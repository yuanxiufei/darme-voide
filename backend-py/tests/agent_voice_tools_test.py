"""S5 工具层自检：音色分配工具集（``agents/tools/voice_tools.py``，3 个工具）。

重点是两条**硬规则**（它们是 Agent 提示词里的铁律，违反必须被工具层挡住，而不是靠模型自觉）：

1. **一角色一音色** —— 同一剧里已有别的角色用了该音色 -> 返回 ``error``（不落库）；
2. **speaker_id 全局唯一** —— 已分配给别的角色 -> 返回 ``error``。

另有三处容易写歪的细节：``infer_gender`` 是**字符类**正则；``role_tag`` 过滤**只在回落分支**
生效；分配时 ``voice_sample_url`` 要**显式置空**（换音色后旧样本失效）。

运行::

    ./.venv/Scripts/python.exe tests/agent_voice_tools_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="voicetools_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import ai_service_configs, ai_voices, characters  # noqa: E402
from app.response import now  # noqa: E402
from app.services.agents.tools.voice_tools import create_voice_tools, infer_gender  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _call(tools: dict, tool_id: str, arguments: dict | None = None):
    return asyncio.run(tools[tool_id].run(arguments))


def main() -> int:  # noqa: C901
    # ================= 纯函数 =================
    check(
        "性别: 命中男声线索（含字符类巧合 —— 'male' 靠 m/a/l/e 命中）",
        infer_gender("male_qn", []) == "男声" and infer_gender("张三", ["低沉 大叔感"]) == "男声",
        infer_gender("male_qn", []),
    )
    # 只含中文线索的走女声；含英文单词的会**先命中男声的字符类**（见下条）
    check("性别: 命中女声线索（中文）", infer_gender("少女音", []) == "女声"
          and infer_gender("御姐", []) == "女声")
    check(
        "性别: 字符类是**按单字符**匹配 —— `voice` 里的 o/e 就够判男声了（不是词匹配）",
        infer_gender("voice", []) == "男声" and infer_gender("male_qn", []) == "男声",
        infer_gender("voice", []),
    )
    check("性别: 真无线索才中性（不含任何命中字符，如 qvz）",
          infer_gender("qvz", []) == "中性" and infer_gender("v2", "不是数组") == "中性",
          (infer_gender("qvz", []), infer_gender("v2", "不是数组")))

    # ================= 造数据 =================
    client = TestClient(app)
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 音色工具"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]

    lin = _insert_character(drama_id, "林昭", role="主角", personality="隐忍")
    wan = _insert_character(drama_id, "阿晚", role="反派")

    tools = create_voice_tools(episode_id, drama_id)
    check("工具: 产出 3 个工具，键即 id",
          set(tools) == {"get_characters", "list_voices", "assign_voice"}
          and all(k == t.id for k, t in tools.items()), sorted(tools))
    check("工具: assign_voice 里 character_id/voice_id 必填，speaker_id/reason 可选",
          tools["assign_voice"].input_schema["required"] == ["character_id", "voice_id"],
          tools["assign_voice"].input_schema["required"])

    # ================= get_characters =================
    listed = _call(tools, "get_characters")
    names = [c["name"] for c in listed["characters"]]
    check("角色: 列出全剧组角色（未软删）", set(names) == {"林昭", "阿晚"}, names)
    check(
        "角色: 未分配音色时 current_voice = 「未分配」、speaker_id 为空串",
        listed["characters"][0]["current_voice"] == "未分配"
        and listed["characters"][0]["speaker_id"] == "",
        listed["characters"][0],
    )

    # ================= list_voices（回落分支）=================
    fallback = _call(tools, "list_voices")
    check(
        "音色: 库里没有音色 -> 回落 6 个通用音色 + instruction",
        len(fallback["voices"]) == 6 and fallback["provider"] == "minimax"
        and fallback["instruction"].startswith("根据角色的性别"),
        (fallback["provider"], len(fallback["voices"])),
    )
    filtered = _call(tools, "list_voices", {"role_tag": "反派"})
    # 回落表里**只有 onyx** 带「反派」标签（原 TS 的 role_tags 就是这么分的）
    check("音色: 回落分支按 role_tag 过滤（反派只有 onyx 1 个）",
          len(filtered["voices"]) == 1 and all("反派" in v["role_tags"] for v in filtered["voices"]),
          [v["id"] for v in filtered["voices"]])

    # 有库数据时：provider 跟着**当前集音频配置**走，且 role_tag **不过滤**
    _set_audio_config(drama_id, episode_id, provider="cosyvoice")
    _insert_voice("cosyvoice", "cosy_a", "小美", '["温柔", "甜美", "女主"]', '["主角"]')
    _insert_voice("cosyvoice", "cosy_b", "大叔", '["低沉"]', '["配角"]')
    _insert_voice("minimax", "mm_x", "别的厂商音色", '["x"]', '["主角"]')
    with_data = _call(tools, "list_voices", {"role_tag": "主角"})
    check(
        "音色: provider 取自**当前集音频配置**（cosyvoice），且不含别家音色",
        with_data["provider"] == "cosyvoice"
        and [v["id"] for v in with_data["voices"]] == ["cosy_a", "cosy_b"],
        (with_data["provider"], [v["id"] for v in with_data["voices"]]),
    )
    check(
        "音色: 库分支**不按 role_tag 过滤**（原 TS 只在回落分支过滤）",
        len(with_data["voices"]) == 2, len(with_data["voices"]),
    )
    check(
        "音色: 描述前 2 条作 traits、第 3 条起作 suitable_for",
        with_data["voices"][0]["traits"] == "温柔、甜美"
        and with_data["voices"][0]["suitable_for"] == "女主",
        with_data["voices"][0],
    )

    # ================= assign_voice 正常路径 =================
    assigned = _call(tools, "assign_voice", {
        "character_id": lin, "voice_id": "cosy_a", "speaker_id": "S1", "reason": "主角"})
    check(
        "分配: 落库音色/厂商/speaker_id，并返回英文回执",
        assigned["message"] == f'Assigned voice "cosy_a" (speaker "S1") to character {lin}'
        and assigned["reason"] == "主角",
        assigned,
    )
    with engine.begin() as conn:
        row = conn.execute(select(characters).where(characters.c.id == lin)).first()
    check("分配: voice_provider 跟随当前集音频配置（cosyvoice）",
          row.voice_style == "cosy_a" and row.voice_provider == "cosyvoice"
          and row.speaker_id == "S1",
          (row.voice_style, row.voice_provider, row.speaker_id))
    check("分配: voice_sample_url 被**显式置空**（换音色后旧样本失效）",
          row.voice_sample_url is None, row.voice_sample_url)

    # ================= 硬规则 1：一角色一音色 =================
    conflict = _call(tools, "assign_voice", {"character_id": wan, "voice_id": "cosy_a"})
    check(
        "硬规则1: 同剧另一角色已用该音色 -> 返回 error（**不落库**）",
        conflict.get("error") == f'音色 "cosy_a" 已分配给角色「林昭」(id={lin})，禁止共用，请改选其他音色',
        conflict,
    )
    with engine.begin() as conn:
        wan_row = conn.execute(select(characters).where(characters.c.id == wan)).first()
    check("硬规则1: 冲突时目标角色的音色**未变**", wan_row.voice_style is None, wan_row.voice_style)
    check(
        "硬规则1: 重新分配给**同一个角色**（改回自己）不算冲突",
        "message" in _call(tools, "assign_voice", {"character_id": lin, "voice_id": "cosy_a"}),
    )

    # ================= 硬规则 2：speaker_id 全局唯一 =================
    speaker_conflict = _call(tools, "assign_voice", {
        "character_id": wan, "voice_id": "cosy_b", "speaker_id": "S1"})
    check(
        "硬规则2: speaker_id 已属于别的角色 -> 返回 error",
        speaker_conflict.get("error") == f'speaker_id "S1" 已分配给角色「林昭」(id={lin})，请勿复用',
        speaker_conflict,
    )
    ok_assign = _call(tools, "assign_voice", {
        "character_id": wan, "voice_id": "cosy_b", "speaker_id": "S2"})
    with engine.begin() as conn:
        wan_row2 = conn.execute(select(characters).where(characters.c.id == wan)).first()
    check("硬规则2: 换独立 speaker_id 后成功", wan_row2.speaker_id == "S2"
          and wan_row2.voice_style == "cosy_b", wan_row2.speaker_id)
    check("分配: 不传 speaker_id 时回执里是 'n/a'（且不改已有 speaker_id）",
          'speaker "n/a"' in _call(tools, "assign_voice",
                                   {"character_id": wan, "voice_id": "cosy_b"})["message"],
          ok_assign["message"])

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


def _insert_character(drama_id: int, name: str, **extra) -> int:
    values: dict[str, object] = {"drama_id": drama_id, "name": name}
    for column in ("created_at", "updated_at"):
        if column in characters.c:
            values[column] = now()
    values.update(extra)
    with engine.begin() as conn:
        return int(conn.execute(characters.insert().values(**values)).lastrowid)


def _insert_voice(provider: str, voice_id: str, name: str, description: str,
                  role_tags: str) -> None:
    with engine.begin() as conn:
        conn.execute(ai_voices.insert().values(
            voice_id=voice_id, voice_name=name, description=description,
            role_tags=role_tags, language="中文", provider=provider, created_at=now(),
        ))


def _set_audio_config(drama_id: int, episode_id: int, provider: str) -> None:
    """给该集挂一个 audio 配置（工具的 provider 取自它）。"""
    from app.models import episodes

    with engine.begin() as conn:
        values = {
            "service_type": "audio", "provider": provider,
            "base_url": "https://api.example.com", "api_key": "k",
            "model": "speech-2.8-hd", "is_active": 1,
        }
        for column in ("name", "created_at", "updated_at"):
            if column in ai_service_configs.c:
                values[column] = f"测试音频-{provider}" if column == "name" else now()
        config_id = int(conn.execute(ai_service_configs.insert().values(**values)).lastrowid)
        conn.execute(episodes.update().where(episodes.c.id == episode_id)
                     .values(audio_config_id=config_id))


if __name__ == "__main__":
    raise SystemExit(main())
