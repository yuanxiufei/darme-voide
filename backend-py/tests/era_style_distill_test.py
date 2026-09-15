"""S6 自检：时代背景提炼（``era-background.ts``）与风格提炼（``style-profiles.ts``）
—— 补上最后两个「中等件」，各一条端点：``POST /dramas/{id}/era-background/extract``、
``POST /style-profiles/{id}/distill``。

两侧都**不打网络**：``text_generation.generate_text`` 整体替换成桩（服务内是惰性导入，
所以替换模块属性即可生效）。锁死的是：

- 时代背景：剧本**自动聚合格式**、**8000 字首尾截断**、```json ``` 围栏剥离、
  「异常一律包一层 ``时代背景提炼失败: ``」而 ``Drama not found``/``暂无剧本`` 不包；
- 风格提炼：**prompt 组装**（测量事实 indent=2/缺省文案）、**四对象三数组的缺省归一**、
  「非 JSON 输出」判定、**所有异常吞成 ok:false**（路由转 400）。

运行::

    ./.venv/Scripts/python.exe tests/era_style_distill_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="esd_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ⚠️ Windows 控制台默认 GBK：检查名里带 `⇒`/emoji 时**打印阶段**会 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import dramas, episodes, style_profiles  # noqa: E402
from app.core.response import now  # noqa: E402
from app.services import era_background as eb  # noqa: E402
from app.services import style_profiles as sp  # noqa: E402
from app.services import text_generation as tg  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_CALLS: list[dict[str, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def stub_returning(payload: object):
    """替换 ``generate_text``：记录入参，返回固定文本（或抛错）。"""
    async def _stub(_conn, prompt, options=None):
        _CALLS.append({"prompt": prompt, "options": options})
        if isinstance(payload, Exception):
            raise payload
        return payload

    return _stub


def main() -> int:  # noqa: C901
    stamp = now()
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="时代剧", created_at=stamp, updated_at=stamp)).lastrowid)
        empty_id = int(conn.execute(dramas.insert().values(
            title="空剧", created_at=stamp, updated_at=stamp)).lastrowid)
        deleted_id = int(conn.execute(dramas.insert().values(
            title="已删剧", created_at=stamp, updated_at=stamp,
            deleted_at=stamp)).lastrowid)
        conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="楔子",
            script_content="第一集正文", created_at=stamp, updated_at=stamp))
        conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=2, title="",  # 空标题：表头应退化成「第2集」
            content="第二集正文", created_at=stamp, updated_at=stamp))
        # 已删的集**不参与**聚合
        conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=3, title="已删集",
            script_content="不该出现", created_at=stamp, updated_at=stamp,
            deleted_at=stamp))
        profile_id = int(conn.execute(style_profiles.insert().values(
            drama_id=drama_id, name="港片风", description="霓虹与蒸汽",
            source=None, preferences="节奏快", created_at=stamp, updated_at=stamp)).lastrowid)

    # ================= 助手 =================
    check("助手: strip_json_fence 剥 ```json 围栏 / 截到最外层大括号 / 原文不损",
          eb.strip_json_fence('```json\n{"a":1}\n```') == '{"a":1}'
          and eb.strip_json_fence('前言{"a":1}后记') == '{"a":1}'
          and eb.strip_json_fence('{"a":1}') == '{"a":1}'
          and eb.strip_json_fence('没有大括号') == "没有大括号")
    check("助手: 整数语义的浮点落成 int（避免 JSON 里 340000.0）",
          sp._int_if_integral(340000.0) == 340000
          and isinstance(sp._int_if_integral(340000.0), int)
          and sp._int_if_integral(1.5) == 1.5)

    # ================= 时代背景提炼 =================
    tg.generate_text = stub_returning(
        '```json\n{"era":"古代仙侠","summary":"长安城","image_style_en":"ancient Chinese"}\n```')

    async def _extract(did: int, source: str | None = None) -> object:
        with engine.begin() as conn:
            return await eb.extract_drama_era_background(conn, did, source)

    _CALLS.clear()
    parsed = asyncio.run(_extract(drama_id))
    check("提炼: 返回解析后的时代背景，历史键 image_style_en 归一成 imageHint",
          parsed == {"era": "古代仙侠", "summary": "长安城", "imageHint": "ancient Chinese"},
          parsed)
    prompt = str(_CALLS[-1]["prompt"])
    check("提炼: 剧本**自动聚合**（集号+标题换行正文、集间空行、软删集不参与）",
          prompt.startswith("剧名：时代剧\n以下是剧本内容：\n")
          and "第1集 楔子\n第一集正文\n\n第2集\n第二集正文" in prompt
          and "不该出现" not in prompt and prompt.endswith("请提炼这部剧的时代背景设定，只输出 JSON。"),
          prompt[:80])
    check("提炼: 系统提示词与温度/上限按 TS 常量传（system + temperature 0.2 + maxTokens 800）",
          _CALLS[-1]["options"] == {"system": eb._ERA_EXTRACT_SYSTEM_PROMPT,
                                    "temperature": 0.2, "maxTokens": 800},
          _CALLS[-1]["options"])

    with engine.begin() as conn:
        stored = conn.execute(dramas.select().where(dramas.c.id == drama_id)).first()
    check("提炼: 落库为**紧凑** JSON（键序 era/summary/imageHint），updated_at 刷新",
          stored.era_background == '{"era":"古代仙侠","summary":"长安城","imageHint":"ancient Chinese"}'
          and stored.updated_at >= stamp, stored.era_background)

    # 显式 source_text：**不聚合**剧本
    _CALLS.clear()
    asyncio.run(_extract(drama_id, "  手打剧本原文  "))
    check("提炼: 传了 source_text 就用它（trim 后），不聚合剧本",
          "手打剧本原文" in str(_CALLS[-1]["prompt"])
          and "第一集正文" not in str(_CALLS[-1]["prompt"]))

    # 截断：首 70% + 尾 30%
    _CALLS.clear()
    long_text = "A" * 8000 + "M" * 100 + "Z" * 4000  # 12100 字，中段标记 M 应被丢掉
    asyncio.run(_extract(drama_id, long_text))
    sent = str(_CALLS[-1]["prompt"])
    check("提炼: 超过 8000 字按「首 5600 + 省略行 + 尾 2400」截断（保设定与结局）",
          "A" * 5600 in sent and "Z" * 2400 in sent and "……（中间省略）……" in sent
          and sent.count("A") == 5600 and sent.count("Z") == 2400 and "M" not in sent,
          (sent.count("A"), sent.count("Z"), sent.count("M")))
    asyncio.run(_extract(drama_id, "短" * 7999))
    check("提炼: 7999 字不动裁剪（阈值是 >8000 才裁）",
          "……（中间省略）……" not in str(_CALLS[-1]["prompt"]))

    # 异常链路
    tg.generate_text = stub_returning("模型说了些废话，没有 JSON")
    try:
        asyncio.run(_extract(drama_id))
        check("提炼: 非 JSON 输出 -> 抛错", False)
    except ValueError as err:
        check("提炼: 非 JSON 输出 -> 包成 `时代背景提炼失败: ...`（含原始报错）",
              str(err).startswith("时代背景提炼失败: "), str(err))

    # ⚠️ 原 TS 的**宽松判据**：只给 era 也会被接受 —— summary/imageHint 会**回退到 era**
    #    （模块 docstring 里标的那个坑：判据是 `summary || imageHint`，不是三字段齐全）。
    tg.generate_text = stub_returning('{"era":"只有时代标签"}')
    filled = asyncio.run(_extract(drama_id))
    check("提炼: 只给 era 也**接受**（era/summary 回退成同一值；imageHint 链上**没有 era** ⇒ 留空）",
          filled == {"era": "只有时代标签", "summary": "只有时代标签", "imageHint": ""},
          filled)

    tg.generate_text = stub_returning('{"foo":1}')
    try:
        asyncio.run(_extract(drama_id))
        check("提炼: 三字段皆空 -> 抛错", False)
    except ValueError as err:
        check("提炼: 合法 JSON 但三字段皆空 -> `AI 输出缺少有效时代背景字段`（被包一层）",
              "时代背景提炼失败: AI 输出缺少有效时代背景字段" == str(err), str(err))

    try:
        asyncio.run(_extract(999999))
        check("提炼: 剧不存在 -> 抛错", False)
    except ValueError as err:
        check("提炼: 剧不存在 -> `Drama not found`（**不被**「提炼失败」包裹）",
              str(err) == "Drama not found", str(err))
    try:
        asyncio.run(_extract(empty_id))
        check("提炼: 无剧本 -> 抛错", False)
    except ValueError as err:
        check("提炼: 无剧本 -> 引导文案（同样不被包裹）",
              str(err).startswith("该剧暂无剧本内容，无法提炼时代背景"), str(err))

    # ================= 风格提炼 =================
    tg.generate_text = stub_returning(
        '好的，结果如下：\n```json\n{"storytelling":{"pace":"快"},"facts":["1080p"],'
        '"preferences":["冷色调"]}\n```')

    async def _distill(pid: object) -> dict:
        with engine.begin() as conn:
            return await sp.distill_style_profile(conn, pid)

    _CALLS.clear()
    out = asyncio.run(_distill(profile_id))
    check("风格提炼: 四对象三数组缺省归一（只给 storytelling/facts/preferences，其余补空）",
          out == {"ok": True, "result": {
              "storytelling": {"pace": "快"}, "shot_patterns": {}, "audio_captions": {},
              "qc_rules": {}, "facts": ["1080p"], "inferences": [], "preferences": ["冷色调"]}},
          out)
    distill_prompt = str(_CALLS[-1]["prompt"])
    check("风格提炼: prompt 组装（素材说明/名称/描述/用户偏好，缺省给（无））",
          distill_prompt.startswith("参考素材说明：（未提供，仅凭 Profile 名称/描述）\n\nProfile 名称：港片风")
          and "\n\n描述：霓虹与蒸汽" in distill_prompt
          and distill_prompt.endswith("已知用户偏好（若有）：节奏快"), distill_prompt)
    check("风格提炼: 无测量事实时**不出现**「测量事实」行",
          "测量事实" not in distill_prompt)
    check("风格提炼: 系统提示词按 TS 逐字传（含四类规则/三分类要求）",
          _CALLS[-1]["options"] == {"system": sp._DISTILL_INSTRUCTIONS}
          and "facts 只放可验证的客观测量" in sp._DISTILL_INSTRUCTIONS)

    # 有测量事实：indent=2
    real_probe = sp._probe_measurement_facts

    async def _fake_probe(source):
        return {"resolution": "1920x1080", "duration_seconds": 12, "file_size_bytes": 340000}

    sp._probe_measurement_facts = _fake_probe  # type: ignore[assignment]
    _CALLS.clear()
    asyncio.run(_distill(profile_id))
    with_facts = str(_CALLS[-1]["prompt"])
    check("风格提炼: 测量事实以 **indent=2 JSON** 注入（对齐 JSON.stringify(x,null,2)）",
          '参考素材测量事实（ffprobe 探测，可信）：{\n  "resolution": "1920x1080",' in with_facts
          and '"file_size_bytes": 340000' in with_facts, with_facts[80:200])
    sp._probe_measurement_facts = real_probe  # type: ignore[assignment]

    check("风格提炼: 探测助手对 None/非媒体文件都返回空事实（不抛）",
          asyncio.run(real_probe(None)) == {}
          and asyncio.run(real_probe("nope.mp4")) == {})

    tg.generate_text = stub_returning("我只想聊聊天")
    bad = asyncio.run(_distill(profile_id))
    check("风格提炼: LLM 输出没有大括号 -> ok:false + 'LLM output is not JSON'",
          bad == {"ok": False, "error": "LLM output is not JSON"}, bad)
    tg.generate_text = stub_returning(RuntimeError("provider 未配置"))
    failed = asyncio.run(_distill(profile_id))
    check("风格提炼: **异常吞成 ok:false**（不向上抛）",
          failed == {"ok": False, "error": "provider 未配置"}, failed)
    check("风格提炼: Profile 不存在 -> ok:false + 'Profile not found'",
          asyncio.run(_distill(999999)) == {"ok": False, "error": "Profile not found"})

    # ================= 端点 =================
    tg.generate_text = stub_returning(
        '{"era":"现代都市","summary":"写字楼","image_style_en":"modern city"}')
    client = TestClient(app)
    check("端点: 非法 id / 剧不存在 / **已软删的剧** -> 404（文案「剧本不存在」）",
          client.post("/api/v1/dramas/abc/era-background/extract").status_code == 404
          and client.post("/api/v1/dramas/999999/era-background/extract").status_code == 404
          and client.post(f"/api/v1/dramas/{deleted_id}/era-background/extract").json()
          == {"code": 404, "message": "剧本不存在"})
    ok_resp = client.post(f"/api/v1/dramas/{drama_id}/era-background/extract",
                          json={"source_text": "写字楼里的故事"}).json()
    check("端点: 提炼成功 -> 200 + data 就是时代背景对象（裸字段，非信封）",
          ok_resp["code"] == 200
          and ok_resp["data"] == {"era": "现代都市", "summary": "写字楼",
                                  "imageHint": "modern city"}, ok_resp)
    fallback = client.post(f"/api/v1/dramas/{drama_id}/era-background/extract",
                           json={"source_text": None, "sourceText": "回退键"}).json()
    check("端点: `source_text ?? sourceText` —— 显式 null 才回退另一个键",
          fallback["code"] == 200)
    hidden = client.post(f"/api/v1/dramas/{drama_id}/era-background/extract",
                         json={"sourceText": "只有 camelCase 键"}).json()
    check("端点: 只给 sourceText（camelCase）也能提炼", hidden["code"] == 200)
    empty_body = client.post(f"/api/v1/dramas/{empty_id}/era-background/extract")
    check("端点: 无剧本剧 -> 400 + 引导文案（空请求体也安全）",
          empty_body.status_code == 400 and empty_body.json()["code"] == 400
          and empty_body.json()["message"].startswith("该剧暂无剧本内容"), empty_body.json())

    client = TestClient(app)
    # ⚠️ 换回 distill 的桩（上面为 era 端点装的是 era 的桩）
    tg.generate_text = stub_returning(
        '{"storytelling":{"pace":"快"},"facts":["1080p"],"preferences":["冷色调"]}')
    distilled = client.post(f"/api/v1/style-profiles/{profile_id}/distill")
    check("端点: distill -> 200 + {result: {...}}（**不落库**，等 /apply 确认）",
          distilled.json()["code"] == 200
          and distilled.json()["data"]["result"]["storytelling"] == {"pace": "快"},
          distilled.json())
    check("端点: distill 不存在的 Profile -> 400（不是 404，原 TS 如此）",
          client.post("/api/v1/style-profiles/999999/distill").json()
          == {"code": 400, "message": "Profile not found"})
    with engine.begin() as conn:
        before = conn.execute(
            style_profiles.select().where(style_profiles.c.id == profile_id)).first()
    check("端点: distill 确实**没写库**（storytelling 仍是空）",
          before.storytelling in (None, "") or not json.loads(before.storytelling or "{}"),
          before.storytelling)

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
