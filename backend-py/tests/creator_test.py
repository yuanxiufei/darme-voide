"""S6 自检：Agent 创建器 + `POST /agent-configs/generate`（``agents/creator.ts`` 169 行）。

一句话需求 → Agent 配置这条链路本身很短，但有三类**静默出错**：

1. **JSON 提取**：LLM 常套 markdown 围栏、或在 JSON 前后夹解释 ⇒ 必须「剥围栏 + 取首个 `{`
   到末个 `}`」；取不到要**报错**（而不是返回空对象让后面静默产出空配置）；
2. **skills 过滤**：只留 `AVAILABLE_SKILL_IDS`（**自有 skill**）里的 id，非法**静默丢弃**；
   过滤后为空要回退该 Agent 的默认绑定 —— 否则会把「外部技能库」的 skill 绑给常驻 Agent，
   注入半截指令；
3. **落库 upsert**：按 ``agent_type`` 取第一条，更新时**只改** name/description/systemPrompt/
   skills/isActive，**保留** model/temperature 等（并且把软删的 `deleted_at` 清回 null 复活）。

外加**两套文案别混**：路由层是 ``agent_type required`` / ``requirement required``，
creator 内部是 ``未知 Agent 类型：…`` / ``requirement 不能为空``。

运行::

    ./.venv/Scripts/python.exe tests/creator_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="creator_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import agent_configs  # noqa: E402
from app.response import now, row_to_dict  # noqa: E402
from app.routers import agent_configs as ac_route  # noqa: E402
from app.services.agent_registry import VALID_AGENT_TYPES, get_default_name  # noqa: E402
from app.services.agents import creator as cr  # noqa: E402
from app.services.agents import skills as sk  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


GOOD_PROMPT = "你是一名分镜师。" + "请严格按镜头类型路由表拆解镜头，并逐镜填写首尾帧提示词。" * 3

_CALLS: list[dict] = []


def _stub_generate(text: str) -> None:
    async def _fake(**kwargs) -> dict:
        _CALLS.append(kwargs)
        return {"text": text, "tool_calls": [], "tool_results": [], "steps": []}

    cr.default_generate = _fake  # type: ignore[assignment]


def main() -> int:  # noqa: C901
    # ================= extract_json =================
    check("提取: 纯 JSON 直取",
          cr.extract_json('{"a": 1}') == {"a": 1})
    check("提取: 剥 ``` 围栏（含语言标注）",
          cr.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
          and cr.extract_json('```\n{"a": 1}\n```') == {"a": 1})
    check("提取: 前后夹解释 -> 取**首个 { 到末个 }**",
          cr.extract_json('好的，结果如下：\n{"a": {"b": 2}}\n以上。') == {"a": {"b": 2}})
    check("提取: 没有花括号 -> 抛 `生成器未返回有效 JSON`",
          _raises(lambda: cr.extract_json("没有 JSON"), "生成器未返回有效 JSON"))
    check("提取: 只有 `}` 或 `}{` 这种 -> 同样抛",
          _raises(lambda: cr.extract_json("}"), "生成器未返回有效 JSON")
          and _raises(lambda: cr.extract_json("}{"), "生成器未返回有效 JSON"))

    # ================= 前置契约 =================
    with engine.begin() as conn:
        unknown = _raises(
            lambda: asyncio.run(cr.generate_agent_config(conn, "没有这种", "需求")),
            None)
    check("校验: 未知 Agent 类型 -> 报错里带「可用：」全列表（6 个）",
          unknown != "" and "未知 Agent 类型：没有这种" in unknown
          and all(t in unknown for t in VALID_AGENT_TYPES),
          unknown[:80])
    with engine.begin() as conn:
        blank = _raises(
            lambda: asyncio.run(cr.generate_agent_config(conn, "extractor", "   ")),
            "requirement 不能为空")
    check("校验: requirement 只有空白 -> `requirement 不能为空`", blank)

    # ================= 生成（打桩 LLM）=================
    client = TestClient(app)
    client.post("/api/v1/ai-configs", json={
        "service_type": "text", "provider": "openai", "base_url": "https://api.example.com",
        "api_key": "k", "model": ["m"], "is_active": True,
    })

    _CALLS.clear()
    _stub_generate(json.dumps({
        "name": " 我的分镜师 ", "description": "  一句话职责  ",
        "system_prompt": GOOD_PROMPT,
        "skills": [sk.list_core_skill_ids()[0], "不存在的 skill", 123],
    }, ensure_ascii=False))
    with engine.begin() as conn:
        candidate = asyncio.run(cr.generate_agent_config(conn, "storyboard_breaker", "更强调运镜"))
    check("生成: name/description 都 trim，systemPrompt 取用",
          candidate.name == "我的分镜师" and candidate.description == "一句话职责"
          and candidate.system_prompt == GOOD_PROMPT)
    check("生成: skills 只留自有 skill（非法/非字符串一律静默丢弃）",
          [s["id"] for s in candidate.skills] == [sk.list_core_skill_ids()[0]]
          and all(s["enabled"] is True for s in candidate.skills)
          and candidate.skills[0]["priority"] == 1,
          candidate.skills)
    check("生成: 走**单发**调用（无工具、max_steps=1、模型取 textConfig.model）",
          _CALLS[0]["tools"].tools == {} and _CALLS[0]["max_steps"] == 1
          and _CALLS[0]["model"] == "m" and _CALLS[0]["model_settings"] == {},
          {k: v for k, v in _CALLS[0].items() if k in ("max_steps", "model", "model_settings")})
    message = _CALLS[0]["message"]
    check("生成: user message 结构完整（Agent 类型 + 分隔标记 + JSON 骨架）",
          message.startswith("Agent 类型：storyboard_breaker\n当前默认提示词：\n--- 开始 ---")
          and message.rstrip().endswith('"skills": ["从可用集合选出的 skill id 列表"]}'),
          message[:40])
    check("生成: user message 里带了默认提示词原文与可用 skill 集合",
          "你是资深影视分镜师" in message
          and f"可用 skill 集合：{', '.join(cr.AVAILABLE_SKILL_IDS)}" in message
          and '"system_prompt"' in message and f"用户需求：更强调运镜" in message)

    _stub_generate(json.dumps({"system_prompt": GOOD_PROMPT}, ensure_ascii=False))
    with engine.begin() as conn:
        fallback = asyncio.run(cr.generate_agent_config(conn, "voice_assigner", "偏少年音"))
    check("生成: skills 缺失/为空 -> **回退该 Agent 的默认绑定**；name 缺失 -> 回落默认显示名",
          [s["id"] for s in fallback.skills] == sk.resolve_default_skills("voice_assigner")
          and fallback.name == get_default_name("voice_assigner"),
          (fallback.skills, fallback.name))

    _stub_generate(json.dumps({"system_prompt": "太短"}, ensure_ascii=False))
    with engine.begin() as conn:
        short = _raises(
            lambda: asyncio.run(cr.generate_agent_config(conn, "extractor", "需求")), None)
    check("生成: systemPrompt < 50 字 -> 报错并带 length",
          "生成器产出无效 systemPrompt（length=2）" in short, short)

    # ================= persist（upsert 两条路径）=================
    # ⚠️ persist 返回的是 **Row**（与 TS 返回 drizzle 行对应）⇒ 用 `row_to_dict` 归一成
    #    snake_case 字典（这正是路由层对客户端做的事）。
    with engine.begin() as conn:
        saved = row_to_dict(cr.persist_agent_config(candidate))
    check("落库: 首次是 insert，带默认 model='' / temperature=0.7 / maxTokens=4096 / maxIterations=10",
          saved["agent_type"] == "storyboard_breaker" and saved["model"] == ""
          and saved["temperature"] == 0.7 and saved["max_tokens"] == 4096
          and saved["max_iterations"] == 10 and saved["is_active"] is True
          and saved["deleted_at"] is None,
          {k: saved[k] for k in ("model", "temperature", "max_tokens", "max_iterations")})
    check("落库: skills 存的是**紧凑 JSON**（与 `_validate_skills` 同口径）",
          json.dumps(json.loads(saved["skills"]), ensure_ascii=False, separators=(",", ":"))
          == saved["skills"], saved["skills"])

    # 改成「有既有配置 + 软删」的状态，验证更新路径与复活
    with engine.begin() as conn:
        conn.execute(agent_configs.update().where(agent_configs.c.id == saved["id"]).values(
            model="keep-me", temperature=0.1, deleted_at=now()))
    updated_candidate = cr.GeneratedAgentConfig(
        agent_type="storyboard_breaker", name="新名字", description="新描述",
        system_prompt=GOOD_PROMPT + "补充规则。", skills=[{"id": "x", "enabled": True,
                                                          "priority": 1}])
    with engine.begin() as conn:
        again = row_to_dict(cr.persist_agent_config(updated_candidate))
        rows = conn.execute(select(agent_configs).where(
            agent_configs.c.agent_type == "storyboard_breaker")).all()
    check("落库: 已有配置走 update（同 agent_type 只一行，不新增）",
          len(rows) == 1 and again["id"] == saved["id"], len(rows))
    check("落库: 更新保留 model/temperature（只改名字/描述/提示词/skills/isActive）",
          again["model"] == "keep-me" and again["temperature"] == 0.1
          and again["name"] == "新名字" and again["description"] == "新描述"
          and "补充规则。" in again["system_prompt"])
    check("落库: 更新会把软删的 deleted_at **清回 null**（复活）",
          again["deleted_at"] is None and again["is_active"] is True, again["deleted_at"])

    # ================= 端点 =================
    check("端点: 缺 agent_type -> 400 `agent_type required`（路由层文案）",
          client.post("/api/v1/agent-configs/generate", json={}).json()
          == {"code": 400, "message": "agent_type required"})
    bad_type = client.post("/api/v1/agent-configs/generate",
                           json={"agent_type": "没有这种", "requirement": "x"})
    check("端点: 未知类型 -> 400 且文案与 creator 一致（带可用列表）",
          bad_type.status_code == 400
          and "未知 Agent 类型：没有这种（可用：" in bad_type.json()["message"],
          bad_type.json())
    no_req = client.post("/api/v1/agent-configs/generate", json={"agent_type": "extractor"})
    check("端点: 缺 requirement -> 400 `requirement required`（**与 creator 的文案不同**）",
          no_req.json() == {"code": 400, "message": "requirement required"}, no_req.json())

    _stub_generate(json.dumps({"name": "预览名", "description": "d",
                               "system_prompt": GOOD_PROMPT, "skills": []}, ensure_ascii=False))
    dry = client.post("/api/v1/agent-configs/generate",
                      json={"agent_type": "grid_prompt_generator", "requirement": "更严格",
                            "dry_run": True})
    check("端点: dry_run=true -> 200 只回 candidate（**不落库**）",
          dry.status_code == 200 and "saved" not in dry.json()["data"]
          and dry.json()["data"]["candidate"]["agentType"] == "grid_prompt_generator"
          and dry.json()["data"]["candidate"]["systemPrompt"] == GOOD_PROMPT,
          dry.json()["data"])
    with engine.begin() as conn:
        leaked = conn.execute(select(agent_configs).where(
            agent_configs.c.agent_type == "grid_prompt_generator")).all()
    check("端点: dry_run 确认没写库", leaked == [], len(leaked))

    done = client.post("/api/v1/agent-configs/generate",
                       json={"agent_type": "grid_prompt_generator", "requirement": "更严格"})
    check("端点: 非 dry_run -> 回 candidate + saved（saved 是 snake_case 行）",
          done.status_code == 200 and set(done.json()["data"]) == {"candidate", "saved"}
          and done.json()["data"]["saved"]["agent_type"] == "grid_prompt_generator"
          and "system_prompt" in done.json()["data"]["saved"],
          sorted(done.json()["data"]["saved"])[:6])
    check("端点: candidate 是 **camelCase**（systemPrompt/agentType），与 TS 接口一致",
          set(done.json()["data"]["candidate"])
          == {"agentType", "name", "description", "systemPrompt", "skills"})

    _stub_generate("没有 JSON")
    failed = client.post("/api/v1/agent-configs/generate",
                         json={"agent_type": "extractor", "requirement": "x"})
    check("端点: 运行期异常 -> 400 且 message 取 str(err)",
          failed.status_code == 400
          and failed.json()["message"] == "生成器未返回有效 JSON", failed.json())

    check("常量: AVAILABLE_SKILL_IDS 就是自有 skill 列表（模块加载快照）",
          cr.AVAILABLE_SKILL_IDS == sk.list_core_skill_ids(), cr.AVAILABLE_SKILL_IDS)

    # ================= 汇总 =================
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


def _raises(fn, expected_message: str | None) -> str:
    """调用 fn 并把异常文案返回（``expected_message`` 给了就断言包含它）。"""
    try:
        fn()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        if expected_message is not None and expected_message not in text:
            return f"<文案不符：{text}>"
        return text
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
