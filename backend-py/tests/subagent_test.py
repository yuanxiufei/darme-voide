"""S7 自检：子 Agent 调度（``agents/subagent.py``，移植 ``agents/subagent.ts``）。

锁住四条**硬约束**与两条传播语义：

1. 未知 ``agent_type`` → 错误文本里带可用清单；
2. **禁止自调用**；3. **环形调用**（链里已有目标）；4. **深度上限 2**；
5. 调用链用 ``ContextVar`` 传播 ⇒ **委托中再委托**会被挡（证明跨 await 传播有效）；
6. 链在成功/失败后都**复位**（否则同一次 run 的后续委托会带着旧链误判）。

``run_agent_with_retry`` 全程打桩（不打网络、不起模型）。

运行::

    ./.venv/Scripts/python.exe tests/subagent_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="subagent_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.agent_registry import VALID_AGENT_TYPES  # noqa: E402
from app.agent import runtime, subagent  # noqa: E402

_R: list[tuple[str, bool, object]] = []
_SEEN: dict[str, object] = {}


def check(name: str, condition: object, detail: object = "") -> None:
    _R.append((name, bool(condition), detail))


_DEFAULT = object()


def stub(*, protocol=_DEFAULT, text="正文", tool_calls=2, raises=None):
    async def fake(conn, agent_type, episode_id, drama_id, message, options=None, **_kw):
        _SEEN.update({"conn": conn, "agentType": agent_type, "episodeId": episode_id,
                      "dramaId": drama_id, "message": message})
        if raises:
            raise raises
        return SimpleNamespace(
            text=text, tool_calls=[{}] * tool_calls, tool_results=[],
            protocol=({"status": "completed", "summary": "子任务完成"}
                      if protocol is _DEFAULT else protocol))

    return fake


def tools(parent: str = "orchestrator") -> dict:
    return subagent.create_run_subagent_tools("CONN", 7, 11, parent_type=parent)


def call(tool_map: dict, tool_id: str, arguments: dict):
    return asyncio.run(tool_map[tool_id].run(arguments))


def main() -> int:  # noqa: C901
    tool_map = tools()

    # ── 1. 注册表与清单 ──
    check("注册表: 5 个领域专家，type 均在 VALID_AGENT_TYPES 内且都有 name/capability",
          len(subagent.SUBAGENT_REGISTRY) == 5
          and all(entry["type"] in VALID_AGENT_TYPES
                  and entry["name"] and entry["capability"]
                  for entry in subagent.SUBAGENT_REGISTRY),
          [entry["type"] for entry in subagent.SUBAGENT_REGISTRY])
    listed = call(tool_map, "list_available_agents", {})
    check("清单: **排除自己**（orchestrator 不在返回里），其余 5 条带 name+capability",
          "agents" in listed and len(listed["agents"]) == 5
          and all(entry["type"] != "orchestrator" for entry in listed["agents"])
          and all("capability" in entry for entry in listed["agents"]), listed)
    other = tools(parent="extractor")
    others = call(other, "list_available_agents", {})["agents"]
    # 注册表 5 条里含 extractor ⇒ 排除自己后剩 4 条（含 orchestrator）
    check("清单: 换 parent ⇒ 只把该条排除（extractor 不在、余 4 条）",
          len(others) == 4 and all(entry["type"] != "extractor" for entry in others),
          [entry["type"] for entry in others])

    # ── 2. 四条硬约束 ──
    runtime.run_agent_with_retry = stub()  # type: ignore[assignment]
    bad = call(tool_map, "run_subagent", {"agent_type": "no_such", "task": "x"})
    check("约束: 未知类型 -> error 文本含 `Unknown agent type` 与可用清单",
          "error" in bad and "Unknown agent type: no_such" in bad["error"]
          and "Available:" in bad["error"], bad)
    me = call(tool_map, "run_subagent", {"agent_type": "orchestrator", "task": "x"})
    check("约束: **禁止自调用** -> `Cannot delegate to self (orchestrator)`",
          me.get("error") == "Cannot delegate to self (orchestrator)", me)

    token = subagent._chain.set(("orchestrator", "extractor"))
    try:
        circular = call(tool_map, "run_subagent", {"agent_type": "extractor", "task": "x"})
        depth = call(tool_map, "run_subagent", {"agent_type": "voice_assigner", "task": "x"})
    finally:
        subagent._chain.reset(token)
    check("约束: **环形调用**被挡（链里已有 extractor）",
          circular.get("error", "").startswith("Circular delegation detected")
          and "orchestrator -> extractor -> extractor" in circular["error"], circular)
    check("约束: **深度上限 2**（链长已达 2 ⇒ 任何新委托都拒）",
          depth.get("error") == f"Max subagent depth reached ({subagent.MAX_SUBAGENT_DEPTH})", depth)

    # ── 3. 正常委托 ──
    ok = call(tool_map, "run_subagent", {"agent_type": "extractor", "task": "提取角色场景"})
    check("委托: 回传 {status, summary, text, tool_calls_count}（取自 protocol 与 result）",
          ok == {"status": "completed", "summary": "子任务完成", "text": "正文",
                 "tool_calls_count": 2}, ok)
    check("委托: 上下文透传（conn/episodeId/dramaId/task 原样交给子 Agent）",
          _SEEN["conn"] == "CONN" and _SEEN["episodeId"] == 11 and _SEEN["dramaId"] == 7
          and _SEEN["agentType"] == "extractor" and _SEEN["message"] == "提取角色场景", _SEEN)
    check("委托后: 调用链**复位**（同一次 run 的后续委托不受影响）",
          subagent.current_chain() is None, subagent.current_chain())

    runtime.run_agent_with_retry = stub(protocol=None)  # type: ignore[assignment]
    none_protocol = call(tool_map, "run_subagent", {"agent_type": "extractor", "task": "x"})
    check("委托: protocol 缺失 -> status 回落 `completed`、summary 回落空串",
          none_protocol["status"] == "completed" and none_protocol["summary"] == "",
          none_protocol)

    # ── 4. 失败与链复位 ──
    runtime.run_agent_with_retry = stub(raises=RuntimeError("模型挂了"))  # type: ignore[assignment]
    failed = call(tool_map, "run_subagent", {"agent_type": "extractor", "task": "x"})
    check("失败: 回 `Subagent extractor failed: 模型挂了`（**返回值**而非抛异常）",
          failed.get("error") == "Subagent extractor failed: 模型挂了", failed)
    check("失败后: 调用链同样**复位**", subagent.current_chain() is None)

    # ── 5. 链跨 await 传播（委托中再委托）──
    inner: dict = {}

    async def nested_fake(conn, agent_type, episode_id, drama_id, message, options=None, **_kw):
        # 子 Agent run 期间，再发起一次委托 ⇒ 应被「深度上限」挡住
        inner["result"] = await tool_map["run_subagent"].run(
            {"agent_type": "voice_assigner", "task": "再委托"})
        return SimpleNamespace(text="t", tool_calls=[], tool_results=[], protocol=None)

    runtime.run_agent_with_retry = nested_fake  # type: ignore[assignment]
    call(tool_map, "run_subagent", {"agent_type": "extractor", "task": "外层"})
    check("传播: 子 Agent run **期间**再委托会被深度上限挡住（ContextVar 跨 await 生效）",
          inner.get("result", {}).get("error")
          == f"Max subagent depth reached ({subagent.MAX_SUBAGENT_DEPTH})", inner)

    # ── 6. schema ──
    run_schema = tool_map["run_subagent"].input_schema
    list_schema = tool_map["list_available_agents"].input_schema
    check("schema: run_subagent 必填 [agent_type, task]；list 无必填、且都不许额外键",
          run_schema["required"] == ["agent_type", "task"]
          and run_schema["additionalProperties"] is False
          and list_schema["required"] == [] and list_schema["properties"] == {}, (run_schema,
                                                                                  list_schema))
    check("schema: tool.id 是蛇形（`run_subagent` / `list_available_agents`）",
          tool_map["run_subagent"].id == "run_subagent"
          and tool_map["list_available_agents"].id == "list_available_agents")

    failed_items = [item for item in _R if not item[1]]
    for name, passed, detail in _R:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_R) - len(failed_items)}/{len(_R)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
