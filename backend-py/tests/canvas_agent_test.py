"""画布 Agent 守卫（源 `canvas-agent` 的 MCP + codex 子进程 ⇒ `services/canvas_agent.py` 薄环）。

钉五件事：

1. **工具不产生副作用** —— 写工具只往 `sink` 里攒 op，回给模型的必须是「已登记 N 条」
   这种人话，**不能**假装已经执行（此刻画布没动 ✗）。
2. **⭐ 只读工具必须真给数据** —— 没快照时如实报 `error`（返回 `{}` 会让模型以为
   画布是空的，然后凭空造节点 ✗）；有快照时返回**压缩后**的，正文截断。
3. **⭐ 指令与实现不打架** —— 指令里点名的工具必须都在 `TOOLS` 里；参数 schema
   必须与 `/canvas/tools` 同源（`param_schema`），不许第二套。
4. **跑通一次完整对话** —— 假驱动里先读状态再生成，断言最终 5 条 op、回话原样透出。
5. **端点与错误** —— `/api/v1/canvas/agent` 挂上；空 message / 无模型配置如实 400。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/canvas_agent_test.py
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:
    sys.path.insert(0, str(BP))
    from app.services import canvas_agent as agent  # noqa: PLC0415
    from app.services import canvas_ops as ops  # noqa: PLC0415

    # ---------------------------------------------------------------- ① 工具不产生副作用
    sink: dict = {}
    tools = agent.build_canvas_tools(sink)
    check("工具: 21 个", len(tools.tools) == len(ops.TOOL_NAMES), len(tools.tools))
    check("工具: id 与清单一致", sorted(tools.tools) == sorted(ops.TOOL_NAMES))
    missing_schema = [
        name for name, tool in tools.tools.items()
        if tool.input_schema.get("type") != "object" or not tool.description.strip()
    ]
    check("工具: 每个都有 object schema 与描述", not missing_schema, missing_schema)

    snapshot = {
        "nodes": [{"id": "n1", "type": "text", "text": "y" * 400, "x": 0, "y": 0, "w": 340, "h": 240}],
        "edges": [],
        "viewport": {"x": 0, "y": 0, "scale": 1},
        "selection": ["n1"],
    }
    result = asyncio.run(tools.get("canvas_generate_image").run({"prompt": "一只猫"}))
    check("工具: 写工具回话是「已登记」不是「已执行」",
          result.get("ok") is True and "已登记" in str(result.get("note")), result)
    check("工具: 写工具没有把 op 结果塞回模型", "ops" in sink and isinstance(sink["ops"], list), list(sink))
    check("工具: 攒了 5 条 op", len(sink["ops"]) == 5, len(sink["ops"]))
    check("工具: 调用被记账", len(sink["calls"]) == 1 and sink["calls"][0]["toolName"] == "canvas_generate_image",
          sink["calls"])

    # ---------------------------------------------------------------- ② 只读工具
    bare: dict = {}
    bare_tools = agent.build_canvas_tools(bare)
    empty = asyncio.run(bare_tools.get("canvas_get_state").run({}))
    check("只读: 没快照时如实报错（不返回空画布）",
          isinstance(empty, dict) and "error" in empty and not empty.get("nodes"), empty)

    with_state: dict = {}
    state_tools = agent.build_canvas_tools(with_state, snapshot)
    view = asyncio.run(state_tools.get("canvas_get_state").run({}))
    check("只读: 有快照时返回节点", view.get("nodes") and view["nodes"][0]["id"] == "n1", view)
    check("只读: 正文被截断（不灌满上下文）",
          len(view["nodes"][0]["text"]) <= ops.SNAPSHOT_TEXT_KEEP + 1, view["nodes"][0].get("text"))
    chosen = asyncio.run(state_tools.get("canvas_get_selection").run({}))
    check("只读: 选中查询只回选中", chosen == {"selection": ["n1"]}, chosen)

    # ---------------------------------------------------------------- ③ 指令与实现
    instruction = agent.CANVAS_INSTRUCTIONS
    named = set(re.findall(r"`(canvas_[a-z_]+)`", instruction))
    check("指令: 点名的工具都在清单里", named and not (named - set(ops.TOOL_NAMES)),
          sorted(named - set(ops.TOOL_NAMES)))
    banned = [word for word in ("codex", "mcp", "stdio", "app-server", "jsonl") if word in instruction.lower()]
    check("指令: 不含源项目的实现词汇", not banned, banned)
    check("指令: 要求先读再动", "canvas_get_state" in named, sorted(named))
    check("指令: 要求优先用大工具", "canvas_generate_image" in named, sorted(named))

    schemas = {item["name"]: item["parameters"]["properties"] for item in ops.tool_schemas()}
    check("同源: ops 参数是 array（不是 string）",
          schemas["canvas_apply_ops"]["ops"]["type"] == "array",
          schemas["canvas_apply_ops"]["ops"])
    check("同源: x 是 number", schemas["canvas_create_text_node"]["x"]["type"] == "number")
    check("同源: autoRun 是 boolean", schemas["canvas_create_generation_flow"]["autoRun"]["type"] == "boolean")
    check("同源: referenceNodeIds 是 array",
          schemas["canvas_create_generation_flow"]["referenceNodeIds"]["type"] == "array")
    check("同源: text 仍是 string", schemas["canvas_create_text_node"]["text"]["type"] == "string")
    check("同源: Tool 的 schema 用的就是同一份",
          agent.build_canvas_tools({}).get("canvas_apply_ops").input_schema
          == {"type": "object", "properties": schemas["canvas_apply_ops"]},
          agent.build_canvas_tools({}).get("canvas_apply_ops").input_schema)

    # ---------------------------------------------------------------- ④ 跑通一次对话
    conn = object()  # 假驱动不碰 DB；get_text_config 也被替换
    agent.get_text_config = lambda _conn: {"model": "demo-model"}  # type: ignore[assignment]

    seen: dict = {}

    async def fake_driver(**kwargs):
        seen.update(kwargs)
        registry = kwargs["tools"]
        await registry.get("canvas_get_state").run({})
        await registry.get("canvas_generate_image").run({"prompt": "一只猫"})
        return {"text": "在右边建好了配置节点并开跑。", "tool_results": [{"ok": True}]}

    out = asyncio.run(agent.run_canvas_agent(conn, "在这张图右边生成一张类似的", snapshot, generate=fake_driver))
    check("对话: 回话原样透出", out["text"] == "在右边建好了配置节点并开跑。", out["text"])
    check("对话: 5 条 op 交给前端", len(out["ops"]) == 5, [item["type"] for item in out["ops"]])
    check("对话: 无错误", not out["errors"], out["errors"])
    check("对话: 记下 2 次工具调用", len(out["toolCalls"]) == 2, out["toolCalls"])
    check("对话: 用本仓指令（不是空串）", seen["instructions"] == agent.CANVAS_INSTRUCTIONS)
    check("对话: 步数上限 8", seen["max_steps"] == agent.MAX_STEPS == 8, seen["max_steps"])
    check("对话: 温度低（摆画布不需要发挥）", seen["model_settings"] == {"temperature": 0.2},
          seen["model_settings"])

    # 参数坏掉时：错误回灌模型，不进 ops
    async def bad_driver(**kwargs):
        result = await kwargs["tools"].get("canvas_apply_ops").run({"ops": [{"type": "nope"}]})
        seen["bad"] = result
        return {"text": "失败", "tool_results": []}

    bad = asyncio.run(agent.run_canvas_agent(conn, "随便", None, generate=bad_driver))
    check("对话: 非法 op 记进 errors", bool(bad["errors"]), bad["errors"])
    check("对话: 非法 op 不落进 ops", bad["ops"] == [], bad["ops"])
    check("对话: 错误回灌给模型", "error" in (seen.get("bad") or {}), seen.get("bad"))

    # ---------------------------------------------------------------- ⑤ 端点与错误
    for label, payload, config in (
        ("空 message", "", {"model": "demo"}),
        ("没有模型配置", "摆两个节点", {}),
    ):
        agent.get_text_config = lambda _conn, _c=config: _c  # type: ignore[assignment]
        raised = False
        try:
            asyncio.run(agent.run_canvas_agent(conn, payload, None, generate=fake_driver))
        except ValueError:
            raised = True
        check(f"错误: {label} ⇒ ValueError（路由转 400）", raised)

    from app.main import app  # noqa: PLC0415

    paths = app.openapi()["paths"]
    check("端点: POST /canvas/agent 已挂", "/api/v1/canvas/agent" in paths, sorted(
        item for item in paths if item.startswith("/api/v1/canvas")))

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed_ok, detail in _RESULTS:
        print(("PASS  " if passed_ok else "FAIL  ") + name + ("" if passed_ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
