"""**画布 Agent** —— 一句话 → 一串画布 op（工具只编译，**不执行**）。

源：`infinite-canvas/canvas-agent`（MCP 服务端 + `codex app-server --stdio` + 34 工具）
⇒ **本文件按本仓重写**：不启子进程、不连 MCP，直接用本仓已有的文本适配器 + 工具环 ✓。

## 与"普通 Agent"最关键的分歧：工具**不产生副作用**

本仓其它 Agent 的工具直接改 DB。画布工具**什么都不做**，只把一个工具调用**编译**成
op 列表交给前端 —— 因为画布状态在前端（`useCanvas` 的 localStorage），服务端看不见也改不了。
所以：

* `execute` 只算 op，`sink` 里攒着，随响应回给前端一次性执行；
* 工具回给模型的是「已登记 N 条 op」这种人话，**不是执行结果**（此刻还没执行 ✗）；
* `errors` 原样回灌 —— 模型能据此重试，比默默丢掉强 ✓。

## 三条硬规则（写进指令，也由代码兜底）

1. **只读工具必须真给数据**：模型看不到画布就会凭空编节点 id ⇒ `canvas_get_state`
   返回**压缩快照**（`compact_snapshot`，正文截断、未知字段不外泄）✓；没传快照时
   如实回 `error`，不返回空对象假装"画布是空的"✗。
2. **不许编 id**：`validate_ops` 带快照校验，指向不存在节点的 op 直接报错 ✓。
3. **指令与实现不打架**：工具清单从 `canvas_ops.TOOLS` **取值**生成，参数 schema 走
   `canvas_ops.param_schema`（同一份）—— 不手抄第二套 ✗。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.engine import Connection

from app.agent.runtime import default_generate
from app.agent.tool import Tool, ToolRegistry
from app.services.ai_providers import get_text_config

from . import canvas_ops

__all__ = ["CANVAS_INSTRUCTIONS", "MAX_STEPS", "build_canvas_tools", "run_canvas_agent"]

#: 步数上限：源 canvas-agent 给 30 步，本仓工具都被"大工具"包过了
#: （一次 `canvas_generate_image` 就顶好几条 add_node），8 步足够；多出来的步数
#: 只会让模型开始折腾坐标 ✗
MAX_STEPS = 8

CANVAS_INSTRUCTIONS = """你是一名无限画布助手，负责把用户的话变成画布操作。

## 你能做什么
画布上有五类节点：文本(text)、图片(image)、视频(video)、配置(config)、分组(group)。
配置节点绑定一个工作流 + 一组参数，是"生成"的入口；文本节点可以作为提示词，被配置节点引用。

## 硬规则
1. **先读再动**：不确定画布现状时先调 `canvas_get_state`。不要凭想象编造节点 id ——
   不存在的 id 会被拒绝。
2. **优先用大工具**：要"生成一张图"就直接 `canvas_generate_image`（它会自己建提示词节点、
   建配置节点、连线、选中并开跑），**不要**手写一串 `canvas_create_node`。
   只有用户要的是"摆几个空节点"这类纯布局需求时才用 `canvas_apply_ops` / `canvas_create_*`。
3. **坐标不要自己编**：省略 `x`/`y`，工具会按现有画布排版。用户明确说了位置才给。
4. **最小改动**：用户没要求的节点不要动、不要删。
5. **一次说完**：把用户的需求在**一轮**里做完，不要反复试探。

## 回话
做完后用一两句中文说明你做了什么（新建了什么、跑了什么），**不要**罗列 op 清单，
也**不要**把你的思考过程写出来。
"""


def build_canvas_tools(sink: dict[str, Any], snapshot: dict[str, Any] | None = None) -> ToolRegistry:
    """把 `canvas_ops.TOOLS` 包成 `Tool` 注册表 ✓。

    ``sink`` 是本次 run 的收集袋：``{"ops": [...], "errors": [...], "calls": [...]}``。
    """
    sink.setdefault("ops", [])
    sink.setdefault("errors", [])
    sink.setdefault("calls", [])

    tools: dict[str, Tool] = {}
    for name, (description, params) in canvas_ops.TOOLS.items():
        tools[name] = Tool(
            id=name,
            description=description,
            input_schema={
                "type": "object",
                "properties": {
                    key: canvas_ops.param_schema(key, desc) for key, desc in params.items()
                },
            },
            execute=_make_execute(name, sink, snapshot),
        )
    return ToolRegistry(tools)


def _make_execute(name: str, sink: dict[str, Any], snapshot: dict[str, Any] | None):
    """生成一个工具的 `execute`（闭包持 name / sink / snapshot）。

    单独成函数是为了**每个工具各持自己的 name** —— 用循环变量闭包会全部指向最后一个 ✗。
    """

    async def execute(arguments: dict[str, Any]) -> dict[str, Any]:
        sink["calls"].append({"toolName": name, "args": arguments})

        # —— 只读工具：把画布**真数据**给模型，否则它会编 id ✗
        if name in canvas_ops.READ_TOOLS:
            if snapshot is None:
                return {"error": "没有画布快照：前端没有把当前画布传上来，读不到内容。"
                                 "请不要凭想象创建/修改节点，先请用户提供画布状态。"}
            try:
                compact = canvas_ops.compact_snapshot(snapshot)
            except ValueError as error:
                return {"error": str(error)}
            if name == "canvas_get_selection":
                return {"selection": compact.get("selection", [])}
            return compact

        # —— 写工具：只编译成 op，不执行
        try:
            compiled = canvas_ops.build_request(name, arguments, snapshot)
        except ValueError as error:
            sink["errors"].append(f"{name}: {error}")
            return {"error": str(error)}

        if compiled.get("errors"):
            sink["errors"].extend(f"{name}: {item}" for item in compiled["errors"])
            # 回灌给模型：它可以改参数重试（比如把流散的 ops 合成一条合法 op）
            return {"error": "；".join(compiled["errors"])}

        ops = compiled.get("ops") or []
        if not ops:
            return {"ok": True, "ops": 0, "note": "这条调用没有产生操作"}
        sink["ops"].extend(ops)
        return {
            "ok": True,
            "ops": len(ops),
            "totalOps": len(sink["ops"]),
            "note": "已登记，前端会按顺序执行（此刻还没有真正落地）",
        }

    return execute


async def run_canvas_agent(
    conn: Connection,
    message: str,
    snapshot: dict[str, Any] | None = None,
    *,
    generate: Any = None,
    max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    """跑一次画布 Agent，返回 ``{text, ops, errors, toolCalls}`` ✓。

    ``generate`` 可注入（测试用假驱动，不联网）。抛错由调用方决定文案
    （路由统一 400 —— 与本仓其它 Agent 端点一致）。
    """
    text = (message or "").strip()
    if not text:
        raise ValueError("message 不能为空")

    text_config = get_text_config(conn)
    model = str(text_config.get("model") or "")
    if not model:
        raise ValueError("没有可用的文本模型配置（app_settings 里未设置）")

    sink: dict[str, Any] = {}
    tools = build_canvas_tools(sink, snapshot)
    driver = generate or default_generate

    result = await driver(
        config=text_config,
        model=model,
        instructions=CANVAS_INSTRUCTIONS,
        tools=tools,
        message=text,
        max_steps=max_steps,
        model_settings={"temperature": 0.2},
    )

    return {
        "text": str(result.get("text") or ""),
        "ops": sink["ops"],
        "errors": sink["errors"],
        "toolCalls": sink["calls"],
        "toolResults": result.get("tool_results") or [],
    }
