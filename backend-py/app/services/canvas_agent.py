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

import json
import re
from typing import Any

from sqlalchemy.engine import Connection

from app.agent.runtime import default_generate
from app.agent.tool import Tool, ToolRegistry
from app.services.ai_providers import get_text_config

from . import canvas_ops, canvas_plan

__all__ = [
    "CANVAS_INSTRUCTIONS",
    "MAX_STEPS",
    "PLANNER_INSTRUCTIONS",
    "build_canvas_tools",
    "run_canvas_agent",
    "run_canvas_planner",
]

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
6. **先看上次为什么没跑起来**：快照里节点带 `lastError`，表示它上一次没跑起来（缺首帧、
   档位类别不符、缺件、空提示词…）。要再跑就先解决那句话里的条件（换对应档位的卡 / 连一张
   图片卡 / 补提示词），**不要**原样重试 —— 同样的卡同样的参数只会再失败一次。

## 回话
做完后用一两句中文说明你做了什么（新建了什么、跑了什么），**不要**罗列 op 清单，
也**不要**把你的思考过程写出来。
"""


def build_canvas_tools(
    sink: dict[str, Any],
    snapshot: dict[str, Any] | None = None,
    scene: str = "canvas",
) -> ToolRegistry:
    """把 `canvas_ops.TOOLS` 包成 `Tool` 注册表 ✓。

    ``sink`` 是本次 run 的收集袋：``{"ops": [...], "errors": [...], "calls": [...]}``。

    ``scene`` 决定模型能看见哪些工具（``canvas_ops.tools_for_scene``）✓。默认 ``canvas``
    = 全量，与历史行为一致；``chat`` 只给读 ＋ 整批落地四件。
    场景配置有病时**抛错**（``ValueError``）—— 空工具表会让模型平静地回一句「我做不到」，
    那种失败在日志里长得跟成功一样 ✗。
    """
    sink.setdefault("ops", [])
    sink.setdefault("errors", [])
    sink.setdefault("calls", [])

    names = canvas_ops.tools_for_scene(scene)
    tools: dict[str, Tool] = {}
    for name in names:
        description, params = canvas_ops.TOOLS[name]
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
    scene: str = "canvas",
) -> dict[str, Any]:
    """跑一次画布 Agent，返回 ``{text, ops, errors, toolCalls}`` ✓。

    ``generate`` 可注入（测试用假驱动，不联网）。抛错由调用方决定文案
    （路由统一 400 —— 与本仓其它 Agent 端点一致）。``scene`` 见 ``build_canvas_tools``。
    """
    text = (message or "").strip()
    if not text:
        raise ValueError("message 不能为空")

    text_config = get_text_config(conn)
    model = str(text_config.get("model") or "")
    if not model:
        raise ValueError("没有可用的文本模型配置（app_settings 里未设置）")

    sink: dict[str, Any] = {}
    tools = build_canvas_tools(sink, snapshot, scene)
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


# ---------------------------------------------------------------- 策划模式（只出计划，不出 op）

#: 策划模式的指令 ✓。关键差别：**不给工具**，要模型回一份 JSON 计划。
#:
#: 为什么值得多这一条通路：`run_canvas_agent` 那条路上模型边想边调工具，
#: 它得自己算坐标、自己编 id、自己决定该连谁 —— 这些都不该是语言模型的活。
#: 而策划模式下它只回答「放哪几个、谁接谁」，坐标与 id 由 `canvas_plan` 确定性地算 ✓。
PLANNER_INSTRUCTIONS = """你是一名短剧画布策划。你的唯一任务：把用户的一句话变成**一份画布计划**。

只输出一个 JSON 对象，不要解释、不要 markdown 代码围栏、不要在 JSON 前后写字。

字段：
{
  "goal": "<逐字复制用户的原话，一个字都不许改>",
  "items": [
    {
      "nodeType": "text" | "image" | "video" | "audio" | "config" | "group",
      "ref": "p1",
      "text": "（文本节点必填）这张卡上要显示的内容",
      "title": "可选：卡片标题",
      "url": "可选：已知的素材地址",
      "workflowId": "可选：配置节点用的工作流 id",
      "values": {"参数名": "值"},
      "after": ["p0"],
      "run": true,
      "mode": "text" | "image" | "video" | "audio",
      "prompt": "（run 时必填）真正送去生成的正向提示词"
    }
  ],
  "options": {"layout": "flow"}
}

硬规则（违反会被编译器直接拒掉）：
1. goal 必须**逐字**是用户的原话。你不是编辑，不许润色、不许补全、不许缩写。
2. **不要写 x / y**。排版由编译器按 options.layout 算（flow = 按 after 分列）。
   只有用户明确说了坐标才写。
3. 先后与连线用 ref + after 表达，不要手写坐标去"暗示"关系。
4. 文本节点必须有 text；这里要写**画面/内容**，不是复述用户的指令。
5. 配置节点（config）必须有 workflowId 或 values，否则它点了什么也不会发生。
6. 只有 config 能写 run: true，且 run 时必须给 mode 与 prompt。
7. 一次最多 8 项。用户要的是一句话的可视化，不是把整集搬上画布。
8. after 可以指向本计划的 ref，也可以指向**画布上已有的节点 id**（用户说"接在选中的图后面"时用）。
   不确定那个 id 存在就不要指 —— 编译器会拒。
9. 画布里已经有的信息不要重复建；这份计划只负责**新建**。
"""

#: 模型有时仍会包一层 ```json ``` —— 剥掉再解析，不为这个把整次请求判失败 ✓
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _parse_plan_json(raw: str, goal_fallback: str) -> dict[str, Any]:
    """把模型回的话解析成计划 ✓。

    两条兜底都很克制：
    - 剥代码围栏（模型的老毛病，不值得据此判失败）；
    - ``goal`` 缺失时**填用户原话**（本来就是逐字复制，缺了就是没照做，补原话比补空串正确 ✓）。

    其余一律不修：items 不合法就让编译器去拒，报错文案比这里猜得准 ✓。
    """
    text = _FENCE.sub("", (raw or "").strip()).strip()
    if not text:
        raise ValueError("模型没有返回计划（空回复）")
    try:
        plan = json.loads(text)
    except json.JSONDecodeError as error:
        head = text[:200].replace("\n", " ")
        raise ValueError(f"模型返回的不是 JSON 计划：{head}（解析失败：{error.msg}）") from error
    if not isinstance(plan, dict):
        raise ValueError("模型返回的计划不是 JSON 对象")
    if not isinstance(plan.get("goal"), str) or not plan.get("goal", "").strip():
        plan["goal"] = goal_fallback
    return plan


async def run_canvas_planner(
    conn: Connection,
    message: str,
    snapshot: dict[str, Any] | None = None,
    *,
    generate: Any = None,
) -> dict[str, Any]:
    """一句话 → **一份已编译的计划**（草案），返回 ``{plan, compiled, text}`` ✓。

    与 `run_canvas_agent` 的分工：
    - 这条**不给工具**、单步完成，模型只回 JSON 计划 ⇒ 可确认、可留痕、可复算；
    - op 的下发不在这里 —— 走 `POST /canvas/plans/{id}/confirm` ✓。

    ``compiled`` 里**有 ops**（编译产物），调用方决定是落盘当草案还是直接下发。
    缺模型配置、模型不按格式回、计划不合法（``PlanRejected``）都会抛错 ✓。
    """
    text = (message or "").strip()
    if not text:
        raise ValueError("message 不能为空")

    text_config = get_text_config(conn)
    model = str(text_config.get("model") or "")
    if not model:
        raise ValueError("没有可用的文本模型配置（app_settings 里未设置）")

    driver = generate or default_generate
    result = await driver(
        config=text_config,
        model=model,
        instructions=PLANNER_INSTRUCTIONS,
        # 空工具表：策划阶段模型不该动手，它一动手就会绕回"自己编坐标"的老路 ✗
        tools=ToolRegistry({}),
        message=text,
        max_steps=1,
        model_settings={"temperature": 0.2},
    )

    raw = str(result.get("text") or "")
    plan = _parse_plan_json(raw, text)
    compiled = canvas_plan.compile_plan(plan, snapshot)
    return {"plan": plan, "compiled": compiled, "text": raw}
