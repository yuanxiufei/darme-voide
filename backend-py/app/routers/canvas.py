"""**画布工具域** —— 把 LLM 的工具调用编译成本仓画布可执行的 op 列表。

与 ``services/canvas_ops.py`` 成对：本文件只做 HTTP 包装（收快照、编译、校验），
**不碰画布**也**不起生成任务** ✓ —— 执行在前端 ``utils/canvasOps.ts``，生成在生成域。

为什么要多一跳而不是让前端自己编 op：op 的形状（``x/y/w/h``、``values`` 全是字符串、
视口夹 0.2–3）是**一整套口径**，散落在前端各处迟早分叉；收在一处才有一份真源 ✓。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.engine import Connection
from starlette.requests import Request

from ..core.db import get_conn
from ..core.request_utils import read_json
from ..core.response import bad_request, success
from ..services import canvas_agent, canvas_ops, canvas_plugins

router = APIRouter()

#: 单次最多接受的 op 条数（再多说明调用方在拿 op 当脚本用 ✗）
MAX_OPS = 200


@router.get("/tools")
def tools() -> object:
    """工具清单（function-calling 形态）—— 给 agent 拼 tools 参数用 ✓。"""
    return success({
        "tools": canvas_ops.tool_schemas(),
        "names": list(canvas_ops.TOOL_NAMES),
        "nodeTypes": list(canvas_ops.NODE_TYPES),
        "generationModes": list(canvas_ops.GENERATION_MODES),
        "opTypes": list(canvas_ops.OP_TYPES),
        "dropped": canvas_ops.DROPPED_TOOLS,
        "note": "dropped 是源有、本仓没有宿主的工具：不是漏迁，逐条都有理由",
    })


@router.post("/tool")
async def call_tool(request: Request) -> object:
    """编译一次工具调用 ✓。

    请求体：``{"name": "canvas_generate_image", "input": {...}, "snapshot": {...}?}``

    ``snapshot`` 缺省时不报错：只影响 ``nextCanvasX``（默认落在 0）与连线存在性校验
    （无快照就不校验节点是否存在，交给前端 ``connect`` 挡）。
    """
    body = await read_json(request)
    if not isinstance(body, dict):
        return bad_request("请求体要是对象")
    name = body.get("name")
    if not isinstance(name, str) or not name:
        return bad_request("缺 name（画布工具名）")
    payload = body.get("input")
    if payload is not None and not isinstance(payload, dict):
        return bad_request("input 要是对象")
    snapshot = body.get("snapshot")
    if snapshot is not None and not isinstance(snapshot, dict):
        return bad_request("snapshot 要是对象")
    try:
        result = canvas_ops.build_request(name, payload or {}, snapshot)
    except ValueError as error:
        return bad_request(str(error))
    return success(result)


@router.post("/ops/validate")
async def validate(request: Request) -> object:
    """单独校验一批 op（前端执行前自检用）✓。**有错误时不执行**，由调用方整批放弃。"""
    body = await read_json(request)
    ops = body.get("ops") if isinstance(body, dict) else None
    if isinstance(ops, list) and len(ops) > MAX_OPS:
        return bad_request(f"一次最多 {MAX_OPS} 条 op，收到 {len(ops)} 条")
    snapshot = body.get("snapshot") if isinstance(body, dict) else None
    clean, errors = canvas_ops.validate_ops(ops, snapshot if isinstance(snapshot, dict) else None)
    return success({"ops": clean, "errors": errors, "ok": not errors})


@router.post("/ops/snapshot")
async def snapshot(request: Request) -> object:
    """压缩画布快照（正文截断、只留判断所需字段）✓，给模型看布局用。"""
    body = await read_json(request)
    raw = body.get("snapshot") if isinstance(body, dict) else None
    if not isinstance(raw, dict):
        return bad_request("缺 snapshot")
    try:
        return success(canvas_ops.compact_snapshot(raw))
    except ValueError as error:
        return bad_request(str(error))


@router.get("/next-x")
def next_x(request: Request = None, snapshot: str = Query(default="")) -> object:  # noqa: B008
    """新节点默认横坐标（便于前端在**没有**快照时也对齐同一口径）。

    收 JSON 字符串而不是对象：GET 带体在部分代理上会被丢掉 ✗。
    """
    import json

    state = None
    if snapshot.strip():
        try:
            state = json.loads(snapshot)
        except json.JSONDecodeError:
            return bad_request("snapshot 不是合法 JSON")
        if not isinstance(state, dict):
            return bad_request("snapshot 要是对象")
    return success({"x": canvas_ops.next_canvas_x(state)})


@router.get("/plugins")
def plugins() -> object:
    """插件注册表（含校验报告与 ``hosted`` 标记）✓。

    ``hosted=false`` 是**如实交代**：契约与组件名已就位，但画布还没接渲染槽。
    """
    return success(canvas_plugins.registry())


@router.post("/plugins/validate")
async def validate_plugins(request: Request) -> object:
    """校验一批插件清单（插件作者提交前自检，或第三方插件准入）✓。

    有错误时返回 ``ok=false`` 且**不给**清洗后的清单：插件不像 op，
    不允许「丢掉错的那条照样装」✗。
    """
    body = await read_json(request)
    plugins = body.get("plugins") if isinstance(body, dict) else None
    if not isinstance(plugins, list):
        return bad_request("缺 plugins 数组")
    return success(canvas_plugins.validate_bundle(plugins))


@router.post("/agent")
async def agent(request: Request, conn: Connection = Depends(get_conn)) -> object:
    """一句话 → 一串画布 op ✓。

    请求体：``{"message": "在这张图右边生成一张类似的", "snapshot": {...}?}``

    响应里的 ``ops`` **尚未执行** —— 由前端 ``utils/canvasOps.ts`` 按顺序落地，
    落地后前端再回一次 ``/ops/validate`` 或自行核对。服务端不碰画布（也碰不到）。
    """
    body = await read_json(request)
    if not isinstance(body, dict):
        return bad_request("请求体要是对象")
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return bad_request("缺 message")
    snapshot = body.get("snapshot")
    if snapshot is not None and not isinstance(snapshot, dict):
        return bad_request("snapshot 要是对象")
    try:
        result = await canvas_agent.run_canvas_agent(conn, message, snapshot)
    except ValueError as error:
        return bad_request(str(error))
    return success(result)
