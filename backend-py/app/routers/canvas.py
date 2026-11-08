"""**画布工具域** —— 把 LLM 的工具调用编译成本仓画布可执行的 op 列表。

与 ``services/canvas_ops.py`` 成对：本文件只做 HTTP 包装（收快照、编译、校验），
**不碰画布**也**不起生成任务** ✓ —— 执行在前端 ``utils/canvasOps.ts``，生成在生成域。

为什么要多一跳而不是让前端自己编 op：op 的形状（``x/y/w/h``、``values`` 全是字符串、
视口夹 0.2–3）是**一整套口径**，散落在前端各处迟早分叉；收在一处才有一份真源 ✓。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.engine import Connection
from starlette.requests import Request

from ..core.db import get_conn
from ..core.request_utils import read_json
from ..core.response import bad_request, not_found, success
from ..services import canvas_agent, canvas_ops, canvas_plan, canvas_plan_store, canvas_plugins

router = APIRouter()

#: 单次最多接受的 op 条数（再多说明调用方在拿 op 当脚本用 ✗）
MAX_OPS = 200


async def _json_body(request: Request) -> dict[str, Any]:
    """读 JSON 体 ✓；**空体当 ``{}``** —— 确认端点允许不带体（幂等重放时前端可以什么都不传）。"""
    try:
        body = await read_json(request)
    except Exception:  # noqa: BLE001 —— 空体/坏体统一按「没给」处理，由各端点自己判缺参
        return {}
    return body if isinstance(body, dict) else {}


def _as_revision(value: Any) -> int | None:
    """画布版本号 ✓；不是非负整数就当没给（不猜、不四舍五入）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


@router.get("/tools")
def tools(scene: str = Query(default="")) -> object:
    """工具清单（function-calling 形态）—— 给 agent 拼 tools 参数用 ✓。

    ``?scene=chat|canvas`` 只返回该场景可见的工具 ✓。不给就是全量（老调用方不变）。
    场景名写错或白名单配坏 ⇒ **400 并说清哪条不对**，不返回空清单 ——
    空清单让调用方以为「这个场景本来就没有工具」，问题被藏进成功响应里 ✗。
    """
    try:
        selected = canvas_ops.tools_for_scene(scene) if scene.strip() else canvas_ops.TOOL_NAMES
    except ValueError as error:
        return bad_request(str(error))
    return success({
        "tools": canvas_ops.tool_schemas(scene.strip() or None),
        "names": list(selected),
        "scene": scene.strip() or None,
        "scenes": list(canvas_ops.SCENES),
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


@router.post("/plan")
async def create_plan(request: Request) -> object:
    """**起草** —— 把一份计划编译成「将要发生的动作」并落盘，但**不下发 op** ✓。

    请求体（两种写法等价）：::

        {"plan": {"goal": "...", "items": [...]}, "snapshot": {...}, "baseRevision": 7}
        {"goal": "...", "items": [...], "snapshot": {...}, "baseRevision": 7}

    草案阶段**就编译一次**：缺参、节点类型不认识、依赖成环、连线指向不存在的项 —— 这些问题
    在这里一次报全（``errors`` 数组），而不是等用户点了确认才翻车 ✗。

    返回里没有 ``ops``：要拿可执行物，走 ``POST /canvas/plans/{planId}/confirm`` ✓。
    ``preflight`` 的数字（建几个 / 连几条 / 跑几次）是给人看的，用来在确认前发现「AI 理解歪了」✓。
    """
    body = await _json_body(request)
    plan = body.get("plan")
    if not isinstance(plan, dict):
        # 扁平写法：省掉一层 {"plan": ...}（前端手写计划时更顺）
        plan = {key: body.get(key) for key in ("kind", "goal", "items", "edges", "options")}
        plan = {key: value for key, value in plan.items() if value is not None}
    if not isinstance(plan, dict) or not plan:
        return bad_request("缺计划：给 {\"plan\": {...}} 或直接给 goal + items")
    snapshot = body.get("snapshot")
    if snapshot is not None and not isinstance(snapshot, dict):
        return bad_request("snapshot 要是对象")

    try:
        compiled = canvas_plan.compile_plan(plan, snapshot)
    except canvas_plan.PlanRejected as error:
        # 一次报全：让模型/用户一轮改完，不玩「改一条报一条」✗
        return bad_request("计划未通过编译", errors=error.errors)

    record = canvas_plan_store.draft_record(
        plan, compiled, expected_revision=_as_revision(body.get("baseRevision"))
    )

    # 同一份计划重复起草 ⇒ 同一个 id（指纹）。若它**已经确认过**，不要把状态偷偷退回 draft ✗
    try:
        existing = canvas_plan_store.load_plan(record["id"])
    except canvas_plan_store.PlanNotFound:
        existing = None
    if existing and existing.get("status") == canvas_plan_store.STATUS_CONFIRMED:
        view = canvas_plan_store.public_view(existing)
        view["alreadyConfirmed"] = True
        view["note"] = "这份计划已经确认过；要重来请改计划内容（id 由计划内容决定）"
        return success(view)

    canvas_plan_store.save_plan(record)
    view = canvas_plan_store.public_view(record)
    view["note"] = "草案已落盘，尚未下发可执行 op；确认请走 POST /canvas/plans/{planId}/confirm"
    return success(view)


@router.get("/plans")
def plans(limit: int = Query(default=20, ge=1, le=100)) -> object:
    """最近的计划草案（新的在前）✓ —— 供「AI 想干什么」列表用。"""
    return success({"plans": canvas_plan_store.list_plans(limit)})


@router.get("/plans/{plan_id}")
def plan_detail(plan_id: str) -> object:
    """单份计划详情 ✓。draft 阶段同样**不含 ops**（要 op 就去确认）。"""
    try:
        record = canvas_plan_store.load_plan(plan_id)
    except canvas_plan_store.PlanNotFound:
        return not_found("没有这份计划")
    return success(canvas_plan_store.public_view(record, with_ops=False))


@router.post("/plans/{plan_id}/confirm")
async def confirm_plan(plan_id: str, request: Request) -> object:
    """**确认** —— 用**当下**的快照重编译，然后才下发 ``ops`` ✓。

    重编译而不是直接放行草案里那份 ops：草案期间用户可能删掉了计划要连的那个节点，
    直接放行会留下一根断线 ✗；重编译就让编译器在确认这一步把问题挡下来 ✓。

    幂等：已确认的计划再确认，返回**上次缓存的 ops** 并标 ``replayed: true`` ——
    前端据此不会把同一批节点建两遍 ✓（本仓画布在前端，后端只能靠这个标记替前端兜底）。

    ``expectedRevision`` 是**前端画布的版本**，后端校验不了（画布状态不在后端）。
    它被原样带回去，前端在 ``applyCanvasOps`` 时用它做 CAS：
    草案期间画布被别人改过 ⇒ 版本对不上 ⇒ 前端整批拒绝并让用户重看 ✓。
    """
    body = await _json_body(request)
    try:
        record = canvas_plan_store.load_plan(plan_id)
    except canvas_plan_store.PlanNotFound:
        return not_found("没有这份计划")

    revision = _as_revision(body.get("baseRevision"))
    if revision is None:
        revision = record.get("expectedRevision")

    if record.get("status") == canvas_plan_store.STATUS_CONFIRMED:
        view = canvas_plan_store.public_view(record, with_ops=True)
        view["replayed"] = True
        view["note"] = "这份计划已确认过：返回的是上次下发的 ops，不会重复建节点"
        return success(view)

    snapshot = body.get("snapshot")
    if snapshot is not None and not isinstance(snapshot, dict):
        return bad_request("snapshot 要是对象")

    try:
        compiled = canvas_plan.compile_plan(record.get("plan") or {}, snapshot)
    except canvas_plan.PlanRejected as error:
        # 计划本身没变，问题一定来自**当下的画布**（比如要连的节点没了）
        return bad_request("确认失败：当下画布与这份计划对不上", errors=error.errors)

    record.update(
        {
            "status": canvas_plan_store.STATUS_CONFIRMED,
            "confirmedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "confirmCount": int(record.get("confirmCount") or 0) + 1,
            "expectedRevision": revision,
            "ops": compiled["ops"],
            "refs": compiled["refs"],
            "preflight": compiled["preflight"],
            "layout": compiled["layout"],
            "warnings": compiled["warnings"],
        }
    )
    canvas_plan_store.save_plan(record)
    view = canvas_plan_store.public_view(record, with_ops=True)
    view["replayed"] = False
    view["note"] = "已确认：把 ops 连同 expectedRevision 交给前端 applyCanvasOps"
    return success(view)


@router.post("/plan/auto")
async def create_plan_auto(request: Request, conn: Connection = Depends(get_conn)) -> object:
    """**一句话 → 草案**（模型只出计划，编译器出 op，此时仍不下发 op）✓。

    请求体：``{"message": "...", "snapshot": {...}?, "baseRevision": 7?}``

    与 ``POST /canvas/plan`` 的分工：那个收**已经写好的计划**（给工具链/测试用），
    这个让**模型写计划**（给用户用）。两条最终落进同一个草案仓 ✓。

    模型没按格式回、或计划不合法 ⇒ 400 + ``errors`` 数组（一次报全）✓。
    ``baseRevision`` 原样存进草案，确认时再带回前端做 CAS ✓。
    """
    body = await _json_body(request)
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return bad_request("缺 message（一句话要画什么）")
    snapshot = body.get("snapshot")
    if snapshot is not None and not isinstance(snapshot, dict):
        return bad_request("snapshot 要是对象")

    try:
        planned = await canvas_agent.run_canvas_planner(conn, message, snapshot)
    except canvas_plan.PlanRejected as error:
        return bad_request("模型给出的计划不合法", errors=error.errors)
    except ValueError as error:
        return bad_request(str(error))

    compiled = planned["compiled"]
    record = canvas_plan_store.draft_record(
        planned["plan"], compiled, expected_revision=_as_revision(body.get("baseRevision"))
    )
    try:
        existing = canvas_plan_store.load_plan(record["id"])
    except canvas_plan_store.PlanNotFound:
        existing = None
    if existing and existing.get("status") == canvas_plan_store.STATUS_CONFIRMED:
        view = canvas_plan_store.public_view(existing)
        view["alreadyConfirmed"] = True
        view["note"] = "这份计划已经确认过（同一句话得到同一份计划）；要重来请改变措辞"
        return success(view)

    canvas_plan_store.save_plan(record)
    view = canvas_plan_store.public_view(record)
    view["modelText"] = planned["text"]
    view["note"] = "草案已就绪：核对 preflight 的数字，确认后才会下发 op"
    return success(view)


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

    请求体：``{"message": "在这张图右边生成一张类似的", "snapshot": {...}?, "scene": "chat"?}``

    响应里的 ``ops`` **尚未执行** —— 由前端 ``utils/canvasOps.ts`` 按顺序落地，
    落地后前端再回一次 ``/ops/validate`` 或自行核对。服务端不碰画布（也碰不到）。

    ``scene`` 决定模型能看见哪些工具（见 ``canvas_ops.SCENE_TOOL_SETS``）：
    ``chat`` 只给读 ＋ 整批落地四件，``canvas``（默认）给全量。场景名写错 ⇒ 400 ✓。
    """
    body = await read_json(request)
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return bad_request("缺 message")
    snapshot = body.get("snapshot")
    if snapshot is not None and not isinstance(snapshot, dict):
        return bad_request("snapshot 要是对象")
    scene = body.get("scene")
    scene = "canvas" if scene is None else str(scene)
    try:
        result = await canvas_agent.run_canvas_agent(conn, message, snapshot, scene=scene)
    except ValueError as error:
        return bad_request(str(error))
    return success(result)
