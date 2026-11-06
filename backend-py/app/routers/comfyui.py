"""ComfyUI 能力路由 —— **本项目自己的入口**（2026-09-17）。

⚠️ 这不是「把别人的接口搬过来」✗：运行的**记录/恢复/产物落地/记账**都在
``services/comfyui.py`` 里（我们自己的 DB 与数据根 ✓），本路由只做「参数收口 + 响应封装」✓。

* ``POST /api/v1/comfyui/runs``             入队一次运行 → ``{id, status}``（**立即返回**，后台跑 ✓）
* ``GET  /api/v1/comfyui/runs``             列出（``?status=`` / ``?limit=`` ✓）
* ``GET  /api/v1/comfyui/runs/retention``   **只算不删**：按保留期摊开「会清掉多少」✓
* ``GET  /api/v1/comfyui/runs/{id}``        单条（含产物、耗时、错误、warnings ✓）
* ``POST /api/v1/comfyui/runs/{id}/cancel`` 取消（**上游打断 + 本地标记** 两半都做 ✓）
* ``GET  /api/v1/comfyui/system``           合并视图：上游体检 + **我们自己的运行统计** ✓
* ``GET  /api/v1/comfyui/catalog/nodes``    节点目录（``?class_name=`` 给单类详情 ✓）
* ``GET  /api/v1/comfyui/catalog/models``   模型类别 / ``?folder=`` 该类文件清单 ✓
* ``POST /api/v1/comfyui/free``             让上游卸载模型、释放显存 ✓
* ``GET  /api/v1/comfyui/catalog/workflows`` 工作流目录（**本地资产** ✓ 不需要上游在线）
* ``GET  /api/v1/comfyui/catalog/workflows/{id}`` 单条详情（变体 / 参数 / **步数估算** ✓）
* ``GET  /api/v1/comfyui/catalog/prompts``  提示词模板目录（增强 / 反推四份，**本地资产** ✓）
* ``POST /api/v1/comfyui/catalog/prompts/{id}/compile`` 填参数 ⇒ **可提交的 API 图** ✓（只编译，不提交）
* ``POST /api/v1/comfyui/workflows/infer``  未登记工作流 ⇒ **参数落点推断** ✓（只读图、不写盘、不碰上游）
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.engine import Connection

from ..core.db import get_conn, get_tx
from ..core.response import bad_request, not_found, parse_param_id, success
from ..core.request_utils import read_json
from ..services import comfyui as svc
from ..services import comfyui_catalog as catalog
from ..services import comfyui_infer as infer
from ..services import comfyui_prompts as prompts

router = APIRouter(prefix="/api/v1/comfyui", tags=["comfyui"])


@router.post("/runs")
async def create_run(request: Request, conn: Connection = Depends(get_tx)):
    """入队一次运行（``kind=h3`` 走 H3 视频协议；``kind=workflow`` 跑任意工作流 ✓）。"""
    try:
        body = await read_json(request)
        if str(body.get("kind") or "workflow") == "workflow" and not (
                body.get("workflow") or body.get("workflowPath")):
            return bad_request("kind=workflow 时必须给 workflow（UI/API 格式 JSON）或 workflowPath ✓")
        run_id = await svc.submit_run(conn, body)
        row = svc.get_run(conn, run_id)
        return success(svc.run_to_dict(row))
    except Exception as err:  # noqa: BLE001
        return bad_request(str(err))


@router.post("/validate")
async def validate_workflow(request: Request):
    """**只校验不提交** ✓（缺节点包 / 连线悬空 / 参数注入点找不到 ⇒ 生成前就说清楚 ✓）。

    body 与 ``POST /runs`` **同一个形状** ✓（``workflowId`` / ``variant`` / ``values`` /
    ``prompt`` / ``workflow`` / ``workflowPath``）——
    两份形状迟早会漂成「校验通过、提交报错」，那校验就成了骗人的绿灯 ✗。
    """
    try:
        body = await read_json(request)
        if not (body.get("workflowId") or body.get("workflow") or body.get("workflowPath")):
            return bad_request("需要 workflowId / workflow / workflowPath 之一")
        # 与提交**同一份前置** ✓：校验时素材也真的传一遍，否则校验的是"没有参考图的图" ✗
        await svc.prepare_reference_media(body)
        return success(await svc.validate_workflow_body(svc.build_submit_body(body)))
    except ValueError as err:
        return bad_request(str(err))
    except Exception as err:  # noqa: BLE001
        return bad_request(f"校验失败：{err}")


@router.get("/runs")
def list_runs(status: str | None = None, limit: int = 50, conn: Connection = Depends(get_conn)):
    """列出运行记录（**读的是我们自己的表** ✓ 不是上游历史 ✗）。"""
    rows = svc.list_runs(conn, status=status, limit=limit)
    return success({"runs": [svc.run_to_dict(row) for row in rows]})


@router.get("/runs/retention")
def runs_retention(days: int | None = None, conn: Connection = Depends(get_conn)):
    """**只算不删**：按保留期摊开「会清掉多少条 / 多少个文件 / 多少字节」✓。

    这条**永远 ``dry_run=True``** ✓ —— 真删只有 ``app/scripts/comfyui_retention.py --apply``
    一个入口 ✓。理由：删是不可逆的，把「能删」做成一个 HTTP GET，等于让人在浏览器地址栏
    里就能毁掉产物 ✗；而「还剩多少要清」是只读信息，前端随时该能看 ✓。

    ``?days=`` 不传就按 ``COMFYUI_RETENTION_DAYS`` ⇒ 缺省 :data:`~app.services.comfyui.RETENTION_DAYS` ✓；
    ``days<=0`` 一律按「不清理」回 ``runs=0``（别把 0 读成"全清" ✗）。

    ⚠️ **必须声明在 ``/runs/{run_id}`` 之前** ✓：FastAPI 按声明顺序匹配，写在后面会被
    ``/runs/{run_id}`` 先吃掉 ⇒ ``retention`` 当 id 解析失败回 400「run_id 非法」✗。
    ``tests/comfyui_retention_test.py::case_preview_route`` 盯着这条顺序 ✓。
    """
    try:
        report = svc.purge_runs(conn, days=days, dry_run=True)
    except Exception as err:  # noqa: BLE001
        return bad_request(f"清理预演失败：{err}")
    return success(report)


@router.get("/runs/{run_id}")
def get_run(run_id: str, conn: Connection = Depends(get_conn)):
    """单条运行详情（含 ``outputs`` / ``localPath`` / ``elapsedMs`` / ``errorMsg`` ✓）。"""
    parsed = parse_param_id(run_id)
    if parsed is None:
        return bad_request("run_id 非法")
    row = svc.get_run(conn, parsed)
    if row is None:
        return not_found("运行记录不存在")
    return success(svc.run_to_dict(row))


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str, conn: Connection = Depends(get_tx)):
    """取消运行（**两半**：让上游打断 + 本地标记 ✓ 只做一半就是假的取消 ✗）。"""
    parsed = parse_param_id(run_id)
    if parsed is None:
        return bad_request("run_id 非法")
    row = await svc.cancel_run(conn, parsed)
    if row is None:
        return not_found("运行记录不存在")
    return success(svc.run_to_dict(row))


@router.get("/system")
async def system(conn: Connection = Depends(get_conn)) -> Any:
    """系统视图：上游（8765/ComfyUI）体检 **+ 我们自己的运行统计** ✓。"""
    return success(await svc.system_snapshot(conn))


@router.get("/catalog/nodes")
async def catalog_nodes(class_name: str | None = None) -> Any:
    """节点目录（``class_name`` 给了就回该类详情 ✓）—— 入口在我们的路由上 ✓。"""
    try:
        path = "/v1/catalog/nodes"
        return success(await svc.upstream_get(path, {"class_name": class_name} if class_name else None))
    except Exception as err:  # noqa: BLE001
        return bad_request(f"上游不可达：{err}")


@router.get("/catalog/models")
async def catalog_models(folder: str | None = None) -> Any:
    """模型类别 / 某类文件清单 ✓。"""
    try:
        return success(await svc.upstream_get("/v1/catalog/models", {"folder": folder} if folder else None))
    except Exception as err:  # noqa: BLE001
        return bad_request(f"上游不可达：{err}")


@router.post("/free")
async def free() -> Any:
    """让上游卸载模型、释放显存 ✓（长片生成前后都值得调一次 ✓）。

    运行链路**自己也会调**：上游报显存不足时先 ``/v1/free`` 再重提交一次（只一次 ✓），
    所以这里的手动口是给「还没跑但已知显存紧」的场合 ✓，不是唯一入口 ✓。
    """
    return success({"ok": await svc.free_upstream()})


def _workflow_brief(row: dict[str, Any]) -> dict[str, Any]:
    """entry → 对外字段（**只挑前端画图要的** ✓，不把整份注册表甩给前端 ✗）。"""
    variants = []
    for key in catalog.variant_ids(row):
        overlay = (row.get("variants") or {}).get(key) or {}
        variants.append({"id": key, "label": overlay.get("label") or key})
    return {
        "id": row.get("id"),
        "name": row.get("name") or row.get("label"),
        "type": row.get("type"),
        "category": row.get("category"),
        "description": row.get("description"),
        "file": row.get("file"),
        "variants": variants,
        "parameters": [item for item in (row.get("parameters") or []) if isinstance(item, dict)],
        "minMedia": row.get("minMedia"),
    }


@router.get("/catalog/workflows")
def catalog_workflows() -> Any:
    """工作流目录（**本地资产** ✓）：不依赖上游在线 ⇒ 上游挂了也能看有哪些工作流 ✓。

    隐藏项（``hidden``）不给 ✓ —— 它们是可选项的旧入口，不该出现在下拉里 ✗。
    """
    try:
        rows = catalog.entries()
    except catalog.CatalogError as err:
        return bad_request(str(err))
    return success({"workflows": [_workflow_brief(row) for row in rows]})


@router.get("/catalog/workflows/{workflow_id}")
def catalog_workflow(workflow_id: str, variant: str | None = None) -> Any:
    """单条详情：变体 / 参数 / **步数估算** ✓（步数用来归一进度条 ✓）。

    ``?variant=`` 给了就按变体读图（变体不在册 ⇒ 400 ✓，不静默回落主形态 ✗）。
    """
    entry = catalog.get_entry(workflow_id)
    if entry is None:
        return not_found(f"没有工作流 {workflow_id}")
    try:
        graph = catalog.load_graph(entry, variant)
    except catalog.CatalogError as err:
        return bad_request(str(err))
    return success(
        {
            "workflow": _workflow_brief(entry),
            "variant": variant,
            "stepPlan": catalog.estimate_step_plan(graph),
            "nodeCount": len(graph),
        }
    )


@router.post("/workflows/infer")
async def infer_workflow(request: Request) -> Any:
    """**未登记**工作流 ⇒ 推断「哪些格是提示词/尺寸/种子/参考图」✓（只读图、不写盘、不碰上游 ✓）。

    body::

        {"workflow": {…API 格式图…}, "name": "可选", "type": "image|video|audio|text"}

    ``type`` 不给就按输出节点猜（``SaveImage`` / ``VHS_VideoCombine`` / ``SaveAudio``）✓。
    回执的 ``bindings`` / ``parameters`` **就是本仓 registry 的形状** ⇒ 前端拿同一套渲染，
    人工过目后写进 ``registry.json`` 即可当正式工作流用 ✓（推断只是起草，不自动登记 ✗）。

    ⚠️ 两种失败必须当场说 ✗ 而不是给一份"看起来能用"的回执：
    ① 交来的是 **UI 格式**（带 ``nodes`` 键）——那不是能直接提交的图，得先在 ComfyUI 里导出 API 格式；
    ② 图里**没有输出节点** —— 没有 ``SaveImage`` 之类的终点，提交上去只会跑一遍什么都不落盘。
    另外 ``warnings`` 里会明说「哪些格子推不出来」，``missing`` 是要人工补的键（前端据此隐掉那几项）。
    """
    try:
        body = await read_json(request)
    except Exception as err:  # noqa: BLE001
        return bad_request(f"body 不是合法 JSON：{err}")
    graph = body.get("workflow") or body.get("graph")
    if not isinstance(graph, dict):
        return bad_request("需要 workflow（API 格式的工作流 JSON）")
    hint = body.get("type") or body.get("hint")
    try:
        report = infer.infer(graph, name=body.get("name"), hint=hint)
    except infer.InferError as err:
        # 断言式前缀：前端/脚本按 `[ui_format]` 这类前缀分流，比读自然语言可靠 ✓
        return bad_request(f"[{err.code}] {err}")
    return success(report)


@router.get("/catalog/prompts")
def catalog_prompts() -> Any:
    """提示词模板目录（**本地资产** ✓）：四份增强 / 反推模板 + 各自参数 + 所需节点类 ✓。

    不需要上游在线、不碰显卡 ✓（模板就是 ``configs/comfyui/prompts/*.json`` ✓）。
    ``requiredNodes`` 拿去和 ``GET /catalog/nodes`` 比，缺哪个插件当场能看出来 ✓ ——
    这四份依赖上游插件（``TE_*`` / ``llama_cpp`` / ``Florence2``）✗，不是本仓那 21 条出图工作流 ✓。
    """
    return success(prompts.catalog())


@router.post("/catalog/prompts/{template_id}/compile")
async def compile_prompt_template(template_id: str, request: Request) -> Any:
    """把参数填进模板 ⇒ **一张可直接提交的 API 图** ✓（**只编译，不提交、不连上游** ✓）。

    请求体：``{"values": {...}}``（也接受把 ``values`` 的键平铺在顶层 ✓）。
    返回的 ``workflow`` 原样塞进 ``POST /api/v1/comfyui/runs`` 的 ``workflow`` 就跑 ✓
    —— 由那条路负责 DB 记录 / 记账 / 产物落地 / 失败归因 ✓（本路由只做"填参数"这一件事 ✓）。
    """
    try:
        body = await read_json(request)
        values = body.get("values") if isinstance(body.get("values"), dict) else body
        return success(prompts.compile_template(template_id, values))
    except prompts.PromptTemplateError as err:
        if err.code in {"unknown_template", "missing_template"}:
            return not_found(str(err))
        return bad_request(str(err))
    except Exception as err:  # noqa: BLE001
        return bad_request(str(err))
