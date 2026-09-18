"""ComfyUI 能力路由 —— **本项目自己的入口**（2026-09-17）。

⚠️ 这不是「把别人的接口搬过来」✗：运行的**记录/恢复/产物落地/记账**都在
``services/comfyui.py`` 里（我们自己的 DB 与数据根 ✓），本路由只做「参数收口 + 响应封装」✓。

* ``POST /api/v1/comfyui/runs``             入队一次运行 → ``{id, status}``（**立即返回**，后台跑 ✓）
* ``GET  /api/v1/comfyui/runs``             列出（``?status=`` / ``?limit=`` ✓）
* ``GET  /api/v1/comfyui/runs/{id}``        单条（含产物、耗时、错误、warnings ✓）
* ``POST /api/v1/comfyui/runs/{id}/cancel`` 取消（**上游打断 + 本地标记** 两半都做 ✓）
* ``GET  /api/v1/comfyui/system``           合并视图：上游体检 + **我们自己的运行统计** ✓
* ``GET  /api/v1/comfyui/catalog/nodes``    节点目录（``?class_name=`` 给单类详情 ✓）
* ``GET  /api/v1/comfyui/catalog/models``   模型类别 / ``?folder=`` 该类文件清单 ✓
* ``POST /api/v1/comfyui/free``             让上游卸载模型、释放显存 ✓
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.engine import Connection

from ..core.db import get_conn, get_tx
from ..core.response import bad_request, not_found, parse_param_id, success
from ..core.request_utils import read_json
from ..services import comfyui as svc

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


@router.get("/runs")
def list_runs(status: str | None = None, limit: int = 50, conn: Connection = Depends(get_conn)):
    """列出运行记录（**读的是我们自己的表** ✓ 不是上游历史 ✗）。"""
    rows = svc.list_runs(conn, status=status, limit=limit)
    return success({"runs": [svc.run_to_dict(row) for row in rows]})


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
    """让上游卸载模型、释放显存 ✓（长片生成前后都值得调一次 ✓）。"""
    return success({"ok": await svc.free_upstream()})
