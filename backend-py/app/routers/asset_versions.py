"""``/api/v1/asset-versions`` —— 与 ``backend/src/routes/asset-versions.ts`` 对齐（2 端点）。

* ``GET /``：资产版本列表（版本号倒序）
* ``POST /{id}/activate``：回滚到指定版本（并把 asset_url 写回主表）

用正常信封（``code: 200``）。返回的版本行是 **camelCase**（drizzle 行形状）；
``GET /`` 的顶层是 snake_case（``asset_type`` / ``asset_id``）+ ``versions``。

⚠️ id 校验用的是 ``Number.isFinite(Number(x))`` —— **不管正负、不管是否整数**，
所以 ``/asset-versions/-1/activate`` 会走到查库才失配（报 ``Version not found``）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.engine import Connection

from ..core.db import get_conn, get_tx
from ..core.response import bad_request, js_number, success
from ..services.asset_versions import activate_asset_version, list_asset_versions

router = APIRouter(prefix="/api/v1/asset-versions", tags=["asset-versions"])


@router.get("")
def list_versions(request: Request, conn: Connection = Depends(get_conn)):
    try:
        asset_type = request.query_params.get("asset_type")
        asset_id_raw = request.query_params.get("asset_id")
        if not asset_type or not asset_id_raw:
            return bad_request("asset_type and asset_id are required")
        asset_id = js_number(asset_id_raw)
        if asset_id is None:
            return bad_request("invalid asset_id")

        versions = list_asset_versions(conn, asset_type, asset_id)
        return success({"asset_type": asset_type, "asset_id": asset_id, "versions": versions})
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc) or "list asset versions failed")


@router.post("/{version_id}/activate")
def activate_version(version_id: str, conn: Connection = Depends(get_tx)):
    try:
        vid = js_number(version_id)
        if vid is None:
            return bad_request("invalid version id")

        result = activate_asset_version(conn, vid)
        if not result.get("ok"):
            return bad_request(result.get("error") or "activate failed")
        return success({"activated": result.get("row")})
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc) or "activate asset version failed")
