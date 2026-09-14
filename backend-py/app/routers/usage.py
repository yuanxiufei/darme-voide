"""``/api/v1/usage`` —— 用量与成本。

**已整域迁移**：``GET /summary``（用量汇总）、``GET /board``（多集成本看板）、
``GET /estimate``（生成前费用预估）—— 后者 2026-09-15 校正：**后来已迁**
（``estimate-service`` / ``cost-catalog`` 均已落地）。
⚠️ 「哪些没迁」以 ``tests/route_parity_test.py`` 的机械扫描为准。

用正常信封（``code: 200``）。注意 ``/summary`` 的顶层是 camelCase、``records`` 与分组条目
内部是 snake_case；``/board`` 整体 snake_case —— 都是原 TS 的形状，别统一。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.engine import Connection

from ..db import get_conn
from ..response import bad_request, js_number, js_truthy, success
from ..services.estimate_service import estimate_pending_costs
from ..services.usage_tracking import get_episode_cost_board, get_usage_summary

router = APIRouter(prefix="/api/v1/usage", tags=["usage"])


@router.get("/summary")
def usage_summary(request: Request, conn: Connection = Depends(get_conn)):
    try:
        q = request.query_params
        drama_raw = q.get("drama_id")
        drama_id = js_number(drama_raw) if drama_raw else None
        if drama_raw and drama_id is None:
            return bad_request("invalid drama_id")

        ep_raw = q.get("episode_id")
        episode_id = js_number(ep_raw) if ep_raw else None
        if ep_raw and episode_id is None:
            return bad_request("invalid episode_id")

        limit_raw = q.get("limit")
        limit = js_number(limit_raw) if limit_raw else 200

        return success(
            get_usage_summary(conn, drama_id=drama_id, episode_id=episode_id, limit=limit)
        )
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc) or "usage summary failed")


@router.get("/estimate")
def usage_estimate(request: Request, conn: Connection = Depends(get_conn)):
    """生成前费用预估（花前估算，与 `/summary` 的花后记账配套）。

    注意 ``storyboard_ids`` 的解析口径：``split(',').map(Number).filter(isFinite)``
    —— 空段会变成 ``0`` 并被保留（``Number('') === 0``），不是"忽略空段"。
    """
    try:
        q = request.query_params
        drama_raw = q.get("drama_id")
        drama_id = js_number(drama_raw) if drama_raw else None
        # `dramaId == null || !isFinite` ⇒ 缺参也报 invalid
        if drama_id is None:
            return bad_request("invalid drama_id")

        ep_raw = q.get("episode_id")
        episode_id = js_number(ep_raw) if ep_raw else None
        if ep_raw and episode_id is None:
            return bad_request("invalid episode_id")

        storyboard_ids = None
        ids_raw = q.get("storyboard_ids")
        if js_truthy(ids_raw):
            storyboard_ids = [
                v for v in (js_number(part) for part in ids_raw.split(",")) if v is not None
            ]

        return success(
            estimate_pending_costs(
                conn, drama_id, episode_id=episode_id, storyboard_ids=storyboard_ids
            )
        )
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc) or "estimate failed")


@router.get("/board")
def usage_board(request: Request, conn: Connection = Depends(get_conn)):
    try:
        drama_raw = request.query_params.get("drama_id")
        drama_id = js_number(drama_raw) if drama_raw else None
        # TS: `dramaId == null || !Number.isFinite(dramaId)` ⇒ 缺参也报 invalid
        if drama_id is None:
            return bad_request("invalid drama_id")
        return success(get_episode_cost_board(conn, drama_id))
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc) or "cost board failed")
