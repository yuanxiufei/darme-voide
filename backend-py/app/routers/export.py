"""``/api/v1/export`` —— 与 ``backend/src/routes/export.ts`` 对齐（**已迁移 2 / 7 端点**）。

**已迁移**：工程账本 JSON/Markdown、断点续作 stale 扫描（这两个是**纯 DB**）
**未迁移**：``/edl``（需 ffprobe 探时长）、``/dramas/:id`` 打包 ZIP、``/jianying-draft``（339 行 +
时长探测）、``/qc-report``（ffprobe）、``/contact-sheet``（515 行 + 媒体探测）

⚠️ **本域没有统一信封**：这些端点返回的是**文件本体**（JSON / Markdown / ZIP / EDL），
并带 ``Content-Disposition: attachment``。所以这里用的是裸 ``Response``，不是 ``success()``。
前端拿它们当下载链接，不是当接口数据用。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..db import get_conn
from ..models import dramas
from ..response import bad_request, js_nullish, js_number, js_truthy, not_found, parse_param_id
from ..services.project_ledger import (
    build_project_ledger,
    build_project_ledger_markdown,
    scan_stale_shots,
)

router = APIRouter(prefix="/api/v1/export", tags=["export"])


def _optional_int(raw: str | None) -> int | float | None:
    """``raw && !Number.isNaN(Number(raw)) ? Number(raw) : undefined``。

    注意与原 TS 同形：空串 / 缺失 / 非数值一律**视为未传**（不是报错）。
    """
    if not js_truthy(raw):
        return None
    return js_number(raw)


def _require_drama(conn: Connection, drama_id: Any) -> bool:
    return (
        conn.execute(select(dramas.c.id).where(dramas.c.id == drama_id)).first() is not None
    )


# ---------------------------------------------------------------------------
# 断点续作扫描（**必须早于同前缀的 project-ledger 注册**，与 TS 的顺序一致）
# ---------------------------------------------------------------------------

@router.get("/dramas/{drama_id}/project-ledger/stale")
def project_ledger_stale(drama_id: str, request: Request, conn: Connection = Depends(get_conn)):
    try:
        did = parse_param_id(drama_id)
        if did is None:
            return not_found("Invalid drama id")

        episode_id = _optional_int(request.query_params.get("episodeId"))
        result = scan_stale_shots(conn, did, episode_id)
        # 原 TS 先设 Content-Type 再 c.json() ⇒ 裸 JSON，无信封
        return Response(
            # ⚠️ 紧凑分隔符：Node 是 Hono 的 `c.json(...)`（内部就是 JSON.stringify），
            #    默认的 `json.dumps` 会多出空格，导出的 JSON 与 Node 不一致
            content=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            media_type="application/json; charset=utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# 工程账本（json / md）
# ---------------------------------------------------------------------------

@router.get("/dramas/{drama_id}/project-ledger")
def project_ledger(drama_id: str, request: Request, conn: Connection = Depends(get_conn)):
    try:
        did = parse_param_id(drama_id)
        if did is None:
            return not_found("Invalid drama id")
        if not _require_drama(conn, did):
            return not_found("Drama not found")

        episode_id = _optional_int(request.query_params.get("episodeId"))
        # `?? 'json'` 只看 null ⇒ 用 js_nullish（空串会原样保留，但两者都落进 JSON 分支）
        fmt = js_nullish(request.query_params.get("format"), "json")

        if fmt in ("md", "markdown"):
            markdown = build_project_ledger_markdown(conn, did, episode_id)
            return Response(
                content=markdown,
                media_type="text/markdown; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="drama-{did}-project-ledger.md"'
                },
            )

        ledger = build_project_ledger(conn, did, episode_id)
        return Response(
            # ⚠️ 同上：`c.json(ledger)` 是紧凑的
            content=json.dumps(ledger, ensure_ascii=False, separators=(",", ":")),
            media_type="application/json; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="drama-{did}-project-ledger.json"'
            },
        )
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))
