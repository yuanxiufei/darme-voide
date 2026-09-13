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
import os
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.engine import Connection
from starlette.background import BackgroundTask

from ..db import get_conn
from ..models import dramas
from ..services.export_service import (
    build_edl,
    build_export_zip,
    collect_drama_export_files,
)
from ..services.jianying_draft import build_jianying_draft
from ..services.task_logger import log_task_error
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


# ---------------------------------------------------------------------------
# GET /dramas/{id}/edl — CMX3600 格式 EDL（Premiere / 达芬奇导入）
# ---------------------------------------------------------------------------

@router.get("/dramas/{drama_id}/edl")
async def export_edl(drama_id: str, request: Request, conn: Connection = Depends(get_conn)):
    """导出 EDL（**裸 Response**，非统一信封 —— 前端当下载链接用）。"""
    try:
        did = parse_param_id(drama_id)
        if did is None:
            return not_found("Invalid drama id")
        if not _require_drama(conn, did):
            return not_found("Drama not found")

        # ⚠️ 两个参数都用 `Number(x)` 语义（NaN -> None；fps 允许小数）；空串 -> 0（falsy，等同未传）
        episode_id = js_number(request.query_params.get("episodeId"))
        fps = js_number(request.query_params.get("fps"))

        edl = await build_edl(conn, did, {"episodeId": episode_id, "fps": fps})
        return Response(
            content=edl,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="drama-{did}.edl"'},
        )
    except Exception as exc:  # noqa: BLE001
        log_task_error("ExportAPI", "edl", {"error": str(exc), "dramaId": drama_id})
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# GET /dramas/{id}?scope=all|video|assets — 打包导出 ZIP
# ---------------------------------------------------------------------------

async def _zip_chunks(path: str) -> AsyncIterator[bytes]:
    """分块吐 ZIP（1 MiB/块）。"""
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            yield chunk


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


@router.get("/dramas/{drama_id}/jianying-draft")
def export_jianying_draft(drama_id: str, request: Request,
                          conn: Connection = Depends(get_conn)):
    """导出剪映草稿（``.draft`` 文件夹打包 ZIP）；用户可在剪映里继续调字幕/配音/节奏。"""
    try:
        did = parse_param_id(drama_id)
        if did is None:
            return not_found("Invalid drama id")
        if not _require_drama(conn, did):
            return not_found("Drama not found")

        episode_id = js_number(request.query_params.get("episodeId"))
        drafts = build_jianying_draft(conn, did, episode_id)
        files = [entry for draft in drafts for entry in draft["files"]]
        if not files:
            return bad_request("No composed videos found for draft export")

        zip_path = build_export_zip(files)
        filename = (f"drama-{did}-episode-{episode_id}-jianying-draft.zip" if episode_id
                    else f"drama-{did}-jianying-draft.zip")
        return StreamingResponse(
            _zip_chunks(zip_path),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Draft-Count": str(len(drafts)),
            },
            background=BackgroundTask(_remove_quietly, zip_path),
        )
    except Exception as exc:  # noqa: BLE001
        log_task_error("ExportAPI", "jianying-draft", {"error": str(exc), "dramaId": drama_id})
        return bad_request(str(exc))


@router.get("/dramas/{drama_id}")
def export_drama_zip(drama_id: str, request: Request, conn: Connection = Depends(get_conn)):
    """打包导出整剧（``scope=video`` 只要成片 / ``assets`` 只要源素材 / ``all`` 默认全要）。"""
    try:
        did = parse_param_id(drama_id)
        if did is None:
            return not_found("Invalid drama id")
        if not _require_drama(conn, did):
            return not_found("Drama not found")

        scope = js_nullish(request.query_params.get("scope"), "all")
        if scope not in ("all", "video", "assets"):
            return bad_request("scope must be one of: all | video | assets")

        files = collect_drama_export_files(conn, did, scope)
        if not files:
            return bad_request("No exportable assets found (videos/assets not generated yet)")

        # ⚠️ 与 TS 的流式压缩等价（这里写临时文件再分块吐，见 export_service 模块头）
        zip_path = build_export_zip(files)
        return StreamingResponse(
            _zip_chunks(zip_path),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="drama-{did}-export.zip"',
                "X-Export-Count": str(len(files)),
            },
            background=BackgroundTask(_remove_quietly, zip_path),
        )
    except Exception as exc:  # noqa: BLE001
        log_task_error("ExportAPI", "download", {"error": str(exc), "dramaId": drama_id})
        return bad_request(str(exc))
