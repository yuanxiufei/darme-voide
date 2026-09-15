"""visual-graph 域 —— 与 ``backend/src/routes/visual-graph.ts``（66 行）对齐。

**4 个端点全部迁移**（前缀 ``/api/v1/visual-graph``）：

* ``GET /``                图谱全量（四类知识节点，可按 ``category`` 过滤）；
* ``GET /resolve?text=``   中文视觉术语 → 英文电影术语（供 prompt 调试）；
* ``GET /terms?category=`` 某类的全部术语；
* ``GET /guidance?drama_id=`` 按剧类型/风格生成拆镜引导文本（``text/plain``）。

⚠️ 三处**与其它域不同**的返回形态（别"顺手统一"成标准信封）：

1. 前三个端点返回的是**裸 JSON**（``{"graph":…,"categories":…}``），**没有** ``code/data/message`` 信封；
2. 显式设置 ``Content-Type: application/json; charset=utf-8``；
3. ``/guidance`` 返回的是 **``text/plain`` 纯文本**（不是 JSON），且 ``drama_id`` 传了不存在时是 404。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..core.db import get_conn
from ..core.models import dramas
from ..core.response import bad_request, js_number, not_found
from ..services.visual_graph import (
    build_visual_graph_guidance,
    get_visual_graph,
    list_visual_terms,
    resolve_visual_term,
)

router = APIRouter(prefix="/api/v1/visual-graph", tags=["visual-graph"])

#: 四类知识节点（顺序即前端展示顺序）
CATEGORIES = ("shot_size", "composition", "movement", "lighting")

_JSON_TYPE = "application/json; charset=utf-8"


def _raw_json(payload: dict) -> JSONResponse:
    """裸 JSON（**不套** code/data/message 信封）+ 显式 charset。"""
    return JSONResponse(content=payload, media_type=_JSON_TYPE)


@router.get("")
def graph_root(request: Request):
    """图谱全量；``category`` 非法（或没传）时返回**全部**四类。"""
    category_raw = request.query_params.get("category")
    category = category_raw if category_raw in CATEGORIES else None
    return _raw_json({"graph": get_visual_graph(category), "categories": list(CATEGORIES)})


@router.get("/resolve")
def graph_resolve(request: Request):
    text = request.query_params.get("text")
    if not text:
        return bad_request('query param "text" is required')
    return _raw_json({"zh": text, "en": resolve_visual_term(text)})


@router.get("/terms")
def graph_terms(request: Request):
    category_raw = request.query_params.get("category")
    if category_raw not in CATEGORIES:
        return bad_request(f"category must be one of: {' | '.join(CATEGORIES)}")
    return _raw_json({"category": category_raw, "terms": list_visual_terms(category_raw)})


@router.get("/guidance")
def graph_guidance(request: Request, conn: Connection = Depends(get_conn)):
    """按剧类型/风格生成引导文本；**没有** ``drama_id``（或非数字）时用全局默认引导。"""
    drama_id_raw = request.query_params.get("drama_id")
    # `dramaIdRaw && !Number.isNaN(Number(dramaIdRaw))`
    if drama_id_raw and js_number(drama_id_raw) is not None:
        drama_id = js_number(drama_id_raw)
        row = conn.execute(select(dramas).where(dramas.c.id == drama_id)).first()
        if row is None:
            return not_found("Drama not found")
        return PlainTextResponse(
            build_visual_graph_guidance(row.genre, row.style), media_type="text/plain; charset=utf-8"
        )
    # 无 drama_id 时使用全局默认引导（无类型匹配，仅通用规则）
    return PlainTextResponse(
        build_visual_graph_guidance(None, None), media_type="text/plain; charset=utf-8"
    )
