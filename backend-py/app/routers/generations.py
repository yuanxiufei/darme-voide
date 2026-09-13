"""``/api/v1/generations`` —— 生成历史聚合（1 端点，对齐 ``routes/generations.ts``）。

聚合 ``image_generations`` + ``video_generations`` 两张表，返回统一结构，
供前端「生成历史」面板展示（按时间倒序，支持类型 / 项目 / 分镜过滤）。

⚠️ 三个要在意的小语义：

* ``Number(q.limit || 100)`` —— ``||`` 而非 ``??``：``limit=''`` 走默认 100，
  但 ``limit='0'`` 是**非空字符串**（JS 真值）⇒ 得到 0 ⇒ 被 clamp 成 1。
* ``Number(q.drama_id || NaN)`` —— 拿不到就 NaN ⇒ ``isFinite`` 为假 ⇒ **不过滤**（而不是报错）。
* ``limit`` 同时用于 SQL 的 ``LIMIT`` 与最后的内存 ``slice``。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..db import get_conn
from ..models import image_generations, video_generations
from ..response import js_number, js_truthy, success

router = APIRouter(prefix="/api/v1/generations", tags=["generations"])


def _parse_js_date(value: Any) -> int | None:
    """对齐 ``new Date(x).getTime()``（整数毫秒；非法 → None，即 JS 的 NaN）。

    * 带 ``Z`` 或显式偏移的按该时区解释；
    * 不带时区的：JS 按**本地时区**解释，Python 的 ``naive.timestamp()`` 同样按本地时区
      ⇒ 行为一致；
    * 仅日期的 ``2026-01-01`` 是已知偏差（JS 按 UTC，Python 按本地），真实数据都是完整 ISO 串。
    """
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return int(round(parsed.timestamp() * 1000))


def _elapsed(created_at: Any, completed_at: Any) -> int | None:
    """对齐 ``elapsed()``：任一缺失/非法，或 end < start ⇒ null。"""
    if not created_at or not completed_at:
        return None
    start = _parse_js_date(created_at)
    end = _parse_js_date(completed_at)
    if start is None or end is None or end < start:
        return None
    return end - start


def _clamp_limit(raw: Any) -> int:
    """``Number.isFinite(x) ? min(max(x,1),500) : 100``。"""
    value = js_number(raw)
    if value is None:
        return 100
    return int(min(max(value, 1), 500))


def _filter_number(raw: Any) -> Any:
    """``Number(q || NaN)`` 后判有限性：不合法一律 None（= 不过滤）。"""
    value = js_number(raw if js_truthy(raw) else float("nan"))
    return None if value is None else value


def _image_item(r: Any) -> dict[str, Any]:
    d = dict(r._mapping)
    return {
        "id": d["id"],
        "type": "image",
        "storyboardId": d["storyboard_id"] if d["storyboard_id"] is not None else None,
        "dramaId": d["drama_id"] if d["drama_id"] is not None else None,
        "provider": d["provider"] if d["provider"] is not None else None,
        "model": d["model"] if d["model"] is not None else None,
        "prompt": d["prompt"] if d["prompt"] is not None else None,
        # `?? 'pending'` ⇒ 空串保留，只有 null 才回退
        "status": d["status"] if d["status"] is not None else "pending",
        "errorMsg": d["error_msg"] if d["error_msg"] is not None else None,
        "taskId": d["task_id"] if d["task_id"] is not None else None,
        "url": d["image_url"] if d["image_url"] is not None else None,
        "createdAt": d["created_at"],
        "completedAt": d["completed_at"] if d["completed_at"] is not None else None,
        "elapsedMs": _elapsed(d["created_at"], d["completed_at"]),
    }


def _video_item(r: Any) -> dict[str, Any]:
    d = dict(r._mapping)
    return {
        "id": d["id"],
        "type": "video",
        "storyboardId": d["storyboard_id"] if d["storyboard_id"] is not None else None,
        "dramaId": d["drama_id"] if d["drama_id"] is not None else None,
        "provider": d["provider"] if d["provider"] is not None else None,
        "model": d["model"] if d["model"] is not None else None,
        "prompt": d["prompt"] if d["prompt"] is not None else None,
        "status": d["status"] if d["status"] is not None else "pending",
        "errorMsg": d["error_msg"] if d["error_msg"] is not None else None,
        "taskId": d["task_id"] if d["task_id"] is not None else None,
        "url": d["video_url"] if d["video_url"] is not None else None,
        "duration": d["duration"] if d["duration"] is not None else None,
        "createdAt": d["created_at"],
        "completedAt": d["completed_at"] if d["completed_at"] is not None else None,
        "elapsedMs": _elapsed(d["created_at"], d["completed_at"]),
    }


@router.get("")
def list_generations(request: Request, conn: Connection = Depends(get_conn)):
    q = request.query_params
    gen_type = q.get("type")
    limit = _clamp_limit(q.get("limit") if js_truthy(q.get("limit")) else 100)
    drama_id = _filter_number(q.get("drama_id"))
    storyboard_id = _filter_number(q.get("storyboard_id"))

    want_image = not js_truthy(gen_type) or gen_type == "image"
    want_video = not js_truthy(gen_type) or gen_type == "video"

    items: list[dict[str, Any]] = []

    if want_image:
        where = []
        if drama_id is not None:
            where.append(image_generations.c.drama_id == drama_id)
        if storyboard_id is not None:
            where.append(image_generations.c.storyboard_id == storyboard_id)
        stmt = select(image_generations)
        if where:
            stmt = stmt.where(*where)
        rows = conn.execute(
            stmt.order_by(image_generations.c.created_at.desc()).limit(limit)
        ).all()
        items.extend(_image_item(r) for r in rows)

    if want_video:
        where = []
        if drama_id is not None:
            where.append(video_generations.c.drama_id == drama_id)
        if storyboard_id is not None:
            where.append(video_generations.c.storyboard_id == storyboard_id)
        stmt = select(video_generations)
        if where:
            stmt = stmt.where(*where)
        rows = conn.execute(
            stmt.order_by(video_generations.c.created_at.desc()).limit(limit)
        ).all()
        items.extend(_video_item(r) for r in rows)

    # TS: String(b.createdAt).localeCompare(String(a.createdAt)) —— ISO 串的字典序即时间倒序
    items.sort(key=lambda i: str(i["createdAt"] or ""), reverse=True)
    return success(items[:limit])
