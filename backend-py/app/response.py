"""统一响应层 —— 与 ``backend/src/utils/response.ts`` 逐字段对齐。

实测契约（前端 ``frontend/app/composables/useApi.ts`` 的消费方式）：

* 成功：HTTP 200 + ``{"code": 200, "data": ..., "message": "success"}``
* 创建：HTTP 201 + ``{"code": 201, "data": ..., "message": "created"}``
* 失败：HTTP 4xx/5xx + ``{"code": <status>, "message": "..."}``  ← **没有 data 字段**

前端判据是 ``!resp.ok || (json.code && json.code >= 400)``，读 ``json.message`` 取错误文案，
返回 ``json.data ?? json``。故上表任何一项写错都会直接导致前端报错文案异常。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi.responses import JSONResponse

# Node 侧 table 名 → 本模块函数名，便于对照阅读
_OK = 200
_CREATED = 201
_BAD_REQUEST = 400
_NOT_FOUND = 404
_CONFLICT = 409
_SERVER_ERROR = 500


def success(data: Any = None) -> JSONResponse:
    return JSONResponse(status_code=_OK, content={"code": _OK, "data": data, "message": "success"})


def created(data: Any = None) -> JSONResponse:
    return JSONResponse(status_code=_CREATED, content={"code": _CREATED, "data": data, "message": "created"})


def bad_request(message: str = "bad request") -> JSONResponse:
    return JSONResponse(status_code=_BAD_REQUEST, content={"code": _BAD_REQUEST, "message": message})


def not_found(message: str = "not found") -> JSONResponse:
    return JSONResponse(status_code=_NOT_FOUND, content={"code": _NOT_FOUND, "message": message})


def conflict(message: str = "conflict") -> JSONResponse:
    return JSONResponse(status_code=_CONFLICT, content={"code": _CONFLICT, "message": message})


def server_error(message: str = "internal error") -> JSONResponse:
    return JSONResponse(status_code=_SERVER_ERROR, content={"code": _SERVER_ERROR, "message": message})


def now() -> str:
    """与 JS ``new Date().toISOString()`` 同形：``2026-09-12T07:11:22.123Z``。"""
    ts = datetime.now(timezone.utc)
    return f"{ts.strftime('%Y-%m-%dT%H:%M:%S')}.{ts.microsecond // 1000:03d}Z"


def parse_param_id(raw: Any, *, positive_only: bool = True) -> int | None:
    """安全解析路由参数 id，非法时返回 None（对齐 Node 的 parseParamId）。"""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if positive_only and value <= 0:
        return None
    return value


# ---------------------------------------------------------------------------
# 行 → dict / 字段映射工具
# ---------------------------------------------------------------------------

def row_to_dict(row: Any) -> dict[str, Any]:
    """SQLAlchemy Row → dict。列名本就是 snake_case，等于契约要求的形状。"""
    return dict(row._mapping) if row is not None else {}


def rows_to_dicts(rows: Any) -> list[dict[str, Any]]:
    return [dict(r._mapping) for r in rows]


def camel_to_snake(key: str) -> str:
    """``episodeNumber`` → ``episode_number``（对齐 Node 的 key.replace(/[A-Z]/g, ...)）。"""
    out: list[str] = []
    for ch in key:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def pick_fields(src: dict[str, Any], snake_keys: list[str]) -> dict[str, Any]:
    """白名单取字段，**snake_case 优先、camelCase 兜底**。

    对齐 Node ``routes/dramas.ts`` 的 ``pickFields``：先看 snake 键是否存在，再看 camel 键。
    目的是兼容历史上前端双写两种命名的调用方，同时避免把未知列写进 SQL 触发 no such column。
    """
    out: dict[str, Any] = {}
    for key in snake_keys:
        camel = "".join(part.capitalize() if i else part for i, part in enumerate(key.split("_")))
        if key in src:
            out[key] = src[key]
        elif camel in src:
            out[key] = src[camel]
    return out


def parse_json_array(raw: Any) -> list[Any]:
    """``tags`` 这类以 JSON 文本落库的列 → list（解析失败回退空数组，和 Node 的语义一致）。"""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        import json

        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except (ValueError, TypeError):
        return []
