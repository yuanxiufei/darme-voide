"""``/api/v1/presets`` —— 与 ``backend/src/routes/presets.ts`` 对齐。

4 个端点：列表（可按 ``type`` 过滤）、创建、更新、删除。用正常信封（``code: 200``）。

⚠️ 返回的是 **camelCase**（drizzle 行），且 ``config`` 是**反序列化后的对象**
（解析失败为 ``null``，不是原始字符串）。``config`` 这个键名在两侧同名，无需转换。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import delete, desc, select, update
from sqlalchemy.engine import Connection

from ..db import get_conn, get_tx
from ..models import presets as presets_table
from ..request_utils import read_json
from ..response import bad_request, not_found, now, parse_param_id, row_to_camel, success
from ..services.color_grade import normalize_color_grade

router = APIRouter(prefix="/api/v1/presets", tags=["presets"])


def _to_api(row: Any) -> dict[str, Any]:
    """对齐 TS ``toApi``：``{...row, config: 解析后的对象或 null}``。"""
    mapped = row_to_camel(row, "presets")
    config = None
    if mapped.get("config"):
        try:
            config = json.loads(mapped["config"])
        except (ValueError, TypeError):
            config = None
    mapped["config"] = config
    return mapped


@router.get("")
def list_presets(type: str | None = None, conn: Connection = Depends(get_conn)):
    try:
        stmt = select(presets_table).order_by(desc(presets_table.c.updated_at))
        if type:
            stmt = (
                select(presets_table)
                .where(presets_table.c.type == type)
                .order_by(desc(presets_table.c.updated_at))
            )
        rows = conn.execute(stmt).all()
        return success([_to_api(r) for r in rows])
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.post("")
async def create_preset(request: Request, conn: Connection = Depends(get_tx)):
    try:
        body = await read_json(request)
        ptype = body.get("type")
        name = body.get("name")
        if not ptype or not name:
            return bad_request("type and name are required")

        # TS: `let config = body.config ?? null` —— 空值合并不做真值回退（false/0/'' 会保留）
        config = body.get("config")
        # colorGrade 类型：规整 config 为合法校色参数（这里用的是真值判断）
        if ptype == "colorGrade" and config:
            config = normalize_color_grade(config)

        ts = now()
        result = conn.execute(
            presets_table.insert().values(
                type=ptype,
                name=name,
                # ⚠️ 紧凑分隔符（Node 是 `JSON.stringify(config)`）
            config=json.dumps(config, ensure_ascii=False, separators=(",", ":")) if config else None,
                created_at=ts,
                updated_at=ts,
            )
        )
        row = conn.execute(
            select(presets_table).where(presets_table.c.id == result.inserted_primary_key[0])
        ).first()
        return success(_to_api(row) if row is not None else None)
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.put("/{preset_id}")
async def update_preset(preset_id: str, request: Request, conn: Connection = Depends(get_tx)):
    try:
        pid = parse_param_id(preset_id)
        if pid is None:
            return not_found("Invalid preset id")

        body = await read_json(request)
        values: dict[str, Any] = {"updated_at": now()}
        # TS 用 `!= null`（宽松）：null 与 undefined 都跳过
        if body.get("name") is not None:
            values["name"] = body["name"]
        if body.get("type") is not None:
            values["type"] = body["type"]
        if body.get("config") is not None:
            existing = conn.execute(
                select(presets_table.c.type).where(presets_table.c.id == pid)
            ).first()
            if existing is None:
                return not_found("Preset not found")
            # TS: body.type ?? existing.type —— nullish，传 null 时用库内值
            ptype = body["type"] if body.get("type") is not None else existing.type
            config = body["config"]
            values["config"] = json.dumps(
                normalize_color_grade(config) if ptype == "colorGrade" else config,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        conn.execute(update(presets_table).where(presets_table.c.id == pid).values(**values))
        row = conn.execute(select(presets_table).where(presets_table.c.id == pid)).first()
        return success(_to_api(row) if row is not None else None)
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.delete("/{preset_id}")
def delete_preset(preset_id: str, conn: Connection = Depends(get_tx)):
    try:
        pid = parse_param_id(preset_id)
        if pid is None:
            return not_found("Invalid preset id")
        conn.execute(delete(presets_table).where(presets_table.c.id == pid))
        return success()
    except Exception as exc:  # noqa: BLE001
        # TS 此处是 {code:500, data:null, message}
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})
