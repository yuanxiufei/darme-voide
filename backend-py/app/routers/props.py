"""物品库（props）域 —— 与 ``backend/src/routes/props.ts`` 对齐。

**已整域迁移**：`GET /`（列表）、`GET /{id}`、`POST /`、`PUT /{id}`、`DELETE /{id}`、
`POST /{id}/generate-image` —— 后者 2026-09-15 校正：**后来已迁**（``services/image_generation.py``）。
⚠️ 「哪些没迁」以 ``tests/route_parity_test.py`` 的机械扫描为准。

⚠️ 三处本域特有、别抄错的地方：

1. **本文件的 id 解析比 characters/scenes 更严**：TS 用的是自定义 ``parseId``
   （``Number.isInteger(n) && n > 0``），而 characters/scenes 用的是 ``parseParamId``
   （``Number.isFinite`` ⇒ **允许小数**）。故 ``GET /props/1.5`` → ``Invalid prop id``，
   而 ``GET /scenes/1.5`` → ``Scene not found``。**两者不能合并。**
2. **``custom_prompt`` 落到 DB 列 ``image_prompt``**：TS 声明为
   ``customPrompt: text('image_prompt')``，是 schema.ts 里唯一「属性名 ≠ camelCase(列名)」的
   字段（493 个字段实测仅此 1 处）。返回时靠 ``response._COLUMN_TO_JS_OVERRIDES`` 还原成
   ``customPrompt``，**不能想当然转成 ``imagePrompt``**。
3. **DELETE 返回 ``{ok: true}``**（不是 data:null），且是软删。

⚠️ ``GET /`` 的过滤顺序也照抄：先取全表（``deleted_at IS NULL``），再按 ``drama_id``
过滤、按 ``id`` 升序排序 —— 都是内存里做的。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import and_, select, update
from sqlalchemy.engine import Connection

from ..core.db import get_conn, get_tx
from ..core.models import prop_templates
from ..core.request_utils import read_json
from ..core.response import bad_request, js_number, not_found, now, row_to_camel, success
from ..services.image_generation import generate_image
from ..services.prompt_utils import (
    UI_PLATE_CATEGORY,
    build_prop_image_prompt,
    build_ui_plate_image_prompt,
)
from ..services.task_logger import log_task_error, log_task_start

router = APIRouter(prefix="/api/v1/props", tags=["props"])

#: 请求体字段 → DB 列（对齐 TS 的 fieldMap）
_FIELD_MAP: dict[str, str] = {
    "name": "name",
    "category": "category",
    "description": "description",
    "appearance": "appearance",
    "size_hint": "size_hint",
    "holder": "holder",
    "key_clue": "key_clue",
    "custom_prompt": "image_prompt",   # ← 属性名与列名不一致的那个字段
    "negative_prompt": "negative_prompt",
    "image_url": "image_url",
}


def parse_prop_id(raw: str) -> int | None:
    """对齐 TS 的 ``parseId``：``Number.isInteger(n) && n > 0``（比 parse_param_id 更严）。"""
    if raw is None:
        return None
    try:
        text = str(raw).strip()
        value = float(text)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    if not value.is_integer() or value <= 0:
        return None
    return int(value)


def _fetch_prop(conn: Connection, prop_id: int):
    return conn.execute(
        select(prop_templates).where(
            and_(prop_templates.c.id == prop_id, prop_templates.c.deleted_at.is_(None))
        )
    ).first()


# ---------------------------------------------------------------------------
# GET / — 列表（?drama_id= 可选过滤）
# ---------------------------------------------------------------------------

@router.get("")
def list_props(drama_id: str | None = None, conn: Connection = Depends(get_conn)):
    try:
        # TS: Number(c.req.query('drama_id') || '0') ⇒ 空/缺省为 0，0 表示不过滤
        try:
            filter_id = int(float(drama_id)) if drama_id else 0
        except (TypeError, ValueError):
            filter_id = 0

        rows = conn.execute(
            select(prop_templates).where(prop_templates.c.deleted_at.is_(None))
        ).all()
        items = [
            row_to_camel(r, "prop_templates")
            for r in rows
            if not filter_id or r.drama_id == filter_id
        ]
        items.sort(key=lambda x: x["id"])
        return success(items)
    except Exception as exc:  # noqa: BLE001
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


# ---------------------------------------------------------------------------
# GET /{id}
# ---------------------------------------------------------------------------

@router.get("/{prop_id}")
def get_prop(prop_id: str, conn: Connection = Depends(get_conn)):
    try:
        pid = parse_prop_id(prop_id)
        if not pid:
            return not_found("Invalid prop id")
        row = _fetch_prop(conn, pid)
        if row is None:
            return not_found("物品不存在")
        return success(row_to_camel(row, "prop_templates"))
    except Exception as exc:  # noqa: BLE001
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


# ---------------------------------------------------------------------------
# POST / — 创建
# ---------------------------------------------------------------------------

@router.post("")
async def create_prop(request: Request, conn: Connection = Depends(get_tx)):
    try:
        body = await read_json(request)
        drama_id = body.get("drama_id")
        name = body.get("name")
        if not drama_id or not name:
            return bad_request("drama_id 和 name 必填")

        ts = now()
        result = conn.execute(
            prop_templates.insert().values(
                drama_id=int(float(drama_id)),
                name=str(name),
                category=body.get("category") or "道具",
                description=body.get("description"),
                appearance=body.get("appearance"),
                size_hint=body.get("size_hint"),
                holder=body.get("holder"),
                key_clue=body.get("key_clue"),
                image_prompt=body.get("custom_prompt"),
                negative_prompt=body.get("negative_prompt"),
                created_at=ts,
                updated_at=ts,
            )
        )
        row = _fetch_prop(conn, int(result.inserted_primary_key[0]))
        # TS 用的是 success（HTTP 200），不是 created
        return success(row_to_camel(row, "prop_templates") if row is not None else None)
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# PUT /{id} — 更新
# ---------------------------------------------------------------------------

@router.put("/{prop_id}")
async def update_prop(prop_id: str, request: Request, conn: Connection = Depends(get_tx)):
    try:
        pid = parse_prop_id(prop_id)
        if not pid:
            return not_found("Invalid prop id")

        body = await read_json(request)
        values: dict[str, Any] = {"updated_at": now()}
        for key, column in _FIELD_MAP.items():
            # TS 判据是 `body[k] !== undefined` ⇒ 显式传 null 也会写入（不是"仅非空才更新"）
            if key in body:
                values[column] = body[key]

        conn.execute(update(prop_templates).where(prop_templates.c.id == pid).values(**values))
        # 注意：更新后不校验软删状态（与 TS 一致）
        row = conn.execute(
            select(prop_templates).where(prop_templates.c.id == pid)
        ).first()
        return success(row_to_camel(row, "prop_templates") if row is not None else None)
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# DELETE /{id} — 软删，返回 {ok: true}
# ---------------------------------------------------------------------------

@router.delete("/{prop_id}")
def delete_prop(prop_id: str, conn: Connection = Depends(get_tx)):
    try:
        pid = parse_prop_id(prop_id)
        if not pid:
            return not_found("Invalid prop id")
        ts = now()
        conn.execute(
            update(prop_templates)
            .where(prop_templates.c.id == pid)
            .values(deleted_at=ts, updated_at=ts)
        )
        return success({"ok": True})
    except Exception as exc:  # noqa: BLE001
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


# ---------------------------------------------------------------------------
# POST /{prop_id}/generate-image — 生成物品设定图
# ---------------------------------------------------------------------------

@router.post("/{prop_id}/generate-image")
async def generate_prop_image(prop_id: str, request: Request,
                              conn: Connection = Depends(get_tx)):
    """生成物品设定图。

    ⚠️ 三处保真点：① 找不到物品是 **404 ``物品不存在``**（中文文案，本域特例）；
    ② ``category == '屏幕留白'`` 走**留白图**构建器（供后期叠字），否则走物品构建器；
    ③ ``config_id`` 是 falsy 判定 —— ``0``/``''`` 都视为未传。
    """
    pid = parse_prop_id(prop_id)
    if pid is None:
        return not_found("Invalid prop id")
    prop = _fetch_prop(conn, pid)
    if prop is None:
        return not_found("物品不存在")
    body = await read_json(request)
    # 屏幕留白类物品（ui_plate）使用留白图 prompt 构建器，供后期叠加文字
    if body.get("prompt"):
        prompt = body["prompt"]
    elif prop.category == UI_PLATE_CATEGORY:
        prompt = build_ui_plate_image_prompt({
            "type": prop.name, "context": prop.description or prop.appearance,
        })
    else:
        prompt = build_prop_image_prompt(row_to_camel(prop))
    raw_config = body.get("config_id")
    config_id = js_number(raw_config) if raw_config else None

    try:
        log_task_start("PropsAPI", "generate-image", {"propId": pid, "name": prop.name})
        image_id = await generate_image(conn, {
            "propId": pid,
            "dramaId": prop.drama_id,
            "prompt": prompt,
            "negativePrompt": prop.negative_prompt or None,
            "configId": config_id,
        })
        return success({"imageGenerationId": image_id, "prompt": prompt})
    except Exception as exc:  # noqa: BLE001
        log_task_error("PropsAPI", "generate-image", {"propId": pid, "error": str(exc)})
        return bad_request(str(exc) or "生成失败")
