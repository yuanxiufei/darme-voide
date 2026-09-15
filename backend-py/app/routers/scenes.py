"""scenes 域 —— 与 ``backend/src/routes/scenes.ts`` 对齐。

**已整域迁移**：`GET /{id}`、`POST /`、`PUT /{id}`、`DELETE /{id}`、
`POST /{id}/generate-image` —— 后者 2026-09-15 校正：**后来已迁**（``services/image_generation.py``）。
⚠️ 「哪些没迁」以 ``tests/route_parity_test.py`` 的机械扫描为准。

⚠️ 三处容易照抄错的语义（原 TS 就是这样，逐条对齐）：

1. ``GET /{id}`` **不过滤软删**（无 ``deletedAt`` 条件）—— 但 scenes 的删除是**硬删**（见 4），
   所以该条件其实是历史遗留；仍照抄，避免行为漂移。
2. ``POST /`` 的 ``time`` 与 ``prompt`` 有回退：``time || ''``、``prompt || body.location``。
   这里是**逻辑或**（空串会回退），与 script-fingerprint 的 ``??`` 不同。
3. ``PUT`` 在更新后若 ``location`` 变化，会先把 ``location_id`` **置空再重新锁定**
   （六键 Bible 的跨集地点复用依赖这一步），不做就等于地点永远锁在旧 ID 上。

⚠️ 本域返回 **camelCase**（drizzle 行），见 ``response.dict_to_camel``。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import and_, select, update
from sqlalchemy.engine import Connection

from ..core.db import get_conn, get_tx
from ..core.models import episodes, scenes
from ..core.request_utils import read_json
from ..core.response import (
    bad_request,
    camel_to_snake,
    created,
    js_number,
    not_found,
    now,
    parse_param_id,
    row_to_camel,
    success,
)
from ..services.bible_ids import ensure_location_id
from ..services.image_generation import generate_image
from ..services.prompt_utils import (
    build_scene_image_prompt,
    build_scene_negative_prompt,
    resolve_effective_art_style,
)
from ..services.task_logger import log_task_error, log_task_start, log_task_success

router = APIRouter(prefix="/api/v1/scenes", tags=["scenes"])

# PUT /scenes/:id 白名单（对齐 TS，camelCase 原样保留以便双写兼容）
_UPDATE_KEYS = [
    "location", "time", "prompt", "description", "atmosphere",
    "lighting", "weather", "season", "style", "customPrompt", "negativePrompt", "imageUrl",
]


def _fetch_scene(conn: Connection, scene_id: int | float):
    return conn.execute(select(scenes).where(scenes.c.id == scene_id)).first()


@router.get("/{scene_id}")
def get_scene(scene_id: str, conn: Connection = Depends(get_conn)):
    try:
        sid = parse_param_id(scene_id)
        if sid is None:
            return not_found("Invalid scene id")
        row = _fetch_scene(conn, sid)
        if row is None:
            return not_found("Scene not found")
        return success(row_to_camel(row, "scenes"))
    except Exception as exc:  # noqa: BLE001
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


@router.post("")
async def create_scene(request: Request, conn: Connection = Depends(get_tx)):
    try:
        body = await read_json(request)
        ts = now()
        result = conn.execute(
            scenes.insert().values(
                drama_id=body.get("drama_id"),
                episode_id=body.get("episode_id"),
                location=body.get("location"),
                # time || '' 、prompt || location 都是逻辑或（空串会回退）
                time=body.get("time") or "",
                prompt=body.get("prompt") or body.get("location"),
                created_at=ts,
                updated_at=ts,
            )
        )
        scene_id = int(result.inserted_primary_key[0])
        ensure_location_id(conn, scene_id)
        row = _fetch_scene(conn, scene_id)
        # TS 用的是 created（HTTP 201），不是 success
        return created(row_to_camel(row, "scenes") if row is not None else None)
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.put("/{scene_id}")
async def update_scene(scene_id: str, request: Request, conn: Connection = Depends(get_tx)):
    try:
        sid = parse_param_id(scene_id)
        if sid is None:
            return not_found("Invalid scene id")

        body = await read_json(request)
        updates: dict[str, Any] = {"updated_at": now()}
        for key in _UPDATE_KEYS:
            snake = camel_to_snake(key)
            if snake in body:
                updates[snake] = body[snake]
            elif key in body:
                # ⚠️ 值取自 camelCase 键，列名仍须写 snake_case（同 characters）——
                # TS 的 key 是 drizzle 属性名，Python Core 只认 DB 列名
                updates[snake] = body[key]

        conn.execute(update(scenes).where(scenes.c.id == sid).values(**updates))

        # 地点变化 → 先清空 location_id 再重新锁定（六键 Bible 跨集锁定）
        if "location" in updates:
            conn.execute(update(scenes).where(scenes.c.id == sid).values(location_id=None))
        ensure_location_id(conn, int(sid))

        row = _fetch_scene(conn, sid)
        return success(row_to_camel(row, "scenes") if row is not None else None)
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.delete("/{scene_id}")
def delete_scene(scene_id: str, conn: Connection = Depends(get_tx)):
    """硬删（物理删除，不同于 characters 的软删）。"""
    try:
        sid = parse_param_id(scene_id)
        if sid is None:
            return not_found("Invalid scene id")
        conn.execute(scenes.delete().where(scenes.c.id == sid))
        return success()
    except Exception as exc:  # noqa: BLE001
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


# ---------------------------------------------------------------------------
# POST /{scene_id}/generate-image
# ---------------------------------------------------------------------------

def _resolve_drama_config_id(conn: Connection, drama_id: Any, field: str) -> Any:
    """drama 级共享生成配置：取该剧**第一个已配置**的 episode configId。

    ⚠️ 与 ``characters.py`` 里的同名函数**故意各存一份** —— 原 TS 就是在
    ``characters.ts`` 与 ``scenes.ts`` 里各写了一遍，保持一致比抽公共件更不容易漂。
    """
    rows = conn.execute(
        select(episodes).where(and_(episodes.c.drama_id == drama_id,
                                    episodes.c.deleted_at.is_(None)))
    ).all()
    for row in rows:
        value = getattr(row, field)
        if value is not None:
            return value
    return None


@router.post("/{scene_id}/generate-image")
async def generate_scene_image(scene_id: str, request: Request,
                               conn: Connection = Depends(get_tx)):
    """生成场景图。

    ⚠️ 四处保真点：① id 非法 **404**、场景不存在 **400**；② 场景查询**不过滤软删**
    （但参考图查询过滤）；③ 置 ``status='processing'`` → 成功不动状态、**失败置 ``failed``**；
    ④ **跨集一致**：把同剧**同地点**（地点非空！）已出图的**其他**场景当参考图（最多 2 张）——
    地点为空时必须跳过，否则会命中所有「无地点」场景造成串图。
    """
    sid = parse_param_id(scene_id)
    if sid is None:
        return not_found("Invalid scene id")
    body = await read_json(request)
    scene = conn.execute(select(scenes).where(scenes.c.id == sid)).first()
    if scene is None:
        return bad_request("Scene not found")

    episode = None
    if body.get("episode_id"):
        number = js_number(body.get("episode_id"))
        episode = (conn.execute(select(episodes).where(episodes.c.id == number)).first()
                   if number is not None else None)
        if episode is None:
            return bad_request("Episode not found")

    # 画风收口：统一解析链（剧集 style → 全局默认 → realistic），并让正/负提示词**同源**
    drama_style = resolve_effective_art_style(conn, scene.drama_id)
    prompt = (body.get("prompt") or scene.custom_prompt
              or build_scene_image_prompt({
                  "location": scene.location, "time": scene.time,
                  "prompt": scene.prompt, "dramaStyle": drama_style,
              }))

    # 跨集一致：自动注入同剧同地点已生成的场景图作为参考图，让同地点不同时段/不同集的新场景
    # 继承既有空间布局与材质，避免同一地点跨集各出一版视觉漂移
    norm_location = (scene.location or "").strip()
    location_refs: list[str] = []
    if norm_location:
        siblings = conn.execute(
            select(scenes).where(and_(scenes.c.drama_id == scene.drama_id,
                                      scenes.c.deleted_at.is_(None)))
        ).all()
        for sibling in siblings:
            if len(location_refs) >= 2:
                break
            if (sibling.id != scene.id and (sibling.location or "").strip() == norm_location
                    and sibling.image_url):
                location_refs.append(sibling.image_url)

    config_id = episode.image_config_id if episode is not None else None
    if config_id is None:
        config_id = _resolve_drama_config_id(conn, scene.drama_id, "image_config_id")

    try:
        log_task_start("SceneImage", "generate", {
            "sceneId": sid, "episodeId": episode.id if episode is not None else None,
            "dramaId": scene.drama_id, "location": scene.location,
            "model": body.get("model") or "default", "locationRefs": len(location_refs),
        })
        conn.execute(update(scenes).where(scenes.c.id == sid)
                     .values(status="processing", updated_at=now()))
        gen_id = await generate_image(conn, {
            "sceneId": sid,
            "dramaId": scene.drama_id,
            "prompt": prompt,
            "negativePrompt": (body.get("negative_prompt") or scene.negative_prompt
                               or build_scene_negative_prompt(drama_style)),
            "model": body.get("model"),
            "configId": config_id,
            "referenceImages": location_refs or None,
        })
        log_task_success("SceneImage", "generate", {"sceneId": sid, "generationId": gen_id})
        return success({"image_generation_id": gen_id})
    except Exception as exc:  # noqa: BLE001
        log_task_error("SceneImage", "generate", {"sceneId": sid, "error": str(exc)})
        conn.execute(update(scenes).where(scenes.c.id == sid)
                     .values(status="failed", updated_at=now()))
        return bad_request(str(exc))
