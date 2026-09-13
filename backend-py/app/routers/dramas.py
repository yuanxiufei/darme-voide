"""dramas 域 —— 与 ``backend/src/routes/dramas.ts`` 逐端点对齐。

**已迁移 9 个端点**

======  ==========================  ==================================
方法    路径                         说明
======  ==========================  ==================================
GET     ``/``                       分页列表（含 progress 聚合）
POST    ``/``                       创建（自动建 1 集 episode）
GET     ``/stats``                  按状态统计（必须声明在 ``/{id}`` 之前）
GET     ``/{id}``                   详情（嵌套 episodes/characters/scenes）
PUT     ``/{id}``                   更新（含 era_background 规范化）
DELETE  ``/{id}``                   软删（管线执行中拒绝）
PUT     ``/{id}/characters``        批量保存角色（白名单字段）
PUT     ``/{id}/episodes``          批量保存剧集（白名单字段）
GET     ``/{id}/prompts``           聚合全剧提示词
======  ==========================  ==================================

**刻意未迁移 2 个端点**（依赖尚未移植的服务层，不注册，交给 ``main.py`` 的兜底路由）：

* ``GET  /{id}/rhythm``                   依赖 ``services/rhythm-phase.ts``
* ``POST /{id}/era-background/extract``   依赖 ``services/text-generation.ts``（LLM 链路）

这两个路径在 ``PROXY_TO_NODE=1`` 时由 Node 后端继续服务 —— 这正是绞杀者模式
「逐域切换、边界可见、随时可回退」的落地方式，比一次性重写安全得多。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import and_, desc, select, update
from sqlalchemy.engine import Connection

from ..db import get_conn, get_tx
from ..models import app_settings, characters, dramas, episodes, scenes, storyboards
from ..request_utils import read_json
from ..response import (
    bad_request,
    conflict,
    created,
    not_found,
    now,
    parse_json_array,
    parse_param_id,
    pick_fields,
    rows_to_dicts,
    row_to_dict,
    success,
)
from ..services.bible_ids import ensure_costume_id, ensure_style_id
from ..services.era_background import era_background_to_json, parse_era_background

router = APIRouter(prefix="/api/v1/dramas", tags=["dramas"])

# PUT /dramas/:id/characters 的白名单（对齐 TS 的 CHAR_KEYS，snake_case 形式）
_CHAR_KEYS = [
    "name", "role", "role_type", "description", "appearance", "personality",
    "voice_style", "image_url", "reference_images", "seed_value", "sort_order",
    "local_path", "voice_sample_url", "voice_provider", "voice_speed", "voice_emotion",
    "voice_pitch", "clothing", "weapons", "custom_prompt", "voice_model", "costume_id",
]

# PUT /dramas/:id/episodes 的白名单（对齐 TS 的 EP_KEYS）
_EP_KEYS = [
    "episode_number", "title", "content", "script_content", "description",
    "duration", "status", "video_url", "thumbnail",
    "image_config_id", "video_config_id", "audio_config_id",
]


def _global_art_style(conn: Connection) -> str | None:
    """读取全局默认画风（``app_settings.art_style``）。

    注：画风**合法性**收口在 ``shared/prompt-utils.ts`` 的 ``resolveEffectiveArtStyle``，
    生成链路上才做「脏值跳过」。CRUD 层与原 TS 的 ``getGlobalArtStyle`` 一样只做读取透传。
    """
    row = conn.execute(
        select(app_settings.c.value).where(app_settings.c.key == "art_style")
    ).first()
    return str(row[0]) if row and row[0] else None


def _progress(ep_rows: list[dict], sb_rows: list[dict]) -> dict[str, int]:
    storyboarded = len({s["episode_id"] for s in sb_rows})
    return {
        "total_episodes": len(ep_rows),
        "scripted_episodes": len([e for e in ep_rows if e.get("script_content")]),
        "storyboarded_episodes": storyboarded,
        "storyboards": len(sb_rows),
        "images": len([s for s in sb_rows if s.get("composed_image") or s.get("first_frame_image")]),
        "videos": len([s for s in sb_rows if s.get("video_url") or s.get("composed_video_url")]),
        "tts": len([s for s in sb_rows if s.get("tts_audio_url")]),
    }


def _episodes_of(conn: Connection, drama_id: int) -> list[dict]:
    return rows_to_dicts(
        conn.execute(
            select(episodes).where(
                and_(episodes.c.drama_id == drama_id, episodes.c.deleted_at.is_(None))
            )
        )
    )


def _characters_of(conn: Connection, drama_id: int) -> list[dict]:
    return rows_to_dicts(
        conn.execute(
            select(characters).where(
                and_(characters.c.drama_id == drama_id, characters.c.deleted_at.is_(None))
            )
        )
    )


def _scenes_of(conn: Connection, drama_id: int) -> list[dict]:
    return rows_to_dicts(
        conn.execute(
            select(scenes).where(
                and_(scenes.c.drama_id == drama_id, scenes.c.deleted_at.is_(None))
            )
        )
    )


def _storyboards_of(conn: Connection, episode_ids: list[int]) -> list[dict]:
    if not episode_ids:
        return []
    return rows_to_dicts(
        conn.execute(
            select(storyboards).where(
                and_(
                    storyboards.c.episode_id.in_(episode_ids),
                    storyboards.c.deleted_at.is_(None),
                )
            )
        )
    )


# ---------------------------------------------------------------------------
# GET / — 分页列表
# ---------------------------------------------------------------------------

@router.get("")
def list_dramas(
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    keyword: str | None = None,
    conn: Connection = Depends(get_conn),
):
    try:
        rows = rows_to_dicts(
            conn.execute(
                select(dramas)
                .where(dramas.c.deleted_at.is_(None))
                .order_by(desc(dramas.c.updated_at))
            )
        )
        # 过滤在内存里做（与 TS 版一致：status 精确匹配、keyword 走 title 包含）
        if status:
            rows = [d for d in rows if d.get("status") == status]
        if keyword:
            rows = [d for d in rows if keyword in (d.get("title") or "")]

        total = len(rows)
        page = page if page >= 1 else 1
        page_size = page_size if page_size >= 1 else 20
        sliced = rows[(page - 1) * page_size : page * page_size]

        items: list[dict[str, Any]] = []
        for drama in sliced:
            eps = _episodes_of(conn, drama["id"])
            chars = _characters_of(conn, drama["id"])
            scns = _scenes_of(conn, drama["id"])
            sbs = _storyboards_of(conn, [e["id"] for e in eps])

            item = dict(drama)
            item["tags"] = parse_json_array(drama.get("tags"))
            item["total_episodes"] = len(eps)  # 覆盖 DB 里的计划集数（TS 版同样覆盖）
            item["episodes"] = eps
            item["characters"] = chars
            item["scenes"] = scns
            item["progress"] = _progress(eps, sbs)
            items.append(item)

        return success(
            {
                "items": items,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": -(-total // page_size),  # 向上取整，等价 Math.ceil
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# POST / — 创建（自动建 episode）
# ---------------------------------------------------------------------------

@router.post("")
async def create_drama(request: Request, conn: Connection = Depends(get_tx)):
    try:
        body = await read_json(request)
        ts = now()

        values: dict[str, Any] = {
            "title": body.get("title"),
            "description": body.get("description"),
            "genre": body.get("genre"),
            # ⚠️ 紧凑分隔符（Node 是 `JSON.stringify(body.tags)`）
            "tags": json.dumps(body["tags"], ensure_ascii=False, separators=(",", ":"))
            if body.get("tags")
            else None,
            "metadata": body.get("metadata"),
            "status": "draft",
            "created_at": ts,
            "updated_at": ts,
        }
        # 创建时未指定画风则用全局默认；两者都为空时**不写该列**，让 DB 默认值 realistic 生效
        # （对齐 TS 的 `body.style || getGlobalArtStyle() || undefined`）
        style = body.get("style") or _global_art_style(conn)
        if style:
            values["style"] = style

        result = conn.execute(dramas.insert().values(**values))
        drama_id = int(result.inserted_primary_key[0])

        ensure_style_id(conn, drama_id)

        row = conn.execute(select(dramas).where(dramas.c.id == drama_id)).first()
        if row is None:
            return bad_request("create drama failed")

        # 默认剧集（对齐 TS：标题「第N集」、状态 draft）
        total_episodes = body.get("total_episodes") or 1
        for i in range(1, int(total_episodes) + 1):
            conn.execute(
                episodes.insert().values(
                    drama_id=drama_id,
                    episode_number=i,
                    title=f"第{i}集",
                    status="draft",
                    created_at=ts,
                    updated_at=ts,
                )
            )

        return created(row_to_dict(row))
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# GET /stats — 必须声明在 /{id} 之前
# ---------------------------------------------------------------------------

@router.get("/stats")
def drama_stats(conn: Connection = Depends(get_conn)):
    try:
        rows = rows_to_dicts(
            conn.execute(select(dramas.c.status).where(dramas.c.deleted_at.is_(None)))
        )
        counts: dict[str, int] = {}
        for r in rows:
            key = r.get("status") or "draft"
            counts[key] = counts.get(key, 0) + 1
        return success(
            {"total": len(rows), "by_status": [{"status": k, "count": v} for k, v in counts.items()]}
        )
    except Exception as exc:  # noqa: BLE001
        # TS 版此处用的是 {code:500, data:null, message}，故不能走 bad_request
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


# ---------------------------------------------------------------------------
# GET /{id} — 详情
# ---------------------------------------------------------------------------

@router.get("/{drama_id}")
def get_drama(drama_id: str, conn: Connection = Depends(get_conn)):
    try:
        did = parse_param_id(drama_id)
        if did is None:
            return not_found("Invalid drama id")

        row = conn.execute(
            select(dramas).where(and_(dramas.c.id == did, dramas.c.deleted_at.is_(None)))
        ).first()
        if row is None:
            return not_found("剧本不存在")

        drama = row_to_dict(row)
        drama["tags"] = parse_json_array(drama.get("tags"))
        drama["episodes"] = _episodes_of(conn, did)
        drama["characters"] = _characters_of(conn, did)
        drama["scenes"] = _scenes_of(conn, did)
        return success(drama)
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# PUT /{id} — 更新
# ---------------------------------------------------------------------------

@router.put("/{drama_id}")
async def update_drama(drama_id: str, request: Request, conn: Connection = Depends(get_tx)):
    try:
        did = parse_param_id(drama_id)
        if did is None:
            return not_found("Invalid drama id")

        exists = conn.execute(
            select(dramas.c.id).where(and_(dramas.c.id == did, dramas.c.deleted_at.is_(None)))
        ).first()
        if exists is None:
            return not_found("Drama not found")

        body = await read_json(request)
        updates: dict[str, Any] = {"updated_at": now()}

        for key in ("title", "description", "genre", "style", "status"):
            if key in body:
                updates[key] = body[key]
        if "tags" in body:
            updates["tags"] = json.dumps(body["tags"], ensure_ascii=False, separators=(",", ":"))
        if "metadata" in body:
            updates["metadata"] = body["metadata"]

        # 时代背景：前端 AI 提炼后回填 / 用户手工编辑。空串 = 清空；非法结构 400 拒绝落库
        era_raw = body["era_background"] if "era_background" in body else body.get("eraBackground")
        if era_raw is not None or "era_background" in body or "eraBackground" in body:
            if era_raw is None or era_raw == "":
                updates["era_background"] = ""
            elif isinstance(era_raw, (str, dict)):
                norm = parse_era_background(
                    era_raw
            if isinstance(era_raw, str)
            else json.dumps(era_raw, ensure_ascii=False, separators=(",", ":"))
                )
                if norm is None:
                    return bad_request(
                        "era_background 不是有效的时代背景 JSON（需 era/summary/imageHint 字段）"
                    )
                updates["era_background"] = era_background_to_json(norm)
            else:
                return bad_request("era_background 类型不合法")

        conn.execute(update(dramas).where(dramas.c.id == did).values(**updates))
        return success()
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# DELETE /{id} — 软删（管线执行中拒绝）
# ---------------------------------------------------------------------------

@router.delete("/{drama_id}")
def delete_drama(drama_id: str, conn: Connection = Depends(get_tx)):
    try:
        did = parse_param_id(drama_id)
        if did is None:
            return not_found("Invalid drama id")

        exists = conn.execute(
            select(dramas.c.id).where(and_(dramas.c.id == did, dramas.c.deleted_at.is_(None)))
        ).first()
        if exists is None:
            return not_found("Drama not found")

        # 删除竞态保护：管线执行中的 episode 拒绝删除（避免已删剧本继续写数据/复活）
        ep_rows = _episodes_of(conn, did)
        in_flight = [
            e
            for e in ep_rows
            if (e.get("status") or "").startswith("auto:")
            and e.get("status") not in ("auto:done", "auto:failed")
        ]
        if in_flight:
            return conflict(
                f"剧本正在自动生成中（{len(in_flight)} 集执行中），请等待完成后再删除"
            )

        conn.execute(update(dramas).where(dramas.c.id == did).values(deleted_at=now()))
        return success()
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# PUT /{id}/characters、PUT /{id}/episodes — 批量白名单保存
# ---------------------------------------------------------------------------

@router.put("/{drama_id}/characters")
async def save_drama_characters(drama_id: str, request: Request, conn: Connection = Depends(get_tx)):
    try:
        did = parse_param_id(drama_id)
        if did is None:
            return not_found("Invalid drama id")

        body = await read_json(request)
        ts = now()
        for char in body.get("characters") or []:
            if not isinstance(char, dict):
                continue
            fields = pick_fields(char, _CHAR_KEYS)
            char_id = char.get("id")
            if char_id:
                values = {**fields, "updated_at": ts}
                conn.execute(
                    update(characters)
                    .where(and_(characters.c.id == char_id, characters.c.deleted_at.is_(None)))
                    .values(**values)
                )
                ensure_costume_id(conn, int(char_id))
            else:
                # 同上：先合并再注入主键/时间戳，避免与白名单字段同名时 TypeError
                values = {**fields, "drama_id": did, "created_at": ts, "updated_at": ts}
                result = conn.execute(characters.insert().values(**values))
                ensure_costume_id(conn, int(result.inserted_primary_key[0]))
        return success()
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


@router.put("/{drama_id}/episodes")
async def save_drama_episodes(drama_id: str, request: Request, conn: Connection = Depends(get_tx)):
    try:
        did = parse_param_id(drama_id)
        if did is None:
            return not_found("Invalid drama id")

        body = await read_json(request)
        ts = now()
        for ep in body.get("episodes") or []:
            if not isinstance(ep, dict):
                continue
            fields = pick_fields(ep, _EP_KEYS)
            ep_id = ep.get("id")
            if ep_id:
                # 注意：TS 版此处不过滤软删（update 无 deleted_at 条件），保持一致
                values = {**fields, "updated_at": ts}
                conn.execute(update(episodes).where(episodes.c.id == ep_id).values(**values))
            else:
                # ⚠️ 必须「先合并 dict 再覆盖」，不能写 values(**fields, title=...)：
                # JS 的对象展开是「后者覆盖前者」，而 Python 的 ** 遇到同名键会直接
                # TypeError（got multiple values for keyword argument），行为并不等价。
                values = {**fields, "drama_id": did, "created_at": ts, "updated_at": ts}
                values["episode_number"] = values.get("episode_number") or 1
                values["title"] = values.get("title") or "未命名"
                conn.execute(episodes.insert().values(**values))
        return success()
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# GET /{id}/prompts — 聚合全剧提示词
# ---------------------------------------------------------------------------

@router.get("/{drama_id}/prompts")
def get_drama_prompts(drama_id: str, conn: Connection = Depends(get_conn)):
    did = parse_param_id(drama_id)
    if did is None:
        return not_found("Invalid drama id")
    try:
        exists = conn.execute(
            select(dramas.c.id).where(and_(dramas.c.id == did, dramas.c.deleted_at.is_(None)))
        ).first()
        if exists is None:
            return not_found("Drama not found")

        char_rows = rows_to_dicts(
            conn.execute(
                select(characters)
                .where(and_(characters.c.drama_id == did, characters.c.deleted_at.is_(None)))
                .order_by(characters.c.sort_order)
            )
        )
        scene_rows = rows_to_dicts(
            conn.execute(
                select(scenes)
                .where(and_(scenes.c.drama_id == did, scenes.c.deleted_at.is_(None)))
                .order_by(scenes.c.id)
            )
        )
        ep_rows = rows_to_dicts(
            conn.execute(
                select(episodes)
                .where(and_(episodes.c.drama_id == did, episodes.c.deleted_at.is_(None)))
                .order_by(episodes.c.episode_number)
            )
        )
        ep_ids = [e["id"] for e in ep_rows]
        sb_rows: list[dict] = []
        if ep_ids:
            sb_rows = rows_to_dicts(
                conn.execute(
                    select(storyboards)
                    .where(storyboards.c.episode_id.in_(ep_ids))
                    .order_by(storyboards.c.episode_id, storyboards.c.storyboard_number)
                )
            )

        return success(
            {
                "characters": [
                    {
                        "id": ch["id"], "name": ch["name"], "role": ch.get("role"),
                        "customPrompt": ch.get("custom_prompt"), "imageUrl": ch.get("image_url"),
                    }
                    for ch in char_rows
                ],
                "scenes": [
                    {
                        "id": sc["id"], "location": sc.get("location"), "time": sc.get("time"),
                        "prompt": sc.get("prompt"), "customPrompt": sc.get("custom_prompt"),
                        "imageUrl": sc.get("image_url"),
                    }
                    for sc in scene_rows
                ],
                "episodes": [
                    {"id": e["id"], "episodeNumber": e["episode_number"], "title": e.get("title")}
                    for e in ep_rows
                ],
                "storyboards": [
                    {
                        "id": sb["id"], "episodeId": sb.get("episode_id"),
                        "storyboardNumber": sb.get("storyboard_number"), "title": sb.get("title"),
                        "imagePrompt": sb.get("image_prompt"), "videoPrompt": sb.get("video_prompt"),
                        "customImagePrompt": sb.get("custom_image_prompt"),
                        "customVideoPrompt": sb.get("custom_video_prompt"),
                        "status": sb.get("status"),
                    }
                    for sb in sb_rows
                ],
            }
        )
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))
