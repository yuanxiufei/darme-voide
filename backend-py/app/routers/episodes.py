"""episodes 域 —— 与 ``backend/src/routes/episodes.ts`` 逐端点对齐。

**已迁移 8 个端点**

======  ==================================  ==================================
方法    路径                                 说明
======  ==================================  ==================================
POST    ``/``                                新建剧集（要求三个 config_id）
PUT     ``/{id}``                            更新（白名单 + 剧本指纹重算）
DELETE  ``/{id}``                            软删
GET     ``/{id}/characters``                 该集关联角色
GET     ``/{id}/scenes``                     该集关联场景
GET     ``/{episode_id}/storyboards``        该集分镜（含角色/服装/道具关联）
GET     ``/{id}/pipeline-status``            流水线十步进度
GET     ``/{id}/script-fingerprint``         剧本指纹门禁状态
======  ==================================  ==================================

**刻意未迁移 2 个端点**（不注册 → 走反代/501）：

* ``POST /{id}/continue-script``   依赖 ``services/text-generation.ts`` 的 LLM 链路
* ``POST /{id}/consistency-qc``    依赖 ``services/consistency-qc.ts``（视觉模型 + 多图输入）

⚠️ ``GET /{id}/pipeline-status`` 与 dramas 列表里的同名概念**判据不同**，不要合并：
前者用 ``composed_image`` 单字段，后者（`GET /dramas` 的 progress）用
``composed_image || first_frame_image``。原 TS 就是两套，照抄。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import and_, select, update
from sqlalchemy.engine import Connection

from ..db import get_conn, get_tx
from ..models import (
    characters,
    episode_characters,
    episode_scenes,
    episodes,
    scenes,
    storyboard_characters,
    storyboard_props,
    storyboards,
    video_merges,
)
from ..request_utils import read_json
from ..response import (
    bad_request,
    not_found,
    now,
    parse_param_id,
    success,
)
from ..services.script_fingerprint import (
    check_episode_fingerprint,
    refresh_episode_script_hash,
)

router = APIRouter(prefix="/api/v1/episodes", tags=["episodes"])

# PUT /episodes/:id 的白名单（对齐 TS 的 allowed，请求体侧一律 snake_case）
_ALLOWED_UPDATE_KEYS = [
    "content", "script_content", "title", "description", "status",
    "image_config_id", "video_config_id", "audio_config_id",
    "bgm_url", "bgm_volume", "bgm_fade_in", "bgm_fade_out",
]

# 需要做 Number() 数值转换的键（对齐 TS 里逐个 Number(...) 的写法）
_NUMERIC_KEYS = {
    "image_config_id", "video_config_id", "audio_config_id",
    "bgm_volume", "bgm_fade_in", "bgm_fade_out",
}


def _js_number(value: Any) -> Any:
    """模拟 JS 的 ``Number(x)``，用于 6 个数值字段。

    已知的唯一有意偏差：JS 对 ``"abc"`` 得到 ``NaN``，而 better-sqlite3 绑定 ``NaN``
    会写进 REAL 列成为 NaN 值；Python 侧返回 ``None``（写 NULL）。
    **比写坏数据更安全**，且这些字段本身就允许为 NULL，故不影响前端契约。
    """
    if value is None:
        return 0                     # Number(null) === 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return 0                 # Number('') === 0
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return None              # JS 此处为 NaN
    return None


def _fetch_episode(conn: Connection, episode_id: int):
    return conn.execute(select(episodes).where(episodes.c.id == episode_id)).first()


# ---------------------------------------------------------------------------
# POST / — 新建剧集
# ---------------------------------------------------------------------------

@router.post("")
async def create_episode(request: Request, conn: Connection = Depends(get_tx)):
    try:
        body = await read_json(request)
        if not body.get("drama_id"):
            return bad_request("drama_id required")
        if not body.get("image_config_id") or not body.get("video_config_id") or not body.get("audio_config_id"):
            return bad_request("image_config_id, video_config_id and audio_config_id are required")

        ts = now()
        existing = conn.execute(
            select(episodes.c.episode_number).where(
                and_(episodes.c.drama_id == body["drama_id"], episodes.c.deleted_at.is_(None))
            )
        ).all()
        # 下一个集号 = 现有最大集号 + 1（排除软删）
        next_num = max((r.episode_number for r in existing), default=0) + 1 if existing else 1

        result = conn.execute(
            episodes.insert().values(
                drama_id=body["drama_id"],
                episode_number=next_num,
                title=body.get("title") or f"第{next_num}集",
                image_config_id=body["image_config_id"],
                video_config_id=body["video_config_id"],
                audio_config_id=body["audio_config_id"],
                created_at=ts,
                updated_at=ts,
            )
        )
        ep = _fetch_episode(conn, int(result.inserted_primary_key[0]))
        if ep is None:
            return bad_request("create episode failed")

        # 注意：原 TS 用的是 success（HTTP 200）不是 created（201）
        return success(
            {
                "id": ep.id,
                "episode_number": ep.episode_number,
                "title": ep.title,
                "image_config_id": ep.image_config_id,
                "video_config_id": ep.video_config_id,
                "audio_config_id": ep.audio_config_id,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# PUT /{id} — 更新（含剧本指纹重算）
# ---------------------------------------------------------------------------

@router.put("/{episode_id}")
async def update_episode(episode_id: str, request: Request, conn: Connection = Depends(get_tx)):
    try:
        eid = parse_param_id(episode_id)
        if eid is None:
            return not_found("Invalid episode id")

        body = await read_json(request)
        filtered: dict[str, Any] = {k: body[k] for k in _ALLOWED_UPDATE_KEYS if k in body}
        if not filtered:
            return bad_request("no valid fields")

        values: dict[str, Any] = {"updated_at": now()}
        for key, value in filtered.items():
            values[key] = _js_number(value) if key in _NUMERIC_KEYS else value

        conn.execute(update(episodes).where(episodes.c.id == eid).values(**values))

        # 剧本内容指纹门禁：剧本变更后立即重算，下游分镜/资产据此标记过期
        # （放在更新之后，与 TS 一致 —— 因此 updated_at 会再被刷新一次）
        if "script_content" in filtered or "content" in filtered:
            refresh_episode_script_hash(conn, eid)

        return success()
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# DELETE /{id} — 软删
# ---------------------------------------------------------------------------

@router.delete("/{episode_id}")
def delete_episode(episode_id: str, conn: Connection = Depends(get_tx)):
    try:
        eid = parse_param_id(episode_id)
        if eid is None:
            return not_found("Invalid episode id")

        ep = _fetch_episode(conn, eid)
        if ep is None or ep.deleted_at:
            return not_found("Episode not found")

        ts = now()
        conn.execute(
            update(episodes).where(episodes.c.id == eid).values(deleted_at=ts, updated_at=ts)
        )
        return success()
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# GET /{id}/characters、GET /{id}/scenes — 关联集合
# ---------------------------------------------------------------------------

def _linked_characters(conn: Connection, episode_id: int) -> list[dict[str, Any]]:
    links = conn.execute(
        select(episode_characters.c.character_id).where(
            episode_characters.c.episode_id == episode_id
        )
    ).all()
    char_ids = [r.character_id for r in links]
    if not char_ids:
        return []
    # 与 TS 一致：取全表再按 id 过滤 + 排除软删（不做 SQL IN，保持结果集与顺序完全一致）
    rows = conn.execute(select(characters)).all()
    return [dict(r._mapping) for r in rows if r.id in char_ids and not r.deleted_at]


def _linked_scenes(conn: Connection, episode_id: int) -> list[dict[str, Any]]:
    links = conn.execute(
        select(episode_scenes.c.scene_id).where(episode_scenes.c.episode_id == episode_id)
    ).all()
    scene_ids = [r.scene_id for r in links]
    if not scene_ids:
        return []
    rows = conn.execute(select(scenes)).all()
    return [dict(r._mapping) for r in rows if r.id in scene_ids and not r.deleted_at]


@router.get("/{episode_id}/characters")
def get_episode_characters(episode_id: str, conn: Connection = Depends(get_conn)):
    try:
        eid = parse_param_id(episode_id)
        if eid is None:
            return not_found("Invalid episode id")
        return success(_linked_characters(conn, eid))
    except Exception as exc:  # noqa: BLE001
        # TS 此处是 {code:500, data:null, message}
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


@router.get("/{episode_id}/scenes")
def get_episode_scenes(episode_id: str, conn: Connection = Depends(get_conn)):
    try:
        eid = parse_param_id(episode_id)
        if eid is None:
            return not_found("Invalid episode id")
        return success(_linked_scenes(conn, eid))
    except Exception as exc:  # noqa: BLE001
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


# ---------------------------------------------------------------------------
# GET /{episode_id}/storyboards — 分镜 + 关联聚合
# ---------------------------------------------------------------------------

@router.get("/{episode_id}/storyboards")
def get_episode_storyboards(episode_id: str, conn: Connection = Depends(get_conn)):
    try:
        eid = parse_param_id(episode_id)
        if eid is None:
            return not_found("Invalid episode id")

        rows = conn.execute(
            select(storyboards)
            .where(storyboards.c.episode_id == eid)
            .order_by(storyboards.c.storyboard_number)
        ).all()

        # ⚠️ 与 TS 一致：关联表是**全表读取**再按 storyboard_id 建索引（不是按集过滤）。
        # 表小、且这样能保证「一个分镜的关联」不因集过滤条件写错而丢失；照抄不改。
        char_ids_by_sb: dict[int, list[int]] = {}
        costumes_by_sb: dict[int, dict[int, str]] = {}
        for link in conn.execute(select(storyboard_characters)).all():
            char_ids_by_sb.setdefault(link.storyboard_id, []).append(link.character_id)
            if link.costume:
                costumes_by_sb.setdefault(link.storyboard_id, {})[link.character_id] = link.costume

        prop_ids_by_sb: dict[int, list[int]] = {}
        for plink in conn.execute(select(storyboard_props)).all():
            prop_ids_by_sb.setdefault(plink.storyboard_id, []).append(plink.prop_id)

        episode_char_ids = [
            r.character_id
            for r in conn.execute(
                select(episode_characters.c.character_id).where(
                    episode_characters.c.episode_id == eid
                )
            ).all()
        ]
        all_chars = [
            dict(r._mapping)
            for r in conn.execute(select(characters)).all()
            if r.id in episode_char_ids and not r.deleted_at
        ]

        out: list[dict[str, Any]] = []
        for row in rows:
            mapped = dict(row._mapping)
            linked = char_ids_by_sb.get(row.id, [])
            mapped["character_ids"] = linked
            mapped["character_costumes"] = costumes_by_sb.get(row.id, {})
            mapped["prop_ids"] = prop_ids_by_sb.get(row.id, [])
            mapped["characters"] = [c for c in all_chars if c["id"] in linked]
            out.append(mapped)
        return success(out)
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# GET /{id}/pipeline-status — 流水线十步进度
# ---------------------------------------------------------------------------

def _step_status(done: bool, partial: bool = False) -> str:
    if done:
        return "done"
    return "partial" if partial else "pending"


@router.get("/{episode_id}/pipeline-status")
def get_pipeline_status(episode_id: str, conn: Connection = Depends(get_conn)):
    try:
        eid = parse_param_id(episode_id)
        if eid is None:
            return not_found("Invalid episode id")

        ep = _fetch_episode(conn, eid)
        if ep is None:
            return not_found("Episode not found")

        # 注意各处过滤条件与原 TS 一致：角色排软删；场景/分镜/合成只按 id 过滤
        chars = [
            dict(r._mapping)
            for r in conn.execute(
                select(characters).where(
                    and_(characters.c.drama_id == ep.drama_id, characters.c.deleted_at.is_(None))
                )
            ).all()
        ]
        scns = [
            dict(r._mapping)
            for r in conn.execute(select(scenes).where(scenes.c.drama_id == ep.drama_id)).all()
        ]
        sbs = [
            dict(r._mapping)
            for r in conn.execute(
                select(storyboards).where(storyboards.c.episode_id == eid)
            ).all()
        ]
        merges = [
            dict(r._mapping)
            for r in conn.execute(
                select(video_merges).where(video_merges.c.episode_id == eid)
            ).all()
        ]

        with_voice = [c for c in chars if c.get("voice_style")]
        with_sample = [c for c in chars if c.get("voice_sample_url")]
        sb_with_image = [s for s in sbs if s.get("composed_image")]
        sb_with_video = [s for s in sbs if s.get("video_url")]
        sb_composed = [s for s in sbs if s.get("composed_video_url")]
        latest_merge = merges[-1] if merges else None

        return success(
            {
                "episode_id": eid,
                "steps": {
                    "script_rewrite": {
                        "status": "done" if ep.script_content else ("ready" if ep.content else "pending")
                    },
                    "extract_characters": {
                        "status": _step_status(len(chars) > 0),
                        "count": len(chars),
                    },
                    "extract_scenes": {
                        "status": _step_status(len(scns) > 0),
                        "count": len(scns),
                    },
                    "assign_voices": {
                        "status": _step_status(
                            len(with_voice) == len(chars) and len(chars) > 0, len(with_voice) > 0
                        ),
                        "assigned": len(with_voice),
                        "total": len(chars),
                    },
                    "generate_voice_samples": {
                        "status": _step_status(
                            len(with_sample) == len(with_voice) and len(with_voice) > 0,
                            len(with_sample) > 0,
                        ),
                        "completed": len(with_sample),
                        "total": len(with_voice),
                    },
                    "extract_storyboards": {
                        "status": _step_status(len(sbs) > 0),
                        "count": len(sbs),
                    },
                    "generate_images": {
                        "status": _step_status(
                            len(sb_with_image) == len(sbs) and len(sbs) > 0, len(sb_with_image) > 0
                        ),
                        "completed": len(sb_with_image),
                        "total": len(sbs),
                    },
                    "generate_videos": {
                        "status": _step_status(
                            len(sb_with_video) == len(sbs) and len(sbs) > 0, len(sb_with_video) > 0
                        ),
                        "completed": len(sb_with_video),
                        "total": len(sbs),
                    },
                    "compose_shots": {
                        "status": _step_status(
                            len(sb_composed) == len(sbs) and len(sbs) > 0, len(sb_composed) > 0
                        ),
                        "completed": len(sb_composed),
                        "total": len(sbs),
                    },
                    "merge_episode": {
                        "status": (
                            "done"
                            if latest_merge and latest_merge.get("status") == "completed"
                            else (latest_merge.get("status") if latest_merge else "pending")
                        ),
                        "merged_url": latest_merge.get("merged_url") if latest_merge else None,
                    },
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# GET /{id}/script-fingerprint — 指纹门禁状态
# ---------------------------------------------------------------------------

@router.get("/{episode_id}/script-fingerprint")
def get_script_fingerprint(episode_id: str, conn: Connection = Depends(get_conn)):
    try:
        eid = parse_param_id(episode_id)
        if eid is None:
            return not_found("Invalid episode id")
        return success(check_episode_fingerprint(conn, eid))
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))
