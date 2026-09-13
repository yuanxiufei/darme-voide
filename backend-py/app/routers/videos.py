"""videos 域 —— 与 ``backend/src/routes/videos.ts``（290 行）对齐。

**6 个端点全部迁移**：`POST /`（提交生成）、`GET /{id}`、`GET /`、`PUT /{id}`、
`POST /{id}/regenerate`、`DELETE /{id}`。

``POST /`` 是整条迁移里最绕的一段，它做了四件事：

1. **上下文富化**（可 ``_skip_enrich`` 跳过）：按分镜注入角色外观 + 场景描述 + 参考图 +
   参考音频，并解析剧集画风（正/负面词共用）；
2. **帧来源统一**：body 显式传帧优先 → 分镜已存帧兜底。**这一步是修 bug 的**：此前
   决策读 DB 帧、生成请求只读 body 帧，两者不同源，会出现「决策判定 first_last/single，
   实发请求却一帧都不带」的坏请求（adapter 的单帧分支只认 ``image_url``）；
3. **逐镜路由决策**（``decide_shot_route``，会回写 ``storyboards.route``）；
4. **referenceMode 降级**：目标模式是 ``first_last``（FL2VA）但首尾帧不齐时降级 ``single``，
   避免产生缺首帧的坏请求。

⚠️ 三处容易"顺手改掉"的保真点：

* ``GET /{id}`` 查不到时返回 ``success(c, null)``，**不是 404**；
* ``PUT /{id}`` 的字段白名单里没有 ``updatedAt``（原 TS 就不更新它）；
* 路由决策的对话判据在**本文件**是 ``dialogue|meeting|argument|conversation|multi``
  （**没有**「多人/中文」），比 ``shot_router.DIALOGUE_PATTERN`` **更窄** —— 两处都要保真。

⚠️ 本域返回 **camelCase**（drizzle 行），见 ``response.row_to_camel``。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import delete as sql_delete
from sqlalchemy import select, update
from sqlalchemy.engine import Connection

from ..db import get_conn, get_tx
from ..models import episodes, storyboards, video_generations
from ..request_utils import read_json
from ..response import (
    bad_request,
    created,
    js_number,
    js_truthy,
    not_found,
    parse_param_id,
    row_to_camel,
    success,
)
from ..services.ai_providers import get_active_config, get_config_by_id
from ..services.prompt_utils import (
    build_storyboard_video_prompt,
    build_video_negative_prompt,
    get_storyboard_character_appearances,
    get_storyboard_reference_audio_urls,
    get_storyboard_reference_images,
    get_storyboard_scene_description,
    resolve_effective_art_style,
)
from ..services.shot_router import VIDEO_ROUTE_DIALOGUE_PATTERN, decide_shot_route
from ..services.task_logger import log_task_error, log_task_payload, log_task_start, log_task_success
from ..services.video_generation import generate_video

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])

#: `PUT /{id}` 白名单：snake_case（API 入参） → 蛇形列名。
#: ⚠️ 原 TS 的 map 值是 drizzle 的 **camelCase 属性名**；这里直接给列名，等价。
_UPDATE_FIELD_MAP: dict[str, str] = {
    "prompt": "prompt",
    "negative_prompt": "negative_prompt",
    "model": "model",
    "duration": "duration",
    "reference_mode": "reference_mode",
    "aspect_ratio": "aspect_ratio",
    "character_ids": "character_ids",
    "image_url": "image_url",
    "first_frame_url": "first_frame_url",
    "last_frame_url": "last_frame_url",
}

#: 支持多参考图的视频厂商（`canMultiRef` 判据之一）
_MULTI_REF_PROVIDERS = ("volcengine", "vidu", "minimax")


def _fetch_video(conn: Connection, video_id: int | float):
    return conn.execute(
        select(video_generations).where(video_generations.c.id == video_id)
    ).first()


@router.post("")
async def create_video(request: Request, conn: Connection = Depends(get_tx)):
    """提交视频生成（含上下文富化 + 逐镜路由决策）。"""
    try:
        body = await read_json(request)
        if not js_truthy(body.get("prompt")):
            return bad_request("prompt is required")

        config_id: Any = body.get("config_id")
        prompt = body.get("prompt")
        first_frame_url = body.get("first_frame_url")
        reference_image_urls = body.get("reference_image_urls")
        scene_type = body.get("scene_type")
        reference_audio_urls = body.get("reference_audio_urls")
        # 分镜行：上下文注入与逐镜路由共用同一次查询，保证「决策输入」与「请求输入」同源
        sb_row = None
        # 画风收口：视频统一解析链（剧集 → 全局 → realistic），供提示词与负面词共用
        drama_style: str | None = None

        if js_truthy(body.get("storyboard_id")):
            sb_id = js_number(body.get("storyboard_id"))
            if sb_id is not None:
                sb_row = conn.execute(
                    select(storyboards).where(storyboards.c.id == sb_id)
                ).first()
            if sb_row is not None:
                episode = conn.execute(
                    select(episodes.c.video_config_id, episodes.c.drama_id).where(
                        episodes.c.id == sb_row.episode_id
                    )
                ).first()
                if episode is not None and episode[0] is not None:
                    config_id = episode[0]
                drama_style = resolve_effective_art_style(
                    conn, episode[1] if episode is not None else None
                )

                # 自动注入角色外观和场景描述到 prompt（除非明确跳过）
                if not js_truthy(body.get("_skip_enrich")):
                    char_appearances = get_storyboard_character_appearances(conn, sb_id)
                    scene_desc = get_storyboard_scene_description(conn, sb_id)

                    prompt = build_storyboard_video_prompt({
                        "description": body.get("prompt"),
                        "storyboardDescription": sb_row.description,
                        "characterAppearances": char_appearances,
                        "scenePrompt": scene_desc,
                        "action": sb_row.action,
                        "movement": sb_row.movement,
                        "dramaStyle": drama_style,
                        # H3 原生 [background_audio] 场景声标记
                        "backgroundAudio": sb_row.sound_effect,
                    })

                    # 自动添加参考图（角色立绘 + 场景图 + 道具图，与 auto-pipeline 批量路径同源，
                    # 保证手动/批量两种触发方式收到同一套跨集一致性参考；首帧已由前端传）
                    if not (reference_image_urls and len(reference_image_urls) > 0):
                        sb_refs = get_storyboard_reference_images(conn, sb_id)
                        if sb_refs:
                            reference_image_urls = sb_refs

                if not scene_type and sb_row.scene_type:
                    scene_type = sb_row.scene_type
                # H3 音视频联合生成：对话类镜头自动带出场角色声线样本作为参考音频（Ref2VA）
                if not (reference_audio_urls and len(reference_audio_urls) > 0) and (
                    VIDEO_ROUTE_DIALOGUE_PATTERN.search(sb_row.scene_type or "")
                ):
                    reference_audio_urls = get_storyboard_reference_audio_urls(conn, sb_id)

        log_task_start("VideoAPI", "generate", {
            "storyboardId": body.get("storyboard_id"),
            "dramaId": body.get("drama_id"),
            "referenceMode": body.get("reference_mode"),
            "duration": body.get("duration"),
        })
        log_task_payload("VideoAPI", "enriched prompt", {
            "original": body.get("prompt"), "enriched": prompt,
            "hasCharRefs": bool(reference_image_urls and len(reference_image_urls) > 0),
        })

        # —— 帧来源统一：body 显式传帧优先 → 分镜已存帧兜底 ——
        sb_first_frame = (getattr(sb_row, "first_frame_image", None) or None) if sb_row is not None else None
        sb_last_frame = (getattr(sb_row, "last_frame_image", None) or None) if sb_row is not None else None
        cand_first_frame_url = first_frame_url or sb_first_frame or None
        cand_last_frame_url = body.get("last_frame_url") or sb_last_frame or None

        # 逐镜路由：按分镜属性决策生成路线，并回写 storyboards.route / route_reason
        route_decision: dict[str, Any] = {}
        if sb_row is not None:
            provider = (
                (get_config_by_id(conn, body.get("config_id")) or {}).get("provider")
                if body.get("config_id")
                else (get_active_config(conn, "video") or {}).get("provider")
            ) or "default"
            can_multi_ref = (
                provider.lower() in _MULTI_REF_PROVIDERS or bool(body.get("reference_mode"))
            )
            route_decision = decide_shot_route(conn, {
                "storyboardId": sb_row.id,
                "sceneType": sb_row.scene_type,
                "firstFrameImage": cand_first_frame_url,
                "lastFrameImage": cand_last_frame_url,
                "keyframeImage": sb_row.keyframe_image,
                "blocked": sb_row.asset_status == "needs_regeneration" and not cand_first_frame_url,
                "provider": provider or "default",
                "canMultiRef": can_multi_ref,
                "referenceImages": reference_image_urls or [],
                "referenceAudioUrls": reference_audio_urls or [],
                # 同场景顺接：body 给了 image_url 但没给首帧时，把它当上一镜尾帧
                "prevTail": body.get("image_url") if (body.get("image_url") and not body.get("first_frame_url")) else None,
            })

        # referenceMode 兜底对齐 adapter 契约：first_last（FL2VA）必须首尾帧齐备才会被 adapter
        # 派发；只有单帧可用而目标模式落 first_last 时降级 single（用首帧起图），避免缺首帧的坏请求。
        requested_mode = body.get("reference_mode") or route_decision.get("referenceMode") or "none"
        if requested_mode == "first_last" and not (cand_first_frame_url and cand_last_frame_url):
            effective_mode = "single" if (cand_first_frame_url or body.get("image_url")) else "none"
        else:
            effective_mode = requested_mode

        # 仅 single / first_last 消费帧字段（multiple 走 reference_image_urls、none 不用帧），
        # 这两种模式才注入兜底帧，保持既有请求体不变。
        frame_mode = effective_mode in ("single", "first_last")
        out_first_frame_url = cand_first_frame_url if frame_mode else first_frame_url
        out_last_frame_url = cand_last_frame_url if effective_mode == "first_last" else body.get("last_frame_url")
        # adapter 单帧分支只读 imageUrl（无 first_frame_url 入参），故 single 时把首帧统一落到 imageUrl
        if effective_mode == "first_last":
            out_image_url = None
        else:
            out_image_url = (
                body.get("image_url")
                or (cand_first_frame_url if frame_mode else None)
                or None
            )

        video_id = await generate_video(conn, {
            "storyboardId": body.get("storyboard_id"),
            "dramaId": body.get("drama_id"),
            "prompt": prompt,
            "negativePrompt": body.get("negative_prompt") or build_video_negative_prompt(drama_style),
            "model": body.get("model"),
            "referenceMode": effective_mode,
            "imageUrl": out_image_url,
            "firstFrameUrl": out_first_frame_url,
            "lastFrameUrl": out_last_frame_url,
            "referenceImageUrls": reference_image_urls,
            "sceneType": scene_type,
            "referenceAudioUrls": reference_audio_urls,
            "duration": body.get("duration"),
            "aspectRatio": body.get("aspect_ratio"),
            "configId": config_id,
            "force": body.get("force"),
            "route": route_decision.get("route"),
            "routeReason": route_decision.get("reason"),
        })

        record = _fetch_video(conn, video_id)
        log_task_success("VideoAPI", "generate",
                         {"generationId": video_id, "provider": getattr(record, "provider", None)})
        return created(row_to_camel(record, "video_generations"))
    except Exception as exc:  # noqa: BLE001
        log_task_error("VideoAPI", "generate", {"error": str(exc)})
        return bad_request(str(exc))


@router.get("/{video_id}")
def get_video(video_id: str, conn: Connection = Depends(get_conn)):
    """⚠️ 查不到时返回 ``success(null)``（**不是 404**）—— 与 TS 一致。"""
    try:
        vid = parse_param_id(video_id)
        if vid is None:
            return not_found("Invalid video id")
        row = _fetch_video(conn, vid)
        return success(row_to_camel(row, "video_generations") if row is not None else None)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


@router.get("")
def list_videos(request: Request, conn: Connection = Depends(get_conn)):
    """按 ``storyboard_id`` / ``drama_id`` 过滤；两个都不是有限数字时**不过滤**。"""
    try:
        storyboard_id = request.query_params.get("storyboard_id")
        drama_id = request.query_params.get("drama_id")

        rows = conn.execute(select(video_generations)).all()

        if storyboard_id:
            sid = js_number(storyboard_id)
            if sid is not None:
                rows = [row for row in rows if row.storyboard_id == sid]
        if drama_id:
            did = js_number(drama_id)
            if did is not None:
                rows = [row for row in rows if row.drama_id == did]

        return success([row_to_camel(row, "video_generations") for row in rows])
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


@router.put("/{video_id}")
async def update_video(video_id: str, request: Request, conn: Connection = Depends(get_tx)):
    """更新视频生成参数（白名单字段）。⚠️ **不更新 updatedAt**（与 TS 一致）。"""
    try:
        vid = parse_param_id(video_id)
        if vid is None:
            return not_found("Invalid video id")
        body = await read_json(request)
        row = _fetch_video(conn, vid)
        if row is None:
            return bad_request("视频记录不存在")

        values: dict[str, Any] = {}
        for api_key, column in _UPDATE_FIELD_MAP.items():
            if api_key in body:
                values[column] = body[api_key]
        if not values:
            return bad_request("No fields to update")

        conn.execute(
            update(video_generations).where(video_generations.c.id == vid).values(**values)
        )
        updated = _fetch_video(conn, vid)
        return success(row_to_camel(updated, "video_generations"))
    except Exception as exc:  # noqa: BLE001
        log_task_error("VideoAPI", "update", {"error": str(exc), "id": video_id})
        return bad_request(str(exc))


@router.post("/{video_id}/regenerate")
async def regenerate_video(video_id: str, request: Request, conn: Connection = Depends(get_tx)):
    """重新生成视频（可覆盖模型与参数）；未给的一律**沿用原记录**。"""
    try:
        vid = parse_param_id(video_id)
        if vid is None:
            return not_found("Invalid video id")
        body = await read_json(request)
        row = _fetch_video(conn, vid)
        if row is None:
            return bad_request("视频记录不存在")

        prompt = body.get("prompt") or row.prompt
        if not js_truthy(prompt):
            return bad_request("prompt is required")

        log_task_start("VideoAPI", "regenerate", {
            "generationId": vid, "storyboardId": row.storyboard_id, "dramaId": row.drama_id,
            "model": body.get("model") or row.model or "default",
        })

        # 获取 configId：优先从请求体 → 从原记录关联的 episode
        config_id: Any = body.get("config_id")
        if not config_id and row.storyboard_id:
            sb_episode = conn.execute(
                select(storyboards.c.episode_id).where(storyboards.c.id == row.storyboard_id)
            ).first()
            if sb_episode is not None:
                episode = conn.execute(
                    select(episodes.c.video_config_id).where(episodes.c.id == sb_episode[0])
                ).first()
                if episode is not None and episode[0] is not None:
                    config_id = episode[0]

        new_id = await generate_video(conn, {
            "storyboardId": row.storyboard_id,
            "dramaId": row.drama_id,
            "prompt": prompt,
            # 原记录负面词优先（可复现）；缺失时按**剧集画风**补，而不是用与画风无关的通用词
            "negativePrompt": body.get("negative_prompt")
            or row.negative_prompt
            or build_video_negative_prompt(resolve_effective_art_style(conn, row.drama_id)),
            "model": body.get("model") or row.model,
            "referenceMode": body.get("reference_mode") or row.reference_mode,
            "imageUrl": body.get("image_url") or row.image_url,
            "firstFrameUrl": body.get("first_frame_url") or row.first_frame_url,
            "lastFrameUrl": body.get("last_frame_url") or row.last_frame_url,
            "referenceImageUrls": body.get("reference_image_urls") or row.reference_image_urls,
            "sceneType": body.get("scene_type") or row.scene_type or None,
            "referenceAudioUrls": body.get("reference_audio_urls")
            or (get_storyboard_reference_audio_urls(conn, row.storyboard_id) if row.storyboard_id else None),
            "duration": body.get("duration") or row.duration,
            "aspectRatio": body.get("aspect_ratio") or row.aspect_ratio,
            "configId": config_id,
            "force": body.get("force"),
        })

        record = _fetch_video(conn, new_id)
        log_task_success("VideoAPI", "regenerate", {
            "generationId": new_id, "oldId": vid, "provider": getattr(record, "provider", None),
        })
        return created(row_to_camel(record, "video_generations"))
    except Exception as exc:  # noqa: BLE001
        log_task_error("VideoAPI", "regenerate", {"generationId": video_id, "error": str(exc)})
        return bad_request(str(exc))


@router.delete("/{video_id}")
def delete_video(video_id: str, conn: Connection = Depends(get_tx)):
    try:
        vid = parse_param_id(video_id)
        if vid is None:
            return not_found("Invalid video id")
        conn.execute(sql_delete(video_generations).where(video_generations.c.id == vid))
        return success()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})
