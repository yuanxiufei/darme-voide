"""images 域 —— 与 ``backend/src/routes/images.ts``（135 行）对齐。

**4 个端点全部迁移**（前缀 ``/api/v1/images``）：``POST /``（提交出图）、``GET /{id}``、
``GET /``（按分镜/剧集过滤）、``DELETE /{id}``。

⚠️ 四处保真点：

* **负面词的二选一**：给了 ``storyboard_id`` 用 ``build_storyboard_negative_prompt(画风)``，
  **否则用 ``NEGATIVE_BASE``** —— 不是同一个东西，别统一；
* 上下文注入（角色外观 + 场景 + 分镜字段）只在 ``storyboard_id`` 且**未** ``_skip_enrich`` 时做；
* ``storyboard_id`` 存在但**分镜查不到**时静默跳过注入（不报错）；
* 本域返回 **camelCase 行**（drizzle 行）—— ``GET /{id}`` 查不到是 ``success(null)``，不是 404。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..db import get_conn, get_tx
from ..models import episodes, image_generations, storyboards
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
from ..services.image_generation import generate_image
from ..services.prompt_utils import (
    NEGATIVE_BASE,
    build_storyboard_image_prompt,
    build_storyboard_negative_prompt,
    get_storyboard_character_appearances,
    get_storyboard_reference_images,
    get_storyboard_scene_description,
    resolve_effective_art_style,
)
from ..services.task_logger import log_task_error, log_task_payload, log_task_start, log_task_success

router = APIRouter(prefix="/api/v1/images", tags=["images"])


@router.post("")
async def create_image(request: Request, conn: Connection = Depends(get_tx)):
    """提交图片生成（可按分镜自动富化 prompt 与参考图）。"""
    try:
        body = await read_json(request)
        if not js_truthy(body.get("prompt")):
            return bad_request("prompt is required")

        config_id: Any = body.get("config_id")
        prompt = body.get("prompt")
        reference_images = body.get("reference_images")
        drama_style: str | None = None

        if js_truthy(body.get("storyboard_id")):
            sb_id = js_number(body.get("storyboard_id"))
            row = None
            if sb_id is not None:
                row = conn.execute(
                    select(storyboards).where(storyboards.c.id == sb_id)
                ).first()
            if row is not None:
                episode = conn.execute(
                    select(episodes.c.image_config_id, episodes.c.drama_id)
                    .where(episodes.c.id == row.episode_id)
                ).first()
                if episode is not None and episode[0] is not None:
                    config_id = episode[0]
                drama_style = resolve_effective_art_style(
                    conn, episode[1] if episode is not None else None
                )

                # 自动注入角色外观和场景描述到 prompt
                if not js_truthy(body.get("_skip_enrich")):
                    char_appearances = get_storyboard_character_appearances(conn, sb_id)
                    scene_desc = get_storyboard_scene_description(conn, sb_id)

                    prompt = build_storyboard_image_prompt({
                        "description": body.get("prompt"),
                        "storyboardDescription": row.description,
                        # ⚠️ 角色外观在这里用**全角分号**连接（别改成逗号）
                        "characterDescription": "；".join(char_appearances) if char_appearances else None,
                        "sceneDescription": scene_desc,
                        "location": row.location,
                        "shotType": row.shot_type,
                        "cameraAngle": row.angle,
                        "dramaStyle": drama_style,
                    })

                    # 自动添加角色图 + 场景图作为参考图，保证人物与场景一致
                    if not (reference_images and len(reference_images)):
                        ref_urls = get_storyboard_reference_images(conn, sb_id)
                        if ref_urls:
                            reference_images = ref_urls

        log_task_start("ImageAPI", "generate", {
            "storyboardId": body.get("storyboard_id"),
            "sceneId": body.get("scene_id"),
            "characterId": body.get("character_id"),
            "dramaId": body.get("drama_id"),
            "frameType": body.get("frame_type"),
        })
        log_task_payload("ImageAPI", "enriched prompt", {
            "original": body.get("prompt"), "enriched": prompt,
            "referenceCount": len(reference_images) if reference_images else 0,
        })

        generation_id = await generate_image(conn, {
            "storyboardId": body.get("storyboard_id"),
            "dramaId": body.get("drama_id"),
            "sceneId": body.get("scene_id"),
            "characterId": body.get("character_id"),
            "prompt": prompt,
            # ⚠️ 有分镜 -> 分镜负面词（含画风对立词）；无分镜 -> NEGATIVE_BASE（通用负面词）
            "negativePrompt": body.get("negative_prompt") or (
                build_storyboard_negative_prompt(drama_style)
                if body.get("storyboard_id") else NEGATIVE_BASE
            ),
            "model": body.get("model"),
            "size": body.get("size"),
            "referenceImages": reference_images,
            "frameType": body.get("frame_type"),
            "configId": config_id,
            "force": body.get("force"),
        })

        record = conn.execute(
            select(image_generations).where(image_generations.c.id == generation_id)
        ).first()
        log_task_success("ImageAPI", "generate", {
            "generationId": generation_id,
            "provider": getattr(record, "provider", None),
        })
        return created(row_to_camel(record, "image_generations") if record is not None else None)
    except Exception as err:  # noqa: BLE001
        log_task_error("ImageAPI", "generate", {"error": str(err)})
        return bad_request(str(err))


@router.get("/{image_id}")
def get_image(image_id: str, conn: Connection = Depends(get_conn)):
    """⚠️ 查不到返回 ``success(null)``（**不是 404**）。"""
    try:
        iid = parse_param_id(image_id)
        if iid is None:
            return not_found("Invalid image id")
        row = conn.execute(
            select(image_generations).where(image_generations.c.id == iid)
        ).first()
        return success(row_to_camel(row, "image_generations") if row is not None else None)
    except Exception as err:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(err)})


@router.get("")
def list_images(request: Request, conn: Connection = Depends(get_conn)):
    """按 ``storyboard_id`` / ``drama_id`` 过滤（非有限数字时**不过滤**）。"""
    try:
        storyboard_id = request.query_params.get("storyboard_id")
        drama_id = request.query_params.get("drama_id")

        rows = conn.execute(select(image_generations)).all()

        if storyboard_id:
            sid = js_number(storyboard_id)
            if sid is not None:
                rows = [row for row in rows if row.storyboard_id == sid]
        if drama_id:
            did = js_number(drama_id)
            if did is not None:
                rows = [row for row in rows if row.drama_id == did]

        return success([row_to_camel(row, "image_generations") for row in rows])
    except Exception as err:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(err)})


@router.delete("/{image_id}")
def delete_image(image_id: str, conn: Connection = Depends(get_tx)):
    try:
        iid = parse_param_id(image_id)
        if iid is None:
            return not_found("Invalid image id")
        conn.execute(sql_delete(image_generations).where(image_generations.c.id == iid))
        return success()
    except Exception as err:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(err)})
