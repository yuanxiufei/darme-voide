"""preset-framework 域 —— 与 ``backend/src/routes/preset-framework.ts``（247 行）对齐。

**6 个端点全部迁移**（前缀 ``/api/v1/preset/framework``）：

* ``GET  /variation-card``   随机生成一张 Variation Card（``?excludeFamily=`` 避免连续重复）；
* ``POST /create``           按 Card 建 Drama + Episode + Storyboards；
* ``POST /generate-images``  批量生首帧（复用 image 管线，**并发 + 失败不中断**）；
* ``POST /generate-videos``  批量生视频（图生视频，需首帧就绪）；
* ``GET  /status/:dramaId``  管线状态与进度汇总；
* ``POST /full-pipeline``    一键：建剧 → 批量生图。

⚠️ **本域响应形态与全项目都不同**（别"统一"成 ``code/data/message`` 信封）：

* 成功 ``{"success": true, "data": …}``；失败 ``{"success": false, "msg": …}``；
* 参数错 400、内部错 **500**（其它域多为 400 走同一信封）。

⚠️ ``GET /status/:dramaId`` 用 ``parseInt``（``'12abc'`` → 12，``'abc'`` → NaN ⇒ 400）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Connection

from ..db import get_conn, get_tx
from ..request_utils import read_json
from ..services.adapters.jscompat import js_parse_int
from ..services.preset_framework import (
    create_preset_drama,
    generate_variation_card,
    get_preset_pipeline_status,
    trigger_image_generation,
    trigger_video_generation,
)
from ..services.task_logger import log_task_error

router = APIRouter(prefix="/api/v1/preset/framework", tags=["preset-framework"])

#: 批量端点的参数校验文案（两个端点**完全相同**）
_BATCH_REQUIRED_MSG = "dramaId, episodeId, storyboardIds, and variationCard are required"


def _ok(data):
    """成功：``{success: true, data}``。"""
    return {"success": True, "data": data}


def _fail(message: str, status_code: int):
    """失败：``{success: false, msg}``。"""
    return JSONResponse(status_code=status_code, content={"success": False, "msg": message})


@router.get("/variation-card")
def variation_card(exclude_family: str | None = Query(None, alias="excludeFamily")):
    """随机生成一张 Variation Card（5 镜配置）。

    ⚠️ 查询参数名必须是 **camelCase ``excludeFamily``**（Node 是
    ``c.req.query('excludeFamily')``，前端按这个拼 URL）。蛇形形参名收不到它，
    而且因为带默认值会被**静默忽略** ⇒ 排除逻辑失效，只能靠「随机偶尔撞上」暴露。
    """
    try:
        return _ok(generate_variation_card(exclude_family or None))
    except Exception as err:  # noqa: BLE001
        return _fail(str(err), 500)


@router.post("/create")
async def create(request: Request):
    """基于 Variation Card 创建 Drama + Episode + N 个 Storyboard。"""


    body = await read_json(request)

    title = body.get("title")
    if not title:
        return _fail("title is required", 400)

    variation_card_payload = body.get("variationCard")
    if not variation_card_payload or not (variation_card_payload.get("shots") or []):
        return _fail("variationCard with shots is required", 400)

    try:
        result = create_preset_drama(title, body.get("description") or "", variation_card_payload)
        return _ok(result)
    except Exception as err:  # noqa: BLE001
        log_task_error("PresetFramework", "create-drama", {"error": str(err)})
        return _fail(str(err), 500)


@router.post("/generate-images")
async def generate_images(request: Request):
    """批量生成各 Shot 的首帧图。"""


    body = await read_json(request)
    drama_id = body.get("dramaId")
    episode_id = body.get("episodeId")
    storyboard_ids = body.get("storyboardIds")
    variation_card_payload = body.get("variationCard")

    if not drama_id or not episode_id or not storyboard_ids or not variation_card_payload:
        return _fail(_BATCH_REQUIRED_MSG, 400)

    try:
        result = await trigger_image_generation(
            drama_id, episode_id, storyboard_ids, variation_card_payload
        )
        return _ok({"dramaId": drama_id, "episodeId": episode_id, **result})
    except Exception as err:  # noqa: BLE001
        return _fail(str(err), 500)


@router.post("/generate-videos")
async def generate_videos(request: Request):
    """批量生成各 Shot 的视频（需首帧已就绪）。"""


    body = await read_json(request)
    drama_id = body.get("dramaId")
    episode_id = body.get("episodeId")
    storyboard_ids = body.get("storyboardIds")
    variation_card_payload = body.get("variationCard")

    if not drama_id or not episode_id or not storyboard_ids or not variation_card_payload:
        return _fail(_BATCH_REQUIRED_MSG, 400)

    try:
        result = await trigger_video_generation(
            drama_id, episode_id, storyboard_ids, variation_card_payload
        )
        return _ok({"dramaId": drama_id, "episodeId": episode_id, **result})
    except Exception as err:  # noqa: BLE001
        log_task_error("PresetFramework", "generate-videos", {"error": str(err)})
        return _fail(str(err), 500)


@router.get("/status/{drama_id}")
def status(drama_id: str, conn: Connection = Depends(get_conn)):
    """管线状态：Drama / Episode / Storyboards / Card / 进度汇总。"""
    parsed = js_parse_int(drama_id)
    if parsed is None:
        return _fail("Invalid dramaId", 400)
    try:
        return _ok(get_preset_pipeline_status(conn, parsed))
    except Exception as err:  # noqa: BLE001
        return _fail(str(err), 500)


@router.post("/full-pipeline")
async def full_pipeline(request: Request):
    """一键：生成 Card → 建剧 → 批量生图。"""


    body = await read_json(request)
    title = body.get("title")
    if not title:
        return _fail("title is required", 400)

    variation_card_payload = body.get("variationCard")
    if not variation_card_payload or not (variation_card_payload.get("shots") or []):
        return _fail("variationCard with shots is required", 400)

    try:
        created = create_preset_drama(title, body.get("description") or "", variation_card_payload)

        image_gen_ids: list[int] = []
        # `autoGenerateImages = true` 是**默认值**，只有显式 false 才跳过
        if body.get("autoGenerateImages", True) is not False:
            result = await trigger_image_generation(
                created["dramaId"], created["episodeId"], created["storyboardIds"],
                variation_card_payload,
            )
            image_gen_ids = result["imageGenIds"]

        return _ok({
            "dramaId": created["dramaId"],
            "episodeId": created["episodeId"],
            "storyboardIds": created["storyboardIds"],
            "imageGenIds": image_gen_ids,
            "variationCard": variation_card_payload,
        })
    except Exception as err:  # noqa: BLE001
        log_task_error("PresetFramework", "full-pipeline", {"error": str(err)})
        return _fail(str(err), 500)
