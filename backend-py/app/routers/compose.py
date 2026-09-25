"""compose 域 —— 与 ``backend/src/routes/compose.ts``（108 行）对齐。

**3 个端点全部迁移**：``POST /storyboards/{id}/compose``（单镜合成）、
``POST /episodes/{id}/compose-all``（批量）、``GET /episodes/{id}/compose-status``（进度）。

⚠️ **路径前缀是 ``/api/v1/compose``** —— Node 的 ``index.ts`` 里是
``api.route('/compose', compose)``，而 compose.ts 自己的子路径又带 ``/storyboards``、
``/episodes``，所以真实路径是 ``/api/v1/compose/episodes/{id}/compose-all``（看着别扭但保真）。

⚠️ 两处保真点：

* 单镜合成返回的是 **snake_case** ``{id, composed_video_url}``（不是 camelCase 行）；
* 批量合成是**火忘型**：立刻返回 ``{message, total}``，真正的循环在后台跑，
  且**只把有视频的分镜**标记为 ``compose_processing``（无视频的保持原状）。
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, update
from sqlalchemy.engine import Connection

from ..core.db import engine, get_conn, get_tx
from ..core.models import storyboards
from ..core.request_utils import read_json
from ..core.response import bad_request, not_found, parse_param_id, success
from ..services.ffmpeg_compose import compose_storyboard
from ..services.task_logger import log_task_error, log_task_start, log_task_success

router = APIRouter(prefix="/api/v1/compose", tags=["compose"])


async def _run_batch(episode_id: int, storyboard_ids: list[int]) -> None:
    """批量合成的后台循环（**每镜一条独立连接**：请求连接早已随响应关闭）。"""
    ok = 0
    fail = 0
    for storyboard_id in storyboard_ids:
        try:
            # compose_storyboard 自己管连接（每步一个短事务），这里不再借出连接
            await compose_storyboard(storyboard_id)
            ok += 1
        except Exception as err:  # noqa: BLE001 —— 单镜失败不影响其余
            fail += 1
            log_task_error("ComposeAPI", "batch-item", {
                "storyboardId": storyboard_id, "episodeId": episode_id, "error": str(err),
            })
    log_task_success("ComposeAPI", "batch-compose", {
        "episodeId": episode_id, "total": len(storyboard_ids), "ok": ok, "fail": fail,
    })


@router.post("/storyboards/{storyboard_id}/compose")
async def compose_one(storyboard_id: str, request: Request):
    """合成单个镜头（把该镜视频 + 配音 + 字幕烧成成片片段）。

    ⚠️ 不注入数据库连接：``compose_storyboard`` 自己开短事务写
    ``status`` / ``composed_video_url`` / ``tts_audio_url`` / ``subtitle_url``。
    若借用请求连接，**失败路径会被依赖的 rollback 抹掉**（写不进 ``compose_failed``）。

    可选 body 字段（2026-09-24 接 ✓，**默认 false ⇒ 行为一字不差** ✗）：
    ``mixModelAudio``（把视频自带的音轨与配音**混合** ✓✗ 顶替会让模型声直接消失 ✓）、
    ``voiceMode``（``mix`` 混合 / ``replace`` 顶替 ✓）、``voiceVolume``（配音音量 ✓）。
    """
    sid = parse_param_id(storyboard_id)
    if sid is None:
        return not_found("Invalid storyboard id")
    body = await read_json(request)
    voice_mode = str(body.get("voiceMode") or "mix").strip().lower()
    if voice_mode not in ("mix", "replace"):
        return bad_request(f"voiceMode 只能是 mix / replace（收到 {voice_mode!r} ✗）")
    raw_volume = body.get("voiceVolume")
    try:
        voice_volume = float(1.0 if raw_volume in (None, "") else raw_volume)
    except (TypeError, ValueError):
        return bad_request(f"voiceVolume 必须是数字（收到 {raw_volume!r} ✗）")
    try:
        log_task_start("ComposeAPI", "single-compose", {"storyboardId": sid})
        composed_url = await compose_storyboard(
            sid, mix_model_audio=bool(body.get("mixModelAudio")),
            voice_mode=voice_mode, voice_volume=voice_volume)
        log_task_success("ComposeAPI", "single-compose", {"storyboardId": sid, "output": composed_url})
        return success({"id": sid, "composed_video_url": composed_url})
    except Exception as err:  # noqa: BLE001
        log_task_error("ComposeAPI", "single-compose", {"storyboardId": sid, "error": str(err)})
        return bad_request(str(err))


@router.post("/episodes/{episode_id}/compose-all")
async def compose_all(episode_id: str, conn: Connection = Depends(get_tx)):
    """批量合成整集（**火忘型**：立即返回，后台逐镜合成）。

    ⚠️ 同样要用 ``get_tx``：这里会把有视频的分镜标记成 ``compose_processing``，
    只读连接不会提交，前端轮询就永远看不到「进行中」。
    """
    eid = parse_param_id(episode_id)
    if eid is None:
        return not_found("Invalid episode id")
    try:
        rows = conn.execute(
            select(storyboards.c.id, storyboards.c.video_url)
            .where(storyboards.c.episode_id == eid)
            .order_by(storyboards.c.storyboard_number)
        ).all()
        if not rows:
            return bad_request("No storyboards found")

        with_video = [row[0] for row in rows if row[1]]
        if not with_video:
            return bad_request("No storyboards have video yet")

        log_task_start("ComposeAPI", "batch-compose", {"episodeId": eid, "total": len(with_video)})

        # 只把有视频的分镜标记为合成中（无视频的保持原状）
        conn.execute(
            update(storyboards).where(storyboards.c.id.in_(with_video))
            .values(status="compose_processing")
        )

        asyncio.create_task(_run_batch(eid, with_video))

        return success({
            "message": f"Started composing {len(with_video)} storyboards",
            "total": len(with_video),
        })
    except Exception as err:  # noqa: BLE001
        log_task_error("ComposeAPI", "batch-compose", {"episodeId": episode_id, "error": str(err)})
        return bad_request(str(err))


@router.get("/episodes/{episode_id}/compose-status")
def compose_status(episode_id: str, conn: Connection = Depends(get_conn)):
    """查询批量合成进度（统计 + 逐镜明细）。"""
    try:
        eid = parse_param_id(episode_id)
        if eid is None:
            return not_found("Invalid episode id")
        rows = conn.execute(
            select(
                storyboards.c.id, storyboards.c.storyboard_number,
                storyboards.c.status, storyboards.c.composed_video_url,
                storyboards.c.video_url,
            )
            .where(storyboards.c.episode_id == eid)
            .order_by(storyboards.c.storyboard_number)
        ).all()

        # ⚠️ 只统计**有视频**的分镜（无视频的不进 total）
        with_video = [row for row in rows if row[4]]
        completed = [r for r in with_video if r[2] == "compose_completed" and r[3]]
        failed = [r for r in with_video if r[2] == "compose_failed"]
        processing = [r for r in with_video if r[2] == "compose_processing"]
        # `!status || !String(status).startsWith('compose_')`
        idle = [r for r in with_video if not r[2] or not str(r[2]).startswith("compose_")]

        return success({
            "total": len(with_video),
            "completed": len(completed),
            "failed": len(failed),
            "processing": len(processing),
            "idle": len(idle),
            # 逐镜明细走 snake_case（TS 用 toSnakeCase 包装字面量）
            "items": [
                {
                    "id": row[0],
                    "storyboard_number": row[1],
                    "status": row[2] or "pending",
                    "composed_video_url": row[3],
                    "error_msg": (
                        "视频合成失败，请检查视频、配音或字幕素材"
                        if row[2] == "compose_failed" else ""
                    ),
                }
                for row in with_video
            ],
        })
    except Exception as err:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(err)})
