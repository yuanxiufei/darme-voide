"""全自动管线编排器 —— 与 ``services/auto-pipeline.ts``（814 行）对齐。

「一句话梗概 → 整集短剧」：自动串联 4 个 LLM Agent
``script_rewriter → extractor → voice_assigner → storyboard_breaker``，
并可选接入媒体阶段：图片首帧 → 图生视频 → 单镜头合成（TTS+字幕）→ 整集拼接。
**全程无人值守**，前端只需轮询 ``episode.status``。

## 幂等与崩溃恢复设计

1. 以 ``episode.status``（``auto:*`` 前缀）为**单向状态机**，每个阶段执行成功才推进到下一阶段；
2. Agent 阶段工具本身幂等（``save_script`` 覆盖、``save_dedup_*`` 去重、``save_storyboards``
   重建、``assign_voice`` 覆盖），**重跑安全**，不产生脏数据；
3. 媒体阶段提交前检查已有产物（``first_frame_image`` / ``video_url`` / ``composed_video_url``）
   或已有 ``processing`` 任务 ⇒ 崩溃重启**不会重复提交**（避免重复扣费）；
4. 启动时 ``recover_auto_pipeline_on_startup()`` 扫描中间态 episode 自动续跑（幂等）。

⚠️ 六处保真点（都极易写错）：

1. **阶段顺序 ≠ 直觉**：``scripting → extracting → **voicing** → storyboarding``（**配音在分镜之前**），
   而 ``STAGE_SEQUENCE`` 里的顺序就是幂等判据，改顺序等于改「跳过哪些阶段」；
2. ``is_stage_completed`` 用 **``next`` 索引**比对（不是当前阶段）：``curIdx >= nextIdx`` 即视为该阶段已完成；
3. **媒体阶段开关有依赖补全**（``media_needs``）：``video`` 隐含 ``image``、``compose`` 隐含
   ``image+video``、``merge`` 隐含全部 —— 只勾 ``withMerge`` 也会把图片阶段跑起来；
4. **只写 ``tail_frame_image``**（真实尾帧）作为下一镜起帧，**绝不碰 ``last_frame_image``**
   （设计尾帧是 FL2VA 的决策依据，不能被产物覆盖）；
5. **同进程防重入 + follow-up 队列**：执行期间再次触发**入队而非丢弃**，本轮结束后自动重跑一轮；
6. ``JSON.stringify(opts)`` 存进 ``dramas.metadata`` ⇒ 必须用**紧凑分隔符**（``json.dumps`` 守卫）。
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any, Callable, Coroutine

from sqlalchemy import and_, asc, select, update
from sqlalchemy.engine import Connection

from app.core.db import engine
from app.core.models import (
    dramas,
    episode_characters,
    episode_scenes,
    episodes,
    image_generations,
    storyboards,
    video_generations,
    video_merges,
)
from app.core.response import now
from app.agent.runtime import run_agent_with_retry
from app.services.ai_providers import get_active_config, get_config_by_id
from app.services.ffmpeg_compose import compose_storyboard
from app.services.ffmpeg_merge import merge_episode_videos
from app.services.frame_extractor import extract_storyboard_tail_frames
from app.services.image_generation import generate_image
from app.services.prompt_utils import (
    build_storyboard_art_style_suffix,
    build_storyboard_negative_prompt,
    build_video_art_style_suffix,
    build_video_negative_prompt,
    get_storyboard_reference_audio_urls,
    get_storyboard_reference_images,
    resolve_effective_art_style,
)
from app.services.shot_router import decide_shot_route
from app.services.sse_hub import publish_pipeline_event
from app.services.task_logger import (
    log_task_error,
    log_task_progress,
    log_task_start,
    log_task_success,
    log_task_warn,
)
from app.services.video_generation import generate_video

__all__ = [
    "AUTO_STATUS",
    "MEDIA_POLL_INTERVAL_MS",
    "MEDIA_WAIT_TIMEOUT_MS",
    "execute_episode_pipeline",
    "execute_pipeline",
    "get_auto_pipeline_status",
    "is_stage_completed",
    "media_needs",
    "normalize_options",
    "recover_auto_pipeline_on_startup",
    "resume_auto_pipeline",
    "run_auto_pipeline",
]

#: 媒体阶段轮询参数
MEDIA_WAIT_TIMEOUT_MS = 30 * 60 * 1000
MEDIA_POLL_INTERVAL_MS = 5000

#: 支持多参考图（``referenceMode='multiple'``）的视频提供商；其余仅首尾帧
MULTI_REFERENCE_PROVIDERS = {"volcengine", "vidu", "minimax"}

#: 管线阶段状态（存 ``episodes.status``，``auto:`` 前缀避免与业务状态 draft/completed 冲突）
AUTO_STATUS: dict[str, str] = {
    "queued": "auto:queued",
    "scripting": "auto:scripting",
    "extracting": "auto:extracting",
    "storyboarding": "auto:storyboarding",
    "voicing": "auto:voicing",
    "imaging": "auto:imaging",
    "videoing": "auto:videoing",
    "composing": "auto:composing",
    "merging": "auto:merging",
    "done": "auto:done",
    "failed": "auto:failed",
}

#: 阶段顺序（幂等判据：当前 status 索引 >= 某阶段 ``next`` 索引即视为该阶段已完成）
#: ⚠️ 注意 ``voicing`` 排在 ``storyboarding`` **之前**（先配音后分镜，别按直觉改）
STAGE_SEQUENCE: list[str] = [
    AUTO_STATUS["queued"],
    AUTO_STATUS["scripting"],
    AUTO_STATUS["extracting"],
    AUTO_STATUS["voicing"],
    AUTO_STATUS["storyboarding"],
    AUTO_STATUS["imaging"],
    AUTO_STATUS["videoing"],
    AUTO_STATUS["composing"],
    AUTO_STATUS["merging"],
    AUTO_STATUS["done"],
]

#: 中间态（崩溃恢复扫描范围）
IN_FLIGHT_STATUSES: list[str] = [
    AUTO_STATUS["queued"],
    AUTO_STATUS["scripting"],
    AUTO_STATUS["extracting"],
    AUTO_STATUS["voicing"],
    AUTO_STATUS["storyboarding"],
    AUTO_STATUS["imaging"],
    AUTO_STATUS["videoing"],
    AUTO_STATUS["composing"],
    AUTO_STATUS["merging"],
]

#: 可注入的睡眠（自检替换成 no-op，避免真等 5 秒 × N 轮）
_sleep = asyncio.sleep

#: 同进程防重入锁：崩溃恢复与手动触发并发时，同一 episode 只执行一次
_running_episodes: set[int] = set()

#: follow-up 队列：执行期间再次触发时**入队**（而非静默丢弃），本轮完成后自动重跑一轮
_queued_episodes: set[int] = set()


# ============================================================
# 选项归一
# ============================================================

def normalize_options(opts: dict[str, Any]) -> dict[str, Any]:
    """归一化管线选项（⚠️ 键名保持 **camelCase** —— 它会序列化进 ``dramas.metadata`` 再读回）。"""
    raw_count = opts.get("episodeCount")
    try:
        count = float(raw_count) if raw_count is not None else 0
    except (TypeError, ValueError):
        count = 0
    episode_count = int(math.floor(count)) if count > 0 else 1

    premise = opts.get("premise") or ""
    title = (opts.get("title") or "").strip() or premise[:24]
    genre = (opts.get("genre") or "").strip() or "短剧"
    style = (opts.get("style") or "").strip() or "realistic"

    return {
        **opts,
        "episodeCount": episode_count,
        "title": title,
        "genre": genre,
        "style": style,
        "withImages": bool(opts.get("withImages")),
        "withVideos": bool(opts.get("withVideos")),
        "withCompose": bool(opts.get("withCompose")),
        "withMerge": bool(opts.get("withMerge")),
    }


def media_needs(opts: dict[str, Any]) -> dict[str, bool]:
    """依赖补全：计算各媒体阶段**是否真正需要执行**（video 隐含 image，compose 隐含 image+video…）。"""
    return {
        "needImage": bool(opts.get("withImages") or opts.get("withVideos")
                           or opts.get("withCompose") or opts.get("withMerge")),
        "needVideo": bool(opts.get("withVideos") or opts.get("withCompose")
                           or opts.get("withMerge")),
        "needCompose": bool(opts.get("withCompose") or opts.get("withMerge")),
        "needMerge": bool(opts.get("withMerge")),
    }


# ============================================================
# 通用读取
# ============================================================

def _fetch_episode(conn: Connection, episode_id: int) -> Any:
    return conn.execute(select(episodes).where(episodes.c.id == episode_id)).first()


def is_drama_deleted(conn: Connection, drama_id: int) -> bool:
    """drama 是否**已软删**（删除竞态保护：已删剧本不再写数据，防止管线复活已删对象）。"""
    drama = conn.execute(select(dramas).where(dramas.c.id == drama_id)).first()
    return drama is None or bool(drama.deleted_at)


def get_storyboards(conn: Connection, episode_id: int) -> list[Any]:
    """本集分镜（按 ``storyboard_number`` 升序）。⚠️ 与原 TS 一致：**不过滤软删**。"""
    return list(conn.execute(
        select(storyboards).where(storyboards.c.episode_id == episode_id)
        .order_by(asc(storyboards.c.storyboard_number))
    ).all())


def set_status(episode_id: int, status: str) -> None:
    """推进阶段状态并**推送 SSE 事件**（publish 永不抛，绝不拖垮管线）。"""
    with engine.begin() as conn:
        conn.execute(
            update(episodes).where(episodes.c.id == episode_id)
            .values(status=status, updated_at=now())
        )
        drama_id = conn.execute(
            select(episodes.c.drama_id).where(episodes.c.id == episode_id)
        ).first()
    if drama_id is not None and drama_id[0] is not None:
        publish_pipeline_event(drama_id[0], {
            "type": "status", "episodeId": episode_id, "status": status,
        })


def is_stage_completed(current_status: str | None, next_status: str) -> bool:
    """当前 status 是否**已越过** ``next_status``（即该阶段已完成）。"""
    if not current_status or current_status == AUTO_STATUS["failed"]:
        return False
    cur_index = STAGE_SEQUENCE.index(current_status) if current_status in STAGE_SEQUENCE else -1
    next_index = STAGE_SEQUENCE.index(next_status) if next_status in STAGE_SEQUENCE else -1
    if next_index == -1:
        return True
    if cur_index == -1:
        return False
    return cur_index >= next_index


def is_storyboard_blocked(conn: Connection, storyboard_id: int) -> bool:
    """分镜是否已被**资产门禁**阻断（存在 ``blocked_by_missing_asset`` 视频记录）。"""
    rows = conn.execute(
        select(video_generations.c.id).where(and_(
            video_generations.c.storyboard_id == storyboard_id,
            video_generations.c.status == "blocked_by_missing_asset",
        ))
    ).all()
    return len(rows) > 0


# ============================================================
# Agent 阶段
# ============================================================

async def run_script_stage(episode_id: int, drama_id: int) -> None:
    with engine.begin() as conn:
        await run_agent_with_retry(
            conn, "script_rewriter", episode_id, drama_id,
            "请读取本集的原始剧情内容（梗概/小说），将其改写为符合平台规范的完整短剧剧本"
            "（含场景划分、对白、动作与心理描写），并保存到本集。",
        )
        episode = _fetch_episode(conn, episode_id)
    if episode is None or not (episode.script_content or "").strip():
        raise ValueError("script_rewriter 未产出剧本（script_content 为空）")


async def run_extract_stage(episode_id: int, drama_id: int) -> None:
    with engine.begin() as conn:
        await run_agent_with_retry(
            conn, "extractor", episode_id, drama_id,
            "请从本集剧本中提取去重后的角色与场景，并保存关联到本集。",
        )


async def run_voice_stage(episode_id: int, drama_id: int) -> None:
    with engine.begin() as conn:
        await run_agent_with_retry(
            conn, "voice_assigner", episode_id, drama_id,
            "请从可用音色库中为本剧角色分配最匹配的音色，并保存。",
        )


async def run_storyboard_stage(episode_id: int, drama_id: int) -> None:
    with engine.begin() as conn:
        await run_agent_with_retry(
            conn, "storyboard_breaker", episode_id, drama_id,
            "请将本集剧本按场景拆分为分镜，为每个分镜生成画面描述（image_prompt）与运镜提示"
            "（video_prompt）及对白，并保存。",
        )
        count = len(get_storyboards(conn, episode_id))
    if count == 0:
        raise ValueError("storyboard_breaker 未产出分镜（storyboards 为空）")


# ============================================================
# 媒体阶段
# ============================================================

async def submit_missing_keyframes(episode_id: int, drama_id: int,
                                   config_id: int | None = None) -> int:
    """提交缺失的**中段关键帧**图（幂等：有 keyframe_prompt 但缺图、且无进行中任务时提交）。"""
    submitted = 0
    with engine.begin() as conn:
        arts = get_storyboards(conn, episode_id)
        # 画风收口：批量管线与手动路由**共用同一套**画风解析与后缀（skill 约定 agent 不写
        # 画风词，故由后端统一注入，保证「批量出图」与「单张重跑」画风一致）
        art_style = resolve_effective_art_style(conn, drama_id)

    for sb in arts:
        if not sb.keyframe_prompt or sb.keyframe_image:
            continue
        with engine.begin() as conn:
            existing = conn.execute(
                select(image_generations.c.status).where(and_(
                    image_generations.c.storyboard_id == sb.id,
                    image_generations.c.frame_type == "keyframe",
                ))
            ).all()
            if any(row[0] in ("processing", "pending", "completed") for row in existing):
                continue
            await generate_image(conn, {
                "storyboardId": sb.id,
                "dramaId": drama_id,
                "prompt": f"{sb.keyframe_prompt}{build_storyboard_art_style_suffix(art_style)}",
                "negativePrompt": build_storyboard_negative_prompt(art_style),
                "frameType": "keyframe",
                "configId": config_id,
            })
        submitted += 1
    return submitted


async def submit_missing_images(episode_id: int, drama_id: int,
                                config_id: int | None = None) -> int:
    """提交缺失的**首帧**图（幂等：已有首帧 / 已有 processing/completed 任务则跳过）。"""
    submitted = 0
    with engine.begin() as conn:
        arts = get_storyboards(conn, episode_id)
        art_style = resolve_effective_art_style(conn, drama_id)

    for sb in arts:
        if sb.first_frame_image:
            continue
        with engine.begin() as conn:
            existing = conn.execute(
                select(image_generations.c.status).where(and_(
                    image_generations.c.storyboard_id == sb.id,
                    image_generations.c.frame_type == "first_frame",
                ))
            ).all()
            if any(row[0] in ("processing", "pending", "completed") for row in existing):
                continue
            prompt = sb.image_prompt or sb.description or "短剧分镜画面，电影感构图"
            await generate_image(conn, {
                "storyboardId": sb.id,
                "dramaId": drama_id,
                "prompt": f"{prompt}{build_storyboard_art_style_suffix(art_style)}",
                "negativePrompt": build_storyboard_negative_prompt(art_style),
                "frameType": "first_frame",
                "configId": config_id,
            })
        submitted += 1
    return submitted


async def submit_missing_videos(episode_id: int, drama_id: int,
                                config_id: int | None = None) -> int:
    """提交缺失的**视频**（幂等：需首帧已就绪 + 无 processing/completed 任务）。"""
    submitted = 0
    with engine.begin() as conn:
        arts = get_storyboards(conn, episode_id)
        art_style = resolve_effective_art_style(conn, drama_id)
        config = (get_config_by_id(conn, config_id) if config_id
                  else get_active_config(conn, "video"))
    provider = str((config or {}).get("provider") or "").lower()
    can_multi_ref = provider in MULTI_REFERENCE_PROVIDERS

    for index, sb in enumerate(arts):
        if sb.video_url:
            continue

        if not sb.first_frame_image:
            # 资产验收门禁：首帧资产标记为 needs_regeneration 时，写入 blocked 视频记录
            # （状态 blocked_by_missing_asset），阻止继续提交并向前端暴露阻断原因
            if sb.asset_status == "needs_regeneration":
                with engine.begin() as conn:
                    existing = conn.execute(
                        select(video_generations.c.status)
                        .where(video_generations.c.storyboard_id == sb.id)
                    ).all()
                    already_blocked = any(
                        row[0] == "blocked_by_missing_asset" for row in existing)
                    if not already_blocked:
                        ts = now()
                        conn.execute(video_generations.insert().values(
                            drama_id=drama_id, storyboard_id=sb.id,
                            status="blocked_by_missing_asset",
                            block_reason="first frame asset needs_regeneration",
                            prompt=sb.video_prompt or sb.image_prompt or sb.description or "",
                            created_at=ts, updated_at=ts,
                        ))
                if not already_blocked:
                    log_task_warn("AutoPipeline", "asset-gate-blocked", {
                        "storyboardId": sb.id, "reason": "first_frame_needs_regeneration",
                    })
            continue

        with engine.begin() as conn:
            existing = conn.execute(
                select(video_generations.c.status)
                .where(video_generations.c.storyboard_id == sb.id)
            ).all()
            if any(row[0] in ("processing", "pending", "completed") for row in existing):
                continue

            # scene_type 路由：对话/多人/争吵类场景在支持多参考图的 provider 上传「角色+场景」
            # 参考图，保证多人同框一致；其余保持首帧图生视频（仅支持首尾帧的 provider 自动降级）
            scene_type = sb.scene_type or ""
            is_dialogue = bool(re.search(r"dialogue|meeting|argument|conversation|multi",
                                         scene_type, re.IGNORECASE))
            reference_images = (get_storyboard_reference_images(conn, sb.id)[:9]
                                if is_dialogue and can_multi_ref else [])
            # 关键帧扩展：有中段关键帧图时并入参考图（锁定动作/道具/机位中间态）
            if can_multi_ref and sb.keyframe_image:
                reference_images = (reference_images + [sb.keyframe_image])[:9]
            # H3 音视频联合生成：对话类镜头带出场角色声线样本作为参考音频（Ref2VA）
            reference_audio_urls = (get_storyboard_reference_audio_urls(conn, sb.id)
                                    if is_dialogue and provider == "minimax" else [])

            # 尾帧衔接（连续性状态机 v3）：同场景顺接时，以「上一镜**真实尾帧**」作为本镜视频
            # 起帧；真实尾帧优先，其次设计尾帧；都没有或跨场景时回退本镜首帧图起帧。
            prev = arts[index - 1] if index > 0 else None
            same_scene = (prev is not None and prev.scene_id is not None
                          and prev.scene_id == sb.scene_id)
            prev_tail = ((prev.tail_frame_image or prev.last_frame_image)
                         if same_scene and prev is not None else None)
            anchor_image = prev_tail or sb.first_frame_image
            if prev_tail and prev is not None:
                log_task_progress("AutoPipeline", "tail-link", {
                    "storyboardId": sb.id, "linkedFrom": prev.id, "fromTail": prev_tail,
                })

            # 逐镜路由：提交前显式决策生成路线（T2V/I2V/FL2VA/R2V/K-Frame/Editor），
            # 并回写 storyboards.route / route_reason，供可复现账本与前端展示
            route_decision = decide_shot_route(conn, {
                "storyboardId": sb.id,
                "sceneType": sb.scene_type,
                "firstFrameImage": sb.first_frame_image,
                "lastFrameImage": sb.last_frame_image,
                "keyframeImage": sb.keyframe_image,
                "blocked": False,
                "provider": provider,
                "canMultiRef": can_multi_ref,
                "referenceImages": reference_images,
                "referenceAudioUrls": reference_audio_urls,
                "prevTail": prev_tail,
            })

            # 画风收口：agent 只写画面/运镜内容，画风英文词由后端统一追加（防同剧画风漂移）
            prompt = (f"{sb.video_prompt or sb.image_prompt or sb.description or '镜头缓慢推进，人物自然表演'}"
                      f"{build_video_art_style_suffix(art_style)}")
            # FL2VA（首尾帧连接）：adapter 契约要求 referenceMode='first_last' 才会派发首尾帧。
            # 起帧 = 同场景顺接的真实尾帧（连续性）或本镜设计首帧；尾帧目标 = 本镜设计尾帧。
            is_first_last = route_decision.get("referenceMode") == "first_last"
            await generate_video(conn, {
                "storyboardId": sb.id,
                "dramaId": drama_id,
                "prompt": prompt,
                "negativePrompt": build_video_negative_prompt(art_style),
                "referenceMode": route_decision.get("referenceMode"),
                "imageUrl": None if is_first_last else anchor_image,
                "firstFrameUrl": ((prev_tail or sb.first_frame_image) if is_first_last
                                  else (sb.first_frame_image if prev_tail else None)),
                "lastFrameUrl": sb.last_frame_image or None,
                "referenceImageUrls": reference_images or None,
                "sceneType": scene_type or None,
                "referenceAudioUrls": reference_audio_urls or None,
                "configId": config_id,
                "route": route_decision.get("route"),
                "routeReason": route_decision.get("reason"),
            })
        submitted += 1
    return submitted


async def wait_for_images(episode_id: int, drama_id: int,
                          timeout_ms: int | None = None) -> bool:
    """轮询等待所有分镜首帧图就绪；超时返回 False（**部分完成不阻断**）。"""
    limit = MEDIA_WAIT_TIMEOUT_MS if timeout_ms is None else timeout_ms
    start = _now_ms()
    while _now_ms() - start < limit:
        with engine.begin() as conn:
            arts = get_storyboards(conn, episode_id)
        ready = len([sb for sb in arts if sb.first_frame_image])
        if ready >= len(arts):
            return True
        publish_pipeline_event(drama_id, {
            "type": "media-progress", "episodeId": episode_id,
            "ready": ready, "total": len(arts),
        })
        await _sleep(MEDIA_POLL_INTERVAL_MS / 1000)
    return False


async def wait_for_keyframes(episode_id: int, drama_id: int,
                             timeout_ms: int | None = None) -> bool:
    """等待所有**有关键帧 prompt**的分镜关键帧就绪；无 keyframe_prompt 或已生成即视为就绪。"""
    limit = MEDIA_WAIT_TIMEOUT_MS if timeout_ms is None else timeout_ms
    start = _now_ms()
    while _now_ms() - start < limit:
        with engine.begin() as conn:
            arts = get_storyboards(conn, episode_id)
        needing = [sb for sb in arts if sb.keyframe_prompt]
        if not needing:
            return True
        ready = len([sb for sb in needing if sb.keyframe_image])
        if ready >= len(needing):
            return True
        await _sleep(MEDIA_POLL_INTERVAL_MS / 1000)
    return False


async def wait_for_videos(episode_id: int, drama_id: int,
                          timeout_ms: int | None = None) -> bool:
    """等待所有分镜视频就绪；**被资产门禁阻断的分镜视为已结束**；超时返回 False。"""
    limit = MEDIA_WAIT_TIMEOUT_MS if timeout_ms is None else timeout_ms
    start = _now_ms()
    while _now_ms() - start < limit:
        with engine.begin() as conn:
            arts = get_storyboards(conn, episode_id)
            blocked = {sb.id for sb in arts if is_storyboard_blocked(conn, sb.id)}
        ready = len([sb for sb in arts if sb.video_url])
        settled = len([sb for sb in arts if sb.video_url or sb.id in blocked])
        if settled >= len(arts):
            return True
        publish_pipeline_event(drama_id, {
            "type": "media-progress", "episodeId": episode_id,
            "ready": ready, "total": len(arts),
        })
        await _sleep(MEDIA_POLL_INTERVAL_MS / 1000)
    return False


def _now_ms() -> float:
    """``Date.now()`` 的等价物（墙钟毫秒；只用于计算经过时间，与单调时钟差别可忽略）。"""
    import time

    return time.time() * 1000


async def run_image_stage(episode_id: int, drama_id: int, opts: dict[str, Any]) -> None:
    with engine.begin() as conn:
        episode = _fetch_episode(conn, episode_id)
    config_id = (episode.image_config_id if episode is not None
                 and episode.image_config_id is not None else opts.get("imageConfigId"))
    submitted = await submit_missing_images(episode_id, drama_id, config_id)
    log_task_progress("AutoPipeline", "image-stage", {"episodeId": episode_id,
                                                      "submitted": submitted})
    ok = await wait_for_images(episode_id, drama_id)
    if not ok:
        with engine.begin() as conn:
            pending = len([sb for sb in get_storyboards(conn, episode_id)
                           if not sb.first_frame_image])
        log_task_warn("AutoPipeline", "image-timeout", {"episodeId": episode_id,
                                                        "pendingCount": pending})
    # 关键帧扩展：首帧就绪后再补齐中段关键帧（不影响首帧等待完成度，失败不阻断）
    if ok:
        key_submitted = await submit_missing_keyframes(episode_id, drama_id, config_id)
        if key_submitted > 0:
            log_task_progress("AutoPipeline", "image-keyframes", {
                "episodeId": episode_id, "submitted": key_submitted,
            })
            await wait_for_keyframes(episode_id, drama_id, MEDIA_WAIT_TIMEOUT_MS)


async def run_video_stage(episode_id: int, drama_id: int, opts: dict[str, Any]) -> None:
    with engine.begin() as conn:
        episode = _fetch_episode(conn, episode_id)
    config_id = (episode.video_config_id if episode is not None
                 and episode.video_config_id is not None else opts.get("videoConfigId"))
    submitted = await submit_missing_videos(episode_id, drama_id, config_id)
    log_task_progress("AutoPipeline", "video-stage", {"episodeId": episode_id,
                                                      "submitted": submitted})
    ok = await wait_for_videos(episode_id, drama_id)
    # 尾帧衔接：从已完成视频提取**真实末帧**写入 tail_frame_image（供下一镜顺接与重跑衔接），
    # **不写** last_frame_image —— 设计尾帧是 FL2VA 的决策依据，不能被产物覆盖
    await extract_storyboard_tail_frames(episode_id, drama_id)
    if not ok:
        with engine.begin() as conn:
            pending = len([sb for sb in get_storyboards(conn, episode_id)
                           if not sb.video_url and not is_storyboard_blocked(conn, sb.id)])
        log_task_warn("AutoPipeline", "video-timeout", {"episodeId": episode_id,
                                                        "pendingCount": pending})


async def run_compose_stage(episode_id: int) -> None:
    with engine.begin() as conn:
        arts = get_storyboards(conn, episode_id)
    for sb in arts:
        if sb.composed_video_url:
            continue
        if not sb.video_url:
            continue
        await compose_storyboard(sb.id)


async def run_merge_stage(episode_id: int, drama_id: int) -> None:
    with engine.begin() as conn:
        episode = _fetch_episode(conn, episode_id)
        if episode is not None and episode.video_url:
            return  # 已拼接
        completed = conn.execute(
            select(video_merges.c.id).where(and_(
                video_merges.c.episode_id == episode_id,
                video_merges.c.status == "completed",
            ))
        ).all()
        if completed:
            return
        arts = get_storyboards(conn, episode_id)
    ready = [sb for sb in arts if sb.composed_video_url]
    if not arts or len(ready) != len(arts):
        log_task_warn("AutoPipeline", "merge-skip-not-ready", {
            "episodeId": episode_id, "ready": len(ready), "total": len(arts),
        })
        return
    merge_episode_videos(episode_id, drama_id)


# ============================================================
# 单集管线（幂等）
# ============================================================

def build_stages(episode_id: int, drama_id: int,
                 opts: dict[str, Any]) -> list[dict[str, Any]]:
    """按媒体开关构造阶段表（``status`` = 进入该阶段写入的状态，``next`` = 完成后的状态）。"""
    needs = media_needs(opts)
    done = AUTO_STATUS["done"]
    stages: list[dict[str, Any]] = [
        {"key": "script", "status": AUTO_STATUS["scripting"],
         "next": AUTO_STATUS["extracting"],
         "run": lambda: run_script_stage(episode_id, drama_id)},
        {"key": "extract", "status": AUTO_STATUS["extracting"],
         "next": AUTO_STATUS["voicing"],
         "run": lambda: run_extract_stage(episode_id, drama_id)},
        {"key": "voice", "status": AUTO_STATUS["voicing"],
         "next": AUTO_STATUS["storyboarding"],
         "run": lambda: run_voice_stage(episode_id, drama_id)},
        {"key": "storyboard", "status": AUTO_STATUS["storyboarding"],
         "next": AUTO_STATUS["imaging"] if needs["needImage"] else done,
         "run": lambda: run_storyboard_stage(episode_id, drama_id)},
    ]
    if needs["needImage"]:
        stages.append({
            "key": "image", "status": AUTO_STATUS["imaging"],
            "next": AUTO_STATUS["videoing"] if needs["needVideo"] else done,
            "run": lambda: run_image_stage(episode_id, drama_id, opts),
        })
    if needs["needVideo"]:
        stages.append({
            "key": "video", "status": AUTO_STATUS["videoing"],
            "next": AUTO_STATUS["composing"] if needs["needCompose"] else done,
            "run": lambda: run_video_stage(episode_id, drama_id, opts),
        })
    if needs["needCompose"]:
        stages.append({
            "key": "compose", "status": AUTO_STATUS["composing"],
            "next": AUTO_STATUS["merging"] if needs["needMerge"] else done,
            "run": lambda: run_compose_stage(episode_id),
        })
    if needs["needMerge"]:
        stages.append({
            "key": "merge", "status": AUTO_STATUS["merging"], "next": done,
            "run": lambda: run_merge_stage(episode_id, drama_id),
        })
    return stages


async def execute_episode_pipeline(episode_id: int, drama_id: int,
                                   opts: dict[str, Any]) -> None:
    """执行单集管线（幂等，可反复调用自动续跑）。**忙时入队而非静默丢弃**。"""
    if episode_id in _running_episodes:
        _queued_episodes.add(episode_id)
        return
    _running_episodes.add(episode_id)
    try:
        while True:
            await _execute_episode_pipeline_inner(episode_id, drama_id, opts)
            if episode_id not in _queued_episodes:
                break
            _queued_episodes.discard(episode_id)
            log_task_progress("AutoPipeline", "follow-up-requeue",
                              {"episodeId": episode_id})
    finally:
        _running_episodes.discard(episode_id)


async def _execute_episode_pipeline_inner(episode_id: int, drama_id: int,
                                          opts: dict[str, Any]) -> None:
    with engine.begin() as conn:
        if is_drama_deleted(conn, drama_id):
            log_task_warn("AutoPipeline", "drama-deleted-skip",
                          {"episodeId": episode_id, "dramaId": drama_id})
            return
        episode = _fetch_episode(conn, episode_id)
    if episode is None:
        raise ValueError(f"Episode {episode_id} not found")
    if episode.status == AUTO_STATUS["done"]:
        return

    stages = build_stages(episode_id, drama_id, opts)
    current_status = episode.status

    log_task_start("AutoPipeline", "episode", {
        "episodeId": episode_id, "dramaId": drama_id, "fromStatus": current_status,
    })

    for stage in stages:
        if is_stage_completed(current_status, stage["next"]):
            continue
        set_status(episode_id, stage["status"])
        current_status = stage["status"]
        log_task_progress("AutoPipeline", f"stage-{stage['key']}", {"episodeId": episode_id})
        try:
            await stage["run"]()
            set_status(episode_id, stage["next"])
            current_status = stage["next"]
            log_task_progress("AutoPipeline", f"stage-{stage['key']}-done",
                              {"episodeId": episode_id})
        except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价：标记失败再抛
            set_status(episode_id, AUTO_STATUS["failed"])
            log_task_error("AutoPipeline", f"stage-{stage['key']}-failed", {
                "episodeId": episode_id, "error": str(err),
            })
            raise

    set_status(episode_id, AUTO_STATUS["done"])
    log_task_success("AutoPipeline", "episode-done", {
        "episodeId": episode_id, "dramaId": drama_id,
    })


# ============================================================
# 整剧执行主体
# ============================================================

def load_pipeline_options(drama_id: int) -> dict[str, Any] | None:
    """从 ``dramas.metadata`` 读回管线选项（解析失败 → None）。"""
    with engine.begin() as conn:
        row = conn.execute(select(dramas.c.metadata).where(
            dramas.c.id == drama_id)).first()
    if row is None or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return None


def save_pipeline_options(drama_id: int, opts: dict[str, Any]) -> None:
    """写回管线选项（⚠️ **紧凑 JSON**，与 ``JSON.stringify`` 对齐）。"""
    with engine.begin() as conn:
        conn.execute(update(dramas).where(dramas.c.id == drama_id).values(
            metadata=json.dumps(opts, ensure_ascii=False, separators=(",", ":")),
            updated_at=now(),
        ))


async def execute_pipeline(drama_id: int, opts: dict[str, Any]) -> None:
    """后台**串行**执行整剧各集（单集失败标记 auto:failed，**不阻断后续集**）。"""
    with engine.begin() as conn:
        rows = conn.execute(
            select(episodes.c.id).where(episodes.c.drama_id == drama_id)
            .order_by(asc(episodes.c.episode_number))
        ).all()
    episode_ids = [row[0] for row in rows]

    log_task_start("AutoPipeline", "drama", {"dramaId": drama_id,
                                            "episodeCount": len(episode_ids)})
    for episode_id in episode_ids:
        try:
            await execute_episode_pipeline(episode_id, drama_id, opts)
        except Exception as err:  # noqa: BLE001 —— 失败隔离：不阻断后续集
            log_task_error("AutoPipeline", "episode-failed", {
                "episodeId": episode_id, "dramaId": drama_id, "error": str(err),
            })
    log_task_success("AutoPipeline", "drama-done", {"dramaId": drama_id})


# ============================================================
# 对外入口
# ============================================================

def _spawn(coro: Coroutine[Any, Any, Any], action: str,
           meta: dict[str, Any], success_suffix: str = "complete") -> None:
    """fire-and-forget 后台执行（成功/失败各记一条日志）。

    ⚠️ 成功日志后缀**按入口不同**：run/resume 是 ``-complete``，recover 是 ``-done``（原 TS 如此）。

    ⚠️ 用 ``get_running_loop()`` 而**不是** ``ensure_future()``：``ensure_future`` 走
    ``get_event_loop()``，在 ASGI 服务器的工作线程里那个线程未必注册过事件循环
    （anyio 起的循环不保证 ``set_event_loop``）⇒ 会抛
    ``There is no current event loop in thread 'AnyIO worker thread'``。
    本函数总是被「循环线程上的同步代码」调用（async 路由处理器内部 / lifespan），
    ``get_running_loop()`` 在这种场景下必定可用。
    """
    task = asyncio.get_running_loop().create_task(coro)

    def _done(future: "asyncio.Future[Any]") -> None:
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            log_task_error("AutoPipeline", f"{action}-failed", {**meta, "error": str(error)})
        else:
            log_task_success("AutoPipeline", f"{action}-{success_suffix}", meta)

    task.add_done_callback(_done)


def run_auto_pipeline(opts: dict[str, Any]) -> dict[str, Any]:
    """创建 Drama + Episodes 并触发后台全自动管线，**立即返回** ``{dramaId, episodeIds}``。"""
    normalized = normalize_options(opts)
    if not (normalized.get("premise") or "").strip():
        raise ValueError("premise 不能为空")

    ts = now()
    premise = normalized["premise"].strip()
    episode_count = normalized["episodeCount"]

    # ⚠️ 自己开短事务并**立即提交**：后台管线任务用的是另一个连接，看不到未提交的数据
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title=normalized["title"], description=premise,
            genre=normalized["genre"], style=normalized["style"],
            total_episodes=episode_count, status="auto_generating",
            metadata=json.dumps(normalized, ensure_ascii=False, separators=(",", ":")),
            created_at=ts, updated_at=ts,
        )).lastrowid)

        episode_ids: list[int] = []
        for index in range(episode_count):
            number = index + 1
            content = f"【第{number}集】{premise}" if episode_count > 1 else premise
            episode_ids.append(int(conn.execute(episodes.insert().values(
                drama_id=drama_id,
                episode_number=number,
                title=f"{normalized['title']} 第{number}集",
                content=content,
                status=AUTO_STATUS["queued"],
                image_config_id=normalized.get("imageConfigId"),
                video_config_id=normalized.get("videoConfigId"),
                audio_config_id=normalized.get("audioConfigId"),
                created_at=ts,
                updated_at=ts,
            )).lastrowid))

    _spawn(execute_pipeline(drama_id, normalized), "run", {"dramaId": drama_id})
    return {"dramaId": drama_id, "episodeIds": episode_ids}


def resume_auto_pipeline(drama_id: int, override: dict[str, Any] | None = None) -> None:
    """续跑（幂等）：用于崩溃恢复、手动重跑、**媒体补跑**。"""
    base = load_pipeline_options(drama_id)
    if not base:
        raise ValueError(f"Drama {drama_id} 缺少 pipeline 配置（metadata）")
    opts = normalize_options({**base, **(override or {})})
    save_pipeline_options(drama_id, opts)

    with engine.begin() as conn:
        rows = conn.execute(
            select(episodes.c.id, episodes.c.status)
            .where(episodes.c.drama_id == drama_id)
        ).all()
    for episode_id, status in rows:
        if status == AUTO_STATUS["done"]:
            missing = media_missing_status(episode_id, opts)
            if missing:
                set_status(episode_id, missing)
        elif status == AUTO_STATUS["failed"]:
            set_status(episode_id, AUTO_STATUS["queued"])  # 从失败恢复：从头幂等重跑

    _spawn(execute_pipeline(drama_id, opts), "resume", {"dramaId": drama_id})


def media_missing_status(episode_id: int, opts: dict[str, Any]) -> str | None:
    """``done`` 状态下，若媒体产物缺失，**精确降级**到缺失的媒体阶段（避免重跑 Agent 链）。"""
    needs = media_needs(opts)
    if not needs["needImage"]:
        return None
    with engine.begin() as conn:
        arts = get_storyboards(conn, episode_id)
        if not arts:
            return None
        blocked = {sb.id for sb in arts if is_storyboard_blocked(conn, sb.id)}
        if needs["needImage"] and any(
                not sb.first_frame_image and sb.id not in blocked for sb in arts):
            return AUTO_STATUS["imaging"]
        if needs["needVideo"] and any(
                not sb.video_url and sb.id not in blocked for sb in arts):
            return AUTO_STATUS["videoing"]
        if needs["needCompose"] and any(not sb.composed_video_url for sb in arts):
            return AUTO_STATUS["composing"]
        if needs["needMerge"]:
            completed = conn.execute(
                select(video_merges.c.id).where(and_(
                    video_merges.c.episode_id == episode_id,
                    video_merges.c.status == "completed",
                ))
            ).all()
            if not completed:
                return AUTO_STATUS["merging"]
    return None


def get_auto_pipeline_status(drama_id: int) -> dict[str, Any] | None:
    """查询整剧管线进度（drama 不存在 → ``None`` ⇒ 路由回 404）。"""
    with engine.begin() as conn:
        drama = conn.execute(select(dramas).where(dramas.c.id == drama_id)).first()
        if drama is None:
            return None
        episodes_rows = conn.execute(
            select(episodes).where(episodes.c.drama_id == drama_id)
            .order_by(asc(episodes.c.episode_number))
        ).all()

        details: list[dict[str, Any]] = []
        for episode in episodes_rows:
            arts = get_storyboards(conn, episode.id)
            characters = conn.execute(
                select(episode_characters.c.character_id)
                .where(episode_characters.c.episode_id == episode.id)
            ).all()
            scenes = conn.execute(
                select(episode_scenes.c.scene_id)
                .where(episode_scenes.c.episode_id == episode.id)
            ).all()
            details.append({
                "id": episode.id,
                "episodeNumber": episode.episode_number,
                "status": episode.status,
                "hasScript": bool((episode.script_content or "").strip()),
                "hasVideo": bool(episode.video_url),
                "storyboardCount": len(arts),
                "characterCount": len(characters),
                "sceneCount": len(scenes),
                "imageReadyCount": len([sb for sb in arts if sb.first_frame_image]),
                "videoReadyCount": len([sb for sb in arts if sb.video_url]),
                "videoBlockedCount": len([sb for sb in arts
                                          if is_storyboard_blocked(conn, sb.id)]),
                "composedCount": len([sb for sb in arts if sb.composed_video_url]),
            })

    done_count = len([item for item in details if item["status"] == AUTO_STATUS["done"]])
    failed_count = len([item for item in details if item["status"] == AUTO_STATUS["failed"]])

    return {
        "dramaId": drama_id,
        "title": drama.title,
        "status": drama.status,
        "totalEpisodes": len(episodes_rows),
        "doneCount": done_count,
        "failedCount": failed_count,
        "running": done_count + failed_count < len(episodes_rows),
        "episodes": details,
    }


def recover_auto_pipeline_on_startup() -> None:
    """启动**崩溃恢复**：扫描中间态 episode 自动续跑（幂等）。"""
    with engine.begin() as conn:
        rows = conn.execute(
            select(episodes.c.id, episodes.c.drama_id)
            .where(episodes.c.status.in_(IN_FLIGHT_STATUSES))
        ).all()
    if not rows:
        return

    # 按 drama 分组，每个 drama 只触发一次 execute_pipeline（内部逐集幂等）
    drama_ids: list[int] = []
    for _episode_id, drama_id in rows:
        if drama_id not in drama_ids:
            drama_ids.append(drama_id)
    log_task_start("AutoPipeline", "recover", {
        "episodeCount": len(rows), "dramaCount": len(drama_ids),
    })

    for drama_id in drama_ids:
        opts = load_pipeline_options(drama_id)
        if not opts:
            log_task_warn("AutoPipeline", "recover-no-config", {"dramaId": drama_id})
            continue
        _spawn(execute_pipeline(drama_id, opts), "recover",
               {"dramaId": drama_id}, success_suffix="done")
