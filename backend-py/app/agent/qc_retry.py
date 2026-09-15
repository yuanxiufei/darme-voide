"""审片重跑闭环 —— 移植 ``services/qc-retry.ts``（175 行）。

镜头 QC 审片未通过时**只重写该失败分镜**，不重建整集：

1. 组装 QC ``issues`` → 交给 ``storyboard_breaker``，**只允许调 ``update_storyboard``** 修这一镜；
2. **软删**该镜旧的图片/视频生成记录（防止 QC 复用旧产物）；
3. 重新提交首帧图 → **轮询等新首帧就绪**（5 分钟上限 / 3 秒一轮）→ 重新提交视频（fire-and-forget）；
4. 视频完成时由 webhook 触发 QC 重新打分 ⇒ 闭环。

⚠️ 五处保真点：

1. **LLM 提示词逐字照抄**（含「禁止 save_storyboards 重建整集」与四条修复要求）；
2. **首帧等待是「值变了才算就绪」**（``firstFrameImage !== oldFirstFrame``）——若重新生成恰好得到同一 URL，
   会一直等到超时（这是原实现的判据，照抄）；
3. **参考图/参考音频的开关条件不同**：参考图要求 ``isDialogue && provider ∈ {volcengine,vidu,minimax}``
   （最多 9 张）；参考音频要求 ``isDialogue && provider == 'minimax'``；
4. **FL2VA 判定用「设计首帧 + **last_frame_image**」**：``tail_frame_image``（真实尾帧）**不参与**决策
   —— 它是可运行产物，不代表设计目标；
5. ``rewrittenFields`` 输出的是 **camelCase 字段名**（``firstFramePrompt`` 等），而库里是 snake_case。

⚠️ **首帧轮询写在请求路径里**（原 TS 就这样：端点 await 整个流程）⇒ 5 分钟上限是**同步阻塞**的。
自检里把 ``FIRST_FRAME_WAIT_TIMEOUT_MS`` 调 0 即可快速走通。
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from sqlalchemy import and_, select, update
from sqlalchemy.engine import Connection

from app.core.models import episodes, image_generations, storyboards, video_generations, video_quality_checks
from app.core.response import now
from app.agent.runtime import run_agent_with_retry
from app.services.ai_providers import get_active_config, get_config_by_id
from app.services.image_generation import generate_image
from app.services.prompt_utils import (
    build_storyboard_negative_prompt,
    build_video_negative_prompt,
    get_storyboard_reference_audio_urls,
    get_storyboard_reference_images,
    resolve_effective_art_style,
)
from app.services.task_logger import log_task_progress, log_task_success, log_task_warn
from app.services.video_generation import generate_video

__all__ = ["retry_failed_storyboard"]

#: 支持多参考图的视频提供商；其余仅首尾帧
MULTI_REFERENCE_PROVIDERS = frozenset({"volcengine", "vidu", "minimax"})

FIRST_FRAME_WAIT_TIMEOUT_MS = 5 * 60 * 1000
FIRST_FRAME_POLL_INTERVAL_MS = 3000

#: ``rewrittenFields`` 输出用的 camelCase 名（库列 → 对外字段名）
_REWRITE_FIELDS: tuple[tuple[str, str], ...] = (
    ("first_frame_prompt", "firstFramePrompt"),
    ("last_frame_prompt", "lastFramePrompt"),
    ("video_prompt", "videoPrompt"),
    ("image_prompt", "imagePrompt"),
    ("negative_prompt", "negativePrompt"),
    ("constraints", "constraints"),
    ("start_state", "startState"),
    ("end_state", "endState"),
)

#: 对白/多人场景的类型判定（原 TS 的裸正则）
_DIALOGUE_RE = re.compile(r"dialogue|meeting|argument|conversation|multi", re.IGNORECASE)


def _parse_issues(raw: Any) -> list[dict[str, Any]]:
    """``qc.issues``（JSON 文本）→ 列表；坏 JSON 回退空列表。"""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


async def _wait_for_new_first_frame(conn: Connection, storyboard_id: Any,
                                    old_first_frame: Any) -> bool:
    """轮询等新首帧就绪：**值变了才算**（超时返回 False）。"""
    started = time.monotonic() * 1000
    while time.monotonic() * 1000 - started < FIRST_FRAME_WAIT_TIMEOUT_MS:
        row = conn.execute(
            select(storyboards.c.first_frame_image)
            .where(storyboards.c.id == storyboard_id)
        ).first()
        if row is not None and row[0] and row[0] != old_first_frame:
            return True
        await asyncio.sleep(FIRST_FRAME_POLL_INTERVAL_MS / 1000)
    return False


async def retry_failed_storyboard(conn: Connection, storyboard_id: Any) -> dict[str, Any]:
    """对一个审片未通过的镜头执行「改写 → 重生成」闭环。"""
    sb = conn.execute(
        select(storyboards).where(and_(storyboards.c.id == storyboard_id,
                                       storyboards.c.deleted_at.is_(None)))
    ).first()
    if sb is None:
        return {"ok": False, "message": "Storyboard not found", "storyboardId": storyboard_id}
    episode = conn.execute(
        select(episodes).where(episodes.c.id == sb.episode_id)).first()
    if episode is None:
        return {"ok": False, "message": "Episode not found", "storyboardId": storyboard_id}

    qcs = sorted(
        conn.execute(select(video_quality_checks).where(
            video_quality_checks.c.storyboard_id == storyboard_id)).all(),
        key=lambda r: r.id, reverse=True,
    )
    issues = _parse_issues(qcs[0].issues if qcs else None)

    # ===== 1. LLM 只重写该失败分镜 =====
    issues_text = ("\n".join(f"[{i.get('severity')}] {i.get('dimension')}: {i.get('message')}"
                            for i in issues) if issues
                   else "（无具体问题描述，请复核画面与对白质量后自行修复明显缺陷）")
    message = (
        f"本镜 QC 审片未通过，需要修复。分镜 ID={storyboard_id}"
        f"（镜头 #{sb.storyboard_number}「{sb.title or ''}」）。\n"
        f"审片问题：\n{issues_text}\n\n"
        "修复要求：\n"
        f"1. 只调用 update_storyboard 工具更新该分镜（storyboard_id={storyboard_id}），"
        "可修改 first_frame_prompt / last_frame_prompt / video_prompt / image_prompt / "
        "negative_prompt / constraints / start_state / end_state / action / description 等字段\n"
        "2. 禁止调用 save_storyboards 重建整集，禁止修改其他分镜\n"
        "3. 针对性修复：画面重复则改构图、首尾帧雷同则强化差异、状态跳变则修正 "
        "start_state/end_state、负面元素则强化 negative_prompt\n"
        "4. 修复完成后报告修改了哪些字段"
    )

    log_task_progress("QcRetry", "rewrite-begin", {
        "storyboardId": storyboard_id, "episodeId": sb.episode_id,
        "issueCount": len(issues),
    })
    agent_result = await run_agent_with_retry(
        conn, "storyboard_breaker", sb.episode_id, episode.drama_id, message, {"maxSteps": 20}
    )

    updated = conn.execute(
        select(storyboards).where(storyboards.c.id == storyboard_id)).first()
    if updated is None:
        return {"ok": False, "message": "Storyboard disappeared during retry",
                "storyboardId": storyboard_id}

    # 报告实际被修改的字段（对外用 camelCase）
    rewritten_fields = [
        public for column, public in _REWRITE_FIELDS
        if getattr(updated, column) != getattr(sb, column)
    ]

    # ===== 2. 软删旧图片/视频生成记录（防止 QC 复用旧产物）=====
    soft_delete_ts = now()
    conn.execute(
        update(image_generations)
        .where(image_generations.c.storyboard_id == storyboard_id)
        .values(deleted_at=soft_delete_ts, updated_at=soft_delete_ts)
    )
    conn.execute(
        update(video_generations)
        .where(video_generations.c.storyboard_id == storyboard_id)
        .values(deleted_at=soft_delete_ts, updated_at=soft_delete_ts)
    )

    # ===== 3. 重新提交首帧图并等待就绪 =====
    # 画风收口：重跑也要与首次生成同一套画风（正/负词同源），否则 QC 重试会变成「换风格重画」
    drama_style = resolve_effective_art_style(conn, episode.drama_id)
    old_first_frame = updated.first_frame_image
    prompt = (updated.first_frame_prompt or updated.image_prompt or updated.description
              or "短剧分镜画面，电影感构图")
    ref_images = get_storyboard_reference_images(conn, storyboard_id)
    image_params: dict[str, Any] = {
        "storyboardId": storyboard_id,
        "dramaId": episode.drama_id,
        "prompt": prompt,
        "negativePrompt": (updated.negative_prompt
                           or build_storyboard_negative_prompt(drama_style)),
        "frameType": "first_frame",
        "configId": episode.image_config_id,
    }
    if ref_images:
        image_params["referenceImages"] = ref_images
    image_id = await generate_image(conn, image_params)

    if not await _wait_for_new_first_frame(conn, storyboard_id, old_first_frame):
        log_task_warn("QcRetry", "first-frame-timeout",
                      {"storyboardId": storyboard_id, "imageGenerationId": image_id})

    # ===== 4. 重新提交视频（fire-and-forget，完成后 webhook 自动重新 QC）=====
    final_sb = conn.execute(
        select(storyboards).where(storyboards.c.id == storyboard_id)).first()
    scene_type = (final_sb.scene_type if final_sb is not None else None) or ""
    is_dialogue = bool(_DIALOGUE_RE.search(str(scene_type)))
    config = (get_config_by_id(conn, episode.video_config_id) if episode.video_config_id
              else get_active_config(conn, "video"))
    provider = str((config or {}).get("provider") or "").lower()
    can_multi_ref = provider in MULTI_REFERENCE_PROVIDERS
    reference_images = (get_storyboard_reference_images(conn, storyboard_id)[:9]
                        if is_dialogue and can_multi_ref else [])
    reference_audio_urls = (get_storyboard_reference_audio_urls(conn, storyboard_id)
                            if is_dialogue and provider == "minimax" else [])

    video_prompt = (final_sb.video_prompt if final_sb is not None else None) or \
        (final_sb.image_prompt if final_sb is not None else None) or \
        (final_sb.description if final_sb is not None else None) or \
        "镜头缓慢推进，人物自然表演"
    # FL2VA（首尾帧连接）重跑：**设计**首帧与**设计**尾帧必须同时存在才能走 first_last
    # （adapter 缺任一端会发坏请求）。真实尾帧 tail_frame_image 不参与决策。
    first_frame = final_sb.first_frame_image if final_sb is not None else None
    last_frame = final_sb.last_frame_image if final_sb is not None else None
    is_fl2va_retry = bool(last_frame) and bool(first_frame)

    video_params: dict[str, Any] = {
        "storyboardId": storyboard_id,
        "dramaId": episode.drama_id,
        "prompt": video_prompt,
        "negativePrompt": build_video_negative_prompt(drama_style),
        "referenceMode": ("first_last" if is_fl2va_retry
                          else ("multiple" if reference_images else "single")),
        "configId": episode.video_config_id,
    }
    if is_fl2va_retry:
        video_params["firstFrameUrl"] = first_frame
        video_params["lastFrameUrl"] = last_frame
    elif first_frame:
        video_params["imageUrl"] = first_frame
    if reference_images:
        video_params["referenceImageUrls"] = reference_images
    if scene_type:
        video_params["sceneType"] = scene_type
    if reference_audio_urls:
        video_params["referenceAudioUrls"] = reference_audio_urls
    video_id = await generate_video(conn, video_params)

    log_task_success("QcRetry", "retry-complete", {
        "storyboardId": storyboard_id,
        "rewrittenFields": ",".join(rewritten_fields),
        "imageGenerationId": image_id,
        "videoGenerationId": video_id,
        "agentToolCalls": len(getattr(agent_result, "tool_calls", None) or []),
    })

    fields_text = "、".join(rewritten_fields) if rewritten_fields else "无"
    return {
        "ok": True,
        "message": f"已重写并重新生成镜头 #{sb.storyboard_number}（修改字段：{fields_text}），"
                   f"新首帧图任务 {image_id}、新视频任务 {video_id} 已提交，完成后将自动重新审片",
        "storyboardId": storyboard_id,
        "storyboardNumber": sb.storyboard_number,
        "rewrittenFields": rewritten_fields,
        "imageGenerationId": image_id,
        "videoGenerationId": video_id,
    }
