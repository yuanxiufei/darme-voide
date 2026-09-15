"""生成前费用预估 —— 移植 ``services/estimate-service.ts``（整文件）。

对齐 ArcReel 的「生成前预估 + 生成后核算」双环：发起批量生成前（整剧/整集/指定分镜），
按当前激活 provider/model 的单价 × 待生成工作量（图片张数 / 视频秒数 / TTS 字符数）预估总费用，
回答「这一批要烧多少钱」。与 ``usage_tracking``（生成后实际记账）配套：**估算是花前，记算是花后**。

⚠️ 与 ``ai_providers.get_active_config`` 的**排序口径不同**，别合并：

* 这里：``is_default`` 优先，**再看 priority 倒序**（``(b.isDefault?1:0)-(a.isDefault?1:0) || (b.priority??0)-(a.priority??0)``）
* ``ai.ts``：只看 priority 倒序

⚠️ 另一个照抄的怪点：这里的 ``model`` 取的是 **``model`` 列的原始 JSON 字符串**
（``cfg.model || ''``，没 ``JSON.parse``），而 ``estimateCost`` 用的是**包含匹配** ⇒
``'["doubao-…"]'`` 里照样能命中 ``doubao`` 前缀。能跑通，但不是"正常"取值方式，别顺手改写。
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..core.models import ai_service_configs, episodes, storyboards, video_generations
from ..core.response import js_round, now
from .cost_catalog import estimate_cost
from .task_logger import log_task_start, log_task_success

_TRUTHY_IS_DEFAULT = 1


def _active_config(conn: Connection, service_type: str) -> dict[str, Any] | None:
    """取该服务类型优先级最高的**激活**配置（is_default 优先，再 priority 倒序）。"""
    rows = [
        row
        for row in conn.execute(
            select(ai_service_configs).where(
                (ai_service_configs.c.service_type == service_type)
                & (ai_service_configs.c.is_active.is_(True))
            )
        ).all()
    ]
    if not rows:
        return None
    # JS 的 `||` 短路：is_default 有差异时按它排，相同才比 priority
    rows.sort(
        key=lambda r: (
            (_TRUTHY_IS_DEFAULT if r.is_default else 0),
            r.priority if r.priority is not None else 0,
        ),
        reverse=True,
    )
    cfg = rows[0]
    settings = None
    if cfg.settings:
        try:
            parsed = json.loads(cfg.settings)
            settings = parsed if isinstance(parsed, dict) else None
        except (ValueError, TypeError):
            settings = None
    return {"provider": cfg.provider or "", "model": cfg.model or "", "settings": settings}


def _has_completed_video(conn: Connection, storyboard_id: int) -> bool:
    return (
        conn.execute(
            select(video_generations.c.id).where(
                (video_generations.c.storyboard_id == storyboard_id)
                & (video_generations.c.status == "completed")
            )
        ).first()
        is not None
    )


def _select_storyboards(
    conn: Connection, drama_id: int, storyboard_ids: list[int] | None, episode_id: int | None
) -> list[Any]:
    columns = (storyboards.c.id, storyboards.c.duration, storyboards.c.dialogue)
    if storyboard_ids:
        return conn.execute(
            select(*columns).where(storyboards.c.id.in_(storyboard_ids))
        ).all()
    if episode_id:
        return conn.execute(
            select(*columns).where(storyboards.c.episode_id == episode_id)
        ).all()
    episode_ids = [
        row.id
        for row in conn.execute(
            select(episodes.c.id).where(episodes.c.drama_id == drama_id)
        ).all()
    ]
    if not episode_ids:
        return []
    return conn.execute(
        select(*columns).where(storyboards.c.episode_id.in_(episode_ids))
    ).all()


def estimate_pending_costs(
    conn: Connection,
    drama_id: int,
    *,
    scope: str | None = None,
    episode_id: int | None = None,
    storyboard_ids: list[int] | None = None,
) -> dict[str, Any]:
    """预估一批待生成分镜的费用。"""
    log_task_start("Estimate", "estimate", {"dramaId": drama_id, "episodeId": episode_id})

    rows = _select_storyboards(conn, drama_id, storyboard_ids, episode_id)
    # 待生成：显式指定分镜时全部计入；否则只统计「没有已完成视频」的分镜
    pending = (
        rows if storyboard_ids else [sb for sb in rows if not _has_completed_video(conn, sb.id)]
    )

    items: list[dict[str, Any]] = []
    unestimatable: list[str] = []
    total = 0.0
    has_cost = False

    # 视频（按秒；未设时长按 5 秒估）
    video_cfg = _active_config(conn, "video")
    video_seconds = sum((sb.duration if sb.duration is not None else 5) for sb in pending)
    if video_cfg:
        cost = estimate_cost(
            "video", video_cfg["provider"], video_cfg["model"], video_seconds, video_cfg["settings"]
        )
        if cost is None:
            unestimatable.append(f"video({video_cfg['provider']}/{video_cfg['model']}) 无单价")
        else:
            total += cost
            has_cost = True
        items.append(
            {
                "service_type": "video",
                "provider": video_cfg["provider"],
                "model": video_cfg["model"],
                "units": video_seconds,
                "unit": "second",
                "cost": cost,
            }
        )
    else:
        unestimatable.append("video 未配置激活服务")

    # 图片（按张 = 待生成分镜数）
    image_cfg = _active_config(conn, "image")
    pending_images = len(pending)
    if image_cfg:
        cost = estimate_cost(
            "image", image_cfg["provider"], image_cfg["model"], pending_images, image_cfg["settings"]
        )
        if cost is None:
            unestimatable.append(f"image({image_cfg['provider']}/{image_cfg['model']}) 无单价")
        else:
            total += cost
            has_cost = True
        items.append(
            {
                "service_type": "image",
                "provider": image_cfg["provider"],
                "model": image_cfg["model"],
                "units": pending_images,
                "unit": "image",
                "cost": cost,
            }
        )
    else:
        unestimatable.append("image 未配置激活服务")

    # 音频 TTS（按字；无对白则不预估）
    audio_cfg = _active_config(conn, "audio")
    audio_chars = sum(len(sb.dialogue) if sb.dialogue else 0 for sb in pending)
    if audio_cfg and audio_chars > 0:
        cost = estimate_cost(
            "audio", audio_cfg["provider"], audio_cfg["model"], audio_chars, audio_cfg["settings"]
        )
        if cost is None:
            unestimatable.append(f"audio({audio_cfg['provider']}/{audio_cfg['model']}) 无单价")
        else:
            total += cost
            has_cost = True
        items.append(
            {
                "service_type": "audio",
                "provider": audio_cfg["provider"],
                "model": audio_cfg["model"],
                "units": audio_chars,
                "unit": "char",
                "cost": cost,
            }
        )
    elif audio_chars == 0:
        pass  # 无对白则不预估音频
    else:
        unestimatable.append("audio 未配置激活服务")

    log_task_success(
        "Estimate",
        "estimated",
        {
            "dramaId": drama_id,
            "episodeId": episode_id,
            "pendingVideos": len(pending),
            "videoSeconds": video_seconds,
            "audioChars": audio_chars,
            "totalCost": total if has_cost else None,
        },
    )

    return {
        "drama_id": drama_id,
        "episode_id": episode_id,
        "scope": scope or (f"episode-{episode_id}" if episode_id else f"drama-{drama_id}"),
        "generated_at": now(),
        "pending_images": pending_images,
        "pending_videos": len(pending),
        "pending_audio_chars": audio_chars,
        "active_image": (
            {"provider": image_cfg["provider"], "model": image_cfg["model"]} if image_cfg else None
        ),
        "active_video": (
            {"provider": video_cfg["provider"], "model": video_cfg["model"]} if video_cfg else None
        ),
        "active_audio": (
            {"provider": audio_cfg["provider"], "model": audio_cfg["model"]} if audio_cfg else None
        ),
        "items": items,
        # `Math.round(total*100)/100` ⇒ 两位小数
        "total_cost": js_round(total * 100) / 100 if has_cost else None,
        "unestimatable": unestimatable,
    }
