"""逐镜路由决策（Shot Routing，移植自 ``backend/src/services/shot-router.ts``，217 行）。

对齐参考项目 H3-Codex-Drama「逐镜路由」：每个分镜在提交视频生成前**显式**决策采用哪条
生成路线，并把结果与原因写入 ``storyboards.route`` / ``route_reason``（同时快照进
``video_generations``），供可复现账本追溯。

路由类型：

* ``text_to_video``        (T2V)：纯文本提示词，无任何参考帧
* ``first_frame_to_video`` (I2V)：以首帧图起帧（本系统默认路线）
* ``first_last_frame``     (FL2VA)：首帧 + 尾帧连接动画化（锁定结束画面）
* ``reference_to_video``   (R2V)：多参考图（角色立绘/场景图/关键帧）+ 可选参考音频（H3 Ref2VA）
* ``keyframe_to_video``    (I2V-K)：以中段关键帧图为主要参考
* ``video_editor``         (Editor)：基于已有视频的精确修改（本系统暂不自动触发，**保留类型**）
* ``blocked``              资产门禁阻断（无首帧且 ``needs_regeneration``），不提交生成

⚠️ ``reason`` 是**用户可见**文案（前端展示 + 落库），逐字对齐，改字即漂移。

⚠️ 优先级里有一个**反直觉但关键**的顺序：``prevTail``（同场景顺接的上一镜尾帧）必须在
``text_to_video`` **之前**消费 —— 否则 ``referenceMode`` 落到 ``none``，调用方传的顺接帧
不会被 adapter 派发，**尾帧顺接形同虚设**。
"""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.engine import Connection

from ..models import storyboards
from ..response import now
from .task_logger import log_task_progress

__all__ = [
    "SHOT_ROUTES",
    "decide_shot_route",
    "persist_route",
    "recompute_episode_routes",
]

#: 路由类型全集（顺序即声明顺序，会出现在类型校验/展示里）
SHOT_ROUTES = (
    "text_to_video",
    "first_frame_to_video",
    "first_last_frame",
    "reference_to_video",
    "keyframe_to_video",
    "video_editor",
    "blocked",
)

#: 对话/多人/争吵类场景关键词（R2V 多参考图适用）
DIALOGUE_PATTERN = re.compile(
    r"dialogue|meeting|argument|conversation|multi|multi-person|多人|对话|争吵|会议",
    re.IGNORECASE,
)

#: ⚠️ `videos.ts` 里**另有一份**更窄的判据（无「多人」与中文）—— 两处都要保真，别合并
VIDEO_ROUTE_DIALOGUE_PATTERN = re.compile(
    r"dialogue|meeting|argument|conversation|multi",
    re.IGNORECASE,
)


def persist_route(conn: Connection, storyboard_id: int, decision: dict[str, str]) -> None:
    """回写分镜 ``route`` / ``route_reason``（幂等，供账本与前端展示）。

    ⚠️ **尽力而为**：写失败只记 progress 日志、**不抛错**（决策记录失败不该阻断生成）。
    """
    try:
        conn.execute(
            update(storyboards)
            .where(storyboards.c.id == storyboard_id)
            .values(route=decision["route"], route_reason=decision["reason"], updated_at=now())
        )
        log_task_progress("ShotRouter", "route-decided",
                          {"storyboardId": storyboard_id, "route": decision["route"]})
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_progress("ShotRouter", "route-persist-failed",
                          {"storyboardId": storyboard_id, "error": str(err)})


def decide_shot_route(conn: Connection, input_data: dict[str, Any]) -> dict[str, str]:
    """逐镜路由决策，返回 ``{route, reason, referenceMode}`` **并回写分镜 route 字段**。

    优先级（高 → 低）：

    1. ``blocked``：资产门禁阻断；
    2. ``text_to_video``：无首帧（纯文本）；**若给了上一镜尾帧 ``prevTail`` 则改用
       ``first_frame_to_video``** 以其起帧（I2V，保证镜头间像素级连续）；
    3. ``first_last_frame``：有尾帧目标（FL2VA 锁定结束画面）；
    4. ``reference_to_video``：对话/多人场景 + 多参考提供商 + 有参考图；
       4b. **H3 Ref2VA**：provider 是 minimax 且有参考音频；
    5. ``keyframe_to_video``：有中段关键帧图；
    6. ``first_frame_to_video``：默认（首帧图生视频）。
    """
    storyboard_id = input_data.get("storyboardId")
    scene_type = input_data.get("sceneType")
    first_frame_image = input_data.get("firstFrameImage")
    last_frame_image = input_data.get("lastFrameImage")
    keyframe_image = input_data.get("keyframeImage")
    provider = input_data.get("provider")
    can_multi_ref = input_data.get("canMultiRef")
    reference_images = input_data.get("referenceImages") or []
    reference_audio_urls = input_data.get("referenceAudioUrls") or []

    # 1. 资产门禁阻断
    if input_data.get("blocked"):
        decision = {
            "route": "blocked",
            "reason": "first frame asset needs_regeneration，视频生成被资产门禁阻断",
            "referenceMode": "none",
        }
        persist_route(conn, storyboard_id, decision)
        return decision

    # 2. 无首帧（prevTail 必须在此消费 —— 见模块头）
    if not first_frame_image:
        if input_data.get("prevTail"):
            decision = {
                "route": "first_frame_to_video",
                "reason": "分镜无首帧图，但同场景顺接提供上一镜尾帧，以其起帧的图生视频（I2V）",
                "referenceMode": "single",
            }
            persist_route(conn, storyboard_id, decision)
            return decision
        decision = {
            "route": "text_to_video",
            "reason": "分镜无首帧图，采用纯文本提示词生成（T2V）",
            "referenceMode": "none",
        }
        persist_route(conn, storyboard_id, decision)
        return decision

    # 3. 有尾帧目标：首尾帧连接（FL2VA）
    if last_frame_image:
        decision = {
            "route": "first_last_frame",
            "reason": "分镜配置了尾帧目标，采用首尾帧连接动画化（FL2VA）以锁定结束画面",
            "referenceMode": "first_last",
        }
        persist_route(conn, storyboard_id, decision)
        return decision

    is_dialogue = bool(DIALOGUE_PATTERN.search(scene_type or ""))

    # 4. R2V：对话/多人场景 + 多参考提供商 + 有参考图
    if can_multi_ref and len(reference_images) > 0 and is_dialogue:
        decision = {
            "route": "reference_to_video",
            "reason": (
                f"对话/多人场景（{scene_type or 'dialogue'}）且提供商 {provider} 支持多参考图，"
                f"采用 R2V（角色+场景参考图 {len(reference_images)} 张）"
            ),
            "referenceMode": "multiple",
        }
        persist_route(conn, storyboard_id, decision)
        return decision

    # 4b. H3 Ref2VA：有参考音频（出场角色声线样本）
    if provider == "minimax" and len(reference_audio_urls) > 0:
        decision = {
            "route": "reference_to_video",
            "reason": (
                f"提供商 minimax 带参考音频（Ref2VA，{len(reference_audio_urls)} 条声线样本），"
                "采用 R2V 音视频联合生成"
            ),
            "referenceMode": "multiple" if len(reference_images) else "single",
        }
        persist_route(conn, storyboard_id, decision)
        return decision

    # 5. 关键帧参考
    if keyframe_image:
        decision = {
            "route": "keyframe_to_video",
            "reason": "分镜配置了中段关键帧图，以其为主要参考锁定动作/道具/机位中间态",
            "referenceMode": "multiple" if (can_multi_ref and len(reference_images)) else "single",
        }
        persist_route(conn, storyboard_id, decision)
        return decision

    # 6. 默认：首帧图生视频
    hint = "；同场景顺接，以上一镜尾帧衔接起帧" if input_data.get("prevTail") else ""
    decision = {
        "route": "first_frame_to_video",
        "reason": f"默认路线：以首帧图起帧的图生视频（I2V）{hint}",
        "referenceMode": "single",
    }
    persist_route(conn, storyboard_id, decision)
    return decision


def recompute_episode_routes(conn: Connection, episode_id: int) -> None:
    """批量计算某集所有分镜的路由（供启动/展示使用）。

    ⚠️ 这里刻意用「无参考图 / 不支持多参考 / provider=default」的**保守输入**，
    结果会**覆盖**分镜上已有的 route —— 与原实现一致。
    """
    rows = conn.execute(
        select(
            storyboards.c.id,
            storyboards.c.scene_type,
            storyboards.c.first_frame_image,
            storyboards.c.last_frame_image,
            storyboards.c.keyframe_image,
        ).where(storyboards.c.episode_id == episode_id)
    ).all()
    for row in rows:
        decide_shot_route(conn, {
            "storyboardId": row[0],
            "sceneType": row[1],
            "firstFrameImage": row[2],
            "lastFrameImage": row[3],
            "keyframeImage": row[4],
            "provider": "default",
            "canMultiRef": False,
            "referenceImages": [],
            "referenceAudioUrls": [],
        })
