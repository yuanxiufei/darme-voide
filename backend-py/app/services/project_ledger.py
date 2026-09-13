"""工程账本 —— 移植 ``services/project-ledger.ts``（整服务，纯 DB）。

对齐 H3-Codex-Drama 的「可复现工程（project.yaml）」：把整剧制作全链路落成一份账本
（剧本哈希 / 分镜提示词 / 逐镜路由决策 / 输入资产引用 / 生成记录 / 成本），用于

1. **可复现导出**：拿到账本就知道「每镜怎么生成、用了什么提示词/参考图/路由、花了多少」
2. **断点续作**：以 ``script_hash`` 比对剧本是否变更，定位 stale 分镜
3. **审计交付**：Markdown 报告交给制片/审计

⚠️ **输出形状是 camelCase**（原 TS 手工构造，没走 ``toSnakeCase``）—— 与多数域的 snake_case 相反，别统一。
⚠️ ``cost`` 字段直接用 ``getEpisodeCostBoard`` 的 ep 行（**它是 snake_case**）⇒ 同一份账本里
camelCase 与 snake_case 混着，这是原契约，照抄。
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..models import (
    characters,
    dramas,
    episodes,
    image_generations,
    prop_templates,
    scenes,
    storyboards,
    video_generations,
)
from ..response import now
from .usage_tracking import get_episode_cost_board

#: 路由标识 → 中文标签（Markdown 用）
ROUTE_LABEL: dict[str, str] = {
    "text_to_video": "T2V 纯文本生成",
    "first_frame_to_video": "I2V 首帧图生视频",
    "first_last_frame": "FL2VA 首尾帧连接",
    "reference_to_video": "R2V 多参考图/参考音频",
    "keyframe_to_video": "I2V-K 关键帧参考",
    "video_editor": "Editor 视频编辑",
    "blocked": "资产门禁阻断",
}


def route_label(route: str | None) -> str:
    """路由摘要（用于 Markdown 表头）—— 未知路由原样返回，空值用全角破折号。"""
    if not route:
        return "—"
    return ROUTE_LABEL.get(route) or route


def _safe_json_array(value: Any) -> list[Any]:
    """损坏 / 非数组一律回退空数组（账本不该因为一个坏字段导不出来）。"""
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def build_project_ledger(
    conn: Connection, drama_id: int, episode_id: int | None = None
) -> dict[str, Any]:
    """构建整剧工程账本（JSON 结构）。``drama_id`` 不存在时抛错。"""
    ts = now()

    drama = conn.execute(select(dramas).where(dramas.c.id == drama_id)).first()
    if drama is None:
        raise ValueError(f"Drama {drama_id} not found")

    asset_characters = conn.execute(
        select(characters).where(characters.c.drama_id == drama_id)
    ).all()
    asset_scenes = conn.execute(select(scenes).where(scenes.c.drama_id == drama_id)).all()
    asset_props = conn.execute(
        select(prop_templates).where(prop_templates.c.drama_id == drama_id)
    ).all()

    ep_rows = sorted(
        conn.execute(select(episodes).where(episodes.c.drama_id == drama_id)).all(),
        key=lambda e: e.episode_number,
    )
    if episode_id:
        ep_rows = [e for e in ep_rows if e.id == episode_id]

    # 多集成本看板失败不该让整份账本导不出（原 TS 用 try/catch 兜住）
    try:
        cost_board = get_episode_cost_board(conn, drama_id)
    except Exception:  # noqa: BLE001
        cost_board = None
    cost_by_episode = {
        row["episode_id"]: row for row in (cost_board or {}).get("episodes", [])
    }

    episode_records: list[dict[str, Any]] = []
    for ep in ep_rows:
        sb_rows = conn.execute(
            select(storyboards)
            .where(storyboards.c.episode_id == ep.id)
            .order_by(storyboards.c.storyboard_number.asc())
        ).all()

        shot_records: list[dict[str, Any]] = []
        for sb in sb_rows:
            images = conn.execute(
                select(image_generations).where(image_generations.c.storyboard_id == sb.id)
            ).all()
            videos = conn.execute(
                select(video_generations).where(video_generations.c.storyboard_id == sb.id)
            ).all()

            shot_records.append(
                {
                    "storyboardId": sb.id,
                    "storyboardNumber": sb.storyboard_number,
                    "title": sb.title,
                    "sceneType": sb.scene_type,
                    "shotType": sb.shot_type,
                    "duration": sb.duration,
                    "rhythmPhase": sb.rhythm_phase,
                    "route": sb.route,
                    "routeReason": sb.route_reason,
                    "scriptHash": sb.script_hash,
                    "imagePrompt": sb.image_prompt,
                    "videoPrompt": sb.video_prompt,
                    "firstFrameImage": sb.first_frame_image,
                    "lastFrameImage": sb.last_frame_image,
                    "keyframeImage": sb.keyframe_image,
                    "referenceImages": sb.reference_images,
                    # 取**最后一条**视频记录的参考音频（原 TS 用 videos[length-1]）
                    "referenceAudioUrls": videos[-1].reference_audio_urls if videos else None,
                    "imageGenerations": [
                        {
                            "id": g.id,
                            "provider": g.provider,
                            "model": g.model,
                            "prompt": g.prompt,
                            "frameType": g.frame_type,
                            "status": g.status,
                            "imageUrl": g.image_url,
                            "createdAt": g.created_at,
                        }
                        for g in images
                    ],
                    "videoGenerations": [
                        {
                            "id": v.id,
                            "provider": v.provider,
                            "model": v.model,
                            "prompt": v.prompt,
                            "referenceMode": v.reference_mode,
                            "route": v.route,
                            "routeReason": v.route_reason,
                            "status": v.status,
                            "videoUrl": v.video_url,
                            "blockReason": v.block_reason,
                            "createdAt": v.created_at,
                        }
                        for v in videos
                    ],
                    "composedVideoUrl": sb.composed_video_url,
                }
            )

        episode_records.append(
            {
                "id": ep.id,
                "episodeNumber": ep.episode_number,
                "title": ep.title,
                "status": ep.status,
                "scriptHash": ep.script_hash,
                "duration": ep.duration,
                "videoUrl": ep.video_url,
                "bgmUrl": ep.bgm_url,
                "imageConfigId": ep.image_config_id,
                "videoConfigId": ep.video_config_id,
                "audioConfigId": ep.audio_config_id,
                "cost": cost_by_episode.get(ep.id),
                "shots": shot_records,
            }
        )

    return {
        "schemaVersion": 1,
        "exportedAt": ts,
        "drama": {
            "id": drama.id,
            "title": drama.title,
            "genre": drama.genre,
            "style": drama.style,
            "totalEpisodes": drama.total_episodes,
            "status": drama.status,
        },
        "assets": {
            "characters": [
                {
                    "id": c.id,
                    "name": c.name,
                    "imageUrl": c.image_url,
                    "voiceSampleUrl": c.voice_sample_url,
                }
                for c in asset_characters
            ],
            "scenes": [
                {"id": s.id, "location": s.location, "imageUrl": s.image_url}
                for s in asset_scenes
            ],
            "props": [
                {"id": p.id, "name": p.name, "category": p.category, "imageUrl": p.image_url}
                for p in asset_props
            ],
        },
        "episodes": episode_records,
    }


def _md_slice(text: str, limit: int = 240) -> str:
    """提示词截断（超长补省略号）——与 TS 的 slice + 条件省略号同形。"""
    return text[:limit] + ("…" if len(text) > limit else "")


def build_project_ledger_markdown(
    conn: Connection, drama_id: int, episode_id: int | None = None
) -> str:
    """生成可复现 Markdown 报告（人类可读，交付制片/审计）。"""
    ledger = build_project_ledger(conn, drama_id, episode_id)
    lines: list[str] = []

    lines.append(f"# 工程账本 · {ledger['drama']['title']}")
    lines.append("")
    lines.append(f"- 导出时间：{ledger['exportedAt']}")
    lines.append(
        f"- 剧集类型：{ledger['drama']['genre'] or '—'} / 风格：{ledger['drama']['style'] or '—'}"
    )
    total_episodes = ledger["drama"]["totalEpisodes"]
    lines.append(f"- 总集数：{total_episodes if total_episodes is not None else '—'}")
    lines.append(f"- 状态：{ledger['drama']['status'] or '—'}")
    lines.append("")
    lines.append(
        "> 本报告为可复现工程账本：每镜的提示词、输入指纹、路由决策、参考资产与成本均留档，"
        "可用于断点续作与审计。"
    )
    lines.append("")

    lines.append("## 资产基线")
    lines.append("")
    chars = ledger["assets"]["characters"]
    scns = ledger["assets"]["scenes"]
    prps = ledger["assets"]["props"]
    lines.append(f"- 角色：{len(chars)} 个（{'、'.join(c['name'] for c in chars) or '—'}）")
    lines.append(f"- 场景：{len(scns)} 个（{'、'.join(s['location'] for s in scns) or '—'}）")
    lines.append(f"- 道具：{len(prps)} 个（{'、'.join(p['name'] for p in prps) or '—'}）")
    lines.append("")

    for ep in ledger["episodes"]:
        lines.append(f"## 第 {ep['episodeNumber']} 集 · {ep['title']}")
        lines.append("")
        duration = ep["duration"]
        lines.append(
            f"- 状态：{ep['status'] or '—'}　|　时长：{duration if duration is not None else '—'}s"
            f"　|　指纹：{ep['scriptHash'] or '—'}"
        )
        if ep["videoUrl"]:
            lines.append(f"- 成片：{ep['videoUrl']}")
        if ep["bgmUrl"]:
            lines.append(f"- BGM：{ep['bgmUrl']}")
        lines.append(
            f"- 配置：image#{ep['imageConfigId'] if ep['imageConfigId'] is not None else '—'}"
            f" / video#{ep['videoConfigId'] if ep['videoConfigId'] is not None else '—'}"
            f" / audio#{ep['audioConfigId'] if ep['audioConfigId'] is not None else '—'}"
        )

        if ep["cost"]:
            cost = ep["cost"]
            total_cost = cost.get("total_cost")
            retry = cost.get("retry_cost") or 0
            retry_part = f"，重拍 {retry} 元" if retry > 0 else ""
            lines.append(
                f"- 成本：{total_cost if total_cost is not None else '—'} 元"
                f"（调用 {cost.get('total_calls') or 0} 次{retry_part}）"
            )
        lines.append("")

        lines.append("| # | 分镜 | 时长 | 相位 | 路由 | 指纹 | 视频状态 |")
        lines.append("|---|------|------|------|------|------|----------|")
        for shot in ep["shots"]:
            video_status = (
                "/".join(
                    v["status"]
                    for v in shot["videoGenerations"]
                    if v["status"] in ("completed", "processing", "blocked_by_missing_asset")
                )
                or "—"
            )
            phase = shot["rhythmPhase"] or "—"
            title = (shot["title"] or f"分镜 {shot['storyboardNumber']}").replace("|", "\\|")
            shot_duration = shot["duration"]
            lines.append(
                f"| {shot['storyboardNumber']} | {title} "
                f"| {shot_duration if shot_duration is not None else '—'}s | {phase} "
                f"| {route_label(shot['route'])} "
                f"| {(shot['scriptHash'] or '')[:8] or '—'} | {video_status} |"
            )
        lines.append("")

        for shot in ep["shots"]:
            suffix = f" · {shot['title']}" if shot["title"] else ""
            lines.append(f"### 分镜 {shot['storyboardNumber']}{suffix}")
            lines.append("")
            if shot["routeReason"]:
                lines.append(
                    f"- **路由**：{route_label(shot['route'])} — {shot['routeReason']}"
                )
            if shot["scriptHash"]:
                lines.append(f"- **指纹**：`{shot['scriptHash']}`")
            if shot["imagePrompt"]:
                lines.append(f"- **图片提示词**：`{_md_slice(shot['imagePrompt'])}`")
            if shot["videoPrompt"]:
                lines.append(f"- **视频提示词**：`{_md_slice(shot['videoPrompt'])}`")

            assets: list[str] = []
            if shot["firstFrameImage"]:
                assets.append(f"首帧 `{shot['firstFrameImage']}`")
            if shot["lastFrameImage"]:
                assets.append(f"尾帧 `{shot['lastFrameImage']}`")
            if shot["keyframeImage"]:
                assets.append(f"关键帧 `{shot['keyframeImage']}`")
            for ref in _safe_json_array(shot["referenceImages"]):
                assets.append(f"参考图 `{ref}`")
            for ref in _safe_json_array(shot["referenceAudioUrls"]):
                assets.append(f"参考音频 `{ref}`")
            if assets:
                lines.append("- **输入资产**：")
                for asset in assets:
                    lines.append(f"  - {asset}")

            if shot["videoGenerations"]:
                lines.append(f"- **生成记录**（{len(shot['videoGenerations'])} 次视频尝试）：")
                for v in shot["videoGenerations"]:
                    tail = f" 阻断：{v['blockReason']}" if v["blockReason"] else ""
                    tail += f" 产物：{v['videoUrl']}" if v["videoUrl"] else ""
                    lines.append(
                        f"  - #{v['id']} `{v['status']}` route={v['route'] or '—'}"
                        f" model={v['model'] or '—'}{tail}"
                    )
            if shot["composedVideoUrl"]:
                lines.append(f"- **成片**：{shot['composedVideoUrl']}")
            lines.append("")

    lines.append("## 可复现导出说明")
    lines.append("")
    lines.append(
        "若需断点续作：以各分镜 `指纹`（script_hash）与当前 `episodes.script_hash` 比对，"
        "不一致即视为剧本变更后的 stale 分镜，需重新拆解。生成时按 `路由` 列选择对应路线，"
        "参考资产按「输入资产」列表注入。"
    )
    lines.append("")

    return "\n".join(lines)


def scan_stale_shots(
    conn: Connection, drama_id: int, episode_id: int | None = None
) -> dict[str, Any]:
    """断点续作扫描：返回「账本指纹 vs 当前剧本」不一致的分镜清单。"""
    ledger = build_project_ledger(conn, drama_id, episode_id)
    stale: list[dict[str, Any]] = []
    for ep in ledger["episodes"]:
        for shot in ep["shots"]:
            if shot["scriptHash"] and shot["scriptHash"] != ep["scriptHash"]:
                stale.append(
                    {
                        "episodeId": ep["id"],
                        "episodeNumber": ep["episodeNumber"],
                        "storyboardId": shot["storyboardId"],
                        "storyboardNumber": shot["storyboardNumber"],
                    }
                )
    return {
        "dramaId": drama_id,
        "exportedAt": ledger["exportedAt"],
        "staleCount": len(stale),
        "stale": stale,
    }
