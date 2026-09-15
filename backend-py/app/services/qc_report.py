"""QC 报告交付物服务 —— 与 ``services/qc-report.ts``（515 行）对齐。**阶段 1/3：探测层**。

三类交付物：

1. **Contact Sheet（联系表）**：每镜缩略图（首帧/尾帧/关键帧）+ 元信息网格，供快速浏览整集节奏；
2. **QC 报告**：聚合 ``video_quality_checks``（唇形/角色一致性/连续性/总体分）+ 技术维度，
   输出每镜 QC 状态与问题清单，支持 Markdown / HTML / JSON；
3. **Media Info（媒体信息）**：每镜视频/音频规格（分辨率/帧率/时长/码率/编码），
   **优先读 ``video_generations`` 的实测值，缺项才回退 ffprobe 探测**。

时间线导出文件已由 EDL（`export_service`）与剪映草稿（`jianying_draft`）承担，
本模块的 QC 报告含 ``timeline`` 汇总（每镜时长/起始/路由/状态），补足「交付物」闭环。

⚠️ 阶段 1 的四条保真点：

1. **配置优先、探测兜底**：``vg.duration`` / ``fps`` / ``resolution`` / ``width`` / ``height``
   有值就用，**缺项**才用 ffprobe 的值（逐字段 ``x ?? info.x ?? null``，不是整体替换）；
2. **``fileSize`` 还有第二道兜底**：ffprobe 拿不到就 ``os.stat`` 直接量文件；都没有才 null；
3. **``durationSec`` 的最后兜底是分镜自身**：``vg`` 与 ffprobe 都空、且 ``sb.duration`` 为真值时才用它
   （``0`` 不算，``sb?.duration`` 是 JS 真值判定）；
4. **``fps`` 的解析有讲究**：``avg_frame_rate`` 优先于 ``r_frame_rate``，``'0/0'`` 视为无效，
   ``n``/``d`` 任一为 0 视为无效，结果 ``Math.round(x * 100) / 100``（**两位小数后截断**）。
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.engine import Connection

from ..core.models import dramas, episodes, storyboards, video_generations, video_quality_checks
from ..core.response import js_round
from .export_service import to_local_abs_path

__all__ = [
    "build_contact_sheet_html",
    "build_media_info",
    "build_media_info_report",
    "build_qc_report",
    "build_qc_report_files",
    "build_qc_report_html",
    "build_qc_report_markdown",
    "escape_html",
    "ffprobe_info",
    "img_tag_or_placeholder",
    "latest_qc",
    "latest_video_gen",
    "parse_json_array",
    "parse_json_obj",
    "pct",
]


def _js_to_fixed(value: float, digits: int = 0) -> str:
    """``Number.toFixed`` —— **half-away-from-zero**。

    ⚠️ Python 的 ``f"{x:.0f}"`` 是**银行家舍入**（``1024.5`` ⇒ ``1024``），
    JS 的 ``toFixed`` 是 ``1025`` ⇒ 码率这类数会差 1，必须自己算。
    """
    factor = 10 ** digits
    scaled = value * factor
    rounded = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    return f"{rounded / factor:.{digits}f}"


def parse_json_array(raw: Any) -> list[Any]:
    """JSON 数组字符串 → list（**坏 JSON / 非数组一律空**）。"""
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return value if isinstance(value, list) else []


def parse_json_obj(raw: Any) -> dict[str, Any] | None:
    """JSON 对象字符串 → dict（坏 JSON / 非对象一律 None）。"""
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def pct(score: Any) -> int | float | None:
    """分数夹到 ``[0, 100]``（``None`` 原样返回 —— 不是 0）。"""
    if score is None:
        return None
    return max(0, min(100, score))


async def ffprobe_info(abs_path: str) -> dict[str, Any] | None:
    """调 ffprobe 拿媒体规格（JSON 输出）；**任何失败都返回 None**（不抛）。"""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            abs_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await process.communicate()
    except Exception:  # noqa: BLE001 —— ffprobe 不存在/被杀，等价于 TS 的 err 分支
        return None
    if process.returncode != 0:
        return None

    try:
        payload = json.loads(stdout.decode("utf-8", errors="replace") or "")
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    video_stream = next((s for s in streams
                         if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams
                         if isinstance(s, dict) and s.get("codec_type") == "audio"), None)
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}

    width = None
    height = None
    if video_stream:
        if video_stream.get("width"):
            width = int(video_stream["width"])
        if video_stream.get("height"):
            height = int(video_stream["height"])

    # ⚠️ ``avg_frame_rate`` 优先于 ``r_frame_rate``；``'0/0'`` 与 n/d 为 0 都视为无效
    fps_raw = (video_stream or {}).get("avg_frame_rate") or (video_stream or {}).get("r_frame_rate")
    fps: float | None = None
    if fps_raw and fps_raw != "0/0":
        parts = str(fps_raw).split("/")
        try:
            numerator = float(parts[0])
            denominator = float(parts[1]) if len(parts) > 1 else 0.0
        except (ValueError, TypeError):
            numerator, denominator = 0.0, 0.0
        if numerator and denominator:
            fps = js_round((numerator / denominator) * 100) / 100

    duration_sec = float(fmt["duration"]) if fmt.get("duration") else None
    bitrate = (f"{_js_to_fixed(float(fmt['bit_rate']) / 1024, 0)} kbps"
               if fmt.get("bit_rate") else None)
    file_size = float(fmt["size"]) if fmt.get("size") else None

    return {
        "durationSec": duration_sec,
        "fps": fps,
        "resolution": f"{width}x{height}" if (width and height) else None,
        "width": width,
        "height": height,
        "bitrate": bitrate,
        "codec": (video_stream or {}).get("codec_name") or None,
        "audioCodec": (audio_stream or {}).get("codec_name") or None,
        "fileSize": file_size,
    }


def latest_video_gen(conn: Connection, storyboard_id: int) -> Any | None:
    """取分镜**最新**一条视频生成记录（按 ``created_at`` 升序后的最后一条）。"""
    rows = conn.execute(
        select(video_generations)
        .where(video_generations.c.storyboard_id == storyboard_id)
        .order_by(video_generations.c.created_at)
    ).all()
    return rows[-1] if rows else None


def latest_qc(conn: Connection, storyboard_id: int) -> Any | None:
    """取分镜**最新**一条 QC 记录（同上：升序取最后一条，**没有 ORDER BY DESC**）。"""
    rows = conn.execute(
        select(video_quality_checks)
        .where(video_quality_checks.c.storyboard_id == storyboard_id)
        .order_by(video_quality_checks.c.created_at)
    ).all()
    return rows[-1] if rows else None


async def build_media_info(sb: Any, vg: Any) -> dict[str, Any]:
    """媒体信息：**优先视频生成记录的实测规格，缺项 fallback ffprobe**（逐字段补）。"""
    duration_sec = (vg.duration if vg is not None else None) or None
    fps = (vg.fps if vg is not None else None) or None
    resolution = (vg.resolution if vg is not None else None) or None
    width = (vg.width if vg is not None else None) or None
    height = (vg.height if vg is not None else None) or None
    bitrate: str | None = None
    codec: str | None = None
    audio_codec: str | None = None
    file_size: float | None = None

    src = to_local_abs_path((sb.video_url if sb is not None else None)
                            or (sb.composed_video_url if sb is not None else None))
    if src:
        info = await ffprobe_info(src)
        if info:
            duration_sec = duration_sec if duration_sec is not None else info["durationSec"]
            fps = fps if fps is not None else info["fps"]
            resolution = resolution if resolution is not None else info["resolution"]
            width = width if width is not None else info["width"]
            height = height if height is not None else info["height"]
            bitrate = bitrate if bitrate is not None else info["bitrate"]
            codec = codec if codec is not None else info["codec"]
            audio_codec = audio_codec if audio_codec is not None else info["audioCodec"]
            file_size = file_size if file_size is not None else info["fileSize"]
        # ⚠️ fileSize 的第二道兜底：ffprobe 没给就直接量文件
        if file_size is None:
            try:
                file_size = os.stat(src).st_size
            except OSError:
                pass

    # ⚠️ 最后兜底是分镜自身时长，且只在「为真值」时用（0 不算）
    sb_duration = sb.duration if sb is not None else None
    if duration_sec is None and sb_duration:
        duration_sec = sb_duration

    return {
        "durationSec": duration_sec,
        "fps": fps,
        "resolution": resolution,
        "width": width,
        "height": height,
        "bitrate": bitrate,
        "codec": codec,
        "audioCodec": audio_codec,
        "fileSize": file_size,
    }


def _iso_now() -> str:
    """``new Date().toISOString()`` —— 毫秒 3 位 + ``Z``（Python 默认给 ``+00:00``）。"""
    stamp = datetime.now(timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z"


def _or_zero(value: Any) -> Any:
    """``value ?? 0`` —— **nullish**（分数 0 要保留 0，不能被 ``or`` 吞成假值语义）。"""
    return 0 if value is None else value


async def _probe_video_duration(video_url: str) -> float:
    """``probeVideoDuration``：媒体路径 → 本地绝对路径 → ffprobe；**失败一律 0**。

    ⚠️ 与 ``export_service._probe_video_duration`` 同源（都映射 TS 的 ``utils/video-probe.ts``），
    这里各留一份是为了对齐「每个 service 自带兜底」的原结构。
    """
    from .frame_extractor import _probe_duration_seconds  # noqa: PLC0415

    absolute = to_local_abs_path(video_url)
    if not absolute:
        return 0.0
    try:
        return float(await _probe_duration_seconds(absolute) or 0)
    except Exception:  # noqa: BLE001
        return 0.0


async def build_qc_report(conn: Connection, drama_id: int,
                          opts: dict[str, Any] | None = None) -> dict[str, Any]:
    """构建 QC 报告（结构化数据；Markdown / HTML 都基于它）。"""
    episode_id_opt = (opts or {}).get("episodeId")

    drama = conn.execute(select(dramas).where(dramas.c.id == drama_id)).first()

    condition = and_(episodes.c.drama_id == drama_id, episodes.c.deleted_at.is_(None))
    if episode_id_opt:
        condition = and_(episodes.c.id == episode_id_opt, condition)
    episode_rows = conn.execute(
        select(episodes).where(condition).order_by(episodes.c.episode_number)
    ).all()

    if not episode_rows:
        raise ValueError("No episodes found for QC report")

    # 支持多集时聚合全部；report.episodeId 为 opts 指定或首集
    # ⚠️ `??`：只有 None 才回退首集（传 0 时**保留 0**，随后 find 不到再回落首集）
    episode_id = episode_id_opt if episode_id_opt is not None else episode_rows[0].id
    episode = next((e for e in episode_rows if e.id == episode_id), None) or episode_rows[0]

    storyboard_rows = conn.execute(
        select(storyboards).where(and_(storyboards.c.episode_id == episode.id,
                                       storyboards.c.deleted_at.is_(None)))
        .order_by(storyboards.c.storyboard_number)
    ).all()

    shots: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    cursor_sec = 0.0

    for sb in storyboard_rows:
        vg = latest_video_gen(conn, sb.id)
        qc = latest_qc(conn, sb.id)
        media = await build_media_info(sb, vg)

        # ⚠️ `sb.duration || media?.durationSec || 0`：JS 的 `||` ⇒ 0 也往下走
        duration_sec = sb.duration or (media["durationSec"] if media else None) or 0
        if duration_sec <= 0 and sb.video_url:
            try:
                duration_sec = await _probe_video_duration(sb.video_url)
            except Exception:  # noqa: BLE001 —— 与 TS 的裸 catch 等价
                pass

        timeline.append({
            "storyboardNumber": sb.storyboard_number,
            "startSec": cursor_sec,
            "durationSec": duration_sec,
            "status": sb.status,
        })
        cursor_sec += duration_sec

        shots.append({
            "storyboardNumber": sb.storyboard_number,
            "title": sb.title,
            "description": sb.description,
            "shotType": sb.shot_type,
            "angle": sb.angle,
            "movement": sb.movement,
            "route": sb.route,
            "routeReason": sb.route_reason,
            "duration": sb.duration,
            "durationSec": duration_sec,
            "dialogue": sb.dialogue,
            "status": sb.status,
            "assetStatus": sb.asset_status,
            "takeCount": sb.take_count,
            "takeBudget": sb.take_budget,
            "videoUrl": sb.video_url,
            "composedVideoUrl": sb.composed_video_url,
            "ttsAudioUrl": sb.tts_audio_url,
            "firstFrameImage": sb.first_frame_image,
            "tailFrameImage": sb.tail_frame_image,
            "lastFrameImage": sb.last_frame_image,
            "keyframeImage": sb.keyframe_image,
            "videoGen": (None if vg is None else {
                "provider": vg.provider,
                "model": vg.model,
                "duration": vg.duration,
                "fps": vg.fps,
                "resolution": vg.resolution,
                "width": vg.width,
                "height": vg.height,
                "status": vg.status,
                "seed": vg.seed,
            }),
            "qc": (None if qc is None else {
                "lipSyncScore": pct(qc.lip_sync_score),
                "characterConsistencyScore": pct(qc.character_consistency_score),
                "continuityScore": pct(qc.continuity_score),
                "overallScore": pct(qc.overall_score),
                "status": qc.status,
                "issues": parse_json_array(qc.issues),
                "dimensions": parse_json_obj(qc.dimensions),
            }),
            "media": media,
        })

    # 汇总
    with_qc = [shot for shot in shots if shot["qc"]]

    def avg(selector: Any) -> float | None:
        values = [value for value in (selector(shot) for shot in with_qc) if value is not None]
        if not values:
            return None
        return js_round((sum(values) / len(values)) * 10) / 10

    passed = len([s for s in shots if s["qc"] and _or_zero(s["qc"]["overallScore"]) >= 60])
    failed = len([s for s in shots if s["qc"] and _or_zero(s["qc"]["overallScore"]) < 60])

    return {
        "dramaId": drama_id,
        "dramaTitle": (drama.title if drama is not None else None) or None,
        "episodeId": episode.id,
        "episodeNumber": episode.episode_number,
        "episodeTitle": episode.title or None,
        "generatedAt": _iso_now(),
        "summary": {
            "totalShots": len(shots),
            "shotsWithVideo": len([s for s in shots if s["videoUrl"] or s["composedVideoUrl"]]),
            "shotsWithQc": len(with_qc),
            "avgLipSync": avg(lambda s: s["qc"]["lipSyncScore"]),
            "avgCharacterConsistency": avg(lambda s: s["qc"]["characterConsistencyScore"]),
            "avgContinuity": avg(lambda s: s["qc"]["continuityScore"]),
            "avgOverall": avg(lambda s: s["qc"]["overallScore"]),
            "passed": passed,
            "failed": failed,
        },
        "shots": shots,
        "timeline": timeline,
    }


# ===========================================================================
# 阶段 3：渲染层（Markdown / HTML / 联系表 / 媒体信息）
# ===========================================================================

def _js_num(value: Any, fallback: str = "—") -> str:
    """JS 数字转字符串：**整数不带 ``.0``**（Python 的 float 会带 ⇒ 报告文本会不一样）。"""
    if value is None:
        return fallback
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _js_str(value: Any) -> str:
    """JS 模板字面量里的 ``${x}``：``None`` 渲染成字面量 ``null``（照抄，不美化）。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return _js_num(value, "null")


def _fixed1(value: Any, fallback: str = "—") -> str:
    """``x?.toFixed(1) || '—'``：``None`` 回退 ``—``，否则**固定 1 位小数**。"""
    if value is None:
        return fallback
    return _js_to_fixed(float(value), 1)


def escape_html(text: Any) -> str:
    """HTML 转义（**只转 ``& < > "``，不转单引号** —— 与原 TS 一致）。"""
    return (str(text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def img_tag_or_placeholder(src: Any, label: str) -> str:
    if not src:
        return f'<div class="thumb placeholder"><span>{label}</span></div>'
    return f'<div class="thumb"><img src="{escape_html(src)}" alt="{label}" loading="lazy" /></div>'


async def build_qc_report_markdown(conn: Connection, drama_id: int,
                                  opts: dict[str, Any] | None = None) -> str:
    """渲染为 Markdown QC 报告（可交付给制片/审计）。"""
    report = await build_qc_report(conn, drama_id, opts)
    summary = report["summary"]
    lines: list[str] = []
    lines.append(f"# QC 报告：{report['dramaTitle'] or f'Drama #{report['dramaId']}'}")
    lines.append("")
    lines.append(f"- 剧集：第 {report['episodeNumber']} 集「{report['episodeTitle'] or ''}」")
    lines.append(f"- 生成时间：{report['generatedAt']}")
    lines.append(f"- 分镜总数：{summary['totalShots']} ｜ 有视频：{summary['shotsWithVideo']}"
                 f" ｜ 有 QC：{summary['shotsWithQc']}")
    lines.append(f"- 平均分：唇形 {_js_num(summary['avgLipSync'])} ｜ "
                 f"角色一致性 {_js_num(summary['avgCharacterConsistency'])} ｜ "
                 f"连续性 {_js_num(summary['avgContinuity'])} ｜ "
                 f"总体 {_js_num(summary['avgOverall'])}")
    lines.append(f"- 达标（≥60）：{summary['passed']} ｜ 未达标：{summary['failed']}")
    lines.append("")
    lines.append("## 分镜明细")
    lines.append("")
    lines.append("| # | 景别 | 机位 | 运镜 | 路由 | 时长s | 视频 | QC总体 | 状态 |")
    lines.append("|---|------|------|------|------|-------|------|--------|------|")
    for shot in report["shots"]:
        # ✅ 有视频 / ⚠️ 标记完成但没视频 / ⛔ 其它
        if shot["videoUrl"]:
            status_emoji = "✅"
        elif shot["status"] == "completed":
            status_emoji = "⚠️"
        else:
            status_emoji = "⛔"
        qc_overall = shot["qc"]["overallScore"] if shot["qc"] else None
        lines.append(
            f"| {shot['storyboardNumber']} | {shot['shotType'] or '—'} | {shot['angle'] or '—'} "
            f"| {shot['movement'] or '—'} | {shot['route'] or '—'} | {_fixed1(shot['durationSec'])} "
            f"| {status_emoji} | {_js_num(qc_overall)} | {shot['status'] or '—'} |"
        )
    lines.append("")
    lines.append("## 问题清单")
    lines.append("")
    with_issues = [s for s in report["shots"] if s["qc"] and s["qc"]["issues"]]
    if not with_issues:
        lines.append("无记录问题。")
    else:
        for shot in with_issues:
            lines.append(f"- **Shot #{shot['storyboardNumber']}**（{shot['title'] or ''}）")
            for issue in shot["qc"]["issues"]:
                lines.append(f"  - {issue}")
    lines.append("")
    lines.append("## 时间线汇总")
    lines.append("")
    lines.append("| # | 起始s | 时长s | 状态 |")
    lines.append("|---|-------|-------|------|")
    for entry in report["timeline"]:
        lines.append(f"| {entry['storyboardNumber']} | {_fixed1(entry['startSec'])} "
                     f"| {_fixed1(entry['durationSec'])} | {entry['status'] or '—'} |")
    return "\n".join(lines)


async def build_contact_sheet_html(conn: Connection, drama_id: int,
                                  opts: dict[str, Any] | None = None) -> str:
    """构建联系表 HTML（每镜缩略图 + 元信息网格，可浏览器打印/转 PDF）。"""
    report = await build_qc_report(conn, drama_id, opts)
    summary = report["summary"]
    lines: list[str] = []
    lines.append(f'<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">'
                 f"<title>Contact Sheet - {escape_html(report['dramaTitle'])}</title>")
    lines.append("""<style>
    body{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:24px;color:#222;background:#fff}
    h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:24px 0 8px;border-bottom:2px solid #eee;padding-bottom:4px}
    .meta{color:#666;font-size:12px;margin:0 0 16px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
    .card{border:1px solid #e3e3e3;border-radius:8px;overflow:hidden;break-inside:avoid}
    .card .header{display:flex;justify-content:space-between;align-items:center;padding:6px 10px;background:#f7f7f7;font-size:12px;font-weight:600}
    .route-tag{font-family:ui-monospace,Consolas,monospace;font-size:10px;padding:1px 6px;border:1px solid #4fc3f7;color:#0288d1;border-radius:4px;background:#e1f5fe}
    .route-tag.bad{border-color:#e57373;color:#c62828;background:#ffebee}
    .thumbs{display:grid;grid-template-columns:1fr 1fr;gap:4px;padding:4px}
    .thumb{aspect-ratio:16/9;background:#fafafa;border:1px solid #eee;display:flex;align-items:center;justify-content:center;font-size:10px;color:#999;overflow:hidden}
    .thumb img{width:100%;height:100%;object-fit:cover}
    .info{padding:0 10px 10px;font-size:11px;color:#444;line-height:1.6}
    .info b{color:#111}
    .qc-ok{color:#2e7d32;font-weight:700}.qc-bad{color:#c62828;font-weight:700}.qc-na{color:#999}
    .summary{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#333;margin-bottom:8px}
    .summary span{padding:4px 10px;border-radius:20px;background:#f0f0f0}
    @media print{.card{break-inside:avoid}body{margin:12px}}
  </style></head><body>""")
    lines.append("<h1>联系表 Contact Sheet</h1>")
    lines.append(f'<p class="meta">{escape_html(report["dramaTitle"])} · 第 {report["episodeNumber"]} 集'
                 f'「{escape_html(report["episodeTitle"])}」 · {report["generatedAt"]}</p>')
    lines.append(f'<div class="summary">\n'
                 f'  <span>分镜 {summary["totalShots"]}</span>\n'
                 f'  <span>有视频 {summary["shotsWithVideo"]}</span>\n'
                 f'  <span>QC 达标 {summary["passed"]} / 未达标 {summary["failed"]}</span>\n'
                 f'  <span>总体均分 {_js_num(summary["avgOverall"])}</span>\n'
                 f"</div>")
    lines.append('<div class="grid">')
    for shot in report["shots"]:
        qc = shot["qc"]
        if qc is None or qc["overallScore"] is None:
            qc_class = "qc-na"
        elif qc["overallScore"] >= 60:
            qc_class = "qc-ok"
        else:
            qc_class = "qc-bad"
        route_tag = ""
        if shot["route"]:
            bad = " bad" if shot["route"] == "blocked" else ""
            route_tag = (f'<span class="route-tag{bad}">'
                         f'{escape_html(shot["route"])}</span>')
        media = shot["media"] or {}
        spec_line = ""
        if media.get("resolution"):
            fps_part = f'@{_js_num(media.get("fps"))}fps' if media.get("fps") else ""
            dur_part = f' {_fixed1(media.get("durationSec"))}s' if media.get("durationSec") else ""
            spec_line = (f'<div>规格：{escape_html(media["resolution"])} '
                         f'{fps_part}{dur_part}</div>')
        issues_line = ""
        if qc and qc["issues"]:
            issues_line = (f'<div style="color:#c62828">问题：'
                           f'{escape_html("；".join(qc["issues"][:3]))}</div>')
        description = shot["description"] or ""
        truncated = description[:120] + ("…" if len(description) > 120 else "")
        lines.append(f"""<div class="card">
      <div class="header"><span>#{shot['storyboardNumber']} {escape_html(shot['title'] or '')}</span>
      {route_tag}</div>
      <div class="thumbs">{img_tag_or_placeholder(shot['firstFrameImage'], '首帧')}{img_tag_or_placeholder(shot['tailFrameImage'] or shot['lastFrameImage'], '尾帧')}</div>
      <div class="info">
        <div>{escape_html(shot['shotType'] or '—')} / {escape_html(shot['angle'] or '—')} / {escape_html(shot['movement'] or '—')} ｜ 时长 {_fixed1(shot['durationSec'])}s</div>
        <div>视频：{'✅ 已生成' if shot['videoUrl'] else '—'} ｜ 合成：{'✅' if shot['composedVideoUrl'] else '—'} ｜ take {_js_str(shot['takeCount'])}/{_js_str(shot['takeBudget'])}</div>
        <div>QC：唇形 <b>{_js_num(qc['lipSyncScore'] if qc else None)}</b> ｜ 角色 <b>{_js_num(qc['characterConsistencyScore'] if qc else None)}</b> ｜ 连续 <b>{_js_num(qc['continuityScore'] if qc else None)}</b> ｜ <span class="{qc_class}">总体 {_js_num(qc['overallScore'] if qc else None, "未跑")}</span></div>
        {spec_line}
        {issues_line}
        <div style="color:#888;font-size:10px">{escape_html(truncated)}</div>
      </div>
    </div>""")
    lines.append("</div></body></html>")
    return "\n".join(lines)


async def build_qc_report_html(conn: Connection, drama_id: int,
                              opts: dict[str, Any] | None = None) -> str:
    """渲染为 HTML QC 报告（Markdown 转义后塞进 ``<pre>``）。"""
    markdown = await build_qc_report_markdown(conn, drama_id, opts)
    return ('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>QC Report</title>\n'
            "<style>body{font-family:ui-monospace,Consolas,monospace;margin:24px;"
            "font-size:13px;line-height:1.7;color:#222}pre{white-space:pre-wrap;word-break:break-all}"
            f"</style>\n</head><body><pre>{escape_html(markdown)}</pre></body></html>")


async def build_qc_report_files(conn: Connection, drama_id: int,
                               opts: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """生成交付物文件清单（联系表 HTML + QC 报告 MD/HTML），供打包下载。"""
    contact_sheet = await build_contact_sheet_html(conn, drama_id, opts)
    markdown = await build_qc_report_markdown(conn, drama_id, opts)
    html = await build_qc_report_html(conn, drama_id, opts)
    episode_id = (opts or {}).get("episodeId")
    label = f"ep{episode_id}" if episode_id else "all"
    return [
        {"name": f"contact-sheet_{label}.html", "content": contact_sheet},
        {"name": f"qc-report_{label}.md", "content": markdown},
        {"name": f"qc-report_{label}.html", "content": html},
    ]


async def build_media_info_report(conn: Connection, drama_id: int,
                                 opts: dict[str, Any] | None = None) -> dict[str, Any]:
    """构建媒体信息报告（纯媒体规格，供交付/转码核对）。"""
    report = await build_qc_report(conn, drama_id, opts)
    return {
        "generatedAt": report["generatedAt"],
        "files": [shot["media"] for shot in report["shots"]],
    }
