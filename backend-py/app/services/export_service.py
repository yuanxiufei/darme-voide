"""导出服务 —— 与 ``services/export-service.ts``（192 行）对齐。

三件事：**收集待导出文件清单**、**打成 ZIP**、**生成 CMX3600 EDL**（Premiere/达芬奇可导入）。

⚠️ 六处保真点：

1. **只导出「已落盘」的媒体**：``data:`` / ``http(s)://`` 一律跳过（``to_local_abs_path`` 返回 None）；
2. **ZIP 内路径要 sanitize**：``\\/:*?"<>|`` 与控制字符换成 ``_``，空则 ``untitled``；
   扩展名从媒体路径推断（**去掉 ``?query``**），推断不出用 ``bin``；
3. **两类产物分组**：``videos/ep01_标题.ext``（整集成片）、``characters/``、``scenes/``、
   ``storyboards/ep01/sb01_first|last|tail.ext``（**真实尾帧 ``tail_frame_image`` 单独导出**，
   不与设计尾帧混用）；
4. **``scope`` 语义**：``video`` 只打成片、``assets`` 只打源素材、``all``（默认）全打；
5. **EDL 时间码**：``HH:MM:SS:FF``，``fps`` 默认 **25**（PAL）；秒 → 帧用 ``round``（不是 floor）；
6. **分镜时长缺失时去探测视频真实时长**（``sb.duration <= 0`` ⇒ ffprobe），**探测失败就跳过该镜**。

⚠️ 与原实现的**一处已知差异**：TS 用 ``archiver`` 流式边压边出；这里用标准库 ``zipfile``
写**临时文件**再由 ``StreamingResponse`` 分块吐（`X-Export-Count` 等头与文件名完全一致），
避免把整包读进内存，同时不引入新依赖。
"""
from __future__ import annotations

import os
import re
import tempfile
import time
import zipfile
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.engine import Connection

from ..core.models import characters, dramas, episodes, scenes, storyboards
from ..core.response import js_round
from .frame_extractor import to_abs_media_path

__all__ = [
    "build_edl",
    "build_export_zip",
    "collect_drama_export_files",
    "export_tmp_dir",
    "ext_of",
    "sanitize_name",
    "seconds_to_timecode",
    "sweep_stale_exports",
    "to_local_abs_path",
]

#: 导出包（**一次性中转**）的专属子目录名 ✓ —— 落在系统临时区里，但**自己一层** ✓，
#: 不和系统临时区里的杂物混住（`%TEMP%` 根上直接扔 zip ⇒ 出事时根本分不清哪个是本仓的 ✓✗）。
EXPORT_TMP_PREFIX = "darme-export"
#: 陈旧判据（小时）：正常路径由 `routers/export.py` 的 `BackgroundTask` 删掉 ✓，
#: 超过这个时长还躺着的，一定是**没删掉**的那份 ✓✗。
EXPORT_STALE_HOURS = 6.0

#: 文件名非法字符（与 TS 的正则一一对应：``\\/:*?"<>|`` + 换行/制表）
_INVALID_NAME_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]')
#: 扩展名推断（**可选 query 要吃掉**）
_EXT_PATTERN = re.compile(r"\.([a-zA-Z0-9]+)(?:\?.*)?$")


def sanitize_name(name: Any) -> str:
    """清理文件名：非法字符换 ``_``，去空白，空值回退 ``untitled``。"""
    return _INVALID_NAME_CHARS.sub("_", str(name or "")).strip() or "untitled"


def ext_of(path: Any) -> str:
    """从媒体路径推断扩展名（去 query），默认 ``bin``。"""
    match = _EXT_PATTERN.search(str(path or ""))
    return match.group(1).lower() if match else "bin"


def to_local_abs_path(path: Any) -> str | None:
    """媒体路径（``static/xxx``）→ 本地绝对路径；**远程 URL / data: 等未落盘形态返回 None**。"""
    if not path:
        return None
    text = str(path)
    if text.startswith(("data:", "http://", "https://")):
        return None
    try:
        absolute = to_abs_media_path(text)
    except Exception:  # noqa: BLE001 —— 与 TS 的裸 catch 等价
        return None
    return absolute if os.path.exists(absolute) else None


def collect_drama_export_files(conn: Connection, drama_id: int,
                              scope: str) -> list[dict[str, str]]:
    """收集该 drama 下需要导出的文件清单（``[{absPath, zipPath}]``）。"""
    files: list[dict[str, str]] = []

    episode_rows = conn.execute(
        select(episodes).where(and_(episodes.c.drama_id == drama_id,
                                    episodes.c.deleted_at.is_(None)))
        .order_by(episodes.c.episode_number)
    ).all()
    ep_number_by_id = {row.id: row.episode_number for row in episode_rows}

    # 成片视频（整集）
    if scope in ("all", "video"):
        for episode in episode_rows:
            if not episode.video_url:
                continue
            absolute = to_local_abs_path(episode.video_url)
            if not absolute:
                continue
            label = str(episode.episode_number).zfill(2)
            files.append({
                "absPath": absolute,
                "zipPath": f"videos/ep{label}_{sanitize_name(episode.title)}."
                           f"{ext_of(episode.video_url)}",
            })

    # 源素材
    if scope in ("all", "assets"):
        # 角色立绘
        char_rows = conn.execute(
            select(characters).where(and_(characters.c.drama_id == drama_id,
                                          characters.c.deleted_at.is_(None)))
        ).all()
        for char in char_rows:
            if not char.image_url:
                continue
            absolute = to_local_abs_path(char.image_url)
            if not absolute:
                continue
            files.append({
                "absPath": absolute,
                "zipPath": f"characters/{sanitize_name(char.name)}.{ext_of(char.image_url)}",
            })

        # 场景图（⚠️ 与 TS 一致：**不过滤软删** —— scenes 的删除本来就是硬删）
        scene_rows = conn.execute(
            select(scenes).where(scenes.c.drama_id == drama_id)
        ).all()
        for scene in scene_rows:
            if not scene.image_url:
                continue
            absolute = to_local_abs_path(scene.image_url)
            if not absolute:
                continue
            files.append({
                "absPath": absolute,
                "zipPath": f"scenes/{sanitize_name(scene.location)}.{ext_of(scene.image_url)}",
            })

        # 分镜关键帧（首帧/设计尾帧/真实尾帧）
        if episode_rows:
            storyboard_rows = conn.execute(
                select(storyboards)
                .where(storyboards.c.episode_id.in_([row.id for row in episode_rows]))
                .order_by(storyboards.c.episode_id, storyboards.c.storyboard_number)
            ).all()
            for sb in storyboard_rows:
                ep_label = str(ep_number_by_id.get(sb.episode_id) or 0).zfill(2)
                sb_label = str(sb.storyboard_number).zfill(2)
                directory = f"storyboards/ep{ep_label}"
                for column, suffix in (("first_frame_image", "first"),
                                       ("last_frame_image", "last"),
                                       ("tail_frame_image", "tail")):
                    value = getattr(sb, column)
                    if not value:
                        continue
                    absolute = to_local_abs_path(value)
                    if absolute:
                        files.append({
                            "absPath": absolute,
                            "zipPath": f"{directory}/sb{sb_label}_{suffix}.{ext_of(value)}",
                        })

    return files


def export_tmp_dir() -> str:
    """导出包落点：系统临时区下的**专属**子目录（自建）✓。

    为什么不用 ``tempfile.TemporaryDirectory``：包要活到**响应分块吐完**之后
    （``StreamingResponse`` 的 background task 才删 ✓），生命周期跨出函数 ⇒
    只能先落盘、再由调用方删 ✓。专属一层是为了出事时能一眼认出是本仓的中转包 ✓。
    """
    path = os.path.join(tempfile.gettempdir(), EXPORT_TMP_PREFIX)
    os.makedirs(path, exist_ok=True)
    return path


def sweep_stale_exports(max_age_hours: float = EXPORT_STALE_HOURS) -> list[str]:
    """清掉**没被删掉**的旧导出包 ✓✗，返回被清掉的路径。

    正常路径由 ``routers/export.py`` 的 ``BackgroundTask`` 删 ✓；但客户端半路断开、
    或建完包之后抛异常时 background 跑不到 ✓✗ —— 一个完整导出包（合集能到几十 GB）
    会永久压在系统临时区，且**没人再指得回它** ✓✗。故每次建包前顺手扫一遍 ✓。
    只清本前缀目录里的 ``*.zip``，别的一律不碰 ✓。
    """
    root = export_tmp_dir()
    cutoff = time.time() - max_age_hours * 3600.0
    removed: list[str] = []
    for name in os.listdir(root):
        if not name.endswith(".zip"):
            continue
        target = os.path.join(root, name)
        try:
            if os.path.getmtime(target) < cutoff:
                os.remove(target)
                removed.append(target)
        except OSError:
            continue  # 别的进程正在吐/正在删 ⇒ 下一轮再说
    return removed


def build_export_zip(files: list[dict[str, str]], *, sweep: bool = True) -> str:
    """把清单打成 ZIP，返回**临时文件路径**（调用方负责分块吐完再删）。

    ⚠️ 与 TS 的 ``archiver`` 流式实现不同（见模块头）：这里用标准库写临时文件，
    既不把整包读进内存，也不新增依赖。``zipfile`` 默认 ``ZIP_STORED``；这里显式
    用 **deflate level=6**，与 TS 的 ``{ zlib: { level: 6 } }`` 对齐。

    落点是 :func:`export_tmp_dir`（专属子目录，不是 ``%TEMP%`` 根上）✓；
    ``sweep=True`` 时先清陈旧件（:func:`sweep_stale_exports`）✓。
    """
    if sweep:
        sweep_stale_exports()
    # mkstemp 而不是 NamedTemporaryFile：只要一个**名字**，句柄立刻关掉，
    # Windows 上也不会有"文件已被打开"的锁 ✓。
    fd, name = tempfile.mkstemp(suffix=".zip", prefix="export-", dir=export_tmp_dir())
    os.close(fd)
    with zipfile.ZipFile(name, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for entry in files:
            if not os.path.exists(entry["absPath"]):
                continue  # 与 TS 的 ENOENT 告警等价：文件被移除就跳过
            archive.write(entry["absPath"], arcname=entry["zipPath"])
    return name


def seconds_to_timecode(seconds: float, fps: int = 25) -> str:
    """秒 → ``HH:MM:SS:FF``（fps 默认 25，PAL）。

    ⚠️ 总帧数用 ``Math.round`` 语义（不是 floor、**也不是 Python 的 round**）：
    原 TS 是 ``Math.round(seconds * fps)`` = ``floor(x + 0.5)``，而 Python ``round`` 是
    **银行家舍入**（``round(0.5) == 0``）—— 差半帧会直接体现在 FF 上，故用仓里的 ``js_round``。
    """
    total_frames = max(0, js_round(seconds * fps))
    frames = total_frames % fps
    total_seconds = total_frames // fps
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return (f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}")


async def build_edl(conn: Connection, drama_id: int, opts: dict[str, Any] | None = None) -> str:
    """生成 CMX3600 格式 EDL：每个**有视频**的分镜一条事件，按「集号 → 分镜号」铺时间线。"""
    opts = opts or {}
    fps = opts.get("fps") or 25
    drama = conn.execute(select(dramas).where(dramas.c.id == drama_id)).first()

    episode_id = opts.get("episodeId")
    condition = and_(episodes.c.drama_id == drama_id, episodes.c.deleted_at.is_(None))
    if episode_id:
        condition = and_(episodes.c.id == episode_id, condition)
    episode_rows = conn.execute(
        select(episodes).where(condition).order_by(episodes.c.episode_number)
    ).all()

    lines: list[str] = [f"TITLE: {(drama.title if drama is not None else None) or 'Drama Studio Export'}",
                        "FCM: NON-DROP FRAME", ""]
    event_no = 0
    record_in_sec = 0.0

    for episode in episode_rows:
        storyboard_rows = conn.execute(
            select(storyboards).where(and_(storyboards.c.episode_id == episode.id,
                                          storyboards.c.deleted_at.is_(None)))
            .order_by(storyboards.c.storyboard_number)
        ).all()
        for sb in storyboard_rows:
            if not sb.video_url:
                continue
            duration = sb.duration or 0
            if duration <= 0:
                duration = await _probe_video_duration(sb.video_url)
            if duration <= 0:
                continue

            event_no += 1
            clip_name = (str(sb.video_url).split("/")[-1]
                         or f"ep{episode.episode_number}_sb{sb.storyboard_number}.mp4")
            src_in = seconds_to_timecode(0, fps)
            src_out = seconds_to_timecode(duration, fps)
            rec_in = seconds_to_timecode(record_in_sec, fps)
            rec_out = seconds_to_timecode(record_in_sec + duration, fps)

            lines.append(f"{str(event_no).zfill(3)}  AX       V     C        "
                         f"{src_in} {src_out} {rec_in} {rec_out}")
            lines.append(f"* FROM CLIP NAME: {clip_name}")
            comment = " - ".join(x for x in (sb.title, sb.description) if x)
            if comment:
                lines.append(f"* COMMENT: {comment}")
            record_in_sec += duration

    lines.append("")
    return "\n".join(lines)


async def _probe_video_duration(video_url: str) -> float:
    """探测分镜视频真实时长（ffprobe）；失败返回 0（调用方会跳过该镜）。"""
    from .frame_extractor import _probe_duration_seconds  # noqa: PLC0415

    try:
        absolute = to_abs_media_path(video_url)
    except Exception:  # noqa: BLE001
        return 0.0
    if not os.path.exists(absolute):
        return 0.0
    try:
        return float(await _probe_duration_seconds(absolute) or 0)
    except Exception:  # noqa: BLE001
        return 0.0
