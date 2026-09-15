"""剪映草稿导出 —— 与 ``services/jianying-draft.ts``（339 行）对齐。**阶段 1/2：助手层**。

目标：把一集的**合成分镜视频 + TTS 对白 + 字幕**导出为剪映可直接打开的草稿，
用户在剪映里继续调整字幕 / 配音 / 节奏 / 转场。

剪映草稿本质是一个 ``<name>.draft/`` 文件夹，必含：

* ``draft_content.json``：时间线与素材仓库（**引用式架构，UUID 关联，时间单位微秒**）
* ``draft_info.json``：草稿元信息
* ``draft_meta_info.json``：剪映打开时自动补全，**无需生成**

素材路径用「**复制进 ``.draft/media/`` 后以绝对路径引用**」策略：本机使用绝对路径直接生效
（主要场景，自托管工作台）；跨机器分发则整个 ``.draft`` 文件夹打包后由剪映自动搜索/手动重链。

⚠️ 兼容性：剪映格式为私有格式且随版本演进，这里按社区通用结构生成（微秒计时、
``canvas_config`` / ``materials`` / ``tracks`` 三段式）；若剪映大版本升级导致不兼容，
以剪映自建的草稿为基准微调字段即可。

⚠️ 四处保真点（本阶段已覆盖前三处）：

1. **``us_of`` 的 ``?? 5`` 是 nullish**：``duration`` 为 ``None`` 才用 5 秒，``0`` 会**原样保留**
   再被 ``max(1, ...)`` 夹到 1 微秒级下限（避免 0 时长素材让剪映报错）；
2. **``sanitize`` 有 40 字符上限**（草稿名过长会让剪映路径超限），空值回退 ``episode``；
3. **素材名是 ``video_001`` / ``audio_001``**（3 位补零）—— 与 ZIP 内路径一一对应，改名会让
   用户重链素材；
4. （阶段 2）``tracks`` 顺序与 ``materials`` 的 UUID 引用必须一一对应。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.engine import Connection

from ..core.config import get_data_root, get_storage_root
from ..core.models import episodes, storyboards, video_generations
from ..core.response import js_round
from .task_logger import log_task_start, log_task_success, log_task_warn

__all__ = [
    "build_jianying_draft",
    "copy_to_media",
    "sanitize",
    "segment",
    "text_segment",
    "to_media_path",
    "us_of",
]

#: 草稿名非法字符（与 TS 的 ``[\\/:*?"<>|\r\n\t]`` 一一对应）
_INVALID_NAME_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]')

#: 无扩展名时的兜底（按素材类型）
_FALLBACK_EXT = {"video": ".mp4", "audio": ".mp3"}


def to_media_path(path: Any) -> str | None:
    """相对/绝对媒体路径 → 本地绝对路径（**不存在返回 None**）。

    三段判定：绝对路径原样用；``static/`` 前缀挂**数据根**；其余挂**存储根**。
    """
    if not path:
        return None
    raw = str(path)
    if os.path.isabs(raw):
        absolute = raw
    elif raw.startswith("static/"):
        absolute = os.path.join(get_data_root(), raw)
    else:
        absolute = os.path.join(get_storage_root(), raw)
    return absolute if os.path.exists(absolute) else None


def copy_to_media(src: str, media_dir: str, index: int, kind: str) -> str | None:
    """把文件复制进 ``media`` 目录，返回复制后的绝对路径（**失败返回 None 且只告警**）。"""
    ext = os.path.splitext(src)[1] or _FALLBACK_EXT.get(kind, "")
    dest = os.path.join(media_dir, f"{kind}_{str(index).zfill(3)}{ext}")
    try:
        os.makedirs(media_dir, exist_ok=True)
        with open(src, "rb") as source, open(dest, "wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
        return dest
    except Exception as exc:  # noqa: BLE001 —— 与 TS 的 catch 一致：只告警不抛
        log_task_warn("JianYingDraft", "copy-failed", {"from": src, "error": str(exc)})
        return None


def us_of(duration: Any) -> int:
    """秒 → 微秒（**不小于 1**，避免 0 时长素材）。

    ⚠️ ``duration ?? 5`` 是 **nullish**：只有 ``None`` 才回退 5 秒；``0`` 保留 0 再被夹到 1。
    ⚠️ 用 ``js_round``（``Math.round`` 语义）—— Python ``round`` 是银行家舍入。
    """
    value = 5 if duration is None else duration
    return max(1, js_round(value * 1_000_000))


def sanitize(name: Any) -> str:
    """草稿名 sanitize（非法字符→``_``、去空白、**截断 40 字符**、空回退 ``episode``）。"""
    cleaned = _INVALID_NAME_CHARS.sub("_", str(name if name is not None else "")).strip()
    return cleaned[:40] or "episode"


def segment(material_id: str, start_us: int, duration_us: int) -> dict[str, Any]:
    """时间线片段（**视频与音频共用**）。"""
    return {
        "id": str(uuid.uuid4()).upper(),
        "material_id": material_id,
        "extra_material_refs": [],
        "source_timerange": {"start": 0, "duration": duration_us},
        "target_timerange": {"start": start_us, "duration": duration_us},
        "clip": {"alpha": 1, "scale": {"x": 1, "y": 1}, "rotation": 0},
        "speed": 1,
        "volume": 1,
        "visible": True,
    }


def text_segment(material_id: str, start_us: int, duration_us: int) -> dict[str, Any]:
    """字幕片段（⚠️ 没有 ``speed``/``volume``，但多一个 ``transform``）。"""
    return {
        "id": str(uuid.uuid4()).upper(),
        "material_id": material_id,
        "extra_material_refs": [],
        "source_timerange": {"start": 0, "duration": duration_us},
        "target_timerange": {"start": start_us, "duration": duration_us},
        "clip": {"alpha": 1, "scale": {"x": 1, "y": 1}, "rotation": 0},
        "transform": {"x": 0, "y": 0},
        "visible": True,
    }


# ===========================================================================
# 阶段 2：入口 + 单集草稿构建
# ===========================================================================

def build_jianying_draft(conn: Connection, drama_id: int,
                        episode_id: Any = None) -> list[dict[str, Any]]:
    """生成一集（或全剧）的剪映草稿。

    ``episode_id`` 给定时只导该集（且必须属于该剧）；缺省导出全剧（**每集一个草稿**）。

    ⚠️ 与 TS 一致：集查询**不过滤软删**；一集都没有时**抛错**（路由转 400）。
    """
    condition = and_(episodes.c.drama_id == drama_id)
    if episode_id:
        condition = and_(episodes.c.id == episode_id, episodes.c.drama_id == drama_id)
    rows = conn.execute(
        select(episodes).where(condition).order_by(episodes.c.episode_number)
    ).all()

    if not rows:
        raise ValueError("No episodes found for draft export")

    return [_build_draft_for_episode(conn, drama_id, episode) for episode in rows]


def _canvas_size(conn: Connection, storyboard_id: int) -> tuple[int, int]:
    """画布尺寸来源：该分镜的 ``video_generations`` 里**最后一条**有宽高的记录。

    ⚠️ TS 是 ``.all().filter(v => v.width && v.height).pop()``：**按 createdAt 升序后取最后一个**
    （即最新一条有尺寸的），缺省 ``1920x1080``。
    """
    rows = conn.execute(
        select(video_generations).where(video_generations.c.storyboard_id == storyboard_id)
        .order_by(video_generations.c.created_at)
    ).all()
    with_size = [row for row in rows if row.width and row.height]
    if not with_size:
        return 1920, 1080
    last = with_size[-1]
    return int(last.width), int(last.height)


def _build_draft_for_episode(conn: Connection, drama_id: int,
                             episode: Any) -> dict[str, Any]:
    """构建单集草稿目录（``<storageRoot>/jianying/<draftName>.draft/``）+ 文件清单。"""
    log_task_start("JianYingDraft", "build", {
        "dramaId": drama_id, "episodeId": episode.id,
        "episodeNumber": episode.episode_number,
    })

    storyboard_rows = conn.execute(
        select(storyboards).where(storyboards.c.episode_id == episode.id)
        .order_by(storyboards.c.storyboard_number)
    ).all()

    ready = [sb for sb in storyboard_rows if sb.composed_video_url]
    if not ready:
        raise ValueError(f"Episode {episode.episode_number} has no composed videos")

    draft_name = (f"drama{drama_id}_ep{str(episode.episode_number).zfill(2)}_"
                  f"{sanitize(episode.title or 'episode')}")
    draft_root = os.path.join(get_storage_root(), "jianying")
    os.makedirs(draft_root, exist_ok=True)
    draft_dir = os.path.join(draft_root, f"{draft_name}.draft")
    media_dir = os.path.join(draft_dir, "media")
    # ⚠️ 每次重建：先整目录删掉（原 TS 是 rmSync recursive+force）
    shutil.rmtree(draft_dir, ignore_errors=True)
    os.makedirs(media_dir, exist_ok=True)

    videos: list[dict[str, Any]] = []
    audios: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    cursor_us = 0
    video_index = 0
    audio_index = 0

    # 先复制素材，把「分镜 → 视频/音频」映射固定下来
    shot_timings: list[dict[str, Any]] = []

    # 取首个分镜的最新已完成视频生成记录作为画布尺寸来源（storyboards 无 resolution 字段）
    canvas_w, canvas_h = _canvas_size(conn, ready[0].id)

    for sb in ready:
        duration_us = us_of(sb.duration)

        # 视频素材：合成镜头
        video_id: str | None = None
        video_src = to_media_path(sb.composed_video_url)
        if video_src:
            copied = copy_to_media(video_src, media_dir, video_index, "video")
            video_index += 1
            if copied:
                video_id = str(uuid.uuid4()).upper()
                videos.append({"id": video_id, "path": copied, "duration": duration_us,
                               "width": canvas_w, "height": canvas_h,
                               "type": "video", "has_audio": True})

        # 音频素材：独立 TTS 对白
        audio_id: str | None = None
        audio_src = to_media_path(sb.tts_audio_url)
        if audio_src:
            copied = copy_to_media(audio_src, media_dir, audio_index, "audio")
            audio_index += 1
            if copied:
                audio_id = str(uuid.uuid4()).upper()
                audios.append({"id": audio_id, "path": copied,
                               "duration": duration_us, "type": "music"})

        # 字幕素材：对白文本
        if sb.dialogue:
            texts.append({
                "id": str(uuid.uuid4()).upper(),
                "content": sb.dialogue,
                "duration": duration_us,
                "font_size": 80,
                "x": canvas_w / 2,
                "y": canvas_h - 120,
                "alignment": 1,
                "type": "text",
                "font_name": "Microsoft YaHei",
                "is_bold": 0,
                "is_italic": 0,
                "outline": {"color": {"r": 0, "g": 0, "b": 0, "a": 1}, "size": 0.15,
                            "softness": 0},
                "stroke": {"color": {"r": 0, "g": 0, "b": 0, "a": 1}, "size": 0,
                           "softness": 0},
                "transform": {"x": 0, "y": 0},
                "fixed_scale": 1,
            })

        shot_timings.append({"startUs": cursor_us, "durationUs": duration_us,
                             "videoId": video_id, "audioId": audio_id})
        cursor_us += duration_us

    if not videos:
        raise ValueError(f"Episode {episode.episode_number}: no video materials copied")

    total_us = cursor_us
    fps = 30

    # 视频/音频/字幕段（分别按素材顺序与 shotTimings 对齐）
    video_segments: list[dict[str, Any]] = []
    audio_segments: list[dict[str, Any]] = []
    text_segments: list[dict[str, Any]] = []
    text_shot_index = 0
    for timing in shot_timings:
        if timing["videoId"]:
            video_segments.append(segment(timing["videoId"], timing["startUs"],
                                          timing["durationUs"]))
        if timing["audioId"]:
            audio_segments.append(segment(timing["audioId"], timing["startUs"],
                                          timing["durationUs"]))
        # ⚠️ 这里按**循环计数**取 texts（原 TS 如此）：中间某个镜头没有对白时，
        #    后续字幕会**错位到更早的 text 素材**上 —— 照抄，不擅自修
        if text_shot_index < len(texts):
            text_segments.append(text_segment(texts[text_shot_index]["id"],
                                              timing["startUs"], timing["durationUs"]))
        text_shot_index += 1

    content = {
        "canvas_config": {"width": canvas_w, "height": canvas_h, "ratio": "original"},
        "color_space": 0,
        "config": {
            "maintrack_adsorb": True,
            "video_mute": False,
            "subtitle_sync": True,
            "lyrics_sync": True,
            "material_save_mode": 0,
        },
        "duration": total_us,
        "fps": fps,
        "id": str(uuid.uuid4()).upper(),
        "keyframes": {},
        "materials": {
            "videos": videos,
            "audios": audios,
            "texts": texts,
            "speeds": [],
            "canvases": [],
            "audio_fades": [],
            "material_animation": [],
            "sound_channel_mapping": [],
            "vocal_separation": [],
        },
        "name": draft_name,
        "platform": {},
        "tracks": [
            {"id": str(uuid.uuid4()).upper(), "type": "video", "attribute": 0,
             "is_default_name": True, "name": "视频", "segments": video_segments},
            {"id": str(uuid.uuid4()).upper(), "type": "audio", "attribute": 0,
             "is_default_name": True, "name": "音频", "segments": audio_segments},
            *([{"id": str(uuid.uuid4()).upper(), "type": "text", "attribute": 0,
                "is_default_name": True, "name": "字幕", "segments": text_segments}]
              if text_segments else []),
        ],
        "version": 1,
    }

    info = {
        "created_at": 0,
        "edited": 0,
        "tm_draft_enter_edit_time": 0,
        "tm_draft_exit_edit_time": 0,
        "tm_draft_last_modified": 0,
        "tm_draft_last_modified_mtime": 0,
        "tm_draft_last_opened": 0,
        "tm_draft_removed": 0,
        "tm_draft_used": 0,
        "draft_fold_path": draft_dir,
        "draft_id": str(uuid.uuid4()).upper(),
        "draft_name": draft_name,
        "draft_removed": False,
        "draft_root_path": draft_dir,
        "draft_selection": [],
        "draft_selection_end": -1,
        "draft_selection_start": -1,
        "draft_timeline_materials": [],
        "draft_used_tracks": [],
        "extension": {},
        "folder_id": "",
        "is_imported_draft": False,
        "is_new_simple_draft": False,
        "is_short_video_mode": False,
        "need_download": False,
        "new_version": 0,
        "path": draft_dir,
        "project_has_audio": len(audios) > 0,
        "project_source": "",
        "purchase": {"pack": "", "version": 0},
        "recover_draft_info": {},
        "storage": "local",
        "use_speeches_to_score": False,
        "use_shortcut_engine": False,
    }

    # ⚠️ 剪映读的是**缩进 2 空格**的 JSON（TS 是 JSON.stringify(x, null, 2)）⇒ 这条
    #    `indent=2` 是**有意**的，已在 route_parity 的 json.dumps 白名单里
    with open(os.path.join(draft_dir, "draft_content.json"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(content, ensure_ascii=False, indent=2))
    with open(os.path.join(draft_dir, "draft_info.json"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(info, ensure_ascii=False, indent=2))

    # 文件清单（供 ZIP 打包）
    files: list[dict[str, str]] = [
        {"absPath": os.path.join(draft_dir, name),
         "zipPath": f"{draft_name}.draft/{name}"}
        for name in ("draft_content.json", "draft_info.json")
    ]
    for material in [*videos, *audios]:
        if material["path"].startswith(draft_dir):
            relative = os.path.relpath(material["path"], draft_dir).replace(os.sep, "/")
            files.append({"absPath": material["path"],
                          "zipPath": f"{draft_name}.draft/{relative}"})

    log_task_success("JianYingDraft", "built", {
        "dramaId": drama_id, "episodeId": episode.id, "draftName": draft_name,
        "durationUs": total_us, "videos": len(videos), "audios": len(audios),
        "texts": len(texts),
    })
    return {
        "draftDir": draft_dir,
        "draftName": draft_name,
        "durationUs": total_us,
        "videoCount": len(videos),
        "audioCount": len(audios),
        "textCount": len(texts),
        "files": files,
    }
