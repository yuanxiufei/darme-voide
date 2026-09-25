"""FFmpeg 单镜头合成（移植自 ``backend/src/services/ffmpeg-compose.ts``，187 行）。

把一个镜头的**视频 + TTS 对白音频 + 烧录字幕**合成成片片段（``composed_video_url``）。
与 ``merge``（整集拼接）区分：merge 把多个已合成镜头串接为完整剧集。

流程：

1. 标记 ``status=compose_processing`` 并清空 ``composedVideoUrl``；
2. **TTS**：台词可解析且非「可忽略」时 —— 已有 ``ttsAudioUrl`` 且文件在就用它；
   否则按「说话人 → 剧组角色 → 音色」匹配拿 voiceId（回落 ``alloy``），调 ``generate_tts``
   并把结果写回 ``storyboards.tts_audio_url``；
3. **字幕**：生成 SRT（**只有一条**：``00:00:00,500 --> 00:00:{min(duration-1,59)},000``），
   写 ``static/subtitles/<uuid>.srt`` 并回写 ``subtitle_url``；
4. **ffmpeg**：``-c:v libx264 -preset fast -crf 23``，有音频则
   ``-map 0:v -map 1:a -c:a aac -shortest``，否则 ``-an``；输出 ``static/composed/<uuid>.mp4``，
   成功写 ``compose_completed``，失败写 ``compose_failed``。

⚠️ 三处保真点：

* 字幕滤镜要**探测**（``ffmpeg -filters`` 里有没有 ``subtitles``，结果**记忆化**）；
  不支持时**只是跳过烧录**，不报错（也不删 SRT）；
* 台词解析**直接复用** ``storyboard_helpers.parse_dialogue_for_tts``（两处 IGNORE 正则逐字相同）；
* ffmpeg 走**子进程**（Node 侧是 ``fluent-ffmpeg``）—— 参数顺序按 fluent-ffmpeg 的产出排列，
  不经 shell 传参（避免路径里的空格/冒号被解析）。
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select, update

from ..core.config import get_storage_root
from ..core.db import engine
from ..core.models import characters, episodes, storyboards
from ..core.response import now
from .character_match import match_character_by_speaker_name
from .file_storage import get_absolute_path
from .segment_audio import SegmentAudioError, mix_model_and_voice
from .storyboard_helpers import parse_dialogue_for_tts
from .task_logger import log_task_error, log_task_progress, log_task_start, log_task_success
from .tts_generation import generate_tts

__all__ = ["compose_storyboard", "supports_subtitle_filter"]

#: 混音中间产物目录名 ✓（放在数据根下的 ``static/mixed`` ✓ —— 与 ``composed`` 同级 ✓）：
#: ⚠️ 用**固定目录**而不是临时目录 ✗：出问题时能**拿到那几路音轨**复盘 ✓（排查"音轨不对"只能靠听 ✓）。
_MIX_SUBDIR = "static/mixed"

#: 字幕滤镜支持探测的**记忆化**缓存（``None`` = 还没探测过）
_subtitle_filter_support: bool | None = None


def _to_abs_path(relative_path: str) -> str:
    """相对路径 → 绝对路径（语义与 TS 的 ``toAbsPath`` 相同，见 ``file_storage``）。"""
    return get_absolute_path(relative_path)


def supports_subtitle_filter() -> bool:
    """当前 ffmpeg 是否带 ``subtitles`` 滤镜（结果记忆化；探测失败按「不支持」处理）。"""
    global _subtitle_filter_support
    if _subtitle_filter_support is not None:
        return _subtitle_filter_support
    try:
        import subprocess

        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        _subtitle_filter_support = bool(re.search(r"\bsubtitles\b", out.stdout or ""))
    except Exception:  # noqa: BLE001 —— 没装 ffmpeg 也走这里
        _subtitle_filter_support = False
    return _subtitle_filter_support


async def _run_ffmpeg(args: list[str]) -> None:
    """跑一次 ffmpeg；非 0 退出时抛错（消息形如 fluent-ffmpeg 的 ``ffmpeg exited with code``）。"""
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = (stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {tail[-2000:]}")


def _write(storyboard_id: int, **values: Any) -> None:
    """**独立事务**写回分镜（对齐 Node 的 ``db.update(...).run()`` 逐语句自动提交）。

    ⚠️ 这里刻意**不用调用方传入的连接**：否则失败路径写 ``compose_failed`` 时会被
    调用方的事务一起回滚（请求依赖是「异常即 rollback」），前端就永远看不到失败态。
    """
    values.setdefault("updated_at", now())
    with engine.begin() as conn:
        conn.execute(
            update(storyboards).where(storyboards.c.id == storyboard_id).values(**values)
        )


def _fetch(storyboard_id: int):
    with engine.begin() as conn:
        return conn.execute(
            select(storyboards).where(storyboards.c.id == storyboard_id)
        ).first()


async def _extract_audio_wav(video_path: str, target: str) -> str:
    """从视频里**抽出音轨**成 16-bit PCM wav ✓（**不重采样、不混声道** ✗ —— 见 :func:`engine.media.extract_wav` ✓）。

    ⚠️ 单独成一个函数是为了**可测** ✓（自检里换掉它就能验混音接线 ✓，不必真装 ffmpeg ✓）。
    ⚠️ 视频**没有音轨** ⇒ 这里会抛 ✓ —— 混音那一路**不当成"那就照旧顶替"** ✗（静默降级 ✓✗）。
    """
    from .engine import media as engine_media   # 局部 import ✓：只有混音这一条路才需要引擎的 ffmpeg 封装 ✓

    await asyncio.to_thread(engine_media.extract_wav, video_path, target)
    return target


async def _build_mixed_track(video_path: str, voice_path: str, *, voice_mode: str,
                             voice_volume: float, tag: str) -> str:
    """把**视频自带音轨**与配音按口径合成一条 ✓ ⇒ 混合后 wav 的**绝对路径** ✓。

    ⚠️ 失败**不静默退回顶替** ✗✗：调用方是**显式**要求混音的 ✓ ⇒ 拼不出来就**明确失败并把原因说清** ✓
    （悄悄换成顶替 ⇒ 用户以为混了、模型声其实没了 ✓）。
    """
    target_dir = Path(get_storage_root()) / _MIX_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)
    model_wav = target_dir / f"{tag}-model.wav"
    mixed_wav = target_dir / f"{tag}-mixed.wav"
    await _extract_audio_wav(video_path, str(model_wav))
    report = await asyncio.to_thread(
        mix_model_and_voice, model_audio_path=str(model_wav), voice_path=voice_path,
        output_path=str(mixed_wav), voice_mode=voice_mode, voice_volume=voice_volume)
    log_task_progress("ComposeTask", "mixed-model-audio", {
        "mixed": report["path"], "voiceMode": report["voiceMode"],
        "targetSeconds": report["targetSeconds"], "modelSeconds": report["modelSeconds"],
        "voiceSeconds": report["voiceSeconds"], "clippedSamples": report["clippedSamples"],
    })
    return str(mixed_wav)


async def compose_storyboard(storyboard_id: int, *, mix_model_audio: bool = False,
                             voice_mode: str = "mix", voice_volume: float = 1.0) -> str:
    """合成单个镜头，返回**相对数据根**的成片路径（``static/composed/<uuid>.mp4``）。

    ⚠️ 本函数**自己管连接**（每次读写一个短事务），不接收外部连接 —— 见 ``_write``。

    ``mix_model_audio=True``（**默认 false ⇒ 与原有行为一字不差** ✗）⇒ 把视频自带的音轨
    （H3 是**联合 AV** ✓ 生成的）与配音**混合**成一条 ✓ 再喂给 ffmpeg ✓ —— 见
    :mod:`app.services.segment_audio` 的三支削波口径 ✓✗（顶替会让模型声**直接消失** ✓）。
    """
    sb = _fetch(storyboard_id)
    if sb is None:
        raise ValueError(f"Storyboard {storyboard_id} not found")
    if not sb.video_url:
        raise ValueError(f"Storyboard {storyboard_id} has no video")

    _write(storyboard_id, status="compose_processing", composed_video_url=None)

    log_task_start("ComposeTask", "storyboard-compose", {
        "storyboardId": storyboard_id,
        "storyboardNumber": sb.storyboard_number,
        "episodeId": sb.episode_id,
    })

    video_path = _to_abs_path(sb.video_url)
    audio_path: str | None = None
    subtitle_path: str | None = None
    parsed = parse_dialogue_for_tts(sb.dialogue)

    try:
        # 1. TTS 音频（有对白时）
        if not parsed["ignorable"]:
            if sb.tts_audio_url:
                existing = _to_abs_path(sb.tts_audio_url)
                if Path(existing).exists():
                    audio_path = existing

            if not audio_path:
                voice_id = "alloy"
                with engine.begin() as conn:
                    episode = conn.execute(
                        select(episodes).where(episodes.c.id == sb.episode_id)
                    ).first()
                    found = None
                    if parsed["speaker"] and episode is not None:
                        rows = conn.execute(
                            select(characters).where(
                                characters.c.drama_id == episode.drama_id,
                                characters.c.deleted_at.is_(None),
                            )
                        ).all()
                        # ⚠️ 这个匹配函数靠 **getattr(c, 'name')** 取名字 ⇒ 必须传 Row，
                        #    传 dict 会静默匹配不到（`getattr(dict,'name')` 是 None）
                        found = match_character_by_speaker_name(rows, parsed["speaker"])
                    if found is not None and getattr(found, "voice_style", None):
                        voice_id = found.voice_style

                pure_dialogue = parsed["pureText"]
                if pure_dialogue:
                    log_task_progress("ComposeTask", "generate-inline-tts", {
                        "storyboardId": storyboard_id, "voiceId": voice_id,
                        "textPreview": pure_dialogue[:40],
                    })
                    with engine.begin() as conn:
                        tts_path = await generate_tts(conn, {
                            "text": pure_dialogue,
                            "voice": voice_id,
                            # `ep?.audioConfigId ?? undefined` —— null 走默认配置
                            "configId": (episode.audio_config_id if episode is not None else None),
                        })
                    audio_path = _to_abs_path(tts_path)
                    _write(storyboard_id, tts_audio_url=tts_path)

        # 2. 字幕文件（SRT）
        if not parsed["ignorable"]:
            srt_dir = Path(get_storage_root()) / "subtitles"
            srt_dir.mkdir(parents=True, exist_ok=True)
            srt_filename = f"{uuid4()}.srt"
            subtitle_path = str(srt_dir / srt_filename)

            duration = sb.duration or 10
            end_seconds = min(duration - 1, 59)
            srt = (
                "1\n"
                f"00:00:00,500 --> 00:00:{str(end_seconds).zfill(2)},000\n"
                f"{parsed['pureText']}\n"
            )
            Path(subtitle_path).write_text(srt, encoding="utf-8")

            srt_relative = f"static/subtitles/{srt_filename}"
            _write(storyboard_id, subtitle_url=srt_relative)

        # 2b. **可选**：把模型自带的音轨与配音**混合** ✓（默认关 ⇒ 与原有行为**一字不差** ✗）
        if mix_model_audio and audio_path:
            try:
                audio_path = await _build_mixed_track(
                    video_path, audio_path, voice_mode=voice_mode, voice_volume=voice_volume,
                    tag=f"{storyboard_id}-{uuid4().hex[:8]}")
            except (SegmentAudioError, OSError) as err:
                # ⭐ 显式要求混音却拼不出来 ⇒ **失败** ✗（不悄悄退回顶替 ✓✗：那等于模型声没了 ✓）
                raise ValueError(
                    f"混音失败（显式要求 ``mixModelAudio`` ✓）：{type(err).__name__}: {err}"
                    f"　—— ⚠️ 不静默退回「配音顶替」✗（那会让模型自带的环境音**直接消失** ✓✗）"
                ) from err

        # 3. FFmpeg 合成
        output_dir = Path(get_storage_root()) / "composed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{uuid4()}.mp4"
        output_path = str(output_dir / output_filename)

        args = ["ffmpeg", "-i", video_path]
        if audio_path:
            args += ["-i", audio_path]

        filters: list[str] = []
        if subtitle_path and supports_subtitle_filter():
            # ⚠️ 转义顺序不能变：先统一斜杠，再转义冒号与单引号（Windows 盘符的 `D:` 必须转义）
            escaped_path = (
                subtitle_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            )
            force_style = "FontSize=20\\,PrimaryColour=&HFFFFFF&\\,OutlineColour=&H000000&\\,Outline=2"
            filters.append(f"subtitles=filename='{escaped_path}':force_style='{force_style}'")
        elif subtitle_path:
            log_task_progress("ComposeTask", "subtitle-filter-unavailable", {
                "storyboardId": storyboard_id, "subtitlePath": subtitle_path,
            })

        if filters:
            args += ["-vf", ",".join(filters)]

        args += ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
        if audio_path:
            args += ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]
        else:
            args += ["-an"]
        args.append(output_path)

        await _run_ffmpeg(args)

        composed_relative = f"static/composed/{output_filename}"
        _write(storyboard_id, composed_video_url=composed_relative, status="compose_completed")
        log_task_success("ComposeTask", "storyboard-compose", {
            "storyboardId": storyboard_id,
            "storyboardNumber": sb.storyboard_number,
            "output": composed_relative,
        })
        return composed_relative
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        # ⚠️ 独立事务写失败态：若借用调用方连接，这里会被它的 rollback 一起抹掉
        _write(storyboard_id, status="compose_failed", composed_video_url=None)
        log_task_error("ComposeTask", "storyboard-compose", {
            "storyboardId": storyboard_id, "error": str(err),
        })
        raise
