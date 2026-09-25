"""**段级音频合成**（模型声 + 配音 → 一条音轨 ✓ 2026-09-24 接 ✓）。

## 为什么需要它（而不是让 ffmpeg 顶替）
本仓原有的单镜合成是 ``-map 0:v -map 1:a`` ✓ —— 那是**顶替** ✗：生成视频里自带的音轨
（H3 是**联合 AV** ✓）会被**直接丢掉** ✓✗。逆向口径要的是「**可选的混合**」✓，而且三支口径
**各不相同** ✗✗（判据全在 :mod:`app.services.engine.audio_mix` ✓，本层**只做 IO** ✓）：

* ``voice_mode="mix"``：模型声 × **0.6**（**写死** ✓）+ 配音 × 音量 ⇒ clamp **±1** ✓；
* 混**环境音**那支 ⇒ clamp **±0.95** ✓（留 0.05 防爆音 ✓）—— ⚠️ 别与上面统一 ✗；
* **纯模型声**那支 ⇒ **根本不削波** ✗（1.5 还是 1.5 ✓）。

## 两条硬口径（错了都会「听着还行、其实变了」✗）
1. ⭐ **长度守恒** ✗✗：输出恒 ``round(duration × sr)``（配音长了**截断** ✓、短了**补零** ✓）——
   音轨**不许把视频拖长** ✓✗；
2. ⭐ **不静默重采样 / 不转声道** ✗：两路采样率或声道数不一致 ⇒ **当场拒** ✓，并**把两个值都印出来** ✓
   （悄悄重采样会让音高与时长都变 ✓✗；正确做法是**加载时**就按段采样率解码 ✓）。

⚠️ 本层**不引 torch / numpy** ✗：wav 用标准库读 ✓ → 嵌套列表 → ``audio_mix`` ✓ → 标准库写 ✓
（compose 那条路本来就不该被 torch 绑住 ✓✗）。
"""
from __future__ import annotations

import array
import sys
import wave
from pathlib import Path
from typing import Any

from .engine import audio_mix as mix_mod

__all__ = ["SegmentAudioError", "mix_model_and_voice", "read_wav_tracks", "write_wav_tracks"]


class SegmentAudioError(RuntimeError):
    """本层自己的错（读不了 / 写不了 / 两路口径不一致 ✓）—— 一律**当场拒** ✗ 不静默处置 ✓。"""


def read_wav_tracks(path: str | Path) -> tuple[list[list[float]], int]:
    """wav → ``([声道][样本], 采样率)`` ✓（**标准库** ✓，值域 ``[-1, 1]`` ✓）。

    ⚠️ 只认 **16-bit PCM** ✗（本仓写的就是它 ✓；别的位宽**没有依据** ⇒ 拒 ✓ 不猜 ✗）。
    """
    target = Path(path)
    if not target.exists():
        raise SegmentAudioError(f"音频文件不存在 ✗：{target}（先确认路径 ✓）")
    try:
        with wave.open(str(target), "rb") as handle:
            channels = int(handle.getnchannels())
            width = int(handle.getsampwidth())
            rate = int(handle.getframerate())
            count = int(handle.getnframes())
            raw = handle.readframes(count)
    except (wave.Error, EOFError) as err:
        raise SegmentAudioError(f"{target.name} 不是可读的 wav ✗：{err}") from err
    if width != 2:
        raise SegmentAudioError(
            f"{target.name} 是 {width * 8} bit ✗ —— 本层只认 **16-bit PCM** ✓（不猜别的格式 ✗）")
    values = array.array("h")
    values.frombytes(raw[: count * channels * 2])
    if sys.byteorder == "big":                       # wav 是**小端** ✓（大端机上不换就是噪声 ✗）
        values.byteswap()
    tracks = [[values[index * channels + channel] / 32767.0 for index in range(count)]
              for channel in range(channels)]
    return tracks, rate


def write_wav_tracks(tracks: Any, sample_rate: int, path: str | Path) -> dict[str, Any]:
    """``[声道][样本]`` → wav ✓（标准库 ✓）⇒ 事实（少写了多少截样 ✓）。

    ⚠️ 越界值**钳到边界并如实回报** ✗（不是让它绕回成大噪声 ✓✗）——
    ⚠️ **三支的削波口径不同** ✓✗（见模块头 ✓）：本层只做**最后一道兜底** ✓，别拿它当混音口径 ✗。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rate = int(sample_rate)
    if rate <= 0:
        raise SegmentAudioError(f"采样率必须为正（收到 {sample_rate!r} ✗）")
    rows = [list(row) for row in tracks]
    if not rows or not rows[0]:
        raise SegmentAudioError("没有样本可写 ✗（空音轨不是音轨 ✓）")
    frames = len(rows[0])
    if len({len(row) for row in rows}) != 1:
        raise SegmentAudioError(f"各声道长度不一致 ✗：{[len(row) for row in rows]} ✓")
    clipped = 0
    out = array.array("h", bytes(2) * frames * len(rows))
    for channel, row in enumerate(rows):
        for index, value in enumerate(row):
            if value > 1.0 or value < -1.0:
                clipped += 1
            scaled = int(round(min(1.0, max(-1.0, float(value))) * 32767))
            out[index * len(rows) + channel] = scaled
    if sys.byteorder == "big":
        out.byteswap()
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(len(rows))
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(out.tobytes())
    return {"path": str(target), "channels": len(rows), "frames": frames, "sampleRate": rate,
            "seconds": round(frames / float(rate), 6), "clippedSamples": clipped}


def mix_model_and_voice(*, model_audio_path: str | Path, voice_path: str | Path,
                        output_path: str | Path, voice_mode: str = "mix",
                        voice_volume: float = 1.0, duration: float | None = None,
                        offset: float = 0.0, trim_start: float = 0.0, trim_end: float = 0.0,
                        trim_mode: str = "keep",
                        ambient_path: str | Path | None = None,
                        ambient_volume: float = mix_mod.DEFAULT_AMBIENT_VOLUME) -> dict[str, Any]:
    """按口径拼一条音轨并落盘 ✓ ⇒ 事实（含**时长与削波** ✓）。

    ``duration`` 省略 ⇒ **跟模型声一样长** ✓（= 视频长度 ✓）—— ⚠️ 不是「跟配音一样长」✗
    （那会把视频拖长/拖短 ✓✗）。
    """
    model_tracks, model_rate = read_wav_tracks(model_audio_path)
    voice_tracks, voice_rate = read_wav_tracks(voice_path)
    ambient_tracks: list[list[float]] | None = None
    ambient_rate: int | None = None
    if ambient_path:
        ambient_tracks, ambient_rate = read_wav_tracks(ambient_path)
    seconds = float(duration) if duration is not None else len(model_tracks[0]) / float(model_rate)
    try:
        mixed = mix_mod.mix_segment_audio(
            duration=seconds, sample_rate=model_rate, model_audio=model_tracks,
            model_rate=model_rate, voice_audio=voice_tracks, voice_rate=voice_rate,
            voice_mode=voice_mode, voice_volume=float(voice_volume), offset=float(offset),
            trim_start=float(trim_start), trim_end=float(trim_end), trim_mode=trim_mode,
            ambient_audio=ambient_tracks, ambient_rate=ambient_rate,
            ambient_volume=float(ambient_volume))
    except mix_mod.AudioMixError as err:
        # ⭐ 采样率/声道不一致就是从这里出来的 ✓ —— 原样抛出（它已经把两个值都印好了 ✓✗）
        raise SegmentAudioError(str(err)) from err
    if mixed is None:
        raise SegmentAudioError("``enabled=False`` 才返回 None ✓ —— 本层不该走到这里（要静音就别拼 ✓）")
    written = write_wav_tracks(mixed.samples, mixed.sample_rate, output_path)
    return {**written, "voiceMode": voice_mode, "voiceVolume": float(voice_volume),
            "modelRate": model_rate, "voiceRate": voice_rate, "targetSeconds": round(seconds, 6),
            "modelSeconds": round(len(model_tracks[0]) / float(model_rate), 6),
            "voiceSeconds": round(len(voice_tracks[0]) / float(voice_rate), 6)}
