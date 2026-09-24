"""**段级音频合成**（纯数学 ✓ 零依赖 ✓ 2026-09-24 补 ✓，口径来自逆向 ✓）。

## 它解决什么
一段视频要配一段音 ✓：模型声 / 用户配音 / 环境音按规则拼起来 ✓。上游规则很清楚 ✓，
本模块把它落成**纯样本序列运算** ✓（不做 IO ✗、不碰 torch ✗ ⇒ 可自检 ✓；读写 wav 是 :mod:`media` 的事 ✓）。

## 规则（逐条对应上游 ✓）
* ⭐ **长度守恒** ✓：输出长度恒等于 ``round(duration × sample_rate)`` ✓（配音偏长**截断** ✓、
  偏短**补静音** ✓✗ —— 音频不能把视频拖长 ✓✗）；`max(1, …)` ✓ 保证至少一帧 ✓；
* ``enabled=False`` ⇒ 返回 **``None``** ✓（**整段静音 = 没有音轨** ✗，不是全零音轨 ✓✗）；
* 配音三件事 ✓：**起点偏移**（前置静音 ✓）、**裁剪**（``keep`` 保留选中区 ✓ / ``cut`` **中间挖掉** ✓）、
  模式 ``replace``（顶替模型声 ✓）或 ``mix`` ✓；
* ⭐ ``mix`` 的模型声系数是**写死的 0.6** ✓ + 配音 × ``voice_volume`` ✓；
* ⭐ 环境音**单独一路** ✓ ⇒ **循环对齐**到段长 ✓ ⇒ 叠加后 clamp **±0.95** ✓✗
  （**留 0.05 防爆音** ✓ —— 而 ``replace`` / ``mix`` 那两支是 clamp **±1** ✓、**纯模型声那支根本不 clamp** ✓✗
  —— 三支口径**不一样** ✗，别统一 ✗）。

## 不猜（本模块的边界 ✓）
* ⚠️ **不做重采样** ✗：上游是在**加载时**就按段采样率解码的 ✓ ⇒ 本模块要求**调用方先统一采样率** ✓
  （给了不一致的 ⇒ **报错** ✗ 不静默重采样 ✓ —— 静默重采样会让音高/时长悄悄变 ✓✗）；
* ⚠️ **不做声道转换** ✗：要求各轨**声道数一致** ✓（不一致 ⇒ 报错 ✓）；
* ⚠️ 音量只做**线性乘 + 削波** ✓（**不做响度归一化** ✗ —— 那是另一件事 ✓）；
* ⚠️ 「替换音轨 ≠ 对口型」✓：换音轨不会让嘴动对上 ✓✗（要口型必须让模型**听着**配音生成 ✓）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["AMBIENT_CEILING", "AudioMixError", "DEFAULT_AMBIENT_VOLUME", "MIX_MODEL_GAIN",
           "SegmentAudio", "fit", "mix_segment_audio", "offset_by", "samples_for_seconds", "trim"]

#: 环境音默认音量 ✓（上游口径 ✓）。
DEFAULT_AMBIENT_VOLUME = 0.25
#: ⭐ ``mix`` 模式下**模型声的固定压低系数** ✓（上游写死 0.6 ✓ —— 不是参数 ✗）。
MIX_MODEL_GAIN = 0.6
#: ⭐ 混**环境音**那支的削波上限 ✓（**0.95** ✓✗ 留 0.05 余量防爆音 ✓ —— 与另外两支不同 ✗）。
AMBIENT_CEILING = 0.95
#: ``replace`` / ``mix`` 两支的削波上限 ✓（±1 ✓）。
CEILING = 1.0

#: 裁剪模式 ✓：``keep`` 保留选中区 ✓ / ``cut`` 把选中区**从中间挖掉** ✓（上游 UI 的两个动作 ✓）。
TRIM_MODES: tuple[str, ...] = ("keep", "cut")


class AudioMixError(ValueError):
    """音频拼不起来 ✓ ⇒ 当场报 ✗（静默重采样/转声道会让音高、时长悄悄变 ✓✗）。"""


@dataclass(frozen=True)
class SegmentAudio:
    """一段音频 ✓（``samples[声道][样本]`` ✓ 不可变 ✓ —— 免得调用方改到中间产物 ✓）。"""

    samples: tuple[tuple[float, ...], ...]
    sample_rate: int

    @property
    def channels(self) -> int:
        return len(self.samples)

    @property
    def frames(self) -> int:
        return len(self.samples[0]) if self.samples else 0

    @property
    def seconds(self) -> float:
        return round(self.frames / float(self.sample_rate), 6) if self.sample_rate else 0.0

    def to_dict(self) -> dict[str, Any]:
        """只报**形状与时长** ✓（⚠️ 不回样本 ✗ —— 日志里塞几十万个浮点数没有任何价值 ✓✗）。"""
        return {"channels": self.channels, "frames": self.frames, "sampleRate": self.sample_rate,
                "seconds": self.seconds}


def samples_for_seconds(seconds: float, sample_rate: int) -> int:
    """秒 → **样本数** ✓（``max(1, round(…)`` ✓ —— 至少一帧 ✓ 不许 0 长度 ✓✗）。"""
    rate = int(sample_rate)
    if rate <= 0:
        raise AudioMixError(f"采样率必须为正（收到 {sample_rate!r} ✗）")
    return max(1, int(round(float(seconds or 0) * rate)))


def _as_tracks(samples: Any) -> list[list[float]]:
    """规整成 ``[声道][样本]`` ✓（单声道/多声道都接 ✓；空 ⇒ 报错 ✗）。"""
    if not samples:
        raise AudioMixError("音频是空的 ✗（空轨没法拼 ✓ —— 要静音请用 ``enabled=False`` ✓）")
    tracks = [list(track) for track in samples]
    widths = {len(track) for track in tracks}
    if len(widths) != 1:
        raise AudioMixError(f"各声道长度不一致 ✗：{[len(track) for track in tracks]} ✓")
    return tracks


def _check_pair(tracks: list[list[float]], sample_rate: int, other: list[list[float]],
                other_rate: int, *, label: str) -> None:
    if int(sample_rate) != int(other_rate):
        raise AudioMixError(
            f"{label}的采样率与本段**不一致** ✗：{other_rate} vs {sample_rate} ✓"
            f" ⇒ ⚠️ 本仓**不静默重采样** ✗（那会让音高/时长悄悄变 ✓✗）："
            f"请在**加载时**就按本段采样率解码 ✓")
    if len(other) != len(tracks):
        raise AudioMixError(
            f"{label}的声道数与本段**不一致** ✗：{len(other)} vs {len(tracks)} ✓"
            f" ⇒ 本仓**不静默转声道** ✗，先在上游统一 ✓")


def fit(samples: Any, frames: int, *, loop: bool = False) -> list[list[float]]:
    """**对齐到指定样本数** ✓：长了**截断** ✓、短了**补零** ✓（``loop=True`` ⇒ 循环填充 ✓，环境音用 ✓）。"""
    tracks = _as_tracks(samples)
    wanted = max(1, int(frames))
    out: list[list[float]] = []
    for track in tracks:
        if len(track) >= wanted:
            out.append(track[:wanted])
        elif loop and track:
            repeated: list[float] = []
            while len(repeated) < wanted:
                repeated.extend(track)
            out.append(repeated[:wanted])
        else:
            out.append(track + [0.0] * (wanted - len(track)))
    return out


def trim(samples: Any, *, sample_rate: int, start: float = 0.0, end: float = 0.0,
         mode: str = "keep") -> list[list[float]]:
    """按秒裁剪 ✓：``keep`` 保留 ``[start, end)`` ✓ / ``cut`` **挖掉** ``[start, end)`` 并接起来 ✓。

    ⚠️ ``end <= start`` ⇒ **取到结尾** ✓（上游口径 ✓ —— 不是「空区间」✗）。
    """
    if mode not in TRIM_MODES:
        raise AudioMixError(f"裁剪模式 {mode!r} 不认识 ✗ ⇒ 只能是 " + " / ".join(TRIM_MODES) + " ✓")
    tracks = _as_tracks(samples)
    rate = int(sample_rate)
    total = len(tracks[0])
    first = max(0, min(total, int(round(float(start or 0) * rate))))
    last = int(round(float(end or 0) * rate))
    if last <= first:
        last = total
    last = max(first, min(total, last))
    if mode == "keep":
        return [track[first:last] for track in tracks]
    return [track[:first] + track[last:] for track in tracks]


def offset_by(samples: Any, *, seconds: float, sample_rate: int) -> list[list[float]]:
    """**起点偏移** ✓：前面**补静音** ✓（= 「配音从段内第 X 秒开始播」✓ —— ⚠️ 不是往后接 ✗）。"""
    tracks = _as_tracks(samples)
    delay = max(0, int(round(float(seconds or 0) * int(sample_rate))))
    return [([0.0] * delay) + track for track in tracks] if delay else tracks


def _clamp(tracks: list[list[float]], ceiling: float) -> list[list[float]]:
    low = -float(ceiling)
    high = float(ceiling)
    return [[min(high, max(low, value)) for value in track] for track in tracks]


def mix_segment_audio(*, duration: float, sample_rate: int, model_audio: Any = None,
                      model_rate: int | None = None, voice_audio: Any = None,
                      voice_rate: int | None = None, voice_mode: str = "replace",
                      voice_volume: float = 1.0, enabled: bool = True, trim_start: float = 0.0,
                      trim_end: float = 0.0, trim_mode: str = "keep", offset: float = 0.0,
                      ambient_audio: Any = None, ambient_rate: int | None = None,
                      ambient_volume: float = DEFAULT_AMBIENT_VOLUME) -> SegmentAudio | None:
    """拼一段音频 ✓ ⇒ :class:`SegmentAudio` ✓；⭐ ``enabled=False`` ⇒ **``None``** ✓（没有音轨 ✓✗）。

    优先级与上游一致 ✓：**有配音 ⇒ 配音说了算**（``replace`` 顶替 / ``mix`` 混合 ✓）；
    **没配音但有环境音** ⇒ 叠在模型声上 ✓；都没有 ⇒ 就模型声 ✓（⚠️ 这一支**不削波** ✗，与另外两支不同 ✓）。
    """
    if not enabled:
        return None
    rate = int(sample_rate)
    frames = samples_for_seconds(duration, rate)
    if voice_mode not in ("replace", "mix"):
        raise AudioMixError(
            f"配音模式 {voice_mode!r} 不认识 ✗ ⇒ 只能是 replace（顶替 ✓）/ mix（混合 ✓）"
            f"—— ⚠️ 两者都与**对口型无关** ✗：换音轨不会让嘴动对上 ✓✗")
    model = fit(model_audio, frames) if model_audio else None
    if model is not None and model_rate is not None:
        _check_pair(model, rate, model, int(model_rate), label="模型声")

    if voice_audio:
        if voice_rate is None:
            raise AudioMixError("给了配音就**必须给采样率** ✗（本仓不猜 ✗ —— 猜错会让音高/时长悄悄变 ✓✗）")
        voice = _as_tracks(voice_audio)
        _check_pair(model if model is not None else voice, rate, voice, int(voice_rate),
                    label="配音")
        voice = trim(voice, sample_rate=rate, start=trim_start, end=trim_end, mode=trim_mode)
        voice = offset_by(voice, seconds=offset, sample_rate=rate)
        voice = fit(voice, frames)
        if voice_mode == "mix":
            base = model if model is not None else [[0.0] * frames for _ in voice]
            _check_pair(base, rate, voice, rate, label="配音")
            merged = [[base[channel][index] * MIX_MODEL_GAIN + voice[channel][index] * float(voice_volume)
                       for index in range(frames)] for channel in range(len(voice))]
            return SegmentAudio(tuple(tuple(row) for row in _clamp(merged, CEILING)), rate)
        scaled = [[value * float(voice_volume) for value in track] for track in voice]
        return SegmentAudio(tuple(tuple(row) for row in _clamp(scaled, CEILING)), rate)

    if ambient_audio:
        if ambient_rate is not None:
            ambient_tracks = _as_tracks(ambient_audio)
            _check_pair(model if model is not None else ambient_tracks, rate, ambient_tracks,
                        int(ambient_rate), label="环境音")
        ambient = fit(ambient_audio, frames, loop=True)
        base = model if model is not None else [[0.0] * frames for _ in ambient]
        _check_pair(base, rate, ambient, rate, label="环境音")
        merged = [[base[channel][index] + ambient[channel][index] * float(ambient_volume)
                   for index in range(frames)] for channel in range(len(ambient))]
        # ⭐ 这一支的削波上限是 **0.95** ✗（留 0.05 余量 ✓）；⚠️ 别与上面两支统一 ✗
        return SegmentAudio(tuple(tuple(row) for row in _clamp(merged, AMBIENT_CEILING)), rate)

    if model is None:
        raise AudioMixError("既没模型声也没配音/环境音 ✗ ⇒ 没东西可拼 ✓（要静音请用 enabled=False ✓）")
    return SegmentAudio(tuple(tuple(row) for row in model), rate)
