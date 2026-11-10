"""**落盘**（帧 → 真 mp4 / 音频 → 真 wav ✓）—— 引擎里"产物变文件"的那一步。

## 为什么要用外部 ffmpeg（而不是纯 Python 编 H.264 ✗）

H.264/H.265 编码器**不在 Python 生态里** ✓（PyAV 也要带 ffmpeg 库 ✓）。而本机的现实是：
**ffmpeg 9.0.1 已在 PATH 上** ✓，且本仓其它地方（`services/color_grade.py` 等 ✓）已经在用它 ✓
⇒ 沿用同一件事 ✓，不引入新依赖 ✓。

## 两条硬要求（本模块的纪律 ✓）

1. **产物必须可被独立复核** ✓ —— 写完立刻能用 :func:`probe`（ffprobe ✓）读出**宽/高/fps/时长/帧数** ✓。
   自检就是这么做的 ✓：不是"函数没报错"就算成功 ✗，而是"文件里确实有那几帧" ✓。
2. **值域必须显式** ✓ —— 张量是 ``[-1, 1]``（扩散模型的常规输出 ✓）还是 ``[0, 1]`` ✓ **由调用方说** ✓
   （默认 ``-1..1`` ✓）。猜错值域 ⇒ 画面全白/全黑 ✗ 而且**不报错** ✗（最难查的那种 ✓）。

⚠️ 另一个真实坑（已显式处理 ✗）：``yuv420p`` 要求**宽高为偶数** ✓ ⇒ 奇数尺寸直接**明确报错** ✓，
**不**偷偷裁掉一行像素 ✗（那会改变产物尺寸却没人知道 ✓）。
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

__all__ = ["MediaError", "extract_wav", "ffmpeg_version", "have_ffmpeg", "load_image_tensor",
           "load_video_tensor", "probe", "write_image", "write_video", "write_wav"]


class MediaError(RuntimeError):
    """落盘失败 ✓（缺工具 / 参数非法 / 编码器报错 ✓ —— 都带上**可行动**的信息 ✓）。"""


def _tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise MediaError(
            f"找不到 {name} ✗ ⇒ 先装 ffmpeg（Windows: `winget install Gyan.FFmpeg` ✓；"
            f"Linux: `apt install ffmpeg` ✓）。本仓的色校正/合成等功能也依赖它 ✓。\n"
            f"⚠️ 若确认**装过**却仍报这个 ⇒ 十有八九是**当前进程的 PATH 里没有它** ✗"
            f"（用 `where.exe {name}` 复核 ✓）；Windows 上 WinGet 装出来的 "
            f"`…\\Microsoft\\WinGet\\Links\\{name}.exe` 是**应用别名（重解析点）** ✗ —— "
            f"它在某些进程里会 `lexists=True` 但 `exists=False` ✓✗（于是 `shutil.which` 找不到 ✓）"
            f"⇒ 把**真实**的 `…\\WinGet\\Packages\\…\\bin` 目录**前置**到 PATH ✓"
            f"（⚠️ **前置**才治本 ✗：追加只让 `which` 找得到 ✓，而**裸名** spawn 仍会先命中坏别名 ✓✗"
            f"⇒ 那一侧报的是 `WinError 448 不受信任的装入点` ✓）。")
    return found


def have_ffmpeg() -> bool:
    """ffmpeg 是否可用 ✓（`describe()` 会报这个 ✓）。

    ⚠️ 判据就是 **PATH 里能不能解析出这两个可执行文件** ✓（本仓**不猜安装路径** ✗ ——
    猜路径在不同机器上会给出「看着能跑」的假绿 ✓✗）。装过却报 `False` 的排查见 :func:`_tool` ✓。
    """
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def ffmpeg_version() -> str:
    if not have_ffmpeg():
        return ""
    try:
        done = subprocess.run([_tool("ffmpeg"), "-version"], capture_output=True, text=True,
                              timeout=20, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return ""
    first = (done.stdout or "").splitlines()[:1]
    return first[0].strip() if first else ""


def _normalize(frames: Any, value_range: str) -> tuple[Any, int, int, int, int]:
    """张量 → ``(uint8 numpy, batch, frames, height, width)`` ✓（值域与形状**都显式校验** ✓）。"""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    shape = tuple(frames.shape)
    if len(shape) == 4:            # (3, T, H, W) ⇒ 补 batch ✓
        frames = frames.unsqueeze(0)
        shape = tuple(frames.shape)
    if len(shape) != 5 or shape[1] != 3:
        raise MediaError(f"帧张量形状应为 (B, 3, T, H, W) ✓，收到 {shape} ✗")
    batch, _channels, count, height, width = shape
    if height % 2 or width % 2:
        # ⚠️ 明确报错而不是悄悄裁 ✓（裁了会改变产物尺寸却没人知道 ✗）
        raise MediaError(f"尺寸必须为偶数（yuv420p 要求 ✓）：收到 {width}×{height} ✗")
    kind = str(value_range or "-1..1").replace(" ", "")
    if kind in ("-1..1", "[-1,1]"):
        scaled = (frames.clamp(-1.0, 1.0) + 1.0) * 0.5
    elif kind in ("0..1", "[0,1]"):
        scaled = frames.clamp(0.0, 1.0)
    else:
        raise MediaError(f"未知值域 {value_range!r}；可用：-1..1 / 0..1 ✓（猜错就是全白或全黑 ✗）")
    array = (scaled * 255.0).round().to("cpu", dtype=torch.uint8)
    # (B, 3, T, H, W) → (B, T, H, W, 3) ✓ 与 ffmpeg 的 rgb24 行序一致 ✓
    return array.permute(0, 2, 3, 4, 1).numpy().astype(np.uint8), batch, count, height, width


def _metadata_pairs(metadata: Any) -> list[tuple[str, str]]:
    """``{k: v}`` → ``[(k, v), …]`` ✓（**归一化键名** ✓ + 换行压平 ✓）。

    ⚠️ 键名归一化是必需的 ✓✗：``-metadata`` 的键是容器规范的一部分 ✓，带空格/中文的键名
    在 mp4 muxer 上会**静默丢弃** ✓✗ ⇒ 只留 ``[0-9A-Za-z_.-]`` ✓（值原样保留 ✓）。
    """
    if not metadata:
        return []
    pairs: list[tuple[str, str]] = []
    for key, value in dict(metadata).items():
        name = re.sub(r"[^0-9A-Za-z_.-]", "_", str(key)).strip("_")
        if not name or value is None:
            continue
        pairs.append((name, str(value).replace(chr(10), " ").replace(chr(13), " ")))
    return pairs


def _png_info(metadata: Any) -> Any:
    """``{k: v}`` → :class:`PIL.PngImagePlugin.PngInfo` ✓（写进 PNG 的 iTXt 块 ✓ 可复读 ✓）。"""
    if not metadata:
        return None
    from PIL import PngImagePlugin  # noqa: PLC0415

    info = PngImagePlugin.PngInfo()
    for key, value in dict(metadata).items():
        name = str(key).strip()
        if not name or value is None:
            continue
        info.add_text(name, str(value))
    return info


def frames_from_duration(facts: dict[str, Any], *, fallback_fps: int = 24) -> int | None:
    """``nbFrames`` 缺失时按「时长 × 帧率」**估**帧数 ✓；估不出来返 ``None`` ✓（**不猜 0** ✗✗）。

    ⚠️ ``nb_frames`` 只在封装**写了 index** 时才有 ✓（分片 mp4 / 某些编码器 / 重封装中间态全没有 ✓），
    ``probe`` 就如实给 ``None`` ✓。而「读不到 nbFrames」曾被两处**当成没事**：一处静默跳过帧数核对
    （``torch_backend.write`` ✗）、一处直接抛「帧数与时长都读不出来」✓（``seedvr2_target`` ✓）。
    ⇒ 抽成本函数，两处共用**同一个估法** ✓✗（各写一遍迟早一处改了另一处不改 ✓）。

    ⚠️ 这是**估计值** ✓：调用方必须自己写明「估的」✓✗ —— 别把它当 ``nbFrames`` 的等价物 ✓。
    ``seedvr2_target`` 估完写进 ``warnings`` ✓，``torch_backend.write`` 估完标
    ``framesVerified="duration×fps"`` 并按 ±1 帧判 ✓。
    """
    try:
        seconds = float(facts.get("durationSeconds") or 0)
        fps = float(facts.get("fps") or fallback_fps or 0)
    except (TypeError, ValueError):
        return None
    if seconds <= 0 or fps <= 0:
        return None
    return int(round(seconds * fps))


#: ⭐ 帧率对账的**容差** ✓✗：请求的 fps 是整数 ✓，而盘上 ``r_frame_rate`` 可能是 ``"2997/100"`` ✓
#:   ⇒ 差 0.01 以内算同一个帧率 ✓（再大就不是舍入了 ✗）。
_FPS_TOLERANCE = 0.01


def compare_fps(requested: Any, on_disk: Any) -> tuple[str | None, str | None]:
    """「**请求的**帧率」与「**盘上真的**帧率」是不是同一个 —— **三档** ✓✗（2026-11-10 补 ✗）。

    ⚠️ 这一格此前**根本不存在** ✓✗：``probe`` 老老实实读回了 ``fps`` ✓（盘上那个 ✓），
    可 ``write_video`` 手里也有请求的 ``fps`` ✓ —— 两个数**没有任何一格对过账** ✓。
    于是「ffmpeg 按我请求的帧率封的」与「它悄悄按别的帧率封的」在回执里**完全同形** ✓，
    而每一格都"有值、能复核" ✓，看不出哪里缺了东西 ✓✗。

    为什么必须单独表态（不止是`多一层保险`）✓：
    * ``videoSeconds`` / ``-t`` / 音轨 ``apad`` 的截断与补长**全按请求的 fps 算** ✓；
    * 而 ``durationSeconds`` 是**盘上真实的** ✓；
    * 两者一旦不等 ⇒ **音画对不上**（画面按 A 帧率播、音轨按 B 算的秒数收尾 ✓）、
      成片时长也不是它该有的长度 ✓ —— 可这些格子在回执里**全都"有值"** ✓✗。

    三档（与 ``framesVerified`` 同一个规矩：读不到**不许**当成通过 ✗）：
    * ``"exact"`` ✓ —— 盘上帧率就是请求的那个 ✓（容差 :data:`_FPS_TOLERANCE` ✓）；
    * ``"drift"`` ✗ —— 真被改了 ✓（同时给一句 :func:`compare_fps` 生成的 ``fpsNote`` ✓）；
    * ``None`` ✗ —— **盘上没读到**帧率 ✓ ⇒ 「有没有被改」**不知道** ✓（不是"一致" ✗）。

    返回 ``(verdict, note)``：``note`` 是可直接贴进回执的中文 ✓（``"drift"`` 时才有 ✓）。
    """
    if requested is None or on_disk is None:
        return None, None
    try:
        want, got = float(requested), float(on_disk)
    except (TypeError, ValueError):
        return None, None
    if want <= 0 or got <= 0:
        return None, None
    if abs(got - want) <= _FPS_TOLERANCE:
        return "exact", None
    return "drift", (
        f"⚠️ 盘上帧率与请求的**不一致** ✗：请求 {want:g} fps ✓、封进文件的是 {got:g} fps ✗"
        f"（差 {round(got - want, 4):g} ✓）—— 画面按 {got:g} fps 播 ✓，"
        f"而混流的 ``-t`` 是按请求的 {want:g} fps 算秒数的 ✓ ⇒ "
        f"**音画会不同步**、成片时长也不是该有的长度 ✗；"
        f"``videoSeconds`` 是按**请求** fps 算的 ✓、``durationSeconds`` 是**盘上真实**的 ✓，"
        f"两者不等就是这件事 ✓")


def frame_verdict(facts: dict[str, Any], *, fallback_fps: int = 24,
                  tolerance: int = 1) -> dict[str, Any]:
    """「盘上到底几帧、这个数**是怎么来的**」—— **一处**判定 ✓✗（2026-11-08 补 ✗）。

    此前「落盘帧数核过没有」这件事在 ``torch_backend.write`` 里**自己写了一遍** ✓，而 ``chain``
    那条成片**一遍都没有** ✓✗（它只在张量层守恒 ✓、不核编码器有没有丢帧 ✓）—— 同一件事两个地方
    各写一半，最典型的下场是一处改了另一处不改 ✓。判定收在这里，两处都读它 ✓。

    三层表态（顺序就是可信度 ✗）：
    1. ``nb_frames`` 有 ⇒ ``verified="nb_frames"`` ✓（封装 index 直读 ✓，**容差 0** ✓）；
    2. 没有但有 ``时长 × fps`` ⇒ ``verified="duration×fps"`` ✓（**估的** ✓✗，容差 ±1 帧 ——
       容器时长是 float 秒 ✓，``round`` 会上下抖一帧 ✓；差更多就不是舍入 ✓）；
    3. 两样都读不到 ⇒ ``verified=None`` ＋ ``matches=None`` ✓ —— ⚠️ ``matches is None`` 表示
       **没有可比读数** ✓✗，**任何调用方都不许把它读成「通过」** ✗（这正是「读数缺失被当成读数说
       没有/没问题」那一族 ✓）。
    """
    try:
        requested = facts.get("requestedFrames")
        written = int(requested) if requested is not None else None
    except (TypeError, ValueError):
        written = None
    read = facts.get("nbFrames")
    if read is not None:
        try:
            encoded, verified = int(read), "nb_frames"
        except (TypeError, ValueError):
            encoded, verified = None, None
        tol = 0
    else:
        encoded = frames_from_duration(facts, fallback_fps=fallback_fps)
        verified = "duration×fps" if encoded is not None else None
        tol = int(tolerance)
    if written is None or encoded is None:
        matches = None
    else:
        matches = abs(int(encoded) - int(written)) <= tol
    return {"framesWritten": written, "encodedFrames": encoded,
            "framesVerified": verified, "framesMatch": matches,
            "framesTolerance": tol}


def write_video(frames: Any, path: str | Path, *, fps: int = 24, value_range: str = "-1..1",
                crf: int = 18, preset: str = "veryfast", pix_fmt: str = "yuv420p",
                audio: str | Path | None = None,
                metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """帧张量 → **真 mp4** ✓；返回可复核的事实（宽/高/fps/时长/字节 ✓，来自 :func:`probe` ✓）。

    ``audio``：给了就把**这条音轨混进同一个 mp4** ✓（H3 双流的正解 ✓ —— 见下），
    不给就仍是纯视频轨 ✓（``-an`` ✓，默认路径一字未变 ✓）。
    ``metadata``：写进容器 ✓（任务号/提示词摘要等 ✓，事后 ``ffprobe -show_format`` 可读回 ✓）。

    ⚠️ **音画长度天然不等**（H3 的音频塔一段只吐 ~0.2 秒 ✓ 而画面 1.6 秒 ✓）⇒ 混流时
    ``apad`` 把音轨补静音到画面长度 ✓ 再用 ``-t <画面秒数>`` 硬性截断 ✓：
    两路等长 ✓、容器时长 = 画面时长 ✓，画面一帧不少 ✓。
    （实测**不能**用 ``-shortest`` ✓✗：``apad`` 补出来的音轨没有尽头 ⇒ 12 帧 0.5 秒的画面出了 0.992 秒的成片 ✓。）
    音轨**原样另存**的 wav 不受影响 ✓（要干净音频仍读那条 ✓）。

    ⚠️⚠️ 但 ``-t`` 是**单向**的 ✗✗（2026-11-08 补 ✗）：音轨比画面**短**时是补静音 ✓（无损 ✓），
    比画面**长**时是**截掉** ✗（内容真丢 ✗）。此前两种都不留痕 ✓✗：``probe`` 只能读**混完之后**
    那条音轨 ✓（被收过尾 ✓ ⇒ 天然等于画面时长 ✓），于是「音轨被剪掉 0.5 秒」与「音轨本来
    0.5 秒」在回执里**完全同形** ✓。现在单列 ``audioSourceSeconds``（源 wav 实测 ✓）/
    ``audioTrimSeconds``（被截 ✗）/ ``audioPadSeconds``（被补 ✓），真截过再给
    ``audioTruncated`` ＋ 一句人话 ✓（阈值＝一帧 ✓，小于它的是帧对齐噪声 ✓ 不喊狼 ✗）。

    ⚠️⚠️ 而「源音轨到底多长」**读不出来**是第三种情况 ✗✗（2026-11-08 补 ✗）：它不是「没截」✗，
    而是**不知道** ✓。``wave`` 只认 PCM ✓（float32 / ``WAVE_FORMAT_EXTENSIBLE`` 这些合法 wav
    它会 ``unknown format`` ✗）⇒ 先 ``wave``、再 **ffprobe 兜底** ✓、再**全解码兜底** ✓；
    三路都没有才给 ``audioLengthVerified: None`` ＋ ``audioLengthNote`` ✓，此时
    ``audioSourceSeconds`` / ``audioTrimSeconds`` / ``audioPadSeconds`` **一格都不写** ✓（不编数 ✗）。
    ⭐ 而「不知道」本身还要分两句 ✓✗（2026-11-09 补 ✗）：``"too_long"``（源超过全解码上限 ⇒
    **这次没量** ✓）说「没有量到底」✓；其余（三路全失败 ✓）才说「读不出来」✗ ——
    把"你自己不让解"写成"读不出来"，用户会去查编码/换封装 ✓，白查一场 ✗。
    ⭐ ffprobe 那一路**只认流自己给的时长** ✗✗（2026-11-09 补 ✗）：``probe`` 在流时长缺失时会回落到
    容器时长 ✓（**带封装余量**：1.0 秒的 mp3 封进 mkv ⇒ 1.045、webm ⇒ 1.008 ✓），
    那是"最长那条流"而不是音轨 ⇒ 拿它做对账会喊一个**假的**「被截掉 0.045 秒」✗。
    ⭐⭐ 但 mkv / webm **不是只能认"不知道"** ✓✗：删掉它做不到的不是"容量"而是"没试"✗ ——
    第四级 **全解码**（``-f null`` 只量不存 ✓）量出来的才是内容长度 ✓
    （同一批源：容器 1.045 ✗ / 包级 span 1.045 ✗ / **解码 1.000** ✓，见
    :func:`_decode_audio_seconds` 的对照表 ✓）⇒ 只有**连解码都不成**（源太长没全解 ✓ /
    ffmpeg 起不来 ✓）才落到「不知道」档 ✓。

    ⭐ 帧率同样要对账 ✓✗（2026-11-10 补 ✗）：回执里的 ``fps`` 是 :func:`probe` 读回来的
    **盘上**帧率 ✓，请求值是参数 ``fps`` ✓ —— 此后另有 ``fpsRequested`` /
    ``fpsVerified``（``exact`` / ``drift`` / ``None`` ✓，判据见 :func:`compare_fps`）✓。
    缺了这一格时，「按 24 fps 封好 ✓」与「它按 12 fps 封了 ✗」同形 ✓，而
    ``durationSeconds``（真实 ✗）与 ``videoSeconds``（按**请求** fps 算 ✓）不等这件事
    没有任何一格说 ✓。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array, batch, count, height, width = _normalize(frames, value_range)
    if batch != 1:
        # 多段拼接属于后续功能 ✓ —— 现在**明确拒绝**而不是只取第一段 ✗（静默丢弃最坏 ✓）
        raise MediaError(f"暂只支持 batch=1（收到 {batch} ✓；多段拼接见后续计划 ✗）")

    command = [
        _tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(int(fps)), "-i", "-",
    ]
    #: 源音轨时长 ✓（没用音轨 ⇒ ``None`` ✓ 不编 0 ✗）＋ **这个数是怎么来的** ✓（``wave`` / ``ffprobe`` ✓）
    audio_seconds: float | None = None
    audio_how: str | None = None
    if audio is None:
        command.append("-an")                    # 纯视频轨 ✓（音频要单独落 wav 时走这条路 ✓）
    else:
        source = Path(audio)
        if not source.exists():
            raise MediaError(f"要混入的音轨不存在 ✗：{source}（先落 wav 再混 ✓）")
        #: ⭐⭐ **先把「源音轨到底多长」量下来** ✗✗（2026-11-08 补 ✗）：下面那句 ``-t`` 只保画面
        #:   ✓ —— 源音轨**比画面长**时多出来的那截被**直接截掉** ✓✗，而 ``probe`` 只能读**混完之后**
        #:   的那条音轨（＝画面长度 ✓）⇒ 回执里 ``audioMuxed: True`` ＋ ``audioDurationSeconds: 0.5``
        #:   与「音轨本来就只有 0.5 秒」**完全同形** ✓✗ ⇒ 「用户那段音频被剪掉一半」这件事
        #:   在任何一格都不存在 ✓。
        audio_seconds, audio_how, audio_miss = _audio_length(source)
        # ⚠️ 长度收尾**必须用 ``-t``（画面秒数）** ✓✗ —— 实测 ``apad`` ＋ ``-shortest`` 并不以画面收尾 ✓✗：
        #    ``apad`` 补出来的音轨没有尽头 ✓，容器时长会跟着音轨涨（12 帧 0.5 秒的画面 ⇒ 成片 0.992 秒 ✓），
        #    播放器在尾部多停半秒 ✓。``-t`` 是**确定的**：画面 12 帧 @24 ⇒ 0.5 秒 ✓，
        #    音轨被 apad 补到 0.5 秒再被 ``-t`` 截断 ✓ ⇒ 两路等长 ✓、容器时长 = 画面时长 ✓。
        command += [
            "-i", str(source),
            "-map", "0:v:0", "-map", "1:a:0",
            "-af", "apad",                       # 音轨补静音到画面长度 ✓（不补 ⇒ 音轨先结束 ✗）
            "-c:a", "aac", "-b:a", "192k",
            "-t", f"{float(count) / float(int(fps)):.6f}",
        ]
    pairs = _metadata_pairs(metadata)
    command += [
        "-c:v", "libx264", "-pix_fmt", pix_fmt, "-crf", str(int(crf)), "-preset", preset,
    ]
    for name, value in pairs:
        command += ["-metadata", f"{name}={value}"]
    # ⚠️ ``use_metadata_tags`` 不是可选优化 ✓✗：mp4 muxer **只认**自己那几张固定标签
    #    （title/comment/artist… ✓），其余键**直接丢掉且不报错** ✓✗（实测 ``engineTask`` 消失 ✓）。
    #    带上这个 movflag 才会落进 ``mdta`` 键表 ✓ —— 元数据"写了但其实没写"正是本轮要消灭的静默失败 ✓。
    command += ["-movflags", "+faststart+use_metadata_tags" if pairs else "+faststart",
                str(target)]
    try:
        done = subprocess.run(command, input=array.tobytes(), capture_output=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as err:
        raise MediaError(f"ffmpeg 调用失败：{type(err).__name__}: {err}") from err
    if done.returncode:
        tail = (done.stderr or b"")[-600:].decode("utf-8", "replace")
        raise MediaError(f"ffmpeg 返回 {done.returncode} ✗：{tail}")
    if not target.exists() or target.stat().st_size == 0:
        raise MediaError(f"ffmpeg 返回 0 但没写出文件（{target} ✗）—— 这种**静默失败**必须当失败处理 ✓")
    report = probe(target)
    report["requestedFrames"] = int(count)
    report["valueRange"] = value_range
    report["audioMuxed"] = bool(audio is not None)
    #: ⭐⭐ 「**核过帧数**」/「**读不到所以没核**」必须分开交出去 ✗✗（2026-11-08 补 ✗）：
    #:   ``nb_frames`` 只在封装**写了 index** 时才有 ✓（分片 mp4 / 某些编码器 / 重封装中间态全没有 ✓），
    #:   ``probe`` 就如实给 ``None`` ✓。而旧的闸门写 ``nbFrames is not None`` ⇒ 读不到就**整条跳过** ✓，
    #:   回执照样写「真 mp4 ✓（ffprobe 可复核 ✓）」✓✗ ⇒ 「逐帧核过了」与「根本没核」**同形** ✓。
    #:   现在把判定（直读 / 估 / 未核 ✓）**记在报告里** ✓，调用方（``torch_backend.write`` ✓ /
    #:   ``chain.run_chain`` ✓）只需看 ``framesMatch``，不必各自重算 ✓。
    report.update(frame_verdict(report, fallback_fps=int(fps)))
    #: ⭐⭐ 「帧率是不是我请求的那个」也是必须单独表态的一格 ✗✗（2026-11-10 补 ✗）：
    #:   ``report["fps"]`` 是 :func:`probe` 从 **``r_frame_rate`` 读回来的盘上读数** ✓，
    #:   请求值是参数 ``fps`` ✓ —— 两者此前**一格都没有对过账** ✓✗，于是
    #:   「按 24 fps 封的 ✓」与「它按 12 fps 封了 ✗（画面慢一倍 ✓）」在回执里同形 ✓，
    #:   而每一格都"有值、ffprobe 可复核" ✓，看不出缺了什么 ✓。
    #:   ⚠️ ``fps`` 这个名字**只指盘上** ✓（下游别拿它当"我请求的"✗）⇒ 请求值另起
    #:   ``fpsRequested`` ✓；``fpsVerified`` 三档（``exact`` / ``drift`` / ``None``＝没读到 ✗）。
    report["fpsRequested"] = int(fps)
    report["fpsVerified"], fps_note = compare_fps(int(fps), report.get("fps"))
    if report["fpsVerified"] == "drift":
        report["fpsNote"] = fps_note
        report["fpsDrift"] = round(float(report["fps"]) - float(int(fps)), 4)
    if pairs:
        # ⚠️ 送了不等于写进去了 ✓✗（见上面 movflag 的注释 ✓）⇒ 用 ffprobe 读回来的 tags 当场对账 ✓
        landed = {str(key).lower() for key in (report.get("tags") or {})}
        missing = [name for name, _ in pairs if name.lower() not in landed]
        report["metadataKeys"] = [name for name, _ in pairs]
        report["metadataDropped"] = missing
    if audio is not None:
        # ⚠️ ffmpeg 返回 0 不等于音轨真在里面 ✓（map 写错只会**静默丢一路** ✓✗）⇒ 用 ffprobe 当场核 ✓
        if not report.get("hasAudio"):
            raise MediaError(
                f"要求混入音轨，但产物里没有音频流 ✗：{target}（ffprobe 读不到 audio ✓）"
                f"—— 静默丢音轨必须当失败处理 ✓")
        report["audioSource"] = str(Path(audio))
        #: ⚠️ 上一版这里是 ``report["audioDurationSeconds"] = report.get("audioDurationSeconds")``
        #:   ✓✗ —— **自己赋给自己** ✓：一行读起来像"在核对音轨时长"的代码 ✓，实际**什么都没干** ✓。
        #:   而 ``probe`` 给的 ``audioDurationSeconds`` 是**混完之后**那条音轨的长度 ✓（被 ``-t`` 收过
        #:   尾 ✓）⇒ 它天然等于画面时长 ✓，拿它"核对"永远自洽 ✓✗。
        #: ⭐⭐ 现在改成**两侧对账** ✓✗：源音轨（``audioSourceSeconds`` ✓）vs 画面（``videoSeconds`` ✓）。
        video_seconds = float(count) / float(int(fps))
        report["videoSeconds"] = round(video_seconds, 4)
        #: ⭐⭐ 「源音轨多长」这个读数**是怎么来的**必须交出去 ✓✗（2026-11-08 补 ✗）：
        #:   ``"wave"`` 直读 wav 头 ✓ / ``"ffprobe"`` 兜底读到 ✓ / ``"decoded"`` 全解码量到 ✓ /
        #:   ``None`` ＝**三路都没读到** ✗。
        #:   ⚠️ ``None`` **不等于**「没有截断」✗✗ —— 旧代码在这里整段跳过 ✓（``if audio_seconds is not None:``
        #:   之后没有 else ✓），于是「音轨被截掉 1 秒」与「音轨有没有被截不知道」在回执里**同形** ✓，
        #:   而 ``torch_backend`` 那句人话照旧写着「音轨补静音到画面长度 ✓」✓✗ —— 又是一次
        #:   「读数缺失被当成读数说没问题」✓（与帧数那个是同一个病根 ✓）。
        report["audioLengthVerified"] = audio_how
        if audio_seconds is not None:
            report["audioSourceSeconds"] = round(float(audio_seconds), 4)
            #: ``-t`` 是**单向**的 ✓：长了就**截**（内容真丢 ✗）、短了就 apad **补静音** ✓。
            #: 两种都不报 ⇒ 「音轨被剪掉 0.5 秒」与「音轨本来 0.5 秒」在回执里**同形** ✓✗。
            #: 阈值取**一帧**（``1/fps`` ✓）：更小的差是浮点/帧对齐噪声 ✓，报出来只会喊狼 ✗。
            trim = max(0.0, float(audio_seconds) - video_seconds)
            pad = max(0.0, video_seconds - float(audio_seconds))
            report["audioTrimSeconds"] = round(trim, 4)
            report["audioPadSeconds"] = round(pad, 4)
            if trim > 1.0 / max(1, int(fps)):
                #: ⚠️ 这是**内容丢失** ✓（不是口径问题 ✓）⇒ 单独给一格 + 一句人话 ✓，
                #:   让上游（``torch_backend.write`` ✓）能把它贴进 ``note`` 而不用自己再算一遍 ✓。
                report["audioTruncated"] = True
                report["audioTruncatedNote"] = (
                    f"音轨比画面长 {round(trim, 3)} 秒 ✓ ⇒ 混流时被 "
                    f"``-t {video_seconds:.3f}`` **截掉** ✗（mp4 里那条只剩画面长度 ✓；"
                    f"源 wav 原样未动 ✓ —— 要全长音频读 ``{Path(audio)}`` ✓）")
        else:
            #: ⚠️ 三路都读不出时长 ⇒ 「**有没有**被截」这次**不知道** ✗✗（不是「没截」✗）。
            #:   一格都不写会让 ``torch_backend`` 只看到「没有 ``audioTruncated``」✓ ⇒ 安静放过 ✓✗；
            #:   所以这里留一句人话 ＋ ``audioLengthVerified: None`` ✓，让它照着念 ✓。
            #: ⭐ 「不知道」要分两种说法 ✓✗（2026-11-09 补 ✗）：``"too_long"`` 是**量过但没量到底**
            #:   ✓（源超过全解码上限 ✓ ⇒ 是"这次没试" ✗，不是"读不出来" ✗）；其余才是三路都失败 ✓。
            #:   混成一句会让「2 小时的播客」看起来像"文件坏了" ✗ —— 用户照着去查编码就白查 ✓✗。
            #: ⭐⭐ 所以把**原因**也交出去 ✓（``audioLengthMiss`` ∈ ``"too_long"`` / ``"unreadable"`` ✓）：
            #:   ``audioLengthNote`` 是写给人看的中文长句 ✓，上游拿它去**判断**只能字符串匹配 ✗
            #:   （``torch_backend`` 那句人话正是这么写死成"读不出来"的 ✓✗）；给一个机器码 ✓，
            #:   上游按它分两句 ✓，两档就不会在半路被拍平 ✓。
            report["audioLengthMiss"] = audio_miss
            if audio_miss == "too_long":
                report["audioLengthNote"] = (
                    f"⚠️ 源音轨时长**没有量到底** ✗：容器声明就超过全解码上限 "
                    f"{_MAX_DECODE_SECONDS:g} 秒 ⇒ **这次没做全解码** ✓（不是读不出来 ✓）"
                    "⇒ 「音轨有没有被 ``-t`` 截掉」**不知道** ✗ —— "
                    "这不是「没截断」✓✗；要确认请拿 ``audioSource`` 那条单独量 ✓")
            else:
                report["audioLengthNote"] = (
                    "⚠️ 源音轨时长**读不出来** ✗（标准库 ``wave``、ffprobe 与全解码都没给出可用时长 ✓）"
                    "⇒ 「音轨有没有被 ``-t`` 截掉」这次**不知道** ✗ —— "
                    "这不是「没截断」✓✗；要确认请拿 ``audioSource`` 那条 wav 单独量 ✓")
    return report


def _audio_length(path: Path, *,
                  max_decode_seconds: float | None = None) -> tuple[float | None, str | None, str | None]:
    """源音轨时长 ＋ **这个数是怎么来的** ✓✗ ＋ **没读到是为什么** ✓✗
    （2026-11-08 补 ✗，2026-11-09 加第四级 ✓，同日补第三项 ✓）。

    四级，最后才认「不知道」：
    1. ``wave`` 直读 wav 头 ✓（不启子进程 ✓，绝大多数 wav 走这条）；
    2. **ffprobe 兜底** ✓✗ —— ``wave`` 只认 PCM ✓，float32 / ``WAVE_FORMAT_EXTENSIBLE``
       这类**完全合法的 wav** 它会 ``wave.Error: unknown format`` ✗（Python 3.11 的实测行为 ✓），
       而 ffmpeg 照混不误 ✓ ⇒ 只靠 ``wave`` 会把「能读到的」说成「读不出来」✓✗（与帧数那次
       「只认 ``nb_frames``」是同一个病根的另一面 ✓：**一次没读到就宣布没有** ✗）；
    3. **全解码兜底** ✓✗（2026-11-09 补 ✗）—— mkv / webm 这类封装**根本没有**流的 ``duration`` ✗，
       前两级都读不到；而 ffmpeg 混流时本来就要解这条音轨 ✓ ⇒ 让 :func:`_decode_audio_seconds`
       解一遍量样本时长 ✓（见那支函数里的对照表 ✓：容器声明 1.045 ✗ vs 解码 1.000 ✓）；
    4. 连解码都失败 / 源太长没全解 ⇒ ``(None, None, <为什么>)`` ✓ 交给调用方**如实说不知道** ✓。

    ⚠️ 返回值里的 ``str`` 不是装饰 ✓：``"wave"``（wav 头直读 ✓）/ ``"ffprobe"``（流时长 ✓）/
    ``"decoded"``（全解码 ✓）三者可信度与成本都不同 ✓，回执里 ``audioLengthVerified``
    必须能分辨 ✓✗。

    ⭐⭐ 第三项 ``miss`` 同样是"必须说出来"的一格 ✓✗（2026-11-09 补 ✗）：``None``＝读到了 ✓、
    ``"too_long"``＝**量过但没量到底** ✓（源长到不值得全解码 ⇒ 这次没试 ✗）、
    ``"unreadable"``＝三路都失败 ✓（文件坏了 / 没有音轨 ✓）。此前两者都塌成 ``(None, None)``
    ⇒ 回执里只有一句"读不出来" ✗，用户会去查编码 ✗ —— 而真正的原因是"你自己不让解" ✓。
    """
    seconds = _wav_seconds(path)
    if seconds is not None:
        return seconds, "wave", None
    seconds = _probe_audio_seconds(path)
    if seconds is not None:
        return seconds, "ffprobe", None
    missed: list[str] = []
    seconds = _decode_audio_seconds(path, max_seconds=max_decode_seconds, missed=missed)
    if seconds is not None:
        return seconds, "decoded", None
    return None, None, (missed[0] if missed else "unreadable")


#: ⭐ 「静音」的绝对门限 ✓✗（≈ −80 dBFS ✓）：比 16-bit 量化底噪（1/32767 ≈ 3.05e-5 ✓）
#:   高约 3 dB ✓，所以 1 个 LSB 的抖动不会被判成"有声音"✗，而**真的全零**一定被抓到 ✓。
#:   用绝对门限而不是"相对画面/相对峰值"✗ —— 后者在整轨都很轻时会把轻声判成静音 ✗。
_PEAK_SILENT = 1e-4


#: ⭐ 全解码兜底的**成本闸门** ✓✗：``-f null`` 会真把音轨解一遍 ✓（不落盘 ✓、不编码 ✓），
#:   音频解码通常 >100× 实时 ✓（1 分钟 ≈ 半秒 ✓）；但用户丢进来一条 2 小时的播客就不再划算 ✗。
#:   超过这个预估时长（看**容器时长**，只用它决定"要不要解" ✓、**不当读数** ✗）就**不解码** ✓，
#:   由 :func:`_audio_length` 落到「不知道」档 ✓ —— 与"静默降级"的区别是它会**说出来** ✓。
_MAX_DECODE_SECONDS = 600.0


def _decode_audio_seconds(path: Path, *, max_seconds: float | None = None,
                          missed: list[str] | None = None) -> float | None:
    """**全解码**一遍音频，按解码出来的样本时长量 ✓✗（第四级；2026-11-09 补 ✗）。

    ⚠️ 为什么非要有这一级 ✗✗：mkv / webm **根本不给**流的 ``duration`` ✗，容器那条又**带封装
    余量** ✗（实测 1.0 秒的 mp3 原样封进去 ⇒ 容器 **1.045** ✓，与它"长得一样"的**包级 span**
    也恰好是 1.045 ✗ —— 所以「数包」这条路**也是错的** ✓，包级 span 就是容器声明，不是内容长度 ✗）。
    只有**解码**口径对得上内容：

    ================  ==========  ===========  ===============
    源（1.0 秒内容）   容器声明      包级 span     **全解码**
    ================  ==========  ===========  ===============
    mp3 → mkv        1.045 ✗     1.045 ✗      1.025 ✓
    mp3 → webm       1.008 ✗     1.008 ✗      **1.000** ✓
    s16 wav          1.0         1.0          1.000 ✓
    mp3              1.0449 ✗    1.0449 ✗     **1.000** ✓
    ================  ==========  ===========  ===============

    （mkv 那条 1.025 是 mp3 delay 帧的残余 ✓ —— 它确实就是**混流会喂进去**的样本量 ✓，
    所以对账用它比用 1.045 对得多一个数量级 ✓。）

    实现：``-f null`` 只量不存 ✓，``-progress pipe:1`` 把 ``out_time_us``（**微秒** ✓）交到 stdout ✓
    —— 不去解析人类可读的 ``time=`` ✗（那要 ``-v info`` 且格式随版本变 ✓✗）。
    超过 ``max_seconds``（按容器时长预估 ✓）**不解码** ⇒ ``None`` ✓（由上层明说"没全解" ✓，不编数 ✗）。

    ⚠️ ``max_seconds`` 默认取 ``None`` 而不是直接写 ``_MAX_DECODE_SECONDS`` ✗✗：函数默认值在
    ``def`` 那一刻就绑死了 ✓，之后 ``patch`` 模块常量对它**毫无作用** ✗ ⇒ 判据里想用"上限 0.001 秒"
    触发"源太长"这条分支就做不到 ✓✗（只能造一条真 2 小时的文件 ✗）。改成运行时读 ✓。
    同理 ``missed`` 是出参 ✓：超限时 append ``"too_long"`` ✓，让 :func:`_audio_length` 能把它
    作为第三项交出去 ✓（不是在回执里编一个时长 ✗）。
    """
    if max_seconds is None:
        max_seconds = _MAX_DECODE_SECONDS
    estimate: float | None = None
    try:
        raw = probe(path).get("durationSeconds")
        estimate = float(raw) if raw else None
    except (MediaError, OSError, subprocess.SubprocessError, TypeError, ValueError):
        estimate = None
    if estimate is not None and estimate > max_seconds:
        if missed is not None:
            #: 这是**没量** ✗（不是量不出来 ✓）：容器自己说源超过上限 ⇒ 上层必须换一句话说 ✓。
            missed.append("too_long")
        return None                          # 源太长 ⇒ 不解 ✗（上层说"没全解" ✓）
    timeout = max(60.0, (estimate * 4.0) if estimate else 300.0)
    try:
        done = subprocess.run(
            [_tool("ffmpeg"), "-v", "error", "-nostats", "-i", str(path),
             "-map", "0:a:0", "-f", "null", "-progress", "pipe:1", "-"],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode:
        return None
    for line in reversed((done.stdout or "").splitlines()):   # 最后一条有效戳＝总长 ✓
        if not line.startswith("out_time_us="):
            continue
        stamp = line.split("=", 1)[1].strip()
        if stamp.isdigit() and int(stamp) > 0:
            return int(stamp) / 1_000_000.0
    return None


def _wav_seconds(path: Path) -> float | None:
    """读**源 wav** 的真实时长 ✓（标准库 ``wave`` ✓，不起子进程 ✓）。

    ⚠️ 读不出来就返 ``None`` 而**不是猜一个 0** ✗✗：``0`` 会让下游算出"音轨比画面短 1.6 秒、补了静音"✓✗
    —— 那是**编出来的事实** ✓。``None`` 只表示"这次不知道" ✓，下游按"没读数"处理 ✓
    （真正的"不知道"由 :func:`_audio_length` 收尾 ✓）。
    """
    try:
        with wave.open(str(path), "rb") as handle:
            rate = float(handle.getframerate() or 0)
            frames = float(handle.getnframes() or 0)
            return (frames / rate) if rate > 0 else None
    except (OSError, EOFError, wave.Error, ValueError):
        return None


def _probe_audio_seconds(path: Path) -> float | None:
    """用 **ffprobe** 读源文件里的音频流时长 ✓（``wave`` 读不了的合法 wav 走这条 ✓）。

    ⚠️⚠️ 只认 ``audioDurationSource == "stream"`` ✗✗（2026-11-09 补 ✗）：``probe`` 在
    **流时长缺失**时会回落到**容器时长** ✓，而容器时长是"最长那条流 ＋ 封装余量" ✗
    （实测 1.0 秒的 mp3 封进 mkv ⇒ 1.045、webm ⇒ 1.008 ✓，两种封装都不给流 ``duration`` ✗）⇒
    拿它当**源音轨长度**与画面做对账，就会报一个**假的**「音轨被截掉 0.045 秒」✗✗。
    ⇒ 回落值一律按**没读到**处理 ✓（``None`` ✓，由 :func:`_audio_length` 落到"不知道"档 ✓，
    ``audioLengthVerified: None`` ＋ 人话点名 ✓）—— **宁可说不知道，也不拿容器数当音轨数** ✓✗。
    读不出 / 非正值 / ffprobe 起不来 ⇒ ``None`` ✓（**不猜 0** ✗）。
    """
    try:
        facts = probe(path)
        if facts.get("audioDurationSource") != "stream":
            return None                      # 容器回落值 ≠ 源音轨长度 ✗
        seconds = float(facts.get("audioDurationSeconds") or 0)
    except (MediaError, OSError, subprocess.SubprocessError, TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def probe(path: str | Path) -> dict[str, Any]:
    """用 **ffprobe** 读出真实产物事实 ✓（自检据此断言"文件里确实有那几帧" ✓）。"""
    target = Path(path)
    if not target.exists():
        raise MediaError(f"文件不存在：{target} ✗")
    command = [_tool("ffprobe"), "-v", "error", "-print_format", "json",
               "-show_streams", "-show_format", str(target)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120,
                          encoding="utf-8", errors="replace")
    if done.returncode:
        raise MediaError(f"ffprobe 返回 {done.returncode} ✗：{(done.stderr or '')[-300:]}")
    payload = json.loads(done.stdout or "{}")
    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    duration = payload.get("format", {}).get("duration")
    frames = video.get("nb_frames")
    # ⚠️ 没有音频流 ⇒ ``音频时长`` 必须是 ``None`` ✓✗（回落成容器时长就是**把画面上报成音轨** ✓✗）
    #: ⭐⭐ 而「**有**音频流、流时长却缺」时旧写法**静默**回落到容器时长 ✓✗（2026-11-09 补 ✗）——
    #:   容器时长是**最长那条流** ✓（mp4 里通常就是画面 ✓）、且**带封装余量** ✗（实测：1.0 秒的
    #:   mp3 原样封进 mkv ⇒ ``format.duration=1.045``、封进 webm ⇒ ``1.008`` ✓；两种封装都
    #:   **不给**流的 ``duration`` ✗、也不给 ``duration_ts`` ✗，只有 ``tags.DURATION`` 那串
    #:   同样带余量的值 ✓✗）。于是回执里 ``audioDurationSeconds`` 与「音轨真就那个长度」
    #:   **完全同形** ✓ —— 而 :func:`_probe_audio_seconds` 正是拿它当**源音轨长度**去和画面
    #:   做截断对账的 ✗ ⇒ 会照着一个**带余量的容器数**报「音轨被截掉 0.045 秒」✗✗（假的 ✓）。
    #:   回落**保留** ✓（对"混流之后的 mp4"无害且必要 ✓：``-t`` 已让两条流同长 ✓），
    #:   但**必须交出它是什么** ✓ —— 这就是 ``audioDurationSource`` 三档的意义 ✓✗。
    if not audio:
        audio_duration, audio_source = None, None
    elif audio.get("duration"):
        audio_duration, audio_source = audio.get("duration"), "stream"
    else:
        audio_duration, audio_source = duration, "container"
    return {
        "path": str(target), "bytes": target.stat().st_size,
        "width": int(video.get("width") or 0), "height": int(video.get("height") or 0),
        "codec": video.get("codec_name"),
        "pixFmt": video.get("pix_fmt"),
        "fps": _fps_of(video.get("r_frame_rate") or video.get("avg_frame_rate")),
        "durationSeconds": round(float(duration), 4) if duration else None,
        "nbFrames": int(frames) if frames else None,
        "hasAudio": bool(audio),
        # 音轨事实：混流后**同一份报告**就能核"里面那条音轨是什么" ✓（不必再 ffprobe 一次 ✓）
        "audioCodec": audio.get("codec_name"),
        "audioChannels": int(audio.get("channels") or 0) or None,
        "audioSampleRate": int(audio.get("sample_rate") or 0) or None,
        "audioDurationSeconds": round(float(audio_duration), 4) if audio_duration else None,
        #: ⭐ 「这个音频时长是**流自己**给的 ✓，还是**容器顶替**的 ✗，还是**压根没有** ✗」——
        #:   三档必须能分辨 ✓✗（``"stream"`` / ``"container"`` / ``None`` ✓）。少了这一格 ⇒
        #:   「源音轨 1.0 秒」与「容器 1.045 秒顶替」在回执里同形 ✓，下游对账会照着后者喊狼 ✗。
        "audioDurationSource": audio_source,
        # 容器标签 ✓（判断"元数据到底写进去了没有"只能靠它 ✓ —— 送了不等于写了 ✗）
        "tags": dict(payload.get("format", {}).get("tags") or {}),
    }


def _fps_of(text: Any) -> float | None:
    """``"24/1"`` → ``24.0`` ✓。"""
    try:
        numerator, _, denominator = str(text).partition("/")
        return round(float(numerator) / float(denominator or 1), 4)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def load_image_tensor(path: str | Path, *, width: int, height: int,
                      value_range: str = "-1..1") -> Any:
    """图片 → 帧张量 ✓ 形状 ``(1, 3, 1, H, W)``（**单帧** ✓ —— 首帧条件要的就是它 ✓）。

    ⚠️ 尺寸是**显式**给的 ✓：本函数**按给定的宽高拉伸** ✓（不做 letterbox/crop ✗ ——
    那些都会改变构图 ✓ 而"悄悄改构图"正是最难看出来的错 ✗）。返回的 ``report`` 里带**原图尺寸** ✓
    便于调用方发现"被拉伸了" ✓。

    值域与 :func:`write_video` 同一套口径 ✓（``-1..1`` 默认 ✓）。
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    target = Path(path)
    if not target.exists():
        raise MediaError(f"图片不存在：{target} ✗")
    try:
        with Image.open(target) as handle:
            original = (int(handle.width), int(handle.height))
            converted = handle.convert("RGB").resize((int(width), int(height)),
                                                     resample=Image.BILINEAR)
            array = np.asarray(converted, dtype=np.float32) / 255.0     # (H, W, 3) ✓
    except (OSError, ValueError) as err:
        raise MediaError(f"图片读不了（{type(err).__name__}: {err}）✗：{target}") from err
    kind = str(value_range or "-1..1").replace(" ", "")
    if kind in ("-1..1", "[-1,1]"):
        array = array * 2.0 - 1.0
    elif kind not in ("0..1", "[0,1]"):
        raise MediaError(f"未知值域 {value_range!r}；可用：-1..1 / 0..1 ✓")
    # (H, W, 3) → (1, 3, 1, H, W) ✓ 单帧单批 ✓
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return tensor.contiguous(), {"originalWidth": original[0], "originalHeight": original[1],
                                 "width": int(width), "height": int(height),
                                 "resized": (original[0], original[1]) != (int(width), int(height)),
                                 "valueRange": value_range}


def write_image(frames: Any, path: str | Path, *, index: int = 0,
                value_range: str = "-1..1", metadata: dict[str, Any] | None = None,
                blank_std: float = 1.0) -> dict[str, Any]:
    """帧张量 → **真 PNG** ✓（调试/预览用 ✓）。

    两种入参都认 ✓：
    * ``(B,3,T,H,W)`` 视频帧 ✓ —— 取第 ``index`` 帧 ✓；
    * ``(B,3,H,W)`` **单张图** ✓ —— 没有时间维 ✓，``index`` 只能是 0 ✓。

    ⚠️ 2026-10-31 补 4 维 ✓：Z-Image 这类**图模型**的 VAE 解出来就是 4 维 ✓，此前这里一刀切判 5 维 ✓✗
    ⇒ **真图落不了盘** ✓✗（而且报错指向"帧张量形状"，看不出是"这是图不是片"✓）。

    ``metadata``：写进 PNG 的 iTXt ✓（提示词/种子/参数摘要 ✓ —— 图脱离作业单后仍能自证怎么来的 ✓）。
    ``blank_std``：**空白图判据** ✓ —— 像素标准差低于它就标 ``degraded`` ✓（纯黑/纯白/一片灰 ⇒ 标准差 ≈ 0 ✓）。
    归一化错、VAE 没解出内容、张量全零这几种失败**都不会抛异常** ✓✗，只落一张"看着正常"的纯色图 ✗
    ⇒ 这里给出机器可读的 ``std`` ✓（**报而不拦** ✓：阈值是创作口径 ✓，拦不拦由调用方定 ✓）。
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    shape = tuple(frames.shape)
    if len(shape) == 5 and shape[1] == 3:
        if not 0 <= int(index) < int(shape[2]):
            raise MediaError(f"帧序号 {index} 越界（共 {int(shape[2])} 帧 ✗）")
        frame = frames[0, :, int(index)]
        height, width = int(shape[3]), int(shape[4])
    elif len(shape) == 4 and shape[1] == 3:
        if int(index) not in (0, -1):
            raise MediaError(f"单张图没有第 {index} 帧 ✗（4 维张量只有一张 ✓，index 用 0 ✓）")
        frame = frames[0]
        height, width = int(shape[2]), int(shape[3])
    else:
        raise MediaError(
            f"帧张量形状应为 (B, 3, T, H, W) 或单图 (B, 3, H, W) ✓，收到 {shape} ✗")
    kind = str(value_range or "-1..1").replace(" ", "")
    scaled = (frame.clamp(-1.0, 1.0) + 1.0) * 0.5 if kind in ("-1..1", "[-1,1]") \
        else frame.clamp(0.0, 1.0)
    array = (scaled.detach().to("cpu", dtype=torch.float32).permute(1, 2, 0).numpy() * 255.0)
    pixels = array.round().clip(0.0, 255.0).astype(np.uint8)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    info = _png_info(metadata)
    picture = Image.fromarray(pixels, "RGB")
    if info is not None:
        picture.save(target, pnginfo=info)
    else:
        picture.save(target)
    #: ⭐⭐ 「元数据**真写进 PNG 了**」必须**从盘上读回来核** ✗✗（2026-11-10 补 ✗）：
    #:   此前 ``metadataKeys`` 列的是**送出去的键名** ✓✗，而 :func:`_png_info` 会**跳过**
    #:   ``value is None`` 的项 ✓、键名 strip 后为空也跳过 ✓ ⇒「送了 3 个键」与「PNG 里真有 3 个键」
    #:   在回执里同形 ✓，甚至列出来的键**根本不在文件里** ✓✗ —— 这正是本族那个
    #:   「读数缺失被当成读数」（只不过这里的"读数"是**我们自己打算写的清单** ✓）。
    #:   mp4 那侧早有 ``metadataDropped`` 对账 ✓，图这侧一直没有 ✓ ⇒ 现在同口径：
    #:   **读回来**才算数 ✓；读不回来（文件都打不开 ✓）⇒ **一个都不算数** ✓，不是"都对" ✗。
    landed: list[str] = []
    missing: list[str] = []
    if metadata:
        try:
            with Image.open(target) as check:
                present = {str(key) for key in check.info}
        except (OSError, ValueError):
            present = set()
        for key in dict(metadata):
            name = str(key).strip()
            if name and name in present:
                landed.append(name)
            else:
                missing.append(name or str(key))
    # 空白图判据 ✓：标准差按 **0..255 的像素**算 ✓（纯色 ⇒ 0 ✓；正常画面通常 30+ ✓）
    spread = float(pixels.std())
    return {"path": str(target), "bytes": target.stat().st_size, "index": int(index),
            "width": int(width), "height": int(height),
            "mean": round(float(pixels.mean()), 3), "std": round(spread, 3),
            "blank": spread < float(blank_std),
            "degraded": spread < float(blank_std),
            #: **真读回来的**键名 ✓（下游拿它能复核 ✓）
            "metadataKeys": sorted(landed),
            #: 送了但**读不回来**的键名 ✗（``value is None`` / 空键名 / PNG 打不开 ✓）
            "metadataMissing": sorted(missing),
            #: 三档 ✓：``True``＝逐键读回来核过 ✓／``None``＝本次**没送** metadata ✓（不是"核过了" ✗）
            "metadataReadback": (True if metadata else None)}


def load_video_tensor(path: str | Path, *, value_range: str = "-1..1") -> tuple[Any, dict[str, Any]]:
    """视频文件 → 帧张量 ✓ 形状 ``(1, 3, T, H, W)`` —— **自己实现** ✓（ffmpeg 解码 ✓ 不引依赖 ✓）。

    ⚠️ 用**原生尺寸与帧数** ✓：**不缩放、不裁剪、不抽帧** ✗（参考视频块用的是**它自己的**网格 ✓
    —— 悄悄改尺寸/丢帧都会改变素材本身 ✓✗，而参考唯一在意的就是它的内容 ✓）。
    尺寸整除性（vae_scale ✓）由**调用方**校验 ✓ —— 这里只管忠实解码 ✓。
    ⚠️ 解出的字节数**不是整数帧** ⇒ 报错 ✗（不静默截断半帧 ✓）。
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    target = Path(path)
    if not target.exists():
        raise MediaError(f"视频文件不存在 ✗：{target}（先确认路径 ✓）")
    facts = probe(target)
    width, height = int(facts["width"]), int(facts["height"])
    if not width or not height:
        raise MediaError(f"文件里没有视频轨 ✗：{target}（ffprobe 读不到宽高 ✓）")
    command = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error",
               "-i", str(target), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        done = subprocess.run(command, capture_output=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as err:
        raise MediaError(f"ffmpeg 调用失败：{type(err).__name__}: {err}") from err
    if done.returncode:
        tail = (done.stderr or b"")[-600:].decode("utf-8", "replace")
        raise MediaError(f"ffmpeg 解码返回 {done.returncode} ✗：{tail}")
    raw = done.stdout or b""
    per_frame = width * height * 3
    if not raw:
        raise MediaError(f"一帧都没解出来 ✗：{target}（ffprobe 说有 {width}×{height} 的视频轨 ✓）")
    if len(raw) % per_frame:
        raise MediaError(
            f"解出的字节数不是整数帧 ✗（{len(raw)} 字节 ÷ 每帧 {per_frame} 有余 ✓）"
            f"—— 不静默截断半帧 ✓：{target}")
    frames = len(raw) // per_frame
    kind = str(value_range or "-1..1").replace(" ", "")
    if kind not in ("-1..1", "[-1,1]", "0..1", "[0,1]"):
        raise MediaError(f"未知值域 {value_range!r}；可用：-1..1 / 0..1 ✓")
    array = np.frombuffer(raw, dtype=np.uint8).reshape(frames, height, width, 3)
    values = torch.from_numpy(array.astype(np.float32) / 255.0)
    if kind in ("-1..1", "[-1,1]"):
        values = values * 2.0 - 1.0
    # (T, H, W, 3) → (1, 3, T, H, W) ✓ 与 write_video 的输入口径一致 ✓（可往返 ✓）
    tensor = values.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return tensor, {"path": str(target), "width": width, "height": height, "frames": frames,
                    "fps": facts["fps"], "hasAudio": bool(facts["hasAudio"]),
                    "valueRange": value_range}


def extract_wav(source: str | Path, target: str | Path) -> dict[str, Any]:
    """从视频容器里**抽出音轨** → 16-bit PCM wav ✓（**原生**采样率/声道 ✓）。

    ⚠️ 与 :func:`load_wav_tensor` **成对** ✓：这里**不重采样、不混声道** ✗
    （不带 ``-ar`` / ``-ac`` ✓）—— 采样率/声道与音频 VAE 不符会在读回那一步**明确报错** ✓
    （那才是该报的地方 ✓，而不是在这里悄悄改掉素材 ✓✗）。
    """
    origin, destination = Path(source), Path(target)
    if not origin.exists():
        raise MediaError(f"视频文件不存在 ✗：{origin}（先确认路径 ✓）")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(origin), "-vn", "-acodec", "pcm_s16le", str(destination)]
    try:
        done = subprocess.run(command, capture_output=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as err:
        raise MediaError(f"ffmpeg 调用失败：{type(err).__name__}: {err}") from err
    if done.returncode:
        tail = (done.stderr or b"")[-600:].decode("utf-8", "replace")
        raise MediaError(f"ffmpeg 抽音轨返回 {done.returncode} ✗：{tail}")
    if not destination.exists() or destination.stat().st_size == 0:
        raise MediaError(
            f"ffmpeg 返回 0 但没抽出音轨（{origin} ✗）—— **静默失败**必须当失败处理 ✓"
            f"（先用 :func:`probe` 的 ``hasAudio`` 判断有没有音轨 ✓）")
    return {"path": str(destination), "bytes": destination.stat().st_size}


def load_wav_tensor(path: str | Path, *, sample_rate: int,
                    channels: int = 2) -> tuple[Any, dict[str, Any]]:
    """wav → 波形张量 ✓ 形状 ``(channels, N)`` ✓（值域 ``[-1, 1]`` ✓）—— **自己实现** ✓（标准库 `wave` ✓ 零依赖 ✓）。

    ⚠️ 与 :func:`write_wav` **成对** ✓（那边写 16-bit PCM ✓ 这边读 16-bit PCM ✓）。
    ⚠️ 采样率 / 声道数与**给定的**不一致 ⇒ **报错** ✓：**不重采样、不混声道** ✗ ——
    那些都会**悄悄改掉素材本身** ✓✗，而参考音频唯一在意的就是它的内容 ✓。
    ⚠️ 位宽不是 2 字节 ⇒ 报错 ✓（本仓只写 2 字节 ✓ ⇒ 别的宽度**没有依据** ✓ 不猜 ✗）。
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    target = Path(path)
    if not target.exists():
        raise MediaError(f"音频文件不存在 ✗：{target}（先确认路径 ✓）")
    with wave.open(str(target), "rb") as handle:
        got_channels, width = handle.getnchannels(), handle.getsampwidth()
        rate, count = handle.getframerate(), handle.getnframes()
        raw = handle.readframes(count)
    if int(width) != 2:
        raise MediaError(f"采样位宽 {int(width) * 8} bit ✗ —— 本仓只认 16-bit PCM ✓（不猜别的格式 ✗）")
    if int(rate) != int(sample_rate) or int(got_channels) != int(channels):
        raise MediaError(
            f"采样率/声道与要求不符 ✗：文件是 {rate} Hz / {got_channels} 声道 ✓，"
            f"要的是 {sample_rate} Hz / {channels} 声道 ✓（**不重采样、不混声道** ✓✗ —— "
            f"那会悄悄改掉参考音频的内容 ✓）")
    values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    tensor = torch.from_numpy(values.reshape(int(count), int(channels)).T.copy())
    return tensor, {"path": str(target), "sampleRate": int(rate), "channels": int(got_channels),
                    "samples": int(count),
                    "durationSeconds": round(int(count) / max(1, int(rate)), 4)}


def write_wav(samples: Any, path: str | Path, sample_rate: int = 32000) -> dict[str, Any]:
    """音频张量 → **真 wav** ✓（用标准库 `wave` ✓ 不引依赖 ✓）；返回时长等事实 ✓。

    ``samples``：``(N,)`` 单声道 或 ``(channels, N)`` ✓，值域 ``[-1, 1]`` ✓（越界会被**钳**到边界 ✓
    并如实回报 ``clipped`` ✓ —— 而不是让它绕回成大噪声 ✗）。

    ⚠️⚠️ **「这段音频到底有没有声音」也必须交出去** ✗✗（2026-11-09 补 ✗）：:func:`write_image`
    早有 ``blank_std`` 判纯色图 ✓，而这里此前只有 ``bytes`` / ``durationSeconds`` ✓ ——
    一个**全零波形**与一段正常音频在回执里**完全同形** ✓✗：wav 照写 ✓（0 字节错误都没有 ✓）、
    混进 mp4 也照成 ✓、回执照说「真 wav ✓ 已混进 mp4 ✓」✓ ⇒ 用户听到一段无声片，而**没有
    任何一格**提示过 ✗。这正是「读数缺失被当成读数说没有」那一族的近亲：**没量过振幅**，
    就等于宣布「音频没问题」✗。
    ⇒ 现在给出 ``peak``（绝对值最大 ✓）/ ``rms`` ✓ / ``peakDbfs`` ✓ / ``silent`` ✓。
    ⚠️ ``peakDbfs`` 在 ``peak == 0`` 时是 **``None``** ✓ 而不是 ``-inf`` ✗（不编数 ✓ ——
    ``-inf`` 过一遍 JSON 会变成 ``-Infinity``，那是**非法 JSON** ✗，前端 ``JSON.parse`` 直接炸 ✗）；
    ``silent`` 用**峰值**判而不是 rms ✓（rms 低但峰值高的脉冲音不是静音 ✗）。
    报而不拦 ✓（与 ``blank_std`` 同口径 ✓：阈值是创作口径 ✓，拦不拦由调用方定 ✓）。
    """
    import numpy as np  # noqa: PLC0415

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = samples
    if len(tuple(array.shape)) == 1:
        array = array.unsqueeze(0)
    if len(tuple(array.shape)) != 2:
        raise MediaError(f"音频张量形状应为 (channels, N) 或 (N,) ✓，收到 {tuple(samples.shape)} ✗")
    channels, length = tuple(array.shape)
    clipped = int((array.abs() > 1.0).sum().item())
    pcm = (array.clamp(-1.0, 1.0) * 32767.0).round().to("cpu", dtype=__import__("torch").int16)
    raw = pcm.numpy().astype(np.int16)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(int(channels))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        # 交错写入 ✓（wave 要的是 ch0,ch1,ch0,ch1… ✓）
        handle.writeframes(raw.T.tobytes() if channels > 1 else raw.tobytes())
    #: ⭐⭐ **从写进文件的 int16 里**量振幅 ✓✗（不是从入参张量量 ✓）：问题要问的是
    #:   「这条 wav 里到底有没有声音」✓ —— 入参再响，量化/钳位之后是 0 也就是没声音 ✓；
    #:   反过来若从 float 张量算，一个 1e-6 的"有值"会被判成非静音 ✗ 而 16-bit 里它就是 0 ✗。
    flat = raw.astype(np.float64) / 32768.0
    peak = float(np.abs(flat).max()) if flat.size else 0.0
    rms = float(np.sqrt(np.mean(flat ** 2))) if flat.size else 0.0
    return {
        "path": str(target), "bytes": target.stat().st_size, "channels": int(channels),
        "sampleRate": int(sample_rate), "samples": int(length),
        "durationSeconds": round(length / max(1, int(sample_rate)), 4), "clipped": clipped,
        #: ⚠️ ``peakDbfs`` 全零时是 ``None`` ✓（不写 ``-inf`` ✗ —— 那是非法 JSON ✗）
        "peak": round(peak, 6), "rms": round(rms, 6),
        "peakDbfs": round(20.0 * math.log10(peak), 2) if peak > 0 else None,
        #: ⚠️ 判据用**峰值**（不是 rms ✓）：一声咳嗽 rms 很低但不是静音 ✗；
        #:   门限 ≈ −80 dBFS ✓ —— 16-bit 的量化底噪本身就有 1/32767 ≈ 3.05e-5 ✓。
        "silent": bool(peak < _PEAK_SILENT),
    }
