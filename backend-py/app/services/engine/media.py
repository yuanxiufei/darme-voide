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
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

__all__ = ["MediaError", "ffmpeg_version", "have_ffmpeg", "load_image_tensor", "probe",
           "write_image", "write_video", "write_wav"]


class MediaError(RuntimeError):
    """落盘失败 ✓（缺工具 / 参数非法 / 编码器报错 ✓ —— 都带上**可行动**的信息 ✓）。"""


def _tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise MediaError(
            f"找不到 {name} ✗ ⇒ 先装 ffmpeg（Windows: `winget install Gyan.FFmpeg` ✓；"
            f"Linux: `apt install ffmpeg` ✓）。本仓的色校正/合成等功能也依赖它 ✓。")
    return found


def have_ffmpeg() -> bool:
    """ffmpeg 是否可用 ✓（`describe()` 会报这个 ✓）。"""
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


def write_video(frames: Any, path: str | Path, *, fps: int = 24, value_range: str = "-1..1",
                crf: int = 18, preset: str = "veryfast", pix_fmt: str = "yuv420p") -> dict[str, Any]:
    """帧张量 → **真 mp4** ✓；返回可复核的事实（宽/高/fps/时长/字节 ✓，来自 :func:`probe` ✓）。"""
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
        "-an",                                   # 视频轨专用 ✓（音频走 write_wav ✓ 再合成是后续事 ✓）
        "-c:v", "libx264", "-pix_fmt", pix_fmt, "-crf", str(int(crf)), "-preset", preset,
        str(target),
    ]
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
    return report


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
    return {
        "path": str(target), "bytes": target.stat().st_size,
        "width": int(video.get("width") or 0), "height": int(video.get("height") or 0),
        "codec": video.get("codec_name"),
        "pixFmt": video.get("pix_fmt"),
        "fps": _fps_of(video.get("r_frame_rate") or video.get("avg_frame_rate")),
        "durationSeconds": round(float(duration), 4) if duration else None,
        "nbFrames": int(frames) if frames else None,
        "hasAudio": bool(audio),
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
                value_range: str = "-1..1") -> dict[str, Any]:
    """帧张量 → **真 PNG** ✓（调试/预览用 ✓；``(B,3,T,H,W)`` 里取第 ``index`` 帧 ✓）。"""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    shape = tuple(frames.shape)
    if len(shape) != 5 or shape[1] != 3:
        raise MediaError(f"帧张量形状应为 (B, 3, T, H, W) ✓，收到 {shape} ✗")
    if not 0 <= int(index) < int(shape[2]):
        raise MediaError(f"帧序号 {index} 越界（共 {int(shape[2])} 帧 ✗）")
    kind = str(value_range or "-1..1").replace(" ", "")
    frame = frames[0, :, int(index)]
    scaled = (frame.clamp(-1.0, 1.0) + 1.0) * 0.5 if kind in ("-1..1", "[-1,1]") \
        else frame.clamp(0.0, 1.0)
    array = (scaled.detach().to("cpu", dtype=torch.float32).permute(1, 2, 0).numpy() * 255.0)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.round().astype(np.uint8), "RGB").save(target)
    return {"path": str(target), "bytes": target.stat().st_size, "index": int(index),
            "width": int(shape[4]), "height": int(shape[3])}


def write_wav(samples: Any, path: str | Path, sample_rate: int = 32000) -> dict[str, Any]:
    """音频张量 → **真 wav** ✓（用标准库 `wave` ✓ 不引依赖 ✓）；返回时长等事实 ✓。

    ``samples``：``(N,)`` 单声道 或 ``(channels, N)`` ✓，值域 ``[-1, 1]`` ✓（越界会被**钳**到边界 ✓
    并如实回报 ``clipped`` ✓ —— 而不是让它绕回成大噪声 ✗）。
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
    return {
        "path": str(target), "bytes": target.stat().st_size, "channels": int(channels),
        "sampleRate": int(sample_rate), "samples": int(length),
        "durationSeconds": round(length / max(1, int(sample_rate)), 4), "clipped": clipped,
    }
