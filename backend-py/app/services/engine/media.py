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
    if audio is None:
        command.append("-an")                    # 纯视频轨 ✓（音频要单独落 wav 时走这条路 ✓）
    else:
        source = Path(audio)
        if not source.exists():
            raise MediaError(f"要混入的音轨不存在 ✗：{source}（先落 wav 再混 ✓）")
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
        report["audioDurationSeconds"] = report.get("audioDurationSeconds")
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
    # ⚠️ 没有音频流 ⇒ ``音频时长`` 必须是 ``None`` ✓✗（回落成容器时长就是**把画面上报成音轨** ✓✗）
    audio_duration = (audio.get("duration") or duration) if audio else None
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
    # 空白图判据 ✓：标准差按 **0..255 的像素**算 ✓（纯色 ⇒ 0 ✓；正常画面通常 30+ ✓）
    spread = float(pixels.std())
    return {"path": str(target), "bytes": target.stat().st_size, "index": int(index),
            "width": int(width), "height": int(height),
            "mean": round(float(pixels.mean()), 3), "std": round(spread, 3),
            "blank": spread < float(blank_std),
            "degraded": spread < float(blank_std),
            "metadataKeys": sorted(str(key) for key in dict(metadata or {}))}


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
