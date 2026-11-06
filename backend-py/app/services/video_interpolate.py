"""视频插帧（FFmpeg ``minterpolate``）—— 移植自上游 ``src/video-interpolation.js`` 的**语义**，实现换成 ffmpeg ✓。

## 为什么不用源那套 RIFE 工作流

源走 ComfyUI 自定义节点 + ``_internal/video-rife.json`` 工作流资产 ✗：那要求用户装了 RIFE 节点、
把工作流文件放对位置 —— 缺一样就是「点了没反应」，而且报错出现在 ComfyUI 日志里、前端只看到超时 ✗。
ffmpeg 的 ``minterpolate`` 同样是**运动补偿 + 双向块匹配**，零额外依赖、装 ffmpeg 就能跑 ✓。

## 三处必须保真的口径

1. ⭐ **倍率只接 2× / 3×** ✓（源同口径）：``minterpolate`` 的耗时随倍率**平方级**涨，
   而且插出来的是"造出来的帧" —— 4 倍之后肉眼全是糊 ✗。与其让用户等二十分钟拿一段糊片，
   不如在入口就拒掉 ✓（错误文案带「仅支持 2× 或 3×」，用户知道该退到哪一档 ✓）。
2. ⭐ **输出帧率 = 源帧率 × 倍率** ✓：源 JS 那句注释（"输出帧率必须跟随源视频，否则 30/60fps 素材会变速"）是
   这一行存在的理由 —— 写死 24 会让 30fps 素材变慢、60fps 变快 ✗。
3. ⭐ **超时必须杀进程** ✓：源里超时只调 ``comfy.cancel()``；ffmpeg 若不 ``kill``，
   进程会继续烧 CPU 跑到底，前端已经报错了、机器还在满负荷 ✗。

⚠️ 尺寸**偶数化**：``libx264`` + ``yuv420p`` 要求宽高为偶数，奇数尺寸直接失败（不是"糊一点"✗）。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.config import get_storage_root
from .file_storage import get_absolute_path
from .task_logger import log_task_error, log_task_start, log_task_success

__all__ = [
    "INTERPOLATE_MULTIPLIERS",
    "build_interpolate_filter",
    "interpolate_video",
    "normalize_multiplier",
    "parse_frame_rate",
    "probe_video_fps",
    "target_fps_of",
]

#: 只接这两档 ✓（见模块头保真点 1）
INTERPOLATE_MULTIPLIERS = (2, 3)

DEFAULT_TIMEOUT_SECONDS = 30 * 60


def normalize_multiplier(value: Any) -> int:
    """倍率归一 ✓：只认 2 / 3，其它一律报（不静默夹取 —— 夹取会让用户以为插了 4 倍 ✗）。"""
    try:
        multiplier = int(value)
    except (TypeError, ValueError):
        raise ValueError("插帧倍率仅支持 2× 或 3×") from None
    if multiplier not in INTERPOLATE_MULTIPLIERS:
        raise ValueError("插帧倍率仅支持 2× 或 3×")
    return multiplier


def parse_frame_rate(text: Any) -> float | None:
    """ffprobe 的帧率是分数串（``"30000/1001"`` ⇒ 29.97 ✓）；解析不出返回 ``None`` ✓。"""
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            value = float(numerator) / float(denominator)
        else:
            value = float(raw)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not value or value != value or value <= 0 or value > 1000:
        return None
    return round(value, 3)


def target_fps_of(source_fps: float, multiplier: int) -> float:
    """源帧率 × 倍率 ✓（保留 3 位小数，FFmpeg 的 ``fps`` 参数接受小数 ✓）。"""
    return round(max(1.0, float(source_fps)) * int(multiplier), 3)


def build_interpolate_filter(source_fps: float, multiplier: int) -> str:
    """滤镜链：先**偶数化**再 ``minterpolate`` ✓（顺序反了会把奇数尺寸带进编码器 ✗）。"""
    fps = target_fps_of(source_fps, multiplier)
    return (
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,"
        f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    )


async def probe_video_fps(file_path: str) -> float | None:
    """读源视频帧率（ffprobe 不可用 / 解析失败 ⇒ ``None``，由调用方回 24 ✓）。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "json", file_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        streams = json.loads(stdout.decode("utf-8", errors="replace") or "{}").get("streams") or []
        return parse_frame_rate(streams[0].get("r_frame_rate")) if streams else None
    except Exception:  # noqa: BLE001 —— 没装 ffprobe / 文件不是视频
        return None


async def _has_audio(file_path: str) -> bool:
    """有音轨就 ``-c:a copy``（不重编码 ✓），没有就 ``-an`` ✓（不然 ffmpeg 会对着无音轨输入报警告）。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
            "-of", "json", file_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return False
        streams = json.loads(stdout.decode("utf-8", errors="replace") or "{}").get("streams") or []
        return any(stream.get("codec_type") == "audio" for stream in streams)
    except Exception:  # noqa: BLE001
        return False


async def _run_ffmpeg(args: list[str], timeout_seconds: float) -> None:
    """跑 ffmpeg；**超时就 kill 并等它真的退出** ✓（只 kill 不 wait 会留僵尸进程 ✗）。"""
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError("插帧超时（已终止 ffmpeg）") from None
    if proc.returncode != 0:
        tail = (stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {tail[-2000:]}")


async def interpolate_video(
    video_url: str,
    multiplier: Any = 2,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """把一条视频插帧到 ``源帧率 × 倍率``，产物落在 ``static/interpolate/`` ✓。

    ``video_url`` 是**相对数据根**的路径（``/uploads/x.mp4`` 或 ``static/...`` ✓），
    走 :func:`get_absolute_path` 做穿越校验 ✓。
    """
    factor = normalize_multiplier(multiplier)
    relative = str(video_url or "").strip().lstrip("/")
    if not relative:
        raise ValueError("videoUrl 必填")
    source_path = get_absolute_path(relative)
    if not Path(source_path).is_file():
        raise FileNotFoundError(f"视频不存在：{video_url}")

    started = time.monotonic()
    log_task_start("InterpolateTask", "start", {"videoUrl": video_url, "multiplier": factor})

    source_fps = await probe_video_fps(source_path) or 24.0
    target_fps = target_fps_of(source_fps, factor)

    out_dir = Path(get_storage_root()) / "static" / "interpolate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uuid4()}.mp4"

    args = ["ffmpeg", "-y", "-i", source_path, "-vf", build_interpolate_filter(source_fps, factor),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    args += ["-c:a", "copy"] if await _has_audio(source_path) else ["-an"]
    args += ["-movflags", "+faststart", str(out_path)]

    try:
        await _run_ffmpeg(args, timeout_seconds)
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise RuntimeError("插帧没有产出文件")
    except Exception as exc:
        # 失败/超时都别留半个文件（留着会被当成产物引用 ✗）
        out_path.unlink(missing_ok=True)
        log_task_error("InterpolateTask", "failed", {"videoUrl": video_url, "error": str(exc)})
        raise

    elapsed_ms = int((time.monotonic() - started) * 1000)
    log_task_success("InterpolateTask", "done", {"videoUrl": video_url, "elapsedMs": elapsed_ms})
    return {
        "url": f"/static/interpolate/{out_path.name}",
        "path": f"static/interpolate/{out_path.name}",
        "multiplier": factor,
        "sourceFps": source_fps,
        "targetFps": target_fps,
        "elapsedMs": elapsed_ms,
    }
