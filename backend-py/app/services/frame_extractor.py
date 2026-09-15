"""尾帧提取（连续性状态机 v3 的衔接链路）—— 与 ``services/frame-extractor.ts``（86 行）对齐。

视频生成完成后，从已产出视频中提取**真实尾帧**写入 ``storyboards.tail_frame_image``，
供下一镜头视频生成时作为「上一镜尾帧」起帧参考（video 链路的 tail-link）。

⚠️ **语义边界（2026-09-03 明确，别混）**：

* ``last_frame_image`` = **设计尾帧**（grid first_last / 手动上传），是 FL2VA 锁定结束画面的
  目标，由用户/设计流程写；
* ``tail_frame_image`` = **真实尾帧**（视频的实际末帧），是可运行产物，随视频重生成随时刷新。

分离后真实尾帧**永远不会冒充设计尾帧**去污染 shot-router 的 FL2VA 决策 —— 所以本模块
**只写 ``tail_frame_image``**。

⚠️ 与 ``video_probe`` 的差异：那里探时长是**整秒**（回填 ``storyboards.duration``），
这里要**浮点秒**（``seek = duration - 0.2``）⇒ 单独实现，别复用 ``probe_video_duration``。
两条路径映射写法不同但**等价**（``storage_root`` 就是 ``<data_root>/static``）。
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from sqlalchemy import and_, select, update

from ..core.config import get_data_root, get_storage_root
from ..core.db import engine
from ..core.models import storyboards
from ..core.response import now
from .task_logger import log_task_success

__all__ = ["extract_frame", "extract_storyboard_tail_frames", "extract_tail_frame",
           "to_abs_media_path"]

#: 尾帧相对 seek（秒）：从时长末尾往前退一点，避免落在**最后一帧之外**取不到图
_TAIL_SEEK_OFFSET = 0.2


def to_abs_media_path(relative_path: str) -> str:
    """相对媒体路径 → 绝对路径。

    * 已是绝对路径 → 原样；
    * ``static/...`` → ``<data_root>/static/...``（即 ``<storage_root>/...``）；
    * 其余 → ``<storage_root>/...``。
    """
    if os.path.isabs(relative_path):
        return relative_path
    if relative_path.startswith("static/") or relative_path.startswith("static\\"):
        return str(Path(get_data_root()) / relative_path)
    return str(Path(get_storage_root()) / relative_path)


async def _probe_duration_seconds(abs_path: str) -> float:
    """``ffprobe`` 读时长（**浮点秒**）；失败 / 取不到一律 0（与 TS 的 ``resolve(0)`` 一致）。"""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            abs_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            return 0.0
        payload = json.loads(stdout.decode("utf-8", errors="replace"))
        raw = (payload.get("format") or {}).get("duration")
        if raw in (None, ""):
            return 0.0
        return float(raw)
    except Exception:  # noqa: BLE001 —— 没装 ffprobe / 文件不存在 / 输出不是 JSON 一律 0
        return 0.0


async def extract_tail_frame(video_url: str) -> str | None:
    """从视频提取**尾帧**（``extract_frame(url, "last_frame")`` 的薄封装，保持既有调用点不变）。"""
    return await extract_frame(video_url, "last_frame")


async def extract_frame(video_url: str, frame_type: str = "first_frame") -> str | None:
    """从视频提取**首帧 / 尾帧**，返回可访问的相对图片路径（``static/frames/<uuid>.jpg``）；失败 ``None``。

    * ``last_frame``：先 ffprobe 拿时长，从**末尾回退 0.2s** 处 seek（避免落在最后一帧之外取不到图）；
    * ``first_frame``：**不 seek**，ffmpeg 默认从 0 取第 1 帧；
    * 时长探测失败（``duration <= 0``）在尾帧路径下**直接返回 None**（原实现如此）。
    """
    is_last = str(frame_type or "") == "last_frame"
    abs_path = to_abs_media_path(video_url)
    if not os.path.exists(abs_path):
        return None

    seek_args: list[str] = []
    if is_last:
        duration = await _probe_duration_seconds(abs_path)
        if duration <= 0:
            return None
        seek_args = ["-ss", str(max(0.0, duration - _TAIL_SEEK_OFFSET))]

    output_dir = Path(get_storage_root()) / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4()}.jpg"
    out_path = output_dir / filename

    try:
        # 对齐 fluent-ffmpeg：`seekInput` 落在 `-i` **之前**，再取 1 帧、质量 2
        await _run_ffmpeg([
            *seek_args, "-i", abs_path,
            "-frames:v", "1", "-q:v", "2", str(out_path),
        ])
    except Exception:  # noqa: BLE001 —— 与 TS 的 catch 等价
        return None
    return f"static/frames/{filename}"


async def _run_ffmpeg(args: list[str]) -> None:
    """跑一次 ffmpeg；非 0 退出时抛错（与 ``ffmpeg_compose._run_ffmpeg`` 同形）。

    独立成函数是**刻意留的测试缝**：自检把它换掉即可在没装 ffmpeg 的机器上锁住参数与状态机。
    """
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        tail = (stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited with code {process.returncode}: {tail[-2000:]}")


async def extract_storyboard_tail_frames(episode_id: int, drama_id: int) -> int:
    """提取本集所有「已有视频」分镜的真实尾帧；返回**成功提取数量**。

    幂等：视频重生成后重跑会刷新（``tail_frame_image`` 是可运行产物）。
    ⚠️ 读写都自管短事务（对齐 Node 的逐语句自动提交；调用方是后台管线）。
    """
    with engine.begin() as conn:
        rows = conn.execute(
            select(storyboards.c.id, storyboards.c.video_url).where(and_(
                storyboards.c.episode_id == episode_id,
                storyboards.c.deleted_at.is_(None),
            ))
        ).all()

    done = 0
    failed = 0
    for row in rows:
        video_url = row[1]
        if not video_url:
            continue
        frame = await extract_tail_frame(video_url)
        if frame is None:
            failed += 1
            continue
        # ⚠️ 只写 `tail_frame_image`：设计尾帧（last_frame_image）**永不被真实产物覆盖**
        with engine.begin() as conn:
            conn.execute(
                update(storyboards).where(storyboards.c.id == row[0]).values(
                    tail_frame_image=frame, updated_at=now()
                )
            )
        done += 1

    if done > 0 or failed > 0:
        log_task_success("TailFrame", "extract", {
            "episodeId": episode_id, "dramaId": drama_id,
            "extracted": done, "failed": failed,
        })
    return done
