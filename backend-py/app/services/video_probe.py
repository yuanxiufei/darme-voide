"""本地视频时长探测（移植自 ``backend/src/utils/video-probe.ts``，15 行）。

用途：**异步提供商**（轮询型 / Webhook 型）在结果里不返回 ``duration`` 时，
用 ``ffprobe`` 读本地文件的实际时长，回填到 ``storyboards.duration``。

⚠️ 与原实现一致的「失败即 0」：Node 的 ``fluent-ffmpeg`` 在报错时 ``resolve(0)``，
所以**没装 ffprobe 也不会抛错**，只是拿不到时长（调用方会保持 duration 为空）。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from ..config import get_storage_root
from ..response import js_round

__all__ = ["probe_video_duration"]

#: 本地路径可能是「相对数据根」的 ``static/xxx``（见 file_storage.get_absolute_path 的说明）
_STATIC_PREFIX_RE = re.compile(r"^static[\\/]")


def _absolute_path(local_path: str) -> str:
    """等价 TS：绝对路径原样用；否则拼到 storageRoot 下（并剥掉开头的 ``static/``）。"""
    if os.path.isabs(local_path):
        return local_path
    return str(Path(get_storage_root()) / _STATIC_PREFIX_RE.sub("", local_path))


async def probe_video_duration(local_path: str) -> int:
    """探测视频实际时长（**整秒**）；失败返回 0。

    ⚠️ 取整用的是 ``response.js_round``（``Math.round`` 的「.5 向 +∞」），
    **不是** Python 内置 ``round`` —— 后者是银行家舍入，``round(2.5) == 2`` 会差 1 秒。
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            _absolute_path(local_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            return 0
        payload = json.loads(stdout.decode("utf-8", errors="replace"))
        raw_duration = (payload.get("format") or {}).get("duration")
        if raw_duration in (None, ""):
            return 0
        return js_round(float(raw_duration))
    except Exception:  # noqa: BLE001 —— 没装 ffprobe / 文件不存在 / 输出不是 JSON 一律 0（与 TS 一致）
        return 0
