"""宫格图切分（移植自 ``backend/src/services/grid-split.ts``，45 行）。

把一张 ``rows x cols`` 的宫格大图按行列切成单格 PNG，落到 ``<storage>/grid-cells/``，
返回每格的 ``index`` 与**相对数据根**的路径（``static/grid-cells/cell_<ts>_<i>.png``）。

⚠️ **实现替换**：Node 用 ``sharp`` 做无损 PNG 裁剪，Python 侧**没有装 Pillow**，
这里改用 **ffmpeg** 的 ``crop`` 滤镜（同样无损 PNG、同样不新增依赖）。
像素内容等价，字节流当然不同（两者编码器不同）—— 下游只把结果当图片读，不受影响。

⚠️ 三处保真点：

* 尺寸校验：行列必须是**正整数**，否则抛
  ``Invalid grid dimensions: rows=R, cols=C. Both must be positive integers.``；
* 格子尺寸用**向下取整**（``floor(w/cols)``），右边/下边多余几个像素**丢掉**（不拉伸）；
* 时间戳 ``ts`` 在循环**外**取一次 ⇒ 同一批格子文件名共享同一个前缀。
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from .file_storage import get_absolute_path

__all__ = ["split_grid_image"]


async def _probe_dimensions(abs_path: str) -> tuple[int, int] | None:
    """读图片宽高（ffprobe）；读不到返回 None。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", abs_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        streams = (json.loads(stdout.decode("utf-8", errors="replace") or "{}")
                   .get("streams") or [])
        if not streams:
            return None
        width = streams[0].get("width")
        height = streams[0].get("height")
        if not width or not height:
            return None
        return int(width), int(height)
    except Exception:  # noqa: BLE001 —— 没装 ffprobe / 解析失败
        return None


async def _crop(abs_path: str, out_path: str, left: int, top: int, width: int, height: int) -> None:
    """用 ffmpeg 裁剪一格（等价 sharp 的 ``extract``）。"""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", abs_path,
        "-vf", f"crop={width}:{height}:{left}:{top}",
        "-frames:v", "1", out_path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = (stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {tail[-1000:]}")


async def split_grid_image(image_path: str, rows: int, cols: int) -> list[dict[str, Any]]:
    """把宫格大图切成 ``rows x cols`` 个单元格，返回 ``[{index, localPath}]``。"""
    if (
        not isinstance(rows, int) or isinstance(rows, bool) or rows < 1
        or not isinstance(cols, int) or isinstance(cols, bool) or cols < 1
    ):
        raise ValueError(
            f"Invalid grid dimensions: rows={rows}, cols={cols}. "
            "Both must be positive integers."
        )

    # 绝对路径：以 `/` 开头视为已是绝对路径（原 TS 就是只看首字符）
    abs_path = image_path if image_path.startswith("/") else get_absolute_path(image_path)

    dimensions = await _probe_dimensions(abs_path)
    if dimensions is None:
        raise ValueError("Cannot read image dimensions")
    width, height = dimensions

    cell_w = width // cols
    cell_h = height // rows

    out_dir = Path(get_absolute_path("grid-cells"))
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    ts = int(time.time() * 1000)  # ⚠️ 循环外取一次（同批文件名共享前缀）

    for r in range(rows):
        for c in range(cols):
            index = r * cols + c
            file_name = f"cell_{ts}_{index}.png"
            await _crop(abs_path, str(out_dir / file_name), c * cell_w, r * cell_h, cell_w, cell_h)
            results.append({
                "index": index,
                "localPath": f"static/grid-cells/{file_name}",
            })

    return results
