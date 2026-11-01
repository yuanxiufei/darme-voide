"""Qwen3-VL 的**像素预处理**（H3 用）✓ —— 把图/帧对变成视觉塔要的 ``flatten_patches`` ✓。

## 逐字对照 `comfy/text_encoders/qwen_vl.py::process_qwen2vl_images`
## 与 `comfy/text_encoders/minimax.py::process_video_block`

两条口径**只差一件事** ✓：单图把**同一帧重复两遍**填满 ``temporal_patch_size=2`` ✓；
视频块把**相邻两帧**放进那两格 ✓。resize / 归一化 / patchify 的顺序完全一样 ✓。

## 四处会静默错的细节

1. **归一化是 0.5 / 0.5** ✓（`minimax.py::QWEN_IMAGE_MEAN/STD` ✓）——**不是** CLIP 的
   ``0.48145466 / 0.26862954`` 那套 ✗。用错 ⇒ 颜色整体偏 ✓ 不报错 ✗（Qwen2.5-VL 才是 CLIP 均值 ✓）。
2. **行序是「merge 块优先」** ✓ 不是「像素行优先」✗：``permute(0, 3, 6, 4, 7, 2, 1, 5, 8)`` ⇒
   行下标按 ``(grid_t, h//merge, w//merge, merge_h, merge_w)`` 走 ✓。视觉塔的
   :func:`~.qwen3vl_vision.VisionTower.forward` 内部正是按这个顺序做位置插值 ✓ ⇒ 两边必须一致 ✓
   （写反 ⇒ 位置编码套错格 ✓ 图糊但不报错 ✗）。
3. **resize 的倍数是 ``patch*merge = 32``** ✓ 且 ``h_bar`` 用 **``round``** ✓（不是 floor ✗）——
   长宽不平衡的图在 ``round`` 与 ``floor`` 下会得到**不同**的网格 ✓ 且都不报错 ✗。
4. **``min_pixels=3136`` 会把小图/视频块放大** ✓（参考缺省就是这样 ✓ 不是 bug ✓）——
   两块加起来 ``max_pixels`` 上限仍生效 ✓。

⚠️ 时间维：`grid_t` 恒为 **1** ✓ —— 一个视频块的 2 帧进 ``temporal_patch_size`` ✓ 不进 ``grid_t`` ✗。
所以「`t` 变大」只发生在**多帧拼一起**的场合 ✓，H3 的每块都是 2 帧、``grid_t=1`` ✓。
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "MAX_PIXELS", "MERGE_SIZE", "MIN_PIXELS", "PATCH_SIZE", "PIXEL_MEAN", "PIXEL_STD",
    "TEMPORAL_PATCH_SIZE", "pad_frames_even", "patchify", "process_image",
    "process_video_block", "resize_plan",
]

PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
#: ⚠️ **0.5 / 0.5** ✓（Qwen3-VL ✓）—— 不是 CLIP 均值方差 ✗
PIXEL_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
PIXEL_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)
MIN_PIXELS = 3136
MAX_PIXELS = 12845056


def resize_plan(height: int, width: int, *, patch_size: int = PATCH_SIZE,
                merge_size: int = MERGE_SIZE, min_pixels: int = MIN_PIXELS,
                max_pixels: int = MAX_PIXELS) -> dict[str, Any]:
    """算出 ``(h_bar, w_bar)`` ✓ —— 逐字照参考（``round`` 起手 ✓、上限按比例缩 ✓、下限按比例放 ✓）。"""
    factor = patch_size * merge_size
    if height <= 0 or width <= 0:
        raise ValueError(f"高宽必须 > 0（收到 {height}×{width} ✗）")
    h_bar = int(round(height / factor)) * factor
    w_bar = int(round(width / factor)) * factor
    reason = "round"
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, int(math.floor(height / beta / factor)) * factor)
        w_bar = max(factor, int(math.floor(width / beta / factor)) * factor)
        reason = "shrink"
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = int(math.ceil(height * beta / factor)) * factor
        w_bar = int(math.ceil(width * beta / factor)) * factor
        reason = "grow"
    if h_bar <= 0 or w_bar <= 0:
        raise ValueError(f"resize 计划算出 {h_bar}×{w_bar} ✗（{height}×{width} 太小 ✓）")
    return {"height": h_bar, "width": w_bar, "factor": factor,
            "gridH": h_bar // patch_size, "gridW": w_bar // patch_size, "reason": reason}


def pad_frames_even(frames: Any) -> Any:
    """奇数帧 ⇒ **重复末帧**补齐到偶数 ✓（参考 ``cat([frames, frames[-1:]])`` ✓）。

    ⚠️ 时间戳也要跟着重复 ✓（``timestamps += [timestamps[-1]]`` ✓）—— 那一步在
    :func:`~.h3_prompt.video_blocks` 的调用方做 ✓（本函数只管像素 ✓）。
    """
    torch = _torch()
    if int(frames.shape[0]) % TEMPORAL_PATCH_SIZE == 0:
        return frames
    return torch.cat((frames, frames[-1:]), dim=0)


def _torch() -> Any:
    import torch

    return torch


def _resize_normalize(planes: Any, height: int, width: int) -> Any:
    """``(n, C, H, W)`` ⇒ 双线性 resize ⇒ ``(x - 0.5) / 0.5`` ✓。"""
    torch = _torch()
    functional = torch.nn.functional
    plan = resize_plan(height, width)
    resized = functional.interpolate(planes, size=(plan["height"], plan["width"]),
                                     mode="bilinear", align_corners=False)
    mean = torch.tensor(PIXEL_MEAN, device=resized.device, dtype=resized.dtype).view(1, 3, 1, 1)
    std = torch.tensor(PIXEL_STD, device=resized.device, dtype=resized.dtype).view(1, 3, 1, 1)
    return (resized - mean) / std, plan


def patchify(planes: Any, grid_h: int, grid_w: int, *, patch_size: int = PATCH_SIZE,
             merge_size: int = MERGE_SIZE, temporal: int = TEMPORAL_PATCH_SIZE) -> Any:
    """``(temporal, C, H, W)`` ⇒ ``((grid_h*grid_w), C*temporal*p*p)`` ✓ —— 顺序与 Conv3d 逐字一致 ✓。

    ⚠️ **行序是「merge 块优先」** ✓：``(gh//m, m, p, gw//m, m, p)`` 里的 ``(gh//m, gw//m)`` 在外层 ✓。
    """
    torch = _torch()
    channel = int(planes.shape[1])
    patches = planes.reshape(
        1, temporal, channel,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(grid_h * grid_w, channel * temporal * patch_size * patch_size)


def process_image(pixels: Any, *, patch_size: int = PATCH_SIZE,
                  merge_size: int = MERGE_SIZE) -> tuple[Any, tuple[int, int, int]]:
    """单图 ``(H, W, C)`` ⇒ ``(patches, (1, grid_h, grid_w))`` ✓（**同一帧重复两遍**填时间维 ✓）。"""
    torch = _torch()
    if pixels.dim() != 3:
        raise ValueError(f"单图必须是 ``(H, W, C)`` 三维 ✗（收到 {tuple(pixels.shape)}）")
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    planes = pixels[..., :3].permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    normalized, plan = _resize_normalize(planes, height, width)
    grid_h, grid_w = int(plan["gridH"]), int(plan["gridW"])
    doubled = normalized[0].unsqueeze(0).repeat(TEMPORAL_PATCH_SIZE, 1, 1, 1)
    return patchify(doubled, grid_h, grid_w, patch_size=patch_size, merge_size=merge_size), \
        (1, grid_h, grid_w)


def process_video_block(frames: Any, *, patch_size: int = PATCH_SIZE,
                        merge_size: int = MERGE_SIZE) -> tuple[Any, tuple[int, int, int]]:
    """2 帧 ``(2, H, W, C)`` ⇒ ``(patches, (1, grid_h, grid_w))`` ✓（两帧进**时间维** ✓ ``grid_t=1`` ✓）。"""
    if frames.dim() != 4:
        raise ValueError(f"视频块必须是 ``(帧, H, W, C)`` 四维 ✗（收到 {tuple(frames.shape)}）")
    count = int(frames.shape[0])
    if count != TEMPORAL_PATCH_SIZE:
        raise ValueError(
            f"一个视频块必须**恰好 {TEMPORAL_PATCH_SIZE} 帧** ✗（收到 {count} ✓ —— "
            f"参考的 ``grid_t`` 恒为 1 ✓：多给的帧会被当成时间维 ✓ 位置表就对不上了 ✗）")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    planes = frames[..., :3].permute(0, 3, 1, 2).to(_torch().float32)
    normalized, plan = _resize_normalize(planes, height, width)
    grid_h, grid_w = int(plan["gridH"]), int(plan["gridW"])
    return patchify(normalized, grid_h, grid_w, patch_size=patch_size,
                    merge_size=merge_size), (1, grid_h, grid_w)
