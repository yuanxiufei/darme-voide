"""VAE **分块缩放 / 分块解码**（tiled scale）—— 大图与长视频解码的**显存阀** ✓。

## 为什么要有它

A5000 可用 **22.49 GiB** ✓。Qwen 2.1 @1.76 MP 解码只吃 0.63 GiB ✓（见 `vae_wan` 的实测 ✓），
但**像素数一上去就是平方级**：4K 单图（8.3 MP ✓）与 Wan 81 帧 720p（≈75 MP ✓）一次性解码会在
**最后一层 up 块**上把显存顶爆 ✓✗ —— 报错是 ``CUDA out of memory`` ✓，指不到"该切块" ✗。

出路只有两条 ✓：**降分辨率**（改作品 ✗）或**切块再拼**（本模块 ✓）。

## 与 ComfyUI 的口径关系（照它的语义 ✓，实现自己写 ✓）

* 名字与单位对齐 ``VAEDecodeTiled`` ✓：``tile``/``overlap`` 都是**像素**单位 ✓
  （默认 512 / 64 ✓，与 ComfyUI 一致 ✓）。⚠️ 用户按**潜像素**填是这里最容易踩的错 ✓✗：
  512 px 在 8× VAE 上只有 64 个潜像素 ✓ ⇒ 由 :func:`latent_tile` 统一换算 ✓，
  并且 ``overlap`` 换算后**不足 1 个潜像素就报错** ✗（否则羽化宽度 = 0 ⇒ 接缝**静默**出现 ✓✗）。
* 拼接用**权重累积 + 除法归一** ✓（:func:`tiled_scale` 的 ``acc``/``div`` ✓）：对**局部**算子
  （逐元素 / 卷积 ✓）在重叠带里是**精确平均** ✓ ⇒ 与整张解码**逐位相同** ✓
  （判据见 ``tests/engine_vae_tiling_test.py`` ✓）。
* 只在**空间**切 ✓✗：视频（``(b,c,t,h,w)``）的**时间维不切** ✓ —— Wan/H3 的解码器带**因果时间
  卷积** ✓，切了接缝会在帧间**闪** ✗（H3 侧按 17 帧整段切是参考实现自带的分段 ✓，是另一回事 ✓，
  见 `vae_h3` ✓）。

## 三条"不报错但结果坏"的坑（本模块一律改成**报错** ✓）

1. ``tile`` 太小（< 2 个潜像素）⇒ 每块只看见自己 ✓，拼回来像打了马赛克 ✓；
2. ``overlap`` 不足 1 个潜像素 ⇒ 羽化宽度 0 ⇒ 接缝**一条条硬边** ✓✗；
3. ``overlap ≥ tile`` ⇒ 步长 ≤ 0 ⇒ **死循环** ✓✗（不是慢，是永远不前进 ✓）。
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Sequence

__all__ = [
    "TileError", "TILE_DEFAULTS", "tile_starts", "tiled_scale_steps",
    "tiled_scale", "latent_tile", "decode_latents_tiled",
]

#: 默认档：**像素**单位 ✓（``VAEDecodeTiled`` 同名同义 ✓）。
#: 512 px / overlap 64 px ⇒ 8× VAE 上是 64 / 8 个潜像素 ✓（接缝带 8 个潜像素宽 ✓）。
TILE_DEFAULTS: dict[str, int] = {"tile": 512, "overlap": 64}


class TileError(ValueError):
    """分块参数或分块结果不对 ✓ —— 一律**抛** ✗（接缝与显存问题都不会自己报错 ✓）。"""


def _torch() -> Any:
    try:
        import torch  # noqa: PLC0415
    except Exception as error:  # noqa: BLE001
        raise TileError(f"没装 torch ✗ ⇒ 分块缩放做不了 ✓（{error} ✓）") from error
    return torch


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """一维的**起点表** ✓（全模块的**唯一真源** ✓ —— 步数统计与真切片都从这里取 ✓）。

    ⚠️ 步数若另算一遍 ✓✗，就会出现「进度条总数 12、实际切了 13 块」那种**对不上** ✓ ——
    所以 :func:`tiled_scale_steps` 只是把这张表乘起来 ✓，不重新推公式 ✓。
    """
    length, tile, overlap = int(length), int(tile), int(overlap)
    if tile <= 0:
        raise TileError(f"tile={tile} ✗ 必须是正整数 ✓（单位见模块头：**像素** ✓）")
    if overlap < 0:
        raise TileError(f"overlap={overlap} ✗ 不能是负数 ✓")
    step = tile - overlap
    if step <= 0:
        raise TileError(
            f"overlap={overlap} ≥ tile={tile} ✗ ⇒ 步长 {step} ≤ 0 ✓（永远不前进 ✓✗，不是慢 ✓）")
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] + tile < length:
        #: 末块**贴住右下角** ✓（不贴 ⇒ 最边上那一条**永远没被解码** ✓✗，且不报错 ✓）。
        starts.append(length - tile)
    return starts


def tiled_scale_steps(spatial_shape: Sequence[int], tile: Sequence[int],
                      overlap: int | Sequence[int]) -> int:
    """一共要切几块 ✓（进度条的**总数** ✓ —— 与实际切片同一张起点表 ✓）。"""
    if len(spatial_shape) != len(tile):
        raise TileError(f"tile 有 {len(tile)} 维 ✗ 但空间维是 {len(spatial_shape)} ✓")
    overlaps = (list(overlap) if isinstance(overlap, (list, tuple))
                else [int(overlap)] * len(tile))
    if len(overlaps) != len(tile):
        raise TileError(f"overlap 有 {len(overlaps)} 项 ✗ 与 tile 的 {len(tile)} 维不符 ✓")
    total = 1
    for length, size, band in zip(spatial_shape, tile, overlaps):
        total *= len(tile_starts(int(length), int(size), int(band)))
    return total


def _axis_mask(length: int, feather: int, torch: Any, *, device: Any, dtype: Any) -> Any:
    """一维羽化权重 ✓：两端各 ``feather`` 个位置按 ``(t+1)/feather`` **线性**抬起 ✓。

    ⚠️ 边界块**也照抬** ✓：最终会除以累积权重 ``div`` ✓ ⇒ 只有一块覆盖的位置结果不变 ✓
    （不除 ``div`` 的实现会把边角**压暗** ✓✗ —— 只暗一点、看着像"光照"，最容易被放过 ✓）。
    """
    mask = torch.ones(int(length), device=device, dtype=dtype)
    band = min(int(feather), int(length) // 2)
    if band > 0:
        ramp = (torch.arange(1, band + 1, device=device, dtype=torch.float64) / float(band))
        ramp = ramp.to(dtype=dtype)
        mask[:band] = ramp
        mask[int(length) - band:] = torch.flip(ramp, dims=(0,))
    return mask


def _tile_mask(slice_spatial: Sequence[int], overlaps: Sequence[int], upscale: int,
               dims: int, lead: int, torch: Any, *, device: Any, dtype: Any) -> Any:
    """**一块**的权重 ✓，形状 ``(1,)*lead + slice_spatial`` ✓（各轴一维权重的外积 ✓）。

    ⚠️ 是**块**尺寸（``tile × upscale`` ✓）不是整张尺寸 ✗✗：拿整张去乘每块产物 ⇒
    ``The size of tensor a (16) must match … (53)`` ✓（本轮实测撞到 ✓，报错指向相乘那行 ✓）。
    """
    mask: Any = None
    for axis in range(dims):
        axis_mask = _axis_mask(slice_spatial[axis], int(overlaps[axis]) * int(upscale),
                              torch, device=device, dtype=dtype)
        view = [1] * (lead + dims)
        view[lead + axis] = slice_spatial[axis]
        axis_mask = axis_mask.view(view)
        mask = axis_mask if mask is None else mask * axis_mask
    return mask


def tiled_scale(samples: Any, function: Callable[[Any], Any], tile: Sequence[int],
                overlap: int | Sequence[int] = 8, *, upscale: int = 1,
                out_channels: int | None = None,
                progress: Callable[[int, int], None] | None = None) -> Any:
    """分块跑 ``function`` 再按**权重归一**拼回 ✓（像素数 ``upscale`` 倍 ✓）。

    ``samples`` 是 ``(b, c, *spatial)`` 或 ``(b, c, t, *spatial)`` ✓，``spatial`` = 末
    ``len(tile)`` 维 ✓；``function`` 是 ``tile_in ⇒ tile_out`` 的**纯函数** ✓（不改通道 ✓）。

    ``overlap`` 按**输入**坐标给 ✓，输出侧羽化宽度 = ``overlap × upscale`` ✓。
    ``upscale`` 是输出相对输入的空间倍数 ✓（VAE 解码 = ``spatial_scale`` ✓，SDXL/Flux 是 **8** ✓）。

    ⚠️ 批维永远**整块过** ✓（只切空间 ✓）：按批拆开会让统计类算子（attention 等）口径变 ✗。
    """
    torch = _torch()
    dims = len(tile)
    if dims not in (2, 3):
        raise TileError(f"tile 只能是 2 维（h,w）或 3 维（t,h,w）✗（收到 {dims} 维 ✓）")
    shape = [int(v) for v in samples.shape]
    if len(shape) - dims not in (2, 3):
        raise TileError(
            f"样本形状 {shape} ✗ 与 tile 的 {dims} 维对不上 ✓：要 ``(b,c,*spatial)`` ✓ 或 "
            f"``(b,c,t,*spatial)`` ✓（视频只切空间时形状是 5 维、tile 给 2 维 ✓）")
    upscale = int(upscale)
    if upscale < 1:
        raise TileError(f"upscale={upscale} ✗ 必须 ≥ 1 ✓（缩小的那种不归本模块管 ✗）")

    sizes = [int(v) for v in tile]
    overlaps = (list(overlap) if isinstance(overlap, (list, tuple))
                else [int(overlap)] * dims)
    if len(overlaps) != dims:
        raise TileError(f"overlap 有 {len(overlaps)} 项 ✗ 与 tile 的 {dims} 维不符 ✓")

    spatial = shape[-dims:]
    #: ⚠️ 先按**起点表**建出来 ✓（顺带把 tile/overlap 的三条非法档当场拒掉 ✓）。
    starts_per_axis = [tile_starts(spatial[i], sizes[i], overlaps[i]) for i in range(dims)]
    plan = list(itertools.product(*starts_per_axis))
    total = len(plan)

    #: ⚠️ 通道恒在 **index 1** ✓（``(b,c,…)`` ✓）：视频的 ``t`` 在 ``index 2`` 且**不参与切分** ✓
    #: ⇒ ``middle``（t 那一截 ✓）原样穿过 ✓，用 ``shape[-dims-1]`` 取通道会把 ``t`` 当通道 ✓✗。
    middle = shape[2:len(shape) - dims]
    channels = int(out_channels if out_channels is not None else shape[1])
    out_spatial = [v * upscale for v in spatial]
    expected_out = tuple([shape[0], channels] + middle + out_spatial)
    #: 每块**自己**的输出形状 ✓（不是整张 ✓ —— 拿整张去比会把正确的 function 判成错的 ✓✗）。
    expected_tile = tuple([shape[0], channels] + middle + [v * upscale for v in sizes])
    lead = 1 + len(middle)          # 权重张量的前置 1 维个数 ✓（b 一维 + middle ✓）

    #: 输出 dtype **惰性取** ✓：只能从 ``function`` 的实际产物得知 ✓ —— 猜 fp32 会把 bf16 权重族
    #: 的产物**悄悄升精度** ✓✗（逐位对比当场对不上 ✓）。
    out_dtype: Any = None
    acc: Any = None
    div: Any = None
    for done, index in enumerate(plan, start=1):
        src = tuple(slice(v, v + sizes[axis]) for axis, v in enumerate(index))
        dst = tuple(slice(v * upscale, (v + sizes[axis]) * upscale)
                    for axis, v in enumerate(index))
        piece = samples[(Ellipsis, *src)]
        out = function(piece)
        if tuple(int(v) for v in out.shape) != expected_tile:
            raise TileError(
                f"function 的输出形状 {tuple(int(v) for v in out.shape)} ✗ 与预期 "
                f"{expected_tile} ✓ 不符 —— 检查 upscale={upscale} 与通道数 ✓"
                f"（⚠️ 形状不对**不会**自己报错 ✗，拼回来就是错位的图 ✓✗）")
        if acc is None:
            out_dtype = out.dtype
            #: ⚠️ **累加器恒 fp64** ✓✗：权重是 1/4、2/5、3/4、1 这类**二进制不友好**的数 ✓，
            #: 在 fp32 里累加再除 ⇒ 恒等函数的结果也会差 1 ulp ✓（本轮实测 4.7e-7 ✓）⇒
            #: 「分块 = 整张」这条**永远判不过** ✓✗。fp64 里这些权重与乘积都**精确** ✓、
            #: 除掉之后回到原位 ✓ ⇒ 逐位可比 ✓。代价只有**一张图**的 f64（4K 单图 ≈200 MB ✓），
            #: 相比整张解码要留的中间激活仍是小头 ✓。
            acc_dtype = torch.float64 if out.dtype.is_floating_point else out.dtype
            acc = torch.zeros(expected_out, device=out.device, dtype=acc_dtype)
            div = torch.zeros(out_spatial, device=out.device, dtype=acc_dtype)
        mask = _tile_mask([v * upscale for v in sizes], overlaps, upscale, dims, lead, torch,
                          device=out.device, dtype=acc.dtype)
        acc[(Ellipsis, *dst)] += out.to(dtype=acc.dtype) * mask
        div[dst] += mask[(0,) * lead]
        if progress is not None:
            progress(done, total)
    if acc is None:  # pragma: no cover  （起点表恒非空 ✓，留个明确的兜底 ✗）
        raise TileError("一块都没切 ✗ ⇒ 空间维为空 ✓")
    return (acc / div.clamp_min(torch.finfo(acc.dtype).eps)).to(dtype=out_dtype)


def latent_tile(pixel_tile: int, pixel_overlap: int, spatial_scale: int) -> tuple[int, int]:
    """**像素**尺寸 ⇒ **潜**尺寸 ✓（:func:`tile_starts` 吃潜坐标 ✓）。

    三条非法档一律报错 ✗（它们在真机上全都"能跑完、只是接缝难看" ✓✗）：
    ``tile`` 不足 2 个潜像素 / ``overlap`` 不足 1 个潜像素 / ``overlap × 2 ≥ tile``。
    """
    scale = int(spatial_scale)
    if scale < 1:
        raise TileError(f"spatial_scale={scale} ✗ 必须 ≥ 1 ✓")
    tile = int(pixel_tile) // scale
    overlap = int(pixel_overlap) // scale
    if tile < 2:
        raise TileError(
            f"tile={pixel_tile} px ÷ spatial_scale={scale} ⇒ {tile} 个潜像素 ✗ —— "
            f"至少要 2 ✓（每块只看见自己 ⇒ 拼回来像打了马赛克 ✓✗）")
    if overlap < 1:
        raise TileError(
            f"overlap={pixel_overlap} px ÷ spatial_scale={scale} ⇒ {overlap} 个潜像素 ✗ —— "
            f"羽化宽度为 0 ⇒ 接缝**一条条硬边** ✓✗（至少给 {scale} px ✓）")
    if overlap * 2 >= tile:
        raise TileError(
            f"overlap {overlap} 个潜像素 ≥ tile {tile} 的一半 ✗ ⇒ 步长 ≤ 0 或几乎不前进 ✓")
    return tile, overlap


def decode_latents_tiled(decode_fn: Callable[[Any], Any], latents: Any, *,
                         spatial_scale: int, tile: int = TILE_DEFAULTS["tile"],
                         overlap: int = TILE_DEFAULTS["overlap"],
                         progress: Callable[[int, int], None] | None = None) -> Any:
    """潜变量 ⇒ 像素，按**像素**尺寸切块 ✓（``decode_fn`` 只负责一块 ⇒ 不含缩放系数 ✓）。

    ⚠️ 只切**空间** ✓✗：5 维（``(b,c,t,h,w)``）也照样只切末两维 ✓ —— 时间维带因果卷积 ✓，
    切了接缝**在帧间闪** ✗（见模块头 ✓）。
    """
    tile_latent, overlap_latent = latent_tile(tile, overlap, spatial_scale)
    return tiled_scale(latents, decode_fn, (tile_latent, tile_latent), overlap_latent,
                       upscale=int(spatial_scale), progress=progress)
