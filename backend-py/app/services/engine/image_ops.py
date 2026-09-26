"""**图像算子集**（纯 ``torch`` ✓ —— **不引 kornia ✗、不引 scipy ✗**，2026-09-26 起）。

## 为什么有这一层

「生成/后处理」这条线上缺的从来不是模型，而是**能把张量改形的那几十个算子**：
缩到目标尺寸、裁一块、两张图并起来、把 source 贴到 destination 的某坐标、按掩罩合成、
形态学去噪、RGB↔YUV 换色域……这些在参考项目里是**节点** ✓，在本仓是**函数** ✓ ——
于是它们能被流水线直接调 ✓，而不被「节点图」绑住 ✗。

## 口径（**本模块最重要的一段**，实现必须与它对得上 ✓）

* **IMAGE** = ``(B, H, W, C)`` ✓、float、值域 ``[0, 1]`` ✓（``C`` 常见 3 / 4 ✓）；
* **MASK** = ``(B, H, W)`` ✓、值域 ``[0, 1]`` ✓ —— ⚠️ **1 = 挖掉/透明** ✓、**0 = 保留** ✓
  （与 alpha **相反** ✗ ⇒ 换算只写一遍 ✓，见 :func:`mask_to_alpha` ✓）；
* **内部一律折成 ``(N, C, H, W)``** ✓（卷积/池化/插值都要这个布局 ✓）；
* **越界像素**：贴图/掩罩只在**可见区域**落笔 ✓（不越界写 ✗）；形态学/滤波在边界按
  **replicate** 延拓 ✓（靠它保住「常量图进 ⇒ 常量图出」✓）。

## 事实来源（**语义对齐参考节点，源码一行没抄** ✓）

``reference/ComfyUI`` 是 **GPL-3.0** ✗ ⇒ 只对齐**行为** ✓：参数名与语义按它的节点 schema 核过 ✓
（``nodes.py`` 与 ``comfy_extras/nodes_{images,mask,compositing,morphology,canny,post_processing}.py`` ✓），
**实现全部自研** ✓。

⚠️ **明确不承诺**（别当成「与参考逐像素相同」✗）：``lanczos`` / ``bislerp`` / ``canny`` / 形态学 ——
参考走 PIL / kornia / scipy ✗ 而本仓没有这些依赖 ✓ ⇒ 按**公开定义**自研 ✓（Lanczos-3 窗、
slerp 球面插值、Canny 四步、核内极值 ✓），边界延拓与插值细节是**本仓口径** ✓。
``quantize`` 用 **PIL 调色板** ✓（Pillow 是本仓既有依赖 ✓），没装时**明确报错** ✗。
"""
from __future__ import annotations

import math
from typing import Any, Sequence

__all__ = [
    "ALPHA_EPS",
    "BLEND_MODES",
    "CANNY_KERNEL_SIZE",
    "CANNY_NOISE_FLOOR",
    "CANNY_SIGMA",
    "CROP_METHODS",
    "MASK_COMPOSITE_OPS",
    "MORPHOLOGY_OPS",
    "PORTER_DUFF_MODES",
    "UPSCALE_METHODS",
    "ImageOpsError",
    "alpha_to_mask",
    "bchw_to_image",
    "bislerp",
    "bislerp_coords",
    "canny",
    "canny_edge_image",
    "common_upscale",
    "composite",
    "crop_mask",
    "feather_mask",
    "gaussian_blur",
    "gaussian_kernel",
    "grow_mask",
    "has_pil",
    "has_torch",
    "image_alpha_fix",
    "image_batch",
    "image_blend",
    "image_color_to_mask",
    "image_composite_masked",
    "image_crop",
    "image_flip",
    "image_invert",
    "image_pad_for_outpaint",
    "image_rgb_to_yuv",
    "image_rotate",
    "image_scale",
    "image_scale_by",
    "image_scale_to_max_dimension",
    "image_scale_to_total_pixels",
    "image_stitch",
    "image_to_bchw",
    "image_to_mask",
    "image_yuv_to_rgb",
    "invert_mask",
    "join_image_with_alpha",
    "lanczos_resize",
    "mask_composite",
    "mask_to_alpha",
    "mask_to_bchw1",
    "mask_to_image",
    "morphology",
    "porter_duff_composite",
    "porter_duff_image_composite",
    "quantize",
    "repeat_to_batch_size",
    "resize_and_pad",
    "resize_mask",
    "rgb_to_grayscale",
    "rgb_to_ycbcr",
    "sharpen",
    "solid_mask",
    "split_image_with_alpha",
    "threshold_mask",
    "ycbcr_to_rgb",
]

#: 重采样方法 —— 前四个是 torch 内建名 ✓，后两个（``bislerp``/``lanczos``）是本仓自研 ✓。
UPSCALE_METHODS: tuple[str, ...] = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp", "lanczos")
#: 裁剪口径：``disabled`` = 直接拉伸（会变形 ✓）；``center`` = 先居中裁成目标比例再缩 ✓。
CROP_METHODS: tuple[str, ...] = ("disabled", "center")
#: 混合模式 ✓（对齐参考 ``ImageBlend`` ✓）。
BLEND_MODES: tuple[str, ...] = ("normal", "multiply", "screen", "overlay", "soft_light", "difference")
#: Porter-Duff 的 18 个模式 ✓（名字与顺序对齐参考枚举 ✓）。
PORTER_DUFF_MODES: tuple[str, ...] = (
    "ADD", "CLEAR", "DARKEN", "DST", "DST_ATOP", "DST_IN", "DST_OUT", "DST_OVER", "LIGHTEN",
    "MULTIPLY", "OVERLAY", "SCREEN", "SRC", "SRC_ATOP", "SRC_IN", "SRC_OUT", "SRC_OVER", "XOR",
)
#: 掩罩合成的六个运算 ✓（对齐参考 ``MaskComposite`` ✓）。
MASK_COMPOSITE_OPS: tuple[str, ...] = ("multiply", "add", "subtract", "and", "or", "xor")
#: 形态学算子 ✓（对齐参考 ``Morphology`` ✓）。
MORPHOLOGY_OPS: tuple[str, ...] = ("erode", "dilate", "open", "close", "gradient", "top_hat", "bottom_hat")
#: 合成后回除 alpha 的**下限** ✓（低于它视为全透明 ⇒ 输出 0 ✓ 不做 0 除 ✗）。
ALPHA_EPS = 1e-5

try:  # pragma: no cover - 环境相关 ✓
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    F = None
    _HAS_TORCH = False

try:  # pragma: no cover - 环境相关 ✓
    from PIL import Image as _PILImage

    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _PILImage = None
    _HAS_PIL = False


class ImageOpsError(RuntimeError):
    """算子**跑不了**（缺 torch/Pillow ✓、参数非法 ✓、形状对不上 ✓）⇒ 当场报 ✗ 不静默降级 ✓。"""


def has_torch() -> bool:
    """装了 torch 吗 ✓（自检据此 SKIP ✓ —— **没跑 ≠ 绿** ✗）。"""
    return _HAS_TORCH


def has_pil() -> bool:
    """装了 Pillow 吗 ✓（只有 :func:`quantize` 需要 ✓）。"""
    return _HAS_PIL


def _need_torch() -> None:
    if not _HAS_TORCH:
        raise ImageOpsError("图像算子需要 torch（本机未装 ⇒ 这些算子跑不了 ✓ 不降级 ✗）")


# ────────────────────────────── 布局 / 口径工具 ──────────────────────────────


def image_to_bchw(image: Any) -> Any:
    """``(B,H,W,C)`` → ``(B,C,H,W)`` ✓（内部布局 ✓）。"""
    _need_torch()
    return image.movedim(-1, 1)


def bchw_to_image(samples: Any) -> Any:
    """``(B,C,H,W)`` → ``(B,H,W,C)`` ✓（对外布局 ✓）。"""
    _need_torch()
    return samples.movedim(1, -1)


def mask_to_bchw1(mask: Any) -> Any:
    """掩罩 ``(B,H,W)`` → ``(B,1,H,W)`` ✓（插值要的布局 ✓）；已是 4 维则原样返回 ✓。"""
    _need_torch()
    if mask.dim() == 4:
        return mask
    return mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))


def repeat_to_batch_size(tensor: Any, batch_size: int, dim: int = 0) -> Any:
    """把 ``tensor`` 在 ``dim`` 上凑成 ``batch_size`` ✓：多了**截断** ✓、少了**循环重复** ✓。

    「一图对一批」的**唯一**口径 ✓（贴图/合成/拼接都走它 ✓）—— 与参考同名函数同义 ✓。
    """
    _need_torch()
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    if tensor.shape[dim] < batch_size:
        reps = [1] * tensor.dim()
        reps[dim] = math.ceil(batch_size / tensor.shape[dim])
        return tensor.repeat(*reps).narrow(dim, 0, batch_size)
    return tensor


def image_alpha_fix(destination: Any, source: Any) -> tuple[Any, Any]:
    """把两图**通道数**对齐 ✓：source 多通道 ⇒ 砍 ✓；少通道 ⇒ 补一列 **1.0**（不透明 ✓）。"""
    _need_torch()
    if destination.shape[-1] < source.shape[-1]:
        source = source[..., : destination.shape[-1]]
    elif destination.shape[-1] > source.shape[-1]:
        source = F.pad(source, (0, 1))
        source[..., -1] = 1.0
    return destination, source


def mask_to_alpha(mask: Any) -> Any:
    """掩罩 → **alpha** ✓（``1 - m`` ✓）—— 这次换位**全仓只在这里**做 ✓（口径唯一 ✓）。"""
    _need_torch()
    return 1.0 - mask


def alpha_to_mask(alpha: Any) -> Any:
    """**alpha** → 掩罩 ✓（``1 - a`` ✓）。"""
    _need_torch()
    return 1.0 - alpha


def resize_mask(mask: Any, shape: Sequence[int]) -> Any:
    """掩罩缩到 ``(h, w)`` ✓（双线性 ✓）—— 维度按输入回推 ✓（3 维进 3 维出 ✓）。"""
    _need_torch()
    out = F.interpolate(mask_to_bchw1(mask), size=(int(shape[0]), int(shape[1])), mode="bilinear")
    return out.squeeze(1) if mask.dim() == 3 else out


# ────────────────────────────── 重采样（缩放的内核） ──────────────────────────────


def bislerp_coords(in_len: int, out_len: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """``align_corners=False`` 下的**源坐标** ✓ —— 返回 ``(整数下界, 小数比)`` ✓（纯 Python ✓）。

    口径：输出像素 ``i`` 的采样中心 = ``(i + 0.5) · in/out − 0.5`` ✓，越界夹到 ``[0, in−1]`` ✓
    （等价于边界延拓 ✓）—— 与 ``F.interpolate(mode='bilinear')`` 的取点一致 ✓。
    """
    scale = in_len / out_len
    top = float(max(in_len - 1, 0))
    idx: list[int] = []
    frac: list[float] = []
    for i in range(out_len):
        pos = min(max((i + 0.5) * scale - 0.5, 0.0), top)
        lo = int(math.floor(pos))
        idx.append(lo)
        frac.append(pos - lo)
    return tuple(idx), tuple(frac)


def _coords_tensor(in_len: int, out_len: int, device: Any) -> tuple[Any, Any]:
    """张量版坐标 ✓（与 :func:`bislerp_coords` 同一公式 ✓）。"""
    _need_torch()
    pos = (torch.arange(out_len, dtype=torch.float64, device=device) + 0.5) * (in_len / out_len) - 0.5
    pos = pos.clamp(0.0, float(max(in_len - 1, 0)))
    lo = pos.floor()
    return lo.long(), pos - lo


def _slerp_rows(a: Any, b: Any, r: Any) -> Any:
    """按**通道向量**球面插值（slerp）✓、模长线性插值 ✓ —— ``a,b: (M,C)``、``r: (M,1)`` ✓。

    退化（方向同向/反向 ⇒ ``sin ω → 0``）⇒ 退回**线性插值** ✓（数学极限即线性 ✓；
    **不**照搬参考里「同向直接取 a」那一步 ✗）。
    """
    _need_torch()
    na = a.norm(dim=-1, keepdim=True)
    nb = b.norm(dim=-1, keepdim=True)
    ea = torch.where(na > 0, a / na.clamp_min(1e-12), torch.zeros_like(a))
    eb = torch.where(nb > 0, b / nb.clamp_min(1e-12), torch.zeros_like(b))
    dot = (ea * eb).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    so = torch.where(so.abs() < 1e-6, torch.ones_like(so), so)
    dirv = torch.sin((1.0 - r) * omega) / so * ea + torch.sin(r * omega) / so * eb
    res = dirv * (na * (1.0 - r) + nb * r)
    linear = a * (1.0 - r) + b * r
    return torch.where(dot.abs() > 1.0 - 1e-5, linear, res)


def _bislerp_axis(x: Any, out_len: int, axis: int) -> Any:
    """沿 ``axis``（``-1``=W ✓ / ``-2``=H ✓）做一次「双线性取点 + slerp 混合」✓。"""
    _need_torch()
    in_len = int(x.shape[axis])
    n, c = int(x.shape[0]), int(x.shape[1])
    if in_len == out_len:
        return x
    lo, frac = _coords_tensor(in_len, out_len, x.device)
    i0 = lo.clamp(0, in_len - 1)
    i1 = (lo + 1).clamp(0, in_len - 1)
    if axis == -1:
        h = int(x.shape[2])
        a = x.index_select(-1, i0).movedim(1, -1).reshape(-1, c)
        b = x.index_select(-1, i1).movedim(1, -1).reshape(-1, c)
        r = frac.unsqueeze(0).expand(n * h, out_len).reshape(-1, 1)
        return _slerp_rows(a, b, r).reshape(n, h, out_len, c).movedim(-1, 1)
    w = int(x.shape[3])
    a = x.index_select(-2, i0).movedim(1, -1).reshape(-1, c)
    b = x.index_select(-2, i1).movedim(1, -1).reshape(-1, c)
    r = frac.unsqueeze(-1).expand(out_len, w).reshape(-1, 1).repeat(n, 1)
    return _slerp_rows(a, b, r).reshape(n, out_len, w, c).movedim(-1, 1)


def bislerp(samples: Any, width: int, height: int) -> Any:
    """``bislerp`` ✓：先在 **W**、再在 **H** 做「双线性 + slerp」✓（纯 torch ✓ 无 PIL/kornia ✓）。"""
    _need_torch()
    out = _bislerp_axis(samples.float(), int(width), -1)
    out = _bislerp_axis(out, int(height), -2)
    return out.to(samples.dtype)


def _lanczos_window(x: Any, a: int = 3) -> Any:
    """Lanczos-a 窗 ✓：``sinc(x)·sinc(x/a)`` ✓（``|x| ≥ a`` ⇒ 0 ✓；``x → 0`` ⇒ 1 ✓）。"""
    _need_torch()
    ax = x.abs()
    inside = ax < (a - 1e-8)
    px = math.pi * ax.clamp_min(1e-12)
    core = torch.where(ax < 1e-8, torch.ones_like(ax), torch.sin(px) / px)
    wide = ax / a
    pw = math.pi * wide.clamp_min(1e-12)
    win = torch.where(ax < 1e-8, torch.ones_like(ax), torch.sin(pw) / pw)
    return torch.where(inside, core * win, torch.zeros_like(ax))


def _lanczos_axis(x: Any, out_len: int, axis: int) -> Any:
    """单轴 Lanczos-3 重采样 ✓（可分离 ✓ 逐抽头累加 ✓ ⇒ 内存与抽头数成正比 ✓）。"""
    _need_torch()
    in_len = int(x.shape[axis])
    if in_len == out_len:
        return x
    scale = in_len / out_len
    support = 3.0 * scale if scale >= 1.0 else 3.0
    radius = int(math.ceil(support))
    taps = 2 * radius
    centers = (torch.arange(out_len, dtype=torch.float64, device=x.device) + 0.5) * scale
    base = centers.floor().long() - (radius - 1)
    idx = base.unsqueeze(1) + torch.arange(taps, device=x.device).unsqueeze(0)
    dist = (idx.to(torch.float64) - centers.unsqueeze(1)) * (1.0 / scale if scale >= 1.0 else 1.0)
    w = _lanczos_window(dist.to(x.dtype))
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)  # 归一化 ⇒ 常量图不变 ✓
    idx_c = idx.clamp(0, in_len - 1)  # 越界抽头夹到边界 ✓
    shape = [-1 if d == (axis % x.dim()) else 1 for d in range(x.dim())]
    acc = None
    for t in range(taps):
        part = x.index_select(axis, idx_c[:, t]) * w[:, t].reshape(shape)
        acc = part if acc is None else acc + part
    return acc


def lanczos_resize(samples: Any, width: int, height: int) -> Any:
    """**Lanczos-3** 重采样 ✓（纯 torch ✓ 无 PIL ✓）—— 缩小按缩放比展宽核 ✓（抗锯齿 ✓）。"""
    _need_torch()
    out = _lanczos_axis(samples.float(), int(width), -1)
    out = _lanczos_axis(out, int(height), -2)
    return out.to(samples.dtype)


def common_upscale(samples: Any, width: int, height: int, upscale_method: str = "bilinear",
                   crop: str = "disabled") -> Any:
    """**统一的缩放入口** ✓（吃 ``(N,C,H,W)`` ✓，也吃视频潜空间 ``(B,C,T,H,W)`` ✓）。

    * ``crop='center'`` ⇒ 先按目标宽高比**居中裁** ✓（再缩 ⇒ **不变形** ✓）；
    * 方法见 :data:`UPSCALE_METHODS` ✓；
    * **高阶张量** ⇒ 把 ``T`` 折进 batch **逐帧**缩 ✓（⚠️ 绝不沿时间维插值 ✗）—— 缩完原样折回 ✓。
    """
    _need_torch()
    if upscale_method not in UPSCALE_METHODS:
        raise ImageOpsError(f"不支持的重采样方法：{upscale_method!r}（可选：{UPSCALE_METHODS}）")
    if crop not in CROP_METHODS:
        raise ImageOpsError(f"不支持的裁剪口径：{crop!r}（可选：{CROP_METHODS}）")
    width, height = max(1, int(width)), max(1, int(height))
    orig_shape = tuple(samples.shape)
    if len(orig_shape) > 4:
        folded = samples.reshape(orig_shape[0], orig_shape[1], -1, orig_shape[-2], orig_shape[-1])
        samples = folded.movedim(2, 1).reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
    s = samples
    if crop == "center":
        old_w, old_h = int(s.shape[-1]), int(s.shape[-2])
        old_aspect, new_aspect = old_w / old_h, width / height
        x = y = 0
        if old_aspect > new_aspect:
            x = round((old_w - old_w * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_h - old_h * (old_aspect / new_aspect)) / 2)
        s = s.narrow(-2, y, old_h - y * 2).narrow(-1, x, old_w - x * 2)
    if upscale_method == "bislerp":
        out = bislerp(s, width, height)
    elif upscale_method == "lanczos":
        out = lanczos_resize(s, width, height)
    else:
        out = F.interpolate(s.to(torch.float32), size=(height, width), mode=upscale_method)
    if len(orig_shape) == 4:
        return out
    out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
    return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))


# ────────────────────────────── 缩放 / 裁剪 / 旋转 / 拼接 ──────────────────────────────


def image_scale(image: Any, upscale_method: str = "bilinear", width: int = 512, height: int = 512,
                crop: str = "disabled") -> Any:
    """``ImageScale`` ✓：宽高**任一为 0** ⇒ 按另一维**保比例**推 ✓（最少 1 px ✓）。

    ``width == height == 0`` ⇒ 原样返回 ✓（与参考一致 ✓）。
    """
    _need_torch()
    if width == 0 and height == 0:
        return image
    samples = image_to_bchw(image)
    if width == 0:
        width = max(1, round(int(samples.shape[3]) * height / int(samples.shape[2])))
    elif height == 0:
        height = max(1, round(int(samples.shape[2]) * width / int(samples.shape[3])))
    return bchw_to_image(common_upscale(samples, int(width), int(height), upscale_method, crop))


def image_scale_by(image: Any, upscale_method: str = "bilinear", scale_by: float = 1.0) -> Any:
    """``ImageScaleBy`` ✓：按倍数缩 ✓（尺寸 = ``round(原尺寸 × 倍数)`` ✓，**不居中裁** ✓）。"""
    _need_torch()
    samples = image_to_bchw(image)
    width = round(int(samples.shape[3]) * float(scale_by))
    height = round(int(samples.shape[2]) * float(scale_by))
    return bchw_to_image(common_upscale(samples, width, height, upscale_method, "disabled"))


def image_scale_to_total_pixels(image: Any, upscale_method: str = "bilinear", megapixels: float = 1.0,
                                resolution_steps: int = 1) -> Any:
    """``ImageScaleToTotalPixels`` ✓：按**总像素预算**定尺寸 ✓ 并按 ``resolution_steps`` 对齐 ✓。"""
    _need_torch()
    samples = image_to_bchw(image)
    w, h = int(samples.shape[3]), int(samples.shape[2])
    scale_by = math.sqrt(float(megapixels) * 1024 * 1024 / (w * h))
    step = max(1, int(resolution_steps))
    width = int(round(w * scale_by / step) * step)
    height = int(round(h * scale_by / step) * step)
    return bchw_to_image(common_upscale(samples, width, height, upscale_method, "disabled"))


def image_scale_to_max_dimension(image: Any, upscale_method: str = "lanczos",
                                 largest_size: int = 512) -> Any:
    """``ImageScaleToMaxDimension`` ✓：把**长边**缩到 ``largest_size`` ✓（保比例 ✓ 不动变形 ✓）。"""
    _need_torch()
    samples = image_to_bchw(image)
    width, height = int(samples.shape[3]), int(samples.shape[2])
    if height > width:
        width, height = round((width / height) * largest_size), int(largest_size)
    elif width > height:
        height, width = round((height / width) * largest_size), int(largest_size)
    else:
        width = height = int(largest_size)
    return bchw_to_image(common_upscale(samples, width, height, upscale_method, "disabled"))


def resize_and_pad(image: Any, target_width: int = 512, target_height: int = 512,
                   padding_color: str = "black", interpolation: str = "area") -> Any:
    """``ResizeAndPadImage`` ✓：**保比例**缩到能装下 ✓，余下**居中填边** ✓（黑/白 ✓）。

    ⚠️ 缩后的尺寸用 ``int()`` **截断** ✓（对齐参考口径 ✓ ⇒ 边缘最多差 1 px 的余地 ✓）。
    """
    _need_torch()
    if padding_color not in ("black", "white"):
        raise ImageOpsError(f"padding_color 只能是 black/white ✓（收到 {padding_color!r} ✗）")
    b, h, w, c = image.shape
    tw, th = int(target_width), int(target_height)
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = common_upscale(image.permute(0, 3, 1, 2), new_w, new_h, interpolation, "disabled")
    padded = torch.full((b, c, th, tw), 0.0 if padding_color == "black" else 1.0,
                        dtype=resized.dtype, device=resized.device)
    y0, x0 = (th - new_h) // 2, (tw - new_w) // 2
    padded[:, :, y0:y0 + new_h, x0:x0 + new_w] = resized
    return padded.permute(0, 2, 3, 1)


def image_crop(image: Any, width: int = 512, height: int = 512, x: int = 0, y: int = 0) -> Any:
    """``ImageCrop`` ✓：从 ``(x, y)`` 裁 ``width × height`` ✓。

    ⚠️ 起点被夹在**图内** ✓（``x ≤ W−1`` ✓）且裁框可能**小于**请求尺寸 ✗（贴到边界就会 ✓）——
    与参考一致 ✓（它也是直接切片 ✓，不做补边 ✗）。
    """
    _need_torch()
    x = min(int(x), int(image.shape[2]) - 1)
    y = min(int(y), int(image.shape[1]) - 1)
    return image[:, y:y + int(height), x:x + int(width), :]


def image_rotate(image: Any, rotation: str = "none") -> Any:
    """``ImageRotate`` ✓：``none`` / ``90`` / ``180`` / ``270 degrees`` ✓（``rot90`` ✓ 尺寸会换 ✓）。"""
    _need_torch()
    k = 0
    if rotation.startswith("90"):
        k = 1
    elif rotation.startswith("180"):
        k = 2
    elif rotation.startswith("270"):
        k = 3
    elif rotation != "none":
        raise ImageOpsError(f"rotation 只能是 none/90/180/270 degrees ✓（收到 {rotation!r} ✗）")
    return torch.rot90(image, k=k, dims=[2, 1])


def image_flip(image: Any, flip_method: str = "y-axis: horizontally") -> Any:
    """``ImageFlip`` ✓：``x-axis: vertically`` ⇒ 上下翻 ✓；``y-axis: horizontally`` ⇒ 左右翻 ✓。"""
    _need_torch()
    if flip_method.startswith("x"):
        return torch.flip(image, dims=[1])
    if flip_method.startswith("y"):
        return torch.flip(image, dims=[2])
    raise ImageOpsError(f"flip_method 只能是 x-axis: vertically / y-axis: horizontally ✓（{flip_method!r} ✗）")


def image_invert(image: Any) -> Any:
    """``ImageInvert`` ✓：``1 − image`` ✓ —— ⚠️ 第 4 通道是 **alpha** ⇒ **不反** ✓（原样保留 ✓）。"""
    _need_torch()
    out = 1.0 - image
    if image.shape[-1] == 4:
        out[..., 3] = image[..., 3]
    return out


def image_batch(image1: Any, image2: Any) -> Any:
    """``ImageBatch`` ✓：**沿 batch 维并** ✓ —— 通道不同补一列 1.0 ✓、尺寸不同按 ``bilinear+center`` 对齐 ✓。"""
    _need_torch()
    if image1.shape[-1] != image2.shape[-1]:
        if image1.shape[-1] > image2.shape[-1]:
            image2 = F.pad(image2, (0, 1), mode="constant", value=1.0)
        else:
            image1 = F.pad(image1, (0, 1), mode="constant", value=1.0)
    if tuple(image1.shape[1:]) != tuple(image2.shape[1:]):
        image2 = bchw_to_image(common_upscale(image_to_bchw(image2), int(image1.shape[2]),
                                              int(image1.shape[1]), "bilinear", "center"))
    return torch.cat((image1, image2), dim=0)


def image_stitch(image1: Any, direction: str = "right", match_image_size: bool = True,
                 spacing_width: int = 0, spacing_color: str = "white", image2: Any = None) -> Any:
    """``ImageStitch`` ✓：把 ``image2`` 按 ``direction`` 拼到 ``image1`` ✓（**空间**拼接 ✓）。

    * ``match_image_size`` ⇒ 先按方向把 image2 **等比**缩到贴合 ✓（走 **lanczos** ✓）；
    * 不匹配尺寸时 ⇒ 只在**拼接方向的垂直边**上**居中补边** ✓（另一个方向必相等 ✓）；
    * ``spacing_width > 0`` ⇒ 中间插一条色带 ✓（宽度**取偶** ✓、颜色可选 ✓）；
    * 通道数不同 ⇒ 补一列 1.0 ✓；batch 不同 ⇒ **复制末帧**补齐 ✓（不是循环 ✓，与参考一致 ✓）。
    """
    _need_torch()
    if direction not in ("right", "down", "left", "up"):
        raise ImageOpsError(f"direction 只能是 right/down/left/up ✓（收到 {direction!r} ✗）")
    if spacing_color not in ("white", "black", "red", "green", "blue"):
        raise ImageOpsError(f"spacing_color 不支持：{spacing_color!r} ✗")
    if image2 is None:
        return image1
    if image1.shape[0] != image2.shape[0]:
        target = max(int(image1.shape[0]), int(image2.shape[0]))
        if image1.shape[0] < target:
            image1 = torch.cat([image1, image1[-1:].repeat(target - image1.shape[0], 1, 1, 1)])
        if image2.shape[0] < target:
            image2 = torch.cat([image2, image2[-1:].repeat(target - image2.shape[0], 1, 1, 1)])
    horiz = direction in ("left", "right")
    if match_image_size:
        h1, w1 = int(image1.shape[1]), int(image1.shape[2])
        h2, w2 = int(image2.shape[1]), int(image2.shape[2])
        aspect = w2 / h2
        if horiz:
            target_w, target_h = int(h1 * aspect), h1
        else:
            target_w, target_h = w1, int(w1 / aspect)
        image2 = bchw_to_image(common_upscale(image_to_bchw(image2), target_w, target_h, "lanczos", "disabled"))
    color_map: dict[str, Any] = {"white": 1.0, "black": 0.0, "red": (1.0, 0.0, 0.0),
                                 "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}
    color_val = color_map[spacing_color]
    if not match_image_size:  # 只对齐「拼接方向的垂直边」✓（另一个方向不碰 ✗）
        h1, w1 = int(image1.shape[1]), int(image1.shape[2])
        h2, w2 = int(image2.shape[1]), int(image2.shape[2])
        pad_value = 0.0 if isinstance(color_val, tuple) else float(color_val)
        if horiz and h1 != h2:
            target = max(h1, h2)
            for img, cur in (("1", h1), ("2", h2)):
                if cur < target:
                    top = (target - cur) // 2
                    pad = (0, 0, 0, 0, top, target - cur - top)
                    if img == "1":
                        image1 = F.pad(image1, pad, mode="constant", value=pad_value)
                    else:
                        image2 = F.pad(image2, pad, mode="constant", value=pad_value)
        elif not horiz and w1 != w2:
            target = max(w1, w2)
            for img, cur in (("1", w1), ("2", w2)):
                if cur < target:
                    left = (target - cur) // 2
                    pad = (0, 0, left, target - cur - left)
                    if img == "1":
                        image1 = F.pad(image1, pad, mode="constant", value=pad_value)
                    else:
                        image2 = F.pad(image2, pad, mode="constant", value=pad_value)
    if image1.shape[-1] != image2.shape[-1]:
        target_c = max(int(image1.shape[-1]), int(image2.shape[-1]))
        if image1.shape[-1] < target_c:
            image1 = torch.cat([image1, torch.ones(*image1.shape[:-1], target_c - image1.shape[-1],
                                                  device=image1.device, dtype=image1.dtype)], dim=-1)
        if image2.shape[-1] < target_c:
            image2 = torch.cat([image2, torch.ones(*image2.shape[:-1], target_c - image2.shape[-1],
                                                  device=image2.device, dtype=image2.dtype)], dim=-1)
    spacing = None
    spacing_width = int(spacing_width)
    if spacing_width > 0:
        spacing_width += spacing_width % 2  # 取偶 ✓（与参考一致 ✓）
        if horiz:
            shape = (image1.shape[0], max(int(image1.shape[1]), int(image2.shape[1])),
                     spacing_width, image1.shape[-1])
        else:
            shape = (image1.shape[0], spacing_width, max(int(image1.shape[2]), int(image2.shape[2])),
                     image1.shape[-1])
        spacing = torch.full(shape, 0.0, device=image1.device, dtype=image1.dtype)
        if isinstance(color_val, tuple):
            for i, c in enumerate(color_val):
                if i < spacing.shape[-1]:
                    spacing[..., i] = c
        else:
            spacing[..., : min(3, spacing.shape[-1])] = float(color_val)
        if spacing.shape[-1] == 4:
            spacing[..., 3] = 1.0
    parts = [image2, image1] if direction in ("left", "up") else [image1, image2]
    if spacing is not None:
        parts.insert(1, spacing)
    return torch.cat(parts, dim=2 if horiz else 1)


def image_pad_for_outpaint(image: Any, left: int = 0, top: int = 0, right: int = 0,
                           bottom: int = 0, feathering: int = 40) -> tuple[Any, Any]:
    """``ImagePadForOutpaint`` ✓：往外扩画布 ✓（填 **0.5 中灰** ✓）+ 返回配套**掩罩** ✓。

    掩罩口径 ✓：**新扩出来的区域 = 1（要生成 ✓）**、**原图区 = 0** ✓；``feathering`` 把
    原图区四边做成**二次衰减** ✓（``v = ((f − d) / f)²`` ✓，``d`` = 到最近「已扩边」的距离 ✓）；
    ⚠️ 只有当 ``2·feathering`` **小于**原图宽高时才feather ✓（否则保持硬边 ✓，与参考一致 ✓）。
    """
    _need_torch()
    b, h, w, c = image.shape
    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    new_h, new_w = h + top + bottom, w + left + right
    new_image = torch.full((b, new_h, new_w, c), 0.5, dtype=torch.float32, device=image.device)
    new_image[:, top:top + h, left:left + w, :] = image
    mask = torch.ones((new_h, new_w), dtype=torch.float32, device=image.device)
    inner = torch.zeros((h, w), dtype=torch.float32, device=image.device)
    if feathering > 0 and feathering * 2 < h and feathering * 2 < w:
        rr = torch.arange(h, dtype=torch.float32, device=image.device).unsqueeze(1)
        cc = torch.arange(w, dtype=torch.float32, device=image.device).unsqueeze(0)
        d_top = rr if top != 0 else torch.full_like(rr, float(h))
        d_bottom = (h - rr) if bottom != 0 else torch.full_like(rr, float(h))
        d_left = cc if left != 0 else torch.full_like(cc, float(w))
        d_right = (w - cc) if right != 0 else torch.full_like(cc, float(w))
        d = torch.minimum(torch.minimum(d_top, d_bottom), torch.minimum(d_left, d_right))
        v = (feathering - d).clamp_min(0.0) / feathering
        inner = torch.where(d >= feathering, torch.zeros_like(d), v * v)
    mask[top:top + h, left:left + w] = inner
    return new_image, mask.unsqueeze(0)


# ────────────────────────────── 贴图 / 合成 ──────────────────────────────


def composite(destination: Any, source: Any, x: int, y: int, mask: Any = None, multiplier: int = 1,
              resize_source: bool = False) -> Any:
    """**通用贴图**（``(N,C,H,W)`` ✓）：把 ``source`` 贴到 ``destination`` 的 ``(x, y)`` ✓。

    这是「图像贴图」与「潜空间贴图」**共用**的那一支 ✓（差别只在 ``multiplier``：图像 1 ✓、潜空间 8 ✓）：
    ``x``/``y`` 以 ``multiplier`` 为单位换算成左上角 ✓、允许负坐标 ⇒ 左上出界时只贴**可见部分** ✓；
    ``mask`` 缺省 ⇒ 全 1（**整块覆盖** ✓）；给了 ⇒ ``mask·source + (1−mask)·destination`` ✓
    （掩罩口径：**1 = 覆盖/挖掉** ✓）。
    """
    _need_torch()
    source = source.to(destination.device)
    if resize_source:
        source = F.interpolate(source, size=(destination.shape[-2], destination.shape[-1]),
                               mode="bilinear")
    source = repeat_to_batch_size(source, int(destination.shape[0]))
    mult = max(1, int(multiplier))
    x = max(-int(source.shape[-1]) * mult, min(int(x), int(destination.shape[-1]) * mult))
    y = max(-int(source.shape[-2]) * mult, min(int(y), int(destination.shape[-2]) * mult))
    left, top = x // mult, y // mult
    right, bottom = left + int(source.shape[-1]), top + int(source.shape[-2])
    if mask is None:
        m = torch.ones((source.shape[0], 1, source.shape[-2], source.shape[-1]),
                       dtype=source.dtype, device=destination.device)
    else:
        m = mask.to(destination.device)
        m = F.interpolate(mask_to_bchw1(m), size=(source.shape[-2], source.shape[-1]),
                          mode="bilinear")
        m = repeat_to_batch_size(m, int(source.shape[0]))
    visible_w = int(destination.shape[-1]) - left + min(0, x)
    visible_h = int(destination.shape[-2]) - top + min(0, y)
    if visible_w <= 0 or visible_h <= 0:
        return destination  # 完全出界 ⇒ 什么都不贴 ✓（不越界写 ✗）
    m = m[:, :, :visible_h, :visible_w]
    if m.dim() < source.dim():
        m = m.unsqueeze(1)
    src_part = m * source[..., :visible_h, :visible_w]
    dst_part = (1.0 - m) * destination[..., top:bottom, left:right]
    destination[..., top:bottom, left:right] = src_part + dst_part
    return destination


def image_composite_masked(destination: Any, source: Any, x: int = 0, y: int = 0,
                           resize_source: bool = False, mask: Any = None) -> Any:
    """``ImageCompositeMasked`` ✓：**图像**层面的贴图 ✓（先对齐 alpha 通道 ✓ 再走 :func:`composite` ✓）。"""
    _need_torch()
    destination, source = image_alpha_fix(destination, source)
    out = composite(destination.clone().movedim(-1, 1), source.movedim(-1, 1), x, y, mask, 1, resize_source)
    return out.movedim(1, -1)


def image_blend(image1: Any, image2: Any, blend_factor: float = 0.5,
                blend_mode: str = "normal") -> Any:
    """``ImageBlend`` ✓：先在两张图上算**混合函数** ✓，再按 ``blend_factor`` 与原图线性混合 ✓。

    ⚠️ 第 4 通道是 alpha ⇒ 混合结果**只取 image1 的 alpha** ✓（不参与色彩混合 ✓）。
    尺寸不同 ⇒ image2 走 ``bicubic + center`` 对齐 ✓。
    """
    _need_torch()
    if blend_mode not in BLEND_MODES:
        raise ImageOpsError(f"不支持的混合模式：{blend_mode!r}（可选：{BLEND_MODES}）")
    image1, image2 = image_alpha_fix(image1, image2)
    image2 = image2.to(image1.device)
    if tuple(image1.shape) != tuple(image2.shape):
        image2 = bchw_to_image(common_upscale(image_to_bchw(image2), int(image1.shape[2]),
                                              int(image1.shape[1]), "bicubic", "center"))
    a = image1[..., :3]
    b = image2[..., :3]
    if blend_mode == "normal":
        mixed = b
    elif blend_mode == "multiply":
        mixed = a * b
    elif blend_mode == "screen":
        mixed = 1.0 - (1.0 - a) * (1.0 - b)
    elif blend_mode == "overlay":
        mixed = torch.where(a <= 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    elif blend_mode == "soft_light":
        g = torch.where(a <= 0.25, ((16.0 * a - 12.0) * a + 4.0) * a, torch.sqrt(a.clamp_min(0.0)))
        mixed = torch.where(b <= 0.5, a - (1.0 - 2.0 * b) * a * (1.0 - a), a + (2.0 * b - 1.0) * (g - a))
    else:  # difference
        mixed = a - b
    out = a * (1.0 - float(blend_factor)) + mixed * float(blend_factor)
    out = out.clamp(0.0, 1.0)
    if image1.shape[-1] == 4:
        out = torch.cat([out, image1[..., 3:4]], dim=-1)
    return out


def porter_duff_composite(src_image: Any, src_alpha: Any, dst_image: Any, dst_alpha: Any,
                          mode: str) -> tuple[Any, Any]:
    """**Porter-Duff** 合成 ✓（``(H,W,C)`` + 掩罩 ``(H,W,1)`` ⇒ 出 ``(图像, 掩罩)`` ✓）。

    ⚠️ 两个换算**必须按顺序**做 ✓（顺序错了整条链都是错的 ✗）：
    ``alpha = 1 − mask`` ✓ ⇒ **预乘** ✓ ⇒ 按 ``mode`` 合成 ✓ ⇒ **回除 alpha** ✓（``≤ ALPHA_EPS`` ⇒ 0 ✓）
    ⇒ 再 ``mask = 1 − alpha`` ✓。
    ``mode`` 见 :data:`PORTER_DUFF_MODES` ✓（18 个 ✓ 名字对齐参考枚举 ✓）。
    """
    _need_torch()
    if mode not in PORTER_DUFF_MODES:
        raise ImageOpsError(f"不支持的 Porter-Duff 模式：{mode!r}（可选：{PORTER_DUFF_MODES}）")
    sa = mask_to_alpha(src_alpha)
    da = mask_to_alpha(dst_alpha)
    src_image = src_image * sa  # 预乘 ✓
    dst_image = dst_image * da
    one = torch.ones_like(sa)
    if mode == "ADD":
        out_alpha = (sa + da).clamp(0.0, 1.0)
        out_image = (src_image + dst_image).clamp(0.0, 1.0)
    elif mode == "CLEAR":
        out_alpha, out_image = torch.zeros_like(da), torch.zeros_like(dst_image)
    elif mode == "DARKEN":
        out_alpha = sa + da - sa * da
        out_image = (one - da) * src_image + (one - sa) * dst_image + torch.minimum(src_image * da, dst_image * sa)
    elif mode == "DST":
        out_alpha, out_image = da, dst_image
    elif mode == "DST_ATOP":
        out_alpha = sa
        out_image = sa * dst_image + (one - da) * src_image
    elif mode == "DST_IN":
        out_alpha = sa * da
        out_image = dst_image * sa
    elif mode == "DST_OUT":
        out_alpha = (one - sa) * da
        out_image = (one - sa) * dst_image
    elif mode == "DST_OVER":
        out_alpha = da + (one - da) * sa
        out_image = dst_image + (one - da) * src_image
    elif mode == "LIGHTEN":
        out_alpha = sa + da - sa * da
        out_image = (one - da) * src_image + (one - sa) * dst_image + torch.maximum(src_image * da, dst_image * sa)
    elif mode == "MULTIPLY":
        out_alpha = sa + da - sa * da
        out_image = (one - da) * src_image + (one - sa) * dst_image + src_image * dst_image
    elif mode == "OVERLAY":
        out_alpha = sa + da - sa * da
        overlap = torch.where(2 * dst_image < da, 2 * src_image * dst_image,
                              sa * da - 2 * (sa - src_image) * (da - dst_image))
        out_image = (one - da) * src_image + (one - sa) * dst_image + overlap
    elif mode == "SCREEN":
        out_alpha = sa + da - sa * da
        out_image = src_image + dst_image - src_image * dst_image
    elif mode == "SRC":
        out_alpha, out_image = sa, src_image
    elif mode == "SRC_ATOP":
        out_alpha = da
        out_image = da * src_image + (one - sa) * dst_image
    elif mode == "SRC_IN":
        out_alpha = sa * da
        out_image = src_image * da
    elif mode == "SRC_OUT":
        out_alpha = (one - da) * sa
        out_image = (one - da) * src_image
    elif mode == "SRC_OVER":
        out_alpha = sa + (one - sa) * da
        out_image = src_image + (one - sa) * dst_image
    else:  # XOR
        out_alpha = (one - da) * sa + (one - sa) * da
        out_image = (one - da) * src_image + (one - sa) * dst_image
    out_image = torch.where(out_alpha > ALPHA_EPS, out_image / out_alpha.clamp_min(ALPHA_EPS),
                            torch.zeros_like(out_image))
    return out_image.clamp(0.0, 1.0), alpha_to_mask(out_alpha)


def porter_duff_image_composite(source: Any, source_alpha: Any, destination: Any,
                                destination_alpha: Any, mode: str = "DST") -> tuple[Any, Any]:
    """``PorterDuffImageComposite`` ✓：**批量**版 ✓ —— 尺寸/掩罩不一致先 ``bicubic+center`` 对齐 ✓。

    ⚠️ batch 取**四者最小** ✓（不是补齐 ✗，与参考一致 ✓）；通道数必须相同 ✓（不同 ⇒ 报错 ✗）。
    """
    _need_torch()
    batch = min(len(source), len(source_alpha), len(destination), len(destination_alpha))
    out_images, out_alphas = [], []
    for i in range(batch):
        src_image, dst_image = source[i], destination[i]
        if src_image.shape[2] != dst_image.shape[2]:
            raise ImageOpsError("Porter-Duff 合成要求源/目标**通道数相同** ✓"
                                f"（{src_image.shape[2]} vs {dst_image.shape[2]} ✗）")
        src_alpha = source_alpha[i].unsqueeze(2)
        dst_alpha = destination_alpha[i].unsqueeze(2)
        if tuple(dst_alpha.shape[:2]) != tuple(dst_image.shape[:2]):
            up = dst_alpha.unsqueeze(0).permute(0, 3, 1, 2)
            dst_alpha = bchw_to_image(common_upscale(up, int(dst_image.shape[1]),
                                                     int(dst_image.shape[0]), "bicubic", "center"))
        if tuple(src_image.shape) != tuple(dst_image.shape):
            up = src_image.unsqueeze(0).permute(0, 3, 1, 2)
            src_image = bchw_to_image(common_upscale(up, int(dst_image.shape[1]),
                                                     int(dst_image.shape[0]), "bicubic", "center"))
        if tuple(src_alpha.shape) != tuple(dst_alpha.shape):
            up = src_alpha.unsqueeze(0).permute(0, 3, 1, 2)
            src_alpha = bchw_to_image(common_upscale(up, int(dst_alpha.shape[1]),
                                                     int(dst_alpha.shape[0]), "bicubic", "center"))
        out_image, out_alpha = porter_duff_composite(src_image, src_alpha, dst_image, dst_alpha, mode)
        out_images.append(out_image)
        out_alphas.append(out_alpha.squeeze(2))
    return torch.stack(out_images), torch.stack(out_alphas)


def split_image_with_alpha(image: Any) -> tuple[Any, Any]:
    """``SplitImageWithAlpha`` ✓：拆成 **RGB** ✓ + **掩罩** ✓（没 alpha 通道 ⇒ 掩罩全 0 = 不透明 ✓）。"""
    _need_torch()
    rgb = torch.stack([i[:, :, :3] for i in image])
    m = torch.stack([i[:, :, 3] if i.shape[2] > 3 else torch.ones_like(i[:, :, 0]) for i in image])
    return rgb, alpha_to_mask(m)


def join_image_with_alpha(image: Any, alpha: Any) -> tuple[Any]:
    """``JoinImageWithAlpha`` ✓：把掩罩并成第 4 通道 ✓（掩罩先缩到图尺寸 ✓、batch 取**最大** ✓）。"""
    _need_torch()
    batch = max(len(image), len(alpha))
    a = alpha_to_mask(resize_mask(alpha.to(image), image.shape[1:3]))
    a = repeat_to_batch_size(a, batch)
    image = repeat_to_batch_size(image, batch)
    return torch.cat((image[..., :3], a.unsqueeze(-1)), dim=-1)


# ────────────────────────────── 掩罩 ──────────────────────────────


def mask_to_image(mask: Any) -> Any:
    """``MaskToImage`` ✓：掩罩 ``(B,H,W)`` → 图 ``(B,H,W,3)`` ✓（三通道同值 ✓ 只读视图 ✓）。"""
    _need_torch()
    m = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1)
    return m.expand(-1, -1, -1, 3)


def image_to_mask(image: Any, channel: str = "red") -> Any:
    """``ImageToMask`` ✓：取某通道当掩罩 ✓（``red``/``green``/``blue``/``alpha`` ✓）。"""
    _need_torch()
    names = ["red", "green", "blue", "alpha"]
    if channel not in names:
        raise ImageOpsError(f"channel 只能是 {names} ✓（收到 {channel!r} ✗）")
    idx = names.index(channel)
    if idx >= image.shape[-1]:
        raise ImageOpsError(f"图只有 {image.shape[-1]} 个通道 ⇒ 取不到 {channel!r} ✗")
    return image[:, :, :, idx]


def image_color_to_mask(image: Any, color: int = 0) -> Any:
    """``ImageColorToMask`` ✓：把**指定 RGB 整数**的像素抠成掩罩 1 ✓（**精确**比对 ✓ 不做容差 ✗）。

    口径：``rgb = (R << 16) + (G << 8) + B`` ✓（先 ``clamp(0,1)·255`` 再 ``round`` ✓ 与参考一致 ✓）。
    """
    _need_torch()
    temp = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    key = (temp[:, :, :, 0] << 16) + (temp[:, :, :, 1] << 8) + temp[:, :, :, 2]
    return torch.where(key == int(color), torch.ones_like(key, dtype=image.dtype),
                       torch.zeros_like(key, dtype=image.dtype))


def solid_mask(value: float = 1.0, width: int = 512, height: int = 512) -> Any:
    """``SolidMask`` ✓：纯色掩罩 ✓（``(1, H, W)`` ✓）。"""
    _need_torch()
    return torch.full((1, int(height), int(width)), float(value), dtype=torch.float32)


def invert_mask(mask: Any) -> Any:
    """``InvertMask`` ✓：``1 − mask`` ✓。"""
    _need_torch()
    return 1.0 - mask


def crop_mask(mask: Any, x: int = 0, y: int = 0, width: int = 512, height: int = 512) -> Any:
    """``CropMask`` ✓：从 ``(x, y)`` 裁 ``width × height`` ✓（先压成 3 维 ✓，同 :func:`image_crop` 一样**不补边** ✗）。"""
    _need_torch()
    m = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    return m[:, int(y):int(y) + int(height), int(x):int(x) + int(width)]


def mask_composite(destination: Any, source: Any, x: int = 0, y: int = 0,
                   operation: str = "multiply") -> Any:
    """``MaskComposite`` ✓：把 ``source`` 贴到 ``(x, y)`` 做**逐像素**运算 ✓（结果夹到 ``[0,1]`` ✓）。

    * ``multiply``/``add``/``subtract`` ⇒ 直接代数 ✓（**允许越界中间值** ✓ 最后统一夹 ✓）；
    * ``and``/``or``/``xor`` ⇒ 先 ``round()`` 成布尔 ✓ 再位运算 ✓（**二值掩罩**口径 ✓）；
    * ⚠️ ``x``/``y`` 为负 ⇒ **报错** ✗（参考 schema 也是 ``min=0`` ✓；负坐标的语义要人来定 ✓ 不猜 ✗）。
    """
    _need_torch()
    if operation not in MASK_COMPOSITE_OPS:
        raise ImageOpsError(f"不支持的掩罩运算：{operation!r}（可选：{MASK_COMPOSITE_OPS}）")
    if x < 0 or y < 0:
        raise ImageOpsError("MaskComposite 的 x/y 不支持负值 ✗（要负坐标请先用 pad 扩画布 ✓）")
    out = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
    src = source.reshape((-1, source.shape[-2], source.shape[-1])).to(out.device)
    left, top = int(x), int(y)
    right = min(left + int(src.shape[-1]), int(destination.shape[-1]))
    bottom = min(top + int(src.shape[-2]), int(destination.shape[-2]))
    if right <= left or bottom <= top:
        return out
    src_part = src[:, : bottom - top, : right - left]
    dst_part = out[:, top:bottom, left:right]
    if operation == "multiply":
        out[:, top:bottom, left:right] = dst_part * src_part
    elif operation == "add":
        out[:, top:bottom, left:right] = dst_part + src_part
    elif operation == "subtract":
        out[:, top:bottom, left:right] = dst_part - src_part
    else:
        a = dst_part.round().bool()
        b = src_part.round().bool()
        combo = {"and": torch.bitwise_and, "or": torch.bitwise_or, "xor": torch.bitwise_xor}[operation]
        out[:, top:bottom, left:right] = combo(a, b).to(out.dtype)
    return out.clamp(0.0, 1.0)


def feather_mask(mask: Any, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> Any:
    """``FeatherMask`` ✓：把掩罩四边做成**线性羽化** ✓（第 ``k`` 行/列乘 ``(k+1)/n`` ✓ ⇒ 边缘**最外**那行最暗 ✓）。

    * 羽化宽度先夹到图内 ✓（``n = min(n, 宽/高)`` ✓ 与参考一致 ✓）；
    * 四边**依次相乘** ✓ ⇒ 角上取**两次衰减的乘积** ✓（不是 min ✓）。
    """
    _need_torch()
    out = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).clone()
    w, h = int(out.shape[-1]), int(out.shape[-2])
    ln, rn = min(int(left), w), min(int(right), w)
    tn, bn = min(int(top), h), min(int(bottom), h)
    if ln > 0:
        rate = torch.arange(1, ln + 1, device=out.device, dtype=out.dtype) / ln
        out[:, :, :ln] = out[:, :, :ln] * rate
    if rn > 0:
        rate = torch.arange(1, rn + 1, device=out.device, dtype=out.dtype) / rn
        out[:, :, w - rn:] = out[:, :, w - rn:] * rate
    if tn > 0:
        rate = (torch.arange(1, tn + 1, device=out.device, dtype=out.dtype) / tn).unsqueeze(1)
        out[:, :tn, :] = out[:, :tn, :] * rate
    if bn > 0:
        rate = (torch.arange(1, bn + 1, device=out.device, dtype=out.dtype) / bn).unsqueeze(1)
        out[:, h - bn:, :] = out[:, h - bn:, :] * rate
    return out


def grow_mask(mask: Any, expand: int = 0, tapered_corners: bool = True) -> Any:
    """``GrowMask`` ✓：掩罩**逐 px** 膨胀（``expand > 0`` ✓）/ 腐蚀（``< 0`` ✓）。

    * ``tapered_corners=True`` ⇒ 结构元是**十字** ✓（角不算 ⇒ 形状圆润 ✓）；``False`` ⇒ **3×3 满格** ✓；
    * 边界按 **replicate** 延拓 ✓（本仓口径 ✓ 与参考的 scipy 边界处理不逐像素对齐 ✗）；
    * ``|expand|`` 次迭代 **= 长/缩 ``|expand|`` 像素** ✓（每次 3×3 结构元 ✓）。
    """
    _need_torch()
    out = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).float()
    steps = abs(int(expand))
    if steps == 0:
        return out
    op = "dilate" if expand > 0 else "erode"
    for _ in range(steps):
        out = _extreme_filter(out.unsqueeze(1), 3, op, tapered_corners).squeeze(1)
    return out


def threshold_mask(mask: Any, value: float = 0.5) -> Any:
    """``ThresholdMask`` ✓：``> value ⇒ 1`` ✓ 否则 0 ✓（**严格大于** ✓ 等于不算 ✓）。"""
    _need_torch()
    return (mask > float(value)).to(mask.dtype)


# ────────────────────────────── 滤波 / 形态学 ──────────────────────────────


def _extreme_filter(x: Any, size: int, op: str = "dilate", tapered: bool = False) -> Any:
    """**灰值**核内极值 ✓（``dilate`` = 取最大 ✓、``erode`` = 取最小 ✓）—— ``x: (N,1,H,W)`` ✓。

    * 方形结构元走 ``max_pool2d`` ✓（快 ✓）；十字结构元走 ``unfold`` + 掩码 ✓（角不参与 ✓）；
    * 边界一律 **replicate** 延拓 ✓；**偶数**核按 ``左 = k//2、右 = k−1−k//2`` 补 ✓ ⇒ 输出尺寸**必等于**输入 ✓。
    """
    _need_torch()
    k = max(1, int(size))
    pl, pr = k // 2, k - 1 - k // 2
    pad = (pl, pr, pl, pr)
    negative = op == "erode"
    if not tapered:
        xp = F.pad(x, pad, mode="replicate")
        pooled = F.max_pool2d(-xp if negative else xp, kernel_size=k, stride=1)
        return -pooled if negative else pooled
    xp = F.pad(x, pad, mode="replicate")
    patches = xp.unfold(2, k, 1).unfold(3, k, 1)  # (N,1,H,W,k,k) ✓
    fp = torch.zeros((k, k), dtype=torch.bool, device=x.device)
    fp[k // 2, :] = True
    fp[:, k // 2] = True
    fill = torch.finfo(patches.dtype).max if negative else torch.finfo(patches.dtype).min
    sel = patches.masked_fill(~fp, fill)
    return sel.amin(dim=(-1, -2)) if negative else sel.amax(dim=(-1, -2))


def morphology(image: Any, operation: str = "erode", kernel_size: int = 3) -> Any:
    """``Morphology`` ✓：**逐通道**形态学 ✓（``(B,H,W,C)`` ⇒ 内部折成 ``(B·C,1,H,W)`` ✓ 通道互不串 ✓）。

    ``open = 先腐后胀`` ✓、``close = 先胀后腐`` ✓、``gradient = 胀 − 腐`` ✓、
    ``top_hat = 原 − open`` ✓、``bottom_hat = close − 原`` ✓ —— 均按**灰值**定义 ✓（不是二值 ✓）。
    """
    _need_torch()
    if operation not in MORPHOLOGY_OPS:
        raise ImageOpsError(f"不支持的形态学算子：{operation!r}（可选：{MORPHOLOGY_OPS}）")
    b, h, w, c = image.shape
    x = image.movedim(-1, 1).reshape(b * c, 1, h, w).float()

    def dilate(t: Any) -> Any:
        return _extreme_filter(t, kernel_size, "dilate")

    def erode(t: Any) -> Any:
        return _extreme_filter(t, kernel_size, "erode")

    if operation == "erode":
        out = erode(x)
    elif operation == "dilate":
        out = dilate(x)
    elif operation == "open":
        out = dilate(erode(x))
    elif operation == "close":
        out = erode(dilate(x))
    elif operation == "gradient":
        out = dilate(x) - erode(x)
    elif operation == "top_hat":
        out = x - dilate(erode(x))
    else:
        out = erode(dilate(x)) - x
    return out.reshape(b, c, h, w).movedim(1, -1).to(image.dtype)


def gaussian_kernel(kernel_size: int, sigma: float, dtype: Any = None, device: Any = None) -> Any:
    """各向同性高斯核 ✓（**口径对齐参考** ✓）：坐标轴固定取 ``linspace(-1, 1, k)`` ✓、
    ``r = √(x²+y²)`` ✓、``g = exp(−r²/(2σ²))`` ✓、**归一化**（和 = 1 ✓）。

    ⚠️ 坐标轴**与 k 无关**地钉在 ``[-1, 1]`` ✓ —— 于是 k 变大时核的**实际展宽不变** ✓，
    这是参考的口径 ✓（照搬语义 ✓），不是「标准 σ 像素单位」✗。
    """
    _need_torch()
    k = max(1, int(kernel_size))
    axis = torch.linspace(-1.0, 1.0, k, device=device)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    d = torch.sqrt(xx * xx + yy * yy)
    g = torch.exp(-(d * d) / (2.0 * float(sigma) ** 2))
    g = g / g.sum()
    if dtype is not None:
        g = g.to(dtype)
    return g


def gaussian_blur(image: Any, blur_radius: int = 1, sigma: float = 1.0) -> Any:
    """``ImageBlur`` ✓：**逐通道**高斯模糊 ✓（深度可分离 ✓ 核 = ``2·radius+1`` ✓）。

    先按 ``reflect`` 补 ``radius`` ✓、再 ``conv2d(padding=k//2)`` 取中间有效区 ✓ —— 补边是为了
    让 ``conv2d`` 自带的那圈**零填充**只落在被裁掉的区域 ✓（与参考同法 ✓）。
    """
    _need_torch()
    radius = int(blur_radius)
    if radius == 0:
        return image
    b, h, w, c = image.shape
    k = radius * 2 + 1
    kern = gaussian_kernel(k, sigma, dtype=image.dtype, device=image.device)
    weight = kern.repeat(c, 1, 1).unsqueeze(1)
    x = image.permute(0, 3, 1, 2)
    padded = F.pad(x, (radius, radius, radius, radius), "reflect")
    blurred = F.conv2d(padded, weight, padding=k // 2, groups=c)
    blurred = blurred[:, :, radius:-radius, radius:-radius]
    return blurred.permute(0, 2, 3, 1)


def sharpen(image: Any, sharpen_radius: int = 1, sigma: float = 1.0, alpha: float = 1.0) -> Any:
    """``ImageSharpen`` ✓：**非锐化掩模** ✓ —— 核 = ``中心调整过的高斯`` ✓（``中心 += 1 − Σ核`` ✓）。

    先建 ``kern = gaussian·(−10α)`` ✓ ⇒ 再 ``kern[中心] = kern[中心] − Σkern + 1`` ✓ ⇒ 和恒为 1 ✓
    （常量图**不变** ✓）；结果夹到 ``[0, 1]`` ✓。
    """
    _need_torch()
    radius = int(sharpen_radius)
    if radius == 0:
        return image
    b, h, w, c = image.shape
    k = radius * 2 + 1
    kern = gaussian_kernel(k, sigma, dtype=image.dtype, device=image.device) * -(float(alpha) * 10.0)
    center = k // 2
    kern[center, center] = kern[center, center] - kern.sum() + 1.0
    weight = kern.repeat(c, 1, 1).unsqueeze(1)
    x = image.permute(0, 3, 1, 2)
    padded = F.pad(x, (radius, radius, radius, radius), "reflect")
    out = F.conv2d(padded, weight, padding=center, groups=c)
    out = out[:, :, radius:-radius, radius:-radius]
    return out.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def quantize(image: Any, colors: int = 256, dither: str = "none") -> Any:
    """``ImageQuantize`` ✓：**调色板量化** ✓（走 Pillow 的中位切分 ✓；本仓不自研调色板 ✗）。

    * ``colors`` 夹到 ``[1, 256]`` ✓，且是**真的**色数上限 ✓（走**中位切分** ✓；
      ⚠️ **不借「灰阶调色板 + ``quantize(palette=...)``」那条写法** ✗ —— 那条 Pillow 分支会
      **忽略 ``colors``** ✗ 且把彩色**压成灰阶** ✓✗ ⇒ ``colors=4`` 却出来几十种色 + 掉色 ✓✗）；
    * ``dither='none'`` ⇒ ``colors`` **减 1** ✓（对齐参考口径 ✓）；``floyd-steinberg`` ✓；
      ``bayer-2/4/8/16`` ⇒ 按 **有序抖动矩阵** 加偏移后再量化 ✓（矩阵按 ``(m−1.5, m+0.5; m+1.5, m−0.5)/4ⁿ``
      递归构造 ✓ 与参考同式 ✓，用 torch 拼 ✓ 不引 numpy 之外的东西 ✓）；
    * 第 4 通道（alpha）⇒ **原样拼回** ✓ 不参与量化 ✓。
    """
    _need_torch()
    if not (_HAS_PIL and _has_numpy()):
        raise ImageOpsError("quantize 需要 Pillow + numpy ✓（本机缺 ⇒ 这个算子跑不了 ✗）")
    import numpy as np

    if dither == "none":
        colors = max(1, min(256, int(colors) - 1))
    elif dither != "floyd-steinberg" and not dither.startswith("bayer-"):
        raise ImageOpsError(f"不支持的抖动方式：{dither!r}（none / floyd-steinberg / bayer-2/4/8/16 ✗）")
    colors = max(1, min(256, int(colors)))
    b, h, w, c = image.shape
    out = []
    for i in range(b):
        rgb = image[i, :, :, :3]
        if dither.startswith("bayer"):
            order = int(dither.split("-")[1])
            if order <= 0:
                raise ImageOpsError(f"bayer 阶数必须 ≥ 1 ✓（收到 {dither!r} ✗）")
            matrix = _bayer_matrix(order, rgb.device, rgb.dtype)
            spread = 2.0 * 256.0 / colors
            m = spread * matrix + 0.5
            m = torch.where(m < 0, torch.zeros_like(m), m)
            tw = math.ceil(h / m.shape[0])
            th = math.ceil(w / m.shape[1])
            tiled = m.repeat(tw, th)[:h, :w].unsqueeze(-1)
            rgb = (rgb * 255.0 + tiled).clamp(0.0, 255.0)
        else:
            rgb = (rgb * 255.0).clamp(0.0, 255.0)
        arr = rgb.detach().cpu().numpy().astype(np.uint8)
        pil = _PILImage.fromarray(arr)  # (H,W,3) uint8 ⇒ 自动按 RGB 解析 ✓
        # ⚠️ 走**中位切分** ✓ ⇒ colors 是**真的**上限 ✓。
        # 不能写成「灰阶调色板 + quantize(palette=...)」✗ —— Pillow 那条分支**忽略 colors** ✗
        # 且把彩色**压成灰阶** ✓✗（``colors=4`` 出来几十种色 + 掉色 ⇒ 看着对的错值 ✓✗）。
        pil = pil.quantize(
            colors=colors,
            method=_PILImage.Quantize.MEDIANCUT,
            dither=(_PILImage.Dither.NONE if dither != "floyd-steinberg"
                    else _PILImage.Dither.FLOYDSTEINBERG),
        )
        arr2 = np.array(pil.convert("RGB")).astype("float32") / 255.0
        quant = torch.from_numpy(arr2).to(image.device, image.dtype)
        if c == 4:
            quant = torch.cat([quant, image[i, :, :, 3:]], dim=-1)
        out.append(quant)
    return torch.stack(out)


def _has_numpy() -> bool:
    """有没有 numpy ✓（量化走 Pillow ⇒ 需要它把张量接送出去 ✓）。"""
    try:  # pragma: no cover - 环境相关 ✓
        import numpy  # noqa: F401

        return True
    except ImportError:  # pragma: no cover
        return False


def _bayer_matrix(order: int, device: Any, dtype: Any) -> Any:
    """递归构造 **Bayer** 有序抖动矩阵 ✓（``order=1`` ⇒ ``[[0,2],[3,1]]/4`` ✓ 与参考同式 ✓）。"""
    _need_torch()
    m = torch.zeros((1, 1), device=device, dtype=dtype)
    for _ in range(int(order)):
        top = torch.cat([m - 1.5, m + 0.5], dim=1)
        bottom = torch.cat([m + 1.5, m - 0.5], dim=1)
        m = torch.cat([top, bottom], dim=0)
    return m / (4.0 ** int(order))


# ────────────────────────────── 色彩 ──────────────────────────────


def rgb_to_grayscale(image: Any) -> Any:
    """RGB → **亮度** ✓（``(B,H,W,3)`` → ``(B,H,W)`` ✓）—— BT.601 亮度系数 ✓（与 Y 通道同口径 ✓）。"""
    _need_torch()
    x = image[..., :3]
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def rgb_to_ycbcr(image: Any) -> Any:
    """RGB → **YCbCr** ✓（``(...,3)`` ✓，值域 ``[0,1]`` ✓，通道序 ``Y, Cb, Cr`` ✓）。

    系数是 **BT.601 全范围** ✓：``Cb = −0.168736R − 0.331264G + 0.5B + 0.5`` ✓、
    ``Cr = 0.5R − 0.418688G − 0.081312B + 0.5`` ✓（与参考节点用的那套一致 ✓）。
    """
    _need_torch()
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack((y, cb, cr), dim=-1)


def ycbcr_to_rgb(image: Any) -> Any:
    """**YCbCr** → RGB ✓（``rgb_to_ycbcr`` 的逆 ✓，输出夹到 ``[0,1]`` ✓）。"""
    _need_torch()
    y, cb, cr = image[..., 0], image[..., 1], image[..., 2]
    cb0, cr0 = cb - 0.5, cr - 0.5
    r = y + 1.402 * cr0
    g = y - 0.344136 * cb0 - 0.714136 * cr0
    b = y + 1.772 * cb0
    return torch.stack((r, g, b), dim=-1).clamp(0.0, 1.0)


def image_rgb_to_yuv(image: Any) -> tuple[Any, Any, Any]:
    """``RGBToYUV`` ✓：一张图 ⇒ **Y/U/V 三张掩罩** ✓（``(B,H,W)`` × 3 ✓，顺序 = ``Y, U=Cb, V=Cr`` ✓）。"""
    _need_torch()
    out = rgb_to_ycbcr(image[..., :3])
    return out[..., 0], out[..., 1], out[..., 2]


def image_yuv_to_rgb(y: Any, u: Any, v: Any) -> Any:
    """``YUVToRGB`` ✓：三张掩罩 ⇒ 一张图 ✓（尺寸/batch 不齐时按 **第一个** 对齐 ✓）。"""
    _need_torch()
    b, h, w = y.shape[0], y.shape[1], y.shape[2]
    parts = []
    for m in (u, v):
        m = m.reshape((-1, m.shape[-2], m.shape[-1]))
        if tuple(m.shape) != (b, h, w):
            m = F.interpolate(m.unsqueeze(1).to(y.device), size=(h, w), mode="bicubic").squeeze(1)
            m = repeat_to_batch_size(m, b)
        parts.append(m)
    stacked = torch.stack((y, parts[0], parts[1]), dim=-1)
    return ycbcr_to_rgb(stacked)


# ────────────────────────────── 边缘（Canny 预处理） ──────────────────────────────

#: Canny 的默认平滑参数 ✓（对齐参考节点的默认值 ✓）。
CANNY_KERNEL_SIZE = 5
CANNY_SIGMA = 1.0
#: ⚠️ **归一化的噪声地板** ✓：整体峰值不超它 ⇒ 这张图「**没有梯度**」✓ ⇒ 幅值直接全 0 ✓。
#: 少了这道闸 ⇒ 平坦图的浮点噪声会被「按本图峰值归一化」**放大成整幅伪边缘** ✓✗。
CANNY_NOISE_FLOOR = 1e-6


def _conv_same(x: Any, weight: Any, groups: int = 1) -> Any:
    """**同尺寸**卷积 ✓（边界按 replicate 补 ✓）—— ``weight`` 的最后一维必须是奇数 ✓。"""
    _need_torch()
    p = int(weight.shape[-1]) // 2
    return F.conv2d(F.pad(x, (p, p, p, p), mode="replicate"), weight.to(x.dtype), groups=groups)


def _sobel_kernels(device: Any) -> tuple[Any, Any]:
    """**Sobel 3×3** ✓（``kx`` 横向一阶导 ✓ ``ky = kxᵀ`` 纵向 ✓）—— 归一化系数不影响结果 ✓
    （幅值随后**整体归一化** ✓ ⇒ 系数被约掉 ✓）。"""
    _need_torch()
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      dtype=torch.float32, device=device).view(1, 1, 3, 3) / 4.0
    return kx, kx.transpose(-1, -2).contiguous()


def _non_max_suppression(mag: Any, gx: Any, gy: Any) -> Any:
    """**非极大值抑制** ✓：把梯度方向**量化到 4 个方向** ✓（``0°/45°/90°/135°`` ✓），
    只保留沿该方向的**局部极大** ✓（``≥`` 两侧 ⇒ 平台**全留** ✓ 本仓口径 ✓）。"""
    _need_torch()
    p = F.pad(mag, (1, 1, 1, 1), mode="replicate")
    m = p[:, :, 1:-1, 1:-1]
    left, right = p[:, :, 1:-1, :-2], p[:, :, 1:-1, 2:]
    up, down = p[:, :, :-2, 1:-1], p[:, :, 2:, 1:-1]
    ul, ur = p[:, :, :-2, :-2], p[:, :, :-2, 2:]
    dl, dr = p[:, :, 2:, :-2], p[:, :, 2:, 2:]
    angle = torch.rad2deg(torch.atan2(gy, gx)) % 180.0
    idx = torch.round(angle / 45.0).long() % 4
    n1 = torch.zeros_like(m)
    n2 = torch.zeros_like(m)
    for d, (a, b) in ((0, (left, right)), (1, (ur, dl)), (2, (up, down)), (3, (ul, dr))):
        sel = idx == d
        n1 = torch.where(sel, a, n1)
        n2 = torch.where(sel, b, n2)
    return torch.where((m >= n1) & (m >= n2), m, torch.zeros_like(m))


def _hysteresis(strong: Any, weak: Any) -> Any:
    """**双阈值滞后** ✓：强边全留 ✓、弱边**只在 8 邻域连着强边时**留 ✓（形态学**膨胀传播** ✓ 直到不变 ✓）。

    口径：连通按 **8 邻域** ✓（图**外**不算连接 ✓），迭代上限 = ``max(H, W)`` ✓（理论最坏路径长 ✓）。
    """
    _need_torch()
    out = strong.clone()
    limit = int(max(strong.shape[-1], strong.shape[-2]))
    for _ in range(max(1, limit)):
        grown = (F.max_pool2d(out.float(), kernel_size=3, stride=1, padding=1) > 0.5) & weak
        new = out | grown
        if torch.equal(new, out):
            break
        out = new
    return out


def canny(image: Any, low_threshold: float = 0.4, high_threshold: float = 0.8,
          kernel_size: int = CANNY_KERNEL_SIZE, sigma: float = CANNY_SIGMA) -> tuple[Any, Any]:
    """**Canny 边缘** ✓（预处理口径 ✓）—— 返回 ``(幅值, 边缘)`` ✓，均为 ``(B, H, W)`` ✓。

    四步 ✓：**亮度** → **高斯平滑** ✓（replicate 边 ✓）→ **Sobel** ✓ → 幅值 ``√(gx²+gy²)`` ✓
    ⇒ **整体归一化到 ``[0,1]``** ✓（⚠️ 峰值不到 :data:`CANNY_NOISE_FLOOR` ⇒ 视为**没有梯度** ✓
    整幅全 0 ✓ —— 不然平坦图的浮点噪声会被归一化**放大成整幅伪边缘** ✓✗）⇒ **非极大值抑制** ✓
    ⇒ **双阈值滞后** ✓。

    ⚠️ **本仓口径**（**不是** kornia 的逐像素复刻 ✗）：阈值吃的是**归一化幅值** ✓
    （``low ≤ 幅值 < high`` = 弱边 ✓、``≥ high`` = 强边 ✓）⇒ 与参考节点「阈值作用在相对幅值上」的
    行为**同向** ✓，但边缘的**精确像素**会与 kornia 版有差 ✓（自研 ✓ 无 kornia 依赖 ✓）。
    """
    _need_torch()
    lo, hi = float(low_threshold), float(high_threshold)
    if not (0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0):
        raise ImageOpsError(f"Canny 阈值必须在 [0, 1] ✓（收到 {lo} / {hi} ✗）")
    if lo > hi:
        raise ImageOpsError(f"Canny 要求 low ≤ high ✓（收到 {lo} > {hi} ✗）")
    k = max(3, int(kernel_size) | 1)  # 取奇数 ✓（核必须奇数 ✓）
    gray = rgb_to_grayscale(image).unsqueeze(1).float()
    kern = gaussian_kernel(k, sigma, dtype=torch.float32, device=gray.device).view(1, 1, k, k)
    blurred = _conv_same(gray, kern)
    kx, ky = _sobel_kernels(gray.device)
    gx = _conv_same(blurred, kx)
    gy = _conv_same(blurred, ky)
    mag = torch.sqrt(gx * gx + gy * gy)
    peak = mag.amax(dim=(-1, -2), keepdim=True)
    # ⚠️ 峰值连**噪声地板**都不到 ⇒ 这张图就是「**没有梯度**」✓ ⇒ 全 0 ✓。
    # 只判 ``peak > 0`` 会把平坦图的浮点噪声**归一化放大**成整幅伪边缘 ✓✗（满屏边缘 ✓✗）。
    norm = torch.where(peak > CANNY_NOISE_FLOOR, mag / peak.clamp_min(CANNY_NOISE_FLOOR),
                       torch.zeros_like(mag))
    nms = _non_max_suppression(norm, gx, gy)
    edges = _hysteresis(nms >= hi, nms >= lo)
    return norm.squeeze(1), edges.squeeze(1).to(image.dtype)


def canny_edge_image(image: Any, low_threshold: float = 0.4, high_threshold: float = 0.8) -> Any:
    """Canny **边缘图** ✓（``(B,H,W,3)`` ✓ 三通道同值 ✓）—— 直接接参考节点 ``Canny`` 的输出口径 ✓。"""
    _need_torch()
    _, edges = canny(image, low_threshold, high_threshold)
    return edges.unsqueeze(-1).expand(-1, -1, -1, 3)
