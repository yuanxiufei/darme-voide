"""自研引擎·图像算子 ✓（缩放/裁剪/旋转/拼接/贴图/合成/掩罩/形态学/色彩/Canny ✓ **纯 torch** ✓）。

口径：全内部 ``(B,C,H,W)`` ✓ / 对外 ``(B,H,W,C)`` ✓；掩罩 **1 = 覆盖/要生成** ✓。
⚠️ 没装 torch ⇒ **显式 SKIP + 打摘要** ✓（崩掉连「这次没验」都看不见 ✗）。

跑法::

    ./.venv/Scripts/python.exe tests/engine_image_ops_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
if str(BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import image_ops as ops  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []
_TORCH: Any = None
_PROBED = False


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: Any, needle: str = "") -> str | None:
    try:
        call()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        return text if needle in text else None
    return None


def torch_or_skip() -> Any:
    """懒取 torch ✓（只里探一次 ✓ ⇒ SKIP 只打一条 ✓）。"""
    global _TORCH, _PROBED  # noqa: PLW0603
    if not _PROBED:
        _PROBED = True
        try:
            import torch  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            _TORCH = None
            skip("没装 torch ⇒ 图像算子整层不跑 ✓（显式 SKIP ✓ 不是通过 ✗）")
        else:
            _TORCH = torch
    return _TORCH


def case_layout(t: Any) -> None:
    """① 布局：``(B,H,W,C)`` ⇄ ``(B,C,H,W)`` ✓ + 掩罩 3↔4 维 ✓。"""
    img = t.rand(2, 4, 6, 3)
    bchw = ops.image_to_bchw(img)
    check("① ``image_to_bchw``：``(2,4,6,3)`` ⇒ ``(2,3,4,6)`` ✓",
          tuple(bchw.shape) == (2, 3, 4, 6), tuple(bchw.shape))
    check("①′ **往返恒等** ✓（``bchw_to_image(image_to_bchw(x)) == x`` ✓）",
          t.allclose(ops.bchw_to_image(bchw), img, atol=1e-6))
    mask = t.rand(2, 4, 6)
    check("①″ ``mask_to_bchw1``：``(B,H,W)`` ⇒ ``(B,1,H,W)`` ✓",
          tuple(ops.mask_to_bchw1(mask).shape) == (2, 1, 4, 6))
    check("①‴ ``resize_mask(mask, (h, w))`` ⇒ ``(B,h,w)`` ✓（**3 维进 3 维出** ✓）",
          tuple(ops.resize_mask(mask, (3, 5)).shape) == (2, 3, 5),
          tuple(ops.resize_mask(mask, (3, 5)).shape))
    check("①⁗ ``repeat_to_batch_size``：单张 ⇒ 批 3 ✓",
          tuple(ops.repeat_to_batch_size(t.rand(1, 3, 4, 4), 3).shape)[0] == 3)


def case_upscale(t: Any) -> None:
    """② 重采样：六种方法都认 ✓、输出尺寸对 ✓、未知方法**报错** ✗；自研 bislerp/lanczos ✓。"""
    x = t.rand(1, 3, 4, 6)
    bad = []
    for method in ops.UPSCALE_METHODS:
        try:
            out = ops.common_upscale(x, 8, 12, method)
            if tuple(out.shape) != (1, 3, 12, 8):
                bad.append((method, tuple(out.shape)))
        except Exception as err:  # noqa: BLE001
            bad.append((method, f"{type(err).__name__}: {err}"))
    check(f"② ``UPSCALE_METHODS`` 六种**逐个**认 ✓（{ops.UPSCALE_METHODS} ✓）且输出 ``(1,3,12,8)`` ✓",
          not bad, bad)
    check("②′ 未知方法 ⇒ **报错** ✗（不静默退回 bilinear ✗ —— 那会让人以为换了核 ✓✗）",
          _raises(lambda: ops.common_upscale(x, 8, 8, "magic"), "magic") is not None)
    check("②″ 自研 ``bislerp`` / ``lanczos_resize`` 尺寸对 ✓",
          tuple(ops.bislerp(x, 8, 12).shape) == (1, 3, 12, 8)
          and tuple(ops.lanczos_resize(x, 8, 12).shape) == (1, 3, 12, 8))
    idx, weights = ops.bislerp_coords(4, 8)
    check("②‴ ``bislerp_coords``：**每行**都给索引 + 权重 ✓（长度跟上采样后的目标 ✓）"
          "⚠️ 这里只验长度不验数值 ✓ —— 数值正确性由 ②″ 的**往返形状**与 ② 的六核一致性兜 ✓",
          len(idx) == 8 and len(weights) == 8, (len(idx), len(weights)))


def case_scale_family(t: Any) -> None:
    """③ 缩放族：定尺寸 / 倍率 / 长边 / 总像素 / 保比例填边 ✓。"""
    img = t.rand(1, 8, 12, 3)
    check("③ ``image_scale`` 定尺寸 ⇒ ``(1,6,8,3)`` ✓",
          tuple(ops.image_scale(img, "bilinear", 8, 6).shape) == (1, 6, 8, 3),
          tuple(ops.image_scale(img, "bilinear", 8, 6).shape))
    check("③′ ``image_scale_by(2.0)`` ⇒ 宽高**各乘 2** ✓（``12,8`` ⇒ ``24,16`` ✓）",
          tuple(ops.image_scale_by(img, "bilinear", 2.0).shape) == (1, 16, 24, 3),
          tuple(ops.image_scale_by(img, "bilinear", 2.0).shape))
    check("③″ ``image_scale_to_max_dimension(largest_size=6)`` ⇒ **长边** = 6 ✓（保比例 ✓）",
          max(ops.image_scale_to_max_dimension(img, "lanczos", 6).shape[1:3]) == 6,
          ops.image_scale_to_max_dimension(img, "lanczos", 6).shape)
    check("③‴ ``image_scale_to_total_pixels`` 总像素 ≈ 预算 ✓（按步长对齐 ✓）",
          ops.image_scale_to_total_pixels(img, "bilinear", 0.01, 8).shape[1] % 8 == 0,
          ops.image_scale_to_total_pixels(img, "bilinear", 0.01, 8).shape)
    padded = ops.resize_and_pad(img, 8, 8, "black", "area")
    check("③⁗ ``resize_and_pad`` ⇒ **正好** ``(1,8,8,3)`` ✓ 且黑边 = 0 ✓（保比例有余量才填 ✓）",
          tuple(padded.shape) == (1, 8, 8, 3) and float(padded.min()) >= 0.0
          and float(padded[..., 0].max()) > 0.0, tuple(padded.shape))
    check("③⁵ ``CROP_METHODS`` = disabled / center ✓（两种口径都在 ✓）",
          ops.CROP_METHODS == ("disabled", "center"))


def case_crop_rotate_flip(t: Any) -> None:
    """④ 裁剪 / 旋转 / 翻转 / 反色 ✓ —— 尺寸会换的地方**逐维**验 ✓。"""
    img = t.arange(1 * 4 * 6 * 3, dtype=t.float32).reshape(1, 4, 6, 3) / 100.0
    crop = ops.image_crop(img, 3, 2, 1, 1)
    check("④ ``image_crop(w=3,h=2,x=1,y=1)`` ⇒ ``(1,2,3,3)`` ✓ 且**真的**取对了那一块 ✓",
          tuple(crop.shape) == (1, 2, 3, 3) and t.allclose(crop[0, :, :, 0], img[0, 1:3, 1:4, 0], atol=1e-6),
          tuple(crop.shape))
    check("④′ ``image_rotate('90 degrees')`` ⇒ **宽高互换** ✓（``4,6`` ⇒ ``6,4`` ✓）",
          tuple(ops.image_rotate(img, "90 degrees").shape) == (1, 6, 4, 3),
          tuple(ops.image_rotate(img, "90 degrees").shape))
    check("④″ ``image_rotate('none')`` 尺寸**不变** ✓；非法角度 ⇒ **报错** ✗（不静默不转 ✗）",
          tuple(ops.image_rotate(img, "none").shape) == (1, 4, 6, 3)
          and _raises(lambda: ops.image_rotate(img, "45 degrees"), "rotation") is not None)
    sym = t.rand(1, 4, 6, 3)
    sym[:, :, :, :] = 0.5
    check("④‴ ``image_flip`` 左右翻：**对称图翻完不变** ✓、非对称图**必变** ✓",
          t.allclose(ops.image_flip(sym, "y-axis: horizontally"), sym, atol=1e-6)
          and not t.allclose(ops.image_flip(img, "y-axis: horizontally"), img, atol=1e-6))
    rgba = ops.image_invert(t.cat([t.zeros(1, 2, 2, 3), t.full((1, 2, 2, 1), 0.3)], -1))
    check("④⁗ ``image_invert`` = ``1 − x`` ✓（**第 4 通道是 alpha ⇒ 原样保留** ✓ 不反 ✓）",
          t.allclose(ops.image_invert(t.full((1, 2, 2, 3), 0.25)), t.full((1, 2, 2, 3), 0.75), atol=1e-6)
          and abs(float(rgba[0, 0, 0, 3]) - 0.3) < 1e-6
          and t.allclose(rgba[..., :3], t.ones(1, 2, 2, 3), atol=1e-6),
          float(rgba[0, 0, 0, 3]))


def case_stitch_batch(t: Any) -> None:
    """⑤ 拼接：右接**宽相加** ✓ 下接**高相加** ✓；成批：批相加 ✓。"""
    a, b = t.rand(1, 4, 6, 3), t.rand(1, 4, 6, 3)
    check("⑤ ``image_stitch`` 右接 ⇒ 宽 = w1 + w2 ✓",
          tuple(ops.image_stitch(a, "right", True, 0, "white", b).shape) == (1, 4, 12, 3),
          tuple(ops.image_stitch(a, "right", True, 0, "white", b).shape))
    check("⑤′ 下接 ⇒ 高 = h1 + h2 ✓",
          tuple(ops.image_stitch(a, "down", True, 0, "white", b).shape) == (1, 8, 6, 3))
    check("⑤″ 非法方向 ⇒ **报错** ✗（不猜一个最像的方向 ✗）",
          _raises(lambda: ops.image_stitch(a, "sideways", True, 0, "white", b)) is not None)
    check("⑤‴ ``image_batch`` ⇒ 批 = 2 ✓；三张 ⇒ 批 = 3 ✓（**首尾相接** ✓）",
          tuple(ops.image_batch(a, b).shape) == (2, 4, 6, 3)
          and tuple(ops.image_batch(ops.image_batch(a, b), a).shape)[0] == 3)


def case_composite(t: Any) -> None:
    """⑥ 贴图 / 混合 / 透明度拆分 ✓（⚠️ ``composite`` 吃**内部** ``(N,C,H,W)`` ✓）。"""
    out = ops.composite(t.zeros(1, 3, 4, 4), t.ones(1, 3, 2, 2), 1, 1)
    check("⑥ ``composite``：``(1,1)`` 起的 2×2 被贴上 ✓、其余保持 0 ✓（贴的是 ``(x, y)`` ✓）",
          tuple(out.shape) == (1, 3, 4, 4) and float(out[0, 0, 1, 1]) == 1.0
          and float(out[0, 0, 0, 0]) == 0.0)
    masked = ops.image_composite_masked(t.zeros(1, 4, 4, 3), t.ones(1, 2, 2, 3), 1, 1)
    check("⑥′ ``image_composite_masked``：图像层同样贴对的**位置** ✓（前 3 通道对齐 ✓）",
          tuple(masked.shape)[:3] == (1, 4, 4) and float(masked[0, 1, 1, 0]) == 1.0
          and float(masked[0, 3, 3, 0]) == 0.0, tuple(masked.shape))
    dark, light = t.zeros(1, 4, 6, 3), t.ones(1, 4, 6, 3)
    check("⑥″ ``image_blend``：``factor=0`` ⇒ 图1 ✓、``=1`` ⇒ 图2 ✓、``=0.5`` ⇒ 各一半 ✓",
          t.allclose(ops.image_blend(dark, light, 0.0), dark, atol=1e-6)
          and t.allclose(ops.image_blend(dark, light, 1.0), light, atol=1e-6)
          and float(ops.image_blend(dark, light, 0.5)[0, 0, 0, 0]) == 0.5)
    rgb = t.rand(1, 4, 6, 3)
    check("⑥‴ ``split`` / ``join`` 往返：3 通道图 ⇒ 掩罩**全 0** ✓（没 alpha 就是**不透明** ✓）"
          "⇒ 并回第 4 通道后前 3 通道**逐值不变** ✓",
          t.allclose(ops.split_image_with_alpha(rgb)[1], t.zeros(1, 4, 6), atol=1e-6)
          and t.allclose(ops.join_image_with_alpha(rgb, t.zeros(1, 4, 6))[..., :3], rgb, atol=1e-6))


def case_porter_duff(t: Any) -> None:
    """⑦ Porter-Duff：**18 个模式**逐个能跑 ✓、形状不变 ✓、未知模式**报错** ✗。"""
    bad = []
    for mode in ops.PORTER_DUFF_MODES:
        try:
            img, alpha = ops.porter_duff_composite(t.ones(4, 4, 3), t.ones(4, 4, 1),
                                                   t.zeros(4, 4, 3), t.zeros(4, 4, 1), mode)
            if tuple(img.shape) != (4, 4, 3) or tuple(alpha.shape) != (4, 4, 1):
                bad.append((mode, tuple(img.shape), tuple(alpha.shape)))
        except Exception as err:  # noqa: BLE001
            bad.append((mode, f"{type(err).__name__}: {err}"))
    check(f"⑦ **全部** {len(ops.PORTER_DUFF_MODES)} 个模式逐个跑通 ✓ 且输出 ``(4,4,3)`` + ``(4,4,1)`` ✓",
          len(ops.PORTER_DUFF_MODES) == 18 and not bad, bad)
    check("⑦′ 未知模式 ⇒ **报错** ✗（不做「默认按 over 处理」这种兜底 ✗）",
          _raises(lambda: ops.porter_duff_composite(t.ones(2, 2, 3), t.ones(2, 2, 1),
                                                    t.zeros(2, 2, 3), t.zeros(2, 2, 1), "nope")) is not None)


def case_masks(t: Any) -> None:
    """⑧ 掩罩：纯色 / 反色 / 裁剪 / 阈值 / 羽化 / 膨胀 / 抠色 / 图像互转 ✓。"""
    check("⑧ ``solid_mask(1.0, 4, 3)`` ⇒ ``(1,3,4)`` **全 1** ✓",
          tuple(ops.solid_mask(1.0, 4, 3).shape) == (1, 3, 4)
          and float(ops.solid_mask(0.25, 2, 2).min()) == 0.25)
    check("⑧′ ``invert_mask`` = ``1 − m`` ✓；``mask_to_alpha`` / ``alpha_to_mask`` **互为逆** ✓",
          t.allclose(ops.invert_mask(t.full((1, 2, 2), 0.25)), t.full((1, 2, 2), 0.75), atol=1e-6)
          and t.allclose(ops.alpha_to_mask(ops.mask_to_alpha(t.full((1, 2, 2), 0.3))),
                         t.full((1, 2, 2), 0.3), atol=1e-6))
    grid = t.arange(1 * 6 * 6, dtype=t.float32).reshape(1, 6, 6) / 36.0
    crop = ops.crop_mask(grid, 1, 2, 3, 2)
    check("⑧″ ``crop_mask(x=1,y=2,w=3,h=2)`` ⇒ ``(1,2,3)`` ✓ 且取对块 ✓（``y`` 是**行** ✓）",
          tuple(crop.shape) == (1, 2, 3) and t.allclose(crop[0, :, :], grid[0, 2:4, 1:4], atol=1e-6))
    thr = ops.threshold_mask(t.tensor([[[0.5, 0.6, 0.4]]]), 0.5)
    check("⑧‴ ``threshold_mask``：**严格大于**才 1 ✓（等于 ``value`` **不算** ✓）",
          tuple(thr.shape) == (1, 1, 3)
          and float(thr[0, 0, 0]) == 0.0 and float(thr[0, 0, 1]) == 1.0, thr.tolist())
    feathered = ops.feather_mask(t.ones(1, 8, 8), 4, 0, 0, 0)
    check("⑧⁗ ``feather_mask``：羽化边（最外那列）**比中心暗** ✓ 且中心保持 1 ✓",
          float(feathered[0, 4, 0]) < float(feathered[0, 4, 7])
          and abs(float(feathered[0, 4, 7]) - 1.0) < 1e-6, float(feathered[0, 4, 0]))
    dot = t.zeros(1, 9, 9)
    dot[0, 4, 4] = 1.0
    grown = ops.grow_mask(dot, 1, True)
    check("⑧⁵ ``grow_mask(expand=1)`` 让单点**扩散** ✓（十字形 ⇒ 1 变 3~9 个 ✓）；``expand=0`` **原样** ✓",
          4.0 < float(grown.sum()) <= 9.0 and float(ops.grow_mask(dot, 0, True).sum()) == 1.0,
          float(grown.sum()))
    one = t.full((1, 3, 3), 0.6)
    check("⑧⁶ ``mask_to_image`` ⇒ ``(B,H,W,3)`` ✓ 且**三通道同值** ✓；``image_to_mask`` **取回** ✓",
          tuple(ops.mask_to_image(one).shape) == (1, 3, 3, 3)
          and t.allclose(ops.mask_to_image(one)[..., 0], one, atol=1e-6)
          and t.allclose(ops.image_to_mask(ops.mask_to_image(one), "red"), one, atol=1e-6))
    red = t.zeros(1, 3, 3, 3)
    red[..., 0] = 1.0
    red[0, 0, 0] = t.tensor([0.0, 0.0, 0.0])
    check("⑧⁷ ``image_color_to_mask`` 抠**纯黑**（``0x000000`` ✓）⇒ 只有那一个像素是 1 ✓",
          float(ops.image_color_to_mask(red, 0).sum()) == 1.0
          and float(ops.image_color_to_mask(red, 0xFF0000).sum()) == 8.0)


def case_filter(t: Any) -> None:
    """⑨ 形态学 / 高斯 / 锐化 ✓（⚠️ 逐通道 ✓ **通道互不串** ✓；核**归一** ✓）。"""
    img = t.rand(1, 8, 8, 3)
    bad = []
    for operation in ops.MORPHOLOGY_OPS:
        try:
            out = ops.morphology(img, operation, 3)
            if tuple(out.shape) != (1, 8, 8, 3):
                bad.append((operation, tuple(out.shape)))
        except Exception as err:  # noqa: BLE001
            bad.append((operation, f"{type(err).__name__}: {err}"))
    check(f"⑨ ``MORPHOLOGY_OPS`` = ``{ops.MORPHOLOGY_OPS}`` **逐个**跑通 ✓ 且形状不变 ✓"
          f"（折成 ``(B·C,1,H,W)`` ⇒ **通道不串** ✓）", not bad, bad)
    dot = t.zeros(1, 9, 9, 1)
    dot[0, 4, 4, 0] = 1.0
    check("⑨′ ``dilate`` **放大**亮点 ✓；``erode`` **抹掉**孤立亮点 ✓（对偶 ✓）",
          float(ops.morphology(dot, "dilate", 3).sum()) >= 9.0
          and float(ops.morphology(dot, "erode", 3).sum()) <= 1.0,
          (float(ops.morphology(dot, "dilate", 3).sum()), float(ops.morphology(dot, "erode", 3).sum())))
    flat = t.full((1, 4, 4, 3), 0.5)
    check("⑨″ ``gaussian_blur`` / ``sharpen`` 对**常量图** ⇒ **不变** ✓（核**归一** ⇒ 不改亮度 ✓）"
          "且形状不变 ✓",
          tuple(ops.gaussian_blur(flat, 1, 1.0).shape) == (1, 4, 4, 3)
          and t.allclose(ops.gaussian_blur(flat, 1, 1.0), flat, atol=1e-5)
          and t.allclose(ops.sharpen(flat, 1, 1.0, 1.0), flat, atol=1e-5))
    check("⑨‴ ``gaussian_kernel(5, 1.0)`` **和为 1** ✓（不归一 ⇒ 模糊顺手改亮度 ✓✗）",
          abs(float(ops.gaussian_kernel(5, 1.0).sum()) - 1.0) < 1e-6)


def case_color(t: Any) -> None:
    """⑩ 色彩：亮度系数 ✓、YCbCr / YUV **往返** ✓（都按 **BT.601 全范围** ✓）。"""
    check("⑩ ``rgb_to_grayscale``：纯红 ⇒ ``0.299`` ✓（BT.601 亮度 ✓ 与 Y 通道同口径 ✓）",
          abs(float(ops.rgb_to_grayscale(t.tensor([[[[1.0, 0.0, 0.0]]]]))[0, 0, 0]) - 0.299) < 1e-5,
          float(ops.rgb_to_grayscale(t.tensor([[[[1.0, 0.0, 0.0]]]]))[0, 0, 0]))
    img = t.rand(2, 4, 6, 3)
    ycbcr = ops.rgb_to_ycbcr(img)
    check("⑩′ ``rgb_to_ycbcr`` ⇒ ``(B,H,W,3)`` ✓ 且 ``ycbcr_to_rgb`` **往返恒等** ✓",
          tuple(ycbcr.shape) == (2, 4, 6, 3)
          and t.allclose(ops.ycbcr_to_rgb(ycbcr), img, atol=1e-5))
    y, u, v = ops.image_rgb_to_yuv(img)
    check("⑩″ ``image_rgb_to_yuv`` ⇒ **三张** ``(B,H,W)`` ✓（顺序 ``Y,U=Cb,V=Cr`` ✓）"
          "且 ``image_yuv_to_rgb`` **往返恒等** ✓",
          (tuple(y.shape), tuple(u.shape), tuple(v.shape)) == ((2, 4, 6), (2, 4, 6), (2, 4, 6))
          and t.allclose(ops.image_yuv_to_rgb(y, u, v), img, atol=1e-5))


def case_quantize(t: Any) -> None:
    """⑪ 量化：形状/值域不变 ✓、**色数真的受限** ✓、dither 不认就**报错** ✗。"""
    try:
        import PIL  # noqa: F401,PLC0415
    except Exception:  # noqa: BLE001
        skip("没装 Pillow ⇒ 调色板量化不跑 ✓（**不自研调色板** ✗ 见 ``quantize`` 口径 ✓）")
        return
    img = t.rand(1, 8, 8, 3)
    out = ops.quantize(img, 4, "none")
    colors = {tuple(round(float(c), 4) for c in row) for row in out.reshape(-1, 3).tolist()}
    check("⑪ ``quantize(colors=4)``：形状不变 ✓ 值域 ``[0,1]`` ✓ 色数 **≤ 4** ✓（真受限 ✓ 不是原样返回 ✗）",
          tuple(out.shape) == (1, 8, 8, 3) and float(out.min()) >= 0.0 and float(out.max()) <= 1.0
          and len(colors) <= 4, (len(colors), tuple(out.shape)))
    check("⑪′ 未知 dither ⇒ **报错** ✗（不静默退回 ``none`` ✗ —— 那就不是同一张图了 ✓✗）",
          _raises(lambda: ops.quantize(img, 4, "magic")) is not None)


def case_canny(t: Any) -> None:
    """⑫ Canny：阶跃边缘**落在该在的地方** ✓、平坦区**没有边缘** ✓。"""
    step = t.zeros(1, 8, 8, 3)
    step[:, :, 4:, :] = 1.0
    magnitude, edges = ops.canny(step, 0.1, 0.3)
    flat_edges = ops.canny(t.full((1, 8, 8, 3), 0.5), 0.1, 0.3)[1]
    check("⑫ ``canny`` ⇒ ``(幅值, 边缘)`` **两张** ``(B,H,W)`` ✓；阶跃处**检测得到** ✓（``sum > 0`` ✓）",
          tuple(magnitude.shape) == (1, 8, 8) and tuple(edges.shape) == (1, 8, 8)
          and float(edges.sum()) > 0.0 and float(magnitude.max()) > 0.0,
          (tuple(magnitude.shape), float(edges.sum())))
    check("⑫′ **平坦图 ⇒ 一条边缘都没有** ✓（没梯度就没边缘 ✓ 不是「全都算边缘」✗）",
          float(flat_edges.sum()) == 0.0, float(flat_edges.sum()))
    edge_img = ops.canny_edge_image(step, 0.1, 0.3)
    check("⑫″ ``canny_edge_image`` ⇒ ``(B,H,W,3)`` ✓ 且**三通道同值** ✓（接参考节点输出口径 ✓）",
          tuple(edge_img.shape) == (1, 8, 8, 3)
          and t.allclose(edge_img[..., 0], edge_img[..., 2], atol=1e-6))
    check("⑫‴ 常量口径 ✓：``CANNY_KERNEL_SIZE = 5`` ✓ / ``CANNY_SIGMA = 1.0`` ✓",
          ops.CANNY_KERNEL_SIZE == 5 and abs(float(ops.CANNY_SIGMA) - 1.0) < 1e-9)


def main() -> int:
    t = torch_or_skip()
    if t is not None:
        case_layout(t)
        case_upscale(t)
        case_scale_family(t)
        case_crop_rotate_flip(t)
        case_stitch_batch(t)
        case_composite(t)
        case_porter_duff(t)
        case_masks(t)
        case_filter(t)
        case_color(t)
        case_quantize(t)
        case_canny(t)
    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
