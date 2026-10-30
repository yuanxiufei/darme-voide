"""**NVFP4**（E2M1 打包 + e4m3 块 scale + f32 二级 scale）的读取与反量化 ✓（2026-10-30）。

## 为什么单开一个模块 ✗

:h3 的唯一文本编码器 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`（14.61 GiB ✓）就是这一族 ✓，
而 :mod:`quant` 的形状判据**认不出它** ✗ —— 它的 `weight` 是 **U8 打包 4 bit**（`[N, K/2]` ✓）、
`weight_scale` 是 **F8_E4M3**（每 16 个元素一个 ✓）、另有 `weight_scale_2`（f32 标量 ✓），
形状上**自证不了**自己是 nvfp4 ✓（`fp8` / `int8` 的判据都会误命中 ✓✗）。

## 规格（逐位对齐，不靠猜 ✓）

参照实现：``reference/lora/musubi-tuner/src/musubi_tuner/modules/nvfp4_utils.py`` ✓
（抬头明写 “verified bit-level against the BF16 reference of the MiniMax-H3 Qwen3-VL text encoder” ✓，
且与 ComfyUI 侧 `comfy/float.py::stochastic_round_quantize_nvfp4_block` 的量化口径互逆 ✓）。

* **值表**（E2M1 ✓ 4 bit = 1 符号 + 2 指数 + 1 尾数 ✓ 共 16 个值 ✓）：
  ``0, .5, 1, 1.5, 2, 3, 4, 6`` 与它们的相反数 ✓；
* **打包顺序**：一个字节两个值 ✓ —— **元素 0 在「高」半字节** ✓✗（反了就是整份权重错位 ✓，
  而且**不报错** ✓✗）；
* **块 scale**：每 16 个元素一个 ✓，存成 **F8_E4M3** ✓，且是 cuBLAS 的
  **128×4 tiled（swizzled）布局** ✓✗ —— 它的 ``shape`` 看起来和「行主序的块表」一模一样 ✓
  （`[rows, cols/16]` ✓），**肉眼看形状以为对 ✓、按行主序解就是错的 ✓✗**；
* **二级 scale**：``weight_scale_2`` 一个 **f32 标量** ✓（per-tensor ✓）；
* **还原公式**：``w = fp4_value * (block_scale * weight_scale_2)`` ✓；
* **AWQ**：``pre_quant_scale`` ``[K]`` 是**乘在输入上**的 ✓（``x * s`` ✓，不是乘权重 ✓✗）——
  等价地可以折进权重**列**（``W[:, k] * s[k]`` ✓ 因为 ``Σ_k (x_k s_k) W_k = Σ_k x_k (W_k s_k)`` ✓），
  本模块两种都给 ✓（见 :func:`fold_pre_quant_scale` ✓）；
* **INT8 embedding**（同一个文件里的另一个族 ✓）：``weight`` I8 + ``weight_scale`` f32 ``[V, 1]``
  ⇒ ``w = q * s`` ✓（逐行 ✓，与 :mod:`quant` 的 ``int8`` 口径一致 ✓，本模块不重复实现 ✗）。

## 两条纪律 ✗

1. **形状必须自证** ✓：`in_features = shape[1] * 2` 要是 16 的倍数 ✓、
   `block_scale.numel() == roundup(rows,128) * roundup(in_features/16,4)` ✓ ——
   对不上就**拒绝** ✓，不按猜的布局硬解 ✓；
2. **整份 32B 不还原** ✗：nvfp4 → bf16 是 4 倍膨胀 ✓（14.61 GiB ⇒ ~64 GiB ✗，22.49 GiB 的卡装不下 ✓✗）
   ⇒ 只提供**逐层**还原（单层最大 ~250 MiB ✓），整份还原留给有内存的机器 ✓。
"""
from __future__ import annotations

import math
from typing import Any, Mapping

__all__ = ["BLOCK_SIZE", "COMFY_SCALE_2_SUFFIX", "COMFY_SCALE_SUFFIX", "E2M1_MAX", "E2M1_VALUES",
           "FORMAT", "PRE_QUANT_SCALE_SUFFIX", "dequantize", "dequantize_state",
           "fold_pre_quant_scale", "from_blocked", "layer_plan", "plan", "to_blocked",
           "unpack_e2m1"]

#: `comfy_quant` 里这一族的名字 ✓（大小写不敏感 ✓ 见 :func:`quant.format_family` ✓）
FORMAT = "nvfp4"

#: 每多少个元素一个块 scale ✓（nvfp4 的硬规定 ✓ 不是"可选参数" ✗）
BLOCK_SIZE = 16

#: E2M1 的最大可表示值 ✓（量化时用它做归一分母 ✓ 见 `comfy/float.py:160` ✓）
E2M1_MAX = 6.0

#: 4 bit 码 → 数值 ✓（code 0..7 为正 ✓、8..15 为负 ✓；**顺序就是规格** ✗ 别按"看起来顺眼"重排 ✓）
E2M1_VALUES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                  -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

COMFY_SCALE_SUFFIX = ".weight_scale"
COMFY_SCALE_2_SUFFIX = ".weight_scale_2"
PRE_QUANT_SCALE_SUFFIX = ".pre_quant_scale"


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _roundup(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple


# ── 纯 Python：体检（没有 torch 也能给出「这份到手后能不能还原」✓）───────────────────────────
def block_scale_elements(rows: int, in_features: int) -> int:
    """swizzled 块 scale 的**元素个数** ✓（= padded 形状的面积 ✓ 不是 ``rows * in_features/16`` ✗）。

    ⚠️ 两个 roundup 都**不能省** ✗：cuBLAS 的 tiled 布局按 128 行 × 4 块对齐 ✓
    ⇒ 行数不是 128 的倍数时（例如 `[2560, 25600]` 的 `up_proj` ✓）**要多出整块 padding** ✓，
    用"真实形状"算出来的数会小于文件里的数 ✓ ⇒ 误报「scale 缺元素」✓✗。
    """
    return _roundup(rows, 128) * _roundup(in_features // BLOCK_SIZE, 4)


def layer_plan(name: str, *, weight_shape: Any, weight_dtype: Any,
               scale_shape: Any, scale_dtype: Any, scale_2_shape: Any = None,
               scale_2_dtype: Any = None,
               pre_quant_shape: Any = None) -> tuple[dict[str, Any] | None, list[str]]:
    """单层体检 ⇒ ``(层信息 或 None, 问题清单)`` ✓（**纯形状** ✓ 一个字节都不用读 ✓）。

    判据全部是**等式** ✓（不猜 ✓）：dtype 对 ✓、`in_features` 是 16 的倍数 ✓、
    块 scale 元素数 == :func:`block_scale_elements` ✓、二级 scale 是标量 ✓、
    `pre_quant_scale` 长度 == `in_features` ✓。
    """
    problems: list[str] = []
    dtype = str(weight_dtype or "").upper()
    if dtype != "U8":
        problems.append(f"`{name}.weight` 的 dtype 是 {weight_dtype!r} ✗，nvfp4 必须是 **U8 打包** ✓")
    shape = tuple(int(dim) for dim in (weight_shape or ()))
    if len(shape) != 2:
        problems.append(f"`{name}.weight` 是 {len(shape)} 维 ✗，nvfp4 权重必须是 2 维 `[N, K/2]` ✓")
        return None, problems
    rows, packed_cols = shape
    in_features = packed_cols * 2
    if in_features % BLOCK_SIZE:
        problems.append(f"`{name}` 的 in_features={in_features} 不是 {BLOCK_SIZE} 的倍数 ✗"
                        f" ⇒ 块 scale 无从对齐 ✓")
    if str(scale_dtype or "").upper() not in ("F8_E4M3", "FLOAT8_E4M3FN"):
        problems.append(f"`{name}.weight_scale` 的 dtype 是 {scale_dtype!r} ✗，"
                        f"nvfp4 的块 scale 必须是 **F8_E4M3** ✓")
    expected = block_scale_elements(rows, in_features)
    got = 1
    for dim in (scale_shape or ()):
        got *= int(dim)
    if got != expected:
        problems.append(
            f"`{name}.weight_scale` 有 {got} 个元素 ✗，按 ({rows}, {in_features}) 的 swizzled 布局"
            f"应当是 {expected} 个 ✓（= roundup({rows},128) × roundup({in_features}/{BLOCK_SIZE},4) ✓）")
    if scale_2_shape is None:
        problems.append(f"`{name}` 缺 `.weight_scale_2`（f32 标量 ✓）⇒ 少一级尺度，还原会整体偏 ✓✗")
    else:
        numel = 1
        for dim in scale_2_shape or ():
            numel *= int(dim)
        if numel != 1:
            problems.append(f"`{name}.weight_scale_2` 有 {numel} 个元素 ✗，nvfp4 的二级 scale 是"
                            f"**每张量一个标量** ✓")
        if str(scale_2_dtype or "").upper() not in ("F32", "FLOAT32"):
            problems.append(f"`{name}.weight_scale_2` 的 dtype 是 {scale_2_dtype!r} ✗，应当是 F32 ✓")
    if pre_quant_shape is not None:
        length = 1
        for dim in pre_quant_shape or ():
            length *= int(dim)
        if length != in_features:
            problems.append(f"`{name}.pre_quant_scale` 长 {length} ✗，应当 == in_features {in_features} ✓"
                            f"（AWQ 是**乘在输入**上的 ✓ 长度对不上就不可能对 ✓）")
    return ({"name": name, "rows": rows, "inFeatures": in_features,
             "blockScaleElements": expected,
             "awq": pre_quant_shape is not None}, problems)


def plan(info: Any, formats: Any = ()) -> dict[str, Any]:
    """整份文件体检 ✓ ⇒ 与 :func:`quant.plan_dequant` **同形状**的结论 ✓（可直接塞进 `dequantPlan` ✓）。

    ``formats``：:func:`loader.quant_formats_of` 读出来的 ``comfy_quant`` 族说明 ✓ ——
    ⚠️ 它**去重后只留几份**（实测同族说明在同一文件里重复 350 次、内容一样 ✓）⇒
    **不能拿它当「层清单」** ✗✗：层清单从 ``info.tensors`` 里所有 ``*.comfy_quant`` 键取 ✓，
    族说明只用来回答「这份文件里到底有没有 nvfp4」✓。
    同文件里还有 **int8 的 `embed_tokens`**（`I8` + f32 行 scale ✓）⇒ 按 `weight` dtype **跳过** ✓
    （它不是 nvfp4 ✗，也不归本模块 ✗）。
    """
    tokens = {str((spec or {}).get("format") or "").strip().lower() for spec in formats or ()}
    tensors = getattr(info, "tensors", {}) or {}
    modules = sorted(name[: -len(".comfy_quant")] for name in tensors
                     if name.endswith(".comfy_quant"))
    layers: list[dict[str, Any]] = []
    problems: list[str] = []
    unresolved: list[str] = []
    if FORMAT not in tokens:
        return {
            "format": FORMAT, "layers": 0, "names": [], "awqLayers": 0, "unresolved": [],
            "unresolvedCount": 0, "declared": False, "problems": [],
            "supported": False,
            "note": f"文件里**没有**任何 `comfy_quant` 说自己是 {FORMAT} ✓ ⇒ 不归本模块管 ✓",
            "layout": {},
        }
    for name in modules:
        weight = tensors.get(f"{name}.weight")
        if weight is None:
            unresolved.append(name)
            problems.append(f"`{name}` 有 `comfy_quant` 但缺 `weight` ✓✗")
            continue
        if str(getattr(weight, "dtype", "")).upper() != "U8":
            continue  # int8 的 embedding 一类 ✓ 不是本族 ✗
        scale = tensors.get(f"{name}{COMFY_SCALE_SUFFIX}")
        if scale is None:
            unresolved.append(name)
            problems.append(f"`{name}` 是 U8 打包权重，但缺 `weight_scale` ✓✗")
            continue
        scale_2 = tensors.get(f"{name}{COMFY_SCALE_2_SUFFIX}")
        pre = tensors.get(f"{name}{PRE_QUANT_SCALE_SUFFIX}")
        item, issues = layer_plan(
            name, weight_shape=getattr(weight, "shape", ()),
            weight_dtype=getattr(weight, "dtype", None),
            scale_shape=getattr(scale, "shape", ()),
            scale_dtype=getattr(scale, "dtype", None),
            scale_2_shape=None if scale_2 is None else getattr(scale_2, "shape", ()),
            scale_2_dtype=None if scale_2 is None else getattr(scale_2, "dtype", None),
            pre_quant_shape=None if pre is None else getattr(pre, "shape", ()),
        )
        problems.extend(issues)
        if item is not None and not issues:
            layers.append(item)
    awq = sum(1 for item in layers if item["awq"])
    return {
        "format": FORMAT,
        "declared": True,
        "layers": len(layers),
        #: 逐层名单 ✓（`quant.plan_dequant` 靠它把这些 stem 从 `unresolved` 里**摘出来** ✓✗ ——
        #: 不摘的话同一次体检既说「能还原」又说「判不出布局」✓✗）
        "names": [item["name"] for item in layers],
        "awqLayers": awq,
        "unresolved": unresolved,
        "unresolvedCount": len(unresolved),
        "problems": problems,
        #: ⭐ 一句话结论 ✓（`True` ＝「这份到手后能自动还原」✓，体检阶段就能答 ✓ 不需要 torch ✓）
        "supported": not problems and bool(layers),
        "note": (f"nvfp4：{len(layers)} 层可还原 ✓（{awq} 层带 AWQ 预缩放 ✓）；"
                 f"w = fp4(E2M1) × (e4m3 块 scale[每 {BLOCK_SIZE} 元素] × f32 二级 scale) ✓"
                 if layers else "nvfp4：文件里没有任何可还原的层 ✗"),
        "layout": {
            "packed": "U8 [N, K/2]，元素 0 在高半字节 ✓",
            "blockScale": "F8_E4M3，每 16 元素一个 ✓，cuBLAS 128×4 tiled（swizzled）✓",
            "perTensorScale": "weight_scale_2，f32 标量 ✓",
            "awq": "pre_quant_scale 乘在**输入**上 ✓（或等价地折进权重列 ✓）",
        },
    }


# ── torch：数值还原（惰性导入 ✓ 顶层不碰 torch ✓）────────────────────────────────────────────
def _torch() -> Any:
    import torch  # noqa: PLC0415 —— 只有真要算时才 import ✓

    return torch


def to_blocked(matrix: Any) -> Any:
    """把行主序的块 scale 表排成 cuBLAS 的 128×4 tiled 布局 ✓（写文件 / 造测试数据用 ✓）。

    ⚠️ 与 :func:`from_blocked` **必须互逆** ✓（判据里做往返 ✓）——
    但"互逆"只证明两者自洽 ✗，不证明与 cuBLAS 一致 ✗ ⇒ 一致性靠与参照实现**对拍** ✓
    （`tests/engine_quant_nvfp4_test.py` ✓）。
    """
    rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    padded = matrix
    if (rows, cols) != (padded_rows, padded_cols):
        torch = _torch()
        padded = torch.zeros((padded_rows, padded_cols), device=matrix.device, dtype=matrix.dtype)
        padded[:rows, :cols] = matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.reshape(padded_rows, padded_cols)


def from_blocked(blocked: Any, num_rows: int, num_cols: int) -> Any:
    """把 cuBLAS tiled 布局的块 scale **还原**成行主序 ``[num_rows, num_cols]`` ✓。"""
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    step1 = blocked.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    unblocked = step5.reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols]


#: 字节 → (高半字节的值, 低半字节的值) ✓ 按 (device, dtype) 缓存 ✓（50 层重建 50 次纯属浪费 ✗）
_BYTE_LUT_CACHE: dict[tuple[str, str], Any] = {}


def _byte_pair_lut(device: Any, dtype: Any) -> Any:
    torch = _torch()
    key = (str(device), str(dtype))
    lut = _BYTE_LUT_CACHE.get(key)
    if lut is None:
        values = torch.tensor(E2M1_VALUES, dtype=torch.float32)
        codes = torch.arange(256, dtype=torch.int64)
        lut = torch.stack([values[codes >> 4], values[codes & 0x0F]], dim=1).to(device=device, dtype=dtype)
        _BYTE_LUT_CACHE[key] = lut
    return lut


def unpack_e2m1(packed: Any, *, dtype: Any = None) -> Any:
    """``U8 [N, K/2]`` → 浮点 ``[N, K]`` ✓（**一次查表解一个字节的两个半字节** ✓ 高半字节在前 ✓）。"""
    torch = _torch()
    out_dtype = dtype or torch.float32
    if packed.dtype != torch.uint8:
        raise ValueError(f"nvfp4 打包权重必须是 uint8 ✗，拿到 {packed.dtype} ✓")
    if packed.ndim != 2:
        raise ValueError(f"nvfp4 打包权重必须是 2 维 ✗，拿到 {packed.ndim} 维 ✓")
    lut = _byte_pair_lut(packed.device, out_dtype)
    return torch.nn.functional.embedding(
        packed.reshape(-1).to(torch.int64), lut).reshape(packed.shape[0], packed.shape[1] * 2)


def dequantize(packed: Any, block_scale: Any, per_tensor_scale: Any, orig_shape: Any,
               *, dtype: Any = None) -> Any:
    """nvfp4 单层权重 → 浮点 ``orig_shape`` ✓（``w = fp4 × (块 scale × 标量 scale)`` ✓）。

    ⚠️ ``block_scale`` 是 **swizzled** 的 ✓（原样喂进来 ✓，本函数负责还原 ✓）——
    若调用方已经还原过，就会**再扭一次** ✓✗（扭完还是同形状的合法张量 ✓ ⇒ 不报错的错 ✓✗）。
    """
    torch = _torch()
    out_dtype = dtype or torch.float32
    rows = int(packed.shape[0])
    cols = int(packed.shape[1]) * 2
    expected = block_scale_elements(rows, cols)
    if int(block_scale.numel()) != expected:
        # ⚠️ 这里必须**当场拒绝** ✗✗：块 scale 少一截时 `from_blocked` 会 reshape 失败 ✓，
        #    但**多**一截时它能默默跑通 ✓ ⇒ 布局整体错位而**不报错** ✓✗（最难查的那种 ✓）。
        raise ValueError(
            f"nvfp4 块 scale 有 {int(block_scale.numel())} 个元素 ✗，按 ({rows}, {cols}) 的"
            f"swizzled 布局应当是 {expected} 个 ✓")
    weight = unpack_e2m1(packed, dtype=out_dtype)
    scales = from_blocked(block_scale, rows, cols // BLOCK_SIZE)
    total = (per_tensor_scale.float() * scales.float()).to(out_dtype)
    weight = (weight.reshape(rows, -1, BLOCK_SIZE) * total.unsqueeze(-1)).reshape(rows, cols)
    want_rows, want_cols = int(orig_shape[0]), int(orig_shape[1])
    return weight[:want_rows, :want_cols]


def fold_pre_quant_scale(weight: Any, pre_quant_scale: Any) -> Any:
    """把 AWQ 的 ``pre_quant_scale`` 折进权重**列** ✓ ⇒ 输入侧不用再乘 ✓（数值等价 ✓）。

    ``Σ_k (x_k · s_k) · W_k = Σ_k x_k · (W_k · s_k)`` ✓ —— 所以折进列与"乘在输入上"**逐位同义** ✓
    （差别只有浮点乘法的结合顺序 ✓，bf16 上约 1 ulp ✓ 可接受 ✓）。

    ⚠️ 长度必须 == ``in_features`` ✓ 否则拒绝 ✓（错一列就是整层错 ✓✗）。
    """
    torch = _torch()
    if pre_quant_scale is None:
        return weight
    length = int(pre_quant_scale.reshape(-1).shape[0])
    if length != int(weight.shape[1]):
        raise ValueError(f"pre_quant_scale 长 {length} ✗ ≠ in_features {int(weight.shape[1])} ✓")
    return weight * pre_quant_scale.reshape(1, -1).to(weight.dtype)


def dequantize_state(state: Mapping[str, Any], *, dtype: Any = None,
                     orig_shapes: Mapping[str, Any] | None = None) -> Any:
    """把一份（**已按模块聚好**的）state 里的 nvfp4 层还原 ✓ ⇒ :class:`quant.DequantResult` ✓。

    ``state`` 的键就是文件里的键 ✓（``<module>.weight`` / ``.weight_scale`` / ``.weight_scale_2`` /
    ``.pre_quant_scale`` ✓），配套张量解完**丢掉** ✓（模型里没有这些键 ✓）。

    ``orig_shapes``：``{模块名: (out, in)}`` ✓ —— 打包权重自身就能推出 ✓（``in = shape[1]*2`` ✓），
    这一项只用于**裁剪 padding** ✓（当 ``rows``/``cols`` 带 tiled padding 时 ✓）。

    ⚠️ **整份 32B 不要用这个** ✗（4 倍膨胀 ✓）⇒ 逐层用 :func:`dequantize` ✓；
    本函数服务"小模型 / 单层调试 / 判据对拍" ✓。
    """
    from app.services.engine import quant as quant_mod  # noqa: PLC0415 —— 复用同一个结果容器 ✓

    torch = _torch()
    result = quant_mod.DequantResult()
    converted = dict(state)
    companions: list[str] = []
    stems = sorted({key[: -len(".weight")] for key in state if key.endswith(".weight")})
    for stem in stems:
        packed = state.get(f"{stem}.weight")
        if packed is None or getattr(packed, "dtype", None) != torch.uint8:
            continue
        scale = state.get(f"{stem}{COMFY_SCALE_SUFFIX}")
        scale_2 = state.get(f"{stem}{COMFY_SCALE_2_SUFFIX}")
        if scale is None or scale_2 is None:
            result.problems.append(
                f"`{stem}` 是 nvfp4 打包权重（U8 ✓）但缺 "
                f"`{'weight_scale' if scale is None else 'weight_scale_2'}` ✗ ⇒ 不还原 ✓")
            continue
        shape = (orig_shapes or {}).get(stem) or (int(packed.shape[0]), int(packed.shape[1]) * 2)
        value = dequantize(packed, scale, scale_2, shape, dtype=dtype or torch.float32)
        pre = state.get(f"{stem}{PRE_QUANT_SCALE_SUFFIX}")
        if pre is not None:
            value = fold_pre_quant_scale(value, pre)
            companions.append(f"{stem}{PRE_QUANT_SCALE_SUFFIX}")
        converted[f"{stem}.weight"] = value
        companions.extend([f"{stem}{COMFY_SCALE_SUFFIX}", f"{stem}{COMFY_SCALE_2_SUFFIX}"])
        result.converted[f"{stem}.weight"] = {
            "layout": "nvfp4", "scale": f"{stem}{COMFY_SCALE_SUFFIX}",
            "inverse": "False", "from": "torch.uint8", "awq": pre is not None,
        }
    for name in companions:
        converted.pop(name, None)
    result.companions.extend(companions)
    result.tensors = converted
    return result
