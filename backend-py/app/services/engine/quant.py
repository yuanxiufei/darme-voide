r"""**反量化**（fp8 / int8 权重 → 可计算的高精度张量 ✓）—— 纯函数 ✓ 只依赖 torch ✓。

## 为什么必须有它 ✗

目标主权重是 **19.53 GiB 的 fp8** ✓（50 层 × [5376/14336] ✓ —— bf16 会是 ~39 GiB ✗）。
而 `load_module_weights` 现在的做法是 `tensor.to(dtype)` ✓✗ —— 对 fp8 张量来说这是**把尺度丢掉** ✗：
fp8 存的是「缩放过的整数/浮点」✓，**不乘 scale 直接 cast** ⇒ 得到一个**看着正常、数值全错**的模型 ✓✗
（而且**不会报错** ✗ —— `load_state_dict` 会照单全收 ✓，症状要到出片才发现「画面是噪声」✓✗）。

⚠️ **本模块的纪律：判不出来就拒绝** ✗（宁可报错 ✓ 也不按猜的布局反量化 ✓）：

* 布局**按形状判** ✓ 而不是按文件名猜 ✗（`scale` 是标量 ⇒ per-tensor ✓、`(out,)` ⇒ per-row ✓、
  `(in,)` ⇒ per-col ✓、与权重同形 ⇒ 逐元素 ✓；`[N,1]`/`[1,M]` 这类**带退化维**的写法与
  `(N,)`/`(M,)` 等价 ✓ —— 见 :func:`_squeeze_scale_shape`）；
* **文件自己说了算** ✓：ComfyUI 的打包格式给每个低精度权重配一个 `*.comfy_quant` 小张量
  （JSON ✓，55~72 字节 ✓）⇒ 里面的 `format` / `convrot` / `convrot_groupsize` 是**权威** ✓
  （见 :func:`parse_format_spec` / :func:`format_family`）；
* 形状**对不上任何一种** 且 `comfy_quant` 没说是**哪一族** ⇒ 拒绝 ✓（若它说了哪一族 ✓，
  拒绝理由就用**那一族的话** ✓ —— 别让人去找不存在的东西 ✗）；
* `zeros`/`qzeros`（int4 一类 ✗）⇒ 拒绝 ✓；
* 低精度权重**没有**配套 scale ⇒ 拒绝 ✓（`loader` 只报「缺 scale」✓，这里**不装** ✗）；
* **能做 vs 不能做**要分开说 ✓：:data:`SUPPORTED_FORMATS`（nvfp4 ⇒ 归 :mod:`quant_nvfp4` ✓）与
  :data:`UNSUPPORTED_FORMATS`（mxfp4 ✓）**两张表** ✓ —— 前者是「本仓做得了、只是形状判据认不出」✓，
  后者是「本仓真没有解码表」✗。2026-10-30 之前 nvfp4 混在后一张表里 ⇒ 报告说「未实现」而
  `quant_nvfp4` 已经能逐位还原 ✓✗（同一份文档两处互相打脸 ✓）。

## 两条 scale 方向约定 ✓

* `*.scale` / `*.scales` / `*.weight_scale` ⇒ ``w = q * s`` ✓；
* `*.weight_scale_inv` ⇒ ``w = q / s`` ✓（DeepSeek 系的写法 ✓ —— 名字里的 `inv` 就是「乘倒数」✓）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Mapping as MappingABC
from typing import Any, Mapping

__all__ = ["CAST_ONLY_DTYPES", "COMPANION_SUFFIXES", "DequantResult", "INPUT_SCALE_SUFFIXES",
           "LOW_PRECISION_DTYPES", "METADATA_QUANT_KEY", "QUANT_META_SUFFIX", "SCALED_FP8_MARKER",
           "SCALE_SUFFIXES", "SUPPORTED_FORMATS", "UNSUPPORTED_FAMILIES", "UNSUPPORTED_FORMATS",
           "build_hadamard", "convrot_group_of", "convrot_of_spec", "dequantize_state",
           "dequantize_weight_rotated", "format_family", "group_for", "is_cast_only",
           "FORMAT_ALIASES", "format_alias", "GENERIC_SCALE_SUFFIXES", "INT4_PACK_TOKENS",
           "INT4_STORAGE_DTYPES", "int4_packed_of", "int4_shape_problem", "int4_stems_of",
           "is_input_scale", "is_int4_storage", "is_low_precision", "is_scaled_fp8_marker",
           "layout_of", "logical_weight_shape", "orphan_scales", "parse_format_spec",
           "plan_dequant", "precheck_dequant", "quant_report", "rotate_activations",
           "rotate_weight", "scale_stem", "specs_from_metadata", "supported_format",
           "unpack_int4", "unsupported_family", "WEIGHT_SCALE_SUFFIXES"]

#: 需要反量化才能计算的 dtype ✓（**只有这些** ✗ —— bf16/fp16/fp32 本来就是可算精度 ✓）
#: ⚠️ 含**两套写法** ✓：torch 的 `float8_e4m3fn` ✓ 与 safetensors 头部的 `F8_E4M3` ✓（见 `is_low_precision` ✓）。
LOW_PRECISION_DTYPES: tuple[str, ...] = ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz",
                                         "float8_e5m2fnuz", "int8", "uint8")

#: 归一化后的低精度名字集合 ✓（`is_low_precision` 用它 ✓）
_LOW_PRECISION_NAMES = frozenset(LOW_PRECISION_DTYPES) | {
    "f8_e4m3", "f8_e5m2", "f8_e4m3fn", "f8_e5m2fnuz", "f8_e4m3fnuz",
    "i8", "u8",
}

#: 名字里**自带 `weight`** 的 scale 后缀 ✓（**顺序有意义** ✗：先找带 `_inv` 的 ✓ —— 否则
#: `weight_scale_inv` 会被 `weight_scale` **前缀命中** ✗ ⇒ 把「除」当成「乘」 ✓✗，那是最坏的一种错 ✓）。
#: ⚠️ 语义**无歧义** ✓：名字里就写着 weight ⇒ 一定是给某个 ``*.weight`` 用的量化 scale ✓。
WEIGHT_SCALE_SUFFIXES: tuple[str, ...] = (
    ".weight_scale_inv", ".weight_scale",
    #: ⚠️ **fp8_scaled 族的写法**（2026-10-30 实测 ✓）：ComfyUI 转换器把 per-tensor scale 写成
    #: ``xxx.scale_weight``（**标量** ``shape=()`` ✓）+ ``xxx.scale_input``（激活侧 ✓）。
    #: 少这一条 ⇒ 整个 fp8_scaled 族都报「找不到配套 scale」✗（盘上 `wan2.2_i2v_high_noise_14B_
    #: fp8_scaled` 406 对、`qwen_2.5_vl_7b_fp8_scaled` 337 对、`umt5_xxl_fp8_e4m3fn_scaled` 168 对 ✓）。
    #: 语义与 ``weight_scale`` **完全相同**（乘 ✓）：实测 ``q.max()≈±448`` 且
    #: ``scale_weight = max(|w|)/448`` ✓ ⇒ ``w = q × scale_weight`` ✓ 用不到「除」✗。
    ".scale_weight",
)

#: **通用名**的 scale 后缀 ✓ —— ⚠️ 这类名字**也被当归一化层的可学习权重**用 ✓✗，
#: 实测打脸（2026-10-30，`:func:`orphan_scales` 第一次跑真文件就抓到的误报 ✓）：
#: `krea2_turbo_int8_convrot.safetensors` 里 `blocks.N.prenorm.scale` / `postnorm.scale` /
#: `attn.qknorm.qnorm.scale` / `txtfusion.*.prenorm.scale` / `txtmlp.0.scale` 共 **138 个**
#: 全是 **RMSNorm 的缩放参数** ✓、一个都不是量化 scale ✓ —— 把它们当孤儿报出来是**纯误报** ✓✗。
#: ⇒ 判据：只有「同 stem 下真有低精度 ``.weight``」才算量化 scale ✓；否则它就是**独立权重** ✓
#: （落点在 :func:`orphan_scales` ✓）。
GENERIC_SCALE_SUFFIXES: tuple[str, ...] = (".scales", ".scale", ".scale_inv")

#: scale 的候选后缀 ✓（**单一真源** ✓：:func:`_scale_for` / :func:`scale_stem` /
#: :func:`orphan_scales` / `loader._SCALE_SUFFIXES` 全读这一张 ✓ —— 手抄第二份就会分叉 ✓✗）
SCALE_SUFFIXES: tuple[str, ...] = WEIGHT_SCALE_SUFFIXES + GENERIC_SCALE_SUFFIXES

#: **激活侧** scale 的写法 ✓（fp8_scaled 的 ``.scale_input`` / 别家的 ``.input_scale``）——
#: ⚠️ 它们**不是**权重的 scale ✗：语义是「激活先按这个标量缩放再进 matmul」✓。
#: 本仓把低精度权重还原成 dense 高层精度来算 ✓ ⇒ 激活侧那一步在数学上**整体抵消** ✓
#: （实测盘上 4 个 fp8_scaled 文件的全部 ``.scale_input`` 都恰好是 ``1.0`` ✓ ⇒ 丢掉是**无损**的 ✓）；
#: 但**不能当权重键留下** ✗（模型里没有这个参数 ⇒ 会变成 ``unexpected`` 噪声 ✓✗）。
INPUT_SCALE_SUFFIXES: tuple[str, ...] = (".scale_input", ".input_scale")

#: 「整份文件都是 fp8_scaled」的**标记张量** ✓（实测 ``scaled_fp8``：``F8_E4M3`` / ``shape=(0,)`` ✓，
#: 一个字节都不占 ✓）—— 它不是模型参数 ✗，读的人只该记下「这文件是这么来的」✓。
SCALED_FP8_MARKER = "scaled_fp8"

#: 明确**不做**的配套张量 ✗（要 group size ✓ 在随附 json 里 ✓ 本仓不猜 ✓）
_GROUP_SUFFIXES: tuple[str, ...] = (".zeros", ".qzeros", ".g_idx", ".qweight")

#: 「低精度权重一定带配套张量」的**整张表** ✓（scale ✓ ＋ 分组量化的 zeros/g_idx ✓）——
#: ⚠️ 体检侧要问「这层缺不缺配套」时必须用**这一张** ✗：手抄第二份就会分叉 ✓✗
#: （`loader._missing_scales` 原先自己写了一份 ✓）。
COMPANION_SUFFIXES: tuple[str, ...] = SCALE_SUFFIXES + _GROUP_SUFFIXES

#: **不带 scale 也能无损还原**的低精度 dtype ✓（只有 fp8 ✓）——
#: ⚠️ 这不是「宽松放行」✗：fp8 的**每一个值**都能被 bf16 精确表示 ✓（bf16 尾数 8 位 ✓，
#: E4M3 只有 3 位 ✓、E5M2 只有 2 位 ✓）⇒ ``q.to(bf16)`` 就是**无损解码** ✓，缺的那一步乘法本来就等于 ×1 ✓。
#: 实测盘上 `seedvr2_ema_3b_fp8_e4m3fn`（301 个权重 ✓）/ `seedvr2_ema_7b*`（最多 1128 个 ✓）
#: **一个 scale 都没有** ✗ —— 旧判据报「找不到配套 scale ⇒ 不装」✓✗ 属**假拒绝**（拒绝不是安全 ✗）。
#: ⚠️ 只对 fp8 开这个口子 ✗：int4/int8 无 scale 是**真坏** ✓（会静默错一个数量级 ✓✗）⇒ 仍拒 ✓。
CAST_ONLY_DTYPES: frozenset[str] = frozenset({"float8_e4m3fn", "float8_e4m3fnuz",
                                              "float8_e5m2", "float8_e5m2fnuz",
                                              #: ⚠️ safetensors 头部写的是**另一套名字** ✗（`F8_E4M3` ✓）——
                                              #: 只按 torch 名字判 ⇒ 盘上文件一个都不认 ✓✗（第一版就踩了 ✓）
                                              "f8_e4m3", "f8_e4m3fn", "f8_e5m2", "f8_e5m2fnuz"})

#: 名字**前缀**兜底 ✓（两套命名 `F8_E4M3` / `torch.float8_e4m3fn` 都能命中 ✓ ——
#: 以后 safetensors 再添个新 fp8 变体也不会被判成「缺 scale」✗）
_CAST_ONLY_PREFIXES: tuple[str, ...] = ("f8_", "float8")


def is_cast_only(dtype: Any) -> bool:
    """这个 dtype 能不能**不带 scale** 直接 cast 还原 ✓（只有 fp8 ✓ —— 见 :data:`CAST_ONLY_DTYPES`）。"""
    name = _normalize_dtype(dtype)
    return name in CAST_ONLY_DTYPES or name.startswith(_CAST_ONLY_PREFIXES)


def is_low_precision(dtype: Any) -> bool:
    """这个 dtype 是不是「必须先反量化」✓（未知 dtype ⇒ ``False`` ✓ 不猜 ✓）。

    ⚠️ **两套写法都要认** ✗（2026-09-22 踩过 ✓）：**装载**路径拿的是 torch 的 ``float8_e4m3fn`` ✓，
    而**体检**路径拿的是 safetensors **头部里的字符串** ``F8_E4M3`` ✓ —— 只认前面那套 ⇒
    体检永远报「没有低精度权重」✓✗（而 `loader._DTYPE_CLASS` 用的是后面那套 ✓ 两处口径不同 ✗）。
    """
    return _normalize_dtype(dtype) in _LOW_PRECISION_NAMES


def _normalize_dtype(dtype: Any) -> str:
    """统一成小写、去掉 ``torch.`` 前缀 ✓（两套写法的**公共分母** ✓）。"""
    return str(dtype).replace("torch.", "").strip().lower()


def is_input_scale(name: Any) -> bool:
    """这个名字是不是**激活侧** scale ✓（``.scale_input`` / ``.input_scale`` ✓）。

    ⚠️ 只看**名字** ✗、不看值 ✓：值恰好是 1.0 只是「丢掉无损」的实测依据 ✓，
    判据不能依赖它 ✗（换一份文件写成 1.7 也照样该丢 —— 见 :data:`INPUT_SCALE_SUFFIXES`)
    """
    return str(name).endswith(INPUT_SCALE_SUFFIXES)


def is_scaled_fp8_marker(name: Any) -> bool:
    """``scaled_fp8`` 这种**整份文件标记** ✓（不是权重 ✗ 也不是谁的 scale ✗）。"""
    return str(name) == SCALED_FP8_MARKER


@dataclass
class DequantResult:
    """一次反量化的结论 ✓（``ok=False`` ⇒ 调用方**不许继续装** ✗）。"""

    tensors: dict[str, Any] = field(default_factory=dict)
    #: 每个被反量化的权重 ⇒ (布局, 来源 scale 名 ✓)——
    #: ⚠️ ``scale`` 可以是 ``None`` ✓（``cast`` 布局：fp8 本来就没有 scale ✓，见 :data:`CAST_ONLY_DTYPES`）
    converted: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: 被吃掉的配套张量（它们**不进模型** ✓ —— 模型里没有这些键 ✓）
    companions: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    #: 有多少层走了 **4bit 解包** ✓（`w4a4`/`int4` ✓）—— 这是**一步真的计算** ✗ 不是记账 ✓：
    #: 不数它的话，「装了 224 层」与「装了 224 层且每层都真的解了包」在报告里长得一样 ✓✗
    int4PackCount: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "count": len(self.converted), "converted": dict(self.converted),
                "companions": list(self.companions), "problems": list(self.problems),
                "int4PackCount": int(self.int4PackCount)}


def _squeeze_scale_shape(scale_shape: Any) -> tuple[int, ...]:
    """把带**退化维**的 scale 写法归一 ✓：``[N,1]`` ⇒ ``(N,)``、``[1,M]`` ⇒ ``(M,)``、``[1,1]`` ⇒ ``(1,)``。

    ⚠️ 为什么必须归一（2026-10-29 实测盘上两个 19.53 GiB 主 DiT ✓）：``*_int8_convrot.safetensors``
    里 ``blocks.0.attn.out_proj.weight`` 是 ``[5376, 7168]``、配套 ``weight_scale`` 是 ``[5376, 1]``
    ✓ —— 这正是 **per-row** 的常见写法（保留尾维 ⇒ 可以直接广播 ✓）。旧判据只认 ``(5376,)`` ⇒
    把它读成「对不上任何布局」✗，与文件自带的 ``{"format": "int8_tensorwise"}`` 直接矛盾 ✓✗
    ⇒ 报告于是让人「去找随附 json 的 group size」✗（而那个 json 根本不存在 ✓）。

    ⚠️ 这是**形状自证** ✓ 不是猜 ✗：``[N,1]`` 与 ``(N,)`` 在广播语义下是同一件事 ✓。
    """
    flat = tuple(int(dim) for dim in scale_shape)
    if len(flat) > 1 and all(dim == 1 for dim in flat[1:]):
        return (flat[0],)
    if len(flat) > 1 and all(dim == 1 for dim in flat[:-1]):
        return (flat[-1],)
    return flat


# ── 4bit **打包**权重（`w4a4` / `int4` ✓）：每字节两个**有符号** 4bit ─────────────────────────
#: 声明「权重是 4bit」的 token ✓ —— 出处**只有一个**：文件自己写的 `comfy_quant`（或文件级
#: `_quantization_metadata` ✓），字段是 ``format`` 或 ``linear_dtype`` ✓。
#: ⚠️ **不能按形状猜** ✗✗：打包后的 ``(6144, 3072)`` + ``weight_scale (6144,)`` 与普通 per-row
#: int8 **逐字相同** ✓（这正是 2026-10-30 抓到的假绿：`:func:`layout_of` 照 per-row 判、装载照
#: int8 乘 ⇒ 数值全错却报「可自动还原」✓✗）。
INT4_PACK_TOKENS: tuple[str, ...] = ("w4a4", "int4")

#: 4bit 打包的**存储** dtype ✓：每元素 1 字节 ⇒ 逻辑维是声明维的 **2 倍** ✓
INT4_STORAGE_DTYPES: tuple[str, ...] = ("i8", "u8", "int8", "uint8", "i4", "u4")


def is_int4_storage(dtype: Any) -> bool:
    """这个 dtype 能不能装「每字节两个 4bit」✓（两套命名都认 ✓：`I8` 与 `torch.int8` ✓）。"""
    return str(dtype).replace("torch.", "").strip().lower() in INT4_STORAGE_DTYPES


def int4_packed_of(spec: Any) -> bool:
    """这一层的说明是否自称**4bit 打包**权重 ✓（``format=convrot_w4a4`` / ``linear_dtype=int4`` ✓）。

    实测的四种写法（盘上 6 个 int4 文件全中 ✓）：``{"format": "convrot_w4a4", "convrot_groupsize":
    256, "linear_dtype": "int4"}`` / ``{"format": "convrot_w4a4", "convrot_groupsize": 256,
    "quant_group_size": 64}`` / ``{"convrot_groupsize": 256, "format": "convrot_w4a4"}`` /
    文件级 ``{"format": "int4_convrot", "layers": {…}}`` ✓ —— 都含 `w4a4` 或 `int4` ✓。
    ``format=int8_tensorwise`` **不含**这两个 token ⇒ 不误伤 ✓。
    """
    if not isinstance(spec, MappingABC):
        return False
    for field in ("linear_dtype", "format"):
        text = str(spec.get(field) or "").strip().lower()
        if any(token in text for token in INT4_PACK_TOKENS):
            return True
    return False


def int4_stems_of(specs: Any) -> frozenset[str]:
    """``{层名: 说明}`` ⇒ 自称 4bit 打包的**层名集合** ✓（判据只在 :func:`int4_packed_of` 一处 ✓）。"""
    if not isinstance(specs, MappingABC):
        return frozenset()
    return frozenset(str(stem) for stem, spec in specs.items() if int4_packed_of(spec))


def logical_weight_shape(shape: Any, *, packed: bool) -> tuple[int, ...]:
    """**逻辑**权重形状 ✓（4bit 打包 ⇒ 最后一维 ×2 ✓；否则原样 ✓）。

    ⚠️ 只对 2D 生效 ✗：打包只发生在 ``Linear`` 权重上 ✓，1D（norm/bias ✓）没有打包形态 ✓。
    """
    dims = tuple(int(dim) for dim in shape)
    if not packed or len(dims) != 2:
        return dims
    return (dims[0], dims[1] * 2)


def int4_shape_problem(name: str, dtype: Any, shape: Any) -> str | None:
    """声明了 4bit 打包、但**存不了**打包 ⇒ 拒绝 ✗（不静默当 int8 乘 ✗）。"""
    dims = tuple(int(dim) for dim in shape)
    if len(dims) != 2:
        return (f"`{name}` 自称 4bit 打包（`w4a4`/`int4` ✓）但形状 {dims} 不是 2D ✗"
                f"⇒ 解包无从下手（拒绝反量化 ✓）")
    if not is_int4_storage(dtype):
        return (f"`{name}` 自称 4bit 打包（`w4a4`/`int4` ✓）但存储 dtype 是 "
                f"`{str(dtype).replace('torch.', '')}` ✗ 不是单字节整型 ✓ ⇒ 拒绝反量化 ✓"
                f"（不按猜的打包方式解 ✗）")
    return None


def unpack_int4(packed: Any) -> Any:
    """把**每字节两个有符号 4bit** 的打包权重解成整型权重 ✓ ⇒ ``(rows, 2*cols)``。

    ## 约定是**实测**出来的，不是猜的（2026-10-30 ✓）

    两种候选 × 两种符号约定共 4 种组合，用**同源 int8 版**做基准逐层比对：
    `krea2_v1.5_anime_int4_w4a4` 对 `krea2_v1.5_anime_int8_convrot`、
    `krea2_turbo_int4_convrot` 对 `krea2_turbo_int8_convrot`（取 `blocks.0.attn.wq`，6144×6144）：

    ======================  ============  ==========
    解包约定                 余弦相似度     L2 相对误差
    ======================  ============  ==========
    **低半字节=偶数列, 有符号**  **0.9946 / 0.9949**  **0.165 / 0.166**
    低半字节=偶数列, 偏移 8     -0.652        4.56
    高半字节=偶数列, 有符号      -0.007        1.43
    高半字节=偶数列, 偏移 8      0.005        3.97
    ======================  ============  ==========

    ⇒ ① **低半字节在前** ✓（`b & 0x0F` ⇒ 偶数列 ✓）；② **有符号**（nibble ≥ 8 ⇒ −16 ✓）；
    ③ 剩下 16.5% 的 L2 误差正是 4bit 对 bf16 的量化误差量级 ✓（4bit 步长是 int8 的 16 倍 ✓）。
    两个**不同模型**给出几乎逐位相同的结论 ✓ ⇒ 不是单文件巧合 ✓。

    ⚠️ 顺带回答「为什么以前没发现」✗：打包后的形状 ``(6144, 3072)`` 与普通 per-row int8
    **完全一样** ✓，而 ``weight_scale`` 是 ``(6144,)`` = 行数 ✓ ⇒ 每一处形状判据都说「对得上」✓✗。
    唯一能识破它的是**文件自己写的 `w4a4`** ✓。
    """
    torch = _torch()
    if int(packed.dim()) != 2:
        raise ValueError(f"int4 打包只作用于 2D 权重，收到 {tuple(packed.shape)} ✗")
    if not is_int4_storage(packed.dtype):
        raise ValueError(f"int4 打包的存储 dtype 必须是单字节整型，收到 {packed.dtype} ✗")
    raw = packed.to(torch.uint8)
    low = (raw & 0x0F).to(torch.int8)
    high = ((raw >> 4) & 0x0F).to(torch.int8)
    #: 有符号 4bit：0..7 原样、8..15 ⇒ −8..−1 ✓（实测见 docstring ✓）
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    #: 低半字节是**偶数列** ⇒ 先 low 后 high ✓
    return torch.stack((low, high), dim=-1).reshape(int(packed.shape[0]), int(packed.shape[1]) * 2)


def layout_of(weight_shape: Any, scale_shape: Any) -> str | None:
    """按**形状**判 scale 的布局 ✓ ⇒ ``per-tensor`` / ``per-row`` / ``per-col`` / ``elementwise`` ✓。

    ⚠️ **纯形状** ✓ 不碰张量 ✗ ⇒ **体检阶段**（只有 safetensors 头部 ✓ 没有 torch 也能跑 ✓）与
    **装载阶段**用的是**同一句话** ✓ —— 判定分两处写就一定会漂 ✓✗（本仓踩过：`Split` 的拒绝理由 ✓）。

    ``None`` ⇒ **判不出来** ✓（块量化一类 ✓ 调用方**必须拒绝** ✗ 不猜 ✓）。
    """
    flat = _squeeze_scale_shape(scale_shape)
    if flat in ((), (1,)):
        return "per-tensor"
    dims = tuple(int(dim) for dim in weight_shape)
    if len(dims) < 2:
        # 一维权重（norm / bias 一类 ✓）：只有**标量** scale 才说得清 ✓（向量 ⇒ 方向不明 ✗）
        return None
    rows, cols = dims[0], dims[1]
    if flat == (rows,):
        return "per-row"
    if flat == (cols,):
        return "per-col"
    if flat == (rows, cols):
        return "elementwise"
    return None


def plan_dequant(tensors: Mapping[str, Any], *, convrot: Any = None,
                 int4: Any = None) -> dict[str, Any]:
    """**体检阶段**就能说的结论 ✓（零依赖 ✓ 不需要 torch ✓）：这份检查点的低精度权重**能不能自动还原** ✓。

    入参 ``tensors``：``{名字: (dtype 字符串, 形状)}`` ✓ —— ⚠️ **两项都要** ✗：
    只看形状说不出哪个张量是低精度 ✓✗，只看 dtype 说不出 scale 的布局 ✓✗。

    返回：``weights`` 低精度权重数 ✓ / ``paired`` 找到配套 scale 的 ✓ / ``layouts`` 各布局计数 ✓ /
    ``unresolved`` **判不出布局**的 ✓ / ``grouped`` 分组量化（带 `zeros`/`g_idx` ✓）的 ✓ /
    ``unpaired`` 缺 scale 的 ✓ / ``orphanScales`` **没人要的 scale** ✓（反向那一半 ✓
    —— 见 :func:`orphan_scales`：融合 QKV 被拆成分片 scale 时就是这个形态 ✓；只报不拦 ✓）。

    ⚠️ 与 :func:`dequantize_state` **同一套判据** ✗（同一个 `layout_of` ✓ 同一张后缀表 ✓）——
    不然会出现「体检说能装、装载时被拒」✓✗（那种矛盾最费时间 ✓）。

    ``convrot``：旋转分组（由 :func:`convrot_group_of` 判出 ✓），三种形状都收 ✓ —— ``None`` /
    ``256``（整文件同组 ✓）/ ``{层名: group}``（**逐层不同** ✓，实测 H3 全量版 64 与 256 并存 ✓）；
    布局名会带 ``+convrot256`` 后缀 ✓；形状不符的进 ``unresolved`` ✓（**绝不静默当普通 per-row** ✗）。
    """
    names = list(tensors)
    packed_stems = frozenset(int4 or ())
    count = 0
    paired = 0
    layouts: dict[str, int] = {}
    unresolved: list[str] = []
    unpaired: list[str] = []
    grouped: list[str] = []
    int4_problems: list[str] = []
    for name in sorted(name for name in names if name.endswith((".weight", ".weight_orig"))):
        dtype, shape = tensors[name]
        stem = name[: -len(".weight")] if name.endswith(".weight") else name[: -len(".weight_orig")]
        #: 4bit **打包**（`w4a4`/`int4` ✓）：逻辑列是声明列的 **2 倍** ✓ ⇒ 后续每一处形状判据
        #: （convrot 整除 ✓ / `layout_of` ✓）都必须拿**逻辑**形状 ✓，否则 6144 被当 3072 ✓✗
        #: ⚠️ 这一段**必须在低精度筛之前** ✗✗：自称 int4 却存成 bf16 时，先被
        #: :func:`is_low_precision` 跳过 ⇒ 报告里连一句「它自相矛盾」都没有 ✓✗
        #: （2026-10-30 抓到：体检说 `supported=True`、装载照 bf16 原样装 ✓）。
        packed = stem in packed_stems
        if packed:
            bad = int4_shape_problem(stem, dtype, shape)
            if bad is not None:
                int4_problems.append(bad)
                unresolved.append(stem)
                continue
            shape = logical_weight_shape(shape, packed=True)
        if not is_low_precision(dtype):
            continue
        count += 1
        if any(other.startswith(stem) and other.endswith(_GROUP_SUFFIXES) for other in names):
            grouped.append(stem)
            continue
        found = _scale_for(stem, tensors)
        if found is None:
            #: fp8 无 scale ⇒ **直接 cast** ✓（见 :data:`CAST_ONLY_DTYPES` ✓ —— 与
            #: :func:`precheck_dequant` / :func:`dequantize_state` 同一条判据 ✓，
            #: 少这一支就会「体检说缺 scale ⇒ 不装 ✗、装载时却能 cast ✓」两处打架 ✓✗）
            if is_cast_only(dtype):
                paired += 1
                layouts["cast"] = layouts.get("cast", 0) + 1
                continue
            unpaired.append(stem)
            continue
        group = group_for(stem, convrot)
        if group is not None and _convrot_shape_problem(stem, shape, group) is not None:
            unresolved.append(stem)
            continue
        paired += 1
        layout = layout_of(shape, tensors[found[0]][1])
        if layout is None:
            unresolved.append(stem)
        else:
            #: `+int4pack` 让**体检结论自己说出**「还得多一步解包」✓ —— 布局名里没有这一步，
            #: 读报告的人会以为 `per-row` 就是全部工作量 ✓✗（2026-10-30 之前正是这样 ✓）
            base = f"{layout}+int4pack" if packed else layout
            named = _with_convrot(base, group)
            layouts[named] = layouts.get(named, 0) + 1
    #: 反过来的那一半 ✓（见 :func:`orphan_scales`）：**没人要的 scale** —— 融合 QKV 被拆成
    #: 分片 scale 时正是这种形态 ✓；它**不进** `supported` ✗（理由在那边写着 ✓）
    orphans = orphan_scales(tensors)
    #: ⚠️ 声明了 4bit 打包、但一份张量都对不上号 ⇒ **解包根本不会发生** ✓✗（层名前缀不一致的常见
    #: 后果 ✓：`model.diffusion_model.` 有没有带 ✓）—— 那就会退回「按 int8 乘」的静默错 ✓，必须报 ✓
    matched_stems = {name[: -len(".weight")] if name.endswith(".weight") else name[: -len(".weight_orig")]
                     for name in names if name.endswith((".weight", ".weight_orig"))}
    int4_problems.extend(
        f"`{stem}` 自称 4bit 打包（`w4a4`/`int4` ✓）但文件里没有同名的 `.weight`/`.weight_orig` ✗"
        f" ⇒ 解包不会发生，反量化会退回「按 int8 乘」（数量级/结构全错 ✗）"
        for stem in sorted(packed_stems - matched_stems))
    return {"weights": count, "paired": paired, "layouts": layouts,
            "unresolved": unresolved[:6], "unresolvedCount": len(unresolved),
            "unpaired": unpaired[:6], "unpairedCount": len(unpaired),
            "grouped": grouped[:6], "groupedCount": len(grouped),
            "orphanScales": orphans[:6], "orphanScaleCount": len(orphans),
            "int4Problems": int4_problems[:6], "int4ProblemCount": len(int4_problems),
            "supported": count > 0 and not (unresolved or unpaired or grouped or int4_problems)}


def _compose(weight: Any, scale: Any, *, inverse: bool) -> Any:
    """``w = q * s``（或 ``q / s`` ✓）—— 布局已由 :func:`layout_of` 定 ✓。

    ⚠️ 归一走 :func:`_squeeze_scale_shape` **同一处** ✓：``[N,1]`` 这种写法不 reshape 也**恰好**
    广播正确 ✓✗ —— 「恰好」是最危险的那类正确 ✓（换个形状就静默错 ✗）⇒ 显式归一 ✓。
    """
    torch = _torch()
    value = weight.to(torch.float32)
    factor = scale.to(torch.float32)
    rows = int(weight.shape[0])
    flat = _squeeze_scale_shape(scale.shape)
    if flat == (rows,):
        factor = factor.reshape(rows, 1)
    elif len(tuple(int(d) for d in weight.shape)) >= 2 and flat == (int(weight.shape[1]),):
        factor = factor.reshape(1, -1)
    return value / factor if inverse else value * factor


# ── convrot（卷积旋转量化 ✓）：Hadamard 反旋转 ───────────────────────────────────────────────
# ⚠️ 口径**逐字**取自权威参考实现 ✓（不是逆向猜的 ✗）：
#    `comfy_kitchen.tensor.int8_utils._build_hadamard` / `_rotate_weight`
#    （参考仓副本在 comfy-studio 仓的 `reference/comfy-kitchen/comfy_kitchen/tensor/int8_utils.py` ✓）。
#    存盘方向 `W_rot = (W 按 group 分块) @ H^T`；H 是 **4×4 REGULAR Hadamard 的 Kronecker 幂**，
#    该 4×4 阵**对称** ✓ ⇒ `H^T == H` 且 `H @ H == I` ✓ ⇒ **再乘一次 H^T 就是还原** ✓
#    （参考实现正是**复用** `_rotate_weight` 做反量化 ✓ —— 那看着像笔误，其实是对的 ✗）。
#    ⇒ 还原 = `((q*s) 按 group 分块) @ H^T` ✓。
_H4: tuple[tuple[int, ...], ...] = ((1, 1, 1, -1),
                                    (1, 1, -1, 1),
                                    (1, -1, 1, 1),
                                    (-1, 1, 1, 1))

#: Hadamard 只取决于 (size, dtype, device) ✓ 纯确定性 ✓ ⇒ 缓存（50 层重建 50 次纯属浪费 ✗）
_HADAMARD_CACHE: dict[tuple[int, str, str], Any] = {}


def build_hadamard(size: int, *, dtype: Any = None, device: Any = None) -> Any:
    """``size`` 阶**归一化** regular Hadamard ✓（``size`` 必须是 **4 的幂** ✗ 不是"2 的幂" ✓）。

    H3 用的 256 = 4^4 ✓（恰好也是 2 的幂 ✓）—— 但判据按参考实现的语义走（每次 `kron(h, h4)`
    是 ×4 ✓），所以 512 这种 2 的幂**不能**用它 ✗。
    """
    if size < 4 or size % 4 != 0:
        raise ValueError(f"convrot 的 group 必须是 4 的幂，收到 {size} ✗")
    power = int(round(math.log(size, 4)))
    if 4 ** power != size:
        raise ValueError(f"convrot 的 group 必须是 4 的幂（4^k），收到 {size} ✗")
    torch = _torch()
    dtype = dtype or torch.float32
    key = (int(size), str(dtype).replace("torch.", ""), str(device) if device is not None else "-")
    cached = _HADAMARD_CACHE.get(key)
    if cached is None:
        h = torch.tensor(_H4, dtype=torch.float32)
        while int(h.shape[0]) < int(size):
            h = torch.kron(h, torch.tensor(_H4, dtype=torch.float32))
        cached = (h / math.sqrt(size)).to(dtype)
        _HADAMARD_CACHE[key] = cached
    return cached if device is None else cached.to(device)


def rotate_weight(weight: Any, h: Any, group_size: int) -> Any:
    """沿**最后一维**按 ``group_size`` 分块后 ``@ H^T`` ✓（存盘与还原**共用本函数** ✓）。"""
    if int(weight.dim()) != 2:
        raise ValueError(f"convrot 只作用于 2D 权重，收到 {tuple(weight.shape)} ✗")
    out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
    if in_f % group_size != 0:
        raise ValueError(f"输入维 {in_f} 不整除 convrot group {group_size} ✗")
    grouped = weight.reshape(out_f, in_f // group_size, group_size)
    rotated = grouped @ h.transpose(-1, -2).to(weight.dtype)
    return rotated.reshape(out_f, in_f)


def dequantize_weight_rotated(weight: Any, scale: Any, *, dtype: Any = None,
                              name: str = "weight") -> Any:
    """把低精度权重还原成**旋转域**的权重 ✓ —— ``w_rot = (q * s)`` ✓，**不做反旋转** ✗。

    ## 为什么不在这里反旋转（2026-10-29 ✓）

    在**权重侧**反旋转的代价是「每个权重元素 × group 次乘加」✓✗：H3 主权重 19.3 G 个元素 ×
    256 ≈ **4.9 TMAC / 步** ✓✗ —— A5000 上光这一步就要 1~2 秒/步 ✗，整段采样直接不可用 ✗。

    convrot 的旋转是**输入维**上的正交变换 ✓ ⇒ 可以**整块搬到激活侧** ✓：

    ``F.linear(x, W) = F.linear(rotate(x), W_rot)`` ✓（``W_rot = W @ H`` ✓，``H`` 对称 ⇒ ``H^T == H`` ✓）

    激活只有 ``[S, in]`` ✓（S 是 token 数 ✓ 远小于 out ✓）⇒ 每层一次 ✓ 而不是每个权重一次 ✓。

    ⚠️ 两者**必须成对使用** ✗：单独用本函数（旋转域权重配未旋转激活）⇒ 数值全错但不报错 ✓✗，
    症状与「漏反旋转」一模一样（画面是噪声 ✓）。成对性由 :class:`h3_form.Int8Linear` 保证 ✓。
    """
    value = _compose(weight, scale, inverse=False)
    return value if dtype is None else value.to(dtype)


def rotate_activations(x: Any, group: int, *, h: Any = None) -> Any:
    """把 convrot 的旋转搬到激活侧 ✓：``x`` 沿**最后一维**按 ``group`` 分块后 ``@ H`` ✓。

    数学等价见 :func:`dequantize_weight_rotated` ✓。同 :func:`rotate_weight` 一样，H 对称 ⇒
    这里乘 ``H`` 与乘 ``H^T`` 是同一个矩阵 ✓（但代码仍显式写转置 ✓ —— 读的人不必自己推 ✓）。
    """
    torch = _torch()
    dims = tuple(int(d) for d in x.shape)
    if not dims:
        raise ValueError("convrot 激活旋转至少要 1 维 ✗")
    last = dims[-1]
    if last % group:
        raise ValueError(f"激活的最后一维 {last} 不整除 convrot group {group} ✗")
    matrix = h if h is not None else build_hadamard(group, dtype=x.dtype, device=x.device)
    grouped = x.reshape(*dims[:-1], last // group, group)
    rotated = grouped @ matrix.transpose(-1, -2).to(x.dtype)
    return rotated.reshape(dims)


def convrot_of_spec(spec: Any) -> tuple[int | None, str | None]:
    """从**单个** ``comfy_quant`` 说明里读出 convrot 分组 ✓（``(None, None)`` ⇒ 这条没说旋转 ✓）。

    ⚠️ 判据必须认**两种写法** ✗（2026-10-30 实测抓到的假绿）：
    ① H3 一族：``{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`` ✓；
    ② Krea2 一族：``{"format": "convrot_w4a4", "convrot_groupsize": 256, "quant_group_size": 64}``
       —— **没有** ``convrot`` 布尔 ✗。只认布尔 ⇒ 这一族被判成「不是 convrot」⇒ 装载时少一步
       反旋转 ⇒ 能装、能跑、画面全是噪声 ✗✗（正是本仓最怕的「像是成功」的失败 ✓）。
    """
    if not spec:
        return None, None
    marked = bool(spec.get("convrot")) or "convrot" in str(spec.get("format") or "").lower()
    if not marked:
        return None, None
    raw = spec.get("convrot_groupsize") or spec.get("convrot_group_size")
    try:
        group = int(raw)
    except (TypeError, ValueError):
        return None, f"这一层写了 convrot 但 group 读不成整数（收到 {raw!r} ✗）"
    if group <= 0 or group % 4 != 0:
        return None, f"convrot 分组 {group} 不是 4 的正整数倍 ✗ ⇒ 拒绝反量化"
    return group, None


def convrot_group_of(specs: Any) -> tuple[int | dict[str, int] | None, str | None]:
    """从文件自带的量化说明里判定「这份文件哪些层带 convrot、group 多大」✓。

    返回 ``(group, None)`` ✓（整文件同组）/ ``({层: group}, None)`` ✓（**逐层不同**）/
    ``(None, None)``（不是 convrot ✓）/ ``(None, 理由)`` ✗（说明自相矛盾 ⇒ 拒绝 ✓）。

    ``specs`` 收两种形状 ✓：``[{...}]``（去重后的族级说明 ✓）或 ``{层名: {...}}``（逐层 ✓）。
    逐层那路**必须**用它 ✗：实测 H3 **全量**版同一文件里 ``blocks.0.adaln_proj`` 是 group **64**、
    ``blocks.0.attn.qkv_proj`` 是 **256** ✓✗ —— 旧实现（整文件必须同组）因此把 31.7 GiB 的
    正品**整个拒了** ✓；而 Krea2 int4 那族连 ``convrot`` 布尔都没写 ⇒ 旧实现判成「不是 convrot」
    ⇒ 少一步反旋转却报「可自动还原」✓✗（假绿 ✓）。两条都是「判据按假设走、没按文件走」✓。
    """
    if isinstance(specs, MappingABC):
        layer_specs: Mapping[Any, Any] = specs
    else:
        layer_specs = {f"#{index}": spec for index, spec in enumerate(specs or ())}
    groups: dict[str, int] = {}
    for stem, spec in layer_specs.items():
        group, gap = convrot_of_spec(spec)
        if gap:
            return None, f"`{stem}` 的说明自相矛盾 ✗：{gap}"
        if group is not None:
            groups[str(stem)] = group
    if not groups:
        return None, None
    if len(set(groups.values())) == 1:
        return next(iter(groups.values())), None
    return dict(groups), None


def group_for(stem: str, convrot: Any) -> int | None:
    """这一层该用多大的 convrot 分组 ✓（``None`` ⇒ 这一层不旋转 ✓）。

    ``convrot`` 允许三种形状 ✓：``None`` / ``int``（整文件同组 ✓）/ ``{层: group}``（逐层 ✓）。
    ⚠️ 逐层表里**查不到**的层 ⇒ 不旋转 ✓（**不是**拿别层的 group 顶上 ✗ —— 那等于瞎转 ✓✗）。
    """
    if convrot is None:
        return None
    if isinstance(convrot, MappingABC):
        raw = convrot.get(stem)
        try:
            return None if raw is None else int(raw)
        except (TypeError, ValueError):
            return None
    try:
        return int(convrot)
    except (TypeError, ValueError):
        return None


def _with_convrot(layout: str, convrot: Any) -> str:
    """把布局名标成 ``per-row+convrot256`` ✓ —— 报告里**必须能看出**它比普通 per-row 多一步旋转 ✗。"""
    if convrot is None:
        return layout
    if isinstance(convrot, MappingABC):
        return f"{layout}+convrot{'/'.join(str(v) for v in sorted(set(convrot.values())))}"
    return f"{layout}+convrot{convrot}"


def _convrot_shape_problem(key: str, shape: Any, group: int) -> str | None:
    """convrot 只作用于 2D、且输入维必须整除 group ✓ ⇒ 不符就**拒绝**（不静默跳过旋转 ✗）。"""
    dims = tuple(int(d) for d in shape)
    if len(dims) != 2:
        return (f"`{key}` 的形状 {dims} 不是 2D ✗ ⇒ convrot 反旋转无从下手"
                f"（本仓不猜 ✗，拒绝反量化 ✓）")
    if dims[1] % group != 0:
        return (f"`{key}` 的形状 {dims} 的输入维 {dims[1]} 不整除 convrot group {group} ✗"
                f"⇒ 与文件声明矛盾 ✓（拒绝反量化 ✓）")
    return None


def _derotate(value: Any, group: int, key: str, problems: list[str]) -> Any | None:
    """convrot 反旋转：``((q*s) 按 group 分块) @ H^T`` ✓（``H`` 由分组确定性重建 ✓，无随机 ✗）。"""
    try:
        h = build_hadamard(group, device=getattr(value, "device", None),
                           dtype=getattr(value, "dtype", None))
        return rotate_weight(value, h, group)
    except (ValueError, RuntimeError) as exc:
        problems.append(f"`{key}` 的 convrot 反旋转失败：{exc}")
        return None


def _scale_for(stem: str, present: Mapping[str, Any]) -> tuple[str, bool] | None:
    """给 ``stem`` 找配套 scale ✓ ⇒ ``(名字, 是否取倒数 ✓)``；找不到 ⇒ ``None`` ✓。

    ⚠️ 顺序**必须**是「长后缀优先」✗（见 `SCALE_SUFFIXES` 的注释 ✓）。
    """
    for suffix in SCALE_SUFFIXES:
        name = f"{stem}{suffix}"
        if name in present:
            return name, suffix.endswith("_inv")
    return None


def scale_stem(name: Any) -> str | None:
    """``xxx.weight_scale`` ⇒ ``xxx`` ✓（不是 scale 名 ⇒ ``None`` ✓）。

    ⚠️ 遍历 :data:`SCALE_SUFFIXES` 的**原顺序** ✗（表本身就是「长后缀优先」✓）——
    自己另排一份顺序，``weight_scale_inv`` 就会被 ``weight_scale`` 前缀命中 ✓✗。
    """
    text = str(name)
    for suffix in SCALE_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return None


def orphan_scales(tensors: Mapping[str, Any]) -> list[str]:
    """**有 scale、却没有它服务的那个权重** 的键 ✓（``["x.weight_scale", …]``）。

    ## 为什么非要单列（2026-10-30 ✓）

    `plan_dequant` 原来只从**权重**那一侧找 scale ✓ —— 「缺 scale 的权重」会进 ``unpaired`` ✓，
    而「**没人要的 scale**」一个都不看 ✗。这不是学术问题：**融合 QKV** 型的检查点里，
    若量化器把 ``attn.qkv.weight``（融合 ✓）的 scale 拆成 ``attn.q.weight_scale`` /
    ``k`` / ``v`` 三份 ✓，那三份**全都**是孤儿 ✓，而体检照样报「全配上 ✓」✗ ——
    正是「静默装错」的温床 ✗（用户纪律：不静默跳过 ✓）。

    ## 为什么它**不**阻断 ``supported``

    真会装错的形态已经被 ``unpaired`` 拦住了 ✓：融合权重**自己**没有 scale ⇒ 缺配套 ⇒ 拒 ✓。
    孤儿 scale 不会让还原算错（本仓根本不用它 ✓），它只说明「这份文件里还有一段量化信息
    我们没理解」✓ —— 报出来供人一眼看见 ✓，但不许据此把能跑的模型拒掉 ✗
    （拒绝不是安全 ✗）。装载侧不用另写一份判据 ✓：这些键留在 state 里 ⇒ 模型里没有 ⇒
    自然落进 :class:`weights.WeightLoadReport.unexpected` ✓（两端口径一致 ✓）。

    ## ⚠️ 第一版被**真文件**当场打脸（2026-10-30 ✓）

    只按「有没有同名 `.weight`」判 ⇒ 盘上 `krea2_turbo_int8_convrot` 立刻报 **138 个**孤儿 ✓✗：
    `blocks.N.prenorm.scale` / `postnorm.scale` / `attn.qknorm.qnorm.scale` /
    `txtfusion.*.prenorm.scale` / `txtmlp.0.scale` —— 全是 **RMSNorm 的缩放参数** ✓，
    它们**自己就是权重** ✓、没有更低一层的 ``prenorm.weight`` ✓。
    ⇒ 后缀必须分两类 ✓（`:data:`WEIGHT_SCALE_SUFFIXES` 「名字自带 weight ⇒ 语义无歧义 ✓」与
    :data:`GENERIC_SCALE_SUFFIXES` 「通用名 ⇒ 可能是 norm 权重 ✓」✓）：**通用名**没有同名 weight
    时视为**独立权重** ✓ 不报；只有名字自带 `weight` 的才无条件算孤儿 ✓。
    ★ 这正是「判据必须被真文件打脸」的价值 ✓：小样本测试（`w.weight_scale` ＋ `attn.q.weight_scale`）
    两种口径都能过 ✓✗，只有真模型上的 norm 命名能暴露 ✓。
    """
    orphans: list[str] = []
    for name in sorted(str(item) for item in tensors):
        #: 激活侧 scale 先排除 ✓。**当前两张表其实不会相撞** ✓（`SCALE_SUFFIXES` 全都带点 ✓，
        #: 而 `.input_scale` 的末 6 个字符是 `_scale` ✓ ⇒ `endswith(".scale")` 是 **False** ✓）——
        #: ⚠️ 那正因如此，这条**不能**只凭"看着该排除"就留着 ✗：它防的是**以后**有人往
        #: `SCALE_SUFFIXES` 里加无点写法（`"_scale"` 一类 ✓）时把每个 fp8_scaled 层的激活侧
        #: scale 全报成孤儿 ✓✗。判据的可验性由 `tests/engine_quant_test.py` 的 ③ 用 monkeypatch
        #: 临时加 `"_scale"` 去锁 ✓（不改这张表就测不出来 ✓✗）。
        if is_input_scale(name):
            continue
        stem = scale_stem(name)
        if stem is None:
            continue
        has_weight = f"{stem}.weight" in tensors or f"{stem}.weight_orig" in tensors
        if has_weight:
            continue
        #: ⚠️ 通用名（裸 `.scale` ✓）**没有**同名 weight 时，它自己多半就是**权重** ✗✗
        #: （RMSNorm 的缩放参数 ✓）—— 实测 `krea2_turbo_int8_convrot` 有 **138 个**
        #: `prenorm.scale` / `qknorm.*.scale` / `txtmlp.0.scale` ✓，第一版判据把它们全报成孤儿 ✓✗。
        #: 而名字自带 `weight` 的（`attn.q.weight_scale` ✓）没有对应 weight ⇒ **就是**孤儿 ✓
        #: （融合 QKV 把三份 scale 留下、weight 融成一份 ⇒ 正是要抓的形态 ✓）。
        if str(name).endswith(GENERIC_SCALE_SUFFIXES) and not str(name).endswith(
                WEIGHT_SCALE_SUFFIXES):
            continue
        orphans.append(name)
    return orphans


def _layout_problem(name: str, weight_shape: Any, scale_shape: Any) -> str:
    """布局判不出来时**唯一的那句话** ✓ —— 预检与装载必须说出**同一句** ✗（否则人会对不上账 ✓✗）。"""
    return (f"`{name}` 的 scale 形状 {tuple(int(d) for d in scale_shape)} 与权重 "
            f"{tuple(int(d) for d in weight_shape)} **对不上任何一种布局** ✗"
            f"（per-tensor / per-row / per-col / 逐元素 ✓）⇒ 大概是**块量化** ✓"
            f"（group size 在随附 json 里 ✓ 本仓**不猜** ✗）⇒ 拒绝反量化 ✓")


def _zero_scale_problem(name: str) -> str:
    # ⚠️ 除以 0 ⇒ inf ✗ ⇒ 提前判（0 是**坏 scale** ✓，不是「合法的 1」✗）
    return f"`{name}` 的 scale 里有 0 ✗ ⇒ 反量化会出 inf ✓（不装 ✗）"


def _apply_scale(weight: Any, scale: Any, *, inverse: bool, name: str,
                 problems: list[str]) -> tuple[Any, str] | None:
    """反量化一个权重 ✓ ⇒ ``(张量, 布局)``；**判不出来 ⇒ 记问题 + ``None``** ✓。

    ⚠️ 布局判定走 :func:`layout_of` **同一处** ✗（体检阶段也用它 ✓ —— 两处各写一份必然会漂 ✓✗）。
    在 **float32** 里算 ✓（scale 往往是小数量级 ✓，低精度里算会掉精度 ✓）。
    """
    torch = _torch()
    layout = layout_of(weight.shape, scale.shape)
    if layout is None:
        problems.append(_layout_problem(name, weight.shape, scale.shape))
        return None
    if inverse and bool((scale.to(torch.float32) == 0).any()):
        problems.append(_zero_scale_problem(name))
        return None
    return _compose(weight, scale, inverse=inverse), layout


def precheck_dequant(meta: Mapping[str, tuple[Any, Any]],
                     scales: Mapping[str, Any], *,
                     convrot: Any = None,
                     int4: Any = None) -> DequantResult:
    """**不读大权重**就能判「这份检查点装不装得上」✓（读头部 + 只读 scale 小张量 ✓）。

    ``meta``：``{名字: (dtype, 形状)}`` ✓ —— dtype 可以是 torch dtype ✓ 也可以是
    safetensors 头部的字符串 ✓（:func:`is_low_precision` 两套都认 ✓）。
    ``scales``：``{名字: 张量}`` ✓ —— **只需要 scale 伙伴** ✓（它们每个只有几十字节到几 KB ✓）。

    ## 为什么需要它（流式装载的前提 ✗）

    整份装载失败时可以「什么都没装 ✓」；但**逐张量**装载一旦在第 30 层才发现
    「scale 里有 0」✗，前 29 层已经写进模型了 ✓✗ ⇒ 半装状态最难查 ✓。
    ⇒ 结论是：**必须在写第一个字节之前**把所有「判不出来」的理由收集完 ✓，
    而这只需要「形状 + dtype + scale 的值」✓ —— 权重本身一个字节都不用读 ✓。

    ## 判据只许有一处

    这里用的 :func:`layout_of` / :func:`_scale_for` / ``_GROUP_SUFFIXES`` /
    两句错误文案 ✓ 与 :func:`dequantize_state` **完全相同** ✗ ——
    否则会出现「预检说能装 ✓、装载时被拒 ✗」（或更糟：预检说不行而实际能装 ✓✗）。
    """
    result = DequantResult()
    names = list(meta)
    packed_stems = frozenset(int4 or ())
    for name in sorted(names):
        if not name.endswith((".weight", ".weight_orig")):
            continue
        dtype, shape = meta[name]
        stem = name[: -len(".weight")] if name.endswith(".weight") else name[: -len(".weight_orig")]
        #: 4bit 打包 ⇒ **逻辑**形状（列 ×2 ✓）—— 见 :func:`unpack_int4`：布局与 convrot 整除
        #: 都要按逻辑维判 ✓，否则 6144 被当 3072 ✓✗
        #: ⚠️ 与 :func:`plan_dequant` 同一处口径：**先判自述再筛低精度** ✗（不然自称 int4
        #: 却存成 bf16 的层会被静默跳过 ✓✗）
        packed = stem in packed_stems
        if packed:
            bad = int4_shape_problem(stem, dtype, shape)
            if bad is not None:
                result.problems.append(bad)
                continue
            shape = logical_weight_shape(shape, packed=True)
        if not is_low_precision(dtype):
            continue
        group = [other for other in names
                 if other.startswith(stem) and other.endswith(_GROUP_SUFFIXES)]
        if group:
            result.problems.append(
                f"`{stem}` 带 {group[:2]} ✗ ⇒ 这是**分组量化**（int4 一类 ✓）⇒ 组大小在随附 json 里 ✓ "
                f"本仓**不猜** ✗（拒绝反量化 ✓ 不装 ✓）")
            continue
        found = _scale_for(stem, meta)
        if found is None and is_cast_only(dtype):
            #: fp8 没有 scale ⇒ **直接 cast** 就是无损解码 ✓（见 :data:`CAST_ONLY_DTYPES`）——
            #: 单独列 ``cast`` 布局 ✗：不许混进 ``per-tensor`` ✓✗（那会让人以为「有个 1.0 的 scale」✓）
            result.converted[name] = {"layout": "cast", "scale": None, "inverse": "False",
                                      "from": str(dtype)}
            continue
        if found is None:
            result.problems.append(
                f"`{name}` 是低精度（{dtype} ✓）但**找不到配套 scale** ✗"
                f"（找过 {[f'{stem}{suffix}' for suffix in SCALE_SUFFIXES][:3]} … ✓）⇒ 不装 ✓")
            continue
        scale_name, inverse = found
        scale = scales.get(scale_name)
        if scale is None:
            result.problems.append(
                f"`{name}` 的配套 scale `{scale_name}` **没有读到值** ✗ ⇒ 判不了布局 ✓（不装 ✓）")
            continue
        scale_shape = getattr(scale, "shape", ())
        group = group_for(stem, convrot)
        if group is not None:
            bad = _convrot_shape_problem(name, shape, group)
            if bad is not None:
                result.problems.append(bad)
                continue
        layout = layout_of(shape, scale_shape)
        if layout is None:
            result.problems.append(_layout_problem(scale_name, shape, scale_shape))
            continue
        if inverse:
            torch = _torch()
            if bool((scale.to(torch.float32) == 0).any()):
                result.problems.append(_zero_scale_problem(scale_name))
                continue
        base = f"{layout}+int4pack" if packed else layout
        result.converted[name] = {"layout": _with_convrot(base, group), "scale": scale_name,
                                  "inverse": str(inverse), "from": str(dtype)}
        result.companions.append(scale_name)
    _note_fp8_aux(result, meta)
    return result


def _note_fp8_aux(result: DequantResult, present: Mapping[str, Any]) -> None:
    """把 fp8_scaled 的**激活侧** scale ＋ 整份标记记成「吃掉的配套」✓（不装进模型 ✓ 也不报 unexpected ✓）。

    ⚠️ 只对**真有低精度权重的层**这么干 ✗：一个普通的 ``x.scale_input`` 也许是别的用途 ✓，
    没有低精度兄弟就**不动它** ✓（宁可留个 unexpected 让人看一眼 ✓，也不悄悄吞掉 ✓）。
    """
    for name in result.converted:
        stem = (name[: -len(".weight")] if name.endswith(".weight")
                else name[: -len(".weight_orig")] if name.endswith(".weight_orig") else name)
        for suffix in INPUT_SCALE_SUFFIXES:
            companion = f"{stem}{suffix}"
            if companion in present and companion not in result.companions:
                result.companions.append(companion)
    if SCALED_FP8_MARKER in present and SCALED_FP8_MARKER not in result.companions:
        result.companions.append(SCALED_FP8_MARKER)


def dequantize_state(state: Mapping[str, Any], *, target_dtype: Any = None,
                     convrot: Any = None, int4: Any = None) -> DequantResult:
    """把一份 ``state_dict`` 里的低精度权重反量化 ✓（**纯函数** ✓ 不改入参 ✓）。

    ⚠️ 只在**真需要**时才动 ✗：整份 bf16 的权重**原样返回** ✓（多一步 cast 只会白花时间 ✓）。

    ``convrot``：整份文件的旋转分组 ✓ ⇒ 每个低精度权重在 ``q*s`` **之后**再做一次分组 ``@ H^T`` ✓
    （顺序不能反 ✗：旋转是在**量化前**施加在 W 上的 ✓，所以还原也在 dequant 之后 ✓）。

    ⚠️ **nvfp4 不走这里** ✗：它的 ``weight`` 是 U8 打包 4 bit、块 scale 是 swizzled 布局 ✓，
    :func:`layout_of` 那套「形状能自证」的判据认不出它 ✓✗ ⇒ 拿这份 state 进来只会得到
    「判不出布局」✗。走 :func:`quant_nvfp4.dequantize_state` ✓（那边按 :func:`quant_nvfp4.plan`
    的逐层体检结果还原 ✓）。
    """
    result = DequantResult()
    converted = dict(state)
    packed_stems = frozenset(int4 or ())
    for key, tensor in list(state.items()):
        if not key.endswith((".weight", ".weight_orig")):
            continue
        stem = key[: -len(".weight")] if key.endswith(".weight") else key[: -len(".weight_orig")]
        #: 4bit 打包自述 ⇒ **先验**「它到底存得下打包吗」✗ —— 必须在低精度筛之前 ✓
        #: （自称 int4 却存成 bf16 的层，若先被跳过就什么都不会报 ✓✗；见 :func:`plan_dequant`）
        packed = stem in packed_stems
        if packed:
            bad = int4_shape_problem(stem, getattr(tensor, "dtype", None),
                                     getattr(tensor, "shape", ()))
            if bad is not None:
                result.problems.append(bad)
                continue
        if not is_low_precision(getattr(tensor, "dtype", None)):
            continue
        group = [name for name in state if name.startswith(stem) and name.endswith(_GROUP_SUFFIXES)]
        if group:
            result.problems.append(
                f"`{stem}` 带 {group[:2]} ✗ ⇒ 这是**分组量化**（int4 一类 ✓）⇒ 组大小在随附 json 里 ✓ "
                f"本仓**不猜** ✗（拒绝反量化 ✓ 不装 ✓）")
            continue
        #: 4bit 打包 ⇒ 先**解包**再乘 scale ✓（顺序不能反 ✗：scale 是给**逻辑**列上的每个值配的 ✓，
        #: 打包字节是两个值挤在一起 ✓ —— 先乘 scale 等于给两个值乘一次 ✓✗）
        source_dtype = str(getattr(tensor, "dtype", "?"))
        if packed:
            tensor = unpack_int4(tensor)
            result.int4PackCount += 1
        found = _scale_for(stem, state)
        if found is None and is_cast_only(getattr(tensor, "dtype", None)):
            #: fp8 没有 scale ⇒ **直接 cast** ✓（与 :func:`precheck_dequant` 同一条判据 ✓）
            value = tensor.to(_torch().float32)
            if target_dtype is not None:
                value = value.to(target_dtype)
            converted[key] = value
            result.converted[key] = {"layout": "cast", "scale": None, "inverse": "False",
                                     "from": str(getattr(tensor, "dtype", "?"))}
            continue
        if found is None:
            result.problems.append(
                f"`{key}` 是低精度（{getattr(tensor, 'dtype', '?')} ✓）但**找不到配套 scale** ✗"
                f"（找过 {[f'{stem}{suffix}' for suffix in SCALE_SUFFIXES][:3]} … ✓）⇒ 不装 ✓")
            continue
        group = group_for(stem, convrot)
        if group is not None:
            bad = _convrot_shape_problem(key, getattr(tensor, "shape", ()), group)
            if bad is not None:
                result.problems.append(bad)
                continue
        scale_name, inverse = found
        scaled = _apply_scale(tensor, state[scale_name], inverse=inverse, name=scale_name,
                              problems=result.problems)
        if scaled is None:
            continue
        value, layout = scaled
        if packed:
            layout = f"{layout}+int4pack"
        if group is not None:
            rotated = _derotate(value, group, key, result.problems)
            if rotated is None:
                continue
            value = rotated
            layout = _with_convrot(layout, group)
        if target_dtype is not None:
            value = value.to(target_dtype)
        converted[key] = value
        result.converted[key] = {"layout": layout, "scale": scale_name,
                                 "inverse": str(inverse), "from": source_dtype}
        result.companions.append(scale_name)
    _note_fp8_aux(result, state)
    # 配套张量**不进模型** ✓（模型里没有 `.scale` 这类键 ✓ ⇒ 留着只会变成 unexpected ✓）
    # ⚠️ 这里连**激活侧** scale 与 `scaled_fp8` 标记一起 pop ✓：它们不是权重 ✗，
    #    留着只是把「真·多出来的键」淹掉 ✓✗（同 `.comfy_quant` 的处理纪律 ✓）
    for name in set(result.companions):
        converted.pop(name, None)
    result.tensors = converted
    return result


def quant_report(state: Mapping[str, Any]) -> dict[str, Any]:
    """**只看不装** ✓：这份 state 里有没有低精度权重 / 有没有配套 ✓（体检报告用 ✓）。"""
    low = sorted(key for key, tensor in state.items()
                 if key.endswith((".weight", ".weight_orig"))
                 and is_low_precision(getattr(tensor, "dtype", None)))
    stems = [key[: -len(".weight")] if key.endswith(".weight") else key[: -len(".weight_orig")]
             for key in low]
    paired = [stem for stem in stems if _scale_for(stem, state) is not None]
    return {"lowPrecision": len(low), "withScale": len(paired),
            "unpaired": [stem for stem in stems if _scale_for(stem, state) is None][:6],
            "dtypes": sorted({str(getattr(state[key], "dtype", "?")).replace("torch.", "")
                              for key in low})}


def _torch() -> Any:
    import torch  # noqa: PLC0415 —— 只有真要算时才 import ✓

    return torch


# ── 未实现的布局族：**认得出来就点名拒绝** ✗（别只说「判不出布局」✗）────────────────────────
# ⚠️ 2026-09-24 补 ✓（口径来自逆向 ✓）：H3 生态里确实存在 ``*_int8_convrot.safetensors`` 这一族
#    （卷积旋转量化 ✓ —— 模型文件名实锤 ✓）。本仓 :func:`layout_of` 只认「**形状能自证**」的几种布局 ✓
#    ⇒ 撞上它只会得到一句「判不出布局」✗ —— 那句话**指不到真因** ✗，还会让人以为「文件坏了 / 多试几次」✗。
#    ⇒ 补一层**具名识别** ✓：认得出来就说清「是什么 + 为什么不做」✓。
#: 已知但**未实现**的布局族 ✓（token → 为什么不做 ✓）。
#: ⚠️ 判据**只有名字** ✗：这些族在形状上自证不了自己 ✓（与 :func:`layout_of` 能自证的那几种不同 ✓）
#: ⇒ 调用方要**先**问本函数 ✓，再把 :func:`layout_of` 的失败当兜底 ✓。
UNSUPPORTED_FAMILIES: dict[str, str] = {
    # ⚠️ 2026-10-29 **convrot 已实现** ✓ ⇒ 本表**不再有** `convrot` 键 ✗（下面保留事实链作依据 ✓）：
    #    盘上 `*_int8_convrot.safetensors` 每个低精度权重都带 `*.comfy_quant`（55~72 字节 JSON ✓），
    #    内容是 `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}` ✓
    #    —— 旋转参数（分组大小 ✓）**就在文件里** ✓。且该旋转是**确定性的** ✓
    #    （`comfy_kitchen.tensor.int8_utils._build_hadamard` 用固定 4×4 REGULAR Hadamard 做
    #    Kronecker 幂再归一 ✓，**没有随机种子** ✗；`W_rot = (W 按 group 分块) @ H^T` ✓）。
    #    ⇒ 本仓已按该参考实现落地 :func:`build_hadamard` / :func:`rotate_weight` ✓，
    #    分组判据统一收在 :func:`convrot_group_of` ✓（⚠️ 2026-10-30 改判：**不再**要求
    #    「整文件同组」✗ —— 实测 H3 全量版同一文件里 64 与 256 并存 ✓，族级假设会把正品拒掉
    #    ⇒ 现在逐层不同就返回 ``{层: group}`` ✓，按 :func:`group_for` 逐层取 ✓）。
    #    ⚠️ 删键还有一个**必要条件** ✓：:func:`unsupported_family` 靠**文件名子串**匹配
    #    ``convrot`` 兜底 ✗ —— 不删的话，文件名叫 `*_int8_convrot` 仍会被它拦下 ✗。
}

#: 已知但**未实现**的量化格式 ✓（`comfy_quant` 的 ``format`` → 为什么不做 ✓）。
#: ⚠️ 与 :data:`UNSUPPORTED_FAMILIES` 分开：那一族是「**形状/命名**认得出来」✓（convrot ✓），
#: 这一族是「**文件自带的 ``comfy_quant`` 明写了 format**」✓ —— 后者比任何猜测都硬 ✓。
UNSUPPORTED_FORMATS: dict[str, str] = {
    "mxfp4": "MXFP4（4 bit + E8M0 共享指数 ✓）—— 共享指数是 **E8M0**（不是 nvfp4 的 e4m3 块 scale ✗）"
             "⇒ 与 nvfp4 的解码表/布局都不同 ✓，本仓**没有这一族的解码表** ⇒ 不猜 ✗",
}

#: 已**实现**的量化格式 ✓（`comfy_quant` 的 ``format`` → 由谁负责还原 ✓）。
#: ⚠️ 2026-10-30 之前 `nvfp4` 在 :data:`UNSUPPORTED_FORMATS` 里 ✓；现在移出来是因为
#: :mod:`quant_nvfp4` 已按参考实现落地（E2M1 值表 ✓ / 高半字节在前 ✓ / cuBLAS 128×4 tiled
#: 块 scale ✓ / f32 二级 scale ✓ / AWQ 折列 ✓），并有逐位对拍判据 ✓✗ ——
#: **口径要跟着实现走** ✓：实现到位了还报「未实现」，等于把能跑的拦下来 ✓✗。
SUPPORTED_FORMATS: dict[str, str] = {
    "nvfp4": "NVFP4（4 bit E2M1 打包 + e4m3 块 scale[每 16 元素] + f32 二级 scale ✓）"
             "⇒ 由 :mod:`quant_nvfp4` 逐层还原 ✓（`w = fp4 × (块 scale × 二级 scale)` ✓）",
    # ⚠️ 2026-10-30 **两轮实测，前一版结论是错的** ✗✗：先记成「这族的 weight **不是** 4 bit 打包」
    #    （凭 `blocks.0.attn.wk.weight` 的 dtype=int8 / min=-112 / max=127 / 222 个不同值 ✓）——
    #    那些数只能说明「字节是 int8 容器」✗，**说明不了字节里装的是 8 bit 还是两个 4 bit** ✓✗
    #    （打包后的 `(6144, 3072)` 与普通 per-row int8 **逐字相同** ✓）。真证词是同源 int8 版：
    #    按「低半字节=偶数列、nibble ≥ 8 减 16、有符号」解包再乘 scale、再反旋转 ⇒ 与
    #    `krea2_v1.5_anime_int8_convrot` 余弦 **0.9878** ✓（换 nibble 顺序 ⇒ 0.0001 ✓）⇒
    #    这族**必须**解包 ✓；不解包就是把两个值挤成一团再乘一次 scale ✓✗。
    "convrot_w4a4": "convrot_w4a4（4 bit 训练的**旋转**量化 ✓）—— 落盘 weight 是**每字节两个有符号 "
                    "4 bit** ✓ ＋ per-row scale ✓ ⇒ 由 :func:`dequantize_state` **先解包**"
                    "（:func:`unpack_int4` ✓）**再**乘 scale、**再** convrot 反旋转 ✓",
    # ⚠️ 2026-10-30 实测这一族的**两种写法都有** ✓✗：
    #    ① `qwen3vl_4b_fp8_scaled` 带 `*.comfy_quant`，写 `{"format": "float8_e4m3fn"}` ✓；
    #    ② `wan2.2_i2v_high_noise_14B_fp8_scaled` / `umt5_xxl_fp8_e4m3fn_scaled` **一个都没有** ✗ ⇒
    #       那条路只能靠「fp8 dtype + `.scale_weight` 标量」形状自证（per-tensor ✓）走
    #       :func:`layout_of` ✓ —— 所以本表只是**把族名说清楚** ✓，不是放行条件 ✗。
    "float8_e4m3fn": "fp8 E4M3（per-tensor `scale_weight` 标量 ✓）⇒ 由 :func:`dequantize_state` 还原 ✓"
                     "（`scale_input` 是激活侧 ✓ 丢掉 ✓，见 `INPUT_SCALE_SUFFIXES`）",
    "float8_e5m2": "fp8 E5M2（同上 ✓）⇒ 由 :func:`dequantize_state` 还原 ✓",
}

#: `comfy_quant` 里 ``format`` 的**拼写名 → 本仓族名** ✓（`quantScheme` 一栏的**单一真源** ✓）。
#:
#: ⚠️ 为什么必须有这张表（2026-10-30 实测抓到的**口径分裂** ✓✗）：同一族 fp8 文件里，
#:    带 `*.comfy_quant` 的（`qwen3vl_4b_fp8_scaled` ✓）走 :func:`supported_format`，
#:    不带的一个都没有的（`wan2.2_i2v_high_precision_14B_fp8_scaled` / `umt5_xxl_fp8_e4m3fn_scaled` ✓）
#:    只能走 dtype 兜底 ✓ ⇒ 若把拼写名当族名，同一族的 `quantScheme` 会一半叫 `fp8` ✓、
#:    一半叫 `float8_e4m3fn` ✓✗ —— 下游任何人按 `quantScheme == "fp8"` 筛都会漏掉一半 ✓✗。
#:    ⇒ `format` 说的是**拼写** ✓（写进 `computeDtype` 的说明里 ✓），族名由本表归一 ✓。
FORMAT_ALIASES: dict[str, str] = {
    "float8_e4m3fn": "fp8",
    "float8_e5m2": "fp8",
    "fp8_e4m3fn": "fp8",
    "fp8_e5m2": "fp8",
}


def format_alias(fmt: Any) -> str:
    """`comfy_quant` 的 ``format`` ⇒ **族名** ✓（没有别名就原样小写返回 ✓）。"""
    token = str(fmt or "").strip().lower()
    return FORMAT_ALIASES.get(token, token)

#: 量化元数据的张量名后缀 ✓（ComfyUI 打包格式 ✓：一个 JSON 小张量紧挨着每个低精度权重 ✓）
QUANT_META_SUFFIX = ".comfy_quant"


def parse_format_spec(raw: Any) -> dict[str, Any] | None:
    """解析 ``*.comfy_quant`` 的原始字节 ⇒ ``{"format": …, "convrot": …, …}`` ✓；解析不了 ⇒ ``None`` ✓。

    ⚠️ 为什么必须读它（2026-10-29 实测 ✓）：这是**文件自己写的量化说明** ✓ ——
    比按形状猜、按文件名猜都硬 ✓。实测内容：``{"format": "int8_tensorwise", "convrot": true,
    "convrot_groupsize": 256}`` 与 ``{"format": "nvfp4", "full_precision_matrix_mult": true}`` ✓。
    ⚠️ 此前本仓**从没读过它** ✗ ⇒ 于是把「本仓没实现这一族」错报成「缺随附 json 的 group size」✗
    ⇒ 而那个 json **根本不存在**（参数就在权重文件里 ✓）⇒ 用户会一直找不到 ✗。
    """
    if isinstance(raw, (bytes, bytearray)):
        text = bytes(raw).decode("utf-8", "replace").strip().strip("\x00")
    else:
        text = str(raw or "").strip()
    if not text.startswith("{"):
        return None
    import json  # noqa: PLC0415 —— 只在真要解析时用 ✓

    try:
        spec = json.loads(text)
    except ValueError:
        return None
    if not isinstance(spec, dict):
        return None
    return spec


#: **文件级**量化说明写在 metadata 里的那个键 ✓（StarNodes/转换器那一族 ✓ —— 不是 `*.comfy_quant` ✗）
METADATA_QUANT_KEY = "_quantization_metadata"


def specs_from_metadata(metadata: Any) -> dict[str, dict[str, Any]]:
    """解析**文件级**量化说明 ⇒ ``{层名: 说明}`` ✓（读不到 ⇒ ``{}`` ✓ 不抛错 ✓）。

    实测形状（2026-10-30 · `Krea2-Moody-Mix-premium_int4_convrot.safetensors` ✓）::

        metadata["_quantization_metadata"] = {
          "format": "int4_convrot",
          "layers": {"model.diffusion_model.blocks.0.attn.wk":
                       {"format": "convrot_w4a4", "convrot_groupsize": 256,
                        "quant_group_size": 64},
                     ... 256 层 ...}}

    ⚠️ 这一路**必须**读 ✗：那个文件里**一个 `*.comfy_quant` 都没有** ✓✗ ⇒ 只看张量名就等于
    「完全没看见旋转说明」⇒ 按普通 per-row 解 ⇒ **能装、能跑、画面全错**却报「可自动还原」
    ✓✗（假绿 ✓）。`quant.supported` 对此**不设防**：形状 `(1536, 3072)` + scale `(1536,)`
    与普通 per-row 逐字相同，`layout_of` 无处可疑 ✓✗。
    """
    if not isinstance(metadata, MappingABC):
        return {}
    raw = metadata.get(METADATA_QUANT_KEY)
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", "replace")
    if isinstance(raw, str):
        import json  # noqa: PLC0415 —— 只在真要解析时用 ✓

        try:
            parsed: Any = json.loads(raw.strip().strip("\x00") or "null")
        except ValueError:
            return {}
    elif isinstance(raw, MappingABC):
        parsed = raw
    else:
        return {}
    layers = parsed.get("layers") if isinstance(parsed, MappingABC) else None
    if not isinstance(layers, MappingABC):
        return {}
    return {str(stem): dict(spec) for stem, spec in layers.items()
            if isinstance(spec, MappingABC)}


def format_family(spec: Mapping[str, Any] | None) -> tuple[str, str] | None:
    """``comfy_quant`` 说得清「这一族本仓做不了」⇒ 返回 ``(族名, 理由)`` ✓；否则 ``None`` ✓。

    ⚠️ **2026-10-29 起 convrot 不在这里判** ✗：它已由 :func:`convrot_group_of` 单独取 group ✓、
    再由 :func:`dequantize_state` 做反旋转 ✓ ⇒ 本函数只管「``format`` 明写着本仓做不了的那几种」✓。
    （``convrot`` 权重常写 ``format: int8_tensorwise`` ✓ —— 那**不是**漏判 ✗，是有意放行 ✓。）
    """
    if not spec:
        return None
    fmt = str(spec.get("format") or "").strip().lower()
    for token, why in UNSUPPORTED_FORMATS.items():
        if token in fmt:
            return token, f"`comfy_quant` 明写 `format={fmt}` ✗：{why}"
    return None


def supported_format(spec: Mapping[str, Any] | None) -> tuple[str, str] | None:
    """``comfy_quant`` 说得出「这一族本仓**做得了**」⇒ 返回 ``(族名, 负责模块说明)`` ✓；否则 ``None`` ✓。

    ⚠️ 与 :func:`format_family` **成对** ✓：先问「能不能做」（本函数 ✓），问不出来再问「为什么不能」
    （:func:`format_family` ✓）。顺序反了会让**能做的族**拿到一句「未实现」✗ ——
    2026-10-30 之前 `nvfp4` 就是这种状态 ✓✗（`quant_nvfp4` 已落地，报告还在说「没解码表」✗）。

    ⚠️ 这里**只认名字** ✗，不判这份文件是否真的合规 ✓ —— 逐层的形状等式在
    :func:`quant_nvfp4.plan` 里 ✓（那边才有 ``weight_scale`` / ``weight_scale_2`` 的形状 ✓）。
    """
    if not spec:
        return None
    fmt = str(spec.get("format") or "").strip().lower()
    for token, why in SUPPORTED_FORMATS.items():
        if token in fmt:
            #: ⚠️ 返回的族名要**归一** ✗✗：`float8_e4m3fn` 与 `fp8` 是**同一族** ✓，
            #: 直接把拼写名当 `quantScheme` ⇒ 同一族两种标签 ✓✗（见 :data:`FORMAT_ALIASES` ✓）
            return format_alias(fmt), why
    return None


def unsupported_family(*, source: str = "", tensor_names: Any = ()) -> str | None:
    """名字里带**已知但未实现**的布局族 ⇒ 返回可行动的拒绝理由 ✓；否则 ``None`` ✓。

    ``source`` 一般是权重**文件名** ✓；``tensor_names`` 是张量名（可迭代 ✓）。只做**子串匹配** ✓
    （大小写不敏感 ✓）—— ⚠️ 这不是「布局判定」✗，是「**这一族我们不支持**」的具名提示 ✓。
    """
    haystack = " ".join([str(source or "")] + [str(name) for name in tensor_names]).lower()
    for token, why in UNSUPPORTED_FAMILIES.items():
        if token in haystack:
            return (f"这个权重属于**未实现的布局族**「{token}」✗：{why}。"
                    f"⚠️ 这**不是**「判不出布局」✗ —— 不是文件坏了 ✓，是本仓**没实现**这一族 ✓；"
                    f"要么换成非 {token} 的量化产物 ✓，要么先把该族需要的参数（如旋转参数 ✓）拿到再实现 ✓。")
    return None
