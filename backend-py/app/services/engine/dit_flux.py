"""Flux.1 **双流 + 单流** MMDiT 前向（族 ``flux`` ✓）。

权威口径：ComfyUI ``comfy/ldm/flux/model.py`` / ``layers.py`` / ``math.py`` ✓
（逐行核过 ✓；盘上 ``flux1-dev.safetensors`` 780 键 = 19 double × 24 + 38 single × 8 + 顶层 20
  = 456 + 304 + 20 ✓ —— 早先这里写「+ 14」，那是**算术不成立**的笔误 ✗：14 项加起来只有 774）。

## 与已接四族都不一样、且「形状对、能跑、图不对」的七处

① **两段结构**：前 19 层**双流**（img/txt 各有 qkv/proj/norm/mlp/mod，只在注意力处拼），
   后 38 层**单流**（``cat(txt, img)`` 共用块，跑完按 ``[:, txt_len:]`` 把文本段切掉）。
② **拼接顺序 = ``(txt, img)``**（不是 img 在前）：联合注意力拼 q/k/v 与单流入口都是 txt 在前。
   拼反了**一步不报错**，图像 token 用文本参数去算，出来「构图对、细节糊」。
③ **调制 6 组 / 3 组**，每组顺序 **(shift, scale, gate)**，应用式 ``x·(1+scale)+shift``，
   残差门 ``gate·f(x)``。写成 (scale, shift) 不报错，亮度随步漂。
④ **时间口径 = ``1000·σ``**（``time_factor=1000``）—— 与 Z-Image 的 ``1000·(1−σ)`` **相反**。
   抄错 ⇒ 去噪方向倒。``guidance`` 也吃同一套正弦、但走**另一根 256 维**输入。
⑤ **一根 vec 由三路相加**：``time_in(σ) + guidance_in(g) + vector_in(y)``（y = CLIP-L pooled 768，
   只取前 ``vec_dim``）。漏 vector_in ⇒ 提示词语义整段缺席；漏 guidance_in ⇒ 蒸馏版丢了 CFG 那一根。
⑥ **位置号不居中**：``t=0``、``h=0…H−1``、``w=0…W−1``（Qwen-Image 那族是**居中**，差一个 −H//2）。
   文本位置三轴全 0 ⇒ 旋转为单位阵（所以「文本不转」是天然结果，不是特例）。
⑦ **``img_in`` 吃拍平后的 patch**：``[3072, 64]`` 的 64 = 16ch×2×2；``final_layer.linear`` 出 64
   且**不再**展开 patch ⇒ unpatchify 只有一次。

## 本文件**不**负责（如实标，不静默降级）

* **文本塔**：需要 T5-XXL（4096）+ CLIP-L（pooled 768）。盘上两座塔都没有 ⇒ 条件由外部给
  （``context`` / ``vector``），本文件只做主干。
* **guidance 数值**：权重只有 ``guidance_in`` 这块投影，数值来自推理配置 ⇒ 默认
  :data:`GUIDANCE_DEFAULT`，文档明写「默认值，不是权重事实」。
* **带 ``txt_norm`` 的变体**：ComfyUI 的 Flux 参数表里有个 ``txt_norm`` 开关（``txt_in`` 前多一层
  RMSNorm）。``flux1-dev`` 没有这层（780 键里不含 ``txt_norm.scale``）⇒ 本骨架不建它；若换成带这层的
  衍生权重，装载会把该键报成 ``unexpected``（**诚实拒绝**，不静默丢键 ✗）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import sdpa
from .dit_generic import ShapeError, apply_rope, rope_freqs, timestep_embedding

__all__ = [
    "PATCH", "AXES_DIM", "ROPE_THETA", "T_FREQ", "T_SCALE", "NORM_EPS",
    "GUIDANCE_DEFAULT", "FluxShape", "default_axes", "infer_shape", "expected_keys",
    "build", "build_from_shapes", "FluxDiT",
]

#: 潜变量侧 patch 边长（权重里没这个键，但 ``img_in`` 入宽 64 = 16ch×2×2 把它钉死了）。
PATCH = 2
#: 三轴 rope 分维（16+56+56 = 128 = head_dim，与参考实现同值）。
AXES_DIM: tuple[int, int, int] = (16, 56, 56)
ROPE_THETA = 10000.0
#: 时间 / guidance 的正弦宽度。
T_FREQ = 256
T_SCALE = 1000.0
NORM_EPS = 1e-6
#: 蒸馏 guidance 默认值（**不是**权重事实：权重里只有投影，数值来自推理配置，dev 惯例 3.5）。
GUIDANCE_DEFAULT = 3.5


def default_axes(head_dim: int) -> tuple[int, int, int]:
    """按 ``head_dim`` 推三轴分维 —— 真权重 128 ⇒ ``(16, 56, 56)``（1 : 3.5 : 3.5）。

    唯一硬约束是「三轴**各为偶数、都非零、和为 head_dim**」（``rope_freqs`` 按 2 分量配对，
    奇数直接抛）。在这个约束下按**从真到假**三级回落：

    1. ``hd % 32 == 0`` ⇒ 真权重比例 ``1 : 3.5 : 3.5``（``128 ⇒ (16, 56, 56)`` ✓）；
    2. ``hd % 8 == 0`` ⇒ ``(¼, ½, ¼)``（``64 ⇒ (8, 28, 28)``、``16 ⇒ (4, 8, 4)``、``8 ⇒ (2, 4, 2)``）；
    3. 其余偶数（``6/10/12/14/…``）⇒ **t 轴保底 2 或 4**，h/w 均分（``12 ⇒ (4, 4, 4)``、
       ``10 ⇒ (2, 4, 4)``）。

    ⚠️ 前两级的判据都是**踩出来的**，各错一次，症状都是「建得成、跑不动」✗：
    第一版判 ``hd % 16`` 走 1:3.5:3.5 ⇒ ``16/48`` 得 ``(2, 7, 7)/(6, 21, 21)``；改成
    第二版判 ``hd % 4`` 走 ¼½¼ ⇒ ``12/20`` 得 ``(3, 6, 3)/(5, 10, 5)``。两次都是 t 轴落奇数、
    ``rope_freqs`` 抛「轴 0/1 的维度必须是偶数」，而**报错点落在 axes 上**，读起来像调用方
    传错了轴 ⇒ 自检 ⑨ 因此把「任意偶数都能建 RoPE」当硬判据 ✓。
    """
    hd = int(head_dim)
    if hd < 6 or hd % 2:
        raise ShapeError(
            f"head_dim {hd} 推不出三轴分维 ✗ —— 三轴各需**非零偶数** ⇒ 和最小 6、且必为偶数；"
            f"奇数 head_dim 请显式给 axes_dim")
    if hd % 32 == 0:
        return (hd // 8, hd * 7 // 16, hd * 7 // 16)
    if hd % 8 == 0:
        return (hd // 4, hd // 2, hd // 4)
    # 走到这里的偶数：hd % 4 == 0（12/20/28…）或 hd % 4 == 2（6/10/14…）。
    # 取 t 使 (hd - t)/2 也是偶数 ⇒ t ≡ hd (mod 4)，故 hd%4==0 取 4、hd%4==2 取 2 ✓。
    body = 4 if hd % 4 == 0 else 2
    return (body, (hd - body) // 2, (hd - body) // 2)


@dataclass(frozen=True)
class FluxShape:
    """从**权重形状**读出来的架构事实（族判定也只认这些）。"""

    features: int
    double_layers: int
    single_layers: int
    heads: int
    head_dim: int
    channels: int
    context_dim: int
    vec_dim: int
    mlp_hidden: int
    patch: int = PATCH
    t_freq: int = T_FREQ
    t_scale: float = T_SCALE
    #: ``None`` ⇒ 按 ``head_dim`` 推（见 :func:`default_axes`）；真权重 128 ⇒ ``(16, 56, 56)`` ✓。
    axes_dim: tuple[int, ...] | None = None
    theta: float = ROPE_THETA
    eps: float = NORM_EPS
    guidance_embed: bool = True
    guidance_default: float = GUIDANCE_DEFAULT

    def __post_init__(self) -> None:
        if self.axes_dim is None:
            # frozen dataclass ⇒ 只能走 object.__setattr__（这里就是唯一一处「建后补形状」）。
            object.__setattr__(self, "axes_dim", default_axes(self.head_dim))
        else:
            object.__setattr__(self, "axes_dim", tuple(int(x) for x in self.axes_dim))
        if self.heads <= 0 or self.heads * self.head_dim != self.features:
            raise ShapeError(
                f"heads×head_dim = {self.heads}×{self.head_dim} ≠ features {self.features} ✗")
        if sum(self.axes_dim) != self.head_dim:
            raise ShapeError(
                f"axes_dim {self.axes_dim} 之和 {sum(self.axes_dim)} ≠ head_dim "
                f"{self.head_dim} ✗（三轴分维必须正好铺满一个头）")
        if self.channels <= 0 or self.patch <= 0:
            raise ShapeError(f"channels/patch 非法：{self.channels}ch × {self.patch}² ✗")
        if self.mlp_hidden <= 0:
            raise ShapeError(f"mlp_hidden {self.mlp_hidden} 非法 ✗")
        # ⚠️ 这里**不能**再校验单流 linear1/linear2 的宽度 ✗✗：两者的期望宽就是由
        #    ``3·features+mlp_hidden`` / ``features+mlp_hidden`` 算出来的 ⇒ 在 dataclass 里比一遍是
        #    **恒真**的（写过一次，`if x != 同一个表达式` 永不成立 ✗，还容易让人以为"这里管着"）。
        #    真校验在 :func:`infer_shape` 里对着**权重形状**做（linear1/linear2 那两条 ✓）。

    @property
    def patch_in(self) -> int:
        """``img_in`` 入宽（= 通道 × patch²，也是 ``final_layer.linear`` 出宽）。"""
        return self.channels * self.patch * self.patch


def _dims(shapes: Mapping[str, Sequence[int]], name: str) -> tuple[int, ...]:
    if name not in shapes:
        raise ShapeError(f"权重里没有 {name} ✗ ⇒ 不是 flux 族（族判定只认形状）")
    return tuple(int(v) for v in shapes[name])


def _count_layers(shapes: Mapping[str, Sequence[int]], prefix: str) -> int:
    """数层数 —— 按**索引连续性**数，不是数键的个数（后者会把每层 24 个键数成 24 层）。"""
    seen: set[int] = set()
    for name in shapes:
        if not name.startswith(prefix):
            continue
        head = name[len(prefix):].split(".", 1)[0]
        if head.isdigit():
            seen.add(int(head))
    if not seen:
        raise ShapeError(f"权重里没有任何 {prefix}<i>.* 键 ✗ ⇒ 不是 flux 族")
    layers = len(seen)
    if seen != set(range(layers)):
        raise ShapeError(f"{prefix} 层号不是 0…{layers - 1} 连续 ✗：{sorted(seen)[:8]}…")
    return layers


def infer_shape(shapes: Mapping[str, Sequence[int]], *,
                heads: int | None = None,
                axes_dim: Sequence[int] | None = None,
                guidance_embed: bool | None = None,
                patch: int | None = None) -> FluxShape:
    """按权重形状推架构；**推不出或有矛盾就抛**，不带默认值硬上（族判定同样的脾气）。"""
    p = int(patch or PATCH)
    spans = p * p
    features, in_width = _dims(shapes, "img_in.weight")
    if in_width % spans:
        raise ShapeError(
            f"img_in 入宽 {in_width} 不是 {spans} 的整数倍 ✗ ⇒ 通道数读不出来"
            f"（这一族的 img_in 吃**拍平后**的宽 = 通道 × {spans}）")
    channels = in_width // spans

    txt_in, txt_out = _dims(shapes, "txt_in.weight")
    if txt_in != features:
        raise ShapeError(f"txt_in 出宽 {txt_in} ✗ ≠ features {features}")
    context_dim = txt_out

    vec_in, vec_out = _dims(shapes, "vector_in.in_layer.weight")
    if vec_in != features:
        raise ShapeError(f"vector_in 出宽 {vec_in} ✗ ≠ features {features}")
    vec_dim = vec_out

    t_in, t_out = _dims(shapes, "time_in.in_layer.weight")
    if (t_in, t_out) != (features, T_FREQ):
        raise ShapeError(f"time_in 形状 ({t_in}, {t_out}) ✗ 与 ({features}, {T_FREQ}) 不符")

    has_guidance = "guidance_in.in_layer.weight" in shapes
    if guidance_embed is not None and bool(guidance_embed) != has_guidance:
        raise ShapeError(
            f"显式说 guidance_embed={guidance_embed} ✗，但权重里"
            f"{'有' if has_guidance else '没有'} guidance_in —— 两者必须一致"
            f"（schnell 没有这根键，dev 有）")
    if has_guidance:
        g_in, g_out = _dims(shapes, "guidance_in.in_layer.weight")
        if (g_in, g_out) != (features, T_FREQ):
            raise ShapeError(f"guidance_in 形状 ({g_in}, {g_out}) ✗ 与 ({features}, {T_FREQ}) 不符")

    head_dim = _dims(shapes, "double_blocks.0.img_attn.norm.query_norm.scale")[0]
    if head_dim <= 0 or features % head_dim:
        raise ShapeError(f"head_dim {head_dim} 不整除 features {features} ✗")
    heads = int(heads or features // head_dim)
    if heads * head_dim != features:
        raise ShapeError(f"显式 heads {heads} × head_dim {head_dim} ≠ features {features} ✗")

    mlp_hidden = _dims(shapes, "double_blocks.0.img_mlp.0.weight")[0]
    qkv_out, qkv_in = _dims(shapes, "double_blocks.0.img_attn.qkv.weight")
    if (qkv_in, qkv_out) != (features, 3 * features):
        raise ShapeError(f"img_attn.qkv 形状 ({qkv_out}, {qkv_in}) ✗ ≠ (3h, h)")
    mod_out, mod_in = _dims(shapes, "double_blocks.0.img_mod.lin.weight")
    if (mod_in, mod_out) != (features, 6 * features):
        raise ShapeError(f"img_mod.lin 形状 ({mod_out}, {mod_in}) ✗ ≠ (6h, h)")

    l1_out, l1_in = _dims(shapes, "single_blocks.0.linear1.weight")
    single1 = 3 * features + mlp_hidden
    if (l1_in, l1_out) != (features, single1):
        raise ShapeError(
            f"single_blocks.0.linear1 形状 ({l1_out}, {l1_in}) ✗ ≠ ({single1}, {features})"
            f"（qkv 与 mlp_in 是**融合**的一条）")
    l2_out, l2_in = _dims(shapes, "single_blocks.0.linear2.weight")
    single2 = features + mlp_hidden
    if (l2_out, l2_in) != (features, single2):
        raise ShapeError(
            f"single_blocks.0.linear2 形状 ({l2_out}, {l2_in}) ✗ ≠ ({features}, {single2})"
            f"（attn_out 与 mlp_out 也是融合的）")

    fin_out, fin_in = _dims(shapes, "final_layer.linear.weight")
    if (fin_in, fin_out) != (features, channels * spans):
        raise ShapeError(
            f"final_layer.linear 形状 ({fin_out}, {fin_in}) ✗ ≠ ({channels * spans}, {features})"
            f" —— 输出端**不再**展开 patch（宽等于 img_in 入宽）")

    return FluxShape(
        features=features,
        double_layers=_count_layers(shapes, "double_blocks."),
        single_layers=_count_layers(shapes, "single_blocks."),
        heads=heads, head_dim=head_dim, channels=channels,
        context_dim=context_dim, vec_dim=vec_dim, mlp_hidden=mlp_hidden,
        patch=p, axes_dim=tuple(int(x) for x in axes_dim) if axes_dim else None,
        guidance_embed=has_guidance,
    )


def expected_keys(shape: FluxShape) -> set[str]:
    """这一族**应该**有哪些键 —— 与盘上真文件逐键对账。

    盘上 ``flux1-dev`` 的 780 键 = 19 double × 24 + 38 single × 8 + 顶层 16 + guidance 4
    （= 456 + 304 + 16 + 4 ✓）。``guidance_embed=False`` 的变体（schnell 系）按此推定少 4 条 = 776
    —— 盘上没有该文件，**未验**，只是算出来的。
    """
    keys = {"img_in.weight", "img_in.bias", "txt_in.weight", "txt_in.bias",
            "time_in.in_layer.weight", "time_in.in_layer.bias",
            "time_in.out_layer.weight", "time_in.out_layer.bias",
            "vector_in.in_layer.weight", "vector_in.in_layer.bias",
            "vector_in.out_layer.weight", "vector_in.out_layer.bias",
            "final_layer.linear.weight", "final_layer.linear.bias",
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.adaLN_modulation.1.bias"}
    if shape.guidance_embed:
        keys |= {"guidance_in.in_layer.weight", "guidance_in.in_layer.bias",
                 "guidance_in.out_layer.weight", "guidance_in.out_layer.bias"}
    for i in range(shape.double_layers):
        p = f"double_blocks.{i}."
        for stream in ("img", "txt"):
            keys |= {f"{p}{stream}_mod.lin.weight", f"{p}{stream}_mod.lin.bias",
                     f"{p}{stream}_attn.qkv.weight", f"{p}{stream}_attn.qkv.bias",
                     f"{p}{stream}_attn.proj.weight", f"{p}{stream}_attn.proj.bias",
                     f"{p}{stream}_attn.norm.query_norm.scale",
                     f"{p}{stream}_attn.norm.key_norm.scale"}
            for idx in (0, 2):
                keys |= {f"{p}{stream}_mlp.{idx}.weight", f"{p}{stream}_mlp.{idx}.bias"}
    for i in range(shape.single_layers):
        p = f"single_blocks.{i}."
        keys |= {f"{p}linear1.weight", f"{p}linear1.bias",
                 f"{p}linear2.weight", f"{p}linear2.bias",
                 f"{p}modulation.lin.weight", f"{p}modulation.lin.bias",
                 f"{p}norm.query_norm.scale", f"{p}norm.key_norm.scale"}
    return keys


# ------------------------------------------------------------------ 组件


class _RMSNorm(nn.Module):
    """**无偏置** RMSNorm，权重叫 ``scale`` 且**按原样乘**（不做 ``scale+1``）。

    ⚠️ 与 ``dit_generic.RMSNorm`` 不是一个东西：那一族（Krea2）的参数存的是「相对 1 的偏移」，
    这一族（Flux / Qwen-Image）存的就是权重本身。互相套用 ⇒ 整网 q/k 尺度差 1 倍，
    不报错、注意力变平、图能出但细节糊。计算升 fp32（bf16 下 rms 累加掉精度）。
    """

    def __init__(self, dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # 只在**低精度**时升 fp32（bf16 下 rms 累加会掉精度）✓；fp32/fp64 原样算 ✓✗ ——
        # 无条件 ``x.float()`` 会把 fp64 自检悄悄降成 fp32，让「双精度对拍」凭空多出 1e-9
        # 量级残差，读起来像结构写错了（2026-11-01 实测：6.1e-9 → 2.1e-9）。
        work = x if x.dtype in (torch.float32, torch.float64) else x.float()
        return F.rms_norm(work, (work.shape[-1],), weight=self.scale.to(work.dtype),
                          eps=self.eps).to(x.dtype)


class _QKNorm(nn.Module):
    """``query_norm`` / ``key_norm`` 两个 head_dim 宽的 RMSNorm（键名就叫这两格）。"""

    def __init__(self, head_dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.query_norm = _RMSNorm(head_dim, eps)
        self.key_norm = _RMSNorm(head_dim, eps)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        # 归完显式回到 v 的精度：不齐时 SDPA 会直接报 query float32 / value bf16。
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class _MLPEmbedder(nn.Module):
    """``in_layer → SiLU → out_layer``（键名 ``in_layer``/``out_layer``）。"""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden, bias=True)
        self.out_layer = nn.Linear(hidden, hidden, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(F.silu(self.in_layer(x)))


class _Modulation(nn.Module):
    """``SiLU(vec) → Linear`` ⇒ 切成 (shift, scale, gate)；双流块切 6 组 = 两套。

    ⚠️ 顺序是 **(shift, scale, gate)** —— ``chunk`` 出来的前三个归 img、后三个归 txt。
    写成 (scale, shift) 不报错，但亮度随步数漂。
    """

    def __init__(self, dim: int, double: bool) -> None:
        super().__init__()
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor], ...]:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        out = self.lin(F.silu(vec)).chunk(self.multiplier, dim=-1)
        first = (out[0], out[1], out[2])
        return (first, (out[3], out[4], out[5])) if self.multiplier == 6 else (first,)


def _attention(q: Tensor, k: Tensor, v: Tensor, freqs: Tensor | None,
               mask: Tensor | None = None) -> Tensor:
    """RoPE 之后走 SDPA ⇒ ``(b, n, heads·head_dim)``。``mask`` 是**加性**偏置 ``(b,1,1,n_kv)``。"""
    if freqs is not None:
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
    out = sdpa.attention(q, k, v, attn_mask=mask)
    b, h, n, d = out.shape
    return out.transpose(1, 2).reshape(b, n, h * d)


class _SelfAttention(nn.Module):
    """融合 ``qkv`` + QK-norm + ``proj``（这一族的 ``proj`` **有** bias）。"""

    def __init__(self, shape: FluxShape) -> None:
        super().__init__()
        self.heads = shape.heads
        d = shape.features
        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.norm = _QKNorm(shape.head_dim, shape.eps)
        self.proj = nn.Linear(d, d, bias=True)

    def forward(self, x: Tensor, freqs: Tensor | None,
                mask: Tensor | None = None) -> Tensor:
        b, n, _ = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)
        return self.proj(_attention(q, k, v, freqs, mask))


class _DoubleBlock(nn.Module):
    """双流块：两条流各自 norm/mod/mlp，只在注意力处拼 ``(txt, img)``，回来再切开。"""

    def __init__(self, shape: FluxShape) -> None:
        super().__init__()
        d, m = shape.features, shape.mlp_hidden
        for stream in ("img", "txt"):
            setattr(self, f"{stream}_mod", _Modulation(d, double=True))
            setattr(self, f"{stream}_norm1", nn.LayerNorm(d, elementwise_affine=False,
                                                         eps=shape.eps))
            setattr(self, f"{stream}_attn", _SelfAttention(shape))
            setattr(self, f"{stream}_norm2", nn.LayerNorm(d, elementwise_affine=False,
                                                         eps=shape.eps))
            setattr(self, f"{stream}_mlp",
                    nn.Sequential(nn.Linear(d, m, bias=True),
                                  nn.GELU(approximate="tanh"),
                                  nn.Linear(m, d, bias=True)))

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, freqs: Tensor | None,
                mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        (img_m1, img_m2), (txt_m1, txt_m2) = self.img_mod(vec), self.txt_mod(vec)
        img_mod = self.img_norm1(img) * (1 + img_m1[1]) + img_m1[0]
        txt_mod = self.txt_norm1(txt) * (1 + txt_m1[1]) + txt_m1[0]
        b, n_img, _ = img_mod.shape
        b2, n_txt, _ = txt_mod.shape
        q_img, k_img, v_img = self.img_attn.qkv(img_mod).view(b, n_img, 3, self.img_attn.heads, -1).permute(2, 0, 3, 1, 4)
        q_txt, k_txt, v_txt = self.txt_attn.qkv(txt_mod).view(b2, n_txt, 3, self.txt_attn.heads, -1).permute(2, 0, 3, 1, 4)
        q_img, k_img = self.img_attn.norm(q_img, k_img, v_img)
        q_txt, k_txt = self.txt_attn.norm(q_txt, k_txt, v_txt)
        # 拼接顺序 (txt, img)：文本在前，图像在后 —— 切回来时也必须按同一个长度切。
        q = torch.cat((q_txt, q_img), dim=2)
        k = torch.cat((k_txt, k_img), dim=2)
        v = torch.cat((v_txt, v_img), dim=2)
        attn = _attention(q, k, v, freqs, mask)
        img_attn, txt_attn = attn[:, n_txt:], attn[:, :n_txt]
        img = img + img_m1[2] * self.img_attn.proj(img_attn)
        img = img + img_m2[2] * self.img_mlp(self.img_norm2(img) * (1 + img_m2[1]) + img_m2[0])
        txt = txt + txt_m1[2] * self.txt_attn.proj(txt_attn)
        txt = txt + txt_m2[2] * self.txt_mlp(self.txt_norm2(txt) * (1 + txt_m2[1]) + txt_m2[0])
        return img, txt


class _SingleBlock(nn.Module):
    """单流块：``linear1`` 一次产出 ``[qkv | mlp_in]``（融合），``linear2`` 吃 ``[attn | mlp_out]``。"""

    def __init__(self, shape: FluxShape) -> None:
        super().__init__()
        d, m = shape.features, shape.mlp_hidden
        self.mlp_hidden = m
        self.heads = shape.heads
        self.linear1 = nn.Linear(d, 3 * d + m, bias=True)
        self.linear2 = nn.Linear(d + m, d, bias=True)
        self.norm = _QKNorm(shape.head_dim, shape.eps)
        self.pre_norm = nn.LayerNorm(d, elementwise_affine=False, eps=shape.eps)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = _Modulation(d, double=False)

    def forward(self, x: Tensor, vec: Tensor, freqs: Tensor | None,
                mask: Tensor | None = None) -> Tensor:
        (mod,) = self.modulation(vec)
        x_mod = self.pre_norm(x) * (1 + mod[1]) + mod[0]
        split = self.linear1.out_features - self.mlp_hidden
        qkv, mlp = torch.split(self.linear1(x_mod), [split, self.mlp_hidden], dim=-1)
        b, n, _ = qkv.shape
        q, k, v = qkv.view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)
        attn = _attention(q, k, v, freqs, mask)
        return x + mod[2] * self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))


class _LastLayer(nn.Module):
    """``adaLN_modulation`` 出 (shift, scale) ⇒ 归一化后一次线性投到 ``patch_in``。"""

    def __init__(self, shape: FluxShape) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(shape.features, elementwise_affine=False,
                                       eps=shape.eps)
        self.linear = nn.Linear(shape.features, shape.patch_in, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(shape.features, 2 * shape.features, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear(self.norm_final(x) * (1 + scale) + shift)


# ------------------------------------------------------------------ 打包 / 组装


def _patchify(x: Tensor, p: int) -> Tensor:
    """``(b,c,h,w)`` ⇒ ``(b, (h/p)·(w/p), c·p²)``。

    ⚠️ **边长为奇数 ⇒ 直接抛**（参考实现 ``pad_to_patch_size`` 会先按比例补到偶数、解完再切回 ✓）。
    本仓不补：补边是「悄悄改了构图」✗，而 1024² 档 latent 恒为 128×128 ✓ ⇒ 正常路径不会碰到。
    """
    b, c, h, w = x.shape
    if h % p or w % p:
        raise ShapeError(f"latent {h}×{w} 不是 patch {p} 的整数倍 ✗")
    x = x.view(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // p) * (w // p), c * p * p)


def _unpatchify(x: Tensor, p: int, channels: int, h: int, w: int) -> Tensor:
    hh, ww = h // p, w // p
    x = x.view(x.shape[0], hh, ww, channels, p, p)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(x.shape[0], channels, hh * p, ww * p)


def _latent_ids(hh: int, ww: int, device: torch.device) -> Tensor:
    """三轴位置号：``t=0``、``h=0…H−1``、``w=0…W−1``（**不居中**，与 Qwen-Image 那族相反）。"""
    t = torch.zeros(hh, ww, device=device)
    h = torch.arange(hh, device=device, dtype=torch.float32)[:, None].expand(hh, ww)
    w = torch.arange(ww, device=device, dtype=torch.float32)[None, :].expand(hh, ww)
    return torch.stack((t, h, w), dim=-1).reshape(1, hh * ww, 3)


class FluxDiT(nn.Module):
    """Flux.1 主干（不含文本塔）。``forward(latents, σ, context, vector=…, guidance=…)``。"""

    def __init__(self, shape: FluxShape) -> None:
        super().__init__()
        self.shape = shape
        d = shape.features
        self.img_in = nn.Linear(shape.patch_in, d, bias=True)
        self.time_in = _MLPEmbedder(shape.t_freq, d)
        self.guidance_in = _MLPEmbedder(shape.t_freq, d) if shape.guidance_embed else None
        self.vector_in = _MLPEmbedder(shape.vec_dim, d)
        self.txt_in = nn.Linear(shape.context_dim, d, bias=True)
        self.double_blocks = nn.ModuleList(
            _DoubleBlock(shape) for _ in range(shape.double_layers))
        self.single_blocks = nn.ModuleList(
            _SingleBlock(shape) for _ in range(shape.single_layers))
        self.final_layer = _LastLayer(shape)
        #: 外部条件（**按需注入** ✓）：盘上既没有 T5-XXL 也没有 CLIP-L ⇒ 条件只能从外面给 ✓。
        #: 不给 ⇒ 零向量 + 默认 guidance ✓（`describe` 里会如实标「条件未注入」✗ 不装成真语义 ✓）。
        self.default_vector: Tensor | None = None
        self.default_guidance: float | None = None

    # -------------------------------------------------------------- 前向

    def forward(self, latents: Tensor, timesteps: Tensor, context: Tensor, *,
                vector: Tensor | None = None, guidance: Tensor | float | None = None,
                attention_mask: Tensor | None = None) -> Tensor:
        s = self.shape
        if latents.ndim != 4:
            raise ShapeError(f"latents 要 (b,c,h,w)，拿到 {tuple(latents.shape)} ✗")
        b, c, h, w = latents.shape
        if c != s.channels:
            raise ShapeError(f"latents 通道 {c} ✗ ≠ 权重 {s.channels}（族的硬事实）")
        if context.ndim != 3 or context.shape[-1] != s.context_dim:
            raise ShapeError(
                f"context 要 (b,n,{s.context_dim})（T5-XXL 宽），拿到 "
                f"{tuple(context.shape)} ✗ —— 少一维就会在 txt_in 处报形状不匹配")
        dtype = self.img_in.weight.dtype
        p = s.patch
        hh, ww = h // p, w // p
        img = _patchify(latents.to(dtype), p)
        ids = torch.cat((torch.zeros(1, context.shape[1], 3, device=latents.device),
                         _latent_ids(hh, ww, latents.device)), dim=1)
        freqs = rope_freqs(ids, list(s.axes_dim), s.theta).unsqueeze(1).to(dtype)

        img = self.img_in(img)
        sigma = timesteps.reshape(-1).to(torch.float32)
        if sigma.numel() == 1:
            sigma = sigma.expand(b)
        # ⚠️ ``time_factor`` 必须**显式**传 ``s.t_scale`` ✓✗：:func:`timestep_embedding` 的默认值
        #    恰好也是 1000 ✓，漏传时行为一模一样 ✓、**但 `FluxShape.t_scale` 就成了没人读的写死字段** ✗
        #    —— 改它不生效、改默认值才生效，这类"配置项是摆设"最难查 ✓✗（同族 ``dit_qwen_image``
        #    就是显式传的 ✓）。
        vec = self.time_in(timestep_embedding(sigma, s.t_freq, time_factor=s.t_scale).to(dtype))
        if self.guidance_in is not None:
            if guidance is None:
                guidance = self.default_guidance
            g = _guidance_values(guidance, b, s.guidance_default, latents.device)
            vec = vec + self.guidance_in(
                timestep_embedding(g, s.t_freq, time_factor=s.t_scale).to(dtype))
        if vector is None:
            vector = self.default_vector
        if vector is None:
            # 诚实降级：没有 CLIP-L pooled 时补零，语义那一路整段缺席（文档里已写死不是默认路径）。
            vector = torch.zeros(b, s.vec_dim, device=latents.device, dtype=dtype)
        elif vector.shape[-1] < s.vec_dim:
            raise ShapeError(
                f"vector 宽 {vector.shape[-1]} ✗ < vec_dim {s.vec_dim}（pooled 768 维）")
        vec = vec + self.vector_in(vector[..., :s.vec_dim].to(dtype))
        txt = self.txt_in(context.to(dtype))
        mask = _attn_mask(attention_mask, txt.shape[1], img.shape[1], dtype)

        for block in self.double_blocks:
            img, txt = block(img, txt, vec, freqs, mask)
        x = torch.cat((txt, img), dim=1)
        for block in self.single_blocks:
            x = block(x, vec, freqs, mask)
        x = x[:, txt.shape[1]:]
        return _unpatchify(self.final_layer(x, vec), p, s.channels, h, w).to(dtype)


def _guidance_values(guidance: Tensor | float | None, batch: int, default: float,
                     device: torch.device) -> Tensor:
    if guidance is None:
        return torch.full((batch,), float(default), device=device, dtype=torch.float32)
    g = torch.as_tensor(guidance, device=device, dtype=torch.float32).reshape(-1)
    return g.expand(batch) if g.numel() == 1 else g


def _attn_mask(mask: Tensor | None, n_txt: int, n_img: int,
               dtype: torch.dtype) -> Tensor | None:
    """``(b, n_txt)`` 的布尔「保留」掩码 ⇒ 加性偏置 ``(b,1,1,n_txt+n_img)``。

    ⚠️ 文本 token 的**补零位**必须挡掉：T5 输出在 pad 位不是 0，而是「有值的垃圾」，
    不挡就对整张图的注意力灌噪声（能出图、语义却糊）。
    """
    if mask is None:
        return None
    m = mask.to(torch.bool)
    if m.ndim == 1:
        m = m[None, :]
    m = m[:, :n_txt]
    bias = torch.zeros(m.shape[0], 1, 1, n_txt + n_img, device=m.device, dtype=dtype)
    bias[:, :, :, :n_txt] = torch.where(m[:, None, None, :], 0.0,
                                        torch.finfo(dtype).min)
    return bias


def build(shape: FluxShape) -> FluxDiT:
    return FluxDiT(shape)


def build_from_shapes(shapes: Mapping[str, Sequence[int]], **explicit) -> FluxDiT:
    return FluxDiT(infer_shape(shapes, **explicit))
