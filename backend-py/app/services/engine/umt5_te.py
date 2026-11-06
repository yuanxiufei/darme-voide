"""Wan2.1 / Wan2.2 的**文本塔** = UMT5-XXL encoder（4096 维、24 层、**双向**、相对位置偏置**每层各一份**）。

事实来源（只参考行为、实现自己写 ✓）：
``ComfyUI/comfy/text_encoders/t5.py`` ＋ 同目录 ``umt5_config_xxl.json``：

* 骨架 ✓：``d_model=4096, d_kv=64, num_heads=64, d_ff=10240, num_layers=24,
  dense_act_fn="gelu_pytorch_tanh", is_gated_act=True, vocab_size=256384`` ✓；
* 每块两份残差：``x += attn(RMSNorm(x))`` ✓、``x += ffn(RMSNorm(x))`` ✓（**pre-norm** ✓）；
* FFN 是**门控**形态 ✓：``wo(gelu(wi_0(x)) * wi_1(x))`` ✓（不是 ReLU 单支 ✗）；
* RMSNorm **无 affine 偏置**、``eps=1e-6`` ✓（只有 ``.weight``，没有 ``.bias`` ✗）；
* 注意力 ``q/k/v/o`` **全部无 bias** ✓；打分前 ``k`` 乘 ``sqrt(head_dim)`` ✓，
  而注意力内部又除一次 ``sqrt(head_dim)`` ✓ ⇒ **两者抵消** ✓（本实现把 scale 一次写进
  ``scaled_dot_product_attention(scale=...)`` ✓，数学等价 ✓、少开一份 k 副本 ✓）；
* 相对位置偏置 32 桶 / ``max_distance=128`` ✓，``bidirectional=True`` ✓；
* ⚠️ **UMT5 与 T5 的唯一结构性差别** ✓✗：``T5Stack`` 里 ``relative_attention_bias=((not relative_attention) or (i == 0))`` ✓
  —— 传 ``relative_attention=False``（``model_type == "umt5"`` ✓）时 ``not False`` 恒真 ✓
  ⇒ **每层都有自己的 bias 表** ✓（T5 系只有第 0 层有 ✗）。盘上这份权重正好 24 份
  ``relative_attention_bias.weight`` ✓ —— 按"只有第 0 层"写会**报 23 个 unexpected** ✓✗；
* 掩码 ✓：``1 - attention_mask`` 后 ``masked_fill(≠0, -finfo.max)`` ✓ —— 是**加性**掩码 ✓，
  不是 bool ✓（用 bool 会把「被 pad 的位置」当成"可看"✓✗）；
* 取层 ✓：``T5.forward`` 直接返回 ``encoder(...)`` 的 **final_layer_norm 之后** ✓
  （``layer="last"`` ✓）—— 与 Z-Image 取 `-2` 层不同 ✗；
* 填空 ✓：``pad_token_id=0`` ✓、``end=1`` ✓、``min_length=512`` ✓ ⇒ 提示词不足 512 一律补齐 ✓，
  **不裁到固定长度**（超过就按实际长度跑 ✓）。

⚠️ 与 :mod:`.llm` 的分工：本模块**只**拼壳（键名与权重文件逐字一致 ✓：
``encoder.block.0.layer.0.SelfAttention.q.weight`` ✓）与前向 ✓，没有 ``lm_head`` ✗
（盘上 ``umt5_xxl_fp8_e4m3fn_scaled.safetensors`` 里只有 ``shared`` ＋ encoder ✓）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "DEFAULT_MAX_LENGTH",
    "END_TOKEN_ID",
    "PAD_TOKEN_ID",
    "ShapeError",
    "Umt5Encoder",
    "Umt5TeRuntime",
    "Umt5TeShape",
    "build_from_shapes",
    "expected_keys",
    "family_title",
    "infer_shape",
]

#: 填空 token ✓（``umt5_config_xxl.json::pad_token_id`` ✓）
PAD_TOKEN_ID = 0

#: 句末 token ✓（``end_token_id=1`` ✓，Wan 的提示词尾部固定带它 ✓）
END_TOKEN_ID = 1

#: 最短序列 ✓（``min_length=512`` ✓）—— 短提示词补到这里，长提示词不截 ✗
DEFAULT_MAX_LENGTH = 512

#: 相对位置分桶 ✓（``T5Attention.relative_attention_num_buckets`` ✓）
NUM_BUCKETS = 32

#: 分桶上限距离 ✓
MAX_DISTANCE = 128

_LAYER_RE = re.compile(r"^encoder\.block\.(\d+)\.(.+)$")


class ShapeError(ValueError):
    """权重形状推不出 UMT5 骨架 ✓（是「不是这一族」或「没下全」✓，两种都要明说 ✓）。"""


@dataclass(frozen=True)
class Umt5TeShape:
    """UMT5 encoder 的超参 ✓（默认值就是 XXL 档 ✓）。"""

    vocab_size: int = 256384
    d_model: int = 4096
    d_kv: int = 64
    num_heads: int = 64
    d_ff: int = 10240
    num_layers: int = 24
    eps: float = 1e-6
    num_buckets: int = NUM_BUCKETS
    max_distance: int = MAX_DISTANCE

    @property
    def inner_dim(self) -> int:
        return self.d_kv * self.num_heads

    @property
    def hidden_size(self) -> int:
        """送进主干的宽度 ✓（Wan 的 ``cross_attn`` 收 4096 ✓）。"""
        return self.d_model

    @property
    def head_dim(self) -> int:
        return self.d_kv

    def label(self) -> str:
        return f"UMT5 {self.num_layers}L · {self.d_model}d · {self.d_ff}ff"


def _suffixes(shape: Umt5TeShape) -> tuple[str, ...]:
    """每层必须**全有**的 10 个键 ✓（缺一个都是"不是这一族"✗）。"""
    del shape  # 键集与超参无关 ✓（每层都是同一套 10 条 ✓）
    return (
        "layer.0.SelfAttention.q.weight",
        "layer.0.SelfAttention.k.weight",
        "layer.0.SelfAttention.v.weight",
        "layer.0.SelfAttention.o.weight",
        # ⚠️ UMT5 **每层**都有相对位置偏置 ✓（T5 系只有第 0 层 ✗ —— 见模块 docstring）
        "layer.0.SelfAttention.relative_attention_bias.weight",
        "layer.0.layer_norm.weight",
        "layer.1.DenseReluDense.wi_0.weight",
        "layer.1.DenseReluDense.wi_1.weight",
        "layer.1.DenseReluDense.wo.weight",
        "layer.1.layer_norm.weight",
    )


def expected_keys(shape: Umt5TeShape) -> set[str]:
    """模型**完整**的键集合 ✓（244 条 = 24×10 ＋ ``final_layer_norm`` ＋ ``shared`` ✓）。"""
    keys = {"shared.weight", "encoder.final_layer_norm.weight"}
    for i in range(shape.num_layers):
        for suf in _suffixes(shape):
            keys.add(f"encoder.block.{i}.{suf}")
    return keys


def infer_shape(shapes: dict[str, tuple[int, ...]]) -> Umt5TeShape:
    """从**权重形状**反推超参 ✓（一个都不猜 ✓）。"""
    shared = shapes.get("shared.weight")
    if shared is None or len(shared) != 2:
        raise ShapeError("找不到 `shared.weight`（两维）✗ —— 这不像 UMT5 文本塔（或者权重没下全）")

    layers: list[int] = []
    for key in shapes:
        m = _LAYER_RE.match(key)
        if m:
            layers.append(int(m.group(1)))
    if not layers:
        raise ShapeError("找不到 `encoder.block.<n>.*` ✗ —— 这不是 UMT5 骨架")
    num_layers = max(layers) + 1
    if len(set(layers)) != num_layers:
        missing = sorted(set(range(num_layers)) - set(layers))
        raise ShapeError(f"层号不连续 ✗：缺第 {missing[:6]} 层（共 {num_layers} 层）")

    q = shapes.get("encoder.block.0.layer.0.SelfAttention.q.weight")
    wi = shapes.get("encoder.block.0.layer.1.DenseReluDense.wi_0.weight")
    rb = shapes.get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
    if q is None or len(q) != 2:
        raise ShapeError("找不到 `block.0.layer.0.SelfAttention.q.weight`（两维）✗")
    if wi is None or len(wi) != 2:
        raise ShapeError("找不到 `block.0.layer.1.DenseReluDense.wi_0.weight`（两维）✗")
    if rb is None or len(rb) != 2:
        raise ShapeError(
            "找不到 `block.0.layer.0.SelfAttention.relative_attention_bias.weight`（两维）✗ "
            "—— UMT5 **每层**都该有这一张（T5 系才只有第 0 层）")

    inner, d_model = int(q[0]), int(q[1])
    num_heads = int(rb[1])
    if num_heads <= 0 or inner % num_heads:
        raise ShapeError(f"inner_dim={inner} 不能被 num_heads={num_heads} 整除 ✗")
    d_kv = inner // num_heads
    if int(rb[0]) != NUM_BUCKETS:
        raise ShapeError(f"相对位置桶数 {int(rb[0])} ≠ {NUM_BUCKETS} ✗")
    # ⚠️ 门控 FFN 的 ``wi_0``/``wi_1`` **各自**都是 ``d_ff`` ✓（不是总和 ✓）⇒ 直接取第 0 维 ✓
    return Umt5TeShape(
        vocab_size=int(shared[0]),
        d_model=d_model,
        d_kv=d_kv,
        num_heads=num_heads,
        d_ff=int(wi[0]),
        num_layers=num_layers,
    )


# --------------------------------------------------------------------------- 骨架


class _RmsNorm(nn.Module):
    """``T5LayerNorm`` ✓：**无 bias** ✓、``eps`` 在**开方前**加 ✓、``weight`` 初始化成 1 ✓。"""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + self.eps)
        return (normed.to(x.dtype) * self.weight.to(x.dtype))


def _relative_position_bucket(relative_position: torch.Tensor, num_buckets: int,
                              max_distance: int) -> torch.Tensor:
    """照抄 ``T5Attention._relative_position_bucket`` ✓（bidirectional=True 固定 ✓）。"""
    relative_buckets = 0
    num_buckets //= 2
    relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
    pos = torch.abs(relative_position)
    max_exact = num_buckets // 2
    is_small = pos < max_exact
    pos_if_large = max_exact + (
        torch.log(pos.float() / max_exact) / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    pos_if_large = torch.min(pos_if_large, torch.full_like(pos_if_large, num_buckets - 1))
    relative_buckets += torch.where(is_small, pos, pos_if_large)
    return relative_buckets


class _SelfAttention(nn.Module):
    def __init__(self, shape: Umt5TeShape) -> None:
        super().__init__()
        inner = shape.inner_dim
        self.q = nn.Linear(shape.d_model, inner, bias=False)
        self.k = nn.Linear(shape.d_model, inner, bias=False)
        self.v = nn.Linear(shape.d_model, inner, bias=False)
        self.o = nn.Linear(inner, shape.d_model, bias=False)
        self.relative_attention_bias = nn.Embedding(shape.num_buckets, shape.num_heads)
        self.num_heads = shape.num_heads
        self.head_dim = shape.d_kv
        self.num_buckets = shape.num_buckets
        self.max_distance = shape.max_distance
        #: comfy 是「先 k×sqrt(d) 再除 sqrt(d)」✓ ⇒ 净效果就是标准缩放 ✓（这里一次写清 ✓）
        self.scale = shape.head_dim ** -0.5
        self._bias_cache: tuple[int, int, torch.Tensor] | None = None

    def position_bias(self, length: int, dtype: torch.dtype) -> torch.Tensor:
        """``compute_bias`` ✓：``(1, heads, L, L)`` ✓（同长度同 dtype 命中缓存 ✓）。"""
        cached = self._bias_cache
        if cached is not None and cached[0] == length and cached[1] == dtype:
            return cached[2]
        pos = torch.arange(length, dtype=torch.long, device=self.relative_attention_bias.weight.device)
        relative = pos[None, :] - pos[:, None]                      # (L, L) = memory - context ✓
        buckets = _relative_position_bucket(relative, self.num_buckets, self.max_distance)
        values = self.relative_attention_bias(buckets)              # (L, L, heads) ✓
        bias = values.permute(2, 0, 1).unsqueeze(0).to(dtype).contiguous()
        self._bias_cache = (length, dtype, bias)
        return bias

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b, length, _ = x.shape
        q = self.q(x).view(b, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, length, self.num_heads, self.head_dim).transpose(1, 2)
        bias = self.position_bias(length, x.dtype)
        if mask is not None:
            bias = bias + mask
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
        out = out.transpose(1, 2).reshape(b, length, self.num_heads * self.head_dim)
        return self.o(out)


class _GatedFfn(nn.Module):
    """``T5DenseGatedActDense`` ✓：``wo(gelu(wi_0(x)) * wi_1(x))`` ✓（激活是 ``gelu_pytorch_tanh`` ✓）。"""

    def __init__(self, shape: Umt5TeShape) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(shape.d_model, shape.d_ff, bias=False)
        self.wi_1 = nn.Linear(shape.d_model, shape.d_ff, bias=False)
        self.wo = nn.Linear(shape.d_ff, shape.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(F.gelu(self.wi_0(x), approximate="tanh") * self.wi_1(x))


class _LayerSelfAttention(nn.Module):
    def __init__(self, shape: Umt5TeShape) -> None:
        super().__init__()
        self.SelfAttention = _SelfAttention(shape)
        self.layer_norm = _RmsNorm(shape.d_model, shape.eps)


class _LayerFfn(nn.Module):
    def __init__(self, shape: Umt5TeShape) -> None:
        super().__init__()
        self.DenseReluDense = _GatedFfn(shape)
        self.layer_norm = _RmsNorm(shape.d_model, shape.eps)


class _Block(nn.Module):
    def __init__(self, shape: Umt5TeShape) -> None:
        super().__init__()
        self.layer = nn.ModuleList([_LayerSelfAttention(shape), _LayerFfn(shape)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        ln = self.layer[0].layer_norm(x)
        x = x + self.layer[0].SelfAttention(ln, mask)
        x = x + self.layer[1].DenseReluDense(self.layer[1].layer_norm(x))
        return x


class _Stack(nn.Module):
    def __init__(self, shape: Umt5TeShape) -> None:
        super().__init__()
        self.block = nn.ModuleList([_Block(shape) for _ in range(shape.num_layers)])
        self.final_layer_norm = _RmsNorm(shape.d_model, shape.eps)


class Umt5Encoder(nn.Module):
    """UMT5 encoder 本体 ✓（键名与 ``umt5_xxl_fp8_e4m3fn_scaled.safetensors`` 逐字一致 ✓）。"""

    def __init__(self, shape: Umt5TeShape) -> None:
        super().__init__()
        self.shape_ = shape
        self.shared = nn.Embedding(shape.vocab_size, shape.d_model)
        self.encoder = _Stack(shape)

    def _dtype(self) -> torch.dtype:
        return self.shared.weight.dtype

    def forward(self, input_ids: torch.Tensor | None = None, *,
                embeds: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``input_ids [B, L]`` 或已算好的 ``embeds [B, L, d]`` ⇒ ``hidden [B, L, d_model]`` ✓。"""
        if input_ids is None:
            if embeds is None:
                raise ValueError("input_ids 与 embeds 至少给一个 ✗")
            x = embeds
        else:
            x = self.shared(input_ids)
        want = self._dtype()
        if x.dtype != want:                       # ⚠️ 输入精度不齐会让第一层 Linear 直接报错 ✓✗
            x = x.to(want)
        mask = None
        if attention_mask is not None:
            # ⚠️ 只屏蔽 **key 那一维** ✓（``(B, 1, 1, L)`` 广播到 ``(B, heads, L, L)`` ✓）。
            # 写成 ``(B, 1, L, 1)`` 就成了"把 pad 的 **query** 整行屏蔽" ✗ —— 形状照样能广播 ✓✗，
            # 症状是「有效位的结果会随 pad 区内容变化」✓✗（本仓 2026-11-01 的 ④′ 判据专抓这一条 ✓）。
            keep = attention_mask.to(torch.bool).reshape(attention_mask.shape[0], 1, 1, -1)
            mask = torch.zeros((keep.shape[0], 1, 1, keep.shape[-1]), dtype=want, device=x.device)
            mask = mask.masked_fill(~keep, -torch.finfo(want).max)
        for block in self.encoder.block:
            x = block(x, mask)
        return self.encoder.final_layer_norm(x)


def build_from_shapes(shapes: dict[str, tuple[int, ...]], *,
                      dtype: torch.dtype | str = torch.bfloat16) -> Umt5Encoder:
    """按权重形状建骨架 ✓（形状推不出就 :class:`ShapeError` ✓，绝不"先建再说"✗）。"""
    shape = infer_shape(shapes)
    module = Umt5Encoder(shape)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, torch.bfloat16)
    return module.to(dtype)


# --------------------------------------------------------------------------- 运行期信息


class Umt5TeRuntime:
    """送进 pipeline 的**口径** ✓（宽度 / 最大长度 / pad 与 eos ✓）—— 不做前向 ✗。"""

    def __init__(self, shape: Umt5TeShape, *, max_length: int = DEFAULT_MAX_LENGTH) -> None:
        self.shape = shape
        self.max_length = int(max_length)

    @property
    def output_dim(self) -> int:
        return self.shape.d_model

    @property
    def vocab_size(self) -> int:
        return self.shape.vocab_size

    @property
    def pad_token_id(self) -> int:
        return PAD_TOKEN_ID

    @property
    def end_token_id(self) -> int:
        return END_TOKEN_ID

    def prompt(self, text: str) -> str:
        """Wan 的提示词**不做模板包装** ✓（T5 吃裸文本 ✓，只在尾部补 `</s>` ✓）。"""
        return (text or "").strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "umt5",
            "label": self.shape.label(),
            "outputDim": self.output_dim,
            "vocabSize": self.vocab_size,
            "numLayers": self.shape.num_layers,
            "numHeads": self.shape.num_heads,
            "dKv": self.shape.d_kv,
            "dFf": self.shape.d_ff,
            "eps": self.shape.eps,
            "maxLength": self.max_length,
            "padTokenId": PAD_TOKEN_ID,
            "endTokenId": END_TOKEN_ID,
            "relativePosition": "每层各一份（UMT5 口径）· 32 桶 / 128 距离 · 双向",
            "masking": "加性掩码（1 - attention_mask → -finfo.max）",
            "zeroOutMasked": True,
        }


def family_title(key: str) -> str:
    """人话族名 ✓（回执里别再写死 ✓）。"""
    return {
        "umt5": "UMT5-XXL",
    }.get(key, key)
