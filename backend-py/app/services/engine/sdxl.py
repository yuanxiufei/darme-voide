"""**SDXL UNet** —— 自研的「图」扩散主干（卷积 UNet + 交叉注意力）。

## 为什么有这一层

`image_generation.py` 的 `provider=engine` 之前只能拿 H3（视频模型）凑静态图 ✗。图这一路要「完全自主」，
而引擎里 VAE / 调度器 / 采样器都已在位 ✓，缺的就是 SDXL 的 UNet 主干 —— 本模块补它。

## 结构事实（逐条有出处，不猜 ✓）

**逐块布局**取自**本机真权重头**（`sd_xl_base_1.0.safetensors` 的 `model.diffusion_model.*`：
**1680 键 / 2,567,463,684 参数**；只读 safetensors 头、不载张量 ✓）：

- `input_blocks`：`0.0`=conv_in；`1.0`/`2.0`=ResBlock(320 无注意力)；`3.0`=下采样；
  `4.0`/`5.0`=ResBlock(320→640) + `.1`=注意力(640，**深度 2**)；`6.0`=下采样；
  `7.0`/`8.0`=ResBlock(640→1280) + `.1`=注意力(1280，**深度 10**)
- `middle_block`：`.0`/`.2`=ResBlock(1280)、`.1`=注意力(1280，**深度 10**)
- `output_blocks`（自上采样序）：`0..2`=1280 层（注意力 **10**）、`3..5`=640 层（注意力 **2**）、
  `6..8`=320 层（**无注意力**）；每层最后一个带 `conv` 上采样
- `out.0`/`out.2`：GroupNorm(32, 320) → SiLU → Conv2d(320→4)
- `time_embed.0/.2`：Linear(320→1280) → SiLU → Linear(1280→1280)
- `label_emb.0.0/.2`：Linear(**2816** → 1280) → SiLU → Linear(1280→1280)

⇒ 上采样路径每个 `output_blocks` 弹一次 skip ⇒ `len(input_blocks) == len(output_blocks)`（各 9）✓。

**归一化 / 注意力 / 前馈口径**取自 `ComfyUI`（只读事实，未抄代码 ✗）：

- `ResBlock`：`in_layers.0` GroupNorm → `.1` SiLU → `.2` Conv3x3；`emb_layers.0` SiLU →
  `.1` Linear(time_embed_dim → out)；`out_layers.0` GroupNorm → `.1` SiLU → `.2` Dropout → `.3` Conv3x3；
  通道不等时 `skip_connection` 是 **1x1** Conv（加在最后）—— `comfy/ldm/modules/diffusionmodules/openaimodel.py`
- `SpatialTransformer`：`norm` = GroupNorm(32, ch, **eps=1e-6**) → `proj_in` → N × BasicTransformerBlock →
  `proj_out` → 残差加回；`use_linear_in_transformer=True` ⇒ 两个 proj 都是 **Linear**（不是 1x1 Conv）
  —— `comfy/ldm/modules/attention.py`（SpatialTransformer）、`comfy/supported_models.py`（SDXL 段）
- `BasicTransformerBlock`：`norm1/norm2/norm3` = **LayerNorm**；`attn1` 自注意力 → 残差；
  `norm2` → `attn2` 交叉（`context_dim=2048`）→ 残差；`norm3` → **GEGLU** 前馈 → 残差
  —— `comfy/ldm/modules/attention.py`（GEGLU / FeedForward / BasicTransformerBlock）
- `CrossAttention`：`to_q`/`to_k`/`to_v` **无偏置**、`to_out.0` 带偏置（`to_out.1` 是 Dropout）
  —— 同上
- 时间嵌入：**cos 在前、sin 在后**（= diffusers 的 `flip_sin_to_cos=True` / `freq_shift=0`）
  —— `comfy/ldm/modules/diffusionmodules/util.py`（`timestep_embedding`）
- 头数：`num_head_channels=64` ⇒ 每层头数 = 通道 / 64、head_dim = 64（1280 层 ⇒ 20 头）
  —— `comfy/supported_models_base.py`（BASE 默认）、`comfy/model_detection.py`（SDXL 段）
- `out.2` **零初始化**（LDM 的 `zero_module`）⇒ 未训练的新模型输出恒 0 ✓
  —— `openaimodel.py`（UNetModel 构造 `out` 处）

## 不猜·边界

- **只实现 SDXL 这一套排布** ✓：索引/深度按上表逐键对齐真权重；别的变体（refiner、量化改版）不认 ✗，
  配置一不符就**响亮报错**而不是猜 ✓。
- 条件向量 `adm`（**2816** = 池化文本 1280 + 6 个时间 id × 256）由调用方拼好传入 ✓（真权重 `label_emb.0.0`
  的输入就是 2816 ✓）；本模块另给 `build_adm()` 纯函数按 LDM 口径算 ✓。
- 只做**架构 + 前向** ✓；文本编码器、采样循环、VAE 解码不在本模块（引擎另有 ✓）。

## 验证口径

`tests/engine_sdxl_test.py`：缩小版走**同一条前向** ✓；本机有真权重时**逐键逐形状**核对 ✓
（键集/形状/参数总量都要与真头一致 ⇒ 布局读错必红 ✓）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

try:  # pragma: no cover - 环境相关
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[assignment,misc]
    _HAS_TORCH = False

__all__ = [
    "SDXL_UNET_KEY_PREFIX",
    "SDXLUnet",
    "SdxlUnetConfig",
    "SdxlUnetError",
    "build_adm",
    "build_sdxl_unet",
    "has_torch",
    "load_sdxl_unet_state_dict",
    "timestep_embedding",
]

#: SDXL 检查点里 UNet 那半的前缀（真权重实测 ✓）。
SDXL_UNET_KEY_PREFIX = "model.diffusion_model."

#: 时间 id 个数（SDXL 官方口径：height / width / crop_top / crop_left / target_h / target_w ✓）。
ADM_TIME_IDS = 6

#: 每个时间 id 的正弦嵌入维度 ✓（6 × 256 + 1280 = 2816 ✓）。
ADM_TIME_DIM = 256


class SdxlUnetError(RuntimeError):
    """SDXL UNet 的配置/装载/前向出错（不静默兜底 ✗）。"""


def has_torch() -> bool:
    """本进程里 torch 是否可用（无 torch 的环境要能 import 本模块 ✓）。"""
    return _HAS_TORCH


@dataclass(frozen=True)
class SdxlUnetConfig:
    """SDXL UNet 的全部架构参数 —— **逐个显式**，默认值即官方 SDXL base ✓。

    每个字段都在真权重里核过（见模块头部表 ✓）；改动任一字段都会改变键集/形状 ⇒ 装载时会立刻暴露 ✓。
    """

    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 320
    channel_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    context_dim: int = 2048
    adm_in_channels: int = 2816
    num_head_channels: int = 64
    #: 下行每层每个 ResBlock 之后的注意力深度（6 个单元 ⇒ 320 层 0 / 640 层 2 / 1280 层 10 ✓）。
    transformer_depth: tuple[int, ...] = (0, 0, 2, 2, 10, 10)
    #: 中间块注意力深度 ✓。
    transformer_depth_middle: int = 10
    #: 上行每层每个 ResBlock 之后的注意力深度（上行序：1280 层 10 / 640 层 2 / 320 层 0 ✓）。
    transformer_depth_output: tuple[int, ...] = (10, 10, 10, 2, 2, 2, 0, 0, 0)
    norm_num_groups: int = 32
    #: ResBlock 里 GroupNorm 的 eps（LDM 用 torch 默认 ✓）。
    norm_eps_block: float = 1e-5
    #: SpatialTransformer 的 `norm` 的 eps（参考实现显式给 1e-6 ✓）。
    norm_eps_spatial: float = 1e-6
    norm_eps_out: float = 1e-5
    dropout: float = 0.0
    use_linear_in_transformer: bool = True
    #: time_embed_dim = model_channels × 该倍数（SDXL = 1280 ✓）。
    time_embed_mult: int = 4
    #: 是否按 LDM 口径把 `out.2` 零初始化 ✓（真权重自带数值，开关只影响新建模型 ✓）。
    zero_init_out: bool = True

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise SdxlUnetError("in_channels / out_channels 必须为正 ✓")
        if self.model_channels <= 0:
            raise SdxlUnetError("model_channels 必须为正 ✓")
        if not self.channel_mult:
            raise SdxlUnetError("channel_mult 不能为空 ✓")
        if self.num_res_blocks <= 0:
            raise SdxlUnetError("num_res_blocks 必须为正 ✓")
        if not self.use_linear_in_transformer:
            raise SdxlUnetError("只实现了 use_linear=True 的 SDXL 排布（proj 是 Linear）✗")
        expect_down = self.num_res_blocks * len(self.channel_mult)
        if len(self.transformer_depth) != expect_down:
            raise SdxlUnetError(
                f"transformer_depth 长度应为 num_res_blocks×层数={expect_down}，实际 {len(self.transformer_depth)} ✗"
            )
        expect_up = (self.num_res_blocks + 1) * len(self.channel_mult)
        if len(self.transformer_depth_output) != expect_up:
            raise SdxlUnetError(
                f"transformer_depth_output 长度应为 (num_res_blocks+1)×层数={expect_up}，"
                f"实际 {len(self.transformer_depth_output)} ✗"
            )
        for level, mult in enumerate(self.channel_mult):
            channels = self.model_channels * mult
            if channels % self.norm_num_groups:
                raise SdxlUnetError(
                    f"第 {level} 层通道 {channels} 不能被 norm_num_groups={self.norm_num_groups} 整除 ✗"
                )
            if self.attends(level) and channels % self.num_head_channels:
                raise SdxlUnetError(
                    f"第 {level} 层通道 {channels} 不能被 num_head_channels={self.num_head_channels} 整除 ✗"
                )
        if self.adm_in_channels <= 0:
            raise SdxlUnetError("adm_in_channels 必须为正 ✓")

    # ---- 派生量（都由显式字段算出来，不再另存一份 ✓） ----

    @property
    def time_embed_dim(self) -> int:
        """时间嵌入维度（SDXL = 1280 ✓）。"""
        return self.model_channels * self.time_embed_mult

    @property
    def num_levels(self) -> int:
        """下/上采样层数（SDXL = 3 ✓）。"""
        return len(self.channel_mult)

    def channels_at(self, level: int) -> int:
        """第 level 层的通道数（自上而下数，第 0 层最浅 ✓）。"""
        return self.model_channels * self.channel_mult[level]

    def attends(self, level: int) -> bool:
        """该层是否带注意力（SDXL：320 层不带 ✓）。"""
        return any(
            self.transformer_depth[index] > 0
            for index in range(level * self.num_res_blocks, (level + 1) * self.num_res_blocks)
        )

    def heads_at(self, channels: int) -> int:
        """该通道数下的注意力头数（= 通道 / num_head_channels ✓）。"""
        return channels // self.num_head_channels

    def describe(self) -> str:
        """一行摘要，给日志/自检用（不参与判据 ✓）。"""
        return (
            f"SDXL UNet {self.model_channels}×{self.channel_mult} 层={self.num_levels} "
            f"res={self.num_res_blocks} ctx={self.context_dim} adm={self.adm_in_channels} "
            f"头={self.num_head_channels}/通道 深度={self.transformer_depth}"
        )


if _HAS_TORCH:

    def timestep_embedding(timesteps: "Tensor", dim: int, max_period: float = 10000.0) -> "Tensor":
        """正弦时间嵌入：**cos 在前、sin 在后** ✓（= diffusers 的 flip_sin_to_cos=True / freq_shift=0 ✓）。

        出处：`ComfyUI/comfy/ldm/modules/diffusionmodules/util.py`（`timestep_embedding`）✓。
        奇数 dim 时末位补 0 ✓（同参考实现 ✓）。
        """
        if dim <= 0:
            raise SdxlUnetError("dim 必须为正 ✓")
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps.reshape(-1)[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def build_adm(text_embeds: "Tensor", time_ids: "Tensor") -> "Tensor":
        """拼 SDXL 的 `adm` 条件向量：(B, 1280) 池化文本 ++ 6 × 256 的时间 id 嵌入 ⇒ (B, 2816) ✓。

        `time_ids` 顺序是 SDXL 官方口径：`[height, width, crop_top, crop_left, target_height, target_width]` ✓
        —— 出处 `ComfyUI/comfy/model_base.py`（`SDXL.encode_adm`：六个 id 各自过同一个嵌入器，
        再与池化文本拼在一起 ✓）。
        """
        if time_ids.ndim != 2 or time_ids.shape[1] != ADM_TIME_IDS:
            raise SdxlUnetError(f"time_ids 形状应为 (B, {ADM_TIME_IDS})，实际 {tuple(time_ids.shape)} ✗")
        if text_embeds.ndim != 2:
            raise SdxlUnetError(f"text_embeds 形状应为 (B, C)，实际 {tuple(text_embeds.shape)} ✗")
        embeds = [timestep_embedding(time_ids[:, index], ADM_TIME_DIM) for index in range(ADM_TIME_IDS)]
        return torch.cat([text_embeds, *embeds], dim=-1)


if _HAS_TORCH:

    def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> "nn.Module":
        """3x3 卷积（padding=1 ✓，同参考实现 ✓）。"""
        return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)

    class _BlockSequential(nn.Sequential):
        """把本模块的三种层串起来：ResBlock 吃 (x, emb)，其余吃 (x, context) ✓。

        等价于参考实现里的 `TimestepEmbedSequential`（按层类型分派 ✓，但只认本模块这三种层 ✓）。
        """

        def forward(
            self,
            x: "Tensor",
            emb: "Tensor | None" = None,
            context: "Tensor | None" = None,
            output_shape: "tuple[int, int] | None" = None,
        ) -> "Tensor":
            for layer in self:
                if isinstance(layer, _ResBlock):
                    x = layer(x, emb)
                elif isinstance(layer, _SpatialTransformer):
                    x = layer(x, context)
                elif isinstance(layer, _Upsample):
                    x = layer(x, output_shape)
                else:  # conv_in / 下采样 —— 只吃 x ✓
                    x = layer(x)
            return x

    class _ResBlock(nn.Module):
        """残差块：`in_layers` → +时间嵌入 → `out_layers` → +`skip_connection` ✓（键名与真权重一致 ✓）。"""

        def __init__(
            self,
            channels: int,
            emb_channels: int,
            out_channels: int,
            *,
            dropout: float,
            eps: float,
            groups: int,
        ) -> None:
            super().__init__()
            self.in_layers = nn.Sequential(
                nn.GroupNorm(groups, channels, eps=eps), nn.SiLU(), _conv3x3(channels, out_channels)
            )
            self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, out_channels))
            self.out_layers = nn.Sequential(
                nn.GroupNorm(groups, out_channels, eps=eps),
                nn.SiLU(),
                nn.Dropout(dropout),
                _conv3x3(out_channels, out_channels),
            )
            # ⚠️ 通道相等时参考实现是 Identity ⇒ 真权重里没有 skip_connection 键 ✓（键数 10 vs 12 ✓）。
            self.skip_connection: "nn.Module" = (
                nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()
            )

        def forward(self, x: "Tensor", emb: "Tensor") -> "Tensor":
            h = self.in_layers(x)
            emb_out = self.emb_layers(emb)
            while emb_out.ndim < h.ndim:
                emb_out = emb_out[..., None]
            h = h + emb_out
            h = self.out_layers(h)
            return self.skip_connection(x) + h

    class _Downsample(nn.Module):
        """`op`：stride 2 的 3x3 卷积 ✓（键名 `op.weight` / `op.bias` ✓）。"""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.op = _conv3x3(channels, channels, stride=2)

        def forward(self, x: "Tensor") -> "Tensor":
            return self.op(x)

    class _Upsample(nn.Module):
        """`conv`：最近邻放大 2 倍再 3x3 卷积 ✓（键名 `conv.weight` / `conv.bias` ✓）。

        ⚠️ 给了 `output_shape` 就按它取尺寸（**不是**无脑 ×2 ✓）：奇数边长下采样后
        （13 → 7）再 ×2 会得到 14 ≠ 13 ⇒ 与 skip 拼不上 ✓；参考实现同样在这里按
        「下一个 skip 的尺寸」插值 ✓（`openaimodel.py` 的 `Upsample.forward(x, output_shape)`）✓。
        """

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv = _conv3x3(channels, channels)

        def forward(self, x: "Tensor", output_shape: "tuple[int, int] | None" = None) -> "Tensor":
            if output_shape is None:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                x = F.interpolate(x, size=tuple(output_shape), mode="nearest")
            return self.conv(x)

    class _GEGLU(nn.Module):
        """GEGLU：`proj` 出两倍宽，切两半，一半过 gelu 当门 ✓（键名 `proj.weight` ✓）。"""

        def __init__(self, dim_in: int, dim_out: int) -> None:
            super().__init__()
            self.proj = nn.Linear(dim_in, dim_out * 2)

        def forward(self, x: "Tensor") -> "Tensor":
            value, gate = self.proj(x).chunk(2, dim=-1)
            return value * F.gelu(gate)

    class _FeedForward(nn.Module):
        """前馈：GEGLU → Dropout → Linear ✓（键名 `net.0.proj` / `net.2` ✓）。"""

        def __init__(self, dim: int, dim_out: int, *, mult: int = 4, dropout: float) -> None:
            super().__init__()
            inner = int(dim * mult)
            self.net = nn.Sequential(_GEGLU(dim, inner), nn.Dropout(dropout), nn.Linear(inner, dim_out))

        def forward(self, x: "Tensor") -> "Tensor":
            return self.net(x)

    class _CrossAttention(nn.Module):
        """注意力：`to_q`/`to_k`/`to_v` 无偏置、`to_out.0` 带偏置 ✓（同参考实现 ✓）。"""

        def __init__(
            self,
            query_dim: int,
            *,
            context_dim: "int | None",
            heads: int,
            dim_head: int,
            dropout: float,
        ) -> None:
            super().__init__()
            inner_dim = heads * dim_head
            key_dim = query_dim if context_dim is None else context_dim
            self.heads = heads
            self.dim_head = dim_head
            self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(key_dim, inner_dim, bias=False)
            self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        def forward(self, x: "Tensor", context: "Tensor | None" = None) -> "Tensor":
            context = x if context is None else context
            batch, tokens = x.shape[0], x.shape[1]
            query = self.to_q(x)
            key = self.to_k(context)
            value = self.to_v(context)

            def split(tensor: "Tensor") -> "Tensor":
                shape = (tensor.shape[0], tensor.shape[1], self.heads, self.dim_head)
                return tensor.view(shape).transpose(1, 2)

            out = F.scaled_dot_product_attention(split(query), split(key), split(value))
            out = out.transpose(1, 2).reshape(batch, tokens, self.heads * self.dim_head)
            return self.to_out(out)

    class _BasicTransformerBlock(nn.Module):
        """`norm1`→自注意力→残差；`norm2`→交叉注意力→残差；`norm3`→GEGLU 前馈→残差 ✓（三层都是 LayerNorm ✓）。"""

        def __init__(
            self,
            dim: int,
            *,
            heads: int,
            dim_head: int,
            context_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn1 = _CrossAttention(dim, context_dim=None, heads=heads, dim_head=dim_head, dropout=dropout)
            self.norm2 = nn.LayerNorm(dim)
            self.attn2 = _CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.norm3 = nn.LayerNorm(dim)
            self.ff = _FeedForward(dim, dim, dropout=dropout)

        def forward(self, x: "Tensor", context: "Tensor") -> "Tensor":
            x = self.attn1(self.norm1(x)) + x
            x = self.attn2(self.norm2(x), context) + x
            return self.ff(self.norm3(x)) + x

    class _SpatialTransformer(nn.Module):
        """空间注意力块：`norm` → `proj_in` → N × Transformer → `proj_out`，再残差加回 ✓。

        ⚠️ `norm` 的 eps 是 **1e-6**（与 ResBlock 的 1e-5 不同 ✓，参考实现显式给的 ✓）。
        ⚠️ 参考实现把 `proj_out` 声明成 `Linear(in_channels, inner_dim)`（写反了，但 SDXL 里 inner_dim == in_channels
        ⇒ 形状一样 ✓）；本实现按语义写成 `Linear(inner_dim, in_channels)` ✓。
        """

        def __init__(
            self,
            dim: int,
            *,
            depth: int,
            context_dim: int,
            num_head_channels: int,
            groups: int,
            eps: float,
            dropout: float,
        ) -> None:
            super().__init__()
            if depth <= 0:
                raise SdxlUnetError("depth 必须为正（不带注意力的层不该构造本块）✗")
            if dim % num_head_channels:
                raise SdxlUnetError(f"通道 {dim} 不能被 num_head_channels={num_head_channels} 整除 ✗")
            self.norm = nn.GroupNorm(groups, dim, eps=eps)
            self.proj_in = nn.Linear(dim, dim)
            self.transformer_blocks = nn.ModuleList(
                [
                    _BasicTransformerBlock(
                        dim,
                        heads=dim // num_head_channels,
                        dim_head=num_head_channels,
                        context_dim=context_dim,
                        dropout=dropout,
                    )
                    for _ in range(depth)
                ]
            )
            self.proj_out = nn.Linear(dim, dim)

        def forward(self, x: "Tensor", context: "Tensor") -> "Tensor":
            residual = x
            batch, channels, height, width = x.shape
            h = self.norm(x).movedim(1, 3).flatten(1, 2)  # (B, C, H, W) → (B, H·W, C) ✓
            h = self.proj_in(h)
            for block in self.transformer_blocks:
                h = block(h, context)
            h = self.proj_out(h)
            h = h.reshape(batch, height, width, channels).movedim(3, 1)
            return h + residual


if _HAS_TORCH:

    class SDXLUnet(nn.Module):
        """SDXL 的 UNet 主干 —— **逐键对齐官方权重** ✓（构造顺序 == 键索引顺序 ✓）。

        - `input_blocks`：9 项（`0` = conv_in ✓；下采样是**独立一项**、只含 `.op` ✓）
        - `middle_block`：3 项（ResBlock / 注意力 / ResBlock ✓）
        - `output_blocks`：9 项（**每层的最后一项**里再带一个 `_Upsample` ✓，即真权重的 `.2.conv` ✓）
        - 上行每个 ResBlock 吃的是 **`cat([h, skip], dim=1)`** ⇒ 它的 `in_layers.2` 输入维是
          **当前通道 + skip 通道**（真权重实测：`output_blocks.0.0.in_layers.2` = (1280, **2560**) ✓、
          `output_blocks.8.0.in_layers.2` = (320, **640**) ✓）—— ⚠️ 不是只吃 skip ✗。
        - ⚠️ `transformer_depth_output` 本模块按**上行顺序**存（1280 层在前 ✓），
          与 ComfyUI `supported_models.py` 里那份**反着写**的 `[0,0,0,2,2,2,10,10,10]` 是同一条事实的两种写法 ✓
          （它那边是从尾部 `pop()` 取 ✓）。
        """

        def __init__(self, config: "SdxlUnetConfig | None" = None) -> None:
            super().__init__()
            if config is None:
                config = SdxlUnetConfig()
            if not isinstance(config, SdxlUnetConfig):
                raise SdxlUnetError(f"config 应是 SdxlUnetConfig，实际 {type(config).__name__} ✗")
            self.config = config

            time_embed_dim = config.time_embed_dim
            groups = config.norm_num_groups
            dropout = config.dropout

            # `time_embed.0` / `.2` ⇒ 中间那个 SiLU 占索引 1 ✓
            self.time_embed = nn.Sequential(
                nn.Linear(config.model_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            # 真权重键是 `label_emb.0.0` / `label_emb.0.2` ⇒ 外层 Sequential 里再套一层 ✓
            self.label_emb = nn.Sequential(
                nn.Sequential(
                    nn.Linear(config.adm_in_channels, time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, time_embed_dim),
                )
            )

            def res_block(in_channels: int, out_channels: int) -> "_ResBlock":
                return _ResBlock(
                    in_channels,
                    time_embed_dim,
                    out_channels,
                    dropout=dropout,
                    eps=config.norm_eps_block,
                    groups=groups,
                )

            def spatial(channels: int, depth: int) -> "_SpatialTransformer":
                return _SpatialTransformer(
                    channels,
                    depth=depth,
                    context_dim=config.context_dim,
                    num_head_channels=config.num_head_channels,
                    groups=groups,
                    eps=config.norm_eps_spatial,
                    dropout=dropout,
                )

            # ---- 下行：conv_in → 每层 num_res_blocks 个 ResBlock(+可选注意力) → 层间下采样 ----
            input_blocks: list["nn.Module"] = [
                _BlockSequential(_conv3x3(config.in_channels, config.model_channels))
            ]
            #: 一路记下每个块的输出通道 ⇒ 上行按**后进先出**弹出来拼 ✓（与参考实现同一套账 ✓）
            skip_channels: list[int] = [config.model_channels]
            channels = config.model_channels
            cursor = 0
            for level in range(config.num_levels):
                level_channels = config.channels_at(level)
                for _ in range(config.num_res_blocks):
                    layers: list["nn.Module"] = [res_block(channels, level_channels)]
                    channels = level_channels
                    depth = config.transformer_depth[cursor]
                    cursor += 1
                    if depth > 0:
                        layers.append(spatial(channels, depth))
                    input_blocks.append(_BlockSequential(*layers))
                    skip_channels.append(channels)
                if level != config.num_levels - 1:
                    input_blocks.append(_BlockSequential(_Downsample(channels)))
                    skip_channels.append(channels)
            self.input_blocks = nn.ModuleList(input_blocks)

            # ---- 中间块：ResBlock → 注意力 → ResBlock ----
            middle: list["nn.Module"] = [res_block(channels, channels)]
            if config.transformer_depth_middle > 0:
                middle.append(spatial(channels, config.transformer_depth_middle))
            middle.append(res_block(channels, channels))
            self.middle_block = _BlockSequential(*middle)

            # ---- 上行：弹 skip → `cat` → ResBlock(+可选注意力) → 层间上采样 ----
            output_blocks: list["nn.Module"] = []
            remaining = list(skip_channels)
            cursor = 0
            for level in reversed(range(config.num_levels)):
                level_channels = config.channels_at(level)
                for index in range(config.num_res_blocks + 1):
                    if not remaining:
                        raise SdxlUnetError("skip 通道栈提前空了 ⇒ 上下行块数不配对 ✗")
                    jump = remaining.pop()
                    # ⚠️ 输入维是 ch + skip（拼接），不是 skip ✗
                    layers = [res_block(channels + jump, level_channels)]
                    channels = level_channels
                    depth = config.transformer_depth_output[cursor]
                    cursor += 1
                    if depth > 0:
                        layers.append(spatial(channels, depth))
                    if level > 0 and index == config.num_res_blocks:
                        layers.append(_Upsample(channels))
                    output_blocks.append(_BlockSequential(*layers))
            if remaining:
                raise SdxlUnetError(f"skip 通道栈还剩 {len(remaining)} 项没弹 ⇒ 布局算错 ✗")
            self.output_blocks = nn.ModuleList(output_blocks)

            self.out = nn.Sequential(
                nn.GroupNorm(groups, channels, eps=config.norm_eps_out),
                nn.SiLU(),
                _conv3x3(channels, config.out_channels),
            )
            if config.zero_init_out:
                # LDM 的 `zero_module`：新建模型的输出恒 0 ✓（真权重自带数值，不受影响 ✓）
                nn.init.zeros_(self.out[2].weight)  # type: ignore[arg-type]
                nn.init.zeros_(self.out[2].bias)  # type: ignore[arg-type]

        # 给自检/日志用：逐块通道与注意力深度 ✓（不参与判据 ✓）
        def structure(self) -> list[str]:
            """把实际装出来的块排布列出来（键索引顺序 ✓）—— 供自检与排障 ✓。"""
            rows: list[str] = []
            for family, blocks in (
                ("input_blocks", self.input_blocks),
                ("middle_block", [self.middle_block]),
                ("output_blocks", self.output_blocks),
            ):
                for index, block in enumerate(blocks):
                    for sub, layer in enumerate(block):
                        depth = len(layer.transformer_blocks) if isinstance(layer, _SpatialTransformer) else 0
                        name = type(layer).__name__.lstrip("_")
                        rows.append(f"{family}.{index}.{sub} {name} 深度={depth}")
            return rows

        def forward(
            self,
            x: "Tensor",
            timesteps: "Tensor",
            context: "Tensor",
            adm: "Tensor | None" = None,
        ) -> "Tensor":
            """SDXL UNet 前向：`(B,4,H,W)` + `(B,)` 时间步 + `(B,N,2048)` 文本 ⇒ `(B,4,H,W)` ✓。

            `adm`（**2816** 维 ✓，见 `build_adm()`）给了就加到时间嵌入上 ✓（真权重有 `label_emb` ⇒
            SDXL base 必须给 ✓；不给也能算，但那是**另一条**前向 ✗，自检只当结构验证用 ✓）。
            """
            if x.ndim != 4:
                raise SdxlUnetError(f"输入应是 (B, C, H, W)，实际 {tuple(x.shape)} ✗")
            if context.ndim != 3:
                raise SdxlUnetError(f"context 应是 (B, N, C)，实际 {tuple(context.shape)} ✗")
            if context.shape[-1] != self.config.context_dim:
                raise SdxlUnetError(
                    f"context 末维应是 {self.config.context_dim}，实际 {context.shape[-1]} ✗"
                )
            if adm is not None and adm.shape[-1] != self.config.adm_in_channels:
                raise SdxlUnetError(
                    f"adm 末维应是 {self.config.adm_in_channels}，实际 {adm.shape[-1]} ✗"
                )

            emb = self.time_embed(
                timestep_embedding(timesteps, self.config.model_channels).to(x.dtype)
            )
            if adm is not None:
                emb = emb + self.label_emb(adm.to(emb.dtype))

            # 下行：**每个**块（含 conv_in 与下采样 ✓）的输出都存起来当 skip ✓（与参考实现同一套账 ✓）
            skips: list["Tensor"] = []
            h = x
            for block in self.input_blocks:
                h = block(h, emb, context)
                skips.append(h)

            h = self.middle_block(h, emb, context)

            for block in self.output_blocks:
                jump = skips.pop()
                if jump.shape[-2:] != h.shape[-2:]:
                    raise SdxlUnetError(
                        f"上行 {tuple(h.shape[-2:])} 与 skip {tuple(jump.shape[-2:])} 尺寸不符 ⇒ 拼不上 ✗"
                        f"（奇数边长请检查 latent 尺寸 ✓）"
                    )
                h = torch.cat([h, jump], dim=1)
                # 参考实现口径：把**下一个** skip 的尺寸交给块里的上采样 ✓（弹完再取 `[-1]` ✓）
                next_shape = tuple(skips[-1].shape[-2:]) if skips else None
                h = block(h, emb, context, output_shape=next_shape)

            return self.out(h)


def build_sdxl_unet(
    config: "SdxlUnetConfig | None" = None,
    *,
    device: "Any" = None,
    dtype: "Any" = None,
) -> "Any":
    """按配置建一个 SDXL UNet ✓（**随机初始化** ⇒ 要出图得先 `load_sdxl_unet_state_dict` ✓）。

    ⚠️ 没装 torch ⇒ **报错** ✗（本仓不静默降级 ✗，同 `upscale_net.build_upscaler` ✓）。
    """
    if not _HAS_TORCH:
        raise SdxlUnetError(
            "没装 torch ✗ ⇒ 建不了 SDXL UNet ✓（引擎的张量层需要 torch ✓；"
            "要纯结构信息请用 `SdxlUnetConfig.describe()` ✓）"
        )
    model = SDXLUnet(config)
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    return model


def load_sdxl_unet_state_dict(
    model: "Any",
    state_dict: "Mapping[str, Any]",
    *,
    strict: bool = True,
) -> list[str]:
    """把检查点里 **UNet 那一半**装进 `model` ✓，返回**未装载**的键（不属于 UNet 的那些 ✓）。

    - 带 `model.diffusion_model.` 前缀（官方 SDXL 检查点 ✓）会自动剥掉 ✓；
    - 没有这个前缀的键一律**原样跳过**并如实返回 ⇒ 调用方可以核对「文本编码器/VAE 各多少键」✓；
    - 一个前缀键都没有 ⇒ 说明这根本不是 SDXL 检查点 ⇒ **报错** ✗；
    - 键集/形状对不上 ⇒ **报错** ✗（⚠️ 不许 `strict=False` 糊过去 ✓，否则留下随机初始化的层 ✓✗）。
    """
    if not _HAS_TORCH:
        raise SdxlUnetError("没装 torch ✗ ⇒ 装不了权重 ✓")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise SdxlUnetError("state_dict 必须是非空映射 ✗")

    prefix_len = len(SDXL_UNET_KEY_PREFIX)
    stripped: dict[str, "Any"] = {}
    others: list[str] = []
    for key, value in state_dict.items():
        if key.startswith(SDXL_UNET_KEY_PREFIX):
            stripped[key[prefix_len:]] = value
        else:
            others.append(key)
    if not stripped:
        raise SdxlUnetError(
            f"没有任何以 {SDXL_UNET_KEY_PREFIX!r} 开头的键 ⇒ 这不是 SDXL 检查点 ✗"
        )

    try:
        report = model.load_state_dict(stripped, strict=bool(strict))
    except RuntimeError as error:
        raise SdxlUnetError(
            f"UNet 张量与架构**不符** ✗：{error} ✓ ⇒ 检查配置或检查点版本 ✓"
            f"（⚠️ 不许 `strict=False` 糊过去 ✗）"
        ) from error
    if report.missing_keys or report.unexpected_keys:
        raise SdxlUnetError(
            f"仍不对齐 ✗：缺 {list(report.missing_keys)[:5]} / 多 {list(report.unexpected_keys)[:5]} ✓"
        )
    return sorted(others)
