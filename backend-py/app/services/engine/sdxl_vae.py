"""**SDXL VAE**（本项目自己的实现 ✓ 真 ``torch.nn.Module`` ✓）—— 图像潜变量 ⇄ 像素（**2 维** ✓）。

## 为什么需要它

自研出图链路缺的最后一环 ✓：SDXL UNet 采出来的潜变量是 **4 通道 × (H/8) × (W/8)** ✓，
要变成能看的图**必须**过 VAE 解码器 ✓。本仓此前只有 **H3 的视频 VAE** ✓（``vae.py`` ✓ 3D ✓）
—— **服务不了 SDXL** ✗：结构、通道、缩放系数（``0.13025`` ✓）没一项是一回事 ✓✗。

## 结构事实（2026-09-26 从**本机真权重**逐键核出 ✓ —— 不是猜的 ✗）

``sd_xl_base_1.0.safetensors`` 里 ``first_stage_model.*`` 共 **248 键** ✓（只读头部量出 ✓ 不载张量 ✗）。
命名是 **CompVis/LDM 老式** ✓（``encoder.down.N.block.M`` / ``decoder.up.N.upsample.conv`` ✓），
**不是** diffusers 那套（``down_blocks.N.resnets.M`` ✗）—— 两套键名对不上 ✗，拿错名字装权重会一装就红 ✓。

| 位置 | 通道流 |
|---|---|
| 编码器 | 3 ⇒ conv_in 128 ⇒ down.0(2 块) ⇒ down.1 256 ⇒ down.2 512 ⇒ down.3(2 块 **不下采样**) ⇒ mid 512 ⇒ conv_out **8** |
| 解码器 | 4 ⇒ conv_in/post_quant 512 ⇒ mid 512 ⇒ up.3 ⇒ up.2 ⇒ up.1 **256** ⇒ up.0(3 块 **不上采样**) **128** ⇒ conv_out **3** |

两处**照抄键名最容易错**的地方 ✗✗：

1. **解码器键序与前向序相反** ✓：权重里 ``decoder.up.3`` 是**先算**的 ✓、``up.0`` 是**最后**的 ✓
   （``up.0.block.0.conv1`` 是 ``[128, 256, 3, 3]`` ✓ —— 按"0 在前"读，mid 的 512 通道进不去 ✓✗）；
2. **两侧块数不同** ✗：编码器每层 ``num_res_blocks``(=2) 块 ✓、解码器每层 **``+1``**(=3) 块 ✓
   （真权重里 ``decoder.up.0.block.2`` 在 ✓）⇒ 抄成一个数，键集立刻对不上 ✓。

## 明确的简化与边界

1. **只做 SDXL 这一套排布** ✓：配置不符就**响亮报错** ✗ 不猜（同 ``sdxl.py`` ✓）；
2. **不做 tiled 解码** ✗：大图一次性解码（显存不够就 OOM ✓ 不悄悄降级 ✗）；
3. **不做变分采样** ✓：``encode`` 回 ``(mean, logvar)`` ✓，「怎么取 z」是采样策略 ✓ 由调用方定 ✗。

## 验证口径

``tests/engine_sdxl_vae_test.py``：缩小版走**同一条前向** ✓（含 non-square ✓）；再拿**本机真权重头**
把 :func:`sdxl_vae_key_names` 与真键集**逐键比对** ✓ —— **248 键全等**才绿 ✓，错一处（块数 / 通道序 /
解码器键序 ✓）**必红** ✓。⚠️ 不做**数值**对拍 ✗（那要真张量 + 参考环境 ✓）：本模块负责**结构与口径** ✓。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:  # pragma: no cover - 环境相关
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _HAS_TORCH = False

__all__ = [
    "SDXL_VAE_FACTS",
    "SDXL_VAE_KEY_PREFIX",
    "SdxlVae",
    "SdxlVaeConfig",
    "SdxlVaeError",
    "build_sdxl_vae",
    "decode_latents",
    "has_torch",
    "load_sdxl_vae_state_dict",
    "sdxl_vae_key_names",
]

#: SDXL 检查点里 VAE 那半的前缀（真权重实测 ✓：``first_stage_model.*`` 恰 **248** 键 ✓）。
SDXL_VAE_KEY_PREFIX = "first_stage_model."

#: 核过的**事实**（供上层写文案/预检用 ✓ —— 与 :class:`SdxlVaeConfig` 的默认值同源 ✓）。
SDXL_VAE_FACTS: dict[str, Any] = {
    "checkpoint": "sd_xl_base_1.0.safetensors",
    "keyPrefix": SDXL_VAE_KEY_PREFIX,
    "keyCount": 248,
    "naming": "CompVis/LDM 老式（非 diffusers ✗）",
    "blockOutChannels": (128, 256, 512, 512),
    "decoderBlocksPerLevel": 3,
    "encoderBlocksPerLevel": 2,
    "scalingFactor": 0.13025,
    "spatialScale": 8,
    "note": "潜空间 4 通道 ✓；解码键序 ``up.3 → up.0`` 与前向一致 ✓（见模块头 ✓）",
}


class SdxlVaeError(RuntimeError):
    """SDXL VAE 的配置/装载/前向出错（**不静默兜底** ✗）。"""


def has_torch() -> bool:
    """本进程里 torch 是否可用（无 torch 的环境要能 import 本模块 ✓）。"""
    return _HAS_TORCH


@dataclass(frozen=True)
class SdxlVaeConfig:
    """SDXL VAE 的架构参数 —— **逐个显式** ✓，默认值即官方 SDXL base ✓（逐键核过 ✓）。"""

    in_channels: int = 3
    out_channels: int = 3
    #: 潜通道（解码器 ``conv_in`` 的输入 ✓ 真权重实测 ``[512, 4, 3, 3]`` ✓）。
    latent_channels: int = 4
    #: 每层通道（真权重实测 ✓）。
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    #: 每层 ResBlock 数 ✓（⚠️ 解码器每层用 ``+1`` 块 ✓ —— 见 ``decoder_blocks`` ✓）。
    layers_per_block: int = 2
    norm_num_groups: int = 32
    #: ``GroupNorm`` 的 eps（LDM ``Normalize`` 显式给 ``1e-6`` ✓，与 UNet 那两档不同 ✗）。
    norm_eps: float = 1e-6
    #: 潜变量缩放（真权重同名口径 ✓ 见 ``latent_formats`` 的 ``sdxl`` 规格 ✓）。
    scaling_factor: float = 0.13025
    shift_factor: float = 0.0

    def __post_init__(self) -> None:
        if not self.block_out_channels:
            raise SdxlVaeError("block_out_channels 不能为空 ✗")
        if any(int(value) <= 0 for value in self.block_out_channels):
            raise SdxlVaeError(f"block_out_channels 必须为正：{self.block_out_channels} ✗")
        if int(self.latent_channels) <= 0 or int(self.in_channels) <= 0 or int(self.out_channels) <= 0:
            raise SdxlVaeError("in/out/latent_channels 必须为正 ✗")
        if int(self.layers_per_block) < 1:
            raise SdxlVaeError(f"layers_per_block 必须 ≥1（收到 {self.layers_per_block} ✗）")
        if float(self.norm_eps) <= 0:
            raise SdxlVaeError(f"norm_eps 必须 >0（收到 {self.norm_eps} ✗）")
        if float(self.scaling_factor) == 0.0:
            raise SdxlVaeError("scaling_factor 不能为 0 ✗（解码要除以它 ✓）")
        # ⚠️ GroupNorm 要求通道能被组数整除 ✓ —— 不整除 ⇒ 真前向会在**第一层**炸 ✓，
        #    但那时报的是 aten 的内部消息 ✓✗ ⇒ 这里提前说清 ✓。
        # ⚠️ **只查 ``block_out_channels``** ✗：``in/out/latent`` 是 ``conv_in``/``conv_out``/
        #    ``quant_conv`` 的通道 ✓，那几个**不过 GroupNorm** ✓ —— 把 3 也拉进来查 ⇒
        #    **官方配置都建不起来** ✓✗（本条是照抄结构时容易顺手写错的 ✓）。
        for channels in self.block_out_channels:
            if channels % int(self.norm_num_groups) != 0:
                raise SdxlVaeError(
                    f"{channels} 不能被 norm_num_groups={self.norm_num_groups} 整除 ✗"
                    f"（GroupNorm 会炸 ✓ —— 提前说清比让它炸在 aten 里强 ✓）")

    @property
    def num_levels(self) -> int:
        """空间层数 ✓（= 通道表长度 ✓；每层一次下采样，最后一层不下 ✓）。"""
        return len(self.block_out_channels)

    @property
    def spatial_scale(self) -> int:
        """像素 ↔ 潜空间的**边长比** ✓。

        ⚠️ **是 ``2^(层数−1)``** ✗ 不是 ``2^层数`` ✓ —— 最后一层**不下采样** ✓（真权重口径 ✓：
        编码器 ``down.3`` 没有 ``downsample`` 键 ✓、解码器 ``up.0`` 没有 ``upsample`` 键 ✓）。
        """
        return 2 ** (self.num_levels - 1)

    @property
    def encoder_blocks(self) -> int:
        """编码器每层块数 ✓（= ``layers_per_block`` ✓）。"""
        return int(self.layers_per_block)

    @property
    def decoder_blocks(self) -> int:
        """解码器每层块数 ✓（= ``layers_per_block + 1`` ✓ —— LDM 口径 ✓ 真权重实测 ✓）。"""
        return int(self.layers_per_block) + 1

    @property
    def latent_channel_count(self) -> int:
        """编码器 ``conv_out`` 的输出通道 ✓（``2 × latent`` ✓ —— 均值 + 对数方差 ✓）。"""
        return 2 * int(self.latent_channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inChannels": int(self.in_channels), "outChannels": int(self.out_channels),
            "latentChannels": int(self.latent_channels),
            "blockOutChannels": list(self.block_out_channels),
            "layersPerBlock": int(self.layers_per_block),
            "decoderBlocksPerLevel": self.decoder_blocks,
            "normNumGroups": int(self.norm_num_groups), "normEps": float(self.norm_eps),
            "scalingFactor": float(self.scaling_factor), "shiftFactor": float(self.shift_factor),
            "spatialScale": self.spatial_scale,
        }


def sdxl_vae_key_names(config: "SdxlVaeConfig | None" = None) -> tuple[str, ...]:
    """按**结构**推出应当存在的**全部键名** ✓（**不含** ``first_stage_model.`` 前缀 ✓）。

    ⚠️ 这是**预检**用的 ✓：拿它跟真权重头逐个比 ✓ ⇒ 块数 / 通道序 / 解码器键序错一处就红 ✓
    （模块头那两处坑 ✓ 都会在这儿暴露 ✓）。**排序后**返回 ✓（稳定输出才便于比对 ✓）。
    """
    spec = config or SdxlVaeConfig()
    channels = tuple(int(value) for value in spec.block_out_channels)
    names: list[str] = []

    def resblock(prefix: str, in_ch: int, out_ch: int) -> None:
        for tail in ("norm1.weight", "norm1.bias", "conv1.weight", "conv1.bias",
                     "norm2.weight", "norm2.bias", "conv2.weight", "conv2.bias"):
            names.append(f"{prefix}.{tail}")
        if in_ch != out_ch:      # ⚠️ 仅当通道变化时才有 ✓（真权重里 down.0 没有 nin_shortcut ✓）
            names.extend((f"{prefix}.nin_shortcut.weight", f"{prefix}.nin_shortcut.bias"))

    def mid(prefix: str, depth: int) -> None:
        resblock(f"{prefix}.block_1", depth, depth)
        # attn：LDM 的 q/k/v/proj_out 是 **1×1 卷积**且**带偏置** ✓
        # （⚠️ 与 SDXL UNet 的 ``to_q`` 无偏置**不同** ✗ —— 别把两处口径串了 ✓）。
        names.extend((f"{prefix}.attn_1.norm.weight", f"{prefix}.attn_1.norm.bias"))
        for branch in ("q", "k", "v", "proj_out"):
            names.extend((f"{prefix}.attn_1.{branch}.weight", f"{prefix}.attn_1.{branch}.bias"))
        resblock(f"{prefix}.block_2", depth, depth)

    # ── 编码器（前向 = 键序 ✓：down.0 ⇒ down.3）────────────────────────────
    names.extend(("encoder.conv_in.weight", "encoder.conv_in.bias"))
    current = channels[0]
    for level in range(spec.num_levels):
        out_ch = channels[level]
        for block in range(spec.encoder_blocks):
            resblock(f"encoder.down.{level}.block.{block}", current, out_ch)
            current = out_ch
        if level != spec.num_levels - 1:      # 最后一层**不下采样** ✓
            names.extend((f"encoder.down.{level}.downsample.conv.weight",
                          f"encoder.down.{level}.downsample.conv.bias"))
    mid("encoder.mid", current)
    names.extend(("encoder.norm_out.weight", "encoder.norm_out.bias",
                  "encoder.conv_out.weight", "encoder.conv_out.bias"))

    # ── 解码器（⚠️ **键序与前向序相反** ✓✗：键 ``up.3`` 先算 ✓ 见模块头 ✓）─────
    last = channels[-1]
    names.extend(("decoder.conv_in.weight", "decoder.conv_in.bias"))
    mid("decoder.mid", last)
    current = last
    for level in reversed(range(spec.num_levels)):
        out_ch = channels[level]
        for block in range(spec.decoder_blocks):
            resblock(f"decoder.up.{level}.block.{block}", current, out_ch)   # 键名用 **level** ✓
            current = out_ch
        if level != 0:                        # up.0 **不上采样** ✓（up.1/2/3 才上 ✓）
            names.extend((f"decoder.up.{level}.upsample.conv.weight",
                          f"decoder.up.{level}.upsample.conv.bias"))
    names.extend(("decoder.norm_out.weight", "decoder.norm_out.bias",
                  "decoder.conv_out.weight", "decoder.conv_out.bias"))
    names.extend(("post_quant_conv.weight", "post_quant_conv.bias",
                  "quant_conv.weight", "quant_conv.bias"))
    return tuple(sorted(names))


# ══════════════════════════════════════════════════════════════════════════
# 网络（**真 torch 模块** ✓ —— 键名与真权重逐键对齐 ✓）
# ══════════════════════════════════════════════════════════════════════════
def _normalize(channels: int, config: SdxlVaeConfig) -> Any:
    """LDM ``Normalize`` ✓：``GroupNorm(affine=True, eps=1e-6)`` ✓。"""
    return nn.GroupNorm(num_groups=int(config.norm_num_groups), num_channels=int(channels),
                        eps=float(config.norm_eps), affine=True)


def _zero_module(module: Any) -> Any:
    """LDM ``zero_module`` ✓：权重/偏置清零 ✓ ⇒ 新模型的 ``attn_1`` 是**恒等** ✓（可自检 ✓）。"""
    for parameter in module.parameters():
        with torch.no_grad():
            parameter.zero_()
    return module


class _ResnetBlock(nn.Module):
    """ResBlock ✓：``norm1 ⇒ silu ⇒ conv1 ⇒ norm2 ⇒ silu ⇒ conv2`` + shortcut ✓（真权重键名如此 ✓）。"""

    def __init__(self, in_channels: int, out_channels: int, config: SdxlVaeConfig) -> None:
        super().__init__()
        self.norm1 = _normalize(in_channels, config)
        self.conv1 = nn.Conv2d(int(in_channels), int(out_channels), 3, stride=1, padding=1)
        self.norm2 = _normalize(out_channels, config)
        self.conv2 = nn.Conv2d(int(out_channels), int(out_channels), 3, stride=1, padding=1)
        self.nin_shortcut = (nn.Conv2d(int(in_channels), int(out_channels), 1, stride=1, padding=0)
                             if int(in_channels) != int(out_channels) else None)

    def forward(self, x: Any) -> Any:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class _AttnBlock(nn.Module):
    """LDM ``AttnBlock`` ✓：**单头全空间注意力** ✓（``q·kᵀ/√C`` 在 ``H*W`` 上做 ✓）。"""

    def __init__(self, channels: int, config: SdxlVaeConfig) -> None:
        super().__init__()
        self.norm = _normalize(channels, config)
        for branch in ("q", "k", "v"):
            setattr(self, branch, nn.Conv2d(int(channels), int(channels), 1, stride=1, padding=0))
        self.proj_out = _zero_module(
            nn.Conv2d(int(channels), int(channels), 1, stride=1, padding=0))

    def forward(self, x: Any) -> Any:
        h = self.norm(x)
        query, key, value = self.q(h), self.k(h), self.v(h)
        batch, channels, height, width = query.shape
        spots = height * width
        query = query.reshape(batch, channels, spots).permute(0, 2, 1)
        key = key.reshape(batch, channels, spots)
        weights = F.softmax(torch.bmm(query, key) * (channels ** -0.5), dim=2).permute(0, 2, 1)
        value = value.reshape(batch, channels, spots)
        out = torch.bmm(value, weights).reshape(batch, channels, height, width)
        return x + self.proj_out(out)


class _Downsample(nn.Module):
    """LDM ``Downsample`` ✓：**右下补零**后 ``stride=2`` 卷积 ✓。

    ⚠️ 是 ``F.pad((0,1,0,1))`` **不是** ``padding=1`` ✗ —— 后者会多算半格 ✓（尺寸对不上 ⇒ 自检必红 ✓）。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(int(channels), int(channels), 3, stride=2, padding=0)

    def forward(self, x: Any) -> Any:
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0))


class _Upsample(nn.Module):
    """LDM ``Upsample`` ✓：``nearest`` 放大 2× 再 3×3 卷积 ✓。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(int(channels), int(channels), 3, stride=1, padding=1)

    def forward(self, x: Any) -> Any:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class _MidBlock(nn.Module):
    """mid ✓：``block_1 ⇒ attn_1 ⇒ block_2`` ✓（真权重键名如此 ✓）。"""

    def __init__(self, channels: int, config: SdxlVaeConfig) -> None:
        super().__init__()
        self.block_1 = _ResnetBlock(channels, channels, config)
        self.attn_1 = _AttnBlock(channels, config)
        self.block_2 = _ResnetBlock(channels, channels, config)

    def forward(self, x: Any) -> Any:
        return self.block_2(self.attn_1(self.block_1(x)))


class _Level(nn.Module):
    """一层（编码器的 ``down.N`` ✓ 或解码器的 ``up.N`` ✓）：``block`` 若干 + 可选 ``downsample``/``upsample`` ✓。"""

    def __init__(self, in_channels: int, out_channels: int, blocks: int,
                 config: SdxlVaeConfig, *, downsample: bool = False, upsample: bool = False) -> None:
        super().__init__()
        layers = []
        current = int(in_channels)
        for _ in range(int(blocks)):
            layers.append(_ResnetBlock(current, out_channels, config))
            current = int(out_channels)
        self.block = nn.ModuleList(layers)
        self.downsample = _Downsample(out_channels) if downsample else None
        self.upsample = _Upsample(out_channels) if upsample else None

    def forward(self, x: Any) -> Any:
        for layer in self.block:
            x = layer(x)
        if self.downsample is not None:
            x = self.downsample(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class _Encoder(nn.Module):
    """编码器 ✓：``conv_in ⇒ down.0 … down.3 ⇒ mid ⇒ norm_out ⇒ conv_out`` ✓（**含** mid ✓）。"""

    def __init__(self, config: SdxlVaeConfig) -> None:
        super().__init__()
        channels = tuple(int(value) for value in config.block_out_channels)
        self.conv_in = nn.Conv2d(int(config.in_channels), channels[0], 3, stride=1, padding=1)
        levels = []
        current = channels[0]
        for level in range(len(channels)):
            levels.append(_Level(current, channels[level], config.encoder_blocks, config,
                                 downsample=level != len(channels) - 1))
            current = channels[level]
        self.down = nn.ModuleList(levels)
        self.mid = _MidBlock(channels[-1], config)
        self.norm_out = _normalize(channels[-1], config)
        self.conv_out = nn.Conv2d(channels[-1], int(config.latent_channel_count),
                                  3, stride=1, padding=1)

    def forward(self, x: Any) -> Any:
        h = self.conv_in(x)
        for level in self.down:
            h = level(h)
        return self.conv_out(F.silu(self.norm_out(self.mid(h))))


class _Decoder(nn.Module):
    """解码器 ✓：``conv_in ⇒ mid ⇒ up.3 … up.0 ⇒ norm_out ⇒ conv_out`` ✓。

    ⚠️ 两个**照抄必错**的点都在这里 ✓：键名按 ``level`` 存（``up.3``/``up.0`` ✓）但**前向反着走** ✓；
    每层块数用 ``decoder_blocks``（= ``layers_per_block + 1`` ✓）而不是编码器那个数 ✓。
    """

    def __init__(self, config: SdxlVaeConfig) -> None:
        super().__init__()
        channels = tuple(int(value) for value in config.block_out_channels)
        last = channels[-1]
        self.conv_in = nn.Conv2d(int(config.latent_channels), last, 3, stride=1, padding=1)
        self.mid = _MidBlock(last, config)
        levels = []
        for level in range(len(channels)):
            # 入通道：**最先算的那层**（最后一个 level ✓）从 mid 拿 last ✓；其余从**上一层**拿 ✓
            in_ch = last if level == len(channels) - 1 else channels[level + 1]
            levels.append(_Level(in_ch, channels[level], config.decoder_blocks, config,
                                 upsample=level != 0))
        self.up = nn.ModuleList(levels)
        self.norm_out = _normalize(channels[0], config)
        self.conv_out = nn.Conv2d(channels[0], int(config.out_channels), 3, stride=1, padding=1)

    def forward(self, x: Any) -> Any:
        h = self.mid(self.conv_in(x))
        for level in reversed(self.up):     # ⚠️ **键序与前向序相反** ✓✗（见模块头 ✓）
            h = level(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class SdxlVae(nn.Module):
    """SDXL 的 ``AutoencoderKL`` ✓（编解码 + ``quant_conv`` / ``post_quant_conv`` ✓）。"""

    def __init__(self, config: "SdxlVaeConfig | None" = None) -> None:
        super().__init__()
        self.config = config or SdxlVaeConfig()
        self.encoder = _Encoder(self.config)
        self.decoder = _Decoder(self.config)
        depth = int(self.config.latent_channel_count)
        self.quant_conv = nn.Conv2d(depth, depth, 1, stride=1, padding=0)
        latent = int(self.config.latent_channels)
        self.post_quant_conv = nn.Conv2d(latent, latent, 1, stride=1, padding=0)

    def encode(self, images: Any) -> tuple[Any, Any]:
        """像素 ⇒ ``(mean, logvar)`` ✓（⚠️ **不**在这里采样 ✗ —— 采样策略是调用方的事 ✓）。"""
        mean, logvar = torch.chunk(self.quant_conv(self.encoder(images)), 2, dim=1)
        return mean, logvar

    def decode(self, latents: Any) -> Any:
        """潜变量 ⇒ 像素 ✓（⚠️ 调用方要**先除缩放系数** ✓ —— 走 :func:`decode_latents` 就不会漏 ✓）。"""
        return self.decoder(self.post_quant_conv(latents))


def build_sdxl_vae(config: "SdxlVaeConfig | None" = None, *,
                   device: Any = None, dtype: Any = None) -> Any:
    """按配置建一个 SDXL VAE ✓（**随机初始化** ⇒ 要出图得先 :func:`load_sdxl_vae_state_dict` ✓）。

    ⚠️ 没装 torch ⇒ **报错** ✗（本仓不静默降级 ✗，同 ``sdxl.build_sdxl_unet`` ✓）。
    """
    if not _HAS_TORCH:
        raise SdxlVaeError(
            "没装 torch ✗ ⇒ 建不了 SDXL VAE ✓（引擎的张量层需要 torch ✓；"
            "要纯结构信息请用 `SdxlVaeConfig().to_dict()` ✓）"
        )
    model = SdxlVae(config)
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    return model


def load_sdxl_vae_state_dict(model: Any, state_dict: "Mapping[str, Any]", *,
                             strict: bool = True) -> list[str]:
    """把检查点里 **VAE 那一半**装进 ``model`` ✓，返回**未装载**的键（不属于 VAE 的那些 ✓）。

    与 :func:`app.services.engine.sdxl.load_sdxl_unet_state_dict` **同一条口径** ✓：带
    ``first_stage_model.`` 前缀的自动剥掉 ✓；没有前缀的键**原样跳过**并如实返回 ✓（调用方据此核对
    "UNet / 文本编码器各多少键" ✓）；一个前缀键都没有 ⇒ **报错** ✗；键集或形状对不上 ⇒ **报错** ✗
    （⚠️ 不许 ``strict=False`` 糊过去 ✓ —— 那会留下**随机初始化的层** ✓✗）。
    """
    if not _HAS_TORCH:
        raise SdxlVaeError("没装 torch ✗ ⇒ 装不了权重 ✓")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise SdxlVaeError("state_dict 必须是非空映射 ✗")

    prefix_len = len(SDXL_VAE_KEY_PREFIX)
    stripped: dict[str, Any] = {}
    others: list[str] = []
    for key, value in state_dict.items():
        if key.startswith(SDXL_VAE_KEY_PREFIX):
            stripped[key[prefix_len:]] = value
        else:
            others.append(key)
    if not stripped:
        raise SdxlVaeError(f"没有任何以 {SDXL_VAE_KEY_PREFIX!r} 开头的键 ⇒ 这不是 SDXL 检查点 ✗")

    try:
        report = model.load_state_dict(stripped, strict=bool(strict))
    except RuntimeError as error:
        raise SdxlVaeError(
            f"VAE 张量与架构**不符** ✗：{error} ✓ ⇒ 检查配置或检查点版本 ✓"
            f"（⚠️ 不许 `strict=False` 糊过去 ✗）"
        ) from error
    if report.missing_keys or report.unexpected_keys:
        raise SdxlVaeError(
            f"仍不对齐 ✗：缺 {list(report.missing_keys)[:5]} / 多 {list(report.unexpected_keys)[:5]} ✓")
    return sorted(others)


def decode_latents(model: Any, latents: Any, *, config: "SdxlVaeConfig | None" = None,
                   to_unit: bool = True) -> Any:
    """潜变量 ⇒ 像素 ✓ —— **按 SDXL 口径先除缩放系数** ✓（``1/0.13025`` ✓）再解码 ✓。

    ⚠️ 这一步**最容易漏** ✗✗：漏了**不会报错** ✓，出来的是"有形状、有颜色、但整体发灰/过曝/糊"的图 ✓✗
    —— 正是本仓最忌讳的"看不出来"那类错 ✓。所以**解码必须走这里** ✓，别直接调 ``model.decode`` ✓。

    ``to_unit=True`` ⇒ 回 ``[0, 1]`` ✓（与 ``image_ops`` 同一口径 ✓）；否则回原始 ``[-1, 1]`` ✓。
    """
    if not _HAS_TORCH:
        raise SdxlVaeError("没装 torch ✗ ⇒ 解不了码 ✓")
    spec = config or getattr(model, "config", None) or SdxlVaeConfig()
    scale = float(spec.scaling_factor)
    if scale == 0.0:
        raise SdxlVaeError("scaling_factor 为 0 ✗ ⇒ 会除以 0 ✓（配置有问题 ✓）")
    with torch.no_grad():
        pixels = model.decode((latents - float(spec.shift_factor)) / scale)
    if not to_unit:
        return pixels
    return ((pixels + 1.0) / 2.0).clamp(0.0, 1.0)
