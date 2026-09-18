"""**视频 VAE**（本项目自己的实现 ✓ 真 `torch.nn.Module` ✓）—— 潜变量 ⇄ 像素。

## 为什么**编码**也要有（不只是解码 ✓）

* **出片**要解码（潜变量 → 帧 ✓）；
* **首帧条件（图生视频）**要编码 ✓ —— `torch_backend.condition_first_frame` 得把首帧图片
  变成潜变量 ✓ 才能按引擎算好的掩码混合 ✓（掩码在 :mod:`conditioning` ✓）。

## 明确的**简化**（都写在这里，免得后来人以为这就是 H3 的 VAE ✗）

1. **只做空间压缩** ✓（``spatial_scale = 2^层数`` ✓）；**时间不压缩** ✓
   —— 时间压缩（``(帧−1)//k+1`` ✓ 见 :func:`geometry.latent_frames` ✓）是**具体 VAE 的行为** ✓，
   本参考实现不做 ✗ ⇒ 接真 VAE 时这一层要换 ✓（`DiTConfig.vae_scale` 与它配套 ✓）。
2. **不做变分采样** ✓（只回归到 ``latent_channels`` ✓）—— 均值/对数方差、KL、per-channel 归一化
   都属于具体权重 ✓，本实现**不假装有** ✗。
3. 结构（3D 卷积 + GroupNorm + SiLU 的对称编解码 ✓）是公开的常规配方 ✓ —— 不是抄某仓库源码 ✗。

⇒ **能验证的是"机制"** ✓：形状自洽、可逆到同形、确定、越界报错 ✓；
**不是**"能还原 H3 的画面" ✗（那要真权重 ✓）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["VideoVAEConfig", "VAEError", "build_vae"]


class VAEError(ValueError):
    """配置/形状不合法 ✓（**明确报错**，不悄悄裁剪或补零 ✗）。"""


@dataclass(frozen=True)
class VideoVAEConfig:
    """结构参数 ✓（**全部显式** ✓ —— 不猜 H3 的配置 ✗）。"""

    in_channels: int = 3
    latent_channels: int = 4
    base_channels: int = 32
    #: 每层的通道倍数 ✓；**长度 = 空间下采样层数** ✓（3 ⇒ 8 倍 ✓）
    channel_multipliers: tuple[int, ...] = (1, 2, 2)

    def __post_init__(self) -> None:
        if not self.channel_multipliers:
            raise VAEError("channel_multipliers 不能为空 ✗")
        if any(multiplier <= 0 for multiplier in self.channel_multipliers):
            raise VAEError(f"channel_multipliers 必须为正：{self.channel_multipliers} ✗")
        if self.base_channels <= 0 or self.latent_channels <= 0:
            raise VAEError("base_channels / latent_channels 必须为正 ✗")

    @property
    def spatial_scale(self) -> int:
        """像素 ↔ 潜空间的**边长比** ✓。

        ⚠️ **是 ``2^(层数−1)``** 而不是 ``2^层数`` ✗ —— 第 0 层**保持原分辨率** ✓
        （标准配方 ✓：只在换层时下采样 ✓）。这条自检当场抓到过 ✗：属性写 8、实现实际 4 ✓
        ⇒ 这种"属性与实现不一致"最阴 ✗（调用方按属性准备尺寸 ✓ 却与模型对不上 ✓）。
        """
        return 2 ** (len(self.channel_multipliers) - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inChannels": self.in_channels, "latentChannels": self.latent_channels,
            "baseChannels": self.base_channels,
            "channelMultipliers": list(self.channel_multipliers),
            "spatialScale": self.spatial_scale,
            "temporalCompression": 1,
        }


def _build() -> type:
    """造出 :class:`VideoVAE` 类 ✓（模块级定义会强制 import torch ✗ ⇒ 放进工厂 ✓）。"""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class ResBlock3D(nn.Module):
        """``GroupNorm → SiLU → Conv3d`` ×2 ✓（VAE 的常规配方 ✓）。"""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.norm1 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
            self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
            self.norm2 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
            self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

        def forward(self, value: Any) -> Any:
            hidden = self.conv1(torch.nn.functional.silu(self.norm1(value)))
            return value + self.conv2(torch.nn.functional.silu(self.norm2(hidden)))  # 残差 ✓

    class VideoVAE(nn.Module):
        """对称的 3D 编解码器 ✓（层数由 ``channel_multipliers`` 决定 ✓）。"""

        def __init__(self, config: VideoVAEConfig) -> None:
            super().__init__()
            self.config = config
            multipliers = list(config.channel_multipliers)
            # ── 编码：逐层「ResBlock + 步长 2 的 3D 卷积」✓（时间维**不下采样** ✓ stride=(1,2,2) ✓）
            blocks: list[Any] = []
            channels = config.base_channels * multipliers[0]
            blocks.append(nn.Conv3d(config.in_channels, channels, 3, padding=1))
            for index, multiplier in enumerate(multipliers):
                out_channels = config.base_channels * multiplier
                if index:
                    blocks.append(ResBlock3D(channels))
                    blocks.append(nn.Conv3d(channels, out_channels, 3, stride=(1, 2, 2), padding=1))
                    channels = out_channels
                else:
                    blocks.append(ResBlock3D(channels))
            blocks.append(ResBlock3D(channels))
            self.encoder = nn.Sequential(*blocks)
            self.to_latent = nn.Conv3d(channels, config.latent_channels, 1)

            # ── 解码：镜像 ✓（``Upsample`` + 卷积 ✓；时间维**不放大** ✓）
            back: list[Any] = [nn.Conv3d(config.latent_channels, channels, 1),
                               ResBlock3D(channels)]
            for multiplier in reversed(multipliers[1:]):
                out_channels = config.base_channels * multiplier
                back.append(nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"))
                back.append(nn.Conv3d(channels, out_channels, 3, padding=1))
                back.append(ResBlock3D(out_channels))
                channels = out_channels
            back.append(ResBlock3D(channels))
            back.append(nn.Conv3d(channels, config.in_channels, 3, padding=1))
            self.decoder = nn.Sequential(*back)

        # ── 形状校验（**报错**而不是悄悄裁剪 ✗）────────────────────────────
        def _check_pixels(self, frames: Any) -> tuple[int, int]:
            if len(tuple(frames.shape)) != 5 or int(frames.shape[1]) != self.config.in_channels:
                raise VAEError(
                    f"帧张量形状应为 (B, {self.config.in_channels}, T, H, W) ✓，"
                    f"收到 {tuple(frames.shape)} ✗")
            height, width = int(frames.shape[3]), int(frames.shape[4])
            scale = self.config.spatial_scale
            if height % scale or width % scale:
                raise VAEError(
                    f"像素尺寸 {width}×{height} 不能被 {scale} 整除 ✗"
                    f"（spatial_scale = {scale} ✓ —— 尺寸要吸附到网格 ✓）")
            return height, width

        def _check_latent(self, latents: Any) -> tuple[int, int]:
            if len(tuple(latents.shape)) != 5 \
                    or int(latents.shape[1]) != self.config.latent_channels:
                raise VAEError(
                    f"潜变量形状应为 (B, {self.config.latent_channels}, T, h, w) ✓，"
                    f"收到 {tuple(latents.shape)} ✗")
            return int(latents.shape[3]), int(latents.shape[4])

        def encode(self, frames: Any) -> Any:
            """``(B,3,T,H,W)`` → ``(B,C_lat,T,H/S,W/S)`` ✓（时间维不变 ✓）。"""
            self._check_pixels(frames)
            return self.to_latent(self.encoder(frames))

        def decode(self, latents: Any) -> Any:
            """``(B,C_lat,T,h,w)`` → ``(B,3,T,h·S,w·S)`` ✓（与 :meth:`encode` 同规则 ✓）。"""
            self._check_latent(latents)
            return self.decoder(latents)

    return VideoVAE


_VAE_CLASS: Any = None


def build_vae(config: VideoVAEConfig) -> Any:
    """构造 VideoVAE ✓（**懒导入 torch** ✓ —— 没装 torch 也能 import 本模块 ✓）。"""
    global _VAE_CLASS
    if _VAE_CLASS is None:
        _VAE_CLASS = _build()
    return _VAE_CLASS(config)
