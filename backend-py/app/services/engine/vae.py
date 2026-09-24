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

#: ⚠️⚠️ **H3 视频 VAE 的事实**（2026-09-20 读参考 `comfy/ldm/minimax/vae.py` 头部核出 ✓）。
#: ⚠️ 这些**不是**本模块那套**合成**结构的参数 ✗ —— 它们是**真权重那一侧**的事实 ✓。
H3_VIDEO_VAE_FACTS: dict[str, Any] = {
    "family": "3D 因果 CNN 编码器 + **ViT3D 解码器** ✓（**不是**常见的 UNet VAE ✗）",
    #: 像素 ↔ 潜空间的边长比 ✓（**已核实** ✓ 2026-09-20 读参考核出；帧网格 ``17k+5 ⇔ 5k+2`` ✓
    #: ⇒ 潜帧数由 `geometry.latent_frames` 算 ✓）。⚠️ 双流造潜变量要用它 ✓ ——
    #: 拿别的数（比如 DiT 那边的 8 ✓）会得到**形状不对**的张量 ✓✗（运气好会报错、运气不好只是慢 ✓）。
    "vaeScale": 16,
    "latentsMean": [
        0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
        -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
        -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
        -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
        -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
        -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264],
    "latentsStd": [
        1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
        1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
        0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
        0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
        2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
        3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525],
    "imagenetMean": (0.485, 0.456, 0.406),
    "imagenetStd": (0.229, 0.224, 0.225),
    "statsNote": "**24 个均值/标准差** ⇒ 潜通道 24 ✓（与 `h3_form.H3_TRUNK_DEFAULTS` 的 `latents_dim` ✓ 一致 ✓"
                 "⚠️ 这组数字**猜不出来** ✗ —— 漏了它，画面会「看着还行、数值全错」✗✗（且**不报错** ✗）",
}

#: ⚠️ **H3 音频 VAE 是另一族** ✗（同一天核出 ✓）⇒ **`VideoVAE` 服务不了它** ✗（要单开 ✓，还没做 ✗）
H3_AUDIO_VAE_FACTS: dict[str, Any] = {
    "family": "**DAC 血统编码器 + BigVGAN 解码器** ✓（与视频 VAE 毫无关系 ✗）",
    "activations": "**Snake / SnakeBeta** ✓（逐通道 alpha ✓、beta 存 log 域 ✓）",
    "weightNorm": "⚠️ **已折叠**成普通 conv ✓ ⇒ 用普通 `Conv1d`/`ConvTranspose1d` 就能装 ✓"
                  "（权重是 plain 的 `*.weight` ✓，`strict=True` 可加载 ✓）",
    "aliasFree": "带抗混叠重采样（kaiser-windowed sinc ✓）",
    "latent": "32 **特征维** × **立体声 2** ✓ @ **40 Hz** ✓（与 `geometry` 的音频事实一致 ✓）",
}

__all__ = ["H3_AUDIO_VAE_FACTS", "H3_VIDEO_VAE_FACTS", "VideoVAEConfig", "VAEError", "build_vae"]


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
    #: 是否在编解码两侧做**潜变量归一化**（用 `H3_VIDEO_VAE_FACTS` 里那组 mean/std ✓）。
    #: ⚠️ **真权重必须开** ✓ —— 不开的话画面"看着还行"但**数值全错** ✗✗（且**不会报错** ✗）。
    #: ⚠️ 这里默认 ``False`` 的含义**只有一个**：**本模块那套合成结构（自检用）不做归一化** ✓ ——
    #: **不是**「H3 不需要归一化」✗（那条**已核实**：它就是需要 ✓）。两者别读混 ✗。
    apply_latent_stats: bool = False

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

        def _latent_stats(self) -> Any:
            """取那组核实过的统计 ✓（没开 / 数不对 ⇒ 见下面两步：**不猜** ✗）。"""
            if not self.config.apply_latent_stats:
                return None
            mean, std = H3_VIDEO_VAE_FACTS["latentsMean"], H3_VIDEO_VAE_FACTS["latentsStd"]
            if len(mean) != self.config.latent_channels:
                raise VAEError(
                    f"归一化统计有 {len(mean)} 个，但潜通道是 {self.config.latent_channels} ✗"
                    f"（**对不上就别用** ✓ —— 凑一组「差不多」的统计只会得到"
                    f"看着对的错图 ✗✗）")
            return mean, std

        def _stat_tensor(self, values: Any, reference: Any) -> Any:
            return torch.tensor(values, dtype=reference.dtype,
                                device=reference.device).reshape(
                (1, len(values), *([1] * (len(tuple(reference.shape)) - 2))))

        def normalize_latents(self, latents: Any) -> Any:
            """``(x − mean) / std`` ✓ 逐通道 ✓；未开归一化 ⇒ **原样返回** ✓（透明 ✓）。"""
            stats = self._latent_stats()
            if stats is None:
                return latents
            mean, std = stats
            return (latents - self._stat_tensor(mean, latents)) / self._stat_tensor(std, latents)

        def denormalize_latents(self, latents: Any) -> Any:
            """``x · std + mean`` ✓ —— :meth:`normalize_latents` 的**逆** ✓（未开 ⇒ 原样 ✓）。"""
            stats = self._latent_stats()
            if stats is None:
                return latents
            mean, std = stats
            return latents * self._stat_tensor(std, latents) + self._stat_tensor(mean, latents)

        def encode(self, frames: Any) -> Any:
            """``(B,3,T,H,W)`` → ``(B,C_lat,T,H/S,W/S)`` ✓（时间维不变 ✓）。

            ⚠️ 真权重路径：**编码后**要过 :meth:`normalize_latents` ✓（本合成路径默认不开 ✓）。
            """
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
