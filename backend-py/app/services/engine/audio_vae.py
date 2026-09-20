"""H3 **音频 VAE**（自己实现 ✓）—— 潜变量 ⇄ **32 kHz 立体声波形** ✓（**能真跑、能出真 wav** ✓）。

## 事实来源（2026-09-20 读参考 `comfy/ldm/minimax/audio_vae.py` 核出 ✓，**不猜** ✗）

* 采样率 **32000 Hz** ✓、编码器 strides ``(2,4,4,5,5)`` ✓ ⇒ ``hop = 800`` 样本/潜帧 ✓
  ⇒ ``32000 // 800 = **40**`` 潜帧/秒 ✓（与 `geometry.AUDIO_LATENT_HZ` ✓ 对得上 ✓✓）；
* 潜变量 ``[B, 32, 2, T]`` ✓（**32 特征维 × 立体声 2** ✓ —— 两声道**各自独立**过单声道编解码 ✓）；
* 解码器是 **BigVGAN 血统** ✓（``upsample_rates=(5,5,2,2,2,2,2)`` ✓ ⇒ 乘积同样 **800** ✓ 与 hop 自洽 ✓）；
* 激活是 **Snake / SnakeBeta** ✓（``x + (1/β)·sin²(αx)`` ✓；beta 存 log 域 ✓）；
* 潜变量**逐通道归一化** ✓（``z*std + mean`` ✓ 统计随检查点走 ✓）。

## ⚠️ 本实现与参考的差别（写清楚 ✗，免得读成"等价"✓）

* ``pre_block`` / ``AttnProjection``：参考那层我**没读内部** ✗ ⇒ 这里用**自己的**跨通道注意力近似 ✓
  （形状契约一致 ✓：``latent_dim → vae_latent_channels`` ✓；**数值不等价** ✗）；
* 编码器的 EncoderBlock 内部结构（抗混叠细节 ✓）也是**按事实重写**的近似 ✓；
* ⇒ **能装真权重的只有一部分键** ✗（要不要逐键对齐，等真权重到手再核 ✓）。
"""
from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["AudioVAEConfig", "AudioVAEError", "H3_AUDIO_VAE_DEFAULTS", "build_audio_vae",
           "write_wav"]


class AudioVAEError(ValueError):
    """配置/形状不合法 ✓（**明确报错**，不悄悄裁剪或补零 ✗）。"""


#: H3 音频 VAE 的出厂尺寸 ✓（**全是核出来的事实** ✓）
H3_AUDIO_VAE_DEFAULTS: dict[str, Any] = {
    "sample_rate": 32000,
    "encoder_rates": (2, 4, 4, 5, 5),
    "decoder_rates": (5, 5, 2, 2, 2, 2, 2),
    "latent_channels": 32,
    "stereo_channels": 2,
    "encoder_dim": 64,
    "latent_dim": 2048,
    "decoder_dim": 1024,
    "resblock_kernels": (3, 7, 11),
    "resblock_dilations": ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
}


@dataclass(frozen=True)
class AudioVAEConfig:
    """音频 VAE 结构 ✓（**全部显式** ✓ —— 不猜 ✗）。"""

    sample_rate: int = 32000
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    latent_channels: int = 32
    stereo_channels: int = 2
    encoder_dim: int = 64
    latent_dim: int = 2048
    decoder_dim: int = 1024
    resblock_kernels: tuple[int, ...] = (3, 7, 11)
    resblock_dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    #: 逐通道统计 ✓（``None`` ⇒ **恒等** ✓ = 合成/自检路径 ✓）。
    #: ⚠️ **真权重必须给** ✗ —— 不给的话波形"听着有声音"但**数值全错** ✗✗（且**不会报错** ✗）；
    #: 所以 `None` 的含义**只有一个**：**这条路径不做归一化** ✓，**不是**「H3 不需要」✗。
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None
    heads: int = 8
    field_extra: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.latent_channels <= 0 or self.stereo_channels <= 0:
            raise AudioVAEError("sample_rate / latent_channels / stereo_channels 必须为正 ✗")
        if not self.encoder_rates or not self.decoder_rates:
            raise AudioVAEError("encoder_rates / decoder_rates 不能为空 ✗")
        if any(rate <= 1 for rate in (*self.encoder_rates, *self.decoder_rates)):
            raise AudioVAEError(f"所有 strides 必须 > 1：{self.encoder_rates} / "
                                f"{self.decoder_rates} ✗")
        if self.hop_length != self.decoder_hop:
            raise AudioVAEError(
                f"编码 hop（{self.hop_length} ✓）与解码上采样乘积（{self.decoder_hop} ✓）"
                f"**必须相等** ✗ —— 不等的话「解出来的时长」与「编进去的时长」对不上 ✓✗"
                f"（而它**不会报错** ✗）")
        if self.sample_rate % self.hop_length:
            raise AudioVAEError(
                f"sample_rate {self.sample_rate} 不能被 hop {self.hop_length} 整除 ✗"
                f"⇒ 潜帧/秒不是整数 ✓（H3 是 32000/800 = 40 ✓）")
        for name, values in (("latents_mean", self.latents_mean),
                             ("latents_std", self.latents_std)):
            if values is None:
                continue
            if len(values) != self.latent_channels:
                raise AudioVAEError(
                    f"{name} 有 {len(values)} 个，但潜通道是 {self.latent_channels} ✗"
                    f"（**对不上就别用** ✓ —— 凑一组「差不多」的统计只会得到"
                    f"听着有声音的错音频 ✗✗）")
        if (self.latents_mean is None) != (self.latents_std is None):
            raise AudioVAEError("latents_mean 与 latents_std 要么都给 ✓、要么都不给 ✗（不给一半 ✗）")
        # ⚠️⚠️ 2026-09-20 由双流自检**当场抓到**的真坑 ✗✓：解码器每级**通道减半** ✓
        #    （`decoder_dim // 2**(index+1)` ✓）⇒ `decoder_dim < 2^级数` 时末级宽度算成 **0** ✗
        #    ⇒ 造出一个 in_channels=0 的卷积 ✓，**构造期不报错** ✗，等到 forward 才炸 ✓
        #    而且报的是**一句 oneDNN 的「反卷积建不出原语」**✓✗ —— 完全指不到真因 ✓。
        #    ⇒ 在本类里就把这条不变量钉住 ✓（真 H3 是 1024 ≥ 2^7 ✓ 本来就不会触发 ✓）。
        needed = 2 ** len(self.decoder_rates)
        if self.decoder_dim < needed:
            raise AudioVAEError(
                f"decoder_dim={self.decoder_dim} 太小 ✗：{len(self.decoder_rates)} 级上采样逐级减半 ✓ "
                f"⇒ 至少要 {needed} ✓（否则末级通道数是 0，造出的卷积**非法** ✗ —— "
                f"而报错要到 forward 里才出现，且是一句看不懂的卷积后端消息 ✗✗）")

    @property
    def hop_length(self) -> int:
        """**样本/潜帧** ✓（= 编码器 strides 的乘积 ✓；H3 = **800** ✓）。"""
        return math.prod(self.encoder_rates)

    @property
    def decoder_hop(self) -> int:
        """解码器上采样乘积 ✓（**必须等于** :attr:`hop_length` ✓ —— 自检钉住 ✓）。"""
        return math.prod(self.decoder_rates)

    @property
    def latents_per_second(self) -> int:
        """**潜帧/秒** ✓（H3 = **40** ✓，与 `geometry.AUDIO_LATENT_HZ` 一致 ✓）。"""
        return self.sample_rate // self.hop_length

    @property
    def uses_stats(self) -> bool:
        return self.latents_mean is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampleRate": self.sample_rate, "hopLength": self.hop_length,
            "latentsPerSecond": self.latents_per_second,
            "latentChannels": self.latent_channels, "stereoChannels": self.stereo_channels,
            "latentDim": self.latent_dim, "decoderDim": self.decoder_dim,
            "usesStats": self.uses_stats, **self.field_extra,
        }


def write_wav(path: str | Path, waveform: Any, sample_rate: int) -> dict[str, Any]:
    """把 ``(channels, samples)`` 的浮点波形写成**真 wav** ✓（16-bit PCM ✓ 标准库 ✓ 无依赖 ✓）。

    ⚠️ 输入按约定在 ``[-1, 1]`` ✓（超范围**夹紧** ✓ 而不是循环回绕 ✗ —— 回绕会把爆音变成噪声 ✓✗）。
    返回写盘回执 ✓（帧数/时长/比特 ✓）—— 方便自检与调用方核对 ✓。
    """
    import numpy as np  # noqa: PLC0415 —— 只有真写盘才需要 ✓

    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 2:
        raise AudioVAEError(f"波形应为 (channels, samples) ✓，收到 {values.shape} ✗")
    channels, frames = int(values.shape[0]), int(values.shape[1])
    clipped = np.clip(values, -1.0, 1.0)
    pcm = (clipped * 32767.0).round().astype("<i2").T.reshape(-1)     # 交错 ✓
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return {"path": str(target), "channels": channels, "frames": frames,
            "sampleRate": int(sample_rate), "bits": 16,
            "durationSeconds": round(frames / float(sample_rate), 6),
            "bytes": target.stat().st_size}


def _build() -> Any:
    """建出模型类 ✓（**懒导入 torch** ✓ —— 没装 torch 也能 import 本模块 ✓）。"""
    import torch  # noqa: PLC0415
    from torch import nn

    def snake(x: Any, alpha: Any, beta: Any) -> Any:
        """``x + (1/β)·sin²(α·x)`` ✓（Snake 激活 ✓；**公式是事实** ✓）。"""
        value = torch.sin(alpha * x)
        return x + value * value / (beta + 1e-9)

    class Snake1d(nn.Module):  # noqa: D101
        """逐通道 ``alpha``（= ``beta`` ✓，编码器侧同值 ✓，事实 ✓）。"""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.alpha = nn.Parameter(torch.ones(1, channels, 1))

        def forward(self, x: Any) -> Any:
            return snake(x, self.alpha, self.alpha)

    class SnakeBeta(nn.Module):  # noqa: D101
        """``alpha`` / ``beta`` **分开**且存 **log 域** ✓（解码器侧 ✓，事实 ✓）。"""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.alpha = nn.Parameter(torch.zeros(channels))
            self.beta = nn.Parameter(torch.zeros(channels))

        def forward(self, x: Any) -> Any:
            return snake(x, torch.exp(self.alpha).view(1, -1, 1), torch.exp(self.beta).view(1, -1, 1))

    class ResBlock1(nn.Module):  # noqa: D101
        """BigVGAN 的残差块 ✓：**每个 kernel 一组** ✓，组内按各自的 dilations 走并相加 ✓，再按组平均 ✓。

        ⚠️ 这里我**理解错过一次** ✗✓：把 ``kernels`` 与 ``dilations`` 当成了**叉乘** ✗，
        实际是「``resblock_dilation_sizes`` 的第 i 项配 ``resblock_kernel_sizes`` 的第 i 个 kernel」✓
        ⇒ 叉乘写法会让 ``dilation`` 变成**元组** ✓ ⇒ ``tuple // 2`` 直接 `TypeError` ✓（**响亮** ✓）。
        """

        def __init__(self, channels: int, kernels: Any, dilations: Any) -> None:
            super().__init__()
            kernels, dilations = tuple(kernels), tuple(dilations)
            if len(kernels) != len(dilations):
                raise AudioVAEError(
                    f"每个 kernel 要配**一组** dilations ✗（收到 {len(kernels)} 个 kernel、"
                    f"{len(dilations)} 组 dilation ✓）")
            self.blocks = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        SnakeBeta(channels),
                        nn.Conv1d(channels, channels, kernel, dilation=dilation,
                                  padding=dilation * (kernel - 1) // 2))
                    for dilation in dilations[index]])
                for index, kernel in enumerate(kernels)])

        def forward(self, x: Any) -> Any:
            groups = []
            for group in self.blocks:
                summed = x
                for path in group:
                    summed = summed + path(x)
                groups.append(summed)
            stacked = torch.stack(groups, dim=0)
            return stacked.mean(dim=0)

    class AttnProjection(nn.Module):  # noqa: D101
        """``latent_dim → latent_channels`` ✓ 用**自己的**注意力近似 ✓（⚠️ 参考那层内部**没读** ✗）。"""

        def __init__(self, latent_dim: int, out_channels: int, heads: int) -> None:
            super().__init__()
            self.heads = max(1, int(heads))
            self.norm = nn.LayerNorm(latent_dim)
            self.attn = nn.MultiheadAttention(latent_dim, self.heads, batch_first=True)
            self.proj = nn.Linear(latent_dim, out_channels)

        def forward(self, x: Any) -> Any:
            pooled = self.norm(x)
            attended, _ = self.attn(pooled, pooled, pooled, need_weights=False)
            return self.proj(attended + pooled)

    class EncoderBlock(nn.Module):  # noqa: D101
        """``Snake → 带 strides 的卷积`` ✓（**下采样 + 通道翻倍** ✓；抗混叠细节是近似 ✓）。

        ⚠️ 这里我**也写错过一次** ✗✓：照抄参考那种「先 ``d_model *= 2`` 再建块」的写法 ✗，
        导致块的**输入宽度**比实际张量多一倍 ✓ ⇒ ``4 vs 8`` 的 size mismatch ✓（**响亮** ✓）。
        ⇒ 本实现自己保证**自洽** ✓：输入 = 当前宽度 ✓、输出 = 翻倍 ✓（与参考的**外部行为**一致 ✓）。
        """

        def __init__(self, in_channels: int, stride: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                Snake1d(in_channels),
                nn.Conv1d(in_channels, in_channels * 2, 2 * stride, stride=stride,
                          padding=stride // 2))

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class BigVGANDecoder(nn.Module):  # noqa: D101
        """**BigVGAN 血统解码器** ✓：``conv_pre`` → 逐级 ``ConvTranspose1d`` + 残差块 → ``tanh`` ✓。

        ⚠️ 输出**夹到 [-1, 1]** ✓（参考如此 ✓ —— 这是"能直接写 wav"的前提 ✓）。
        """

        def __init__(self, config: AudioVAEConfig) -> None:
            super().__init__()
            self.num_kernels = len(config.resblock_kernels)
            self.conv_pre = nn.Conv1d(config.latent_dim, config.decoder_dim, 7, padding=3)
            ups, resblocks = nn.ModuleList(), nn.ModuleList()
            for index, rate in enumerate(config.decoder_rates):
                ups.append(nn.ConvTranspose1d(
                    config.decoder_dim // (2 ** index),
                    config.decoder_dim // (2 ** (index + 1)), 2 * rate, stride=rate,
                    padding=rate // 2 + rate % 2))
                resblocks.append(nn.ModuleList([
                    ResBlock1(config.decoder_dim // (2 ** (index + 1)), config.resblock_kernels,
                              config.resblock_dilations)
                    for _ in range(self.num_kernels)]))
            self.ups, self.resblocks = ups, resblocks
            self.activation_post = SnakeBeta(config.decoder_dim // (2 ** len(config.decoder_rates)))
            self.conv_post = nn.Conv1d(config.decoder_dim // (2 ** len(config.decoder_rates)),
                                       1, 7, padding=3)

        def forward(self, x: Any) -> Any:
            x = self.conv_pre(x)
            for index, (up, resblock_set) in enumerate(zip(self.ups, self.resblocks)):
                x = up(x)
                if index == len(self.ups) - 1:
                    x = x[..., :x.shape[-1]]
                summed = resblock_set[0](x)
                for resblock in resblock_set[1:]:
                    summed = summed + resblock(x)
                x = summed / len(resblock_set)
            return torch.tanh(self.conv_post(self.activation_post(x)))

    class Encoder(nn.Module):  # noqa: D101
        def __init__(self, config: AudioVAEConfig) -> None:
            super().__init__()
            blocks = [nn.Conv1d(1, config.encoder_dim, 7, padding=3)]
            width = config.encoder_dim
            for stride in config.encoder_rates:
                blocks.append(EncoderBlock(width, stride))   # 输入 = 当前宽度 ✓
                width *= 2                                   # 输出翻倍 ⇒ 下一块的输入 ✓
            blocks += [Snake1d(width), nn.Conv1d(width, config.latent_dim, 3, padding=1)]
            self.blocks = nn.Sequential(*blocks)

        def forward(self, x: Any) -> Any:
            return self.blocks(x)

    class AudioVAE(nn.Module):  # noqa: D101
        """**潜变量 ⇄ 波形** ✓（``[B,32,2,T] ⇄ [B,2,L]`` ✓，``L = T × hop`` ✓）。"""

        def __init__(self, config: AudioVAEConfig) -> None:
            super().__init__()
            self.config = config
            self.encoder = Encoder(config)
            self.pre_block = AttnProjection(config.latent_dim, config.latent_channels,
                                            config.heads)
            self.mean_proj = nn.Conv1d(config.latent_channels, config.latent_channels, 1)
            self.dec_in_proj = nn.Conv1d(config.latent_channels, config.latent_dim, 1)
            self.decoder = BigVGANDecoder(config)

        # ── 归一化（统计随检查点走 ✓；没给 ⇒ 恒等 ✓）──
        def _stats(self, reference: Any, name: str) -> Any:
            values = getattr(self.config, name)
            if values is None:
                return None
            return torch.tensor(values, dtype=reference.dtype,
                                device=reference.device).view(1, -1, 1)

        def normalize_latents(self, latents: Any) -> Any:
            """``(z − mean) / std`` ✓（**逐通道** ✓）；没给统计 ⇒ **原样** ✓。"""
            mean, std = self._stats(latents, "latents_mean"), self._stats(latents, "latents_std")
            if mean is None:
                return latents
            return (latents - mean) / std

        def denormalize_latents(self, latents: Any) -> Any:
            """``z · std + mean`` ✓（:meth:`normalize_latents` 的逆 ✓）。"""
            mean, std = self._stats(latents, "latents_mean"), self._stats(latents, "latents_std")
            if mean is None:
                return latents
            return latents * std + mean

        def decode(self, latents: Any) -> Any:
            """``[B, 32, 2, T]``（**已归一化** ✓）⇒ ``[B, 2, L]`` ✓（``L = T × hop`` ✓）。

            ⚠️ 两声道的处理照事实来 ✓：先 ``permute`` 成 ``[B·2, 32, T]`` ✓（**各走各的** ✓），
            解完再 ``reshape`` 回立体声 ✓。
            """
            shape = tuple(latents.shape)
            if len(shape) != 4 or int(shape[1]) != self.config.latent_channels \
                    or int(shape[2]) != self.config.stereo_channels:
                raise AudioVAEError(
                    f"潜变量应为 (B, {self.config.latent_channels}, "
                    f"{self.config.stereo_channels}, T) ✓，收到 {shape} ✗")
            batch, channels, stereo, frames = (int(value) for value in shape)
            flat = latents.permute(0, 2, 1, 3).reshape(batch * stereo, channels, frames)
            flat = self.denormalize_latents(flat)
            wave_out = self.decoder(self.dec_in_proj(flat))
            return wave_out.reshape(batch, stereo, -1)

        def encode(self, waveform: Any) -> Any:
            """``[B, 2, L]``（``[-1,1]`` ✓）⇒ ``[B, 32, 2, T]`` ✓（**已归一化** ✓）。

            ⚠️ ``L`` **不是** ``hop`` 的整数倍时**右侧补零** ✓（事实如此 ✓ —— 补零而不是报错 ✓，
            因为这是"把任意长度音频编进去"的正常做法 ✓）；``T = ceil(L/hop)`` ✓。
            """
            shape = tuple(waveform.shape)
            if len(shape) != 3 or int(shape[1]) != self.config.stereo_channels:
                raise AudioVAEError(
                    f"波形应为 (B, {self.config.stereo_channels}, L) ✓，收到 {shape} ✗")
            batch, stereo, length = (int(value) for value in shape)
            pad = -length % self.config.hop_length
            padded = (nn.functional.pad(waveform, (0, pad)) if pad else waveform)
            flat = padded.reshape(batch * stereo, 1, -1)
            hidden = self.encoder(flat)                              # [B·2, latent_dim, T] ✓
            projected = self.pre_block(hidden.transpose(1, 2)).transpose(1, 2)
            latents = self.mean_proj(projected).reshape(batch, stereo, -1, projected.shape[-1])
            return self.normalize_latents(latents.permute(0, 2, 1, 3))

    return AudioVAE


_AUDIO_VAE_CLASS: Any = None


def build_audio_vae(config: AudioVAEConfig | None = None) -> Any:
    """构造音频 VAE ✓（**懒导入 torch** ✓）；``config`` 省略 ⇒ 用 H3 出厂事实 ✓。"""
    global _AUDIO_VAE_CLASS
    if _AUDIO_VAE_CLASS is None:
        _AUDIO_VAE_CLASS = _build()
    if config is None:
        config = AudioVAEConfig(**{key: value for key, value in H3_AUDIO_VAE_DEFAULTS.items()
                                   if key != "sample_rate"} | {"sample_rate": 32000})
    return _AUDIO_VAE_CLASS(config)


def _unused_struct_guard() -> None:  # pragma: no cover - 仅为保留 stdlib 依赖的显式痕迹 ✓
    """（``struct`` 留着以便将来写 24-bit 时用 ✓ —— 现在没有调用点 ✓ 故不触发 ✓。）"""
    _ = struct
