"""H3 **音频 VAE**（自己实现 ✓）—— 潜变量 ⇄ **32 kHz 立体声波形** ✓（**能真跑、能出真 wav** ✓）。

## 事实来源（2026-09-20 读参考 `comfy/ldm/minimax/audio_vae.py` 核出 ✓，**不猜** ✗）

* 采样率 **32000 Hz** ✓、编码器 strides ``(2,4,4,5,5)`` ✓ ⇒ ``hop = 800`` 样本/潜帧 ✓
  ⇒ ``32000 // 800 = **40**`` 潜帧/秒 ✓（与 `geometry.AUDIO_LATENT_HZ` ✓ 对得上 ✓✓）；
* 潜变量 ``[B, 32, 2, T]`` ✓（**32 特征维 × 立体声 2** ✓ —— 两声道**各自独立**过单声道编解码 ✓）；
* 解码器是 **BigVGAN 血统** ✓（``upsample_rates=(5,5,2,2,2,2,2)`` ✓ ⇒ 乘积同样 **800** ✓ 与 hop 自洽 ✓）；
* 激活是 **Snake / SnakeBeta** ✓（``x + (1/β)·sin²(αx)`` ✓；beta 存 log 域 ✓）；
* 潜变量**逐通道归一化** ✓（``z*std + mean`` ✓ 统计随检查点走 ✓）。

## 结构（2026-11-01 **按参考逐行对齐** ✓ ⇒ 键名与数值都对齐 ✓，不再有"近似层"✗）

* 编码器 = DAC 血统 ✓：``Conv(1→d,7)`` → 逐级 ``EncoderBlock``（**块内宽度 = dim//2** ✓，含三个
  ``ResidualUnit``）+ ``Snake1d`` + ``Conv(→latent_dim,3)`` ✓，名字挂在 ``encoder.block`` ✓；
* 后验头 = ``AttnProjection`` ✓（``LayerNorm`` + **因果** ``CausalAttention`` + ``GeGluMlp`` ✓，
  ``qkv`` 无 bias 而 ``q_bias``/``v_bias`` 是参数 ✓、``zero_k_bias`` 是 buffer ✓）⇒ 只在编码侧 ✓；
  ``logs_proj`` 参考里**推理不用** ✓ 但**检查点里有** ⇒ 必须建出来 ✓（否则白多一组键 ✗）；
* 解码器 = **BigVGAN** ✓：``ups`` 是**嵌套** ``ModuleList`` ✓、``resblocks`` 是**扁平**
  ``级数×kernel`` 个 ``AMPBlock1`` ✓、激活一律 ``Activation1d``（**升采样→Snake→降采样** ✓，
  kaiser-sinc ✓）、末尾 ``activation_post → conv_post(无 bias) → clamp(-1,1)`` ✓；
* 上采样核是 ``(9,9,4,4,4,4,4)`` ✓（**不是** ``2·rate`` ✗✓）⇒ ``L = T × hop`` **精确成立** ✓
  （旧写法每级少 1 个样本 ⇒ 8 帧解出 6208 而不是 6400 ✗）；
* 激活是 **Snake / SnakeBeta** ✓（``x + (1/β)·sin²(αx)`` ✓ 含 ``+1e-9`` ✓；beta 存 log 域 ✓）；
* 潜变量**逐通道归一化** ✓（``z*std + mean`` ✓ 统计是检查点里的 ``latents_mean``/``latents_std``
  buffer ✓ —— 真权重直接盖过来 ✓，``config`` 给了就先写进去 ✓）。

⇒ 结论：**真权重可以逐键装进来** ✓（`tests/engine_audio_vae_test.py` 的 ⑧ 把每处结构判断都钉成键名 ✓）。
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
    #: 上采样**核** ✓（参考 H3 档是 ``(9,9,4,4,4,4,4)`` ✓）；``None`` ⇒ 按 stride 推 ✓（见
    #: :attr:`upsample_kernels` ✓，只有自检的小配置会走到 ✓）。**不能**用 ``2·rate`` 顶替 ✗✗。
    decoder_kernels: tuple[int, ...] | None = None
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
        # ⚠️ 上采样**核**是参考的硬事实 ✓（H3 档 ``(9,9,4,4,4,4,4)`` ✓，不是 ``2*rate`` ✗✓）：
        #    核与 stride 的**奇偶必须同** ✓，否则 ``(k−rate)//2`` 的补边让长度差 1 ✓
        #    ⇒ 解出来的样本数比 ``T × hop`` 少/多几个 ✓（**不报错** ✓，只是时长悄悄不对 ✓✗）。
        for rate, kernel in zip(self.decoder_rates, self.upsample_kernels):
            if kernel <= rate or (kernel - rate) % 2:
                raise AudioVAEError(
                    f"上采样核 {kernel} 与 stride {rate} 不匹配 ✗：要 kernel > stride ✓ 且"
                    f"「kernel − stride」为**偶数** ✓（参考的配方是 4/2 ✓、9/5 ✓）")

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
    def upsample_kernels(self) -> tuple[int, ...]:
        """上采样核 ✓：显式给了就用 ✓，否则按参考配方推 ✓（``5 → 9`` ✓、``2 → 4`` ✓、其它 ``2·rate`` ✓）。

        ⚠️ 参考 H3 是 ``(9,9,4,4,4,4,4)`` ✓ —— **不是** ``2·rate`` ✗✓（``2·5=10`` 长度会差 ✓）。
        只有自检用的小配置（``decoder_rates=(2,2)``）走推导分支 ✓。
        """
        if self.decoder_kernels is not None:
            return tuple(self.decoder_kernels)
        return tuple(9 if rate == 5 else 4 if rate == 2 else 2 * rate
                     for rate in self.decoder_rates)

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

    # ── 抗混叠重采样 ✓（参考 `kaiser_sinc_filter1d` / `UpSample1d` / `LowPassFilter1d` /
    #    `DownSample1d` / `Activation1d` ✓ —— 2026-11-01 **逐行照参考** ✓，此前是「无重采样的
    #    Snake」✗ ⇒ 键名（`…act.alpha` / `…upsample.filter`）与数值**双双**对不上真权重 ✓✗）──
    def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> Any:
        """Kaiser 窗 sinc 低通 ✓（``[1, 1, kernel_size]`` ✓，和为 1 ✓）。"""
        even = kernel_size % 2 == 0
        half_size = kernel_size // 2
        attenuation = 2.285 * (half_size - 1) * math.pi * 4 * half_width + 7.95
        if attenuation > 50.0:
            beta = 0.1102 * (attenuation - 8.7)
        elif attenuation >= 21.0:
            beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
        else:
            beta = 0.0
        window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
        if even:
            time = torch.arange(-half_size, half_size) + 0.5
        else:
            time = torch.arange(kernel_size) - half_size
        response = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        return (response / response.sum()).view(1, 1, kernel_size)

    class UpSample1d(nn.Module):  # noqa: D101
        """转置卷积升采样 ✓（``× ratio`` 缩放 ✓ + 两侧裁掉边缘 ✓，与参考逐项相同 ✓）。"""

        def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
            super().__init__()
            self.ratio = ratio
            self.stride = ratio
            self.pad = kernel_size // ratio - 1
            self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
            self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
            self.register_buffer("filter", kaiser_sinc_filter1d(
                cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size))

        def forward(self, x: Any) -> Any:
            channels = x.shape[1]
            value = torch.nn.functional.pad(x, (self.pad, self.pad), mode="replicate")
            value = torch.nn.functional.conv_transpose1d(
                value, self.filter.expand(channels, -1, -1).to(x.dtype),
                stride=self.stride, groups=channels) * self.ratio
            return value[..., self.pad_left:-self.pad_right]

    class LowPassFilter1d(nn.Module):  # noqa: D101
        """sinc 低通 + 跨步 ✓（降采样先滤波 ✓，否则会混叠 ✓）。"""

        def __init__(self, cutoff: float = 0.5, half_width: float = 0.6, stride: int = 1,
                     kernel_size: int = 12) -> None:
            super().__init__()
            self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
            self.pad_right = kernel_size // 2
            self.stride = stride
            self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

        def forward(self, x: Any) -> Any:
            channels = x.shape[1]
            value = torch.nn.functional.pad(x, (self.pad_left, self.pad_right), mode="replicate")
            return torch.nn.functional.conv1d(
                value, self.filter.expand(channels, -1, -1).to(x.dtype),
                stride=self.stride, groups=channels)

    class DownSample1d(nn.Module):  # noqa: D101
        """``LowPassFilter1d`` 的薄封装 ✓（cutoff 随 ratio 收窄 ✓）。"""

        def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
            super().__init__()
            self.ratio = ratio
            self.kernel_size = kernel_size
            self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio,
                                           stride=ratio, kernel_size=kernel_size)

        def forward(self, x: Any) -> Any:
            return self.lowpass(x)

    class Activation1d(nn.Module):  # noqa: D101
        """**升采样 → 激活 → 降采样** ✓（alias-free ✓；参考的 ``Activation1d`` 逐项相同 ✓）。"""

        def __init__(self, activation: Any, up_ratio: int = 2, down_ratio: int = 2,
                     up_kernel_size: int = 12, down_kernel_size: int = 12) -> None:
            super().__init__()
            self.act = activation
            self.upsample = UpSample1d(up_ratio, up_kernel_size)
            self.downsample = DownSample1d(down_ratio, down_kernel_size)

        def forward(self, x: Any) -> Any:
            return self.downsample(self.act(self.upsample(x)))

    # ── DAC 编码器 ✓（参考 `ResidualUnit` / `EncoderBlock` / `Encoder` 逐行 ✓）──
    class ResidualUnit(nn.Module):  # noqa: D101
        """``Snake→Conv(dil)→Snake→Conv1`` ✓ + 残差 ✓（长度差**两侧对称**裁掉 ✓，参考如此 ✓）。"""

        def __init__(self, dim: int, dilation: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                Snake1d(dim),
                nn.Conv1d(dim, dim, 7, dilation=dilation, padding=((7 - 1) * dilation) // 2),
                Snake1d(dim),
                nn.Conv1d(dim, dim, 1))

        def forward(self, x: Any) -> Any:
            value = self.block(x)
            pad = (x.shape[-1] - value.shape[-1]) // 2
            if pad > 0:
                x = x[..., pad:-pad]
            return value + x

    class EncoderBlock(nn.Module):  # noqa: D101
        """三个 ``ResidualUnit`` ＋ ``Snake`` ＋ **跨步卷积** ✓（宽度是块宽的一半 ✓ —— 参考如此 ✓）。"""

        def __init__(self, dim: int, stride: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                ResidualUnit(dim // 2, 1),
                ResidualUnit(dim // 2, 3),
                ResidualUnit(dim // 2, 9),
                Snake1d(dim // 2),
                nn.Conv1d(dim // 2, dim, 2 * stride, stride=stride,
                          padding=math.ceil(stride / 2)))

        def forward(self, x: Any) -> Any:
            return self.block(x)

    # ── 后验头 ✓（参考 `GeGluMlp` / `CausalAttention` / `AttnProjection` 逐行 ✓ ——
    #    **只走编码侧** ✓：解码不经过它 ✓，但它挂在检查点里 ✓，缺了就是一组键装不上 ✓）──
    class GeGluMlp(nn.Module):  # noqa: D101
        """``LayerNorm → (gelu(w0·x) · w1·x) → w2`` ✓（GELU 用 **tanh 近似** ✓，参考如此 ✓）。"""

        def __init__(self, in_features: int, hidden_features: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(in_features)
            self.act = nn.GELU(approximate="tanh")
            self.w0 = nn.Linear(in_features, hidden_features)
            self.w1 = nn.Linear(in_features, hidden_features)
            self.w2 = nn.Linear(hidden_features, in_features)

        def forward(self, x: Any) -> Any:
            value = self.norm(x)
            return self.w2(self.act(self.w0(value)) * self.w1(value))

    class CausalAttention(nn.Module):  # noqa: D101
        """**因果**自注意力 ✓ ⇒ 各头取均值 ✓ ⇒ 沿特征维池化到 ``out_dim`` ✓ ⇒ ``proj`` ✓。

        ⚠️ 三个偏置的形状是参考的**硬事实** ✓：``qkv`` 本身 ``bias=False`` ✓，而
        ``q_bias`` / ``v_bias`` 是参数 ✓、``zero_k_bias`` 是 **buffer** ✓（k 侧恒 0 ✓）——
        写成「一个普通 bias」会让键名与形状都对不上 ✓✗。
        """

        def __init__(self, in_dim: int, out_dim: int, num_heads: int) -> None:
            super().__init__()
            self.head_dim = in_dim // num_heads
            self.num_heads = num_heads
            self.out_dim = out_dim
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer("zero_k_bias", torch.zeros(in_dim))
            self.proj = nn.Linear(out_dim, out_dim)

        def forward(self, x: Any) -> Any:
            batch, count, _ = x.shape
            bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)).to(x.dtype)
            qkv = torch.nn.functional.linear(x, self.qkv.weight.to(x.dtype), bias)
            q, k, v = qkv.reshape(batch, count, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
            attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            pooled = torch.nn.functional.adaptive_avg_pool1d(
                torch.mean(attended, dim=1), self.out_dim)
            return self.proj(pooled)

    class AttnProjection(nn.Module):  # noqa: D101
        """``[B,T,latent_dim] → [B,T,latent_channels]`` ✓（**残差**两跳 ✓，参考 ``AttnProjection`` ✓）。"""

        def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(in_dim)
            self.attn = CausalAttention(in_dim, out_dim, num_heads)
            self.proj = nn.Linear(in_dim, out_dim)
            self.norm3 = nn.LayerNorm(in_dim)
            self.norm2 = nn.LayerNorm(out_dim)
            self.mlp = GeGluMlp(in_features=out_dim,
                                hidden_features=int(out_dim * mlp_ratio))

        def forward(self, x: Any) -> Any:
            value = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
            return value + self.mlp(self.norm2(value))

    # ── BigVGAN 解码器 ✓（参考 `AMPBlock1` / `BigVGAN` 逐行 ✓）──
    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)

    class AMPBlock1(nn.Module):  # noqa: D101
        """**抗混叠**残差块 ✓：``convs1[i]`` 带膨胀 ✓、``convs2[i]`` 不带 ✓，两者各配一个激活 ✓。"""

        def __init__(self, channels: int, kernel_size: int = 3,
                     dilation: Any = (1, 3, 5)) -> None:
            super().__init__()
            self.convs1 = nn.ModuleList([
                nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=dilation,
                          padding=get_padding(kernel_size, dilation))
                for dilation in dilation])
            self.convs2 = nn.ModuleList([
                nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=1,
                          padding=get_padding(kernel_size, 1))
                for _ in range(len(tuple(dilation)))])
            self.num_layers = len(self.convs1) + len(self.convs2)
            self.activations = nn.ModuleList([
                Activation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)])

        def forward(self, x: Any) -> Any:
            acts1, acts2 = self.activations[::2], self.activations[1::2]
            for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
                x = x + conv2(act2(conv1(act1(x))))
            return x

    class BigVGAN(nn.Module):  # noqa: D101
        """**BigVGAN 解码器** ✓：``conv_pre`` → 逐级（``ConvTranspose1d`` ＋ 该级全部残差块取均值）
        → ``activation_post`` → ``conv_post`` → **``clamp(-1, 1)``** ✓。

        ⚠️ 三处**键名/数值的硬事实** ✓（2026-11-01 照参考改 ✓，此前全都不对 ✗）：
        ① ``ups`` 是**嵌套** ``ModuleList([ConvTranspose1d])`` ✓ ⇒ 键是 ``ups.0.0.weight`` ✓；
        ② ``resblocks`` 是**扁平**的 ✓（``num_ups × num_kernels`` 个 ✓）⇒ 键是 ``resblocks.7.convs1.0.weight`` ✓；
        ③ 末尾是 ``clamp`` **不是** ``tanh`` ✗✓，且 ``conv_post`` **无 bias** ✓ ⇒ 多一个
        ``conv_post.bias`` 键 ✓✗。①②③ 任一写错 ⇒ 真权重装进去就是**形状缺失/静默走错层** ✓。
        """

        def __init__(self, config: AudioVAEConfig) -> None:
            super().__init__()
            self.num_kernels = len(config.resblock_kernels)
            self.num_upsamples = len(config.decoder_rates)
            self.conv_pre = nn.Conv1d(config.latent_dim, config.decoder_dim, 7, padding=3)
            self.ups = nn.ModuleList()
            for index, (rate, kernel) in enumerate(zip(config.decoder_rates,
                                                       config.upsample_kernels)):
                self.ups.append(nn.ModuleList([nn.ConvTranspose1d(
                    config.decoder_dim // (2 ** index),
                    config.decoder_dim // (2 ** (index + 1)),
                    kernel, rate, padding=(kernel - rate) // 2)]))
            self.resblocks = nn.ModuleList()
            for index in range(self.num_upsamples):
                width = config.decoder_dim // (2 ** (index + 1))
                for kernel, dilations in zip(config.resblock_kernels, config.resblock_dilations):
                    self.resblocks.append(AMPBlock1(width, kernel, dilations))
            width = config.decoder_dim // (2 ** self.num_upsamples)
            self.activation_post = Activation1d(activation=SnakeBeta(width))
            self.conv_post = nn.Conv1d(width, 1, 7, padding=3, bias=False)

        def forward(self, x: Any) -> Any:
            x = self.conv_pre(x)
            for index in range(self.num_upsamples):
                for up in self.ups[index]:
                    x = up(x)
                summed = None
                for offset in range(self.num_kernels):
                    block = self.resblocks[index * self.num_kernels + offset]
                    summed = block(x) if summed is None else summed + block(x)
                x = summed / self.num_kernels
            x = self.activation_post(x)
            return self.conv_post(x).clamp(-1.0, 1.0)

    class Encoder(nn.Module):  # noqa: D101
        """DAC 编码器 ✓：``Conv(1→d,7)`` → 逐级 ``EncoderBlock``（**先翻倍再建块** ✓）→ ``Snake`` ＋ ``Conv(→latent)`` ✓。

        ⚠️ 参考是「``d_model *= 2`` 之后建块」✓，而块的**内部宽度是 ``dim // 2``** ✓ ⇒
        「翻倍」与「半宽」正好抵消 ✓，这才是它能接上前一级的原因 ✓（我此前按「输入宽度翻倍」
        写过一版 ✗ ⇒ 8 vs 4 尺寸不匹配 ✓；按参考的**块内半宽**写就自洽 ✓）。
        """

        def __init__(self, config: AudioVAEConfig) -> None:
            super().__init__()
            width = config.encoder_dim
            blocks: list[Any] = [nn.Conv1d(1, width, 7, padding=3)]
            for stride in config.encoder_rates:
                width *= 2
                blocks.append(EncoderBlock(width, stride))
            blocks += [Snake1d(width), nn.Conv1d(width, config.latent_dim, 3, padding=1)]
            self.block = nn.Sequential(*blocks)

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class AudioVAE(nn.Module):  # noqa: D101
        """**潜变量 ⇄ 波形** ✓（``[B,32,2,T] ⇄ [B,2,L]`` ✓，``L = T × hop`` ✓）。"""

        def __init__(self, config: AudioVAEConfig) -> None:
            super().__init__()
            self.config = config
            self.sample_rate = config.sample_rate
            self.hop_length = config.hop_length
            self.encoder = Encoder(config)
            self.pre_block = AttnProjection(config.latent_dim, config.latent_channels,
                                            config.heads)
            self.mean_proj = nn.Conv1d(config.latent_channels, config.latent_channels, 1)
            #: ``logs_proj`` **推理不用** ✓（参考 ``encode`` 只回后验均值 ✓），但**检查点里有** ✓
            #: ⇒ 不建它，真权重装进来就多出一组没人认领的键 ✓✗（不建 ⇏ 省显存，只 ⇏ 装得上 ✓）。
            self.logs_proj = nn.Conv1d(config.latent_channels, config.latent_channels, 1)
            self.dec_in_proj = nn.Conv1d(config.latent_channels, config.latent_dim, 1)
            self.decoder = BigVGAN(config)
            #: 统计量在参考里是 **buffer** ✓（``latents_mean`` / ``latents_std`` 各 ``latent_channels`` 个 ✓）
            #: ⇒ 真权重会**直接盖过来** ✓；``config`` 里给了就**先写进去** ✓（两条来源同一处 ✓）。
            self.register_buffer("latents_mean", torch.zeros(config.latent_channels))
            self.register_buffer("latents_std", torch.ones(config.latent_channels))
            if config.uses_stats:
                self.latents_mean.copy_(torch.tensor(config.latents_mean, dtype=torch.float32))
                self.latents_std.copy_(torch.tensor(config.latents_std, dtype=torch.float32))

        # ── 归一化（统计随检查点走 ✓；恒等值 ⇒ 原样 ✓）──
        def _has_stats(self) -> bool:
            """检查点里是不是带**非平凡**统计 ✓（默认全 0 / 全 1 ⇒ 数学上就是恒等 ✓）。"""
            if not bool(torch.count_nonzero(self.latents_mean)):
                return not bool(torch.allclose(self.latents_std,
                                               torch.ones_like(self.latents_std)))
            return True

        def _stats(self, reference: Any, name: str) -> Any:
            if not self._has_stats():
                return None
            buffer = getattr(self, name)
            return buffer.to(dtype=reference.dtype, device=reference.device).view(1, -1, 1)

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
