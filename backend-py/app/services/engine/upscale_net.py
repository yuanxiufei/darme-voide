"""**超清放大器的网络本体**（clean-latent 2× ✓ V2 主干 + V3 因子化注意力 ✓ 2026-09-24 补 ✓）。

## 出处与许可（⚠️ 这条决定了能不能写 ✓）
上游实现把许可写在文件头 ✓：**Mamad8 权重、MIT 代码**（`github.com/mamad8c/ComfyUI-H3-Latent-Upscaler-Mamad8` ✓）
⇒ 本仓**可以实现** ✓（保留出处署名 ✓）。检查点格式串已核到真值 ✓
（`minimax_h3_clean_latent_upscaler_v3_factorized_attention` ✓ ⇒ 结构 = V2 + V3 ✓），
字段集/值域的**严格校验**在 :mod:`app.services.engine.upscale` ✓（本模块只负责**结构** ✓）。

## 结构（与规格逐条对应 ✓，规格见 `.codebuddy/memory/TOPICS.md` §超清放大器的架构规格 ✓）
* **两层残差** ✓：V2 内 ``forward = bilinear2x(low) + correction(low)`` ✓；V3 外 ``base + delta`` ✓；
* ⭐ **绝不在时间维插值** ✗ 有两层保险 ✓：双线性把 ``T`` 折进 batch ✓、残差块时间 padding = ``k//2`` ✓；
* 通道→空间用 ``pixel_shuffle`` ✓（``to_high`` 出 ``refine×4`` ✓；V3 的 ``to_delta`` 出 ``24×4`` ✓）；
* refine 段把**双线性基底拼进通道** ✓（``refine_channels + 24`` ✓）；
* V3 的窗口注意力 ⚠️ **shifted 按 ``index % 2`` 交替** ✓（算完要**裁回** ✗，否则窗口边界永远在同一处 ✓✗）；
* 时间注意力把 ``H/W`` 折进 batch ✓；局部分支是**深度可分离**卷积 ✓；MLP 用 **GELU** ✓。

## 不猜 / 边界
* ⚠️ **没装 torch** ⇒ 本模块能 import ✓，但 :func:`build_upscaler` 会**明确报错** ✗（不静默降级 ✓）；
* ⚠️ 装载一律 **``strict=True``** ✓：缺键/多键 ⇒ 报错并写明「张量与声明的架构不符」✓✗；
* ⚠️ 本模块只保证**结构正确** ✓ —— 「画质」要真权重 + 真采样才谈得上 ✓✗。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["H3LatentUpscalerV2", "H3LatentUpscalerV3", "RESIDUAL_DILATIONS", "UpscalerNetError",
           "build_upscaler", "groups_for", "has_torch", "spatial_bilinear_2x",
           "spatial_pixel_shuffle_2x"]

#: 低清段的**空间膨胀循环** ✓（口径：上游按 ``i % 4`` 取 ✓）。
RESIDUAL_DILATIONS: tuple[int, ...] = (1, 2, 1, 3)

try:  # pragma: no cover - 环境相关 ✓（没装 torch 时下面那三行是 None ✓）
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False


class UpscalerNetError(RuntimeError):
    """网络**建不起来 / 装不上** ✓ ⇒ 当场报 ✗（静默降级会让人以为超清开着 ✓✗）。"""


def has_torch() -> bool:
    """装了 torch 吗 ✓（自检据此 SKIP ✓ —— **没跑 ≠ 绿** ✗）。"""
    return _HAS_TORCH


def groups_for(channels: int) -> int:
    """``GroupNorm`` 的组数 ✓：``min(16, C)`` 起**往下**找能整除的 ✓（口径来自上游 ✓）。"""
    groups = min(16, max(1, int(channels)))
    while groups > 1 and int(channels) % groups:
        groups -= 1
    return groups


if _HAS_TORCH:

    def spatial_bilinear_2x(x: Any) -> Any:
        """空间 2× 双线性 ✓（⚠️ **不在时间维插值** ✗：把 ``T`` 折进 batch 再插 ✓）。"""
        batch, channels, frames, height, width = x.shape
        folded = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        up = F.interpolate(folded, size=(height * 2, width * 2), mode="bilinear",
                           align_corners=False)
        return up.view(batch, frames, channels, height * 2, width * 2).permute(0, 2, 1, 3, 4)

    def spatial_pixel_shuffle_2x(x: Any, out_channels: int) -> Any:
        """通道→空间 2× ✓（输入通道必须 ``out_channels * 4`` ✓，不然**报错** ✗ 不硬来 ✓）。"""
        batch, channels, frames, height, width = x.shape
        if int(channels) != int(out_channels) * 4:
            raise UpscalerNetError(
                f"pixel_shuffle 要求输入通道 = {out_channels} × 4 ✗，实得 {channels} ✓"
                f"（通道数对不上说明结构或权重装错了 ✓✗）")
        x = x.view(batch, out_channels, 2, 2, frames, height, width)
        return x.permute(0, 1, 4, 5, 2, 6, 3).reshape(batch, out_channels, frames, height * 2,
                                                     width * 2)

    class ResidualBlock3d(nn.Module):
        """``norm→SiLU→conv`` 做两遍 + **残差** ✓（⚠️ 时间 padding = ``k//2`` ✓ 保住时间对齐 ✗）。"""

        def __init__(self, channels: int, temporal_kernel: int = 3, spatial_dilation: int = 1):
            super().__init__()
            pad = (temporal_kernel // 2, spatial_dilation, spatial_dilation)
            dilation = (1, spatial_dilation, spatial_dilation)
            self.norm1 = nn.GroupNorm(groups_for(channels), channels, eps=1e-6)
            self.conv1 = nn.Conv3d(channels, channels, (temporal_kernel, 3, 3), padding=pad,
                                   dilation=dilation)
            self.norm2 = nn.GroupNorm(groups_for(channels), channels, eps=1e-6)
            self.conv2 = nn.Conv3d(channels, channels, (temporal_kernel, 3, 3), padding=pad,
                                   dilation=dilation)

        def forward(self, x: Any) -> Any:
            residual = self.conv1(F.silu(self.norm1(x)))
            residual = self.conv2(F.silu(self.norm2(residual)))
            return x + residual

    class H3LatentUpscalerV2(nn.Module):
        """V2 主干 ✓：低清特征 → （膨胀循环残差块）→ 通道→空间 2× → 拼基底精修 → 出修正量 ✓。"""

        def __init__(self, in_channels: int = 24, hidden_channels: int = 128, num_blocks: int = 6,
                     refine_channels: int = 64, refine_blocks: int = 2, temporal_kernel: int = 3):
            super().__init__()
            pad = (temporal_kernel // 2, 1, 1)
            self.in_channels = int(in_channels)
            self.refine_channels = int(refine_channels)
            self.temporal_kernel = int(temporal_kernel)
            self.low_stem = nn.Conv3d(in_channels, hidden_channels, (temporal_kernel, 3, 3),
                                      padding=pad)
            self.low_blocks = nn.ModuleList(
                ResidualBlock3d(hidden_channels, temporal_kernel,
                                RESIDUAL_DILATIONS[index % len(RESIDUAL_DILATIONS)])
                for index in range(num_blocks))
            self.low_norm = nn.GroupNorm(groups_for(hidden_channels), hidden_channels, eps=1e-6)
            self.to_high = nn.Conv3d(hidden_channels, refine_channels * 4,
                                     (temporal_kernel, 3, 3), padding=pad)
            self.refine_stem = nn.Conv3d(refine_channels + in_channels, refine_channels,
                                         (temporal_kernel, 3, 3), padding=pad)
            self.refine_blocks = nn.ModuleList(
                ResidualBlock3d(refine_channels, temporal_kernel) for _ in range(refine_blocks))
            self.out_norm = nn.GroupNorm(groups_for(refine_channels), refine_channels, eps=1e-6)
            self.out = nn.Conv3d(refine_channels, in_channels, (temporal_kernel, 3, 3), padding=pad)

        def correction(self, low: Any) -> Any:
            base = spatial_bilinear_2x(low)
            features = self.low_stem(low)
            for block in self.low_blocks:
                features = block(features)
            features = self.to_high(F.silu(self.low_norm(features)))
            features = spatial_pixel_shuffle_2x(features, self.refine_channels)
            features = self.refine_stem(torch.cat([features, base], dim=1))
            for block in self.refine_blocks:
                features = block(features)
            return self.out(F.silu(self.out_norm(features)))

        def forward(self, low: Any) -> Any:
            """⭐ **残差式超分** ✓：双线性基底 + 学出来的修正量 ✓。"""
            return spatial_bilinear_2x(low) + self.correction(low)

    class FactorizedBlock(nn.Module):
        """**因子化注意力** ✓：空间窗口注意力（可 shifted ✓）+ 时间注意力 + 深度可分离卷积 + MLP ✓。"""

        def __init__(self, width: int, heads: int, window: int, shifted: bool = False,
                     mlp_ratio: int = 4):
            super().__init__()
            self.window = int(window)
            self.shift = self.window // 2 if shifted else 0
            self.spatial_norm = nn.LayerNorm(width, eps=1e-6)
            self.spatial = nn.MultiheadAttention(width, heads, batch_first=True)
            self.temporal_norm = nn.LayerNorm(width, eps=1e-6)
            self.temporal = nn.MultiheadAttention(width, heads, batch_first=True)
            self.local_norm = nn.GroupNorm(1, width, eps=1e-6)
            self.local = nn.Conv3d(width, width, 3, padding=1, groups=width)
            hidden = width * int(mlp_ratio)
            self.mlp_norm = nn.LayerNorm(width, eps=1e-6)
            self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

        def _spatial(self, x: Any) -> Any:
            batch, channels, frames, height, width = x.shape
            shift, window = self.shift, self.window
            if shift:
                # ⚠️ shifted 版先挪半窗 ✓ —— 否则窗口边界**永远落在同一处** ✗✗
                x = F.pad(x, (shift, 0, shift, 0))
            pad_h, pad_w = x.shape[-2:]
            x = F.pad(x, (0, (-pad_w) % window, 0, (-pad_h) % window))
            full_h, full_w = x.shape[-2:]
            tokens = (x.permute(0, 2, 3, 4, 1)
                      .reshape(batch * frames, full_h // window, window, full_w // window, window,
                               channels)
                      .permute(0, 1, 3, 2, 4, 5)
                      .reshape(-1, window * window, channels))
            query = self.spatial_norm(tokens)
            tokens = tokens + self.spatial(query, query, query, need_weights=False)[0]
            x = (tokens.reshape(batch * frames, full_h // window, full_w // window, window, window,
                                channels)
                 .permute(0, 1, 3, 2, 4, 5)
                 .reshape(batch, frames, full_h, full_w, channels)
                 .permute(0, 4, 1, 2, 3))
            x = x[..., :pad_h, :pad_w]
            if shift:
                # ⚠️ 挪过就必须**裁回原位置** ✓✗
                return x[..., shift:shift + height, shift:shift + width]
            return x[..., :height, :width]

        def forward(self, x: Any) -> Any:
            x = self._spatial(x)
            batch, channels, frames, height, width = x.shape
            tokens = x.permute(0, 3, 4, 2, 1).reshape(batch * height * width, frames, channels)
            query = self.temporal_norm(tokens)
            tokens = tokens + self.temporal(query, query, query, need_weights=False)[0]
            x = tokens.reshape(batch, height, width, frames, channels).permute(0, 4, 3, 1, 2)
            x = x + self.local(F.silu(self.local_norm(x)))
            tokens = x.permute(0, 2, 3, 4, 1)
            return x + self.mlp(self.mlp_norm(tokens)).permute(0, 4, 1, 2, 3)

    class H3LatentUpscalerV3(nn.Module):
        """V3 ✓ = **V2 当 base** + 因子化注意力算出的 ``delta`` ✓（``base + delta`` ✓）。"""

        def __init__(self, base_config: Mapping[str, Any], config: Mapping[str, Any]):
            super().__init__()
            self.base_config = dict(base_config)
            self.config = dict(config)
            self.base = H3LatentUpscalerV2(**{key: int(value) for key, value in base_config.items()})
            width = int(config["width"])
            in_channels = int(base_config["in_channels"])
            self.stem = nn.Conv3d(in_channels, width, 3, padding=1)
            self.blocks = nn.ModuleList(
                FactorizedBlock(width, int(config["heads"]), int(config["window"]),
                                bool(index % 2), int(config["mlp_ratio"]))
                for index in range(int(config["blocks"])))
            self.norm = nn.GroupNorm(1, width, eps=1e-6)
            self.to_delta = nn.Conv3d(width, in_channels * 4, 3, padding=1)

        def forward(self, low: Any) -> Any:
            base = self.base(low)
            x = self.stem(low)
            for block in self.blocks:
                x = block(x)
            delta = spatial_pixel_shuffle_2x(self.to_delta(F.silu(self.norm(x))),
                                             self.base_config["in_channels"])
            return base + delta


def build_upscaler(state_dict: Mapping[str, Any], *, base_config: Mapping[str, Any],
                   config: Mapping[str, Any], strict: bool = True) -> Any:
    """按契约装配并**严格装载** ✓ ⇒ ``eval()`` 的模型 ✓（⚠️ 没装 torch / 键对不上 ⇒ **报错** ✗）。

    ⚠️ ``strict=True`` 是**故意的** ✗：缺键/多键说明「张量与声明的架构不符」✓ ⇒ 宁可拒 ✓
    （静默 ``strict=False`` 会留下随机初始化的层，画面看着像超分、其实是噪声 ✓✗）。
    """
    if not _HAS_TORCH:
        raise UpscalerNetError(
            "没装 torch ✗ ⇒ 建不了超清放大器 ✓（本仓**不静默降级** ✗：要么装 torch，"
            "要么在计划层按 `upscale.plan_upscale(contract=None)` 走**回退普通模式** ✓）")
    model = H3LatentUpscalerV3(base_config, config)
    try:
        model.load_state_dict(dict(state_dict), strict=bool(strict))
    except RuntimeError as err:
        raise UpscalerNetError(
            f"放大器的张量与声明的架构**不符** ✗：{err} ✓ ⇒ 检查点与契约不是一套 ✓✗"
            f"（⚠️ 不许 ``strict=False`` 糊过去 ✗）") from err
    return model.eval().requires_grad_(False)
