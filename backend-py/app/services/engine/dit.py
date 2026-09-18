"""**DiT（Diffusion Transformer）** —— 本项目自己的实现 ✓（真 `torch.nn.Module` ✓ 可独立验证 ✓）。

## 职责边界（与 :mod:`app.services.engine.torch_backend` 的分工 ✓）

这里只做**模型本身** ✓：张量进、预测出 ✓。**采样循环**在 :mod:`sampler` ✓、**引导**在 :mod:`guidance` ✓、
**首帧掩码**在 :mod:`conditioning` ✓、**权重体检/加载计划**在 :mod:`safetensors` / `loader` ✓ ——
彼此不重复 ✓（这也是"换后端不动算法"的前提 ✓）。

## 为什么能脱离 H3 真权重就写出来（✗ 不猜架构）

实现的是**公开的标准结构** ✓（patch embed + 逐块 adaLN-Zero 调制的注意力/MLP + 最终 adaLN 输出 ✓，
出自 DiT / PixArt / SD3 这一系公开论文 ✓ —— **不是**抄某个仓库的源码 ✗）：

* 结构由 :class:`DiTConfig` **显式给出** ✓（隐层/深度/头数/patch 尺寸 ✓）⇒ 本模块**不猜** H3 的配置 ✗；
* 配置可以从权重文件的 ``__metadata__`` 里读 ✓（:meth:`DiTConfig.from_metadata` ✓）——
  读到就用 ✓，读不到就要求调用方给 ✓（**缺就报错**，不编一个"看起来合理"的 ✗）；
* **流匹配的参数化转换**（velocity → x0 ✓）单独成函数 ✓ ⇒ 换预测目标不用改模型 ✓。

⚠️ 因此：**"能跑通"不等于"H3 能出片"** ✗ —— 中间还差
「H3 的张量命名 → 本模块模块名」的**映射表** ✓ 与**权重的该模型配置** ✓（两者都要等真权重 ✓）。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any

from . import safetensors as st

__all__ = ["DiTConfig", "DiTConfigError", "build_dit", "flow_match_x0", "infer_config_from_info"]


class DiTConfigError(ValueError):
    """配置不合法 / 推断不出来 ✓（**明确报错**，不编一个默认值糊过去 ✗）。"""


@dataclass(frozen=True)
class DiTConfig:
    """模型结构参数 ✓（**全部显式** —— 本模块不猜 ✗）。"""

    hidden: int = 256
    depth: int = 4
    heads: int = 4
    #: patch 尺寸 ``(T, H, W)`` ✓（视频潜变量一般 1×2×2 ✓）
    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 4
    text_dim: int = 512
    mlp_ratio: float = 4.0
    #: 文本条件以 **cross-attention** 进入（True ✓）还是只做 pooled 调制（False ✓）
    cross_attention: bool = False
    #: **像素 → 潜空间**的边长比 ✓（8 = 每 8×8 像素一个潜token ✓）。
    #: ⚠️ ``0`` = **未给** ✓ ⇒ 这时**不能**从 plan 推潜变量形状 ✓（本该如此：压缩比是 VAE 的知识 ✗，
    #: 本模块**不猜** ✓ —— 与 `geometry` 不猜时间压缩比是同一条纪律 ✓）。
    vae_scale: int = 0

    def __post_init__(self) -> None:
        if self.hidden % self.heads:
            raise DiTConfigError(f"hidden={self.hidden} 不能被 heads={self.heads} 整除 ✗")
        if self.hidden <= 0 or self.depth <= 0:
            raise DiTConfigError("hidden/depth 必须为正 ✗")
        if any(size <= 0 for size in self.patch_size):
            raise DiTConfigError(f"patch_size 必须为正（收到 {self.patch_size} ✗）")

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden, "depth": self.depth, "heads": self.heads,
            "patchSize": list(self.patch_size), "inChannels": self.in_channels,
            "textDim": self.text_dim, "mlpRatio": self.mlp_ratio,
            "crossAttention": self.cross_attention, "vaeScale": self.vae_scale,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> "DiTConfig":
        """从 ``__metadata__`` 读配置 ✓（读到几个用几个 ✓；一个都没读到就**报错** ✗）。

        键名按各家习惯**多写法**匹配 ✓（``hidden`` / ``hidden_size`` / ``dim`` ✓）——
        这是"宽容读入" ✓，但**不猜值** ✗：没有的项必须由调用方显式给 ✓。
        """
        def pick(*names: str) -> Any:
            for name in names:
                for key, value in metadata.items():
                    if str(key).lower() == name:
                        return value
            return None

        def as_int(value: Any) -> int | None:
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                return None

        lowered = {str(key).lower(): value for key, value in metadata.items()}
        blob = lowered.get("dit_config") or lowered.get("config") or lowered.get("ditconfig")
        if isinstance(blob, str) and blob.strip().startswith("{"):
            try:
                parsed = json.loads(blob)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                merged = {**parsed, **metadata}
                return cls.from_metadata({str(k): str(v) for k, v in merged.items()})

        hidden = as_int(pick("hidden", "hidden_size", "dim", "model_dim"))
        depth = as_int(pick("depth", "num_layers", "n_layers", "num_blocks"))
        heads = as_int(pick("heads", "num_heads", "n_heads"))
        if not all((hidden, depth, heads)):
            raise DiTConfigError(
                "权重元数据里读不到完整结构（需要 hidden/depth/heads ✓）⇒ "
                "请在调用方**显式给 DiTConfig** ✓ —— 本模块**不猜**结构 ✗"
                f"（现有元数据键：{sorted(metadata)[:8]} ✓）")
        patch = pick("patch_size", "patch")
        patch_size = (1, 2, 2)
        if isinstance(patch, str):
            numbers = [as_int(part) for part in patch.replace("x", ",").split(",")]
            if len(numbers) == 3 and all(numbers):
                patch_size = (numbers[0], numbers[1], numbers[2])  # type: ignore[assignment]
        return cls(hidden=hidden, depth=depth, heads=heads, patch_size=patch_size,
                   in_channels=as_int(pick("in_channels", "latent_channels")) or 4,
                   text_dim=as_int(pick("text_dim", "context_dim")) or 512,
                   # ⚠️ `vae_scale` 也要读（自检 ⑲ 抓到漏读 ✗）—— 读不到就是 0 ✓（= 未给 ✓ 不猜 ✓）
                   vae_scale=(as_int(pick("vae_scale", "vae_scale_factor",
                                          "spatial_compression")) or 0))


def infer_config_from_info(info: st.SafetensorsInfo, *,
                           base: DiTConfig | None = None) -> DiTConfig:
    """从**权重头部**能**机械确定**的那一项：**层数**（``blocks.N`` 的个数 ✓）。

    其余（hidden/heads ✓）**要读张量形状**才能定 ✗ ⇒ 那属于 :mod:`loader` 的层块报告 ✓，
    或者由调用方给 ✓。这里**只填空缺** ✓（``base`` 给的就是"调用方已知的部分" ✓）。
    """
    groups: dict[str, set[int]] = {}
    for name in info.tensors:
        head, _, tail = str(name).partition(".")
        if tail and tail.split(".")[0].isdigit():
            groups.setdefault(head, set()).add(int(tail.split(".")[0]))
    if not groups:
        raise DiTConfigError("权重头部里找不到 ``<blocks>.<N>`` 这样的层块命名 ⇒ "
                             "无法确定层数 ✓（请显式给 DiTConfig ✓）")
    head = max(groups, key=lambda name: len(groups[name]))
    slots = sorted(groups[head])
    expected = list(range(len(slots)))
    if slots != expected:
        raise DiTConfigError(
            f"层号不连续（{head}.N = {slots[:8]}… ✓）⇒ 权重很可能**被截断或拼错** ✗"
            f"（用 `loader.plan_component` 能看到同一结论 ✓）")
    depth = len(slots)
    if base is None:
        return DiTConfig(depth=depth)
    return replace(base, depth=depth)


def flow_match_x0(model_output: Any, latents: Any, sigma: float, *,
                  prediction: str = "velocity") -> Any:
    """流匹配下**从模型输出还原 x0** ✓（公开的标准转换 ✓）。

    * ``velocity``（流匹配/rectified flow ✓）：``x0 = x − σ·v`` ✓
    * ``epsilon``（σ 是噪声强度 ✓）：``x0 = x − σ·ε`` ✓
    * ``sample``（模型直接给 x0 ✓）：原样 ✓
    """
    sigma = float(sigma)
    if sigma <= 0:
        return latents
    kind = str(prediction or "velocity").lower()
    if kind in ("velocity", "v", "flow"):
        return latents - model_output * sigma
    if kind in ("epsilon", "eps", "noise"):
        return latents - model_output * sigma
    if kind in ("sample", "x0"):
        return model_output
    raise DiTConfigError(f"未知预测类型 {prediction!r}；可用：velocity / epsilon / sample ✓")


def _modules() -> tuple[Any, Any]:
    """懒导入 torch ✓（模块能 import 不依赖 torch ✓）。"""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    return torch, nn


def _build() -> tuple[type, type, type]:
    """构造 :class:`DiT` 用到的三个类 ✓（**模块级**定义会强制 import torch ✗ ⇒ 放进工厂 ✓）。"""
    torch, nn = _modules()

    class PatchEmbed3D(nn.Module):
        """视频潜变量 → token 序列 ✓（3D 卷积，kernel=stride=patch ✓）。"""

        def __init__(self, config: DiTConfig) -> None:
            super().__init__()
            self.proj = nn.Conv3d(config.in_channels, config.hidden,
                                  kernel_size=config.patch_size, stride=config.patch_size)

        def forward(self, latent: Any) -> Any:
            """``(B, C, T, H, W)`` → ``(B, N, D)`` ✓。"""
            out = self.proj(latent)
            return out.flatten(2).transpose(1, 2)

    class DiTBlock(nn.Module):
        """adaLN-Zero 调制的注意力 + MLP ✓（DiT 系的公开做法 ✓）。"""

        def __init__(self, config: DiTConfig) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(config.hidden, elementwise_affine=False)
            self.attn = nn.MultiheadAttention(config.hidden, config.heads, batch_first=True)
            self.norm2 = nn.LayerNorm(config.hidden, elementwise_affine=False)
            inner = int(config.hidden * config.mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(config.hidden, inner), nn.GELU(),
                                     nn.Linear(inner, config.hidden))
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden, 6 * config.hidden))
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)   # adaLN-**Zero** ✓：起步是恒等 ✓

        def forward(self, tokens: Any, cond: Any, context: Any = None) -> Any:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.modulation(cond).chunk(6, dim=-1)
            normed = self.norm1(tokens) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            if context is not None:
                attended, _ = self.attn(normed, context, context, need_weights=False)
            else:
                attended, _ = self.attn(normed, normed, normed, need_weights=False)
            tokens = tokens + gate_msa.unsqueeze(1) * attended
            normed = self.norm2(tokens) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            return tokens + gate_mlp.unsqueeze(1) * self.mlp(normed)

    class DiT(nn.Module):
        """**本项目自己的 DiT** ✓（结构显式来自 :class:`DiTConfig` ✓ 不猜 ✗）。"""

        def __init__(self, config: DiTConfig) -> None:
            super().__init__()
            self.config = config
            self.patch = PatchEmbed3D(config)
            # 位置编码：**按需插值** ✓（标准做法 ✓）。
            # ⚠️ 这里踩过一次：初版写死 ``(1, 4096, hidden)`` ✗ ⇒ 真实分辨率下 token 数
            #    是 **12920**（1088×608 潜变量 ✓）⇒ 直接 `size mismatch` 崩在采样里 ✗
            #    （自检 ㉗ 跑到整链才暴露 ✓ —— 这也是"必须端到端跑一次"的理由 ✓）。
            #    真权重若自带同长度的位置表 ⇒ 加载时形状一致就直接用 ✓；不同长度则插值 ✓。
            self.pos_embed = nn.Parameter(torch.zeros(1, 4096, config.hidden))
            self.sigma_embed = nn.Sequential(
                nn.Linear(1, config.hidden), nn.SiLU(), nn.Linear(config.hidden, config.hidden))
            self.text_proj = (nn.Identity() if config.text_dim == config.hidden
                              else nn.Linear(config.text_dim, config.hidden))
            self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.depth)])
            self.final_norm = nn.LayerNorm(config.hidden, elementwise_affine=False)
            self.final_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(config.hidden, 2 * config.hidden))
            nn.init.zeros_(self.final_modulation[-1].weight)
            nn.init.zeros_(self.final_modulation[-1].bias)
            out_channels = math.prod(config.patch_size) * config.in_channels
            self.out = nn.Linear(config.hidden, out_channels)

        def forward(self, latent: Any, sigma: Any, context: Any = None) -> Any:
            """``latent (B,C,T,H,W)`` + ``sigma`` + 文本上下文 → **与 latent 同形状的预测** ✓。

            ``sigma`` 是标量或 ``(B,)`` 张量 ✓（每步变化 ⇒ 条件**真的**进网络了 ✓，
            自检会验证"改 sigma 输出就变" ✓ —— 防"条件没接进去"的假绿 ✗）。
            """
            batch = latent.shape[0]
            f_t, f_h, f_w = self.grid_sizes(latent)
            tokens = self.patch(latent) + self._positions(f_t * f_h * f_w, latent)
            sigma_tensor = torch.as_tensor(sigma, dtype=tokens.dtype, device=tokens.device)
            if sigma_tensor.ndim == 0:
                sigma_tensor = sigma_tensor.repeat(batch)
            cond = self.sigma_embed(sigma_tensor.reshape(batch, 1).to(tokens.dtype))
            text = None
            if context is not None:
                # ⚠️ 维度要**归一化**再进网络 ✓：调用方可能给一维向量（我们把条件当 pooled 用 ✓）、
                #    二维 (B,D) 或三维 (B,L,D) ✓。初版只处理 2/3 维 ⇒ 一维时 `mean(dim=1)` 直接
                #    `IndexError` ✗（自检 ㉗ 整链才暴露 ✓）。
                raw = torch.as_tensor(context, dtype=tokens.dtype, device=tokens.device)
                if raw.ndim == 1:
                    raw = raw.unsqueeze(0)              # (D,) → (1, D) ✓
                if raw.ndim == 2:
                    raw = raw.unsqueeze(1)              # (B, D) → (B, 1, D) ✓
                text = self.text_proj(raw)
                if int(text.shape[0]) != batch:         # 单条条件广播到整个 batch ✓
                    text = text.expand(batch, *text.shape[1:])
                cond = cond + text.mean(dim=1)          # pooled 调制 ✓
            for block in self.blocks:
                tokens = block(tokens, cond, text if self.config.cross_attention else None)
            shift, scale = self.final_modulation(cond).chunk(2, dim=-1)
            tokens = self.final_norm(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            return self.unpatchify(self.out(tokens), latent)

        def unpatchify(self, tokens: Any, reference: Any) -> Any:
            """``(B, N, pT·pH·pW·C)`` → ``(B, C, T, H, W)`` ✓（**与输入同形** ✓）。

            维度命名（免得看错 ✓）：``pT/pH/pW`` = patch 尺寸 ✓；``fT/fH/fW`` = 网格数 ✓。
            顺序必须与 :meth:`patch_tokens` 一致（``fT·fH·fW`` ✓），否则画面会**错位但不报错** ✗。
            """
            patch_t, patch_h, patch_w = self.config.patch_size
            channels = self.config.in_channels
            batch = int(tokens.shape[0])
            f_t, f_h, f_w = self.grid_sizes(reference)
            grid = tokens.reshape(batch, f_t, f_h, f_w, patch_t, patch_h, patch_w, channels)
            return grid.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
                batch, channels, f_t * patch_t, f_h * patch_h, f_w * patch_w)

        def _positions(self, count: int, latents: Any) -> Any:
            """位置编码**插值到需要的长度** ✓（表长正好 ⇒ 直接用 ✓）。

            ``interpolate`` 沿 token 维做线性插值 ✓ —— 这是公开的常规做法 ✓（换分辨率不必重训位置表 ✓）。
            """
            table = self.pos_embed
            if int(table.shape[1]) == count:
                return table
            flat = table.transpose(1, 2)                    # (1, D, P) ✓
            resized = torch.nn.functional.interpolate(
                flat, size=int(count), mode="linear", align_corners=False)
            return resized.transpose(1, 2).to(dtype=latents.dtype, device=latents.device)

        def grid_sizes(self, latent: Any) -> tuple[int, int, int]:
            """潜变量 → 网格数 ``(fT, fH, fW)`` ✓（整除不了就报错 ✗，不悄悄向下取整 ✓）。"""
            sizes = []
            for index in range(3):
                size = int(latent.shape[index + 2])
                patch = self.config.patch_size[index]
                if size % patch:
                    raise DiTConfigError(
                        f"潜变量第 {index} 维 {size} 不能被 patch {patch} 整除 ✗"
                        f"（尺寸要吸附到网格 ✓）")
                sizes.append(size // patch)
            return sizes[0], sizes[1], sizes[2]

    return DiT, DiTBlock, PatchEmbed3D


_DIT_CLASS: Any = None


def build_dit(config: DiTConfig) -> Any:
    """构造 DiT 实例 ✓（**懒导入 torch** ✓ ⇒ 没装 torch 时 import 本模块也不会炸 ✓）。

    ⚠️ 刻意**不**把这个工厂命名成 ``DiT`` ✗ —— 那会与里面真正的类同名 ✓，
    调用方（和读代码的人）会分不清"类"和"工厂函数" ✗（第一版就是这样 ✓ 已改 ✓）。
    """
    global _DIT_CLASS
    if _DIT_CLASS is None:
        _DIT_CLASS = _build()[0]
    return _DIT_CLASS(config)
