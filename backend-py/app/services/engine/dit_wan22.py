"""Wan2.2 I2V 14B（A14B 高/低噪声双专家之一）DiT 骨架 —— **逐行对着 ComfyUI 转写**。

盘上：``models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors``
（**1908 键** = 40 块 × 47 + 顶层 28；真权重键 1095 = 15 + 40×27，其余是 fp8 的
``*.scale_weight`` / ``*.scale_input`` 与 0 元素哨兵 ``scaled_fp8``，都由
:mod:`app.services.engine.weights` 的 fp8 通路吃掉，不进 ``state_dict``）。

## 这一族的形状（全部从权重形状读出，只有 ``heads`` 是查表来的）

============================  ==========================================
``dim``                       5120（``patch_embedding.weight`` 首维）
``in_dim``                    36（``patch_embedding.weight`` 第二维）
``ffn_dim``                   13824（``blocks.0.ffn.0.weight`` 首维）
``out_dim``                   16（``head.head.weight`` 首维 64 ÷ patch 体积 4）
``text_dim``                  4096（``text_embedding.0.weight`` 次维，umt5-XXL）
``freq_dim``                  256（``time_embedding.0.weight`` 次维）
``layers``                    40（数 ``blocks.N``）
``heads``                     **40** —— 见下面的「唯一的查表项」
============================  ==========================================

## 唯一的查表项：``heads``

``q``/``k``/``v`` 在盘上都是**单条** ``[5120, 5120]``（没有分组维可数），``norm_q`` 又是
``RMSNorm(5120)``（宽 = ``dim`` 而不是 ``head_dim``）⇒ **头数在权重里读不出来**，这与 Flux 不同
（Flux 的 ``query_norm.scale`` 宽就等于 ``head_dim``，可反推）。

所以 ``heads`` 走两条路，二者都要过同一道校验（``dim % heads == 0`` 且三轴 RoPE 必须是**非零偶数**）：

1. 调用方显式给 ``heads=``；
2. 不给则按 ``(dim, ffn_dim, layers)`` 三元组查 :data:`HEADS_BY_GEOMETRY`。

三元组约束下是**可核的架构事实**而不是猜：14B 就是 ``(5120, 13824, 40)``，而 ``dim``/``ffn_dim``/
``layers`` 三个数各自都是权重直接读出的 ⇒ 三者同时对上才认，任一不符就抛 ``ShapeError`` 要求显式给
（宁可不建，也不按 ``dim // 128`` 之类"经验比例"瞎猜 ✗）。

## 36 通道的 I2V 输入

``in_dim = 36 = 16 + 4 + 16``，三段依次是 :data:`I2V_LAYOUT`：**噪声 latent 16 通道 + 时间掩码 4
通道 + 条件图 latent 16 通道**。拼接**不在骨架内**（参考实现也在 ``model_base.concat_cond`` 里拼，
``WanModel`` 只吃拼好的 36 通道）—— 骨架只做三件事：校验 ``x.shape[1] == in_dim``、提供
:func:`i2v_concat` 让上层按**同一份布局**拼、提供 :func:`split_i2v` 让人能把 36 通道切回三段来核。

★**掩码在条件图之前**（这条曾写反过，是最难从症状反推的一种错）：参考实现
``comfy/model_base.py::WAN21.concat_cond`` 在 ``concat_cond_index == 0`` 时先
``cond_concat.append(torch.cat((cond, kwargs["concat_image"]), dim=1))``，而 ``cond`` 就是 ``1 - mask``
的 4 通道 ⇒ 模型看到的是 ``[噪声 16, 掩码 4, 条件图 16]``。``WanImageToVideo`` 节点产出的
``concat_latent_image`` 是 16 通道、``concat_mask`` 是 1 通道（``concat_cond`` 里 ``repeat`` 成 4），
两者顺序写反不会报任何错（通道数照样是 36），只是条件图被当成掩码读、出片完全不受首帧约束。

★掩码为什么是 4 通道：像素帧按 4 倍时间下采样并进残差帧后，被并掉的 4 帧落在通道维（参考实现
``mask.view(b, T//4, 4, H, W).transpose(1, 2)``），所以它和 latent 帧数同长。语义是
**首帧 latent 那一拍为 1、其余为 0**（``WanImageToVideo`` 里 ``mask`` 先置 1 再把首拍置 0，
``concat_cond`` 又取 ``1 - mask`` 翻回来）。

## 与 Flux 族的三处口径差别（都\"不报错只是图不对\"）

1. **``t`` 的 1000 倍在哪乘**：参考实现里 ``sinusoidal_embedding_1d`` 吃的是 ``1000·σ``，但那次缩放**不在
   模型里** —— 它发生在采样器（``model_base._apply_model`` 先 ``model_sampling.timestep(t)``，
   Wan 用 ``ModelSamplingDiscreteFlow``，其 ``timestep = σ·multiplier``，``multiplier`` 默认 **1000**）。
   本仓 ``TorchBackend.denoise`` 交给骨架的是**裸 σ**（与 Z-Image / Flux 一致）⇒ 骨架必须自己
   :data:`TIMESTEP_SCALE` 补这一次乘法。漏了不会报错：``e`` 的频率只对应 σ∈[0,1]，
   σ=0.9 时等于让模型以为自己是 σ=0.0009 ⇒ 整段采样都在"接近干净图"的假设下走，出片是糊/噪声。
   频率底数是 ``10000^(-i/half)``（``half = freq_dim//2 = 128``，分母是 half 不是 dim）。
2. **位置号从 0 起、三轴都**不**居中**，且三轴宽度是 ``[44, 42, 42]``（``head_dim=128`` 按
   ``[d-4*(d//6), 2*(d//6), 2*(d//6)]`` 拆）—— 不是等分三份。
3. **``e`` 是 4 维的**：``time_projection`` 出 ``6·dim`` 后 ``unflatten(2, (6, dim))`` ⇒
   ``[B, 1, 6, dim]``，块内沿 ``dim=2`` 的 ``unbind(2)`` 才是 6 路调制（Flux 是 ``chunk(6, dim=1)``）。
   照抄 Flux 的 ``chunk(6, dim=1)`` 会得到 6 个 ``[B,1,6,dim]``、**形状在广播时才对不上**，报错点
   落在 ``addcmul`` 上、读起来像 e 没算对。

## 已知边界（如实标）

* **双专家**：A14B 是 high/low noise 两份独立权重，接力在采样器侧（:mod:`app.services.engine.dit_build`
  只负责"建一个专家"）。``high_noise`` 与 ``low_noise`` 的键集合完全一样，骨架无法分辨、也不需要分辨。
* **``patch_embedding`` 的精度**：参考实现让它常驻 ``float32`` 并在前向里 ``x.float()``（fp16 时代
  防溢出）；本仓全链统一精度（bf16/fp32 由设备决定），conv 在权重精度下算。bf16 的指数范围与 fp32
  同为 8 位 ⇒ 溢出风险不存在，差异只在 ~1e-3 相对量级，且参考实现算完也要 cast 回 bf16。
* **``img_emb`` / ``k_img`` / ``v_img`` 不存在**：那是 Wan2.1 I2V 的写法（CLIP 视觉特征走单独一路
  交叉注意力）。Wan2.2 14B I2V 改成**通道拼接**（``in_dim=36``），盘上 47 键/块里也确实没有
  ``k_img``/``v_img``/``norm_k_img`` ⇒ 本骨架按 ``t2v_cross_attn`` 形态实现。
  :func:`expected_keys` 是**逐键清单**，真有这些键的变体会被当场报 ``missing``，不会静默丢。
* **``pad_to_patch_size`` 不补边**：参考实现在 ``t/h/w`` 上补到 patch 的整数倍再切回；本仓直接抛
  ``ShapeError``（补边是悄悄改构图 ✗）。Wan 的 latent 空间下采样是 8×、``h/w = 像素/8``，
  720p 档 90×160、480p 档 60×80 都是 patch(2) 的整数倍 ⇒ 正常路径碰不到。
* **RoPE 的 ``rope_options`` / ``source_id`` / ``context_latents`` 扩展**（Bernini 参考帧、shot 缩放）
  不在骨架里：那是 ``transformer_options`` 的注入项，本仓没有对应通路。

## 接口（与其它族一致）

``infer_shape(shapes, *, heads=None)`` / ``expected_keys(shape)`` / ``build(shape, *, dtype=None)`` /
``build_from_headers(...)``（:mod:`dit_build` 用前者）/ ``WanDiT.forward(latents, sigma, context)``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .dit_generic import ShapeError, apply_rope, rope_freqs as _rope_freqs_nd

__all__ = [
    "WAN_EPS",
    "ROPE_THETA",
    "PATCH_SIZE",
    "I2V_LAYOUT",
    "HEADS_BY_GEOMETRY",
    "WanShape",
    "WanBlock",
    "WanDiT",
    "apply_rope",
    "build",
    "expected_keys",
    "i2v_concat",
    "i2v_condition_mask",
    "infer_shape",
    "rope_freqs",
    "sinusoidal_embedding_1d",
    "split_i2v",
    "unpatchify",
]

#: 参考实现写死的 eps（``WanModel(eps=1e-6)``），所有 LayerNorm/RMSNorm 共用。
WAN_EPS = 1e-6
#: σ ⇒ 时刻的倍数（``ModelSamplingDiscreteFlow.multiplier`` 默认 1000）。
#:
#: ⚠️ 参考实现的这次乘法在**采样器**里（``model_base._apply_model`` 的
#: ``model_sampling.timestep(t)``），不在 ``WanModel`` 里 ✗ —— 本仓骨架收裸 σ（同 Z-Image /
#: Flux）⇒ 必须自己补。补错的症状：不报错、图糊（详见模块抬头第 1 条）。
TIMESTEP_SCALE = 1000.0
#: RoPE 底数（``EmbedND(theta=10000.0)``）。
ROPE_THETA = 10000.0
#: ``(t, h, w)`` 三轴的 patch 尺寸（Wan 只在空间上切 2×2）。
PATCH_SIZE = (1, 2, 2)

#: I2V 的 36 通道布局：**顺序固定**，`split_i2v` / `i2v_concat` 都按它走。
#:
#: ★顺序以 ``comfy/model_base.py::WAN21.concat_cond`` 为准：掩码在前、条件图在后
#: （``cond_concat_index == 0`` ⇒ ``cat((1 - mask, concat_image))``）。
I2V_LAYOUT: tuple[tuple[str, int], ...] = (("noise", 16), ("mask", 4), ("image", 16))

#: ``heads`` 的查表：键是**三个都能从权重直接读出的数**拼成的三元组。
#: 只登记能确证的几何；不在表里的必须显式给 ``heads=``（不按"每头 128 维"之类的经验比例猜 ✗）。
HEADS_BY_GEOMETRY: Mapping[tuple[int, int, int], int] = {
    (5120, 13824, 40): 40,  # Wan2.1/Wan2.2 A14B（T2V 与 I2V 同几何）
}


@dataclass(frozen=True)
class WanShape:
    """把权重形状翻译成前向需要的一切（并当场校验不自洽的组合）。"""

    dim: int
    ffn_dim: int
    in_dim: int
    out_dim: int
    text_dim: int
    freq_dim: int
    heads: int
    layers: int
    patch: tuple[int, int, int] = PATCH_SIZE
    eps: float = WAN_EPS
    theta: float = ROPE_THETA

    def __post_init__(self) -> None:
        for name in ("dim", "ffn_dim", "in_dim", "out_dim", "text_dim", "freq_dim",
                     "heads", "layers"):
            value = getattr(self, name)
            if value <= 0:
                raise ShapeError(f"{name}={value} 非法 ✗（必须是正整数）")
        if len(self.patch) != 3 or any(p <= 0 for p in self.patch):
            raise ShapeError(f"patch={self.patch} 非法 ✗（要三个正整数）")
        if self.dim % self.heads:
            raise ShapeError(
                f"dim={self.dim} 不能被 heads={self.heads} 整除 ✗ ⇒ head_dim 不是整数，"
                f"RoPE 的三轴拆不开")
        if self.freq_dim % 2:
            raise ShapeError(f"freq_dim={self.freq_dim} 是奇数 ✗（正弦嵌入要成对切分）")
        # 三轴必须**非零偶数**：``rope`` 按 2x2 旋转块配对，奇数轴会在 einsum 后 reshape 失败
        # （症状是报错落在 rope_freqs 里、像是位置号算错 —— 与 Flux 那边同一种坑 ✓）。
        object.__setattr__(self, "_axes", self._axes_of(self.head_dim))

    @staticmethod
    def _axes_of(head_dim: int) -> tuple[int, int, int]:
        """参考实现的拆法：``[d - 4*(d//6), 2*(d//6), 2*(d//6)]``（``d=128 ⇒ 44/42/42``）。"""
        d = int(head_dim)
        six = d // 6
        axes = (d - 4 * six, 2 * six, 2 * six)
        if sum(axes) != d:
            raise ShapeError(f"三轴 {axes} 之和 ≠ head_dim {d} ✗")
        for index, axis in enumerate(axes):
            if axis <= 0 or axis % 2:
                raise ShapeError(
                    f"head_dim={d} 拆出的轴 {index}={axis} 不是正偶数 ✗ "
                    f"（三轴 {axes}；rope 按 2×2 配对，奇数轴建不出）")
        return axes

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def axes_dim(self) -> tuple[int, int, int]:
        """三轴 RoPE 宽度（``t``/``h``/``w``，和为 ``head_dim``）。"""
        return self._axes  # type: ignore[attr-defined]

    @property
    def patch_volume(self) -> int:
        return self.patch[0] * self.patch[1] * self.patch[2]

    @property
    def latent_channels(self) -> int:
        """去噪侧通道数（= ``out_dim``，与 ``vae_wan`` 的 16 通道对得上）。"""
        return self.out_dim

    @property
    def i2v_layout(self) -> tuple[tuple[str, int], ...]:
        return I2V_LAYOUT


def split_i2v(shape: WanShape, x: Tensor) -> dict[str, Tensor]:
    """把 ``(b, 36, t, h, w)`` 切回 ``{"noise": 16, "mask": 4, "image": 16}``。

    只做**切分**、不构造（拼接与掩码的生成属于上层，见模块 docstring「36 通道」一节）。
    """
    if x.ndim != 5:
        raise ShapeError(f"I2V 输入要 5 维 (b,c,t,h,w) ✗，实际 {tuple(x.shape)}")
    if x.shape[1] != shape.in_dim:
        raise ShapeError(f"I2V 输入通道 {x.shape[1]} ✗ ≠ 形状表声明的 in_dim {shape.in_dim}")
    pieces: dict[str, Tensor] = {}
    start = 0
    for name, width in I2V_LAYOUT:
        pieces[name] = x[:, start:start + width]
        start += width
    if start != shape.in_dim:
        raise ShapeError(
            f"I2V 布局 {I2V_LAYOUT} 加起来是 {start} ✗ ≠ in_dim {shape.in_dim}"
            f"（这套布局只对 14B I2V 成立，别的变体必须显式说明通道含义）")
    return pieces


def i2v_condition_mask(*, frames: int, height: int, width: int, dtype=None,
                       device=None, batch: int = 1) -> Tensor:
    """造 ``[b, 4, T_lat, h_lat, w_lat]`` 的 I2V 掩码：**首拍为 1、其余为 0**。

    与 ``comfy/model_base.py::WAN21.concat_cond`` 的 ``1 - denoise_mask`` 等价：
    ``WanImageToVideo`` 造掩码时先把全部置 1、再把首拍置 0，``concat_cond`` 再翻一次 ⇒
    模型看到的掩码在**被条件住的那一拍是 1**。4 通道是把 ``mask.view(b, T//4, 4, H, W)``
    的那个残差帧维展开（像素 4 帧并进 1 个 latent 帧）。
    """
    if frames < 1 or height < 1 or width < 1:
        raise ShapeError(
            f"I2V 掩码尺寸非法 ✗：frames={frames} height={height} width={width}")
    mask = torch.zeros((batch, 1, frames, height, width), dtype=dtype or torch.float32,
                       device=device)
    mask[:, :, 0] = 1.0
    return mask.repeat(1, 4, 1, 1, 1)


def i2v_concat(noise: Tensor, image: Tensor, mask: Tensor) -> Tensor:
    """按 :data:`I2V_LAYOUT` 拼 36 通道：``cat([noise, mask, image], dim=1)``。

    三段必须同 ``ndim``、同 ``(b, t, h, w)``；通道宽严格按布局表核，**不按入参顺序猜**
    （顺序写反不报错、只是出片不受首帧约束 —— 见模块 docstring「36 通道」一节）。
    """
    if noise.ndim != 5:
        raise ShapeError(f"noise 要 5 维 (b,c,t,h,w) ✗，实际 {tuple(noise.shape)}")
    wanted = {"noise": noise, "mask": mask, "image": image}
    pieces: list[Tensor] = []
    for name, width in I2V_LAYOUT:
        piece = wanted[name]
        if piece.ndim != 5:
            raise ShapeError(f"I2V 的 {name} 要 5 维 ✗，实际 {tuple(piece.shape)}")
        if int(piece.shape[1]) != int(width):
            raise ShapeError(
                f"I2V 的 {name} 通道 {int(piece.shape[1])} ✗ ≠ 布局表要求的 {width}")
        if tuple(piece.shape[:1]) + tuple(piece.shape[2:]) != \
                tuple(noise.shape[:1]) + tuple(noise.shape[2:]):
            raise ShapeError(
                f"I2V 的 {name} 空间/时间维 {tuple(piece.shape[2:])} ✗ 与 noise "
                f"{tuple(noise.shape[2:])} 不一致")
        pieces.append(piece.to(dtype=noise.dtype, device=noise.device))
    return torch.cat(pieces, dim=1)


# --------------------------------------------------------------------------- 基础算子


def sinusoidal_embedding_1d(dim: int, position: Tensor) -> Tensor:
    """``cat([cos, sin])`` 的正弦时刻嵌入（**裸位置、不缩放**）。

    分母是 ``dim//2``，不是 ``dim``；内部用 ``float64`` 算完再 cast 回入参精度
    （参考实现就是 ``position.type(torch.float64)``，见 ``sinusoidal_embedding_1d``）。
    """
    if dim % 2:
        raise ShapeError(f"正弦嵌入维度 {dim} 是奇数 ✗")
    half = dim // 2
    pos = position.to(torch.float64).reshape(-1)
    exponent = -torch.arange(half, dtype=torch.float64, device=pos.device) / half
    sinusoid = torch.outer(pos, torch.pow(float(10000.0), exponent))
    out = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return out.to(position.dtype)


def rope_freqs(shape: WanShape, t_len: int, h_len: int, w_len: int, *,
               device: torch.device | None = None) -> Tensor:
    """``[1, L, 1, head_dim/2, 2, 2]`` 的三轴 RoPE 表，``L = t_len·h_len·w_len``。

    与参考实现 ``WanModel.rope_encode`` 逐句对应：先造 ``img_ids``（``[1, L, 3]``，三列 =
    ``t/h/w`` 位置号，**各自从 0 起、不居中**），过 ``EmbedND``（内部 ``cat(dim=-3)`` 后
    ``unsqueeze(1)``），再 ``movedim(1, 2)`` ⇒ 第 2 维是**头维广播位**（长度 1），
    这样 ``apply_rope`` 里 ``(1,L,1,d/2,2,2)`` 能与 ``(b,L,heads,d/2,2)`` 对上。

    频率表本身复用 :func:`app.services.engine.dit_generic.rope_freqs`（与 Flux 同一条实现：
    fp64 算频率、``linspace(0, (d-2)/d, d/2)``、2×2 块），三轴宽度取 ``shape.axes_dim``
    （``128 ⇒ 44/42/42``）。``theta`` 由形状给（Wan 是 10000）。
    """
    if min(t_len, h_len, w_len) <= 0:
        raise ShapeError(f"网格 ({t_len},{h_len},{w_len}) 有非正轴 ✗")
    axes = [torch.arange(length, dtype=torch.float64, device=device)
            for length in (t_len, h_len, w_len)]
    ids = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, -1, 3)
    freqs = _rope_freqs_nd(ids, list(shape.axes_dim), shape.theta)     # [1, L, d/2, 2, 2]
    return freqs.unsqueeze(1).movedim(1, 2)                            # [1, L, 1, d/2, 2, 2]


class _Attention(nn.Module):
    """Wan 的注意力：``qk_norm=True`` 时 q/k 各过 RMSNorm，``rope`` 只作用于自注意力。"""

    def __init__(self, shape: WanShape, *, rope: bool) -> None:
        super().__init__()
        dim = shape.dim
        self.dim = dim
        self.heads = shape.heads
        self.head_dim = shape.head_dim
        self.rope = rope
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        # RMSNorm 的参数名就是 ``weight`` ⇒ 键 ``...norm_q.weight`` ✓
        self.norm_q = nn.RMSNorm(dim, eps=shape.eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim, eps=shape.eps, elementwise_affine=True)

    def _split(self, x: Tensor) -> Tensor:
        b, s = x.shape[0], x.shape[1]
        return x.view(b, s, self.heads, self.head_dim)

    def _merge(self, x: Tensor) -> Tensor:
        b, s = x.shape[0], x.shape[1]
        return x.reshape(b, s, self.heads * self.head_dim)

    def _attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        b, length, heads, head_dim = q.shape
        q = q.transpose(1, 2)                       # [b, heads, s, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2).reshape(b, length, heads * head_dim)

    def forward(self, x: Tensor, freqs: Tensor | None = None,
                context: Tensor | None = None) -> Tensor:
        if self.rope:                               # 自注意力
            if freqs is None:
                raise ShapeError("自注意力要 freqs ✗")
            q = apply_rope(self._split(self.norm_q(self.q(x))), freqs)
            k = apply_rope(self._split(self.norm_k(self.k(x))), freqs)
            v = self._split(self.v(x))
        else:                                       # 交叉注意力：k/v 吃 context，无 RoPE
            if context is None:
                raise ShapeError("交叉注意力要 context ✗")
            q = self._split(self.norm_q(self.q(x)))
            k = self._split(self.norm_k(self.k(context)))
            v = self._split(self.v(context))
        return self.o(self._attend(q, k, v))


class WanBlock(nn.Module):
    """一个 Wan 双流块：自注意力 → 交叉注意力 → FFN，三处都是「门控残差」。"""

    def __init__(self, shape: WanShape) -> None:
        super().__init__()
        dim = shape.dim
        # norm1/norm2 **无参数**（elementwise_affine=False）⇒ 盘上 47 键里没有它们的影子 ✓
        self.norm1 = nn.LayerNorm(dim, eps=shape.eps, elementwise_affine=False)
        self.self_attn = _Attention(shape, rope=True)
        # cross_attn_norm=True（盘上有 norm3.weight/bias）⇒ 交叉注意力前还有一层带权 LayerNorm
        self.norm3 = nn.LayerNorm(dim, eps=shape.eps, elementwise_affine=True)
        self.cross_attn = _Attention(shape, rope=False)
        self.norm2 = nn.LayerNorm(dim, eps=shape.eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, shape.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(shape.ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))

    def forward(self, x: Tensor, e: Tensor, freqs: Tensor, context: Tensor) -> Tensor:
        # e: [B, 1, 6, dim] ⇒ modulation [1, 1, 6, dim] 加完沿 dim=2 unbind 出 6 路 [B, 1, dim]
        # （**不是** chunk(6, dim=1) ✗ —— 那是 Flux 的形态；这里 chunk 出来还是 [B,1,6,dim]，
        #  形状一直到 addcmul 才炸，报错点看着像 e 没算对）
        # modulation 跟 x 的精度走（参考实现 ``cast_to(self.modulation, dtype=x.dtype)``）：
        # 它是 ``torch.zeros`` 起手的 Parameter，装载前精度无关，装载后与权重同精度 ⇒ 通常是 no-op。
        mod = self.modulation.to(dtype=x.dtype, device=x.device)
        e6 = (mod.unsqueeze(0) + e).unbind(2)

        y = self.self_attn(
            torch.addcmul(e6[0], self.norm1(x), 1 + e6[1]), freqs=freqs)
        x = torch.addcmul(x, y, e6[2])

        x = x + self.cross_attn(self.norm3(x), context=context)

        y = self.ffn(torch.addcmul(e6[3], self.norm2(x), 1 + e6[4]))
        return torch.addcmul(x, y, e6[5])


class WanHead(nn.Module):
    """输出头：``norm → (1+scale)··+shift → Linear(dim, out_dim·patch体积)``。"""

    def __init__(self, shape: WanShape) -> None:
        super().__init__()
        self.out_dim = shape.out_dim
        self.patch = shape.patch
        self.norm = nn.LayerNorm(shape.dim, eps=shape.eps, elementwise_affine=False)
        self.head = nn.Linear(shape.dim, shape.patch_volume * shape.out_dim)
        self.modulation = nn.Parameter(torch.zeros(1, 2, shape.dim))

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        dim = self.modulation.shape[-1]
        # 参考实现按 ``e.ndim < 3`` 分两支，两支语义完全相同（都是每批一行 ``dim`` 宽的向量），
        # 这里统一压成 [B, dim]：modulation [1,2,dim] + [B,1,dim] ⇒ [B,2,dim] ⇒ chunk 2×[B,1,dim]。
        if e.shape[-1] != dim or e.numel() != x.shape[0] * dim:
            raise ShapeError(
                f"head 的 e 要 [B, dim] 或 [B, 1, dim]（dim={dim}）✗，实际 {tuple(e.shape)}")
        vec = e.reshape(x.shape[0], dim).to(x.dtype)
        shift, scale = (self.modulation.to(dtype=x.dtype) + vec.unsqueeze(1)).chunk(2, dim=1)
        return self.head(torch.addcmul(shift, self.norm(x), 1 + scale))


def unpatchify(x: Tensor, grid: Sequence[int], patch: Sequence[int],
               out_dim: int) -> Tensor:
    """``[B, L, out_dim·p·q·r]`` ⇒ ``[B, out_dim, T', H', W']·patch``（与参考实现同一个 einsum）。"""
    t_len, h_len, w_len = (int(g) for g in grid)
    p, q, r = (int(i) for i in patch)
    b = x.shape[0]
    u = x.view(b, t_len, h_len, w_len, p, q, r, out_dim)
    u = torch.einsum("bfhwpqrc->bcfphqwr", u)
    return u.reshape(b, out_dim, t_len * p, h_len * q, w_len * r)


class WanDiT(nn.Module):
    """Wan2.2 I2V 主干（单专家）。前向吃**拼好的** ``(b, in_dim, t, h, w)``。"""

    def __init__(self, shape: WanShape, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.shape = shape
        dim = shape.dim
        p, q, r = shape.patch
        self.patch_embedding = nn.Conv3d(shape.in_dim, dim, kernel_size=(p, q, r),
                                         stride=(p, q, r))
        self.text_embedding = nn.Sequential(
            nn.Linear(shape.text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(shape.freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        self.blocks = nn.ModuleList([WanBlock(shape) for _ in range(shape.layers)])
        self.head = WanHead(shape)
        if dtype is not None:
            self.to(dtype)

    # -- 前向 ---------------------------------------------------------------
    def forward(self, latents: Tensor, sigma: Tensor | float | None = None,
                context: Tensor | None = None, **_: Any) -> Tensor:
        """``latents``: ``(b, in_dim, t, h, w)``；``sigma``: 标量或 ``(b,)``；``context``: ``(b, L, text_dim)``。"""
        shape = self.shape
        if latents.ndim != 5:
            raise ShapeError(
                f"Wan 的 latent 是 5 维 (b,c,t,h,w) ✗，实际 {tuple(latents.shape)}"
                f"（4 维 = 出图族的口径，装错了族）")
        b, c, t, h, w = latents.shape
        if c != shape.in_dim:
            raise ShapeError(
                f"输入通道 {c} ✗ ≠ in_dim {shape.in_dim}"
                f"（36 = 噪声 16 + 掩码 4 + 条件图 16，拼接走 i2v_concat）")
        if context is None:
            raise ShapeError("Wan 前向必须有 context（文本塔 4096 维）✗")
        if context.ndim != 3 or context.shape[-1] != shape.text_dim:
            #: 不拦的话报错点是 ``text_embedding.0`` 的 matmul（"mat1 and mat2 shapes
            #: cannot be multiplied"），读起来像装配坏了 ✗
            raise ShapeError(
                f"context 要 (b, L, {shape.text_dim}) ✗，实际 {tuple(context.shape)}"
                f"（4096 = umt5-XXL 的隐宽，别拿 Qwen 的 2560/3584 顶）")
        if sigma is None:
            raise ShapeError("Wan 前向必须有 sigma ✗")
        p, q, r = shape.patch
        for name, size, step in (("t", t, p), ("h", h, q), ("w", w, r)):
            if size % step:
                raise ShapeError(
                    f"{name}={size} 不是 patch 的整数倍 ✗（参考实现会补边再切回，"
                    f"本仓不补 —— 补边等于悄悄改构图）")

        pw = self.patch_embedding.weight
        #: ★ 计算精度**跟权重走**、而且**整条前向只这一种精度** ✗✗（2026-11-02 真跑撞的 ✓）：
        #: 旧写法 ``conv(x.to(pw.dtype)).to(latents.dtype)`` 把 x 换回**latent 的**精度 ✓（fp32 ✓），
        #: 而盘上权重（fp8_scaled 反量化）是 **bf16** ✓ ⇒ 第一层 ``blocks.0.self_attn.q`` 当场
        #: ``RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`` ✓
        #: （读起来像"装配时精度混了" ✓✗，其实是**同一个前向里两种精度** ✓：conv 走 bf16、
        #: 紧接着的 Linear 吃 fp32 的 x ✓）。采样起点是 fp32 噪声 ✓、权重是 bf16 ✓ ⇒ 只能在这里对齐 ✓。
        #: 与 :class:`~app.services.engine.dit_qwen21.Qwen21DiT` 同一口径（那边也是 ``latents.to(wdtype)``
        #: 进去、按权重精度出来 ✓）。参考实现让 conv 常驻 fp32 是把 latent 也留在 fp32 才有意义 ✓，
        #: 本仓权重是 bf16 ⇒ 学它只会造出两种精度 ✓✗。
        dtype = pw.dtype
        x = self.patch_embedding(latents.to(dtype))
        grid = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)                       # [B, L, dim]

        ctx = self.text_embedding(context.to(self.text_embedding[0].weight.dtype))

        # σ ⇒ 时刻：×1000 在参考实现里由采样器做（见模块抬头第 1 条 / :data:`TIMESTEP_SCALE`），
        # 本仓骨架收裸 σ ⇒ 在这里补。漏了不报错，只是 e 整体错（出片糊/噪声）。
        t_vec = torch.as_tensor(sigma, dtype=torch.float32).reshape(-1) * TIMESTEP_SCALE
        if t_vec.numel() == 1 and b > 1:
            t_vec = t_vec.expand(b)
        if t_vec.numel() != b:
            raise ShapeError(f"sigma 给了 {t_vec.numel()} 个 ✗ ≠ 批 {b}")
        e = self.time_embedding(sinusoidal_embedding_1d(shape.freq_dim, t_vec).to(x.dtype))
        e = e.reshape(b, -1, e.shape[-1])                      # [B, 1, dim]
        e0 = self.time_projection(e).unflatten(2, (6, shape.dim))   # [B, 1, 6, dim]

        freqs = rope_freqs(shape, *grid, device=x.device).to(x.dtype)

        for block in self.blocks:
            x = block(x, e=e0, freqs=freqs, context=ctx)

        x = self.head(x, e)
        return unpatchify(x, grid, shape.patch, shape.out_dim)


# --------------------------------------------------------------------------- 形状/键表


def _shape_of_shapes(shapes: Mapping[str, Sequence[int]],
                     heads: int | None = None) -> WanShape:
    """从**键名 → 形状**的表里推导 :class:`WanShape`（不认识/缺键 ⇒ ``ShapeError``）。"""
    def need(key: str) -> tuple[int, ...]:
        try:
            return tuple(int(i) for i in shapes[key])
        except KeyError as exc:
            raise ShapeError(f"缺 {key} ✗（不是 Wan 族，或键名被前缀污染）") from exc

    pe = need("patch_embedding.weight")
    if len(pe) != 5:
        raise ShapeError(f"patch_embedding.weight 应 5 维 ✓，实际 {pe} ✗（不是视频族）")
    dim, in_dim = pe[0], pe[1]
    patch = (pe[2], pe[3], pe[4])
    out_dim = need("head.head.weight")[0] // (patch[0] * patch[1] * patch[2])
    text_dim = need("text_embedding.0.weight")[1]
    freq_dim = need("time_embedding.0.weight")[1]
    ffn_dim = need("blocks.0.ffn.0.weight")[0]
    layers = len({k.split(".")[1] for k in shapes
                  if k.startswith("blocks.") and k.split(".")[1].isdigit()})
    if not layers:
        raise ShapeError("一个 blocks.N 都没有 ✗")
    if "blocks.0.norm3.weight" not in shapes:
        raise ShapeError(
            "blocks.0.norm3.weight 不在 ✗ ⇒ cross_attn_norm=False 的变体本骨架不建"
            "（那样交叉注意力吃的是未归一化的 x，结构与这里不同）")
    if "blocks.0.self_attn.norm_q.weight" not in shapes:
        raise ShapeError("blocks.0.self_attn.norm_q.weight 不在 ✗ ⇒ qk_norm=False，本骨架不建")
    for banned in ("blocks.0.cross_attn.k_img.weight", "img_emb.0.weight"):
        if banned in shapes:
            raise ShapeError(
                f"{banned} 在 ✗ ⇒ 这是 Wan2.1 I2V 的写法（CLIP 视觉特征走单独一路交叉注意力），"
                f"本骨架按 Wan2.2 的通道拼接（in_dim=36）实现，建不出")

    resolved = heads
    if resolved is None:
        resolved = HEADS_BY_GEOMETRY.get((dim, ffn_dim, layers))
        if resolved is None:
            raise ShapeError(
                f"heads 在权重里读不出来（q/k/v 都是单条 {dim}×{dim}，norm_q 宽 = dim 不是 head_dim）"
                f"✗；(dim, ffn_dim, layers)=({dim}, {ffn_dim}, {layers}) 不在查表"
                f" {sorted(HEADS_BY_GEOMETRY)} 里 ⇒ 必须显式给 heads=")
    return WanShape(dim=dim, ffn_dim=ffn_dim, in_dim=in_dim, out_dim=out_dim,
                    text_dim=text_dim, freq_dim=freq_dim, heads=int(resolved),
                    layers=layers, patch=patch)


def infer_shape(info: Mapping[str, Any], *, heads: int | None = None) -> WanShape:
    """``info`` 可以是 ``{键: 形状}`` 或 reader（有 ``.keys()``/``.shape_of``）。"""
    shapes = getattr(info, "shapes", None)
    if shapes is None:
        if hasattr(info, "keys"):
            shapes = {k: (info.shape_of(k) if hasattr(info, "shape_of") else info[k])
                      for k in info.keys()}
        else:
            shapes = {k: v for k, v in dict(info).items()}
    return _shape_of_shapes(shapes, heads=heads)


def expected_keys(shape: WanShape) -> set[str]:
    """这一族**应该**有哪些真权重键（``1095 = 15 + 40×27``；fp8 的 ``*.scale_*`` 不算）。

    逐键列出来的好处：换成 Wan2.1（多 ``k_img``/``v_img``/``img_emb``）或换专家档位时，
    装载会当场报 ``missing``/``unexpected``，不会"能装、能跑、只是图不对" ✗。
    """
    keys: list[str] = [
        "head.head.weight", "head.head.bias",
        "head.modulation",
        "patch_embedding.weight", "patch_embedding.bias",
    ]
    for layer in ("text_embedding", "time_embedding"):
        for index in (0, 2):
            keys += [f"{layer}.{index}.weight", f"{layer}.{index}.bias"]
    keys += ["time_projection.1.weight", "time_projection.1.bias"]
    block = [
        # 自注意力（qk_norm=True ⇒ 两条 RMSNorm 有权重）
        "self_attn.q.weight", "self_attn.q.bias",
        "self_attn.k.weight", "self_attn.k.bias",
        "self_attn.v.weight", "self_attn.v.bias",
        "self_attn.o.weight", "self_attn.o.bias",
        "self_attn.norm_q.weight", "self_attn.norm_k.weight",
        # 交叉注意力（同形）
        "cross_attn.q.weight", "cross_attn.q.bias",
        "cross_attn.k.weight", "cross_attn.k.bias",
        "cross_attn.v.weight", "cross_attn.v.bias",
        "cross_attn.o.weight", "cross_attn.o.bias",
        "cross_attn.norm_q.weight", "cross_attn.norm_k.weight",
        # FFN
        "ffn.0.weight", "ffn.0.bias", "ffn.2.weight", "ffn.2.bias",
        # 交叉注意力前的带权 LayerNorm + 六路调制
        "norm3.weight", "norm3.bias",
        "modulation",
    ]
    for index in range(shape.layers):
        keys += [f"blocks.{index}.{name}" for name in block]
    return set(keys)


def header_facts(shapes: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """给 ``dit_families`` 之类的清单脚本用：不建模型，只报从权重读出的架构事实。"""
    shape = _shape_of_shapes(shapes)
    return {
        "dim": shape.dim,
        "ffnDim": shape.ffn_dim,
        "inDim": shape.in_dim,
        "outDim": shape.out_dim,
        "textDim": shape.text_dim,
        "freqDim": shape.freq_dim,
        "heads": shape.heads,
        "headDim": shape.head_dim,
        "layers": shape.layers,
        "patch": list(shape.patch),
        "axesDim": list(shape.axes_dim),
        "i2vLayout": [list(item) for item in shape.i2v_layout],
        "expectedKeys": len(expected_keys(shape)),
    }


def build(shape: WanShape, *, dtype: torch.dtype | None = None) -> WanDiT:
    return WanDiT(shape, dtype=dtype)


def build_from_headers(shapes: Mapping[str, Sequence[int]], *,
                       dtype: torch.dtype | None = None,
                       heads: int | None = None) -> WanDiT:
    return build(infer_shape(shapes, heads=heads), dtype=dtype)


def param_count(shape: WanShape) -> int:
    """参数个数（不建模型时也能算，用于文档里的体量核对）。"""
    dim, ffn = shape.dim, shape.ffn_dim
    per_block = (4 * dim * dim + 4 * dim                 # 自注意力 q/k/v/o（含 bias）
                 + 2 * dim                               # 自注意力 norm_q/norm_k
                 + 4 * dim * dim + 4 * dim               # 交叉注意力 q/k/v/o
                 + 2 * dim                               # 交叉注意力 norm_q/norm_k
                 + 2 * dim                               # norm3（weight+bias）
                 + dim * ffn + ffn + ffn * dim + dim     # FFN
                 + 6 * dim)                              # modulation
    top = (shape.in_dim * dim * math.prod(shape.patch) + dim   # patch_embedding
           + dim * shape.text_dim + dim + dim * dim + dim      # text_embedding
           + dim * shape.freq_dim + dim + dim * dim + dim      # time_embedding
           + dim * 6 * dim + 6 * dim                          # time_projection
           + dim * shape.patch_volume * shape.out_dim + shape.patch_volume * shape.out_dim
           + 2 * dim)                                        # head
    return per_block * shape.layers + top

