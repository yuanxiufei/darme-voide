"""Qwen-Image 2.1 单流 DiT 前向（``qwen_image21`` 族）。

为什么单独一份：2.1 有四处的错法都是**形状对、能跑、出图与提示词无关**：

① 全局共享调制（``modulation.1`` 一次出 ``4×dim``）＋**两张时间行**：``temb`` 是
   ``[采样 t, 0]`` 两行，文本 token 用 t=0 那行、图像 token 用采样 t 那行。全序列用同一组
   scale ⇒ 文本被时间调制拖着走。也因此**只支持 batch=1**（参考实现按"最后一行是真 t"分派）。
② 块内因果：文本段内部**带因果掩码**，图像段看全部。写成全可见 ⇒ 提示词末句开始影响前几句。
③ ``ZeroCenteredRMSNorm``：存的是 ``scale − 1``，用时要 ``w = weight + 1`` 且在 fp32 里算。
   当普通 RMSNorm 用 ⇒ 初值 0 的通道全被清零 ⇒ 文本投影全零，**不报错**。
④ SwiGLU 融合矩阵：``img_mlp.gate_up`` 出 ``2×hidden``，**gate 在前**，``silu(gate) * up``。

两个"权重里有、容易读错"的事实：``img_in`` 是 ``Linear(64 → dim)``，latent **本来就是
64 通道**（VAE 直出 64ch，``spacial_downscale_ratio=16``），**没有 conv patchify**；
RoPE 是三轴 ``(16,56,56)``/theta 1e4，和 = ``head_dim=128``，图像 h/w **中心化**。

权威口径：ComfyUI ``comfy/ldm/qwen_image21/model.py``（2026-10-31 逐行核过）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dit_generic import (ShapeError, apply_rope, rope_freqs, timestep_embedding)

__all__ = ["AXES_DIM", "ROPE_THETA", "T_FREQ", "Qwen21Shape", "Qwen21DiT",
           "expected_keys", "infer_shape"]

#: 三轴 RoPE 分轴宽度（t / h / w），**和必须 = head_dim**（16+56+56 = 128 ✓）
AXES_DIM: tuple[int, int, int] = (16, 56, 56)
ROPE_THETA = 10000.0
#: 时间正弦宽度（进 ``linear_1`` 之前）
T_FREQ = 256
#: 全族统一 eps，**不是** torch 默认的 1e-5
NORM_EPS = 1e-6

_TOP_KEYS: tuple[str, ...] = (
    "img_in.weight", "modulation.1.weight", "norm_out.linear.weight", "proj_out.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "txt_in.in_layer.weight", "txt_in.out_layer.weight", "txt_in.text_norm.weight",
)
_BLOCK_KEYS: tuple[str, ...] = (
    "attn.norm_k.weight", "attn.norm_q.weight", "attn.to_k.weight", "attn.to_out.0.weight",
    "attn.to_q.weight", "attn.to_v.weight", "img_mlp.gate_up.weight", "img_mlp.out.weight",
)


@dataclass(frozen=True)
class Qwen21Shape:
    """结构参数 —— **每一项都从权重形状推出来**。"""

    features: int
    layers: int
    heads: int
    head_dim: int
    hidden: int
    channels: int
    context_dim: int
    t_freq: int = T_FREQ
    t_inner: int = 0
    axes_dim: tuple[int, ...] = AXES_DIM
    theta: float = ROPE_THETA
    eps: float = NORM_EPS

    def __post_init__(self) -> None:
        if self.t_inner == 0:
            object.__setattr__(self, "t_inner", self.features)
        if self.heads * self.head_dim != self.features:
            raise ShapeError(
                f"heads×head_dim（{self.heads}×{self.head_dim}）≠ features（{self.features}）")
        if sum(self.axes_dim) != self.head_dim:
            raise ShapeError(
                f"三轴 RoPE 宽度和 {sum(self.axes_dim)} ≠ head_dim {self.head_dim}")


def _block_index(name: str) -> int | None:
    m = re.match(r"transformer_blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _dims(shapes: Mapping[str, tuple[int, ...]], name: str) -> tuple[int, ...]:
    got = shapes.get(name)
    if got is None:
        raise ShapeError(f"缺键 `{name}` —— 不是 2.1 的权重，或键名带了没剥掉的前缀")
    return tuple(int(x) for x in got)


def infer_shape(shapes: Mapping[str, tuple[int, ...]], *,
                heads: int | None = None,
                axes_dim: tuple[int, ...] | None = None) -> Qwen21Shape:
    """从权重形状表反推全部结构参数。

    ``heads`` 一般**不用给**：``attn.norm_q`` 是 per-head 宽（实测 ``[128]``）⇒
    ``heads = to_q 输出宽 ÷ 128`` 算得出来（Krea2 的 qknorm 是 ``[hidden]``，推不出来）。
    给了就核一遍，对不上立刻拒。

    ``axes_dim`` 同样**不用给**（默认 :data:`AXES_DIM` ✓）—— 它**权重里没有** ✗，是族事实；
    只有自检把结构缩小时才需要一起缩 ✓（缩了 ``head_dim`` 就必须缩它，否则
    ``sum(axes_dim) ≠ head_dim`` 当场拒 ✓✗ —— 这正是它该有的守门方式 ✓）。
    """
    #: ⚠️ ``nn.Linear(in, out).weight`` 是 ``[out, in]`` ⇒ 第一维是 **features**、第二维才是通道。
    #: 写反了会得到 ``features=64``、然后在 ``img_mlp.out`` 那条校验上报"hidden 对不上"（假警报）。
    features, channels = _dims(shapes, "img_in.weight")
    layers = {i for k in shapes if (i := _block_index(k)) is not None}
    if not layers:
        raise ShapeError("一个 `transformer_blocks.N.` 都没有 —— 不是 2.1 的权重")
    prefix = f"transformer_blocks.{min(layers)}."
    head_dim = _dims(shapes, prefix + "attn.norm_q.weight")[0]
    to_q = _dims(shapes, prefix + "attn.to_q.weight")
    if to_q[0] % head_dim:
        raise ShapeError(f"to_q 输出宽 {to_q[0]} 不是 norm_q 宽 {head_dim} 的整数倍")
    derived = to_q[0] // head_dim
    if heads is not None and heads != derived:
        raise ShapeError(f"给的 heads={heads} 与权重算出的 {derived} 不一致（别覆盖权重的事实）")

    gate_up = _dims(shapes, prefix + "img_mlp.gate_up.weight")
    if gate_up[0] % 2:
        raise ShapeError(f"gate_up 输出宽 {gate_up[0]} 是奇数 —— 切不出 gate/up")
    hidden = gate_up[0] // 2
    out_w = _dims(shapes, prefix + "img_mlp.out.weight")
    if (out_w[0], out_w[1]) != (features, hidden):
        raise ShapeError(f"img_mlp.out {out_w} 与 gate_up 推的 hidden={hidden} 对不上")

    ctx = _dims(shapes, "txt_in.in_layer.weight")
    text_norm = _dims(shapes, "txt_in.text_norm.weight")[0]
    if ctx[0] != features or text_norm != ctx[1]:
        raise ShapeError(
            f"txt_in 形状不一致：in_layer {ctx}、features {features}、text_norm {text_norm}")

    t_inner = _dims(shapes, "time_text_embed.timestep_embedder.linear_2.weight")[0]
    t_freq = _dims(shapes, "time_text_embed.timestep_embedder.linear_1.weight")[1]
    mod = _dims(shapes, "modulation.1.weight")
    if mod != (4 * features, t_inner):
        raise ShapeError(
            f"modulation.1 形状 {mod} ≠ (4×features={4 * features}, t_inner={t_inner})"
            " —— 2.1 的调制是**一组共享的 4 份**，对不上说明不是这一族")

    return Qwen21Shape(features=features, layers=len(layers), heads=derived,
                       head_dim=head_dim, hidden=hidden, channels=channels,
                       context_dim=ctx[1], t_freq=t_freq, t_inner=t_inner,
                       axes_dim=tuple(int(x) for x in (axes_dim or AXES_DIM)))


def expected_keys(shape: Qwen21Shape) -> set[str]:
    """这一族应有多少键、叫什么（不含量化伴生键）。"""
    keys = set(_TOP_KEYS)
    for i in range(shape.layers):
        keys.update(f"transformer_blocks.{i}.{s}" for s in _BLOCK_KEYS)
    return keys


# ------------------------------------------------------------------ 组件


class _ZeroCenteredRMSNorm(nn.Module):
    """``w = weight + 1``，fp32 里算。⚠️ 存的是 ``scale − 1``：直接用 ``weight`` ⇒
    初值 0 的通道全清零 ⇒ 文本投影全零、不报错。"""

    def __init__(self, dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.float() + 1.0
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * w).to(x.dtype)


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """普通 RMSNorm（``attn.norm_q/norm_k``），权重直接乘，fp32 里算。"""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


class _TextProjection(nn.Module):
    """``out_layer(gelu_tanh(in_layer(text_norm(x))))`` —— GELU 在 ``in_layer`` **之后**。"""

    def __init__(self, in_dim: int, features: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.text_norm = _ZeroCenteredRMSNorm(in_dim, eps)
        self.in_layer = nn.Linear(in_dim, features, bias=False)
        self.out_layer = nn.Linear(features, features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class _TimestepEmbed(nn.Module):
    """正弦（``time_factor=1000``）→ linear_1 → SiLU → linear_2。

    ⚠️ 类名/层级要和权重键对上：键是 ``time_text_embed.timestep_embedder.linear_1.weight``，
    所以外层属性必须叫 ``timestep_embedder``（少一层就加载不上，且报的是"缺键"）。
    """

    def __init__(self, freq_dim: int, inner: int, out_dim: int) -> None:
        super().__init__()
        self.freq_dim = int(freq_dim)
        self.linear_1 = nn.Linear(freq_dim, inner, bias=False)
        self.linear_2 = nn.Linear(inner, out_dim, bias=False)

    def forward(self, t: Tensor) -> Tensor:
        # ⚠️ 正弦在 fp32 里算完再降精度（bf16 算正弦会把位置信息糊掉）
        emb = timestep_embedding(t.float(), self.freq_dim).to(self.linear_1.weight.dtype)
        return self.linear_2(F.silu(self.linear_1(emb)))


class _SwiGLUFeedForward(nn.Module):
    """``out(silu(gate) * up)``，``gate_up`` 是一个 ``[2h, dim]`` 矩阵。"""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)  # ⚠️ gate 在前
        return self.out(F.silu(gate) * up)


class _Weight(nn.Module):
    """只有一根 ``weight`` 的壳 —— 存在的唯一理由：让**参数名带 ``.weight`` 后缀**。

    ⚠️ 直接 ``self.norm_q = nn.Parameter(...)`` 的参数名是 ``attn.norm_q``，而权重键是
    ``attn.norm_q.weight`` ⇒ 加载时这一格静默落空（q/k 没归一，形状全对、结果全偏）。
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))


class _Attention(nn.Module):
    """``to_q/k/v`` 分开 ＋ per-head RMSNorm ＋ 三轴 RoPE ＋ ``to_out.0``。"""

    def __init__(self, dim: int, heads: int, head_dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.heads, self.head_dim, self.eps = heads, head_dim, float(eps)
        inner = heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim, bias=False)])
        self.norm_q = _Weight(head_dim)
        self.norm_k = _Weight(head_dim)

    def forward(self, x: Tensor, freqs: Tensor, mask: Tensor | None) -> Tensor:
        b, n, _ = x.shape
        h, d = self.heads, self.head_dim
        q = self.to_q(x).view(b, n, h, d).transpose(1, 2)
        k = self.to_k(x).view(b, n, h, d).transpose(1, 2)
        v = self.to_v(x).view(b, n, h, d).transpose(1, 2)
        # ⚠️ 先 norm 再 rope（参考实现同序：norm 作用在 per-head 原始分量上）
        q = apply_rope(_rms_norm(q, self.norm_q.weight, self.eps), freqs)
        k = apply_rope(_rms_norm(k, self.norm_k.weight, self.eps), freqs)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.to_out[0](out.transpose(1, 2).reshape(b, n, h * d))


def _modulated(norm: nn.Module, x: Tensor, scale: Tensor, n_prefix: int) -> Tensor:
    """``norm(x) * (1 + scale)``：前缀行用 t=0 的 scale，目标行用采样 t 的 scale。

    ``scale`` 形状 ``(2, dim)``：第 0 行是采样 t、最后一行是 t=0。
    """
    y = norm(x)
    return torch.cat([y[:, :n_prefix] * (1 + scale[-1:]), y[:, n_prefix:] * (1 + scale[:-1])],
                     dim=1)


def _gated(x: Tensor, y: Tensor, gate: Tensor, n_prefix: int) -> Tensor:
    """``x + y * gate``（前缀/目标两行分开）。"""
    out = torch.empty_like(x)
    out[:, :n_prefix] = x[:, :n_prefix] + y[:, :n_prefix] * gate[-1:]
    out[:, n_prefix:] = x[:, n_prefix:] + y[:, n_prefix:] * gate[:-1]
    return out


class _Block(nn.Module):
    """``x += attn(norm1(x)*(1+s1))*g1`` → ``x += mlp(norm2(x)*(1+s2))*g2``。"""

    def __init__(self, shape: Qwen21Shape) -> None:
        super().__init__()
        d = shape.features
        # ⚠️ elementwise_affine=False：这两个 norm **没有参数**（权重里也确实没有）
        self.img_norm1 = nn.LayerNorm(d, eps=shape.eps, elementwise_affine=False)
        self.img_norm2 = nn.LayerNorm(d, eps=shape.eps, elementwise_affine=False)
        self.attn = _Attention(d, shape.heads, shape.head_dim, shape.eps)
        self.img_mlp = _SwiGLUFeedForward(d, shape.hidden)

    def forward(self, x: Tensor, mods: tuple[Tensor, ...], freqs: Tensor,
                mask: Tensor | None, n_prefix: int) -> Tensor:
        s1, g1, s2, g2 = mods
        x = _gated(x, self.attn(_modulated(self.img_norm1, x, s1, n_prefix), freqs, mask),
                   g1, n_prefix)
        return _gated(x, self.img_mlp(_modulated(self.img_norm2, x, s2, n_prefix)), g2, n_prefix)


class _LastLayer(nn.Module):
    """只作用在**目标行**：``norm(x) * (1 + linear(silu(temb)))``。"""

    def __init__(self, shape: Qwen21Shape) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(shape.features, eps=shape.eps, elementwise_affine=False)
        self.linear = nn.Linear(shape.t_inner, shape.features, bias=False)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        return self.norm(x) * (1 + self.linear(F.silu(temb)).unsqueeze(1))


class Qwen21DiT(nn.Module):
    """2.1 主干。``forward(latents (1,64,h,w), t (1,), context (1,n,ctx)) -> (1,64,h,w)``。"""

    def __init__(self, shape: Qwen21Shape) -> None:
        super().__init__()
        self.shape = shape
        d = shape.features
        self.img_in = nn.Linear(shape.channels, d, bias=False)
        self.txt_in = _TextProjection(shape.context_dim, d, shape.eps)
        self.time_text_embed = nn.ModuleDict(
            {"timestep_embedder": _TimestepEmbed(shape.t_freq, shape.t_inner, shape.t_inner)})
        #: ``Sequential(SiLU, Linear)`` ⇒ 权重键是 ``modulation.1.weight`` ✓
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(shape.t_inner, 4 * d, bias=False))
        self.transformer_blocks = nn.ModuleList(_Block(shape) for _ in range(shape.layers))
        self.norm_out = _LastLayer(shape)
        self.proj_out = nn.Linear(d, shape.channels, bias=False)

    # -------------------------------------------------------------- 位置
    @staticmethod
    def _ids(n_txt: int, h: int, w: int, *, device: torch.device) -> Tensor:
        """``(n_txt + h·w, 3)`` 位置号：文本三轴同值从 **0** 起，图像 ``(n_txt, hh, ww)`` 中心化。"""
        txt = torch.arange(0, n_txt, dtype=torch.float32, device=device)[:, None].expand(n_txt, 3)
        #: ⚠️ 中心化：``arange(h) - (h - h//2)`` ⇒ 偶数高是 ``-h/2 … h/2-1``。
        #: 参考实现里还有个 ``0.5*(h%2 - H%2)`` 项，只在**多参考图**时才非零；t2i 无参考图 ⇒ 0。
        hh = torch.arange(h, dtype=torch.float32, device=device) - (h - h // 2)
        ww = torch.arange(w, dtype=torch.float32, device=device) - (w - w // 2)
        img = torch.stack([torch.full((h, w), float(n_txt), device=device),
                           hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1)
        return torch.cat([txt, img.reshape(h * w, 3)], dim=0)

    @staticmethod
    def _mask(n_txt: int, total: int, *, device: torch.device) -> Tensor:
        """文本段内部**因果**、不看图像；图像段看全部（t2i 只有两段）。"""
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        mask[:n_txt, :n_txt] = torch.ones(n_txt, n_txt, dtype=torch.bool,
                                          device=device).tril()
        mask[n_txt:, :] = True
        return mask

    def forward(self, latents: Tensor, timesteps: Tensor, context: Tensor) -> Tensor:
        shape = self.shape
        wdtype = self.img_in.weight.dtype
        if latents.dim() != 4:
            raise ShapeError(f"latents 要 (B,C,H,W)，实际 {tuple(latents.shape)}")
        b, c, h, w = latents.shape
        if b != 1:
            #: ⚠️ 共享调制是 ``cat([t, 0])`` 两行、按"最后一行是真 t"分派 ⇒ 多 batch 会被分错。
            raise ShapeError(f"2.1 的共享调制只支持 batch=1，实际 {b}")
        if c != shape.channels:
            raise ShapeError(f"latent 通道 {c} ≠ 权重通道 {shape.channels}（2.1 的 latent 就是 64ch）")
        device = latents.device
        ctx = context.to(wdtype)
        n_txt = ctx.shape[1]
        if n_txt == 0:
            raise ShapeError("context 一个 token 都没有")

        # ⚠️ 参考实现先把 t 乘 1000 再除回去，纯粹为了让 t 落在采样 dtype 的格点上
        t = ((timesteps.reshape(-1)[:1].to(wdtype) * 1000).to(wdtype) / 1000).to(wdtype)
        temb = self.time_text_embed["timestep_embedder"](torch.cat([t, t.new_zeros(1)]))  # (2, dim)
        s1, g1, s2, g2 = self.modulation(temb).chunk(4, dim=-1)
        mods = (s1, g1.tanh(), s2, g2.tanh())  # ⚠️ gate 要 tanh

        x = torch.cat([self.txt_in(ctx),
                       self.img_in(latents.to(wdtype).flatten(2).transpose(1, 2))], dim=1)
        freqs = rope_freqs(self._ids(n_txt, h, w, device=device).unsqueeze(0),
                           list(shape.axes_dim), shape.theta).unsqueeze(1)
        mask = self._mask(n_txt, x.shape[1], device=device)
        for block in self.transformer_blocks:
            x = block(x, mods, freqs, mask, n_txt)

        # ⚠️ 输出只取目标行；``temb[:-1]`` = 采样 t 那行（``temb[-1]`` 是给文本用的 0）
        out = self.proj_out(self.norm_out(x[:, n_txt:], temb[:-1]))
        return out.transpose(1, 2).reshape(b, shape.channels, h, w)


def build(shape: Qwen21Shape) -> Qwen21DiT:
    """按形状建一个**未加载权重**的主干。"""
    return Qwen21DiT(shape)
