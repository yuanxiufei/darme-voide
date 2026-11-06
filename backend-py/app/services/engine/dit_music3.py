"""MiniMax Music3 的 DiT（**音频/歌曲主干**）—— 第十族骨架。

族内三段（本文件只负责中间那段 ✓）::

    文本+歌词 --(TE+AR)--> context (b, F, 8x4096) --(本文件)--> 速度 (b, 128, L) --(DAV)--> 44.1k 立体声

一次前向：``latents (b,128,L)`` ／ ``sigma (b,)`` ／ ``context (b,F,8*4096)`` ／
``conditioning_scale (b,1,1)`` ⇒ ``(b,128,L)``，**返回前已取负** ✓✗。

三处「形状全对、只有歌不对」的口径（都写死在本文件，不让调用方猜 ✓）：

1. **时间步要翻**：参考侧 ``process_timestep`` 是 ``1 − σ`` ✓✗；喂原样 σ 不报错，
   只是把去噪方向整体反过来（听感＝人声噪声）✓✗。
2. **输出要取负**：``return -out`` ✓✗。与第 1 条是**一对**，只改一边立刻崩且不报错 ✓✗。
3. **latent 帧数 = 条件帧数 × 3.4453125**（``44100/24000 × 960/512`` ✓）：条件帧 25 Hz、
   latent 86.13 Hz ✓。当两者同帧率（多数族是 1:1 ✓✗）⇒ 长度对不上后被 pad/crop 静默吃掉 ✓✗。

与其它族不同的结构（通用骨架装不下 ⇒ 一族一份 ✓）：LLM 风格 pre-norm 双残差、**无 ada 调制**
（时间步作**第 0 个 token** 拼进序列 ✓✗）；LN 带 gamma+beta 且 **eps=1e-5**；
GLU 是 ``value * silu(gate)``（**value 在前** ✓✗）；RoPE 只转 head_dim 的**前半 32 维**且是
**split-half** 配对 ✓✗。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "CONDITION_HOP_FRAMES",
    "COND_HIDDEN",
    "COND_LAYERS",
    "DEFAULT_DIM_HEADS",
    "LATENT_CHANNELS",
    "MAX_CONDITION_FRAMES",
    "Music3DiT",
    "Music3DiffusionTransformer",
    "Music3Shape",
    "NORM_EPS",
    "ROPE_THETA",
    "ShapeError",
    "build",
    "infer_shape",
    "latent_length",
    "param_count",
]


class ShapeError(ValueError):
    """权重里的形状自相矛盾（推不出可装的结构 ✓）—— 调用方转成「缺能力」的人话 ✓。"""


LATENT_CHANNELS = 128
COND_LAYERS = 8
COND_HIDDEN = 4096
#: 条件帧率：24000 采样 ÷ 960 跳 = **25 Hz** ✓
CONDITION_SAMPLE_RATE = 24000
CONDITION_HOP = 960
#: 音频 latent：44100 ÷ 512 = **86.13 Hz** ✓
LATENT_SAMPLE_RATE = 44100
LATENT_HOP = 512
#: 单次前向最多吃 200 条件帧（8 秒 ✓），超过滑窗；步长 100 帧（4 秒 ✓）
MAX_CONDITION_FRAMES = 200
CONDITION_HOP_FRAMES = 100
#: 头宽（**权重里没有** ✓✗ ⇒ 只能 ``dim // 64``）
DEFAULT_DIM_HEADS = 64
#: ⚠️ LayerNorm eps 是 **1e-5**（Wan/Z-Image 那套是 1e-6 ✓✗）
NORM_EPS = 1e-5
ROPE_THETA = 10_000.0
#: 参考实现 ``return -out`` ✓
OUTPUT_SIGN = -1.0


def latent_length(audio_frames: int) -> int:
    """条件帧（25 Hz）⇒ latent 帧（86.13 Hz）✓ —— 算式**逐字照抄**参考实现 ✓✗。

    先约分成 ``× 3.4453125`` 会在个别帧数上差 1（截断位置变了 ✓），
    而差 1 帧会被后面的 pad/crop 静默吃掉 ✓✗。
    """
    return max(1, int(audio_frames * LATENT_SAMPLE_RATE / CONDITION_SAMPLE_RATE
                      * CONDITION_HOP / LATENT_HOP))


@dataclass(frozen=True)
class Music3Shape:
    """从**权重形状**推出来的结构 ✓。

    ``heads`` / ``rope_theta`` 权重里没有 ✓✗ ⇒ 走 :data:`DEFAULT_DIM_HEADS` /
    :data:`ROPE_THETA`（可用 ``explicit["heads"]`` 覆盖），报告里要写明「这两格是常量」✓。
    """

    dim: int = 2048
    ffn_dim: int = 8192
    latent_channels: int = LATENT_CHANNELS
    cond_layers: int = COND_LAYERS
    cond_hidden: int = COND_HIDDEN
    cond_proj: int = 2048
    layers: int = 36
    heads: int = 32
    rotary_dim: int = 32
    freq_dim: int = 128
    rope_theta: float = ROPE_THETA

    @property
    def in_dim(self) -> int:
        """入口宽度 ＝ ``x(128) + zeros(128) + condition(2048)`` ＝ 2304 ✓。"""
        return self.latent_channels * 2 + self.cond_proj

    @property
    def context_dim(self) -> int:
        """**上游编码出来的**条件宽度 ＝ ``cond_layers × cond_hidden`` ＝ 32768 ✓。

        ⚠️ 与 :attr:`cond_proj`（2048 ✓）是两件事 ✗✗：32768 是 TE+AR 逐帧拼出来的原始宽度 ✓
        （基座隐状态 4096 ＋ RVQ 7 层各 4096 ✓）；2048 是主干**内部**对齐后的宽度 ✓
        （``latent_conditioners`` ＋ 层 softmax 加权求和 ⇒ 2048 ✓）。
        把 2048 当"text_dim"报给上游 ⇒ 编码侧产出 2048 宽 ✓、``Music3DiT.condition_for`` 当场
        判形状不符 ✗（还是好的）；更糟的是按 2048 造条件 ✓ 而层权重是 8 个 ✓ ⇒ 静默错位 ✓✗。
        """
        return self.cond_layers * self.cond_hidden

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads


def _shape_of(shapes: Mapping[str, Sequence[int]], suffix: str) -> tuple[str, tuple[int, ...]]:
    """按**后缀**找唯一键 ⇒ ``(键名, 形状)``；找不到 / 命中多个都当场拒 ✓（不猜 ✗）。"""
    hits = [k for k in shapes if k == suffix or k.endswith("." + suffix)]
    if not hits:
        raise ShapeError(f"权重里找不到 ``{suffix}`` ✗ —— 这不是 MiniMax Music3 的 DiT ✓")
    if len(hits) > 1:
        raise ShapeError(f"``{suffix}`` 命中 {len(hits)} 个键（{hits[:3]}…）✗ —— 分不清用哪个 ✓")
    return hits[0], tuple(int(v) for v in shapes[hits[0]])


def _layer_indexes(shapes: Iterable[str], prefix: str) -> list[int]:
    """``...layers.<n>.`` 的所有 ``n``（升序去重 ✓）。"""
    found: set[int] = set()
    for key in shapes:
        at = key.find(prefix)
        if at < 0:
            continue
        head = key[at + len(prefix):].split(".", 1)[0]
        if head.isdigit():
            found.add(int(head))
    return sorted(found)


def infer_shape(shapes: Mapping[str, Sequence[int]], *, heads: Any = None,
                **_: Any) -> Music3Shape:
    """从形状字典推结构 ✓（**自相矛盾**就 :class:`ShapeError` ✓，不硬装 ✗）。"""
    _, post = _shape_of(shapes, "diffusion_transformer.postprocess_conv.weight")
    if len(post) != 3:
        raise ShapeError(f"``postprocess_conv.weight`` 应当 3 维，实际 {post} ✗")
    latent_channels = int(post[0])

    _, pre = _shape_of(shapes, "diffusion_transformer.preprocess_conv.weight")
    if len(pre) != 3:
        raise ShapeError(f"``preprocess_conv.weight`` 应当 3 维，实际 {pre} ✗")
    in_dim = int(pre[1])

    _, project_in = _shape_of(shapes, "diffusion_transformer.transformer.project_in.weight")
    dim = int(project_in[0])
    _, project_out = _shape_of(shapes, "diffusion_transformer.transformer.project_out.weight")
    _, ts_first = _shape_of(shapes, "diffusion_transformer.to_timestep_embed.0.weight")
    _, ts_second = _shape_of(shapes, "diffusion_transformer.to_timestep_embed.2.weight")
    _, fourier = _shape_of(shapes, "diffusion_transformer.timestep_features.weight")
    freq_dim = int(fourier[0])
    _, rotary = _shape_of(shapes, "diffusion_transformer.transformer.rotary_pos_emb.inv_freq")

    _, cond_w = _shape_of(shapes, "latent_conditioners.0.weight")
    if len(cond_w) != 3:
        raise ShapeError(f"``latent_conditioners.0.weight`` 应当 3 维，实际 {cond_w} ✗")
    cond_proj, cond_hidden = int(cond_w[0]), int(cond_w[1])
    _, logits = _shape_of(shapes, "cond_layer_logits")
    cond_layers = int(logits[0])

    indexes = _layer_indexes(shapes, "diffusion_transformer.transformer.layers.")
    if not indexes:
        raise ShapeError("权重里没有 ``...transformer.layers.<n>.`` ✗ —— 分不清层数 ✓")
    if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
        raise ShapeError(f"层号不连续（{indexes[:3]}…{indexes[-1]}）✗ —— 权重可能被裁剪过 ✓")

    _, glu = _shape_of(shapes, "transformer.layers.0.ff.ff.0.proj.weight")
    _, ff_out = _shape_of(shapes, "transformer.layers.0.ff.ff.2.weight")
    ffn_dim = int(ff_out[1])

    want_in = latent_channels * 2 + cond_proj
    problems: list[str] = []
    if in_dim != want_in:
        problems.append(f"入口宽度 {in_dim} ≠ {latent_channels}×2＋{cond_proj}（＝{want_in}）")
    if int(project_in[1]) != in_dim:
        problems.append(f"``project_in`` 入宽 {project_in[1]} ≠ 入口宽度 {in_dim}")
    if int(project_out[1]) != dim:
        problems.append(f"``project_out`` 入宽 {project_out[1]} ≠ {dim}")
    if int(project_out[0]) != latent_channels:
        problems.append(f"``project_out`` 出宽 {project_out[0]} ≠ latent 通道 {latent_channels}")
    if (int(ts_first[0]), int(ts_first[1])) != (dim, freq_dim * 2):
        problems.append(f"``to_timestep_embed.0`` {tuple(ts_first)} ≠ ({dim}, {freq_dim * 2})")
    if (int(ts_second[0]), int(ts_second[1])) != (dim, dim):
        problems.append(f"``to_timestep_embed.2`` {tuple(ts_second)} ≠ ({dim}, {dim})")
    if (int(glu[0]), int(glu[1])) != (ffn_dim * 2, dim):
        problems.append(f"GLU 投影 {tuple(glu)} ≠ ({ffn_dim * 2}, {dim})（GLU 是 2 倍宽 ✓）")
    if int(ff_out[0]) != dim:
        problems.append(f"``ff.ff.2`` 出宽 {ff_out[0]} ≠ {dim}")
    if problems:
        raise ShapeError("；".join(problems))

    if heads:
        heads_val = int(heads)
        if dim % heads_val:
            raise ShapeError(f"显式 heads={heads_val} 除不尽 dim={dim} ✗")
    else:
        if dim % DEFAULT_DIM_HEADS:
            raise ShapeError(f"dim={dim} 除不尽头宽 {DEFAULT_DIM_HEADS} ✗，"
                             f"而**头数权重里没有** ✓✗ ⇒ 请显式给 heads（别猜 ✓）")
        heads_val = dim // DEFAULT_DIM_HEADS

    rotary_dim = int(rotary[0]) * 2 if len(rotary) else 32
    head_dim = dim // heads_val
    if rotary_dim % 2 or rotary_dim > head_dim:
        raise ShapeError(f"``inv_freq`` {tuple(rotary)} ⇒ 旋转 {rotary_dim} 维 ✗，"
                         f"与头宽 {head_dim} 不匹配（必须是头宽的前半 ✓）")

    return Music3Shape(dim=dim, ffn_dim=ffn_dim, latent_channels=latent_channels,
                       cond_layers=cond_layers, cond_hidden=cond_hidden, cond_proj=cond_proj,
                       layers=len(indexes), heads=heads_val, rotary_dim=rotary_dim,
                       freq_dim=freq_dim)


def param_count(shape: Music3Shape) -> int:
    """参数量（报告用 ✓，不含 buffer ✓）。"""
    dim, ffn = shape.dim, shape.ffn_dim
    per_layer = (4 * dim * dim                      # qkv(3·dim²) + out(dim²)
                 + 4 * dim                          # 两组 LN 的 gamma+beta
                 + 2 * ffn * dim + 2 * ffn          # GLU
                 + dim * ffn + dim)                 # ff.2
    head = (shape.in_dim * dim + dim * shape.latent_channels
            + shape.in_dim ** 2 + shape.latent_channels ** 2
            + shape.freq_dim
            + 2 * shape.freq_dim * dim + dim + dim * dim + dim
            + shape.cond_hidden * shape.cond_proj * 3 + shape.cond_proj
            + shape.cond_layers + 1)
    return head + per_layer * shape.layers


# ── 归一化：gamma + **beta** ─────────────────────────────────────────────────

class Music3Norm(nn.Module):
    """``F.layer_norm(x, (d,), gamma, beta)``，eps 用 PyTorch 默认 **1e-5** ✓。

    ⚠️ Wan / Z-Image 那一套是 1e-6 ✓✗：写错**不报错**，只是数值逐层漂 ✓
    （36 层够让出片「听着像、但不是那首歌」✓✗）。

    ⚠️ ``beta`` 在参考实现里登记成 **buffer** ✓（不是 Parameter ✓）。这里写成 Parameter ✓
    —— 两者**装载与精度行为完全一致** ✓（名字相同 ✓、``state_dict`` 都含 ✓、
    ``.to(dtype)`` 同理 ✓、``materialize_meta`` 参数与 buffer 两条都换 ✓）。
    唯一的差别是「它会不会出现在 ``named_parameters`` 里 ✓」—— 本仓只做推理 ✓，
    不影响产物 ✓。**记在这里是为了别把它当成"抄错了"再改回去** ✓✗。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.gamma, self.beta)


class Music3Rotary(nn.Module):
    """``inv_freq`` 是**权重里的 buffer** ✓ ⇒ 装载后就是参考实现的实测值（不自算 ✓）。"""

    def __init__(self, rotary_dim: int, theta: float = ROPE_THETA) -> None:
        super().__init__()
        freq = 1.0 / (float(theta) ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        self.register_buffer("inv_freq", freq.to(torch.float32))
        self.rotary_dim = int(rotary_dim)

    def forward_from_seq_len(self, length: int, dtype: torch.dtype) -> torch.Tensor:
        """``(1, 1, L, rotary_dim/2, 2, 2)``，栈顺序 ``cos,-sin,sin,cos`` ✓。"""
        positions = torch.arange(length, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq.float()).to(dtype)
        cos, sin = freqs.cos(), freqs.sin()
        return torch.stack((cos, -sin, sin, cos), dim=-1).reshape(
            1, 1, length, freqs.shape[-1], 2, 2)


def _apply_rope(x: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    """**split-half** 配对旋转 ✓：``(i, i+d/2)`` 配，不是相邻偶奇配 ✓✗。

    交错配对照样算得出张量、也不报错 ✓✗，只是相位整体打乱（听感：结构在、段落乱 ✓）。
    """
    x_dtype = x.dtype
    x = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(rotation_matrix.dtype)
    x = rotation_matrix[..., 0] * x[..., 0] + rotation_matrix[..., 1] * x[..., 1]
    return x.movedim(-1, -2).flatten(-2).to(x_dtype)


class Music3Attention(nn.Module):
    """融合 qkv、**无 qk-norm**、无 bias ✓（权重里 qkv/out 都只有 ``.weight`` ✓）。"""

    def __init__(self, shape: Music3Shape) -> None:
        super().__init__()
        self.heads = shape.heads
        self.head_dim = shape.head_dim
        self.to_qkv = nn.Linear(shape.dim, shape.dim * 3, bias=False)
        self.to_out = nn.Linear(shape.dim, shape.dim, bias=False)

    def forward(self, x: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        rotary = rotation_matrix.shape[-3] * 2
        q = torch.cat((_apply_rope(q[..., :rotary], rotation_matrix), q[..., rotary:]), dim=-1)
        k = torch.cat((_apply_rope(k[..., :rotary], rotation_matrix), k[..., rotary:]), dim=-1)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.to_out(out.transpose(1, 2).reshape(batch, length, dim))


class Music3GLU(nn.Module):
    """``value * silu(gate)`` —— **value 在前** ✓✗（反了照样出片、只是糊成一团 ✓）。"""

    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * F.silu(gate)


class Music3FeedForward(nn.Module):
    """``Sequential(GLU, Identity, Linear)`` —— 中间那格 Identity 是**参考实现的形状** ✓：
    去掉它键名会从 ``ff.ff.2`` 变 ``ff.ff.1`` ⇒ 装载时整层对不上 ✓✗。"""

    def __init__(self, shape: Music3Shape) -> None:
        super().__init__()
        self.ff = nn.Sequential(Music3GLU(shape.dim, shape.ffn_dim), nn.Identity(),
                                nn.Linear(shape.ffn_dim, shape.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class Music3Block(nn.Module):
    """LLM 风格：pre-norm 两处、残差两条、**没有任何调制** ✓。"""

    def __init__(self, shape: Music3Shape) -> None:
        super().__init__()
        self.pre_norm = Music3Norm(shape.dim)
        self.self_attn = Music3Attention(shape)
        self.ff_norm = Music3Norm(shape.dim)
        self.ff = Music3FeedForward(shape)

    def forward(self, x: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.pre_norm(x), rotation_matrix)
        return x + self.ff(self.ff_norm(x))


class Music3FourierFeatures(nn.Module):
    """``2π·v @ Wᵀ`` 后 ``cat((cos, sin))`` ✓✗ —— 顺序反了（sin 在前）不报错 ✓。"""

    def __init__(self, freq_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(freq_dim, 1) * 0.02)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(value.dtype)
        features = 2.0 * math.pi * (value @ weight.T)
        return torch.cat((features.cos(), features.sin()), dim=-1)


class Music3Transformer(nn.Module):
    """``project_in → 拼时间步 token → RoPE → N × block → project_out`` ✓。"""

    def __init__(self, shape: Music3Shape) -> None:
        super().__init__()
        self.project_in = nn.Linear(shape.in_dim, shape.dim, bias=False)
        self.project_out = nn.Linear(shape.dim, shape.latent_channels, bias=False)
        self.rotary_pos_emb = Music3Rotary(shape.rotary_dim, shape.rope_theta)
        self.layers = nn.ModuleList([Music3Block(shape) for _ in range(shape.layers)])

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        #: ⭐ 时间步是**第 0 个 token** ✓✗ —— 位置反了、或加成偏置都不报错，只是时间信息错位 ✓
        x = torch.cat((timestep_embedding.unsqueeze(1), x), dim=1)
        rotation_matrix = self.rotary_pos_emb.forward_from_seq_len(x.shape[1], x.dtype)
        for layer in self.layers:
            x = layer(x, rotation_matrix)
        return self.project_out(x[:, 1:])


class Music3DiffusionTransformer(nn.Module):
    """``[x | 0 | condition] → 1×1 conv 残差 → 主干 → 1×1 conv 残差`` ✓。"""

    def __init__(self, shape: Music3Shape) -> None:
        super().__init__()
        self.shape_info = shape
        self.transformer = Music3Transformer(shape)
        self.timestep_features = Music3FourierFeatures(shape.freq_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(shape.freq_dim * 2, shape.dim),
            nn.SiLU(),
            nn.Linear(shape.dim, shape.dim),
        )
        self.preprocess_conv = nn.Conv1d(shape.in_dim, shape.in_dim, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(shape.latent_channels, shape.latent_channels, 1,
                                          bias=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        #: ⭐ 中间那一路是 **zeros** ✓✗（不是 condition、也不是 x）：
        #: 三路等宽、拼错一路形状照样对 ✓，只是主干看到的是「错的自变量」✓
        full = torch.cat((x, torch.zeros_like(x), condition), dim=1)
        full = self.preprocess_conv(full) + full
        features = self.timestep_features(timestep[:, None]).to(dtype=x.dtype)
        embedding = self.to_timestep_embed(features)
        out = self.transformer(full.transpose(1, 2), embedding).transpose(1, 2)
        return self.postprocess_conv(out) + out


class Music3DiT(nn.Module):
    """族入口：条件对齐 + 主干 + 长音频滑窗 ✓。

    ``forward(latents, sigma, context, conditioning_scale=None)`` ⇒ 返回**已取负**的速度 ✓✗。
    """

    def __init__(self, shape: Music3Shape, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.shape_info = shape
        self.latent_conditioners = nn.Sequential(
            nn.Conv1d(shape.cond_hidden, shape.cond_proj, kernel_size=3, padding=1))
        self.diffusion_transformer = Music3DiffusionTransformer(shape)
        self.cond_layer_logits = nn.Parameter(torch.zeros(shape.cond_layers))
        self.cond_layer_scale = nn.Parameter(torch.ones(1))
        if dtype is not None:
            self.to(dtype)

    def aligned_condition(self, hidden: torch.Tensor) -> torch.Tensor:
        """``(b, F, 8×4096)`` ⇒ ``(b, cond_proj, latent_length(F))`` ✓。

        ⚠️ 通道排布是**帧主序**（``frame → layer → 4096`` ✓）：``transpose(1,2).reshape``
        只在这个排布下才对 ✓✗；若上游按 ``layer → frame`` 拼，reshape 照样成功 ✓、
        只是把 8 层权重加到了错的帧上（听感：条件几乎不生效 ✓）。
        """
        frames = hidden.shape[1]
        layers, width = self.shape_info.cond_layers, self.shape_info.cond_hidden
        hidden = hidden.transpose(1, 2).reshape(hidden.shape[0], layers, width, frames)
        weights = torch.softmax(self.cond_layer_logits.to(hidden.dtype), dim=0)
        hidden = torch.einsum("blht,l->bht", hidden, weights)
        hidden = self.cond_layer_scale.to(hidden.dtype) * hidden
        condition = self.latent_conditioners(hidden)
        return F.interpolate(condition, size=latent_length(frames), mode="nearest")

    def condition_for(self, hidden: torch.Tensor, length: int,
                      conditioning_scale: torch.Tensor | None = None) -> torch.Tensor:
        """对齐 + 缩放 + **pad/crop 到 latent 长度** ✓（顺序与参考实现一致 ✓）。"""
        condition = self.aligned_condition(hidden)
        if conditioning_scale is not None:
            condition = condition * conditioning_scale[:, :1, :1].to(condition.dtype)
        if condition.shape[-1] < length:
            condition = F.pad(condition, (0, length - condition.shape[-1]))
        else:
            condition = condition[..., :length]
        return condition

    def velocity(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """**未取负**的主干速度 ✓（长音频在这里滑窗 ✓）。``t`` 是**已翻转**的 ``1 − σ`` ✓。"""
        length = x.shape[-1]
        window = latent_length(MAX_CONDITION_FRAMES)
        hop = latent_length(CONDITION_HOP_FRAMES)
        if length <= window:
            return self.diffusion_transformer(x, t, condition)
        #: 长音频滑窗 ✓：逐窗累加后除以命中次数 ✓（取负放在最后一次性做，
        #: 与参考实现"每窗取负再相加"逐位等价 —— 取负是线性的 ✓）
        total = torch.zeros_like(x)
        weight = torch.zeros_like(x)
        start = 0
        while start < length:
            end = min(start + window, length)
            total[..., start:end] += self.diffusion_transformer(
                x[..., start:end], t, condition[..., start:end])
            weight[..., start:end] += 1.0
            if end >= length:
                break
            start += hop
        return total / weight

    def forward(self, latents: torch.Tensor, sigma: Any, context: torch.Tensor,
                conditioning_scale: torch.Tensor | None = None, **_: Any) -> torch.Tensor:
        #: ⚠️ 计算精度**跟权重走**且**整条前向只留一种** ✓✗（同 `dit_wan22` 那条坑 ✓）
        dtype = self.diffusion_transformer.postprocess_conv.weight.dtype
        if latents.dim() != 3:
            raise ValueError(f"Music3 的 latent 必须是 (b, {self.shape_info.latent_channels}, L)，"
                             f"实际 {tuple(latents.shape)} ✗ —— 这是**音频**族，不是图像族 ✓✗")
        x = latents.to(dtype)
        length = x.shape[-1]
        condition = self.condition_for(context.to(dtype), length, conditioning_scale)

        #: ⭐ 时间步翻转 ✓✗：喂的是 ``t = 1 − σ``，且**在 float32 里算** ✓（参考侧 .float() ✓）
        t = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).reshape(-1)
        if t.numel() == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0])
        if t.numel() != x.shape[0]:
            raise ValueError(f"σ 的批 {tuple(t.shape)} 与 latent 批 {x.shape[0]} 不一致 ✗")
        t = 1.0 - t

        #: ⭐ 取负 ✓✗（参考实现 ``return -out``；与入口的时间步翻转是一对 ✓）
        return self.velocity(x, t, condition) * OUTPUT_SIGN


def build(shape: Music3Shape, *, patch: Any = None, dtype: torch.dtype | None = None) -> Music3DiT:
    """按形状建网 ✓（``patch`` 只占位：音乐族没有 patchify ✓）。"""
    del patch
    return Music3DiT(shape, dtype=dtype)

