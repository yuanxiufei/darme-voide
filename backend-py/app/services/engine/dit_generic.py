"""通用单流 DiT 骨架 —— 把「判得出族、建不出结构」的那些权重变成能跑的前向。

盘上 45 份 DiT 只有 5 份 H3 与 1 份 SDXL 有实现，其余 **39 份**（`krea2` 15 /
`seedvr2` 13 / `wan` 3 / `qwen_image` 2 / `flux` 1 / `music3` 1 ……）多数只能被
:mod:`app.services.engine.dit_family` 判出族名 —— 判族脚本自己就写着「前向骨架未实现
⇒ 只能判族、不能前向」。

⚠️ 这份名单是**会过期的**（2026-10-31 已过期两次）✗：`zimage` 2 份走
:mod:`dit_lumina`、`qwen_image21` 2 份走 :mod:`dit_qwen21`，都已有前向骨架 ✓。
剩下**真正还只有骨架缺口**的是 `qwen_image` 2 份（2512 / Edit 2511，**双流 MMDiT**
`dual_stream=True`，本模块是单流 ⇒ **装不进这一份** ✗）、`wan` 3、`seedvr2` 13、
`flux` 1、`music3` 1。改这里之前先跑
``python -X utf8 -m app.scripts.dit_families``（这才是**真入口** ✓ —— `dit_family.py`
自己没有 ``__main__`` ✗，`-m app.services.engine.dit_family` 会静默空跑 ✓✗），别照这段文字数数。

本模块是那份骨架的第一块：单流（single-stream）Transformer，先接住结构最简单、盘上最多
的一族 —— `krea2`（Lumina-2 / NextDiT 同构）。

**维度一个都不写死**：全部从权重形状反推（:func:`infer_shape`）。参数名照权重逐字对齐，
所以 :meth:`Krea2DiT.state_dict` 的键集合可以直接与权重键集合双向对账。

语义照权威实现（comfy-studio 仓的 `ComfyUI/comfy/ldm/krea2/model.py`）：
`RMSNorm` 是 ``weight = scale + 1``；`SwiGLU` 中间维按权重取；`Attention` 的输出闸门是
``sigmoid(gate(x))`` 乘在 ``wo`` 之后；`mod.lin` 是加在 t 向量上的逐块共享偏移。

**没做的**（明确留白）：参考图分支（`ref_latents` / `index_timestep_zero`）、时间维
（5 维输入）、`transformer_options` 补丁钩子。这三块与「能不能算出数」正交，接法是纯加法。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#: RMSNorm 的 eps —— 照参考实现写死 ``1e-5``（不是 torch 默认的 ``None``/``1e-6``）。
RMS_EPS = 1e-5

#: 权重里「不是参数、是量化伴生」的键尾（对账时要滤掉，否则骨架永远少这些键）。
COMPANION_SUFFIXES = (
    ".weight_scale",
    ".comfy_quant",
    ".scale_weight",
    ".input_scale",
    ".scale_input",
)


class ShapeError(ValueError):
    """权重形状自相矛盾（同族不同层宽、`first` 与 `last` 对不上…）⇒ 不建半个结构。"""


class RMSNorm(nn.Module):
    """``weight = scale + 1`` 的 RMSNorm（Krea2/Lumina-2 约定，与 `nn.RMSNorm` 不同）。

    参数名必须是 ``scale``：权重里是 ``blocks.0.prenorm.scale``，叫 `weight` 就对不上。
    计算一律升 fp32（bf16 下 ``rms_norm`` 会掉精度，参考实现也这么做）。
    """

    def __init__(self, dim: int, eps: float = RMS_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        weight = self.scale.to(torch.float32) + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=self.eps).to(dtype)


class QKNorm(nn.Module):
    """按 head_dim 归一化 q/k（权重是 ``attn.qknorm.qnorm.scale``，形状 ``[head_dim]``）。"""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.qnorm = RMSNorm(head_dim)
        self.knorm = RMSNorm(head_dim)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k)


def rope_freqs(pos: Tensor, axes_dim: list[int], theta: float = 1e3) -> Tensor:
    """三维位置 id ``(b, n, 3)`` ⇒ 旋转矩阵 ``(b, n, Σdᵢ/2, 2, 2)``（照 Flux ``rope``）。

    ``scale = linspace(0, (d-2)/d, d/2)``、``omega = theta ** -scale``，用 fp64 算频率再降
    回 fp32 —— 位置 id 可到几千，fp32 累乘频率会丢低位（参考实现同样走 fp64）。
    """
    parts: list[Tensor] = []
    for axis, dim in enumerate(axes_dim):
        if dim % 2:
            raise ShapeError(f"轴 {axis} 的维度 {dim} 必须是偶数（每两个分量为一个旋转对）")
        scale = torch.linspace(0, (dim - 2) / dim, dim // 2, dtype=torch.float64, device=pos.device)
        omega = 1.0 / (theta ** scale)
        angles = torch.einsum("...n,d->...nd", pos[..., axis].to(torch.float64), omega)
        # (..., d/2, 4) ⇒ (..., d/2, 2, 2)：[[cos, -sin], [sin, cos]] 按列存
        stacked = torch.stack([angles.cos(), -angles.sin(), angles.sin(), angles.cos()], dim=-1)
        parts.append(stacked.reshape(*angles.shape, 2, 2))
    return torch.cat(parts, dim=-3).to(torch.float32)


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """``x (b,h,n,d)`` 与 ``freqs (b,1,n,d/2,2,2)`` ⇒ 旋转后的 x。

    旋转是**相邻两维配对**（``reshape(..., d//2, 2)`` 后第 0、1 个分量成对），不是 split-half
    —— 与 ``comfy.ldm.flux.math._apply_rope1`` 的 reshape 约定逐字一致。
    """
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    cos = freqs[..., 0, 0]
    sin = freqs[..., 1, 0]
    out = torch.stack([pairs[..., 0] * cos - pairs[..., 1] * sin,
                       pairs[..., 0] * sin + pairs[..., 1] * cos], dim=-1)
    # ⚠️ **必须显式降回 ``x.dtype``** ✗✗：``freqs`` 是 fp32（见 :func:`rope_freqs` ✓ —— 位置 id
    #    可到几千，频率要 fp64/fp32 算 ✓），乘上去会把 bf16 的 q/k **悄悄升成 fp32** ✓✗ ⇒
    #    `scaled_dot_product_attention` 直接报 "query float32 / value bf16" ✗（SDPA 要求三者同
    #    dtype ✓）。2026-10-31 在盘上真权重 11 GiB 跑 Z-Image 时撞的 ✓ —— 小配置 fp32 自检
    #    全绿 ✗，因为整条链都是 fp32 时"升成 fp32"看不出来 ✓✗（典型：判据缺了精度这一维 ✓）。
    return out.reshape(*x.shape).to(x.dtype)


def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0,
                       time_factor: float = 1000.0) -> Tensor:
    """正弦时间嵌入（``time_factor=1000`` 是 Krea2 的 t∈[0,1] 约定）。"""
    scaled = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, dtype=torch.float32, device=scaled.device) / half)
    args = scaled.float()[:, None] * freqs[None]
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb.to(scaled.dtype) if torch.is_floating_point(scaled) else emb


class SwiGLU(nn.Module):
    """``down(silu(gate(x)) * up(x))``；中间维由调用方按权重形状给出。"""

    def __init__(self, features: int, mlpdim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(features, mlpdim, bias=False)
        self.up = nn.Linear(features, mlpdim, bias=False)
        self.down = nn.Linear(mlpdim, features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """GQA：q/k 先 per-head RMSNorm、再 3 轴 RoPE、再 SDPA，输出乘 ``sigmoid(gate)``。

    闸门在 **``wo`` 之后**（参考：``out = self.wo(out) * sigmoid(gate)``），不是加在残差上
    —— 顺序错了数值仍然「看着合理」，只有逐位对拍才抓得住。
    """

    def __init__(self, dim: int, heads: int, head_dim: int, kvheads: int) -> None:
        super().__init__()
        self.heads, self.kvheads, self.head_dim = heads, kvheads, head_dim
        self.wq = nn.Linear(dim, heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, kvheads * head_dim, bias=False)
        self.wv = nn.Linear(dim, kvheads * head_dim, bias=False)
        self.wo = nn.Linear(heads * head_dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.qknorm = QKNorm(head_dim)

    def forward(self, x: Tensor, freqs: Tensor | None = None) -> Tensor:
        b, n, _ = x.shape
        q = self.wq(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, n, self.kvheads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, n, self.kvheads, self.head_dim).transpose(1, 2)
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=self.heads != self.kvheads)
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim)
        return self.wo(out) * torch.sigmoid(self.gate(x))


class SimpleModulation(nn.Module):
    """``out = tvec + lin`` 再切 2 份 ⇒ ``(scale, shift)``（``lin`` 形状 ``[2, dim]``）。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))

    def forward(self, vec: Tensor) -> tuple[Tensor, Tensor]:
        #: ⚠️ 切分维是 ``-2`` 不是 ``-1``：``lin`` 形状 ``(2, dim)`` 要先 ``unsqueeze(0)`` 才能与
        #: ``vec (b, 1, dim)`` 广播成 ``(b, 2, dim)``，此时「哪一半」在**倒数第二维** ✓。
        #: 照 ``chunk(2, dim=-1)`` 写会把 dim 切成两半 ⇒ 最后一层报 ``a (32) vs b (64)`` ✓✗
        #: （参考实现：``out = vec + self.lin.unsqueeze(0); out.chunk(2, dim=1)``）。
        if vec.ndim == 2:
            vec = vec.unsqueeze(1)
        scale, shift = (vec + self.lin.to(vec.dtype).unsqueeze(0)).chunk(2, dim=-2)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    """``out = tvec + lin`` 再切 6 份（``lin`` 形状 ``[6*dim]``，一维、加在 t 向量上）。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: Tensor) -> tuple[Tensor, ...]:
        return tuple((vec + self.lin.to(vec.dtype)).chunk(6, dim=-1))


class TextFusionBlock(nn.Module):
    """文本侧单流块（无调制）：``x += attn(prenorm(x))``；``x += mlp(postnorm(x))``。"""

    def __init__(self, features: int, heads: int, head_dim: int, kvheads: int, mlpdim: int) -> None:
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, head_dim, kvheads)
        self.mlp = SwiGLU(features, mlpdim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.prenorm(x))
        return x + self.mlp(self.postnorm(x))


class TextFusionTransformer(nn.Module):
    """把 N 层文本特征压成 1 层：逐层块（``(b·l, n, d)`` 展平）→ ``projector`` → 精修块。

    ``projector`` 是 ``Linear(num_txt_layers, 1)``（权重形状 ``[1, 12]``）—— 它给每一层学一个
    标量权重再求和，不是把特征投影到别的维。
    """

    def __init__(self, txtlayers: int, txtdim: int, heads: int, head_dim: int,
                 kvheads: int, mlpdim: int, refiner_layers: int, blocks: int = 0) -> None:
        super().__init__()
        #: ``blocks`` 缺省等于 ``txtlayers`` 只是为了让旧调用点还能跑 ✓；krea2 一律显式传
        #: ``shape.txt_blocks`` ✓（盘上 2 个块对 12 层特征 ✓）。
        count = blocks or txtlayers
        self.layerwise_blocks = nn.ModuleList([
            TextFusionBlock(txtdim, heads, head_dim, kvheads, mlpdim) for _ in range(count)])
        #: ⚠️ ``bias=False``：真实权重里**没有** ``txtfusion.projector.bias``（对账抓出来的）✓
        self.projector = nn.Linear(txtlayers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([
            TextFusionBlock(txtdim, heads, head_dim, kvheads, mlpdim)
            for _ in range(refiner_layers)])

    def forward(self, x: Tensor) -> Tensor:
        b, n, layers, dim = x.shape
        #: 核的是**输入层数 == `projector` 的入维** ✓，不是「层数 == 块数」✗（盘上 12 vs 2）。
        if layers != self.projector.in_features:
            raise ShapeError(f"文本特征层数 {layers} ≠ `projector` 入维 "
                             f"{self.projector.in_features}")
        flat = x.reshape(b * layers, n, dim)
        for block in self.layerwise_blocks:
            flat = block(flat)
        back = flat.view(b, layers, n, dim).transpose(1, -1)   # (b, d, n, l)
        x = self.projector(back).squeeze(-1).transpose(1, 2)   # (b, n, d)
        for block in self.refiner_blocks:
            x = block(x)
        return x


class LastLayer(nn.Module):
    """``(1 + scale) * norm(x) + shift`` ⇒ 线性到 ``channels * patch²``（带 bias）。"""

    def __init__(self, features: int, patch: int, channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        return self.linear((1 + scale) * self.norm(x) + shift)


class SingleStreamBlock(nn.Module):
    """``x += pregate·attn((1+prescale)·prenorm(x)+preshift)``；``x += postgate·mlp(…)``。"""

    def __init__(self, dim: int, heads: int, head_dim: int, kvheads: int,
                 mlpdim: int, modulation: bool = True) -> None:
        super().__init__()
        self.prenorm = RMSNorm(dim)
        self.postnorm = RMSNorm(dim)
        self.attn = Attention(dim, heads, head_dim, kvheads)
        self.mlp = SwiGLU(dim, mlpdim)
        #: 没有调制参数的族（扁平 DiT）就关掉这块，键集才对得上。
        self.mod = DoubleSharedModulation(dim) if modulation else None

    def forward(self, x: Tensor, tvec: Tensor | None = None,
                freqs: Tensor | None = None) -> Tensor:
        if self.mod is None or tvec is None:
            x = x + self.attn(self.prenorm(x), freqs)
            return x + self.mlp(self.postnorm(x))
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(tvec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs)
        return x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)


@dataclass(frozen=True)
class Krea2Shape:
    """一组**从权重形状反推**出来的结构常量（所有维度都有出处，没有手抄值）。"""

    features: int
    layers: int
    heads: int
    head_dim: int
    kvheads: int
    mlpdim: int
    patch: int
    channels: int
    tdim: int
    txtdim: int
    txtlayers: int
    #: ⚠️ **两个不同的数**：``txtlayers`` 是文本特征的层数（``projector`` 的入维，12 ✓），
    #: ``txt_blocks`` 是逐层块**模块**的个数（盘上是 2 ✓）。它们**不是一回事** ✗ ——
    #: 盘上 2 个块要对 12 层特征跑（``reshape(b·l, n, d)`` 后一次吃 12・b 条序列 ✓）。
    #: 照 ``txtlayers`` 建块会凭空多出 10 个块的参数 ✓✗（真实权重对账当场抓到 ✓）。
    txt_blocks: int
    txt_heads: int
    txt_head_dim: int
    txt_kvheads: int
    txt_mlpdim: int
    refiner_layers: int
    theta: float = 1e3

    @property
    def axes_dim(self) -> list[int]:
        """RoPE 三轴 = ``[head_dim-12u, 6u, 6u]``，``u = head_dim//16``；128 ⇒ ``[32,48,48]``。"""
        unit = self.head_dim // 16
        return [self.head_dim - 12 * unit, 6 * unit, 6 * unit]


def _shape_of(shapes: dict[str, tuple[int, ...]], key: str) -> tuple[int, ...]:
    try:
        return tuple(shapes[key])
    except KeyError:
        raise ShapeError(f"权重里缺 `{key}`：这份文件不是 krea2 单流结构") from None


def _block_indexes(shapes: dict[str, tuple[int, ...]], prefix: str) -> list[int]:
    """从 ``<prefix>.<i>.…`` 取块号并排序（不数键名、不假设连续）。"""
    head = prefix + "."
    found: set[int] = set()
    for key in shapes:
        if not key.startswith(head):
            continue
        head_part = key[len(head):].split(".", 1)[0]
        if head_part.isdigit():
            found.add(int(head_part))
    if not found:
        raise ShapeError(f"权重里没有 `{prefix}.<i>.…` 的任何键")
    if found != set(range(len(found))):
        raise ShapeError(f"`{prefix}` 的块号不连续：{sorted(found)}（缺层不许当没看见）")
    return sorted(found)


def infer_shape(shapes: dict[str, tuple[int, ...]], *, patch: int = 2,
                theta: float = 1e3) -> Krea2Shape:
    """从权重形状反推全套结构常量；任何一处对不上就抛 :class:`ShapeError`。

    ``patch`` 是唯一需要外部告知的数（``first.weight`` 的入维 = ``channels·patch²``，单看形状
    分不出「16ch×4」与「64ch×1」）。它的存在性必须能被形状证实：``first`` 入维与
    ``last.linear`` 出维都得能整除 ``patch²``。
    """
    if patch < 1:
        raise ShapeError(f"patch 必须 ≥ 1，拿到 {patch}")
    blocks = _block_indexes(shapes, "blocks")
    first = _shape_of(shapes, "first.weight")
    last = _shape_of(shapes, "last.linear.weight")
    if first[1] != last[0]:
        raise ShapeError(f"`first` 入维 {first[1]} ≠ `last` 出维 {last[0]}：首尾不属于同一结构")
    if first[1] % (patch * patch):
        raise ShapeError(f"`first` 入维 {first[1]} 不能按 patch={patch} 拆成 channels·patch²")
    head_dim = _shape_of(shapes, "blocks.0.attn.qknorm.qnorm.scale")[0]
    features = first[0]
    q_out = _shape_of(shapes, "blocks.0.attn.wq.weight")[0]
    kv_out = _shape_of(shapes, "blocks.0.attn.wk.weight")[0]
    if q_out % head_dim or kv_out % head_dim:
        raise ShapeError(f"q/k 输出维 {q_out}/{kv_out} 不是 head_dim={head_dim} 的整数倍")
    if q_out % features and q_out != features:
        raise ShapeError(f"q 输出维 {q_out} 与 hidden {features} 不成比例")
    mlpdim = _shape_of(shapes, "blocks.0.mlp.gate.weight")[0]
    # 逐块核一致：一个文件里层宽混着来，说明「同族」判断本身就错了。
    for i in blocks:
        for key, want in (("attn.wq.weight", q_out), ("attn.wk.weight", kv_out),
                          ("mlp.gate.weight", mlpdim), ("prenorm.scale", features)):
            got = _shape_of(shapes, f"blocks.{i}.{key}")[0]
            if got != want:
                raise ShapeError(f"`blocks.{i}.{key}` 宽度 {got} ≠ 块 0 的 {want}（层宽不一致）")
    txt_blocks = _block_indexes(shapes, "txtfusion.layerwise_blocks")
    refiners = _block_indexes(shapes, "txtfusion.refiner_blocks")
    txt_head_dim = _shape_of(shapes, "txtfusion.layerwise_blocks.0.attn.qknorm.qnorm.scale")[0]
    return Krea2Shape(
        features=features,
        layers=len(blocks),
        heads=q_out // head_dim,
        head_dim=head_dim,
        kvheads=kv_out // head_dim,
        mlpdim=mlpdim,
        patch=patch,
        channels=first[1] // (patch * patch),
        tdim=_shape_of(shapes, "tmlp.0.weight")[1],
        txtdim=_shape_of(shapes, "txtmlp.1.weight")[1],
        txtlayers=_shape_of(shapes, "txtfusion.projector.weight")[1],
        txt_blocks=len(txt_blocks),
        txt_heads=_shape_of(shapes, "txtfusion.layerwise_blocks.0.attn.wq.weight")[0] // txt_head_dim,
        txt_head_dim=txt_head_dim,
        txt_kvheads=_shape_of(shapes, "txtfusion.layerwise_blocks.0.attn.wk.weight")[0]
        // txt_head_dim,
        txt_mlpdim=_shape_of(shapes, "txtfusion.layerwise_blocks.0.mlp.gate.weight")[0],
        refiner_layers=len(refiners),
        theta=theta,
    ) if txt_blocks else _raise_missing_text(shapes)


def _raise_missing_text(shapes: dict[str, tuple[int, ...]]) -> Krea2Shape:
    raise ShapeError("权重里缺 `txtfusion.layerwise_blocks`：不是 krea2 结构")


def weight_tensor_keys(shapes: dict[str, tuple[int, ...]]) -> set[str]:
    """权重文件里属于骨架的参数键（滤掉量化伴生键：它们不是参数，反量化时被吃掉）。"""
    return {k for k in shapes if not k.endswith(COMPANION_SUFFIXES)}


def expected_keys(shape: Krea2Shape) -> set[str]:
    """骨架**应该**产出的权重键集合（与 :func:`weight_tensor_keys` 双向对账用）。"""
    keys = {"first.weight", "first.bias",
            "tmlp.0.weight", "tmlp.0.bias", "tmlp.2.weight", "tmlp.2.bias",
            "tproj.1.weight", "tproj.1.bias",
            "txtmlp.0.scale", "txtmlp.1.weight", "txtmlp.1.bias",
            "txtmlp.3.weight", "txtmlp.3.bias",
            "txtfusion.projector.weight",
            "last.norm.scale", "last.linear.weight", "last.linear.bias",
            "last.modulation.lin"}
    for i in range(shape.layers):
        keys |= _block_keys(f"blocks.{i}.")
        #: 单流块的调制向量是**逐块**参数（`blocks.N.mod.lin`，一维 6·hidden）—— 文本侧的块
        #: 没有它（`TextFusionBlock` 无调制），所以只在这一处补。
        keys.add(f"blocks.{i}.mod.lin")
    for group, count in (("layerwise_blocks", shape.txt_blocks),
                         ("refiner_blocks", shape.refiner_layers)):
        for i in range(count):
            keys |= _block_keys(f"txtfusion.{group}.{i}.")
    return keys


def _block_keys(head: str) -> set[str]:
    tail = {"prenorm.scale", "postnorm.scale",
            "attn.wq.weight", "attn.wk.weight", "attn.wv.weight",
            "attn.wo.weight", "attn.gate.weight",
            "attn.qknorm.qnorm.scale", "attn.qknorm.knorm.scale",
            "mlp.gate.weight", "mlp.up.weight", "mlp.down.weight"}
    return {head + name for name in tail}


def _pad_to_patch(x: Tensor, patch: int) -> Tensor:
    """右/下补零到 patch 整数倍（参考实现同样做），返回新张量、不改入参。"""
    h, w = x.shape[-2], x.shape[-1]
    ph, pw = (-h) % patch, (-w) % patch
    return x if not ph and not pw else F.pad(x, (0, pw, 0, ph))


class Krea2DiT(nn.Module):
    """Krea2 / Lumina-2 同构的单流 DiT（结构由 :class:`Krea2Shape` 驱动）。

    前向口径（与参考实现逐条对应）：

    1. patchify：``(b, c, h, w)`` ⇒ ``(b, h·w, c·p²)``，顺序 ``(c, ph, pw)``；
    2. 图像位置 id 是 ``(0, 行, 列)``、文本位置 id **全 0**，两者拼在序列维；
    3. 文本序列在前、图像序列在后 ⇒ 输出只取图像那一段；
    4. 裁剪回原尺寸（补零的部分不能吐给下游）。
    """

    def __init__(self, shape: Krea2Shape) -> None:
        super().__init__()
        self.shape_ = shape
        self.patch = shape.patch
        self.channels = shape.channels
        self.first = nn.Linear(shape.channels * shape.patch * shape.patch, shape.features)
        self.tmlp = nn.Sequential(
            nn.Linear(shape.tdim, shape.features), nn.GELU(approximate="tanh"),
            nn.Linear(shape.features, shape.features))
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"), nn.Linear(shape.features, 6 * shape.features))
        self.txtmlp = nn.Sequential(
            RMSNorm(shape.txtdim), nn.Linear(shape.txtdim, shape.features),
            nn.GELU(approximate="tanh"), nn.Linear(shape.features, shape.features))
        self.txtfusion = TextFusionTransformer(
            txtlayers=shape.txtlayers, txtdim=shape.txtdim, heads=shape.txt_heads,
            head_dim=shape.txt_head_dim, kvheads=shape.txt_kvheads, mlpdim=shape.txt_mlpdim,
            refiner_layers=shape.refiner_layers, blocks=shape.txt_blocks)
        self.blocks = nn.ModuleList([
            SingleStreamBlock(shape.features, shape.heads, shape.head_dim, shape.kvheads,
                              shape.mlpdim) for _ in range(shape.layers)])
        self.last = LastLayer(shape.features, shape.patch, shape.channels)

    # ------------------------------------------------------------------ 前向

    def process_img(self, x: Tensor, index: int = 0) -> tuple[Tensor, Tensor, int, int]:
        """``(b,c,h,w)`` ⇒ ``(tokens, ids, h_tok, w_tok)``；返回的 h/w 是**补零后**的 token 网格。"""
        patch = self.patch
        padded = _pad_to_patch(x, patch)
        b, c, h, w = padded.shape
        hh, ww = h // patch, w // patch
        tokens = (padded.reshape(b, c, hh, patch, ww, patch)
                  .permute(0, 2, 4, 1, 3, 5)          # (b, hh, ww, c, ph, pw)
                  .reshape(b, hh * ww, c * patch * patch))
        ids = torch.zeros(hh, ww, 3, device=x.device, dtype=torch.float32)
        ids[..., 0] = index
        ids[..., 1] = torch.arange(hh, device=x.device, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(ww, device=x.device, dtype=torch.float32)[None, :]
        return tokens, ids.reshape(1, hh * ww, 3).repeat(b, 1, 1), hh, ww

    def _unpack_context(self, context: Tensor) -> Tensor:
        """文本条件：``(b, seq, txtlayers·txtdim)`` 或已展开的 ``(b, seq, txtlayers, txtdim)``。"""
        if context.ndim == 4:
            if context.shape[2] != self.shape_.txtlayers:
                raise ShapeError(f"文本条件层数 {context.shape[2]} ≠ "
                                 f"{self.shape_.txtlayers}（拿错 TE 了）")
            return context
        b, seq, fused = context.shape
        want = self.shape_.txtlayers * self.shape_.txtdim
        if fused != want:
            raise ShapeError(f"文本条件宽度 {fused} ≠ {self.shape_.txtlayers}×"
                             f"{self.shape_.txtdim}={want}（应来自 {self.shape_.txtlayers} 层 "
                             f"Qwen3-VL 栈）")
        return context.reshape(b, seq, self.shape_.txtlayers, self.shape_.txtdim)

    def forward(self, x: Tensor, timesteps: Tensor, context: Tensor) -> Tensor:
        """``x (b,c,h,w)`` ＋ ``timesteps (b,)`` ＋ ``context`` ⇒ 速度预测 ``(b,c,h,w)``。"""
        if x.ndim != 4:
            raise ShapeError(f"只支持 4 维 latent（图像），拿到 {x.ndim} 维")
        if x.shape[1] != self.channels:
            raise ShapeError(f"latent 通道 {x.shape[1]} ≠ 结构里的 {self.channels}")
        b, _, h0, w0 = x.shape
        img, imgpos, hh, ww = self.process_img(x)
        img_tokens = img.shape[1]
        img = self.first(img)

        t = self.tmlp(timestep_embedding(timesteps, self.shape_.tdim).unsqueeze(1)
                      .to(img.dtype))
        tvec = self.tproj(t)

        ctx = self.txtmlp(self.txtfusion(self._unpack_context(context)))
        txtlen = ctx.shape[1]
        txtpos = torch.zeros(b, txtlen, 3, device=ctx.device, dtype=torch.float32)

        combined = torch.cat((ctx, img), dim=1)
        pos = torch.cat((txtpos, imgpos), dim=1)
        freqs = rope_freqs(pos, self.shape_.axes_dim, self.shape_.theta).unsqueeze(1)
        for block in self.blocks:
            combined = block(combined, tvec, freqs)

        out = self.last(combined, t)[:, txtlen:txtlen + img_tokens, :]
        out = out.reshape(b, hh, ww, self.channels, self.patch, self.patch)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(b, self.channels, hh * self.patch,
                                                    ww * self.patch)
        return out[:, :, :h0, :w0]


def build_from_shapes(shapes: dict[str, tuple[int, ...]], *, patch: int = 2,
                      dtype: torch.dtype = torch.float32) -> Krea2DiT:
    """按权重**形状**建出骨架（不读权重内容）；``dtype`` 只影响初始化的参数类型。"""
    shape = infer_shape(shapes, patch=patch)
    model = Krea2DiT(shape)
    return model if dtype is torch.float32 else model.to(dtype)


__all__ = [
    "Attention",
    "COMPANION_SUFFIXES",
    "DoubleSharedModulation",
    "Krea2DiT",
    "Krea2Shape",
    "LastLayer",
    "QKNorm",
    "RMS_EPS",
    "RMSNorm",
    "ShapeError",
    "SimpleModulation",
    "SingleStreamBlock",
    "SwiGLU",
    "TextFusionBlock",
    "TextFusionTransformer",
    "apply_rope",
    "build_from_shapes",
    "expected_keys",
    "infer_shape",
    "rope_freqs",
    "timestep_embedding",
    "weight_tensor_keys",
]
