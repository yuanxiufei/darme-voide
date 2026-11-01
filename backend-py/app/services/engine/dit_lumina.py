"""**NextDiT / Lumina-2 形态的单流 DiT 前向骨架**（Z-Image 一族，盘上 2 份）。

与 :mod:`dit_generic`（Krea2 形态）是**两种键布局**，不能混用：

* Krea2：``first/last`` ＋ ``blocks.N`` ＋ ``txtfusion.*``（文本特征先过融合栈再进块）；
* Z-Image：``x_embedder/cap_embedder`` ＋ ``context_refiner/noise_refiner/layers`` ＋
  ``final_layer``（文本侧**只有** 2 层 context refiner，没有 12 层特征融合）。

参考实现：``comfy-studio/ComfyUI/comfy/ldm/lumina/model.py``（``NextDiT``）与
``comfy/model_detection.py`` 里 Z-Image 的构造参数。前向口径逐条对应：

1. ``t = 1 - timesteps``（**先取反**再乘 ``time_scale=1000``）—— 少这一步形状完全正常、
   数值全错，是这一族最容易写错的一行；
2. 文本侧：``cap_embedder``（RMSNorm → Linear）⇒ ``context_refiner``（**无调制**，纯
   norm/attn/ffn 残差）。⚠️ 参考实现**不补位** ✗：``pad_tokens_multiple=None`` ⇒
   那两个 ``*_pad_token`` 参数**建了但从不使用** ✓（见 ``LuminaShape.pad_tokens``）；
3. 图像侧：patchify（顺序 ``(c, ph, pw)``）⇒ ``x_embedder`` ⇒ ``noise_refiner``（**有** adaLN 调制）；
4. 位置 id：文本 ``t = 1…L``（注意**从 1 起** ✓ 不是 0 ✗）、图像 ``t = L+1``、图像内 ``(行, 列)``；
   三轴 ``[32,48,48]``、``theta=256``（不是 Krea2 的 1e3），相邻两维配对旋转；
5. 序列拼成 ``[文本; 图像]`` 过 ``layers``（整段自注意力；参考里 ``cap_mask = None`` ⇒
   文本**不遮罩** ✓，所以填充会直接污染注意力 ✓✗ —— 这也是「不填位」必须对齐的原因）；
   ``final_layer`` 对整段算完再从文本长度处切开；
6. 输出取负（参考 ``return -img``）。

权重里**没有**、只能由调用方告知的超参（全部收进 :class:`LuminaShape` 并在此写明）：
``theta=256.0``、``time_scale=1000.0``、``pad_tokens=0``（**不填位** ✓，出处见该字段注释）、
``norm_eps=1e-5``、
``qk_eps``（per-head qk-norm，参考走 ``eps=None`` ⇒ 量化路径取 fp32 eps）、
``mod_silu=False``（Z-Image 的块调制是裸 Linear，没有 SiLU；Lumina-2 原版有）。

未实现（有意留白）：omni 多图/音视频分支、``siglip`` 视觉条件、
``clip_text_pooled_proj``、``x_mask`` 遮罩推理。缺口登记在
:mod:`app.scripts.dit_families` 的 ``ENGINE_STATE``。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dit_generic import (COMPANION_SUFFIXES, ShapeError, apply_rope, rope_freqs,
                          timestep_embedding)

__all__ = [
    "COMPANION_SUFFIXES",
    "FeedForward",
    "FinalLayer",
    "JointAttention",
    "LuminaBlock",
    "LuminaDiT",
    "LuminaShape",
    "PlainRMSNorm",
    "ShapeError",
    "build_from_shapes",
    "expected_keys",
    "infer_shape",
    "weight_tensor_keys",
]

#: 层归一化 eps：参考实现 ``NextDiT(norm_eps=1e-5)`` 显式传入。
NORM_EPS = 1e-5
#: ``final_layer.norm_final`` 是 ``nn.LayerNorm(eps=1e-6)``（torch 默认值），与 RMSNorm 那套不同。
FINAL_EPS = 1e-6
#: per-head qk-norm：参考写 ``RMSNorm(head_dim)``（``eps=None``），量化路径回落 fp32 eps。
QK_EPS = torch.finfo(torch.float32).eps


class PlainRMSNorm(nn.Module):
    """普通 RMSNorm：权重**直接**乘（``weight`` 而不是 ``scale + 1``）。

    ⚠️ 这一族与 Krea2 的约定相反 —— 权重名是 ``attention_norm1.weight``、``q_norm.weight``，
    值就是乘数；照 :class:`dit_generic.RMSNorm` 那样再 ``+1`` 会让每层输出偏大且无处报错。
    """

    def __init__(self, dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class JointAttention(nn.Module):
    """qkv **合在一个线性层**（``attention.qkv``）里，再按 heads/kvheads 切三段。

    顺序：切段 ⇒ per-head qk-norm ⇒ 三轴 RoPE ⇒ SDPA ⇒ ``out``。
    这一族**没有** Krea2 那种 ``sigmoid(gate)`` 输出闸门，照抄会多出参数。
    """

    def __init__(self, dim: int, heads: int, head_dim: int, kvheads: int,
                 qk_eps: float = QK_EPS) -> None:
        super().__init__()
        self.heads, self.head_dim, self.kvheads = heads, head_dim, kvheads
        self.qkv = nn.Linear(dim, (heads + 2 * kvheads) * head_dim, bias=False)
        self.out = nn.Linear(heads * head_dim, dim, bias=False)
        self.q_norm = PlainRMSNorm(head_dim, qk_eps)
        self.k_norm = PlainRMSNorm(head_dim, qk_eps)

    def forward(self, x: Tensor, freqs: Tensor) -> Tensor:
        b, n, _ = x.shape
        q_size, kv_size = self.heads * self.head_dim, self.kvheads * self.head_dim
        q, k, v = torch.split(self.qkv(x), [q_size, kv_size, kv_size], dim=-1)
        #: qk-norm 归一化的**最后一维是 head_dim**：必须先拆头再归一化，按整段归一化会
        #: 报 ``weight [head_dim] vs normalized_shape [heads·head_dim]``。
        q = self.q_norm(q.view(b, n, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(b, n, self.kvheads, self.head_dim)).transpose(1, 2)
        v = v.view(b, n, self.kvheads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        out = F.scaled_dot_product_attention(
            q, k, v, enable_gqa=self.heads != self.kvheads)
        return self.out(out.transpose(1, 2).reshape(b, n, q_size))


class FeedForward(nn.Module):
    """SwiGLU，参数名是 ``w1``（门）/``w3``（升）/``w2``（降）—— 顺序照权重，不照习惯。"""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LuminaBlock(nn.Module):
    """``norm ⇒ attention ⇒ norm2`` 残差 ＋ ``norm ⇒ ffn ⇒ norm2`` 残差。

    ``modulated=True`` 时多一个 ``adaLN_modulation``（``t 向量 ⇒ 4·dim``），按
    ``(1 + scale)`` 调制、``tanh(gate)`` 收缩 —— **调制在残差支路内部**，加在残差上就错。
    """

    def __init__(self, dim: int, heads: int, head_dim: int, kvheads: int, hidden: int,
                 mod_dim: int, *, modulated: bool, mod_silu: bool = False,
                 norm_eps: float = NORM_EPS, qk_eps: float = QK_EPS) -> None:
        super().__init__()
        self.attention_norm1 = PlainRMSNorm(dim, norm_eps)
        self.attention_norm2 = PlainRMSNorm(dim, norm_eps)
        self.ffn_norm1 = PlainRMSNorm(dim, norm_eps)
        self.ffn_norm2 = PlainRMSNorm(dim, norm_eps)
        self.attention = JointAttention(dim, heads, head_dim, kvheads, qk_eps)
        self.feed_forward = FeedForward(dim, hidden)
        self.modulated = modulated
        if modulated:
            layers: list[nn.Module] = [nn.Linear(mod_dim, 4 * dim)]
            if mod_silu:
                #: 有 SiLU 时 Linear 落在 ``.1``（Lumina-2 原版权重），没有时在 ``.0``（Z-Image）。
                layers.insert(0, nn.SiLU())
            self.adaLN_modulation = nn.Sequential(*layers)

    def forward(self, x: Tensor, freqs: Tensor, tvec: Tensor | None = None) -> Tensor:
        if self.modulated:
            if tvec is None:
                raise ShapeError("这个块有调制，必须给 t 向量")
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                tvec).chunk(4, dim=1)
            gate_msa = gate_msa.tanh().unsqueeze(1)
            gate_mlp = gate_mlp.tanh().unsqueeze(1)
            scale_msa = scale_msa.unsqueeze(1)
            scale_mlp = scale_mlp.unsqueeze(1)

            h = self.attention_norm1(x) * (1.0 + scale_msa)
            h = self.attention_norm2(self.attention(h, freqs))
            x = x + h * gate_msa
            h = self.ffn_norm1(x) * (1.0 + scale_mlp)
            h = self.ffn_norm2(self.feed_forward(h))
            return x + h * gate_mlp

        x = x + self.attention_norm2(self.attention(self.attention_norm1(x), freqs))
        return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))


class FinalLayer(nn.Module):
    """``norm_final``（无仿射 ⇒ 不出参数）⇒ 调制 ⇒ ``linear`` 回 ``channels·patch²``。"""

    def __init__(self, dim: int, patch: int, channels: int, mod_dim: int,
                 eps: float = FINAL_EPS) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.linear = nn.Linear(dim, patch * patch * channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(mod_dim, dim))

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale = self.adaLN_modulation(tvec)
        return self.linear(self.norm_final(x) * (1.0 + scale.unsqueeze(1)))


class TimestepEmbedder(nn.Module):
    """``正弦嵌入(256) ⇒ Linear ⇒ SiLU ⇒ Linear``；正弦那步**不再**乘 1000（外面乘过了）。"""

    def __init__(self, freq_dim: int, inner: int, out_dim: int) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, inner), nn.SiLU(),
                                 nn.Linear(inner, out_dim))

    def forward(self, t: Tensor) -> Tensor:
        #: ⚠️ 两件事挤在这一行里，缺一条就崩 ✓✗：
        #: ① 正弦**在 f32 里算** ✓（``t.float()`` ✓）—— 用 bf16 算 ``cos/sin`` 会把相邻时间步糊成
        #:    同一个数 ✓✗（8 步 turbo 里那等于丢掉一半步进信息 ✓，症状是"能跑但糊" ✓）；
        #: ② 进 ``mlp`` 前**对齐到权重精度** ✓ —— 参考实现是 ``.to(self.dtype)`` ✓；不做则当场
        #:    ``mat1 and mat2 must have the same dtype, but got Float and BFloat16`` ✓✗。
        #:    注意**不要**反过来把时间向量降精度 ✗：采样侧给 f32 是对的 ✓（见 `_timesteps_of` ✓），
        #:    该 cast 的地方是模型内部这一处 ✓。
        return self.mlp(timestep_embedding(t.float(), self.freq_dim, time_factor=1.0)
                        .to(self.mlp[0].weight.dtype))


@dataclass
class LuminaShape:
    """Z-Image / Lumina-2 的全套结构常量（能反推的全从权重形状反推）。

    后面几个是**超参**（权重里没有）：``pad_tokens``/``theta``/``time_scale``/
    ``mod_silu``/``norm_eps``/``qk_eps`` —— 改它们不会改变键集合，只会改变数值。
    ⚠️ 例外是 ``has_pad_tokens`` ✗：它**由权重决定** ✓（有没有 ``*_pad_token`` 这两个键 ✓），
    改它会改变 :func:`expected_keys` ⇒ 与真权重对不上 ⇒ 装载当场报缺键/多键 ✓。
    """

    features: int
    layers: int
    heads: int
    head_dim: int
    kvheads: int
    mlpdim: int
    patch: int
    channels: int
    cap_dim: int
    t_freq: int
    t_inner: int
    t_out: int
    refiner_layers: int
    context_refiner_layers: int
    mod_dim: int
    #: ⚠️ **两件事，别混成一格** ✗✗（这是我上轮估错过的地方 ✓）：
    #: ① ``has_pad_tokens`` = 权重里**有没有** ``x_pad_token``/``cap_pad_token`` ✓ ——
    #:    有就必须建（不建则键集合对不上、装载报缺键 ✓✗）；
    #: ② ``pad_tokens`` = 前向**填不填**、按多少对齐 ✓ —— 参考实现的真值是 **0（不填）** ✓✗：
    #:    ``comfy/ldm/lumina/model.py`` 的 ``NextDiT.__init__(pad_tokens_multiple=None)`` ✓
    #:    且 `model_base.py` / `supported_models.py` **从不给 ZImage / Lumina2 传这个值** ✓
    #:    ⇒ ``if self.pad_tokens_multiple is not None`` 那段恒不执行 ✓。
    #:    我上轮把超参估成 32 ✗ ⇒ 每张图多填最多 31 个 cap token，而参考的 ``cap_mask = None`` ✓✗
    #:    ⇒ 这些填充 token **不遮罩、直接参与注意力** ✓ ⇒ 形状对、不报错、结果只是与参考不同 ✓✗。
    has_pad_tokens: bool = True
    pad_tokens: int = 0
    theta: float = 256.0
    time_scale: float = 1000.0
    mod_silu: bool = False
    norm_eps: float = NORM_EPS
    qk_eps: float = QK_EPS

    @property
    def axes_dim(self) -> list[int]:
        """三轴 RoPE 维度：``[d-12u, 6u, 6u]``，``u = head_dim // 16``（128 ⇒ [32,48,48]）。"""
        unit = self.head_dim // 16
        return [self.head_dim - 12 * unit, 6 * unit, 6 * unit]

    @property
    def modulation_index(self) -> int:
        """块调制里 Linear 在 ``nn.Sequential`` 的位置（无 SiLU ⇒ 0 ／有 ⇒ 1）。"""
        return 1 if self.mod_silu else 0


def _shape_of(shapes: dict[str, tuple[int, ...]], key: str) -> tuple[int, ...]:
    try:
        return tuple(shapes[key])
    except KeyError:
        raise ShapeError(f"权重里缺 `{key}`：这份文件不是 z-image/lumina2 单流结构") from None


def _block_indexes(shapes: dict[str, tuple[int, ...]], prefix: str,
                   *, required: bool = True) -> list[int]:
    head = prefix + "."
    found: set[int] = set()
    for key in shapes:
        if not key.startswith(head):
            continue
        part = key[len(head):].split(".", 1)[0]
        if part.isdigit():
            found.add(int(part))
    if not found:
        if required:
            raise ShapeError(f"权重里没有 `{prefix}.<i>.…` 的任何键")
        return []
    if found != set(range(len(found))):
        raise ShapeError(f"`{prefix}` 的块号不连续：{sorted(found)}（缺层不许当没看见）")
    return sorted(found)


def infer_shape(shapes: dict[str, tuple[int, ...]], *, patch: int = 2,
                theta: float = 256.0, time_scale: float = 1000.0, pad_tokens: int = 0,
                mod_silu: bool = False) -> LuminaShape:
    """从权重形状反推全套常量；任何一处对不上就抛 :class:`ShapeError`。

    ``patch`` 是唯一必须外部告知的数（``x_embedder`` 入维 = ``channels·patch²``，单看形状
    分不出「16ch×4」与「64ch×1」）；它必须同时整除 ``x_embedder`` 入维与 ``final_layer`` 出维。
    """
    if patch < 1:
        raise ShapeError(f"patch 必须 ≥ 1，拿到 {patch}")
    layers = _block_indexes(shapes, "layers")
    noise = _block_indexes(shapes, "noise_refiner", required=False)
    context = _block_indexes(shapes, "context_refiner", required=False)

    x_in = _shape_of(shapes, "x_embedder.weight")
    out_in = _shape_of(shapes, "final_layer.linear.weight")
    if x_in[1] != out_in[0]:
        raise ShapeError(f"`x_embedder` 入维 {x_in[1]} ≠ `final_layer` 出维 {out_in[0]}"
                         "：首尾不属于同一结构")
    if x_in[1] % (patch * patch):
        raise ShapeError(f"`x_embedder` 入维 {x_in[1]} 不能按 patch={patch} 拆成 channels·patch²")

    features = x_in[0]
    head_dim = _shape_of(shapes, "layers.0.attention.q_norm.weight")[0]
    attn_out = _shape_of(shapes, "layers.0.attention.out.weight")[1]
    if attn_out % head_dim:
        raise ShapeError(f"attention 输出维 {attn_out} 不是 head_dim={head_dim} 的整数倍")
    heads = attn_out // head_dim
    qkv_out = _shape_of(shapes, "layers.0.attention.qkv.weight")[0]
    rest = qkv_out - attn_out
    if rest < 0 or rest % (2 * head_dim):
        raise ShapeError(f"qkv 输出维 {qkv_out} 减 q 的 {attn_out} 不能平分给 k/v"
                         f"（head_dim={head_dim}）")
    mlpdim = _shape_of(shapes, "layers.0.feed_forward.w1.weight")[0]

    mod_index = 1 if mod_silu else 0
    mod = _shape_of(shapes, f"layers.0.adaLN_modulation.{mod_index}.weight")
    if mod[0] != 4 * features:
        raise ShapeError(f"块调制输出维 {mod[0]} ≠ 4×hidden={4 * features}：不是 4 段调制")
    final_mod = _shape_of(shapes, "final_layer.adaLN_modulation.1.weight")
    if final_mod[1] != mod[1] or final_mod[0] != features:
        raise ShapeError(f"末层调制 {final_mod} 与块调制入维 {mod[1]} / hidden {features} 对不上")
    t_out = _shape_of(shapes, "t_embedder.mlp.2.weight")[0]
    if mod[1] != t_out:
        #: 参考实现里块调制的入维是 ``min(dim, 256)``、时间嵌入输出是 ``256``（z_image 档），
        #: 两者只有在 ``dim ≥ 256`` 时才相等 —— Z-Image 的 dim=3840 恒成立。dim<256 的自造配置
        #: 两条路宽度不同，直接跑会在 ``adaLN_modulation`` 上炸，这里提前说清楚。
        raise ShapeError(f"块调制入维 {mod[1]} ≠ 时间嵌入输出维 {t_out}：这份权重的时间嵌入与"
                         "块调制接不上（Z-Image 的 dim≥256 时两者都是 256）")

    for i in layers:
        for key, want in (("attention.out.weight", attn_out),
                          ("attention.qkv.weight", qkv_out),
                          ("feed_forward.w1.weight", mlpdim),
                          ("attention_norm1.weight", features)):
            got = _shape_of(shapes, f"layers.{i}.{key}")[0]
            if got != want:
                raise ShapeError(f"`layers.{i}.{key}` 宽度 {got} ≠ 块 0 的 {want}（层宽不一致）")

    for prefix, indexes in (("noise_refiner", noise), ("context_refiner", context)):
        for i in indexes:
            has_mod = f"{prefix}.{i}.adaLN_modulation.0.weight" in shapes
            if prefix == "noise_refiner" and not has_mod:
                raise ShapeError(f"`{prefix}.{i}` 缺调制（noise_refiner 必须带 adaLN）")
            if prefix == "context_refiner" and has_mod:
                raise ShapeError(f"`{prefix}.{i}` 多出调制（context_refiner 不带 adaLN）")

    has_pad = "x_pad_token" in shapes
    if has_pad and "cap_pad_token" not in shapes:
        raise ShapeError("有 `x_pad_token` 却没有 `cap_pad_token`：补位 token 必须成对")
    return LuminaShape(
        features=features,
        layers=len(layers),
        heads=heads,
        head_dim=head_dim,
        kvheads=rest // (2 * head_dim),
        mlpdim=mlpdim,
        patch=patch,
        channels=x_in[1] // (patch * patch),
        cap_dim=_shape_of(shapes, "cap_embedder.1.weight")[1],
        t_freq=_shape_of(shapes, "t_embedder.mlp.0.weight")[1],
        t_inner=_shape_of(shapes, "t_embedder.mlp.0.weight")[0],
        t_out=t_out,
        refiner_layers=len(noise),
        context_refiner_layers=len(context),
        mod_dim=mod[1],
        has_pad_tokens=has_pad,
        pad_tokens=pad_tokens if has_pad else 0,
        theta=theta,
        time_scale=time_scale,
        mod_silu=mod_silu,
    )


def weight_tensor_keys(shapes: dict[str, tuple[int, ...]]) -> set[str]:
    """权重里属于骨架的参数键（滤掉量化伴生键：反量化时被吃掉，不是参数）。"""
    return {k for k in shapes if not k.endswith(COMPANION_SUFFIXES)}


def _block_keys(head: str, *, modulated: bool, mod_index: int = 0) -> set[str]:
    keys = {
        head + "attention_norm1.weight", head + "attention_norm2.weight",
        head + "ffn_norm1.weight", head + "ffn_norm2.weight",
        head + "attention.qkv.weight", head + "attention.out.weight",
        head + "attention.q_norm.weight", head + "attention.k_norm.weight",
        head + "feed_forward.w1.weight", head + "feed_forward.w2.weight",
        head + "feed_forward.w3.weight",
    }
    if modulated:
        keys |= {f"{head}adaLN_modulation.{mod_index}.weight",
                 f"{head}adaLN_modulation.{mod_index}.bias"}
    return keys


def expected_keys(shape: LuminaShape) -> set[str]:
    """骨架**应该**产出的键集合（与 :func:`weight_tensor_keys` 双向对账用）。"""
    keys = {"x_embedder.weight", "x_embedder.bias",
            "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias",
            "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias",
            "cap_embedder.0.weight",
            "cap_embedder.1.weight", "cap_embedder.1.bias",
            "final_layer.linear.weight", "final_layer.linear.bias",
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.adaLN_modulation.1.bias"}
    if shape.has_pad_tokens:
        keys |= {"x_pad_token", "cap_pad_token"}
    index = shape.modulation_index
    for i in range(shape.layers):
        keys |= _block_keys(f"layers.{i}.", modulated=True, mod_index=index)
    for i in range(shape.refiner_layers):
        keys |= _block_keys(f"noise_refiner.{i}.", modulated=True, mod_index=index)
    for i in range(shape.context_refiner_layers):
        keys |= _block_keys(f"context_refiner.{i}.", modulated=False)
    return keys


def _pad_to_patch(x: Tensor, patch: int) -> Tensor:
    """右/下补零到 patch 整数倍（参考实现同样做），返回新张量、不改入参。"""
    h, w = x.shape[-2], x.shape[-1]
    ph, pw = (-h) % patch, (-w) % patch
    return x if not ph and not pw else F.pad(x, (0, pw, 0, ph))


class LuminaDiT(nn.Module):
    """Z-Image / Lumina-2 单流 DiT（结构由 :class:`LuminaShape` 驱动）。"""

    def __init__(self, shape: LuminaShape) -> None:
        super().__init__()
        self.shape_ = shape
        self.patch = shape.patch
        self.channels = shape.channels
        self.x_embedder = nn.Linear(shape.channels * shape.patch * shape.patch,
                                    shape.features)
        self.t_embedder = TimestepEmbedder(shape.t_freq, shape.t_inner, shape.t_out)
        self.cap_embedder = nn.Sequential(
            PlainRMSNorm(shape.cap_dim, shape.norm_eps),
            nn.Linear(shape.cap_dim, shape.features))
        self.context_refiner = nn.ModuleList([
            LuminaBlock(shape.features, shape.heads, shape.head_dim, shape.kvheads,
                        shape.mlpdim, shape.mod_dim, modulated=False,
                        norm_eps=shape.norm_eps, qk_eps=shape.qk_eps)
            for _ in range(shape.context_refiner_layers)])
        self.noise_refiner = nn.ModuleList([
            LuminaBlock(shape.features, shape.heads, shape.head_dim, shape.kvheads,
                        shape.mlpdim, shape.mod_dim, modulated=True,
                        mod_silu=shape.mod_silu, norm_eps=shape.norm_eps,
                        qk_eps=shape.qk_eps)
            for _ in range(shape.refiner_layers)])
        self.layers = nn.ModuleList([
            LuminaBlock(shape.features, shape.heads, shape.head_dim, shape.kvheads,
                        shape.mlpdim, shape.mod_dim, modulated=True,
                        mod_silu=shape.mod_silu, norm_eps=shape.norm_eps,
                        qk_eps=shape.qk_eps)
            for _ in range(shape.layers)])
        self.final_layer = FinalLayer(shape.features, shape.patch, shape.channels,
                                      shape.mod_dim)
        if shape.has_pad_tokens:
            #: 补位 token 是**可学习参数**（权重里就是 ``x_pad_token``/``cap_pad_token``）✓ ——
            #: ⚠️ 建它 ≠ 用它 ✗：参考实现建了这两个参数 ✓、但 ``pad_tokens_multiple=None`` 从不填 ✓。
            #: 所以判据用 ``has_pad_tokens``（有没有这个键 ✓），不是 ``pad_tokens``（填不填 ✓）。
            self.x_pad_token = nn.Parameter(torch.zeros(1, shape.features))
            self.cap_pad_token = nn.Parameter(torch.zeros(1, shape.features))

    # ------------------------------------------------------------------ 前向

    def _fill_pad(self, tokens: Tensor, pad_token: Tensor) -> Tensor:
        extra = (-tokens.shape[1]) % self.shape_.pad_tokens
        if not extra:
            return tokens
        filler = pad_token.to(tokens.dtype).expand(tokens.shape[0], extra, tokens.shape[2])
        return torch.cat((tokens, filler), dim=1)

    def _patchify(self, x: Tensor) -> tuple[Tensor, int, int]:
        """``(b,c,h,w)`` ⇒ ``(b, h·w, p²·c)``，每 token 内顺序是 ``(ph, pw, c)``。

        ⚠️ 不是 ``(c, ph, pw)``：参考实现是
        ``view(b,c,F,hh,ph,ww,pw).permute(0,2,3,5,4,6,1).flatten(4)`` —— 写反了键名全对、
        形状全对，结果只是「同一批权重读错了像素」。
        """
        patch = self.patch
        padded = _pad_to_patch(x, patch)
        b, c, h, w = padded.shape
        hh, ww = h // patch, w // patch
        tokens = (padded.reshape(b, c, hh, patch, ww, patch)
                  .permute(0, 2, 4, 3, 5, 1)          # (b, hh, ww, ph, pw, c)
                  .reshape(b, hh * ww, patch * patch * c))
        return tokens, hh, ww

    def _spatial_ids(self, start_t: float, hh: int, ww: int,
                     b: int, device: torch.device) -> Tensor:
        ids = torch.zeros(hh, ww, 3, device=device, dtype=torch.float32)
        ids[..., 0] = start_t
        ids[..., 1] = torch.arange(hh, device=device, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(ww, device=device, dtype=torch.float32)[None, :]
        return ids.reshape(1, hh * ww, 3).repeat(b, 1, 1)

    def forward(self, x: Tensor, timesteps: Tensor, context: Tensor) -> Tensor:
        """``x (b,c,h,w)`` ＋ ``timesteps (b,)`` ＋ ``context (b,L,cap_dim)`` ⇒ 速度 ``(b,c,h,w)``。"""
        if x.ndim != 4:
            raise ShapeError(f"只支持 4 维 latent（图像），拿到 {x.ndim} 维")
        if x.shape[1] != self.channels:
            raise ShapeError(f"latent 通道 {x.shape[1]} ≠ 结构里的 {self.channels}")
        if context.ndim != 3 or context.shape[-1] != self.shape_.cap_dim:
            raise ShapeError(f"文本条件形状 {tuple(context.shape)} 与结构里的 "
                             f"(b, L, {self.shape_.cap_dim}) 不符（拿错 TE 了）")
        b, _, h0, w0 = x.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != b:
            raise ShapeError(f"timesteps 形状 {tuple(timesteps.shape)} 与 batch={b} 不符")

        #: ⚠️ 潜变量**故意是 f32**（``torch.randn`` 的默认 ✓，采样全程也保持 f32 ✓ 更准 ✓），
        #: 而权重是 bf16/fp16 ✓ ⇒ **必须在这里对齐到权重精度** ✓✗：不做的话 `x_embedder` 当场报
        #: ``mat1 and mat2 must have the same dtype, but got Float and BFloat16`` ✓ ——
        #: 报错指向 ``nn.Linear`` ✓，读起来像「权重坏了」✓，其实是「精度没对齐」✓（参考实现也是在
        #: 模型内 cast ✓，不是在采样侧把潜变量降精度 ✗）。
        x = x.to(self.x_embedder.weight.dtype)

        #: ① 先取反再乘 time_scale —— 参考：``t = 1 - timesteps``、``t_embedder(t * time_scale)``。
        tvec = self.t_embedder((1.0 - timesteps) * self.shape_.time_scale).to(x.dtype)

        cap = self.cap_embedder(context.to(x.dtype))
        if self.shape_.pad_tokens:
            cap = self._fill_pad(cap, self.cap_pad_token)
        cap_len = cap.shape[1]
        cap_pos = torch.zeros(b, cap_len, 3, device=x.device, dtype=torch.float32)
        cap_pos[..., 0] = torch.arange(cap_len, device=x.device, dtype=torch.float32) + 1.0
        cap_freqs = rope_freqs(cap_pos, self.shape_.axes_dim, self.shape_.theta).unsqueeze(1)
        for block in self.context_refiner:
            cap = block(cap, cap_freqs, None)

        tokens, hh, ww = self._patchify(x)
        img = self.x_embedder(tokens)
        img_pos = self._spatial_ids(cap_len + 1.0, hh, ww, b, x.device)
        if self.shape_.pad_tokens:
            #: 参考实现是 ``x_pos_ids = F.pad(x_pos_ids, (0,0,0,pad_extra))`` —— 补位 token 拿
            #: **全零**位置，不是顺着网格往后延（顺着延会让 pad 参与旋转、与参考不一致）。
            extra = (-img.shape[1]) % self.shape_.pad_tokens
            if extra:
                img_pos = F.pad(img_pos, (0, 0, 0, extra))
            img = self._fill_pad(img, self.x_pad_token)
        img_freqs = rope_freqs(img_pos, self.shape_.axes_dim, self.shape_.theta).unsqueeze(1)
        for block in self.noise_refiner:
            img = block(img, img_freqs, tvec)

        pos = torch.cat((cap_pos, img_pos), dim=1)
        freqs = rope_freqs(pos, self.shape_.axes_dim, self.shape_.theta).unsqueeze(1)
        seq = torch.cat((cap, img), dim=1)
        for block in self.layers:
            seq = block(seq, freqs, tvec)

        out = self.final_layer(seq, tvec)[:, cap_len:cap_len + hh * ww, :]
        out = out.reshape(b, hh, ww, self.patch, self.patch, self.channels)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(b, self.channels, hh * self.patch,
                                                    ww * self.patch)
        #: ② 参考实现最后一步是 ``return -img``（预测的是「负速度」）。
        return -out[:, :, :h0, :w0]


def build_from_shapes(shapes: dict[str, tuple[int, ...]], *, patch: int = 2,
                      dtype: torch.dtype = torch.float32) -> LuminaDiT:
    """按权重**形状**建出骨架（不读权重内容）；``dtype`` 只影响初始化的参数类型。"""
    shape = infer_shape(shapes, patch=patch)
    model = LuminaDiT(shape)
    return model if dtype == torch.float32 else model.to(dtype)
