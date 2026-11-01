"""Qwen-Image **2512 / Edit 2511** 双流 MMDiT 前向（族 ``qwen_image`` ✓）。

权威口径：ComfyUI ``comfy/ldm/qwen_image/model.py`` ✓（2026-10-31 逐行核过 ✓）。
与 ``dit_qwen21``（2.1 单流 649 键 ✓）**不是同一个东西** ✓✗ —— 盘上 2512（1933 键 ✓）与
Edit 2511（3614 键 ✓）走的是这一份 ✓。

## 四条「形状对、能跑、结果却与提示词无关」的错法

① **单流化**（拿 2.1 的写法套 ✓✗）：这一族是**双流** —— 文本与图像各一套调制
   （``img_mod``/``txt_mod`` 各 6 组 ✓）**与各一套 MLP** ✓。合并成单流 ⇒ 文本被图像的时间
   调制拖着走 ✓，提示词跟随明显变差 ✓、不报错 ✓。
② **调制顺序写反**：6 组先 ``chunk(2)`` 成两组、每组 ``chunk(3)`` = **(shift, scale, gate)** ✓
   （``addcmul(shift, x, 1 + scale)`` ✓）。写成 (scale, shift) ⇒ 图能出 ✓，但亮度随步数漂 ✓✗
   （adaLN 的 shift 是**加性偏置** ✓，不是乘性 ✓）。
③ **参考图走通道拼接** ✗✗：Edit 的参考图是 **token 串联** ✓，位置号 ``t`` 分量取 ``0``（目标）/
   ``1, 2, …``（第几张参考 ✓）。拼通道 ⇒ 装载时 ``img_in`` 就报形状错 ✓（会自己报 ✓，
   但报在那儿指不到「Edit 与 t2i 是两种拼法」✓）。
④ **参考图的 timestep 用错行**：``index_timestep_zero``（**Edit 2511 就这档** ✓，由权重里的
   ``__index_timestep_zero__`` 自述 ✓）下 ``timestep = cat([t, 0])`` ✓ ⇒ 调制张量**两行** ✓：
   目标 token 用行 0（采样 t ✓）、参考 token 用行 1（**t=0** ✓）。全用采样 t ⇒ 参考图被当成
   「要一起去噪的图」✓ ⇒ 编辑强度虚高 ✓、**且不报错** ✓✗。

## 两处会自己报错、但报错位置指不到原因的

* 输出必须**丢掉参考 token**（``hidden[:, :num_embeds]`` ✓）再 unpatchify ✓ —— 忘了丢 ⇒
  reshape 报形状错 ✓；
* ``in_channels = 64`` 是**拍平后的宽度** ✓（= 16ch × 2×2 ✓），不是 latent 通道数 ✓✗。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dit_generic import ShapeError, apply_rope, rope_freqs, timestep_embedding

__all__ = [
    "PATCH", "AXES_DIM", "ROPE_THETA", "T_FREQ", "T_SCALE", "NORM_EPS",
    "REF_INDEX", "REF_INDEX_TZERO", "REF_NEGATIVE", "REF_METHODS", "EDIT_SENTINEL",
    "QwenImageShape", "infer_shape", "expected_keys", "build", "QwenImageDiT",
]

#: 潜变量侧 patch 边长 ✓（固定 2×2 ✓ —— 盘上没有这个键 ✓，猜 1 或 4 只会得到隐式的 token 数错 ✓✗）。
PATCH = 2
#: 三轴 rope 分维 ✓（16 + 56 + 56 = 128 = head_dim ✓）。与 2.1 同值只是巧合 ✓。
AXES_DIM: tuple[int, int, int] = (16, 56, 56)
ROPE_THETA = 10000.0
#: 时间正弦 256 维 ✓（``Timesteps(num_channels=256, flip_sin_to_cos=True, scale=1000)`` ✓）。
T_FREQ = 256
T_SCALE = 1000.0
NORM_EPS = 1e-6

#: 参考图（Edit）的三种口径 ✓（名字照抄参考实现 ✓，别自创 ✓）。
#: ⚠️ 参考实现里还有第四种（``offset``：按 h/w 偏移把参考图**排在目标图右侧/下方** ✓）——
#: 本层**不支持** ✓，传进来会由 :meth:`QwenImageShape.__post_init__` 当场拒 ✗
#: （**显式拒 > 按 index 猜** ✗✗：口径猜错不会报错，只会出一张「像但不对」的图 ✓✗）。
REF_INDEX = "index"
REF_INDEX_TZERO = "index_timestep_zero"
#: ``negative_index``：与 ``index`` 只差**序号方向**（-1、-2…✓），参考实现里是给「参考图排前面」的
#: 权重用的 ✓（2026-10-31 补 ✗ —— 原先只认前两种 ⇒ 这类权重只能靠显式报错挡下 ✓）。
REF_NEGATIVE = "negative_index"
REF_METHODS: tuple[str, ...] = (REF_INDEX, REF_INDEX_TZERO, REF_NEGATIVE)
#: Edit 2511 在权重里**自述**口径 ✓（``model_detection.py`` 就是读这根键 ✓）⇒ 不靠文件名猜 ✓。
EDIT_SENTINEL = "__index_timestep_zero__"


@dataclass(frozen=True)
class QwenImageShape:
    """从**权重形状**读出来的架构事实 ✓（族判定也只认这些 ✓）。"""

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
    ref_method: str = REF_INDEX
    ref_index_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.heads * self.head_dim != self.features:
            raise ShapeError(
                f"heads×head_dim = {self.heads}×{self.head_dim} ≠ features {self.features} ✗")
        if sum(self.axes_dim) != self.head_dim:
            raise ShapeError(
                f"axes_dim {self.axes_dim} 之和 {sum(self.axes_dim)} ≠ head_dim "
                f"{self.head_dim} ✗（三轴分维必须正好铺满一个头 ✓）")
        if self.ref_method not in REF_METHODS:
            raise ShapeError(f"未知 ref_method {self.ref_method!r}；可用：{list(REF_METHODS)}")

    @property
    def ref_timestep_zero(self) -> bool:
        """参考 token 是否走 **t=0** 那行调制 ✓（Edit 2511 = 是 ✓）。"""
        return self.ref_method == REF_INDEX_TZERO


def _dims(shapes: Mapping[str, Sequence[int]], name: str) -> tuple[int, ...]:
    if name not in shapes:
        raise ShapeError(f"权重里没有 {name} ✗ ⇒ 不是 qwen_image 族 ✓（族判定只认形状 ✓）")
    return tuple(int(v) for v in shapes[name])


def _block_dims(shapes: Mapping[str, Sequence[int]], name: str) -> tuple[int, ...]:
    """块内键在盘上是 ``transformer_blocks.0.<name>`` ✓；测试里传**已剥前缀**的小表也认 ✓。

    ⚠️ 只写不带前缀的名字 ✗ 会让真权重在这里报「没有这根键」✓（错的是查法、不是权重 ✓）。
    """
    prefixed = f"transformer_blocks.0.{name}"
    if prefixed in shapes:
        return _dims(shapes, prefixed)
    return _dims(shapes, name)


def _block_index(name: str) -> int | None:
    if not name.startswith("transformer_blocks."):
        return None
    head = name.split(".", 2)
    return int(head[1]) if len(head) > 1 and head[1].isdigit() else None


def infer_shape(shapes: Mapping[str, Sequence[int]], *,
                heads: int | None = None, ref_method: str | None = None,
                axes_dim: Sequence[int] | None = None,
                ref_index_scale: float = 1.0) -> QwenImageShape:
    """按权重形状推架构 ✓；**推不出或有矛盾就抛** ✓（不带默认值硬上 ✗）。

    ``heads`` 读不到时用 ``features / head_dim`` ✓（``attn.norm_q.weight`` 就是 head_dim ✓ ——
    这也是 ``dit_family`` 里那条 ``unknown: heads`` 的答案 ✓）。
    """
    features, patch_in = _dims(shapes, "img_in.weight")
    spans = PATCH * PATCH
    if patch_in % spans:
        raise ShapeError(
            f"img_in 输入宽 {patch_in} 不是 {spans} 的整数倍 ✗ ⇒ latent 通道数读不出来 ✓"
            f"（这一族的 img_in 吃的是**拍平后**的宽 = 通道 × {spans} ✓）")
    channels = patch_in // spans
    proj_out, features_out = _dims(shapes, "proj_out.weight")
    if (proj_out, features_out) != (channels * spans, features):
        raise ShapeError(
            f"proj_out 形状 ({proj_out}, {features_out}) ✗ 与 img_in 推出的 "
            f"({channels * spans}, {features}) ✓ 不符 —— 输入输出 patch 口径必须一致 ✓")

    head_dim = _block_dims(shapes, "attn.norm_q.weight")[0]
    if head_dim <= 0 or features % head_dim:
        raise ShapeError(f"head_dim {head_dim} 不整除 features {features} ✗")
    heads = int(heads or features // head_dim)

    context_dim = _dims(shapes, "txt_norm.weight")[0]
    txt_in, txt_out = _dims(shapes, "txt_in.weight")
    if (txt_in, txt_out) != (features, context_dim):
        raise ShapeError(
            f"txt_in 形状 ({txt_in}, {txt_out}) ✗ 与 (features, txt_norm 宽) = "
            f"({features}, {context_dim}) ✓ 不符")

    hidden = _block_dims(shapes, "img_mlp.net.0.proj.weight")[0]
    mlp_out, mlp_in = _block_dims(shapes, "img_mlp.net.2.weight")
    if (mlp_out, mlp_in) != (features, hidden):
        raise ShapeError(
            f"img_mlp.net.2 形状 ({mlp_out}, {mlp_in}) ✗ 与 ({features}, {hidden}) ✓ 不符")

    #: ⚠️ 6 组调制是**这一族**的指纹 ✓：2.1 是 4 组 + 单流 ✓，把它的权重当这一族装进来会
    #: 在这里当场拒绝 ✓（形状读得到、语义不对 ✓✗ —— 最省事的判别就是这一行 ✓）。
    mod_out, mod_in = _block_dims(shapes, "img_mod.1.weight")
    if mod_out != 6 * features or mod_in != features:
        raise ShapeError(
            f"img_mod 形状 ({mod_out}, {mod_in}) ✗ ≠ ({6 * features}, {features}) "
            f"⇒ 不是 qwen_image 族 ✓（双流 MMDiT 的图片调制是 6 组 ✓）")

    t_freq = _dims(shapes, "time_text_embed.timestep_embedder.linear_1.weight")[1]
    t_inner = _dims(shapes, "time_text_embed.timestep_embedder.linear_2.weight")[1]
    if t_freq != T_FREQ:
        raise ShapeError(f"时间正弦 {t_freq} 维 ✗ ≠ {T_FREQ} ✓（这一族固定 256 ✓）")

    layers = len([i for i in {_block_index(n) for n in shapes} if i is not None])
    if layers <= 0:
        raise ShapeError("权重里没有任何 transformer_blocks.N ✗")

    method = str(ref_method or (REF_INDEX_TZERO if EDIT_SENTINEL in shapes else REF_INDEX))
    return QwenImageShape(
        features=features, layers=layers, heads=heads, head_dim=head_dim, hidden=hidden,
        channels=channels, context_dim=context_dim, t_freq=t_freq, t_inner=t_inner,
        axes_dim=tuple(int(v) for v in (axes_dim or AXES_DIM)),
        ref_method=method, ref_index_scale=float(ref_index_scale))


def expected_keys(shape: QwenImageShape) -> set[str]:
    """这一族**应该**有哪些键 ✓（两条流各 6 组调制、各一套 MLP 都在里面 ✓）。"""
    keys = {
        "img_in.weight", "img_in.bias",
        "txt_in.weight", "txt_in.bias", "txt_norm.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "time_text_embed.timestep_embedder.linear_1.bias",
        "time_text_embed.timestep_embedder.linear_2.weight",
        "time_text_embed.timestep_embedder.linear_2.bias",
        "norm_out.linear.weight", "norm_out.linear.bias",
        "proj_out.weight", "proj_out.bias",
    }
    if shape.ref_timestep_zero:
        keys.add(EDIT_SENTINEL)
    for i in range(shape.layers):
        p = f"transformer_blocks.{i}."
        for prefix in ("img_mod.1", "txt_mod.1"):
            keys |= {f"{p}{prefix}.weight", f"{p}{prefix}.bias"}
        for name in ("to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj",
                     "add_v_proj", "to_add_out"):
            keys |= {f"{p}attn.{name}.weight", f"{p}attn.{name}.bias"}
        for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            keys.add(f"{p}attn.{name}.weight")
        for stream in ("img_mlp", "txt_mlp"):
            keys |= {f"{p}{stream}.net.0.proj.weight", f"{p}{stream}.net.0.proj.bias",
                     f"{p}{stream}.net.2.weight", f"{p}{stream}.net.2.bias"}
    return keys


# ------------------------------------------------------------------ 组件


class _RMSNorm(nn.Module):
    """**非**零中心 RMSNorm（权重直接乘 ✓、初值 1 ✓），``txt_norm`` 与四处 per-head norm 都用它。

    ⚠️ 与 ``dit_qwen21._ZeroCenteredRMSNorm``（存 ``scale − 1`` ✓）**不是一回事** ✓✗：
    混用 ⇒ 文本特征被整体平移一个常数 ✓，形状全对、不报错 ✓✗（出图偏灰/偏曝 ✓）。
    """

    def __init__(self, dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        return _rms_norm(x, self.weight, self.eps)


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """fp32 里算完再降回 ``x.dtype`` ✓（bf16 直接算 ``rsqrt`` 会掉精度 ✗ —— 参考实现同样升精度 ✓）。"""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


class _Weight(nn.Module):
    """只有一根 ``weight`` 的壳 —— 唯一目的是让参数名**带 ``.weight`` 后缀** ✓。

    ⚠️ 写成 ``self.norm_q = nn.Parameter(...)`` 参数名是 ``attn.norm_q`` ✗，而权重键是
    ``attn.norm_q.weight`` ✓ ⇒ 加载时这一格**静默落空** ✓✗（q/k 没归一，形状全对、结果全偏 ✓）。
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))


class _GeluProj(nn.Module):
    """``linear`` 之后接 **tanh 近似** 的 GELU ✓（``FeedForward.net.0`` ✓，不是 SiLU ✗）。"""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class _FeedForward(nn.Module):
    """``Linear → GELU(tanh) → Linear`` ✓（两条流各一套 ✓ —— 共用一套就是单流化 ✗）。"""

    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        #: ``net`` 是 ``ModuleList`` ⇒ 键是 ``net.0.proj`` / ``net.2`` ✓（层号要留空位 ✓）
        self.net = nn.ModuleList([_GeluProj(dim, inner), nn.Dropout(0.0),
                                  nn.Linear(inner, dim, bias=True)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.net:
            x = layer(x)
        return x


class _Attention(nn.Module):
    """**联合注意力**：文本与图像拼在序列维一次算 ✓，输出按 ``seq_txt`` 切回两条流 ✓。

    ⚠️ 三个「形状对、结果错」的坑都在这里：
    ① 拼在**序列维**（``dim=2``，此时已是 BHND ✓）而不是特征维 ✗；
    ② ``norm_added_*`` 是**文本流**的 q/k 归一 ✓，拿 ``norm_q`` 代替不会报错（宽度同为 head_dim ✓✗）；
    ③ ``to_out`` 是 ``ModuleList([Linear, Dropout])`` ⇒ 键是 ``to_out.0`` ✓，用单个 ``Linear``
       会让 ``to_out.0.weight`` 落空 ✓✗（注意力输出白算 ✓）。
    """

    def __init__(self, shape: "QwenImageShape") -> None:
        super().__init__()
        dim, heads, head_dim = shape.features, shape.heads, shape.head_dim
        self.heads, self.head_dim = heads, head_dim
        self.eps = float(shape.eps)
        inner = heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=True)
        self.to_k = nn.Linear(dim, inner, bias=True)
        self.to_v = nn.Linear(dim, inner, bias=True)
        self.add_q_proj = nn.Linear(dim, inner, bias=True)
        self.add_k_proj = nn.Linear(dim, inner, bias=True)
        self.add_v_proj = nn.Linear(dim, inner, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim, bias=True), nn.Dropout(0.0)])
        self.to_add_out = nn.Linear(inner, dim, bias=True)
        #: 四处都是 **per-head**（宽 = head_dim ✓），普通 RMSNorm ✓
        self.norm_q = _Weight(head_dim)
        self.norm_k = _Weight(head_dim)
        self.norm_added_q = _Weight(head_dim)
        self.norm_added_k = _Weight(head_dim)

    def _split(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, img: Tensor, txt: Tensor, freqs: Tensor,
                mask: Tensor | None) -> tuple[Tensor, Tensor]:
        b, si, _ = img.shape
        st = txt.shape[1]
        q = torch.cat((self._split(self.add_q_proj(txt)), self._split(self.to_q(img))), dim=2)
        k = torch.cat((self._split(self.add_k_proj(txt)), self._split(self.to_k(img))), dim=2)
        v = torch.cat((self._split(self.add_v_proj(txt)), self._split(self.to_v(img))), dim=2)
        q_t, q_i = q[:, :, :st], q[:, :, st:]
        k_t, k_i = k[:, :, :st], k[:, :, st:]
        #: ⚠️ 归一按**各自那条流**的权重 ✓；先归一、后 RoPE ✓（参考实现同序 ✓）
        q = torch.cat((_rms_norm(q_t, self.norm_added_q.weight, self.eps),
                       _rms_norm(q_i, self.norm_q.weight, self.eps)), dim=2)
        k = torch.cat((_rms_norm(k_t, self.norm_added_k.weight, self.eps),
                       _rms_norm(k_i, self.norm_k.weight, self.eps)), dim=2)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        joint = out.transpose(1, 2).reshape(b, st + si, self.heads * self.head_dim)
        img_out = self.to_out[0](joint[:, st:])
        return self.to_out[1](img_out), self.to_add_out(joint[:, :st])


def _modulate(x: Tensor, shift: Tensor, scale: Tensor, gate: Tensor,
              tz: int | None) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
    """``(1 + scale)·x + shift`` ⇒ ``(调制后的 x, gate)`` ✓。

    ⚠️ 是 ``addcmul(shift, x, 1 + scale)`` ✓ —— shift 是**加性偏置** ✗ 不是乘性 ✓。
    写反（先乘后加、或把 shift 当 scale）⇒ 图能出、亮度随步数漂 ✓✗。

    ``tz``（``timestep_zero_index``）非空时调制张量有**两行** ✓：行 0 给目标 token（采样 t ✓）、
    行 1 给参考 token（**t=0** ✓）。此时返回的 gate 也是两行 ✓。
    """
    if tz is None:
        return torch.addcmul(shift.unsqueeze(1), x, 1 + scale.unsqueeze(1)), gate.unsqueeze(1)
    half = shift.size(0) // 2
    s_t, s_z = shift[:half], shift[half:]
    c_t, c_z = scale[:half], scale[half:]
    g_t, g_z = gate[:half], gate[half:]
    reg = torch.addcmul(s_t.unsqueeze(1), x[:, :tz], 1 + c_t.unsqueeze(1))
    zero = torch.addcmul(s_z.unsqueeze(1), x[:, tz:], 1 + c_z.unsqueeze(1))
    return torch.cat((reg, zero), dim=1), (g_t.unsqueeze(1), g_z.unsqueeze(1))


def _gated(y: Tensor, x: Tensor, gate: Tensor | tuple[Tensor, Tensor],
           tz: int | None) -> Tensor:
    """``x + gate·y`` ✓（前缀段用 t=0 的 gate、目标段用采样 t 的 gate ✓）。"""
    if tz is None:
        return torch.addcmul(x, gate, y)
    g_t, g_z = gate
    return x + torch.cat((y[:, :tz] * g_t, y[:, tz:] * g_z), dim=1)


class _Block(nn.Module):
    """双流块：两条流**各一套** 6 组调制 ✓、各一套 MLP ✓，注意力是联合的 ✓。

    顺序（与参考实现逐条对应 ✓）：
    ``img += gate1·attn(调制(norm1(img)), 调制(norm1(txt)))`` →
    ``txt += gate1·attn_out`` → ``img += gate2·mlp(调制(norm2(img)))`` →
    ``txt += gate2·mlp(调制(norm2(txt)))``。
    """

    def __init__(self, shape: "QwenImageShape") -> None:
        super().__init__()
        d = shape.features
        #: ⚠️ ``elementwise_affine=False``：这四个 LayerNorm **没有参数** ✓（权重里也确实没有）
        self.img_norm1 = nn.LayerNorm(d, eps=shape.eps, elementwise_affine=False)
        self.img_norm2 = nn.LayerNorm(d, eps=shape.eps, elementwise_affine=False)
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d, bias=True))
        self.img_mlp = _FeedForward(d, shape.hidden)
        self.txt_norm1 = nn.LayerNorm(d, eps=shape.eps, elementwise_affine=False)
        self.txt_norm2 = nn.LayerNorm(d, eps=shape.eps, elementwise_affine=False)
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d, bias=True))
        self.txt_mlp = _FeedForward(d, shape.hidden)
        self.attn = _Attention(shape)

    @staticmethod
    def _six(mod: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
        """``6d`` ⇒ 先 ``chunk(2)`` 成两组、每组再 ``chunk(3)`` = (shift, scale, gate) ✓。"""
        first, second = mod.chunk(2, dim=-1)
        return first.chunk(3, dim=-1), second.chunk(3, dim=-1)

    def forward(self, img: Tensor, txt: Tensor, temb: Tensor, temb_txt: Tensor,
                freqs: Tensor, mask: Tensor | None,
                tz: int | None) -> tuple[Tensor, Tensor]:
        #: ⚠️ 调制是**逐块**的（每块一套 ``img_mod``/``txt_mod`` 权重 ✓）⇒ 必须在块内投影 ✓，
        #: 拿到块外算一次再传进来 ⇒ 所有块共用第一块的调制 ✓✗（形状全对、提示词跟随塌掉 ✓）。
        img1, img2 = self._six(self.img_mod(temb))
        txt1, txt2 = self._six(self.txt_mod(temb_txt))

        x_in, g1 = _modulate(self.img_norm1(img), *img1, tz)
        t_in, tg1 = _modulate(self.txt_norm1(txt), *txt1, None)
        attn_img, attn_txt = self.attn(x_in, t_in, freqs, mask)
        img = _gated(attn_img, img, g1, tz)
        txt = txt + tg1 * attn_txt

        x_in, g2 = _modulate(self.img_norm2(img), *img2, tz)
        img = _gated(self.img_mlp(x_in), img, g2, tz)
        t_in, tg2 = _modulate(self.txt_norm2(txt), *txt2, None)
        txt = torch.addcmul(txt, tg2, self.txt_mlp(t_in))
        return img, txt


class _LastLayer(nn.Module):
    """``norm(x)·(1 + (1+scale)) + shift``（``addcmul`` 同款 ✓）⇒ 线性到 ``channels·patch²``。

    ⚠️ ``linear`` 出 ``2·d`` 后 ``chunk(2)`` 得 **(scale, shift)** ✓ —— 顺序反了不报错 ✓✗。
    """

    def __init__(self, shape: "QwenImageShape") -> None:
        super().__init__()
        d = shape.features
        self.norm = nn.LayerNorm(d, eps=shape.eps, elementwise_affine=False)
        self.linear = nn.Linear(d, 2 * d, bias=True)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        scale, shift = self.linear(F.silu(temb)).chunk(2, dim=-1)
        return torch.addcmul(shift.unsqueeze(1), self.norm(x), (1 + scale).unsqueeze(1))


# ------------------------------------------------------------------ 主干


class _TimestepProj(nn.Module):
    """``正弦(256) → Linear → SiLU → Linear`` ✓（键名是 ``linear_1``/``linear_2`` ✓）。"""

    def __init__(self, freq: int, inner: int) -> None:
        super().__init__()
        self.freq = int(freq)
        self.linear_1 = nn.Linear(freq, inner, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(inner, inner, bias=True)

    def forward(self, t: Tensor) -> Tensor:
        #: ⚠️ 这一族的时间口径是 **1000·σ** ✓（``Timesteps(scale=1000)`` ✓ 且 comfy 传进来的就是 σ、
        #: 不翻转 ✓）。与 Z-Image（lumina ✓）的 **1000·(1−σ)** ✓ 正好相反 ✓✗ —— 抄错那一族
        #: ⇒ 形状全对、图能出 ✓，但等于「去噪方向反着走」✓（噪点越采越多/糊成一团 ✓）。
        emb = timestep_embedding(t.reshape(-1).float(), self.freq, time_factor=T_SCALE)
        return self.linear_2(self.act(self.linear_1(emb.to(self.linear_1.weight.dtype))))


class QwenImageDiT(nn.Module):
    """双流 MMDiT 主体 ✓（2512 t2i 与 Edit 2511 同一份 ✓，差别只在 ``ref_latents`` ✓）。

    输入（与其它族同口径 ✓，**不 patchify** ✓ 由本模块内部按 :data:`PATCH` 拍平 ✓）：
    ``latents (b, channels, h, w)``、``timesteps (b,)``（**σ 域 [0,1]** ✓）、
    ``context (b, n_txt, context_dim)``（Qwen2.5-VL 最后一层隐状态 ✓，**不做末层归一** ✓）。
    """

    def __init__(self, shape: QwenImageShape) -> None:
        super().__init__()
        self.shape = shape
        self.spans = PATCH * PATCH
        width = shape.channels * self.spans
        self.img_in = nn.Linear(width, shape.features, bias=True)
        #: ⚠️ ``txt_norm`` 是 **RMSNorm**（非零中心 ✓）且作用在 **context_dim** 上 ✓，
        #: 不是 features ✗ —— 用 features 宽度会装载报形状错 ✓（这条会自己报 ✓）。
        self.txt_norm = _RMSNorm(shape.context_dim, shape.eps)
        self.txt_in = nn.Linear(shape.context_dim, shape.features, bias=True)
        self.time_text_embed = nn.ModuleDict(
            {"timestep_embedder": _TimestepProj(shape.t_freq, shape.features)})
        self.transformer_blocks = nn.ModuleList([_Block(shape) for _ in range(shape.layers)])
        self.norm_out = _LastLayer(shape)
        self.proj_out = nn.Linear(shape.features, width, bias=True)
        if shape.ref_timestep_zero:
            #: 空张量哨兵 ✓（盘上是 0 元素的键 ✓）：存在的意义就是**自述** Edit 口径 ✓
            self.register_buffer(EDIT_SENTINEL, torch.tensor([]))

    # ---------------------------------------------------------- 位置与 patch

    def _patch(self, x: Tensor) -> tuple[Tensor, tuple[int, int, int, int]]:
        b, c, h, w = x.shape
        p = PATCH
        hh, ww = -(-h // p), -(-w // p)
        if (hh * p, ww * p) != (h, w):
            x = F.pad(x, (0, ww * p - w, 0, hh * p - h))
        tokens = x.reshape(b, c, hh, p, ww, p).permute(0, 2, 4, 1, 3, 5).reshape(b, hh * ww, c * p * p)
        return tokens, (hh, ww, h, w)

    def _unpatch(self, tokens: Tensor, hh: int, ww: int, c: int, h: int,
                 w: int) -> Tensor:
        p = PATCH
        b = tokens.shape[0]
        x = tokens.reshape(b, hh, ww, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, hh * p, ww * p)
        #: ⚠️ 回裁到**原始** h/w ✓（补齐的边不能漏到输出 ✓，否则 VAE 解码尺寸对不上 ✓）
        return x[:, :, :h, :w]

    @staticmethod
    def _img_ids(hh: int, ww: int, b: int, index: float, device,
                 dtype) -> Tensor:
        """图像三轴位置号 ✓：``t`` 分量 = **参考图序号**（目标 0 ✓、第 k 张 = k ✓），h/w 居中 ✓。"""
        grid = torch.zeros((hh, ww, 3), device=device, dtype=torch.float32)
        grid[:, :, 0] = float(index)
        #: ⚠️ h 沿**行**铺、w 沿**列**铺 ⇒ 各自 unsqueeze 到 (hh,1)/(1,ww) ✓；
        #: 直接写 (hh,) 在 hh≠ww 时会被广播判成尺寸冲突 ✓（hh==ww 的方形尺寸下**不会**露头 ✓✗）。
        grid[:, :, 1] = (torch.arange(hh, device=device).unsqueeze(1) - (hh // 2)).float()
        grid[:, :, 2] = (torch.arange(ww, device=device).unsqueeze(0) - (ww // 2)).float()
        return grid.reshape(1, hh * ww, 3).expand(b, -1, -1)

    @staticmethod
    def _txt_ids(n_txt: int, hh: int, ww: int, b: int, device: torch.device) -> Tensor:
        """文本三轴**同值** ✓、起点 = ``max(hh, ww)//2`` 附近的整数 ✓（不是 0 ✗、也不居中 ✗）。"""
        start = round(max(hh // 2, ww // 2))
        return torch.arange(start, start + n_txt, device=device).reshape(1, -1, 1).expand(b, -1, 3)

    # ---------------------------------------------------------- 前向

    def forward(self, latents: Tensor, timesteps: Tensor, context: Tensor, *,
                ref_latents: Sequence[Tensor] | None = None,
                attention_mask: Tensor | None = None,
                ref_method: str | None = None) -> Tensor:
        """``ref_latents`` 是 **token 串联** ✓（``cat(dim=1)`` ✓），不是通道拼接 ✗。"""
        shape = self.shape
        b = latents.shape[0]
        t = timesteps.reshape(1) if timesteps.dim() == 0 else timesteps.reshape(-1)
        if t.numel() == 1 and b > 1:
            t = t.expand(b)

        method = str(ref_method or shape.ref_method)
        if method not in REF_METHODS:
            raise ShapeError(f"未知参考图口径 {method!r}；可用：{list(REF_METHODS)}")

        hidden, (hh, ww, h0, w0) = self._patch(latents)
        n_target = hidden.shape[1]
        ids = [self._txt_ids(context.shape[1], hh, ww, b, latents.device)]
        ids.append(self._img_ids(hh, ww, b, 0, latents.device, latents.dtype))

        tz: int | None = None
        if ref_latents:
            #: ⚠️ 序号**方向**与**缩放**都要照口径走 ✗（2026-10-31 补 ✓）：
            #: ``negative_index`` ⇒ 1、2… 反过来是 -1、-2… ✓；``ref_index_scale`` 原先只存不用 ✗✗
            #: ⇒ 权重自述 scale≠1 时位置号会悄悄错 ✓✗（默认 1.0 ✓ 所以只在特殊权重上露头 ✓）。
            step = -1 if method == REF_NEGATIVE else 1
            index = 0
            for ref in ref_latents:
                index += step
                piece, (rh, rw, _, _) = self._patch(ref)
                if piece.shape[0] != b:
                    raise ShapeError(
                        f"参考图 batch {piece.shape[0]} ✗ ≠ 目标 batch {b} ✓（Edit 不支持混合批 ✓）")
                hidden = torch.cat((hidden, piece), dim=1)
                ids.append(self._img_ids(rh, rw, b, float(index) * float(shape.ref_index_scale),
                                         latents.device, latents.dtype))
            if method == REF_INDEX_TZERO:
                #: 目标 token 用行 0（采样 σ）、参考 token 用行 1（**σ=0** ✓）
                tz = n_target
                t = torch.cat((t, t * 0), dim=0)

        freqs = rope_freqs(torch.cat(ids, dim=1), list(shape.axes_dim), shape.theta).unsqueeze(1)

        mask = self._mask(attention_mask, b, context.shape[1], hidden.shape[1], latents.dtype)

        img = self.img_in(hidden)
        txt = self.txt_in(self.txt_norm(context))
        temb = self.time_text_embed["timestep_embedder"](t)
        #: ⚠️ 文本流**不用**那第二行（σ=0）✓ —— 用错行不会报错 ✓✗，只会让参考图的「无噪」假设漏到文本上 ✓
        temb_txt = temb.chunk(2, dim=0)[0] if tz is not None else temb

        for block in self.transformer_blocks:
            img, txt = block(img, txt, temb, temb_txt, freqs, mask, tz)

        if tz is not None:
            temb = temb.chunk(2, dim=0)[0]
        out = self.proj_out(self.norm_out(img, temb))
        #: ⚠️ **必须丢掉参考 token** ✓（``[:, :n_target]`` ✓）再 unpatchify ✗ → 否则 reshape 报形状错 ✓
        return self._unpatch(out[:, :n_target], hh, ww, shape.channels, h0, w0)

    def _mask(self, attention_mask: Tensor | None, b: int, n_txt: int, n_img: int,
              dtype) -> Tensor | None:
        """文本侧 padding 掩码 ✓ ⇒ 加性偏置 ``(b, 1, 1, n_txt + n_img)`` ✓（SDPA 认这个形状 ✓）。

        ⚠️ 掩的是 **key 侧** ✓（参考实现也是按 key 铺 ✓）：文本 padding 位不许被任何 query 看见 ✓；
        图像位恒 0 ✓。整行都被掩的光景不会出现（文本 query 至少能看图像 ✓），所以不会出 NaN ✓。
        """
        if attention_mask is None:
            return None
        m = attention_mask
        if not torch.is_floating_point(m):
            m = (m - 1).to(dtype) * torch.finfo(dtype).max
        m = m.to(dtype).reshape(b, 1, 1, -1)
        if m.shape[-1] < n_txt:
            raise ShapeError(f"attention_mask 宽 {m.shape[-1]} ✗ < 文本 token 数 {n_txt} ✓")
        pad = torch.zeros((b, 1, 1, n_img), dtype=m.dtype, device=m.device)
        return torch.cat((m[:, :, :, :n_txt], pad), dim=-1)


def build(shape: QwenImageShape) -> QwenImageDiT:
    """按形状表建模型 ✓（初值随机 ✓，由装载器逐键覆盖 ✓ —— 漏键由键对账报 ✓，不会静默留初值 ✓）。"""
    return QwenImageDiT(shape)


def build_from_shapes(shapes: Mapping[str, Sequence[int]], **kwargs) -> QwenImageDiT:
    """形状表 ⇒ 模型 ✓（``kwargs`` 直通 :func:`infer_shape` 的 ``heads``/``ref_method`` 等 ✓）。"""
    return build(infer_shape(shapes, **kwargs))

