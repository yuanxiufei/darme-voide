"""SEEDVR2（NaDiT）骨架 —— ByteDance SeedVR2 视频修复/超分主干。

权威参考是只读的 ``comfy/ldm/seedvr/model.py``（``NaDiT`` 一族），本仓不 import 它，
只按权重结构反推几何并把前向逐行镜像到自己的模块里。盘上三档实测（2026-11-01）：

* ``seedvr2_ema_3b_fp8_e4m3fn`` / ``_3b_fp16`` / ``_3b_convrot_w4a4`` —— 32 块、
  ``vid_dim=2560``、``head_dim=128``、``heads=20``、``patch=(1,2,2)``、
  ``in=33``（16 latent + 17 条件潜）、``out=16``、``ffn=6912``、``emb=15360=6*vid_dim``。
* 前 10 块双塔（``vid`` + ``txt`` 各一套），后 22 块共享（``all``）⇒ ``mm_layers=10``。
* ``txt_dim`` 恒等于 ``vid_dim``；``txt_in`` 只负责 ``txt_in_dim=5120 -> 2560``。

## 三处「形状自洽、键名全对、只有数值错」

1. **SwiGLU 的门控顺序**：``proj_out(silu(proj_in_gate(x)) * proj_in(x))``。
   写成 ``silu(proj_in(x)) * proj_in_gate(x)`` 形状全对，但两路角色互换。
2. **``AdaSingle`` 是乘法调制不是加法**：``hid * (scale_vid + scale_txt) + (shift_vid + shift_txt)``，
   ``mode="out"`` 是 ``hid * (gate_vid + gate_txt)``。写成 ``hid * (1 + scale)`` 不报错、结果钝化。
3. **末端 ``vid_out_ada`` 用的调制切片**：参考实现的 ``AdaSingle(layers=["out"])`` 会按
   ``emb.reshape(B, -1, 1, 3)`` 切成 ``2*dim`` 宽，与 ``dim`` 宽的 hid 相乘必崩；
   它能跑通纯属 ``Cache`` 的键 ``"emb_repeat_0_vid"`` 已被第 0 块 attn 组写过 ⇒ 拿到的是
   **attn 组（``layers=["attn","mlp"]`` 的 idx=0）**那份 ``[L, dim, 3]``。本仓显式按 attn 组切。

## 两处精度交接点（照抄参考，不许「顺手提高精度」）

* ``_to_flux_freqs_cis`` 里 ``angles = freqs[..., ::2].float()`` —— 旋转角一律降到 fp32 再求 cos/sin。
  角度量级可达 ``±402 rad``，fp32 的 1e-7 相对误差会变成 4e-5 弧度，32 层后足以让成图漂移。
* ``get_timestep_embedding`` 的正弦在 fp32 里算完再 ``.to(dtype)``。

## 前向契约

* ``latents``：``[B, 16, T, H, W]``（原生 5 维，不做 4 维折叠）。
* ``conditions``：``[B, 17, T, H, W]`` —— 16 通道条件潜 + 1 通道 mask，batch/T/H/W 必须与 latents 一致。
* ``context``：文本条件 ``[L, C]`` 或 ``[B, L, C]``；``ema_*``/``fp8_*`` 三档把
  ``positive_conditioning`` 从权重里剥掉了（只有 ``convrot`` 档自带）⇒ 本仓**要求显式传入**，
  缺了就当场拒，绝不拿未初始化的 buffer 兜底。
* ``sigma``：flow 口径 ``[0,1]``，内部换算 ``timestep = sigma * 1000``
  （参考侧 ``ModelSamplingDiscreteFlow.multiplier == 1000``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from . import sdpa

SEEDVR2_LATENT_CHANNELS = 16
SEEDVR2_CONDITION_CHANNELS = 17
SEEDVR2_IN_CHANNELS = SEEDVR2_LATENT_CHANNELS + SEEDVR2_CONDITION_CHANNELS  # 33

SEEDVR2_NORM_EPS = 1e-5
SEEDVR2_TIMESTEP_SCALE = 1000.0

SEEDVR2_720P_REF_AREA = 45 * 80
SEEDVR2_MAX_TEMPORAL_WINDOW = 30
SEEDVR2_ROPE_MAX_FREQ = 256.0
#: ``MMRotaryEmbeddingBase`` 真正生效的是 lang 档，底数 10000（``BYTEDANCE_ROPE_MAX_FREQ`` 那档被覆盖）。
SEEDVR2_ROPE_THETA = 10000.0

#: 7B 档走 ``rope_type="rope3d"`` + 分块 MLP，本仓未镜像，见到就拒。
SEEDVR2_7B_VID_DIM = 3072

#: 只有 ``convrot`` / ``w4a8`` 档把这两组文本条件烤进了权重；其余档必须由调用方给 context。
SEEDVR2_POS_COND_SHAPE = (58, 5120)
SEEDVR2_NEG_COND_SHAPE = (64, 5120)

SEEDVR2_DEFAULT_WINDOW: Tuple[int, int, int] = (4, 3, 3)
SEEDVR2_WINDOW_METHODS: Tuple[str, ...] = (
    "720pwin_by_size_bysize",
    "720pswin_by_size_bysize",
)

SEEDVR2_ROPE_CHUNK_TOKENS = 4096


class ShapeError(ValueError):
    """权重结构与本族任何已知几何都对不上时抛这个（带人话原因）。"""


@dataclass(frozen=True)
class SeedVR2Shape:
    """SEEDVR2 主干的全部几何（全部可由权重反推，没有一项是猜的）。"""

    layers: int
    mm_layers: int
    vid_dim: int
    txt_dim: int
    txt_in_dim: int
    in_channels: int
    out_channels: int
    heads: int
    head_dim: int
    ffn_dim: int
    expand_ratio: int
    patch: Tuple[int, int, int]
    rope_freqs_per_axis: int
    sinusoidal_dim: int
    emb_dim: int
    emb_hidden_dim: int
    norm_eps: float = SEEDVR2_NORM_EPS
    window: Tuple[int, int, int] = SEEDVR2_DEFAULT_WINDOW
    window_methods: Tuple[str, ...] = SEEDVR2_WINDOW_METHODS
    vid_out_norm_ada: bool = True
    mlp_type: str = "swiglu"
    has_conditioning: bool = False
    """权重里自带 ``positive_conditioning`` / ``negative_conditioning``（只有 convrot/w4a8 档）。"""

    @property
    def latent_channels(self) -> int:
        """待修复视频的潜变量通道（``out_channels`` 的别名，前向守门读这个名）。"""
        return self.out_channels

    @property
    def condition_channels(self) -> int:
        """条件潜通道 = 总输入 − 潜变量（16 潜 + 17 条件 = 33）。"""
        return self.in_channels - self.out_channels

    @property
    def patch_volume(self) -> int:
        return self.patch[0] * self.patch[1] * self.patch[2]

    @property
    def rope_dim(self) -> int:
        """``rope_dim // 3`` 是每轴频率宽；实际参与旋转的是 ``6 * per_axis`` 维。"""
        return 6 * self.rope_freqs_per_axis

    @property
    def rot_dim(self) -> int:
        return 6 * self.rope_freqs_per_axis

    def window_of(self, layer: int) -> Tuple[Tuple[int, int, int], str]:
        return self.window, self.window_methods[layer % len(self.window_methods)]


def _triple(value) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    seq = tuple(int(v) for v in value)
    if len(seq) != 3:
        raise ShapeError(f"SEEDVR2 需要三维 patch/window，收到 {value!r}")
    return seq  # type: ignore[return-value]


def _expand_dims(x: torch.Tensor, dim: int, ndim: int) -> torch.Tensor:
    shape = x.shape
    return x.reshape(shape[:dim] + (1,) * (ndim - len(shape)) + shape[dim:])


def _flatten(hid) -> Tuple[torch.Tensor, torch.Tensor]:
    """``hid`` → ``(摊平后的 (L,C), 逐样本 ``(**shape)``)``。

    ⚠️ ``hid_shape`` **必须跟数据同设备** ✓✗：它下游会被当**索引张量**用
    （``_window_idx`` 里 ``arange(total, device=vid_shape.device)`` ⇒ 再喂
    ``torch.index_select``），CPU 上建就会在真卡上当场报
    ``Expected all tensors to be on the same device, but got index is on cpu,
    different from other tensors on cuda:0`` ✓ —— 小尺寸 CPU 测试**永远试不出来** ✗
    （2026-11-01 CUDA 首跑踩到 ✓）。参考实现把 shape 常留 CPU，是因为它在 CPU 上拼索引；
    本仓按设备走，两条路在单卡下逐位等价 ✓。
    """
    if isinstance(hid, torch.Tensor):
        return hid.flatten(0, -2), torch.as_tensor([list(hid.shape[:-1])], device=hid.device)
    device = hid[0].device if hid else None
    shape = torch.as_tensor([list(x.shape[:-1]) for x in hid], device=device)
    return torch.cat([x.reshape(-1, x.shape[-1]) for x in hid], dim=0), shape


def _unflatten(hid: torch.Tensor, hid_shape: torch.Tensor) -> List[torch.Tensor]:
    """``(L, ...c)`` → 逐样本 ``(***, c)``。

    ⚠️ 必须用 ``unflatten(0, sizes)``（只拆第 0 维、**保留**后面的通道维），不能写
    ``reshape(*sizes)`` —— ``hid_shape`` 里没有通道维，reshape 会把通道数乘进去而报
    「shape [3, 8, 8] is invalid for input of size 6336」。
    """
    shape = hid_shape.tolist()
    sizes = [int(math.prod(s)) for s in shape]
    return [x.unflatten(0, s) for x, s in zip(torch.split(hid, sizes), shape)]


def _cumulative_lengths(lengths: Sequence[int]) -> List[int]:
    out = [0]
    for value in lengths:
        out.append(out[-1] + int(value))
    return out


def _repeat_concat(vid: torch.Tensor, txt: torch.Tensor, vid_len: torch.Tensor, txt_len: torch.Tensor, txt_repeat: Sequence[int]):
    """把每个窗口的视频段与「该样本的文本段」按窗口交替拼起来。

    ⚠️ ``txt`` 先按 ``txt_repeat`` **摊平成逐窗口的列表**，再与 ``vid`` 逐块交替。漏掉这步
    摊平（直接 ``zip(逐窗口 vid, 逐样本 txt)``）会被 zip 截断成「只处理第一个窗口」——
    症状是 ``q`` 只有 31 个 token 而 ``cu_seqlens`` 说 63，报「cu_seqlens 必须止于 31」。
    """
    vid_parts = torch.split(vid, vid_len.tolist())
    txt_parts = torch.split(txt, txt_len.tolist())
    txt_parts = [part for x, n in zip(txt_parts, txt_repeat) for part in [x] * int(n)]
    flat: List[torch.Tensor] = []
    for v, t in zip(vid_parts, txt_parts):
        flat.append(v)
        flat.append(t)
    return torch.cat(flat)


def _repeat_concat_idx(vid_len: torch.Tensor, txt_len: torch.Tensor, txt_repeat: torch.Tensor):
    device = vid_len.device
    #: ⚠️ ``txt_repeat`` 由 :func:`_window_count_tensor` 建 ⇒ **默认在 CPU** ✓✗：第 214 行的
    #: ``txt_len * txt_repeat`` 一相乘就跨设备 ⇒ ``Expected all tensors to be on the same device,
    #: but found at least two devices, cuda:0 and cpu`` ✓（2026-11-01 CUDA 首跑踩到 ✓，
    #: CPU 上**永远试不出来** ✗）。这里统一搬到 ``vid_len`` 的设备 ✓（只读 ``.tolist()`` ✓ 不参与计算 ✓），
    #: 比让每个调用点自己搬更稳 ✓。
    txt_repeat = txt_repeat.to(device=device, dtype=torch.long)
    vid_idx = torch.arange(int(vid_len.sum()), device=device)
    txt_idx = torch.arange(len(vid_idx), len(vid_idx) + int(txt_len.sum()), device=device)
    txt_repeat_list = txt_repeat.tolist()
    tgt_idx = _repeat_concat(vid_idx, txt_idx, vid_len, txt_len, txt_repeat_list)
    src_idx = torch.argsort(tgt_idx)
    txt_idx_len = len(tgt_idx) - len(vid_idx)
    repeat_txt_len = (txt_len * txt_repeat).tolist()

    def unconcat_coalesce(all_: torch.Tensor):
        vid_out, txt_out = all_[src_idx].split([len(vid_idx), txt_idx_len])
        pieces = []
        for txt, repeat_time in zip(txt_out.split(repeat_txt_len), txt_repeat_list):
            txt = txt.reshape(-1, repeat_time, *txt.shape[1:]).mean(1)
            pieces.append(txt)
        return vid_out, torch.cat(pieces)

    return (
        lambda vid, txt: torch.cat([vid, txt])[tgt_idx],
        unconcat_coalesce,
    )


def _make_720p_windows(size: Tuple[int, int, int], num_windows: Tuple[int, int, int]):
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    scale = math.sqrt(SEEDVR2_720P_REF_AREA / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = math.ceil(resized_h / resized_nh), math.ceil(resized_w / resized_nw)
    wt = math.ceil(min(t, SEEDVR2_MAX_TEMPORAL_WINDOW) / resized_nt)
    nt, nh, nw = math.ceil(t / wt), math.ceil(h / wh), math.ceil(w / ww)
    return [
        (
            slice(it * wt, min((it + 1) * wt, t)),
            slice(ih * wh, min((ih + 1) * wh, h)),
            slice(iw * ww, min((iw + 1) * ww, w)),
        )
        for iw in range(nw)
        if min((iw + 1) * ww, w) > iw * ww
        for ih in range(nh)
        if min((ih + 1) * wh, h) > ih * wh
        for it in range(nt)
        if min((it + 1) * wt, t) > it * wt
    ]


def _make_shifted_720p_windows(size: Tuple[int, int, int], num_windows: Tuple[int, int, int]):
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    scale = math.sqrt(SEEDVR2_720P_REF_AREA / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = math.ceil(resized_h / resized_nh), math.ceil(resized_w / resized_nw)
    wt = math.ceil(min(t, SEEDVR2_MAX_TEMPORAL_WINDOW) / resized_nt)

    st, sh, sw = (
        0.5 if wt < t else 0,
        0.5 if wh < h else 0,
        0.5 if ww < w else 0,
    )
    nt, nh, nw = math.ceil((t - st) / wt), math.ceil((h - sh) / wh), math.ceil((w - sw) / ww)
    nt, nh, nw = (
        nt + 1 if st > 0 else 1,
        nh + 1 if sh > 0 else 1,
        nw + 1 if sw > 0 else 1,
    )
    return [
        (
            slice(max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t)),
            slice(max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)),
            slice(max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)),
        )
        for iw in range(nw)
        if min(int((iw - sw + 1) * ww), w) > max(int((iw - sw) * ww), 0)
        for ih in range(nh)
        if min(int((ih - sh + 1) * wh), h) > max(int((ih - sh) * wh), 0)
        for it in range(nt)
        if min(int((it - st + 1) * wt), t) > max(int((it - st) * wt), 0)
    ]


_WINDOW_FNS: Dict[str, Callable] = {
    "720pwin_by_size_bysize": _make_720p_windows,
    "720pswin_by_size_bysize": _make_shifted_720p_windows,
}


def _window(hid: torch.Tensor, hid_shape: torch.Tensor, window_fn):
    """按窗口切分并摊平成 ``(总 token, c)``。

    ⚠️ ``window_fn`` 的签名是**吃张量、还切口后的张量列表** ✓（唯一实现是
    ``_Patcher._make_window``：它从 ``x.shape`` 里自己取 ``(t, h, w)`` 再套 ``window_op``）。
    所以这里传的是**带通道维的整张量** ✓ —— 别把 ``_WINDOW_FNS`` 里的裸函数（签名是
    吃 ``(t, h, w)`` 元组）直接当 ``window_fn`` 递进来 ✗，那会在 ``t, h, w, _ = x.shape``
    上炸 ``'torch.Size' object has no attribute 'shape'``（口径混用的症状 ✗）。
    """
    hid = _unflatten(hid, hid_shape)
    hid = [window_fn(x) for x in hid]
    hid_windows_list = [len(x) for x in hid]
    hid_windows = torch.as_tensor(hid_windows_list, device=hid_shape.device)
    flat: List[torch.Tensor] = []
    for x in hid:
        flat.extend(x)
    hid_len_list = [int(math.prod(x.shape[:-1])) for x in flat]
    hid, hid_shape = _flatten(flat)
    return hid, hid_shape, hid_windows, hid_len_list, hid_windows_list


def _window_idx(vid_shape: torch.Tensor, window_fn):
    """返回 (分区, 还原, 窗口形状, 每窗口长度, 每样本窗口数)。

    参考实现靠 ``torch.argsort(tgt_idx)`` 还原，要求 ``tgt_idx`` 是 ``arange(total)`` 的置换
    （窗口不重叠）。两种窗口法整数化后都恰好无缝不重叠，本仓仍显式校验，真重叠就当场报错。
    """
    total = int(vid_shape.prod(-1).sum())
    hid_idx = torch.arange(total, device=vid_shape.device).unsqueeze(-1)
    tgt_idx, tgt_shape, tgt_windows, tgt_len_list, tgt_windows_list = _window(hid_idx, vid_shape, window_fn)
    tgt_idx = tgt_idx.squeeze(-1)
    if tgt_idx.numel() != total:
        raise ShapeError(f"SEEDVR2 窗口重叠：分区后 {tgt_idx.numel()} 个 token，原始只有 {total} 个")
    src_idx = torch.argsort(tgt_idx)
    return (
        lambda hid: torch.index_select(hid, 0, tgt_idx),
        lambda hid: torch.index_select(hid, 0, src_idx),
        tgt_shape,
        tgt_len_list,
        tgt_windows_list,
    )


class _RotaryEmbedding(nn.Module):
    """只有 ``freqs`` 一个 buffer（参考侧 ``RotaryEmbedding``），两档公式：

    * ``freqs_for="lang"``：``1/θ^(arange(0,dim,2)/dim)``、位置取 ``arange``，**底数 θ=10000**；
    * ``freqs_for="pixel"``：``linspace(1, max_freq/2, dim/2)·π``、位置取 ``linspace(-1,1)``。

    ⚠️ 本骨架只走 ``lang``：``NaMMRotaryEmbedding3d`` 走 ``MMRotaryEmbeddingBase``，它在
    ``RotaryEmbeddingBase`` 之后**又覆盖了一次** ``self.rope``（pixel → lang）。两档的 buffer
    长度同为 ``per_axis = (rope_dim // 3) // 2``（3b: 128//3=42 ⇒ 21），**形状一模一样、数值不同**，
    照 7B 那条 ``rope3d`` 写会「跑通但出片糊」。``pixel`` 分支留着只为标注那个分叉点（7B 已被拒）。
    """

    def __init__(self, per_axis: int, *, freqs_for: str = "lang",
                 theta: float = SEEDVR2_ROPE_THETA, max_freq: float = SEEDVR2_ROPE_MAX_FREQ):
        super().__init__()
        if per_axis <= 0:
            raise ShapeError(f"SEEDVR2 旋转频率宽度必须为正，收到 {per_axis}")
        width = 2 * per_axis                       # 参考侧传给 RotaryEmbedding 的 dim
        if freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, width, 2)[:per_axis].float() / width))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, per_axis) * math.pi
        else:
            raise ShapeError(f"未知的 SEEDVR2 旋转频率档：{freqs_for!r}")
        self.freqs_for = freqs_for
        self.register_buffer("freqs", freqs, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs
        freqs = torch.einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        return freqs.unsqueeze(-1).expand(*freqs.shape, 2).flatten(-2)

    def axis_freqs(self, dim: int) -> torch.Tensor:
        if self.freqs_for == "lang":
            pos = torch.arange(dim, device=self.freqs.device)
        else:
            pos = torch.linspace(-1, 1, steps=dim, device=self.freqs.device)
        return self.forward(pos)

    def axial_freqs(self, *dims: int) -> torch.Tensor:
        colon = slice(None)
        all_freqs = []
        for ind, dim in enumerate(dims):
            freqs = self.axis_freqs(dim)
            all_axis = [None] * len(dims)
            all_axis[ind] = colon
            all_freqs.append(freqs[(Ellipsis, *all_axis, colon)])
        return torch.cat(torch.broadcast_tensors(*all_freqs), dim=-1)


def _to_flux_freqs_cis(freqs_interleaved: torch.Tensor) -> torch.Tensor:
    # 精度交接点：参考有意在 fp32 里求 cos/sin（见模块文档）。
    angles = freqs_interleaved[..., ::2].float()
    out = torch.stack([torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2)


def _apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = x.to(freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, : x_.shape[2]]
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


def _apply_rope1_partial(t: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    out = t.clone()
    rot_d = 2 * freqs_cis.shape[-3]
    seq_len = out.shape[-2]
    for start in range(0, seq_len, SEEDVR2_ROPE_CHUNK_TOKENS):
        end = min(start + SEEDVR2_ROPE_CHUNK_TOKENS, seq_len)
        chunk = freqs_cis[start:end]
        if rot_d == out.shape[-1]:
            out[..., start:end, :] = _apply_rope1(out[..., start:end, :], chunk).to(out.dtype)
        else:
            out[..., start:end, :rot_d] = _apply_rope1(out[..., start:end, :rot_d], chunk).to(out.dtype)
    return out


class _MMRotary3d(nn.Module):
    """视频 3 轴 + 文本 1 轴的旋转位置；``rope.rope.freqs`` 是唯一持久 buffer。"""

    def __init__(self, per_axis: int):
        super().__init__()
        self.rope = _RotaryEmbedding(per_axis)
        self.per_axis = per_axis

    def get_freqs(self, vid_shape: torch.Tensor, txt_shape: torch.Tensor):
        frames, heights, widths = vid_shape.unbind(-1)
        txt_lens = txt_shape[:, 0]
        max_temporal = int((txt_lens + frames).max())
        max_height = int(heights.max())
        max_width = int(widths.max())
        max_txt_len = int(txt_lens.max())

        vid_freqs = self.rope.axial_freqs(max_temporal + 16, max_height + 4, max_width + 4).float()
        txt_freqs = self.rope.axial_freqs(max_txt_len + 16)

        vid_list, txt_list = [], []
        width = vid_freqs.size(-1)
        for (f, h, w), l in zip(vid_shape.tolist(), txt_lens.tolist()):
            vid_list.append(vid_freqs[l : l + f, :h, :w].reshape(-1, width))
            txt_list.append(txt_freqs[:l].repeat(1, 3).reshape(-1, width))
        return (
            _to_flux_freqs_cis(torch.cat(vid_list, dim=0)),
            _to_flux_freqs_cis(torch.cat(txt_list, dim=0)),
        )

    def forward(self, vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape):
        vid_freqs, txt_freqs = self.get_freqs(vid_shape, txt_shape)
        vid_freqs = vid_freqs.to(vid_q.device)
        txt_freqs = txt_freqs.to(txt_q.device)
        vid_q = _apply_rope1_partial(vid_q.transpose(0, 1), vid_freqs).transpose(0, 1)
        vid_k = _apply_rope1_partial(vid_k.transpose(0, 1), vid_freqs).transpose(0, 1)
        txt_q = _apply_rope1_partial(txt_q.transpose(0, 1), txt_freqs).transpose(0, 1)
        txt_k = _apply_rope1_partial(txt_k.transpose(0, 1), txt_freqs).transpose(0, 1)
        return vid_q, vid_k, txt_q, txt_k


# --------------------------------------------------------------------------- #
# 双塔模块（键名对齐 ``*.all.*`` / ``*.vid.*`` / ``*.txt.*``）
# --------------------------------------------------------------------------- #


class _MMLinear(nn.Module):
    def __init__(self, vid_in: int, vid_out: int, txt_in: int, txt_out: int, *, bias: bool, shared: bool, vid_only: bool, dtype=None):
        super().__init__()
        self.shared = shared
        self.vid_only = vid_only
        if shared:
            if vid_in != txt_in or vid_out != txt_out:
                raise ShapeError(f"SEEDVR2 共享线性层要求双塔同宽，收到 vid {vid_in}->{vid_out} / txt {txt_in}->{txt_out}")
            self.all = nn.Linear(vid_in, vid_out, bias=bias, dtype=dtype)
        else:
            self.vid = nn.Linear(vid_in, vid_out, bias=bias, dtype=dtype)
            if not vid_only:
                self.txt = nn.Linear(txt_in, txt_out, bias=bias, dtype=dtype)

    def for_vid(self, x: torch.Tensor) -> torch.Tensor:
        return self.all(x) if self.shared else self.vid(x)

    def for_txt(self, x: torch.Tensor) -> torch.Tensor:
        if self.vid_only:
            return x
        return self.all(x) if self.shared else self.txt(x)


class _MMWeight(nn.Module):
    """只装一个 ``weight`` 参数的容器（供 RMSNorm 的 affine 形态对齐键名）。"""

    def __init__(self, width: int, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=dtype))


class _MMRMSNorm(nn.Module):
    def __init__(self, vid_width: int, txt_width: int, *, eps: float, shared: bool, vid_only: bool, dtype=None):
        super().__init__()
        self.eps = eps
        self.shared = shared
        self.vid_only = vid_only
        self.vid_width = vid_width
        self.txt_width = txt_width
        if shared:
            self.all = _MMWeight(vid_width, dtype)
        else:
            self.vid = _MMWeight(vid_width, dtype)
            if not vid_only:
                self.txt = _MMWeight(txt_width, dtype)

    def for_vid(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.all.weight if self.shared else self.vid.weight
        return F.rms_norm(x, (self.vid_width,), weight.to(x.dtype), self.eps)

    def for_txt(self, x: torch.Tensor) -> torch.Tensor:
        if self.vid_only:
            return x
        weight = self.all.weight if self.shared else self.txt.weight
        return F.rms_norm(x, (self.txt_width,), weight.to(x.dtype), self.eps)


class _AdaBranch(nn.Module):
    """``{layer}_{shift,scale}``（``mode="in"``）与 ``{layer}_gate``（``mode="out"``）。"""

    def __init__(self, dim: int, layers: Sequence[str], modes: Sequence[str], dtype=None):
        super().__init__()
        for layer in layers:
            if "in" in modes:
                self.register_parameter(f"{layer}_shift", nn.Parameter(torch.zeros(dim, dtype=dtype)))
                self.register_parameter(f"{layer}_scale", nn.Parameter(torch.zeros(dim, dtype=dtype)))
            if "out" in modes:
                self.register_parameter(f"{layer}_gate", nn.Parameter(torch.zeros(dim, dtype=dtype)))


class _AdaMM(nn.Module):
    """全局调制（``AdaSingle``）：``emb`` 按 ``[B, dim, n_layers, 3]`` 切 shift/scale/gate。"""

    def __init__(self, dim: int, txt_dim: int, layers: Sequence[str], modes: Sequence[str], *, shared: bool, vid_only: bool, dtype=None):
        super().__init__()
        self.layers = tuple(layers)
        self.modes = tuple(modes)
        self.shared = shared
        self.vid_only = vid_only
        self.dim = dim
        self.txt_dim = txt_dim
        if shared:
            self.all = _AdaBranch(dim, layers, modes, dtype)
        else:
            self.vid = _AdaBranch(dim, layers, modes, dtype)
            if not vid_only:
                self.txt = _AdaBranch(txt_dim, layers, modes, dtype)

    def _branch(self, key: str) -> _AdaBranch:
        return self.all if self.shared else getattr(self, key)

    def _slice(self, emb: torch.Tensor, layer: str, hid, hid_len: torch.Tensor):
        idx = self.layers.index(layer)
        emb = emb.reshape(emb.shape[0], -1, len(self.layers), 3)[:, :, idx, :]
        if hid_len.numel() == 1:
            # B=1 是常态：`(dim, 3)` 直接广播，省掉参考侧那份 ``3·dim·L`` 的物化
            # （720p 视频 L 可达十万级，逐次物化要 1 GiB 以上）
            return emb[0].unbind(-1)
        emb = _expand_dims(emb, 1, hid.ndim + 1)
        return torch.repeat_interleave(emb, hid_len, dim=0).unbind(-1)

    def apply_in(self, hid: torch.Tensor, emb: torch.Tensor, layer: str, hid_len: torch.Tensor, *, branch: str) -> torch.Tensor:
        if branch == "txt" and self.vid_only:
            return hid            # 末块的 mlp 组只跑视频，文本必须原样透传（参考 MMModule 跳过）
        mod = self._branch(branch)
        shift, scale, _ = self._slice(emb, layer, hid, hid_len)
        base_shift = getattr(mod, f"{layer}_shift").to(hid.dtype)
        base_scale = getattr(mod, f"{layer}_scale").to(hid.dtype)
        return hid.mul_(scale + base_scale).add_(shift + base_shift)

    def apply_out(self, hid: torch.Tensor, emb: torch.Tensor, layer: str, hid_len: torch.Tensor, *, branch: str) -> torch.Tensor:
        if branch == "txt" and self.vid_only:
            return hid
        mod = self._branch(branch)
        _, _, gate = self._slice(emb, layer, hid, hid_len)
        base_gate = getattr(mod, f"{layer}_gate").to(hid.dtype)
        return hid.mul_(gate + base_gate)


class _MMNorm(nn.Module):
    """``elementwise_affine=False`` 的 RMSNorm：块内 ``attn_norm`` / ``mlp_norm`` 用这个。

    ⚠️ 参考侧这两个 norm 是 ``RMSNorm(dim, eps, elementwise_affine=False)``，**没有权重**；
    若误用带 affine 的 ``_MMRMSNorm``，会凭空多出 ``blocks.N.attn_norm.*.weight`` 键，加载时
    报「意外的键」——症状看着像权重文件不对，其实是骨架建错了。
    """

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), None, self.eps)


def _var_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                   heads: int, cu_seqlens: Sequence[int]) -> torch.Tensor:
    """逐段 SDPA（参考 ``optimized_var_attention`` 的 split 实现）。

    参考侧每个窗口段是一次独立的注意力（窗口之间互不可见），所以不能拼成一个
    因果/全注意力算——拼错了「形状全对、画面串味」。
    """
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ShapeError(f"SEEDVR2 变长注意力：q/k/v token 数不一致（{tuple(q.shape)} / {tuple(k.shape)} / {tuple(v.shape)}）")
    if len(cu_seqlens) < 2 or cu_seqlens[0] != 0 or cu_seqlens[-1] != q.shape[0]:
        raise ShapeError(f"SEEDVR2 变长注意力：cu_seqlens 必须从 0 起、止于 {q.shape[0]}，收到 {list(cu_seqlens)}")
    split_indices = list(cu_seqlens[1:-1])
    q_splits = torch.tensor_split(q, split_indices, dim=0)
    k_splits = torch.tensor_split(k, split_indices, dim=0)
    v_splits = torch.tensor_split(v, split_indices, dim=0)
    if not (len(q_splits) == len(k_splits) == len(v_splits)):
        raise ShapeError("SEEDVR2 变长注意力：q/k/v 的段数不一致")
    outs: List[torch.Tensor] = []
    for q_i, k_i, v_i in zip(q_splits, k_splits, v_splits):
        # (L, H, D) → (1, H, L, D)
        q_i = q_i.permute(1, 0, 2).unsqueeze(0)
        k_i = k_i.permute(1, 0, 2).unsqueeze(0)
        v_i = v_i.permute(1, 0, 2).unsqueeze(0)
        out_i = sdpa.attention(q_i, k_i, v_i)
        outs.append(out_i.squeeze(0).permute(1, 0, 2))
    return torch.cat(outs, dim=0)


class _SwinAttention(nn.Module):
    """双塔 Swin 窗口注意力（参考 ``NaMMAttention`` + ``NaSwinAttention``）。

    文本 token 在注意力里被**复制进每个窗口**（``repeat_concat_idx`` 的最小索引语义），
    窗口内做完再按窗口 ``mean`` 聚合回原长度——所以文本是「每个窗口都能被看到」。
    """

    def __init__(self, shape: SeedVR2Shape, *, layer: int, shared: bool, dtype=None):
        super().__init__()
        inner = shape.heads * shape.head_dim
        self.proj_qkv = _MMLinear(shape.vid_dim, 3 * inner, shape.txt_dim, 3 * inner,
                                  bias=False, shared=shared, vid_only=False, dtype=dtype)
        self.proj_out = _MMLinear(inner, shape.vid_dim, inner, shape.txt_dim,
                                  bias=True, shared=shared, vid_only=False, dtype=dtype)
        self.norm_q = _MMRMSNorm(shape.head_dim, shape.head_dim, eps=shape.norm_eps,
                                 shared=shared, vid_only=False, dtype=dtype)
        self.norm_k = _MMRMSNorm(shape.head_dim, shape.head_dim, eps=shape.norm_eps,
                                 shared=shared, vid_only=False, dtype=dtype)
        self.rope = _MMRotary3d(shape.rope_freqs_per_axis)
        self.heads = shape.heads
        self.head_dim = shape.head_dim
        self.window, method = shape.window_of(layer)
        self.window_method = method
        self.window_op = _WINDOW_FNS[method]

    def _make_window(self, x: torch.Tensor):
        t, h, w, _ = x.shape
        windows = self.window_op((t, h, w), self.window)
        return [x[st, sh, sw] for (st, sh, sw) in windows]

    def forward(self, vid: torch.Tensor, txt: torch.Tensor,
                vid_shape: torch.Tensor, txt_shape: torch.Tensor):
        vid_qkv = self.proj_qkv.for_vid(vid)
        txt_qkv = self.proj_qkv.for_txt(txt)

        window_partition, window_reverse, window_shape, vid_len_win_list, window_count_list = _window_idx(
            vid_shape, self._make_window)

        vid_qkv_win = window_partition(vid_qkv).reshape(-1, 3, self.heads, self.head_dim)
        txt_qkv = txt_qkv.reshape(-1, 3, self.heads, self.head_dim)
        vid_q, vid_k, vid_v = vid_qkv_win.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)
        vid_q = self.norm_q.for_vid(vid_q)
        txt_q = self.norm_q.for_txt(txt_q)
        vid_k = self.norm_k.for_vid(vid_k)
        txt_k = self.norm_k.for_txt(txt_k)

        txt_len = txt_shape.prod(-1)
        vid_q, vid_k, txt_q, txt_k = self._repeat_rope(
            vid_q, vid_k, txt_q, txt_k, window_shape, txt_shape, window_count_list)

        txt_len_win_list = [
            int(tl) for tl, n in zip(txt_len.tolist(), window_count_list) for _ in range(int(n))
        ]
        all_len_win = [a + b for a, b in zip(vid_len_win_list, txt_len_win_list)]
        concat_win, unconcat_win = _repeat_concat_idx(
            window_shape.prod(-1), txt_len, _window_count_tensor(window_count_list))

        out = _var_attention(
            concat_win(vid_q, txt_q),
            concat_win(vid_k, txt_k),
            concat_win(vid_v, txt_v),
            heads=self.heads,
            cu_seqlens=_cumulative_lengths(all_len_win),
        )
        vid_out, txt_out = unconcat_win(out)
        vid_out = window_reverse(vid_out.flatten(1, 2))
        txt_out = txt_out.flatten(1, 2)
        return self.proj_out.for_vid(vid_out), self.proj_out.for_txt(txt_out)

    def _repeat_rope(self, vid_q, vid_k, txt_q, txt_k, window_shape, txt_shape, window_count_list):
        """文本 q/k 先按窗口数复制、再与视频一起做 rope（参考在同一步里干这件事）。"""
        num_h = txt_q.shape[1]
        repeated: List[torch.Tensor] = []
        for item in (txt_q, txt_k):
            parts = _unflatten(item.flatten(1, 2), txt_shape)
            parts = [x for x, n in zip(parts, window_count_list) for _ in range(int(n))]
            flat, rep_shape = _flatten(parts)
            repeated.append((flat.reshape(flat.shape[0], num_h, self.head_dim), rep_shape))
        txt_q_rep, txt_shape_rep = repeated[0]
        txt_k_rep, _ = repeated[1]
        return self.rope(vid_q, vid_k, window_shape, txt_q_rep, txt_k_rep, txt_shape_rep)


def _window_count_tensor(counts: Sequence[int]) -> torch.Tensor:
    """窗口计数转 1-D long 张量（``_repeat_concat_idx`` 只读 ``.tolist()``）。"""
    return torch.as_tensor(list(counts), dtype=torch.long)


class _MLP(nn.Module):
    """单塔前馈：``swiglu`` 与普通 ``gelu`` 两档（由权重里有没有 ``proj_in_gate`` 决定）。"""

    def __init__(self, dim: int, ffn_dim: int, *, kind: str, dtype=None):
        super().__init__()
        self.kind = kind
        if kind == "swiglu":
            self.proj_in = nn.Linear(dim, ffn_dim, bias=False, dtype=dtype)
            self.proj_in_gate = nn.Linear(dim, ffn_dim, bias=False, dtype=dtype)
            self.proj_out = nn.Linear(ffn_dim, dim, bias=False, dtype=dtype)
        elif kind == "gelu":
            self.proj_in = nn.Linear(dim, ffn_dim, bias=True, dtype=dtype)
            self.proj_out = nn.Linear(ffn_dim, dim, bias=True, dtype=dtype)
            self.act = nn.GELU(approximate="tanh")
        else:
            raise ShapeError(f"未知的 SEEDVR2 前馈档：{kind!r}（只支持 swiglu / gelu）")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "swiglu":
            # ⚠️ 门控顺序：silu(门) * 值。写成 silu(值) * 门 形状全对、数值全错。
            return self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))
        return self.proj_out(self.act(self.proj_in(x)))


class _MMDual(nn.Module):
    """把「单塔模块工厂」包成双塔：``shared`` ⇒ ``.all``；否则 ``.vid`` + ``.txt``。"""

    def __init__(self, build: Callable[[str], nn.Module], *, shared: bool, vid_only: bool):
        super().__init__()
        self.shared = shared
        self.vid_only = vid_only
        if shared:
            self.all = build("all")
        else:
            self.vid = build("vid")
            if not vid_only:
                self.txt = build("txt")

    def for_vid(self, x: torch.Tensor) -> torch.Tensor:
        return (self.all if self.shared else self.vid)(x)

    def for_txt(self, x: torch.Tensor) -> torch.Tensor:
        if self.vid_only:
            return x
        return (self.all if self.shared else self.txt)(x)


class _Block(nn.Module):
    """``NaMMSRTransformerBlock`` 镜像（前 10 块双塔、后 22 块共享、末块只跑视频）。"""

    def __init__(self, shape: SeedVR2Shape, *, layer: int, is_last_layer: bool, dtype=None):
        super().__init__()
        shared = layer >= shape.mm_layers
        self.attn_norm = _MMNorm(shape.norm_eps)
        self.mlp_norm = _MMNorm(shape.norm_eps)
        self.attn = _SwinAttention(shape, layer=layer, shared=shared, dtype=dtype)
        self.mlp = _MMDual(
            lambda key: _MLP(
                shape.vid_dim if key != "txt" else shape.txt_dim,
                shape.ffn_dim,
                kind=shape.mlp_type,
                dtype=dtype,
            ),
            shared=shared,
            vid_only=is_last_layer,
        )
        self.ada = _AdaMM(shape.vid_dim, shape.txt_dim, ("attn", "mlp"), ("in", "out"),
                          shared=shared, vid_only=is_last_layer, dtype=dtype)
        self.vid_only = is_last_layer

    def forward(self, vid: torch.Tensor, txt: torch.Tensor,
                vid_shape: torch.Tensor, txt_shape: torch.Tensor, emb: torch.Tensor):
        vid_len = vid_shape.prod(-1)
        txt_len = txt_shape.prod(-1)

        vid_attn, txt_attn = self.attn_norm(vid), self.attn_norm(txt)
        vid_attn = self.ada.apply_in(vid_attn, emb, "attn", vid_len, branch="vid")
        txt_attn = self.ada.apply_in(txt_attn, emb, "attn", txt_len, branch="txt")
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape)
        vid_attn = self.ada.apply_out(vid_attn, emb, "attn", vid_len, branch="vid")
        txt_attn = self.ada.apply_out(txt_attn, emb, "attn", txt_len, branch="txt")
        vid_attn = vid_attn + vid
        txt_attn = txt_attn + txt

        # 末块 `mlp_norm` / `mlp` / `ada` 都是 vid_only ⇒ 文本原样透传（参考靠 MMModule 的
        # ``if not self.vid_only`` 跳过；这里必须同样跳过，否则白算一遍文本分支）
        vid_mlp = self.mlp_norm(vid_attn)
        txt_mlp = txt_attn if self.vid_only else self.mlp_norm(txt_attn)
        vid_mlp = self.ada.apply_in(vid_mlp, emb, "mlp", vid_len, branch="vid")
        txt_mlp = self.ada.apply_in(txt_mlp, emb, "mlp", txt_len, branch="txt")
        vid_mlp = self.mlp.for_vid(vid_mlp)
        if not self.vid_only:
            txt_mlp = self.mlp.for_txt(txt_mlp.to(dtype=vid.dtype))
        vid_mlp = self.ada.apply_out(vid_mlp, emb, "mlp", vid_len, branch="vid")
        txt_mlp = self.ada.apply_out(txt_mlp, emb, "mlp", txt_len, branch="txt")
        return vid_mlp + vid_attn, txt_mlp + txt_attn


# --------------------------------------------------------------------------- #
# patch 进出 / 时间嵌入 / 末端调制
# --------------------------------------------------------------------------- #


def _timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """参考 ``get_timestep_embedding(flip_sin_to_cos=False, downscale_freq_shift=0)``。

    ⚠️ 顺序是 ``[sin, cos]``（不是常见的 cos 在前）；分母是 ``half_dim``（不减 1）；
    全程 fp32 算，最后才 ``.to(dtype)`` —— 这就是第一处精度交接点。
    """
    if timesteps.ndim != 1:
        raise ShapeError(f"SEEDVR2 时间步必须是一维，收到 {tuple(timesteps.shape)}")
    half = dim // 2
    emb = math.log(10000) / half
    freqs = torch.exp(torch.arange(half, dtype=torch.float32, device=timesteps.device) * -emb)
    args = timesteps.float()[:, None] * freqs[None]
    out = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        out = F.pad(out, (0, 1, 0, 0))
    return out


class _TimeEmbedding(nn.Module):
    """``emb_in``：256 维正弦 → 隐层 → 隐层 → ``emb_dim``（两层隐层，别写成一层）。"""

    def __init__(self, emb_dim: int, hidden: int, sinusoidal_dim: int, dtype=None):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden, bias=True, dtype=dtype)
        self.proj_hid = nn.Linear(hidden, hidden, bias=True, dtype=dtype)
        self.proj_out = nn.Linear(hidden, emb_dim, bias=True, dtype=dtype)
        self.act = nn.SiLU()

    def forward(self, timestep: torch.Tensor, *, dtype) -> torch.Tensor:
        emb = _timestep_embedding(timestep, self.sinusoidal_dim).to(dtype=dtype)
        emb = self.act(self.proj_in(emb))
        emb = self.act(self.proj_hid(emb))
        return self.proj_out(emb)


class _PatchIn(nn.Module):
    """``NaPatchIn``：逐样本做时空 patchify，再线性投影到 ``dim``。"""

    def __init__(self, in_channels: int, patch: Sequence[int], dim: int, dtype=None):
        super().__init__()
        self.patch = _triple(patch)
        t, h, w = self.patch
        self.proj = nn.Linear(in_channels * t * h * w, dim, bias=True, dtype=dtype)

    def forward(self, vid: torch.Tensor, vid_shape: torch.Tensor):
        t, h, w = self.patch
        before = vid_shape
        if (t, h, w) != (1, 1, 1):
            parts = _unflatten(vid, vid_shape)
            for i, part in enumerate(parts):
                if t > 1 and int(before[i, 0]) % t != 0:
                    # 参考按「首帧复制」补零，且补的帧数用 len % t 算（不是 shape - len % t）
                    part = torch.cat([part[:1]] * (t - part.size(0) % t) + [part], dim=0)
                tt, hh, ww, cc = part.shape
                if tt % t or hh % h or ww % w:
                    raise ShapeError(
                        f"SEEDVR2 第 {i} 个样本补帧后仍不能整除 patch{(t, h, w)}：{(tt, hh, ww)}")
                parts[i] = part.view(tt // t, t, hh // h, h, ww // w, w, cc).permute(
                    0, 2, 4, 1, 3, 5, 6).reshape(tt // t, hh // h, ww // w, t * h * w * cc)
            vid, vid_shape = _flatten(parts)
        return self.proj(vid), vid_shape, before


class _PatchOut(nn.Module):
    """``NaPatchOut``：线性投影回 ``out_channels·t·h·w``，再还原成逐样本 ``(T,H,W,c)``。"""

    def __init__(self, out_channels: int, patch: Sequence[int], dim: int, dtype=None):
        super().__init__()
        self.patch = _triple(patch)
        t, h, w = self.patch
        self.proj = nn.Linear(dim, out_channels * t * h * w, bias=True, dtype=dtype)

    def forward(self, vid: torch.Tensor, vid_shape: torch.Tensor,
                before_patchify: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        t, h, w = self.patch
        vid = self.proj(vid)
        parts = _unflatten(vid, vid_shape)
        if (t, h, w) == (1, 1, 1):
            return parts
        out: List[torch.Tensor] = []
        for i, part in enumerate(parts):
            tt, hh, ww, channels = part.shape
            if channels % (t * h * w):
                raise ShapeError(
                    f"SEEDVR2 第 {i} 个样本解 patch 时通道数 {channels} 不能整除 {t * h * w}")
            c = channels // (t * h * w)
            part = part.view(tt, hh, ww, t, h, w, c).permute(0, 3, 1, 4, 2, 5, 6).reshape(
                tt * t, hh * h, ww * w, c)
            if t > 1 and before_patchify is not None and int(before_patchify[i, 0]) % t != 0:
                part = part[(t - int(before_patchify[i, 0]) % t):]
            out.append(part)
        return out


class _OutAda(nn.Module):
    """末端 ``vid_out_ada``（``AdaSingle(layers=["out"], modes=["in"])``）。

    ⚠️ 参考侧这里按 **1 层** 切 ``emb``（``reshape(B,-1,1,3)``），宽度会是 ``2·dim`` 而崩；
    它能跑通只因 ``Cache`` 键 ``emb_repeat_0_vid`` 已被第 0 块的 attn 写出，回读到的是那份
    ``[..., dim, 3]``。本仓不复刻这个缓存副作用，直接按 attn 组（2 层里的 idx=0）切——
    与参考「命中缓存」那条路的数值逐位相同。
    """

    def __init__(self, dim: int, dtype=None):
        super().__init__()
        self.out_shift = nn.Parameter(torch.zeros(dim, dtype=dtype))
        self.out_scale = nn.Parameter(torch.zeros(dim, dtype=dtype))

    def forward(self, hid: torch.Tensor, emb: torch.Tensor, hid_len: torch.Tensor) -> torch.Tensor:
        sel = emb.reshape(emb.shape[0], -1, 2, 3)[:, :, 0, :]
        if hid_len.numel() == 1:
            shift, scale, _ = sel[0].unbind(-1)
        else:
            sel = _expand_dims(sel, 1, hid.ndim + 1)
            shift, scale, _ = torch.repeat_interleave(sel, hid_len, dim=0).unbind(-1)
        return hid.mul_(scale + self.out_scale.to(hid.dtype)).add_(shift + self.out_shift.to(hid.dtype))


# --------------------------------------------------------------------------- #
# 顶层
# --------------------------------------------------------------------------- #


class SeedVR2DiT(nn.Module):
    """SeedVR2 / NaDiT 主干（3B 档：``vid_dim=2560``、32 块、20 头）。

    ``forward(latents, sigma, context, conditioning)``：

    * ``latents``：``(B, 16, T, H, W)`` 待修复视频的潜变量；
    * ``sigma``：``(B,)``，**σ∈[0,1]**，内部乘 1000 才是主干 timestep；
    * ``context``：``(B, L, 5120)``（或 B=1 时的 ``(L, 5120)``）文本嵌入；
    * ``conditioning``：``(B, 17, T, H, W)`` 条件潜（现状视频 + loss mask），与潜变量拼成 33 通道。
    """

    def __init__(self, shape: SeedVR2Shape, dtype=None):
        super().__init__()
        self.shape = shape
        self.vid_in = _PatchIn(shape.in_channels, shape.patch, shape.vid_dim, dtype=dtype)
        if shape.txt_in_dim == shape.txt_dim:
            self.txt_in = nn.Identity()
        else:
            self.txt_in = nn.Linear(shape.txt_in_dim, shape.txt_dim, bias=True, dtype=dtype)
        self.emb_in = _TimeEmbedding(shape.emb_dim, shape.emb_hidden_dim, shape.sinusoidal_dim, dtype=dtype)
        self.blocks = nn.ModuleList([
            _Block(shape, layer=i, is_last_layer=(i == shape.layers - 1), dtype=dtype)
            for i in range(shape.layers)
        ])
        self.vid_out = _PatchOut(shape.latent_channels, shape.patch, shape.vid_dim, dtype=dtype)
        if shape.vid_out_norm_ada:
            self.vid_out_norm = _MMWeight(shape.vid_dim, dtype=dtype)
            self.vid_out_ada = _OutAda(shape.vid_dim, dtype=dtype)
        else:
            self.vid_out_norm = None
            self.vid_out_ada = None
        if shape.has_conditioning:
            # 只有 convrot / w4a8 档把这两块烤进权重；其余档由调用方给 context，
            # 绝不注册「未初始化的缓冲」——参考侧那样会拿脏内存做条件。
            self.register_buffer("positive_conditioning",
                                 torch.zeros(*SEEDVR2_POS_COND_SHAPE), persistent=True)
            self.register_buffer("negative_conditioning",
                                 torch.zeros(*SEEDVR2_NEG_COND_SHAPE), persistent=True)

    # -- 形状守门 ---------------------------------------------------------- #
    def _check_latents(self, latents: torch.Tensor, conditioning: torch.Tensor):
        if latents.ndim != 5:
            raise ShapeError(f"SEEDVR2 潜变量必须是 5 维 (B,C,T,H,W)，收到 {tuple(latents.shape)}")
        if latents.shape[1] != self.shape.latent_channels:
            raise ShapeError(
                f"SEEDVR2 潜变量通道数应为 {self.shape.latent_channels}，收到 {latents.shape[1]}")
        if conditioning.ndim != 5:
            raise ShapeError(f"SEEDVR2 条件潜必须是 5 维 (B,C,T,H,W)，收到 {tuple(conditioning.shape)}")
        if conditioning.shape[1] != self.shape.condition_channels:
            raise ShapeError(
                f"SEEDVR2 条件潜通道数应为 {self.shape.condition_channels}，收到 {conditioning.shape[1]}")
        if tuple(conditioning.shape[0:1] + conditioning.shape[2:]) != tuple(latents.shape[0:1] + latents.shape[2:]):
            raise ShapeError(
                f"SEEDVR2 条件潜与潜变量的 B/T/H/W 必须一致，收到 {tuple(conditioning.shape)} vs {tuple(latents.shape)}")

    def _resolve_context(self, context: Optional[torch.Tensor], batch: int, device, dtype) -> torch.Tensor:
        if context is None:
            if not self.shape.has_conditioning:
                raise ShapeError(
                    "SEEDVR2 该档权重没有内置文本条件（只有 convrot/w4a8 档自带 "
                    "positive_conditioning），必须显式传 context")
            context = self.positive_conditioning
        ctx = context.to(device=device, dtype=dtype)
        if ctx.ndim == 2:
            if batch != 1:
                raise ShapeError(f"SEEDVR2 二维 context 只适用于 B=1，收到 B={batch}")
            ctx = ctx.unsqueeze(0)
        if ctx.ndim != 3:
            raise ShapeError(f"SEEDVR2 context 必须是 (B,L,D)，收到 {tuple(ctx.shape)}")
        if ctx.shape[0] != batch:
            raise ShapeError(f"SEEDVR2 context 批大小 {ctx.shape[0]} 与潜变量 {batch} 不一致")
        if ctx.shape[-1] != self.shape.txt_in_dim:
            raise ShapeError(
                f"SEEDVR2 context 宽度应为 {self.shape.txt_in_dim}，收到 {ctx.shape[-1]}")
        return ctx

    # -- 前向 -------------------------------------------------------------- #
    def forward(self, latents: torch.Tensor, sigma: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        if conditioning is None:
            raise ShapeError(
                "SEEDVR2 必须给 conditioning（17 通道：现状视频潜变量 + loss mask），"
                "与本模型无关的通用出图管线不能直接调用它")
        self._check_latents(latents, conditioning)
        #: ⚠️ 潜变量在采样侧**故意保持 f32** ✓（``torch.randn`` 默认 ✓、逐步累积更准 ✓），
        #: 权重却是 bf16/fp16 ✓ ⇒ **必须在这里对齐到权重精度** ✗✗，不能拿 ``latents.dtype``
        #: 当 ``dtype`` ✓。不做的症状：``mat1 and mat2 must have the same dtype, but got
        #: Float and BFloat16`` ✓ —— 报错指向 ``nn.Linear``（本条是 ``emb_in`` 里第一个），
        #: 读起来像「权重坏了 / 装错了档」✓✗，其实是精度没对齐（2026-11-01 真权重首跑踩到 ✓）。
        #: 口径与 :meth:`dit_qwen_image.QwenImageDiT.forward` 逐字相同 ✓ —— 别改成
        #: 「把采样侧潜变量降成 bf16」✗：那会丢掉逐步累积的精度 ✓。
        dtype = self.vid_in.proj.weight.dtype
        latents = latents.to(dtype=dtype)
        conditioning = conditioning.to(dtype=dtype)
        device = latents.device
        batch, _, frames, height, width = latents.shape

        ctx = self._resolve_context(context, batch, device, dtype)
        sigma = sigma.to(device=device, dtype=torch.float32).reshape(-1)
        if sigma.numel() == 1 and batch != 1:
            sigma = sigma.expand(batch)
        if sigma.numel() != batch:
            raise ShapeError(f"SEEDVR2 sigma 个数 {sigma.numel()} 与批大小 {batch} 不一致")
        timestep = sigma * SEEDVR2_TIMESTEP_SCALE
        emb = self.emb_in(timestep, dtype=dtype)

        vid = torch.cat([latents, conditioning.to(dtype=dtype)], dim=1)       # (B,33,T,H,W)
        vid = vid.permute(0, 2, 3, 4, 1).contiguous()                          # (B,T,H,W,C)
        vid, vid_shape = _flatten(list(vid))
        txt, txt_shape = _flatten(list(ctx))
        # ⚠️ ``txt_in`` 必须排在**块之前**（参考 1306 行）：块里的双塔模块吃的是 ``txt_dim``
        # 宽度的文本；放到块后 ⇒ 文本一直是 ``txt_in_dim``（5120），第 0 块调制当场报
        # 「tensor a (5120) vs b (2560)」。
        txt = self.txt_in(txt)

        vid, vid_shape, before_patchify = self.vid_in(vid, vid_shape)

        for block in self.blocks:
            vid, txt = block(vid, txt, vid_shape, txt_shape, emb)

        if self.vid_out_norm is not None:
            vid = F.rms_norm(vid, (vid.shape[-1],), self.vid_out_norm.weight.to(vid.dtype),
                             self.shape.norm_eps)
            vid = self.vid_out_ada(vid, emb, vid_shape.prod(-1))

        patched = self.vid_out(vid, vid_shape, before_patchify)
        out = torch.stack(patched)                                             # (B,T,H,W,16)
        return out.permute(0, 4, 1, 2, 3).contiguous()


# --------------------------------------------------------------------------- #
# 形状反推 / 装配
# --------------------------------------------------------------------------- #


def _as_shape_map(info: Any) -> Dict[str, Tuple[int, ...]]:
    """把「reader / ``SafetensorsInfo`` / ``{键: 形状}``」统一成形状字典。

    ``dit_build`` 传进来的是 ``{键: tuple(shape)}``；直接吃 ``safetensors.inspect()`` 的
    结论（``.tensors`` 里是 :class:`TensorEntry`）也认 ✓ —— 探针与测试都这么用。
    """
    tensors = getattr(info, "tensors", None)
    source = tensors if isinstance(tensors, Mapping) else info
    if not isinstance(source, Mapping):
        raise ShapeError(f"SEEDVR2 需要形状字典，收到 {type(info).__name__}")
    out: Dict[str, Tuple[int, ...]] = {}
    for key, value in source.items():
        shape = getattr(value, "shape", value)
        out[str(key)] = tuple(int(v) for v in shape)
    return out


def _shape_of(shapes: Mapping[str, Tuple[int, ...]], key: str) -> Tuple[int, ...]:
    if key not in shapes:
        raise ShapeError(f"SEEDVR2 权重缺关键张量 {key}（不是这一族或文件不完整）")
    return shapes[key]


def _blocks_index(shapes: Mapping[str, Any]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for key in shapes:
        if not key.startswith("blocks."):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            raise ShapeError(f"SEEDVR2 块键解析失败：{key}") from None
        out.setdefault(idx, []).append(key)
    if not out:
        raise ShapeError("SEEDVR2 权重里没有任何 blocks.* 键")
    found = sorted(out)
    if found != list(range(found[0], found[0] + len(found))) or found[0] != 0:
        raise ShapeError(f"SEEDVR2 块编号不连续：{found[:4]}…{found[-4:]}（共 {len(found)}）")
    return out


def _swiglu_hidden(dim: int, expand: int = 4, multiple_of: int = 256) -> int:
    hidden = int(dim * expand * 2 / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


def infer_shape(info: Mapping[str, Any]) -> SeedVR2Shape:
    """从权重形状反推结构（``info`` 可以是 ``{键: 形状}`` 或 reader）。"""
    shapes = _as_shape_map(info)
    blocks = _blocks_index(shapes)
    # ⚠️ 块编号必须是 ``0..N-1`` 连续 ✓：``_blocks_index`` 只按出现过的编号归类 ✗，
    #    缺一块（比如少 ``blocks.1.``）会**静默**建成一座浅一层的模型 —— 键名全对、形状全对、
    #    跑得通、只是出片全糊 ✗（这类「少一块」在量化转换/裁剪产物里真会出现 ✓）。
    idxs = sorted(blocks)
    if idxs != list(range(len(idxs))):
        raise ShapeError(
            f"SEEDVR2 块编号不连续：{idxs[:8]}{'…' if len(idxs) > 8 else ''}"
            f"（共 {len(idxs)} 个），缺块会被当成「模型就这么深」静默建成浅模型")
    num_layers = len(blocks)
    mm_layers = sum(1 for idx in blocks if any(".txt." in k for k in blocks[idx]))
    if mm_layers == 0:
        raise ShapeError("SEEDVR2 双塔块数为 0：不是 NaMM 这一族")
    # 双塔块必须**占据前 mm_layers 块**（后半段一块都不许再有 `.txt.` ✓）。
    # ⚠️ 原判据前半截 ``mm_layers != len([i for i in range(mm_layers)])`` 恒为假 ✗（死代码 ✓），
    #    真正起作用的只有后半截 ⇒ 已删 ✗。
    if any(any(".txt." in k for k in blocks[i]) for i in range(mm_layers, num_layers)):
        raise ShapeError("SEEDVR2 双塔块必须连续占据前 mm_layers 块")

    vid_in = _shape_of(shapes, "vid_in.proj.weight")
    vid_out = _shape_of(shapes, "vid_out.proj.weight")
    vid_dim = int(vid_in[0])
    latent_channels = SEEDVR2_LATENT_CHANNELS
    if vid_out[0] % latent_channels:
        raise ShapeError(f"SEEDVR2 vid_out 输出维 {vid_out[0]} 不能整除潜变量通道 {latent_channels}")
    patch_volume = vid_out[0] // latent_channels
    if vid_in[1] % patch_volume:
        raise ShapeError(f"SEEDVR2 vid_in 输入维 {vid_in[1]} 不能整除 patch 体量 {patch_volume}")
    in_channels = vid_in[1] // patch_volume
    if in_channels != SEEDVR2_IN_CHANNELS:
        raise ShapeError(
            f"SEEDVR2 输入通道推算为 {in_channels}，与 {SEEDVR2_IN_CHANNELS}"
            f"（16 潜变量 + 17 条件）不符")
    if patch_volume != 4:
        raise ShapeError(
            f"SEEDVR2 patch 体量为 {patch_volume}，只镜像了 1×2×2 那一档"
            "（(1,2,2)=4 的分解无法从权重反推）")
    if vid_dim == SEEDVR2_7B_VID_DIM:
        raise ShapeError(
            f"SEEDVR2 7B 档（vid_dim={vid_dim}）走 rope3d + 分块 MLP，本仓未镜像："
            "见到就拒，别静默按 3B 几何建（会跑通但出片全糊）")

    head_dim = int(_shape_of(shapes, "blocks.0.attn.norm_q.all.weight")[0]) \
        if "blocks.0.attn.norm_q.all.weight" in shapes \
        else int(_shape_of(shapes, "blocks.0.attn.norm_q.vid.weight")[0])
    qkv_key = "blocks.0.attn.proj_qkv.all.weight" if "blocks.0.attn.proj_qkv.all.weight" in shapes \
        else "blocks.0.attn.proj_qkv.vid.weight"
    qkv_out = int(_shape_of(shapes, qkv_key)[0])
    if qkv_out % (3 * head_dim):
        raise ShapeError(f"SEEDVR2 qkv 输出维 {qkv_out} 不能整除 3×{head_dim}")
    heads = qkv_out // (3 * head_dim)

    mlp_key = "blocks.0.mlp.all.proj_in.weight" if "blocks.0.mlp.all.proj_in.weight" in shapes \
        else "blocks.0.mlp.vid.proj_in.weight"
    ffn_dim = int(_shape_of(shapes, mlp_key)[0])
    gate_key = mlp_key.replace("proj_in.weight", "proj_in_gate.weight")
    if gate_key in shapes:
        mlp_type = "swiglu"
        expected = _swiglu_hidden(vid_dim)
        if ffn_dim != expected:
            raise ShapeError(
                f"SEEDVR2 SwiGLU 隐层 {ffn_dim} 与公式算出的 {expected}（dim={vid_dim}）不符")
    else:
        mlp_type = "gelu"
        if vid_dim == SEEDVR2_7B_VID_DIM:
            raise ShapeError(
                f"SEEDVR2 vid_dim={vid_dim} 且无 proj_in_gate：这是 7B 档（rope3d + 分块 MLP），"
                "本仓未镜像")

    rope_freqs = _shape_of(shapes, "blocks.0.attn.rope.rope.freqs")
    if len(rope_freqs) != 1:
        raise ShapeError(f"SEEDVR2 rope.freqs 形状异常：{rope_freqs}")
    per_axis = int(rope_freqs[0])
    if 2 * per_axis * 3 > head_dim:
        raise ShapeError(
            f"SEEDVR2 旋转维度 {6 * per_axis} 超过 head_dim {head_dim}（freqs 长度 {per_axis}）")

    emb_out = _shape_of(shapes, "emb_in.proj_out.weight")
    emb_in_w = _shape_of(shapes, "emb_in.proj_in.weight")
    txt_w = _shape_of(shapes, "txt_in.weight")

    return SeedVR2Shape(
        layers=num_layers,
        mm_layers=mm_layers,
        vid_dim=vid_dim,
        txt_dim=int(txt_w[0]),
        txt_in_dim=int(txt_w[1]),
        in_channels=in_channels,
        out_channels=latent_channels,
        heads=heads,
        head_dim=head_dim,
        ffn_dim=ffn_dim,
        expand_ratio=4,                    # 只用于核对 ffn_dim 的公式，见 _swiglu_hidden
        patch=(1, 2, 2),
        rope_freqs_per_axis=per_axis,
        sinusoidal_dim=int(emb_in_w[1]),
        emb_dim=int(emb_out[0]),
        emb_hidden_dim=int(emb_out[1]),
        mlp_type=mlp_type,
        vid_out_norm_ada="vid_out_norm.weight" in shapes,
        has_conditioning="positive_conditioning" in shapes,
    )


def build(shape: SeedVR2Shape, *, dtype: torch.dtype = None) -> SeedVR2DiT:
    return SeedVR2DiT(shape, dtype=dtype)


def build_from_headers(shapes: Mapping[str, Sequence[int]], *,
                       dtype: torch.dtype = None) -> SeedVR2DiT:
    return build(infer_shape(shapes), dtype=dtype)


def param_count(shape: SeedVR2Shape) -> int:
    """参数个数（不建模型也能算，用于体量核对）。"""
    dim, txt, ffn = shape.vid_dim, shape.txt_dim, shape.ffn_dim
    inner = shape.heads * shape.head_dim
    per_axis = shape.rope_freqs_per_axis

    def mlp(width: int) -> int:
        if shape.mlp_type == "swiglu":
            return 3 * ffn * width
        return 2 * ffn * width + ffn + width

    def block(idx: int, is_last: bool) -> int:
        if idx >= shape.mm_layers:
            return (6 * dim                       # ada.all
                    + 3 * inner * dim + dim * inner + dim   # qkv + proj_out（bias 宽 = 输出宽）
                    + 2 * shape.head_dim          # norm_q / norm_k
                    + mlp(dim) + per_axis)
        torso = (6 * dim + 6 * txt                 # ada.vid / ada.txt
                 + 3 * inner * dim + 3 * inner * txt
                 + dim * inner + dim + txt * inner + txt
                 + 2 * shape.head_dim * 2
                 + mlp(dim) + (0 if is_last else mlp(txt))
                 + per_axis)
        return torso

    total = sum(block(i, i == shape.layers - 1) for i in range(shape.layers))
    t, h, w = shape.patch
    total += dim * (shape.in_channels * t * h * w) + dim              # vid_in
    total += shape.txt_dim * shape.txt_in_dim + shape.txt_dim         # txt_in
    total += (shape.sinusoidal_dim * shape.emb_hidden_dim + shape.emb_hidden_dim
              + shape.emb_hidden_dim * shape.emb_hidden_dim + shape.emb_hidden_dim
              + shape.emb_hidden_dim * shape.emb_dim + shape.emb_dim)   # emb_in
    total += (shape.latent_channels * t * h * w) * dim + shape.latent_channels * t * h * w
    if shape.vid_out_norm_ada:
        total += dim + 2 * dim                                        # vid_out_norm + vid_out_ada
    if shape.has_conditioning:
        total += SEEDVR2_POS_COND_SHAPE[0] * SEEDVR2_POS_COND_SHAPE[1]
        total += SEEDVR2_NEG_COND_SHAPE[0] * SEEDVR2_NEG_COND_SHAPE[1]
    return total
