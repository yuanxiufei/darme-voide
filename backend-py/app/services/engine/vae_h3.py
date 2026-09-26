"""H3 **真视频 VAE** ✓ —— 3D 因果 CNN 编码器 ✓ + **ViT3D 解码器** ✓（两侧**不对称** ✓）。

为什么单开一个模块（而不是改 :mod:`vae` ✓）：:mod:`vae` 那套是**合成参考结构** ✓，只为「验管道」✓；
真权重**装不进**它 ✗（键名/形状逐条不符 ✓）⇒ `strict=True` 报一整页缺键 ✗，而 `strict=False` 更坏 ✗✗
（静默跳过 ✓ 画面"看着有东西" ✓ 其实是随机权重 ✗）。本模块装**真的**那份 fp16 ✓。

出处与许可 ⚠️：事实读自**上游只读源码** ``ComfyUI/comfy/ldm/minimax/vae.py``（711 行 ✓，
GPL-3.0 ✓）。本模块按「公开的架构事实」自写 ✓（张量名/形状/超参都能从真权重独立核出 ✓），
**没有逐行照抄** ✗ —— 但事实同源 ✓ ⇒ 将来若要再分发，**许可需自行评估** ✓✗（留给人定 ✓）。

⚠️ 本模块**只**管「潜变量 ⇄ 像素」✓，**不会**让画面变对 ✗（那是真 TE + 真 DiT 的事 ✓）。
"""

from __future__ import annotations

import math
import mmap
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import safetensors as safetensors_mod

__all__ = [
    "H3_VIDEO_VAE_ARCH_FACTS",
    "H3_VIDEO_VAE_WEIGHT_KEYS",
    "H3VideoVAEConfig",
    "H3VideoVAEError",
    "build_h3_video_vae",
    "decode_temporal_plan",
    "frames_for_latents",
    "h3_video_vae_weights_on_disk",
    "latents_for_frames",
    "read_state_dict",
    "verify_latents_stats",
]


class H3VideoVAEError(RuntimeError):
    """真 VAE 装不起来 ✓（缺权重 / 键不符 / 文件不完整 ✓）—— **明确报错** ✗ 不静默兜底 ✓。"""


#: ⚠️ 每条都能从**真权重文件**里独立核出来 ✓（自检逐条钉死 ✓），出处见 docstring ✓。
H3_VIDEO_VAE_ARCH_FACTS: dict[str, Any] = {
    "spatialScale": 16,
    "temporalScale": 4,
    "latentChannels": 24,
    "encoder": "6 级 3D 因果 CNN ✓（每级 2 ResBlock ✓）`space_down=(2,2,2,2,1,1)` / "
               "`time_down=(1,2,2,1,1,1)` ⇒ 16× 空间、4× 时间 ✓；出 `2×24=48` ✓（变分 ✓）",
    "decoder": "**ViT3D** ✓：`x_embedder` 24→2048 → 36 层（32 头×64 维、SwiGLU 8192、adaLN "
               "`scale1/scale2`、无仿射 `RMSNorm`）→ `proj_out` 2048→**3072**（=3×4×16×16 ✓）"
               "⇒ 一个潜格 = **4 帧 × 16×16 像素** ✓",
    "tokens": "4 枚 register token + 1 枚**全零** token ✓（`mask_token` 只为让 ckpt 整体装得进 ✓，"
              "前向**不用**它 ✗）",
    "rope": "**split-half** ✓：前 **48** 维（=64×0.75 ✓）配对 **[i]↔[i+24]** ✓（**不是**交织 ✗），"
            "`theta=100`、坐标归一化 [-1,1] 后 ×2π ✓",
    "chunking": "编码按 **17 帧**一段（末段重复末帧补齐）⇒ 丢最后 **3** 个潜帧 ✓；"
                "解码按 **5 潜帧**一段且多带 **2** 帧 ⇒ 叠 **5 帧**线性混合 ✓",
    "grid": "**17k+5 帧 ⇔ 5k+2 潜帧** ✓（与 `geometry` 同源 ✓）",
}

@dataclass(frozen=True)
class H3VideoVAEConfig:
    """真 VAE 的装载口径 ✓。默认值**就是**真权重里的事实 ✓（自检拿它们反查 ckpt ✓）。"""

    #: 真权重路径 ✓ —— **必给** ✗（不给就报错 ✓：随机初始化的"VAE"只会骗自己 ✗）
    weights: str | Path | None = None
    device: str = "cpu"
    #: ``None`` ⇒ **按设备选** ✓：cuda 用 fp16 ✓（权重本来就是 fp16 ✓、显存也省 ✓）、cpu 用 fp32 ✓
    dtype: str | None = None
    #: 分块（省显存/内存 ✓）。⚠️ 分块是**近似** ✗ 不是等价 ✗ —— 块边界上「空间 reflect padding ✓」
    #: 与「**跨块的注意力** ✗」都看不到了 ✓ ⇒ 出片会有细微差异 ✓（如实标注 ✓ 不假装等价 ✗）
    tiling: bool = True
    tile_size: int = 256
    tile_overlap_min: int = 64
    clip_length: int = 17
    token_drop: int = 3
    #: 装载/自检的严格度 ✓（默认严格 ✓ —— 缺键、形状不符一律**报错** ✓ 不静默跳过 ✗）
    strict: bool = True
    #: 出片像素落在哪 ✓（默认 **CPU** ✓）：整段视频**不进显存** ✗，而且出片本来就是 CPU 侧写盘 ✓
    output_device: str = "cpu"

    def __post_init__(self) -> None:
        """⚠️ 分块参数必须**整除空间倍率** ✗：不是整数倍 ⇒ 切出来的块在潜空间里对不齐 ✓✗
        （像素域多出来那几行只能靠丢/补凑 ✓ ⇒ 画面错位 ✓，而且**哪里都不会报错** ✗✗）。"""
        if self.tile_size <= 0 or self.tile_size % self.spatial_scale:
            raise H3VideoVAEError(
                f"tile_size={self.tile_size} 必须是 {self.spatial_scale} 的**正整数倍** ✗"
                f"（否则分块在潜空间对不齐 ⇒ 画面错位 ✓ 且不会报错 ✗）")
        if self.tile_overlap_min <= 0 or self.tile_overlap_min % self.spatial_scale:
            raise H3VideoVAEError(
                f"tile_overlap_min={self.tile_overlap_min} 必须是 {self.spatial_scale} 的"
                f"**正整数倍** ✗")

    @property
    def latent_channels(self) -> int:
        return 24

    @property
    def spatial_scale(self) -> int:
        return 16

    @property
    def temporal_scale(self) -> int:
        return 4

    @property
    def token_chunk(self) -> int:
        """一段 17 帧 ⇒ **5** 个潜帧 ✓（= ceil(17/4) ✓，出自参考实现的 ``tokens_chunk_size`` ✓）。"""
        return math.ceil(self.clip_length / self.temporal_scale)

    @property
    def token_overlap(self) -> int:
        """解码每段**多带**的潜帧数 ✓（= ``(-token_drop) % token_chunk`` = **2** ✓）。"""
        return (-self.token_drop) % self.token_chunk

    @property
    def frame_overlap(self) -> int:
        """相邻两段在**像素域**的重叠帧数 ✓（= ``token_overlap*4 - token_drop`` = **5** ✓）。"""
        return max(self.token_overlap * self.temporal_scale - self.token_drop, 0)

    @property
    def frame_pre_padding(self) -> int:
        """解码每段**开头**要丢掉的帧数 ✓（= ``(-17) % 4`` = **3** ✓ —— 见 drop_first_n_frames ✓）。"""
        return (-self.clip_length) % self.temporal_scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": str(self.weights) if self.weights else None,
            "device": self.device, "dtype": self.dtype, "tiling": self.tiling,
            "tileSize": self.tile_size, "tileOverlapMin": self.tile_overlap_min,
            "clipLength": self.clip_length,
            "tokenDrop": self.token_drop, "latentChannels": self.latent_channels,
            "spatialScale": self.spatial_scale, "temporalScale": self.temporal_scale,
            "tokenChunk": self.token_chunk, "tokenOverlap": self.token_overlap,
            "frameOverlap": self.frame_overlap, "framePrePadding": self.frame_pre_padding,
            "outputDevice": self.output_device, "strict": self.strict,
        }


class _Impl:  # noqa: N801 —— 就是个装类的盒子 ✓
    """一次 ``_build()`` 的产物盒 ✓（类必须现造 ✓ —— torch 是**可选**依赖 ✓，导入期不许碰它 ✓）。"""

    __slots__ = ("VideoVAE",)

    def __init__(self, video_vae: Any) -> None:
        self.VideoVAE = video_vae


_IMPL: _Impl | None = None


def _impl() -> _Impl:
    global _IMPL
    if _IMPL is None:
        _IMPL = _build()
    return _IMPL

def _build() -> _Impl:
    """把 torch 类**现造**出来 ✓（torch 是可选依赖 ✓ —— 导入本模块时一个字节都不碰它 ✓）。"""
    import torch  # noqa: PLC0415 —— 故意延迟 ✓
    import torch.nn as nn  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    def _cast(value: Any, like: Any) -> Any:
        """参数/缓冲 ⇒ 对齐到 ``like`` 的 **dtype + device** ✓（＝参考里的 ``cast_to_input`` ✓）。

        ⚠️ 为什么必须显式转 ✗：模型可以半精度跑 ✓，而 ``scale1/scale2`` 这类参数在 ckpt 里是 fp16 ✓、
        但**运算里出现的张量**可能是 fp32 ✓；不转的话 ``addcmul_`` 会抛 dtype 不匹配 ✗。
        """
        if value.device == like.device and value.dtype == like.dtype:
            return value
        return value.to(device=like.device, dtype=like.dtype)

    def _causal_conv3d(x: Any, weight: Any, bias: Any, stride: Any, autopad: bool = False) -> Any:
        """⚠️ 卷积本体**不加 pad** ✓（padding 全在因果那层手工做 ✓）；``autopad`` = 单帧**截断时间抽头** ✓。"""
        if autopad:
            weight = weight[:, :, -x.shape[2]:, :, :]
        return F.conv3d(x, weight, bias, stride=stride, padding=0)

    class _CausalConv3d(nn.Conv3d):
        """**因果** 3D 卷积 ✓：空间靠 reflect 补边 ✓、时间只补**前面** ✓（不会看到未来帧 ✓）。"""

        def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: Any = 1,
                     padding: Any = 0) -> None:
            super().__init__(in_ch, out_ch, kernel_size, stride)
            self.causal_padding: tuple[int, int, int] = (
                (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
            )

        def forward(self, x: Any) -> Any:
            pad = self.causal_padding
            if sum(pad) == 0:
                return super().forward(x)
            x = F.pad(x, (pad[2], pad[2], pad[1], pad[1], 0, 0), mode="reflect")
            if x.shape[2] == 1:  # 单帧 ⇒ 时间抽头**截断** ✓（等价于"把前面的零帧卷积" ✓，但更省 ✓）
                return _causal_conv3d(x, self.weight, self.bias, self.stride, autopad=True)
            x = F.pad(x, (0, 0, 0, 0, pad[0] * 2, 0))
            return _causal_conv3d(x, self.weight, self.bias, self.stride)

    class _TemporalIsolatedGroupNorm(nn.GroupNorm):
        """GroupNorm **按帧独立**做 ✓（T 折进 batch ✓）—— 否则时间轴会互相污染 ✗。"""

        def forward(self, x: Any) -> Any:
            b, c, t, h, w = x.shape
            y = x.transpose(1, 2).reshape(b * t, c, h, w)
            return super().forward(y).view(b, t, c, h, w).transpose(1, 2)

    class _Downsample3D(nn.Module):
        """空间步长 2 时**右边/下边** reflect 补 1 ✓（配合因果卷积的 stride ✓）。"""

        def __init__(self, ch: int, time_stride: int = 2, space_stride: int = 2) -> None:
            super().__init__()
            self.space_stride = space_stride
            self.conv = _CausalConv3d(ch, ch, 3, stride=(time_stride, space_stride, space_stride),
                                      padding=(1, 0, 0))

        def forward(self, x: Any) -> Any:
            if self.space_stride == 2:
                x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
            return self.conv(x)

    class _ResnetBlock3D(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.norm1 = _TemporalIsolatedGroupNorm(32, in_ch, eps=1e-6, affine=True)
            self.conv1 = _CausalConv3d(in_ch, out_ch, kernel_size=3, padding=1)
            self.norm2 = _TemporalIsolatedGroupNorm(32, out_ch, eps=1e-6, affine=True)
            self.conv2 = _CausalConv3d(out_ch, out_ch, kernel_size=3, padding=1)
            if in_ch != out_ch:  # 通道变了才有 shortcut ✓（键名与 ckpt 一致 ✓）
                self.nin_shortcut = _CausalConv3d(in_ch, out_ch, kernel_size=1)

        def forward(self, x: Any) -> Any:
            h = self.conv1(F.silu(self.norm1(x), inplace=True))
            h = self.conv2(F.silu(self.norm2(h), inplace=True))
            if x.shape[1] != h.shape[1]:
                x = self.nin_shortcut(x)
            return h.add_(x)

    class _EncoderFCN3D(nn.Module):
        """6 级 3D 因果编码器 ✓（超参就是 ckpt 里的事实 ✓ —— 见 ``H3_VIDEO_VAE_ARCH_FACTS`` ✓）。"""

        def __init__(self, ch: int = 128, ch_mult: tuple[int, ...] = (1, 2, 2, 4, 4, 8),
                     num_res_blocks: int = 2, space_down: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
                     time_down: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
                     z_channels: int = 24, double_z: bool = True) -> None:
            super().__init__()
            self.conv_in = _CausalConv3d(3, ch, 3, padding=1)
            mid = [ch * mult for mult in ch_mult]
            #: ⚠️ 每级**第一个** ResBlock 的入通道 = 上一级出通道 ✓（= `[mid[0], *mid[:-1]]` ✓，
            #: 这条不写对 ⇒ 键名会整体错位 ✓✗）
            block_in = [mid[0], *mid[:-1]]
            #: ⚠️ 每级必须是**普通 Module + 两个属性** ✓ ⇒ 键名才是 ``down.i.block.j.*`` ✓ /
            #: ``down.i.downsample.conv.*`` ✓。用 ``ModuleList`` 平铺（``down.0.0.*`` ✗）会让**全部 122 个**
            #: 编码器键都对不上 ✓✗ —— 而且 `strict=False` 时**一个都不报** ✗✗（本仓最怕的那种"装上了"✗）。
            self.down = nn.ModuleList()
            for level, out_ch in enumerate(mid):
                down = nn.Module()
                down.block = nn.ModuleList()
                for index in range(num_res_blocks):
                    down.block.append(_ResnetBlock3D(
                        block_in[level] if index == 0 else out_ch, out_ch))
                if space_down[level] * time_down[level] > 1:
                    down.downsample = _Downsample3D(out_ch, time_stride=time_down[level],
                                                    space_stride=space_down[level])
                self.down.append(down)
            self.norm_out = _TemporalIsolatedGroupNorm(32, mid[-1], eps=1e-6, affine=True)
            self.conv_out = _CausalConv3d(mid[-1], 2 * z_channels if double_z else z_channels, 3,
                                          padding=1)

        def forward(self, x: Any) -> Any:
            h = self.conv_in(x)
            for down in self.down:
                for block in down.block:
                    h = block(h)
                if hasattr(down, "downsample"):
                    h = down.downsample(h)
            return self.conv_out(F.silu(self.norm_out(h), inplace=True))

    def _apply_split_half_rope(x: Any, table: Any) -> Any:
        """**split-half** 旋转 ✓：把 ``rot`` 维按 **[i] ↔ [i+rot/2]** 配对 ✓（**不是**相邻交织 ✗）。

        ``table`` 每格是 2×2 旋转矩阵 ✓（``[c,-s;s,c]`` ✓）⇒ 等价于
        ``[c*x1 - s*x2, s*x1 + c*x2]`` ✓（x1/x2 = 前半/后半 ✓）。
        ⚠️ 头 48 维参与、后 16 维**原样带上** ✗（调用方负责拼回 ✓）。
        """
        half = table.shape[-3]
        lo, hi = x[..., :half], x[..., half:2 * half]
        cos, sin = table[..., 0, 0], table[..., 1, 0]
        return torch.cat([cos * lo - sin * hi, sin * lo + cos * hi], dim=-1)

    def _rope_table(dim: int, base: float, n_dim: int, ids: Any) -> Any:
        """坐标 ⇒ 2×2 旋转表 ✓（形状 ``[B,S,1,rot/2,2,2]`` ✓）。theta/base 与旋转范围都由 ckpt 口径定 ✓。

        ⚠️ 角度必须**在 fp32 里算** ✓、最后一步才转回模型精度 ✗ —— 反过来的话（先在低精度取 cos/sin ✗）
        角度一多就丢有效位 ✓（``rot/2=24`` 个频率里高频那几个最吃亏 ✓）。
        """
        inv_freq = 1.0 / (base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32,
                                              device=ids.device))
        angles = 2 * math.pi * ids.to(torch.float32).unsqueeze(-1) * inv_freq
        angles = angles.flatten(2, 3)
        table = torch.stack([torch.cos(angles), -torch.sin(angles),
                             torch.sin(angles), torch.cos(angles)], dim=-1)
        return table.reshape(*ids.shape[:-1], 1, angles.shape[-1], 2, 2).to(ids.dtype)

    def _token_ids(grid: tuple[int, ...], *, dtype: Any, device: Any) -> Any:
        """每个潜格的三维坐标 ✓ ⇒ 归一化到 **[-1,1]** ✓（末尾 +0.5 ⇒ 取格子中心 ✓）。

        ⚠️ ``dtype`` 由调用方给 ✓（就是输入张量的精度 ✓）—— 别在这里"顺手升 fp32" ✗：升了之后
        旋转表也跟着升 ✓ ⇒ 与 ckpt 口径不一致 ✓✗（数值小差、但会一路放大到 36 层之外 ✓）。
        """
        axes = [((torch.arange(0.5, size, dtype=dtype, device=device) / size) * 2 - 1)
                for size in grid]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, len(grid) - 1)

    class _Attention(nn.Module):
        """多头自注意力 ✓：q/k 做**无仿射 RMSNorm** ✓（ckpt 里没有它的权重 ✓）+ split-half RoPE ✓。"""

        def __init__(self, dim: int, heads: int, dim_head: int, eps: float) -> None:
            super().__init__()
            self.heads, self.dim_head = heads, dim_head
            self.to_qkv = nn.Linear(dim, heads * dim_head * 3, bias=True)
            self.to_out = nn.Linear(heads * dim_head, dim, bias=True)
            #: ⚠️ ``weight=None`` ⇒ ckpt 里**不该**有 ``norm_q.weight`` ✓（有就说明结构错了 ✗）
            self.rms_eps = eps

        def forward(self, x: Any, table: Any) -> Any:
            batch, seq, _ = x.shape
            qkv = self.to_qkv(x).view(batch, seq, self.heads, self.dim_head * 3)
            query, key, value = qkv.chunk(3, dim=-1)
            query = F.rms_norm(query, (self.dim_head,), None, self.rms_eps)
            key = F.rms_norm(key, (self.dim_head,), None, self.rms_eps)
            rot = table.shape[-3] * 2
            query = torch.cat([_apply_split_half_rope(query[..., :rot], table), query[..., rot:]],
                              dim=-1)
            key = torch.cat([_apply_split_half_rope(key[..., :rot], table), key[..., rot:]], dim=-1)
            out = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2),
                                                 value.transpose(1, 2)).transpose(1, 2)
            return self.to_out(torch.nan_to_num(out.reshape(batch, seq, -1), nan=0.0))

    class _FeedForward(nn.Module):
        """**门控 SiLU** FFN ✓：``w1`` 一次出两半 ✓ ⇒ ``silu(前半) * 后半`` ✓（顺序反了画面就废 ✗）。"""

        def __init__(self, dim: int, mult: int = 4) -> None:
            super().__init__()
            inner = dim * mult
            self.w1 = nn.Linear(dim, inner * 2, bias=True)
            self.w2 = nn.Linear(inner, dim, bias=True)

        def forward(self, x: Any) -> Any:
            gate, rest = self.w1(x).chunk(2, dim=-1)
            return self.w2(F.silu(gate).mul_(rest))

    class _TransformerBlock(nn.Module):
        """⚠️ 两条容易搞错的地方：① 两个 norm 是 **RMSNorm + 权重** ✓（**不是** LayerNorm ✗，
        也不是"无仿射" ✗ —— ckpt 里只有 ``.weight`` 没有 ``.bias`` ✓ 正说明是它 ✓）；
        ② ``scale1/scale2`` 是**逐通道** adaLN 缩放 ✓（``addcmul_`` 广播到最后那维 ✓）。"""

        def __init__(self, dim: int, heads: int, dim_head: int, eps: float) -> None:
            super().__init__()
            self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
            self.attn = _Attention(dim, heads, dim_head, eps)
            self.scale1 = nn.Parameter(torch.empty(dim))
            self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
            self.ff = _FeedForward(dim)
            self.scale2 = nn.Parameter(torch.empty(dim))

        def forward(self, x: Any, table: Any) -> Any:
            x = x.addcmul_(self.attn(self.norm1(x), table), _cast(self.scale1, x))
            return x.addcmul_(self.ff(self.norm2(x)), _cast(self.scale2, x))

    class _ViT3DDecoder(nn.Module):
        """**解码器是 transformer** ✓（编码器是 CNN ✓ —— 两侧不对称正是这个模型的特点 ✓）。"""

        def __init__(self, patch_size: int = 16, patch_size_t: int = 4, in_channels: int = 24,
                     out_channels: int = 3, num_layers: int = 36, heads: int = 32,
                     dim_head: int = 64, rope_theta: float = 100.0, rope_dim_ratio: float = 0.75,
                     eps: float = 1e-5, num_register_tokens: int = 4) -> None:
            super().__init__()
            dim = heads * dim_head
            self.patch_size, self.patch_size_t = patch_size, patch_size_t
            self.out_channels, self.num_register_tokens = out_channels, num_register_tokens
            self.rope_dim = int(dim_head * rope_dim_ratio)
            self.rope_theta = rope_theta
            self.x_embedder = nn.Linear(in_channels, dim, bias=True)
            self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, dim))
            #: ⚠️ 推理**不用**它 ✗ —— 留着只是为了 ckpt 能整体装进来 ✓（前向喂的是**全零** token ✓）
            self.register_buffer("mask_token", torch.empty(1, 1, dim))
            self.transformer_blocks = nn.ModuleList(
                [_TransformerBlock(dim, heads, dim_head, eps) for _ in range(num_layers)])
            self.norm_out = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
            self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size,
                                      bias=True)
            self.heads, self.dim_head = heads, dim_head

        def forward(self, x: Any) -> Any:
            batch, _, latent_t, latent_h, latent_w = x.shape
            h = self.x_embedder(x.flatten(2).transpose(1, 2))
            num_patches = h.shape[1]
            num_suffix = 1 + self.num_register_tokens
            h = torch.cat([h, _cast(self.register_tokens, h).expand(batch, -1, -1),
                           torch.zeros_like(h[:, 0:1, :])], dim=1)
            ids = _token_ids((latent_t, latent_h, latent_w), dtype=x.dtype, device=x.device)
            ids = ids.expand(batch, -1, -1)
            suffix_ids = torch.zeros((batch, num_suffix, 3), device=x.device, dtype=ids.dtype)
            table = _rope_table(self.rope_dim, self.rope_theta, 3, torch.cat([ids, suffix_ids], 1))
            for block in self.transformer_blocks:
                h = block(h, table)
            out = self.proj_out(self.norm_out(h))[:, :num_patches, :]
            out = out.view(batch, latent_t, latent_h, latent_w, self.out_channels,
                           self.patch_size_t, self.patch_size, self.patch_size)
            out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
            return out.reshape(batch, self.out_channels, latent_t * self.patch_size_t,
                               latent_h * self.patch_size, latent_w * self.patch_size)

    class _VideoVAE(nn.Module):
        """**真**视频 VAE ✓ = 3D 因果 CNN 编码器 ✓ + ViT3D 解码器 ✓ + 两个 1×1 Conv ✓。

        ⚠️ 对外**不**走 ``forward`` ✗ —— 用 :meth:`encode` / :meth:`decode` ✓：只有它们才是完整的
        「像素 ⇄ 潜变量」口径 ✓（ImageNet 归一化 ✓ / 潜变量 mean-std 归一化 ✓ / 空间分块 ✓ /
        时间叠合 ✓ / 出 [0,1] 的 fp32 像素 ✓ 全在里面 ✓）。

        ⚠️ 常量 ``_LATENTS_MEAN`` 等在**模块末尾** ✓（本函数执行时不需要它们 ✓，方法体是**运行期**才取 ✓）。
        """

        def __init__(self, config: H3VideoVAEConfig) -> None:
            super().__init__()
            self.config = config
            self.vae_ratio = config.spatial_scale
            self.vae_ratio_t = config.temporal_scale
            self.clip_length = config.clip_length
            self.token_drop = config.token_drop
            self.frame_pre_padding = config.frame_pre_padding
            self.tokens_chunk_size = config.token_chunk
            self.token_overlap = config.token_overlap
            self.frame_overlap = config.frame_overlap
            self.tiling = bool(config.tiling)
            self.tile_size = int(config.tile_size)
            self.tile_overlap_min = int(config.tile_overlap_min)
            z_channels = config.latent_channels
            self.encoder = _EncoderFCN3D(z_channels=z_channels, double_z=True)
            self.quant_conv = nn.Conv3d(2 * z_channels, 2 * z_channels, 1)
            self.post_quant_conv = nn.Conv3d(z_channels, z_channels, 1)
            self.decoder = _ViT3DDecoder(patch_size=self.vae_ratio, patch_size_t=self.vae_ratio_t,
                                         in_channels=z_channels, out_channels=3)
            #: 出厂常量 ✓（**真值也在 ckpt 里** ✓ ⇒ 装载时会被覆盖 ✓；这里先放着只是让结构自洽 ✓）
            self.register_buffer("latents_mean", torch.tensor(_LATENTS_MEAN, dtype=torch.float32))
            self.register_buffer("latents_std", torch.tensor(_LATENTS_STD, dtype=torch.float32))
            #: ⚠️ ``persistent=False`` 是**必须的** ✗：ckpt 里**没有** pixel_mean/pixel_std ✓
            #: （它们只是 ImageNet 常量 ✓）—— 用默认的 ``True`` ⇒ 装载期立刻报「缺 2 个键」✗✓。
            self.register_buffer("pixel_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1, 1),
                                 persistent=False)
            self.register_buffer("pixel_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1, 1),
                                 persistent=False)

        @property
        def weight_device(self) -> Any:
            """参数所在设备 ✓（调用方不必自己记 ✓；没有参数 ⇒ ``cpu`` ✓ 不抛错 ✓）。"""
            try:
                return next(self.parameters()).device
            except StopIteration:  # pragma: no cover - 本结构不可能没有参数 ✓
                return torch.device("cpu")

        # ── 像素归一化（ImageNet ✓）与潜变量归一化（mean/std ✓）是**两件事** ✗ 别混 ✓

        def _normalize_pixels(self, x: Any) -> Any:
            """``[-1,1]`` 像素 ⇒ ImageNet 归一化 ✓（编码器**只认**这一路 ✓）。"""
            return x.add(1.0).mul_(0.5).sub_(self.pixel_mean.to(x)).div_(self.pixel_std.to(x))

        def _finalize_pixels(self, part: Any) -> Any:
            """解码器原始输出 ⇒ **fp32 / [0,1]** 像素 ✓（出片侧要的就是这个 ✓）。"""
            part = part * self.pixel_std.to(device=part.device, dtype=torch.float32)
            return part.add_(self.pixel_mean.to(device=part.device,
                                                dtype=torch.float32)).clamp_(0.0, 1.0)

        def _latents_stats(self, like: Any) -> tuple[Any, Any]:
            """``latents_mean/std`` 的 ``[1,C,1,1,1]`` 视图 ✓（对齐 ``like`` 的 dtype+device ✓）。"""
            return (self.latents_mean.view(1, -1, 1, 1, 1).to(like),
                    self.latents_std.view(1, -1, 1, 1, 1).to(like))

        # ── 单趟前向（不分块 ✓）

        def _encode_moments(self, x: Any) -> Any:
            """归一化像素 ⇒ ``quant_conv`` 出来的 **mean‖logvar**（48 通道 ✓）。"""
            return self.quant_conv(self.encoder(x))

        def _decode_pixels(self, z: Any) -> Any:
            """潜变量 ⇒ 原始像素 ✓（**没有**反归一化 ✗ —— 那一步在 :meth:`_finalize_pixels` ✓）。"""
            return self.decoder(self.post_quant_conv(z))

        # ── 空间分块（省显存 ✓；⚠️ **近似** ✗ 不是等价 ✗ —— 边界上「空间 reflect padding ✓」和
        #    「跨块注意力 ✗」都丢了 ✓ ⇒ 叠合只是把缝做得看不出来 ✓ 不是把它消掉 ✓）

        def _adaptive_encode(self, x: Any) -> Any:
            return self.tiled_encode(x) if self.tiling else self._encode_moments(x)

        def _adaptive_decode(self, z: Any) -> Any:
            return self.tiled_decode(z) if self.tiling else self._decode_pixels(z)

        def split_tiles(self, input_len: int) -> tuple[list[int], list[int], list[int]]:
            """一段边长 ⇒ ``(起始, 长度, 重叠)`` ✓。

            ⚠️ 重叠**必须是** ``vae_ratio`` 的整数倍 ✗ —— 不是的话切出来的块在潜空间里**对不齐** ✓✗
            （像素域多 1 行 ⇒ 潜域多 1/16 格 ✓ 只能靠丢/补像素来圆 ✓ ⇒ 画面错位 ✓）。
            构造口径：先按 ``tile_size`` 铺满、再把**多出来的余量**平摊进重叠 ✓（余量也是整除后摊 ✓）。
            """
            tile_size = self.tile_size
            if tile_size >= input_len:
                return [0], [input_len], []
            count = math.ceil(input_len / tile_size)
            while True:
                overlaps = [self.tile_overlap_min] * (count - 1)
                remaining = tile_size * count - sum(overlaps) - input_len
                if remaining < 0:  # 铺不下 ⇒ 再加一块 ✓（多出来的靠重叠平分 ✓）
                    count += 1
                else:
                    break
            for index in range(remaining // self.vae_ratio):
                overlaps[index % (count - 1)] += self.vae_ratio
            starts = [0]
            for index in range(count - 1):
                starts.append(starts[-1] + tile_size - overlaps[index])
            return starts, [tile_size] * count, overlaps

        def blend(self, a: Any, b: Any, blend_extent: int, dim: int) -> Any:
            """``a`` 的**尾巴**与 ``b`` 的**头**按线性权重叠合 ✓（``a`` 1⇒0 ✓、``b`` 0⇒1 ✓）。"""
            blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
            if blend_extent <= 0:
                # ⚠️ 0 长度**不能**走下面的切片 ✓：``[-0:]`` 在 Python 里是**整段** ✗ ⇒ 静默错位 ✓✗
                return b
            positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
            shape = [1] * a.ndim
            shape[dim] = blend_extent
            weight_a = (1 - positions / blend_extent).view(shape)
            weight_b = (positions / blend_extent).view(shape)
            head = [slice(None)] * a.ndim
            head[dim] = slice(-blend_extent, None)
            tail = [slice(None)] * b.ndim
            tail[dim] = slice(0, blend_extent)
            blended = a[tuple(head)] * weight_a + b[tuple(tail)] * weight_b
            if blend_extent < b.shape[dim]:
                rest = [slice(None)] * b.ndim
                rest[dim] = slice(blend_extent, None)
                return torch.cat([blended, b[tuple(rest)]], dim=dim)
            return blended

        def tiled_encode(self, x: Any) -> Any:
            """按 ``tile_size`` 切块编码 ✓，块间**在潜空间**叠合 ✓（重叠恰好是整格 ✓ ⇒ 不丢不补 ✓）。"""
            y_starts, y_lens, y_overlaps = self.split_tiles(x.shape[-2])
            x_starts, x_lens, x_overlaps = self.split_tiles(x.shape[-1])
            rows: list[list[Any]] = []
            for y_start, y_len in zip(y_starts, y_lens):
                row: list[Any] = []
                for x_start, x_len in zip(x_starts, x_lens):
                    tile = x[..., y_start:y_start + y_len, x_start:x_start + x_len]
                    row.append(self._encode_moments(tile))
                rows.append(row)
            y_lat = [overlap // self.vae_ratio for overlap in y_overlaps]
            x_lat = [overlap // self.vae_ratio for overlap in x_overlaps]
            result_rows: list[Any] = []
            for i, row in enumerate(rows):
                result_row: list[Any] = []
                for j, tile in enumerate(row):
                    if i > 0:
                        tile = self.blend(rows[i - 1][j], tile, y_lat[i - 1], dim=-2)
                    if j > 0:
                        tile = self.blend(row[j - 1], tile, x_lat[j - 1], dim=-1)
                    if i < len(rows) - 1:
                        tile = tile[..., :-y_lat[i], :]
                    if j < len(row) - 1:
                        tile = tile[..., :, :-x_lat[j]]
                    result_row.append(tile)
                result_rows.append(torch.cat(result_row, dim=-1))
            return torch.cat(result_rows, dim=-2)

        def tiled_decode(self, z: Any) -> Any:
            """按 ``tile_size`` 切块解码 ✓，块间**在像素域**叠合 ✓（直接写进预分配画布 ✓ 整幅不就位 ✓）。

            ⚠️ 顺序很讲究 ✓：**叠合前**先把"要留给下一块的尾巴"拷出来 ✓ —— 叠完再拷就拷到**混合过**的
            像素了 ✗ ⇒ 接缝会出现"两次曝光" ✓✗。四条尾巴（下 / 右）各走各的 ✓。
            """
            height = z.shape[-2] * self.vae_ratio
            width = z.shape[-1] * self.vae_ratio
            y_starts, y_lens, y_overlaps = self.split_tiles(height)
            x_starts, x_lens, x_overlaps = self.split_tiles(width)
            canvas = None
            tile = None
            row_tails: list[Any] = []
            out_y = 0
            for i, (y_start, y_len) in enumerate(zip(y_starts, y_lens)):
                zy, zl = y_start // self.vae_ratio, y_len // self.vae_ratio
                new_tails: list[Any] = []
                left_tail = None
                out_x = 0
                for j, (x_start, x_len) in enumerate(zip(x_starts, x_lens)):
                    zx, zw = x_start // self.vae_ratio, x_len // self.vae_ratio
                    tile = self._decode_pixels(z[..., zy:zy + zl, zx:zx + zw])
                    if i < len(y_starts) - 1:
                        new_tails.append(tile[..., -y_overlaps[i]:, :].clone())
                    next_left = (tile[..., :, -x_overlaps[j]:].clone()
                                 if j < len(x_starts) - 1 else None)
                    if i > 0:
                        tile = self.blend(row_tails[j], tile, y_overlaps[i - 1], dim=-2)
                    if j > 0:
                        tile = self.blend(left_tail, tile, x_overlaps[j - 1], dim=-1)
                    left_tail = next_left
                    if i < len(y_starts) - 1:
                        tile = tile[..., :-y_overlaps[i], :]
                    if j < len(x_starts) - 1:
                        tile = tile[..., :, :-x_overlaps[j]]
                    if canvas is None:
                        canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype,
                                             device=tile.device)
                    canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
                    out_x += tile.shape[-1]
                row_tails = new_tails
                out_y += tile.shape[-2]
            return canvas

        # ── 时间分块（编码按 17 帧一段 ✓ / 解码按 5 潜帧一段 ✓ —— 口径见 H3_VIDEO_VAE_ARCH_FACTS ✓）

        def encode_temporal(self, x: Any, device: Any) -> Any:
            """``[B,3,T,H,W]``（**[-1,1]** 像素 ✓）⇒ 拼接后的 **mean‖logvar** ✓（编码侧分段 ✓）。

            ⚠️ 两处"看起来可以省、其实不能"的 ✗：
            ① 末段**不满 17 帧要重复末帧补齐** ✓（**不是**补零 ✗ —— 补零等于给模型喂黑帧 ✓
               最后一段的统计会整体偏掉 ✓）；
            ② 补出来的多余潜帧**最后整段丢掉** ✓（``token_drop=3`` ✓ —— 它们是从重复帧"编"出来的 ✓
               不是真信息 ✓，留着会让解码器看到 3 个重复格 ✓✗）。
            """
            latents = []
            for index in range(math.ceil(x.shape[2] / self.clip_length)):
                clip = x[:, :, index * self.clip_length:(index + 1) * self.clip_length, :, :]
                if clip.shape[2] < self.clip_length:
                    pad = clip[:, :, -1:].repeat(1, 1, self.clip_length - clip.shape[2], 1, 1)
                    clip = torch.cat([clip, pad], dim=2)
                latents.append(self._adaptive_encode(self._normalize_pixels(clip.to(device))))
            out = torch.cat(latents, dim=2)
            if self.token_drop > 0:
                out = out[:, :, :-self.token_drop]
            return out

        def decode_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, int, int, int, int]:
            """潜变量形状 ⇒ **出片形状** ✓（帧数走 :func:`frames_for_latents` ✓ —— 同一套分段算式 ✓
            不是另写一条闭式 ✗，那样两边**一定会漂** ✓✗）。"""
            batch, _channels, latents, height, width = input_shape
            return (batch, 3, frames_for_latents(latents, clip_length=self.clip_length,
                                                 temporal_scale=self.vae_ratio_t,
                                                 token_drop=self.token_drop),
                    height * self.vae_ratio, width * self.vae_ratio)

        def _write_part(self, canvas: Any, part: Any, write_pos: int) -> int:
            """一段像素写进画布 ✓：先反归一化成 ``[0,1]`` ✓；越界部分**截断** ✓（分段算式会多算几帧 ✓）。"""
            if part.shape[2] <= 0:
                return write_pos
            part = self._finalize_pixels(part)
            count = min(part.shape[2], max(0, canvas.shape[2] - write_pos))
            if count > 0:
                canvas[:, :, write_pos:write_pos + count, :, :].copy_(part[:, :, :count, :, :])
            return write_pos + count

        def decode_temporal(self, z: Any) -> Any:
            """``[B,24,T_L,H_L,W_L]``（**已反归一化** ✓）⇒ fp32 像素画布 ``[B,3,T,H,W]`` ✓。

            每段多带 ``token_overlap`` 个潜帧 ✓、段间在像素域叠 ``frame_overlap`` 帧 ✓ ——
            这就是为什么"干脆不分段、整段一次解码"会得到接缝 ✗：因果解码每段开头都要**重新起步** ✓，
            起步那一小段并不等于"接在上一段后面"✓。

            ⚠️ 画布落在 ``config.output_device`` ✓（默认 **CPU** ✓）：整段视频**不进显存** ✗，
            而且出片本来就是 CPU 侧写盘 ✓。⚠️ 画布尺寸必须按**补 padding 之前**的 ``z`` 算 ✓
            （:meth:`decode_output_shape` 内部自己会重算 padding ✓，两边同一套算式 ✓）。
            """
            canvas = torch.empty(self.decode_output_shape(z.shape), dtype=torch.float32,
                                 device=self.config.output_device)
            plan = decode_temporal_plan(z.shape[2], clip_length=self.clip_length,
                                        temporal_scale=self.vae_ratio_t, token_drop=self.token_drop)
            if plan["padTokens"] > 0:
                pad = z[:, :, -1:].repeat(1, 1, plan["padTokens"], 1, 1)
                z = torch.cat([z, pad], dim=2)
            chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
            splits = (1 if self.token_drop > 0 else 0) + 1
            overlap_pixels = None
            write_pos = 0
            for index in range(plan["chunks"]):
                start = index * self.tokens_chunk_size
                clip = z[:, :, start:start + self.tokens_chunk_size + self.token_overlap, :, :]
                clip_dec = self._adaptive_decode(clip)
                for split in range(splits):
                    f_start = split * chunk_dec
                    f_end = min(f_start + chunk_dec, clip_dec.shape[2])
                    part = clip_dec[:, :, f_start:f_end, :, :]
                    # 每段开头都要丢掉 frame_pre_padding 帧 ✓（那几帧对应的潜格是**上一段**已经出过的 ✓）
                    part = part[:, :, self.frame_pre_padding:, :, :]
                    if split == 0:
                        if overlap_pixels is not None:
                            part = self.blend(overlap_pixels, part, self.frame_overlap, dim=-3)
                            overlap_pixels = None
                        write_pos = self._write_part(canvas, part, write_pos)
                    else:  # 第二半先**扣住** ✓：它是叠合用的尾巴 ✓ 不是最终像素 ✓
                        overlap_pixels = part.contiguous()
                if index == plan["chunks"] - 1 and overlap_pixels is not None:
                    write_pos = self._write_part(canvas, overlap_pixels, write_pos)
                    overlap_pixels = None
                del clip_dec, clip
            return canvas

        # ── 对外两个入口（**完整口径** ✓）

        def encode(self, x: Any, device: Any = None) -> Any:
            """``[B,3,T,H,W]``（**[-1,1]** 像素 ✓）⇒ **归一化**潜变量 ``[B,24,T_L,H/16,W/16]`` ✓。

            ⚠️ 只取 ``mean`` ✓、**不用 logvar** ✗（推理是确定性的 ✓ —— 取后验均值 ✓ 不采样 ✓）。
            ⚠️ 单帧走**另一条路** ✓（不分段 ✓ 只保留最后一个潜格 ✓）：分段那条会重复末帧凑满 17 ✓✗。
            """
            if x.ndim == 4:
                x = x.unsqueeze(2)
            device = self.weight_device if device is None else device
            if x.shape[2] == 1:
                moments = self._adaptive_encode(self._normalize_pixels(x.to(device)))[:, :, -1:, :, :]
            else:
                moments = self.encode_temporal(x, device)
            mean = torch.chunk(moments.float(), 2, dim=1)[0]
            latents_mean, latents_std = self._latents_stats(mean)
            return (mean - latents_mean) / latents_std

        def decode(self, z: Any) -> Any:
            """**归一化**潜变量 ⇒ fp32 像素 ``[B,3,T,H,W]``（**[0,1]** ✓ 落在 ``output_device`` ✓）。"""
            z = z.to(self.weight_device)
            latents_mean, latents_std = self._latents_stats(z)
            z = z * latents_std + latents_mean
            if z.shape[2] == 1:
                return self._finalize_pixels(self._adaptive_decode(z)[:, :, -1:, :, :])
            return self.decode_temporal(z)

    return _Impl(_VideoVAE)

# ─────────────────────────────────────────────────────────────────────────────
# 常量（**从 ckpt 抄下来** ✓：fp16 可精确表示 ✓ 一位不差 ✓）
# ⚠️ ckpt 里**也有一份** ✓ 且装载时**覆盖**这里 ✓ ⇒ 它们只是让结构先自洽 ✓、
#    顺便让"没装权重"这件事**一眼看得出来**（数值不对 ✓）；本模块**不支持**不装权重的用法 ✗。
# ─────────────────────────────────────────────────────────────────────────────

#: 潜变量归一化：``latents = (mean ‖ logvar) 的 mean`` ⇒ ``(latents - mean) / std`` ✓
_LATENTS_MEAN: tuple[float, ...] = (
    0.85791015625, -0.96044921875, 1.06640625, -0.5087890625, -0.272705078125, -1.3671875,
    -0.25537109375, -0.26904296875, -0.53759765625, -0.046417236328125, 0.66552734375,
    0.1968994140625, -0.5458984375, -0.403564453125, -0.23681640625, 0.25927734375,
    -0.30126953125, 0.2113037109375, -1.12109375, 0.358154296875, -0.042236328125,
    0.260498046875, 0.2286376953125, 0.70556640625,
)
_LATENTS_STD: tuple[float, ...] = (
    1.22265625, 1.2763671875, 1.68359375, 1.7548828125, 1.5634765625, 2.193359375,
    0.96533203125, 1.056640625, 0.841796875, 0.77294921875, 1.8955078125, 0.94677734375,
    0.7998046875, 0.449951171875, 0.7197265625, 0.69384765625, 2.9609375, 2.76953125,
    3.048828125, 2.109375, 3.275390625, 3.162109375, 2.28125, 2.61328125,
)
#: ⚠️ ImageNet 常量 **不在** ckpt 里 ✓ —— 它们若被注册成 ``persistent=True`` 的 buffer，
#: 装载期会报「缺 pixel_mean / pixel_std」✗✓（结构那边已经设成 False ✓，这里只是把数写清楚 ✓）。
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: safetensors 的 dtype 名 ⇒ torch dtype ✓（只列 ``torch.frombuffer`` **收得下**的那些 ✓；
#: fp8 那两种**故意不收** ✗ —— 它得先反量化 ✓，直接 ``frombuffer`` 会变成"没尺度的字节" ✓✗）
_TORCH_DTYPES: dict[str, str] = {
    "BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "U16": "uint16",
    "F16": "float16", "BF16": "bfloat16", "I32": "int32", "U32": "uint32",
    "F32": "float32", "I64": "int64", "U64": "uint64", "F64": "float64",
}


# ─────────────────────────────────────────────────────────────────────────────
# 时间几何（**公开算式** ✓）：像素帧数 ⇄ 潜帧数
# ⚠️ 两个方向**必须共用同一套分段算式** ✗ —— 一边写算式、一边写"闭式" ⇒ 两边迟早会漂 ✓✗
# ─────────────────────────────────────────────────────────────────────────────


def latents_for_frames(frames: int, *, clip_length: int = 17, temporal_scale: int = 4,
                       token_drop: int = 3) -> int:
    """像素帧数 ⇒ 潜帧数（**编码侧口径** ✓）。

    每 ``clip_length``（17 ✓）帧编出 ``ceil(17/4) = 5`` 个潜帧 ✓，最后**整段丢掉** ``token_drop``（3 ✓）
    ⇒ ``5 * ceil(T/17) - 3`` ✓。单帧是特例 ✓（走单帧那条路 ✓ 只留 1 个潜格 ✓）。
    """
    if frames <= 1:
        return 1
    per_clip = -(-clip_length // temporal_scale)
    return -(-frames // clip_length) * per_clip - token_drop


def decode_temporal_plan(latents: int, *, clip_length: int = 17, temporal_scale: int = 4,
                         token_drop: int = 3) -> dict[str, int]:
    """潜帧数 ⇒ 解码分段计划 ✓ ``{"padTokens", "chunks"}`` ✓（**与 :meth:`decode_temporal` 同源** ✓）。

    补 padding 的口径很绕 ✓，但它是**闭环**的 ✓：先按 ``token_drop`` 把潜帧数补成 ``5`` 的整数倍 ✓
    （末尾**重复最后一格** ✓ 不是补零 ✗），段数再减 1 ✓ —— 因为最后那一段在解码时是**叠合用的尾巴** ✓
    （它对应的像素已经由上一段出过 ✓）。
    """
    per_clip = -(-clip_length // temporal_scale)
    pad_tokens = (-(latents + token_drop)) % per_clip
    total = latents + token_drop + pad_tokens
    chunks = total // per_clip - (1 if token_drop > 0 else 0)
    if chunks < 1:
        pad_tokens += per_clip
        chunks += 1
    return {"padTokens": pad_tokens, "chunks": chunks}


def _decode_temporal_pad_frames(z_len: int, pad_tokens: int, *, clip_length: int = 17,
                                temporal_scale: int = 4, token_drop: int = 3) -> int:
    """补出来的那些潜格，**一共会多出几帧像素** ✓（要从总帧数里减掉 ✓）。

    ⚠️ ``z_len`` 是**补完 padding 之后**的潜帧数 ✗（真数 = ``z_len - pad_tokens`` ✓）—— 传错这一个参数
    不会报错 ✗、只会让帧数差几帧 ✓✗（上层按需求裁剪 ⇒ 症状是"尾帧被削掉"✓ 很难追 ✓）。

    ``clip_length % temporal_scale == 0`` 时每个补格都是满 4 帧 ✓；有余数（这里是 ``17%4=1`` ✓）时，
    补格里**正好对在片尾**的那一格只值 1 帧 ✓ —— 不区分就会每格多算 3 帧 ✓✗。
    """
    if pad_tokens <= 0:
        return 0
    intra_tail = clip_length % temporal_scale
    if intra_tail == 0:
        return pad_tokens * temporal_scale
    before = z_len - pad_tokens
    per_clip = -(-clip_length // temporal_scale)
    return sum(intra_tail if (before + index) % per_clip == 0 else temporal_scale
               for index in range(pad_tokens))


def frames_for_latents(latents: int, *, clip_length: int = 17, temporal_scale: int = 4,
                       token_drop: int = 3) -> int:
    """潜帧数 ⇒ 像素帧数（**解码侧口径** ✓ —— 就是 :meth:`decode_temporal` **实际会写出来的**那一堆 ✓）。

    ⚠️ 别期待它是 :func:`latents_for_frames` 的严格逆 ✗：编码末尾会丢 3 个潜格 ✓，
    所以 ``frames_for_latents(latents_for_frames(T)) == T`` 只对 ``T = 17k+5`` 成立 ✓（这也正是
    节点口径里"帧数 = 5 + 17k"的由来 ✓）。其余长度解码会**多出几帧** ✓，由上层按需求裁剪 ✓。
    """
    if latents <= 1:
        return 1
    per_clip = -(-clip_length // temporal_scale)
    chunk_dec = per_clip * temporal_scale
    token_overlap = (-token_drop) % per_clip
    splits = (1 if token_drop > 0 else 0) + 1
    frame_pre_padding = (-clip_length) % temporal_scale
    plan = decode_temporal_plan(latents, clip_length=clip_length, temporal_scale=temporal_scale,
                                token_drop=token_drop)
    z_len = latents + plan["padTokens"]
    total = 0
    tail = 0
    for index in range(plan["chunks"]):
        start = index * per_clip
        end = start + per_clip + token_overlap
        clip_tokens = max(0, min(end, z_len) - min(start, z_len))
        clip_frames = clip_tokens * temporal_scale
        for split in range(splits):
            f_start = split * chunk_dec
            f_end = min(f_start + chunk_dec, clip_frames)
            got = max(0, f_end - f_start - frame_pre_padding)
            if split == 0:
                total += got
            else:
                tail = got
    total += tail
    return total - _decode_temporal_pad_frames(z_len, plan["padTokens"], clip_length=clip_length,
                                               temporal_scale=temporal_scale, token_drop=token_drop)


# ─────────────────────────────────────────────────────────────────────────────
# 装载与体检
# ─────────────────────────────────────────────────────────────────────────────

#: 清单（``configs/models.json``）里"视频 VAE"的候选键 ✓ —— **按优先级** ✓（官方 fp16 在前 ✓）
H3_VIDEO_VAE_WEIGHT_KEYS: tuple[str, ...] = ("vae_video_fp16", "vae_video_fp8mix")


def _require_torch() -> Any:
    """torch 在本仓是**可选**依赖 ✓ —— 真要用时才报「怎么装」✗，**不在**导入期炸 ✓。"""
    try:
        import torch  # noqa: PLC0415 —— 故意延迟 ✓
    except ImportError as err:  # pragma: no cover - 取决于环境 ✓
        raise H3VideoVAEError(f"算真视频 VAE 需要 torch ✗（{err}）") from err
    return torch


def _resolve_torch_dtype(torch: Any, dtype: str | None, device: str) -> Any:
    """``dtype=None`` ⇒ **按设备选** ✓：cuda 用 fp16 ✓（权重本来就是 fp16 ✓）、cpu 用 fp32 ✓。

    ⚠️ CPU 上**别**用 fp16 ✗：PyTorch 的 CPU fp16 卷积/attention 会退化成很慢的实现 ✓✗，
    而且真 VAE 的 fp16 权重转 fp32 是**无损**的 ✓（数值只多不少 ✓）。
    """
    if dtype:
        resolved = getattr(torch, str(dtype), None)
        if resolved is None:
            raise H3VideoVAEError(f"不认识的 dtype {dtype!r} ✗（要 torch 的属性名，如 'float16' ✓）")
        return resolved
    return torch.float16 if str(device).startswith(("cuda", "mps")) else torch.float32


def read_state_dict(path: str | Path, *, dtype: str | None = None, device: str = "cpu",
                    info: Any = None) -> dict[str, Any]:
    """把 safetensors **整份读成 torch 张量** ✓（单文件 ✓ 总 5.2 GiB ✓ 一次读完 ✓）。

    ⚠️ 为什么不用 :func:`safetensors.read_tensor_bytes` ✗：它按张量**限 64 MiB** ✓（防手滑 ✓），
    而解码器最大的那两个张量各 **67 MiB** ✓（``ff.w1`` = 16384×2048 fp16 ✓）⇒ 会被自家护栏拦下 ✓✗。
    这里按头部已经声明好的区间读 ✓，不猜偏移 ✓。
    """
    torch = _require_torch()
    target = Path(path)
    info = safetensors_mod.inspect(target) if info is None else info
    if not info.ok:
        raise H3VideoVAEError(
            f"权重体检不过 ✗：{target.name} —— {'；'.join(info.problems[:4])}")
    base = 8 + info.header_bytes
    resolved = _resolve_torch_dtype(torch, dtype, device)
    state: dict[str, Any] = {}
    with target.open("rb") as handle:
        for name, entry in info.tensors.items():
            torch_dtype = _TORCH_DTYPES.get(entry.dtype)
            if torch_dtype is None:
                raise H3VideoVAEError(
                    f"张量 {name!r} 的 dtype ``{entry.dtype}`` 不能直接 ``frombuffer`` ✗ "
                    f"（低精度权重得**先反量化** ✓ —— 那是 :mod:`.quant` 的活儿 ✓，不在这里猜 ✗）")
            handle.seek(base + entry.begin)
            raw = handle.read(entry.nbytes)
            if len(raw) != entry.nbytes:
                raise H3VideoVAEError(
                    f"读 {name!r} 只拿到 {len(raw)}/{entry.nbytes} 字节 ⇒ 文件被截断 ✗")
            tensor = torch.frombuffer(bytearray(raw), dtype=getattr(torch, torch_dtype))
            state[name] = tensor.reshape(entry.shape).to(device=device, dtype=resolved)
    return state


def h3_video_vae_weights_on_disk(*, root: str | Path | None = None) -> dict[str, Any]:
    """本机哪里能找到**真视频 VAE** ✓（走清单 ✓ 不写死路径 ✓）。

    用途是**把原因说出来** ✗：装不起来时，先说清楚「清单里没有这个键 / 期望落在哪 / 盘上有没有」✓
    —— ⚠️ ``present=False`` 是**结论** ✓，不是"没查" ✗（这条纪律全仓一致 ✓）。
    """
    from . import inventory as inventory_mod  # noqa: PLC0415 —— 只有这条查询路需要清单层 ✓

    catalog = inventory_mod.load_catalog()
    entries: dict[str, Any] = {str(item.get("key")): item for item in catalog["models"]}
    resolved_root = Path(root) if root is not None else None
    candidates: list[dict[str, Any]] = []
    for key in H3_VIDEO_VAE_WEIGHT_KEYS:
        entry = entries.get(key)
        if entry is None:
            candidates.append({"key": key, "path": None, "present": False, "bytes": 0,
                               "note": "清单里没有这个键 ✗"})
            continue
        # ⚠️ **同一条解析** ✗（2026-09-27 修 ✓）：给了 ``root`` ⇒ 只认那个根 ✓；否则清单落点
        #    （``models_dir``）优先 → 动态探测根 ✓（= 引擎加载与体检用的那条 ✓）。
        #    以前只看 ``models_dir`` ✗ ⇒ 权重在探测根里也会被报成"不在盘上" ✓✗（同一件事两个说法 ✓）。
        if resolved_root is not None:
            found: Path | None = inventory_mod.component_path(entry, resolved_root)
        else:
            found = inventory_mod.resolve_component(entry)[0]
        present = bool(found is not None and found.is_file())
        candidates.append({
            "key": key, "path": None if found is None else str(found), "present": present,
            "bytes": found.stat().st_size if present and found is not None else 0,
            "required": bool(entry.get("required")),
            "expectedGiB": entry.get("size_gib"),
        })
    best = next((item for item in candidates if item["present"]), None)
    notes: list[str] = []
    if catalog.get("error"):
        notes.append(f"清单读不了：{catalog['error']}")
    if best is None:
        notes.append("盘上**没有**任何候选视频 VAE ✓ —— 期望落点见 candidates 的 path ✓"
                     "（官方键 " + " / ".join(f"`{k}`" for k in H3_VIDEO_VAE_WEIGHT_KEYS) + " ✓）")
    else:
        notes.append(f"用 `{best['key']}` ✓；⚠️ 社区 fp8mix 版**要先反量化** ✗ "
                     f"（本模块只认 fp16/bf16/fp32 这类能直接读的 ✓）")
    return {"key": None if best is None else best["key"],
            "path": None if best is None else best["path"],
            "candidates": candidates, "catalogError": catalog.get("error"), "notes": notes}


def build_h3_video_vae(config: H3VideoVAEConfig | None = None, **overrides: Any) -> Any:
    """装**真**视频 VAE ✓ —— 装不满就 :class:`H3VideoVAEError` ✗（**不返回"半个模型"** ✓）。

    返回的模块上额外挂了 ``loadReport`` ✓（:class:`~app.services.engine.weights.WeightLoadReport` ✓）：
    ``loadReport.complete=False`` 就是"这不是能用的模型" ✓ —— 本仓的纪律是**说出来** ✗ 不是沉默 ✓。
    """
    from . import weights as weights_mod  # noqa: PLC0415 —— 只有这条路要用 torch 侧 ✓

    torch = _require_torch()
    resolved_config = H3VideoVAEConfig() if config is None else config
    if overrides:
        resolved_config = replace(resolved_config, **overrides)
    if not resolved_config.weights:
        raise H3VideoVAEError(
            "没给权重路径 ✗ —— 随机初始化的\"VAE\"只会骗自己 ✓✗"
            f"（不知道路径可以先问 :func:`h3_video_vae_weights_on_disk` ✓）")

    info = safetensors_mod.inspect(resolved_config.weights)
    if not info.ok:
        raise H3VideoVAEError(
            f"权重体检不过 ✗：{Path(resolved_config.weights).name} —— "
            f"{'；'.join(info.problems[:4])}")

    dtype = _resolve_torch_dtype(torch, resolved_config.dtype, resolved_config.device)
    module = _impl().VideoVAE(resolved_config)
    # ⚠️ 先 ``to(dtype)`` **再** 装载 ✗：反过来的话张量会被**塞进** fp32 参数里 ✓（值对 ✓ 但显存白占一倍 ✗）
    module.to(device=resolved_config.device, dtype=dtype)
    state = read_state_dict(resolved_config.weights, dtype=resolved_config.dtype,
                            device=resolved_config.device, info=info)

    report = weights_mod.diff_state_dict(module, state)
    report.path = str(resolved_config.weights)
    report.dtypeCasts = {str(dtype): len(state)}
    if not report.complete:
        summary = (f"真视频 VAE 装不满 ✗：missing={len(report.missing)} / "
                   f"shapeMismatch={len(report.shapeMismatch)} / unexpected={len(report.unexpected)}")
        if resolved_config.strict:
            raise H3VideoVAEError(
                f"{summary} ⇒ 缺的前几个 {report.missing[:6]} ✓、形状不符 {report.shapeMismatch[:3]} ✓"
                f"（`strict=False` 可以硬装 ✓，但那样出来的**不是**真 VAE ✗）")
        warnings.warn(f"{summary} ⇒ strict=False 继续 ✓，但产物**不可信** ✗✗", stacklevel=2)

    module.load_state_dict(state, strict=False)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.loadReport = report
    return module


def verify_latents_stats(latents: Any, *, tolerance: float = 0.75) -> dict[str, Any]:
    """潜变量的**量级体检** ✓：逐通道 ``mean/std`` ⇄ 出厂归一化常量 ✓。

    怎么读这份报告 ✓：潜变量**反归一化之后**（= ``latents * std + mean`` ✓）应当逐通道接近
    ``latents_mean / latents_std`` ✓ —— 那本来就是**训练时用的**归一化 ✓ ⇒ 差得离谱说明
    「这份潜变量不是这个 VAE 该有的东西」✓（多半是别处不对：没装真权重 ✗ / 条件还是噪声 ✗ / 拼接维度错了 ✗）。

    ⚠️ 这是**量级核对** ✗ 不是逐位对照 ✗：``ok=True`` **不能**证明对 ✓，``ok=False`` 也只是**可疑** ✓
    （短视频的逐通道统计本来就抖 ✓）。容差是工程经验值 ✓，要更严就调 ``tolerance`` ✓。
    """
    torch = _require_torch()
    if getattr(latents, "ndim", 0) != 5:
        raise H3VideoVAEError(
            f"潜变量形状必须是 ``[B,C,T,H,W]`` ✗（收到 {tuple(getattr(latents, 'shape', ()))} ✓）")
    channels = int(latents.shape[1])
    if channels != len(_LATENTS_MEAN):
        raise H3VideoVAEError(
            f"通道数 {channels} ≠ 真 VAE 的 {len(_LATENTS_MEAN)} ✗ ⇒ 这不是它的潜变量 ✓")
    ref_mean = torch.tensor(_LATENTS_MEAN, dtype=torch.float32)
    ref_std = torch.tensor(_LATENTS_STD, dtype=torch.float32)
    view = (1, -1, 1, 1, 1)
    raw = latents.detach().to(device="cpu", dtype=torch.float32)
    raw = raw * ref_std.view(view) + ref_mean.view(view)
    per_mean = raw.mean(dim=(0, 2, 3, 4))
    per_std = raw.std(dim=(0, 2, 3, 4), unbiased=False)
    mean_dev = float((per_mean - ref_mean).abs().max()) if channels else 0.0
    std_dev = float(((per_std - ref_std).abs() / ref_std.clamp_min(1e-6)).max()) if channels else 0.0
    ok = mean_dev <= tolerance and std_dev <= tolerance
    return {
        "ok": ok, "tolerance": tolerance, "channels": channels,
        "frames": int(latents.shape[2]),
        "maxAbsMeanDev": round(mean_dev, 6), "maxRelStdDev": round(std_dev, 6),
        "referenceMean": [float(v) for v in ref_mean],
        "referenceStd": [float(v) for v in ref_std],
        "observedMean": [round(float(v), 6) for v in per_mean],
        "observedStd": [round(float(v), 6) for v in per_std],
        "note": ("量级核对 ✓ 不是逐位对照 ✗：ok=True 不证明对 ✓、ok=False 只代表**可疑** ✓"
                 "（短视频的逐通道统计本来就抖 ✓）"),
    }
