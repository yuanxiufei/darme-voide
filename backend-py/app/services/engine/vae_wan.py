"""Wan 系 VAE 解码骨架 —— 先接住 Qwen-Image **2.1** 的 64 通道 VAE ✓。

盘上 ``vae/qwen_image_2.1_vae_bf16.safetensors`` 的 238 键与 comfy
``comfy/ldm/wan/vae2_2.py::WanVAE`` 逐键同名 ✓（``tests/vae_wan_ref.py`` 用原文
``load_state_dict`` 装过，0 缺 0 多 ✓）。comfy ``sd.py`` 里 2.1 的分支写着
「Wan 2.2 layout, temporal kernel 1, no patchify, RGBA」，参数全部从权重形状读：

- ``dim`` = ``encoder.conv1.shape[0]`` = 96；``dec_dim`` = ``decoder.head.0.gamma`` = 144
- ``z_dim`` = ``conv2.shape[0]`` = 64（``conv1`` 是 [128,128] = 2×64 ✓）
- ``dim_mult`` = (1,2,4,8,8)；``num_res_blocks`` = 2；``temperal_downsample`` = (F,T,T,T)
- ``image_channels`` = 4（**RGBA** ✓）；``patch_size`` = 1；``temporal_kernel`` = 1

## 只做单帧解码 ✓ —— 这一族本来就没有时间维 ✓

``temporal_kernel == 1`` ⇒ 所有 ``CausalConv3d`` 的 ``padding[0]`` 都是 0、因果补齐是空
操作 ✓；``time_conv`` 在无 ``feat_cache`` 的单帧路径上不执行 ✓。唯一真需要时间维的是
``AvgDown3D`` / ``DupUp3D`` 这两个无参重排（t=1 时前者补 1 帧再平均、后者复制后让
``first_chunk`` 切掉第一帧 ✓），本文件照实现 ✓。

**留白**：``encode()`` / ``feat_cache`` 流式 / comfy 的 ``strip_apply`` 行分块
（后者是省显存的等价改写，2048×1536 时 decoder 最大中间量约 0.9 GiB/份，卡上装得下 ✓）。

四个「不报错只是图不对」：① ``RMS_norm`` 是 ``F.normalize(x,dim=1)*sqrt(dim)*gamma`` ✓
不是均方根；② ``head`` 出 **4 通道 RGBA**，出图要丢第 4 通道 ✓；③ ``DupUp3D`` 的
``first_chunk`` 切的是**第一帧** ✓；④ **解码前必须反缩放** ✓✗ —— DiT 吃的是**归一化后**的
潜变量 ✓，直接喂 ``decode()`` 形状全对、图是灰噪 ✓、一个错都不报 ✓✗
（``decode_latents()`` 就是那条正路 ✓，见 :func:`latents_stats` ✓）。

## 反缩放的数在哪 ✗✓

**不在权重里** ✗（``qwen_image_2.1_vae_bf16.safetensors`` 的 238 键全是 conv/gamma ✓、
没有任何 ``latents_mean`` / ``latents_std`` ✓）—— 它们是**潜空间口径**的事实 ✓，
在 comfy ``comfy/latent_formats.py`` 的 ``class QwenImage21`` 里 ✓（64 个数逐字抄在
:data:`QWEN21_LATENT_MEAN` / :data:`QWEN21_LATENT_STD` ✓，注释写明行号 ✓）。
⇒ 换算统一走 :mod:`latent_formats` 的 ``process_out("qwen_image21", …)`` ✓（**不在这里
另写一份乘加** ✗：两份数学迟早漂 ✓✗）。

权威来源：``comfy-studio/ComfyUI/comfy/ldm/wan/vae2_2.py``（逐层对应 ✓）
＋ ``comfy-studio/ComfyUI/comfy/latent_formats.py::QwenImage21``（反缩放 ✓）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["WanVaeShape", "FlatBlock", "infer_shape", "WanVAE", "build", "patchify", "unpatchify",
           "latents_stats", "family_title", "FAMILY_TITLES", "decode_latents", "encode_latents",
           "QWEN21_LATENT_MEAN", "QWEN21_LATENT_STD",
    "WAN21_LATENT_MEAN", "WAN21_LATENT_STD", "WAN22_LATENT_MEAN", "WAN22_LATENT_STD",
           "SUGGESTED_ATTN_CHUNK", "SUGGESTED_TILE"]

#: 显存真的不够时**建议**喂给 :meth:`WanVAE.decode` 的两个值 ✓（**默认不生效** ✓✗）。
#: 为什么只是「建议」而不做默认：两条都会改数值（见 :class:`AttentionBlock` / :class:`Decoder3d` ✓），
#: 而本仓对「同 seed 同文本逐像素复现」有硬要求 ✓ ⇒ 只有调用方**明确知道**要换内存、且愿意
#: 接受产物字节变化时才打开 ✓。``attn`` 取 1024：8× 那份从 20.96 GiB 降到 3.94 GiB ✓，
#: 再小只省一点、白丢并行度 ✓。``tile`` 取 512（像素）：2.1 的 2048×1536 从 8.69 降到 3.21 GiB ✓。
SUGGESTED_ATTN_CHUNK = 1024
SUGGESTED_TILE = 512


@dataclass(frozen=True)
class FlatBlock:
    """平铺代的**一级** ✓：若干 ``ResidualBlock`` ＋ 可选升/降采样器 ✓（通道照权重逐个记 ✓）。"""

    #: 每个残差块的 ``(in, out)`` ✓（实测同级内会「先升后降」⇒ 不能用单个数概括 ✓）。
    blocks: tuple[tuple[int, int], ...]
    #: 采样器的 ``(in, out)`` ✓ —— **in ≠ out 是常态** ✓（``decoder.upsamples.3.resample.1`` 就是 384→192 ✓）。
    down: tuple[int, int] | None = None
    #: 采样器带不带时间下/上采样（键里有 ``time_conv.weight`` ✓）。
    temporal: bool = False


@dataclass(frozen=True)
class WanVaeShape:
    """从权重形状反推的结构参数 ✓（一个都不写死 ✓）。"""

    dim: int
    dec_dim: int
    z_dim: int
    dim_mult: tuple[int, ...]
    num_res_blocks: int
    temperal_downsample: tuple[bool, ...]
    image_channels: int
    patch_size: int = 1
    temporal_kernel: int = 3
    #: ⭐ **两代结构**（2026-11-01 接 Krea2 时逼出来的 ✓）：
    #: ``False`` = 「每级一个 ``DownResidualBlock`` 外壳」（键形
    #: ``encoder.downsamples.0.downsamples.0.residual.0.gamma`` ✓，2.1 的 64ch 那份 ✓）；
    #: ``True`` = 「全平铺」（键形 ``encoder.downsamples.0.residual.0.gamma`` ＋
    #: 下一格直接是 ``encoder.downsamples.2.resample.1.weight`` ✓，盘上全部 16ch Wan 系
    #: VAE（``qwen_image_vae`` / ``wan_2.1_vae`` / ``Wan2_1_VAE_bf16`` / ``..._comfy``）都是这代 ✓）。
    #: ⚠️ 两代的**参数张量逐个同名同形** ✗ 差别只在**层级** ⇒ 认错代只会报「缺 N 个键」✓，
    #: 看不出是结构代数不对 ✓✗（所以必须由 :func:`infer_shape` 从键名反推 ✓，不许按文件名猜 ✗）。
    flat: bool = False
    #: 平铺代专用的**逐级结构快照** ✓（``flat=False`` 时为空 ✓）。
    #: ⚠️ 为什么不按 ``dim_mult`` 公式反推：平铺代的解码器实测是「384→384 三块 → 升采样 384→**192**」
    #: 之后**下一级首块又从 192 升回 384** ✓（盘上 ``qwen_image_vae`` 逐键如此 ✓）——
    #: 这类「通道先降再升」的排布用公式套不出来 ✗，只能照权重逐块记 ✓。
    encoder_flat: tuple[FlatBlock, ...] = ()
    decoder_flat: tuple[FlatBlock, ...] = ()
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def scale(self) -> int:
        """latent 边长 → 像素边长的倍数 ✓。

        ★ **不是** ``2 ** len(dim_mult)`` ✗：最后一级 ``down_flag=False`` 不降采样，
        所以真降采样次数是 ``len(dim_mult) - 1`` ⇒ 本族 = 2⁴ = **16** ✓
        （与 ``latent_formats.QwenImage21`` 的 ``spacial_downscale_ratio 16`` 对上 ✓）。
        """
        return 2 ** (len(self.dim_mult) - 1)

    @property
    def latent_channels(self) -> int:
        """潜通道数 ✓（别名 ✗：:meth:`TorchBackend._check_attached_vaes` 只认这个名字 ✓）。"""
        return self.z_dim

    @property
    def spatial_scale(self) -> int:
        """空间倍率 ✓（别名 ✗：同上，校验口读的是 ``spatial_scale`` ✓）。"""
        return self.scale

    def to_dict(self) -> dict[str, Any]:
        """给 ``torch_backend.describe()/decode()`` 的报告用 ✓（键名与其它 VAE 配置同口径 ✓）。"""
        return {"name": f"wan-vae/{self.z_dim}ch", "kind": "wan",
                "latentChannels": self.z_dim, "spatialScale": self.scale,
                "dim": self.dim, "decDim": self.dec_dim, "dimMult": list(self.dim_mult),
                "numResBlocks": self.num_res_blocks, "imageChannels": self.image_channels,
                "patchSize": self.patch_size, "temporalKernel": self.temporal_kernel,
                "evidence": list(self.evidence)}


def infer_shape(shapes: dict[str, tuple[int, ...]]) -> WanVaeShape:
    """从 ``{键: 形状}`` 反推 ✓；读不全就抛 ``ValueError`` 并列出缺的键 ✗。"""
    need = ["encoder.conv1.weight", "conv1.weight", "conv2.weight",
            "decoder.conv1.weight", "decoder.head.0.gamma", "decoder.head.2.weight"]
    lack = [k for k in need if k not in shapes]
    if lack:
        raise ValueError(f"这不是 Wan 结构 VAE：缺 {lack} ✗")

    dim = shapes["encoder.conv1.weight"][0]
    temporal_kernel = shapes["encoder.conv1.weight"][2]
    in_total = shapes["encoder.conv1.weight"][1]
    z_dim = shapes["conv2.weight"][0]
    dec_dim = shapes["decoder.head.0.gamma"][0]
    out_total = shapes["decoder.head.2.weight"][0]

    # image_channels 与 patch_size 只能联合解：out_total = c·ps²，in_total = c·ps²·ps²
    patch_size = 1
    image_channels = out_total
    for candidate in (1, 2, 3):
        if out_total % (candidate ** 2) == 0:
            channels = out_total // (candidate ** 2)
            if channels and in_total == channels * candidate ** 4 and channels in (1, 3, 4):
                patch_size, image_channels = candidate, channels
                break

    levels: list[int] = []
    temporal: list[bool] = []
    blocks = 0
    index = 0
    while f"encoder.downsamples.{index}.downsamples.0.residual.2.weight" in shapes:
        out_dim = shapes[f"encoder.downsamples.{index}.downsamples.0.residual.2.weight"][0]
        levels.append(out_dim // dim)
        temporal.append(any(key.startswith(f"encoder.downsamples.{index}.downsamples.")
                            and key.endswith(".time_conv.weight") for key in shapes))
        count = 0
        while f"encoder.downsamples.{index}.downsamples.{count}.residual.0.gamma" in shapes:
            count += 1
        blocks = max(blocks, count)
        index += 1
    #: ⭐ **平铺代**（2026-11-01 加 ✓）：嵌套那套一格都读不到时才试 ✓ ——
    #: 两代的键名**互相不会误命中** ✓（嵌套要求 ``downsamples.i.downsamples.0.residual`` ✓，
    #: 平铺是 ``downsamples.i.residual`` ✓），所以"先嵌套后平铺"不会把两代混着认 ✓。
    flat = False
    encoder_flat: tuple[FlatBlock, ...] = ()
    decoder_flat: tuple[FlatBlock, ...] = ()
    if not levels:
        encoder_flat = _flat_levels_of(shapes, "encoder.downsamples", "downsample")
        if not encoder_flat:
            raise ValueError("读不到 encoder.downsamples ✗（既不是嵌套代、也不是平铺代的 Wan 结构 ✓）")
        decoder_flat = _flat_levels_of(shapes, "decoder.upsamples", "upsample")
        if not decoder_flat:
            raise ValueError("平铺代的编码器认得、解码器一格都读不到 ✗（`decoder.upsamples.*` 缺 ✓）")
        for record in encoder_flat:
            levels.append(record.blocks[-1][1] // dim)      #: 每级**末块**的 out / dim ✓
            temporal.append(record.temporal)
        blocks = max(len(record.blocks) for record in encoder_flat)
        flat = True

    # ★ 只留前 len(levels)-1 项 ✗：最后一级 `down_flag=False` **本来就不降采样**，
    # 它那个「没有 time_conv」不是配置的一部分 ✓。若把它也算进 temperal_downsample，
    # 反转后 `temperal_upsample` 会整体错位一格 ⇒ 该建 `upsample3d` 的块建成了 2d，
    # 装载时表现为 2 个 `time_conv` 键「缺」（模型多建了 ✓ 权重没有 ✓）。
    temporal = temporal[: len(levels) - 1]

    shape_of = ("各级 `downsamples.i.residual.2.weight` 的 out/dim" if flat
                else "各级 `downsamples.i.downsamples.0.residual.2.weight` 的 out/dim")
    return WanVaeShape(
        dim=dim, dec_dim=dec_dim, z_dim=z_dim, dim_mult=tuple(levels),
        num_res_blocks=blocks, temperal_downsample=tuple(temporal),
        image_channels=image_channels, patch_size=patch_size, temporal_kernel=temporal_kernel,
        flat=flat, encoder_flat=encoder_flat, decoder_flat=decoder_flat,
        evidence=(
            f"dim={dim} ← `encoder.conv1.weight`",
            f"z_dim={z_dim} ← `conv2.weight`（`conv1` = {tuple(shapes['conv1.weight'][:2])} = 2×{z_dim} ✓）",
            f"dec_dim={dec_dim} ← `decoder.head.0.gamma`",
            f"结构={'平铺代（`downsamples.i.residual` ✓）' if flat else '嵌套代（`downsamples.i.downsamples.j` ✓）'}"
            " ← 由 encoder.downsamples 的键层级反推",
            f"dim_mult={tuple(levels)} ← {shape_of}",
            f"num_res_blocks={blocks} ← 各级 `residual.0.gamma` 的个数",
            f"temperal_downsample={tuple(temporal)} ← 各级有没有 `time_conv.weight`",
            f"image_channels={image_channels} patch_size={patch_size} ← conv1 in={in_total} / head out={out_total}",
            f"temporal_kernel={temporal_kernel} ← `encoder.conv1.weight` 的时间核",
        ),
    )


def _flat_levels_of(shapes: dict[str, tuple[int, ...]], side: str, kind: str) -> tuple[FlatBlock, ...]:
    """把平铺代的一侧（``encoder.downsamples`` / ``decoder.upsamples``）摊成「一级一条」✓。

    平铺代的样子（盘上 16ch 那几份 ✓）：``downsamples.0/1.residual`` 是同一级的两个残差块 ✓、
    ``downsamples.2.resample`` ＋ ``downsamples.2.time_conv`` 是这一级的降采样 ✓，
    然后 ``3/4.residual`` ＋ ``5.resample`` …… ⇒ **相邻同级、不嵌套** ✓；最后一级没有采样器 ✓。

    ★ 每块 ``(in, out)`` 一律**照权重读**，不按 ``dim_mult`` 公式推 ✗：
    解码器实测有「级首块 192→384 升通道、末尾升采样又 384→192 降回来」这种排布 ✓
    （``qwen_image_vae`` 逐键如此 ✓），公式套不出来 ✗。
    """
    levels: list[FlatBlock] = []
    index = 0
    while f"{side}.{index}.residual.0.gamma" in shapes or f"{side}.{index}.resample.1.weight" in shapes:
        blocks: list[tuple[int, int]] = []
        down: tuple[int, int] | None = None
        temporal = False
        while True:
            head = f"{side}.{index}"
            if f"{head}.residual.0.gamma" in shapes:
                #: ⚠️ 权重形状是 ``(out, in, k...)`` ✗ 而 :class:`FlatBlock` 记的是 ``(in, out)`` ✓
                #: —— 顺序写反的症状是「级间断链：96 接不上 192」✓，看着像结构不对 ✗。
                out_dim, in_dim = shapes[f"{head}.residual.2.weight"][:2]
                blocks.append((in_dim, out_dim))
                index += 1
                continue
            if f"{head}.resample.1.weight" in shapes:
                #: ★ 采样器一到就**收尾这一级** ✗：不 break 会把下一级的残差块也数进来 ✓✗
                #: （实测症状：`encoder` 报每级 8 块、`decoder` 报 12 块 ✓，看着像"结构不自洽" ✓，
                #: 实际是**分组没切** ✗）。
                out_dim, in_dim = shapes[f"{head}.resample.1.weight"][:2]
                down = (in_dim, out_dim)
                temporal = f"{head}.time_conv.weight" in shapes
                index += 1
            break
        if not blocks:
            raise ValueError(f"平铺代结构读不通 ✗：`{side}.{index}` 这一级没有残差块 ✓")
        levels.append(FlatBlock(blocks=tuple(blocks), down=down, temporal=temporal))
    if levels:
        #: 级间必须**首尾相接** ✓（上一级产出 = 下一级输入 ✓）；断链说明分组还是错的 ✓。
        for position, (left, right) in enumerate(zip(levels, levels[1:])):
            tail = left.down[1] if left.down else left.blocks[-1][1]
            if tail != right.blocks[0][0]:
                raise ValueError(
                    f"平铺代结构断链 ✗：{kind} 第 {position} 级产出 {tail} 通道 ✓，"
                    f"而第 {position + 1} 级要 {right.blocks[0][0]} 通道 ✗")
    return tuple(levels)


def patchify(x: Tensor, patch_size: int) -> Tensor:
    """``b c (h q) (w r) -> b (c r q) h w`` ✓（ps=1 时恒等 ✓）。"""
    if patch_size == 1:
        return x
    q = r = patch_size
    b, c, hq, wr = x.shape
    return x.reshape(b, c, hq // q, q, wr // r, r).permute(0, 1, 5, 3, 2, 4).reshape(b, c * r * q, hq // q, wr // r)


def unpatchify(x: Tensor, patch_size: int) -> Tensor:
    """``b (c r q) h w -> b c (h q) (w r)`` ✓。"""
    if patch_size == 1:
        return x
    q = r = patch_size
    b, crq, h, w = x.shape
    c = crq // (r * q)
    return x.reshape(b, c, r, q, h, w).permute(0, 1, 4, 3, 5, 2).reshape(b, c, h * q, w * r)


class CausalConv3d(nn.Conv3d):
    """因果 3D 卷积 ✓ —— ``padding[0] == 0``（``temporal_kernel == 1`` 那族 ✓）时就是普通 ``Conv3d``。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pad_t = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x: Tensor, cache_x: Tensor | None = None) -> Tensor:
        """时间维**只在左边**补零 ✓（因果 ✓、输出帧数与输入相同 ✓）。

        ⚠️ **单帧（``T == 1``）走另一条** ✗（2026-11-01 Krea2 单图解码实测 ✓）：
        ``temporal_kernel == 3`` 的族（``qwen_image_vae`` ✓）喂 ``T=1`` 时，左边补 2 帧虽然
        长度算得通（``1+2-3+1 = 1`` ✓），但 comfy 走的是**另一条等价路径** ——
        ``autopad="causal_zero"``：``weight = weight[:, :, -x.shape[2]:, :, :]`` ✓
        （见 ``comfy/ops.py:615`` ✓），即**把时间核截到最后一拍**、时间上**不补** ✓。
        两者在 ``T>=3`` 时逐位相同 ✓、在 ``T==1`` 时也**恰好**相同 ✓（左补 2 个零后只剩最后一拍
        非零 ✓）；差别在 ``T==2`` ✓ —— 截核会让输出掉成 1 帧 ✗，补零才保住 2 帧 ✓
        ⇒ 判据取 ``T == 1`` ✓（与 comfy 的 ``x.shape[2] == 1`` 同一条件 ✓）。

        不这么写会当场
        ``Calculated padded input size per channel: (1 x 66 x 66). Kernel size: (3 x 3 x 3)`` ✓
        —— 报错说"核比输入大" ✓，读起来像"尺寸算错了" ✓，其实是**单帧没走截核那条路** ✓✗。
        """
        if self._pad_t:
            if x.shape[2] == 1:
                weight = self.weight[:, :, -1:, :, :]
                return F.conv3d(x, weight, self.bias, self.stride, self.padding,
                                self.dilation, self.groups)
            x = F.pad(x, [0, 0, 0, 0, self._pad_t, 0])
        return super().forward(x)


class RMSNorm(nn.Module):
    """Wan 的 ``RMS_norm`` ✓：**L2 归一化乘 sqrt(dim)**，不是均方根 ✗。"""

    def __init__(self, dim: int, images: bool = False) -> None:
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1) if images else (dim, 1, 1, 1)))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma.to(dtype=x.dtype)


class AttentionBlock(nn.Module):
    """单头自注意力 ✓（在 ``h*w`` 上做，无位置编码 ✓）。

    ⭐ ``chunk``（2026-11-02 加 ✓）：**按 query 分块**调 SDPA ✓，把 ``n²`` 那个矩阵从
    「整张一次建」变成「一次一块」✓。

    ⚠️ **它不保证逐位相同** ✓✗ —— 同一批 key 的 softmax 每行独立 ✓ 是**数学上**成立的，
    但 SDPA 的 math 后端内部也会按 query 再切块（归约顺序随块大小变 ✓），所以我们外部切的块
    与它内部的块**对不上**时就会有差异 ✓。实测（fp32、真权重、2048×1536）：

    | VAE | n | chunk | 与整量比 |
    |---|---|---|---|
    | 2.1 | 12288 | **4096** | **0.000e+00（逐位 ✓）** |
    | 2.1 | 12288 | 1024 | 相对 8.55e-03 |
    | 8× | 49152 | 4096 | 相对 1.81e-03 |
    | 8× | 49152 | 1024 | 相对 3.66e-03 |

    ⇒ 「逐位」是**某个 chunk 恰好与内部块对齐**的巧合 ✓✗，**不是**这条路的性质 ✓。
    所以默认**关** ✓；开了就必须在回执里如实报 ✓（见 `docs/reference-audit.md` ✓）。

    为什么需要它（实测 2026-11-02）：``head_dim = c``（384/1152）远超 efficient 后端上限
    ⇒ SDPA 只能走 **math** 后端 ⇒ 显式建 ``[b*t, 1, n, n]`` 的 fp32 矩阵 ✓。
    8× 那份 VAE 在 2048×1536 画布下 ``n = 192*256 = 49152`` ⇒ 整次解码峰值 **20.96 GiB**
    （卡 22.49 ✓）；按 query 切成 1024 ⇒ **3.94 GiB** ✓ —— 这才是「VAE 解码爆显存」的真因 ✓✗，
    与 ``upsamples`` 的卷积无关 ✓（探针实证：只切 up 链，该峰值**一点不降** ✓）。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor, chunk: int | None = None) -> Tensor:
        identity = x
        b, c, t, h, w = x.shape
        flat = self.norm(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        q, k, v = self.to_qkv(flat).chunk(3, dim=1)
        q, k, v = (e.view(b * t, 1, c, h * w).transpose(2, 3) for e in (q, k, v))
        out = _attend(q, k, v, chunk)
        out = self.proj(out.transpose(2, 3).reshape(b * t, c, h, w))
        return out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4) + identity


def _attend(q: Tensor, k: Tensor, v: Tensor, chunk: int | None) -> Tensor:
    """``chunk`` 个 query 一次 ✓；``None``/``0``/``≥n`` ⇒ 整量 ✓（与旧路径逐位不变 ✓）。"""
    n = q.shape[2]
    if not chunk or chunk <= 0 or chunk >= n:
        return F.scaled_dot_product_attention(q, k, v)
    return torch.cat([F.scaled_dot_product_attention(q[:, :, i:i + chunk], k, v)
                      for i in range(0, n, chunk)], dim=2)


def conv3(in_dim: int, out_dim: int, temporal_kernel: int) -> CausalConv3d:
    return CausalConv3d(in_dim, out_dim, (temporal_kernel, 3, 3), padding=(temporal_kernel // 2, 1, 1))


class Resample(nn.Module):
    """2D/3D 上下采样 ✓ —— 3D 两条分支**必须真跑 ``time_conv``** ✓（否则时间维一动不动 ✓✗）。"""

    def __init__(self, dim: int, mode: str, temporal_kernel: int = 3,
                 *, out_dim: int | None = None) -> None:
        super().__init__()
        self.mode = mode
        #: ⭐ ``out_dim``（2026-11-01 加 ✓）：**平铺代**的采样器**会换通道** ✓
        #: （``qwen_image_vae`` 的 ``decoder.upsamples.3.resample.1.weight`` 实测 384→192 ✓）。
        #: 嵌套代传 ``None`` ⇒ 恒等旧行为 ✓。
        out = dim if out_dim is None else int(out_dim)
        if mode == "upsample2d":
            self.resample = nn.Sequential(nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                                          nn.Conv2d(dim, out, 3, padding=1))
        elif mode == "upsample3d":
            self.resample = nn.Sequential(nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                                          nn.Conv2d(dim, out, 3, padding=1))
            #: ★ 时间卷积的输出通道是 ``2 × in`` ✗ 不是 ``2 × out`` ✓ —— 盘上
            #: ``decoder.upsamples.3.time_conv`` 实测 (768, 384) 而该级 conv 是 384→192 ✓，
            #: 按 ``2×out`` 建就只有 384 ⇒ 装载报「形状不符」✓（读起来像权重坏了 ✗）。
            self.time_conv = CausalConv3d(dim, dim * 2, (temporal_kernel, 1, 1),
                                          padding=(temporal_kernel // 2, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, out, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, out, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, out, (temporal_kernel, 1, 1), stride=(2, 1, 1))
        elif mode == "none":
            self.resample = nn.Identity()
        else:
            raise ValueError(f"未知 resample 模式：{mode}")

    def _per_frame(self, x: Tensor) -> Tensor:
        """逐帧 2D 采样 ✓（``[b,c,t,h,w]`` ⇒ ``[b,c',t,h',w']`` ✓，时间维**原样不动** ✓）。"""
        b, c, t, h, w = x.shape
        flat = self.resample(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        #: ★ 新通道数只能从 ``flat`` 读 ✗ 不能用输入的 ``c``（会换通道时它是旧的 ✓✗）。
        return flat.reshape(b, t, flat.shape[1], flat.shape[-2], flat.shape[-1]).permute(0, 2, 1, 3, 4)

    def forward(self, x: Tensor, first_chunk: bool = False) -> Tensor:
        """2D / 3D 采样 ✓ —— **3D 那两个分支必须真跑 ``time_conv``** ✗✗。

        ## 为什么这条是硬缺陷（2026-11-02 真跑撞出来的 ✓）

        旧实现**从头到尾没调过 ``time_conv``** ✓✗ ⇒ ``downsample3d`` / ``upsample3d`` 退化成纯空间
        采样 ✓，时间维**一动不动** ✓ ⇒ Wan 2.2 I2V 的首帧条件在 ``encode`` 段
        ``参考 latent 时间维 17 ✗ ≠ 潜变量 5 ✓`` ✓（读起来像"帧数填错了" ✗，其实是**VAE 没做时间压缩** ✓）。
        反向更要命：``decode`` 出片会得到 **T 帧而不是 4(T-1)+1 帧** ✓✗ —— 而且不报错 ✓。

        ## 本实现的口径：**无 ``feat_cache`` 的整段因果等价形** ✓

        参考实现（``comfy/ldm/wan/vae.py::WanVAE.encode``）是把像素按 ``1, 4, 4, …`` 切块、
        每块带 ``feat_cache`` 跑同一套卷积 ✓。逐层看，缓存只是"把上一块的尾巴接回来" ✓ ⇒
        **整段跑一遍**与它**逐窗口相同** ✓（每个输出窗口仍是 3 帧、步长 2 ✓）。

        唯一的差别在**首帧** ✓：参考里第一块只有 1 帧 ⇒ ``time_conv`` 那一步被跳过 ✓
        （``feat_cache[idx] = x`` 直接返回 ✓）。所以本实现也**让首帧不过时间卷积** ✓：

        * ``downsample3d``：``[首帧的纯空间采样] ＋ conv(整段) 的全部输出`` ✓
          ⇒ ``17 → 1+8 = 9`` ✓、再一级 ``9 → 1+4 = 5`` ✓（＝ 4 倍压缩 ＋ 1 ✓）；
        * ``upsample3d``：``time_conv`` 出 ``2c`` ⇒ 交错重排成 ``2T`` 帧 ✓，``first_chunk`` 时
          **丢掉头两份、换上首帧的纯空间采样** ✓ ⇒ ``2T−1`` ✓（``5 → 9 → 17`` ✓）。

        ``T == 1`` 时两条时间分支都**一个窗口都凑不出** ⇒ 只剩空间采样 ✓，与旧行为**逐位相同** ✓
        （所有单帧解码 / 单图编码不受影响 ✓，这条是硬要求 ✓）。
        """
        if self.mode == "downsample3d":
            flat = self._per_frame(x)
            if flat.shape[2] < 3:                      # 凑不出 (3 帧 / 步长 2) 的窗口
                return flat
            return torch.cat([flat[:, :, :1], self.time_conv(flat)], dim=2)
        if self.mode == "upsample3d":
            if x.shape[2] <= 1:
                return self._per_frame(x)
            doubled = self._duplicate(self.time_conv(x))     # [b, c, 2T, h, w]
            if not first_chunk:
                return self._per_frame(doubled)
            #: 首帧不过时间卷积 ✓（参考的 ``'Rep'`` 分支 ✓）；``doubled`` 的头两份属于第 0 帧 ✓。
            return torch.cat([self._per_frame(x[:, :, :1]), self._per_frame(doubled[:, :, 2:])], dim=2)
        return self._per_frame(x)

    @staticmethod
    def _duplicate(y: Tensor) -> Tensor:
        """``time_conv`` 吐的 ``2c`` 交错重排成两倍帧数 ✓（comfy ``stack(...,3).reshape`` 同义 ✓）。

        ``y.reshape(b, 2, c, T, h, w)`` ⇒ 后半段（第 2 个 ``c``）落在**奇数帧** ✓
        （``out[2j]`` 取半 0、``out[2j+1]`` 取半 1 ✓）。
        """
        b, c2, t, h, w = y.shape
        c = c2 // 2
        y = y.reshape(b, 2, c, t, h, w)
        return y.permute(0, 2, 3, 1, 4, 5).reshape(b, c, t * 2, h, w)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temporal_kernel: int) -> None:
        super().__init__()
        #: ★ 第二个 conv 前**有一个无参 Dropout** ✓ —— 少了它权重索引会全错位
        #: （权重里是 ``residual.0.gamma / .2.weight / .3.gamma / .6.weight`` ✓，
        #: 直接连写就成了 0/2/3/5 ✗，装载时 60 个键「缺」✗）。
        self.residual = nn.Sequential(
            RMSNorm(in_dim), nn.SiLU(), conv3(in_dim, out_dim, temporal_kernel),
            RMSNorm(out_dim), nn.SiLU(), nn.Dropout(0.0),
            conv3(out_dim, out_dim, temporal_kernel),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        short = self.shortcut(x)
        return self.residual(x) + (short if isinstance(short, Tensor) else x)


class AvgDown3D(nn.Module):
    """无参降采样 ✓：时空块**取均值**（不是丢弃 ✓）。"""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1) -> None:
        super().__init__()
        self.out_channels, self.factor_t, self.factor_s = out_channels, factor_t, factor_s
        self.factor = factor_t * factor_s * factor_s
        if in_channels * self.factor % out_channels:
            raise ValueError(f"AvgDown3D 不整除：{in_channels}×{self.factor} % {out_channels}")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: Tensor) -> Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        if pad_t:
            x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        x = x.view(b, c, t // self.factor_t, self.factor_t,
                   h // self.factor_s, self.factor_s, w // self.factor_s, self.factor_s)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, c * self.factor, t // self.factor_t, h // self.factor_s, w // self.factor_s)
        x = x.view(b, self.out_channels, self.group_size, t // self.factor_t,
                   h // self.factor_s, w // self.factor_s)
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    """无参升采样 ✓：通道复制再重排；``first_chunk`` 时切掉**第一帧** ✓。"""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1) -> None:
        super().__init__()
        self.out_channels, self.factor_t, self.factor_s = out_channels, factor_t, factor_s
        self.factor = factor_t * factor_s * factor_s
        if out_channels * self.factor % in_channels:
            raise ValueError(f"DupUp3D 不整除：{out_channels}×{self.factor} % {in_channels}")
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: Tensor, first_chunk: bool = False) -> Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        b, _, t, h, w = x.shape
        x = x.view(b, self.out_channels, self.factor_t, self.factor_s, self.factor_s, t, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(b, self.out_channels, t * self.factor_t, h * self.factor_s, w * self.factor_s)
        if first_chunk:
            x = x[:, :, self.factor_t - 1:, :, :]
        return x


class DownResidualBlock(nn.Module):
    """🔴 ``down_flag`` 是**最后一级不降采样**的开关 ✓ —— 忘了它就会多降一次（边长只放大 8 倍 ✗）。"""

    def __init__(self, in_dim: int, out_dim: int, mult: int, temperal_downsample: bool,
                 down_flag: bool, temporal_kernel: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        first = in_dim
        for index in range(mult):
            layers.append(ResidualBlock(in_dim, out_dim, temporal_kernel))
            in_dim = out_dim
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            layers.append(Resample(out_dim, mode, temporal_kernel))
        self.downsamples = nn.Sequential(*layers)
        self.avg_shortcut = AvgDown3D(first, out_dim,
                                      factor_t=2 if temperal_downsample else 1,
                                      factor_s=2 if down_flag else 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.downsamples(x) + self.avg_shortcut(x)


class UpResidualBlock(nn.Module):
    """``up_flag=False`` 时**没有** ``avg_shortcut`` ✓（参考实现建的也是 ``None`` ✓）。"""

    def __init__(self, in_dim: int, out_dim: int, mult: int, temperal_upsample: bool,
                 up_flag: bool, temporal_kernel: int) -> None:
        super().__init__()
        # ★ ``first`` 必须**先存** ✗：循环里 ``in_dim = out_dim`` 会把它覆盖，
        # 用覆盖后的值建 ``DupUp3D`` ⇒ 复制倍数算成 ``out*factor/out`` = 2× ✗，
        # 症状是 ``view`` 报「输入元素是期望的 2 倍」✓（不是通道不整除，更难看出）。
        first = in_dim
        layers: list[nn.Module] = []
        for index in range(mult):
            layers.append(ResidualBlock(in_dim, out_dim, temporal_kernel))
            in_dim = out_dim
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            layers.append(Resample(out_dim, mode, temporal_kernel))
        self.upsamples = nn.Sequential(*layers)
        self.avg_shortcut = (
            DupUp3D(first, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2)
            if up_flag else None
        )

    def forward(self, x: Tensor, first_chunk: bool = False) -> Tensor:
        #: ★ 必须**逐层**传 ``first_chunk`` ✗ 不能 ``self.upsamples(x)`` 一把过 ✗✗：
        #: ``nn.Sequential`` 只传位置参数 ⇒ 里面那个 ``Resample`` 收不到 ``first_chunk`` ✓、
        #: 按 ``2T`` 吐帧 ✓，而 ``DupUp3D`` 按 ``first_chunk`` 吐 ``2T−1`` ✓ ⇒
        #: ``out + self.avg_shortcut(...)`` 当场
        #: ``RuntimeError: The size of tensor a (10) must match the size of tensor b (9)`` ✓
        #: （报得像尺寸算错 ✗，其实是**同一个开关只传到了一半** ✓）。
        out = x
        for layer in self.upsamples:
            out = layer(out, first_chunk=first_chunk) if isinstance(layer, Resample) else layer(out)
        if self.avg_shortcut is None:
            return out
        return out + self.avg_shortcut(x, first_chunk=first_chunk)


class Encoder3d(nn.Module):
    """编码器 ✓ —— 权重里**一半是它** ✓（不建就装不满 ``state_dict`` ✓），且编码路径也要用它 ✓。

    ⚠️ 2026-11-01 前这个类**只有 ``__init__``、没有 ``forward``** ✗✗ —— 于是
    :meth:`WanVAE.encode` 也**不存在** ✓✗ ⇒ 任何"图 → 潜变量"的路（单流首帧条件 /
    参考图 / img2img）在 Wan 族上都只能撞 ``AttributeError: 'WanVAE' object has no attribute 'encode'`` ✓
    （报得像"调错方法" ✓，其实是**这条能力没接** ✓）。``head`` 输出 **2×z_dim** ✓
    （``mean‖logvar`` ✓，与 :class:`Decoder3d` 的输入侧对称 ✓）就是最硬的证据 ✓。
    """

    def __init__(self, dim: int, z_dim: int, image_channels: int, patch_size: int,
                 dim_mult: tuple[int, ...], num_res_blocks: int,
                 temperal_downsample: tuple[bool, ...], temporal_kernel: int,
                 *, flat: bool = False, levels: tuple[FlatBlock, ...] = ()) -> None:
        super().__init__()
        #: ★ ``[1] + dim_mult`` ⇒ 比 ``dim_mult`` **多一级** ✓（每级之间升一次通道 ✓）
        dims = [dim] + [dim * level for level in dim_mult]
        self.conv1 = CausalConv3d(image_channels * patch_size ** 2, dims[0], (temporal_kernel, 3, 3),
                                  padding=(temporal_kernel // 2, 1, 1))
        blocks: list[nn.Module] = []
        if flat:
            #: ⭐ **平铺代**（2026-11-01 ✓）：残差块**直接**挂在 ``downsamples.i`` 上 ✓，
            #: 没有 ``DownResidualBlock`` 那层外壳、也**没有** ``avg_shortcut`` ✓ ——
            #: 键名就是 ``downsamples.i.residual.*`` / ``downsamples.i.shortcut.*``
            #: / ``downsamples.i.resample.*`` / ``downsamples.i.time_conv.*`` ✓。
            #: 通道一律照 ``levels`` 逐块走 ✓（理由见 :class:`FlatBlock` 的注 ✓）。
            if not levels:
                raise ValueError("平铺代必须带 `levels` ✗（逐块 (in,out) 快照 ✓）")
            for record in levels:
                for in_dim, out_dim in record.blocks:
                    blocks.append(ResidualBlock(in_dim, out_dim, temporal_kernel))
                if record.down is not None:
                    mode = "downsample3d" if record.temporal else "downsample2d"
                    blocks.append(Resample(record.down[0], mode, temporal_kernel, out_dim=record.down[1]))
        else:
            for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
                flag = temperal_downsample[index] if index < len(temperal_downsample) else False
                blocks.append(DownResidualBlock(in_dim, out_dim, num_res_blocks, flag,
                                                down_flag=index != len(dim_mult) - 1,
                                                temporal_kernel=temporal_kernel))
        self.downsamples = nn.Sequential(*blocks)
        self.middle = nn.Sequential(
            ResidualBlock(dims[-1], dims[-1], temporal_kernel),
            AttentionBlock(dims[-1]),
            ResidualBlock(dims[-1], dims[-1], temporal_kernel),
        )
        self.head = nn.Sequential(RMSNorm(dims[-1]), nn.SiLU(),
                                  CausalConv3d(dims[-1], z_dim * 2, (temporal_kernel, 3, 3),
                                               padding=(temporal_kernel // 2, 1, 1)))
        #: ★ ``z_dim`` **存下来** ✓：编码后要 ``chunk(2)`` 取均值半 ✓（不靠猜 ``head`` 的输出通道 ✓）。
        self.z_dim = int(z_dim)

    def forward(self, x: Tensor) -> Tensor:
        """``[B, C_in, T, H, W]`` ⇒ ``[B, 2·z_dim, T, h, w]``（``mean‖logvar`` ✓）。

        与 :meth:`Decoder3d.forward` **同一条骨架** ✓（conv1 → 逐级 → middle → head ✓）；
        ⚠️ 收 ``first_chunk`` 的层只有 :class:`Resample` ✓（且它收下即弃 ✓），
        :class:`DownResidualBlock` / :class:`ResidualBlock` **不收** ✗ ⇒ 这里不能无脑透传 ✗。
        """
        x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x)
        x = self.middle(x)
        return self.head(x)


def _upsamples_spatially(layer: nn.Module) -> bool:
    """该层的**出口空间边长**是否 ×2 ✓（只判结构、不跑数据 ✓）。

    两种挂法都要认 ✓：平铺代把 :class:`Resample` 直接挂在 ``upsamples`` 上 ✓；
    嵌套代是 :class:`UpResidualBlock` 把它藏在内部 ✓（``up_flag`` ⇔ ``avg_shortcut`` 非空 ✓）。
    """
    if isinstance(layer, Resample):
        return layer.mode.startswith("upsample")
    if isinstance(layer, UpResidualBlock):
        return layer.avg_shortcut is not None
    return False


class Decoder3d(nn.Module):
    """Wan 式解码器 ✓。

    ⭐ **空间分块**（``tile`` / ``halo`` ✓，2026-11-02 补）：这项与「按输出张量重叠平均」
    （:mod:`.vae_tiling`）是**两件不同的事** ✓ —— 后者要求算子全程局部，而本类的
    ``middle.1`` 是**全图单头注意力** ✓✗ ⇒ 在 latent 上切块再拼，整张都会变（实测相对差 1.35 ✓）。

    正确的切法只有一种 ✓：**``conv1`` ＋ ``middle``（含注意力）在整张上跑** ✓，
    之后的 ``upsamples`` ＋ ``head``（全 3×3 卷积 ＋ 逐点 RMSNorm/SiLU ＋ 逐点上采样）**分块跑** ✓。
    块之间留 ``halo``（latent 单位）⇒ 块内靠内圈计算，只裁掉外圈 ✓。
    halo 的下限由 :meth:`receptive_latent` 解析算出（不是拍脑袋 ✓）：不够就有接缝、而且**不报错** ✗✗。

    ⚠️ **它不是逐位不变的** ✓✗（2026-11-02 实测纠正）：接缝算术是精确的 ✓，但 conv 对**不同窗口
    尺寸**会选不同 cuDNN 算法 ⇒ 浮点归约顺序变 ✓。fp32、2048×1536 实测相对差 ``5.8e-03``，
    而且**加 halo 消不掉**（halo 从 16 加到 24 仍有 ``5e-2`` ✗✗）⇒ 这不是「halo 不够」✓。
    另一个量测陷阱：逐级看，绝对差一路放大（``5e-2 → 3e+02``），但**相对差稳定在 1e-4~6e-4**
    （因为残差链本身把幅值放大到 1e5+ ✓）⇒ 判「接缝」必须看**相对**差 ✗ 绝对值 ✓。

    为什么非做不可（2026-11-02 实测，bf16、单帧、卡 22.49 GiB ✓）：

    | VAE | 画布 | 整张解码峰值 | ``attn_chunk=1024`` | 再加 ``tile=512`` |
    |---|---|---|---|---|
    | 2.1（16×，64ch 潜） | 2048×1536 | **8.69 GiB** | 8.69（**没用** ✗） | **3.21 GiB** |
    | 2.1 | 3072×2304 | **18.75 GiB** | 18.75（没用 ✗） | **3.28 GiB** |
    | 2.1 | 4096×3072 | 32.73 GiB（超卡，靠系统内存回落 ⇒ 极慢） | — | — |
    | 2512（8×，16ch 潜） | 2048×1536 | **20.96 GiB**（顶到 93% 卡） | **3.94 GiB** | **1.03 GiB** |
    | 2512 | 3072×2304 | **OOM** | — | — |

    ⇒ 两个旋钮**各治一半** ✓：2.1 的峰值几乎全在 ``upsamples``（它那份注意力 ``n=12288`` 只 ~0.6 GiB ✓），
    8× 的峰值几乎全在注意力（只切 up 链**一点不降** ✓）。「几百 MB」是错的 ✓✗：
    8× 那份在**现行主力画布**上就已经把卡吃满 ✓。
    """

    def __init__(self, dim: int, z_dim: int, out_channels: int, patch_size: int,
                 dim_mult: tuple[int, ...], num_res_blocks: int,
                 temperal_downsample: tuple[bool, ...], temporal_kernel: int,
                 *, flat: bool = False, levels: tuple[FlatBlock, ...] = ()) -> None:
        super().__init__()
        #: ★ ``[dim_mult[-1]] + dim_mult[::-1]`` ⇒ 也是**多一级** ✓；且 up 侧 ``mult = num_res_blocks + 1`` ✓
        dims = [dim * dim_mult[-1]] + [dim * level for level in dim_mult[::-1]]
        self.conv1 = CausalConv3d(z_dim, dims[0], (temporal_kernel, 3, 3),
                                  padding=(temporal_kernel // 2, 1, 1))
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], temporal_kernel),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], temporal_kernel),
        )
        temperal_upsample = list(reversed(temperal_downsample))
        blocks: list[nn.Module] = []
        if flat:
            #: ⭐ **平铺代**：升采样器直接挂 ``upsamples.i.resample`` ✓、**没有** ``avg_shortcut`` ✓；
            #: 通道照 ``levels`` 逐块走 ✓（级首块会「升通道」、末尾升采样再降回来 ✓，公式套不出来 ✗）。
            if not levels:
                raise ValueError("平铺代必须带 `levels` ✗（逐块 (in,out) 快照 ✓）")
            for record in levels:
                for in_dim, out_dim in record.blocks:
                    blocks.append(ResidualBlock(in_dim, out_dim, temporal_kernel))
                if record.down is not None:
                    mode = "upsample3d" if record.temporal else "upsample2d"
                    blocks.append(Resample(record.down[0], mode, temporal_kernel, out_dim=record.down[1]))
        else:
            for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
                flag = temperal_upsample[index] if index < len(temperal_upsample) else False
                blocks.append(UpResidualBlock(in_dim, out_dim, num_res_blocks + 1, flag,
                                              up_flag=index != len(dim_mult) - 1,
                                              temporal_kernel=temporal_kernel))
        self.upsamples = nn.Sequential(*blocks)
        self.head = nn.Sequential(RMSNorm(dims[-1]), nn.SiLU(),
                                  CausalConv3d(dims[-1], out_channels * patch_size ** 2,
                                               (temporal_kernel, 3, 3),
                                               padding=(temporal_kernel // 2, 1, 1)))

    def receptive_latent(self) -> int:
        """``upsamples`` ＋ ``head`` 折回 latent 坐标的**感受野半径** ✓（latent 像素 ✓）。

        解析算、不跑数据 ✓：逐级累加「该级 3×3 卷积的像素半径 ÷ 该级相对 latent 的倍数」✓。
        ``Resample`` 里那个 conv 算在**升级前**的倍数上 ⇒ 偏**保守**（更大）✓ —— 宁可多算，
        少算就是接缝 ✓✗。时间维核为 1 的（``CausalConv3d(…,1)`` / ``time_conv``）不计数 ✓
        （空间核仍是 ``(1,3,3)`` 的要看最后一维 ✓）。
        """
        total = 0.0
        level = 1
        for layer in self.upsamples:
            for module in layer.modules():
                if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                    kernel = int(module.kernel_size[-1])
                    if kernel > 1:
                        total += (kernel // 2) / level
            if _upsamples_spatially(layer):
                level *= 2
        total += 1 / level                      # head 的 3×3 在最高分辨率
        return int(math.ceil(total)) + 1        # ＋1：把 Resample/边界补零的半步也兜住

    def _run_ups(self, hidden: Tensor, first_chunk: bool) -> Tensor:
        """``middle`` 之后的整条链 ✓（``upsamples`` ＋ ``head`` ✓）。"""
        x = hidden
        for layer in self.upsamples:
            #: 平铺代这里的元素是裸 ``ResidualBlock`` ✓（不收 ``first_chunk``）⇒ 按类型分派 ✓；
            #: 嵌套代是 ``UpResidualBlock`` / ``Resample`` ✓ 两者都收 ✓。
            x = layer(x, first_chunk=first_chunk) if isinstance(layer, (UpResidualBlock, Resample)) else layer(x)
        return self.head(x)

    def _run_middle(self, x: Tensor, attn_chunk: int | None) -> Tensor:
        """``middle`` 三段 ✓，只把 ``attn_chunk`` 递给 :class:`AttentionBlock` ✓。

        按**类型**分派 ✓、不按索引 ✗ —— 盘上两种 VAE 的 ``middle`` 都是
        ``ResidualBlock → AttentionBlock → ResidualBlock`` ✓，但索引写死就等于
        把「这个结构恰好如此」当契约 ✓✗（换一族就静默跳过注意力 ✓）。
        """
        for layer in self.middle:
            x = layer(x, chunk=attn_chunk) if isinstance(layer, AttentionBlock) else layer(x)
        return x

    def forward(self, x: Tensor, first_chunk: bool = False, *,
                tile: int | None = None, halo: int | None = None,
                attn_chunk: int | None = None) -> Tensor:
        """``tile``＝每块在 **latent** 空间的边长 ✓（``None`` ＝整张 ✓）。

        ``halo`` 也按 latent 算 ✓；给不够会被**抬到** :meth:`receptive_latent` ✓（不是信调用方 ✓）——
        这一条是关键：halo 差一点就是一条**看不见报错**的接缝 ✓✗。

        ``attn_chunk`` 透给注意力 ✓（见 :class:`AttentionBlock` 的 ``chunk`` 注 ✓）：
        与 ``tile`` 是**两件独立**的事 ✓ —— 只切 up 链治不了注意力的 ``n²`` 峰值 ✓✗，
        只切注意力也治不了 up 链的峰值 ✓✗（两张表见类头 ✓）。
        """
        #: ⚠️ ``conv1`` ＋ ``middle`` 里的**注意力**必须看整张 ✓✗ —— 切了它每个 query 就只看块内。
        #: （把 query 分块只是「怎么算」，key/value 仍是全量 ✓ ⇒ 数值不变 ✓。）
        hidden = self._run_middle(self.conv1(x), attn_chunk)
        if tile is None:
            return self._run_ups(hidden, first_chunk)
        return self._forward_tiled(hidden, int(tile), halo, first_chunk)

    def _forward_tiled(self, hidden: Tensor, tile: int, halo: int | None,
                       first_chunk: bool) -> Tensor:
        height, width = hidden.shape[-2:]
        pad = self.receptive_latent() if halo is None else max(int(halo), self.receptive_latent())
        tile = max(int(tile), 1)
        if tile >= max(height, width):
            return self._run_ups(hidden, first_chunk)
        out: Tensor | None = None
        span = 0
        for y0 in range(0, height, tile):
            y1 = min(y0 + tile, height)
            for x0 in range(0, width, tile):
                x1 = min(x0 + tile, width)
                wy0, wy1 = max(0, y0 - pad), min(height, y1 + pad)
                wx0, wx1 = max(0, x0 - pad), min(width, x1 + pad)
                piece = self._run_ups(hidden[..., wy0:wy1, wx0:wx1], first_chunk)
                if out is None:
                    #: ★ 比例**从第一块实测** ✓（各级 up 是否 ×2 由权重结构定 ✓，不靠假设 ✓）。
                    span = piece.shape[-1] // (wx1 - wx0)
                    out = piece.new_empty(piece.shape[0], piece.shape[1], piece.shape[2],
                                          height * span, width * span)
                #: 只把「非 halo 的内圈」写进画布 ✓ —— 外圈带着真实上下文算出来的，裁掉即弃 ✓。
                out[..., y0 * span:y1 * span, x0 * span:x1 * span] = piece[
                    ..., (y0 - wy0) * span:(y0 - wy0) * span + (y1 - y0) * span,
                    (x0 - wx0) * span:(x0 - wx0) * span + (x1 - x0) * span]
        return out


class WanVAE(nn.Module):
    """Wan 系 VAE ✓（本仓只走 ``decode`` 单帧路径 ✓，其余参数也要能装上 ✓）。"""

    def __init__(self, shape: WanVaeShape) -> None:
        super().__init__()
        self.shape = shape
        self.encoder = Encoder3d(shape.dim, shape.z_dim, shape.image_channels, shape.patch_size,
                                 shape.dim_mult, shape.num_res_blocks, shape.temperal_downsample,
                                 shape.temporal_kernel, flat=shape.flat, levels=shape.encoder_flat)
        self.conv1 = CausalConv3d(shape.z_dim * 2, shape.z_dim * 2, 1)
        self.conv2 = CausalConv3d(shape.z_dim, shape.z_dim, 1)
        self.decoder = Decoder3d(shape.dec_dim, shape.z_dim, shape.image_channels, shape.patch_size,
                                 shape.dim_mult, shape.num_res_blocks, shape.temperal_downsample,
                                 shape.temporal_kernel, flat=shape.flat, levels=shape.decoder_flat)

    @property
    def weight_dtype(self) -> Any:
        """参数所在 dtype ✓ —— **第一个参数**为准 ✓（``shape`` 是 dataclass、不带 dtype ✗）。

        没有参数（自检里的恒等桩 ✓）就回 ``float32`` ✓。
        """
        for parameter in self.parameters():
            return parameter.dtype
        return torch.float32

    # ---------------------------------------------------------------- 编码
    def encode(self, x: Tensor) -> Tensor:
        """``[B, 3, T, H, W]``（``[-1,1]`` 像素 ✓）⇒ ``[B, z_dim, T, H/S, W/S]`` ✓（**未归一化** ✗）。

        ⚠️ 这里返回的是 **VAE 模型空间**的潜变量 ✓✗ —— 喂给 DiT 之前**必须**过
        :func:`encode_latents`（它做 ``process_in`` ✓）；直接用的人会得到「形状全对、
        首帧条件几乎不起作用、一个错都不报」✓✗（与 :func:`decode_latents` 是同一类坑 ✓）。

        取 ``mean`` **不取 logvar** ✓（推理是确定性的 ✓ 后验均值 ✓ 不采样 ✓）。
        """
        target = self.weight_dtype
        if x.dtype != target:
            x = x.to(target)
        squeeze = x.ndim == 4
        if squeeze:
            x = x.unsqueeze(2)
        #: ⚠️ 与 :meth:`decode` 的 ``unpatchify`` **对称** ✓（少了这一步，ps>1 的族形状就错 ✗）。
        #: 这里**不**复用模块级的 :func:`patchify` ✓：它只吃 ``[B,C,H,W]`` ✓，而这一层是 5 维 ✓
        #: （T=1 时硬 ``squeeze(2)`` 能过、T>1 时直接炸 ✗ ⇒ 显式按 5 维折叠 ✓）。
        patch = self.shape.patch_size
        if patch != 1:
            b, c, t, h, w = x.shape
            x = x.reshape(b, c, t, h // patch, patch, w // patch, patch)
            x = x.permute(0, 1, 4, 6, 2, 3, 5).reshape(
                b, c * patch * patch, t, h // patch, w // patch)
        hidden = self.conv1(self.encoder(x))
        mean = hidden.chunk(2, dim=1)[0]
        return mean.squeeze(2) if squeeze else mean

    # ---------------------------------------------------------------- 解码
    def decode(self, z: Tensor, *, tile: int | None = None, halo: int | None = None,
               attn_chunk: int | None = None) -> Tensor:
        """``[B, z_dim, H, W]`` → ``[B, image_channels, H*scale, W*scale]`` ✓（RGBA ✓）。

        **两个省显存的旋钮，默认全关 ✓（关着 ⇒ 逐位等于旧路径 ✓）**：

        * ``tile``（**像素**单位、与 ComfyUI 的 ``VAEDecodeTiled`` 同口径 ✓）：``upsamples`` ＋
          ``head`` 按块跑 ✓。**改数值** ✓✗：实测 fp32 相对差 5.8e-03（2048×1536 ✓），
          且**与 halo 够不够无关** ✓ —— halo 加到 24 仍有 ``5e-2`` ✓（cuDNN 对不同窗口尺寸
          选不同卷积算法 ✓）。裁接缝是精确的 ✓，差的是浮点归约 ✓。
        * ``attn_chunk``：注意力按 query 分块 ✓。**也改数值** ✓✗（见 :class:`AttentionBlock`
          的表 ✓）；只在 ``n > chunk`` 时真的分块 ✓。
        两者**治的毛病不同** ✓：8× 那份的峰值几乎全在注意力（只切 up 链 **一点不降** ✓），
        2.1 的峰值全在 up 链（注意力只 ~0.6 GiB ✓）。
        ``halo`` 给不够会被抬到感受野下限 ✓（见 :meth:`Decoder3d.forward` ✓）。
        

        ``z`` 的单帧形状 (4 维) 就走 comfy 的 ``decode`` 分支：``unsqueeze(2)`` 后
        ``first_chunk=True`` ✓；返回时再 ``squeeze(2)`` ✓，与 comfy 逐位一致 ✓。

        ⚠️ **入口先把 ``z`` 对齐到权重精度** ✓（2026-11-01 Krea2 端到端实测 ✓）：
        采样侧的潜变量**有意保 f32** ✓（见 ``init_latents`` ✓），而这族的 VAE 权重在卡上
        多是 **bf16** ✓ —— 不对齐当场
        ``Input type (float) and bias type (struct c10::BFloat16) should be the same`` ✓，
        报错指向 ``conv`` ✓、读起来像"通道/权重坏了" ✓，其实是**精度没对齐** ✓✗。
        对齐**只能在这里**（模型内部）✗：在采样侧把潜变量降成 bf16 会连带毁掉
        逆过程的数值 ✓✗（同 `sdxl_vae.decode_latents` / `dit_lumina.forward` 的教训 ✓）。
        """
        target = self.weight_dtype
        if z.dtype != target:
            z = z.to(target)
        squeeze = z.ndim == 4
        if squeeze:
            z = z.unsqueeze(2)
        latent_tile, latent_halo = self._tile_in_latent(tile, halo)
        out = self.decoder(self.conv2(z), first_chunk=True,
                           tile=latent_tile, halo=latent_halo, attn_chunk=attn_chunk)
        out = unpatchify(out.squeeze(2), self.shape.patch_size)
        return out

    def _tile_in_latent(self, tile: int | None, halo: int | None) -> tuple[int | None, int | None]:
        """像素口径 → latent 口径 ✓（不足一格按一格、并对齐到 ``patch_size`` ✓）。

        ``patch_size>1`` 时块边界必须落在 patch 网格上 ✓，否则 ``unpatchify`` 会把接缝
        整条 patch 挪位 ✓✗；``halo`` 同样要对齐 ✓（它决定块的**起点偏移** ✓）。
        """
        if tile is None:
            return None, None
        scale = max(1, int(self.shape.spatial_scale))
        patch = max(1, int(self.shape.patch_size))

        def align(value: int) -> int:
            steps = max(1, int(round(value / patch)))
            return steps * patch

        return align(int(tile) // scale), (None if halo is None else align(int(halo) // scale))

    def decode_to_image(self, z: Tensor) -> Tensor:
        """解到可直接落盘的 RGB ✓：``[-1,1] → [0,1]``、丢 alpha ✓。

        ⚠️ ``z`` 必须是**已反缩放**的潜变量 ✗（DiT 吐的是归一化后的 ✓）——
        归一化值直接进来 ⇒ 图发灰、不报错 ✓✗。用 :func:`decode_latents` ✓ 它替你做那一步。
        """
        pixels = self.decode(z)
        pixels = pixels[:, :3]
        return (pixels.float() + 1.0) / 2.0


def build(shape: WanVaeShape, *, dtype: Any = None) -> WanVAE:
    """按形状建 VAE ✓。

    ``dtype`` 走 ``set_default_dtype`` ✓（同 :func:`~.qwen3_te.build_from_shapes` ✓）：
    ⚠️ 建完再 ``.to(dtype)`` 也能用 ✓，但**先 fp32 建一遍**就是两份峰值 ✓✗；
    而且 2026-10-31 实测的坑是**反过来**的 ✗：建 fp32 + 装载 ``dtype=bf16`` ⇒ 权重被
    ``copy_`` 回 float32 param ✓✗ ⇒ 报告写 ``dtype=bf16`` ✓、**参数全是 fp32** ✓✗，
    喂 bf16 潜变量当场报 ``Input type (BFloat16) and bias type (float) should be the same`` ✓
    （报得像"输入错了" ✓，其实是**建的时候没定精度** ✗✗）。
    """
    previous = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(dtype)             # ⚠️ 全局状态 ⇒ 必须 finally 复原 ✓
    try:
        module = WanVAE(shape)
    finally:
        torch.set_default_dtype(previous)
    return module if dtype is None else module.to(dtype)


# ----------------------------------------------------------------- 反缩放（口径事实）
#: ⚠️ **不在权重里** ✗：``qwen_image_2.1_vae_bf16.safetensors`` 的 238 键全是 conv/gamma ✓。
#: 这 64 个数逐字抄自 comfy ``comfy/latent_formats.py::QwenImage21.__init__``（``latents_mean``
#: 行 984–992 / ``latents_std`` 行 994–1002 ✓）。改这里之前先去那边核一遍 ✓。
QWEN21_LATENT_MEAN: tuple[float, ...] = (
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
    1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
    2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
    -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
    -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
    0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
)

QWEN21_LATENT_STD: tuple[float, ...] = (
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910,
    3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756,
    3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467,
    3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
    3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959,
    3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161,
)


#: Wan**2.1/2.2** 的 16 个数 ✓ —— 逐字抄自 comfy ``comfy/latent_formats.py::Wan21.__init__``
#: （``latents_mean`` 行 761–764 / ``latents_std`` 行 766–769 ✓）。
#: ⚠️ 别拿上面的 2.1（Qwen-Image 2.1）那 64 个顶替 ✗✗：两族**连通道数都不同**（16 vs 64 ✓），
#:    数值更是各算各的 ✓ —— 抄错一串 ⇒ 解码照跑、形状全对、图偏色发灰、一个错都不报 ✓✗。
WAN21_LATENT_MEAN: tuple[float, ...] = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)

WAN21_LATENT_STD: tuple[float, ...] = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)

#: Wan**2.2** 的 **48** 个数 ✓ —— 逐字抄自 comfy ``comfy/latent_formats.py::Wan22.__init__``
#: （``latents_mean`` 行 841–846 / ``latents_std`` 行 849–854 ✓、``scale_factor`` 仍 1.0 ✓）。
#: ⚠️ Wan22 **不是**「Wan21 换 48 通道」✗✗：``class Wan22(Wan21)`` 虽继承 ✓，但它**自己重写**了
#:    这两串 ✓ —— mean 全部重来（-0.2289… ✓ 不是 -0.7571… ✓）、std 更是从 2.8 量级掉到 0.4～1.6 ✓。
#:    拿 Wan21 那 16 个数顶 48 通道 ⇒ 现成就会撞通道数校验 ✓；真正危险的是"随手补 32 个 0/1"✗
#:    ⇒ 解码照跑、颜色偏、一个错都不报 ✓✗。
WAN22_LATENT_MEAN: tuple[float, ...] = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)

WAN22_LATENT_STD: tuple[float, ...] = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)


#: 族名 → **人话标题** ✓（只给回执/报错用 ✓，不参与任何数学 ✓）。
#: ⚠️ 为什么要这张表 ✗（2026-11-01 补 ✓）：回执里只写 ``qwen_image21`` ✓ 对机器够用 ✓、
#: 对人不够 ✓ —— 创作者读到的应该是「Qwen-Image 2.1」✓，而**不是** `H3` ✗（历史缺陷：
#: 老文案写死「否则 H3」，挂 2.1 也报 H3 ✓✗）。查不到就**原样回显族名** ✓（不猜族 ✓）。
FAMILY_TITLES: dict[str, str] = {
    "qwen_image21": "Qwen-Image 2.1",
    "qwen_image_2_1": "Qwen-Image 2.1",
    "qwen_image_21": "Qwen-Image 2.1",
    "wan21": "Wan 2.1",
    "wan_2_1": "Wan 2.1",
    "wan2_1": "Wan 2.1",
    "wan22": "Wan 2.2",
    "wan_2_2": "Wan 2.2",
    "wan2_2": "Wan 2.2",
    "qwen_image": "Qwen-Image 2512",
    "qwen_image_2512": "Qwen-Image 2512",
}


def _family_key(family: str) -> str:
    """族名 → **查表键** ✓：大小写、``-``、``.``、空格一律折成 ``_`` ✓，多余 ``_`` 压成一个 ✓。

    ⚠️ 为什么要这么折 ✗（2026-11-01 审出来的真缺陷 ✓）：本模块原先只折 ``-`` ✓
    ⇒ ① 表里刻意登记的 ``wan2.1`` 这种点号写法**永远命中不了** ✓✗（登记了却是死配置 ✓）；
    ② ``qwen-image-2.1`` 这种「连字符 ＋ 点号」组合谁都不命中 ✓ ⇒ 回执里回显一串机器名 ✓✗。
    两项都不是「错图」级 ✓，但都是「写了等于没写」✓✗ —— 表与折法必须是同一套口径 ✓。
    """
    key = str(family or "").strip().lower()
    for ch in ("-", ".", " "):
        key = key.replace(ch, "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


def family_title(family: str) -> str:
    """族名 → 人话标题 ✓；没登记 ⇒ **原样回显** ✓（绝不猜成别族 ✗✗）。"""
    title = FAMILY_TITLES.get(_family_key(family))
    return title if title is not None else str(family or "").strip()


def latents_stats(family: str, z_dim: int | None = None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """族名 → ``(mean, std)`` ✓；**没有就报错** ✗（**不猜、不回退成恒等** ✗✗）。

    ⚠️ 回退成 ``(0, 1)`` 是最贵的省事法 ✗：解码照跑 ✓、形状全对 ✓、图发灰 ✓、不报错 ✓✗。
    """
    key = _family_key(family)
    table: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
        "qwen_image21": (QWEN21_LATENT_MEAN, QWEN21_LATENT_STD),
        "qwen_image_2_1": (QWEN21_LATENT_MEAN, QWEN21_LATENT_STD),
        #: ⭐ **Wan 2.1 / 2.2 视频**（16ch ✓）：参考实现**有** 16 个 mean/std ✓（不是恒等 ✗✗）。
        #: 2026-10-31 修：此条原先写成 16 个 0/1，理由抄的是"``Wan21`` 没覆写 ``scale_factor``"
        #: —— 那句**混淆了两件事** ✓✗：``scale_factor`` 是 1.0 ✓ 不假，可 ``latents_mean/std``
        #: 是**另一组**数（-0.7571…/2.8184… ✓，见 `WAN21_LATENT_MEAN` ✓）。
        #: 后果就是"解码照跑、形状全对、颜色整体偏灰"✓✗，一个错都不报 ✓。
        "wan21": (WAN21_LATENT_MEAN, WAN21_LATENT_STD),
        "wan_2_1": (WAN21_LATENT_MEAN, WAN21_LATENT_STD),
        "wan2.1": (WAN21_LATENT_MEAN, WAN21_LATENT_STD),
        #: ⭐ **Wan 2.2 视频**（48ch ✓，`wan2.2` 盘上是 5B/14B 两档 DiT ✓）：**另一组** 48 个数 ✓
        #: （见 `WAN22_LATENT_MEAN` ✓），``scale_factor`` 仍是 1.0 ✓。别拿 ``wan21`` 顶 ✗✗。
        "wan22": (WAN22_LATENT_MEAN, WAN22_LATENT_STD),
        "wan_2_2": (WAN22_LATENT_MEAN, WAN22_LATENT_STD),
        "wan2.2": (WAN22_LATENT_MEAN, WAN22_LATENT_STD),
        #: ⭐ **Qwen-Image 2512**（16ch ✓）走**恒等** ✓ —— 与上面 ``wan21`` 是**两族** ✗✗：
        #: ComfyUI 里 2512 用的是 ``LatentFormat`` **基类**（``supported_models.QwenImage`` 没有
        #: 覆写 ``latent_format`` ✓ ⇒ ``scale_factor=1.0``、无 mean/std ✓）。
        #: ⚠️ 这 16 个 0/1 是**族事实** ✓、不是"取不到就回退成恒等" ✗✗ —— 两者在这张表上长得
        #: 一样、在语义上是反的 ✓：回退会让"没登记的新族"静默按恒等解码 ✓✗（图发灰、不报错 ✓）；
        #: 登记下来的这一条缺了就是缺了 ✓（报错 ✓）。
        #: ⚠️ 2512 曾借 ``wan21`` 这个族名（只因结构同是 16ch/8× ✓）—— 那时两条数值都是 0/1 ✓
        #: 才没露馅 ✓；``wan21`` 一改成真值，借名就会让 2512 **吃到 Wan 的 mean/std** ✓✗（图整体偏绿）。
        "qwen_image": ((0.0,) * 16, (1.0,) * 16),
        "qwen_image_2512": ((0.0,) * 16, (1.0,) * 16),
    }
    #: ⚠️ **两张表的键口径不同** ✗✗（2026-11-01 实测撞出来的 ✓）：本表既有 ``wan_2_1`` 也有
    #: ``wan2.1``（点号**不折** ✓），而 :data:`FAMILY_TITLES` 全是折后写法 ✓ ⇒ 归一后查不到时
    #: **按原样再查一次** ✓（只折 ``-`` 与大小写 ✓）。两把钥匙都试过才判「没登记」✗ ——
    #: 只试一把会把本来能命中的族判死 ✓✗（``latents_stats("wan2.1")`` 曾因此直接抛错 ✓）。
    if key not in table:
        raw = str(family or "").strip().lower().replace("-", "_")
        key = raw if raw in table else key
    if key not in table:
        raise ValueError(
            f"没有 {family!r} 的 ``latents_mean/std`` ✗ —— 这一族还没登记的数是**口径事实** ✓，"
            f"不许拿 (0, 1) 顶上 ✗✗（解码照跑、图发灰、不报错 ✓✗）。已登记：{sorted(table)} ✓")
    mean, std = table[key]
    if z_dim is not None and int(z_dim) != len(mean):
        raise ValueError(
            f"{family!r} 的均值/标准差是 {len(mean)} 通道 ✓，但 VAE 自述 z_dim={z_dim} ✗ "
            f"⇒ 不是同一族 ✓✗（形状对不上就别硬套 ✓）")
    return mean, std


def decode_latents(module: WanVAE, latents: Tensor, *, family: str = "qwen_image21",
                   mean: tuple[float, ...] | list[float] | None = None,
                   std: tuple[float, ...] | list[float] | None = None,
                   to_unit: bool = False,
                   tile: int | None = None, halo: int | None = None,
                   attn_chunk: int | None = None) -> Tensor:
    """**归一化潜变量** → 像素 ✓（先反缩放，再解码 ✓）。

    ⚠️ 顺序是硬的 ✗：``decode(反缩放(z))`` ✓；反过来或省掉 ⇒ 形状全对、图是灰噪、不报错 ✓✗。

    ``to_unit=True`` ⇒ 返 ``[0,1]`` 的 3 通道 ✓（丢 alpha ✓）；否则返 ``[-1,1]`` 的 4 通道 ✓
    （与 ``torch_backend.decode`` 的 ``value_range="-1..1"`` 口径一致 ✓）。

    ``tile`` / ``halo`` / ``attn_chunk`` 原样转给 :meth:`WanVAE.decode` ✓（**默认全关** ✓，
    关着逐位等于旧路径 ✓；开着**改数值** ✓✗，见那个方法的表 ✓）。

    ⚠️ **整段按 latent 切块是错的** ✓✗（2026-11-02 实测 ✓）：解码器 ``middle.1`` 是**全图
    单头注意力** ✓ ⇒ 潜变量一切块，注意力就只看得到自己那块 ✓，拼回来的**整张**都变了 ✓✗
    （真 2.1 权重实测：10×10 潜变量切 6×6 块 ⇒ 相对差 **1.35** ✓，不是接缝问题 ✓，是全局
    重排 ✓）。ComfyUI 的 ``WanVAE`` 同样**没有** ``decode_tiled`` ✓。
    ⇒ 本仓的 ``tile`` 只切「注意力**之后**的那几级 up 块」✓（那之后才是纯局部卷积 ✓），
    注意力那一拍另由 ``attn_chunk`` 按 query 分块 ✓ —— **两个旋钮治两半，缺一不可** ✓：
    2048×1536(fp32) 整张解 8× 那份峰值 **20.96 GiB** ✓（卡 22.49 ✓ ⇒ 3072×2304 直接 OOM ✓），
    ``attn_chunk=1024`` ⇒ **3.94 GiB** ✓，再叠 ``tile=512`` ⇒ **1.03 GiB** ✓；
    2.1(16×) 峰值几乎全在 up 链 ✓（``tile=512`` ⇒ 8.69 → **3.21 GiB** ✓）。
    """
    from app.services.engine import latent_formats as latent_formats_mod  # noqa: PLC0415

    shape = getattr(module, "shape", None)
    z_dim = int(getattr(shape, "z_dim", latents.shape[1]))
    if mean is None or std is None:
        mean, std = latents_stats(family, z_dim=z_dim)
    if len(mean) != int(latents.shape[1]):
        #: ⚠️ 这里**不能写死通道数** ✗（原先写"是 64 通道" ✓✗）：``latents_stats`` 现在有
        #: 16 通道的 ``wan21`` / ``qwen_image`` 与 64 通道的 ``qwen_image21`` ✗ —— 写死 64 会让
        #: 16 通道那两族的报错**自己说错话** ✓✗（读到"是 64 通道"会以为串了 2.1 的权重 ✓，
        #: 实际只是 ``family`` 名字填错 ✓）。报**登记了几个** ✓，那才是这一格里的事实 ✓。
        raise ValueError(
            f"潜通道 {int(latents.shape[1])} 与 ``mean/std`` 的 {len(mean)} 对不上 ✗"
            f"（{family!r} 登记的是 {len(mean)} 个 ✓）—— 别按广播硬算 ✓✗，"
            f"先看 ``family`` 是不是填成了别的一族 ✓（``wan21``/``qwen_image`` 16 ✓、"
            f"``qwen_image21`` 64 ✓）")
    # ⚠️ 必须显式 ``view(1, C, 1, 1)`` ✗：comfy 的 ``latents_mean`` 就是这么存的 ✓；
    #    一维 ``[C]`` 广播到 ``[B,C,H,W]`` 会在**维度 3** 上撞 ✓✗（报错指不到"少了 reshape" ✗）。
    view = (1, -1) + (1,) * max(0, latents.ndim - 2)
    mean_t = torch.tensor(list(mean), dtype=latents.dtype, device=latents.device).view(view)
    std_t = torch.tensor(list(std), dtype=latents.dtype, device=latents.device).view(view)
    scaled = latent_formats_mod.process_out(family, latents, mean=mean_t, std=std_t)
    #: ⚠️ 三个旋钮只在调用方**明确**给了才往下传 ✓（``tile=None`` ⇒ `decode` 走旧路径 ✓，
    #:    逐位不变 ✓）—— 不在这里替它挑默认值 ✗（见 :data:`SUGGESTED_TILE` ✓）。
    pixels = module.decode(scaled, tile=tile, halo=halo, attn_chunk=attn_chunk)
    if not to_unit:
        return pixels
    return ((pixels[:, :3].float() + 1.0) / 2.0).clamp(0.0, 1.0)


def encode_latents(module: WanVAE, frames: Tensor, *, family: str = "qwen_image21",
                   mean: tuple[float, ...] | list[float] | None = None,
                   std: tuple[float, ...] | list[float] | None = None) -> Tensor:
    """像素 → **归一化潜变量** ✓（:func:`decode_latents` 的对称面 ✓：先编码，再正向缩放 ✓）。

    ⚠️ 2026-11-01 前**没有这一层** ✓✗：``latent_formats.process_in`` 写好了却**全仓零调用** ✓，
    编码侧于是直接把 **VAE 模型空间**的潜变量当 DiT 输入 ✓✗ —— Flux 族差 0.3611 倍、
    Wan 族差 16 组 mean/std ✓ ⇒ 首帧条件/参考图**数值全歪**、形状全对、一个错不报 ✓（
    与 :mod:`latent_formats` 抬头警告的"编码侧没接"完全一致 ✓）。

    ⚠️ 顺序同样是硬的 ✗：``process_in(encode(x))`` ✓；省掉正向缩放 ⇒ 上面那个静默错 ✓。
    """
    from app.services.engine import latent_formats as latent_formats_mod  # noqa: PLC0415

    shape = getattr(module, "shape", None)
    z_dim = int(getattr(shape, "z_dim", 0) or 0)
    if mean is None or std is None:
        mean, std = latents_stats(family, z_dim=z_dim or None)
    encoded = module.encode(frames)
    if len(mean) != int(encoded.shape[1]):
        raise ValueError(
            f"编码出的潜通道 {int(encoded.shape[1])} 与 ``mean/std`` 的 {len(mean)} 对不上 ✗"
            f"（{family!r} 登记的是 {len(mean)} 个 ✓）—— ``family`` 大概填成了别的一族 ✓"
            f"（``wan21``/``qwen_image`` 16 ✓、``qwen_image21`` 64 ✓）")
    view = (1, -1) + (1,) * max(0, encoded.ndim - 2)
    mean_t = torch.tensor(list(mean), dtype=encoded.dtype, device=encoded.device).view(view)
    std_t = torch.tensor(list(std), dtype=encoded.dtype, device=encoded.device).view(view)
    return latent_formats_mod.process_in(family, encoded, mean=mean_t, std=std_t)
