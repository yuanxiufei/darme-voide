"""SEEDVR2 VAE（ByteDance ``VideoAutoencoderKL``）前向 ✓ —— 3D 因果卷积 + 逐帧空间自注意力 ✓。

本模块是 ``comfy/ldm/seedvr/vae.py``（ComfyUI 侧参考实现）的**逐位重写** ✓，只服务
SEEDVR2 修复链的「降质视频 → 潜变量 → 复原潜变量 → 像素」这一段 ✓。

结构事实（全部可从权重反推 ✓，见 :func:`infer_shape`）：

* 4 组下采样 / 4 组上采样，通道 ``(128, 256, 512, 512)`` ✓，每组 2 个 ResNet（解码侧 3 个）✓。
* **没有** ``quant_conv`` / ``post_quant_conv`` ✗ —— 盘上 250 键里就没有它们 ✓，
  编码器直接吐 32 通道（16 均值 + 16 logvar），取均值即 latent ✓。
* 时间口径：**两**次时间下采样 ⇒ 时间比 4 ✓；空间比 8 ✓。帧数铁律 ``T = 4n + 1`` ✓
  （``T=1`` 也合法 = ``4·0+1``）✓。
* 卷积是**因果**的：时间 padding 换成「把首帧复制若干份顶到前面」✓
  （``times = temporal_padding * 2``）✓。
* 上采样在**时间翻倍后丢掉第 2 帧**（``remove_head``）✓ —— 这就是 4n+1 的来源 ✓。
* 注意力只在 mid block ✓、**逐帧**做 ✓（``(B,T,C,H,W)`` 拍平成 ``(B·T, C, H, W)``）✓，单头 ✓。
* 潜变量缩放 ``0.9152`` ✗✗ —— **VAE 空间 ≠ 模型空间** ✓：DiT 吃的是 ``raw × 0.9152`` ✓，
  ``decode`` 要 ``÷ 0.9152`` ✓。本模块把它交给 :mod:`latent_formats` 的 ``seedvr2`` 口径 ✓，
  **不在这里另写一份乘加** ✓✗。

时间维分块**已接通** ✓（:class:`_MemoryState` ＋逐层 ``memory_cache`` ✓，机制抄自参考实现的
``MemoryState`` ✓，对外口＝:meth:`SeedVR2Vae.decode` 的 ``temporal_chunk`` ✓）；它**不是**「顺手接上」
就能做对的那类 ✗：``_extend_head`` 在块跑时会把**块首帧**当左 padding，因果链把那点误差送到整块
✓✗ —— 所以块首必须接**块外真实帧**，且要**逐层**带 ✓。反面教材（四个变体全差 1.8~2.8 ✓）与验收
判据写在 :func:`decode_latents` 的文档里 ✓，动手改这块前先读它 ✓。

有意留白：参考实现的 ``memory_limit`` / ``tiling`` **空间**分块（本仓 ``_forward_tiled`` 更严 ✓）
与 ``memory_limit_conv`` 未搬 ✓。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import latent_formats

__all__ = [
    "BYTEDANCE_BLOCK_OUT_CHANNELS",
    "SEEDVR2_LATENT_CHANNELS",
    "SEEDVR2_LATENT_SCALE",
    "SEEDVR2_SPATIAL_SCALE",
    "SEEDVR2_TEMPORAL_SCALE",
    "SEEDVR2_TEMPORAL_SCALE_NUM",
    "SUGGESTED_TEMPORAL_CHUNK",
    "SeedVR2Vae",
    "chunk_ladder",
    "shrink_for_peak",
    "SeedVR2VaeError",
    "SeedVR2VaeShape",
    "build",
    "decode_latents",
    "encode_latents",
    "infer_shape",
    "latent_frames",
    "pixel_frames",
    "readout",
]

#: 四组上下采样通道 ✓（参考实现 ``BYTEDANCE_BLOCK_OUT_CHANNELS`` ✓）。
BYTEDANCE_BLOCK_OUT_CHANNELS: tuple[int, ...] = (128, 256, 512, 512)
#: 潜变量通道数 ✓（16 ✓；编码器吐 32 = 均值 + logvar ✓）。
SEEDVR2_LATENT_CHANNELS = 16
#: 时间下采样/上采样块个数 ✓（2 ⇒ 时间比 2² = 4 ✓）。
SEEDVR2_TEMPORAL_SCALE_NUM = 2
#: 空间下采样比 ✓（3 次空间下采样 ⇒ 8 ✓）。
SEEDVR2_SPATIAL_SCALE = 8
#: 时间下采样比 ✓（``2 ** SEEDVR2_TEMPORAL_SCALE_NUM`` ✓）。
SEEDVR2_TEMPORAL_SCALE = 2**SEEDVR2_TEMPORAL_SCALE_NUM
#: 潜变量缩放因子 ✓（参考实现 ``BYTEDANCE_VAE_SCALING_FACTOR`` ✓；偏移恒 0 ✓）。
SEEDVR2_LATENT_SCALE = 0.9152
#: 组归一化的组数 ✓ —— **权重里读不出来** ✗（形状与组数无关 ✓），按参考实现钉 32 ✓。
SEEDVR2_NORM_GROUPS = 32

#: 显存真的不够时**建议**喂给 :meth:`SeedVR2Vae.decode` 的时间分块（**潜帧** ✓，**默认不生效** ✓✗）。
#: 取 4 的理由与 Wan 那两个建议值**不同** ✓✗：那两个是「拿数值换显存」✓，这个**不动数值** ✓
#: （块首接块外真实帧 ⇒ 与整段相等 ✓，见测试 ⑫ ✓）⇒ 唯一的代价是吞吐/峰值权衡，不是画质 ✓。
#: 为什么不是 1：每块都要把**整条解码器**的前几层重跑一遍上文 ✓（首帧 ＋ 块间重叠的计算量按块数
#: 上升 ✓），切到 1 帧在 13 潜帧的片子上是 12 块 ✓ ⇒ 时间 ×2 那几层的重复计算约翻倍 ✓；
#: 4 潜帧 ≈ 每块 ~13 像素帧 ✓，把「时间 ×2 上采样后」那几笔 17.3 GiB 的重排张量压到 1/3 上下 ✓。
#:
#: ⚠️ 它只是**兜底的起点**，不是「一定装得下」✗：峰值 ≈ 与**每块像素帧数**（``4L`` ✓）成正比 ✓，
#: 而 720×480 这种真分辨率下**空间**那几笔本身就很大 ✓ —— 一块 4 潜帧（17 像素帧 ✓）实测已接近
#: 22.49 GiB 的卡 ✓。因此兜底要走 :func:`chunk_ladder` 逐档下探 ✓，别写成「降一档就收工」✗✗。
SUGGESTED_TEMPORAL_CHUNK = 4


def chunk_ladder(start: int | None = None) -> tuple[int, ...]:
    """OOM 兜底的**下探阶梯** ✓：从 ``start``（默认 :data:`SUGGESTED_TEMPORAL_CHUNK`）逐档减半到 1 ✓。

    只减半、不跳档 ✓：峰值 ≈ 与每块像素帧数成正比 ✓ ⇒ 减半就是唯一的整数级台阶 ✓；
    ``1`` 是地板 ✓（切到 1 每块仍吐 ``4L-3=1`` 帧 ✓、仍胜过整段 ✗）。``4`` ⇒ ``(4, 2, 1)`` ✓。
    """
    top = SUGGESTED_TEMPORAL_CHUNK if start is None else int(start)
    if top < 1:
        raise SeedVR2VaeError(
            f"时间分块的起点必须 ≥1 ✗（收到 {top} ✓）：``None``/``0`` 是「整段」的写法 ✓，"
            f"要分块就给正数 ✓")
    steps: list[int] = []
    while top >= 1:
        steps.append(top)
        top //= 2
    return tuple(dict.fromkeys(steps))


def shrink_for_peak(chunk: int, peak_bytes: int, total_bytes: int) -> int | None:
    """按「峰值**超出物理显存**」反推该降到哪一档 ✓ —— 纯函数 ✓、不测量不猜 ✓。

    ⚠️ 为什么需要它 ✗✗：这份卡上 ``temporalChunk=4`` 在 720×480 下**根本不报 OOM** ✓ ——
    Windows 驱动把超出的部分落到主机内存 ✓（实测 reserved 86.67 GiB / 卡 22.49 GiB ✓），
    代价是 **231.1 秒 vs 一档 8.3 秒（28 倍）** ✓✗ ⇒ 只等 OOM 的兜底**永远等不到触发** ✗。

    估算口径：峰值 ≈ 与每块像素帧数（``4L``）成正比 ✓ ⇒ 需要降到 ``chunk × 总显存 ÷ 实测峰值`` ✓，
    再**向下取阶梯档** ✓。用 ``reserved`` 而不是 ``allocated`` 是**故意偏保守** ✓（切小只会更省 ✓，
    切大了才要再慢一遍 ✓）。返 ``None`` ＝ 没溢出、不用动 ✓。
    """
    if chunk < 1 or total_bytes <= 0 or peak_bytes <= total_bytes:
        return None
    want = max(1, int(chunk * total_bytes / peak_bytes))
    steps = [step for step in chunk_ladder(chunk) if step < chunk and step <= want]
    return steps[0] if steps else None


class SeedVR2VaeError(ValueError):
    """SEEDVR2 VAE 的结构 / 输入不合口径 ✓（消息写人话 ✓）。"""


# ---------------------------------------------------------------- 形状反推 ----

@dataclass(frozen=True)
class SeedVR2VaeShape:
    """从权重反推出来的结构事实 ✓（一个通道数都不写死 ✓）。"""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = SEEDVR2_LATENT_CHANNELS
    block_out_channels: tuple[int, ...] = BYTEDANCE_BLOCK_OUT_CHANNELS
    layers_per_block: int = 2
    norm_num_groups: int = SEEDVR2_NORM_GROUPS
    temporal_scale_num: int = SEEDVR2_TEMPORAL_SCALE_NUM
    scale_factor: float = SEEDVR2_LATENT_SCALE
    #: 反推时留下的凭据 ✓（"哪个数字是从哪个键读的" ✓）。
    evidence: tuple[str, ...] = field(default=(), compare=False)

    @property
    def temporal_downscale_ratio(self) -> int:
        return 2**self.temporal_scale_num

    @property
    def spatial_downscale_ratio(self) -> int:
        return 2 ** (len(self.block_out_channels) - 1)

    #: ⚠️ ``torch_backend._check_attached_vaes`` 读的是**这两个名字** ✓（`vae_wan` /
    #: `qwen_image_vae` 也都长这样 ✓）—— 少写一个属性 ⇒ 挂 VAE 时对账被静默跳过 ✗
    #: （"没检查"不是绿灯 ✗）。
    @property
    def spatial_scale(self) -> int:
        return self.spatial_downscale_ratio

    @property
    def temporal_scale(self) -> int:
        return self.temporal_downscale_ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            "inChannels": self.in_channels,
            "outChannels": self.out_channels,
            "latentChannels": self.latent_channels,
            "blockOutChannels": list(self.block_out_channels),
            "layersPerBlock": self.layers_per_block,
            "normNumGroups": self.norm_num_groups,
            "temporalScaleNum": self.temporal_scale_num,
            "scaleFactor": self.scale_factor,
            "spatialDownscaleRatio": self.spatial_downscale_ratio,
            "temporalDownscaleRatio": self.temporal_downscale_ratio,
            "spatialScale": self.spatial_scale,
            "temporalScale": self.temporal_scale,
        }


def _shape_of(shapes: Mapping[str, Iterable[int]], key: str) -> tuple[int, ...] | None:
    value = shapes.get(key)
    if value is None:
        return None
    return tuple(int(x) for x in value)


def _children(shapes: Mapping[str, Iterable[int]], prefix: str) -> set[int]:
    """``prefix`` 后面紧跟的整数编号集合 ✓（只认数字段 ✓）。"""
    found: set[int] = set()
    for key in shapes:
        text = str(key)
        if not text.startswith(prefix):
            continue
        head = text[len(prefix):].split(".", 1)[0]
        if head.isdigit():
            found.add(int(head))
    return found


def shape_problems(shape: SeedVR2VaeShape,
                   shapes: Mapping[str, Iterable[int]] | None = None) -> tuple[str, ...]:
    """结构自洽检查 ✓（返回人话问题清单 ✓，空 = 通过 ✓）。"""
    problems: list[str] = []
    for value in (shape.in_channels, shape.out_channels, shape.latent_channels,
                  *shape.block_out_channels):
        if value <= 0:
            problems.append(f"通道数必须为正 ✗（收到 {value}）")
    if shape.layers_per_block < 1:
        problems.append(f"每层 ResNet 个数必须 ≥1 ✗（收到 {shape.layers_per_block}）")
    if shape.temporal_scale_num < 1:
        problems.append("时间下采样次数必须 ≥1 ✗（否则帧数铁律 4n+1 不成立 ✓）")
    if len(shape.block_out_channels) < 2:
        problems.append("下采样级数必须 ≥2 ✗（空间比 8 需要 3 次 ✓）")
    for value in shape.block_out_channels:
        if value % shape.norm_num_groups:
            problems.append(
                f"通道 {value} 不能被组数 {shape.norm_num_groups} 整除 ✗"
                "（组归一化会当场报错 ✓）")
            break
    if shapes is not None:
        for level, value in enumerate(shape.block_out_channels):
            #: 用 ``norm2``（**输出**通道 ✓）—— ``norm1`` 是这一级的输入通道 ✓，
            #: 拿它比会得到"每级都差一档"的假警报 ✗。
            key = f"encoder.down_blocks.{level}.resnets.0.norm2.weight"
            got = _shape_of(shapes, key)
            if got is not None and got[0] != value:
                problems.append(
                    f"{key} 通道 {got[0]} ≠ 反推的 {value} ✗（不是同一个结构 ✓）")
    return tuple(problems)


def infer_shape(shapes: Mapping[str, Iterable[int]]) -> SeedVR2VaeShape:
    """从 ``{键: 形状}`` 反推结构 ✓（读不出来就抛 ✗，不猜 ✓）。

    ``shapes`` 只要有键名 + 形状 ✓ —— safetensors 头、``state_dict``、普通字典都行 ✓。
    """
    evidence: list[str] = []
    conv_in = _shape_of(shapes, "encoder.conv_in.weight")
    conv_out = _shape_of(shapes, "decoder.conv_out.weight")
    if conv_in is None or conv_out is None:
        raise SeedVR2VaeError(
            "这不是 SEEDVR2 VAE 的权重 ✗ —— 少了 encoder.conv_in / decoder.conv_out ✓"
            f"（收到 {len(shapes)} 个键 ✓，顶层前缀 "
            f"{sorted({str(k).split('.')[0] for k in shapes})} ✗）")
    if len(conv_in) != 5 or len(conv_out) != 5:
        raise SeedVR2VaeError(
            f"卷积核不是 3D ✗（encoder.conv_in 是 {len(conv_in)} 维 ✓）—— "
            "SEEDVR2 是视频 VAE ✓，4 维核说明拿错了文件 ✗✗")

    in_channels = conv_in[1]
    #: ✗ 坑：``conv_in`` 的输出通道 == 第 0 级的通道 ✓（128 == 128 ✓），把它当成
    #: "第 0 级"再逐个 append 会凭空多出一级 ✓ ⇒ 级数 4 变 5 ✓、空间比 8 变 16 ✗。
    block_out_channels: list[int] = []
    level = 0
    while f"encoder.down_blocks.{level}.resnets.0.conv2.weight" in shapes:
        ids = _children(shapes, f"encoder.down_blocks.{level}.resnets.")
        last = max(ids)
        conv2 = _shape_of(
            shapes, f"encoder.down_blocks.{level}.resnets.{last}.conv2.weight")
        block_out_channels.append(conv2[0])
        level += 1
    if not block_out_channels:
        raise SeedVR2VaeError("找不到任何 encoder.down_blocks.N ✗（不是 SEEDVR2 VAE ✓）")
    evidence.append(f"in_channels={in_channels}（encoder.conv_in.weight[1] ✓）")
    if block_out_channels[0] != conv_in[0]:
        raise SeedVR2VaeError(
            f"encoder.conv_in 输出 {conv_in[0]} ≠ 第 0 级通道 {block_out_channels[0]} ✗"
            "（结构不自洽 ✓）")
    evidence.append(
        f"block_out_channels={tuple(block_out_channels)}（逐级 resnets.*.conv2.weight[0] ✓）")

    layers = len(_children(shapes, "encoder.down_blocks.0.resnets."))
    evidence.append(f"layers_per_block={layers}（数 encoder.down_blocks.0.resnets.* ✓）")

    temporal_scale_num = 0
    for index in range(len(block_out_channels)):
        down = _shape_of(shapes, f"encoder.down_blocks.{index}.downsamplers.0.conv.weight")
        if down is None:
            continue
        if len(down) != 5:
            raise SeedVR2VaeError(f"encoder.down_blocks.{index} 的卷积不是 3D ✗")
        if down[2] == 3:
            temporal_scale_num += 1
    if temporal_scale_num == 0:
        raise SeedVR2VaeError(
            "所有下采样都是纯空间核（t=1）✗ —— 这是 2D 图 VAE 的写法 ✓，不是 SEEDVR2 ✓")
    evidence.append(f"temporal_scale_num={temporal_scale_num}（数时间核 3 的下采样 ✓）")

    dec_conv_in = _shape_of(shapes, "decoder.conv_in.weight")
    if dec_conv_in is None:
        raise SeedVR2VaeError("缺 decoder.conv_in ✗（解码侧不完整 ✓）")
    evidence.append(f"latent_channels={dec_conv_in[1]}（decoder.conv_in.weight[1] ✓）")

    up_temporal: list[int] = []
    for index in range(len(block_out_channels)):
        up = _shape_of(shapes,
                       f"decoder.up_blocks.{index}.upsamplers.0.upscale_conv.weight")
        if up is None:
            up_temporal.append(0)
            continue
        ratio = up[0] // up[1] if up[1] else 0
        if ratio % 4:
            raise SeedVR2VaeError(
                f"decoder.up_blocks.{index} 的像素重排倍率 {ratio} 不是 4 的整数倍 ✗"
                "（参考实现固定 spatial² × temporal ✓）")
        up_temporal.append(2 if ratio // 4 == 2 else 1)
    evidence.append(f"upsamplers 时间倍率={tuple(up_temporal)}（upscale_conv 出/入 ✓）")
    if sum(1 for x in up_temporal if x == 2) != temporal_scale_num:
        raise SeedVR2VaeError(
            f"编解码时间口径不对称 ✗ —— 下采样 {temporal_scale_num} 次、上采样 "
            f"{sum(1 for x in up_temporal if x == 2)} 次 ✓（结构坏了 ✓）")

    shape = SeedVR2VaeShape(
        in_channels=in_channels,
        out_channels=conv_out[0],
        latent_channels=dec_conv_in[1],
        block_out_channels=tuple(block_out_channels),
        layers_per_block=layers,
        temporal_scale_num=temporal_scale_num,
        evidence=tuple(evidence),
    )
    problems = shape_problems(shape, shapes)
    if problems:
        raise SeedVR2VaeError("；".join(problems))
    return shape


# ------------------------------------------------------------------ 基础件 ----

class _MemoryState(Enum):
    """时间维分块的「缓存状态」✓（与参考实现 ``MemoryState`` 同义 ✓，取自
    ``comfy/ldm/seedvr/vae.py`` 的 ``MemoryState`` ✓ / ``InflatedCausalConv3d.basic_forward`` ✓）。

    * :attr:`DISABLED` —— 整段前向 ✓（今天的老路 ✓）：不记缓存、左 padding 复制首帧 ✓；
    * :attr:`INITIALIZING` —— **第一块** ✓：它是真·首帧 ⇒ 照旧复制首帧 ✓，但**要记**缓存 ✓；
    * :attr:`ACTIVE` —— 后续块 ✓：左 padding 换成缓存里的**块外真实帧** ✓（长度同为
      ``|stride - kernel|`` ⇒ 输出帧数不变 ✓），这才是逐位相等的关键 ✓✗。
    """

    DISABLED = "disabled"
    INITIALIZING = "initializing"
    ACTIVE = "active"


def _extend_head(tensor: torch.Tensor, times: int,
                 memory: torch.Tensor | None = None) -> torch.Tensor:
    """把首帧复制 ``times`` 份顶到最前 ✓（因果卷积的「左侧 padding」✓）。

    ``memory`` 非空 ⇒ **不复制首帧**，改成把块外的真实上文帧接在前面 ✓（``times`` 此时被
    忽略 ✓）—— 参考实现是同一个入口 ✓（``extend_head(tensor, times=-1, memory=memory)`` ✓）。
    """
    if memory is not None:
        return torch.cat((memory.to(device=tensor.device, dtype=tensor.dtype), tensor), dim=2)
    if times <= 0:
        return tensor
    return torch.cat((tensor[:, :, :1].repeat(1, 1, times, 1, 1), tensor), dim=2)


def _remove_head(tensor: torch.Tensor, times: int = 1) -> torch.Tensor:
    """保留首帧、丢掉第 ``1..times`` 帧 ✓ —— 时间翻倍后回到 4n+1 ✓。"""
    if times <= 0:
        return tensor
    return torch.cat((tensor[:, :, :1], tensor[:, :, times + 1:]), dim=2)


class _InflatedCausalConv3d(nn.Conv3d):
    """时间 padding 换成「复制首帧」的 3D 卷积 ✓。

    参考实现只在 :class:`torch.nn.Conv3d` 外面套这一层 ✓ —— 权重形状与普通 ``Conv3d``
    **逐字相同** ✓，所以原始键名可以直进 ✓。``inflation_mode``（pad/tail）在参考实现里
    **不参与前向** ✗（只影响 2D 权重如何被"充气"，此处权重本来就是 3D ✓）⇒ 不必带 ✓。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.padding, str):
            raise SeedVR2VaeError(
                f"padding 不能是 {self.padding!r} ✗ —— 因果卷积要显式三元组 ✓")
        self.temporal_padding = int(self.padding[0])
        #: 时间维不再零填充 ✓（0）✓，空间维照旧 ✓。
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x: torch.Tensor,  # type: ignore[override]
                memory_state: _MemoryState | None = None,
                memory_cache: dict[Any, torch.Tensor] | None = None) -> torch.Tensor:
        """⚠️ 三个形状全对、只有**跨块缓存**错就会「块与块之间留一道」✗ —— 见 :class:`_MemoryState` ✓。

        缓存长度的口径与参考实现逐字相同 ✓：``mem_size = stride[0] - kernel[0]`` ✓
        （3×3×3 / 步长 1 ⇒ ``-2`` ⇒ 记**最后 2 帧** ✓；1×1×1 与时间核 1 ⇒ ``0`` ⇒ **不进缓存** ✓；
        时间步长 2 的下采样 ⇒ ``-1`` ⇒ 记最后 1 帧 ✓）。
        """
        x = x.to(self.weight.dtype)
        mem_size = self.stride[0] - self.kernel_size[0]
        memory = memory_cache.get(self) if memory_cache is not None else None
        if memory is not None and memory_state is _MemoryState.ACTIVE:
            x = _extend_head(x, times=-1, memory=memory)
        else:
            x = _extend_head(x, times=self.temporal_padding * 2)
        if memory_cache is not None and memory_state is not _MemoryState.DISABLED:
            #: ⚠️ 记的是**本层输入**（含刚接上的上文 ✓）的最后几帧 ✓ —— 记输出就错 ✗：
            #: 下一块要补的是「这一层看到的原始上文」✓，不是混合过的特征 ✓。
            if mem_size:
                #: ⚠️⚠️ 必须 ``.clone()`` ✗✗：切片的 ``.detach()`` **不复制存储** ✓ ⇒ 缓存里那几帧
                #: 是**整块激活的视图** ✓ ⇒ 每层都把这一块的整份激活钉住不放 ✓ ⇒ 自建缓存变成
                #: 「块内整份前向激活全程常驻」✗，实测 720×480 / 7 潜帧 chunk=2 峰值 22.62 GiB，
                #: 反而高于整段 4 潜帧（13 帧）的 13.48 GiB ✗✗ —— 分块比整段还费 ✓，兜底白搭 ✓。
                #: 拷出来只有 ``mem_size`` 帧（≤2 帧）✓，代价可以忽略 ✓。
                memory_cache[self] = x[:, :, mem_size:].detach().clone()
            else:
                memory_cache.pop(self, None)
        return F.conv3d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class _GroupNorm5d(nn.GroupNorm):
    """组归一化 ✓ —— **直接在 5D 上按 (组内通道, 高, 宽) 统计** ✓（逐帧统计 ✓，绝不跨帧 ✓）。

    写这样是为了**不拷贝** ✓：原先走「``transpose(1,2).reshape(B*T,C,H,W)`` → ``F.group_norm``
    → 再 transpose 回来」✓，数值一样 ✓（`tests/engine_seedvr2_vae_test.py` ⑧ 的金标就是它 ✓），
    但转置后非连续 ⇒ ``reshape`` 必拷 ✓、``group_norm`` 再出一份 ✓、回程再一次 ✓ ⇒ 全分辨率
    那一层光归一化就多三份整张量 ✓。现在 ``reshape`` 只是**视图** ✓、只出 ``out`` 一份 ✓。

    ⚠️ 但这**不是** 720p 解码慢的主因 ✓（实测换成这版后峰值仍是 47.60 GiB / 336 秒 ✓，见
    :func:`decode_latents` 的账 ✓）—— 大头在「时间 ×2 上采样之后」的单笔 17.3 GiB 张量 ✓，
    别把这条当性能优化成果去汇报 ✗。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() != 5:
            return super().forward(x)
        dtype = self.weight.dtype
        batch, channels, frames, height, width = x.shape
        groups = self.num_groups
        work = x.to(dtype)
        shaped = work.reshape(batch, groups, channels // groups, frames, height, width)
        #: ⚠️ 归约维必须是 ``(组内通道, 高, 宽)`` ✓，**把帧维放过** ✓ —— 写成
        #: ``dim=(2,3,4,5)`` 就是把**整段所有帧**一起归一 ✓✗（数值上等于 ``F.group_norm``
        #: 直接吃 5D ✓，看着"也说得通" ✓，但那是**跨帧统计** ✗，会毁掉因果性 ✓：
        #: 末帧一改，首帧跟着变 ✗）。
        mean = shaped.mean(dim=(2, 4, 5), keepdim=True)
        var = shaped.var(dim=(2, 4, 5), unbiased=False, keepdim=True)
        out = (shaped - mean) / torch.sqrt(var + self.eps)
        out = out.reshape(batch, channels, frames, height, width)
        scale = self.weight.reshape(1, channels, 1, 1, 1)
        shift = self.bias.reshape(1, channels, 1, 1, 1)
        return out * scale + shift


class _Attention(nn.Module):
    """逐帧空间自注意力 ✓（单头 ✓，``SDPA`` 默认缩放 ✓，残差 + 归一化 ✓）。

    ✗ 坑：归一化作用在 ``(B, C, N)`` 上（通道在前 ✓）—— 写成 ``(B, N, C)`` 会按 token
    归一到「所有通道同一个均值」✓，形状全对、数值全错 ✗。
    """

    def __init__(self, query_dim: int, *, heads: int, dim_head: int,
                 norm_num_groups: int, eps: float = 1e-6,
                 residual_connection: bool = True,
                 rescale_output_factor: float = 1.0) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.inner_dim = inner_dim
        self.residual_connection = residual_connection
        self.rescale_output_factor = rescale_output_factor
        self.group_norm = _GroupNorm5d(norm_num_groups, query_dim, eps=eps, affine=True)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=True)
        #: ✗ 键名是 ``to_out.0``（ModuleList：线性 + dropout ✓）—— 直接写 ``to_out``
        #: 会让权重整份"多出来" ✓，症状是 4 个 unexpected ✗。
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim, bias=True),
                                     nn.Dropout(0.0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise SeedVR2VaeError(
                f"注意力只吃逐帧 4D 张量 ✗（收到 {tuple(x.shape)} ✓）—— "
                "中段块会先把 (B,T,C,H,W) 拍平成 (B·T,C,H,W) ✓")
        residual = x
        batch, channels, height, width = x.shape
        hidden = x.reshape(batch, channels, height * width).transpose(1, 2)
        hidden = self.group_norm(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = hidden.to(self.to_q.weight.dtype)

        head_dim = self.inner_dim // self.heads
        query = self.to_q(hidden).view(batch, -1, self.heads, head_dim).transpose(1, 2)
        key = self.to_k(hidden).view(batch, -1, self.heads, head_dim).transpose(1, 2)
        value = self.to_v(hidden).view(batch, -1, self.heads, head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(query, key, value)
        out = out.transpose(1, 2).reshape(batch, -1, self.inner_dim).to(query.dtype)
        out = self.to_out[1](self.to_out[0](out))
        out = out.transpose(-1, -2).reshape(batch, channels, height, width)

        if self.residual_connection:
            out = out + residual.to(out.dtype)
        return out / self.rescale_output_factor


class _ResnetBlock3d(nn.Module):
    """3D 残差块 ✓（``norm→SiLU→conv`` 两遍 ✓；通道不等时 1×1 短路 ✓）。

    本族权重里**没有** ``time_emb_proj`` ✗（``temb_channels=None`` ✓）⇒ 不建、不猜 ✓。
    """

    def __init__(self, in_channels: int, out_channels: int, *, groups: int,
                 eps: float = 1e-6, output_scale_factor: float = 1.0,
                 temporal_kernel: int = 3) -> None:
        super().__init__()
        self.output_scale_factor = output_scale_factor
        self.norm1 = _GroupNorm5d(groups, in_channels, eps=eps, affine=True)
        self.norm2 = _GroupNorm5d(groups, out_channels, eps=eps, affine=True)
        self.conv1 = _InflatedCausalConv3d(
            in_channels, out_channels, kernel_size=(temporal_kernel, 3, 3),
            stride=1, padding=(temporal_kernel // 2, 1, 1))
        self.conv2 = _InflatedCausalConv3d(out_channels, out_channels,
                                           kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = _InflatedCausalConv3d(in_channels, out_channels,
                                                       kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor,
                memory_state: _MemoryState | None = None,
                memory_cache: dict[Any, torch.Tensor] | None = None) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)), memory_state, memory_cache)
        hidden = self.conv2(F.silu(self.norm2(hidden)), memory_state, memory_cache)
        shortcut = (x if self.conv_shortcut is None
                    else self.conv_shortcut(x, memory_state, memory_cache))
        return (shortcut + hidden) / self.output_scale_factor


class _Downsample3d(nn.Module):
    """空间（可选时间）下采样 ✓ —— 空间侧先「右下」补一行一列再 3×3 步长 2 ✓。

    ✗ 坑：补边是**单侧** ✓（``(0,1,0,1)`` = 右 1 / 下 1 ✓），不是对称补 ✓；
    对称补会让 3×3 窗口跨到左上零区 ✓，形状一样、数值错 ✗。
    """

    def __init__(self, channels: int, *, temporal_down: bool,
                 spatial_down: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.temporal_down = temporal_down
        self.spatial_down = spatial_down
        temporal_kernel = 3 if temporal_down else 1
        spatial_kernel = 3 if spatial_down else 1
        self.conv = _InflatedCausalConv3d(
            channels, channels,
            kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
            stride=(2 if temporal_down else 1,
                    2 if spatial_down else 1,
                    2 if spatial_down else 1),
            padding=(1 if temporal_down else 0, 0, 0))

    def forward(self, x: torch.Tensor,
                memory_state: _MemoryState | None = None,
                memory_cache: dict[Any, torch.Tensor] | None = None) -> torch.Tensor:
        if self.spatial_down:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x, memory_state, memory_cache)


class _Upsample3d(nn.Module):
    """像素重排上采样 ✓（空间 ×2 ✓；时间 ×2 之后**丢第 2 帧** ✓）。

    ``view(B, sr, sr, tr, C, T, H, W)`` 的维度顺序是硬口径 ✓：
    ``permute(0,4,5,3,6,1,7,2)`` = ``(B, C, T·tr, H·sr, W·sr)`` ✓。
    """

    def __init__(self, channels: int, *, temporal_up: bool,
                 spatial_up: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.temporal_up = temporal_up
        self.spatial_up = spatial_up
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2 if spatial_up else 1
        ratio = self.temporal_ratio * self.spatial_ratio * self.spatial_ratio
        self.upscale_conv = nn.Conv3d(channels, channels * ratio, kernel_size=1,
                                      padding=0)
        self.conv = _InflatedCausalConv3d(channels, channels, kernel_size=3,
                                          stride=1, padding=1)

    def forward(self, x: torch.Tensor,
                memory_state: _MemoryState | None = None,
                memory_cache: dict[Any, torch.Tensor] | None = None) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise SeedVR2VaeError(
                f"上采样期望 {self.channels} 通道 ✗（收到 {x.shape[1]} ✓）")
        out = self.upscale_conv(x)
        batch, channels, frames, height, width = out.shape
        spatial, temporal = self.spatial_ratio, self.temporal_ratio
        inner = channels // (spatial * spatial * temporal)
        out = out.view(batch, spatial, spatial, temporal, inner, frames, height, width)
        out = out.permute(0, 4, 5, 3, 6, 1, 7, 2).reshape(
            batch, inner, frames * temporal, height * spatial, width * spatial)
        #: ⚠️ 「丢第 2 帧」是**全局首帧**才有的修正 ✓✗：块跑时（``ACTIVE``）那块首帧不是真首帧 ✓
        #: ⇒ **不能丢** ✗ —— 丢了就少一帧、拼回来整条错位 ✓；而块是**恰好平铺**潜帧的 ✓，
        #: 每块吐 ``4·L`` 帧、首块吐 ``1 + 4·(L-1)`` 帧 ✓，合起来正好 ``4T-3`` ✓（参考实现同一个
        #: 判据：``if self.temporal_up and memory_state != MemoryState.ACTIVE: remove_head`` ✓）。
        if self.temporal_up and memory_state is not _MemoryState.ACTIVE:
            out = _remove_head(out)
        return self.conv(out, memory_state, memory_cache)


class _MidBlock3d(nn.Module):
    """中段 ✓：ResNet → 逐帧注意力 → ResNet ✓。"""

    def __init__(self, channels: int, *, groups: int, attention_head_dim: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([
            _ResnetBlock3d(channels, channels, groups=groups, eps=eps),
            _ResnetBlock3d(channels, channels, groups=groups, eps=eps),
        ])
        self.attentions = nn.ModuleList([
            _Attention(channels, heads=channels // attention_head_dim,
                       dim_head=attention_head_dim, norm_num_groups=groups, eps=eps),
        ])

    def forward(self, x: torch.Tensor,
                memory_state: _MemoryState | None = None,
                memory_cache: dict[Any, torch.Tensor] | None = None) -> torch.Tensor:
        #: 注意力是**逐帧**空间注意力 ✓ ⇒ 天然可以按时间分块 ✓（不需要缓存 ✓）。
        x = self.resnets[0](x, memory_state, memory_cache)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            batch, channels, frames, height, width = x.shape
            flat = x.transpose(1, 2).reshape(batch * frames, channels, height, width)
            flat = attention(flat)
            x = flat.reshape(batch, frames, channels, height, width).transpose(1, 2)
            x = resnet(x, memory_state, memory_cache)
        return x


# ------------------------------------------------------------------ 编解码 ----

class _DownBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, num_layers: int,
                 groups: int, add_downsample: bool, temporal_down: bool,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([
            _ResnetBlock3d(in_channels if index == 0 else out_channels, out_channels,
                           groups=groups, eps=eps)
            for index in range(num_layers)
        ])
        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                _Downsample3d(out_channels, temporal_down=temporal_down),
            ])

    def forward(self, x: torch.Tensor,
                memory_state: _MemoryState | None = None,
                memory_cache: dict[Any, torch.Tensor] | None = None) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, memory_state, memory_cache)
        if self.downsamplers is not None:
            for down in self.downsamplers:
                x = down(x, memory_state, memory_cache)
        return x


class _UpBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, num_layers: int,
                 groups: int, add_upsample: bool, temporal_up: bool,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([
            _ResnetBlock3d(in_channels if index == 0 else out_channels, out_channels,
                           groups=groups, eps=eps)
            for index in range(num_layers)
        ])
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                _Upsample3d(out_channels, temporal_up=temporal_up),
            ])

    def forward(self, x: torch.Tensor,
                memory_state: _MemoryState | None = None,
                memory_cache: dict[Any, torch.Tensor] | None = None) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, memory_state, memory_cache)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                #: ⚠️ 逐层**显式**传 ✗✗ —— ``upsamplers`` 是 ``nn.ModuleList`` ✓，
                #: 写 ``up(x)`` 就会把状态丢掉 ✓（症状：能跑、帧数还对、块缝在 ✓）。
                x = up(x, memory_state, memory_cache)
        return x


class _Encoder3d(nn.Module):
    def __init__(self, shape: SeedVR2VaeShape) -> None:
        super().__init__()
        groups = shape.norm_num_groups
        block = shape.block_out_channels
        last = len(block) - 1
        self.conv_in = _InflatedCausalConv3d(shape.in_channels, block[0],
                                             kernel_size=3, stride=1, padding=1)
        downs = []
        output_channel = block[0]
        for index, out_channels in enumerate(block):
            downs.append(_DownBlock3d(
                output_channel, out_channels,
                num_layers=shape.layers_per_block, groups=groups,
                add_downsample=index != last,
                #: 参考实现：后 ``temporal_scale_num`` 个下采样才压时间 ✓
                #: （``index >= 级数 - 1 - temporal_scale_num`` ✓）✓。
                temporal_down=index >= last - shape.temporal_scale_num,
            ))
            output_channel = out_channels
        self.down_blocks = nn.ModuleList(downs)
        self.mid_block = _MidBlock3d(block[-1], groups=groups,
                                     attention_head_dim=block[-1])
        self.conv_norm_out = _GroupNorm5d(groups, block[-1], eps=1e-6, affine=True)
        #: 出 32 通道 = 16 均值 + 16 logvar ✓（**没有** quant_conv ✗）。
        self.conv_out = _InflatedCausalConv3d(block[-1], 2 * shape.latent_channels,
                                              kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor,
                memory_state: _MemoryState | None = None,
                memory_cache: dict[Any, torch.Tensor] | None = None) -> torch.Tensor:
        x = self.conv_in(x, memory_state, memory_cache)
        for block in self.down_blocks:
            x = block(x, memory_state, memory_cache)
        x = self.mid_block(x, memory_state, memory_cache)
        return self.conv_out(F.silu(self.conv_norm_out(x)), memory_state, memory_cache)


class _Decoder3d(nn.Module):
    def __init__(self, shape: SeedVR2VaeShape) -> None:
        super().__init__()
        groups = shape.norm_num_groups
        block = shape.block_out_channels
        reversed_block = tuple(reversed(block))
        last = len(reversed_block) - 1
        self.conv_in = _InflatedCausalConv3d(shape.latent_channels, reversed_block[0],
                                             kernel_size=3, stride=1, padding=1)
        self.mid_block = _MidBlock3d(reversed_block[0], groups=groups,
                                     attention_head_dim=reversed_block[0])
        ups = []
        output_channel = reversed_block[0]
        for index, out_channels in enumerate(reversed_block):
            ups.append(_UpBlock3d(
                output_channel, out_channels,
                #: 解码侧每组多一层 ✓（参考实现 ``layers_per_block + 1`` ✓）。
                num_layers=shape.layers_per_block + 1, groups=groups,
                add_upsample=index != last,
                temporal_up=index < shape.temporal_scale_num,
            ))
            output_channel = out_channels
        self.up_blocks = nn.ModuleList(ups)
        self.conv_norm_out = _GroupNorm5d(groups, reversed_block[-1], eps=1e-6,
                                          affine=True)
        self.conv_out = _InflatedCausalConv3d(reversed_block[-1], shape.out_channels,
                                              kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor,
                memory_state: _MemoryState | None = None,
                memory_cache: dict[Any, torch.Tensor] | None = None) -> torch.Tensor:
        x = self.conv_in(x, memory_state, memory_cache)
        x = self.mid_block(x, memory_state, memory_cache)
        for block in self.up_blocks:
            x = block(x, memory_state, memory_cache)
        return self.conv_out(F.silu(self.conv_norm_out(x)), memory_state, memory_cache)


def latent_frames(frames: int, shape: SeedVR2VaeShape | None = None) -> int:
    """像素帧数 → 潜变量帧数 ✓（``(T-1)/4 + 1`` ✓；不合 4n+1 当场人话拒绝 ✗）。"""
    ratio = (shape.temporal_downscale_ratio if shape is not None
             else SEEDVR2_TEMPORAL_SCALE)
    if frames < 1:
        raise SeedVR2VaeError(f"帧数必须 ≥1 ✗（收到 {frames}）")
    if (frames - 1) % ratio:
        raise SeedVR2VaeError(
            f"帧数必须是 4n+1 ✗（收到 {frames} ✓，时间比 {ratio} ✓）—— "
            f"最近的两个合法值是 {(frames - 1) // ratio * ratio + 1} 与 "
            f"{(frames - 1) // ratio * ratio + ratio + 1} ✓，先补帧再进来 ✓")
    return (frames - 1) // ratio + 1


def pixel_frames(frames: int, shape: SeedVR2VaeShape | None = None) -> int:
    """潜变量帧数 → 像素帧数 ✓（``(T-1)*4 + 1`` ✓）。"""
    ratio = (shape.temporal_downscale_ratio if shape is not None
             else SEEDVR2_TEMPORAL_SCALE)
    if frames < 1:
        raise SeedVR2VaeError(f"潜变量帧数必须 ≥1 ✗（收到 {frames}）")
    return (frames - 1) * ratio + 1


class SeedVR2Vae(nn.Module):
    """整条 VAE ✓（编码器 + 解码器 ✓，无 quant/post_quant ✓）。"""

    def __init__(self, shape: SeedVR2VaeShape) -> None:
        super().__init__()
        problems = shape_problems(shape)
        if problems:
            raise SeedVR2VaeError("；".join(problems))
        self.shape = shape
        self.encoder = _Encoder3d(shape)
        self.decoder = _Decoder3d(shape)

    # -- 尺寸口径 --------------------------------------------------------
    def latent_frames(self, frames: int) -> int:
        return latent_frames(frames, self.shape)

    def pixel_frames(self, frames: int) -> int:
        return pixel_frames(frames, self.shape)

    def latent_size(self, height: int, width: int) -> tuple[int, int]:
        ratio = self.shape.spatial_downscale_ratio
        if height % ratio or width % ratio:
            raise SeedVR2VaeError(
                f"宽高必须是 {ratio} 的整数倍 ✗（收到 {width}×{height} ✓）—— "
                "先补到整数倍再进来 ✓（补边会写进 mask ✓，别在这里偷偷裁 ✓）")
        return height // ratio, width // ratio

    # -- 前向 ------------------------------------------------------------
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """``(B, 3, T, H, W)``（[0,1] 或 [-1,1] 由调用方负责 ✓）→ 原始潜变量 ✓。"""
        if pixels.dim() != 5:
            raise SeedVR2VaeError(
                f"编码输入必须是 (B,C,T,H,W) ✗（收到 {tuple(pixels.shape)} ✓）")
        if pixels.shape[1] != self.shape.in_channels:
            raise SeedVR2VaeError(
                f"编码输入通道应为 {self.shape.in_channels} ✗（收到 {pixels.shape[1]} ✓）")
        self.latent_frames(pixels.shape[2])
        self.latent_size(pixels.shape[3], pixels.shape[4])
        moments = self.encoder(pixels.to(self.encoder.conv_in.weight.dtype))
        #: 只取均值 ✓（``posterior.mode()`` ✓）；logvar 半边在确定性修复链里不用 ✓。
        return moments[:, :self.shape.latent_channels]

    def decode(self, latents: torch.Tensor, *,
               temporal_chunk: int | None = None) -> torch.Tensor:
        """原始潜变量 ✓ → ``(B, 3, T, H, W)`` ✓（宽高裁到偶数 ✓，与参考实现一致 ✓）。

        ``temporal_chunk``＝每块多少**潜帧** ✓（``None``/``0`` ⇒ 整段 ✓，逐位老路 ✓）。分块走
        :meth:`_decode_temporal_chunked` ✓（带块外真实上文 ⇒ **结果与整段相等** ✓）。
        """
        if latents.dim() != 5:
            raise SeedVR2VaeError(
                f"解码输入必须是 (B,C,T,H,W) ✗（收到 {tuple(latents.shape)} ✓）")
        if latents.shape[1] != self.shape.latent_channels:
            raise SeedVR2VaeError(
                f"解码输入通道应为 {self.shape.latent_channels} ✗"
                f"（收到 {latents.shape[1]} ✓）")
        latent = latents.to(self.decoder.conv_in.weight.dtype)
        chunk = int(temporal_chunk or 0)
        if chunk > 0 and latent.shape[2] - 1 > chunk:
            out = self._decode_temporal_chunked(latent, chunk)
        else:
            out = self.decoder(latent, _MemoryState.DISABLED)
        out = out.float()
        height, width = out.shape[-2] - out.shape[-2] % 2, out.shape[-1] - out.shape[-1] % 2
        return out[:, :, :, :height, :width]

    def _decode_temporal_chunked(self, latents: torch.Tensor,
                                 chunk: int) -> torch.Tensor:
        """沿时间维按 ``chunk`` 个**潜帧**切块解码 ✓ —— 每块的左 padding 用**块外真实帧** ✓。

        做法与参考实现（``comfy/ldm/seedvr/vae.py::slicing_decode`` ✓）同构：

        1. 首帧单独拎出来 ✓ ⇒ 第一块 = ``[真首帧] + 前 chunk 个后续帧`` ✓，跑
           ``INITIALIZING``（复制首帧 + **记**缓存 ✓）；
        2. 后续每块跑 ``ACTIVE`` ✓ ⇒ 每个因果卷积的左 padding 从缓存里取**上一块的真实输入帧** ✓，
           长度与"复制首帧"完全相同 ⇒ 输出帧数不变 ✓；
        3. 各块沿时间维直接拼 ✓ —— 块是**恰好平铺**潜帧的 ✓，首块吐 ``4L-3``、后续每块吐 ``4L`` ✓，
           合起来就是 ``4T-3`` ✓，一帧不多一帧不少 ✓（这一点由测试 ⑫ 的帧数与逐位两条判据兜住 ✓）。
        """
        cache: dict[Any, torch.Tensor] = {}
        head, rest = latents[:, :, :1], latents[:, :, 1:]
        pieces: list[torch.Tensor] = []
        for index, piece in enumerate(rest.split(chunk, dim=2)):
            if index == 0:
                pieces.append(self.decoder(torch.cat((head, piece), dim=2),
                                           _MemoryState.INITIALIZING, cache))
            else:
                pieces.append(self.decoder(piece, _MemoryState.ACTIVE, cache))
        return torch.cat(pieces, dim=2)


def build(shape: SeedVR2VaeShape, *, dtype: torch.dtype = torch.float32,
          device: torch.device | str | None = None) -> SeedVR2Vae:
    """按反推出来的结构建网 ✓（**不**在这里载权重 ✓）。"""
    vae = SeedVR2Vae(shape)
    vae = vae.to(device=device, dtype=dtype) if device is not None else vae.to(dtype=dtype)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def encode_latents(vae: SeedVR2Vae, pixels: torch.Tensor, *,
                   family: str = "seedvr2") -> torch.Tensor:
    """像素 → **模型空间**潜变量 ✓（VAE 原始值 × 0.9152 ✓，走 :mod:`latent_formats` ✓）。"""
    return latent_formats.process_in(family, vae.encode(pixels))


def decode_latents(vae: SeedVR2Vae, latents: torch.Tensor, *,
                   family: str = "seedvr2", to_unit: bool = False,
                   temporal_chunk: int | None = None) -> torch.Tensor:
    """**模型空间**潜变量 → 像素 ✓（先 ÷ 0.9152 回到 VAE 空间 ✓）。

    ``temporal_chunk``＝每块多少**潜帧** ✓（``None``/``0`` ⇒ 整段 ✓ = 逐位老路 ✓）。分块**不会**
    改变结果 ✓：块与块之间带的是**块外真实上文** ✓（:class:`_MemoryState` ＋逐层 ``memory_cache`` ✓，
    机制抄自 ``comfy/ldm/seedvr/vae.py`` 的 ``MemoryState``／``extend_head(memory=...)`` ✓，
    只搬机制、对外口按本仓口径重写 ✓），由测试 ⑫ 的「帧数 ``4T-3`` ＋ ``max|Δ| ≤ 1e-5``」兜住 ✓✗
    （残下的 ~1e-6 是 fp32 卷积在不同 ``T`` 上选的累加顺序不同 ✓，不是口径差 ✓；而"块首帧伪造"
    那种错是 ``1.8~2.8`` ✓ —— 差六个数量级 ✓，别拿这条门槛当挡箭牌 ✗）。
    结构性收益 ✓：峰值只与**每块帧数**有关 ✓（时间 ×2 上采样后的重排张量按块生成 ✓），
    与整段帧数无关 ✓ —— 实测账待真跑补 ✓（别先写数字 ✗）。

    **下面这段是两轮失败的反面教材 ✓✗，别再照着改一遍 ✓**：

    * 分块失败**不是**丢帧数算错 ✗：把丢帧数按因果相位数正成 ``4k-3``、k 加到 2/3/4/5/6、
      甚至让非首块整个跳过 :func:`_remove_head`（改丢 ``4k``），最大差仍是 **1.8~2.8** ✓；
      对整段做 ±8 帧平移扫描，**一帧都找不到逐位** ✓。
    * 根因在**因果链本身** ✓：:class:`_InflatedCausalConv3d` 用 :func:`_extend_head` 把**最左端那一帧
      复制 2 份**当左 padding ✓ —— 整段跑时它正好是真首帧 ✓；块跑时它变成**块首帧** ✗ ⇒ 块首帧
      **自己**的输出就错 ✓，严格因果把这点错送到块内**全部后续帧** ✓✗。实测（小模型、两级时间 ×2）：
      改输入第 8 帧 ⇒ 输出从第 ``4·8-3=29`` 帧起一变到底；改末帧 ⇒ 只变最后 4 帧 ✓。
      所以「补多少上文都救不回来」✗（补进来的上文帧自己也被当成块首 ✓）。
    * 正解＝**块首帧的左 padding 用块外真实帧** ✓，而且必须**逐层**带 ✓ —— 每一层都有自己的
      :func:`_extend_head` ✓，只在输入层补不够 ✗。这条已落地 ✓（见 :meth:`SeedVR2Vae.decode` ✓）。

    实测账（720×480、49 帧、13 潜帧、真权重 ✓，只装 VAE ✓）：整段解码 **336 秒** ✓、
    分配器峰值 **47.60 GiB** ✓（卡 22.49 ✓）。最大那几笔是「时间 ×2 上采样之后」的同名张量
    （``upscale_conv`` 出的 512/1024 通道 + ``view/permute/reshape`` 的重排 ✓），单笔 ≈ 17.3 GiB ✓
    ⇒ 全程在 OOM 重试里打滚 ✓ 但**能出对的片子** ✓（`ok=True` ✓，e2e 1.27 MB mp4 ✓）。
    别用降精度/降分辨率蒙混 ✗（修复族的输出分辨率就是交付物 ✓）。
    """
    out = vae.decode(latent_formats.process_out(family, latents),
                     temporal_chunk=temporal_chunk)
    if to_unit:
        out = out.add(1).div(2).clamp(0, 1)
    return out


def readout(shape: SeedVR2VaeShape) -> str:
    """一句话人话 ✓（写进回执 ✓）。"""
    channels = "/".join(str(x) for x in shape.block_out_channels)
    return (
        f"SEEDVR2 VAE：{len(shape.block_out_channels)} 级 {channels} ✓，"
        f"潜变量 {shape.latent_channels} 通道 ✓，"
        f"空间 ÷{shape.spatial_downscale_ratio} ✓、时间 ÷{shape.temporal_downscale_ratio} ✓"
        f"（帧数 4n+1 ✓），潜变量缩放 ×{shape.scale_factor:g} ✓，"
        "因果 3D 卷积 ✓，注意力只在逐帧空间 ✓")
