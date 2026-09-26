"""**潜空间口径表** ✓（scale / shift / 通道数 / 维数 / 下采样比 —— 一份表说清 ✓ 零依赖 ✓）。

## 解决什么
潜变量在不同模型族里**不是一个东西** ✗：SD15 是 4 通道 × ``0.18215`` ✓，SDXL 是 4 通道 × ``0.13025`` ✓，
SD3 / Flux 是 16 通道且**先减 shift 再乘 scale** ✓，H3 视频是 24 通道 3 维 ✓，音频是 1 维 ✓。
这些数字**各写一遍** ⇒ 必然漂 ✓✗（改采样端忘解码端 = 出「看着对的错图」✓✗）⇒ 本模块给**一张表** ✓。

## 口径（搬语义 ✓ 不搬源码 ✗）
* **一个 spec 一个数**：``scale_factor`` / ``shift_factor`` / ``latent_channels`` / ``latent_dimensions`` ✓；
* **``process_in``**：潜变量 → 模型空间（SD15/SDXL **乘** scale ✓；SD3/Flux **先减 shift 再乘** ✓）；
* **``process_out``**：模型空间 → 潜变量（**逆运算** ✓）；
* ⚠️ 换算**不止「乘个数」** ✗：还有「按均值/标准差归一」✓（Wan / Mochi / SDXL-Playground ✓）与
  「恒等」✓（像素空间族 ✓）⇒ 用 :data:`PROCESS_*` 标签区分 ✓，**不拿一个公式硬套所有族** ✗✗。

## 不猜
* **``PROCESS_MEANSTD``**：mean/std 属于**具体权重** ✓ ⇒ 本表**不假装有** ✗，调用方自带 ✓，不带就**报错** ✗；
* **``PROCESS_REARRANGE``**：换算里夹着 permute/reshape/拼帧 ✓ ⇒ 本模块**不实现** ✗ 且调用即**报错** ✓
  （只做「乘个数」= 少了一半重排的错值 ✓✗，比失败更贵 ✗）；
* **认不出的名字** ⇒ **报错** ✗（**不静默退回默认 4 通道** ✗✗ —— 那等于把所有族当 SD1.5 用 ✓✗）；
* **H3 的 24 通道不是这里发明的** ✓：引用 :data:`latent_container.H3_VIDEO_CHANNELS` ✓
  （与 ``h3_form.H3_TRUNK_DEFAULTS["latents_dim"]`` 同一个事实 ✓，自检逐值比对 ✓）；
* **音频 32 通道同理** ✓：与 ``h3_form.H3_TRUNK_DEFAULTS["audio_latents_dim"]`` 同源 ✓（自检比对 ✓）；
* ⚠️ **下采样比只有 H3 是实测的** ✓（16× 空间 / 4× 时间 ✓，见 ``vae_h3`` ✓）；其余族保持基类默认
  （8 / 1 ✓）—— **别当实测值拿去算尺寸** ✗（尺寸/帧数走 ``vae_h3`` 等既有口径 ✓）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .latent_container import H3_VIDEO_CHANNELS

__all__ = [
    "H3_AUDIO_LATENT_CHANNELS",
    "LATENT_FORMATS",
    "PROCESS_AFFINE",
    "PROCESS_IDENTITY",
    "PROCESS_MEANSTD",
    "PROCESS_REARRANGE",
    "PROCESS_SCALE",
    "LatentFormatError",
    "LatentFormatSpec",
    "get_latent_format",
    "list_latent_formats",
    "normalize_latent_format",
    "process_in",
    "process_out",
]

#: 值域换算：**乘** ``scale_factor`` ✓（SD15 / SDXL 一族 ✓）。
PROCESS_SCALE = "scale"
#: 值域换算：**先减** ``shift_factor`` **再乘** ``scale_factor`` ✓（SD3 / Flux ✓）。
PROCESS_AFFINE = "affine"
#: 值域换算：**原样返回** ✓（像素空间族 ✓）。
PROCESS_IDENTITY = "identity"
#: 值域换算：``(x − mean) × scale / std`` ✓ —— ⚠️ mean/std 属**权重**，本表不存 ✗（调用方自带 ✓）。
PROCESS_MEANSTD = "meanstd"
#: 值域换算：缩放**之外**还夹张量重排 ✓ —— ⚠️ 本模块**不实现** ✗（调用即报错 ✓）。
PROCESS_REARRANGE = "rearrange"

#: 全部合法换算标签 ✓（``__post_init__`` 校验用 ✓）。
PROCESS_MODES = (PROCESS_SCALE, PROCESS_AFFINE, PROCESS_IDENTITY, PROCESS_MEANSTD, PROCESS_REARRANGE)

#: H3 **音频**侧潜通道数 ✓ —— ⚠️ 与 ``h3_form.H3_TRUNK_DEFAULTS["audio_latents_dim"]`` 是**同一个事实** ✗
#: （自检逐值比对 ✓；谁改了另一边这里会红 ✓）。注意：**不是** ``geometry.AUDIO_LATENT_CHANNELS`` ✓
#: —— 那个是**立体声声声道数 = 2** ✓，两码事 ✓✗。
H3_AUDIO_LATENT_CHANNELS = 32


class LatentFormatError(ValueError):
    """口径**认不出 / 换算做不了** ✓ ⇒ 当场报 ✗（给个「差不多」的数比失败更贵 ✓✗）。"""


@dataclass(frozen=True)
class LatentFormatSpec:
    """一个模型族的潜空间口径 ✓（**不可变** ✓ —— 口径是事实，不是可调参数 ✓✗）。"""

    #: 规范名 ✓（表里唯一 ✓，也是 :func:`get_latent_format` 的主键 ✓）。
    name: str
    #: 数值缩放因子 ✓（``process_in`` 乘它 ✓ / ``process_out`` 除它 ✓）。
    scale_factor: float = 1.0
    #: 偏移因子 ✓（仅 :data:`PROCESS_AFFINE` 用 ✓，其余必须为 ``None`` ✗）。
    shift_factor: float | None = None
    #: 潜通道数 ✓（第 2 维 ✓）。
    latent_channels: int = 4
    #: 潜维数 ✓（2 = 图像 ✓ / 3 = 视频 ✓ / 1 = 音频 ✓）。
    latent_dimensions: int = 2
    #: 空间下采样比 ✓（⚠️ 除 H3 外均为**基类默认 8** ✗ —— 未采集到，别当实测值 ✓✗）。
    spatial_downscale_ratio: int = 8
    #: 时间下采样比 ✓（⚠️ 同上，除 H3 外为默认 1 ✗）。
    temporal_downscale_ratio: int = 1
    #: 配套 TAESD 解码器名 ✓（``None`` = 没有 ✓）。
    taesd_decoder_name: str | None = None
    #: 换算语义 ✓（:data:`PROCESS_*` 之一 ✓）。
    process: str = PROCESS_SCALE
    #: 别名 ✓（大小写/写法变体 ✓，供 :func:`normalize_latent_format` 用 ✓）。
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise LatentFormatError("口径名不能为空 ✗")
        if self.process not in PROCESS_MODES:
            raise LatentFormatError(
                f"换算标签 {self.process!r} 不合法 ✗ —— 只能是 {PROCESS_MODES} ✓")
        if self.latent_channels <= 0 or self.latent_dimensions <= 0:
            raise LatentFormatError(
                f"{self.name!r}：通道数 / 维数必须为正 ✗（收到 {self.latent_channels} / "
                f"{self.latent_dimensions} ✓）")
        if self.spatial_downscale_ratio <= 0 or self.temporal_downscale_ratio <= 0:
            raise LatentFormatError(f"{self.name!r}：下采样比必须为正 ✗")
        if self.process == PROCESS_AFFINE and self.shift_factor is None:
            raise LatentFormatError(
                f"{self.name!r}：标签是 {PROCESS_AFFINE} 却没给 ``shift_factor`` ✗ —— "
                f"少减一个数就是**偏移的**潜变量 ✓✗")
        if self.process == PROCESS_AFFINE and not self.scale_factor:
            raise LatentFormatError(f"{self.name!r}：``scale_factor`` 不能为 0 ✗（除零 ✓✗）")
        if self.process not in (PROCESS_AFFINE, PROCESS_REARRANGE) and self.shift_factor is not None:
            raise LatentFormatError(
                f"{self.name!r}：换算标签是 {self.process} 却填了 ``shift_factor="
                f"{self.shift_factor}`` ✗ —— 这个标签**不用它** ✓，填了说明标签选错 ✓✗")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "process": self.process,
                "scaleFactor": self.scale_factor, "shiftFactor": self.shift_factor,
                "latentChannels": self.latent_channels, "latentDimensions": self.latent_dimensions,
                "spatialDownscaleRatio": self.spatial_downscale_ratio,
                "temporalDownscaleRatio": self.temporal_downscale_ratio,
                "taesdDecoderName": self.taesd_decoder_name, "aliases": list(self.aliases)}


def _build_specs() -> tuple[LatentFormatSpec, ...]:
    """全部口径 ✓（表就是这张 ✓ —— 顺序无关 ✓，重名/别名冲突在索引期报错 ✓）。"""
    S = LatentFormatSpec  # noqa: N806 —— 局部别名只为让这张表读起来像表格 ✓
    return (
        # ── 图像（2 维 ✓）───────────────────────────────────────────────
        S("sd15", scale_factor=0.18215, latent_channels=4, latent_dimensions=2,
          taesd_decoder_name="taesd_decoder", aliases=("sd1.5", "sd1_5", "sd_1_5", "sd15_inpaint")),
        S("sdxl", scale_factor=0.13025, latent_channels=4, latent_dimensions=2,
          taesd_decoder_name="taesdxl_decoder", aliases=("sd_xl", "sd_xl_base")),
        S("sdxl_playground_2_5", scale_factor=0.5, latent_channels=4, latent_dimensions=2,
          taesd_decoder_name="taesdxl_decoder", process=PROCESS_MEANSTD,
          aliases=("playground_2_5",)),
        S("sd3", scale_factor=1.5305, shift_factor=0.0609, latent_channels=16, latent_dimensions=2,
          taesd_decoder_name="taesd3_decoder", process=PROCESS_AFFINE,
          aliases=("sd3_5", "stable_diffusion_3", "sd3.5")),
        S("flux", scale_factor=0.3611, shift_factor=0.1159, latent_channels=16, latent_dimensions=2,
          taesd_decoder_name="taef1_decoder", process=PROCESS_AFFINE, aliases=("flux1", "flux_1")),
        S("flux2", scale_factor=1.0, latent_channels=128, latent_dimensions=2,
          taesd_decoder_name="taef2_decoder", process=PROCESS_IDENTITY, aliases=("flux_2",)),
        S("hunyuan_image_21", scale_factor=0.75289, latent_channels=64, latent_dimensions=2,
          aliases=("hunyuan_image_2_1",)),
        S("hunyuan_image_21_refiner", scale_factor=1.03682, latent_channels=64, latent_dimensions=3,
          process=PROCESS_REARRANGE, aliases=("hunyuan_image_2_1_refiner",)),
        S("sd_x4", scale_factor=1.0 / 12.0, latent_channels=4, latent_dimensions=2,
          aliases=("sd_x4_upscaler", "esrgan_x4")),  # 1/12 ≈ 0.08333 ✓（结构超分 ✓）
        S("sc_prior", scale_factor=1.0, latent_channels=16, latent_dimensions=2,
          aliases=("stable_cascade_prior",)),
        S("sc_b", scale_factor=1.0 / 0.43, latent_channels=4, latent_dimensions=2,
          aliases=("stable_cascade_b",)),  # ≈ 2.3256 ✓
        S("pixel_space", scale_factor=1.0, latent_channels=3, latent_dimensions=2,
          process=PROCESS_IDENTITY,
          aliases=("chroma_radiance", "z_image_pixel", "hidream_o1_pixel", "pixel_dit_pixel")),
        # ── 视频（3 维 ✓）───────────────────────────────────────────────
        S("h3_video", scale_factor=1.0, latent_channels=H3_VIDEO_CHANNELS, latent_dimensions=3,
          spatial_downscale_ratio=16, temporal_downscale_ratio=4, taesd_decoder_name="taeh3",
          aliases=("minimax_h3", "minimax_h3_video", "h3")),  # 24 通道引 latent_container ✓
        S("hunyuan_video", scale_factor=0.476986, latent_channels=16, latent_dimensions=3,
          taesd_decoder_name="taehv", aliases=("hunyuanvideo", "hunyuan_video_1")),
        S("hunyuan_video_15", scale_factor=1.03682, latent_channels=32, latent_dimensions=3,
          taesd_decoder_name="lighttaehy1_5", aliases=("hunyuan_video_1_5",)),
        S("wan21", scale_factor=1.0, latent_channels=16, latent_dimensions=3, process=PROCESS_MEANSTD,
          taesd_decoder_name="lighttaew2_1", aliases=("wan_2_1", "wan2.1")),
        S("wan22", scale_factor=1.0, latent_channels=48, latent_dimensions=3, process=PROCESS_MEANSTD,
          taesd_decoder_name="lighttaew2_2", aliases=("wan_2_2", "wan2.2")),
        S("ltxv", scale_factor=1.0, latent_channels=128, latent_dimensions=3,
          aliases=("ltx_video", "ltx")),
        S("ltxav", scale_factor=1.0, latent_channels=128, latent_dimensions=3,
          aliases=("ltx_av",)),
        S("mochi", scale_factor=1.0, latent_channels=12, latent_dimensions=3,
          process=PROCESS_MEANSTD, aliases=("mochi_1",)),
        S("cosmos_1_c_v8x8x8", scale_factor=1.0, latent_channels=16, latent_dimensions=3,
          aliases=("cosmos", "cosmos1")),
        S("cogvideox", scale_factor=1.15258426, latent_channels=16, latent_dimensions=3,
          aliases=("cog_video_x",)),
        S("cogvideox_1_5", scale_factor=0.7, latent_channels=16, latent_dimensions=3,
          aliases=("cog_video_x_1_5",)),
        S("seedvr2", scale_factor=1.0, latent_channels=16, latent_dimensions=3,
          aliases=("seed_vr2",)),
        # ── 音频（1 维 ✓）───────────────────────────────────────────────
        S("h3_audio", scale_factor=1.0, latent_channels=H3_AUDIO_LATENT_CHANNELS,
          latent_dimensions=3, spatial_downscale_ratio=1,
          # 32 通道与 h3_form 同源 ✓；⚠️ 本仓 torch_backend 另带**立体声**那一维 ⇒ 实造 B×32×2×T ✓
          aliases=("minimax_h3_audio", "h3_audio_latents")),
        S("audio_16khz_2ch", scale_factor=1.0, latent_channels=8, latent_dimensions=1,
          spatial_downscale_ratio=1, aliases=("audio_16k",)),
        S("audio_44khz_2ch", scale_factor=1.0, latent_channels=8, latent_dimensions=1,
          spatial_downscale_ratio=1, aliases=("audio_44k", "audio_44_1khz")),
        S("audio_48khz_2ch", scale_factor=1.0, latent_channels=8, latent_dimensions=1,
          spatial_downscale_ratio=1, aliases=("audio_48k",)),
        S("stable_audio_1", scale_factor=1.0, latent_channels=64, latent_dimensions=1,
          spatial_downscale_ratio=1, aliases=("stable_audio",)),
        S("stable_audio_3", scale_factor=1.0, latent_channels=256, latent_dimensions=1,
          spatial_downscale_ratio=1, aliases=("stable_audio3",)),
        S("ace_audio", scale_factor=1.0, latent_channels=8, latent_dimensions=2,
          aliases=("ace_step_audio",)),
        S("ace_audio_15", scale_factor=1.0, latent_channels=64, latent_dimensions=1,
          spatial_downscale_ratio=1, aliases=("ace_audio_1_5",)),
        S("yue2", scale_factor=1.0, latent_channels=64, latent_dimensions=1,
          spatial_downscale_ratio=1, aliases=("yue_2",)),
        S("minimax_music3", scale_factor=1.0, latent_channels=128, latent_dimensions=1,
          spatial_downscale_ratio=1, aliases=("minimax_music_3", "music3")),
    )


def _index(specs: tuple[LatentFormatSpec, ...]) -> tuple[dict[str, LatentFormatSpec], dict[str, str]]:
    """建主键表 + 别名表 ✓（⚠️ **重名/别名撞车一律报错** ✗ —— 静默让后一条盖掉前一条 = 表里有幽灵 ✓✗）。"""
    table: dict[str, LatentFormatSpec] = {}
    aliases: dict[str, str] = {}
    for spec in specs:
        if spec.name in table:
            raise LatentFormatError(f"口径 {spec.name!r} 定义了两次 ✗（表里有幽灵 ✓✗）")
        table[spec.name] = spec
        for alias in spec.aliases:
            key = alias.strip().lower()
            if not key:
                raise LatentFormatError(f"{spec.name!r} 的空别名 ✗")
            owner = table.get(key) or aliases.get(key)
            if owner is not None:
                raise LatentFormatError(
                    f"别名 {alias!r} 撞车 ✗：既要指 {spec.name!r} 又已属于 "
                    f"{getattr(owner, 'name', owner)!r} ✓ —— 别名有歧义就只能靠猜 ✓✗")
            aliases[key] = spec.name
    return table, aliases


LATENT_FORMATS, _ALIAS_TO_NAME = _index(_build_specs())


def list_latent_formats() -> list[str]:
    """全部**规范名** ✓（排序后 ✓ —— 稳定输出便于比对 ✓）。"""
    return sorted(LATENT_FORMATS)


def normalize_latent_format(name: str) -> str:
    """别名/大小写 → **规范名** ✓；认不出 ⇒ **报错** ✗（**不返回默认** ✗✗）。"""
    key = str(name or "").strip().lower()
    if not key:
        raise LatentFormatError("口径名是空的 ✗ —— 认不出就别接着算 ✓✗")
    if key in LATENT_FORMATS:
        return key
    if key in _ALIAS_TO_NAME:
        return _ALIAS_TO_NAME[key]
    sample = ", ".join(list_latent_formats()[:8])
    raise LatentFormatError(
        f"认不出潜空间口径 {name!r} ✗（大小写不敏感 ✓、别名也认 ✓）—— "
        f"⚠️ **不回退默认 4 通道** ✗✗（那等于把它当 SD1.5 用 ✓✗）。表里有：{sample} … ✓")


def get_latent_format(name: str | LatentFormatSpec) -> LatentFormatSpec:
    """取口径 ✓（传 :class:`LatentFormatSpec` 原样返回 ✓ —— 便于内部复用 ✓）。"""
    if isinstance(name, LatentFormatSpec):
        return name
    return LATENT_FORMATS[normalize_latent_format(name)]


def _need_torch() -> Any:
    """⚠️ 没装 torch ⇒ **报错** ✗（本仓不静默降级 ✗，同 ``image_ops`` ✓）。"""
    try:
        import torch  # noqa: PLC0415 —— 懒加载 ✓（查表不需要 torch ✓）
    except Exception as err:  # noqa: BLE001 —— 任何导入失败都算「装不上」✓
        raise LatentFormatError(
            f"值域换算要 torch ✗，但 import torch 失败：{type(err).__name__}: {err} ✓"
            f"（⚠️ **查表/列名不需要** torch ✓ —— 只有真换算才需要 ✓）") from err
    return torch


def _as_tensor(latent: Any) -> Any:
    torch = _need_torch()
    if isinstance(latent, torch.Tensor):
        return torch, latent
    try:
        return torch, torch.as_tensor(latent)
    except Exception as err:  # noqa: BLE001 —— 转不成就报 ✗
        raise LatentFormatError(
            f"潜变量既不是张量也转不成张量 ✗：{type(latent).__name__} ✓"
            f"（{type(err).__name__}: {err} ✓）") from err


def _mean_std(spec: LatentFormatSpec, mean: Any, std: Any, torch: Any) -> tuple[Any, Any]:
    """``PROCESS_MEANSTD`` 的 mean/std ✓ —— ⚠️ **必须由调用方给** ✗（本表不假装有 ✓）。"""
    if mean is None or std is None:
        raise LatentFormatError(
            f"{spec.name!r} 是**按均值/标准差归一**的口径 ✓ ⇒ 必须传 ``mean`` / ``std`` ✗"
            f"（它们是**具体权重**的事实 ✓，本表不存 ✓）—— ⚠️ 拿默认值凑只会得到看着对的错值 ✓✗")
    mean_t = mean if isinstance(mean, torch.Tensor) else torch.as_tensor(mean)
    std_t = std if isinstance(std, torch.Tensor) else torch.as_tensor(std)
    return mean_t, std_t


def _rearrange_guard(spec: LatentFormatSpec) -> None:
    raise LatentFormatError(
        f"{spec.name!r} 的换算里**夹着张量重排** ✗（permute / reshape / 拼帧 ✓）—— "
        f"本模块**不做** ✗✗：只乘一个数会给出「少了一半重排」的错值 ✓✗（比失败更贵 ✓）")


def process_in(name: str | LatentFormatSpec, latent: Any, *,
               mean: Any = None, std: Any = None) -> Any:
    """潜变量 → **模型空间** ✓（方向：``latent * scale`` ✓ / ``(latent − shift) * scale`` ✓）。

    ⚠️ ``PROCESS_IDENTITY`` 原样返回 ✓；``PROCESS_MEANSTD`` 要 ``mean``/``std`` ✓；
    ``PROCESS_REARRANGE`` **直接报错** ✗（本模块不做重排 ✓✗）。
    """
    spec = get_latent_format(name)
    if spec.process == PROCESS_REARRANGE:
        _rearrange_guard(spec)
    torch, tensor = _as_tensor(latent)
    if spec.process == PROCESS_IDENTITY:
        return tensor
    if spec.process == PROCESS_SCALE:
        return tensor * spec.scale_factor
    if spec.process == PROCESS_AFFINE:
        return (tensor - spec.shift_factor) * spec.scale_factor
    mean_t, std_t = _mean_std(spec, mean, std, torch)
    return (tensor - mean_t) * spec.scale_factor / std_t


def process_out(name: str | LatentFormatSpec, latent: Any, *,
                mean: Any = None, std: Any = None) -> Any:
    """模型空间 → **潜变量** ✓（:func:`process_in` 的**逆运算** ✓）。

    ⚠️ 逆运算含义：``scale`` ⇒ 除 ✓；``affine`` ⇒ **先除再加** shift ✓；
    ``meanstd`` ⇒ ``x * std / scale + mean`` ✓。
    """
    spec = get_latent_format(name)
    if spec.process == PROCESS_REARRANGE:
        _rearrange_guard(spec)
    torch, tensor = _as_tensor(latent)
    if spec.process == PROCESS_IDENTITY:
        return tensor
    if spec.process == PROCESS_SCALE:
        return tensor / spec.scale_factor
    if spec.process == PROCESS_AFFINE:
        return tensor / spec.scale_factor + spec.shift_factor
    mean_t, std_t = _mean_std(spec, mean, std, torch)
    return tensor * std_t / spec.scale_factor + mean_t

