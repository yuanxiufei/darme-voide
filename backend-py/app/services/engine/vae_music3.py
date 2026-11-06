"""**MiniMax Music3 DAV**（Deep Audio VAE，本仓自己的实现 ✓）—— 音频潜变量 ⇄ **44.1 kHz 立体声** ✓。

## 为什么单独一份，不并进 ``audio_vae``（H3 那条）

两条音频 VAE 是**两个族** ✓，数字全不同 ✓（并进来就会「装得上、解得出、只是声音全错」✓✗）：

===========  ==================  =========  ============  ==================
              H3 ``audio_vae``    本模块      出处         影响
===========  ==================  =========  ============  ==================
采样率        32000 Hz ✓          44100 Hz ✓  参考 ``sd.py``  ``audio_sample_rate`` ✓
hop           800 样本/潜帧 ✓     512 ✓        ``upscale_ratio=512`` ✓
潜帧率        40 Hz ✓             86.1328 Hz ✓ 44100/512 ✓   时长整体 ×2.15 ✓✗
潜通道        32 ✓（+ 立体声 2）   **128** ✓   ``latent_channels=128`` ✓
声道折叠      先卷积再分声道 ✓     **通道维折半** ✓  ``reshape(b*2, 64, T)`` ✓
===========  ==================  =========  ============  ==================

⚠️ **128 = 2 声道 × 64** ✓ —— ``decode`` 里 ``latent.reshape(batch*2, 64, frames)`` ✓ 是**行主序折半** ✓：
通道 ``0..63`` → 左声道 ✓、``64..127`` → 右声道 ✓、``reshape(batch, 2, -1)`` 折回来 ✓。
把 128 当"128 个特征通道"直接卷积 ⇒ **形状全对、声音是两声道混在一起的糊** ✓✗。

## 结构事实（2026-11-02 从本机真权重头逐键量出 ✓ —— 不是猜的 ✗）

``vae/minimax_music3_dav.safetensors`` 共 **121** 键 ✓：

* ``dec_in_proj`` = ``Conv1d(64 → 1024, k=1)`` ✓（**64** ✓ 就是折半后的声道数 ✓）；
* ``decoder.model.0`` = ``Conv1d(1024 → 1536, k=7, p=3)`` ✓；
* ``decoder.model.{1..4}`` = 4 个 :class:`_DecoderBlock` ✓，通道 ``1536→768→384→192→96`` ✓
  （逐级 ``//2`` ✓），**步长 ``(8, 8, 4, 2)`` ✓ ⇒ 乘积 = 512** ✓；
* 每块里 3 个 :class:`_ResidualUnit` ✓（扩张 ``1 / 3 / 9`` ✓，``dilation=d`` 且 ``padding=3d`` ✓
  ⇒ 卷积**不改长度** ✓，参考里那段裁切是死代码但**照抄保留** ✓）；
* ``decoder.model.5`` = 收尾 Snake ✓、``model.6`` = ``Conv1d(96 → 1, k=7, p=3)`` ✓、``Tanh`` ✓
  ⇒ 输出逐声道落在 ``[-1, 1]`` ✓。

### 权重名是**旧式** ``weight_g`` / ``weight_v`` ✓（照抄会「装不上」的坑）

参考用 ``nn.utils.parametrizations.weight_norm`` ✓，而**新版** parametrize 的参数名是
``parametrizations.weight.original0/1`` ✓，检查点里却是 ``weight_g`` / ``weight_v`` ✓。
两者**并不冲突** ✓：torch 的 ``ParametrizationList._load_from_state_dict`` 认旧名 ✓
（本机 torch 2.14 实测 ``strict=True`` 下 0 missing / 0 unexpected ✓，且
``weight == v * g / ‖v‖`` ✓ 逐位成立 ✓）。⇒ 本模块**不自己实现 weight_norm** ✗、
也不在手装前把键改名 ✗，只把两套写法**归一到一处再对账** ✓（:func:`_canonical_key` ✓）。

## 缩放口径

``latent_formats.minimax_music3`` ✓：``scale_factor = 1.0`` ✓（**恒等** ✓，没有 Flux 那种
``0.3611 / 0.1159`` ✓✗）。所以 ``decode`` 前**不做任何缩放** ✓ —— 但也不许"顺手乘一个 0.18215" ✗
（那是 SD1.5 的 ✓）。

## 验证口径

``tests/engine_music3_test.py``：键名表与真权重头**逐键比对**（121 键全等 ✓）、
``T`` 帧 → ``512·T`` 样本 ✓、``T → 44.1 kHz 秒数`` 与 ``dit_music3.latent_length`` 互逆 ✓、
折半顺序（左声道取通道 ``0..63`` ✓）、缩小版同构往返 ✓。
"""
from __future__ import annotations

import functools
import math
from typing import Any, Mapping

from .latent_formats import get_latent_format

__all__ = [
    "KEY_MAP",
    "MUSIC3_DAV_FACTS",
    "Music3Dav",
    "as_legacy_name",
    "build_music3_dav",
    "dav_key_names",
    "decode_latents",
    "has_torch",
    "latent_frames_for_seconds",
    "load_music3_dav_state_dict",
    "samples_for_frames",
    "seconds_for_frames",
    "shape_problems",
]

#: 检查点（**旧式**）``…weight_g`` / ``…weight_v`` ⇒ ``state_dict``（**新式**）略写 ✓。
#: ⚠️ 这是**两个方向共用的唯一真源** ✗✗：``_canonical_key``（旧 → 新 ✓）与
#: ``weights.load_module_weights(key_map=…)``（流式装载时逐键改名 ✓）都读它 ✓ ——
#: 各写一份必然漂 ✓（2026-11-06 首次真跑就栽在这里：`shape_problems` 归一对账全绿 ✓、
#: 转头 `attach_music3_dav` 走通用装载器**按原键硬对** ⇒ 报「缺 60 键」✓✗）。
KEY_MAP: dict[str, str] = {
    r"\.weight_g$": ".parametrizations.weight.original0",
    r"\.weight_v$": ".parametrizations.weight.original1",
}

#: 潜空间口径 ✓（通道数 / 时间下采样比**只从** :mod:`latent_formats` 读 ✓ —— 不在这里再写一遍 ✗）。
_SPEC = get_latent_format("minimax_music3")

#: 核过的**事实**（来源：本机 ``minimax_music3_dav.safetensors`` 头部 ✓ +
#: 参考 ``comfy/ldm/minimax_music/dav.py`` ✓ + 参考 ``comfy/sd.py`` 的 DAV 分支 ✓）。
MUSIC3_DAV_FACTS: dict[str, Any] = {
    "checkpoint": "vae/minimax_music3_dav.safetensors",
    "keyCount": 121,
    "latentChannels": int(_SPEC.latent_channels),        # 128 ✓
    "foldChannels": int(_SPEC.latent_channels) // 2,    # 64 ✓（折半后逐声道的特征数）
    "stereo": 2,
    "decoderInChannels": 1024,
    "channelLadder": (1536, 768, 384, 192, 96),
    "strides": (8, 8, 4, 2),
    "residualDilations": (1, 3, 9),
    "upscaleRatio": int(_SPEC.temporal_downscale_ratio),  # 512 ✓
    "sampleRate": 44100,
    "latentHz": 44100.0 / float(_SPEC.temporal_downscale_ratio),
    "scaleFactor": float(_SPEC.scale_factor),             # 1.0 ✓ 恒等
    "weightNormKeys": ("weight_g", "weight_v"),
    "outputRange": (-1.0, 1.0),
    "note": "128 通道 = 2 声道 × 64；reshape(batch*2, 64, T) 是行主序折半 ✓",
}


def has_torch() -> bool:
    """本仓的「torch 在不在」唯一判据 ✓（与其它 VAE 模块同一口径 ✓）。"""
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - 只在缺依赖的环境里走
        return False
    return True


def _torch():
    import torch

    return torch


def _nn():
    import torch.nn as nn

    return nn


# --------------------------------------------------------------------------- #
# 结构（逐字照抄参考 dav.py ✓）
# --------------------------------------------------------------------------- #
def _snake(x: Any, alpha: Any) -> Any:
    """``x + sin(αx)² / (α + 1e-9)`` ✓（⚠️ **1e-9 不是装饰** ✗：α=0 会让分母归零 ✓）。"""
    torch = _torch()
    shape = x.shape
    flat = x.reshape(shape[0], shape[1], -1)
    alpha = alpha.to(flat.dtype)
    flat = flat + (alpha + 1e-9).reciprocal() * torch.sin(alpha * flat).pow(2)
    return flat.reshape(shape)


@functools.lru_cache(maxsize=1)
def _make_modules():
    """按需造 ``nn.Module`` 子类 ✓（模块导入时不 require torch ✓ —— 与全仓同一口径 ✓）。

    ⚠️ 只造**一次** ✓（``lru_cache``）：每次调都新建一批类 ⇒ ``isinstance`` 与
    ``state_dict`` 键集虽然一致 ✓，但 ``pickle`` / 类型比较会静默失配 ✓✗。
    """
    nn = _nn()

    class Snake1d(nn.Module):
        def __init__(self, channels: int, dtype: Any = None, device: Any = None) -> None:
            super().__init__()
            #: ⚠️ 必须给**确定初值** ✓✗：``torch.empty`` 是未初始化内存 ✓，本机实测抽到过
            #: ``alpha ≈ 3.5e7`` ✓ —— 随机壳（:func:`build_music3_dav` 不装权重时）会解出
            #: 满是高频抖动的「声音」✓✗，看着像模型在跑、其实全是垃圾初值 ✓。
            #: 真权重装载时这一格会被覆盖 ✓（``decoder.model.*.block.*.alpha`` ✓），
            #: 与 :mod:`audio_vae` 的 ``Snake1d`` 同一初值口径 ✓。
            self.alpha = nn.Parameter(_torch().ones(1, channels, 1, dtype=dtype, device=device))

        def forward(self, x: Any) -> Any:
            return _snake(x, self.alpha)

    def _wn_conv(dim_in: int, dim_out: int, *, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, dtype: Any = None,
                 device: Any = None) -> Any:
        conv = nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, dtype=dtype, device=device)
        return nn.utils.parametrizations.weight_norm(conv)

    def _wn_conv_transpose(dim_in: int, dim_out: int, *, kernel_size: int, stride: int,
                           padding: int, dtype: Any = None, device: Any = None) -> Any:
        conv = nn.ConvTranspose1d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dtype=dtype, device=device)
        return nn.utils.parametrizations.weight_norm(conv)

    class ResidualUnit(nn.Module):
        def __init__(self, dim: int, dilation: int, dtype: Any = None,
                     device: Any = None) -> None:
            super().__init__()
            self.block = nn.Sequential(
                Snake1d(dim, dtype, device),
                _wn_conv(dim, dim, kernel_size=7, dilation=dilation, padding=3 * dilation,
                         dtype=dtype, device=device),
                Snake1d(dim, dtype, device),
                _wn_conv(dim, dim, kernel_size=1, dtype=dtype, device=device),
            )

        def forward(self, x: Any) -> Any:
            residual = self.block(x)
            if residual.shape[-1] != x.shape[-1]:
                padding = (x.shape[-1] - residual.shape[-1]) // 2
                x = x[..., padding:x.shape[-1] - padding]
            return x + residual

    class DecoderBlock(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, stride: int,
                     dtype: Any = None, device: Any = None) -> None:
            super().__init__()
            self.block = nn.Sequential(
                Snake1d(input_dim, dtype, device),
                _wn_conv_transpose(input_dim, output_dim, kernel_size=2 * stride,
                                   stride=stride, padding=math.ceil(stride / 2),
                                   dtype=dtype, device=device),
                ResidualUnit(output_dim, 1, dtype, device),
                ResidualUnit(output_dim, 3, dtype, device),
                ResidualUnit(output_dim, 9, dtype, device),
            )

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class Decoder(nn.Module):
        def __init__(self, dtype: Any = None, device: Any = None) -> None:
            super().__init__()
            facts = MUSIC3_DAV_FACTS
            channels = int(facts["channelLadder"][0])
            layers: list[Any] = [
                _wn_conv(int(facts["decoderInChannels"]), channels, kernel_size=7, padding=3,
                         dtype=dtype, device=device),
            ]
            output_dim = channels
            for index, stride in enumerate(tuple(facts["strides"])):
                input_dim = channels // (2 ** index)
                output_dim = channels // (2 ** (index + 1))
                layers.append(DecoderBlock(input_dim, output_dim, int(stride), dtype, device))
            layers.extend((
                Snake1d(output_dim, dtype, device),
                _wn_conv(output_dim, 1, kernel_size=7, padding=3, dtype=dtype, device=device),
                nn.Tanh(),
            ))
            self.model = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.model(x)

    class Music3Dav(nn.Module):
        """潜变量 ``(B, 128, T)`` ⇒ 波形 ``(B, 2, 512·T)`` ✓。"""

        def __init__(self, dtype: Any = None, device: Any = None) -> None:
            super().__init__()
            facts = MUSIC3_DAV_FACTS
            self.dec_in_proj = nn.Conv1d(int(facts["foldChannels"]),
                                         int(facts["decoderInChannels"]),
                                         kernel_size=1, dtype=dtype, device=device)
            self.decoder = Decoder(dtype, device)

        def decode(self, latent: Any) -> Any:
            #: ⚠️⚠️ **入口先把潜变量对齐到权重精度** ✗✗（2026-11-06 首次真跑栽的第三处 ✓）：
            #: 采样侧**有意**保 f32 ✓，DAV 权重在卡上是 bf16 ✓ ⇒ 不对齐当场
            #: ``Input type (float) and bias type (struct c10::BFloat16) should be the same`` ✓，
            #: 报错指向 ``conv`` ✓、读起来像"权重坏了" ✓，其实是**精度没对齐** ✓。
            #: 对齐**只能在这里**（模型内部）✗：把潜变量降成 bf16 再回传会毁掉逆过程的数值 ✓✗
            #: （与 `vae_wan.WanVae.decode` / `sdxl_vae.decode_latents` 同一条教训 ✓）。
            target = self.dec_in_proj.weight.dtype
            if latent.dtype != target:
                latent = latent.to(target)
            batch, channels, frames = latent.shape
            want = int(MUSIC3_DAV_FACTS["latentChannels"])
            if int(channels) != want:
                raise ValueError(
                    f"DAV 潜通道 {int(channels)} ≠ {want} ✗ —— 128 = 2 声道 × 64 ✓"
                    f"（4 / 16 / 32 说明拿成了别的族的 VAE ✗）")
            folded = latent.reshape(batch * 2, want // 2, frames)
            waveform = self.decoder(self.dec_in_proj(folded))
            return waveform.reshape(batch, 2, -1)

        forward = decode

    return Music3Dav


def build_music3_dav(*, dtype: Any = None, device: Any = None) -> Any:
    """建一个 DAV 壳（**随机初始化** ⇒ 出流前必须 :func:`load_music3_dav_state_dict` ✓）。"""
    if not has_torch():
        raise RuntimeError("没装 torch ✗ ⇒ 建不了 Music3 DAV ✓")
    return _make_modules()(dtype=dtype, device=device)


@functools.lru_cache(maxsize=1)
def _compiled_key_map() -> tuple[tuple[Any, str], ...]:
    """``KEY_MAP`` 预编译一次 ✓（与 ``weights._compile_key_map`` 同一形 ✓ 但**读同一张表** ✓）。"""
    import re  # noqa: PLC0415 —— 只在真对账路径才需要 ✓（模块导入零依赖 ✓）

    return tuple((re.compile(pattern), target) for pattern, target in KEY_MAP.items())


def _canonical_key(name: str) -> str:
    """**旧式** ``weight_g`` / ``weight_v`` ⇒ **新式** parametrize 名 ✓（:func:`as_legacy_name` 的逆向 ✓）。

    ⚠️ 两套写法在本机 torch 上都能 ``load_state_dict`` 成功 ✓，但**键集对账**不做归一
    ⇒ 121 键会被判成「缺 80 / 多 80」✓✗（全是 weight_norm 那批 ✓）。
    ⚠️ 规则表**只在** :data:`KEY_MAP` 一处 ✓（它同时喂 ``weights.load_module_weights`` 的
    ``key_map`` ✓）—— 这里再手写一遍 ``endswith`` 就是第二份真源 ✓✗。
    """
    for pattern, target in _compiled_key_map():
        if pattern.search(name):
            return pattern.sub(target, name)
    return name


def as_legacy_name(name: str) -> str:
    """**新式** parametrize 名 ⇒ **旧式** ``weight_g`` / ``weight_v`` ✓（检查点那一套 ✓）。

    ⚠️ 两个方向都要有 ✓✗：:func:`shape_problems` 的基准是**检查点**名字 ✓，而调用方手里
    常常直接是 ``model.state_dict()`` ✓（新名字 ✓）—— 少了这一向，「自证零问题」会变成
    「缺 60 / 多 60」✓✗（实测正是这样栽的 ✓）。
    """
    return (name.replace(".parametrizations.weight.original0", ".weight_g")
            .replace(".parametrizations.weight.original1", ".weight_v"))


def dav_key_names(*, legacy: bool = True) -> tuple[str, ...]:
    """应当存在的**全部键名** ✓。

    ``legacy=True`` ⇒ 返回检查点那一套（``weight_g`` / ``weight_v`` ✓，用于跟真权重头比对 ✓）；
    ``legacy=False`` ⇒ 返回 ``nn.Module.state_dict()`` 那一套 ✓（用于装载前的另一半对账 ✓）。
    """
    return _key_names(legacy=bool(legacy))


@functools.lru_cache(maxsize=2)
def _key_names(*, legacy: bool = True) -> tuple[str, ...]:
    module = build_music3_dav(dtype=None, device="meta")
    names = list(module.state_dict().keys())
    if legacy:
        names = [as_legacy_name(n) for n in names]
    return tuple(sorted(names))


def shape_problems(shapes: Mapping[str, Any]) -> list[str]:
    """从**权重头**（键 → 形状）核口径 ✓ —— 装载前先跑，比装完再炸清楚 ✓。

    ``shapes`` 的键**两种写法都收** ✓（检查点 ``weight_g`` ✓ / ``state_dict`` 的
    ``parametrizations.weight.original0`` ✓）—— 先归一到检查点那一套再对账 ✓。
    """
    problems: list[str] = []
    keys = {as_legacy_name(str(k)) for k in shapes}
    expected = set(dav_key_names(legacy=True))
    missing = [k for k in expected if k not in keys]
    extra = sorted(keys - expected)
    if missing:
        problems.append(f"缺 {len(missing)} 键（例：{missing[:3]}）✗")
    if extra:
        problems.append(f"多 {len(extra)} 键（例：{extra[:3]}）✗")
    wanted = int(MUSIC3_DAV_FACTS["foldChannels"])
    proj = shapes.get("dec_in_proj.weight")
    if proj is not None and int(tuple(proj)[1]) != wanted:
        problems.append(
            f"dec_in_proj 输入通道 {int(tuple(proj)[1])} ≠ {wanted} ✗"
            f"（{MUSIC3_DAV_FACTS['latentChannels']} 潜通道要折半成 2 × {wanted} ✓）")
    total = 1
    for key in (f"decoder.model.{i}.block.1.weight_v" for i in range(1, 5)):
        shape = shapes.get(key)
        if shape is None:
            continue
        # ConvTranspose1d 的权重是 (in, out, k) 且 kernel = 2·stride ⇒ stride = k // 2
        total *= max(1, int(tuple(shape)[2]) // 2)
    if total != 1 and total != int(MUSIC3_DAV_FACTS["upscaleRatio"]):
        problems.append(
            f"转置卷积步长乘积 {total} ≠ {MUSIC3_DAV_FACTS['upscaleRatio']} ✗"
            f"（512 定死 ⇒ 潜帧率 44100/512 = {MUSIC3_DAV_FACTS['latentHz']:.4f} Hz ✓）")
    return problems


def load_music3_dav_state_dict(model: Any, state_dict: Mapping[str, Any]) -> int:
    """把 DAV 检查点装进 ``model`` ✓，返回装载键数 ✓。

    ⚠️ 不许 ``strict=False`` 糊过去 ✗ —— 那会留下**随机初始化**的 Snake 层 ✓✗
    （症状：声音出来是噪声包着一段像样的轮廓 ✓，很难看出是权重没装 ✓）。
    """
    if not has_torch():
        raise RuntimeError("没装 torch ✗ ⇒ 装不了权重 ✓")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError("state_dict 必须是非空映射 ✗")
    want = set(model.state_dict())
    got = {_canonical_key(str(k)) for k in state_dict}
    missing, extra = sorted(want - got), sorted(got - want)
    if missing or extra:
        raise RuntimeError(
            f"键集与 Music3 DAV 架构**不符** ✗：缺 {len(missing)}（例 {missing[:3]}）✓ / "
            f"多 {len(extra)}（例 {extra[:3]}）✓ ⇒ 检查是不是拿成了 H3 的 audio_vae（32 通道 ✓）")
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise RuntimeError(f"Music3 DAV 张量与架构**不符** ✗：{error} ✓") from error
    return len(got)


# --------------------------------------------------------------------------- #
# 时长换算（**唯一真源**在 dit_music3.latent_length 的同一套常数 ✓）
# --------------------------------------------------------------------------- #
def samples_for_frames(latent_frames: int) -> int:
    """潜帧数 ⇒ 波形样本数 ✓（``512 · T`` ✓，逐级转置卷积**恰好**整除 ✓ 不是四舍五入 ✓）。"""
    if latent_frames < 0:
        raise ValueError(f"潜帧数不能为负 ✗（收到 {latent_frames} ✓）")
    return int(latent_frames) * int(MUSIC3_DAV_FACTS["upscaleRatio"])


def seconds_for_frames(latent_frames: int) -> float:
    """潜帧数 ⇒ 秒 ✓。"""
    return samples_for_frames(latent_frames) / float(MUSIC3_DAV_FACTS["sampleRate"])


def latent_frames_for_seconds(seconds: float) -> int:
    """秒 ⇒ 潜帧数（**向下取整** ✓ —— 向上齐会让音频比请求长一个潜帧 ✓✗）。"""
    if seconds < 0:
        raise ValueError(f"秒数不能为负 ✗（收到 {seconds} ✓）")
    return int(seconds * float(MUSIC3_DAV_FACTS["latentHz"]))


# --------------------------------------------------------------------------- #
# 解码入口
# --------------------------------------------------------------------------- #
def decode_latents(model: Any, latents: Any, *, clamp: bool = True) -> Any:
    """潜变量 ⇒ 波形 ✓，返回 ``(B, 2, 512·T)`` ✓、值域 ``[-1, 1]`` ✓。

    ⚠️ **不做任何缩放** ✓（本族 ``scale_factor = 1.0`` ✓，见 :data:`MUSIC3_DAV_FACTS` ✓）；
    但会**核对** ``latent_formats`` 里仍是 1.0 ✗ —— 哪天有人把表改成 0.3611 那种，
    这里宁可当场报 ✗，也不要静默解出一段音量偏了 3 倍的声音 ✓✗。

    ``clamp``：模块尾部已有 ``Tanh`` ✓ ⇒ 默认再夹一次只是防线 ✓（不会改数值 ✓）。
    """
    if not has_torch():
        raise RuntimeError("没装 torch ✗ ⇒ 解不了音频 ✓")
    scale = float(get_latent_format("minimax_music3").scale_factor)
    if abs(scale - 1.0) > 1e-9:
        raise RuntimeError(
            f"minimax_music3 的 scale_factor 变成 {scale} ✗ —— 本模块按**恒等**接线 ✓，"
            f"先把缩放接上再解 ✓（否则音量整体偏 ✓✗、不报错 ✓）")
    waveform = model.decode(latents)
    if clamp:
        waveform = waveform.clamp(-1.0, 1.0)
    return waveform
