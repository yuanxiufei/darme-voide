"""**Flux VAE**（本仓自己的实现 ✓）—— Flux / **Z-Image** 的潜变量 ⇄ 像素（**2 维** ✓）。

## 为什么需要它（2026-10-31）

Z-Image 的 DiT 能装、能前向 ✓（``dit_lumina`` ✓），但潜变量是 **16 通道** ✓；本仓此前只有
SDXL VAE（4 通道 ✓ 带 ``quant_conv`` ✓）和 H3 视频 VAE（3D ✓）⇒ **16 通道这一档解不出来** ✗。
参考实现里 Z-Image **直接继承 Lumina2** ✓，而 ``Lumina2.latent_format = latent_formats.Flux`` ✓
（``reference/comfy-org/ComfyUI/comfy/supported_models.py`` 的 ``class ZImage(Lumina2)`` ✓）
⇒ 它用的就是 **Flux 的 ae.safetensors** ✓，不是 ``qwen_image_vae`` ✗（那条属 Qwen-Image 家族 ✗）。

## 结构事实（从**本机真权重头**逐键量出 ✓ —— 不是猜的 ✗）

``vae/ae.safetensors`` 共 **244** 键 ✓，命名与 SDXL VAE（248 键 ✓）**同一套 LDM 老式** ✓：

* ``encoder.down.{0..3}.block.{0,1}`` ✓ / ``decoder.up.{0..3}.block.{0,1,2}`` ✓ —— 每层块数与 SDXL **完全一致** ✓；
* ``encoder.mid`` / ``decoder.mid`` 各 ``block_1 + attn_1 + block_2`` ✓ 同构 ✓；
* ``decoder.conv_in.weight`` = ``[512, 16, 3, 3]`` ✓（潜 **16** 通道 ✓）/ ``encoder.conv_out.weight`` = ``[32, 512, 3, 3]`` ✓；
* **唯一差异**：没有 ``quant_conv`` / ``post_quant_conv`` 这 4 个键 ✗ ⇒ 244 = 248 − 4 ✓。

⚠️ 所以本模块**不重抄编解码器** ✗：直接复用 ``sdxl_vae`` 那套模块 ✓，只把配置换成
``latent_channels=16`` + ``quant_conv=False`` ✓（见 ``SdxlVaeConfig.quant_conv`` 的注释 ✓）。
少抄一份 250 行 = 少一份会漂的结构 ✓。

## 缩放口径（**最容易漏、漏了不报错** ✗✗）

``latent_formats.Flux`` ✓：``scale_factor = 0.3611`` ✓、``shift_factor = 0.1159`` ✓。
解码前必须把**模型空间**换回 **VAE 空间** ✓：
``x_vae = x / 0.3611 + 0.1159`` ✓（＝ ``Flux.process_out`` ✓ 的逆过程 ✓，出处见
:func:`sdxl_vae.decode_latents` 的 docstring ✓）。
⚠️ **不是** ``(x − 0.1159) / 0.3611`` ✗✗ —— 那个式子在 SDXL 上恰好等价（``shift=0`` ✓）⇒
在 SDXL 上永远试不出来 ✓✗，换到 Flux 族就是常数偏移 ``≈0.437`` ✓✗：不报错、图偏色 ✓。

## 验证口径

``tests/engine_vae_flux_test.py``：① 键名表与本机真权重头**逐键比对** ✓（**244 键全等**才绿 ✓）；
② 缩小版同构前向（编解码往返、非方形 ✓）；③ 缩放系数与 ``latent_formats.Flux`` 常量一致 ✓。
"""
from __future__ import annotations

from typing import Any, Mapping

from . import sdxl_vae
from .sdxl_vae import SdxlVaeConfig, SdxlVaeError, has_torch

__all__ = [
    "FLUX_VAE_FACTS",
    "build_flux_vae",
    "decode_latents",
    "flux_vae_config",
    "flux_vae_key_names",
    "load_flux_vae_state_dict",
    "shape_problems",
]

#: 核过的**事实**（来源：本机 ``vae/ae.safetensors`` 头部 ✓ + ``comfy/latent_formats.py::Flux`` ✓）。
FLUX_VAE_FACTS: dict[str, Any] = {
    "checkpoint": "vae/ae.safetensors",
    "keyPrefix": "",             # ⚠️ 裸键 ✓（不像 SDXL 检查点带 ``first_stage_model.`` ✗）
    "keyCount": 244,
    "naming": "CompVis/LDM 老式，与 SDXL VAE 同构",
    "blockOutChannels": (128, 256, 512, 512),
    "encoderBlocksPerLevel": 2,
    "decoderBlocksPerLevel": 3,
    "latentChannels": 16,
    "scalingFactor": 0.3611,
    "shiftFactor": 0.1159,
    "spatialScale": 8,
    "note": "与 SDXL VAE 的唯一差异 = 无 quant_conv/post_quant_conv（244 = 248 − 4）✓",
}


def flux_vae_config(*, block_out_channels: tuple[int, ...] | None = None,
                    layers_per_block: int = 2) -> SdxlVaeConfig:
    """Flux VAE 配置 ✓（默认值即真权重口径 ✓；``block_out_channels`` / ``layers_per_block``
    只在测试里缩小用 ✓ —— 缩小版走**同一条前向** ✓）。"""
    return SdxlVaeConfig(
        in_channels=3, out_channels=3, latent_channels=int(FLUX_VAE_FACTS["latentChannels"]),
        block_out_channels=tuple(block_out_channels or FLUX_VAE_FACTS["blockOutChannels"]),
        layers_per_block=int(layers_per_block), norm_num_groups=32, norm_eps=1e-6,
        scaling_factor=float(FLUX_VAE_FACTS["scalingFactor"]),
        shift_factor=float(FLUX_VAE_FACTS["shiftFactor"]),
        quant_conv=False,
    )


def flux_vae_key_names(config: SdxlVaeConfig | None = None) -> tuple[str, ...]:
    """应当存在的**全部键名** ✓（无前缀 ✓）。拿它跟真权重头逐键比 ✓ = 预检 ✓。"""
    return sdxl_vae.sdxl_vae_key_names(config or flux_vae_config())


def build_flux_vae(config: SdxlVaeConfig | None = None, *,
                   device: Any = None, dtype: Any = None) -> Any:
    """建一个 Flux VAE 壳（**随机初始化** ⇒ 出图前必须 :func:`load_flux_vae_state_dict` ✓）。

    ⚠️ 没装 torch ⇒ **报错** ✗（不静默降级 ✗，同 ``build_sdxl_vae`` ✓）。
    """
    if not has_torch():
        raise SdxlVaeError("没装 torch ✗ ⇒ 建不了 Flux VAE ✓")
    spec = config or flux_vae_config()
    return sdxl_vae.build_sdxl_vae(spec, device=device, dtype=dtype)


def shape_problems(shapes: Mapping[str, Any]) -> list[str]:
    """从**权重头**（键 → 形状）核配置是否对得上 ✓ —— 装载前先跑，比装完再炸清楚 ✓。

    只查**口径性质**的四处 ✓（通道 / 潜通道 / 卷积核 / 键数 ✓），不查每个 ResBlock ✗
    （那是 :func:`flux_vae_key_names` 逐键比对的活 ✓）。
    """
    problems: list[str] = []
    keys = set(shapes)
    expected = set(flux_vae_key_names())
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing:
        problems.append(f"缺 {len(missing)} 键（例：{missing[:3]}）✗")
    if extra:
        problems.append(f"多 {len(extra)} 键（例：{extra[:3]}）✗")
    latent = int(FLUX_VAE_FACTS["latentChannels"])
    decoder_in = shapes.get("decoder.conv_in.weight")
    if decoder_in is not None and int(tuple(decoder_in)[1]) != latent:
        problems.append(
            f"decoder.conv_in 潜通道 {int(tuple(decoder_in)[1])} ≠ {latent} ✗"
            f"（16 是 Flux/Z-Image ✓；4 说明拿成了 SDXL VAE ✗）")
    conv_out = shapes.get("encoder.conv_out.weight")
    if conv_out is not None and int(tuple(conv_out)[0]) != 2 * latent:
        problems.append(f"encoder.conv_out 通道 {int(tuple(conv_out)[0])} ≠ {2 * latent} ✗")
    return problems


def load_flux_vae_state_dict(model: Any, state_dict: "Mapping[str, Any]", *,
                             strict: bool = True) -> int:
    """把 ``ae.safetensors`` 的裸键装进 ``model`` ✓，返回装载键数 ✓。

    ⚠️ 与 ``load_sdxl_vae_state_dict`` 不同 ✗：这里**没有前缀** ✓ ⇒ 不去剥 ✗，而是先拿
    :func:`flux_vae_key_names` 做**键集对账** ✓，多了少了都**报错**（带差集，不是一句 "size mismatch" ✓）。
    ⚠️ 不许 ``strict=False`` 糊过去 ✗ —— 那会留下**随机初始化的层** ✓✗。
    """
    if not has_torch():
        raise SdxlVaeError("没装 torch ✗ ⇒ 装不了权重 ✓")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise SdxlVaeError("state_dict 必须是非空映射 ✗")
    expected = set(flux_vae_key_names(getattr(model, "config", None)))
    got = set(state_dict)
    missing, extra = sorted(expected - got), sorted(got - expected)
    if missing or extra:
        raise SdxlVaeError(
            f"键集与 Flux VAE 架构**不符** ✗：缺 {len(missing)}（例 {missing[:3]}）✓ / "
            f"多 {len(extra)}（例 {extra[:3]}）✓ ⇒ 检查是不是拿成了 SDXL VAE（带 quant_conv ✓）")
    try:
        model.load_state_dict(dict(state_dict), strict=bool(strict))
    except RuntimeError as error:
        raise SdxlVaeError(f"Flux VAE 张量与架构**不符** ✗：{error} ✓") from error
    return len(got)


def decode_latents(model: Any, latents: Any, *, config: SdxlVaeConfig | None = None,
                   to_unit: bool = True, tile: int | None = None, overlap: int = 64,
                   progress: Any = None) -> Any:
    """潜变量 ⇒ 像素 ✓ —— 先 ``latents / 0.3611 + 0.1159`` ✓（``Flux.process_out`` ✓）再解码 ✓。

    ⚠️ 千万别直接调 ``model.decode`` ✗（漏缩放**不报错** ✓，图会发灰 ✓✗，见模块头 ✓）。

    ``tile`` / ``overlap`` 是**像素**单位的分块解码 ✓（默认关 ✓，见 :func:`sdxl_vae.decode_latents` ✓）。
    """
    spec = config or getattr(model, "config", None) or flux_vae_config()
    return sdxl_vae.decode_latents(model, latents, config=spec, to_unit=to_unit,
                                   tile=tile, overlap=overlap, progress=progress)
