"""S8 自检：**SDXL VAE**（自研的「图」解码器 ✓ 2026-09-26）。

判据四类，都是**可判真假的事实** ✓（本仓纪律：不把没验的说成验过 ✗）：

* **口径类**：解码**必须先除缩放系数** `1/0.13025` ✗✗ —— 用**恒等解码桩**把算术钉死 ✓；
  `spatial_scale = 2^(层数−1)` ✓（**不是** `2^层数` ✗）、编码器 2 块 / 解码器 **3** 块 ✓；
* **结构类**：缩小版走**同一条编解码前向** ✓（含 non-square `8×12` ✓）；
  新模型 `mid.attn_1` 因 `proj_out` 零初始化 ⇒ **恒等** ✗✗（最硬的一条 ✓）；
* **键名类**：:func:`sdxl_vae_key_names` 对默认配置必须给 **248** 键 ✓（**纯结构推** ✓ ⇒ 不靠真权重 ✓）；
* **真权重类**：拿**本机真权重头**（只读元数据 ✗ 不载张量 ✓）把 **248 键的名字 + 形状**逐个对齐 ✓。

⚠️ 检查点是**扫描器找出来的** ✓（本文件**没有机器路径** ✗）；没装 torch / 扫不到 ⇒ 显式 SKIP ✓
（**没跑 ≠ 绿** ✗、不是通过 ✗）。⚠️ `ComfyUI` 是 **GPL-3.0** ⇒ **一行都不抄** ✗（只当规格书读 ✓）。
⚠️ 不做**数值**对拍 ✗ ⇒ 本套验**结构与口径** ✓，结论按实测说 ✓ 不夸大 ✗。

运行::

    ./.venv/Scripts/python.exe tests/engine_sdxl_vae_test.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import sdxl_vae as mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: Any, needle: str = "") -> str | None:
    try:
        call()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        return text if needle in text else None
    return None


#: SDXL **主权重**的文件名口径 ✓：先**扫描**出来 ✓ 再按名字筛 ✓（⚠️ 不写死机器路径 ✗）。
SDXL_BASE_NAME = re.compile(r"sd_?xl[_-]?base", re.IGNORECASE)
#: 真权重实测：`first_stage_model.*` 的键数 ✓（只读头部量出来的 ✓）
REAL_KEYS = 248

#: 缩小版：**同一条编解码前向** ✓、CPU 秒级 ✓（通道满足 `norm_num_groups` 整除 ✓）。
#: 两层 ⇒ `spatial_scale = 2^(2-1) = 2` ✓（**故意不是 8** ✓ ⇒ 顺手验「不是写死 8」✗）。
TOY = dict(
    in_channels=3, out_channels=3, latent_channels=4,
    block_out_channels=(8, 16), layers_per_block=1, norm_num_groups=4,
)


def find_sdxl_base_checkpoint() -> tuple[Path | None, str]:
    """找本机那份 SDXL base 主权重 —— **走本仓扫描器** ✓，不写死路径 ✗（同 ``engine_sdxl_test`` ✓）。"""
    try:
        from app.services import local_model_scan as scan  # noqa: PLC0415
    except Exception as error:  # noqa: BLE001
        return None, f"导不进本仓扫描器（{error}）⇒ 真权重对齐不跑 ✓"
    roots = list(scan.get_default_roots())
    try:
        scanned = scan.scan_local_models({"roots": roots, "kinds": ["image"], "maxFiles": 20000})
    except Exception as error:  # noqa: BLE001
        return None, f"扫描器报错（{error}）⇒ 真权重对齐不跑 ✓"
    hits = sorted(
        (item for item in scanned["models"]
         if item.get("role") == "standalone"
         and SDXL_BASE_NAME.search(str(item.get("filename") or ""))),
        key=lambda item: str(item.get("path") or ""),
    )
    if not hits:
        cut = ("（⚠️ 扫描被 `maxFiles` 截断了 ⇒「扫不到」也可能只是**没扫完** ✓ 不许当通过 ✗）"
               if scanned["truncated"] else "")
        return None, (f"扫描器扫了 {len(roots)} 个根 / {scanned['total']} 个图类权重，"
                      f"没有 `sd_xl_base*` ⇒ 真权重对齐不跑 ✓{cut}")
    return Path(str(hits[0]["path"])), f"扫描器命中 {len(hits)} 份"


def case_config() -> None:
    """① 配置口径 ✓：默认即官方 SDXL base ✓、派生量**算出来** ✓、坏配置**响亮报错** ✗。"""
    default = mod.SdxlVaeConfig()
    check("① 默认 = 官方 SDXL base ✓（通道 (128,256,512,512) ✓ / 4 潜通道 ✓ / 32 组 ✓ / 0.13025 ✓）",
          (default.block_out_channels, default.latent_channels, default.norm_num_groups,
           default.scaling_factor, default.layers_per_block)
          == ((128, 256, 512, 512), 4, 32, 0.13025, 2), default.to_dict())
    check("①′ 派生量：`num_levels=4` ✓、`spatial_scale` = **2^(4−1) = 8** ✓"
          "（⚠️ **不是** 2^4 ✗ —— 最后一层不下采样 ✓ 真权重 `down.3` 没有 `downsample` 键 ✓）、"
          "编码器 **2** 块 / 解码器 **3** 块 ✓（LDM 的 `+1` ✓）、`latent_channel_count=8` ✓",
          default.num_levels == 4 and default.spatial_scale == 8
          and default.encoder_blocks == 2 and default.decoder_blocks == 3
          and default.latent_channel_count == 8)
    check("①″ 缩小版 `spatial_scale=2` ✓ ⇒ 是**算出来**的（写死 8 会红 ✓）",
          mod.SdxlVaeConfig(**TOY).spatial_scale == 2)
    check("①‴ 事实表与配置**同源** ✓（248 键 / 0.13025 / 8 倍 ✓）",
          mod.SDXL_VAE_FACTS["keyCount"] == REAL_KEYS
          and mod.SDXL_VAE_FACTS["scalingFactor"] == default.scaling_factor
          and mod.SDXL_VAE_FACTS["spatialScale"] == default.spatial_scale
          and mod.SDXL_VAE_KEY_PREFIX == "first_stage_model.")
    check("①⁗ ⚠️ **只有 `block_out_channels` 受 GroupNorm 整除约束** ✗✗：`in/out_channels = 3` 是"
          "**卷积**通道、**不过 GroupNorm** ✓ ⇒ **官方配置必须建得起来** ✓（把 3 也拉进来查 ⇒ 官方配置都炸 ✓✗）",
          mod.SdxlVaeConfig() is not None and mod.SdxlVaeConfig(**TOY) is not None)
    check("①⁵ 坏配置**当场点名** ✗：通道表空 / 非整除 / eps≤0 / 缩放为 0 ✓",
          _raises(lambda: mod.SdxlVaeConfig(block_out_channels=()), "不能为空") is not None
          and _raises(lambda: mod.SdxlVaeConfig(block_out_channels=(7,)), "整除") is not None
          and _raises(lambda: mod.SdxlVaeConfig(norm_eps=0.0), "norm_eps") is not None
          and _raises(lambda: mod.SdxlVaeConfig(scaling_factor=0.0), "scaling_factor") is not None)


def case_key_names() -> None:
    """② 键名表 ✓：**纯结构推**出来的 ✓ ⇒ 不依赖真权重也能判 ✓。"""
    default = mod.sdxl_vae_key_names()
    check(f"② 默认配置 **{len(default)} 键** == 真权重 **{REAL_KEYS}** ✓"
          f"（⚠️ 块数 / 通道序 / 解码器键序错一处 ⇒ 这条必红 ✓）",
          len(default) == REAL_KEYS, len(default))
    check("②′ 排序后返回 ✓（稳定输出才便于比对 ✓）、无重复 ✓",
          list(default) == sorted(default) and len(set(default)) == len(default))
    check("②″ ⚠️ **两个「照抄必错」的点** ✗✗：① 编码器**最后一层没有** `downsample` ✓ 而 `down.0` 有 ✓；"
          "② 解码器 `up.0` **没有** `upsample` ✓ 而 `up.3` 有 ✓（键序与前向序相反 ✓）",
          "encoder.down.3.downsample.conv.weight" not in default
          and "encoder.down.0.downsample.conv.weight" in default
          and "decoder.up.0.upsample.conv.weight" not in default
          and "decoder.up.3.upsample.conv.weight" in default)
    check("②‴ ⚠️ **解码器每层 3 块** ✗（`up.0.block.2` 在 ✓）；编码器那侧只有 2 块 ✓（`down.0.block.2` 不在 ✓）",
          "decoder.up.0.block.2.conv1.weight" in default
          and "encoder.down.0.block.2.conv1.weight" not in default)
    check("②⁗ ⚠️ `nin_shortcut` **只在通道变化时**才有 ✓✗：解码器 `up.0` 有（256→128 ✓）✓、"
          "编码器 `down.0` 没有（128→128 ✓）✓",
          "decoder.up.0.block.0.nin_shortcut.weight" in default
          and "encoder.down.0.block.0.nin_shortcut.weight" not in default)
    check("②⁵ attn 的 `q/k/v/proj_out` 是**带偏置**的 1×1 卷积 ✓"
          "（⚠️ 与 SDXL UNet 的 `to_q` **无偏置** ✗ 是两套口径 ✓ 别串 ✓）",
          "decoder.mid.attn_1.proj_out.bias" in default
          and "decoder.mid.attn_1.q.weight" in default and "quant_conv.weight" in default)
    check("②⁶ 缩小版键名与默认**同构** ✓（只是少层/换通道 ✓）—— 数**从结构算** ✓："
          "2 层 × (编码器 1 块 / 解码器 2 块 ✓) = **124** 键 ✓",
          len(mod.sdxl_vae_key_names(mod.SdxlVaeConfig(**TOY))) == 124,
          len(mod.sdxl_vae_key_names(mod.SdxlVaeConfig(**TOY))))


def case_structure() -> None:
    """③ 结构 ✓：模块装出来就是那份键集 ✓、`proj_out` 零初始化 ⇒ 注意力**恒等** ✗✗。"""
    if not mod.has_torch():
        skip("没装 torch ⇒ ③~⑥ 不跑 ✓（**没跑 ≠ 绿** ✗）")
        return
    import torch  # noqa: PLC0415

    spec = mod.SdxlVaeConfig(**TOY)
    model = mod.build_sdxl_vae(spec)
    mine = dict(model.state_dict())
    expected = mod.sdxl_vae_key_names(spec)
    check(f"③ ⭐ 模块 `state_dict` 与**结构推出的键集**逐个全等 ✗✗（{len(expected)} 键 ✓ ⇒ "
          f"少一个块 / 通道序错 / 解码器方向反了，都会在这条露 ✓）",
          sorted(mine) == list(expected), [k for k in expected if k not in mine][:5])
    check("③′ 层级数如实 ✓：`encoder.down` / `decoder.up` 各 == 层数 ✓；每层块数 = "
          "`encoder_blocks`(1 ✓) 与 `decoder_blocks`(2 ✓) —— ⚠️ **从配置算** ✓ 不写死 2/3 ✗"
          "（写死 ⇒ 换个 `layers_per_block` 就假绿 ✓✗）",
          (len(model.encoder.down), len(model.decoder.up)) == (spec.num_levels, spec.num_levels)
          and all(len(level.block) == spec.encoder_blocks for level in model.encoder.down)
          and all(len(level.block) == spec.decoder_blocks for level in model.decoder.up))
    check("③″ ⚠️ 解码器**入通道按「前向反序」接** ✓✗：`up.1`（先算 ✓）从 mid 拿 16 ✓、"
          "`up.0`（后算 ✓）从 `up.1` 拿 16 ✓ ⇒ 最后收到 8 ✓",
          (model.decoder.up[1].block[0].conv1.in_channels,
           model.decoder.up[1].block[0].conv1.out_channels,
           model.decoder.up[0].block[0].conv1.in_channels,
           model.decoder.up[0].block[0].conv1.out_channels) == (16, 16, 16, 8))
    check("③‴ ⭐⭐ `proj_out` 零初始化 ⇒ 新模型的 mid 注意力是**恒等** ✗✗"
          "（漏了这一步 ⇒ 会有非零残差 ⇒ 这条必红 ✓）",
          float(model.decoder.mid.attn_1.proj_out.weight.abs().sum()) == 0.0
          and float(model.encoder.mid.attn_1.proj_out.weight.abs().sum()) == 0.0)
    probe = torch.randn(1, 16, 4, 6)
    with torch.no_grad():
        out = model.decoder.mid.attn_1(probe)
    check("③⁗ 上条的**前向**验证 ✓：`attn_1(x) == x` ✓（真走了算子 ✓ 不是短路 ✓）",
          torch.equal(out, probe))


def case_forward() -> None:
    """④ 前向 ✓：**non-square** 也走同一条路 ✓、尺寸账与潜空间口径对得上 ✓。"""
    if not mod.has_torch():
        skip("没装 torch ⇒ ③~⑥ 不跑 ✓（**没跑 ≠ 绿** ✗）")
        return
    import torch  # noqa: PLC0415

    spec = mod.SdxlVaeConfig(**TOY)
    model = mod.build_sdxl_vae(spec)
    model.eval()
    images = torch.rand(2, 3, 8, 12)
    with torch.no_grad():
        mean, logvar = model.encode(images)
        pixels = model.decode(mean)
    check("④ 编码：`8×12` ⇒ 潜变量 `4×6` ✓（`spatial_scale=2` ✓）、**均值/对数方差各 4 通道** ✓"
          "（`2×latent` 切开 ✓）、**non-square 照走** ✓",
          tuple(mean.shape) == (2, 4, 4, 6) and tuple(logvar.shape) == (2, 4, 4, 6),
          (tuple(mean.shape), tuple(logvar.shape)))
    check("④′ 解码回 `8×12` ✓（上采样次数 = 层数−1 = 1 ✓）；输出 3 通道 ✓ 且**没有** clamp 到 [0,1] ✗"
          "（原始口径是 `[-1,1]` ✓、单位化在 `decode_latents` ✓）",
          tuple(pixels.shape) == (2, 3, 8, 12), tuple(pixels.shape))
    with torch.no_grad():
        unit = mod.decode_latents(model, mean)
    check("④″ `decode_latents` 回 `[0,1]` ✓（与 `image_ops` 同口径 ✓）且形状不变 ✓",
          tuple(unit.shape) == (2, 3, 8, 12)
          and float(unit.min()) >= 0.0 and float(unit.max()) <= 1.0,
          (float(unit.min()), float(unit.max())))
    check("④‴ ⚠️ `to_unit=False` 回原始口径 ✓ ⇒ 与 `to_unit=True` **只差单位化 + 夹取** ✓"
          "（**不是两套网络** ✓）—— ⚠️ **必须带 `.clamp` 比** ✗：随机初始化下解码值会大幅越界 ✓，"
          "漏掉夹取 ⇒ 这条假红 ✓（早先就是漏了 ✓）",
          torch.allclose((mod.decode_latents(model, mean, to_unit=False) * 0.5 + 0.5)
                         .clamp(0.0, 1.0), unit, atol=1e-6))


class _Identity:
    """恒等「解码器」桩 ✓：`decode(z) == z` ✓ ⇒ 出来的就是**缩放后**的潜变量 ✓（把算术钉死 ✓）。"""

    def __init__(self, spec: Any) -> None:
        self.config = spec

    def decode(self, latents: Any) -> Any:
        return latents


def case_scale() -> None:
    """⑤ 缩放口径 ✗✗：**解码必须先除** `1/0.13025` ✓ —— 用**恒等桩**把算术钉死 ✓。"""
    if not mod.has_torch():
        skip("没装 torch ⇒ ③~⑥ 不跑 ✓（**没跑 ≠ 绿** ✗）")
        return
    import torch  # noqa: PLC0415

    spec = mod.SdxlVaeConfig()
    latents = torch.tensor([[[[0.13025, -0.13025]]]], dtype=torch.float32)
    got = mod.decode_latents(_Identity(spec), latents, config=spec, to_unit=False)
    check("⑤ ⭐⭐ **先除缩放系数** ✗✗：`0.13025 ⇒ 1.0` ✓、`-0.13025 ⇒ -1.0` ✓"
          "（⚠️ 漏了这一步**不会报错** ✗ —— 出来的是「有形状、有颜色、但整体发灰/过曝」的图 ✓✗ "
          "属本仓最忌的「看不出来」那类错 ✓）",
          torch.allclose(got, torch.tensor([[[[1.0, -1.0]]]])), got.tolist())
    unit = mod.decode_latents(_Identity(spec), latents, config=spec)
    check("⑤′ `to_unit=True` 即 `(x+1)/2` 且**夹到 [0,1]** ✓（超界被 clamp ✓ 不是溢出 ✓）",
          torch.allclose(unit, torch.tensor([[[[1.0, 0.0]]]])), unit.tolist())
    shifted = mod.SdxlVaeConfig(scaling_factor=0.5, shift_factor=2.0)
    check("⑤″ `shift_factor` 也参与 ✓（口径是 `(z − shift) / scale` ✓ —— SDXL 的 shift 是 0 ✓ "
          "但**别写死成 0** ✗）",
          torch.allclose(mod.decode_latents(_Identity(shifted), torch.tensor([[[[3.0]]]]),
                                            config=shifted, to_unit=False),
                         torch.tensor([[[[2.0]]]])))
    zero = type("_Raw", (), {"scaling_factor": 0.0, "shift_factor": 0.0})()
    check("⑤‴ 缩放为 0 ⇒ **响亮报错** ✗（绝不除以 0 静默出 NaN ✓）—— ⚠️ 配置层已拦一次 ✓，"
          "这里是**第二道**：拿**绕过校验**的裸配置对象来，也必须拦 ✓（判据要够得着 ✓）",
          _raises(lambda: mod.decode_latents(_Identity(zero), torch.zeros(1, 4, 2, 2),
                                             config=zero), "除") is not None)


def case_load() -> None:
    """⑥ 装载 ✓：前缀剥掉 ✓、外来键**如实返回** ✓、缺键/错形状/没前缀 ⇒ **报错** ✗。"""
    if not mod.has_torch():
        skip("没装 torch ⇒ ③~⑥ 不跑 ✓（**没跑 ≠ 绿** ✗）")
        return
    import torch  # noqa: PLC0415

    spec = mod.SdxlVaeConfig(**TOY)
    source = mod.build_sdxl_vae(spec).state_dict()
    prefixed = {mod.SDXL_VAE_KEY_PREFIX + key: value for key, value in source.items()}
    target = mod.build_sdxl_vae(spec)
    others = mod.load_sdxl_vae_state_dict(target, dict(prefixed))
    loaded = target.state_dict()
    check("⑥ 带前缀的**全装上了** ✓（`strict` 下缺一个都会红 ✓）、外来键为空 ✓、逐张量相等 ✓",
          others == [] and sorted(loaded) == sorted(source)
          and all(torch.equal(loaded[key], source[key]) for key in source))
    mixed = dict(prefixed)
    mixed["model.diffusion_model.input_blocks.0.0.weight"] = torch.zeros(1)
    mixed["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"] = torch.zeros(1)
    check("⑥′ ⚠️ **不属于 VAE 的键原样跳过并如实返回** ✓✗（调用方据此核对"
          "「UNet / 文本编码器各多少键」✓ —— 假装认识它们会埋雷 ✓）",
          mod.load_sdxl_vae_state_dict(mod.build_sdxl_vae(spec), mixed)
          == ["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
              "model.diffusion_model.input_blocks.0.0.weight"])
    missing = {key: value for key, value in prefixed.items()
               if key != mod.SDXL_VAE_KEY_PREFIX + "decoder.up.0.block.0.nin_shortcut.weight"}
    check("⑥″ ⚠️ 缺一个键 ⇒ **报错** ✗（不许 `strict=False` 糊过去 ✓✗ —— 那会留下"
          "**随机初始化的层** ✓：图上多一片雪花而**不报错** ✓）",
          _raises(lambda: mod.load_sdxl_vae_state_dict(mod.build_sdxl_vae(spec), missing),
                  "不符") is not None)
    wrong = dict(prefixed)
    wrong[mod.SDXL_VAE_KEY_PREFIX + "decoder.conv_out.weight"] = torch.zeros(1, 1, 1, 1)
    check("⑥‴ 形状对不上 ⇒ **报错** ✗（同上：不许静默 ✓）",
          _raises(lambda: mod.load_sdxl_vae_state_dict(mod.build_sdxl_vae(spec), wrong),
                  "不符") is not None)
    check("⑥⁗ 前缀一个都没有 / 空映射 ⇒ **报错** ✗（这时它根本不是 SDXL 检查点 ✓）",
          _raises(lambda: mod.load_sdxl_vae_state_dict(mod.build_sdxl_vae(spec),
                                                       {"a": torch.zeros(1)}), "检查点") is not None
          and _raises(lambda: mod.load_sdxl_vae_state_dict(mod.build_sdxl_vae(spec), {}),
                      "非空") is not None)


def case_real_weights() -> None:
    """⑦ 真权重对齐 ✓：**248 键的名字 + 形状**逐个全等 ✓（只读头部 ✓ 不载张量 ✗）。"""
    if not mod.has_torch():
        skip("没装 torch ⇒ 真权重对齐不跑 ✓（**没跑 ≠ 绿** ✗）")
        return
    path, why = find_sdxl_base_checkpoint()
    if path is None:
        skip(why)
        return

    import torch  # noqa: PLC0415
    from app.services.engine.safetensors import read_header  # noqa: PLC0415

    _, header = read_header(path)
    prefix = mod.SDXL_VAE_KEY_PREFIX
    real = {key[len(prefix):]: tuple(header[key]["shape"])
            for key in header if key.startswith(prefix)}
    check(f"⑦ 真权重 `first_stage_model.*` 恰 **{REAL_KEYS}** 键 ✓（实测 {len(real)} ✓）",
          len(real) == REAL_KEYS, len(real))

    # ⚠️ 用 **meta 设备**建模型 ✓：拿到全部形状但**不占内存** ✗（83M 参数 ⇒ 别真建 ✓）。
    with torch.device("meta"):
        model = mod.build_sdxl_vae(mod.SdxlVaeConfig())
    mine = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    missing = sorted(set(real) - set(mine))
    extra = sorted(set(mine) - set(real))
    mismatched = sorted(key for key in set(real) & set(mine) if real[key] != mine[key])
    check("⑦′ ⭐⭐ **键集全等** ✗✗（块数 / 通道序 / 解码器键序错一处，这条必红 ✓）",
          not missing and not extra, {"缺": missing[:5], "多": extra[:5]})
    check("⑦″ ⭐⭐ **逐键形状全等** ✗✗（通道账 / `nin_shortcut` 的位置 / "
          "下采样的位置全在这一条里 ✓）", not mismatched,
          [(key, real[key], mine[key]) for key in mismatched[:5]])
    check("⑦‴ 参数量与真权重**算出来的**一致 ✓（少一个块 / 多个块必然对不上 ✓）",
          sum(torch.Size(shape).numel() for shape in mine.values())
          == sum(torch.Size(shape).numel() for shape in real.values()),
          sum(torch.Size(shape).numel() for shape in mine.values()))
    check("⑦⁗ 缩放的**事实**与真权重口径一致 ✓（0.13025 ✓ / 潜空间 4 通道 ✓）",
          mod.SDXL_VAE_FACTS["scalingFactor"] == 0.13025
          and (4,) == (mine["decoder.conv_in.weight"][1],))


def main() -> int:
    if not mod.has_torch():
        skip("没装 torch ⇒ ③~⑥ 不跑 ✓（**没跑 ≠ 绿** ✗）")
    else:
        for case in (case_config, case_key_names, case_structure, case_forward,
                     case_scale, case_load):
            case()
    case_real_weights()

    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      实测: {detail}")
    for reason in _SKIPS:
        print(f"SKIP  {reason}")
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
