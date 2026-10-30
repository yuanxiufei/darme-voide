r"""自研引擎 · **反量化**（fp8 / int8 → 可算精度 ✓）—— 目标主权重就是 fp8 ✓ 这条在关键路径上 ✓。

覆盖：
* 四种布局（per-tensor / per-row / per-col / 逐元素 ✓）**各自**按形状判对 ✓ 且**往返**（量化→反量化 ≈ 原值 ✓）；
* 两种 scale 方向（`*.scale` 乘 / `*.weight_scale_inv` 除 ✓）—— ⚠️ 长后缀优先 ✗（否则 `_inv` 会被
  `weight_scale` 前缀命中 ⇒ 把「除」当「乘」 ✓✗，那是最坏的一种错 ✓）；
* ⭐ **判不出来必须拒绝** ✗：块量化形状 / 分组量化（`qzeros` ✓）/ 缺 scale / scale 里有 0 ✓；
* ⭐ 接进 `weights.load_module_weights`：**fp8 检查点能真装** ✓（数值对得上 ✓）且**拒绝时中止装载**
  ✓（不污染目标模块 ✓、不静默 cast ✗）；
* ⭐ **按盘上的真文件**判：`convrot_w4a4`（说明里没有 `convrot` 布尔 ✗）/ 说明只写在 metadata
  `_quantization_metadata` 里 ✗ / 同一文件里 convrot 分组**逐层不同** ✗ —— 三条都曾判错
  （前两条⇒少一步反旋转＝假绿 ✓✗，第三条⇒把 31.7 GiB 正品整个拒掉 ✗）。真文件不在盘上
  ⇒ **单列 SKIP** ✓，绝不混进 PASS ✗。

运行::

    ./.venv/Scripts/python.exe tests/engine_quant_test.py
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from app.services.engine import quant as quant_mod  # noqa: E402
from app.services.engine import weights as weights_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPPED: list[str] = []

#: 真实文件（盘上未必有 ⇒ 找不到就**单列 SKIP** ✓ 不混进 PASS ✗ —— 混进去就是假绿 ✓）
REAL_KREA2_W4A4 = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models\Krea2"
                       r"\krea2_v1.5_anime_int4_w4a4.safetensors")
REAL_KREA2_MOODY = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models\Krea2"
                        r"\Krea2-Moody-Mix-premium_int4_convrot.safetensors")
#: ⚠️ **全量**版（31.7 GiB）—— 它的 convrot 分组**逐层不同**（64 与 256 并存 ✓）
REAL_H3_FULL = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models"
                    r"\minimax_h3_fl2va_int8_convrot.safetensors")
#: fp8_scaled 的**大件**（406 层 ✓ 且文件里**一个 `*.comfy_quant` 都没有** ✗ ⇒ 全靠形状自证 ✓）
REAL_WAN_FP8 = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models"
                    r"\wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
#: **纯 fp8、一个 scale 都没有**的那一族（301 个权重 ✓）—— 旧判据把整族判成「缺配套 ⇒ 不装」✗
REAL_SEEDVR2_FP8 = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models\SEEDVR2"
                        r"\seedvr2_ema_3b_fp8_e4m3fn.safetensors")
#: 第三个变体：`scale_weight` ✓ + `scaled_fp8` 标记 ✓ + `spiece_model`（分词器 proto ✓ 不是权重 ✓）
REAL_UMT5_FP8 = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\text_encoders"
                     r"\umt5_xxl_fp8_e4m3fn_scaled.safetensors")
#: ⭐ convrot 分组**逐层不同**的第二个实例（2026-10-30 ✓）：310 个权重里 language_model 是 256、
#: `visual.blocks.*` 是 64、另有 27 层是 16 ✓ —— 旧判据按**文件级**取一个分组 ⇒ visual 的
#: 输入维 1152 不整除 256 ⇒ 108 个 `model.visual.blocks.N.{attn.qkv,attn.proj,mlp.linear_fc1,
#: mlp.linear_fc2}` 全判「配不上」（27×4＝108 ✓，正是那一串 unresolved ✓✗）
REAL_QWEN35_PE = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\text_encoders"
                      r"\qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors")


def check(name: str, ok: bool, detail: object = None) -> None:
    _RESULTS.append((name, bool(ok), detail))


def skip(name: str, why: str) -> None:
    """跳过的判据**单独计数并打印** ✓✗ —— 混进 PASS 里就是假绿 ✓。"""
    _SKIPPED.append(f"{name}（{why}）")


def _torch() -> Any:
    import torch  # noqa: PLC0415

    return torch


def _skips() -> list[str]:
    try:
        _torch()
    except ImportError:
        return ["未装 torch ⇒ 反量化自检跳过（本模块要真张量 ✓）"]
    return []


# ══════════════════════════════════════════════════════════════════════════
# ① 四种布局 + 两种方向：往返要**真的**对得上
# ══════════════════════════════════════════════════════════════════════════
def case_layouts() -> None:
    torch = _torch()
    torch.manual_seed(7)
    weight = torch.randn(6, 4)
    # 用**同一条**量化路径造 fp8：`q = (w / s).to(fp8)` ⇒ 反量化 `q * s` ⇒ 与 w 的误差应当很小 ✓
    # ⚠️ 每个布局的 scale **广播方式不同** ✗（per-row 要 `(rows,1)✓`、per-col 要 `(1,cols)✓`
    #    —— 直接 `weight / scale` 会在 per-row 上因为维度对不上**当场抛** ✓，第一版就踩了 ✓）。
    cases: dict[str, tuple[Any, Any]] = {
        "per-tensor": (torch.tensor(0.02), None),
        # ⚠️ per-row 的 scale **各不相等** ✗（第一版全填 0.02 ✓✗ ⇒ ①″ 的"用错布局"根本看不出差别 ✓
        #    —— 均一的 scale 让"只取第一行"也数值相同 ✓ ⇒ 反向证明变成恒真 ✓✗，当场红 ✓）
        "per-row": (torch.linspace(0.01, 0.06, 6), (6, 1)),
        "per-col": (torch.linspace(0.01, 0.04, 4), (1, 4)),
        "elementwise": (torch.full((6, 4), 0.02), None),
    }
    #: ⚠️ fp8 **e4m3** 的相对精度约 `2^-3`（12.5% ✓）⇒ |w| ≤ 4 时绝对误差可到 ~0.1 ✓
    #: ⇒ 容差取 0.15 ✓（第一版写 0.05 ✗ ⇒ 明明对的实现被误判成 FAIL ✓✗）
    tol = 0.15
    layouts: dict[str, str] = {}
    errors: dict[str, float] = {}
    for label, (scale, shape) in cases.items():
        scaled = weight / (scale if shape is None else scale.reshape(shape))
        fp8 = scaled.to(torch.float8_e4m3fn)
        state = {"w.weight": fp8, "w.scale": scale}
        result = quant_mod.dequantize_state(state)
        if not result.ok:
            errors[label] = float("inf")
            continue
        restored = result.tensors["w.weight"]
        layouts[label] = result.converted["w.weight"]["layout"]
        errors[label] = float((restored - weight).abs().max())
    check("① ⭐ 四种布局**各自被按形状判对** ✓（per-tensor / per-row / per-col / 逐元素 ✓）",
          layouts == {"per-tensor": "per-tensor", "per-row": "per-row",
                      "per-col": "per-col", "elementwise": "elementwise"}, layouts)
    check("①′ ⭐ 四种布局**往返都对得上** ✓（误差 ≤ 0.15 ✓ = fp8 e4m3 的精度量级 ✓ —— "
          "布局判错会到 10⁰ 量级 ✓✗）",
          all(value < tol for value in errors.values()), errors)

    # ⚠️ 布局判错的**反面**：故意把 per-row 的 scale 当 per-tensor 用 ⇒ 误差必须**大得多** ✓
    scale = torch.linspace(0.01, 0.06, 6)
    fp8 = (weight / scale.reshape(6, 1)).to(torch.float8_e4m3fn)
    right = float((fp8.to(torch.float32) * scale.reshape(6, 1) - weight).abs().max())
    wrong = float((fp8.to(torch.float32) * scale[:1] - weight).abs().max())
    check("①″ 反向证明：**布局判错**（per-row 当 per-tensor ✓）⇒ 误差确实大得多 ✓"
          "（⇒ 第 ①′ 条比的是真行为 ✓ 不是恒真 ✓）",
          right < tol and wrong > 10 * right, (right, wrong))

    inverse = quant_mod.dequantize_state({"w.weight": (weight * 50.0).to(torch.float8_e4m3fn),
                                          "w.weight_scale_inv": torch.tensor(50.0)})
    check("② ⭐ `weight_scale_inv` 走**除** ✓（DeepSeek 系写法 ✓）：`w = q / s` ✓",
          inverse.ok and inverse.converted["w.weight"]["inverse"] == "True"
          and float((inverse.tensors["w.weight"] - weight).abs().max()) < tol,
          inverse.to_dict())

    both = quant_mod.dequantize_state({"w.weight": torch.ones(2, 2).to(torch.float8_e4m3fn),
                                       "w.weight_scale": torch.tensor(2.0),
                                       "w.weight_scale_inv": torch.tensor(0.5)})
    check("③ ⚠️ 两种 scale 同时存在 ⇒ **长后缀优先** ✗（用 `_inv` ✓ —— 若按短后缀前缀命中，"
          "会把「除」当「乘」✓✗：1/0.5 = 2 而不是 1×2 ✓）",
          both.ok and both.converted["w.weight"]["scale"].endswith("_inv")
          and float(both.tensors["w.weight"][0, 0]) == 2.0, both.to_dict())

    # ⚠️ **带退化维**的 scale 写法（2026-10-29 实测盘上两个 19.53 GiB 主 DiT ✓）：
    #    `[5376, 7168]` 权重配 `[5376, 1]` scale ⇒ 旧判据只认 `(5376,)` ⇒ 报「对不上任何布局」✗，
    #    与文件自带的 `{"format": "int8_tensorwise"}` 直接矛盾 ✓✗ ⇒ 报告让人去找不存在的 json ✗。
    check("④ ⭐ `[N,1]` / `[1,M]` 这类**带退化维**的 scale 与 `(N,)` / `(M,)` **等价** ✓"
          "（形状自证 ✓ 不是猜 ✗）",
          quant_mod.layout_of((5376, 7168), (5376, 1)) == "per-row"
          and quant_mod.layout_of((5376, 7168), (1, 7168)) == "per-col"
          and quant_mod.layout_of((5376, 7168), (1, 1)) == "per-tensor",
          (quant_mod.layout_of((5376, 7168), (5376, 1)),
           quant_mod.layout_of((5376, 7168), (1, 7168))))

    # ⚠️ **文件自己说的**比按形状猜硬 ✓：ComfyUI 给每个低精度权重配 `*.comfy_quant`（JSON ✓）
    spec_convrot = quant_mod.parse_format_spec(
        b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}\x00')
    spec_nvfp4 = quant_mod.parse_format_spec(b'{"format": "nvfp4", "full_precision_matrix_mult": true}')
    # ⚠️ convrot **2026-10-29 起已实现** ✓、nvfp4 **2026-10-30 起也已实现** ✓ ⇒ 两族在
    #    `format_family`（＝「做不了的族」栏 ✓）里都必须回到 `None` ✗ —— nvfp4 改由
    #    `supported_format` 认领 ✓（`quant_nvfp4` 负责逐层还原 ✓），`mxfp4` 才是真做不了的那个 ✗。
    group_convrot = quant_mod.convrot_group_of([spec_convrot])
    nvfp4 = quant_mod.format_family(spec_nvfp4)
    nvfp4_ok = quant_mod.supported_format(spec_nvfp4)
    mxfp4 = quant_mod.format_family({"format": "mxfp4"})
    check("⑤ ⭐ `comfy_quant` 能解析 ✓ 且**两张表各认各的** ✓：`convrot` 判出 group=256 ✓、"
          "`nvfp4` 落 `SUPPORTED_FORMATS` ✓（不在「未实现族」栏 ✗）、`mxfp4` 才是未实现的族 ✗、"
          "`int8_tensorwise` 两栏都不进 ✓",
          spec_convrot is not None
          and quant_mod.format_family(spec_convrot) is None
          and group_convrot == (256, None)
          and nvfp4 is None and nvfp4_ok is not None and nvfp4_ok[0] == "nvfp4"
          and mxfp4 is not None and mxfp4[0] == "mxfp4"
          and quant_mod.supported_format({"format": "mxfp4"}) is None
          and quant_mod.format_family({"format": "int8_tensorwise"}) is None
          and quant_mod.parse_format_spec(b"not json") is None,
          (group_convrot, nvfp4, nvfp4_ok, mxfp4))

    # ⚠️ 分组**逐层不同** ⇒ 必须给出逐层表 ✓（2026-10-30 改判 ✗：旧版这里判「拒绝」✓✗，
    #    结果是 H3 **全量**版 31.7 GiB 的正品被整份拒掉 ✓ —— 实测它 `blocks.0.adaln_proj` 是 64、
    #    `attn.qkv_proj` 是 256 ✓。拒绝**不是**安全，是把真能干的事说成干不了 ✓）。
    mixed = quant_mod.convrot_group_of({
        "blocks.0.adaln_proj.linear": {"format": "int8_tensorwise", "convrot": True,
                                       "convrot_groupsize": 64},
        "blocks.0.attn.qkv_proj": {"format": "int8_tensorwise", "convrot": True,
                                   "convrot_groupsize": 256}})
    check("⑤″ 同一文件里 convrot 分组**逐层不同** ⇒ 给**逐层表** ✓（不是拒绝、也不是猜一个 ✓）",
          mixed == ({"blocks.0.adaln_proj.linear": 64, "blocks.0.attn.qkv_proj": 256}, None), mixed)

    #: ⭐ 本条是「假绿」的正面判据 ✗✗：Krea2 那族写 ``{"format": "convrot_w4a4"}`` 而**没有**
    #: ``convrot: true`` 布尔 ✓ —— 只认布尔 ⇒ 判成「不是 convrot」⇒ 装载少一步反旋转 ⇒ 能装能跑、
    #: 画面全错却报「可自动还原」✓✗。所以判据必须是「**说明里写着 convrot 就算**」✓。
    w4a4 = quant_mod.convrot_group_of(
        {"blocks.0.attn.wk": {"format": "convrot_w4a4", "convrot_groupsize": 256,
                              "quant_group_size": 64}})
    check("⑤‴ `convrot_w4a4` 一族**没写** `convrot` 布尔也算旋转 ✓（只认布尔 ⇒ 假绿 ✓✗）",
          w4a4 == (256, None), w4a4)
    check("⑤⁗ 这族进了「已支持格式」表 ✓（否则报告会自相矛盾：认得出又报不认识 ✓✗）",
          quant_mod.supported_format({"format": "convrot_w4a4"}) is not None
          and quant_mod.supported_format({"format": "convrot_w4a4"})[0] == "convrot_w4a4",
          quant_mod.supported_format({"format": "convrot_w4a4"}))

    #: ⭐ 逐层表**查不到**的层 ⇒ **不旋转** ✓✗（拿别层的 group 顶上＝瞎转 ✓✗，而且形状可能恰好
    #: 整除 ⇒ 静默出错 ✓）。这是「逐层」这个改动的**唯一**安全边界 ✓。
    per_layer = {"blocks.0.attn.qkv_proj": 256}
    check("⑤⁵ 逐层表里没有的层 ⇒ `group_for` 返回 None ✓（不拿别层的 group 顶上 ✗）",
          quant_mod.group_for("blocks.0.attn.qkv_proj", per_layer) == 256
          and quant_mod.group_for("blocks.9.nope", per_layer) is None
          and quant_mod.group_for("anything", 256) == 256
          and quant_mod.group_for("anything", None) is None,
          (quant_mod.group_for("blocks.0.attn.qkv_proj", per_layer),
           quant_mod.group_for("blocks.9.nope", per_layer)))

    #: ⭐ **文件级**说明（StarNodes 那一族写在 metadata 里 ✓ —— 文件里一个 `*.comfy_quant` 都没有 ✗）。
    meta = {"_quantization_metadata": (
        '{"format": "int4_convrot", "layers": {'
        '"model.diffusion_model.blocks.0.attn.wk": {"format": "convrot_w4a4", '
        '"convrot_groupsize": 256, "quant_group_size": 64}}}')}
    specs = quant_mod.specs_from_metadata(meta)
    check("⑤⁶ metadata `_quantization_metadata` ⇒ 逐层说明能读出来 ✓（实测该文件 256 层全在这 ✓）",
          list(specs) == ["model.diffusion_model.blocks.0.attn.wk"]
          and quant_mod.convrot_group_of(specs) == (256, None), specs)
    check("⑤⁷ 坏 metadata / 无 metadata ⇒ 空表 ✓（不抛错、也不编一个 group 出来 ✗）",
          quant_mod.specs_from_metadata({}) == {}
          and quant_mod.specs_from_metadata(None) == {}
          and quant_mod.specs_from_metadata({"_quantization_metadata": "not json"}) == {}
          and quant_mod.specs_from_metadata(
              {"_quantization_metadata": '{"layers": {"a": {"format": "fp8"}}}'}) == {"a": {"format": "fp8"}},
          quant_mod.specs_from_metadata({"_quantization_metadata": "not json"}))


# ══════════════════════════════════════════════════════════════════════════
# ①′ convrot：Hadamard 归一化 ✓ / 反旋转**逐位**等于手算 ✓（不是"看着差不多" ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_convrot() -> None:
    torch = _torch()
    torch.manual_seed(11)
    group = 256
    base = torch.randn(8, group)
    h = quant_mod.build_hadamard(group)
    eye_gap = float((h @ h.transpose(-1, -2) - torch.eye(group)).abs().max())
    check("①′ ⭐ convrot 的 Hadamard **归一化**正确 ✓（`H@H^T == I` ✓ —— 少了 `/sqrt(n)` ⇒ "
          "反旋转会把数值整体放大 256 倍 ✓✗）", eye_gap < 1e-6, eye_gap)

    twice = quant_mod.rotate_weight(quant_mod.rotate_weight(base, h, group), h, group)
    check("①″ ⭐ H **对称** ⇒ `rotate` 两次就是还原 ✓（存盘与还原**共用同一个算子** ✓ —— "
          "参考实现 `_dequantize_convrot_int8` 正是复用 `_rotate_weight` ✓，看着像笔误其实对 ✓）",
          float((twice - base).abs().max()) < 1e-5, float((twice - base).abs().max()))

    # 端到端：真走 `dequantize_state` ✓ —— 造一份**带旋转**的低精度检查点 ✓
    scale = torch.full((8, 1), 0.02)
    rotated = quant_mod.rotate_weight(base, h, group)
    packed = (rotated / scale).to(torch.float8_e4m3fn)
    state = {"w.weight": packed, "w.scale": scale.reshape(8)}
    restored = quant_mod.dequantize_state(state, convrot=group)
    plain = quant_mod.dequantize_state(state)
    manual = quant_mod.rotate_weight(packed.to(torch.float32) * scale, h, group)
    check("①‴ ⭐⭐ convrot 反解 = `(q*s) 按 group 分块 @ H^T` ✓ **逐位**对上 ✓"
          "（布局标成 `per-row+convrot256` ✓ —— 报告里必须看得见多了一步旋转 ✓）",
          restored.ok and float((restored.tensors["w.weight"] - manual).abs().max()) < 1e-6
          and restored.converted["w.weight"]["layout"] == "per-row+convrot256",
          (restored.converted, restored.problems))

    # ⚠️ 反面：**不传** convrot ⇒ 少一步反旋转 ⇒ 差到同一个量级 ✓（证明那一步不能省 ✓ 不是恒真 ✓）
    gap = float((plain.tensors["w.weight"] - base).abs().max())
    check("①⁗ 反向证明：**漏传** convrot ⇒ 反解结果与真权重差得远 ✓（少一步反旋转不是小事 ✗）",
          gap > 1.0, gap)

    # 形状与声明冲突 ⇒ **拒绝** ✓（不静默跳过旋转 ✗）；体检侧布局名带后缀 ✓
    conflict = quant_mod.dequantize_state(state, convrot=512)
    plan = quant_mod.plan_dequant({"w.weight": ("int8", (8, group)), "w.scale": ("F32", (8, 1))},
                                 convrot=group)
    check("①⁵ convrot 形状冲突（输入维不整除 group ✓）⇒ **拒绝** ✓ / 体检布局名带 `+convrot256` ✓",
          not conflict.ok and plan["supported"] is True
          and plan["layouts"] == {"per-row+convrot256": 1},
          (conflict.problems, plan["layouts"]))


# ══════════════════════════════════════════════════════════════════════════
# ④ 真实文件：判据必须**按盘上的文件**走（不在盘上 ⇒ 单列 SKIP ✓ 不混进 PASS ✗）
# ══════════════════════════════════════════════════════════════════════════
def _real(path: Path) -> tuple[Any, dict[str, Any], Any] | None:
    """读一个真文件的**逐层说明** ⇒ ``(info, specs, (convrot, gap))`` ✓（只读头部，不装载 ✓）。"""
    from app.services.engine import loader as loader_mod  # noqa: PLC0415
    from app.services.engine import safetensors as st  # noqa: PLC0415

    if not path.is_file():
        return None
    info = st.inspect(path)
    specs = loader_mod.quant_specs_of(path, info)
    return info, specs, quant_mod.convrot_group_of(specs)


def case_real_files() -> None:
    from collections import Counter  # noqa: PLC0415

    from app.services.engine import facade as facade_mod  # noqa: PLC0415

    #: ① **假绿的那一份**（2026-10-30 实测）：224 层说明**全在 `*.comfy_quant`** 里 ✓ 且
    #:    格式串是 ``convrot_w4a4``（**没有** `convrot` 布尔 ✗）⇒ 只看布尔 ⇒ 少一步反旋转却报
    #:    「可自动还原」✓✗；落盘的 `weight` 是 **int8 值域 + per-row scale**（不是 4bit 打包 ✗）
    #:    ⇒ 缺的只有反旋转 ✓。
    real = _real(REAL_KREA2_W4A4)
    if real is None:
        skip("④ Krea2 `convrot_w4a4` 真文件（224 层逐层说明 ⇒ 同组 256 + 反旋转在布局名里）",
             f"不在盘上：{REAL_KREA2_W4A4.name}")
    else:
        info, specs, (convrot, gap) = real
        report = facade_mod.inspect_weights(REAL_KREA2_W4A4)
        quant = report["quant"]
        check("④ ⭐ 真文件 `convrot_w4a4`（224 层逐层说明）⇒ **判出 convrot=256** ✓ 且布局名带 "
              "`+convrot256` ✓（只认 `convrot` 布尔 ⇒ 这里 `convrot` 会是 None ✓✗、布局退化成 "
              "`per-row` ⇒ 少一步反旋转、画面全错却报「可还原」✓✗）",
              len(specs) == 224
              and Counter(str(spec.get("format")) for spec in specs.values()) == {"convrot_w4a4": 224}
              and convrot == 256 and gap is None
              and quant["supported"] is True and quant["weights"] == 224
              and quant["layouts"] == {"per-row+convrot256": 224} and not quant["unresolved"],
              (len(specs), convrot, gap, quant["supported"], quant["layouts"], info.tensor_count))

    #: ② **说明只写在 metadata 里**的那一份（文件里 `*.comfy_quant` 数 = **0** ✓）——
    #:    不读 `_quantization_metadata` 就等于「没看见旋转」✓✗。
    #:    同时它又是「**诚实拒绝**」的样本：4 层 `mlp.down` 形状 (2560, 3456)、3456 不整除 256 ✓
    #:    ⇒ 与文件声明矛盾 ⇒ `supported=False` ✓（不是靠猜放行 ✗）。
    real = _real(REAL_KREA2_MOODY)
    if real is None:
        skip("④′ Krea2 Moody（metadata 里的逐层说明 + 4 层不整除 ⇒ 诚实拒绝）",
             f"不在盘上：{REAL_KREA2_MOODY.name}")
    else:
        info, specs, (convrot, gap) = real
        comfy_quant = [name for name in info.tensors if name.endswith(".comfy_quant")]
        report = facade_mod.inspect_weights(REAL_KREA2_MOODY)
        quant = report["quant"]
        unresolved = list(quant["unresolved"])
        #: 未还原清单里是**层名**（不带 `.weight` ✓）⇒ 查头部时两种写法都试 ✓（别让键名差异
        #: 把「不整除」这个证据悄悄变成空列表 ✗ —— 第一版就踩了：`gaps == []` 让断言恒假 ✓✗）
        gaps = [int(info.tensors[name if name in info.tensors else f"{name}.weight"].shape[-1]) % 256
                for name in unresolved
                if (name in info.tensors or f"{name}.weight" in info.tensors)]
        check("④′ ⭐ 说明**只在 metadata `_quantization_metadata`** 里（`*.comfy_quant` 张量数 = 0 ✓）"
              "⇒ 仍判出 convrot=256 ✓ + 256 层逐层说明 ✓（读不到 metadata ⇒ `convrot` 是 None ✓✗）",
              comfy_quant == [] and len(specs) == 256 and convrot == 256 and gap is None,
              (len(comfy_quant), len(specs), convrot, gap))
        check("④″ ⭐ 4 层 `mlp.down`（3456 不整除 256 ✓）⇒ **诚实拒绝** ✓（`supported=False` + 逐层点名 ✓"
              " —— 不许静默跳过旋转或猜一个 group ✗）",
              quant["supported"] is False and len(unresolved) == 4
              and all(name.endswith("mlp.down") for name in unresolved)
              and all(gap_item != 0 for gap_item in gaps) and len(gaps) == 4,
              (quant["supported"], unresolved, gaps))

    #: ③ **全量版被整份拒掉**的那一份：同一文件里 64 与 256 **并存** ✓✗ ⇒ 族级「必须同组」
    #:    的判法会把它（31.7 GiB 的正品 ✓）判成「说明自相矛盾」⇒ 拒绝 ✓✗。拒绝**不是**安全 ✗。
    real = _real(REAL_H3_FULL)
    if real is None:
        skip("④‴ H3 全量版（convrot 分组逐层不同 64/256 ⇒ 必须给逐层表）",
             f"不在盘上：{REAL_H3_FULL.name}")
    else:
        info, specs, (convrot, gap) = real
        report = facade_mod.inspect_weights(REAL_H3_FULL)
        quant = report["quant"]
        expected = {64: 50, 256: 200}
        counts = Counter(convrot.values()) if isinstance(convrot, dict) else {}
        sample = next(iter(convrot)) if isinstance(convrot, dict) else ""
        check("④‴ ⭐⭐ H3 **全量**版同一文件里 convrot **64 与 256 并存** ⇒ 给**逐层表** ✓ 且"
              "`supported=True` ✓（旧判据判「自相矛盾」⇒ 整个 31.7 GiB 正品被拒 ✗ —— "
              "拒绝不是安全 ✓）",
              gap is None and isinstance(convrot, dict) and dict(counts) == expected
              and quant["supported"] is True and quant["weights"] == 250
              and quant["layouts"] == {"per-row+convrot64": 50, "per-row+convrot256": 200}
              and quant_mod.group_for(sample, convrot) == convrot[sample]
              and quant_mod.group_for("model.diffusion_model.blocks.999.nope", convrot) is None,
              (gap, dict(counts), quant["supported"], quant["layouts"], info.tensor_count))

    #: ④ fp8_scaled 的**大件**（14B ✓）：`scale_weight` 是**标量** ⇒ per-tensor ✓；
    #:    而这份文件**一个 `*.comfy_quant` 都没有** ✗ ⇒ 判据只能靠「fp8 dtype ＋ 标量 scale」形状自证 ✓。
    if not REAL_WAN_FP8.is_file():
        skip("④⁗ wan2.2 14B fp8_scaled（406 层 per-tensor 标量 scale ⇒ 靠形状自证）",
             f"不在盘上：{REAL_WAN_FP8.name}")
    else:
        from app.services.engine import safetensors as st  # noqa: PLC0415

        info = st.inspect(REAL_WAN_FP8)
        quant = facade_mod.inspect_weights(REAL_WAN_FP8)["quant"]
        check("④⁗ ⭐⭐ wan2.2 14B `fp8_scaled`：406 层 `per-tensor` ✓ 且**一个 `*.comfy_quant` 都没有**"
              " ✗（⇒ 判据只能是「fp8 ＋ 标量 `.scale_weight`」✓ —— 旧判据认不出 `.scale_weight`"
              " ⇒ 整族报「缺配套 scale」⇒ 不装 ✓✗）",
              quant["supported"] is True and quant["weights"] == 406
              and quant["layouts"] == {"per-tensor": 406}
              and not [name for name in info.tensors if name.endswith(".comfy_quant")]
              and quant_mod.SCALED_FP8_MARKER in info.tensors,
              (quant["supported"], quant["weights"], quant["layouts"]))

        #: ⭐ 真装一层：数值 = `q × scale` ✓，且 `.scale_weight` / `.scale_input` **不许**落进
        #: `unexpected` ✓✗（一个 14B 文件 800+ 个 ✓ —— 淹掉真·多余键就是假绿 ✓✗）
        from torch import nn  # noqa: PLC0415

        from app.services.engine import weights as weights_mod  # noqa: PLC0415

        shape = tuple(int(dim) for dim in info.tensors["blocks.0.cross_attn.k.weight"].shape)
        #: ⚠️ 改名要**保住末位那个点** ✗：映射成 ``"w."``（不是 ``"w"``、更不是 ``""``）——
        #: 逻辑名才是 `w.weight` ✓；空串会让逻辑名退化成 `weight` / `scale_weight` ✓✗，
        #: 于是「配套键属于哪一层」在**逻辑名空间**里对不上 ⇒ `consumedScales` 只有 1 ✓✗（第一版踩了）
        layer = nn.Module()
        layer.w = nn.Linear(shape[1], shape[0], bias=False, dtype=_torch().bfloat16)
        loaded = weights_mod.load_module_weights(
            layer, REAL_WAN_FP8, key_map={r"^blocks[.]0[.]cross_attn[.]k[.]": "w."})
        scale = struct.unpack("<f", st.read_tensor_bytes(
            REAL_WAN_FP8, "blocks.0.cross_attn.k.scale_weight", max_bytes=64))[0]
        #: ⚠️ `frombuffer` 给的是**一维** ✓ ⇒ 必须 reshape 成头部那个形状 ✗（否则相减当场抛 ✓✗）
        raw = _torch().frombuffer(st.read_tensor_bytes(
            REAL_WAN_FP8, "blocks.0.cross_attn.k.weight"),
            dtype=_torch().float8_e4m3fn).reshape(shape)
        expect = raw.float() * scale
        #: 容差按**目标 bf16 的精度**算 ✓（尾数 8 位 ⇒ 相对误差 ~0.4% ✓）：拿 `scale` 当基准会太小 ✗ ——
        #: 权重本身在 0.1 量级、bf16 间距就有 5e-4 ✓✗（第一版写 `0.02*scale` ⇒ 明明对的实现被判 FAIL ✓）
        tol = 0.01 * float(expect.abs().max()) + 1e-6
        check("④⁵ ⭐⭐ fp8_scaled **真装一层**：值 = `q × scale` ✓（方向错了会差 5 个数量级 ✓）、"
              "配套键（`scale_weight` ✓ / `scale_input` ✓ / `scaled_fp8` ✓ 共 3 个）被吃掉并**计数** ✓、"
              "`unexpected` 里**没有**它们 ✓✗",
              #: ⚠️ `error` 的默认值是**空串** ✗（不是 `None` ✓）—— 写 `is None` 会恒假 ✓✗（踩过）
              not loaded.error and loaded.loadedKeys == 1 and loaded.consumedScales == 3
              #: ⚠️ 两个条件都要 ✗：只看**后缀** ⇒ 另外 405 层没映射的 `.scale_weight` 会误判 ✓✗；
              #: 只看**前缀** ⇒ 那一层的 `bias`（Linear 里没建 ✓）也会误判 ✓✗
              and not [name for name in loaded.unexpected
                       if name.startswith("w.")
                       and name.endswith((".scale_weight", ".scale_input"))]
              and float((layer.w.weight.float() - expect).abs().max()) < tol,
              (loaded.error, loaded.loadedKeys, loaded.consumedScales, len(loaded.unexpected),
               float((layer.w.weight.float() - expect).abs().max()), tol))

    #: ⑤ **纯 fp8、一个 scale 都没有**的那一族 ✓（301 个权重 ✓）—— `cast` 布局 ✓
    if not REAL_SEEDVR2_FP8.is_file():
        skip("④⁶ seedvr2 3b 纯 fp8（无 scale ⇒ cast）", f"不在盘上：{REAL_SEEDVR2_FP8.name}")
    else:
        from app.services.engine import safetensors as st  # noqa: PLC0415

        info = st.inspect(REAL_SEEDVR2_FP8)
        quant = facade_mod.inspect_weights(REAL_SEEDVR2_FP8)["quant"]
        has_scale = [name for name in info.tensors
                     if name.endswith(tuple(quant_mod.SCALE_SUFFIXES))]
        check("④⁶ ⭐⭐ `seedvr2` 纯 fp8（**一个 scale 都没有** ✓）⇒ `cast` 布局 ＋ 放行 ✓"
              "（旧判据整族「缺配套 scale ⇒ 不装」✗ 假拒绝 ✓✗）",
              quant["supported"] is True and quant["layouts"] == {"cast": 301}
              and not has_scale,
              (quant["supported"], quant["weights"], quant["layouts"], has_scale[:3]))

    #: ⑥ 第三个 fp8 变体：`scaled_fp8` 标记 ＋ `spiece_model`（分词器 proto ✓ 不是权重 ✓）
    if not REAL_UMT5_FP8.is_file():
        skip("④⁷ umt5-XXL fp8_scaled（`scaled_fp8` 标记 + `spiece_model`）",
             f"不在盘上：{REAL_UMT5_FP8.name}")
    else:
        from app.services.engine import safetensors as st  # noqa: PLC0415

        info = st.inspect(REAL_UMT5_FP8)
        quant = facade_mod.inspect_weights(REAL_UMT5_FP8)["quant"]
        check("④⁷ ⭐ umt5-XXL `fp8_scaled`：168 层 `per-tensor` ✓ 且带 `scaled_fp8` 标记 ＋ "
              "`spiece_model`（后者是分词器 ✓ **不许**当权重算 ✓✗）",
              quant["supported"] is True and quant["weights"] == 168
              and quant["layouts"] == {"per-tensor": 168}
              and quant_mod.SCALED_FP8_MARKER in info.tensors
              and "spiece_model" in info.tensors,
              (quant["supported"], quant["weights"], quant["layouts"]))

    #: ⑦ convrot 分组**逐层不同** + 融合 QKV（`attn.qkv` 一个权重配一个整行 scale ✓）——
    #:    旧判据按文件级取 256 ⇒ `visual.blocks.*` 的输入维 1152 不整除 ⇒ 108 个 unresolved ✓✗
    if not REAL_QWEN35_PE.is_file():
        skip("④⁸ Qwen3.5-9B prompt-encoder（逐层 convrot 64/16/256 ＋ 融合 QKV）",
             f"不在盘上：{REAL_QWEN35_PE.name}")
    else:
        from app.services.engine import safetensors as st  # noqa: PLC0415

        info = st.inspect(REAL_QWEN35_PE)
        quant = facade_mod.inspect_weights(REAL_QWEN35_PE)["quant"]
        check("④⁸ ⭐⭐ Qwen3.5-9B PE：310 层里 convrot **64 / 16 / 256 三种并存** ⇒ 全配上 ✓"
              "（`unresolved` / `unpaired` / `orphanScales` **三个都空** ✓）且 `visual` 那 108 层"
              "带 `+convrot64` ✓（按文件级取 256 ⇒ 1152 不整除 256 ⇒ 108 个 `unresolved` ✓✗）",
              quant["supported"] is True and quant["weights"] == 310
              and not quant["unresolved"] and not quant["unpaired"] and not quant["orphanScales"]
              and quant["layouts"] == {"per-row+convrot256": 202, "per-row+convrot64": 81,
                                       "per-row+convrot16": 27},
              (quant["supported"], quant["weights"], quant["layouts"],
               quant["unresolved"][:3], quant["orphanScales"][:3]))

        #: ⭐ 真装一层**融合 QKV**（`attn.qkv` 3456×1152 ✓）—— 判据不看"值等于多少"（原权重不在文件里 ✓），
        #:    改用**旋转的数学不变量** ✓：Hadamard 是正交的组内变换 ⇒ 每行每 `group` 段的 L2 范数不变 ✓；
        #:    再拿**正向**旋转回去必须回到 `q × s` ✓（往返 10⁻⁷ 级 ✓）—— 两条一起才能证明
        #:    「确实反旋转了」✓ 且「用的是对的那个 H」✓✗（漏旋转、或 group 取错，两条各炸一条 ✓）
        from torch import nn  # noqa: PLC0415

        stem = "model.visual.blocks.0.attn.qkv"
        shape = tuple(int(dim) for dim in info.tensors[f"{stem}.weight"].shape)
        weight = _torch().frombuffer(
            st.read_tensor_bytes(REAL_QWEN35_PE, f"{stem}.weight"),
            dtype=_torch().int8).reshape(shape)
        scale = _torch().frombuffer(
            st.read_tensor_bytes(REAL_QWEN35_PE, f"{stem}.weight_scale"),
            dtype=_torch().float32).reshape(shape[0], 1)
        state = {f"{stem}.weight": weight, f"{stem}.weight_scale": scale}
        #: ⚠️ 文件级 256 必须**判出问题**（不是静默跳过旋转 ✗）—— 这条同时锁住「逐层分组」不许回归 ✗
        #: ⚠️ meta 的键**必须**是 ``…​.weight`` ✗（`precheck_dequant` 只认那个后缀 ✓ —— 传裸 `stem`
        #: 会被整段跳过 ⇒ 一条 problem 都没有 ⇒ 断言看着过、其实什么都没判 ✓✗，第一版踩了 ✓）
        wrong = quant_mod.precheck_dequant(
            {f"{stem}.weight": (info.tensors[f"{stem}.weight"].dtype, shape),
             f"{stem}.weight_scale": (info.tensors[f"{stem}.weight_scale"].dtype, (shape[0], 1))},
            {f"{stem}.weight_scale": scale}, convrot=256)
        restored = quant_mod.dequantize_state(state, convrot=64)
        value = restored.tensors[f"{stem}.weight"]
        scaled = weight.float() * scale
        group = 64
        norm_ok = bool(_torch().allclose(
            value.reshape(shape[0], -1, group).norm(dim=-1),
            scaled.reshape(shape[0], -1, group).norm(dim=-1), rtol=1e-4, atol=1e-4))
        back = quant_mod.rotate_weight(value, quant_mod.build_hadamard(group), group)
        layer = nn.Linear(shape[1], shape[0], bias=False, dtype=_torch().float32)
        layer.weight.data.copy_(value)
        check("④⁸′ ⭐⭐ 融合 QKV 真装一层：布局 `per-row+convrot64` ✓、**保范**（每组 L2 不变 ✓）、"
              "正向旋转回去 = `q × s`（误差 10⁻⁷ 级 ✓）、与「不反旋转」差得远 ✓"
              "（漏了旋转 ⇒ 数值全错却不报错 ✓✗）；且文件级 `convrot=256` **必须判出问题** ✗",
              restored.ok and restored.converted[f"{stem}.weight"]["layout"] == "per-row+convrot64"
              and bool(_torch().isfinite(value).all()) and norm_ok
              and float((back - scaled).abs().max()) < 1e-5
              and float((value - scaled).abs().max()) > 0.1
              and not wrong.ok and any("不整除" in item for item in wrong.problems)
              and tuple(layer.weight.shape) == shape,
              (restored.problems, restored.converted.get(f"{stem}.weight"),
               norm_ok, float((back - scaled).abs().max()),
               float((value - scaled).abs().max()), wrong.problems[:1]))


# ══════════════════════════════════════════════════════════════════════════
# ①″ fp8 的三态：带标量 scale ✓ / **不带** scale（cast ✓）/ int8 不带 scale（仍拒 ✓✗）
# ══════════════════════════════════════════════════════════════════════════
def case_fp8() -> None:
    torch = _torch()
    torch.manual_seed(19)

    #: ① `fp8_scaled` 族的写法（2026-10-30 实盘实测 ✓）：``xxx.scale_weight`` 是**标量** ⇒ per-tensor
    #:    ✓；``xxx.scale_input`` 是**激活侧** ✓；``scaled_fp8`` 是**整份文件标记**（``shape=(0,)`` ✓）。
    q = torch.randn(6, 4).to(torch.float8_e4m3fn)
    state = {"w.weight": q, "w.scale_weight": torch.tensor(0.5),
             "w.scale_input": torch.tensor(1.0),
             "scaled_fp8": torch.empty(0, dtype=torch.float8_e4m3fn)}
    #: ⚠️ 用 **safetensors 头部那套 dtype 名**（`F8_E4M3` ✓）而不是 torch 名 ✗ —— 体检阶段拿到的
    #:    就是这一套 ✓；只按 `float8_e4m3fn` 判 ⇒ 盘上文件一个都不认 ✓✗
    meta = {name: ("F8_E4M3" if tensor.dtype == torch.float8_e4m3fn else "F32",
                   tuple(tensor.shape)) for name, tensor in state.items()}
    plan = quant_mod.plan_dequant(meta)
    check("⑥ ⭐ `fp8_scaled` 的 `weight.scale_weight`（**标量** ✓）⇒ 判成 `per-tensor` ✓ 且"
          "**只算 1 个权重** ✓（`scale_input` / `scaled_fp8` 不许被当成权重 ✗）",
          plan["weights"] == 1 and plan["paired"] == 1
          and plan["layouts"] == {"per-tensor": 1} and plan["supported"] is True
          and not plan["unpaired"] and not plan["unresolved"],
          plan)

    pre = quant_mod.precheck_dequant(meta, state)
    result = quant_mod.dequantize_state(state)
    check("⑥′ ⭐ 装载侧**同一口径** ✓：乘 0.5 ✓；`scale_input` / `scaled_fp8` 进 `companions`、"
          "**不进模型** ✓（留下来 ⇒ 每个 14B 文件多 400+ 个 `unexpected` ✓✗）",
          pre.converted["w.weight"]["layout"] == "per-tensor"
          and pre.converted["w.weight"]["scale"] == "w.scale_weight"
          and set(pre.companions) == {"w.scale_weight", "w.scale_input", "scaled_fp8"}
          and set(result.tensors) == {"w.weight"}
          and float((result.tensors["w.weight"] - q.float() * 0.5).abs().max()) < 1e-9,
          (sorted(pre.companions), sorted(result.tensors), result.problems))

    #: ② **不带 scale 的 fp8**（盘上 `seedvr2_ema_3b_fp8_e4m3fn` 301 个权重全是这种 ✓）——
    #:    fp8 的每个值都能被 bf16 **精确**表示 ✓ ⇒ `cast` 就是无损解码 ✓；旧判据报「缺配套 ⇒ 不装」✗
    #:    是**假拒绝**（拒绝不是安全 ✗）。
    plain_plan = quant_mod.plan_dequant({"w.weight": ("F8_E4M3", tuple(q.shape))})
    plain = quant_mod.dequantize_state({"w.weight": q})
    check("⑦ ⭐⭐ 纯 fp8（**没有** scale ✓）⇒ `cast` 布局 ✓ + 放行 ✓ + 数值**逐位**等于直接 cast ✓"
          "（`F8_E4M3` 与 `torch.float8_e4m3fn` **两套名字都要认** ✗ —— 只认 torch 名 ⇒ 一个都不认 ✓✗）",
          plain_plan["layouts"] == {"cast": 1} and plain_plan["supported"] is True
          and plain.converted["w.weight"]["layout"] == "cast"
          and plain.converted["w.weight"]["scale"] is None
          and bool(torch.equal(plain.tensors["w.weight"], q.float()))
          and quant_mod.is_cast_only("F8_E4M3")
          and quant_mod.is_cast_only(torch.float8_e4m3fn),
          plain_plan)

    #: ③ **守住口子** ✗：int8 没有 scale 是**真坏** ✓（会静默错一个数量级 ✓✗）⇒ 仍然拒绝 ✓
    broken = {"w.weight": torch.ones(4, 2, dtype=torch.int8)}
    broken_meta = {"w.weight": ("I8", (4, 2))}
    broken_plan = quant_mod.plan_dequant(broken_meta)
    broken_pre = quant_mod.precheck_dequant(broken_meta, broken)
    check("⑦′ ⭐ int8 **没有** scale ⇒ 仍然拒绝 ✓（`unpaired` ＋ `problems` ✓ —— 别让 fp8 这个口子"
          "把 int4/int8 也放了 ✗：那种静默错是数量级的 ✓✗）",
          broken_plan["unpairedCount"] == 1 and broken_plan["supported"] is False
          and not broken_pre.converted and bool(broken_pre.problems)
          and not quant_mod.is_cast_only("I8") and not quant_mod.is_cast_only("U8"),
          (broken_plan, broken_pre.problems))

    #: ④ `comfy_quant` 的 `format` 是**拼写**、不是族名 ✓ —— 必须归一 ✓✗
    #:    （不归一 ⇒ 同一族 fp8 一半叫 `fp8` ✓、一半叫 `float8_e4m3fn` ✓✗：
    #:    带 `*.comfy_quant` 的走 `supported_format` ✓，不带的一个都没有的只能走 dtype 兜底 ✓）
    check("⑦″ ⭐ `format` 拼写名归一到族名 ✓（`float8_e4m3fn` / `float8_e5m2` ⇒ 一律 `fp8` ✓；"
          "没别名的原样返回 ✓）—— 不归一 ⇒ `quantScheme` 同族两标签 ✓✗，"
          "下游按 `quantScheme == \"fp8\"` 筛会**漏掉一半** ✓✗",
          quant_mod.supported_format({"format": "float8_e4m3fn"})[0] == "fp8"
          and quant_mod.supported_format({"format": "float8_e5m2"})[0] == "fp8"
          and quant_mod.format_alias("Float8_E4M3FN") == "fp8"
          and quant_mod.format_alias("nvfp4") == "nvfp4"
          and quant_mod.format_alias(None) == "",
          [quant_mod.supported_format({"format": name})[0]
           for name in ("float8_e4m3fn", "float8_e5m2", "nvfp4")])


# ══════════════════════════════════════════════════════════════════════════
# ①‴ **反过来的那一半**：有 scale、没人要（融合 QKV 被拆成分片 scale 时正是这样 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_qkv_scales() -> None:
    """`plan_dequant` 只看「缺 scale 的权重」⇒ 「没人要的 scale」一个都不看 ✗。

    2026-10-30 盘上 23 个量化文件实测：孤儿 scale **一个都没有** ✓ ⇒ 原来那条
    「Qwen3.5-9B 的 108 个 unresolved 是融合 QKV 配不上分片 scale」的诊断是**错归因** ✗ ——
    真因是 convrot 分组被当成**文件级**（`visual` 输入维 1152 不整除 256 ✓✗）。
    但仍然要立这条判据 ✓：分片 scale 一旦真出现，旧判据会**静默报全绿** ✗。
    """
    torch = _torch()

    #: ① 融合 QKV ＋ 三份分片 scale（`q`/`k`/`v` 各自的 `weight_scale`）—— 三份全是孤儿 ✓，
    #:    而**融合那个权重自己**没有 scale ⇒ 缺配套 ⇒ `supported=False` ✓（这条才是真拦截 ✓）
    fused = {"attn.qkv.weight": torch.ones(12, 8, dtype=torch.int8),
             "attn.q.weight_scale": torch.ones(4, 1),
             "attn.k.weight_scale": torch.ones(4, 1),
             "attn.v.weight_scale": torch.ones(4, 1)}
    meta = {name: ("I8" if "weight_scale" not in name else "F32", tuple(t.shape))
            for name, t in fused.items()}
    plan = quant_mod.plan_dequant(meta)
    check("① ⭐ 融合 QKV ＋ 三份**分片** `weight_scale`（各自没有对应权重 ✓）⇒ 三份都报成"
          " `orphanScales` ✓ 且缺 scale 的融合权重进 `unpaired` ⇒ **拒绝** ✓"
          "（旧判据对这三份**一个都不看** ✗ ⇒ 只剩「全绿」✓✗）",
          plan["orphanScaleCount"] == 3
          and set(plan["orphanScales"]) == {"attn.q.weight_scale", "attn.k.weight_scale",
                                            "attn.v.weight_scale"}
          and plan["unpairedCount"] == 1 and plan["supported"] is False,
          plan)

    #: ② 报告归报告、**不许因此拒掉能跑的模型** ✗（拒绝不是安全 ✗）：融合权重**自己有** scale
    #:    ⇒ 能还原 ✓ ⇒ `supported=True`，孤儿只是「这段量化信息我们没用上」的提示 ✓
    ok_state = {"attn.qkv.weight": torch.ones(12, 8, dtype=torch.int8),
                "attn.qkv.weight_scale": torch.ones(12, 1),
                "attn.q.weight_scale": torch.ones(4, 1)}
    ok_meta = {name: ("I8" if name.endswith(".weight") else "F32", tuple(t.shape))
               for name, t in ok_state.items()}
    ok_plan = quant_mod.plan_dequant(ok_meta)
    check("② ⭐ 融合权重**自己有** scale ⇒ 能还原 ✓ `supported=True` ✓；多出来的那份孤儿 scale"
          "**照报**但**不拦** ✓（拦了就是把能跑的模型拒掉 ✗ —— 拒绝不是安全 ✗）",
          ok_plan["supported"] is True and ok_plan["paired"] == 1
          and ok_plan["orphanScales"] == ["attn.q.weight_scale"],
          ok_plan)

    #: ③ 激活侧 scale **不是**孤儿 ✗ —— ⚠️ 但**现行两张表其实不相撞** ✓（`SCALE_SUFFIXES` 全带点 ✓，
    #:    `w.input_scale` 的末 6 字符是 `_scale` ✓ ⇒ `endswith(".scale")` 为 False ✓）⇒ 只用现行表
    #:    测这条**测不出任何东西** ✓✗。所以临时加一个无点后缀 `"_scale"` 去逼它 ✗（这才是那条排除
    #:    真正防的坏法 ✓：以后往表里加无点写法 ⇒ 每个 fp8_scaled 层的激活侧 scale 全变孤儿 ✓✗）
    act = {"w.weight": torch.ones(4, 2, dtype=torch.int8), "w.weight_scale": torch.ones(4, 1),
           "w.input_scale": torch.ones(()), "w.scale_input": torch.ones(())}
    act_meta = {name: ("I8" if name.endswith(".weight") else "F32", tuple(t.shape))
                for name, t in act.items()}
    act_plan = quant_mod.plan_dequant(act_meta)
    saved = quant_mod.SCALE_SUFFIXES
    try:
        quant_mod.SCALE_SUFFIXES = saved + ("_scale",)
        guarded = quant_mod.orphan_scales({"w.weight": None, "w.input_scale": None})
    finally:
        quant_mod.SCALE_SUFFIXES = saved
    check("③ ⭐ **激活侧** scale（`input_scale` ✓ / `scale_input` ✓）**不算**孤儿 ✓；且把无点后缀"
          " `\"_scale\"` 临时塞进 `SCALE_SUFFIXES` 也**不**误报 ✓（不排除 ⇒ 每个 fp8_scaled 层都会"
          "多报一个孤儿 ✓✗；只用现行表测这条是**空转** ✗ —— 现行表不相撞 ✓）",
          act_plan["orphanScaleCount"] == 0 and act_plan["supported"] is True
          and guarded == [] and quant_mod.SCALE_SUFFIXES == saved,
          (act_plan, guarded, quant_mod.SCALE_SUFFIXES == saved))

    #: ④ 长后缀优先：`weight_scale_inv` 必须剥成 `x` ✓（剥成 `x.weight_scale` ⇒ 谁也配不上 ✓✗）
    check("④ ⭐ `scale_stem` 按**长后缀优先**剥 ✓（`weight_scale_inv` ⇒ `x` ✓；`weight_scale` ⇒ `x` ✓；"
          "普通键 ⇒ `None` ✓）",
          quant_mod.scale_stem("a.b.weight_scale_inv") == "a.b"
          and quant_mod.scale_stem("a.b.weight_scale") == "a.b"
          and quant_mod.scale_stem("a.b.scale_weight") == "a.b"
          and quant_mod.scale_stem("a.b.weight") is None
          and quant_mod.orphan_scales({"a.b.weight_scale_inv": None, "a.b.weight": None}) == [],
          [quant_mod.scale_stem(name) for name in
           ("a.b.weight_scale_inv", "a.b.weight_scale", "a.b.scale_weight", "a.b.weight")])

    #: ⑤ ⚠️ **真文件打脸**（2026-10-30 ✓）：第一版只按「有没有同名 `.weight`」判 ⇒
    #:    `krea2_turbo_int8_convrot` 上立刻报 **138 个**孤儿 ✓✗（全是 RMSNorm 的 `prenorm.scale` /
    #:    `attn.qknorm.qnorm.scale` / `txtmlp.0.scale` ✓ —— 它们**自己就是权重** ✓）
    check("⑤ ⭐ 归一化层的 `.scale`（`prenorm.scale` / `qknorm.qnorm.scale` / `txtmlp.0.scale` ✓）"
          "**不是**孤儿 ✓（自己就是权重 ✓，没有更低一层的 `.weight` ✓）；而名字**自带 `weight`** 的"
          " `attn.q.weight_scale` 没有对应权重 ⇒ **必须**报 ✓（融合 QKV 留分片 scale 就是这形态 ✓）—— "
          "第一版不分这两类 ⇒ 真模型上 138 个误报 ✓✗",
          quant_mod.orphan_scales({"blocks.0.prenorm.scale": None,
                                   "blocks.0.attn.qknorm.qnorm.scale": None,
                                   "txtmlp.0.scale": None}) == []
          and quant_mod.orphan_scales({"attn.q.weight_scale": None}) == ["attn.q.weight_scale"]
          and quant_mod.orphan_scales({"w.scale": None, "w.weight": None}) == [],
          [quant_mod.orphan_scales({"attn.q.weight_scale": None}),
           quant_mod.orphan_scales({"blocks.0.prenorm.scale": None})])


# ══════════════════════════════════════════════════════════════════════════
# ② 判不出来 ⇒ **拒绝**（宁可报错也不按猜的布局算 ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_refusals() -> None:
    torch = _torch()
    fp8 = torch.ones(8, 4).to(torch.float8_e4m3fn)

    block = quant_mod.dequantize_state({"w.weight": fp8, "w.scale": torch.ones(2, 2)})
    check("④ 块量化（scale 形状 (2,2) 与权重 (8,4) 对不上 ✓）⇒ **拒绝** ✓ 且说清是块量化 ✓"
          "（group size 在随附 json 里 ✓ 本仓不猜 ✗）",
          not block.ok and any("块量化" in item for item in block.problems), block.problems)

    group = quant_mod.dequantize_state({"w.weight": fp8, "w.qzeros": torch.zeros(2, 2, dtype=torch.int8)})
    check("⑤ 分组量化（带 `qzeros` ✓）⇒ **拒绝** ✓（组大小同样在 json 里 ✓）",
          not group.ok and any("分组量化" in item for item in group.problems), group.problems)

    #: ⚠️ 这一条**原来拿 fp8 当样本** ✓✗ —— 2026-10-30 起 fp8 没 scale 是**合法**的（`cast` ✓：
    #: 盘上 `seedvr2_ema_*` 整族一个 scale 都没有 ✓）⇒ 样本必须换成 **int8**（没有 scale 才是真坏 ✓：
    #: 值域整体丢了 ✓✗）。fp8 那一支的正例在 `case_fp8` 的 ⑦ ✓。
    orphan = quant_mod.dequantize_state({"w.weight": torch.ones(8, 4, dtype=torch.int8)})
    check("⑥ 低精度权重（**int8** ✓）**没有**配套 scale ⇒ **拒绝** ✓（不装 ✓ —— 装了只会算出错数 ✓✗）",
          not orphan.ok and any("找不到配套 scale" in item for item in orphan.problems),
          orphan.problems)

    zero = quant_mod.dequantize_state({"w.weight": fp8, "w.weight_scale_inv": torch.zeros(())})
    check("⑦ scale 里有 0 ⇒ **拒绝** ✓（除 0 出 inf ✓ —— 0 是**坏 scale** ✗，不是「合法的 1」✗）",
          not zero.ok and any("0" in item for item in zero.problems), zero.problems)

    plain = quant_mod.dequantize_state({"w.weight": torch.ones(4, 2), "b.bias": torch.ones(2)})
    check("⑧ bf16/fp32 的权重**原样返回** ✓（多一步 cast 只会白花时间 ✓）且 `converted` 为空 ✓",
          plain.ok and not plain.converted and "w.weight" in plain.tensors, plain.to_dict())

    companions = quant_mod.dequantize_state({"w.weight": fp8, "w.scale": torch.tensor(2.0)})
    check("⑨ 被吃掉的配套张量**不进模型** ✓（模型里没有 `.scale` 键 ✓ —— 留着只会变 unexpected ✓）",
          "w.scale" not in companions.tensors and companions.companions == ["w.scale"],
          (sorted(companions.tensors), companions.companions))

    check("⑪ ⭐ **两套 dtype 写法都认** ✓：torch 的 `float8_e4m3fn` ✓ 与 safetensors **头部**的 `F8_E4M3` ✓"
          " —— 只认一套 ⇒ **体检永远报「没有低精度权重」** ✓✗（装载在路上 ✓、体检在头部 ✓ 两者口径不同 ✓）",
          all(quant_mod.is_low_precision(item) for item in
              ("float8_e4m3fn", "torch.float8_e4m3fn", "F8_E4M3", "I8", "f8_e5m2", "U8"))
          and not quant_mod.is_low_precision("BF16") and not quant_mod.is_low_precision("F32"),
          [quant_mod.is_low_precision(item) for item in ("F8_E4M3", "torch.float8_e4m3fn", "BF16")])

    # ⭐ 体检（`plan_dequant` ✓ 只看形状）与装载（`dequantize_state` ✓ 真算）**必须同一口径** ✗
    shapes = {"w.weight": ("F8_E4M3", (8, 8)), "w.scale": ("F32", (2, 2))}
    plan = quant_mod.plan_dequant(shapes)
    verdict = quant_mod.dequantize_state({"w.weight": fp8.reshape(8, 8) if fp8.shape == (8, 8)
                                          else torch.ones(8, 8).to(torch.float8_e4m3fn),
                                          "w.scale": torch.ones(2, 2)})
    check("⑫ ⭐⭐ 体检与装载**同一口径** ✓：块量化在两边都被判成「不行」✗"
          "（不然会出现「体检说能装、装载时被拒」✓✗ —— 那种矛盾最费时间 ✓）",
          plan["unresolvedCount"] == 1 and plan["supported"] is False and verdict.ok is False,
          (plan, verdict.problems))

    report = quant_mod.quant_report({"w.weight": fp8, "w.scale": torch.tensor(2.0),
                                     "v.weight": fp8})
    check("⑩ `quant_report` **只看不装** ✓：报低精度数量 / 已配套数 / 没配套的名字 ✓ + 原 dtype ✓",
          report["lowPrecision"] == 2 and report["withScale"] == 1
          and report["unpaired"] == ["v"] and report["dtypes"] == ["float8_e4m3fn"], report)


# ══════════════════════════════════════════════════════════════════════════
# ③ 接进装载：fp8 检查点能真装 ✓ / 判不了就**中止** ✓
# ══════════════════════════════════════════════════════════════════════════
def case_load(root: Path) -> None:
    torch = _torch()
    from safetensors.torch import save_file  # noqa: PLC0415

    from app.services.engine import dit as dit_mod  # noqa: PLC0415

    config = dit_mod.DiTConfig(hidden=8, depth=1, heads=2, patch_size=(1, 2, 2),
                               in_channels=2, text_dim=8)
    torch.manual_seed(3)
    model = dit_mod.build_dit(config)
    full = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}

    # 把其中**一个**权重做成 fp8 + per-row scale ✓（其余保持 bf16 ✓ —— 真实检查点也常是混合的 ✓）
    target = "blocks.0.mlp.0.weight"
    weight = full[target].to(torch.float32)
    scale = torch.full((weight.shape[0],), 0.01)
    full[target] = (weight / scale.reshape(-1, 1)).to(torch.float8_e4m3fn)
    full[f"{target}_scale"] = scale
    fp8_path = root / "fp8.safetensors"
    save_file(full, str(fp8_path), metadata={})

    fresh = dit_mod.build_dit(config)
    with torch.no_grad():
        fresh.load_state_dict({key: tensor.clone() for key, tensor in model.state_dict().items()})
    report = weights_mod.load_module_weights(fresh, fp8_path)
    check("⑪ ⭐⭐ **fp8 权重真的装进去了** ✓ 且报告里有**反量化明细** ✓"
          "（布局 / 来源 scale / 原 dtype ✓ —— 「装上了」与「按对的布局还原」是两件事 ✓）",
          report.complete and report.dequant.get("count") == 1
          and report.dequant["converted"][target]["layout"] == "per-row"
          and report.dequant["converted"][target]["from"] == "torch.float8_e4m3fn",
          report.to_dict()["dequant"])

    # ⭐ 数值对得上：与**没被量化过**的原模型比，误差应当只在 fp8 精度内 ✓
    with torch.no_grad():
        latent = torch.randn(1, config.in_channels, 2, 8, 8)
        before = model(latent, 0.5, None)
        after = fresh(latent, 0.5, None)
    check("⑫ ⭐⭐ fp8 反量化后**前向数值对得上** ✓（与未量化的原模型最大差 < 0.05 ✓ —— "
          "**静默 cast**（不乘 scale ✓）会差到 10⁰–10² 量级 ✓✗）",
          float((after - before).abs().max()) < 0.05, float((after - before).abs().max()))

    bad = dict(full)
    bad[f"{target}_scale"] = torch.ones(3, 3)                 # ✗ 形状对不上任何布局
    bad_path = root / "bad.safetensors"
    save_file(bad, str(bad_path), metadata={})
    victim = dit_mod.build_dit(config)
    with torch.no_grad():
        victim.load_state_dict({key: tensor.clone() for key, tensor in model.state_dict().items()})
        untouched = victim(latent, 0.5, None).clone()
    bad_report = weights_mod.load_module_weights(victim, bad_path)
    check("⑬ ⭐ 反量化判不出来 ⇒ **中止装载** ✓（`error` 说清 + `complete=False` ✓）且"
          "**目标模块没被污染** ✓（半装状态最难查 ✗）",
          bad_report.complete is False and "中止装载" in bad_report.error
          and bool(torch.allclose(victim(latent, 0.5, None), untouched)),
          bad_report.error[:120])

    plain_path = root / "plain.safetensors"
    save_file({key: tensor.detach().clone() for key, tensor in model.state_dict().items()},
              str(plain_path), metadata={})
    plain_report = weights_mod.load_module_weights(dit_mod.build_dit(config), plain_path)
    check("⑭ 纯 bf16 检查点**不受影响** ✓（`dequant.count == 0` ✓ 且仍装得上 ✓ —— "
          "新逻辑不许给老路径添乱 ✗）",
          plain_report.complete and plain_report.dequant.get("count", 0) == 0,
          plain_report.to_dict()["dequant"])


def main() -> int:
    skips = _skips()
    if skips:
        for reason in skips:
            check(f"SKIP: {reason}", True, reason)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_layouts()
            case_convrot()
            case_fp8()
            case_qkv_scales()
            case_refusals()
            case_load(root)
            case_real_files()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    #: ⚠️ SKIP **单独一行**打 ✓✗ —— 与 PASS 混在一起时，「盘上没有真文件」会被读成「判据过了」✓✗
    for item in _SKIPPED:
        print(f"SKIP  {item}")
    print(f"\nSUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f", {len(_SKIPPED)} skipped" if _SKIPPED else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
