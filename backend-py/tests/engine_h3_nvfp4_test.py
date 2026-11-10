r"""自研引擎 · **nvfp4 打包权重常驻**（`NvFp4Linear` ✓）—— 「14.61 GiB 装下 Qwen3-VL-32B」的那条路 ✓。

## 为什么单独一个自检（2026-11-09 ✓）

盘上 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`（**14.61 GiB** ✓）此前**没有任何挂载点** ✓✗：
`attach_h3_text` 只认 int8_convrot 那份 ✓，见到 nvfp4 直接报「一个 int8 权重都没有 ⇒ 拒装」✓
（而 25.44 GiB 的 int8 那份要分时换装 ✓）。整份反量化是 **≈64 GiB** ✗ ⇒ 22.49 GiB 的卡装不下 ✗。
唯一排法是「**打包字节原样常驻** ＋ **前向逐层反量化**」✓ ⇒ 驻留 **15.33 GiB**（实测卡上 15.35 ✓）。

## 五个静默失败面（全部「形状对、不报错、数值错」✗）

1. **半字节顺序**：元素 0 在**高半字节** ✓（`0x21` ⇒ `[1.0, 0.5]` ✓）；反了每列错配 ✓✗；
2. **块 scale 布局**：`weight_scale` 是 cuBLAS **128×4 swizzled** ✓，必须先 `from_blocked` 再乘 ✓
   —— 按行主序直接乘，形状全合法、数值全错 ✓✗（本文件 ② 用**手算数**钉死 ✓，第 ③ 条用参考实现对拍 ✓）；
3. **少一级尺度**：`weight_scale_2`（f32 标量 ✓）漏乘 ⇒ 整层整体偏一个常数 ✓✗；
4. **AWQ 折错侧**：`pre_quant_scale` 乘在**输入**上 ✓，等价折进权重**列** ✓；折进行 / 忘了折 ⇒
   形状全对、数值偏 ✓✗（本文件 ②c 的手算值 40.0 专门钉这一条 ✓）；
5. **装载时被"常驻"漏判**：`weights` 原先只认 `int8` ⇒ `uint8` 被打成"不能原样写入"✗；
   反过来把打包层放进 `nv_layers`（装载时就还原 ✓）⇒ 一装 62 GiB 且打包壳白留 ✓✗。

运行::

    ./.venv/Scripts/python.exe tests/engine_h3_nvfp4_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os  # noqa: E402

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from app.services.engine import h3_form  # noqa: E402
from app.services.engine import quant_nvfp4 as nv  # noqa: E402
from app.services.engine import torch_backend as tb  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPPED: list[str] = []

#: 真实文件（盘上未必有 ⇒ 找不到就**单列 SKIP** ✓ 不混进 PASS ✗）
REAL_NVFP4 = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\text_encoders"
                  r"\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")


def _torch() -> Any:
    """惰性取 torch ✓（顶层 `import torch` 会让 structure_test ㊁ 判红 ✗）。"""
    import torch  # noqa: PLC0415

    return torch


def check(name: str, ok: bool, detail: object = None) -> None:
    _RESULTS.append((name, bool(ok), detail))


def skip(name: str, why: str) -> None:
    _SKIPPED.append(f"{name} —— {why}")


def case_shell_shape() -> None:
    """① 壳的形状、拒绝规则、参数集合 ✓（全部"照文件"✓，不许自算 ✗）。"""
    torch = _torch()
    cls = h3_form.NvFp4Linear
    shell = cls(64, 8, bias=True, scale_shape=(128, 4), awq=True, dtype=torch.float32)
    check("①¹ weight 是 U8 且列数只有逻辑的一半",
          shell.weight.dtype == torch.uint8
          and tuple(shell.weight.shape) == (8, 32)
          and int(shell.weight.numel()) * 2 == 8 * 64,
          (str(shell.weight.dtype), tuple(shell.weight.shape)))
    check("①² 块 scale 照文件形状建（带 padding 也算 ✓）",
          tuple(shell.weight_scale.shape) == (128, 4)
          and shell.weight_scale.dtype == torch.float8_e4m3fn,
          (tuple(shell.weight_scale.shape), str(shell.weight_scale.dtype)))
    check("①³ 二级 scale 是 **0 维** f32（建成 [1] 会被形状闸口拦 ✗）",
          tuple(shell.weight_scale_2.shape) == () and shell.weight_scale_2.dtype == torch.float32,
          (tuple(shell.weight_scale_2.shape), str(shell.weight_scale_2.dtype)))
    check("①⁴ AWQ 的 pre_quant_scale 长 = in_features（不是 out ✗）",
          tuple(shell.pre_quant_scale.shape) == (64,)
          and shell.pre_quant_scale.dtype == torch.bfloat16,
          (tuple(shell.pre_quant_scale.shape), str(shell.pre_quant_scale.dtype)))
    plain = cls(64, 8, bias=False, scale_shape=(128, 4))
    check("①⁵ 没有 bias / 没有 AWQ 时是 None（不建空张量 ✗）",
          plain.bias is None and plain.pre_quant_scale is None,
          (plain.bias, plain.pre_quant_scale))
    raised = []
    for kwargs, why in (((62, 8), "奇数 in_features"), ((8, 8), "不是 16 的倍数")):
        try:
            cls(*kwargs, scale_shape=(128, 4))
        except ValueError:
            raised.append(why)
    check("①⁶ in_features 奇数 / 非 16 倍数都当场拒（不补齐、不猜 ✓）", len(raised) == 2, raised)


def case_forward_hand_math() -> None:
    """② **手算**数（不经 `quant_nvfp4` ✓）：0x21 ⇒ `[1.0, 0.5]` ✓ ⇒ 列和 24 / 48 / 40 ✓。"""
    torch = _torch()
    cls = h3_form.NvFp4Linear
    rows, cols = 4, 32

    def build(*, scale2: float = 1.0, awq: bool = False) -> Any:
        shell = cls(cols, rows, bias=False, scale_shape=(128, 4), awq=awq, dtype=torch.float32)
        shell.weight.data.fill_(0x21)                    # 高半字节 2 ⇒ 1.0 ✓、低半字节 1 ⇒ 0.5 ✓
        shell.weight_scale.data.fill_(1.0)               # F8 的 1.0 是精确值 ✓
        shell.weight_scale_2.data.fill_(scale2)
        if awq:
            pqs = torch.ones(cols, dtype=torch.bfloat16)
            pqs[0::2] = 2.0                              # 偶数列（值 1.0 那些）× 2 ✓
            shell.pre_quant_scale.data.copy_(pqs)
        return shell

    x = torch.ones((1, cols), dtype=torch.float32)
    y = build()(x)
    check("②¹ 半字节顺序：偶数位置是 1.0、奇数位置是 0.5 ⇒ 列和 24.0",
          bool(torch.equal(y, torch.full_like(y, 24.0))), y.tolist())
    y2 = build(scale2=2.0)(x)
    check("②² 二级 scale（f32 标量）确实乘进去了 ⇒ 48.0",
          bool(torch.equal(y2, torch.full_like(y2, 48.0))), y2.tolist())
    y3 = build(awq=True)(x)
    #: 输入 x=1 ⇒ Σ W·s = 16×1.0×2 + 16×0.5×1 = 40 ✓（折进**列**才对 ✓；折进行会得 16×2+16×0.5×2=48 ✗）
    check("②³ AWQ 折进权重**列**（不是行 ✗）⇒ 40.0", bool(torch.equal(y3, torch.full_like(y3, 40.0))),
          y3.tolist())


def case_forward_reference() -> None:
    """③ 与 `quant_nvfp4.dequantize` ＋ `fold_pre_quant_scale` 的**参考实现**对拍 ✓（随机权重 ✓）。"""
    torch = _torch()
    cls = h3_form.NvFp4Linear
    torch.manual_seed(20261109)
    rows, cols = 64, 64
    packed = torch.randint(0, 256, (rows, cols // 2), dtype=torch.int32).to(torch.uint8)
    scale = (torch.rand(128, 4, dtype=torch.float32) + 0.5).to(torch.float8_e4m3fn)
    scale2 = torch.tensor(0.0731, dtype=torch.float32)
    x = torch.randn(3, cols, dtype=torch.float32)
    reference = nv.dequantize(packed, scale, scale2, (rows, cols), dtype=torch.float32)
    want = torch.nn.functional.linear(x, reference)
    shell = cls(cols, rows, bias=False, scale_shape=(128, 4), dtype=torch.float32)
    shell.weight.data.copy_(packed)
    shell.weight_scale.data.copy_(scale)
    shell.weight_scale_2.data.copy_(scale2)
    got = shell(x)
    check("③¹ 无 AWQ：壳的前向 == 参考（1e-5）",
          bool(torch.allclose(got, want, atol=1e-5, rtol=1e-4)),
          float((got - want).abs().max()))
    pqs = torch.rand(cols, dtype=torch.float32).to(torch.bfloat16) + 0.5
    want2 = torch.nn.functional.linear(x, nv.fold_pre_quant_scale(reference, pqs))
    shell2 = cls(cols, rows, bias=False, scale_shape=(128, 4), awq=True, dtype=torch.float32)
    shell2.weight.data.copy_(packed)
    shell2.weight_scale.data.copy_(scale)
    shell2.weight_scale_2.data.copy_(scale2)
    shell2.pre_quant_scale.data.copy_(pqs)
    got2 = shell2(x)
    check("③² 有 AWQ：壳的前向 == 参考（折列 ✓，1e-5）",
          bool(torch.allclose(got2, want2, atol=1e-5, rtol=1e-4)),
          float((got2 - want2).abs().max()))


def case_upgrade() -> None:
    """④ `upgrade_linear_to_nvfp4`：只换点名的层 ✓、bias/训练态保留 ✓、非 Linear 当场拒 ✓。"""
    torch = _torch()
    nn = torch.nn
    module = nn.Sequential()
    module.add_module("a", nn.Linear(64, 8, bias=True))
    module.add_module("b", nn.Linear(64, 8, bias=False))
    module.add_module("conv", nn.Conv1d(4, 4, 3))
    swapped = h3_form.upgrade_linear_to_nvfp4(module, {"a": ((128, 4), False)})
    check("④¹ 只换点名的 1 层", swapped == ["a"] and isinstance(module.a, h3_form.NvFp4Linear),
          swapped)
    check("④² 非目标层原样是 nn.Linear（不连带换 ✗）",
          isinstance(module.b, nn.Linear) and type(module.b) is nn.Linear, type(module.b).__name__)
    check("④³ bias 有无跟随原层", module.a.bias is not None and module.b.bias is None,
          (module.a.bias is not None, module.b.bias is None))
    refused = ""
    try:
        h3_form.upgrade_linear_to_nvfp4(module, {"conv": ((128, 4), False)})
    except TypeError as error:
        refused = str(error)
    check("④⁴ 点名非 nn.Linear ⇒ TypeError（不静默跳过 ✗）", "不是 nn.Linear" in refused, refused)


def _entry(dtype: str, shape: tuple[int, ...]) -> Any:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    size = {"U8": 1, "I8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4, "F16": 2}[dtype]
    return SimpleNamespace(dtype=dtype, shape=shape, nbytes=numel * size, element_count=numel)


def case_plan() -> None:
    """⑤ `nvfp4_scale_plan`：三件配套齐全才点名 ✓；`I8` 嵌入不点 ✓；没自述就不猜 ✓。"""
    info = SimpleNamespace(tensors={
        "model.layers.0.mlp.up_proj.weight": _entry("U8", (8, 32)),
        "model.layers.0.mlp.up_proj.weight_scale": _entry("F8_E4M3", (128, 4)),
        "model.layers.0.mlp.up_proj.weight_scale_2": _entry("F32", ()),
        "model.layers.0.mlp.down_proj.weight": _entry("U8", (8, 32)),
        "model.layers.0.mlp.down_proj.weight_scale": _entry("F8_E4M3", (128, 4)),
        "model.layers.0.mlp.down_proj.weight_scale_2": _entry("F32", ()),
        "model.layers.0.mlp.down_proj.pre_quant_scale": _entry("BF16", (64,)),
        "model.embed_tokens.weight": _entry("I8", (100, 64)),
        "model.embed_tokens.weight_scale": _entry("F32", (100, 1)),
    })
    all_stems = {"model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj"}
    plan = tb.nvfp4_scale_plan(info, packed_stems=all_stems)
    check("⑤¹ 两层的 scale 形状照文件带出来，且 AWQ 只认 down_proj",
          plan == {"model.layers.0.mlp.up_proj": ((128, 4), False),
                   "model.layers.0.mlp.down_proj": ((128, 4), True)},
          plan)
    only_up = tb.nvfp4_scale_plan(info, packed_stems={"model.layers.0.mlp.up_proj"})
    check("⑤² 文件没自述的层不点名（不按 dtype 猜 ✗）", set(only_up) == {"model.layers.0.mlp.up_proj"},
          only_up)
    no_scale2 = SimpleNamespace(tensors=dict(info.tensors))
    no_scale2.tensors.pop("model.layers.0.mlp.up_proj.weight_scale_2")
    check("⑤³ 缺 weight_scale_2 ⇒ 不点名（让装载闸口报\"缺哪一件\"✓）",
          "model.layers.0.mlp.up_proj" not in tb.nvfp4_scale_plan(no_scale2, packed_stems=all_stems),
          tb.nvfp4_scale_plan(no_scale2, packed_stems=all_stems))
    stripped = tb.nvfp4_scale_plan(info, packed_stems=all_stems, strip_prefix="model.")
    check("⑤⁴ strip_prefix 按模块名剥（前缀不对一个都点不上 ✓）",
          set(stripped) == {"layers.0.mlp.up_proj", "layers.0.mlp.down_proj"}, set(stripped))
    check("⑤⁵ I8 嵌入（同文件里的另一个族 ✓）一个都不点",
          not any("embed_tokens" in key for key in plan), list(plan))


def case_resident_bytes() -> None:
    """⑥ 驻留账：打包层 **1 B/2参** ✓、配套按文件字节 ✓、int8 嵌入按 bf16 记 ✓（0.72 GiB 那个坑 ✓）。"""
    info = SimpleNamespace(tensors={
        # 打包层：8×64 逻辑 = 8×32 字节 U8 ✓ ⇒ 计 256 B（若 ×2 会成 512 ✗）
        "m.up.weight": _entry("U8", (8, 32)),
        "m.up.weight_scale": _entry("F8_E4M3", (128, 4)),
        "m.up.weight_scale_2": _entry("F32", ()),
        # int8 嵌入：100×64 I8 = 6400 B ✓ ⇒ 记 **12800 B** ✓（本仓没有 int8 嵌入壳 ✓）
        "embed_tokens.weight": _entry("I8", (100, 64)),
        "embed_tokens.weight_scale": _entry("F32", (100, 1)),
    })
    #: ⚠️ `specs` 的值是**文件自述的字典** ✓（`packed_weight_of` 读 `format`/`linear_dtype` ✓），
    #: 写字符串 ⇒ 打包层认不出来、按 int8 记 ✓✗（这个测试自己第一版就踩了 ✓）。
    specs = {"m.up": {"format": "nvfp4"},
             "embed_tokens": {"format": "int8_tensorwise"}}
    total, ledger = tb.resident_bytes_of(info, adapter="h3", specs=specs)
    want = 8 * 32 + 128 * 4 + 4 + (100 * 64 * 2) + (100 * 4)
    check("⑥¹ 总账 == 打包字节 + 配套字节 + 嵌入(bf16) + 嵌入 scale", total == want,
          (total, want))
    check("⑥² 打包层计 1 层、int8 嵌入单列一格（不混进 int8Resident ✗）",
          ledger.get("nvfp4Layers") == 1 and ledger.get("int8ResidentLayers") == 0
          and ledger.get("int8EmbedLayers") == 1, ledger)


def case_int8_bytes_of() -> None:
    """⑦ `_int8_bytes_of` 必须把 **uint8** 也算进去 ✓（否则钉 CPU 前不问内存 ✓✗）。"""
    torch = _torch()
    nn = torch.nn
    module = nn.Module()
    module.register_parameter("w8", nn.Parameter(torch.zeros(10, dtype=torch.int8),
                                                 requires_grad=False))
    module.register_parameter("w4", nn.Parameter(torch.zeros(30, dtype=torch.uint8),
                                                 requires_grad=False))
    module.register_parameter("w16", nn.Parameter(torch.zeros(100, dtype=torch.bfloat16),
                                                  requires_grad=False))
    check("⑦ int8 + uint8 都算、bf16 不算", tb._int8_bytes_of(module, torch) == 40,
          tb._int8_bytes_of(module, torch))


def case_infer_shape() -> None:
    """⑧ 形状反推：打包层**列数是逻辑宽度的一半** ✓ ⇒ 不 ×2 会把 hidden 读成一半 ✓✗。"""
    from app.services.engine import qwen3vl_te, qwen3vl_vision

    shapes = {
        "model.embed_tokens.weight": (100, 5120),
        "model.layers.0.self_attn.q_proj.weight": (8192, 2560),
        "model.layers.0.mlp.gate_proj.weight": (25600, 2560),
        "model.layers.0.self_attn.o_proj.weight": (5120, 8192),
    }
    packed = {"model.layers.0.self_attn.q_proj", "model.layers.0.mlp.gate_proj",
              "model.layers.0.self_attn.o_proj"}
    good = qwen3vl_te.infer_shape(shapes, packed_stems=packed)
    check("⑧¹ 认打包列数 ⇒ hidden 仍是 5120（heads 不会变 32 ✓）",
          good.get("hidden") == 5120, good.get("hidden"))
    bad = qwen3vl_te.infer_shape(shapes)
    check("⑧² 反证：不递 packed_stems ⇒ hidden 被读成 2560（这就是那个不报错的错 ✗）",
          bad.get("hidden") == 2560, bad.get("hidden"))
    vision_shapes = {"visual.merger.linear_fc1.weight": (4096, 5120)}
    vision = qwen3vl_vision.infer_shape(vision_shapes,
                                        packed_stems={"visual.merger.linear_fc1"})
    check("⑧³ 视觉塔 merger 也认打包列数（unit=2 ✓ 不退回常量 ✓）",
          vision.get("spatial_merge_size") == 2, vision.get("spatial_merge_size"))


def case_real_file() -> None:
    """⑨ 真实 nvfp4 文件头部：350 层打包 ✓、精算 15.33 GiB ✓、档位判成 nvfp4 ✓（20 秒全装那次见文档 ✓）。"""
    if not REAL_NVFP4.exists():
        skip("⑨ 真实 nvfp4 文件", f"盘上没有 {REAL_NVFP4}")
        return
    from app.services.engine import loader, safetensors as st

    info = st.inspect(REAL_NVFP4)
    specs = loader.quant_specs_of(REAL_NVFP4, info)
    packed = __import__("app.services.engine.quant", fromlist=["x"]).packed_weight_stems_of(specs)
    plan = tb.nvfp4_scale_plan(info, packed_stems=packed)
    check("⑨¹ 350 层打包、其中 100 层带 AWQ", len(plan) == 350
          and sum(1 for value in plan.values() if value[1]) == 100,
          (len(plan), sum(1 for value in plan.values() if value[1])))
    total, ledger = tb.resident_bytes_of(info, adapter="h3", specs=specs)
    gib = total / 2 ** 30
    check("⑨² 精算驻留 15.33 GiB（= 文件 14.61 + 嵌入按 bf16 的那 0.72 ✓）",
          15.2 <= gib <= 15.5,
          (round(gib, 3), ledger))
    need, why = tb.part_need_gib(REAL_NVFP4, fallback=25.44, adapter="h3", assume="int8")
    check("⑨³ 档位判成 nvfp4 且**不用表值**（25.44 是 int8 那份的账 ✗）",
          why.get("mode") == "nvfp4" and why.get("source") == "file" and 15.2 <= need <= 15.5,
          (round(need, 3), why.get("mode"), why.get("source")))


def case_place_module() -> None:
    """⑩ `place_module(park_int8_on="cpu")` 也要把 **uint8** 钉住 ✓（否则"名义驻 CPU"✓✗）。"""
    torch = _torch()
    if not torch.cuda.is_available():
        skip("⑩ place_module 钉 uint8", "没有 CUDA")
        return
    nn = torch.nn
    module = nn.Module()
    module.register_parameter("w8", nn.Parameter(torch.zeros(16, dtype=torch.int8),
                                                 requires_grad=False))
    module.register_parameter("w4", nn.Parameter(torch.zeros(16, dtype=torch.uint8),
                                                 requires_grad=False))
    module.register_parameter("w16", nn.Parameter(torch.zeros(16, dtype=torch.bfloat16),
                                                  requires_grad=False))
    report = h3_form.place_module(module, "cuda", park_int8_on="cpu")
    check("⑩¹ int8 与 uint8 都留在 CPU、bf16 上卡",
          module.w8.device.type == "cpu" and module.w4.device.type == "cpu"
          and module.w16.device.type == "cuda",
          (module.w8.device.type, module.w4.device.type, module.w16.device.type))
    check("⑩² 报告分得清两类（parkedNvfp4 ✓）",
          report.get("parked") == 2 and report.get("parkedNvfp4") == 1, report)


def case_shell_forward_parked() -> None:
    """⑪ 权重钉在 CPU 时前向**现搬** ✓（峰值是单层 ✓，不是整份浮点副本 ✗）。"""
    torch = _torch()
    if not torch.cuda.is_available():
        skip("⑪ 驻 CPU 前向", "没有 CUDA")
        return
    cls = h3_form.NvFp4Linear
    shell = cls(64, 8, bias=False, scale_shape=(128, 4), dtype=torch.bfloat16)
    shell.weight.data.fill_(0x21)
    shell.weight_scale.data.fill_(1.0)
    shell.weight_scale_2.data.fill_(1.0)
    report = h3_form.place_module(shell, "cuda", park_int8_on="cpu")
    y = shell(torch.ones((1, 64), dtype=torch.bfloat16, device="cuda"))
    check("⑪¹ 驻 CPU 也能算（64 列 ⇒ 32×1.0 + 32×0.5 = 48 ✓）",
          bool(torch.equal(y, torch.full_like(y, 48.0))) and report.get("parked") == 1,
          (y.tolist(), report.get("parked")))
    check("⑪² 前向搬完打包权重**留在** CPU（没有悄悄常驻 ✓）",
          shell.weight.device.type == "cpu", str(shell.weight.device))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    case_shell_shape()
    case_forward_hand_math()
    case_forward_reference()
    case_upgrade()
    case_plan()
    case_resident_bytes()
    case_int8_bytes_of()
    case_infer_shape()
    case_real_file()
    case_place_module()
    case_shell_forward_parked()

    failed = 0
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failed += 1
            print(f"      detail: {detail}")
    for line in _SKIPPED:
        print(f"SKIP  {line}")
    print(f"\nSUMMARY: {len(_RESULTS) - failed}/{len(_RESULTS)} passed")
    print(f"  （skip {len(_SKIPPED)}）")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
