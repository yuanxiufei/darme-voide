r"""自研引擎 · **int8 权重常驻**（前向逐层反量化 ✓）—— 「22.49 GiB 显存装 19.53 GiB 权重」的唯一路径 ✓。

## 为什么单独一个自检（这一条决定「能不能起来 ✓」）

主 DiT 的 int8 权重是 **17.94 GiB** ✓，整份反量化成 bf16 = **35.9 GiB** ✗ ⇒ A5000 可用显存
22.49 GiB **连装都装不进去** ✗（`loader.residency_plan` 早就点过 ✓）。所以走「int8 原样常驻 +
前向时逐层反量化」✓（:class:`h3_form.Int8Linear` ✓）。

这条路上有**四个**静默失败面 ✓✗，必须逐个钉住：

1. **旋转搬边**：convrot 的旋转从权重侧搬到激活侧 ✓ —— 搬错（漏搬 / 只搬一半）不报错 ✗，
   症状是「权重形状全对、出片全是噪声」✗ ⇒ 必须与「权重侧反旋转」的参考算法**数值对齐** ✓；
2. **装载被 cast**：`copy_` 把 bf16 静默截断进 int8 参数 ⇒ 权重全 0 ✗（不报错 ✗）；
3. **scale 被降精度**：`weight_scale` 是 per-row 尺度 ✓，降成 bf16 丢 0.4% ✗ —— 比 int8
   本身的量化误差还大 ✗；
4. **元数据被当权重**：`*.comfy_quant`（55~72 字节 json ✓）一个模板 400+ 个 ✗ ⇒ 落进
   `unexpected` 会把真正多出来的键淹掉 ✗。

运行::

    ./.venv/Scripts/python.exe tests/engine_int8_resident_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from app.services.engine import quant as quant_mod  # noqa: E402
from app.services.engine import weights as weights_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, ok: bool, detail: object = None) -> None:
    _RESULTS.append((name, bool(ok), detail))


def _torch() -> Any:
    import torch  # noqa: PLC0415

    return torch


def _skips() -> list[str]:
    try:
        _torch()
        from safetensors.torch import save_file  # noqa: F401, PLC0415
    except ImportError:
        return ["未装 torch / safetensors ⇒ int8 常驻自检跳过（它要造真 safetensors ✓）"]
    return []


def _h3() -> Any:
    from app.services.engine import h3_form  # noqa: PLC0415

    return h3_form


def _quant_meta(group: int = 256) -> Any:
    import json  # noqa: PLC0415

    torch = _torch()
    text = json.dumps({"format": "int8_tensorwise", "convrot": True,
                       "convrot_groupsize": group}, separators=(",", ":"))
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)


def _save(root: Path, name: str, tensors: dict[str, Any]) -> Path:
    from safetensors.torch import save_file  # noqa: PLC0415

    path = root / name
    save_file({key: tensor.detach().clone().contiguous() for key, tensor in tensors.items()},
              str(path), metadata={})
    return path


def _fill(layer: Any, *, seed: int) -> None:
    """给 int8 层填**真实感**的权重 ✓（int8 值 ✓ + fp32 per-row 尺度 ✓ + bias ✓）。"""
    torch = _torch()
    gen = torch.Generator().manual_seed(seed)
    layer.weight.copy_(torch.randint(-127, 127, tuple(layer.weight.shape), dtype=torch.int8,
                                     generator=gen))
    layer.weight_scale.copy_(torch.rand(tuple(layer.weight_scale.shape), dtype=torch.float32,
                                        generator=gen) * 0.01 + 0.001)
    if layer.bias is not None:
        layer.bias.data = torch.randn(tuple(layer.bias.shape), dtype=layer.bias.dtype,
                                      generator=gen)


# ══════════════════════════════════════════════════════════════════════════
# ① 旋转搬边：激活侧旋转 + 旋转域权重 == 权重侧反旋转 ✓（**数值对齐** ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_rotation_side() -> None:
    torch = _torch()
    quant_meta = quant_mod
    h3 = _h3()
    group = 256
    layer = h3.Int8Linear(2 * group, 64, bias=True, scale_shape=(64,), convrot=group,
                          dtype=torch.float32)
    _fill(layer, seed=101)
    x = torch.randn(5, 2 * group, dtype=torch.float32)

    got = layer(x)                                     # 激活侧旋转 ✓（本仓实现 ✓）

    # 参考：**权重侧**反旋转 ✓（教科书做法 ✓ —— 慢 15 倍但数学正确 ✓，正好当尺子 ✓）
    w_rot = layer.weight.to(torch.float32) * layer.weight_scale.reshape(-1, 1)
    h = quant_meta.build_hadamard(group)
    w_true = quant_meta.rotate_weight(w_rot, h, group)
    want = torch.nn.functional.linear(x, w_true, layer.bias)

    scale = float(want.abs().max()) or 1.0
    gap = float((got - want).abs().max()) / scale

    # ⚠️ **反向**：漏搬旋转（直接用未旋转的 x）⇒ 必须差得远 ✓（否则上一条判据是恒真的 ✗）
    wrong = torch.nn.functional.linear(x, w_rot, layer.bias)
    wrong_gap = float((wrong - want).abs().max()) / scale
    check("① ⭐ convrot 旋转搬到**激活侧** ✓ 与「权重侧反旋转」数值一致 ✓"
          "（相对误差 < 1e-5 ✓）；反向：漏搬旋转 ⇒ 差得远 ✓（判据不是恒真 ✓）",
          gap < 1e-5 and wrong_gap > 1e-2, (gap, wrong_gap))


# ══════════════════════════════════════════════════════════════════════════
# ② upgrade：nn.Linear → Int8Linear ✓（形状/偏置/精度都对 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_upgrade() -> None:
    torch = _torch()
    from torch import nn  # noqa: PLC0415

    h3 = _h3()

    class Holder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Module()])
            self.blocks[0].fc = nn.Linear(2 * 256, 64, bias=True)

    holder = Holder().to(torch.bfloat16)
    swapped = h3.upgrade_linear_to_int8(holder, {"blocks.0.fc": (64,)}, convrot=256)
    layer = holder.blocks[0].fc
    _fill(layer, seed=202)

    x = torch.randn(4, 2 * 256, dtype=torch.bfloat16)
    out = layer(x)
    # 参考走**另一条**路 ✓：手工 bf16 乘尺度 ✓（实现里先 fp32 乘再降 bf16 ✓）⇒ 两者只应有
    # bf16 尾数级差异（0.4% 量级 ✓）；⚠️ 容差放到 5% 仍能抓住「漏旋转」（那是 100% 量级 ✗）
    w = layer.weight.to(torch.bfloat16) * layer.weight_scale.reshape(-1, 1).to(torch.bfloat16)
    x_rot = quant_mod.rotate_activations(x, 256)
    ref = torch.nn.functional.linear(x_rot, w, layer.bias)
    scale = float(ref.abs().max()) or 1.0
    gap = float((out - ref).abs().max()) / scale
    wrong = torch.nn.functional.linear(x, w, layer.bias)          # 漏搬旋转 ✗
    wrong_gap = float((wrong - ref).abs().max()) / scale
    check("② upgrade 换上去的层 ✓：参数是 int8 ✓、scale 是 **fp32** ✓、前向与**另一条**参考路径"
          "一致 ✓（bf16 尾数级 ✓，且漏旋转会被抓住 ✓ —— 判据有区分度 ✓）",
          swapped == ["blocks.0.fc"]
          and isinstance(layer, h3.Int8Linear)
          and layer.weight.dtype == torch.int8
          and layer.weight_scale.dtype == torch.float32
          and gap < 5e-2 and wrong_gap > 1e-1,
          (swapped, layer.weight.dtype, layer.weight_scale.dtype, gap, wrong_gap))


# ══════════════════════════════════════════════════════════════════════════
# ③ 装载：int8 **原样**进参数 ✓、`comfy_quant` 被当元数据吃掉 ✓、scale 不被降 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_resident_load(root: Path) -> None:
    torch = _torch()
    from torch import nn  # noqa: PLC0415

    h3 = _h3()

    class Holder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = h3.Int8Linear(2 * 256, 64, bias=True, scale_shape=(64,), convrot=256,
                                    dtype=torch.bfloat16)

    module = Holder()
    _fill(module.fc, seed=303)
    want_weight = module.fc.weight.detach().clone()
    want_scale = module.fc.weight_scale.detach().clone()

    tensors = {
        "fc.weight": want_weight,
        "fc.weight_scale": want_scale,
        "fc.bias": module.fc.bias.detach().clone(),
        "fc.weight.comfy_quant": _quant_meta(256),
    }
    path = _save(root, "resident.safetensors", tensors)
    report = weights_mod.load_module_weights(module, path)

    check("③ ⭐ 装载 **int8 常驻** ✓：报告 int8Resident=1 ✓、`comfy_quant` 被吃掉 ✓、"
          "`unexpected` 空 ✓、值逐位相等 ✓（**没被反量化**✗ 也没被 cast ✗）",
          report.complete
          and report.int8Resident == 1
          and report.consumedQuantMeta == 1
          and not report.unexpected
          and module.fc.weight.dtype == torch.int8
          and bool(torch.equal(module.fc.weight, want_weight))
          and module.fc.weight_scale.dtype == torch.float32
          and bool(torch.equal(module.fc.weight_scale, want_scale)),
          (report.to_dict(), module.fc.weight.dtype, module.fc.weight_scale.dtype))

    dequant = report.dequant or {}
    check("③′ 报告**不自相矛盾** ✓：int8 常驻的键**不出现**在「反量化明细」里 ✓ "
          "（否则同一份报告会说「反量化了 1 个」又说「1 个原样常驻」✗✗）",
          dequant.get("resident") == 1 and "fc.weight" not in (dequant.get("converted") or {}),
          dequant)


# ══════════════════════════════════════════════════════════════════════════
# ④ 反向：文件里不是 int8 ⇒ **拒绝**写进 int8 参数 ✓（不静默截断 ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_refuse_cast(root: Path) -> None:
    torch = _torch()
    from torch import nn  # noqa: PLC0415

    h3 = _h3()

    class Holder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = h3.Int8Linear(2 * 256, 64, bias=True, scale_shape=(64,), convrot=None,
                                    dtype=torch.bfloat16)

    module = Holder()
    before = module.fc.weight.detach().clone()
    path = _save(root, "wrong-dtype.safetensors", {
        "fc.weight": torch.randn(64, 2 * 256, dtype=torch.bfloat16),
        "fc.weight_scale": torch.rand(64, dtype=torch.float32),
        "fc.bias": torch.randn(64, dtype=torch.bfloat16),
    })
    report = weights_mod.load_module_weights(module, path)

    check("④ ⭐ 文件里是 **bf16** 权重而参数是 int8 常驻 ⇒ **拒绝写入** ✓（`copy_` 会静默截断成 0 ✗✗）"
          "且参数**没被动过** ✓",
          bool(report.error) and "截断" in str(report.error)
          and bool(torch.equal(module.fc.weight, before)),
          (report.error, module.fc.weight.dtype))


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 驻 CPU 那一档（`int8_offload` ✓）：int8 **真留在 CPU** ✓、其它参数搬走 ✓、前向照样对 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_offload() -> None:
    """⚠️ 这一档是**唯一**能让 720×1280 跑起来的排法 ✓✗（17.94 GiB 全常驻后只剩 2.87 GiB ✓）。

    它有两个静默失败面 ✓✗：
    1. ``place_module`` 把 **int8 也搬上卡** ✗ ⇒ 报告说"驻 CPU"、显存照爆 ✓✗
       （⇒ 必须断言**权重真在 CPU** ✓，而不是只信返回值 ✓）；
    2. 权重在 CPU、激活在卡，``forward`` 忘了搬 ✗ ⇒ 不报"设备不一致"而是**算错** ✓✗
       （⇒ 必须与"全在卡上"的结果**数值对齐** ✓）。
    """
    torch = _torch()
    from torch import nn  # noqa: PLC0415

    h3 = _h3()

    class Holder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = h3.Int8Linear(2 * 256, 64, bias=True, scale_shape=(64,), convrot=256,
                                    dtype=torch.bfloat16)
            self.norm = nn.LayerNorm(16).to(torch.bfloat16)   # 非 int8：它**必须**被搬走 ✓

    target = "cuda" if torch.cuda.is_available() else "cpu"
    module = Holder()
    _fill(module.fc, seed=505)
    x = torch.randn(3, 2 * 256, dtype=torch.bfloat16)
    ref = module.fc(x).detach().clone()

    report = h3.place_module(module, target, park_int8_on="cpu")
    int8_bytes = module.fc.weight.numel() * module.fc.weight.element_size()

    check("⑤ ⭐ `place_module(park_int8_on='cpu')`：int8 权重**真留在 CPU** ✓（不信返回值 ✓ 直接看 "
          "`weight.device` ✗）、其它参数**搬去** target ✓、报告字节数与 int8 实际字节**相等** ✓",
          report["parked"] == 1
          and report["parkedBytes"] == int8_bytes
          and module.fc.weight.device.type == "cpu"
          and report["moved"] >= 1
          and module.norm.weight.device.type == target.split(":")[0],
          (report, module.fc.weight.device.type, module.norm.weight.device.type))

    if target == "cpu":
        check("SKIP: 本机只有 CPU ⇒ 「激活在卡、权重在 CPU」这一路**无法验证**（不是「通过」✗）",
              True, "cpu-only")
        return

    # ⚠️ 激活在**卡**上、权重在 **CPU** ⇒ 本层现搬 ✓（这才是 offload 的真实形状 ✓）
    got = module.fc(x.to(target)).to("cpu")
    scale = float(ref.abs().max()) or 1.0
    gap = float((got - ref).abs().max()) / scale
    check("⑤′ 跨设备前向 ✓：权重在 CPU、激活在卡 ⇒ 输出与「全在卡上」**数值一致** ✓"
          "（bf16 尾数级 ✓；忘了搬权重会直接炸 / 搬错会差一个量级 ✗）",
          gap < 2e-2, (gap, tuple(got.shape)))


# ══════════════════════════════════════════════════════════════════════════
# ⑦ 峰值必须是「单层」✓ 不随层数线性涨 ✗ —— offload 的全部价值就在这一条 ✓✗
# ══════════════════════════════════════════════════════════════════════════
def case_peak() -> None:
    """⚠️ offload 唯一的意义 = **同时在显存里的只有单层** ✓✗（不是整份 17.94 GiB ✓）。

    这条假设有两种最容易被写坏的形状 ✓✗，而**所有单层数值判据都会照样通过** ✗：

    1. ``forward`` 里先把**全部层**搬上来再算（或把搬上来的副本缓存在 ``self`` 上）✗
       ⇒ 显存照样吃满整份 int8 + 反量化 ✓✗；
    2. 反量化出的 bf16 副本被某处长期持有（缓存 / ``self._last`` / 日志）✗
       ⇒ **层数越多涨越多** ✓✗。

    所以这里**不看数值**，只看 ``max_memory_allocated``（**活跃**峰值 ✓ —— 缓存分配器里
    的空块不算 ✗），并拿「同一结构、int8 全搬上卡」那份当**对照尺子** ✓：
    两份的峰值差必须≈**整份 int8** ✓（差得出来才证明 offload 真把那 17.94 GiB 省下来了 ✓）。
    """
    torch = _torch()
    from torch import nn  # noqa: PLC0415

    if not torch.cuda.is_available():
        check("SKIP: 本机没有 CUDA ⇒ 「多层 offload 的峰值」这一条**无法验证**（不是「通过」✗）",
              True, "no-cuda")
        return

    h3 = _h3()

    # ⚠️ **单层尺寸不变、只变层数** ✓✗ —— 这是本组唯一能分辨「累积 vs 单层」的办法 ✓：
    #    单层的瞬时峰值与层数**无关** ✓（同一层、同一中间张量 ✓），
    #    而累积型的写法会随层数**线性**涨 ✓✗。⚠️ 不能用「峰值 < 某个绝对阈值」✗：
    #    实测单层就是 **152 MiB**（反量化内部有 3~4 个中间张量 ✓）—— 六层累积才 192 MiB，
    #    两者**贴太近** ⇒ 那种阈值既可能误杀 ✓ 也可能放过累积 ✗（2026-10-29 本组第一版就栽在这 ✓）。
    dim = 4096
    group = 256
    few, many = 4, 12

    class Stack(nn.Module):
        def __init__(self, count: int) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([
                h3.Int8Linear(dim, dim, bias=False, scale_shape=(dim,), convrot=group,
                              dtype=torch.bfloat16) for _ in range(count)])

        def forward(self, x: Any) -> Any:
            for block in self.blocks:
                x = block(x)
            return x

    def _make(count: int) -> Any:
        stack = Stack(count)
        for index, block in enumerate(stack.blocks):
            _fill(block, seed=700 + index)      # 同下标权重逐字节相同 ✓（对照才成立 ✓）
        return stack

    def _forward_peak(module: Any) -> int:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(2, dim, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            out = module(x)
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
        del x, out
        return peak

    module = _make(many)
    one_int8 = module.blocks[0].weight.numel() * module.blocks[0].weight.element_size()
    one_bf16 = one_int8 * 2                     # int8 → bf16 正好两倍 ✓

    report = h3.place_module(module, "cuda", park_int8_on="cpu")
    check("⑦ ⭐ 十二层 int8 **全在 CPU** ✓、`parkedBytes` == 十二层之和 ✓、`applied` 为真 ✓、"
          "scale（float32 buffer）**搬去卡上** ✓"
          "（⚠️ 只断言\"报告说驻 CPU\" 不够 ✗ —— 必须 `weight.device` 本身在 CPU ✓）",
          report["applied"] is True and report["parked"] == many
          and report["parkedBytes"] == one_int8 * many
          and all(block.weight.device.type == "cpu" for block in module.blocks)
          and all(block.weight_scale.device.type == "cuda" for block in module.blocks),
          (report, one_int8, one_int8 * many))

    peak_many = _forward_peak(module)
    torch.cuda.empty_cache()

    small = _make(few)
    h3.place_module(small, "cuda", park_int8_on="cpu")
    peak_few = _forward_peak(small)
    del small
    torch.cuda.empty_cache()

    check("⑦′ ⭐⭐ **峰值不随层数增长** ✓（4 层 vs 12 层：多出的 8 层若被累积 ⇒ 至少多 256 MiB ✗；"
          "实测差值应接近 0 ✓）—— 这是 offload 唯一的价值 ✓✗，而累积型写法的**数值全对** ✗",
          peak_many - peak_few < 2 * one_bf16,
          {"fewMiB": round(peak_few / 2**20, 1), "manyMiB": round(peak_many / 2**20, 1),
           "deltaMiB": round((peak_many - peak_few) / 2**20, 1),
           "oneLayerMiB": round(one_bf16 / 2**20, 1)})

    resident = _make(many)
    h3.place_module(resident, "cuda", park_int8_on=None)     # 同结构、int8 全搬上卡 ✓
    peak_resident = _forward_peak(resident)
    del resident
    torch.cuda.empty_cache()

    check("⑦″ ⭐⭐ **对照尺子** ✓：int8 全搬上卡那份的峰值必须比 offload 那份**高出≈整份 int8** ✓"
          "（高不出来 ⇒ offload 根本没省到显存 ✓✗；这个差值就是 17.94 GiB 那件事的小样 ✓）",
          peak_resident - peak_many > 0.8 * one_int8 * many,
          {"residentMiB": round(peak_resident / 2**20, 1),
           "offloadMiB": round(peak_many / 2**20, 1),
           "deltaMiB": round((peak_resident - peak_many) / 2**20, 1),
           "expectMiB": round(one_int8 * many / 2**20, 1)})

    check("⑦‴ 跑完**权重仍在 CPU** ✓（`forward` 里若写成 `self.weight.data = …` 就永久驻卡了 ✗ ——"
          "第一次跑没问题 ✓、显存账却已经变 ✗✗）",
          all(block.weight.device.type == "cpu" for block in module.blocks))


# ══════════════════════════════════════════════════════════════════════════
# ⑧ int8 的账：层数 / int8 参数 / 文件侧常驻数 —— 三个数必须自洽 ✓✗
# ══════════════════════════════════════════════════════════════════════════
def case_int8_ledger() -> None:
    """⚠️ `upgrade_linear_to_int8` 的返回名单一直被装载侧**丢掉** ✗ ⇒ 「计划换掉 N 层、
    实际只换成 N-1 层」这种错**不会当场报错** ✗：那层按 bf16 落盘（显存 2 倍 ✓✗），
    几十分钟后在前向里以「CUDA out of memory」露面 ✗✗（指不到真原因 ✓）。
    报告里 `parked` 与 `int8Resident` 也各看各的 ✓✗ —— 两个数**没人对过账** ✓。
    """
    torch = _torch()
    from torch import nn  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    h3 = _h3()

    class Pair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = h3.Int8Linear(2 * 256, 64, bias=True, scale_shape=(64,), convrot=256,
                                   dtype=torch.bfloat16)
            self.b = h3.Int8Linear(2 * 256, 64, bias=True, scale_shape=(64,), convrot=256,
                                   dtype=torch.bfloat16)

    def _ledger(module: Any, resident: int) -> tuple[dict[str, Any], str | None]:
        return tb._int8_residency_problems(module, SimpleNamespace(int8Resident=resident))

    ledger, problem = _ledger(Pair(), 2)
    check("⑧ 两个常驻层 ＋ 文件侧 2 个 int8 权重 ⇒ **无问题** ✓，账目四项都在 ✓"
          "（这条检查本身得不误杀 ✓ 才有用 ✓）",
          problem is None
          and ledger == {"layers": 2, "int8Params": 2, "int8OnCpu": 2, "int8OnGpu": 0,
                         "int8Resident": 2},
          (ledger, problem))

    ledger, problem = _ledger(Pair(), 1)
    check("⑧′ ⭐ 文件侧只有 1 个 int8 权重装进 2 个常驻层 ⇒ **报错** ✓ 且说清「差的那层会按 bf16 落盘」✓"
          "（静默后果＝显存 2 倍、几十分钟后才 OOM ✗✗）",
          problem is not None and "1 个" in problem and "2 层" in problem and "bf16" in problem,
          (ledger, problem))

    broken = Pair()
    broken.a.weight.data = broken.a.weight.data.to(torch.bfloat16)   # 模拟「那层其实没换成常驻层」✗
    ledger, problem = _ledger(broken, 2)
    check("⑧″ 层数 2 而 int8 参数只有 1 个 ⇒ **报结构自相矛盾** ✓"
          "（这个形状 `resident` 那条抓不到 ✗ —— 两条判据各管一个方向 ✓）",
          problem is not None and "自相矛盾" in problem, (ledger, problem))

    ledger, problem = _ledger(Pair(), 3)
    check("⑧‴ **不误杀** ✓：文件侧 3 > 层数 2（将来出现非 Linear 的 int8 品类，如量化 embedding ✓）"
          "⇒ 不报错 ✓（只查漏装 `resident < layers` ✓，不查多 ✗）",
          problem is None, (ledger, problem))


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 开关：`int8_offload` **只认显式 True** ✓（字符串 "yes" / 1 不许悄悄打开 ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_switch() -> None:
    import inspect  # noqa: PLC0415

    from app.services.engine import bridge as bridge_mod  # noqa: PLC0415
    from app.services.engine import runtime as runtime_mod  # noqa: PLC0415

    sig = inspect.signature(runtime_mod.EngineRuntime.ensure_loaded)
    check("⑥ `ensure_loaded(int8_offload=…)` 在签名上 ✓ 且**默认关** ✓（默认开会让每台机器都慢一截 ✗）",
          "int8_offload" in sig.parameters
          and sig.parameters["int8_offload"].default is False,
          sig.parameters.get("int8_offload"))

    # ⚠️ `assembly_from_config` 读的是 ``config["settings"]`` ✓（不是 config 顶层 ✗；
    #    key 也会被 `normalize_settings` 归一成 snake_case ✓ —— `int8Offload` 与 `int8_offload` 都认 ✓）
    on = bridge_mod.assembly_from_config(
        {"settings": {"ditPath": "D:/x.safetensors", "int8Offload": True}}, stage="h3")
    loose = [bridge_mod.assembly_from_config(
        {"settings": {"ditPath": "D:/x.safetensors", "int8_offload": value}}, stage="h3")
        for value in ("yes", 1, "true", [1])]
    check("⑥′ 桥：给 ``True`` ⇒ 进装配 ✓；给 ``\"yes\"`` / ``1`` / ``\"true\"`` / ``[1]`` ⇒ **不进** ✓"
          "（显存账不许被「看着像真」的值改写 ✗）",
          on.get("int8_offload") is True
          and not any(a.get("int8_offload") for a in loose),
          ([a.get("int8_offload") for a in loose], on.get("int8_offload")))


def case_frozen() -> None:
    """⑨ 一族：**推理不许建图** ✓✗（2026-11-01 端到端实测 83.7 GiB 的那个根因 ✓）。

    这条与 int8 常驻是同一件事的两面 ✓：``Int8Linear.forward`` 每层现反量化的 bf16 权重是
    **局部变量** ✓ —— 正常情况下 `forward` 一返回就释放 ✓；一旦 autograd 建了图，它就被
    ``AddmmBackward`` 当 saved tensor 吊住 ✓✗（探针实测：378 次反量化里 28 个副本一直活着 ✓，
    累计 37.18 GiB ✓）。而建图的开关就是**骨架里 norm 权重的 `requires_grad`** ✓（默认 True ✓）。
    """
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    from app.services.engine import h3_form, torch_backend  # noqa: PLC0415

    #: ⚠️ 取积木走**模块级懒加载** ✓（`h3_form.RMSNorm` ✓ —— 模块 `__getattr__` 已登记 ✓），
    #:    别再套一层"先建 dict 再取键"✗（那样拿反了 `_build_torch_parts()` 的返回值 ✓ 会 `TypeError` ✓）。
    fresh = h3_form.RMSNorm(8)
    check("⑨ ⭐ 骨架的 norm 权重**默认取梯度** ✓（`nn.Parameter(torch.ones(dim))` 不写 False ✓ —— "
          "这就是整条采样在**建图**的根因 ✓，不是「某个模块写错」✗）",
          fresh.weight.requires_grad is True, fresh.weight.requires_grad)

    sample = torch.randn(2, 8)
    before = fresh(sample).requires_grad
    frozen = torch_backend._freeze_inference(fresh)
    after = fresh(sample).requires_grad
    check("⑨′ ⭐⭐ 冻**之前**前向带梯度 ✓（⇒ 每个中间激活都被钉住 ✓✗）、冻**之后**不带 ✓（图建不起来 ✓）"
          "—— 并且 `_freeze_inference` 报出冻了几个 ✓（1 ✓）",
          before is True and after is False and frozen == 1,
          (before, after, frozen))

    #: ⚠️ 已经是不取梯度的参数**不重复冻** ✓（`Int8Linear` 的 weight/bias 建时就写了 False ✓）——
    #: 计数要是把"本来就是 False"的也算进去，报告上就分不清"真冻了几个" ✓✗
    layer = h3_form.Int8Linear(8, 4)
    check("⑨″ `Int8Linear` 建时就不取梯度 ⇒ 再冻一次**计数为 0** ✓（不误计 ✓）",
          torch_backend._freeze_inference(layer) == 0)

    #: ⚠️ **挂完 LoRA 要再冻一次** ✓✗：LoRA 的 A/B 是挂载那一刻才建出来的 ✓、默认取梯度 ✓ ⇒
    #: 只冻底模 = LoRA 那几层照样把整张图吊起来 ✓（症状与"完全不冻"一模一样 ✓✗）。
    holder = nn.Sequential(nn.Linear(4, 4))
    torch_backend._freeze_inference(holder)
    holder[0].weight.requires_grad_(True)          # 模拟"挂完 LoRA 又多出来的取梯度参数" ✓
    again = torch_backend._freeze_inference(holder)
    check("⑨‴ 挂 LoRA 之后再冻一次 ⇒ **新参数也收掉** ✓ 且计数只报新冻的那一个 ✓",
          again == 1 and not any(p.requires_grad for p in holder.parameters()), again)

    check("⑨⁗ `_freeze_inference(None)` 返回 0 ✓（没模型可冻不算失败 ✓）",
          torch_backend._freeze_inference(None) == 0)


def main() -> int:
    skips = _skips()
    if skips:
        for reason in skips:
            check(f"SKIP: {reason}", True, reason)
    else:
        case_rotation_side()
        case_upgrade()
        case_offload()
        case_peak()
        case_int8_ledger()
        case_switch()
        case_frozen()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_resident_load(root)
            case_refuse_cast(root)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print(f"\nSUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
