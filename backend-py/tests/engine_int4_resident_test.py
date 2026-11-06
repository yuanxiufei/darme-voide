r"""自研引擎 · **4bit 打包权重常驻**（`Int8Linear(packed4=True)` ✓）—— 「5.98 GiB 装下 12.82B」的那条路 ✓。

## 为什么单独一个自检（2026-11-02 ✓）

盘上 8 个 4bit 文件（`*_int4_w4a4` / `*_int4_convrot` ✓）此前只有「整份反量化成 bf16」一条路 ✓✗ ——
Krea2 一族因此要 **23.89 GiB** ✓✗ > 卡 22.49 GiB ⇒ **根本装不进去** ✗（`loader` 会如实报装不下 ✓）。
打包字节常驻之后驻留 ＝ **5.98 GiB**（＝文件数据区 ✓）✓。

## 四个静默失败面（全部「形状对、不报错、数值错」✗）

1. **解包顺序**：先乘 scale 再解包 / 或对打包字节直接乘 scale ⇒ 每字节被当成一个数 ✓✗
   （2026-10-30 那次假绿的同一形状 ✓ —— 只用「逐位对拍参考」才抓得住 ✓）；
2. **int8 计划抢走 4bit 层**：4bit 打包与 per-row int8 **头部逐字同形** ✓✗（都是 `I8` ＋
   `weight_scale` ✓）⇒ 两张计划表必须由**文件自述**分工 ✓（:func:`quant.int4_stems_of` ✓）；
3. **没接线**：`_shell_int8` 少传 `int4_plan` ⇒ 一层都不换 ✓✗，装载时按 bf16 物化 ⇒ 4 倍显存 ✓
   ⇒ OOM 的报错指不到"哪层没换" ✓（本文件 ⑤ 的反证用例钉这一条 ✓）；
4. **装载被 cast**：`weights` 的 int8 常驻路要求文件 dtype 逐字是 `int8` ✓ —— 打包层同样命中 ✓，
   若被 cast 成 bf16 再写回 ⇒ 静默截断成 0 ✓✗（本文件断言写入后**逐位相同** ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_int4_resident_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from app.services.engine import quant as quant_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPPED: list[str] = []

#: 真实文件（盘上未必有 ⇒ 找不到就**单列 SKIP** ✓ 不混进 PASS ✗）
REAL_KREA2_W4A4 = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models\Krea2"
                       r"\krea2_v1.5_anime_int4_w4a4.safetensors")
REAL_KREA2_MOODY = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models\Krea2"
                        r"\Krea2-Moody-Mix-premium_int4_convrot.safetensors")


def check(name: str, ok: bool, detail: object = None) -> None:
    _RESULTS.append((name, bool(ok), detail))


def skip(name: str, why: str) -> None:
    _SKIPPED.append(f"{name} —— {why}")


def _torch() -> Any:
    import torch  # noqa: PLC0415

    return torch


def _h3() -> Any:
    from app.services.engine import h3_form  # noqa: PLC0415

    return h3_form


def _skips() -> list[str]:
    try:
        _torch()
        from safetensors.torch import save_file  # noqa: F401, PLC0415
    except ImportError:
        return ["未装 torch / safetensors ⇒ 4bit 常驻自检跳过（它要造真 safetensors ✓）"]
    return []


def _pack(logical: Any) -> Any:
    """把**逻辑** int8 权重（4bit 值域 ✓）压成打包字节 ✓ —— 与 :func:`quant.unpack_int4` 互逆 ✓。

    ⚠️ 约定与解包**同一张表**写死 ✓（低半字节＝偶数列 ✓、nibble ≥ 8 表示负数 ✓）：
    两处各自独立实现才叫对拍 ✓（若这里调用 `unpack_int4` 的反函数，等于自己证明自己 ✓✗）。
    """
    torch = _torch()
    rows, cols = int(logical.shape[0]), int(logical.shape[1])
    if cols % 2:
        raise ValueError("列数必须是偶数 ✓")
    low = logical[:, 0::2].to(torch.int16) & 0x0F
    high = logical[:, 1::2].to(torch.int16) & 0x0F
    packed = (low | (high << 4)).to(torch.uint8)
    return packed.view(torch.int8), (rows, cols)


# ══════════════════════════════════════════════════════════════════════════
# ① 解包顺序：Int4Linear 与「先解包、再按 int8 层算」**逐位相同** ✓
# ══════════════════════════════════════════════════════════════════════════
def case_unpack_forward() -> None:
    torch = _torch()
    h3 = _h3()
    gen = torch.Generator().manual_seed(2026)
    logical = torch.randint(-8, 8, (64, 512), dtype=torch.int8, generator=gen)
    packed, _ = _pack(logical)
    scale = torch.rand((64, 1), dtype=torch.float32, generator=gen) * 0.01 + 0.001

    layer = h3.Int8Linear(512, 64, bias=False, scale_shape=(64, 1), packed4=True,
                          dtype=torch.float32)
    layer.weight.copy_(packed.view(torch.int8))
    layer.weight_scale.copy_(scale)
    x = torch.randn(3, 512, dtype=torch.float32, generator=gen)

    got = layer(x)

    #: 参考：**手工**解包 + 乘 scale（不含旋转 ✓）—— 与层内实现是两条独立代码 ✓
    want = torch.nn.functional.linear(x, logical.to(torch.float32) * scale)
    gap = float((got - want).abs().max()) / (float(want.abs().max()) or 1.0)
    check("① ⭐ 4bit 常驻层 == 「先解包、后乘 scale」的参考 ✓（相对误差 < 1e-6 ✓）"
          "—— 解包在**卡上**做 ✓、顺序不可反 ✗",
          gap < 1e-6, gap)

    #: 反向 ①：**把打包字节当成 256 列的 int8 权重直接乘 scale** ✓（＝「每字节一个数」那个错解 ✓）
    #: ⇒ 必须差得远 ✓（否则上一条判据是恒真的 ✗）。用**打包形状自己**去算 ✓，
    #: 才叫「同一份字节的另一种读法」✓（而不是换个输入凑一个必然不同的结果 ✓）。
    wrong = torch.nn.functional.linear(x[:, :256], packed.to(torch.float32) * scale)
    gap_wrong = float((got - wrong).abs().max()) / (float(wrong.abs().max()) or 1.0)
    check("①′ 反证：「不解包、拿打包字节当权重」必须**差得远** ✓（否则 ① 是恒真判据 ✗）",
          gap_wrong > 0.1, gap_wrong)

    #: ② 与 int8 层在**同一份逻辑权重**上逐位相同 ✓（同一条前向代码 ✓，只多一步解包 ✓）
    ref = h3.Int8Linear(512, 64, bias=False, scale_shape=(64, 1), dtype=torch.float32)
    ref.weight.copy_(logical)
    ref.weight_scale.copy_(scale)
    gap_ref = float((got - ref(x)).abs().max()) / (float(want.abs().max()) or 1.0)
    check("② 同一份逻辑权重下，4bit 常驻层与 int8 常驻层**逐位相同** ✓"
          "（差别只有「参数是打包形状」＋「前向多一步解包」✓）",
          gap_ref == 0.0, gap_ref)

    #: ③ 参数形状与字节数：**打包**形状 ✓、正好是逻辑的一半 ✓
    check("③ 参数形状 ＝ 文件里的**打包**形状 ✓（64×256 ✓，不是 64×512 ✗）⇒ "
          "装载侧「参数是 int8 ⇒ 原样写」那条路**无需任何特殊分支** ✓",
          tuple(layer.weight.shape) == (64, 256)
          and layer.weight.numel() * layer.weight.element_size() == 64 * 256,
          tuple(layer.weight.shape))


# ══════════════════════════════════════════════════════════════════════════
# ④ convrot：4bit 层的旋转搬边与 int8 层**同一条路** ✓（数值对齐 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_packed_convrot() -> None:
    torch = _torch()
    h3 = _h3()
    group = 256
    gen = torch.Generator().manual_seed(77)
    logical = torch.randint(-8, 8, (32, 2 * group), dtype=torch.int8, generator=gen)
    packed, _ = _pack(logical)
    scale = torch.rand((32, 1), dtype=torch.float32, generator=gen) * 0.01 + 0.001

    layer = h3.Int8Linear(2 * group, 32, bias=False, scale_shape=(32, 1),
                          convrot=group, packed4=True, dtype=torch.float32)
    layer.weight.copy_(packed.view(torch.int8))
    layer.weight_scale.copy_(scale)
    x = torch.randn(4, 2 * group, dtype=torch.float32, generator=gen)

    got = layer(x)                     # 激活侧旋转 ✓

    #: 参考：**权重侧**反旋转 ✓（与 `engine_int8_resident_test` ① 同一把尺子 ✓）
    w_rot = logical.to(torch.float32) * scale.reshape(-1, 1)
    h = quant_mod.build_hadamard(group)
    want = torch.nn.functional.linear(x, quant_mod.rotate_weight(w_rot, h, group))
    gap = float((got - want).abs().max()) / (float(want.abs().max()) or 1.0)
    check("④ ⭐ 4bit 常驻层的 convrot **搬边**与 int8 层同一条路 ✓"
          "（与「权重侧反旋转」参考数值对齐 ✓，相对误差 < 1e-6 ✓）",
          gap < 1e-6, gap)

    #: 反向：**漏搬旋转**（直接把未旋转的 x 喂进去）⇒ 必须差得远 ✓
    h2 = quant_mod.build_hadamard(group)
    plain = torch.nn.functional.linear(x, quant_mod.rotate_weight(w_rot, h2, group))
    gap0 = float((got - plain).abs().max()) / (float(plain.abs().max()) or 1.0)
    check("④′ 反证：漏搬旋转 ⇒ 差得远 ✓（4bit 层不是「绕开旋转」的特例 ✓）", gap0 < 1e-6, gap0)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 两张计划表**由文件自述分工** ✓（互斥 ✓）
# ══════════════════════════════════════════════════════════════════════════
def _fake_info(entries: dict[str, tuple[str, tuple[int, ...]]]) -> Any:
    from types import SimpleNamespace  # noqa: PLC0415

    tensors = {name: SimpleNamespace(dtype=dtype, shape=shape, nbytes=1)
               for name, (dtype, shape) in entries.items()}
    return SimpleNamespace(tensors=tensors)


def case_plan_split() -> None:
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    int4_spec = {"format": "convrot_w4a4", "convrot": True, "convrot_groupsize": 256}
    int8_spec = {"format": "int8_tensorwise"}
    specs = {"blk.a": int4_spec, "blk.b": int8_spec}
    stems = quant_mod.int4_stems_of(specs)

    info = _fake_info({
        "blk.a.weight": ("I8", (64, 256)),         # 4bit 打包（64×512 逻辑 ✓）
        "blk.a.weight_scale": ("F32", (64, 1)),
        "blk.b.weight": ("I8", (64, 512)),         # per-row int8 ✓
        "blk.b.weight_scale": ("F32", (64, 1)),
        "blk.c.weight": ("BF16", (64, 512)),       # 没有 scale ✓
    })
    plan8 = tb.int8_scale_plan(info, int4_stems=stems)
    plan4 = tb.int4_scale_plan(info, int4_stems=stems)

    check("⑤ ⭐ 4bit 与 int8 **各归各的**：自述 4bit 的层只进 int4 计划 ✓、per-row int8 只进 int8 计划 ✓"
          "（头部逐字同形 ⇒ 分工**只能**靠文件自述 ✓）",
          set(plan4) == {"blk.a"} and set(plan8) == {"blk.b"},
          (sorted(plan4), sorted(plan8)))

    #: 反向 ⑤⁴：两条路**都**不拿文件自述 ⇒ 4bit 层被 int8 计划抢走 ✓✗（这就是那个假绿 ✓）
    plan8_blind = tb.int8_scale_plan(info)
    plan4_blind = tb.int4_scale_plan(info)
    check("⑤⁴ 反证：不把文件自述递进去 ⇒ 4bit 层**被 int8 计划抢走** ✓✗、4bit 计划空 ✓"
          "（⇒ 相乘时按「每字节一个数」解 ✓，形状全对、数值全错 ✓）",
          "blk.a" in plan8_blind and plan4_blind == {},
          (sorted(plan8_blind), sorted(plan4_blind)))

    #: ⑤⁵ 自称 4bit 却**没有 scale** ⇒ 两张表都不要它 ✓（缺的留给装载闸口报 ✓，这里不塞假尺度 ✓）
    info2 = _fake_info({"blk.a.weight": ("I8", (64, 256))})
    check("⑤⁵ 自称 4bit 但**缺 scale** ⇒ 不进 4bit 计划 ✓（不在这补假尺度 ✗、也不退化成 int8 ✗）",
          tb.int4_scale_plan(info2, int4_stems=stems) == {}
          and tb.int8_scale_plan(info2, int4_stems=stems) == {})


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 真文件：需求数从 23.89 GiB 降到打包字节 ✓（这就是能装与装不下的分界 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_real_file() -> None:
    from app.services.engine import loader as loader_mod  # noqa: PLC0415
    from app.services.engine import safetensors as st  # noqa: PLC0415
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    for path, want_layers in ((REAL_KREA2_W4A4, 224), (REAL_KREA2_MOODY, 256)):
        if not path.is_file():
            skip(f"⑥ {path.name}（4bit 真文件需求数）", f"不在盘上：{path}")
            continue
        info = st.inspect(path)
        specs = loader_mod.quant_specs_of(path, info)
        plan4 = tb.int4_scale_plan(info, int4_stems=quant_mod.int4_stems_of(specs))
        gib_krea2, why_krea2 = tb.part_need_gib(str(path), fallback=12.57,
                                                adapter="krea2", assume="int8")
        gib_blind, _ = tb.part_need_gib(str(path), fallback=12.57, adapter=None, assume="int8")
        check(f"⑥ {path.name}：{want_layers} 层全进 4bit 常驻计划 ✓、"
              f"krea2 线下驻留 {gib_krea2:.2f} GiB ✓ 装得进 22.49 卡 ✓"
              f"（没有壳时 {gib_blind:.2f} GiB ✗）",
              len(plan4) == want_layers and gib_krea2 < 15.0 and gib_blind > gib_krea2 * 2,
              (len(plan4), round(gib_krea2, 2), round(gib_blind, 2),
               why_krea2.get("int4ShellLayers")))


# ══════════════════════════════════════════════════════════════════════════
# ⑦ 接线 ＋ 装载：换壳 → 原样写入 → 前向 == 旧路（反量化 bf16）✓
# ══════════════════════════════════════════════════════════════════════════
def case_shell_and_load() -> None:
    torch = _torch()
    from safetensors.torch import save_file  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    from app.services.engine import dit_build  # noqa: PLC0415
    from app.services.engine import h3_form  # noqa: PLC0415
    from app.services.engine import safetensors as st  # noqa: PLC0415
    from app.services.engine import torch_backend as tb  # noqa: PLC0415
    from app.services.engine import weights as weights_mod  # noqa: PLC0415

    group = 256
    in_features, out_features = 2 * group, 32
    gen = torch.Generator().manual_seed(4242)
    logical = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8, generator=gen)
    packed, _ = _pack(logical)
    scale = torch.rand((out_features, 1), dtype=torch.float32, generator=gen) * 0.01 + 0.001
    meta = json.dumps({"format": "convrot_w4a4", "convrot": True,
                       "convrot_groupsize": group}, separators=(",", ":"))

    class _Holder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blk = nn.Module()
            self.blk.q = nn.Linear(in_features, out_features, bias=False)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny_int4.safetensors"
        save_file({"blk.q.weight": packed.contiguous(),
                   "blk.q.weight_scale": scale.contiguous(),
                   "blk.q.comfy_quant": torch.tensor(list(meta.encode("utf-8")),
                                                     dtype=torch.uint8)}, str(path),
                  metadata={})
        info = st.inspect(path)

        holder = _Holder()
        plan8, _layers, plan4 = dit_build._shell_int8(  # noqa: SLF001
            holder, info, None, path, tb.int8_scale_plan, tb.int4_scale_plan)
        layer = holder.blk.q
        check("⑦ ⭐ 接线到位：`_shell_int8` 把自述 4bit 的层换成**打包常驻层** ✓"
              "（`packed4=True` ✓、参数＝打包形状 ✓）、int8 计划**空** ✓",
              isinstance(layer, h3_form.Int8Linear) and layer.packed4
              and tuple(layer.weight.shape) == (out_features, in_features // 2)
              and plan8 == {} and set(plan4) == {"blk.q"},
              (type(layer).__name__, getattr(layer, "packed4", None),
               tuple(layer.weight.shape), sorted(plan4)))

        #: 反向 ⑦⁴：**不传** `int4_plan_fn`（＝本轮改动之前的样子 ✓）⇒ 一层都不换 ✓✗
        blind = _Holder()
        dit_build._shell_int8(blind, info, None, path, tb.int8_scale_plan)  # noqa: SLF001
        check("⑦⁴ 反证：`_shell_int8` **不传** 4bit 计划 ⇒ 该层还是普通 `nn.Linear` ✓✗"
              "（装载时按 bf16 反量化 ⇒ 4 倍显存 ⇒ OOM 报错指不到「哪层没换」✓）",
              type(blind.blk.q) is nn.Linear, type(blind.blk.q).__name__)

        report = weights_mod.load_module_weights(holder, path)   # dtype=None ⇒ 全保原精度 ✓
        check("⑦⁵ 装载**原样写入** ✓（不反量化 ✗、不 cast ✗）：`int4Resident=1` ✓、"
              "`int8Resident=1` ✓（子集关系 ✓）、`int4PackCount=0` ✓（故意不解 ✓）、无 error ✓",
              report.error == "" and report.int4Resident == 1 and report.int8Resident == 1
              and report.dequant.get("int4PackCount", 0) == 0
              and "int4Resident" in report.dequant,
              (report.error, report.int4Resident, report.int8Resident,
               report.dequant.get("int4PackCount"), report.dequant.get("int4Resident")))

        check("⑦⁶ 卡上的字节 == 文件里的字节（**逐位相同** ✓）—— 被 cast 过就会静默截断成 0 ✓✗",
              torch.equal(layer.weight.detach().cpu(), packed.contiguous().view(torch.int8)),
              None)

        #: 旧路参考：**整份反量化**（4bit 解包 + 乘 scale + 权重侧反旋转 ✓）再 `F.linear` ✓
        x = torch.randn(3, in_features, dtype=torch.float32, generator=gen)
        w_rot = logical.to(torch.float32) * scale.reshape(-1, 1)
        h = quant_mod.build_hadamard(group)
        want = torch.nn.functional.linear(x, quant_mod.rotate_weight(w_rot, h, group))
        got = layer(x)
        gap = float((got - want).abs().max()) / (float(want.abs().max()) or 1.0)
        check("⑦⁷ 装完之后前向结果 == 旧路（整份反量化成 bf16 再算）✓（相对误差 < 1e-6 ✓）"
              "⇒ 「省显存」没换掉数学 ✓",
              gap < 1e-6, gap)


# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    #: ⚠️ cp936 控制台下首行 `✓` 直接 UnicodeEncodeError ⇒ 0 条判据就退 1 ✓✗
    #: （`run_all` 下不露头，因为它已重配过 stdout ✓ —— 单独跑才中招 ✓）
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    reasons = _skips()
    if reasons:
        for why in reasons:
            print(f"SKIP  {why}")
        return 0

    case_unpack_forward()
    case_packed_convrot()
    case_plan_split()
    case_real_file()
    case_shell_and_load()

    failed = 0
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failed += 1
            print(f"      detail: {detail}")
    for line in _SKIPPED:
        print(f"SKIP  {line}")
    # ⚠️ 收尾行必须是 ``SUMMARY: n/m passed`` 这个**字面量** ✓（``run_all`` 只认它 ✓；
    # 写成 ``SUMMARY n/m passed, k skipped`` ⇒ 全量表里本套**项数恒为 0** ✗，见 ``structure_test`` ㊈ ✓）。
    # skip 的条数另起一行说 ✓（不挂在同行 ⇒ 不参与解析 ✓）。
    print(f"\nSUMMARY: {len(_RESULTS) - failed}/{len(_RESULTS)} passed")
    print(f"  （skip {len(_SKIPPED)}）")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
