r"""自研引擎 · **反量化**（fp8 / int8 → 可算精度 ✓）—— 目标主权重就是 fp8 ✓ 这条在关键路径上 ✓。

覆盖：
* 四种布局（per-tensor / per-row / per-col / 逐元素 ✓）**各自**按形状判对 ✓ 且**往返**（量化→反量化 ≈ 原值 ✓）；
* 两种 scale 方向（`*.scale` 乘 / `*.weight_scale_inv` 除 ✓）—— ⚠️ 长后缀优先 ✗（否则 `_inv` 会被
  `weight_scale` 前缀命中 ⇒ 把「除」当「乘」 ✓✗，那是最坏的一种错 ✓）；
* ⭐ **判不出来必须拒绝** ✗：块量化形状 / 分组量化（`qzeros` ✓）/ 缺 scale / scale 里有 0 ✓；
* ⭐ 接进 `weights.load_module_weights`：**fp8 检查点能真装** ✓（数值对得上 ✓）且**拒绝时中止装载**
  ✓（不污染目标模块 ✓、不静默 cast ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_quant_test.py
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

    orphan = quant_mod.dequantize_state({"w.weight": fp8})
    check("⑥ 低精度权重**没有**配套 scale ⇒ **拒绝** ✓（不装 ✓ —— 装了只会算出错数 ✓✗）",
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
            case_refusals()
            case_load(root)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print(f"\nSUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
