"""LoRA **补丁层**的判据：挂载 ✓ / 搬家 ✓ / 卸载 ✓（2026-11-01 ✓）。

`engine_lora_test` 那套用**假张量**验算法 ✓（零 torch ✓）；这一套反过来，只验**生命周期** ✓，
全部走真 torch 的真模块 ✓：

1. **挂载不是折进**：装完 ``weight`` 逐位不变 ✓ —— 一旦悄悄 merge 进权重 ✓✗，
   ``strength`` 就改不回来了 ✓✗（量化底模上还会被反量化吃掉 ✓）。
2. **搬家后仍然对** ✗✗：因子活在钩子闭包里 ✓，而权重随后会被搬走 ✓（``enter_phase`` 换装 ✓、
   ``balanced`` 分时驻留的 split ✓、``module.to(dtype)`` ✓）—— 只在装载时对齐一次的写法，
   搬完就在前向炸 ``Expected all tensors to be on the same device`` ✓✗。
3. **卸载是真摘** ✗✗：``release_loRA`` 以前只还**张量租约** ✓，钩子还赖在模块上 ✓✗ ——
   重挂一次就是「增量叠两遍」✓✗（不报错、只是越来越"重" ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_lora_patch_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

#: ⚠️ cp936 控制台下 ``✓`` 会 ``UnicodeEncodeError`` ⇒ 判据还没跑就崩在第一条打印上 ✗
#: （``run_all`` 已重配 stdout，只有单独跑才露头 ⇒ 每个 ``__main__`` 测试都自带这一句 ✓）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


import torch  # noqa: E402

from app.services.engine import lora as lora_mod  # noqa: E402
from app.services.engine import torch_backend as tb  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _source(name: str) -> str:
    return (BACKEND_PY / "app" / "services" / "engine" / name).read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 一个能挂 LoRA 的最小模型：`blocks.0.mlp.gate`（名字要与 LoRA 键**逐字**对上 ✓）
# ══════════════════════════════════════════════════════════════════════════════

_IN, _OUT, _RANK = 3, 2, 2
_SCALE = 1.0                                   # alpha(2) / rank(2) ✓


def build_model(seed: int = 7) -> Any:
    torch.manual_seed(seed)
    block = torch.nn.Module()
    block.mlp = torch.nn.Module()
    block.mlp.gate = torch.nn.Linear(_IN, _OUT, bias=False)
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([block])
    return model


def build_tensors() -> dict[str, Any]:
    """LoRA 张量表（盘上布局：``down`` 是 ``(rank, in)`` ✓、``up`` 是 ``(out, rank)`` ✓）。"""
    torch.manual_seed(11)
    return {
        "blocks.0.mlp.gate.lora_down.weight": torch.randn(_RANK, _IN),
        "blocks.0.mlp.gate.lora_up.weight": torch.randn(_OUT, _RANK),
        "blocks.0.mlp.gate.alpha": torch.tensor(2.0),
    }


def reference(base_in: Any, weight: Any, down: Any, up: Any, scale: float = _SCALE) -> Any:
    """判据**另写一遍** ✓（不复用被测的 ``bypass_output`` ✓）。"""
    return base_in @ weight.T + (base_in @ down.T) @ up.T * scale


def attach(model: Any) -> Any:
    tensors = build_tensors()
    parsed = lora_mod.parse_lora(tensors)
    assert parsed.ok, parsed.error
    return lora_mod.apply_to_model(model, parsed, ops=lora_mod.torch_ops())


def close(left: Any, right: Any, tol: float = 1e-5) -> bool:
    return tuple(left.shape) == tuple(right.shape) and bool((left - right).abs().max() <= tol)


def bare_backend() -> Any:
    """只要 `release_loras` / `_detach_lora_hooks` 那两件 ✓ ⇒ 不跑真装配 ✓（省十几秒 ✓）。"""
    backend = object.__new__(tb.TorchBackend)
    backend._loraLeases = []
    backend._loraHooks = []
    return backend


# ══════════════════════════════════════════════════════════════════════════════
# ① 挂载：weights 逐位不变，输出 = base + delta
# ══════════════════════════════════════════════════════════════════════════════

def case_attach() -> None:
    model = build_model()
    tensors = build_tensors()
    module = model.blocks[0].mlp.gate
    before = module.weight.detach().clone()
    x = torch.randn(4, _IN, dtype=before.dtype)
    base = x @ before.T

    result = attach(model)
    after = module.weight.detach()
    check("① 挂载不折进权重（`weight` 逐位不变 ✓）",
          torch.equal(before, after), {"maxDiff": float((before - after).abs().max())})
    check("① 钩子挂上了 ✓（bypass 路 ✓）",
          len(result.hooks) == 1 and result.merged == [],
          {"hooks": len(result.hooks), "merged": result.merged, "error": result.error})
    expected = reference(x, before, tensors["blocks.0.mlp.gate.lora_down.weight"],
                         tensors["blocks.0.mlp.gate.lora_up.weight"])
    check("① 前向 = base + up(down(x))·scale ✓（钩子真在算 ✓）",
          close(module(x), expected), {"got": module(x).tolist(), "want": expected.tolist()})
    check("① 输出确实变了 ✓（不是「挂了但没生效」✓）", not close(module(x), base))

    # 卸载复原（钩子摘掉 ⇒ 回到 base ✓）
    backend = bare_backend()
    backend._loraHooks = list(result.hooks)
    removed = backend.release_loras(reason="自检 ⇒ 摘钩子 ✓")
    check("① 卸掉后回到 base ✓（钩子真摘 ✓）",
          close(module(x), base) and removed.get("hooks") == 1 and not backend._loraHooks,
          removed)


# ══════════════════════════════════════════════════════════════════════════════
# ② 搬家：模块与输入换设备/精度之后，钩子仍然对齐（旧写法在这里炸 ✗✗）
# ══════════════════════════════════════════════════════════════════════════════

def case_move() -> None:
    model = build_model()
    tensors = build_tensors()
    result = attach(model)
    module = model.blocks[0].mlp.gate
    weight = module.weight.detach().clone()
    down = tensors["blocks.0.mlp.gate.lora_down.weight"]
    up = tensors["blocks.0.mlp.gate.lora_up.weight"]
    x = torch.randn(4, _IN)

    #: ② 精度搬家 ✓ —— 因子留在装时的 dtype ✓，权重换成 double ✓
    module.double()
    model.blocks[0].mlp.gate = module
    try:
        got = module(x.to(torch.float64))
        moved_ok, detail = close(got, reference(x.to(torch.float64), weight.double(),
                                                down.double(), up.double())), got.tolist()
    except Exception as err:  # noqa: BLE001 —— 就是来看它会怎么炸的 ✓
        moved_ok, detail = False, f"{type(err).__name__}: {err}"
    check("② 换精度（权重 double、因子 float）之后前向仍然对 ✓", moved_ok, detail)

    if not torch.cuda.is_available():
        skip("② 设备搬家：本机没有 cuda ⇒ 只能跳过 ✓（精度那条已覆盖对齐路径 ✓）")
        return
    model2 = build_model()
    attach(model2)
    module2 = model2.blocks[0].mlp.gate
    weight2 = module2.weight.detach().clone()
    model2.cuda()
    x2 = x.cuda()
    try:
        got2 = model2.blocks[0].mlp.gate(x2)
        #: 判据在 cuda 上独立算一遍 ✓（`weight2` 是搬之前抓的 CPU 副本 ✓ ⇒ 一并搬 ✓）
        want2 = reference(x2, weight2.cuda(), down.cuda(), up.cuda())
        device_ok, detail2 = close(got2, want2, tol=1e-4), got2.device.type
    except Exception as err:  # noqa: BLE001
        device_ok, detail2 = False, f"{type(err).__name__}: {err}"
    check("② 设备搬家（模块上卡、因子在 CPU）之后前向仍然对 ✓", device_ok, detail2)

    backend = bare_backend()
    backend._loraHooks = list(result.hooks)
    backend.release_loras(reason="自检收尾 ✓")


# ══════════════════════════════════════════════════════════════════════════════
# ③ 卸载与重挂：叠两次 = 两倍增量（所以「不摘钩子」是真缺陷 ✗），摘了就不残留
# ══════════════════════════════════════════════════════════════════════════════

def case_reattach() -> None:
    model = build_model()
    tensors = build_tensors()
    module = model.blocks[0].mlp.gate
    weight = module.weight.detach().clone()
    down = tensors["blocks.0.mlp.gate.lora_down.weight"]
    up = tensors["blocks.0.mlp.gate.lora_up.weight"]
    x = torch.randn(4, _IN)
    base = x @ weight.T
    once = reference(x, weight, down, up)

    first = attach(model)
    second = attach(model)
    twice = module(x)
    check("③ 挂两遍 = 增量叠两遍 ✓（这就是「必须能摘钩子」的理由 ✓）",
          close(twice, base + (once - base) * 2) and len(first.hooks) + len(second.hooks) == 2,
          {"got": twice.tolist(), "want": (base + (once - base) * 2).tolist()})

    backend = bare_backend()
    backend._loraHooks = list(first.hooks) + list(second.hooks)
    report = backend.release_loras(reason="自检 ⇒ 摘两遍 ✓")
    check("③ 全部摘掉后回到 base ✓（不残留 ✓）",
          close(module(x), base) and report.get("hooks") == 2, report)

    #: ⚠️ 同一个 handle 摘两次不许炸 ✗✗（`release_loras` 在装载中止路径上会被调两次 ✓）
    again = backend.release_loras(reason="再摘一次 ✓")
    check("③ 重复卸载是空操作 ✓（不抛 ✓）", again.get("hooks") == 0, again)


# ══════════════════════════════════════════════════════════════════════════════
# ④ int8 的 weight 不能当对齐参考（它是整数量化因子 ✗）
# ══════════════════════════════════════════════════════════════════════════════

def case_int8_reference() -> None:
    x = torch.randn(2, _IN)
    int8_weight = torch.randint(-127, 127, (_OUT, _IN), dtype=torch.int8)
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(int8_weight, requires_grad=False)
    check("④ int8 层对齐到**输入**而不是 int8 权重 ✓（退化成整数量化因子就不对了 ✓）",
          lora_mod._patch_reference(module, x) is x)
    float_module = torch.nn.Linear(_IN, _OUT, bias=False)
    check("④ 浮点层仍对齐到权重 ✓",
          lora_mod._patch_reference(float_module, x) is float_module.weight)


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ 源码级：两处收口不许被改回去（这两条都是"改了不报错、只是悄悄变错"✗）
# ══════════════════════════════════════════════════════════════════════════════

def case_source() -> None:
    lora_src = _source("lora.py")
    hook = lora_src.split("def _bypass_hook", 1)[-1].split("\ndef ", 1)[0]
    check("⑤ `_bypass_hook` 每次前向都把因子对齐一次 ✓（删回「只在装载时对齐」⇒ 搬家即炸 ✗）",
          "_patch_reference(" in hook and "_align_delta(" in hook,
          {"hasRef": "_patch_reference(" in hook, "hasAlign": "_align_delta(" in hook})

    backend_src = _source("torch_backend.py")
    release = backend_src.split("def release_loras", 1)[-1].split("\n    def ", 1)[0]
    check("⑤ `release_loras` 先摘钩子再还租约 ✓（只还租约 ⇒ 钩子指着已释放的张量 ✗）",
          "_detach_lora_hooks(" in release and release.index("_detach_lora_hooks(")
          < release.index("_release_leases("), release.strip()[:120])
    apply_body = backend_src.split("def _apply_loras", 1)[-1].split("\n    def ", 1)[0]
    check("⑤ 装成功要**把钩子入账** ✓（不入账 ⇒ `release_loras` 摘不到 ✗）",
          "_loraHooks.extend(" in apply_body)
    check("⑤ 半装失败也要摘 ✓（`_detach_lora_hooks(list(result.hooks))` ✓）",
          "_detach_lora_hooks(list(result.hooks))" in apply_body)


def main() -> int:
    for case in (case_attach, case_move, case_reattach, case_int8_reference, case_source):
        try:
            case()
        except Exception as err:  # noqa: BLE001 —— 一例炸不该带走其余 ✓
            check(f"{case.__name__} 未抛异常 ✗", False, f"{type(err).__name__}: {err}")

    for name, ok, detail in _RESULTS:
        print(f"{'OK   ' if ok else 'FAIL '}{name}" + ("" if ok else f"   <<< {detail}"))
    for reason in _SKIPS:
        print(f"SKIP {reason}")
    failed = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"SUMMARY: {len(_RESULTS) - failed}/{len(_RESULTS)} passed"
          + (f"（{len(_SKIPS)} 项 SKIP ✓）" if _SKIPS else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
