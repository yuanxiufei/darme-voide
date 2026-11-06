"""``foresight``（层预取上卡）的判据。

盯六类事（每条都能**独立失败** ✓）：

1. **判据本身**（不碰 torch ✓）：没落地不给用、同一组不重复发、槽不够记 ``busy``、
   窗口配额记 ``overBudget``、流水号对不上不算落地；
2. **"copy 没落地就不用"是硬约束**：把落地事件换成"永远为假" ⇒ 那层必须照原路在 CPU 上算 ✓
   且数值不变 ✓（不是等 ✗、更不是拿半份数据算 ✗✗）；
3. **落地的那些层真在卡上算**：多跑两趟 ⇒ ``usedLayers > 0``，且权重**算完换回 CPU** ✓；
4. **一大块"计划之外"的显存**：槽要**从预算里扣** ✓（同预算下卡上少一层 ✓），
   扣完太亏 ⇒ 不启用并写明理由 ✓；
5. **老路零变化**：不给 ``foresight`` ⇒ 报告里那个字段是 ``None`` ✓、一根多余的 hook 都没有 ✓；
6. **退件把显存还回来**：``detach()`` 之后槽里的缓冲全放掉 ✓（否则下次装配 OOM 找不到主 ✓）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # noqa: E402 - cp936 控制台下 ✓ 会 UnicodeEncodeError ✓

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from app.services.engine import foresight as fs  # noqa: E402
from app.services.engine import residency as r  # noqa: E402
from app.services.engine import torch_backend as tb  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
HAS_CUDA = torch.cuda.is_available()


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


class _Block(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(dim, dim)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.lin(x)) + x


class _Tiny(torch.nn.Module):
    def __init__(self, dim: int = 16, depth: int = 3) -> None:
        super().__init__()
        self.x_embedder = torch.nn.Linear(dim, dim)
        self.layers = torch.nn.ModuleList([_Block(dim) for _ in range(depth)])
        self.final = torch.nn.Linear(dim, dim)
        self.pad = torch.nn.Parameter(torch.zeros(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.x_embedder(x) + self.pad
        for block in self.layers:
            h = block(h)
        return self.final(h)


def _seeded(dim: int = 16, depth: int = 3, seed: int = 5) -> _Tiny:
    torch.manual_seed(seed)
    model = _Tiny(dim, depth)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.3)
    return model


def _budget_for(model: _Tiny, layers: int) -> int:
    groups = r.survey(model)
    non_layer = sum(g.nbytes for g in groups if not g.is_layer)
    one = next(g.nbytes for g in groups if g.name == "layers.0")
    return non_layer + one * layers


def _rel_error(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.cpu().double() - want.cpu().double()).abs().max().item()
            / max(want.cpu().double().abs().max().item(), 1e-12))


# ── ① 判据本身（纯逻辑，不碰 torch）──────────────────────────────────────────────


def case_judgement() -> None:
    bad = fs.selftest()
    check("① 判据自检全过（没落地不给用 / 不重复发 / busy / overBudget / 流水号）", bad == 0,
          f"{bad} 条不过")
    check("① 默认 0 槽 ⇒ 这一级**不存在**（与参考实现的 SLOTS>0 同一条）",
          not fs.Foresight().enabled and fs.of_spec(None).slots == 0)
    slots = fs.of_spec(True)
    check("① 只给 True ⇒ 2 个槽（1 个在用 + 1 个在拷，**1 个槽没有重叠** ✓）",
          slots.slots == 2, slots.slots)


# ── ⑤ 老路零变化（不用卡也能验）────────────────────────────────────────────────


def case_zero_change() -> None:
    model = _seeded()
    want = model(torch.randn(2, 7, 16))
    handle = r.place(model, main_device=torch.device("cpu"), budget=0, mode="split")
    check("⑤ 不给 foresight ⇒ 报告里那个字段是 None（不是空结构）",
          handle.report()["foresight"] is None, handle.report().get("foresight"))
    check("⑤ 没卡时预算 0 ⇒ 照样不启用、不报错", handle.plan.budget == 0, handle.plan.reason)
    del want
    handle.detach()

    conf = tb._foresight_of("off")
    check("⑤ residency=off ⇒ 环境变量开了也不启用（没有卡就没有「上卡」）",
          conf is None or not conf.enabled, conf)


# ── ②③④⑥ 真跑（有卡才走这一段）────────────────────────────────────────────────


def case_split_prefetch() -> None:
    if not HAS_CUDA:
        check("本机没有 CUDA ⇒ 跳过层预取真跑（只验判据）", True, "no cuda")
        return
    main = torch.device("cuda")
    model = _seeded(seed=11)
    x = torch.randn(2, 7, 16)
    with torch.no_grad():
        want = model(x).clone()

    budget = _budget_for(model, 2)
    conf = fs.Foresight(slots=1, minBytes=0)
    handle = r.place(model, main_device=main, budget=budget, mode="split", foresight=conf)
    report = handle.report()
    one = next(g.nbytes for g in r.survey(model) if g.name == "layers.0")

    check("③ 报告里有 foresight（≠ None ⇒ 这一级真的开着）",
          report["foresight"] is not None and report["foresight"]["enabled"], report["foresight"])
    check("④ 槽从预算里真扣掉了：plan.budget == 给的 − 1 层的字节",
          handle.plan.budget == budget - one, (handle.plan.budget, budget, one))
    check("④ 报告写明预留了多少（不是扣了却看不出来；按原始字节核，不按四舍五入后的 GiB）",
          conf.reserveBytes == one, (conf.reserveBytes, one))
    check("③ CPU 侧还剩两层 ⇒ 才有「下一层」可预取",
          handle.plan.layers_on_cpu == 2, report["layersOnCpu"])

    with torch.no_grad():
        first = model(x)
    #: 小模型上"拷完"比"这一层算完"还快 ✓ ⇒ 第一趟就可能已经用上 ✓（不能拿"第一趟必为 0"当判据 ✗）
    check("③ 第一趟就把拷贝发出去了（不是挂了个空 hook）",
          handle.slots.prefetched >= 1 and handle.report()["foresight"]["copies"] >= 1,
          handle.report()["foresight"]["copies"])
    check("③ 第一趟算完没有半换回去的权重（slotPairs 空、CPU 层仍在 CPU）",
          not handle.slotPairs
          and all(str(p.device) == "cpu" for p in model.layers[1].parameters()),
          handle.slotPairs)

    with torch.no_grad():
        for _ in range(3):
            got = model(x)
    report = handle.report()
    check("③ 多跑几趟之后确有层在卡上算（usedLayers > 0）",
          handle.usedSlots > 0, report["foresight"])
    check("③ 预取真搬了权重（按原始字节核 —— 这是 split 下唯一的权重搬运开销）",
          handle.prefetchBytes > 0 and handle.usedSlots > 0,
          (handle.prefetchBytes, handle.usedSlots))
    err = _rel_error(got, want)
    check("② 数值与全 CPU 逐元素一致（< 1e-5）—— 预取不改结果", err < 1e-5, f"相对差 {err:.3e}")
    del first

    check("③ 算完把权重**换回 CPU**（槽只在那一层的前向期间是「在卡上」）",
          all(str(p.device) == "cpu" for p in model.layers[1].parameters())
          and not handle.slotPairs, handle.placement())
    check("③ 槽里的缓冲用完就放（不是一层层攒到把卡撑爆）",
          len(handle.slots.tensors) <= 1, list(handle.slots.tensors))

    handle.detach()
    check("⑥ detach 之后槽里的显存全部还回去", not handle.slots.tensors and handle.slots.bytes == 0,
          handle.slots.tensors)


def case_never_landed() -> None:
    """② 的硬约束：把落地事件换成"永远为假" ⇒ 必须照原路在 CPU 上算 ✓、数值不变 ✓。"""
    if not HAS_CUDA:
        check("本机没有 CUDA ⇒ 跳过「没落地就不用」用例", True, "no cuda")
        return
    main = torch.device("cuda")
    model = _seeded(seed=13)
    x = torch.randn(2, 7, 16)
    with torch.no_grad():
        want = model(x).clone()

    conf = fs.Foresight(slots=1, minBytes=0)
    handle = r.place(model, main_device=main, budget=_budget_for(model, 2), mode="split",
                     foresight=conf)

    class _Never:
        """永远没落地 ✓（拷流上的 Event 在真机上查一次就真了 ⇒ 这里必须靠替身验第二条路 ✓）。"""

        def query(self) -> bool:
            return False

        def synchronize(self) -> None:
            return None

    real = handle.slots.prefetch

    def fake(name: str, node: object) -> bool:
        ok = real(name, node)
        if ok and name in handle.slots.flying:
            token, _event = handle.slots.flying[name]
            handle.slots.flying[name] = (token, _Never())
        return ok

    handle.slots.prefetch = fake          # type: ignore[method-assign]
    with torch.no_grad():
        for _ in range(3):
            got = model(x)
    report = handle.report()
    check("② 一次都没用上（usedLayers == 0）—— 没落地就是不用，绝不等",
          handle.usedSlots == 0, report["foresight"])
    check("② 报告里能看出是「在等」而不是「没发」（pending > 0）",
          report["foresight"]["pending"] > 0, report["foresight"])
    err = _rel_error(got, want)
    check("② 没落地时结果与全 CPU 一致（走了原路，不是拿半份数据算）", err < 1e-5,
          f"相对差 {err:.3e}")
    handle.detach()


def case_park_and_refuse() -> None:
    if not HAS_CUDA:
        check("本机没有 CUDA ⇒ 跳过 park / 太亏两条", True, "no cuda")
        return
    main = torch.device("cuda")
    model = _seeded(seed=17)
    conf = fs.Foresight(slots=1, minBytes=0)
    handle = r.place(model, main_device=main, budget=_budget_for(model, 2), mode="park",
                     foresight=conf)
    report = handle.report()
    check("⑤ park 下不启用（它本来就逐层搬权重，再占一层槽是纯浪费）",
          report["foresight"] is not None and not report["foresight"]["enabled"]
          and handle.slots is None, report["foresight"])
    handle.detach()

    tiny = _seeded(seed=19)
    conf2 = fs.Foresight(slots=2, minBytes=0)
    handle2 = r.place(tiny, main_device=main, budget=_budget_for(tiny, 2), mode="split",
                      foresight=conf2)
    check("④ 槽比预算一半还大 ⇒ 不启用并写明理由（不是硬扣把层全赶去 CPU）",
          not handle2.report()["foresight"]["enabled"]
          and "赔本" in str(handle2.report()["foresight"]["note"]),
          handle2.report()["foresight"].get("note"))
    handle2.detach()


def case_robustness() -> None:
    """两条"坏在半路"的路 ✓：槽里缺项（不许半换 ✗）、预取本身抛错（不许带崩前向 ✗✗）。"""
    if not HAS_CUDA:
        check("本机没有 CUDA ⇒ 跳过坏在半路两条", True, "no cuda")
        return
    main = torch.device("cuda")

    model = _seeded(seed=23)
    x = torch.randn(2, 7, 16)
    with torch.no_grad():
        want = model(x).clone()
    handle = r.place(model, main_device=main, budget=_budget_for(model, 2), mode="split",
                     foresight=fs.Foresight(slots=1, minBytes=0))

    #: 槽里**少一项**（组在后加的成员 / 名字变了 ✓）⇒ 整份放弃 ✓，绝不能换一半 ✗✗
    handle.slots.tensors["layers.1"] = {"__不存在的张量": torch.zeros(1, device=main)}
    check("② 槽里缺项 ⇒ swap_in 整份放弃（不是换一半）",
          handle.slots.swap_in("layers.1", r._node(model, "layers.1")) is None
          and all(str(p.device) == "cpu" for p in model.layers[1].parameters()))
    handle.slots.tensors.pop("layers.1", None)

    real = handle.slots.prefetch

    def fake(name: str, node: object) -> bool:
        ok = real(name, node)
        if ok and name in handle.slots.tensors:
            bag = handle.slots.tensors[name]
            keys = list(bag)
            if len(keys) > 1:
                handle.slots.tensors[name] = {k: bag[k] for k in keys[:-1]}
        return ok

    handle.slots.prefetch = fake       # type: ignore[method-assign]
    with torch.no_grad():
        for _ in range(2):
            got = model(x)
    torch.cuda.synchronize()
    err = _rel_error(got, want)
    check("② 槽里缺项时照原路在 CPU 上算（数值不变、权重没被换走）",
          err < 1e-5 and all(str(p.device) == "cpu" for p in model.layers[1].parameters()),
          f"相对差 {err:.3e}")
    handle.detach()

    model2 = _seeded(seed=29)
    with torch.no_grad():
        want2 = model2(x).clone()
    handle2 = r.place(model2, main_device=main, budget=_budget_for(model2, 2), mode="split",
                      foresight=fs.Foresight(slots=1, minBytes=0))

    def boom(name: str, node: object) -> bool:
        raise RuntimeError("拷流挂了")

    handle2.slots.prefetch = boom      # type: ignore[method-assign]
    with torch.no_grad():
        for _ in range(2):
            got2 = model2(x)
    report = handle2.report()
    check("② 预取抛错 ⇒ 当场停用并留理由，前向照原路跑完（优化坏不许带崩推理）",
          not report["foresight"]["enabled"] and "停用" in str(report["foresight"]["note"])
          and _rel_error(got2, want2) < 1e-5, report["foresight"].get("note"))
    handle2.detach()


def main() -> int:
    case_judgement()
    case_zero_change()
    case_split_prefetch()
    case_never_landed()
    case_park_and_refuse()
    case_robustness()

    fails = 0
    for name, ok, detail in _RESULTS:
        if not ok:
            fails += 1
            print(f"FAIL {name}  <<< {detail}")
    print(f"\nSUMMARY: {len(_RESULTS) - fails}/{len(_RESULTS)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
