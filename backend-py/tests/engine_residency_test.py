"""``residency``（GPU + CPU 联合驻留）的判据。

盯五类事：

1. **切分数学对**：非层参数先上卡、层按**前向顺序**取连续前缀（不能挑小的塞，否则激活每步过界）；
   预算连非层都放不下时 ``fits=False`` 且说清原因，不许假装切成功；
2. **真的两台设备一起算**（本机有卡就真跑）：split 下 CPU 那几层的权重**不搬**、数值与全 CPU 一致、
   ``transfers`` 只在跨界处涨一次；
3. **hook 不是装饰**：同样的布局**不挂 hook** 必须报错 —— 否则「跑通了」可能只是碰巧同设备；
4. **老路零开销**：全在一处时不挂任何 hook、``transfers == 0``；
5. **park 与 split 的区别是「搬什么」**：park 搬权重（能精确报每步 MiB），split 搬激活（报不出、不猜）。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from app.services.engine import residency as r  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
HAS_CUDA = torch.cuda.is_available()


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ── 被测模型（三层，够切出「卡上 1 层 / CPU 2 层」）────────────────────────────────


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
        self.pad = torch.nn.Parameter(torch.zeros(1, dim))   # 顶层裸参数 ⇒ ‹root› 组

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.x_embedder(x) + self.pad
        for block in self.layers:
            h = block(h)
        return self.final(h)


def _bytes(n: int) -> str:
    return f"{n / 2**20:.1f} MiB"


# ── ① 分组与切分数学 ────────────────────────────────────────────────────────────


def case_plan() -> None:
    model = _Tiny()
    groups = r.survey(model)
    names = [g.name for g in groups]
    check("分组按直接子模块＋顶层裸参数（‹root› 必须在，它每步都用）",
          names[0] == "‹root›" and "layers.0" in names and names[-1] == "final", names)
    check("层组被标记成 layer（`layers.N` ✓）",
          [g.name for g in groups if g.is_layer] == ["layers.0", "layers.1", "layers.2"],
          [g.name for g in groups if g.is_layer])
    exact = sum(g.nbytes for g in groups)
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    check("字节数与 parameters 逐字节相同（不许漏 buffer / 漏裸参数）", exact == params,
          (exact, params))

    non_layer = sum(g.nbytes for g in groups if not g.is_layer)
    one_layer = groups[[g.name for g in groups].index("layers.0")].nbytes

    full = r.plan(groups, budget=exact)
    check("预算够 ⇒ 全上卡、on_cpu 空、needs_split False（全在一处就别挂 hook）",
          not full.on_cpu and not full.needs_split and full.fits, full)

    tight = r.plan(groups, budget=non_layer + one_layer + one_layer)
    check("预算只够两层 ⇒ 卡上是**连续前缀** layers.0/1（不是挑最小的塞）",
          [n for n in tight.on_gpu if n.startswith("layers.")] == ["layers.0", "layers.1"],
          tight.on_gpu)
    check("切完两层 ⇒ on_cpu 只剩 layers.2、needs_split True、层数报得对",
          list(tight.on_cpu) == ["layers.2"] and tight.needs_split
          and (tight.layers_on_gpu, tight.layers_on_cpu) == (2, 1),
          (tight.on_cpu, tight.layers_on_gpu, tight.layers_on_cpu))

    no_room = r.plan(groups, budget=non_layer - 1)
    check("预算连非层都放不下 ⇒ fits False 并点名「非层参数」（不许假装切成功）",
          not no_room.fits and "非层参数" in no_room.reason, no_room.reason)

    zero = r.plan(groups, budget=0)
    check("预算 ≤ 0 ⇒ fits False（没显存就是没显存）", not zero.fits and zero.on_gpu == (),
          zero.reason)

    #: 反向验证：把「依次上卡」改成「挑最小的塞」会怎样 —— 手工模拟，证明判据能分辨两者。
    picked = sorted([g for g in groups if g.is_layer], key=lambda g: g.nbytes)
    check("本用例的层等大 ⇒ 「挑小的」与「取前缀」结果相同，此判据靠上面顺序断言成立",
          [g.nbytes for g in picked] == [g.nbytes for g in groups if g.is_layer],
          [g.nbytes for g in picked])


# ── ②③④⑤ 真跑（有卡走 CUDA，没卡降级成 CPU↔CPU 的同设备对照）─────────────────────────


def case_split_forward() -> None:
    torch.manual_seed(5)
    model = _Tiny()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.3)
    want = model(torch.randn(2, 7, 16))

    if not HAS_CUDA:
        check("本机没有 CUDA ⇒ 跳过跨设备用例（只验切分数学）", True, "no cuda")
        return

    main = torch.device("cuda")
    groups = r.survey(model)
    non_layer = sum(g.nbytes for g in groups if not g.is_layer)
    one = next(g.nbytes for g in groups if g.name == "layers.0")
    handle = r.place(model, main_device=main, budget=non_layer + one,
                     mode="split")
    check("按预算切出「卡上 1 层 / CPU 2 层」",
          (handle.plan.layers_on_gpu, handle.plan.layers_on_cpu) == (1, 2), handle.report())
    check("split 下权重**不搬**（CPU 层就留在 CPU 上算）",
          str(next(model.layers[2].parameters()).device) == "cpu"
          and str(next(model.layers[0].parameters()).device).startswith("cuda"),
          handle.placement())

    x = torch.randn(2, 7, 16)          # 特意留在 CPU：入口 hook 负责搬到卡上
    with torch.no_grad():
        got = model(x)
    with torch.no_grad():
        ref_model = _Tiny()
        ref_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
        want = ref_model(x)
    rel = (got.cpu().double() - want.double()).abs().max().item() / max(
        want.double().abs().max().item(), 1e-12)
    check("跨设备前向与全 CPU 结果一致（< 1e-5）", rel < 1e-5, f"相对差 {rel:.3e}")
    check("真的跨过界（transfers ≥ 1）—— 全在卡上时不该涨", handle.transfers >= 1,
          handle.transfers)

    report = handle.report()
    check("报告如实：入口 1 根 ＋ 每个 CPU 层 2 根、每步搬运量报不出就不编",
          report["hooks"] == 1 + 2 * report["layersOnCpu"] and report["entryHook"]
          and report["layersOnCpu"] == 2 and report["perStepTransferMiB"] is None, report)

    #: ③ hook 不是装饰：同样的布局、不挂 hook ⇒ 必须报错。
    broken = _Tiny()
    broken.load_state_dict(model.state_dict())
    broken.to(main)
    broken.layers[2].to("cpu")
    failed = False
    try:
        with torch.no_grad():
            broken(torch.randn(2, 7, 16).to(main))
    except RuntimeError:
        failed = True
    check("同布局**不挂 hook** 必报设备不一致（证明跑通是 hook 的功劳，不是巧合）", failed)

    handle.detach()
    check("detach 之后 hook 清空、权重回到计划位置（CPU 层仍是 CPU，这是设计不是残留）",
          not handle.hooks
          and str(next(model.layers[2].parameters()).device) == "cpu", handle.hooks)


def case_park_forward() -> None:
    if not HAS_CUDA:
        check("本机没有 CUDA ⇒ 跳过 park 用例", True, "no cuda")
        return
    torch.manual_seed(9)
    model = _Tiny()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.3)
    x = torch.randn(2, 7, 16)
    with torch.no_grad():
        want = model(x)
    handle = r.place(model, main_device=torch.device("cuda"), budget=1 << 40, mode="park")
    with torch.no_grad():
        got = model(x.to("cuda"))
    rel = (got.cpu().double() - want.double()).abs().max().item() / max(
        want.double().abs().max().item(), 1e-12)
    check("park 前向与 CPU 参考一致（权重搬上卡算、算完搬回）", rel < 1e-5, f"相对差 {rel:.3e}")
    report = handle.report()
    check("park 能报出每步搬运量（搬的是权重 ⇒ 算得出来；小模型 MiB 会取整成 0 ⇒ 认字节）",
          report["perStepTransferBytes"] == 2 * handle.plan.layer_bytes
          and report["perStepTransferBytes"] > 0, report)
    check("park 跑完权重回到内存（峰值＝单层，不是整份）",
          str(next(model.layers[0].parameters()).device) == "cpu", handle.placement())
    handle.restore()
    handle.detach()


def case_no_split_no_hooks() -> None:
    torch.manual_seed(3)
    model = _Tiny()
    handle = r.place(model, main_device=torch.device("cpu"), budget=1 << 40, mode="split")
    check("全在一处（预算够）⇒ 一根 hook 都不挂", handle.hooks == [], handle.report())
    with torch.no_grad():
        model(torch.randn(2, 7, 16))
    check("同设备前向 transfers 恒为 0（老路零开销）", handle.transfers == 0, handle.transfers)
    handle.detach()


def case_layer_names() -> None:
    """⭐ 真主干的编排容器叫 ``blocks`` —— 只认 ``layers.N`` 会让联合驻留在真模型上**一层都不识别**。

    2026-10-31 实测：`dit.py` 的骨架与 `h3_form.H3FormTrunk` 的堆叠**都叫 `blocks`** ✓。
    改前 `survey` 把它们全算成"非层参数" ⇒ 非层字节 = 整份 ⇒ 任何预算都判 `fits=False`
    ⇒ 整份退回 CPU ✓✗ —— 联合驻留**静默变成纯 CPU**，正是这次要消灭的那一档 ✓✗。
    （而 `_Tiny` 这种测试模型用的是 ``layers`` ⇒ **判据全绿也发现不了** ✓✗ —— 所以这条单独立案 ✓。）
    """

    class _Blocks(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([_Block(16) for _ in range(3)])
            self.head = torch.nn.Linear(16, 16)

    groups = r.survey(_Blocks())
    check("blocks.N 也认作层（DiT 与 H3 主干的真形态）",
          [g.name for g in groups if g.is_layer] == ["blocks.0", "blocks.1", "blocks.2"],
          [g.name for g in groups if g.is_layer])
    non_layer = sum(g.nbytes for g in groups if not g.is_layer)
    one = next(g.nbytes for g in groups if g.name == "blocks.0")
    tight = r.plan(groups, budget=non_layer + one)
    check("预算 = 非层 + 一层 ⇒ 只有 blocks.0 上卡（改前会整份退回 CPU）",
          tight.on_gpu[-1] == "blocks.0" and (tight.layers_on_gpu, tight.layers_on_cpu) == (1, 2),
          (tight.on_gpu, tight.on_cpu))
    check("父名不是容器的索引**不许**被当层（`final_modulation.1` / `layers` 本身）",
          not r.is_layer("final_modulation.1") and not r.is_layer("layers"),
          (r.is_layer("final_modulation.1"), r.is_layer("layers")))
    check("模式清单只有一份（`MODES` ⇒ off/split/park；`WORK_MODES` ⇒ 真干活的两种）",
          r.MODES == ("off", "split", "park") and r.WORK_MODES == ("split", "park"), r.MODES)


def case_backend() -> None:
    """接进 `TorchBackend` 之后仍然成立：报告看得见、老路没变、切不动时**不假装**。

    盯的是"接线"（不是切分数学 ✓——那在 ① 与 ②③ 里）：装载报告里有没有 `residency`、
    `off` 时是不是**真的一字未变**、权重装载后是不是**仍停在切好的位置**（`copy_` 落在 param 的
    device 上 ✓）、以及 `applied=False` 的两种情形（没有第二台设备 / 预算为 0）会不会**如实说** ✓。
    """
    import tempfile

    from app.services.engine import dit as dit_mod
    from app.services.engine import torch_backend as tb
    from app.services.engine import weights as weights_mod

    cfg = dit_mod.DiTConfig(hidden=32, depth=2, heads=4, patch_size=(1, 2, 2),
                            in_channels=4, text_dim=16, mlp_ratio=2.0, vae_scale=8)
    root = Path(tempfile.mkdtemp(prefix="residency_wire_"))
    path = root / "tiny.safetensors"
    weights_mod.save_module_weights(dit_mod.build_dit(cfg), path)

    #: ① 老路：`off` ⇒ 报告里 None、describe 自述 off（**不是**"开了但没切"）
    plain = tb.TorchBackend(device="cpu")
    plain_report = plain.load_weights(path=str(path), config=cfg)
    check("residency=off ⇒ 装载报告里 `residency` 是 None（老路一字未变）",
          plain_report.get("residency") is None, plain_report.get("residency"))
    check("describe 自述 mode=off 且没有把手（省下的不只是显存，还有每步一次判断）",
          plain.describe()["residency"]["mode"] == "off"
          and plain.describe()["residency"]["report"] is None,
          plain.describe().get("residency"))

    #: ② CPU 设备上开 `split` ⇒ 没有第二台设备可"联" ⇒ 如实说，不假装省了显存
    cpu = tb.TorchBackend(device="cpu", residency="split")
    cpu_info = cpu.load_weights(path=str(path), config=cfg).get("residency") or {}
    check("CPU 设备上开 residency ⇒ applied=False 且点明「没有第二台设备」",
          cpu_info.get("applied") is False and "没有第二台设备" in str(cpu_info.get("note")),
          cpu_info)

    #: ③ 不认的模式**当场炸**（静默回落 off = 用户以为开了联合）
    refused = False
    try:
        tb.TorchBackend(device="cpu", residency="half")
    except ValueError:
        refused = True
    check("residency 写错 ⇒ 构造就报错（不静默当成 off）", refused)

    if not HAS_CUDA:
        check("本机没有 CUDA ⇒ 跳过真切与退化的两条", True, "no cuda")
        return

    #: ④ 真切：预算 = 非层 + 一层 ⇒ 卡上 blocks.0、CPU blocks.1，**装载后仍停在这两处**
    probe = dit_mod.build_dit(cfg)
    groups = r.survey(probe)
    non_layer = sum(g.nbytes for g in groups if not g.is_layer)
    one = next(g.nbytes for g in groups if g.name == "blocks.0")
    split = tb.TorchBackend(device="cuda", residency="split",
                            residency_budget_mib=(non_layer + one) / 2**20)
    split_info = split.load_weights(path=str(path), config=cfg).get("residency") or {}
    check("CUDA 上真切：报告 applied/fits 都真、层数 1/1",
          split_info.get("applied") is True and split_info.get("fits") is True
          and (split_info.get("layersOnGpu"), split_info.get("layersOnCpu")) == (1, 1),
          split_info)
    model = split._model
    check("装载走 copy_ ⇒ 权重停在切好的位置（不会被装回整份到卡上）",
          str(next(model.blocks[0].parameters()).device).startswith("cuda")
          and str(next(model.blocks[1].parameters()).device) == "cpu",
          [str(next(block.parameters()).device) for block in model.blocks])
    with torch.no_grad():
        out = model(torch.randn(1, cfg.in_channels, 2, 8, 8), torch.tensor(0.5),
                    torch.randn(1, 5, cfg.text_dim))
    check("装好的模型**真能跨设备前向**（hook 在装载之后仍然挂得住）",
          tuple(out.shape) == (1, cfg.in_channels, 2, 8, 8), tuple(out.shape))

    #: ⑤ 预算 0 ⇒ 整份留在内存 ⇒ 报 applied=False（不是 `fits=True` 就当切过了）
    zero = tb.TorchBackend(device="cuda", residency="split", residency_budget_mib=0)
    zero_info = zero.load_weights(path=str(path), config=cfg).get("residency") or {}
    check("预算 0 ⇒ applied=False 且点明「等于纯 CPU」（不许报成切成功）",
          zero_info.get("applied") is False and "纯 CPU" in str(zero_info.get("note")),
          zero_info)


def case_budget_from_cuda() -> None:
    budget = r.budget_from_cuda(torch)
    if HAS_CUDA:
        check("预算取「空闲显存 ×(1−余量)」且大于 0", budget > 0, _bytes(budget))
        check("默认余量 25%（激活十几 GiB 的族不能只留 5%）", 0.2 <= r.DEFAULT_RESERVE <= 0.35,
              r.DEFAULT_RESERVE)
    else:
        check("没卡 ⇒ 预算 0（不许瞎猜一个数）", budget == 0, budget)


def main() -> int:
    case_plan()
    case_layer_names()
    case_split_forward()
    case_park_forward()
    case_no_split_no_hooks()
    case_backend()
    case_budget_from_cuda()
    width = max(len(name) for name, _, _ in _RESULTS)
    fails = 0
    for name, ok, detail in _RESULTS:
        mark = "ok  " if ok else "FAIL"
        line = f"{mark} {name.ljust(width)}"
        if not ok:
            fails += 1
            line += f"  <<< {detail}"
        print(line)
    print(f"\nSUMMARY: {len(_RESULTS) - fails}/{len(_RESULTS)} passed"
          + ("" if not fails else f"  ·  FAILS {fails}"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
