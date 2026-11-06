"""⑧ 内存这一级的判据（`host_cache` ＋ `residency` 的卸/填 ＋ `weights` 的跳过/重读 ✓）。

三级内存里**只有这一级原先没有上限** ✗：`residency.survey()` 遍历全部参数 ⇒ 整份常驻 ✓
⇒ `_host_guard` 核不过就只剩一条路 —— 抛 `InsufficientResources` 中止装载 ✓✗。
补上之后要盯五件事，少一件就会得到"报告说卸了、其实没卸"或"卸了再也读不回来"：

1. **决策对**（`host_cache` ✓）：装得下不动别人、装不下按 LRU 只卸**层**、卸不够就一件都不卸 ✓；
2. **顺序对**（`residency._host_prepare` ✓）：快照**先于**卸载 ⇒ 重读回来能核形状 ✓
   （反了就会拿 `(0,)` 当基准 ✓✗）；
3. **装载对**（`weights._load_streaming` ✓）：卸过的键**不读不写** ✓、且**不许**因为占位是
   `(0,)` 被判成 `shapeMismatch` ✓✗（那会让"内存不够"变成"整份装不上" ✓）；
4. **重读真把数填回去**（端到端 ✓）：装载 → 前向 → 逐元素等于整份常驻那条路 ✓✗
   （这条是整份判据里唯一能抓"卸了但没填"的 ✓）；
5. **不启用时零变化**（`report()["hostCache"] is None` ✓）：不给配额 ⇒ 一层都不卸、一根 hook 都不多 ✓。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from app.services.engine import host_cache as hc  # noqa: E402
from app.services.engine import residency as r  # noqa: E402
from app.services.engine import scheduler  # noqa: E402
from app.services.engine import weights as w  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ── 被测模型（4 个组 ＋ 3 个层组；每层 4 个参数 ⇒ 卸一层就要跳过 4 个键）──────────────


class _Block(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(dim, dim)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.lin(x)) + x


class _Tiny(torch.nn.Module):
    def __init__(self, dim: int = 32, depth: int = 3) -> None:
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


def _tight_quota(model: torch.nn.Module) -> tuple[int, int]:
    """算出「正好装不下最后两个层组」的配额 ✓ ⇒ 返回 ``(配额, 单层字节)`` ✓。

    按 `survey` 的顺序（‹root› → x_embedder → layers.0… → final ✓）走配额账，
    「两个组被卸」是**算出来的**，不是写死的数 ✓✗（写死字节数会让这个测试变成
    "记不记得住 4352 这个神奇数字" ✓）。
    """
    groups = r.survey(model)
    non_layer = sum(g.nbytes for g in groups if not g.is_layer)
    layer = min(g.nbytes for g in groups if g.is_layer)
    return non_layer + 2 * layer - 1, layer


def _make_pair(tmp: Path, name: str) -> tuple[torch.nn.Module, torch.nn.Module, Path]:
    """``(参考模型, 待装模型, 权重文件)`` ✓ —— 参考模型的权重先落盘 ✓，右边那份从盘上装 ✓。

    ⚠️ 每个用例一份**独立文件名** ✓✗：Windows 上重读是 mmap 打开的 ✓，同一个路径再写一次
    会撞「拒绝访问」✓ —— 那是**上一例还没退件**留下的句柄 ✓，不是这里写坏了 ✓。
    """
    torch.manual_seed(1234)
    reference = _Tiny().eval()
    path = Path(w.save_module_weights(reference, tmp / f"{name}.safetensors"))
    return reference, _Tiny().eval(), path


# ── ① 配额算术 ────────────────────────────────────────────────────────────────


def case_quota() -> None:
    check("⑧ 配额＝总内存 × 75%（不是拿 free 现算：那是「此刻」、不是「上限」）",
          hc.quota_of(8 * 2**30) == int(8 * 2**30 * hc.HOST_USE_RATIO),
          hc.quota_of(8 * 2**30))
    check("⑧ 小内存补下限（4 GiB × 0.75 在 1 GiB 的机器上等于没有）",
          hc.quota_of(1024) == hc.MIN_QUOTA_BYTES, hc.quota_of(1024))
    check("⑧ ratio=1.0 不夹（「我明确要全部」不许被静默改掉）",
          hc.quota_of(2**30, ratio=1.0) == 2**30, hc.quota_of(2**30, ratio=1.0))
    raised: BaseException | None = None
    try:
        hc.quota_of(2**30, ratio=0)
    except ValueError as exc:  # noqa: PERF203 —— 就是要抓它 ✓
        raised = exc
    check("⑧ ratio=0 当场拒（配额 0 ＝ 一层都不许留 ⇒ 一定要报，不许静默）",
          isinstance(raised, ValueError), raised)


# ── ② 淘汰决策（三条硬约束 ✓）────────────────────────────────────────────────


def case_decide() -> None:
    cache = hc.HostCache(quota=100, name="decide")
    first = cache.admit("layers.0", 40, layers=1)
    check("⑧ 装得下 ⇒ resident 且不动别人", first.status == hc.RESIDENT
          and not first.victims, first.to_dict())
    cache.admit("layers.1", 40, layers=1)
    cache.touch("layers.0")
    third = cache.admit("layers.2", 40, layers=1)
    check("⑧ 装不下 ⇒ 按 LRU 卸（刚用过的 layers.0 不许被挑中）",
          third.victims == ("layers.1",), third.victims)
    check("⑧ 卸完账对得上（resident 就占 80）", cache.used_bytes == 80, cache.used_bytes)

    pinned = hc.HostCache(quota=100, name="decide-pin")
    pinned.admit("x_embedder", 60, layers=0)
    pinned.admit("layers.0", 30, layers=1)
    pinned.touch("layers.0")
    verdict = pinned.admit("layers.1", 30, layers=1)
    check("⑧ 钉住的组（非层）永不被卸 —— 每步都用的东西卸掉就是抖动的定义",
          "x_embedder" not in verdict.victims and verdict.status == hc.RESIDENT,
          verdict.to_dict())

    stuck = hc.HostCache(quota=100, name="decide-stuck")
    stuck.admit("layers.0", 30, layers=1)
    blocked = stuck.admit("big", 200, layers=1)
    check("⑧ 卸了也不够 ⇒ 一件都不卸（白卸一场什么都没换来）",
          blocked.status == hc.SPILLED and not blocked.victims, blocked.to_dict())
    check("⑧ 不白卸：原位那层还在", "layers.0" in stuck.resident(), stuck.resident())

    snap = stuck.snapshot()
    check("⑧ 快照能看出「卸了几层、卸过几次、配额多少」",
          set(snap) >= {"quotaGiB", "usedGiB", "layers", "spilled", "evicted", "restores"}
          and snap["spilled"] == ["big"], snap)


# ── ③ 端到端：卸 → 装载跳过 → 前向重读 → 数值不变 ─────────────────────────────


def case_spill_restore(tmp: Path) -> None:
    reference, model, path = _make_pair(tmp, "e2e-spill")
    quota, layer_bytes = _tight_quota(model)
    cache = hc.HostCache(quota=quota, name="e2e")
    slot = r.ReloadSlot()
    param_ref = model.layers[0].lin.weight     #: 就地换 data 的凭据 ✓（不是 `del`/`setattr(None)` ✓）

    handle = r.place(model, main_device="cpu", budget=0, mode="split",
                     host_cache=cache, reload=slot)
    report = handle.report()
    check("⑧ 配额紧到装不下 ⇒ 真卸了两层（不是「记了个数」）",
          sorted(handle.spilled) == ["layers.0", "layers.1"], sorted(handle.spilled))
    check("⑧ 只卸层：‹root›／x_embedder／final 一个都不许动",
          not (set(handle.spilled) & {"‹root›", "x_embedder", "final"}),
          sorted(handle.spilled))
    check("⑧ 报告里 hostCache 不是 None（真卸了却报 null ＝ 假账）",
          isinstance(report.get("hostCache"), dict)
          and report["hostCache"]["spilled"] == ["layers.0", "layers.1"],
          report.get("hostCache"))
    check("⑧ 卸掉的字节单独记账且**逐字节对得上**（不是只记了个层名）",
          int(handle.spillBytes) == 2 * int(layer_bytes),
          (handle.spillBytes, layer_bytes))
    check("⑧ 快照**先于**卸载（形状是 (32,32) 而不是 (0,)）",
          handle.hostMeta["layers.0"]["lin.weight"][0] == (32, 32),
          handle.hostMeta["layers.0"]["lin.weight"][0] if "layers.0" in handle.hostMeta else None)
    check("⑧ 模块上留了两个约定（装载靠它们跳过、装完靠它们接还原器）",
          getattr(model, "_residency_spilled", None) == ("layers.0", "layers.1")
          and getattr(model, "_residency_reload", None) is slot,
          (getattr(model, "_residency_spilled", None), type(getattr(model, "_residency_reload", None))))

    #: 装载：**必须**跳过被卸的 8 个键（每层 4 个）✓，且不许把它们判成 shapeMismatch ✗
    loaded = w.load_module_weights(model, path, read_ahead=0)
    check("⑧ 装载成功（卸过的组不再让它半装中止）",
          loaded.error == "" and not loaded.shapeMismatch, (loaded.error, loaded.shapeMismatch[:2]))
    check("⑧ 跳过的键单独记账（8 ＝ 2 层 × 4 参数），不混进 missing",
          loaded.skipped == 8 and loaded.missing == [], (loaded.skipped, loaded.missing))
    check("⑧ 还原器造出来了（没有它，卸了的层再也读不回来）",
          loaded.restorer is not None, type(loaded.restorer).__name__)
    check("⑧ 装载后重读入口已绑定（ReloadSlot 被 weights 接上）", slot.bound is True, slot.bound)

    x = torch.arange(32, dtype=torch.float32).reshape(1, 32) / 7.0
    with torch.no_grad():
        want = reference(x)
        got = model(x)
    check("⑧ 前向数值与整份常驻**逐元素相同**（卸了又填回来，不是静默算错）",
          torch.equal(got, want), (float((got - want).abs().max()), handle.restores))
    check("⑧ 重读真的发生了（每层一次 ⇒ ≥ 2）", handle.restores >= 2, handle.restores)
    check("⑧ 卸/填都是就地换 data（Parameter 对象自始至终是同一个，hook 表才一直有效）",
          model.layers[0].lin.weight is param_ref
          and model.layers[0].lin.weight.dtype == torch.float32,
          (type(model.layers[0].lin.weight), model.layers[0].lin.weight.dtype))

    with torch.no_grad():
        again = model(x)
    check("⑧ 第二次前向仍然逐元素相同（第二次要重新读盘，不是靠上一轮的残留）",
          torch.equal(again, want), float((again - want).abs().max()))
    check("⑧ 每趟都记账（restores 单调涨）", handle.restores >= 4, handle.restores)
    check("⑧ 单层字节与 survey 一致（配额不是靠猜出来的）",
          layer_bytes > 0 and cache.quota == quota, (layer_bytes, cache.quota))

    handle.detach()


# ── ④ 不启用时零变化 ─────────────────────────────────────────────────────────


def case_no_cache(tmp: Path) -> None:
    reference, model, path = _make_pair(tmp, "e2e-nocache")
    handle = r.place(model, main_device="cpu", budget=0, mode="split")
    report = handle.report()
    check("⑧ 不给配额 ⇒ 一层都不卸、报告里整个字段是 null（不是 evicted=0）",
          report.get("hostCache") is None and not handle.spilled, report.get("hostCache"))
    loaded = w.load_module_weights(model, path, read_ahead=0)
    check("⑧ 不给配额 ⇒ 装载一个键都不跳（老路逐字未变）",
          loaded.skipped == 0 and loaded.error == "", (loaded.skipped, loaded.error))
    x = torch.arange(32, dtype=torch.float32).reshape(1, 32) / 7.0
    with torch.no_grad():
        check("⑧ 老路数值一致", torch.equal(model(x), reference(x)), "")
    handle.detach()

    raised: BaseException | None = None
    try:
        r.place(_Tiny(), main_device="cpu", budget=0, host_cache=hc.HostCache(quota=1))
    except ValueError as exc:  # noqa: PERF203 —— 就是要抓它 ✓
        raised = exc
    check("⑧ 给了配额却没给 reload ⇒ 当场拒（只记账不重读 ＝ 换成 0 元素之后永不回填）",
          isinstance(raised, ValueError), raised)


# ── ⑤ 没接上还原器时必须炸，不许静默算错 ──────────────────────────────────────


def case_unbound() -> None:
    torch.manual_seed(7)
    model = _Tiny().eval()
    quota, _layer = _tight_quota(model)
    handle = r.place(model, main_device="cpu", budget=0, mode="split",
                     host_cache=hc.HostCache(quota=quota, name="unbound"),
                     reload=r.ReloadSlot())
    raised: BaseException | None = None
    try:
        with torch.no_grad():
            model(torch.zeros(1, 32))
    except RuntimeError as exc:  # noqa: PERF203 —— 就是要抓它 ✓
        raised = exc
    check("⑧ 还原器没绑 ⇒ 前向当场抛（不返回空字典、不静默算错）",
          isinstance(raised, RuntimeError) and "重读入口没接上" in str(raised), raised)
    handle.detach()


# ── ⑥ 后端接线：每件一本、读不到内存就不启用 ──────────────────────────────────


def case_backend(monkeypatch_free: bool = True) -> None:
    from app.services.engine.torch_backend import TorchBackend  # noqa: PLC0415 —— 只为拿一个类 ✓

    backend = TorchBackend.__new__(TorchBackend)
    backend._loadEvents = []           # noqa: SLF001 —— 只借 `_record_event` 的账 ✓
    backend._hostCaches = {}           # noqa: SLF001

    original = scheduler.host_memory
    try:
        scheduler.host_memory = lambda: (48 * 2**30, 64 * 2**30)  # type: ignore[assignment]
        dit = backend._host_cache_of("dit")      # noqa: SLF001
        te = backend._host_cache_of("te")        # noqa: SLF001
        expect = min(hc.quota_of(64 * 2**30), 48 * 2**30 - scheduler.host_reserve_bytes(64 * 2**30))
        check("⑧ 配额取 min（总量×0.75, 此刻剩余）（两个数各管一头）",
              isinstance(dit, hc.HostCache) and dit.quota == expect,
              (getattr(dit, "quota", None), expect))
        check("⑧ **每件一本**（dit 与 te 的组名都叫 blocks.0，合成一本会卸掉对方的层）",
              dit is not te and backend._host_cache_of("dit") is dit,   # noqa: SLF001
              (dit.name, getattr(te, "name", None)))
        check("⑧ 建表记一条事件（配额从哪来、读到了多少，报告里查得到）",
              any(event.get("kind") == "host-cache" for event in backend._loadEvents),  # noqa: SLF001
              backend._loadEvents)  # noqa: SLF001

        backend._hostCaches = {}       # noqa: SLF001 —— 重来一次测"读不到" ✓
        scheduler.host_memory = lambda: None  # type: ignore[assignment]
        none = backend._host_cache_of("dit")   # noqa: SLF001
        check("⑧ 内存读数取不到 ⇒ 这一级**不启用**（不拿假数去卸层）", none is None, none)
        check("⑧ 不启用也照样缓存结论（别每次 place 都去探一次）",
              "dit" in backend._hostCaches and backend._hostCaches["dit"] is None,  # noqa: SLF001
              backend._hostCaches)  # noqa: SLF001
    finally:
        scheduler.host_memory = original  # type: ignore[assignment]


# ── 收尾 ─────────────────────────────────────────────────────────────────────


def main() -> int:
    case_quota()
    case_decide()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_spill_restore(root)
        case_no_cache(root)
    case_unbound()
    case_backend()

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
