"""``scheduler``（GPU 优先 ⇒ 不足则 GPU/CPU 协同 ⇒ 都不够就拒）的判据 ✓。

盯六类事：

1. **指令档不许被估算覆盖** ✓：策略 ``cpu`` / 显式 ``cpu`` / 策略 ``gpu`` / 没卡 —— 这四档先于
   显存读数决定 ✓✗（否则「用户明确要 CPU」会被「卡上还空着」劝回去 ✓）；
2. **协同的两本账** ✓：切分后 ``gpuBytes + cpuBytes`` 必须**正好等于**需求 ✓（少算就是采样期 OOM ✓），
   且卡上只给到「余量 − 1 GiB」✓（再挤一步就炸 ✓）；
3. **第二台设备也要核** ✓✗：内存放不下 CPU 那一半 ⇒ **加卡减内存** ✓，加完还放不下 ⇒
   :class:`InsufficientResources` ✓（不许交给交换文件慢慢拖 ✓）；
4. **滞回** ✓：单次超硬阈值不动作 ✓、连续两次才动作 ✓、回落立即归位 ✓、读不到压力不改状态 ✓；
5. **回收判据是纯函数** ✓：``reclaim_for`` 在「阶段表没登记」时一个都不退 ✓✗（猜着退就会卸掉在用的 ✓）；
6. **后端真的接了这几件** ✓：``describe()["scheduling"]`` 有账本与观察者 ✓、``tick()`` 在 CPU 后端
   不探显存 ✓、``_host_guard`` 巨量需求当场拒 ✓、事件表**限长** ✓（每步都探 ⇒ 无上限会吃光内存 ✓✗）。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.services.engine import device_policy as dp  # noqa: E402
from app.services.engine import scheduler as s  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []

GIB = s.GIB


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _press(total_gib: float, free_gib: float) -> dp.Pressure:
    return dp.Pressure(totalBytes=int(total_gib * GIB), freeBytes=int(free_gib * GIB))


def _host(total_gib: float, free_gib: float) -> tuple[int, int]:
    return int(free_gib * GIB), int(total_gib * GIB)


# ── ① 指令档（先于估算）───────────────────────────────────────────────────────


def case_orders() -> None:
    need = int(20 * GIB)
    wide = _press(22.49, 21.0)          # 卡上很空
    tight = _press(22.49, 0.2)          # 卡上快满了

    slots = s.assign(need, pressure=wide, host=_host(64, 60), policy="cpu")
    check("策略 cpu ⇒ 整份内存，且**不看**显存（卡上还空着也不回去）",
          (slots.device, slots.residency, slots.degraded) == ("cpu", "off", False),
          slots.to_dict())

    slots = s.assign(need, pressure=wide, host=_host(64, 60), requested_device="cpu")
    check("显式 device=cpu ⇒ 同样整份内存", slots.device == "cpu", slots.to_dict())

    slots = s.assign(need, pressure=tight, host=_host(8, 1), policy="gpu")
    check("策略 gpu ⇒ 再紧也上卡（不够就 OOM，不悄悄转移）",
          (slots.device, slots.residency, slots.cpuBytes) == ("cuda", "off", 0),
          slots.to_dict())

    slots = s.assign(need, pressure=None, host=_host(64, 60), gpu_available=False)
    check("没卡 ⇒ 整份 CPU 且如实标降级", (slots.device, slots.degraded) == ("cpu", True),
          slots.to_dict())

    slots = s.assign(need, pressure=None, host=_host(64, 60), gpu_available=False,
                     policy="gpu")
    check("没卡但策略 gpu ⇒ 照上卡（让装载当场失败，不假装成功）",
          slots.device == "cuda", slots.to_dict())

    bad = ""
    try:
        s.assign(need, pressure=wide, host=_host(64, 60), policy="gpuu")
    except ValueError as err:
        bad = str(err)
    check("策略名拼错 ⇒ 当场 ValueError（不静默按默认跑）", bool(bad), bad)

    slots = s.assign(need, pressure=wide, host=_host(64, 60), policy="GPU-FIRST")
    check("`gpu-first` 是 `auto` 的别名（不算拼错）",
          slots.policy == "auto" and slots.device == "cuda", slots.to_dict())


# ── ② 协同账 / ③ 内存侧也要核 ─────────────────────────────────────────────────


def case_split() -> None:
    need = int(25.28 * GIB)             # H3 的 32B 文本塔：卡上装不下
    free = 21.28
    pressure = _press(22.49, free)
    slots = s.assign(need, pressure=pressure, host=_host(93, 70))

    check("装不下但能切 ⇒ 判协同（split）",
          (slots.device, slots.residency) == ("cuda", "split"), slots.to_dict())
    check("协同后两边之和**正好**等于需求（少算就是采样期 OOM）",
          slots.gpuBytes + slots.cpuBytes == need, slots.to_dict())
    check("卡上只给到「余量 − 1 GiB」—— 再挤一步就炸",
          slots.gpuBytes == int(free * GIB) - s.GPU_HEADROOM_BYTES, slots.to_dict())
    check("被动切分如实标 degraded=True", slots.degraded is True, slots.to_dict())

    #: ⚠️ 反向：内存放不下 CPU 那一半 ⇒ **拒** ✓✗（卡上早已给到「余量 − 1 GiB」✓，
    #: 没有"再加卡顶上"的空间 ✓ —— 早先那个 `extra_gpu` 死分支已删 ✓）。
    tight_host = _host(16, 3.0)         # 3 GiB 可用，扣 2 GiB 余量 ⇒ 只放得下 1 GiB
    raised = ""
    try:
        s.assign(need, pressure=pressure, host=tight_host)
    except s.InsufficientResources as err:
        raised = str(err)
    check("内存放不下 CPU 那一半 ⇒ 拒（并算出 卡上 + 内存 < 需求）",
          "合计" in raised and "拒跑" in raised, raised[:160])

    #: 同样两边都紧、但卡的余量更小 ⇒ 仍然拒 ✓（不同余量读数不该改结论 ✓）
    raised = ""
    try:
        s.assign(int(40 * GIB), pressure=_press(22.49, 1.0), host=_host(16, 2.0))
    except s.InsufficientResources as err:
        raised = str(err)
    check("两边都放不下 ⇒ InsufficientResources（不靠交换文件硬跑）", bool(raised), raised[:160])

    slots = s.assign(need, pressure=_press(22.49, 0.3), host=_host(93, 70))
    check("卡上连有意义的块都放不下 ⇒ 整件转 CPU 且标降级",
          (slots.device, slots.residency, slots.degraded) == ("cpu", "off", True),
          slots.to_dict())

    raised = ""
    try:
        s.assign(need, pressure=_press(22.49, 0.3), host=_host(16, 2.0))
    except s.InsufficientResources as err:
        raised = str(err)
    check("卡切不出 + 内存也没有 ⇒ 拒", bool(raised), raised)


def case_balanced() -> None:
    #: ⚠️ 「装得下」与「已用 ≥80%」要同时成立 ⇒ 件必须**小** ✓：8 GiB 的件在 22.49 GiB 的卡上
    #: 一旦要留 1 GiB 余量 ✓，余量最大也只有 8.5 GiB ⇒ 占用率必然 < 66% ✓ ⇒ 永远测不到这一档 ✗✗
    #: （第一版就写成了 8 GiB ✗ —— 它走的是「装不下」那条被动切分 ✓，标签自然是 `degraded=True` ✓）。
    need = int(2 * GIB)
    hot = _press(22.49, 4.4)            # 已用 ≈80.4% 且 4.4 ≥ 2 + 1 ⇒ 既装得下、又过软阈值
    slots = s.assign(need, pressure=hot, host=_host(93, 70), policy="balanced")
    check("balanced 装得下但已用 ≥80% ⇒ 主动协同", slots.residency == "split",
          slots.to_dict())
    check("主动协同**不是降级** ⇒ degraded=False", slots.degraded is False,
          slots.to_dict())

    slots = s.assign(need, pressure=_press(22.49, 20.0), host=_host(93, 70),
                     policy="balanced")
    check("balanced 压力不高 ⇒ 照常整份上卡，不无谓切", slots.residency == "off",
          slots.to_dict())


# ── ④ 宿主内存：检测 + 闸口 ───────────────────────────────────────────────────


def case_host() -> None:
    check("内存余量 = max(2 GiB, 8%)（大机器按比例、小机器按下限）",
          (s.host_reserve_bytes(int(93.6 * GIB)) > 2 * GIB
           and s.host_reserve_bytes(int(8 * GIB)) == int(2 * GIB)),
          [round(s.host_reserve_bytes(int(n * GIB)) / GIB, 2) for n in (8, 93.6)])

    read = s.host_memory()
    check("本机能读到（可用, 总量）且 0 ≤ 可用 ≤ 总量（不是提交额度）",
          read is not None and 0 <= read[0] <= read[1] and read[1] > 0,
          None if read is None else [round(x / GIB, 1) for x in read])

    total, free = int(93.6 * GIB), int(60 * GIB)
    check("余量要扣掉给系统留的那份",
          s.host_room_bytes((free, total)) == free - s.host_reserve_bytes(total),
          s.host_room_bytes((free, total)))

    got = s.ensure_host_room(int(8 * GIB), host=(int(60 * GIB), total), what="t")
    check("放得下 ⇒ checked=True 并报可用量", got["checked"] is True, got)

    bad = ""
    try:
        s.ensure_host_room(int(80 * GIB), host=(int(4 * GIB), total), what="t")
    except s.InsufficientResources as err:
        bad = str(err)
    check("放不下 ⇒ InsufficientResources 且点名要多少 / 只有多少", bool(bad), bad)

    saved = s.host_memory
    s.host_memory = lambda: None          # 冒充"这台机器读不到物理内存"
    try:
        got = s.ensure_host_room(int(80 * GIB), what="t")
    finally:
        s.host_memory = saved
    check("读数取不到 ⇒ 放行但如实说没核（不假装核过，也不假装拒过）",
          got["checked"] is False and "没核" in str(got.get("note")), got)


# ── ⑤ 账本 ──────────────────────────────────────────────────────────────────


def case_ledger() -> None:
    ledger = s.Ledger()
    ledger.reserve("dit", bytes_=int(17.94 * GIB), device="cuda", phase="sample")
    ledger.reserve("text#host", bytes_=int(4.4 * GIB), device="cpu")
    check("账本两侧分开算",
          (ledger.gpuBytes, ledger.ramBytes) == (int(17.94 * GIB), int(4.4 * GIB)),
          ledger.to_dict())

    other = s.Ledger()
    other.reserve("x", bytes_=7, device="cuda:0")
    check("cuda:0 也算卡上", other.gpuBytes == 7, other.to_dict())

    ledger.reserve("dit", bytes_=int(1 * GIB), device="cuda")
    check("同名再记 ⇒ 覆盖而不是叠加（一件东西只有一份现状）",
          ledger.gpuBytes == int(1 * GIB), ledger.to_dict())

    entry = ledger.release("dit")
    check("释放 ⇒ 真的从账上消失", entry is not None and "dit" not in ledger.items,
          ledger.to_dict())

    ledger.clear()
    check("清空 ⇒ 两栏归零", ledger.gpuBytes == 0 and ledger.ramBytes == 0, ledger.to_dict())


# ── ⑥ 运行期滞回 ─────────────────────────────────────────────────────────────


def case_watch() -> None:
    watch = s.LoadWatch()
    #: 三段要落进 **软硬之间的那 7 个百分点** ✓✗（88% ≤ 占用 < 95%）：22.49 GiB 的卡上就是
    #: 余量 2.25 ~ 1.12 GiB ✓ —— 第一版取 4.0 GiB（≈82%）落在软阈值**以下** ✓，测的其实是 normal ✗。
    hot = _press(22.49, 0.5)            # ≈98% ⇒ 硬以上
    warm = _press(22.49, 2.0)           # ≈91% ⇒ 软硬之间
    cool = _press(22.49, 12.0)          # ≈47%

    watch.sample(hot)
    check("单次超硬阈值**不动手**（采样期抖动天然存在）", watch.wants_reclaim is False,
          watch.to_dict())
    watch.sample(hot)
    check("连续两次超硬阈值 ⇒ 才认 hard", watch.wants_reclaim is True, watch.to_dict())
    watch.sample(cool)
    check("回落到软阈值以下 ⇒ 立即回 normal（保守方向不延迟）",
          (watch.state, watch.wants_reclaim) == ("normal", False), watch.to_dict())
    watch.sample(warm)
    check("软硬之间 ⇒ pressured 只记录、不动手",
          (watch.state, watch.wants_reclaim) == ("pressured", False), watch.to_dict())
    state_before, source_before = watch.state, watch.to_dict()["source"]
    watch.sample(None)
    check("读不到压力 ⇒ 状态不动，且读数来源照实报",
          (watch.state, watch.to_dict()["source"]) == (state_before, source_before),
          watch.to_dict())

    quiet = s.LoadWatch()
    quiet.sample(None)
    check("从没读到过压力 ⇒ source=unknown（测不到 ≠ 没压力）",
          quiet.to_dict()["source"] == "unknown" and quiet.state == "normal",
          quiet.to_dict())


# ── ⑦ 回收判据（纯函数）──────────────────────────────────────────────────────


def case_reclaim() -> None:
    present = {"dit": True, "text": True, "vae": False}
    check("阶段表没登记 ⇒ 一个都不退（不猜着卸）",
          s.reclaim_for("sample", present, None) == [], present)
    check("sample 只要 dit ⇒ 退掉 text（vae 本来就不在场）",
          s.reclaim_for("sample", present, ("dit",)) == ["text"], present)
    check("本阶段要的都留着",
          s.reclaim_for("encode", present, ("dit", "text")) == [], present)


# ── ⑧ 后端接线 ──────────────────────────────────────────────────────────────


class _Plan:
    """冒充 `residency.Plan` ✓：闸口只读 ``cpu_bytes`` ✓。"""

    def __init__(self, cpu_bytes: int) -> None:
        self.cpu_bytes = cpu_bytes


def case_backend() -> None:
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    backend = tb.TorchBackend(device="cpu")
    scheduling = backend.describe().get("scheduling") or {}
    check("describe().scheduling 有口径名、账本、观察者、事件表",
          scheduling.get("policy") == s.POLICY_NAME
          and set(scheduling) >= {"policy", "ledger", "watch", "events"}, scheduling)
    check("口径名恒为 gpu-first（不是可选项）", s.POLICY_NAME == "gpu-first", s.POLICY_NAME)

    check("CPU 后端上 tick() 不探显存 ⇒ 返回 None（不炸、不瞎报压力）",
          backend.tick("sample") is None, backend.describe().get("scheduling"))

    guard = backend._host_guard("探针")  # noqa: SLF001 - 验的就是这条闸 ✓
    problems = ""
    try:
        guard(_Plan(0))
    except Exception as err:  # noqa: BLE001
        problems = f"意外：{err}"
    check("闸口：CPU 侧为 0 ⇒ 直接放行（不白跑一次读数）", problems == "", problems)

    raised = ""
    try:
        guard(_Plan(10 ** 6 * GIB))     # 100 万 GiB ⇒ 必然放不下
    except s.InsufficientResources as err:
        raised = str(err)
    check("闸口：放不下 ⇒ InsufficientResources（装载中止，不是先搬进去再说）",
          bool(raised), raised)

    for index in range(205):
        backend._record_event("probe", {"i": index})  # noqa: SLF001
    events = backend.describe()["scheduling"]["events"]
    check("事件表限长（每步都探 ⇒ 无上限会自己把内存吃光）",
          len(events) <= 200 and bool(events),
          {"kept": len(events), "tail": events[-1].get("i")})


def case_module_selftest() -> None:
    problems = s.selftest()
    check("模块自检全绿（够 / 切 / 拒 / 没卡 / gpu / cpu / 滞回 / 账本）",
          problems == [], problems)


def main() -> int:
    case_orders()
    case_split()
    case_balanced()
    case_host()
    case_ledger()
    case_watch()
    case_reclaim()
    case_backend()
    case_module_selftest()
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
