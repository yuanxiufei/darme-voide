"""层预取上卡（``foresight_swap`` ✓）—— **额外槽 + 独立拷流 + copy 没落地就不用** ✓。

来源：``include/strata/core/foresight_swap.hpp`` ✓。那边解决的是 MoE 的
"专家刚错过"✓：每个专家层在每张卡上留**几个与缓存不相干的槽** ✓，用**自己的拷流**把刚错过的
专家补进去 ✓；池子对每一个"否则要交 CPU"的专家先问一句"有没有哪个槽的 copy 已经落地" ✓，
落地了它就变成普通的 GPU 组 ✓（plan 指哪算哪 ⇒ 核函数一行不改 ✓）。

本仓（DiT 扩散）的对应物是**层**而不是专家 ✓：

* **额外槽** ＝ 卡上单独一块够装**一整层权重**的显存 ✓（与"常驻层"分开记账 ✓
  —— 它不改变 plan ✓，只在计划之外多一块可用的地 ✓）；
* **独立拷流** ＝ 一条不与计算流同步的 ``torch.cuda.Stream`` ✓（拷贝与上一层计算并行 ✓）；
* **copy 没落地就不用** ＝ 进这一层时先问那个 ``Event`` ✓：落地了才把权重换成槽里的 ✓，
  没落地就**照原计划在 CPU 上算** ✓（不是等 ✗、不是猜 ✗、更不是拿半份数据算 ✗✗）。

⚠️ **为什么"没落地就不用"是硬约束而不是优化** ✗✗：槽里的字节由拷贝填 ✓，拷一半就拿来算
会得到**不报错的错数** ✓（本仓最不能接受的那一类 ✓）。所以判定只有一条：``event.query()``
为真且槽里装的**就是这一组**（流水号对得上 ✓）⇒ 可用 ✓；否则一律走原路 ✓。

⚠️ **默认关** ✓✗（与参考实现逐字同一条纪律：``SLOTS > 0`` 才存在，不设时引擎一字不变 ✓）：
它花的是一整层的显存 ✓，而"该不该多占这一层"是策略问题不是正确性问题 ✓
—— 由 ``place(foresight=…, )`` 显式打开 ✓，并且**从预算里扣掉** ✓（否则装配期就会 OOM ✗）。

⚠️ 本模块**不碰张量、不 import torch** ✓（只做"该不该发、发没发、落地没落地、能不能用" ✓）：
分配与拷贝在 :mod:`residency` 里 ✓（那里才是唯一允许碰 torch 的地方 ✓）。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

#: 槽的四个状态 ✓（``missed`` ＝ 拷贝失败了/被丢弃 ✓ —— 与"正在拷"分开 ✓：前者不该再等 ✓，
#: 后者下一拍可能就好了 ✓，混成一个状态会让"拷贝失败"变成"永久 pending"✓✗）。
EMPTY = "empty"
LOADING = "loading"
READY = "ready"
MISSED = "missed"

#: 一个窗口里最多发几次拷贝 ✓（参考实现 ``STRATA_FS_BUDGET`` 默认 32 ✓；我们一次前向只
#: 走一遍层 ⇒ 默认 ``None`` ＝ 不设上限 ✓ —— 只有"一层要发十几份"的族才需要它 ✓）。
DEFAULT_COPIES_PER_WINDOW: int | None = None


@dataclass
class Slot:
    """一个槽 ✓（同一时刻只装一组 ✓）。"""

    group: str = ""
    token: int = 0
    nbytes: int = 0
    state: str = EMPTY
    taken: int = 0          # 被**用**过几次（落地且换成槽里那份 ✓）
    predicted: bool = False  # 这一次是"看一眼就发"还是"刚错过才发"（报告分得开 ✓）

    def to_dict(self) -> dict[str, Any]:
        return {"group": self.group, "state": self.state, "nbytes": self.nbytes,
                "taken": self.taken, "predicted": self.predicted, "token": self.token}


@dataclass
class Foresight:
    """槽表 ＋ 判据 ✓（线程安全：前向线程发、拷流回调报落地 ✓ ⇒ 两把线程都会碰 ✓）。

    ⚠️ ``slots`` **不是**"每层几个" ✗：本仓一个槽就装**一整层** ✓，所以 ``slots=1`` 就是
    "往前看一层" ✓（``2`` ＝ 两层流水 ✓，代价是两层的显存 ✓）。参考实现那边是"每层几个专家" ✓，
    两者名字一样、意思不一样 ✓（写清楚，免得照抄参数 ✗）。
    """

    slots: int = 0
    copiesPerWindow: int | None = DEFAULT_COPIES_PER_WINDOW
    #: 少于这么多字节的组不值得占一个槽 ✓（拷贝开销 + 槽的碎片 ✓）
    minBytes: int = 1 << 20
    #: 启动时为这些槽预留的显存 ✓（报告用；真正的分配在 `residency` ✓）
    reserveBytes: int = 0

    _table: list[Slot] = field(default_factory=list)
    _seq: int = 0
    _copiesThisWindow: int = 0
    _lock: Any = field(default_factory=threading.Lock)
    hits: int = 0
    pending: int = 0
    missed: int = 0
    busy: int = 0
    overBudget: int = 0
    copies: int = 0
    small: int = 0
    dropped: int = 0
    disabledReason: str = ""

    def __post_init__(self) -> None:
        self.slots = max(0, int(self.slots))
        if self.slots and not self._table:
            self._table = [Slot() for _ in range(self.slots)]

    # -- 判据 --------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.slots > 0

    def free_slot(self) -> int | None:
        for index, slot in enumerate(self._table):
            if slot.state in (EMPTY, MISSED):
                return index
        return None

    def request(self, group: str, nbytes: int, *, predicted: bool = False) -> int | None:
        """想给 ``group`` 发一次拷贝 ✓ ⇒ **流水号**（要拿它去 :meth:`landed` ✓）或 ``None`` ✓。

        五种"不发"分别记账 ✓✗（合成一个数就再也查不出"为什么不发"✓）：

        * ``slots == 0`` ⇒ 这一级根本没开 ✓（``None`` ✓，连计数都不涨 ✓）；
        * 已经装着同一组且 ``LOADING``／``READY`` ⇒ 重复发 ✓（白占带宽 ✓）⇒ 记 ``dropped`` ✓；
        * ``nbytes < minBytes`` ⇒ 太小不值得 ✓ ⇒ 记 ``small`` ✓；
        * 本窗口配额用完 ⇒ ``overBudget`` ✓（**不是** ``busy`` ✗：那个是"槽不够"✓）；
        * 没有空槽 ⇒ ``busy`` ✓（参考实现同名字 ✓）。
        """
        if not self.enabled:
            return None
        with self._lock:
            for slot in self._table:
                if slot.group == group and slot.state in (LOADING, READY):
                    self.dropped += 1
                    return None
            if nbytes < self.minBytes:
                self.small += 1
                return None
            if (self.copiesPerWindow is not None
                    and self._copiesThisWindow >= self.copiesPerWindow):
                self.overBudget += 1
                return None
            index = self.free_slot()
            if index is None:
                self.busy += 1
                return None
            self._seq += 1
            slot = self._table[index]
            slot.group = group
            slot.token = self._seq
            slot.nbytes = int(nbytes)
            slot.state = LOADING
            slot.predicted = bool(predicted)
            self._copiesThisWindow += 1
            self.copies += 1
            return slot.token

    def landed(self, token: int, *, ok: bool = True) -> bool:
        """拷贝回来了 ✓ ⇒ ``True`` ✓（``ok=False`` ⇒ 记 ``missed`` 并标记 ``MISSED`` ✓）。

        ⚠️ 只认**流水号** ✗✗（不认组名 ✓）：一组可能被"先发后废"再发一次 ✓，
        按组名认会让**上一次**的落地把**这一次**标成 ready ✓ ⇒ 槽里是旧字节、plan 说可以用 ✓✗。
        """
        if not self.enabled:
            return False
        with self._lock:
            for slot in self._table:
                if slot.token != token or slot.token == 0:
                    continue
                if slot.state != LOADING:      # 已经被人丢掉 / 用过了 ✓
                    return False
                if not ok:
                    slot.state = MISSED
                    self.missed += 1
                    return False
                slot.state = READY
                return True
            return False

    def take(self, group: str) -> int | None:
        """这一层要算了 ✓ ⇒ 槽里**确实**装着它、而且**已经落地** ⇒ 流水号 ✓，否则 ``None`` ✓。

        这是整条纪律的门口 ✓✗：``LOADING`` 一律 **不用**（记 ``pending`` ✓ 并**放弃**这一份 ✓
        —— 等它就是把 GPU 空在那里等人 ✓，而"照原计划在 CPU 上算"永远是安全的那条 ✓）。
        """
        if not self.enabled:
            return None
        with self._lock:
            for slot in self._table:
                if slot.group != group:
                    continue
                if slot.state == READY:
                    slot.taken += 1
                    self.hits += 1
                    return slot.token
                if slot.state == LOADING:
                    self.pending += 1
                return None             # MISSED / 没装着 ⇒ 走原路 ✓
            return None

    def release(self, group: str) -> int:
        """用完了 ✓ ⇒ 槽腾出来 ✓（返回释放的字节数 ✓，调用方要把它从显存里真放掉 ✓）。"""
        if not self.enabled:
            return 0
        with self._lock:
            for slot in self._table:
                if slot.group != group:
                    continue
                freed = slot.nbytes
                slot.group, slot.token, slot.nbytes = "", 0, 0
                slot.state, slot.predicted, slot.taken = EMPTY, False, 0
                return freed
            return 0

    def discard(self, group: str) -> bool:
        """把装在 ``group`` 上的那份**作废** ✓（显存层面由调用方释放 ✓，这里只记状态 ✓）。"""
        if not self.enabled:
            return False
        with self._lock:
            for slot in self._table:
                if slot.group != group:
                    continue
                if slot.state == LOADING:
                    self.dropped += 1
                slot.group, slot.token, slot.nbytes = "", 0, 0
                slot.state, slot.predicted, slot.taken = EMPTY, False, 0
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._table = [Slot() for _ in range(self.slots)] if self.slots else []

    def new_window(self) -> None:
        """一趟前向开始 ✓ ⇒ 把"本窗口发了几次"清零 ✓（层数是固定的 ✓，不清等于只允许第一趟 ✓✗）。"""
        with self._lock:
            self._copiesThisWindow = 0

    # -- 报告 --------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled, "slots": self.slots,
                "reservedGiB": round(self.reserveBytes / 2**30, 3),
                "hits": self.hits, "pending": self.pending, "missed": self.missed,
                "busy": self.busy, "overBudget": self.overBudget, "copies": self.copies,
                "tooSmall": self.small, "dropped": self.dropped,
                "inSlots": [slot.to_dict() for slot in self._table if slot.state != EMPTY],
                "note": (self.disabledReason or
                         "层预取上卡 ✓：上一层在 CPU 上算的时候，下一层的权重从内存经**独立拷流**"
                         "搬进额外槽 ✓；进那一层时 copy 已落地 ⇒ 那层在卡上算 ✓，"
                         "没落地 ⇒ 照原计划在 CPU 上算 ✓（绝不等、绝不拿半份数据算 ✗）"),
            }


#: 「开启了」时给几个槽 ✓ ＝ **2** ✗✗：一个槽正被这一层用着 ✓，下一层的拷贝就得另有一块地 ✓ ——
#: 只有 1 个槽也能跑 ✓，但每层都退化成"先拷完再算" ✓（``busy`` 涨满 ✓），一层重叠都没有 ✓✗。
DEFAULT_SLOTS = 2


def of_spec(spec: Any) -> Foresight:
    """把 ``place(foresight=…)`` 能收的几种写法归一成一个 :class:`Foresight` ✓。

    收 ``None``／``bool``／``int``／``Foresight`` ✓（``0`` 与 ``None`` 同义 ⇒ 这一级不存在 ✓，
    与参考实现的 ``SLOTS > 0`` 同一条 ✓；``True`` ⇒ :data:`DEFAULT_SLOTS` ✓）。
    """
    if isinstance(spec, Foresight):
        return spec
    if spec is None or spec is False:
        return Foresight(slots=0)
    if spec is True:
        return Foresight(slots=DEFAULT_SLOTS)
    return Foresight(slots=int(spec))


def groups_of(plan_: Iterable[str]) -> tuple[str, ...]:
    """薄壳：把计划的 CPU 组转成"往前看"用的序列 ✓（保留顺序 ✓ —— 层序就是执行序 ✓）。"""
    return tuple(plan_)


def selftest() -> int:
    """自检 ✓（不 import torch ✓、不碰盘 ✓）：只验判据本身 ✓。"""
    bad = 0

    def expect(name: str, cond: object, detail: object = "") -> None:
        nonlocal bad
        if not cond:
            bad += 1
            print(f"FAIL {name}  <<< {detail}")

    off = Foresight(slots=0)
    expect("关掉时 request 返回 None 且一个计数都不涨",
           off.request("layers.0", 1 << 30) is None and off.copies == 0, off.snapshot())
    expect("关掉时 take 也是 None（不能靠它绕过 enabled）",
           off.take("layers.0") is None and not off.enabled, off.snapshot())

    fs = Foresight(slots=1)
    token = fs.request("layers.3", 8 << 20)
    expect("空槽 ⇒ 发得出去（拿到流水号）", isinstance(token, int) and token > 0, token)
    expect("还在拷 ⇒ take 不用（pending +1，绝不等）",
           fs.take("layers.3") is None and fs.pending == 1, fs.snapshot())
    expect("同一组不重复发（白占带宽）", fs.request("layers.3", 8 << 20) is None and fs.dropped == 1,
           fs.snapshot())
    expect("槽不够时记 busy（不是 overBudget）", fs.request("layers.4", 8 << 20) is None
           and fs.busy == 1, fs.snapshot())
    expect("流水号对不上 ⇒ 不算落地（同一组重发时最要命）", fs.landed(999) is False, fs.snapshot())
    expect("落地后 take 给得出", fs.landed(token) is True and fs.take("layers.3") == token,
           fs.snapshot())
    expect("用完 release 把槽腾干净", fs.release("layers.3") > 0 and fs.free_slot() == 0, fs.snapshot())

    token2 = fs.request("layers.5", 8 << 20)
    expect("拷贝失败 ⇒ 记 missed 且不再 pending", fs.landed(token2, ok=False) is False
           and fs.missed == 1 and fs.take("layers.5") is None, fs.snapshot())
    expect("失败过的槽可以复用（MISSED 算空槽）", fs.request("layers.6", 8 << 20) is not None,
           fs.snapshot())

    tiny = Foresight(slots=2, minBytes=4 << 20)
    expect("太小的组不值得占槽（记 tooSmall）",
           tiny.request("x_embedder", 1024) is None and tiny.small == 1, tiny.snapshot())

    capped = Foresight(slots=2, copiesPerWindow=1)
    capped.request("a", 8 << 20)
    expect("窗口配额用完 ⇒ overBudget（与 busy 分开）",
           capped.request("b", 8 << 20) is None and capped.overBudget == 1, capped.snapshot())
    capped.new_window()
    expect("换一趟前向 ⇒ 配额回来", capped.request("b", 8 << 20) is not None, capped.snapshot())

    expect("spec 归一：None/0 关、True 给 DEFAULT_SLOTS（2，一层在用一层在拷）、int 照给",
           not of_spec(None).enabled and not of_spec(0).enabled
           and of_spec(True).slots == DEFAULT_SLOTS == 2 and of_spec(3).slots == 3,
           [of_spec(x).slots for x in (None, 0, True, 3)])
    return bad


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(1 if selftest() else 0)
