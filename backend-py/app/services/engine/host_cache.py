"""**宿主层缓存**：内存这一级也有上限 ✓✗（三级内存的第二级 ✓）。

## 为什么要有这个文件 ✗

本仓的"三级"一直是**缺一级**的 ✓✗：

| 级 | 谁管 | 现状 |
|---|---|---|
| SSD | `win_io` ＋ `weights._load_streaming` ＋ `prefetch` | ✅ 逐张量读、`FILE_FLAG_NO_BUFFERING` 绕页缓存、后台预取 |
| **内存** | **这里** | ⚠️ **原先没有上限** ✗ —— `residency.survey()` 遍历 `module.parameters()` ⇒ 整份常驻 |
| 显存 | `residency.attach` | ✅ 按预算切片 + hook 过界搬激活 |

缺这一级的后果不是"慢" ✓，是**装不上** ✗：`torch_backend._host_guard` 在切完分之后核内存 ✓，
放不下就只有一条路 —— 抛 `scheduler.InsufficientResources` 中止装载 ✓✗。
于是 96 GiB 的机器上「H3 三件同驻 50 GiB」能跑 ✓，而 32 GiB 的机器上同一套**一行都跑不了** ✗✗，
尽管它的 SSD 上明明躺着全部权重 ✓。

补上这一级之后，内存装不下就**卸到盘上**（`spill` ✓），前向到那一层时再读回来 ✓ ——
这正是三级内存的本义 ✓：**任何一级装不下，就往下一级放，而不是失败** ✓。

## 这里只做**决策与记账**，不搬数据 ✗✗

本模块是**纯逻辑** ✓：不 `import torch` ✗、不读盘 ✗（与 :mod:`residency` 的分工一致 ✓ ——
那一层管"落在哪台设备"✓、这一层管"内存里留几层"✓）。

真正把参数从内存里放掉 / 从盘上读回来，由 :mod:`residency` 执行 ✓：
它才知道模块的 `_node` 怎么取、参数怎么 `copy_` 回去 ✓。**决策与动作分开**的理由：
决策要能单测（喂假字节数就能验 LRU ✓），动作必须碰 torch 与盘 ✗（测起来要真文件 ✓✗）。

## 淘汰规则（三条都是硬约束 ✗）

1. **只淘汰层** ✓✗：`layers` 为 0 的组（`x_embedder` / `final` / 顶层裸参数 ✓）**钉住** ——
   它们每步都用 ✓，卸掉之后**每一步**都要重读一次 ✓✗（一层 DiT 上百 MiB × 30 步 = 抖动的定义 ✗）。
2. **LRU，不按大小挑** ✓✗：按"最近一次被用到"排序 ✓。按大小挑会把小层先卸光 ✓、
   留下最大的那几层反复重读 ✓✗（和 `residency.plan` 里"层必须取连续前缀"是同一个道理 ✓：
   局部性比贪心值钱 ✓）。
3. **淘汰前先看"卸了能不能装下"** ✓：要腾 3 GiB 却只卸得出 200 MiB ⇒ **一件都不卸** ✓✗
   —— 白卸一场等于让后面几步都多读一次盘 ✓，还什么都没换来 ✓✗。

## 与 `scheduler.host_room_bytes` 的关系 ✗

**不重复算配额** ✓：:func:`quota_of` 只是把"宿主机总内存 × 比例"算成一个数 ✓，
"此刻还剩多少"仍由 :mod:`scheduler` 的 `host_memory()` / `host_room_bytes()` 回答 ✓
（两处口径打架的代价见 `residency.RESERVE_BY_KIND` 那次 ✓✗：同一张卡两套余量口径 ✓）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "HOST_USE_RATIO", "MIN_QUOTA_BYTES", "RESIDENT", "SPILLED",
    "quota_of", "Slot", "Verdict", "HostCache", "selftest",
]

#: 宿主内存里**允许给权重**的比例 ✓✗（其余留给 torch 的 allocator、激活、图像缓冲、Python 本体 ✓）。
#:
#: ⚠️ 默认 **0.75** 不是拍的 ✓：一次 720p 采样在 CPU 侧的激活 + 帧缓冲实测到 3～6 GiB ✓，
#: 加上 torch 的宿主 allocator 碎片（`PYTORCH_CUDA_ALLOC_CONF` 只管卡上 ✓）⇒
#: 留 25% 是"装得下也不许装满" ✓；**装满的代价是进程被系统换页** ✓✗ ——
#: 那种慢没有报错、也没有堆栈 ✓✗（症状是"今天特别卡"，查一周都查不出来 ✓）。
HOST_USE_RATIO = 0.75

#: 配额的**下限** ✓✗（同 `residency.MIN_MARGIN_BYTES` 的理由 ✓）：比例在很小的内存上会缩没 ✓
#: —— 4 GiB 内存的机器 × 0.75 = 3 GiB ✓，看着合理 ✓，可那台机器上一件 llm 就装不下 ✓✗。
MIN_QUOTA_BYTES = 512 * 1024 * 1024

#: :meth:`HostCache.admit` 的两种结论 ✓（字符串常量：报告 / 日志能直接读 ✓）。
RESIDENT = "resident"
SPILLED = "spilled"


def quota_of(total_bytes: int, *, ratio: float | None = None,
             floor: int = MIN_QUOTA_BYTES) -> int:
    """宿主机总内存 ⇒ **给权重的配额** ✓✗（纯算术 ✓，不读系统 ✓）。

    ⚠️ 比例**不夹到 (0, 1)** ✗✗：``ratio=1.0`` 是"我明确要全部" ✓（A/B 测速用 ✓，
    与 `residency.decompose(reserve=0)` 同一条纪律 ✓ —— 硬夹会静默改掉调用方的决定 ✗）。
    只挡非正数 ✓（那是真没意义 ✓）。
    """
    use = HOST_USE_RATIO if ratio is None else float(ratio)
    if use <= 0:
        raise ValueError(f"ratio 必须 > 0 ✓（拿到 {ratio!r} ⇒ 配额为 0 等于「一层都不许留」✗）")
    total = max(0, int(total_bytes))
    return max(int(floor), int(total * use))


@dataclass
class Slot:
    """一个驻留组的内存账 ✓（可变：``last`` 会被 :meth:`HostCache.touch` 改写 ✓）。"""

    name: str
    nbytes: int
    #: 0 ⇒ 这是**非层组**（钉住 ✓，永不淘汰 ✓）
    layers: int = 0
    last: int = 0          # 最近一次被用到的逻辑时刻 ✓（单调递增 ✓，不是 wall clock ✓✗）
    resident: bool = True

    @property
    def pinned(self) -> bool:
        return self.layers <= 0


@dataclass(frozen=True)
class Verdict:
    """:meth:`HostCache.admit` 的回执 ✓ —— 调用方照着它执行动作 ✓。"""

    name: str
    status: str                      # RESIDENT / SPILLED
    #: 为了腾地方**已划掉**的组 ✓（调用方按它依次释放 ✓；空元组＝没动别人 ✓）
    victims: tuple[str, ...] = ()
    nbytes: int = 0
    used: int = 0
    quota: int = 0
    #: 一句话说清这次为什么是这个结论 ✓（报告里直接给人看 ✓，不给"按策略处理" ✗）
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "status": self.status, "victims": list(self.victims),
            "MiB": round(self.nbytes / 2**20, 1),
            "usedGiB": round(self.used / 2**30, 2), "quotaGiB": round(self.quota / 2**30, 2),
            "reason": self.reason,
        }


def _noop_gc(module: Any) -> None:
    """占位 ✓（真实现由 `residency` 传进来 ✓；这里不 import torch ✗）。"""
    return None


@dataclass
class HostCache:
    """按层配额的宿主驻留表 ✓（LRU + 只淘汰层 + 不白卸 ✓，见模块头三条 ✓）。

    ⚠️ **不拥有**这些张量 ✗：它只记"谁在内存里、多大、多久没用" ✓。
    释放和重读是 :mod:`residency` 的事 ✓✗ —— 这里擅自 `del` 一下只会得到一个
    「报告说卸了、参数还在」的假账 ✓✗（本仓已经踩过同形状的坑：`release_part` 只置 None
    而显存不退 ✓）。
    """

    quota: int
    name: str = "host"
    _slots: dict[str, Slot] = field(default_factory=dict, repr=False)
    _clock: int = 0
    _bytes: int = 0                  # 当前驻留字节 ✓（只算 resident ✓）
    evicted: int = 0
    restores: int = 0

    # ---- 登记 -----------------------------------------------------------
    def admit(self, name: str, nbytes: int, *, layers: int = 0) -> Verdict:
        """要不要把 ``name`` 留在内存 ✓ ⇒ :class:`Verdict` ✓（**不搬任何字节** ✗）。

        三种情形 ✓：

        * 装得下 ⇒ ``RESIDENT`` ✓、不动别人 ✓；
        * 装不下但**卸得够**（LRU 挑出来的层 ≥ 差额 ✓）⇒ ``RESIDENT`` ＋ 带 ``victims`` ✓；
        * 卸完也不够 ⇒ ``SPILLED`` ✓ 且 **``victims`` 为空** ✓✗（白卸一场没意义 ✓）。
        """
        size = max(0, int(nbytes))
        self._clock += 1
        slot = self._slots.get(name)
        if slot is None:
            #: ⚠️ 新登记的一律 ``resident=False`` ✓✗：**登记 ≠ 已经在内存里** ✓ ——
            #: `admit` 问的是"要不要留"✓，这一刻它的字节还不在 `_bytes` 里 ✓。
            #: 写成 True 会让这次的候选**把自己算进可淘汰集** ✓ ⇒ "要 200 只腾得出 30" 也照样
            #: 报"卸够了"✓（`_pick` 把 ``big`` 自己挑进去、`evict` 再把自己划掉 ✓）——症状是
            #: 一个比配额还大的层被判成"留内存" ✓，然后 `used` 比 `quota` 还大 ✓✗。
            slot = Slot(name=name, nbytes=size, layers=int(layers), resident=False)
            self._slots[name] = slot
        else:
            if slot.resident:
                self._bytes -= slot.nbytes
            slot.nbytes = size
            slot.layers = int(layers)
        slot.last = self._clock

        if self._bytes + size <= self.quota:
            self._bytes += size
            slot.resident = True
            return Verdict(name=name, status=RESIDENT, nbytes=size, used=self._bytes,
                           quota=self.quota,
                           reason="" if slot.pinned or size == 0
                           else f"装得下（{self._used_gib():.2f} / {self._quota_gib():.2f} GiB）")

        #: 差额 ✓ —— 只有"卸得出"才动手 ✓✗（模块头第 3 条 ✓）
        need = self._bytes + size - self.quota
        victims = self._pick(need)
        if victims:
            for victim in victims:
                self.evict(victim)
            self._bytes += size
            slot.resident = True
            return Verdict(name=name, status=RESIDENT, victims=tuple(victims), nbytes=size,
                           used=self._bytes, quota=self.quota,
                           reason=(f"装不下 ⇒ 先卸 {len(victims)} 层（LRU ✓）腾出 "
                                   f"{sum(self._slots[v].nbytes for v in victims if v in self._slots) / 2**20:.0f} MiB"))
        slot.resident = False
        return Verdict(name=name, status=SPILLED, nbytes=size, used=self._bytes,
                       quota=self.quota,
                       reason=(f"卸不出足够空间 ⇒ 这层留盘上 ✓（需要 {need / 2**20:.0f} MiB，"
                               f"可卸的只有 {self._evictable() / 2**20:.0f} MiB ✓）"))

    def touch(self, name: str) -> bool:
        """记一次"用到了" ✓ ⇒ 它在 LRU 里的位置往后挪 ✓（返回它此前在不在 ✓）。"""
        slot = self._slots.get(name)
        if slot is None:
            return False
        self._clock += 1
        slot.last = self._clock
        return slot.resident

    def evict(self, name: str) -> bool:
        """卸掉 ✓（**只记账** ✓：真释放由调用方做 ✓）。"""
        slot = self._slots.get(name)
        if slot is None or not slot.resident:
            return False
        slot.resident = False
        self._bytes = max(0, self._bytes - slot.nbytes)
        self.evicted += 1
        return True

    def restore(self, name: str) -> Verdict:
        """读回来了 ✓（调用方**先**把参数填回去、成功之后再调这里 ✓✗ —— 顺序反了会得到一个
        「账上回来了、内存里没有」的假账 ✓）。

        ⚠️ 与 :meth:`admit` 的差别只有一条 ✓✗：``admit`` 可以答"不留"（那就别读 ✓），
        而**一旦已经决定读回来**，这层就**必须**算 resident ✓ —— 前向缺它根本算不了 ✓✗。
        所以配额真不够时这里如实**超额记账**并在 ``reason`` 里写清 ✓，
        **不是**返回 ``SPILLED`` 让调用方以为"没读"✓（那会把
        `residency` 的 `ensure` 逼成一个当场炸的硬失败 ✓，而它手里其实已经有数据了 ✓）。
        """
        slot = self._slots.get(name)
        if slot is None:
            return self.admit(name, 0)
        self.restores += 1
        verdict = self.admit(name, slot.nbytes, layers=slot.layers)
        if verdict.status == RESIDENT:
            return verdict
        slot.resident = True
        self._bytes += slot.nbytes
        return Verdict(name=name, status=RESIDENT, nbytes=slot.nbytes, used=self._bytes,
                       quota=self.quota,
                       reason=(f"配额不够仍读回（超额 {max(0, self._bytes - self.quota) / 2**20:.0f} MiB ✓）："
                               f"这层这一拍一定要用，卸了它前向就算不了 ✓"))

    # ---- 查询 -----------------------------------------------------------
    @property
    def used_bytes(self) -> int:
        return self._bytes

    @property
    def headroom(self) -> int:
        return max(0, self.quota - self._bytes)

    def resident(self) -> tuple[str, ...]:
        return tuple(n for n, s in self._slots.items() if s.resident)

    def spilled(self) -> tuple[str, ...]:
        return tuple(n for n, s in self._slots.items() if not s.resident)

    def _evictable(self) -> int:
        return sum(s.nbytes for s in self._slots.values() if s.resident and not s.pinned)

    def _pick(self, need: int) -> list[str]:
        """按 LRU 挑出**刚好够**的那几层 ✓（跳过钉住的 ✓）；不值得就一件都不挑 ✓。"""
        if need <= 0:
            return []
        candidates = sorted((s for s in self._slots.values() if s.resident and not s.pinned),
                            key=lambda s: s.last)
        picked: list[str] = []
        freed = 0
        for slot in candidates:
            if freed >= need:
                break
            picked.append(slot.name)
            freed += slot.nbytes
        if freed < need:
            return []
        return picked

    def _used_gib(self) -> float:
        return self._bytes / 2**30

    def _quota_gib(self) -> float:
        return self.quota / 2**30

    # ---- 可见面 ---------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """喂装载报告 ✓：「内存这一级现在装了几层、卸了几层、卸过几次」✓。"""
        res = [s for s in self._slots.values() if s.resident]
        return {
            "name": self.name,
            "quotaGiB": round(self._quota_gib(), 2),
            "usedGiB": round(self._used_gib(), 2),
            "headroomGiB": round(self.headroom / 2**30, 2),
            "layers": sum(1 for s in res if s.layers > 0),
            "pinned": sum(1 for s in res if s.pinned),
            "spilled": [s.name for s in self._slots.values() if not s.resident],
            "evicted": self.evicted,
            "restores": self.restores,
        }


def selftest() -> int:
    """自检 ✓（纯计数 ✓：不碰 torch、不碰盘 ✓）。"""
    failures: list[str] = []

    def expect(label: str, condition: object, detail: object = "") -> None:
        if not condition:
            failures.append(f"{label}：{detail}")

    cache = HostCache(quota=100, name="selftest")
    first = cache.admit("layers.0", 40, layers=1)
    expect("装得下", first.status == RESIDENT and not first.victims, first)
    expect("装得下不动别人", cache.used_bytes == 40, cache.used_bytes)
    cache.admit("layers.1", 40, layers=1)
    cache.touch("layers.0")
    #: 第 3 层要 40 ⇒ 差 20 ⇒ 按 LRU 该卸 layers.1（layers.0 刚被用过 ✓）
    third = cache.admit("layers.2", 40, layers=1)
    expect("LRU 卸最近最少用的那层", third.victims == ("layers.1",), third.victims)
    expect("卸完账对得上", cache.used_bytes == 80, cache.used_bytes)

    #: 非层组钉住 ✓：即使它最旧也不许卸 ✓
    pinned = HostCache(quota=100, name="selftest-pin")
    pinned.admit("x_embedder", 60, layers=0)
    pinned.admit("layers.0", 30, layers=1)
    pinned.touch("layers.0")
    verdict = pinned.admit("layers.1", 30, layers=1)
    expect("钉住的组不许被淘汰", "x_embedder" not in verdict.victims, verdict.victims)
    expect("卸一层就够 ⇒ 装下", verdict.status == RESIDENT, verdict)

    #: 卸得出但**不够** ⇒ 一件都不卸 ✓（白卸一场没意义 ✓）
    stuck = HostCache(quota=100, name="selftest-stuck")
    stuck.admit("layers.0", 30, layers=1)
    blocked = stuck.admit("big", 200, layers=1)
    expect("卸不够就不卸", blocked.status == SPILLED and not blocked.victims, blocked)
    expect("不白卸（原层还在）", "layers.0" in stuck.resident(), stuck.resident())

    #: restore 走回 resident ✓
    stuck.restore("big")
    expect("重读之后账上回来", "big" in stuck.resident() and stuck.restores == 1, stuck.snapshot())

    #: 配额算术
    expect("比例照算", quota_of(8 * 2**30, ratio=0.5) == 4 * 2**30, quota_of(8 * 2**30, ratio=0.5))
    expect("小内存补下限", quota_of(1024, ratio=0.5) == MIN_QUOTA_BYTES, quota_of(1024, ratio=0.5))
    try:
        quota_of(2**30, ratio=0)
        failures.append("ratio=0 没被拒（配额 0 ＝ 一层都不许留 ✗）")
    except ValueError:
        pass

    for line in failures:
        print("FAIL  " + line)
    print(f"SUMMARY: {'OK' if not failures else 'FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
