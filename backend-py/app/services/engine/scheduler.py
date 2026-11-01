"""**任务级**调度 ✓：GPU 优先 ⇒ 显存不足自动 **GPU ＋ CPU 协同** ⇒ 两边都放不下就**当场拒** ✓。

## 与 `device_policy` / `residency` 的分工 ✗✗（三处不许互相抄 ✓）

| 层 | 回答什么问题 |
|---|---|
| :mod:`device_policy` | **这次**该把一件东西放哪 ✓（纯函数 ✓ 只看显存读数 ✓） |
| :mod:`residency` | **放在卡上的那部分怎么切** ✓（按层切 / 参数钉在 CPU ✓ ＋ 跨界 hook ✓） |
| **本模块** | **谁在占、占多少、什么时候还** ✓ ＋ 运行期压力状态 ✓ |

⇒ 一层的结论当下一层的输入 ✓，**不重复实现**别层已有的判据 ✗（否则两套阈值迟早打架 ✓✗）。

## 本次补的是两个真洞 ✓

### ① 「搬去 CPU」从来不问 CPU 有没有地方 ✗✗

`device_policy.decide()` 只看**显存** ✓：显存紧 ⇒ 判 `split` ⇒ 剩下的层搬进内存 ✓。
可**内存也会不够** ✓ —— 2026-11-01 实测：32B 文本塔的 bf16 骨架把宿主内存吃到
**74 / 93 GiB** ✓，系统只剩 **6 GiB** ✓，进程随即在页错误里停滞十几分钟 ✓、连权重文件都
没开始读 ✓✗。那一刻 `decide()` 的结论完全"正确"，坏的是**没人核过第二台设备** ✗。

⇒ :func:`assign` 在这里补 CPU 侧余量（:func:`host_memory` ✓ 纯 stdlib ✓）：
放不下 ⇒ 把能放的那部分**多给卡** ✓；卡也接不住 ⇒ :class:`InsufficientResources` ✓
（当场说"两边都装不下，需要多少 / 只有多少" ✓，不许交给交换文件慢慢拖 ✗✗）。

### ② 运行期压力没有任何观察者 ✗

阶段边界（`TorchBackend.enter_phase` ✓）会重新探一次 ✓，可**采样 30 步的中途**别人把卡
抢了、激活涨上去了 ✓ —— 旧实现一路跑到 OOM 才炸 ✓✗。⇒ :class:`LoadWatch`（带**滞回** ✓）
＋ `TorchBackend.tick()`：超硬阈值才动手 ✓、动手也只做**回收**（退掉本阶段用不着的件 ✓），
**不做**搬来搬去的"优化" ✗（那是抖动之源 ✓✗）。

## 滞回（为什么不是一超阈值就切）✗✗

`usedRatio` 在采样过程中天然抖动 ✓（临时激活、allocator 预留块 ✓）。单次读数就降级 ⇒
一步上一步下 ✓✗（每次都重装 14 GiB ✓）。⇒ 连涨 ``CONFIRM_SAMPLES`` 次才判 `hard` ✓；
回到 ``SOFT_USED_RATIO`` 以下立即判 `normal` ✓（保守方向不延迟 ✓：早回 GPU 只是下一次装
载的决定 ✓，不会打断正在跑的活 ✓）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from . import device_policy

#: 调度的口径名 ✓（**恒为** GPU 优先 ✓ —— 这是创作者定的硬规则 ✓，不是可选项 ✗）
POLICY_NAME = "gpu-first"

GIB = device_policy.GIB
MIB = 1 << 20

#: 卡上留的余量 ✓（与 `device_policy.HEADROOM_BYTES` 同一个数 ✓ —— 借过来不另立 ✗）
GPU_HEADROOM_BYTES = device_policy.HEADROOM_BYTES
#: 卡上至少要有这么多，切分才有意义 ✓（少于它 ⇒ 整件转 CPU ✓，同 `decide()` 口径 ✓）
MIN_SPLIT_FREE_BYTES = device_policy.MIN_SPLIT_FREE_BYTES

#: 内存侧要留的余量 ✓：比例 ＋ 下限取**大的那个** ✓
#: （比例照顾大内存机器 ✓ —— 64 GiB 机器上留 5 GiB 不算多 ✓；下限照顾小机器 ✓）
HOST_RESERVE_RATIO = 0.08
HOST_RESERVE_MIN_BYTES = 2 * GIB

#: 运行期压力档 ✓（`soft` 以下不动作 ✓、`hard` 才动手 ✓）
SOFT_USED_RATIO = 0.88
HARD_USED_RATIO = 0.95
#: 连续多少次读数超硬阈值才认 ✓（防抖 ✓，见模块头）
CONFIRM_SAMPLES = 2

STATES: tuple[str, ...] = ("normal", "pressured", "hard")


class InsufficientResources(RuntimeError):
    """**两边都放不下** ✓ —— 当场拒 ✓（不静默降级成"跑得极慢" ✗✗）。"""


# ── 资源检测 ────────────────────────────────────────────────────────────────


def os_sysconf(name: str) -> int:  # pragma: no cover - 平台分支
    """`os.sysconf` 的薄封装 ✓（只为把 import 收在一处 ✓ 便于 `host_memory` 一条 try 兜住 ✓）。"""
    import os  # noqa: PLC0415

    return int(os.sysconf(name))


def host_memory() -> tuple[int, int] | None:
    """宿主**物理内存** ``(free, total)`` 字节 ✓；量不到 ⇒ ``None`` ✓（不猜 ✗）。

    ⚠️ 只用 stdlib ✓（`psutil` 不是必装依赖 ✓，为了一个读数把整条装载链挂上它不值 ✗）：
    * Windows ⇒ ``GlobalMemoryStatusEx`` ✓（``ullAvailPhys`` ✓ 就是"当前可用物理内存" ✓）；
    * 类 Unix ⇒ ``sysconf('SC_AVPHYS_PAGES')`` ✓。

    ⚠️ **不是**「虚拟内存 / 提交上限」✗：Windows 把 pagefile 也算进提交额度 ✓ ⇒ 拿它判会
    得出"还有 100 GiB" ✓✗，而真实内存早满了 ✓（那种"看起来能装"正是把机器拖进 swap 的原因 ✗✗）。
    """
    try:
        if sys.platform.startswith("win"):  # pragma: no cover - 平台分支
            import ctypes  # noqa: PLC0415

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.ullAvailPhys), int(status.ullTotalPhys)
        page = int(os_sysconf("SC_PAGE_SIZE"))  # noqa: F821 —— 见下面的 `os_sysconf`
        available = int(os_sysconf("SC_AVPHYS_PAGES"))  # noqa: F821
        return page * available, page * int(os_sysconf("SC_PHYS_PAGES"))  # noqa: F821
    except Exception:  # noqa: BLE001 —— 读数失败不是致命错 ✓（退回"未知" ✓ 见 `assign` ✓）
        return None


def host_reserve_bytes(total_bytes: int) -> int:
    """内存侧留多少 ✓（比例与下限取大 ✓）。"""
    return max(int(HOST_RESERVE_MIN_BYTES), int(float(total_bytes) * HOST_RESERVE_RATIO))


def host_room_bytes(host: tuple[int, int] | None = None) -> int | None:
    """**现在**还能放多少进内存 ✓（已扣掉留给系统的余量 ✓）；读数取不到 ⇒ ``None`` ✓。"""
    read = host if host is not None else host_memory()
    if read is None:
        return None
    free, total = read
    return max(0, int(free) - host_reserve_bytes(total))


def ensure_host_room(need_bytes: int, *, host: tuple[int, int] | None = None,
                     what: str = "") -> dict[str, Any]:
    """**整份放内存**之前核一次 ✓✗（放不下 ⇒ :class:`InsufficientResources` ✓）。

    ⚠️ 为什么必须有这一道 ✗：:func:`device_policy.decide` 只看**显存** ✓ —— 它判"转 CPU"
    时并不知道内存还剩多少 ✓✗。2026-11-01 实测 32B 文本塔把宿主吃到 74/93 GiB、系统只剩
    6 GiB ✓，进程在页错误里停了十几分钟 ✓、连权重文件都没开始读 ✓✗ —— 那一刻 `decide()` 的
    结论完全"正确" ✓，坏的是没人核过第二台设备 ✗。

    读数取不到 ⇒ **放行但如实说没核** ✓（`checked=False` ✓）—— 宁可少拦一次，也不假装核过 ✓✗。
    """
    read = host if host is not None else host_memory()
    need = max(0, int(need_bytes))
    if read is None:
        return {"what": what, "needGiB": round(need / GIB, 2), "checked": False,
                "note": "宿主内存读数取不到 ⇒ 这一道**没核** ✓（不假装核过 ✗）"}
    ok, why = _fit_host(need, read)
    if not ok:
        raise InsufficientResources(
            f"{what or '这一件'}要**整份放进内存** ✓，但 {why} ⇒ **拒装** ✓✗"
            f"（不交给交换文件硬跑 ✗✗：那会先卡死十几分钟、再在别处报一个看不出根因的错 ✓）"
            f"　⇒ 先关掉别的占内存的进程 ✓，或换更小的档位 ✓")
    return {"what": what, "needGiB": round(need / GIB, 2), "checked": True,
            "hostFreeGiB": round(read[0] / GIB, 2),
            "hostRoomGiB": round(host_room_bytes(read) / GIB, 2)}


# ── 分配 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Slots:
    """**这次**的落位 ✓ —— 每个字段都要能进报告 ✓（静默改落点 = 用户以为在上卡 ✗）。"""

    device: str
    residency: str
    #: 放卡上的字节数 ✓（``off`` 时 = 全部 ✓；``split`` 时是**能给卡的那部分** ✓）
    gpuBytes: int
    #: 放内存的字节数 ✓
    cpuBytes: int
    needBytes: int
    degraded: bool
    reason: str
    policy: str = POLICY_NAME
    hostFreeBytes: int | None = None
    gpuFreeBytes: int | None = None

    @property
    def needGiB(self) -> float:
        return round(int(self.needBytes) / GIB, 2)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "device": self.device, "residency": self.residency,
            "gpuGiB": round(int(self.gpuBytes) / GIB, 2),
            "cpuGiB": round(int(self.cpuBytes) / GIB, 2),
            "needGiB": self.needGiB, "degraded": bool(self.degraded),
            "reason": self.reason, "policy": self.policy,
        }
        if self.gpuFreeBytes is not None:
            out["gpuFreeGiB"] = round(int(self.gpuFreeBytes) / GIB, 2)
        if self.hostFreeBytes is not None:
            out["hostFreeGiB"] = round(int(self.hostFreeBytes) / GIB, 2)
        return out


def _fit_host(need_cpu: int, host: tuple[int, int] | None) -> tuple[bool, str]:
    """内存放不放得下 ``need_cpu`` ✓（量不到 ⇒ 放行 ✓ 但如实说"没测到" ✓）。"""
    if need_cpu <= 0:
        return True, ""
    if host is None:
        return True, "（宿主内存读数取不到 ⇒ 这一条没核 ✓，不假装核过 ✗）"
    free, total = host
    reserve = host_reserve_bytes(total)
    if free >= need_cpu + reserve:
        return True, ""
    return False, (f"宿主内存只剩 {free / GIB:.2f} GiB ✗ < 要放 {need_cpu / GIB:.2f} GiB ＋ "
                   f"留 {reserve / GIB:.2f} GiB ✓")


def assign(need_bytes: int, *, pressure: device_policy.Pressure | None = None,
           host: tuple[int, int] | None = None,
           policy: Any = device_policy.DEFAULT_POLICY,
           requested_device: str | None = None, gpu_available: bool = True,
           reserve_gpu_bytes: int = 0) -> Slots:
    """**GPU 优先** ⇒ 不足则协同 ⇒ 都放不下就拒 ✓（纯函数 ✓ 不读盘、不碰 torch ✓）。

    ``reserve_gpu_bytes``：账本里**别人**还占着多少 ✓（只进报告与说明 ✓ —— 空闲显存读数
    已经含了它 ✓，再扣一遍会把自己越推越往 CPU ✓✗）。

    ⚠️ 判据顺序**不许调** ✗✗：显式 CPU / 没卡 / 强制 GPU 这三档先走 ✓（那是**指令** ✓）；
    之后才是「够 ⇢ 切 ⇢ 都不够」✓（那是**估算** ✓）—— 混起来会出现"用户明确要 CPU、
    却被显存读数劝回卡上" ✓✗。
    """
    name = device_policy.normalize_policy(policy)
    need = max(0, int(need_bytes))
    gpu_free = None if pressure is None else int(pressure.freeBytes)
    host_free = None if host is None else int(host[0])

    def _slots(device: str, residency: str, gpu: int, cpu: int, degraded: bool,
               reason: str) -> Slots:
        return Slots(device=device, residency=residency, gpuBytes=int(gpu), cpuBytes=int(cpu),
                     needBytes=need, degraded=degraded, reason=reason, policy=name,
                     hostFreeBytes=host_free, gpuFreeBytes=gpu_free)

    requested = str(requested_device or "").strip().lower()
    # ① 指令档：显式 CPU / 策略 cpu ⇒ 尊重 ✓（不是降级 ✓）
    if name == "cpu" or requested.startswith("cpu"):
        ok, why = _fit_host(need, host)
        if not ok:
            raise InsufficientResources(
                f"要求整份放 CPU ✓，但 {why} ⇒ **拒跑** ✓（不交给交换文件慢慢拖 ✗✗）")
        return _slots("cpu", "off", 0, need, False,
                      "策略/请求指定 CPU ⇒ 整份在内存 ✓（不是降级 ✓）")

    # ② 没卡
    if not gpu_available:
        if name == "gpu":
            return _slots("cuda", "off", 0, 0, False,
                          "策略 `gpu` ⇒ 照上卡 ✓（本机没有可用 CUDA ✗ ⇒ 装载会当场失败 ✓）")
        ok, why = _fit_host(need, host)
        if not ok:
            raise InsufficientResources(f"本机没有可用 CUDA ✓，退回 CPU 也不行 ✗：{why}")
        return _slots("cpu", "off", 0, need, True, "本机没有可用 CUDA ✓ ⇒ 整条链在 CPU ✓（降级 ✓）")

    # ③ 强制 GPU（显存不够就 OOM ✓ 这是策略要求的 ✓）
    if name == "gpu":
        return _slots("cuda", "off", need, 0, False,
                      "策略 `gpu` ⇒ 只上卡 ✓（不够就直接 OOM ✓ 不转移 ✗）")

    # ④ 测不到卡况 ⇒ 按默认优先 GPU ✓（没有依据降级 ✗，但如实说没测到 ✓）
    if pressure is None:
        return _slots("cuda", "off", need, 0, False,
                      "测不到显存读数 ✓ ⇒ 按默认**优先 GPU** ✓（没有依据降级 ✗）")

    free = int(pressure.freeBytes)
    used_ratio = pressure.usedRatio
    pressured = used_ratio >= device_policy.HIGH_USED_RATIO
    others = "" if reserve_gpu_bytes <= 0 else (
        f"；账本里另有 {reserve_gpu_bytes / GIB:.2f} GiB 占用 ✓")

    # ⑤ 够：整份上卡 ✓（`balanced` 高压时主动留余量 ⇒ 协同 ✓）
    if free >= need + GPU_HEADROOM_BYTES:
        if pressured and name == "balanced":
            #: ⚠️ ``degraded=False`` ✓：这一档是**用户选的策略**在主动留余量 ✓，不是被动降级 ✗ ——
            #: 报告里 `degraded` 是给"原计划没做到"用的 ✓✗（把它也标 True，用户会以为出事 ✓）。
            return _split_split(need, free, host, _slots,
                                f"装得下 ✓，但显存已用 {used_ratio:.0%} ≥ "
                                f"{device_policy.HIGH_USED_RATIO:.0%} ⇒ 按 `balanced` 主动协同 ✓{others}",
                                degraded=False)
        return _slots("cuda", "off", need, 0, False,
                      f"装得下 ✓：余 {free / GIB:.2f} GiB ≥ 需 {need / GIB:.2f} ＋ 1 GiB ✓ "
                      f"⇒ 整份上卡 ✓{others}")

    # ⑥ 不够：能切就切 ✓（切出来那部分要核**内存**放不放得下 ✓✗ —— 本次补的洞 ✓）
    if free >= MIN_SPLIT_FREE_BYTES:
        return _split_split(need, free, host, _slots,
                            f"余 {free / GIB:.2f} GiB < 需 {need / GIB:.2f} GiB ＋ 1 GiB ✗ "
                            f"⇒ **GPU/CPU 协同** ✓{others}")

    # ⑦ 卡上连有意义的切片都放不下 ⇒ 整件转 CPU ✓（同样要核内存 ✓）
    ok, why = _fit_host(need, host)
    if not ok:
        raise InsufficientResources(
            f"显存只剩 {free / GIB:.2f} GiB ✗（< {MIN_SPLIT_FREE_BYTES / GIB:.2f} GiB ⇒ 切不出"
            f"有意义的块 ✓），而整件放 CPU 也放不下 ✗：{why}　⇒ **两边都装不下** ✓："
            f"需要 {need / GIB:.2f} GiB，请先关掉别的占卡/占内存的进程再试 ✓")
    return _slots("cpu", "off", 0, need, True,
                  f"显存只剩 {free / GIB:.2f} GiB ✗ < {MIN_SPLIT_FREE_BYTES / GIB:.2f} GiB "
                  f"⇒ 切不出有意义的块 ⇒ 整件转 CPU ✓（内存放得下 ✓）{others}")


def _split_split(need: int, free: int, host: tuple[int, int] | None,
                 make: Any, reason: str, *, degraded: bool = True) -> Slots:
    """切分档的公共算法 ✓：先给卡 ``free − 余量`` ✓，再把 CPU 侧缩到内存放得下的量 ✓✗。

    ⚠️ **不许多给卡** ✗：卡上给到刚好留 1 GiB 余量 ✓ —— 再挤就是采样期 OOM（那正是
    「装载成功、第一步炸」的形状 ✗✗）。
    ⚠️ 内存放不下 CPU 那部分 ⇒ **没有**「加卡顶上」这一步 ✗（2026-11-01 复核 ✓）：卡上在
    :func:`_split_split` 第一行就给到 ``余量 − 1 GiB`` 了 ✓，再往上加＝把保命的余量吃掉 ✗✗。
    早先那段 ``extra_gpu = min(缺口, 余量 − 1 GiB − gpu)`` **恒等于 0** ✓ —— 是**死分支** ✓✗
    （留着只会让人以为还有余量可调度 ✓）。⇒ 这里直接拒 ✓ 并把账算给用户看 ✓。
    """
    gpu = max(0, min(int(free) - GPU_HEADROOM_BYTES, int(need)))
    cpu = max(0, int(need) - gpu)
    if cpu > 0 and host is not None:
        host_free, host_total = host
        reserve = host_reserve_bytes(host_total)
        room = max(0, int(host_free) - reserve)
        if cpu > room:
            raise InsufficientResources(
                f"GPU/CPU 协同也放不下 ✗：卡上能给 {gpu / GIB:.2f} GiB（余 "
                f"{free / GIB:.2f} GiB ✓），内存只剩 {host_free / GIB:.2f} GiB、留 "
                f"{reserve / GIB:.2f} GiB 之后还能放 {room / GIB:.2f} GiB ✗ ⇒ 合计 "
                f"{(gpu + room) / GIB:.2f} GiB < 需 {need / GIB:.2f} GiB ✓"
                "　⇒ **拒跑** ✓（不许靠交换文件硬跑 ✗✗）")
    return make("cuda", "split", gpu, cpu, degraded, reason)


def reclaim_for(phase: str, present: dict[str, bool],
                wanted: tuple[str, ...] | None) -> list[str]:
    """到硬阈值时**该退掉谁** ✓（纯函数 ✓ 便于验 ✓ 也免得判据留在 `TorchBackend` 里没法测 ✗）。

    ⚠️ ``wanted is None`` ⇒ **一个都不退** ✓✗：阶段表没登记这一拍到底要用哪几件 ⇒ 猜"哪件
    用不着"就会把正在跑的那件卸掉 ✓✗。宁可只报压力 ✓，也不做这种"看运气"的回收 ✗。
    ⚠️ **只退不在本阶段清单里的件** ✓ —— 不搬、不换、不重装 ✓（见模块头的防抖理由 ✓）。
    """
    if wanted is None:
        return []
    keep = {str(item) for item in wanted}
    return [str(part) for part, here in present.items() if here and str(part) not in keep]


# ── 任务账本（谁在占、占多少、几时还）────────────────────────────────────────


@dataclass
class Entry:
    """账本里的一条 ✓。"""

    part: str
    bytes: int
    device: str
    residency: str
    phase: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = {"part": self.part, "GiB": round(self.bytes / GIB, 2),
               "device": self.device, "residency": self.residency}
        if self.phase:
            out["phase"] = self.phase
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class Ledger:
    """**谁现在占着资源** ✓ —— 装配、换装、回收三处都过这一本 ✓。

    ⚠️ 这本账**不是**判据 ✗✗：判据永远是**现测**的读数（`device_policy.probe` ✓）——
    账本只回答"这些字节是谁的" ✓（换装泄漏、忘记回收、两个件重叠占卡都能从这看出来 ✓）。
    拿账本当"还有多少余量"用，就会跟 allocator 的预留块对不上 ✓✗。
    """

    items: dict[str, Entry] = field(default_factory=dict)

    def reserve(self, part: str, *, bytes_: int, device: str, residency: str = "off",
                phase: str | None = None, note: str = "") -> Entry:
        entry = Entry(part=str(part), bytes=int(bytes_), device=str(device),
                      residency=str(residency), phase=phase, note=str(note))
        self.items[entry.part] = entry
        return entry

    def release(self, part: str) -> Entry | None:
        return self.items.pop(str(part), None)

    def clear(self) -> None:
        self.items.clear()

    def bytes_on(self, *devices: str) -> int:
        wanted = {str(item) for item in devices}
        return sum(entry.bytes for entry in self.items.values() if entry.device in wanted)

    @property
    def gpuBytes(self) -> int:
        return self.bytes_on("cuda") + self.bytes_on("cuda:0")

    @property
    def ramBytes(self) -> int:
        return self.bytes_on("cpu")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpuGiB": round(self.gpuBytes / GIB, 2),
            "ramGiB": round(self.ramBytes / GIB, 2),
            "items": {name: entry.to_dict() for name, entry in self.items.items()},
        }


# ── 运行期压力（滞回）────────────────────────────────────────────────────────


@dataclass
class LoadWatch:
    """运行期压力状态机 ✓（**带滞回** ✓，见模块头）。

    ⚠️ 读数取不到 ⇒ 状态**不动** ✓ 且 ``source="unknown"`` ✓（"测不到"不等于"没压力" ✗）。
    """

    soft: float = SOFT_USED_RATIO
    hard: float = HARD_USED_RATIO
    confirm: int = CONFIRM_SAMPLES
    state: str = "normal"
    hits: int = 0
    samples: int = 0
    lastRatio: float | None = None
    changed: int = 0

    def sample(self, pressure: device_policy.Pressure | None) -> str:
        self.samples += 1
        if pressure is None:
            return self.state
        ratio = float(pressure.usedRatio)
        self.lastRatio = round(ratio, 4)
        previous = self.state
        if ratio >= self.hard:
            self.hits += 1
            if self.hits >= max(1, int(self.confirm)):
                self.state = "hard"
        elif ratio <= self.soft:
            self.hits = 0
            self.state = "normal"
        else:
            self.hits = 0
            self.state = "pressured"
        if self.state != previous:
            self.changed += 1
        return self.state

    @property
    def wants_reclaim(self) -> bool:
        """要不要**动手回收** ✓（只认 `hard` ✓ —— `pressured` 只记录不动作 ✓ 防抖 ✓）。"""
        return self.state == "hard"

    def reset(self) -> None:
        self.state = "normal"
        self.hits = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state, "usedRatio": self.lastRatio,
            "soft": self.soft, "hard": self.hard, "confirm": self.confirm,
            "samples": self.samples, "changed": self.changed,
            "source": "memory" if self.lastRatio is not None else "unknown",
        }


def selftest() -> list[str]:
    """本模块自检 ✓（纯逻辑 ✓ 无 torch ✓）⇒ 返回问题清单 ✓（空 ⇒ 全绿 ✓）。"""
    problems: list[str] = []

    def _press(total_gib: float, free_gib: float) -> device_policy.Pressure:
        return device_policy.Pressure(totalBytes=int(total_gib * GIB),
                                      freeBytes=int(free_gib * GIB))

    def _host(total_gib: float, free_gib: float) -> tuple[int, int]:
        return int(free_gib * GIB), int(total_gib * GIB)

    need = int(13.26 * GIB)
    # ① 够 ⇒ 整份上卡 ✓
    slots = assign(need, pressure=_press(22.49, 20.0), host=_host(64, 60))
    if slots.device != "cuda" or slots.residency != "off" or slots.degraded:
        problems.append(f"够了却没整份上卡：{slots.to_dict()}")
    # ② 差一点 ⇒ 协同，且两边加起来正是需求 ✓
    slots = assign(need, pressure=_press(22.49, 6.0), host=_host(64, 50))
    if slots.device != "cuda" or slots.residency != "split" or not slots.degraded:
        problems.append(f"差一点却没协同：{slots.to_dict()}")
    elif slots.gpuBytes + slots.cpuBytes != need:
        problems.append(f"协同后两边之和 ≠ 需求：{slots.to_dict()}")
    # ③ 内存放不下 ⇒ 拒（本次补的洞 ✓）
    try:
        assign(need, pressure=_press(22.49, 6.0), host=_host(16, 2.0))
    except InsufficientResources:
        pass
    else:
        problems.append("内存吃紧却照样判了协同（应当拒）")
    # ④ 没卡 ⇒ CPU ✓
    slots = assign(need, pressure=None, host=_host(64, 60), gpu_available=False)
    if slots.device != "cpu" or not slots.degraded:
        problems.append(f"没卡却没退 CPU：{slots.to_dict()}")
    # ⑤ 策略 gpu ⇒ 再紧也上卡 ✓
    slots = assign(need, pressure=_press(22.49, 0.2), host=_host(8, 1), policy="gpu")
    if slots.device != "cuda":
        problems.append(f"策略 gpu 却没上卡：{slots.to_dict()}")
    # ⑥ 策略 cpu ⇒ 不看显存 ✓
    slots = assign(need, pressure=_press(22.49, 20.0), host=_host(64, 60), policy="cpu")
    if slots.device != "cpu":
        problems.append(f"策略 cpu 却上了卡：{slots.to_dict()}")

    # ⑦ 滞回：单次 hard 不动作，连续两次才动作 ✓
    watch = LoadWatch(confirm=2)
    watch.sample(_press(22.49, 1.0))          # 95%+ ⇒ 第一次
    if watch.wants_reclaim:
        problems.append("单次超硬阈值就动手了（会抖）")
    watch.sample(_press(22.49, 1.0))          # 第二次 ⇒ 认
    if not watch.wants_reclaim:
        problems.append("连续两次超硬阈值却没认")
    watch.sample(_press(22.49, 8.0))          # 回到软阈值以下 ⇒ 立即 normal ✓
    if watch.state != "normal":
        problems.append(f"压力回落后没回 normal：{watch.to_dict()}")
    watch.sample(None)                        # 测不到 ⇒ 状态不动 ✓
    if watch.state != "normal":
        problems.append("读不到压力却改了状态")

    # ⑧ 账本：记、还、分开算 ✓
    ledger = Ledger()
    ledger.reserve("dit", bytes_=int(13.26 * GIB), device="cuda")
    ledger.reserve("text", bytes_=int(14.79 * GIB), device="cpu")
    if abs(ledger.gpuBytes - int(13.26 * GIB)) > 1:
        problems.append(f"账本 GPU 侧算错：{ledger.to_dict()}")
    if abs(ledger.ramBytes - int(14.79 * GIB)) > 1:
        problems.append(f"账本内存侧算错：{ledger.to_dict()}")
    ledger.release("dit")
    if ledger.gpuBytes != 0 or "dit" in ledger.items:
        problems.append(f"释放没生效：{ledger.to_dict()}")

    # ⑨ 回收判据：阶段表没登记 ⇒ 一个都不退 ✓（猜着退就会卸掉正在用的那件 ✗）
    present = {"dit": True, "text": True, "vae": False}
    if reclaim_for("sample", present, None):
        problems.append("阶段表没登记却退了件（会卸掉正在用的）")
    if reclaim_for("sample", present, ("dit",)) != ["text"]:
        problems.append("本阶段用不着的件没退干净")
    if reclaim_for("encode", present, ("dit", "text")):
        problems.append("本阶段要用的件被退了")
    return problems


if __name__ == "__main__":  # pragma: no cover - 手工入口
    issues = selftest()
    print("\n".join(issues) if issues else "scheduler selftest: ok ✓")
    raise SystemExit(1 if issues else 0)
