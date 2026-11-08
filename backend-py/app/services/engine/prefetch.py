"""**预取单槽**：趁 GPU 在算，把下一件权重先读到内存 ✓（移植自 ``airllm`` 上游 ✓）。

## 为什么照搬它 ✗

本仓的装配是**串行**的 ✓：DiT 装完才装 TE ✓、TE 装完才装 VAE ✓ —— 实测 Z-Image 整链 40.5 秒 ✓，
其中**绝大部分是等盘** ✓✗，而这 40 秒里 GPU 一直空着 ✓✗。``airllm_base`` 那份实现就干一件事：
进层之前先在**后台线程**把下一层读到 CPU ✓，用到时直接取 ✓（``_prefetch_stream`` / ``_materialize`` ✓），
取不到就退回同步读 ✓（**永不阻塞主路径** ✓）。

## 本仓口径下的四处改动（都不是能力，是纪律 ✓）

1. **不绑 torch / 不绑层** ✗：原件预取的是 ``state_dict`` ✓；这里只约 ``fn: () -> T`` ✓
   ⇒ 既能预取权重 ✓、也能预取 tokenizer / 配置 / 反量化结果 ✓；
2. **取到就作废** ✓：原件取走后 ``_prefetch_future=None`` ✓；这里同款 ✓（**同一份不许取两次** ✗✗ ——
   取两次＝两份常量驻留 ✓，19 GiB 的账立刻翻倍 ✓）；
3. **提交永不抛** ✓✗：原件在热路径里裸调 ``executor.submit`` ✓；本仓所有装载错误都要能落到
   ``loadReport`` ✓ ⇒ ``submit`` 一律返回回执 ✓（``scheduled`` / ``reason`` ✓），异常留到 **取的时候** 抛 ✓；
4. **要能看见** ✓：原件靠 ``print`` ✓✗；这里给 :meth:`Prefetcher.snapshot` ✓（在飞哪几件 / 命中几次 /
   几条错误 ✓）⇒ 直接喂 ``GET /engine/runtime`` 的加载状态反馈 ✓。

## 预取深度按盘速分档（§4.3 ✓，2026-11-07 ✓）

``max_inflight`` 原先由调用方写死 ✓✗（``weights._READ_AHEAD = 2`` ✓）。现在唯一口径是
:func:`resolve_depth` ✓：**显式参数 > ``DARME_READ_AHEAD`` > 盘档默认** ✓，盘档由 :func:`probe_disk`
真读 32 MiB 量顺序读吞吐得出 ✓（每块盘只量一次 ✓，量不出 ⇒ ``unknown`` ⇒ 沿用 2 ✓，**不改现状** ✓）。
计划的区间（NVMe 2–4 / SATA 1–2 ✓）只用来判**上限** ✓：机械盘上要 4 条并行读是有害的 ✓ ⇒ 截断并
如实报 ``capped`` ✓。§6 只认实测 ✓：本机 NVMe 上 2 槽已喂饱队列 ✓，4 留给别人自己量 ✓。

## 边界（**不做**的事 ✗）

* 不猜"该预取什么" ✗：调用方给 key 与函数 ✓（预测器是另一层 ✓，``expert-stream`` 的教训是
  预测打不中时**只会更慢** ✓✗）；
* 不做淘汰策略 ✗：只有"在飞"与"已就绪"两格 ✓，就绪的东西**必须**被取走或 :meth:`Prefetcher.drop` 掉 ✓；
* 不跨进程 ✗。

## 移植来源

* 上游：``airllm``（Apache-2.0），commit ``75a6bb3c41c6``；
* 原件：``air_llm/airllm/airllm_base.py`` 的 ``_prefetch_stream`` / ``_materialize`` / ``_evict``；
* 本文件不取原件的 ``ModelPersister`` / ``bitsandbytes`` 压缩路径 ✓（那两样本仓不需要 ✓）。
"""
from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

__all__ = ["PrefetchError", "Prefetcher", "READY", "PENDING", "MISSING", "FAILED",
           "DISK_CLASSES", "MAX_SLOTS", "normalize_depth", "probe_disk", "resolve_depth",
           "selftest"]

#: :meth:`Prefetcher.try_take` 的四种答案 ✓（字符串常量，前端/日志能直接读 ✓）。
READY = "ready"
PENDING = "pending"
MISSING = "missing"
FAILED = "failed"


class PrefetchError(RuntimeError):
    """预取**取的时候**失败 ✓（提交是成功的 ✓ —— 这是两件事 ✓✗）。"""

    def __init__(self, message: str, *, key: str, cause: str) -> None:
        super().__init__(message)
        self.key = key
        self.cause = cause


@dataclass
class _Slot:
    key: str
    future: Future
    started: float
    took: float = 0.0


@dataclass
class _Stats:
    submitted: int = 0
    hits: int = 0
    misses: int = 0
    failures: int = 0
    refused: int = 0
    lastError: str = ""


class Prefetcher:
    """单槽/多槽预取 ✓（``max_inflight=1`` 就是原件的行为 ✓）。

    ⚠️ 线程数是 ``max_inflight`` ✓：读盘是 I/O ✓ 不是 CPU ✓ ⇒ 2 槽就够把 NVMe 队列喂起来 ✓，
    再多只是让**同时驻留**的半成品变多 ✓✗（内存账）。
    """

    def __init__(self, *, name: str = "prefetch", max_inflight: int = 2) -> None:
        self.name = name
        self._max_inflight = max(1, int(max_inflight))
        self._executor = ThreadPoolExecutor(max_workers=self._max_inflight,
                                            thread_name_prefix=f"{name}-prefetch")
        self._lock = threading.Lock()
        self._slots: list[_Slot] = []
        self._closed = False
        self._stats = _Stats()
        self._created = time.time()

    # ---- 提交 -----------------------------------------------------------
    def submit(self, key: str, fn: Callable[[], Any]) -> dict[str, Any]:
        """后台把 ``fn()`` 跑掉 ✓ ⇒ 回执（**不抛** ✓✗）。

        拒绝的三种情形都是**正常**的 ✓：``inflight``（同一件已经在飞 ✓）、``ready``（已经就绪 ✓）、
        ``busy``（槽满了 ✓ —— 调用方当场同步读即可 ✓，别等 ✓）。
        """
        key = str(key)
        with self._lock:
            if self._closed:
                self._stats.refused += 1
                return {"scheduled": False, "reason": "closed", "key": key}
            self._reap_locked()
            if any(slot.key == key for slot in self._slots):
                self._stats.refused += 1
                reason = PENDING if not self._done_locked(key) else READY
                return {"scheduled": False, "reason": reason, "key": key}
            if len(self._slots) >= self._max_inflight:
                self._stats.refused += 1
                return {"scheduled": False, "reason": "busy", "key": key}
            try:
                future = self._executor.submit(self._run, key, fn)
            except RuntimeError as err:  # 执行器已关（极少见：close 与 submit 撞上）
                self._stats.refused += 1
                self._stats.lastError = str(err)
                return {"scheduled": False, "reason": "closed", "key": key}
            self._slots.append(_Slot(key=key, future=future, started=time.time()))
            self._stats.submitted += 1
            return {"scheduled": True, "reason": "submitted", "key": key}

    def _run(self, key: str, fn: Callable[[], Any]) -> Any:
        """工作线程里跑 ✓ —— 异常**不吞** ✓（存进 future ✓，取的时候再说 ✓✗）。"""
        try:
            return fn()
        except BaseException as err:  # noqa: BLE001 —— 必须原样带到取的那一端 ✓
            raise PrefetchError(f"预取 {key!r} 失败：{err}", key=key, cause=str(err)) from err

    # ---- 取 -------------------------------------------------------------
    def try_take(self, key: str) -> tuple[str, Any]:
        """**非阻塞**取 ✓ ⇒ ``(状态, 值)`` ✓（``ready`` 才有值 ✓；失败也**不抛** ✓✗）。"""
        key = str(key)
        with self._lock:
            slot = self._find_locked(key)
            if slot is None:
                self._stats.misses += 1
                return MISSING, None
            if not slot.future.done():
                return PENDING, None
            return self._take_locked(slot)

    def take(self, key: str, *, timeout: float | None = None) -> Any:
        """**阻塞**取 ✓（等盘读完 ✓）。失败 ⇒ :class:`PrefetchError` ✓；没这件 ⇒ ``KeyError`` ✓。"""
        key = str(key)
        with self._lock:
            slot = self._find_locked(key)
            if slot is None:
                self._stats.misses += 1
                raise KeyError(f"没有预取 {key!r}（没提交过，或已经被取走了 ✓✗）")
        try:
            value = slot.future.result(timeout=timeout)
        except BaseException as err:  # noqa: BLE001
            with self._lock:
                self._discard_locked(slot)
                self._stats.failures += 1
                self._stats.lastError = str(err)
            if isinstance(err, PrefetchError):
                raise
            raise PrefetchError(f"预取 {key!r} 失败：{err}", key=key, cause=str(err)) from err
        with self._lock:
            self._discard_locked(slot)
            self._stats.hits += 1
        return value

    def drop(self, key: str) -> bool:
        """不要了 ✓（就绪的东西不取就等于**一直占着** ✓✗ ⇒ 必须有这个口 ✓）。"""
        key = str(key)
        with self._lock:
            slot = self._find_locked(key)
            if slot is None:
                return False
            self._discard_locked(slot)
            return True

    # ---- 内部 -----------------------------------------------------------
    def _find_locked(self, key: str) -> _Slot | None:
        for slot in self._slots:
            if slot.key == key:
                return slot
        return None

    def _done_locked(self, key: str) -> bool:
        slot = self._find_locked(key)
        return bool(slot is not None and slot.future.done())

    def _take_locked(self, slot: _Slot) -> tuple[str, Any]:
        slot.took = time.time() - slot.started
        try:
            value = slot.future.result()
        except BaseException as err:  # noqa: BLE001
            self._discard_locked(slot)
            self._stats.failures += 1
            self._stats.lastError = str(err)
            return FAILED, None
        self._discard_locked(slot)
        self._stats.hits += 1
        return READY, value

    def _discard_locked(self, slot: _Slot) -> None:
        try:
            self._slots.remove(slot)
        except ValueError:  # pragma: no cover —— 同一槽被取两次
            pass

    def _reap_locked(self) -> None:
        """只清**已经失败**的槽 ✓✗（在飞的绝不动 ✗、就绪的也绝不动 ✗ —— 就绪是资产 ✓）。"""
        for slot in list(self._slots):
            if not slot.future.done():
                continue
            error = slot.future.exception()
            if error is None:
                continue  # 就绪 ≠ 该死 ✓：等取 ✓
            self._discard_locked(slot)
            self._stats.failures += 1
            self._stats.lastError = str(error)

    # ---- 可见面 ---------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """加载状态反馈 ✓（在飞哪几件 / 就绪哪几件 / 计数 / 最后一次错误 ✓）。"""
        with self._lock:
            inflight = [slot.key for slot in self._slots if not slot.future.done()]
            ready = [slot.key for slot in self._slots if slot.future.done()]
            stats = _Stats(**self._stats.__dict__)
            closed = self._closed
            max_inflight = self._max_inflight
        return {
            "name": self.name,
            "closed": closed,
            "maxInflight": max_inflight,
            "inflight": inflight,
            "ready": ready,
            "submitted": stats.submitted,
            "hits": stats.hits,
            "misses": stats.misses,
            "failures": stats.failures,
            "refused": stats.refused,
            "lastError": stats.lastError,
            "elapsedSeconds": round(time.time() - self._created, 3),
        }

    def close(self, *, wait: bool = True) -> None:
        """关掉 ✓（``wait=True`` 等在飞的读完 ✓ —— 释放 19 GiB 前必须等 ✓✗）。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            slots = list(self._slots)
            self._slots.clear()
        for slot in slots:
            slot.future.cancel()
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "Prefetcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---- 预取深度：按盘速分档（§4.3）------------------------------------------
#
#: 盘档 ⇒ ``(默认深度, 允许上限, 人话)`` ✓✗（计划原文：「预取并发度按盘速自适应，
#: **NVMe 2–4 / SATA 1–2**」✓）。
#:
#: * **默认不取区间上沿** ✗：2 槽就够把盘的队列喂起来 ✓（见 :class:`Prefetcher` 的说明 ✓），
#:   上沿 4 只作为**显式**能给到的值 ✓ —— 别人机器上要不要 4 得他自己量 ✓，本仓不猜 ✗（§6 只认实测 ✓）；
#: * SATA 的 1 同理：两条能把「读」与「反量化／写卡」叠起来 ✓ ⇒ 默认 2（区间上沿 ✓）；
#: * **机械盘必须 1** ✓✗：两条并行读只会让磁头来回跑 ✓，比顺序读更慢 ✓；
#: * ``unknown`` **不改现状** ✓：测不出盘速就退回本仓一直在用的 2 ✓（**不是**猜一个新数 ✓）。
DISK_CLASSES: dict[str, tuple[int, int, str]] = {
    "nvme": (2, 4, "NVMe（顺序读 ≥ 1000 MiB/s）"),
    "sata": (2, 2, "SATA SSD（250–1000 MiB/s）"),
    "hdd": (1, 1, "机械盘（< 250 MiB/s）"),
    "unknown": (2, 2, "测不出盘速（小文件／读不了）⇒ 沿用 2"),
}

#: 显式值／``DARME_READ_AHEAD`` 的合法上限 ✓（再多只是让**同时驻留**的半成品变多 ✗：单件上限 256 MiB ✓）。
MAX_SLOTS = 8
#: 测速取样：32 MiB 够分辨三档 ✓（本机 NVMe ≈11 ms ✓、机械盘 ≈0.2 s ✓），**每块盘只量一次** ✓。
PROBE_BYTES = 32 * 1024 * 1024
#: ⚠️ 块**必须小** ✗✗（2026-11-07 实测 ✓）：``chunk = handle.read(n)`` 是「先申请新的、再释放旧的」✓
#: ⇒ 瞬时**同时活着两块** ✓ ⇒ 峰值 ＝ ``2 × PROBE_CHUNK`` ✓。用 4 MiB 块时探一次就要 8 MiB ✓✗，
#: 把 ``engine_weights_streaming_test`` 的「关预取时峰值 ≈ 单件（< 8 MiB）」顶红了 ✓（8.01 MiB ✓）。
#: 1 MiB 块 ⇒ 探盘峰值 2 MiB ✓，远低于任何一份权重的装载峰值 ✓（读数本身不变 ✗：只与总字节/时长有关 ✓）。
PROBE_CHUNK = 1 * 1024 * 1024
#: 小于 8 MiB 的文件不测 ✓✗：临时文件／小权重测出来的数只反映页缓存 ✓（会得出「这块盘 3 GB/s」的笑话 ✓）。
PROBE_MIN_BYTES = 8 * 1024 * 1024
#: 分档线（MiB/s ✓）：`win_io` 在本机 NVMe 实测 2.79~3.44 GiB/s ✓、SATA SSD 约 450~550 ✓、机械 120~200 ✓。
NVME_MIN_MIB = 1000.0
SATA_MIN_MIB = 250.0
#: 盘 ⇒ 量过的结论 ✓（键＝盘符／挂载点 ✓）。这不是「缓存结论」✗，是「别每装一件就再读 32 MiB」✓。
_DISK_CACHE: dict[str, dict[str, Any]] = {}


def _drive_key(path: Any) -> str:
    """盘符／挂载点 ✓（同一块盘上装十件只该量一次 ✓✗）。"""
    try:
        anchor = Path(path).resolve().anchor
        return (str(anchor) or str(Path(path).parent)).lower()
    except (OSError, ValueError, RuntimeError):  # pragma: no cover - 路径怪到 resolve 都拒
        return str(path).lower()


def _classify(mbps: float) -> str:
    if mbps >= NVME_MIN_MIB:
        return "nvme"
    if mbps >= SATA_MIN_MIB:
        return "sata"
    return "hdd"


def _fake_mbps() -> float | None:
    """G4 夹具 ✓：``DARME_FAKE_DISK_MBPS`` 假装量到某个吞吐 ✓（写错**当场炸** ✓✗，不许静默忽略 ✓）。"""
    raw = os.environ.get("DARME_FAKE_DISK_MBPS")
    if raw is None or not str(raw).strip():
        return None
    try:
        value = float(str(raw).strip())
    except ValueError as err:
        raise ValueError(
            f"DARME_FAKE_DISK_MBPS={raw!r} 不是数 ✗（夹具变量写错就该当场炸 ✓，"
            f"静默忽略会让人以为「已经切到 SATA 档」✓✗）") from err
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"DARME_FAKE_DISK_MBPS={raw!r} 必须是正的有限数 ✗")
    return value


def probe_disk(path: Any = None) -> dict[str, Any]:
    """真读一小段量**顺序读吞吐** ⇒ ``{"disk", "mbps", "how", "why"}`` ✓（量不出 ⇒ ``disk="unknown"`` ✓）。

    ⚠️ 目的是**分档**、不是评分 ✓：只需要回答「≥1000 / ≥250 MiB/s」这两个门 ✓，所以取文件中段、
    1 MiB 一块（见 :data:`PROBE_CHUNK` ✓）、最多 32 MiB ✓。**不做随机读** ✗（那是体检／§7 的事 ✓），
    也不写盘 ✗。
    """
    fake = _fake_mbps()
    if fake is not None:
        return {"disk": _classify(fake), "mbps": fake, "how": "fake",
                "why": f"夹具 DARME_FAKE_DISK_MBPS={fake:g} MiB/s"}
    if path is None:
        return {"disk": "unknown", "mbps": None, "how": "none", "why": "没给文件 ⇒ 无从量起"}
    key = _drive_key(path)
    hit = _DISK_CACHE.get(key)
    if hit is not None:
        return {**hit, "how": "cache"}
    try:
        size = Path(path).stat().st_size
    except OSError as err:
        return {"disk": "unknown", "mbps": None, "how": "none", "why": f"读不到文件大小：{err}"}
    if size < PROBE_MIN_BYTES:
        return {"disk": "unknown", "mbps": None, "how": "none",
                "why": f"文件只有 {size} 字节（< {PROBE_MIN_BYTES >> 20} MiB）⇒ 量出来的和盘速无关"}
    offset = (size // 2 // PROBE_CHUNK) * PROBE_CHUNK
    read_total = 0
    started = time.perf_counter()
    try:
        with open(path, "rb", buffering=0) as handle:
            handle.seek(offset)
            while read_total < PROBE_BYTES:
                chunk = handle.read(min(PROBE_CHUNK, PROBE_BYTES - read_total))
                if not chunk:
                    break
                read_total += len(chunk)
    except OSError as err:
        return {"disk": "unknown", "mbps": None, "how": "none", "why": f"测速读失败：{err}"}
    elapsed = max(time.perf_counter() - started, 1e-6)
    if read_total <= 0:
        return {"disk": "unknown", "mbps": None, "how": "none", "why": "一个字节都没读到"}
    mbps = read_total / float(1 << 20) / elapsed
    answer = {"disk": _classify(mbps), "mbps": mbps, "how": "measured",
              "why": f"读 {read_total >> 20} MiB 用了 {elapsed * 1000:.0f} ms"}
    _DISK_CACHE[key] = answer
    return answer


def normalize_depth(value: Any, *, field: str = "readAhead") -> int | None:
    """预取深度 → ``int | None`` ✓（``None``／空串／``"auto"`` ＝ **没提** ✓ ⇒ 交给盘档表 ✓）。

    坏值**一律当场抛** ✓✗（静默回落到默认值 ⇒ 调用方以为旋钮生效了 ✓，实际什么也没发生 ✓✗）。
    口径照 :func:`residency.normalize_gpu_layers` 的同一条纪律：**不抄 llama.cpp 的负数**（那是
    ``--n-gpu-layers`` 的「全上卡」✓✗），本仓的负数没有含义 ✓、``0`` 才是**明确关掉** ✓。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} 不接受布尔值 {value!r} ✗（要关预取就给 0 ✓）")
    slots: int
    if isinstance(value, int):
        slots = value
    elif isinstance(value, float):
        raise ValueError(f"{field}={value!r} 不是整数 ✗（槽数只能是整数 ✓）")
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "auto":
            return None
        try:
            slots = int(text)
        except ValueError as err:
            raise ValueError(
                f"{field}={value!r} 不是整数 ✗（**不写**＝按盘速自适应 ✓，关掉给 0 ✓）") from err
    else:
        raise ValueError(f"{field} 的类型不认识：{type(value).__name__} ✗")
    if slots < 0:
        raise ValueError(f"{field}={slots} 是负数 ✗（负数在别的工具里表示「全都上」，本仓没有这个口径 ✗；"
                         f"要关预取给 0 ✓）")
    if slots > MAX_SLOTS:
        raise ValueError(f"{field}={slots} 超上限 {MAX_SLOTS} ✗（再多只是让同时驻留的半成品变多 ✓）")
    return slots


def resolve_depth(path: Any = None, *, explicit: Any = None) -> dict[str, Any]:
    """这一趟装载**该用多深的预取** ✓ ⇒ 深度 ＋ 凭据（回执／观测面直接读 ✓）。

    优先级：**显式参数 > ``DARME_READ_AHEAD`` > 盘档默认** ✓，但显式值也要**截到该盘档的上限** ✓✗
    —— 在机械盘上要 4 条并行读是**有害**的 ✓（磁头来回跑 ✓），所以截断并如实报 ``capped`` ✓
    （`capped=True` 时 ``value < requested`` ✓，不是静默改成别的数 ✓）。
    """
    disk = probe_disk(path)
    default, hi, label = DISK_CLASSES.get(str(disk["disk"]), DISK_CLASSES["unknown"])
    asked: int | None = None
    source = "disk"
    explicit_slots = normalize_depth(explicit, field="read_ahead")
    if explicit_slots is not None:
        asked, source = explicit_slots, "caller"
    else:
        env_slots = normalize_depth(os.environ.get("DARME_READ_AHEAD"),
                                    field="DARME_READ_AHEAD")
        if env_slots is not None:
            asked, source = env_slots, "env"
    if asked is None:
        value, capped = default, False
        why = f"没明说 ⇒ 按盘档取默认 {default}（{label} ✓）"
    else:
        value = min(asked, hi)
        capped = value != asked
        who = "调用方" if source == "caller" else "DARME_READ_AHEAD"
        if capped:
            why = f"{who}给 {asked} 槽，超过 {disk['disk']} 档上限 {hi} ⇒ 截到 {value}（{label} ✓）"
        else:
            why = f"{who}给 {asked} 槽，盘档允许（上限 {hi} ✓）⇒ 照给"
    return {
        "value": value,
        "source": source,
        "requested": asked,
        "capped": capped,
        "default": default,
        "hi": hi,
        "disk": disk["disk"],
        "diskClass": label,
        "mbps": None if disk["mbps"] is None else round(float(disk["mbps"]), 1),
        "how": disk["how"],
        "why": why,
    }


def selftest() -> int:
    """自检 ✓（不碰盘、不碰 torch ✓：跑三个纯计数任务 ✓）。"""
    import concurrent.futures

    failures: list[str] = []
    calls: list[str] = []
    gate = threading.Event()

    def job(name: str) -> str:
        calls.append(name)
        gate.wait(2.0)
        return f"value-{name}"

    with Prefetcher(name="selftest", max_inflight=2) as pf:
        first = pf.submit("a", lambda: job("a"))
        second = pf.submit("a", lambda: job("a"))  # 同一件重复提交
        if first["scheduled"] is not True:
            failures.append("首次提交没被接受")
        if second["scheduled"] is not False or second["reason"] != PENDING:
            failures.append(f"重复提交没被拒：{second}")
        if pf.try_take("a")[0] != PENDING:
            failures.append("未完成时 try_take 不是 pending")
        if pf.try_take("nope")[0] != MISSING:
            failures.append("没提交过的 key 不是 missing")
        gate.set()
        state, value = (PENDING, None)
        deadline = time.time() + 5
        while time.time() < deadline:
            state, value = pf.try_take("a")
            if state == READY:
                break
            time.sleep(0.01)
        if state != READY or value != "value-a":
            failures.append(f"取到的不是就绪值：{state} {value!r}")
        if pf.try_take("a")[0] != MISSING:
            failures.append("取过还能再取（⚠️ 同一份取两次＝两份驻留 ✗）")
        if calls != ["a"]:
            failures.append(f"同一件被执行了多次：{calls}")
        snap = pf.snapshot()
        if snap["hits"] != 1 or snap["submitted"] != 1:
            failures.append(f"计数不对：{snap}")

    # 失败要留到「取」的那一端 ✓
    with Prefetcher(name="selftest-fail", max_inflight=1) as pf:
        pf.submit("boom", lambda: (_ for _ in ()).throw(RuntimeError("炸了")))
        deadline = time.time() + 5
        answer: tuple[str, Any] = (PENDING, None)
        while time.time() < deadline:
            answer = pf.try_take("boom")
            if answer[0] in (READY, FAILED, MISSING):
                break
            time.sleep(0.01)
        if answer[0] != FAILED:
            failures.append(f"失败没被报成 failed：{answer}")
        if not pf.snapshot()["lastError"]:
            failures.append("失败没进 lastError")

    # 盘档分档（§4.3 ✓）：只验表与纯函数 ✓，**不读盘** ✗（真读由 `probe_disk` 的调用方决定 ✓）。
    for cls in ("nvme", "sata", "hdd", "unknown"):
        if cls not in DISK_CLASSES:
            failures.append(f"盘档表缺 {cls}")
            continue
        default, hi, _label = DISK_CLASSES[cls]
        if not 1 <= default <= hi:
            failures.append(f"{cls} 档默认 {default} 不在 1..上限 {hi}")
        if cls == "hdd" and default != 1:
            failures.append(f"机械盘默认深度不是 1（{default}）⇒ 磁头会来回跑")
    if DISK_CLASSES["nvme"][1] < 4 or DISK_CLASSES["sata"][1] < 2:
        failures.append("上限低于计划原文的 NVMe 4 / SATA 2")
    saved_fake = os.environ.pop("DARME_FAKE_DISK_MBPS", None)
    saved_slots = os.environ.pop("DARME_READ_AHEAD", None)
    try:
        os.environ["DARME_FAKE_DISK_MBPS"] = "120"
        if probe_disk("x")["disk"] != "hdd" or resolve_depth("x")["value"] != 1:
            failures.append("120 MiB/s 没被判成机械盘 / 深度不是 1")
        os.environ["DARME_FAKE_DISK_MBPS"] = "2500"
        plan = resolve_depth("x")
        if plan["disk"] != "nvme" or plan["value"] != 2:
            failures.append(f"2500 MiB/s 该判 NVMe 且默认 2：{plan}")
        os.environ["DARME_FAKE_DISK_MBPS"] = "2500"
        bad = resolve_depth("x", explicit=6)
        if bad["value"] != 4 or bad["capped"] is not True:
            failures.append(f"NVMe 上要 6 槽没被截到 4 并报 capped：{bad}")
        os.environ["DARME_READ_AHEAD"] = "3"
        if resolve_depth("x")["value"] != 3:
            failures.append("DARME_READ_AHEAD=3 没赢过盘档默认")
        os.environ["DARME_READ_AHEAD"] = "0"
        if resolve_depth("x")["value"] != 0:
            failures.append("DARME_READ_AHEAD=0 没被当成「明确关掉」")
        os.environ["DARME_READ_AHEAD"] = "abc"
        try:
            resolve_depth("x")
            failures.append("坏环境变量没当场抛")
        except ValueError:
            pass
    finally:
        os.environ.pop("DARME_FAKE_DISK_MBPS", None)
        os.environ.pop("DARME_READ_AHEAD", None)
        if saved_fake is not None:
            os.environ["DARME_FAKE_DISK_MBPS"] = saved_fake
        if saved_slots is not None:
            os.environ["DARME_READ_AHEAD"] = saved_slots

    for line in failures:
        print("FAIL  " + line)
    print(f"SUMMARY: {'OK' if not failures else 'FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
