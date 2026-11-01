"""**预取单槽**：趁 GPU 在算，把下一件权重先读到内存 ✓（移植自 ``reference/airllm`` ✓）。

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

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["PrefetchError", "Prefetcher", "READY", "PENDING", "MISSING", "FAILED", "selftest"]

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

    for line in failures:
        print("FAIL  " + line)
    print(f"SUMMARY: {'OK' if not failures else 'FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
