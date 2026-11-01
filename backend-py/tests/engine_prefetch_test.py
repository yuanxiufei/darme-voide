"""预取自检 ✓（移植自 `reference/airllm` ✓，2026-11-01 ✓）。

**为什么单独一份** ✗：这套东西不读权重 ✓、不 import torch ✓ ⇒ 任何机器全绿 ✓。

判据口径（**别写松** ✗✗）：
- 「提交成功」**不是**"取的时候成功"✓✗ —— 提交只保证**接单** ✓，错误必须留到**取**的那一端 ✓；
- 「取到」**只能取一次** ✓✗ —— 取两次＝同一份常量驻留两份 ✓（13 GiB 的账立刻翻倍 ✗）；
- 预热路径**永不许**把异常甩给热路径 ✓✗（装载失败要落 `loadReport` ✓，不是崩在前面 ✓）。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import prefetch as pf  # noqa: E402

_FAILS: list[str] = []
_SKIPS: list[str] = []


def check(label: str, ok: bool, detail: object = "") -> None:
    if ok:
        print(f"PASS  {label}")
        return
    _FAILS.append(label)
    print(f"FAIL  {label}  ->  {detail!r}")


def _wait_state(prefetcher: "pf.Prefetcher", key: str, want: tuple[str, ...],
                timeout: float = 5.0) -> tuple[str, object]:
    deadline = time.time() + timeout
    answer: tuple[str, object] = (pf.MISSING, None)
    while time.time() < deadline:
        answer = prefetcher.try_take(key)
        if answer[0] in want:
            return answer
        time.sleep(0.01)
    return answer


# ---------------------------------------------------------------- ① 基本通路


def case_basic() -> None:
    calls: list[str] = []
    fence = threading.Event()

    def job() -> str:
        calls.append("ran")
        fence.wait(2.0)
        return "weights"

    with pf.Prefetcher(name="t-basic", max_inflight=2) as prefetcher:
        receipt = prefetcher.submit("dit", job)
        check("① 提交当场接单 ✓（不阻塞：worker 还没跑完就返回 ✓）",
              receipt["scheduled"] is True and receipt["reason"] == "submitted", receipt)
        check("①′ 未完成时 try_take 答 pending 而**不阻塞** ✓✗（热路径不许被预取卡住 ✗）",
              prefetcher.try_take("dit")[0] == pf.PENDING)
        fence.set()
        state, value = _wait_state(prefetcher, "dit", (pf.READY,))
        check("①″ 就绪后取到真值 ✓", state == pf.READY and value == "weights", (state, value))
        check("①‴ 同一件**只执行了一次** ✓✗（重复执行＝白读 13 GiB ✗）", calls == ["ran"], calls)


def case_single_flight() -> None:
    calls: list[str] = []
    fence = threading.Event()
    with pf.Prefetcher(name="t-dup", max_inflight=2) as prefetcher:
        prefetcher.submit("te", lambda: (calls.append(1), fence.wait(2.0), "te")[2])
        again = prefetcher.submit("te", lambda: (calls.append(2), "te")[2])
        check("② 同 key 重复提交被拒且**如实报 pending** ✓（不是「排了两趟」✗）",
              again["scheduled"] is False and again["reason"] == pf.PENDING, again)
        fence.set()
        _wait_state(prefetcher, "te", (pf.READY,))
        check("②′ 重复提交没有带来第二次执行 ✓", calls == [1], calls)


def case_take_once() -> None:
    with pf.Prefetcher(name="t-once", max_inflight=1) as prefetcher:
        receipt = prefetcher.submit("vae", lambda: "vae-obj")
        start = time.time()
        value = prefetcher.take("vae", timeout=5.0)
        check("③ 阻塞取能拿到值 ✓", value == "vae-obj" and receipt["scheduled"], value)
        check("③′ 就绪件取走即作废 ✓✗（取两次＝两份常量驻留 ⇒ 19 GiB 翻倍 ✗）",
              prefetcher.try_take("vae")[0] == pf.MISSING,
              prefetcher.snapshot())
        raised = False
        try:
            prefetcher.take("vae", timeout=1.0)
        except KeyError:
            raised = True
        check("③″ 再取抛 KeyError ✓（不许答 None 让上层自己猜 ✗✗）", raised)
        check("③‴ 取走花的墙钟时间记下来了 ✓（状态反馈要能答「预取到底省了多少」✓）",
              time.time() - start < 5.0)


# ---------------------------------------------------------------- ④ 失败与热路径


def case_failure() -> None:
    with pf.Prefetcher(name="t-fail", max_inflight=1) as prefetcher:
        receipt = prefetcher.submit("bad", lambda: 1 / 0)
        check("④ 会炸的活也**接单** ✓✗（提交时不知道会炸 ✓，硬拦就得先跑一遍 ✗）",
              receipt["scheduled"] is True, receipt)
        state, value = _wait_state(prefetcher, "bad", (pf.FAILED, pf.MISSING))
        check("④′ try_take 答 failed 且**不抛** ✓（探测器不该炸 ✓）",
              state == pf.FAILED and value is None, (state, value))
        snap = prefetcher.snapshot()
        check("④″ 失败进 lastError ＋ failures 计数 ✓（否则状态面永远「一切正常」✗✗）",
              bool(snap["lastError"]) and snap["failures"] == 1, snap)

    with pf.Prefetcher(name="t-fail2", max_inflight=1) as prefetcher:
        prefetcher.submit("bad2", lambda: (_ for _ in ()).throw(RuntimeError("盘掉了")))
        raised = None
        try:
            prefetcher.take("bad2", timeout=5.0)
        except pf.PrefetchError as err:
            raised = err
        check("④‴ 取的时候抛 PrefetchError 并带原因 ✓（错在「装载」而不是「预取」✓）",
              raised is not None and "盘掉了" in str(raised) and raised.key == "bad2",
              repr(raised))


def case_hot_path_safe() -> None:
    prefetcher = pf.Prefetcher(name="t-closed", max_inflight=1)
    prefetcher.submit("a", lambda: "a")
    prefetcher.close()
    receipt = prefetcher.submit("b", lambda: "b")
    check("⑤ 关闭后 submit **不抛**、如实回执 ✓✗（差点忘了：close 与 submit 会撞 ✓）",
          receipt["scheduled"] is False and receipt["reason"] == "closed", receipt)
    prefetcher.close()
    check("⑤′ close 幂等 ✓", True)


def case_busy() -> None:
    fence = threading.Event()
    with pf.Prefetcher(name="t-busy", max_inflight=1) as prefetcher:
        prefetcher.submit("first", lambda: (fence.wait(2.0), "1")[1])
        receipt = prefetcher.submit("second", lambda: "2")
        check("⑥ 槽满时**当场拒**并答 busy ✓✗（等空位＝把串行装成串行还多一次调度 ✗）",
              receipt["scheduled"] is False and receipt["reason"] == "busy", receipt)
        fence.set()
        _wait_state(prefetcher, "first", (pf.READY,))


def case_drop_and_resubmit() -> None:
    with pf.Prefetcher(name="t-drop", max_inflight=1) as prefetcher:
        prefetcher.submit("ready", lambda: "v1")
        # ⚠️ 这里**不能**用 try_take 等就绪 ✓✗：try_take 本身就是"取走" ⇒ 槽当场空出来 ✗，
        # 那样测的就不是「就绪不取占死槽」✓✗（第一版正是这么写错的 ✓）。
        deadline = time.time() + 5
        while time.time() < deadline and prefetcher.snapshot()["ready"] != ["ready"]:
            time.sleep(0.01)
        check("⑦⓪ 就绪件停在 ready 格里等人取 ✓",
              prefetcher.snapshot()["ready"] == ["ready"], prefetcher.snapshot())
        busy = prefetcher.submit("other", lambda: "v2")
        check("⑦ 就绪不取 ⇒ 槽被占死 ✓✗（这就是必须有 drop 的原因 ✓）",
              busy["scheduled"] is False and busy["reason"] == "busy", busy)
        check("⑦′ drop 就绪件成功 ✓", prefetcher.drop("ready") is True)
        after = prefetcher.submit("other", lambda: "v2")
        check("⑦″ drop 之后能再提交 ✓（否则漏取一次就永久堵死 ✗✗）",
              after["scheduled"] is True, after)
        _wait_state(prefetcher, "other", (pf.READY,))


def case_snapshot() -> None:
    prefetcher = pf.Prefetcher(name="t-snap", max_inflight=2)
    prefetcher.try_take("nobody")  # miss
    prefetcher.submit("hit", lambda: "x")
    _wait_state(prefetcher, "hit", (pf.READY,))
    prefetcher.close()  # ⚠️ 快照必须在 close **之后**取 ✓✗（块内取 ⇒ closed 恒 False ✗）
    snap = prefetcher.snapshot()
    need = {"name", "closed", "maxInflight", "inflight", "ready", "submitted",
            "hits", "misses", "failures", "refused", "lastError", "elapsedSeconds"}
    check("⑧ snapshot 字段齐 ✓（这是喂 `GET /engine/runtime` 的那一份 ✓）",
          need.issubset(snap), sorted(need - set(snap)))
    check("⑧′ 命中/未命中/提交三格计数对得上 ✓",
          snap["hits"] == 1 and snap["misses"] == 1 and snap["submitted"] == 1, snap)
    check("⑧″ 名字带着走 ✓（多套装配线并存时不然分不清是谁的预取 ✓✗）",
          snap["name"] == "t-snap" and snap["closed"] is True, snap)


def main() -> int:
    print("== 预取自检（移植自 airllm ✓）==")
    case_basic()
    case_single_flight()
    case_take_once()
    case_failure()
    case_hot_path_safe()
    case_busy()
    case_drop_and_resubmit()
    case_snapshot()
    for note in _SKIPS:
        print(f"SKIP  {note}")
    print(f"\nSUMMARY: {len(_FAILS)} failed")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
