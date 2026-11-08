"""参数调优的**后台任务表** —— 让 ``/tune`` 从「一次 7 分钟的同步请求」变成「入队 + 查进度」。

要解决的问题
============
``param_optimizer.optimize_agent_temperature`` 是 ``repeat × (1 + 候选数)`` 次**真模型评测**。
真机实测（``_bench/tune_probe.py``，2026-11-07）：eval-case-001 / repeat=2 / 1 个候选，
单跑 85–165 秒、四跑合计 **423.9 秒**。把它挂在 ``POST`` 的同步返回链上意味着：

1. 前端 fetch 不设超时会一直等，**中间代理/网关的超时却会先到** ⇒ 用户看到的是失败，
   而后台其实还在烧卡（这是最难查的一种：日志显示跑完了，界面显示超时）；
2. 这 7 分钟里没有任何中间态 —— 分不清「在跑第 1 跑」还是「卡死了」；
3. 用户点两次就排队跑两轮 7 分钟（同一个 case、同一份提示词，纯浪费）。

所以：**入队立刻返回 ``taskId``**，进度靠 ``GET /tune/{taskId}`` 拿，跑完的回执也在那里。

三条口径（都不是可选的）
========================
* **单工作线程 + 队列**：这台机器只有一张卡，两个调优并发跑只会互相抢显存、两个都变慢甚至 OOM。
  串行是唯一诚实的并发度 —— 排队就明说在排队（``queuePosition``）。
* **取消只在两次评测之间生效**：一次评测内部的真模型生成（85–165 秒）在别的线程里跑采样循环，
  从外面掐不断（见 ``engine.llm.generate_llm`` 的 ``abort`` 口径）。所以 ``cancel`` 的语义是
  **「不再开下一跑」**，不是「立刻停」。宁可这么写，也不假装能秒停。
* **重连自己的 DB 连接**：任务跑在一个**新的线程**里，而 ``get_conn`` 那条连接的生命周期跟着
  **HTTP 请求**走 —— 请求一返回连接就归还了。拿它继续跑后台任务，症状是任务跑到一半
  ``connection is closed``。故工作线程自己 ``engine.connect()``（每次任务一条，跑完就关）。

⚠️ 任务表是**进程内存**：进程重启后 ``GET /tune/{taskId}`` 就是 404。这是刻意的 —— 调优的产物
（判决与落库）已经写进 ``agent_optimizations`` / ``agent_configs``，重启后要查的「上次调成什么」
去那里看，本表只负责「这次跑到哪了」。
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from typing import Any, Awaitable, Callable, Mapping

from app.core import db as db_module
from app.agent.evaluation.param_optimizer import TuneCancelled, optimize_agent_temperature

__all__ = [
    "TERMINAL",
    "cancel",
    "get",
    "list_tasks",
    "reset",
    "submit",
    "wait",
]

#: 终态：到这四态之后不会再变（``queued``/``running`` 之外的都算）
TERMINAL = ("succeeded", "failed", "cancelled")

#: 最多留多少个任务（进程内存不许无限长 —— 掉的是**最老的已结束**那批）
_MAX_TASKS = 50

#: 任务快照里对外暴露的键（内部 ``_`` 开头的一律不出去：``_abort`` 只给 worker 看）
_RUNNER = "runner"

#: 一次任务要连的库连接（每次新开，见模块头第 3 条）
Runner = Callable[[Any, dict[str, Any]], Awaitable[dict[str, Any]]]

_tasks: dict[str, dict[str, Any]] = {}
_queue: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None
_lock = threading.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _stage_text(event: Mapping[str, Any]) -> str:
    """把进度事件翻成一句人话（前端就算不解析字段，显示这一句也不会误解）。"""
    kind = str(event.get("type") or "")
    done, total = event.get("doneRuns"), event.get("totalRuns")
    if kind == "start":
        return f"开始：共 {total} 跑（每档 {event.get('repeat')} 次）"
    if kind == "run":
        phase = "基线" if event.get("phase") == "baseline" else f"温度 {event.get('temperature')}"
        return f"{phase}：第 {event.get('run')}/{event.get('repeat')} 跑（总 {done}/{total}）"
    if kind == "baseline":
        return f"基线完成：均值 {event.get('mean')}，抖动带 {event.get('jump')}"
    if kind == "candidate":
        record = event.get("record") or {}
        return f"温度 {record.get('temperature')}：{record.get('reasonLabel') or record.get('reason')}"
    if kind == "done":
        return "已结束（回执见 receipt）"
    return "运行中"


def _snapshot(job: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(job.get("taskId") or "")
    out: dict[str, Any] = {
        "taskId": task_id,
        "caseId": job.get("caseId"),
        "agentType": job.get("agentType"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "createdAt": job.get("createdAt"),
        "startedAt": job.get("startedAt"),
        "finishedAt": job.get("finishedAt"),
        "repeat": (job.get("options") or {}).get("repeat"),
        "maxCandidates": (job.get("options") or {}).get("maxCandidates"),
        "autoPersist": (job.get("options") or {}).get("autoPersist"),
        "doneRuns": job.get("doneRuns") or 0,
        "totalRuns": job.get("totalRuns") or 0,
        "progress": job.get("progress"),
        "pollUrl": f"/api/v1/evaluation/tune/{task_id}",
    }
    if job.get("receipt") is not None:
        out["receipt"] = job["receipt"]
    if job.get("error"):
        out["error"] = job["error"]
    return out


def _queue_position(task_id: str) -> int:
    """排在它前面还有几个等着跑（``0`` = 下一个就是它；跑着的那个不算排队）。"""
    position = 0
    for other in _tasks.values():
        if other.get("taskId") == task_id:
            continue
        if other.get("status") == "queued" and int(other.get("createdAt") or 0) < \
                int(_tasks.get(task_id, {}).get("createdAt") or 0):
            position += 1
    return position


def _trim() -> None:
    """超出上限时丢掉**最老的已结束**任务（队列中/在跑的绝不丢）。"""
    if len(_tasks) <= _MAX_TASKS:
        return
    finished = [
        item for item in _tasks.values() if item.get("status") in TERMINAL
    ]
    finished.sort(key=lambda item: int(item.get("createdAt") or 0))
    for item in finished[: len(_tasks) - _MAX_TASKS]:
        _tasks.pop(str(item.get("taskId")), None)


# ---------------------------------------------------------------------------
# 工作线程
# ---------------------------------------------------------------------------


def _ensure_worker() -> None:
    """第一次提交时才起线程（import 这个模块不该顺手起线程）。"""
    global _worker
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(target=_loop, name="tune-worker", daemon=True)
        _worker.start()


def _loop() -> None:
    while True:
        task_id = _queue.get()
        job = _tasks.get(task_id)
        if job is None:
            continue
        if job.get("status") == "cancelled":
            continue        # 排队期间就被取消了 ⇒ 连跑都不跑
        _execute(job)


def _execute(job: dict[str, Any]) -> None:
    job["status"] = "running"
    job["startedAt"] = _now_ms()
    job["stage"] = "正在装载评测上下文"
    runner: Runner = job.get(_RUNNER) or _default_runner

    def progress(event: dict[str, Any]) -> None:
        job["progress"] = dict(event)
        if event.get("doneRuns") is not None:
            job["doneRuns"] = int(event["doneRuns"])
        if event.get("totalRuns") is not None:
            job["totalRuns"] = int(event["totalRuns"])
        job["stage"] = _stage_text(event)

    try:
        # ⚠️ 新连接：请求那条连接在响应返回时就归还了（模块头第 3 条）
        with db_module.engine.connect() as conn:
            job["progressCallback"] = progress
            receipt = asyncio.run(runner(conn, job))
    except TuneCancelled as err:
        job["status"] = "cancelled"
        job["error"] = str(err)
        job["stage"] = f"已取消（{err}）"
    except Exception as err:  # noqa: BLE001 —— 任何失败都要落到任务表上，不许只打日志
        job["status"] = "failed"
        job["error"] = f"{err.__class__.__name__}: {err}"
        job["stage"] = f"失败：{job['error']}"
    else:
        job["status"] = "succeeded"
        job["receipt"] = receipt
        job["stage"] = "已完成（回执见 receipt）"
    finally:
        job["finishedAt"] = _now_ms()
        job.pop("progressCallback", None)
        _trim()


async def _default_runner(conn: Any, job: dict[str, Any]) -> dict[str, Any]:
    """真正跑调优：把进度/取消接进优化器（优化器本身不认识队列）。"""
    return await optimize_agent_temperature(
        conn,
        str(job["agentType"]),
        job["caseDef"],
        job.get("options") or {},
        progress=job.get("progressCallback"),
        should_stop=lambda: bool(job.get("abort")),
    )


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------


def submit(
    *,
    case_id: str,
    case_def: Mapping[str, Any],
    agent_type: str,
    options: Mapping[str, Any],
    runner: Runner | None = None,
) -> dict[str, Any]:
    """入队一次调优 ⇒ 任务快照（含 ``taskId`` / ``queuePosition`` / ``pollUrl``）。

    ⚠️ **同一个 (agentType, caseId) 已在排队或正在跑 ⇒ 不重复入队**，直接把它还回去并标
    ``deduped=True``：同一个 case、同一份提示词，跑第二遍只是再烧 7 分钟。
    参数不同也不另起 —— 真想换参数，先取消手上这个（``POST /tune/{taskId}/cancel``）。
    """
    for item in _tasks.values():
        if item.get("status") in ("queued", "running") \
                and item.get("agentType") == agent_type \
                and item.get("caseId") == case_id:
            return {**_snapshot(item), "deduped": True,
                    "queuePosition": _queue_position(str(item.get("taskId")))}

    task_id = f"tune-{uuid.uuid4().hex[:12]}"
    job: dict[str, Any] = {
        "taskId": task_id,
        "caseId": case_id,
        "agentType": agent_type,
        "caseDef": dict(case_def),
        "options": {k: options[k] for k in ("repeat", "maxCandidates", "autoPersist")
                    if k in options},
        "status": "queued",
        "stage": "排队中",
        "createdAt": _now_ms(),
        "startedAt": None,
        "finishedAt": None,
        "doneRuns": 0,
        "totalRuns": int(options.get("repeat") or 0)
        * (1 + int(options.get("maxCandidates") or 0)),
        "progress": None,
        "receipt": None,
        "error": None,
        "abort": False,
        "deduped": False,
    }
    if runner is not None:
        job[_RUNNER] = runner
    _tasks[task_id] = job
    _ensure_worker()
    _queue.put(task_id)
    return {**_snapshot(job), "deduped": False, "queuePosition": _queue_position(task_id)}


def get(task_id: str) -> dict[str, Any] | None:
    """任务快照（不认识 ⇒ ``None``，由路由翻成 404）。"""
    job = _tasks.get(str(task_id or ""))
    return _snapshot(job) if job is not None else None


def list_tasks(*, limit: int = 20) -> list[dict[str, Any]]:
    """最近的调优任务（新的在前；不带回执正文，要看点开 ``GET /tune/{taskId}``）。"""
    items = sorted(_tasks.values(), key=lambda item: int(item.get("createdAt") or 0),
                   reverse=True)
    return [_snapshot(item) for item in items[: max(0, int(limit))]]


def cancel(task_id: str) -> dict[str, Any]:
    """取消 ⇒ 快照（不认识 ⇒ ``None``）。

    ``queued`` ⇒ 立刻是 ``cancelled``（worker 排到它时直接跳过）；
    ``running`` ⇒ 置标志，**下一次评测之前**才停（见模块头第 2 条）；
    终态 ⇒ 原样返回（``changed: False``，不报错 —— 「已经结束了」不是失败）。
    """
    job = _tasks.get(str(task_id or ""))
    if job is None:
        return None
    before = job.get("status")
    changed = False
    if before == "queued":
        job["status"] = "cancelled"
        job["stage"] = "已取消（还没开跑）"
        job["finishedAt"] = _now_ms()
        changed = True
    elif before == "running":
        job["abort"] = True
        job["stage"] = "取消中：跑完当前这一跑就停"
        changed = True
    return {**_snapshot(job), "changed": changed, "wasStatus": before}


def wait(task_id: str, *, timeout: float = 900.0, poll: float = 0.2) -> dict[str, Any] | None:
    """等到终态（给 ``wait=true`` 的同步调用方用；超时 ⇒ 返回当时快照并标 ``timedOut``）。"""
    deadline = time.time() + max(0.0, float(timeout))
    while True:
        job = _tasks.get(str(task_id or ""))
        if job is None:
            return None
        if job.get("status") in TERMINAL:
            return _snapshot(job)
        if time.time() >= deadline:
            return {**_snapshot(job), "timedOut": True}
        time.sleep(max(0.02, float(poll)))


def reset() -> None:
    """清空任务表（**测试专用**：让每个用例从空表开始；不影响已起的 worker 线程）。"""
    _tasks.clear()
