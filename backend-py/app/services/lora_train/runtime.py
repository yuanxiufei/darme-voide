"""**训练任务运行时** ✓ —— 排队 / 单任务串行 / 取消 / 日志 / 自动关机 ✓。

## 为什么是"单任务串行" ✗

参考实现（``reference/lora/LoRAMaster`` ✓）是 NiceGUI 单体 ✓ ⇒ 它全局只有**一份**
``train_process`` 变量 ✓（``wan_lora_train.py`` 第 471 行附近 ✓）⇒ **本来就没有并发语义** ✓。
本仓照抄这条 ✓：一张卡同时跑两份训练只会互相抢显存 ✓✗（本仓 ``errors`` 里的
:class:`~app.services.lora_train.errors.LoraTrainBusyError` 就是为它准备的 ✓）。

## 任务的两步结构 ✓

一个任务 = **预缓存（0~2 条命令 ✓）+ 训练（1 条命令 ✓）** ✓ ——
这正是参考实现 ``run_cache()`` 里那段"先 1/2 再 2/2、然后才训练"的顺序 ✓。
FLUX 没有预缓存 ✓（``cache=()`` ✓，见 ``options._flux`` 的说明 ✓）⇒ 那个任务只有 1 步 ✓。

## 与参考实现**刻意不同** ✓

1. **工作目录 = 任务目录** ✓：参考实现用相对路径 ``./output`` / ``./logs`` ✓ 而 cwd 是它自己的仓库 ✓✗
   ⇒ 产物会落在代码目录里 ✓。本仓把子进程 ``cwd`` 设成**任务自己的目录** ✓，
   于是 ``./output``、``./logs`` 自然各归各任务 ✓（本仓「产物落点必须说清」✓）。
2. **提交前把错拦掉** ✓：``validate``（参数 ✓）与 ``build_plan``（工具在不在 ✓）都在
   :meth:`submit` 里**同步**做完 ✓ ⇒ 请求线程就能回 400 / 409 ✓，
   不用等后台线程跑起来才说"你参数错了" ✓✗。
3. **日志有界 + 可增量取** ✓：见 :mod:`app.services.lora_train.progress` ✓。
4. **自动关机带撤单** ✓：见 :mod:`app.services.lora_train.shutdown` ✓。
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from app.core import cpu_budget

from . import config, shutdown
from .commands import build_plan
from .errors import (
    LoraTrainConfigError,
    LoraTrainNotFoundError,
    LoraTrainRunError,
)
from .options import merge, spec
from .paths import ResolvedTool, environment_report, resolve_tool
from .progress import LogBuffer
from .runner import CommandRunner, OutputLine, StepRecord, child_env

#: 终结态 ✓（到了这三种就不再变 ✓ —— 别用 ``status != "running"`` 判 ✗：``queued`` 也不是终结 ✓✗）
TERMINAL = ("succeeded", "failed", "cancelled")

#: 每一步跑完时给日志文件补的那句时间戳格式 ✓
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class LoraTrainTask:
    """一次训练任务的全部事实 ✓（**成功/失败都留着** ✓ —— 前端要能回看日志与产物 ✓）。"""

    id: str
    modelKey: str
    #: 本次任务**实际用的**完整取值 ✓（默认 + 文件 + 请求覆盖 ✓，来源可追溯 ✓）
    values: dict[str, Any]
    workspace: Path
    logPath: Path
    #: ``CommandPlan.as_dict()`` ✓
    plan: dict[str, Any]
    #: ``cpu_budget.describe()`` 的事实 ✓（"用几个线程、凭什么" ✓）
    cpu: dict[str, Any]
    status: str = "queued"
    stage: str = "排队中"
    steps: list[StepRecord] = field(default_factory=list)
    createdAt: int = 0
    startedAt: int | None = None
    finishedAt: int | None = None
    error: str | None = None
    #: 自动关机的**事实** ✓（没开就是 ``None`` ✓ —— 不是 ``False`` ✓）
    shutdownFacts: dict[str, Any] | None = None
    logs: LogBuffer = field(default_factory=LogBuffer)
    cancelEvent: threading.Event = field(default_factory=threading.Event)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "modelKey": self.modelKey,
            "status": self.status,
            "stage": self.stage,
            "terminal": self.terminal,
            "createdAt": self.createdAt,
            "startedAt": self.startedAt,
            "finishedAt": self.finishedAt,
            "error": self.error,
            "steps": [step.as_dict() for step in self.steps],
            "progress": self.logs.facts,
        }

    def to_dict(self, *, since: int = 0, log_limit: int = 200) -> dict[str, Any]:
        payload = self.to_summary()
        payload.update({
            "workspace": str(self.workspace),
            "logPath": str(self.logPath),
            "values": dict(self.values),
            "plan": dict(self.plan),
            "cpu": dict(self.cpu),
            "shutdown": self.shutdownFacts,
            "log": self.logs.snapshot(since=since, limit=log_limit),
        })
        return payload


def resolve_values(model_key: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """算出一份**完整取值** ✓ ⇒ 交给 ``commands`` ✓。

    * ``payload`` 省略 ⇒ 读该模型 ``settings.toml`` + 默认值 ✓（``config.load_values`` ✓）；
    * ``payload`` 给了 ⇒ **文件为底、请求覆盖** ✓ —— 前端"改两个字段就跑"时不用把整份都传上来 ✓；
      ⚠️ 两段都过 :func:`options.merge` ✓ ⇒ 未知键照样当场报错 ✓（不静默丢 ✗）。
    """
    if payload is None:
        return config.load_values(model_key)
    base = config.read_toml(config.settings_path(model_key))
    return merge(model_key, {**base, **dict(payload)})


class LoraTrainRuntime:
    """**单任务串行**的训练运行时 ✓（全局单例 :data:`lora_train_runtime` ✓）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._tasks: dict[str, LoraTrainTask] = {}
        #: 提交顺序 ✓（列表按它倒序 ✓ —— 不按时间戳排 ✗：同毫秒提交会乱序 ✓✗）
        self._order: list[str] = []
        self._queue: list[str] = []
        self._current: str | None = None
        self._runner: CommandRunner | None = None
        self._worker: threading.Thread | None = None

    # ── 队列 ✓ ───────────────────────────────────────────────────────────
    def _ensure_worker(self) -> None:
        """懒启守护线程 ✓（导入本模块**不**起线程 ✓ —— 没提交过任务就一个线程都不多 ✓）。"""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, name="lora-train-runtime", daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:  # pragma: no cover —— 线程循环（自检经 submit 走到 ✓）
        while True:
            with self._wake:
                while not self._queue:
                    self._wake.wait()
                task_id = self._queue.pop(0)
                task = self._tasks.get(task_id)
                if task is None:  # 理论不发生 ✓（清任务表时也清队列 ✓）
                    continue
                if task.cancelEvent.is_set():
                    task.status = "cancelled"
                    task.stage = "已取消（还没轮到它 ✓）"
                    task.finishedAt = _now_ms()
                    self._wake.notify_all()
                    continue
                self._current = task_id
                task.status = "running"
                task.startedAt = _now_ms()

            status, error = "failed", None
            try:
                status, error = self._execute(task)
            except BaseException as err:  # noqa: BLE001 - 后台线程的异常必须落到任务上 ✓（不静默吞 ✗）
                status, error = "failed", f"运行时内部错误 ✗：{err!r}"

            with self._wake:
                task.status = status
                task.error = error
                if status == "cancelled":
                    task.stage = "已取消"
                elif status == "succeeded":
                    task.stage = "已完成"
                task.finishedAt = _now_ms()
                self._current = None
                self._runner = None
                self._wake.notify_all()

    # ── 执行 ✓ ───────────────────────────────────────────────────────────
    def _execute(self, task: LoraTrainTask) -> tuple[str, str | None]:
        """按「预缓存 1/2 ⇒ 预缓存 2/2 ⇒ 训练」跑 ✓ ⇒ ``(终结状态, 错误说明)`` ✓。"""
        resolved = spec(task.modelKey)
        # ⚠️ ``submit`` 里已经 :func:`resolve_tool` 过一次 ✓（那时是为了把"没装"提前报出来 ✓）；
        # 这里再解析一次是为了拿到 ``pythonpath`` ✓（env 用 ✓）—— 同一份候选表 ✓，结果必然相同 ✓。
        train_tool = resolve_tool(resolved.train_script, toolkit=resolved.toolkit)
        cache_tools = [resolve_tool(name, toolkit=resolved.toolkit) for name in resolved.cache_scripts]

        runner = CommandRunner(on_line=task.logs.append)
        with self._lock:
            self._runner = runner

        total = len(task.plan.get("cache", [])) + 1
        for index, argv in enumerate(task.plan.get("cache", [])):
            if task.cancelEvent.is_set():
                return ("cancelled", None)
            tool = cache_tools[index] if index < len(cache_tools) else train_tool
            outcome = self._run_step(task, runner, f"预缓存 {index + 1}/{total - 1}",
                                     list(argv), tool)
            if outcome is not None:
                return (outcome, None)

        if task.cancelEvent.is_set():
            return ("cancelled", None)
        outcome = self._run_step(task, runner, f"训练 {total}/{total}",
                                 list(task.plan.get("train", [])), train_tool)
        if outcome is not None:
            return (outcome, None)

        if task.values.get("auto_shutdown"):
            task.shutdownFacts = shutdown.schedule()
            task.logs.append(_note_line(
                "自动关机：已排上 ✓" if task.shutdownFacts.get("scheduled")
                else f"自动关机：没排上 ✗（{task.shutdownFacts.get('reason')}）"
            ))
            task.logs.flush_progress()
        return ("succeeded", None)

    def _run_step(self, task: LoraTrainTask, runner: CommandRunner, name: str,
                  argv: list[str], tool: ResolvedTool | None) -> str | None:
        """跑一步 ✓ ⇒ ``None`` = 这步过了 ✓；否则返回终结状态 ✓。"""
        if not argv:
            raise LoraTrainRunError(f"第「{name}」步的命令是空的 ✗ —— 这是命令组装层的问题 ✓")
        step = StepRecord(name=name, argv=argv, state="running")
        task.steps.append(step)
        task.stage = name
        task.logs.append(_note_line(f"$ {' '.join(argv)}"))
        task.logs.flush_progress()

        with self._lock:
            still_cancelled = task.cancelEvent.is_set()
        if still_cancelled:
            step.state = "cancelled"
            return "cancelled"

        if tool is None:
            raise LoraTrainRunError(f"第「{name}」步没定位到工具脚本 ✗ —— 这是提交前就该拦掉的 ✓")

        result = runner.run(argv, env=child_env(tool), cwd=task.workspace)
        step.returncode = result.returncode
        step.seconds = result.seconds
        step.lines = result.lines
        step.kill = result.kill
        task.logs.flush_progress()
        task.logs.append(_note_line(
            f"「{name}」结束：返回码 {result.returncode} ✓ 用时 {result.seconds:.1f}s ✓ "
            f"输出 {result.lines} 行 ✓" + (f" 终止方式 {result.kill} ✓" if result.kill else "")
        ))
        task.logs.flush_progress()

        if result.cancelled:
            step.state = "cancelled"
            return "cancelled"
        if result.returncode != 0:
            step.state = "failed"
            return f"第「{name}」步返回码 {result.returncode} ✗（看日志与上面的命令行 ✓）"
        step.state = "done"
        return None

    # ── 对外 API ✓ ────────────────────────────────────────────────────────
    def submit(self, model_key: str, payload: Mapping[str, Any] | None = None) -> str:
        """入队一次训练 ✓ ⇒ ``taskId`` ✓（**立即返回** ✗ 不等它跑完 ✓）。

        ⚠️ 三件事都在**这里同步**做完 ✓（见模块头"刻意不同"第 2 条 ✓）：
        :func:`options.merge`（默认+覆盖+逐键校验 ✓）⇒
        :func:`commands.build_plan`（``validate`` ✓ + 工具在不在 ✓ + 拼命令 ✓）。
        任何一件不过 ⇒ **当场抛** ✓，任务根本不会入队 ✓✗。
        """
        values = resolve_values(model_key, payload)
        task_id = f"lora-{uuid.uuid4().hex[:12]}"
        workspace = config.data_dir() / "tasks" / task_id
        workspace.mkdir(parents=True, exist_ok=True)
        log_path = workspace / "train.log"
        # ⚠️ ``build_plan`` 会在这里写一个 ``prompt.txt`` ✓（要出样图且没用自定义文件时 ✓）——
        # 所以它必须在"确认能跑"之后 ✓、在任务真正入队之前 ✓，顺序见 ``commands.build_plan`` ✓。
        plan = build_plan(model_key, values, workspace=workspace)
        try:
            cpu = cpu_budget.describe()
        except cpu_budget.CpuBudgetError as err:
            raise LoraTrainConfigError(f"CPU 线程预算不可用 ✗：{err}") from err

        def _persist(text: str) -> None:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")

        task = LoraTrainTask(
            id=task_id,
            modelKey=spec(model_key).key,
            values=values,
            workspace=workspace,
            logPath=log_path,
            plan=plan.as_dict(),
            cpu=cpu,
            logs=LogBuffer(persist=_persist),
            createdAt=_now_ms(),
        )
        with self._wake:
            self._ensure_worker()
            self._tasks[task.id] = task
            self._order.append(task.id)
            self._queue.append(task.id)
            self._wake.notify_all()
        return task.id

    def get_task(self, task_id: str, *, since: int = 0, log_limit: int = 200) -> dict[str, Any]:
        """取任务快照 ✓；不认识 ⇒ :class:`LoraTrainNotFoundError` ✓（**不返回 ``None``** ✗）。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise LoraTrainNotFoundError(f"没有这个训练任务 ✗：{task_id!r}")
            return task.to_dict(since=since, log_limit=log_limit)

    def list_tasks(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(reversed(self._order))[:max(0, int(limit))]
            return [self._tasks[task_id].to_summary() for task_id in ids]

    def cancel(self, task_id: str) -> dict[str, Any]:
        """请求取消 ✓ —— **排队中的**立刻终结 ✓ / **跑着的**杀整棵进程树 ✓。"""
        with self._wake:
            task = self._tasks.get(task_id)
            if task is None:
                raise LoraTrainNotFoundError(f"没有这个训练任务 ✗：{task_id!r}")
            task.cancelEvent.set()
            if not task.terminal and task.id in self._queue:
                self._queue.remove(task.id)
                task.status = "cancelled"
                task.stage = "已取消（还没轮到它 ✓）"
                task.finishedAt = _now_ms()
            runner = self._runner
            self._wake.notify_all()
        if runner is not None and runner.running:
            # ⚠️ 杀进程树**不在锁里**做 ✗：``cancel`` 里有 ``wait`` ✓，持锁等会让
            #    ``get_task`` 一起卡住 ✓✗（本仓 runtime 的教训：临界区只放状态 ✓）。
            runner.cancel()
        return self.get_task(task_id, log_limit=0)

    def wait(self, task_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        """等任务终结 ✓（``timeout`` 秒；到点还没完 ⇒ 返回**当前**状态 ✓ —— 不抛 ✗）。"""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._wake:
            while True:
                task = self._tasks.get(task_id)
                if task is None:
                    raise LoraTrainNotFoundError(f"没有这个训练任务 ✗：{task_id!r}")
                if task.terminal:
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._wake.wait(timeout=remaining)
        return self.get_task(task_id, log_limit=0)

    def busy(self) -> bool:
        """有没有任务在跑/在排 ✓（自动关机这类"破坏性动作"的前置检查 ✓）。"""
        with self._lock:
            return bool(self._queue) or self._current is not None

    def status(self) -> dict[str, Any]:
        """**体检 + 队列** ✓ —— 前端"能不能跑、现在在跑什么" ✓。"""
        with self._lock:
            current = self._current
            queued = list(self._queue)
            task_count = len(self._tasks)
        return {
            "busy": bool(queued) or current is not None,
            "current": current,
            "queued": queued,
            "taskCount": task_count,
            "cpu": cpu_budget.describe(),
            "environment": environment_report(),
        }


#: 全局单例 ✓ —— 全进程**只有这一份**队列与任务表 ✓（与 ``services/engine`` 的 ``engine_runtime`` 同一形状 ✓）
lora_train_runtime = LoraTrainRuntime()


def _note_line(text: str) -> OutputLine:
    """本运行器自己往日志里补的一行 ✓（带时间戳 ✓ ⇒ 和子进程输出区分得开 ✓）。"""
    return OutputLine(f"[{time.strftime(_TIME_FORMAT)}] {text}", overwrite=False)


__all__ = [
    "TERMINAL",
    "LoraTrainRuntime",
    "LoraTrainTask",
    "lora_train_runtime",
    "resolve_values",
]
