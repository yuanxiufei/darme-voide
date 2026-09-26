"""**素材任务的运行时** ✓ —— 改名 / 转格式 / 打标 的排队与执行 ✓。

形状与 :mod:`app.services.lora_train.runtime` **完全一致** ✓
（同样是「提交时同步校验 ✓、单任务串行 ✓、日志有界可增量取 ✓、可取消 ✓」✓）
—— 因为这两边面对的是同一件事：**一次只能有一个东西在动同一批素材** ✓。

⚠️ **为什么素材任务与训练任务是两条独立的队列** ✗（不合成一条 ✓）：
* 训练吃 GPU ✓（一跑几小时 ✓），素材里除了打标都是纯 CPU/IO ✓（几十秒 ✓）；
* 合在一起会让"改个文件名"排在"训练 8 小时"后面 ✓✗；
* 而它们本来也**不共享同一份稀缺资源** ✓ —— 除了打标要显存 ✓，
  那一种冲突由路由层显式处理 ✓（打标前先卸推理引擎 ✓，见 ``routers/lora_train.py`` ✓）。

⚠️ **取消的粒度是诚实的** ✓（不是"点了就立刻停"✗）：
* 打标 ✓ / 转格式 ✓ ⇒ **每张图之间**查一次停止标记 ✓ ⇒ 停得住 ✓；
* 改名 ✗ ⇒ **不响应取消** ✓ —— 它本来就是秒级的 ✓，而"改到一半停下"会留下
  「前 30 张新名、后 70 张旧名」的目录 ✓✗ 比不停更糟 ✓。
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..errors import LoraTrainConfigError, LoraTrainNotFoundError
from ..progress import LogBuffer
from ..runner import OutputLine
from . import caption, fields, files

#: 终结态 ✓（与训练侧同一组词 ✓）
TERMINAL = ("succeeded", "failed", "cancelled")

#: 只读动作（不改磁盘 ✓）—— 前端可以放心自动跑 ✓
READ_ONLY_ACTIONS = ("scan",)

#: 每个工具的默认动作 ✓
DEFAULT_ACTIONS = {
    fields.TOOL_RENAME: "rename",
    fields.TOOL_CONVERT: "convert",
    fields.TOOL_CAPTION: "caption",
}

#: 每个工具允许的动作 ✓（不认 ⇒ 提交时报错点名 ✓）
ACTIONS = {
    fields.TOOL_RENAME: ("rename",),
    fields.TOOL_CONVERT: ("convert",),
    fields.TOOL_CAPTION: ("caption", "trigger", "filter", "scan"),
}

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _note(text: str) -> OutputLine:
    return OutputLine(f"[{time.strftime(_TIME_FORMAT)}] {text}", overwrite=False)


def release_inference_vram(*, force: bool = False) -> dict[str, Any]:
    """**打标前腾显存** ✓ —— 把自研推理引擎的张量卸掉 ✓。

    ⚠️ 为什么要这一步 ✗（参考实现没有 ✗ 因为它压根没有进程内引擎 ✓）：
    JoyCaption 是 8B ✓，推理引擎也常驻在**同一块卡**上 ✓ ⇒ 两边都想要显存时
    不是"慢一点" ✓，是**直接 OOM** ✓✗。⇒ 本仓**显式**做交接 ✓，
    而且**引擎正忙就报错** ✗（不 ``force`` 强卸 ✗ —— 那会把用户正在跑的视频任务搞失败 ✓✗）。

    返回：引擎卸之前/之后的显存事实 ✓（没装引擎 ⇒ ``{"unloaded": False, ...}`` ✓ ——
    这正是「本来就没占」✓，不是失败 ✗）。
    """
    from ...engine import runtime as engine_runtime_module  # noqa: PLC0415 - 重模块，按需 import ✓

    engine = engine_runtime_module.engine_runtime
    try:
        report = engine.unload(force=force)
    except engine_runtime_module.EngineBusy as err:
        raise LoraTrainConfigError(
            f"推理引擎正在跑任务 ✗ ⇒ 现在开打标会和它抢显存 ✓✗：{err}\n"
            "⇒ 要么等它跑完 / 先取消它 ✓，要么把「开跑前卸掉推理引擎」关掉、自己保证显存够 ✓"
        ) from err
    return {"unloaded": bool(report.get("unloaded", True)), "reclaimedGiB": report.get("reclaimedGiB"),
            "freeGiB": report.get("freeGiB")}


@dataclass
class DatasetTask:
    """一次素材任务 ✓。"""

    id: str
    tool: str
    action: str
    values: dict[str, Any]
    status: str = "queued"
    stage: str = "排队中"
    createdAt: int = 0
    startedAt: int | None = None
    finishedAt: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    logs: LogBuffer = field(default_factory=LogBuffer)
    cancelEvent: threading.Event = field(default_factory=threading.Event)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "action": self.action,
            "status": self.status,
            "stage": self.stage,
            "terminal": self.terminal,
            "createdAt": self.createdAt,
            "startedAt": self.startedAt,
            "finishedAt": self.finishedAt,
            "error": self.error,
            "result": self.result,
        }

    def to_dict(self, *, since: int = 0, log_limit: int = 200) -> dict[str, Any]:
        payload = self.to_summary()
        payload["log"] = self.logs.snapshot(since=since, limit=log_limit)
        return payload


class DatasetRuntime:
    """**单任务串行**的素材运行时 ✓（全局单例 :data:`dataset_runtime` ✓）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._tasks: dict[str, DatasetTask] = {}
        self._order: list[str] = []
        self._queue: list[str] = []
        self._current: str | None = None
        self._worker: threading.Thread | None = None

    # ── 队列 ✓ ───────────────────────────────────────────────────────────
    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, name="lora-train-dataset",
                                        daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:  # pragma: no cover - 线程循环（自检经 submit 走到 ✓）
        while True:
            with self._wake:
                while not self._queue:
                    self._wake.wait()
                task_id = self._queue.pop(0)
                task = self._tasks.get(task_id)
                if task is None:
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

            status, error, result = "failed", None, None
            try:
                status, error, result = self._execute(task)
            except BaseException as err:  # noqa: BLE001 - 后台线程的异常必须落到任务上 ✓
                status, error = "failed", f"运行时内部错误 ✗：{err!r}"

            with self._wake:
                task.status = status
                task.error = error
                task.result = result
                task.stage = {"succeeded": "已完成", "cancelled": "已取消"}.get(status, task.stage)
                task.finishedAt = _now_ms()
                self._current = None
                self._wake.notify_all()

    # ── 执行 ✓ ───────────────────────────────────────────────────────────
    def _execute(self, task: DatasetTask) -> tuple[str, str | None, dict[str, Any] | None]:
        emit: Callable[[str], None] = lambda text: task.logs.append(_note(text))
        stop: Callable[[], bool] = task.cancelEvent.is_set
        values = task.values

        if task.action == "scan":
            dataset = files.require_directory(values.get("dataset_path", ""))
            stats = files.caption_stats(dataset)
            emit(f"扫到 {stats['images']} 张图 ✓；有标签 {stats['labeled']} ✓、缺标签 {stats['missing']} ✓")
            return "succeeded", None, {"kind": "scan", **stats, "dataset": str(dataset)}

        if task.action == "rename":
            dataset = files.require_directory(values.get("dataset_path", ""))
            task.stage = "规划改名"
            result = files.run_rename(
                dataset,
                suffix=str(values.get("target_suffix", "")),
                prefix=str(values.get("target_prefix", "")),
                digits=int(values.get("target_num", 4)),
                rename_caption=bool(values.get("rename_caption", True)),
                emit=emit,
            )
            return "succeeded", None, {"kind": "rename", **result, "dataset": str(dataset)}

        if task.action == "convert":
            dataset = files.require_directory(values.get("dataset_path", ""))
            task.stage = "备份 + 转格式"
            result = files.run_convert(
                dataset,
                target_suffix=str(values.get("target_suffix", "")),
                backup=bool(values.get("backup", True)),
                remove_original=bool(values.get("remove_original", False)),
                overwrite=bool(values.get("overwrite", True)),
                quality=int(values.get("quality", 95)),
                emit=emit,
                should_stop=stop,
            )
            status = "cancelled" if result.get("stopped") else "succeeded"
            return status, None, {"kind": "convert", **result, "dataset": str(dataset)}

        if task.action == "trigger":
            dataset = files.require_directory(values.get("dataset_path", ""))
            task.stage = "写触发词"
            result = caption.run_trigger_caption(
                dataset, str(values.get("trigger_word", "")), emit=emit, should_stop=stop,
            )
            return ("cancelled" if task.cancelEvent.is_set() else "succeeded"), None, {
                "kind": "trigger", **result, "dataset": str(dataset)}

        if task.action == "filter":
            dataset = files.require_directory(values.get("dataset_path", ""))
            task.stage = "过滤标签"
            result = caption.run_filter_caption(
                dataset, str(values.get("filter_word", "")), emit=emit, should_stop=stop,
            )
            return ("cancelled" if task.cancelEvent.is_set() else "succeeded"), None, {
                "kind": "filter", **result, "dataset": str(dataset)}

        if task.action == "caption":
            request = caption.CaptionRequest.from_values(values)
            task.stage = "加载打标模型"
            emit(f"素材目录：{request.dataset}")
            if bool(values.get("free_vram", True)) and request.device.startswith("cuda"):
                facts = release_inference_vram()
                emit(f"已卸掉推理引擎腾显存 ✓：{facts}")
            result = caption.run_caption(request, emit=emit, should_stop=stop)
            return ("cancelled" if task.cancelEvent.is_set() else "succeeded"), None, {
                "kind": "caption", **result, "dataset": str(request.dataset)}

        raise LoraTrainConfigError(
            f"素材工具「{task.tool}」没有这个动作 ✗：{task.action!r}；它有：{list(ACTIONS[task.tool])}"
        )

    # ── 对外 API ✓ ────────────────────────────────────────────────────────
    def submit(self, tool: str, action: str | None = None,
               payload: Mapping[str, Any] | None = None) -> str:
        """入队一次素材任务 ✓ ⇒ ``taskId`` ✓（校验与取值都在**这里同步**做完 ✓）。"""
        resolved = fields.spec(tool)
        chosen = (action or DEFAULT_ACTIONS[resolved.key]).strip()
        if chosen not in ACTIONS[resolved.key]:
            raise LoraTrainConfigError(
                f"素材工具「{resolved.label}」没有这个动作 ✗：{chosen!r}；"
                f"它有：{list(ACTIONS[resolved.key])}"
            )
        values = fields.merge(resolved.key, payload)
        # ⚠️ 目录与打标参数**在提交时就验** ✓（见模块头"同步校验" ✓）——
        #    这样"目录写错了"是 400 ✓，而不是任务跑起来才失败 ✓。
        files.require_directory(values.get("dataset_path", ""))
        if chosen == "caption":
            caption.CaptionRequest.from_values(values)
        if chosen == "rename":
            int(values.get("target_num", 4))
            if not str(values.get("target_suffix", "")).strip():
                raise LoraTrainConfigError("改名：没填素材格式 ✗")
            if not str(values.get("target_prefix", "")).strip():
                raise LoraTrainConfigError("改名：没填命名前缀 ✗（否则会把文件改成纯数字名 ✓✗）")
        if chosen == "convert" and not str(values.get("target_suffix", "")).strip():
            raise LoraTrainConfigError("转格式：没填目标格式 ✗")
        if chosen == "trigger" and not str(values.get("trigger_word", "")).strip():
            raise LoraTrainConfigError("写触发词：触发词是空的 ✗")
        if chosen == "filter" and not str(values.get("filter_word", "")).strip():
            raise LoraTrainConfigError("过滤标签：过滤词是空的 ✗")

        task_id = f"data-{uuid.uuid4().hex[:12]}"
        task = DatasetTask(
            id=task_id, tool=resolved.key, action=chosen, values=values, createdAt=_now_ms(),
        )
        with self._wake:
            self._ensure_worker()
            self._tasks[task.id] = task
            self._order.append(task.id)
            self._queue.append(task.id)
            self._wake.notify_all()
        return task.id

    def get_task(self, task_id: str, *, since: int = 0, log_limit: int = 200) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise LoraTrainNotFoundError(f"没有这个素材任务 ✗：{task_id!r}")
            return task.to_dict(since=since, log_limit=log_limit)

    def list_tasks(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(reversed(self._order))[:max(0, int(limit))]
            return [self._tasks[task_id].to_summary() for task_id in ids]

    def cancel(self, task_id: str) -> dict[str, Any]:
        with self._wake:
            task = self._tasks.get(task_id)
            if task is None:
                raise LoraTrainNotFoundError(f"没有这个素材任务 ✗：{task_id!r}")
            task.cancelEvent.set()
            if not task.terminal and task.id in self._queue:
                self._queue.remove(task.id)
                task.status = "cancelled"
                task.stage = "已取消（还没轮到它 ✓）"
                task.finishedAt = _now_ms()
            self._wake.notify_all()
        return self.get_task(task_id, log_limit=0)

    def wait(self, task_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._wake:
            while True:
                task = self._tasks.get(task_id)
                if task is None:
                    raise LoraTrainNotFoundError(f"没有这个素材任务 ✗：{task_id!r}")
                if task.terminal:
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._wake.wait(timeout=remaining)
        return self.get_task(task_id, log_limit=0)

    def busy(self) -> bool:
        with self._lock:
            return bool(self._queue) or self._current is not None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "busy": bool(self._queue) or self._current is not None,
                "current": self._current,
                "queued": list(self._queue),
                "taskCount": len(self._tasks),
                "captionModel": caption.caption_engine.describe(),
                "actions": {key: list(value) for key, value in ACTIONS.items()},
                "readOnlyActions": list(READ_ONLY_ACTIONS),
            }


#: 全局单例 ✓（与 ``lora_train_runtime`` 同一形状 ✓）
dataset_runtime = DatasetRuntime()


__all__ = ["ACTIONS", "DEFAULT_ACTIONS", "DatasetRuntime", "DatasetTask", "dataset_runtime"]
