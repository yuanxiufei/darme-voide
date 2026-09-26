"""自研引擎的**进程内运行时** ✓ —— 装载 / 排队 / 跑 / 卸载，**全在本进程里完成** ✓。

## 为什么必须有它（此前缺的就是这一层）

`torch_backend.TorchBackend` 只是「一个会算张量的后端」✓ —— 它**不知道**：权重从哪来（清单 ✓）、
TE 与 VAE 要不要挂 ✓、同一时刻能不能跑两个任务（显存 ✗）、跑完什么时候把显存还回去 ✓。
这些**装配与调度**的事实需要一个持有者 ✗ —— 此前**没有** ✗ ⇒ 自研引擎只有 ``/engine/dry-run``
干跑那条路 ✓✗，「真跑一次」连**入口**都没有 ✗（`torch_backend` 模块头 ``PENDING_PARTS`` 第 2 条 ✓）。

## ⚠️ 底线：**不依赖任何外部服务** ✗✗

不用 ComfyUI(8188) ✓、不用 SD WebUI ✓、不用 Ollama ✓、不起任何子进程 ✓。
对照两条老路：``services/comfyui.py`` 是「提交工作流给外部实例」✗；``gpu_manager`` 的卸载靠 **HTTP
通知**外部服务（Ollama ``keep_alive=0`` / SD ``unload-checkpoint`` / ComfyUI ``/free`` ✓）✗ ⇒
那套对**进程内**的引擎**不适用** ✗：我们的卸载就是**丢张量 + 清缓存** ✓（:meth:`EngineRuntime.unload` ✓）。

## 三条纪律

1. **一次只跑一个** ✓（FIFO ✓）：模型驻留显存，两个任务并行 = OOM ✗ ⇒ 串行不是"省事" ✗，
   是**唯一正确**的做法 ✓（这正是"不依赖外部"的代价：没有外部服务替我们排队 ✓）。
2. **装载幂等 + 可换** ✓：同参数复用 ✓（大权重只装一次 ✗）；参数变了才重装 ✓；
   有任务在跑时**不许卸载** ✗（:class:`EngineBusy` ✓）。
3. **如实自述** ✓：``synthetic`` **永远** ``True`` ✓✗ —— 参考 TE / VAE **未经训练** ✗ ⇒
   产物是「真张量 + 真文件 + 噪声画面」✓；真权重到位前不许改口说是真画面 ✗✗。

## 失败口径

装配失败 ⇒ **抛** :class:`EngineUnavailable`（带 ``reason`` ✓：``deps`` 依赖没装 ✓ /
``pending`` 权重或结构没就绪 ✓ / ``busy`` 正忙 ✓），且**不半装** ✗：失败时模块一个字不变 ✓
（与 `torch_backend` 装载「校验失败就中止 + 不污染模块」同一条规矩 ✓）。

⚠️ **本模块全是同步的** ✗（worker 是守护线程 ✓）⇒ 调用方负责别卡事件循环 ✓
（``await asyncio.to_thread(...)`` ✓ 或 :meth:`EngineRuntime.submit` 入队 ✓）。
"""
from __future__ import annotations

import gc
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import audio_vae as audio_vae_mod
from . import pipeline as pipe
from . import vae as vae_mod
from .dryrun import DryRunBackend
from .torch_backend import (
    PENDING_PARTS,
    TorchBackend,
    TorchBackendUnavailable,
    dependency_status,
    torch_available,
)

__all__ = [
    "EngineBusy",
    "EngineRuntime",
    "EngineTask",
    "EngineUnavailable",
    "engine_runtime",
]

#: 终结态 ✓（到了这三种就不再变 ✓ —— 别用 `status != "running"` 判 ✗：``queued`` 也不是终结 ✓✗）
_TERMINAL = ("done", "failed", "cancelled")


class EngineUnavailable(RuntimeError):
    """引擎不可用 ✓ —— ``reason`` 说明**缺什么** ✓。"""

    def __init__(self, message: str, *, reason: str = "pending") -> None:
        super().__init__(message)
        self.reason = reason


class EngineBusy(EngineUnavailable):
    """引擎**正忙** ✓（有任务在跑/在排 ⇒ 此刻不许卸载 ✓）—— 与"没装好"分开报 ✓✗。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="busy")


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class EngineTask:
    """一次引擎任务的全部事实 ✓（**成功了也留着** ✓ —— 前端要能回看产物与阶段耗时 ✓）。"""

    id: str
    stage: str
    request: Any
    #: 本次任务的装配口径 ✓（路径 / 设备 / 取整口径 ✓ —— 与"当时到底装了什么"一一对应 ✓）
    assembly: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    createdAt: int = 0
    startedAt: int | None = None
    finishedAt: int | None = None
    #: 管线事件 ✓（每步一条 ✓ —— 前端轮询它画进度 ✓）
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    #: 装载报告 ✓（``None`` = 干跑 / 还没装 ✓ —— 别默认 ``{}`` ✗：分不清"没装"与"装了没报告" ✓✗）
    loadReport: dict[str, Any] | None = None
    #: 取消旗标 ✓ —— 管线每步查一次 ✓（``pipeline.run_sync(cancel=…)`` ✓）
    cancelEvent: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL

    def elapsed_ms(self) -> int:
        return max(0, (self.finishedAt or _now_ms()) - self.createdAt)

    def to_summary(self) -> dict[str, Any]:
        """列表用的瘦身版 ✓（**不带事件** ✗ —— 否则 30 步的任务会把列表撑爆 ✓✗）。"""
        outputs = (self.result or {}).get("outputs") or {}
        return {
            "id": self.id, "status": self.status, "stage": self.stage,
            "createdAt": self.createdAt, "startedAt": self.startedAt,
            "finishedAt": self.finishedAt, "elapsedMs": self.elapsed_ms(),
            "eventCount": len(self.events), "terminal": self.terminal,
            "videoPath": outputs.get("videoPath"), "audioPath": outputs.get("audioPath"),
            "primaryPath": outputs.get("primaryPath"), "error": self.error,
        }

    def to_dict(self, *, events_limit: int = 200) -> dict[str, Any]:
        """完整版 ✓ —— 事件**截断但报总数** ✓（截了不说 = 看着像"就这些步" ✗✗）。"""
        keep = self.events[-events_limit:] if events_limit > 0 else []
        return {
            **self.to_summary(),
            "request": self.request.to_dict() if hasattr(self.request, "to_dict") else self.request,
            "assembly": dict(self.assembly),
            "loadReport": self.loadReport,
            "result": self.result,
            "events": keep,
            "eventsTruncated": len(self.events) - len(keep),
        }


def _load_key(**kwargs: Any) -> tuple[Any, ...]:
    """装配参数的**可比较快照** ✓（dict 不可哈希 ✗ ⇒ 摊成排序元组 ✓）。"""
    return tuple(sorted((key, repr(value)) for key, value in kwargs.items()))


def _trunk_dim(trunk: dict[str, Any], key: str) -> int:
    """从**已装主干的形态**里取一个尺寸 ✓ —— 取不到就**拒装** ✗（不猜一个"差不多"的 ✓✗）。"""
    value = (trunk or {}).get(key)
    if not isinstance(value, int) or value <= 0:
        raise TorchBackendUnavailable(
            f"从主干形态里读不出 `{key}` ✗（``describe().config`` 里没有它 ✓）⇒ **拒装参考 VAE** ✗。\n"
            f"⚠️ 宁可拒装也不猜 ✗：挂上**不是那一族**的 VAE，会「形状自洽、装得进去、"
            f"**出来不对而且不报错**」✗✗（倍率/通道数两边各算各的 ✓）。",
            reason="pending")
    return int(value)


def _attach_reference_video_vae(backend: TorchBackend, trunk: dict[str, Any]) -> dict[str, Any]:
    """挂 **H3 口径**的**参考**（⚠️ **未经训练** ✗）视频 VAE ✓ ⇒ 返回装载报告 ✓。

    ⚠️ 级数**从 H3 的 `vaeScale` 推** ✗（不是随手写个 5 ✓）：``spatial_scale = 2^(级数−1)`` ✓
    ⇒ 级数 = ``log2(vaeScale) + 1`` ✓。挂完**再核一遍**报告里的 ``spatialScale`` ✓ ——
    对不上就报 ✗（``attach_vae`` 自己也会拒 ✓，这里再钉一道 ✓ —— 一条事实两个来源 ✓）。
    """
    scale = int(vae_mod.H3_VIDEO_VAE_FACTS["vaeScale"])
    levels = int(math.log2(scale)) + 1
    channels = _trunk_dim(trunk, "latents_dim")
    report = backend.attach_vae(vae_mod.VideoVAEConfig(
        base_channels=channels, latent_channels=channels,
        channel_multipliers=tuple(1 for _ in range(levels))))
    if int(report.get("spatialScale") or 0) != scale:
        raise TorchBackendUnavailable(
            f"参考视频 VAE 的 spatial_scale={report.get('spatialScale')} ✗ ≠ H3 的 {scale} ✓ "
            f"⇒ **拒用** ✓（否则潜尺寸按一边算、解码按另一边放 ⇒ 出片尺寸与请求的不是一回事 ✓✗）",
            reason="pending")
    return report


def _attach_reference_audio_vae(backend: TorchBackend, trunk: dict[str, Any]) -> dict[str, Any]:
    """挂 **H3 口径**的**参考**（⚠️ **未经训练** ✗）音频 VAE ✓ ⇒ 返回装载报告 ✓。

    ⚠️ ``latent_channels`` **必须** = 主干 ``audio_latents_dim`` ✓（对不上解码端就报形状 ✓）；
    其余尺寸用 :data:`audio_vae.H3_AUDIO_VAE_DEFAULTS` 那套**出厂事实** ✓
    （32 潜通道 × 立体声 2 @ 40 Hz ✓ —— 与 ``geometry`` 的音频事实同源 ✓）。
    """
    channels = _trunk_dim(trunk, "audio_latents_dim")
    return backend.attach_audio_vae(audio_vae_mod.AudioVAEConfig(latent_channels=channels))


def _release_torch_cache() -> bool:
    """丢掉计算缓存 ✓ ⇒ 返回"真的清了吗" ✓（**没有 torch 就如实 False** ✗ 不抛 ✗）。"""
    try:
        import torch  # noqa: PLC0415 —— 可选依赖 ✓：没有它也要能报错而不是崩 ✓
    except Exception:
        return False
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return True
    except Exception:
        return False


class EngineRuntime:
    """进程内引擎的**唯一持有者** ✓（全局单例 :data:`engine_runtime` ✓）。

    ⚠️ 刻意**不是** asyncio 的形状 ✗：装载与推理**都阻塞** ✗ ⇒ 调用方负责别卡事件循环 ✓。
    """

    #: ⚠️ 与 :class:`~app.services.engine.torch_backend.TorchBackend` **同一个理由**保持 True ✓✗：
    #: 参考 TE / VAE **未经训练** ✗ ⇒ 权重齐了、链路通了，画面仍是噪声 ✓（如实标 ✓）。
    synthetic = True

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._backend: Any = None
        self._loadKeyTuple: tuple[Any, ...] | None = None
        self._loadReport: dict[str, Any] | None = None
        self._tasks: dict[str, EngineTask] = {}
        #: 提交顺序 ✓（列表按它倒序 ✓ —— 不按时间戳排 ✗：同毫秒提交会乱序 ✓✗）
        self._order: list[str] = []
        self._queue: list[str] = []
        self._current: str | None = None
        self._worker: threading.Thread | None = None

    # ── 队列 ✓ ───────────────────────────────────────────────────────────
    def _ensure_worker(self) -> None:
        """懒启守护线程 ✓（导入本模块**不**起线程 ✓ —— 没提交过任务就一个线程都不多 ✓）。"""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, name="engine-runtime", daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:  # pragma: no cover —— 线程循环（自检经 submit 走到 ✓）
        while True:
            with self._wake:
                while not self._queue:
                    self._wake.wait()
                task_id = self._queue.pop(0)
                task = self._tasks.get(task_id)
                if task is None:            # 理论不发生 ✓（清任务表时也清队列 ✓）
                    continue
                if task.cancelEvent.is_set():
                    task.status = "cancelled"
                    task.finishedAt = _now_ms()
                    self._wake.notify_all()
                    continue
                self._current = task_id
                task.status = "loading"
                task.startedAt = _now_ms()
            try:
                self._execute(task)
            except BaseException as err:  # noqa: BLE001 —— worker **绝不允许**死掉 ✗✗
                #  线程一死，后面所有任务永久 queued ✓✗ —— 那是最难查的静默故障 ✓
                self._finish_failed(task, f"{type(err).__name__}: {err}")
            finally:
                with self._wake:
                    self._current = None
                    self._wake.notify_all()

    def _execute(self, task: EngineTask) -> None:
        """装载（除非干跑 ✓）⇒ 跑管线 ⇒ 收口 ✓。**任何失败都变成任务里的 error** ✓ 不抛 ✗。"""
        dry = bool(task.assembly.get("dryRun"))
        # ⚠️ ``dryRun`` 是**运行时**的开关 ✗ ⇒ 不进 ``ensure_loaded`` ✓（否则 TypeError ✓✗）
        assembly = {key: value for key, value in task.assembly.items() if key != "dryRun"}
        backend: Any = DryRunBackend() if dry else None
        if not dry:
            try:
                report = self._ensure_loaded_internal(**assembly)
                task.loadReport = report
                task.events.append({
                    "kind": "load", "loaded": True, "reused": bool(report.get("reused")),
                    "device": report.get("device"),
                    "components": sorted((report.get("components") or {}).keys()),
                })
            except TorchBackendUnavailable as err:
                task.events.append({"kind": "load", "loaded": False,
                                    "reason": getattr(err, "reason", ""), "error": str(err)})
                self._finish_failed(task, f"装载失败：{err}")
                return
            backend = self._backend
            if backend is None:             # 报告成功却没后端 = 内部矛盾 ⇒ 响亮报 ✗
                self._finish_failed(task, "装载报告成功但运行时没有后端 ✗（内部状态不一致 ✓）")
                return
        if task.cancelEvent.is_set():
            task.status = "cancelled"
            task.finishedAt = _now_ms()
            return
        task.status = "running"
        result = pipe.run_sync(task.request, backend, on_event=task.events.append,
                              cancel=task.cancelEvent.is_set)
        task.result = result.to_dict()
        task.finishedAt = _now_ms()
        if result.cancelled:
            task.status = "cancelled"
        elif result.ok:
            task.status = "done"
        else:
            task.status = "failed"
            task.error = (result.error or {}).get("message") or "生成失败 ✓（看 result.error ✓）"

    def _finish_failed(self, task: EngineTask, message: str) -> None:
        task.status = "failed"
        task.error = message
        task.finishedAt = _now_ms()

    def submit(self, request: Any, *, assembly: dict[str, Any] | None = None,
               stage: str = "h3") -> str:
        """入队一次生成 ✓ ⇒ 返回 ``taskId`` ✓（**立即返回** ✗ 不等它跑完 ✓）。

        ⚠️ 装配是**任务自己的**事实 ✗ —— 不查"现在装了没" ✓：前面那个任务可能把装配换掉 ✓✗
        （同参数会复用 ✓，见 :meth:`ensure_loaded` ✓）。
        """
        task = EngineTask(id=f"eng-{uuid.uuid4().hex[:12]}", stage=stage, request=request,
                          assembly=dict(assembly or {}), createdAt=_now_ms())
        with self._wake:
            self._ensure_worker()
            self._tasks[task.id] = task
            self._order.append(task.id)
            self._queue.append(task.id)
            self._wake.notify_all()
        return task.id

    def get_task(self, task_id: str, *, events_limit: int = 200) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.to_dict(events_limit=events_limit) if task is not None else None

    def list_tasks(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(reversed(self._order))[:max(0, limit)]
            return [self._tasks[task_id].to_summary() for task_id in ids]

    def cancel(self, task_id: str) -> dict[str, Any] | None:
        """请求取消 ✓ —— **排队中的**立刻终结 ✓ / **跑着的**由管线在下一步停 ✗（不是瞬停 ✓）。"""
        with self._wake:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.cancelEvent.set()
            if not task.terminal and task.id in self._queue:
                self._queue.remove(task.id)
                task.status = "cancelled"
                task.finishedAt = _now_ms()
            self._wake.notify_all()
        return self.get_task(task_id, events_limit=0)

    def wait(self, task_id: str, *, timeout: float | None = None) -> dict[str, Any] | None:
        """等任务终结 ✓（``timeout`` 秒；到点还没完就返回**当前**状态 ✓ —— 不抛 ✗）。"""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._wake:
            while True:
                task = self._tasks.get(task_id)
                if task is None or task.terminal:
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._wake.wait(timeout=remaining)
        return self.get_task(task_id)

    # ── 装载 / 卸载 ✓ ────────────────────────────────────────────────────
    def ensure_loaded(self, *, stage: str = "h3", dit_path: str | None = None,
                      tokenizer_path: str | None = None, device: str | None = None,
                      audio_latent_mode: str | None = None, attach_video_vae: bool = True,
                      attach_audio_vae: bool = True, force: bool = False) -> dict[str, Any]:
        """把引擎**装齐** ✓（同名参数 ⇒ **复用**已装的 ✓）⇒ 返回装载报告 ✓。

        ⚠️ 有任务在跑时**不许换装配** ✗（那是把在跑的模型从底下抽走 ✓✗）⇒ :class:`EngineBusy` ✓。
        ⚠️ 装不上 ⇒ ``TorchBackendUnavailable``（``reason`` 见模块头 ✓），且**不半装** ✗。
        """
        key = _load_key(stage=stage, dit_path=dit_path, tokenizer_path=tokenizer_path,
                        device=device, audio_latent_mode=audio_latent_mode,
                        attach_video_vae=attach_video_vae, attach_audio_vae=attach_audio_vae)
        with self._lock:
            if not force and self._backend is not None and key == self._loadKeyTuple:
                return {**dict(self._loadReport or {}), "reused": True}
            if self._current is not None:
                raise EngineBusy(f"引擎正在跑任务 {self._current} ✗ ⇒ 此刻不换装配 ✓"
                                 f"（先等它结束，或卸载后再装 ✓）")
        return self._ensure_loaded_internal(stage=stage, dit_path=dit_path,
                                            tokenizer_path=tokenizer_path, device=device,
                                            audio_latent_mode=audio_latent_mode,
                                            attach_video_vae=attach_video_vae,
                                            attach_audio_vae=attach_audio_vae)

    def _ensure_loaded_internal(self, *, stage: str = "h3", dit_path: str | None = None,
                               tokenizer_path: str | None = None, device: str | None = None,
                               audio_latent_mode: str | None = None,
                               attach_video_vae: bool = True,
                               attach_audio_vae: bool = True) -> dict[str, Any]:
        """真正装配 ✓（**不带排队守卫** ✗ —— worker 自己持有当前任务，见 :meth:`_execute` ✓）。

        顺序**按依赖来** ✓（⚠️ 不是随便排的 ✗）：主 DiT 先装 ✓ ⇒ ``attach_text_encoder`` 才能
        用主干 ``text_dim`` **自动配** ``output_dim`` ✓；VAE 只做**互核** ✓（``load_weights`` 与已挂
        的 VAE 彼此校验 ✓ ⇒ 先挂后装 / 先装后挂都对 ✓）。
        ⚠️ 显存排班（谁先谁后、谁能在下一步前释放 ✓）是**另一件事** ✗ —— 见 ``loader.residency_plan`` ✓。
        """
        key = _load_key(stage=stage, dit_path=dit_path, tokenizer_path=tokenizer_path,
                        device=device, audio_latent_mode=audio_latent_mode,
                        attach_video_vae=attach_video_vae, attach_audio_vae=attach_audio_vae)
        with self._lock:
            if self._backend is not None and key == self._loadKeyTuple:
                return {**dict(self._loadReport or {}), "reused": True}

        backend = TorchBackend(device=device, audio_latent_mode=audio_latent_mode)
        report: dict[str, Any] = {"stage": stage, "reused": False, "components": {},
                                  "synthetic": True}
        try:
            report["weights"] = backend.load_weights(path=dit_path)
            report["components"]["dit"] = (report["weights"] or {}).get("path") or True
            # ⚠️ 参考 VAE 的尺寸**必须从已装主干现推** ✗（`attach_vae()` 不给配置 ⇒ 拿默认那套
            #    `channel_multipliers=(1,2,2)` ⇒ `spatial_scale=4` ✗ ≠ H3 的 16 ✓ ⇒ **当场被拒** ✓✗）
            trunk = (backend.describe().get("config") or {})
            if tokenizer_path:
                # ⚠️ 只有给了词表才挂真分词器 ✗ —— 不给就挂桩 ✓（桩的分词**没有语义** ✓，如实标 ✓）
                report["components"]["textEncoder"] = backend.attach_text_encoder(
                    tokenizer_path=tokenizer_path)
            else:
                report["components"]["textEncoder"] = backend.attach_text_encoder()
                report["textEncoderNote"] = "未给词表 ⇒ 挂了**桩**分词器 ✓（分词无语义 ✓）"
            if attach_video_vae:
                report["components"]["vae"] = _attach_reference_video_vae(backend, trunk)
            if attach_audio_vae:
                report["components"]["audioVae"] = _attach_reference_audio_vae(backend, trunk)
        except TorchBackendUnavailable as err:
            # ⚠️ **不半装** ✗：失败 ⇒ 丢掉整个候选后端 ✓（模块一个字不变 ✓）
            del backend
            _release_torch_cache()
            raise TorchBackendUnavailable(str(err), reason=getattr(err, "reason", "pending")) from err
        except BaseException as err:  # noqa: BLE001
            del backend
            _release_torch_cache()
            raise TorchBackendUnavailable(f"装配失败：{type(err).__name__}: {err}",
                                          reason="pending") from err

        described = backend.describe()
        report["device"] = described.get("device") or device
        report["backend"] = described.get("name")
        #: ⚠️ 与 `TorchBackend.write` 同一条诚实口径 ✓：文件是真的 ✓，画面来自未训练 VAE ✗
        report["note"] = ("DiT 主权重**是真的** ✓；参考 TE / VAE **未训练** ✗ ⇒ "
                          "产物是真文件 + 噪声画面 ✓（`synthetic` 保持 True ✓）")
        with self._lock:
            self._backend = backend
            self._loadKeyTuple = key
            self._loadReport = report
        return report

    def unload(self, *, force: bool = False) -> dict[str, Any]:
        """**丢掉张量 + 清缓存** ✓ —— 这就是"不依赖外部"的卸载 ✓（不通知任何人 ✗）。"""
        with self._lock:
            if (self._current is not None or self._queue) and not force:
                raise EngineBusy(f"引擎正忙（current={self._current}，queued={len(self._queue)}）✗ "
                                 f"⇒ 先等它们结束，或 force 强卸 ✓")
            had = self._backend is not None
            device = (self._loadReport or {}).get("device")
            backend = self._backend
            self._backend = None
            self._loadKeyTuple = None
            self._loadReport = None
        del backend                          # 引用一断 ⇒ 大张量才真被回收 ✓（只置 None 不算 ✓✗）
        cleared = _release_torch_cache()
        return {"unloaded": had, "device": device, "cacheCleared": cleared}

    # ── 自述 ✓ ───────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        """运行时的**全部事实** ✓（依赖 / 装了没 / 忙不忙 / 队列 / 最近任务 ✓）。"""
        available, reason = torch_available()
        with self._lock:
            current = self._current
            queued = len(self._queue)
            tasks = [self._tasks[task_id].to_summary() for task_id in list(reversed(self._order))[:10]]
            report = dict(self._loadReport or {}) or None
        # ⚠️ 没装也要能答"**将会**在哪跑" ✗——``cpu`` 与 ``cuda`` 差着两个数量级 ✓✗
        #    （cpu 上一条视频要按小时算 ✓）⇒ 前端必须一眼看见 ✓，否则会把"慢"当成"卡住了" ✓。
        #    探设备的代价 = 一次依赖探测 ✓（毫秒级 ✓）。
        probe: dict[str, Any] | None = None
        if self._backend is not None:
            described = self._backend.describe()
        else:
            described = None
            if available:
                try:
                    probe = TorchBackend().describe()
                except Exception:  # noqa: BLE001 —— 自述**不许**把 status 打挂 ✗
                    probe = None
        return {
            "name": "engine",
            "available": available, "reason": reason,
            "dependencies": dependency_status(),
            "device": (described or probe or {}).get("device") or (report or {}).get("device"),
            #: ⚠️ ``probe`` = **还没装、只是探到的** ✗（别当成"已经在这儿跑了" ✓✗）
            "deviceSource": ("loaded" if described else ("probe" if probe else None)),
            "loaded": self._loaded_flags(),
            "loadReport": report,
            "backend": described,
            "busy": current is not None or queued > 0,
            "current": current, "queued": queued,
            "tasks": tasks,
            "synthetic": True,
            "externalDependencies": [],
            "note": ("**不依赖任何外部服务** ✓（不用 ComfyUI / SD WebUI / Ollama ✓，不起子进程 ✓）；"
                     "⚠️ 参考 TE / VAE 未训练 ✗ ⇒ 画面是噪声 ✓"),
            "pendingParts": list(PENDING_PARTS),
        }

    def _loaded_flags(self) -> dict[str, bool]:
        """逐组件"挂上了没" ✓ —— 不写"支持"这类能力声明 ✗（那是 `backend.describe` 的事 ✓）。"""
        backend = self._backend
        if backend is None:
            return {"dit": False, "textEncoder": False, "vae": False, "audioVae": False}
        return {
            "dit": getattr(backend, "_model", None) is not None,
            "textEncoder": getattr(backend, "_textEncoder", None) is not None,
            "vae": getattr(backend, "_vae", None) is not None,
            "audioVae": getattr(backend, "_audioVae", None) is not None,
        }


#: 全局单例 ✓ —— 进程内**只能有一份**模型驻留 ✓✗（两份 = 双倍显存 ✓）。
engine_runtime = EngineRuntime()
