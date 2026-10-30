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
import hashlib
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import audio_vae as audio_vae_mod
from . import chain
from . import lora as lora_mod
from . import lora_cache
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


def _env_int(name: str, default: int, *, low: int = 1) -> int:
    """读整数环境变量；**读不懂就用默认** ✓（不抛 ✗ —— 为一个限额配置把引擎起不来，得不偿失 ✓✗）。"""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= low else default


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


def _lora_key(loras: Any) -> tuple[tuple[str, float], ...]:
    """LoRA 清单 → **可比较**的快照 ✓（``LoraSpec`` / dict / 路径字符串都认 ✓）。

    ⚠️ 必须进装载键 ✗✗：LoRA 是**模型的一部分** ✓ ⇒ 不进键就会出现「换了 LoRA 却复用上一套
    模型」，而且**一声不响** ✓✗（画面上只表现为"不是那个档" ✓）。排序 ⇒ 顺序无关 ✓
    （加法式增量可交换 ✓）；换文件 / 换强度 ⇒ 键必变 ✓。
    """
    items: list[tuple[str, float]] = []
    for item in (loras or ()):
        #: ⚠️ **写法与 ``strength`` 口径只在 `lora.coerce_spec` 一处** ✓✗ —— 这里以前自己解析 ✓，
        #:    而且**只认 ``strength``** ✓✗ ⇒ ``{"strength_model": 0.5}`` 被记成 **1.0** 而实际按
        #:    0.5 挂 ✓✗ ⇒ 「键说 1.0、实挂 0.5」✓✗，正好踩中文档里那条禁忌 ✓：先按 1.0 装过 ⇒
        #:    再来一条 0.5 的**复用旧模型、一声不响** ✓✗（"不是那个档"却查不出原因 ✓）
        #:    （2026-10-29 实测：``key = (('/a', 1.0),)`` 而 ``apply = 0.5`` ✓）。
        spec = lora_mod.coerce_spec(item)
        items.append((spec.path, float(spec.strength)))
    return tuple(sorted(items))


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


def _attach_video_vae(backend: TorchBackend, trunk: dict[str, Any], *,
                      real: bool = False, device: str | None = None,
                      weights: str | None = None) -> dict[str, Any]:
    """挂 **H3 口径**的视频 VAE ✓ ⇒ 返回装载报告 ✓ —— **参考族**（默认 ✓）/ **真权重族**（``real=True`` ✓）。

    ⚠️ 级数**从 H3 的 `vaeScale` 推** ✗（不是随手写个 5 ✓）：``spatial_scale = 2^(级数−1)`` ✓
    ⇒ 级数 = ``log2(vaeScale) + 1`` ✓。挂完**再核一遍**报告里的 ``spatialScale`` ✓ ——
    对不上就报 ✗（``attach_vae`` 自己也会拒 ✓，这里再钉一道 ✓ —— 一条事实两个来源 ✓）。

    ⚠️ 真权重那条路**不猜文件** ✗：``weights`` 不给 ⇒ 走
    :func:`vae_h3.h3_video_vae_weights_on_disk` 的**发现** ✓；发现不到 ⇒ **当场拒** ✗ 并把候选摊开 ✓
    （静默回落成参考族会让"以为在出真画面"这件事**永远查不出来** ✗✗）。
    ⚠️ 真 VAE 默认落 **cpu** ✓（``device`` 可改 ✓）：它 5.2 GiB 权重与 DiT 的 19.62 GiB int8 常驻
    抢同一块 22.49 GiB 卡 ✓✗ ⇒ 想上卡得先让 DiT 下卡 ✓（``loader.residency_plan`` ✓：两族**不重叠** ✓）。
    """
    scale = int(vae_mod.H3_VIDEO_VAE_FACTS["vaeScale"])
    if real:
        from . import vae_h3 as vae_h3_mod  # noqa: PLC0415 —— 只有真族那条路要它 ✓
        target = str(weights or "").strip()
        if not target:
            found = vae_h3_mod.h3_video_vae_weights_on_disk()
            target = str((found or {}).get("path") or "").strip()
            if not target:
                candidates = (found or {}).get("candidates") or []
                raise TorchBackendUnavailable(
                    "要真 H3 视频 VAE，但盘上**没发现**权重 ✗"
                    f"（扫过 {len(candidates)} 处候选 ✓）。\n"
                    f"候选：{list(candidates)[:8] or '（一处都没有 ✓ —— 搜索根都没找到）'}\n"
                    "⚠️ **不回落**成参考 VAE ✗：那样出来的仍是噪声 ✓ 而调用方会以为在出真画面 ✗✗。",
                    reason="pending")
        channels = (trunk or {}).get("latents_dim")
        wanted = vae_h3_mod.H3_VIDEO_VAE_ARCH_FACTS.get("latentChannels")
        if isinstance(channels, int) and channels > 0 and isinstance(wanted, int) \
                and channels != wanted:
            raise TorchBackendUnavailable(
                f"主干 latent 通道数 {channels} ✗ ≠ H3 真 VAE 的 {wanted} ✓ ⇒ **拒装** ✓",
                reason="pending")
        report = backend.attach_vae(vae_h3_mod.H3VideoVAEConfig(
            weights=target, device=str(device or "cpu"), strict=True))
        if not report.get("complete"):
            raise TorchBackendUnavailable(
                f"真 H3 视频 VAE 装不齐 ✗（缺 {len(report.get('missing') or [])} 个键 ✓）"
                f"⇒ **拒用** ✓：{target}", reason="pending")
    else:
        channels = _trunk_dim(trunk, "latents_dim")
        levels = int(math.log2(scale)) + 1
        report = backend.attach_vae(vae_mod.VideoVAEConfig(
            base_channels=channels, latent_channels=channels,
            channel_multipliers=tuple(1 for _ in range(levels))))
    if int(report.get("spatialScale") or 0) != scale:
        raise TorchBackendUnavailable(
            f"视频 VAE（{report.get('kind')}）的 spatial_scale={report.get('spatialScale')} ✗ "
            f"≠ H3 的 {scale} ✓ "
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


def _vram_facts() -> dict[str, Any] | None:
    """显存台账（GiB ✓）—— **没有 torch / 没有卡 ⇒ ``None``** ✗（不编数字 ✗）。"""
    try:
        import torch  # noqa: PLC0415 —— 可选依赖 ✓
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "allocatedGiB": round(torch.cuda.memory_allocated() / 2**30, 2),
            "reservedGiB": round(torch.cuda.memory_reserved() / 2**30, 2),
            "freeGiB": round(free_bytes / 2**30, 2),
            "totalGiB": round(total_bytes / 2**30, 2),
        }
    except Exception:  # noqa: BLE001 —— 台账拿不到**不许**把主流程搞崩 ✗
        return None


def _reclaim_cuda_cache() -> dict[str, Any]:
    """把**保留着但没在用**的显存块还给驱动 ✓ ⇒ 返回前后事实 ✓。

    ⚠️ 只还"空着的" ✗：**在用的张量（权重 / 本步激活）一个不动** ✓（见自检的不变量 ✓）。

    为什么非做不可 ✗（2026-09-26 实测 ✓，RTX A5000 22.49 GiB ✓，
    本机 User 环境预设 ``PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024`` ✓）：

    * 装完 SDXL 四件套：**真占用 6.73 GiB** ✓，而缓存**保留 13.88 GiB** ✗ ⇒ 驱动视角只剩 **7.37 GiB** ✗；
    * 再采样一张 1024²：保留涨到 **15.76 GiB** ✗ ⇒ 驱动视角只剩 **5.47 GiB** ✗✗；
    * 还一次：保留 **15.76 → 8.24 GiB** ✓、驱动视角 **5.47 → 12.98 GiB** ✓（**还回 7.52 GiB** ✓）
      —— 而在用的 6.73 GiB **一点没少** ✓、同条件采样耗时**不变** ✓。

    ⇒ 不还的后果是「**缓存看起来占了整卡**」✗：与同机 ComfyUI 抢卡时先撞的就是它 ✓✗，
      也撞「不许逼近显存上限」的口径 ✗。还回去的代价 = 下次几块 ``cudaMalloc`` ✓（毫秒级 ✓）。
    """
    before = _vram_facts()
    if before is None:
        # 没有 torch / 没有卡 ⇒ 连 ``gc.collect`` 都不必跑 ✓（如实说"没得还" ✓）
        return {"released": False, "reclaimedGiB": 0.0,
                "note": "没有 torch / 没有 CUDA ⇒ 无缓存可还 ✓"}
    _release_torch_cache()
    after = _vram_facts() or {}
    return {
        "released": True,
        "reclaimedGiB": round(float(before["reservedGiB"]) - float(after.get("reservedGiB", 0.0)), 2),
        "reservedGiB": after.get("reservedGiB"),
        "allocatedGiB": after.get("allocatedGiB"),
        "freeGiB": after.get("freeGiB"),
        "totalGiB": after.get("totalGiB"),
    }


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
        #: 装配量计 ✓（E3.4 ✓，2026-10-29 ✓）—— **进程内**的复用/真装/丢弃次数 ✓
        #: （⚠️ 只有计数 + 最近几条事件 ✗ 不留全量历史：长跑进程会把它变成内存泄漏 ✓✗）
        self._cacheHits = 0
        self._cacheMisses = 0
        self._cacheEvictions = 0
        self._cacheEvents: list[dict[str, Any]] = []
        self._cacheSince = _now_ms()
        self._tasks: dict[str, EngineTask] = {}
        #: 提交顺序 ✓（列表按它倒序 ✓ —— 不按时间戳排 ✗：同毫秒提交会乱序 ✓✗）
        self._order: list[str] = []
        self._queue: list[str] = []
        self._current: str | None = None
        self._worker: threading.Thread | None = None
        #: ⚠️ 队列**必须有上限** ✗✗：`submit` 立即返回 ✓、worker 单线程 ✓ ⇒ 前端连点 30 次就会排 30 个 ✓；
        #: 每个 task 都留一份 `request`（带参考图时是**十 MB 级** ✓），而单条视频要分钟级 ✓ ——
        #: 排到第 30 个时用户早不知道自己在等什么 ✓✗，内存还一直涨 ✗。「排不下」要**当场说** ✓，
        #: 不能让请求**悄悄堆**在列表里 ✓✗。
        #: 默认 32：单机单卡按分钟计的生产任务，32 已远超「今天要出的量」✓；环境变量可覆盖 ✓。
        self._maxQueue = _env_int("DARME_ENGINE_MAX_QUEUE", 32, low=1)
        #: 终结任务的**历史**上限 ✓：不裁 ⇒ `_tasks` 只增不减 ✓✗（长跑进程连查询都变钝 ✓）。
        self._maxTasks = _env_int("DARME_ENGINE_MAX_TASKS", 200, low=self._maxQueue)

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
            # ⚠️ 先给个兜底 ✗：`_execute` 自己炸了也得有东西可封口 ✓（否则 finally 里取不到它 ✓✗）
            outcome: tuple[str, str | None] = ("failed", "任务没给出结果 ✗（内部矛盾 ✓）")
            try:
                outcome = self._execute(task)
            except BaseException as err:  # noqa: BLE001 —— worker **绝不允许**死掉 ✗✗
                #  线程一死，后面所有任务永久 queued ✓✗ —— 那是最难查的静默故障 ✓
                outcome = ("failed", f"{type(err).__name__}: {err}")
            finally:
                # ⚠️ 一张跑完就把**空着的**缓存还回去 ✗（采样同样会把碎片顶起来 ✓✗：实测一张 1024²
                #    之后 reserved 15.76 GiB ⇒ 驱动视角只剩 5.47 GiB ✓✗ ⇒ 还完 12.98 GiB ✓）。
                # ⚠️ 顺序要紧 ✗✗：**先还 → 记事件 → 最后封口状态 → 再唤醒** ✓。
                #    只写"先还、后唤醒"**不够** ✗✗（2026-09-26 被实测证伪 ✓）：它只保证了
                #    **被唤醒**的读者看得见 ✓ —— 而读者不止是被唤醒那种 ✗（`bridge.run_job` 是
                #    **带超时轮询**的 ✓）⇒ 终结态一可见它当场收工走人 ✓ ⇒ 此刻事件里若还没有
                #    "还了多少"，那条事实就**永远**漏掉 ✓✗。实测：同进程连跑两张，
                #    **第 1 张的 `reclaimEvent` 为空、第 2 张有** ✓✗ —— 窗口就在这里 ✓。
                reclaim = _reclaim_cuda_cache()
                status, error = outcome
                with self._wake:
                    if reclaim.get("released"):
                        task.events.append({"kind": "reclaim", **reclaim})
                    self._settle(task, status, error)
                    self._current = None
                    self._wake.notify_all()

    def _execute(self, task: EngineTask) -> tuple[str, str | None]:
        """装载（除非干跑 ✓）⇒ 跑管线 ⇒ **返回"该怎么封口"**（``status`` / ``error``）✓ 不抛 ✗。

        ⚠️ **自己不写终结态** ✗✗：封口是 :meth:`_settle` 的事 ✓，而且必须排在
        "这一张还回多少显存"记完之后 ✓（理由见 :meth:`_worker_loop` 的 ``finally`` ✓）。
        """
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
                return ("failed", f"装载失败：{err}")
            backend = self._backend
            if backend is None:             # 报告成功却没后端 = 内部矛盾 ⇒ 响亮报 ✗
                return ("failed", "装载报告成功但运行时没有后端 ✗（内部状态不一致 ✓）")
        if task.cancelEvent.is_set():
            return ("cancelled", None)
        task.status = "running"
        # ⭐ 多段成片 ✓（2026-09-26 接 ✓）：超过单段原生上限的请求**不再**一路上调帧数直到显存爆掉 ✗
        #    —— 而是拆成几段**顺序**生成（每段带上一段保留尾帧当首帧 ✓）再拼成一条 ✓
        #    （分段计划见 app.services.engine.chain ✓、每段输入见 app.services.engine.segments ✓）。
        #    ⚠️ 单段请求**原路一字不改** ✓：`plan_chain` 回 None ⇒ 那个分支与从前完全一样 ✓。
        chain_plan = chain.plan_chain(task.request)
        if chain_plan is None:
            result = pipe.run_sync(task.request, backend, on_event=task.events.append,
                                  cancel=task.cancelEvent.is_set)
        else:
            # ⚠️ 拆段这件事**必须让用户看见** ✗（悄悄替人拆 = 他以为一次就出 ✓✗）
            task.events.append({
                "kind": "stage", "stage": "chain-plan",
                "note": f"请求 {chain_plan.seconds:.2f}s 超过单段上限 "
                        f"（{chain_plan.maxFrames} 帧 ≈ "
                        f"{chain_plan.maxFrames / max(1, chain_plan.fps):.2f}s ✓）"
                        f"⇒ 拆成 {chain_plan.segmentCount} 段顺序生成、再拼成一条 ✓",
                "chainPlan": chain_plan.to_dict(),
            })
            result = chain.run_chain(task.request, backend, run_segment=pipe.run_sync,
                                     on_event=task.events.append,
                                     cancel=task.cancelEvent.is_set)
        task.result = result.to_dict()
        if result.cancelled:
            return ("cancelled", None)
        if result.ok:
            return ("done", None)
        return ("failed", (result.error or {}).get("message") or "生成失败 ✓（看 result.error ✓）")

    def _settle(self, task: EngineTask, status: str, error: str | None = None) -> None:
        """**封口**：写终结态 ✓ —— ⚠️ **必须持 ``self._wake`` 调用** ✗✗（它与唤醒同属一段临界区 ✓）。

        ⚠️ 为什么不许在 :meth:`_execute` 里随手写 ✗：``terminal`` 一旦为真，读者就可以走了 ✓
        ⇒ 那些"跑完才知道"的事实（本张还回多少显存 ✓）会被整个漏掉 ✓✗
        （2026-09-26 实测的窗口 ✓，见 :meth:`_worker_loop` ✓）。
        """
        task.status = status
        task.error = error
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
            #: ⚠️ 拒绝必须发生在**建 task 之前/之前不入表** ✗：先入表再判，被拒的请求仍留一条记录 ✓✗
            #: ⇒ 连点 100 次照样把 `_tasks` 撑到 100 条 ⇒ 背压白做 ✓✗（内存才是要防的那一头 ✓）。
            if len(self._queue) >= self._maxQueue:
                raise EngineBusy(
                    f"引擎队列已满（排队 {len(self._queue)} / 上限 {self._maxQueue}）⇒ 这次请求"
                    f"**没有入队、也没有开始跑**；等前一批跑完再提交（或先 cancel 掉不要的）")
            self._ensure_worker()
            self._tasks[task.id] = task
            self._order.append(task.id)
            self._queue.append(task.id)
            self._prune_tasks()
            self._wake.notify_all()
        return task.id

    def _prune_tasks(self) -> None:
        """裁掉**最老的已终结**任务（在跑/在排/正在被读的一个都不动 ✓）。调用方必须持锁 ✓。

        ⚠️ 只裁 ``terminal`` 的 ✗：把还在排队的裁掉，`cancel`/`wait` 就会找不到任务 ✓✗（表现为
        「取消返回 None」—— 比不裁更难查 ✓）。
        """
        overflow = len(self._order) - self._maxTasks
        if overflow <= 0:
            return
        keep: list[str] = []
        dropped = 0
        for task_id in self._order:          # 老 → 新
            task = self._tasks.get(task_id)
            evictable = (dropped < overflow and task is not None and bool(getattr(task, "terminal", False))
                         and task_id not in self._queue and task_id != self._current)
            if evictable:
                self._tasks.pop(task_id, None)
                dropped += 1
                continue
            keep.append(task_id)
        self._order = keep

    def get_task(self, task_id: str, *, events_limit: int = 200) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.to_dict(events_limit=events_limit) if task is not None else None

    def list_tasks(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(reversed(self._order))[:max(0, limit)]
            return [self._tasks[task_id].to_summary() for task_id in ids]

    def cancel(self, task_id: str) -> dict[str, Any] | None:
        """请求取消 ✓ ⇒ **快照 + 裁决**（``cancel.verdict`` ✓）。**排队中的**立刻终结 ✓ /
        **跑着的**由管线在下一步停 ✗（不是瞬停 ✓）。没有这个任务 ⇒ ``None`` ✓。

        ⚠️ 裁决**必须在这里定** ✗✗（2026-10-29 补 ✓，E3.2 缺口 ✓）：以前只回快照 ✓，调用方自己
        比 ``status`` —— 那是**另一瞬间**的事实 ✓✗。这中间 worker 完全可能刚好收尾 ✓，于是
        「点了取消 → 查到 ``done`` → 以为没取消成功」，而真相是「取消来得及、只是晚一步」✓✗；
        反过来也有：看到 ``cancelled`` 就以为是自己停的，其实是上一次取消早就在读之前落了账 ✓。
        三值只回答一件事 —— **这一个取消请求算不算数** ✓：

        * ``accepted`` ✓ —— 还在排队 ⇒ **已当场终结**（``stopped: true`` ✓，保证停了 ✓）。
          ⚠️ 这个字读作「已受理**且已生效**」✗ 不是「只收到」✓ —— 只收到、停不停还得看管线的
          那种叫 ``too_late`` ✓（同一个词两种读法 = 下一个坑 ✓✗）。
        * ``too_late`` ✓ —— 已经在跑 ⇒ 旗标已置、管线在**下一步**停 ✗（``stopped: false`` ✓，
          **不保证**停：这一步可能正把产物写完 ✓）。⚠️ 这不是错误 ✗、不许翻成失败报错 ✓✗：
          用户点的那一刻请求完全合法 ✓，报错会让人以为「重试就好」✓✗。
        * ``completed`` ✓ —— 早已终结（``done`` / ``failed`` / 已经 ``cancelled``）⇒ 取消无从生效 ✓
          （``stopped: false`` ✓ —— 第二次点取消**不许**谎称又停了一次 ✓✗）。

        裁决与快照**同一瞬间** ✓：两者都在同一把锁里取 ✓，所以 ``accepted`` 的 ``status`` 必是
        ``cancelled`` ✓、``too_late`` 的必**不是** ``cancelled`` ✓（判据见 `engine_runtime_test` ✓）。
        """
        with self._wake:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.terminal:
                verdict, stopped = "completed", False
                reason = (f"任务早已终结（``status={task.status}`` ✓）⇒ 取消无从生效 ✓"
                          "（⚠️ 不是「取消失败」✗：本来就没有可停的东西 ✓）")
            elif task.id in self._queue:
                self._queue.remove(task.id)
                task.status = "cancelled"
                task.finishedAt = _now_ms()
                task.cancelEvent.set()
                verdict, stopped = "accepted", True
                reason = "还在排队 ✓ ⇒ 已当场终结（一步都没跑，产物一条都没出 ✓）"
            else:
                task.cancelEvent.set()
                verdict, stopped = "too_late", False
                reason = ("已经在跑 ✓ ⇒ 只置了取消旗标，管线在**下一步**停 ✗；"
                          "这一步可能已经把产物写完了 ⇒ **不保证**停下 ✓")
            self._wake.notify_all()
        snapshot = self.get_task(task_id, events_limit=0) or {}
        return {**snapshot,
                "cancel": {"verdict": verdict, "stopped": stopped, "reason": reason}}

    def wait(self, task_id: str, *, timeout: float | None = None) -> dict[str, Any] | None:
        """等任务终结 ✓（``timeout`` 秒；到点还没完就返回**当前**状态 ✓ —— 不抛 ✗）。

        ⚠️ 一旦这里报 ``terminal`` ✓，**这一张的收尾事实就已经在事件里了** ✓（"还回多少显存" ✓）
        —— 封口与那条 append 在同一段临界区 ✓，见 :meth:`_settle` ✓：
        轮询式读者（``bridge.run_job`` ✓）一到终结就走人 ✗，顺序错了那条事实就永远漏掉 ✓✗。
        """
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
                      attach_audio_vae: bool = True, loras: Any = None,
                      real_video_vae: bool = False, video_vae_device: str | None = None,
                      video_vae_weights: str | None = None,
                      int8_offload: bool = False,
                      force: bool = False) -> dict[str, Any]:
        """把引擎**装齐** ✓（同名参数 ⇒ **复用**已装的 ✓）⇒ 返回装载报告 ✓。

        ⚠️ 有任务在跑时**不许换装配** ✗（那是把在跑的模型从底下抽走 ✓✗）⇒ :class:`EngineBusy` ✓。
        ⚠️ 装不上 ⇒ ``TorchBackendUnavailable``（``reason`` 见模块头 ✓），且**不半装** ✗。
        ⚠️ ``loras`` **必须进装载键** ✗✗：它是**模型的一部分** ✓ —— 不进键就会出现
        「换了 turbo LoRA 却复用上一套模型」✓✗，而且**一声不响** ✓✗（出图只是"不是那个档"）。

        ``real_video_vae``（2026-10-29 新增 ✓，**默认关** ✗）：开 ⇒ 挂 **H3 真视频 VAE** ✓
        （``video_vae_weights`` 不给就按清单**发现** ✓；发现不到**当场拒** ✓ 不回落 ✗）。
        三个新参数**都进装载键** ✓ —— 换 VAE 就是换模型的一部分 ✓✗。

        ``int8_offload``（2026-10-29 新增 ✓，**默认关** ✗）：开 ⇒ int8 常驻权重**钉在 CPU**
        （锁页 ✓）、前向逐层搬上卡 ✓ —— 720×1280 那种长度**只有这一档装得下** ✓✗（见
        `h3_form.place_module` 的实测 ✓）。⚠️ 它也**必须进装载键** ✗✗：同一份权重、
        同一张卡，「int8 在卡上」与「int8 在 CPU」显存差 **17.94 GiB** ✓✗ ⇒ 不进键就会复用错的那个 ✓✗。
        """
        key = _load_key(stage=stage, dit_path=dit_path, tokenizer_path=tokenizer_path,
                        device=device, audio_latent_mode=audio_latent_mode,
                        attach_video_vae=attach_video_vae, attach_audio_vae=attach_audio_vae,
                        real_video_vae=real_video_vae, video_vae_device=video_vae_device,
                        video_vae_weights=video_vae_weights,
                        int8_offload=int8_offload,
                        loras=_lora_key(loras))
        with self._lock:
            same = self._backend is not None and key == self._loadKeyTuple
            if same and not force:
                self._note_cache("hit", key=key, reason="装配键相同 ⇒ 复用已装的 ✓（一次都不用花 ✓）")
                return {**dict(self._loadReport or {}), "reused": True}
            if self._current is not None:
                # ⚠️ **不记账** ✗：忙到拒装 ⇒ 什么都没发生 ✓（记成 miss 会把"排队等"和"真装一遍"混一起 ✓✗）
                raise EngineBusy(f"引擎正在跑任务 {self._current} ✗ ⇒ 此刻不换装配 ✓"
                                 f"（先等它结束，或卸载后再装 ✓）")
            if same:
                # ⚠️ ``force=True`` 但装配键**完全相同** ⇒ :meth:`_ensure_loaded_internal` 一进门就
                #    **当场复用**（那里也有一道同键短路 ✓）⇒ 实际发生的是「复用一次」✓✗。
                #    这里若按 ``miss`` + ``eviction`` 记 ✗✗ 就是**两条假账**：既没真装、旧的也没丢 ✓✗
                #    —— 量与事实不一致比"没有量"更坏 ✓（调参时会去追一个不存在的重装 ✓✗）。
                self._note_cache("hit", key=key,
                                 reason=("force=True 但装配键相同 ⇒ 内部**当场复用**已装的 ✓"
                                         "（没有真装 ✗ —— force 对同键不生效 ✓✗，要重装请改装配键或先 unload ✓）"))
                return {**dict(self._loadReport or {}), "reused": True}
            self._note_cache("miss", key=key, reason="装配键不同 / 还没装 ⇒ 真装一次 ✓")
            if self._backend is not None:
                # ⚠️ 这条要记在**动手之前** ✗✗：`_ensure_loaded_internal` 会**先把旧的放掉**再装 ✓
                #    （两套大模型同时在内存里 = 峰值翻倍 = OOM ✓）⇒ 哪怕随后装失败，旧的也确实没了 ✓
                #    —— 记在成功之后就会漏掉"装失败、旧的也丢了"这种最该看见的情形 ✓✗。
                self._note_cache("eviction", key=self._loadKeyTuple,
                                 reason="换装配 ⇒ 先把旧的放掉 ✓（两套同时在内存里会 OOM ✓✗）")
        return self._ensure_loaded_internal(stage=stage, dit_path=dit_path,
                                            tokenizer_path=tokenizer_path, device=device,
                                            audio_latent_mode=audio_latent_mode,
                                            attach_video_vae=attach_video_vae,
                                            attach_audio_vae=attach_audio_vae,
                                            real_video_vae=real_video_vae,
                                            video_vae_device=video_vae_device,
                                            video_vae_weights=video_vae_weights,
                                            int8_offload=int8_offload,
                                            loras=loras)

    def _ensure_loaded_internal(self, *, stage: str = "h3", dit_path: str | None = None,
                               tokenizer_path: str | None = None, device: str | None = None,
                               audio_latent_mode: str | None = None,
                               attach_video_vae: bool = True,
                               attach_audio_vae: bool = True,
                               loras: Any = None,
                               real_video_vae: bool = False,
                               video_vae_device: str | None = None,
                               video_vae_weights: str | None = None,
                               int8_offload: bool = False) -> dict[str, Any]:
        """真正装配 ✓（**不带排队守卫** ✗ —— worker 自己持有当前任务，见 :meth:`_execute` ✓）。

        顺序**按依赖来** ✓（⚠️ 不是随便排的 ✗）：主 DiT 先装 ✓ ⇒ ``attach_text_encoder`` 才能
        用主干 ``text_dim`` **自动配** ``output_dim`` ✓；VAE 只做**互核** ✓（``load_weights`` 与已挂
        的 VAE 彼此校验 ✓ ⇒ 先挂后装 / 先装后挂都对 ✓）。
        ⚠️ 显存排班（谁先谁后、谁能在下一步前释放 ✓）是**另一件事** ✗ —— 见 ``loader.residency_plan`` ✓。
        ⚠️ 但"装完把**空着的**缓存还给驱动"是**这一件** ✓（装之前先 ``_release`` ✓ ⇒ 装完再
        :func:`_reclaim_cuda_cache` ✓ —— 一头一尾都还 ✓，见报告里的 ``cache`` 事实 ✓）。
        """
        key = _load_key(stage=stage, dit_path=dit_path, tokenizer_path=tokenizer_path,
                        device=device, audio_latent_mode=audio_latent_mode,
                        attach_video_vae=attach_video_vae, attach_audio_vae=attach_audio_vae,
                        real_video_vae=real_video_vae, video_vae_device=video_vae_device,
                        video_vae_weights=video_vae_weights,
                        int8_offload=int8_offload,
                        loras=_lora_key(loras))
        with self._lock:
            if self._backend is not None and key == self._loadKeyTuple:
                return {**dict(self._loadReport or {}), "reused": True}

        if loras and str(stage or "").strip().lower() == pipe.IMAGE_STAGE:
            # ⚠️ 出图那套（SDXL）还没接 LoRA ✗ ⇒ **当场报** ✗，不许静默丢掉这层增量 ✓✗
            raise TorchBackendUnavailable(
                f"出图（stage={pipe.IMAGE_STAGE}）还不支持 LoRA ✗（收到 {len(list(loras))} 条 ✓）；"
                f"视频侧已支持 ✓ —— 要么去掉 loras，要么等出图侧接上 ✓。", reason="pending")

        if str(stage or "").strip().lower() == pipe.IMAGE_STAGE:
            # ⚠️ 出图是**另一套组件** ✗（SDXL 四件套 ✓ ≠ H3 DiT + 参考 TE/VAE ✓）⇒ 分开装配 ✓。
            return self._ensure_image_loaded(key=key, dit_path=dit_path,
                                             tokenizer_path=tokenizer_path, device=device)

        # ⚠️ 装新的之前**先把旧的放掉** ✗✗（与 :meth:`_ensure_image_loaded` 同一条规矩 ✓）：
        #    H3 主干 int8 是 **17.94 GiB** ✓（offload 时全在 CPU 锁页 ✓）、全量 19.53 GiB ✓，
        #    而装配期还要**临时建一份 fp32 主干**（78 GiB ✓，见 :func:`_prepare_h3_module` ✓）✗
        #    ⇒ 旧的活着就装新的 = 同时压两份大模型 ✓✗：96 GB 的机器也炸 ✓
        #    （⚠️ 老路更硬：17.94 GiB 还在显存里 ⇒ 22.49 GiB 的卡**必 OOM** ✗）。
        #    丢旧的可恢复（重装一次 ✓）；⚠️ 代价是"装失败时旧的也没了" ✓ —— 刻意取舍 ✓，
        #    与出图那条路的口径一致 ✓（不静默降级 ✗）。
        with self._lock:
            previous = self._backend
            had_previous = previous is not None
            self._backend = None
            self._loadKeyTuple = None
            self._loadReport = None
        del previous                         # 引用一断 ⇒ 大张量才真被回收 ✓（只置 None 不算 ✓✗）
        _release_torch_cache()

        backend = TorchBackend(device=device, audio_latent_mode=audio_latent_mode,
                               int8_offload=int8_offload)
        report: dict[str, Any] = {"stage": stage, "reused": False, "components": {},
                                  "replaced": had_previous, "synthetic": True}
        try:
            report["weights"] = backend.load_weights(path=dit_path, loras=loras)
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
                report["components"]["vae"] = _attach_video_vae(
                    backend, trunk, real=bool(real_video_vae),
                    device=video_vae_device, weights=video_vae_weights)
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
        #: ⚠️ 与 `TorchBackend.write` 同一条诚实口径 ✓：文件是真的 ✓，画面来自哪一族要说清 ✓
        vae_kind = str(((report.get("components") or {}).get("vae") or {}).get("kind") or "")
        report["vaeKind"] = vae_kind or None
        if vae_kind == "h3-real":
            report["note"] = ("DiT 主权重**是真的** ✓；视频 VAE = **H3 真权重** ✓；"
                              "但 TE / 音频 VAE **未训练** ✗ ⇒ 解码链路是真的 ✓、潜变量来源仍是噪声 ✗ "
                              "⇒ `synthetic` 保持 True ✓（区别：换成真 TE 就能直接出片 ✓，"
                              "不必再换 VAE ✓）")
        else:
            report["note"] = ("DiT 主权重**是真的** ✓；参考 TE / VAE **未训练** ✗ ⇒ "
                              "产物是真文件 + 噪声画面 ✓（`synthetic` 保持 True ✓）")
        #: ⚠️ 装完**还一次缓存** ✗（G 级权重是一件件塞进去的 ⇒ 缓存被碎片顶起来 ✓✗，见
        #: :func:`_reclaim_cuda_cache` 的实测 ✓）—— 只还空着的块 ✓，在用的张量一个不动 ✓。
        report["cache"] = _reclaim_cuda_cache()
        with self._lock:
            self._backend = backend
            self._loadKeyTuple = key
            self._loadReport = report
        return report

    def _ensure_image_loaded(self, *, key: tuple[Any, ...], dit_path: str | None,
                             tokenizer_path: str | None,
                             device: str | None) -> dict[str, Any]:
        """装 **SDXL 出图后端** ✓（``stage="sdxl"`` ✓ ⇒ :class:`~.sdxl_backend.SdxlBackend` ✓）。

        ⚠️ 与视频那条路**不是同一套组件** ✗✗：那边是 H3 DiT + 参考 TE/VAE ✓；
        这边是 SDXL 的 **UNet + VAE + 双文本塔**四件套 ✓ —— 四件都在**同一份**主权重里 ✓，
        由 :func:`~.sdxl_backend.load_sdxl_components` 按前缀切开 ✓（实测四个前缀**恰好完整划分** ✓）。
        ⚠️ 四件**缺一不可** ✗：任一件不齐 ⇒ ``load_weights`` 当场报错 ✓（**不半装** ✗ ——
        半装出来的图「有形状、有颜色、但语义全错」✓✗，本仓最忌这类 ✓）。
        ⚠️ 词表**必须给** ✗：CLIP 词表是**权重的一部分** ✓，引擎不内置 ✗
        （不给 ⇒ 报错 ✓，**不挂桩词表** ✗ —— 桩分词没有语义，出图会「看着像、完全不听话」✓✗；
        这一点与视频那条路**刻意不同** ✓：那边没词表会挂桩且如实标注 ✓）。
        ⚠️ 装完同样**还一次缓存** ✓（见报告里的 ``cache`` 事实 ✓ ⇒ 驱动视角不逼近上限 ✓）。
        """
        from .sdxl_backend import SdxlBackend  # 局部 import ✓：torch 可选 ✓（同视频那条路 ✓）
        from .tokenizer_bpe import load_tokenizer

        if not dit_path:
            raise TorchBackendUnavailable(
                "自研引擎出图缺 **SDXL 主权重** ✗：装配里没给 ``ditPath`` ✓。\n"
                "  · 怎么修：在出图服务的 settings 里配 ``sdxlPath`` ✓，或按清单把权重放进 "
                "``models_dir`` ✓\n"
                "  · ⚠️ 这里**不猜路径** ✗：猜错会「装得进去、跑得出来、结果不对」✓✗",
                reason="pending")
        if not tokenizer_path:
            raise TorchBackendUnavailable(
                "自研引擎出图缺 **CLIP 词表** ✗：装配里没给 ``tokenizerPath`` ✓。\n"
                "  · 词表是**权重的一部分** ✓（引擎不内置 ✗）；请指向含 ``tokenizer.json`` ✓ "
                "或 ``vocab.json``+``merges.txt`` ✓ 的目录",
                reason="pending")

        # ⚠️ 装新的之前**先把旧的放掉** ✗✗：SDXL 四件套与 H3 DiT 都是 GB 级 ✓
        #    两套同时在内存里 ⇒ 峰值翻倍 ⇒ OOM ✓（丢旧的可恢复：重装一次即可 ✓；
        #    ⚠️ 代价是**装失败时旧的也没了** ✓ —— 这是刻意的取舍 ✓，不静默降级 ✓）。
        with self._lock:
            previous = self._backend
            self._backend = None
        del previous
        _release_torch_cache()

        backend = SdxlBackend(device=device)
        report: dict[str, Any] = {"stage": pipe.IMAGE_STAGE, "reused": False,
                                  "components": {}, "synthetic": True}
        try:
            report["weights"] = backend.load_weights(
                path=dit_path, tokenizer=load_tokenizer(tokenizer_path))
            # 报告里只留**事实** ✓（四件各自的张量数 ✓ —— 与事实表逐个可比 ✓）
            for name, component in (report["weights"].get("components") or {}).items():
                report["components"][name] = dict(component).get("keys") or True
        except BaseException as err:  # noqa: BLE001
            del backend
            _release_torch_cache()
            raise TorchBackendUnavailable(
                f"SDXL 装配失败：{type(err).__name__}: {err}", reason="pending") from err

        described = backend.describe()
        report["device"] = described.get("device") or device or "cpu"
        report["backend"] = described.get("name")
        report["synthetic"] = bool(described.get("synthetic"))
        report["facts"] = described.get("facts")
        #: ⚠️ 与 `SdxlBackend.synthetic` 同一条诚实口径 ✓：**真权重**装齐 ⇒ False ✓
        report["note"] = ("SDXL 四件套（UNet / VAE / CLIP-L / CLIP-G）**全是真权重** ✓ ⇒ "
                          "`synthetic` 如实为 **False** ✓"
                          "（与视频那条路的参考 TE/VAE 未训练 ⇒ 恒 True 不同 ✓）")
        #: ⚠️ 这条**尤其**要还 ✗：SDXL 是 1680+248+197+390 件小张量 ✗ ⇒ 装机就把缓存顶到
        #: **13.88 GiB（真占用只有 6.73 GiB）** ✓✗ ⇒ 不还就等于**白白占掉半张卡** ✗。
        report["cache"] = _reclaim_cuda_cache()
        with self._lock:
            self._backend = backend
            self._loadKeyTuple = key
            self._loadReport = report
        return report

    def unload(self, *, force: bool = False) -> dict[str, Any]:
        """**丢掉张量 + 清缓存** ✓ —— 这就是"不依赖外部"的卸载 ✓（不通知任何人 ✗）。

        ⚠️ ``cacheCleared`` 只说"清缓存这步成没成" ✓；**到底还回多少**看 ``reclaimedGiB`` /
        ``reservedGiB`` / ``freeGiB`` ✓（与装载报告里的 ``cache`` **同一套事实** ✓ 不另立口径 ✗）。
        """
        with self._lock:
            if (self._current is not None or self._queue) and not force:
                raise EngineBusy(f"引擎正忙（current={self._current}，queued={len(self._queue)}）✗ "
                                 f"⇒ 先等它们结束，或 force 强卸 ✓")
            had = self._backend is not None
            device = (self._loadReport or {}).get("device")
            backend = self._backend
            old_key = self._loadKeyTuple
            if had:
                self._note_cache("eviction", key=old_key,
                                 reason="显式卸载 ✓（`unload` 就是要丢掉它 ✓）")
            self._backend = None
            self._loadKeyTuple = None
            self._loadReport = None
        # ⭐ LoRA 租约必须**先还** ✓✗（E2.5 ✓）：`release_all` 只回答"账本里还有没有未归还的引用" ✓，
        #    不替后端还账 ✓ —— 模型都拆了还报 `outstanding > 0`，那个字段就成了**假泄漏** ✓✗
        #    （以后真泄漏时没人会再信它 ✓）。
        lora_leases: dict[str, Any] | None = None
        if hasattr(backend, "release_loras"):
            lora_leases = backend.release_loras(reason="模型卸载 ⇒ 先还租约 ✓")
        del backend                          # 引用一断 ⇒ 大张量才真被回收 ✓（只置 None 不算 ✓✗）
        # ⚠️ **必须再显式回收一次** ✗：`int8_offload` 把 **17.94 GiB 钉在 CPU 锁页内存**里 ✓，
        #    它靠引用计数回收 ✓ —— 只要有**一处循环引用**（LoRA 补丁 ✓ / 模型上的缓存字段 ✓），
        #    `del` 之后这 17.94 GiB 仍会吊着 ✓✗，而症状跑到**下一次装载**去：
        #    锁页配额不足 / 内存直接翻倍 ✗（读不到"上一份没放掉"这个原因 ✓）。
        #    成本几十毫秒 ✓ —— 而必须排在 `_reclaim_cuda_cache()` **之前** ✓：
        #    后者只还**已释放**的缓存块 ✗（顺序反了 ⇒ 显存 reserved 还挂着 ✓✗）。
        #    ⚠️ 别把这两行当成重复删除 ✗：`_reclaim_cuda_cache()` 内部确实也会 gc 一次 ✓，
        #    但那发生在它自己的流程里 ✓✗ —— 这里要的是「**先**把 CPU 锁页清掉、再进还缓存」✓，
        #    并且把这个计数**报出去** ✓（否则它只发生在黑盒里 ✗，出问题时无从判断 ✓✗）。
        #
        # ⭐ 2026-10-29 **真机实测**（A5000 22.49 GiB / 96 GB / ``int8_offload=True`` ✓ ——
        #    不是推理 ✓）。跑法三步、照做即可复现 ✓（一次性探针本机跑完即弃 ✗ 不进版本库 ✗）：
        #    **装 → 卸 →** ``SetProcessWorkingSetSize(GetCurrentProcess(), -1, -1)`` **再量 RSS →**
        #    **接着再装一份看是否翻倍** ✓（少任一步都判不出来 ✓✗ —— 见下面 ② ✓）：
        #    ① 装：RSS **0.04 → 28.76 GiB** ✓，其中 ``int8Residency.parkedBytes`` = **17.94 GiB** ✓、
        #       ``int8Ledger`` **200 层全在 CPU**（``int8OnGpu=0`` ✓）；装配峰值 **75.86 GiB** ✓
        #       （＝ :func:`_prepare_h3_module` 那份临时 fp32 主干 ✓，与注释里的 78 GiB 对得上 ✓）。
        #    ② 卸：RSS **28.76 → 28.76（一点没动）** ✗ —— ⚠️ **但这不代表没还** ✗✗：
        #       Windows 的工作集**不会**因为 ``free`` 就收缩 ✓，页进 standby 仍算在 RSS 里 ✓✗。
        #       用 ``SetProcessWorkingSetSize(handle, -1, -1)`` 收缩之后 ⇒ **0.00 GiB** ✓
        #       （比起点还低 ✓）⇒ 那 28.72 GiB 的物理页**真没了** ✓。
        #    ③ 再装同一份：**成功 ✓**、RSS 只 **+18.02**（≈ 第一份的量 ✓，**不是翻倍** ✗✗）、
        #       且 ``gcCollected=0`` ✓ —— 印证「对象是**引用计数**走的、没有循环垃圾」✓
        #       ⇒ **原来担心的"循环引用吊着 17.94 GiB"实测没发生 ✓**。
        #    ⇒ 但**另一个东西确实不还** ✗：CPU 侧**提交内存**（``PagefileUsage``）卸完仍是
        #      **30.32 GiB** ✓✗ —— 因为那 17.94 GiB 锁页是 PyTorch **CachingHostAllocator 的池** ✓，
        #      而 :func:`_reclaim_cuda_cache` 只管 ``torch.cuda.*``（**显存** ✓），CPU 一个字节都不还 ✓✗。
        #    ⚠️ 这**不是缺陷、别去"修"** ✗✗：池里留着的**正是下次装载复用的那 17.94 GiB** ✓
        #      （实测第二次装完 RSS 只 18.06 ✓ ＝ 池被复用、没再分一份 ✓）。
        #      真要清得用 ``torch._C._accelerator_emptyHostCache()`` ✓，代价是下次重新
        #      ``cudaHostAlloc`` 17.94 GiB —— **秒级** ✓（不是下面 :func:`_reclaim_cuda_cache`
        #      里描述的"毫秒级 cudaMalloc" ✗，那条讲的是**显存**块 ✓）⇒ 只有**要把内存腾给别的进程**时才值得 ✓。
        #    ⚠️ 顺带记一笔：那次脚本里第二次装载花了 **110.9s** ✗ —— **不作数** ✗✗，
        #      因为它前面刚做过 ``SetProcessWorkingSetSize`` ⇒ Python 代码与 torch 的页
        #      全要被重新 page in ✓；**不 trim** 的正常换装配没有这份开销 ✓。
        # ⭐ LoRA 账本 ✓（E2.5 ✓）：模型一没，借来的 LoRA 张量就该还 ✓ —— 在此之前**根本没有账本** ✗，
        #    同一个 LoRA 挂两次就是两份张量 ✓✗（`lora_cache` 现在按**内容 sha256** 只留一份 ✓）。
        #    ⚠️ 三个读数各管一段 ✓✗，别只看最后一个（它混了 gc + allocator 缓存 ✓）：
        #    ``before`` → ``afterRelease`` 的 **allocated 差**才是**LoRA 张量真放掉**的那部分 ✓；
        #    ``afterReclaim`` 再降的是 allocator 还回驱动的缓存块 ✓（两件事 ✓）。
        vram_before = _vram_facts()
        lora_freed = lora_cache.shared().release_all(reason="模型卸载 ⇒ LoRA 张量整份清掉 ✓")
        vram_released = _vram_facts() if vram_before is not None else None
        collected = gc.collect()
        reclaim = _reclaim_cuda_cache()
        vram_after = _vram_facts() if vram_before is not None else None
        return {"unloaded": had, "device": device, "gcCollected": collected,
                "cacheCleared": bool(reclaim.get("released")),
                "lora": {**lora_freed,
                         "leases": lora_leases,
                         "vramBefore": vram_before,
                         "vramAfterRelease": vram_released,
                         "vramAfterReclaim": vram_after,
                         "note": ("`vramBefore → vramAfterRelease` 的 allocated 差是 **LoRA 张量**"
                                  "真放掉的部分 ✓；再往后是 allocator 缓存 ✓；"
                                  "三项为 ``None`` ⇒ 没 torch / 没卡 ✓（**不是 0** ✓✗）")},
                **reclaim}

    # ── 装配量计 ✓（E3.4 ✓，2026-10-29 ✓）────────────────────────────────
    #: `status().cache.recent` 留几条 ✓（够看出"刚才发生了什么"✓；留全量 ✗ 长跑进程会涨 ✓✗）
    _CACHE_EVENT_KEEP = 8

    @staticmethod
    def _key_digest(key: Any) -> str | None:
        """装配键 ⇒ 短摘要 ✓（键里是绝对路径 + 布尔 ✓，太长不宜直接给前端；完整键在
        ``loadReport`` 与装配参数里 ✓ —— 这里只要"是否同一个"✓）。"""
        if not key:
            return None
        return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:12]

    def _note_cache(self, kind: str, *, key: Any, reason: str) -> None:
        """记一笔装配量计 ✓ —— ⚠️ **必须在 ``self._lock`` 里调用** ✗✗（与 :meth:`status` 的快照
        同一把锁 ✓，否则「计数涨了、事件还没进列表」这种半截状态会被读到 ✓✗）。"""
        if kind == "hit":
            self._cacheHits += 1
        elif kind == "miss":
            self._cacheMisses += 1
        elif kind == "eviction":
            self._cacheEvictions += 1
        else:
            raise ValueError(f"不认识的量计种类 {kind!r} ✗（只有 hit / miss / eviction ✓）")
        self._cacheEvents.append({"at": _now_ms(), "kind": kind,
                                  "key": self._key_digest(key), "reason": reason})
        del self._cacheEvents[:-self._CACHE_EVENT_KEEP]

    def _cache_facts(self) -> dict[str, Any]:
        """装配量计的事实 ✓（⚠️ 调用方持锁 ✓ —— 三个数与 ``recent`` 必须是同一瞬间的 ✓）。

        ⚠️ **口径只到「装配复用门」** ✗✗，别读成"引擎的缓存全在这"：
        `cache_guard` / `latent_container` 是**纯函数模块** ✓ 一个计数器都没有 ✗；
        torch 自己的 allocator 缓存也不在这里 ✓（那是 ``reclaim_cache`` / 装载报告里 ``cache``
        那份事实的活 ✓ 两回事 ✗）。
        """
        return {
            "hits": self._cacheHits,
            "misses": self._cacheMisses,
            "evictions": self._cacheEvictions,
            "since": self._cacheSince,
            "recent": [dict(event) for event in self._cacheEvents],
            "scope": ("本进程内 EngineRuntime 的**装配复用门**（复用 / 真装 / 丢弃 ✓）；"
                      "不含 torch allocator 缓存、不含 hybrid 合并文件缓存 ✓"),
            "note": ("⚠️ 进程重启即归零 ✗ ⇒ 读到 0 只说明「这一程还没复用过」✓，"
                     "**不是**「从没复用」✓✗"),
        }

    # ── 自述 ✓ ───────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        """运行时的**全部事实** ✓（依赖 / 装了没 / 忙不忙 / 队列 / 最近任务 / **装配量计** ✓ /
        **LoRA 账本** ✓）。"""
        available, reason = torch_available()
        with self._lock:
            current = self._current
            queued = len(self._queue)
            tasks = [self._tasks[task_id].to_summary() for task_id in list(reversed(self._order))[:10]]
            report = dict(self._loadReport or {}) or None
            cache = self._cache_facts()
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
            #: ⭐ 复用了几次 / 真装了几次 / 丢掉了几次 ✓（E3.4 ✓）—— 以前只能从 `loadReport.reused`
            #: 看"这一次"，看不出"这一程"✓✗（`loadReport` 是**最后一份**的报告 ✓，装过几次就只留最后一次 ✓）。
            "cache": cache,
            #: ⭐ LoRA 账本 ✓（E2.5 ✓）：同内容只留**一份**张量 ✓、几个引用 ✓、丢了几个字节 ✓。
            #: ⚠️ 与上面的 `cache` 是**两件事** ✓✗：`cache` 记的是"主干装配复用了几次"（次数 ✓），
            #: 这里记的是"LoRA 张量现在占多少"（字节 ✓）—— 混着看会得出"复用越多占越多" ✓✗。
            "lora": lora_cache.shared().facts(),
            "backend": described,
            "busy": current is not None or queued > 0,
            "current": current, "queued": queued,
            #: ⭐ 限额也要**可观测** ✓：只报「排了 3 个」而不报「上限 32」，读者没法判断「还剩多少余量」✓✗。
            "queueLimit": self._maxQueue,
            "taskHistoryLimit": self._maxTasks,
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
