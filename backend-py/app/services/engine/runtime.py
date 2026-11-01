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
from dataclasses import dataclass, field, replace
from typing import Any, NoReturn

from . import audio_vae as audio_vae_mod
from . import chain
from . import device_policy
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
#: ⚠️ 判族常量**单独引** ✗（不整包铺开 ✓）：装配时要拿它核「`ditPath` 到底是不是 Z-Image」✓ ——
#: 判错的后果是「装得进去、跑得出来、图是噪声」✓✗，见 :meth:`EngineRuntime._ensure_zimage_loaded` ✓。
#: ⚠️ ``QWEN21_FAMILY`` 同理 ✓：2.1 与 2512 的 `text_dim`/VAE 通道**全一样** ✓✗ ⇒ 三道互核全过 ✓，
#: 只有族名拦得住 ✓（见 :meth:`EngineRuntime._ensure_qwen21_loaded` ✓）。
from .torch_backend import QWEN21_FAMILY, QWEN_IMAGE_FAMILY, ZIMAGE_FAMILY

__all__ = [
    "EngineBusy",
    "EngineRuntime",
    "EngineTask",
    "EngineUnavailable",
    "engine_runtime",
]

#: 终结态 ✓（到了这三种就不再变 ✓ —— 别用 `status != "running"` 判 ✗：``queued`` 也不是终结 ✓✗）
_TERMINAL = ("done", "failed", "cancelled")

#: ``stage=zimage`` 的装配要**三份权重 + 一本词表** ✓（2026-10-31 起装配口收齐 ✓）：
#: ``ditPath`` ✓ / ``textWeights`` ✓ / ``imageVaeWeights`` ✓ / ``tokenizerPath`` ✓。
#: ⚠️ 这份文案只在**确实缺格**时出现 ✗ —— 收齐了就真装（不拒 ✓）。
_ZIMAGE_STAGE_NOTE = (
    "Z-Image 的装配要**三份权重 + 一本词表** ✓：DiT（`ditPath` ✓）+ **Qwen3 文本塔**"
    "（`textWeights` ✗）+ **Flux VAE**（`imageVaeWeights` ✗）+ Qwen3 词表（`tokenizerPath` ✗）。"
    "⚠️ 这里**不猜路径** ✗：猜错会「装得进去、跑得出来、结果是噪声」✓✗（比装不上糟得多 ✓）。"
    "三份权重各是什么：`unet/z_image_turbo_bf16.safetensors`（453 键 ✓）、"
    "`text_encoders/qwen_3_4b.safetensors`（398 键 ✓，**不是** Qwen3-VL ✗）、"
    "`vae/ae.safetensors`（244 键 ✓，**是 Flux 的 ae** ✗ 不是 `qwen_image_vae`）；"
    "词表是 `tokenizers/qwen3vl_32b/` 这种**目录** ✓（含 `tokenizer.json` ✓，不是 SDXL 的 "
    "`sd1_tokenizer` ✗）。"
)

#: ``stage=qwen_image21`` 的装配同样要**三份权重 + 一本词表** ✓（2026-10-31 ✓）：
#: ``ditPath`` ✓ / ``textWeights`` ✓ / ``imageVaeWeights`` ✓ / ``tokenizerPath`` ✓。
#: ⚠️ 这三份**与 Z-Image 那三份一件都不能互换** ✗✗ —— 两族都是「DiT + 文本塔 + VAE」，
#: 形状还都"看着能对上" ✓：拿 Z-Image 的 Qwen3-4B（`text_dim=2560` ✓）配 2.1 主干
#: （`text_dim=4096` ✓）会在 `_check_text_encoder` 当场被拒 ✓（这条能拦住 ✓），
#: 但拿 **25/12 的 qwen_image VAE** 顶 2.1 的 **64 通道** VAE ✓✗ ⇒ 通道数一核就拒 ✓；
#: 真正拦不住的是「同族不同代」的 DiT 权重 ✓✗（2.1 与 2512 的键数 649 vs 1933 ⇒ 键集能区分 ✓）。
_QWEN21_STAGE_NOTE = (
    "Qwen-Image **2.1** 的装配要**三份权重 + 一本词表** ✓：DiT（`ditPath` ✓）+ **Qwen3.5 文本塔**"
    "（`textWeights` ✓）+ **64 通道 VAE**（`imageVaeWeights` ✓）+ Qwen3.5 词表（`tokenizerPath` ✓）。"
    "⚠️ **不猜路径** ✗：猜错会「装得进去、跑得出来、图是噪声/糊」✓✗。三份长这样："
    "`diffusion_models/qwen_image_2.1_int8_convrot.safetensors`（**649** 键 ✓，不是 2512 的 1933 ✗）、"
    "`text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors`"
    "（语言塔 + **视觉塔 + lm_head** 同文件 ✓ —— 加载器会自动单列后两族 ✓）、"
    "`vae/qwen_image_2.1_vae_bf16.safetensors`（**238** 键 ✓，Wan 式 3D 因果 VAE ✓、"
    "**不是** `qwen_image_vae`（那是 2512 的 ✗ 通道数不对 ✓））。"
)

#: ⭐ H3 那三件的**驻留账** ✓（GiB ✓）—— 与 `TorchBackend.PART_NEED_GIB`（Qwen-Image 2.1 的 ✓）
#: **必须分开** ✗✗（两张表差 11 GiB ✓✗）。三个数是**盘上文件大小 ＋ dtype 直接读出来的** ✓：
#:
#: | 部件 | 文件 | 盘上 | 驻留 | 为什么 |
#: |---|---|---|---|---|
#: | DiT | ``h3_ref2va_pruned_int8_convrot`` ✓ | 19.53 GiB ✓ | **19.62** ✓ | int8 常驻（`Int8Linear` ✓ 不反量化 ✓）＋ fp32 岛/输出头/buffers 抬到 bf16 ✓ |
#: | 文本塔 | ``qwen3vl_32b_minimax_h3_int8_convrot`` ✓ | 25.28 GiB ✓ | **25.44** ✓ | 同上 ✓；其中 int8 部分 **22.7 GiB** ✓、非 int8（视觉塔 ＋ 嵌入 ＋ 范数）**2.57 GiB** ✓ |
#: | 视频 VAE | ``h3_video_vae`` ✓ | 5.2 GiB ✓ | **5.2** ✓ | 默认落 **cpu** ✓（不占卡 ✓） |
#:
#: ⚠️ **驻留数不许按文件大小抄** ✗✗（2026-11-01 端到端实测纠了一遍 ✓）：DiT 原来记的是
#: ``17.94``（＝更早那份 pruned 文件的字节数 ✓），可实测 `memory_allocated` 是 **19.62 GiB** ✓
#: —— 差的 1.68 GiB 是"文件字节 ≠ 驻留字节"（int8 之外的那些参数要抬到 bf16 ✓）。
#: 记小了会**误判装得下** ✓✗：按 17.94 判「余 21.28 ≥ 17.94+1 ⇒ 整份上卡」✓，实际装上占 19.62 ⇒
#: 只剩 **1.3 GiB** 给采样工作区 ✓（这才是「明明装下了、采样第 1 步就 OOM」的真因 ✓）。
#: TE 同样按实测 25.44 记（差 0.16 GiB 属毛刺 ✓ 但一律以实测为准 ✓）。
#:
#: ⚠️ 拿 2.1 的 ``{"dit": 13.26, "text": 14.79}`` 判 H3 ✗✗：DiT 少算 6.4 GiB ✓、TE 少算 10.6 GiB ✓
#: ⇒ 决策说"装得下" ✓、`residency` 留 ``off`` ✓（= 不切 ✗）、装到一半 OOM ✓，且 OOM 落在
#: **后装的那件**上 ✓✗（读起来像"TE 坏了" ✗，实际是两张表混用 ✓）。
H3_PART_NEED_GIB: dict[str, float] = {"dit": 19.62, "text": 25.44, "vae": 5.2}

_QWEN_IMAGE_STAGE_NOTE = (
    "Qwen-Image **2512 / Edit 2511** 的装配要**三份权重 + 一本词表** ✓：DiT（`ditPath` ✓）+ "
    "**Qwen2.5-VL-7B 文本塔**（`textWeights` ✓）+ **16 通道 VAE**（`imageVaeWeights` ✓）+ "
    "Qwen2.5 词表（`tokenizerPath` ✓）。"
    "⚠️ **不猜路径** ✗：这一族与 2.1 的 `text_dim` / VAE 通道**都不一样** ✓，配错就是「装得进去、"
    "图是糊的/脱钩的」✓✗。五份长这样："
    "`diffusion_models/qwen_image_2512_bf16.safetensors`（**1933** 键 ✓、bf16 38.05 GiB ✓；"
    "Edit 是 `qwen_image_edit_2511_int8_convrot.safetensors` ✓、带 `__index_timestep_zero__` ✓）、"
    "`text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`（语言塔 + **视觉塔 + lm_head** 同文件 ✓ —— "
    "只装语言塔 ✓、后两族单列不装 ✓）、"
    "`vae/qwen_image_vae.safetensors`（**16** 通道 ✓、8 倍 ✓ —— **不是** 2.1 的 64 通道 ✗）、"
    "词表是 Qwen2.5-VL 的**目录** ✓（含 `tokenizer.json` ✓，特殊符 `<|im_start|>=151644` ✓）。"
)


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


def _abandon_assembly(backend: Any, err: BaseException, *, what: str = "") -> NoReturn:
    """装配失败 ⇒ **丢掉整个候选后端** ✓✗（模块一个字不变 ✓），并把 ``reason`` **原样**传下去 ✓。

    ⚠️⚠️ 三件事必须**一起**做 ✗✗（2026-10-31 收口 ✓）：四条装配线各自手抄了一段
    ``except TorchBackendUnavailable`` ＋ 一段 ``except BaseException`` ✓✗，其中 SDXL 那条
    **少抄了一整段** ✓✗ —— 见下面第 3 条。

    1. ``del backend`` ✓✗：只靠"局部变量出作用域"**不算** ✗ —— 异常仍挂在 traceback 里被引用 ✓，
       不显式删就留到栈帧销毁才可能回收 ✓（G 级张量 ⇒ 装下一个模型时峰值翻倍 ✓）。
    2. ``_release_torch_cache()`` ✓✗：不还缓存时 ``reserved`` 仍占着整卡 ✓（症状：
       "明明卸载了，新装配还是 OOM" ✓）。
    3. **``reason`` 必须原样往下传** ✗✗：SDXL 那条只有 ``except BaseException`` 一段 ⇒
       它内部抛的 ``TorchBackendUnavailable(reason="deps")``（缺 torch）被重包成
       ``reason="pending"``（权重没就绪）✓✗ —— 调用方照文档去补权重白忙 ✓，
       而真正该做的是装依赖 ✓。**报错分类错 = 指错方向** ✓，比不报更贵 ✓。

    ⚠️ 正因为第 3 条要分派，调用方**只需要一段** ``except BaseException`` ✓✗
    （两段 except 是历史写法 ✓；``str(err)`` ＋ 换 reason 的路径在下面统一 ✓）。
    """
    del backend                              # 引用一断 ⇒ traceback 里的那份才真被回收 ✓
    _release_torch_cache()
    reason = str(getattr(err, "reason", "") or "pending")
    if isinstance(err, TorchBackendUnavailable):
        raise TorchBackendUnavailable(str(err), reason=reason) from err
    prefix = f"{what} " if what else ""
    raise TorchBackendUnavailable(f"{prefix}装配失败：{type(err).__name__}: {err}",
                                  reason=reason) from err


#: 「**没传**」与「传了个 ``None``」要分开 ✓✗：``pressure=None`` 是**有意义的**输入 ✓
#: （＝测不到卡况 ✓），不能拿来当"没传"的哨兵 ✗（否则注入测不到卡况的场景时，
#: 函数反而会自己去探一次真卡 —— 测试就变成了「看这台机器卡忙不忙」✓✗）。
_UNSET: Any = object()


def _chain_device_decision(*, part: str, device: str | None, need_bytes: int,
                           policy: Any = None, pressure: Any = _UNSET,
                           gpu_available: bool | None = None,
                           split_note: str) -> dict[str, Any] | None:
    """**整条链**该在哪台设备上装配 ✓（``None`` ⇒ 用户钉死了设备 ✓ / 不需要决策 ✓）。

    三条线共用这一份 ✓✗（SDXL 四件套 ✓ / Z-Image 三件同驻 ✓ / 2.1 装配前那一次复判 ✓），
    为的是把「判成 ``split`` 但**这条线落不了地** ⇒ 按装不下处理、整份转 CPU」这条规则
    **只写一处** ✓ —— 三处各写一遍必然改一处忘两处 ✓✗（那种错不报异常 ✓，只是在卡上硬塞到 OOM ✓）。

    ``need_bytes``：**只接受已经量出来的数** ✓（SDXL 取 ``os.stat`` 的文件字节 ✓、Z-Image 取
    :attr:`TorchBackend.CHAIN_NEED_GIB` 的实测账 ✓）；量不出来 ⇒ 传 ``0`` 走「需求未知」的
    占用率粗判 ✓（**不编一个 GiB 数** ✗✗ —— 那会让报告看着精确、其实全是猜的 ✓✗）。

    ``split_note``：这条线**为什么**落不了 ``split`` ✓ —— 会拼进 ``reason`` 里一起报出去 ✓✗
    （只写"降级"的话，读者分不清「卡不够」还是「这条线本来就不支持」✓✗）。

    ``pressure`` / ``gpu_available`` 是**注入位** ✓（自检必需 ✓✗）：不注入就现读真卡 ✓，
    注入后**不再碰真机** ✓ —— 否则「装不下 ⇒ 转 CPU」这条断言在卡空着的机器上永远走不到 ✓✗。
    """
    if str(device or "").strip().lower() not in ("", "auto"):
        return None
    engine_torch: Any = None
    if gpu_available is None:
        gpu_ok = False
        try:
            import torch as engine_torch  # noqa: PLC0415 —— 可选依赖 ✓
            gpu_ok = bool(engine_torch.cuda.is_available())
        except Exception:  # noqa: BLE001 —— 没装 torch ⇒ 当「没卡」✓（各后端自己会报依赖缺 ✓）
            gpu_ok = False
    else:
        gpu_ok = bool(gpu_available)
        if gpu_ok:
            try:
                import torch as engine_torch  # noqa: PLC0415
            except Exception:  # noqa: BLE001
                engine_torch = None
    if pressure is _UNSET:
        pressure = device_policy.probe(engine_torch) if gpu_ok else None
    picked = device_policy.decide(
        policy=device_policy.normalize_policy(policy), requested_device=device,
        need_bytes=max(0, int(need_bytes)), pressure=pressure,
        gpu_available=gpu_ok, phase="assemble", part=part)
    if picked.residency == "split":
        #: ⚠️ ``split`` 有**两个来源** ✓✗，不能一并推去 CPU ✗✗：
        #:   · 「**装不下**整件、但还剩 ≥0.5 GiB」⇒ 联合是唯一不 OOM 的活法 ✓
        #:     ⇒ 这条线落不了地就只能整份转 CPU ✓（慢，但跑得完 ✓）；
        #:   · 「**装得下**，但 `balanced` 主动想省显存」⇒ 整份上卡**不是**降级 ✓✗
        #:     ⇒ 落不了地就照常上卡 ✓（为了"没能省那点余量"把速度砍一个数量级是错的 ✓✗）。
        fits = (pressure is not None
                and int(pressure.freeBytes) >= max(0, int(need_bytes)) + device_policy.HEADROOM_BYTES)
        if fits:
            picked = replace(picked, residency="off", degraded=False,
                             reason=picked.reason + "；⚠️ " + split_note
                             + " ⇒ 实际按**整份上卡** ✓（装得下 ✓，只是没能按联合摊开 ✗）")
        else:
            picked = replace(picked, device="cpu", residency="off", degraded=True,
                             reason=picked.reason + "；⚠️ " + split_note + " ⇒ 这一档落到 CPU ✓")
    return picked.to_dict()


def _image_device_decision(*, dit_path: str | None, device: str | None,
                           policy: Any = None, need_bytes: int | None = None,
                           pressure: Any = _UNSET,
                           gpu_available: bool | None = None) -> dict[str, Any] | None:
    """SDXL 出图这条线的**设备决策** ✓（判据与口径见 :func:`_chain_device_decision` ✓）。

    ⚠️ 这里只管一件事 ✗：把需求量**量出来** ✓（:func:`os.stat` 盘上真字节 ✓）——
    SDXL 是**单个** checkpoint ✓、权重无量化 ✓ ⇒ 文件大小就是驻留量的下界 ✓
    （实测装载后真占 6.73 GiB ✓）。读不到大小 ⇒ ``0`` ⇒ 走占用率粗判 ✓（不猜 ✗）。
    """
    if str(device or "").strip().lower() not in ("", "auto"):
        return None
    if need_bytes is None:
        need_bytes = 0
        try:
            if dit_path:
                need_bytes = int(os.stat(dit_path).st_size)  # noqa: PTH116 —— 只要大小 ✓
        except OSError:
            need_bytes = 0                  # 读不到 ⇒ 「需求未知」✓（不猜 ✗）
    return _chain_device_decision(
        part="sdxl", device=device, need_bytes=need_bytes, policy=policy,
        pressure=pressure, gpu_available=gpu_available,
        split_note="SDXL 线没有分层驻留（四件套整份落位 ✓）")


def _zimage_device_decision(*, device: str | None, policy: Any = None,
                            pressure: Any = _UNSET,
                            gpu_available: bool | None = None) -> dict[str, Any] | None:
    """Z-Image 这条线的**设备决策** ✓（判据见 :func:`_chain_device_decision` ✓）。

    ⚠️ 需求量取 :attr:`TorchBackend.CHAIN_NEED_GIB` 的**整链同驻实测账** ✓✗（19.28 GiB ✓）——
    这条线 DiT ＋ TE ＋ VAE **同时在卡上** ✓，不是 2.1 那种一段一段 ✓。
    拿「最重的那一件（11.46）」去判会以为卡上剩 12 GiB 就够 ✓✗ ⇒ DiT 装进去了 ✓、
    装 TE 时才 OOM ✓，报错落在 TE 上、看着像"文本塔坏了" ✓✗（这个误导实测踩过一次 ✓）。

    ⚠️ 落不了 ``split`` 的理由与 SDXL 不同 ✗：这条线的三件**各自整份 ``to(device)``** ✓，
    没有"按条切前缀"这种口径 ✓ ⇒ 判成联合也落不了地 ✓✗（真要联合得按件拆预算 ✓，
    本仓没有这条实现 ⇒ 不假装有 ✗✗）。
    """
    need_gib = float(TorchBackend.CHAIN_NEED_GIB.get("zimage") or 0.0)
    return _chain_device_decision(
        part="zimage", device=device,
        need_bytes=int(need_gib * device_policy.GIB), policy=policy,
        pressure=pressure, gpu_available=gpu_available,
        split_note="Z-Image 三件**同时在卡上**（DiT＋TE＋VAE 各自整份落位 ✓、没有按条拆分 ✗）")


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
                      text_weights: str | None = None,
                      image_vae_weights: str | None = None,
                      int8_offload: bool = False,
                      residency: str = "off", residency_budget_mib: float | None = None,
                      policy: str = "auto",
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

        ``residency`` / ``residency_budget_mib``（2026-10-31 新增 ✓，**默认 off** ✗）：开 ⇒ **卡 + CPU
        联合跑** ✓（``split`` 卡上算前缀、CPU 算剩下的 ✓；``park`` 权重留内存、逐层搬上卡 ✓，
        口径见 `residency` ✓）。创作者原话：「以后不能只走 CPU，也要可以 GPU 和 CPU 联合跑模型，
        单个 CPU 很容易超载」✓。⚠️ 两者**必须进装载键** ✗✗：同一份权重、同一张卡，
        「卡上 30 层」与「卡上 0 层」的运行速度差**一个数量级** ✓✗ ⇒ 不进键就会复用错的那一套 ✓✗。

        ``text_weights`` / ``image_vae_weights``（2026-10-31 新增 ✓，**只对 ``stage=zimage`` 有意义** ✓）：
        Z-Image 是**三份权重**拼起来的一条链 ✓（DiT ✓ + Qwen3-4B 文本塔 ✓ + Flux VAE ✓），
        三者**不在同一个文件里** ✗✗ ⇒ 必须各给一格 ✓。缺任一格 ⇒ **当场拒** ✗ 并说清缺哪个
        （**不猜路径** ✗：猜错是「装齐了、跑得出、图是噪声」✓✗，比装不上糟得多 ✓）。
        ⚠️ 两者**必须进装载键** ✗✗：换 TE 或换 VAE 就是**换了一个模型** ✓ —— 不进键就会
        「给了新的 qwen_3_4b 却复用上一套」✓✗，而且一声不响 ✓✗。
        """
        #: ⚠️ 先**归一**再进键 ✗✗：``None`` / ``"auto"`` / ``"gpu-first"`` 是同一种策略 ✓，
        #: 但 ``repr`` 进键后是**三个不同的键** ✓✗ ⇒ 会白装三遍，而且 "改了个写法" 的复用键也不稳 ✓✗。
        #: 拼错的策略名在这里**当场抛** ✓（不静默回落 ✓）。
        policy = device_policy.normalize_policy(policy)
        key = _load_key(stage=stage, dit_path=dit_path, tokenizer_path=tokenizer_path,
                        device=device, audio_latent_mode=audio_latent_mode,
                        attach_video_vae=attach_video_vae, attach_audio_vae=attach_audio_vae,
                        real_video_vae=real_video_vae, video_vae_device=video_vae_device,
                        video_vae_weights=video_vae_weights,
                        text_weights=text_weights,
                        image_vae_weights=image_vae_weights,
                        int8_offload=int8_offload,
                        residency=residency, residency_budget_mib=residency_budget_mib,
                        #: ⚠️ **策略必须进键** ✗✗：同一份权重、同一张卡，`auto` 与 `cpu` 装出来的
                        #: 是**两套设备布局** ✓（一个在卡上、一个在内存里 ✓）⇒ 不进键就会「改了策略却复用
                        #: 上一套」✓✗，而且一声不响 ✓✗（症状是"改了没效果"，比报错难查得多 ✓）。
                        policy=policy,
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
                                            text_weights=text_weights,
                                            image_vae_weights=image_vae_weights,
                                            int8_offload=int8_offload,
                                            residency=residency,
                                            residency_budget_mib=residency_budget_mib,
                                            policy=policy,
                                            loras=loras)

    def _reset_loaded(self) -> bool:
        """丢掉当前装载的**三格账** ✓（``_backend`` / ``_loadKeyTuple`` / ``_loadReport``）
        ⇒ 返回「原来有没有装」✓（调用方的 ``report["replaced"]`` 读它 ✓）。

        ⚠️⚠️ **三格必须一起清** ✗✗（2026-10-31 收口 ✓）：这段话此前在**四条装配线**里各手抄了一遍 ✓✗，
        而 SDXL 那条（``_ensure_image_loaded``）只抄到 ``self._backend = None`` ✓✗ ——
        后果不是炸，而是**报告撒谎** ✗：装失败之后 ``_loadKeyTuple`` / ``_loadReport`` 还停在
        **上一个**模型上 ⇒ ``/engine/runtime`` 与 ``unload()`` 的回执答的是已经不在场的模型 ✓✗。
        （假复用**不会**发生 ✓：``reused`` 判据同时看 ``self._backend is not None`` ✓ ——
        所以这个错永远不报错，只是让人按错的模型估余量 ✓✗，正是本仓最贵的那一类 ✓。）

        ⚠️ 只置 ``None`` **不会**让 G 级张量被回收 ✓✗（本仓实测过 ✓）：``previous`` 是本函数的
        局部变量 ⇒ 返回时引用自动断 ✓，调用方随后调 :func:`_release_torch_cache` 才真还回去 ✓。

        ⚠️ ``unload()`` **不用**这个助手 ✓：它要在清空**之前**记 ``eviction`` 与读旧 ``device`` ✓
        （记账时机是那条口径的一部分 ✓），所以那里保持原样 ✓。
        """
        with self._lock:
            had = self._backend is not None
            self._backend = None
            self._loadKeyTuple = None
            self._loadReport = None
        return had

    def _require_slots(self, what: str, note: str,
                       slots: list[tuple[Any, str]]) -> None:
        """四格（或任意 N 格）缺任一 ⇒ **当场拒** ✓✗，且**首行只列真缺的几格** ✓。

        ⚠️⚠️ 收口理由（2026-10-31 ✓）：Z-Image 与 Qwen-Image 2.1 两条线各自手抄了一段
        "``missing: list[str] = []`` ＋ 四个 ``if not xxx: missing.append(...)`` ＋ ``if missing: raise``"
        共 **23 行** ✓✗。这段的坏处不在啰嗦，而在**纪律**：报错首行必须只列真缺的那几格 ✓
        （四格全列 = 调用方照着补也会漏、还把真缺的埋进说明里 ✓✗；测试也只读首行 ✓）。
        两处各写一遍 ⇒ 改一处口径另一处不会跟着改 ✓✗。

        ⚠️ 调用方登记顺序要**保持原有顺序** ✗（报错文本逐字进过测试与文档 ✓）。
        """
        missing = [label for value, label in slots if not value]
        if missing:
            raise TorchBackendUnavailable(
                f"{what} 装配缺格 ✗：还没给 " + "、".join(missing) + f"\n  · {note}",
                reason="pending")

    def _commit_load(self, key: tuple[Any, ...], backend: Any,
                     report: dict[str, Any]) -> None:
        """装配成功后**一次性**提交三格账 ✓（``_backend`` / ``_loadKeyTuple`` / ``_loadReport``）。

        ⚠️⚠️ 收口理由同 :meth:`_reset_loaded` ✓✗（2026-10-31）：这段 ``with self._lock:`` 五行
        在四条装配线里逐字抄了四遍 ✓✗。少抄一格的后果**方向相反**但一样静默：
        漏 ``_loadKeyTuple`` ⇒ 复用判据永远不成立 ⇒ **每次调用都白装一遍**（Z-Image 40 秒／19 GiB ✓✗）；
        漏 ``_loadReport`` ⇒ ``/load`` 回执与 ``/engine/runtime`` 的 ``loadReport`` 是空的 ✓✗
        （上一轮刚修完的「报告必须如实」在这里会整段消失 ✓）。

        ⚠️ 必须在**装配全部成功之后**调 ✓，不许提前 ✓：提前提交 ⇒ 随后抛异常时
        ``self._backend`` 指着半装的模型 ⇒ 下一次调用 ``reused=True`` 直接拿去出图 ✓✗。
        """
        with self._lock:
            self._backend = backend
            self._loadKeyTuple = key
            self._loadReport = report

    def _ensure_loaded_internal(self, *, stage: str = "h3", dit_path: str | None = None,
                               tokenizer_path: str | None = None, device: str | None = None,
                               audio_latent_mode: str | None = None,
                               attach_video_vae: bool = True,
                               attach_audio_vae: bool = True,
                               loras: Any = None,
                               real_video_vae: bool = False,
                               video_vae_device: str | None = None,
                               video_vae_weights: str | None = None,
                               text_weights: str | None = None,
                               image_vae_weights: str | None = None,
                               int8_offload: bool = False,
                               residency: str = "off",
                               residency_budget_mib: float | None = None,
                               policy: str = "auto") -> dict[str, Any]:
        """真正装配 ✓（**不带排队守卫** ✗ —— worker 自己持有当前任务，见 :meth:`_execute` ✓）。

        顺序**按依赖来** ✓（⚠️ 不是随便排的 ✗）：主 DiT 先装 ✓ ⇒ ``attach_text_encoder`` 才能
        用主干 ``text_dim`` **自动配** ``output_dim`` ✓；VAE 只做**互核** ✓（``load_weights`` 与已挂
        的 VAE 彼此校验 ✓ ⇒ 先挂后装 / 先装后挂都对 ✓）。
        ⚠️ 显存排班（谁先谁后、谁能在下一步前释放 ✓）是**另一件事** ✗ —— 见 ``loader.residency_plan`` ✓。
        ⚠️ 但"装完把**空着的**缓存还给驱动"是**这一件** ✓（装之前先 ``_release`` ✓ ⇒ 装完再
        :func:`_reclaim_cuda_cache` ✓ —— 一头一尾都还 ✓，见报告里的 ``cache`` 事实 ✓）。
        ⚠️ ``stage=zimage`` 走**独立的第三条装配线** ✓（:meth:`_ensure_zimage_loaded` ✓）——
        绝不能落到 H3 那条线上 ✗✗：那会挂上**参考 TE / 参考 VAE**（synthetic ✓）⇒
        「装齐了、跑得出、图是噪声」✓✗。而"缺格（``textWeights`` / ``imageVaeWeights`` /
        ``tokenizerPath``）"⇒ **当场拒** ✗ 并说清缺哪个 ✓（**不猜路径** ✗ —— 猜错就是
        「装得进去、跑得出来、结果是噪声」✓✗，比装不上糟得多 ✓）。
        """
        #: 归一（幂等 ✓）：本方法也可能被 :meth:`ensure_loaded` 之外的路径直接调 ✓ —— 键必须一致 ✓✗
        policy = device_policy.normalize_policy(policy)
        key = _load_key(stage=stage, dit_path=dit_path, tokenizer_path=tokenizer_path,
                        device=device, audio_latent_mode=audio_latent_mode,
                        attach_video_vae=attach_video_vae, attach_audio_vae=attach_audio_vae,
                        real_video_vae=real_video_vae, video_vae_device=video_vae_device,
                        video_vae_weights=video_vae_weights,
                        text_weights=text_weights,
                        image_vae_weights=image_vae_weights,
                        int8_offload=int8_offload,
                        residency=residency, residency_budget_mib=residency_budget_mib,
                        policy=policy,
                        loras=_lora_key(loras))
        with self._lock:
            if self._backend is not None and key == self._loadKeyTuple:
                return {**dict(self._loadReport or {}), "reused": True}

        #: ⭐ 2026-10-31：出图（SDXL）侧**已接** LoRA ✓✗（此前这里直接拒 ✓）——
        #: 三处分派 UNet / CLIP-L / CLIP-G 归 `SdxlBackend._apply_loras` ✓；
        #: kohya 那种下划线分层键由 `lora.kohya_module_name` 翻译 ✓。
        #: 挂不全 ⇒ `load_weights` 当场炸 ✓（不留「四件齐、档不对」的模型 ✓✗）。

        if str(stage or "").strip().lower() == pipe.ZIMAGE_STAGE:
            # ⚠️ Z-Image 是**第三套组件** ✗（DiT + Qwen3-4B 文本塔 + Flux VAE ✓）——
            #    既不等于 H3 那套（参考 TE/VAE 未训练 ✗）也不等于 SDXL 四件套 ✓。
            return self._ensure_zimage_loaded(
                key=key, dit_path=dit_path, tokenizer_path=tokenizer_path,
                text_weights=text_weights, image_vae_weights=image_vae_weights, device=device,
                policy=policy, loras=loras)

        if str(stage or "").strip().lower() == pipe.QWEN21_STAGE:
            # ⚠️ Qwen-Image 2.1 是**第四套组件** ✗（DiT 649 键 + Qwen3.5-9B + 64 通道 VAE ✓）——
            #    与 Z-Image 同为三件套 ✓，但**三件一件都不能互换** ✗✗（见 `_QWEN21_STAGE_NOTE` ✓）。
            return self._ensure_qwen21_loaded(
                key=key, dit_path=dit_path, tokenizer_path=tokenizer_path,
                text_weights=text_weights, image_vae_weights=image_vae_weights, device=device,
                policy=policy, loras=loras)

        if str(stage or "").strip().lower() == pipe.QWEN_IMAGE_STAGE:
            #: ⚠️ **族名比较一律 `==`** ✗ 不许 `startswith` ✓：``"qwen_image21"`` 以
            #: ``"qwen_image"`` 开头 ✓ —— 用前缀判，2.1 会连同上面那条一起命中 ✓✗
            #: （两条装配线抢着装同一次请求，后装的把先装的顶掉 ✓，报错还落在"消失的那件"上 ✓✗）。
            #    2512 / Edit 2511 与 2.1 是**两代** ✓：双流 vs 单流、16ch vs 64ch、
            #    `1000·σ` 不翻转 vs 翻转、3.1 vs 0.69、Qwen2.5 3584 vs Qwen3.5 4096 ✓✗
            #    —— 一件都不能互换（见 `_QWEN_IMAGE_STAGE_NOTE` ✓）。
            return self._ensure_qwen_image_loaded(
                key=key, dit_path=dit_path, tokenizer_path=tokenizer_path,
                text_weights=text_weights, image_vae_weights=image_vae_weights, device=device,
                policy=policy, loras=loras)

        if str(stage or "").strip().lower() == pipe.IMAGE_STAGE:
            # ⚠️ 出图是**另一套组件** ✗（SDXL 四件套 ✓ ≠ H3 DiT + 参考 TE/VAE ✓）⇒ 分开装配 ✓。
            report = self._ensure_image_loaded(key=key, dit_path=dit_path,
                                               tokenizer_path=tokenizer_path, device=device,
                                               policy=policy, loras=loras)
            if str(residency or "off").strip().lower() != "off":
                # ⚠️ **不生效要说** ✗（同 `int8_offload` 在 DiT 上那条口径 ✓）：出图那条路（SDXL）
                #    还没接联合驻留 ✓ —— 它本来也不需要（四件套几 GB ✓），但"配了没生效"必须看得见 ✗。
                report["residency"] = {
                    "mode": residency, "applied": False,
                    "note": "出图（SDXL 四件套）那条路**还没接**联合驻留 ✗ —— `residency` 只在 "
                            "H3 / DiT 视频那条路上生效 ✓（它本来就不需要：四件套几 GB ✓）"}
            return report

        # ⚠️ 装新的之前**先把旧的放掉** ✗✗（与 :meth:`_ensure_image_loaded` 同一条规矩 ✓）：
        #    H3 主干 int8 是 **17.94 GiB** ✓（offload 时全在 CPU 锁页 ✓）、全量 19.53 GiB ✓
        #    ⇒ 旧的活着就装新的 = 同时压两份大模型 ✓✗（⚠️ 22.49 GiB 的卡上 17.94 GiB 还在显存里
        #    时再装一份**必 OOM** ✗）。
        #    ✅ 2026-11-01 起装配期**不再**临时建 fp32 主干 ✓（``_prepare_h3_module`` 改成 meta 建壳
        #    + 按最终精度物化 ✓，峰值 ≈ 19.5 GiB 而不是 78 GiB ✓）—— 但"先丢旧的"仍然必要 ✓：
        #    两份 19.5 GiB 一样会把这台 96 GB 的机器压到只剩十几 GiB ✓✗。
        #    丢旧的可恢复（重装一次 ✓）；⚠️ 代价是"装失败时旧的也没了" ✓ —— 刻意取舍 ✓，
        #    与出图那条路的口径一致 ✓（不静默降级 ✗）。
        #: ⚠️ **先核再丢旧模型** ✓✗：`text_weights`（真文本塔 ✓）离了词表就装得上、编不出 ✓
        #: —— 这种「请求本身不成立」要在 `_reset_loaded()` **之前**拒 ✓，否则白丢一台已装好的 ✓
        #: （重装 17.94 GiB ≈ 20 秒 ✓，而调用方只是漏了个参数 ✓）。
        if str(text_weights or "").strip() and not str(tokenizer_path or "").strip():
            raise TorchBackendUnavailable(
                "给了 `text_weights`（真 H3 文本塔 ✓）却**没给词表** ✗ ⇒ 装得上、编不出 ✓："
                "`<|image_pad|>` 这类特殊 token 的 id 只有 Qwen3-VL 的 tokenizer 里才有 ✓"
                "（拿别族的词表 ⇒ id 全错 ✓、形状全对 ✓✗）。请一并给 `tokenizer_path` ✓",
                reason="pending")
        had_previous = self._reset_loaded()   # 三格一起清 ✓（引用断在函数返回时 ⇒ 才真能回收 ✓✗）
        _release_torch_cache()

        backend = TorchBackend(device=device, audio_latent_mode=audio_latent_mode,
                               int8_offload=int8_offload,
                               residency=residency,
                               residency_budget_mib=residency_budget_mib,
                               policy=policy)
        report: dict[str, Any] = {"stage": stage, "reused": False, "components": {},
                                  "replaced": had_previous, "synthetic": True}
        try:
            report["weights"] = backend.load_weights(path=dit_path, loras=loras)
            report["components"]["dit"] = (report["weights"] or {}).get("path") or True
            # ⚠️ 参考 VAE 的尺寸**必须从已装主干现推** ✗（`attach_vae()` 不给配置 ⇒ 拿默认那套
            #    `channel_multipliers=(1,2,2)` ⇒ `spatial_scale=4` ✗ ≠ H3 的 16 ✓ ⇒ **当场被拒** ✓✗）
            trunk = (backend.describe().get("config") or {})
            if str(text_weights or "").strip():
                # ⭐⭐ **第四条装配线：真文本塔** ✓✗（2026-11-01 ✓）—— Qwen3-VL-32B 双塔 ✓
                #    （语言塔 `model.*` ＋ 视觉塔 `visual.*` 同文件 ✓，`attach_h3_text` ✓）。
                #
                # ⚠️ 为什么**不在装配期**把它装上来 ✗✗：TE 是 **25.28 GiB**（int8 常驻 22.7 ✓）、
                #    DiT 是 **17.94 GiB**（int8 ✓）—— 两件之和 43.2 GiB ✓✗，而卡只有 22.49 GiB ✓
                #    ⇒ 同驻**必然 OOM** ✓，且 OOM 落在**后装的那件**上 ✓（读起来像"TE 坏了" ✗）。
                #    好在这三段本来就不交叠 ✓（`encode` 只用 TE ✓、`sample` 只用 DiT ✓）⇒
                #    **按阶段互斥换装** ✓，沿用 Qwen-Image 2.1 的那套机制 ✓（`declare_chain` ＋
                #    `enter_phase` ✓），但驻留账必须换一份 ✗✗（见 `H3_PART_NEED_GIB` ✓）。
                report["chain"] = backend.declare_chain(
                    dit_path=dit_path, text_weights=str(text_weights),
                    tokenizer_path=tokenizer_path,
                    #: ``vaePath`` **留空** ✓：H3 的真 VAE 在下面按 `real_video_vae` 挂 ✓，且
                    #: 这条链的 `phase_parts` 里**没有 decode** ✓ ⇒ `enter_phase("decode")` 不换装 ✓。
                    vae_path="", vae_family="h3", text_kind="h3-te",
                    #: ⚠️ **`decode` 要单列** ✓✗（2026-11-01 补 ✓）：它这一段要用视频 VAE ✓ ——
                    #: 列进来才会「把 DiT 退掉 ✓、把常驻的 VAE 搬上卡 ✓」（见 `_relocate_keep_parts` ✓）；
                    #: 不列的话 `enter_phase("decode")` 直接返回 ``None`` ✓ ⇒ DiT 仍占着 19.62 GiB ✓、
                    #: VAE 留在 cpu ✓ ⇒ 整段解码在 cpu 上跑 ✓（实测 320×320 × 34 帧 **94.6 秒** ✓✗）。
                    #: ⚠️ 列进来**不会**重读 5.2 GiB ✓：VAE 是 `keep_parts` ✓，`present` 命中就只搬不装 ✓。
                    phase_parts={"encode": ("text",), "sample": ("dit",),
                                 "decode": ("vae",)},
                    #: ⭐ **VAE 是常驻件** ✓✗（2026-11-01 端到端逼出来的 ✓）：它在装配期由
                    #: `_attach_video_vae` 挂到 **cpu** ✓（5.2 GiB ✓ 不占卡 ✓）⇒ 换装翻不到它 ✓、
                    #: 也不该被"这一阶段用不着"退掉 ✓✗。⚠️ 千万别把它塞进 `phase_parts` ✗✗ ——
                    #: 一进 `decide_for` 的账，`encode` 阶段就会凭空多要 5.2 GiB ⇒ 判"装不下"
                    #: ⇒ 整座 25.44 GiB 的 TE 被推去 CPU ✓✗（症状＝编码慢十倍、卡空着 ✓）。
                    keep_parts=("vae",),
                    part_need_gib=H3_PART_NEED_GIB)
                report["components"]["textEncoder"] = {
                    "kind": "h3-te", "path": str(text_weights), "phased": True,
                    "note": ("真文本塔**不在装配期装** ✓：它 25.28 GiB、DiT 17.94 GiB 同驻装不下 ✗ "
                             "⇒ `encode` 阶段由 `enter_phase(\"encode\")` 现装 ✓（形状与键都在那时核 ✓）")}
                report["synthetic"] = False if real_video_vae else True
            elif tokenizer_path:
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
        except BaseException as err:  # noqa: BLE001
            # ⚠️ **不半装** ✗：失败 ⇒ 丢掉整个候选后端 ✓（模块一个字不变 ✓）
            _abandon_assembly(backend, err)

        described = backend.describe()
        report["device"] = described.get("device") or device
        report["backend"] = described.get("name")
        #: ⚠️ 与 `TorchBackend.write` 同一条诚实口径 ✓：文件是真的 ✓，画面来自哪一族要说清 ✓
        vae_kind = str(((report.get("components") or {}).get("vae") or {}).get("kind") or "")
        report["vaeKind"] = vae_kind or None
        if vae_kind == "h3-real":
            if str(text_weights or "").strip():
                report["note"] = (
                    "DiT 主权重**是真的** ✓、TE = **真 Qwen3-VL-32B 双塔** ✓（`encode` 阶段现装 ✓）、"
                    "视频 VAE = **H3 真权重** ✓ —— 三件全真 ✓；唯一未训练的只剩**音频 VAE** ✗ "
                    "（音频潜帧的来源仍无色 ✓）")
            else:
                report["note"] = ("DiT 主权重**是真的** ✓；视频 VAE = **H3 真权重** ✓；"
                                  "但 TE / 音频 VAE **未训练** ✗ ⇒ 解码链路是真的 ✓、潜变量来源仍是噪声 ✗ "
                                  "⇒ `synthetic` 保持 True ✓（区别：换成真 TE 就能直接出片 ✓，"
                                  "不必再换 VAE ✓；现在给 `textWeights` 就走第四条装配线 ✓）")
        elif str(text_weights or "").strip():
            report["note"] = ("DiT 主权重**是真的** ✓、TE = **真 Qwen3-VL-32B 双塔** ✓（`encode` 阶段现装 ✓）、"
                              "但**视频 VAE 仍是参考族** ✗（未训练 ✓）⇒ 解码链路是假的 ✓ "
                              "⇒ `synthetic` 保持 True ✓（补 `realVideoVae` 即三件全真 ✓）")
        else:
            report["note"] = ("DiT 主权重**是真的** ✓；参考 TE / VAE **未训练** ✗ ⇒ "
                              "产物是真文件 + 噪声画面 ✓（`synthetic` 保持 True ✓）")
        #: ⚠️ 装完**还一次缓存** ✗（G 级权重是一件件塞进去的 ⇒ 缓存被碎片顶起来 ✓✗，见
        #: :func:`_reclaim_cuda_cache` 的实测 ✓）—— 只还空着的块 ✓，在用的张量一个不动 ✓。
        report["cache"] = _reclaim_cuda_cache()
        self._commit_load(key, backend, report)   # 三格一起提交 ✓（少一格 ⇒ 白装或回执空 ✓✗）
        return report

    def _ensure_image_loaded(self, *, key: tuple[Any, ...], dit_path: str | None,
                             tokenizer_path: str | None,
                             device: str | None,
                             policy: str | None = None,
                             loras: Any = None) -> dict[str, Any]:
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
        had_previous = self._reset_loaded()   # ⚠️ 原来这里只清 `_backend` ✓✗ ⇒ 报告会停在旧模型上
        _release_torch_cache()

        #: ⭐ **出图这条线也走设备策略** ✓✗（2026-10-31 ✓）：SDXL 四件套装完真占 **6.73 GiB** ✓，
        #: 卡被别人占着时老路照样硬上 ⇒ `CUDA out of memory` ✓✗。细节见
        #: :func:`_image_device_decision` ✓（需求取盘上真字节 ✓、`split` 落不了地 ⇒ 转 CPU ✓）。
        decision = _image_device_decision(dit_path=dit_path, device=device, policy=policy)
        if decision is not None:
            device = decision["device"]

        backend = SdxlBackend(device=device)
        report: dict[str, Any] = {"stage": pipe.IMAGE_STAGE, "reused": False,
                                  "components": {}, "replaced": had_previous,
                                  "synthetic": True}
        try:
            #: ⭐ LoRA 从 ``ensure_loaded(loras=…)`` 一路透到这里 ✓（2026-10-31 补 ✓）——
            #: 挂不全 ``load_weights`` 自己会炸 ✓（不会留下「四件齐、档不对」的模型 ✓✗）。
            report["weights"] = backend.load_weights(
                path=dit_path, tokenizer=load_tokenizer(tokenizer_path), loras=loras)
            # 报告里只留**事实** ✓（四件各自的张量数 ✓ —— 与事实表逐个可比 ✓）
            for name, component in (report["weights"].get("components") or {}).items():
                report["components"][name] = dict(component).get("keys") or True
        except BaseException as err:  # noqa: BLE001
            _abandon_assembly(backend, err, what="SDXL")

        described = backend.describe()
        report["device"] = described.get("device") or device or "cpu"
        report["backend"] = described.get("name")
        report["synthetic"] = bool(described.get("synthetic"))
        report["facts"] = described.get("facts")
        #: ⭐ 装配期那次**设备决策** ✓✗（`auto` 下可能就是「卡上只剩 0.4 GiB ⇒ 转 CPU」✓）——
        #: 只报 `device` 会让人以为「这台机器只能 CPU」✓✗（与另两条装配线同一口径 ✓）。
        report["devicePolicy"] = decision
        #: ⚠️ 与 `SdxlBackend.synthetic` 同一条诚实口径 ✓：**真权重**装齐 ⇒ False ✓
        report["note"] = ("SDXL 四件套（UNet / VAE / CLIP-L / CLIP-G）**全是真权重** ✓ ⇒ "
                          "`synthetic` 如实为 **False** ✓"
                          "（与视频那条路的参考 TE/VAE 未训练 ⇒ 恒 True 不同 ✓）")
        #: ⚠️ 这条**尤其**要还 ✗：SDXL 是 1680+248+197+390 件小张量 ✗ ⇒ 装机就把缓存顶到
        #: **13.88 GiB（真占用只有 6.73 GiB）** ✓✗ ⇒ 不还就等于**白白占掉半张卡** ✗。
        report["cache"] = _reclaim_cuda_cache()
        self._commit_load(key, backend, report)   # 三格一起提交 ✓（少一格 ⇒ 白装或回执空 ✓✗）
        return report

    def _ensure_zimage_loaded(self, *, key: tuple[Any, ...], dit_path: str | None,
                              tokenizer_path: str | None, text_weights: str | None,
                              image_vae_weights: str | None,
                              device: str | None,
                              policy: str | None = None,
                              loras: Any = None) -> dict[str, Any]:
        """装 **Z-Image 出图链** ✓（``stage=zimage`` ⇒ :class:`~.torch_backend.TorchBackend` ✓）。

        ## 为什么是**单独一条线** ✗（不塞进 H3 / SDXL ✓）

        三个组件**不是同一份权重** ✓✗：DiT（`z_image_turbo_bf16` ✓ **453** 键 ✓）、
        Qwen3-4B 文本塔（`qwen_3_4b` ✓ **398** 键 ✓）、Flux VAE（`ae` ✓ **244** 键 ✓）。
        H3 那条线装的是「H3 DiT + **参考** TE/VAE（未训练 ✓）」⇒ 拿它装 Z-Image 会得到
        「形状顺、跑得动、图是噪声」✓✗（正是本仓最忌的那类 ✓）；SDXL 那条线是四件套切分 ✓，
        更不是一回事 ✓。

        ## 缺格 ⇒ **当场拒** ✗（不猜 ✓）

        四格缺任一：``ditPath`` / ``tokenizerPath`` / ``textWeights`` / ``imageVaeWeights`` ✓。
        ⚠️ **不猜路径** ✗✗：盘上同时有 `qwen_3_vl_32b`（不是它 ✓）、`qwen_image_vae`（不是它 ✓）
        —— 猜错的后果不是装不上 ✓，而是**装上了、出噪声** ✓✗，那比装不上难查得多 ✓。

        ## 与 :meth:`_ensure_image_loaded` 同一条纪律

        1. **先放旧的** ✓（GB 级模型不许两套同时在内存里 ✓：DiT 11.46 + TE 7.49 + VAE 0.33
           ≈ 19.3 GiB ✓，卡上 22.49 可用 ⇒ **刚好装得下** ✓，再多一套必 OOM ✗）；
        2. **不半装** ✗：任一步失败 ⇒ 丢掉整个候选后端 ✓（模块一个字不变 ✓）；
        3. **报告说真话** ✓：``synthetic=False`` ✓（三份都是真权重 ✓ ⇒ 出来的是真图 ✓）。
        """
        self._require_slots("Z-Image", _ZIMAGE_STAGE_NOTE, [
            (dit_path, "`ditPath`（DiT 主干 ✓ `unet/z_image_turbo_bf16.safetensors` ✓）"),
            (text_weights, "`textWeights`（Qwen3-4B 文本塔 ✓ `text_encoders/qwen_3_4b.safetensors` ✓）"),
            (image_vae_weights, "`imageVaeWeights`（Flux VAE ✓ `vae/ae.safetensors` ✓）"),
            (tokenizer_path, "`tokenizerPath`（Qwen3 词表**目录** ✓ 含 `tokenizer.json` ✓）"),
        ])

        # ⚠️ 装新的之前**先把旧的放掉** ✗✗（同 H3 / SDXL 两条线 ✓）：三份权重合计 ≈19.3 GiB ✓，
        #    旧的活着就装新的 ⇒ 峰值翻倍 ⇒ 96 GB 内存 + 22.49 GiB 卡**必炸** ✓✗。
        had_previous = self._reset_loaded()   # 三格一起清 ✓（引用断在函数返回时 ⇒ 才真能回收 ✓✗）
        _release_torch_cache()

        #: ⭐ 装配前按**整链同驻**的实测账判一次 ✓✗（19.28 GiB ✓，见
        #: :func:`_zimage_device_decision` ✓）：构造期那次 `decide_for(())` 的 need 是"**未知**" ✓
        #: ⇒ 只能按占用率粗判 ✓，卡上剩 12 GiB（占用率 47%）照样判"能上卡" ✓✗ ⇒ 装 TE 时 OOM ✓。
        #: ⚠️ 必须落在 `TorchBackend(...)` **构造之前** ✗✗：设备决定精度（cuda⇒bf16 / cpu⇒fp32 ✓），
        #: 构造完再改就是"按 bf16 建的模块搬到 CPU 上跑"✓✗（CPU 上 conv 未实现 ✓）。
        decision = _zimage_device_decision(device=device, policy=policy)
        if decision is not None:
            device = decision["device"]
        backend = TorchBackend(device=device, policy=policy)
        report: dict[str, Any] = {"stage": pipe.ZIMAGE_STAGE, "reused": False,
                                  "components": {}, "replaced": had_previous,
                                  "synthetic": False}
        try:
            # ① DiT 主干 ✓（`load_weights` 自己按**键名**判族 ✓ ⇒ 不是 Z-Image 的形态会走别的族 ✓）
            #: ⭐ LoRA 必须**往下传** ✗✗（2026-10-31 修 ✓）：这条线原先不带 ``loras`` ✓，调用方送
            #: 了也**静默丢掉** ✓✗ —— 装出来是「没挂 LoRA 的底模」✓，跑得动、出得来、不是那个档 ✓✗。
            report["weights"] = backend.load_weights(path=dit_path, loras=loras)
            report["components"]["dit"] = (report["weights"] or {}).get("path") or True
            #: ⚠️ **必须显式核一次族** ✗✗：`load_weights` 对判不出的族会走 `dit_generic` ✓ ——
            #: 拿 Krea2 的权重配上 Qwen3 TE + Flux VAE 也能"装得进去"✓✗（TE 的 `output_dim`
            #: 与它要的 `text_dim` 恰好也可能对得上 ✓），出来是**噪声图**且一声不响 ✓✗。
            family = str(((backend.describe().get("config") or {}) or {}).get("family") or "")
            if family != ZIMAGE_FAMILY:
                raise TorchBackendUnavailable(
                    f"`ditPath` 这份权重判出来是 **{family or '未知'}** 族 ✗，不是 Z-Image（`{ZIMAGE_FAMILY}` ✓）"
                    f"\n  · 给的：{dit_path} ✓\n"
                    f"  · Z-Image 的 DiT 长这样：`unet/z_image_turbo_bf16.safetensors` ✓（453 键 ✓）"
                    f"\n  ⚠️ 这里**不将就** ✗：配错族的权重「装得进去、跑得出来、图是噪声」✓✗",
                    reason="pending")
            # ② Qwen3-4B 文本塔 ✓（`_check_text_encoder` 会拿主干 `text_dim` 互核 ✓ 2560 ✓）
            report["components"]["textEncoder"] = backend.attach_qwen3_te(
                str(text_weights), tokenizer_path=str(tokenizer_path))
            # ③ Flux VAE ✓（`_check_attached_vaes` 会核 16 通道 ↔ `in_channels` ✓、8 倍 ↔ `vae_scale` ✓）
            report["components"]["vae"] = backend.attach_flux_vae(str(image_vae_weights))
        except BaseException as err:  # noqa: BLE001
            _abandon_assembly(backend, err, what="Z-Image")

        described = backend.describe()
        report["device"] = described.get("device") or device or "cpu"
        report["backend"] = described.get("name")
        #: ⭐ 装配期那一次**设备决策** ✓✗（`auto` 下可能就是"因为卡上只剩 0.2 GiB ⇒ 转 CPU" ✓）——
        #: 只报 `device` 会让人以为「这台机器只能 CPU」✓✗（2026-10-31 加策略 ✓）。
        report["devicePolicy"] = described.get("devicePolicy")
        #: ⭐ 装配前那次**整链**决策 ✓✗（与上面那格是两个问题 ✓）：上面答「这台后端现在什么策略、
        #: 逐阶段怎么判的」✓，这格答「**装这条链之前**按 19.28 GiB 总账判成了什么」✓。
        #: 只有前者时，「为什么这次整条链在 CPU 上」答不出来 ✓✗（后端是被传进 `device="cpu"` 的 ✓）。
        report["chainDevicePolicy"] = decision
        report["vaeKind"] = str((report["components"].get("vae") or {}).get("kind") or "") or None
        #: ⚠️ 与 H3 那条线**刻意不同** ✓：这边 TE 与 VAE 都要求**真权重** ✓（缺格就拒 ✗，
        #: 没有"挂桩出图"这条路 ✓）⇒ ``synthetic`` 如实为 **False** ✓。
        report["note"] = ("Z-Image 三份权重**全是真的** ✓（DiT 453 键 ✓ / Qwen3-4B 398 键 ✓ / "
                          "Flux VAE 244 键 ✓）⇒ 出来的是**真图** ✓，`synthetic` 如实为 False ✓。"
                          "装齐之后出图直接走 `pipeline.run_sync(stage=\"zimage\")` ✓"
                          "（2026-10-31 实测：装配 40.5 秒 ✓、1024² / 8 步 13.1 秒 ✓、"
                          "与手动 `attach_*` 那条路的产物**逐字节相同** ✓）。"
                          "⚠️ 卡上占用 ≈19.39 GiB / 22.49 ✓（余 1.77 GiB ✓）—— 这个余量只够 1024² 那档 ✓，"
                          "再往上加分辨率要掂量 ✓。")
        report["cache"] = _reclaim_cuda_cache()
        self._commit_load(key, backend, report)   # 三格一起提交 ✓（少一格 ⇒ 白装或回执空 ✓✗）
        return report

    def _ensure_qwen21_loaded(self, *, key: tuple[Any, ...], dit_path: str | None,
                              tokenizer_path: str | None, text_weights: str | None,
                              image_vae_weights: str | None,
                              device: str | None,
                              policy: str | None = None,
                              loras: Any = None) -> dict[str, Any]:
        """装 **Qwen-Image 2.1 出图链** ✓（``stage=qwen_image21`` ⇒ :class:`~.torch_backend.TorchBackend` ✓）。

        ## 为什么是**第四条线** ✗（不塞进 Z-Image / H3 / SDXL ✓）

        三件组件没有一件能与前三条线共用 ✓✗：

        * DiT —— 2.1 的主干（**649** 键 ✓、`in_channels=64` ✓、`img_in` 是 ``Linear`` **没有
          patchify** ✗、`vae_scale=16` ✓，见 `dit_qwen21` 模块头 ✓）；
        * 文本塔 —— **Qwen3.5-9B**（24 层 Gated DeltaNet ＋ 8 层 full attention ✓、
          `text_dim=4096` ✓、取 `layer_idx=-2` ✓），Z-Image 那座是 Qwen3-4B（`text_dim=2560` ✓）✗；
        * VAE —— **64 通道**的 Wan 式 3D 因果 VAE ✓（238 键 ✓），Flux 的 ae 是 16 通道 ✗。

        ⚠️ 任何一对互换都在"不报错"那一类里 ✓✗：文本塔换错 ⇒ `_check_text_encoder` 能拦 ✓（宽度对不上 ✓）；
        VAE 换错 ⇒ 通道数一核就拦 ✓；**同一族不同代**的 DiT（2.1 与 2512）才是真难分的那对 ✓✗ ——
        2.1 只有 **649** 键（2512 是 1933 ✓）⇒ 键集能区分 ✓，所以这里**显式核族** ✓。

        ## 缺格 ⇒ **当场拒** ✗（不猜 ✓）

        四格缺任一：``ditPath`` / ``tokenizerPath`` / ``textWeights`` / ``imageVaeWeights`` ✓。

        ## 与 :meth:`_ensure_zimage_loaded` 同一条纪律

        1. **先放旧的** ✓（GB 级模型不许两套同在内存 ✓）；
        2. **不半装** ✗（任一步失败 ⇒ 丢掉整个候选后端 ✓，模块一个字不变 ✓）；
        3. **报告说真话** ✓（``synthetic=False`` ✓、``vaeKind`` 如实 ✓ —— 分时驻留下它是 ``None`` ✓）。

        ## ⚠️ 这里**只装 DiT** ✓（与 Z-Image 那条不同 ✗✗）

        三件**反量化到 bf16 之后**的驻留量是 DiT **13.26 GiB** ＋ TE **14.79 GiB** ＋ VAE
        **0.63 GiB** ＝ **28.68 GiB** ✓，而这张卡可用 **22.49 GiB** ✓（连 TE+DiT 的 28.05 都过不去 ✗）。
        ⚠️ 盘上那三个文件是 **int8** ✓（6.76 / 8.82 / 0.63 GiB ✓）—— 拿它们相加得 ≈16.2 GiB ✓
        是**装进来的字节数** ✗✗，**不是驻留量** ✓（2 倍关系 ✓）。按 16.2 判"装得下"就是这条线
        最容易犯的错 ✓✗（OOM 会落在最后一件 ⇒ 报错指向 VAE ✓，看着像"VAE 坏了" ✓）。
        ⇒ 本装配线只装 DiT ＋ :meth:`TorchBackend.declare_chain` 登记另外两件 ✓，管线在
        encode / decode 边界换装 ✓（峰值 ≈14.8 GiB ✓）。
        """
        self._require_slots("Qwen-Image 2.1", _QWEN21_STAGE_NOTE, [
            (dit_path, "`ditPath`（DiT 主干 ✓ `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` ✓）"),
            (text_weights, "`textWeights`（Qwen3.5-9B 文本塔 ✓ "
                           "`text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors` ✓）"),
            (image_vae_weights, "`imageVaeWeights`（64 通道 VAE ✓ `vae/qwen_image_2.1_vae_bf16.safetensors` ✓）"),
            (tokenizer_path, "`tokenizerPath`（Qwen3.5 词表**目录** ✓ 含 `tokenizer.json` ✓）"),
        ])

        # ⚠️ 装新的之前**先把旧的放掉** ✗✗（同前三条线 ✓）：光 DiT 一件就 13.26 GiB ✓，
        #    旧的活着就装新的 ⇒ 峰值翻倍 ⇒ 22.49 GiB 卡**必炸** ✓✗。
        had_previous = self._reset_loaded()   # 三格一起清 ✓（引用断在函数返回时 ⇒ 才真能回收 ✓✗）
        _release_torch_cache()

        #: ⭐ 策略随装配下去 ✓✗：`policy="auto"`（默认）⇒ 装配这件时**现测**卡况 ✓，
        #: 卡上没地方就自动转 CPU / 联合 ✓（老路只看"有没有卡" ⇒ 卡满也硬上 ⇒ OOM ✗）。
        backend = TorchBackend(device=device, policy=policy)
        #: ⭐ 装配这件之前再按**实测账**判一次 ✓✗（`PART_NEED_GIB["dit"] = 13.26` ✓）：构造期那次
        #: 的 need 是"**未知**" ✓ ⇒ 只能按占用率粗判 ✓，7 成占用照样敢上卡 ✓✗ ⇒ 装到一半 OOM ✓。
        #: 判成 `split` ⇒ 权重按层切给卡＋内存 ✓；判成 `cpu` ⇒ 整件在内存 ✓（慢，但不失败 ✓）。
        #: ⚠️ 只对**有实测账**的族这么做 ✗（目前只有 2.1 三件 ✓）—— 别的族没有账，编一个数字进去
        #: 就等于"看着精确、其实是猜" ✓✗（见 `PART_NEED_GIB` 注释 ✓）。
        backend._apply_decision(backend.decide_for(("dit",), phase="assemble"))  # noqa: SLF001
        report: dict[str, Any] = {"stage": pipe.QWEN21_STAGE, "reused": False,
                                  "components": {}, "replaced": had_previous,
                                  "synthetic": False}
        try:
            # ① DiT 主干 ✓（`load_weights` 按键名判族 ✓）
            #: ⭐ LoRA 必须**往下传** ✗✗（2026-10-31 修 ✓）：原先不带 ``loras`` ⇒ 调用方送了
            #: 也静默丢掉 ✓✗（装出来是没挂 LoRA 的底模 ✓，跑得动但不是那个档 ✓✗）。
            report["weights"] = backend.load_weights(path=dit_path, loras=loras)
            report["components"]["dit"] = (report["weights"] or {}).get("path") or True
            #: ⚠️ **必须显式核族** ✗✗：`load_weights` 判不出就走 `dit_generic` ✓ ——
            #: 把 2512 的 1933 键（同族不同代 ✓）配 Qwen3.5 TE + 64 通道 VAE 也"装得进去" ✓✗：
            #: `text_dim` 都是 4096 ✓、VAE 通道也都是 64 ✓ ⇒ 三道互核**全部通过** ✓✗，
            #: 出来是「结构错代」的图 ✓ —— 只有族名这一处能说话 ✓。
            family = str(((backend.describe().get("config") or {}) or {}).get("family") or "")
            if family != QWEN21_FAMILY:
                raise TorchBackendUnavailable(
                    f"`ditPath` 这份权重判出来是 **{family or '未知'}** 族 ✗，不是 Qwen-Image 2.1"
                    f"（`{QWEN21_FAMILY}` ✓）\n  · 给的：{dit_path} ✓\n"
                    f"  · 2.1 的 DiT 长这样：`diffusion_models/qwen_image_2.1_int8_convrot.safetensors` ✓"
                    f"（**649** 键 ✓；2512 是 1933 键 ✗ —— 两代 `text_dim` / VAE 通道都一样 ✓✗，"
                    f"只有键集与族名能分 ✓）", reason="pending")
            # ② 三件**不同时驻留** ✗✗：只登记链 ✓，由 `pipeline` 在 encode/sample/decode
            #    三个边界上调 `enter_phase` 换装 ✓（账见 `TorchBackend.PHASE_PARTS` ✓）。
            #    ⚠️ 这里**不再** `attach_qwen35_te` / `attach_wan_vae` ✗✗：三件 bf16 驻留
            #    合计 **28.68 GiB** ✓（DiT 13.26 ＋ TE 14.79 ＋ VAE 0.63 ✓）> 卡上 22.49 ✓
            #    ⇒ 「一次装完」**必然 OOM** ✓✗，而且 OOM 落在最后一件 ⇒ 报错指向 VAE ✓✗。
            report["components"]["textEncoder"] = {"phased": True, "loaded": False,
                                                   "phase": "encode",
                                                   "path": str(text_weights),
                                                   "tokenizer": str(tokenizer_path)}
            report["components"]["vae"] = {"phased": True, "loaded": False,
                                           "phase": "decode",
                                           "path": str(image_vae_weights)}
            report["chain"] = backend.declare_chain(
                dit_path=str(dit_path), text_weights=str(text_weights),
                tokenizer_path=str(tokenizer_path), vae_path=str(image_vae_weights),
                vae_family=QWEN21_FAMILY)
        except BaseException as err:  # noqa: BLE001
            _abandon_assembly(backend, err, what="Qwen-Image 2.1")

        described = backend.describe()
        report["device"] = described.get("device") or device or "cpu"
        report["backend"] = described.get("name")
        #: ⭐ 装配期决策 ✓（同 zimage 线 ✓）：`auto` 下"卡满 ⇒ 转 CPU"必须看得见 ✓✗。
        report["devicePolicy"] = described.get("devicePolicy")
        #: ⚠️ 此时 VAE **还没装** ✓（分时驻留 ✓）⇒ ``vaeKind`` 如实是 ``None`` ✓✗
        #: —— 不许从"登记了链"推"VAE 已就位" ✓✗（那时解码还没发生 ✓）。
        report["vaeKind"] = None
        #: ⚠️ 「在场没在场」只认后端自己的自述 ✓✗ —— 2026-10-31 实测踩过：这里原本写
        #: ``components[name] is not None`` ✓，而 ``components["textEncoder"]`` 是一个**登记
        #: 用的 dict** ✓（有 path ✓）⇒ 三件全报 true ✓✗（看着像"分时驻留没生效"✓，其实只是
        #: 判据读错了对象 ✓）。真正的答案是 ``describe()`` 的 ``*Loaded`` 三格 ✓。
        report["phasedComponents"] = {"dit": bool(described.get("modelLoaded")),
                                      "textEncoder": bool(described.get("textEncoderLoaded")),
                                      "vae": bool(described.get("vaeLoaded"))}
        report["note"] = (
            "Qwen-Image 2.1 三份权重**全是真的** ✓（DiT 649 键 ✓ / Qwen3.5-9B 语言塔 ✓ / "
            "64 通道 VAE 238 键 ✓）⇒ 出来的是**真图** ✓，`synthetic` 如实为 False ✓。"
            "出图走 `pipeline.run_sync(stage=\"qwen_image21\")` ✓；"
            "⚠️ 采样 shift 是 **0.69** ✓（`pipeline.QWEN21_FLOW_SHIFT` ✓，**不是** Z-Image 的 3 ✓✗）、"
            "**没有** patchify ⇒ 尺寸只吸附到 16 ✓✗。"
            "⭐ **分时驻留** ✓：三件 bf16 驻留合计 **28.68 GiB** ✗（DiT 13.26 ＋ TE 14.79 ＋ "
            "VAE 0.63 ✓）装不进 22.49 GiB 的卡 ✓✗ ⇒ 现在**只装了 DiT** ✓，TE 与 VAE 由管线在 "
            "encode / decode 边界换装 ✓（`TorchBackend.enter_phase` ✓）。"
            "峰值 ≈**14.8 GiB** ✓（余 7.7 ✓）。"
            "⚠️ 代价：**每次出图多两段装配** ✓（TE 14.79 GiB ＋ VAE 0.63 GiB 读盘 ✓）"
            "—— 这是装不下的必付成本 ✓，不是可省的优化 ✗。"
        )
        report["cache"] = _reclaim_cuda_cache()
        self._commit_load(key, backend, report)   # 三格一起提交 ✓（少一格 ⇒ 白装或回执空 ✓✗）
        return report

    def _ensure_qwen_image_loaded(self, *, key: tuple[Any, ...], dit_path: str | None,
                                 tokenizer_path: str | None, text_weights: str | None,
                                 image_vae_weights: str | None,
                                 device: str | None,
                                 policy: str | None = None,
                                 loras: Any = None) -> dict[str, Any]:
        """装 **Qwen-Image 2512 / Edit 2511 出图链** ✓（``stage=qwen_image`` ⇒
        :class:`~.torch_backend.TorchBackend` ✓）。

        ## 为什么是**第五条线** ✗（不塞进 2.1 那条 ✓）

        逐件列一遍"能不能共用"（**全不能** ✗✗，见 :data:`_QWEN_IMAGE_STAGE_NOTE` ✓）：

        | | 2.1（2509） ✓ | **2512 / Edit** ✓ |
        |---|---|---|
        | DiT | 649 键 / 单流 / 64ch ✓ | **1933 键**（Edit 3614 ✓）/ **双流** / **16ch** ✓ |
        | 文本塔 | Qwen3.5-9B / **4096** / 取 −2 ✓ | **Qwen2.5-VL-7B / 3584 / 取 −1** ✓ |
        | VAE | 64 通道（238 键 ✓） | **16 通道**（`qwen_image_vae` ✓） |
        | 时间口径 | lumina 式翻转 ✓ | **`1000·σ` 不翻转** ✓ |
        | flow shift | 0.69 ✓ | **3.1** ✓ |

        ⚠️ 词表也要换 ✓（Qwen2.5 的 `<|im_start|>=151644` ✓ / vocab 152064 ✓ —— 与 Qwen3.5 不同 ✗）。

        ## 分时驻留（**这一族更狠** ✗✗）

        | 部件 | 驻留 | 出处 |
        |---|---|---|
        | DiT | **38.05 GiB** ✓ | 盘上 `qwen_image_2512_bf16` 本来就是 bf16 ⇒ 1:1 ✓ |
        | 文本塔 | ≈**16.1 GiB** ✓ | `qwen_2.5_vl_7b_fp8_scaled` 8.74 GiB × 2 − 视觉塔 ✓ |
        | VAE | **0.24 GiB** ✓ | 本体即最终精度 ✓ |

        合计 ≈**54.4 GiB**（卡 22.49 ✓、内存 96 GB ✓）⇒ **三件绝无可能同驻** ✓✗，而且
        **DiT 单件就超卡** ✓✗（38.05 > 22.49）⇒ `auto` 必然判 `split`（按层切 ✓，跨界前向慢 ✓）
        或 `cpu`（整件在内存 ✓，更慢 ✓）。⚠️ **这是必付成本** ✗，报告里如实写 ✓，
        不许写成"已优化" ✗✗。

        ⚠️ **Edit 2511 的参考图通路还没接** ✗✗（视觉塔 + 参考图 VAE 编码 + `ref_latents` ✓）：
        这里**照装**（键对账与形状现在就能核 ✓），但 :meth:`TorchBackend.denoise` 会在
        Edit 权重上**明确报错** ✓✗ —— 静默按 t2i 出图＝编辑能力凭空消失 ✓✗。
        """
        self._require_slots("Qwen-Image 2512", _QWEN_IMAGE_STAGE_NOTE, [
            (dit_path, "`ditPath`（DiT 主干 ✓ `diffusion_models/qwen_image_2512_bf16.safetensors` ✓"
                       " / Edit `qwen_image_edit_2511_int8_convrot.safetensors` ✓）"),
            (text_weights, "`textWeights`（**Qwen2.5-VL-7B** 文本塔 ✓ "
                           "`text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` ✓）"),
            (image_vae_weights, "`imageVaeWeights`（**16 通道** VAE ✓ `vae/qwen_image_vae.safetensors` ✓）"),
            (tokenizer_path, "`tokenizerPath`（Qwen2.5 词表**目录** ✓ 含 `tokenizer.json` ✓）"),
        ])

        had_previous = self._reset_loaded()   # 三格一起清 ✓（引用断在函数返回时 ⇒ 才真能回收 ✓✗）
        _release_torch_cache()

        backend = TorchBackend(device=device, policy=policy)
        #: ⭐ 按**本族**的账判一次 ✓✗（`QWEN_IMAGE_PART_NEED_GIB["dit"] = 38.05` ✓）：
        #: 拿 2.1 的 13.26 判 ⇒ "装得下" ✓✗、`residency` 留 `full` ✓、装到 20 GiB 才 OOM ✓。
        #: ⚠️ 账挂在 :class:`TorchBackend` 上（`QWEN_IMAGE_PART_NEED_GIB` ✓ 与 2.1 的
        #: `PART_NEED_GIB` 并列 ✓，两族**必须分开** ✗✗）—— 这里用**实例**取，不另抄一份 ✓。
        backend._apply_decision(backend.decide_for(  # noqa: SLF001
            ("dit",), phase="assemble", need_gib=backend.QWEN_IMAGE_PART_NEED_GIB))
        report: dict[str, Any] = {"stage": pipe.QWEN_IMAGE_STAGE, "reused": False,
                                  "components": {}, "replaced": had_previous,
                                  "synthetic": False}
        try:
            #: ⭐ LoRA 必须**往下传** ✗✗（2026-10-31 修 ✓）：原先不带 ``loras`` ⇒ 调用方送了
            #: 也静默丢掉 ✓✗（装出来是没挂 LoRA 的底模 ✓，跑得动但不是那个档 ✓✗）。
            report["weights"] = backend.load_weights(path=dit_path, loras=loras)
            report["components"]["dit"] = (report["weights"] or {}).get("path") or True
            #: ⚠️ **必须显式核族** ✗✗，而且这次是**双向**的：2.1 与 2512 的 `text_dim`
            #: 与 VAE 通道**都不一样** ✓（4096/64 vs 3584/16 ✓）⇒ 文本塔与 VAE 那一关也能拦 ✓；
            #: 但 **Edit 2511 与 2512 t2i 的配置完全相同** ✓✗（同族同代 ✓，只差参考图通路 ✓）
            #: ⇒ 族名拦不住这一对 ✓，只能靠 `ref_timestep_zero` 那一格区分 ✓（见下面的警告 ✓）。
            config = (backend.describe().get("config") or {}) or {}
            family = str(config.get("family") or "")
            if family != QWEN_IMAGE_FAMILY:
                raise TorchBackendUnavailable(
                    f"`ditPath` 这份权重判出来是 **{family or '未知'}** 族 ✗，不是 Qwen-Image 2512 / Edit 2511"
                    f"（`{QWEN_IMAGE_FAMILY}` ✓）\n  · 给的：{dit_path} ✓\n"
                    f"  · 2512 的 DiT 长这样：`diffusion_models/qwen_image_2512_bf16.safetensors` ✓"
                    f"（**1933** 键 ✓、bf16 ✓）；Edit 是 `qwen_image_edit_2511_int8_convrot.safetensors` ✓"
                    f"（3614 键 ✓）\n"
                    f"  · ⚠️ **不是** `qwen_image_2.1_int8_convrot.safetensors` ✗（那是 649 键的单流 2.1 ✓，"
                    f"走 `stage=\"qwen_image21\"` ✓）", reason="pending")
            #: ⚠️ Edit 权重**照装**但把事实写在报告里 ✓（`denoise` 会拒跑 ✓，见方法 docstring ✓）：
            #: 若这里静默放行、报告也不提，读者只会看到"装成功了、图不对" ✓✗。
            is_edit = bool(config.get("ref_timestep_zero"))
            report["components"]["textEncoder"] = {"phased": True, "loaded": False,
                                                   "phase": "encode",
                                                   "path": str(text_weights),
                                                   "tokenizer": str(tokenizer_path),
                                                   "kind": "qwen25-te"}
            report["components"]["vae"] = {"phased": True, "loaded": False,
                                           "phase": "decode",
                                           "path": str(image_vae_weights)}
            #: ⚠️ 三个按族参数**一个都不能省** ✗✗：
            #: ``text_kind="qwen25"`` ⇒ encode 时接 Qwen2.5 塔 ✓（缺省是 qwen35 ✗ ⇒ 报「缺 linear_attn」✓✗）；
            #: ``vae_family="qwen_image"`` ⇒ 反缩放取**恒等** ✓（`vae_wan.latents_stats` ✓）。
            #: ⚠️ 这个族名 **不能用 ``wan21``** ✗✗（2026-10-31 修）：2512 的 VAE 只是**结构**与 Wan21
            #: 同档（16ch / 8× ✓），**反缩放不同** ✓ —— Wan21 有 16 个真 mean/std（-0.7571… ✓），
            #: 而 2512 走 ComfyUI 的 ``LatentFormat`` 基类（恒等 ✓）。借名的话 ``wan21`` 一旦登真值，
            #: 2512 就吃到 Wan 的 mean/std ⇒ 图整体偏色、不报错 ✓✗。给 2.1 的 `qwen_image21` 也不行
            #: （64 通道 ⇒ 长度对不上当场报 ✓）。
            #: ``part_need_gib`` ⇒ 每阶段决策按 38.05 / 16.1 / 0.24 判 ✓（缺省是 2.1 的 13.26 ✗）。
            report["chain"] = backend.declare_chain(
                dit_path=str(dit_path), text_weights=str(text_weights),
                tokenizer_path=str(tokenizer_path), vae_path=str(image_vae_weights),
                vae_family="qwen_image", text_kind="qwen25",
                part_need_gib=backend.QWEN_IMAGE_PART_NEED_GIB)
        except BaseException as err:  # noqa: BLE001
            _abandon_assembly(backend, err, what="Qwen-Image 2512")

        described = backend.describe()
        report["device"] = described.get("device") or device or "cpu"
        report["backend"] = described.get("name")
        report["devicePolicy"] = described.get("devicePolicy")
        report["vaeKind"] = None      #: 分时驻留下 VAE 还没装 ✓（同 2.1 那条 ✓）
        report["phasedComponents"] = {"dit": bool(described.get("modelLoaded")),
                                      "textEncoder": bool(described.get("textEncoderLoaded")),
                                      "vae": bool(described.get("vaeLoaded"))}
        report["ditVariant"] = "edit-2511" if is_edit else "t2i-2512"
        report["warnings"] = ([] if not is_edit else [
            "⚠️ 装的是 **Edit 2511** 权重（带 `__index_timestep_zero__` 哨兵 ✓），而参考图通路"
            "（Qwen2.5-VL 视觉塔 + 参考图 VAE 编码 + `ref_latents` ✓）**尚未接** ✗ ⇒ 出图会在 "
            "`TorchBackend.denoise` 前**明确报错** ✓（不会静默按纯 t2i 画 ✓✗）。"
            "要用编辑能力：换 2512 的 t2i 权重，或先把视觉塔接上 ✓"
        ])
        report["note"] = (
            "Qwen-Image 2512 三份权重**全是真的** ✓（DiT 1933 键 ✓ / Qwen2.5-VL-7B 语言塔 28 层 3584 ✓ / "
            "16 通道 VAE ✓）⇒ `synthetic` 如实为 False ✓。"
            "出图走 `pipeline.run_sync(stage=\"qwen_image\")` ✓；"
            "⚠️ 采样 shift 是 **3.1** ✓（`pipeline.QWEN_IMAGE_FLOW_SHIFT` ✓，**不是** 2.1 的 0.69 ✗）、"
            "尺寸吸附 **16** ✓（VAE 8× × DiT patch 2 ✓）、时间口径 **`1000·σ` 不翻转** ✓✗。"
            "⭐ **分时驻留，且这一族更狠** ✓✗：DiT **38.05 GiB** ＋ TE ≈16.1 ＋ VAE 0.24 ＝ ≈54.4 GiB ✓，"
            "而**单是 DiT 就超卡**（22.49 ✓）⇒ `auto` 会判 `split`（按层切给卡＋内存 ✓）或 `cpu` ✓。"
            "**这是必付成本** ✗，不是可省的优化 ✗：逐层跨界前向会比 Z-Image 慢一个量级 ✓。"
        )
        report["cache"] = _reclaim_cuda_cache()
        self._commit_load(key, backend, report)   # 三格一起提交 ✓（少一格 ⇒ 白装或回执空 ✓✗）
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
