"""GPU 显存管理器 —— 移植 ``services/gpu-manager.ts``（473 行），云端/本地双模式下的显存调度。

核心职责（与原 TS 一致）：

1. 追踪当前 GPU 上加载了哪些本地模型及其**估算**显存；
2. 切模型前**主动卸载**上一个模型的显存（避免 OOM）；
3. **互斥锁 + FIFO 队列**确保同一时刻只有一个 GPU 重负载任务在跑；
4. 提供 GPU 状态监控（``nvidia-smi`` 实时数据 + 管理器内部快照）。

用法（对应 TS 的 ``withGpuLease``）::

    async with gpu_lease('image', provider, model, base_url):
        ... 执行图片生成 ...

本地服务卸载机制：

* **Ollama**：``POST /api/generate {keep_alive: 0}`` ⇒ 响应后立即卸载；
* **SD WebUI**：``POST /sdapi/v1/unload-checkpoint``（best-effort，部分版本支持）；
* **CosyVoice**：轻量（~4GB），**不强制卸载**，仅跟踪状态（``passive``）；
* **MiniMax H3 本地**：经 **ComfyUI** ``POST /free`` 卸载（权重实际驻留 ComfyUI 显存）。

⚠️ 五处保真/移植要点：

1. **只对「本地配置」申请租约** —— 调用方先用 ``is_local_config(base_url, provider)`` 判断
   （``ollama``/``local-sd``/``cosyvoice`` 三个 provider 名，或 baseUrl 指向 localhost/127.0.0.1/192.168.*）；
2. **策略分支的返回值是有讲究的**：``ollama-keep-alive`` / ``passive`` / ``sd-unload-checkpoint``
   一律返回 ``True``（**即使请求失败**），``comfyui-free`` 看响应码，``none``/默认返回 ``False``。
   驱逐逻辑据此累加「已释放显存」⇒ 改动这些返回值会改变驱逐行为；
3. **释放时分流**：``ollama-keep-alive`` / ``comfyui-free`` 走**立即卸载**（fire-and-forget + 从表里删除），
   其余策略只是打上 ``releasedAt`` 时间戳、等 TTL（10 分钟）过期后由 ``prune_expired`` 清理；
4. **驱逐顺序是从小到大**（先卸小的快速腾地方），且**跳过即将加载的那个模型**；每卸一个都要
   ``sleep(cooldownMs)`` 给服务端缓冲；
5. ⚠️ Ollama 卸载的模型名取自 ``modelKey.split(':')[1]`` ⇒ ``ollama:qwen3:14b`` 会得到
   **``qwen3``**（丢掉 ``:14b`` 标签）。这是**原 TS 的行为**（照抄），真卸载失败也不报错。

⚠️ 两处**有意差异**：

* TTL 用 ``time.monotonic()`` 而不是墙钟（TS 是 ``Date.now()``）—— 只用于「过了多久」的比较，
  单调钟不受系统时间调整影响，更稳；
* ``release()`` 里的 fire-and-forget 用 ``asyncio.get_running_loop().create_task``
  （**不能用 ``ensure_future`**，见 ``auto_pipeline`` 里踩过的 AnyIO 工作线程坑）。
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx

from ..core.response import js_round
from .ai_configs import LOCAL_PROVIDERS as _AI_CONFIGS_LOCAL_PROVIDERS
from .ai_configs import is_local_config
from .task_logger import log_task_error, log_task_progress, log_task_warn

__all__ = [
    "COMFYUI_BASE_URL",
    "GPU_SAFE_MARGIN_GB",
    "GPU_TOTAL_GB",
    "LOCAL_PROVIDERS",
    "MODEL_REUSE_TTL_MS",
    "VRAM_ESTIMATES",
    "GpuLease",
    "GpuMemoryManager",
    "get_nvidia_smi",
    "gpu_lease",
    "gpu_manager",
    "is_local_config",
    "is_local_provider",
]

# ─── 本地 Provider 标记 ────────────────────────────────────────
# ⚠️ 白名单与 ``is_local_config`` 的**唯一实现在 `services/ai_configs.py`**（它已在那边
#    带着「与 JS 的 URL 解析等价性」分析落地并被路由使用）⇒ 这里只做再导出，避免两份实现漂移。

#: 本地 provider 集合（这些是运行在本机 GPU 上的服务）
#: ⚠️ ``openai`` **不在**此列：Ollama 的 OpenAI 兼容接口 baseUrl 指向 localhost，
#: 会命中 ``is_local_config`` 的 hostname 判断；真实云端 OpenAI 不该被误判为本地服务。
#: ``minimax`` 本地 H3 同理（走 minimax provider + localhost baseUrl）。
LOCAL_PROVIDERS = set(_AI_CONFIGS_LOCAL_PROVIDERS)


def is_local_provider(provider: str) -> bool:
    """判断 provider 是否为本地 GPU 服务。"""
    return str(provider or "").lower() in LOCAL_PROVIDERS


# ─── VRAM 配置 ────────────────────────────────────────────────

#: ComfyUI HTTP 服务地址（用于主动卸载模型，可用 ``COMFYUI_URL`` 覆盖）
COMFYUI_BASE_URL = os.environ.get("COMFYUI_URL") or "http://localhost:8188"

#: 每个 provider+model 组合的显存估算（GB）+ 卸载策略 + 冷却时间（ms）
VRAM_ESTIMATES: dict[str, dict[str, Any]] = {
    # Ollama 文本模型
    "ollama:qwen3:14b": {"vramGB": 9, "unloadStrategy": "ollama-keep-alive", "cooldownMs": 2000},
    "ollama:qwen3:32b": {"vramGB": 16, "unloadStrategy": "ollama-keep-alive", "cooldownMs": 3000},
    "ollama:qwen3:8b": {"vramGB": 5, "unloadStrategy": "ollama-keep-alive", "cooldownMs": 1000},
    "openai:qwen3:14b": {"vramGB": 9, "unloadStrategy": "ollama-keep-alive", "cooldownMs": 2000},
    "openai:qwen3:32b": {"vramGB": 16, "unloadStrategy": "ollama-keep-alive", "cooldownMs": 3000},
    # SD WebUI 图片模型
    "local-sd:sdxl-base": {"vramGB": 12, "unloadStrategy": "sd-unload-checkpoint", "cooldownMs": 3000},
    "local-sd:sd15": {"vramGB": 5, "unloadStrategy": "sd-unload-checkpoint", "cooldownMs": 1500},
    # CosyVoice 语音模型（轻量）
    "cosyvoice:cosyvoice-v2": {"vramGB": 4, "unloadStrategy": "passive", "cooldownMs": 500},
    # MiniMax H3 本地视频模型（权重驻留 ComfyUI 显存，卸载走 ComfyUI /free）
    "minimax:hailuo-02": {"vramGB": 15, "unloadStrategy": "comfyui-free",
                          "unloadBaseUrl": COMFYUI_BASE_URL, "cooldownMs": 5000},
    # 默认兜底
    "default": {"vramGB": 8, "unloadStrategy": "none", "cooldownMs": 1000},
}

GPU_TOTAL_GB = 24
GPU_SAFE_MARGIN_GB = 2  # 安全余量，避免精确填满
#: 模型复用窗口：释放后超过该时长未再使用则视为过期清理（避免显存估算只增不减）
MODEL_REUSE_TTL_MS = 10 * 60 * 1000


async def _post_json(url: str, payload: dict[str, Any] | None, timeout_seconds: float) -> int:
    """POST 一个 JSON（返回状态码）。

    ⚠️ 抽成独立函数是**为了自检可替换**（把厂商卸载请求打桩，不发真网络）。
    ``payload=None`` 时发送空对象（对齐 TS 的 ``body: '{}'``）。
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
        response = await client.post(url, json=(payload if payload is not None else {}))
        return response.status_code


class GpuLease:
    """GPU 租约：使用完毕**必须**调用 ``release()``。"""

    __slots__ = ("serviceType", "modelKey", "vramGB", "_release")

    def __init__(self, service_type: str, model_key: str, vram_gb: float,
                 release: Any) -> None:
        self.serviceType = service_type
        self.modelKey = model_key
        self.vramGB = vram_gb
        self._release = release

    def release(self) -> None:
        """释放 GPU 占用（同步；内部按策略决定是否异步卸载）。"""
        self._release()


class GpuMemoryManager:
    """GPU 管理器（进程内单例，见 ``gpu_manager``）。"""

    def __init__(self) -> None:
        self._holder: str | None = None
        #: FIFO 等待队列：``{"serviceType": str, "future": asyncio.Future}``
        self._queue: list[dict[str, Any]] = []
        #: 已加载模型：``modelKey -> {"profile": {...}, "releasedAt": float | None}``
        self._loaded: dict[str, dict[str, Any]] = {}

    # ─── 只读状态 ───

    @property
    def total_vram_gb(self) -> float:
        """当前 GPU 总占用估算（GB）。"""
        return js_round(sum(entry["profile"]["vramGB"] for entry in self._loaded.values()) * 10) / 10

    @property
    def available_vram_gb(self) -> float:
        """当前可用显存（GB）。"""
        return js_round((GPU_TOTAL_GB - GPU_SAFE_MARGIN_GB - self.total_vram_gb) * 10) / 10

    @property
    def is_locked(self) -> bool:
        return self._holder is not None

    @property
    def current_holder(self) -> str | None:
        return self._holder

    @property
    def loaded_model_keys(self) -> list[str]:
        return list(self._loaded.keys())

    # ─── 内部：过期清理 / 配置 / 卸载 / 驱逐 ───

    def prune_expired(self) -> None:
        """清理超过复用窗口未再使用的已释放模型，避免显存估算只增不减。"""
        now_ms = time.monotonic() * 1000
        for key in list(self._loaded.keys()):
            entry = self._loaded[key]
            released_at = entry.get("releasedAt")
            if released_at is not None and now_ms - released_at > MODEL_REUSE_TTL_MS:
                del self._loaded[key]
                log_task_progress("GpuManager", "pruned-expired", {"modelKey": key})

    def get_profile(self, provider: str, model: str) -> dict[str, Any]:
        """取 VRAM 配置（未登记的 provider:model 落 ``default``）。"""
        key = f"{str(provider or '').lower()}:{model}"
        return VRAM_ESTIMATES.get(key) or VRAM_ESTIMATES["default"]

    async def unload_model(self, model_key: str, profile: dict[str, Any]) -> bool:
        """尝试卸载指定模型（返回值语义见模块 docstring 第 2 条）。"""
        strategy = profile.get("unloadStrategy")
        log_task_progress("GpuManager", "unload-attempt",
                          {"modelKey": model_key, "strategy": strategy})
        try:
            if strategy == "ollama-keep-alive":
                # ⚠️ 照抄 TS：只取 split(':')[1]，``ollama:qwen3:14b`` ⇒ ``qwen3``（丢掉标签）
                parts = model_key.split(":")
                model_name = parts[1] if len(parts) > 1 else None
                ollama_url = profile.get("baseUrl") or "http://localhost:11434"
                status = await _post_json(f"{ollama_url}/api/generate", {
                    "model": model_name or "qwen3:14b",
                    "prompt": ".",
                    "keep_alive": 0,
                    "max_tokens": 1,
                    "stream": False,
                }, 10)
                if 200 <= status < 300:
                    log_task_progress("GpuManager", "unload-ollama-ok", {"modelKey": model_key})
                # 注意：**无论响应码如何都算释放成功**（原 TS 如此）—— 它只看请求有没有抛错
                return True

            if strategy == "sd-unload-checkpoint":
                sd_url = profile.get("baseUrl") or "http://localhost:7860"
                try:
                    await _post_json(f"{sd_url}/sdapi/v1/unload-checkpoint", None, 5)
                    log_task_progress("GpuManager", "unload-sd-ok", {"modelKey": model_key})
                except Exception:  # noqa: BLE001 —— 部分版本不支持该接口
                    log_task_warn("GpuManager", "unload-sd-unsupported", {"modelKey": model_key})
                return True

            if strategy == "comfyui-free":
                comfy_url = profile.get("unloadBaseUrl") or COMFYUI_BASE_URL
                status = await _post_json(f"{comfy_url}/free",
                                          {"unload_models": True, "free_memory": True}, 10)
                if 200 <= status < 300:
                    log_task_progress("GpuManager", "unload-comfyui-ok", {"modelKey": model_key})
                    return True
                log_task_warn("GpuManager", "unload-comfyui-failed",
                              {"modelKey": model_key, "status": status})
                return False

            if strategy == "passive":
                log_task_progress("GpuManager", "unload-passive", {"modelKey": model_key})
                return True

            return False
        except Exception as exc:  # noqa: BLE001
            log_task_warn("GpuManager", "unload-error",
                          {"modelKey": model_key, "error": str(exc)})
            return False

    async def evict_if_needed(self, needed_gb: float, new_model_key: str) -> None:
        """释放所有冲突模型，为新的 GPU 任务腾出空间。"""
        if self.total_vram_gb + needed_gb <= GPU_TOTAL_GB - GPU_SAFE_MARGIN_GB:
            return  # 空间足够，无需驱逐

        need_to_free = self.total_vram_gb + needed_gb - (GPU_TOTAL_GB - GPU_SAFE_MARGIN_GB)
        log_task_progress("GpuManager", "evict-needed", {
            "currentVram": self.total_vram_gb,
            "needed": needed_gb,
            "needToFree": js_round(need_to_free * 10) / 10,
        })

        # 按显存占用**从小到大**排序，优先卸载小的（快速释放）；跳过即将加载的新模型
        candidates = sorted(
            ((key, entry) for key, entry in self._loaded.items() if key != new_model_key),
            key=lambda item: item[1]["profile"]["vramGB"],
        )

        freed = 0
        for key, entry in candidates:
            if freed >= need_to_free:
                break
            if await self.unload_model(key, entry["profile"]):
                freed += entry["profile"]["vramGB"]
                del self._loaded[key]
                log_task_progress("GpuManager", "evicted",
                                  {"modelKey": key, "freedGB": entry["profile"]["vramGB"]})
            await asyncio.sleep(entry["profile"]["cooldownMs"] / 1000)
        if freed < need_to_free:
            log_task_warn("GpuManager", "evict-insufficient", {
                "needToFree": js_round(need_to_free * 10) / 10,
                "freed": js_round(freed * 10) / 10,
                "remainingModels": list(self._loaded.keys()),
            })

    # ─── 核心：申请 / 释放 ───

    async def acquire(self, service_type: str, provider: str, model: str,
                      base_url: str | None = None) -> GpuLease:
        """获取 GPU 独占锁（必要时排队等待）。"""
        self.prune_expired()
        model_key = f"{str(provider or '').lower()}:{model}"
        # ⚠️ 浅拷贝：防止修改全局共享的 VRAM_ESTIMATES 配置对象
        profile = dict(self.get_profile(provider, model))
        if base_url:
            profile["baseUrl"] = base_url

        log_task_progress("GpuManager", "acquire-request", {
            "serviceType": service_type, "modelKey": model_key,
            "vramGB": profile["vramGB"], "currentVram": self.total_vram_gb,
            "holder": self._holder,
        })

        # 等待锁释放（FIFO 队列，避免竞态）
        if self._holder is not None:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._queue.append({"serviceType": service_type, "future": future})
            log_task_progress("GpuManager", "queued", {
                "serviceType": service_type, "queueLength": len(self._queue),
                "holder": self._holder,
            })
            await future

        self._holder = model_key
        await self.evict_if_needed(profile["vramGB"], model_key)
        self._loaded[model_key] = {"profile": profile, "releasedAt": None}

        log_task_progress("GpuManager", "acquired", {
            "serviceType": service_type, "modelKey": model_key,
            "vramGB": profile["vramGB"], "totalVram": self.total_vram_gb,
        })
        return GpuLease(service_type, model_key, profile["vramGB"],
                        lambda: self.release(model_key, profile))

    def release(self, model_key: str, profile: dict[str, Any]) -> None:
        """释放 GPU 锁（同步；主动卸载类策略会起后台任务）。"""
        log_task_progress("GpuManager", "release",
                          {"modelKey": model_key, "strategy": profile.get("unloadStrategy")})

        # ollama-keep-alive / comfyui-free：任务结束后**立即卸载**释放显存
        if profile.get("unloadStrategy") in ("ollama-keep-alive", "comfyui-free"):
            task = asyncio.get_running_loop().create_task(self.unload_model(model_key, profile))
            task.add_done_callback(_log_unload_failure(model_key))
            self._loaded.pop(model_key, None)
        else:
            # 其他策略：保留复用，记录释放时间，超过 TTL 后由 prune_expired 清理
            entry = self._loaded.get(model_key)
            if entry is not None:
                entry["releasedAt"] = time.monotonic() * 1000

        self._holder = None
        if self._queue:
            nxt = self._queue.pop(0)
            log_task_progress("GpuManager", "dequeue",
                              {"next": nxt["serviceType"], "remaining": len(self._queue)})
            if not nxt["future"].done():
                nxt["future"].set_result(None)

    async def release_all(self) -> None:
        """批量释放所有已加载模型（调试用）。"""
        log_task_progress("GpuManager", "release-all", {"count": len(self._loaded)})
        for key, entry in list(self._loaded.items()):
            await self.unload_model(key, entry["profile"])
        self._loaded.clear()
        self._holder = None
        # 清空队列（唤醒所有等待者）
        for item in self._queue:
            if not item["future"].done():
                item["future"].set_result(None)
        self._queue = []

    def get_status(self) -> dict[str, Any]:
        """当前状态快照（供 ``GET /gpu/status`` 返回）。"""
        self.prune_expired()
        return {
            "totalVRAM_GB": GPU_TOTAL_GB,
            "safeMargin_GB": GPU_SAFE_MARGIN_GB,
            "usedVRAM_GB": self.total_vram_gb,
            "availableVRAM_GB": self.available_vram_gb,
            "isLocked": self.is_locked,
            "holder": self._holder,
            "queueLength": len(self._queue),
            "loadedModels": self.loaded_model_keys,
        }


def _log_unload_failure(model_key: str):
    """生成 ``release()`` 里后台卸载任务的 done 回调（失败记一条 error）。"""
    def _done(future: "asyncio.Future[Any]") -> None:
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            log_task_error("GpuManager", "unload-on-release",
                           {"modelKey": model_key, "error": str(error)})

    return _done


#: 单例（与原 TS 的 ``export const gpuManager`` 对应）
gpu_manager = GpuMemoryManager()


@asynccontextmanager
async def gpu_lease(service_type: str, provider: str, model: str,
                    base_url: str | None = None) -> AsyncIterator[GpuLease]:
    """``withGpuLease`` 的 Python 形态：**只在本地配置时**申请，退出时必定释放。

    ⚠️ 与原 TS 的差异：TS 的 ``withGpuLease`` 无条件申请，是否本地由**调用方**先判断
    （``text-generation`` 的 ``if (isLocal)``）；这里把判断收进来，避免每个调用点重复写错。
    """
    if not is_local_config(base_url, provider):
        yield GpuLease(service_type, f"{str(provider or '').lower()}:{model}", 0, lambda: None)
        return
    lease = await gpu_manager.acquire(service_type, provider, model, base_url)
    try:
        yield lease
    finally:
        lease.release()


async def get_nvidia_smi() -> dict[str, Any] | None:
    """``nvidia-smi`` 实时数据（Windows/Linux 同一条命令）；失败返回 ``None``。

    ⚠️ 原 TS 用 ``child_process.exec`` + 5 秒超时；Python 侧用 ``create_subprocess_exec``，
    非零退出/命令不存在/超时都返回 ``None``（调用方据此判 ``isLocalMode``）。
    """
    query = ("nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,"
             "utilization.gpu,temperature.gpu --format=csv,noheader,nounits")
    try:
        process = await asyncio.create_subprocess_shell(
            query, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=5)
    except Exception:  # noqa: BLE001 —— 没装驱动/不是本机 GPU
        return None
    if process.returncode != 0:
        return None

    lines = (stdout.decode("utf-8", errors="replace") or "").strip().split("\n")
    fields = [part.strip() for part in lines[0].split(",")]
    # ⚠️ 字段数不足时返回 None（原 TS 会给出 parseInt(undefined)=NaN 的对象 ——
    #    那既不可用也无法序列化成合法 JSON，属原实现的坏路径，这里**有意加固**）
    if len(fields) < 6:
        return None

    def _parse_int(raw: str) -> int | None:
        """``parseInt(x, 10)``：取前导整数部分（``'8192 MiB' -> 8192``），无数字则 None。"""
        match = re.match(r"\s*([+-]?\d+)", raw)
        return int(match.group(1)) if match else None

    return {
        "gpuName": fields[0],
        "totalMemoryMB": _parse_int(fields[1]),
        "usedMemoryMB": _parse_int(fields[2]),
        "freeMemoryMB": _parse_int(fields[3]),
        "utilizationPercent": _parse_int(fields[4]),
        "temperatureC": _parse_int(fields[5]),
    }
