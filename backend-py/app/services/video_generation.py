"""视频生成链路（移植自 ``backend/src/services/video-generation.ts``，498 行）。

与图片链路（``image_generation.py``）结构同构 —— 入队即返回、后台跑、多模型 fallback、
崩溃恢复。**差异点**（都是这一层独有的）：

* **prompt 提交前要洗**：先剥掉分镜视频提示词里的结构化标签（``<location>``/``<n>``…，
  那是给程序解析的 DSL，视频扩散模型不认），再注入逐镜禁止变化清单；
* **H3 原生环境音标记**：分镜配了 ``sound_effect`` 且 prompt 里没有 ``[background_audio]``
  时注入（**幂等**：``videos.ts`` 富化路径注入过就不重复加）；
* **Vidu 是 Webhook 型**：提交成功即返回，**不轮询**（``parse_poll_response`` 恒为 processing）；
* **轮询更慢更长**：300 次 × 10 秒（图片是 120 × 5 秒）；
* **完成后要用 ffprobe 补时长**（异步提供商不返回 duration 时）。

⚠️ 与原实现一致的两处**反直觉行为**（都有用例锁住，别"顺手修好"）：

1. **同步完成路径不更新分镜的 ``video_url``** —— ``handleVideoComplete`` 的
   ``storyboardId`` 形参在同步路径是 ``undefined``，而只有它才会去更新分镜行；
   版本留档与 QC 走的是「形参 ?? 记录里的 storyboardId」所以仍然生效。
2. ``duration ?? undefined`` 与 ``meta ? {...} : undefined`` —— 值为空时**整个键不写**
   （对齐 JS ``JSON.stringify`` 丢 ``undefined``），**不是**写 ``null``。

⚠️ 已知的、有意的差异：**镜头 QC 打分未迁**（见 ``_run_qc_after_video_complete``）。
✅ **GPU 显存租约已接线**（本地配置才申请；**长租约**：提交时持有、完成/失败/重试时释放，
见 ``_video_gpu_leases`` 与 ``release_video_gpu_lease``）。
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.engine import Connection, Row

from ..core.db import engine
from ..core.models import api_usage, storyboards, video_generations
from ..core.response import js_truthy, now
from .adapters.registry import get_video_adapter, video_adapters
from .ai_configs import is_local_config
from .ai_providers import get_active_config, get_active_config_by_provider, get_config_by_id
from .asset_versions import record_asset_version
from .file_storage import download_file, read_image_as_compressed_data_url
from .gpu_manager import gpu_manager
from .prompt_utils import strip_video_prompt_tags
from .qc_scoring import run_qc_after_video_complete
from .script_fingerprint import check_storyboard_gate
from .take_budget import check_take_budget, consume_take
from .task_logger import (
    log_task_error,
    log_task_payload,
    log_task_progress,
    log_task_start,
    log_task_success,
    log_task_warn,
    redact_url,
)
from .usage_tracking import record_usage
from .vendor_errors import (
    _get_client,
    fetch_with_retry,
    format_vendor_http_error,
    format_vendor_task_error,
    is_non_retryable_http_error,
)
from .video_probe import probe_video_duration

__all__ = [
    "generate_video",
    "recover_video_tasks_on_startup",
]

#: 轮询上限：300 次 × 10 秒（≈50 分钟）
_POLL_MAX_ATTEMPTS = 300
_POLL_INTERVAL_SECONDS = 10.0

#: 恢复超时阈值：processing 超过该时长（毫秒）无进展，判为孤儿任务直接失败（60 分钟）
_RECOVER_TIMEOUT_MS = 60 * 60 * 1000

#: 后台任务的强引用集合（**不持引用会被 GC**，表现为「永远停在 processing」）
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _tx():
    """短写事务（与 Node 逐语句自动提交等价；轮询期间不能长持写事务）。"""
    return engine.begin()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _col(record: Any, name: str) -> Any:
    return getattr(record, name, None)


def _stringify(value: Any) -> str:
    """等价 ``JSON.stringify``（紧凑 + 不转义非 ASCII），见 image_generation._stringify。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ===========================================================================
# 入队
# ===========================================================================

#: 视频任务的 GPU 租约（**长租约**：提交时申请、完成/失败/重试时释放）。
#: ⚠️ 与 text/tts 的「请求内即用即放」不同 —— 视频是**提交后轮询**型，租约必须跨轮询持有；
#: 对应原 TS 的 ``const videoGpuLeases = new Map<number, GpuLease>()``。
_video_gpu_leases: dict[int, Any] = {}


def release_video_gpu_lease(video_id: int) -> None:
    """释放视频任务的 GPU 租约（幂等：没有就什么都不做）。

    调用点与原 TS 一致：**重试前**、**最后一次尝试失败**、**完成**。
    """
    lease = _video_gpu_leases.pop(video_id, None)
    if lease is not None:
        lease.release()


async def generate_video(conn: Connection, params: dict[str, Any]) -> int:
    """入队一次视频生成，返回 ``video_generations.id``。

    ``params`` 见原 ``GenerateVideoParams``（storyboardId / dramaId / prompt / referenceMode /
    imageUrl / firstFrameUrl / lastFrameUrl / referenceImageUrls / sceneType /
    referenceAudioUrls / duration / aspectRatio / route / routeReason / force / configId）。
    """
    ts = now()
    config = (
        get_config_by_id(conn, params["configId"])
        if params.get("configId")
        else get_active_config(conn, "video")
    )
    if not config:
        raise ValueError("No active video AI config")

    storyboard_id = params.get("storyboardId")

    # per-shot take 预算（force 可放行）
    if storyboard_id and not params.get("force"):
        budget = check_take_budget(conn, storyboard_id)
        if not budget["allowed"]:
            log_task_warn("VideoTask", "take-budget-exhausted",
                          {"storyboardId": storyboard_id, "reason": budget["reason"]})
            raise ValueError(budget["reason"])

    # 剧本内容指纹门禁（force 可放行）
    if storyboard_id and not params.get("force"):
        gate = check_storyboard_gate(conn, storyboard_id)
        if not gate["allowed"]:
            log_task_warn("VideoTask", "gate-blocked",
                          {"storyboardId": storyboard_id, "reason": gate["reason"]})
            raise ValueError(gate["reason"])

    # 剥离分镜视频提示词中的结构化标签（<location>/<role>/<voice>/<n>），
    # 让视频生成模型拿到干净的纯自然语言 prompt（标签是给 agent/程序解析用的 DSL）
    prompt = strip_video_prompt_tags(params.get("prompt"))

    if storyboard_id:
        sb_row = conn.execute(
            select(storyboards.c.constraints, storyboards.c.sound_effect).where(
                storyboards.c.id == storyboard_id
            )
        ).first()

        # 连续性状态机 v3：逐镜禁止变化清单注入
        constraints = sb_row[0] if sb_row is not None else None
        if constraints:
            prompt = (
                f"{prompt} -- 逐镜禁止变化（画面中以下元素必须始终保持不变，不得增减或改变）: {constraints}"
            )

        # H3 原生场景声标记 [background_audio]（对齐 minimax-h3-comfyui 语法）：
        # 分镜配了 sound_effect 且 prompt 尚未包含该标记时注入，H3 会在成片时同步合成环境音。
        # **幂等**：videos.ts 富化路径已注入过则不再重复追加。
        sound_effect = sb_row[1] if sb_row is not None else None
        if (sound_effect or "").strip() and "[background_audio]" not in prompt:
            prompt = f"{prompt} [background_audio] {sound_effect.strip()}"

    negative_prompt = params.get("negativePrompt")
    if negative_prompt is None:
        negative_prompt = config.get("negativePrompt")
    reference_image_urls = params.get("referenceImageUrls")
    reference_audio_urls = params.get("referenceAudioUrls")

    result = conn.execute(
        video_generations.insert().values(
            storyboard_id=storyboard_id,
            drama_id=params.get("dramaId"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=params.get("model") or config.get("model"),
            provider=config.get("provider"),
            # `params.referenceMode || 'none'` 是 **truthy**：空串会回落到 'none'
            reference_mode=params.get("referenceMode") or "none",
            image_url=params.get("imageUrl"),
            first_frame_url=params.get("firstFrameUrl"),
            last_frame_url=params.get("lastFrameUrl"),
            reference_image_urls=_stringify(reference_image_urls) if js_truthy(reference_image_urls) else None,
            # `params.sceneType || null`：空串也落 null（truthy 判断）
            scene_type=params.get("sceneType") or None,
            # ⚠️ 参考音频判的是 **`.length`**（非空数组）—— 空数组在 JS 是真值，
            #    所以不能用 `js_truthy`，否则 `[]` 会被落成 "[]"（与 TS 不一致）
            reference_audio_urls=_stringify(reference_audio_urls)
            if reference_audio_urls and len(reference_audio_urls) > 0
            else None,
            route=params.get("route") or None,
            route_reason=params.get("routeReason") or None,
            # `params.duration || 5`：0 也回落 5（truthy）
            duration=params.get("duration") or 5,
            aspect_ratio=params.get("aspectRatio") or "16:9",
            status="processing",
            created_at=ts,
            updated_at=ts,
        )
    )
    last_id = int(result.lastrowid)

    # 任务提交成功即消耗一次 take（无论成败）
    if storyboard_id:
        consume_take(conn, storyboard_id)

    log_task_start("VideoTask", "enqueue", {
        "id": last_id,
        "provider": config.get("provider"),
        "storyboardId": storyboard_id,
        "dramaId": params.get("dramaId"),
        "referenceMode": params.get("referenceMode") or "none",
        "duration": params.get("duration") or 5,
    })
    log_task_payload("VideoTask", "enqueue params", {
        "id": last_id,
        "config": {
            "provider": config.get("provider"),
            "model": config.get("model"),
            "baseUrl": config.get("baseUrl"),
        },
        "params": params,
    })

    _spawn(_run_video_generation(last_id, config))
    return last_id


async def _run_video_generation(video_id: int, config: dict[str, Any]) -> None:
    try:
        await _process_video_generation(video_id, config)
    except Exception as err:  # noqa: BLE001
        log_task_error("VideoTask", "process", {"id": video_id, "error": str(err)})
        print(f"Video generation {video_id} failed: {err}")


def _fetch_record(conn: Connection, video_id: int) -> Row | None:
    return conn.execute(
        select(video_generations).where(video_generations.c.id == video_id)
    ).first()


async def _process_video_generation(video_id: int, config: dict[str, Any]) -> None:
    """处理视频生成任务（含模型自动 fallback）。"""
    with _tx() as conn:
        record = _fetch_record(conn, video_id)
    if record is None:
        return

    resolved_image_url = await _normalize_video_reference_url(_col(record, "image_url"))
    resolved_first_frame = await _normalize_video_reference_url(_col(record, "first_frame_url"))
    resolved_last_frame = await _normalize_video_reference_url(_col(record, "last_frame_url"))
    resolved_reference_images = await _normalize_video_reference_urls(
        _col(record, "reference_image_urls")
    )

    models = list(config["models"]) if config.get("models") else [config.get("model")]

    for attempt, model in enumerate(models):
        attempt_config = {**config, "model": model}

        with _tx() as conn:
            conn.execute(
                video_generations.update()
                .where(video_generations.c.id == video_id)
                .values(model=model, updated_at=now())
            )

        # ── 非首次尝试：释放上一次的 GPU 租约 ──
        if attempt > 0:
            release_video_gpu_lease(video_id)

        is_local = is_local_config(config.get("baseUrl") or "", config.get("provider") or "")

        # ── 本地 GPU 模型：申请显存租约（**失败只告警，不中断任务** —— 与原 TS 一致）──
        if is_local:
            try:
                _video_gpu_leases[video_id] = await gpu_manager.acquire(
                    "video", config.get("provider"), model, config.get("baseUrl")
                )
            except Exception as err:  # noqa: BLE001
                log_task_warn("VideoTask", "gpu-acquire-failed",
                              {"id": video_id, "model": model, "error": str(err)})

        # 用量记账：**units = 时长（秒）**（图片是 1 张）—— 视频按秒计费
        with _tx() as conn:
            record_usage(conn, {
                "serviceType": "video",
                "provider": config.get("provider"),
                "model": model,
                "dramaId": _col(record, "drama_id"),
                "storyboardId": _col(record, "storyboard_id"),
                "videoGenerationId": video_id,
                "units": _col(record, "duration"),
                "isLocal": is_local,
                "status": "submitted",
                "retryCount": attempt,
                "settings": config.get("settings"),
            })

        try:
            log_task_progress("VideoTask", "build-request", {
                "id": video_id, "provider": config.get("provider"), "model": model,
                "attempt": attempt + 1,
                "storyboardId": _col(record, "storyboard_id"),
                "referenceMode": _col(record, "reference_mode"),
            })

            adapter = get_video_adapter(config.get("provider"))
            request = adapter.build_generate_request(attempt_config, {
                "id": _col(record, "id"),
                "model": model,
                "prompt": _col(record, "prompt"),
                "negativePrompt": _col(record, "negative_prompt"),
                "referenceMode": _col(record, "reference_mode"),
                "imageUrl": resolved_image_url,
                "firstFrameUrl": resolved_first_frame,
                "lastFrameUrl": resolved_last_frame,
                "referenceImageUrls": _stringify(resolved_reference_images)
                if js_truthy(resolved_reference_images)
                else None,
                "sceneType": _col(record, "scene_type"),
                "referenceAudioUrls": _col(record, "reference_audio_urls"),
                "duration": _col(record, "duration"),
                "aspectRatio": _col(record, "aspect_ratio"),
            })

            url = request["url"]
            method = request["method"]
            headers = request["headers"]
            body = request["body"]
            log_task_progress("VideoTask", "request", {
                "id": video_id, "provider": config.get("provider"), "method": method,
                "url": redact_url(url), "model": model,
                "referenceMode": _col(record, "reference_mode"),
            })
            log_task_payload("VideoTask", "request payload", {
                "id": video_id, "method": method, "url": url, "headers": headers, "body": body,
            })

            # ⚠️ 这里**不传 timeout_ms**（原实现如此）—— 视频创建请求可能很慢
            response = await fetch_with_retry(
                url,
                {
                    "method": method,
                    "headers": headers,
                    "body": _stringify(body) if body is not None else None,
                },
                "video",
                on_retry=lambda retry_attempt, delay_ms, reason: log_task_warn(
                    "VideoTask", "request-retry",
                    {"id": video_id, "model": model, "attempt": retry_attempt,
                     "delayMs": delay_ms, "reason": reason},
                ),
            )
            result = response.json()

            parsed = adapter.parse_generate_response(result)
            is_async = parsed.get("isAsync")
            task_id = parsed.get("taskId")
            video_url = parsed.get("videoUrl")

            if not is_async and video_url:
                log_task_progress("VideoTask", "sync-complete",
                                  {"id": video_id, "model": model, "videoUrl": video_url})
                # ⚠️ 同步路径**不传 storyboardId**（原实现如此）⇒ 分镜行不会被更新，
                #    只有版本留档与 QC 会通过记录里的 storyboardId 生效。见模块头。
                await _handle_video_complete(video_id, video_url, _col(record, "duration"))
                return

            # 异步模式：更新 taskId
            with _tx() as conn:
                conn.execute(
                    video_generations.update()
                    .where(video_generations.c.id == video_id)
                    .values(task_id=task_id, status="processing", updated_at=now())
                )
            log_task_progress("VideoTask", "poll-start", {
                "id": video_id, "taskId": task_id,
                "provider": config.get("provider"), "model": model,
            })

            # Vidu 没有轮询端点：提交成功后直接返回，**依赖 Webhook 回调完成**
            if adapter.provider == "vidu":
                log_task_progress("VideoTask", "webhook-wait",
                                  {"id": video_id, "taskId": task_id, "provider": adapter.provider})
                return

            await _poll_video_task(video_id, attempt_config, task_id, _col(record, "storyboard_id"))
            return
        except Exception as err:  # noqa: BLE001
            is_last_attempt = attempt == len(models) - 1
            warn_meta: dict[str, Any] = {
                "id": video_id, "attempt": attempt + 1, "totalModels": len(models),
                "failedModel": model, "error": str(err),
            }
            if not is_last_attempt:
                warn_meta["nextModel"] = models[attempt + 1]
            log_task_warn("VideoTask", "all-models-failed" if is_last_attempt else "model-fallback", warn_meta)

            if is_last_attempt:
                release_video_gpu_lease(video_id)
                with _tx() as conn:
                    _mark_usage_by_video_gen(conn, video_id, "failed")
                    log_task_error("VideoTask", "process", {
                        "id": video_id, "provider": config.get("provider"),
                        "attemptedModels": models, "error": str(err),
                    })
                    conn.execute(
                        video_generations.update()
                        .where(video_generations.c.id == video_id)
                        .values(
                            status="failed",
                            error_msg=f"All models failed. Last error: {err}",
                            updated_at=now(),
                        )
                    )
                # 注意：**视频失败不标记分镜资产**（图片链路才有 markStoryboardAssetNeedsRegeneration）


async def _normalize_video_reference_url(value: Any) -> str | None:
    """单个参考图 URL 归一化：``static/`` 相对路径转 data URL；空/读失败返回 None。"""
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.startswith("data:image/"):
        return raw
    if raw.startswith("static/") or raw.startswith("/static/"):
        local_path = raw[1:] if raw.startswith("/static/") else raw
        try:
            return await read_image_as_compressed_data_url(
                local_path, {"maxWidth": 768, "maxHeight": 768, "quality": 68}
            )
        except Exception as err:  # noqa: BLE001
            log_task_warn("VideoTask", "reference-read-failed",
                          {"path": local_path, "error": str(err)})
            return None
    return raw


async def _normalize_video_reference_urls(raw: Any) -> list[str]:
    """多个参考图 URL：去空、去重后逐个归一化，**丢掉归一化失败的**。"""
    if not raw:
        return []
    try:
        refs = json.loads(raw)
    except (ValueError, TypeError):
        refs = []
    if not isinstance(refs, list):
        refs = []

    deduped: list[str] = []
    for item in refs:
        value = str(item or "").strip()
        if value and value not in deduped:
            deduped.append(value)

    normalized: list[str] = []
    for value in deduped:
        result = await _normalize_video_reference_url(value)
        if result:
            normalized.append(result)
    return normalized


async def _poll_video_task(
    video_id: int, config: dict[str, Any], task_id: str, storyboard_id: Any = None
) -> None:
    """轮询异步视频任务直到完成或失败。

    ⚠️ 与图片链路的两点不同（原实现如此）：

    * **没有「总时长」二次判定**，只有 300 次的循环计数；
    * 最后一次（``i == 299``）出错时**直接抛出**（给外层 fallback 循环），不去看耗时。
    """
    adapter = get_video_adapter(config.get("provider"))

    for index in range(_POLL_MAX_ATTEMPTS):
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        try:
            poll_request = adapter.build_poll_request(config, task_id)
            url = poll_request["url"]
            method = poll_request["method"]
            headers = poll_request["headers"]
            log_task_progress("VideoTask", "poll-request", {
                "id": video_id, "taskId": task_id, "provider": config.get("provider"),
                "method": method, "url": redact_url(url), "attempt": index + 1,
            })
            response = await _get_client().request(method, url, headers=headers)
            if not response.is_success:
                body_text = response.text
                if is_non_retryable_http_error(response.status_code, body_text):
                    raise ValueError(
                        format_vendor_http_error(response.status_code, body_text, "video")
                    )
                continue
            result = response.json()

            poll_response = adapter.parse_poll_response(result)

            if poll_response.get("status") == "completed" and poll_response.get("videoUrl"):
                log_task_success("VideoTask", "poll-complete", {
                    "id": video_id, "taskId": task_id, "videoUrl": poll_response["videoUrl"],
                })
                # 轮询路径**传了 storyboardId** ⇒ 会更新分镜行（与同步路径不同）
                await _handle_video_complete(
                    video_id, poll_response["videoUrl"], None, storyboard_id
                )
                return
            if poll_response.get("status") == "failed":
                message = format_vendor_task_error(poll_response.get("error"), "video")
                log_task_error("VideoTask", "poll-failed",
                               {"id": video_id, "taskId": task_id, "error": message})
                raise ValueError(message)
        except Exception as err:  # noqa: BLE001
            if index == _POLL_MAX_ATTEMPTS - 1:
                raise  # 重新抛出给外层 fallback 循环
            log_task_warn("VideoTask", "poll-retry",
                          {"id": video_id, "taskId": task_id, "attempt": index + 1, "error": str(err)})


# ===========================================================================
# 完成 / 恢复
# ===========================================================================

def _mark_usage_by_video_gen(conn: Connection, video_generation_id: int, status: str) -> None:
    conn.execute(
        api_usage.update()
        .where(
            and_(
                api_usage.c.video_generation_id == video_generation_id,
                api_usage.c.status == "submitted",
            )
        )
        .values(status=status)
    )


async def _handle_video_complete(
    video_id: int,
    video_url: str,
    duration: Any,
    storyboard_id: Any = None,
) -> None:
    """下载视频、补时长、更新记录、回写分镜、留档、触发 QC。

    ⚠️ ``storyboard_id`` 为 None 时**不更新分镜行**（同步完成路径就是这样）——
    但版本留档与 QC 会用「形参 ?? 记录里的 storyboard_id」回退，仍然生效。
    """
    release_video_gpu_lease(video_id)
    with _tx() as conn:
        _mark_usage_by_video_gen(conn, video_id, "completed")
    local_path = await download_file(video_url, "videos")

    # 异步提供商（轮询/Webhook）不返回时长时，用 ffprobe 探测本地文件实际时长
    resolved_duration = duration if duration is not None else None
    if resolved_duration is None:
        probed = await probe_video_duration(local_path)
        if probed > 0:
            resolved_duration = probed

    with _tx() as conn:
        conn.execute(
            video_generations.update()
            .where(video_generations.c.id == video_id)
            .values(
                video_url=video_url,
                local_path=local_path,
                status="completed",
                completed_at=now(),
                updated_at=now(),
            )
        )
    log_task_success("VideoTask", "downloaded", {
        "id": video_id, "localPath": local_path,
        "storyboardId": storyboard_id, "duration": resolved_duration,
    })

    if storyboard_id:
        sb_values: dict[str, Any] = {"video_url": local_path, "updated_at": now()}
        # ⚠️ `duration: resolvedDuration ?? undefined` —— 拿不到时长时**整个键不写**
        #    （对齐 JSON.stringify 丢 undefined），不是写 null
        if resolved_duration is not None:
            sb_values["duration"] = resolved_duration
        with _tx() as conn:
            conn.execute(
                storyboards.update().where(storyboards.c.id == storyboard_id).values(**sb_values)
            )

    with _tx() as conn:
        gen = _fetch_record(conn, video_id)

    # 资产版本历史留档（分镜视频）。`frameType: null` 是**显式**传的（视频没有帧类型概念）。
    version_sb_id = storyboard_id or (_col(gen, "storyboard_id") if gen is not None else None)
    if version_sb_id:
        gen_duration = _col(gen, "duration") if gen is not None else None
        with _tx() as conn:
            record_asset_version(
                conn,
                asset_type="storyboard",
                asset_id=version_sb_id,
                media_type="video",
                frame_type=None,
                asset_url=local_path,
                provider=_col(gen, "provider") if gen is not None else None,
                model=_col(gen, "model") if gen is not None else None,
                prompt=_col(gen, "prompt") if gen is not None else None,
                generation_id=video_id,
                # `gen?.duration ? { duration } : undefined` —— 无时长时不传 meta
                meta={"duration": gen_duration} if js_truthy(gen_duration) else None,
            )

    # 触发镜头级 QC 打分（fire-and-forget，**未迁**）
    qc_storyboard_id = storyboard_id or (_col(gen, "storyboard_id") if gen is not None else None)
    if qc_storyboard_id:
        _run_qc_after_video_complete(qc_storyboard_id, video_id)


def _run_qc_after_video_complete(storyboard_id: Any, video_generation_id: int) -> None:
    """视频完成后的镜头级 QC（**fire-and-forget 增强**，失败只 warn、绝不影响成片）。

    ✅ **规则打分 + 技术维度都已接线**（``qc_scoring.run_qc_after_video_complete`` ⇒ 写
    ``video_quality_checks``；技术维度见 ``technical_qc``，2026-09-15 起不再是 ``tech-qc-skipped`` 存根）。
    ⚠️ 技术维度是**同事务同步**跑的（Node 侧是 fire-and-forget 不等待）——理由见 ``technical_qc``
    模块 docstring；失败只 warn，绝不影响成片。
    """
    try:
        with _tx() as conn:
            run_qc_after_video_complete(conn, storyboard_id, video_generation_id)
    except Exception as err:  # noqa: BLE001 —— fire-and-forget：绝不让 QC 影响成片流程
        log_task_warn("VideoTask", "qc-failed", {
            "storyboardId": storyboard_id,
            "videoGenerationId": video_generation_id,
            "error": str(err),
        })


def _mark_recover_failed(conn: Connection, video_id: int, reason: str) -> None:
    conn.execute(
        video_generations.update()
        .where(video_generations.c.id == video_id)
        .values(status="failed", error_msg=reason, updated_at=now())
    )
    log_task_error("VideoTask", "recover-failed", {"id": video_id, "reason": reason})
    # 注意：视频恢复失败**不**标记分镜资产（与图片链路的 markImageRecoverFailed 不同）


def recover_video_tasks_on_startup() -> None:
    """服务启动时恢复被中断的视频生成任务（崩溃恢复）。

    与图片链路唯一的结构差异：**Vidu 是 Webhook 型**，恢复时保持 ``processing`` 等外部回调
    （外部回调仍会到达，不该判失败）。

    恢复策略（幂等，**绝不重新提交**，避免按次计费厂商重复扣费）：
    超时（60 分钟）→ 无 taskId → 未知 provider → vidu 保持等待 →
    按 provider 匹配活跃配置续跑轮询；匹配不到配置则保持 processing。
    """
    with _tx() as conn:
        orphans = conn.execute(
            select(video_generations).where(video_generations.c.status == "processing")
        ).all()

        if not orphans:
            return
        log_task_warn("VideoTask", "recover-start", {"count": len(orphans)})

        for row in orphans:
            try:
                if _col(row, "updated_at") and (
                    _now_ms() - _parse_iso_ms(_col(row, "updated_at")) > _RECOVER_TIMEOUT_MS
                ):
                    _mark_recover_failed(
                        conn,
                        _col(row, "id"),
                        f"Interrupted and expired (idle > {_RECOVER_TIMEOUT_MS // 60000}min)",
                    )
                    continue

                provider = (_col(row, "provider") or "").lower()

                if not _col(row, "task_id"):
                    _mark_recover_failed(conn, _col(row, "id"), "Interrupted before task submit")
                    continue

                # 未知 provider 显式失败（get_video_adapter 已改为 throw，不再静默 fallback minimax）
                if provider not in video_adapters:
                    _mark_recover_failed(
                        conn,
                        _col(row, "id"),
                        f"Unknown video provider: {_col(row, 'provider') or '(empty)'}",
                    )
                    continue

                if provider == "vidu":
                    # Webhook 型：外部回调仍会到达，保持 processing 等待
                    log_task_warn("VideoTask", "recover-webhook-wait", {
                        "id": _col(row, "id"), "taskId": _col(row, "task_id"), "provider": provider,
                    })
                    continue

                config = get_active_config_by_provider(conn, "video", provider)
                if not config:
                    log_task_warn("VideoTask", "recover-no-config", {
                        "id": _col(row, "id"), "taskId": _col(row, "task_id"), "provider": provider,
                    })
                    continue
                if _col(row, "model"):
                    config["model"] = _col(row, "model")

                log_task_warn("VideoTask", "recover-resume-poll", {
                    "id": _col(row, "id"), "taskId": _col(row, "task_id"),
                    "provider": provider, "model": _col(row, "model"),
                })
                _spawn(_resume_poll(
                    _col(row, "id"), config, _col(row, "task_id"), _col(row, "storyboard_id")
                ))
            except Exception as err:  # noqa: BLE001
                log_task_error("VideoTask", "recover-error",
                               {"id": _col(row, "id"), "error": str(err)})


async def _resume_poll(
    video_id: int, config: dict[str, Any], task_id: str, storyboard_id: Any = None
) -> None:
    """恢复轮的续跑包装：失败时标记 failed（与 TS 的 catch 一致）。"""
    try:
        await _poll_video_task(video_id, config, task_id, storyboard_id)
    except Exception as err:  # noqa: BLE001
        log_task_error("VideoTask", "recover-poll-failed", {"id": video_id, "error": str(err)})
        with _tx() as conn:
            _mark_recover_failed(conn, video_id, f"Recover poll failed: {err}")


def _parse_iso_ms(value: str) -> int:
    """ISO 时间串 → 毫秒时间戳（等价 ``new Date(s).getTime()``）。"""
    from datetime import datetime, timezone

    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return 0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)
