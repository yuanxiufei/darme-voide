"""图片生成链路（移植自 ``backend/src/services/image-generation.ts``，753 行）。

调用形状与 Node 完全一致 —— **入队即返回、后台跑**：

    image_id = await generate_image(conn, params)   # 插入 processing 记录并返回 id
    # 之后由后台协程跑：多模型 fallback → 适配器构请求 → 同步/异步（轮询）→ 落盘 → 回写关联表

三条与原实现**刻意保持一致**的设计：

1. **fire-and-forget**：入队接口不等生成完成（一次生图可能轮询 10 分钟）。Python 侧用
   ``asyncio.create_task`` + 模块级任务集合持引用（**不持引用会被 GC 掉**，表现为
   「生图永远停在 processing」这种极难排查的问题）。
2. **多模型 fallback**：``config.models`` 里任一模型成功即完成；每次尝试单独记一条
   ``submitted`` 用量（``retry_count`` 从 0 起），最后收口成 completed / failed。
3. **崩溃恢复绝不重提交**：重启后只对「已提交拿到 ``taskId``」的任务**续跑轮询**，
   其余判 failed —— 避免按次计费的厂商**重复扣费**。

⚠️ 两处已知的、**有意**的行为差异（详见 README「已知差异」表）：

* **校色**与**参考图压缩**未迁（都需要 sharp / Pillow）。校色走「与原实现**失败时**相同」
  的分支（告警 + 保留原图），参考图退化为原图 data URL。

✅ **GPU 显存租约已接线**（本地配置才申请，**长租约**：提交时持有、完成/失败/重试时释放，
见 ``_image_gpu_leases`` 与 ``release_image_gpu_lease``）。
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.engine import Connection, Row

from ..db import engine
from ..models import (
    api_usage,
    characters,
    image_generations,
    prop_templates,
    scenes,
    storyboards,
)
from ..response import js_truthy, now
from .adapters.registry import get_image_adapter, image_adapters
from .ai_configs import is_local_config
from .ai_providers import get_active_config, get_active_config_by_provider, get_config_by_id
from .asset_versions import record_asset_version, resolve_storyboard_frame_type
from .color_grade import apply_color_grade_to_file, has_color_grade
from .era_background import apply_era_image_clause
from .file_storage import (
    download_file,
    read_image_as_compressed_data_url,
    save_base64_image,
)
from .gpu_manager import gpu_manager
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

__all__ = [
    "generate_image",
    "recover_image_tasks_on_startup",
]

#: 轮询上限：10 分钟 / 每次间隔 5 秒 ⇒ 最多 120 次
_POLL_MAX_ATTEMPTS = 120
_POLL_INTERVAL_SECONDS = 5.0
_POLL_MAX_DURATION_MS = 600_000

#: 恢复超时阈值：processing 超过该时长（毫秒）无进展，判为孤儿任务直接失败（60 分钟）
_RECOVER_TIMEOUT_MS = 60 * 60 * 1000

#: 后台任务的强引用集合 —— 见模块头第 1 条
_background_tasks: set[asyncio.Task] = set()

#: 完成阶段回写 JSON 列时，每类资产的「项内字段名」（原 TS 里三个分支各写一个字段名）
_ITEM_KEY_FIELD = {
    "item_images": "type",
    "three_views": "view",
    "equip_images": "type",
}


def _spawn(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _now_ms() -> int:
    """等价 ``Date.now()``（毫秒时间戳）。"""
    return int(time.time() * 1000)


def _col(record: Any, name: str) -> Any:
    """取记录字段（不存在时 None）。

    ``drizzle`` 的行对象字段齐全，而这里用 ``getattr(..., None)`` 兜底，避免模型缺列就炸。
    """
    return getattr(record, name, None)


def _tx():
    """开一个短写事务（与 Node 逐语句自动提交等价）。

    ⚠️ 刻意**不**全程持有一个事务：轮询最长 10 分钟，SQLite 下长时间持有写事务会阻塞
    Node 侧的写入（绞杀期两个后端共用同一个库）；better-sqlite3 本来就是逐语句提交。
    """
    return engine.begin()


def _stringify(value: Any) -> str:
    """等价 JS ``JSON.stringify``：**紧凑分隔符** + 不转义非 ASCII。

    ⚠️ Python 的 ``json.dumps`` 默认分隔符是 ``(', ', ': ')``，会**多出空格**
    （``{"exposure": 0.5}`` vs JS 的 ``{"exposure":0.5}``）。这些值会写进**与 Node 共用**的列
    （``image_generations.reference_images`` / ``color_grade``、``characters.*`` 的 JSON 列），
    两边字符串不一致会让「同一条记录」看起来不同 —— 而且**前端可能会按原文比对**。
    这个偏差是靠自检「colorGrade 有调整 -> 紧凑 JSON」那条用例抓出来的。
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ===========================================================================
# 入队
# ===========================================================================

#: 图片任务的 GPU 租约（**长租约**：提交时申请、完成/失败/重试时释放）。
#: ⚠️ 与 text/tts 的「请求内即用即放」不同 —— 图片是**提交后轮询**型，租约必须跨轮询持有；
#: 对应原 TS 的 ``const imageGpuLeases = new Map<number, GpuLease>()``。
_image_gpu_leases: dict[int, Any] = {}


def release_image_gpu_lease(image_id: int) -> None:
    """释放图片任务的 GPU 租约（幂等：没有就什么都不做）。

    调用点与原 TS 一致：**重试前**、**最后一次尝试失败**、**完成**（下载/base64 两条路径）。
    """
    lease = _image_gpu_leases.pop(image_id, None)
    if lease is not None:
        lease.release()


async def generate_image(conn: Connection, params: dict[str, Any]) -> int:
    """入队一次图片生成，返回 ``image_generations.id``。

    ``params`` 字段见原 ``GenerateImageParams``（storyboardId / dramaId / sceneId /
    characterId / propId / prompt / negativePrompt / model / size / referenceImages /
    frameType / configId / costume / colorGrade / viewType / equipType / itemType /
    expression / force）。**字段名保持 camelCase**（跨语言契约）。
    """
    ts = now()
    config = (
        get_config_by_id(conn, params["configId"])
        if params.get("configId")
        else get_active_config(conn, "image")
    )
    if not config:
        raise ValueError("No active image AI config")

    storyboard_id = params.get("storyboardId")

    # per-shot take 预算：超预算则阻断（force 可放行）
    if storyboard_id and not params.get("force"):
        budget = check_take_budget(conn, storyboard_id)
        if not budget["allowed"]:
            log_task_warn("ImageTask", "take-budget-exhausted",
                          {"storyboardId": storyboard_id, "reason": budget["reason"]})
            raise ValueError(budget["reason"])

    # 剧本内容指纹门禁：分镜媒体生成前校验剧本未变更，过期分镜阻断（force 可放行）
    if storyboard_id and not params.get("force"):
        gate = check_storyboard_gate(conn, storyboard_id)
        if not gate["allowed"]:
            log_task_warn("ImageTask", "gate-blocked",
                          {"storyboardId": storyboard_id, "reason": gate["reason"]})
            raise ValueError(gate["reason"])

    # 连续性状态机 v3：逐镜禁止变化清单注入（该镜画面中必须保持不变的元素）
    prompt = params.get("prompt")
    if storyboard_id:
        row = conn.execute(
            select(storyboards.c.constraints).where(storyboards.c.id == storyboard_id)
        ).first()
        if row is not None and row[0]:
            prompt = (
                f"{prompt} -- 逐镜禁止变化（画面中以下元素必须保持不变，不得增减或改变）: {row[0]}"
            )

    # 剧集时代背景自动注入：整剧所有视觉资产统一携带时代/环境美术指令
    if params.get("dramaId"):
        prompt = apply_era_image_clause(conn, prompt, params["dramaId"])

    color_grade = params.get("colorGrade")
    reference_images = params.get("referenceImages")
    negative_prompt = params.get("negativePrompt")
    if negative_prompt is None:
        # `params.negativePrompt ?? config.negativePrompt ?? null` —— **nullish 链**
        negative_prompt = config.get("negativePrompt")

    result = conn.execute(
        image_generations.insert().values(
            storyboard_id=storyboard_id,
            drama_id=params.get("dramaId"),
            scene_id=params.get("sceneId"),
            character_id=params.get("characterId"),
            prop_id=params.get("propId"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=params.get("model") or config.get("model"),
            provider=config.get("provider"),
            size=params.get("size") or "1920x1080",
            frame_type=params.get("frameType"),
            # `params.referenceImages ? JSON.stringify(...) : null`：空数组在 JS 是**真值**
            # ⇒ 传 `[]` 也会落成 `"[]"`（不是 null）
            reference_images=_stringify(reference_images) if js_truthy(reference_images) else None,
            costume=params.get("costume"),
            color_grade=_stringify(color_grade)
            if js_truthy(color_grade) and has_color_grade(color_grade)
            else None,
            view_type=params.get("viewType"),
            equip_type=params.get("equipType"),
            item_type=params.get("itemType"),
            expression=params.get("expression"),
            status="processing",
            created_at=ts,
            updated_at=ts,
        )
    )
    last_id = int(result.lastrowid)

    # per-shot take 预算：任务提交成功即消耗一次 take（**无论成败，算一次尝试**）
    if storyboard_id:
        consume_take(conn, storyboard_id)

    log_task_start("ImageTask", "enqueue", {
        "id": last_id,
        "provider": config.get("provider"),
        "storyboardId": storyboard_id,
        "sceneId": params.get("sceneId"),
        "characterId": params.get("characterId"),
        "frameType": params.get("frameType"),
        "model": params.get("model") or config.get("model"),
    })
    log_task_payload("ImageTask", "enqueue params", {
        "id": last_id,
        "config": {
            "provider": config.get("provider"),
            "model": config.get("model"),
            "baseUrl": config.get("baseUrl"),
        },
        "params": params,
    })

    _spawn(_run_image_generation(last_id, config))
    return last_id


async def _run_image_generation(image_id: int, config: dict[str, Any]) -> None:
    """后台包装：异常在这里落日志（等价 TS 的 ``.catch(err => …)``）。"""
    try:
        await _process_image_generation(image_id, config)
    except Exception as err:  # noqa: BLE001
        log_task_error("ImageTask", "process", {"id": image_id, "error": str(err)})
        print(f"Image generation {image_id} failed: {err}")


def _fetch_record(conn: Connection, image_id: int) -> Row | None:
    return conn.execute(
        select(image_generations).where(image_generations.c.id == image_id)
    ).first()


async def _process_image_generation(image_id: int, config: dict[str, Any]) -> None:
    """处理图片生成任务（含模型自动 fallback）。"""
    with _tx() as conn:
        record = _fetch_record(conn, image_id)
    if record is None:
        return

    resolved_reference_images = await _normalize_reference_images(_col(record, "reference_images"))

    models = list(config["models"]) if config.get("models") else [config.get("model")]

    for attempt, model in enumerate(models):
        attempt_config = {**config, "model": model}

        # 更新 DB 当前使用的模型
        with _tx() as conn:
            conn.execute(
                image_generations.update()
                .where(image_generations.c.id == image_id)
                .values(model=model, updated_at=now())
            )

        # ── 非首次尝试：释放上一次的 GPU 租约 ──
        if attempt > 0:
            release_image_gpu_lease(image_id)

        is_local = is_local_config(config.get("baseUrl") or "", config.get("provider") or "")

        # ── 本地 GPU 模型：申请显存租约（**失败只告警，不中断任务** —— 与原 TS 一致）──
        if is_local:
            try:
                _image_gpu_leases[image_id] = await gpu_manager.acquire(
                    "image", config.get("provider"), model, config.get("baseUrl")
                )
            except Exception as err:  # noqa: BLE001
                log_task_warn("ImageTask", "gpu-acquire-failed",
                              {"id": image_id, "model": model, "error": str(err)})

        # 用量记账：每次模型尝试（含 fallback）记一条 submitted，完成/失败后收口
        with _tx() as conn:
            record_usage(conn, {
                "serviceType": "image",
                "provider": config.get("provider"),
                "model": model,
                "dramaId": _col(record, "drama_id"),
                "storyboardId": _col(record, "storyboard_id"),
                "sceneId": _col(record, "scene_id"),
                "characterId": _col(record, "character_id"),
                "imageGenerationId": image_id,
                "units": 1,  # 每次调用产出 1 张图
                "isLocal": is_local,
                "status": "submitted",
                "retryCount": attempt,
                "settings": config.get("settings"),
            })

        try:
            log_task_progress("ImageTask", "build-request", {
                "id": image_id, "provider": config.get("provider"), "model": model,
                "attempt": attempt + 1,
                "storyboardId": _col(record, "storyboard_id"),
                "sceneId": _col(record, "scene_id"),
                "characterId": _col(record, "character_id"),
                "frameType": _col(record, "frame_type"),
            })

            adapter = get_image_adapter(config.get("provider"))
            request = adapter.build_generate_request(attempt_config, {
                "id": _col(record, "id"),
                "model": model,
                "prompt": _col(record, "prompt"),
                "negativePrompt": _col(record, "negative_prompt"),
                "size": _col(record, "size"),
                "frameType": _col(record, "frame_type"),
                # ⚠️ `resolved ? JSON.stringify(...) : null`：**空数组也是真值** ⇒ 无参考图时
                #    传的是 `"[]"` 而非 null。这会连带影响 Ali 是否加 seed，必须保真。
                "referenceImages": _stringify(resolved_reference_images)
                if js_truthy(resolved_reference_images)
                else None,
            })

            url = request["url"]
            method = request["method"]
            headers = request["headers"]
            body = request["body"]
            log_task_progress("ImageTask", "request", {
                "id": image_id, "provider": config.get("provider"), "method": method,
                "url": redact_url(url), "model": model,
            })
            log_task_payload("ImageTask", "request payload", {
                "id": image_id, "method": method, "url": url, "headers": headers, "body": body,
            })

            response = await fetch_with_retry(
                url,
                {
                    "method": method,
                    "headers": headers,
                    "body": _stringify(body) if body is not None else None,
                },
                "image",
                timeout_ms=600_000,
                on_retry=lambda retry_attempt, delay_ms, reason: log_task_warn(
                    "ImageTask", "request-retry",
                    {"id": image_id, "model": model, "attempt": retry_attempt,
                     "delayMs": delay_ms, "reason": reason},
                ),
            )
            result = response.json()
            log_task_payload("ImageTask", "response payload",
                             {"id": image_id, "provider": config.get("provider"), "result": result})

            parsed = adapter.parse_generate_response(result)
            is_async = parsed.get("isAsync")
            task_id = parsed.get("taskId")
            image_url = parsed.get("imageUrl")

            if not is_async and image_url:
                log_task_progress("ImageTask", "sync-complete",
                                  {"id": image_id, "model": model, "imageUrl": image_url})
                await _handle_image_complete(image_id, config.get("provider"), image_url)
                return

            if not is_async and not image_url:
                b64 = adapter.extract_image_base64(result)
                if b64:
                    log_task_progress("ImageTask", "sync-base64-complete",
                                      {"id": image_id, "model": model, "mimeType": b64["mimeType"]})
                    await _handle_image_complete_base64(
                        image_id, config.get("provider"), b64["data"], b64["mimeType"]
                    )
                    return
                raise ValueError("No image URL or base64 data in response")

            # 异步模式：更新 taskId，阻塞等待轮询完成
            with _tx() as conn:
                conn.execute(
                    image_generations.update()
                    .where(image_generations.c.id == image_id)
                    .values(task_id=task_id, status="processing", updated_at=now())
                )
            log_task_progress("ImageTask", "poll-start", {
                "id": image_id, "taskId": task_id,
                "provider": config.get("provider"), "model": model,
            })

            await _poll_image_task(image_id, attempt_config, task_id)
            return  # 轮询成功完成
        except Exception as err:  # noqa: BLE001
            is_last_attempt = attempt == len(models) - 1
            warn_meta: dict[str, Any] = {
                "id": image_id, "attempt": attempt + 1, "totalModels": len(models),
                "failedModel": model, "error": str(err),
            }
            if not is_last_attempt:
                warn_meta["nextModel"] = models[attempt + 1]
            log_task_warn("ImageTask", "all-models-failed" if is_last_attempt else "model-fallback", warn_meta)

            if is_last_attempt:
                with _tx() as conn:
                    release_image_gpu_lease(image_id)
                    _mark_usage_by_image_gen(conn, image_id, "failed")
                    log_task_error("ImageTask", "process", {
                        "id": image_id, "provider": config.get("provider"),
                        "attemptedModels": models, "error": str(err),
                    })
                    conn.execute(
                        image_generations.update()
                        .where(image_generations.c.id == image_id)
                        .values(
                            status="failed",
                            error_msg=f"All models failed. Last error: {err}",
                            updated_at=now(),
                        )
                    )
                    # 资产验收门禁：图片生成失败 → 分镜资产标记 needs_regeneration
                    _mark_storyboard_asset_needs_regeneration(conn, image_id)


async def _normalize_reference_images(raw: Any) -> list[str]:
    """归一化参考图：去空、**去重**、``static/`` 相对路径转 data URL、截断到 **6 张**。"""
    if not raw:
        return []
    try:
        refs = json.loads(raw)
    except (ValueError, TypeError):
        refs = []
    if not isinstance(refs, list):
        refs = []

    # 去重（对齐 `new Set(...)`：保持首次出现顺序）
    deduped: list[str] = []
    for item in refs:
        value = str(item or "").strip()
        if value and value not in deduped:
            deduped.append(value)

    normalized: list[str] = []
    for value in deduped:
        if value.startswith("data:image/"):
            normalized.append(value)
            continue
        if value.startswith("static/") or value.startswith("/static/"):
            local_path = value[1:] if value.startswith("/static/") else value
            try:
                normalized.append(
                    await read_image_as_compressed_data_url(
                        local_path, {"maxWidth": 768, "maxHeight": 768, "quality": 68}
                    )
                )
            except Exception as err:  # noqa: BLE001
                log_task_warn("ImageTask", "reference-read-failed",
                              {"path": local_path, "error": str(err)})
            continue
        normalized.append(value)

    return [item for item in normalized if item][:6]


async def _poll_image_task(image_id: int, config: dict[str, Any], task_id: str) -> None:
    """轮询异步图片任务直到完成或失败。

    成功时落盘并返回；**最终失败时抛错**，由外层 fallback 循环捕获后切换下一个模型。
    """
    adapter = get_image_adapter(config.get("provider"))
    started_at = _now_ms()

    for index in range(_POLL_MAX_ATTEMPTS):
        if _now_ms() - started_at >= _POLL_MAX_DURATION_MS:
            raise ValueError("Polling exceeded 10 minutes")
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        if _now_ms() - started_at >= _POLL_MAX_DURATION_MS:
            raise ValueError("Polling exceeded 10 minutes")
        try:
            poll_request = adapter.build_poll_request(config, task_id)
            url = poll_request["url"]
            method = poll_request["method"]
            headers = poll_request["headers"]
            log_task_progress("ImageTask", "poll-request", {
                "id": image_id, "taskId": task_id, "provider": config.get("provider"),
                "method": method, "url": redact_url(url), "attempt": index + 1,
            })

            remaining_ms = max(1_000, _POLL_MAX_DURATION_MS - (_now_ms() - started_at))
            # ⚠️ 轮询**不走 fetchWithRetry**（原实现如此）：失败留给下面的 except
            #    「睡一觉再试」，与 create 阶段的指数退避是两套策略。
            response = await _get_client().request(
                method, url, headers=headers, timeout=remaining_ms / 1000
            )
            if not response.is_success:
                body_text = response.text
                if is_non_retryable_http_error(response.status_code, body_text):
                    raise ValueError(
                        format_vendor_http_error(response.status_code, body_text, "image")
                    )
                continue
            result = response.json()

            poll_response = adapter.parse_poll_response(result)

            if poll_response.get("status") == "completed" and poll_response.get("imageUrl"):
                log_task_success("ImageTask", "poll-complete", {
                    "id": image_id, "taskId": task_id, "imageUrl": poll_response["imageUrl"],
                })
                await _handle_image_complete(image_id, config.get("provider"), poll_response["imageUrl"])
                return
            # Gemini 是同步的；解析出 completed 却不带 URL 时走 base64 分支
            if poll_response.get("status") == "completed" and adapter.provider == "gemini":
                b64 = adapter.extract_image_base64(result)
                if b64:
                    log_task_success("ImageTask", "poll-base64-complete",
                                     {"id": image_id, "taskId": task_id, "mimeType": b64["mimeType"]})
                    await _handle_image_complete_base64(
                        image_id, config.get("provider"), b64["data"], b64["mimeType"]
                    )
                    return
            if poll_response.get("status") == "failed":
                message = format_vendor_task_error(poll_response.get("error"), "image")
                log_task_error("ImageTask", "poll-failed",
                               {"id": image_id, "taskId": task_id, "error": message})
                raise ValueError(message)
        except Exception as err:  # noqa: BLE001
            if index == _POLL_MAX_ATTEMPTS - 1 or _now_ms() - started_at >= _POLL_MAX_DURATION_MS:
                raise  # 重新抛出给外层 fallback 循环
            log_task_warn("ImageTask", "poll-retry",
                          {"id": image_id, "taskId": task_id, "attempt": index + 1, "error": str(err)})


# ===========================================================================
# 完成后回写
# ===========================================================================

def _update_character_image(
    conn: Connection, record: Row, local_path: str, prompt: Any = None
) -> None:
    """回写角色图片。

    ⚠️ **分支顺序即优先级**（原实现如此，不要重排）：
    ``itemType`` → ``viewType`` → ``expression`` → ``equipType`` → ``costume`` → 主图。
    每个分支都是「读 JSON → 改一项 → 写回」，所以**不会覆盖同角色其它类型的图**。
    """
    character_id = _col(record, "character_id")
    if not character_id:
        return
    row = conn.execute(
        select(characters)
        .where(and_(characters.c.id == character_id, characters.c.deleted_at.is_(None)))
    ).first()
    if row is None:
        return
    char = dict(row._mapping)
    prompt = _col(record, "prompt")

    def merge(column: str, item_key: str) -> None:
        current: dict[str, Any] = {}
        try:
            parsed = json.loads(char.get(column) or "{}")
            if isinstance(parsed, dict):
                current = parsed
        except (ValueError, TypeError):
            current = {}
        current[item_key] = {
            _ITEM_KEY_FIELD[column]: item_key,
            "imageUrl": local_path,
            "prompt": prompt,
            "generatedAt": now(),
        }
        conn.execute(
            characters.update()
            .where(characters.c.id == character_id)
            .values(**{column: _stringify(current), "updated_at": now()})
        )

    # 单件高清道具图（服装/武器/首饰单品、纯物品无人物）
    if _col(record, "item_type") in ("clothing", "weapon", "accessory"):
        merge("item_images", _col(record, "item_type"))
        return

    # 三视图（combined = 正面/侧面/背面合成的一张横向长图）
    if _col(record, "view_type") in ("front", "side", "back", "combined"):
        merge("three_views", _col(record, "view_type"))
        return

    # 表情头像特写组
    if _col(record, "expression"):
        key = str(_col(record, "expression"))
        current_exp: dict[str, Any] = {}
        try:
            parsed_exp = json.loads(char.get("expressions") or "{}")
            if isinstance(parsed_exp, dict):
                current_exp = parsed_exp
        except (ValueError, TypeError):
            current_exp = {}
        current_exp[key] = {
            "key": key, "imageUrl": local_path, "prompt": prompt, "generatedAt": now(),
        }
        conn.execute(
            characters.update()
            .where(characters.c.id == character_id)
            .values(expressions=_stringify(current_exp), updated_at=now())
        )
        return

    # 装备特写（服装/武器/首饰）
    if _col(record, "equip_type") in ("clothing", "weapon", "accessory"):
        merge("equip_images", _col(record, "equip_type"))
        return

    # 服装变体：只改 variations 对应项的 imageUrl，**不覆盖主图**
    costume = _col(record, "costume")
    if costume:
        variations: list[dict[str, Any]] = []
        try:
            parsed_var = json.loads(char.get("variations") or "[]")
            if isinstance(parsed_var, list):
                variations = parsed_var
        except (ValueError, TypeError):
            variations = []
        index = next(
            (
                i
                for i, item in enumerate(variations)
                if isinstance(item, dict) and item.get("name") == costume
            ),
            -1,
        )
        if index >= 0:
            variations[index] = {**variations[index], "imageUrl": local_path}
        else:
            variations.append({"name": costume, "imageUrl": local_path})
        conn.execute(
            characters.update()
            .where(characters.c.id == character_id)
            .values(variations=_stringify(variations), updated_at=now())
        )
    else:
        conn.execute(
            characters.update()
            .where(characters.c.id == character_id)
            .values(image_url=local_path, updated_at=now())
        )


def _record_asset_version_for_generation(
    conn: Connection, record: Row, image_id: int, provider: str, final_path: str
) -> None:
    """生成成功后为关联资产留档版本历史（storyboard / character / scene / prop）。

    ⚠️ 一次生成可能同时给**多个**资产留档（例如分镜图也带 characterId）——
    原实现是 4 个独立 ``if``，不是 else-if。
    """
    meta: dict[str, Any] = {}
    for key, column in (
        ("frameType", "frame_type"),
        ("costume", "costume"),
        ("viewType", "view_type"),
        ("equipType", "equip_type"),
        ("expression", "expression"),
        ("imageType", "image_type"),
    ):
        value = _col(record, column)
        if value:
            meta[key] = value

    storyboard_id = _col(record, "storyboard_id")
    if storyboard_id:
        record_asset_version(
            conn,
            asset_type="storyboard",
            asset_id=storyboard_id,
            media_type="image",
            frame_type=resolve_storyboard_frame_type(_col(record, "frame_type")),
            asset_url=final_path,
            provider=provider,
            model=_col(record, "model"),
            prompt=_col(record, "prompt"),
            generation_id=image_id,
            meta=meta,
        )
    for asset_type, column in (
        ("character", "character_id"),
        ("scene", "scene_id"),
        ("prop", "prop_id"),
    ):
        asset_id = _col(record, column)
        if asset_id:
            record_asset_version(
                conn,
                asset_type=asset_type,
                asset_id=asset_id,
                media_type="image",
                asset_url=final_path,
                provider=provider,
                model=_col(record, "model"),
                prompt=_col(record, "prompt"),
                generation_id=image_id,
                meta=meta,
            )


def _mark_usage_by_image_gen(conn: Connection, image_generation_id: int, status: str) -> None:
    """收口某个图片生成任务的**所有** ``submitted`` 用量记录（fallback 会留下多条）。"""
    conn.execute(
        api_usage.update()
        .where(
            and_(
                api_usage.c.image_generation_id == image_generation_id,
                api_usage.c.status == "submitted",
            )
        )
        .values(status=status)
    )


async def _finalize_image(
    conn: Connection,
    record: Row,
    image_id: int,
    provider: str,
    *,
    image_url: str | None,
    local_path: str,
) -> None:
    """两个完成函数（URL / base64）的公共收尾：校色 → 更新记录 → 回写关联表 → 留档。

    ⚠️ 与原实现的差异都在这里，且**都是刻意的**：

    * ``image_url`` 只有 URL 模式才写（base64 模式不写远程 URL，与 TS 一致）；
    * 校色的 ``try/except`` 保留 —— 校色未移植会抛错，被记成 ``color-grade-failed`` 并
      **保留未校色图**，这正是 Node 校色失败时的行为。
    """
    final_path = local_path
    color_grade = _col(record, "color_grade")
    if color_grade:
        try:
            final_path = await apply_color_grade_to_file(local_path, color_grade)
        except Exception as err:  # noqa: BLE001
            log_task_warn("ImageTask", "color-grade-failed", {"id": image_id, "error": str(err)})

    values: dict[str, Any] = {"local_path": final_path, "status": "completed", "updated_at": now()}
    if image_url is not None:
        values["image_url"] = image_url
    conn.execute(image_generations.update().where(image_generations.c.id == image_id).values(**values))

    storyboard_id = _col(record, "storyboard_id")
    if storyboard_id:
        sb_update: dict[str, Any] = {"updated_at": now()}
        frame_type = _col(record, "frame_type")
        if frame_type == "first_frame":
            sb_update["first_frame_image"] = final_path
        elif frame_type == "last_frame":
            sb_update["last_frame_image"] = final_path
        elif frame_type == "keyframe":
            sb_update["keyframe_image"] = final_path
        else:
            sb_update["composed_image"] = final_path
        # 资产验收门禁：分镜首/尾帧资产生成成功即视为验收通过
        if frame_type in ("first_frame", "last_frame"):
            sb_update["asset_status"] = "approved"
        conn.execute(storyboards.update().where(storyboards.c.id == storyboard_id).values(**sb_update))

    if _col(record, "character_id"):
        _update_character_image(conn, record, final_path)
    if _col(record, "scene_id"):
        conn.execute(
            scenes.update()
            .where(scenes.c.id == _col(record, "scene_id"))
            .values(image_url=final_path, status="completed", updated_at=now())
        )
    if _col(record, "prop_id"):
        conn.execute(
            prop_templates.update()
            .where(prop_templates.c.id == _col(record, "prop_id"))
            .values(image_url=final_path, updated_at=now())
        )

    _record_asset_version_for_generation(conn, record, image_id, provider, final_path)


async def _handle_image_complete(image_id: int, provider: str, image_url: str) -> None:
    """下载远程图片并落盘，然后收尾。"""
    release_image_gpu_lease(image_id)
    with _tx() as conn:
        _mark_usage_by_image_gen(conn, image_id, "completed")
        record = _fetch_record(conn, image_id)
    local_path = await download_file(image_url, "images")
    log_task_success("ImageTask", "downloaded",
                     {"id": image_id, "provider": provider, "localPath": local_path})
    if record is None:
        return
    with _tx() as conn:
        await _finalize_image(conn, record, image_id, provider, image_url=image_url, local_path=local_path)


async def _handle_image_complete_base64(
    image_id: int, provider: str, base64_data: str, mime_type: str
) -> None:
    """保存 base64 图片，然后收尾（**不写 image_url**，与 TS 一致）。"""
    release_image_gpu_lease(image_id)
    with _tx() as conn:
        _mark_usage_by_image_gen(conn, image_id, "completed")
        record = _fetch_record(conn, image_id)
    local_path = save_base64_image(base64_data, mime_type, "images")
    log_task_success("ImageTask", "saved-base64",
                     {"id": image_id, "provider": provider, "mimeType": mime_type, "localPath": local_path})
    if record is None:
        return
    with _tx() as conn:
        await _finalize_image(conn, record, image_id, provider, image_url=None, local_path=local_path)


# ===========================================================================
# 崩溃恢复
# ===========================================================================

def _mark_image_recover_failed(conn: Connection, image_id: int, reason: str) -> None:
    """标记恢复失败（复用 process 的 failed 更新语义）。"""
    conn.execute(
        image_generations.update()
        .where(image_generations.c.id == image_id)
        .values(status="failed", error_msg=reason, updated_at=now())
    )
    log_task_error("ImageTask", "recover-failed", {"id": image_id, "reason": reason})
    _mark_storyboard_asset_needs_regeneration(conn, image_id)


def _mark_storyboard_asset_needs_regeneration(conn: Connection, image_id: int) -> None:
    """图片任务失败时把关联分镜的资产标记为 ``needs_regeneration``。

    **仅首/尾帧类任务生效** —— 角色/场景/物品图失败不影响分镜门禁。
    """
    try:
        record = _fetch_record(conn, image_id)
        if record is None or not _col(record, "storyboard_id"):
            return
        if _col(record, "frame_type") not in ("first_frame", "last_frame"):
            return
        conn.execute(
            storyboards.update()
            .where(storyboards.c.id == _col(record, "storyboard_id"))
            .values(asset_status="needs_regeneration", updated_at=now())
        )
    except Exception as err:  # noqa: BLE001
        log_task_error("ImageTask", "mark-asset-status", {"imageId": image_id, "error": str(err)})


def recover_image_tasks_on_startup() -> None:
    """服务启动时恢复被中断的图片生成任务（崩溃恢复）。

    根因：``generate_image`` 是 fire-and-forget，轮询靠内存里的循环；进程一旦重启这些
    循环直接蒸发，``status='processing'`` 的记录会**永久卡死**。

    恢复策略（**幂等，绝不重新提交**，避免按次计费的厂商重复扣费）：

    * ``updated_at`` 超阈值（60 分钟）→ 判 failed；
    * 无 ``task_id``：提交请求前就崩了 → 判 failed；
    * 未知 provider → 判 failed；
    * 其余轮询型：按记录 provider 精确匹配活跃配置，恢复轮询续跑；
    * 匹配不到配置：**保持 processing**，等配置补上后下次重启再恢复。
    """
    with _tx() as conn:
        orphans = conn.execute(
            select(image_generations).where(image_generations.c.status == "processing")
        ).all()

        if not orphans:
            return
        log_task_warn("ImageTask", "recover-start", {"count": len(orphans)})

        for row in orphans:
            try:
                if _col(row, "updated_at") and (
                    _now_ms() - _parse_iso_ms(_col(row, "updated_at")) > _RECOVER_TIMEOUT_MS
                ):
                    _mark_image_recover_failed(
                        conn,
                        _col(row, "id"),
                        f"Interrupted and expired (idle > {_RECOVER_TIMEOUT_MS // 60000}min)",
                    )
                    continue

                provider = (_col(row, "provider") or "").lower()

                if not _col(row, "task_id"):
                    _mark_image_recover_failed(
                        conn, _col(row, "id"), "Interrupted before task submit"
                    )
                    continue

                # 未知 provider 显式失败（get_image_adapter 已改为 throw，不再静默 fallback）
                if provider not in image_adapters:
                    _mark_image_recover_failed(
                        conn,
                        _col(row, "id"),
                        f"Unknown image provider: {_col(row, 'provider') or '(empty)'}",
                    )
                    continue

                config = get_active_config_by_provider(conn, "image", provider)
                if not config:
                    # 无匹配配置：保持 processing，待配置补上后下次重启再恢复
                    log_task_warn("ImageTask", "recover-no-config", {
                        "id": _col(row, "id"), "taskId": _col(row, "task_id"), "provider": provider,
                    })
                    continue
                # 用记录里当初提交的模型覆盖，保证续跑同一模型
                if _col(row, "model"):
                    config["model"] = _col(row, "model")

                log_task_warn("ImageTask", "recover-resume-poll", {
                    "id": _col(row, "id"), "taskId": _col(row, "task_id"),
                    "provider": provider, "model": _col(row, "model"),
                })
                _spawn(_resume_poll(_col(row, "id"), config, _col(row, "task_id")))
            except Exception as err:  # noqa: BLE001
                log_task_error("ImageTask", "recover-error",
                               {"id": _col(row, "id"), "error": str(err)})


async def _resume_poll(image_id: int, config: dict[str, Any], task_id: str) -> None:
    """恢复轮的续跑包装：失败时标记 failed（与 TS 的 catch 一致）。"""
    try:
        await _poll_image_task(image_id, config, task_id)
    except Exception as err:  # noqa: BLE001
        log_task_error("ImageTask", "recover-poll-failed", {"id": image_id, "error": str(err)})
        with _tx() as conn:
            _mark_image_recover_failed(conn, image_id, f"Recover poll failed: {err}")


def _parse_iso_ms(value: str) -> int:
    """ISO 时间串 → 毫秒时间戳（等价 ``new Date(s).getTime()``）。"""
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return 0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)
