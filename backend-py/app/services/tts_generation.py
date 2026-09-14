"""TTS 语音合成（移植自 ``backend/src/services/tts-generation.ts``，145 行）。

支持 MiniMax TTS（**hex 音频响应**）与 CosyVoice（本地，见 ``cosyvoice-tts`` 适配器）。

与图片/视频链路**不同**：TTS 是**同步**的（没有任务 id、没有轮询）——
调一次就拿到音频字节，直接落盘返回相对路径。

⚠️ 三个容易写错的点：

1. ``params.model || models[attempt]``：**传了 model 就等于关掉 fallback**
   （每次尝试都用同一个模型）—— 原实现如此，照实保真；
2. hex 解码用 :func:`_hex_to_bytes` 而不是 ``bytes.fromhex``：Node 的
   ``Buffer.from(s, 'hex')`` 遇到非法字符是**截断**，Python 会抛 ``ValueError``；
3. CosyVoice 零样本复用：音色库里存了参考音频时才走 ``/inference_zero_shot``，
   且**文件必须真实存在**（否则静默退回普通合成）。

⚠️ 有意的差异：**GPU 显存租约未迁**（``gpuManager.acquire('audio', …)``）。
"""
from __future__ import annotations

import base64
import json
import re
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..config import get_storage_root
from ..models import ai_voices
from .adapters.registry import get_tts_adapter
from .ai_configs import is_local_config
from .ai_providers import get_audio_config_by_id
from .gpu_manager import gpu_manager
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
from .vendor_errors import fetch_with_retry

__all__ = ["generate_tts", "generate_voice_sample"]

_HEX_PAIR_RE = re.compile(r"[0-9a-fA-F]{2}")
_STATIC_PREFIX_RE = re.compile(r"^static/")


def _hex_to_bytes(text: str) -> bytes:
    """镜像 ``Buffer.from(text, 'hex')``：**成对解析，遇到非法字符即停止**。

    ⚠️ 别用 ``bytes.fromhex``：它对奇数长度/非法字符直接抛 ``ValueError``，
    而 Node 是**静默截断**（拿到半截音频也比整个 TTS 失败好）。
    """
    out = bytearray()
    raw = text or ""
    for index in range(0, len(raw) - 1, 2):
        pair = raw[index : index + 2]
        if not _HEX_PAIR_RE.fullmatch(pair):
            break
        out.append(int(pair, 16))
    return bytes(out)


def _load_cosyvoice_zero_shot(conn: Connection, voice_id: Any) -> dict[str, str]:
    """CosyVoice 零样本复用：音色库里若存有**真实存在**的参考音频，则走 ``/inference_zero_shot``。"""
    if not voice_id:
        return {}
    row = conn.execute(
        select(ai_voices.c.reference_audio, ai_voices.c.prompt_text).where(
            ai_voices.c.voice_id == voice_id
        )
    ).first()
    if row is None or not row[0]:
        return {}
    audio_path = Path(get_storage_root()) / _STATIC_PREFIX_RE.sub("", row[0])
    if not audio_path.exists():
        return {}
    return {
        "promptAudio": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
        "promptText": row[1] or "",
    }


async def generate_tts(conn: Connection, params: dict[str, Any]) -> str:
    """生成 TTS 音频，返回**相对数据根**的路径（``static/audio/<uuid>.<fmt>``）。

    支持多模型自动 fallback：一个模型失败时自动尝试配置中的下一个模型。
    """
    config = get_audio_config_by_id(conn, params.get("configId"))
    models = list(config["models"]) if config.get("models") else [config.get("model")]
    is_local = is_local_config(config.get("baseUrl") or "", config.get("provider") or "")
    text = params.get("text") or ""

    for attempt in range(len(models)):
        # ⚠️ `params.model || models[attempt]`：传了 model 就等于每次都用它（fallback 失效）
        model = params.get("model") or models[attempt]
        attempt_config = {**config, "model": model}

        log_task_start("AudioTask", "tts-generate", {
            "provider": config.get("provider"),
            "voice": params.get("voice"),
            "model": model,
            "attempt": attempt + 1,
            "textPreview": text[:50],
            "textLength": len(text),
            "isLocal": is_local,
        })
        log_task_payload("AudioTask", "tts params", {
            "config": {
                "provider": config.get("provider"),
                "model": model,
                "baseUrl": config.get("baseUrl"),
            },
            "params": params,
        })

        # ── 本地 GPU 模型：申请显存租约 ──
        lease = None
        if is_local:
            lease = await gpu_manager.acquire("audio", config.get("provider"), model,
                                              config.get("baseUrl"))

        try:
            adapter = get_tts_adapter(config.get("provider"))

            zero_shot: dict[str, str] = {}
            if config.get("provider") == "cosyvoice":
                zero_shot = _load_cosyvoice_zero_shot(conn, params.get("voice"))

            request = adapter.build_generate_request(
                attempt_config, {**params, "model": model, **zero_shot}
            )
            url = request["url"]
            method = request["method"]
            headers = request["headers"]
            body = request["body"]
            log_task_progress("AudioTask", "request", {
                "provider": config.get("provider"), "voice": params.get("voice"),
                "method": method, "url": redact_url(url), "model": model,
            })
            log_task_payload("AudioTask", "request payload",
                             {"method": method, "url": url, "headers": headers, "body": body})

            response = await fetch_with_retry(
                url,
                {
                    "method": method,
                    "headers": headers,
                    "body": _stringify(body) if body is not None else None,
                },
                "audio",
                on_retry=lambda retry_attempt, delay_ms, reason: log_task_warn(
                    "AudioTask", "request-retry",
                    {"model": model, "attempt": retry_attempt, "delayMs": delay_ms, "reason": reason},
                ),
            )
            result = response.json()
            parsed = adapter.parse_response(result)

            # 将 hex 解码为二进制
            audio_bytes = _hex_to_bytes(parsed.get("audioHex") or "")

            # 保存到本地
            directory = Path(get_storage_root()) / "audio"
            directory.mkdir(parents=True, exist_ok=True)
            filename = f"{uuid.uuid4()}.{parsed.get('format') or 'mp3'}"
            (directory / filename).write_bytes(audio_bytes)
            relative_path = f"static/audio/{filename}"

            # 用量记账：TTS 同步成功即记 completed（**按合成字符数计费**）
            record_usage(conn, {
                "serviceType": "audio",
                "provider": config.get("provider"),
                "model": model,
                "units": len(text),
                "isLocal": is_local,
                "status": "completed",
                "settings": config.get("settings"),
            })

            log_task_success("AudioTask", "tts-saved", {
                "provider": config.get("provider"), "model": model,
                "voice": params.get("voice"), "path": relative_path,
                "bytes": len(audio_bytes), "audioMs": parsed.get("audioLength"),
            })
            return relative_path
        except Exception as err:  # noqa: BLE001
            is_last_attempt = attempt == len(models) - 1
            meta: dict[str, Any] = {
                "provider": config.get("provider"), "voice": params.get("voice"),
                "model": model, "attempt": attempt, "error": str(err),
            }
            if not is_last_attempt:
                meta["nextModel"] = models[attempt + 1]
            log_task_error("AudioTask", "all-models-failed" if is_last_attempt else "model-fallback", meta)
            if is_last_attempt:
                raise
        finally:
            # ⚠️ 与 TS 的 `finally { if (lease) lease.release() }` 一致（每轮尝试各自释放）
            if lease is not None:
                lease.release()

    raise ValueError("All TTS models failed")


async def generate_voice_sample(
    conn: Connection, character_name: str, voice_id: str, config_id: Any = None
) -> str:
    """为角色生成试听音频（固定文案）。"""
    sample_text = f"你好，我是{character_name}。很高兴认识你，这是我的声音试听。"
    return await generate_tts(conn, {"text": sample_text, "voice": voice_id, "configId": config_id})


def _stringify(value: Any) -> str:
    """等价 ``JSON.stringify``（紧凑 + 不转义非 ASCII）—— 见 ``image_generation._stringify``。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
