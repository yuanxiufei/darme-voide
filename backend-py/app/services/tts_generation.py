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

## 2026-09-24 接线：情绪/语速**契约**（:mod:`app.services.voice_contract` ✓）

此前情绪/语速是**每个适配器自己**处理（MiniMax ``|| 'happy'`` ✗ / CosyVoice ``|| 'neutral'`` ✗）
⇒ 两条「出片不对还查不出来」的路 ✓✗：① 表里没有的情绪**静默换成 happy** ✓；② 引擎根本不接的
参数（CosyVoice 两条路径都不转发 ``emotion``/``speed`` ✓）**静默丢掉** ✓✗。

现在**过契约**收口在 :func:`_resolve_voice_params` ✓（**提交之前** ✗）：非法**当场拒** ✓、
引擎不认的**如实报告**（``dropped`` + ``notes`` 进任务日志 ✓✗）、**不替用户默认** ✗。
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

from ..core.config import get_storage_root
from ..core.models import ai_voices
from . import voice_contract
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


def _cosyvoice_reference(conn: Connection, voice_id: Any) -> tuple[Path, Any] | None:
    """音色库里的克隆参考 ✓ ⇒ ``(参考音频路径, 参考文本)``（行在 ✓、字段有 ✓、**文件真在** ✓）。

    ⚠️ **不读文件** ✗（只判存在性 ✓）—— 所以既能当「这单是不是克隆路径 ✓」的**廉价探针** ✓，
    也是 :func:`_load_cosyvoice_zero_shot` 的唯一数据来源 ✓（两处口径**同源** ✓✗）。
    """
    if not voice_id:
        return None
    row = conn.execute(
        select(ai_voices.c.reference_audio, ai_voices.c.prompt_text).where(
            ai_voices.c.voice_id == voice_id
        )
    ).first()
    if row is None or not row[0]:
        return None
    audio_path = Path(get_storage_root()) / _STATIC_PREFIX_RE.sub("", row[0])
    if not audio_path.exists():
        return None
    return audio_path, row[1]


def _load_cosyvoice_zero_shot(conn: Connection, voice_id: Any) -> dict[str, str]:
    """CosyVoice 零样本复用：音色库里若存有**真实存在**的参考音频，则走 ``/inference_zero_shot``。"""
    found = _cosyvoice_reference(conn, voice_id)
    if found is None:
        return {}
    audio_path, prompt_text = found
    return {
        "promptAudio": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
        "promptText": prompt_text or "",
    }


def _settings_speed_range(config: dict[str, Any]) -> tuple[float, float] | None:
    """产品侧声明的语速区间 ✓（``settings.speedRange = [low, high]`` ✓）。

    ⚠️ 语速区间**只能由调用方给** ✗（好取值取决于引擎 ✓ 契约层不内置 ✗）⇒ 三道来源：
    ① 本次调用显式传 ✓、② 配置里的 ``speedRange`` ✓（这里读 ✓）、③ 适配器声明 ✓；
    **都没有 ⇒ 「区间没查」** ✗✗（只做有限性校验 ✓ 并如实报告 ✓ —— **不当通过** ✗）。
    """
    settings = config.get("settings")
    if not isinstance(settings, dict):
        return None
    raw = settings.get("speedRange")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


def _resolve_voice_params(config: dict[str, Any], params: dict[str, Any], *,
                          clone: bool = False) -> dict[str, Any]:
    """情绪/语速过契约 ✓ ⇒ 一份「做了什么 / 没做什么」的规范化参数 ✓（**发请求之前** ✗）。

    两种失败**必须分开** ✗✗：**用户给的值不合法** ⇒ 抛 :class:`VoiceContractError` ✓（当场拒 ✓，
    **不许**顺手改成 ``happy`` ✓✗）；**引擎不认这个参数** ⇒ 进任务日志的 ``dropped`` ✓
    （丢的是「这版引擎做不到」✓，不是「用户写错了」✓ —— 混在一起就查不出真因 ✓✗）。
    """
    try:
        adapter = get_tts_adapter(config.get("provider"))
    except Exception:  # noqa: BLE001 —— 未知 provider ✓：**别在这里另造一个说法** ✗
        adapter = None   # ⇒ 交给下面循环里原来的报错路径 ✓（它才是这条链的既有判据 ✓）

    try:
        plan = voice_contract.resolve_voice_params(
            params, adapter=adapter, clone=clone,
            speed_range=_settings_speed_range(config) or None,
        )
    except voice_contract.VoiceContractError as err:
        log_task_error("AudioTask", "voice-contract-rejected", {
            "provider": config.get("provider"), "voice": params.get("voice"),
            "emotion": params.get("emotion"), "speed": params.get("speed"),
            "error": str(err),
        })
        raise

    log_task_payload("AudioTask", "voice contract", plan)
    for item in plan["dropped"]:
        log_task_warn("AudioTask", "voice-contract-dropped",
                      {"provider": config.get("provider"), "voice": params.get("voice"), **item})
    for note in plan["notes"]:
        log_task_warn("AudioTask", "voice-contract-note",
                      {"provider": config.get("provider"), "note": note})

    resolved = dict(params)
    for field in ("emotion", "speed"):
        if plan[field] is None:
            # ⚠️ 「没给」与「引擎不认」都**不留字段** ✓✗ —— 留着会让适配器按自己的老写法补默认值 ✓✗
            resolved.pop(field, None)
        else:
            resolved[field] = plan[field]
    return resolved


async def generate_tts(conn: Connection, params: dict[str, Any]) -> str:
    """生成 TTS 音频，返回**相对数据根**的路径（``static/audio/<uuid>.<fmt>``）。

    支持多模型自动 fallback：一个模型失败时自动尝试配置中的下一个模型。
    """
    config = get_audio_config_by_id(conn, params.get("configId"))
    models = list(config["models"]) if config.get("models") else [config.get("model")]
    is_local = is_local_config(config.get("baseUrl") or "", config.get("provider") or "")
    text = params.get("text") or ""

    # ── ⭐ 情绪/语速过契约：**一次、且在重试循环之外** ✗（在循环里做 ⇒ 同一个非法值会被重试 N 次 ✓✗）──
    #    ⚠️ 克隆判断只做**廉价探针**（不读文件、不 base64 ✓）—— 真正的加载仍在循环里原处 ✓✗：
    #    探针只为「该不该把情绪/语速报成 dropped ✓」服务 ✓；探不出来 ⇒ 按「不是克隆」报 ✓
    #    （不在这里抛 ✗ —— 真错误留给循环里那条**既有**路径去报 ✓，别抢它的活 ✓）。
    clone = False
    if config.get("provider") == "cosyvoice":
        try:
            clone = _cosyvoice_reference(conn, params.get("voice")) is not None
        except Exception:  # noqa: BLE001 —— 探针失败**不改变**本函数的行为 ✓（下面照样会真加载 ✓）
            clone = False
    params = _resolve_voice_params(config, params, clone=clone)

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
