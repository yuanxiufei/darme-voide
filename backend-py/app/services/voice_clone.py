"""音色快速复刻（移植自 ``backend/src/services/voice-clone.ts``，135 行）。

* **MiniMax**：``POST /v1/files/upload``（上传样本）→ ``POST /v1/voice_clone``（复刻）
* **CosyVoice 2（本地）**：``POST /inference_zero_shot``，零样本即时模仿

复刻得到的 ``voice_id`` 写入音色库后，可直接用于 TTS 合成。

⚠️ 与 TTS 主链路一样：**这是「提交型」请求（create 阶段）**，所以用
``fetch_with_retry`` 且 **maxRetries=2**（比生成任务少一次）—— 上传大文件时
多试几次代价高。
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..response import js_number
from .task_logger import log_task_error, log_task_start, log_task_success
from .vendor_errors import fetch_with_retry

__all__ = ["clone_voice", "clone_voice_cosyvoice", "infer_audio_mime"]

_WAV_RE = re.compile(r"\.wav$", re.IGNORECASE)
_M4A_RE = re.compile(r"\.m4a$|\.mp4$", re.IGNORECASE)


def infer_audio_mime(filename: str) -> str:
    """根据文件名推断音频 MIME（MiniMax 支持 mp3/m4a/wav）；默认 ``audio/mpeg``。"""
    if _WAV_RE.search(filename or ""):
        return "audio/wav"
    if _M4A_RE.search(filename or ""):
        return "audio/m4a"
    return "audio/mpeg"


def _strip_trailing_slashes(base_url: str) -> str:
    return re.sub(r"/+$", "", base_url or "")


async def _upload_file_to_minimax(
    base_url: str, api_key: str, file_buffer: bytes, filename: str
) -> str:
    """上传音频样本到 MiniMax，返回 ``file_id``。"""
    url = f"{_strip_trailing_slashes(base_url)}/v1/files/upload"

    response = await fetch_with_retry(
        url,
        {"method": "POST", "headers": {"Authorization": f"Bearer {api_key}"}},
        "audio",
        max_retries=2,
        # multipart：字段名 purpose + file（Content-Type 按文件名推断）
        data={"purpose": "voice_clone"},
        files={"file": (filename, file_buffer, infer_audio_mime(filename))},
    )

    try:
        data = response.json()
    except ValueError:
        data = {}
    # ⚠️ `status_code !== 0`：缺 base_resp 时是 undefined !== 0 ⇒ **照样抛错**
    if (data.get("base_resp") or {}).get("status_code") != 0:
        raise ValueError((data.get("base_resp") or {}).get("status_msg") or "MiniMax 文件上传失败")
    file_id = (data.get("file") or {}).get("file_id")
    if not file_id:
        raise ValueError("MiniMax 未返回 file_id")
    return str(file_id)


async def clone_voice(input_data: dict[str, Any]) -> dict[str, Any]:
    """调用 MiniMax 音色快速复刻。

    ``input_data``：``baseUrl`` / ``apiKey`` / ``fileBuffer`` / ``filename`` / ``voiceId`` /
    ``voiceName?`` / ``demoText?`` / ``model?``。
    """
    voice_id = input_data.get("voiceId")
    log_task_start("VoiceClone", "clone",
                   {"voiceId": voice_id, "filename": input_data.get("filename")})
    try:
        file_id = await _upload_file_to_minimax(
            input_data.get("baseUrl"),
            input_data.get("apiKey"),
            input_data.get("fileBuffer"),
            input_data.get("filename"),
        )

        url = f"{_strip_trailing_slashes(input_data.get('baseUrl'))}/v1/voice_clone"
        body: dict[str, Any] = {
            # ⚠️ `Number(fileId)`：Node 里非数字会变 NaN，JSON.stringify 后是 `null`
            "file_id": js_number(file_id),
            "voice_id": voice_id,
            "need_noise_reduction": True,
            "need_volume_normalization": True,
        }
        if input_data.get("voiceName"):
            body["voice_name"] = input_data["voiceName"]
        if input_data.get("demoText"):
            body["text"] = input_data["demoText"]
            body["model"] = input_data.get("model") or "speech-2.8-hd"

        response = await fetch_with_retry(
            url,
            {
                "method": "POST",
                "headers": {
                    "Authorization": f"Bearer {input_data.get('apiKey')}",
                    "Content-Type": "application/json",
                },
                "body": json.dumps(body, ensure_ascii=False, separators=(",", ":")),
            },
            "audio",
            max_retries=2,
        )

        try:
            data = response.json()
        except ValueError:
            data = {}
        if (data.get("base_resp") or {}).get("status_code") != 0:
            raise ValueError((data.get("base_resp") or {}).get("status_msg") or "MiniMax 音色复刻失败")

        result = {"voiceId": voice_id, "demoAudio": data.get("demo_audio") or None}
        log_task_success("VoiceClone", "clone", {"voiceId": result["voiceId"]})
        return result
    except Exception as err:  # noqa: BLE001
        log_task_error("VoiceClone", "clone", {"error": str(err)})
        raise


async def clone_voice_cosyvoice(input_data: dict[str, Any]) -> dict[str, Any]:
    """CosyVoice 2 **零样本克隆**（本地 ``/inference_zero_shot``）。

    无需训练，传入参考音频（``prompt_audio``）+ 参考文本（``prompt_text``）即时模仿音色合成 demo。

    ⚠️ 接口约定参考 CosyVoice 官方 FastAPI 封装；本地服务部署后**需按实际接口核对**。
    """
    import base64

    prompt_text = input_data.get("promptText") or ""
    log_task_start("VoiceClone", "cosyvoice-zero-shot", {"promptText": prompt_text[:20]})
    try:
        url = f"{_strip_trailing_slashes(input_data.get('baseUrl'))}/inference_zero_shot"
        body: dict[str, Any] = {
            "tts_text": input_data.get("demoText"),
            "prompt_text": prompt_text,
            "prompt_audio": base64.b64encode(input_data.get("fileBuffer") or b"").decode("ascii"),
            "model": input_data.get("model") or "cosyvoice-v2",
            "stream": False,
            "speed": 1.0,
        }

        response = await fetch_with_retry(
            url,
            {
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(body, ensure_ascii=False, separators=(",", ":")),
            },
            "audio",
            max_retries=2,
        )

        try:
            data = response.json()
        except ValueError:
            data = {}
        audio = data.get("audio") or (data.get("data") or {}).get("audio")
        log_task_success("VoiceClone", "cosyvoice-zero-shot", {})
        return {"demoAudio": audio or None}
    except Exception as err:  # noqa: BLE001
        log_task_error("VoiceClone", "cosyvoice-zero-shot", {"error": str(err)})
        raise
