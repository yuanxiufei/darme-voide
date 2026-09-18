"""语音合成（TTS）适配器（逐字移植）。

来源::

    backend/src/services/adapters/minimax-tts.ts
    backend/src/services/adapters/cosyvoice-tts.ts

⚠️ 两家返回的音频**表示不同**，解析层负责归一成统一的 hex：

* MiniMax：``data.audio`` 已经是 hex 字符串；
* CosyVoice：可能是 hex，也可能是 base64 —— 靠「前 100 字符是不是纯 hex」来判，
  非纯 hex 就当 base64 解码后转 hex（Node 侧是 ``Buffer.from(raw,'base64').toString('hex')``）。
  下游统一 ``Buffer.from(audioHex, 'hex')``，所以这一步不能省。
"""
from __future__ import annotations

from typing import Any

from ...core.response import js_nullish
from .jscompat import as_dict, dig, is_pure_hex, js_base64_to_hex
from .url import join_provider_url

__all__ = [
    "CosyVoiceTTSAdapter",
    "MiniMaxTTSAdapter",
    "TTSResult",
]

#: 通用 emotion 标签 → MiniMax 平台支持的 emotion 值
_MINIMAX_EMOTION_MAP = {
    "happy": "happy",
    "joyful": "happy",
    "sad": "sad",
    "sorrowful": "sad",
    "angry": "angry",
    "furious": "angry",
    "excited": "surprised",
    "surprised": "surprised",
    "calm": "calm",
    "peaceful": "calm",
    "serious": "serious",
    "solemn": "serious",
    "fearful": "sad",
    "nervous": "sad",
    "neutral": "neutral",
}

#: 与 TS 的 ``TTSResult`` 同名（这里用 TypeDict 语义的普通 dict，仅作标注用）
TTSResult = dict[str, Any]


def map_emotion(emotion: str) -> str:
    """通用 emotion 标签 → MiniMax 平台值；未知标签回落 ``happy``。"""
    return _MINIMAX_EMOTION_MAP.get(str(emotion).lower()) or "happy"


class MiniMaxTTSAdapter:
    """MiniMax TTS（``POST /v1/t2a_v2``）。"""

    provider = "minimax"

    def build_generate_request(self, config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        url = join_provider_url(config.get("baseUrl"), "/v1", "/t2a_v2")
        headers = {
            "Authorization": f"Bearer {config.get('apiKey')}",
            "Content-Type": "application/json",
        }

        voice_setting: dict[str, Any] = {
            "voice_id": params.get("voice"),
            # ⚠️ `??` 是 **nullish**：speed 传 0 要保留 0（用 `or` 会变成 1）
            "speed": js_nullish(params.get("speed"), 1),
            "vol": 1,
            "pitch": js_nullish(params.get("pitch"), 0),
        }
        if params.get("emotion"):
            voice_setting["emotion"] = map_emotion(params["emotion"])

        body = {
            "model": params.get("model") or "speech-2.8-hd",
            "text": params.get("text"),
            "stream": False,
            "voice_setting": voice_setting,
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1,
            },
            "subtitle_enable": False,
        }

        return {"url": url, "method": "POST", "headers": headers, "body": body}

    def parse_response(self, result: Any) -> TTSResult:
        data = as_dict(result)
        # ⚠️ `status_code !== 0`：缺失 base_resp 时是 undefined !== 0 ⇒ **照样抛错**
        if dig(data, "base_resp", "status_code") != 0:
            raise ValueError(dig(data, "base_resp", "status_msg") or "TTS generation failed")

        audio = dig(data, "data", "audio")
        if not audio:
            raise ValueError("No audio data in response")

        return {
            "audioHex": audio,
            "audioLength": dig(data, "data", "extra_info", "audio_length") or 0,
            "sampleRate": dig(data, "data", "extra_info", "audio_sample_rate") or 32000,
            "bitrate": dig(data, "data", "extra_info", "bitrate") or 128000,
            "format": dig(data, "data", "extra_info", "audio_format") or "mp3",
            "channel": dig(data, "data", "extra_info", "audio_channel") or 1,
        }


class CosyVoiceTTSAdapter:
    """CosyVoice 2 本地 TTS —— ⚠️ **面向本地薄封装**（``app/local_services/cosyvoice/server.py``）。

    * 普通合成 → ``POST /tts``（JSON）
    * **带参考音频时走零样本克隆** → ``POST /inference_zero_shot``（JSON；``prompt_audio`` 是 **base64** ✓）

    ✅ **2026-09-16 已核实**（读上游 ``runtime/python/fastapi/server.py`` 源码，不再靠猜）：
    官方 FastAPI 服务**根本没有 ``/tts``** ✗，零样本那条收的是 **multipart 文件**（字段名
    ``prompt_wav``，不是 ``prompt_audio`` ✗）、返回的是**裸 int16 PCM 流**（不是 JSON 里的 ``audio`` ✗）、
    默认端口是 **50000**（不是 9880 ✗）⇒ **直接打官方服务四项都不兼容**，普通合成必然 404 ✗。

    所以本适配器的 baseUrl 应当指向**薄封装**（``PRESET_SERVICES`` 里本地音频就是
    ``http://localhost:9880`` ✓）：封装负责把这些差异**全部吸收**（JSON → form-data / multipart、
    裸 PCM → 补 44 字节 WAV 头 → hex、并如实回报 ``format="wav"`` ⇒ 下游据此定文件扩展名 ✓）。
    这条契约由 ``tests/cosyvoice_seam_test.py`` **机械守卫**（起真 HTTP stub 按官方形态收请求，
    断言字段名/文件/字节数/WAV 头，并含「包装真的转发了」的反套套逻辑 ✓）。
    """

    provider = "cosyvoice"

    def build_generate_request(self, config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        # 零样本模式：带参考音频时走 CosyVoice 官方 /inference_zero_shot
        if params.get("promptAudio"):
            return {
                "url": join_provider_url(config.get("baseUrl"), "", "/inference_zero_shot"),
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {
                    "tts_text": params.get("text"),
                    "prompt_text": params.get("promptText") or "",
                    "prompt_audio": params.get("promptAudio"),
                    "model": config.get("model") or "cosyvoice-v2",
                    "stream": False,
                    # ⚠️ nullish：speed 传 0 要保留
                    "speed": js_nullish(params.get("speed"), 1.0),
                },
            }

        return {
            "url": join_provider_url(config.get("baseUrl"), "", "/tts"),
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": {
                "text": params.get("text"),
                "voice_id": params.get("voice"),
                "speed": js_nullish(params.get("speed"), 1.0),
                # ⚠️ 这个用的是 `||`（truthy）：emotion 传空串要回退 'neutral'
                "emotion": params.get("emotion") or "neutral",
                "model": config.get("model") or "cosyvoice-v2",
                "format": "mp3",
                "sample_rate": 32000,
            },
        }

    def parse_response(self, result: Any) -> TTSResult:
        data = as_dict(result)
        raw = data.get("audio") or dig(data, "data", "audio") or ""
        if not raw:
            raise ValueError("CosyVoice 未返回音频数据")

        # 返回的可能是 base64，自动转 hex（兼容下游 Buffer.from(audioHex, 'hex')）
        raw_text = str(raw)
        audio_hex = raw_text
        if not is_pure_hex(raw_text[:100]):
            audio_hex = js_base64_to_hex(raw_text)

        return {
            "audioHex": audio_hex,
            "audioLength": data.get("audio_length") or data.get("duration") or 0,
            "sampleRate": data.get("sample_rate") or 32000,
            "bitrate": data.get("bitrate") or 128000,
            "format": data.get("format") or "mp3",
            "channel": data.get("channel") or 1,
        }
