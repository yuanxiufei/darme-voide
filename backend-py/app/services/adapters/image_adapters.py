"""图片生成适配器（逐字移植）。

来源（一一对应）::

    backend/src/services/adapters/minimax-image.ts
    backend/src/services/adapters/openai-image.ts
    backend/src/services/adapters/gemini-image.ts
    backend/src/services/adapters/volcengine-image.ts
    backend/src/services/adapters/ali-image.ts
    backend/src/services/adapters/local-sd-image.ts

**整层是纯函数**：这些适配器只「构建请求描述」与「解析响应」，
真正发 HTTP 的是调用方（``image-generation.ts`` 那一层）。
所以本层可以逐字比对、全量单测，不需要任何网络或密钥。

保真说明
--------
* ``body: undefined`` 统一写成 ``body: None`` —— 调用方按「无请求体」处理，与 TS 等价。
* JS 里**显式写 ``undefined``** 的键（如 Ali 的 ``seed``）会被 ``JSON.stringify`` 丢掉，
  这里就**不加这个键**，而不是放 ``None``（否则会变成 JSON ``null`` 发给厂商）。
  来自 DB 行的可空字段（``null``）则如实保留为 ``None``。
"""
from __future__ import annotations

import json
from typing import Any

from ...core.response import js_number, js_truthy
from ..file_storage import parse_data_url
from .jscompat import (
    add_query_param,
    as_dict,
    dig,
    gcd,
    js_json_stringify,
    js_length,
    js_num_str,
    js_parse_int,
    random_seed,
    split_xy,
)
from .url import join_provider_url

__all__ = [
    "AliImageAdapter",
    "GeminiImageAdapter",
    "LocalSDImageAdapter",
    "MiniMaxImageAdapter",
    "OpenAIImageAdapter",
    "VolcEngineImageAdapter",
]

_JSON_MIME = "application/json"


def _bearer(api_key: str) -> dict[str, str]:
    return {"Content-Type": _JSON_MIME, "Authorization": f"Bearer {api_key}"}


def _json_array_or_none(raw: Any) -> Any:
    """``JSON.parse(raw)``；解析失败返回 None（原 TS 的 ``catch {}`` 静默分支）。"""
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


class MiniMaxImageAdapter:
    """MiniMax 图片生成（API 风格与 OpenAI 兼容，零改动）。"""

    provider = "minimax"

    def build_generate_request(self, config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": record.get("model") or config.get("model"),
            "prompt": record.get("prompt"),
            "size": record.get("size") or "1920x1080",
            "n": 1,
        }

        if record.get("negativePrompt"):
            body["negative_prompt"] = record["negativePrompt"]

        # MiniMax 支持 reference_images（参考图）
        if record.get("referenceImages"):
            refs = _json_array_or_none(record["referenceImages"])
            # 原 TS 判的是 `refs.length > 0`：字符串按字符数、对象与数字为 undefined（假）
            if refs is not None and (js_length(refs) or 0) > 0:
                body["image"] = refs

        # aspect_ratio 参数（MiniMax 支持）
        if record.get("size"):
            width, height = split_xy(record["size"])
            if width and height:
                body["aspect_ratio"] = f"{width}/{height}"

        return {
            "url": join_provider_url(config.get("baseUrl"), "/v1", "/image_generation"),
            "method": "POST",
            "headers": _bearer(config.get("apiKey")),
            "body": body,
        }

    def parse_generate_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        # 异步模式：返回 task_id
        task_id = data.get("task_id") or data.get("id")
        if task_id:
            return {"isAsync": True, "taskId": task_id}
        # 同步模式：直接返回图片 URL
        image_url = dig(data, "data", 0, "url") or data.get("url")
        if image_url:
            return {"isAsync": False, "imageUrl": image_url}
        raise ValueError("No image URL or task_id in response")

    def build_poll_request(self, config: dict[str, Any], task_id: str) -> dict[str, Any]:
        return {
            "url": join_provider_url(config.get("baseUrl"), "/v1", f"/image_generation/task/{task_id}"),
            "method": "GET",
            "headers": {"Authorization": f"Bearer {config.get('apiKey')}"},
            "body": None,
        }

    def parse_poll_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        status = data.get("status") or data.get("state")
        if status in ("completed", "succeeded"):
            return {
                "status": "completed",
                "imageUrl": data.get("image_url")
                or dig(data, "data", "image_url")
                or data.get("url")
                or dig(data, "data", "url"),
            }
        if status in ("failed", "error"):
            return {"status": "failed", "error": data.get("error_msg") or data.get("error") or "Generation failed"}
        return {"status": status or "processing"}

    def extract_image_url(self, result: Any) -> str | None:
        return (
            dig(result, "image_url")
            or dig(result, "data", "image_url")
            or dig(result, "url")
            or dig(result, "data", "url")
            or None
        )

    def extract_image_base64(self, result: Any) -> dict[str, str] | None:
        # MiniMax 通常返回 URL，不返回 base64
        return None


class OpenAIImageAdapter:
    """OpenAI DALL-E 图片生成。

    端点 ``/v1/images/generations``；
    响应 ``{data:[{url}]}`` 或 ``{data:[{b64_json}]}``。
    """

    provider = "openai"

    def build_generate_request(self, config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        size = record.get("size") or "1024x1024"
        body: dict[str, Any] = {
            "model": record.get("model") or "dall-e-3",
            "prompt": record.get("prompt"),
            "size": size,
            "n": 1,
            "response_format": "url",  # 默认返回 URL，可选 'b64_json'
        }
        return {
            "url": join_provider_url(config.get("baseUrl"), "/v1", "/images/generations"),
            "method": "POST",
            "headers": _bearer(config.get("apiKey")),
            "body": body,
        }

    def parse_generate_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        # OpenAI DALL-E 3 目前是同步返回，但规范上也有异步 task 模式
        task_id = data.get("task_id") or data.get("id")
        if task_id:
            return {"isAsync": True, "taskId": task_id}
        image_url = dig(data, "data", 0, "url") or data.get("url")
        if image_url:
            return {"isAsync": False, "imageUrl": image_url}
        # b64_json 模式：标记完成但不给 URL，真正的数据在 extract_image_base64 里取
        if dig(data, "data", 0, "b64_json"):
            return {"isAsync": False, "imageUrl": None}
        raise ValueError("No image URL in response")

    def build_poll_request(self, config: dict[str, Any], task_id: str) -> dict[str, Any]:
        return {
            "url": join_provider_url(config.get("baseUrl"), "/v1", f"/images/task/{task_id}"),
            "method": "GET",
            "headers": {"Authorization": f"Bearer {config.get('apiKey')}"},
            "body": None,
        }

    def parse_poll_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        if data.get("status") == "completed":
            return {
                "status": "completed",
                "imageUrl": data.get("image_url") or dig(data, "data", 0, "url") or None,
            }
        if data.get("status") == "failed":
            return {"status": "failed", "error": dig(data, "error", "message") or "Generation failed"}
        return {"status": data.get("status") or "processing"}

    def extract_image_url(self, result: Any) -> str | None:
        return dig(result, "data", 0, "url") or dig(result, "image_url") or None

    def extract_image_base64(self, result: Any) -> dict[str, str] | None:
        b64 = dig(result, "data", 0, "b64_json")
        if b64:
            return {"data": b64, "mimeType": "image/png"}
        return None


#: Gemini 图片生成被内容安全拒绝时可能出现的 finishReason（需归因成可行动的中文）
_GEMINI_SAFETY_REASONS = frozenset({
    "SAFETY",
    "IMAGE_SAFETY",
    "SPII",
    "RECITATION",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "SEXUALLY_EXPLICIT",
    "HATE_SPEECH",
    "DANGEROUS_CONTENT",
})

_GEMINI_SAFETY_MESSAGE = (
    "图片生成被内容安全拦截：Gemini 判定该提示词或参考图触发了安全策略。"
    "请修改关键帧/角色提示词或更换参考图后重试。"
)


class GeminiImageAdapter:
    """Gemini 图片生成（Google REST 风格）。

    * 认证**同时**给出两种方式：URL ``?key=`` 与 Header（``x-goog-api-key`` + ``Authorization``）
    * 请求体是 ``contents[].parts[]`` 结构
    * 响应是 **base64**（``inlineData.data``），没有 URL
    """

    provider = "gemini"

    def build_generate_request(self, config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        # 模型名允许写成 "models/xxx" 或直接 "xxx"，这里统一补前缀
        model_name = record.get("model") or config.get("model") or "gemini-2.5-flash-image"
        model = model_name if str(model_name).startswith("models/") else f"models/{model_name}"

        parts: list[dict[str, Any]] = []
        if record.get("referenceImages"):
            refs = _json_array_or_none(record["referenceImages"])
            # 原 TS 是 `for (const ref of refs)`：不可迭代会抛错并被 catch 吞掉（等于不处理）
            if isinstance(refs, (list, str)):
                for ref in refs:
                    parsed = parse_data_url(str(ref or ""))
                    if parsed:
                        parts.append({
                            "inline_data": {
                                "mime_type": parsed["mimeType"],
                                "data": parsed["data"],
                            },
                        })
        parts.append({"text": record.get("prompt") or "Generate an image"})

        body = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
                "imageConfig": {
                    "aspectRatio": self._parse_aspect_ratio(record.get("size")),
                    "imageSize": self._parse_image_size(record.get("size")),
                },
            },
        }

        url = add_query_param(
            join_provider_url(config.get("baseUrl"), "/v1beta", f"/{model}:generateContent"),
            "key",
            config.get("apiKey"),
        )
        return {
            "url": url,
            "method": "POST",
            "headers": {
                "Content-Type": _JSON_MIME,
                "x-goog-api-key": config.get("apiKey"),
                "Authorization": f"Bearer {config.get('apiKey')}",
            },
            "body": body,
        }

    def parse_generate_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        first_candidate = dig(data, "candidates", 0)
        finish_reason = dig(first_candidate, "finishReason") or dig(first_candidate, "finish_reason")
        finish_message = dig(first_candidate, "finishMessage") or dig(first_candidate, "finish_message")

        if finish_reason and finish_reason != "STOP" and finish_reason != "MAX_TOKENS":
            # 因安全策略被拒时 finishMessage 是英文技术描述，归因成可行动的中文
            reason = str(finish_reason).upper()
            if reason in _GEMINI_SAFETY_REASONS:
                raise ValueError(_GEMINI_SAFETY_MESSAGE)
            raise ValueError(finish_message or f"Gemini generation stopped: {finish_reason}")

        # ⚠️ 这里的判定顺序与其它厂商**不同**：先 URL → 再 base64 → 再 task_id → 再 error
        image_url = self.extract_image_url(data)
        if image_url:
            return {"isAsync": False, "imageUrl": image_url}
        if self.extract_image_base64(data):
            return {"isAsync": False, "imageUrl": None}

        task_id = data.get("task_id") or data.get("id")
        if task_id:
            return {"isAsync": True, "taskId": task_id}

        if data.get("error"):
            raise ValueError(dig(data, "error", "message") or "Gemini generation failed")
        raise ValueError("No image data in Gemini response")

    def parse_poll_response(self, result: Any) -> dict[str, Any]:
        # Gemini 是同步的，通常不会走到这里
        return {"status": "completed"}

    def build_poll_request(self, config: dict[str, Any], task_id: str) -> dict[str, Any]:
        # Gemini 不需要轮询，但实现接口以保持一致
        url = add_query_param(
            join_provider_url(config.get("baseUrl"), "/v1beta", f"/{task_id}"), "key", config.get("apiKey")
        )
        return {
            "url": url,
            "method": "GET",
            "headers": {
                "x-goog-api-key": config.get("apiKey"),
                "Authorization": f"Bearer {config.get('apiKey')}",
            },
            "body": None,
        }

    def extract_image_url(self, result: Any) -> str | None:
        return (
            dig(result, "data", 0, "url")
            or dig(result, "image_url")
            or dig(result, "url")
            or None
        )

    def extract_image_base64(self, result: Any) -> dict[str, str] | None:
        data = as_dict(result)
        b64 = dig(data, "data", 0, "b64_json")
        if b64:
            return {"data": b64, "mimeType": "image/png"}

        parts = dig(data, "candidates", 0, "content", "parts") or []
        if not isinstance(parts, list):
            return None
        for part in parts:
            inline = dig(part, "inlineData") or dig(part, "inline_data")
            if inline:
                return {
                    "data": inline.get("data"),
                    "mimeType": inline.get("mimeType") or inline.get("mime_type") or "image/png",
                }
        return None

    @staticmethod
    def _parse_aspect_ratio(size: Any = None) -> str:
        if not size:
            return "16:9"
        width_raw, height_raw = split_xy(size)
        width, height = js_number(width_raw), js_number(height_raw)
        if not width or not height:
            return "16:9"
        divisor = gcd(width, height)
        return f"{js_num_str(width / divisor)}:{js_num_str(height / divisor)}"

    @staticmethod
    def _parse_image_size(size: Any = None) -> str:
        if not size:
            return "1K"
        width_raw, _ = split_xy(size)
        width = js_number(width_raw)
        if not width:
            return "1K"
        if width >= 2048:
            return "4K"
        if width >= 1024:
            return "2K"
        if width >= 512:
            return "1K"
        return "512"


class VolcEngineImageAdapter:
    """火山引擎 veImageX 图片生成（端点 ``/api/v3/images/generations``）。"""

    provider = "volcengine"

    def build_generate_request(self, config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        model = record.get("model") or config.get("model") or "doubao-seedream-5-0-lite"
        body: dict[str, Any] = {"model": model, "prompt": record.get("prompt")}

        if record.get("negativePrompt"):
            body["negative_prompt"] = record["negativePrompt"]

        # Seedream 参考图锚定：record.referenceImages 是 normalize 后的 JSON 数组（URL 或 data URL）
        if record.get("referenceImages"):
            refs = _json_array_or_none(record["referenceImages"])
            if isinstance(refs, list) and len(refs) > 0:
                body["image"] = refs

        # 尺寸参数
        if record.get("size"):
            width, height = split_xy(record["size"])
            if width and height:
                body["width"] = js_parse_int(width)
                body["height"] = js_parse_int(height)

        return {
            "url": join_provider_url(config.get("baseUrl"), "/api/v3", "/images/generations"),
            "method": "POST",
            "headers": _bearer(config.get("apiKey")),
            "body": body,
        }

    def parse_generate_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        # 火山引擎可能返回 task_id 进行轮询
        task_id = data.get("task_id") or data.get("id")
        if task_id:
            return {"isAsync": True, "taskId": task_id}
        image_url = dig(data, "data", 0, "url") or data.get("url")
        if image_url:
            return {"isAsync": False, "imageUrl": image_url}
        raise ValueError("No image URL in response")

    def build_poll_request(self, config: dict[str, Any], task_id: str) -> dict[str, Any]:
        return {
            "url": join_provider_url(config.get("baseUrl"), "/api/v3", f"/images/generations/{task_id}"),
            "method": "GET",
            "headers": {"Authorization": f"Bearer {config.get('apiKey')}"},
            "body": None,
        }

    def parse_poll_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        status = data.get("status")
        if status == "succeeded":
            return {"status": "completed", "imageUrl": dig(data, "data", 0, "url") or data.get("image_url")}
        if status == "failed":
            return {"status": "failed", "error": data.get("error") or "Generation failed"}
        return {"status": status or "processing"}

    def extract_image_url(self, result: Any) -> str | None:
        return dig(result, "data", 0, "url") or dig(result, "image_url") or None

    def extract_image_base64(self, result: Any) -> dict[str, str] | None:
        return None


class AliImageAdapter:
    """阿里云百炼（万相）图片生成。"""

    provider = "ali"

    _DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"

    def build_generate_request(self, config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        base_url = config.get("baseUrl") or self._DEFAULT_BASE_URL
        url = join_provider_url(base_url, "/api/v1", "/services/aigc/image-generation/generation")

        headers = {
            "Authorization": f"Bearer {config.get('apiKey')}",
            "Content-Type": _JSON_MIME,
            "X-DashScope-Async": "enable",
        }

        # 解析 size 参数（如 "1920x1080" → "1696*960"）
        size = self._normalize_size(record.get("size") or "1280*1280")

        parameters: dict[str, Any] = {
            "size": size,
            "n": 1,
            "negative_prompt": record.get("negativePrompt") or "",
            "prompt_extend": True,
            "watermark": False,
        }
        # ⚠️ 有参考图时 seed 在 TS 里是**字面 undefined**，会被 JSON.stringify 丢掉 ⇒ 这里不加键
        if not record.get("referenceImages"):
            parameters["seed"] = random_seed()

        body = {
            "model": record.get("model") or "wan2.6-t2i",
            "input": {"messages": [{"role": "user", "content": [{"text": record.get("prompt")}]}]},
            "parameters": parameters,
        }

        return {"url": url, "method": "POST", "headers": headers, "body": body}

    def parse_generate_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        # PENDING 表示异步任务已创建
        if dig(data, "output", "task_status") == "PENDING" and dig(data, "output", "task_id"):
            return {"isAsync": True, "taskId": dig(data, "output", "task_id")}

        # 同步模式
        image = dig(data, "output", "choices", 0, "message", "content", 0, "image")
        if image:
            return {"isAsync": False, "imageUrl": image}

        raise ValueError(
            f"Unexpected Ali image response: {js_json_stringify(result)[:200]}"
        )

    def build_poll_request(self, config: dict[str, Any], task_id: str) -> dict[str, Any]:
        base_url = config.get("baseUrl") or self._DEFAULT_BASE_URL
        return {
            "url": join_provider_url(base_url, "/api/v1", f"/tasks/{task_id}"),
            "method": "GET",
            "headers": {
                "Authorization": f"Bearer {config.get('apiKey')}",
                "Content-Type": _JSON_MIME,
            },
            "body": None,
        }

    def parse_poll_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        status = dig(data, "output", "task_status")
        if status == "SUCCEEDED":
            return {
                "status": "completed",
                "imageUrl": dig(data, "output", "choices", 0, "message", "content", 0, "image"),
            }
        if status == "FAILED":
            return {"status": "failed", "error": data.get("message") or "Generation failed"}
        if status in ("PENDING", "RUNNING"):
            return {"status": "processing"}
        return {"status": "pending"}

    def extract_image_base64(self, result: Any) -> dict[str, str] | None:
        # Ali 目前不支持直接返回 base64
        return None

    def extract_image_url(self, result: Any) -> str | None:
        return dig(result, "output", "choices", 0, "message", "content", 0, "image") or None

    @staticmethod
    def _normalize_size(size: str) -> str:
        """把 ``"1920x1080"`` 转换成阿里需要的 ``"1696*960"`` 格式。"""
        width_raw, height_raw = split_xy(size)
        width, height = js_number(width_raw), js_number(height_raw)
        if width and height:
            aspect = width / height
            if aspect > 1.7:
                return "1696*960"  # 16:9
            if aspect < 0.8:
                return "960*1696"  # 9:16
            return "1280*1280"  # 1:1
        return "1280*1280"


class LocalSDImageAdapter:
    """本地 Stable Diffusion（SD WebUI API）。

    ``POST /sdapi/v1/txt2img`` 或 ``/sdapi/v1/img2img``；**同步返回 base64**，不轮询。
    """

    provider = "local-sd"

    _DEFAULT_NEGATIVE = (
        "low quality, blurry, text, watermark, distorted face, deformed, bad anatomy, disfigured"
    )

    def build_generate_request(self, config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        prompt = record.get("prompt") or ""
        negative = record.get("negativePrompt") or self._DEFAULT_NEGATIVE

        width_raw, height_raw = split_xy(record.get("size") or "1024x1024")
        width = js_number(width_raw) or 1024
        height = js_number(height_raw) or 1024

        body: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": 25,
            "width": width,
            "height": height,
            "cfg_scale": 7,
            "sampler_name": "DPM++ 2M Karras",
            "seed": -1,
        }

        # 有参考图 → img2img 模式
        endpoint = "/sdapi/v1/txt2img"
        if record.get("referenceImages"):
            refs = _json_array_or_none(record["referenceImages"])
            if isinstance(refs, list) and len(refs) > 0:
                body["init_images"] = [refs[0]]
                body["denoising_strength"] = 0.55
                endpoint = "/sdapi/v1/img2img"

        return {
            "url": join_provider_url(config.get("baseUrl"), "", endpoint),
            "method": "POST",
            "headers": {"Content-Type": _JSON_MIME},
            "body": body,
        }

    def parse_generate_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        images = data.get("images")
        # 原 TS：`if (!result.images || result.images.length === 0)`
        # ⚠️ 空数组/空对象在 JS 是**真值**，所以必须走 js_truthy + length 两个判定，
        #    写 Python 的 `not images` 会让 `{}` 与 `[]` 都误判成「没返回」
        if not js_truthy(images) or js_length(images) == 0:
            raise ValueError("SD 未返回图片")
        # SD 同步返回 base64，直接标记完成
        return {"isAsync": False}

    def build_poll_request(self, config: dict[str, Any], task_id: str) -> dict[str, Any]:
        raise ValueError("SD WebUI 同步模式，无需轮询")

    def parse_poll_response(self, result: Any) -> dict[str, Any]:
        raise ValueError("SD WebUI 同步模式，无需轮询")

    def extract_image_url(self, result: Any) -> str | None:
        # SD 返回的是 base64，不是 URL
        return None

    def extract_image_base64(self, result: Any) -> dict[str, str] | None:
        b64 = dig(result, "images", 0)
        if not b64 or not isinstance(b64, str):
            return None
        return {"data": b64, "mimeType": "image/png"}
