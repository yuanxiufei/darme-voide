"""视频生成适配器（逐字移植）。

来源（一一对应）::

    backend/src/services/adapters/minimax-video.ts
    backend/src/services/adapters/volcengine-video.ts
    backend/src/services/adapters/vidu-video.ts
    backend/src/services/adapters/ali-video.ts

与图片层同样是**纯函数**：只构建请求描述与解析响应。

各家的参考图字段差异很大（这是本层存在的理由）：

===========  =========================================================
厂商          参考图 / 音频承载方式
===========  =========================================================
MiniMax      ``first_frame_image`` / ``last_frame_image`` /
             ``subject_reference``（多参考图，≤9）/ ``reference_audio``（≤3）
volcengine   ``content[]`` 数组里逐个 ``{type:'image_url'}``，首尾帧带 ``role``
vidu         ``images[]`` 纯 URL 数组；**不提供轮询接口**，靠 Webhook 回调
ali          ``input.img_url`` / ``input.last_img_url``
===========  =========================================================
"""
from __future__ import annotations

import json
import re
from typing import Any

from ...core.response import js_number, js_round
from .jscompat import as_dict, dig, js_json_stringify, random_seed
from .url import join_provider_url

__all__ = [
    "AliVideoAdapter",
    "MiniMaxVideoAdapter",
    "ViduVideoAdapter",
    "VolcEngineVideoAdapter",
    "parse_callback_state",
]

_JSON_MIME = "application/json"


def _json_or_none(raw: Any) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def _iterate_like_js(value: Any) -> list[Any]:
    """镜像 JS 的 ``for (const x of value)``。

    数组按元素、**字符串按字符**（JS 真的会这样）；其它类型会抛错并被调用方的
    ``catch {}`` 吞掉 ⇒ 这里返回空列表（净效果一致）。
    """
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        return list(value)
    return []


class MiniMaxVideoAdapter:
    """MiniMax H3 视频生成（原生 API 格式，prompt 是纯字符串）。"""

    provider = "minimax"

    def build_generate_request(self, config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": resolve_h3_checkpoint(config, record),
            "prompt": record.get("prompt") or "",
            "aspect_ratio": record.get("aspectRatio") or "16:9",
            "duration": record.get("duration") or 5,
        }

        # 参考图：MiniMax 原生字段 first_frame_image / last_frame_image
        reference_mode = record.get("referenceMode")
        if reference_mode == "single" and record.get("imageUrl"):
            body["first_frame_image"] = record["imageUrl"]
        elif reference_mode == "first_last":
            if record.get("firstFrameUrl"):
                body["first_frame_image"] = record["firstFrameUrl"]
            if record.get("lastFrameUrl"):
                body["last_frame_image"] = record["lastFrameUrl"]
        elif reference_mode == "multiple" and record.get("referenceImageUrls"):
            # Ref2VA：多参考图（角色立绘 + 场景图）用 subject_reference 锁定主体
            refs = _json_or_none(record["referenceImageUrls"])
            if isinstance(refs, list) and len(refs) > 0:
                body["subject_reference"] = [str(x) for x in refs][:9]

        # 参考音频（H3 音视频联合生成 / Ref2VA reference conditioning）：角色声线样本，最多 3 条
        # 字段名 reference_audio 是本地 H3 服务的约定；云端 MiniMax 不支持时会忽略该字段
        if record.get("referenceAudioUrls"):
            audios = _json_or_none(record["referenceAudioUrls"])
            if isinstance(audios, list) and len(audios) > 0:
                body["reference_audio"] = [str(x) for x in audios][:3]

        return {
            "url": join_provider_url(config.get("baseUrl"), "/v1", "/video_generation"),
            "method": "POST",
            "headers": {"Content-Type": _JSON_MIME, "Authorization": f"Bearer {config.get('apiKey')}"},
            "body": body,
        }

    def parse_generate_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        task_id = data.get("task_id") or data.get("id") or dig(data, "data", "id")
        if not task_id:
            # 同步返回
            video_url = (
                data.get("video_url") or dig(data, "data", "video_url") or dig(data, "content", "video_url")
            )
            if video_url:
                return {"isAsync": False, "videoUrl": video_url}
            raise ValueError("No task_id or video_url in response")
        return {"isAsync": True, "taskId": task_id}

    def build_poll_request(self, config: dict[str, Any], task_id: str) -> dict[str, Any]:
        return {
            "url": join_provider_url(config.get("baseUrl"), "/v1", f"/video_generation/task/{task_id}"),
            "method": "GET",
            "headers": {"Authorization": f"Bearer {config.get('apiKey')}"},
            "body": None,
        }

    def parse_poll_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        status = data.get("status") or data.get("state") or dig(data, "data", "status")
        if status in ("completed", "succeeded"):
            return {
                "status": "completed",
                "videoUrl": data.get("video_url")
                or dig(data, "data", "video_url")
                or dig(data, "content", "video_url"),
            }
        if status in ("failed", "error"):
            return {
                "status": "failed",
                "error": data.get("error_msg") or data.get("error") or "Video generation failed",
            }
        return {"status": status or "processing"}

    def extract_video_url(self, result: Any) -> str | None:
        return (
            dig(result, "video_url")
            or dig(result, "data", "video_url")
            or dig(result, "content", "video_url")
            or None
        )


def resolve_h3_checkpoint(config: dict[str, Any], record: dict[str, Any]) -> str:
    """H3 双 checkpoint 路由：按 ``scene_type`` 选 FL2VA / Ref2VA。

    需在 ``ai_service_configs.settings`` 里声明 ``checkpoint_map``，例如::

        {"checkpoint_map": {"fl2va": "...", "ref2va": "..."}}

    未声明时退化为 ``record.model / config.model``（单 checkpoint 模式，云端亦然）。
    """
    base = record.get("model") or config.get("model")
    settings = config.get("settings")
    checkpoint_map = dig(settings, "checkpoint_map")
    if not isinstance(checkpoint_map, dict):
        return base

    scene_type = (record.get("sceneType") or "").lower()
    # FL2VA（图生视频/首尾帧）：动作 / 静默 / 空镜 / 转场；其余走 Ref2VA
    is_fl2va = re.search(r"action|silent|transition|establishing|empty", scene_type) is not None
    key = "fl2va" if is_fl2va else "ref2va"
    mapped = checkpoint_map.get(key) or checkpoint_map.get("fl2va") or checkpoint_map.get("ref2va")
    return mapped if isinstance(mapped, str) and mapped else base


class VolcEngineVideoAdapter:
    """火山引擎 Seedance 视频生成（端点 ``/api/v3/contents/generations/tasks``）。"""

    provider = "volcengine"

    def build_generate_request(self, config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        model = record.get("model") or config.get("model") or "doubao-seedance-1-5-pro-251215"
        content: list[dict[str, Any]] = [{"type": "text", "text": record.get("prompt") or ""}]

        reference_mode = record.get("referenceMode")
        if reference_mode == "single" and record.get("imageUrl"):
            content.append({"type": "image_url", "image_url": {"url": record["imageUrl"]}})
        elif reference_mode == "first_last":
            if record.get("firstFrameUrl"):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": record["firstFrameUrl"]},
                    "role": "first_frame",
                })
            if record.get("lastFrameUrl"):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": record["lastFrameUrl"]},
                    "role": "last_frame",
                })
        elif reference_mode == "multiple" and record.get("referenceImageUrls"):
            refs = _json_or_none(record["referenceImageUrls"])
            for url in _iterate_like_js(refs):
                content.append({"type": "image_url", "image_url": {"url": url}})

        body: dict[str, Any] = {
            "model": model,
            "content": content,
            "generate_audio": True,
            "ratio": record.get("aspectRatio") or "adaptive",
            "duration": self._normalize_duration(record.get("duration")),
            "watermark": False,
        }
        if record.get("negativePrompt"):
            body["negative_prompt"] = record["negativePrompt"]

        return {
            "url": join_provider_url(config.get("baseUrl"), "/api/v3", "/contents/generations/tasks"),
            "method": "POST",
            "headers": {"Content-Type": _JSON_MIME, "Authorization": f"Bearer {config.get('apiKey')}"},
            "body": body,
        }

    def parse_generate_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        if data.get("id"):
            return {"isAsync": True, "taskId": data["id"]}
        video_url = data.get("video_url") or dig(data, "content", "video_url") or dig(data, "data", "video_url")
        if video_url:
            return {"isAsync": False, "videoUrl": video_url}
        raise ValueError("No task_id or video_url in response")

    def build_poll_request(self, config: dict[str, Any], task_id: str) -> dict[str, Any]:
        return {
            "url": join_provider_url(config.get("baseUrl"), "/api/v3", f"/contents/generations/tasks/{task_id}"),
            "method": "GET",
            "headers": {"Authorization": f"Bearer {config.get('apiKey')}"},
            "body": None,
        }

    def parse_poll_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        status = data.get("status")
        if status == "succeeded":
            return {
                "status": "completed",
                "videoUrl": data.get("video_url")
                or dig(data, "content", "video_url")
                or dig(data, "data", "video_url"),
            }
        if status == "failed":
            return {"status": "failed", "error": data.get("error") or "Video generation failed"}
        return {"status": status or "processing"}

    def extract_video_url(self, result: Any) -> str | None:
        return (
            dig(result, "video_url")
            or dig(result, "content", "video_url")
            or dig(result, "data", "video_url")
            or None
        )

    @staticmethod
    def _normalize_duration(duration: Any = None) -> int:
        """``Math.round(Number(duration || 5))`` 后夹到 ``[4, 12]``；非有限值回落 5。"""
        value = js_number(duration or 5)
        if value is None:  # 原 JS 的 `!Number.isFinite(parsed)`
            return 5
        parsed = js_round(value)
        return min(12, max(4, parsed))


class ViduVideoAdapter:
    """Vidu 视频生成。

    ⚠️ 两个反直觉之处（都是原样保留）：

    1. 认证头是 ``Authorization: Token {apiKey}``，**不是 Bearer**；
    2. **不提供轮询接口** —— ``build_poll_request`` 返回一个不可达的
       ``vidu://`` 伪 URL，``parse_poll_response`` 永远回 ``processing``，
       真实状态由 Webhook 回调（``parse_callback_state``）推进。
    """

    provider = "vidu"

    def build_generate_request(self, config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        model = record.get("model") or config.get("model") or "viduq3-turbo"
        body: dict[str, Any] = {"model": model, "images": [], "prompt": record.get("prompt")}

        if record.get("negativePrompt"):
            body["negative_prompt"] = record["negativePrompt"]

        reference_mode = record.get("referenceMode")
        if reference_mode == "single" and record.get("imageUrl"):
            body["images"].append(record["imageUrl"])
        elif reference_mode == "first_last":
            if record.get("firstFrameUrl"):
                body["images"].append(record["firstFrameUrl"])
            if record.get("lastFrameUrl"):
                body["images"].append(record["lastFrameUrl"])
        elif reference_mode == "multiple" and record.get("referenceImageUrls"):
            refs = _json_or_none(record["referenceImageUrls"])
            body["images"].extend(_iterate_like_js(refs))

        # 可选参数
        if record.get("duration"):
            body["duration"] = record["duration"]
        if record.get("aspectRatio"):
            # Vidu 使用 resolution 参数而非 aspect ratio（三种比例都映射到 720p）
            ratio_map = {"16:9": "720p", "9:16": "720p", "1:1": "720p"}
            body["resolution"] = ratio_map.get(record["aspectRatio"]) or "720p"

        return {
            "url": join_provider_url(config.get("baseUrl"), "", "/ent/v2/img2video"),
            "method": "POST",
            "headers": {
                "Content-Type": _JSON_MIME,
                "Authorization": f"Token {config.get('apiKey')}",  # 注意: 不是 Bearer!
            },
            "body": body,
        }

    def parse_generate_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        if data.get("task_id"):
            return {"isAsync": True, "taskId": data["task_id"]}
        # 同步返回（不太可能发生）
        if data.get("video_url"):
            return {"isAsync": False, "videoUrl": data["video_url"]}
        raise ValueError("No task_id in Vidu response")

    def build_poll_request(self, config: dict[str, Any], task_id: str) -> dict[str, Any]:
        # Vidu 没有轮询端点：返回一个不可达的 URL，让轮询尽快结束并改依赖 Webhook
        return {"url": "vidu://no-polling-endpoint", "method": "GET", "headers": {}, "body": None}

    def parse_poll_response(self, result: Any) -> dict[str, Any]:
        return {"status": "processing"}

    def extract_video_url(self, result: Any) -> str | None:
        return dig(result, "video_url") or None


def parse_callback_state(body: Any) -> dict[str, Any]:
    """Vidu 回调状态映射（原 ``ViduVideoAdapter.parseCallbackState`` 静态方法）。

    Webhook 路由用它解析回调；**未知 state 归为 failed**（而不是静默忽略）。
    """
    data = as_dict(body)
    state = data.get("state")
    if state == "success":
        return {"status": "completed", "videoUrl": data.get("video_url")}
    if state == "failed":
        return {"status": "failed", "error": data.get("error") or "Vidu generation failed"}
    return {"status": "failed", "error": f"Unknown state: {state}"}


class AliVideoAdapter:
    """阿里云百炼（万相）视频生成。"""

    provider = "ali"

    _DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"

    def build_generate_request(self, config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        base_url = config.get("baseUrl") or self._DEFAULT_BASE_URL
        url = join_provider_url(base_url, "/api/v1", "/services/aigc/video-generation/video-synthesis")

        headers = {"Authorization": f"Bearer {config.get('apiKey')}", "Content-Type": _JSON_MIME}

        # ⚠️ `imageUrl ?? firstFrameUrl ?? ''` 是 **nullish** 链：传空串会**保留空串**
        image_url = record.get("imageUrl")
        if image_url is None:
            image_url = record.get("firstFrameUrl")
        if image_url is None:
            image_url = ""

        body: dict[str, Any] = {
            "model": record.get("model") or "wan2.6-i2v-flash",
            "input": {"prompt": record.get("prompt"), "img_url": image_url},
            "parameters": {
                "resolution": self._normalize_resolution(record.get("aspectRatio") or "16:9"),
                "duration": record.get("duration") or 5,
                "watermark": False,
                "seed": random_seed(),
            },
        }

        # 尾帧模式
        if record.get("lastFrameUrl"):
            body["input"]["last_img_url"] = record["lastFrameUrl"]

        return {"url": url, "method": "POST", "headers": headers, "body": body}

    def parse_generate_response(self, result: Any) -> dict[str, Any]:
        data = as_dict(result)
        if dig(data, "output", "task_status") == "PENDING" and dig(data, "output", "task_id"):
            return {"isAsync": True, "taskId": dig(data, "output", "task_id")}
        if dig(data, "output", "video_url"):
            return {"isAsync": False, "videoUrl": dig(data, "output", "video_url")}
        raise ValueError(f"Unexpected Ali video response: {js_json_stringify(result)[:200]}")

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
            return {"status": "completed", "videoUrl": dig(data, "output", "video_url")}
        if status == "FAILED":
            return {"status": "failed", "error": data.get("message") or "Video generation failed"}
        if status in ("PENDING", "RUNNING"):
            return {"status": "processing"}
        return {"status": "pending"}

    def extract_video_url(self, result: Any) -> str | None:
        return dig(result, "output", "video_url") or None

    @staticmethod
    def _normalize_resolution(aspect_ratio: Any = None) -> str:
        ratio = aspect_ratio or "16:9"
        if ratio == "9:16":
            return "720P"
        if ratio == "1:1":
            return "720P"
        return "1080P"
