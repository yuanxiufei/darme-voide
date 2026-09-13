"""Provider 适配器层（移植自 ``backend/src/services/adapters/``，共 16 个文件 / 1611 行）。

**这一层全是纯函数**：只把「配置 + 记录」翻译成「HTTP 请求描述」，再把响应解析成
统一结构；真正发请求的是 ``image-generation`` / ``video-generation`` /
``text-generation`` / ``tts-generation`` 那几个服务。因此本层可以完全离线单测。

覆盖的 17 家厂商（按能力分）::

    图片  minimax / openai / gemini / volcengine / ali / chatfire / local-sd
    视频  minimax / volcengine / vidu / ali
    TTS   minimax / cosyvoice
    文本  openai / openrouter / chatfire / ollama / volcengine / ali / minimax / gemini

（chatfire 的图片复用 OpenAI 实现；文本那 7 家 OpenAI 兼容的共用一个实例。）

⚠️ 全层字段名 **camelCase**，与 TS 契约一致，别改成 snake_case。
"""
from __future__ import annotations

from .jscompat import as_dict, dig, js_json_stringify, js_length, js_num_str, js_parse_int
from .registry import (
    UnknownProviderError,
    get_image_adapter,
    get_text_adapter,
    get_tts_adapter,
    get_video_adapter,
    image_adapters,
    text_adapters,
    tts_adapters,
    video_adapters,
)
from .types import (
    AIConfig,
    ImageGenResponse,
    ImageGenerationRecord,
    ImagePollResponse,
    ProviderRequest,
    TextGenerateParams,
    VideoGenResponse,
    VideoGenerationRecord,
    VideoPollResponse,
)
from .url import join_provider_url

__all__ = [
    "AIConfig",
    "ImageGenResponse",
    "ImageGenerationRecord",
    "ImagePollResponse",
    "ProviderRequest",
    "TextGenerateParams",
    "UnknownProviderError",
    "VideoGenResponse",
    "VideoGenerationRecord",
    "VideoPollResponse",
    "as_dict",
    "dig",
    "get_image_adapter",
    "get_text_adapter",
    "get_tts_adapter",
    "get_video_adapter",
    "image_adapters",
    "join_provider_url",
    "js_json_stringify",
    "js_length",
    "js_num_str",
    "js_parse_int",
    "text_adapters",
    "tts_adapters",
    "video_adapters",
]
