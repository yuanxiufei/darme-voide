"""Provider Adapter 注册表（移植自 ``backend/src/services/adapters/registry.ts``）。

按 provider 名称取适配器实例。**未知 provider 显式抛错**（原 TS 注释：
「不再静默 fallback 到 MiniMax」）—— 让配置里的拼写错误在提交时就暴露，
而不是"成功走到 MiniMax"这种极难排查的降级。
"""
from __future__ import annotations

from typing import Any, TypeVar

from .image_adapters import (
    AliImageAdapter,
    GeminiImageAdapter,
    LocalSDImageAdapter,
    MiniMaxImageAdapter,
    OpenAIImageAdapter,
    VolcEngineImageAdapter,
)
from .text_adapters import GeminiTextAdapter, OpenAICompatibleTextAdapter
from .tts_adapters import CosyVoiceTTSAdapter, MiniMaxTTSAdapter
from .video_adapters import (
    AliVideoAdapter,
    MiniMaxVideoAdapter,
    ViduVideoAdapter,
    VolcEngineVideoAdapter,
)

__all__ = [
    "UnknownProviderError",
    "get_image_adapter",
    "get_text_adapter",
    "get_tts_adapter",
    "get_video_adapter",
    "image_adapters",
    "text_adapters",
    "tts_adapters",
    "video_adapters",
]


class UnknownProviderError(ValueError):
    """未知 provider —— 单独成类，方便调用方把「配置写错」与「上游报错」分开处理。"""


# 图片 Adapter 注册表
image_adapters: dict[str, Any] = {
    "minimax": MiniMaxImageAdapter(),
    "openai": OpenAIImageAdapter(),
    "gemini": GeminiImageAdapter(),
    "volcengine": VolcEngineImageAdapter(),
    "ali": AliImageAdapter(),
    # 第三方代理 - 兼容 OpenAI 格式
    "chatfire": OpenAIImageAdapter(),
    # 本地 SD WebUI
    "local-sd": LocalSDImageAdapter(),
}

# 视频 Adapter 注册表
# 注意：MiniMax H3 开源后本地启动也走 minimax Adapter，只需改 baseUrl 指向 localhost
video_adapters: dict[str, Any] = {
    "minimax": MiniMaxVideoAdapter(),
    "volcengine": VolcEngineVideoAdapter(),
    "vidu": ViduVideoAdapter(),
    "ali": AliVideoAdapter(),
}

# TTS Adapter 注册表
tts_adapters: dict[str, Any] = {
    "minimax": MiniMaxTTSAdapter(),
    "cosyvoice": CosyVoiceTTSAdapter(),
}

# 文本 Adapter 注册表
# openai/openrouter/chatfire/ollama/volcengine/ali/minimax 均为 OpenAI 兼容（仅端点前缀不同），
# **共享同一个实例**（前缀差异在适配器内部按 config.provider 查表得到）
_openai_compatible_text = OpenAICompatibleTextAdapter()
text_adapters: dict[str, Any] = {
    "openai": _openai_compatible_text,
    "openrouter": _openai_compatible_text,
    "chatfire": _openai_compatible_text,
    "ollama": _openai_compatible_text,
    "volcengine": _openai_compatible_text,
    "ali": _openai_compatible_text,
    "minimax": _openai_compatible_text,
    "gemini": GeminiTextAdapter(),
}

_T = TypeVar("_T")


def _resolve(registry: dict[str, _T], provider: str, kind: str) -> _T:
    key = str(provider or "").lower()
    adapter = registry.get(key)
    if adapter is None:
        available = ", ".join(registry)
        raise UnknownProviderError(f'Unknown {kind} provider "{provider}". Available: {available}')
    return adapter


def get_image_adapter(provider: str) -> Any:
    return _resolve(image_adapters, provider, "image")


def get_video_adapter(provider: str) -> Any:
    return _resolve(video_adapters, provider, "video")


def get_tts_adapter(provider: str) -> Any:
    return _resolve(tts_adapters, provider, "TTS")


def get_text_adapter(provider: str) -> Any:
    return _resolve(text_adapters, provider, "text")
