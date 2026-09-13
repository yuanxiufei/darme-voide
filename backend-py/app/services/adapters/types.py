"""适配器层的契约类型（移植自 ``backend/src/services/adapters/types.ts``）。

TS 侧是 ``interface``；Python 侧用 ``TypedDict`` 保留字段名与可选性 ——
**既是文档，也让 IDE 能提示**，运行时不做校验（与 TS 一致，都是编译期约束）。

⚠️ 全层的字段名都是 **camelCase**（``baseUrl`` / ``referenceImages`` / ``taskId``…），
这是跨语言契约的一部分，不要"顺手"改成 snake_case。
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

__all__ = [
    "AIConfig",
    "ImageGenResponse",
    "ImageGenerationRecord",
    "ImagePollResponse",
    "ProviderRequest",
    "TTSResult",
    "TextGenerateParams",
    "VideoGenResponse",
    "VideoGenerationRecord",
    "VideoPollResponse",
]


class ProviderRequest(TypedDict):
    """适配器构建出的 HTTP 请求描述 —— **适配器自己不发起请求**。"""

    url: str
    method: str
    headers: dict[str, str]
    #: ``None`` 表示无请求体（对应 TS 里的 ``body: undefined``）
    body: Any


class AIConfig(TypedDict, total=False):
    provider: str
    baseUrl: str
    apiKey: str
    model: str
    #: 所有可用模型（按优先级排序），失败时自动 fallback
    models: list[str]
    #: 图片生成负面提示词（从配置 settings JSON 解析，可空）
    negativePrompt: str | None
    #: 配置原始 settings JSON 解析结果（含 checkpoint_map 等扩展配置，供 adapter 使用）
    settings: dict[str, Any] | None


class ImageGenerationRecord(TypedDict, total=False):
    id: int
    model: str | None
    prompt: str | None
    negativePrompt: str | None
    size: str | None
    frameType: str | None
    referenceImages: str | None


class VideoGenerationRecord(TypedDict, total=False):
    id: int
    model: str | None
    prompt: str | None
    negativePrompt: str | None
    referenceMode: Literal["single", "first_last", "multiple"] | None
    imageUrl: str | None
    firstFrameUrl: str | None
    lastFrameUrl: str | None
    referenceImageUrls: str | None
    sceneType: str | None
    referenceAudioUrls: str | None
    duration: int | None
    aspectRatio: str | None


class ImageGenResponse(TypedDict, total=False):
    isAsync: bool
    taskId: str
    #: 同步模式下直接返回的图片 URL
    imageUrl: str


class ImagePollResponse(TypedDict, total=False):
    status: Literal["pending", "processing", "completed", "failed"]
    imageUrl: str
    error: str


class VideoGenResponse(TypedDict, total=False):
    isAsync: bool
    taskId: str
    videoUrl: str


class VideoPollResponse(TypedDict, total=False):
    status: Literal["pending", "processing", "completed", "failed"]
    videoUrl: str
    error: str


class TTSResult(TypedDict):
    audioHex: str
    audioLength: int
    sampleRate: int
    bitrate: int
    format: str
    channel: int


class TextGenerateParams(TypedDict, total=False):
    model: str
    messages: list[dict[str, str]]
    temperature: float
    maxTokens: int
