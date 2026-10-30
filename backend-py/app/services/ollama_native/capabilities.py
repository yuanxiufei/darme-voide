"""模型**能力面**判定 —— 上游 ``server/images.go`` 里那串 ``*Capabilities()`` 的对等移植。

⭐ 为什么单独成模块：Ollama 的「这个模型能干什么」**不是**存在某个字段里的，而是由六路信号
叠出来的（上游 ``Model.capabilitiesForTemplate``）：

    config.capabilities → gguf 元数据 → projector 层 → 模板变量 → parser 名 → 模型族
      ↓ 最后还要过一道「抑制」：safetensors + nemotron 不许自称 vision

本仓离线也要能判（前端要在没启动引擎时列出模型能干什么），所以这里只落**不需要读 GGUF
二进制**的那几路：``config`` / ``projector 层`` / ``Go 模板`` / ``parser 名`` / ``模型族``。
⚠️ **有意不移植的两路**（都要求真读 GGUF 头部，属另一个量级的实现）：

    ① ``ggufCapabilities`` —— 要读 ``pooling_type``（embedding 判据）、``vision.block_count``
       （视觉塔判据）、``tokenizer.chat_template``（工具/思考判据）。这些元数据只有把几 GB 的
       blob 头部解析出来才有，仓内没有 GGUF 解析器，**编不出来就只能不判**（宁可漏报不误报 ✓）。
    ② ``thinking.InferTags`` —— 上游是从模板里**推断**思考标签对（启发式找 assistant 段前后
       的标记字面量）。这里退化成「模板里出现已知标签字面量才判思考」，同样宁可漏报 ✓。

⚠️ 因此本模块的结论口径是：``source`` 字段必须跟着结果一起给出去 ✓ —— 同一个 ``vision``
在「projector 层在场」时是**硬证据**，在「模板字面量」时只是**推测**，前端别把两者画成一样重。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from . import manifest as manifest_module
from .gotemplate import parse as parse_template

__all__ = [
    "CAPABILITY_COMPLETION",
    "CAPABILITY_TOOLS",
    "CAPABILITY_INSERT",
    "CAPABILITY_VISION",
    "CAPABILITY_EMBEDDING",
    "CAPABILITY_THINKING",
    "CAPABILITY_AUDIO",
    "CAPABILITY_ORDER",
    "CAPABILITY_LABELS",
    "PARSER_CAPABILITIES",
    "THINK_TAG_LITERALS",
    "PLUGIN_MEDIA_TYPES",
    "capabilities_from",
    "check_capabilities",
    "describe",
    "infer_from_config",
    "is_video_reader",
    "normalize_capabilities",
    "parser_capabilities",
    "plugin_layers",
    "projector_capabilities",
    "template_capabilities",
]


# ── 能力名（``types/model/capability.go`` 逐字）──────────────────────────────
CAPABILITY_COMPLETION = "completion"
CAPABILITY_TOOLS = "tools"
CAPABILITY_INSERT = "insert"
CAPABILITY_VISION = "vision"
CAPABILITY_EMBEDDING = "embedding"
CAPABILITY_THINKING = "thinking"
CAPABILITY_AUDIO = "audio"

#: 归一化后的展示顺序（上游是**插入序**，会让同一个模型因判定路径不同而排序抖动 ✗）
CAPABILITY_ORDER: tuple[str, ...] = (
    CAPABILITY_COMPLETION,
    CAPABILITY_EMBEDDING,
    CAPABILITY_VISION,
    CAPABILITY_AUDIO,
    CAPABILITY_TOOLS,
    CAPABILITY_THINKING,
    CAPABILITY_INSERT,
)

CAPABILITY_LABELS: dict[str, str] = {
    CAPABILITY_COMPLETION: "对话/补全",
    CAPABILITY_EMBEDDING: "向量化",
    CAPABILITY_VISION: "读图（可读视频帧）",
    CAPABILITY_AUDIO: "读音频",
    CAPABILITY_TOOLS: "工具调用",
    CAPABILITY_THINKING: "思考模式",
    CAPABILITY_INSERT: "中间填充",
}

#: 上游 ``parser.ParserForName`` 里**已逐个核过**的 ``(hasToolSupport, hasThinkingSupport)``。
#: ⚠️ 表外名字一律**不加任何能力**（与上游「``ParserForName`` 返回 nil 就直接返回」等价 ✓）；
#: 想扩表必须回上游 ``model/parsers/*.go`` 读那个类型的两个 ``Has*Support()``，别猜 ✗。
PARSER_CAPABILITIES: dict[str, tuple[bool, bool]] = {
    "passthrough": (False, False),
    "harmony": (True, True),
    "cogito": (True, True),
    "cohere": (True, True),
    "deepseek3": (True, True),
    "functiongemma": (True, False),
    "gemma4": (True, True),
    "gemma4-no-thinking": (True, False),
    "glimmer": (True, True),
    "glm-4.6": (True, True),
    "laguna": (True, True),
    "lfm2": (True, False),
    "lfm2-thinking": (True, True),
    "ministral": (True, False),
    "nemotron-3-nano": (True, True),
    "nemotron-3.5-nano": (True, True),
    "olmo3": (True, False),
    "olmo3-think": (False, True),
    "ornith": (True, True),
    "qwen3": (True, False),
    "qwen3-thinking": (True, True),
    "qwen3-coder": (True, False),
    "qwen3.5": (True, True),
    "qwen3-vl-instruct": (True, False),
    "qwen3-vl-thinking": (True, True),
}

#: 出现这些字面量才判「会思考」。⚠️ 这是 ``thinking.InferTags`` 的**降级替代**（见模块抬头）。
THINK_TAG_LITERALS: tuple[str, ...] = (
    "<|think|>",
    "<|channel|>",
    "<think>",
    "<|start_think|>",
    "thinking",
)

#: 层 mediaType → 插件种类。**项目组的「插件」就是这些附加层** ——
#: 权重之外的每一层都是可以单独换掉的能力扩展（LoRA / 视觉投影器 / 投机草稿 …）。
PLUGIN_MEDIA_TYPES: dict[str, str] = {
    manifest_module.MEDIA_TYPE_ADAPTER: "adapter",       # LoRA / 适配器
    manifest_module.MEDIA_TYPE_PROJECTOR: "projector",   # 多模态投影器（读图/读视频帧）
    manifest_module.MEDIA_TYPE_DRAFT: "draft",           # 投机解码草稿模型
    manifest_module.MEDIA_TYPE_MESSAGES: "messages",     # 预制消息（系统提示词模板）
    manifest_module.MEDIA_TYPE_TEMPLATE: "template",     # Go 对话模板
    manifest_module.MEDIA_TYPE_SYSTEM: "system",
    manifest_module.MEDIA_TYPE_PARAMS: "params",
    manifest_module.MEDIA_TYPE_LICENSE: "license",
    manifest_module.MEDIA_TYPE_EMBED: "embed",
    manifest_module.MEDIA_TYPE_PROMPT: "prompt",
}

#: 这几类插件会**改变能力面**（其余只是配置/元数据）；catalog 的「为什么判成可读图」靠它解释。
CAPABILITY_BEARING_PLUGINS: tuple[str, ...] = ("projector", "adapter", "draft")


def normalize_capabilities(capabilities: Iterable[str]) -> list[str]:
    """去重 + 按 :data:`CAPABILITY_ORDER` 排序；**不丢未知字符串**（排到最后，保住前向兼容 ✓）。"""
    seen: list[str] = []
    for capability in capabilities:
        text = str(capability or "").strip()
        if text and text not in seen:
            seen.append(text)
    known = [c for c in CAPABILITY_ORDER if c in seen]
    return known + [c for c in seen if c not in CAPABILITY_ORDER]


def _append(capabilities: list[str], capability: str) -> None:
    if capability not in capabilities:
        capabilities.append(capability)


def infer_from_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """从清单的 config 层 JSON 抽「自述能力」与几个关键标识（不判、只取 ✓）。

    ⚠️ config 层的形状有两代：新版是扁平字段（``model_family`` / ``model_format`` …），
    旧版/兼容 docker 写法把同一批字段塞在 ``config`` 子对象里。两处都找一遍 ✓ ——
    只认一种的实现会在另一代清单上**静默判成空**（最难查的那种错 ✗）。
    """
    source = config if isinstance(config, Mapping) else {}
    nested = source.get("config")
    merged: dict[str, Any] = dict(nested) if isinstance(nested, Mapping) else {}
    merged.update({k: v for k, v in source.items() if k != "config"})

    declared = merged.get("capabilities")
    capabilities = normalize_capabilities(declared) if isinstance(declared, (list, tuple)) else []
    families = merged.get("model_families")
    return {
        "capabilities": capabilities,
        "model_family": str(merged.get("model_family") or ""),
        "model_families": [str(x) for x in families] if isinstance(families, (list, tuple)) else [],
        "model_format": str(merged.get("model_format") or ""),
        "renderer": str(merged.get("renderer") or ""),
        "parser": str(merged.get("parser") or ""),
        "architecture": str(merged.get("architecture") or ""),
    }


def projector_capabilities(media_types: Sequence[str]) -> list[str]:
    """有 ``projector`` 层 ⇒ vision（上游 ``projectorCapabilities`` 的硬证据那一路 ✓）。

    ⚠️ 音频那一路（``projectorHasAudio`` + 抑制表）要读投影器 GGUF 头，**未移植** ✓；
    所以这里最多只会多判一个 ``vision``，不会误报 ``audio`` ✓。
    """
    if any(str(t) == manifest_module.MEDIA_TYPE_PROJECTOR for t in media_types):
        return [CAPABILITY_VISION]
    return []


def template_capabilities(template: str) -> list[str]:
    """Go 模板变量/字面量能判出来的能力（上游 ``goTemplateCapabilities`` ✓）。

    - ``tools`` 变量 ⇒ 工具调用；``suffix`` 变量 ⇒ 中间填充（``insert``）
    - 思考：上游用 ``InferTags`` 推标签对，这里退化成字面量匹配（见模块抬头，宁可漏报 ✓）

    ⚠️ 模板**解析失败**也必须能返回空表而不是抛错：一份手写坏的模板不该让整份清单列不出来 ✗。
    """
    text = str(template or "")
    if not text.strip():
        return []

    capabilities: list[str] = []
    names: list[str] = []
    try:
        names = parse_template(text).vars()
    except Exception:                       # noqa: BLE001 - 坏模板 ⇒ 只做字面量兜底 ✓
        names = []

    lowered = [n.lower() for n in names]
    if "tools" in lowered or ".tools" in text.lower():
        _append(capabilities, CAPABILITY_TOOLS)
    if "suffix" in lowered or ".suffix" in text.lower():
        _append(capabilities, CAPABILITY_INSERT)
    if any(tag in text for tag in THINK_TAG_LITERALS):
        _append(capabilities, CAPABILITY_THINKING)
    return capabilities


def parser_capabilities(parser_name: str) -> list[str]:
    """**内置** parser 名 ⇒ 能力（上游 ``parserCapabilities``：表外名字不加任何能力 ✓）。"""
    flags = PARSER_CAPABILITIES.get(str(parser_name or "").strip())
    if not flags:
        return []
    has_tools, has_thinking = flags
    capabilities: list[str] = []
    if has_tools:
        _append(capabilities, CAPABILITY_TOOLS)
    if has_thinking:
        _append(capabilities, CAPABILITY_THINKING)
    return capabilities


def _family_capabilities(family: str) -> list[str]:
    """上游 ``modelFamilyCapabilities``：只认 gpt-oss ⇒ thinking（就这一条，别扩 ✗）。"""
    return [CAPABILITY_THINKING] if str(family or "") in ("gptoss", "gpt-oss") else []


def _suppressed(capabilities: list[str], config: Mapping[str, Any]) -> list[str]:
    """上游 ``filterUnsupportedCapabilities`` 的**已核那一条** ✓。

    ``safetensors + nemotron`` 的 MLX 路径只有文本，不许自称 vision（上游注释原文：
    "The current MLX Nemotron path is text-only"）。音频抑制那条要模型架构串，未移植。
    """
    model_format = str(config.get("model_format") or "")
    family = str(config.get("model_family") or "")
    if model_format == "safetensors" and "nemotron" in family:
        return [c for c in capabilities if c != CAPABILITY_VISION]
    return capabilities


def plugin_layers(media_types: Sequence[str]) -> list[dict[str, Any]]:
    """把层的 mediaType 翻成**插件清单**（本项目口径：权重之外的附加层都是插件 ✓）。

    返回顺序＝层在清单里的顺序 ✓（同一个模型多次 ``FROM`` 会叠出多个 adapter，顺序有意义 ✓）。
    """
    plugins: list[dict[str, Any]] = []
    for media_type in media_types:
        kind = PLUGIN_MEDIA_TYPES.get(str(media_type))
        if kind:
            plugins.append({"kind": kind, "mediaType": str(media_type)})
    return plugins


def capabilities_from(
    *,
    config: Mapping[str, Any] | None = None,
    media_types: Sequence[str] = (),
    template: str = "",
) -> dict[str, Any]:
    """六路信号里**可离线复现的五路**合流，返回能力面 + 每条的来源 ✓。

    ``sources`` 里记的是「这条能力是哪一路判出来的」，空列表＝无依据（不会出现在结果里）✓。
    前端要解释「为什么说它能读图」时读这里，不要去猜。
    """
    facts = infer_from_config(config)
    capabilities: list[str] = []
    sources: dict[str, list[str]] = {}

    def merge(origin: str, items: Sequence[str]) -> None:
        for capability in items:
            _append(capabilities, capability)
            sources.setdefault(capability, [])
            if origin not in sources[capability]:
                sources[capability].append(origin)

    merge("config", facts["capabilities"])
    merge("projector", projector_capabilities(media_types))
    merge("template", template_capabilities(template))
    merge("parser", parser_capabilities(facts["parser"]))
    merge("family", _family_capabilities(facts["model_family"]))
    capabilities = _suppressed(capabilities, facts)

    # 上游 ``ggufCapabilities`` 的分支：GGUF 若没声明 ``pooling_type`` 就**默认能对话** ✓。
    # ⚠️ 判据是「没有 embedding 也没有 completion」，**不是**「一个能力都没有」——
    # 摄像投影器在场时只有 vision，那种模型不存在（视觉塔总得挂在一个能出字的骨架上 ✓）；
    # 写成 `if not capabilities` 会让「只有 projector 的清单」判出一个不能对话的模型 ✗。
    if CAPABILITY_EMBEDDING not in capabilities and CAPABILITY_COMPLETION not in capabilities:
        merge("default", [CAPABILITY_COMPLETION])

    normalized = normalize_capabilities(capabilities)
    return {
        "capabilities": normalized,
        "sources": {c: sources.get(c, []) for c in normalized},
        "model_family": facts["model_family"],
        "model_families": facts["model_families"],
        "model_format": facts["model_format"],
        "renderer": facts["renderer"],
        "parser": facts["parser"],
        "plugins": plugin_layers(media_types),
        "video_reader": is_video_reader(normalized),
    }


def is_video_reader(capabilities: Sequence[str]) -> bool:
    """能不能吃**视频帧**：判据就是 ``vision`` ✓。

    ⚠️ Ollama 本身**没有视频模型**（没有任何视频生成/编码层）；「读视频」在本项目里的真实
    含义＝「把视频抽成若干帧，逐帧喂给视觉模型」——所以能力判据是 vision，而不是去找一个
    不存在的 ``video`` 能力名 ✓。视频**生成**走 ComfyUI/H3 那条线（见 ``ollama_catalog``）。
    """
    return CAPABILITY_VISION in capabilities


def check_capabilities(available: Sequence[str], required: Sequence[str]) -> list[str]:
    """差集：``required`` 里 ``available`` 没有的那些（顺序跟 ``required`` ✓）。

    用途：调用方在真正发请求**之前**就能给出人话错误（"这个模型不支持读图，换一个"），
    而不是让上游回一句 400 让人猜 ✓。
    """
    have = set(available)
    return [c for c in required if c not in have]


def describe(capabilities: Sequence[str]) -> str:
    """一句人话摘要（前端/日志用）：``对话/补全 · 读图（可读视频帧）``。"""
    return " · ".join(CAPABILITY_LABELS.get(c, c) for c in capabilities) or "（无）"
