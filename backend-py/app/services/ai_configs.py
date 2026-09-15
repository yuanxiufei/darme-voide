"""AI 服务配置 —— 移植 ``routes/aiConfigs.ts`` 的**纯 DB 部分**所需的辅助。

移植范围（已迁移）：`GET /`、`POST /`、`POST /quick-preset`、`POST /quick-local`、
`GET /:id`、`PUT /:id`、`DELETE /:id`、`GET /configs/local`、以及 `GET /ai-providers`。
未迁移：`/ollama/*`（子进程 + HTTP 探测）、`POST /models`（厂商 HTTP）、`POST /test`（HTTP 探测）、
`GET /gpu/status`（nvidia-smi）、`POST /gpu/release-all`（GPU 管理器）、`GET /runtime/health`（HTTP 探测）。

⚠️ 两处 JSON 语义照抄：

* ``parse_settings_object`` 解析失败**静默回退空对象** —— settings 是「扩展配置容器」，
  损坏时不该让整条接口挂掉。
* ``build_settings`` 的关键在于**合并而非覆盖**：`negative_prompt` / `checkpoint_map` 是一等字段，
  `settings` 整体兜底合并；任一出现即重算，从而保住 `checkpoint_map` 等既有扩展配置不被抹掉。

⚠️ ``is_local_config`` 来自 ``services/gpu-manager.ts``（那里的其余部分依赖 GPU/HTTP，未迁）。
它按「provider 白名单 + baseUrl 主机名」双判据识别本地服务 —— 注释里明确 **openai 不在白名单**：
Ollama 的 OpenAI 兼容接口 baseUrl 指向 localhost，会被主机名判据命中；而真实云端 OpenAI
不该被误判成本地。后续 ``local-models`` 域也会用到它。
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from sqlalchemy.engine import Row

from ..core.response import js_truthy, row_to_dict

#: 本地 provider 白名单（运行在本机 GPU 上的服务）
LOCAL_PROVIDERS = {"ollama", "local-sd", "cosyvoice"}

#: 内部哨兵：区分「调用方没传这个键」与「传了 null」（对齐 TS 的 ``undefined``）
UNSET: Any = object()

#: 「一键配置」预设（对齐 TS 的 PRESET_SERVICES）
PRESET_SERVICES = (
    {"service_type": "text", "label": "文本", "provider": "chatfire", "base_url": "https://api.chatfire.site", "model": "gemini-3-pro-preview", "priority": 100},
    {"service_type": "image", "label": "图片", "provider": "gemini", "base_url": "https://api.chatfire.site", "model": "gemini-3-pro-image-preview", "priority": 99},
    {"service_type": "video", "label": "视频", "provider": "volcengine", "base_url": "https://api.chatfire.site/volcengine", "model": "doubao-seedance-1-5-pro-251215", "priority": 98},
    {"service_type": "audio", "label": "音频", "provider": "minimax", "base_url": "https://api.chatfire.site/minimax", "model": "speech-2.8-hd", "priority": 97},
)

#: 本地模型预设（无需 API Key）
LOCAL_PRESET_SERVICES = (
    {"service_type": "text", "label": "文本(本地)", "provider": "openai", "base_url": "http://localhost:11434", "model": "qwen3:14b", "priority": 85},
    {"service_type": "image", "label": "图片(本地)", "provider": "local-sd", "base_url": "http://localhost:7860", "model": "sdxl-base", "priority": 84},
    {"service_type": "video", "label": "视频(本地H3)", "provider": "minimax", "base_url": "http://localhost:8765", "model": "hailuo-02", "priority": 83},
    {"service_type": "audio", "label": "音频(本地)", "provider": "cosyvoice", "base_url": "http://localhost:9880", "model": "cosyvoice-v2", "priority": 82},
)

#: 一键配置时同步写入的 Agent 默认
PRESET_AGENT_DEFAULTS = (
    {"agent_type": "script_rewriter", "name": "剧本改写"},
    {"agent_type": "extractor", "name": "角色场景提取"},
    {"agent_type": "storyboard_breaker", "name": "分镜拆解"},
    {"agent_type": "voice_assigner", "name": "音色分配"},
    {"agent_type": "grid_prompt_generator", "name": "图片提示词生成"},
)

PRESET_AGENT_MODEL = "gemini-3-pro-preview"


def is_local_config(base_url: str, provider: str) -> bool:
    """是否为本地服务配置。对齐 ``gpu-manager.ts`` 的 ``isLocalConfig``。

    注意与 JS 的两点等价性：``new URL('')`` 在 JS 里**抛错**（→ false），
    Python 的 ``urlparse('')`` 得到 ``hostname is None``（→ 同样 false）；
    ``new URL('localhost:7860')`` 不抛错但 hostname 为空（JS 把 ``localhost:`` 当 scheme），
    Python 的 ``urlparse`` 同样得到空 hostname —— 两边一致。
    """
    if (provider or "").lower() in LOCAL_PROVIDERS:
        return True
    host = urlparse(base_url or "").hostname
    if not host:
        return False
    return host in ("localhost", "127.0.0.1") or host.startswith("192.168.")


def parse_settings_object(settings: Any) -> dict[str, Any]:
    """``JSON.parse(settings || '{}')``，损坏时回退空对象。"""
    try:
        parsed = json.loads(settings if settings else "{}")
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_settings(
    *,
    existing: Any = None,
    negative_prompt: Any = UNSET,
    checkpoint_map: Any = UNSET,
    settings: Any = UNSET,
) -> str:
    """组装 ``settings`` JSON：保留既有扩展字段，只合并被显式传入的部分。"""
    next_obj = parse_settings_object(existing)

    if negative_prompt is not UNSET:
        next_obj["negative_prompt"] = negative_prompt if js_truthy(negative_prompt) else ""
    if checkpoint_map is not UNSET:
        # `== null` ⇒ 显式传 null 表示「删除该键」
        if checkpoint_map is None:
            next_obj.pop("checkpoint_map", None)
        else:
            next_obj["checkpoint_map"] = checkpoint_map
    # `input.settings && typeof === 'object'`：空对象在 JS 里为真 ⇒ 用 js_truthy
    if settings is not UNSET and isinstance(settings, dict) and js_truthy(settings):
        next_obj.update(settings)

    # 与原 TS 的 JSON.stringify 一致：紧凑、不转义非 ASCII
    return json.dumps(next_obj, ensure_ascii=False, separators=(",", ":"))


def parse_model_json(raw: Any) -> list[Any]:
    """``row.model ? JSON.parse(row.model) : []`` —— 损坏时**抛错**（原 TS 会冒到 500）。"""
    if js_truthy(raw):
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else parsed
    return []


def map_config_row(
    row: Row, *, with_settings_fields: bool = False, with_is_local: bool = False
) -> dict[str, Any]:
    """``ai_service_configs`` 行 → 契约形状（snake_case + model 反序列化）。

    三个可选字段对应三个接口的差异，别统一：
    ``negative_prompt`` / ``checkpoint_map`` 只在 ``GET /`` 与 ``GET /:id`` 出现；
    ``is_local`` **只在 ``GET /`` 出现**（详情与 /configs/local 都没有）。
    """
    d = row_to_dict(row)
    d["model"] = parse_model_json(d.get("model"))

    if with_settings_fields:
        settings = parse_settings_object(d.get("settings"))
        value = settings.get("negative_prompt")
        d["negative_prompt"] = value if js_truthy(value) else ""
        d["checkpoint_map"] = settings.get("checkpoint_map")  # `?? null` 与直接取值等价

    if with_is_local:
        d["is_local"] = is_local_config(d.get("base_url") or "", d.get("provider") or "")

    return d
