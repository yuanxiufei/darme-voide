"""AI 服务配置解析层 —— 移植 ``services/ai.ts``（整文件）。

**所有适配器与 Agent 的入口**：从 ``ai_service_configs`` 里选出「当前生效」的配置，
解析出 ``provider / baseUrl / apiKey / models``（含 settings 里的 negative_prompt 与扩展配置）。

⚠️ 四个容易踩的点：

1. **筛选顺序**：``is_active`` 过滤 → 按 ``priority`` 倒序 → 取第一个。
   ``priority`` 为 null 时按 0 参与排序（``(b.priority || 0) - (a.priority || 0)``）。
2. ``model`` 列存的是 **JSON 数组字符串**；解析失败**静默保留空数组**（不让脏数据打断生成）。
3. ``models`` 会 ``filter(Boolean)`` 去空 ⇒ ``model``（首个）与 ``models[0]`` 可能不同：
   ``model`` 取的是**未过滤**的首项（原 TS 如此，别"顺手修正"）。
4. ``get_text_provider_base_url`` 是**文本**专用的前缀规则，与图片/视频的原生前缀不同
   （阿里文本走 OpenAI 兼容的 ``/compatible-mode/v1``，MiniMax 文本走 ``/v1``）。
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..core.models import ai_service_configs
from .provider_probe import join_provider_url
from .task_logger import log_task_progress, log_task_warn


def _parse_negative_prompt(settings: Any) -> str | None:
    """从 settings JSON 取 ``negative_prompt``（去掉空白后非空才算）。"""
    if not settings:
        return None
    try:
        parsed = json.loads(settings)
    except (ValueError, TypeError):
        return None
    value = parsed.get("negative_prompt") if isinstance(parsed, dict) else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_settings(settings: Any) -> dict[str, Any] | None:
    if not settings:
        return None
    try:
        parsed = json.loads(settings)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _models_of(raw: Any) -> list[str]:
    """``JSON.parse(row.model)``，失败静默空数组。"""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _to_config(row: Any) -> dict[str, Any]:
    models = _models_of(row.model)
    return {
        "provider": row.provider or "",
        "baseUrl": row.base_url,
        "apiKey": row.api_key,
        "model": models[0] if models else "",  # ⚠️ 未过滤的首项（见模块头第 3 点）
        "models": [m for m in models if m],
        "negativePrompt": _parse_negative_prompt(row.settings),
        "settings": _parse_settings(row.settings),
    }


def get_active_config(conn: Connection, service_type: str) -> dict[str, Any] | None:
    """按 service_type 取生效配置（is_active + priority 倒序取首个）。"""
    rows = [
        row
        for row in conn.execute(
            select(ai_service_configs).where(ai_service_configs.c.service_type == service_type)
        ).all()
        if row.is_active
    ]
    rows.sort(key=lambda r: r.priority or 0, reverse=True)

    if not rows:
        log_task_warn("AIConfig", "active-config-missing", {"serviceType": service_type})
        return None

    active = rows[0]
    config = _to_config(active)
    log_task_progress(
        "AIConfig",
        "active-config-selected",
        {
            "serviceType": service_type,
            "configId": active.id,
            "provider": active.provider,
            "model": config["model"],
            "priority": active.priority,
        },
    )
    return config


def get_config_by_id(conn: Connection, config_id: Any) -> dict[str, Any] | None:
    """按 id 取配置（**必须 is_active**，否则视为取不到）。"""
    row = conn.execute(
        select(ai_service_configs).where(ai_service_configs.c.id == config_id)
    ).first()
    if row is None or not row.is_active:
        log_task_warn("AIConfig", "config-by-id-missing", {"configId": config_id})
        return None
    config = _to_config(row)
    log_task_progress(
        "AIConfig",
        "config-by-id-selected",
        {
            "configId": config_id,
            "provider": row.provider,
            "model": config["model"],
            "serviceType": row.service_type,
        },
    )
    return config


def get_active_config_by_provider(
    conn: Connection, service_type: str, provider: str
) -> dict[str, Any] | None:
    """按 provider 精确匹配活跃配置。

    用于**崩溃恢复**：生成记录里只存了 provider，需据此找回对应厂商的 baseUrl/apiKey 续跑。
    """
    rows = [
        row
        for row in conn.execute(
            select(ai_service_configs).where(ai_service_configs.c.service_type == service_type)
        ).all()
        if row.is_active and (row.provider or "").lower() == (provider or "").lower()
    ]
    rows.sort(key=lambda r: r.priority or 0, reverse=True)

    if not rows:
        log_task_warn(
            "AIConfig",
            "active-config-by-provider-missing",
            {"serviceType": service_type, "provider": provider},
        )
        return None
    return _to_config(rows[0])


def get_text_config(conn: Connection) -> dict[str, Any]:
    config = get_active_config(conn, "text")
    if not config:
        raise RuntimeError("No active text AI config")
    return config


def get_audio_config(conn: Connection) -> dict[str, Any]:
    config = get_active_config(conn, "audio")
    if not config:
        raise RuntimeError("No active audio AI config — 请在设置中添加音频服务")
    return config


def get_audio_config_by_id(conn: Connection, config_id: Any = None) -> dict[str, Any]:
    """指定 id 优先，取不到则回退到生效的音频配置。"""
    if config_id:
        config = get_config_by_id(conn, config_id)
        if config:
            return config
    return get_audio_config(conn)


def get_text_provider_base_url(config: dict[str, Any]) -> str:
    """文本服务的 baseUrl 前缀规则（与图片/视频的**不同**，见模块头第 4 点）。"""
    provider = (config.get("provider") or "").lower()
    base_url = config.get("baseUrl") or ""

    if provider in ("openai", "openrouter", "chatfire", "ollama"):
        return join_provider_url(base_url, "/v1", "")
    if provider == "volcengine":
        return join_provider_url(base_url, "/api/v3", "")
    if provider == "ali":
        # 阿里百炼文本走 OpenAI 兼容模式端点（区别于图片/视频的 DashScope 原生 /api/v1）
        return join_provider_url(base_url, "/compatible-mode/v1", "")
    if provider == "minimax":
        # MiniMax 文本走 OpenAI 兼容 /v1（与 openai-compatible-text 适配器的前缀表对齐）
        return join_provider_url(base_url, "/v1", "")
    return base_url
