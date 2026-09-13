"""文本生成适配器（逐字移植）。

来源::

    backend/src/services/adapters/openai-compatible-text.ts
    backend/src/services/adapters/gemini-text.ts

``OpenAICompatibleTextAdapter`` 一个类覆盖 7 家（只差端点前缀），
**不是** 7 份复制体 —— 注册表里它们共享同一个实例。
"""
from __future__ import annotations

from typing import Any

from ...response import js_nullish
from .jscompat import add_query_param, as_dict, dig
from .url import join_provider_url

__all__ = ["GeminiTextAdapter", "OpenAICompatibleTextAdapter"]

#: provider → 端点前缀（OpenAI 兼容家族的全部差异就这一张表）
PREFIX_BY_PROVIDER = {
    "openai": "/v1",
    "openrouter": "/v1",
    "chatfire": "/v1",
    "ollama": "/v1",
    "volcengine": "/api/v3",
    "ali": "/compatible-mode/v1",
    "minimax": "/v1",
}


class OpenAICompatibleTextAdapter:
    """OpenAI 兼容文本生成（``/chat/completions``）。"""

    provider = "openai-compatible"

    def build_request(self, config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        prefix = PREFIX_BY_PROVIDER.get(str(config.get("provider") or "").lower()) or "/v1"

        body: dict[str, Any] = {
            "model": params.get("model"),
            "messages": params.get("messages"),
            # ⚠️ nullish：temperature 传 0 要保留 0
            "temperature": js_nullish(params.get("temperature"), 0.7),
        }
        if params.get("maxTokens"):
            body["max_tokens"] = params["maxTokens"]

        return {
            "url": join_provider_url(config.get("baseUrl"), prefix, "/chat/completions"),
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.get('apiKey')}",
            },
            "body": body,
        }

    def parse_response(self, result: Any) -> str:
        content = dig(result, "choices", 0, "message", "content")
        if isinstance(content, str):
            return content

        # 某些兼容实现返回数组 content（多模态），取第一段 text
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    return part["text"]
        return ""


class GeminiTextAdapter:
    """Gemini 原生文本生成（``:generateContent``，非 OpenAI 兼容）。

    * system 消息映射到 ``systemInstruction``；assistant 角色映射成 ``model``
    * 认证走 URL ``?key=`` + Header ``x-goog-api-key``（**没有** Authorization）
    * 响应把所有 ``parts[].text`` **拼成一个字符串**（不是取第一段）
    """

    provider = "gemini"

    def build_request(self, config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        model = params.get("model") or ""
        model_name = model if str(model).startswith("models/") else f"models/{model}"

        messages = params.get("messages") or []
        system_parts = [
            {"text": m.get("content")} for m in messages if isinstance(m, dict) and m.get("role") == "system"
        ]
        chat_contents = [
            {
                "role": "model" if m.get("role") == "assistant" else "user",
                "parts": [{"text": m.get("content")}],
            }
            for m in messages
            if isinstance(m, dict) and m.get("role") != "system"
        ]

        generation_config: dict[str, Any] = {
            "temperature": js_nullish(params.get("temperature"), 0.7),
        }
        body: dict[str, Any] = {"contents": chat_contents, "generationConfig": generation_config}
        if len(system_parts) > 0:
            body["systemInstruction"] = {"parts": system_parts}
        if params.get("maxTokens"):
            generation_config["maxOutputTokens"] = params["maxTokens"]

        url = add_query_param(
            join_provider_url(config.get("baseUrl"), "/v1beta", f"/{model_name}:generateContent"),
            "key",
            config.get("apiKey"),
        )
        return {
            "url": url,
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
                "x-goog-api-key": config.get("apiKey"),
            },
            "body": body,
        }

    def parse_response(self, result: Any) -> str:
        data = as_dict(result)
        parts = dig(data, "candidates", 0, "content", "parts") or []
        if not isinstance(parts, list):
            return ""
        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "".join(texts)
