"""文本生成适配器（逐字移植）。

来源::

    backend/src/services/adapters/openai-compatible-text.ts
    backend/src/services/adapters/gemini-text.ts

``OpenAICompatibleTextAdapter`` 一个类覆盖 7 家（只差端点前缀），
**不是** 7 份复制体 —— 注册表里它们共享同一个实例。
"""
from __future__ import annotations

from typing import Any

from ...core.response import js_nullish
from .jscompat import add_query_param, as_dict, dig
from .url import join_provider_url

__all__ = ["GeminiTextAdapter", "OllamaTextAdapter", "OpenAICompatibleTextAdapter"]

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


class OllamaTextAdapter:
    """Ollama **原生**文本生成（``/api/chat``）—— **不**走 OpenAI 兼容端点。

    ⚠️ 为什么必须单列一家（2026-09-16 活体实测，证据确凿）：
    本机 ``qwen3:14b`` 是**思考模型**，走 Ollama 的 **OpenAI 兼容**端点（``/v1/chat/completions``）时
    响应形如 ``{"choices":[{"message":{"content":"","reasoning":"好的，用户让我…"}}]}`` ——
    **``content`` 恒为空串** ✗，正文全落在 ``reasoning`` 字段里（``max_tokens`` 给到 128 也一样把
    预算全花在思考上，``finish_reason='length'``）⇒ 走那条路的调用方**拿不到任何文本** ✗。
    同一模型走**原生** ``/api/chat`` 并显式 ``think: false`` ⇒ ``message.content`` 正常返回 ✓。
    （实测脚本见 ``tests/local_services_live_test.py`` 的活体断言 ✓，模板是 ``self.think = False``。）

    端点与字段（Ollama 原生，非 OpenAI 兼容）：
    * ``POST {base}/api/chat``，body ``{model, messages, stream: false, think: false, options: {...}}``
    * 响应 ``{"message": {"role": "assistant", "content": "..."}}``
    """

    provider = "ollama"

    #: 关掉思考链（本项目要的是**可用文本**，不是思考过程）
    think: bool = False

    def build_request(self, config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {
            # ⚠️ nullish：temperature 传 0 要保留 0
            "temperature": js_nullish(params.get("temperature"), 0.7),
        }
        if params.get("maxTokens"):
            options["num_predict"] = params["maxTokens"]   # ollama 的上限参数名与 OpenAI 不同
        body: dict[str, Any] = {
            "model": params.get("model"),
            "messages": params.get("messages"),
            "stream": False,
            "think": self.think,
            "options": options,
        }
        return {
            "url": join_provider_url(config.get("baseUrl"), "", "/api/chat"),
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": body,
        }

    def parse_response(self, result: Any) -> str:
        """原生响应取 ``message.content``；⚠️ 兼容「万一被反代成 OpenAI 形状」的情况。"""
        data = as_dict(result)
        content = dig(data, "message", "content")
        if isinstance(content, str) and content:
            return content
        # 兜底：若网关把 /api/chat 反代成 OpenAI 形状，仍能取到文本
        return OpenAICompatibleTextAdapter.parse_response(self, result)


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
