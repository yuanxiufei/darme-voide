"""厂商 URL 拼接 / 连通性探测描述 —— 移植 ``services/adapters/url.ts`` +
``routes/aiConfigs.ts`` 的 ``buildProbe`` / ``buildListModels`` / ``redactUrl``。

服务对象是「设置」页的两个能力：

* ``POST /ai-configs/test``   —— 按厂商拼出「最小可判活请求」，发出去看状态码
* ``POST /ai-configs/models`` —— 按厂商拼出「列出可用模型」请求并解析

⚠️ ``redact_url`` 会**出现在 ``/test`` 的响应体里**（``url`` 字段），不是纯日志装饰，
所以必须照抄 —— 尤其别用 ``urlencode``：JS 的 ``searchParams.set('key','***')`` 不会把
``*`` 转成 ``%2A``，而 Python 的 ``quote_plus`` 会，前端会把 api key 的遮蔽显示成乱码。
"""

from __future__ import annotations

import re
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

#: 需要遮蔽的 query 参数名（大小写敏感，对齐 JS 的 URLSearchParams.has）
_REDACT_KEYS = ("key", "api_key", "apikey", "token", "access_token")


def _normalize_segment(segment: str) -> str:
    if not segment:
        return ""
    return segment if segment.startswith("/") else f"/{segment}"


def join_provider_url(base_url: str, required_prefix: str, path: str) -> str:
    """把 ``baseUrl`` + ``requiredPrefix`` + ``path`` 拼成最终 URL（避免重复前缀）。

    关键点：若 ``baseUrl`` 的 path **已经以 requiredPrefix 结尾**，就不再重复追加
    —— 例如 baseUrl 已写成 ``https://api.chatfire.site/volcengine`` 时，
    拼 ``/api/v3/...`` 不会变成 ``/volcengine/api/v3`` 之外的怪路径。
    """
    normalized_base = re.sub(r"/+$", "", base_url or "")
    prefix = _normalize_segment(required_prefix)
    norm_path = _normalize_segment(path)

    if not normalized_base:
        return f"{prefix}{norm_path}"

    try:
        parts = urlsplit(normalized_base)
        if not parts.scheme or not parts.netloc:
            raise ValueError("not an absolute url")
    except ValueError:
        base_path = (
            normalized_base
            if normalized_base.endswith(prefix)
            else f"{normalized_base}{prefix}"
        )
        return f"{base_path}{norm_path}"

    current_path = re.sub(r"/+$", "", parts.path)
    merged_prefix = (
        current_path if current_path.endswith(prefix) else f"{current_path}{prefix}"
    )
    new_path = re.sub(r"/{2,}", "/", f"{merged_prefix}{norm_path}")
    return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, parts.fragment))


def redact_url(raw_url: str) -> str:
    """遮蔽 query 里的 api key（出现在响应体里，见模块头）。"""
    try:
        parts = urlsplit(raw_url)
        if not parts.scheme or not parts.netloc:
            raise ValueError("not an absolute url")
    except ValueError:
        return re.sub(
            r"([?&](?:key|api_key|apikey|token|access_token)=)[^&]+",
            r"\1***",
            raw_url,
            flags=re.IGNORECASE,
        )

    # 只在 query 上做定向替换：其余部分字节保持，避免 urlencode 重新编码带来的偏差
    query = parts.query
    for key in _REDACT_KEYS:
        query = re.sub(rf"(^|&){re.escape(key)}=[^&]*", rf"\1{key}=***", query)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _bearer_headers(api_key: str | None = None, with_json: bool = False) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if with_json:
        headers["Content-Type"] = "application/json"
    return headers


def _gemini_headers(api_key: str | None = None, with_json: bool = False) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-goog-api-key"] = api_key
    if with_json:
        headers["Content-Type"] = "application/json"
    return headers


def _vidu_headers(api_key: str | None = None, with_json: bool = False) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Token {api_key}"
    if with_json:
        headers["Content-Type"] = "application/json"
    return headers


def build_probe(
    service_type: str,
    provider: str,
    base_url: str,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """按厂商拼出「最小可判活请求」：``{method, url, headers, body}``。"""
    p = (provider or "").lower()
    m = model or ""

    if p == "gemini":
        url = join_provider_url(
            base_url, "/v1beta", f"/models/{m or 'gemini-2.5-flash'}:generateContent"
        )
        if api_key:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}key={api_key}"
        return {"method": "POST", "url": url, "headers": _gemini_headers(api_key, True), "body": {}}

    if p in ("openai", "openrouter", "chatfire"):
        return {
            "method": "GET",
            "url": join_provider_url(base_url, "/v1", "/models"),
            "headers": _bearer_headers(api_key),
            "body": None,
        }

    if p == "ali":
        return {
            "method": "POST",
            "url": join_provider_url(
                base_url,
                "/api/v1",
                "/services/aigc/video-generation/video-synthesis"
                if service_type == "video"
                else "/services/aigc/image-generation/generation",
            ),
            "headers": _bearer_headers(api_key, True),
            "body": {},
        }

    if p == "volcengine":
        path = (
            "/contents/generations/tasks" if service_type == "video" else "/images/generations"
        )
        return {
            "method": "POST",
            "url": join_provider_url(base_url, "/api/v3", path),
            "headers": _bearer_headers(api_key, True),
            "body": {},
        }

    if p == "minimax":
        if service_type == "audio":
            path = "/t2a_v2"
        elif service_type == "video":
            path = "/video_generation"
        else:
            path = "/image_generation"
        return {
            "method": "POST",
            "url": join_provider_url(base_url, "/v1", path),
            "headers": _bearer_headers(api_key, True),
            "body": {},
        }

    if p == "vidu":
        return {
            "method": "POST",
            "url": join_provider_url(base_url, "", "/ent/v2/img2video"),
            "headers": _vidu_headers(api_key, True),
            "body": {},
        }

    if p == "local-sd":
        return {
            "method": "GET",
            "url": join_provider_url(base_url, "", "/sdapi/v1/samplers"),
            "headers": {},
            "body": None,
        }

    if p == "cosyvoice":
        return {"method": "GET", "url": join_provider_url(base_url, "", "/"), "headers": {}, "body": None}

    if p == "ollama":
        return {"method": "GET", "url": join_provider_url(base_url, "", "/api/tags"), "headers": {}, "body": None}

    return {
        "method": "GET",
        "url": join_provider_url(base_url, "", f"/{m}" if m else "/"),
        "headers": _bearer_headers(api_key),
        "body": None,
    }


def build_list_models(
    provider: str, base_url: str, api_key: str | None = None
) -> dict[str, Any] | None:
    """构造「列出可用模型」的请求描述；不支持在线列举的厂商返回 None。"""
    p = (provider or "").lower()

    if p in ("openai", "openrouter", "chatfire"):
        return {
            "method": "GET",
            "url": join_provider_url(base_url, "/v1", "/models"),
            "headers": _bearer_headers(api_key),
            "parse": lambda d: (
                [m.get("id") for m in d["data"] if isinstance(m, dict) and m.get("id")]
                if isinstance(d.get("data"), list)
                else []
            ),
        }

    if p == "gemini":
        url = join_provider_url(base_url, "/v1beta", "/models")
        if api_key:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}key={api_key}"
        return {
            "method": "GET",
            "url": url,
            "headers": {},
            "parse": lambda d: (
                [
                    re.sub(r"^models/", "", str(m.get("name") or ""))
                    for m in d["models"]
                    if isinstance(m, dict) and m.get("name")
                ]
                if isinstance(d.get("models"), list)
                else []
            ),
        }

    if p == "ollama":
        return {
            "method": "GET",
            "url": join_provider_url(base_url, "", "/api/tags"),
            "headers": {},
            "parse": lambda d: (
                [m.get("name") for m in d["models"] if isinstance(m, dict) and m.get("name")]
                if isinstance(d.get("models"), list)
                else []
            ),
        }

    # 阿里百炼 / 火山方舟 / MiniMax / Vidu / 本地 SD / CosyVoice 等无公开 list models 接口
    return None


def parse_models_safely(parse: Callable[[dict[str, Any]], list[str]], data: Any) -> list[str]:
    """``desc.parse(data)`` —— data 是「JSON 解析失败即 {}」，解析器内部一律做类型判断。"""
    try:
        return parse(data if isinstance(data, dict) else {})
    except Exception:  # noqa: BLE001
        return []
