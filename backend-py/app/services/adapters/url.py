"""Provider 基础 URL 拼接（移植自 ``backend/src/services/adapters/url.ts``）。

规则：把 ``baseUrl`` 尾部斜杠去掉，再把「必填前缀」和「路径」各补一个前导 ``/``
拼上去；**若 baseUrl 的路径已经以该前缀结尾，就不再重复拼**（这样用户在配置里
写 ``https://api.x.com/v1`` 或 ``https://api.x.com`` 都能用）。
"""
from __future__ import annotations

import re

__all__ = ["join_provider_url"]

_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def _normalize_segment(segment: str | None) -> str:
    if not segment:
        return ""
    return segment if segment.startswith("/") else f"/{segment}"


def join_provider_url(base_url: str | None, required_prefix: str, path: str) -> str:
    normalized_base = re.sub(r"/+$", "", base_url or "")
    normalized_prefix = _normalize_segment(required_prefix)
    normalized_path = _normalize_segment(path)

    if not normalized_base:
        return f"{normalized_prefix}{normalized_path}"

    # JS 的 `new URL(base)` 只接受**带 scheme 的绝对地址**，否则抛错并走 catch 分支。
    # Python 的 urlsplit 对 "example.com/x" 会把 example.com 当成 scheme，
    # 所以这里显式判 scheme + "://"，与 JS 的抛错条件对齐。
    if not _SCHEME.match(normalized_base):
        base_path = (
            normalized_base
            if normalized_base.endswith(normalized_prefix)
            else f"{normalized_base}{normalized_prefix}"
        )
        return f"{base_path}{normalized_path}"

    scheme, _, rest = normalized_base.partition("://")
    netloc, slash, raw_path = rest.partition("/")
    url_path = f"/{raw_path}" if slash else ""
    current_path = re.sub(r"/+$", "", url_path)
    merged_prefix = (
        current_path if current_path.endswith(normalized_prefix) else f"{current_path}{normalized_prefix}"
    )
    new_path = re.sub(r"/{2,}", "/", f"{merged_prefix}{normalized_path}")
    # JS 的 URL 会把 scheme 与 host 转小写（`url.toString()` 的规范化行为）
    return f"{scheme.lower()}://{netloc.lower()}{new_path}"
