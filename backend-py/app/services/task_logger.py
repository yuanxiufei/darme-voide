"""任务日志与可追踪执行（trace）—— 移植 ``utils/task-logger.ts``（整文件）。

对齐 PenguinHarness 第 9 章的「单一事实来源」可观测性：同一 trace 内的所有日志自动附带
``traceId`` 与 ``elapsedMs``，把一次 Agent run / 生成任务的多条日志串起来。

⚠️ ``sanitize_value`` 是这个模块**最要紧**的部分：trace 会落盘并被前端回放，
所以它要负责**遮蔽密钥**（``token`` / ``authorization`` / ``api_key``…）、
**脱敏 URL 里的 key**、以及**截断 base64 / data-url**（否则一条 trace 能写出几十 MB）。
别为了"看起来更全"把它删掉。

⚠️ ``redact_url`` 原 TS 定义在本文件，Python 侧先在 ``services/provider_probe.py`` 落地
（``/test`` 的响应体要用它），这里直接复用，避免两份实现漂移。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from ..response import now
from .provider_probe import redact_url  # noqa: F401  (再导出，保持与原 TS 的模块归属直觉一致)
from .trace_store import append_trace_event

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"
_CYAN = "\x1b[36m"
_BLUE = "\x1b[34m"

#: 需要整体遮蔽的键名（小写包含匹配用）
_SECRET_KEYS = ("authorization", "api_key", "apikey", "apiKey", "token", "access_token")


def _color_for(level: str) -> str:
    if level == "SUCCESS":
        return _GREEN
    if level == "WARN":
        return _YELLOW
    if level == "ERROR":
        return _RED
    return _CYAN


def _time_text() -> str:
    """``toLocaleTimeString('zh-CN', {hour12:false})`` ⇒ ``HH:MM:SS``。"""
    return datetime.now().strftime("%H:%M:%S")


def _js_string(value: Any) -> str:
    """``${value}`` 的 JS 语义：``null`` / ``true`` / ``false`` 的写法与 Python 不同。"""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _truncate_string(value: str, edge: int = 120) -> str:
    """超长字符串保留首尾各 ``edge`` 字符，中间标注被裁掉多少。"""
    if len(value) <= edge * 2 + 24:
        return value
    return f"{value[:edge]}...<trimmed {len(value)} chars>...{value[-edge:]}"


def sanitize_value(value: Any) -> Any:
    """落盘前脱敏：遮蔽密钥、脱敏 URL、截断 base64。"""
    if value is None:
        return None
    if isinstance(value, str):
        return _truncate_string(value)
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, raw in value.items():
            lower = str(key).lower()
            if key in _SECRET_KEYS or any(
                part in lower for part in ("authorization", "token", "apikey", "api_key")
            ):
                out[key] = "***"
                continue
            if isinstance(raw, str) and (lower == "url" or lower.endswith("url")):
                out[key] = redact_url(raw)
                continue
            if isinstance(raw, str) and (
                lower == "data"
                or lower == "b64_json"
                or "base64" in lower
                or "audiohex" in lower
                or "inline" in lower
                or raw.startswith("data:image/")
            ):
                out[key] = _truncate_string(raw, 48)
                continue
            out[key] = sanitize_value(raw)
        return out
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return str(value)


def _safe_value(value: Any) -> Any:
    """console 输出用：对象转紧凑 JSON，其余原样。"""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _format_meta(meta: dict[str, Any] | None) -> str:
    if not meta:
        return ""
    entries = [f"{key}={_js_string(_safe_value(v))}" for key, v in meta.items()]
    return f" | {' '.join(entries)}" if entries else ""


def log_task(
    scope: str, action: str, meta: dict[str, Any] | None = None, level: str = "INFO"
) -> None:
    color = _color_for(level)
    print(f"{_DIM}{_time_text()}{_RESET} {color}[{scope}]{_RESET} {action}{_format_meta(meta)}")


def log_task_start(scope: str, action: str, meta: dict[str, Any] | None = None) -> None:
    log_task(scope, f"START {action}", meta, "INFO")


def log_task_progress(scope: str, action: str, meta: dict[str, Any] | None = None) -> None:
    log_task(scope, action, meta, "INFO")


def log_task_success(scope: str, action: str, meta: dict[str, Any] | None = None) -> None:
    log_task(scope, f"DONE {action}", meta, "SUCCESS")


def log_task_warn(scope: str, action: str, meta: dict[str, Any] | None = None) -> None:
    log_task(scope, action, meta, "WARN")


def log_task_error(scope: str, action: str, meta: dict[str, Any] | None = None) -> None:
    log_task(scope, f"ERROR {action}", meta, "ERROR")


def log_task_payload(scope: str, action: str, payload: Any) -> None:
    sanitized = sanitize_value(payload)
    serialized = (
        sanitized
        if isinstance(sanitized, str)
        else json.dumps(sanitized, ensure_ascii=False, indent=2)
    )
    print(f"{_DIM}{_time_text()}{_RESET} {_BLUE}[{scope}]{_RESET} {action}\n{serialized}")


class TraceHandle:
    """一次可追踪执行 —— 方法名与原 TS 的 TraceHandle 保持一致。"""

    __slots__ = ("trace_id", "_scope", "_meta", "_start_at")

    def __init__(self, scope: str, meta: dict[str, Any] | None) -> None:
        # `randomUUID().slice(0, 8)` ⇒ UUID 的前 8 个十六进制字符
        self.trace_id = uuid.uuid4().hex[:8]
        self._scope = scope
        self._meta = meta
        self._start_at = datetime.now()

    def _elapsed_ms(self) -> int:
        return int((datetime.now() - self._start_at).total_seconds() * 1000)

    def _with_trace(self, meta: dict[str, Any] | None) -> dict[str, Any]:
        return {**(meta or {}), "traceId": self.trace_id, "elapsedMs": self._elapsed_ms()}

    def _emit(self, level: str, action: str, meta: dict[str, Any]) -> None:
        """单点 emit：console 输出 + append-only JSONL 落盘（同一份数据，两个消费者）。"""
        log_task(self._scope, action, meta, level)
        append_trace_event(
            {
                "ts": now(),
                "traceId": self.trace_id,
                "scope": self._scope,
                "level": level,
                "action": action,
                "elapsedMs": meta.get("elapsedMs"),
                "meta": sanitize_value(meta),
            }
        )

    def progress(self, action: str, meta: dict[str, Any] | None = None) -> None:
        self._emit("INFO", action, self._with_trace(meta))

    def success(self, action: str, meta: dict[str, Any] | None = None) -> None:
        self._emit("SUCCESS", f"DONE {action}", self._with_trace(meta))

    def warn(self, action: str, meta: dict[str, Any] | None = None) -> None:
        self._emit("WARN", action, self._with_trace(meta))

    def error(self, action: str, meta: dict[str, Any] | None = None) -> None:
        self._emit("ERROR", f"ERROR {action}", self._with_trace(meta))

    def end(self) -> None:
        self._emit(
            "INFO",
            f"END trace {self.trace_id}",
            {**self._with_trace(self._meta), "totalMs": self._elapsed_ms()},
        )


def start_trace(
    scope: str, action: str, meta: dict[str, Any] | None = None
) -> TraceHandle:
    """开启一次 trace（返回句柄，首条事件为 ``START <action>``）。"""
    handle = TraceHandle(scope, meta)
    handle._emit("INFO", f"START {action}", handle._with_trace(meta))
    return handle
