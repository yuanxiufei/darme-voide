"""请求日志中间件 —— 移植 ``backend/src/middleware/logger.ts`` 的 ``requestLogger``（68 行）。

Node 侧全局 ``app.use('*', requestLogger)``：**请求前**打一行（``POST/PUT/PATCH`` 附带请求体，
>500 字符截断加 ``...``），**响应后**再打一行（状态码按类别着色 + 耗时 ms）。

⚠️ 与原 TS 的一处**有意差异**：多了 ``HTTP_LOG=0`` 开关（**默认开**，即与 Node 同行为）。
理由：本仓自检大量走 ``TestClient``，一次全量会打上千行噪音淹没 ``SUMMARY``；Node 侧没有
类似痛点（它不把测试灌进自己的 stdout）⇒ 这是**为测试环境**加的，不影响线上行为。

⚠️ 请求体是**在这里读一次**的：Starlette 会把 ``request.body()`` 缓存到 ``request._body``
⇒ 后续路由里 ``await request.body()`` / ``read_json()`` 仍能拿到同一份（已自检覆盖）。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from starlette.requests import Request

__all__ = ["request_logger", "status_color", "format_time"]

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"
_CYAN = "\x1b[36m"

#: 会打印请求体的方法（与原 TS 一致）
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})

#: 请求体截断长度（原 TS 是 500）
_BODY_LIMIT = 500

#: ⚠️ **有意加固**：请求体超过这个大小就**不读**（原 TS 无条件 `clone().text()`）。
#: 读 body 会让 Starlette 把整个请求体缓存进内存 —— 普通 JSON 无所谓，但本仓有
#: multipart 上传，几百 MB 的视频会在中间件里先被完整缓冲一次。这里按 `Content-Length`
#: 预判并跳过（**只影响超大请求体的日志**，正常请求与 Node 完全一致）。
_BODY_READ_LIMIT = 256 * 1024


def status_color(status: int) -> str:
    """状态码着色：5xx 红 / 4xx 黄 / 3xx 青 / 其余绿（原 TS ``statusColor``）。"""
    if status >= 500:
        return _RED
    if status >= 400:
        return _YELLOW
    if status >= 300:
        return _CYAN
    return _GREEN


def format_time() -> str:
    """``new Date().toLocaleTimeString('zh-CN', { hour12: false })`` 等价物（24 小时制）。"""
    return datetime.now().strftime("%H:%M:%S")


def _enabled() -> bool:
    """``HTTP_LOG=0`` 关日志（**默认开**，与 Node 同行为）；每次调用都读 ⇒ 便于测试切换。"""
    return os.environ.get("HTTP_LOG", "1") != "0"


async def request_logger(request: Request, call_next: Any) -> Any:
    """请求前 + 响应后各打一行（对齐原 TS 的格式与配色）。"""
    if not _enabled():
        return await call_next(request)

    method = request.method
    path = request.url.path
    started = datetime.now()
    # ⚠️ 与原 TS 一致：**先取时间戳**，请求行与响应行打的是**同一个**时间（不是响应时刻）
    stamp = format_time()

    body_info = ""
    if method in _BODY_METHODS:
        try:
            declared = request.headers.get("content-length")
            oversized = (declared is not None and declared.isdigit()
                         and int(declared) > _BODY_READ_LIMIT)
            text = "" if oversized else (await request.body()).decode("utf-8", errors="replace")
            if text:
                truncated = text if len(text) <= _BODY_LIMIT else text[:_BODY_LIMIT] + "..."
                body_info = f"\n  {_DIM}body: {truncated}{_RESET}"
        except Exception:  # noqa: BLE001 —— 与 TS 的空 catch 等价
            body_info = ""

    print(f"{_DIM}{stamp}{_RESET} {_CYAN}{method}{_RESET} {path}{body_info}")

    response = await call_next(request)

    elapsed_ms = int((datetime.now() - started).total_seconds() * 1000 + 0.5)
    color = status_color(response.status_code)
    print(f"{_DIM}{stamp}{_RESET} {_CYAN}{method}{_RESET} {path} "
          f"{color}{response.status_code}{_RESET} {_DIM}{elapsed_ms}ms{_RESET}")
    return response
