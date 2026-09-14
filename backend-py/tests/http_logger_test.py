"""S7 自检：请求日志中间件（``app/http_logger.py``，移植 ``middleware/logger.ts``）。

锁四件事：**格式与配色**（请求行/响应行同一时间戳、状态码按类别着色、耗时带 ms）、
**body 截断规则**（仅 POST/PUT/PATCH，>500 字符加 ``...``）、**读完 body 后路由仍能取到**
（Starlette 缓存），以及 ``HTTP_LOG=0`` 开关。

运行::

    ./.venv/Scripts/python.exe tests/http_logger_test.py
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="httplog_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from starlette.responses import JSONResponse  # noqa: E402

from app import http_logger  # noqa: E402
from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import dramas  # noqa: E402

_R: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _R.append((name, bool(condition), detail))


class _FakeRequest:
    """最小 Request 替身（只用到 method / url.path / body()）。"""

    def __init__(self, method: str, path: str, body: bytes = b"",
                 headers: dict[str, str] | None = None) -> None:
        self.method = method
        self.url = type("U", (), {"path": path})()
        self.headers = headers or {}
        self._body = body

    async def body(self) -> bytes:
        return self._body


def run_middleware(method: str, path: str, body: bytes = b"", status: int = 200,
                   headers: dict[str, str] | None = None) -> str:
    async def call_next(_request):
        return JSONResponse({"ok": True}, status_code=status)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        asyncio.run(http_logger.request_logger(
            _FakeRequest(method, path, body, headers), call_next))
    return buffer.getvalue()


def main() -> int:  # noqa: C901
    # ── 1. 格式与配色 ──
    out = run_middleware("GET", "/api/v1/dramas")
    lines = out.splitlines()
    check("格式: GET 打两行（请求行 + 响应行），且**没有** body 行",
          len(lines) == 2 and "body:" not in out, lines)
    check("格式: 请求行 = `dim(时间) cyan(GET) 路径`；响应行追加 `绿(200) dim(3ms)`",
          http_logger._DIM in lines[0] and http_logger._CYAN in lines[0]
          and "/api/v1/dramas" in lines[0] and http_logger._GREEN in lines[1]
          and "200" in lines[1] and "ms" in lines[1], lines)
    check("格式: 两行用**同一个**时间戳（对齐原 TS 的 stamp 只取一次）",
          lines[0].split(http_logger._RESET)[1] == lines[1].split(http_logger._RESET)[1], lines)
    check("配色: 4xx 黄 / 5xx 红 / 3xx 青",
          http_logger.status_color(404) == http_logger._YELLOW
          and http_logger.status_color(500) == http_logger._RED
          and http_logger.status_color(302) == http_logger._CYAN
          and http_logger.status_color(200) == http_logger._GREEN)

    # ── 2. body 打印与截断 ──
    out = run_middleware("POST", "/api/v1/dramas", b'{"title":"x"}')
    check("body: POST 有 body 时多打一行 `body: {...}`（GET 不打）",
          "body:" in out and '{"title":"x"}' in out and len(out.splitlines()) == 3, out)
    long_body = ("A" * 700).encode()
    out = run_middleware("POST", "/api/v1/x", long_body)
    printed = [line for line in out.splitlines() if "body:" in line][0]
    # ⚠️ 行尾还有 reset 色码 ⇒ 不能直接 endsWith("...")，按「500 个 A 紧跟省略号」判
    check("body: >500 字符截断为 500 + `...`（长度与后缀都要对）",
          printed.count("A") == 500 and ("A" * 500 + "...") in printed, len(printed))
    out = run_middleware("POST", "/api/v1/x", b"")
    check("body: 空 body 不打 body 行", "body:" not in out and len(out.splitlines()) == 2)
    # ⚠️ 有意加固：超大请求体（按 Content-Length 预判）**不读**，避免 multipart 大文件被整包缓冲
    out = run_middleware("POST", "/api/v1/x", b"x" * 10, headers={"content-length": "99999999"})
    check("body: Content-Length 超阈值 -> **不读也不打** body（避免大上传被缓冲进内存）",
          "body:" not in out and len(out.splitlines()) == 2, out)
    out = run_middleware("POST", "/api/v1/x", b'{"a":1}', headers={"content-length": "7"})
    check("body: Content-Length 正常 -> 照常打 body（加固不影响常规请求）",
          "body:" in out and '{"a":1}' in out, out)

    # ── 3. 读完 body 后路由仍能取到（Starlette 缓存）──
    os.environ.pop("HTTP_LOG", None)
    from fastapi.testclient import TestClient  # noqa: PLC0415

    client = TestClient(app)
    stamp = "2026-01-01T00:00:00.000Z"
    with engine.begin() as conn:
        created = conn.execute(dramas.insert().values(
            title="日志剧", created_at=stamp, updated_at=stamp))
        drama_id = int(created.lastrowid)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        response = client.post("/api/v1/dramas", json={"title": "新剧"})
    # ⚠️ `POST /dramas` 成功是 **201**（原 TS 的 `created`），不是 200
    check("缓存: 中间件读过 body 后，路由**仍能解析 JSON**（不是空 body 报 400）",
          response.status_code == 201, (response.status_code, response.text[:80]))
    check("缓存: 同时确实打出了含请求体的日志行",
          "body:" in buffer.getvalue() and "新剧" in buffer.getvalue(),
          buffer.getvalue()[-200:])

    # ── 4. 开关 ──
    os.environ["HTTP_LOG"] = "0"
    check("开关: HTTP_LOG=0 -> 一行都不打（默认是开，与 Node 同行为）",
          run_middleware("GET", "/api/v1/dramas") == "")
    os.environ.pop("HTTP_LOG", None)
    check("开关: 去掉后恢复打印", run_middleware("GET", "/api/v1/dramas") != "")

    failed = [item for item in _R if not item[1]]
    for name, passed, detail in _R:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_R) - len(failed)}/{len(_R)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
