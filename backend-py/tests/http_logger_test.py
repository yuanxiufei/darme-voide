"""S7 自检：请求日志中间件（``app/http_logger.py``，移植 ``middleware/logger.ts``）。

锁五件事：**格式与配色**（请求行/响应行同一时间戳、状态码按类别着色、耗时带 ms）、
**body 截断规则**（仅 POST/PUT/PATCH，>500 字符加 ``...``）、**读完 body 后路由仍能取到**
（Starlette 缓存）、**编码安全**（控制台编码打不出的字符不许把请求炸成 500），
以及 ``HTTP_LOG=0`` 开关。

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
from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import dramas  # noqa: E402

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


def run_middleware_stream(method: str, path: str, body: bytes = b"",
                          headers: dict[str, str] | None = None,
                          stream: object | None = None) -> tuple[BaseException | None, str]:
    """用**真会撞倒的** stdout 跑一遍中间件，返回 ``(抛出的异常, 输出文本)``。

    与 ``run_middleware`` 的区别只有 stdout：这里用 ``gbk + errors="strict"``
    （Windows 控制台的真实形状），``print`` 遇到 GBK 外字符会**真的抛**
    ``UnicodeEncodeError`` —— 这正是把上传请求炸成 500 的那个洞。
    ``StringIO`` 从不挑字符，所以老判据一直看不到它。
    """
    async def call_next(_request):
        return JSONResponse({"ok": True}, status_code=200)

    buffer = io.BytesIO()
    target = stream or io.TextIOWrapper(buffer, encoding="gbk", errors="strict", newline="")
    raised: BaseException | None = None
    with contextlib.redirect_stdout(target):  # type: ignore[arg-type]
        try:
            asyncio.run(http_logger.request_logger(
                _FakeRequest(method, path, body, headers), call_next))
        except BaseException as exc:  # noqa: BLE001 —— 这里要的就是「有没有炸」
            raised = exc
        try:
            target.flush()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    return raised, buffer.getvalue().decode("gbk", errors="replace")


class _BrokenConsole:
    """write 一律 OSError 的 stdout（控制台被关掉／管道已断）—— 日志也不许把请求带崩。"""

    encoding = "gbk"

    def write(self, *_args: object, **_kwargs: object) -> int:
        raise OSError("console gone")

    def flush(self, *_args: object, **_kwargs: object) -> None:
        raise OSError("console gone")


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

    # ── 2b. multipart 不打 body ──
    multipart_body = (b"--b\r\nContent-Disposition: form-data; name=\"files\"; filename=\"a.txt\""
                      b"\r\n\r\n\xef\xbb\xbfhello\r\n--b--\r\n")
    out = run_middleware("POST", "/api/v1/knowledge/upload", multipart_body,
                         headers={"content-type": "multipart/form-data; boundary=b"})
    check("body: multipart 请求**不打** body（文件内容是二进制噪音，还专藏 BOM/emoji）",
          "body:" not in out and len(out.splitlines()) == 2, out[:120])

    # ── 2c. 编码安全（本轮实测事故）──
    #   真机：上传一份记事本存的 UTF-8-BOM txt ⇒ 500。栈顶只是「打日志」那一行，
    #   根因是 cp936 控制台编不出 \ufeff ⇒ print 抛 ⇒ 中间件是必经之路 ⇒ 整请求炸。
    raised, text = run_middleware_stream(
        "POST", "/api/v1/x", '{"t":"\ufeff😀生僻字𠀋"}'.encode("utf-8"),
        headers={"content-type": "application/json"})
    check("编码: body 含 BOM/emoji/GBK 外字符时**不抛**（否则整个请求 500、路由一行没跑）",
          raised is None, type(raised).__name__ if raised else "")
    check("编码: 打不出的字符降级成 `?`，能打出的中文照常显示（不是整行丢掉）",
          "生僻字" in text and "?" in text, text[-160:])
    raised, _ = run_middleware_stream(
        "GET", "/api/v1/knowledge/tree?dir=剧本知识库")
    check("编码: 纯中文路径在 cp936 下本来就没问题（加固不误伤）", raised is None, raised)
    raised, _ = run_middleware_stream("GET", "/api/v1/dramas", stream=_BrokenConsole())
    check("编码: stdout 整个坏掉（OSError）也不许把请求带崩（日志不是业务的一部分）",
          raised is None, type(raised).__name__ if raised else "")

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
    # ⚠️ **显式发 UTF-8 字节** ✓（2026-09-25）：用 `client.post(json=...)` 时，httpx 走
    #   `json.dumps` 默认 `ensure_ascii=True` ⇒ **线上字节里是 `\u65b0\u5267`** ✓ ⇒ 中间件
    #   如实回显就**不含"新剧"** ✗（此前这条在 Windows 上一直红 ✓✗ —— 判据问的和它想验的不是一回事 ✗）。
    #   自己发字节 ⇒ 既钉住"日志里能看到请求体" ✓，又顺带钉住**按 UTF-8 解码** ✓
    #   （若按 locale 解（zh-CN ⇒ GBK ✓）会成乱码 ⇒ 这条当场红 ✓）。
    with contextlib.redirect_stdout(buffer):
        response = client.post("/api/v1/dramas", content='{"title": "新剧"}'.encode("utf-8"),
                               headers={"content-type": "application/json"})
    # ⚠️ `POST /dramas` 成功是 **201**（原 TS 的 `created`），不是 200
    check("缓存: 中间件读过 body 后，路由**仍能解析 JSON**（不是空 body 报 400）",
          response.status_code == 201, (response.status_code, response.text[:80]))
    check("缓存: 同时确实打出了含请求体的日志行（且按 UTF-8 解 ✓ 不是乱码 ✓）",
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
