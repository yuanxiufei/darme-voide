"""FastAPI 入口 —— 对齐 ``backend/src/index.ts`` 的挂载结构。

路由注册顺序是**语义的一部分**，不要随意调整：

1. ``GET /api/v1/health``
2. 已迁移的业务域（当前只有 ``/api/v1/dramas``）
3. ``/static/*``           —— 生成文件（Range 分段，视频进度条依赖）
4. ``/api/v1/{path}``      —— **兜底**：未迁移的域 → 反代 Node（或返回 501）
5. ``/webhooks/{path}``    —— 同上（Node 侧挂在 /api/v1 之外）
6. ``/{path}``             —— 前端产物（SPA fallback 到 index.html）

第 4/5 步就是绞杀者模式的接缝：**已迁移的域由 Python 服务，未迁移的域透明转发给 Node**，
每迁完一个域就把它的路由注册上去，Node 侧对应模块即可停用 —— 全程系统可用、随时可回退。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles  # noqa: F401  (保留给后续完全迁移后使用)
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import (
    FRONTEND_DIST,
    NODE_BACKEND_URL,
    PROXY_TO_NODE,
    get_storage_root,
    server,
)
from .response import not_found
from .routers.dramas import router as dramas_router
from .routers.episodes import router as episodes_router

# Node ``index.ts`` 的 STATIC_MIME：显式声明以覆盖 mimetypes 猜不到的扩展名
# （尤其 .srt / .vtt 字幕 —— 猜错会让字幕轨加载失败）
STATIC_MIME: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
    ".gif": "image/gif", ".svg": "image/svg+xml", ".bmp": "image/bmp", ".ico": "image/x-icon",
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".mkv": "video/x-matroska",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".srt": "application/x-subrip", ".vtt": "text/vtt", ".txt": "text/plain",
    ".json": "application/json", ".pdf": "application/pdf", ".zip": "application/zip",
}

# 逐跳首部（RFC 7230）：转发时必须剥掉，否则会污染下游连接语义
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}

# 反代到 Node 的共享客户端（连接池复用；SSE 长连接也走它）
_proxy_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _proxy_client
    # 启动日志刻意用 ASCII：Windows 控制台默认代码页会把中文打成乱码（见 .codebuddy/memory）
    if PROXY_TO_NODE:
        _proxy_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))
        print(f"[py] strangler ON: unmigrated domains -> {NODE_BACKEND_URL}")
    else:
        print("[py] PROXY_TO_NODE=0: unmigrated domains return 501")
    try:
        yield
    finally:
        if _proxy_client is not None:
            await _proxy_client.aclose()
            _proxy_client = None


app = FastAPI(
    title="Drama Studio API (Python)",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=server["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 统一异常 → 契约信封
# 前端只认 {code, message}，FastAPI 默认的 {"detail": ...} 会让错误文案变成
# 「请求失败 (422)」—— 所以这里必须全部收口。
# ---------------------------------------------------------------------------

@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "request failed"
    return JSONResponse(status_code=exc.status_code, content={"code": exc.status_code, "message": message})


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
    message = f"参数校验失败: {loc} {first.get('msg', '')}".strip()
    return JSONResponse(status_code=400, content={"code": 400, "message": message})


@app.exception_handler(Exception)
async def _unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"code": 500, "message": str(exc)})


# ---------------------------------------------------------------------------
# 1) 健康检查（注意：Node 侧**没有**包信封，直接返回裸对象，保持一致）
# ---------------------------------------------------------------------------

@app.get("/api/v1/health")
def health() -> dict[str, str]:
    from .response import now

    return {"status": "ok", "timestamp": now()}


# ---------------------------------------------------------------------------
# 2) 已迁移的业务域
# ---------------------------------------------------------------------------

app.include_router(dramas_router)
app.include_router(episodes_router)
# 下一个域迁完后在这里 include（characters / scenes / storyboards / props / ...）


# ---------------------------------------------------------------------------
# 3) 静态文件（Range 分段）
# ---------------------------------------------------------------------------

@app.get("/static/{rel:path}")
def serve_static(rel: str) -> Response:
    storage_root = Path(get_storage_root()).resolve()
    abs_path = (storage_root / rel).resolve()

    # 路径穿越防护（对齐 Node：既不能在根之外，也不能是根本身）
    if abs_path != storage_root and not str(abs_path).startswith(str(storage_root) + "\\") and not str(
        abs_path
    ).startswith(str(storage_root) + "/"):
        return JSONResponse(status_code=404, content={"code": 404, "message": "not found"})
    if not abs_path.is_file():
        return JSONResponse(status_code=404, content={"code": 404, "message": "not found"})

    # 刻意用 FileResponse 而不是手写 Range：Starlette 原生支持 Range/If-Range/416，
    # 且不会像 Node 版那样把整个视频读进内存（大文件更稳）。行为与原实现一致。
    media_type = STATIC_MIME.get(abs_path.suffix.lower(), "application/octet-stream")
    return FileResponse(
        abs_path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ---------------------------------------------------------------------------
# 4/5) 绞杀者接缝：未迁移的域
# ---------------------------------------------------------------------------

async def _proxy_to_node(request: Request, path: str) -> Response:
    assert _proxy_client is not None
    target = f"{NODE_BACKEND_URL}/{path}"
    headers = [
        (k, v) for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
    ]
    try:
        upstream_request = _proxy_client.build_request(
            request.method,
            target,
            headers=headers,
            content=await request.body(),
            params=list(request.query_params.multi_items()),
        )
        upstream = await _proxy_client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "code": 502,
                "message": (
                    f"反代到 Node 后端失败（{NODE_BACKEND_URL}）：{exc}。"
                    "请确认 Node 后端已启动，或关闭 PROXY_TO_NODE。"
                ),
            },
        )

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP}
    # aiter_raw 保证 SSE 不被缓冲（管线进度流依赖），且透传压缩体与 content-encoding 一致
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=resp_headers,
        background=BackgroundTask(upstream.aclose),
    )


def _not_migrated(path: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "code": 501,
            "message": (
                f"/{path} 尚未迁移到 Python 后端。"
                "设置 PROXY_TO_NODE=1 可反代到 Node 后端保持可用。"
            ),
        },
    )


@app.api_route(
    "/api/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    include_in_schema=False,
)
async def api_v1_fallback(request: Request, path: str) -> Response:
    if not PROXY_TO_NODE:
        return _not_migrated(f"api/v1/{path}")
    return await _proxy_to_node(request, f"api/v1/{path}")


@app.api_route(
    "/webhooks/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    include_in_schema=False,
)
async def webhooks_fallback(request: Request, path: str) -> Response:
    if not PROXY_TO_NODE:
        return _not_migrated(f"webhooks/{path}")
    return await _proxy_to_node(request, f"webhooks/{path}")


# ---------------------------------------------------------------------------
# 6) 前端产物（SPA fallback）—— 必须最后注册
# ---------------------------------------------------------------------------

@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str) -> Response:
    index = FRONTEND_DIST / "index.html"
    if not FRONTEND_DIST.is_dir():
        return JSONResponse(
            status_code=404,
            content={
                "code": 404,
                "message": (
                    f"前端产物目录不存在：{FRONTEND_DIST}。"
                    "开发模式请单独启动前端（cd frontend && npx nuxt dev --port 3013）。"
                ),
            },
        )

    if full_path:
        candidate = (FRONTEND_DIST / full_path).resolve()
        # 目录穿越防护：只有确实落在 dist 内的文件才直出
        if str(candidate).startswith(str(FRONTEND_DIST.resolve())) and candidate.is_file():
            return FileResponse(candidate)

    if index.is_file():
        return FileResponse(index, media_type="text/html")
    return not_found("index.html not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=server["host"],
        port=server["port"],
        reload=True,
    )
