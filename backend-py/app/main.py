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

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles  # noqa: F401  (保留给后续完全迁移后使用)
from starlette.exceptions import HTTPException as StarletteHTTPException

from .http_logger import request_logger
from .config import (
    FRONTEND_DIST,
    NODE_BACKEND_URL,
    PROXY_TO_NODE,
    get_storage_root,
    server,
)
from .passthrough import (
    close_proxy_client,
    delegate,
    init_proxy_client,
)
from .response import not_found
from .routers.ai_voices import router as ai_voices_router
from .routers.characters import router as characters_router
from .routers.dramas import router as dramas_router
from .routers.episodes import router as episodes_router
from .routers.export import router as export_router
from .routers.app_settings import router as app_settings_router
from .routers.asset_versions import router as asset_versions_router
from .routers.ai_configs import providers_router as ai_providers_router
from .routers.ai_configs import router as ai_configs_router
from .routers.agent_configs import router as agent_configs_router
from .routers.generations import router as generations_router
from .routers.storage import router as storage_router
from .routers.style_profiles import router as style_profiles_router
from .routers.traces import router as traces_router
from .routers.usage import router as usage_router
from .routers.libraries import (
    character_library_router,
    costume_library_router,
    scene_library_router,
    weapon_library_router,
)
from .routers.presets import router as presets_router
from .routers.agent import router as agent_router
from .routers.auto_pipeline import router as auto_pipeline_router
from .routers.local_models import router as local_models_router
from .routers.compose import router as compose_router
from .routers.grid import router as grid_router
from .routers.images import router as images_router
from .routers.visual_graph import router as visual_graph_router
from .routers.webhooks import router as webhooks_router
from .routers.merge import router as merge_router
from .routers.preset_framework import router as preset_framework_router
from .routers.evaluation import router as evaluation_router
from .services.auto_pipeline import recover_auto_pipeline_on_startup
from .services.evaluation_scheduler import start_evaluation_scheduler
from .routers.mcp import router as mcp_router
from .routers.props import router as props_router
from .routers.videos import router as videos_router
from .routers.scenes import router as scenes_router
from .routers.skills import router as skills_router
from .routers.upload import router as upload_router
from .routers.storyboards import router as storyboards_router

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
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # 启动日志刻意用 ASCII：Windows 控制台默认代码页会把中文打成乱码（见 .codebuddy/memory）
    init_proxy_client()
    # 评测→优化 无人值守调度器（**默认关闭**，仅 evaluation.auto_optimize.enabled=true 时启动；
    # 关闭时只打一条 warn —— 与原 TS 的 index.ts 行为一致）
    start_evaluation_scheduler()
    # 全自动管线**崩溃恢复**：扫描中间态 episode 自动续跑（幂等，空库时直接返回）
    recover_auto_pipeline_on_startup()
    if PROXY_TO_NODE:
        print(f"[py] strangler ON: unmigrated domains -> {NODE_BACKEND_URL}")
    else:
        print("[py] PROXY_TO_NODE=0: unmigrated domains return 501")
    try:
        yield
    finally:
        await close_proxy_client()


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

# 请求日志（对齐 Node 的 `app.use('*', requestLogger)`）；`HTTP_LOG=0` 可关（自检降噪）
app.middleware("http")(request_logger)


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
app.include_router(characters_router)
app.include_router(ai_voices_router)
app.include_router(scenes_router)
app.include_router(props_router)
app.include_router(preset_framework_router)
app.include_router(storyboards_router)
app.include_router(videos_router)
app.include_router(compose_router)
app.include_router(agent_router)
app.include_router(auto_pipeline_router)
app.include_router(local_models_router)
app.include_router(mcp_router)
app.include_router(evaluation_router)
app.include_router(merge_router)
app.include_router(grid_router)
app.include_router(images_router)
app.include_router(visual_graph_router)
app.include_router(webhooks_router)
app.include_router(character_library_router)
app.include_router(scene_library_router)
app.include_router(weapon_library_router)
app.include_router(costume_library_router)
app.include_router(presets_router)
app.include_router(app_settings_router)
app.include_router(asset_versions_router)
app.include_router(traces_router)
app.include_router(storage_router)
app.include_router(usage_router)
app.include_router(agent_configs_router)
app.include_router(style_profiles_router)
app.include_router(generations_router)
app.include_router(ai_configs_router)
app.include_router(ai_providers_router)
app.include_router(skills_router)
app.include_router(upload_router)
app.include_router(export_router)
# 下一个域迁完后在这里 include（ai-configs / 本地模型 / evaluation / ...）


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
    return await delegate(request, path)


@app.api_route(
    "/api/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    include_in_schema=False,
)
async def api_v1_fallback(request: Request, path: str) -> Response:
    """兜底：未被任何已迁移路由命中的 /api/v1/* 一律走委派（反代 Node 或 501）。

    ⚠️ 兜底只能兜「没被别的路由先命中」的路径。像 ``GET /agent-configs/defaults``
    会被 ``GET /agent-configs/{id}`` 抢走 —— 那类遮蔽必须在各自 router 里显式声明委派，
    见 ``passthrough.py`` 与 ``tests/route_parity_test.py``。
    """
    return await delegate(request, f"api/v1/{path}")


@app.api_route(
    "/webhooks/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    include_in_schema=False,
)
async def webhooks_fallback(request: Request, path: str) -> Response:
    return await delegate(request, f"webhooks/{path}")


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
