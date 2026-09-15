"""全自动管线编排路由 —— 与 ``routes/auto-pipeline.ts``（101 行）对齐。**4 端点**。

* ``POST /api/v1/auto-pipeline/run``           创建 Drama + 触发后台管线
* ``GET  /api/v1/auto-pipeline/status/{id}``   查询整剧进度
* ``GET  /api/v1/auto-pipeline/stream/{id}``   **SSE 进度流**（全仓唯一的 SSE 端点）
* ``POST /api/v1/auto-pipeline/resume/{id}``   幂等续跑（崩溃恢复 / 媒体补跑）

⚠️ 五处保真点：

1. 两个 POST 的错误是 **500 + ``{code, message}``**（``server_error``），**不是** 400；
   且文案兜底分别是 ``run failed`` / ``resume failed``；
2. ``run`` 的 premise 校验是 **truthy**（非字符串也算缺）⇒ ``premise: 123`` 不会被拦；
3. ``status`` / ``resume`` 的 dramaId 用 ``parseParamId``（``Number.isFinite && > 0``），
   非法回 **400 ``invalid dramaId``**（不是 404）；
4. **SSE 回放协议**：**先 subscribe 再发快照** —— 否则「订阅→快照」窗口内的事件会丢；
   历史状态由快照提供，流只推增量（不缓存、不回放 buffer）；
5. 心跳 **20s 一次** ``event: ping``（保持长连接不被代理/浏览器断开）；写失败即断连，
   ``finally`` 里清理订阅与心跳任务；``X-Accel-Buffering: no`` 禁用反向代理缓冲。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..core.request_utils import read_json
from ..core.response import bad_request, not_found, parse_param_id, server_error, success
from app.agent.auto_pipeline import (
    get_auto_pipeline_status,
    resume_auto_pipeline,
    run_auto_pipeline,
)
from ..services.sse_hub import subscribe_pipeline

router = APIRouter(prefix="/api/v1/auto-pipeline", tags=["auto-pipeline"])

#: SSE 心跳间隔（秒；原 TS 是 ``setInterval(..., 20000)``）
_HEARTBEAT_INTERVAL_SECONDS = 20.0


def _sse_frame(event: str, data: str) -> str:
    """SSE 帧（``event:`` + ``data:`` + 空行结尾）。"""
    return f"event: {event}\ndata: {data}\n\n"


def _compact(payload: Any) -> str:
    """``JSON.stringify(payload)`` —— **紧凑分隔符**（守卫在盯）。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@router.post("/run")
async def run_pipeline(request: Request):
    """创建 Drama + Episodes 并触发后台全自动管线，**立即返回** ``{dramaId, episodeIds}``。"""
    try:
        body = await read_json(request)
        premise = body.get("premise")
        if not (isinstance(premise, str) and premise.strip()):
            return bad_request("premise 不能为空")
        return success(run_auto_pipeline(body))
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价（返回 500）
        return server_error(str(err) or "run failed")


@router.get("/status/{drama_id}")
def pipeline_status(drama_id: str):
    """查询整剧管线进度（drama 不存在 → 404）。"""
    parsed = parse_param_id(drama_id)
    if not parsed:
        return bad_request("invalid dramaId")
    status = get_auto_pipeline_status(int(parsed))
    if status is None:
        return not_found("drama not found")
    return success(status)


@router.get("/stream/{drama_id}")
async def stream_pipeline(drama_id: str):
    """SSE 进度流（对齐 PenguinHarness 16.4 节「SSE 端点的交付保证」）。"""
    parsed = parse_param_id(drama_id)
    if not parsed:
        return bad_request("invalid dramaId")
    if get_auto_pipeline_status(int(parsed)) is None:
        return not_found("drama not found")

    target = int(parsed)
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    # ⚠️ 先订阅再发快照：避免「订阅 → 快照」窗口内发生的事件丢失
    unsubscribe = subscribe_pipeline(target, lambda event: queue.put_nowait(("event", event)))

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
            await queue.put(("ping", None))

    # ⚠️ 同 `auto_pipeline._spawn`：必须用 `get_running_loop()`（本函数是 async ⇒ 一定在循环上）
    heartbeat = asyncio.get_running_loop().create_task(_heartbeat())

    async def _relay() -> AsyncIterator[str]:
        try:
            # 初始快照（历史状态走快照，流只推增量）
            yield _sse_frame("snapshot", _compact(get_auto_pipeline_status(target)))
            while True:
                kind, payload = await queue.get()
                if kind == "ping":
                    yield _sse_frame("ping", "")
                else:
                    yield _sse_frame(str(payload.get("type")), _compact(payload))
        finally:
            # 客户端断开（生成器被取消）= 原 TS 的 stream.onAbort：清定时器与订阅
            heartbeat.cancel()
            unsubscribe()

    return StreamingResponse(_relay(), media_type="text/event-stream", headers={
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })


@router.post("/resume/{drama_id}")
async def resume_pipeline(drama_id: str, request: Request):
    """续跑（幂等）：崩溃恢复 / 手动重跑 / **媒体补跑**（body 可带 override）。"""
    parsed = parse_param_id(drama_id)
    if not parsed:
        return bad_request("invalid dramaId")
    try:
        body = await read_json(request)
        resume_auto_pipeline(int(parsed), body)
        return success({"dramaId": int(parsed), "resumed": True})
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价（返回 500）
        return server_error(str(err) or "resume failed")
