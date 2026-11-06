"""ComfyUI 运行的**进度/预览**通道（``/api/v1/comfyui/runs/{id}/events`` 与 ``/progress``）。

与 ``routers/comfyui.py`` 的分工：
* ``comfyui.py`` 管**提交与查询**（一次性 JSON，状态以 DB 为准 ✓）；
* 本文件管**在途过程**（百分比 + 采样预览帧，SSE 增量 ✓）—— 两者读的是同一条 run ✓，
  但过程数据**不落库**（每秒几条，落库会把 SQLite 写满 ✗，与 ``sse_hub`` 同口径 ✓）。

⚠️ 三处与 ``auto_pipeline`` 的 SSE 同形，别改歪：
1. **先订阅再发快照** ✓：反过来的话，订阅与快照之间发生的事件会丢 ✗（那一窗口正好是"提交完成"的高频段）；
2. **心跳**（默认 15s）✓：本地 nginx/浏览器代理会掐掉长时间没字节的连接 ✗；
3. **客户端断开时清订阅与定时器** ✓：不清就是每刷一次页面漏一个订阅（内存慢慢涨 ✗）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.engine import Connection

from ..core.db import get_conn
from ..core.response import bad_request, not_found, parse_param_id, success
from ..services.comfyui import get_run
from ..services.comfyui_progress import snapshot, subscribe, websockets_available

router = APIRouter(prefix="/api/v1/comfyui", tags=["comfyui-events"])

_HEARTBEAT_INTERVAL_SECONDS = 15.0

#: 事件名映射 ✓：``error`` 是 ``EventSource`` 的**连接错误**同名词 ——
#: 直接用它发"工作流失败"，前端会把两者混成一条（还会互相覆盖错误文案 ✗）⇒ 收尾失败叫 ``failed`` ✓
_EVENT_NAMES = {"error": "failed"}


def _compact(payload: Any) -> str:
    """紧凑 JSON ✓（与 TS ``JSON.stringify`` 一致；``ensure_ascii=False`` 保住中文 ✓）。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _sse_frame(event: str, data: str) -> str:
    """SSE 一帧 ✓（``data`` 为空 ⇒ 只发事件名，用于心跳 ✓）。"""
    head = f"event: {event}\n"
    return f"{head}data: {data}\n\n" if data else f"{head}\n"


def _parse_run_id(run_id: str) -> int | None:
    parsed = parse_param_id(run_id)
    return int(parsed) if parsed else None


@router.get("/runs/{run_id}/progress")
def run_progress(run_id: str, conn: Connection = Depends(get_conn)):
    """进度快照（**不是流** ✓）：没有订阅需求的调用方（列表页、轮询兜底）用这个 ✓。

    ``progressAvailable`` 如实反映「本进程有没有 ``websockets``」：
    ``false`` ⇒ 只有 ``pullStatus``（排队/运行），**没有百分比** ✓ —— 别让前端画一个永远 0% 的条 ✗。
    """
    parsed = _parse_run_id(run_id)
    if parsed is None:
        return bad_request("invalid run id")
    row = get_run(conn, parsed)
    if row is None:
        return not_found("run not found")
    snap = snapshot(parsed)
    return success({
        **snap,
        "runId": parsed,
        "status": row.status,
        "step": row.step,
        "progressAvailable": websockets_available(),
    })


@router.get("/runs/{run_id}/events")
async def run_events(run_id: str, request: Request, conn: Connection = Depends(get_conn)):
    """SSE：``snapshot`` → 增量（``progress`` / ``preview`` / ``done`` / ``error`` / ``cancelled``）✓。"""
    parsed = _parse_run_id(run_id)
    if parsed is None:
        return bad_request("invalid run id")
    row = get_run(conn, parsed)
    if row is None:
        return not_found("run not found")

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    # ⚠️ 先订阅再发快照 ✓（见模块头第 1 条）
    unsubscribe = subscribe(parsed, lambda event: queue.put_nowait(("event", event)))

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
            await queue.put(("ping", None))

    heartbeat = asyncio.get_running_loop().create_task(_heartbeat())
    initial = {
        **snapshot(parsed),
        "runId": parsed,
        "status": row.status,
        "step": row.step,
        "progressAvailable": websockets_available(),
    }

    async def _relay() -> AsyncIterator[str]:
        try:
            yield _sse_frame("snapshot", _compact(initial))
            while True:
                if await request.is_disconnected():
                    break
                kind, payload = await queue.get()
                if kind == "ping":
                    yield _sse_frame("ping", "")
                else:
                    kind_name = str(payload.get("type"))
                    yield _sse_frame(_EVENT_NAMES.get(kind_name, kind_name), _compact(payload))
        finally:
            heartbeat.cancel()
            unsubscribe()

    return StreamingResponse(_relay(), media_type="text/event-stream", headers={
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })
