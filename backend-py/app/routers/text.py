"""文本域 —— **流式**与**结构化**文本生成（2026-11-01 新增，P0-1 / P0-3）。

为什么单开一个域（而不是塞进 ``agent`` / ``engine``）✗：

* ``/api/v1/agent/{type}/chat`` 是**智能体**（带工具循环、上下文预算、写回 nudge ✓），
  它要的是「跑完给终态」✓；本域要的是「**边跑边看**」✗ —— 两者对延迟的容忍度差一个数量级 ✓；
* ``/api/v1/engine/generate`` 是**出图/出片**的任务队列 ✓，不是文本 ✗。

两个端点：

* ``POST /api/v1/text/stream``   —— SSE ✓，事件 ``meta`` / ``thinking`` / ``content`` / ``usage`` / ``done`` / ``error``；
* ``POST /api/v1/text/generate`` —— 一次性 ✓（可选 ``schema`` ⇒ 走校验+回喂重试 ✓）。

⚠️ SSE 四守则（少一条前端就"看起来卡住了" ✓✗）：

1. **``event:`` + ``data:`` + 空行** ✓（与 ``auto_pipeline`` 同一帧格式 ✓，前端只写一份解析 ✓）；
2. **心跳**：本地推理可能几十秒不出字 ✓（装载权重 ⏳）⇒ 每 15 秒发一个 ``: ping`` 注释帧 ✓；
3. **思考与正文分事件** ✓（``thinking`` / ``content``）—— 合成一个就把 qwen3 的推理当正文上屏了 ✗；
4. **收尾必到** ✓：成功发 ``done`` ✓、失败发 ``error`` ✓（**只发一次** ✗ —— 服务层发过这里就不补 ✗）、
   客户端断开则**取消任务** ✓；⚠️ 取消只是第一步 ✗✗：线程里的同步采样循环要靠 abort 事件才真停 ✓
   （否则「cancel 完还在烧卡 ⏳」—— 见 :mod:`..services.text_stream` 第 4 条 ✓）。
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.engine import Connection

from ..core.db import get_conn
from ..core.request_utils import read_json
from ..core.response import bad_request, success
from ..services.task_logger import log_task_error, log_task_start
from ..services.text_generation import generate_json, generate_text
from ..services.text_stream import stream_text

router = APIRouter(prefix="/api/v1/text", tags=["text"])

#: SSE 心跳间隔（秒）—— 首 token 之前**必须有东西**在动 ✓，否则代理/浏览器会以为断了 ✗
_HEARTBEAT_SECONDS = 15.0


def _sse_frame(event: str, data: str) -> str:
    """SSE 帧（``event:`` + ``data:`` + 空行结尾）—— 与 ``auto_pipeline._sse_frame`` 逐字同形 ✓。"""
    return f"event: {event}\ndata: {data}\n\n"


def _compact(payload: Any) -> str:
    """``JSON.stringify(payload)`` —— **紧凑分隔符** ✓（与全仓同口径 ✓）。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _text_options(body: dict[str, Any]) -> dict[str, Any]:
    """从请求体取生成参数 ✓（只认白名单 ✓，未知键不下传 ✗）。"""
    options: dict[str, Any] = {}
    for key in ("system", "temperature", "maxTokens", "topP", "topK", "stop"):
        if body.get(key) is not None:
            options[key] = body[key]
    stops = options.get("stop")
    if isinstance(stops, str):
        options["stop"] = [stops]
    elif isinstance(stops, list):
        options["stop"] = [str(item) for item in stops if str(item)]
    return options


@router.post(
    "/stream",
    response_class=StreamingResponse,
    # 契约写进 openapi ✓：调用方（含前端代码生成）能看出这**不是**一次性 JSON ✗
    responses={200: {
        "description": "SSE 事件流（event: meta / thinking / content / usage / done / error）",
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    }},
)
async def text_stream(request: Request, conn: Connection = Depends(get_conn)):
    """**流式**文本生成（SSE ✓）。体：``{prompt, system?, temperature?, maxTokens?, stop?}``。"""
    body = await read_json(request)
    prompt = body.get("prompt") if body.get("prompt") is not None else body.get("message")
    if not isinstance(prompt, str) or not prompt.strip():
        return bad_request("prompt is required")

    options = _text_options(body)
    log_task_start("TextStream", "http", {"chars": len(prompt), "stop": options.get("stop") or []})

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    #: 服务层已经发过 ``error`` 事件 ⇒ 路由**不再补发** ✗（同一句报错上屏两次＝前端弹两个红条 ✗）
    seen: dict[str, bool] = {"error": False}

    def emit(name: str, data: dict[str, Any]) -> None:
        # ⚠️ 自研引擎那条路会在**工作线程**里回调 ✓ ⇒ 必须经 ``call_soon_threadsafe`` ✗ 别直接 put ✗
        if name == "error":
            seen["error"] = True
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (name, data))
        except RuntimeError:
            # 事件循环已关（进程退出/测试收尾）⇒ 工作线程里再抛就是「没人接的异常」noise ✗
            pass

    def _close() -> None:
        """放"没有更多事件了"的哨兵 ✓（``None`` 而不是元组 —— ``_relay`` 靠它收流 ✓）。"""
        try:
            loop.call_soon_threadsafe(queue.put_nowait, None)
        except RuntimeError:
            pass

    async def _run() -> None:
        try:
            await stream_text(conn, prompt.strip(), options, emit)
        except Exception as err:  # noqa: BLE001 —— 收尾统一成 error 帧 ✓（HTTP 头早发出去了 ✗ 不能再改状态码 ✗）
            log_task_error("TextStream", "http-failed", {"error": str(err)[:200]})
            # ⚠️ ``stream_text`` 抛错前**已经** ``_push("error")`` 过一次 ✓ ⇒ 这里只补「它没发过」的情形 ✗
            if not seen["error"]:
                emit("error", {"message": str(err)[:400]})
        finally:
            _close()

    task = asyncio.create_task(_run())

    async def _relay():
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                if item is None:
                    return
                name, data = item
                yield _sse_frame(name, _compact(data))
                if name in ("done", "error"):
                    continue          # 让 ``_run`` 自己收尾（``done`` 之后可能还有日志 ✓）
        finally:
            # ⚠️ 客户端断开 ⇒ **取消推理** ✗✗（不取消的话本地引擎会继续把 max_new_tokens 烧完 ✗）
            # 注意这里是**两段式** ✓：``cancel()`` 只停 asyncio 这一侧 ✓，线程里的采样循环靠
            # :mod:`..services.text_stream` 收到 ``CancelledError`` 后置位 abort 事件才真停 ✓
            # （只写 cancel() 而不管线程，就是"取消之后还在烧卡"那个坑 ⏳✗）
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    return StreamingResponse(
        _relay(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关掉 nginx 一类反代的缓冲 ✓（否则「流式」会被攒成一整段再吐 ✗✗）
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/generate")
async def text_generate(request: Request, conn: Connection = Depends(get_conn)):
    """一次性文本生成 ✓（带 ``schema`` 时走「校验 → 回喂重试」✓，见 :func:`generate_json` ✓）。"""
    body = await read_json(request)
    prompt = body.get("prompt") if body.get("prompt") is not None else body.get("message")
    if not isinstance(prompt, str) or not prompt.strip():
        return bad_request("prompt is required")

    options = _text_options(body)
    schema = body.get("schema")
    log_task_start("TextGen", "http", {"chars": len(prompt), "schema": schema is not None})
    try:
        if schema is None:
            text = await generate_text(conn, prompt.strip(), options)
            return success({"text": text, "schemaOk": None, "repairs": 0})
        data, meta = await generate_json(conn, prompt.strip(), schema, options)
    except Exception as err:  # noqa: BLE001 —— 与 agent 域同口径：业务失败 400 ✓ 不 500 ✗
        log_task_error("TextGen", "http-failed", {"error": str(err)[:200]})
        return bad_request(str(err) or "text generation failed")

    # ⚠️ 校验过不去**不许静默返回** ✗✗（宁可 400 让调用方知道 ✓，也别把不合规的 JSON 当成功 ✓）
    if not meta.get("schemaOk"):
        return bad_request(f"模型输出不符合 schema：{meta.get('schemaError') or '未知原因'}")
    return success({"text": meta.get("raw") or "", "data": data,
                    "schemaOk": True, "repairs": meta.get("repairs", 0)})
