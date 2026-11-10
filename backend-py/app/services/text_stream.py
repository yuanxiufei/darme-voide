"""**流式**文本生成（2026-11-01 新增 ✓，P0-1）。

与 :mod:`.text_generation` 的分工（**不要混用** ✓✗）：

* ``text_generation.generate_text``：**一次性任务** ✓（出提示词、拆镜、打标…）—— 调用方要的是
  「最后那段 JSON / 那句话」✓，中间过程没人看 ✓；
* 本模块：**人在看** ✓（对话、长文续写、智能体旁白）—— 每几百毫秒就得有字上屏 ✓。

三条设计口径（都是踩过/能踩的坑 ✓）：

1. **事件形状统一** ✓：``meta`` / ``thinking`` / ``content`` / ``usage`` / ``done`` / ``error``
   —— 本地（Ollama 原生、自研引擎）与远程（OpenAI 兼容）都归一到这六种 ✓，
   ````router`` 只做转发、不认 provider ✗。
2. **思考与正文分两栏** ✗✗：qwen3 这类思考模型的原生回复把推理写进 ``message.thinking`` ✓
   （Ollama 兼容端点则写进 ``reasoning_content`` ✓），合成一栏就是「content 恒为空」那个老坑的流式版 ✓。
3. **``emit`` 可能在工作线程里被调用** ✗（自研引擎的推理是同步阻塞的 ✓，只能 ``asyncio.to_thread``）——
   所以调用方**必须**自己转线程安全（router 走 ``loop.call_soon_threadsafe`` ✓）；
   本模块**不**替调用方猜事件循环 ✓。
4. **取消要落到线程里** ✗✗：调用方 ``task.cancel()`` 只能打断 ``await`` ✓，线程里的同步采样循环照跑 ✓✗
   （本地推理＝真烧卡 ⏳）。⇒ 本模块在 ``CancelledError`` 时置位传给引擎的 ``threading.Event`` ✓，
   引擎在**每一步**之前看一眼 ✓（见 :meth:`..engine.llm_backend.LlmBackend.generate_stream` 的 ``abort``）。
   ⚠️ 「取消任务」与「停止推理」是两件事 ✗ —— 只做前者等于没停 ✓✗。
"""
from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from typing import Any, Callable

import httpx
from sqlalchemy.engine import Connection

from ..core.response import js_truthy
from .adapters.registry import get_text_adapter
from .ai_providers import get_text_config
from .task_logger import log_task_error, log_task_start, log_task_success
from .vendor_errors import format_vendor_http_error

__all__ = ["EVENTS", "stream_text"]

#: 归一事件名 ✓（``router`` 与前端都按这一组认 ✓ —— 名字改了就是破坏性变更 ✗）
EVENTS = ("meta", "thinking", "content", "usage", "done", "error")

Emit = Callable[[str, dict[str, Any]], None]

#: 本地推理慢 ✓（装载 + 首 token 可能是几十秒 ✓）⇒ 读超时给足 ✓，连接超时照旧短 ✓
_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


async def stream_text(conn: Connection, user_prompt: str,
                      options: dict[str, Any] | None = None,
                      emit: Emit | None = None) -> dict[str, Any]:
    """流式生成 ✓ —— 边生成边 ``emit`` ✓，返回**汇总**（也用于 ``done`` 事件 ✓）。

    ``options``：``system`` / ``temperature`` / ``maxTokens`` / ``stop``（文字级停用词列表 ✓）。

    返回::

        {"text": 整段正文, "thinking": 整段思考, "model": 用了哪个模型,
         "usage": {...} | None, "streamed": 是否真流式, "elapsedMs": int, "firstTokenMs": int | None}
    """
    options = options or {}
    notify: Emit = emit or (lambda _name, _data: None)
    started = time.monotonic()
    text_parts: list[str] = []
    think_parts: list[str] = []
    state: dict[str, Any] = {"firstTokenMs": None, "usage": None}

    def _push(name: str, payload: dict[str, Any]) -> None:
        if name in ("content", "thinking") and state["firstTokenMs"] is None:
            state["firstTokenMs"] = int((time.monotonic() - started) * 1000)
        notify(name, payload)

    config = get_text_config(conn)
    provider = str(config.get("provider") or "").lower()
    models = (list(config["models"]) if config.get("models")
              else [m for m in [config.get("model")] if js_truthy(m)])
    if not models and provider != "engine":
        raise ValueError("文本模型未配置 — 请在设置中添加文本服务")

    adapter = None if provider == "engine" else get_text_adapter(provider)
    streamed = provider == "engine" or hasattr(adapter, "build_stream_request")
    _push("meta", {
        "provider": provider or "(未配置)",
        "models": models,
        "streamed": streamed,
        # ⚠️ 「不支持流式」不是错 ✓（gemini 那条路只能整段返回 ✓）—— 但必须**明说** ✗ 不许假装在流 ✗
        "note": "" if streamed else f"{provider} 不支持流式 ⇒ 整段返回后再一次性上屏",
    })

    log_task_start("TextStream", "generate", {"provider": provider, "streamed": streamed})
    try:
        if provider == "engine":
            model, usage = await _stream_engine(config, user_prompt, options, _push,
                                                text_parts, think_parts)
        elif streamed:
            model, usage = await _stream_http(config, adapter, models, user_prompt, options,
                                              _push, text_parts, think_parts)
        else:
            model, usage = await _whole_http(conn, models, user_prompt, options,
                                             _push, text_parts)
    except Exception as err:  # noqa: BLE001 —— 统一成 error 事件 + 上抛（由 router 收尾 ✓）
        log_task_error("TextStream", "failed", {"error": str(err)[:200]})
        _push("error", {"message": str(err)[:400]})
        raise

    state["usage"] = state["usage"] or usage
    result = {
        "text": "".join(text_parts).strip(),
        "thinking": "".join(think_parts).strip(),
        "model": model,
        "usage": state["usage"],
        "streamed": streamed,
        "elapsedMs": int((time.monotonic() - started) * 1000),
        "firstTokenMs": state["firstTokenMs"],
    }
    log_task_success("TextStream", "done",
                     {"model": model, "chars": len(result["text"]),
                      "firstTokenMs": result["firstTokenMs"], "elapsedMs": result["elapsedMs"]})
    _push("done", result)
    return result


# ── 自研引擎（进程内推理 ✓ 同步 ⇒ 必须进线程 ✗）


def _accepts_abort(fn: Any) -> bool:
    """这个后端收不收 ``abort=`` ✓（判**签名**，不是试一遍 ✗ —— 试错会把一次正常生成跑掉 ✓✗）。"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):     # C 实现/内置可调用拿不到签名 ⇒ 保守按"不收"✓
        return False
    if "abort" in params:
        return True
    # ``**kwargs`` 形：能收 ✓（替身/包装器多是这种 ✓）；但别把它当"支持取消"的证据 ✗ ——
    # 送进去总比 TypeError 强 ✓，真后端是不是每步看，是它自己的事 ✓。
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


async def _stream_engine(config: dict[str, Any], prompt: str, options: dict[str, Any],
                         push: Emit, text_parts: list[str], think_parts: list[str]
                         ) -> tuple[str, dict[str, Any] | None]:
    # 同包私有函数是有意的 ✓：后端**缓存与装载只有那一处** ✓（自己再建一份 ⇒ 显存里两份权重 ✗✗）
    from .text_generation import (  # noqa: PLC0415
        _get_engine_backend, engine_fast_kwargs, engine_gpu_layers, engine_graph_kwargs,
        engine_kv_cache_bytes, engine_kv_mode, engine_kv_window_kwargs, engine_spec_kwargs,
    )

    settings = config.get("settings") or {}
    gguf_path = settings.get("ggufPath") or settings.get("gguf_path")
    tokenizer_path = settings.get("tokenizerPath") or settings.get("tokenizer_path")
    if not gguf_path:
        raise ValueError("自研文本后端缺少 ggufPath（在文本服务 settings 里配 ✓）")
    device = str(settings.get("device") or "auto")
    dtype = settings.get("dtype")
    backend = _get_engine_backend(str(gguf_path), str(tokenizer_path) if tokenizer_path else None,
                                  device=device, dtype=str(dtype) if dtype else None,
                                  gpu_layers=engine_gpu_layers(settings),
                                  kv_mode=engine_kv_mode(settings),
                                  kv_cache_bytes=engine_kv_cache_bytes(settings),
                                  **engine_spec_kwargs(settings),
                                  **engine_fast_kwargs(settings),
                                  **engine_graph_kwargs(settings),
                                  **engine_kv_window_kwargs(settings))

    def _on_event(name: str, data: dict[str, Any]) -> None:
        # ⚠️ 这里在工作线程里 ✓（``generate_stream`` 是同步的 ✗）⇒ ``push`` 由调用方保证线程安全 ✓
        if name == "delta":
            chunk = data.get("text") or ""
            if chunk:
                text_parts.append(chunk)
                push("content", {"text": chunk})

    # ⚠️ 「取消任务」≠「停止推理」✗✗：``asyncio.to_thread`` 只是把同步函数丢进线程 ✓，
    # 取消 await 之后线程仍然会把 ``max_new_tokens`` 跑完 ✗（本地推理＝实打实烧卡 ✓）。
    # ⇒ 真正的止损口是采样循环里的这个 Event ✓，由取消路径置位 ✓（:meth:`LlmBackend.generate_stream` 每步看一眼）。
    # ⚠️ 只给**确实收**这个参数的后端送 ✓：老/替身后端没这一格时硬塞一个 kwarg 会 TypeError ✗
    # （那样"取消功能"反倒把正常流式弄崩了 ✓✗）—— 不收就退回「取消不了线程」的老行为 ✓，且明说 ✓。
    abort = threading.Event()
    extra: dict[str, Any] = {"abort": abort} if _accepts_abort(backend.generate_stream) else {}
    try:
        text = await asyncio.to_thread(
            backend.generate_stream, prompt,
            system=options.get("system"),
            temperature=float(options.get("temperature")
                              if options.get("temperature") is not None else 1.0),
            max_new_tokens=int(options.get("maxTokens") or 256),
            top_p=float(options.get("topP") or 1.0),
            top_k=int(options.get("topK") or 0),
            stop=tuple(str(s) for s in (options.get("stop") or []) if str(s)),
            on_event=_on_event,
            **extra,
        )
    except asyncio.CancelledError:
        abort.set()
        raise
    # ⚠️ 「拼回来的 must equal 返回的」✗：引擎侧若某轮增量被 hold（半个 UTF-8 ✓），
    # 拼接与返回值会短暂不一致 ✓ —— 以**返回值**为准校正 ✓，别让 done 与前面流出去的字对不上 ✗✗。
    streamed_text = "".join(text_parts)
    if text != streamed_text:
        text_parts.clear()
        text_parts.append(text)
    # ⚠️ 一个字都没出 ⇒ **不能当成功** ✗✗（本地引擎最常见的两种原因：context 撑爆、权重装错族 ✓）
    if not "".join(text_parts).strip():
        raise ValueError("文本模型返回为空（自研引擎）— 检查 ggufPath / 上下文长度 / 停用词设置")
    return str(settings.get("model") or gguf_path), None


# ── Ollama 原生 / OpenAI 兼容（真·流式 HTTP）


async def _stream_http(config: dict[str, Any], adapter: Any, models: list[Any], prompt: str,
                       options: dict[str, Any], push: Emit, text_parts: list[str],
                       think_parts: list[str]) -> tuple[Any, dict[str, Any] | None]:
    settings = config.get("settings") or {}
    sys_prompt = options.get("system")
    last_error: Exception | None = None
    usage: dict[str, Any] | None = None

    for model in models:
        messages: list[dict[str, str]] = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})
        request = adapter.build_stream_request(config, {
            "model": model,
            "messages": messages,
            "temperature": options.get("temperature"),
            "maxTokens": options.get("maxTokens"),
            "stop": options.get("stop"),
            "numCtx": settings.get("numCtx") or settings.get("num_ctx"),
        })
        body = json.dumps(request["body"], ensure_ascii=False, separators=(",", ":"))
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                async with client.stream("POST", request["url"], content=body,
                                         headers=request["headers"]) as response:
                    if response.status_code >= 400:
                        raw = (await response.aread()).decode("utf-8", "replace")
                        raise ValueError(format_vendor_http_error(response.status_code, raw, "text"))
                    async for line in response.aiter_lines():
                        event = adapter.parse_stream_line(line)
                        if event.get("error"):
                            raise ValueError(str(event["error"])[:300])
                        if event.get("thinking"):
                            think_parts.append(event["thinking"])
                            push("thinking", {"text": event["thinking"]})
                        if event.get("content"):
                            text_parts.append(event["content"])
                            push("content", {"text": event["content"]})
                        if event.get("usage"):
                            usage = event["usage"]
                            push("usage", usage)
                        if event.get("done"):
                            break
            if not "".join(text_parts).strip():
                # ⚠️ 流里**一个字都没有** ⇒ 不能当成功 ✗（这就是「拿不到文本」那个老坑的新表现 ✓）
                last_error = ValueError("文本模型返回为空")
                log_task_error("TextStream", "empty-stream", {"model": model})
                continue
            return model, usage
        except Exception as err:  # noqa: BLE001 —— 多模型 fallback：换下一个模型再试 ✓
            last_error = err
            # 已经流出去的字不能回滚 ✗ ⇒ 换模型只允许发生在「还没发出任何正文」时 ✓
            if "".join(text_parts).strip():
                raise
            log_task_error("TextStream", "model-error", {"model": model, "error": str(err)[:200]})
    if last_error is not None:
        raise last_error
    raise ValueError("文本生成失败")


# ── 不支持流式的 provider（gemini 等）✓ 明说 + 整段返回


async def _whole_http(conn: Connection, models: list[Any], prompt: str,
                      options: dict[str, Any], push: Emit,
                      text_parts: list[str]) -> tuple[Any, dict[str, Any] | None]:
    from .text_generation import generate_text  # noqa: PLC0415

    text = await generate_text(conn, prompt, options)
    text_parts.append(text)
    push("content", {"text": text})
    return (models[0] if models else ""), None
