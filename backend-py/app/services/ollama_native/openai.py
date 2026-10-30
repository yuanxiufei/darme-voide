"""**OpenAI 兼容层** —— 上游 ``openai/openai.go`` 的对等实现（转换函数，不含 HTTP 路由）。

本仓为它存在的具体理由：``app/services`` 里的 agent / 前端都用 **OpenAI 形态**的
``messages``（``{"role": "user", "content": "…"}``），而 ollama 原生 ``/api/chat`` 是另一套
（``images`` 独立字段、``tool_calls.function.arguments`` 是对象而非 JSON 串）。这层转换
就是那条缝。

**口径**：全部用 ``dict`` 进出（JSON 边界上再包 dataclass 只是多一层搬运）。凡是 Go 用
``omitempty`` 的字段，这里靠「**不加键**」实现 —— 差别很实在：``"content": ""`` 与没有
``content`` 键在 OpenAI 客户端里是两种行为（前者会覆盖已累积的正文，后者不会）。

⚠️ 五个照抄上游、但直觉会写错的地方，逐条在函数内注明：

1. ``temperature``/``top_p`` **缺省要填 1.0**（OpenAI 的默认值语义），不是「不传」；
2. 多段 content 会拆成**多条** ollama message，而 ``tool_calls`` 只挂到**最后一条**；
3. ``tool`` 角色的 ``name`` 为空时，从历史 ``tool_calls`` 里按 ``tool_call_id`` **倒序**反查；
4. 流式时 thinking 与正文混在一条响应里，必须**拆成两个 chunk**（否则某些客户端把
   ``reasoning`` 当前缀拼进正文）；
5. ``finish_reason`` 只有 ``stop`` 才允许被改写成 ``tool_calls`` —— 未完成的原因不能被吞。

**未搬**（并说明为什么不）：``responses.go``(1786 行)/``responses_compact.go``(917 行)
是 OpenAI **Responses API**（``/v1/responses``）与上下文压缩，本仓无任何消费方；
``FromTranscriptionRequest`` 属语音转写，本仓不做音频，且它只是拼一条 system prompt。
"""

from __future__ import annotations

import base64
import json
import struct
import time
from datetime import datetime, timezone
from typing import Any, Iterable

from . import names

__all__ = [
    "DEFAULT_MODEL_PATH",
    "FromChatRequest",
    "FromCompleteRequest",
    "FromCompletionToolCall",
    "FinishChunk",
    "ToChatCompletion",
    "ToCompleteChunk",
    "ToCompletion",
    "ToEmbeddingList",
    "ToListCompletion",
    "ToModel",
    "ToStreamChunks",
    "ToToolCalls",
    "ToUsage",
    "ToUsageGenerate",
    "decodeImageURL",
    "embeddingInputs",
    "floatsToBase64",
    "nameFromToolCallID",
    "NewError",
    "thinkFromReasoningEffort",
]

#: ``ToModel`` 里 ``owned_by`` 的取值来源：``model.ParseName(m).Namespace``。
DEFAULT_MODEL_PATH = "library"

FINGERPRINT = "fp_ollama"

_SUPPORTED_IMAGE_TYPES = ("jpeg", "jpg", "png", "webp")


class ConversionError(ValueError):
    """请求体不合法 —— 调用方应回 400（对应上游各处 ``errors.New`` + ``http.StatusBadRequest``）。"""


# --------------------------------------------------------------------------- #
# 错误
# --------------------------------------------------------------------------- #


def NewError(code: int, message: str) -> dict[str, Any]:
    """对应 ``NewError`` —— 把 HTTP 状态码映射成 OpenAI 的 ``error.type``。

    404 必须映射成 ``not_found_error``，否则 OpenAI SDK 不会抛 ``NotFoundError``，
    调用方的「模型不存在 → 自动 pull」分支就永远不触发。
    """
    if code == 400:
        error_type = "invalid_request_error"
    elif code == 404:
        error_type = "not_found_error"
    else:
        error_type = "api_error"
    return {"error": {"message": message, "type": error_type, "param": None, "code": None}}


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def _unix(text: Any, *, default_now: bool = False) -> int:
    """``created_at``（RFC3339 串）→ Unix 秒；解析不了时按需回落「现在」。"""
    if isinstance(text, (int, float)):
        return int(text)
    if isinstance(text, str) and text:
        raw = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
    return int(time.time()) if default_now else 0


def _metrics(response: dict[str, Any]) -> dict[str, Any]:
    return response.get("metrics") or {}


def _message(response: dict[str, Any]) -> dict[str, Any]:
    return response.get("message") or {}


def floatsToBase64(floats: Iterable[float]) -> str:
    """``[]float32`` → base64（**小端**，OpenAI 的 ``encoding_format: base64`` 就是这个格式）。"""
    values = list(floats)
    raw = struct.pack(f"<{len(values)}f", *values) if values else b""
    return base64.b64encode(raw).decode("ascii")


# --------------------------------------------------------------------------- #
# ollama → OpenAI
# --------------------------------------------------------------------------- #


def ToUsage(response: dict[str, Any]) -> dict[str, Any]:
    """对应 ``ToUsage``（``/api/chat`` 的响应）。"""
    metrics = _metrics(response)
    prompt = int(metrics.get("prompt_eval_count") or 0)
    completion = int(metrics.get("eval_count") or 0)
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    cached = metrics.get("prompt_eval_cached_count")
    if cached is not None:
        # 有缓存命中才给 details 键；给了却填 0 会让客户端的「缓存省钱」统计失真
        usage["prompt_tokens_details"] = {"cached_tokens": int(cached)}
    return usage


def ToUsageGenerate(response: dict[str, Any]) -> dict[str, Any]:
    """对应 ``ToUsageGenerate``（``/api/generate`` 的响应，形状相同）。"""
    return ToUsage(response)


def ToToolCalls(tool_calls: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """``api.ToolCall`` → OpenAI ``ToolCall``。

    ⚠️ ``arguments`` 必须是 **JSON 字符串**（OpenAI 的约定），而 ollama 给的是对象。
    序列化失败时上游是 ``slog.Error`` + ``continue``（跳过这条），这里同样跳过而不是抛
    —— 抛会让整轮对话因为一个畸形工具调用而全废。
    """
    out: list[dict[str, Any]] = []
    for call in tool_calls or ():
        function = call.get("function") or {}
        index = function.get("index", 0) or 0
        try:
            # 紧凑分隔符：`arguments` 是 OpenAI 约定的 JSON **字符串**，客户端会原样
            # json.loads 后再用；带空格虽也合法，但与 `JSON.stringify` 的输出不一致
            arguments = json.dumps(function.get("arguments"), ensure_ascii=False,
                                   separators=(",", ":"))
        except (TypeError, ValueError):
            continue
        out.append({
            "id": call.get("id") or "",
            "index": index,
            "type": "function",
            "function": {"name": function.get("name") or "", "arguments": arguments},
        })
    return out


def _logprobs(response: dict[str, Any]) -> dict[str, Any] | None:
    content = response.get("logprobs")
    return {"content": content} if content else None


def ToChatCompletion(identifier: str, response: dict[str, Any]) -> dict[str, Any]:
    """对应 ``ToChatCompletion``（非流式）。"""
    message = _message(response)
    tool_calls = ToToolCalls(message.get("tool_calls"))

    reason = response.get("done_reason") or ""
    if reason == "stop" and tool_calls:
        reason = "tool_calls"

    choice: dict[str, Any] = {
        "index": 0,
        "message": {
            "role": message.get("role") or "assistant",
            "content": message.get("content") or "",
            **({"reasoning": message["thinking"]} if message.get("thinking") else {}),
            **({"tool_calls": tool_calls} if tool_calls else {}),
        },
        "finish_reason": reason or None,
    }
    logprobs = _logprobs(response)
    if logprobs is not None:
        choice["logprobs"] = logprobs

    return {
        "id": identifier,
        "object": "chat.completion",
        "created": _unix(response.get("created_at")),
        "model": response.get("model") or "",
        "system_fingerprint": FINGERPRINT,
        "choices": [choice],
        "usage": ToUsage(response),
        **({"_debug_info": response["debug_info"]} if response.get("debug_info") else {}),
    }


def _toChunk(identifier: str, response: dict[str, Any], include_role: bool) -> dict[str, Any]:
    message = _message(response)
    tool_calls = ToToolCalls(message.get("tool_calls"))
    delta: dict[str, Any] = {}
    if include_role:
        delta["role"] = "assistant"
    # 上游把 content 声明成 any 且只有 omitempty：空串要**保留**（首块 / 纯思考块），
    # 否则客户端分不清「这一块没有正文」和「正文就是空的」
    delta["content"] = message.get("content") or ""
    if message.get("thinking"):
        delta["reasoning"] = message["thinking"]
    if tool_calls:
        delta["tool_calls"] = tool_calls

    choice: dict[str, Any] = {"index": 0, "delta": delta}
    logprobs = _logprobs(response)
    if logprobs is not None:
        choice["logprobs"] = logprobs

    return {
        "id": identifier,
        "object": "chat.completion.chunk",
        "created": _unix(response.get("created_at"), default_now=True),
        "model": response.get("model") or "",
        "system_fingerprint": FINGERPRINT,
        "choices": [choice],
    }


def ToStreamChunks(identifier: str, response: dict[str, Any],
                   include_role: bool) -> list[dict[str, Any]]:
    """对应 ``ToStreamChunks`` —— 必要时要拆成「思考块 + 内容块」两条。

    ⚠️ 拆开是**必须**的：OpenAI 的 ``delta.reasoning`` 与 ``delta.content`` 由客户端
    分别拼接，混在一条里会让支持 reasoning 的客户端把思考当正文的前缀。两条 chunk
    还必须**共用同一个 created**，否则前端时间线会跳。
    """
    message = _message(response)
    mixed = bool(message.get("thinking")) and (
        bool(message.get("content")) or bool(message.get("tool_calls")))
    if not mixed:
        return [_toChunk(identifier, response, include_role)]

    reasoning_chunk = _toChunk(identifier, response, include_role)
    reasoning_delta = reasoning_chunk["choices"][0]["delta"]
    reasoning_delta.pop("content", None)
    reasoning_delta.pop("tool_calls", None)
    # 思考块的 logprobs 会混进不属于它的 token，直接不给
    reasoning_chunk["choices"][0].pop("logprobs", None)

    content_chunk = _toChunk(identifier, response, False)
    content_chunk["created"] = reasoning_chunk["created"]
    content_chunk["choices"][0]["delta"].pop("reasoning", None)
    content_chunk["choices"][0].pop("logprobs", None)

    return [reasoning_chunk, content_chunk]


def FinishChunk(identifier: str, response: dict[str, Any],
                tool_call_sent: bool) -> dict[str, Any]:
    """终止块：``delta`` 为空对象，只带 ``finish_reason``（OpenAI 的规范形状）。"""
    reason = response.get("done_reason") or "stop"
    if reason == "stop" and tool_call_sent:
        reason = "tool_calls"
    return {
        "id": identifier,
        "object": "chat.completion.chunk",
        "created": _unix(response.get("created_at"), default_now=True),
        "model": response.get("model") or "",
        "system_fingerprint": FINGERPRINT,
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
    }


def ToCompletion(identifier: str, response: dict[str, Any]) -> dict[str, Any]:
    """对应 ``ToCompletion``（``/v1/completions`` 非流式）。"""
    reason = response.get("done_reason") or ""
    return {
        "id": identifier,
        "object": "text_completion",
        "created": _unix(response.get("created_at")),
        "model": response.get("model") or "",
        "system_fingerprint": FINGERPRINT,
        "choices": [{"text": response.get("response") or "", "index": 0,
                     "finish_reason": reason or None}],
        "usage": ToUsageGenerate(response),
    }


def ToCompleteChunk(identifier: str, response: dict[str, Any]) -> dict[str, Any]:
    """对应 ``ToCompleteChunk``。

    ⚠️ 上游这里用的是 ``time.Now()`` 而不是响应的 ``created_at``（与 ``ToCompletion``
    **不一致**）。照抄：流式分片的 created 每片都不同，客户端本来也不读它。
    """
    reason = response.get("done_reason") or ""
    return {
        "id": identifier,
        "object": "text_completion",
        "created": int(time.time()),
        "model": response.get("model") or "",
        "system_fingerprint": FINGERPRINT,
        "choices": [{"text": response.get("response") or "", "index": 0,
                     "finish_reason": reason or None}],
    }


def ToModel(response: dict[str, Any], model_name: str) -> dict[str, Any]:
    """对应 ``ToModel`` —— ``owned_by`` 取模型名的 namespace（默认 ``library``）。"""
    return {
        "id": model_name,
        "object": "model",
        "created": _unix(response.get("modified_at")),
        "owned_by": names.parse_name(model_name).namespace,
    }


def ToListCompletion(response: dict[str, Any]) -> dict[str, Any]:
    """对应 ``ToListCompletion`` —— ``id`` 优先取 ``model``，回落到 ``name``。"""
    data: list[dict[str, Any]] = []
    for item in response.get("models") or ():
        identifier = item.get("model") or item.get("name") or ""
        data.append({
            "id": identifier,
            "object": "model",
            "created": _unix(item.get("modified_at")),
            "owned_by": names.parse_name(identifier).namespace,
        })
    return {"object": "list", "data": data}


def ToEmbeddingList(model_name: str, response: dict[str, Any],
                    encoding_format: str = "") -> dict[str, Any]:
    """对应 ``ToEmbeddingList``。``encoding_format`` 为 ``base64`` 时转小端 float32 串。"""
    embeddings = response.get("embeddings")
    if embeddings is None:
        return {}
    use_base64 = (encoding_format or "").lower() == "base64"
    data = [
        {
            "object": "embedding",
            "embedding": floatsToBase64(vector) if use_base64 else list(vector),
            "index": index,
        }
        for index, vector in enumerate(embeddings)
    ]
    prompt_tokens = int(response.get("prompt_eval_count") or 0)
    return {
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


# --------------------------------------------------------------------------- #
# OpenAI → ollama
# --------------------------------------------------------------------------- #


def thinkFromReasoningEffort(effort: str) -> Any:
    """``reasoning_effort`` → ollama 的 ``think``（``False`` / 档位串 / ``None``）。

    OpenAI 的档位比 ollama 多两端（``minimal`` 更低、``xhigh``/``ultra`` 更高），上游选择
    **夹到最近档而不是 400** —— 客户端认为合法的值不该被拒。
    """
    if not effort:
        return None
    if effort == "none":
        return False
    if effort == "minimal":
        return "low"
    if effort in ("low", "medium", "high", "max"):
        return effort
    if effort in ("xhigh", "ultra"):
        return "max"
    raise ConversionError(
        f'invalid reasoning value: {effort!r} (must be "minimal", "low", "medium", '
        f'"high", "xhigh", "ultra", "max", or "none")')


def nameFromToolCallID(messages: list[dict[str, Any]], tool_call_id: str) -> str:
    """按 ``tool_call_id`` 反查工具名（**倒序**遍历 = 重复 id 时最后一个赢）。"""
    for message in reversed(messages):
        for call in message.get("tool_calls") or ():
            if (call.get("id") or "") == tool_call_id:
                return (call.get("function") or {}).get("name") or ""
    return ""


def decodeImageURL(url: str) -> bytes:
    """``data:image/png;base64,…`` → 原始字节。

    ⚠️ 只认 data URI：``http(s)`` 直接报错（上游明写 "image URLs are not currently
    supported"），因为 ollama 内部只吃字节，替用户下载图片会引入 SSRF 面。
    """
    if url.startswith(("http://", "https://")):
        raise ConversionError(
            "image URLs are not currently supported, please use base64 encoded data instead")

    if url.startswith("data:;base64,"):
        payload = url[len("data:;base64,"):]
    else:
        payload = None
        for kind in _SUPPORTED_IMAGE_TYPES:
            prefix = f"data:image/{kind};base64,"
            if url.startswith(prefix):
                payload = url[len(prefix):]
                break
        if payload is None:
            raise ConversionError("invalid image input")

    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as exc:
        raise ConversionError("invalid image input") from exc


def FromCompletionToolCall(tool_calls: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """OpenAI ``ToolCall`` → ``api.ToolCall``（``arguments`` 从 JSON 串解成对象）。"""
    out: list[dict[str, Any]] = []
    for call in tool_calls or ():
        function = call.get("function") or {}
        raw = function.get("arguments")
        if isinstance(raw, dict):
            arguments = raw              # 宽容：有的客户端直接给对象
        else:
            try:
                arguments = json.loads(raw or "{}")
            except (json.JSONDecodeError, TypeError) as exc:
                raise ConversionError("invalid tool call arguments") from exc
        out.append({
            "id": call.get("id") or "",
            "function": {"name": function.get("name") or "", "arguments": arguments},
        })
    return out


def _stop_option(stop: Any) -> dict[str, Any]:
    if isinstance(stop, str):
        return {"stop": [stop]}
    if isinstance(stop, (list, tuple)):
        return {"stop": [item for item in stop if isinstance(item, str)]}
    return {}


def FromChatRequest(body: dict[str, Any]) -> dict[str, Any]:
    """OpenAI ``ChatCompletionRequest`` → ollama ``ChatRequest``。

    返回的 dict 可以直接喂给 ``/api/chat``（字段名用 ollama 的 snake_case）。
    """
    raw_messages = body.get("messages") or []
    messages: list[dict[str, Any]] = []

    for message in raw_messages:
        role = message.get("role") or ""
        tool_name = ""
        if role.lower() == "tool":
            tool_name = message.get("name") or ""
            if not tool_name and message.get("tool_call_id"):
                tool_name = nameFromToolCallID(raw_messages, message["tool_call_id"])

        content = message.get("content")

        if isinstance(content, str):
            entry: dict[str, Any] = {"role": role, "content": content}
            if message.get("reasoning"):
                entry["thinking"] = message["reasoning"]
            if message.get("tool_calls"):
                entry["tool_calls"] = FromCompletionToolCall(message["tool_calls"])
            if tool_name:
                entry["tool_name"] = tool_name
            if message.get("tool_call_id"):
                entry["tool_call_id"] = message["tool_call_id"]
            messages.append(entry)
            continue

        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise ConversionError("invalid message format")
                kind = part.get("type")
                if kind == "text":
                    text = part.get("text")
                    if not isinstance(text, str):
                        raise ConversionError("invalid message format")
                    messages.append({"role": role, "content": text})
                elif kind == "image_url":
                    container = part.get("image_url")
                    if isinstance(container, dict):
                        url = container.get("url")
                    else:
                        url = container
                    if not isinstance(url, str):
                        raise ConversionError("invalid message format")
                    messages.append({"role": role, "content": "",
                                     "images": [decodeImageURL(url)]})
                elif kind == "input_audio":
                    container = part.get("input_audio")
                    if not isinstance(container, dict):
                        raise ConversionError("invalid input_audio format")
                    payload = container.get("data")
                    if not isinstance(payload, str):
                        raise ConversionError("invalid input_audio format: missing data")
                    try:
                        audio = base64.b64decode(payload, validate=True)
                    except (ValueError, TypeError) as exc:
                        raise ConversionError(f"invalid input_audio base64 data: {exc}") from exc
                    messages.append({"role": role, "content": "", "images": [audio]})
                else:
                    raise ConversionError("invalid message format")

            # 多段内容已拆成多条消息；工具调用只能挂到最后一条上（否则角色归属错乱）
            if messages and message.get("tool_calls"):
                last = messages[-1]
                last["tool_calls"] = FromCompletionToolCall(message["tool_calls"])
                if tool_name:
                    last["tool_name"] = tool_name
                if message.get("tool_call_id"):
                    last["tool_call_id"] = message["tool_call_id"]
                if message.get("reasoning"):
                    last["thinking"] = message["reasoning"]
            continue

        # content 为 None：只有带 tool_calls 时才合法
        if not message.get("tool_calls"):
            raise ConversionError(f"invalid message content type: {type(content).__name__}")
        entry = {
            "role": role,
            "tool_calls": FromCompletionToolCall(message["tool_calls"]),
        }
        if message.get("reasoning"):
            entry["thinking"] = message["reasoning"]
        if message.get("tool_call_id"):
            entry["tool_call_id"] = message["tool_call_id"]
        messages.append(entry)

    options: dict[str, Any] = {}
    options.update(_stop_option(body.get("stop")))
    if body.get("max_tokens") is not None:
        options["num_predict"] = body["max_tokens"]
    # ⚠️ 缺省填 1.0 —— OpenAI 的默认温度就是 1.0；不填会让 ollama 用自己更低的默认值，
    # 同一个请求在两条链路上产出完全不同
    options["temperature"] = body["temperature"] if body.get("temperature") is not None else 1.0
    if body.get("seed") is not None:
        options["seed"] = body["seed"]
    if body.get("frequency_penalty") is not None:
        options["frequency_penalty"] = body["frequency_penalty"]
    if body.get("presence_penalty") is not None:
        options["presence_penalty"] = body["presence_penalty"]
    options["top_p"] = body["top_p"] if body.get("top_p") is not None else 1.0

    fmt: Any = None
    response_format = body.get("response_format")
    if isinstance(response_format, dict):
        kind = (response_format.get("type") or "").strip().lower()
        if kind == "json_object":
            fmt = "json"
        elif kind == "json_schema":
            schema = response_format.get("json_schema") or {}
            fmt = schema.get("schema")

    effort = ""
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort") or ""
    elif body.get("reasoning_effort"):
        effort = body["reasoning_effort"]

    request: dict[str, Any] = {
        "model": body.get("model") or "",
        "messages": messages,
        "options": options,
        "stream": bool(body.get("stream")),
        "tools": body.get("tools") or [],
        "logprobs": bool(body.get("logprobs")),
        "top_logprobs": body.get("top_logprobs") or 0,
    }
    if fmt is not None:
        request["format"] = fmt
    think = thinkFromReasoningEffort(effort)
    if think is not None:
        request["think"] = think
    if body.get("_debug_render_only"):
        request["_debug_render_only"] = True
    return request


def FromCompleteRequest(body: dict[str, Any]) -> dict[str, Any]:
    """OpenAI ``CompletionRequest`` → ollama ``GenerateRequest``（legacy ``/v1/completions``）。"""
    options: dict[str, Any] = {}
    stop = body.get("stop")
    if isinstance(stop, str):
        options["stop"] = [stop]
    elif isinstance(stop, (list, tuple)):
        stops: list[str] = []
        for item in stop:
            if not isinstance(item, str):
                raise ConversionError(f"invalid type for 'stop' field: {type(item).__name__}")
            stops.append(item)
        options["stop"] = stops

    if body.get("max_tokens") is not None:
        options["num_predict"] = body["max_tokens"]
    options["temperature"] = body["temperature"] if body.get("temperature") is not None else 1.0
    if body.get("seed") is not None:
        options["seed"] = body["seed"]
    # ⚠️ 这两个字段在这个接口里**不是指针**，缺省就是 0，于是会被无条件写进 options。
    # 照抄上游（不判断 None），否则 0 值请求的采样行为会跟 ollama 原生不一致。
    options["frequency_penalty"] = body.get("frequency_penalty", 0)
    options["presence_penalty"] = body.get("presence_penalty", 0)
    top_p = body.get("top_p") or 0.0
    options["top_p"] = top_p if top_p != 0.0 else 1.0

    logprobs = False
    top_logprobs = 0
    requested = body.get("logprobs")
    if requested is not None and int(requested) > 0:
        logprobs = True
        top_logprobs = int(requested)

    request: dict[str, Any] = {
        "model": body.get("model") or "",
        "prompt": body.get("prompt") or "",
        "options": options,
        "stream": bool(body.get("stream")),
        "suffix": body.get("suffix") or "",
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if body.get("_debug_render_only"):
        request["_debug_render_only"] = True
    return request


#: ``/v1/embeddings`` 的 ``input`` 允许四种形态；ollama 的 ``input`` 只吃 str 或 list[str]。
def embeddingInputs(body: dict[str, Any]) -> list[str]:
    """把 OpenAI 的 ``input`` 归一化成「一批字符串」。

    * ``"text"`` → ``["text"]``
    * ``["a", "b"]`` → 原样
    * ``[1, 2, 3]``（单条 token 序列）与 ``[[1,2],[3,4]]``（多条）**无法**还原成文本，
      会抛 :class:`ConversionError` —— 上游同样不支持（ollama 内部按文本分词）。
    """
    value = body.get("input")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return list(value)
        raise ConversionError(
            "token-id input is not supported; pass text or a list of texts")
    raise ConversionError("invalid input field")
