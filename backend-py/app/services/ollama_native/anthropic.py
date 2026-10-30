"""**Anthropic Messages 兼容层** —— 上游 ``anthropic/anthropic.go`` 的对等实现。

把 Claude 的 ``/v1/messages`` 请求体翻成 ollama 的 ``/api/chat``，把响应翻回去，含**流式
事件状态机**。与 :mod:`.openai` 同样的口径：``dict`` 进出，``omitempty`` 靠「不加键」。

⚠️ 六个必须照抄、否则静默出错的点：

1. ``tool_result`` 在 **user 轮**里必须排在正文**之前**（Anthropic 的语义），
   在 assistant 轮里才排在后面 —— 顺序错了模型会以为工具还没返回就继续说话；
2. ``input_tokens`` **不含**缓存命中的部分（``total - cached``），直接给总数会让
   客户端的「缓存省了多少」统计变成负数；
3. ``ContentBlock`` 的 ``text``/``thinking`` 是**指针语义**：空串也要输出字段，
   因为 Anthropic SDK 靠「字段存在」累积流式分片；
4. 流式里 thinking 与 text 是两个 block，切换时必须发 ``content_block_stop`` 并让 index 前进；
5. ``mapStopReason`` 里 **有 tool_calls 就一律 ``tool_use``**，压过 ``done_reason``；
6. ``max_tokens`` 是 Anthropic 的必填项，要直接写进 ``num_predict``（缺了就变 ollama 默认值）。

**未搬**：``WebSearch``（1297 行附近）—— 它要请求 ``ollama.com`` 的云搜索接口，凭据来源
（``~/.ollama/id_ed25519`` 签名 + 云端账号）不在本仓职责内，硬编只会得到一个 401 的转发层。
``TraceMessagesRequest`` 系列同理（上游是 slog 结构化追踪，本仓用标准 ``logging``）。
"""

from __future__ import annotations

import base64
import json
import secrets
from typing import Any

__all__ = [
    "StreamConverter",
    "EstimateInputTokens",
    "FromMessagesRequest",
    "GenerateMessageID",
    "NewError",
    "ToMessagesResponse",
    "UsageFromMetrics",
    "convertMessage",
    "convertTool",
    "convertToolResultContent",
    "mapStopReason",
    "resolveImageSource",
]

WEB_SEARCH_TOOL_NAME = "web_search"
WEB_SEARCH_DESCRIPTION = (
    "Search the web for current information. Use this to find up-to-date information "
    "about any topic.")


class ConversionError(ValueError):
    """请求体不合法 —— 调用方回 400（Anthropic 的 ``invalid_request_error``）。"""


# --------------------------------------------------------------------------- #
# 错误
# --------------------------------------------------------------------------- #

_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


def NewError(code: int, message: str) -> dict[str, Any]:
    """对应 ``NewError`` —— 形状是 ``{"type": "error", "error": {...}}``。"""
    return {
        "type": "error",
        "error": {"type": _ERROR_TYPES.get(code, "api_error"), "message": message},
    }


def GenerateMessageID() -> str:
    """``msg_<24位hex>``（对应 ``GenerateMessageID``）。"""
    return "msg_" + secrets.token_bytes(12).hex()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_bytes(12).hex()}"


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def _metrics(response: dict[str, Any]) -> dict[str, Any]:
    return response.get("metrics") or {}


def intValue(value: int | None) -> int:
    """对应 ``intValue`` —— ``None`` 当 0（Python 里 ``None`` 不能参与减法）。"""
    return 0 if value is None else int(value)


def UsageFromMetrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """对应 ``UsageFromMetrics`` —— **input_tokens 不含缓存命中**。

    ⚠️ 缓存数还必须 clamp 到 ``[0, total]``：上游某些后端会报出比总 prompt 还大的
    缓存计数（跨请求共享 KV 时的计数口径问题），不 clamp 就会得到负的 input_tokens。
    """
    total = max(0, int(metrics.get("prompt_eval_count") or 0))
    cached: int | None = None
    raw_cached = metrics.get("prompt_eval_cached_count")
    if raw_cached is not None:
        cached = min(max(0, int(raw_cached)), total)
    usage: dict[str, Any] = {"input_tokens": total - intValue(cached),
                             "output_tokens": int(metrics.get("eval_count") or 0)}
    if cached is not None:
        usage["cache_read_input_tokens"] = cached
    return usage


def resolveImageSource(source: dict[str, Any] | None) -> bytes:
    """只接受 ``{"type": "base64", "media_type": …, "data": …}``（URL 一律拒）。"""
    if not source:
        raise ConversionError("invalid image source")
    kind = source.get("type") or ""
    if kind != "base64":
        raise ConversionError(
            f"invalid image source type: {kind}. Only base64 images are supported.")
    data = source.get("data")
    if not isinstance(data, str):
        raise ConversionError("invalid base64 image data: missing data")
    try:
        return base64.b64decode(data, validate=True)
    except (ValueError, TypeError) as exc:
        raise ConversionError(f"invalid base64 image data: {exc}") from exc


def convertToolResultContent(content: Any) -> tuple[str, list[bytes]]:
    """``tool_result.content`` → ``(文本, 图片列表)``。只认 ``text`` 与 ``image`` 两种块。"""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []

    texts: list[str] = []
    images: list[bytes] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
        elif kind == "image":
            source = block.get("source")
            if not isinstance(source, dict):
                raise ConversionError("invalid tool_result image source")
            images.append(resolveImageSource(source))
    return "".join(texts), images


def formatWebSearchToolResultContent(content: Any) -> str:
    """把 ``web_search_tool_result`` 拼成给模型看的一串 ``- 标题: 链接``。"""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if content.get("type") == "web_search_tool_result_error":
            code = content.get("error_code") or ""
            return "web_search_tool_result_error" + (f": {code}" if code else "")
        # 紧凑分隔符：这段文本会**原样喂给模型**，Go 侧 `json.Marshal` 也不带空格
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    if isinstance(content, list):
        lines: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "web_search_result":
                lines.append(f"- {item.get('title') or ''}: {item.get('url') or ''}\n")
            elif kind == "web_search_tool_result_error":
                code = item.get("error_code") or ""
                return "web_search_tool_result_error" + (f": {code}" if code else "")
        return "".join(lines)
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def convertTool(tool: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """对应 ``convertTool`` —— 返回 ``(ollama 工具, 是否内置服务端工具)``。"""
    kind = tool.get("type") or ""
    if kind.startswith("web_search"):
        return {
            "type": "function",
            "function": {
                "name": WEB_SEARCH_TOOL_NAME,
                "description": WEB_SEARCH_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": ["string"],
                            "description": "The search query to look up on the web",
                        },
                    },
                },
            },
        }, True

    schema = tool.get("input_schema")
    parameters: dict[str, Any] = {}
    if schema:
        if isinstance(schema, str):
            try:
                parameters = json.loads(schema)
            except json.JSONDecodeError as exc:
                raise ConversionError(
                    f"invalid input_schema for tool {tool.get('name')!r}: {exc}") from exc
        else:
            parameters = dict(schema)
    return {
        "type": "function",
        "function": {
            "name": tool.get("name") or "",
            "description": tool.get("description") or "",
            "parameters": parameters,
        },
    }, False


# --------------------------------------------------------------------------- #
# 请求转换
# --------------------------------------------------------------------------- #


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def convertMessage(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """对应 ``convertMessage`` —— 一条 Anthropic 消息可能变**多条** ollama 消息。"""
    role = (msg.get("role") or "").lower()

    content = msg.get("content")
    if isinstance(content, str):
        blocks: list[dict[str, Any]] = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        blocks = [block for block in content if isinstance(block, dict)]
    else:
        blocks = []

    texts: list[str] = []
    images: list[bytes] = []
    tool_calls: list[dict[str, Any]] = []
    thinking = ""
    tool_results: list[dict[str, Any]] = []

    for block in blocks:
        kind = block.get("type")

        if kind == "text":
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)

        elif kind == "image":
            images.append(resolveImageSource(block.get("source")))

        elif kind in ("tool_use", "server_tool_use"):
            call_id = block.get("id") or ""
            name = block.get("name") or ""
            if kind == "tool_use":
                # 内置的 server_tool_use 允许缺字段（它是服务端自己产生的），
                # 但用户侧 tool_use 缺 id/name 就是坏请求，必须报错
                if not call_id:
                    raise ConversionError("tool_use block missing required 'id' field")
                if not name:
                    raise ConversionError("tool_use block missing required 'name' field")
            tool_calls.append({
                "id": call_id,
                "function": {"name": name, "arguments": block.get("input") or {}},
            })

        elif kind == "tool_result":
            text, result_images = convertToolResultContent(block.get("content"))
            entry: dict[str, Any] = {"role": "tool", "content": text}
            if result_images:
                entry["images"] = result_images
            if block.get("tool_use_id"):
                entry["tool_call_id"] = block["tool_use_id"]
            tool_results.append(entry)

        elif kind == "thinking":
            value = block.get("thinking")
            if isinstance(value, str):
                thinking = value

        elif kind == "web_search_tool_result":
            tool_results.append({
                "role": "tool",
                "content": formatWebSearchToolResultContent(block.get("content")),
                **({"tool_call_id": block["tool_use_id"]} if block.get("tool_use_id") else {}),
            })

        # 其他块（redacted_thinking、document 等）静默跳过，同上游 default 计数后忽略

    messages: list[dict[str, Any]] = []

    # ⚠️ user 轮的工具结果必须排在最前 —— Anthropic 语义是「先把工具结果交给模型，
    # 再补用户这一轮的话」。顺序颠倒会让模型重复调用工具。
    if role == "user" and tool_results:
        messages.extend(tool_results)

    if texts or images or tool_calls or thinking:
        entry = {"role": role, "content": "".join(texts)}
        if images:
            entry["images"] = images
        if tool_calls:
            entry["tool_calls"] = tool_calls
        if thinking:
            entry["thinking"] = thinking
        messages.append(entry)

    if not (role == "user" and tool_results):
        messages.extend(tool_results)

    return messages


def FromMessagesRequest(body: dict[str, Any]) -> dict[str, Any]:
    """Anthropic ``MessagesRequest`` → ollama ``ChatRequest``。"""
    messages: list[dict[str, Any]] = []

    system = _system_text(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})

    for index, raw in enumerate(body.get("messages") or ()):
        if not isinstance(raw, dict):
            raise ConversionError(f"invalid message at index {index}")
        try:
            messages.extend(convertMessage(raw))
        except ConversionError as exc:
            raise ConversionError(f"messages[{index}] ({raw.get('role')}): {exc}") from exc

    options: dict[str, Any] = {}
    if body.get("max_tokens") is not None:
        options["num_predict"] = body["max_tokens"]
    if body.get("temperature") is not None:
        options["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        options["top_p"] = body["top_p"]
    if body.get("top_k") is not None:
        options["top_k"] = body["top_k"]
    stops = body.get("stop_sequences")
    if stops:
        options["stop"] = list(stops)

    raw_tools = [tool for tool in (body.get("tools") or ()) if isinstance(tool, dict)]
    has_builtin_web_search = any(
        (tool.get("type") or "").startswith("web_search") for tool in raw_tools)

    tools: list[dict[str, Any]] = []
    for tool in raw_tools:
        # 内置 web_search 与用户自定义同名工具会互相抢调用；内置优先，丢掉自定义的
        if (has_builtin_web_search
                and not (tool.get("type") or "").startswith("web_search")
                and tool.get("name") == WEB_SEARCH_TOOL_NAME):
            continue
        converted, _ = convertTool(tool)
        tools.append(converted)

    effort = ""
    output_config = body.get("output_config")
    if isinstance(output_config, dict):
        effort = (output_config.get("effort") or "").strip().lower()
        if effort == "xhigh":
            effort = "high"

    think: Any = None
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("type") == "enabled":
            think = True
        elif thinking.get("type") == "disabled":
            think = False
    if think is None and isinstance(output_config, dict) and effort in ("high", "medium", "low", "max"):
        think = effort

    request: dict[str, Any] = {
        "model": body.get("model") or "",
        "messages": messages,
        "options": options,
        "stream": bool(body.get("stream")),
        "tools": tools,
    }
    if think is not None:
        request["think"] = think
    return request


# --------------------------------------------------------------------------- #
# 响应转换
# --------------------------------------------------------------------------- #


def mapStopReason(reason: str, hasToolCalls: bool) -> str:
    """对应 ``mapStopReason`` —— 有工具调用时**压过** ``done_reason``。"""
    if hasToolCalls:
        return "tool_use"
    if reason == "stop":
        return "end_turn"
    if reason == "length":
        return "max_tokens"
    if reason:
        return "stop_sequence"
    return ""


def ToMessagesResponse(identifier: str, response: dict[str, Any]) -> dict[str, Any]:
    """对应 ``ToMessagesResponse``（非流式）。"""
    message = response.get("message") or {}
    content: list[dict[str, Any]] = []

    if message.get("thinking"):
        content.append({"type": "thinking", "thinking": message["thinking"]})
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or ():
        function = call.get("function") or {}
        content.append({
            "type": "tool_use",
            "id": call.get("id") or "",
            "name": function.get("name") or "",
            "input": function.get("arguments") or {},
        })

    return {
        "id": identifier,
        "type": "message",
        "role": "assistant",
        "model": response.get("model") or "",
        "content": content,
        "stop_reason": mapStopReason(response.get("done_reason") or "",
                                     bool(message.get("tool_calls"))),
        "usage": UsageFromMetrics(_metrics(response)),
    }


# --------------------------------------------------------------------------- #
# 流式
# --------------------------------------------------------------------------- #


class StreamConverter:
    """对应 ``StreamConverter`` —— 有状态的事件生成器。

    用法：对每个 ollama 分片调 :meth:`process`，把返回的 ``(event, data)`` 逐条写成
    ``event: …\\ndata: …\\n\\n``（SSE）。状态（当前 block index、是否已开思考/正文块、
    已发过的 tool_call id）都在实例里，**不能**每片新建。
    """

    __slots__ = ("id", "model", "first_write", "content_index", "input_tokens",
                 "cache_read_tokens", "output_tokens", "estimated_input_tokens",
                 "thinking_started", "thinking_done", "text_started", "tool_calls_sent")

    def __init__(self, identifier: str, model: str, estimated_input_tokens: int = 0) -> None:
        self.id = identifier
        self.model = model
        self.first_write = True
        self.content_index = 0
        self.input_tokens = 0
        self.cache_read_tokens: int | None = None
        self.output_tokens = 0
        self.estimated_input_tokens = estimated_input_tokens
        self.thinking_started = False
        self.thinking_done = False
        self.text_started = False
        self.tool_calls_sent: set[str] = set()

    # -- 事件小工具 --
    def _stop(self) -> tuple[str, dict[str, Any]]:
        event = {"type": "content_block_stop", "index": self.content_index}
        return "content_block_stop", event

    def _start(self, block: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        event = {"type": "content_block_start", "index": self.content_index,
                 "content_block": block}
        return "content_block_start", event

    def _delta(self, delta: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        event = {"type": "content_block_delta", "index": self.content_index, "delta": delta}
        return "content_block_delta", event

    def process(self, response: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """对应 ``Process`` —— 一个 ollama 分片 → 若干 Anthropic 事件。"""
        events: list[tuple[str, dict[str, Any]]] = []
        message = response.get("message") or {}
        thinking = message.get("thinking") or ""
        content = message.get("content") or ""

        if self.first_write:
            self.first_write = False
            usage = UsageFromMetrics(_metrics(response))
            self.input_tokens = usage["input_tokens"]
            self.cache_read_tokens = usage.get("cache_read_input_tokens")
            # 首片常常还没有 metrics；用请求侧的粗估撑住 input_tokens，别让客户端看到 0
            if (self.input_tokens == 0 and intValue(self.cache_read_tokens) == 0
                    and self.estimated_input_tokens > 0):
                self.input_tokens = self.estimated_input_tokens
            events.append(("message_start", {
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "usage": {
                        "input_tokens": self.input_tokens,
                        **({"cache_read_input_tokens": self.cache_read_tokens}
                           if self.cache_read_tokens is not None else {}),
                        "output_tokens": 0,
                    },
                },
            }))

        if thinking and not self.thinking_done:
            if self.text_started:
                events.append(self._stop())
                self.content_index += 1
                self.text_started = False
            if not self.thinking_started:
                self.thinking_started = True
                events.append(self._start({"type": "thinking", "thinking": ""}))
            events.append(self._delta({"type": "thinking_delta", "thinking": thinking}))

        if content:
            if self.thinking_started and not self.thinking_done:
                self.thinking_done = True
                events.append(self._stop())
                self.content_index += 1
            if not self.text_started:
                self.text_started = True
                events.append(self._start({"type": "text", "text": ""}))
            events.append(self._delta({"type": "text_delta", "text": content}))

        for call in message.get("tool_calls") or ():
            call_id = call.get("id") or ""
            if call_id in self.tool_calls_sent:
                continue      # 同一 id 会重复出现（流式重发），只能发一次

            # thinking → tool_use 中间没有正文时，思考块也要在这里关掉
            if self.thinking_started and not self.thinking_done:
                self.thinking_done = True
                events.append(self._stop())
                self.content_index += 1
            if self.text_started:
                events.append(self._stop())
                self.content_index += 1
                self.text_started = False

            function = call.get("function") or {}
            try:
                # `partial_json` 是**线上格式**，客户端会拿它拼参数：必须紧凑且不带空格
                arguments = json.dumps(function.get("arguments") or {}, ensure_ascii=False,
                                       separators=(",", ":"))
            except (TypeError, ValueError):
                continue      # 畸形参数跳过这一条，不能让整条流断掉

            events.append(self._start({
                "type": "tool_use",
                "id": call_id,
                "name": function.get("name") or "",
                "input": {},
            }))
            events.append(self._delta({"type": "input_json_delta", "partial_json": arguments}))
            events.append(self._stop())
            self.tool_calls_sent.add(call_id)
            self.content_index += 1

        if response.get("done"):
            if self.text_started:
                events.append(self._stop())
            elif self.thinking_started and not self.thinking_done:
                events.append(self._stop())

            usage = UsageFromMetrics(_metrics(response))
            self.input_tokens = usage["input_tokens"]
            self.cache_read_tokens = usage.get("cache_read_input_tokens")
            self.output_tokens = usage["output_tokens"]

            events.append(("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": mapStopReason(
                    response.get("done_reason") or "", bool(self.tool_calls_sent))},
                "usage": {
                    "input_tokens": self.input_tokens,
                    **({"cache_read_input_tokens": self.cache_read_tokens}
                       if self.cache_read_tokens is not None else {}),
                    "output_tokens": self.output_tokens,
                },
            }))
            events.append(("message_stop", {"type": "message_stop"}))

        return events


# --------------------------------------------------------------------------- #
# token 粗估（/v1/messages/count_tokens）
# --------------------------------------------------------------------------- #


def _count_content(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                total += len(text)
            thinking = item.get("thinking")
            if isinstance(thinking, str):
                total += len(thinking)
            if item.get("type") in ("tool_use", "tool_result"):
                # 这里只是**字符计数**，但分隔符要和线上格式一致，否则估出来的 token 偏大
                total += len(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
        return total
    return len(json.dumps(content, ensure_ascii=False, separators=(",", ":")))


def EstimateInputTokens(body: dict[str, Any]) -> int:
    """对应 ``EstimateInputTokens`` —— ``len/4`` 的粗估（上游自己的 TODO 也承认不准）。

    用途只有一个：流式首片还没有 metrics 时，``message_start`` 的 ``input_tokens``
    需要有个非零值占位，否则 Claude SDK 会显示「0 tokens」。
    """
    total = 0
    system = body.get("system")
    if isinstance(system, str):
        total += len(system)
    elif isinstance(system, list):
        total += _count_content(system)

    messages = body.get("messages") or ()
    for message in messages:
        if not isinstance(message, dict):
            continue
        total += len(message.get("role") or "")
        total += _count_content(message.get("content"))

    for tool in body.get("tools") or ():
        if not isinstance(tool, dict):
            continue
        total += len(tool.get("name") or "") + len(tool.get("description") or "")
        schema = tool.get("input_schema")
        if isinstance(schema, str):
            total += len(schema)
        elif schema is not None:
            total += len(json.dumps(schema, ensure_ascii=False, separators=(",", ":")))

    tokens = total // 4
    if tokens == 0 and (messages or body.get("system") is not None):
        tokens = 1
    return tokens
