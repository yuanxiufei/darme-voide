"""工具调用的**流式**解析器 —— 逐字移植 ``reference/ollama/tools/tools.go``。

背景：Ollama 原生 ``/api/chat`` 里的工具调用不是 OpenAI 那种结构化 JSON 字段，
而是**混在正文文本里的一段带标签文本**，形如::

    <tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>

标签本身由模型的 chat template 决定（``<tool_call>`` / ``[TOOL_CALLS]`` / 直接 ``{``），
所以官方先从 template 里把标签抠出来（``tools/template.go`` 的 ``parseTag``），
再拿这个标签做流式切分。

⚠️ 与 ``thinking.py`` 同一类问题：chunk 边界会切在标签中间、也会切在 JSON 中间，
所以必须「不确定就缓冲」。这里的 ``find_tool`` / ``find_arguments`` / ``done``
三段分别处理「工具名可能只是前缀」「JSON 还没闭合」「tag 是 ``{`` / ``[`` 时要等括号配平」。

用法::

    parser = ToolCallParser(tools=api_tools, tag="<tool_call>")
    for chunk in stream:
        calls, content = parser.add(chunk)   # calls: [{"function": {...}}]
"""

from __future__ import annotations

import json
from typing import Any


def parse_tag_from_template(chat_template: str | None) -> str:
    """从 chat template 文本里抠出工具调用标签 —— ``tools/template.go::parseTag`` 的等价实现。

    Go 侧是解析 ``text/template`` 的 AST，找「条件里含 ``.ToolCalls`` 的 if 节点」，
    再取该节点第一个**非空白文本节点**。Python 侧没有对应 AST，改用等价扫描：

    1. 定位 ``.ToolCalls`` 出现处；
    2. 向后扫描，跳过全部 ``{{ … }}`` 动作，取第一段**非空白**文本；
    3. 按第一个 ``{`` 截断（``{`` 之后是本次工具调用本身，不是标签）；
    4. 抠不到就回落到 ``"{"``（表示「尝试把任何 JSON 对象当工具调用」）。
    """
    if not chat_template:
        return "{"

    index = chat_template.find(".ToolCalls")
    if index < 0:
        return "{"

    total = len(chat_template)
    cursor = index
    while cursor < total:
        if chat_template.startswith("{{", cursor):
            end = chat_template.find("}}", cursor + 2)
            if end < 0:
                return "{"
            cursor = end + 2
            continue

        stop = cursor
        while stop < total and not chat_template.startswith("{{", stop):
            stop += 1
        text = chat_template[cursor:stop]
        if text.strip():
            tag = text.split("{", 1)[0].strip()
            return tag or "{"
        cursor = stop

    return "{"


def find_tool_entry(tools: list[dict[str, Any]], buf: bytes) -> tuple[dict[str, Any] | None, int]:
    """在缓冲区里找**最早出现**的工具名。

    返回 ``(tool, 结束位置)``；找不到返回 ``(None, 0)``。

    ⚠️ 开头那段「后缀是不是某个工具名的前缀」检查不能省：否则看到 ``get`` 就会
    立刻匹配上 ``get``，而下一个 chunk 其实是 ``_weather`` —— 官方注释里点明了这点。
    """
    if not buf:
        return None, 0

    longest = 0
    for tool in tools:
        longest = max(longest, len(str(tool.get("function", {}).get("name") or "")))

    for i in range(1, min(len(buf), longest) + 1):
        tail = buf[len(buf) - i :]
        for tool in tools:
            name = str(tool.get("function", {}).get("name") or "").encode("utf-8")
            if len(tail) < len(name) and name.startswith(tail):
                return None, 0

    found: dict[str, Any] | None = None
    start = -1
    end = -1
    found_name_len = 0

    for tool in tools:
        name = str(tool.get("function", {}).get("name") or "").encode("utf-8")
        position = buf.find(name)
        if position == -1:
            continue
        if start != -1:
            if position > start:
                continue
            if position == start and len(name) <= found_name_len:
                continue
        found = tool
        start = position
        end = position + len(name)
        found_name_len = len(name)

    if found is not None:
        return found, end
    return None, 0


def _find_object(obj: dict[str, Any], tool_name: str) -> tuple[dict[str, Any] | None, bool]:
    """在任意 JSON 结构里递归找参数对象（对应 Go 的闭包 ``findObject``）。"""

    def find_map(key: str, node: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
        value = node.get(key)
        if isinstance(value, dict):
            return value, True
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                return None, False
            if isinstance(parsed, dict):
                return parsed, True
        return None, False

    if "name" in obj:
        args, ok = find_map("arguments", obj)
        if ok:
            return args, True
        args, ok = find_map("parameters", obj)
        if ok:
            return args, True
        # 有 name 但没参数 ⇒ 认为找到了调用（无参工具）
        return None, True

    args, ok = find_map(tool_name, obj)
    if ok:
        return args, True

    for value in obj.values():
        if isinstance(value, dict):
            result, found = _find_object(value, tool_name)
            if found:
                return result, True
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    result, found = _find_object(item, tool_name)
                    if found:
                        return result, True

    return None, False


def find_arguments(tool: dict[str, Any], buffer: bytes) -> tuple[dict[str, Any] | None, int]:
    """在缓冲区里找该工具调用的参数对象，返回 ``(参数, 结束位置)``。

    ⚠️ 用一个手写扫描器而不是 ``json.loads(整段)``：流式下 JSON 往往**还没闭合**，
    整段解析必失败；而正则找花括号会被字符串里的 ``}`` 骗到，所以这里自己跟
    ``in_string`` / ``escaped`` / 括号深度。
    """
    if not buffer:
        return None, 0

    tool_name = str(tool.get("function", {}).get("name") or "")
    start = -1
    braces = 0
    in_string = False
    escaped = False

    for i, byte in enumerate(buffer):
        if escaped:
            escaped = False
            continue
        if byte == 0x5C:  # 反斜杠
            escaped = True
            continue
        if byte == 0x22:  # 双引号
            in_string = not in_string
            continue
        if in_string:
            continue

        if byte == 0x7B:  # {
            if braces == 0:
                start = i
            braces += 1
        elif byte == 0x7D:  # }
            braces -= 1
            if braces == 0 and start != -1:
                raw = buffer[start : i + 1]
                try:
                    data = json.loads(raw)
                except (TypeError, ValueError):
                    start = -1
                    continue
                if not isinstance(data, dict):
                    start = -1
                    continue

                args, found = _find_object(data, tool_name)
                if found:
                    return args, i
                return data, i

            if braces < 0:
                braces = 0

    return None, 0


class ToolCallParser:
    """有状态的工具调用解析器（一次流一个实例）。

    :param tools: 官方 tools 数组（``[{"type": "function", "function": {"name": …}}]``）
    :param tag: 工具调用标签；给 ``"{"`` / ``"["`` 表示「把 JSON 对象直接当工具调用」
    """

    def __init__(self, tools: list[dict[str, Any]] | None = None, tag: str = "{") -> None:
        self.tools = tools or []
        self.tag = tag
        self._state = "looking_for_tag"
        self._buffer = b""
        self._count = 0

    # ── 对外 ────────────────────────────────────────────────────────────
    @property
    def buffer(self) -> bytes:
        return self._buffer

    def add(self, chunk: str) -> tuple[list[dict[str, Any]], str]:
        """喂入一段文本，返回 ``(已完成的调用, 可直接给用户的内容)``。"""
        if self._state == "done":
            return [], chunk

        self._buffer += chunk.encode("utf-8")

        if self._state == "looking_for_tag":
            index, found = self._find_tag()

            if index == -1:
                content = self._buffer.decode("utf-8", errors="replace")
                self._buffer = b""
            else:
                content = self._buffer[:index].decode("utf-8", errors="replace")
                self._buffer = self._buffer[index:]

            # tag 是 { 或 [ 的模型：只有「第一个非空白字符就是它」才算工具调用，
            # 否则（比如正文里提到一个 JSON 例子）就该当普通内容吐回去。
            if self.tag in ("{", "["):
                if content.strip() != "":
                    self._state = "done"
                    return [], content + self._buffer.decode("utf-8", errors="replace")

            if not found:
                return [], content

            self._state = "tool_calling"

        calls: list[dict[str, Any]] = []
        while True:
            call = self._parse_tool_call()
            if call is None:
                break
            calls.append(call)

        content = ""
        if self._is_done():
            self._state = "done"
            content = self._buffer.decode("utf-8", errors="replace")
            self._buffer = b""

        return calls, content

    def remaining_content(self) -> str:
        """流结束时调用：返回缓冲里剩下的、应当还给用户的内容。"""
        if self._count > 0:
            return ""
        if self.tag in ("{", "["):
            return self._buffer.decode("utf-8", errors="replace")
        return ""

    # ── 内部 ────────────────────────────────────────────────────────────
    def _find_tag(self) -> tuple[int, bool]:
        raw_tag = self.tag.encode("utf-8")
        index = self._buffer.find(raw_tag)
        if index > -1:
            return index, True

        limit = min(len(self._buffer), len(raw_tag))
        for i in range(limit, 0, -1):
            if self._buffer.endswith(raw_tag[:i]):
                return len(self._buffer) - i, False
        return -1, False

    def _parse_tool_call(self) -> dict[str, Any] | None:
        tool, end = find_tool_entry(self.tools, self._buffer)
        if tool is None:
            return None

        args, args_end = find_arguments(tool, self._buffer)
        if args is None:
            # 参数还没到齐 ⇒ 等下一个 chunk。
            # ⚠️ 官方在这里**不支持**「省略 arguments 对象的无参调用」
            # （``tools.go`` 的 TODO 明写了），移植时保持同一限制：
            # 自作主张补一个 {} 会把「正文里恰好出现一个完整 JSON」误判成无参调用。
            return None
        if args_end > end:
            end = args_end

        call = {
            "function": {
                "name": str(tool.get("function", {}).get("name") or ""),
                "arguments": args,
                "index": self._count,
            }
        }
        self._count += 1
        self._buffer = self._buffer[end:]
        return call

    def _is_done(self) -> bool:
        """tag 是 ``{`` / ``[`` 时，括号配平才算这一串工具调用结束。"""
        if self.tag == "{":
            opening, closing = 0x7B, 0x7D
        elif self.tag == "[":
            opening, closing = 0x5B, 0x5D
        else:
            return False

        depth = 0
        in_string = False
        escaped = False
        for byte in self._buffer:
            if escaped:
                escaped = False
                continue
            if byte == 0x5C:
                escaped = True
                continue
            if byte == 0x22:
                in_string = not in_string
                continue
            if in_string:
                continue
            if byte == opening:
                depth += 1
            elif byte == closing:
                depth -= 1
                if depth == 0:
                    return True
        return False


__all__ = [
    "ToolCallParser",
    "find_arguments",
    "find_tool_entry",
    "parse_tag_from_template",
]
