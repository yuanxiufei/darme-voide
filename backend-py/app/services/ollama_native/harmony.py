"""**gpt-oss harmony 协议解析** —— 上游 ``harmony/harmonyparser.go`` 的对等实现。

harmony 是 gpt-oss 系列的输出格式，流长这样：

```
<|start|>assistant<|channel|>analysis<|message|>先想一想…<|end|>
<|start|>assistant<|channel|>final<|message|>答案<|end|>
<|start|>assistant<|channel|>analysis to=functions.get_weather<|constrain|>json<|message|>{"city":"北京"}<|end|>
```

三层结构，逐层对等：

* :class:`HarmonyParser` —— **纯状态机**，把字节流切成
  `MessageStart → HeaderComplete → ContentEmitted* → MessageEnd` 事件。
  ⚠️ 关键细节是 ``overlap``：流可能在某次 ``add`` 结束时停在 ``<|en`` 这种半个结束标记
  上，此时**不能**把这几个字符当正文吐出去，否则前端会看到 ``<|en`` 残片。
* :class:`HarmonyMessageHandler` —— 把 channel（``analysis``/``commentary``/``final``）
  映射成 ollama 的三条语义流：正文 / 思考 / 工具调用。
* :class:`FunctionNameMap` —— 工具名在两套命名空间之间转（harmony 要求 TypeScript
  标识符风格，用户可能给 ``get weather`` 这种名字），且**必须**有反向表才能还原。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable

__all__ = [
    "HarmonyEventContentEmitted",
    "HarmonyEventHeaderComplete",
    "HarmonyEventMessageEnd",
    "HarmonyEventMessageStart",
    "HarmonyHeader",
    "HarmonyMessageHandler",
    "HarmonyParser",
    "HarmonyParseError",
    "HarmonyToolCallAccumulator",
    "FunctionNameMap",
]


class HarmonyParseError(ValueError):
    """工具调用参数不是合法 JSON（上游在 ``Add`` 里返回 error）。"""


#: 三个结构标记（上游写死在 ``NewHarmonyMessageHandler`` 里）。
MESSAGE_START_TAG = "<|start|>"
MESSAGE_END_TAG = "<|end|>"
HEADER_END_TAG = "<|message|>"

#: ``PreservedTokens()`` —— llama-server 不该把这些当普通文本吞掉的标记。
PRESERVED_TOKENS = ("<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|constrain|>")


# --------------------------------------------------------------------------- #
# 事件
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class HarmonyHeader:
    role: str = ""
    channel: str = ""
    recipient: str = ""


@dataclass(slots=True)
class HarmonyEventMessageStart:
    pass


@dataclass(slots=True)
class HarmonyEventHeaderComplete:
    header: HarmonyHeader


@dataclass(slots=True)
class HarmonyEventContentEmitted:
    content: str


@dataclass(slots=True)
class HarmonyEventMessageEnd:
    pass


# --------------------------------------------------------------------------- #
# 状态机
# --------------------------------------------------------------------------- #

LOOKING_FOR_MESSAGE_START = "LookingForMessageStart"
PARSING_HEADER = "ParsingHeader"
PARSING_CONTENT = "ParsingContent"


def _overlap(text: str, delim: str) -> int:
    """``s`` 的后缀与 ``delim`` 的前缀的最长重叠长度（逐字对等上游 ``overlap``）。"""
    for length in range(min(len(delim), len(text)), 0, -1):
        if text.endswith(delim[:length]):
            return length
    return 0


class HarmonyParser:
    """把 harmony 标记流切成事件（逐字对等上游 ``HarmonyParser``）。"""

    __slots__ = ("state", "message_start_tag", "message_end_tag", "header_end_tag",
                 "acc", "lifetime_acc", "warnings")

    def __init__(self, message_start_tag: str = MESSAGE_START_TAG,
                 message_end_tag: str = MESSAGE_END_TAG,
                 header_end_tag: str = HEADER_END_TAG) -> None:
        self.state = LOOKING_FOR_MESSAGE_START
        self.message_start_tag = message_start_tag
        self.message_end_tag = message_end_tag
        self.header_end_tag = header_end_tag
        self.acc = ""
        self.lifetime_acc = ""
        #: 上游用 ``slog.Warn`` 报「正文中间冒出 start 标记」；这里攒起来给调用方看，不刷屏
        self.warnings: list[str] = []

    def add_implicit_start(self) -> None:
        """补上模型该输出的开头 —— 有些推理后端不回显 ``<|start|>assistant``。"""
        self.acc += "<|start|>assistant"

    def add_implicit_start_or_prefill(self, last_message: Any | None) -> None:
        """按最后一条消息决定从哪续（prefill 场景）。

        * 上一条是 ``assistant`` 且**有正文** ⇒ 直接从 ``final`` 通道续写；
        * 上一条是 ``assistant`` 且**只有思考** ⇒ 从 ``analysis`` 通道续写；
        * 否则 ⇒ 普通起点。
        """
        if last_message is not None:
            role = _get(last_message, "role") or ""
            if role == "assistant":
                if _get(last_message, "content"):
                    self.acc += "<|start|>assistant<|channel|>final<|message|>"
                    return
                if _get(last_message, "thinking"):
                    self.acc += "<|start|>assistant<|channel|>analysis<|message|>"
                    return
        self.add_implicit_start()

    def add_content(self, content: str) -> list[Any]:
        """喂一段流，返回本次能确定的所有事件（可能一次跨多个状态）。"""
        self.lifetime_acc += content
        self.acc += content
        events: list[Any] = []
        while True:
            produced, keep = self._eat()
            events.extend(produced)
            if not keep:
                break
        return events

    def _eat(self) -> tuple[list[Any], bool]:
        if self.state == LOOKING_FOR_MESSAGE_START:
            if self.message_start_tag in self.acc:
                head, _, after = self.acc.partition(self.message_start_tag)
                if head:
                    self.warnings.append(
                        f"harmony parser: message start tag in the middle of content: {self.acc!r}")
                self.acc = after
                self.state = PARSING_HEADER
                return [HarmonyEventMessageStart()], True
            return [], False

        if self.state == PARSING_HEADER:
            if self.header_end_tag in self.acc:
                head, _, after = self.acc.partition(self.header_end_tag)
                self.acc = after
                self.state = PARSING_CONTENT
                return [HarmonyEventHeaderComplete(parse_header(head))], True
            return [], False

        if self.state == PARSING_CONTENT:
            if self.message_end_tag in self.acc:
                head, _, after = self.acc.partition(self.message_end_tag)
                self.acc = after
                self.state = LOOKING_FOR_MESSAGE_START
                events: list[Any] = []
                if head:
                    events.append(HarmonyEventContentEmitted(head))
                events.append(HarmonyEventMessageEnd())
                return events, True

            overlap = _overlap(self.acc, self.message_end_tag)
            if overlap > 0:
                # 尾巴可能是结束标记的前半截 ⇒ 只吐确定的部分，剩下的继续攒着
                head = self.acc[:-overlap]
                self.acc = self.acc[-overlap:]
                if not head:
                    return [], False
                return [HarmonyEventContentEmitted(head)], False

            content = self.acc
            if not content:
                return [], False
            self.acc = ""
            return [HarmonyEventContentEmitted(content)], False

        return [], False

    def parse_header(self, raw: str) -> HarmonyHeader:
        return parse_header(raw)


def parse_header(raw: str) -> HarmonyHeader:
    """解析 ``<|message|>`` 之前那段头（逐字对等上游 ``parseHeader``）。

    头形如 ``assistant<|channel|>analysis to=functions.foo<|constrain|>json``：
    role、channel、recipient 三个字段的出现顺序**不固定**，所以先摘 channel、再按
    空白切 token 取 role、最后从剩下的 token 里找 ``to=``。
    """
    header = HarmonyHeader()

    if "<|constrain|>" in raw:
        # 模型常把 ``<|constrain|>`` 直接粘在前一个词后面；补个空格才会被切成独立 token
        raw = raw.replace("<|constrain|>", " <|constrain|>", 1).strip()

    index = raw.find("<|channel|>")
    if index != -1:
        before = raw[:index]
        after = raw[index + len("<|channel|>"):]
        cut = len(after)
        for position, char in enumerate(after):
            if char.isspace():
                cut = position
                break
        header.channel = after[:cut]
        raw = (before + after[cut:]).strip()

    tokens = raw.split()
    if not tokens:
        return header

    role = tokens[0]
    tokens = tokens[1:]
    if role.startswith("to="):
        # 特例：``to=xxx`` 直接当头 ⇒ 这是一次工具调用，不是 assistant 说话
        header.recipient = role[3:]
        header.role = "tool"
    else:
        header.role = role

    if not header.recipient and tokens and tokens[0].startswith("to="):
        header.recipient = tokens[0][3:]
    return header


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


# --------------------------------------------------------------------------- #
# 工具名映射
# --------------------------------------------------------------------------- #


class FunctionNameMap:
    """工具名 ⇄ harmony 标识符的双向表（对等上游 ``FunctionNameMap``）。

    ⚠️ 反查**只能**靠这张表：``get weather`` → ``get_weather`` 之后无法从字符串还原出
    原来的空格，所以每次转换都必须登记。
    """

    #: 内置函数名不重命名（harmony 训练时就用这几个名字）。
    BUILTINS = ("browser.open", "browser.search", "browser.find", "python")

    __slots__ = ("user_to_harmony", "harmony_to_user")

    def __init__(self) -> None:
        self.user_to_harmony: dict[str, str] = {}
        self.harmony_to_user: dict[str, str] = {}

    def convert_and_add(self, user_function_name: str) -> str:
        harmony_name = self.derive_name(user_function_name)
        if user_function_name in self.BUILTINS:
            harmony_name = user_function_name
        self.user_to_harmony[user_function_name] = harmony_name
        self.harmony_to_user[harmony_name] = user_function_name
        return harmony_name

    def original_from_converted(self, harmony_function_name: str) -> str:
        return self.harmony_to_user.get(harmony_function_name, harmony_function_name)

    @staticmethod
    def convert_to_valid_chars(user_function_name: str) -> str:
        """转成合法 TypeScript 标识符（空格/``-``/``.`` → ``_``，其余非字母数字删掉）。"""
        out: list[str] = []
        for char in user_function_name:
            if char in " -.":
                out.append("_")
            elif char.isalnum() or char in "_$":
                out.append(char)
        candidate = "".join(out)
        if not candidate:
            return "unnamed"
        if candidate[0].isdigit():
            candidate = "_" + candidate
        return candidate

    def derive_name(self, user_function_name: str) -> str:
        """撞名时追加 ``_2``/``_3``（从 2 起 —— 第一个不改名）。"""
        original = self.convert_to_valid_chars(user_function_name)
        candidate = original
        count = 2
        while candidate in self.harmony_to_user:
            candidate = f"{original}_{count}"
            count += 1
        return candidate


# --------------------------------------------------------------------------- #
# 工具调用累积器
# --------------------------------------------------------------------------- #

TOOL_CALL_NORMAL = "Normal"
TOOL_CALL_CALLING = "ToolCalling"


class HarmonyToolCallAccumulator:
    """把工具调用通道的内容攒起来，``done`` 时一次性吐出（上游说工具调用一次只有一个）。"""

    __slots__ = ("state", "acc", "current_tool_name")

    def __init__(self) -> None:
        self.state = TOOL_CALL_NORMAL
        self.acc = ""
        self.current_tool_name: str | None = None

    def set_tool_name(self, tool_name: str) -> None:
        self.current_tool_name = tool_name

    def add(self, content: str) -> None:
        self.acc += content

    def drain(self) -> tuple[str | None, str]:
        text = self.acc
        self.state = TOOL_CALL_NORMAL
        self.acc = ""
        return self.current_tool_name, text

    def content(self) -> str:
        return self.acc


# --------------------------------------------------------------------------- #
# 高层处理器：harmony 事件 → ollama 的 content/thinking/tool_calls
# --------------------------------------------------------------------------- #

MESSAGE_NORMAL = "Normal"
MESSAGE_THINKING = "Thinking"
MESSAGE_TOOL_CALLING = "ToolCalling"


class HarmonyMessageHandler:
    """对等上游 ``HarmonyMessageHandler`` —— 也是 ``api.ChatResponse`` 的流式粘合层。

    用法（与 ollama 的 ``Parser`` 接口同形）：:

        handler = HarmonyMessageHandler()
        tools = handler.init(tools, last_message, think)
        for chunk in stream:
            content, thinking, calls = handler.add(chunk, done=False)
    """

    __slots__ = ("state", "parser", "function_names", "tool_accumulator", "converted_tools")

    def __init__(self) -> None:
        self.state = MESSAGE_NORMAL
        self.parser = HarmonyParser()
        self.function_names = FunctionNameMap()
        self.tool_accumulator: HarmonyToolCallAccumulator | None = None
        self.converted_tools: set[str] = set()

    # -- Parser 接口 --
    def init(self, tools: Iterable[Any] | None, last_message: Any | None,
             think: Any | None = None) -> list[Any]:
        """初始化并返回**改名后**的工具列表（调用方必须用返回的这份去请求模型）。"""
        self.parser.add_implicit_start_or_prefill(last_message)
        self.tool_accumulator = HarmonyToolCallAccumulator()

        items = list(tools or ())
        if not items:
            return items
        processed: list[Any] = []
        for tool in items:
            fn = _get(tool, "function") or _get(tool, "Function") or {}
            name = _get(fn, "name") or _get(fn, "Name") or ""
            if name:
                new_name = self.function_names.convert_and_add(name)
                self.converted_tools.add(name)
                tool = _rename_tool(tool, new_name)
            processed.append(tool)
        return processed

    def has_tool_support(self) -> bool:
        return True

    def has_thinking_support(self) -> bool:
        return True

    def preserved_tokens(self) -> tuple[str, ...]:
        return PRESERVED_TOKENS

    def create_tool_parser(self) -> HarmonyToolCallAccumulator:
        return HarmonyToolCallAccumulator()

    def add_content(self, content: str,
                    tool_parser: HarmonyToolCallAccumulator) -> tuple[str, str, str]:
        """返回 ``(正文, 思考, 待解析的工具调用文本)``。"""
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_parts: list[str] = []

        for event in self.parser.add_content(content):
            if isinstance(event, HarmonyEventHeaderComplete):
                channel = event.header.channel
                if channel == "analysis":
                    if event.header.recipient:
                        self.state = MESSAGE_TOOL_CALLING
                        tool_parser.set_tool_name(event.header.recipient)
                    else:
                        self.state = MESSAGE_THINKING
                elif channel == "commentary":
                    if event.header.recipient:
                        self.state = MESSAGE_TOOL_CALLING
                        tool_parser.set_tool_name(event.header.recipient)
                    else:
                        self.state = MESSAGE_NORMAL
                elif channel == "final":
                    self.state = MESSAGE_NORMAL
            elif isinstance(event, HarmonyEventContentEmitted):
                if self.state == MESSAGE_NORMAL:
                    content_parts.append(event.content)
                elif self.state == MESSAGE_THINKING:
                    thinking_parts.append(event.content)
                elif self.state == MESSAGE_TOOL_CALLING:
                    tool_parts.append(event.content)
            elif isinstance(event, HarmonyEventMessageEnd):
                self.state = MESSAGE_NORMAL

        return "".join(content_parts), "".join(thinking_parts), "".join(tool_parts)

    def add(self, chunk: str, done: bool = False) -> tuple[str, str, list[dict[str, Any]]]:
        """流式喂入，返回 ``(content, thinking, tool_calls)``。

        ``tool_calls`` 元素与 OpenAI 同形：``{"function": {"name": …, "arguments": {…}}}``。
        """
        accumulator = self.tool_accumulator
        if accumulator is None:
            accumulator = self.create_tool_parser()
            self.tool_accumulator = accumulator

        content, thinking, tool_content = self.add_content(chunk, accumulator)
        if tool_content:
            accumulator.add(tool_content)

        calls: list[dict[str, Any]] = []
        if done:
            name, raw = accumulator.drain()
            if name is not None:
                # ⚠️ 前缀是 ``functions.`` 十个字符，不是三个 —— 切错会得到 ``ctions.xxx``
                short = name[len("functions."):] if name.startswith("functions.") else name
                short = self.function_names.original_from_converted(short)
                try:
                    arguments = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError as exc:
                    raise HarmonyParseError(
                        f"工具调用参数不是合法 JSON：raw={raw!r}，err={exc}") from exc
                calls.append({
                    "function": {
                        "name": short,
                        "arguments": arguments if isinstance(arguments, dict) else {"value": arguments},
                    },
                })
        return content, thinking, calls


def _rename_tool(tool: Any, new_name: str) -> Any:
    """返回一份把 ``function.name`` 换成 harmony 名字的工具定义（不改原对象）。"""
    if isinstance(tool, dict):
        clone = dict(tool)
        fn = dict(clone.get("function") or clone.get("Function") or {})
        fn["name"] = new_name
        clone["function"] = fn
        clone.pop("Function", None)
        return clone
    return tool


def parse_stream(chunks: Iterable[str]) -> tuple[str, str, list[dict[str, Any]]]:
    """便利函数：把整段 harmony 流一次解析完（非流式调用方用）。"""
    handler = HarmonyMessageHandler()
    handler.init(None, None, None)
    content: list[str] = []
    thinking: list[str] = []
    calls: list[dict[str, Any]] = []
    for chunk in chunks:
        part_content, part_thinking, part_calls = handler.add(chunk, done=False)
        content.append(part_content)
        thinking.append(part_thinking)
        calls.extend(part_calls)
    part_content, part_thinking, part_calls = handler.add("", done=True)
    content.append(part_content)
    thinking.append(part_thinking)
    calls.extend(part_calls)
    return "".join(content), "".join(thinking), calls
