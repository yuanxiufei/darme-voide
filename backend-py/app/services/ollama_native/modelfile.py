"""Modelfile 的解析与渲染 —— 逐状态移植 ``reference/ollama/parser/parser.go``。

Modelfile 是 ollama 的「模型配方」文本格式::

    FROM ./qwen3-14b.Q4_K_M.gguf
    PARAMETER num_ctx 8192
    PARAMETER temperature 0.7
    SYSTEM 你是一个严谨的编剧助手。
    TEMPLATE <模板原文>

（值也允许用三引号包裹成多行文本，这里为免与本 docstring 的引号冲突不再演示。）

为什么不用「按行 split 再正则」：值可以是**三引号包起来的多行文本**，也可以在
引号之间包含任意字符；而且 ollama 允许 ``FROM`` 与文件名之间出现**不换行**的空格
拼接。Go 侧用的是一个逐字符状态机（``stateNil/Name/Value/Parameter/Message/Comment``），
唯一的正确做法就是照抄它 —— 下面每个分支都对应 Go 里同名分支，改逻辑前请先读那一份。

移植时做的**唯一**简化：字符编码只认 UTF-8（去 BOM），不再处理 UTF-16
（Go 用 ``unicode.BOMOverride`` 支持两者，本项目所有输入均为 UTF-8）。
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from .params import format_params

#: 允许出现的指令名（小写）。注意 ``from`` 会被规范成内部的 ``model``。
VALID_COMMANDS: frozenset[str] = frozenset(
    {
        "from",
        "license",
        "template",
        "system",
        "adapter",
        "draft",
        "renderer",
        "parser",
        "parameter",
        "message",
        "requires",
    }
)

#: ``MESSAGE`` 允许的角色
VALID_MESSAGE_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})

INVALID_COMMAND_MSG = (
    'command must be one of "from", "license", "template", "system", "adapter", '
    '"draft", "renderer", "parser", "parameter", "message", or "requires"'
)
INVALID_MESSAGE_ROLE_MSG = 'message role must be one of "system", "user", or "assistant"'


class ParserError(Exception):
    """解析失败，带行号（与 Go 的 ``ParserError`` 同构，文案也保持一致）。"""

    def __init__(self, line_number: int, msg: str) -> None:
        self.line_number = line_number
        self.msg = msg
        super().__init__(self.__str__())

    def __str__(self) -> str:
        if self.line_number > 0:
            return f"(line {self.line_number}): {self.msg}"
        return self.msg


class UnexpectedEOFError(ParserError):
    """输入在命令中途结束（引号没闭合、``PARAMETER`` 后面缺值等）。"""


class ModelNotFoundError(Exception):
    """Modelfile 里没有 ``FROM``，或者 ``FROM`` 指向的文件不存在。"""


class _State(IntEnum):
    NIL = 0
    NAME = 1
    VALUE = 2
    PARAMETER = 3
    MESSAGE = 4
    COMMENT = 5


_SENTINEL_EOF = object()


def _is_alpha(ch: str) -> bool:
    """只有 ASCII 字母算「指令名/参数名/角色名」的合法字符（Go 侧写死 a-zA-Z）。"""
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _is_number(ch: str) -> bool:
    """Go 用的是 ``unicode.IsNumber``；Python 的 ``isdigit()`` 覆盖日常场景（含全角数字）。"""
    return ch.isdigit()


def _is_newline(ch: str) -> bool:
    return ch in ("\r", "\n")


def _is_space(ch: str) -> bool:
    """⚠️ 只认空格与 tab，**不含换行** —— 与 Go 的 ``isSpace`` 严格一致。

    这个区别决定成败：``stateValue`` 分支里「遇到空格就继续累积、遇到换行就结束」
    全靠它区分。若把换行也算 space，值永远不会结束，整个 Modelfile 会被吞成
    第一条指令的参数。
    """
    return ch in (" ", "\t")


def _next_state(ch: str, curr: _State) -> tuple[_State, str | None, BaseException | None]:
    """推进一步状态机，返回 ``(下一状态, 要写入缓冲的字符, 错误)``。

    ``emit`` 为 ``None`` 表示这个字符**不**进缓冲区（分隔符、注释符等）。
    """
    if curr == _State.NIL:
        if ch == "#":
            return _State.COMMENT, None, None
        if _is_newline(ch) or _is_space(ch):
            return _State.NIL, None, None
        return _State.NAME, ch, None

    if curr == _State.NAME:
        if _is_alpha(ch):
            return _State.NAME, ch, None
        if _is_space(ch):
            return _State.VALUE, None, None
        # 换行也落到这里：``FROM`` 后面没有值就等于「指令不完整」
        return _State.NIL, None, ParserError(0, INVALID_COMMAND_MSG)

    if curr == _State.VALUE:
        # 换行与空格都算值结束，但字符本身要留着（由拦截逻辑决定是「接着累积」还是「落指令」）
        if _is_newline(ch) or _is_space(ch):
            return _State.NIL, ch, None
        return _State.VALUE, ch, None

    if curr == _State.PARAMETER:
        if _is_alpha(ch) or _is_number(ch) or ch == "_":
            return _State.PARAMETER, ch, None
        if _is_space(ch):
            return _State.VALUE, None, None
        return _State.NIL, None, UnexpectedEOFError(0, "unexpected EOF")

    if curr == _State.MESSAGE:
        if _is_alpha(ch):
            return _State.MESSAGE, ch, None
        if _is_space(ch):
            return _State.VALUE, None, None
        return _State.NIL, None, UnexpectedEOFError(0, "unexpected EOF")

    if curr == _State.COMMENT:
        if _is_newline(ch):
            return _State.NIL, None, None
        return _State.COMMENT, None, None

    return _State.NIL, None, ParserError(0, "")


def _unquote(text: str) -> tuple[str, bool]:
    """去掉三引号/单层双引号包裹，返回 ``(值, 是否合法)``。"""
    if len(text) >= 3 and text[:3] == '"""':
        if len(text) >= 6 and text[-3:] == '"""':
            return text[3:-3], True
        return "", False

    if len(text) >= 1 and text[0] == '"':
        if len(text) >= 2 and text[-1] == '"':
            return text[1:-1], True
        return "", False

    return text, True


def quote(text: str) -> str:
    """按需加引号（渲染 Modelfile 时用）。

    ⚠️ 注意这里的判断与直觉相反：**含双引号**时反而用三引号 —— 因为单层双引号
    会被内部的引号截断。与 Go 的 ``parser.quote`` 一致。
    """
    if "\n" in text or text.startswith(" ") or text.endswith(" "):
        if '"' in text:
            return f'"""{text}"""'
        return f'"{text}"'
    return text


@dataclass
class Command:
    """一条指令。``args`` 在解析阶段恒为字符串（``MESSAGE`` 已拼成 ``"角色: 内容"``）。"""

    name: str
    args: str = ""

    def render(self) -> str:
        name = self.name
        if name == "model":
            return f"FROM {self.args}"
        if name in ("license", "template", "system", "adapter", "renderer", "parser", "requires", "draft"):
            return f"{name.upper()} {quote(self.args)}"
        if name == "message":
            role, _, message = self.args.partition(": ")
            return f"MESSAGE {role} {quote(message)}"
        return f"PARAMETER {name} {quote(self.args)}"


@dataclass
class Modelfile:
    commands: list[Command] = field(default_factory=list)

    # ── 渲染 ────────────────────────────────────────────────────────────
    def render(self) -> str:
        """渲染回文本形式（每行一个 ``\\n``，与 ``Modelfile.String()`` 一致）。"""
        return "".join(f"{command.render()}\n" for command in self.commands)

    def __str__(self) -> str:  # pragma: no cover - 仅是 render 的别名
        return self.render()

    # ── 查询 ────────────────────────────────────────────────────────────
    def get(self, name: str) -> list[Command]:
        return [command for command in self.commands if command.name == name]

    @property
    def from_path(self) -> str:
        """``FROM`` 的原始写法（未展开、未算摘要）。"""
        for command in self.commands:
            if command.name == "model":
                return command.args
        return ""

    # ── 转成 /api/create 的请求体 ────────────────────────────────────────
    def to_create_request(self, relative_dir: str = "") -> dict[str, Any]:
        """组装 ``POST /api/create`` 的请求体（对应 Go 的 ``CreateRequest``）。

        文件类指令（``FROM`` / ``ADAPTER`` / ``DRAFT``）会被展开成绝对路径并算
        ``sha256`` 摘要 —— 这与服务端「先 ``POST /api/blobs/<digest>`` 上传、
        再引用摘要」的流程对应。

        :raises ModelNotFoundError: 没有 ``FROM``，或 ``FROM`` 的文件不存在
        :raises ParserError: 参数名未登记 / 取值类型不对
        """
        request: dict[str, Any] = {
            "parameters": {},
            "messages": [],
            "files": {},
            "adapters": {},
        }
        licenses: list[str] = []
        raw_params: dict[str, list[str]] = {}

        for command in self.commands:
            name, args = command.name, command.args

            if name == "model":
                path = _expand_path(args, relative_dir)
                digest = _make_digest(path)
                request["from"] = digest
                request["files"][path] = digest
            elif name == "adapter":
                path = _expand_path(args, relative_dir)
                digest = _make_digest(path)
                request["adapters"][args] = digest
                request["files"][path] = digest
            elif name == "draft":
                path = _expand_path(args, relative_dir)
                digest = _make_digest(path)
                request["draft_files"] = {path: digest}
            elif name == "license":
                licenses.append(args)
            elif name in ("template", "system", "renderer", "parser", "requires"):
                request[name] = args
            elif name == "message":
                role, _, content = args.partition(": ")
                request["messages"].append({"role": role, "content": content})
            else:
                raw_params.setdefault(name, []).append(args)

        request["parameters"] = format_params(raw_params)
        if licenses:
            request["license"] = licenses
        return request


def parse_file(path: str) -> Modelfile:
    """读文件并解析（``FROM`` 的相对路径以该文件所在目录为基准）。"""
    with open(path, "rb") as handle:
        raw = handle.read()
    return parse(raw.decode("utf-8-sig", errors="replace"))


def parse(text: str) -> Modelfile:
    """解析 Modelfile 文本 —— ``parser.ParseFile`` 的逐字符等价实现。

    :raises ParserError: 出现非法指令名 / 非法 ``MESSAGE`` 角色 / 中途截断
    :raises ModelNotFoundError: 全文没有 ``FROM``
    """
    if text.startswith("\ufeff"):
        text = text[1:]

    commands: list[Command] = []
    cmd_name = ""
    cmd_args = ""
    role = ""
    curr = _State.NIL
    line = 1
    buffer = ""

    for ch in text:
        if _is_newline(ch):
            line += 1

        next_state, emit, error = _next_state(ch, curr)

        if isinstance(error, UnexpectedEOFError):
            raise UnexpectedEOFError(line, f"{error.msg}: {buffer}")
        if error is not None:
            raise ParserError(line, error.msg)

        if next_state != curr:
            if curr == _State.NAME:
                lowered = buffer.lower()
                if lowered not in VALID_COMMANDS:
                    raise ParserError(line, INVALID_COMMAND_MSG)
                if lowered == "from":
                    cmd_name = "model"
                elif lowered == "parameter":
                    # 转到 PARAMETER 状态，由它来定名字
                    next_state = _State.PARAMETER
                elif lowered == "message":
                    next_state = _State.MESSAGE
                    cmd_name = lowered
                else:
                    cmd_name = lowered
            elif curr == _State.PARAMETER:
                cmd_name = buffer
            elif curr == _State.MESSAGE:
                if buffer not in VALID_MESSAGE_ROLES:
                    raise ParserError(line, INVALID_MESSAGE_ROLE_MSG)
                role = buffer
            elif curr in (_State.COMMENT, _State.NIL):
                pass  # 注释与空行不产生指令
            elif curr == _State.VALUE:
                value, ok = _unquote(buffer.strip())
                if (not ok) or (emit is not None and _is_space(emit)):
                    # ⚠️ 值还没结束（引号未闭合，或遇到普通空格）⇒ 留在 VALUE 状态继续累积。
                    # 这里对应 Go 的 ``continue``：**不**清空缓冲区、**不**切换状态。
                    buffer += emit or ""
                    continue
                if role:
                    value = f"{role}: {value}"
                    role = ""
                cmd_args = value
                commands.append(Command(cmd_name, cmd_args))

            buffer = ""
            curr = next_state

        if emit is not None and emit.isprintable():
            buffer += emit

    # 收尾：把最后一条指令 flush 出来
    if curr in (_State.COMMENT, _State.NIL):
        pass
    elif curr == _State.VALUE:
        value, ok = _unquote(buffer.strip())
        if not ok:
            raise UnexpectedEOFError(line, "unexpected EOF")
        if role:
            value = f"{role}: {value}"
        commands.append(Command(cmd_name, value))
    else:
        raise UnexpectedEOFError(line, "unexpected EOF")

    if not any(command.name == "model" for command in commands):
        raise ModelNotFoundError("no Modelfile or safetensors files found")

    return Modelfile(commands)


# ── 文件摘要 ────────────────────────────────────────────────────────────
def _expand_path(path: str, relative_dir: str) -> str:
    """相对路径展开（``~`` 展开 + 以 Modelfile 所在目录为基准）。"""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    if relative_dir:
        return os.path.normpath(os.path.join(relative_dir, expanded))
    return os.path.abspath(expanded)


def _make_digest(path: str, chunk_size: int = 1 << 20) -> str:
    """算文件的 ``sha256:<hex>`` 摘要（与 ollama blobs 的 digest 同格式）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def inspect_safetensors(path: str) -> dict[str, Any]:
    """读 safetensors 头部元信息 —— 对应 Go 的 ``parseFromSafetensors``。

    safetensors 的布局是「8 字节小端长度 + 该长度的 JSON 头 + 张量数据」，
    头部里的 ``__metadata__`` 若带 ``format: pt`` 就说明这是 transformers 存出来的
    权重（而不是已转换好的 gguf）。这里只做校验与信息读取，不改动文件。
    """
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        if header_len <= 0 or header_len > 100 * 1024 * 1024:
            raise ValueError(f"safetensors 头部长度异常: {header_len}")
        header = json.loads(handle.read(header_len).decode("utf-8"))

    if not isinstance(header, dict):
        raise ValueError("safetensors 头部不是 JSON 对象")
    return header


__all__ = [
    "Command",
    "INVALID_COMMAND_MSG",
    "INVALID_MESSAGE_ROLE_MSG",
    "ModelNotFoundError",
    "Modelfile",
    "ParserError",
    "UnexpectedEOFError",
    "VALID_COMMANDS",
    "VALID_MESSAGE_ROLES",
    "inspect_safetensors",
    "parse",
    "parse_file",
    "quote",
]
