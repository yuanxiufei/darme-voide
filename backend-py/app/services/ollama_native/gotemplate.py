"""**Go ``text/template`` 子集引擎** —— ollama 的 chat 模板语言。

ollama 的模板库（``template/*.gotmpl``）与 Modelfile 的 ``TEMPLATE`` 指令都用 Go 的
``text/template``，**不是 Jinja2**：GGUF 元数据里的 ``tokenizer.chat_template`` 才是 Jinja
（那一头由 ``app/services/engine/chat_template.py`` 走 Jinja2 负责）。两者语法不通，
Jinja 引擎渲染不了 Go 模板，所以 Modelfile 解析出来的 ``TEMPLATE`` 必须有这一层才能出骨架。

支持面＝ollama 20 份内置模板 + 用户自写 Modelfile 的实际用到面：
``{{ .Field.Sub }}``、``$x``/``$x.y``、``$x := …``/``$x = …``、``$i, $e := …``、
``if/else if/else/end``、``range/with/end``（含 ``else``）、``continue``/``break``、
``{{/* 注释 */}}``、``{{-`` ``-}}`` 空白裁剪、``|`` 管道、``(pipe)`` 分组。

函数：``json currentDate yesterdayDate toTypeScriptType``（＝Go 的 ``funcs``）
＋ Go 内建 ``eq ne lt le gt ge not and or len index slice print printf println``。

刻意**不做**（遇到即报错，不静默降级）：``define``/``template``/``block``（ollama 模板
一份都没用）、channel、方法调用反射。字段解析按「dict 键 → 对象属性 → ``missingkey=zero``
返回 ``None``」处理。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from datetime import date, timedelta
from typing import Any, Iterator

__all__ = ["FUNCS", "GoTemplateError", "Template", "parse", "sprint", "truth"]


class GoTemplateError(ValueError):
    """模板**语法错**或**执行错**（带偏移/说明，便于对照 Go 的报错）。"""


# --------------------------------------------------------------------------- #
# 词法
# --------------------------------------------------------------------------- #

TEXT, SPACE, IDENT, FIELD, VARIABLE, STRING, NUMBER = (
    "text", "space", "ident", "field", "variable", "string", "number")
LPAREN, RPAREN, PIPE, COMMA, DECLARE, ASSIGN, EOF = (
    "lparen", "rparen", "pipe", "comma", "declare", "assign", "eof")
COMMENT = "comment"

_WS = " \t\r\n"
_IDENT_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_IDENT_CHAR = _IDENT_START | set("0123456789")
_DIGITS = set("0123456789")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'",
            "0": "\0", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}


@dataclass(slots=True)
class Token:
    kind: str
    value: Any
    pos: int


def _scan_string(src: str, start: int, base: int, quote: str = '"') -> tuple[str, int]:
    buf: list[str] = []
    i = start + 1
    while i < len(src):
        ch = src[i]
        if ch == "\\":
            if i + 1 >= len(src):
                raise GoTemplateError(f"转义符在串尾（偏移 {base + i}）")
            nxt = src[i + 1]
            if nxt in _ESCAPES:
                buf.append(_ESCAPES[nxt])
                i += 2
                continue
            if nxt == "x" and i + 3 < len(src):
                buf.append(chr(int(src[i + 2:i + 4], 16)))
                i += 4
                continue
            if nxt == "u" and i + 5 < len(src):
                buf.append(chr(int(src[i + 2:i + 6], 16)))
                i += 6
                continue
            raise GoTemplateError(f"认不出的转义 \\{nxt}（偏移 {base + i}）")
        if ch == quote:
            return "".join(buf), i + 1
        buf.append(ch)
        i += 1
    raise GoTemplateError(f"字符串没有闭合（起始偏移 {base + start}）")


def _parse_number(text: str, pos: int) -> Any:
    low = text.lower()
    try:
        if low.lstrip("+-").startswith("0x"):
            return int(text, 16)
        if any(c in low for c in (".", "e", "p")):
            return float(text)
        return int(text, 10)
    except ValueError as exc:
        raise GoTemplateError(f"不是合法的数字字面量 {text!r}（偏移 {pos}）") from exc


def lex_action(inner: str, base: int) -> list[Token]:
    """动作内部词法（``inner`` 是 ``{{`` 与 ``}}`` 之间的原文）。"""
    out: list[Token] = []
    i, n = 0, len(inner)
    while i < n:
        ch = inner[i]
        if ch in _WS:
            j = i
            while j < n and inner[j] in _WS:
                j += 1
            out.append(Token(SPACE, inner[i:j], base + i))
            i = j
            continue
        two = {"|": PIPE, "(": LPAREN, ")": RPAREN, ",": COMMA}.get(ch)
        if two:
            out.append(Token(two, ch, base + i))
            i += 1
            continue
        if inner.startswith(":=", i):
            out.append(Token(DECLARE, ":=", base + i))
            i += 2
            continue
        if ch == "=" and not inner.startswith("==", i):
            out.append(Token(ASSIGN, "=", base + i))
            i += 1
            continue
        if ch == '"':
            value, i = _scan_string(inner, i, base)
            out.append(Token(STRING, value, base + i))
            continue
        if ch == "`":
            end = inner.find("`", i + 1)
            if end < 0:
                raise GoTemplateError(f"原始字符串没有闭合的反引号（偏移 {base + i}）")
            out.append(Token(STRING, inner[i + 1:end], base + i))
            i = end + 1
            continue
        if ch == "'":
            value, i = _scan_string(inner, i, base, "'")
            if len(value) != 1:
                raise GoTemplateError(f"字符字面量必须是单个字符（偏移 {base + i}）")
            out.append(Token(NUMBER, ord(value), base + i))
            continue
        if ch == "$":
            j = i + 1
            while j < n and inner[j] in _IDENT_CHAR:
                j += 1
            while j < n and inner[j] == ".":
                k = j + 1
                while k < n and inner[k] in _IDENT_CHAR:
                    k += 1
                if k == j + 1:
                    raise GoTemplateError(f"变量 ``{inner[i:j]}.`` 后面缺字段名（偏移 {base + j}）")
                j = k
            out.append(Token(VARIABLE, inner[i:j], base + i))
            i = j
            continue
        if ch == ".":
            j = i + 1
            if j < n and inner[j] in _IDENT_CHAR:
                while j < n and (inner[j] in _IDENT_CHAR or inner[j] == "."):
                    j += 1
                if inner[j - 1] == ".":
                    raise GoTemplateError(f"字段 ``{inner[i:j]}`` 以点结尾（偏移 {base + i}）")
            out.append(Token(FIELD, inner[i:j], base + i))
            i = j
            continue
        if ch in _DIGITS or (ch in "+-" and i + 1 < n and inner[i + 1] in _DIGITS):
            j = i + 1
            while j < n:
                c = inner[j]
                if c in _DIGITS or c in "abcdefABCDEF":
                    j += 1
                elif c in ".xXpP":
                    j += 1
                elif c in "+-" and inner[j - 1] in "eEpP":
                    j += 1
                else:
                    break
            out.append(Token(NUMBER, _parse_number(inner[i:j], base + i), base + i))
            i = j
            continue
        if ch in _IDENT_START:
            j = i
            while j < n and inner[j] in _IDENT_CHAR:
                j += 1
            out.append(Token(IDENT, inner[i:j], base + i))
            i = j
            continue
        raise GoTemplateError(f"动作里出现认不出的字符 {ch!r}（偏移 {base + i}）")
    out.append(Token(EOF, None, base + n))
    return out


# --------------------------------------------------------------------------- #
# 语法树
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TextNode:
    text: str


@dataclass(slots=True)
class FieldNode:
    ident: list[str]


@dataclass(slots=True)
class VariableNode:
    name: str
    ident: list[str]


@dataclass(slots=True)
class LiteralNode:
    value: Any


@dataclass(slots=True)
class FunctionRef:
    name: str


@dataclass(slots=True)
class GroupNode:
    pipe: "PipeNode"


@dataclass(slots=True)
class CommandNode:
    args: list[Any]


@dataclass(slots=True)
class PipeNode:
    decls: list[tuple[str, str]] = dc_field(default_factory=list)
    cmds: list[CommandNode] = dc_field(default_factory=list)


@dataclass(slots=True)
class ActionNode:
    pipe: PipeNode


@dataclass(slots=True)
class BranchNode:
    pipe: PipeNode
    list: list[Any]
    else_list: list[Any] = dc_field(default_factory=list)


@dataclass(slots=True)
class IfNode(BranchNode):
    pass


@dataclass(slots=True)
class RangeNode(BranchNode):
    pass


@dataclass(slots=True)
class WithNode(BranchNode):
    pass


@dataclass(slots=True)
class BreakNode:
    pass


@dataclass(slots=True)
class ContinueNode:
    pass


_KEYWORD_LITERALS = {"true": True, "false": False, "nil": None}


def _describe(tok: Token) -> str:
    return {"eof": "动作结尾", "ident": f"标识符 {tok.value!r}", "field": f"字段 {tok.value!r}",
            "variable": f"变量 {tok.value!r}", "string": "字符串", "number": f"数字 {tok.value!r}",
            "lparen": "``(``", "rparen": "``)``", "pipe": "``|``", "comma": "``,``",
            "declare": "``:=``", "assign": "``=``"}.get(tok.kind, tok.kind)


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._toks = tokens
        self._i = 0

    def peek(self) -> Token:
        while self._toks[self._i].kind == SPACE:
            self._i += 1
        return self._toks[self._i]

    def next(self) -> Token:
        tok = self.peek()
        self._i += 1
        return tok

    def expect(self, kind: str, what: str) -> Token:
        tok = self.next()
        if tok.kind != kind:
            raise GoTemplateError(f"期望 {what}，实际是 {_describe(tok)}（偏移 {tok.pos}）")
        return tok

    def parse_pipe(self) -> PipeNode:
        pipe = PipeNode()
        if self.peek().kind == VARIABLE:
            save = self._i
            first = self.next()
            if self.peek().kind == COMMA:
                self.next()
                second = self.expect(VARIABLE, "第二个循环变量")
                op = self._expect_assign()
                pipe.decls.append((first.value, op))
                pipe.decls.append((second.value, op))
            elif self.peek().kind in (DECLARE, ASSIGN):
                pipe.decls.append((first.value, self._expect_assign()))
            else:
                self._i = save
        pipe.cmds.append(self.parse_command())
        while self.peek().kind == PIPE:
            self.next()
            pipe.cmds.append(self.parse_command())
        return pipe

    def _expect_assign(self) -> str:
        tok = self.next()
        if tok.kind not in (DECLARE, ASSIGN):
            raise GoTemplateError(f"期望 ``:=`` 或 ``=``，实际是 {_describe(tok)}（偏移 {tok.pos}）")
        return ":=" if tok.kind == DECLARE else "="

    def parse_command(self) -> CommandNode:
        args: list[Any] = []
        while self.peek().kind not in (EOF, PIPE, RPAREN, COMMA, DECLARE, ASSIGN):
            args.append(self.parse_operand())
        if not args:
            raise GoTemplateError("``{{ }}`` 里是空的 —— Go 也不接受空动作")
        return CommandNode(args)

    def parse_operand(self) -> Any:
        tok = self.next()
        if tok.kind == FIELD:
            return FieldNode(tok.value[1:].split(".") if len(tok.value) > 1 else [])
        if tok.kind == VARIABLE:
            name, _, rest = tok.value.partition(".")
            return VariableNode(name, rest.split(".") if rest else [])
        if tok.kind in (STRING, NUMBER):
            return LiteralNode(tok.value)
        if tok.kind == IDENT:
            if tok.value in _KEYWORD_LITERALS:
                return LiteralNode(_KEYWORD_LITERALS[tok.value])
            return FunctionRef(tok.value)
        if tok.kind == LPAREN:
            pipe = self.parse_pipe()
            self.expect(RPAREN, "``)``")
            return GroupNode(pipe)
        raise GoTemplateError(f"认不出的操作数 {_describe(tok)}（偏移 {tok.pos}）")


def _split_source(source: str) -> list[tuple[str, str, int]]:
    """切成 ``[(kind, payload, offset)]``，并处理 ``{{-`` / ``-}}`` 的空白裁剪。"""
    raw: list[tuple[str, str, int]] = []
    pos, n = 0, len(source)
    while pos < n:
        start = source.find("{{", pos)
        if start < 0:
            raw.append((TEXT, source[pos:], pos))
            break
        if start > pos:
            raw.append((TEXT, source[pos:start], pos))

        inner_start = start + 2
        trim_left = (inner_start + 1 < n and source[inner_start] == "-"
                     and source[inner_start + 1] in _WS)
        if trim_left:
            inner_start += 1
        end = source.find("}}", inner_start)
        if end < 0:
            raise GoTemplateError(f"``{{{{`` 没有配对的 ``}}}}``（偏移 {start}）")
        inner = source[inner_start:end]
        trim_right = len(inner) >= 2 and inner[-1] == "-" and inner[-2] in _WS
        if trim_right:
            inner = inner[:-1]

        if trim_left and raw and raw[-1][0] == TEXT:
            kind, payload, off = raw[-1]
            raw[-1] = (kind, payload.rstrip(_WS), off)

        stripped = inner.strip()
        if stripped.startswith("/*"):
            if not stripped.endswith("*/"):
                raise GoTemplateError(f"注释没有闭合（偏移 {start}）")
            raw.append((COMMENT, stripped, inner_start))
        else:
            raw.append(("action", inner, inner_start))
        pos = end + 2
        if trim_right:
            raw.append(("trim_next", "", pos))

    cleaned: list[tuple[str, str, int]] = []
    pending = False
    for kind, payload, off in raw:
        if kind == "trim_next":
            pending = True
            continue
        if pending and kind == TEXT:
            payload = payload.lstrip(_WS)
            pending = False
        cleaned.append((kind, payload, off))
    return cleaned


def _make_branch(frame: list[Any]) -> Any:
    """把一层块折成节点。

    frame ＝ ``[kind, pipes, then_lists, else_list, offset]``。``if`` 支持 ``else if`` 链，
    所以可能有多个 (pipe, then_list) 对 —— 折叠成「每一级 ``else`` 里嵌一个 ``if``」的嵌套结构，
    这正是 Go 的 ``else if`` 语义（且**只消耗一个 ``end``**）。
    """
    kind, pipes, then_lists, else_list, _off = frame
    cls = {"if": IfNode, "range": RangeNode, "with": WithNode}[kind]
    tail: list[Any] = list(else_list) if else_list is not None else []
    if kind != "if" or len(pipes) == 1:
        return cls(pipe=pipes[0], list=then_lists[0], else_list=tail)
    node: Any = cls(pipe=pipes[-1], list=then_lists[-1], else_list=tail)
    for index in range(len(pipes) - 2, -1, -1):
        node = cls(pipe=pipes[index], list=then_lists[index], else_list=[node])
    return node


def _frame_target(frame: list[Any]) -> list[Any]:
    """当前正在往里写节点的列表：有 ``else`` 就写 else，否则写最后一个 then。"""
    return frame[3] if frame[3] is not None else frame[2][-1]


def build_nodes(parts: list[tuple[str, str, int]]) -> list[Any]:
    root: list[Any] = []
    stack: list[list[Any]] = [["root", [], [root], None, None]]

    for kind, payload, off in parts:
        current = _frame_target(stack[-1])
        if kind == TEXT:
            if payload:
                current.append(TextNode(payload))
            continue
        if kind == COMMENT:
            continue

        parser = _Parser(lex_action(payload, off))
        tok = parser.peek()

        if tok.kind == IDENT and tok.value in ("if", "range", "with"):
            parser.next()
            stack.append([tok.value, [parser.parse_pipe()], [[]], None, off])
            continue

        if tok.kind == IDENT and tok.value == "else":
            parser.next()
            frame = stack[-1]
            if frame[0] == "root":
                raise GoTemplateError(f"``else`` 没有对应的 ``if``/``range``（偏移 {off}）")
            nxt = parser.peek()
            if nxt.kind == IDENT and nxt.value == "if":
                # ``else if`` 是**同一个块**的 else 链（只消耗一个 ``end``），
                # ⚠️ 不是嵌套块 —— 当嵌套处理会要两个 end，alpaca 那类模板直接报「缺 end」
                if frame[3] is not None:
                    raise GoTemplateError(f"``else`` 之后不能再 ``else if``（偏移 {off}）")
                if frame[0] != "if":
                    raise GoTemplateError(f"``{frame[0]}`` 不支持 ``else if``（偏移 {off}）")
                parser.next()
                frame[1].append(parser.parse_pipe())
                frame[2].append([])
                continue
            if nxt.kind != EOF:
                raise GoTemplateError(f"``else`` 后面只跟 ``if`` 或直接 ``end``（偏移 {off}）")
            if frame[3] is not None:
                raise GoTemplateError(f"同一个块里出现了两个 ``else``（偏移 {off}）")
            frame[3] = []
            continue

        if tok.kind == IDENT and tok.value == "end":
            parser.next()
            if parser.peek().kind != EOF:
                raise GoTemplateError(f"``end`` 后面不该再有内容（偏移 {off}）")
            if len(stack) == 1:
                raise GoTemplateError(f"多余的 ``end``（偏移 {off}）")
            node = _make_branch(stack.pop())
            _frame_target(stack[-1]).append(node)
            continue

        if tok.kind == IDENT and tok.value in ("continue", "break"):
            parser.next()
            current.append(ContinueNode() if tok.value == "continue" else BreakNode())
            continue

        if tok.kind == IDENT and tok.value in ("define", "template", "block"):
            raise GoTemplateError(
                f"``{{{{ {tok.value} }}}}`` 不在本引擎支持面内 —— ollama 内置模板一份都没用；"
                f" 需要复用片段请把内容直接写进模板（偏移 {off}）")

        current.append(ActionNode(parser.parse_pipe()))

    if len(stack) != 1:
        kind_name, _pipes, _thens, _e, off = stack[-1]
        raise GoTemplateError(f"``{kind_name}`` 缺 ``end``（起始偏移 {off}）")
    return root


def parse(source: str) -> "Template":
    """编译模板（语法错当场抛 :class:`GoTemplateError`）。"""
    return Template(source, build_nodes(_split_source(source.replace("\r\n", "\n"))))


# --------------------------------------------------------------------------- #
# 值语义
# --------------------------------------------------------------------------- #


def sprint(value: Any) -> str:
    """``{{ .X }}`` 的输出 —— 对齐 Go 的 ``fmt.Sprint`` + ollama 的 ``String()``。

    ``str`` 原样、``bool`` → ``true``/``false``、数字按 ``%v``、``None`` → ``<no value>``
    （Go ``text/template`` 对 nil 的既有输出）；dict/list → **JSON**（Go 侧
    ``templateArgs``/``templateTools`` 都实现了 ``String()`` 返回 ``json.Marshal``）。
    """
    if value is None:
        return "<no value>"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value == int(value) and abs(value) < 1e16 else repr(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                          default=lambda v: v.isoformat() if isinstance(v, date) else str(v))
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


def truth(value: Any) -> bool:
    """``if``/``and``/``or`` 的真值 —— 对应 Go ``text/template`` 的 ``isTrue``。"""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, (str, bytes, bytearray, list, tuple, dict, set)):
        return len(value) > 0
    return True


def _snake(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def resolve_field(base: Any, name: str) -> Any:
    """取一个字段：dict 键 → 对象属性 → ``missingkey=zero`` 给 ``None``。"""
    if base is None:
        return None
    if isinstance(base, dict):
        if name in base:
            return base[name]
        alt = _snake(name)          # Go 导出字段是大写；放宽到 snake_case 一次
        return base.get(alt) if alt in base else None
    if isinstance(base, (list, tuple)) and name.isdigit():
        idx = int(name)
        return base[idx] if 0 <= idx < len(base) else None
    try:
        return getattr(base, name)
    except AttributeError:
        alt = _snake(name)
        if alt != name:
            try:
                return getattr(base, alt)
            except AttributeError:
                pass
        return None


def _walk(base: Any, ident: list[str]) -> Any:
    value = base
    for name in ident:
        value = resolve_field(value, name)
        if value is None:
            return None
    return value


# --------------------------------------------------------------------------- #
# 内建函数（Go 的 ``builtins`` + ollama 的 ``funcs``）
# --------------------------------------------------------------------------- #


def _to_typescript_type(prop: Any) -> str:
    """``api.ToolProperty::ToTypeScriptType``（``api/types.go``）逐分支对等。"""
    if not isinstance(prop, dict):
        return "any"
    any_of = prop.get("anyOf") or prop.get("any_of")
    if any_of:
        return " | ".join(_to_typescript_type(item) for item in any_of)
    types = prop.get("type")
    if not types:
        return "any"
    if isinstance(types, str):
        types = [types]
    return " | ".join(_JSON_TO_TS.get(t, "any") for t in types)


_JSON_TO_TS = {"string": "string", "number": "number", "integer": "number",
               "boolean": "boolean", "array": "any[]", "object": "Record<string, any>",
               "null": "null"}


def _go_eq(*args: Any) -> bool:
    if len(args) < 2:
        raise GoTemplateError("``eq`` 至少要两个参数")
    first = args[0]
    return any(first == other for other in args[1:])


def _compare(args: tuple[Any, ...], op: str) -> bool:
    if len(args) != 2:
        raise GoTemplateError(f"``{op}`` 要正好两个参数（收到 {len(args)} 个）")
    left, right = args
    try:
        if op == "lt":
            return left < right
        if op == "le":
            return left <= right
        if op == "gt":
            return left > right
        return left >= right
    except TypeError as exc:
        raise GoTemplateError(f"``{op}`` 的两个参数不可比：{left!r} / {right!r}") from exc


def _printf(*argv: Any) -> str:
    """Go ``fmt.Sprintf`` 的子集（``%v %s %q %d %f %t %x %X %c %%`` + 宽度/精度）。"""
    if not argv:
        raise GoTemplateError("``printf`` 至少要一个格式串")
    fmt, args = argv[0], list(argv[1:])
    text = str(fmt)
    out: list[str] = []
    i, n, argi = 0, len(text), 0
    while i < n:
        ch = text[i]
        if ch != "%":
            out.append(ch)
            i += 1
            continue
        j = i + 1
        while j < n and (text[j] in "-+ 0123456789.#"):
            j += 1
        if j >= n:
            raise GoTemplateError(f"``printf`` 的格式串以 ``%`` 结尾：{text!r}")
        flags, verb = text[i + 1:j], text[j]
        i = j + 1
        if verb == "%":
            out.append("%")
            continue
        if not verb.isalpha():
            raise GoTemplateError(f"``printf`` 的动词必须是字母，这里遇到 {verb!r}（格式串 {text!r}）")
        if argi >= len(args):
            raise GoTemplateError(f"``printf`` 的实参不够：格式串 {text!r} 需要更多参数")
        value = args[argi]
        argi += 1
        out.append(_format_one(value, flags, verb, text))
    if argi < len(args):
        out.append("".join(sprint(a) for a in args[argi:]))  # Go: 多余实参追加 %!(EXTRA …)，这里只追加值
    return "".join(out)


def _format_one(value: Any, flags: str, verb: str, fmt: str) -> str:
    spec = "%" + flags + verb
    if verb in ("d", "i"):
        return spec % int(value)
    if verb in ("f", "F", "e", "E", "g", "G"):
        return spec % float(value)
    if verb in ("x", "X", "o", "b"):
        number = int(value) if not isinstance(value, str) else int(value, 16)
        return spec % number
    if verb == "c":
        return chr(value if isinstance(value, int) else ord(str(value)))
    if verb == "t":
        return "true" if truth(value) else "false"
    if verb == "q":
        # Go 的 `%q` 是字符串字面量，永远不带分隔符；这里显式写紧凑分隔符只为与
        # `json.Marshal` 的输出口径一致（`sprint` 恒返回 str，实际不影响结果）
        quoted = json.dumps(sprint(value), ensure_ascii=False, separators=(",", ":"))
        return spec.replace("%q", "%s") % quoted
    if verb == "s":
        return spec % sprint(value)
    if verb == "v":
        return spec.replace("%v", "%s") % sprint(value)
    raise GoTemplateError(f"``printf`` 不支持动词 ``%{verb}``（格式串 {fmt!r}）")


def _fn_slice(*args: Any) -> Any:
    if not args:
        raise GoTemplateError("``slice`` 至少要一个参数")
    target, rest = args[0], args[1:]
    if not rest:
        return target
    if len(rest) == 1:
        return target[rest[0]:]
    if len(rest) == 2:
        return target[rest[0]:rest[1]]
    raise GoTemplateError("``slice`` 最多接 ``[start, end]`` 两个下标")


def _fn_index(*args: Any) -> Any:
    if len(args) < 2:
        raise GoTemplateError("``index`` 至少要两个参数")
    value = args[0]
    for key in args[1:]:
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError):
            return None
    return value


#: ``{{ json … }}`` 等 —— 与 Go 侧 ``template.funcs`` 逐个同名同义。
FUNCS = {
    "json": lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                    default=lambda v: v.isoformat() if isinstance(v, date) else str(v)),
    # ⚠️ Go 侧**忽略**格式参数（注释里写了留给以后用），默认 YYYY-MM-DD —— 这里照抄
    "currentDate": lambda *args: date.today().isoformat(),
    "yesterdayDate": lambda *args: (date.today() - timedelta(days=1)).isoformat(),
    "toTypeScriptType": _to_typescript_type,
    "eq": _go_eq,
    "ne": lambda *args: not _go_eq(args),
    "lt": lambda *args: _compare(args, "lt"),
    "le": lambda *args: _compare(args, "le"),
    "gt": lambda *args: _compare(args, "gt"),
    "ge": lambda *args: _compare(args, "ge"),
    "not": lambda value: not truth(value),
    "and": lambda *args: _and(args),
    "or": lambda *args: _or(args),
    "len": lambda value: len(value) if value is not None else 0,
    "index": _fn_index,
    "slice": _fn_slice,
    "print": lambda *args: "".join(sprint(a) for a in args),
    "println": lambda *args: " ".join(sprint(a) for a in args) + "\n",
    "printf": _printf,
}


def _and(args: list[Any]) -> Any:
    """Go 的 ``and`` 返回**第一个假值**，全真则返回**最后一个**（不是 bool）。"""
    value: Any = True
    for item in args:
        value = item
        if not truth(item):
            return item
    return value


def _or(args: list[Any]) -> Any:
    """Go 的 ``or`` 返回**第一个真值**，全假则返回**最后一个**。"""
    value: Any = False
    for item in args:
        value = item
        if truth(item):
            return item
    return value


# --------------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------------- #

_MISSING = object()


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Frame:
    """一层作用域：``dot`` + 变量表。``=`` 赋值要能改到**外层**，所以变量表是链式查找。"""

    __slots__ = ("dot", "root", "vars", "parent")

    def __init__(self, dot: Any, root: Any, vars: dict[str, Any],
                 parent: "_Frame | None" = None) -> None:
        self.dot = dot
        self.root = root
        self.vars = vars
        self.parent = parent

    def lookup(self, name: str) -> Any:
        frame: _Frame | None = self
        while frame is not None:
            if name in frame.vars:
                return frame.vars[name]
            frame = frame.parent
        return _MISSING

    def assign(self, name: str, value: Any, *, declare: bool) -> None:
        if declare:
            self.vars[name] = value
            return
        frame: _Frame | None = self
        while frame is not None:  # ``=`` 改已有那一层（模板里的累加靠它）
            if name in frame.vars:
                frame.vars[name] = value
                return
            frame = frame.parent
        self.vars[name] = value


def _collect_idents(node: Any, names: set[str]) -> None:
    if isinstance(node, (TextNode, BreakNode, ContinueNode)):
        return
    if isinstance(node, ActionNode):
        _collect_pipe(node.pipe, names)
        return
    if isinstance(node, BranchNode):
        _collect_pipe(node.pipe, names)
        for child in node.list:
            _collect_idents(child, names)
        for child in node.else_list:
            _collect_idents(child, names)
        return


def _collect_pipe(pipe: PipeNode, names: set[str]) -> None:
    # ⚠️ 声明式管道（`$x := …`）里的字段**也要**收集：Go 的 ``Identifiers`` 不区分声明，
    #    而 ollama 正是靠 ``vars`` 里有没有 ``messages`` 来选分支（见 template.py）。
    for cmd in pipe.cmds:
        for arg in cmd.args:
            if isinstance(arg, FieldNode):
                if arg.ident:
                    names.add(arg.ident[0])
            elif isinstance(arg, GroupNode):
                _collect_pipe(arg.pipe, names)
            elif isinstance(arg, CommandNode):
                _collect_pipe(PipeNode(cmds=[arg]), names)


class Template:
    """一份编译好的 Go 模板（对应上游 ``template.Template``）。"""

    __slots__ = ("raw", "nodes", "_vars")

    def __init__(self, raw: str, nodes: list[Any]) -> None:
        self.raw = raw
        self.nodes = nodes
        self._vars: list[str] | None = None

    def __str__(self) -> str:
        return self.raw

    def contains(self, needle: str) -> bool:
        return needle in self.raw

    def vars(self) -> list[str]:
        """模板引用到的**根标识符**（小写、去重、有序），对应 Go 的 ``Template.Vars()``。"""
        if self._vars is None:
            names: set[str] = set()
            for node in self.nodes:
                _collect_idents(node, names)
            self._vars = sorted(n.lower() for n in names)
        return self._vars

    def render(self, data: dict[str, Any] | None = None) -> str:
        payload = data or {}
        out: list[str] = []
        self._exec(self.nodes, _Frame(payload, payload, {"$": payload}), out)
        return "".join(out)

    # -- 内部 --
    def _exec(self, nodes: list[Any], frame: _Frame, out: list[str]) -> None:
        for node in nodes:
            if isinstance(node, TextNode):
                out.append(node.text)
            elif isinstance(node, ActionNode):
                value = self._pipe(node.pipe, frame)
                if not node.pipe.decls:
                    out.append(sprint(value))
            elif isinstance(node, IfNode):
                if truth(self._pipe(node.pipe, frame)):
                    self._exec(node.list, frame, out)
                else:
                    self._exec(node.else_list, frame, out)
            elif isinstance(node, WithNode):
                value = self._pipe(node.pipe, frame)
                if truth(value):
                    self._exec(node.list, _Frame(value, frame.root, {}, frame), out)
                else:
                    self._exec(node.else_list, frame, out)
            elif isinstance(node, RangeNode):
                self._range(node, frame, out)
            elif isinstance(node, ContinueNode):
                raise _Continue()
            elif isinstance(node, BreakNode):
                raise _Break()
            else:  # pragma: no cover
                raise GoTemplateError(f"未知节点 {type(node).__name__}")

    def _range(self, node: RangeNode, frame: _Frame, out: list[str]) -> None:
        items = list(_range_items(self._pipe(node.pipe, frame)))
        if not items:
            self._exec(node.else_list, frame, out)
            return
        for key, value in items:
            inner_vars: dict[str, Any] = {}
            if len(node.pipe.decls) == 1:
                inner_vars[node.pipe.decls[0][0]] = value
            elif len(node.pipe.decls) >= 2:
                inner_vars[node.pipe.decls[0][0]] = key
                inner_vars[node.pipe.decls[1][0]] = value
            inner = _Frame(value, frame.root, inner_vars, frame)
            try:
                self._exec(node.list, inner, out)
            except _Continue:
                continue
            except _Break:
                break

    def _pipe(self, pipe: PipeNode, frame: _Frame) -> Any:
        value: Any = None
        piped: Any = _NO_PIPE
        for cmd in pipe.cmds:
            value = self._command(cmd, frame, piped)
            piped = value            # 前一命令的返回值是后一命令的**最后一个参数**（Go 语义）
        for name, op in pipe.decls:
            frame.assign(name, value, declare=(op == ":="))
        return value

    def _command(self, cmd: CommandNode, frame: _Frame, piped: Any) -> Any:
        head = cmd.args[0]
        if isinstance(head, FunctionRef):
            args = [self._operand(arg, frame) for arg in cmd.args[1:]]
            if piped is not _NO_PIPE:
                args.append(piped)
            func = FUNCS.get(head.name)
            if func is None:
                raise GoTemplateError(f"没有名为 ``{head.name}`` 的函数（Go 侧也不认）")
            return func(*args)
        if len(cmd.args) > 1:
            raise GoTemplateError(
                f"一条命令里出现了多个操作数（{len(cmd.args)} 个）但第一个不是函数名 ——"
                f" Go 也不接受这种写法")
        return self._operand(head, frame)

    def _operand(self, node: Any, frame: _Frame) -> Any:
        if isinstance(node, FieldNode):
            return frame.dot if not node.ident else _walk(frame.dot, node.ident)
        if isinstance(node, VariableNode):
            value = frame.lookup(node.name)
            if value is _MISSING:
                if node.name == "$":
                    value = frame.root
                else:
                    raise GoTemplateError(f"变量 ``{node.name}`` 未定义")
            return _walk(value, node.ident) if node.ident else value
        if isinstance(node, LiteralNode):
            return node.value
        if isinstance(node, GroupNode):
            return self._pipe(node.pipe, frame)
        if isinstance(node, FunctionRef):
            # 函数名单独出现在管道尾部（无参调用）
            func = FUNCS.get(node.name)
            if func is None:
                raise GoTemplateError(f"没有名为 ``{node.name}`` 的函数")
            return func()
        raise GoTemplateError(f"认不出的操作数节点 {type(node).__name__}")


_NO_PIPE = object()


def _range_items(value: Any) -> Iterator[tuple[Any, Any]]:
    if value is None:
        return
    if isinstance(value, dict):
        # ⚠️ Go 的 text/template 对 map **按 key 排序**遍历 —— 不排序顺序就与 Go 不一致
        for key in sorted(value, key=str):
            yield key, value[key]
        return
    if isinstance(value, (list, tuple, str, bytes)):
        for index, item in enumerate(value):
            yield index, item
        return
    raise GoTemplateError(f"``range`` 不支持 {type(value).__name__}（Go 只接受 array/slice/map）")
