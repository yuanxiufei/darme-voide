"""**ollama 模板层** —— 上游 ``template/template.go`` 的 Python 对等实现。

三件事：

1. **内置模板库**：20 份 ``*.gotmpl``（Go 模板）+ 同名 ``*.json``（stop 参数）+
   ``index.json``（每份对应的 **Jinja 原文**，只用于识别）。:func:`Named` 按 Levenshtein
   距离「认领」一份 GGUF 里带的 Jinja 模板属于哪一族。
2. **渲染**：:func:`parse` 编译，:meth:`Template.execute` 走 ollama 的两条分支 ——
   * **messages 分支**（模板里出现 ``.Messages``）：一次喂 ``System``/``Messages``/``Tools``；
   * **legacy 分支**（老模板 ``{{ .Prompt }}``/``{{ .Response }}``）：逐条累积
     ``prompt``/``response``，最后把 ``{{ .Response }}`` 之后整段截掉再渲染一次，
     得到「等 assistant 开口」的骨架 —— ollama 边生成边喂回模型就靠这个。
3. **数据整形**：``collate``（合并同角色相邻消息、收集 system）与
   ``convert_messages_for_template``/``convert_tools_for_template``。

⚠️ 与 Go 只有一处差异，且是有意的：上游 ``deleteNode`` 在某些嵌套位置会类型断言 panic；
这里改成「找到 Response 节点，只截断它之后的兄弟」，正常输入逐字相同、异常输入不崩。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from . import gotemplate
from .gotemplate import ActionNode, BranchNode, FieldNode, GoTemplateError, TextNode

__all__ = [
    "NAMED_THRESHOLD",
    "DefaultTemplate",
    "Named",
    "Template",
    "Values",
    "collate",
    "convert_messages_for_template",
    "convert_tools_for_template",
    "named",
    "named_templates",
    "parse",
]

_TEMPLATE_DIR = Path(__file__).with_name("templates")

#: ``Named()`` 的收录阈值 —— 上游写死 ``score < 100``，照抄。
NAMED_THRESHOLD = 100


# --------------------------------------------------------------------------- #
# 内置模板库
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class named:
    """对应上游 ``template.named``：名字 + Jinja 原文 + Go 模板字节 + stop 参数。"""

    name: str
    template: str = ""
    bytes: bytes = b""
    stop: list[str] = dc_field(default_factory=list)

    def reader(self) -> str:
        """对应 ``named.Reader()`` —— 已做 ``\\r\\n`` → ``\\n`` 归一化。"""
        return self.bytes.decode("utf-8")


@lru_cache(maxsize=1)
def named_templates() -> tuple[named, ...]:
    """加载 ``index.json`` + 每份 ``.gotmpl``/``.json``（对应上游 ``templatesOnce``）。"""
    index = json.loads((_TEMPLATE_DIR / "index.json").read_text(encoding="utf-8"))
    out: list[named] = []
    for entry in index:
        name = entry["name"]
        gotmpl = _TEMPLATE_DIR / f"{name}.gotmpl"
        if not gotmpl.exists():
            continue
        item = named(
            name=name,
            template=entry.get("template", ""),
            bytes=gotmpl.read_bytes().replace(b"\r\n", b"\n"),
        )
        params = _TEMPLATE_DIR / f"{name}.json"
        if params.exists():
            item.stop = list((json.loads(params.read_text(encoding="utf-8")) or {}).get("stop") or [])
        out.append(item)
    if not out:
        raise GoTemplateError(f"内置模板库是空的：{_TEMPLATE_DIR}")
    return tuple(out)


def _distance(a: str, b: str, limit: int) -> int:
    """Levenshtein 距离，超过 ``limit`` 提前返回 ``limit``（只用于「小于 100」的判定）。

    两处加速都**不改变结果**：

    * 长度差 ≥ limit ⇒ 距离必 ≥ limit，直接返回；
    * 剥掉公共前缀/后缀 —— 首字符相同时最优对齐必让它们对齐（标准结论），26 KB 的
      模板剥完通常只剩几百字符，带状 DP 再按 ``|i-j| ≤ limit`` 剪一遍。
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) >= limit:
        return limit

    n, m = len(a), len(b)
    prefix = 0
    while prefix < n and prefix < m and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < n - prefix and suffix < m - prefix
           and a[n - 1 - suffix] == b[m - 1 - suffix]):
        suffix += 1
    a, b = a[prefix:n - suffix], b[prefix:m - suffix]
    n, m = len(a), len(b)
    if not n:
        return min(m, limit)
    if not m:
        return min(n, limit)
    if abs(n - m) >= limit:
        return limit

    best = limit
    prev = [min(j, limit) for j in range(m + 1)]
    for i in range(1, n + 1):
        ca = a[i - 1]
        lo = max(1, i - limit)
        hi = min(m, i + limit)
        cur = [limit] * (m + 1)
        cur[0] = min(i, limit)
        row_min = limit
        for j in range(lo, hi + 1):
            value = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != b[j - 1]))
            cur[j] = value
            if value < row_min:
                row_min = value
        if row_min >= best:
            return best          # 行最小值单调不减 ⇒ 再往下也不可能更小
        prev = cur
    return min(prev[m], limit)


@lru_cache(maxsize=64)
def Named(source: str) -> named:
    """按 Jinja 原文识别模板族（对应 ``template.Named``）。

    ⚠️ 上游在 ``score >= 100`` 时返回 error；这里同样抛 :class:`GoTemplateError`
    —— 静默回落成某个模板会让整段提示词格式错掉，且几乎查不出来。
    """
    best: named | None = None
    score = NAMED_THRESHOLD
    for candidate in named_templates():
        if abs(len(source) - len(candidate.template)) >= score:
            continue            # 长度差已 ≥ 当前最优 ⇒ 不可能更近
        distance = _distance(source, candidate.template, score)
        if distance < score:
            score = distance
            best = candidate
    if best is None:
        raise GoTemplateError("no matching template found")
    return best


# --------------------------------------------------------------------------- #
# 渲染入参
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Values:
    """对应上游 ``template.Values``。``messages`` 元素可以是 dict 也可以是对象。"""

    messages: Sequence[Any] = ()
    tools: Sequence[Any] | None = None
    prompt: str = ""
    suffix: str = ""
    think: bool = False
    think_level: str = ""
    is_think_set: bool = False
    force_legacy: bool = False


def _snake(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """从 dict 或对象取字段（消息在本仓既有 dict 形态也有 dataclass 形态）。"""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        return obj.get(_snake(key), default)
    try:
        value = getattr(obj, key)
    except AttributeError:
        try:
            value = getattr(obj, _snake(key))
        except AttributeError:
            return default
    return value


def collate(msgs: Sequence[Any]) -> tuple[str, list[dict[str, Any]]]:
    """合并同角色相邻消息并收集 system（逐条对等 ``template.collate``）。

    ⚠️ ``tool`` 角色的相邻消息**不合并** —— 各自带 call id，合并会丢掉配对。
    """
    system: list[str] = []
    collated: list[dict[str, Any]] = []
    for msg in msgs:
        role = _get(msg, "Role") or ""
        content = _get(msg, "Content") or ""
        if role == "system":
            system.append(content)
        if collated and collated[-1]["Role"] == role and role != "tool":
            collated[-1]["Content"] += "\n\n" + content
        else:
            collated.append({
                "Role": role,
                "Content": content,
                "Thinking": _get(msg, "Thinking") or "",
                "Images": list(_get(msg, "Images") or ()),
                "ToolCalls": list(_get(msg, "ToolCalls") or ()),
                "ToolName": _get(msg, "ToolName") or "",
                "ToolCallID": _get(msg, "ToolCallID") or "",
            })
    return "\n\n".join(system), collated


def _arguments_map(args: Any) -> dict[str, Any]:
    """``ToolCall.Function.Arguments`` 可能是已解析的 dict，也可能是原始 JSON 串。"""
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    to_map = getattr(args, "to_map", None) or getattr(args, "ToMap", None)
    if callable(to_map):
        return dict(to_map())
    return {}


def convert_messages_for_template(messages: Sequence[Any] | None) -> list[dict[str, Any]] | None:
    """对应 ``convertMessagesForTemplate``：``ToolCalls`` 转成模板能 ``range`` 的形态。"""
    if messages is None:
        return None
    out: list[dict[str, Any]] = []
    for msg in messages:
        tool_calls: list[dict[str, Any]] = []
        for call in _get(msg, "ToolCalls") or ():
            fn = _get(call, "Function") or {}
            tool_calls.append({
                "ID": _get(call, "ID") or "",
                "Function": {
                    "Index": _get(fn, "Index", 0) or 0,
                    "Name": _get(fn, "Name") or "",
                    "Arguments": _arguments_map(_get(fn, "Arguments")),
                },
            })
        out.append({
            "Role": _get(msg, "Role") or "",
            "Content": _get(msg, "Content") or "",
            "Thinking": _get(msg, "Thinking") or "",
            "Images": list(_get(msg, "Images") or ()),
            "ToolCalls": tool_calls,
            "ToolName": _get(msg, "ToolName") or "",
            "ToolCallID": _get(msg, "ToolCallID") or "",
        })
    return out


def _properties_of(params: Any) -> dict[str, Any]:
    """``ToolFunctionParameters.Properties`` —— ``ToMap()`` 之后的普通 dict。"""
    props = _get(params, "Properties")
    if props is None:
        return {}
    to_map = getattr(props, "to_map", None) or getattr(props, "ToMap", None)
    if callable(to_map):
        return dict(to_map())
    return dict(props)


def convert_tools_for_template(tools: Sequence[Any] | None) -> list[dict[str, Any]] | None:
    """对应 ``convertToolsForTemplate``（``Properties`` 换成普通 dict 以便 ``range``）。"""
    if tools is None:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        fn = _get(tool, "Function") or {}
        params = _get(fn, "Parameters") or {}
        parameters: dict[str, Any] = {
            "type": _get(params, "Type") or "",
            "properties": _properties_of(params),
        }
        defs = _get(params, "Defs")
        if defs:
            parameters["$defs"] = defs
        p_items = _get(params, "Items")
        if p_items:
            parameters["items"] = p_items
        required = _get(params, "Required")
        if required:
            parameters["required"] = list(required)
        entry: dict[str, Any] = {
            "type": _get(tool, "Type") or "",
            "function": {
                "name": _get(fn, "Name") or "",
                "description": _get(fn, "Description") or "",
                "parameters": parameters,
            },
        }
        items = _get(tool, "Items")
        if items:
            entry["items"] = items
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# 模板
# --------------------------------------------------------------------------- #

#: 上游 ``DefaultTemplate`` —— 没有可用模板时的兜底（`Modelfile` 里既无 TEMPLATE 也猜不出家族）。
DefaultTemplate = "{{ .Prompt }}"


def _is_response_field(node: Any) -> bool:
    return isinstance(node, FieldNode) and "Response" in node.ident


def _cut_pipe(pipe: "gotemplate.PipeNode") -> tuple["gotemplate.PipeNode", bool]:
    commands: list[Any] = []
    for cmd in pipe.cmds:
        args: list[Any] = []
        for arg in cmd.args:
            if _is_response_field(arg):
                args.append(arg)
                commands.append(gotemplate.CommandNode(args))
                return gotemplate.PipeNode(decls=list(pipe.decls), cmds=commands), True
            args.append(arg)
        commands.append(gotemplate.CommandNode(args))
    return gotemplate.PipeNode(decls=list(pipe.decls), cmds=commands), False


def _cut_nodes(nodes: list[Any]) -> tuple[list[Any], bool]:
    """复制一棵节点树，遇到第一个 ``{{ .Response }}`` 就只保留它、丢掉后面的兄弟。"""
    out: list[Any] = []
    for node in nodes:
        if _is_response_field(node):
            out.append(node)
            return out, True
        if isinstance(node, BranchNode):
            then_list, hit = _cut_nodes(node.list)
            if hit:
                out.append(type(node)(pipe=node.pipe, list=then_list, else_list=[]))
                return out, True
            else_list, hit = _cut_nodes(node.else_list)
            out.append(type(node)(pipe=node.pipe, list=then_list, else_list=else_list))
            if hit:
                return out, True
            continue
        if isinstance(node, ActionNode):
            pipe, hit = _cut_pipe(node.pipe)
            out.append(ActionNode(pipe))
            if hit:
                return out, True
            continue
        out.append(node)
    return out, False


class Template:
    """对应上游 ``template.Template`` —— 一层薄包装，真正的引擎在 :mod:`.gotemplate`。"""

    __slots__ = ("inner", "raw")

    def __init__(self, inner: gotemplate.Template, raw: str) -> None:
        self.inner = inner
        self.raw = raw

    def __str__(self) -> str:
        return self.raw

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Template({len(self.raw)} chars, vars={self.vars()})"

    def vars(self) -> list[str]:
        return self.inner.vars()

    def contains(self, needle: str) -> bool:
        return needle in self.raw

    def subtree(self, predicate) -> "Template | None":
        """找出第一个满足谓词的节点，包成单节点模板（对应上游 ``Subtree``）。

        ollama 自己用它探测「模板里有没有 ``{{ .Tools }}``」来决定要不要插工具段。
        """
        for node in _walk_nodes(self.inner.nodes):
            if predicate(node):
                return Template(gotemplate.Template(str(node), [node]), str(node))
        return None

    # -- 渲染 --
    def execute(self, values: "Values | dict[str, Any]") -> str:
        if isinstance(values, dict):
            values = Values(**values)
        system, messages = collate(values.messages)
        var_names = self.vars()
        think = {
            "Think": values.think,
            "ThinkLevel": values.think_level,
            "IsThinkSet": values.is_think_set,
        }

        if values.prompt and values.suffix:
            # 上游：fill-in-the-middle 场景只喂 Prompt/Suffix，不进 messages 分支
            return self.inner.render({"Prompt": values.prompt, "Suffix": values.suffix,
                                      "Response": "", **think})

        if not values.force_legacy and "messages" in var_names:
            return self.inner.render({
                "System": system,
                "Messages": convert_messages_for_template(messages),
                "Tools": convert_tools_for_template(values.tools),
                "Response": "",
                **think,
            })

        return self._execute_legacy(messages, values)

    def _execute_legacy(self, messages: list[dict[str, Any]], values: "Values") -> str:
        """legacy 逐条渲染 —— 对等上游 ``Template.Execute`` 的循环 + 最后一次截断渲染。"""
        out: list[str] = []
        state = {"system": "", "prompt": "", "response": ""}

        def emit() -> None:
            out.append(self.inner.render({
                "System": state["system"],
                "Prompt": state["prompt"],
                "Response": state["response"],
                "Think": values.think,
                "ThinkLevel": values.think_level,
                "IsThinkSet": values.is_think_set,
            }))
            state["system"] = state["prompt"] = state["response"] = ""

        for msg in messages:
            role = msg["Role"]
            if role == "system":
                if state["prompt"] or state["response"]:
                    emit()
                state["system"] = msg["Content"]
            elif role == "user":
                if state["response"]:
                    emit()
                state["prompt"] = msg["Content"]
            elif role == "assistant":
                state["response"] = msg["Content"]

        # 最后一步：截掉 ``{{ .Response }}`` 之后的全部内容，等模型从这里续写
        saved = self.inner.nodes
        self.inner.nodes = _cut_nodes(saved)[0]
        try:
            out.append(self.inner.render({
                "System": state["system"],
                "Prompt": state["prompt"],
                "Response": state["response"],
                "Think": values.think,
                "ThinkLevel": values.think_level,
                "IsThinkSet": values.is_think_set,
            }))
        finally:
            self.inner.nodes = saved
        return "".join(out)


def _walk_nodes(nodes: list[Any]) -> "list[Any]":
    for node in nodes:
        yield node
        if isinstance(node, BranchNode):
            yield from _walk_nodes(node.list)
            yield from _walk_nodes(node.else_list)
        elif isinstance(node, ActionNode):
            for cmd in node.pipe.cmds:
                yield from cmd.args


def parse(source: str) -> Template:
    """编译模板；若它既没有 ``{{ .Messages }}`` 也没有 ``{{ .Response }}``，自动补一个。

    上游的逻辑是 ``Execute`` 时按 ``vars`` 判定分支，**补 ``{{ .Response }}`` 是在 Parse**
    （见 ``template.go`` 的 ``default`` 分支）—— 少了这一步，legacy 模板最后那次截断渲染
    会把续写位点吃掉，模型永远只能拿到「完整问答」而不是「等它开口」。
    """
    normalized = source.replace("\r\n", "\n")
    inner = gotemplate.parse(normalized)
    var_names = inner.vars()
    if "messages" not in var_names and "response" not in var_names:
        inner.nodes.append(ActionNode(gotemplate.PipeNode(
            cmds=[gotemplate.CommandNode([FieldNode(["Response"])])])))
        inner._vars = None
        normalized = normalized + "{{ .Response }}"
    return Template(inner=inner, raw=normalized)


def for_model(chat_template: str = "") -> Template:
    """按「GGUF 里的 Jinja 串 → 认领模板族 → 读 .gotmpl」来选模板，选不到就用默认。

    ⚠️ 这是本仓**新增**的一层胶水（上游在 ``server/`` 里做同一件事）：调用方拿到的是
    模型元数据里的 Jinja 模板串，只有认出族才能落到 Go 模板上。认不出时**不静默**：
    走 :data:`DefaultTemplate` 并在返回值里留痕（调用方可用 ``str(tpl) == DefaultTemplate`` 判断）。
    """
    if not chat_template:
        return parse(DefaultTemplate)
    try:
        return parse(Named(chat_template).reader())
    except GoTemplateError:
        return parse(DefaultTemplate)
