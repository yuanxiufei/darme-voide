"""Agent 工具基座（S5）。

Node 侧每个工具用 ``createTool({ id, description, inputSchema: z.object(...), execute })``
（Mastra + zod）定义，工厂函数把 ``episodeId`` / ``dramaId`` **注入闭包**。Python 侧没有 Mastra，
这里用等价的轻量结构：

* :class:`Tool` —— ``id`` / ``description`` / **JSON Schema** / ``execute``；
* :class:`ToolRegistry` —— 按 id 取用，并能一次性导出**两种 LLM 工具契约**：
  OpenAI 兼容（``{"type":"function","function":{…}}``）与 Gemini
  （``{"name","description","parameters"}``）。

**为什么用 JSON Schema 而不是 zod 的等价物**：function-calling 契约本来就吃 JSON Schema，
写死在这一层可以同时喂给两条 text 适配器，不用为每个工具维护两套声明。

⚠️ 工厂约定与 Node **完全同形**：``create_xxx_tools(episode_id, drama_id) -> dict[str, Tool]``——
闭包只注入 id，**每个工具自己开短事务**（Node 用全局 db）。这样 agent 循环期间不会长期
占着写事务（SQLite 单写者），也不会踩「连接已被调用方关闭」。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

__all__ = [
    "Tool",
    "ToolRegistry",
    "array_of",
    "object_schema",
    "json_number",
    "json_string",
]


# ---------------------------------------------------------------------------
# 极简 JSON Schema 构造器（让工具定义读起来接近 zod）
# ---------------------------------------------------------------------------

def json_string(description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if description:
        schema["description"] = description
    return schema


def json_number(description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number"}
    if description:
        schema["description"] = description
    return schema


def array_of(item: dict[str, Any], description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": item}
    if description:
        schema["description"] = description
    return schema


def object_schema(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    """``z.object({...})`` 的等价物（``required`` 缺省 = 全部必填，与 zod 一致）。"""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    schema["required"] = list(properties) if required is None else list(required)
    return schema


# ---------------------------------------------------------------------------
# 工具与注册表
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """一个 Agent 工具（``id`` 即 LLM 侧的函数名）。"""

    id: str
    description: str
    input_schema: dict[str, Any]
    execute: Callable[[dict[str, Any]], Any | Awaitable[Any]]

    async def run(self, arguments: dict[str, Any] | None = None) -> Any:
        """执行工具（``execute`` 可同步可异步，这里统一成 await）。"""
        arguments = arguments or {}
        result = self.execute(arguments)
        if inspect.isawaitable(result):
            result = await result
        return result


@dataclass
class ToolRegistry:
    """一组工具（一个 ``create_xxx_tools`` 工厂的产物）。"""

    tools: dict[str, Tool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for tool in list(self.tools.values()):
            if not tool.id:
                raise ValueError("tool id is required")

    @classmethod
    def of(cls, *registries: "ToolRegistry") -> "ToolRegistry":
        """合并多组工具（后注册的覆盖同 id 的）。"""
        merged: dict[str, Tool] = {}
        for registry in registries:
            merged.update(registry.tools)
        return cls(merged)

    def get(self, tool_id: str) -> Tool | None:
        """按 ``tool.id`` 取工具（**先按键，再按 id 兜底**）。

        ⚠️ 驱动拿到的是 LLM 回传的**函数名 = ``tool.id``**（蛇形），而注册表的键是工厂
        返回的对象键 —— 两者**约定等同**，但一旦某个工厂用了 camelCase 键（TS 侧就是
        这样：``readScriptForExtraction`` 的 id 是 ``read_script_for_extraction``），
        只按键查就会让工具变成「Unknown tool」且**不报错**。这里按 id 兜底，杜绝这类静默失败。
        """
        tool = self.tools.get(tool_id)
        if tool is not None:
            return tool
        for candidate in self.tools.values():
            if candidate.id == tool_id:
                return candidate
        return None

    def ids(self) -> list[str]:
        return list(self.tools)

    def names(self) -> list[str]:
        """LLM 侧看到的函数名（``tool.id``，蛇形）。

        ⚠️ 与 :meth:`ids` **不是同一件事**：``ids`` 是注册表的**键**，``names`` 是
        下发给模型的 ``function.name``。评测与优化要判「模型能调哪些工具」，问的是后者。
        """
        return [tool.id for tool in self.tools.values()]

    def only(self, wanted: Iterable[str]) -> tuple["ToolRegistry", list[str]]:
        """按名册裁剪出一个子集（**评测「假设性工具集」的唯一入口**）。

        返回 ``(子集, 名册里有但本注册表没有的名字)`` —— 后者必须往上报，不能吞：
        「想摘掉一个本来就不存在的工具」与「摘掉成功」是两件事，静默当成功会让
        优化结论看着有依据、实际什么都没变。

        名字既可以写 ``tool.id``（蛇形，模型侧看到的名字）也可以写注册表键：先按键查、
        再按 id 兜底 —— 与 :meth:`get` 同一个两套匹配口径，否则评测里「按键能摘、按 id
        摘不掉」那种偏差会直接变成假结论。

        ⚠️ 顺序跟着**本注册表**走（不是跟着名册）：工具声明顺序就是提示词里 tool 列表的
        顺序，评测跑次必须与线上同一序，否则 A/B 之间多了一个不相关的变量。
        """
        # 名字 → 键：先按键（工厂返回的键），再按 ``tool.id``（蛇形）兜底 ——
        # 与 :meth:`get` 的两套匹配口径一致，否则评测里会「按键能摘、按 id 摘不掉」。
        by_id = {tool.id: key for key, tool in self.tools.items()}
        keep_keys: set[str] = set()
        missing: list[str] = []
        for item in wanted:
            name = str(item)
            key = name if name in self.tools else by_id.get(name)
            if key is None:
                missing.append(name)
                continue
            keep_keys.add(key)
        kept = {key: tool for key, tool in self.tools.items() if key in keep_keys}
        return ToolRegistry(kept), missing

    def as_openai_tools(self) -> list[dict[str, Any]]:
        """OpenAI / OpenAI 兼容（``/chat/completions``）的 ``tools`` 参数。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.id,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.tools.values()
        ]

    def as_gemini_tools(self) -> list[dict[str, Any]]:
        """Gemini ``generateContent`` 的 ``tools[].functionDeclarations``。"""
        return [{
            "functionDeclarations": [
                {
                    "name": tool.id,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
                for tool in self.tools.values()
            ]
        }]
