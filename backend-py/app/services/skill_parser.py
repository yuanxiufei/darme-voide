"""SKILL.md 结构化解析器 —— 移植 ``agents/skill-parser.ts``（整文件）。

对齐 PenguinHarness 第 11 章（自进化闭环）的 SKILL.md 规范：

* frontmatter 承载元数据（name / description / preconditions / protocol / workflows / agents / priority）
* 正文承载「如何做」的规范
* **前置契约**（preconditions）：执行前必须满足的条件，缺失则停下询问
* **协议字段**（protocol）：完成后必须在 YAML 协议块中额外汇报的字段，实现 skill 与 agent 解耦

单一事实来源：``skills.py``（注入 Agent prompt）与 ``routes/skills.py``（元数据 API）共用本解析器，
避免两处用脆弱正则重复解析 frontmatter。

⚠️ **YAML 方言差异（已知、低风险）**：Node 用 ``yaml`` 包（YAML 1.2），Python 用 PyYAML（YAML 1.1）。
两者对 ``yes/no/on/off``（PyYAML 当布尔）与裸日期（PyYAML 当 date 对象）处理不同。
本解析器对这些字段都做 ``isinstance(x, str)`` 判定，所以差异表现为「回退到默认值」而非崩溃；
实测 36 个 SKILL.md 未出现这类写法。
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from ..core.response import js_number

#: 开头的 YAML frontmatter（--- 包裹），非贪婪，避免误匹配正文里的 --- 分隔线
FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.S)

#: 「外来宿主平台」的工具名前缀。外部技能库按另一套宿主平台编写，工具统一带 ``hub_`` 前缀；
#: 本项目宿主工具（``read_script_for_extraction`` / ``save_dedup_characters`` …）无一以此开头。
#: 刻意**只按前缀识别**，不做「正文里未知 snake_case token 即工具」的宽泛猜测 ——
#: 正文里大量反引号 token 其实是字段名与文件名（``shot_type`` / ``bgm_url`` / ``references/``），
#: 宽泛匹配会把它们统统判成缺失工具，让预警彻底失去意义。
FOREIGN_TOOL_PREFIXES = ("hub_",)

#: 正文中反引号包裹的 token（要求至少 3 字符，避免 `` `a` `` 这类噪声）
_BACKTICK_TOKEN_RE = re.compile(r"`([a-z][a-z0-9_]{2,})`")


def _to_str_array(value: Any) -> list[str]:
    """把 frontmatter 的任意值规范化为字符串数组（兼容字符串 / 数组 / 省略三种写法）。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [s for s in (str(x).strip() for x in value) if s]
    if isinstance(value, str):
        # 字符串写法：按换行切，剥掉列表符 `-`
        return [s for s in (re.sub(r"^-\s*", "", line.strip()) for line in value.split("\n")) if s]
    return [str(value)]


def _to_tool_list(value: Any) -> list[str]:
    """把 frontmatter ``allowed-tools`` 规范化为工具名列表。

    必须同时兼容三种写法（外部库三种都出现过）::

        allowed-tools: [hub_read, hub_write]   → YAML 解析为数组
        allowed-tools: hub_read, hub_write     → 解析为**纯量字符串**（YAML 不按逗号切分）
        allowed-tools:\\n  - hub_read           → 换行列表

    故在数组/字符串归一后，统一按「换行 + 逗号」切分，再剥掉列表符与方括号/引号，最后去重。
    """
    raw = value if isinstance(value, list) else [value]
    parts: list[str] = []
    for item in raw:
        if item is None:
            continue
        parts.extend(re.split(r"[\n,]", str(item)))

    names: list[str] = []
    for part in parts:
        s = re.sub(r"^-\s*", "", part.strip())
        s = re.sub(r"^\[|\]$", "", s)
        s = re.sub(r"^[\"']|[\"']$", "", s).strip()
        if s:
            names.append(s)
    return list(dict.fromkeys(names))  # 去重且保序（对齐 JS Set）


def extract_foreign_tool_refs(body: str) -> list[str]:
    """从**正文**提取外来宿主工具引用（反引号包裹，如「先调用 ``hub_list_capabilities``」）。

    与 frontmatter ``allowed-tools`` 互补，缺一不可：实测 13 个外部 skill 正文出现 ``hub_*``，
    其中 6 个**根本没声明** ``allowed-tools`` ⇒ 只查声明会整片漏掉。
    """
    found = {token for token in _BACKTICK_TOKEN_RE.findall(body) if token.startswith(FOREIGN_TOOL_PREFIXES)}
    return sorted(found)


def _normalize_priority(value: Any) -> int | float:
    """``Number(v)`` 有限且 > 0 才采用，否则默认 100（越小越靠前）。"""
    n = js_number(value)
    return n if n is not None and n > 0 else 100


def parse_skill(content: str, fallback_id: str) -> dict[str, Any]:
    """解析 SKILL.md 原始文本为 ``{metadata, body, foreignToolRefs}``。

    frontmatter 解析失败时**降级为空元数据**（name 回退 ``fallback_id``），正文照常使用。
    """
    match = FRONTMATTER_RE.match(content)
    raw_front = match.group(1) if match else ""
    body = content[match.end() :] if match else content

    fm: dict[str, Any] = {}
    if raw_front.strip():
        try:
            parsed = yaml.safe_load(raw_front)
            # `!Array.isArray` ⇒ 只接受映射
            if isinstance(parsed, dict):
                fm = parsed
        except Exception:  # noqa: BLE001 - 与原 TS 一致：损坏的 frontmatter 不阻断解析
            fm = {}

    name = fm.get("name")
    description = fm.get("description")
    metadata = {
        "name": name.strip() if isinstance(name, str) and name.strip() else fallback_id,
        "description": description.strip() if isinstance(description, str) else "",
        "preconditions": _to_str_array(fm.get("preconditions")),
        "protocol": _to_str_array(fm.get("protocol")),
        "workflows": _to_str_array(fm.get("workflows")),
        "agents": _to_str_array(fm.get("agents")),
        "priority": _normalize_priority(fm.get("priority")),
        "allowedTools": _to_tool_list(fm.get("allowed-tools")),
    }
    return {
        "metadata": metadata,
        "body": body.strip(),
        "foreignToolRefs": extract_foreign_tool_refs(body),
    }


def render_skill(parsed: dict[str, Any]) -> str:
    """渲染为注入 Agent prompt 的文本：正文 + 前置契约段 + 输出协议字段段。

    协议字段段与 ``protocol.ts`` 的 buildProtocolContract（全局 status/summary 契约）衔接 ——
    skill 只声明「额外」字段，全局契约保证 status/summary 始终存在，二者解耦。
    """
    metadata = parsed["metadata"]
    parts = [parsed["body"]]

    if metadata["preconditions"]:
        parts.append(
            "## 前置契约（执行前必须满足）\n"
            "以下条件不满足时，停下并向用户说明缺失项，不要臆造数据继续执行：\n"
            + "\n".join(f"- {p}" for p in metadata["preconditions"])
        )

    if metadata["protocol"]:
        parts.append(
            "## 输出协议字段\n"
            "在收尾的 YAML 协议块中，除 status / summary 外，还必须额外汇报以下字段：\n"
            + "\n".join(f"- {p}" for p in metadata["protocol"])
        )

    return "\n\n".join(parts)
