"""SKILL.md 结构化解析器 —— **唯一权威实现**（2026-09-15 去重：此前 `app/agent/skill_parser.py`
与之是同一套逻辑的两份实现 ✗，改一处忘另一处即静默分叉）。

⚠️ 位置按**层级**定：`agent → services` 允许、反向禁止 ⇒ 权威版落在 `services/`；
`app/agent/skill_parser.py` 现在只是**转发壳**（re-export）✓。

对外契约：**数据类** `ParsedSkill`（`metadata.*` / `body` / `foreign_tool_refs`），
字段名一律 snake_case（去重前 dict 版用 camelCase ✗ —— 已统一）。

与 ``agents/skill-parser.ts``（172 行）对齐。

对齐 PenguinHarness 架构参考文档第 11 章（自进化闭环）的 SKILL.md 规范：

* **frontmatter** 承载元数据（``name`` / ``description`` / ``preconditions`` / ``protocol`` / ``agents`` …）；
* **正文**承载「如何做」的规范；
* **前置契约**（``preconditions``）：执行该 Skill 前必须满足的条件，缺失则**停下询问**；
* **协议字段**（``protocol``）：完成后必须在 YAML 协议块中额外汇报的字段 —— skill 与 agent 解耦。

单一事实来源：``skills.py``（注入 Agent prompt）与 ``routes/skills``（元数据 API）**共用本解析器**，
避免两处各写一套脆弱正则去解析 frontmatter。

⚠️ 两处「宁可漏报也不误报」的取舍（都是刻意的，别"优化"）：

* 外来工具**只按 ``hub_`` 前缀识别**（``FOREIGN_TOOL_PREFIXES``）—— 正文里大量反引号 token
  其实是字段名/文件名（``shot_type`` / ``bgm_url`` / ``references/``…），宽泛猜「未知 snake_case
  即工具」会把它们全判成缺失，让预警彻底失去意义。代价：正文裸写的 ``read``/``write``/``task``
  识别不到（这类词在正文中噪声极大），只能靠 frontmatter ``allowed-tools`` 覆盖。
* ``to_tool_list`` 必须**同时**兼容三种写法（外部库三种都出现过）：YAML 数组、逗号纯量字符串
  （YAML **不按逗号切分**，得自己切）、换行列表。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

__all__ = [
    "FOREIGN_TOOL_PREFIXES",
    "ParsedSkill",
    "SkillMetadata",
    "extract_foreign_tool_refs",
    "parse_skill",
    "render_skill",
]

#: 匹配**开头**的 YAML frontmatter（``---`` 包裹）；非贪婪，避免误吞正文里的 ``---`` 分隔线
FRONTMATTER_RE = re.compile(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?")

#: 「外来宿主平台」的工具名前缀（本项目宿主工具无一以此开头 ⇒ 命中即外来）
FOREIGN_TOOL_PREFIXES: tuple[str, ...] = ("hub_",)

#: 正文里的反引号 token（``\`hub_xxx\```）；**只认小写开头 + ≥3 字符**，避免命中单个字符
_BACKTICK_TOKEN = re.compile(r"`([a-z][a-z0-9_]{2,})`")

#: frontmatter ``priority`` 缺省值（越小越靠前）
DEFAULT_PRIORITY = 100


@dataclass
class SkillMetadata:
    """frontmatter 元数据（缺失项都有确定的兜底值）。"""

    name: str
    description: str = ""
    #: 前置契约（空 = 无）
    preconditions: list[str] = field(default_factory=list)
    #: 输出协议字段（空 = 仅 status/summary）
    protocol: list[str] = field(default_factory=list)
    #: 适用工作流/制作阶段
    workflows: list[str] = field(default_factory=list)
    #: **默认注入哪些 Agent**（skill 自描述绑定关系；空 = 只能手动绑定）
    agents: list[str] = field(default_factory=list)
    #: 注入顺序（越小越靠前）
    priority: int = DEFAULT_PRIORITY
    #: 假设宿主提供的工具（是否可用由宿主工具集比对判定）
    allowed_tools: list[str] = field(default_factory=list)


@dataclass
class ParsedSkill:
    metadata: SkillMetadata
    #: 纯正文（不含 frontmatter），用于注入 Agent prompt
    body: str
    #: 正文中引用的**外来宿主工具**名（与 ``metadata.allowed_tools`` 互补）
    foreign_tool_refs: list[str] = field(default_factory=list)


def to_str_array(value: Any) -> list[str]:
    """frontmatter 任意值 → 字符串数组（兼容**字符串 / 数组 / 省略**三种写法）。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [
            re.sub(r"^-\s*", "", line.strip())
            for line in value.split("\n")
            if re.sub(r"^-\s*", "", line.strip())
        ]
    return [str(value)]


def to_tool_list(value: Any) -> list[str]:
    """``allowed-tools`` → 工具名列表（兼容数组 / 逗号纯量 / 换行列表；**去重保序**）。"""
    raw_items = list(value) if isinstance(value, (list, tuple)) else [value]
    parts: list[str] = []
    for item in raw_items:
        if item is None:
            continue
        parts.extend(re.split(r"[\n,]", str(item)))
    names: list[str] = []
    for part in parts:
        name = part.strip()
        name = re.sub(r"^-\s*", "", name)
        name = re.sub(r"^\[|\]$", "", name)
        name = re.sub(r'^["\']|["\']$', "", name).strip()
        if name and name not in names:  # `Array.from(new Set(...))`：去重**保序**
            names.append(name)
    return names


def normalize_priority(value: Any) -> int:
    """``priority`` → 正数（缺省 / 非法 → 100，越小越靠前）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return DEFAULT_PRIORITY
    if number != number or number in (float("inf"), float("-inf")) or number <= 0:
        return DEFAULT_PRIORITY
    return int(number)


def extract_foreign_tool_refs(body: str) -> list[str]:
    """从**正文**提取外来宿主工具引用（反引号包裹，如「先调用 ``hub_list_capabilities``」）。

    与 frontmatter ``allowed-tools`` **互补，缺一不可**：实测若干外部 skill 正文出现 ``hub_*``
    却根本没声明 ``allowed-tools`` ⇒ 只查声明会整片漏掉；也有正文调用的工具与声明**不一致**的。
    """
    found: set[str] = set()
    for match in _BACKTICK_TOKEN.finditer(body):
        token = match.group(1)
        if any(token.startswith(prefix) for prefix in FOREIGN_TOOL_PREFIXES):
            found.add(token)
    return sorted(found)


def parse_skill(content: str, fallback_id: str) -> ParsedSkill:
    """解析 SKILL.md 原文为「元数据 + 正文」。

    frontmatter 解析失败 ⇒ **降级为空元数据**（``name`` 回退 ``fallback_id``），正文照常可用。
    """
    match = FRONTMATTER_RE.match(content)
    raw_front = match.group(1) if match else ""
    body = content[match.end():] if match else content

    front: dict[str, Any] = {}
    if raw_front.strip():
        try:
            parsed = yaml.safe_load(raw_front)
            if isinstance(parsed, dict):
                front = parsed
        except Exception:  # noqa: BLE001 —— 与 TS 的 catch 等价
            front = {}

    name = front.get("name")
    metadata = SkillMetadata(
        name=(str(name).strip() if isinstance(name, str) and str(name).strip() else fallback_id),
        description=(str(front["description"]).strip()
                     if isinstance(front.get("description"), str) else ""),
        preconditions=to_str_array(front.get("preconditions")),
        protocol=to_str_array(front.get("protocol")),
        workflows=to_str_array(front.get("workflows")),
        agents=to_str_array(front.get("agents")),
        priority=normalize_priority(front.get("priority")),
        # ⚠️ YAML 键是 `allowed-tools`（连字符），不是 snake_case
        allowed_tools=to_tool_list(front.get("allowed-tools")),
    )
    return ParsedSkill(
        metadata=metadata,
        body=body.strip(),
        foreign_tool_refs=extract_foreign_tool_refs(body),
    )


def render_skill(parsed: ParsedSkill) -> str:
    """渲染成注入 Agent prompt 的文本：**正文 + 前置契约段 + 输出协议字段段**。

    协议字段段与 ``protocol.build_protocol_contract``（全局 status/summary 契约）衔接 ——
    skill 只声明「额外」字段，全局契约保证 status/summary 始终存在，二者解耦。
    """
    parts: list[str] = [parsed.body]

    if parsed.metadata.preconditions:
        parts.append(
            "## 前置契约（执行前必须满足）\n"
            "以下条件不满足时，停下并向用户说明缺失项，不要臆造数据继续执行：\n"
            + "\n".join(f"- {item}" for item in parsed.metadata.preconditions)
        )

    if parsed.metadata.protocol:
        parts.append(
            "## 输出协议字段\n"
            "在收尾的 YAML 协议块中，除 status / summary 外，还必须额外汇报以下字段：\n"
            + "\n".join(f"- {item}" for item in parsed.metadata.protocol)
        )

    return "\n\n".join(parts)
