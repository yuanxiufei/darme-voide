"""转发壳：真正的实现在 :mod:`app.services.skill_parser`（2026-09-15 去重）。

⚠️ 为什么留壳而不是删：`agent/` 里多处按旧路径 import，留壳可避免一次性改一片；
新代码请直接 `from app.services.skill_parser import …`（层级方向 `agent → services` ✓）。
"""

from __future__ import annotations

from app.services.skill_parser import (  # noqa: F401
    DEFAULT_PRIORITY,
    FRONTMATTER_RE,
    ParsedSkill,
    SkillMetadata,
    extract_foreign_tool_refs,
    normalize_priority,
    parse_skill,
    render_skill,
    to_str_array,
    to_tool_list,
)

__all__ = [
    "DEFAULT_PRIORITY",
    "FRONTMATTER_RE",
    "ParsedSkill",
    "SkillMetadata",
    "extract_foreign_tool_refs",
    "normalize_priority",
    "parse_skill",
    "render_skill",
    "to_str_array",
    "to_tool_list",
]
