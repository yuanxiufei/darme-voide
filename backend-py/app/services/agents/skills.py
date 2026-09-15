"""Agent Skill 加载器 —— 与 ``agents/skills.ts``（218 行）对齐。

支持 **DB 配置（用户可控）+ SKILL.md 自描述默认绑定（兜底）**。

## 默认绑定由 SKILL.md 自己声明，代码里**不维护**任何「谁绑谁」的映射

```yaml
agents:   [storyboard_breaker]   # 默认注入哪些 Agent（不写 = 不默认注入，仅按需手动绑）
priority: 20                     # 注入顺序，越小越靠前（默认 100）
```

于是「新增 / 调整默认绑定」= 改 SKILL.md，**零代码改动**；绑定关系与 skill 同生共死
（不会出现「代码里绑了一个不存在的 skill」这种漂移）。

## 目录约定（决定 skill 属于「自有」还是「外部库」）

* ``skills/<name>/SKILL.md`` → **自有（core）**：与工作流、工具名、字段契约对齐，**可默认注入**；
* ``skills/<lib>/library.yaml`` → **外部技能库（vendor）**：库内 skill **不默认注入**。

外部库不默认注入的三条理由（体量失控 ~40KB/篇、语义错位、只读 SKILL.md 会注入半截指令）
见原 TS 文件头；仍可在前端「Agent 配置 → 绑定 Skills」手动启用（DB ``agent_configs.skills``
**优先于**默认绑定）。

⚠️ 目录定位：``<项目根>/skills``（与 cwd 解耦，兼容任意启动方式）—— **注意它不在
``backend/`` 之下，所以 S7 删 ``backend/`` 不受影响**（对比：``evaluation`` 的
``benchmarks/`` 在 ``backend/`` 里，那边需要搬）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .skill_parser import ParsedSkill, parse_skill, render_skill

__all__ = [
    "SKILL_CHAR_BUDGET_DEFAULT",
    "list_core_skill_ids",
    "list_skill_ids",
    "load_agent_skills",
    "load_skill",
    "resolve_default_skills",
    "skill_char_budget",
    "skills_dir",
]

#: Skill 注入体量上限（字符数，默认 60000 ≈ 3 万 token 量级）——给「skill 越挂越多」设硬闸
SKILL_CHAR_BUDGET_DEFAULT = 60_000


def skills_dir() -> Path:
    """``backend-py/skills``（可用 ``SKILLS_DIR`` 覆盖，便于测试隔离）。

    ⚠️ **唯一权威在 ``app/config.py`` 的 ``skills_dir()``**（2026-09-15 收口：此前本文件、
    ``services/skills.py`` 的常量、守卫脚本三处各写一遍 ✗）。这里只转发 —— 保留同名函数是因为
    运行链（``load_agent_skills`` 等）与测试都按这个名字调用，且它们需要**每次调用都重读
    ``SKILLS_DIR``**（常量版做不到，测试隔离时就会失灵）。
    """
    from ...config import skills_dir as _skills_dir  # noqa: PLC0415 —— 惰性导入，避免任何循环依赖风险

    return _skills_dir()


def skill_char_budget() -> int:
    """``AGENT_SKILL_BUDGET`` 覆盖，非正数 / 非法 → 默认 60000；**设为 0 关闭限制**。

    ⚠️ 与 TS 的**有意差异**：TS 在模块加载时算一次常量；这里每次调用读 env ——
    行为只在你运行期改 env 时才不同（换来的是可测试性）。
    """
    raw = os.environ.get("AGENT_SKILL_BUDGET")
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return SKILL_CHAR_BUDGET_DEFAULT
    if number != number or number in (float("inf"), float("-inf")) or number <= 0:
        return SKILL_CHAR_BUDGET_DEFAULT
    return int(number)


@dataclass
class SkillBinding:
    """一条 skill 绑定（DB 配置或默认绑定）。"""

    id: str
    enabled: bool = True
    #: 默认按数组顺序（见 ``default_bindings``）
    priority: int = 0


def parse_skills_config(raw: str | None) -> list[SkillBinding]:
    """解析 DB ``agent_configs.skills`` 的 JSON 字符串（**只留 id 是字符串的项**）。"""
    if not raw:
        return []
    try:
        parsed: Any = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    bindings: list[SkillBinding] = []
    for item in parsed:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        priority = item.get("priority")
        bindings.append(SkillBinding(
            id=item["id"],
            enabled=item.get("enabled") is not False,
            priority=int(priority) if isinstance(priority, (int, float)) and not isinstance(priority, bool) else 0,
        ))
    return bindings


def default_bindings(skill_ids: list[str]) -> list[SkillBinding]:
    """默认 skillId 列表 → 绑定（``priority`` 取**数组序号 + 1**，保序）。"""
    return [SkillBinding(id=skill_id, enabled=True, priority=index + 1)
            for index, skill_id in enumerate(skill_ids)]


#: SKILL.md 解析缓存（按 mtime 失效）：默认绑定查询会在**每次 Agent 生成**时扫一遍目录
_parsed_cache: dict[str, tuple[float, ParsedSkill | None]] = {}


def load_skill(skill_id: str) -> ParsedSkill | None:
    """加载并解析指定 Skill（带 mtime 缓存）。文件缺失 / 读取失败 → ``None``。"""
    file_path = skills_dir() / skill_id / "SKILL.md"
    try:
        mtime = file_path.stat().st_mtime
    except OSError:
        _parsed_cache.pop(skill_id, None)
        return None

    cached = _parsed_cache.get(skill_id)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    parsed: ParsedSkill | None
    try:
        parsed = parse_skill(file_path.read_text(encoding="utf-8"), skill_id)
    except OSError:
        parsed = None
    _parsed_cache[skill_id] = (mtime, parsed)
    return parsed


def list_core_skill_ids() -> list[str]:
    """**自有** skill（顶层目录自身含 ``SKILL.md``）；顶层无 SKILL.md 的目录视为外部库容器。"""
    directory = skills_dir()
    if not directory.exists():
        return []
    return sorted([
        entry.name for entry in directory.iterdir()
        if entry.is_dir() and (entry / "SKILL.md").exists()
    ])


def resolve_default_skills(agent_type: str) -> list[str]:
    """某 Agent 的默认绑定 = 所有 frontmatter 里声明了该 agent 的自有 skill。

    按 ``priority`` 升序（同序按 id 字典序）⇒ 注入顺序**稳定可复现**。
    """
    if not agent_type:
        return []
    entries: list[tuple[str, int]] = []
    for skill_id in list_core_skill_ids():
        parsed = load_skill(skill_id)
        if parsed is None or agent_type not in parsed.metadata.agents:
            continue
        entries.append((skill_id, parsed.metadata.priority))
    # ⚠️ TS 用 `localeCompare`：ASCII id 上与字典序一致（有意差异仅限非 ASCII id）
    entries.sort(key=lambda item: (item[1], item[0]))
    return [skill_id for skill_id, _ in entries]


def load_agent_skills(agent_type: str, db_skills_raw: str | None = None) -> str | None:
    """为指定 Agent 加载 Skill 指令文本；无可用 Skill → ``None``。

    优先级：

    1. ``db_skills_raw`` 能解析出**配置项** → 只加载 ``enabled`` 的（**全部关闭则不注入，不回退默认**）；
    2. 空 / 空数组 / 解析失败 → 回退该 Agent 的默认绑定（SKILL.md frontmatter ``agents:``）；
    3. 都没有 → ``None``。
    """
    bindings: list[SkillBinding] = []
    # DB 里存在「配置项」= 用户已做过选择，其意图优先 —— 即使最终一个都不注入，也不回退默认
    user_configured = False

    if db_skills_raw and db_skills_raw.strip():
        parsed_config = parse_skills_config(db_skills_raw)
        # ⚠️ 判据必须是「**解析出的配置项**是否非空」，不能看「过滤 enabled 后是否为空」：
        # 后者会把「用户全部取消勾选」误判成「从来没配过」→ 回退默认绑定，
        # 于是「一个都不注入」被反向执行为「注入全部默认 skill」。
        if parsed_config:
            user_configured = True
            bindings = sorted([b for b in parsed_config if b.enabled],
                              key=lambda b: b.priority or 0)

    if not user_configured:
        default_ids = resolve_default_skills(agent_type)
        if not default_ids:
            return None
        bindings = default_bindings(default_ids)

    # 用户显式关闭了全部 skill → 不注入任何 skill（这是「最小化注入」的正当诉求）
    if not bindings:
        return None

    budget = skill_char_budget()
    sections: list[str] = []
    skipped: list[str] = []
    used = 0
    for binding in bindings:
        parsed = load_skill(binding.id)
        if parsed is None:
            skipped.append(f"{binding.id}（缺失）")
            continue
        rendered = render_skill(parsed)
        if budget > 0 and used + len(rendered) > budget:
            skipped.append(f"{binding.id}（超预算）")
            continue
        used += len(rendered)
        sections.append(rendered)

    if not sections:
        return None

    notice = ([f"> 未注入：{'、'.join(skipped)}（文件缺失或超出 {budget} 字符预算）", ""]
              if skipped else [])

    return "\n".join([
        "## Available Skills",
        "",
        *notice,
        *[f"---\n{section}\n" for section in sections],
        "---",
    ])


def list_skill_ids() -> list[str]:
    """扫描 ``skills/``，返回**全部**可用 skill id（含外部库的嵌套目录，如 ``<lib>/installed/<name>``）。

    ⚠️ 默认注入请用 ``resolve_default_skills``（**只含自有 core skill**）—— 外部库是
    「按触发词独立启动」的会话式技能，绑给常驻 Agent 只会注入半截指令。

    有意差异：TS 用 ``readdirSync`` 顺序（非确定）；这里**排序**输出（确定性更好）。
    """
    directory = skills_dir()
    if not directory.exists():
        return []
    found: list[str] = []

    def scan(current: Path, prefix: str = "") -> None:
        for entry in sorted(current.iterdir()):
            if not entry.is_dir():
                continue
            skill_id = f"{prefix}/{entry.name}" if prefix else entry.name
            if (entry / "SKILL.md").exists():
                found.append(skill_id)
            scan(entry, skill_id)

    scan(directory)
    return found
