"""Skill 目录扫描与默认绑定解析 —— 移植 ``agents/skills.ts``（除 ``loadAgentSkills``）。

**默认绑定由 SKILL.md 自己声明**（frontmatter ``agents:``），代码里不维护任何「谁绑谁」的映射：
``agents: [storyboard_breaker]`` + ``priority: 20`` ⇒ 新增/调整默认绑定是**纯文件操作、零代码改动**，
且绑定关系与 skill 同生共死（不会出现「代码里绑了一个不存在的 skill」这种漂移）。

目录约定（决定 skill 属「自有」还是「外部库」）::

    backend-py/app/skills/<name>/SKILL.md      → 自有（core）：可被 frontmatter agents 默认注入
    backend-py/app/skills/<lib>/library.yaml   → 外部技能库（vendor）：**库由声明文件识别**，库内 skill 不默认注入

✅ ``loadAgentSkills``（注入文本组装 + 体量预算闸）**已迁** —— 见 ``services/agents/skills.py``
的 ``load_agent_skills``（Agent 运行链在 ``agents/runtime.py`` 里真调用它，skill 段 2026-09-15 起
**真的会注入**）。本文件只搬**路由实际用到的三个能力**：core 列表 / 默认绑定 / 全量列表
（外加预算常量的读法）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..core.config import skills_dir
from .skill_parser import parse_skill

#: 技能库目录 —— **唯一权威是 ``app/config.py`` 的 ``skills_dir()``**（2026-09-15 收口：此前
#: 本文件、``services/agents/skills.py``、守卫脚本三处各写一遍 ✗）。
#: 这里在**导入期**绑定为常量，与迁移前行为一致（要让 ``SKILLS_DIR`` 环境变量生效，需在 import 前设好）；
#: 运行链里需要「每次调用都重读 env」的地方，用 ``services/agents/skills.py`` 的 ``skills_dir()``。
SKILLS_DIR: Path = skills_dir()

#: SKILL.md 解析缓存（按 mtime 失效）：默认绑定查询会反复扫同一批文件
_parsed_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}


def skill_char_budget() -> int:
    """注入体量上限（字符，默认 60000 ≈ 3 万 token 量级），``AGENT_SKILL_BUDGET`` 可覆盖。

    目的是给「skill 越挂越多」设一道硬闸 —— 超出的 skill 按优先级跳过并在注入文本里注明。
    """
    raw = os.environ.get("AGENT_SKILL_BUDGET")
    try:
        value = float(raw) if raw else float("nan")
    except ValueError:
        value = float("nan")
    return int(value) if value == value and value > 0 else 60_000


#: 与原 TS 的 ``SKILL_CHAR_BUDGET`` 同义（模块级常量，启动时求值一次）
SKILL_CHAR_BUDGET = skill_char_budget()


def list_core_skill_ids() -> list[str]:
    """扫描 ``skills/`` 顶层，返回**自有 skill** id（= 目录自身含 SKILL.md 的那些）。

    顶层不含 SKILL.md 的目录视为外部技能库容器（vendor），其子项不进 core 列表。
    这条规则不依赖任何外部库的名字，换库/加库都无需改代码。
    """
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(
        entry.name
        for entry in SKILLS_DIR.iterdir()
        if entry.is_dir() and (entry / "SKILL.md").exists()
    )


def load_skill(skill_id: str) -> dict[str, Any] | None:
    """加载并解析指定 Skill（带 mtime 缓存）。文件缺失/读取失败 → None。"""
    file_path = SKILLS_DIR / skill_id / "SKILL.md"
    try:
        mtime = file_path.stat().st_mtime
    except OSError:
        _parsed_cache.pop(skill_id, None)
        return None

    hit = _parsed_cache.get(skill_id)
    if hit is not None and hit[0] == mtime:
        return hit[1]

    try:
        parsed: dict[str, Any] | None = parse_skill(file_path.read_text(encoding="utf-8"), skill_id)
    except OSError:
        parsed = None
    _parsed_cache[skill_id] = (mtime, parsed)
    return parsed


def resolve_default_skills(agent_type: str) -> list[str]:
    """某 Agent 的默认绑定 = 所有 frontmatter 声明了该 agent 的**自有** skill。

    按 priority 升序（同序按 id 字典序），保证注入顺序稳定可复现。
    """
    if not agent_type:
        return []
    entries = []
    for skill_id in list_core_skill_ids():
        parsed = load_skill(skill_id)
        if parsed and agent_type in parsed["metadata"]["agents"]:
            entries.append((parsed["metadata"]["priority"], skill_id))
        elif parsed is None:
            continue
    entries.sort(key=lambda e: (e[0], e[1]))
    return [skill_id for _, skill_id in entries]


def list_skill_ids() -> list[str]:
    """扫描 ``skills/``，返回全部可用 skill id（**含外部库的嵌套目录**）。

    ⚠️ 默认注入请用 ``resolve_default_skills``（只含自有 core skill）—— 外部库是
    「按触发词独立启动」的会话式技能，绑定给常驻 Agent 只会注入半截指令。
    """
    if not SKILLS_DIR.is_dir():
        return []

    out: list[str] = []

    def scan(directory: Path, prefix: str = "") -> None:
        for entry in sorted(directory.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue
            skill_id = f"{prefix}/{entry.name}" if prefix else entry.name
            if (entry / "SKILL.md").exists():
                out.append(skill_id)
            scan(entry, skill_id)

    scan(SKILLS_DIR)
    return out
