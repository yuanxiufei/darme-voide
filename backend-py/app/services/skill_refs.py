"""Skill 附件的**按需读取**通道（只读、有边界）。

为什么需要它
============

``app/agent/skills.py::load_agent_skills`` 只读每个 skill 的 ``SKILL.md``，
于是 ``references/`` 下的正文**没有任何通路会去读** —— 那批文件等于死料。
外部技能库（``data/playbook/film-expert-bridge``）的 ``combat/``／``general/``／``ultimate/``
133 份素材正是这种形态：搬进本仓若没有读取通路，就是「看起来装了很多、其实一份也用不上」。

同一件事本仓已经踩过一次：``docs/feature-map.md`` 里 ``/scenes`` 那条写着
「不在端点里触发解析，一份要读全本几百页」——**能不能读到**和**读得起不起**
是两个问题，本模块只解决前者：给模型一个明确的「列目录 / 读某一份」的手。

三条边界（缺一条就等于开了个任意文件读取口）
============================================

1. **根只能在** ``skills_dir()`` 下（``SKILLS_DIR`` 可覆盖，测试隔离用）；
2. **只允许进** ``references/``：``SKILL.md`` 已由加载器注入，放开整个 skill 目录
   等于让人把注入过的内容再读一遍，白烧上下文；
3. **后缀白名单 ＋ 单次上限**：非文本文件（图片/字体/权重）一律拒绝并说明，
   不静默读成乱码；超长按行截断并如实标 ``truncated``。

``available=False``（这个 skill 压根没有 ``references/``）与「列出来是空目录」
必须能区分 —— 与 ``routers/knowledge.py`` 的回执纪律同一条：不许把「没有这东西」
和「有但是空」混成一个空数组。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agent.skills import skills_dir

__all__ = [
    "SkillRefError",
    "list_refs",
    "read_ref",
]

#: 允许读取的后缀（都是纯文本）。图片/音频/权重不进白名单 —— 那一类要走媒体接口，不是这条路。
TEXT_SUFFIXES = (".md", ".txt", ".yaml", ".yml", ".json", ".csv")

#: 单文件字节上限：超过就说明它不是"一段可读的方法论文本"（或者根本是二进制改了后缀）。
MAX_BYTES = 512 * 1024

#: 单次读取行数上限。
MAX_LINES = 400


class SkillRefError(ValueError):
    """skill 名 / 附件路径不合法，或文件不可读。"""


def _skill_root(skill_id: str) -> Path:
    """``skill_id``（``库名/技能名`` 形式）→ 解析后的 skill 根目录。"""
    clean = (skill_id or "").strip().replace("\\", "/").strip("/")
    if not clean:
        raise SkillRefError("缺少 skill 名（形如 genre-templates/xxx）")
    if ".." in clean.split("/"):
        raise SkillRefError(f"skill 名里不许出现 ..：{skill_id!r}")
    base = skills_dir().resolve()
    root = (base / clean).resolve()
    #: ``resolve()`` 之后再比 —— 符号链接指到库外的情形必须在这里挡住
    if root != base and not root.is_relative_to(base):
        raise SkillRefError(f"skill 名越界：{skill_id!r}")
    if not root.is_dir():
        raise SkillRefError(f"没有这个 skill：{clean}")
    return root


def _asset_path(skill_id: str, rel: str) -> Path:
    """附件相对路径 → 解析后的文件路径（**必须**落在 ``references/`` 内）。"""
    root = _skill_root(skill_id)
    assets = (root / "references")
    if not assets.is_dir():
        raise SkillRefError(f"{skill_id} 没有 references/ 目录（它没有可读的附件）")
    clean = (rel or "").strip().replace("\\", "/").lstrip("/")
    if not clean:
        raise SkillRefError("缺少附件路径（先用 list 看有哪些）")
    if ".." in clean.split("/"):
        raise SkillRefError(f"附件路径里不许出现 ..：{rel!r}")
    target = (root / clean).resolve()
    #: ⚠️ 判据是「解析后仍在 references/ 内」，不是「字符串以 references/ 开头」
    #: —— 后者会被 ``references/x/../../SKILL.md`` 绕过。
    if target != assets and not target.is_relative_to(assets):
        raise SkillRefError(f"只允许读 references/ 下的文件：{rel!r}")
    suffix = target.suffix.lower()
    if suffix not in TEXT_SUFFIXES:
        raise SkillRefError(
            f"不读 {suffix or '（无后缀）'} 文件：这类资源走媒体接口，不走文本读取"
            f"（白名单：{'、'.join(TEXT_SUFFIXES)}）"
        )
    if not target.is_file():
        raise SkillRefError(f"没有这个附件：{rel!r}")
    return target


def _first_line(path: Path) -> str:
    """文件首行（当摘要用）：截 100 字，空行跳过。"""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if text:
            return text[:100]
    return ""


def list_refs(skill_id: str, *, prefix: str | None = None) -> dict[str, Any]:
    """列出某 skill 的 ``references/`` 附件（可只列某个子目录）。

    三档回执，**不许混**：

    * ``available=false`` ＋ ``hint`` —— 这个 skill 没有 ``references/``（不是空，是没有）；
    * ``files=[]`` ＋ ``total=0``     —— 有目录但（该前缀下）一份都没有；
    * ``files=[...]``                —— 真实清单，``bytes``/``firstLine`` 让模型不用逐份试读。
    """
    root = _skill_root(skill_id)
    assets = root / "references"
    if not assets.is_dir():
        return {
            "skill": skill_id, "available": False, "total": 0, "files": [],
            "hint": "这个 skill 没有 references/ 资产目录（它只有 SKILL.md 正文）",
        }

    entries: list[dict[str, Any]] = []
    for path in sorted(assets.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        if prefix and not relative.startswith(prefix.strip().replace("\\", "/")):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        entries.append({
            "path": relative,
            "bytes": size,
            "firstLine": _first_line(path) if size <= MAX_BYTES else "",
            #: 超上限的文件不预读首行，但**照样列出** —— 否则模型会以为它不存在
            "readable": size <= MAX_BYTES,
        })

    return {
        "skill": skill_id,
        "available": True,
        "total": len(entries),
        "files": entries,
        "hint": "" if entries else f"references/ 下{'该前缀内' if prefix else ''}没有文本附件",
    }


def read_ref(
    skill_id: str, rel: str, *, offset: int = 1, limit: int = 200,
) -> dict[str, Any]:
    """读 ``references/`` 下某一份的正文（按行分页）。

    回执带 ``totalLines`` 与 ``truncated``：模型要知道「还有多少没看」，
    否则会拿半份当全部（与 ``text_slice`` 的截断纪律一致）。
    """
    path = _asset_path(skill_id, rel)
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise SkillRefError(
            f"{rel!r} 有 {size} 字节，超过单份上限 {MAX_BYTES}（{MAX_BYTES // 1024} KB）"
            "—— 这类文件请用知识库检索，不要整份读"
        )

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    start = max(0, int(offset) - 1)
    count = max(1, min(int(limit), MAX_LINES))
    chunk = lines[start:start + count]

    return {
        "skill": skill_id,
        "path": rel,
        "totalLines": total,
        "offset": start + 1,
        "limit": count,
        "truncated": start + count < total,
        "content": "\n".join(chunk),
    }
