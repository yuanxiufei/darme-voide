"""Skill 附件的按需读取工具（两个：列目录 / 读一份）。

**只做转发**：路径校验、后缀白名单、字节与行数上限全在 ``services/skill_refs.py``
（单一真源）。本文件不重复任何口径 —— 否则 HTTP 层、Agent 工具、CLI 三处各写一遍
路径校验，迟早有一处写成「字符串以 references/ 开头」而被 ``../`` 绕过。

为什么挂在编剧与分镜两个 Agent 上：技能库里的方法论文本（镜头语法、打戏编排、
布光判据…）正是这两个 Agent 干活时要查的东西；而 skill 加载器只注入 ``SKILL.md``，
附件正文得由模型自己按需读 —— 给它手，不替它把整库塞进上下文。
"""
from __future__ import annotations

from typing import Any

from app.agent.tool import Tool, json_number, json_string, object_schema
from app.services import skill_refs
from app.services.skill_refs import SkillRefError

__all__ = ["create_skill_ref_tools"]


def create_skill_ref_tools() -> dict[str, Tool]:
    """技能库附件工具（编剧/分镜 Agent 共用；无状态、纯只读）。"""

    async def list_skill_refs(arguments: dict[str, Any]) -> dict[str, Any]:
        skill = str(arguments.get("skill") or "").strip()
        prefix = arguments.get("prefix")
        try:
            return skill_refs.list_refs(
                skill, prefix=str(prefix).strip() if prefix else None
            )
        except SkillRefError as error:
            return {"error": str(error)}

    async def read_skill_ref(arguments: dict[str, Any]) -> dict[str, Any]:
        skill = str(arguments.get("skill") or "").strip()
        path = str(arguments.get("path") or "").strip()
        offset = arguments.get("offset")
        limit = arguments.get("limit")
        try:
            offset = int(offset) if offset is not None else 1
        except (TypeError, ValueError):
            offset = 1
        try:
            limit = int(limit) if limit is not None else 200
        except (TypeError, ValueError):
            limit = 200
        try:
            return skill_refs.read_ref(skill, path, offset=max(1, offset), limit=limit)
        except SkillRefError as error:
            return {"error": str(error)}

    return {
        "list_skill_refs": Tool(
            id="list_skill_refs",
            description=(
                "列某个 skill 的 references/ 附件清单（只读）。先列再读：回执里每份都有"
                " bytes 与 firstLine，凭这两样决定读哪份，不要逐份试读。"
                "available=false 表示这个 skill 根本没有附件目录（不是空）。"
                "skill 名形如 genre-templates/xxx（与 /skills 接口的 id 同形）。"
            ),
            input_schema=object_schema(
                {
                    "skill": json_string("skill id，形如 genre-templates/xxx"),
                    "prefix": json_string(
                        "可选：只看某个子目录，如 references/library/combat/"
                    ),
                },
                required=["skill"],
            ),
            execute=list_skill_refs,
        ),
        "read_skill_ref": Tool(
            id="read_skill_ref",
            description=(
                "读某个 skill 的 references/ 下某一份正文（按行分页，只读）。"
                "回执带 totalLines 与 truncated：truncated=true 说明还有没看完，"
                "要用 offset 接着读，**别把半份当全部**。"
            ),
            input_schema=object_schema(
                {
                    "skill": json_string("skill id，形如 genre-templates/xxx"),
                    "path": json_string(
                        "附件相对路径，取自 list_skill_refs 的 path 字段（如 references/library/combat/x.md）"
                    ),
                    "offset": json_number("从第几行开始（1 起），默认 1"),
                    "limit": json_number("读多少行，默认 200，最大 400"),
                },
                required=["skill", "path"],
            ),
            execute=read_skill_ref,
        ),
    }
