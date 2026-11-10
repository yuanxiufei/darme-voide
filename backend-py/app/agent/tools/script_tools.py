"""剧本改写工具（移植自 ``agents/tools/script-tools.ts``，83 行）。

四个工具组成「读 → 改写 → 存」闭环：

* ``read_episode_script``   读本集原文（**超长自动滑窗截断**，见 ``text_slice``）；
* ``rewrite_to_screenplay`` 读原文 **+ 注入格式化剧本规范**（``SCREENPLAY_FORMAT_RULES``）
  一并返回给模型 —— 规范只在 ``prompt_blocks`` 定义一次，杜绝副本漂移；
* ``edit_script``           **增量改动**（P1-2）：只送改了的那几句，``oldText`` 必须在
  原文里逐字唯一出现，整批原子生效（实现见 :mod:`app.services.markdown_patch`）；
* ``save_script``           整篇覆盖落库并**重算剧本指纹**（指纹门禁：改写后旧指纹必须失效）。

⚠️ ``edit_script`` 与 ``save_script`` **不是二选一的关系**：改一两句用小改动（省上下文、
且能发现自己记错原文），真要重写整场才用整篇覆盖。两个都刷新指纹。

⚠️ 三处保真点：

* 原文取值链是 ``content || scriptContent``（两个字段都可能承载原文）；
* 三个工具的报错文案各不相同（``… (id=N)`` 带 id 的两条只在 ``read`` 里）；
* 保存后**必须**刷新指纹，否则下游（分镜/提取）会拿着过期指纹判断"剧本没变"。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select, update
from sqlalchemy.engine import Connection

from app.core.db import engine
from app.core.models import episodes
from app.core.response import now
from app.services.markdown_patch import apply_edits, coerce_edits, render_result
from app.services.prompt_blocks import SCREENPLAY_FORMAT_RULES
from app.services.script_fingerprint import refresh_episode_script_hash
from app.services.text_slice import slice_long_text
from app.agent.tool import Tool, array_of, json_string, object_schema

__all__ = ["create_script_tools"]


def create_script_tools(episode_id: int) -> dict[str, Tool]:
    """工厂（闭包注入 ``episodeId``；每个工具自开短事务）。"""

    def _episode_source(conn: Connection) -> tuple[bool, str]:
        """返回 ``(是否找到剧集, 原文)``。"""
        row = conn.execute(
            select(episodes.c.content, episodes.c.script_content)
            .where(episodes.c.id == episode_id)
        ).first()
        if row is None:
            return False, ""
        return True, (row[0] or row[1] or "")

    async def read_episode_script(_arguments: dict[str, Any]) -> dict[str, Any]:
        with engine.begin() as conn:
            found, content = _episode_source(conn)
        if not found:
            return {"error": f"Episode not found (id={episode_id})"}
        if not content:
            return {"error": f"Episode has no content (id={episode_id})"}
        sliced = slice_long_text(content)
        return {
            "content": sliced["text"],
            "word_count": len(content),
            "episode_id": episode_id,
            "truncated": sliced["truncated"],
            "total_chars": sliced["total_chars"],
        }

    async def rewrite_to_screenplay(arguments: dict[str, Any]) -> dict[str, Any]:
        instructions = arguments.get("instructions")
        with engine.begin() as conn:
            found, source = _episode_source(conn)
        if not found:
            return {"error": "Episode not found"}
        if not source:
            return {"error": "Episode has no content to rewrite"}
        sliced = slice_long_text(source)
        return {
            "source_content": sliced["text"],
            "truncated": sliced["truncated"],
            "total_chars": sliced["total_chars"],
            "instruction": (
                "请将以下内容改写为格式化剧本。\n\n"
                f"{SCREENPLAY_FORMAT_RULES}\n\n"
                f"{instructions or ''}\n\n"
                "【原始内容】\n"
                f"{sliced['text']}"
            ),
        }

    async def edit_script(arguments: dict[str, Any]) -> dict[str, Any]:
        """按增量改动改本集剧本（``oldText`` 必须逐字唯一出现，整批原子）。"""
        edits, problem = coerce_edits(arguments.get("edits"))
        if problem:
            return {"error": problem, "appliedCount": 0}
        if not edits:
            return {"error": "edits 是空数组，没有要改的内容。", "appliedCount": 0}
        with engine.begin() as conn:
            found, content = _episode_source(conn)
            if not found:
                return {"error": f"Episode not found (id={episode_id})", "appliedCount": 0}
            if not content:
                return {"error": f"Episode has no content (id={episode_id})", "appliedCount": 0}
            result = apply_edits(content, edits)
            if not result.ok:
                # 一条都没落：把"第几条、命中几次、怎么改"原样回灌，模型才有第二次机会
                return {
                    "error": result.error,
                    "failedIndex": result.failed_index,
                    "appliedCount": 0,
                    "note": "本次**没有**修改剧本，请按提示改正 oldText 后重送整批。",
                }
            conn.execute(
                update(episodes).where(episodes.c.id == episode_id).values(
                    script_content=result.text, updated_at=now()
                )
            )
            refresh_episode_script_hash(conn, episode_id)
        return {
            "message": render_result(result),
            "word_count": len(result.text),
            "appliedCount": len(result.applied),
            "applied": result.applied,
            "warnings": result.warnings,
        }

    async def save_script(arguments: dict[str, Any]) -> dict[str, Any]:
        content = arguments.get("content") or ""
        with engine.begin() as conn:
            conn.execute(
                update(episodes).where(episodes.c.id == episode_id).values(
                    script_content=content, updated_at=now()
                )
            )
            # 剧本内容指纹门禁：改写落库后重算指纹
            refresh_episode_script_hash(conn, episode_id)
        return {"message": "Script saved", "word_count": len(content)}

    return {
        "read_episode_script": Tool(
            id="read_episode_script",
            description="Read the script content of the current episode.",
            input_schema=object_schema({}, required=[]),
            execute=read_episode_script,
        ),
        "rewrite_to_screenplay": Tool(
            id="rewrite_to_screenplay",
            description=(
                "Read the original content for AI rewriting. "
                "Returns the source text with formatting instructions."
            ),
            input_schema=object_schema(
                {"instructions": json_string("Additional rewrite instructions")}, required=[]
            ),
            execute=rewrite_to_screenplay,
        ),
        "edit_script": Tool(
            id="edit_script",
            description=(
                "Edit the current episode script with small, targeted changes. Each edit's "
                "oldText must appear exactly once in the script; the whole batch is applied "
                "atomically, and nothing is written if any edit fails to match."
            ),
            input_schema=object_schema(
                {
                    "edits": array_of(
                        object_schema(
                            {
                                "oldText": json_string(
                                    "Existing text copied verbatim from the script; must occur "
                                    "exactly once (include enough surrounding context)"
                                ),
                                "newText": json_string("Replacement text"),
                                "reason": json_string("Why this line changes (one sentence)"),
                            },
                            required=["oldText", "newText"],
                        ),
                        "Edits applied in order",
                    )
                }
            ),
            execute=edit_script,
        ),
        "save_script": Tool(
            id="save_script",
            description="Save the rewritten screenplay content to the current episode.",
            input_schema=object_schema(
                {"content": json_string("The formatted screenplay content to save")}
            ),
            execute=save_script,
        ),
    }
