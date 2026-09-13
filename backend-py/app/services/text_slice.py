"""超长输入保护（移植自 ``backend/src/utils/text-slice.ts``，39 行）。

剧本/小说超长时整段注入 LLM 会触发 context length 超限（**fatal、不可重试**），
这里用「**保留前 70% + 后 30%，中间省略**」的确定性滑窗截断：

* 剧本开头（人物出场、设定）与结尾（冲突高潮、结局）通常最关键，优先保留；
* 中间省略处**显式提示**，让 Agent 基于首尾关键情节继续，而不是拿到被硬截断的半句。

默认 24000 字符（中文约 2 字/token ⇒ 约 12k token），远低于常见 32k 上下文。
"""
from __future__ import annotations

from typing import Any

__all__ = ["DEFAULT_MAX_CHARS", "slice_long_text"]

#: 默认上限（字符数）
DEFAULT_MAX_CHARS = 24000


def slice_long_text(content: str | None, max_chars: int = DEFAULT_MAX_CHARS) -> dict[str, Any]:
    """确定性滑窗截断；未超限时原样返回（``truncated=False``）。"""
    total = len(content) if content else 0
    if not content or total <= max_chars:
        return {
            "text": content or "",
            "truncated": False,
            "total_chars": total,
            "kept_chars": total,
        }

    head = int(max_chars * 0.7)  # JS Math.floor
    tail = max_chars - head
    omitted = total - max_chars
    text = (
        content[:head]
        + f"\n\n……（中间 {omitted} 字因内容超长已省略，请基于上文开头与下文结尾的关键情节继续）……\n\n"
        + (content[-tail:] if tail else "")
    )
    return {"text": text, "truncated": True, "total_chars": total, "kept_chars": max_chars}
