"""台词说话人 → 角色匹配 —— 移植 ``backend/src/shared/character-match.ts``。

对白常用简称/昵称/带称谓（「阿晚」「晚晚」「林小姐」），与角色全名（「林晚」）
做轻量别名归一，保证跨集对白落到同一角色/音色。**仅在「全剧唯一命中」时采用**，
避免误配。

⚠️ 两处 JS 语义必须对齐，否则匹配结果会多出/漏掉角色：

1. **JS 的 ``$``（无 ``m`` 标志）只匹配字符串结尾，Python 的 ``$`` 还会匹配
   「结尾换行符之前」**。若写成 Python 原生的 ``$``，``"音效\\n"`` 会被误判成命中。
   因此本文件一律用 ``\\Z`` 收尾。
2. **JS ``String.replace(re, '')`` 不带 ``g`` 标志时只替换第一处**，Python 的
   ``re.sub`` 默认替换全部 ⇒ 必须显式 ``count=1``。
"""

from __future__ import annotations

import re
from typing import Any

#: 常见称谓后缀/前缀，用于说话人别名归一。
#: 结尾用 \Z（不是 $）—— 见模块头第 1 条。
SPEAKER_TITLE_SUFFIX = re.compile(
    r"(?:小姐|夫人|太太|先生|公子|老爷|少爷|姑娘|师傅|师父|前辈|老师|阿姨|哥哥|姐姐|妹妹|弟弟|妈妈|爸爸|母亲|父亲|奶奶|爷爷|祖母|祖父)\Z"
)
SPEAKER_TITLE_PREFIX = re.compile(r"\A(?:阿|小|老)")


def strip_speaker_affixes(name: str) -> str:
    """循环剥离称谓前后缀（对齐 TS 的 while 循环 + 单次替换语义）。"""
    out = name.strip()
    prev = ""
    while prev != out and len(out) > 0:
        prev = out
        # count=1：JS 的 replace 无 g 标志，只替换第一处
        out = SPEAKER_TITLE_SUFFIX.sub("", out, count=1)
        out = SPEAKER_TITLE_PREFIX.sub("", out, count=1)
    return out


def match_character_by_speaker_name(chars: list[Any], speaker: str) -> Any | None:
    """全剧唯一命中才返回；无法唯一确定时返回 None（调用方按 not_found 处理）。

    判据顺序（照抄 TS）：

    1. 全名精确（唯一才采用）
    2. 说话人去称谓词缀后与全名精确（如「阿晚」「林小姐」→「林晚」）
    3. 双向包含（如「晚晚」「晚儿」⊂「林晚」，**须长度 >= 2 且全剧唯一**）
    """
    raw = (speaker or "").strip()
    if not raw:
        return None

    def _name(c: Any) -> str:
        return (getattr(c, "name", None) or "").strip()

    exact = [c for c in chars if _name(c) == raw]
    if len(exact) == 1:
        return exact[0]

    norm = strip_speaker_affixes(raw)
    if norm and norm != raw:
        candidates: list[Any] = []
        for c in chars:
            n = _name(c)
            if not n or n == raw:
                continue
            if n == norm or strip_speaker_affixes(n) == norm:
                candidates.append(c)
                continue
            a = len(norm) >= 2 and norm in n
            b = len(n) >= 2 and n in norm
            if a or b:
                candidates.append(c)
        if len(candidates) == 1:
            return candidates[0]
    return None
