"""H3 视频提示词的**说话人标签**校验与补齐（``speaker-tags`` 语义 ✓，自研重写 ✓）。

H3 是**音视频同生成**的模型：提示词里谁说话要写成 ``<d>男人(S1)：……</d>`` 这种带
``(S1)`` 标签的形式 —— 没有标签，模型就分不清「这句是谁说的」，口型与音色会串；
而**画外音/旁白**还必须标 ``off-screen``，否则模型会**硬要在画面里找这个人**
（症状：凭空多出一个人，或把旁白安到某个角色脸上 ✗）。

⚠️ 口径来源（2026-11-02 逐条核对 ✓）：上游 ``ComfyUI-Local-Infinite-Canvas``（MIT ✓）
的 ``comfyui-gateway/src/speaker-tags.js`` ⇒ 只搬**语义与正则**，代码自己写 ✗。

## 四处保真点

1. ⭐ **只动「描述段」** ✓：从 ``integrated_multimodal_description:`` /
   ``detailed_description:`` 起，到 ``overall_soundscape:`` / ``non_diegetic_music:`` 止；
   段外（全局音景、非剧情音乐）**一字不碰** ✗（在那里补标签会把音乐当台词 ✗）。
2. ⭐ **同一主体复用同一个 id** ✓：``男人`` 第二次出现仍是 ``S1``（靠
   ``主体(`` 的既有配对记下来 ✓）；新主体才从 ``S1..S9`` 里取**最小未用** ✓
   （取最大会让 id 随提示词长度乱跳 ✗）。
3. ⭐ **取该段里**最后一个**主体** ✓：``<d>前是「男人……，女人(S2)……」`` 指的是
   离标签最近那个人 ⇒ 用 ``subjects.at(-1)``，用第一个会**张冠李戴** ✗。
4. ⭐ **画外说明只加一次** ✓：全文第一处 ``off-screen`` 后面挂
   「（不在画面中，不对应任何参考图）」，后面几处不再重复 ✓ ——
   重复会让提示词变长而**稀释**其它约束 ✗。中英文按该处文本语言选 ✓。

## 一个「宁可记问题，不硬猜」的边界

段里连一个主体词（男人/女人/旁白/old man…）都没有 ⇒ **不编 id**，只把这句记进
``issues`` ✓ —— 编一个 ``S1`` 上去等于凭空造人 ✗。

运行::

    from app.services.comfyui_speaker_tags import ensure_speaker_tags
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["ensure_speaker_tags", "normalize_speaker_ids"]

#: 一段台词（``<d …>…</d>`` ✓）
TAG_RE = re.compile(r"<d(?:\s[^>]*)?>.*?</d>", re.IGNORECASE | re.DOTALL)
#: 已有说话人 id：``(S1)`` / ``(S1, S2)`` ✓
SPEAKER_ID_RE = re.compile(r"\((S\d+(?:\s*[,，]\s*S\d+)*)\)")
#: 画外/旁白的措辞 ✓
OFFSCREEN_RE = re.compile(r"画(?:面)?[^，。！？!?；;\n]{0,10}外|画外音|旁白|\boff-screen\b|\bvoice-?over\b", re.IGNORECASE)
#: 主体词（顺序**不能调**：``老头儿`` 要在 ``老头`` 前，否则切出 ``老头`` + ``儿`` ✗）
SUBJECT_RE = re.compile(
    r"(老头儿|老头|老太太|老人|男人|女人|男孩|女孩|孩子|少年|少女|旁白|\bold man\b|\bold woman\b|\bman\b|\bwoman\b|\bboy\b|\bgirl\b|\bnarrator\b)",
    re.IGNORECASE,
)
#: 主体 + 已带 id：``男人(S1)`` ⇒ 记下「男人 → S1」✓
SUBJECT_ID_RE = re.compile(
    r"(老头儿|老头|老太太|老人|男人|女人|男孩|女孩|孩子|少年|少女|旁白|\bold man\b|\bold woman\b|\bman\b|\bwoman\b|\bboy\b|\bgirl\b|\bnarrator\b)\s*\((S\d+)",
    re.IGNORECASE,
)
#: 描述段的起止锚点 ✓
_DESC_START_RE = re.compile(r"(integrated_multimodal_description|detailed_description)\s*:", re.IGNORECASE)
_DESC_END_RE = re.compile(r"\n\s*(?:overall_soundscape|non_diegetic_music)\s*:", re.IGNORECASE)
#: 一对括号前后各留一点上下文，用来判断「是不是画外」✓
_PAIR_RE = re.compile(r"([^，。！？!?；;\n]{0,60}?)(\((?:S\d+)(?:\s*[,，]\s*S\d+)*\))([^，。！？!?；;\n]{0,30})")
_TRAILING_RE = re.compile(r"[\s，。！？!?；;、：:）)]")
_FULLWIDTH_ID_RE = re.compile(r"（\s*(S\d+(?:\s*[,，]\s*S\d+)*)\s*）")

OFFSCREEN_NOTE_ZH = "（不在画面中，不对应任何参考图）"
OFFSCREEN_NOTE_EN = " (off-screen voice, not in any reference picture)"


def _is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fa5]", text or ""))


def normalize_speaker_ids(text: str) -> str:
    """全角括号与空格归一 ✓：``（ S1， S2 ）`` → ``(S1,S2)`` ✓。"""
    return _FULLWIDTH_ID_RE.sub(lambda match: f"({re.sub(r'\\s*[,，]\\s*', ',', match.group(1))})", text or "")


def _description_bounds(prompt: str) -> tuple[int, int]:
    """描述段的 ``(start, end)`` ✓（找不到锚点 ⇒ 整段都算 ✓）。"""
    match = _DESC_START_RE.search(prompt or "")
    if not match:
        return 0, len(prompt or "")
    start = match.end()
    rest = (prompt or "")[start:]
    next_match = _DESC_END_RE.search(rest)
    return start, start + next_match.start() if next_match else len(prompt or "")


def _subject_key(value: str) -> str:
    text = (value or "").lower()
    if "老头" in text or "old man" in text:
        return "old man"
    if "老太太" in text or "old woman" in text:
        return "old woman"
    for token, key in (("男人", "man"), ("女人", "woman"), ("男孩", "boy"), ("女孩", "girl"), ("旁白", "narrator")):
        if token in text:
            return key
    return text


def _next_speaker_id(used: set[str]) -> str:
    for index in range(1, 10):
        candidate = f"S{index}"
        if candidate not in used:
            return candidate
    return "S1"


def _add_missing_speaker_ids(description: str) -> tuple[str, int, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    fixed = 0
    used = {match.group(1).split(",")[0].strip() for match in SPEAKER_ID_RE.finditer(description)}
    subject_ids = {_subject_key(match.group(1)): match.group(2) for match in SUBJECT_ID_RE.finditer(description)}
    insertions: list[tuple[int, str]] = []
    last_tag_end = 0
    for tag in TAG_RE.finditer(description):
        segment = description[last_tag_end : tag.start()]
        segment_start = tag.start() - len(segment)
        last_tag_end = tag.end()
        if re.search(r"\(S\d+", segment):
            continue
        subjects = list(SUBJECT_RE.finditer(segment))
        if not subjects:
            issues.append({"reason": "missing-speaker", "text": tag.group(0)[:80]})
            continue
        subject = subjects[-1]  # ⭐ 取**最近**那个主体（用第一个会张冠李戴 ✗）
        key = _subject_key(subject.group(1))
        speaker = subject_ids.get(key) or _next_speaker_id(used)
        if key not in subject_ids:
            subject_ids[key] = speaker
            used.add(speaker)
        at = segment_start + subject.end()
        next_char = description[at] if at < len(description) else ""
        trailing = " " if next_char and not _TRAILING_RE.match(next_char) else ""
        insertions.append((at, f" ({speaker}){trailing}"))
        fixed += 1
    output = description
    for at, text in sorted(insertions, key=lambda item: item[0], reverse=True):
        output = output[:at] + text + output[at:]
    return output, fixed, issues


def _add_off_screen_markers(description: str) -> tuple[str, int]:
    fixed = 0
    has_note = bool(re.search(r"(?:不在画面|不对应任何参考图|not in any reference picture)", description, re.IGNORECASE))
    state = {"has_note": has_note, "fixed": 0}

    def replace(match: re.Match) -> str:
        before, speaker_id, after = match.group(1), match.group(2), match.group(3)
        if not OFFSCREEN_RE.search(before):
            return match.group(0)
        if re.search(r"\boff-screen\b", before, re.IGNORECASE) or re.search(r"\boff-screen\b", after, re.IGNORECASE):
            return match.group(0)
        note = "" if state["has_note"] else (OFFSCREEN_NOTE_ZH if _is_chinese(before + after) else OFFSCREEN_NOTE_EN)
        state["has_note"] = True
        state["fixed"] += 1
        return f"{before}{speaker_id} off-screen{note}{after}"

    output = _PAIR_RE.sub(replace, description)
    fixed = state["fixed"]
    if not state["has_note"] and re.search(r"\boff-screen\b", output, re.IGNORECASE):
        # 说明**只补一次** ✓（每处都补会把提示词撑长、稀释其它约束 ✗）
        note = OFFSCREEN_NOTE_ZH if _is_chinese(output) else OFFSCREEN_NOTE_EN
        output = re.sub(r"\boff-screen\b", lambda match: f"{match.group(0)}{note}", output, count=1, flags=re.IGNORECASE)
        fixed += 1
    return output, fixed


def ensure_speaker_tags(output: Any) -> dict[str, Any]:
    """给 H3 提示词补说话人标签 ✓，返回 ``{prompt, fixed, issues}``。

    * ``prompt``：补齐后的整段提示词（**段外一字不动** ✓）；
    * ``fixed``：**实际改动**了几处（0 ⇒ 原本就合规 ✓）；
    * ``issues``：认不出主体、**没敢编**的地方 ✓（要人来看，不是报错 ✗）。
    """
    prompt = str(output or "")
    start, end = _description_bounds(prompt)
    description = normalize_speaker_ids(prompt[start:end])
    description, missing_fixed, issues = _add_missing_speaker_ids(description)
    description, offscreen_fixed = _add_off_screen_markers(description)
    return {
        "prompt": prompt[:start] + description + prompt[end:],
        "fixed": missing_fixed + offscreen_fixed,
        "issues": issues,
    }
