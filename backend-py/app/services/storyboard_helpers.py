"""分镜的关联同步 + 台词解析/TTS 匹配校验。

移植自两处 TS（原代码分散在 ``routes/storyboards.ts`` 的本地函数 与
``agents/tools/storyboard-tools.ts`` 的同名实现）：

* 关联同步：`syncStoryboardProps` / `getStoryboardPropIds`
  （原在 agents/tools 里，只是被路由 import；Python 侧统一放这里，避免路由依赖 Agent 目录）
* 本地工具：`syncStoryboardCharacters` / `getStoryboardCharacterIds` /
  `getStoryboardCharacterCostumes` / `validateStoryboardBindings`
* 台词处理：`parseDialogueLines` / `parseDialogueForTTS` / `validateTTSSpeaker` /
  `validateDialogueLines`

⚠️ 正则的 JS 语义对齐（与 ``character_match.py`` 同源）：

1. JS ``$``（无 ``m``）只匹配字符串**结尾**，Python ``$`` 还会匹配结尾换行之前 ⇒ **一律用 ``\\Z``**。
2. JS ``replace`` 无 ``g`` 时只替换首处，Python ``re.sub`` 默认全替换 ⇒ 显式 ``count=1``。
3. 全局匹配用 ``re.finditer``（与 JS 的 `while (re.exec)` 同为「从左到右、不重叠」）。
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.engine import Connection

from ..models import (
    characters as characters_table,
    episode_characters,
    episode_scenes,
    storyboard_characters,
    storyboard_props,
)
from .character_match import match_character_by_speaker_name

#: 这些「说话人」不是角色，是音效/环境音标记 —— 不参与 TTS 角色匹配。
#: 结尾用 \Z（不是 $），见模块头第 1 条。
IGNORE_TTS_SPEAKERS = re.compile(
    r"\A(环境音|环境声|音效|效果音|sfx|sound ?effect|bgm|背景音|背景音乐|ambient)\Z",
    re.IGNORECASE,
)

#: 这些台词内容无需配音。
IGNORE_TTS_TEXT = re.compile(
    r"\A(无|无对白|无台词|无旁白|无需配音|无需对白|none|null|n/a|na|环境音|环境声|音效|效果音|纯音效|纯环境音|只有环境音|仅环境音|背景音|背景音乐|bgm|sfx|ambient)\Z",
    re.IGNORECASE,
)

#: 旁白/画外音直接视为匹配通过。
_NARRATOR = re.compile(r"\A(旁白|画外音|narrator)\Z", re.IGNORECASE)

#: 对话行拆分：角色名(不含冒号，1-10 字) + 冒号 + 至少 8 字符台词(不含冒号)
_DIALOGUE_LINE = re.compile(r"([^\n:：]{1,10}?)[:：]([^:：\n]{8,})")

#: 去掉台词里的括号补充说明（TS 带 ``g`` 标志 ⇒ 全替换）
_PAREN = re.compile(r"[（(].+?[)）]")
#: 单人模式：捕获开头「角色名：」里的角色名（TS: ``/^(.+?)[:：]/``）
_SPEAKER_CAPTURE = re.compile(r"\A(.+?)[:：]")
#: 单人模式：剥掉开头「角色名：」及紧随空白（TS: ``/^.+?[:：]\s*/``，注意含 ``\s*``）
_SPEAKER_STRIP = re.compile(r"\A.+?[:：]\s*")


# ---------------------------------------------------------------------------
# 关联同步
# ---------------------------------------------------------------------------

def sync_storyboard_props(conn: Connection, storyboard_id: int, prop_ids: list[Any]) -> None:
    conn.execute(
        delete(storyboard_props).where(storyboard_props.c.storyboard_id == storyboard_id)
    )
    unique_ids = {p for p in (prop_ids or []) if p}
    for prop_id in unique_ids:
        conn.execute(
            storyboard_props.insert().values(storyboard_id=storyboard_id, prop_id=prop_id)
        )


def get_storyboard_prop_ids(conn: Connection, storyboard_id: int) -> list[int]:
    rows = conn.execute(
        select(storyboard_props.c.prop_id).where(storyboard_props.c.storyboard_id == storyboard_id)
    ).all()
    return [r.prop_id for r in rows]


def sync_storyboard_characters(
    conn: Connection,
    storyboard_id: int,
    character_ids: list[Any],
    costumes: dict[Any, str] | None = None,
) -> None:
    conn.execute(
        delete(storyboard_characters).where(storyboard_characters.c.storyboard_id == storyboard_id)
    )
    # 去重且保序（对齐 TS 的 [...new Set(...)]）
    seen: set[Any] = set()
    unique_ids: list[Any] = []
    for cid in character_ids or []:
        if not cid or cid in seen:
            continue
        seen.add(cid)
        unique_ids.append(cid)
    if not unique_ids:
        return
    for character_id in unique_ids:
        # 注意 TS 用 costumes?.[characterId] ?? null ⇒ 缺省/undefined 写 null，
        # 但空字符串 '' 会原样保留
        costume = None
        if costumes is not None and character_id in costumes:
            costume = costumes[character_id]
        conn.execute(
            storyboard_characters.insert().values(
                storyboard_id=storyboard_id, character_id=character_id, costume=costume
            )
        )


def get_storyboard_character_ids(conn: Connection, storyboard_id: int) -> list[int]:
    rows = conn.execute(
        select(storyboard_characters.c.character_id).where(
            storyboard_characters.c.storyboard_id == storyboard_id
        )
    ).all()
    return [r.character_id for r in rows]


def get_storyboard_character_costumes(conn: Connection, storyboard_id: int) -> dict[int, str]:
    """镜头级角色服装变体映射 ``{character_id: costume}``（只含非空 costume）。"""
    rows = conn.execute(
        select(storyboard_characters.c.character_id, storyboard_characters.c.costume).where(
            storyboard_characters.c.storyboard_id == storyboard_id
        )
    ).all()
    return {r.character_id: r.costume for r in rows if r.costume}


def validate_storyboard_bindings(
    conn: Connection,
    episode_id: Any,
    scene_id: Any,
    character_ids: list[Any] | None,
) -> None:
    """校验 scene_id / character_ids 均来自当前集已关联的场景与角色，否则抛 ValueError。

    ⚠️ 注意 ``agents/tools/storyboard-tools.ts`` 里有个**同名但文案不同**的实现
    （``scene_id {id} 不属于当前集``）；路由用的是本版本（``必须来自当前集已关联场景``）。
    """
    episode_scene_ids = {
        r.scene_id
        for r in conn.execute(
            select(episode_scenes.c.scene_id).where(episode_scenes.c.episode_id == episode_id)
        ).all()
    }
    episode_character_ids = {
        r.character_id
        for r in conn.execute(
            select(episode_characters.c.character_id).where(
                episode_characters.c.episode_id == episode_id
            )
        ).all()
    }

    if scene_id is not None and scene_id not in episode_scene_ids:
        raise ValueError("scene_id 必须来自当前集已关联场景")

    invalid = [cid for cid in (character_ids or []) if cid not in episode_character_ids]
    if invalid:
        raise ValueError("character_ids 必须来自当前集已关联角色")


# ---------------------------------------------------------------------------
# 台词解析
# ---------------------------------------------------------------------------

def parse_dialogue_lines(dialogue: str | None) -> list[dict[str, str]]:
    """把「角色名：较长台词」的多角色对话文本拆成独立对话行。

    只匹配真正像对话的模式：冒号后必须 >= 8 个字符且不含冒号，
    避免把句子里的标点/短句误拆。
    """
    raw = (dialogue or "").strip()
    if not raw:
        return []

    lines: list[dict[str, str]] = []
    # finditer 与 JS 的 `while (re.exec)` 同为「从左到右、不重叠」的全局匹配
    for match in _DIALOGUE_LINE.finditer(raw):
        speaker = _PAREN.sub("", match.group(1)).strip()
        text = match.group(2).strip()
        if not speaker or len(speaker) > 10 or IGNORE_TTS_SPEAKERS.search(speaker):
            continue
        if IGNORE_TTS_TEXT.search(text):
            continue
        lines.append({"speaker": speaker, "text": text})
    return lines


def parse_dialogue_for_tts(dialogue: str | None) -> dict[str, Any]:
    """单人模式解析：取开头「角色名：」后的正文。"""
    raw = (dialogue or "").strip()
    if not raw:
        return {"speaker": "", "pureText": "", "ignorable": True}

    speaker_match = _SPEAKER_CAPTURE.match(raw)
    speaker = _PAREN.sub("", speaker_match.group(1)).strip() if speaker_match else ""
    # TS: raw.replace(/^.+?[:：]\s*/, '').replace(/[（(].+?[)）]/g, '').trim()
    # 去前缀那条无 g 标志 ⇒ count=1；去括号那条带 g ⇒ 全替换
    pure_text = _PAREN.sub("", _SPEAKER_STRIP.sub("", raw, count=1)).strip()
    ignorable = (
        (bool(speaker) and bool(IGNORE_TTS_SPEAKERS.search(speaker)))
        or not pure_text
        or bool(IGNORE_TTS_TEXT.search(pure_text))
    )
    return {"speaker": speaker, "pureText": pure_text, "ignorable": ignorable}


def validate_tts_speaker(conn: Connection, speaker: str, drama_id: int) -> dict[str, Any]:
    """台词行 → 剧组角色 → 音色 三连匹配。

    状态：``matched``（已配音色）/ ``no_voice``（角色存在但无音色，回落 alloy）/
    ``not_found``（角色名不在剧组）/ ``narrator``（旁白，无需匹配）。
    """
    if _NARRATOR.match(speaker or ""):
        return {"match_status": "narrator", "voiceId": "alloy"}

    chars = conn.execute(
        select(characters_table).where(
            characters_table.c.drama_id == drama_id, characters_table.c.deleted_at.is_(None)
        )
    ).all()
    found = match_character_by_speaker_name(chars, speaker)

    if found is None:
        return {"match_status": "not_found", "voiceId": "alloy", "characterId": None}
    if not found.voice_style:
        return {"match_status": "no_voice", "voiceId": "alloy", "characterId": found.id}
    return {"match_status": "matched", "voiceId": found.voice_style, "characterId": found.id}


def validate_dialogue_lines(
    conn: Connection, speaker_lines: list[dict[str, str]], drama_id: int
) -> list[dict[str, Any]]:
    """验证整段 dialogue 中所有台词行的匹配状态。"""
    out: list[dict[str, Any]] = []
    for line in speaker_lines:
        v = validate_tts_speaker(conn, line["speaker"], drama_id)
        warning: str | None = None
        if v["match_status"] == "not_found":
            warning = f'角色"{line["speaker"]}"在剧组角色列表中不存在，将使用默认音色'
        elif v["match_status"] == "no_voice":
            warning = f'角色"{line["speaker"]}"尚未配置音色，将使用默认音色'

        item: dict[str, Any] = {
            "speaker": line["speaker"],
            "text": line["text"],
            "match_status": v["match_status"],
            "voice_id": v["voiceId"],
            "character_id": v.get("characterId"),
        }
        # TS 是条件展开 `...(warning ? { warning } : {})` ⇒ 无 warning 时**不带该键**
        if warning:
            item["warning"] = warning
        out.append(item)
    return out
