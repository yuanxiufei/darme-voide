"""角色音色分配工具（移植自 ``agents/tools/voice-tools.ts``，148 行）。

三个工具 + 两条**硬规则**（违反会被系统拒绝，对应 ``DEFAULT_PROMPTS.voice_assigner`` 里的铁律）：

* ``get_characters`` 列出剧组角色与当前音色（``current_voice`` 为「未分配」时表示还没定）；
* ``list_voices``    列出可用音色（**库里没有时回落到 6 个 OpenAI 通用音色**）；
* ``assign_voice``   分配音色：**一角色一音色**（禁止两个角色共用）＋ **speaker_id 全局唯一**。

⚠️ 三处保真点：

1. ``role_tag`` 过滤**只在回落分支生效**（原 TS 就是这么写的 —— 从库里取到的音色不过滤）；
2. ``infer_gender`` 的两个正则是**字符类**（``[男|青年|…|male]``）—— 语义是「任意一个字符命中」，
   不是备选词匹配；``male``/``man`` 能命中纯属字符集巧合。照抄，别"修正"成 ``(男|青年|…)``；
3. 分配时把 ``voice_sample_url`` **显式置空**（换了音色，旧的声线样本必须失效）。
"""
from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import and_, select, update
from sqlalchemy.engine import Connection

from app.core.db import engine
from app.core.models import ai_service_configs, ai_voices, characters, episodes
from app.core.response import now
from app.services.task_logger import log_task_progress, log_task_success, log_task_warn
from app.agent.tool import Tool, json_number, json_string, object_schema

__all__ = ["create_voice_tools"]

#: ⚠️ 字符类（不是备选词）——与 TS 的正则同形
_MALE_HINT = re.compile(r"[男|青年|大爷|学长|boy|man|male]", re.IGNORECASE)
_FEMALE_HINT = re.compile(r"[女|少女|御姐|奶奶|girl|woman|female]", re.IGNORECASE)

#: 库里没有任何音色时的回落（OpenAI 通用音色）
_FALLBACK_VOICES: list[dict[str, Any]] = [
    {"id": "alloy", "name": "Alloy", "gender": "中性", "traits": "平衡自然",
     "suitable_for": "旁白、通用", "role_tags": ["旁白", "配角"], "language": "多语言"},
    {"id": "echo", "name": "Echo", "gender": "男声", "traits": "低沉稳重",
     "suitable_for": "成熟男性、旁白", "role_tags": ["旁白", "主角"], "language": "多语言"},
    {"id": "fable", "name": "Fable", "gender": "男声", "traits": "温暖富有表现力",
     "suitable_for": "年轻男性、故事叙述", "role_tags": ["主角", "旁白"], "language": "多语言"},
    {"id": "onyx", "name": "Onyx", "gender": "男声", "traits": "深沉有力",
     "suitable_for": "权威角色、反派", "role_tags": ["反派", "配角"], "language": "多语言"},
    {"id": "nova", "name": "Nova", "gender": "女声", "traits": "温柔甜美",
     "suitable_for": "年轻女性、女主", "role_tags": ["主角"], "language": "多语言"},
    {"id": "shimmer", "name": "Shimmer", "gender": "女声", "traits": "明亮活泼",
     "suitable_for": "活泼女性、少女", "role_tags": ["主角", "配角"], "language": "多语言"},
]

_LIST_INSTRUCTION = (
    "根据角色的性别、性格、年龄来匹配最合适的音色，并且只能从当前集音频配置可用的音色列表中选择。"
)


def infer_gender(name: str, description: Any) -> str:
    """从音色名 + 描述推断性别（⚠️ 字符类命中，见模块头）。"""
    joined = " ".join(description) if isinstance(description, list) else ""
    text = f"{name} {joined}"
    if _MALE_HINT.search(text):
        return "男声"
    if _FEMALE_HINT.search(text):
        return "女声"
    return "中性"


def parse_role_tags(raw: Any) -> list[str]:
    """容错解析 role_tags（非数组/脏数据 → 空）。"""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def create_voice_tools(episode_id: int, drama_id: int) -> dict[str, Tool]:
    """工厂（闭包注入 episodeId/dramaId；每个工具自开短事务）。"""

    def _episode_audio_provider(conn: Connection) -> str | None:
        """当前集音频配置的 provider（未配则 None）。"""
        episode = conn.execute(
            select(episodes.c.audio_config_id).where(episodes.c.id == episode_id)
        ).first()
        if episode is None or not episode[0]:
            return None
        config = conn.execute(
            select(ai_service_configs.c.provider).where(ai_service_configs.c.id == episode[0])
        ).first()
        return (config[0] if config else None) or None

    async def get_characters(_arguments: dict[str, Any]) -> dict[str, Any]:
        with engine.begin() as conn:
            rows = conn.execute(
                select(characters).where(and_(
                    characters.c.drama_id == drama_id,
                    characters.c.deleted_at.is_(None),
                ))
            ).all()
        payload = {"characters": [
            {
                "id": row.id,
                "name": row.name,
                "role": row.role,
                "personality": row.personality,
                "description": row.description,
                # `c.voiceStyle || '未分配'`
                "current_voice": row.voice_style or "未分配",
                "speaker_id": row.speaker_id or "",
            }
            for row in rows
        ]}
        log_task_success("VoiceTool", "get-characters", {
            "episodeId": episode_id, "dramaId": drama_id,
            "count": len(payload["characters"]),
        })
        return payload

    async def list_voices(arguments: dict[str, Any]) -> dict[str, Any]:
        role_tag = arguments.get("role_tag")
        with engine.begin() as conn:
            provider = _episode_audio_provider(conn) or "minimax"
            rows = conn.execute(
                select(ai_voices).where(ai_voices.c.provider == provider)
            ).all()

        if rows:
            voices = []
            for row in rows:
                desc = json.loads(row.description) if row.description else []
                desc_list = desc if isinstance(desc, list) else []
                voices.append({
                    "id": row.voice_id,
                    "name": row.voice_name,
                    "gender": infer_gender(row.voice_name, desc),
                    "traits": ("、".join(str(x) for x in desc_list[:2]) if desc_list
                               else f"{row.language or '多语言'}音色"),
                    "suitable_for": ("、".join(str(x) for x in desc_list[2:])
                                     if len(desc_list) > 2 else f"{row.language or '通用'}角色"),
                    "role_tags": parse_role_tags(row.role_tags),
                    "language": row.language,
                    "provider": provider,
                })
        else:
            # ⚠️ 过滤**只在这里**（原 TS 如此）：从库里取到的音色不做 role_tag 过滤
            voices = [
                {**voice, "provider": provider}
                for voice in _FALLBACK_VOICES
                if not role_tag or role_tag in voice["role_tags"]
            ]

        payload = {"provider": provider, "voices": voices, "instruction": _LIST_INSTRUCTION}
        log_task_success("VoiceTool", "list-voices", {
            "episodeId": episode_id, "provider": provider, "count": len(voices),
        })
        return payload

    async def assign_voice(arguments: dict[str, Any]) -> dict[str, Any]:
        character_id = arguments.get("character_id")
        voice_id = arguments.get("voice_id")
        speaker_id = arguments.get("speaker_id")
        reason = arguments.get("reason")

        with engine.begin() as conn:
            provider = _episode_audio_provider(conn) or "minimax"
            log_task_progress("VoiceTool", "assign-begin", {
                "episodeId": episode_id, "dramaId": drama_id, "characterId": character_id,
                "voiceId": voice_id, "speakerId": speaker_id, "provider": provider,
                "reason": reason,
            })

            # 硬规则 1：一角色一音色（禁止两个角色共用）
            voice_conflict = next((
                row for row in conn.execute(
                    select(characters.c.id, characters.c.name).where(and_(
                        characters.c.drama_id == drama_id,
                        characters.c.voice_style == voice_id,
                        characters.c.deleted_at.is_(None),
                    ))
                ).all() if row[0] != character_id
            ), None)
            if voice_conflict is not None:
                log_task_warn("VoiceTool", "assign-voice-conflict", {
                    "episodeId": episode_id, "characterId": character_id,
                    "voiceId": voice_id, "conflictName": voice_conflict[1],
                })
                return {"error": (
                    f'音色 "{voice_id}" 已分配给角色「{voice_conflict[1]}」'
                    f"(id={voice_conflict[0]})，禁止共用，请改选其他音色"
                )}

            # 硬规则 2：speaker_id 全局唯一（一个说话人 = 一个角色，跨集稳定）
            if speaker_id:
                speaker_conflict = next((
                    row for row in conn.execute(
                        select(characters.c.id, characters.c.name).where(and_(
                            characters.c.drama_id == drama_id,
                            characters.c.speaker_id == speaker_id,
                            characters.c.deleted_at.is_(None),
                        ))
                    ).all() if row[0] != character_id
                ), None)
                if speaker_conflict is not None:
                    log_task_warn("VoiceTool", "assign-speaker-conflict", {
                        "episodeId": episode_id, "characterId": character_id,
                        "speakerId": speaker_id, "conflictName": speaker_conflict[1],
                    })
                    return {"error": (
                        f'speaker_id "{speaker_id}" 已分配给角色「{speaker_conflict[1]}」'
                        f"(id={speaker_conflict[0]})，请勿复用"
                    )}

            values: dict[str, Any] = {
                "voice_style": voice_id,
                "voice_provider": provider,
                # 换了音色 ⇒ 旧的声线样本必须失效（显式置空，不是"不改"）
                "voice_sample_url": None,
                "updated_at": now(),
            }
            if speaker_id:
                values["speaker_id"] = speaker_id
            conn.execute(update(characters).where(characters.c.id == character_id).values(**values))

        log_task_success("VoiceTool", "assign-complete", {
            "episodeId": episode_id, "characterId": character_id,
            "voiceId": voice_id, "speakerId": speaker_id, "provider": provider,
        })
        return {
            "message": (f'Assigned voice "{voice_id}" '
                        f'(speaker "{speaker_id or "n/a"}") to character {character_id}'),
            "reason": reason,
        }

    return {
        "get_characters": Tool(
            id="get_characters",
            description="Get all characters for the current drama with their current voice assignments.",
            input_schema=object_schema({}, required=[]),
            execute=get_characters,
        ),
        "list_voices": Tool(
            id="list_voices",
            description="List all available voice options for TTS, optionally filtered by role type.",
            input_schema=object_schema(
                {"role_tag": json_string("按角色类型筛选音色：旁白 / 主角 / 反派 / 配角")},
                required=[],
            ),
            execute=list_voices,
        ),
        "assign_voice": Tool(
            id="assign_voice",
            description="Assign a voice to a character.",
            input_schema=object_schema(
                {
                    "character_id": json_number("Character ID"),
                    "voice_id": json_string("Voice ID from list_voices"),
                    "speaker_id": json_string(
                        "Global speaker ID (S1, S2, ...) unique per character and stable across episodes"
                    ),
                    "reason": json_string("Why this voice fits"),
                },
                required=["character_id", "voice_id"],
            ),
            execute=assign_voice,
        ),
    }
