"""分镜拆解工具（移植自 ``agents/tools/storyboard-tools.ts``，517 行）—— S5 六组工具的最后一组。

四个工具：

* ``read_storyboard_context`` 读剧本 + 角色 + 场景 + 已有分镜 + **连续性状态** + 物品（拆镜上下文）；
* ``save_storyboards``        **整集重建**：删光旧分镜与关联 → 逐镜校验/插入/绑定 → 回写集时长 → 三件后置；
* ``save_continuity_states``  **整体替换**连续性状态（跨镜一致性状态机）；
* ``update_storyboard``       单镜增量更新（**按字段是否出现**决定是否写，不是"空值覆盖"）。

⚠️ 六处保真点：

1. ``read_storyboard_context`` 的原文取值链是 **``scriptContent || content``**（与 ``extract_tools`` 一致、
   与 ``script_tools`` **相反**）；
2. 角色/场景的过滤是「**本集已关联的优先，未关联任何东西时全给**」——
   ``!linkedIds.size || linkedIds.has(id)``，即**空关联集视为"不过滤"**；
3. ``save_storyboards`` 会**先删光旧分镜及其角色关联**（整集重建），且 ``props`` 关联**不删**
   （原 TS 只删了 ``storyboardCharacters``）；
4. **说话人自动绑定**：``speaker_id`` 未填时从对白首个「角色名：台词」解析，再用
   「角色名 → speaker_id」映射补上；旁白/画外音/narrator **返回 None**（不绑）；
5. **对白一致性校验**只告警不改数据：统计对白里出现但既不在本镜 ``character_ids``
   也不在本集角色名单里的名字（``dialogue_issues`` 计数，并在回执里提示）；
6. ``update_storyboard`` 的每次写入都**以字段出现为准**（``'x' in fields``）⇒
   传 ``null`` 会**真的清空**该列，而不传则保持原值。

⚠️ ``assign_rhythm_phases``（`rhythm-phase.ts`）**尚未移植** ⇒ 保留调用点为**空实现**
（与 compose/merge 的 QC 占位同样处置），``storyboards.rhythm_phase`` 暂时留空。
"""
from __future__ import annotations

import math
import re
from typing import Any

from sqlalchemy import and_, delete as sql_delete, select, update

from app.core.db import engine
from app.core.models import (
    characters,
    continuity_states,
    dramas,
    episode_characters,
    episode_props,
    episode_scenes,
    episodes,
    prop_templates,
    scenes,
    storyboard_characters,
    storyboards,
)
from app.core.response import now
from app.services.script_fingerprint import stamp_storyboards_script_hash
from app.services.storyboard_helpers import (
    sync_storyboard_characters,
    sync_storyboard_props,
    validate_storyboard_bindings,
)
from app.services.take_budget import reset_take_budget_for_episode
from app.services.task_logger import log_task_progress, log_task_success
from app.services.text_slice import slice_long_text
from app.agent.tool import Tool, array_of, json_number, json_string, object_schema

__all__ = ["create_storyboard_tools", "extract_speaker_name"]

#: 对白里「角色名：台词」的说话人前缀（**12 字以内**，与 TS 同形）
_SPEAKER_PREFIX = re.compile(r"^\s*([^\n:：]{1,12}?)[:：]")

#: 对白一致性扫描：角色名（≤10 字）+ 冒号 + **至少 8 字**台词
_DIALOGUE_LINE = re.compile(r"([^\n:：]{1,10}?)[:：]([^:：\n]{8,})")

#: 旁白/画外音不算"角色"（不绑定、也不参与一致性校验）
_NARRATOR_NAMES = re.compile(r"^(旁白|画外音|narrator)$", re.IGNORECASE)

#: 分镜默认时长（秒）——原 TS 两处都用 10
_DEFAULT_DURATION = 10


def extract_speaker_name(dialogue: str | None) -> str | None:
    """从对白解析第一个「角色名：台词」的说话人；旁白/画外音返回 None。"""
    if not dialogue:
        return None
    match = _SPEAKER_PREFIX.match(dialogue)
    if match is None:
        return None
    name = match.group(1).strip()
    if not name or _NARRATOR_NAMES.match(name):
        return None
    return name


def assign_rhythm_phases(episode_id: int) -> None:
    """多集节奏相位分配（``rhythm-phase.ts``）——**保留调用点，逻辑未移植**。"""
    return None


def create_storyboard_tools(episode_id: int, drama_id: int) -> dict[str, Tool]:
    """工厂（闭包注入 id；每个工具自开短事务）。"""

    async def read_storyboard_context(_arguments: dict[str, Any]) -> dict[str, Any]:
        with engine.begin() as conn:
            episode = conn.execute(
                select(episodes).where(episodes.c.id == episode_id)
            ).first()
            if episode is None:
                return {"error": "Episode not found"}
            # ⚠️ 这里是 **scriptContent 优先**（与 script_tools 相反）
            script = episode.script_content or episode.content
            if not script:
                return {"error": "Episode has no script"}
            sliced = slice_long_text(script)

            char_links = conn.execute(
                select(episode_characters.c.character_id)
                .where(episode_characters.c.episode_id == episode_id)
            ).all()
            scene_links = conn.execute(
                select(episode_scenes.c.scene_id)
                .where(episode_scenes.c.episode_id == episode_id)
            ).all()
            linked_character_ids = {row[0] for row in char_links}
            linked_scene_ids = {row[0] for row in scene_links}

            char_rows = conn.execute(
                select(characters).where(and_(
                    characters.c.drama_id == drama_id,
                    characters.c.deleted_at.is_(None),
                ))
            ).all()
            scene_rows = conn.execute(
                select(scenes).where(scenes.c.drama_id == drama_id)
            ).all()

            def char_view(row: Any) -> dict[str, Any]:
                return {
                    "id": row.id,
                    "name": row.name,
                    "role": row.role or "",
                    "description": row.description or "",
                    "appearance": row.appearance or "",
                    "personality": row.personality or "",
                    "voice_style": row.voice_style or "",
                    "speaker_id": row.speaker_id or "",
                    "costume_id": row.costume_id or "",
                    "image_url": row.image_url or "",
                    "reference_images": row.reference_images or "",
                    # 生成 image_prompt / video_prompt 时必须用它保证角色视觉一致
                    "appearance_hint": (f"{row.name}: "
                                        f"{row.appearance or row.description or row.role or ''}"),
                }

            def scene_view(row: Any) -> dict[str, Any]:
                return {
                    "id": row.id,
                    "location": row.location,
                    "location_id": row.location_id or "",
                    "time": row.time,
                    "prompt": row.prompt or "",
                    "image_url": row.image_url or "",
                    "storyboard_count": row.storyboard_count or 0,
                }

            storyboard_rows = conn.execute(
                select(storyboards).where(storyboards.c.episode_id == episode_id)
            ).all()

            existing_storyboards = []
            for row in storyboard_rows:
                if row.deleted_at:
                    continue
                character_ids = [
                    link[0] for link in conn.execute(
                        select(storyboard_characters.c.character_id)
                        .where(storyboard_characters.c.storyboard_id == row.id)
                    ).all()
                ]
                existing_storyboards.append({
                    "id": row.id,
                    "shot_number": row.storyboard_number,
                    "title": row.title or "",
                    "scene_id": row.scene_id,
                    "character_ids": character_ids,
                    "shot_type": row.shot_type or "",
                    "duration": row.duration or 0,
                    "start_state": row.start_state or "",
                    "end_state": row.end_state or "",
                })

            state_rows = conn.execute(
                select(continuity_states)
                .where(continuity_states.c.episode_id == episode_id)
                .order_by(continuity_states.c.id)
            ).all()

            prop_links = conn.execute(
                select(episode_props.c.prop_id)
                .where(episode_props.c.episode_id == episode_id)
            ).all()
            props = []
            for link in prop_links:
                row = conn.execute(
                    select(prop_templates).where(prop_templates.c.id == link[0])
                ).first()
                if row is None or row.deleted_at:
                    continue
                props.append({
                    "id": row.id, "name": row.name, "category": row.category,
                    "appearance": row.appearance, "holder": row.holder,
                })

            drama = conn.execute(select(dramas).where(dramas.c.id == drama_id)).first()

        payload = {
            "style_id": (drama.style_id or "") if drama is not None else "",
            "episode": {
                "id": episode.id,
                "title": episode.title,
                "episode_number": episode.episode_number,
                "description": episode.description or "",
            },
            "script": sliced["text"],
            "script_truncated": sliced["truncated"],
            "script_total_chars": sliced["total_chars"],
            # ⚠️ 空关联集 = **不过滤**（`!linkedIds.size || linkedIds.has(id)`）
            "characters": [
                char_view(row) for row in char_rows
                if not linked_character_ids or row.id in linked_character_ids
            ],
            "scenes": [
                scene_view(row) for row in scene_rows
                if not row.deleted_at
                and (not linked_scene_ids or row.id in linked_scene_ids)
            ],
            "existing_storyboards": existing_storyboards,
            "continuity_states": [
                {
                    "state_type": row.state_type,
                    "entity_key": row.entity_key,
                    "state_value": row.state_value,
                    "constraints": row.constraints or "",
                    "storyboard_id": row.storyboard_id,
                }
                for row in state_rows
            ],
            "props": props,
        }
        log_task_success("StoryboardTool", "read-context", {
            "episodeId": episode_id, "dramaId": drama_id,
            "characters": len(payload["characters"]), "scenes": len(payload["scenes"]),
            "existingStoryboards": len(existing_storyboards),
            "scriptLength": len(script),
        })
        return payload

    async def save_continuity_states(arguments: dict[str, Any]) -> dict[str, Any]:
        states = arguments.get("states") or []
        ts = now()
        with engine.begin() as conn:
            conn.execute(
                sql_delete(continuity_states).where(continuity_states.c.episode_id == episode_id)
            )
            for state in states:
                conn.execute(continuity_states.insert().values(
                    episode_id=episode_id,
                    storyboard_id=state.get("storyboard_id"),
                    scene_id=state.get("scene_id"),
                    state_type=state.get("state_type"),
                    entity_key=state.get("entity_key"),
                    state_value=state.get("state_value"),
                    constraints=state.get("constraints") or "",
                    created_at=ts, updated_at=ts,
                ))
        log_task_success("StoryboardTool", "save-continuity", {
            "episodeId": episode_id, "dramaId": drama_id, "count": len(states),
            # 去重保序（`[...new Set(...)]`）
            "types": ",".join(dict.fromkeys(str(s.get("state_type")) for s in states)),
        })
        return {"message": f"Saved {len(states)} continuity states", "count": len(states)}

    async def save_storyboards(arguments: dict[str, Any]) -> dict[str, Any]:
        payload_storyboards = arguments.get("storyboards") or []
        ts = now()
        log_task_progress("StoryboardTool", "save-begin", {
            "episodeId": episode_id, "dramaId": drama_id,
            "count": len(payload_storyboards),
            "shotNumbers": ",".join(str(sb.get("shot_number")) for sb in payload_storyboards),
        })

        total_duration = 0
        dialogue_issue_count = 0
        with engine.begin() as conn:
            # 整集重建：先删光旧分镜的**角色关联**，再删分镜本身（⚠️ 物品关联不删）
            existing_ids = [
                row[0] for row in conn.execute(
                    select(storyboards.c.id).where(storyboards.c.episode_id == episode_id)
                ).all()
            ]
            for storyboard_id in existing_ids:
                conn.execute(sql_delete(storyboard_characters).where(
                    storyboard_characters.c.storyboard_id == storyboard_id
                ))
            conn.execute(sql_delete(storyboards).where(storyboards.c.episode_id == episode_id))

            # 本集角色名单 + 「角色名 → speaker_id」映射
            episode_char_names: set[str] = set()
            name_to_speaker: dict[str, str] = {}
            for link in conn.execute(
                select(episode_characters.c.character_id)
                .where(episode_characters.c.episode_id == episode_id)
            ).all():
                char = conn.execute(
                    select(characters).where(and_(
                        characters.c.id == link[0],
                        characters.c.deleted_at.is_(None),
                    ))
                ).first()
                if char is not None and char.name:
                    episode_char_names.add(char.name)
                    if char.speaker_id:
                        name_to_speaker[char.name] = char.speaker_id

            for shot in payload_storyboards:
                validate_storyboard_bindings(
                    conn, episode_id, shot.get("scene_id"), shot.get("character_ids")
                )

                # 说话人绑定闭环：speaker_id 未填时从对白自动解析并绑定
                resolved_speaker_id = shot.get("speaker_id") or ""
                if not resolved_speaker_id and shot.get("dialogue"):
                    speaker_name = extract_speaker_name(shot.get("dialogue"))
                    if speaker_name:
                        resolved_speaker_id = name_to_speaker.get(speaker_name) or ""
                        if resolved_speaker_id:
                            log_task_progress("StoryboardTool", "speaker-auto-bind", {
                                "shotNumber": shot.get("shot_number"),
                                "speakerName": speaker_name,
                                "speakerId": resolved_speaker_id,
                            })

                # 对白一致性校验（只告警不改数据）
                if shot.get("dialogue"):
                    shot_char_names: set[str] = set()
                    for character_id in shot.get("character_ids") or []:
                        char = conn.execute(
                            select(characters.c.name).where(and_(
                                characters.c.id == character_id,
                                characters.c.deleted_at.is_(None),
                            ))
                        ).first()
                        if char is not None and char[0]:
                            shot_char_names.add(char[0])
                    for match in _DIALOGUE_LINE.finditer(shot["dialogue"]):
                        name = match.group(1).strip()
                        if _NARRATOR_NAMES.match(name):
                            continue
                        if name not in shot_char_names and name not in episode_char_names:
                            dialogue_issue_count += 1
                            log_task_progress("StoryboardTool", "dialogue-warn", {
                                "shotNumber": shot.get("shot_number"),
                                "unknownCharacter": name,
                                "availableNames": ",".join(
                                    [*shot_char_names, *episode_char_names]
                                ),
                            })

                duration = shot.get("duration") or _DEFAULT_DURATION
                storyboard_id = int(conn.execute(storyboards.insert().values(
                    episode_id=episode_id,
                    storyboard_number=shot.get("shot_number"),
                    title=shot.get("title"),
                    shot_type=shot.get("shot_type"),
                    angle=shot.get("angle"),
                    movement=shot.get("movement"),
                    location=shot.get("location"),
                    time=shot.get("time"),
                    action=shot.get("action"),
                    dialogue=shot.get("dialogue"),
                    description=shot.get("description"),
                    result=shot.get("result"),
                    atmosphere=shot.get("atmosphere"),
                    image_prompt=shot.get("image_prompt"),
                    first_frame_prompt=shot.get("first_frame_prompt"),
                    last_frame_prompt=shot.get("last_frame_prompt"),
                    video_prompt=shot.get("video_prompt"),
                    negative_prompt=shot.get("negative_prompt"),
                    bgm_prompt=shot.get("bgm_prompt"),
                    sound_effect=shot.get("sound_effect"),
                    scene_id=shot.get("scene_id"),
                    duration=duration,
                    scene_type=shot.get("scene_type"),
                    speaker_id=resolved_speaker_id,
                    start_state=shot.get("start_state"),
                    end_state=shot.get("end_state"),
                    constraints=shot.get("constraints"),
                    transition_motive=shot.get("transition_motive"),
                    keyframe_prompt=shot.get("keyframe_prompt"),
                    created_at=ts, updated_at=ts,
                )).lastrowid)

                # 关联用现成 helper（语义与 TS 局部实现一致，另支持 costume 映射）
                sync_storyboard_characters(conn, storyboard_id, shot.get("character_ids") or [])
                sync_storyboard_props(conn, storyboard_id, shot.get("prop_ids") or [])
                total_duration += duration

            # 集时长：向上取整到分钟（`Math.ceil(total/60)`）
            conn.execute(episodes.update().where(episodes.c.id == episode_id).values(
                duration=math.ceil(total_duration / 60), updated_at=ts
            ))

            # 三个后置：剧本指纹盖章 / take 预算重置 / 节奏相位
            stamp_storyboards_script_hash(conn, episode_id)
            reset_take_budget_for_episode(conn, episode_id)
            assign_rhythm_phases(episode_id)  # 占位（rhythm-phase.ts 未迁）

        log_task_success("StoryboardTool", "save-complete", {
            "episodeId": episode_id, "count": len(payload_storyboards),
            "totalDuration": total_duration, "dialogueIssues": dialogue_issue_count,
        })
        suffix = (f" ({dialogue_issue_count} dialogue character mismatches detected)"
                  if dialogue_issue_count else "")
        return {
            "message": f"Saved {len(payload_storyboards)} storyboards{suffix}",
            "count": len(payload_storyboards),
            "total_duration": total_duration,
            "dialogue_issues": dialogue_issue_count,
        }

    async def update_storyboard(arguments: dict[str, Any]) -> dict[str, Any]:
        storyboard_id = arguments.get("storyboard_id")
        fields = {key: value for key, value in arguments.items() if key != "storyboard_id"}

        with engine.begin() as conn:
            storyboard = conn.execute(
                select(storyboards).where(storyboards.c.id == storyboard_id)
            ).first()
            if storyboard is None:
                return {"error": f"Storyboard {storyboard_id} not found"}

            log_task_progress("StoryboardTool", "update-begin", {
                "episodeId": episode_id, "storyboardId": storyboard_id,
                "fields": list(fields),
            })

            # 绑定校验：显式传了就用传的，没传就用**库里已有的**
            if "character_ids" in fields:
                character_ids: list[Any] | None = fields.get("character_ids")
            else:
                character_ids = [
                    row[0] for row in conn.execute(
                        select(storyboard_characters.c.character_id)
                        .where(storyboard_characters.c.storyboard_id == storyboard_id)
                    ).all()
                ]
            validate_storyboard_bindings(
                conn, episode_id,
                fields["scene_id"] if "scene_id" in fields else storyboard.scene_id,
                character_ids,
            )

            # ⚠️ **按字段是否出现**决定是否写：传 null 会真的清空该列
            mapping = (
                ("title", "title"), ("shot_type", "shot_type"), ("angle", "angle"),
                ("movement", "movement"), ("location", "location"), ("time", "time"),
                ("action", "action"), ("result", "result"), ("atmosphere", "atmosphere"),
                ("image_prompt", "image_prompt"),
                ("first_frame_prompt", "first_frame_prompt"),
                ("last_frame_prompt", "last_frame_prompt"),
                ("video_prompt", "video_prompt"), ("negative_prompt", "negative_prompt"),
                ("bgm_prompt", "bgm_prompt"), ("sound_effect", "sound_effect"),
                ("description", "description"), ("dialogue", "dialogue"),
                ("scene_id", "scene_id"), ("duration", "duration"),
                ("scene_type", "scene_type"), ("speaker_id", "speaker_id"),
                ("start_state", "start_state"), ("end_state", "end_state"),
                ("constraints", "constraints"),
                ("transition_motive", "transition_motive"),
                ("keyframe_prompt", "keyframe_prompt"),
            )
            values: dict[str, Any] = {"updated_at": now()}
            for field, column in mapping:
                if field in fields:
                    values[column] = fields[field]
            conn.execute(
                update(storyboards).where(storyboards.c.id == storyboard_id).values(**values)
            )
            if "character_ids" in fields:
                sync_storyboard_characters(conn, storyboard_id, fields.get("character_ids") or [])

        log_task_success("StoryboardTool", "update-complete", {
            "episodeId": episode_id, "storyboardId": storyboard_id,
            "updatedFields": list(values),
            "characterIds": (",".join(str(x) for x in fields["character_ids"])
                             if "character_ids" in fields and fields.get("character_ids") else None),
        })
        return {"message": f"Storyboard {storyboard_id} updated"}

    # ── 入参 schema（JSON Schema；只列出 zod 的必填项）──────────────────
    nullable_number = {"type": ["number", "null"]}
    nullable_string = {"type": ["string", "null"]}

    storyboard_item = object_schema(
        {
            "shot_number": json_number(),
            "title": json_string(), "shot_type": json_string(), "angle": json_string(),
            "movement": json_string(), "location": json_string(), "time": json_string(),
            "action": json_string(), "dialogue": json_string(), "description": json_string(),
            "result": json_string(), "atmosphere": json_string(),
            "image_prompt": json_string(), "first_frame_prompt": json_string(),
            "last_frame_prompt": json_string(), "video_prompt": json_string(),
            "negative_prompt": json_string(), "bgm_prompt": json_string(),
            "sound_effect": json_string(), "duration": json_number(),
            "scene_id": nullable_number, "character_ids": array_of({"type": "number"}),
            "scene_type": json_string(), "speaker_id": json_string(),
            "start_state": json_string(), "end_state": json_string(),
            "constraints": json_string(), "transition_motive": json_string(),
            # 关键帧（中段）：锁定动作/道具/机位的中间状态，供视频生成参考
            "keyframe_prompt": json_string(),
            "prop_ids": array_of({"type": "number"}),
        },
        required=["shot_number"],
    )

    update_item = object_schema(
        {
            "storyboard_id": json_number(),
            "title": json_string(), "shot_type": json_string(), "angle": json_string(),
            "movement": json_string(), "location": json_string(), "time": json_string(),
            "action": json_string(), "result": json_string(), "atmosphere": json_string(),
            "image_prompt": json_string(), "first_frame_prompt": json_string(),
            "last_frame_prompt": json_string(), "video_prompt": json_string(),
            "negative_prompt": json_string(), "bgm_prompt": json_string(),
            "sound_effect": json_string(), "description": json_string(),
            "dialogue": json_string(), "scene_id": nullable_number,
            "character_ids": array_of({"type": "number"}), "duration": json_number(),
            "scene_type": nullable_string, "speaker_id": nullable_string,
            "start_state": json_string(), "end_state": json_string(),
            "constraints": json_string(), "transition_motive": json_string(),
        },
        required=["storyboard_id"],
    )

    return {
        "read_storyboard_context": Tool(
            id="read_storyboard_context",
            description="Read the screenplay, characters, and scenes for storyboard breakdown.",
            input_schema=object_schema({}, required=[]),
            execute=read_storyboard_context,
        ),
        "save_storyboards": Tool(
            id="save_storyboards",
            description=(
                "Save generated storyboards. Replaces all existing storyboards for this episode."
            ),
            input_schema=object_schema({"storyboards": array_of(storyboard_item)}),
            execute=save_storyboards,
        ),
        "update_storyboard": Tool(
            id="update_storyboard",
            description="Update a specific storyboard shot.",
            input_schema=update_item,
            execute=update_storyboard,
        ),
        "save_continuity_states": Tool(
            id="save_continuity_states",
            description=(
                "Save/replace the persistent continuity states (scene space layout, character pose, "
                "prop state, clue reveal) for this episode. Call after save_storyboards to lock "
                "cross-shot consistency. Replaces all previous states (idempotent)."
            ),
            input_schema=object_schema({"states": array_of(object_schema(
                {
                    "state_type": json_string(),
                    "entity_key": json_string(),
                    "state_value": json_string(),
                    "constraints": json_string(),
                    "storyboard_id": nullable_number,
                    "scene_id": nullable_number,
                },
                required=["state_type", "entity_key", "state_value"],
            ))}),
            execute=save_continuity_states,
        ),
    }
