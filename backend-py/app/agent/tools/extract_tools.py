"""角色/场景/物品提取工具（移植自 ``agents/tools/extract-tools.ts``，445 行）。

**单 Agent 一步流程**：读剧本 → 读已有角色/场景（供去重）→ 提取并**智能去重后直接保存**。

────────── 移植进度（本文件）──────────
✅ 关联辅助：``_link_char_to_episode`` / ``_link_scene_to_episode`` / ``_link_episode_to_prop``
✅ ``_infer_role_type``：自由文本 role → 标准化角色类型（**顺序敏感**的判据链）
✅ ``read_script_for_extraction``（⚠️ 取值链是 ``scriptContent || content``，与剧本工具**相反**）
✅ ``read_existing_characters`` / ``read_existing_scenes``（返回**整行**供去重）
✅ ``read_existing_props``（关联表反查 prop_templates，过滤软删）
✅ ``save_dedup_characters`` / ``save_dedup_scenes`` / ``save_dedup_props``（去重合并保存）

⚠️ 三处**必须照抄**的细节（保存工具已实现，这三条仍是判定依据）：

* ``core_features`` / ``costumes`` 落库要**紧凑 JSON**（与 Node 共用该列）；
* 场景去重键是 ``location + (time || '')``（同地点**不同时段**算新场景）；
* 物品保存时 ``negative_prompt`` 兜底是**空串**（不是构建出来的负面词），``keyClue`` 新增兜底 ``'否'``。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import and_, select, update  # `update` 供去重合并保存使用（见 save_dedup_*）
from sqlalchemy.engine import Connection

from app.core.db import engine
from app.core.models import (
    app_settings,
    characters,
    dramas,
    episode_characters,
    episode_props,
    episode_scenes,
    episodes,
    prop_templates,
    scenes,
)
from app.core.response import now, row_to_camel
from app.services.prompt_utils import (  # noqa: F401  (下批保存工具的提示词兜底要用)
    SCENE_IMAGE_NEGATIVE,
    build_character_image_prompt,
    build_character_negative_prompt,
    build_prop_image_prompt,
    build_scene_image_prompt,
)
from app.services.task_logger import log_task_progress, log_task_success  # noqa: F401
from app.services.text_slice import slice_long_text
from app.agent.tool import Tool, array_of, json_string, object_schema

__all__ = ["create_extract_tools", "infer_role_type"]

#: 角色类型判据（**顺序敏感**：主角 → 反派 → 龙套 → 配角 → 旁白）
_ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"主角|男主|女主|hero|protagonist|lead", "主角"),
    (r"反派|坏|恶|villain|antagonist|boss", "反派"),
    (r"龙套|路人", "龙套"),
    (r"配角|supporting|side", "配角"),
    (r"旁白|叙述|narrator|画外音", "旁白"),
)


def infer_role_type(role: str | None) -> str:
    """从自由文本 role 推断标准化角色类型（无法判断且**非空**时是「其他」）。"""
    if not role:
        return ""
    import re

    for pattern, role_type in _ROLE_PATTERNS:
        if re.search(pattern, role, re.IGNORECASE):
            return role_type
    return "其他"


# ─── 关联辅助（幂等：已关联就不重复插）────────────────────────────


def _link_char_to_episode(conn: Connection, episode_id: int, character_id: int) -> None:
    existing = conn.execute(
        select(episode_characters.c.id).where(and_(
            episode_characters.c.episode_id == episode_id,
            episode_characters.c.character_id == character_id,
        ))
    ).first()
    if existing is None:
        conn.execute(episode_characters.insert().values(
            episode_id=episode_id, character_id=character_id, created_at=now()
        ))


def _link_scene_to_episode(conn: Connection, episode_id: int, scene_id: int) -> None:
    existing = conn.execute(
        select(episode_scenes.c.id).where(and_(
            episode_scenes.c.episode_id == episode_id,
            episode_scenes.c.scene_id == scene_id,
        ))
    ).first()
    if existing is None:
        conn.execute(episode_scenes.insert().values(
            episode_id=episode_id, scene_id=scene_id, created_at=now()
        ))


def _link_episode_to_prop(conn: Connection, episode_id: int, prop_id: int) -> None:
    # ⚠️ 物品关联**不写 created_at**（原 TS 只插两列）
    # ⚠️ 且 `episode_props` 是**复合主键**（无 `id` 列，另两张关联表才有）⇒ 按 prop_id 查
    existing = conn.execute(
        select(episode_props.c.prop_id).where(and_(
            episode_props.c.episode_id == episode_id,
            episode_props.c.prop_id == prop_id,
        ))
    ).first()
    if existing is None:
        conn.execute(episode_props.insert().values(episode_id=episode_id, prop_id=prop_id))


def create_extract_tools(episode_id: int, drama_id: int) -> dict[str, Tool]:
    """工厂（闭包注入 id；每个工具自开短事务）。"""

    async def read_script_for_extraction(_arguments: dict[str, Any]) -> dict[str, Any]:
        with engine.begin() as conn:
            row = conn.execute(
                select(episodes.c.content, episodes.c.script_content)
                .where(episodes.c.id == episode_id)
            ).first()
        if row is None:
            return {"error": "Episode not found"}
        # ⚠️ 这里是 **scriptContent 优先**（与 script_tools 的 `content || scriptContent` 相反）
        content = row[1] or row[0] or ""
        if not content:
            return {"error": "Episode has no script content"}
        sliced = slice_long_text(content)
        log_task_success("ExtractTool", "read-script", {
            "episodeId": episode_id, "dramaId": drama_id, "scriptLength": len(content),
        })
        return {
            "script": sliced["text"],
            "truncated": sliced["truncated"],
            "total_chars": sliced["total_chars"],
        }

    async def read_existing_characters(_arguments: dict[str, Any]) -> dict[str, Any]:
        with engine.begin() as conn:
            linked_ids = {
                row[0] for row in conn.execute(
                    select(episode_characters.c.character_id)
                    .where(episode_characters.c.episode_id == episode_id)
                ).all()
            }
            rows = conn.execute(
                select(characters).where(and_(
                    characters.c.drama_id == drama_id,
                    characters.c.deleted_at.is_(None),
                ))
            ).all()
        payload = {
            "count": len(rows),
            # 原 TS 直接返回整行（drizzle 行 = camelCase）
            "characters": [row_to_camel(row, "characters") for row in rows],
            "current_episode_characters": [
                row_to_camel(row, "characters") for row in rows if row.id in linked_ids
            ],
        }
        log_task_success("ExtractTool", "read-characters", {
            "episodeId": episode_id, "dramaId": drama_id,
            "projectCharacters": payload["count"],
            "episodeCharacters": len(payload["current_episode_characters"]),
        })
        return payload

    async def read_existing_scenes(_arguments: dict[str, Any]) -> dict[str, Any]:
        with engine.begin() as conn:
            linked_ids = {
                row[0] for row in conn.execute(
                    select(episode_scenes.c.scene_id)
                    .where(episode_scenes.c.episode_id == episode_id)
                ).all()
            }
            rows = conn.execute(
                select(scenes).where(and_(
                    scenes.c.drama_id == drama_id,
                    scenes.c.deleted_at.is_(None),
                ))
            ).all()
        payload = {
            "count": len(rows),
            "scenes": [row_to_camel(row, "scenes") for row in rows],
            "current_episode_scenes": [
                row_to_camel(row, "scenes") for row in rows if row.id in linked_ids
            ],
        }
        log_task_success("ExtractTool", "read-scenes", {
            "episodeId": episode_id, "dramaId": drama_id,
            "projectScenes": payload["count"],
            "episodeScenes": len(payload["current_episode_scenes"]),
        })
        return payload

    async def read_existing_props(_arguments: dict[str, Any]) -> dict[str, Any]:
        with engine.begin() as conn:
            link_rows = conn.execute(
                select(episode_props.c.prop_id)
                .where(episode_props.c.episode_id == episode_id)
            ).all()
            if not link_rows:
                return {"props": []}
            props = []
            for link in link_rows:
                row = conn.execute(
                    select(prop_templates).where(prop_templates.c.id == link[0])
                ).first()
                if row is None or row.deleted_at:
                    continue
                props.append({
                    "id": row.id,
                    "name": row.name,
                    "category": row.category,
                    "description": row.description,
                    "holder": row.holder,
                    "keyClue": row.key_clue,
                })
        return {"props": props}

    def _compact(value: Any) -> str:
        """与 Node 共用的 JSON 列 ⇒ 紧凑分隔符。"""
        import json

        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _drama_style(conn: Connection) -> str | None:
        """画风：剧集 style → 全局 ``app_settings.art_style``（提取兜底 prompt 与生成链路保持一致）。"""
        drama_row = conn.execute(select(dramas.c.style).where(dramas.c.id == drama_id)).first()
        global_row = conn.execute(
            select(app_settings.c.value).where(app_settings.c.key == "art_style")
        ).first()
        return ((drama_row[0] if drama_row else None)
                or (global_row[0] if global_row else None)
                or None)

    def _live_rows(conn: Connection, table) -> list[Any]:
        """本剧未软删的行（去重基准，每次重新查——与原 TS 一致）。"""
        return conn.execute(
            select(table).where(and_(table.c.drama_id == drama_id, table.c.deleted_at.is_(None)))
        ).all()

    async def save_dedup_characters(arguments: dict[str, Any]) -> dict[str, Any]:
        ts = now()
        payload_chars = arguments.get("characters") or []
        created = 0
        merged_count = 0
        with engine.begin() as conn:
            drama_style = _drama_style(conn)
            log_task_progress("ExtractTool", "save-characters-begin", {
                "episodeId": episode_id, "dramaId": drama_id,
                "names": ",".join(str(c.get("name")) for c in payload_chars),
            })
            for char in payload_chars:
                existing = next(
                    (row for row in _live_rows(conn, characters) if row.name == char.get("name")),
                    None,
                )
                # 合并字段：agent 本次提供的值优先，缺省回退已有数据
                role = char.get("role") or getattr(existing, "role", None) or ""
                merged = {
                    "name": char.get("name"),
                    "role": role,
                    "roleType": char.get("role_type") or infer_role_type(role),
                    "description": char.get("description") or getattr(existing, "description", None) or "",
                    "appearance": char.get("appearance") or getattr(existing, "appearance", None) or "",
                    "personality": char.get("personality") or getattr(existing, "personality", None) or "",
                    "clothing": char.get("clothing") or getattr(existing, "clothing", None) or "",
                    "weapons": char.get("weapons") or getattr(existing, "weapons", None) or "",
                    "accessories": char.get("accessories") or getattr(existing, "accessories", None) or "",
                    "coreFeatures": (_compact(char["core_features"]) if char.get("core_features")
                                     else (getattr(existing, "core_features", None) or "")),
                    "costumes": (_compact(char["costumes"]) if char.get("costumes")
                                 else (getattr(existing, "costumes", None) or "")),
                }
                # 提示词兜底：agent 值 > 已有值 > **基于合并字段自动构建**（不依赖 LLM 必填）
                custom_prompt = (
                    char.get("image_prompt")
                    or getattr(existing, "custom_prompt", None)
                    or build_character_image_prompt({
                        "name": merged["name"],
                        "appearance": merged["appearance"],
                        "description": merged["description"],
                        "personality": merged["personality"],
                        "coreFeatures": merged["coreFeatures"] or None,
                        "clothing": merged["clothing"],
                        "costumes": merged["costumes"] or None,
                        "dramaStyle": drama_style,
                    })
                )
                negative_prompt = (
                    char.get("negative_prompt")
                    or getattr(existing, "negative_prompt", None)
                    or build_character_negative_prompt(drama_style)
                )
                values = {
                    "role": merged["role"], "role_type": merged["roleType"],
                    "description": merged["description"], "appearance": merged["appearance"],
                    "personality": merged["personality"], "clothing": merged["clothing"],
                    "weapons": merged["weapons"], "accessories": merged["accessories"],
                    "core_features": merged["coreFeatures"], "costumes": merged["costumes"],
                    "custom_prompt": custom_prompt, "negative_prompt": negative_prompt,
                    "updated_at": ts,
                }
                if existing is not None:
                    conn.execute(update(characters).where(characters.c.id == existing.id).values(**values))
                    _link_char_to_episode(conn, episode_id, existing.id)
                    merged_count += 1
                else:
                    char_id = int(conn.execute(characters.insert().values(
                        **values, name=merged["name"], drama_id=drama_id, created_at=ts,
                    )).lastrowid)
                    _link_char_to_episode(conn, episode_id, char_id)
                    created += 1

        log_task_success("ExtractTool", "save-characters-complete", {
            "episodeId": episode_id, "created": created, "merged": merged_count,
        })
        return {
            "message": f"角色保存完成：新增 {created}，合并更新 {merged_count}",
            "created": created, "merged": merged_count,
        }

    async def save_dedup_scenes(arguments: dict[str, Any]) -> dict[str, Any]:
        ts = now()
        payload_scenes = arguments.get("scenes") or []
        created = 0
        reused = 0
        with engine.begin() as conn:
            log_task_progress("ExtractTool", "save-scenes-begin", {
                "episodeId": episode_id, "dramaId": drama_id,
                "scenes": ",".join(f"{s.get('location')}@{s.get('time') or ''}" for s in payload_scenes),
            })
            for scene in payload_scenes:
                # 去重键：**地点 + 时间段**（同地点不同时段算新场景）
                existing = next(
                    (row for row in _live_rows(conn, scenes)
                     if row.location == scene.get("location")
                     and row.time == (scene.get("time") or "")),
                    None,
                )
                fallback = build_scene_image_prompt({
                    "location": scene.get("location"),
                    "time": scene.get("time"),
                    "prompt": scene.get("prompt") or scene.get("description"),
                })
                if existing is not None:
                    conn.execute(update(scenes).where(scenes.c.id == existing.id).values(
                        description=scene.get("description") or existing.description,
                        atmosphere=scene.get("atmosphere") or existing.atmosphere,
                        lighting=scene.get("lighting") or existing.lighting,
                        weather=scene.get("weather") or existing.weather,
                        season=scene.get("season") or existing.season,
                        style=scene.get("style") or existing.style,
                        custom_prompt=(scene.get("image_prompt") or existing.custom_prompt or fallback),
                        negative_prompt=(scene.get("negative_prompt") or existing.negative_prompt
                                         or SCENE_IMAGE_NEGATIVE),
                        updated_at=ts,
                    ))
                    _link_scene_to_episode(conn, episode_id, existing.id)
                    reused += 1
                else:
                    scene_id = int(conn.execute(scenes.insert().values(
                        drama_id=drama_id,
                        location=scene.get("location"),
                        time=scene.get("time") or "",
                        # ⚠️ prompt 兜底是 location（不是空串）
                        prompt=scene.get("prompt") or scene.get("location"),
                        description=scene.get("description") or "",
                        atmosphere=scene.get("atmosphere") or "",
                        lighting=scene.get("lighting") or "",
                        weather=scene.get("weather") or "",
                        season=scene.get("season") or "",
                        style=scene.get("style") or "",
                        custom_prompt=scene.get("image_prompt") or fallback,
                        negative_prompt=scene.get("negative_prompt") or SCENE_IMAGE_NEGATIVE,
                        created_at=ts, updated_at=ts,
                    )).lastrowid)
                    _link_scene_to_episode(conn, episode_id, scene_id)
                    created += 1

        log_task_success("ExtractTool", "save-scenes-complete", {
            "episodeId": episode_id, "created": created, "reused": reused,
        })
        return {
            "message": f"场景保存完成：新增 {created}，复用已有 {reused}",
            "created": created, "reused": reused,
        }

    async def save_dedup_props(arguments: dict[str, Any]) -> dict[str, Any]:
        ts = now()
        payload_props = arguments.get("props") or []
        created = 0
        reused = 0
        with engine.begin() as conn:
            log_task_progress("ExtractTool", "save-props-begin", {
                "episodeId": episode_id, "dramaId": drama_id,
                "props": ",".join(str(p.get("name")) for p in payload_props),
            })
            for prop in payload_props:
                existing = next(
                    (row for row in _live_rows(conn, prop_templates) if row.name == prop.get("name")),
                    None,
                )
                fallback = build_prop_image_prompt({
                    "name": prop.get("name"),
                    "category": prop.get("category"),
                    "description": prop.get("description"),
                    "appearance": prop.get("appearance"),
                    "sizeHint": prop.get("size_hint"),
                    "holder": prop.get("holder"),
                })
                if existing is not None:
                    conn.execute(update(prop_templates).where(
                        prop_templates.c.id == existing.id
                    ).values(
                        category=prop.get("category") or existing.category or "道具",
                        description=prop.get("description") or existing.description,
                        appearance=prop.get("appearance") or existing.appearance,
                        size_hint=prop.get("size_hint") or existing.size_hint,
                        holder=prop.get("holder") or existing.holder,
                        key_clue=prop.get("key_clue") or existing.key_clue,
                        # ⚠️ prop_templates 的提示词列叫 image_prompt（scenes 才叫 custom_prompt）
                        image_prompt=(prop.get("image_prompt") or existing.image_prompt or fallback),
                        # ⚠️ 物品的负面词兜底是**空串**（不是构建出来的）
                        negative_prompt=prop.get("negative_prompt") or existing.negative_prompt or "",
                        updated_at=ts,
                    ))
                    _link_episode_to_prop(conn, episode_id, existing.id)
                    reused += 1
                else:
                    prop_id = int(conn.execute(prop_templates.insert().values(
                        drama_id=drama_id,
                        name=prop.get("name"),
                        category=prop.get("category") or "道具",
                        description=prop.get("description") or "",
                        appearance=prop.get("appearance") or "",
                        size_hint=prop.get("size_hint") or "",
                        holder=prop.get("holder") or "",
                        # ⚠️ 新增时 keyClue 兜底 `'否'`
                        key_clue=prop.get("key_clue") or "否",
                        image_prompt=prop.get("image_prompt") or fallback,
                        negative_prompt=prop.get("negative_prompt") or "",
                        created_at=ts, updated_at=ts,
                    )).lastrowid)
                    _link_episode_to_prop(conn, episode_id, prop_id)
                    created += 1

        log_task_success("ExtractTool", "save-props-complete", {
            "episodeId": episode_id, "created": created, "reused": reused,
        })
        return {
            "message": f"物品保存完成：新增 {created}，复用已有 {reused}",
            "created": created, "reused": reused,
        }

    return {
        "read_script_for_extraction": Tool(
            id="read_script_for_extraction",
            description="Read the formatted screenplay for character/scene extraction.",
            input_schema=object_schema({}, required=[]),
            execute=read_script_for_extraction,
        ),
        "read_existing_characters": Tool(
            id="read_existing_characters",
            description="Read all characters already existing in this drama project (for deduplication).",
            input_schema=object_schema({}, required=[]),
            execute=read_existing_characters,
        ),
        "read_existing_scenes": Tool(
            id="read_existing_scenes",
            description="Read all scenes already existing in this drama project (for deduplication).",
            input_schema=object_schema({}, required=[]),
            execute=read_existing_scenes,
        ),
        "read_existing_props": Tool(
            id="read_existing_props",
            description=(
                "Read props (items/clues) already extracted for this episode, "
                "for dedup when saving new props."
            ),
            input_schema=object_schema({}, required=[]),
            execute=read_existing_props,
        ),
        "save_dedup_characters": Tool(
            id="save_dedup_characters",
            description=(
                "Save extracted characters with deduplication. Existing characters (same name) "
                "are merged/updated; new ones are created. All are linked to the current episode."
            ),
            input_schema=object_schema({"characters": array_of(object_schema(
                {
                    "name": json_string(),
                    "role": json_string(),
                    "role_type": json_string(),
                    "description": json_string(),
                    "appearance": json_string(),
                    "personality": json_string(),
                    "clothing": json_string(),
                    "weapons": json_string(),
                    "accessories": json_string(),
                    "core_features": array_of(json_string()),
                    "costumes": array_of(json_string()),
                    "image_prompt": json_string(),
                    "negative_prompt": json_string(),
                },
                required=["name"],
            ))}),
            execute=save_dedup_characters,
        ),
        "save_dedup_scenes": Tool(
            id="save_dedup_scenes",
            description=(
                "Save extracted scenes with deduplication. Existing scenes (same location+time) "
                "are reused; new ones are created. All are linked to the current episode."
            ),
            input_schema=object_schema({"scenes": array_of(object_schema(
                {
                    "location": json_string(),
                    "time": json_string(),
                    "prompt": json_string(),
                    "description": json_string(),
                    "atmosphere": json_string(),
                    "lighting": json_string(),
                    "weather": json_string(),
                    "season": json_string(),
                    "style": json_string(),
                    "image_prompt": json_string(),
                    "negative_prompt": json_string(),
                },
                required=["location"],
            ))}),
            execute=save_dedup_scenes,
        ),
        "save_dedup_props": Tool(
            id="save_dedup_props",
            description=(
                "Save extracted props (items/clues/treasure/props) with deduplication by name "
                "within the drama. All are linked to the current episode."
            ),
            input_schema=object_schema({"props": array_of(object_schema(
                {
                    "name": json_string(),
                    "category": json_string(),
                    "description": json_string(),
                    "appearance": json_string(),
                    "size_hint": json_string(),
                    "holder": json_string(),
                    "key_clue": json_string(),
                    "image_prompt": json_string(),
                    "negative_prompt": json_string(),
                },
                required=["name"],
            ))}),
            execute=save_dedup_props,
        ),
    }
