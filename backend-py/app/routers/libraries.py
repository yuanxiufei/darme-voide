"""四个资源库的**规格声明**（角色 / 场景 / 兵器 / 服装）—— 与 Node 侧四个文件对齐。

实现全在 ``app/services/resource_library.py``（规格驱动）；本文件只放差异点与
apply / from-* 的去向，便于与 TS 逐个字段对照。

⚠️ 这四处用的信封与项目其余部分**不同**：成功 ``code: 0``、错误 ``code: 400/404/500``
且 **HTTP 恒 200**。详见 ``resource_library`` 模块头。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.responses import JSONResponse
from sqlalchemy.engine import Connection

from ..response import now
from ..services.resource_library import (
    LibrarySpec,
    build_library_router,
    js_parse_int,
    lib_err,
    lib_ok,
    q_all,
    q_one,
    q_run,
    safe_stringify,
)

# ===== 前端下拉的兜底枚举（原 TS 常量，仅 weapon/costume 的 filter-options 用）=====
COSTUME_STYLES = ["古风", "现代", "仙侠", "武侠", "科幻", "宫廷", "民俗", "其他"]
BODY_PARTS = ["全身", "上衣", "下装", "外套", "鞋履", "头饰", "配饰"]
SEASONS = ["春", "夏", "秋", "冬", "通用"]
WEAPON_CATEGORIES = ["剑", "刀", "枪", "棍", "斧", "锤", "弓", "弩", "扇", "鞭", "杖", "暗器", "法宝", "其他"]
WEAPON_TYPES = ["近战", "远程", "暗器", "法宝"]
WEAPON_RANKS = ["凡品", "灵品", "仙品", "神品"]


def _distinct(conn: Connection, table: str, column: str, *, not_null: bool = False) -> list[Any]:
    """``SELECT DISTINCT <col> ... ORDER BY <col>`` 的公共实现。"""
    cond = f" AND {column} IS NOT NULL" if not_null else ""
    rows = q_all(
        conn,
        f"SELECT DISTINCT {column} FROM {table} WHERE deleted_at IS NULL{cond} ORDER BY {column}",
    )
    return [r[column] for r in rows]


def _pick_index(arr: list[Any], raw: Any) -> dict[str, Any]:
    """对齐 JS 的 ``arr[Number(raw)] || {}``：非法下标得到 ``{}``（而不是抛错）。"""
    if raw is None:
        idx = 0
    else:
        try:
            idx = int(float(raw))
        except (TypeError, ValueError):
            idx = -1
    if idx < 0 or idx >= len(arr):
        return {}
    item = arr[idx]
    return item if isinstance(item, dict) else {}


# ---------------------------------------------------------------------------
# apply / from-* 处理函数
# ---------------------------------------------------------------------------

def character_apply(conn: Connection, item_id: str, body: dict[str, Any]) -> JSONResponse:
    drama_id = body.get("dramaId")
    if not drama_id:
        return lib_err(400, "请提供目标剧组ID")
    tpl = q_one(
        conn,
        "SELECT * FROM character_templates WHERE id = ? AND deleted_at IS NULL",
        (js_parse_int(item_id),),
    )
    if tpl is None:
        return lib_err(404, "角色模板不存在")
    drama = q_one(
        conn, "SELECT id FROM dramas WHERE id = ? AND deleted_at IS NULL", (drama_id,)
    )
    if drama is None:
        return lib_err(404, "剧组不存在")

    t = now()
    result = q_run(
        conn,
        "INSERT INTO characters (drama_id, name, role, description, appearance, personality,"
        " voice_style, voice_provider, image_url, reference_images, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            drama_id, tpl["name"], tpl["category"] or "", tpl["description"] or "",
            tpl["appearance"], tpl["personality"] or "", tpl["voice_style"] or "",
            tpl["voice_provider"] or "", tpl["image_url"] or "", tpl["reference_images"] or "", t, t,
        ),
    )
    q_run(
        conn,
        "UPDATE character_templates SET usage_count = usage_count + 1 WHERE id = ?",
        (tpl["id"],),
    )
    return lib_ok({"characterId": result.lastrowid}, f'角色 "{tpl["name"]}" 已应用到剧组')


def scene_apply(conn: Connection, item_id: str, body: dict[str, Any]) -> JSONResponse:
    drama_id = body.get("dramaId")
    if not drama_id:
        return lib_err(400, "请提供目标剧组ID")
    tpl = q_one(
        conn,
        "SELECT * FROM scene_templates WHERE id = ? AND deleted_at IS NULL",
        (js_parse_int(item_id),),
    )
    if tpl is None:
        return lib_err(404, "场景模板不存在")

    t = now()
    # prompt 缺省是「地点，氛围，光线」三段的拼接
    prompt = tpl["prompt"] or f"{tpl['location']}，{tpl['atmosphere']}，{tpl['lighting']}"
    result = q_run(
        conn,
        "INSERT INTO scenes (drama_id, episode_id, location, time, prompt, status,"
        " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            drama_id, body.get("episodeId") or None, tpl["location"] or tpl["name"],
            tpl["time_of_day"] or "白天", prompt, "pending", t, t,
        ),
    )
    q_run(
        conn,
        "UPDATE scene_templates SET usage_count = usage_count + 1 WHERE id = ?",
        (tpl["id"],),
    )
    return lib_ok({"sceneId": result.lastrowid}, f'场景 "{tpl["name"]}" 已应用到剧组')


def character_from_character(
    conn: Connection, source_id: str, _body: dict[str, Any]
) -> JSONResponse:
    char = q_one(
        conn,
        "SELECT * FROM characters WHERE id = ? AND deleted_at IS NULL",
        (js_parse_int(source_id),),
    )
    if char is None:
        return lib_err(404, "角色不存在")

    t = now()
    result = q_run(
        conn,
        "INSERT INTO character_templates (name, category, description, appearance, personality,"
        " clothing_style, image_url, reference_images, voice_style, voice_provider,"
        " source_drama_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            char["name"], char["role"] or "通用", char["description"] or "", char["appearance"] or "",
            char["personality"] or "", "", char["image_url"] or "", char["reference_images"] or "",
            char["voice_style"] or "", char["voice_provider"] or "", char["drama_id"], t, t,
        ),
    )
    return lib_ok({"templateId": result.lastrowid}, f'角色 "{char["name"]}" 已保存到角色库')


def scene_from_scene(conn: Connection, source_id: str, _body: dict[str, Any]) -> JSONResponse:
    sc = q_one(
        conn,
        "SELECT * FROM scenes WHERE id = ? AND deleted_at IS NULL",
        (js_parse_int(source_id),),
    )
    if sc is None:
        return lib_err(404, "场景不存在")

    t = now()
    result = q_run(
        conn,
        "INSERT INTO scene_templates (name, category, location, time_of_day, prompt, image_url,"
        " source_drama_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (sc["location"] or "未命名场景", "通用", sc["location"], sc["time"], sc["prompt"],
         sc["image_url"] or "", sc["drama_id"], t, t),
    )
    return lib_ok({"templateId": result.lastrowid}, "场景已保存到场景库")


def weapon_from_character(
    conn: Connection, source_id: str, body: dict[str, Any]
) -> JSONResponse:
    """从角色 ``weapons`` 保存到兵器库（该字段可能是 JSON 数组，也可能是纯文本）。"""
    char = q_one(
        conn,
        "SELECT * FROM characters WHERE id = ? AND deleted_at IS NULL",
        (js_parse_int(source_id),),
    )
    if char is None:
        return lib_err(404, "角色不存在")

    weapons: list[Any] = []
    raw_text = ""
    raw_value = char["weapons"]
    if raw_value:
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                weapons = parsed
            else:
                raw_text = str(raw_value)
        except (ValueError, TypeError):
            raw_text = str(raw_value)

    w = _pick_index(weapons, body.get("weaponIndex"))
    # TS: body.name || w.name || (rawText && weapons.length === 0 ? rawText.slice(0,40) : '') || '未命名武器'
    name = (
        body.get("name")
        or w.get("name")
        or ((raw_text[:40] if raw_text else "") if not weapons else "")
        or "未命名武器"
    )

    t = now()
    result = q_run(
        conn,
        "INSERT INTO weapon_templates (name, category, type, description, appearance, material,"
        " attributes, rank, owner_character_name, image_url, reference_images, tags, metadata,"
        " source_drama_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            name, body.get("category") or w.get("category") or "其他",
            body.get("type") or w.get("type") or "",
            body.get("description") or w.get("description") or "",
            body.get("appearance") or w.get("appearance") or "",
            body.get("material") or w.get("material") or "",
            safe_stringify(body.get("attributes") or w.get("attributes")),
            body.get("rank") or w.get("rank") or "", char["name"],
            body.get("imageUrl") or w.get("imageUrl") or "",
            safe_stringify(body.get("referenceImages") or w.get("referenceImages")),
            safe_stringify(body.get("tags") or w.get("tags")),
            safe_stringify(body.get("metadata") or w.get("metadata")),
            char["drama_id"], t, t,
        ),
    )
    return lib_ok({"id": result.lastrowid}, f'武器 "{name}" 已保存到武器库')


def costume_from_character(
    conn: Connection, source_id: str, body: dict[str, Any]
) -> JSONResponse:
    """从角色的 ``costumes``（JSON 数组，元素含 name/imageUrl）按 ``costumeIndex`` 保存到服装库。"""
    char = q_one(
        conn,
        "SELECT * FROM characters WHERE id = ? AND deleted_at IS NULL",
        (js_parse_int(source_id),),
    )
    if char is None:
        return lib_err(404, "角色不存在")

    costumes: list[Any] = []
    try:
        parsed = json.loads(char["costumes"] or "")
        if isinstance(parsed, list):
            costumes = parsed
    except (ValueError, TypeError):
        costumes = []

    cst = _pick_index(costumes, body.get("costumeIndex"))
    # TS: body.name || cst.name || (character.clothing ? String(clothing).slice(0,40) : '') || '未命名服装'
    clothing_snippet = str(char["clothing"])[:40] if char["clothing"] else ""
    name = body.get("name") or cst.get("name") or clothing_snippet or "未命名服装"

    t = now()
    result = q_run(
        conn,
        "INSERT INTO costume_templates (name, category, description, style, body_part, material,"
        " color_scheme, season, appearance, image_url, reference_images, tags, metadata,"
        " source_drama_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            name, body.get("category") or "通用",
            body.get("description") or cst.get("description") or char["clothing"] or "",
            body.get("style") or cst.get("style") or "",
            body.get("bodyPart") or "",
            body.get("material") or cst.get("material") or "",
            body.get("colorScheme") or cst.get("colorScheme") or "",
            body.get("season") or cst.get("season") or "",
            body.get("appearance") or cst.get("appearance") or "",
            body.get("imageUrl") or cst.get("imageUrl") or char["image_url"] or "",
            safe_stringify(body.get("referenceImages") or cst.get("referenceImages")),
            safe_stringify(body.get("tags") or cst.get("tags")),
            safe_stringify(body.get("metadata") or cst.get("metadata")),
            char["drama_id"], t, t,
        ),
    )
    return lib_ok({"id": result.lastrowid}, f'服装 "{name}" 已保存到服装库')


# ---------------------------------------------------------------------------
# 四个规格
# ---------------------------------------------------------------------------

CHARACTER_LIBRARY = LibrarySpec(
    path="/api/v1/character-library",
    table="character_templates",
    label="角色模板",
    search_columns=("name", "description", "tags", "appearance", "personality", "voice_style"),
    allowed_sorts=("name", "category", "usage_count", "updated_at", "created_at"),
    columns=("name", "category", "description", "appearance", "personality", "clothing_style",
             "expression", "gender", "age_group", "image_url", "reference_images", "voice_style",
             "voice_provider", "voice_config", "tags", "metadata", "source_drama_id"),
    json_columns=("reference_images", "voice_config", "tags", "metadata"),
    required=("name", "appearance"),
    required_message="名称和外貌描述为必填项",
    create_defaults={"category": "通用"},
    out_aliases=("voice_config", "reference_images"),
    apply=character_apply,
    from_route=("/from-character/{source_id}", character_from_character),
)

SCENE_LIBRARY = LibrarySpec(
    path="/api/v1/scene-library",
    table="scene_templates",
    label="场景模板",
    search_columns=("name", "description", "tags", "location", "atmosphere", "lighting", "prompt"),
    allowed_sorts=("name", "category", "time_of_day", "style", "usage_count", "updated_at", "created_at"),
    columns=("name", "category", "description", "location", "atmosphere", "lighting", "time_of_day",
             "style", "season", "weather", "image_url", "reference_images", "prompt", "tags",
             "metadata", "source_drama_id"),
    json_columns=("reference_images", "tags", "metadata"),
    required=("name",),
    exact_filters=(("timeOfDay", "time_of_day"), ("style", "style")),
    create_defaults={"category": "通用"},
    out_aliases=("reference_images",),
    filter_options=lambda conn: {
        "timeOfDay": _distinct(conn, "scene_templates", "time_of_day", not_null=True),
        "styles": _distinct(conn, "scene_templates", "style", not_null=True),
    },
    apply=scene_apply,
    from_route=("/from-scene/{source_id}", scene_from_scene),
)

WEAPON_LIBRARY = LibrarySpec(
    path="/api/v1/weapon-library",
    table="weapon_templates",
    label="兵器模板",
    search_columns=("name", "description", "tags", "appearance", "material", "attributes"),
    allowed_sorts=("name", "category", "type", "rank", "usage_count", "updated_at", "created_at"),
    columns=("name", "category", "type", "description", "appearance", "material", "attributes",
             "rank", "owner_character_name", "image_url", "reference_images", "tags", "metadata",
             "source_drama_id"),
    json_columns=("attributes", "reference_images", "tags", "metadata"),
    required=("name",),
    exact_filters=(("type", "type"), ("rank", "rank")),
    create_defaults={"category": "剑"},
    out_aliases=("reference_images",),
    filter_options=lambda conn: {
        "categories": _distinct(conn, "weapon_templates", "category", not_null=True) or WEAPON_CATEGORIES,
        "types": _distinct(conn, "weapon_templates", "type", not_null=True) or WEAPON_TYPES,
        "ranks": _distinct(conn, "weapon_templates", "rank", not_null=True) or WEAPON_RANKS,
    },
    categories_fallback=tuple(WEAPON_CATEGORIES),
    from_route=("/from-character/{source_id}", weapon_from_character),
)

COSTUME_LIBRARY = LibrarySpec(
    path="/api/v1/costume-library",
    table="costume_templates",
    label="服装模板",
    search_columns=("name", "description", "tags", "appearance", "material", "color_scheme"),
    allowed_sorts=("name", "category", "style", "body_part", "season", "usage_count", "updated_at",
                   "created_at"),
    columns=("name", "category", "description", "style", "body_part", "material", "color_scheme",
             "season", "appearance", "image_url", "reference_images", "tags", "metadata",
             "source_drama_id"),
    json_columns=("reference_images", "tags", "metadata"),
    required=("name",),
    exact_filters=(("style", "style"), ("bodyPart", "body_part")),
    create_defaults={"category": "通用"},
    out_aliases=("reference_images",),
    filter_options=lambda conn: {
        "styles": _distinct(conn, "costume_templates", "style", not_null=True) or COSTUME_STYLES,
        "bodyParts": _distinct(conn, "costume_templates", "body_part", not_null=True) or BODY_PARTS,
        "seasons": _distinct(conn, "costume_templates", "season", not_null=True) or SEASONS,
    },
    from_route=("/from-character/{source_id}", costume_from_character),
)

character_library_router = build_library_router(CHARACTER_LIBRARY)
scene_library_router = build_library_router(SCENE_LIBRARY)
weapon_library_router = build_library_router(WEAPON_LIBRARY)
costume_library_router = build_library_router(COSTUME_LIBRARY)
