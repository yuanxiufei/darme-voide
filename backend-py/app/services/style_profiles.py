"""风格 Profile CRUD —— 移植 ``backend/src/services/style-profiles.ts`` 的**非 LLM 部分**。

从参考素材提炼可复用的 house style，分四类规则：``storytelling``（叙事节奏）、
``shot_patterns``（景别/机位/运镜）、``audio_captions``（音效/配乐/字幕）、``qc_rules``（验收标准）。
来源三分类：measurement facts（客观测量）/ visual inference（模型推断）/ user preference（用户偏好）。

⚠️ ``distillStyleProfile``（LLM 分析 + ffprobe 探测）**未移植** —— 它依赖 ``@mastra`` 的 Agent、
``@ai-sdk`` 的 OpenAI provider 与 ``fluent-ffmpeg`` 的 ffprobe，属媒体/Agent 域。
⇒ ``POST /style-profiles/:id/distill`` 不注册（走反代），但 ``/apply`` 已迁：
   用户可以把**在 Node 侧提炼好**的结果贴回来落库，链路不阻塞。

⚠️ 行形状是 **camelCase**（原 TS 用 ``mapRow`` 显式改名，返回的不是 drizzle 原始行）。
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import and_, select, update
from sqlalchemy.engine import Connection

from ..models import style_profiles
from ..response import js_truthy, now, row_to_dict

#: 五个 JSON 列的列名（原 TS 里 ``preferences`` 与其余四个一样走 JSON.stringify）
_JSON_FIELDS = ("storytelling", "shot_patterns", "audio_captions", "qc_rules", "preferences")
_PLAIN_FIELDS = ("name", "description", "source")

#: 内部哨兵：区分「调用方没传这个键」与「传了 null」（对齐 TS 的 ``undefined``）
UNSET: Any = object()


def _dump_json(value: Any) -> str | None:
    """对齐 ``value ? JSON.stringify(value) : null`` —— 注意用 ``js_truthy``，
    显式传入的空 ``{}`` / ``[]`` 在 JS 里是真值，必须落成 ``"{}"`` / ``"[]"`` 而不是 null。"""
    if not js_truthy(value):
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _map_row(row: Any) -> dict[str, Any]:
    """对齐 TS 的 ``mapRow``：列名 → camelCase，``isActive`` 转真布尔。"""
    d = row_to_dict(row)
    return {
        "id": d["id"],
        "dramaId": d["drama_id"],
        "name": d["name"],
        "description": d["description"],
        "source": d["source"],
        "storytelling": d["storytelling"],
        "shotPatterns": d["shot_patterns"],
        "audioCaptions": d["audio_captions"],
        "qcRules": d["qc_rules"],
        "facts": d["facts"],
        "inferences": d["inferences"],
        "preferences": d["preferences"],
        "isActive": bool(d["is_active"]),
        "createdAt": d["created_at"],
        "updatedAt": d["updated_at"],
        "deletedAt": d["deleted_at"],
    }


def list_style_profiles(conn: Connection, drama_id: Any = None) -> list[dict[str, Any]]:
    """列表（激活的排前面）。

    ⚠️ ``drama_id`` 走 JS 真值判断：``0`` / ``NaN`` 都视为「不过滤」
    （原 TS 是 ``dramaId ? eq(...) : undefined``，而 NaN 在 JS 里是假值）。
    """
    conds = [style_profiles.c.deleted_at.is_(None)]
    if js_truthy(drama_id):
        conds.append(style_profiles.c.drama_id == drama_id)

    rows = conn.execute(select(style_profiles).where(and_(*conds))).all()
    mapped = [_map_row(r) for r in rows]
    # TS 的比较器 (a.isActive?-1:1)-(b.isActive?-1:1) 等价于「激活的排前，其余保持原序」
    return sorted(mapped, key=lambda p: 0 if p["isActive"] else 1)


def get_active_profile_for_drama(conn: Connection, drama_id: Any = None) -> dict[str, Any] | None:
    """当前生效的风格 Profile（对齐 TS ``getActiveProfileForDrama``）。

    ⚠️ **两级回退**，顺序不能反：先找**本剧专属**且激活的；找不到再找**全局**
    （``drama_id IS NULL``）且激活的。只挑其中一份，不合并。
    与 ``list_style_profiles`` **不同**（那个是「列全部、激活的排前」，没有全局兜底）。
    """
    if drama_id:
        row = conn.execute(
            select(style_profiles).where(and_(
                style_profiles.c.drama_id == drama_id,
                style_profiles.c.is_active.is_(True),
                style_profiles.c.deleted_at.is_(None),
            ))
        ).first()
        if row is not None:
            return _map_row(row)
    global_row = conn.execute(
        select(style_profiles).where(and_(
            style_profiles.c.drama_id.is_(None),
            style_profiles.c.is_active.is_(True),
            style_profiles.c.deleted_at.is_(None),
        ))
    ).first()
    return _map_row(global_row) if global_row is not None else None


def get_style_profile(conn: Connection, profile_id: Any) -> dict[str, Any] | None:
    if profile_id is None or profile_id != profile_id:  # None / NaN
        return None
    row = conn.execute(
        select(style_profiles).where(
            and_(style_profiles.c.id == profile_id, style_profiles.c.deleted_at.is_(None))
        )
    ).first()
    return _map_row(row) if row is not None else None


def create_style_profile(
    conn: Connection,
    *,
    name: Any = None,
    drama_id: Any = None,
    description: Any = None,
    source: Any = None,
    storytelling: Any = None,
    shot_patterns: Any = None,
    audio_captions: Any = None,
    qc_rules: Any = None,
    preferences: Any = None,
) -> int | None:
    """创建（失败返回 None —— 路由据此回 400 ``create failed``）。

    注：``name`` 缺失时按原 TS 一样不补默认值 ⇒ 触发 NOT NULL 约束 ⇒ 返回 None ⇒ 400，
    与 Node 的可观察行为一致。
    """
    ts = now()
    try:
        result = conn.execute(
            style_profiles.insert().values(
                drama_id=None if drama_id is None else drama_id,
                name=name,
                description=None if description is None else description,
                source=None if source is None else source,
                storytelling=_dump_json(storytelling),
                shot_patterns=_dump_json(shot_patterns),
                audio_captions=_dump_json(audio_captions),
                qc_rules=_dump_json(qc_rules),
                preferences=_dump_json(preferences),
                is_active=False,
                created_at=ts,
                updated_at=ts,
            )
        )
        return int(result.inserted_primary_key[0])
    except Exception:  # noqa: BLE001 - 与原 TS 一致：失败仅告警
        return None


def update_style_profile(conn: Connection, profile_id: Any, patch_input: dict[str, Any]) -> bool:
    """更新。``patch_input`` 里**没有的键**视为 ``undefined``（不动）；
    值为 ``null`` 则按原逻辑落成 null。"""
    if get_style_profile(conn, profile_id) is None:
        return False

    patch: dict[str, Any] = {"updated_at": now()}
    for key in _JSON_FIELDS:
        value = patch_input.get(key, UNSET)
        if value is UNSET:
            continue
        patch[key] = _dump_json(value)
    for key in _PLAIN_FIELDS:
        value = patch_input.get(key, UNSET)
        if value is UNSET:
            continue
        patch[key] = value

    conn.execute(update(style_profiles).where(style_profiles.c.id == profile_id).values(**patch))
    return True


def delete_style_profile(conn: Connection, profile_id: Any) -> bool:
    """软删除（同时取消激活）。"""
    if get_style_profile(conn, profile_id) is None:
        return False
    conn.execute(
        update(style_profiles)
        .where(style_profiles.c.id == profile_id)
        .values(deleted_at=now(), is_active=False)
    )
    return True


def activate_style_profile(conn: Connection, profile_id: Any) -> dict[str, Any] | None:
    """激活（**同 drama 内仅一个**；未绑定 drama 的全局 Profile 不受此约束）。"""
    profile = get_style_profile(conn, profile_id)
    if profile is None:
        return None

    if js_truthy(profile["dramaId"]):
        conn.execute(
            update(style_profiles)
            .where(
                and_(
                    style_profiles.c.drama_id == profile["dramaId"],
                    style_profiles.c.is_active.is_(True),
                )
            )
            .values(is_active=False)
        )

    conn.execute(
        update(style_profiles)
        .where(style_profiles.c.id == profile_id)
        .values(is_active=True, updated_at=now())
    )
    return get_style_profile(conn, profile_id)


def apply_distill_result(conn: Connection, profile_id: Any, result: dict[str, Any]) -> bool:
    """把（用户在 Node 侧提炼并确认后的）结果写入 Profile。

    ⚠️ 与原 TS 一致：这里五个键**总是**写入（不做 undefined 判断），
    所以 ``facts`` / ``inferences`` 两列**不在**写入范围内（它们由提炼流程单独维护）。
    """
    return update_style_profile(
        conn,
        profile_id,
        {
            "storytelling": result.get("storytelling", {}),
            "shot_patterns": result.get("shot_patterns", {}),
            "audio_captions": result.get("audio_captions", {}),
            "qc_rules": result.get("qc_rules", {}),
            "preferences": result.get("preferences", []),
        },
    )
