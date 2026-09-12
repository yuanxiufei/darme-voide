"""六键 Bible 的 ID 收口 —— 移植 ``backend/src/services/bible-ids.ts``（纯 DB 逻辑，无 LLM）。

三个 ID 都是「跨集锁定」的稳定标识：
* ``STYLE_{dramaId:03d}``  一剧一 ID，挂在 ``dramas.style_id``
* ``COST_{characterId:03d}`` 一角色一 ID，换装时才显式变更，挂在 ``characters.costume_id``
* ``LOC_{seq:03d}``        同剧内**按 location 文本复用**，挂在 ``scenes.location_id``

⚠️ LOC 的复用判据是「同剧 + 未软删 + location 文本 trim 后相同」，且序号取
现有 ``LOC_数字`` 的**最大值 + 1**（不是计数 + 1）—— 照抄 TS 版，避免删过场景后 ID 撞车。
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import and_, select, update
from sqlalchemy.engine import Connection

from ..models import characters, dramas, scenes

_LOC_RE = re.compile(r"^LOC_(\d+)$")


def _pad3(n: int) -> str:
    return f"{n:03d}"


def ensure_style_id(conn: Connection, drama_id: int) -> str:
    row = conn.execute(select(dramas.c.style_id).where(dramas.c.id == drama_id)).first()
    if row is None:
        return ""
    if row[0]:
        return str(row[0])
    style_id = f"STYLE_{_pad3(drama_id)}"
    conn.execute(update(dramas).where(dramas.c.id == drama_id).values(style_id=style_id))
    return style_id


def ensure_costume_id(conn: Connection, character_id: int) -> str:
    row = conn.execute(select(characters.c.costume_id).where(characters.c.id == character_id)).first()
    if row is None:
        return ""
    if row[0]:
        return str(row[0])
    costume_id = f"COST_{_pad3(character_id)}"
    conn.execute(update(characters).where(characters.c.id == character_id).values(costume_id=costume_id))
    return costume_id


def ensure_location_id(conn: Connection, scene_id: int) -> str:
    scene = conn.execute(
        select(scenes.c.id, scenes.c.drama_id, scenes.c.location, scenes.c.location_id).where(
            scenes.c.id == scene_id
        )
    ).first()
    if scene is None:
        return ""
    if scene.location_id:
        return str(scene.location_id)

    norm = (scene.location or "").strip()
    siblings: list[Any] = conn.execute(
        select(scenes.c.id, scenes.c.location, scenes.c.location_id).where(
            and_(scenes.c.drama_id == scene.drama_id, scenes.c.deleted_at.is_(None))
        )
    ).all()

    # 复用同地点已有 ID
    location_id = ""
    for sib in siblings:
        if sib.id != scene_id and sib.location_id and (sib.location or "").strip() == norm:
            location_id = str(sib.location_id)
            break

    if not location_id:
        max_seq = 0
        for sib in siblings:
            m = _LOC_RE.match(str(sib.location_id or ""))
            if m:
                max_seq = max(max_seq, int(m.group(1)))
        location_id = f"LOC_{_pad3(max_seq + 1)}"

    conn.execute(update(scenes).where(scenes.c.id == scene_id).values(location_id=location_id))
    return location_id
