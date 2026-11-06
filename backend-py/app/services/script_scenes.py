"""项目剧本的场次结构（B 侧事实）—— ``episodes`` 正文 → ``script_scenes``。

与 A 侧（``scene_index``）的分工
==============================
* A：库里 5403 份**参考资料**，纯派生，``DELETE`` 后重跑即可，写在独立 sidecar；
* B：本仓**自己的剧本**，是项目事实，写在 ``darme.db``，带指纹，剧本一改就标 stale。

两侧共用同一个解析器 ``scene_parser.parse_scenes``（一套 schema、两种归属键），
但**不共用表** —— 混在一张表里，"重跑解析"会意外清掉项目数据。

偏移契约（前端切片必须照这个来）
==============================
``char_start`` / ``char_end`` **相对原文、未 strip**。而剧本指纹是按 ``strip`` 后的内容算的
（``script_fingerprint.compute_script_hash`` 的既有口径，不能改）⇒ 两者差的只是首尾空白，
偏移以原文为准、哈希以 strip 为准。回执里的 ``sourceField`` 说明该按哪个字段切片
（``scriptContent`` 优先，回退 ``content``）。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.engine import Connection

from app.core.models import characters, episodes, script_scene_docs, script_scenes
from app.core.response import now
from app.services.scene_parser import parse_scenes
from app.services.script_fingerprint import (
    compute_script_hash,
    get_episode_script_source,
)

__all__ = ["list_episode_scenes", "parse_episode_scenes"]


def _episode_source(conn: Connection, episode_id: int) -> dict[str, Any] | None:
    """取这一集的正文 + 该按哪个字段切片 + 当前指纹。"""
    row = conn.execute(
        select(episodes.c.script_content, episodes.c.content, episodes.c.drama_id)
        .where(episodes.c.id == episode_id)
    ).first()
    if row is None:
        return None
    has_script = row.script_content is not None
    return {
        "raw": row.script_content if has_script else (row.content or ""),
        "sourceField": "scriptContent" if has_script else "content",
        "hash": compute_script_hash(get_episode_script_source(row.script_content, row.content)),
        "dramaId": row.drama_id,
    }


def parse_episode_scenes(conn: Connection, episode_id: int, *, force: bool = False) -> dict[str, Any]:
    """解析这一集 → 覆盖写 ``script_scenes``。指纹没变且非 ``force`` 时跳过（可反复调用）。

    跳过判据用**指纹**而不是时间戳：``PUT /episodes`` 每次都会刷新 ``updated_at``，
    拿它判断会把"只改了标题"也当成"剧本变了"。
    """
    source = _episode_source(conn, episode_id)
    if source is None:
        return {"available": False, "hint": f"剧集 {episode_id} 不存在。"}

    digest = source["hash"]
    if digest is None:
        return {"available": True, "episodeId": episode_id, "scenes": 0, "skipped": False,
                "hint": "这一集还没有剧本正文（``script_content`` 与 ``content`` 都是空的）。",
                "notes": []}

    existing = conn.execute(
        select(script_scene_docs.c.script_hash, script_scene_docs.c.scenes)
        .where(script_scene_docs.c.episode_id == episode_id)
    ).first()
    if not force and existing is not None and existing.script_hash == digest:
        return {"available": True, "episodeId": episode_id, "scenes": int(existing.scenes or 0),
                "skipped": True, "scriptHash": digest,
                "hint": "剧本指纹未变，场次已是最新（要重算请带 ``force=true``）。",
                "notes": []}

    #: 项目角色表是本仓独有的 ground truth —— 拿它放宽名字长度上限，
    #: 否则"欧阳明月：…"这种 4 字以上的名字会被长度启发式当成动作行丢掉。
    known: list[str] = []
    if source["dramaId"] is not None:
        known = [
            row.name for row in conn.execute(
                select(characters.c.name).where(characters.c.drama_id == source["dramaId"])
            ) if row.name
        ]

    parsed = parse_scenes(source["raw"], known_cast=known)
    items = parsed["scenes"]
    ts = now()
    conn.execute(delete(script_scenes).where(script_scenes.c.episode_id == episode_id))
    if items:
        conn.execute(
            script_scenes.insert(),
            [{
                "episode_id": episode_id,
                "scene_no": item["scene_no"],
                "no_source": item["no_source"],
                "int_ext": item["int_ext"],
                "location": item["location"],
                "time": item["time"],
                "cast": ", ".join(item["cast"]) if item["cast"] else None,
                "opening": item["opening"],
                "props": None,
                "core_event": item["core_event"],
                "scene_tail": item["scene_tail"],
                "char_start": item["char_start"],
                "char_end": item["char_end"],
                "chars": item["chars"],
                "confidence": item["confidence"],
                "extractor": item["extractor"],
                "source_script_hash": digest,
                "status": "current",
                "created_at": ts,
                "updated_at": ts,
            } for item in items],
        )

    conn.execute(delete(script_scene_docs).where(script_scene_docs.c.episode_id == episode_id))
    conn.execute(
        script_scene_docs.insert().values(
            episode_id=episode_id,
            script_hash=digest,
            strategy=parsed["strategy"],
            scenes=len(items),
            confidence=parsed["confidence"],
            chars=len(source["raw"]),
            notes="\n".join(parsed["notes"])[:2000] or None,
            source_field=source["sourceField"],
            status="current",
            created_at=ts,
            updated_at=ts,
        )
    )
    return {
        "available": True, "episodeId": episode_id, "scenes": len(items), "skipped": False,
        "scriptHash": digest, "strategy": parsed["strategy"],
        "confidence": parsed["confidence"], "sourceField": source["sourceField"],
        "notes": parsed["notes"], "hint": None,
    }


def list_episode_scenes(conn: Connection, episode_id: int) -> dict[str, Any]:
    """读回场次表，并**当场**判 stale（不靠定时任务刷新状态列）。

    为什么不在写入时才更新 ``status``：剧本可以通过 ``PUT /episodes``、续写、甚至直接改库
    变成另一副样子，每一条路径都去更新场次状态就是 N 个漏点。读时用当前指纹比一次，
    只有一处判据，不会漏。
    """
    source = _episode_source(conn, episode_id)
    if source is None:
        return {"available": False, "scenes": [], "hint": f"剧集 {episode_id} 不存在。"}

    doc = conn.execute(
        select(script_scene_docs).where(script_scene_docs.c.episode_id == episode_id)
    ).first()
    #: 按 ``id``（= 插库顺序 = 剧本顺序）排，不按 ``scene_no``：未分场的那一段是
    #: ``scene_no=NULL``，按场号排会把它甩到最后，正文顺序就断了。
    rows = conn.execute(
        select(script_scenes)
        .where(script_scenes.c.episode_id == episode_id)
        .order_by(script_scenes.c.id)
    ).all()

    #: 用**当前正文现算**的指纹，不用 ``episodes.script_hash`` 那一列：
    #: 列是缓存，走 ``PUT /episodes`` 才刷新；直接改库（脚本/迁移/手工）时它是陈的，
    #: 于是"剧本变了"被判成没变 —— 正是最该报 stale 的时候不报。
    current_hash = source["hash"]
    parsed_hash = doc.script_hash if doc is not None else None
    stale = bool(rows) and parsed_hash is not None and current_hash != parsed_hash

    scenes = [{
        "sceneNo": row.scene_no,
        "noSource": row.no_source,
        "intExt": row.int_ext,
        "location": row.location,
        "time": row.time,
        "cast": [name.strip() for name in (row.cast or "").split(",") if name.strip()],
        "opening": row.opening,
        "props": [name.strip() for name in (row.props or "").split(",") if name.strip()],
        "coreEvent": row.core_event,
        "sceneTail": row.scene_tail,
        "charStart": row.char_start,
        "charEnd": row.char_end,
        "chars": row.chars,
        "confidence": row.confidence,
        "extractor": row.extractor,
    } for row in rows]

    hint = None
    if not rows:
        hint = ("这一集还没解析过场次（属按需生成，不是漏做）；"
                "调 ``POST /episodes/{id}/script-scenes`` 生成。")
    elif stale:
        hint = "剧本已改动，场次结构已过期 —— 重新解析前不要拿它做分镜。"

    return {
        "available": True,
        "episodeId": episode_id,
        "sourceField": (doc.source_field if doc is not None else source["sourceField"]),
        "scriptHash": current_hash,
        "parsedHash": parsed_hash,
        "stale": stale,
        "strategy": (doc.strategy if doc is not None else None),
        "confidence": (doc.confidence if doc is not None else None),
        "notes": (doc.notes.split("\n") if doc is not None and doc.notes else []),
        "parsedAt": (doc.updated_at if doc is not None else None),
        "scenes": scenes,
        "hint": hint,
    }
