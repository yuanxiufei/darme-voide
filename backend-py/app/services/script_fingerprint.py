"""剧本内容指纹门禁 —— 完整移植 ``backend/src/services/script-fingerprint.ts``。

**为什么值得整服务搬过来**：它是纯逻辑（sha256 + 查库），无 LLM、无子进程、无长任务
⇒ 行为可完全确定地复现；且被媒体生成链路多处依赖（分镜/图片/视频的门禁）。
不搬它，`PUT /episodes/:id` 就会少写 `script_hash` ⇒「剧本改了但下游不提示过期」的
**静默失效** —— 属宁可多搬一段也不能省的那类服务。

⚠️ 两处 JS 语义必须逐字对齐，否则行为漂移：

1. ``ep.scriptContent ?? ep.content ?? ''`` 是 **空值合并（``??``）而非逻辑或（``||``）**：
   ``script_content = ''``（空串）时**不会**回退到 ``content``，结果就是空串 ⇒ 指纹为 null。
   写成 Python 的 ``a or b`` 会变成回退，门禁判定将与 Node 不一致。
2. ``content.replace(/\\r\\n/g, '\\n')`` **只替换 CRLF**，孤立的 ``\\r`` / ``\\n`` 不动。
"""

from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.engine import Connection

from ..core.models import episodes, storyboards
from ..core.response import now


def get_episode_script_source(script_content: str | None, content: str | None) -> str:
    """取剧本指纹内容源：优先 ``script_content``，回退 ``content``。

    用 ``is None`` 而非真值判断 —— 见模块头第 1 条（``??`` ≠ ``||``）。
    """
    source = script_content if script_content is not None else content
    return (source if source is not None else "").strip()


def compute_script_hash(content: str) -> str | None:
    """计算剧本内容指纹（sha256 前 16 位十六进制）；空内容返回 None。"""
    normalized = content.replace("\r\n", "\n").strip()
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def refresh_episode_script_hash(conn: Connection, episode_id: int) -> str | None:
    """重算某集当前剧本指纹并写回 ``episodes.script_hash``；返回新指纹。"""
    row = conn.execute(
        select(episodes.c.script_content, episodes.c.content).where(episodes.c.id == episode_id)
    ).first()
    if row is None:
        return None
    # ⚠️ 与原 TS 一致：这里同时刷新 updated_at
    # ⇒ `PUT /episodes/:id` 改剧本后的 updated_at 实际是「指纹刷新时刻」
    hash_value = compute_script_hash(get_episode_script_source(row.script_content, row.content))
    conn.execute(
        update(episodes)
        .where(episodes.c.id == episode_id)
        .values(script_hash=hash_value, updated_at=now())
    )
    return hash_value


def get_episode_script_hash(conn: Connection, episode_id: int) -> str | None:
    """取某集当前剧本指纹（已存则直读，不重算）。"""
    row = conn.execute(
        select(episodes.c.script_hash, episodes.c.script_content, episodes.c.content).where(
            episodes.c.id == episode_id
        )
    ).first()
    if row is None:
        return None
    if row.script_hash:
        return str(row.script_hash)
    return compute_script_hash(get_episode_script_source(row.script_content, row.content))


def stamp_storyboards_script_hash(conn: Connection, episode_id: int) -> None:
    """把当前剧本指纹写入该集全部分镜（`save_storyboards` 之后调用）。"""
    hash_value = get_episode_script_hash(conn, episode_id)
    rows = conn.execute(select(storyboards.c.id).where(storyboards.c.episode_id == episode_id)).all()
    ts = now()
    for row in rows:
        conn.execute(
            update(storyboards)
            .where(storyboards.c.id == row.id)
            .values(script_hash=hash_value, updated_at=ts)
        )


def check_episode_fingerprint(conn: Connection, episode_id: int) -> dict[str, Any]:
    """检查某集剧本指纹状态：当前指纹 vs 各分镜指纹。"""
    ep = conn.execute(
        select(episodes.c.script_hash, episodes.c.script_content, episodes.c.content).where(
            episodes.c.id == episode_id
        )
    ).first()

    if ep is None:
        return {
            "episode_id": episode_id,
            "current_script_hash": None,
            "has_script": False,
            "storyboard_count": 0,
            "stale_count": 0,
            "stale_storyboards": [],
            "stale": False,
            "stale_with_assets": False,
            "message": "Episode not found",
        }

    current_hash = ep.script_hash or compute_script_hash(
        get_episode_script_source(ep.script_content, ep.content)
    )
    sbs = conn.execute(
        select(
            storyboards.c.id,
            storyboards.c.storyboard_number,
            storyboards.c.script_hash,
            storyboards.c.composed_image,
            storyboards.c.video_url,
            storyboards.c.composed_video_url,
        ).where(storyboards.c.episode_id == episode_id)
    ).all()

    stale_sbs: list[dict[str, Any]] = []
    if current_hash:
        for sb in sbs:
            # 分镜无指纹视为「门禁启用前产物」，不计为过期（保持向后兼容）
            if sb.script_hash and sb.script_hash != current_hash:
                stale_sbs.append(
                    {
                        "id": sb.id,
                        "storyboard_number": sb.storyboard_number,
                        "has_assets": bool(
                            sb.composed_image or sb.video_url or sb.composed_video_url
                        ),
                    }
                )

    return {
        "episode_id": episode_id,
        "current_script_hash": current_hash,
        "has_script": bool(current_hash),
        "storyboard_count": len(sbs),
        "stale_count": len(stale_sbs),
        "stale_storyboards": stale_sbs,
        "stale": len(stale_sbs) > 0,
        "stale_with_assets": any(s["has_assets"] for s in stale_sbs),
        "message": (
            f"剧本已变更：{len(stale_sbs)}/{len(sbs)} 个分镜基于旧剧本，需重新拆解分镜"
            if stale_sbs
            else f"剧本指纹一致：{len(sbs)} 个分镜与当前剧本匹配"
        ),
    }


def check_storyboard_gate(conn: Connection, storyboard_id: int) -> dict[str, Any]:
    """单分镜门禁检查：媒体生成前调用。"""
    sb = conn.execute(
        select(storyboards.c.episode_id, storyboards.c.script_hash).where(
            storyboards.c.id == storyboard_id
        )
    ).first()
    if sb is None:
        return {"allowed": False, "reason": "Storyboard not found"}

    current_hash = get_episode_script_hash(conn, sb.episode_id)
    # 无剧本指纹或分镜无指纹 → 不设门禁
    if not current_hash or not sb.script_hash:
        return {"allowed": True}
    if sb.script_hash != current_hash:
        return {
            "allowed": False,
            "reason": "分镜基于旧剧本生成（剧本已变更），请先重新拆解分镜。如确认继续请传 force=true。",
        }
    return {"allowed": True}
