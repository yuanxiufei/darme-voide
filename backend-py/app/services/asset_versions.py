"""资产版本历史 / 回滚 —— 移植 ``backend/src/services/asset-versions.ts``（整服务）。

每次图片/视频生成成功留档一条 ``asset_versions`` 记录（``current``）；同一资产再次生成时
旧版本降为 ``historical``，新版本成为 ``current``。**回滚 = 把某历史版本重新置为 ``current``，
并把它的 ``asset_url`` 写回主表字段**。

资产标识 = ``asset_type``（storyboard/character/scene/prop）+ ``asset_id``。
分镜图片还按 ``frame_type``（composed/first_frame/last_frame/keyframe）分组版本，
即同一条分镜的合成图、首帧、尾帧各自独立版本历史。

⚠️ 两处语义照抄：

* 分组判据里 ``frame_type`` 用的是「**IS NULL** 或 =」，因为同一资产「无帧类型」也是一组，
  写成 ``= None`` 在 SQL 里恒不成立 ⇒ 会把无帧类型的版本漏出分组。
* ``record_asset_version`` **不抛错**（失败仅告警返回 None）—— 它挂在生成主流程上，
  留档失败不该让已经成功的生成任务报错。
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import and_, select, update
from sqlalchemy.engine import Connection

from ..core.models import asset_versions, characters, prop_templates, scenes, storyboards
from ..core.response import now, row_to_camel

STORYBOARD_FRAME_TYPES = ("composed", "first_frame", "last_frame", "keyframe")


def _frame_condition(frame_type: str | None):
    """帧类型分组条件：None ⇒ ``IS NULL``（不能用 ``= NULL``，那样恒不成立）。"""
    column = asset_versions.c.frame_type
    return column.is_(None) if frame_type is None else column == frame_type


def resolve_storyboard_frame_type(frame_type: str | None) -> str:
    """从生成记录推导分镜图片的 frame_type（无法识别时视为 composed）。"""
    if frame_type in ("first_frame", "last_frame", "keyframe"):
        return frame_type
    return "composed"


def record_asset_version(
    conn: Connection,
    *,
    asset_type: str,
    asset_id: int,
    media_type: str,
    asset_url: str,
    frame_type: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    prompt: str | None = None,
    generation_id: int | None = None,
    meta: dict[str, Any] | None = None,
) -> int | None:
    """生成成功时留档：旧版本降级 historical，新版本成为 current。失败返回 None（不抛错）。"""
    try:
        frame = frame_type if frame_type is not None else None
        existing = conn.execute(
            select(asset_versions.c.id, asset_versions.c.version, asset_versions.c.status).where(
                and_(
                    asset_versions.c.asset_type == asset_type,
                    asset_versions.c.asset_id == asset_id,
                    asset_versions.c.media_type == media_type,
                    _frame_condition(frame),
                )
            )
        ).all()

        for row in existing:
            if row.status == "current":
                conn.execute(
                    update(asset_versions)
                    .where(asset_versions.c.id == row.id)
                    .values(status="historical")
                )

        next_version = max((r.version for r in existing), default=0) + 1 if existing else 1
        result = conn.execute(
            asset_versions.insert().values(
                asset_type=asset_type,
                asset_id=asset_id,
                media_type=media_type,
                frame_type=frame,
                version=next_version,
                asset_url=asset_url,
                provider=provider,
                model=model,
                prompt=prompt,
                generation_id=generation_id,
                # ⚠️ 紧凑分隔符：Node 是 `JSON.stringify(meta)`，Python 默认会多出空格
                meta=json.dumps(meta, ensure_ascii=False, separators=(",", ":")) if meta else None,
                status="current",
                created_at=now(),
            )
        )
        return int(result.inserted_primary_key[0])
    except Exception:  # noqa: BLE001 - 与原 TS 一致：留档失败不阻断主流程
        return None


def list_asset_versions(conn: Connection, asset_type: str, asset_id: int) -> list[dict[str, Any]]:
    """资产版本列表（版本号倒序，最新在前）。返回 camelCase（drizzle 行形状）。"""
    rows = conn.execute(
        select(asset_versions)
        .where(
            and_(
                asset_versions.c.asset_type == asset_type,
                asset_versions.c.asset_id == asset_id,
            )
        )
        .order_by(asset_versions.c.version.desc())
    ).all()
    return [row_to_camel(r, "asset_versions") for r in rows]


def apply_asset_to_entity(conn: Connection, row: Any, ts: str) -> None:
    """把资产 URL 写回对应主表字段（回滚生效）。"""
    asset_type = row.asset_type
    asset_id = row.asset_id
    asset_url = row.asset_url

    if asset_type == "storyboard":
        values: dict[str, Any] = {"updated_at": ts}
        if row.media_type == "video":
            values["video_url"] = asset_url
        elif row.frame_type == "first_frame":
            values["first_frame_image"] = asset_url
        elif row.frame_type == "last_frame":
            values["last_frame_image"] = asset_url
        elif row.frame_type == "keyframe":
            values["keyframe_image"] = asset_url
        else:
            values["composed_image"] = asset_url
        conn.execute(update(storyboards).where(storyboards.c.id == asset_id).values(**values))
        return

    if asset_type == "character":
        conn.execute(
            update(characters)
            .where(characters.c.id == asset_id)
            .values(image_url=asset_url, updated_at=ts)
        )
        return

    if asset_type == "scene":
        # 场景回滚额外把 status 置 completed（原 TS 如此）
        conn.execute(
            update(scenes)
            .where(scenes.c.id == asset_id)
            .values(image_url=asset_url, status="completed", updated_at=ts)
        )
        return

    if asset_type == "prop":
        conn.execute(
            update(prop_templates)
            .where(prop_templates.c.id == asset_id)
            .values(image_url=asset_url, updated_at=ts)
        )
        return
    # 未知 asset_type：原 TS 只记一条告警，不做任何写入


def activate_asset_version(conn: Connection, version_id: int | float) -> dict[str, Any]:
    """回滚：把指定版本置为 current，``asset_url`` 写回主表对应字段。

    返回 ``{ok, error?, row?}`` —— ``row`` 为 camelCase（drizzle 行形状）。
    """
    row = conn.execute(
        select(asset_versions).where(asset_versions.c.id == version_id)
    ).first()
    if row is None:
        return {"ok": False, "error": "Version not found"}

    ts = now()

    # 同组其他 current 降级
    conn.execute(
        update(asset_versions)
        .where(
            and_(
                asset_versions.c.asset_type == row.asset_type,
                asset_versions.c.asset_id == row.asset_id,
                asset_versions.c.media_type == row.media_type,
                _frame_condition(row.frame_type),
                asset_versions.c.status == "current",
            )
        )
        .values(status="historical")
    )
    conn.execute(
        update(asset_versions).where(asset_versions.c.id == version_id).values(status="current")
    )

    try:
        apply_asset_to_entity(conn, row, ts)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "row": row_to_camel(row, "asset_versions")}

    return {"ok": True, "row": row_to_camel(row, "asset_versions")}
