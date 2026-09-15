"""``/api/v1/app-settings`` —— 与 ``backend/src/routes/app-settings.ts`` 对齐。

全局设置是 KV 表（``app_settings``），目前只承载 ``art_style``（全局默认画风）。
用正常信封（``code: 200``），不是资源库那套。

⚠️ **照抄了一个已知缺陷（有意不"顺手修"）**：原 TS 的 ``ART_STYLE_KEYS`` 只列了
**6 种**画风（realistic / anime / ghibli / cinematic / comic / watercolor），
而画风体系在 2026-09-11 已扩到 **10 种** ⇒ 把全局默认画风设成
``noir`` / ``ink-wash`` / ``cyberpunk`` / ``pixar3d`` 会被 **400 拒绝**。

之所以先照抄：绞杀者迁移期两个后端并存，若这里「顺手修好」，切域的那一刻
行为会静默改变（本来被拒的请求突然通过），排障时极难定位。**修的话要两边一起修。**

权威的 10 种 key 在 ``backend/src/shared/prompt-utils.ts`` 的 ``ART_STYLE_CATALOG``
与 ``frontend/app/utils/artStyles.ts``（本文件的 6 项列表是**过期的第三份副本**）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..core.db import get_conn, get_tx
from ..core.models import app_settings as app_settings_table
from ..core.request_utils import read_json
from ..core.response import bad_request, now, success

router = APIRouter(prefix="/api/v1/app-settings", tags=["app-settings"])

#: ⚠️ 过期副本：实际画风体系有 10 种，这里只有 6 种。见模块头说明。
ART_STYLE_KEYS = ["realistic", "anime", "ghibli", "cinematic", "comic", "watercolor"]


def _all_settings(conn: Connection) -> dict[str, str | None]:
    rows = conn.execute(select(app_settings_table)).all()
    return {r.key: r.value for r in rows}


@router.get("")
def get_settings(conn: Connection = Depends(get_conn)):
    return success(_all_settings(conn))


@router.put("")
async def put_settings(request: Request, conn: Connection = Depends(get_tx)):
    body = await read_json(request)
    if "art_style" in body:
        # TS: String(body.art_style || '') —— 逻辑或，null/undefined/'' 都变成 ''
        raw = body["art_style"]
        value = "" if not raw else str(raw)
        if value and value not in ART_STYLE_KEYS:
            return bad_request(f"art_style 必须是 {'/'.join(ART_STYLE_KEYS)} 之一或留空")

        ts = now()
        existing = conn.execute(
            select(app_settings_table.c.key).where(app_settings_table.c.key == "art_style")
        ).first()
        if existing is not None:
            conn.execute(
                app_settings_table.update()
                .where(app_settings_table.c.key == "art_style")
                .values(value=value or None, updated_at=ts)
            )
        else:
            conn.execute(
                app_settings_table.insert().values(
                    key="art_style", value=value or None, updated_at=ts
                )
            )
    return success(_all_settings(conn))
