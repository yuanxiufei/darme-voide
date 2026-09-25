"""``/api/v1/app-settings`` —— 与 ``backend/src/routes/app-settings.ts`` 对齐。

全局设置是 KV 表（``app_settings``），目前只承载 ``art_style``（全局默认画风）。
用正常信封（``code: 200``），不是资源库那套。

✅ **2026-09-25 修掉了继承自 Node 的过期白名单**：原 TS 的 ``ART_STYLE_KEYS`` 只列了 **6 种**
画风（realistic / anime / ghibli / cinematic / comic / watercolor），而画风体系早在 2026-09-11
就扩到 **10 种** ⇒ 前端设置页「黑色电影 / 国风水墨 / 赛博朋克 / 三维动画」四张卡片点了必 400
（前端选项取自 ``frontend/app/utils/artStyles.ts`` 的 10 项，后端只放行 6 项）。

「照抄不修」的原始理由是绞杀期两个后端并存、单边修会让切域那一刻行为静默改变；**Node 后端已于
2026-09-15 删除** ⇒ 该约束消失。现改为直接引用唯一权威 ``services.prompt_utils.ART_STYLE_KEYS``
（由 ``ART_STYLE_CATALOG`` 派生），本文件不再保留第三份副本。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..core.db import get_conn, get_tx
from ..core.models import app_settings as app_settings_table
from ..core.request_utils import read_json
from ..core.response import bad_request, now, success
# 画风白名单的唯一权威（``ART_STYLE_CATALOG`` 派生，10 种）—— 别再在本文件另抄一份列表。
from ..services.prompt_utils import ART_STYLE_KEYS

router = APIRouter(prefix="/api/v1/app-settings", tags=["app-settings"])


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
