"""per-shot take 预算（移植自 ``backend/src/services/take-budget.ts``，86 行）。

每个分镜（shot）允许的生成尝试次数由 ``storyboards.take_budget`` 控制（默认 3）。
每次提交图片/视频生成（**无论成败**）消耗一次 take；超预算后阻断继续生成，
强制用户重新拆解分镜（此时重置预算）或显式 ``force`` 覆盖。

价值：防止失败重试/反复调参导致同一分镜**无限消耗算力**，把资源收敛到
「预算内重试 → 预算耗尽 → 重新拆解」的收敛循环。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection, Row

from ..core.models import storyboards
from ..core.response import now

__all__ = [
    "DEFAULT_TAKE_BUDGET",
    "check_take_budget",
    "consume_take",
    "get_take_status",
    "reset_take_budget",
    "reset_take_budget_for_episode",
]

DEFAULT_TAKE_BUDGET = 3


def _fetch(conn: Connection, storyboard_id: int | None) -> Row | None:
    return conn.execute(
        select(storyboards).where(storyboards.c.id == storyboard_id)
    ).first()


def get_take_status(conn: Connection, storyboard_id: int | None) -> dict[str, Any] | None:
    """取分镜 take 状态；分镜不存在返回 None。

    ⚠️ 返回的是 **snake_case**（``take_count``/``take_budget``/``remaining``/``exhausted``），
    与 TS 一致 —— 它最终会出现在 400 报错体与前端提示里。
    """
    row = _fetch(conn, storyboard_id)
    if row is None:
        return None
    # `sb.takeBudget ?? DEFAULT_TAKE_BUDGET` / `sb.takeCount ?? 0` 是 **nullish**
    # —— 库里存 0 时要保留 0（用 `or` 会把 take_count=0 变成默认值，但那两处恰好等价，
    #    唯独 take_budget 传 0 时语义不同：0 预算 = 一律阻断）
    budget = row.take_budget if row.take_budget is not None else DEFAULT_TAKE_BUDGET
    count = row.take_count if row.take_count is not None else 0
    return {
        "storyboard_id": storyboard_id,
        "take_count": count,
        "take_budget": budget,
        "remaining": max(0, budget - count),
        "exhausted": count >= budget,
    }


def consume_take(conn: Connection, storyboard_id: int | None) -> None:
    """消耗一次 take（媒体生成提交成功后调用）。

    ⚠️ **任务提交成功即消耗**，与后续成败无关 —— 算的是「尝试次数」。
    """
    if not storyboard_id:
        return
    row = _fetch(conn, storyboard_id)
    if row is None:
        return
    count = row.take_count if row.take_count is not None else 0
    conn.execute(
        storyboards.update()
        .where(storyboards.c.id == storyboard_id)
        .values(take_count=count + 1, updated_at=now())
    )


def check_take_budget(
    conn: Connection, storyboard_id: int | None, force: bool | None = None
) -> dict[str, Any]:
    """门禁检查：预算是否耗尽。返回 ``{allowed, reason?, status?}``。

    未绑定 storyboard 或分镜不存在时一律放行（与 TS 一致）。
    """
    if not storyboard_id:
        return {"allowed": True}
    status = get_take_status(conn, storyboard_id)
    if status is None:
        return {"allowed": True}
    if status["exhausted"] and not force:
        return {
            "allowed": False,
            "reason": (
                f"该分镜 take 预算已耗尽（{status['take_count']}/{status['take_budget']}）。"
                "请重新拆解分镜以重置预算，或传 force=true 强制继续。"
            ),
            "status": status,
        }
    return {"allowed": True, "status": status}


def reset_take_budget(conn: Connection, storyboard_id: int) -> None:
    """重置分镜 take 预算（重新拆解分镜后调用）。"""
    conn.execute(
        storyboards.update()
        .where(storyboards.c.id == storyboard_id)
        .values(take_count=0, updated_at=now())
    )


def reset_take_budget_for_episode(conn: Connection, episode_id: int) -> None:
    """重置某集所有分镜 take 预算。"""
    rows = conn.execute(
        select(storyboards.c.id).where(storyboards.c.episode_id == episode_id)
    ).all()
    for row in rows:
        reset_take_budget(conn, row[0])
