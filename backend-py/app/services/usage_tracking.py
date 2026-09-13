"""用量统计与成本看板 —— 移植 ``backend/src/services/usage-tracking.ts`` 的**读取部分**。

对齐 ArcReel ``usage_repo``：每次模型调用记一条账；本地模型 ``is_local=1`` 不计费；
汇总口径 = 按服务类型 / 提供商 / 按天 / 按剧集。

本模块含**读**与**写**两侧：

* 读：``getUsageSummary`` / ``getEpisodeCostBoard``（路由用）
* 写：``recordUsage``（媒体生成链路用，S4 随 ``image-generation`` 一起接入）
  ⇒ 依赖 ``cost_catalog.estimate_cost`` 的单价目录（已迁）。

⚠️ ``recordUsage`` **不抛错**：记账失败只告警，绝不影响主流程（与 TS 一致）。

⚠️ **金额一律用 ``js_round``**：原 TS 是 ``Math.round(x * 10000) / 10000``，
而 Python 内置 ``round`` 是银行家舍入（``round(2.5) == 2``）⇒ 会与 Node 差 1 个最小单位。
"""

from __future__ import annotations

import json
from typing import Any, Literal

from sqlalchemy import insert, select
from sqlalchemy.engine import Connection

from ..models import api_usage, episodes
from ..response import js_round, js_truthy, now
from .cost_catalog import estimate_cost
from .task_logger import log_task_warn

#: 用量记录状态（``submitted`` 是「已提交、结果未定」）
UsageStatus = Literal["submitted", "completed", "failed"]


def get_usage_summary(
    conn: Connection,
    *,
    drama_id: int | float | None = None,
    episode_id: int | float | None = None,
    limit: int | float | None = 200,
) -> dict[str, Any]:
    """汇总：按项目/集过滤，返回总数、总成本、分类口径与明细（默认最近 200 条）。

    ⚠️ 顶层是 **camelCase**（``totalCount``/``totalCost``/``byService``...），
    但 ``records`` 与各分组条目内部是 **snake_case**（``service_type``/``cost_amount``...）——
    原 TS 就是这样混着的，照抄。
    """
    rows = [dict(r._mapping) for r in conn.execute(select(api_usage)).all()]
    rows = [
        r
        for r in rows
        if (drama_id is None or r["drama_id"] == drama_id)
        and (episode_id is None or r["episode_id"] == episode_id)
    ]

    by_service: dict[str, dict[str, float]] = {}
    by_provider: dict[str, dict[str, float]] = {}
    by_day: dict[str, dict[str, float]] = {}
    total_cost = 0.0
    costed = 0

    for r in rows:
        cost = r["cost_amount"] or 0
        total_cost += cost
        if r["cost_amount"] is not None:
            costed += 1

        svc = by_service.setdefault(r["service_type"], {"count": 0, "cost": 0})
        svc["count"] += 1
        svc["cost"] += cost

        prov = by_provider.setdefault(r["provider"], {"count": 0, "cost": 0})
        prov["count"] += 1
        prov["cost"] += cost

        date = (r["created_at"] or "")[:10]
        day = by_day.setdefault(date, {"count": 0, "cost": 0})
        day["count"] += 1
        day["cost"] += cost

    # 按 count 倒序（JS sort 稳定 ⇒ 同 count 保持首次出现顺序；Python sorted 同样稳定）
    sorted_service = dict(sorted(by_service.items(), key=lambda kv: kv[1]["count"], reverse=True))
    sorted_provider = dict(sorted(by_provider.items(), key=lambda kv: kv[1]["count"], reverse=True))

    # TS: Math.max(1, filter.limit ?? 200)；limit 为 NaN 时会原样传播，slice(0, NaN) 得空数组
    if limit is None or limit != limit:  # None / NaN
        recent: list[dict[str, Any]] = []
    else:
        ordered = sorted(rows, key=lambda r: r["id"], reverse=True)
        recent = ordered[: int(max(1, limit))]

    return {
        "totalCount": len(rows),
        "totalCost": js_round(total_cost * 10000) / 10000,
        "checkedCost": costed > 0,
        "byService": [
            {"service_type": k, **v} for k, v in sorted_service.items()
        ],
        "byProvider": [{"provider": k, **v} for k, v in sorted_provider.items()],
        "byDay": [
            {"date": k, **v}
            # TS 用 localeCompare 排序 —— 日期是 ASCII，等价于字典序
            for k, v in sorted(by_day.items(), key=lambda kv: kv[0])
        ],
        "records": [
            {
                "id": r["id"],
                "service_type": r["service_type"],
                "provider": r["provider"],
                "model": r["model"],
                "drama_id": r["drama_id"],
                "episode_id": r["episode_id"],
                "storyboard_id": r["storyboard_id"],
                "units": r["units"],
                "cost_amount": r["cost_amount"],
                "is_local": bool(r["is_local"]),
                "status": r["status"] or "",
                "retry_count": r["retry_count"] or 0,
                "created_at": r["created_at"],
            }
            for r in recent
        ],
    }


def get_episode_cost_board(conn: Connection, drama_id: int | float) -> dict[str, Any]:
    """多集成本看板：按剧聚合每集成本（总成本 / 总调用 / 重拍成本 / 按服务类型拆分）。

    重拍成本 = 同一任务 ``retry_count >= 1`` 的调用成本，用来回答「重拍烧了多少钱」。
    **整体 snake_case**（与原 TS 一致）。
    """
    usage_rows = [
        dict(r._mapping)
        for r in conn.execute(select(api_usage)).all()
        if r._mapping["drama_id"] == drama_id and r._mapping["episode_id"] is not None
    ]

    ep_rows = conn.execute(
        select(episodes.c.id, episodes.c.episode_number, episodes.c.title)
        .where(episodes.c.drama_id == drama_id)
        .order_by(episodes.c.episode_number)
    ).all()

    board: dict[int, dict[str, Any]] = {}
    for ep in ep_rows:
        board[ep.id] = {
            "episode_id": ep.id,
            "episode_number": ep.episode_number,
            "title": ep.title,
            "total_cost": 0.0,
            "total_calls": 0,
            "retry_cost": 0.0,
            "by_service": [],
        }

    drama_total = 0.0
    drama_calls = 0
    drama_retry = 0.0

    for r in usage_rows:
        ep_row = board.get(r["episode_id"] if r["episode_id"] is not None else -1)
        if ep_row is None:
            continue
        cost = r["cost_amount"] or 0
        is_retry = (r["retry_count"] or 0) >= 1

        ep_row["total_cost"] += cost
        ep_row["total_calls"] += 1
        if is_retry:
            ep_row["retry_cost"] += cost

        drama_total += cost
        drama_calls += 1
        if is_retry:
            drama_retry += cost

        found = next(
            (s for s in ep_row["by_service"] if s["service_type"] == r["service_type"]), None
        )
        if found is not None:
            found["count"] += 1
            found["cost"] += cost
        else:
            ep_row["by_service"].append(
                {"service_type": r["service_type"], "count": 1, "cost": cost}
            )

    for ep_row in board.values():
        ep_row["total_cost"] = js_round(ep_row["total_cost"] * 10000) / 10000
        ep_row["retry_cost"] = js_round(ep_row["retry_cost"] * 10000) / 10000
        ep_row["by_service"].sort(key=lambda s: s["count"], reverse=True)

    return {
        "drama_id": drama_id,
        "total_cost": js_round(drama_total * 10000) / 10000,
        "total_calls": drama_calls,
        "retry_cost": js_round(drama_retry * 10000) / 10000,
        "episodes": list(board.values()),
    }


def record_usage(conn: Connection, input_data: dict[str, Any]) -> int | None:
    """记录一次模型调用，返回新行 id；**失败只告警、不抛错**（绝不影响主流程）。

    ⚠️ 几个保真点：

    * ``units ?? null`` 与 ``isLocal ?? false`` 都是 **nullish** —— 传 0 要保留 0
      （``units=0`` 表示「没产出任何单位」，与「未提供」不同）；
    * **本地模型一律不记成本**（``costAmount = null``），因为不产生费用；
    * ``meta`` 用 truthy 判断：传 ``{}`` 也会落成 ``"{}"``（JS 空对象是真值）；
    * 单价查不到时 ``estimate_cost`` 返回 None ⇒ ``cost_amount`` 落 NULL（**不是 0**），
      这样成本看板才能区分「免费/本地」与「单价未知」。
    """
    try:
        units = input_data.get("units")
        cost_amount = (
            None
            if input_data.get("isLocal")
            else estimate_cost(
                input_data.get("serviceType"),
                input_data.get("provider"),
                input_data.get("model"),
                units,
                input_data.get("settings"),
            )
        )
        meta = input_data.get("meta")
        result = conn.execute(
            insert(api_usage).values(
                service_type=input_data.get("serviceType"),
                provider=input_data.get("provider"),
                model=input_data.get("model"),
                drama_id=input_data.get("dramaId"),
                episode_id=input_data.get("episodeId"),
                storyboard_id=input_data.get("storyboardId"),
                image_generation_id=input_data.get("imageGenerationId"),
                video_generation_id=input_data.get("videoGenerationId"),
                units=units,
                cost_amount=cost_amount,
                is_local=js_truthy(input_data.get("isLocal"))
                if input_data.get("isLocal") is not None
                else False,
                status=input_data.get("status") or "submitted",
                retry_count=input_data.get("retryCount")
                if input_data.get("retryCount") is not None
                else 0,
                meta=json.dumps(meta, ensure_ascii=False, separators=(",", ":")) if js_truthy(meta) else None,
                created_at=now(),
            )
        )
        return int(result.lastrowid)
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch (err: any) 等价
        log_task_warn(
            "Usage",
            "record-failed",
            {"serviceType": input_data.get("serviceType"), "error": str(err)},
        )
        return None
