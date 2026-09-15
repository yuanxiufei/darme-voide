"""评测→优化 无人值守定时调度器 —— 与 ``services/evaluation-scheduler.ts``（126 行）对齐。

* 原生定时实现「每日 HH:MM」（无外部 cron 依赖）；
* **串行**跑全部基准 case（避免并发 LLM 调用引发限流 / 费用激增）；
* **防重入**：上一轮未跑完则跳过本次触发；
* **失败隔离**：单个 case 失败不影响其余；
* 仅在 ``evaluation.auto_optimize.enabled`` 时启动（**默认关闭**，避免意外烧钱）。

⚠️ 三处保真点（都别"顺手修"）：

1. **``running`` 没有 try/finally 兜底**（原 TS 如此）：若循环**之外**出错（典型：
   ``listBenchmarkCases`` 撞上基准目录里的坏 JSON ⇒ 直接抛），``running`` 会**永远停在 true**
   ⇒ 之后所有触发都被「防重入」吃掉。这是原实现的真实缺陷，已记录待评估；迁移期保持同形。
2. ``下一轮时间`` 用**本地时间**算（``setHours`` 语义）：今天的 HH:MM 已过则顺延到次日。
3. 一轮里**每个 case 各开一个短事务**（Python 与 TS 的结构性差异：TS 是全局 ``db``，
   没有连接生命周期的概念）—— 别把整轮包进一个事务，否则 LLM 调用期间会一直占着 SQLite 写锁。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import evaluation as evaluation_config
from app.core.db import engine
from app.core.response import now
from app.agent.evaluation.catalog import list_benchmark_cases, load_case_by_id
from app.agent.evaluation.optimizer import optimize_agent_prompt
from app.agent.evaluation.types import AGENT_BY_KIND
from app.services.task_logger import log_task, log_task_error, log_task_success, log_task_warn

__all__ = [
    "get_scheduler_state",
    "next_run_delay",
    "run_evaluation_cycle",
    "start_evaluation_scheduler",
]

#: 配置在**模块加载时**读一次（原 TS 是顶层 ``const schedule = config.evaluation.autoOptimize``）
_schedule: dict[str, Any] = evaluation_config["auto_optimize"]

#: 调度器状态（``getSchedulerState`` 直接把它交给前端，键名必须 camelCase）
_state: dict[str, Any] = {
    "enabled": bool(_schedule["enabled"]),
    "running": False,
    "lastRunAt": None,
    "nextRunAt": None,
    "lastResults": [],
    "schedule": {
        "hour": int(_schedule["hour"]),
        "minute": int(_schedule["minute"]),
        "iterations": int(_schedule["iterations"]),
        "runOnStartup": bool(_schedule["run_on_startup"]),
    },
}

#: 当前挂着的定时器（``asyncio.TimerHandle``）
_timer: asyncio.TimerHandle | None = None


def next_run_delay(hour: int, minute: int) -> float:
    """距下一个本地 ``HH:MM`` 的**毫秒数**（已过则顺延到次日；秒/微秒归零）。"""
    current = datetime.now()
    target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= current:
        target += timedelta(days=1)
    return (target - current).total_seconds() * 1000


def _iso_in(delay_ms: float) -> str:
    """``new Date(Date.now() + delay).toISOString()`` —— 与 ``response.now()`` 同形。"""
    moment = datetime.now(timezone.utc) + timedelta(milliseconds=delay_ms)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def run_evaluation_cycle() -> list[dict[str, Any]]:
    """跑一轮全量优化：**串行**遍历全部基准 case，返回 ``RunResult[]``。"""
    if _state["running"]:
        log_task_warn("EvalScheduler", "上一轮仍在运行，跳过本次触发")
        return list(_state["lastResults"])

    _state["running"] = True
    _state["lastRunAt"] = now()
    results: list[dict[str, Any]] = []
    cases = list_benchmark_cases()
    log_task("EvalScheduler", f"START 全量优化（{len(cases)} 个 case）")

    for case in cases:
        try:
            case_def = load_case_by_id(case["id"])
            if case_def is None:
                raise ValueError(f"case 加载失败：{case['id']}")
            agent_type = AGENT_BY_KIND.get(case_def.get("kind") or "")
            if not agent_type:
                raise ValueError(f"未知 case kind：{case_def.get('kind')}")

            # ⚠️ 每个 case 一个短事务（见模块头第 3 点）
            with engine.begin() as conn:
                history = await optimize_agent_prompt(
                    conn, agent_type, case_def,
                    {"iterations": _state["schedule"]["iterations"], "autoPersist": True},
                )
            results.append({
                "caseId": case["id"],
                "agentType": agent_type,
                "ok": True,
                "persisted": history.get("persisted") or None,
            })
            log_task("EvalScheduler", f"DONE {case['id']}", {
                "agentType": agent_type,
                "bestScore": history["best"]["score"],
                "persisted": "yes" if history.get("persisted") else "no",
            })
        except Exception as err:  # noqa: BLE001 —— 失败隔离：单个 case 挂了不影响其余
            log_task_error("EvalScheduler", f"case {case['id']} 优化失败", {"error": str(err)})
            results.append({
                "caseId": case["id"],
                "agentType": case["agentType"],
                "ok": False,
                "error": str(err),
            })

    _state["lastResults"] = results
    _state["running"] = False
    ok_count = len([item for item in results if item["ok"]])
    log_task_success("EvalScheduler", f"全量优化完成 {ok_count}/{len(results)}")
    return results


def _on_timer() -> None:
    """定时器回调：跑一轮再重新挂上（``call_later`` 是一次性的）。"""
    async def _run_and_rearm() -> None:
        await run_evaluation_cycle()
        arm()

    asyncio.ensure_future(_run_and_rearm())


def arm() -> None:
    """调度下一次运行。"""
    global _timer
    if not _state["enabled"]:
        return
    delay_ms = next_run_delay(_state["schedule"]["hour"], _state["schedule"]["minute"])
    _state["nextRunAt"] = _iso_in(delay_ms)
    log_task("EvalScheduler", f"下次优化定于 {_state['nextRunAt']}")
    _timer = asyncio.get_event_loop().call_later(delay_ms / 1000, _on_timer)


def get_scheduler_state() -> dict[str, Any]:
    """查询调度器状态（供 ``GET /evaluation/scheduler`` 展示）。"""
    return _state


def start_evaluation_scheduler() -> None:
    """启动调度器（应在应用启动时调用一次）。

    * ``enabled`` 时按每日 HH:MM 定时；
    * ``run_on_startup`` 时启动后**立即跑一轮**（与定时互不冲突，靠防重入保证）。
    """
    if not _state["enabled"]:
        log_task_warn("EvalScheduler", "未启用（evaluation.auto_optimize.enabled=false），跳过")
        return
    hour = _state["schedule"]["hour"]
    minute = _state["schedule"]["minute"]
    log_task(
        "EvalScheduler",
        f"启用每日定时优化（{hour}:{minute:02d}，iterations={_state['schedule']['iterations']}）",
    )
    arm()
    if _state["schedule"]["runOnStartup"]:
        log_task("EvalScheduler", "run_on_startup=true，启动即跑一轮")
        asyncio.ensure_future(run_evaluation_cycle())
