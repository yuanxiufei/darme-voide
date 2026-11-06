"""知识蒸馏的**定时复盘** —— 把「跑次里的统计信号」每天自动整理成待审经验。

为什么需要它
============
``knowledge_distill`` 只提供了「跑一次」的能力（HTTP 上是 ``POST /distill``）。
没人点，它就永远不跑 —— 而"知识积累"这件事最怕的恰恰是**依赖人记得点**：
一次集中失败之后没人复盘，二十集之后同样的错还在犯，台账里却一条相关经验都没有。

所以补一个定时器，形态与 ``agent/evaluation_scheduler.py`` 一致（这个是本仓已有成法，
不要另造一套调度）：原生每日 ``HH:MM``、**串行**、**防重入**、**失败隔离**、默认关。

它**只写 ``proposed``**，而且**永不调用模型**
============================================
* 写 ``proposed`` 是 ``knowledge_distill.distill`` 的既定行为：定时任务不是审核者。
  无人值守时能做的只有"把现象记下来"，能不能变成 ``accepted``（进而被注进 system prompt）
  必须由人或者后续的 ``lift`` 证据决定 ✓。
* **不调模型**是刻意的：无人值守 + 自动烧模型调用 = 账单事故，且模型生成的"经验"
  在没人看的夜里直接堆进台账，第二天人只会一眼扫过就批量通过（审阅质量形同取消）。
  规则式模板（``source_kind="distill-rule"``、``confidence`` 封顶 0.5）足够把"该看什么"摆出来；
  想用模型归纳就手动点 ``POST /distill`` 带 ``useLlm=true`` —— 那时候有人在场 ✓。

与本仓另外两条定时任务纪律的关系
================================
1. **别照抄 ``evaluation_scheduler`` 的一个已知缺陷**：它把 ``running`` 的复位写在
   正常路径上（没有 ``try/finally``），循环之外抛错就会**永远卡在 running** ⇒
   此后每一轮都被「防重入」吃掉，而且不报错。那份是 TS 迁移期的保真选择，
   本模块是**新写**的，没有保真义务 ⇒ 一律 ``try/finally`` 复位 ✓。
2. ``lift`` 单独一格开关（默认也关）：蒸馏只**增加待审条目**，而 lift 会
   **把 ``lift<0`` 的既有条目降级**（``deprecated``）—— 「加一条待审」与「撤一条旧结论」
   是两种力度的动作，不该绑在同一个开关上。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from app.core.config import knowledge as knowledge_config
from app.core.response import now
from app.services import knowledge_distill
from app.services.task_logger import log_task, log_task_error, log_task_success, log_task_warn

__all__ = [
    "arm",
    "run_cycle",
    "start",
    "state",
    "stop",
]

_SCOPE = "KnowledgeScheduler"

#: 配置在**模块加载时**读一次（与 ``evaluation_scheduler`` 同形：调度参数不该在运行中漂移）
_schedule: dict[str, Any] = knowledge_config["distill"]

_state: dict[str, Any] = {
    "enabled": bool(_schedule["enabled"]),
    "running": False,
    "lastRunAt": None,
    "nextRunAt": None,
    "lastResult": None,
    "schedule": {
        "hour": int(_schedule["hour"]),
        "minute": int(_schedule["minute"]),
        "limit": int(_schedule["limit"]),
        "lift": bool(_schedule["lift"]),
        "runOnStartup": bool(_schedule["run_on_startup"]),
    },
}

#: 当前挂着的定时器（``asyncio.TimerHandle``）
_timer: asyncio.TimerHandle | None = None


def state() -> dict[str, Any]:
    """调度器状态（交给 ``GET /api/v1/knowledge-loop/scheduler``）。

    **关着也照实答** ``enabled=false``：前端要能区分「没开」与「开了但失败」，
    两者都表现为「台账没有新条目」✗。
    """
    return {**_state, "schedule": dict(_state["schedule"])}


# ---------------------------------------------------------------------------
# 一轮
# ---------------------------------------------------------------------------

def _default_distill(limit: int, dry_run: bool) -> dict[str, Any]:
    """规则式蒸馏（不花模型调用）。候选为空时**如实回 0 条**，不伪造。"""
    pool = knowledge_distill.candidates()[:limit]
    if not pool:
        return {"candidates": 0, "written": 0, "rejected": 0, "items": [],
                "hint": "没有达到阈值的统计信号（样本不足或确实没有异常）"}
    return knowledge_distill.distill(candidates_list=pool, limit=limit, dry_run=dry_run)


async def run_cycle(
    *,
    dry_run: bool = False,
    limit: int | None = None,
    lift: bool | None = None,
    distill_fn: Callable[[int, bool], dict[str, Any]] | None = None,
    lift_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """跑一轮：先蒸馏，再（可选）刷新 ``lift``。**哪一步炸都不影响另一步**。

    ``distill_fn`` / ``lift_fn`` 只是为了测试能注入桩；生产路径不传，
    走 ``knowledge_distill`` 的真实现（同一套逻辑，不写第二份）。
    """
    at = now()
    if _state["running"]:
        log_task_warn(_SCOPE, "上一轮仍在运行，跳过本次触发")
        return {"ok": False, "skipped": "running", "at": at, "dryRun": bool(dry_run),
                "distill": None, "lift": None, "errors": [], "elapsedMs": 0}

    cap = max(1, int(limit if limit is not None else _state["schedule"]["limit"]))
    want_lift = bool(_state["schedule"]["lift"]) if lift is None else bool(lift)
    started = time.monotonic()
    errors: list[str] = []
    summary: dict[str, Any] = {
        "ok": True, "skipped": None, "at": at, "dryRun": bool(dry_run),
        "distill": None, "lift": None, "errors": errors, "elapsedMs": 0,
    }

    _state["running"] = True
    _state["lastRunAt"] = at
    try:
        runner = distill_fn or _default_distill
        try:
            summary["distill"] = runner(cap, dry_run)
            log_task(_SCOPE, "蒸馏完成", {
                "candidates": summary["distill"].get("candidates"),
                "written": summary["distill"].get("written"),
                "rejected": summary["distill"].get("rejected"),
            })
        except Exception as err:  # noqa: BLE001 —— 失败隔离：蒸馏炸了也别拦着 lift
            errors.append(f"蒸馏失败：{err}")
            log_task_error(_SCOPE, "蒸馏失败", {"error": str(err)})

        if want_lift:
            refresher = lift_fn or knowledge_distill.refresh_lift
            try:
                summary["lift"] = refresher()
                log_task(_SCOPE, "lift 刷新完成", {
                    "judged": summary["lift"].get("judged"),
                    "deprecated": summary["lift"].get("deprecated"),
                })
            except Exception as err:  # noqa: BLE001 —— 同上
                errors.append(f"lift 刷新失败：{err}")
                log_task_error(_SCOPE, "lift 刷新失败", {"error": str(err)})
    finally:
        #: ⚠️ 复位必须在 ``finally``：这是与 ``evaluation_scheduler`` 唯一的**有意差异**
        #: （那一份缺这段，抛错就永久卡死防重入）。少了这一行，一轮异常之后
        #: 定时任务就**静默停摆**，回执里还写着 ``enabled=true``。
        _state["running"] = False
        summary["elapsedMs"] = int((time.monotonic() - started) * 1000)
        summary["ok"] = not errors
        _state["lastResult"] = summary

    if summary["ok"]:
        log_task_success(_SCOPE, "本轮完成")
    return summary


# ---------------------------------------------------------------------------
# 定时
# ---------------------------------------------------------------------------

def _event_loop() -> asyncio.AbstractEventLoop:
    """当前循环（``start()`` 在 lifespan 里被调用；测试里在 ``asyncio.run`` 里）。"""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:  # 没有运行中的循环（脚本式调用 / 单元测试）
        return asyncio.get_event_loop()


def _on_timer() -> None:
    """定时器回调：跑一轮再重新挂上（``call_later`` 是一次性的）。"""
    async def _run_and_rearm() -> None:
        try:
            await run_cycle()
        finally:
            arm()

    asyncio.ensure_future(_run_and_rearm())


def arm() -> None:
    """调度下一次运行（``enabled=false`` 时直接返回，不挂空定时器）。"""
    global _timer
    if not _state["enabled"]:
        return
    #: 延迟与 ISO 格式都取 ``evaluation_scheduler`` 那份现成的（同一套 ``HH:MM`` 语义、
    #: 同一套 ``toISOString`` 形状）；在这里 lazy import 是为了避开启动期的循环依赖。
    from app.agent.evaluation_scheduler import iso_in, next_run_delay

    delay_ms = next_run_delay(_state["schedule"]["hour"], _state["schedule"]["minute"])
    _state["nextRunAt"] = iso_in(delay_ms)
    log_task(_SCOPE, f"下次复盘定于 {_state['nextRunAt']}")
    if _timer is not None:  # 重新 arm 时先撤旧的，否则一个进程里会叠出两个定时器
        _timer.cancel()
    _timer = _event_loop().call_later(delay_ms / 1000, _on_timer)


def stop() -> dict[str, Any]:
    """停掉定时器并清掉 ``nextRunAt``（不存在的"下一轮"比未来时间更难排查）。"""
    global _timer
    if _timer is not None:
        _timer.cancel()
        _timer = None
    _state["nextRunAt"] = None
    return state()


def start() -> dict[str, Any]:
    """启动（应用启动时调用一次）。**默认关**，关着只打一条 warn。

    返回回执而不是 ``None``：启动路径也要能回答「到底有没有跑起来、为什么没跑」。
    """
    result: dict[str, Any] = {"started": False, "why": "", "scheduler": None}
    if not _state["enabled"]:
        log_task_warn(_SCOPE, "未启用（knowledge.distill.enabled=false），跳过")
        result["why"] = "未启用（knowledge.distill.enabled=false）"
        result["scheduler"] = state()
        return result

    hour = _state["schedule"]["hour"]
    minute = _state["schedule"]["minute"]
    log_task(_SCOPE, f"启用每日定时复盘（{hour}:{minute:02d}，limit={_state['schedule']['limit']}"
                     f"，lift={'on' if _state['schedule']['lift'] else 'off'}）")
    arm()
    if _state["schedule"]["runOnStartup"]:
        log_task(_SCOPE, "run_on_startup=true，启动即跑一轮")
        asyncio.ensure_future(run_cycle())
    result.update({"started": True, "why": "", "scheduler": state()})
    return result
