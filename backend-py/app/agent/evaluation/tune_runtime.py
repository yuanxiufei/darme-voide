"""调优的**后台任务表** —— 让 ``/tune`` 从「一次 7 分钟的同步请求」变成「入队 + 查进度」。

本表同时服务**四个目标**（这也是为什么它叫 ``tune_runtime`` 而不是 ``tune_params``）：

* ``target="params"`` ⇒ 温度调优（``param_optimizer.optimize_agent_temperature``）；
* ``target="skills"`` ⇒ 技能绑定调优（``skill_optimizer.optimize_agent_skills``）；
* ``target="tools"`` ⇒ 工具集调优（``tool_optimizer.optimize_agent_tools``，2026-11-08）；
* ``target="system_prompt"`` ⇒ 提示词优化（``optimizer.optimize_agent_prompt``，2026-11-09）。

四者的任务形状一样（多次真模型评测、都能取消、都要进度），差别只有「谁当 runner」与「进度那句话
怎么说」。所以 runner 按 ``target`` 选，别的共用一份 —— 复制一份任务表会让「取消只在两次评测之间
生效」「任务自己开连接」这些口径迟早一边改一边没改。

⚠️ **别把这句读成「四格都是 ``repeat × (1 + 候选数)`` 跑」**（2026-11-09 更正：原文就是这么写的，
提示词格照抄它就会报出 2 倍的分母）：``repeat`` 在四格里**语义不同** —— 温度/技能两格是「同一档位
重跑几次」；``tools`` 每候选只评一次、抖动取自候选自己那 ``repeat`` 个样本，基线只跑 1 次；
``system_prompt`` 的 ``repeat`` 是**Reference 复测次数**（默认 1 ＝不测抖动带＝零容差，2 起才用
样本极差），候选每轮只评一次 ⇒ 它的分母是 ``repeat + iterations``（2026-11-10 起这一格也有
``repeat`` 了，旧注释「连 repeat 这一格都没有」已作废）。所以入队时那条 ``totalRuns`` 只是
**上限估**，真值一律由优化器的 ``start`` / ``baseline`` 事件下发；`/tune-prompt` 路由会传
``total_runs`` 把这一格的估法一次算准（别在 ``submit`` 里按 target 分叉算术）。

⚠️ 四个目标的**基线跑法**并不一样，这点由各自优化器负责，任务表不掺和：温度那格的基线是
「现在 DB 里那个温度」，技能那格是「现在 DB 里那份绑定」，工具集那格**不传覆盖**（= 该 Agent
的全部工具，因为工具集的真源是代码而不是配置列），提示词那格是「**运行时在用的那份**提示词
（DB 里的 `agent_configs.system_prompt` 优先，没有配置行才回退出厂默认）」且**会真的评测一次**
（所以那 1 跑要算进分母）。

⚠️ ``system_prompt`` 这条路是**新增的入队口**，``POST /optimize/{case_id}`` 那个同步端点只做了
一处同步补齐（基线同样改成 DB 优先）：两条走的是**同一个** ``optimize_agent_prompt``，区别只在
投递方式。同步那条留给 CLI/脚本（不受浏览器/代理超时约束）；要进度、要取消、怕代理超时就走这里。

⚠️ **去重键必须带上 ``target``**：同一个 (agent, case) 上「正在调温度」与「正在调绑定」是两件
不同的事，只按 (agent, case) 去重会把刚提交的技能调优当成温度任务的重复而**静默丢弃**
（回一个别人的 taskId，用户看到的是「温度」的进度与回执）。

要解决的问题
============
``param_optimizer.optimize_agent_temperature`` 是 ``repeat × (1 + 候选数)`` 次**真模型评测**。
真机实测（``_bench/tune_probe.py``，2026-11-07）：eval-case-001 / repeat=2 / 1 个候选，
单跑 85–165 秒、四跑合计 **423.9 秒**。把它挂在 ``POST`` 的同步返回链上意味着：

1. 前端 fetch 不设超时会一直等，**中间代理/网关的超时却会先到** ⇒ 用户看到的是失败，
   而后台其实还在烧卡（这是最难查的一种：日志显示跑完了，界面显示超时）；
2. 这 7 分钟里没有任何中间态 —— 分不清「在跑第 1 跑」还是「卡死了」；
3. 用户点两次就排队跑两轮 7 分钟（同一个 case、同一份提示词，纯浪费）。

所以：**入队立刻返回 ``taskId``**，进度靠 ``GET /tune/{taskId}`` 拿，跑完的回执也在那里。

三条口径（都不是可选的）
========================
* **单工作线程 + 队列**：这台机器只有一张卡，两个调优并发跑只会互相抢显存、两个都变慢甚至 OOM。
  串行是唯一诚实的并发度 —— 排队就明说在排队（``queuePosition``）。
* **取消只在两次评测之间生效**：一次评测内部的真模型生成（85–165 秒）在别的线程里跑采样循环，
  从外面掐不断（见 ``engine.llm.generate_llm`` 的 ``abort`` 口径）。所以 ``cancel`` 的语义是
  **「不再开下一跑」**，不是「立刻停」。宁可这么写，也不假装能秒停。
* **重连自己的 DB 连接**：任务跑在一个**新的线程**里，而 ``get_conn`` 那条连接的生命周期跟着
  **HTTP 请求**走 —— 请求一返回连接就归还了。拿它继续跑后台任务，症状是任务跑到一半
  ``connection is closed``。故工作线程自己 ``engine.connect()``（每次任务一条，跑完就关）。

⚠️ 任务表是**进程内存**：进程重启后 ``GET /tune/{taskId}`` 就是 404。这是刻意的 —— 调优的产物
（判决与落库）已经写进 ``agent_optimizations`` / ``agent_configs``，重启后要查的「上次调成什么」
去那里看，本表只负责「这次跑到哪了」。
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from typing import Any, Awaitable, Callable, Mapping

from app.core import db as db_module
from app.agent.evaluation.optimizer import optimize_agent_prompt
from app.agent.evaluation.param_optimizer import TuneCancelled, optimize_agent_temperature
from app.agent.evaluation.skill_optimizer import optimize_agent_skills
from app.agent.evaluation.tool_optimizer import optimize_agent_tools

__all__ = [
    "TARGETS",
    "TERMINAL",
    "cancel",
    "get",
    "list_tasks",
    "reset",
    "submit",
    "wait",
]

#: 终态：到这四态之后不会再变（``queued``/``running`` 之外的都算）
TERMINAL = ("succeeded", "failed", "cancelled")

#: 本表认的调优目标（与 ``optimization_log.TARGETS`` 的落库值同名，别另起一套叫法）
TARGETS: tuple[str, ...] = ("params", "skills", "system_prompt", "tools")

#: 最多留多少个任务（进程内存不许无限长 —— 掉的是**最老的已结束**那批）
_MAX_TASKS = 50

#: 任务快照里对外暴露的键（内部 ``_`` 开头的一律不出去：``_abort`` 只给 worker 看）
_RUNNER = "runner"

#: 一次任务要连的库连接（每次新开，见模块头第 3 条）
Runner = Callable[[Any, dict[str, Any]], Awaitable[dict[str, Any]]]

_tasks: dict[str, dict[str, Any]] = {}
_queue: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None
_lock = threading.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _stage_text(event: Mapping[str, Any], target: str = "params") -> str:
    """把进度事件翻成一句人话（前端就算不解析字段，显示这一句也不会误解）。

    ⚠️ 必须按 ``target`` 分流：四个目标的 ``run`` / ``candidate`` 事件带着**不同的键**
    （温度是 ``temperature``，技能绑定是 ``skill``，工具集是 ``tool``，提示词是 ``version`` 与
    ``phase``）。写成温度专用时，技能任务会显示「温度 None」—— 它不是崩溃，所以没人会去修，
    但后台页从此看不出在干什么；工具集那格更隐蔽：它的候选要**基线跑完**才存在，`start` 事件里
    根本没有 ``candidates``；提示词那格则是**每轮只跑 1 次**，照温度格式印「第 1/1 跑」会让人
    以为还有重复档没跑完。
    """
    kind = str(event.get("type") or "")
    done, total = event.get("doneRuns"), event.get("totalRuns")
    if kind == "start":
        if target == "system_prompt":
            # 本格的 ``repeat`` 只加在 Reference 上（候选每轮只评 1 次）⇒ 写清「复测 N 次」
            repeat = int(event.get("repeat") or 1)
            baseline = (f"Reference 复测 {repeat} 次取抖动带" if repeat > 1 else "先评 Reference")
            return (f"开始：共 {total} 跑（{baseline}，再逐轮改提示词，"
                    f"共 {event.get('iterations')} 轮）")
        if target == "skills":
            return (f"开始：共 {total} 跑（当前绑定 {len(event.get('current') or [])} 个 skill，"
                    f"候选 {len(event.get('candidates') or [])} 个，每个 {event.get('repeat')} 次）")
        if target == "tools":
            # 工具集的候选要等基线跑完才知道（候选源是轨迹），所以这里只说「几跑」
            return (f"开始：共 {total} 跑（先跑基线取工具集，每个候选 {event.get('repeat')} 次）")
        return f"开始：共 {total} 跑（每档 {event.get('repeat')} 次）"
    if kind == "run":
        if target == "system_prompt":
            # 候选每轮只跑 1 次 ⇒ 不印「第 1/1 跑」（那会让人以为还有重复档没跑完）；
            # Reference 在 ``repeat>1`` 时才真是多跑，这时候才该写出第几跑。
            if str(event.get("phase") or "") == "reference":
                repeat = int(event.get("repeat") or 1)
                if repeat > 1:
                    return (f"评 Reference（v0）：第 {event.get('run')}/{repeat} 跑"
                            f"（总 {done}/{total}）")
                return f"评 Reference（v0）：总 {done}/{total}"
            return (f"第 {event.get('run')} 轮候选（v{event.get('version')}）："
                    f"评测中（总 {done}/{total}）")
        if event.get("phase") == "baseline":
            phase = "基线"
        elif target == "skills":
            phase = f"摘掉 {event.get('skill')}"
        elif target == "tools":
            phase = f"摘掉 {event.get('tool')}"
        else:
            phase = f"温度 {event.get('temperature')}"
        return f"{phase}：第 {event.get('run')}/{event.get('repeat')} 跑（总 {done}/{total}）"
    if kind == "baseline":
        if target == "system_prompt":
            # 抖动带**按 `jumpKind` 分支**：`none` = 没测（repeat=1，零容差），不是「抖动带 0」；
            # `within` = 同配置那几次测量的极差（repeat≥2）。别把两者印成同一句话。
            # `baselineSource` 说清这分数是「现在在跑的那份」还是「出厂默认」：同一句话里
            # 不写，用户会以为测的总是默认（2026-11-09 基线改成 DB 优先之后尤其要写明）
            where = {"config": "基线＝当前生效的提示词",
                     "default": "基线＝出厂默认（该 Agent 还没保存过提示词）"}.get(
                         str(event.get("baselineSource") or ""), "")
            if str(event.get("jumpKind") or "none") == "within":
                rule = (f"抖动带 {event.get('jump')}（同配置 {event.get('repeat')} 次测量的极差），"
                        f"候选要超过它才采纳")
            else:
                rule = "本格没测抖动带（repeat=1），候选严格更高才采纳"
            return (f"Reference 完成：{event.get('score')}"
                    f"（{where + '；' if where else ''}{rule}）")
        if target == "tools":
            # 工具集的基线事件带的是 score / tools / steps（**没有** mean/jump：抖动带要等候选跑完）
            steps = event.get("steps") or {}
            if steps.get("unavailable"):
                # 缺明细时 `tools` 必然是空的（`current_tools` 读不到账）⇒ 只写「出现 0 个工具」
                # 会把**读数缺失**说成「模型没调过任何工具」，两个事实正相反
                return (f"基线完成：{event.get('score')}，但逐步账没有明细"
                        f"（{steps.get('unavailable')}）⇒ 判不了，理由见回执")
            return (f"基线完成：{event.get('score')}，轨迹里出现 "
                    f"{len(event.get('tools') or [])} 个工具")
        return f"基线完成：均值 {event.get('mean')}，抖动带 {event.get('jump')}"
    if kind == "candidate":
        record = event.get("record") or {}
        why = record.get("reasonLabel") or record.get("reason")
        if target == "system_prompt":
            return f"v{record.get('version')}：{why}"
        if target == "skills":
            return f"摘掉 {record.get('skill')}：{why}"
        if target == "tools":
            if why:
                return f"摘掉 {record.get('tool')}：{why}"
            # 判据尚未出来的那次（每候选刚跑完）—— 报告均值与逐步账，别印出 "：None"
            return (f"摘掉 {record.get('tool')}：均值 {record.get('mean')}"
                    f"（unknownTool {record.get('unknownTool')} 次）")
        return f"温度 {record.get('temperature')}：{why}"
    if kind == "done":
        return "已结束（回执见 receipt）"
    return "运行中"


def _snapshot(job: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(job.get("taskId") or "")
    out: dict[str, Any] = {
        "taskId": task_id,
        "caseId": job.get("caseId"),
        "agentType": job.get("agentType"),
        #: ``params``（温度）/ ``skills``（技能绑定）/ ``tools``（工具集）——
        #: 前端靠它决定「哪一格被这次调优动了」，也决定该去读哪个优化史目标
        "target": job.get("target") or "params",
        "status": job.get("status"),
        "stage": job.get("stage"),
        "createdAt": job.get("createdAt"),
        "startedAt": job.get("startedAt"),
        "finishedAt": job.get("finishedAt"),
        "repeat": (job.get("options") or {}).get("repeat"),
        "iterations": (job.get("options") or {}).get("iterations"),
        "maxCandidates": (job.get("options") or {}).get("maxCandidates"),
        "autoPersist": (job.get("options") or {}).get("autoPersist"),
        "doneRuns": job.get("doneRuns") or 0,
        "totalRuns": job.get("totalRuns") or 0,
        "progress": job.get("progress"),
        "pollUrl": f"/api/v1/evaluation/tune/{task_id}",
    }
    if job.get("receipt") is not None:
        out["receipt"] = job["receipt"]
    if job.get("error"):
        out["error"] = job["error"]
    return out


def _queue_position(task_id: str) -> int:
    """排在它前面还有几个等着跑（``0`` = 下一个就是它；跑着的那个不算排队）。"""
    position = 0
    for other in _tasks.values():
        if other.get("taskId") == task_id:
            continue
        if other.get("status") == "queued" and int(other.get("createdAt") or 0) < \
                int(_tasks.get(task_id, {}).get("createdAt") or 0):
            position += 1
    return position


def _trim() -> None:
    """超出上限时丢掉**最老的已结束**任务（队列中/在跑的绝不丢）。"""
    if len(_tasks) <= _MAX_TASKS:
        return
    finished = [
        item for item in _tasks.values() if item.get("status") in TERMINAL
    ]
    finished.sort(key=lambda item: int(item.get("createdAt") or 0))
    for item in finished[: len(_tasks) - _MAX_TASKS]:
        _tasks.pop(str(item.get("taskId")), None)


# ---------------------------------------------------------------------------
# 工作线程
# ---------------------------------------------------------------------------


def _ensure_worker() -> None:
    """第一次提交时才起线程（import 这个模块不该顺手起线程）。"""
    global _worker
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(target=_loop, name="tune-worker", daemon=True)
        _worker.start()


def _loop() -> None:
    while True:
        task_id = _queue.get()
        job = _tasks.get(task_id)
        if job is None:
            continue
        if job.get("status") == "cancelled":
            continue        # 排队期间就被取消了 ⇒ 连跑都不跑
        _execute(job)


def _execute(job: dict[str, Any]) -> None:
    job["status"] = "running"
    job["startedAt"] = _now_ms()
    job["stage"] = "正在装载评测上下文"
    target = str(job.get("target") or "params")
    runner: Runner = job.get(_RUNNER) or _runner_for(target)

    def progress(event: dict[str, Any]) -> None:
        job["progress"] = dict(event)
        if event.get("doneRuns") is not None:
            job["doneRuns"] = int(event["doneRuns"])
        if event.get("totalRuns") is not None:
            job["totalRuns"] = int(event["totalRuns"])
        job["stage"] = _stage_text(event, target)

    try:
        # ⚠️ 新连接：请求那条连接在响应返回时就归还了（模块头第 3 条）。
        # ⚠️⚠️ 必须是 ``begin()`` 而不是 ``connect()``：SQLAlchemy 2.0 下 ``connect()`` 里的写入
        # 在 ``close()`` 时**一律回滚**（2026-11-08 实测：`with engine.connect() as c` 里 INSERT 一行，
        # 另开连接读回 `rows = 0`）。而优化器整轮做的事就是「写优化史 + 采纳后 UPDATE 配置」——
        # 用 ``connect()`` 等于**回执写 persisted=True，库里一个字没有**：页面说生效了、线上还是旧值，
        # 且没有任何报错。这条路径的温度版从入队改造（2026-11-07）起就是这样，全程静默。
        with db_module.engine.begin() as conn:
            job["progressCallback"] = progress
            receipt = asyncio.run(runner(conn, job))
    except TuneCancelled as err:
        job["status"] = "cancelled"
        job["error"] = str(err)
        job["stage"] = f"已取消（{err}）"
    except Exception as err:  # noqa: BLE001 —— 任何失败都要落到任务表上，不许只打日志
        job["status"] = "failed"
        job["error"] = f"{err.__class__.__name__}: {err}"
        job["stage"] = f"失败：{job['error']}"
    else:
        job["status"] = "succeeded"
        job["receipt"] = receipt
        job["stage"] = "已完成（回执见 receipt）"
    finally:
        job["finishedAt"] = _now_ms()
        job.pop("progressCallback", None)
        _trim()


async def _default_runner(conn: Any, job: dict[str, Any]) -> dict[str, Any]:
    """真正跑温度调优：把进度/取消接进优化器（优化器本身不认识队列）。"""
    return await optimize_agent_temperature(
        conn,
        str(job["agentType"]),
        job["caseDef"],
        job.get("options") or {},
        progress=job.get("progressCallback"),
        should_stop=lambda: bool(job.get("abort")),
    )


async def _skills_runner(conn: Any, job: dict[str, Any]) -> dict[str, Any]:
    """真正跑技能绑定调优（与上面逐条同构 —— 两个优化器的进度/取消签名本来就是一份口径）。"""
    return await optimize_agent_skills(
        conn,
        str(job["agentType"]),
        job["caseDef"],
        job.get("options") or {},
        progress=job.get("progressCallback"),
        should_stop=lambda: bool(job.get("abort")),
    )


async def _tools_runner(conn: Any, job: dict[str, Any]) -> dict[str, Any]:
    """真正跑工具集调优（与上面逐条同构）。

    ⚠️ 本格**没有** ``autoPersist`` 可传：工具集没有配置列，结论只落优化史
    （见 ``tool_optimizer`` 模块头最后一段）。``optimize_agent_tools`` 会把开关钉成 False。
    """
    return await optimize_agent_tools(
        conn,
        str(job["agentType"]),
        job["caseDef"],
        job.get("options") or {},
        progress=job.get("progressCallback"),
        should_stop=lambda: bool(job.get("abort")),
    )


async def _prompt_runner(conn: Any, job: dict[str, Any]) -> dict[str, Any]:
    """真正跑提示词优化（与上面三条逐条同构）。

    ⚠️ 本格的 ``repeat`` 是 **Reference 复测次数**（候选仍每轮只评一次）⇒ 分母是
    ``repeat + iterations``，不是 ``repeat × (1 + n)``（照抄温度格会报出 2 倍分母）。
    ⚠️ ``autoPersist`` 与同步端点同义（默认 True）：best **严格高于 Reference + 抖动带** 才写进
    ``agent_configs`` —— 本格是四格里**唯一会真改配置**的一格，别照工具集那格理解成只记历史。
    """
    return await optimize_agent_prompt(
        conn,
        str(job["agentType"]),
        job["caseDef"],
        job.get("options") or {},
        progress=job.get("progressCallback"),
        should_stop=lambda: bool(job.get("abort")),
    )


def _runner_for(target: str) -> Runner:
    """按目标选默认 runner（不认识的目标 ⇒ 当场拒，别静默退成温度）。"""
    if target == "skills":
        return _skills_runner
    if target == "tools":
        return _tools_runner
    if target == "system_prompt":
        return _prompt_runner
    if target == "params":
        return _default_runner
    raise ValueError(f"未知调优目标：{target!r}（只认 {'/'.join(TARGETS)}）")


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------


def submit(
    *,
    case_id: str,
    case_def: Mapping[str, Any],
    agent_type: str,
    options: Mapping[str, Any],
    target: str = "params",
    candidate_count: int | None = None,
    total_runs: int | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """入队一次调优 ⇒ 任务快照（含 ``taskId`` / ``queuePosition`` / ``pollUrl``）。

    ``target``：``params``（温度，默认）/ ``skills``（技能绑定）/ ``tools``（工具集）/
    ``system_prompt``（提示词）。它决定**谁当 runner**，也决定进度那句话怎么说
    （见 :func:`_stage_text`）。

    ``candidate_count``：可选，**只用于入队那一刻的 ``totalRuns`` 预估**。技能绑定的候选数
    取决于当前绑定里有几个 skill（可能比 ``maxCandidates`` 少），路由手里有库连接能算准；
    提示词格的候选数就是 ``iterations``（每轮评一次）；不传就按 ``maxCandidates`` 估。
    估大了不致命 —— 优化器发事件时会把它校正成真值。
    ⚠️ 下面这个式子默认「基线也跑 ``repeat`` 次」（温度/技能两格如此），而**工具集那格的基线
    只跑 1 次**、候选数还要等基线跑完才知道 ⇒ 对 ``target="tools"`` 它是个偏大的**上限估**，
    真值由优化器 ``baseline`` 事件下发（``1 + repeat × 真候选数``）；别在这里按 target 分叉，
    那是把「本格跑几跑」的知识复制到第二处。
    ``total_runs``：可选，**直接给总跑数**（真值已知时用）。存在的理由只有一个 ——
    ``system_prompt`` 那格的分母是 ``repeat + iterations``（``repeat`` 只加在 Reference 上），
    而下面那条通式是 ``repeat × (1 + 候选数)``（``repeat`` 加在每一档上）。两种 ``repeat``
    语义不同，**这个差别不属于本模块**（本模块不该知道「某格跑几跑」），所以由调用方算好传进来，
    而不是在这里按 target 分叉算术。不传时按通式估（只影响入队那一刻的显示，真值由 ``start``
    事件校正）。

    ⚠️ ``system_prompt`` 的 ``repeat`` 缺省按 1 兜底（写 0 会让入队快照显示 ``0/0``）。

    ⚠️ **同一个 (agentType, caseId, target) 已在排队或正在跑 ⇒ 不重复入队**，直接把它还回去并标
    ``deduped=True``：同一个 case、同一份提示词，跑第二遍只是再烧 7 分钟。
    参数不同也不另起 —— 真想换参数，先取消手上这个（``POST /tune/{taskId}/cancel``）。
    ⚠️ 去重键**必须带 ``target``**：温度与技能绑定是两件不同的事，只按 (agent, case) 去重会把
    刚提交的技能调优当成温度任务的重复而静默丢弃。
    """
    if target not in TARGETS:
        raise ValueError(f"未知调优目标：{target!r}（只认 {'/'.join(TARGETS)}）")

    for item in _tasks.values():
        if item.get("status") in ("queued", "running") \
                and item.get("agentType") == agent_type \
                and item.get("caseId") == case_id \
                and (item.get("target") or "params") == target:
            return {**_snapshot(item), "deduped": True,
                    "queuePosition": _queue_position(str(item.get("taskId")))}

    expected = (int(candidate_count) if candidate_count is not None
                else int(options.get("maxCandidates") or 0))
    task_id = f"tune-{uuid.uuid4().hex[:12]}"
    job: dict[str, Any] = {
        "taskId": task_id,
        "caseId": case_id,
        "agentType": agent_type,
        "target": target,
        "caseDef": dict(case_def),
        # ⚠️ 保留键是**白名单**：少列一个键 ⇒ runner 里读到的是默认值，任务照跑不报错。
        #    `iterations` 属提示词格（漏了它就会永远按默认 3 轮跑，用户传 5 也没人理）。
        "options": {k: options[k] for k in
                    ("repeat", "maxCandidates", "autoPersist", "iterations")
                    if k in options},
        "status": "queued",
        "stage": "排队中",
        "createdAt": _now_ms(),
        "startedAt": None,
        "finishedAt": None,
        "doneRuns": 0,
        # 真值已知就照它写（提示词格 `repeat + iterations` 这类「repeat 语义不同」的分母）；
        # 否则按通式估。`repeat` 缺省按 1 —— 按 0 算会让快照永远显示 0/0。
        "totalRuns": (max(0, int(total_runs)) if total_runs is not None
                      else int(options.get("repeat") or 1) * (1 + max(0, expected))),
        "progress": None,
        "receipt": None,
        "error": None,
        "abort": False,
        "deduped": False,
    }
    if runner is not None:
        job[_RUNNER] = runner
    _tasks[task_id] = job
    _ensure_worker()
    _queue.put(task_id)
    return {**_snapshot(job), "deduped": False, "queuePosition": _queue_position(task_id)}


def get(task_id: str) -> dict[str, Any] | None:
    """任务快照（不认识 ⇒ ``None``，由路由翻成 404）。"""
    job = _tasks.get(str(task_id or ""))
    return _snapshot(job) if job is not None else None


def list_tasks(*, limit: int = 20) -> list[dict[str, Any]]:
    """最近的调优任务（新的在前；不带回执正文，要看点开 ``GET /tune/{taskId}``）。"""
    items = sorted(_tasks.values(), key=lambda item: int(item.get("createdAt") or 0),
                   reverse=True)
    return [_snapshot(item) for item in items[: max(0, int(limit))]]


def cancel(task_id: str) -> dict[str, Any]:
    """取消 ⇒ 快照（不认识 ⇒ ``None``）。

    ``queued`` ⇒ 立刻是 ``cancelled``（worker 排到它时直接跳过）；
    ``running`` ⇒ 置标志，**下一次评测之前**才停（见模块头第 2 条）；
    终态 ⇒ 原样返回（``changed: False``，不报错 —— 「已经结束了」不是失败）。
    """
    job = _tasks.get(str(task_id or ""))
    if job is None:
        return None
    before = job.get("status")
    changed = False
    if before == "queued":
        job["status"] = "cancelled"
        job["stage"] = "已取消（还没开跑）"
        job["finishedAt"] = _now_ms()
        changed = True
    elif before == "running":
        job["abort"] = True
        job["stage"] = "取消中：跑完当前这一跑就停"
        changed = True
    return {**_snapshot(job), "changed": changed, "wasStatus": before}


def wait(task_id: str, *, timeout: float = 900.0, poll: float = 0.2) -> dict[str, Any] | None:
    """等到终态（给 ``wait=true`` 的同步调用方用；超时 ⇒ 返回当时快照并标 ``timedOut``）。"""
    deadline = time.time() + max(0.0, float(timeout))
    while True:
        job = _tasks.get(str(task_id or ""))
        if job is None:
            return None
        if job.get("status") in TERMINAL:
            return _snapshot(job)
        if time.time() >= deadline:
            return {**_snapshot(job), "timedOut": True}
        time.sleep(max(0.02, float(poll)))


def reset() -> None:
    """清空任务表（**测试专用**：让每个用例从空表开始；不影响已起的 worker 线程）。"""
    _tasks.clear()
