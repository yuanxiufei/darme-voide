"""参数优化器 —— 知识闭环 §1 缺口 4 的**第一格**：优化「生成参数」而不是提示词。

为什么单开一个模块
==================
提示词优化器（``optimizer.py``）的候选是 LLM 改写的**文本**，一次一版、天然有父子链。参数不一样：
候选是**几个数**，全都得拿同一份提示词、同一个 case 跑一遍才比得出高低 ⇒ 它是「一批平级候选」
而不是「一条演化链」。混在提示词优化器里，那套 version/parent 语义会立刻失真。

为什么只调 ``temperature`` 一格
==============================
分数是**确定性判分**（维度命中 + 格式），能由分数驱动的旋钮只有一个方向：

* ``max_tokens`` 调小 ⇒ 工具调用被截断，分数掉了不是模型变差 ✗；调大只烧钱 ✗。
  它该由成本预算定，不该被分数推着走；
* ``max_iterations`` 同理 —— 调小会静默截断工具链（表现为「该调的第二个工具没调」⇒ 像是模型
  变笨），调大只是多花钱 ✗；
* ``temperature`` 是唯一「同一份提示词、同一个 case、分数会真变」的旋钮，且在确定性判分下方向
  单一：**更确定 ⇒ 格式漂移更少**。故候选只往更低温度扫，**不往高扫**（往高温试只在模型卡在
  重复循环时有用，那种情况分数落差会远超抖动带，属于改提示词该管的事，不是调参该管的 ✗）。

三条硬约束（都为了「别把噪声当成提升」）
========================================
1. **单次评测不许拿来比参数**：LLM 每次输出不同 ⇒ ``repeat < 2`` 直接拒（抛 ``ValueError``），
   比较的是**均值**；
2. **涨的那点必须大于抖动带**：``delta > (基线最高分 − 基线最低分)`` 才算提升，否则记
   ``reason="below_noise"``（「涨了但没超过抖动带」）—— 这是参数调优最典型的假绿：
   同一档位重跑两次差 5 分，就敢说「换温度提升了 5 分」✗；
3. **同模型才可比**：候选的 ``runtimeModel`` 与基线不一致 ⇒ 记 ``runtime_mismatch``、不采纳
   （与提示词优化器同一口径）。

落库口径
========
采纳后只 update ``agent_configs.temperature`` 那一列（``persist_agent_temperature``）。**没有配置
行就不写**（返回 0）⇒ 此时照实记 ``persisted=None`` ＋ 理由，绝不为了「让优化看起来生效了」去插
一行只带温度的配置（那会把该 agent 的提示词留空）。这正是 ``accepted`` 与 ``persisted`` 分成两个
布尔的意义。
"""
from __future__ import annotations

import hashlib
from typing import Any, Awaitable, Callable, Mapping

from sqlalchemy import func, select
from sqlalchemy.engine import Connection

from app.core.models import agent_configs, agent_optimizations
from app.services import optimization_log
from app.services.agent_prompts import get_default_instructions
from app.agent.creator import persist_agent_temperature
from app.agent.runtime import DEFAULT_TEMPERATURE
from app.agent.evaluation.evaluator import evaluate_case

__all__ = [
    "MAX_REPEAT",
    "TEMPERATURE_GRID",
    "TuneCancelled",
    "current_temperature",
    "normalize_options",
    "optimize_agent_temperature",
    "temperature_candidates",
]


class TuneCancelled(Exception):
    """调优被调用方中止（**只在两次评测之间生效** —— 见 :mod:`tune_runtime` 的取消口径）。

    不继承 ``ValueError``：取消是**正常收场**，不是参数错 ⇒ 别让它走进 ``bad_request`` 那条
    400 通路（那里会把「用户自己按了取消」报成「你的参数不对」✗）。
    """


#: 进度回调：**同步**、**不许抛**（簿记动作拖垮调优是本末倒置，实现里已自吞异常 ✓）
ProgressFn = Callable[[dict[str, Any]], None]

#: 温度候选网格（升序）。**只取严格小于当前值的档**，故网格本身就是「往确定性方向」的梯子。
TEMPERATURE_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

#: 同一档位最多重跑几次。上限存在的理由：``repeat × (1 + 候选数)`` 次真·LLM 评测，
#: 参数填个 50 就是一次几百次评测的长跑 —— 那是无人值守任务，不该被一个查询参数触发。
MAX_REPEAT = 5

#: 默认每档重跑 2 次：够看出抖动带（1 次连抖动都看不见 ⇒ 约束 2 形同虚设）。
DEFAULT_REPEAT = 2

#: 默认最多试 2 个更低档：**最近的先试**（离当前值越远，风险越大、翻盘概率越小）。
DEFAULT_MAX_CANDIDATES = 2

#: 评测函数签名（默认 :func:`evaluate_case`；测试注入假实现，免得真拉模型）。
EvaluateFn = Callable[..., Awaitable[dict[str, Any] | None]]


def temperature_candidates(current: Any, *, max_candidates: int = DEFAULT_MAX_CANDIDATES) -> list[float]:
    """当前温度之下的候选（离当前最近的先试），**绝不包含当前值本身、也不含更高值**。

    已是最确定档（0.0）时返回空列表 —— 空列表不是错误，是「没有更低的档可试」这个事实
    （调用方据此直接收工，不去跑一轮注定没有候选的评测）。
    """
    try:
        cur = float(current)
    except (TypeError, ValueError):
        return []
    if max_candidates <= 0:
        return []
    below = [t for t in TEMPERATURE_GRID if t < cur - 1e-9]
    below.sort(reverse=True)  # 最近的先试
    return below[:max_candidates]


def current_temperature(conn: Connection, agent_type: str) -> float:
    """当前生效温度：``agent_configs.temperature`` 优先，没有配置行/列为空则用运行期默认值。

    与 ``runtime.run_agent_with_instructions`` 读的是同一格 —— 基线必须拿**真正会生效**的那个值跑，
    否则「基线」是编的。
    """
    row = conn.execute(
        select(agent_configs.c.temperature).where(agent_configs.c.agent_type == agent_type)
    ).first()
    if row is not None and row[0] is not None:
        return float(row[0])
    return float(DEFAULT_TEMPERATURE)


def _sha16(text: str) -> str:
    """提示词指纹（与 ``optimization_log`` 同算法：sha1 前 16 位）。

    参数候选**共用同一份提示词**，所以这一轮所有行的指纹都一样 —— 这不是巧合，是这条通路
    唯一的防混淆凭据：页面上若看到指纹不一致，说明「固定提示词」这个前提被破坏了。
    """
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


def _next_version(conn: Connection, agent_type: str, case_id: str) -> int:
    """下一个 ``version``：取该 (agent, case) 已存在的**最大 version + 1**。

    为什么跨目标取 max（而不是按 ``target`` 各自从 1 数）：``mark_persisted`` 是按
    (agent_type, version, case_id) 定位行的，两套编号撞上就会把「参数采纳」标到「提示词那行」
    头上 —— 判决与生效错位，正是这张表最不能出的错。
    """
    current = conn.execute(
        select(func.max(agent_optimizations.c.version)).where(
            agent_optimizations.c.agent_type == agent_type,
            agent_optimizations.c.case_id == case_id,
        )
    ).scalar()
    return int(current or 0) + 1


def _effective_instructions(conn: Connection, agent_type: str) -> str:
    """调参时**固定不动**的那份提示词：DB 里的当前值优先，没有配置行就用默认提示词。"""
    row = conn.execute(
        select(agent_configs.c.system_prompt).where(agent_configs.c.agent_type == agent_type)
    ).first()
    if row is not None and row[0]:
        return str(row[0])
    return get_default_instructions(agent_type)


def normalize_options(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """把调用方给的 ``options`` 收成**唯一一份**校验过的形状。

    为什么单独一个函数：入队（路由）与执行（后台线程）隔着一条队列，两边各写一遍校验迟早分叉。
    分叉的两种表现都很难查 —— 入队时 400 的写法排到了才炸（用户以为提交成功了），或者入队放行、
    跑到一半才抛（前面烧掉的那几次评测白跑）。所以：**校验只此一处**，路由在入队前就调它，
    优化器开头再调一次（直接调库的人也不会绕过）。

    返回的键固定三个：``repeat`` / ``maxCandidates`` / ``autoPersist``（``historyDir`` 之类的
    其它键一律**不认** —— 参数候选没有提示词全文要落盘）。
    """
    opts = dict(options or {})
    repeat = opts.get("repeat") if opts.get("repeat") is not None else DEFAULT_REPEAT
    if not isinstance(repeat, int) or isinstance(repeat, bool):
        raise ValueError(f"repeat 必须是整数，收到 {type(repeat).__name__}")
    if repeat < 2:
        raise ValueError(
            f"repeat 必须 ≥ 2（收到 {repeat}）：单次评测的结果差异里分不清「换温度了」与"
            f"「这次恰好抖高了」，一次就下结论等于把噪声当提升"
        )
    if repeat > MAX_REPEAT:
        raise ValueError(f"repeat 最多 {MAX_REPEAT}（收到 {repeat}）：这是 repeat×候选数 次真模型评测")

    max_candidates = opts.get("maxCandidates")
    if max_candidates is None:
        max_candidates = DEFAULT_MAX_CANDIDATES
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates < 0:
        raise ValueError(f"maxCandidates 必须是非负整数，收到 {max_candidates!r}")
    auto_persist = opts["autoPersist"] if opts.get("autoPersist") is not None else True
    return {"repeat": repeat, "maxCandidates": int(max_candidates),
            "autoPersist": bool(auto_persist)}


def _notify(progress: ProgressFn | None, **event: Any) -> None:
    """发一条进度（**永不抛** —— 订阅方写崩了不许把正在跑的调优带走）。"""
    if progress is None:
        return
    try:
        progress(event)
    except Exception:  # noqa: BLE001
        pass


async def optimize_agent_temperature(
    conn: Connection,
    agent_type: str,
    case_def: dict[str, Any],
    options: dict[str, Any] | None = None,
    *,
    evaluate: EvaluateFn | None = None,
    progress: ProgressFn | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """在**同一份提示词**上扫更低的温度，按「均值涨幅 > 抖动带」判采纳，采纳才落库。

    ``options``：

    * ``repeat``（默认 2，必须 ≥2 且 ≤ :data:`MAX_REPEAT`）—— 每档重跑几次；
    * ``maxCandidates``（默认 2）—— 最多试几个更低档；
    * ``autoPersist``（默认 True）—— 采纳后是否写回 ``agent_configs.temperature``；
    * ``historyDir`` 等其它键**不认**（参数候选没有提示词全文要落盘，历史就是优化史那张表）。

    ``progress``（可选）：每跑一次评测前后各发一条 dict（``type``/``doneRuns``/``totalRuns`` 等），
    给后台任务表拿去画进度 —— **同步、不许抛**（:func:`_notify` 已自吞）。

    ``should_stop``（可选）：每跑之前问一次「还要继续吗」；为真就抛 :class:`TuneCancelled`。
    粒度就是「两次评测之间」—— 一次评测内部的 100 秒真模型生成**无法**从外面掐断
    （采样循环在别的线程里，见 ``engine.llm.generate_llm`` 的 ``abort`` 口径），
    所以取消的语义是「不再开下一跑」，不是「立刻停」。

    返回回执（可直接打印/返回 HTTP）：基线样本与均值、抖动带、每个候选的样本与 Δ、采纳了谁、
    落库成没成（含没成的理由）。
    """
    opts = normalize_options(options)
    repeat = opts["repeat"]
    max_candidates = opts["maxCandidates"]
    auto_persist = opts["autoPersist"]

    run_eval: EvaluateFn = evaluate or evaluate_case
    case_id = str(case_def.get("id") or "")
    instructions = _effective_instructions(conn, agent_type)
    prompt_sha = _sha16(instructions)
    base_temp = current_temperature(conn, agent_type)
    candidates = temperature_candidates(base_temp, max_candidates=max_candidates)

    print(f"\n===== 参数调优：温度（agent={agent_type}, case={case_id}）=====")
    print(f"当前温度 {base_temp}，候选 {candidates or '（已是 0.0，没有更低的档）'}"
          f"，每档跑 {repeat} 次（提示词指纹 {prompt_sha}）")

    # 进度口径：**早就定死**总跑数（基线 + 每个候选各 repeat 次）⇒ 前端能画出「第 3/6 跑」，
    # 而不是只能转圈。取消同理：只在两次评测之间生效，故每跑开头问一次。
    total_runs = repeat * (1 + len(candidates))
    done_runs = 0
    _notify(progress, type="start", caseId=case_id, agentType=agent_type, repeat=repeat,
            candidates=list(candidates), currentTemperature=base_temp, promptSha=prompt_sha,
            doneRuns=0, totalRuns=total_runs)

    def _guard_stop() -> None:
        if should_stop is not None and should_stop():
            raise TuneCancelled(f"在完成 {done_runs}/{total_runs} 跑之后被取消")

    # 1. 基线：用当前温度跑 repeat 次。它同时给出「抖动带」——没有它，后面任何涨幅都无法判断真假。
    baseline_scores: list[float] = []
    baseline_model = ""
    for index in range(repeat):
        _guard_stop()
        _notify(progress, type="run", phase="baseline", temperature=base_temp,
                run=index + 1, repeat=repeat, doneRuns=done_runs, totalRuns=total_runs)
        report = await run_eval(conn, case_def, instructions, None)
        if report is None:
            raise ValueError(f"基线第 {index + 1} 次评测没有返回报告（case={case_id} 未知类型？）")
        baseline_scores.append(float(report["total"]))
        baseline_model = baseline_model or str(report.get("runtimeModel") or "")
        done_runs += 1

    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    jump = max(baseline_scores) - min(baseline_scores)
    receipt: dict[str, Any] = {
        "agentType": agent_type,
        "caseId": case_id,
        "target": "params",
        "tunable": "temperature",
        "promptSha": prompt_sha,
        "repeat": repeat,
        "current": {"temperature": base_temp},
        "baseline": {
            "scores": baseline_scores,
            "mean": round(baseline_mean, 4),
            "jump": round(jump, 4),
            "runtimeModel": baseline_model,
        },
        "candidates": [],
        "accepted": None,
        "persisted": None,
    }
    print(f"基线：{baseline_scores} ⇒ 均值 {baseline_mean:.2f}，抖动带 {jump:.2f}")
    _notify(progress, type="baseline", scores=list(baseline_scores),
            mean=round(baseline_mean, 4), jump=round(jump, 4),
            doneRuns=done_runs, totalRuns=total_runs)

    if not candidates:
        # 已是最确定档：如实收工。**不**为了「跑出点东西」去试更高温度（那是改提示词的事）。
        receipt["reason"] = "当前温度已是最确定档（0.0），没有更低的候选"
        print(f"[SKIP] {receipt['reason']}")
        _notify(progress, type="done", reason="no_candidate", doneRuns=done_runs,
                totalRuns=total_runs)
        return receipt

    # 2. 逐个候选跑 repeat 次（同提示词、只换温度），按「均值涨幅 > 抖动带」判。
    parent_version: int | None = None
    version = _next_version(conn, agent_type, case_id)
    best: dict[str, Any] | None = None
    for offset, temp in enumerate(candidates):
        scores: list[float] = []
        candidate_model = ""
        for index in range(repeat):
            _guard_stop()
            _notify(progress, type="run", phase="candidate", temperature=temp,
                    run=index + 1, repeat=repeat, doneRuns=done_runs, totalRuns=total_runs)
            try:
                report = await run_eval(conn, case_def, instructions, {"temperature": temp})
            except Exception as err:  # noqa: BLE001
                # 单个候选跑崩（模型拒绝/超时）不该把整轮调优废掉：如实记一条然后继续下一个候选。
                scores = []
                print(f"[ERR] 温度 {temp} 第 {index + 1} 次评测失败：{err}")
                break
            if report is None:
                scores = []
                print(f"[ERR] 温度 {temp} 第 {index + 1} 次评测没有返回报告，跳过该候选")
                break
            scores.append(float(report["total"]))
            candidate_model = candidate_model or str(report.get("runtimeModel") or "")
            done_runs += 1

        row_version = version + offset
        if len(scores) != repeat:
            record = {
                "temperature": temp, "scores": [], "mean": None, "delta": None,
                "accepted": False, "reason": "eval_failed", "reasonLabel": "评测没跑完，不作数",
                "version": row_version,
            }
            receipt["candidates"].append(record)
            optimization_log.record(
                conn, agent_type=agent_type, version=row_version, case_id=case_id,
                parent_version=parent_version, score_before=baseline_mean, score_after=None,
                accepted=False, reason="eval_failed", target="params",
                runtime_model=candidate_model, prompt=instructions,
                evidence={"params": {"temperature": temp}, "samples": scores,
                          "baseline": baseline_scores},
            )
            _notify(progress, type="candidate", record=record, doneRuns=done_runs,
                    totalRuns=total_runs)
            continue

        mean = sum(scores) / len(scores)
        delta = mean - baseline_mean
        mismatched = bool(baseline_model and candidate_model and candidate_model != baseline_model)
        if mismatched:
            accepted, reason = False, "runtime_mismatch"
        elif delta > jump and delta > 0:
            accepted, reason = True, "accepted"
        elif delta > 0:
            # 涨了，但涨幅落在「同一档位自己重跑的差值」里 ⇒ 不能说这是温度带来的。
            accepted, reason = False, "below_noise"
        else:
            accepted, reason = False, "no_gain"

        record = {
            "temperature": temp,
            "scores": scores,
            "mean": round(mean, 4),
            "delta": round(delta, 4),
            "accepted": accepted,
            "reason": reason,
            "reasonLabel": optimization_log.REASONS.get(reason, reason),
            "version": row_version,
            "runtimeModel": candidate_model,
        }
        receipt["candidates"].append(record)
        print(f"温度 {temp}：{scores} ⇒ 均值 {mean:.2f}（Δ {delta:+.2f}）⇒ {record['reasonLabel']}")
        _notify(progress, type="candidate", record=record, doneRuns=done_runs,
                totalRuns=total_runs)

        optimization_log.record(
            conn, agent_type=agent_type, version=row_version, case_id=case_id,
            parent_version=parent_version, score_before=baseline_mean, score_after=mean,
            accepted=accepted, reason=reason, target="params",
            runtime_model=candidate_model, prompt=instructions,
            evidence={"params": {"temperature": temp}, "samples": scores,
                      "baseline": baseline_scores, "jump": jump,
                      "runtimeMismatch": mismatched},
        )

        if accepted and (best is None or delta > best["delta"]):
            best = record

    if best is None:
        print("[KEEP] 没有候选超过抖动带，温度保持原值")
        _notify(progress, type="done", reason="keep", doneRuns=done_runs, totalRuns=total_runs)
        return receipt

    receipt["accepted"] = {**best, "persisted": False}
    parent_version = best["version"]

    if not auto_persist:
        receipt["accepted"]["persisted"] = False
        receipt["accepted"]["persistSkip"] = "autoPersist=false，只给结论不落库"
        print(f"[NO-PERSIST] 采纳温度 {best['temperature']}，但 autoPersist=false")
        _notify(progress, type="done", reason="accepted_no_persist", doneRuns=done_runs,
                totalRuns=total_runs)
        return receipt

    # 3. 落库：只动 temperature 一列。写 0 行 = 没有配置行 ⇒ 照实报「采纳了但没落库」。
    # ⚠️ 必须传 ``conn``：本连接此刻还攥着上面那些优化史 INSERT 的写锁，而 SQLite WAL 只允许
    #    一个写者 ⇒ 另开连接去 UPDATE 会白等 busy_timeout（30 秒）再抛 ``database is locked``。
    #    复用同一连接还让「判决」与「生效」落进同一个事务（要么都落，要么都不落）。
    written = persist_agent_temperature(agent_type, float(best["temperature"]), conn)
    if written:
        optimization_log.mark_persisted(
            conn, agent_type=agent_type, version=best["version"], case_id=case_id,
        )
        receipt["accepted"]["persisted"] = True
        print(f"[PERSIST] 温度 {best['temperature']} 已写入 agent_configs"
              f"（agent={agent_type}）")
    else:
        receipt["accepted"]["persisted"] = False
        receipt["accepted"]["persistSkip"] = (
            "agent_configs 里没有该 agent 的配置行 ⇒ 温度无处可写"
            "（故意不新建：只带温度的行会把 system_prompt 留空）"
        )
        print(f"[MISS] 采纳温度 {best['temperature']}，但 {receipt['accepted']['persistSkip']}")

    print("\n===== 参数调优结束 =====")
    _notify(progress, type="done", reason="accepted" if written else "accepted_not_persisted",
            doneRuns=done_runs, totalRuns=total_runs)
    return receipt
