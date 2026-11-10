"""技能绑定优化器 —— 知识闭环 §1 缺口 4 的**第二格**：优化「注入哪几个 skill」。

为什么需要单开一格
==================
提示词（缺口 4 第 0 格）与 ``temperature``（第 1 格）之外，**真正决定 Agent 跑出来什么样**的还有
一格：这一跑里到底把哪些 SKILL.md 塞进了上下文。``agent_configs.skills`` 早就可以配（页面能勾选），
但「配哪几个」一直靠人凭感觉勾；而 ``skill_usage`` 又只统计「注入了什么」，不判断「注入了有没用」。
于是接上实测分数，把「该不该注入」变成可判的事 —— 这就是本模块。

判据方向与温度**相反**（这是本模块最重要的一条）
==============================================
温度那一格的判据是「均值涨幅 > 抖动带才采纳」（更确定 ⇒ 格式漂移更少 ⇒ 涨）；
这一格动的是**注入量**，而少注入本身就是收益（省预算、少串味、少踩别的技能的规则）：

* 摘掉某个 skill 后分数**没掉**（跌幅不超过基线抖动带）⇒ **没有证据说明它有用** ⇒ 采纳这一摘，
  记 ``reason="pruned"``；
* 摘掉后分数**明显下降**（跌幅超过抖动带）⇒ 那个 skill 真在起作用 ⇒ 保留，记
  ``reason="skill_needed"``。

两者不能共用一个理由码：若写成「accepted / no_gain」，页面上会出现「分数没涨却采纳了」这种看着
自相矛盾的记录 —— 而它其实完全正确。语义不同就分码，别让文案去弥补语义。

一次只摘一个（为什么不做累积删减）
==================================
候选之间**有交互**：同时摘掉 A、B 与先摘 A 再摘 B 不是同一件事，而且一旦摘掉一个，后面所有候选的
基线都变了。所以本模块：

1. 一次调用**至多采纳一个**（采纳即 ``break``）；
2. 每次采纳都重新跑基线 ⇒ 想再精简就再调一次（幂等、可反复跑、每步都有独立证据）。

这比「一口气摘三个」慢一点，但每一步都能对着优化史解释清楚，也避免了把 agent 的能力连着删没。

三条护栏（都不是保守，而是「这条通路自己会骗人」的地方）
======================================================
1. **不许摘成空**：只剩一个 skill 时没有候选（``skill_candidates`` 返回空）。摘到空之后分数大概率
   也不会掉 —— 那说明这套 case 测不出技能风格差异，**不等于**线上也没差别；真删空了是能力倒退；
2. **``autoPersist`` 默认 False**（与温度那格的默认 True 相反）：温度调低了最坏只是输出更死板，
   而删掉一个 skill 最坏是「该知道的规范它不知道了」。删减默认只出结论、由人点头；
3. **评测覆盖不写 ``skill_usage``**：``runtime`` 收到 ``options["skillsRaw"]`` 时不记使用统计
   （假设性绑定若入库，下一轮 ``penalty`` 的裁剪顺序就照着评测假数据走）。

落库口径
========
采纳后只 update ``agent_configs.skills`` 那一列（``creator.persist_agent_skills``），**没有配置行
就不写**（返回 0 ⇒ 照实记 ``persisted=False`` ＋ 理由），绝不为了「看起来生效了」插一行空配置。
写进去的值必须是 ``[{"id","enabled","priority"}]`` 形状 —— 纯字符串数组会被运行时的
``skills.parse_skills_config`` 整条跳过、静默回退默认绑定（见 ``creator._normalize_skills``）。
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.engine import Connection

from app.core.models import agent_configs
from app.services import optimization_log, skill_usage
from app.agent.skills import parse_skills_config, resolve_default_skills
from app.agent.creator import persist_agent_skills
from app.agent.evaluation.evaluator import evaluate_case
from app.agent.evaluation.param_optimizer import (
    DEFAULT_MAX_CANDIDATES,
    TuneCancelled,
    effective_instructions,
    next_version,
    normalize_options,
    notify,
    sha16,
)

__all__ = [
    "current_binding",
    "normalize_skill_options",
    "optimize_agent_skills",
    "skill_candidates",
]

#: 一次评测的签名。**必须接受 ``skills`` 关键字**（候选就是靠它换绑定的）；
#: 生产实现是 ``evaluator.evaluate_case``，测试里注入的替身也要照这个形状来。
EvaluateFn = Callable[..., Awaitable[dict[str, Any] | None]]
ProgressFn = Callable[[dict[str, Any]], None]


def current_binding(conn: Connection, agent_type: str) -> tuple[list[str], str]:
    """当前**真正生效**的绑定：``(skill id 列表, 来源)``，来源是 ``"db"`` 或 ``"default"``。

    必须与 ``skills.load_agent_skills`` 同口径，否则「基线」是编的：

    * DB 里能解析出**配置项** ⇒ 只取 ``enabled`` 的（全部关掉就是空列表，**不回退默认**）；
    * 没有配置行 / 空串 / 解析不出配置项 ⇒ 回退该 Agent 的默认绑定（SKILL.md frontmatter ``agents:``）。
    """
    row = conn.execute(
        select(agent_configs.c.skills).where(agent_configs.c.agent_type == agent_type)
    ).first()
    raw = row[0] if row is not None else None
    parsed = parse_skills_config(raw)
    if parsed:
        enabled = sorted((b for b in parsed if b.enabled), key=lambda b: b.priority or 0)
        return [b.id for b in enabled], "db"
    return list(resolve_default_skills(agent_type) or []), "default"


def skill_candidates(current: list[str], *, agent_type: str,
                     max_candidates: int = DEFAULT_MAX_CANDIDATES) -> list[str]:
    """leave-one-out 候选：**一次只摘一个**，按「最可疑」排前。

    排序依据两条，都不看分数（分数要跑评测才知道，那是候选之后的事）：

    1. ``skill_usage.penalty`` 高的先摘 —— 那一格统计的是「历史真实跑次里，注入了它反而掉分」，
       是最有依据的可疑度；样本不足时 ``penalty`` 返回 0，自然退到第 2 条；
    2. 同等可疑时**绑定顺序靠后的先摘** —— 顺序即 ``priority``，排在后面的本来就是「补充材料」。

    ``len(current) <= 1`` 或 ``max_candidates <= 0`` ⇒ 返回空（护栏 1：不许摘成空）。
    """
    if max_candidates <= 0 or len(current) <= 1:
        return []

    ranked: list[tuple[int, int, str]] = []
    for index, skill_id in enumerate(current):
        try:
            points, _why = skill_usage.penalty(agent_type, skill_id)
        except Exception:      # noqa: BLE001 —— 统计库不可用不该让「一个候选都没有」
            points = 0
        # 第一键取负号 ⇒ penalty 大的排前；第二键取负 index ⇒ 绑定靠后的排前
        ranked.append((-int(points), -index, skill_id))
    ranked.sort()
    return [item[2] for item in ranked[:max_candidates]]


def normalize_skill_options(options: dict[str, Any] | None = None) -> dict[str, Any]:
    """校验并补默认值 —— 与 ``param_optimizer.normalize_options`` 同一份实现，只改一处默认。

    **唯一差异**：``autoPersist`` 缺省是 **False**（温度那边是 True）。理由见模块头「三条护栏」第 2 条：
    温度调低最坏是更死板，删一个 skill 最坏是「该知道的规范不知道了」⇒ 删减要人点头。
    """
    opts = dict(options or {})
    normalized = normalize_options(opts)
    if opts.get("autoPersist") is None:
        normalized["autoPersist"] = False
    return normalized


def _skills_json(skill_ids: list[str]) -> str:
    """绑定 → 运行时可解析的 JSON（形状同 ``agent_configs.skills``：``[{"id","enabled","priority"}]``）。

    这里传的是 ``skillsRaw`` 覆盖值，不是落库值，但**形状必须一样** —— 两者都由
    ``skills.parse_skills_config`` 解析，形状分叉等于「评测时试的那个绑定」和「采纳后写进去的绑定」
    不是一回事。
    """
    return json.dumps(
        [{"id": s, "enabled": True, "priority": index + 1} for index, s in enumerate(skill_ids)],
        ensure_ascii=False, separators=(",", ":"),
    )


async def optimize_agent_skills(
    conn: Connection,
    agent_type: str,
    case_def: dict[str, Any],
    options: dict[str, Any] | None = None,
    *,
    evaluate: EvaluateFn | None = None,
    progress: ProgressFn | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """在当前绑定的基础上**逐个摘**，按「摘掉后没掉分」判采纳，采纳才落库。

    ``options``：``repeat``（默认 2，必须 ≥2）/ ``maxCandidates``（默认 2）/
    ``autoPersist``（**默认 False**）。其它键不认（同一份校验函数）。

    ``evaluate``：评测函数替身，签名 ``(conn, case_def, instructions, settings, *, skills=None)``。
    不传就用 ``evaluator.evaluate_case``。

    ``progress`` / ``should_stop``：与 ``param_optimizer`` 同语义（同步发进度、永不抛；
    取消只在两次评测之间生效）。

    返回回执：当前绑定与来源、基线样本与抖动带、每个候选的样本与 Δ、采纳了谁、落库成没成。
    """
    opts = normalize_skill_options(options)
    repeat = opts["repeat"]
    max_candidates = opts["maxCandidates"]
    auto_persist = opts["autoPersist"]

    run_eval: EvaluateFn = evaluate or evaluate_case
    case_id = str(case_def.get("id") or "")
    instructions = effective_instructions(conn, agent_type)
    prompt_sha = sha16(instructions)
    current, source = current_binding(conn, agent_type)
    candidates = skill_candidates(current, agent_type=agent_type, max_candidates=max_candidates)

    print(f"\n===== 技能绑定优化（agent={agent_type}, case={case_id}）=====")
    print(f"当前绑定 {len(current)} 个（来源 {source}）{current}，"
          f"候选 {candidates or '（没有可摘的）'}，每个跑 {repeat} 次（提示词指纹 {prompt_sha}）")

    total_runs = repeat * (1 + len(candidates))
    done_runs = 0
    notify(progress, type="start", caseId=case_id, agentType=agent_type, repeat=repeat,
           candidates=list(candidates), current=current, bindingSource=source,
           promptSha=prompt_sha, doneRuns=0, totalRuns=total_runs)

    def _guard_stop() -> None:
        if should_stop is not None and should_stop():
            raise TuneCancelled(f"在完成 {done_runs}/{total_runs} 跑之后被取消")

    # 1. 基线：**不传 skills 覆盖**（走 DB 里那套，即 current_binding 读到的同一份）。
    baseline_scores: list[float] = []
    baseline_model = ""
    for index in range(repeat):
        _guard_stop()
        notify(progress, type="run", phase="baseline", skills=list(current),
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
        "target": "skills",
        "tunable": "skills",
        "promptSha": prompt_sha,
        "repeat": repeat,
        "current": {"skills": list(current), "source": source},
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
    notify(progress, type="baseline", scores=list(baseline_scores),
           mean=round(baseline_mean, 4), jump=round(jump, 4),
           doneRuns=done_runs, totalRuns=total_runs)

    if not candidates:
        # 如实收工：只剩 ≤1 个（护栏 1）或 maxCandidates=0。
        if len(current) <= 1:
            receipt["reason"] = (f"当前只绑定了 {len(current)} 个 skill，再摘就是空绑定 —— "
                                 f"那会让 agent 变成裸模型，且分数不掉只说明这套 case 测不出差别")
        else:
            receipt["reason"] = "maxCandidates=0，没试任何候选"
        print(f"[SKIP] {receipt['reason']}")
        notify(progress, type="done", reason="no_candidate", doneRuns=done_runs,
               totalRuns=total_runs)
        return receipt

    # 2. 逐个候选（同提示词、同温度，只换注入集），按「跌幅不超过抖动带」判。
    version = next_version(conn, agent_type, case_id)
    best: dict[str, Any] | None = None
    for offset, skill_id in enumerate(candidates):
        remaining = [s for s in current if s != skill_id]
        if not remaining:      # skill_candidates 已挡住，这里只是「别把护栏写进注释就完事」
            continue
        override = _skills_json(remaining)
        scores: list[float] = []
        candidate_model = ""
        for index in range(repeat):
            _guard_stop()
            notify(progress, type="run", phase="candidate", skill=skill_id,
                   remaining=list(remaining), run=index + 1, repeat=repeat,
                   doneRuns=done_runs, totalRuns=total_runs)
            try:
                report = await run_eval(conn, case_def, instructions, None, skills=override)
            except Exception as err:      # noqa: BLE001
                # 单个候选跑崩不该废掉整轮：如实记一条再继续下一个。
                scores = []
                print(f"[ERR] 摘掉 {skill_id} 后第 {index + 1} 次评测失败：{err}")
                break
            if report is None:
                scores = []
                print(f"[ERR] 摘掉 {skill_id} 后第 {index + 1} 次评测没返回报告，跳过该候选")
                break
            scores.append(float(report["total"]))
            candidate_model = candidate_model or str(report.get("runtimeModel") or "")
            done_runs += 1

        row_version = version + offset
        if len(scores) != repeat:
            record = {
                "skill": skill_id, "remaining": remaining, "scores": [], "mean": None,
                "delta": None, "accepted": False, "reason": "eval_failed",
                "reasonLabel": optimization_log.REASONS.get("eval_failed", "eval_failed"),
                "version": row_version,
            }
            receipt["candidates"].append(record)
            optimization_log.record(
                conn, agent_type=agent_type, version=row_version, case_id=case_id,
                parent_version=None, score_before=baseline_mean, score_after=None,
                accepted=False, reason="eval_failed", target="skills",
                runtime_model=candidate_model, prompt=instructions,
                evidence={"skills": {"removed": skill_id, "remaining": remaining},
                          "samples": scores, "baseline": baseline_scores},
            )
            notify(progress, type="candidate", record=record, doneRuns=done_runs,
                   totalRuns=total_runs)
            continue

        mean = sum(scores) / len(scores)
        delta = mean - baseline_mean
        mismatched = bool(baseline_model and candidate_model and candidate_model != baseline_model)
        if mismatched:
            accepted, reason = False, "runtime_mismatch"
        elif delta >= -jump:
            # 跌幅不超过抖动带 ⇒ 没有证据说明它有用 ⇒ 采纳这一摘（方向与温度相反，见模块头）
            accepted, reason = True, "pruned"
        else:
            accepted, reason = False, "skill_needed"

        record = {
            "skill": skill_id,
            "remaining": remaining,
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
        print(f"摘掉 {skill_id}：{scores} ⇒ 均值 {mean:.2f}（Δ {delta:+.2f}）"
              f"⇒ {record['reasonLabel']}")
        notify(progress, type="candidate", record=record, doneRuns=done_runs,
               totalRuns=total_runs)

        optimization_log.record(
            conn, agent_type=agent_type, version=row_version, case_id=case_id,
            parent_version=None, score_before=baseline_mean, score_after=mean,
            accepted=accepted, reason=reason, target="skills",
            runtime_model=candidate_model, prompt=instructions,
            evidence={"skills": {"removed": skill_id, "remaining": remaining},
                      "samples": scores, "baseline": baseline_scores,
                      "jump": jump, "runtimeMismatch": mismatched},
        )

        if accepted:
            best = record
            break      # 一次只采纳一个：绑定变了，后面的候选基线已不同（见模块头）

    if best is None:
        print("[KEEP] 每个候选摘掉后分数都明显下降，绑定保持不变")
        notify(progress, type="done", reason="keep", doneRuns=done_runs, totalRuns=total_runs)
        return receipt

    receipt["accepted"] = {**best, "persisted": False}

    if not auto_persist:
        receipt["accepted"]["persistSkip"] = "autoPersist=false，只给结论不落库"
        print(f"[NO-PERSIST] 建议摘掉 {best['skill']}，但 autoPersist=false")
        notify(progress, type="done", reason="accepted_no_persist", doneRuns=done_runs,
               totalRuns=total_runs)
        return receipt

    # 3. 落库：只动 skills 一列。写 0 行 = 没有配置行 ⇒ 照实报「采纳了但没落库」。
    # ⚠️ 必须传 ``conn``：本连接还攥着优化史 INSERT 的写锁，SQLite WAL 只允许一个写者。
    written = persist_agent_skills(agent_type, best["remaining"], conn)
    if written:
        optimization_log.mark_persisted(
            conn, agent_type=agent_type, version=best["version"], case_id=case_id,
            target="skills",
        )
        receipt["accepted"]["persisted"] = True
        print(f"[PERSIST] 已从绑定里摘掉 {best['skill']}"
              f"（agent={agent_type}，剩 {len(best['remaining'])} 个）")
    else:
        receipt["accepted"]["persistSkip"] = (
            "agent_configs 里没有该 agent 的配置行 ⇒ 绑定无处可写"
            "（故意不新建：只带 skills 的行会把 system_prompt 留空）"
        )
        print(f"[MISS] 采纳摘掉 {best['skill']}，但 {receipt['accepted']['persistSkip']}")

    print("\n===== 技能绑定优化结束 =====")
    notify(progress, type="done", reason="accepted" if written else "accepted_not_persisted",
           doneRuns=done_runs, totalRuns=total_runs)
    return receipt
