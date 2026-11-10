"""工具集优化器 —— 知识闭环 §1 缺口 4 的**第三格**：优化「这一跑暴露哪几个工具」。

为什么这格必须等「单步对照」先落地
====================================
提示词 / 温度 / 技能绑定三格都能用「整轮总分」当目标函数，工具集不行 ——
现有 case 是**整轮**打分，摘掉 ``search_script_library`` 之后，扣分可能落在
「第 3 步没料可查」也可能落在「最后一句写得不好」上，两种原因在同一个数字里。
更要紧的是**耦合**：摘掉检索工具，模型会改用别的路径把活干完，整轮分可能一点不掉
⇒ 用整轮分判 leave-one-out，结论必然是「所有工具都可以摘」。

所以本模块的判决**同时**读两把尺子（``evaluator.step_trace`` 提供的逐步账）：

* **整轮分**（``report["total"]``）—— 摘掉之后活还干不干得成；
* **逐步账**（``report["stepStats"]``）—— 摘掉之后模型**还有没有再伸手去调它**
  （``unknownTool`` 计数）。后者才是「这个工具在流程里占着什么位置」的直接证据。

候选从哪来：**只看基线跑次里真被调用过的工具**
==============================================
静态枚举「该 Agent 装配了哪些工具」在这条路上拿不到：装配要真 ``episode_id``
（``create_agent_tools(conn, type, episode_id, drama_id)``），而 seed 是
``evaluate_case`` 内部造的，优化器手上没有。

但这不是妥协，反而更准：**从没被调用过的工具，摘掉它 Δ 必然 ≈ 0**，那种候选学不到
任何东西（它只说明「这套 case 用不到它」，不说明「线上用不到它」）。而轨迹里出现过的工具
是「这套 case 上真被用到的」，摘它才有信息量。

判据与护栏
==========
* ``delta < -jump`` ⇒ ``tool_needed``（真在用）；
* ``delta >= -jump`` **但**逐步账里出现 ``unknownTool`` ⇒ ``tool_wanted`` (**不采纳**) ——
  分数没掉只是别的工具补位了，轨迹明说「模型还在调它」，这时候摘掉属于「拿一次跑次的运气
  换线上的风险」，本模块不许这么判；
* ``delta >= -jump`` 但**逐步账本身没明细**（``stepStats.unavailable`` 非空）⇒ ``tool_unjudged``
  (**不采纳**) —— 这一档是「读数缺失」不是「读数说没调」：没有明细时 ``unknownTool`` 恒 0，
  与「模型真没再伸手」在数字上一模一样。第二把尺子哑了就只剩整轮分，而只靠整轮分必然得出
  「所有工具都能摘」（见模块头第一段）—— 所以宁可不判，也不许拿缺证据当证据；
  ⚠️ **部分**缺账（``stepStats.unjudged > 0``：有调用、没回灌结果）同属这一档：``unknownTool``
  是靠**结果的 error 正文**认出来的，那几步没有结果 ⇒ 模型真在其中调了被摘的工具也读不到。
  整格缺与部分缺一起挡，理由码仍是 ``tool_unjudged``（回执里两个字段分开写，便于分辨是哪一种）；
* ``delta >= -jump`` 且没有 ``unknownTool`` ⇒ ``pruned``（采纳这一摘）。

护栏与技能那格同：**不许摘成空**、**一次只采纳一个**（采纳即停，否则基线已变）。

为什么不落库（与技能那格的分岔，别当半成品看）
==============================================
``agent_configs`` **没有 tools 列**，而且工具集的真源是代码（``runtime._TOOL_FACTORIES``：
哪个 Agent 挂哪几组工具是装配语句，不是配置项）。所以本格的结论落 ``agent_optimizations``
（``target="tools"``）供人照着改代码，``persisted`` 恒 ``False`` ＋ 写明理由 ——
这与「缺口 5 只出候选、不自动应用」同一条口径：**能自动改的才自动改**。
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from sqlalchemy.engine import Connection

from app.services import optimization_log
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
    "current_tools",
    "normalize_tool_options",
    "optimize_agent_tools",
    "tool_candidates",
]

#: 一次评测的签名。**必须接受 ``tools`` 关键字**（候选就是靠它换暴露集）；
#: 生产实现是 ``evaluator.evaluate_case``，测试里注入的替身也要照这个形状来。
EvaluateFn = Callable[..., Awaitable[dict[str, Any] | None]]
ProgressFn = Callable[[dict[str, Any]], None]


def current_tools(report: dict[str, Any]) -> list[str]:
    """基线跑次里**真被调用过**的工具（按首次出现顺序去重）。

    取 ``stepStats.tools``；老报告没有这一栏时退回逐 ``steps`` 扫（两个都读不到 ⇒ 空列表，
    调用方据此如实收工，不猜一套工具名出来）。
    """
    stats = report.get("stepStats") or {}
    names = [str(item) for item in (stats.get("tools") or []) if str(item)]
    if names:
        return names
    order: list[str] = []
    for step in report.get("steps") or []:
        name = str((step or {}).get("tool") or "")
        if name and name not in order:
            order.append(name)
    return order


def tool_candidates(
    report: dict[str, Any],
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[str]:
    """leave-one-out 候选：**一次只摘一个**，按「最可疑」排前。

    排序三键（都不看分数，分数要跑评测才知道）：

    1. 失败步多的先摘 —— ``steps`` 里 ``ok is False`` 的那些步是「调了但没用」的直接证据；
    2. 调用次数**少**的先摘 —— 只被顺手调过一次的工具，证据最薄；
    3. 出现靠后的先摘 —— 后补进来的工具通常是补丁。

    ``len(current) <= 1`` 或 ``max_candidates <= 0`` ⇒ 返回空（不许摘成空）。
    """
    if max_candidates <= 0:
        return []
    order = current_tools(report)
    if len(order) <= 1:
        return []
    stats: dict[str, list[int]] = {}
    for step in report.get("steps") or []:
        name = str((step or {}).get("tool") or "")
        if name not in stats:
            stats[name] = [0, 0]
        stats[name][0] += 1                       # 调用次数
        if (step or {}).get("ok") is False:
            stats[name][1] += 1                   # 失败次数
    ranked: list[tuple[int, int, int, str]] = []
    for index, name in enumerate(order):
        calls, fails = stats.get(name, [0, 0])
        ranked.append((-fails, calls, -index, name))
    ranked.sort()
    return [item[3] for item in ranked[:max_candidates]]


def normalize_tool_options(options: dict[str, Any] | None = None) -> dict[str, Any]:
    """校验并补默认值 —— 与 ``param_optimizer.normalize_options`` 同一份实现。

    **唯一差异**：``autoPersist`` 恒为 ``False``（传 ``True`` 也不认）。本格没有可写的列，
    接受「自动落库」这个开关只会让回执出现一个永远不会发生的承诺 —— 干脆不认。
    """
    opts = dict(options or {})
    normalized = normalize_options(opts)
    normalized["autoPersist"] = False
    return normalized


def _tools_json(tool_ids: list[str]) -> str:
    """暴露集 → ``toolsRaw`` 覆盖值（JSON 数组原文，元素是工具名）。"""
    return json.dumps([str(item) for item in tool_ids], ensure_ascii=False,
                      separators=(",", ":"))


def _step_stats(report: dict[str, Any]) -> dict[str, Any]:
    """回执/落库只留四个计数 ＋ **两个「证据缺不缺」的标记**。

    ⚠️ 别把 ``unavailable`` / ``unjudged`` 在这里丢掉：本格判 ``pruned`` 的第二把尺子就是
    ``unknownTool``，而这条尺子在**没有逐步明细**时读数恒为 0 —— 看着像「模型没再调它」，
    其实只是**没记账**。丢掉这两个标记 ⇒ 证据缺失被读成证据充分，判决会误采纳。
    """
    stats = report.get("stepStats") or {}
    return {
        "total": int(stats.get("total") or 0),
        "ok": int(stats.get("ok") or 0),
        "failed": int(stats.get("failed") or 0),
        "unjudged": int(stats.get("unjudged") or 0),
        "unknownTool": int(stats.get("unknownTool") or 0),
        "unavailable": stats.get("unavailable"),
    }


async def optimize_agent_tools(
    conn: Connection,
    agent_type: str,
    case_def: dict[str, Any],
    options: dict[str, Any] | None = None,
    *,
    evaluate: EvaluateFn | None = None,
    progress: ProgressFn | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """在当前暴露集上**逐个摘**，判据同时读整轮分与逐步账，采纳即停。

    ``options``：``repeat``（默认 2，必须 ≥2）/ ``maxCandidates``（默认 2）；
    ``autoPersist`` 不认（见 :func:`normalize_tool_options`）。

    ``evaluate``：评测函数替身，签名 ``(conn, case_def, instructions, settings, *, tools=None)``。
    不传就用 ``evaluator.evaluate_case``。

    返回回执：基线暴露集（**实际被调用的那些**）、抖动带、每个候选的样本/Δ/逐步账、采纳了谁。
    落库恒为 ``False``，理由写在 ``persistSkip``。
    """
    opts = normalize_tool_options(options)
    repeat = opts["repeat"]
    max_candidates = opts["maxCandidates"]

    run_eval: EvaluateFn = evaluate or evaluate_case
    case_id = str(case_def.get("id") or "")
    instructions = effective_instructions(conn, agent_type)
    prompt_sha = sha16(instructions)

    print(f"\n===== 工具集优化（agent={agent_type}, case={case_id}）=====")
    print(f"提示词指纹 {prompt_sha}，每个候选跑 {repeat} 次（基线 1 轮先跑，用来取暴露集）")

    #: 进度分母。**基线只跑 1 次**（它不吃 ``repeat``：这一跑只用来取暴露集与基线分），
    #: 而候选数要**基线跑完**才知道（``tool_candidates`` 依赖轨迹）⇒ 先按上限估，
    #: 基线一落地就校正成 ``1 + repeat × 真候选数``。
    #: ⚠️ 别照抄技能那格的 ``repeat * (1 + n)``：它基线也跑 ``repeat`` 次，本格不是 ——
    #: 抄过来会「基线多算 repeat−1 跑 ＋ 候选按上限算」，于是任务已 done、页面永远显示 5/6。
    total_runs = 1 + repeat * max_candidates
    done_runs = 0
    notify(progress, type="start", caseId=case_id, agentType=agent_type,
           repeat=repeat, target="tools", promptSha=prompt_sha,
           doneRuns=0, totalRuns=total_runs)

    def _guard_stop() -> None:
        if should_stop is not None and should_stop():
            raise TuneCancelled(f"在完成 {done_runs}/{total_runs} 跑之后被取消")

    # 1. 基线：**不传 tools 覆盖**（就是该 Agent 的全部工具，即 current_tools 的出处）。
    _guard_stop()
    notify(progress, type="run", phase="baseline", run=1, repeat=1,
           doneRuns=done_runs, totalRuns=total_runs)
    baseline_report = await run_eval(conn, case_def, instructions, None)
    if baseline_report is None:
        raise ValueError(f"基线评测没有返回报告（case={case_id} 未知类型？）")
    baseline_score = float(baseline_report["total"])
    done_runs += 1
    exposed = current_tools(baseline_report)
    baseline_steps = _step_stats(baseline_report)
    baseline_model = str(baseline_report.get("runtimeModel") or "")

    receipt: dict[str, Any] = {
        "agentType": agent_type,
        "caseId": case_id,
        "target": "tools",
        "tunable": "tools",
        "promptSha": prompt_sha,
        "repeat": repeat,
        "current": {"tools": list(exposed), "source": "trace"},
        "baseline": {
            "score": round(baseline_score, 4),
            "runtimeModel": baseline_model,
            "steps": baseline_steps,
        },
        "candidates": [],
        "accepted": None,
        "persisted": False,
    }
    # 基线这一跑就没有逐步明细 ⇒ 本格的第二把尺子**整格哑掉**：候选跑多少次都只会得
    # ``tool_unjudged``（证据缺失，恒不采纳）⇒ 白烧 ``repeat × 候选数`` 跑（8 分钟量级）
    # 才换来一句「判不了」。所以在这里就如实收工，且**不许把理由写成「空工具集」**：
    # 缺明细时 ``current_tools`` 也是空的，照下面的空分支走会说「只出现了 0 个工具」
    # —— 那是**读数缺失**被读成「模型没调过任何工具」。
    baseline_unavailable = baseline_steps.get("unavailable")
    if baseline_unavailable:
        candidates = []
    else:
        # 候选数到此才算得出（基线那一跑的内存逐步账是唯一来源）⇒ 顺手把进度分母校正成
        # 真值，让 `baseline` 这条事件起就是准的（入队那一刻的估算只在头几秒里被看见）。
        candidates = tool_candidates(baseline_report, max_candidates=max_candidates)
    total_runs = 1 + repeat * len(candidates)

    print(f"基线：{baseline_score:.2f}，轨迹里出现 {len(exposed)} 个工具 {exposed}，"
          f"逐步账 {baseline_steps}")
    notify(progress, type="baseline", score=round(baseline_score, 4), tools=list(exposed),
           steps=baseline_steps, doneRuns=done_runs, totalRuns=total_runs)

    if baseline_unavailable:
        receipt["reason"] = (
            f"基线这一跑没有逐步明细（{baseline_unavailable}）⇒ 判 pruned 的第二把尺子是"
            f"「候选那一跑里模型还有没有再伸手调它」，它整格哑掉 ⇒ 候选跑再多也只能得出"
            f"「判不了」。先让驱动回灌逐步调用**列表**（`tool_calls` 逐条，不是只给个数）再来跑。"
        )
        print(f"[SKIP] {receipt['reason']}")
        notify(progress, type="done", reason="no_candidate", doneRuns=done_runs,
               totalRuns=total_runs)
        return receipt

    if not candidates:
        if len(exposed) <= 1:
            receipt["reason"] = (f"基线轨迹里只出现了 {len(exposed)} 个工具，再摘就是空工具集 —— "
                                 f"那让 agent 连保存都做不了，且分数不掉只说明这套 case 测不出差别")
        else:
            receipt["reason"] = "maxCandidates=0，没试任何候选"
        print(f"[SKIP] {receipt['reason']}")
        notify(progress, type="done", reason="no_candidate", doneRuns=done_runs,
               totalRuns=total_runs)
        return receipt

    # 2. 逐个候选（同提示词、同温度，只换暴露集）。抖动带见下面 `jump` 那段：
    #    只能用**同一配置重复测量**的离散度，不能用候选之间的差距（那是信号）。
    version = next_version(conn, agent_type, case_id)
    pending: list[dict[str, Any]] = []
    for offset, tool_id in enumerate(candidates):
        remaining = [name for name in exposed if name != tool_id]
        if not remaining:
            continue
        override = _tools_json(remaining)
        scores: list[float] = []
        steps_acc: list[dict[str, Any]] = []
        candidate_model = ""
        for index in range(repeat):
            _guard_stop()
            notify(progress, type="run", phase="candidate", tool=tool_id,
                   remaining=list(remaining), run=index + 1, repeat=repeat,
                   doneRuns=done_runs, totalRuns=total_runs)
            try:
                report = await run_eval(conn, case_def, instructions, None, tools=override)
            except Exception as err:      # noqa: BLE001
                print(f"[ERR] 摘掉 {tool_id} 后第 {index + 1} 次评测失败：{err}")
                scores = []
                break
            if report is None:
                print(f"[ERR] 摘掉 {tool_id} 后第 {index + 1} 次评测没返回报告，跳过该候选")
                scores = []
                break
            scores.append(float(report["total"]))
            steps_acc.append(_step_stats(report))
            candidate_model = candidate_model or str(report.get("runtimeModel") or "")
            done_runs += 1

        row_version = version + offset
        if len(scores) != repeat:
            record = {
                "tool": tool_id, "remaining": remaining, "scores": [], "mean": None,
                "delta": None, "accepted": False, "reason": "eval_failed",
                "reasonLabel": optimization_log.REASONS.get("eval_failed", "eval_failed"),
                "version": row_version,
            }
            receipt["candidates"].append(record)
            optimization_log.record(
                conn, agent_type=agent_type, version=row_version, case_id=case_id,
                parent_version=None, score_before=baseline_score, score_after=None,
                accepted=False, reason="eval_failed", target="tools",
                runtime_model=candidate_model, prompt=instructions,
                evidence={"tools": {"removed": tool_id, "remaining": remaining},
                          "samples": scores, "steps": steps_acc},
            )
            notify(progress, type="candidate", record=record, doneRuns=done_runs,
                   totalRuns=total_runs)
            continue

        mean = sum(scores) / len(scores)
        unknown = sum(int(item.get("unknownTool") or 0) for item in steps_acc)
        # 逐步明细缺没缺（任一样本缺就算缺：判 pruned 要**每个**样本都看得见轨迹）
        unavailable = next((item.get("unavailable") for item in steps_acc
                            if item.get("unavailable")), None)
        # **部分**缺账：`unjudged` 是「调用有、结果没有」的步数（驱动少回灌/截断）。
        # ⚠️ 它和 `unavailable` 一样能让第二把尺子读瞎 —— `unknownTool` 是靠**结果的 error
        # 正文**认出来的，那几步没有结果 ⇒ 就算模型真在其中调了被摘的工具，也读不到。
        # 所以取**最坏样本**的步数：任一跑看不见的调用，都不能算进「模型没再调它」。
        unjudged = max((int(item.get("unjudged") or 0) for item in steps_acc), default=0)
        pending.append({
            "tool": tool_id, "remaining": remaining, "scores": scores, "mean": mean,
            "steps": steps_acc, "unknownTool": unknown, "stepsUnavailable": unavailable,
            "stepsUnjudged": unjudged,
            "version": row_version, "runtimeModel": candidate_model,
        })
        print(f"摘掉 {tool_id}：{scores} ⇒ 均值 {mean:.2f}，"
              f"逐步账吸掉 unknownTool {unknown} 次"
              + (f"（⚠️ 逐步账缺明细：{unavailable}）" if unavailable
                 else f"（⚠️ 有 {unjudged} 步没回灌结果、读不到 unknownTool）"
                 if unjudged else ""))
        notify(progress, type="candidate", record={
            "tool": tool_id, "remaining": remaining, "scores": scores,
            "mean": round(mean, 4), "unknownTool": unknown, "version": row_version,
        }, doneRuns=done_runs, totalRuns=total_runs)

    if not pending:
        print("[KEEP] 没有可判的候选（全部评测失败），工具集保持不变")
        notify(progress, type="done", reason="keep", doneRuns=done_runs,
               totalRuns=total_runs)
        return receipt

    # 抖动带 = **同一配置重复测量**之间的极差（各候选自己那 repeat 个样本的 `max − min`，
    # 取最大者），幅值口径与参数格/技能格**逐字一致**（它们都是
    # `max(baseline_scores) - min(baseline_scores)`，**不折半**）。
    # ⚠️ 别拿「候选之间的极差」当噪声：候选本来就该差很多（摘掉关键工具 vs 摘掉边缘工具），
    # 那是**信号**。实测反例（2026-11-09 真模型 qwen3:14b / sb-case-001）：基线 95.00；
    # 候选 A 摘 `save_continuity_states` 跑出 [80, 88]（均值 84，真掉了 11 分）；候选 B 摘
    # `read_storyboard_context` 崩到 [0, 0]。旧口径 jump=(84−0)/2=42 ⇒ A 的 −11 被当成
    # 「噪声内、分数没掉」⇒ **错摘一个真在用的工具并记成已采纳**。按候选内部重复样本估：
    # A 的 [80,88] ⇒ 8、B 的 [0,0] ⇒ 0 ⇒ jump=8 ⇒ −11 < −8 ⇒ 判 `tool_needed` ✓。
    # 唯一的来源差异（如实写在这里，别在文档里说成「同源」）：参数格/技能格的抖动带取自
    # **基线**重复跑，本格基线只跑 1 次（见上面 1.），拿不到基线抖动 ⇒ 只能取自**候选**自己
    # 那 repeat 个样本 —— 两者都是「同一配置重复测量」，合法同源，但不是同一批样本。
    # `repeat=1` 时估不出抖动 ⇒ 按 0 计（与参数格「抖动带为 0 时 Δ>0 即算提升」同一口径：
    # 没有抖动，就不存在「掉的那点可能是噪声」）。`jumpKind` 如实写来源。
    spreads = [(max(item["scores"]) - min(item["scores"]))
               for item in pending if len(item["scores"]) > 1]
    jump = max(spreads) if spreads else 0.0
    receipt["jump"] = round(jump, 4)
    receipt["jumpKind"] = "within" if spreads else "none"

    best: dict[str, Any] | None = None
    for item in pending:
        delta = item["mean"] - baseline_score
        mismatched = bool(baseline_model and item["runtimeModel"]
                          and item["runtimeModel"] != baseline_model)
        if mismatched:
            accepted, reason = False, "runtime_mismatch"
        elif delta < -jump:
            accepted, reason = False, "tool_needed"
        elif item["stepsUnavailable"] or item["stepsUnjudged"]:
            # 逐步账没明细（整格缺）**或有步没回灌结果**（部分缺）⇒ ``unknownTool`` 在这两处
            # 都是**读数缺失**，不是「模型没再调它」。第二把尺子哑了就不许判 pruned
            # （分数这条尺子单独用必然得出「都能摘」，见模块头）。
            accepted, reason = False, "tool_unjudged"
        elif item["unknownTool"] > 0:
            # 分数没掉，但模型还在伸手调它 ⇒ 别的工具补位而已，不是「不需要」（见模块头）
            accepted, reason = False, "tool_wanted"
        else:
            accepted, reason = True, "pruned"

        record = {
            "tool": item["tool"],
            "remaining": item["remaining"],
            "scores": item["scores"],
            "mean": round(item["mean"], 4),
            "delta": round(delta, 4),
            "unknownTool": item["unknownTool"],
            "stepsUnavailable": item["stepsUnavailable"],
            "stepsUnjudged": item["stepsUnjudged"],
            "accepted": accepted,
            "reason": reason,
            "reasonLabel": optimization_log.REASONS.get(reason, reason),
            "version": item["version"],
            "runtimeModel": item["runtimeModel"],
        }
        receipt["candidates"].append(record)
        print(f"⇒ {record['reasonLabel']}（Δ {delta:+.2f}，unknownTool {item['unknownTool']}）")

        optimization_log.record(
            conn, agent_type=agent_type, version=item["version"], case_id=case_id,
            parent_version=None, score_before=baseline_score, score_after=item["mean"],
            accepted=accepted, reason=reason, target="tools",
            runtime_model=item["runtimeModel"], prompt=instructions,
            evidence={"tools": {"removed": item["tool"], "remaining": item["remaining"]},
                      "samples": item["scores"], "steps": item["steps"],
                      "unknownTool": item["unknownTool"], "jump": jump,
                      "stepsUnavailable": item["stepsUnavailable"],
                      "stepsUnjudged": item["stepsUnjudged"],
                      "jumpKind": receipt["jumpKind"], "runtimeMismatch": mismatched},
        )
        notify(progress, type="candidate", record=record, doneRuns=done_runs,
               totalRuns=total_runs)
        if accepted and best is None:
            best = record

    if best is None:
        # ⚠️ 别写死「要么掉分、要么模型还在调它」：全部候选都栽在 ``tool_unjudged`` 时
        # 那句话是**假话**（模型可能根本没在调，只是账读不到）⇒ 按真实理由码逐条报。
        labels = sorted({str(record.get("reasonLabel") or record.get("reason") or "?")
                         for record in receipt["candidates"]})
        print(f"[KEEP] 没有任何候选被采纳（{'、'.join(labels) or '全部评测失败'}），"
              f"工具集保持不变")
        notify(progress, type="done", reason="keep", doneRuns=done_runs,
               totalRuns=total_runs)
        return receipt

    receipt["accepted"] = {**best, "persisted": False, "persistSkip": (
        "agent_configs 没有 tools 列，且工具集真源是代码（runtime._TOOL_FACTORIES 的装配语句）"
        "⇒ 本格只出结论、不自动改代码"
    )}
    print(f"[NO-PERSIST] 建议摘掉 {best['tool']}：{receipt['accepted']['persistSkip']}")
    print("\n===== 工具集优化结束 =====")
    notify(progress, type="done", reason="accepted_no_persist", doneRuns=done_runs,
           totalRuns=total_runs)
    return receipt
