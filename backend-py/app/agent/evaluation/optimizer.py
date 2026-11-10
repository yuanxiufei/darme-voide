"""提示词优化器：状态机闭环（口径源自 ``evaluation/optimizer.ts`` 216 行）。

```text
evidence（评测报告）→ hypothesis（失分维度）→ candidate（LLM 改写）
  → evaluation（跑评测）→ accept（总分严格更高）/ rollback（保留最佳）
```

⚠️ **防作弊隔离**：优化器只能看到「维度名 + 分数」，看不到 rubric 的金标准（如「漏提了谁」）
—— ``format_feedback`` **故意不输出 detail**（detail 里可能含金标准答案）。候选提示词直接
传入评测器、**不写 DB**（优化史只存 sha1 指纹与分数 ✓，全文仍在 ``history/*.json``），
回滚零成本（无 tar.gz 快照的必要）。

⚠️ 第六处保真点：每一轮都写一行 ``agent_optimizations``（``services/optimization_log.py``）✓——
``history/*.json`` 是**按 case 覆盖**的（同名文件重跑即冲掉 ✗），而 ``agent_configs`` 是**覆盖式**
落库（上一次生效的提示词与分数当场就没了 ✗）；两边都不回答「这次改动提升了多少」。写口失败
只记 warn、不中断优化 ✓（长跑任务不许被日志拖废）。

⚠️ **溯源现状**：``evaluation/optimizer.ts`` **已随 Node 后端移出本仓**（全仓无此文件），
所以「对齐」指的是**当时的口径**，不是一份能跑的对照物。下面五条现在是本仓**自守**的判据，
改动它们不必同步任何外部文件。第 3 条（采纳判据）2026-11-10 起可带**抖动带**：实测同一配置
重复测量极差 **8 分**（``[80,88]``），零容差会把这种量级的噪声当改进写进配置；但**不设固定
阈值**（拍 8 分会把真实的 5 分改进也判成无效），而是让调用方显式传 ``repeat=2–5``、用**本次
实测的 Reference 极差**当抖动带（``repeat=1`` 仍是零容差，行为与历史逐字一致）。
对照表见 ``docs/knowledge-loop.md`` §4.2。

⚠️ **基线取谁 / 版本号怎么排**（2026-11-09 两处收口，都是「定位键不够真」的同一类病，且都
不抛异常、只把结论写错人）：① v0 基线＝**运行时在用的那份提示词**（`agent_configs.
system_prompt` 优先，没有配置行才回退出厂默认，见 `effective_prompt`）—— 原先一律取出厂
默认，重跑时拿「默认那份的分数」当靶子，候选只要高过默认就写回配置，把已落库的更好版本静默
覆盖掉；② `version` 是 (agent, case) 上的**轮次序号**，四格共用 `param_optimizer.
next_version` 递增 —— 原先每次运行都从 v1 数，同一 case 重跑会让 `mark_persisted` 按
`(agent, version, case, target)` 一次标中**多行**。两条的判据都在 `optimizer_test.py`。

⚠️ 五处保真点：

1. **nullish 语义**（``??``）：``iterations: 0`` 就是 0（不回落 3）；``autoPersist: false`` 生效；
2. **同一模型校验**：候选与 Reference 的 ``runtimeModel`` 不一致时**本轮不 accept**（直接 push
   ``accepted: false`` 并 continue）—— 评测期间换了模型，分数不可比；
3. **accept 判据是「严格高于当前 best」**（不是高于 Reference）⇒ 允许连续晋升；
4. **落库判据是「best 严格高于 Reference」+ ``best.version > 0``**（v0 是自己，不能算晋升）
   —— 注意 Reference 是**当前生效的那份**，所以这条读作「要赢过正在跑的版本」；
5. 历史文件写 ``{history_dir}/{agentType}-{caseId}.json``，``JSON.stringify(history, null, 2)``
   ⇒ ``ensure_ascii=False, indent=2``（**不是紧凑格式**，已加进 ``json.dumps`` 守卫白名单）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.engine import Connection

from app.core.models import agent_configs
from app.services import optimization_log
from app.services.agent_prompts import get_default_instructions
from app.agent.evaluation.catalog import benchmarks_dir
from app.services.agent_registry import get_default_name
from app.agent.creator import GeneratedAgentConfig, persist_agent_config
from app.agent.runtime import default_generate
from app.agent.skills import resolve_default_skills
from app.agent.tool import ToolRegistry
from app.services.ai_providers import get_text_config
from app.agent.evaluation.evaluator import evaluate_case
#: 进度/取消口径与另外三格（温度/技能/工具集）共用一份实现 —— 别在这里另写一个 `_notify`
#: `next_version` 同理：`version` 是 (agent, case) 上的**轮次序号**，四格共用一个递增器
from app.agent.evaluation.param_optimizer import TuneCancelled, next_version, notify

__all__ = [
    "MAX_ITERATIONS",
    "MAX_PROMPT_REPEAT",
    "OPTIMIZER_INSTRUCTIONS",
    "build_persist_candidate",
    "effective_prompt",
    "format_feedback",
    "generate_candidate_prompt",
    "optimize_agent_prompt",
    "print_report",
    "strip_code_fence",
]

#: 一次调优最多改写几轮。**越界即拒，不静默夹**（与 ``creator.TEMPERATURE_RANGE`` 同一条纪律：
#: 「你给 30 我悄悄按 20 跑」会让回执上的轮数与调用方要的对不上，而这一格还要真改配置）。
#: 依据是可算的：本格每轮 = 一次 LLM 改写 + 一次实测 **85–165 秒**的评测 ⇒ 20 轮 ≈ 28–55 分钟；
#: 而后台任务表是**进程内**的、没有超时回收（跑长了只能人工 cancel，进程一重启任务就没了），
#: 再往上就不该由一个 HTTP 查询参数触发。
MAX_ITERATIONS = 20

#: **Reference 复测**上限（不是「每档重跑」—— 本格每轮候选只评一次，见函数文档）。
#: 默认 1 ⇒ 抖动带 0（零容差，与历史行为逐字一致）；传 2–5 才用样本极差当抖动带。
#: 上限取 5，与温度格的 ``param_optimizer.MAX_REPEAT`` 同量级：Reference 那几跑是 47–165 秒/跑，
#: 5 跑已经 4–14 分钟，再多就该先问「是不是该换 case 而不是加样本」。
MAX_PROMPT_REPEAT = 5

OPTIMIZER_INSTRUCTIONS = """你是一名专业的 AI Agent 提示词优化专家。你的任务是改进给定 Agent 的系统提示词，使其在评测中得分更高。

约束：
1. 保持原有的工作流程与工具调用语义不变（不修改工具名、调用顺序）
2. 只针对给定的失分维度做针对性强化（明确字段要求、补充示例、强调易错点）
3. 输出必须是完整的、可直接作为系统提示词的文本
4. 只输出新提示词本身，不要输出任何解释、前言，也不要包 markdown 代码块围栏"""

_FENCE = re.compile(r"^```[a-zA-Z]*\n([\s\S]*?)\n```$")

#: 候选提示词的最短长度（低于此视为无效产出）
_MIN_CANDIDATE_LENGTH = 50


def format_feedback(report: dict[str, Any]) -> str:
    """只暴露「维度名 + 分数」，**不暴露 detail**（detail 里可能含金标准答案）。"""
    return "\n".join(
        f"- {dim['name']}：{dim['score']}/{dim['max']}"
        for dim in (report.get("dimensions") or [])
    )


def strip_code_fence(text: str) -> str:
    """LLM 偶尔会套 markdown 围栏，剥掉（没套则原样 trim）。"""
    stripped = (text or "").strip()
    match = _FENCE.match(stripped)
    return match.group(1).strip() if match else stripped


def _noise_clause(jump: float, jump_kind: str) -> str:
    """stdout 里那句抖动带：``repeat=1`` 说**没测**，不要说「抖动带 0」。

    「没测出抖动带」（``jumpKind="none"``）与「测出来是 0」（``within`` 且极差 0）是两件事实，
    印成同一句会让读终端的人以为这轮真量过噪声；与 ``tune_runtime._stage_text`` 的 `baseline`
    分支同一口径（那里写「本格没测抖动带（repeat=1）」）。
    """
    if jump_kind == "within":
        return f" + 抖动带 {jump}"
    return "（repeat=1，没测抖动带）"


def print_report(report: dict[str, Any]) -> None:
    """把评测报告打到 stdout（原 TS 用 ``console.log``，在 HTTP 路径里也一样打）。"""
    print(f"  总分 {report.get('total')}/100")
    for dim in report.get("dimensions") or []:
        print(f"    {dim['name']}: {dim['score']}/{dim['max']}  {dim['detail']}")


def build_persist_candidate(
    conn: Connection, agent_type: str, best_prompt: str
) -> GeneratedAgentConfig:
    """把优化出的最佳提示词组装成可落库配置。

    优化器**只改 systemPrompt** ⇒ name/description/skills **优先沿用 DB 既有值**，
    避免误覆盖用户自定义的显示名、描述与 skill 绑定；无既有值或 skills 损坏时回退默认绑定。
    """
    existing = conn.execute(
        select(agent_configs).where(agent_configs.c.agent_type == agent_type)
    ).all()
    row = existing[0] if existing else None

    skills: list[dict[str, Any]] = [
        {"id": skill_id, "enabled": True, "priority": index + 1}
        for index, skill_id in enumerate(resolve_default_skills(agent_type))
    ]
    if row is not None and row.skills:
        try:
            parsed = json.loads(row.skills)
            if isinstance(parsed, list):
                skills = parsed
        except (ValueError, TypeError):
            pass  # skills JSON 损坏 ⇒ 静默回退默认绑定

    return GeneratedAgentConfig(
        agent_type=agent_type,
        name=(row.name if row is not None and row.name else "") or get_default_name(agent_type),
        description=(row.description if row is not None else "") or "",
        system_prompt=best_prompt,
        skills=skills,
    )


async def generate_candidate_prompt(
    conn: Connection,
    agent_type: str,
    current_prompt: str,
    version: int,
    feedback_report: dict[str, Any],
) -> str:
    """让 LLM 按「失分维度」改写出新的候选提示词（**单发**调用，无工具）。"""
    text_config = get_text_config(conn)
    user_message = "\n".join([
        f"Agent 类型：{agent_type}",
        f"当前提示词（v{version}）：",
        "--- 开始 ---",
        current_prompt,
        "--- 结束 ---",
        "",
        "最近一次评测的失分维度（只看分数，不要猜测金标准答案）：",
        format_feedback(feedback_report),
        "",
        "请针对上述失分维度改进提示词，输出改进后的完整提示词。",
    ])

    result = await default_generate(
        config=text_config,
        model=str(text_config.get("model") or ""),
        instructions=OPTIMIZER_INSTRUCTIONS,
        tools=ToolRegistry({}),
        message=user_message,
        max_steps=1,
        model_settings={},
    )
    candidate = strip_code_fence(result.get("text") or "")
    if len(candidate) < _MIN_CANDIDATE_LENGTH:
        # ⚠️ 这条文案是**英文**的（原 TS 如此，与 creator 的中文文案不同）
        raise ValueError(f"Optimizer produced invalid candidate (length={len(candidate)})")
    return candidate


def effective_prompt(conn: Connection, agent_type: str) -> tuple[str, str]:
    """本次优化的**基线提示词** ⇒ ``(prompt, source)``，``source`` ∈ ``{"config", "default"}``。

    基线必须取**运行时真正在用的那一份**：``agent_configs.system_prompt`` 优先，没有配置行
    （或配置里是空串）才回退出厂默认。这是姊妹三格早就有的口径 —— 温度的基线是 DB 里那个
    温度、技能的基线是 DB 里那份绑定、工具集的基线是代码里装配出来的真工具集；只有本格原先
    拿 ``get_default_instructions`` 当 v0。

    ⚠️ 那处不一致的代价是**静默覆盖更优的现网提示词**（2026-11-09 修）：手工改过提示词、
    或上一次优化已落库一版 88 分的，再跑一次优化时基线仍是出厂默认（80 分）⇒ 候选只要高过
    80 就被判 ``accepted`` 并写回配置，88 分那份当场被 82 分顶掉；而优化史上那行的
    ``score_before=80`` 还会让人读成「涨了 2 分」。取在用的那份之后，判据自动变成
    「要赢过正在跑的版本」，没有提升就一行都不写 ✗。

    返回的 ``source`` 只描述**基线从哪来**（默认值/已保存的），不描述好坏：两者都是真值。
    """
    row = conn.execute(
        select(agent_configs.c.system_prompt).where(
            agent_configs.c.agent_type == str(agent_type or ""))
    ).first()
    live = str(row[0] or "").strip() if row is not None else ""
    if live:
        return live, "config"
    return get_default_instructions(agent_type), "default"


def normalize_prompt_options(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """把调用方给的 ``options`` 收成**唯一一份**校验过的形状（与 ``param_optimizer.normalize_options``
    同构：入队口与执行口共用，不然「入队放行、排到才炸」的错要等轮询才知道）。

    返回的键固定四个：``iterations`` / ``repeat`` / ``autoPersist`` / ``historyDir``
    （比另三格多一个 ``historyDir`` —— 本格有提示词全文要落盘；其余键一律不认）。

    两条边界**越界即拒**，理由都可算：

    * ``iterations > :data:`MAX_ITERATIONS``` —— 每轮是一次改写 + 一次 85–165 秒评测，
      20 轮 ≈ 28–55 分钟，而任务表在进程内、没有超时回收；
    * ``repeat`` 不在 1–:data:`MAX_PROMPT_REPEAT` —— 它是 Reference 的复测次数，
      0 跑不出均值、6 跑起该先问「是不是该换 case 而不是加样本」。

    ⚠️ 与 ``param_optimizer.normalize_options`` 的 ``repeat<2`` 硬拒**不同**：那边的 repeat 是
    「同一档位重跑几次，单次结果不可比」，这边是「Reference 复测几次」，``1`` 合法且有意义
    （＝不测抖动带、零容差）。照抄那条约束会把本格的默认值判成非法。
    """
    opts = dict(options or {})
    # ⚠️ `??` 语义：显式传 0 就是 0（不回落默认 3）
    iterations = opts["iterations"] if opts.get("iterations") is not None else 3
    iterations = max(0, int(iterations))
    if iterations > MAX_ITERATIONS:
        # 不夹逼：夹到上限会让回执的 `iterations` 与调用方要的对不上（见 MAX_ITERATIONS 注释）。
        # 负数仍夹 0（「只评一次 Reference」的老口径，与 `/tune-prompt` 路由「负数回默认 3」不冲突）。
        raise ValueError(
            f"iterations 最多 {MAX_ITERATIONS}（收到 {iterations}）：本格每轮是一次 LLM 改写"
            f"＋一次 85–165 秒的评测 ⇒ {MAX_ITERATIONS} 轮 ≈ 28–55 分钟；后台任务表在进程内、"
            f"没有超时回收，更长的无人值守长跑不该由一次请求触发"
        )
    repeat = opts.get("repeat") if opts.get("repeat") is not None else 1
    if not isinstance(repeat, int) or isinstance(repeat, bool):
        raise ValueError(f"repeat 必须是整数，收到 {type(repeat).__name__}")
    if not 1 <= repeat <= MAX_PROMPT_REPEAT:
        raise ValueError(
            f"repeat 必须在 1–{MAX_PROMPT_REPEAT} 之间（收到 {repeat}）：这是 Reference 的复测次数，"
            f"1 = 不测抖动带（零容差），2 起才用样本极差当抖动带"
        )
    auto_persist = opts["autoPersist"] if opts.get("autoPersist") is not None else True
    return {"iterations": iterations, "repeat": repeat,
            "autoPersist": bool(auto_persist), "historyDir": opts.get("historyDir")}


async def optimize_agent_prompt(
    conn: Connection,
    agent_type: str,
    case_def: dict[str, Any],
    options: dict[str, Any] | None = None,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """优化指定 Agent 的提示词（状态机），返回 ``OptimizationHistory``（camelCase 键）。

    ``progress`` / ``should_stop`` 是**入队口**（``POST /tune-prompt/{case_id}`` ⇒
    ``tune_runtime``）用的，与另外三格（温度/技能/工具集）逐条同构：事件类型
    ``start`` / ``run`` / ``baseline`` / ``candidate`` / ``done``，取消**只在两次评测之间生效**。
    两者默认 ``None`` ⇒ 老调用方（``POST /optimize/{case_id}`` 与 CLI）行为**逐字不变**。

    ``options``：

    * ``iterations``（默认 3；``0`` 合法＝只评一次 Reference）—— 改写几轮。**上限
      :data:`MAX_ITERATIONS`（20），越界即抛 ``ValueError``**，不夹逼；
    * ``repeat``（默认 **1**，1–:data:`MAX_PROMPT_REPEAT`）—— **Reference 复测几次**，
      取样本极差当抖动带。默认 1 ⇒ 抖动带 0（零容差、与历史行为逐字一致）；
      判据是「候选 > 当前 best **+ 抖动带**」，所以想要「噪声不算提升」就显式传 2–5，
      代价是 Reference 那几跑（47–165 秒/跑）；
    * ``historyDir`` / ``autoPersist``（默认 True）—— 与历史一致。

    ⚠️ **候选每轮只评一次**（不像温度格对每个候选重复测），所以分母是
    ``repeat + iterations``：那 ``repeat`` 跑**全在 Reference 上**。照温度格的
    ``repeat × (1 + n)`` 写会报出 2 倍分母、done 了页面还停在半程。

    ⚠️ 抖动带口径与另三格一致：**样本极差，不折半**（``jump = max − min``）；
    ``repeat=1`` 时极差必然是 0 ⇒ 回执 ``jumpKind="none"``（如实写「没测出抖动带」，
    与「算了但算出 0」不是一回事），``repeat≥2`` 才是 ``"within"``。
    """
    # 校验只此一处：`/tune-prompt` 入队前也调它（「入队放行、排到才炸」的错不该等轮询才暴露）
    opts = normalize_prompt_options(options)
    iterations = opts["iterations"]
    #: Reference 复测次数（候选每轮只评一次，见函数文档）
    repeat = opts["repeat"]
    # ⚠️ 默认值与 TS 的 `Path.cwd()/benchmarks/history` **行为等价但不再依赖 cwd**：
    #    Node 那边 cwd 恒为 `backend/` ⇒ 历史就写在 case 文件旁的 `benchmarks/history`；
    #    S7 把 case 搬到 `<项目根>/benchmarks` 后，这里改成同一个目录下的 `history`
    #    （否则从 `backend-py/` 启动会另开一份历史，与 case 目录脱节）。
    history_dir = opts["historyDir"] or str(benchmarks_dir() / "history")
    auto_persist = opts["autoPersist"]
    # 基线 = **运行时在用的那份**（DB 优先、无配置行才回退默认），见 `effective_prompt`
    reference_prompt, baseline_source = effective_prompt(conn, agent_type)
    #: 优化史的行键之一（同一个 agent 会跑不同 case，分数不可跨 case 比 ✓）。
    case_id = str(case_def.get("id") or "")

    # 进度分母 = Reference 的 `repeat` 跑 + 每轮候选各 1 跑。**别照抄温度格的
    # `repeat × (1 + n)`**：本格的 repeat 只加在 Reference 上（候选每轮只评一次），
    # 照抄会报出 2 倍分母、done 了还停在半程。
    # ⚠️ `iterations` 已在上面夹到非负（`iterations=-2` 会让分母变成 -1，页面显示 `0/-1`、
    #    `done` 时 `doneRuns=1 > totalRuns=-1`，进度条永远走不满，读起来像卡住）。
    total_runs = repeat + iterations
    done_runs = 0
    notify(progress, type="start", caseId=case_id, agentType=agent_type,
           repeat=repeat, iterations=iterations, doneRuns=0, totalRuns=total_runs)

    def _guard_stop() -> None:
        # 取消只在**两次评测之间**生效：跑到一半硬断会留下半条不进优化史的钱。
        if should_stop is not None and should_stop():
            raise TuneCancelled(f"在完成 {done_runs}/{total_runs} 跑之后被取消")

    # 1. 评测 Reference 基准（v0）—— 跑 `repeat` 次，样本极差就是本轮的抖动带
    print("\n===== 评测 Reference 提示词（v0，"
          f"{'当前生效的配置' if baseline_source == 'config' else '出厂默认'}）=====")
    reference_reports: list[dict[str, Any]] = []
    reference_samples: list[float] = []
    for index in range(1, repeat + 1):
        _guard_stop()
        notify(progress, type="run", phase="reference", run=index, repeat=repeat,
               doneRuns=done_runs, totalRuns=total_runs)
        measured = await evaluate_case(conn, case_def, reference_prompt)
        print_report(measured)
        reference_reports.append(measured)
        reference_samples.append(float(measured["total"]))
        done_runs += 1
    reference_report = reference_reports[0] if repeat == 1 else reference_reports[-1]
    # 分数取**样本均值**：repeat=1 时就是那唯一一次（保留原类型 —— 88 变 88.0 会把既有判据
    # 与历史 JSON 的读法带偏）；repeat≥2 才是真均值，与另三格「比均值」同口径。
    reference_score = (reference_report["total"] if repeat == 1
                       else round(sum(reference_samples) / len(reference_samples), 2))
    # 抖动带 = 同一份提示词自己那几次测量的**极差**（不折半，与另三格的 jump 同口径）。
    # repeat=1 拿不出散布 ⇒ 0，并如实标 `jumpKind="none"`（「没测出抖动带」不是「算出来是 0」）。
    jump = round(max(reference_samples) - min(reference_samples), 2) if repeat >= 2 else 0
    jump_kind = "within" if repeat >= 2 else "none"
    reference_model = reference_report.get("runtimeModel")
    notify(progress, type="baseline", score=reference_score, mean=reference_score,
           samples=reference_samples, jump=jump, jumpKind=jump_kind,
           baselineSource=baseline_source,
           doneRuns=done_runs, totalRuns=total_runs)

    history: dict[str, Any] = {
        "agentType": agent_type,
        "caseId": case_def.get("id"),
        "reference": {
            "score": reference_score,
            "dimensions": reference_report["dimensions"],
            "prompt": reference_prompt,
            # 基线从哪来（"config" = 运行时在用的、已保存的那份；"default" = 出厂默认）
            "source": baseline_source,
            # 抖动带的**凭据**：样本原值 + 极差。只写 jump 一个数，读档的人没法复核它算得对不对
            # （另三格的样本也在回执里，本格原本连 repeat 都没有）。
            "repeat": repeat,
            "samples": reference_samples,
            "jump": jump,
            "jumpKind": jump_kind,
        },
        "iterations": [],
        "best": {"version": 0, "score": reference_score, "prompt": reference_prompt},
    }
    best_report = reference_report

    target_dir = Path(history_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{agent_type}-{case_def.get('id')}.json"

    def _flush_history() -> None:
        # 每轮结束都落一次盘，而不是只在循环之后写一次：取消/中途失败时，已跑完那几轮的
        # 候选全文（优化史表只有 sha，唯一能回放的东西就在这份 JSON 里）不会跟着丢掉。
        # ⚠️ `JSON.stringify(history, null, 2)` ⇒ indent=2（**不紧凑**，已进 json.dumps 守卫白名单）
        out_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. 状态机：evidence → hypothesis → candidate → evaluation → accept/rollback
    # ⚠️ 版本号**跨运行连续**（与另三格共用 `param_optimizer.next_version` 的同一个递增器）：
    #    原先是 `range(1, iterations + 1)`，重跑同一个 (agent, case) 会再产出一套 v1/v2 ⇒
    #    优化史里同 target 同 case 出现两个 v1，而 `mark_persisted` 按
    #    (agent, version, case, target) 定位，**一次能标中多行** —— 上次那些并没落库的轮次
    #    被标成「已生效」。序号连续后每行定位唯一 ✓（`case` 与 `target` 已进 WHERE）。
    for round_index in range(1, iterations + 1):
        _guard_stop()
        version = next_version(conn, agent_type, case_id)
        notify(progress, type="run", phase="candidate", version=version, run=round_index,
               repeat=1, doneRuns=done_runs, totalRuns=total_runs)
        print(f"\n===== 第 {round_index}/{iterations} 轮（记作 v{version}）：生成候选提示词 =====")
        # `stage` 记失败发生在哪一步：生成就失败 ⇒ **根本没有候选**；评测失败 ⇒ 候选已经
        # 拿到、模型调用已经花掉了，那份全文必须留住（`history` / 优化史是它唯一的记录点，
        # 另三格的 `eval_failed` 也都带 `prompt=instructions` 而非空串）。
        candidate: str | None = None
        stage = "generate"
        try:
            candidate = await generate_candidate_prompt(
                conn, agent_type, history["best"]["prompt"], history["best"]["version"], best_report
            )
            stage = "evaluate"
            print(f"===== 第 {round_index}/{iterations} 轮（v{version}）：评测候选 =====")
            candidate_report = await evaluate_case(conn, case_def, candidate)
            if candidate_report is None:
                raise ValueError("评测没有返回报告")
        except TuneCancelled:
            # 取消是正常收场，不是「这一轮跑崩」：原样抛出，让 `tune_runtime` 记 cancelled。
            raise
        except Exception as err:  # noqa: BLE001
            # 单轮候选生成/评测失败（模型输出过短、超时、被拒）**不该废掉整轮** ✗：
            # 另三格早在候选层就把单点失败收进 `eval_failed` 继续跑（`param_optimizer` /
            # `skill_optimizer` / `tool_optimizer` 各有一处），本格原先没有这层 ⇒ 第 2 轮模型
            # 抽风输出过短时，第 1 轮已经花掉的评测钱全丢，连 `history` 文件都一个字不落。
            # 这一轮**照样占一个 version**（上面已取号）并进优化史：没跑成的轮次是分母的一部分。
            print(f"[ERR] 第 {round_index} 轮（v{version}）没跑成（{stage}）：{err}")
            kept_prompt = candidate if stage == "evaluate" else None
            history["iterations"].append({
                "version": version, "score": None, "accepted": False,
                "prompt": kept_prompt, "dimensions": [], "reason": "eval_failed",
            })
            optimization_log.record(
                conn, agent_type=agent_type, version=version, case_id=case_id,
                target="system_prompt",
                parent_version=history["best"]["version"],
                score_before=history["best"]["score"], score_after=None,
                accepted=False, reason="eval_failed", prompt=kept_prompt or "",
                evidence={"stage": stage, "error": str(err)},
            )
            done_runs += 1
            notify(progress, type="candidate",
                   record={"version": version, "score": None, "accepted": False,
                           "reason": "eval_failed",
                           "reasonLabel": optimization_log.REASONS.get("eval_failed", "eval_failed")},
                   doneRuns=done_runs, totalRuns=total_runs)
            _flush_history()
            continue
        print_report(candidate_report)

        # runtime 一致性校验：候选与 Reference 必须同模型，否则分数不可比（评测期间 runtime 漂移）
        candidate_model = candidate_report.get("runtimeModel")
        if reference_model and candidate_model and candidate_model != reference_model:
            print(f"[RUNTIME-MISMATCH] 候选={candidate_model} vs Reference={reference_model}，"
                  f"本轮结果不可比，跳过（不 accept）")
            history["iterations"].append({
                "version": version,
                "score": candidate_report["total"],
                "accepted": False,
                # 每条都带 `reason`：这份 JSON 是**唯一存着候选全文**的审计物，只写
                # accepted=False 分不清「分数没涨」与「评测中途换了模型」（后者换镜重跑才该改）。
                "reason": "runtime_mismatch",
                "prompt": candidate,
                "dimensions": candidate_report["dimensions"],
            })
            # 不可比的一轮也进优化史 ✓：它没被采纳，但"评测中途换了模型"这件事本身必须留痕 ✗
            # （否则从页面上只看到"这几轮分数都没涨"⇒ 误判成提示词改得不好 ✗）。
            optimization_log.record(
                conn, agent_type=agent_type, version=version, case_id=case_id,
                # 显式写 target：`record` 的默认值就是 "system_prompt"，但让它隐式兜着，
                # 一旦默认值被改，这一格会**静默**记进别的 target（详见 `mark_persisted` 的注释）
                target="system_prompt",
                parent_version=history["best"]["version"],
                score_before=history["best"]["score"], score_after=candidate_report["total"],
                accepted=False, reason="runtime_mismatch", runtime_model=str(candidate_model or ""),
                prompt=candidate, evidence=candidate_report["dimensions"],
            )
            done_runs += 1
            # 事件形状与温度/技能/工具集三格一致：一律用 ``record`` 子字典（``_stage_text`` 只认它）
            notify(progress, type="candidate",
                   record={"version": version, "score": candidate_report["total"],
                           "accepted": False, "reason": "runtime_mismatch",
                           "reasonLabel": optimization_log.REASONS.get(
                               "runtime_mismatch", "runtime_mismatch")},
                   doneRuns=done_runs, totalRuns=total_runs)
            _flush_history()
            continue

        # 判据：候选 > **当前 best + 抖动带**（repeat=1 ⇒ 抖动带 0，与历史行为逐字一致：
        # 严格更高即采纳）。跟 best 比而不是跟 Reference 比 —— 连续晋升时第 3 轮要跟 v2 比，
        # 跟 v0 比会把 v2 已经涨过的部分重复计入（与 `score_before` 的口径同源）。
        threshold = history["best"]["score"] + jump
        accepted = candidate_report["total"] > threshold
        history["iterations"].append({
            "version": version,
            "score": candidate_report["total"],
            "accepted": accepted,
            # 与 `optimization_log.record` 的 `reason` 同一套枚举（accepted / no_gain / …），
            # 这样 `history` 与优化史表逐轮对得上号。
            "reason": "accepted" if accepted else "no_gain",
            "prompt": candidate,
            "dimensions": candidate_report["dimensions"],
        })

        # 每一轮（不论采纳还是回滚）都进优化史 ✓：只记"被采纳的那几版"就答不了
        # 「试了 5 轮、涨了多少」✗ —— 分母（试过几次）跟分子（成了几次）一样是结论的一部分 ✓。
        # ⚠️ ``score_before`` 取**当时的 best 分数**（不是 Reference 分数 ✗）：连续晋升时
        #    第 3 轮要跟 v2 比，跟 v0 比会把"v2 已经涨过的那部分"重复计入 delta ✗。
        optimization_log.record(
            conn, agent_type=agent_type, version=version, case_id=case_id,
            target="system_prompt",
            parent_version=history["best"]["version"],
            score_before=history["best"]["score"], score_after=candidate_report["total"],
            accepted=accepted, reason="accepted" if accepted else "no_gain",
            runtime_model=str(candidate_model or ""), prompt=candidate,
            evidence=candidate_report["dimensions"],
        )
        done_runs += 1
        why = "accepted" if accepted else "no_gain"
        notify(progress, type="candidate",
               record={"version": version, "score": candidate_report["total"],
                       "accepted": accepted, "reason": why,
                       "reasonLabel": optimization_log.REASONS.get(why, why)},
               doneRuns=done_runs, totalRuns=total_runs)

        if accepted:
            print(f"[ACCEPT] {candidate_report['total']} > 最佳 {history['best']['score']}"
                  f"{_noise_clause(jump, jump_kind)}，候选晋升为 v{version}")
            history["best"] = {
                "version": version, "score": candidate_report["total"], "prompt": candidate,
            }
            best_report = candidate_report
        else:
            print(f"[ROLLBACK] {candidate_report['total']} 没有超过 最佳 {history['best']['score']}"
                  f"{_noise_clause(jump, jump_kind)}，保留 v{history['best']['version']}")
        # 这一轮（含 best 晋升）已定型 ⇒ 立刻落盘，取消/下一轮崩掉都不影响已完成的审计记录
        _flush_history()

    # 3. 持久化历史（含候选提示词，便于审计回放）—— `_flush_history` 的收尾一次
    _flush_history()

    # 4. 自动落库：best **严格高于 Reference + 抖动带** 才生效。
    # ⚠️ 这处 `+ jump` 目前是**冗余**的、不是补漏洞：任何被采纳的 best 都必然 > best_prev + jump，
    #    而 best_prev ≥ reference ⇒「采纳过」本身就蕴含 `best > reference + jump`，所以去掉它
    #    当前跑不出差别（别写「不加就会被噪声覆盖」这种测不到的话）。
    #    留着它的理由是「落库是**持久**动作、判据要能独立成句」——将来若放宽采纳判据（比如按维度
    #    分档），这行仍然守住底线；而不是指望它现在拦住什么。
    if auto_persist and history["best"]["version"] > 0 \
            and history["best"]["score"] > history["reference"]["score"] + jump:
        candidate_cfg = build_persist_candidate(conn, agent_type, history["best"]["prompt"])
        # ⚠️ 必须传 ``conn``：本连接此刻还攥着上面那些优化史 INSERT 的写锁（SQLite WAL 只允许
        #    一个写者）⇒ 另开连接做这条 UPSERT 会白等 busy_timeout 再抛 ``database is locked``。
        row = persist_agent_config(candidate_cfg, conn)
        history["persisted"] = {
            "version": history["best"]["version"],
            "score": history["best"]["score"],
            "name": row.name,
        }
        # 判决 ≠ 生效 ✓：这一版真的写进了 ``agent_configs``，优化史上要能看出来 ✗。
        # ``target`` 必传：只按 (agent, version, case) 定位会把「提示词落库」标到温度/技能那行
        optimization_log.mark_persisted(conn, agent_type=agent_type, target="system_prompt",
                                        version=history["best"]["version"], case_id=case_id)
        print(f"[PERSIST] 最佳候选 v{history['best']['version']} 已自动落库"
              f"（agent_type={agent_type}, name={row.name}）")
    else:
        history["persisted"] = None
        if auto_persist:
            print(f"[SKIP] 无严格更优候选，不落库"
                  f"（best={history['best']['score']}, reference={history['reference']['score']}"
                  f"{_noise_clause(jump, jump_kind)}）")

    # 上一步的 `_flush_history()` 发生在**落库之前** ⇒ 不再补这一次，盘上那份 JSON 里
    # 就永远没有 `persisted` 字段，读档的人看不出这一版到底生效没有（内存里的 `history`
    # 有、文件里没有，两者不一致 —— 而这份文件正是「候选全文 + 是否生效」的唯一审计物）。
    _flush_history()

    print("\n===== 优化结束 =====")
    print(f"最佳：v{history['best']['version']}，得分 {history['best']['score']}"
          f"（Reference v0 得分 {history['reference']['score']}"
          f"{_noise_clause(jump, jump_kind)}）")
    print(f"历史已写入：{out_path}")
    notify(progress, type="done", doneRuns=done_runs, totalRuns=total_runs,
           bestVersion=history["best"]["version"], bestScore=history["best"]["score"],
           referenceScore=history["reference"]["score"],
           jump=jump, jumpKind=jump_kind, repeat=repeat,
           persisted=bool(history.get("persisted")))
    return history
