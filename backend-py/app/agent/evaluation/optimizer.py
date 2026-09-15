"""提示词优化器：状态机闭环 —— 与 ``evaluation/optimizer.ts``（216 行）对齐。

```text
evidence（评测报告）→ hypothesis（失分维度）→ candidate（LLM 改写）
  → evaluation（跑评测）→ accept（总分严格更高）/ rollback（保留最佳）
```

⚠️ **防作弊隔离**：优化器只能看到「维度名 + 分数」，看不到 rubric 的金标准（如「漏提了谁」）
—— ``format_feedback`` **故意不输出 detail**（detail 里可能含金标准答案）。候选提示词直接
传入评测器、**不写 DB**，回滚零成本（无 tar.gz 快照的必要）。

⚠️ 五处保真点：

1. **nullish 语义**（``??``）：``iterations: 0`` 就是 0（不回落 3）；``autoPersist: false`` 生效；
2. **同一模型校验**：候选与 Reference 的 ``runtimeModel`` 不一致时**本轮不 accept**（直接 push
   ``accepted: false`` 并 continue）—— 评测期间换了模型，分数不可比；
3. **accept 判据是「严格高于当前 best」**（不是高于 Reference）⇒ 允许连续晋升；
4. **落库判据是「best 严格高于 Reference」+ ``best.version > 0``**（v0 是自己，不能算晋升）；
5. 历史文件写 ``{history_dir}/{agentType}-{caseId}.json``，``JSON.stringify(history, null, 2)``
   ⇒ ``ensure_ascii=False, indent=2``（**不是紧凑格式**，已加进 ``json.dumps`` 守卫白名单）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from app.core.models import agent_configs
from app.services.agent_prompts import get_default_instructions
from app.agent.evaluation.catalog import benchmarks_dir
from app.services.agent_registry import get_default_name
from app.agent.creator import GeneratedAgentConfig, persist_agent_config
from app.agent.runtime import default_generate
from app.agent.skills import resolve_default_skills
from app.agent.tool import ToolRegistry
from app.services.ai_providers import get_text_config
from app.agent.evaluation.evaluator import evaluate_case

__all__ = [
    "OPTIMIZER_INSTRUCTIONS",
    "build_persist_candidate",
    "format_feedback",
    "generate_candidate_prompt",
    "optimize_agent_prompt",
    "print_report",
    "strip_code_fence",
]

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


async def optimize_agent_prompt(
    conn: Connection,
    agent_type: str,
    case_def: dict[str, Any],
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """优化指定 Agent 的提示词（状态机），返回 ``OptimizationHistory``（camelCase 键）。"""
    options = options or {}
    # ⚠️ `??` 语义：显式传 0 就是 0
    iterations = options["iterations"] if options.get("iterations") is not None else 3
    # ⚠️ 默认值与 TS 的 `Path.cwd()/benchmarks/history` **行为等价但不再依赖 cwd**：
    #    Node 那边 cwd 恒为 `backend/` ⇒ 历史就写在 case 文件旁的 `benchmarks/history`；
    #    S7 把 case 搬到 `<项目根>/benchmarks` 后，这里改成同一个目录下的 `history`
    #    （否则从 `backend-py/` 启动会另开一份历史，与 case 目录脱节）。
    history_dir = options.get("historyDir") or str(benchmarks_dir() / "history")
    auto_persist = (options["autoPersist"] if options.get("autoPersist") is not None else True)
    reference_prompt = get_default_instructions(agent_type)

    # 1. 评测 Reference 基准（v0）
    print("\n===== 评测 Reference 提示词（v0，当前默认）=====")
    reference_report = await evaluate_case(conn, case_def, reference_prompt)
    print_report(reference_report)
    reference_model = reference_report.get("runtimeModel")

    history: dict[str, Any] = {
        "agentType": agent_type,
        "caseId": case_def.get("id"),
        "reference": {
            "score": reference_report["total"],
            "dimensions": reference_report["dimensions"],
            "prompt": reference_prompt,
        },
        "iterations": [],
        "best": {"version": 0, "score": reference_report["total"], "prompt": reference_prompt},
    }
    best_report = reference_report

    # 2. 状态机：evidence → hypothesis → candidate → evaluation → accept/rollback
    for version in range(1, int(iterations) + 1):
        print(f"\n===== 迭代 {version}/{iterations}：生成候选提示词 =====")
        candidate = await generate_candidate_prompt(
            conn, agent_type, history["best"]["prompt"], history["best"]["version"], best_report
        )

        print(f"===== 迭代 {version}/{iterations}：评测候选 =====")
        candidate_report = await evaluate_case(conn, case_def, candidate)
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
                "prompt": candidate,
                "dimensions": candidate_report["dimensions"],
            })
            continue

        accepted = candidate_report["total"] > history["best"]["score"]
        history["iterations"].append({
            "version": version,
            "score": candidate_report["total"],
            "accepted": accepted,
            "prompt": candidate,
            "dimensions": candidate_report["dimensions"],
        })

        if accepted:
            print(f"[ACCEPT] {candidate_report['total']} > 最佳 {history['best']['score']}，"
                  f"候选晋升为 v{version}")
            history["best"] = {
                "version": version, "score": candidate_report["total"], "prompt": candidate,
            }
            best_report = candidate_report
        else:
            print(f"[ROLLBACK] {candidate_report['total']} <= 最佳 {history['best']['score']}，"
                  f"保留 v{history['best']['version']}")

    # 3. 持久化历史（含候选提示词，便于审计回放）
    target_dir = Path(history_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{agent_type}-{case_def.get('id')}.json"
    # ⚠️ `JSON.stringify(history, null, 2)` ⇒ indent=2（**不紧凑**，已进 json.dumps 守卫白名单）
    out_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4. 自动落库：best **严格高于** Reference 才生效
    if auto_persist and history["best"]["version"] > 0 \
            and history["best"]["score"] > history["reference"]["score"]:
        candidate_cfg = build_persist_candidate(conn, agent_type, history["best"]["prompt"])
        row = persist_agent_config(candidate_cfg)
        history["persisted"] = {
            "version": history["best"]["version"],
            "score": history["best"]["score"],
            "name": row.name,
        }
        print(f"[PERSIST] 最佳候选 v{history['best']['version']} 已自动落库"
              f"（agent_type={agent_type}, name={row.name}）")
    else:
        history["persisted"] = None
        if auto_persist:
            print(f"[SKIP] 无严格更优候选，不落库"
                  f"（best={history['best']['score']}, reference={history['reference']['score']}）")

    print("\n===== 优化结束 =====")
    print(f"最佳：v{history['best']['version']}，得分 {history['best']['score']}"
          f"（Reference v0 得分 {history['reference']['score']}）")
    print(f"历史已写入：{out_path}")
    return history
