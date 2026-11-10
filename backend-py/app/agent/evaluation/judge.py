"""LLM 语义判分（学自 lobehub ``eval-rubric/src/matchers/llmRubric.ts``）。

**为什么本仓需要它**：``app/agent/evaluation/scorer.py`` 那五把尺子全是**确定性规则**
（正则/结构/覆盖度/词表），它们能判「有没有 12 个镜头」「角色名在不在表里」，
判不了「这场戏的冲突够不够」「这句台词有没有 AI 味」—— 而提示词优化器的目标函数
恰恰是后者。没有语义判分，优化器只能优化"格式合规"，那会把模型推向更模板化的输出。

三件**真机制**（缺一件这套就退化成"模型随便给个分"）：

1. **证据单列** ``[Evidence]``：``[Output]`` 是模型自己说它做了什么，``[Evidence]`` 是运行
   轨迹证明它做了什么（:func:`evidence_from_steps` 从 ``step_trace`` 渲染）。判"有没有
   真调工具 / 真落盘"这类标准时只能看 Evidence —— 参考实现的原文就是这么写的：
   "Criteria about what the agent did (call a tool, save something) can only be judged
   from the actions themselves."
2. **0 分合法**：``score == 0`` 是「完全不符合」的正当判决，不是解析失败。
   参考实现专门留了一条注释：``!result.score`` would drop real zeros ⇒ 判的是
   ``isinstance(score, (int, float))``，**不是真值**。本模块同口径。
3. **只对"判官翻车"重试**：抛错 / 没给 score / score 不是有限数 / 越界 / reason 不是字符串
   ⇒ 重试（默认 3 次）；**分数不理想不重试**（重试一个 0.2 分直到它给 0.9，那是把
   判分变成抽奖）。三次都翻车 ⇒ ``inconclusive``（判不出来），**不是 0 分**：
   ``0.0`` 与"没判成"在优化器里是两种完全不同的结论（前者该改，后者该重跑判分）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from app.services.text_generation import generate_text, parse_json_object

__all__ = [
    "DEFAULT_SYSTEM_ROLE",
    "DEFAULT_THRESHOLD",
    "DEFAULT_MAX_ATTEMPTS",
    "JUDGE_SCHEMA",
    "JudgeVerdict",
    "build_judge_prompt",
    "evidence_from_steps",
    "judge_output",
    "verdict_from_payload",
]

#: 判官 system role（逐字对拍参考 ``DEFAULT_SYSTEM_ROLE``）
DEFAULT_SYSTEM_ROLE = "\n".join([
    "You are an expert evaluation judge. Your task is to score how well an AI output meets the given criteria.",
    "",
    "Scoring rules:",
    "- Score 1.0: The output fully satisfies the criteria.",
    "- Score 0.0: The output completely fails to meet the criteria.",
    "- Use intermediate values (e.g. 0.3, 0.5, 0.7) for partial matches.",
    "",
    'Respond with a JSON object containing "score" (number 0-1) and "reason" (brief explanation).',
])

#: 参考 ``JUDGE_SCORE_SCHEMA``（``minimum`` / ``maximum`` 由 ``validate_json_schema`` 新增支持）
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "reason"],
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1,
                  "description": "Score from 0.0 to 1.0"},
        "reason": {"type": "string", "minLength": 1,
                   "description": "Brief explanation for the score"},
    },
}

#: 判定「通过」的默认门槛（参考 ``rubric.threshold ?? 0.6``）
DEFAULT_THRESHOLD = 0.6
#: 判官翻车重试上限（参考 ``context.judgeMaxAttempts ?? 3``）
DEFAULT_MAX_ATTEMPTS = 3

_EVIDENCE_HEADER = (
    "[Evidence]\n"
    "What the AI did while producing the output. Use it for criteria about actions "
    "or resulting state.\n"
)


@dataclass
class JudgeVerdict:
    """一次判分结论（``score=None`` ＋ ``verdict="inconclusive"`` ＝ 判官没判成）。"""

    score: float | None = None
    reason: str = ""
    passed: bool | None = None
    attempts: int = 0
    criteria: str = ""
    threshold: float = DEFAULT_THRESHOLD
    error: str = ""
    raw: str = ""

    @property
    def verdict(self) -> str:
        return "inconclusive" if self.score is None else "scored"

    @property
    def scored(self) -> bool:
        return self.score is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": self.score,
            "reason": self.reason,
            "passed": self.passed,
            "attempts": self.attempts,
            "threshold": self.threshold,
            "criteria": self.criteria,
            # 判不出来时必须带原因：``inconclusive`` 不带原因就成了"神秘失败"
            "error": self.error,
        }


def build_judge_prompt(
    criteria: str,
    output: str,
    *,
    input_text: str | None = None,
    evidence: str | None = None,
    expected: str | None = None,
) -> str:
    """五段式判分 prompt（顺序与参考 ``buildJudgeUserPrompt`` 逐段对拍）。

    ``[Criteria]`` 与 ``[Output]`` **恒在**；``[Input]`` / ``[Evidence]`` / ``[Expected]``
    为空时整段不出现（参考同口径：不写一个空标题去浪费判官的注意力）。
    """
    parts = [f"[Criteria]\n{criteria}"]
    if input_text:
        parts.append(f"[Input]\n{input_text}")
    parts.append(f"[Output]\n{output}")
    if evidence:
        parts.append(_EVIDENCE_HEADER + evidence)
    if expected:
        parts.append(f"[Expected]\n{expected}")
    return "\n\n".join(parts)


def evidence_from_steps(steps: Sequence[Mapping[str, Any]] | None,
                        *,
                        limit: int = 60) -> str:
    """把 ``evaluator.step_trace`` 的逐步账渲染成 Evidence 段。

    只留**可核对的事实**：调了哪个工具、参数是什么、结果字节数/错误、每步耗时。
    不把整套 trace dump 进去（长 trace 会盖掉标准本身）；超出 ``limit`` 步时留头留尾
    并明说截断了几步 —— 截断这件事必须让判官知道，否则它会以为"只有这些动作"。
    """
    rows: list[str] = []
    items = list(steps or [])
    head = items if len(items) <= limit else items[: limit // 2]
    tail = [] if len(items) <= limit else items[-(limit // 2):]
    for index, step in enumerate(head, start=1):
        rows.append(_render_step(index, step))
    if tail:
        rows.append(f"…（中间省略 {len(items) - len(head) - len(tail)} 步）…")
        for offset, step in enumerate(tail, start=len(items) - len(tail) + 1):
            rows.append(_render_step(offset, step))
    return "\n".join(row for row in rows if row)


def _render_step(index: int, step: Mapping[str, Any]) -> str:
    if not isinstance(step, Mapping):
        return f"{index}. {step!r}"
    kind = step.get("kind") or step.get("type") or "step"
    bits = [f"{index}. [{kind}]"]
    for key in ("tool", "toolName", "name"):
        if step.get(key):
            bits.append(str(step[key]))
            break
    args = step.get("args")
    if args:
        bits.append("参数=" + _short(args))
    for key in ("error", "problem"):
        if step.get(key):
            bits.append(f"错误={step[key]}")
    for key in ("chars", "bytes", "count", "tokens"):
        if isinstance(step.get(key), (int, float)):
            bits.append(f"{key}={step[key]}")
    if step.get("ok") is False:
        bits.append("失败")
    return " ".join(bits)


def _short(value: Any, limit: int = 240) -> str:
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def verdict_from_payload(
    payload: Any,
    *,
    attempts: int,
    criteria: str = "",
    threshold: float = DEFAULT_THRESHOLD,
    raw: str = "",
) -> JudgeVerdict:
    """把一次判官输出变成结论（**只判"能不能用"，不判"分数好不好"**）。"""
    if not isinstance(payload, Mapping):
        return JudgeVerdict(attempts=attempts, criteria=criteria, threshold=threshold,
                            error="判官没有给出 JSON 对象", raw=raw)
    score = payload.get("score")
    # ⚠️ 这里**绝不能**写 ``if not score`` —— 0 分是合法判决（参考实现的注释同一条）
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return JudgeVerdict(attempts=attempts, criteria=criteria, threshold=threshold,
                            error="判官没有给出数字 score", raw=raw)
    if not math.isfinite(float(score)):
        return JudgeVerdict(attempts=attempts, criteria=criteria, threshold=threshold,
                            error="判官的 score 不是有限数", raw=raw)
    if float(score) < 0 or float(score) > 1:
        return JudgeVerdict(attempts=attempts, criteria=criteria, threshold=threshold,
                            error=f"判官的 score 越界（{score}，应在 0..1）", raw=raw)
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return JudgeVerdict(attempts=attempts, criteria=criteria, threshold=threshold,
                            error="判官没有给出 reason", raw=raw)
    value = round(float(score), 4)
    return JudgeVerdict(score=value, reason=reason.strip(), passed=value >= threshold,
                        attempts=attempts, criteria=criteria, threshold=threshold, raw=raw)


async def judge_output(
    conn: Any,
    *,
    criteria: str,
    output: str,
    input_text: str | None = None,
    evidence: str | Sequence[Mapping[str, Any]] | None = None,
    expected: str | None = None,
    options: dict[str, Any] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    threshold: float = DEFAULT_THRESHOLD,
    system_role: str | None = None,
) -> JudgeVerdict:
    """让 LLM 按 ``criteria`` 给 ``output`` 打分（``evidence`` 可以是文本或 step 列表）。

    ``criteria`` 为空 ⇒ **不调模型**，直接 ``inconclusive``：没有标准就没有判分，
    编一个"通用好坏标准"去评创作者的戏是最坏的一种将就。
    """
    criteria = (criteria or "").strip()
    if not criteria:
        return JudgeVerdict(criteria="", threshold=threshold,
                            error="没有判定标准（criteria 为空），不做判分")
    evidence_text = (evidence if isinstance(evidence, str)
                     else evidence_from_steps(evidence)) if evidence else None
    prompt = build_judge_prompt(criteria, str(output or ""), input_text=input_text,
                                evidence=evidence_text, expected=expected)
    base = dict(options or {})
    base["system"] = system_role or base.get("system") or DEFAULT_SYSTEM_ROLE
    # 判分要可复现：除非调用方显式给温度，否则 0（同一份输出重复判分应当稳定）
    base.setdefault("temperature", 0.0)

    attempts = max(1, int(max_attempts))
    last = JudgeVerdict(criteria=criteria, threshold=threshold,
                        error="判官没有返回可用结果")
    for attempt in range(1, attempts + 1):
        try:
            raw = await generate_text(conn, prompt, base)
        except Exception as err:  # noqa: BLE001 —— 判官翻车要重试，但不能把异常当分数
            last = JudgeVerdict(attempts=attempt, criteria=criteria, threshold=threshold,
                                error=f"判官调用失败：{type(err).__name__}: {err}")
            continue
        payload, parse_error = parse_json_object(raw)
        if payload is None:
            last = JudgeVerdict(attempts=attempt, criteria=criteria, threshold=threshold,
                                error=f"判官输出不是 JSON：{parse_error}", raw=raw)
            continue
        verdict = verdict_from_payload(payload, attempts=attempt, criteria=criteria,
                                      threshold=threshold, raw=raw)
        if verdict.scored:
            return verdict
        last = verdict
    return last


def verdict_summary(verdict: JudgeVerdict) -> str:
    """一行人话（日志/报告用）。"""
    if not verdict.scored:
        return f"语义判分：判不出来（{verdict.error}）"
    return (f"语义判分：{verdict.score:.2f}（门槛 {verdict.threshold:.2f}"
            f"{'，通过' if verdict.passed else '，未通过'}）：{verdict.reason}")
