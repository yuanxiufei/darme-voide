"""LLM 语义判分（P0-3）的守卫测试。

被测：``app/agent/evaluation/judge.py`` ＋ ``text_generation.validate_json_schema`` 的
``minimum``/``maximum`` 扩展 ＋ ``evaluator._judge_spec`` 的开关口径。

四条**"看起来实现了、其实没生效"**的形状是本文件的靶子：

* ``score == 0`` 被当成"没判出来"（``if not score`` 那个经典错）⇒ 真实的 0 分被吞掉；
* 分数不理想就重试 ⇒ 判分变成抽奖（重试的触发条件只能是"判官翻车"）；
* 判不出来时编一个 0 分 ⇒ ``inconclusive`` 与 ``0.0`` 混成一个数，优化器据此改错方向；
* 证据与正文混在一段 ⇒ 判官被模型的自述带跑，"有没有真调工具"这类标准永远判不准。

跑法：``.venv/Scripts/python.exe tests/agent_judge_test.py``。
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA_ROOT = Path(tempfile.mkdtemp(prefix="agent_judge_test_"))
os.environ["DATA_ROOT"] = str(_DATA_ROOT)
os.environ.setdefault("SKILLS_DIR", str(_DATA_ROOT / "skills"))

from app.agent.evaluation import judge  # noqa: E402
from app.agent.evaluation import evaluator  # noqa: E402
from app.services.text_generation import validate_json_schema  # noqa: E402

_failures: list[str] = []
_passed = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global _passed
    if ok:
        _passed += 1
        print(f"  ✓ {label}")
        return True
    _failures.append(f"{label}{(' —— ' + detail) if detail else ''}")
    print(f"  ✗ {label}{(' —— ' + detail) if detail else ''}")
    return False


class _Scripted:
    """按顺序吐预设输出的假 ``generate_text``（并记录每次收到的 prompt 与 options）。"""

    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    async def __call__(self, conn, prompt, options=None):
        self.calls.append({"prompt": prompt, "options": dict(options or {})})
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        value = self.outputs[index]
        if isinstance(value, BaseException):
            raise value
        return value

    @property
    def count(self) -> int:
        return len(self.calls)


def with_generator(fake: _Scripted):
    """临时替换 ``judge.generate_text``（judge 是 ``from ... import`` 拿的引用，只能补它自己的名字）。"""
    original = judge.generate_text
    judge.generate_text = fake
    return original


# ---------------------------------------------------------------------------
# 一、prompt 五段
# ---------------------------------------------------------------------------

def case_prompt() -> None:
    print("\n[1] 五段式 prompt（[Input]/[Evidence]/[Expected] 可选，[Criteria]/[Output] 恒在）")
    full = judge.build_judge_prompt("冲突要够", "正文", input_text="原始剧本",
                                    evidence="1. [tool] read 参数={}", expected="至少三句")
    order = ["[Criteria]", "[Input]", "[Output]", "[Evidence]", "[Expected]"]
    positions = [full.index(tag) for tag in order]
    check("五段顺序与参考一致", positions == sorted(positions), f"{positions}")
    check("Evidence 段带用途说明（判官才知道拿它判'做了什么'）",
          "What the AI did while producing the output" in full)

    minimal = judge.build_judge_prompt("冲突要够", "正文")
    check("只给必填两段时不出现空标题",
          "[Input]" not in minimal and "[Evidence]" not in minimal
          and "[Expected]" not in minimal)
    check("必填两段仍在", "[Criteria]" in minimal and "[Output]" in minimal)

    empty_evidence = judge.build_judge_prompt("c", "o", evidence="")
    check("空证据不落一个空 [Evidence]", "[Evidence]" not in empty_evidence)


def case_evidence() -> None:
    print("\n[2] 证据渲染（只留可核对的事实，截断必须说出来）")
    steps = [
        {"kind": "tool", "tool": "read_script", "args": {"episodeId": 7}, "chars": 812},
        {"kind": "tool", "tool": "save_shots", "ok": False, "error": "permission denied"},
        {"kind": "nudge", "problem": "missing-write"},
    ]
    text = judge.evidence_from_steps(steps)
    check("工具名进证据", "read_script" in text and "save_shots" in text)
    check("参数进证据", "episodeId" in text)
    check("失败与错误进证据", "permission denied" in text and "失败" in text)
    check("其它字段（problem）不丢", "missing-write" in text)

    many = [{"kind": "tool", "tool": f"t{i}"} for i in range(100)]
    truncated = judge.evidence_from_steps(many, limit=10)
    check("超长截断时明说省略了多少步", "中间省略" in truncated and "90" in truncated)
    check("截断后留头留尾", "t0" in truncated and "t99" in truncated)
    check("空步骤 ⇒ 空证据", judge.evidence_from_steps(None) == "")


# ---------------------------------------------------------------------------
# 二、判决解析（0 分合法）
# ---------------------------------------------------------------------------

def case_payload() -> None:
    print("\n[3] 判决解析：0 分合法 / 越界与 NaN 拒 / 只判能不能用")
    zero = judge.verdict_from_payload({"score": 0, "reason": "完全没做到"}, attempts=1)
    check("**score=0 是合法判决**（不是解析失败）", zero.scored and zero.score == 0.0,
          f"score={zero.score} verdict={zero.verdict}")
    check("0 分 ⇒ passed=False", zero.passed is False)
    check("0 分保留理由", zero.reason == "完全没做到")

    one = judge.verdict_from_payload({"score": 1, "reason": "到位"}, attempts=1)
    check("1 分 ⇒ passed=True", one.scored and one.passed is True)

    edge = judge.verdict_from_payload({"score": 0.6, "reason": "勉强"}, attempts=1,
                                      threshold=0.6)
    check("恰好等于门槛 ⇒ 通过（>= 而非 >）", edge.passed is True)

    below = judge.verdict_from_payload({"score": 0.59, "reason": "差一点"}, attempts=1,
                                       threshold=0.6)
    check("差一点 ⇒ 不通过", below.passed is False)

    for label, payload in (
        ("没有 score", {"reason": "x"}),
        ("score 是字符串", {"score": "0.5", "reason": "x"}),
        ("score 是 bool", {"score": True, "reason": "x"}),
        ("score 是 NaN", {"score": float("nan"), "reason": "x"}),
        ("score 越界 1.5", {"score": 1.5, "reason": "x"}),
        ("score 负数", {"score": -0.1, "reason": "x"}),
        ("没有 reason", {"score": 0.5}),
        ("reason 空白", {"score": 0.5, "reason": "   "}),
        ("payload 不是对象", ["0.5"]),
    ):
        verdict = judge.verdict_from_payload(payload, attempts=1)
        check(f"{label} ⇒ inconclusive 且带原因",
              (not verdict.scored) and verdict.verdict == "inconclusive" and bool(verdict.error),
              f"{verdict.as_dict()}")

    check("判决回执给优化器读的字段齐",
          {"verdict", "score", "reason", "passed", "attempts", "threshold", "error"}
          <= set(zero.as_dict()))


def case_schema_bounds() -> None:
    print("\n[4] schema 的 minimum/maximum（判分 schema 的 0..1 界真的在拦）")
    schema = judge.JUDGE_SCHEMA
    check("合规 payload 通过", validate_json_schema({"score": 0.5, "reason": "ok"}, schema) == [])
    check("score=0 通过（不被当缺字段）",
          validate_json_schema({"score": 0, "reason": "ok"}, schema) == [])
    check("score=5 被拦", bool(validate_json_schema({"score": 5, "reason": "ok"}, schema)))
    check("score=-1 被拦", bool(validate_json_schema({"score": -1, "reason": "ok"}, schema)))
    check("缺 reason 被拦", bool(validate_json_schema({"score": 0.5}, schema)))
    check("空 reason 被 minLength 拦",
          bool(validate_json_schema({"score": 0.5, "reason": ""}, schema)))
    check("bool 不会被当成 number 混过 minimum",
          bool(validate_json_schema(True, {"type": "number", "minimum": 0})))
    check("整数型的界照样生效",
          bool(validate_json_schema(3, {"type": "integer", "maximum": 2})))


# ---------------------------------------------------------------------------
# 三、判分主流程（只对"判官翻车"重试）
# ---------------------------------------------------------------------------

def case_judge_loop() -> None:
    print("\n[5] 判分主流程（重试只针对判官翻车）")
    conn = object()

    fake = _Scripted('{"score": 0.8, "reason": "冲突清楚"}')
    original = with_generator(fake)
    try:
        verdict = asyncio.run(judge.judge_output(conn, criteria="冲突要够", output="正文",
                                                 input_text="原剧本", expected="三句内进冲突"))
    finally:
        judge.generate_text = original
    check("一次成功 ⇒ attempts=1", verdict.attempts == 1 and verdict.scored)
    check("分数与理由原样带回", verdict.score == 0.8 and verdict.reason == "冲突清楚")
    check("默认温度 0（重复判分要稳）", fake.calls[0]["options"].get("temperature") == 0.0)
    check("默认 system role 是判官口径",
          "expert evaluation judge" in str(fake.calls[0]["options"].get("system")))
    check("prompt 里五段都在（给了 Input/Expected）",
          all(tag in fake.calls[0]["prompt"] for tag in
              ("[Criteria]", "[Input]", "[Output]", "[Expected]")))

    low = _Scripted('{"score": 0.1, "reason": "很弱"}')
    original = with_generator(low)
    try:
        verdict = asyncio.run(judge.judge_output(conn, criteria="c", output="o"))
    finally:
        judge.generate_text = original
    check("**低分不重试**（只跑一次，分数就是分数）", low.count == 1 and verdict.score == 0.1,
          f"calls={low.count}")

    flaky = _Scripted("这不是 JSON", '{"score": "高", "reason": "x"}',
                      '{"score": 0.7, "reason": "第三次才对"}')
    original = with_generator(flaky)
    try:
        verdict = asyncio.run(judge.judge_output(conn, criteria="c", output="o"))
    finally:
        judge.generate_text = original
    check("前两次翻车 ⇒ 第三次成功，attempts=3",
          verdict.scored and verdict.attempts == 3 and flaky.count == 3,
          f"attempts={verdict.attempts} calls={flaky.count}")

    dead = _Scripted("完全不是 JSON")
    original = with_generator(dead)
    try:
        verdict = asyncio.run(judge.judge_output(conn, criteria="c", output="o"))
    finally:
        judge.generate_text = original
    check("三次都翻车 ⇒ **inconclusive、不是 0 分**",
          (not verdict.scored) and verdict.score is None and verdict.attempts == 3,
          f"score={verdict.score} error={verdict.error}")
    check("inconclusive 带得上原因", "JSON" in verdict.error)
    check("默认重试上限 3 次（参考 maxAttempts=3）", dead.count == 3, f"{dead.count}")

    boom = _Scripted(RuntimeError("模型 500"), '{"score": 0.9, "reason": "补上了"}')
    original = with_generator(boom)
    try:
        verdict = asyncio.run(judge.judge_output(conn, criteria="c", output="o"))
    finally:
        judge.generate_text = original
    check("调用抛错也重试（异常不能当分数）",
          verdict.scored and verdict.score == 0.9 and boom.count == 2)

    out_of_range = _Scripted('{"score": 5, "reason": "越界"}',
                             '{"score": 0.4, "reason": "改好了"}')
    original = with_generator(out_of_range)
    try:
        verdict = asyncio.run(judge.judge_output(conn, criteria="c", output="o"))
    finally:
        judge.generate_text = original
    check("越界分触发重试（不是直接收下）",
          out_of_range.count == 2 and verdict.score == 0.4, f"{out_of_range.count}")

    no_criteria = _Scripted('{"score": 0.9, "reason": "x"}')
    original = with_generator(no_criteria)
    try:
        verdict = asyncio.run(judge.judge_output(conn, criteria="   ", output="o"))
    finally:
        judge.generate_text = original
    check("**没有标准 ⇒ 一次模型都不调**", no_criteria.count == 0, f"{no_criteria.count}")
    check("没有标准 ⇒ inconclusive 并说明原因",
          (not verdict.scored) and "criteria" in verdict.error, verdict.error)

    custom = _Scripted('{"score": 0.5, "reason": "x"}')
    original = with_generator(custom)
    try:
        asyncio.run(judge.judge_output(conn, criteria="c", output="o",
                                       options={"temperature": 0.7, "maxTokens": 64},
                                       system_role="你是审片"))
    finally:
        judge.generate_text = original
    check("调用方显式给的温度不被覆盖", custom.calls[0]["options"]["temperature"] == 0.7)
    check("其余 options 原样透传", custom.calls[0]["options"]["maxTokens"] == 64)
    check("system_role 可换", custom.calls[0]["options"]["system"] == "你是审片")

    summary = judge.verdict_summary(
        judge.JudgeVerdict(score=0.5, reason="一半", passed=False, threshold=0.6))
    check("人话摘要能读", "语义判分" in summary and "0.50" in summary, summary)
    check("判不出来时摘要把原因说出来",
          "判不出来" in judge.verdict_summary(judge.JudgeVerdict(error="超时")))


def case_evidence_wired() -> None:
    print("\n[6] step 列表直接喂进 [Evidence]（不要求调用方自己渲染）")
    conn = object()
    fake = _Scripted('{"score": 0.6, "reason": "调了工具"}')
    original = with_generator(fake)
    try:
        asyncio.run(judge.judge_output(
            conn, criteria="必须真调工具", output="我已保存",
            evidence=[{"kind": "tool", "tool": "save_shots", "chars": 120}]))
    finally:
        judge.generate_text = original
    prompt = fake.calls[0]["prompt"]
    check("证据段出现", "[Evidence]" in prompt)
    check("证据里能看见工具名与产物量", "save_shots" in prompt and "120" in prompt)
    check("正文与证据仍是两段（不去重、不合并）",
          prompt.index("[Output]") < prompt.index("[Evidence]"))


# ---------------------------------------------------------------------------
# 四、接进评测（开关口径）
# ---------------------------------------------------------------------------

def case_evaluator_switch() -> None:
    print("\n[7] evaluator 的开关是 `judge` 而**不是** `criteria`")
    check("没写 judge ⇒ 不判", evaluator._judge_spec({"id": "x"}) is None)
    check("judge 写成字符串 ⇒ 就是 criteria",
          evaluator._judge_spec({"judge": "冲突要够"})["criteria"] == "冲突要够")
    spec = evaluator._judge_spec({"judge": {"criteria": "c", "expected": "e", "threshold": 0.7}})
    check("judge 写成 dict ⇒ 三段都带上",
          spec == {"criteria": "c", "expected": "e", "threshold": 0.7}, f"{spec}")
    check("judge 里 criteria 空 ⇒ 不判（不拿通用标准凑）",
          evaluator._judge_spec({"judge": {"criteria": "  "}}) is None)
    check("judge 写成别的类型 ⇒ 不判", evaluator._judge_spec({"judge": 5}) is None)

    legacy_case = {"criteria": ["保留人物关系", "开场三句内进冲突"]}
    check("现存 case 的 `criteria`（规则词表）**不会**意外触发判分",
          evaluator._judge_spec(legacy_case) is None)

    text = evaluator._judge_input({
        "statement": {"script": "原始剧本", "mode": "storyboard",
                      "characters": ["凌云", "若云"]},
    })
    check("[Input] 段带剧本与模式", "原始剧本" in text and "storyboard" in text)
    check("[Input] 段带角色表", "凌云" in text)
    check("空 statement ⇒ 空 Input", evaluator._judge_input({}) == "")

    source = (Path(_ROOT) / "app" / "agent" / "evaluation" / "evaluator.py").read_text(
        encoding="utf-8")
    check("evaluate_case 真调 judge_output", "await judge_output(" in source)
    check("judge 结果并排挂 report[\"judge\"]（不与规则分混算）",
          'report["judge"]' in source)
    check("判分的 options 不透传被测模型的设置（温度要固定）",
          "options={}" in source)


def case_report_answer() -> None:
    print("\n[8] 报告里留了判分要用的 [Output] 正文")
    from app.agent.runtime import AgentRunResult

    result = AgentRunResult(model="m", text="我保存了 12 个镜头", tool_calls=[], tool_results=[],
                            protocol=None, protocol_errors=[], usage=None)
    report: dict = {"caseId": "x"}
    evaluator._report_steps(report, result)
    check("answer 取模型的最终正文", report["answer"] == "我保存了 12 个镜头")
    check("steps 与 stepStats 照旧",
          "steps" in report and "stepStats" in report)

    empty = AgentRunResult(model="m", text="", tool_calls=[], tool_results=[],
                          protocol=None, protocol_errors=[], usage=None)
    report2: dict = {}
    evaluator._report_steps(report2, empty)
    check("正文为空 ⇒ 空串而不是 None（下游 str() 不会得到 'None'）",
          report2["answer"] == "")


def main() -> int:
    case_prompt()
    case_evidence()
    case_payload()
    case_schema_bounds()
    case_judge_loop()
    case_evidence_wired()
    case_evaluator_switch()
    case_report_answer()

    total = _passed + len(_failures)
    print(f"\nSUMMARY: {_passed}/{total} passed")
    if _failures:
        print("失败项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    sys.exit(main())
