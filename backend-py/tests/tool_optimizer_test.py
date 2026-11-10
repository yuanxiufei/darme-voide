"""工具集调优（缺口 4 **第三格**）的守卫测试。

被测：``tool_optimizer.py`` + ``evaluator.step_trace``（单步对照）+ ``evaluate_case(tools=…)``
+ ``runtime`` 的 ``toolsRaw`` 按次覆盖 + ``ToolRegistry.only`` + ``tune_runtime`` 三目标 + 路由。

盯的是这几处「看起来和技能那格一样、实际不一样」：判据要**两把尺子**（整轮分 + 逐步账里的
``unknownTool``）；候选**不是静态枚举**（源头是基线这一跑真被调用过的工具）；覆盖体是**工具名
数组**而非对象数组；**落库恒 false**（``agent_configs`` 没有 tools 列，真源是代码装配语句）。

跑法：``.venv/Scripts/python.exe tests/tool_optimizer_test.py``。
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: 数据根必须在 **import app.core.db 之前**指到临时目录（否则动真库）
_DATA_ROOT = Path(tempfile.mkdtemp(prefix="tool_optimizer_test_"))
os.environ["DATA_ROOT"] = str(_DATA_ROOT)
os.environ.setdefault("SKILLS_DIR", str(_DATA_ROOT / "skills"))

from sqlalchemy import text  # noqa: E402

from app.core import db  # noqa: E402
from app.agent import runtime  # noqa: E402
from app.agent.evaluation import evaluator  # noqa: E402
from app.agent.evaluation import tool_optimizer as to  # noqa: E402
from app.agent.evaluation import tune_runtime  # noqa: E402
from app.agent.tool import Tool, ToolRegistry  # noqa: E402
from app.services import optimization_log  # noqa: E402

AGENT = "script_rewriter"
CASE = {"id": "script-rewriter-001", "kind": "script_rewriter",
        "input": {"chapter": "第一章", "idea": "把开场改得更狠"},
        "criteria": ["保留人物关系", "开场三句内进冲突"]}
PROMPT = "你是剧本改写助手。只输出 Markdown 正文，不要解释。"

#: 基线轨迹：search 调两次、其中一次失败 ⇒ 它是最可疑候选（第一排序键是「失败多的先摘」）
BASE_STEPS = [
    {"index": 0, "tool": "search_script_library", "ok": True, "unknownTool": False, "error": None},
    {"index": 1, "tool": "search_script_library", "ok": False, "unknownTool": False,
     "error": "query 太短"},
    {"index": 2, "tool": "read_episode", "ok": True, "unknownTool": False, "error": None},
    {"index": 3, "tool": "save_script", "ok": True, "unknownTool": False, "error": None},
]
REMAINING = ("read_episode", "save_script")

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


def _stats_of(steps: list[dict]) -> dict:
    """按 ``step_trace`` 口径自算（**不调它**，免得用被测物证被测物）。"""
    order: list[str] = []
    for step in steps:
        if step["tool"] not in order:
            order.append(step["tool"])
    return {"total": len(steps), "ok": sum(1 for s in steps if s["ok"] is True),
            "failed": sum(1 for s in steps if s["ok"] is False),
            "unjudged": sum(1 for s in steps if s["ok"] is None),
            "unknownTool": sum(1 for s in steps if s.get("unknownTool")), "tools": order}


def _report(score: float, *, steps: list[dict] | None = None, unknown: int = 0,
            unavailable: str | None = None) -> dict:
    """造一份与 ``evaluate_case`` 同形的报告；``unknown``>0 时把末尾几步改写成撞 Unknown tool。

    ``unavailable`` 非空 ⇒ 造「这一跑没有逐步明细」的报告（``step_trace`` 收到计数时的真形状：
    ``steps=[]``、计数全 0、``stats["unavailable"]`` 说明原因）。
    """
    steps = [dict(step) for step in (steps if steps is not None else BASE_STEPS)]
    for index in range(min(unknown, len(steps))):
        step = steps[len(steps) - 1 - index]
        step.update({"ok": False, "unknownTool": True, "error": f"Unknown tool: {step['tool']}"})
    stats = _stats_of(steps)
    if unavailable:
        steps = []
        stats = {"total": 0, "ok": 0, "failed": 0, "unjudged": 0, "unknownTool": 0,
                 "tools": [], "unavailable": unavailable}
    return {"caseId": CASE["id"], "total": score, "runtimeModel": "ollama/qwen3:14b",
            "steps": steps, "stepStats": stats}


def _reset() -> None:
    with db.engine.begin() as conn:
        conn.execute(text("DELETE FROM agent_optimizations"))
        conn.execute(text("DELETE FROM agent_configs"))


def _seed_config() -> None:
    now = "2026-11-08T00:00:00+00:00"
    with db.engine.begin() as conn:
        conn.execute(
            text("INSERT INTO agent_configs (agent_type, name, description, system_prompt, "
                 "temperature, max_tokens, skills, is_active, created_at, updated_at) "
                 "VALUES (:t, :n, '', :p, 0.7, 4096, '[]', 1, :now, :now)"),
            {"t": AGENT, "n": "剧本改写助手", "p": PROMPT, "now": now},
        )


def _opt_rows() -> list[dict]:
    with db.engine.connect() as conn:
        rows = conn.execute(text("SELECT target, reason, version, score_before, score_after "
                                 "FROM agent_optimizations ORDER BY id")).mappings().all()
    return [dict(row) for row in rows]


def make_eval(script: dict, *, unknown: int = 0, fail_keys: set | None = None,
              steps: list[dict] | None = None, unavailable: str | None = None,
              baseline_unavailable: str | None = None):
    """假评测器：按「这一跑实际暴露的工具集」给分（``script`` 键＝工具名元组，基线是 None）。

    覆盖体若形状不对会当场 ``AssertionError`` —— 静默退化成「全开」会让判决全是假的。
    """
    used: dict = {}
    calls: list[dict] = []

    async def run(conn, case_def, instructions, settings=None, *, tools=None):
        ids = None if tools is None else tuple(json.loads(tools))
        calls.append({"tools": ids, "settings": settings, "prompt": instructions})
        if fail_keys and ids in fail_keys:
            raise RuntimeError(f"评测炸了：{ids}")
        if ids not in script:
            raise AssertionError(f"没给工具集 {ids} 安排分数")
        index = used.get(ids, 0)
        used[ids] = index + 1
        totals = script[ids]
        return _report(totals[index % len(totals)], steps=steps,
                       unknown=0 if ids is None else unknown,
                       unavailable=(baseline_unavailable if ids is None else unavailable))

    run.calls = calls  # type: ignore[attr-defined]
    return run


def _optimize(options: dict, *, evaluate, agent_type: str = AGENT, case_def=None, **kwargs):
    with db.engine.begin() as conn:
        return asyncio.run(to.optimize_agent_tools(
            conn, agent_type, case_def or CASE, options, evaluate=evaluate, **kwargs))


# ---------------------------------------------------------------- ① 单步对照


def case_step_trace() -> None:
    print("\n① 单步对照：整轮轨迹拆成逐步的账（工具集那格的唯一依据）")
    ok_r, err_r = json.dumps({"ok": True}), json.dumps({"error": "query 太短"})
    steps, stats = evaluator.step_trace(
        [{"toolName": "a"}, {"toolName": "b"}, {"toolName": "a"}],
        [{"toolName": "a", "result": ok_r}, {"toolName": "b", "result": err_r},
         {"toolName": "a", "result": ok_r}])
    check("①-1 三步都判出来", stats["total"] == 3 and stats["ok"] == 2 and stats["failed"] == 1,
          str(stats))
    check("①-2 失败步带出错误正文", steps[1]["error"] == "query 太短", str(steps[1]))
    check("①-3 工具名去重保序", stats["tools"] == ["a", "b"], str(stats["tools"]))

    steps, stats = evaluator.step_trace(
        [{"toolName": "search_script_library"}],
        [{"toolName": "search_script_library",
          "result": json.dumps({"error": "Unknown tool: search_script_library"})}])
    check("①-4 Unknown tool 单独标记（「摘掉后模型还在调它」的直接证据）",
          steps[0]["unknownTool"] is True and stats["unknownTool"] == 1, str(steps[0]))

    steps, stats = evaluator.step_trace([{"toolName": "a"}, {"toolName": "b"}],
                                        [{"toolName": "a", "result": ok_r}])
    check("①-5 少了结果的那步判「未判」而非失败", steps[1]["ok"] is None
          and stats["unjudged"] == 1 and stats["failed"] == 0, str(stats))

    steps, _ = evaluator.step_trace([{"toolName": "a"}], [{"toolName": "a", "result": "纯文本"}])
    check("①-6 结果不是 JSON 时不臆造错误", steps[0]["ok"] is True, str(steps[0]))

    steps, stats = evaluator.step_trace(
        [{"toolName": "search_corpus"}],
        [{"toolName": "search_corpus", "result": json.dumps({"available": False})}])
    check("①-7 工具降级（available:false）不算失败", steps[0]["ok"] is True
          and stats["failed"] == 0, str(steps[0]))

    # ①-8/9：真事故回归 —— 盘上 `<dataRoot>/traces/*.jsonl` 的 `meta.toolCalls` 是**整数计数**
    # （2026-11-08 实测 61/61 条真实轨迹都是 int，逐步明细根本没落盘）。
    steps, stats = evaluator.step_trace(7, None)  # type: ignore[arg-type]
    check("①-8 传进来计数（真轨迹的形状）不炸，且如实说没有逐步明细",
          steps == [] and stats["unavailable"] == "toolCallsNotList" and stats["total"] == 0,
          str(stats))
    check("①-9 正常数组时 unavailable=None（别把「有明细」也标成缺）",
          evaluator.step_trace([{"toolName": "a"}], [])[1]["unavailable"] is None)

    steps, stats = evaluator.step_trace(["裸字符串", None], [None])  # type: ignore[list-item]
    check("①-10 坏形状的步照旧占位（不会因为一条脏数据丢整轮账）",
          stats["total"] == 2 and steps[0]["tool"] == "" and steps[1]["ok"] is None, str(stats))


# ---------------------------------------------------------------- ② 候选


def case_candidates() -> None:
    print("\n② 候选：只从基线轨迹取，按可疑度排，不许摘成空")
    report = _report(80.0)
    check("②-1 current_tools 取 stepStats.tools",
          to.current_tools(report) == ["search_script_library", "read_episode", "save_script"],
          str(to.current_tools(report)))
    check("②-2 老报告无 stepStats 时退回 steps 扫",
          to.current_tools({"total": 80.0, "steps": report["steps"]}) ==
          ["search_script_library", "read_episode", "save_script"])
    cands = to.tool_candidates(report, max_candidates=3)
    check("②-3 失败步多的先摘", cands[0] == "search_script_library", str(cands))
    check("②-4 其余按「次数少 → 靠后」",
          cands == ["search_script_library", "save_script", "read_episode"], str(cands))
    check("②-5 max_candidates 截断",
          to.tool_candidates(report, max_candidates=1) == ["search_script_library"])
    check("②-6 max_candidates=0 ⇒ 零候选", to.tool_candidates(report, max_candidates=0) == [])
    check("②-7 只剩一个工具 ⇒ 零候选（不许摘成空）",
          to.tool_candidates(_report(80.0, steps=[BASE_STEPS[0]])) == [])
    check("②-8 没有轨迹 ⇒ 零候选（不从别处猜工具名）",
          to.tool_candidates({"total": 80.0}) == [])


# ---------------------------------------------------------------- ③ 判据


def case_direction() -> None:
    print("\n③ 判据：整轮分 + 逐步账（tool_wanted 这档技能那格没有）")
    _reset()
    _seed_config()
    receipt = _optimize({"repeat": 2, "maxCandidates": 1},
                        evaluate=make_eval({None: [80.0], REMAINING: [80.0, 80.0]}))
    row = receipt["candidates"][0]
    check("③-1 分数没掉且轨迹无 Unknown tool ⇒ pruned",
          row["accepted"] is True and row["reason"] == "pruned", str(row))
    check("③-2 accepted 带 persistSkip", "tools 列" in str((receipt["accepted"] or {}).get("persistSkip")))
    check("③-3 persisted 恒 false", receipt["persisted"] is False)
    check("③-4 优化史落 target=tools / reason=pruned",
          any(r["target"] == "tools" and r["reason"] == "pruned" for r in _opt_rows()),
          str(_opt_rows()))

    _reset()
    _seed_config()
    row = _optimize({"repeat": 2, "maxCandidates": 1},
                    evaluate=make_eval({None: [80.0], REMAINING: [40.0, 38.0]}))["candidates"][0]
    check("③-5 明显掉分 ⇒ tool_needed（不采纳）",
          row["accepted"] is False and row["reason"] == "tool_needed", str(row))

    _reset()
    _seed_config()
    receipt = _optimize({"repeat": 2, "maxCandidates": 1},
                        evaluate=make_eval({None: [80.0], REMAINING: [80.0, 80.0]}, unknown=2))
    row = receipt["candidates"][0]
    check("③-6 分数没掉但仍有 Unknown tool ⇒ **不采纳**（tool_wanted + 全部候选）",
          row["accepted"] is False and row["reason"] == "tool_wanted"
          and receipt["accepted"] is None, str(row))
    check("③-7 unknownTool 计数进回执（判决证据可复核）", row["unknownTool"] == 4, str(row["unknownTool"]))
    check("③-8 tool_wanted 也如实落优化史",
          any(r["reason"] == "tool_wanted" and r["target"] == "tools" for r in _opt_rows()))

    _reset()
    _seed_config()
    receipt = _optimize({"repeat": 2, "maxCandidates": 2}, evaluate=make_eval(
        {None: [80.0], REMAINING: [80.0, 80.0],
         ("search_script_library", "read_episode"): [60.0, 60.0]}))
    check("③-9 抖动带取「同配置重复样本」的离散度（两候选各自两跑都一致 ⇒ 0），且写明来源",
          receipt["jumpKind"] == "within" and receipt["jump"] == 0.0,
          f"jump={receipt['jump']} kind={receipt['jumpKind']}")
    check("③-10 采纳即停（只认最先判采纳的那个）",
          len([r for r in receipt["candidates"] if r["accepted"]]) == 1)

    # ③-9b～d：⭐ 真模型实测反例（2026-11-09，qwen3:14b / sb-case-001）。
    # 基线 95.00；候选 A 摘 `save_continuity_states` ⇒ [80, 88]（均值 84，**真掉了 11 分**）；
    # 候选 B 摘 `read_storyboard_context` ⇒ 崩到 [0, 0]。旧口径把「候选之间的极差」当噪声：
    # jump=(84−0)/2=42 ⇒ A 的 −11 落在带内 ⇒ 判 pruned **采纳摘除** —— 那是在真数据上
    # 错摘一个真在用的工具。跨候选差距是**信号**（摘关键工具 vs 摘边缘工具本来就该差很多），
    # 只有同一配置重复测量之间的差才是噪声。
    _reset()
    _seed_config()
    receipt = _optimize({"repeat": 2, "maxCandidates": 2}, evaluate=make_eval(
        {None: [95.0], REMAINING: [80.0, 88.0],
         ("search_script_library", "read_episode"): [0.0, 0.0]}))
    by_tool = {r["tool"]: r for r in receipt["candidates"]}
    first = by_tool.get("search_script_library")
    check("③-9b ⭐ 真掉 11 分不许判 pruned（跨候选极差不得充噪声）",
          first is not None and first["delta"] == -11.0
          and first["accepted"] is False and first["reason"] == "tool_needed",
          str(first))
    check("③-9c 抖动带按候选内部重复样本算（[80,88] ⇒ max−min=8，不折半），不被崩到 0 的候选撑大",
          receipt["jump"] == 8.0 and receipt["jumpKind"] == "within",
          f"jump={receipt['jump']} kind={receipt['jumpKind']}")
    check("③-9d 该轮零采纳（「没测出差别」不能覆盖「真掉分」）",
          receipt["accepted"] is None, str(receipt["accepted"]))

    _reset()
    _seed_config()
    receipt = _optimize({"repeat": 2}, evaluate=make_eval({None: [80.0]}, steps=[BASE_STEPS[0]]))
    check("③-11 只剩一个工具 ⇒ 不跑候选、回执给理由",
          receipt["candidates"] == [] and "空工具集" in str(receipt.get("reason")),
          str(receipt.get("reason")))

    # ③-12 / 13：**逐步账没明细 ≠ 模型没再调它**。候选那一跑的驱动只回灌了个数时，
    # `unknownTool` 恒 0 —— 只靠整轮分必然得出「所有工具都能摘」，所以这时**不许判 pruned**。
    _reset()
    _seed_config()
    receipt = _optimize({"repeat": 2, "maxCandidates": 1},
                        evaluate=make_eval({None: [80.0], REMAINING: [80.0, 80.0]},
                                           unavailable="toolCallsNotList"))
    row = receipt["candidates"][0]
    check("③-12 逐步账没明细 ⇒ 不采纳（tool_unjudged，不是 pruned）",
          row["accepted"] is False and row["reason"] == "tool_unjudged"
          and receipt["accepted"] is None, str(row))
    check("③-13 缺明细如实进回执（`stepsUnavailable`）与优化史",
          row["stepsUnavailable"] == "toolCallsNotList"
          and any(r["reason"] == "tool_unjudged" and r["target"] == "tools" for r in _opt_rows()),
          f"row={row.get('stepsUnavailable')} rows={_opt_rows()}")
    check("③-14 有明细时 `stepsUnavailable` 是 None（别把「有账」也标成缺）",
          _optimize({"repeat": 2, "maxCandidates": 1},
                    evaluate=make_eval({None: [80.0], REMAINING: [80.0, 80.0]}))["candidates"][0]
          ["stepsUnavailable"] is None)

    # ③-15 / 16 / 17：进度分母。本格**基线只跑 1 次**（不吃 repeat），候选数要基线跑完才知道
    # ⇒ 抄技能那格的 `repeat*(1+n)` 会「基线多算 repeat−1 跑 + 候选按上限算」，
    # 症状是任务已 done、页面永远停在 5/6（不是崩溃，所以没人会去修）。
    _reset()
    _seed_config()
    events: list[dict] = []
    _optimize({"repeat": 2, "maxCandidates": 2}, progress=events.append, evaluate=make_eval(
        {None: [80.0], REMAINING: [80.0, 80.0],
         ("search_script_library", "read_episode"): [60.0, 60.0]}))
    done = [e for e in events if e.get("type") == "done"]
    check("③-15 分母＝1＋repeat×真候选数（2 候选取 5，不是 repeat×(1+2)=6）⇒ 终态 100%",
          bool(done) and done[-1]["doneRuns"] == done[-1]["totalRuns"] == 5,
          str([(e.get("doneRuns"), e.get("totalRuns")) for e in events]))

    _reset()
    _seed_config()
    events = []
    _optimize({"repeat": 2}, progress=events.append,
              evaluate=make_eval({None: [80.0]}, steps=[BASE_STEPS[0]]))
    done = [e for e in events if e.get("type") == "done"]
    check("③-16 没有候选时也是 100%（分母＝那 1 跑基线，不是 6）",
          bool(done) and done[-1]["doneRuns"] == done[-1]["totalRuns"] == 1,
          str([(e.get("doneRuns"), e.get("totalRuns")) for e in events]))
    check("③-17 进度不倒退、doneRuns 不越过分母（页面按它算百分比）",
          all(int(e.get("doneRuns") or 0) <= int(e.get("totalRuns") or 0) for e in events),
          str([(e.get("doneRuns"), e.get("totalRuns")) for e in events]))

    # ③-18 / 19 / 20：基线就没有逐步明细 ⇒ 本格**整格判不了**（第二把尺子没了）。
    # 必须在基线处收工：否则烧完 repeat×候选数 跑才换来一句「判不了」。而且理由**不许**
    # 写成「空工具集」—— 缺明细时 `current_tools` 同样是空的，照空分支走会把
    # 「没记账」说成「模型没调过任何工具」，两个事实正相反。
    _reset()
    _seed_config()
    events = []
    ev = make_eval({None: [80.0]}, baseline_unavailable="toolCallsNotList")
    receipt = _optimize({"repeat": 2, "maxCandidates": 2}, evaluate=ev, progress=events.append)
    check("③-18 基线缺明细 ⇒ 一个候选都不跑（不烧 4 跑换一句「判不了」）",
          len(ev.calls) == 1 and receipt["candidates"] == [],
          f"calls={len(ev.calls)} candidates={receipt['candidates']}")
    check("③-19 理由写「缺明细」而不是「空工具集」，缺的是什么如实进回执",
          "明细" in str(receipt.get("reason")) and "空工具集" not in str(receipt.get("reason"))
          and receipt["baseline"]["steps"]["unavailable"] == "toolCallsNotList",
          str(receipt.get("reason")))
    done = [e for e in events if e.get("type") == "done"]
    check("③-20 早退也是 100%（分母＝1，不是估的 1＋repeat×maxCandidates）",
          bool(done) and done[-1]["doneRuns"] == done[-1]["totalRuns"] == 1,
          str([(e.get("doneRuns"), e.get("totalRuns")) for e in events]))

    # ③-21 / 22：**部分**缺账（`unjudged > 0`）也必须挡。`unknownTool` 是靠**结果的 error
    # 正文**认出来的 ⇒ 那几步没有结果时，模型真在其中调了被摘的工具也读不到 ⇒ 与整格缺账
    # 同罪。旧逻辑只挡 `unavailable`，会把「部分没记账」当完整账，进而**错摘**工具。
    _reset()
    _seed_config()
    partial = [dict(step) for step in BASE_STEPS] + [
        {"index": 4, "tool": "save_script", "ok": None, "unknownTool": False, "error": None}]
    receipt = _optimize({"repeat": 2, "maxCandidates": 1},
                        evaluate=make_eval({None: [80.0, 80.0], REMAINING: [80.0, 80.0]},
                                           steps=partial))
    check("③-21 有步没回灌结果（unjudged>0）⇒ 不判 pruned，走 tool_unjudged",
          receipt["candidates"][0]["reason"] == "tool_unjudged"
          and receipt["candidates"][0]["stepsUnjudged"] == 1,
          str(receipt["candidates"][0]))
    rejected = [r for r in _opt_rows() if r["target"] == "tools"]
    check("③-22 这条不采纳也如实落库（accepted=0、reason=tool_unjudged）",
          bool(rejected) and rejected[0]["reason"] == "tool_unjudged", str(rejected))

    # ③-23 / 24：③-21 那条件成立的前提有两条，都必须钉住 ——
    # ① 驱动侧 `tool_calls`/`tool_results` **成对 append**（只在 `_run_tool` 里、相邻无条件）：
    #    否则「有调用没结果」会变成常态，③-21 的拦截反过来变成误拦；
    # ② 报告里那份账要经 `AgentRunResult.tool_results` 带出来：少这一行 ⇒ 每次评测都成了
    #    「有调用没结果」⇒ 本格永远判不了（而且看不出是缺账还是模型真没调）。
    runtime_src = (Path(_ROOT) / "app" / "agent" / "runtime.py").read_text(encoding="utf-8")
    check("③-23 运行时 calls/results 只在 `_run_tool` 里成对追加（③-21 的前提 ✓）",
          runtime_src.count("tool_calls_out.append") == 1
          and runtime_src.count("tool_results_out.append") == 1,
          f"calls={runtime_src.count('tool_calls_out.append')} "
          f"results={runtime_src.count('tool_results_out.append')}")
    check("③-24 `AgentRunResult` 把 generate 的 `tool_results` 带出（漏 ⇒ 本格永远判不了 ✗）",
          'tool_results=list(result.get("tool_results") or [])' in runtime_src)

    # ③-25：第二把尺子的**触发器**也是硬编码，而且是跨模块的两半 ——
    # 产出只有一处（`runtime._run_tool` 的 `f"Unknown tool: {name}"`），识别只有一处
    # （`evaluator._UNKNOWN_TOOL_PREFIX`）。改任何一半都不会报错，只会让 `unknownTool`
    # 恒 0 而 `failed` 照常计数 ⇒ 判决链读成「模型没再调它」⇒ **错摘**。
    # 实测反向验证：把常量改成 `No such tool` ⇒ `evaluation_route_test` 那条端到端判据
    # 立刻红，实得 `failed=1, unknownTool=0`（两把尺子当场分家）。
    evaluator_src = (Path(_ROOT) / "app" / "agent" / "evaluation"
                     / "evaluator.py").read_text(encoding="utf-8")
    check("③-25 evaluator 的前缀常量与 runtime 抛出的字面量一致"
          "（改了任何一半 ⇒ `unknownTool` 恒 0、第二把尺子静默哑掉 ✗）",
          '_UNKNOWN_TOOL_PREFIX = "Unknown tool"' in evaluator_src
          and '{"error": f"Unknown tool: {name}"}' in runtime_src,
          (f"evaluator 命中 {evaluator_src.count('_UNKNOWN_TOOL_PREFIX = \"Unknown tool\"')} 次 / "
           f"runtime {'命中' if '{\"error\": f\"Unknown tool: {name}\"}' in runtime_src else '未命中'}"))

    # ③-26：**三格抖动带的幅值口径必须一致**。2026-11-09 真跑抓到的 bug 根因正是「工具格折了半、
    # 姊妹格没折」：参数格/技能格都写 `jump = max(baseline_scores) - min(baseline_scores)`（全幅），
    # 工具格却按「两个候选均值极差的一半」算 ⇒ 同一个词「抖动带」在三格里量纲不同。
    # 只判机械事实（取极差、不折半），不钉字面量：**来源合法地不同**（参数/技能格取基线重复跑，
    # 本格基线只跑 1 次 ⇒ 取候选自己那 repeat 个样本），但幅值不许一边折半。
    # 反向验证：把本格的 spreads 改回 `/ 2` ⇒ 本条精确报红。
    def _assign_value(src_path: Path, name: str) -> ast.AST | None:
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return node.value
        return None

    def _has_maxmin(node: ast.AST | None) -> bool:
        calls = {n.func.id for n in ast.walk(node)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)} if node else set()
        return {"max", "min"} <= calls

    def _no_half(node: ast.AST | None) -> bool:
        if node is None:
            return False
        return not any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) for n in ast.walk(node))

    _eval_dir = Path(_ROOT) / "app" / "agent" / "evaluation"
    param_jump = _assign_value(_eval_dir / "param_optimizer.py", "jump")
    skill_jump = _assign_value(_eval_dir / "skill_optimizer.py", "jump")
    tool_jump = _assign_value(_eval_dir / "tool_optimizer.py", "jump")
    tool_spreads = _assign_value(_eval_dir / "tool_optimizer.py", "spreads")
    check("③-26 三格抖动带同口径：都取极差（max−min）、都不折半",
          _has_maxmin(param_jump) and _no_half(param_jump)
          and _has_maxmin(skill_jump) and _no_half(skill_jump)
          and _no_half(tool_jump) and _has_maxmin(tool_spreads) and _no_half(tool_spreads),
          f"param={_has_maxmin(param_jump) and _no_half(param_jump)} "
          f"skill={_has_maxmin(skill_jump) and _no_half(skill_jump)} "
          f"tool={_no_half(tool_jump) and _has_maxmin(tool_spreads) and _no_half(tool_spreads)}")


def case_candidate_errors() -> None:
    print("\n③′ 候选评测失败 ⇒ eval_failed，不静默当「没掉分」")
    _reset()
    _seed_config()
    row = _optimize({"repeat": 2, "maxCandidates": 1},
                    evaluate=make_eval({None: [80.0]}, fail_keys={REMAINING}))["candidates"][0]
    check("③′-1 抛错 ⇒ eval_failed 且不采纳",
          row["reason"] == "eval_failed" and row["accepted"] is False, str(row))
    check("③′-2 照样落优化史", any(r["reason"] == "eval_failed" for r in _opt_rows()))


def case_not_persisted() -> None:
    print("\n④ 落库恒 false + 基线不传覆盖")
    _reset()
    _seed_config()
    receipt = _optimize({"repeat": 2, "maxCandidates": 1, "autoPersist": True},
                        evaluate=make_eval({None: [80.0], REMAINING: [80.0, 80.0]}))
    check("④-1 autoPersist=True 也不认", receipt["persisted"] is False)
    check("④-2 开关被钉死成 False",
          to.normalize_tool_options({"autoPersist": True})["autoPersist"] is False)
    with db.engine.connect() as conn:
        skills = conn.execute(text("SELECT skills FROM agent_configs")).scalar()
        rows = conn.execute(text("SELECT COUNT(*) FROM agent_configs")).scalar()
    check("④-3 agent_configs 一字未改", rows == 1 and skills == "[]", f"rows={rows} skills={skills}")

    _reset()
    _seed_config()
    ev = make_eval({None: [80.0], REMAINING: [80.0, 80.0]})
    _optimize({"repeat": 2, "maxCandidates": 1}, evaluate=ev)
    check("④-4 基线那一跑不传 tools 覆盖", ev.calls[0]["tools"] is None, str(ev.calls[0]["tools"]))
    check("④-5 候选那几跑真换了名册", ev.calls[1]["tools"] == REMAINING, str(ev.calls[1]["tools"]))
    check("④-6 总跑次 = 基线 1 + 候选 × repeat", len(ev.calls) == 3, str(len(ev.calls)))

    _reset()
    _seed_config()
    try:
        _optimize({"repeat": 1}, evaluate=make_eval({None: [80.0]}))
        check("④-7 repeat<2 当场拒", False, "没有抛")
    except ValueError as err:
        check("④-7 repeat<2 当场拒", "repeat" in str(err), str(err))


# ---------------------------------------------------------------- ⑤ 覆盖体形状


def case_override_shape() -> None:
    print("\n⑤ 覆盖体：工具名 JSON 数组（与技能的对象数组不同，形状必须对拍）")
    raw = to._tools_json(["a", "b"])
    check("⑤-1 _tools_json 产出名字数组", json.loads(raw) == ["a", "b"], raw)
    check("⑤-2 runtime 能解析回去（跨模块对拍）", runtime._parse_tool_names(raw) == ["a", "b"])
    check("⑤-3 直接给 list 也收", runtime._parse_tool_names(["a"]) == ["a"])
    for bad, label in ((r'{"a":1}', "对象"), ("[a, b]", "非法 JSON"), ('"a"', "字符串")):
        try:
            runtime._parse_tool_names(bad)
            check(f"⑤-4 {label} ⇒ 当场拒", False, "没有抛")
        except ValueError:
            check(f"⑤-4 {label} ⇒ 当场拒", True)

    registry = ToolRegistry({
        "k_search": Tool(id="search_script_library", description="", input_schema={},
                         execute=lambda args: None),
        "k_read": Tool(id="read_episode", description="", input_schema={}, execute=lambda args: None),
        "k_save": Tool(id="save_script", description="", input_schema={}, execute=lambda args: None)})
    subset, _ = registry.only(["save_script", "k_read"])
    check("⑤-5 按 id（蛇形）也摘得掉，键跟着注册表走",
          subset.names() == ["read_episode", "save_script"], str(subset.names()))
    check("⑤-6 顺序跟注册表不跟名册（A/B 之间不多变量）",
          subset.ids() == ["k_read", "k_save"], str(subset.ids()))
    _, missing = registry.only(["search_script_library", "不存在"])
    check("⑤-7 名册里没有的名字如实进 missing", missing == ["不存在"], str(missing))
    check("⑤-8 names() 是 id 不是键", registry.names() ==
          ["search_script_library", "read_episode", "save_script"], str(registry.names()))


# ---------------------------------------------------------------- ⑥ 接线


def case_wiring() -> None:
    print("\n⑥ 接线：裁剪排在 MCP 合并之后 / 五个 kind 都挂 steps / 三目标登记 / 路由")
    source = (_ROOT / "app" / "agent" / "runtime.py").read_text(encoding="utf-8")
    merged_at = source.find("merged = ToolRegistry({**built.tools.tools, **discovered})")
    cut_at = source.find('if options.get("toolsRaw") is not None:')
    check("⑥-1 裁剪排在 MCP 合并之后（MCP 工具也要能摘）",
          merged_at > 0 and cut_at > merged_at, f"merged@{merged_at} cut@{cut_at}")
    check("⑥-2 覆盖走 options['toolsRaw']", cut_at > 0)

    eval_source = (_ROOT / "app" / "agent" / "evaluation" / "evaluator.py").read_text(encoding="utf-8")
    returns = [node for node in ast.walk(ast.parse(eval_source))
               if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
               and getattr(node.value.func, "id", "") == "_report_steps"]
    check("⑥-3 五个 kind 分支全部经 _report_steps 收口", len(returns) == 5, str(len(returns)))
    check("⑥-4 tools 覆盖与 skills 同口径透传",
          'run_options["toolsRaw"] = tools' in eval_source)

    check("⑥-5 tune_runtime 四目标登记（提示词格 2026-11-09 接入）",
          tune_runtime.TARGETS == ("params", "skills", "system_prompt", "tools"),
          str(tune_runtime.TARGETS))
    try:
        tune_runtime.submit(case_id=CASE["id"], case_def=CASE, agent_type=AGENT,
                            options={}, target="lora", runner=None)
        check("⑥-6 未知目标仍然当场拒", False, "没有抛")
    except ValueError as err:
        check("⑥-6 未知目标仍然当场拒", "未知调优目标" in str(err), str(err))

    check("⑥-7 optimization_log 记了 tools 这一格",
          optimization_log.TARGETS.get("tools") == "工具集")
    check("⑥-8 两个新理由码都在册",
          "tool_needed" in optimization_log.REASONS and "tool_wanted" in optimization_log.REASONS)

    router_source = (_ROOT / "app" / "routers" / "evaluation.py").read_text(encoding="utf-8")
    check("⑥-9 路由登记（POST /tune-tools/{case_id}）",
          '@router.post("/tune-tools/{case_id}")' in router_source)
    # 上界取自己的装饰器行，不取 "/tune" —— 中间可能再插别的端点（2026-11-09 /tune-prompt 就夹在两者之间）
    tune_block = router_source.split('"/tune-tools/{case_id}"')[1].split(
        '@router.post("/tune-prompt/{case_id}")')[0]
    has_target = 'target="tools"' in tune_block
    has_persist_arg = '"auto_persist"' in tune_block
    check("⑥-10 该端点只走 target=tools 且不提供 autoPersist",
          has_target and not has_persist_arg,
          f"target={has_target} autoPersist={has_persist_arg}")

    run_text = tune_runtime._stage_text(
        {"type": "run", "phase": "candidate", "tool": "search_script_library", "run": 1,
         "repeat": 2, "doneRuns": 3, "totalRuns": 4}, "tools")
    check("⑥-11 工具进度说「摘掉 x」而不是「温度 None」",
          "摘掉 search_script_library" in run_text and "None" not in run_text, run_text)
    pending_text = tune_runtime._stage_text(
        {"type": "candidate", "record": {"tool": "x", "mean": 80.0, "unknownTool": 0}}, "tools")
    check("⑥-12 判据未出那次印均值/unknownTool，不印 None",
          "均值 80.0" in pending_text and "None" not in pending_text, pending_text)


# ---------------------------------------------------------------- ⑦ HTTP


def case_http() -> None:
    print("\n⑦ HTTP 入口：错误路径不启动模型 / 正常路径只交 target 不交候选数 / 回执写明不落库")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    path = f"/api/v1/evaluation/tune-tools/{CASE['id']}"

    check("⑦-1 未知 case ⇒ 404",
          client.post("/api/v1/evaluation/tune-tools/nope-999", json={}).status_code == 404)
    resp = client.post(path, json={"repeat": 1})
    check("⑦-2 repeat=1 ⇒ 400（校验在入队前，不会白跑一轮）", resp.status_code == 400,
          resp.text[:200])

    # 正常路径：把 submit 打桩，只看它收到的参数（否则会真起后台线程拉模型）
    captured: dict = {}
    original_submit = tune_runtime.submit

    def fake_submit(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {"taskId": "tune-stub", "target": kwargs.get("target"), "status": "queued"}

    tune_runtime.submit = fake_submit  # type: ignore[assignment]
    try:
        # 非整数**被忽略**（与 repeat 那格同一口径）：不进 options，回落到默认值 ——
        # 若当成 0 就是一次静默的空调优（maxCandidates=0 ⇒ 候选为空 ⇒ 什么都没调），
        # 比报错更难查；若整条拒掉又会与兄弟端点（温度/技能）不一致。
        default_caps = to.normalize_tool_options({})["maxCandidates"]
        resp = client.post(path, json={"max_candidates": "两个"})
        check("⑦-3 非整数 max_candidates 被忽略而不是当 0", resp.status_code == 200, resp.text[:200])
        check("⑦-4 被忽略的那格回落到默认候选数（不是 0、也不是字面量）",
              (captured.get("options") or {}).get("maxCandidates") == default_caps,
              f"{captured.get('options')} 默认={default_caps}")
        captured.clear()

        resp = client.post(path, json={"repeat": 2, "max_candidates": 2})
        data = resp.json().get("data") or {}
        check("⑦-5 入队成功", resp.status_code == 200, resp.text[:200])
        check("⑦-6 submit 收到 target=tools", captured.get("target") == "tools", str(captured))
        check("⑦-7 不传 candidate_count（候选要等基线跑完才定，先编一个数是假的）",
              "candidate_count" not in captured, str(captured))
        check("⑦-8 options.autoPersist 恒 False",
              captured.get("options", {}).get("autoPersist") is False, str(captured.get("options")))
        preview = data.get("preview") or {}
        check("⑦-9 回执 preview.persist=False", preview.get("persist") is False, str(preview))
        check("⑦-10 回执写明为什么不落库（persistSkip 非空且点名 agent_configs）",
              "agent_configs" in str(preview.get("persistSkip") or ""), str(preview.get("persistSkip")))
        check("⑦-11 回执标明候选源是轨迹（candidateSource=trace）",
              preview.get("candidateSource") == "trace", str(preview))
        check("⑦-12 回执没有 preview.candidates（这一格的候选此刻还不存在）",
              "candidates" not in preview, str(preview))
    finally:
        tune_runtime.submit = original_submit  # type: ignore[assignment]
        tune_runtime.reset()


def main() -> int:
    print("工具集调优（缺口 4 第三格）守卫测试")
    print(f"临时数据根：{_DATA_ROOT}")
    case_step_trace()
    case_candidates()
    case_direction()
    case_candidate_errors()
    case_not_persisted()
    case_override_shape()
    case_wiring()
    case_http()
    tune_runtime.reset()

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
