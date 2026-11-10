"""技能绑定调优（缺口 4 第二格）的守卫测试。

被测：``app/agent/evaluation/skill_optimizer.py`` + 它接线的四处
（``evaluator.evaluate_case(skills=…)`` / ``runtime`` 的按次覆盖 / ``tune_runtime`` 的双目标 /
``routers/evaluation.py`` 的 ``POST /tune-skills/{case_id}``）。

这一格与温度调优**共用全部机制，只差一个判据方向**，所以测试的重点不是「能不能跑」，
而是那些「看起来一样、实际不一样」的地方：

* 判据方向相反：温度是「涨过抖动带才采纳」，技能是「**跌幅不超过抖动带就采纳（摘掉）**」；
* 覆盖体不是温度而是一段 JSON —— 形状错（纯字符串数组）会被 ``parse_skills_config``
  **静默跳过**，优化器以为摘掉了、运行时其实还在用默认绑定，两边都不报错；
* 候选是**一遍一个**（摘掉一个之后基线就变了），不是一次删到底；
* 删除默认**不落库**（``autoPersist`` 默认 False）——温度最坏是更死板，删 skill 最坏是
  该知道的规范不知道了。

跑法：``.venv/Scripts/python.exe tests/skill_optimizer_test.py``（cp936 控制台下也直接可用）。
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
_DATA_ROOT = Path(tempfile.mkdtemp(prefix="skill_optimizer_test_"))
os.environ["DATA_ROOT"] = str(_DATA_ROOT)
os.environ.setdefault("SKILLS_DIR", str(_DATA_ROOT / "skills"))

from sqlalchemy import text  # noqa: E402

from app.core import db  # noqa: E402
from app.agent import creator  # noqa: E402
from app.agent.evaluation import param_optimizer as po  # noqa: E402
from app.agent.evaluation import skill_optimizer as so  # noqa: E402
from app.agent.evaluation import tune_runtime  # noqa: E402
from app.agent.skills import parse_skills_config  # noqa: E402
from app.services import optimization_log  # noqa: E402

AGENT = "script_rewriter"
CASE = {
    "id": "script-rewriter-001",
    "kind": "script_rewriter",
    "input": {"chapter": "第一章", "idea": "把开场改得更狠"},
    "criteria": ["保留人物关系", "开场三句内进冲突"],
}
PROMPT = "你是剧本改写助手。只输出 Markdown 正文，不要解释。"

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


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- 夹具


def _reset() -> None:
    with db.engine.begin() as conn:
        conn.execute(text("DELETE FROM agent_optimizations"))
        conn.execute(text("DELETE FROM agent_configs"))


def _skills_payload(ids: list[str]) -> str:
    """把绑定写成 ``agent_configs.skills`` 那种对象数组（**真源形状**）。"""
    return json.dumps(
        [{"id": i, "enabled": True, "priority": n + 1} for n, i in enumerate(ids)],
        ensure_ascii=False,
    )


def _seed_config(*, skills: list[str] | None, temperature: float = 0.7) -> None:
    now = "2026-11-08T00:00:00+00:00"
    with db.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO agent_configs "
                "(agent_type, name, description, system_prompt, temperature, max_tokens, "
                " skills, is_active, created_at, updated_at) "
                "VALUES (:t, :n, :d, :p, :temp, 4096, :s, 1, :now, :now)"
            ),
            {"t": AGENT, "n": "剧本改写助手", "d": "", "p": PROMPT, "temp": temperature,
             "s": _skills_payload(skills or []), "now": now},
        )


def _db_skills() -> list[str]:
    with db.engine.connect() as conn:
        raw = conn.execute(
            text("SELECT skills FROM agent_configs WHERE agent_type = :t"), {"t": AGENT},
        ).scalar()
    return [item.id for item in parse_skills_config(raw)]


def _config_rows() -> int:
    with db.engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM agent_configs")).scalar() or 0


def _opt_rows() -> list[dict]:
    with db.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT target, reason, version, persisted, score_before, score_after "
                "FROM agent_optimizations ORDER BY id"
            )
        ).mappings().all()
    return [dict(r) for r in rows]


def make_eval(script: dict[tuple[str, ...] | None, list[float]], models=None):
    """假评测器：按「这一跑实际注入的绑定」给分。

    ``script`` 的键是**注入的 skill id 元组**（基线是 ``None``），值是个分数列表（按次取、循环）。
    这是一个刻意的形状探针：它按 ``b["id"]`` 读覆盖体，覆盖体若被写成纯字符串数组就会当场
    ``TypeError``（真实后果却是**静默**回退默认绑定，见 ①）。
    """
    used: dict[object, int] = {}
    calls: list[dict] = []

    async def run(conn, case_def, instructions, settings=None, *, skills=None):
        ids = None if skills is None else tuple(b["id"] for b in json.loads(skills))
        calls.append({"skills": ids, "settings": settings, "prompt": instructions})
        if ids not in script:
            raise AssertionError(f"没给绑定 {ids} 安排分数")
        idx = used.get(ids, 0)
        used[ids] = idx + 1
        totals = script[ids]
        return {"caseId": case_def.get("id"), "total": totals[idx % len(totals)],
                "runtimeModel": (models or {}).get(ids, "ollama/qwen3:14b")}

    run.calls = calls  # type: ignore[attr-defined]
    return run


def _optimize(options: dict, *, evaluate, agent_type: str = AGENT, case_def=None, **kwargs):
    with db.engine.begin() as conn:
        return _run(so.optimize_agent_skills(
            conn, agent_type, case_def or CASE, options, evaluate=evaluate, **kwargs,
        ))


# ---------------------------------------------------------------- ① 形状


def case_binding_shape() -> None:
    print("\n① 覆盖体与落库体都必须是对象数组（纯字符串数组会被静默跳过）")
    # 覆盖体：优化器实际发给 evaluator 的那段 JSON
    raw = so._skills_json(["a", "b"])
    parsed = parse_skills_config(raw)
    check("①-1 覆盖体可被 parse_skills_config 读回", [p.id for p in parsed] == ["a", "b"],
          f"原文 {raw}")
    check("①-2 覆盖体每项是对象且带 enabled/priority",
          all(isinstance(json.loads(raw)[i], dict) for i in (0, 1))
          and all(p.enabled for p in parsed),
          raw)

    # 落库体：走 creator 的归一（与 /agents/{type} 那个 handler 同一份实现）
    _reset()
    _seed_config(skills=["x"])
    written = creator.persist_agent_skills(AGENT, ["a", "b"])
    with db.engine.connect() as conn:
        stored = conn.execute(
            text("SELECT skills FROM agent_configs WHERE agent_type = :t"), {"t": AGENT},
        ).scalar()
    check("①-3 persist_agent_skills 报写了 1 行", written == 1, f"written={written}")
    check("①-4 落库体可被 parse_skills_config 读回（不是纯字符串数组）",
          [p.id for p in parse_skills_config(stored)] == ["a", "b"], str(stored))
    check("①-5 落库体首项是 dict", isinstance(json.loads(stored)[0], dict), str(stored))
    check("①-6 落库后 _db_skills 读到新绑定", _db_skills() == ["a", "b"])


# ---------------------------------------------------------------- ② 候选


def case_candidates() -> None:
    print("\n② 候选：一遍一个、按可疑度排、剩一个时不摘、从不摘成空")
    cands = so.skill_candidates(["a", "b", "c"], agent_type=AGENT, max_candidates=2)
    check("②-1 默认取 max_candidates 个", len(cands) == 2, str(cands))
    check("②-2 同分时绑定靠后的先摘", cands == ["c", "b"], str(cands))
    check("②-3 剩一个 ⇒ 没有候选（不许摘成空）",
          so.skill_candidates(["a"], agent_type=AGENT) == [])
    check("②-4 空绑定 ⇒ 没有候选",
          so.skill_candidates([], agent_type=AGENT) == [])
    check("②-5 max_candidates=0 ⇒ 没有候选",
          so.skill_candidates(["a", "b"], agent_type=AGENT, max_candidates=0) == [])

    # penalty 高（可疑）的排前面，即使它在绑定里靠前
    original = so.skill_usage.penalty

    def fake_penalty(agent_type, skill_id):  # noqa: ANN001
        return (9.0, "失败率高") if skill_id == "a" else (0.0, "")

    so.skill_usage.penalty = fake_penalty
    try:
        cands = so.skill_candidates(["a", "b", "c"], agent_type=AGENT, max_candidates=1)
        check("②-6 可疑度高的先摘（不受绑定先后影响）", cands == ["a"], str(cands))
    finally:
        so.skill_usage.penalty = original


# ---------------------------------------------------------------- ③ 判据方向


def case_direction() -> None:
    print("\n③ 判据方向：跌幅不超过抖动带 ⇒ 采纳摘除；跌幅明显 ⇒ 保留")

    # A：摘掉不损失 ⇒ pruned
    _reset()
    _seed_config(skills=["a", "b", "c"])
    evaluate = make_eval({None: [80.0], ("a", "b"): [80.0, 80.0]})
    receipt = _optimize({"repeat": 2, "maxCandidates": 1}, evaluate=evaluate)
    accepted = receipt.get("accepted") or {}
    check("③-1 分数没掉 ⇒ 采纳", accepted.get("accepted") is True, str(receipt.get("accepted")))
    check("③-2 理由码是 pruned", accepted.get("reason") == "pruned", str(accepted.get("reason")))
    check("③-3 摘的是候选里那个 skill", accepted.get("skill") == "c", str(accepted.get("skill")))
    check("③-4 采纳记录带 remaining（真摘掉了）",
          accepted.get("remaining") == ["a", "b"], str(accepted.get("remaining")))

    # B：摘掉明显掉分 ⇒ skill_needed、不采纳
    _reset()
    _seed_config(skills=["a", "b", "c"])
    evaluate = make_eval({None: [80.0], ("a", "b"): [40.0, 40.0]})
    receipt = _optimize({"repeat": 2, "maxCandidates": 1}, evaluate=evaluate)
    check("③-5 分数掉出抖动带 ⇒ 不采纳", receipt.get("accepted") is None,
          str(receipt.get("accepted")))
    record = (receipt.get("candidates") or [{}])[0]
    check("③-6 理由码是 skill_needed", record.get("reason") == "skill_needed",
          str(record.get("reason")))
    check("③-7 不采纳时库里绑定原样", _db_skills() == ["a", "b", "c"], str(_db_skills()))

    # C：跌幅正好落在抖动带内 ⇒ 仍算「没有证据说明它有用」
    _reset()
    _seed_config(skills=["a", "b", "c"])
    evaluate = make_eval({None: [78.0, 82.0], ("a", "b"): [76.0, 76.0]})
    receipt = _optimize({"repeat": 2, "maxCandidates": 1}, evaluate=evaluate)
    accepted = receipt.get("accepted") or {}
    check("③-8 抖动带 4 时跌 4 分仍采纳（边界取 >=）",
          accepted.get("accepted") is True and accepted.get("delta") == -4.0,
          f"accepted={accepted.get('accepted')} delta={accepted.get('delta')} "
          f"jump={(receipt.get('baseline') or {}).get('jump')}")

    # D：只剩一个绑定时不许摘（否则 agent 变裸模型）
    _reset()
    _seed_config(skills=["a"])
    evaluate = make_eval({None: [70.0, 70.0]})
    receipt = _optimize({"repeat": 2}, evaluate=evaluate)
    check("③-9 剩一个 ⇒ 不收工成空绑定，且如实写理由",
          receipt.get("accepted") is None and "只绑定了 1 个" in str(receipt.get("reason")),
          str(receipt.get("reason")))
    check("③-10 剩一个时只跑了基线（没有候选评测）", len(evaluate.calls) == 2,
          str(len(evaluate.calls)))


# ---------------------------------------------------------------- ④⑤ 落库


def case_no_persist_by_default() -> None:
    print("\n④ 默认不落库（删 skill 需人点头），但结论照出")
    _reset()
    _seed_config(skills=["a", "b", "c"])
    evaluate = make_eval({None: [80.0, 80.0], ("a", "b"): [80.0, 80.0]})
    receipt = _optimize({"repeat": 2, "maxCandidates": 1}, evaluate=evaluate)
    accepted = receipt.get("accepted") or {}
    check("④-1 采纳了", accepted.get("accepted") is True, str(accepted))
    check("④-2 但 persisted=False", accepted.get("persisted") is False, str(accepted.get("persisted")))
    check("④-3 给了不落库的理由", bool(accepted.get("persistSkip")), str(accepted.get("persistSkip")))
    check("④-4 库里绑定原样", _db_skills() == ["a", "b", "c"], str(_db_skills()))
    rows = _opt_rows()
    check("④-5 优化史只有 1 行（被采纳那条）", len(rows) == 1, str(rows))
    check("④-6 该行 target=skills、persisted=0",
          bool(rows) and rows[0]["target"] == "skills" and rows[0]["persisted"] == 0, str(rows))


def case_persist() -> None:
    print("\n⑤ autoPersist=true ⇒ 摘除真的落进 agent_configs")
    _reset()
    _seed_config(skills=["a", "b", "c"])
    evaluate = make_eval({None: [80.0, 80.0], ("a", "b"): [80.0, 80.0]})
    receipt = _optimize({"repeat": 2, "maxCandidates": 1, "autoPersist": True}, evaluate=evaluate)
    accepted = receipt.get("accepted") or {}
    check("⑤-1 persisted=True", accepted.get("persisted") is True, str(accepted))
    check("⑤-2 库里只剩 a、b", _db_skills() == ["a", "b"], str(_db_skills()))
    rows = _opt_rows()
    check("⑤-3 优化史那行标了 persisted=1", bool(rows) and rows[-1]["persisted"] == 1, str(rows))
    check("⑤-4 版本号从 1 起", bool(rows) and rows[-1]["version"] == 1, str(rows))
    with db.engine.connect() as conn:
        stored = conn.execute(
            text("SELECT skills FROM agent_configs WHERE agent_type = :t"), {"t": AGENT},
        ).scalar()
    check("⑤-5 落库体仍是对象数组且 priority 保序",
          [p.priority for p in parse_skills_config(stored)] == [1, 2], str(stored))


def case_no_config_row() -> None:
    print("\n⑥ 没有配置行 ⇒ 不新建、如实报「采纳了但没落库」")
    _reset()
    original = so.resolve_default_skills
    so.resolve_default_skills = lambda agent_type: ["d1", "d2"]  # noqa: ARG005
    try:
        evaluate = make_eval({None: [60.0, 60.0], ("d1",): [60.0, 60.0]})
        receipt = _optimize({"repeat": 2, "maxCandidates": 1, "autoPersist": True},
                            evaluate=evaluate)
        accepted = receipt.get("accepted") or {}
        check("⑥-1 绑定来源是默认绑定",
              (receipt.get("current") or {}).get("source") == "default",
              str(receipt.get("current")))
        check("⑥-2 采纳了", accepted.get("accepted") is True, str(accepted))
        check("⑥-3 但 persisted=False", accepted.get("persisted") is False, str(accepted))
        check("⑥-4 理由点名「没有配置行」",
              "没有该 agent 的配置行" in str(accepted.get("persistSkip")),
              str(accepted.get("persistSkip")))
        check("⑥-5 没凭空造配置行", _config_rows() == 0, f"rows={_config_rows()}")
    finally:
        so.resolve_default_skills = original


# ---------------------------------------------------------------- ⑦ 一遍一个


def case_one_at_a_time() -> None:
    print("\n⑦ 一次只摘一个：摘掉后基线就变了，不许在同一轮里接着摘")
    _reset()
    _seed_config(skills=["a", "b", "c"])
    # 两个候选都能「摘掉不掉分」，但只应采纳第一个（c），评测次数 = 2 次基线 + 2 次候选
    evaluate = make_eval({None: [80.0], ("a", "b"): [80.0], ("a", "c"): [80.0]})
    receipt = _optimize({"repeat": 2, "maxCandidates": 2, "autoPersist": True}, evaluate=evaluate)
    cand_records = receipt.get("candidates") or []
    check("⑦-1 录取记录只有 1 条（采纳即停）", len(cand_records) == 1, str(cand_records))
    check("⑦-2 那次评测总次数是 4（不是 6）", len(evaluate.calls) == 4, str(len(evaluate.calls)))
    check("⑦-3 落库只剩 a、b", _db_skills() == ["a", "b"], str(_db_skills()))


# ---------------------------------------------------------------- ⑧ 接线


def case_wiring() -> None:
    print("\n⑧ 接线：透传 / 覆盖不污染使用统计 / 后台任务双目标 / 提交用 begin()")

    # ⑧-1~3：优化器确实把「摘掉后的绑定」送进评测，基线那一跑不送覆盖
    _reset()
    _seed_config(skills=["a", "b", "c"])
    evaluate = make_eval({None: [80.0, 80.0], ("a", "b"): [80.0, 80.0]})
    _optimize({"repeat": 2, "maxCandidates": 1}, evaluate=evaluate)
    check("⑧-1 基线那一跑不带覆盖", evaluate.calls[0]["skills"] is None, str(evaluate.calls[0]))
    check("⑧-2 候选那一跑注入的正是「摘掉 c 之后」的绑定",
          evaluate.calls[-1]["skills"] == ("a", "b"), str(evaluate.calls[-1]))
    check("⑧-3 候选那一跑也没改温度（settings 不传）",
          evaluate.calls[-1]["settings"] is None, str(evaluate.calls[-1]))

    # ⑧-4/5：evaluator 的 skills 形参是 kwonly，且只在非 None 时才进 run_options
    src = (_ROOT / "app/agent/evaluation/evaluator.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "evaluate_case"
    )
    kwonly = [a.arg for a in fn.args.kwonlyargs]
    check("⑧-4 evaluate_case 有 kwonly 形参 skills", "skills" in kwonly, str(kwonly))
    check("⑧-5 evaluator 用 is not None 判（'[]' 与不传是两回事）",
          "if skills is not None:" in src and 'run_options["skillsRaw"] = skills' in src)

    # ⑧-6/7：runtime 按次覆盖，且覆盖时**不写** skill_usage
    rt = (_ROOT / "app/agent/runtime.py").read_text(encoding="utf-8")
    check("⑧-6 runtime 认 skillsRaw 覆盖", 'options.get("skillsRaw")' in rt)
    seg_start = rt.index("def _record_skills(")
    seg = rt[seg_start: seg_start + 400]
    check("⑧-7 覆盖时不写使用统计（防假设性绑定污染裁剪顺序）",
          "if skills_overridden:" in seg and seg.index("return") < seg.index("record_skill_usage"),
          seg[:200].replace("\n", " | "))

    # ⑧-8/9：后台任务表认双目标，且**提交必须 begin()**（connect() 会回滚写入）
    tr = (_ROOT / "app/agent/evaluation/tune_runtime.py").read_text(encoding="utf-8")
    check("⑧-8 tune_runtime 注册了 skills 目标与专用 runner",
          '"skills"' in tr and "async def _skills_runner(" in tr and "def _runner_for(" in tr)
    check("⑧-9 优化任务用 engine.begin()（connect() 的写入在 close 时回滚）",
          "db_module.engine.begin()" in tr and "db_module.engine.connect()" not in tr,
          "见 tune_runtime._execute 注释里的实测")

    # ⑧-10/11：路由入口存在且入队前只校验一处
    rv = (_ROOT / "app/routers/evaluation.py").read_text(encoding="utf-8")
    check("⑧-10 路由有 POST /tune-skills/{case_id}",
          '@router.post("/tune-skills/{case_id}")' in rv)
    check("⑧-11 路由用 normalize_skill_options 校验（复用温度那格的实现）",
          "options = normalize_skill_options(options)" in rv)


# ---------------------------------------------------------------- ⑨ 取消


def case_cancel() -> None:
    print("\n⑨ 取消：两次评测之间生效，不落库、不留半条记录")
    _reset()
    _seed_config(skills=["a", "b", "c"])
    evaluate = make_eval({None: [80.0], ("a", "b"): [80.0]})
    try:
        _optimize({"repeat": 2, "maxCandidates": 1}, evaluate=evaluate,
                  should_stop=lambda: True)
        check("⑨-1 取消时抛 TuneCancelled", False, "没有抛")
    except po.TuneCancelled:
        check("⑨-1 取消时抛 TuneCancelled", True)
    check("⑨-2 取消后库里绑定原样", _db_skills() == ["a", "b", "c"], str(_db_skills()))
    check("⑨-3 取消后没写优化史", _opt_rows() == [], str(_opt_rows()))


# ---------------------------------------------------------------- ⑩ 版本与出口


def case_version_and_surface() -> None:
    print("\n⑩ 版本跨目标接续 + /optimizations 能读出 target=skills")
    _reset()
    _seed_config(skills=["a", "b", "c"])
    # 先占一个「提示词版本 3」，技能行必须接在 4 而不是从头数
    with db.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO agent_optimizations "
                "(agent_type, case_id, target, version, prompt_sha, reason, score_before, "
                " score_after, delta, persisted, created_at) "
                "VALUES (:t, :c, 'system_prompt', 3, 'abc', 'adopted', 70, 80, 10, 1, :now)"
            ),
            {"t": AGENT, "c": CASE["id"], "now": "2026-11-08T00:00:00+00:00"},
        )
    evaluate = make_eval({None: [80.0, 80.0], ("a", "b"): [80.0, 80.0]})
    _optimize({"repeat": 2, "maxCandidates": 1}, evaluate=evaluate)
    rows = _opt_rows()
    skills_row = next((r for r in rows if r["target"] == "skills"), None)
    check("⑩-1 技能行版本接在 4", skills_row and skills_row["version"] == 4, str(rows))
    check("⑩-2 理由人话在册",
          optimization_log.REASONS["pruned"] and optimization_log.REASONS["skill_needed"])
    check("⑩-3 TARGETS 里有 skills 的人话名",
          optimization_log.TARGETS.get("skills") == "技能绑定", str(optimization_log.TARGETS))

    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    # 优化史**只读不写**，出口在 knowledge-loop（缺口 3 就落在那），不在 evaluation 下
    resp = client.get("/api/v1/knowledge-loop/optimizations",
                      params={"agentType": AGENT, "caseId": CASE["id"]})
    data = resp.json().get("data") or {}
    check("⑩-4 /knowledge-loop/optimizations 200", resp.status_code == 200, resp.text[:200])
    targets = [i.get("target") for i in (data.get("items") or [])]
    check("⑩-5 出口能读出 target=skills", "skills" in targets, str(targets))
    check("⑩-6 汇总按目标分桶（byTarget 里有 skills）",
          (data.get("summary") or {}).get("byTarget", {}).get("skills") == 1,
          str((data.get("summary") or {}).get("byTarget")))
    check("⑩-7 优化史行带人话理由（pruned 有中文名）",
          any(i.get("reasonLabel") == optimization_log.REASONS["pruned"]
              for i in (data.get("items") or [])),
          str([i.get("reasonLabel") for i in (data.get("items") or [])]))


# ---------------------------------------------------------------- ⑪ HTTP


def case_http() -> None:
    print("\n⑪ HTTP 入口：错误路径不启动模型，正常路径把 target 与候选数交给队列")
    from fastapi.testclient import TestClient
    from app.main import app
    from app.routers import evaluation as ev_router

    client = TestClient(app)
    path = f"/api/v1/evaluation/tune-skills/{CASE['id']}"

    check("⑪-1 未知 case ⇒ 404",
          client.post("/api/v1/evaluation/tune-skills/nope-999", json={}).status_code == 404)
    resp = client.post(path, json={"repeat": 1})
    check("⑪-2 repeat=1 ⇒ 400（校验在入队前，不会白跑一轮）", resp.status_code == 400,
          resp.text[:200])

    # 正常路径：把 submit 打桩，只看它收到的参数（否则会真起后台线程拉模型）
    captured: dict = {}
    original_submit = tune_runtime.submit

    def fake_submit(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {"taskId": "tune-stub", "target": kwargs.get("target"), "status": "queued",
                "totalRuns": 0}

    tune_runtime.submit = fake_submit  # type: ignore[assignment]
    _reset()
    _seed_config(skills=["a", "b", "c"])
    try:
        resp = client.post(path, json={"repeat": 2, "max_candidates": 2})
        data = resp.json().get("data") or {}
        check("⑪-3 入队成功", resp.status_code == 200, resp.text[:200])
        check("⑪-4 submit 收到 target=skills", captured.get("target") == "skills", str(captured))
        check("⑪-5 submit 收到 candidate_count=2（按真绑定算的）",
              captured.get("candidate_count") == 2, str(captured.get("candidate_count")))
        check("⑪-6 autoPersist 默认 False（删 skill 需人点头）",
              captured.get("options", {}).get("autoPersist") is False, str(captured.get("options")))
        preview = data.get("preview") or {}
        check("⑪-7 回执带 preview.current", preview.get("current") == ["a", "b", "c"],
              str(preview))
        check("⑪-8 回执带 preview.candidates", preview.get("candidates") == ["c", "b"],
              str(preview))

        client.post(path, json={"repeat": 2, "max_candidates": 2, "auto_persist": True})
        check("⑪-9 auto_persist=true 时才为 True",
              captured.get("options", {}).get("autoPersist") is True, str(captured.get("options")))
    finally:
        tune_runtime.submit = original_submit  # type: ignore[assignment]
        ev_router  # noqa: B018 —— 保持 import 被使用（路由模块就是被测对象）
        tune_runtime.reset()


# ---------------------------------------------------------------- ⑫ 双目标语义


def case_runtime_targets() -> None:
    print("\n⑫ tune_runtime 的双目标语义：去重键带 target、进度文案分流、非法目标当场拒")

    async def stub(conn, job):  # noqa: ANN001, ARG001
        return {"ok": True, "target": job.get("target")}

    tune_runtime.reset()
    first = tune_runtime.submit(
        case_id=CASE["id"], case_def=CASE, agent_type=AGENT,
        options={"repeat": 2, "maxCandidates": 2}, target="params",
        candidate_count=None, runner=stub,
    )
    tune_runtime.wait(str(first["taskId"]), timeout=10)
    check("⑫-1 params 任务快照带 target", first.get("target") == "params", str(first))
    check("⑫-2 没给候选数时按 maxCandidates 估 totalRuns",
          first.get("totalRuns") == 2 * 3, str(first.get("totalRuns")))

    second = tune_runtime.submit(
        case_id=CASE["id"], case_def=CASE, agent_type=AGENT,
        options={"repeat": 2, "maxCandidates": 2}, target="skills",
        candidate_count=1, runner=stub,
    )
    check("⑫-3 同一 case 换目标不判重复（去重键带 target）",
          second.get("deduped") is not True and second["taskId"] != first["taskId"], str(second))
    check("⑫-4 candidate_count 参与 totalRuns 预估",
          second.get("totalRuns") == 2 * 2, str(second.get("totalRuns")))

    third = tune_runtime.submit(
        case_id=CASE["id"], case_def=CASE, agent_type=AGENT,
        options={"repeat": 2, "maxCandidates": 2}, target="skills",
        candidate_count=1, runner=stub,
    )
    check("⑫-5 同 case 同目标同状态 ⇒ 判重复并回同一个 taskId",
          third.get("deduped") is True and third["taskId"] == second["taskId"], str(third))
    tune_runtime.wait(str(second["taskId"]), timeout=10)

    try:
        tune_runtime.submit(
            case_id=CASE["id"], case_def=CASE, agent_type=AGENT, options={}, target="lora",
            runner=stub,
        )
        check("⑫-6 未知目标 ⇒ ValueError", False, "没有抛")
    except ValueError:
        check("⑫-6 未知目标 ⇒ ValueError", True)

    skills_text = tune_runtime._stage_text(
        {"type": "run", "phase": "candidate", "skill": "c", "run": 1, "repeat": 2,
         "doneRuns": 3, "totalRuns": 4}, "skills")
    params_text = tune_runtime._stage_text(
        {"type": "run", "phase": "candidate", "temperature": 0.5, "run": 1, "repeat": 2,
         "doneRuns": 3, "totalRuns": 4}, "params")
    check("⑫-7 技能进度说「摘掉 c」而不是「温度 None」",
          "摘掉 c" in skills_text and "None" not in skills_text, skills_text)
    check("⑫-8 温度进度照旧说温度", "温度 0.5" in params_text, params_text)
    tune_runtime.reset()


def main() -> int:
    print("技能绑定调优（缺口 4 第二格）守卫测试")
    print(f"临时数据根：{_DATA_ROOT}")

    case_binding_shape()
    case_candidates()
    case_direction()
    case_no_persist_by_default()
    case_persist()
    case_no_config_row()
    case_one_at_a_time()
    case_wiring()
    case_cancel()
    case_version_and_surface()
    case_http()
    case_runtime_targets()

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
