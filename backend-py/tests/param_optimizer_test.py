"""参数（温度）调优守卫（`docs/knowledge-loop.md` §1 缺口 4 第一格）。

为什么要有它
============
提示词优化是「一次一版、单次评测定输赢」——那个场景下噪声被当成提升还算能忍（人还要再审一遍）。
调参不行：同一档位自己重跑两次就能差好几分，把这点抖动当提升写进 ``agent_configs``，等于给
"这次改动提升了多少"这张表灌脏数，而且**没有第二个人会去复核一个温度**。

本套盯的是三条口径与它们的落点，不是"函数没抛异常"：

① 候选只往**更低**温度扫（网格过滤、最近优先、0.0 时没有候选是事实不是错误）；
② ``repeat < 2`` **当场拒**（单次评测分不清"换温度了"与"这次恰好抖高了"）；
③ 判采纳用的是**均值涨幅 > 基线抖动带**，涨幅没超过抖动带记 ``below_noise`` 而不是 ``accepted``；
④ 采纳才落库、落库只动 ``temperature`` 一列；**没有配置行就不写**（照实 ``persisted=False``，
   绝不为了"看起来生效了"插一行只带温度的配置，那会把 ``system_prompt`` 留空）；
⑤ 版本号跨目标取 max+1（``version`` 是 (agent, case) 上的**轮次序号**，三格共用一个递增器；
   防串标**不靠它** —— 提示词那格的号来自 history、不经过它，撞号本来就躲不开，靠
   ``optimization_log.mark_persisted`` 的 WHERE 带 ``target`` 兜住）；
⑥ 同提示词：一轮内所有行的 ``prompt_sha`` 必须一致（不一样就说明"固定提示词"这个前提破了）；
⑦ 接线：``runtime`` 的覆盖按 nullish（``temperature=0.0`` 不许被默认值顶掉）、
   ``evaluator`` 把 settings 传给 runtime、路由 ``POST /evaluation/tune/{case}`` 存在且
   ``repeat=1`` 判 400（**不是**悄悄抬到 2）、未知 case 判 404；
⑧ ``/optimizations`` 端点能读出 ``target=params`` 且人话标签在。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/param_optimizer_test.py
"""
from __future__ import annotations

import ast
import asyncio
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"
sys.path.insert(0, str(BP))
os.environ.setdefault("PROXY_TO_NODE", "0")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

TMP = Path(tempfile.mkdtemp(prefix="param-optimizer-"))
os.environ["DATA_ROOT"] = TMP.as_posix()

from sqlalchemy import text  # noqa: E402

from app.core import db  # noqa: E402
from app.services import optimization_log  # noqa: E402
from app.agent import creator  # noqa: E402
from app.agent.evaluation import param_optimizer as po  # noqa: E402
from app.agent.runtime import DEFAULT_TEMPERATURE, merge_model_settings  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []

AGENT = "script_rewriter"
CASE = {"id": "rewrite-basic", "kind": "script_rewrite"}
#: HTTP 那两组必须用**真的**基准 case id（路由先用 ``load_case_by_id`` 查，假 id 一律 404）。
HTTP_CASE = "script-rewriter-001"


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _reset() -> None:
    with db.engine.begin() as conn:
        conn.execute(text("DELETE FROM agent_optimizations"))
        conn.execute(text("DELETE FROM agent_configs"))


def _seed_config(*, agent_type: str = AGENT, temperature: float | None = 0.7,
                 prompt: str = "你是剧本改写助手……") -> None:
    now = "2026-11-07T00:00:00.000Z"
    with db.engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO agent_configs (agent_type, name, description, system_prompt, temperature,"
            " max_tokens, is_active, created_at, updated_at)"
            " VALUES (:a, '脚本改写', 'desc', :p, :t, 4096, 1, :ts, :ts)"
        ), {"a": agent_type, "p": prompt, "t": temperature, "ts": now})


def make_eval(script: dict[object, list[float]], models: dict[object, str] | None = None):
    """假评测器：``script`` 的键是温度（``None``＝基线），值是这一档依次吐出的总分。

    只做一件事：把"同一档位重跑会抖"这件真实特性如实演出来（按出现顺序消费列表）。
    """
    used: dict[object, int] = {key: 0 for key in script}
    calls: list[object] = []

    async def run(conn, case_def, instructions, settings=None):  # noqa: ANN001
        key = (settings or {}).get("temperature")
        calls.append(key)
        totals = script.get(key)
        if totals is None:
            raise AssertionError(f"没给温度 {key} 安排分数")
        index = used[key]
        used[key] += 1
        model = (models or {}).get(key, "ollama/qwen3:14b")
        return {"caseId": case_def.get("id"), "total": totals[index % len(totals)],
                "runtimeModel": model}

    run.calls = calls  # type: ignore[attr-defined]
    return run


def _run(coro):
    return asyncio.run(coro)


def _optimize(**options):
    def factory(conn, case_def, instructions, settings=None):  # noqa: ANN001
        return options["evaluate"](conn, case_def, instructions, settings)

    opts = {k: v for k, v in options.items() if k != "evaluate"}
    if "agent_type" in options:
        opts.pop("agent_type")
        agent = options["agent_type"]
    else:
        agent = AGENT
    with db.engine.begin() as conn:
        return _run(po.optimize_agent_temperature(
            conn, agent, options.get("case", CASE), opts,
            evaluate=factory,
        ))


# ===================================================================== ①
def case_candidates() -> None:
    check("① 当前 0.7 ⇒ 候选是更低的最近两档 [0.5, 0.3] ✓",
          po.temperature_candidates(0.7) == [0.5, 0.3], po.temperature_candidates(0.7))
    check("① 候选里**没有当前值、也没有更高值** ✓（往高温扫不是调参该管的事 ✗）",
          all(t < 0.7 for t in po.temperature_candidates(0.7)), po.temperature_candidates(0.7))
    check("① 当前 0.2 ⇒ [0.1, 0.0]（含 0.0 贪心解码）✓",
          po.temperature_candidates(0.2) == [0.1, 0.0], po.temperature_candidates(0.2))
    check("① 当前已是 0.0 ⇒ 空列表（没有更低的档，这是事实不是错误）✓",
          po.temperature_candidates(0.0) == [], po.temperature_candidates(0.0))
    check("① max_candidates=0 ⇒ 空列表 ✓（只量抖动带、不下候选的用法）",
          po.temperature_candidates(0.7, max_candidates=0) == [], "非空")
    check("① 非法输入（None/字符串）⇒ 空列表，不抛 ✓",
          po.temperature_candidates(None) == [] and po.temperature_candidates("热") == [],
          "抛了")


# ===================================================================== ②
def case_runtime_override() -> None:
    base = {"temperature": 0.7, "maxOutputTokens": 100}
    check("② runtime 覆盖：temperature=0.0 必须保留 0.0 ✓（写 `or` 会被默认值顶掉 ⇒ 试 0.0 实际试 0.7 ✗）",
          merge_model_settings(base, {"temperature": 0.0})["temperature"] == 0.0,
          merge_model_settings(base, {"temperature": 0.0}))
    check("② runtime 覆盖：None 视同没给 ✓",
          merge_model_settings(base, {"temperature": None}) == base,
          merge_model_settings(base, {"temperature": None}))
    check("② runtime 覆盖：maxOutputTokens 也能覆盖 ✓",
          merge_model_settings(base, {"maxOutputTokens": 8})["maxOutputTokens"] == 8,
          "没生效")
    check("② runtime 覆盖：非 dict（None/字符串）原样返回 ✓",
          merge_model_settings(base, None) == base and merge_model_settings(base, "x") == base,
          "改了基座")
    check("② runtime 覆盖：不改原字典 ✓（避免同一 run 里污染后续读取）",
          merge_model_settings(base, {"temperature": 0.1}) is not base, "返回了同一个对象")


# ===================================================================== ③
def case_repeat_guard() -> None:
    _reset()
    _seed_config()
    for bad in (1, 0, -3):
        try:
            _optimize(evaluate=make_eval({None: [80]}), repeat=bad)
            ok, detail = False, f"repeat={bad} 竟然跑完了"
        except ValueError as err:
            ok, detail = "repeat" in str(err), str(err)
        check(f"③ repeat={bad} ⇒ 当场 ValueError ✓（单次评测不许拿来比参数 ✗）", ok, detail)
    try:
        _optimize(evaluate=make_eval({None: [80]}), repeat=po.MAX_REPEAT + 1)
        ok, detail = False, "超上限竟然跑完了"
    except ValueError as err:
        ok, detail = "repeat" in str(err), str(err)
    check("③ repeat 超上限 ⇒ ValueError ✓（repeat×候选数 次真模型评测，不许被一个查询参数触发）",
          ok, detail)


# ===================================================================== ④
def case_noise_band() -> None:
    _reset()
    _seed_config(temperature=0.7)
    # 基线 [80, 84] ⇒ 抖动带 4；候选 [85, 86] 均值 85.5，Δ=1.5 < 4 ⇒ 涨了但不算提升
    report = _optimize(evaluate=make_eval({None: [80.0, 84.0], 0.5: [85.0, 86.0]}),
                       repeat=2, maxCandidates=1)
    cand = report["candidates"][0]
    check("④ 基线抖动带算出来了（max−min）✓",
          report["baseline"]["jump"] == 4.0, report["baseline"])
    check("④ Δ>0 但没超过抖动带 ⇒ 不采纳 ✓（这正是参数调优最典型的假绿 ✗）",
          cand["delta"] > 0 and cand["accepted"] is False, cand)
    check("④ 理由记 `below_noise`（「涨了，但没超过抖动带」）✓ 而不是 `no_gain` ✗",
          cand["reason"] == "below_noise"
          and cand["reasonLabel"] == optimization_log.REASONS["below_noise"], cand)
    check("④ 没采纳 ⇒ 温度一格没被改 ✓", _temperature() == 0.7, _temperature())
    check("④ 没采纳 ⇒ 优化史里这一行 `accepted=0 / persisted=0` ✓（试过就要留痕 ✓）",
          _rows() and all(r["accepted"] == 0 and r["persisted"] == 0 for r in _rows()), _rows())


def _temperature() -> float:
    """当前生效温度（短连接读完即还 —— 泄漏连接会在连接池耗尽时变成「读到一半卡住」）。"""
    with db.engine.connect() as conn:
        return po.current_temperature(conn, AGENT)


def _rows() -> list[dict]:
    with db.engine.connect() as conn:
        return [dict(row._mapping) for row in  # noqa: SLF001
                conn.execute(text("SELECT * FROM agent_optimizations ORDER BY version"))]


# ===================================================================== ⑤
def case_accept_and_persist() -> None:
    _reset()
    _seed_config(temperature=0.7)
    report = _optimize(
        evaluate=make_eval({None: [80.0, 80.0], 0.5: [86.0, 87.0], 0.3: [79.0, 79.0]}),
        repeat=2, maxCandidates=2,
    )
    check("⑤ 抖动带为 0 时 Δ>0 即算提升 ✓（没有抖动就不存在「涨的那点可能是噪声」）",
          report["baseline"]["jump"] == 0.0 and report["candidates"][0]["accepted"] is True,
          report["baseline"])
    check("⑤ 两个候选都记了行 ✓（失败/落选的候选也是事实 ✓）",
          len(_rows()) == 2 and {r["target"] for r in _rows()} == {"params"}, len(_rows()))
    check("⑤ 采纳的是**涨幅最大**那个（0.5，Δ=6.5）✓ 而不是先跑的哪个 ✗",
          report["accepted"]["temperature"] == 0.5 and report["accepted"]["delta"] == 6.5,
          report["accepted"])
    check("⑤ 采纳 ⇒ `agent_configs.temperature` 真被写成 0.5 ✓",
          _temperature() == 0.5, _temperature())
    accepted_row = [r for r in _rows() if r["accepted"] == 1]
    check("⑤ 采纳行 `persisted=1` ✓（判决与生效在这一行同时为真 ✓）",
          len(accepted_row) == 1 and accepted_row[0]["persisted"] == 1, accepted_row)
    check("⑤ 落选行不置 `persisted` ✓（没生效的版本不许标成生效 ✗）",
          [r["persisted"] for r in _rows() if r["accepted"] == 0] == [0], _rows())
    check("⑤ 只动温度一列：提示词、技能、模型一个没变 ✓",
          _config_snapshot() == {"system_prompt": "你是剧本改写助手……", "skills": None,
                                 "model": None, "temperature": 0.5, "max_tokens": 4096},
          _config_snapshot())
    check("⑤ 回执里写好落库结果 ✓（`accepted.persisted=True`）",
          report["accepted"]["persisted"] is True, report["accepted"])


def _config_snapshot() -> dict:
    with db.engine.connect() as conn:
        row = conn.execute(text(
            "SELECT system_prompt, skills, model, temperature, max_tokens FROM agent_configs"
            " WHERE agent_type = :a"
        ), {"a": AGENT}).first()
    return dict(row._mapping) if row else {}


# ===================================================================== ⑥
def case_no_config_row() -> None:
    _reset()  # 没有任何 agent_configs 行
    report = _optimize(evaluate=make_eval({None: [80.0, 80.0], 0.5: [90.0, 91.0]}),
                       repeat=2, maxCandidates=1)
    check("⑥ 没有配置行 ⇒ 基线温度退回运行期默认值 ✓（拿真正会生效的值当基线 ✓）",
          report["current"]["temperature"] == DEFAULT_TEMPERATURE, report["current"])
    check("⑥ 采纳了但**没落库** ✓（照实 `persisted=False`，不是谎称生效 ✗）",
          report["accepted"]["accepted"] is True and report["accepted"]["persisted"] is False,
          report["accepted"])
    check("⑥ 失败理由写清楚了 ✓", "没有该 agent 的配置行" in report["accepted"].get("persistSkip", ""),
          report["accepted"].get("persistSkip"))
    with db.engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM agent_configs")).scalar()
    check("⑥ 绝不为了「看起来生效了」新建一行只带温度的配置 ✓（那会把 system_prompt 留空 ✗）",
          count == 0, count)
    check("⑥ 采纳行 `persisted=0` ✓（判决 ≠ 生效，这是两次不同的判断 ✓）",
          [r["persisted"] for r in _rows()] == [0], _rows())


# ===================================================================== ⑦
def case_version_and_prompt() -> None:
    _reset()
    _seed_config(temperature=0.7)
    # 造一条既有的提示词行：version 3。参数行的版本必须从 4 起，不许撞 3。
    with db.engine.begin() as conn:
        optimization_log.record(conn, agent_type=AGENT, version=3, case_id=CASE["id"],
                                score_before=70.0, score_after=75.0, accepted=True,
                                reason="accepted", prompt="旧的提示词")
    _optimize(evaluate=make_eval({None: [80.0, 80.0], 0.5: [80.0, 80.0], 0.3: [80.0, 80.0]}),
              repeat=2, maxCandidates=2)
    versions = [r["version"] for r in _rows()]
    check("⑦ 参数行版本接在既有最大 version 之后 ✓（[3, 4, 5]，不重号 ✓）",
          versions == [3, 4, 5], versions)
    shas = {r["prompt_sha"] for r in _rows() if r["target"] == "params"}
    check("⑦ 一轮内所有行的提示词指纹**完全一致** ✓（不一致就说明「固定提示词」这个前提破了 ✗）",
          len(shas) == 1 and "" not in shas, shas)
    check("⑦ 同分不采纳（Δ=0 记 `no_gain`）✓", all(
        r["reason"] == "no_gain" for r in _rows() if r["target"] == "params"), _rows())


# ===================================================================== ⑧
def case_no_candidate() -> None:
    _reset()
    _seed_config(temperature=0.0)
    evaluate = make_eval({None: [80.0, 81.0]})
    report = _optimize(evaluate=evaluate, repeat=2, maxCandidates=2)
    check("⑧ 当前已是 0.0 ⇒ 只跑基线、一个候选都不跑 ✓（不为了出结果去试更高温度 ✗）",
          len(evaluate.calls) == 2 and report["candidates"] == [], evaluate.calls)  # type: ignore[attr-defined]
    check("⑧ 如实说明为什么收工 ✓", "0.0" in report.get("reason", ""), report.get("reason"))
    check("⑧ 没跑候选 ⇒ 优化史一行都不写 ✓（没试过的事不记账 ✓）", _rows() == [], _rows())


# ===================================================================== ⑨
def case_runtime_mismatch() -> None:
    _reset()
    _seed_config(temperature=0.7)
    report = _optimize(
        evaluate=make_eval({None: [80.0, 80.0], 0.5: [95.0, 96.0]},
                           models={None: "ollama/qwen3:14b", 0.5: "ollama/qwen3:32b"}),
        repeat=2, maxCandidates=1,
    )
    cand = report["candidates"][0]
    check("⑨ 候选与基线换了模型 ⇒ 分数不可比 ⇒ 一律不采纳 ✓",
          cand["accepted"] is False and cand["reason"] == "runtime_mismatch", cand)
    check("⑨ 理由是人话 ✓",
          cand["reasonLabel"] == optimization_log.REASONS["runtime_mismatch"], cand)


# ===================================================================== ⑩
def case_wiring() -> None:
    def read(rel: str) -> str:
        return (BP / rel).read_text(encoding="utf-8")

    runtime_src = read("app/agent/runtime.py")
    check("⑩ runtime 真的用上了覆盖（`merge_model_settings(...)` 出现在装配处）✓",
          "merge_model_settings({" in runtime_src and 'options.get("modelSettings")' in runtime_src,
          "没接上")

    evaluator_src = read("app/agent/evaluation/evaluator.py")
    tree = ast.parse(evaluator_src)
    fn = next((n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == "evaluate_case"), None)
    check("⑩ `evaluate_case` 有 `settings` 形参 ✓", fn is not None and
          "settings" in [a.arg for a in fn.args.args + fn.args.kwonlyargs], "没有")
    builds_options = [n for n in ast.walk(fn) if isinstance(n, ast.Dict)
                      and any(isinstance(k, ast.Constant) and k.value == "modelSettings"
                              for k in n.keys)] if fn else []
    check("⑩ `evaluate_case` 把 settings 包成 `{\"modelSettings\": ...}` ✓",
          len(builds_options) == 1, len(builds_options))
    # 每个 `kind == "..."` 分支各调一次 `run_agent_with_instructions`；最后一个位置实参必须是
    # run_options，否则就是「只加形参、不传下去」⇒ 候选与基线跑的都是 DB 里那套温度（同一档跑两遍 ✗）。
    # ⚠️ 分支数**不写死** ✗（原先写 `== 4`：2026-11-07 加进 `grid_prompt` 第五档后这条就恒红，
    # 而"恒红"会被当成环境噪音忽略 —— 真正漏传 run_options 时也没人看见了 ✗✗）。
    # 判据改成三条：分支数与 `kind ==` 的档数对齐 ✓、下限守住历史四档 ✓、**每一个**都传了 ✓。
    kinds = {n.comparators[0].value for n in ast.walk(fn)
             if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name)
             and n.left.id == "kind" and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq)
             and isinstance(n.comparators[0], ast.Constant)
             and isinstance(n.comparators[0].value, str)} if fn else set()
    forwards = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "run_agent_with_instructions"] if fn else []
    wired = [n for n in forwards if n.args and isinstance(n.args[-1], ast.Name)
             and n.args[-1].id == "run_options"]
    check("⑩ `evaluate_case` 的每个 kind 分支都有一条 `run_agent_with_instructions` ✓ "
          "（加档却不加调用＝那一档跑不出来 ✗）",
          len({k for k in kinds}) >= 4 and len(forwards) == len(kinds),
          (len(kinds), len(forwards), sorted(kinds)))
    check("⑩ 每一档都把 run_options 传下去了 ✓（只加形参不传＝永远跑默认温度 ✗）",
          len(wired) == len(forwards) and len(wired) >= 4, (len(forwards), len(wired)))

    po_src = read("app/agent/evaluation/param_optimizer.py")
    check("⑩ 候选评测**必须带 temperature 覆盖** ✓（不带就等于把同一档位跑两遍 ✗）",
          '{"temperature": temp}' in po_src, "没带")
    check("⑩ 判采纳用的是抖动带 ✓（`delta > jump`）", "delta > jump" in po_src, "没比抖动带")

    creator_src = read("app/agent/creator.py")
    creator_tree = ast.parse(creator_src)
    fn2 = next((n for n in creator_tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "persist_agent_temperature"), None)
    check("⑩ `persist_agent_temperature` 存在 ✓", fn2 is not None, "没有")
    check("⑩ 它有 `conn` 形参 ✓（SQLite WAL 只允许一个写者：调用方攥着写锁时另开连接必 "
          "`database is locked`，实测白等 30 秒后炸）",
          fn2 is not None
          and "conn" in [a.arg for a in fn2.args.args + fn2.args.kwonlyargs], "没有")
    writer = next((n for n in creator_tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "_write_temperature"), None)
    update_cols = set()
    if writer is not None:
        for node in ast.walk(writer):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "values":
                for kw in node.keywords:
                    if kw.arg:
                        update_cols.add(kw.arg)
    check("⑩ 它只 update 温度与时间戳两格 ✓（参数调优凭什么动提示词 ✗）",
          update_cols == {"temperature", "updated_at"}, update_cols)
    check("⑩ 落库那一步真的把 conn 传下去了 ✓（不传＝自己另开连接＝WAL 写锁死等 ✗）",
          "persist_agent_temperature(agent_type, float(best[\"temperature\"]), conn)" in po_src,
          "没传")

    router_src = read("app/routers/evaluation.py")
    # 2026-11-07 起路由**不再直接 await 优化器**：一轮 423.9 秒的同步请求会被代理掐断（界面上报错、
    # 后台还在烧卡），改成「先入队、查进度」。所以这条判据跟着改口径：路由里必须能看见
    # 入队（`tune_runtime.submit`）与共用校验（`normalize_options`），而**真正的执行**在
    # `tune_runtime` 的默认 runner 里 —— 两边各验一半，断在哪一头都能定位。
    runtime_src = read("app/agent/evaluation/tune_runtime.py")
    check("⑩ 路由 `POST /evaluation/tune/{case_id}` 已接 ✓ 且改成**入队** ✓"
          "（直接 await 优化器＝那个会被掐断的 7 分钟请求 ✗）",
          '@router.post("/tune/{case_id}")' in router_src
          and "tune_runtime.submit(" in router_src
          and "await optimize_agent_temperature(" not in router_src, "没接")
    check("⑩ 入队前就用 `normalize_options` 校验 ✓（放行到跑的时候才炸＝白烧几轮评测 ✗）",
          "options = normalize_options(options)" in router_src, "没有")
    check("⑩ 后台 runner 真的调到优化器 ✓（路由只入队、runner 不调＝任务永远是空的 ✗）",
          "await optimize_agent_temperature(" in runtime_src, "没调")
    check("⑩ 路由注释写明「本仓新增、不是 TS 对齐项」✓（否则会被当成漏改 ✗）",
          "本仓新增" in router_src, "没写")


# ===================================================================== ⑪
def case_http() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    _reset()
    _seed_config(temperature=0.7)
    client = TestClient(app)

    bad = client.post(f"/api/v1/evaluation/tune/{HTTP_CASE}", json={"repeat": 1})
    check("⑪ HTTP: repeat=1 ⇒ 400 且带原因 ✓（**不是**悄悄抬到 2 —— 那等于替调用方撒谎 ✗）",
          bad.status_code == 400 and "repeat" in bad.text, (bad.status_code, bad.text[:120]))

    unknown = client.post("/api/v1/evaluation/tune/没这个case", json={"repeat": 2})
    check("⑪ HTTP: 未知 case ⇒ 404 ✓",
          unknown.status_code == 404, unknown.status_code)


# ===================================================================== ⑫
def case_endpoint_reads_params() -> None:
    _reset()
    _seed_config(temperature=0.7)
    _optimize(evaluate=make_eval({None: [80.0, 80.0], 0.5: [90.0, 91.0]}),
              repeat=2, maxCandidates=1)
    listing = optimization_log.list_records(agent_type=AGENT)
    item = (listing.get("items") or [{}])[0]
    check("⑫ `/optimizations` 能读出 `target=params` ✓ 且人话标签在 ✓"
          "（前端那一列不用改，直接按 TARGETS 渲染 ✓）",
          item.get("target") == "params"
          and item.get("targetLabel") == optimization_log.TARGETS["params"], item)
    check("⑫ 参数行没有提示词正文 ⇒ 指纹照实有值（同一份提示词）✓、页面上不会显示成「编的」✓",
          bool(item.get("promptSha")), item.get("promptSha"))


# ===================================================================== ⑬
def case_real_payload_temperature() -> None:
    """⑬ 真发一次 HTTP：温度必须原样出现在请求体里。

    为什么非得起个本地服务：「`merge_model_settings` 传对了」「`evaluate_case` 也透下去了」这两条
    静态检查都绿，也完全可能最后写进 payload 的是另一个值 —— adapter 里一个 `or 0.7` 就能把
    ``0.0`` 顶掉，而整轮结论照样算得出来、照样写进优化史，只是「试 0.0」实际试的是 0.7。
    这条把请求体抓下来看，是「这一格旋钮真的拧到了」的唯一实证。
    """
    import http.server
    import json as _json
    import threading

    from app.agent.runtime import default_generate
    from app.agent.tool import ToolRegistry

    seen: list[object] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 —— BaseHTTPRequestHandler 的口
            length = int(self.headers.get("Content-Length") or 0)
            payload = _json.loads(self.rfile.read(length) or b"{}")
            seen.append((payload.get("options") or {}).get("temperature"))
            reply = _json.dumps({
                "message": {"role": "assistant", "content": "好的"},
                "done": True, "prompt_eval_count": 1, "eval_count": 1,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for temp in (0.0, 0.7):
            asyncio.run(default_generate(
                config={"provider": "ollama", "baseUrl": f"http://127.0.0.1:{port}"},
                model="stub", instructions="只回一句", tools=ToolRegistry({}), message="跑一下",
                max_steps=1, model_settings={"temperature": temp, "maxOutputTokens": 64},
            ))
    finally:
        server.shutdown()
        server.server_close()

    check("⑬ 真发 HTTP：0.0 与 0.7 都原样进了请求体 ✓（0.0 被顶成 0.7＝这一格白调、结论还照样写 ✗）",
          seen == [0.0, 0.7], seen)


def main() -> int:
    case_candidates()
    case_runtime_override()
    case_repeat_guard()
    case_noise_band()
    case_accept_and_persist()
    case_no_config_row()
    case_version_and_prompt()
    case_no_candidate()
    case_runtime_mismatch()
    case_wiring()
    case_http()
    case_endpoint_reads_params()
    case_real_payload_temperature()

    failures = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAIL {name}  ← {detail!r}")
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
