"""S6 自检：提示词优化器 + `POST /evaluation/optimize/{caseId}`（口径源自 ``evaluation/optimizer.ts`` 216 行）。

⚠️ 那个 TS 文件**已随 Node 后端移出本仓**，所以下面这些判据现在只守本仓自己的口径，
没有可跑的对拍物 —— 改判据不必同步外部文件，但也别因此以为「随便改都行」：
第 3 条的默认仍是零容差（``repeat=1``，与历史逐字一致），但 2026-11-10 起可显式传
``repeat=2–5``，用 Reference 自己的样本极差当抖动带（实测同配置重复测量极差 8 分）。

优化器是一条**状态机**，四类判据错了都不会报错、只会悄悄产出错误结论：

1. **防作弊**：``format_feedback`` **只给「维度名 + 分数」，绝不能带 detail**（detail 里
   可能含 rubric 的金标准答案）；
2. **runtime 一致性**：候选与 Reference 的 ``runtimeModel`` 不同 -> 本轮**不 accept**
   （哪怕分数更高 —— 分数不可比）；
3. **accept 判据**：严格高于**当前 best**（不是 Reference）-> 允许连续晋升；
4. **落库判据**：``autoPersist`` 且 ``best.version > 0`` 且 ``best.score > reference.score``。

外加 ``??``（nullish）语义：``iterations: 0`` 就是 0、``autoPersist: false`` 生效。

运行::

    ./.venv/Scripts/python.exe tests/optimizer_test.py
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="optimizer_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ⚠️ 单独跑时 cp936 控制台会把判据名里的 `✓` 打成 UnicodeEncodeError、一条判据都没统计就退 1
# （`run_all` 下不露头，因为它已经重配过 stdout —— 与 `engine_win_io_test` 同一个坑）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import func, select, update  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import agent_configs, agent_optimizations  # noqa: E402
from app.core.response import now  # noqa: E402
from app.routers import evaluation as ev_route  # noqa: E402
from app.services import optimization_log  # noqa: E402
from app.services.agent_prompts import get_default_instructions  # noqa: E402
from app.agent import skills as sk  # noqa: E402
from app.agent.evaluation import catalog as cat  # noqa: E402
from app.agent.evaluation import optimizer as opt  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


LONG_PROMPT = "你是一名分镜师。" + "保持工作流程不变，逐镜填写首尾帧提示词，注意运镜与景别。" * 3


def _report(total: int, model: str = "m1", dims: int = 2) -> dict:
    return {
        "caseId": "c1", "kind": "storyboard", "total": total, "runtimeModel": model,
        "dimensions": [
            {"name": f"维度{i}", "score": total / dims, "max": 50, "detail": f"金标准提示{i}"}
            for i in range(dims)
        ],
    }


class Harness:
    """把「评测」与「生成候选」都打桩，只留状态机本身真实执行。"""

    def __init__(self, scores: list[int], models: list[str] | None = None,
                 candidates: list[str] | None = None,
                 score_by_prompt: dict[str, int] | None = None,
                 generate_errors: dict[int, Exception] | None = None,
                 evaluate_errors: dict[int, Exception] | None = None) -> None:
        self.scores = scores
        self.models = models or ["m1"] * len(scores)
        self.candidates = candidates or [f"{LONG_PROMPT}候选{i}" for i in range(len(scores))]
        # 「同一份提示词总是同一个分」的桩：要判「基线取谁」的案子必须让分数**跟提示词绑定**，
        # 否则无论 v0 是哪份都返回同一个数，改错基线也看不出来（假绿）。
        self.score_by_prompt = dict(score_by_prompt or {})
        # 按**调用次数**（1 起）指定要抛的异常：复现「第 N 轮模型输出过短 / 评测超时」。
        # `scores` 只按**成功**的评测依次取，否则一次失败会让后面所有分数错位。
        self.generate_errors = dict(generate_errors or {})
        self.evaluate_errors = dict(evaluate_errors or {})
        self.evaluate_calls: list[str] = []
        self.generate_calls: list[dict] = []
        self._ok_calls = 0

    async def evaluate_case(self, conn, case_def, instructions):  # noqa: ANN001
        self.evaluate_calls.append(instructions)
        error = self.evaluate_errors.get(len(self.evaluate_calls))
        if error is not None:
            raise error
        self._ok_calls += 1
        if instructions in self.score_by_prompt:
            return _report(self.score_by_prompt[instructions])
        index = self._ok_calls - 1
        return _report(self.scores[index], self.models[index])

    async def generate_candidate_prompt(self, conn, agent_type, current, version, report):  # noqa: ANN001
        # ! 注意 `version` 是**当前 best 的版本**（第 1 轮是 0，即 Reference），不是迭代号 ->
        #    桩里必须按「第几次调用」取候选，用 `candidates[version - 1]` 会踩负索引。
        self.generate_calls.append({"current": current, "version": version})
        call_index = len(self.generate_calls)
        error = self.generate_errors.get(call_index)
        if error is not None:
            raise error
        return self.candidates[call_index - 1]


def _install(harness: Harness) -> None:
    opt.evaluate_case = harness.evaluate_case  # type: ignore[assignment]
    opt.generate_candidate_prompt = harness.generate_candidate_prompt  # type: ignore[assignment]


def main() -> int:  # noqa: C901
    # ================= format_feedback（防作弊）=================
    feedback = opt.format_feedback(_report(60))
    check("反馈: 只有「维度名：分数/满分」，**不含 detail**（防作弊隔离）",
          feedback == "- 维度0：30.0/50\n- 维度1：30.0/50"
          and "金标准提示" not in feedback,
          feedback)
    check("反馈: 没有维度时给空串", opt.format_feedback({}) == "")

    # ================= strip_code_fence =================
    check("围栏: 套了 ``` 就剥，没套则原样 trim",
          opt.strip_code_fence(f"```markdown\n{LONG_PROMPT}\n```") == LONG_PROMPT
          and opt.strip_code_fence(f"  {LONG_PROMPT}  ") == LONG_PROMPT)

    # ================= build_persist_candidate =================
    client = TestClient(app)
    client.post("/api/v1/ai-configs", json={
        "service_type": "text", "provider": "openai", "base_url": "https://api.example.com",
        "api_key": "k", "model": ["m"], "is_active": True,
    })
    with engine.begin() as conn:
        fresh = opt.build_persist_candidate(conn, "extractor", "新提示词")
    check("落库候选: 无既有配置 -> 默认显示名 + 默认 skill 绑定，systemPrompt 用最佳提示词",
          fresh.name == "角色场景提取" and fresh.system_prompt == "新提示词"
          and [s["id"] for s in fresh.skills] == sk.resolve_default_skills("extractor"),
          (fresh.name, fresh.skills))

    with engine.begin() as conn:
        conn.execute(agent_configs.insert().values(
            agent_type="extractor", name="我自己的名字", description="我自己的描述",
            skills='[{"id":"keep","enabled":true,"priority":9}]', is_active=True,
            created_at=now(), updated_at=now(),
        ))
        kept = opt.build_persist_candidate(conn, "extractor", "新提示词")
    check("落库候选: 有既有配置 -> name/description/skills **沿用 DB**（只改 systemPrompt）",
          kept.name == "我自己的名字" and kept.description == "我自己的描述"
          and kept.skills == [{"id": "keep", "enabled": True, "priority": 9}],
          (kept.name, kept.skills))

    with engine.begin() as conn:
        conn.execute(agent_configs.update().where(
            agent_configs.c.agent_type == "extractor").values(skills="{坏 JSON"))
        broken = opt.build_persist_candidate(conn, "extractor", "新提示词")
    check("落库候选: DB 里的 skills 损坏 -> **静默回退默认绑定**（不抛）",
          all(s.get("id") != "keep" for s in broken.skills), broken.skills)

    # ================= generate_candidate_prompt =================
    _LLM: list[dict] = []

    def _stub_llm(text: str) -> None:
        async def _fake(**kwargs) -> dict:
            _LLM.append(kwargs)
            return {"text": text, "tool_calls": [], "tool_results": [], "steps": []}

        opt.default_generate = _fake  # type: ignore[assignment]

    _stub_llm(f"```\n{LONG_PROMPT}\n```")
    with engine.begin() as conn:
        candidate = asyncio.run(opt.generate_candidate_prompt(
            conn, "storyboard_breaker", "旧提示词", 2, _report(40)))
    check("生成候选: 剥围栏后返回；走单发调用（无工具、模型取 textConfig.model）",
          candidate == LONG_PROMPT and _LLM[0]["tools"].tools == {}
          and _LLM[0]["max_steps"] == 1,
          candidate[:20])
    message = _LLM[0]["message"]
    check("生成候选: user message 里是「当前提示词（v2）」+ 只有分数的失分维度",
          "当前提示词（v2）：" in message and "旧提示词" in message
          and "- 维度0：20.0/50" in message and "金标准提示" not in message
          and message.rstrip().endswith("请针对上述失分维度改进提示词，输出改进后的完整提示词。"),
          message[-60:])
    check("生成候选: 用的是优化器指令（不是 creator 的）",
          _LLM[0]["instructions"] == opt.OPTIMIZER_INSTRUCTIONS)

    _stub_llm("太短")
    with engine.begin() as conn:
        short = ""
        try:
            asyncio.run(opt.generate_candidate_prompt(conn, "extractor", "旧", 1, _report(10)))
        except Exception as err:  # noqa: BLE001
            short = str(err)
    check("生成候选: 过短 -> **英文**报错 `Optimizer produced invalid candidate (length=2)`",
          short == "Optimizer produced invalid candidate (length=2)", short)

    # ================= 状态机 =================
    case_def = cat.load_case_by_id("sb-case-001")
    history_dir = Path(tempfile.mkdtemp(prefix="opt_hist_"))

    harness = Harness(scores=[50, 60, 55])
    _install(harness)
    with engine.begin() as conn:
        history = asyncio.run(opt.optimize_agent_prompt(
            conn, "storyboard_breaker", case_def,
            {"iterations": 2, "historyDir": str(history_dir), "autoPersist": True}))
    check("状态机: reference=v0；v1 接受（60>50）、v2 回滚（55<=60）",
          history["reference"]["score"] == 50
          and [i["accepted"] for i in history["iterations"]] == [True, False]
          and [i["score"] for i in history["iterations"]] == [60, 55],
          history["iterations"])
    check("状态机: best 停在 v1（版本/分数/提示词三件套）",
          history["best"] == {"version": 1, "score": 60, "prompt": harness.candidates[0]},
          history["best"])
    check("状态机: `iterations` 每条都带 `reason` ✓（accepted / no_gain / runtime_mismatch / "
          "eval_failed 同一套枚举）—— 这份 JSON 是唯一存着候选全文的审计物，只写 accepted 分不清成因",
          [i.get("reason") for i in history["iterations"]] == ["accepted", "no_gain"],
          history["iterations"])
    check("状态机: camelCase 键与 caseId/agentType 齐全",
          set(history) == {"agentType", "caseId", "reference", "iterations", "best", "persisted"}
          and history["caseId"] == "sb-case-001"
          and history["agentType"] == "storyboard_breaker",
          sorted(history))
    check("状态机: 生成候选时传入的是**当前 best 的提示词与版本**（第 1 轮是 v0 Reference，"
          "第 2 轮是刚接受的 v1）",
          [call["current"] for call in harness.generate_calls]
          == [get_default_instructions("storyboard_breaker"), harness.candidates[0]]
          and [call["version"] for call in harness.generate_calls] == [0, 1],
          harness.generate_calls)
    check("状态机: 评测顺序是 Reference → v1 候选 → v2 候选（共 3 次评测）",
          len(harness.evaluate_calls) == 3
          and harness.evaluate_calls[0] == get_default_instructions("storyboard_breaker")
          and harness.evaluate_calls[1] == harness.candidates[0]
          and harness.evaluate_calls[2] == harness.candidates[1],
          len(harness.evaluate_calls))

    hist_file = history_dir / "storyboard_breaker-sb-case-001.json"
    check("状态机: 历史写进 `{historyDir}/{agentType}-{caseId}.json`，且是 **indent=2**（非紧凑）",
          hist_file.exists()
          and '\n  "agentType"' in hist_file.read_text(encoding="utf-8")
          and json.loads(hist_file.read_text(encoding="utf-8"))["best"]["score"] == 60,
          hist_file.name)

    # 落库判据：best 严格高于 reference
    with engine.begin() as conn:
        row = conn.execute(select(agent_configs).where(
            agent_configs.c.agent_type == "storyboard_breaker")).first()
    check("状态机: best > reference -> 自动落库，persisted 带 version/score/name",
          history["persisted"] is not None and history["persisted"]["version"] == 1
          and history["persisted"]["score"] == 60 and row is not None
          and row.system_prompt == harness.candidates[0],
          history["persisted"])

    # 全部回滚 -> 不落库
    harness2 = Harness(scores=[50, 40, 30])
    _install(harness2)
    with engine.begin() as conn:
        history2 = asyncio.run(opt.optimize_agent_prompt(
            conn, "extractor", cat.load_case_by_id("ext-case-001"),
            {"iterations": 2, "historyDir": str(history_dir), "autoPersist": True}))
    check("状态机: 候选都没超过 Reference -> persisted=null（`best.score > reference.score` 不成立）",
          history2["persisted"] is None and history2["best"]["version"] == 0,
          (history2["persisted"], history2["best"]))

    # runtime 漂移 -> 不 accept（即使分数更高）
    harness3 = Harness(scores=[50, 99], models=["m1", "m2"])
    _install(harness3)
    with engine.begin() as conn:
        history3 = asyncio.run(opt.optimize_agent_prompt(
            conn, "extractor", cat.load_case_by_id("ext-case-001"),
            {"iterations": 1, "historyDir": str(history_dir), "autoPersist": True}))
    check("状态机: runtimeModel 与 Reference 不同 -> 本轮 **accepted=false 且 best 不变**（99 分也不认）",
          history3["iterations"][0]["accepted"] is False
          and history3["iterations"][0].get("reason") == "runtime_mismatch"
          and history3["best"]["version"] == 0
          and history3["iterations"][0]["score"] == 99,
          history3["iterations"])

    # autoPersist=false / iterations=0（nullish）
    harness4 = Harness(scores=[50, 80])
    _install(harness4)
    with engine.begin() as conn:
        no_persist = asyncio.run(opt.optimize_agent_prompt(
            conn, "voice_assigner", cat.load_case_by_id("voice-assigner-001"),
            {"iterations": 1, "historyDir": str(history_dir), "autoPersist": False}))
    check("状态机: autoPersist=false -> 不落库（persisted=null），但 best 照常晋升",
          no_persist["persisted"] is None and no_persist["best"]["version"] == 1,
          (no_persist["persisted"], no_persist["best"]))
    with engine.begin() as conn:
        saved_prompts = [
            row.system_prompt or ""
            for row in conn.execute(select(agent_configs).where(
                agent_configs.c.agent_type == "voice_assigner")).all()
        ]
    check("状态机: autoPersist=false 时真的没写 voice_assigner 配置",
          no_persist["best"]["prompt"] not in saved_prompts, saved_prompts)

    harness5 = Harness(scores=[50])
    _install(harness5)
    with engine.begin() as conn:
        zero = asyncio.run(opt.optimize_agent_prompt(
            conn, "extractor", cat.load_case_by_id("ext-case-001"),
            {"iterations": 0, "historyDir": str(history_dir), "autoPersist": False}))
    check("状态机: **iterations=0 就是 0**（`??` nullish，不是回落 3）—— 只评测 Reference",
          zero["iterations"] == [] and len(harness5.evaluate_calls) == 1,
          (zero["iterations"], len(harness5.evaluate_calls)))

    # ========== 进度与取消（入队口 `POST /tune-prompt/{case_id}` 的接线前提）==========
    # 这三条盯的是 2026-11-09 新增的 `progress` / `should_stop` 形参：默认 `None` 时
    # 老调用方行为逐字不变（上面所有用例就是证据，它们一个都没传），而一旦传了就必须
    # 逐跑发事件、且取消**只在两次评测之间**生效。
    events: list[dict] = []
    h_prog = Harness(scores=[50, 60, 55])
    _install(h_prog)
    with engine.begin() as conn:
        asyncio.run(opt.optimize_agent_prompt(
            conn, "storyboard_breaker", case_def,
            {"iterations": 2, "historyDir": str(history_dir), "autoPersist": False},
            progress=events.append))
    kinds = [item.get("type") for item in events]
    check("进度: start → run×3 → baseline / candidate×2 → done ✓（顺序即页面时间轴）",
          kinds[:1] == ["start"] and kinds.count("run") == 3
          and kinds.count("baseline") == 1 and kinds.count("candidate") == 2
          and kinds[-1] == "done", kinds)
    check("进度: 分母 repeat + iterations = 3 ✓（`repeat` 只加在 Reference 上；"
          "照抄温度格的 `repeat × (1 + n)` 会报 6）",
          events[0].get("totalRuns") == 3 and events[0].get("repeat") == 1
          and events[-1].get("doneRuns") == events[-1].get("totalRuns") == 3,
          (events[0], events[-1]))
    _baselines = [e for e in events if e.get("type") == "baseline"]
    check("进度: repeat=1 时 baseline 是 `jumpKind=\"none\"` + `jump=0` + 样本 1 个 ✓"
          "（「没测抖动带」与「测出来是 0」靠 jumpKind 分，不靠 jump 是不是 None）",
          bool(_baselines) and _baselines[0].get("jumpKind") == "none"
          and _baselines[0].get("jump") == 0
          and _baselines[0].get("samples") == [50.0],
          _baselines)
    cand_runs = [e for e in events if e.get("type") == "run" and e.get("phase") == "candidate"]
    check("进度: `run` 事件能分出 Reference 与「第 N 轮候选」✓（都印 `1/1` 会分不清在跑哪一跑）",
          [e for e in events if e.get("type") == "run" and e.get("phase") == "reference"]
          and [e.get("run") for e in cand_runs] == [1, 2]
          and all(isinstance(e.get("version"), int) and e["version"] >= 1 for e in cand_runs),
          [e for e in events if e.get("type") == "run"])
    check("进度: `run` 与 `version` 是**两个数** ✓（本轮是第 1 轮、而它在优化史里记作 v3 —— "
          "storyboard_breaker/sb-case-001 前面已经跑过两版；印「迭代 3/2」会让人以为轮次算错了）",
          cand_runs[0].get("version") != cand_runs[0].get("run")
          and cand_runs[1].get("version") == cand_runs[0].get("version") + 1,
          [{"run": e.get("run"), "version": e.get("version")} for e in cand_runs])

    h_stop = Harness(scores=[50, 60])
    _install(h_stop)
    try:
        with engine.begin() as conn:
            asyncio.run(opt.optimize_agent_prompt(
                conn, "storyboard_breaker", case_def,
                {"iterations": 2, "historyDir": str(history_dir), "autoPersist": False},
                should_stop=lambda: True))
        stopped = False
    except Exception as err:  # noqa: BLE001 —— 只认 `TuneCancelled` 这一类
        stopped = isinstance(err, opt.TuneCancelled)
    check("取消: `should_stop` 在**第一跑之前**就问 ✓ 且一次评测都没跑（先跑再问＝白等 100 秒）",
          stopped and not h_stop.evaluate_calls, len(h_stop.evaluate_calls))
    check("取消: `TuneCancelled` **不是** `ValueError` 子类 ✓"
          "（继承了会被路由那层 `except` 翻成 400「你的参数不对」）",
          not issubclass(opt.TuneCancelled, ValueError), opt.TuneCancelled.__mro__[:3])

    # 负数 `iterations`：`/tune-prompt` 路由层会挡掉（不认、回默认 3），但 `/optimize` 与 CLI
    # 直接把 options 交进来 ⇒ 分母会算成 `1 + (-2) = -1`，`done` 时 `doneRuns=1 > totalRuns=-1`，
    # 进度条永久走不满、读起来像卡住。夹到 0 就是「只评一次 Reference」，语义与 iterations=0 同。
    events_neg: list[dict] = []
    h_neg = Harness(scores=[50])
    _install(h_neg)
    with engine.begin() as conn:
        asyncio.run(opt.optimize_agent_prompt(
            conn, "storyboard_breaker", case_def,
            {"iterations": -2, "historyDir": str(history_dir), "autoPersist": False},
            progress=events_neg.append))
    neg_kinds = [item.get("type") for item in events_neg]
    check("进度: 负数 iterations 夹到 0 ✓（不夹 ⇒ totalRuns=-1、done 时 1 > -1）",
          events_neg and events_neg[0].get("totalRuns") == 1
          and events_neg[0].get("iterations") == 0
          and events_neg[-1].get("doneRuns") == 1 and neg_kinds.count("candidate") == 0,
          (events_neg[0].get("totalRuns") if events_neg else None,
           events_neg[-1].get("doneRuns") if events_neg else None, neg_kinds))

    # ================= 基线取谁 / 版本号怎么排（2026-11-09） =================
    # ① 基线＝**运行时在用的那份**：`agent_configs.system_prompt` 优先，没有配置行才回退默认。
    #    原先一律取出厂默认 ⇒ 重跑时拿默认那份的分数当靶子，候选只要高过默认就被采纳并写回配置，
    #    把已经更好的现网版本顶掉（88 → 82），且**不抛任何异常**、优化史还写着 score_before=80。
    live_prompt = LONG_PROMPT + "【现网已经生效的这一版】"
    with engine.begin() as conn:
        conn.execute(update(agent_configs)
                     .where(agent_configs.c.agent_type == "storyboard_breaker")
                     .values(system_prompt=""))
        empty = opt.effective_prompt(conn, "storyboard_breaker")
    check("基线: 配置行存在但 `system_prompt` 是空串 ⇒ 回退默认、`source=\"default\"` ✓"
          "（空串不许读成「用户就想要空提示词」，那会拿一个空 v0 去评测）",
          empty == (get_default_instructions("storyboard_breaker"), "default"), empty[1])
    with engine.begin() as conn:
        conn.execute(update(agent_configs)
                     .where(agent_configs.c.agent_type == "storyboard_breaker")
                     .values(system_prompt=live_prompt))
    # ⚠️ 分数必须**跟提示词绑定**：现网那份 88、出厂默认 80、候选 82。若基线误取出厂默认，
    #    82 > 80 就被判采纳并写回配置（88 分那份被顶掉）；绑定之后这条判据才不是假绿。
    live_candidate = LONG_PROMPT + "候选"
    h_live = Harness(scores=[0, 0], candidates=[live_candidate], score_by_prompt={
        live_prompt: 88,
        get_default_instructions("storyboard_breaker"): 80,
        live_candidate: 82,
    })
    _install(h_live)
    with engine.begin() as conn:
        hist_live = asyncio.run(opt.optimize_agent_prompt(
            conn, "storyboard_breaker", case_def,
            {"iterations": 1, "historyDir": str(history_dir), "autoPersist": True}))
    check("基线: 评的是 `agent_configs.system_prompt`（现网在用的那份），不是出厂默认 ✓",
          h_live.evaluate_calls[0] == live_prompt
          and hist_live["reference"]["prompt"] == live_prompt
          and hist_live["reference"]["source"] == "config"
          and hist_live["reference"]["score"] == 88,
          (h_live.evaluate_calls[:1], hist_live["reference"]))
    check("基线: 候选 82 < 现网 88 ⇒ 不采纳、`persisted=null` ✓（零容差反过来也成立）",
          hist_live["iterations"][0]["accepted"] is False
          and hist_live["persisted"] is None,
          (hist_live["iterations"], hist_live["persisted"]))
    with engine.begin() as conn:
        live_now = conn.execute(select(agent_configs.c.system_prompt).where(
            agent_configs.c.agent_type == "storyboard_breaker")).first()
    check("基线: 库里仍是 88 分那一版 ✓（拿默认当基线就会在这里被 82 分顶掉）",
          live_now is not None and live_now[0] == live_prompt, live_now)

    # ② 版本号跨运行连续：同一 (agent, case) 再跑**不重新从 v1 数** —— 否则优化史里同 target
    #    同 case 出现两个 v1，而 `mark_persisted` 按 `(agent, version, case_id, target)` 定位，
    #    一次会标中多行（上次没落库的那几轮被标成「已生效」，且不抛异常）。
    with engine.begin() as conn:
        before = conn.execute(select(func.max(agent_optimizations.c.version)).where(
            agent_optimizations.c.agent_type == "storyboard_breaker",
            agent_optimizations.c.case_id == "sb-case-001")).scalar() or 0
    h_seq = Harness(scores=[88, 95], candidates=[LONG_PROMPT + "更好的候选"])
    _install(h_seq)
    with engine.begin() as conn:
        hist_seq = asyncio.run(opt.optimize_agent_prompt(
            conn, "storyboard_breaker", case_def,
            {"iterations": 1, "historyDir": str(history_dir), "autoPersist": True}))
    check(f"版本号: 跨运行连续 ⇒ 这次记作 v{before + 1}（上一行已经是 v{before}），不重复用 v1 ✓",
          hist_seq["iterations"][0]["version"] == before + 1
          and hist_seq["persisted"] is not None
          and hist_seq["persisted"]["version"] == before + 1,
          (before, hist_seq["iterations"][0].get("version"), hist_seq["persisted"]))
    with engine.begin() as conn:
        versions = conn.execute(select(agent_optimizations.c.version).where(
            agent_optimizations.c.agent_type == "storyboard_breaker",
            agent_optimizations.c.case_id == "sb-case-001",
            agent_optimizations.c.target == "system_prompt")).scalars().all()
    check("版本号: 同一 (agent, case, target) 里 `version` **不重复** ✓"
          f"（已跑 {len(versions)} 轮 / {len(set(versions))} 个号；重复 ⇒ `mark_persisted` 一次标中多行）",
          len(versions) == len(set(versions)), sorted(versions))
    with engine.begin() as conn:
        marked = optimization_log.mark_persisted(
            conn, agent_type="storyboard_breaker", case_id="sb-case-001",
            version=before + 1, target="system_prompt")
    check("版本号: 最新那版 `mark_persisted` 命中**恰好 1 行** ✓（重跑过的表上也不许一次标两行）",
          marked == 1, marked)

    # ============= 单轮候选跑崩：不许废掉整轮（与温度/技能/工具集三格同口径） =============
    # 病根：本格原先 `generate_candidate_prompt` / `evaluate_case` 是**裸调用** ⇒ 抛错直接
    # 冒泡到 `tune_runtime`、任务 failed。而 `generate_candidate_prompt` 那句 ValueError
    # （模型改写出来太短 / 剥完围栏是空串）是**设计内的正常失败**：`iterations=3` 时第 2 轮
    # 撞上，第 1 轮已经花掉的评测钱全丢、`history` 文件一个字不落（原写入在循环之后）。
    gen_events: list[dict] = []
    h_gen_fail = Harness(
        scores=[50, 62],
        candidates=[LONG_PROMPT + "会被生成器丢掉的那份", LONG_PROMPT + "乙"],
        generate_errors={1: ValueError("候选提示词太短（< 20 字），已丢）")},
    )
    _install(h_gen_fail)
    with engine.begin() as conn:
        hist_gen_fail = asyncio.run(opt.optimize_agent_prompt(
            conn, "voice_assigner", cat.load_case_by_id("voice-assigner-001"),
            {"iterations": 2, "historyDir": str(history_dir), "autoPersist": True},
            progress=gen_events.append))
    check("单轮失败: 生成候选抛错 ⇒ 那一轮记 `eval_failed`（score 为 None），"
          "**不废掉整轮** —— 第 2 轮照跑并采纳；且 `prompt` 留空（生成就失败 ⇒ 压根没有候选，"
          "拿 best 顶上去会让人读成「这一轮试的就是 best 那份」）",
          hist_gen_fail["iterations"][0]["reason"] == "eval_failed"
          and hist_gen_fail["iterations"][0]["score"] is None
          and hist_gen_fail["iterations"][0]["prompt"] is None
          and hist_gen_fail["iterations"][1]["accepted"] is True
          and hist_gen_fail["best"]["score"] == 62,
          hist_gen_fail["iterations"])
    check("单轮失败: 失败轮仍占分母 ⇒ `doneRuns == totalRuns == 3`（1 跑 Reference + 2 轮），"
          "不许「任务成功但进度停在一半」",
          gen_events[-1].get("doneRuns") == gen_events[-1].get("totalRuns") == 3,
          (gen_events[-1].get("doneRuns"), gen_events[-1].get("totalRuns")))
    with engine.begin() as conn:
        failed_rows = conn.execute(select(agent_optimizations.c.version)
                                   .where(agent_optimizations.c.agent_type == "voice_assigner",
                                          agent_optimizations.c.target == "system_prompt",
                                          agent_optimizations.c.reason == "eval_failed",
                                          agent_optimizations.c.score_after.is_(None))).all()
    check("单轮失败: 优化史里有一条 `eval_failed`、`score_after` 为空 ✓"
          "（写成 0 会在页面上显示「掉了 50 分」）",
          len(failed_rows) == 1, failed_rows)

    h_eval_fail = Harness(scores=[50, 66], evaluate_errors={2: RuntimeError("评测超时")})
    _install(h_eval_fail)
    with engine.begin() as conn:
        hist_eval_fail = asyncio.run(opt.optimize_agent_prompt(
            conn, "voice_assigner", cat.load_case_by_id("voice-assigner-001"),
            {"iterations": 2, "historyDir": str(history_dir), "autoPersist": True}))
    check("单轮失败: 评测抛错只作废**那一轮**（第 1 轮失败、第 2 轮 66 > 50 照常采纳）"
          "—— `scores` 按成功次数取，不会因为一次失败让后面全部错位",
          hist_eval_fail["iterations"][0]["reason"] == "eval_failed"
          and hist_eval_fail["iterations"][1]["accepted"] is True
          and hist_eval_fail["best"]["score"] == 66,
          hist_eval_fail["iterations"])

    # 评测阶段失败时，候选**已经生成好了**（模型调用已经花掉）⇒ 那份全文必须留住：
    # 另三格的 `eval_failed` 都带 `prompt=instructions`（非空），本格原先一律写空串 ⇒
    # 优化史那行的 `prompt_sha` 是 `""`，页面上读不出「这一轮试了什么」。
    failed_version = hist_eval_fail["iterations"][0]["version"]
    with engine.begin() as conn:
        failed_sha = conn.execute(select(agent_optimizations.c.prompt_sha)
                                  .where(agent_optimizations.c.agent_type == "voice_assigner",
                                         agent_optimizations.c.target == "system_prompt",
                                         agent_optimizations.c.version == failed_version,
                                         agent_optimizations.c.reason == "eval_failed")).scalar()
    expected_sha = hashlib.sha1(h_eval_fail.candidates[0].encode("utf-8")).hexdigest()[:16]
    check("单轮失败: 评测抛错时那份**已生成**的候选全文被留住 ✓"
          "（`history` 里存全文、优化史里存指纹；写成空串等于把花掉的那次模型调用一起丢掉）",
          hist_eval_fail["iterations"][0]["prompt"] == h_eval_fail.candidates[0]
          and failed_sha == expected_sha,
          (hist_eval_fail["iterations"][0]["prompt"], failed_sha, expected_sha))

    h_cancel_in_eval = Harness(scores=[50, 60],
                               evaluate_errors={2: opt.TuneCancelled("别跑了")})
    _install(h_cancel_in_eval)
    try:
        with engine.begin() as conn:
            asyncio.run(opt.optimize_agent_prompt(
                conn, "voice_assigner", cat.load_case_by_id("voice-assigner-001"),
                {"iterations": 2, "historyDir": str(history_dir), "autoPersist": False}))
        cancel_bubbled = False
    except Exception as err:  # noqa: BLE001 —— 只认 `TuneCancelled` 这一类
        cancel_bubbled = isinstance(err, opt.TuneCancelled)
    check("单轮失败: 评测途中的 `TuneCancelled` **原样抛出** ✓"
          "（新加的 `except Exception` 不许把它降级成 `eval_failed` —— 那会把「用户取消」"
          "记成「这一轮评测失败」然后接着跑）",
          cancel_bubbled, cancel_bubbled)

    # 取消只在两次评测之间生效 ⇒ 取消那一刻，已跑完的轮次必须已经在盘上
    stop_calls = {"n": 0}

    def _stop_after_first_round() -> bool:
        stop_calls["n"] += 1
        return stop_calls["n"] >= 3  # 1=Reference 前、2=第 1 轮前、3=第 2 轮前 ⇒ 取消

    h_stop2 = Harness(scores=[50, 60])
    _install(h_stop2)
    stop_dir = Path(tempfile.mkdtemp(prefix="opt_stop_"))
    try:
        with engine.begin() as conn:
            asyncio.run(opt.optimize_agent_prompt(
                conn, "voice_assigner", cat.load_case_by_id("voice-assigner-001"),
                {"iterations": 3, "historyDir": str(stop_dir), "autoPersist": False},
                should_stop=_stop_after_first_round))
    except Exception:  # noqa: BLE001 —— 取消路径本身已有判据，这里只关心盘上落下什么
        pass
    stopped_file = stop_dir / "voice_assigner-voice-assigner-001.json"
    check("取消: 已跑完的那一轮**已经落盘** ✓（逐轮 flush；原来只在循环之后写一次 ⇒ 取消即丢档，"
          "而这份 JSON 是唯一存着候选提示词全文的地方）",
          stopped_file.exists()
          and len(json.loads(stopped_file.read_text(encoding="utf-8"))["iterations"]) == 1,
          stopped_file.exists())

    # 落库结论也必须写进盘上那份 JSON：逐轮 flush 发生在落库**之前**，不补最后一次，
    # 文件里就永远没有 `persisted` 字段（内存里有、文件里没有）⇒ 读档的人看不出这一版到底生效没有。
    persisted_dir = Path(tempfile.mkdtemp(prefix="opt_persist_"))
    h_persist = Harness(scores=[50, 70])
    _install(h_persist)
    with engine.begin() as conn:
        hist_persist = asyncio.run(opt.optimize_agent_prompt(
            conn, "voice_assigner", cat.load_case_by_id("voice-assigner-001"),
            {"iterations": 1, "historyDir": str(persisted_dir), "autoPersist": True}))
    on_disk = json.loads((persisted_dir / "voice_assigner-voice-assigner-001.json")
                         .read_text(encoding="utf-8"))
    check("落库结论也落盘: 盘上 JSON 的 `persisted` 与回执一致 ✓"
          "（否则「这一版真写进配置了吗」只有内存里有答案，而顺手 `cat` 的那份文件里没有）",
          on_disk.get("persisted") is not None
          and on_disk["persisted"]["version"] == hist_persist["best"]["version"]
          and on_disk["persisted"]["score"] == 70
          and bool(on_disk["persisted"]["name"]),
          on_disk.get("persisted"))

    # 两份审计物不许分叉：盘上 JSON 的每一轮，都要能在优化史里找到**同版本、同判决**的一行。
    # 现有判据是「history 的 reason 列表」与「表里 eval_failed 行数」各查一份 —— 那只能证明
    # 两份账**各自**对，证不了它们**逐轮对得上**：某处少调一次 `record`，history 有 3 条、
    # 表里只有 2 条，上面那些判据全绿，而页面上整整少一轮。
    audit_dir = Path(tempfile.mkdtemp(prefix="opt_audit_"))
    h_audit = Harness(scores=[50, 70, 60],
                      evaluate_errors={4: RuntimeError("第 3 轮评测超时")})
    _install(h_audit)
    with engine.begin() as conn:
        asyncio.run(opt.optimize_agent_prompt(
            conn, "voice_assigner", cat.load_case_by_id("voice-assigner-001"),
            {"iterations": 3, "historyDir": str(audit_dir), "autoPersist": False}))
    audit_json = json.loads((audit_dir / "voice_assigner-voice-assigner-001.json")
                            .read_text(encoding="utf-8"))
    audit_versions = [item["version"] for item in audit_json["iterations"]]
    with engine.begin() as conn:
        audit_rows = conn.execute(
            select(agent_optimizations.c.version,
                   agent_optimizations.c.accepted,
                   agent_optimizations.c.reason)
            .where(agent_optimizations.c.agent_type == "voice_assigner",
                   agent_optimizations.c.target == "system_prompt",
                   agent_optimizations.c.version.in_(audit_versions))
        ).all()
    audit_by_version = {int(row[0]): (bool(row[1]), str(row[2])) for row in audit_rows}
    check("审计对账: 盘上 JSON 的每一轮都能在优化史里找到**同版本、同判决**的一行 ✓"
          "（两份审计物分叉时，上面那些「各查一份」的判据全绿，而页面上少一整轮）",
          len(audit_versions) == 3
          and all(audit_by_version.get(item["version"])
                  == (bool(item["accepted"]), item["reason"])
                  for item in audit_json["iterations"])
          and [audit_by_version.get(v, (None, None))[1] for v in audit_versions]
          == ["accepted", "no_gain", "eval_failed"],
          (audit_json["iterations"], audit_by_version))

    # ---------- repeat：用 Reference 自己的样本极差当抖动带 ----------
    # 实测同配置重复测量极差 **8 分**，所以这里刻意造「候选比均值高 3 分、但没超过 8 分」的局面：
    # 零容差会采纳（把噪声写进配置），实测抖动带不该采纳。
    ref_dir = Path(tempfile.mkdtemp(prefix="opt_repeat_"))
    noise_events: list[dict] = []
    noise_out = io.StringIO()
    _install(Harness(scores=[80, 72, 76, 79]))
    with engine.begin() as conn, contextlib.redirect_stdout(noise_out):
        noised = asyncio.run(opt.optimize_agent_prompt(
            conn, "voice_assigner", cat.load_case_by_id("voice-assigner-001"),
            {"iterations": 1, "repeat": 3, "historyDir": str(ref_dir), "autoPersist": True},
            progress=noise_events.append))
    check("抖动带: repeat=3 取 Reference 样本**极差 8**（80/72/76）、分数取均值 76 ✓"
          "（不折半、与另三格的 jump 同口径）",
          noised["reference"]["samples"] == [80.0, 72.0, 76.0]
          and noised["reference"]["jump"] == 8.0
          and noised["reference"]["score"] == 76.0
          and noised["reference"]["jumpKind"] == "within"
          and noised["reference"]["repeat"] == 3,
          noised["reference"])
    check("抖动带: 候选 79 > 均值 76、但没超过 76+8 ⇒ no_gain、不落库 ✓"
          "（零容差会把这 3 分噪声当成改进写进配置）",
          noised["iterations"][0]["accepted"] is False
          and noised["iterations"][0]["reason"] == "no_gain"
          and noised["best"]["version"] == 0
          and noised.get("persisted") is None,
          noised["iterations"])
    check("抖动带: 进度分母是 repeat + iterations = 4 ✓（不是温度格的 3×(1+1)=6）",
          noise_events[0].get("totalRuns") == 4
          and noise_events[-1].get("doneRuns") == 4
          and [e.get("run") for e in noise_events
               if e.get("type") == "run" and e.get("phase") == "reference"] == [1, 2, 3],
          (noise_events[0], noise_events[-1]))

    _install(Harness(scores=[80, 72, 76, 85]))
    with engine.begin() as conn:
        lifted = asyncio.run(opt.optimize_agent_prompt(
            conn, "voice_assigner", cat.load_case_by_id("voice-assigner-001"),
            {"iterations": 1, "repeat": 3, "historyDir": str(ref_dir), "autoPersist": True}))
    check("抖动带: 候选 85 > 76+8 ⇒ 采纳**且落库** ✓（同一条判据的两侧，不是只测保守那侧）",
          lifted["iterations"][0]["accepted"] is True
          # 版本号是**跨运行连续**的（`next_version`），这里只断言「best 就是那一轮」
          and lifted["best"]["version"] == lifted["iterations"][0]["version"]
          and bool(lifted.get("persisted")),
          lifted["iterations"])

    # stdout 同一口径：`repeat=1` 印「没测抖动带」，不印「抖动带 0」（「算了但算出 0」是另一件事；
    # 与 `tune_runtime._stage_text` 的 baseline 分支同一条纪律，两个面不许一个诚实一个糊）
    plain_out = io.StringIO()
    _install(Harness(scores=[70, 75]))
    with engine.begin() as conn, contextlib.redirect_stdout(plain_out):
        asyncio.run(opt.optimize_agent_prompt(
            conn, "voice_assigner", cat.load_case_by_id("voice-assigner-001"),
            {"iterations": 1, "repeat": 1, "historyDir": str(ref_dir), "autoPersist": False}))
    check("抖动带: stdout 在 `repeat=1` 时说**没测**、`repeat=3` 时印实测 8 ✓"
          "（「没测出抖动带」与「测出来是 0」印成同一句，读终端的人没法分辨）",
          "没测抖动带" in plain_out.getvalue()
          and "抖动带 0" not in plain_out.getvalue()
          and "抖动带 8" in noise_out.getvalue()
          # `[SKIP]` 那行（未采纳、不落库）原先自己拼了一句 `抖动带={jump}` ⇒ 只钉住上面
          # 三处照样漏它：同一条口径在四个打印点上各写一遍，就是这个下场
          and "抖动带=" not in noise_out.getvalue(),
          (plain_out.getvalue().strip().splitlines()[-1:] or "",
           [ln for ln in noise_out.getvalue().splitlines() if "[SKIP]" in ln] or ""))

    # `iterations=0` 是被文档反复推荐的那个用法（「只量 Reference 的抖动带、不生成候选」），
    # 但此前只测过 `normalize_prompt_options` 认它合法 —— 真跑这条路径一条判据都没有，
    # 状态机在「零轮候选还能不能正常收尾」上是从未被执行过的。
    quiet_events: list[dict] = []
    quiet_dir = Path(tempfile.mkdtemp(prefix="opt_quiet_"))
    _install(Harness(scores=[60, 70]))
    with engine.begin() as conn, contextlib.redirect_stdout(io.StringIO()):
        quiet = asyncio.run(opt.optimize_agent_prompt(
            conn, "voice_assigner", cat.load_case_by_id("voice-assigner-001"),
            {"iterations": 0, "repeat": 2, "historyDir": str(quiet_dir)},
            progress=quiet_events.append))
    check("`iterations=0` 真跑得通：零轮候选也正常收尾 ✓（回执轮次空、best 仍是 v0、不落库）",
          quiet["iterations"] == [] and quiet["best"]["version"] == 0
          and quiet.get("persisted") is None
          and quiet["reference"]["samples"] == [60.0, 70.0]
          and quiet["reference"]["jump"] == 10.0,
          quiet)
    check("`iterations=0` 的分母只是 `repeat`（2），跑完 doneRuns == totalRuns ✓"
          "（照抄成 `repeat + iterations` 后再漏掉零轮那一支，进度条会永远停在 0/2）",
          quiet_events[0].get("totalRuns") == 2
          and quiet_events[-1].get("doneRuns") == 2
          and quiet_events[-1].get("totalRuns") == 2
          and not [e for e in quiet_events if e.get("type") == "candidate"],
          quiet_events)

    # ---------- 两条边界：越界即拒，不夹逼 ----------
    def _rejects(**kwargs: object) -> bool:
        try:
            opt.normalize_prompt_options(kwargs)
        except ValueError:
            return True
        return False

    check("边界: `iterations` > 20 越界即拒、不夹到 20 ✓（夹了回执的轮数与调用方要的对不上；"
          "上限依据＝20 轮 × 85–165 秒 ≈ 28–55 分钟，而任务表在进程内没有超时回收）",
          _rejects(iterations=opt.MAX_ITERATIONS + 1)
          and not _rejects(iterations=opt.MAX_ITERATIONS)
          and opt.normalize_prompt_options({"iterations": 0})["iterations"] == 0
          and opt.normalize_prompt_options({})["iterations"] == 3,
          opt.normalize_prompt_options({}))
    check("边界: `repeat` 只收 1–5 的整数 ✓（1＝不测抖动带是**合法**值，别照抄温度格的 `repeat≥2` 硬拒）",
          _rejects(repeat=0) and _rejects(repeat=opt.MAX_PROMPT_REPEAT + 1)
          and _rejects(repeat=1.0) and _rejects(repeat=True)
          and opt.normalize_prompt_options({"repeat": 1})["repeat"] == 1
          and opt.normalize_prompt_options({})["repeat"] == 1,
          opt.normalize_prompt_options({"repeat": 2}))

    # ================= 端点 =================
    check("端点: case 不存在 -> 404 `未知基准 case：x`",
          client.post("/api/v1/evaluation/optimize/不存在").status_code == 404
          and client.post("/api/v1/evaluation/optimize/不存在").json()["message"]
          == "未知基准 case：不存在")

    _ROUTE_ARGS: list[dict] = []

    async def _stub_optimize(conn, agent_type, case_def, options):  # noqa: ANN001
        _ROUTE_ARGS.append({"agentType": agent_type, "options": options})
        return {"agentType": agent_type, "caseId": case_def.get("id"),
                "reference": {"score": 1, "dimensions": [], "prompt": "p"},
                "iterations": [], "best": {"version": 0, "score": 1, "prompt": "p"},
                "persisted": None}

    ev_route.optimize_agent_prompt = _stub_optimize  # type: ignore[assignment]
    ok = client.post("/api/v1/evaluation/optimize/sb-case-001", json={"iterations": 2})
    check("端点: 成功 -> 200 成功信封 + history 原样（iterations 透传）",
          ok.status_code == 200 and ok.json()["data"]["caseId"] == "sb-case-001"
          and _ROUTE_ARGS[-1]["options"] == {"iterations": 2, "autoPersist": True},
          (_ROUTE_ARGS[-1], ok.status_code))
    client.post("/api/v1/evaluation/optimize/sb-case-001", json={"iterations": 0})
    zero_options = _ROUTE_ARGS[-1]["options"]
    client.post("/api/v1/evaluation/optimize/sb-case-001",
                json={"iterations": "x", "auto_persist": False})
    invalid_options = _ROUTE_ARGS[-1]["options"]
    check("端点: iterations 非法（0/负数/字符串）-> 回落 3；auto_persist 只有显式 false 才关",
          zero_options == {"iterations": 3, "autoPersist": True}
          and invalid_options == {"iterations": 3, "autoPersist": False},
          (zero_options, invalid_options))

    client.post("/api/v1/evaluation/optimize/sb-case-001", json={"iterations": 2.0})
    float_options = _ROUTE_ARGS[-1]["options"]
    check("端点: `iterations: 2.0` 在 JS 眼里是整数 -> 这里也当 2（不是回落 3）",
          float_options["iterations"] == 2, float_options)

    client.post("/api/v1/evaluation/optimize/sb-case-001", content="{坏".encode("utf-8"),
                headers={"Content-Type": "application/json"})
    bad_body_options = _ROUTE_ARGS[-1]["options"]
    client.post("/api/v1/evaluation/optimize/sb-case-001")
    empty_body_options = _ROUTE_ARGS[-1]["options"]
    check("端点: body 宽容（坏 JSON / 空 body -> 缺省 iterations=3、autoPersist=true）",
          bad_body_options == {"iterations": 3, "autoPersist": True}
          and empty_body_options == {"iterations": 3, "autoPersist": True},
          (bad_body_options, empty_body_options))

    async def _boom(conn, agent_type, case_def, options):  # noqa: ANN001
        raise RuntimeError("优化炸了")

    ev_route.optimize_agent_prompt = _boom  # type: ignore[assignment]
    failed = client.post("/api/v1/evaluation/optimize/sb-case-001")
    check("端点: 运行期异常 -> 400 且 message 取 str(err)",
          failed.status_code == 400 and failed.json()["message"] == "优化炸了", failed.json())

    # ---------- `/tune-prompt` 入队口：preview 的分母与两条「入队前就拒」 ----------
    _TUNE_CALLS: list[dict] = []

    def _stub_submit(**kwargs):  # noqa: ANN003, ANN202
        _TUNE_CALLS.append(kwargs)
        return {"taskId": "tune-stub", "status": "queued", "totalRuns": kwargs.get("total_runs")}

    _real_submit = ev_route.tune_runtime.submit
    ev_route.tune_runtime.submit = _stub_submit  # type: ignore[assignment]
    try:
        queued = client.post("/api/v1/evaluation/tune-prompt/sb-case-001",
                             json={"iterations": 2, "repeat": 3})
        payload = queued.json()["data"]
        check("/tune-prompt: repeat=3 时 preview.runs = 3+2 = 5、jumpKind=within，"
              "并把 5 传给 submit ✓（不是温度格的 3×(1+2)=9）",
              queued.status_code == 200 and payload["preview"]["runs"] == 5
              and payload["preview"]["repeat"] == 3
              and payload["preview"]["jumpKind"] == "within"
              and _TUNE_CALLS[-1].get("total_runs") == 5
              and (_TUNE_CALLS[-1].get("options") or {}).get("repeat") == 3,
              (payload.get("preview"), _TUNE_CALLS[-1] if _TUNE_CALLS else None))

        too_many = client.post("/api/v1/evaluation/tune-prompt/sb-case-001",
                               json={"iterations": opt.MAX_ITERATIONS + 1})
        bad_repeat = client.post("/api/v1/evaluation/tune-prompt/sb-case-001",
                                 json={"repeat": opt.MAX_PROMPT_REPEAT + 1})
        check("/tune-prompt: `iterations > 20` / `repeat > 5` 都在**入队前** 400 ✓"
              "（等到排上队才炸，用户会以为已经提交成功）",
              too_many.status_code == 400 and bad_repeat.status_code == 400
              and len(_TUNE_CALLS) == 1,
              (too_many.json(), bad_repeat.json(), len(_TUNE_CALLS)))
    finally:
        ev_route.tune_runtime.submit = _real_submit  # type: ignore[assignment]

    # ================= 汇总 =================
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok_flag, detail in _RESULTS:
        print(("PASS  " if ok_flag else "FAIL  ") + name + ("" if ok_flag else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
