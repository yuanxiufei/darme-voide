"""S6 自检：提示词优化器 + `POST /evaluation/optimize/{caseId}`（``evaluation/optimizer.ts`` 216 行）。

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
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="optimizer_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import agent_configs  # noqa: E402
from app.response import now  # noqa: E402
from app.routers import evaluation as ev_route  # noqa: E402
from app.services.agent_prompts import get_default_instructions  # noqa: E402
from app.services.agents import skills as sk  # noqa: E402
from app.services.evaluation import catalog as cat  # noqa: E402
from app.services.evaluation import optimizer as opt  # noqa: E402

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
                 candidates: list[str] | None = None) -> None:
        self.scores = scores
        self.models = models or ["m1"] * len(scores)
        self.candidates = candidates or [f"{LONG_PROMPT}候选{i}" for i in range(len(scores))]
        self.evaluate_calls: list[str] = []
        self.generate_calls: list[dict] = []

    async def evaluate_case(self, conn, case_def, instructions):  # noqa: ANN001
        self.evaluate_calls.append(instructions)
        index = len(self.evaluate_calls) - 1
        return _report(self.scores[index], self.models[index])

    async def generate_candidate_prompt(self, conn, agent_type, current, version, report):  # noqa: ANN001
        # ! 注意 `version` 是**当前 best 的版本**（第 1 轮是 0，即 Reference），不是迭代号 ->
        #    桩里必须按「第几次调用」取候选，用 `candidates[version - 1]` 会踩负索引。
        self.generate_calls.append({"current": current, "version": version})
        return self.candidates[len(self.generate_calls) - 1]


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

    # ================= 汇总 =================
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok_flag, detail in _RESULTS:
        print(("PASS  " if ok_flag else "FAIL  ") + name + ("" if ok_flag else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
