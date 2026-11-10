"""参数调优的**后台任务表**守卫（``app/agent/evaluation/tune_runtime.py``）。

为什么要有它
============
2026-11-07 真机跑 `/tune`（``_bench/tune_probe.py``）：eval-case-001 / repeat=2 / 1 个候选，
四跑合计 **423.9 秒**。挂在同步 ``POST`` 上的后果不是"慢一点"，而是：

* 中间代理先超时、后台还在烧卡 ⇒ 界面报失败、日志显示跑完（最难查的一种）；
* 七分钟里没有任何中间态 ⇒ 分不清"在跑第 1 跑"还是"卡死了"；
* 连点两次就排队跑两轮七分钟（同一份提示词，纯浪费）。

改成"入队 + 查进度"之后，新的失效方式也出现了，本套盯的就是这些**新**失效方式：

① 入队必须**立刻**返回（把 ``submit`` 写成同步等，等于没改）；
② 真跑完时 ``status=succeeded`` 且 ``receipt`` 在；失败要把原因写进 ``error``（不许只打日志）；
③ 进度要能画出"第几跑/共几跑"，且**总数早定**（基线 + 每个候选各 repeat 次）；
④ 串行：两个任务同时提交，第二个必须排队等第一个（一张卡，并发只会互相抢显存）；
⑤ 同一个 (agent, case) 已在跑 ⇒ 不重复入队（``deduped=True``）；
⑥ 取消排队中的 ⇒ 立刻 ``cancelled`` 且**一次都不跑**；取消运行中的 ⇒ 在**两次评测之间**停；
⑦ 任务用自己的 DB 连接（请求那条连接响应返回就归还了 —— 拿它跑后台，症状是跑到一半
   ``connection is closed``）；
⑧ 取消是**正常收场**（``TuneCancelled``）不是参数错（不许翻成 400 "你的参数不对"）。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/tune_runtime_test.py
"""
from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
import tempfile
import time
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

TMP = Path(tempfile.mkdtemp(prefix="tune-runtime-"))
os.environ["DATA_ROOT"] = TMP.as_posix()

from sqlalchemy import text  # noqa: E402

from app.core import db  # noqa: E402
from app.agent.evaluation import param_optimizer as po  # noqa: E402
from app.agent.evaluation import tune_runtime  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []

AGENT = "script_rewriter"
CASE = {"id": "rewrite-basic", "kind": "script_rewrite"}
OPTS = {"repeat": 2, "maxCandidates": 1, "autoPersist": True}


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def read(rel: str) -> str:
    return (BP / rel).read_text(encoding="utf-8")


def _reset_db() -> None:
    with db.engine.begin() as conn:
        conn.execute(text("DELETE FROM agent_optimizations"))
        conn.execute(text("DELETE FROM agent_configs"))


def _seed_config(*, agent_type: str = AGENT, temperature: float = 0.7,
                 prompt: str = "你是剧本改写助手……") -> None:
    now = "2026-11-07T00:00:00.000Z"
    with db.engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO agent_configs (agent_type, name, description, system_prompt, temperature,"
            " max_tokens, is_active, created_at, updated_at)"
            " VALUES (:a, '脚本改写', 'desc', :p, :t, 4096, 1, :ts, :ts)"
        ), {"a": agent_type, "p": prompt, "t": temperature, "ts": now})


def _wait(task_id: str, *, want: tuple[str, ...] = tune_runtime.TERMINAL,
          timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    snapshot = tune_runtime.get(task_id) or {}
    while time.time() < deadline:
        snapshot = tune_runtime.get(task_id) or {}
        if snapshot.get("status") in want:
            return snapshot
        time.sleep(0.02)
    return snapshot


def make_runner(*, runs: int = 2, seconds: float = 0.12, calls: list | None = None,
                fail: bool = False, label: str = ""):
    """假 runner：按 ``runs`` 次模拟"每跑之前问一次取消"的真优化器形状。

    必须**自己也走** ``job["progressCallback"]`` —— 否则量到的只是提交/完成两端，
    中间那七分钟的进度条是不是活的没人验。
    """

    async def runner(conn, job):  # noqa: ANN001
        cb = job.get("progressCallback")
        total = int(job.get("totalRuns") or runs)
        for index in range(total):
            if job.get("abort"):
                raise po.TuneCancelled(f"在完成 {index}/{total} 跑之后被取消")
            if cb is not None:
                cb({"type": "run", "phase": "baseline", "run": index + 1, "repeat": runs,
                    "doneRuns": index, "totalRuns": total})
            await asyncio.sleep(seconds)
            if calls is not None:
                calls.append(f"{label}{index + 1}")
        if fail:
            raise RuntimeError("假失败：模型没回来")
        if cb is not None:
            cb({"type": "done", "doneRuns": total, "totalRuns": total})
        return {"agentType": job["agentType"], "caseId": job["caseId"],
                "baseline": [80.0] * runs, "candidates": [], "accepted": None,
                "reason": "fake", "label": label or None}

    return runner


# ===================================================================== ①
def case_async_submit() -> None:
    tune_runtime.reset()
    started = time.time()
    queued = tune_runtime.submit(case_id="case-a", case_def={"id": "case-a"}, agent_type=AGENT,
                                 options=OPTS, runner=make_runner(seconds=0.4))
    elapsed = time.time() - started
    check("① 入队**立刻**返回 ✓（写成同步等就还是那个会被代理掐断的 7 分钟请求 ✗）",
          elapsed < 0.3, round(elapsed, 3))
    check("① 返回带 `taskId` / `pollUrl` / `queuePosition` ✓（前端凭这三样就能查进度）",
          bool(queued.get("taskId")) and str(queued.get("pollUrl") or "").endswith(queued["taskId"])
          and "queuePosition" in queued, queued)
    check("① 提交时状态是 queued ✓（还没开跑就报 running 是在骗进度条）",
          queued.get("status") in ("queued", "running"), queued.get("status"))

    done = _wait(str(queued["taskId"]))
    check("② 跑完 `status=succeeded` ✓ 且回执在 ✓（回执是这次调优唯一的产物，丢了就等于白跑）",
          done.get("status") == "succeeded" and isinstance(done.get("receipt"), dict),
          (done.get("status"), done.get("error")))
    check("② 跑完 `finishedAt` 有值 ✓（没有它，前端只能一直显示「进行中」）",
          bool(done.get("finishedAt")), done.get("finishedAt"))


# ===================================================================== ③
def case_progress_and_failure() -> None:
    tune_runtime.reset()
    queued = tune_runtime.submit(case_id="case-c", case_def={"id": "case-c"}, agent_type=AGENT,
                                 options=OPTS, runner=make_runner(runs=2, seconds=0.1))
    done = _wait(str(queued["taskId"]))
    check("③ 进度总数**早定**：repeat×(1+候选) = 2×2 = 4 ✓（跑到一半才知道总数画不出进度条 ✗）",
          done.get("totalRuns") == 4, done.get("totalRuns"))
    check("③ 逐跑推进 `doneRuns` 走到 4 ✓、`stage` 是人话而不是空串 ✓",
          done.get("doneRuns") == 4 and bool(done.get("stage")), (done.get("doneRuns"),
                                                                  done.get("stage")))

    tune_runtime.reset()
    queued = tune_runtime.submit(case_id="case-d", case_def={"id": "case-d"}, agent_type=AGENT,
                                 options=OPTS, runner=make_runner(seconds=0.05, fail=True))
    failed = _wait(str(queued["taskId"]))
    check("② 失败 ⇒ `status=failed` 且 `error` 写明原因 ✓（只打日志＝界面永远转圈 ✗）",
          failed.get("status") == "failed" and "假失败" in str(failed.get("error")),
          (failed.get("status"), failed.get("error")))


# ===================================================================== ③b
def case_wait() -> None:
    """``wait=true`` 那条同步兼容路径（脚本/探针用它最省事，所以它也得是好的）。"""
    tune_runtime.reset()
    quick = tune_runtime.submit(case_id="case-w1", case_def={"id": "case-w1"}, agent_type=AGENT,
                                options=OPTS, runner=make_runner(seconds=0.02))
    done = tune_runtime.wait(str(quick["taskId"]), timeout=20)
    check("③ `wait()` 等到终态就返回快照 ✓ 且回执在 ✓（脚本靠它拿结果，不能只等到 `running`）",
          done is not None and done.get("status") == "succeeded"
          and isinstance(done.get("receipt"), dict), done and done.get("status"))

    slow = tune_runtime.submit(case_id="case-w2", case_def={"id": "case-w2"}, agent_type=AGENT,
                               options=OPTS, runner=make_runner(seconds=0.3))
    partial = tune_runtime.wait(str(slow["taskId"]), timeout=0.05)
    check("③ `wait()` 超时 ⇒ 回当时快照并标 `timedOut` ✓、任务**不停**（把超时当成取消会白烧一半）",
          partial is not None and partial.get("timedOut") is True
          and partial.get("status") in ("queued", "running"), partial)
    _wait(str(slow["taskId"]), timeout=20)
    after = tune_runtime.get(str(slow["taskId"])) or {}
    check("③ 超时之后任务仍自己跑完 ✓（`timedOut` 只是「这次没等到」，不是结果丢了）",
          after.get("status") == "succeeded", after.get("status"))


# ===================================================================== ④
def case_serial() -> None:
    tune_runtime.reset()
    calls: list[str] = []
    first = tune_runtime.submit(case_id="case-s1", case_def={"id": "case-s1"}, agent_type=AGENT,
                                options=OPTS,
                                runner=make_runner(runs=2, seconds=0.2, calls=calls, label="A"))
    second = tune_runtime.submit(case_id="case-s2", case_def={"id": "case-s2"}, agent_type=AGENT,
                                options=OPTS,
                                 runner=make_runner(runs=2, seconds=0.05, calls=calls, label="B"))
    _wait(str(first["taskId"]), want=("running",), timeout=5)
    snapshot = tune_runtime.get(str(second["taskId"])) or {}
    check("④ 第二个任务在第一个跑着时**仍在排队** ✓（一张卡并发只会互相抢显存 ✗）",
          snapshot.get("status") == "queued", snapshot.get("status"))

    _wait(str(first["taskId"]))
    _wait(str(second["taskId"]))
    check("④ 跑完顺序＝入队顺序 ✓（A 全跑完才轮到 B，一次都没交错）",
          [item[0] for item in calls] == ["A"] * 4 + ["B"] * 4, calls)


# ===================================================================== ⑤
def case_dedup() -> None:
    tune_runtime.reset()
    first = tune_runtime.submit(case_id="case-d1", case_def={"id": "case-d1"}, agent_type=AGENT,
                                options=OPTS, runner=make_runner(seconds=0.25))
    again = tune_runtime.submit(case_id="case-d1", case_def={"id": "case-d1"}, agent_type=AGENT,
                                options={"repeat": 3, "maxCandidates": 0, "autoPersist": True},
                                runner=make_runner(seconds=0.25))
    check("⑤ 同一 (agent, case) 已在跑 ⇒ **不重复入队**，返回同一个 taskId 且标 deduped ✓ "
          "（同一份提示词跑第二遍只是再烧七分钟 ✗）",
          again.get("taskId") == first.get("taskId") and again.get("deduped") is True, again)
    _wait(str(first["taskId"]))

    other = tune_runtime.submit(case_id="case-d2", case_def={"id": "case-d2"}, agent_type=AGENT,
                                options=OPTS, runner=make_runner(seconds=0.02))
    check("⑤ 换了 case 就不是重复 ⇒ 正常入队 ✓（去重只认 agent+case，不许把别的活吞掉 ✗）",
          other.get("taskId") != first.get("taskId") and other.get("deduped") is False, other)
    _wait(str(other["taskId"]))


# ===================================================================== ⑥
def case_cancel() -> None:
    tune_runtime.reset()
    calls: list[str] = []
    blocking = tune_runtime.submit(case_id="case-x1", case_def={"id": "case-x1"}, agent_type=AGENT,
                                   options=OPTS,
                                   runner=make_runner(runs=4, seconds=0.25, calls=calls, label="X"))
    queued = tune_runtime.submit(case_id="case-x2", case_def={"id": "case-x2"}, agent_type=AGENT,
                                 options=OPTS,
                                 runner=make_runner(runs=2, seconds=0.05, calls=calls, label="Y"))
    cancelled = tune_runtime.cancel(str(queued["taskId"]))
    check("⑥ 取消**排队中**的 ⇒ 立刻 cancelled ✓、且 `changed=true` ✓",
          cancelled.get("status") == "cancelled" and cancelled.get("changed") is True,
          cancelled.get("status"))

    time.sleep(0.1)
    check("⑥ 排队时就被取消的那个**一次都没跑** ✓（跑起来再停＝白烧一次评测 ✗）",
          not any(item.startswith("Y") for item in calls), calls)

    running = tune_runtime.cancel(str(blocking["taskId"]))
    check("⑥ 取消**运行中**的 ⇒ 返回仍 running 但 stage 说明「跑完这一跑就停」✓ "
          "（一次真模型生成从外面掐不断，假装能秒停才是骗人 ✗）",
          running.get("status") == "running" and "取消中" in str(running.get("stage")),
          (running.get("status"), running.get("stage")))

    done = _wait(str(blocking["taskId"]))
    check("⑥ 取消后 `status=cancelled`、`receipt` 不在 ✓（半截结果不许冒充完整回执 ✗）",
          done.get("status") == "cancelled" and done.get("receipt") is None,
          (done.get("status"), bool(done.get("receipt"))))
    check("⑥ 取消是**在两次评测之间**生效：已开跑的那一跑跑完才停 ✓（不是杀掉正在生成的那次）",
          0 < len([item for item in calls if item.startswith("X")]) < 4, calls)

    again = tune_runtime.cancel(str(blocking["taskId"]))
    check("⑥ 已是终态再取消 ⇒ 不报错、`changed=false` ✓（「已经结束了」不是失败 ✗）",
          again is not None and again.get("changed") is False, again and again.get("changed"))


# ===================================================================== ⑦
def case_real_optimizer_wiring() -> None:
    """真优化器（**不 mock 顶层**）接上 ``progress`` / ``should_stop`` 之后的行为。"""
    _reset_db()
    _seed_config(temperature=0.7)
    events: list[dict] = []

    def make_eval(script: dict[object, list[float]]):
        used = {key: 0 for key in script}

        async def run(conn, case_def, instructions, settings=None):  # noqa: ANN001
            key = (settings or {}).get("temperature")
            seq = script[key]
            index = used[key]
            used[key] = index + 1
            return {"total": seq[min(index, len(seq) - 1)], "sections": [], "penalties": []}

        return run

    with db.engine.begin() as conn:
        asyncio.run(po.optimize_agent_temperature(
            conn, AGENT, CASE, {"repeat": 2, "maxCandidates": 1, "autoPersist": False},
            evaluate=make_eval({None: [80.0, 80.0], 0.5: [70.0, 70.0]}),
            progress=events.append))

    kinds = [item.get("type") for item in events]
    check("③ 优化器真的逐跑发进度 ✓（顺序：start → 逐跑 run×4 → baseline / candidate → done）",
          kinds[:1] == ["start"] and kinds.count("run") == 4 and "baseline" in kinds
          and "candidate" in kinds and kinds[-1] == "done", kinds)
    check("③ `start` 就给出总跑数 4 ✓（repeat 2 × (1 基线 + 1 候选) = 4；早定才能画进度）",
          events and events[0].get("totalRuns") == 4, events[0] if events else None)
    check("③ 最后一跑的 `doneRuns` ＝ totalRuns ✓（中途丢计数会让进度条永远差一截）",
          events and events[-1].get("doneRuns") == events[-1].get("totalRuns") == 4,
          events[-1] if events else None)

    calls: list[object] = []

    async def never(*_args, **_kwargs):  # noqa: ANN002, ANN003
        calls.append(1)
        return {"total": 99.0}

    with db.engine.begin() as conn:
        try:
            asyncio.run(po.optimize_agent_temperature(
                conn, AGENT, CASE, {"repeat": 2, "maxCandidates": 1, "autoPersist": False},
                evaluate=never, should_stop=lambda: True))
            raised = False
        except po.TuneCancelled:
            raised = True
    check("⑧ `should_stop` 在**第一跑之前**就问 ⇒ 抛 `TuneCancelled` 且一次评测都没跑 ✓"
          "（先跑一轮再问＝取消白等 100 秒 ✗）",
          raised and not calls, (raised, len(calls)))
    check("⑧ `TuneCancelled` **不是** `ValueError` 子类 ✓（继承 ValueError 会被路由那层 "
          "`except` 翻成 400「你的参数不对」✗）",
          not issubclass(po.TuneCancelled, ValueError), po.TuneCancelled.__mro__[:3])


# ===================================================================== ⑨
def case_wiring() -> None:
    runtime_src = read("app/agent/evaluation/tune_runtime.py")
    check("⑦ 任务自己开连接、且必须 `begin()` ✓（拿请求那条 conn 跑后台＝响应一返回就 "
          "`connection is closed` ✗；`connect()` 则不提交 ✗）",
          "db_module.engine.begin()" in runtime_src
          and "db_module.engine.connect()" not in runtime_src,
          "connect() 里的写入在 close() 时一律回滚 ⇒ 回执写 persisted=true、库里一个字没有")
    check("④ 单工作线程 + 队列 ✓（线程按任务起＝并发抢显存 ✗）",
          "threading.Thread(" in runtime_src and "queue.Queue()" in runtime_src
          and runtime_src.count("threading.Thread(") == 1, "没有")
    check("⑦ 默认 runner 把 `progress` 与 `should_stop` 都透给优化器 ✓",
          "progress=job.get(\"progressCallback\")" in runtime_src
          and "should_stop=lambda: bool(job.get(\"abort\"))" in runtime_src, "没透")

    # ⚠️ 上面那条只是**存在性**断言：三处写的是同一串字面量，`in` 只要命中一处就算过 ⇒
    # 删掉 `_tools_runner` 里那处仍然绿。而后果是工具集那跑（8–12 分钟）**取消失效**：
    # `cancel()` 只置 `job["abort"]=True`，没人去问 ⇒ 跑完落进 `else` 分支，
    # 把用户看到的 `cancelled` 直接盖成 `succeeded`（而且结论可能已采纳）。
    # 所以必须**逐个 runner** 判，不是判「字面量存在过」。
    tree = ast.parse(runtime_src)
    wired: dict[str, bool] = {}
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name.endswith("_runner"):
            keywords: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    keywords |= {kw.arg for kw in sub.keywords if kw.arg}
            wired[node.name] = {"progress", "should_stop"} <= keywords
    check("⑦ 四个 runner **各自**都透 `progress` 与 `should_stop` ✓"
          "（只判字面量存在过 ⇒ 删掉 tools 那处仍绿，取消会白等 8–12 分钟并被盖成 succeeded ✗）",
          wired == {"_default_runner": True, "_skills_runner": True,
                    "_tools_runner": True, "_prompt_runner": True},
          str(wired))

    # 工具集基线缺明细时，那句话不许写「出现 0 个工具」：缺明细 ⇒ `current_tools` 读不到账
    # ⇒ tools 必然空，而「0 个工具」在页面上读作「模型一次都没调工具」，与事实正相反。
    stage = getattr(tune_runtime, "_stage_text")
    bad = stage({"type": "baseline", "score": 80.0, "tools": [], "doneRuns": 1, "totalRuns": 1,
                 "steps": {"unavailable": "toolCallsNotList"}}, "tools")
    check("⑦ 工具集基线缺明细 ⇒ 那句话如实说「没有明细」，不写「出现 0 个工具」✓"
          "（缺明细时 tools 必然空，写成 0 个＝把读数缺失说成模型没调过 ✗）",
          "0 个工具" not in bad and "明细" in bad, bad)
    good = stage({"type": "baseline", "score": 80.0, "tools": ["a", "b"], "steps": {},
                  "doneRuns": 1, "totalRuns": 5}, "tools")
    check("⑦ 反验：有明细时照常报工具数 ✓（判据不许见谁都喊 ✗）", "2 个工具" in good, good)

    # import 就起线程 ⇒ 每个进程（含 CLI/定时器）都白挂一个阻塞在空队列上的线程 ✗
    probe = subprocess.run(
        [sys.executable, "-c",
         "import threading;from app.agent.evaluation import tune_runtime;"
         "print([t.name for t in threading.enumerate() if t.name == 'tune-worker'])"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(BP),
        env={**os.environ, "DATA_ROOT": TMP.as_posix()},
    )
    check("④ import **不**起工作线程 ✓（第一次 submit 才起；否则 CLI/定时器进程里白挂一个线程 ✗）",
          probe.returncode == 0 and probe.stdout.strip().endswith("[]"),
          (probe.returncode, probe.stdout.strip()[-120:], probe.stderr.strip()[-120:]))

    router_src = read("app/routers/evaluation.py")
    check("⑨ 路由改成 **入队** ✓（`tune_runtime.submit(` 在、且不再在请求里直接 await 优化器 ✓）",
          "tune_runtime.submit(" in router_src
          and "await optimize_agent_temperature(" not in router_src, "没改")
    check("⑨ 校验抽到 `normalize_options` 且**入队前**调用 ✓（入队放行、排到才炸＝白烧几轮 ✗）",
          "options = normalize_options(options)" in router_src, "没有")
    check("⑨ 查进度 / 取消两个端点都在 ✓",
          '@router.get("/tune/{task_id}")' in router_src
          and '@router.post("/tune/{task_id}/cancel")' in router_src, "没有")
    check("⑧ 取消走 `success(...)` **不是** `bad_request` ✓（用户自己按取消，不是他参数写错 ✗）",
          "def cancel_tune_task" in router_src
          and "bad_request" not in router_src.split("def cancel_tune_task")[1].split("@router")[0],
          "有 bad_request")


# ===================================================================== ⑩
def case_http() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    _reset_db()
    _seed_config(temperature=0.7)
    tune_runtime.reset()
    client = TestClient(app)

    created = client.post("/api/v1/evaluation/tune/script-rewriter-001", json={"repeat": 2})
    payload = created.json().get("data") or {}
    check("⑨ HTTP: 默认 POST ⇒ 200 且回 `taskId`（不是回执）✓ —— 同步等 7 分钟正是要修的那个坑",
          created.status_code == 200 and bool(payload.get("taskId"))
          and "receipt" not in payload and payload.get("pollUrl"), payload)

    task_id = str(payload.get("taskId") or "")
    polled = client.get(f"/api/v1/evaluation/tune/{task_id}")
    check("⑨ HTTP: `GET /tune/{taskId}` 能查 ✓ 且带 `status` / `doneRuns` / `totalRuns` ✓",
          polled.status_code == 200
          and {"status", "doneRuns", "totalRuns"} <= set((polled.json().get("data") or {})),
          polled.json().get("data"))

    listed = client.get("/api/v1/evaluation/tune?limit=5")
    check("⑨ HTTP: `GET /tune` 能列 ✓（并发多轮时，光有一个 taskId 找不到别的）",
          listed.status_code == 200
          and isinstance((listed.json().get("data") or {}).get("tasks"), list),
          listed.status_code)

    unknown = client.get("/api/v1/evaluation/tune/tune-不存在")
    check("⑨ HTTP: 未知 taskId ⇒ 404 ✓（不是 200 + 空 body —— 那会让前端以为还在跑 ✗）",
          unknown.status_code == 404, unknown.status_code)
    unknown_cancel = client.post("/api/v1/evaluation/tune/tune-不存在/cancel")
    check("⑨ HTTP: 取消未知任务 ⇒ 404 ✓", unknown_cancel.status_code == 404,
          unknown_cancel.status_code)

    stopped = client.post(f"/api/v1/evaluation/tune/{task_id}/cancel")
    check("⑨ HTTP: 取消真任务 ⇒ 200 且 `status` 落在 cancelled/running 上 ✓",
          stopped.status_code == 200
          and (stopped.json().get("data") or {}).get("status") in
          ("cancelled", "running", "succeeded", "failed"),
          (stopped.status_code, stopped.json().get("data")))
    _wait(task_id, timeout=10)


# ===================================================================== ⑫
def case_prompt_end_to_end() -> None:
    """提示词格**真链路**：``submit(target="system_prompt")`` → worker → ``_prompt_runner``
    → 真 ``optimize_agent_prompt``（只把「评测」和「LLM 改写」两件外设打桩）。

    ⚠️ 为什么非要有这一条：⑩ 喂的是手写事件字典、⑪ 把 ``submit`` 整个打桩了、
    ``optimizer_test`` 那六条不经过任务表 —— 三处都绿的情况下，下面这些事可以同时全错：
    ① ``_runner_for`` 把 ``system_prompt`` 落到温度 runner（温度格的 receipt 是另一套键）；
    ② ``submit`` 的 options 白名单漏了 ``iterations``（永远按默认 3 轮跑，传 2 也没人理）；
    ③ 分母照抄 ``repeat × (1 + n)``（任务 done 了进度条还停在半程）；
    ④ ``autoPersist`` 传进了却没到优化器（「只出结论」变成真写配置，或反过来）。
    这四件都只在**真跑一遍**时露头。
    """
    from app.agent.evaluation import optimizer as opt

    def report(total: int) -> dict:
        return {"caseId": CASE["id"], "kind": CASE["kind"], "total": total,
                "runtimeModel": "m1",
                "dimensions": [{"name": "维度", "score": total, "max": 100, "detail": "金标准"}]}

    scores = [65, 70, 75]
    seen: list[str] = []

    async def fake_evaluate(conn, case_def, instructions):  # noqa: ANN001
        seen.append(str(instructions))
        return report(scores[min(len(seen) - 1, len(scores) - 1)])

    async def fake_candidate(conn, agent_type, current, version, best_report):  # noqa: ANN001
        # `version` 是**当前 best 的版本**（第 1 轮即 0）⇒ 候选文本按它 +1 命名
        return f"候选 v{version + 1}"

    saved = (opt.evaluate_case, opt.generate_candidate_prompt, opt.benchmarks_dir)
    opt.evaluate_case = fake_evaluate  # type: ignore[assignment]
    opt.generate_candidate_prompt = fake_candidate  # type: ignore[assignment]
    # 默认落点是仓库里的 benchmarks/history —— 测试必须隔离，否则每跑一次就往真目录塞一个文件
    opt.benchmarks_dir = lambda: TMP / "benchmarks"  # type: ignore[assignment]
    try:
        # ---- 第一轮：iterations=2 ⇒ 3 跑，两轮都涨 ⇒ 连续晋升到 v2 ----
        _reset_db()
        _seed_config()
        tune_runtime.reset()
        started = tune_runtime.submit(
            case_id=CASE["id"], case_def=CASE, agent_type=AGENT,
            options={"iterations": 2, "autoPersist": False},
            target="system_prompt", candidate_count=2)
        done = _wait(str(started["taskId"]))
        receipt = done.get("receipt") or {}

        check("⑫ 真链路跑完 ⇒ `succeeded` 且带 receipt ✓（不是排在队列里就返回）",
              done.get("status") == "succeeded" and isinstance(receipt, dict)
              and bool(receipt), (done.get("status"), done.get("error")))
        check("⑫ 分母 = 1 + iterations = 3 且跑满 ✓（`repeat` 那格根本没有，按 0 算会报 0/3）",
              done.get("totalRuns") == 3 and done.get("doneRuns") == 3,
              (done.get("doneRuns"), done.get("totalRuns")))
        check("⑫ 快照里 `repeat` 是 None、`iterations` 原样带出 ✓"
              "（repeat 写 1 会被读成「每档跑 1 次」）",
              done.get("repeat") is None and done.get("iterations") == 2,
              (done.get("repeat"), done.get("iterations")))
        check("⑫ 跑的是**提示词** runner 而不是温度那格 ✓（温度格的 receipt 里没有 reference/best）",
              "reference" in receipt and "best" in receipt, sorted(receipt.keys())[:6])
        check("⑫ `iterations=2` 真的传到优化器 ✓（白名单漏键 ⇒ 静默按默认 3 轮跑、多烧一次钱）",
              len(seen) == 3, len(seen))
        check("⑫ 采纳判据比的是**当前 best** 不是 Reference ✓"
              "（65→70→75 两轮都该晋升；跟 Reference 比会得到同样的版本号，故另判分数链）",
              receipt.get("best", {}).get("version") == 2
              and [item.get("score") for item in receipt.get("iterations") or []] == [70, 75],
              receipt.get("best"))
        check("⑫ `autoPersist=false` ⇒ 只出结论、`persisted` 为 None ✓（本格是唯一会写配置的一格，"
              "开关传丢了就是静默改线上提示词）",
              receipt.get("persisted") is None, receipt.get("persisted"))

        # ---- 第二轮：autoPersist 默认 True + 候选更高 ⇒ **真写** agent_configs ----
        _reset_db()
        _seed_config(prompt="旧提示词")
        tune_runtime.reset()
        seen.clear()
        started = tune_runtime.submit(
            case_id=CASE["id"], case_def=CASE, agent_type=AGENT,
            options={"iterations": 1, "autoPersist": True},
            target="system_prompt", candidate_count=1)
        done = _wait(str(started["taskId"]))
        receipt = done.get("receipt") or {}
        with db.engine.begin() as conn:
            stored = conn.execute(text(
                "SELECT system_prompt FROM agent_configs WHERE agent_type = :a"),
                {"a": AGENT}).scalar()
            marks = conn.execute(text(
                "SELECT COUNT(*) FROM agent_optimizations WHERE persisted = 1")
            ).scalar()
        check("⑫ `autoPersist=true` 且严格更高 ⇒ 真写进 agent_configs ✓"
              "（本格与另三格的根本差别：它们只记优化史）",
              (receipt.get("persisted") or {}).get("version") == 1
              and stored == "候选 v1", (receipt.get("persisted"), stored))
        check("⑫ 真写之后优化史那一行要标 `persisted_at` ✓"
              "（判决 ≠ 生效；只写配置不标史，页面会显示「采纳了但没生效」）",
              marks == 1, marks)

        # ---- 第三轮：iterations=0 ⇒ 只评 Reference（合法值，用来验通路）----
        _reset_db()
        _seed_config()
        tune_runtime.reset()
        seen.clear()
        started = tune_runtime.submit(
            case_id=CASE["id"], case_def=CASE, agent_type=AGENT,
            options={"iterations": 0, "autoPersist": False},
            target="system_prompt", candidate_count=0)
        done = _wait(str(started["taskId"]))
        receipt = done.get("receipt") or {}
        check("⑫ `iterations=0` 只跑 1 跑 ✓（顺手反验分母没被写成固定 4：那样这里会卡在半程）",
              done.get("status") == "succeeded" and done.get("totalRuns") == 1
              and done.get("doneRuns") == 1 and len(seen) == 1,
              (done.get("status"), done.get("doneRuns"), done.get("totalRuns"), len(seen)))
        check("⑫ 0 轮时没有候选、也不落库 ✓（`best.version=0` ⇒ `persisted` 必须为 None）",
              receipt.get("iterations") == [] and receipt.get("persisted") is None,
              (receipt.get("iterations"), receipt.get("persisted")))
    finally:
        opt.evaluate_case, opt.generate_candidate_prompt, opt.benchmarks_dir = saved  # type: ignore[assignment]


# ===================================================================== ⑪
def case_http_prompt() -> None:
    """第四格的 HTTP 面：提交的参数口径与回执形状（submit 打桩，不真起后台跑模型）。"""
    from fastapi.testclient import TestClient

    from app.main import app

    tune_runtime.reset()
    client = TestClient(app)
    path = "/api/v1/evaluation/tune-prompt/sb-case-001"

    check("⑪ HTTP: 未知 case ⇒ 404（与另三格同口径）",
          client.post("/api/v1/evaluation/tune-prompt/nope-999", json={}).status_code == 404)

    captured: dict = {}
    original_submit = tune_runtime.submit

    def fake_submit(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {"taskId": "tune-stub", "target": kwargs.get("target"), "status": "queued"}

    tune_runtime.submit = fake_submit  # type: ignore[assignment]
    try:
        # 非法值**不认**、回默认 3（与 `/optimize` 同口径，不是 400）：那条同步端点的不认行为
        # 已由 `optimizer_test` 端点组守着，这里只核入队口没抄错。
        resp = client.post(path, json={"iterations": "三轮"})
        check("⑪ HTTP: 非整数 iterations 被忽略而不是当 0 ✓（当 0 ＝一次空调优，静默无结论）",
              resp.status_code == 200 and captured.get("options", {}).get("iterations", 3) == 3,
              (resp.status_code, captured.get("options")))
        check("⑪ HTTP: 分母按 1 + 默认 3 = 4 报 ✓（不是 repeat × (1+n) 的 6）",
              (resp.json().get("data") or {}).get("preview", {}).get("runs") == 4,
              (resp.json().get("data") or {}).get("preview"))
        captured.clear()

        resp = client.post(path, json={"iterations": 0})
        data = resp.json().get("data") or {}
        preview = data.get("preview") or {}
        check("⑪ HTTP: `iterations=0` **是合法值** ✓（只评一次 Reference，用来验通路）",
              resp.status_code == 200 and captured.get("options", {}).get("iterations") == 0,
              (resp.status_code, captured.get("options")))
        check("⑪ HTTP: 0 轮时 `runs=1` ✓（Reference 那 1 跑也要钱，不许报 0 跑）",
              preview.get("runs") == 1, preview)
        captured.clear()

        resp = client.post(path, json={"iterations": 2})
        data = resp.json().get("data") or {}
        preview = data.get("preview") or {}
        check("⑪ HTTP: submit 收到 `target=system_prompt` ✓（写 `prompt` 会在落库那侧多开一格）",
              captured.get("target") == "system_prompt", str(captured.get("target")))
        check("⑪ HTTP: `candidate_count` 就是 iterations ✓（本格候选数此刻已知，不必等基线）",
              captured.get("candidate_count") == 2, str(captured.get("candidate_count")))
        check("⑪ HTTP: `autoPersist` **默认 True** ✓（四格里唯一真写 agent_configs 的一格）",
              captured.get("options", {}).get("autoPersist") is True,
              str(captured.get("options")))
        check("⑪ HTTP: 回执 `repeat=1` ＋ `jumpKind=\"none\"` ✓"
              "（写 None 会被读成「这格没有 repeat」；写 0 会被读成「抖动带算出来是 0」）",
              preview.get("repeat") == 1 and preview.get("jumpKind") == "none", preview)
        captured.clear()

        client.post(path, json={"iterations": 1, "auto_persist": False})
        check("⑪ HTTP: 显式 `auto_persist=false` 只出结论、不写配置 ✓",
              captured.get("options", {}).get("autoPersist") is False,
              str(captured.get("options")))
    finally:
        tune_runtime.submit = original_submit  # type: ignore[assignment]


# ===================================================================== ⑩
def case_prompt_target() -> None:
    """⑩ 第四格：``system_prompt``（提示词优化）接进同一张任务表。

    这一格的坑全在「照抄姊妹格」上：

    * **分母**：本格的 ``repeat`` 是「Reference 复测几次」（候选每轮只评一次）⇒
      ``repeat + iterations``。照抄温度格的 ``repeat × (1 + n)`` 会报出 2 倍分母，
      任务 done 了页面还停在半程；
    * **配置**：本格 ``autoPersist`` 默认 True ⇒ **会真写** ``agent_configs.system_prompt``，
      别照工具集那格理解成「只记历史」；
    * **抖动带**：可选。``repeat=1``（默认）⇒ 没测（``jumpKind="none"``），文案不许印
      「抖动带 None」——那会让人以为是 0；``repeat≥2`` ⇒ 印出实测极差（``jumpKind="within"``）。
    """
    from app.services import optimization_log

    # 两处闭集必须同一套叫法：一边 `system_prompt`、一边 `prompt` 会让落库出现两格
    check("⑩ 任务表认的目标 == 落库的目标（同一套叫法，不另起 `prompt` 这种别名）",
          set(tune_runtime.TARGETS) == set(optimization_log.TARGETS),
          (tune_runtime.TARGETS, tuple(optimization_log.TARGETS)))

    check("⑩ `_runner_for('system_prompt')` 指向 `_prompt_runner` ✓",
          getattr(tune_runtime, "_runner_for")("system_prompt")
          is getattr(tune_runtime, "_prompt_runner"), "指错了")
    try:
        tune_runtime._runner_for("prompt")
        check("⑩ 别名 `prompt` 当场拒 ✓（两条叫法并存 ⇒ 落库会多出一格）", False, "没抛")
    except ValueError:
        check("⑩ 别名 `prompt` 当场拒 ✓（两条叫法并存 ⇒ 落库会多出一格）", True)

    stage = tune_runtime._stage_text
    start = stage({"type": "start", "iterations": 3, "totalRuns": 4, "doneRuns": 0},
                  "system_prompt")
    check("⑩ 开场说清「1 跑 Reference + N 轮改写」（不是「每档 N 次」）",
          "共 4 跑" in start and "3 轮" in start and "每档" not in start, start)

    ref = stage({"type": "run", "phase": "reference", "run": 1, "repeat": 1,
                 "doneRuns": 0, "totalRuns": 4}, "system_prompt")
    cand = stage({"type": "run", "phase": "candidate", "version": 2, "run": 2, "repeat": 1,
                  "doneRuns": 2, "totalRuns": 4}, "system_prompt")
    check("⑩ 进度不印「第 1/1 跑」（本格没有 repeat，那读起来像还有重复档没跑完）",
          "Reference" in ref and "1/1" not in ref and "第 2 轮候选" in cand, (ref, cand))

    base = stage({"type": "baseline", "score": 78.3, "jump": 0, "jumpKind": "none",
                  "doneRuns": 1, "totalRuns": 4}, "system_prompt")
    check("⑩ `jumpKind=none` 时不印「抖动带 None / 抖动带 0」，而说**没测**抖动带 ✓"
          "（「没测」与「测出来是 0」不是一回事）",
          "抖动带 None" not in base and "抖动带 0" not in base
          and "没测抖动带" in base, base)

    base_noise = stage({"type": "baseline", "score": 76.0, "samples": [80, 72, 76], "repeat": 3,
                        "jump": 8, "jumpKind": "within",
                        "doneRuns": 3, "totalRuns": 4}, "system_prompt")
    check("⑩ `jumpKind=within` 时印出**实测极差**和样本数 ✓（repeat=3 ⇒ 抖动带 8）",
          "抖动带 8" in base_noise and "3 次" in base_noise, base_noise)

    base_cfg = stage({"type": "baseline", "score": 78.3, "jump": None, "jumpKind": "none",
                      "baselineSource": "config", "doneRuns": 1, "totalRuns": 4},
                     "system_prompt")
    base_dft = stage({"type": "baseline", "score": 78.3, "jump": None, "jumpKind": "none",
                      "baselineSource": "default", "doneRuns": 1, "totalRuns": 4},
                     "system_prompt")
    check("⑩ `baseline` 那句说清这分数是**哪份提示词**测出来的 ✓（基线改成 DB 优先后，"
          "不写会让人以为测的总是出厂默认；两者差 8 分就足以改变采纳结论）",
          "当前生效的提示词" in base_cfg and "出厂默认" in base_dft
          and "当前生效的提示词" not in base_dft,
          (base_cfg, base_dft))

    rec = stage({"type": "candidate",
                 "record": {"version": 2, "score": 80.0, "accepted": False, "reason": "no_gain",
                            "reasonLabel": "分数没涨，保持原版"}}, "system_prompt")
    check("⑩ 候选那句是「v2：<理由>」（不印「温度 None」）",
          rec.startswith("v2：") and "None" not in rec, rec)

    # 入队快照里的分母：提示词格 repeat 缺省 ⇒ 必须按 1 兜底（按 0 会永远显示 0/0）
    async def fake(conn, job):  # noqa: ANN001 —— 只回显，真跑要 GPU
        return {"echo": dict(job.get("options") or {})}

    started = tune_runtime.submit(
        case_id="prompt-basic", case_def={"id": "prompt-basic", "kind": "prompt"},
        agent_type=AGENT, options={"iterations": 3, "autoPersist": True},
        target="system_prompt", candidate_count=3, runner=fake,
    )
    check("⑩ 入队分母 = repeat(兜底 1) + iterations ✓（按 0 算会显示 0/0）",
          started.get("totalRuns") == 4, started.get("totalRuns"))

    # 提示词格的 repeat 通式算出来是错的（它只加在 Reference 上）⇒ 调用方传 total_runs 一次算准
    # ⚠️ 必须等上一条**跑完**再入队：同 (agent, case, target) 会被去重，返回的是上一条的快照
    #    （第一版就是踩了这个：拿到 totalRuns=4、repeat=None，读起来像参数没生效）
    _wait(str(started["taskId"]), timeout=10)
    explicit = tune_runtime.submit(
        case_id="prompt-basic", case_def={"id": "prompt-basic", "kind": "prompt"},
        agent_type=AGENT, options={"iterations": 3, "repeat": 3, "autoPersist": True},
        target="system_prompt", candidate_count=3, total_runs=6, runner=fake,
    )
    check("⑩ 显式 `total_runs` 优先于通式 ✓（通式 3×(1+3)=12，真值 3+3=6；"
          "别在 submit 里按 target 分叉算术）",
          explicit.get("totalRuns") == 6 and explicit.get("repeat") == 3
          and not explicit.get("deduped"),
          (explicit.get("totalRuns"), explicit.get("repeat"), explicit.get("deduped")))
    _wait(str(explicit["taskId"]), timeout=10)
    # 快照是**摊平**的（`repeat`/`iterations` 都是顶层键，没有 `options` 子字典）
    check("⑩ submit 的白名单不漏 `iterations` ✓（漏了 ⇒ 永远按默认 3 轮跑，传 5 也没人理）",
          started.get("iterations") == 3, started.get("iterations"))


# ===================================================================== main
def main() -> int:
    for fn in (case_async_submit, case_progress_and_failure, case_wait, case_serial, case_dedup,
               case_cancel, case_real_optimizer_wiring, case_wiring, case_prompt_target,
               case_http, case_http_prompt, case_prompt_end_to_end):
        tune_runtime.reset()
        try:
            fn()
        except Exception as err:  # noqa: BLE001
            _RESULTS.append((f"{fn.__name__} 抛异常", False, f"{err.__class__.__name__}: {err}"))

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        mark = "✓" if ok else "✗"
        if not ok:
            print(f"FAIL  {mark} {name} ← {detail!r}")
    # ⚠️ 收尾行必须是 `run_all` 认的口径（`structure_test` ㊈ 会逐套核）：写 `SUMMARY 36/36`
    # 这种自创格式，总表里这一项的**项数恒为 0** —— 看着绿、其实是没被统计进去 ✓✗
    print(f"SUMMARY: {passed}/{len(_RESULTS)} passed")
    return 0 if passed == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
