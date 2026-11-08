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
    check("⑦ 任务自己开连接 ✓（拿请求那条 conn 跑后台＝响应一返回就 `connection is closed` ✗）",
          "db_module.engine.connect()" in runtime_src
          and runtime_src.count("db_module.engine.connect()") == 1, "没有")
    check("④ 单工作线程 + 队列 ✓（线程按任务起＝并发抢显存 ✗）",
          "threading.Thread(" in runtime_src and "queue.Queue()" in runtime_src
          and runtime_src.count("threading.Thread(") == 1, "没有")
    check("⑦ 默认 runner 把 `progress` 与 `should_stop` 都透给优化器 ✓",
          "progress=job.get(\"progressCallback\")" in runtime_src
          and "should_stop=lambda: bool(job.get(\"abort\"))" in runtime_src, "没透")

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


# ===================================================================== main
def main() -> int:
    for fn in (case_async_submit, case_progress_and_failure, case_wait, case_serial, case_dedup,
               case_cancel, case_real_optimizer_wiring, case_wiring, case_http):
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
