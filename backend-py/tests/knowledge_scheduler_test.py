"""定时复盘守卫（2026-11-06 立）—— 知识蒸馏的每日定时任务。

为什么单开一套
==============
这一类代码的失效方式**全都不报错**：

* 定时器没挂上（模块写了但没人调 ``start``）⇒ 台账安静地一条不涨，看起来"最近没问题"；
* ``running`` 没在 ``finally`` 里复位（``evaluation_scheduler`` 就有这个缺陷）⇒ 一轮抛错后
  **永久停摆**，而回执依然写着 ``enabled=true``；
* 默认写成开 ⇒ 用户第一次启动就在无人值守状态下往台账写条目、甚至把旧经验降级；
* 无人值守还去调模型 ⇒ 夜里烧钱，第二天人只能批量点"通过"。

四类都不会在日志里留下"我坏了"的痕迹，所以必须有机器判据。十一组：

① 默认关（只有显式 ``true`` 才开）+ 关着时 ``start()`` 照实答原因、不挂空定时器；
② 一轮真跑：走 ``knowledge_distill``（AST 判据：不写第二份实现、不直连 sqlite）；
   空索引时**如实回 0 条**，不凭空造一条知识；
③ 防重入：上一轮未结束 ⇒ 跳过，且**不覆盖** ``lastResult``；
④ 失败隔离：蒸馏炸了 lift 仍跑；错误进 ``errors`` 且 ``ok=false``；
⑤ 复位：``running`` 必须在 ``finally`` 里复位（结构化判据 + 反验）；
⑥ ``lift`` 是独立开关（默认 off）；
⑦ 定时器：``arm`` 的下一轮时间是**未来**且是 ``toISOString`` 形（``…Z`` 带三位毫秒）；
   ``stop`` 之后 ``nextRunAt`` 清空（"不存在的下一轮"比未来时间更难排查）；
⑧ ``start`` 回执：enabled 时 ``started=true``，``runOnStartup`` 只跑一轮；
⑨ 回执不可被外部改坏：``state()`` 返回副本；
⑩ 接线：``GET /api/v1/knowledge-loop/scheduler`` 走统一信封；
⑪ 启动路径真调了它（文本判据：``main.py`` 的 lifespan 里有这一行）。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/knowledge_scheduler_test.py
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

#: cp936 控制台下打印中文会 ``UnicodeEncodeError``（``run_all`` 里不露头，单独跑必踩）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # 非常规流（已被重定向）时不强求
        pass

#: 与 ``knowledge_loop_test`` 同一条纪律：夹具库与创作者的真实库必须分开 ——
#: 这套会真写台账（`proposed` 条目），跑到真数据根上就是把测试的假经验堆进检索面。
TMP = Path(tempfile.mkdtemp(prefix="kb-sched-"))
os.environ["OUTCOME_INDEX_DB"] = (TMP / "outcome-index.sqlite").as_posix()
os.environ["KNOWLEDGE_LEDGER_DB"] = (TMP / "knowledge-ledger.sqlite").as_posix()

from app.core.config import knowledge as knowledge_config  # noqa: E402
from app.services import knowledge_distill, knowledge_ledger, knowledge_scheduler  # noqa: E402

#: 台账落点另有 ``KNOWLEDGE_LEDGER_DB`` 环境变量直覆盖（上面已设 ⇒ 不靠打补丁，
#: 也不许顺手打 ``get_data_root`` 补丁把"其实写的是真目录"这件事盖住）
assert knowledge_ledger.db_path().parent.as_posix().startswith(TMP.as_posix()), knowledge_ledger.db_path()

HERE = Path(__file__).resolve().parent
SCHED_SRC = (BP / "app" / "services" / "knowledge_scheduler.py").read_text(encoding="utf-8")
CONFIG_SRC = (BP / "app" / "core" / "config.py").read_text(encoding="utf-8")
MAIN_SRC = (BP / "app" / "main.py").read_text(encoding="utf-8")

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _reset_state(**overrides: object) -> None:
    """把调度器状态摆到确定的起点（不依赖本机 config.yaml 到底写了什么）。"""
    knowledge_scheduler._state.update({
        "enabled": False,
        "running": False,
        "lastRunAt": None,
        "nextRunAt": None,
        "lastResult": None,
    })
    knowledge_scheduler._state["schedule"].update({
        "hour": 4, "minute": 30, "limit": 3, "lift": False, "runOnStartup": False,
    })
    knowledge_scheduler._state.update(overrides)


def _finally_resets_running(source: str) -> bool:
    """``run_cycle`` 是否把 ``_state["running"] = False`` 放在某个 ``try/finally`` 里。

    只认**结构化**位置：写在 ``finally`` 之外（比如正常路径末尾）等于没写 —— 那正是
    ``evaluation_scheduler`` 的既有缺陷（抛错就永久卡住防重入）。
    """
    tree = ast.parse(source)
    target = next((node for node in tree.body
                   if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_cycle"), None)
    if target is None:
        return False
    for node in ast.walk(target):
        if not isinstance(node, ast.Try) or not node.finalbody:
            continue
        for inner in node.finalbody:
            for sub in ast.walk(inner):
                if not isinstance(sub, ast.Assign):
                    continue
                for tgt in sub.targets:
                    if (isinstance(tgt, ast.Subscript)
                            and isinstance(tgt.value, ast.Name)
                            and tgt.value.id == "_state"
                            and isinstance(tgt.slice, ast.Constant)
                            and tgt.slice.value == "running"):
                        return True
    return False


def main() -> int:
    # ================================================================ ① 默认关
    check("默认关: 配置里的开关只能由显式 `true` 打开（不许 `or True` 之类的默认开 ✗）",
          'os.environ.get("KNOWLEDGE_DISTILL_ENABLED") == "true"' in CONFIG_SRC
          and '_KNOWLEDGE_DISTILL.get("enabled") is True' in CONFIG_SRC,
          None)

    _reset_state()
    offline = knowledge_scheduler.start()
    check("默认关: 关着时 start() 照实答原因（不是静默返回 None ✗）",
          offline["started"] is False and "未启用" in offline["why"]
          and offline["scheduler"]["nextRunAt"] is None, offline)

    knowledge_scheduler.arm()
    check("默认关: 关着时不挂空定时器（挂着就会在半夜跑起来，而且回执还说没开 ✗）",
          knowledge_scheduler._timer is None, knowledge_scheduler.state())

    # ======================================================== ② 一轮真跑（空索引）
    before = knowledge_ledger.list_entries(limit=200).get("total")
    empty_run = asyncio.run(knowledge_scheduler.run_cycle())
    after = knowledge_ledger.list_entries(limit=200).get("total")
    check("一轮: 空索引 ⇒ 候选 0 / 写入 0，且**不凭空造一条知识**（台账条数不变）",
          empty_run["distill"]["candidates"] == 0 and empty_run["distill"]["written"] == 0
          and before == after, {"before": before, "after": after, "run": empty_run})

    #: AST 判据：真实现只能来自 ``knowledge_distill``（不许在调度器里抄一份统计/写库逻辑）
    sched_tree = ast.parse(SCHED_SRC)
    called = {node.func.attr for node in ast.walk(sched_tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name) and node.func.value.id == "knowledge_distill"}
    check("一轮: 蒸馏与 lift 都转调 `knowledge_distill`（同一套判据，不写第二份实现 ✓）",
          {"candidates", "distill"} <= called
          and ("refresh_lift" in called or "refresh_lift" in SCHED_SRC),
          sorted(called))
    check("一轮: 调度器不直连 sqlite（写库口径只有台账一处 ✓）",
          "import sqlite3" not in SCHED_SRC and "connect(" not in SCHED_SRC, None)

    # ================================================================ ③ 防重入
    _reset_state(running=True, lastResult={"ok": True, "marker": "上一轮"})
    calls: list[tuple[int, bool]] = []

    async def scenario() -> dict:
        return await knowledge_scheduler.run_cycle(
            distill_fn=lambda cap, dry: calls.append((cap, dry)) or {"candidates": 0})

    reentrant = asyncio.run(scenario())
    check("防重入: 上一轮未结束 ⇒ 跳过（skipped=running），且**不覆盖** lastResult ✓",
          reentrant["skipped"] == "running" and reentrant["ok"] is False
          and calls == [] and knowledge_scheduler._state["lastResult"] == {
              "ok": True, "marker": "上一轮"},
          {"run": reentrant, "calls": calls})

    # ============================================================ ④ 失败隔离
    _reset_state()
    lift_calls: list[bool] = []

    def _boom(cap: int, dry: bool) -> dict:
        raise RuntimeError("统计炸了")

    async def scenario_fail() -> dict:
        return await knowledge_scheduler.run_cycle(
            lift=True, distill_fn=_boom,
            lift_fn=lambda: lift_calls.append(True) or {"judged": 0, "deprecated": 0})

    failed = asyncio.run(scenario_fail())
    check("失败隔离: 蒸馏炸了 ⇒ lift 仍然跑（两步互不牵连 ✓）",
          lift_calls == [True] and failed["lift"] == {"judged": 0, "deprecated": 0}, failed)
    check("失败隔离: 错误如实进 errors 且 ok=false（不许把失败报成成功 ✗）",
          failed["ok"] is False and any("蒸馏失败" in item for item in failed["errors"]),
          failed["errors"])
    check("失败隔离: 一轮失败之后 running 必须已复位（否则下一天起永久停摆 ✗✗）",
          knowledge_scheduler._state["running"] is False, knowledge_scheduler.state())

    # ========================================================== ⑤ finally 判据
    check("复位: `run_cycle` 把 `_state[\"running\"] = False` 写在 `finally` 里 ✓", 
          _finally_resets_running(SCHED_SRC), None)
    #: 反验：把 ``finally`` 改成 ``except``（复位挪进异常分支，正常路径就没有复位了 ——
    #: 等价于 ``evaluation_scheduler`` 的既有写法）⇒ 判据必须报否。注意不能改成 ``if True:``
    #: 那会让源码直接 **SyntaxError**（``try`` 后必须跟 except/finally），反验自己先炸。
    doctored = SCHED_SRC.replace("    finally:\n", "    except Exception:\n", 1)
    check("复位: 反验 —— 复位挪出 finally ⇒ 判据必须报（这正是「静默永久停摆」的坏法 ✗）",
          doctored != SCHED_SRC and _finally_resets_running(doctored) is False, None)

    # ============================================================== ⑥ lift 开关
    _reset_state()
    quiet: list[bool] = []

    async def scenario_quiet() -> dict:
        return await knowledge_scheduler.run_cycle(
            distill_fn=lambda cap, dry: {"candidates": 0, "written": 0, "rejected": 0},
            lift_fn=lambda: quiet.append(True) or {"judged": 0, "deprecated": 0})

    no_lift = asyncio.run(scenario_quiet())
    check("lift: 默认不跑（它会把旧条目降级 —— 与「加一条待审」不是同一个力度 ✓）",
          no_lift["lift"] is None and quiet == [], no_lift)

    # ================================================================ ⑦ 定时器
    _reset_state()
    knowledge_scheduler._state["enabled"] = True

    async def scenario_arm() -> dict:
        knowledge_scheduler.arm()
        return knowledge_scheduler.state()

    armed = asyncio.run(scenario_arm())
    next_at = str(armed["nextRunAt"] or "")
    check("定时器: arm 之后有下一轮时间，且是 toISOString 形（…Z + 三位毫秒 ✓）",
          next_at.endswith("Z") and next_at[-5] == "." and armed["schedule"]["hour"] == 4,
          next_at)
    check("定时器: 下一轮时间必须**在未来**（回退到昨天 ⇒ 立即触发、连夜刷库 ✗）",
          next_at > "2026-11-06" and bool(knowledge_scheduler._timer), next_at)

    stopped = knowledge_scheduler.stop()
    check("定时器: stop 之后 nextRunAt 清空（留一个不存在的下一轮 ⇒ 排查时被它带偏 ✗）",
          stopped["nextRunAt"] is None and knowledge_scheduler._timer is None, stopped)

    # ============================================================== ⑧ start 回执
    _reset_state()
    knowledge_scheduler._state["enabled"] = True
    knowledge_scheduler._state["schedule"]["runOnStartup"] = True  # 注意在 schedule 里

    async def scenario_start() -> dict:
        receipt = knowledge_scheduler.start()
        await asyncio.sleep(0.05)  # 让 runOnStartup 那一轮跑完
        return receipt

    try:
        started = asyncio.run(scenario_start())
    finally:
        knowledge_scheduler.stop()
    check("start: enabled 时 started=true 且回执带调度器状态（启动路径也要能回答「跑没跑」✓）",
          started["started"] is True and started["scheduler"]["nextRunAt"] is not None, started)
    check("start: runOnStartup=true 时立刻跑过一轮（lastRunAt 落上、且只一轮 ✓）",
          knowledge_scheduler._state["lastRunAt"] is not None, knowledge_scheduler.state())

    # =========================================================== ⑨ 回执不可改坏
    knowledge_scheduler._state["enabled"] = False
    snapshot = knowledge_scheduler.state()
    snapshot["schedule"]["limit"] = 999
    snapshot["enabled"] = True
    fresh = knowledge_scheduler.state()
    check("回执: state() 是副本（外部改它不许污染调度器真状态 ✓）",
          fresh["schedule"]["limit"] != 999 and fresh["enabled"] is False, fresh)

    # ================================================================ ⑩ 端点接线
    _reset_state()  # 端点读的是活状态 ⇒ 先摆到「关」，判据不受本机 config.yaml 影响
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import app  # noqa: PLC0415

    client = TestClient(app)
    resp = client.get("/api/v1/knowledge-loop/scheduler")
    body = resp.json()
    data = body.get("data") or {}
    check("端点: GET /scheduler 走统一信封，键名 camelCase（前端直接读，不猜 snake_case ✓）",
          resp.status_code == 200 and body.get("code") == 200
          and {"enabled", "running", "lastRunAt", "nextRunAt", "lastResult", "schedule"} <= set(data),
          sorted(data))
    check("端点: 关着时如实回 enabled=false（不是 404、也不是假装在跑 ✓）",
          data.get("enabled") is False, data.get("enabled"))

    # ============================================================ ⑪ 启动路径接线
    check("接线: `main.py` 的 lifespan 真调了 `start_knowledge_scheduler()`（写了没人调＝死模块 ✗）",
          "start_knowledge_scheduler()" in MAIN_SRC
          and "from app.services.knowledge_scheduler import start as start_knowledge_scheduler"
          in MAIN_SRC, None)
    check("接线: 回执里的配置项与 config 同源（不另抄一份默认值 ✓）",
          set(knowledge_config["distill"]) >= {"enabled", "hour", "minute", "limit", "lift",
                                               "run_on_startup"},
          sorted(knowledge_config["distill"]))

    # ================================================================ 结果
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed_ok, detail in _RESULTS:
        print(f"{'PASS' if passed_ok else 'FAIL'}  {name}"
              + ("" if passed_ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
