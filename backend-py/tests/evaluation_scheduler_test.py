"""S6 自检：评测→优化 定时调度器 + 最后两个端点（``services/evaluation-scheduler.ts`` 126 行）。

这层是「**无人值守**」的自动化：默认**关闭**（避免意外烧钱），开启后每日 HH:MM 串行跑全量。
四件事必须钉住：

1. **防重入**：上一轮没跑完就跳过本次触发（返回上一轮结果，不重跑）；
2. **失败隔离**：单个 case 挂了不中断其余；
3. **串行**：一个 case 一次、顺序遍历（并发会引发限流 / 费用激增）；
4. **`running` 没有 try/finally 兜底**（原 TS 的真实缺陷）：循环**外**出错时 `running`
   会永远停在 true -> 之后所有触发都被防重入吃掉。! 这条是**保真**，别当 bug 顺手改。

运行::

    ./.venv/Scripts/python.exe tests/evaluation_scheduler_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="sched_test_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.core.response import now as response_now  # noqa: E402
from app.routers import evaluation as ev_route  # noqa: E402
from app.agent import evaluation_scheduler as sch  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _reset_state(**overrides) -> None:
    sch._state.update({  # noqa: SLF001 —— 测试直接改状态
        "running": False, "lastRunAt": None, "nextRunAt": None, "lastResults": [],
    })
    sch._state.update(overrides)  # noqa: SLF001


def main() -> int:  # noqa: C901
    # ================= 延时计算（本地时间，setHours 语义）=================
    zero = sch.next_run_delay(0, 0)
    check("延时: 今天 00:00 已过 -> 顺延次日（0 ~ 1440 分钟之间）",
          0 < zero <= 24 * 60 * 60 * 1000, zero)
    current = datetime.now()
    two_hours_later = (current.hour + 2) % 24
    later = sch.next_run_delay(two_hours_later, current.minute)
    check("延时: 今天稍后的时刻 -> 约 2 小时（容忍分钟级误差）",
          110 * 60 * 1000 <= later <= 130 * 60 * 1000, later)
    same_minute = sch.next_run_delay(current.hour, current.minute)
    check("延时: **秒/微秒被归零** -> 当前这一分钟也算「已过」→ 顺延次日（≈23.98h）",
          same_minute > (24 * 60 - 2) * 60 * 1000, same_minute)
    check("延时: 返回的是**毫秒**（float/int 皆可）",
          isinstance(sch.next_run_delay(3, 0), (int, float)))

    check("时间格式: `Date.now()+delay` 的 ISO 串与 `response.now()` 同形",
          sch._iso_in(0).endswith("Z") and "." in sch._iso_in(0)  # noqa: SLF001
          and len(sch._iso_in(0)) == len(response_now()))  # noqa: SLF001

    # ================= 初始状态 =================
    state = sch.get_scheduler_state()
    check("状态: 六个键 + camelCase（runOnStartup）",
          set(state) == {"enabled", "running", "lastRunAt", "nextRunAt", "lastResults", "schedule"}
          and set(state["schedule"]) == {"hour", "minute", "iterations", "runOnStartup"},
          sorted(state))
    check("状态: **默认关闭**（避免意外烧钱）+ 未运行 + 无历史结果",
          state["enabled"] is False and state["running"] is False
          and state["lastRunAt"] is None and state["lastResults"] == [],
          {k: state[k] for k in ("enabled", "running", "lastRunAt")})

    # ================= run_evaluation_cycle =================
    calls: list[dict] = []

    async def _stub_optimize(conn, agent_type, case_def, options):  # noqa: ANN001
        calls.append({"agentType": agent_type, "caseId": case_def.get("id"), "options": options})
        if case_def.get("id") == "ext-case-001":
            raise RuntimeError("这个 case 炸了")
        return {"best": {"version": 1, "score": 60, "prompt": "p"},
                "persisted": {"version": 1, "score": 60, "name": "n"}
                if agent_type == "storyboard_breaker" else None}

    sch.optimize_agent_prompt = _stub_optimize  # type: ignore[assignment]
    _reset_state()
    results = asyncio.run(sch.run_evaluation_cycle())
    # ⚠️ 期望值**从目录取**，不写死 ✗：原先硬编码 4 个 id，2026-11-07 目录加进第 5 类
    # `grid-prompt-001` 之后这条恒红 —— 而"恒红"会被当成环境噪音忽略，真出问题（少跑一个 case、
    # 并发跑、顺序乱）时也没人看得见了 ✗✗。
    catalog_ids = [str(item.get("id")) for item in sch.list_benchmark_cases()]
    check("跑一轮: 基准目录里**每个 case 都跑**（顺序＝目录顺序、串行；目录加档这里自动跟上 ✓）",
          [c["caseId"] for c in calls] == catalog_ids and len(catalog_ids) >= 4,
          ([c["caseId"] for c in calls], catalog_ids))
    check("跑一轮: iterations 取自配置、autoPersist 固定 True（自动落库）",
          all(c["options"] == {"iterations": 3, "autoPersist": True} for c in calls),
          calls[0]["options"])
    by_id = {str(r.get("caseId")): r for r in results}
    crashed = by_id.get("ext-case-001") or {}
    check("跑一轮: **失败隔离** —— 挂掉的 case ok=false + error，其余照常 ok=true"
          "（按 caseId 取，不按下标 ✗：目录顺序一变下标就对不上）",
          crashed.get("ok") is False and crashed.get("error") == "这个 case 炸了"
          and crashed.get("agentType") == "extractor"
          and all(r.get("ok") is True for r in results if r.get("caseId") != "ext-case-001"),
          results)
    check("跑一轮: 结果四键齐全（caseId/agentType/ok/persisted）",
          all(set(r) >= {"caseId", "agentType", "ok"} for r in results)
          and (by_id.get("sb-case-001") or {}).get("persisted")
          == {"version": 1, "score": 60, "name": "n"},
          by_id.get("sb-case-001"))
    check("跑一轮: 结束后 running 复位、lastRunAt/lastResults 写回",
          sch._state["running"] is False and sch._state["lastRunAt"] is not None  # noqa: SLF001
          and sch._state["lastResults"] == results)  # noqa: SLF001

    # 防重入
    calls.clear()
    _reset_state(running=True, lastResults=[{"caseId": "上一轮"}])
    blocked = asyncio.run(sch.run_evaluation_cycle())
    check("防重入: 上一轮仍在跑 -> 直接返回上一轮结果、**不重跑**",
          calls == [] and blocked == [{"caseId": "上一轮"}], (calls, blocked))
    check("防重入: running 保持 true（等上一轮自己复位）", sch._state["running"] is True)  # noqa: SLF001

    # ! 保真：循环**外**出错时 running 不会复位（原 TS 缺陷，别顺手修）
    _reset_state()
    original_catalog = sch.list_benchmark_cases
    sch.list_benchmark_cases = lambda: (_ for _ in ()).throw(ValueError("基准目录坏 JSON"))  # type: ignore[assignment]
    stuck = ""
    try:
        asyncio.run(sch.run_evaluation_cycle())
    except Exception as err:  # noqa: BLE001
        stuck = str(err)
    sch.list_benchmark_cases = original_catalog  # type: ignore[assignment]
    check("保真: 循环外抛错 -> running **卡在 true**（原 TS 没有 try/finally，别当 bug 修）",
          stuck == "基准目录坏 JSON" and sch._state["running"] is True,  # noqa: SLF001
          (stuck, sch._state["running"]))  # noqa: SLF001

    # ================= arm / start =================
    _reset_state()  # running 复位
    sch._state["enabled"] = False  # noqa: SLF001
    sch._timer = None  # noqa: SLF001
    sch.arm()
    check("定时: 未启用 -> arm() 不挂定时器、不写 nextRunAt",
          sch._timer is None and sch._state["nextRunAt"] is None)  # noqa: SLF001

    async def _arm_enabled() -> None:
        sch._state["enabled"] = True  # noqa: SLF001
        sch.arm()
        await asyncio.sleep(0)

    asyncio.run(_arm_enabled())
    check("定时: 启用后 arm() 写入 nextRunAt（ISO 串）并挂上定时器",
          isinstance(sch._state["nextRunAt"], str)  # noqa: SLF001
          and sch._state["nextRunAt"].endswith("Z")  # noqa: SLF001
          and sch._timer is not None,  # noqa: SLF001
          sch._state["nextRunAt"])  # noqa: SLF001
    if sch._timer is not None:  # noqa: SLF001
        sch._timer.cancel()  # noqa: SLF001
    sch._timer = None  # noqa: SLF001
    sch._state["enabled"] = False  # noqa: SLF001

    sch.start_evaluation_scheduler()
    check("启动: 未启用时 start() 直接返回（不挂定时器）", sch._timer is None)  # noqa: SLF001

    # ================= 端点 =================
    client = TestClient(app)
    sched = client.get("/api/v1/evaluation/scheduler")
    check("端点: GET /scheduler -> 200 成功信封 + camelCase 状态",
          sched.status_code == 200
          and set(sched.json()["data"]) == {"enabled", "running", "lastRunAt", "nextRunAt",
                                            "lastResults", "schedule"}
          and "runOnStartup" in sched.json()["data"]["schedule"],
          sched.json().get("data"))

    async def _stub_cycle() -> list[dict]:
        return [{"caseId": "x", "agentType": "extractor", "ok": True, "persisted": None}]

    ev_route.run_evaluation_cycle = _stub_cycle  # type: ignore[assignment]
    run_resp = client.post("/api/v1/evaluation/run")
    check("端点: POST /run -> 200 + results 原样（与定时**共用同一入口**）",
          run_resp.status_code == 200
          and run_resp.json()["data"] == [{"caseId": "x", "agentType": "extractor",
                                           "ok": True, "persisted": None}],
          run_resp.json().get("data"))

    async def _boom() -> list[dict]:
        raise RuntimeError("全量优化炸了")

    ev_route.run_evaluation_cycle = _boom  # type: ignore[assignment]
    failed = client.post("/api/v1/evaluation/run")
    check("端点: POST /run 异常 -> 400 且 message 取 str(err)",
          failed.status_code == 400 and failed.json()["message"] == "全量优化炸了",
          failed.json())

    check("端点: 评测域**已无未迁端点**（5/5：cases/evaluate/optimize/scheduler/run 全注册）",
          all(getattr(client, method)(path).status_code != 501 for method, path in (
              ("get", "/api/v1/evaluation/cases"),
              ("post", "/api/v1/evaluation/evaluate/x"),
              ("post", "/api/v1/evaluation/optimize/x"),
              ("get", "/api/v1/evaluation/scheduler"),
              ("post", "/api/v1/evaluation/run"),
          )))

    # ================= 汇总 =================
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
