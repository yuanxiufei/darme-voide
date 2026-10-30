r"""自研引擎 · **队列背压与任务历史上限**（E3.1）。

`submit` 是「立即返回」的：不设上限时，前端连点 N 次就等于排 N 个 —— 每个 task 都留一份
`request`（带参考图是十 MB 级），单条视频又要分钟级。所以这里验三件事：

1. **排不下就当场拒**（`EngineBusy`），且被拒的请求**不留任何 task 记录**（否则连点仍然撑内存）；
2. **拒了不误伤**：原有队列长度不变、不悄悄丢掉已有请求；
3. **任务历史有上限**：只裁**已终结**的，在跑/在排的一个都不动（裁掉会让 `cancel` 静默返回 None）。

运行::

    ./.venv/Scripts/python.exe tests/engine_queue_backpressure_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from app.services.engine import facade  # noqa: E402
from app.services.engine.runtime import EngineBusy, EngineRuntime, EngineTask, _env_int  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, ok: bool, detail: object = None) -> None:
    _RESULTS.append((name, bool(ok), detail))


def _frozen_runtime(max_queue: int, max_tasks: int | None = None) -> EngineRuntime:
    """造一个**不会被消费**的运行时：把 worker 启动换成 no-op ⇒ 队列原样留着好数。

    ⚠️ 必须连 `_ensure_worker` 一起停 ✗：worker 一起来就会 `pop(0)` 去跑，队列长度随测试机快慢
    抖动 ✓✗，测出来的数字没法判 ✓。
    """
    rt = EngineRuntime()
    rt._maxQueue = max_queue
    rt._maxTasks = max_tasks if max_tasks is not None else max(max_queue * 4, max_queue)
    rt._ensure_worker = lambda: None  # type: ignore[method-assign]
    return rt


def case_reject() -> None:
    rt = _frozen_runtime(3)
    for index in range(3):
        rt.submit({"prompt": f"ok-{index}"})
    check("① 上限内照常入队（3/3 全部收下）",
          len(rt._queue) == 3 and len(rt._tasks) == 3, (len(rt._queue), len(rt._tasks)))

    raised: BaseException | None = None
    try:
        rt.submit({"prompt": "过了"})
    except BaseException as err:  # noqa: BLE001
        raised = err
    check("② 超限抛 `EngineBusy`（不是静默排队）", isinstance(raised, EngineBusy),
          f"{type(raised).__name__}: {raised}")

    check("③ 被拒的请求**没有**留下 task 记录（拒绝发生在入表之前）",
          len(rt._tasks) == 3 and len(rt._order) == 3, (len(rt._tasks), len(rt._order)))
    check("④ 被拒后原有队列**一条不少**（没有顺手丢别人的）",
          len(rt._queue) == 3, len(rt._queue))

    #: 空出位置后应能继续收
    rt._queue.pop(0)
    rt.submit({"prompt": "腾出位置了"})
    check("⑤ 队列腾出位置后可以继续入队（拒绝不是永久的）",
          len(rt._queue) == 3, len(rt._queue))


def case_limits_visible() -> None:
    rt = _frozen_runtime(8, 40)
    rt.submit({"prompt": "a"})
    status = rt.status()
    check("⑥ `status()` 报出限额（只报「排了几个」没法判余量）",
          status.get("queueLimit") == 8 and status.get("taskHistoryLimit") == 40,
          {"queueLimit": status.get("queueLimit"), "taskHistoryLimit": status.get("taskHistoryLimit")})
    check("⑦ `status()` 仍报实际排队数", status.get("queued") == 1, status.get("queued"))


def case_prune() -> None:
    rt = _frozen_runtime(100, max_tasks=5)
    #: 手工灌 4 个**已终结**的老任务（不经过 submit，免得 worker/队列干扰）
    for index in range(4):
        task = EngineTask(id=f"old-{index}", stage="h3", request={}, assembly={}, createdAt=index)
        task.status = "done"
        rt._tasks[task.id] = task
        rt._order.append(task.id)
    #: 再灌 1 个**还在排队**的（绝不能被裁）
    keep_queued = EngineTask(id="queued-1", stage="h3", request={}, assembly={}, createdAt=99)
    rt._tasks[keep_queued.id] = keep_queued
    rt._order.append(keep_queued.id)
    rt._queue.append(keep_queued.id)

    rt.submit({"prompt": "触发裁剪"})
    check("⑧ 超过历史上限时裁掉**最老的已终结**任务（`_order` 与 `_tasks` 同步缩）",
          len(rt._order) <= 5 and len(rt._tasks) == len(rt._order),
          {"order": len(rt._order), "tasks": len(rt._tasks)})
    check("⑨ 还在**排队**的任务没被裁（裁掉会让 cancel 静默返回 None）",
          "queued-1" in rt._tasks and "queued-1" in rt._order, sorted(rt._tasks))
    check("⑩ 最新的任务一定还在（裁的是老的一头）",
          rt._order[-1] in rt._tasks, rt._order)


def case_env_and_translation() -> None:
    check("⑪ 环境变量可调上限", _env_int("DARME_TEST_LIMIT", 7) == 7)
    os.environ["DARME_TEST_LIMIT"] = "2"
    try:
        check("⑫ 环境变量生效", _env_int("DARME_TEST_LIMIT", 7) == 2, _env_int("DARME_TEST_LIMIT", 7))
        os.environ["DARME_TEST_LIMIT"] = "abc"
        check("⑬ 读不懂就回落默认（不为一个限额把引擎起不来）",
              _env_int("DARME_TEST_LIMIT", 7) == 7, _env_int("DARME_TEST_LIMIT", 7))
        os.environ["DARME_TEST_LIMIT"] = "0"
        check("⑭ 小于下限也回落默认", _env_int("DARME_TEST_LIMIT", 7) == 7)
    finally:
        os.environ.pop("DARME_TEST_LIMIT", None)

    #: 门面必须把「忙/满」翻成**机器可判**的码，而不是一句中文
    translated = facade._friendly(EngineBusy("队列已满"))
    check("⑮ 门面把 `EngineBusy` 翻成 `E_ENGINE_BUSY`（前端按码分流）",
          translated.code == "E_ENGINE_BUSY" and translated.code in facade.CODES, translated.code)

    #: ⚠️ 显存不够**单独成码** ✗✗ —— 老行为是落到 fallback ``E_TASK_FAILED``
    #:    「引擎内部异常」✓✗（用户照着"内部异常"去查实现，而真相是这张卡装不下 ✓）。
    OomError = type("OutOfMemoryError", (Exception,), {})
    vram = facade._friendly(OomError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    check("⑮′ ⭐ ``OutOfMemoryError`` ⇒ `E_VRAM_EXHAUSTED`（**不许**落到「引擎内部异常」✗✗），"
          "且 hint 点名 `int8Offload`（本仓唯一知道自己答案的那条 ✓）",
          vram.code == "E_VRAM_EXHAUSTED" and vram.code in facade.CODES
          and "int8Offload" in vram.hint, (vram.code, vram.hint[:80]))
    check("⑮″ ⚠️ 其它异常**不许**被当成显存问题 ✗（否则真因会被这三条建议盖住 ✓✗）",
          facade._friendly(RuntimeError("其它炸了")).code != "E_VRAM_EXHAUSTED")


def case_router_wiring() -> None:
    """HTTP 侧必须**接住**队满（不然背压变成 500）。

    ⚠️ 这里扫 AST 而不是起 TestClient ✗：起 app 会连带初始化数据库/日志 ✓✗，为一个「有没有
    except」的判据付这个代价不值 ✓。AST 只看两件事：`generate` 里 `submit` 是否被 try 包住、
    那个 try 是否接 `EngineBusy` ✓。
    """
    import ast

    source = (Path(__file__).resolve().parents[1] / "app" / "routers" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    generate = next((n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "generate"), None)
    check("⑯ `routers/engine.py::generate` 存在", generate is not None)

    guarded = False
    if generate is not None:
        for node in ast.walk(generate):
            if not isinstance(node, ast.Try):
                continue
            calls_submit = any(isinstance(inner, ast.Call)
                               and isinstance(inner.func, ast.Attribute)
                               and inner.func.attr == "submit"
                               for stmt in node.body for inner in ast.walk(stmt))
            catches_busy = any(
                isinstance(handler.type, ast.Attribute) and handler.type.attr == "EngineBusy"
                for handler in node.handlers if handler.type is not None)
            guarded = guarded or (calls_submit and catches_busy)
    check("⑰ 队满在 HTTP 层被接住并转 400（否则背压表现为 500）", guarded)

    #: 翻码必须走门面，不许在 router 里手写 "E_ENGINE_BUSY"（E1.5 单一真源）
    check("⑱ router 用 `facade.translate` 翻码，而不是手写错误码字面量",
          "facade.translate(" in source and '"E_ENGINE_BUSY"' not in source,
          {"translate": "facade.translate(" in source, "literal": '"E_ENGINE_BUSY"' in source})
    check("⑲ `facade.translate` 与 `_friendly` 同一套规则",
          facade.translate(EngineBusy("满")).code == facade._friendly(EngineBusy("满")).code == "E_ENGINE_BUSY")


def case_generate_returns_400() -> None:
    """真调一次路由函数：队满 ⇒ **400 + engineCode**，不是 500。

    ⚠️ 为什么能直接 await 路由 ✗：`read_json` 只要 ``await request.json()`` ✓，一个带 async
    ``json()`` 的替身就够 ✓ —— 不必起 FastAPI/TestClient ✓（起 app 会连带初始化数据库 ✓✗）。
    """
    import asyncio
    import json

    from app.routers import engine as router_mod

    class _FakeRequest:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        async def json(self) -> dict:
            return self._payload

    original = router_mod.engine_runtime.submit

    def _boom(*_args, **_kwargs):
        raise EngineBusy("引擎队列已满（排队 1 / 上限 1）⇒ 这次请求**没有入队、也没有开始跑**")

    router_mod.engine_runtime.submit = _boom  # type: ignore[method-assign]
    try:
        response = asyncio.run(router_mod.generate(_FakeRequest(
            {"dryRun": True, "prompt": "probe", "outputsDir": str(Path.cwd() / "tmp")})))
    finally:
        router_mod.engine_runtime.submit = original  # type: ignore[method-assign]

    body = json.loads(bytes(response.body).decode("utf-8"))
    check("⑳ 队满时 `/generate` 返回 400（不是 500）", response.status_code == 400, response.status_code)
    check("㉑ 400 里带机器可判的 `engineCode`", body.get("engineCode") == "E_ENGINE_BUSY", body)
    check("㉒ 文案点明「没有入队」（用户据此决定重试而不是查日志）",
          "没有入队" in str(body.get("message") or ""), body.get("message"))


def main() -> int:
    case_reject()
    case_limits_visible()
    case_prune()
    case_env_and_translation()
    case_router_wiring()
    case_generate_returns_400()

    failed = 0
    for name, ok, detail in _RESULTS:
        print(("OK   " if ok else "FAIL ") + name)
        if not ok:
            failed += 1
            print(f"     ↳ {detail}")
    print(f"\nSUMMARY: {len(_RESULTS) - failed}/{len(_RESULTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
