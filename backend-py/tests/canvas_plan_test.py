"""画布「计划层」守卫（意图 → 计划 → op，中间有一道人工确认）。

钉六件事：

1. **工具面按场景收敛，且坏配置会当场炸** —— 白名单点名了不存在的工具、或与工具表
   交集为空，都必须抛错。⚠️ 这两条是照着 `dramaclaw` 踩过的坑写的：那时 `tools/list`
   静默返回 0 个工具，模型平静地回「我做不到」，日志里一切正常 ✗。
2. **缺参当场拒，且一次报全** —— 不许给默认值糊过去，也不许改一条报一条
   （模型的步数根本不够试探 ✗）。
3. **坐标与 id 由编译器算** —— 同一份计划编译两次得到同一批 id；item 不写 x/y 时按
   `layout` 排，成环要拒（否则分层算宽度会无限递归 ✗）。
4. **`after` 可指向画布上已有节点** —— 「接在选中的这张图后面」是最常见的需求，
   只认计划内 ref 的话这类需求根本无从表达。
5. **草案不下发 op，确认才下发，重放不重复** —— 三件事各自独立：
   草案阶段拿到 op 就等于没有确认环节 ✗；重复确认建两套节点/重复起任务 ✗✗。
6. **前端同源** —— `canvasOps.ts` 必须认 `baseRevision` / `applyId`，
   `useCanvas.ts` 必须提供 `revision`；后端编出来、前端不认就是白做 ✗。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/canvas_plan_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _plan(**overrides) -> dict:
    plan = {
        "goal": "女侠在雨夜走进客栈",
        "items": [
            {"nodeType": "text", "ref": "p1", "text": "雨夜，客栈门口"},
            {"nodeType": "image", "ref": "p2", "after": ["p1"]},
            {
                "nodeType": "config",
                "ref": "p3",
                "workflowId": "wf-video",
                "after": ["p2"],
                "run": True,
                "mode": "video",
                "prompt": "推门而入，雨水顺着檐角落下",
            },
        ],
        "options": {"layout": "flow"},
    }
    plan.update(overrides)
    return plan


def _errors_of(fn, *args, **kwargs) -> list[str]:
    """跑一次、把 `PlanRejected.errors` 取出来（没拒返回空）✓。"""
    from app.services import canvas_plan as plan_mod  # noqa: PLC0415

    try:
        fn(*args, **kwargs)
    except plan_mod.PlanRejected as error:
        return list(error.errors)
    return []


def main() -> int:
    sys.path.insert(0, str(BP))
    from app.services import canvas_ops as ops  # noqa: PLC0415
    from app.services import canvas_plan as plan_mod  # noqa: PLC0415
    from app.services import canvas_plan_store as store  # noqa: PLC0415

    # ---------------------------------------------------------------- ① 场景工具面
    check("场景: chat 只给 4 件", len(ops.tools_for_scene("chat")) == 4, ops.tools_for_scene("chat"))
    check("场景: canvas 给全量", len(ops.tools_for_scene("canvas")) == len(ops.TOOL_NAMES))
    check("场景: chat 是 canvas 的真子集",
          set(ops.tools_for_scene("chat")) < set(ops.tools_for_scene("canvas")))
    check("场景: chat 只要读 + 整批落地（不放排版类工具）",
          set(ops.tools_for_scene("chat")) == {
              "canvas_get_state", "canvas_get_selection",
              "canvas_export_snapshot", "canvas_apply_ops",
          }, ops.tools_for_scene("chat"))
    check("场景: tool_schemas('chat') 与白名单条数一致",
          len(ops.tool_schemas("chat")) == 4, len(ops.tool_schemas("chat")))
    check("场景: tool_schemas() 不带参数仍是全量（老调用方不变）",
          len(ops.tool_schemas()) == len(ops.TOOL_NAMES))

    # ⭐ 两道守卫：坏配置必须炸，不能返回空表
    try:
        ops.tools_for_scene("nope")
        check("场景: 未知场景要抛错", False, "没抛")
    except ValueError as error:
        check("场景: 未知场景要抛错", "未知场景" in str(error), str(error))

    original = ops.SCENE_TOOL_SETS["chat"]
    try:
        ops.SCENE_TOOL_SETS["chat"] = frozenset({"canvas_get_state", "freezone_get_state"})
        problems = ops.scene_problems("chat")
        check("场景: 白名单点了不存在的工具 ⇒ 报配置漂移",
              any("不存在" in p for p in problems), problems)

        ops.SCENE_TOOL_SETS["chat"] = frozenset({"freezone_get_state"})
        problems = ops.scene_problems("chat")
        check("场景: 交集为空 ⇒ 报静默归零（这就是 dramaclaw 那个坑）",
              any("交集为空" in p for p in problems), problems)
        try:
            ops.tools_for_scene("chat")
            check("场景: 交集为空时 tools_for_scene 要抛错", False, "没抛")
        except ValueError:
            check("场景: 交集为空时 tools_for_scene 要抛错", True)
    finally:
        ops.SCENE_TOOL_SETS["chat"] = original

    # ---------------------------------------------------------------- ② 编译 happy path
    compiled = plan_mod.compile_plan(_plan())
    preflight = compiled["preflight"]
    check("编译: 建 3 个", preflight["creates"] == 3, preflight)
    check("编译: 连 2 条", preflight["links"] == 2, preflight)
    check("编译: 跑 1 次", preflight["runs"] == 1, preflight)
    check("编译: 节点类型计数正确",
          preflight["byNodeType"] == {"text": 1, "image": 1, "config": 1}, preflight["byNodeType"])
    check("编译: goal 逐字保留（不许润色）", compiled["goal"] == "女侠在雨夜走进客栈", compiled["goal"])
    check("编译: 所有 op 类型前端都认",
          all(op["type"] in ops.OP_TYPES for op in compiled["ops"]),
          [op["type"] for op in compiled["ops"]])

    again = plan_mod.compile_plan(_plan())
    check("编译: 同一计划两次得到同一批 id（重放不建两套）",
          compiled["refs"] == again["refs"], (compiled["refs"], again["refs"]))
    check("编译: id 不是随机的（两次字节相同）",
          json.dumps(compiled["ops"], sort_keys=True) == json.dumps(again["ops"], sort_keys=True))

    # ---------------------------------------------------------------- ③ 布局
    def xs(result: dict) -> list[float]:
        return [op["x"] for op in result["ops"] if op["type"] == "add_node"]

    def ys(result: dict) -> list[float]:
        return [op["y"] for op in result["ops"] if op["type"] == "add_node"]

    flow = plan_mod.compile_plan(_plan(options={"layout": "flow"}))
    check("布局: flow 按 after 分三列", len(set(xs(flow))) == 3, xs(flow))
    check("布局: flow 列内同 x（p1→p2→p3 逐列右移）",
          xs(flow)[0] < xs(flow)[1] < xs(flow)[2], xs(flow))

    row = plan_mod.compile_plan(_plan(options={"layout": "row"}))
    check("布局: row 同 y、x 递增", len(set(ys(row))) == 1 and xs(row) == sorted(xs(row)), (xs(row), ys(row)))

    column = plan_mod.compile_plan(_plan(options={"layout": "column"}))
    check("布局: column 同 x、y 递增",
          len(set(xs(column))) == 1 and ys(column) == sorted(ys(column)), (xs(column), ys(column)))

    manual = _plan()
    manual["items"][0]["x"] = 5000.0
    honored = plan_mod.compile_plan(manual)
    check("布局: item 明确给了 x 就尊重它",
          any(op.get("x") == 5000.0 for op in honored["ops"] if op["type"] == "add_node"), xs(honored))

    # ---------------------------------------------------------------- ④ 拒绝：一次报全
    errors = _errors_of(plan_mod.compile_plan, _plan(goal="   "))
    check("拒绝: 缺 goal", any("goal" in e for e in errors), errors)

    empty_items = _errors_of(plan_mod.compile_plan, {"goal": "x", "items": []})
    check("拒绝: items 为空", any("非空数组" in e for e in empty_items), empty_items)

    bad_layout = _errors_of(plan_mod.compile_plan, _plan(options={"layout": "spiral"}))
    check("拒绝: 未知 layout", any("layout" in e for e in bad_layout), bad_layout)

    no_text = _errors_of(plan_mod.compile_plan, {
        "goal": "x", "items": [{"nodeType": "text", "ref": "p1"}],
    })
    check("拒绝: 文本节点没 text（空文本卡毫无意义）",
          any("没给 text" in e for e in no_text), no_text)

    run_on_text = _errors_of(plan_mod.compile_plan, {
        "goal": "x",
        "items": [{"nodeType": "text", "ref": "p1", "text": "a", "run": True, "mode": "image"}],
    })
    check("拒绝: 只有 config 能 run",
          any("只有 config 节点是生成入口" in e for e in run_on_text), run_on_text)

    run_no_mode = _errors_of(plan_mod.compile_plan, {
        "goal": "x",
        "items": [{"nodeType": "config", "ref": "p1", "workflowId": "wf", "run": True}],
    })
    check("拒绝: run 不给 mode（不许默认成 image）",
          any("mode 不合法" in e for e in run_no_mode), run_no_mode)

    empty_config = _errors_of(plan_mod.compile_plan, {
        "goal": "x", "items": [{"nodeType": "config", "ref": "p1"}],
    })
    check("拒绝: config 既没 workflowId 也没 values",
          any("不会生成任何东西" in e for e in empty_config), empty_config)

    dup_ref = _errors_of(plan_mod.compile_plan, {
        "goal": "x",
        "items": [{"nodeType": "text", "ref": "a", "text": "1"}, {"nodeType": "text", "ref": "a", "text": "2"}],
    })
    check("拒绝: ref 重复", any("ref 重复" in e for e in dup_ref), dup_ref)

    ghost = _errors_of(plan_mod.compile_plan, {
        "goal": "x",
        "items": [{"nodeType": "text", "ref": "a", "text": "1", "after": ["ghost"]}],
    }, {"nodes": [{"id": "node-real"}]})
    check("拒绝: after 指向不存在的东西（有快照时）",
          any("不存在的东西" in e for e in ghost), ghost)

    ghost_no_snapshot = _errors_of(plan_mod.compile_plan, {
        "goal": "x",
        "items": [{"nodeType": "text", "ref": "a", "text": "1", "after": ["node-real"]}],
    })
    check("拒绝: 没给快照时如实说「把 snapshot 传上来」，而不是猜",
          any("snapshot" in e for e in ghost_no_snapshot), ghost_no_snapshot)

    cycle = _errors_of(plan_mod.compile_plan, {
        "goal": "x",
        "items": [
            {"nodeType": "text", "ref": "a", "text": "1", "after": ["b"]},
            {"nodeType": "text", "ref": "b", "text": "2", "after": ["a"]},
        ],
    })
    check("拒绝: 依赖成环", any("成环" in e for e in cycle), cycle)

    many = _errors_of(plan_mod.compile_plan, {
        "goal": "  ",
        "items": [
            {"nodeType": "text", "ref": "a"},
            {"nodeType": "config", "ref": "a", "run": True},
            {"nodeType": "whiteboard", "ref": "c"},
        ],
        "options": {"layout": "spiral"},
    })
    check("拒绝: 一次报全（不是改一条报一条）", len(many) >= 5, f"{len(many)} 条: {many}")

    # ---------------------------------------------------------------- ⑤ after 指向既有节点
    snapshot = {"nodes": [{"id": "node-existing", "type": "image", "x": 0, "y": 0}], "edges": []}
    attach = plan_mod.compile_plan(
        {"goal": "接在这张图后面", "items": [{"nodeType": "video", "ref": "v1", "after": ["node-existing"]}]},
        snapshot,
    )
    connections = [op for op in attach["ops"] if op["type"] == "connect_nodes"][0]["connections"]
    check("既有节点: 连线从快照里的 id 出发",
          connections[0]["from"] == "node-existing", connections)

    # ---------------------------------------------------------------- ⑥ 草案仓：draft → confirm → replay
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = store.plans_dir
        store.plans_dir = lambda: Path(tmp)  # type: ignore[assignment]
        try:
            record = store.draft_record(_plan(), compiled, expected_revision=7)
            store.save_plan(record)
            check("草案: id 是 16 位指纹", len(record["id"]) == 16, record["id"])
            check("草案: 对外视图**不含 ops**",
                  "ops" not in store.public_view(record), list(store.public_view(record)))
            check("草案: 对外视图带 preflight（确认前靠数字核对意图）",
                  store.public_view(record)["preflight"]["creates"] == 3)
            check("草案: requiresConfirmation = True",
                  store.public_view(record)["requiresConfirmation"] is True)

            loaded = store.load_plan(record["id"])
            check("草案: 落盘后可读回", loaded["id"] == record["id"])

            same = store.draft_record(_plan(), compiled, expected_revision=7)
            check("草案: 同一份计划重复起草得到同一个 id", same["id"] == record["id"])

            loaded.update({"status": store.STATUS_CONFIRMED, "confirmCount": 1})
            store.save_plan(loaded)
            confirmed = store.public_view(store.load_plan(record["id"]), with_ops=True)
            check("确认: 带 ops", len(confirmed["ops"]) == 3, len(confirmed["ops"]))
            check("确认: requiresConfirmation = False", confirmed["requiresConfirmation"] is False)

            try:
                store.load_plan("../../etc/passwd")
                check("草案: 非法 id 要被挡住（防路径穿越）", False, "没抛")
            except store.PlanNotFound:
                check("草案: 非法 id 要被挡住（防路径穿越）", True)

            listed = store.list_plans()
            check("草案: 列表能列到", len(listed) == 1, len(listed))
        finally:
            store.plans_dir = original_dir  # type: ignore[assignment]

    # ---------------------------------------------------------------- ⑦ 指纹
    check("指纹: 同一份计划稳定", plan_mod.plan_digest(_plan()) == plan_mod.plan_digest(_plan()))
    check("指纹: 改 goal 就变",
          plan_mod.plan_digest(_plan()) != plan_mod.plan_digest(_plan(goal="换一句")))
    check("指纹: 编译产物（refs/preflight）不参与指纹",
          plan_mod.plan_digest(_plan()) == plan_mod.plan_digest(_plan()))

    # ---------------------------------------------------------------- ⑧ 前端同源
    canvas_ops_src = (REPO / "frontend" / "app" / "utils" / "canvasOps.ts").read_text(encoding="utf-8")
    for token in ("baseRevision", "applyId", "conflict", "replayed"):
        check(f"同源: canvasOps.ts 认 {token}", token in canvas_ops_src)

    use_canvas_src = (REPO / "frontend" / "app" / "composables" / "useCanvas.ts").read_text(encoding="utf-8")
    check("同源: useCanvas.ts 有画布版本 revision", "revision" in use_canvas_src)
    check("同源: useCanvas.ts 导出 canvasRevision()", "export function canvasRevision" in use_canvas_src)
    check("同源: 版本只随 nodes/edges 变（视口不算）",
          "watch([nodes, edges]" in use_canvas_src)

    agent_src = (REPO / "frontend" / "app" / "utils" / "canvasAgent.ts").read_text(encoding="utf-8")
    check("同源: canvasAgent.ts 有草案通路", "draftCanvasPlan" in agent_src)
    check("同源: canvasAgent.ts 有确认通路", "applyCanvasPlan" in agent_src)

    # ---------------------------------------------------------------- 汇总
    failed = [(name, detail) for name, ok, detail in _RESULTS if not ok]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  ⇒ {detail}"))
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
