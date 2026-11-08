"""**画布计划编译层** —— 意图 → 计划 → 可执行 op（确定性、纯函数、不调模型）。

## 为什么要多这一层

现在 `canvas_agent` 的通路是「模型逐条调工具 → 每调一次攒几条 op → 前端顺着落」。
它有两个治不好的毛病：

1. **模型得自己算坐标、自己编 id、自己判断该连谁** —— 这些都不该是语言模型的活；
2. **用户没有拒绝的机会** —— op 一返回就落地了，用户看到画布被改完才知道 AI 想干什么 ✗。

所以中间插一层**计划**：模型只负责说「要放哪几个节点、谁接谁」（有严格 schema），
本模块负责把它们**确定性地**编译成 op —— 算坐标、生成稳定 id、做交叉校验、出 preflight 计数。

三段于是变成：

==================================  ==============================  ==========================
段                                  谁做                            在哪
==================================  ==============================  ==========================
话 → 计划                           模型（结构化输出 / 工具调用）    `canvas_agent`
计划 → op                           本模块（纯函数、可单测）        `canvas_plan.compile_plan`
op → 状态                           `applyCanvasOps`                `utils/canvasOps.ts`
==================================  ==============================  ==========================

## 四条硬口径（都有对应的失败案例）

1. **不改用户原话**：``goal`` 进计划、出结果，逐字不动。模型可以补充 ``items``，
   但不能把用户说的「雨夜客栈」润色成「夜晚的客栈」—— 那是用户的稿子 ✓。
2. **缺参当场拒，一次报全**：不给默认值糊过去。报错要**一次列全所有问题**，
   否则模型改一条、再报下一条，8 步步数全耗在试探上 ✗。
3. **坐标由编译器算**：item 不给 ``x/y`` 就按 ``options.layout`` 排；给了就尊重。
   让模型自由发挥坐标的后果是节点叠成一坨 ✓✗。
4. **确定性 id**：同一份计划编译两次得到同一批节点 id ⇒ 重放不会长出两套节点 ✓。

## 谁执行

与 `canvas_ops` 一致：**只产出 op，不碰画布**。落地在前端。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from . import canvas_ops

__all__ = [
    "DEFAULT_LAYOUT",
    "MAX_ITEMS",
    "PLAN_KIND",
    "PlanRejected",
    "compile_plan",
    "plan_digest",
    "preflight_of",
]

#: 计划的结构版本 ✓（前端与落盘都靠它分流）
PLAN_KIND = "canvas_plan.v1"

#: 一份计划最多几项。超过就不是「一次说清要什么」，而是把 op 当脚本用 ✗
MAX_ITEMS = 60

#: 布局模式 ✓。``flow`` 按 ``after`` 依赖分列，是唯一能表达「这个接那个后面」的排法
DEFAULT_LAYOUT = "flow"
LAYOUTS: tuple[str, ...] = ("flow", "row", "column")

#: 布局算宽高时插件节点没有后端尺寸表 ⇒ 用文本尺寸兜底（**只用于排版，不写进 op** ✓）
_FALLBACK_SIZE = (340.0, 240.0)

#: 列/行间距下限（再小节点就叠上了）
_MIN_GAP = 20.0


class PlanRejected(ValueError):
    """计划被拒 ✓ —— ``errors`` 是**全部**问题，不是第一条。

    继承 ``ValueError``：路由层现有的 ``except ValueError`` 会自动接住并转 400 ✓，
    不必给「计划被拒」单开一条错误分支。
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("；".join(self.errors) or "计划不合法")


# ---------------------------------------------------------------- 小工具


def _number(value: Any, label: str, errors: list[str]) -> float | None:
    """取数字；类型不对记一条错并返回 ``None`` ✓（不抛，攒齐了一起报）。"""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} 要是数字，收到 {type(value).__name__}")
        return None
    return float(value)


def _text_of(raw: dict, key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value


def _refs_of(raw: Any) -> list[str]:
    """``after`` / ``edges.from`` 允许写成单个字符串或数组 ✓（模型两种都爱用）。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if isinstance(item, (str, int)) and str(item).strip()]
    return []


def _size_of(node_type: str, raw: dict) -> tuple[float, float]:
    """该项的排版尺寸 ✓ —— 内置类型用同一张默认尺寸表，插件/未知类型用兜底。"""
    base = canvas_ops.DEFAULT_NODE_SIZE.get(node_type, _FALLBACK_SIZE)
    width = _as_number(raw.get("w")) or base[0]
    height = _as_number(raw.get("h")) or base[1]
    return width, height


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _existing_ids(snapshot: dict | None) -> set[str]:
    """快照里**已有**的节点 id ✓ —— ``after`` 允许指向它们（"接在这张图后面"）。"""
    if not isinstance(snapshot, dict):
        return set()
    nodes = snapshot.get("nodes")
    if not isinstance(nodes, list):
        return set()
    return {
        str(node.get("id"))
        for node in nodes
        if isinstance(node, dict) and node.get("id")
    }


# ---------------------------------------------------------------- 依赖分层（flow 布局）


def _depths(refs: list[str], after: dict[str, list[str]], errors: list[str]) -> dict[str, int]:
    """算每个 ref 的层号（``flow`` 布局的列号）✓。有环 ⇒ 记错并退回 0 层。

    环必须**报出来**：``A after B`` 且 ``B after A`` 时按层号排会出现
    「两列互相等对方先算宽度」，递归就会栈溢出；更重要的是这种计划在语义上就是错的 ✗。
    """
    depth: dict[str, int] = {}
    visiting: set[str] = set()

    def visit(ref: str) -> int:
        if ref in depth:
            return depth[ref]
        if ref in visiting:
            errors.append(f"依赖成环：{' → '.join([*visiting, ref])}")
            return 0
        visiting.add(ref)
        parents = after.get(ref) or []
        value = 0 if not parents else 1 + max(visit(parent) for parent in parents)
        visiting.discard(ref)
        depth[ref] = value
        return value

    for ref in refs:
        visit(ref)
    return depth


def _positions(
    refs: list[str],
    sizes: dict[str, tuple[float, float]],
    after: dict[str, list[str]],
    layout: str,
    start_x: float,
    start_y: float,
    gap: float,
    errors: list[str],
) -> dict[str, tuple[float, float]]:
    """按布局算每个 ref 的左上角坐标 ✓（这是「坐标不要模型编」的落实处）。"""
    if layout == "row":
        out: dict[str, tuple[float, float]] = {}
        cursor = start_x
        for ref in refs:
            out[ref] = (cursor, start_y)
            cursor += sizes[ref][0] + gap
        return out

    if layout == "column":
        out = {}
        cursor = start_y
        for ref in refs:
            out[ref] = (start_x, cursor)
            cursor += sizes[ref][1] + gap
        return out

    # flow：按层号分列，列内竖排；列宽取该列最宽的那个
    depth = _depths(refs, after, errors)
    columns: dict[int, list[str]] = {}
    for ref in refs:
        columns.setdefault(depth.get(ref, 0), []).append(ref)

    out = {}
    cursor_x = start_x
    for level in sorted(columns):
        members = columns[level]
        widest = max(sizes[ref][0] for ref in members)
        cursor_y = start_y
        for ref in members:
            out[ref] = (cursor_x, cursor_y)
            cursor_y += sizes[ref][1] + gap
        cursor_x += widest + gap
    return out


# ---------------------------------------------------------------- 单项校验


def _normalize_item(
    raw: Any,
    index: int,
    default_ref: str,
    errors: list[str],
) -> dict[str, Any] | None:
    """把一项计划条目清洗成内部形状 ✓；有硬伤返回 ``None``（错误已记）。"""
    where = f"items[{index}]"
    if not isinstance(raw, dict):
        errors.append(f"{where} 不是对象")
        return None

    node_type = raw.get("nodeType") or raw.get("node_type")
    problem = canvas_ops.node_type_problem(node_type)
    if problem:
        errors.append(f"{where}.{problem}")
        return None

    ref = _text_of(raw, "ref")
    ref = ref.strip() if ref and ref.strip() else default_ref

    item: dict[str, Any] = {"ref": ref, "nodeType": str(node_type)}

    for key in ("title", "url", "workflowId", "engineStage"):
        value = _text_of(raw, key)
        if value:
            item[key] = value

    text = _text_of(raw, "text")
    if text is not None:
        item["text"] = text

    run_mode = raw.get("runMode")
    if run_mode is not None:
        if run_mode not in ("engine", "comfyui"):
            errors.append(f"{where}.runMode 只能是 engine / comfyui，收到 {run_mode!r}")
        else:
            item["runMode"] = run_mode

    values = raw.get("values")
    if values is not None:
        if not isinstance(values, dict):
            errors.append(f"{where}.values 要是对象")
        else:
            item["values"] = {
                str(key): str(value) for key, value in values.items() if value is not None
            }

    # ⭐ 内容硬要求：文本节点没字 = 一张空卡，用户看不出 AI 想干什么 ✗
    if node_type == "text" and not (item.get("text") or "").strip():
        errors.append(f"{where} 是文本节点但没给 text —— 空文本卡在画布上没有意义")

    # ⭐ 配置节点是**生成入口**：既没工作流也没参数，点它什么都不会发生 ✗
    if node_type == "config" and not item.get("workflowId") and not item.get("values"):
        errors.append(f"{where} 是配置节点但既没 workflowId 也没 values —— 它不会生成任何东西")

    # ⭐ 只有配置节点能开跑；别的类型标 run 是模型理解错了
    run = raw.get("run")
    if run is not None and not isinstance(run, bool):
        errors.append(f"{where}.run 要是布尔，收到 {type(run).__name__}")
        run = None
    if run:
        if node_type != "config":
            errors.append(f"{where} 标了 run，但只有 config 节点是生成入口（它是 {node_type}）")
        mode = raw.get("mode")
        if mode not in canvas_ops.GENERATION_MODES:
            # 不给默认 image：源那边未知模式静默当图片，结果是「跑音乐变成跑图」且不报错 ✗
            errors.append(
                f"{where} 标了 run 但 mode 不合法：{mode!r}"
                f"（允许 {', '.join(canvas_ops.GENERATION_MODES)}）"
            )
        else:
            item["run"] = mode
            prompt = _text_of(raw, "prompt")
            if prompt:
                item["prompt"] = prompt

    for key in ("x", "y", "w", "h"):
        number = _number(raw.get(key), f"{where}.{key}", errors)
        if number is not None:
            item[key] = number

    item["after"] = _refs_of(raw.get("after") or raw.get("dependsOn"))
    return item


# ---------------------------------------------------------------- 编译


def compile_plan(plan: Any, snapshot: dict | None = None) -> dict[str, Any]:
    """把一份计划编译成**已校验**的 op 列表 ✓。有硬伤抛 ``PlanRejected``。

    ``snapshot`` 给了就参与交叉校验（连线指向的节点必须存在）✓；不给也能编译，
    只是少一层存在性检查 —— 与 `canvas_ops.validate_ops` 的口径一致。

    返回::

        {"kind": "canvas_plan.v1", "goal": <原话>, "ops": [...], "refs": {ref: id},
         "preflight": {...}, "layout": {...}, "warnings": [...]}
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(plan, dict):
        raise PlanRejected(["计划要是对象"])

    kind = plan.get("kind")
    if kind is not None and kind != PLAN_KIND:
        errors.append(f"kind 不认识：{kind!r}（要 {PLAN_KIND}）")

    # ⭐ 原话逐字保留：这是用户的稿子，编译器只有搬运权，没有润色权 ✓
    goal = _text_of(plan, "goal")
    if not goal or not goal.strip():
        errors.append("缺 goal（用户原话）—— 计划必须能追溯到用户说了什么")
        goal = ""

    raw_items = plan.get("items")
    if not isinstance(raw_items, (list, tuple)) or not raw_items:
        errors.append("items 要是非空数组（至少要放一个节点，否则这份计划什么也不做）")
        raw_items = []
    elif len(raw_items) > MAX_ITEMS:
        errors.append(f"items 最多 {MAX_ITEMS} 项，收到 {len(raw_items)} 项（再多请拆成多次计划）")

    options = plan.get("options") if isinstance(plan.get("options"), dict) else {}
    layout = options.get("layout") or DEFAULT_LAYOUT
    if layout not in LAYOUTS:
        errors.append(f"options.layout 不认识：{layout!r}（允许 {', '.join(LAYOUTS)}）")
        layout = DEFAULT_LAYOUT
    gap = _as_number(options.get("gap"))
    gap = _MIN_GAP if gap is None else max(_MIN_GAP, gap)

    # —— 逐项清洗（错误攒着，一次报全 ✓）
    items: list[dict[str, Any]] = []
    seen_refs: dict[str, int] = {}
    for index, raw in enumerate(raw_items):
        item = _normalize_item(raw, index, f"i{index + 1}", errors)
        if item is None:
            continue
        ref = item["ref"]
        if ref in seen_refs:
            errors.append(f"ref 重复：{ref!r}（items[{seen_refs[ref]}] 与 items[{index}]）")
            continue
        seen_refs[ref] = index
        items.append(item)

    # —— 依赖只能指向「本计划里的 ref」或「快照里已有的节点 id」✓
    #
    # 允许后者很关键：最常见的一句是「在选中的这张图后面接一个视频节点」，
    # 那个节点在计划之外、在画布之上。只认计划内 ref 的话这类需求根本无从表达 ✗。
    known_refs = set(seen_refs)
    existing_ids = _existing_ids(snapshot)
    for item in items:
        for parent in item["after"]:
            if parent in known_refs or parent in existing_ids:
                continue
            if existing_ids:
                errors.append(
                    f"{item['ref']} 的 after 指向不存在的东西：{parent!r}"
                    f"（既不是本计划的 ref，也不在当前画布上）"
                )
            else:
                # 没有快照就分不清「写错了」和「指现有节点」⇒ 如实说明缺什么，别猜 ✗
                errors.append(
                    f"{item['ref']} 的 after 指向 {parent!r}：它不是本计划的 ref；"
                    f"若要连接到画布上已有的节点，请把 snapshot 一起传上来"
                )
        if item["ref"] in item["after"]:
            errors.append(f"{item['ref']} 不能 after 自己")

    # —— 显式 edges（可省：`after` 已经能表达连线）
    edges: list[tuple[str, str]] = [(parent, item["ref"]) for item in items for parent in item["after"]]
    raw_edges = plan.get("edges")
    if raw_edges is not None:
        if not isinstance(raw_edges, (list, tuple)):
            errors.append("edges 要是数组")
        else:
            for index, raw in enumerate(raw_edges):
                if not isinstance(raw, dict):
                    errors.append(f"edges[{index}] 不是对象")
                    continue
                source, target = raw.get("from"), raw.get("to")
                if not isinstance(source, str) or not isinstance(target, str):
                    errors.append(f"edges[{index}] 的 from/to 要是 ref 字符串")
                    continue
                if source not in known_refs or target not in known_refs:
                    errors.append(f"edges[{index}] 指向不存在的项：{source!r} → {target!r}")
                    continue
                if source == target:
                    errors.append(f"edges[{index}] 是自连：{source!r}")
                    continue
                edges.append((source, target))

    if errors:
        # 一次全报：让模型一轮改完，而不是改一条报一条（8 步根本不够试）
        raise PlanRejected(errors)

    # —— 坐标：给了尊重，不给按布局算 ✓
    sizes = {item["ref"]: _size_of(item["nodeType"], item) for item in items}
    start_x = _as_number(options.get("startX"))
    start_x = canvas_ops.next_canvas_x(snapshot) if start_x is None else start_x
    start_y = _as_number(options.get("startY")) or 0.0
    computed = _positions(
        [item["ref"] for item in items], sizes, {i["ref"]: i["after"] for i in items},
        layout, start_x, start_y, gap, errors,
    )
    if errors:
        raise PlanRejected(errors)

    # —— 生成 op（确定性 id：同一计划重放不会建两套节点 ✓）
    refs: dict[str, str] = {}
    ops: list[dict[str, Any]] = []
    for item in items:
        node_id = canvas_ops._stable_id("node", goal, item["ref"])  # noqa: SLF001
        refs[item["ref"]] = node_id
        x = item.get("x", computed[item["ref"]][0])
        y = item.get("y", computed[item["ref"]][1])
        payload = {
            "id": node_id,
            "title": item.get("title"),
            "text": item.get("text"),
            "url": item.get("url"),
            "workflowId": item.get("workflowId"),
            "runMode": item.get("runMode"),
            "engineStage": item.get("engineStage"),
            "values": item.get("values"),
            "w": item.get("w"),
            "h": item.get("h"),
        }
        ops.append(
            canvas_ops._add_node_op(  # noqa: SLF001
                {key: value for key, value in payload.items() if value is not None},
                node_type=item["nodeType"],
                x=x,
                y=y,
                title=item.get("title"),
            )
        )

    if edges:
        # ``after`` 指向快照里的既有节点时，source 不是本计划的 ref ⇒ 原样用它的 id ✓
        ops.append(
            {
                "type": "connect_nodes",
                "connections": [
                    {"from": refs.get(source, source), "to": refs[target]} for source, target in edges
                ],
            }
        )

    if options.get("selectAll"):
        ops.append({"type": "select_nodes", "ids": list(refs.values())})

    viewport = options.get("viewport")
    if isinstance(viewport, dict):
        scale = _as_number(viewport.get("scale")) or 1.0
        clamped = min(canvas_ops.MAX_SCALE, max(canvas_ops.MIN_SCALE, scale))
        if clamped != scale:
            warnings.append(
                f"viewport.scale {scale} 超出 {canvas_ops.MIN_SCALE}–{canvas_ops.MAX_SCALE}，已夹到 {clamped}"
            )
        ops.append(
            {
                "type": "set_viewport",
                "viewport": {"x": _as_number(viewport.get("x")) or 0.0,
                             "y": _as_number(viewport.get("y")) or 0.0,
                             "scale": clamped},
            }
        )

    for item in items:
        if not item.get("run"):
            continue
        op: dict[str, Any] = {
            "type": "run_generation",
            "nodeId": refs[item["ref"]],
            "mode": item["run"],
        }
        if item.get("prompt"):
            op["prompt"] = item["prompt"]
        ops.append(op)

    # —— 最后一道闸门：用与前端完全同一套校验，编出来的一定能落地 ✓
    clean, problems = canvas_ops.validate_ops(ops, snapshot)
    if problems:
        raise PlanRejected([f"编译结果未通过画布校验：{problem}" for problem in problems])

    return {
        "kind": PLAN_KIND,
        "goal": goal,
        "ops": clean,
        "refs": refs,
        "preflight": preflight_of(clean),
        "layout": {"mode": layout, "startX": start_x, "startY": start_y, "gap": gap,
                   # 列数 = 实际落到的不同 x（flow 一列一 x；row 每项一个 x）
                   "columns": len({round(pos[0], 3) for pos in computed.values()})},
        "warnings": warnings,
    }


def preflight_of(ops: list[dict]) -> dict[str, Any]:
    """执行前计数 ✓ —— 让人在点确认之前就能看出「这次要建几个、连几条、跑几次」。

    数字对不上就是计划理解错了，这种情况**必须**在落地前发现，而不是落地后看着画布发愣 ✓。
    """
    by_op: dict[str, int] = {}
    by_node: dict[str, int] = {}
    creates = links = runs = 0
    for op in ops:
        kind = str(op.get("type"))
        by_op[kind] = by_op.get(kind, 0) + 1
        if kind == "add_node":
            node_type = str(op.get("nodeType"))
            by_node[node_type] = by_node.get(node_type, 0) + 1
            creates += 1
        elif kind == "connect_nodes":
            links += len(op.get("connections") or [])
        elif kind == "run_generation":
            runs += 1
    return {
        "total": len(ops),
        "creates": creates,
        "links": links,
        "runs": runs,
        "byOp": by_op,
        "byNodeType": by_node,
    }


def plan_digest(plan: dict[str, Any]) -> str:
    """计划的稳定指纹 ✓ —— 同一份计划重复提交可据此判重（幂等键）。

    只对 ``kind/goal/items/options`` 求值：``refs`` / ``preflight`` 是编译产物，
    把它们算进指纹会让「同一份计划编译两次」得到两个指纹 ✗。
    """
    payload = {
        "kind": plan.get("kind"),
        "goal": plan.get("goal"),
        "items": plan.get("items"),
        "edges": plan.get("edges"),
        "options": plan.get("options"),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
