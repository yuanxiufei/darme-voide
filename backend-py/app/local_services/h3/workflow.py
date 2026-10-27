"""ComfyUI 工作流：**UI 图 → API 格式** 转换 + 参数注入（零依赖，纯函数）。

## 为什么必须有这一步

ComfyUI 里流通的 workflow 有两种形态，**互不通用** ✗：

* **UI 格式**（``{"nodes": [...], "links": [...]}``）—— 界面里能打开、能看连线，但**不能** POST；
  参考项目（``minimax-h3-comfyui/workflows/MiniMax_H3_Fast_T2V.json`` 等）导出的都是这种 ✓。
* **API 格式**（``{"3": {"class_type": "...", "inputs": {...}}, ...}``）—— 只有它能 ``POST /prompt`` ✓。

界面上的「导出 → API 格式」是**前端**做的：它拿 ``/object_info`` 里每个节点的
``INPUT_TYPES()``（**有序** ✓）把 ``widgets_values`` 逐个对回**字段名**，再把连线（``links``）
解成 ``["<上游节点id>", <输出槽位>]`` ✓。本文件就是把这个算法搬到后端 ✓ ——
这样**参考 workflow 可以原样用**，不需要人工再导一遍 ✗。

## 关键判据（都来自 ComfyUI 前端/服务端的真实行为，不是猜 ✓）

1. **哪些输入是 widget**：``INPUT_TYPES()`` 里该项的 ``type`` 是**数组**（``["COMBO", {...}]``
   或旧的 ``[选项数组, {...}]``）或是标量类型（INT / FLOAT / STRING / BOOLEAN / COMBO），
   且**没有** ``forceInput: True`` ✓。widget 的**顺序** = ``required`` 全部、再 ``optional`` 全部 ✓。
2. **``control_after_generate`` 会多吃一个槽位**：INT 项带该选项时，``widgets_values`` 里会多一个
   模式字符串（``fixed`` / ``randomize`` / ``increment`` ✓）。参考 T2V 里 ``RandomNoise`` 的
   ``['424242424242', 'randomize']`` 就是这两槽 ✓ ⇒ 只把**第一槽**当种子，并把模式语义实现出来 ✓。
3. **node mode**：``0``=正常、``2``=mute、``4``=bypass ✓。**非 0 的节点不转换**（它的输入也一起丢掉 ✓）——
   参考 I2V 工作流就是靠这个把「可选尾帧通道」**默认关掉**的 ✓（要用就显式把它的 mode 改回 0 ✓）。
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

__all__ = [
    "ASPECT_RATIO_OPTIONS",
    "MAIN_VIDEO_CLASSES",
    "MODE_ACTIVE",
    "MODE_MUTE",
    "REF_IMAGE_PORT_PREFIX",
    "RESOLUTION_SELECTOR_MIN_MEGAPIXELS",
    "RESOLUTION_SELECTOR_MULTIPLE",
    "SUBGRAPH_INPUT_ID",
    "SUBGRAPH_OUTPUT_ID",
    "WorkflowConversionError",
    "activate_nodes",
    "active_modes_default",
    "check_combo_choices",
    "combo_options",
    "expected_resolution",
    "find_nodes",
    "flatten_subgraphs",
    "frames_for_seconds",
    "image_source",
    "main_video_node",
    "node_classes",
    "port_source",
    "ratio_terms",
    "reference_slot_names",
    "resolution_selector_size",
    "scalar_source",
    "seconds_for_frames",
    "set_combo",
    "set_input",
    "set_node_input",
    "set_seed",
    "ui_to_api",
    "video_plan",
]

#: 判为「标量 widget」的类型名（与 ComfyUI 前端一致 ✓）
_WIDGET_SCALAR_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
#: node mode：0 正常 / 2 mute / 4 bypass
MODE_ACTIVE = 0
MODE_MUTE = 2
MODE_BYPASS = 4

#: H3 的帧网格（搬自核心节点 ``comfy_extras/nodes_minimax_h3.py`` 的 schema ✓）：
#: ``io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps,
#: snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)"``
H3_FPS = 24
H3_MIN_FRAMES = 5
H3_FRAME_GRID = 17


class WorkflowConversionError(RuntimeError):
    """UI 图无法可靠转成 API 格式（**宁可报错也不要静默换个错工作流** ✗）。"""


def active_modes_default() -> tuple[int, ...]:
    """默认只转换 mode=0 的节点（mute/bypass 的一律跳过 ✓）。"""
    return (MODE_ACTIVE,)


#: 子图（subgraph）的**虚拟节点 id**：ComfyUI 前端固定用 ``-10`` 当「子图入参」、``-20`` 当「子图出参」✓
SUBGRAPH_INPUT_ID = "-10"
SUBGRAPH_OUTPUT_ID = "-20"


# ── 读取 ────────────────────────────────────────────────────────────────
def _as_graph(source: Any) -> dict[str, Any]:
    """接收 路径 / dict / JSON 字符串 ⇒ UI 图 dict（**顺手把子图摊平** ✓ 见 :func:`flatten_subgraphs` ✓）。"""
    if isinstance(source, dict):
        return flatten_subgraphs(source)
    if isinstance(source, (str, Path)):
        return flatten_subgraphs(json.loads(Path(source).read_text(encoding="utf-8")))
    raise WorkflowConversionError(f"不认识的 workflow 形态：{type(source).__name__}")


def find_nodes(graph: dict[str, Any], *, class_type: str | None = None,
               title_contains: str | None = None, mode: int | None = None) -> list[dict[str, Any]]:
    """按类名/标题片段/mode 找节点（**参数注入的入口** ✓）。

    参考 workflow 里节点常带语义标题（如 ``«DURATION (seconds)»`` ✓）⇒ 用标题定位比记 node id 稳 ✓。
    """
    out: list[dict[str, Any]] = []
    for node in _as_graph(graph).get("nodes") or []:
        if class_type and node.get("type") != class_type:
            continue
        if title_contains and title_contains not in str(node.get("title") or ""):
            continue
        if mode is not None and node.get("mode", MODE_ACTIVE) != mode:
            continue
        out.append(node)
    return out


def node_classes(graph: dict[str, Any], modes: tuple[int, ...] = active_modes_default()) -> set[str]:
    """图谱里**会参与执行**的节点类集合（供 ``missing_nodes`` 预检 ✓）。"""
    return {
        str(node.get("type"))
        for node in _as_graph(graph).get("nodes") or []
        if node.get("mode", MODE_ACTIVE) in modes and node.get("type")
    }


# ── widget 解析 ─────────────────────────────────────────────────────────
def _is_widget(spec: Any) -> bool:
    """``INPUT_TYPES()`` 的一项是否为 widget（判据见模块头 ✓）。"""
    if not isinstance(spec, (list, tuple)) or not spec:
        return False
    first = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if opts.get("forceInput"):
        return False
    if isinstance(first, (list, tuple, dict)):
        return True
    return str(first).upper() in _WIDGET_SCALAR_TYPES


def _widget_specs(node_info: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """某一节点类的 widget 输入 **(字段名, 选项)**，顺序 = required 后 optional ✓。"""
    inputs = (node_info or {}).get("input") or {}
    out: list[tuple[str, dict[str, Any]]] = []
    for group in ("required", "optional"):
        for name, spec in (inputs.get(group) or {}).items():
            if _is_widget(spec):
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                out.append((str(name), opts))
    return out


def combo_options(spec: Any) -> list[Any] | None:
    """``INPUT_TYPES()`` 的一项若是 **COMBO** ⇒ 回它的**选项表** ✓；不是 ⇒ ``None`` ✓。

    两种声明形态都收 ✓：老式 ``[["16:9 (Widescreen)", …], {}]``（**选项本身就是第一项** ✓）、
    新式 ``["COMBO", {"options": […]}]`` ✓。
    """
    if not _is_widget(spec):
        return None
    first = spec[0]
    if isinstance(first, (list, tuple, dict)):
        return list(first)
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    options = opts.get("options")
    return list(options) if isinstance(options, (list, tuple)) else None


def check_combo_choices(api_prompt: Any, object_info: dict[str, Any]) -> list[str]:
    """扫 API 格式：**凡是要喂 COMBO 的输入**，值必须在该节点声明的选项表里 ✓ ⇒ 回问题清单 ✓。

    **只报不改** ✗ —— 静默换一个串 = 静默换一个产物 ✓✗：ComfyUI 的 COMBO 是**白名单** ✓，
    表外的串会被 ``/prompt`` 400 拒 ✓，而生产里"表外的串"最常见的来源就是**抄了别人的展示标签** ✗
    （``comfy-org/workflow_templates`` 的 ``site/src/lib/demos/mmh3/config.ts`` 写 ``"21:9 (Cinemascope)"`` ✗，
    节点真枚举是 ``"21:9 (Ultrawide)"`` ✓ —— 见 ``comfy_extras/nodes_resolution.py`` ✓）。
    """
    problems: list[str] = []
    for node_id, node in sorted((api_prompt or {}).items(), key=lambda item: str(item[0])):
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        info = (object_info or {}).get(class_type)
        if not info:
            continue                                     # 类都不在 ⇒ 由 ``missing_nodes`` 那条报 ✓
        groups = info.get("input") or {}
        specs: dict[str, Any] = {}
        for group in ("required", "optional"):
            specs.update(groups.get(group) or {})
        for name, value in (node.get("inputs") or {}).items():
            if isinstance(value, list):
                continue                                 # 连线（值与选项无关 ✓）
            options = combo_options(specs.get(name))
            if options is None or value in options:
                continue
            problems.append(
                f"节点 {node_id}({class_type}) 的 {name}={value!r} 不在该节点的选项表里"
                f"（合法值举例 {options[:3]!r}）")
    return problems


def _resolve_widget_value(spec_opts: dict[str, Any], raw: Any) -> Any:
    """widget 原始值 → API 值（处理 ``control_after_generate`` 的 INT 语义 ✓）。"""
    control = spec_opts.get("control_after_generate")
    if not control:
        return raw
    mode = "fixed"
    if isinstance(raw, str) and raw.lower() in ("fixed", "randomize", "increment", "decrement"):
        # 该槽位本身是模式串（正常情况下值在上一槽 ✓，走到这里说明工作流只给了一槽）
        return _apply_control(None, raw.lower())
    return raw


def _apply_control(value: Any, mode: str) -> Any:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = int(time.time() * 1000) % (2 ** 63)
    if mode == "randomize":
        return random.randint(0, 2 ** 63 - 1)
    if mode == "increment":
        return number + 1
    if mode == "decrement":
        return number - 1
    return number


# ── 转换 ────────────────────────────────────────────────────────────────
def _link_table(graph: dict[str, Any]) -> dict[Any, tuple[str, int, str | None]]:
    """``links`` → ``{link_id: (源节点id, 源输出槽, 目标节点id)}``。

    顶层写法 ``links: [[id, origin_id, origin_slot, target_id, target_slot, type], ...]`` ✓
    （UI 图导出的就是这种 ✓），也兼容 dict 形态 ✓。
    ⚠️ **目标 id 也要取**：判断「某节点是否被消费」时，只有**目标还活着**的连线才算数 ✓ ——
    否则「指向一个已被剔除/不存在的节点的连线」会把它上游误判成「被别人消费」✗（自检实测踩到 ✓）。
    """
    table: dict[Any, tuple[str, int, str | None]] = {}
    for link in graph.get("links") or []:
        if isinstance(link, (list, tuple)) and len(link) >= 3:
            target = str(link[3]) if len(link) > 3 else None
            table[link[0]] = (str(link[1]), int(link[2]), target)
        elif isinstance(link, dict) and "id" in link:
            target_id = link.get("target_id")
            table[link["id"]] = (str(link.get("origin_id")),
                                 int(link.get("origin_slot") or 0),
                                 str(target_id) if target_id is not None else None)
    return table


def _link_parts(link: Any) -> tuple[Any, str, int, str, int] | None:
    """连线 → ``(link_id, 源id, 源槽, 目标id, 目标槽)`` ✓（顶层数组形态 + 子图 dict 形态都收 ✓）。"""
    if isinstance(link, (list, tuple)) and len(link) >= 5:
        return link[0], str(link[1]), int(link[2]), str(link[3]), int(link[4])
    if isinstance(link, dict) and "id" in link:
        return (link["id"], str(link.get("origin_id")), int(link.get("origin_slot") or 0),
                str(link.get("target_id")), int(link.get("target_slot") or 0))
    return None


def _port_inputs(node: dict[str, Any]) -> list[dict[str, Any]]:
    """节点的 ``inputs`` 数组（保证是 list ✓，便于按下标取端口 ✓）。"""
    ports = node.get("inputs")
    return ports if isinstance(ports, list) else []


def _fresh_node_id(*graphs: dict[str, Any]) -> int:
    """比图里**所有**节点 id 都大的整数（新节点用 ✓；负数（虚拟节点 ✓）不参与 ✓）。"""
    biggest = 0
    for graph in graphs:
        for node in graph.get("nodes") or []:
            try:
                biggest = max(biggest, int(node.get("id")))
            except (TypeError, ValueError):
                continue
        for sub in (graph.get("definitions") or {}).get("subgraphs") or []:
            for node in sub.get("nodes") or []:
                try:
                    biggest = max(biggest, int(node.get("id")))
                except (TypeError, ValueError):
                    continue
    return biggest + 1


def _fresh_link_id(*graphs: dict[str, Any]) -> int:
    """比图里所有连线 id 都大的整数（新连线用 ✓）。"""
    biggest = 0
    for graph in graphs:
        lists = [graph.get("links") or []]
        lists += [sub.get("links") or []
                  for sub in (graph.get("definitions") or {}).get("subgraphs") or []]
        for links in lists:
            for link in links:
                parts = _link_parts(link)
                if parts is not None:
                    try:
                        biggest = max(biggest, int(parts[0]))
                    except (TypeError, ValueError):
                        continue
    return biggest + 1


def flatten_subgraphs(graph: dict[str, Any]) -> dict[str, Any]:
    """把 **子图（subgraph）摊平进顶层图** ✓（**就地**改 ``graph`` 的 ``nodes``/``links`` ✓，
    并返回**同一个**对象 ✓）；没有子图 ⇒ 一个字都不动 ✓。

    为什么必须有这一步：新版 ComfyUI 的模板会把主节点**包进子图** ✓ ——
    ``comfy-org/workflow_templates`` 的 4 个 H3 模板（``video_minimax_h3_t2v`` /
    ``_i2v`` / ``_i2v_continuation`` / ``_r2v`` ✓）就是这样 ✓：顶层只有一个 ``type`` 等于
    **子图 id** 的节点 ✗、真正的 ``MiniMaxH3ImageToVideo`` 躺在
    ``definitions.subgraphs[*].nodes`` 里 ✗ ⇒ 不摊平的话 ``ui_to_api`` 会因为
    「类名是个 uuid、object_info 里没有」而**整图转换失败** ✗。

    结构（``ComfyUI_frontend`` 的 ``src/lib/litegraph/src/subgraph/`` ✓ + 模板实测 ✓）：

    * 子图 = ``definitions.subgraphs[]`` 的一项 ✓，自带 ``nodes``/``links``/``inputs``/``outputs`` ✓，
      以及两个**虚拟节点** ``inputNode``（id 恒为 ``-10`` ✓）与 ``outputNode``（``-20`` ✓）；
    * 子图**入参**在内部表现为「``origin_id = -10`` 的一条连线」✓ —— 哪条入参对应它，
      由 ``subgraph.inputs[i].linkIds`` 给出 ✓（**不是**按下标对 ✓）；
    * 子图**出参**同理：内部连线 ``target_id = -20`` ✓，归属由 ``subgraph.outputs[i].linkIds`` 给出 ✓；
    * 顶层节点上的 ``inputs[j].widget`` 标明该入参是**被提升的 widget** ✓ —— 它的值按
      ``subgraph.inputs`` 的**顺序**、从顶层节点的 ``widgets_values`` 里**逐个取** ✓
      （实测：T2V 模板顶层 ``widgets_values`` 13 个值 ↔ 13 个「内部端口带 widget」的入参 ✓，
      顺序**逐个对上** ✓；socket 类入参（如 ``first_frame`` ✓）**不占**槽位 ✓）；
    * 提升的 widget 若**又**被外层拉了线 ✗ ⇒ **连线赢** ✓（同 :func:`ui_to_api` 那条 ✓）。
    * ⚠️ **「子图实例」≠「子图定义」** ✗：定义在 ``definitions.subgraphs[]`` 里、键是它的 **uuid** ✓；
      而顶层图上那个**实例节点**的 id 是**另一个数**（实测 T2V 模板：定义 uuid
      ``79dd8a95-…`` ✓ 实例节点 id ``140`` ✓）⇒ 摘/接外层连线时**只认实例节点的 id** ✓
      （在 :func:`_flatten_once` 里先算 ``parent_ids`` ✓）。

    实现上的两个选择（都为了「别猜」✓）：

    1. 提升的 widget 值是**按名字**记进内部节点的 ``widget_values_override`` ✓，
       **不**去猜它在 ``widgets_values`` 里的下标 ✗ —— 下标要 object_info 的 widget 顺序才能算 ✓，
       而 `ComfyUI 的 widgets_values 并不总把 widget 列全`（如 ``UNETLoader`` 的 ``weight_dtype`` ✓）
       ⇒ 按下标写**会错位** ✗；:func:`ui_to_api` 会在**按序消费之后**再按名字覆盖 ✓。
    2. 父节点若被 **mute/bypass**（mode ≠ 0 ✓）⇒ 摊平出来的内部节点**一并**置成该 mode ✓
       （子图整体关掉时，内部不该偷偷跑 ✗）。
    """
    for _ in range(8):                                   # 子图可以套子图 ⇒ 反复摊 ✓（8 层足够 ✓）
        nodes = graph.get("nodes") or []
        subs = {str(item.get("id")): item
                for item in (graph.get("definitions") or {}).get("subgraphs") or []}
        if not subs or not any(str(node.get("type")) in subs for node in nodes):
            return graph
        flat = _flatten_once(graph, subs, nodes)
        graph["nodes"] = flat["nodes"]                   # ← **就地**换掉顶层两张表 ✓
        graph["links"] = flat["links"]
    raise WorkflowConversionError("子图嵌套超过 8 层 ⇒ 拒绝（别把图摊坏 ✗）")


def _flatten_once(graph: dict[str, Any], subs: dict[str, Any],
                  nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """摊**一层**子图（:func:`flatten_subgraphs` 的干活函数 ✓）。"""
    counter = _fresh_node_id(graph)
    fresh_link = _fresh_link_id(graph)
    out_nodes: list[Any] = []
    out_links: list[Any] = []
    #: 谁是「子图**实例**」节点（顶层那个 `type` = 子图 id 的节点 ✓）—— 摘线判据要用它的 **id** ✓，
    #: ⚠️ **不是** `subs` 的**键**（那些是子图**定义**的 uuid ✗）⇒ 曾经拿键去比连线端点 ⇒ 恒不成立 ✗✗
    #: ⇒ 外层这三条线（`115→140` 的 `width`/`height` 与 `140→92` 的 `VIDEO` ✓）被**原样留下** ✗，
    #: 而 140 已经不在摊平后的图里了 ⇒ **悬空连线** ✓✗（后果很静 ✗：`ui_to_api` 见上游不在 `live` 里
    #: 就跳过那条输入 ✓ ⇒ `SaveVideo` **丢视频输入** ✗，而 `width`/`height` 退回吃顶层
    #: `widgets_values` 里的**旧值 1344×768** ✗ —— 正是 `MiniMax_H3_Fast_T2V.json` 那个坑的同款 ✓）。
    #: 实测证据：`tests/h3_templates_test.py` 的连线自洽那几条 ✓（先红后绿 ✓）。
    parent_ids = {str(node.get("id")) for node in nodes if str(node.get("type")) in subs}
    # 顶层连线：只要一端挂在子图实例节点上 ⇒ 先摘掉 ✓（重接后重建 ✓）
    pending: list[tuple[Any, str, int, str, int]] = []
    for link in graph.get("links") or []:
        parts = _link_parts(link)
        if parts is None:
            out_links.append(link)
            continue
        if str(parts[1]) in parent_ids or str(parts[3]) in parent_ids:
            pending.append(parts)
            continue
        out_links.append(link)

    for node in nodes:
        sub = subs.get(str(node.get("type")))
        if sub is None:
            out_nodes.append(node)
            continue
        parent_id = str(node.get("id"))
        parent_mode = node.get("mode", MODE_ACTIVE)
        clones: dict[str, dict[str, Any]] = {}
        for inner in sub.get("nodes") or []:
            clone = json.loads(json.dumps(inner))        # 深拷贝 ✓（别改到源图 ✓）
            clones[str(inner.get("id"))] = clone
            clone["id"] = counter
            counter += 1
            if parent_mode != MODE_ACTIVE:
                clone["mode"] = parent_mode              # 父节点关掉 ⇒ 内部一并关 ✓
            out_nodes.append(clone)

        inner_links = {parts[0]: parts for link in (sub.get("links") or [])
                       if (parts := _link_parts(link)) is not None}
        # 入参：`subgraph.inputs[i].linkIds` ⇒ 它落到**内部哪个端口** ✓（按名 / 按连线各存一份 ✓：
        # 按名那份给「吃 widgets_values 的顺序」用 ✓，按连线那份给「接实」用 ✓（同一条入参对应多个
        # 内部端口时按连线才准 ✓））
        promoted: dict[str, tuple[str, int]] = {}
        promoted_by_link: dict[Any, tuple[str, str, int]] = {}
        for port in sub.get("inputs") or []:
            name = str(port.get("name"))
            for link_id in port.get("linkIds") or []:
                parts = inner_links.get(link_id)
                if parts is None or parts[1] != SUBGRAPH_INPUT_ID:
                    continue
                promoted_by_link[link_id] = (name, parts[3], parts[4])
                promoted.setdefault(name, (parts[3], parts[4]))
        # 出参：`subgraph.outputs[i].linkIds` ⇒ 内部**哪个输出**扛着它 ✓
        produced: dict[int, tuple[str, int]] = {}
        for index, port in enumerate(sub.get("outputs") or []):
            for link_id in port.get("linkIds") or []:
                parts = inner_links.get(link_id)
                if parts is not None and parts[3] == SUBGRAPH_OUTPUT_ID:
                    produced[index] = (parts[1], parts[2])
                    break

        # ── 提升的 widget：按 `subgraph.inputs` 顺序逐个吃顶层 widgets_values ✓ ──
        values = list(node.get("widgets_values") or [])
        override: dict[str, Any] = {}
        cursor = 0
        for port in sub.get("inputs") or []:
            name = str(port.get("name"))
            target = promoted.get(name)
            if target is None:
                continue                                  # 悬空入参（没人用 ✓）
            ports = _port_inputs(clones[target[0]])
            if target[1] >= len(ports):
                raise WorkflowConversionError(
                    f"子图节点 {parent_id} 的入参 {name!r} 指向内部端口 {target[1]}，"
                    f"但该内部节点只有 {len(ports)} 个端口 ⇒ 结构对不上 ✗")
            if ports[target[1]].get("widget") is None:
                continue                                  # socket ⇒ 不占槽位 ✓
            if cursor >= len(values):
                raise WorkflowConversionError(
                    f"子图节点 {parent_id} 的提升 widget 值不够："
                    f"入参要 {cursor + 1} 个，顶层只给了 {len(values)} 个 ⇒ 结构对不上 ✗")
            override[name] = values[cursor]
            cursor += 1

        # 哪些入参**被外层拉了线**（有 ⇒ **连线赢** ✓，提升的 widget 值丢掉 ✓ 同 :func:`ui_to_api` ✓）
        incoming: dict[str, tuple[str, int]] = {}
        for _link_id, origin_id, origin_slot, target_id, target_slot in pending:
            if target_id != parent_id:                    # 两边都是**字符串** id ✓（见 :func:`_link_parts` ✓）
                continue
            ports = _port_inputs(node)
            if target_slot >= len(ports):
                raise WorkflowConversionError(
                    f"子图节点 {parent_id} 的入参下标 {target_slot} 越界（该节点只有 {len(ports)} 个端口 ✓）"
                    f"⇒ 结构对不上 ✗")
            incoming[str(ports[target_slot].get("name"))] = (origin_id, origin_slot)

        def _new_link() -> int:
            nonlocal fresh_link
            value = fresh_link
            fresh_link += 1
            return value

        # ── 内部连线：两端换新 id ✓；碰虚拟进出节点 ⇒ 接外面 / 交给出参 ✓ ──
        for link_id, origin_id, origin_slot, target_id, target_slot in inner_links.values():
            if origin_id == SUBGRAPH_INPUT_ID:
                target = promoted_by_link.get(link_id)
                if target is None:
                    raise WorkflowConversionError(
                        f"子图节点 {parent_id} 的内部连线 {link_id!r} 来自子图入参节点、"
                        f"但不属于任何 ``subgraph.inputs[*].linkIds`` ⇒ 结构不认识 ✗")
                name, inner_id, slot = target
                port = _port_inputs(clones[inner_id])[slot]
                if name in incoming:                      # 外面接了线 ⇒ 真接一条出去 ✓
                    origin = incoming[name]
                    port["link"] = _new_link()
                    out_links.append([port["link"], origin[0], origin[1],
                                      clones[inner_id]["id"], slot])
                else:                                     # 否则这是**提升的 widget** ✓
                    port["link"] = None
                    if name in override:
                        widget = port.get("widget") or {}
                        clones[inner_id].setdefault("widget_values_override", {})[
                            str(widget.get("name") or port.get("name"))] = override[name]
                continue
            if origin_id not in clones or target_id not in clones:
                continue                                  # 出参（target = -20 ✓）/ 悬空 ⇒ 不生成 ✓
            _port_inputs(clones[target_id])[target_slot]["link"] = _new_link()
            out_links.append([clones[target_id]["inputs"][target_slot]["link"],
                              clones[origin_id]["id"], origin_slot,
                              clones[target_id]["id"], target_slot])

        # ── 外层连线「出去」：改指内部输出 ✓ ──
        for link_id, origin_id, origin_slot, target_id, target_slot in pending:
            if origin_id != parent_id:                    # 同上：两边都是字符串 ✓
                continue
            source = produced.get(origin_slot)
            if source is None:
                continue                                  # 出参悬空 ⇒ 丢 ✓
            out_links.append([link_id, clones[source[0]]["id"], source[1], target_id, target_slot])
    return {"nodes": out_nodes, "links": out_links,
            "groups": graph.get("groups") or [], "extra": graph.get("extra") or {},
            **({"definitions": graph["definitions"]} if "definitions" in graph else {})}


def ui_to_api(source: Any, object_info: dict[str, Any], *,
              modes: tuple[int, ...] = active_modes_default(),
              strict: bool = True,
              dropped_out: list[str] | None = None) -> dict[str, Any]:
    """UI 图 → **API 格式**（可直接 ``POST /prompt`` ✓）。

    ``object_info`` 是 ``GET /object_info`` 的返回（也可只给需要的那几类的子集 ✓）。
    转换不来的情况（缺类的字段表、widget 数量对不上）**直接抛错并指名节点** ✓ ——
    静默产出一个「字段错位的工作流」比失败更糟 ✗。

    ``dropped_out``（可选 list）会收到被丢掉的节点 id：**只在「该类不在 object_info 里、
    且它没有任何输出被别的活节点消费」时丢**（例如界面用的便签 ``MarkdownNote`` ✓ ——
    它在 ComfyUI 里根本不是执行节点 ✗，但它不产任何连线 ⇒ 丢掉对结果零影响 ✓）。
    若这类节点**有输出被消费**，说明它是真缺的节点包 ⇒ **照样抛错** ✓（不静默丢 ✓）。
    """
    graph = _as_graph(source)
    nodes = graph.get("nodes") or []
    links = _link_table(graph)
    live = {str(node.get("id")): node for node in nodes
            if node.get("mode", MODE_ACTIVE) in modes and node.get("type")}
    #: 谁被消费（有连线拉它的输出、且**连线目标也活着** ⇒ 它是真执行节点，不能丢 ✓）
    consumed = {origin[0] for origin in links.values()
                if origin[0] in live and (origin[2] is None or origin[2] in live)}

    api: dict[str, Any] = {}
    for node_id, node in live.items():
        class_type = str(node["type"])
        info = object_info.get(class_type)
        if info is None:
            if node_id not in consumed:
                if dropped_out is not None:
                    dropped_out.append(node_id)
                continue
            raise WorkflowConversionError(
                f"节点 {node_id} 的类 {class_type!r} 不在 object_info 里 —— "
                f"要么该类没装（自定义节点缺失 ✗），要么 object_info 传得不全 ✓")

        inputs: dict[str, Any] = {}
        # ① widget 输入：按 object_info 的顺序吃 widgets_values ✓
        specs = _widget_specs(info)
        values = list(node.get("widgets_values") or [])
        if isinstance(node.get("widgets_values"), dict):  # 少量节点存 dict ✓
            values = []
        index = 0
        for name, opts in specs:
            if index >= len(values):
                break  # 剩下的用 ComfyUI 自己的默认值 ✓（服务端会补 ✓）
            value = values[index]
            index += 1
            if opts.get("control_after_generate"):
                # 多吃一槽（模式串 ✓），并把它实现成真正的种子行为 ✓
                mode = "fixed"
                if index < len(values) and isinstance(values[index], str):
                    candidate = values[index].lower()
                    if candidate in ("fixed", "randomize", "increment", "decrement"):
                        mode = candidate
                        index += 1
                inputs[name] = _apply_control(value, mode)
                continue
            if isinstance(value, dict):
                # 动态 widget 的**占位值**（如 rgthree 的 header 标记 ✓）：不写进 inputs ✓，
                # 槽位照常消费（上面已 `index += 1` ✓）⇒ 不会让后面的值错位 ✓。
                continue
            inputs[name] = value

        # ③ **子图摊平**带出来的「提升 widget」值：**按名字**覆盖 ✓（见 :func:`flatten_subgraphs` ✓）。
        #    ⚠️ 为什么不是按下标写进 widgets_values ✗：下标取决于 object_info 的 widget 顺序 ✓，
        #    而 UI 图的 widgets_values **并不总把 widget 列全**（如 ``UNETLoader`` 的 ``weight_dtype`` ✗）
        #    ⇒ 按下标**会错位** ✓✗。按名字则与版本无关 ✓；类里没声明的名字也照写 ✓
        #    （该 widget 来自同一个 ComfyUI 的图 ✓，类自己最清楚自己有哪些 widget ✓）。
        for key, value in (node.get("widget_values_override") or {}).items():
            inputs[str(key)] = value

        # ⚠️ 判据是「**值多出来了**」：widgets_values 比 object_info 声明的 widget 还多 ⇒ 说明两边对不上
        #    （版本不匹配 / 字段错位 ✗）。两点分寸：
        #    ① 值**偏少**可以接受（服务端补默认值 ✓）；
        #    ② 多出来的值**若全是 dict**，那是「伪 widget 标记」（少数节点/第三方扩展会在
        #       widgets_values 尾部塞 header 之类、而 INPUT_TYPES 里并不声明它 ✓）⇒ **容忍** ✓，
        #       否则真机上会因为一个装饰性标记而整个任务失败 ✗。
        #    但只要有**一个非 dict** 的多余值，就仍按「字段错位」拒绝 ✓。
        leftover = values[index:]
        if strict and any(not isinstance(item, dict) for item in leftover):
            raise WorkflowConversionError(
                f"节点 {node_id}({class_type}) 的 widget 值多出来了："
                f"values={len(values)} 但 object_info 只声明 {len(specs)} 个 widget（消费 {index} 个）"
                f"，多余部分={leftover[:3]!r} ⇒ 字段可能错位，拒绝静默转换 ✗")

        # ② 连线输入：从节点声明的 inputs[].link 反查源（API 格式要 ["id", slot] ✓）
        #
        # ⚠️⚠️ **必须放在 widget 之后** ✓✗ —— 顺序就是语义：同一个名字**两边都有**时**连线赢** ✓。
        #    出处：``ComfyUI_frontend/src/utils/executionUtil.ts`` 的 ``graphToPrompt`` ✓ ——
        #    它先 ``for (widget of widgets) inputs[widget.name] = …`` ✓（对应上面 ① ✓），
        #    **再** ``for (input of node.inputs) if (!resolvedInput.widgetInfo) inputs[input.name] = [origin_id, origin_slot]`` ✓
        #    （对应下面 ② ✓）⇒ 后写者覆盖先写者 ⇒ **连线优先** ✓。
        #    为什么这不是抠字眼 ✗：UI 里**把 widget 转成输入**（拖一条线进 widget 端口 ✓）之后，
        #    ``widgets_values`` 里那个值**并不会被清掉** ✓ —— 它停在**上一次手填的旧值**上 ✓✗，
        #    于是「widget 优先」就会**把旧值发出去**、**图照样出得来** ✓✗（属"看不出来"那类 ✓）。
        #    本仓**实测踩到** ✓：``workflows/MiniMax_H3_Fast_T2V.json`` 的
        #    ``MiniMaxH3ImageToVideo.width/height`` 被 ``ResolutionSelector`` 拉着线 ✓，
        #    而 ``widgets_values`` 里躺着**旧值 1344×768** ✓ ⇒ 旧写法会把 1344×768 发出去 ✗；
        #    按连线算真值是 **1088×608** ✓（0.65MP @16:9 @multiple=32 ✓ 见 :func:`resolution_selector_size` ✓）。
        for port in node.get("inputs") or []:
            name = port.get("name")
            link_id = port.get("link")
            if not name or link_id is None:
                continue
            origin = links.get(link_id)
            if origin is None or origin[0] not in live:
                # 上游被 mute/bypass 掉（或在图外）⇒ 该输入无值 ✓
                continue
            inputs[str(name)] = [origin[0], origin[1]]

        api[node_id] = {"class_type": class_type, "inputs": inputs,
                        "_meta": {"title": node.get("title") or class_type}}
    return api


# ── 参数注入 ────────────────────────────────────────────────────────────
def set_input(prompt: dict[str, Any], class_type: str, field: str, value: Any, *,
              title_contains: str | None = None, required: bool = True) -> int:
    """把 API 工作流里某类节点（可再用标题收窄 ✓）的某输入设为 ``value``；返回改了几处。

    ⚠️ 顶层是 **API 格式**（``ui_to_api`` 的产物 ✓），别传 UI 图 ✗。
    """
    changed = 0
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") != class_type:
            continue
        if title_contains and title_contains not in str((node.get("_meta") or {}).get("title") or ""):
            continue
        if required and field not in (node.get("inputs") or {}):
            continue
        node.setdefault("inputs", {})[field] = value
        changed += 1
    return changed


def set_seed(prompt: dict[str, Any], class_type: str, value: int | str = "random",
             field: str = "noise_seed") -> int:
    """给采样种子节点设种子（``"random"`` = 每次不同 ✓）。"""
    seed = random.randint(0, 2 ** 63 - 1) if value == "random" else int(value)
    return set_input(prompt, class_type, field, seed)


def _combo_options(node_info: dict[str, Any]) -> dict[str, list[str]]:
    """从 ``object_info`` 里取出该类的 **COMBO** widget：``{字段名: [选项…]}`` ✓。

    形态：``["COMBO", {"options": [...]}]``（新）或 ``[["a","b"], {...}]``（旧）✓ —— 两种都认 ✓。
    """
    out: dict[str, list[str]] = {}
    inputs = (node_info or {}).get("input") or {}
    for group in ("required", "optional"):
        for name, spec in (inputs.get(group) or {}).items():
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            first = spec[0]
            if isinstance(first, (list, tuple)):
                out[str(name)] = [str(x) for x in first]
            elif str(first).upper() == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
                out[str(name)] = [str(x) for x in (spec[1].get("options") or [])]
    return out


def set_combo(prompt: dict[str, Any], class_type: str, wanted: str,
              object_info: dict[str, Any], *, title_contains: str | None = None) -> str | None:
    """把某类的 **COMBO** widget 改成「以 ``wanted`` 开头」的那个选项；返回选中值（没找到 ``None`` ✓）。

    为什么要 ``object_info``：参考模板用 ``ResolutionSelector«RATIO»`` 承载画幅，现值是
    ``"16:9 (Widescreen)"`` ✗ 而调用方只会给 ``"9:16"`` ✓ ⇒ **必须知道完整选项表**才能换 ✓
    （不看选项表就无法安全地改 COMBO ✗，硬塞一个「看起来像」的字符串只会让 ComfyUI 校验失败 ✗）。

    字段识别靠「**当前值 ∈ 该字段的选项集**」✓ —— 这样不必把第三方节点的字段名写死进代码 ✓。
    """
    spec = _combo_options(object_info.get(class_type) or {})
    if not spec:
        return None
    needle = wanted.strip().upper()
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") != class_type:
            continue
        if title_contains and title_contains not in str((node.get("_meta") or {}).get("title") or ""):
            continue
        inputs = node.get("inputs") or {}
        for field, options in spec.items():
            current = inputs.get(field)
            if not isinstance(current, str) or current not in options:
                continue  # 该字段不是画幅类（或当前值不在选项里）⇒ 跳过 ✓
            for option in options:
                if option.strip().upper() == needle or option.strip().upper().startswith(needle + " "):
                    inputs[field] = option
                    return option
    return None


def activate_nodes(graph: Any, *, title_contains: str | None = None,
                   class_type: str | None = None,
                   modes: tuple[int, ...] = (MODE_MUTE, MODE_BYPASS)) -> list[str]:
    """把**被关掉**的节点重新激活（mode → 0），返回被激活的节点 id 列表 ✓。

    场景：参考 I2V 模板把「可选尾帧通道」默认 **mute** 了（节点标题带 ``[DISABLED]`` ✗）——
    要用尾帧就得先把它打开 ✓（改的是**图**，不是 API 格式 ⇒ 要在 ``ui_to_api`` **之前**调 ✓）。
    """
    activated: list[str] = []
    for node in _as_graph(graph).get("nodes") or []:
        if node.get("mode", MODE_ACTIVE) not in modes:
            continue
        if title_contains and title_contains not in str(node.get("title") or ""):
            continue
        if class_type and node.get("type") != class_type:
            continue
        node["mode"] = MODE_ACTIVE
        activated.append(str(node.get("id")))
    return activated


def frames_for_seconds(seconds: float, *, fps: int = H3_FPS,
                       min_frames: int = H3_MIN_FRAMES,
                       grid: int = H3_FRAME_GRID) -> int:
    """秒 → H3 的 **length（帧数）**，向上取到 ``grid*k + min_frames`` 网格 ✓。

    来源：核心节点 ``MiniMaxH3ImageToVideo`` 的 schema（``min=5, step=17`` + tooltip
    「Frame count at 24 fps, snapped up to the model's 17k+5 grid」✓）。
    校验：``seconds=5`` ⇒ ``round(5*24)=120`` ⇒ 向上到 ``5+7×17=124`` ✓（与参考模板默认值一致 ✓）。
    """
    raw = max(int(min_frames), int(round(float(seconds or 0) * fps)))
    steps = -(-(raw - min_frames) // grid)  # 向上取整
    return min_frames + steps * grid


def seconds_for_frames(frames: int, *, fps: int = H3_FPS) -> float:
    """帧数 → 秒（24fps ✓）；用于把「实际时长」回给调用方 ✓。"""
    return round(int(frames) / float(fps), 3)


# ══════════════════════════════════════════════════════════════════════════
# 分辨率：模板里的 ResolutionSelector 会算出多大
# ══════════════════════════════════════════════════════════════════════════
#: ComfyUI 核心 ``ResolutionSelector`` 节点的**画幅选项表**（选项串 → 宽高比 ✓）。
#: 出处：``comfy_extras/nodes_resolution.py`` 的 ``AspectRatio`` 枚举 / ``ASPECT_RATIOS`` 字典 ✓。
#: ⚠️ **别拿展示名当选项串** ✗：``comfy-org/workflow_templates`` 的
#: ``site/src/lib/demos/mmh3/config.ts`` 里那几张表写的是**界面标签**（``"9:16 (Vertical)"``、
#: ``"21:9 (Cinemascope)"`` ✗）⇒ 喂给节点会被 **COMBO 校验拒** ✗（同 :func:`set_combo` 那条 ✓）。
ASPECT_RATIO_OPTIONS: dict[str, tuple[int, int]] = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}
#: ``ResolutionSelector`` 的 ``megapixels`` 下限（schema ``min=0.1`` ✓）
RESOLUTION_SELECTOR_MIN_MEGAPIXELS = 0.1
#: 节点的 ``multiple`` 默认值（schema ✓）—— H3 模板在图上显式写 **32** ✓（与 ``step=32`` 同源 ✓）
RESOLUTION_SELECTOR_MULTIPLE = 8


def _leading_number(text: str) -> float:
    """取字符串**开头**的数字（``"16"``⇒16 ✓、``"9 (Ultrawide)"``⇒9 ✓、认不出⇒0 ✓）。"""
    digits = ""
    for char in str(text).strip():
        if char.isdigit() or (char == "." and "." not in digits):
            digits += char
        else:
            break
    try:
        return float(digits)
    except ValueError:
        return 0.0


def ratio_terms(ratio: Any) -> tuple[float, float]:
    """画幅 → ``(宽比, 高比)`` ✓ —— 与「一个比值」分开 ✗：``ResolutionSelector`` 的公式是
    ``√(总像素 ÷ (w×h))`` ✓ ⇒ 要的是**两个分子分母** ✓，压扁成 ``w/h`` 会把面积项算错 ✗
    （``16:9`` 的 ``w×h = 144`` ✓ 不是 ``16/9`` ✗）。

    收：选项表里的**完整选项串** ✓（``"16:9 (Widescreen)"`` ✓，尾部括号标签丢掉 ✓）、
    ``"16:9"``/``"16：9"`` ✓、``(16, 9)`` ✓、纯浮点（按 ``w:h = v:1`` 解 ✓）。
    解析不了 ⇒ 回落 ``16:9`` ✓（与 ``geometry.parse_ratio`` 同一口径 ✓）。
    """
    fallback = (16.0, 9.0)
    if ratio is None:
        return fallback
    if isinstance(ratio, (tuple, list)) and len(ratio) == 2:
        width, height = _leading_number(ratio[0]), _leading_number(ratio[1])
        return (width, height) if width > 0 and height > 0 else fallback
    if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
        return (float(ratio), 1.0) if float(ratio) > 0 else fallback
    text = str(ratio).strip().replace("：", ":")
    if text in ASPECT_RATIO_OPTIONS:                 # 选项串先查表 ✓（省得去啃括号 ✓）
        width, height = ASPECT_RATIO_OPTIONS[text]
        return float(width), float(height)
    if ":" in text:
        head, _, tail = text.partition(":")
        width, height = _leading_number(head), _leading_number(tail)
        return (width, height) if width > 0 and height > 0 else fallback
    value = _leading_number(text)
    return (value, 1.0) if value > 0 else fallback


def resolution_selector_size(megapixels: float, ratio: Any = "16:9", *,
                             multiple: int = RESOLUTION_SELECTOR_MULTIPLE) -> tuple[int, int]:
    """像素预算 + 画幅 → ``(宽, 高)``，**逐字对齐 ComfyUI 的 ``ResolutionSelector``** ✓。

    出处（**权威实现** ✓）：``comfy_extras/nodes_resolution.py`` 的 ``ResolutionSelector.execute`` ✓ ——

    .. code-block:: python

        w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
        total_pixels = megapixels * 1024 * 1024
        scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
        width = round(w_ratio * scale / multiple) * multiple
        height = round(h_ratio * scale / multiple) * multiple

    三处**必须照抄**的细节（差一点在画面上都看不出来 ✓✗）：

    1. 预算基数是 ``megapixels × 1024²`` ✓（**不是** ``×1e6`` ✗ —— 差 **4.86%** 像素 ✓）；
    2. 取整是 **``round``** ✓（Python 的银行家舍入 ✓ —— 节点是 Python ✓ **不是**向上取 ✗）；
    3. ``multiple`` 由调用方给 ✓：节点默认 **8** ✓（:data:`RESOLUTION_SELECTOR_MULTIPLE` ✓），
       H3 模板在图上显式写 **32** ✓ ⇒ 本仓 H3 路径传 :data:`RESOLUTION_MULTIPLE` ✓。

    ⚠️ **与 ``geometry.size_for_megapixels`` 的关系**（两套都留着 ✗ **不是**重复 ✓）：

    * 本函数 = **权威**（上引源码 ✓）⇒ 凡是要「和 ComfyUI 对齐」的地方用它 ✓
      （见 :func:`expected_resolution` ✓）；
    * ``geometry.size_for_megapixels`` = 本仓早先**按参考实测输出反推**的口径 ✓，
      留在**本仓自己的像素预算**那条路上 ✓（SDXL 训练预算与实测显存/耗时绑死 ✓）。
    * 两者在**两个实测锚点上重合** ✓（``0.65MP@16:9 ⇒ 1088×608`` ✓、``0.98MP@16:9 ⇒ 1344×768`` ✓），
      所以当初「反推」看着是对的 ✓；但**一般情形会发散** ✓✗ —— 例：``1.0MP@16:9``
      ⇒ 本函数 **1376×768** ✓、``size_for_megapixels`` **1344×768** ✓
      （``tests/h3_templates_test.py`` 把**两套式子都钉住** ✓，别把一个的数当另一个的 ✓）。
    """
    w_ratio, h_ratio = ratio_terms(ratio)
    step = max(1, int(multiple))
    budget = max(RESOLUTION_SELECTOR_MIN_MEGAPIXELS, float(megapixels or 0.0))
    total_pixels = budget * 1024 * 1024                    # ← 节点写的是 1024² ✓ 不是 1e6 ✗
    scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
    width = round(w_ratio * scale / step) * step          # ← 节点是 round ✓ 不是 ceil ✗
    height = round(h_ratio * scale / step) * step
    return int(width), int(height)


def expected_resolution(api_prompt: Any, *, class_type: str = "ResolutionSelector",
                        title_contains: str | None = None) -> tuple[int, int] | None:
    """**从 API 格式的提示词里**读出「模板会算出多大」✓（对账用 ✓，与
    :func:`frames_for_seconds` 是同一类"预测模板产物"的职责 ✓）。

    模板里的真值由 ``ResolutionSelector`` 现场算 ✓ ⇒ 这里照 :func:`resolution_selector_size` 复算一遍 ✓。
    三件事要读，**读不出就回 ``None``** ✓（不猜 ✗、**不**静默回落某档 ✓）：

    1. ``aspect_ratio`` —— 必须是**字面选项串** ✓；被牵着线（值是 ``["id", slot]``）⇒ 回 ``None`` ✗
       （上游算出来的串这里解不了 ✓）；
    2. ``megapixels`` —— 允许**一跳**取自上游 ``Primitive*`` 节点的 ``value`` ✓
       （H3 模板就是这么接的 ✓：``PrimitiveFloat``（标题 ``QUALITY`` ✓）→ 端口 ✓）；
    3. ``multiple`` —— 同上；节点上没写 ⇒ 用节点的 schema 默认 8 ✓。
    """
    node = _find_node(api_prompt, class_type=class_type, title_contains=title_contains)
    if node is None:
        return None
    inputs = node.get("inputs") or {}
    ratio = inputs.get("aspect_ratio")
    if ratio is None or isinstance(ratio, list):     # 牵着线 ⇒ 解不出字面串 ✓
        return None
    megapixels = _upstream_scalar(api_prompt, inputs.get("megapixels"))
    if megapixels is None:
        return None
    multiple = _upstream_scalar(api_prompt, inputs.get("multiple"))
    return resolution_selector_size(float(megapixels), str(ratio),
                                    multiple=int(multiple or RESOLUTION_SELECTOR_MULTIPLE))


def _upstream_scalar(api_prompt: Any, value: Any) -> float | None:
    """把一个输入值解成标量 ✓：本身是数 ⇒ 直接用 ✓；``["id", slot]`` ⇒ **一跳**读上游
    ``Primitive*`` 节点的 ``value`` ✓；其它（多数值端口 / 非 Primitive / 解不出）⇒ ``None`` ✓。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not (isinstance(value, list) and len(value) == 2):
        return None
    upstream = (api_prompt or {}).get(str(value[0]))
    if not isinstance(upstream, dict):
        return None
    if not str(upstream.get("class_type") or "").startswith("Primitive"):
        return None                                  # 别的节点算出来的 ⇒ 不假装能算 ✓
    raw = (upstream.get("inputs") or {}).get("value")
    return float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None


def _find_node(api_prompt: Any, *, class_type: str | None = None,
               title_contains: str | None = None) -> dict[str, Any] | None:
    """在 API 格式里按类名 / 标题片找**第一个**节点 ✓（找不到回 ``None`` ✓）。"""
    for node in (api_prompt or {}).values():
        if not isinstance(node, dict):
            continue
        if class_type and node.get("class_type") != class_type:
            continue
        title = str((node.get("_meta") or {}).get("title") or "")
        if title_contains and title_contains not in title:
            continue
        return node
    return None


# ══════════════════════════════════════════════════════════════════════════
# 注入点发现：**图判据**（零依赖 ✓ 不靠标题、不靠节点 id ✗）
# ══════════════════════════════════════════════════════════════════════════
#: H3 的两个「主节点」类：前者吃 T2V / I2V / 续写 ✓，后者吃参考素材（R2V ✓）。
MAIN_VIDEO_CLASSES = ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo")
#: 参考图槽位的端口名前缀（真名形如 ``ref_images.ref_image_0`` ✓，见官方 R2V 模板 ✓）。
#:
#: ⚠️ 两个数别混淆 ✓✗：**节点 schema** 允许 ``ref_image_0..8``（``io.Autogrow.Input("ref_images",
#: …, min=0, max=9)`` ✓，出处 ``comfy_extras/nodes_minimax_h3.py`` 的
#: ``MiniMaxH3ReferenceToVideo.define_schema`` ✓）；**盘上那个模板**只接了 3 格
#: （``ref_image_0`` / ``_1`` 各接一个 ``LoadImage``、``_2`` 是空插座 ✓）⇒ 本模块一律**以图为准** ✓
#: （模板哪天加槽位，这里不用改 ✓；节点类还支持 ``ref_videos`` / ``ref_audios``，都在模板里没接 ✗）。
REF_IMAGE_PORT_PREFIX = "ref_images.ref_image_"


def main_video_node(source: Any) -> tuple[str, str] | None:
    """唯一的「主节点」``(id, 类名)`` ✓；**0 个或多于 1 个 ⇒ ``None``** ✗（不猜哪个是主节点 ✓）。"""
    found = [(str(node.get("id")), str(node.get("type")))
             for node in _as_graph(source).get("nodes") or []
             if str(node.get("type")) in MAIN_VIDEO_CLASSES]
    return found[0] if len(found) == 1 else None


def _node_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node.get("id")): node for node in graph.get("nodes") or []}


def _upstream_ids(graph: dict[str, Any], node_id: Any, *, depth: int = 6) -> list[str]:
    """**全部祖先**节点 id（BFS ✓ 不含自身 ✓ 去重 ✓ 深度上限防环 ✓ 顺序 = 由近到远 ✓）。"""
    links = _link_table(graph)
    nodes = _node_map(graph)
    out: list[str] = []
    seen = {str(node_id)}
    frontier = [str(node_id)]
    for _ in range(max(0, depth)):
        nxt: list[str] = []
        for current in frontier:
            for port in (nodes.get(current) or {}).get("inputs") or []:
                link_id = port.get("link")
                origin = links.get(link_id) if link_id is not None else None
                if not origin:
                    continue
                up = str(origin[0])
                if up in seen or up not in nodes:
                    continue
                seen.add(up)
                out.append(up)
                nxt.append(up)
        if not nxt:
            break
        frontier = nxt
    return out


def _unique_upstream(graph: dict[str, Any], root_id: Any, want: Any, *,
                     depth: int = 6, include_root: bool = True) -> str | None:
    """``root_id``（含自身 ✓）及其祖先里**唯一**满足 ``want(节点)`` 的 id；0 个或 ≥2 个 ⇒ ``None`` ✗。

    ⚠️ **只认唯一** ✗：不唯一说明这个端口上挂着两个候选 ⇒ 选错就是「注入到一个没人用的节点」✓✗
    （图看起来还是对的 ✓）⇒ 与其猜，不如回 ``None`` 让调用方**响亮告警** ✓。
    """
    nodes = _node_map(graph)
    candidates = ([str(root_id)] if include_root else []) + _upstream_ids(graph, root_id, depth=depth)
    hits = [cid for cid in candidates if want(nodes.get(cid) or {})]
    return hits[0] if len(hits) == 1 else None


def port_source(source: Any, node_id: Any, port: str) -> tuple[str, int] | None:
    """某个输入端口的连线源 ``(上游 id, 输出槽)`` ✓（没这个端口 / 没接线 ⇒ ``None`` ✓）。"""
    graph = _as_graph(source)
    for node in graph.get("nodes") or []:
        if str(node.get("id")) != str(node_id):
            continue
        for item in node.get("inputs") or []:
            if str(item.get("name")) != port:
                continue
            link_id = item.get("link")
            if link_id is None:
                return None
            origin = _link_table(graph).get(link_id)
            return (str(origin[0]), int(origin[1])) if origin else None
    return None


def image_source(source: Any, node_id: Any, port: str, *, depth: int = 6) -> str | None:
    """``port`` 上游（**可穿中继节点** ✓）**唯一**的 ``LoadImage`` 节点 id ✓。

    ⚠️ 为什么**不能按标题找** ✗✗：A 批模板给 ``LoadImage`` 起了 ``«START FRAME (required)»`` ✓，
    而官方模板里的 ``LoadImage`` **一个标题都没有** ✗ ⇒ 按标题找在官方模板上**恒找不到** ✓✗
    （表现是首帧**静默不生效** ✓✗）。图判据两边都对 ✓。

    ⚠️ 为什么必须**穿中继** ✓：A 批的 ``first_frame`` 根本不是 ``LoadImage`` 直连 ✗ ——
    实测链是 ``LoadImage → ImageResizeKJv2（按画幅裁 ✓）→ PreviewImage → first_frame`` ✓
    ⇒ 只认直连同样找不到 ✗（这层中继还顺手解释了「A 批首帧会被裁成画幅」✓）。
    """
    graph = _as_graph(source)
    src = port_source(graph, node_id, port)
    if src is None:
        return None
    return _unique_upstream(graph, src[0],
                            lambda node: str(node.get("type") or "") == "LoadImage",
                            depth=depth)


def scalar_source(source: Any, node_id: Any, port: str, *, prefix: str = "Primitive",
                  depth: int = 6) -> str | None:
    """``port`` 上游（**可穿表达式节点** ✓）**唯一**的 ``Primitive*`` 节点 id ✓（时长就是这么接的 ✓）。

    ⚠️ 为什么**不能按标题找** ✗✗：实测三种模板对同一个「时长」用了**三种标题** ✓ ——
    A 批 ``«DURATION (seconds)»`` ✓、官方 T2V/I2V ``«Float (duration)»`` ✓、官方 R2V ``«Float (Duration)»`` ✓
    （连大小写都不统一 ✗）⇒ 标题判据在官方模板上直接失效 ✓✗。
    ⚠️ 为什么**要穿表达式** ✓：``length`` 端口接的是 ``ComfyMathExpression``
    （``max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) + 5) % 17)`` ⇒ 17k+5 网格 ✓），
    真正可注入的秒数在它的上游 ``PrimitiveFloat`` 上 ✓ ⇒ 顺连线**找源头**才拿到对的那个 ✓
    （顺带不会误伤同图里的 ``QUALITY (megapixels)`` 那个 ``PrimitiveFloat`` ✗ —— 它喂的是分辨率 ✓）。
    """
    graph = _as_graph(source)
    src = port_source(graph, node_id, port)
    if src is None:
        return None
    return _unique_upstream(graph, src[0],
                            lambda node: str(node.get("type") or "").startswith(prefix),
                            depth=depth)


def reference_slot_names(source: Any, node_id: Any) -> list[str]:
    """主节点上的 ``ref_images.ref_image_*`` 端口名 ✓（**按槽位序** ✓ 未接线的空槽也算 ✓）。

    「按槽位序」是**排序后**的 ✓（``_2`` 不会排到 ``_10`` 前面 ✗ —— 图形状里端口的先后不可靠 ✓✗）；
    「空槽也算」也是**故意**的 ✓：调用方要按下标对齐地往 ``ref_image_k`` 上落图 ✓
    ⇒ 两张表（名字 / 上游 id）长度必须一致 ✓，空槽在 ``ref_image_ids`` 里是 ``None`` ✓（不是被挤掉 ✗）。
    """
    names: list[str] = []
    for node in _as_graph(source).get("nodes") or []:
        if str(node.get("id")) != str(node_id):
            continue
        names = [str(port.get("name")) for port in node.get("inputs") or []
                 if str(port.get("name")).startswith(REF_IMAGE_PORT_PREFIX)]
        break

    def slot(name: str) -> int:
        tail = name.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    return sorted(names, key=slot)


def video_plan(source: Any) -> dict[str, Any]:
    """把「该往哪儿注入」**一次算清** ✓（纯图判据 ⇒ 参考模板与官方模板**共用一套** ✓）。

    返回的每一项都是**从图上读出来的** ✓；读不出的**一律 ``None``** ✗（不猜 ✗、不回落某处 ✗）：

    ==========================  ==================================================
    ``form``                    形态 ✓，用的是 H3 自己的叫法 ✓（``t2va`` / ``i2va`` / ``fl2va`` /
                                ``r2va`` ✓ —— 与官方模板文件名的后缀同一套 ✓）；
                                ⚠️ 这是**图上事实**（接线状态 ✓），不是调用方想要的形态 ✗
    ``main_id`` / ``main_class`` 主节点 ✓（不唯一 ⇒ 两个都是 ``None`` ✗）
    ``prompt_owner``            提示词该写进谁 ✓：``(节点id, 字段)`` —— 主节点的 ``prompt`` 是**自身 widget**
                                时写主节点 ✓；**被拉着线**时写上游那个字符串源 ✓（R2V 就是这样 ✓，
                                往主节点上硬塞字面量会把连线**覆盖掉** ✗）
    ``duration_id``             时长该写进哪个 ``PrimitiveFloat`` ✓（见 :func:`scalar_source` ✓）
    ``resolution_id``           画幅节点 ✓（喂 ``width`` 的那个 ✓）
    ``first_frame_id``          首帧该落在哪个 ``LoadImage`` ✓
    ``last_frame_id``           尾帧同理 ✓（模板里**没接线**的槽位 ⇒ ``None`` ✗）
    ``ref_slot_names``/``ref_image_ids``  参考图槽位名 + 已接线的 ``LoadImage``（空槽 ⇒ ``None`` ✓）
    ==========================  ==================================================
    """
    graph = _as_graph(source)
    plan: dict[str, Any] = {
        "form": None, "main_id": None, "main_class": None, "prompt_owner": None,
        "prompt_wired": False, "duration_id": None, "resolution_id": None,
        "first_frame_id": None, "last_frame_id": None,
        "ref_slot_names": [], "ref_image_ids": [],
    }
    main = main_video_node(graph)
    if main is None:
        return plan
    main_id, main_class = main
    plan["main_id"], plan["main_class"] = main_id, main_class

    # 提示词：**先看主节点自己的 prompt 端口是否被接线** ✓（接线 ⇒ 写上游字符串源 ✓）
    prompt_src = port_source(graph, main_id, "prompt")
    if prompt_src is None:
        plan["prompt_owner"] = (main_id, "prompt")
    else:
        plan["prompt_wired"] = True
        owner = _unique_upstream(graph, prompt_src[0],
                                 lambda node: str(node.get("type") or "").startswith("PrimitiveString"),
                                 depth=2)
        plan["prompt_owner"] = (owner, "value") if owner else None

    plan["duration_id"] = scalar_source(graph, main_id, "length")
    width = port_source(graph, main_id, "width")
    plan["resolution_id"] = width[0] if width else None
    plan["first_frame_id"] = image_source(graph, main_id, "first_frame")
    plan["last_frame_id"] = image_source(graph, main_id, "last_frame")
    plan["ref_slot_names"] = reference_slot_names(graph, main_id)
    plan["ref_image_ids"] = [
        (port_source(graph, main_id, name) or (None, None))[0] for name in plan["ref_slot_names"]
    ]
    plan["form"] = ("r2va" if main_class == "MiniMaxH3ReferenceToVideo"
                    else "fl2va" if plan["last_frame_id"]
                    else "i2va" if plan["first_frame_id"] else "t2va")
    return plan


def set_node_input(api_prompt: dict[str, Any], node_id: Any, field: str, value: Any) -> bool:
    """**按节点 id** 写 API 图里某个节点的某个输入 ✓（节点不在图里 ⇒ ``False`` ✗ 由调用方告警 ✓）。

    ⚠️ 与 :func:`set_input` 的分工：那个按**类名**批量改 ✓（适合"图里只有一类这种节点"✓），
    这个按**图判据算出来的 id** 精改一个 ✓ —— 官方 R2V 模板里 ``prompt`` 是**连线端口** ✗，
    往主节点上写就是覆盖连线 ✗，必须精确落到上游那个字符串节点上 ✓。
    """
    node = api_prompt.get(str(node_id))
    if not isinstance(node, dict):
        return False
    node.setdefault("inputs", {})[field] = value
    return True
