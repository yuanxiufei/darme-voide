"""ComfyUI 工作流：**UI 图 → API 格式** 转换 + 参数注入（零依赖，纯函数）。

## 为什么必须有这一步

ComfyUI 里流通的 workflow 有两种形态，**互不通用** ✗：

* **UI 格式**（``{"nodes": [...], "links": [...]}``）—— 界面里能打开、能看连线，但**不能** POST；
  参考项目（``reference/minimax-h3-comfyui/workflows/MiniMax_H3_Fast_T2V.json`` 等）导出的都是这种 ✓。
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
import random
import time
from pathlib import Path
from typing import Any

__all__ = [
    "MODE_ACTIVE",
    "MODE_MUTE",
    "WorkflowConversionError",
    "activate_nodes",
    "active_modes_default",
    "frames_for_seconds",
    "find_nodes",
    "node_classes",
    "seconds_for_frames",
    "set_combo",
    "set_input",
    "set_seed",
    "ui_to_api",
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


# ── 读取 ────────────────────────────────────────────────────────────────
def _as_graph(source: Any) -> dict[str, Any]:
    """接收 路径 / dict / JSON 字符串 ⇒ UI 图 dict。"""
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, Path)):
        return json.loads(Path(source).read_text(encoding="utf-8"))
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
        # ① 连线输入：从节点声明的 inputs[].link 反查源（API 格式要 ["id", slot] ✓）
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

        # ② widget 输入：按 object_info 的顺序吃 widgets_values ✓
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
