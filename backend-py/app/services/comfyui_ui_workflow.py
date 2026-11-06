"""ComfyUI **UI 格式 → API 格式**工作流转换（``ui-workflow`` 语义 ✓，自研重写 ✓）。

为什么需要这一层：在 ComfyUI 界面里「保存/导出」得到的是 **UI 格式**
（``nodes[]`` + ``links[]`` + ``widgets_values``，带坐标、注释、组开关…），
而 ``POST /prompt`` 只吃 **API 格式**（``{"3": {"class_type":…, "inputs":{…}}}``）。
没有转换器 ⇒ 用户手头最常见那种图**全是废图**（本仓原本只能判定、不能转 ✗）。

⚠️ 口径来源（2026-11-02 逐条核对 ✓）：上游 ``ComfyUI-Local-Infinite-Canvas``（MIT ✓）
的 ``comfyui-gateway/src/ui-workflow.js`` ⇒ 只搬**语义与顺序表**，代码自己写 ✗。

## 六处「转得出来但图是错的」的保真点

1. ⭐ **UI 辅助节点必须丢掉，且不能当成「缺自定义节点」** ✗：``Note`` /
   ``MarkdownNote`` / rgthree 的组开关旁路器… 只在前端存在，后端没有它们；
   若按 ``object_info`` 查不到就报「缺节点」，一张好图会被误判成装不全 ✗。
   判据是「名字命中 **且** 没有接任何连线」✓（接了线的同名节点是**真节点** ✓）。
2. ⭐ **``Reroute`` 与 ``PrimitiveNode`` 要穿透** ✓：它们是连线中转，不是执行节点；
   转出来时要把它们**塌缩成上游的实际端点**（Reroute）或**常量值**（Primitive）✓，
   否则下游拿到的是一个不存在的节点 id ✗。
3. ⭐ **``mode`` 两种语义** ✓：``2``=静音（整节点不执行 ⇒ 丢弃 ✓）、
   ``4``=旁路（**不执行但要把输入直接接到输出** ✓，不是删掉 ✗ ——
   删了会让下游断链）。旁路时若同类型输入有多个，只取第一个并**记警告** ✓。
4. ⭐ **widget 值按** ``WIDGET_ORDER`` **对号入座** ✓：老式 UI 图只有
   ``widgets_values``（**位置数组**，没有名字）；而 ``object_info`` 里的顺序
   对这些节点并不显然（``KSampler`` 是 ``seed, control_after_generate, steps, …`` ✓）。
   对不上号 ⇒ ``steps`` 拿到 seed 的值，**不报错、图照样出、只是完全不对** ✗✗。
   有 ``widgets_values_named`` 时优先用名字 ✓（更可靠）。
5. ⭐ **combo 值要把「显示标签」还原成选项值** ✓：界面上显示
   ``Ref2VA — 参考生音视频``，API 只认 ``ref2va`` 这类**短值**；
   用「选项与标签互为前缀」还原 ✓，还原不了就**原样留着**（留着至少能报出可读错误 ✗）。
6. ⭐ **转换后要剪掉游离节点** ✓：ComfyUI 会校验**提交的每一个**节点，
   一条没连到输出的旁支若引用了缺失模型/节点 ⇒ **整次生成失败** ✗
   （症状是「明明改好了主链路还是报错」）。剪枝只保留**能回溯到输出**的那张子图，
   多个输出互不相连时取**覆盖节点最多**的那条 ✓。

未搬：``detectFreeBindings``（从图里**猜**可绑参数）—— 本仓已有 registry 的
``bindings`` 契约作为绑定真源 ✓，再引入一套猜测口径会变成两个真源 ✗。

运行::

    from app.services.comfyui_ui_workflow import convert_ui_workflow, prune_unreachable_nodes
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

__all__ = [
    "WIDGET_ORDER",
    "UiWorkflowError",
    "canonical_aspect_ratio",
    "convert_ui_workflow",
    "is_ui_workflow",
    "prune_unreachable_nodes",
]

#: 纯前端节点（后端没有，必须从转换结果里丢掉 ✓）
NOTE_TYPES = frozenset(
    {
        "Note",
        "MarkdownNote",
        "Note Plus (mtb)",
        "Label (rgthree)",
        "Fast Groups Bypasser (rgthree)",
        "Fast Groups Muter (rgthree)",
        "Bookmark (rgthree)",
    }
)
REROUTE_TYPE = "Reroute"
PRIMITIVE_TYPE = "PrimitiveNode"
#: ``mode``: 2=静音（丢弃 ✓） / 4=旁路（**直连** ✓，不是丢弃 ✗）
MUTE_MODE = 2
BYPASS_MODE = 4

_BASIC_WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"})

#: widget 顺序表：``object_info`` 推不出顺序的节点，按这张表对号入座 ✓
WIDGET_ORDER: dict[str, tuple[str, ...]] = {
    "KSampler": ("seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"),
    "KSamplerAdvanced": (
        "add_noise", "noise_seed", "control_after_generate", "steps", "cfg",
        "sampler_name", "scheduler", "start_at_step", "end_at_step", "return_with_leftover_noise",
    ),
    "LoadImage": ("image", "upload"),
    "SaveVideo": ("filename_prefix", "format", "format.codec", "codec"),
    "VHS_VideoCombine": ("frame_rate", "loop_count", "filename_prefix", "format", "pingpong", "save_output"),
    "MiniMaxH3ImageToVideo": ("prompt", "width", "height", "length"),
    "MiniMaxH3ReferenceToVideo": ("prompt", "width", "height", "length", "ref_image_size"),
}

_IMAGE_OUTPUT_RE = re.compile(r"^SaveImage$|^PreviewImage$|save.*(?:image|img)|(?:image|img).*save|(?:image|img).*output|(?:save|output).*(?:image|img)", re.IGNORECASE)
_VIDEO_OUTPUT_RE = re.compile(r"SaveVideo|VHS_VideoCombine|SaveWebM", re.IGNORECASE)
_ROOT_RE = re.compile(r"save|preview|output|combine", re.IGNORECASE)
_ASPECT_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*[:/x]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_UI_ONLY_RE = re.compile(r"ignore.*group|group.*ignore|fast.*group|group.*bypass|group.*mute", re.IGNORECASE)


class UiWorkflowError(Exception):
    """转换失败（都是 **4xx** 性质：不是 UI 图 / 含子图 / 缺节点定义 / 转完为空 ✓）。"""

    def __init__(self, message: str, *, code: str = "ui_workflow_error", details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


class _Spec(NamedTuple):
    name: str
    type: str
    opts: Any
    required: bool


def is_ui_workflow(value: Any) -> bool:
    """是不是 UI 格式（``nodes`` + ``links`` 都是数组 ✓）。"""
    return isinstance(value, dict) and isinstance(value.get("nodes"), list) and isinstance(value.get("links"), list)


def canonical_aspect_ratio(value: Any) -> str:
    """``"16:9 (Widescreen)"`` → ``"16:9"`` ✓（认不出返回 ``""`` ✓）。"""
    match = _ASPECT_RE.match(str(value or "").strip())
    if not match:
        return ""
    return f"{float(match.group(1)):g}:{float(match.group(2)):g}"


def is_image_output_class(class_type: Any) -> bool:
    return bool(_IMAGE_OUTPUT_RE.search(str(class_type or "")))


def is_video_output_class(class_type: Any) -> bool:
    return bool(_VIDEO_OUTPUT_RE.search(str(class_type or "")))


def _normalize_type(value: Any) -> str:
    if isinstance(value, list):
        return "COMBO"
    if isinstance(value, dict) and isinstance(value.get("options"), list):
        return "COMBO"
    return str(value or "")


def _is_widget_type(value: Any) -> bool:
    text = _normalize_type(value)
    return text in _BASIC_WIDGET_TYPES or "combo" in text.lower()


def _input_specs(object_info: dict[str, Any] | None, class_type: Any) -> list[_Spec]:
    """从 ``object_info`` 取输入规格（required 在前 ✓，与源一致）。"""
    info = (object_info or {}).get(str(class_type or ""))
    if not isinstance(info, dict) or not isinstance(info.get("input"), dict):
        return []
    specs: list[_Spec] = []
    for group in ("required", "optional"):
        for name, value in (info["input"].get(group) or {}).items():
            if isinstance(value, list):
                specs.append(_Spec(name, _normalize_type(value[0] if value else None), value[1] if len(value) > 1 else {}, group == "required"))
            else:
                specs.append(_Spec(name, _normalize_type((value or {}).get("type")), value or {}, group == "required"))
    return specs


def _normalize_combo_value(spec: _Spec | None, value: Any) -> Any:
    """把界面显示的**长标签**还原成 API 认的**短选项值** ✓（还原不了原样返回 ✓）。"""
    if not isinstance(value, str) or spec is None:
        return value
    options = spec.opts.get("options") if isinstance(spec.opts, dict) else None
    if not isinstance(options, list) or not options or value in options:
        return value
    text = value.strip()
    for option in options:
        if isinstance(option, str) and (option == text or text.startswith(option) or option.startswith(text)):
            return option
    return value


def _is_ui_only_node(node: dict[str, Any]) -> bool:
    """前端辅助节点（组开关/旁路器…）✓ —— 但**接了线**的同名节点是真节点 ✗。"""
    node_type = str(node.get("type") or "")
    if node_type in NOTE_TYPES:
        return True
    if not _UI_ONLY_RE.search(node_type):
        return False
    ports = list(node.get("inputs") or []) + list(node.get("outputs") or [])
    return not any(isinstance(port, dict) and port.get("link") is not None for port in ports)


def _primitive_value(node: dict[str, Any]) -> Any:
    named = node.get("widgets_values_named")
    if isinstance(named, dict):
        for key in ("value", "primitive"):
            if key in named:
                return named[key]
        if named:
            return next(iter(named.values()))
    values = node.get("widgets_values")
    return values[0] if isinstance(values, list) and values else None


def _normalize_links(links: Any) -> dict[str, dict[str, Any]]:
    """``links`` 两种写法都认 ✓：数组 `[id, from, fromSlot, to, toSlot, type]` 与对象 `{id, origin_id, …}`。"""
    out: dict[str, dict[str, Any]] = {}
    for raw in links or []:
        if isinstance(raw, list):
            item = {
                "id": raw[0] if len(raw) > 0 else None,
                "from": raw[1] if len(raw) > 1 else None,
                "fromSlot": raw[2] if len(raw) > 2 else 0,
            }
        elif isinstance(raw, dict):
            item = {"id": raw.get("id"), "from": raw.get("origin_id"), "fromSlot": raw.get("origin_slot") or 0}
        else:
            continue
        if item["id"] is None:
            continue
        out[str(item["id"])] = {"from": str(item["from"]), "fromSlot": int(item["fromSlot"] or 0)}
    return out


def convert_ui_workflow(ui: dict[str, Any], object_info: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[str]]:
    """UI 格式 → API 格式 ✓，返回 ``(图, 警告)``。

    ``object_info``（``GET /object_info``）**建议给**：没有它只能靠
    ``widgets_values_named`` 与顺序表，老式图会缺字段 ✗。
    """
    if not is_ui_workflow(ui):
        raise UiWorkflowError("不是 ComfyUI UI 工作流（需要 nodes + links）", code="not_ui_workflow")
    if isinstance(ui.get("definitions"), dict) and (ui["definitions"].get("subgraphs") or []):
        raise UiWorkflowError("暂不支持包含 Subgraph 的工作流；请在 ComfyUI 里展开子图后保存，或导出 API 格式", code="subgraph_unsupported")
    raw_nodes = ui.get("nodes") or []
    if any(re.match(r"^workflow>|^Subgraph", str(node.get("type") or ""), re.IGNORECASE) for node in raw_nodes if isinstance(node, dict)):
        raise UiWorkflowError("暂不支持子图/组节点工作流；请展开后保存，或导出 API 格式", code="subgraph_unsupported")

    nodes = {str(node.get("id")): node for node in raw_nodes if isinstance(node, dict)}
    links = _normalize_links(ui.get("links"))
    warnings: list[str] = []
    missing_classes: list[str] = []

    def resolve_link(link_id: Any, seen: frozenset[str]) -> dict[str, Any] | None:
        link = links.get(str(link_id))
        if not link:
            return None
        return resolve_endpoint(link["from"], link["fromSlot"], seen)

    def resolve_endpoint(node_id: Any, slot: Any, seen: frozenset[str]) -> dict[str, Any] | None:
        key = f"{node_id}:{slot}"
        if key in seen:
            return None
        seen = seen | {key}
        node = nodes.get(str(node_id))
        if not node:
            return None
        node_type = str(node.get("type") or "")
        if node_type == PRIMITIVE_TYPE:
            return {"value": _primitive_value(node)}
        if node_type == REROUTE_TYPE:
            linked = next((port for port in (node.get("inputs") or []) if isinstance(port, dict) and port.get("link") is not None), None)
            return resolve_link(linked["link"], seen) if linked else None
        mode = int(node.get("mode") or 0)
        if mode == MUTE_MODE:
            return None
        if mode == BYPASS_MODE:
            outputs = node.get("outputs") or []
            output_type = outputs[int(slot or 0)].get("type") if int(slot or 0) < len(outputs) and isinstance(outputs[int(slot or 0)], dict) else None
            candidates = [
                port for port in (node.get("inputs") or [])
                if isinstance(port, dict) and port.get("type") == output_type and port.get("link") is not None
            ]
            if len(candidates) > 1:
                warnings.append(f"节点「{node.get('title') or node_type}」有多个 {output_type} 输入，旁路时只取第一个。")
            if candidates:
                return resolve_link(candidates[0]["link"], seen)
            widget = next(
                (
                    port for port in (node.get("inputs") or [])
                    if isinstance(port, dict) and port.get("type") == output_type and port.get("widget")
                    and isinstance(node.get("widgets_values_named"), dict) and port.get("name") in node["widgets_values_named"]
                ),
                None,
            )
            if widget is not None:
                return {"value": node["widgets_values_named"][widget["name"]]}
            warnings.append(f"旁路节点「{node.get('title') or node_type}」的 {output_type} 输出没有同类型输入，相关输入会被省略。")
            return None
        return {"link": [str(node_id), int(slot or 0)]}

    workflow: dict[str, Any] = {}
    for node_id, node in nodes.items():
        node_type = str(node.get("type") or "")
        if node_type in NOTE_TYPES or node_type in (REROUTE_TYPE, PRIMITIVE_TYPE):
            continue
        mode = int(node.get("mode") or 0)
        if mode in (MUTE_MODE, BYPASS_MODE):
            continue
        if _is_ui_only_node(node):
            warnings.append(f"已忽略 UI 辅助节点「{node.get('title') or node_type}」")
            continue
        specs = _input_specs(object_info, node_type)
        if not specs:
            if node_type not in missing_classes:
                missing_classes.append(node_type)
            continue
        spec_by_name = {spec.name: spec for spec in specs}
        ui_inputs = {port.get("name"): port for port in (node.get("inputs") or []) if isinstance(port, dict)}
        named = node.get("widgets_values_named") if isinstance(node.get("widgets_values_named"), dict) else None
        values = node.get("widgets_values") if isinstance(node.get("widgets_values"), list) else []
        input_order = list(WIDGET_ORDER.get(node_type) or [port.get("widget", {}).get("name") for port in (node.get("inputs") or []) if isinstance(port, dict) and port.get("widget")])
        fallback_order = input_order or [spec.name for spec in specs if _is_widget_type(spec.type)]

        inputs: dict[str, Any] = {}

        def set_resolved(name: Any, resolved: dict[str, Any] | None) -> None:
            if resolved is None:
                return
            if "value" in resolved:
                inputs[name] = resolved["value"]
            elif resolved.get("link"):
                inputs[name] = resolved["link"]

        for port in node.get("inputs") or []:
            if not isinstance(port, dict) or port.get("link") is None:
                continue
            resolved = resolve_link(port["link"], frozenset())
            if resolved is None:
                if spec_by_name.get(port.get("name")) and spec_by_name[port.get("name")].required:
                    warnings.append(f"节点「{node.get('title') or node_type}」的输入 {port.get('name')} 无法解析。")
                continue
            set_resolved(port.get("name"), resolved)

        if named:
            for name, value in named.items():
                if value is None or name in inputs:
                    continue
                if name not in spec_by_name and name not in ui_inputs:
                    continue
                spec = spec_by_name.get(name)
                port_type = spec.type if spec else (ui_inputs.get(name) or {}).get("type")
                if not _is_widget_type(port_type):
                    continue
                inputs[name] = _normalize_combo_value(spec, value)
        else:
            for index, name in enumerate(fallback_order):
                if not name or name in ("control_after_generate", "upload") or name in inputs:
                    continue
                spec = spec_by_name.get(name)
                if not spec or not _is_widget_type(spec.type):
                    continue
                if index < len(values):
                    inputs[name] = _normalize_combo_value(spec, values[index])

        # 有些自定义节点把 prompt/width/height 只放在 widget 值里（不在 node.inputs 上）
        for spec in specs:
            if spec.name in inputs or not _is_widget_type(spec.type):
                continue
            if named and spec.name in named:
                inputs[spec.name] = _normalize_combo_value(spec, named[spec.name])

        entry: dict[str, Any] = {"class_type": node_type, "inputs": inputs}
        title = node.get("title") or (node.get("_meta") or {}).get("title")
        if title:
            entry["_meta"] = {"title": title}
        workflow[node_id] = entry

    if missing_classes:
        raise UiWorkflowError(
            f"缺少节点定义：{', '.join(missing_classes[:8])}。请先在 ComfyUI 安装对应自定义节点。",
            code="missing_class",
            details=missing_classes,
        )
    if not workflow:
        raise UiWorkflowError("转换后没有可执行节点", code="empty_result")
    return workflow, warnings


def _link_target(value: Any) -> str | None:
    return str(value[0]) if isinstance(value, (list, tuple)) and len(value) >= 2 and isinstance(value[0], str) else None


def _reachable(workflow: dict[str, Any], roots: list[str]) -> set[str]:
    found: set[str] = set()
    stack = list(roots)
    while stack:
        node_id = stack.pop()
        if not node_id or node_id in found:
            continue
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            continue
        found.add(node_id)
        for value in (node.get("inputs") or {}).values():
            target = _link_target(value)
            if target:
                stack.append(target)
    return found


def prune_unreachable_nodes(workflow: dict[str, Any]) -> dict[str, Any]:
    """剪掉**回溯不到输出节点**的游离节点 ✓（不剪会被 ComfyUI 一并校验而整次失败 ✗）。

    多个输出互不相连时，只保留**覆盖节点最多**的那条链路 ✓（源注释：否则残留的
    ``SaveImage`` 仍会被执行并因缺模型/缺输入而失败）。
    """
    if not workflow:
        return workflow
    roots = [node_id for node_id, node in workflow.items() if isinstance(node, dict) and _ROOT_RE.search(str(node.get("class_type") or ""))]
    if not roots:
        return workflow
    best: set[str] = set()
    for root in roots:
        reachable = _reachable(workflow, [root])
        if len(reachable) > len(best):
            best = reachable
    if not best or len(best) == len(workflow):
        return workflow
    return {node_id: node for node_id, node in workflow.items() if node_id in best}
