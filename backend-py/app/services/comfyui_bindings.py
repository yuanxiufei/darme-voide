"""ComfyUI 工作流**参数注入**（``bindings`` 语义 ✓，自研重写 ✓）。

把「人给的参数」（提示词 / 尺寸 / 秒数 / 参考图 / 质量档…）按注册表里 entry 的
``bindings`` 写进图上的**节点输入**，产出一张可直接 ``POST /prompt`` 的图 ✓。

⚠️ 口径来源（2026-11-02 逐条核对 ✓）：上游 ``ComfyUI-Local-Infinite-Canvas``（MIT ✓）
的 ``comfyui-gateway/src/bindings.js`` ⇒ 只搬**语义**，代码自己写 ✗。

## 五处「形状对、数值错」的保真点（都实测过 ✓，改错**不报错** ✗）

1. ⭐ **``round`` 不是 Python 的 ``round``** ✗✗：JS ``Math.round(2.5) = 3`` ✓，
   Python ``round(2.5) = 2``（银行家舍入 ✗）⇒ 直接影响 ``snap32`` / ``secondsToFrames``
   的尺寸与帧数 ✗。本模块一律用 ``math.floor(x + 0.5)`` 复刻 JS 口径 ✓。
2. ⭐ **``snap16`` 用上取整、``snap32`` 用四舍五入** ✓（两者**不同** ✗）：
   ``snap16 = max(16, ceil(v/16)*16)`` ✓（尺寸只会变大，不会把用户要的 832 压成 816 ✗）；
   ``snap32 = max(32, round(v/32)*32)`` ✓。混用会让 Krea2 的 1040×1040 变 1024 ✗。
3. ⭐ **随机种子是右开区间** ✓：JS ``crypto.randomInt(1, 2147483647)`` ⇒ ``[1, 2147483647)``，
   Python ``randint`` 是闭区间 ⇒ 这里取 ``randint(1, 2147483646)`` 对齐 ✗（写成 2147483647
   会多出一个上游永远取不到的值 ✗）。
4. ⭐ **H3 帧数是 17k+5**（不是常见的 4n+1 ✗）：``secondsToFrames17`` 向上取到
   ``raw + ((5 - raw % 17) + 17) % 17`` ✓；而 Wan/LTX 类是 ``4n+1`` ✓
   （``secondsToFrames`` ✓）⇒ 两个函数**不能互相顶替** ✗。
5. ⭐ **Qwen 2.1 的步数走 sigma 插值** ✓：8 步是官方轨迹（9 个 sigma ✓），
   其他步数在**这条轨迹上线性插值** ✓ —— 不是简单改 ``steps`` ✗
   （改 steps 而不给 sigma ⇒ 画风偏离官方档 ✗）。

## 参考素材的三种挂法（决定了「能不能出图」✓）

* **文件名**（``LoadImage`` 类节点的 ``image`` 输入）：直接塞 ``"subfolder/name"`` ✓；
* **连线**（``IMAGE`` socket ✗ 不能直接塞文件名）：必须**新建一个 ``LoadImage`` 节点**
  再把它的输出连上去 ✓（``asLink`` / ``frameMode`` / ``array`` 都是这条路 ✓）；
* **多参考**（``array``）：按 ``inputPrefix + (indexBase + i)`` 铺开，
  视频还可顺带挂音轨（``audioPrefix`` → 第 2 槽 ✓）与源长度（``sourceLengthInput`` ✓）。

新节点的 id 从 **9101** 起找空位（``asLink`` 从 **9001** ✓）——
**不能**复用已有 id，否则会把原图的节点覆盖掉 ✗。

运行::

    from app.services.comfyui_bindings import build_context, apply_bindings
"""

from __future__ import annotations

import copy
import math
import random
import re
from typing import Any

from . import media_size

__all__ = [
    "BindingError",
    "QWEN21_BASE_SIGMAS",
    "apply_bindings",
    "apply_transform",
    "build_context",
    "parse_size",
    "qwen21_sigmas",
    "resolve_path",
    "seconds_to_frames",
    "seconds_to_frames_17",
]

#: Qwen-Image 2.1 的**官方 8 步 sigma 轨迹**（9 个值 ⇒ 8 步 ✓）
QWEN21_BASE_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0)

#: 种子上界（**右开**，与 ``crypto.randomInt(1, 2147483647)`` 同区间 ✓）
SEED_MAX_EXCLUSIVE = 2147483647

_SIZE_RE = re.compile(r"^(\d+)\s*[xX×*]\s*(\d+)$")


class BindingError(Exception):
    """参数注入失败（都是 **4xx** 性质：缺参考图 / 缺必填 / 首尾帧不够 ✓）。

    带 ``code`` 让 HTTP 层直接翻译成状态码，不用匹配中文文案 ✗。
    """

    def __init__(self, message: str, *, code: str = "binding_error", details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def _js_round(value: float) -> int:
    """JS ``Math.round`` 口径（**.5 一律向上** ✓）—— Python 的 ``round`` 是银行家舍入 ✗。"""
    return math.floor(value + 0.5)


def parse_size(value: Any) -> dict[str, Any] | None:
    """解析尺寸：``"832x1216"`` / ``"832×1216"`` / ``{"width":..,"height":..}`` ✓。

    认不出来返回 ``None`` ✓（由调用方决定「回落默认还是报错」✓）。
    """
    if value is None:
        return None
    if isinstance(value, dict):
        width = value.get("width")
        height = value.get("height")
        if width and height:
            return {"width": int(width), "height": int(height), "raw": f"{width}x{height}"}
        return None
    match = _SIZE_RE.match(str(value).strip())
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    return {"width": width, "height": height, "raw": f"{width}x{height}"}


def build_context(
    *,
    prompt: str = "",
    negative: str | None = None,
    size: Any = None,
    count: Any = 1,
    quality: str | None = None,
    seconds: Any = None,
    mode: str | None = None,
    background: str | None = None,
    images: list[Any] | None = None,
    uploaded_images: list[dict[str, Any]] | None = None,
    uploaded_videos: list[dict[str, Any]] | None = None,
    uploaded_audios: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
    presets: dict[str, Any] | None = None,
    source_video_frames: Any = None,
) -> dict[str, Any]:
    """构造注入用的上下文（**字段全集**即``from`` 可引用的路径 ✓）。

    ``count`` 夹在 **1–15** ✓（一次最多 15 张 ✗ 防把显存打爆）；
    ``quality`` 一律 lower+strip ✓（``"Auto"`` / ``" auto "`` 都认 ✓）；
    ``presets.quality[quality]`` 的键会**并进** ``context["quality"]`` ✓
    （于是 ``from: "quality.steps"`` 既能拿预设步数 ✓ 又保留 ``quality.value`` ✓）。
    """
    # ⭐ 客户端交来的多是**档位/比例**（"2k" / "16:9" / "auto"），不是像素 ✗
    # 直接交给 parse_size 会得 None ⇒ 图按默认尺寸出、**不报错** ✗
    parsed = parse_size(media_size.resolve_size(size, scale=(params or {}).get("scale"), ratio=(params or {}).get("ratio")))
    quality_value = str(quality or "auto").strip().lower() or "auto"
    bucket = (presets or {}).get("quality") or {}
    preset = bucket.get(quality_value) if isinstance(bucket, dict) else None
    try:
        count_value = int(count) if count is not None else 1
    except (TypeError, ValueError):
        count_value = 1
    quality_block: dict[str, Any] = {"value": quality_value}
    if isinstance(preset, dict):
        quality_block.update(preset)
    return {
        "prompt": str(prompt or ""),
        "negative": None if negative is None else str(negative),
        "size": parsed,
        "outputSize": parse_size((params or {}).get("editCanvasSize")),
        "count": max(1, min(15, count_value)),
        "quality": quality_block,
        "seconds": float(seconds) if _as_float(seconds) else None,
        "mode": str(mode) if mode else None,
        "background": str(background) if background else None,
        "images": list(images or []),
        "uploadedImages": list(uploaded_images or []),
        "uploadedVideos": list(uploaded_videos or []),
        "uploadedAudios": list(uploaded_audios or []),
        "params": dict(params) if isinstance(params, dict) else {},
        "sourceVideoFrames": source_video_frames,
    }


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def resolve_path(context: dict[str, Any], path: str | None) -> Any:
    """按 ``"size.width"`` 这种点路径取值 ✓（中途断了返回 ``None`` ✓，不抛 ✗）。"""
    if not path:
        return None
    current: Any = context
    for part in str(path).split("."):
        if current is None or not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def seconds_to_frames(seconds: Any, fps: Any = 16) -> int:
    """秒 → 帧（**Wan / LTX 类**：``4n+1`` ✓）。"""
    rate = _as_float(fps) or 16
    raw = max(1, _js_round((_as_float(seconds) or 0) * rate))
    return 4 * _js_round((raw - 1) / 4) + 1


def seconds_to_frames_17(seconds: Any, fps: Any = 24) -> int:
    """秒 → 帧（**MiniMax-H3**：向上取到 ``17k+5`` ✓，124 ≈ 5s@24fps ✓）。"""
    rate = _as_float(fps) or 24
    raw = max(5, _js_round((_as_float(seconds) or 0) * rate))
    return raw + ((5 - (raw % 17) + 17) % 17)


def _js_number_str(value: float) -> str:
    """复刻 JS ``Number(x.toFixed(6))`` 转字符串 ✓：先定 6 位，再**去掉尾随零** ✓。

    直接写 ``f"{x:.6f}"`` 会得到 ``"0.500000"``，而 JS 那边是 ``"0.5"`` ✗ ——
    这个串要**原样**塞进 ComfyUI 的 ``sigmas`` 输入框，多一个零就是另一条轨迹 ✗。
    """
    text = f"{value:.6f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def qwen21_sigmas(value: Any) -> str:
    """步数 → Qwen-Image 2.1 的 sigma 串（8 步用官方轨迹 ✓，其余按它**线性插值** ✓）。"""
    try:
        steps = int(value)
    except (TypeError, ValueError):
        steps = 8
    steps = max(1, min(50, steps))
    last = len(QWEN21_BASE_SIGMAS) - 1
    if steps == last:
        return ", ".join(_js_number_str(item) for item in QWEN21_BASE_SIGMAS)
    sigmas: list[float] = []
    for index in range(steps + 1):
        position = (index / steps) * last
        lower = math.floor(position)
        upper = min(last, lower + 1)
        fraction = position - lower
        sigmas.append(QWEN21_BASE_SIGMAS[lower] + (QWEN21_BASE_SIGMAS[upper] - QWEN21_BASE_SIGMAS[lower]) * fraction)
    return ", ".join(_js_number_str(item) for item in sigmas)


def apply_transform(value: Any, binding: dict[str, Any]) -> Any:
    """按 ``transform`` 变换取值 ✓（不认识的 transform **原样返回** ✓，不报错 ✗）。"""
    kind = binding.get("transform")
    if kind == "qwenSigmas":
        return qwen21_sigmas(value)
    if kind == "secondsToFrames":
        return seconds_to_frames(value, binding.get("fps") or 16)
    if kind == "secondsToFrames17":
        return seconds_to_frames_17(value, binding.get("fps") or 24)
    if kind == "snap16":
        number = _as_float(value) or 0
        return max(16, math.ceil(number / 16) * 16)
    if kind == "snap32":
        number = _as_float(value) or 0
        return max(32, _js_round(number / 32) * 32)
    if kind == "round":
        return _js_round(_as_float(value) or 0)
    if kind == "int":
        return int(_as_float(value) or 0)
    if kind == "invert10":
        return 10 - (_as_float(value) or 0)
    if kind == "boolean":
        return bool(value)
    return value


def _next_free_id(graph: dict[str, Any], start: int) -> str:
    """从 ``start`` 起找**没被占用**的节点 id ✓（防覆盖原图节点 ✗）。"""
    node_id = start
    while str(node_id) in graph:
        node_id += 1
    return str(node_id)


def _media_value(uploaded: dict[str, Any] | None) -> str | None:
    """ComfyUI 的素材引用名 ✓（有 subfolder 要拼上 ✗，否则找不到文件 ✓）。"""
    if not uploaded or not isinstance(uploaded, dict):
        return None
    name = uploaded.get("name")
    if not name:
        return None
    subfolder = uploaded.get("subfolder")
    return f"{subfolder}/{name}" if subfolder else str(name)


def apply_bindings(graph: dict[str, Any], entry: dict[str, Any], context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """把 context 按 entry 的 ``bindings`` 写进图，返回 ``(新图, meta)``。

    **不改原图** ✓（深拷贝后再改 ✓）—— 图是共享资产，污染一次后面全错 ✗。

    ``meta`` 目前只带 ``seed`` ✓（随机种子**必须回传给调用方**，否则用户复现不了刚才那张 ✗）。
    """
    next_graph = copy.deepcopy(graph)
    meta: dict[str, Any] = {}

    # 1) defaults：``"7.width": 832`` 这种**节点.输入**直写（只在输入已存在时写 ✓）
    for key, value in (entry.get("defaults") or {}).items():
        node_id, _, input_name = str(key).partition(".")
        node = next_graph.get(node_id)
        if isinstance(node, dict) and input_name in (node.get("inputs") or {}):
            node["inputs"][input_name] = value

    # 2) bindings
    for key, binding in (entry.get("bindings") or {}).items():
        if not isinstance(binding, dict) or not binding.get("node") or not binding.get("input"):
            continue
        node = next_graph.get(str(binding["node"]))
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if inputs is None:
            continue
        is_image_binding = binding.get("from") == "image" or binding.get("imageIndex") is not None
        if binding.get("input") not in inputs and not is_image_binding and not binding.get("array"):
            # 图上没有这个输入 ⇒ 跳过（记一笔，让调用方能看见「配了却没生效」✗）
            meta.setdefault("skipped", []).append(f"{key} → {binding['node']}.{binding['input']}")
            continue

        # 2a) 首尾帧模式：按 first/last/both 建 1–2 个 LoadImage 并连线 ✓
        if binding.get("frameMode"):
            images = list(context.get("uploadedImages") or [])[: int(binding.get("max") or 2)]
            mode = str((context.get("params") or {}).get("frameMode") or ("both" if len(images) >= 2 else "first"))
            inputs.pop(binding.get("firstInput") or "first_frame", None)
            inputs.pop(binding.get("lastInput") or "last_frame", None)
            if mode == "text":
                continue
            need = 2 if mode == "both" else 1
            if len(images) < need:
                raise BindingError(
                    f"首尾帧模式需要 {need} 张图，但请求里只有 {len(images)} 张。",
                    code="missing_reference_image",
                )
            targets: list[tuple[str, dict[str, Any]]] = []
            if mode in ("first", "both"):
                targets.append((binding.get("firstInput") or "first_frame", images[0]))
            if mode == "last":
                targets.append((binding.get("lastInput") or "last_frame", images[0]))
            if mode == "both":
                targets.append((binding.get("lastInput") or "last_frame", images[1]))
            for input_name, uploaded in targets:
                load_id = _next_free_id(next_graph, 9101)
                value = _media_value(uploaded)
                if value is None:
                    continue
                next_graph[load_id] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": value},
                    "_meta": {"title": f"Gateway Frame {input_name}"},
                }
                inputs[input_name] = [load_id, 0]
            continue

        # 2b) 单视频（非数组）：直接塞文件名 ✓
        if binding.get("from") == "video" and not binding.get("array"):
            index = binding.get("videoIndex") if isinstance(binding.get("videoIndex"), int) else 0
            value = _media_value((context.get("uploadedVideos") or [])[index] if index < len(context.get("uploadedVideos") or []) else None)
            if value is not None:
                inputs[binding["input"]] = value
            elif binding.get("required"):
                raise BindingError(f"工作流 {entry.get('name') or entry.get('id')} 缺少输入视频。", code="missing_reference_media")
            continue

        # 2c) 多参考素材（array）：按前缀铺开，必要时建 Load* 节点并连线 ✓
        if binding.get("array"):
            is_video = binding.get("loadType") == "video" or binding.get("from") == "video"
            is_audio = binding.get("loadType") == "audio" or binding.get("from") == "audio"
            media = list((context.get("uploadedVideos") if is_video else context.get("uploadedAudios") if is_audio else context.get("uploadedImages")) or [])
            selected = media[: int(binding.get("max") or 8)]
            minimum = int(binding.get("min") or 0)
            if len(selected) < minimum:
                raise BindingError(
                    f"工作流需要至少 {minimum} 个参考素材，但请求里只有 {len(selected)} 个。",
                    code="missing_reference_media",
                )
            prefix = str(binding.get("inputPrefix") or binding.get("input") or "ref_images.ref_image_")
            index_base = binding.get("indexBase") if isinstance(binding.get("indexBase"), int) else 0
            for existing in [name for name in inputs if str(name).startswith(prefix)]:
                inputs.pop(existing, None)
            for offset, uploaded in enumerate(selected):
                load_id = _next_free_id(next_graph, 9101)
                value = _media_value(uploaded)
                if value is None:
                    continue
                if is_video and binding.get("loader") == "native":
                    next_graph[load_id] = {
                        "class_type": "LoadVideo",
                        "inputs": {"file": value},
                        "_meta": {"title": f"Gateway Video {offset + 1}"},
                    }
                    components_id = _next_free_id(next_graph, 9101)
                    next_graph[components_id] = {
                        "class_type": "GetVideoComponents",
                        "inputs": {"video": [load_id, 0]},
                        "_meta": {"title": f"Gateway Video Frames {offset + 1}"},
                    }
                    inputs[f"{prefix}{index_base + offset}"] = [components_id, 0]
                    if binding.get("sourceLengthInput"):
                        frames = _js_round(_as_float(context.get("sourceVideoFrames")) or 0)
                        if frames > 0:
                            inputs[binding["sourceLengthInput"]] = frames
                    continue
                if is_video:
                    next_graph[load_id] = {
                        "class_type": "VHS_LoadVideo",
                        "inputs": {
                            "video": value,
                            "force_rate": int(binding.get("forceRate") or 0),
                            "custom_width": 0,
                            "custom_height": 0,
                            "frame_load_cap": 0,
                            "skip_first_frames": 0,
                            "select_every_nth": 1,
                            "format": "AnimateDiff",
                        },
                        "_meta": {"title": f"Gateway Video {offset + 1}"},
                    }
                elif is_audio:
                    next_graph[load_id] = {
                        "class_type": "LoadAudio",
                        "inputs": {"audio": value},
                        "_meta": {"title": f"Gateway Audio {offset + 1}"},
                    }
                else:
                    next_graph[load_id] = {
                        "class_type": "LoadImage",
                        "inputs": {"image": value},
                        "_meta": {"title": f"Gateway Ref {offset + 1}"},
                    }
                inputs[f"{prefix}{index_base + offset}"] = [load_id, 0]
                if is_video and binding.get("audioPrefix"):
                    inputs[f"{binding['audioPrefix']}{index_base + offset}"] = [load_id, 2]
                if is_video and binding.get("sourceLengthInput"):
                    slot = binding.get("sourceLengthOutput") if isinstance(binding.get("sourceLengthOutput"), int) else 1
                    inputs[binding["sourceLengthInput"]] = [load_id, slot]
            continue

        # 2d) 单图：可塞文件名，也可建 LoadImage 连线（IMAGE socket 必须连线 ✓）
        if is_image_binding:
            index = binding.get("imageIndex") if isinstance(binding.get("imageIndex"), int) else 0
            gallery = context.get("uploadedImages") or []
            value = _media_value(gallery[index] if index < len(gallery) else None)
            if value is not None:
                if binding.get("asLink"):
                    start = int(binding["loadNodeId"]) if str(binding.get("loadNodeId") or "").isdigit() else 9001
                    load_id = binding.get("loadNodeId") if binding.get("loadNodeId") else _next_free_id(next_graph, start)
                    load_id = str(load_id)
                    next_graph[load_id] = {
                        "class_type": "LoadImage",
                        "inputs": {"image": value},
                        "_meta": {"title": f"Gateway Load {index + 1}"},
                    }
                    inputs[binding["input"]] = [load_id, 0]
                else:
                    inputs[binding["input"]] = value
            elif binding.get("required"):
                raise BindingError(
                    f"工作流 {entry.get('name') or entry.get('id')} 需要第 {index + 1} 张参考图，但请求里没有。",
                    code="missing_reference_image",
                )
            continue

        # 2e) 常规参数
        value = resolve_path(context, binding.get("from")) if binding.get("from") else context.get(key)
        if _is_blank(value) and binding.get("random"):
            value = random.randint(1, SEED_MAX_EXCLUSIVE - 1)
        if _is_blank(value) and binding.get("default") is not None:
            value = binding["default"]
        if value is None:
            if binding.get("required"):
                raise BindingError(f"缺少必填参数：{key}", code="missing_parameter")
            continue
        mapping = binding.get("map")
        if isinstance(mapping, dict) and str(value) in mapping:
            value = mapping[str(value)]
        value = apply_transform(value, binding)
        inputs[binding["input"]] = value
        if key == "seed":
            meta["seed"] = value

    # 3) outputSize：把出图尺寸写进尺寸节点，并把 latent 源头接过去 ✓
    entry_output = entry.get("outputSize")
    if isinstance(entry_output, dict):
        output_size = context.get("size") if entry_output.get("fromSize") else context.get("outputSize")
        resolution = _as_float((context.get("params") or {}).get("resolution")) or 1024
        native = max(256, _js_round(resolution / 32) * 32)
        skip = bool(
            entry_output.get("skipNativeSquare")
            and output_size
            and output_size.get("width") == output_size.get("height")
            and _js_round(output_size["width"] / 32) * 32 == native
        )
        size_node = next_graph.get(str(entry_output.get("node")))
        latent_node = next_graph.get(str(entry_output.get("latentNode")))
        if output_size and not skip and isinstance(size_node, dict) and isinstance(latent_node, dict):
            size_inputs = size_node.setdefault("inputs", {})
            size_inputs["width"] = max(256, _js_round((_as_float(output_size.get("width")) or 0) / 32) * 32)
            size_inputs["height"] = max(256, _js_round((_as_float(output_size.get("height")) or 0) / 32) * 32)
            latent_node.setdefault("inputs", {})[entry_output.get("latentInput") or "latent_image"] = [
                str(entry_output.get("latentSource") or entry_output.get("node")),
                0,
            ]

    return next_graph, meta


def _is_blank(value: Any) -> bool:
    """``undefined / null / ""`` 的并集 ✓（``0`` 与 ``False`` **不算空** ✓）。"""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False
