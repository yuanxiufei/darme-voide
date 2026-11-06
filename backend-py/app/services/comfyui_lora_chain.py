"""LoRA 链注入：把 ``loras`` 参数**串到模型加载器后面**，返回克隆后的工作流（``lora-chain`` 语义 ✓，自研重写 ✓）。

为什么要有这一层：ComfyUI 工作流里 LoRA 是**节点**，不是参数 ✗ —— 用户在界面上加不了节点，
只能给后端一份「要哪几个 LoRA、各自多强」的清单 ✓。把清单**变成节点链**这件事必须在这里做，
否则前端只能自己拼图（拼错一个 ``model`` 连线就是「出图了但没用 LoRA」✗，不报错 ✗）。

## 五处「拼得出图、但 LoRA 不生效」的地方

1. ⭐ **模型源要按"被谁引用"选，不能按节点 id 选** ✓：一份图里可能同时有 ``CheckpointLoaderSimple``
   与 ``UNETLoader``（前者只给 CLIP/VAE），拿错那个当源头 ⇒ 链子挂在一个没人用的模型上 ✗。
   先数「下游有几处引用第 0 槽」，多的优先；平票时 ``UNETLoader`` 优先 ✓（那才是 DiT 本体 ✓）。
2. ⭐ **下游要**全部**重接** ✓：只改第一处 ⇒ 有的分支走 LoRA、有的不走，出图一半生效 ✗。
3. ⭐ **链是"串联"** ✓：第 2 个 LoRA 的 ``model`` 必须指向第 1 个 LoRA 的输出
   （写成两个都指模型加载器 ⇒ 只有后一个生效，前一个被**静默丢掉** ✗✗）。
4. ⭐ **新节点 id 从 9000 起** ✓（源口径）：和用户图里的 1/2/3 撞号会让整份图**互相顶替** ✗。
   取 ``max(本身最大值+1, 9000)``，且**跳过已占用的号** ✓。
5. ⭐ **没有 LoRA 时返回原对象、不克隆** ✓：克隆会让调用方的引用失效（前端缓存比对不上 ✗），
   也白贵一次深拷贝 ✓。

⚠️ 校验是「宁缺毋错」✓：``filter`` 不匹配、``available`` 里没有 ⇒ **报错**，不静默跳过
（静默跳过 = 用户以为加了 LoRA、其实没加 ✗）。

运行::

    from app.services.comfyui_lora_chain import apply_lora_chain
    graph, applied = apply_lora_chain(graph, {"loras": [{"name": "krea2_turbo.safetensors", "strength": 0.8}]})
"""

from __future__ import annotations

import copy
import re
from typing import Any

__all__ = [
    "LoraChainError",
    "apply_lora_chain",
    "find_model_source",
    "normalize_loras",
    "next_node_id",
]

#: 算「模型加载器」的 class_type（与源同表 ✓）
MODEL_LOADERS = frozenset({"UNETLoader", "CheckpointLoaderSimple", "CheckpointLoader"})

#: 新节点 id 的起跳值 ✓（源的 9000 口径）
NODE_ID_FLOOR = 9000


class LoraChainError(Exception):
    """LoRA 注入失败（都是 4xx 性质：没加载器 / 没下游 / 名字不对 ✓）。"""

    def __init__(self, message: str, *, code: str = "lora_injection_failed") -> None:
        super().__init__(message)
        self.code = code


def normalize_loras(params: Any) -> list[dict[str, Any]]:
    """归一 ``params.loras`` ✓：跳过 ``enabled=false`` 与空名、``strength`` 非法回 1.0 ✓。"""
    if not isinstance(params, dict):
        return []
    items = params.get("loras")
    result: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get("enabled") is False:
            continue
        name = str(item.get("name") or item.get("lora_name") or "").strip()
        if not name:
            continue
        try:
            strength = float(item.get("strength", item.get("strength_model", 1)))
        except (TypeError, ValueError):
            strength = 1.0
        if strength != strength or strength in (float("inf"), float("-inf")):  # NaN / Inf
            strength = 1.0
        result.append({"name": name, "strength": strength})
    return result


def next_node_id(graph: dict[str, Any]) -> str:
    """新节点 id ✓：``max(数字 id 最大值+1, 9000)``，并跳过已占用的号 ✓（字符串比较口径同源）。"""
    maximum = 0
    for key in graph or {}:
        if re.fullmatch(r"\d+", str(key)):
            maximum = max(maximum, int(key))
    nxt = max(maximum + 1, NODE_ID_FLOOR)
    while graph.get(str(nxt)):
        nxt += 1
    return str(nxt)


def find_model_source(graph: dict[str, Any], preferred_id: str | None = None) -> str | None:
    """找「该往哪个模型加载器后面挂 LoRA」✓。

    ⭐ 判据是**被引用次数**（下游有几处吃它的第 0 槽），不是节点 id 大小；
    平票时 ``UNETLoader`` 优先 ✓（``CheckpointLoaderSimple`` 常常只提供 CLIP/VAE ✓）。
    """
    if preferred_id and graph.get(str(preferred_id)):
        return str(preferred_id)
    candidates = [
        str(node_id)
        for node_id, node in (graph or {}).items()
        if isinstance(node, dict) and node.get("class_type") in MODEL_LOADERS
    ]
    if not candidates:
        return None
    references = dict.fromkeys(candidates, 0)
    for node in (graph or {}).values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            # 只认「第 0 个输出」的连线 ✓（其他槽不是模型）
            if not isinstance(value, list) or len(value) < 2 or _as_int(value[1]) != 0:
                continue
            source = str(value[0])
            if source in references:
                references[source] += 1
    used = [node_id for node_id in candidates if references.get(node_id, 0) > 0]
    # 引用多的优先；平票 UNETLoader 优先；再平票按 id 字符串序（与源 `localeCompare` 同效 ✓）
    used.sort(key=lambda node_id: (
        -references.get(node_id, 0),
        0 if str(graph[node_id].get("class_type")) == "UNETLoader" else 1,
        node_id,
    ))
    return used[0] if used else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def apply_lora_chain(
    graph: dict[str, Any],
    params: Any,
    *,
    available: list[str] | None = None,
    filter: str | None = None,
    source_node: str | None = None,
    loader_class: str = "LoraLoaderModelOnly",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """把 ``params.loras`` 串成节点链插到模型加载器后面 ✓。

    返回 ``(工作流, applied)``；**没有 LoRA 时原样返回同一个对象** ✓（不克隆 ✓）。
    """
    loras = normalize_loras(params)
    if not loras:
        return graph, []

    # ⚠️ 校验放在**克隆之前**：报错时不留下半成品 ✓
    allowed = re.compile(filter, re.IGNORECASE) if filter else None
    for lora in loras:
        if allowed is not None and not allowed.search(lora["name"]):
            raise LoraChainError(f"LoRA {lora['name']} 不适用于当前模型")
        if available and lora["name"] not in available:
            raise LoraChainError(f"LoRA 不存在：{lora['name']}")

    next_graph = copy.deepcopy(graph or {})
    source_id = find_model_source(next_graph, source_node)
    if source_id is None:
        raise LoraChainError("LoRA 无法注入：工作流里没有可识别的模型加载器")

    # 记下所有「吃模型第 0 槽」的下游，最后统一重接到链尾 ✓（只改一处的图是半生效 ✗）
    targets: list[tuple[dict[str, Any], str]] = []
    for node in next_graph.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_name, value in list(inputs.items()):
            if isinstance(value, list) and len(value) >= 2 and str(value[0]) == source_id and _as_int(value[1]) == 0:
                targets.append((inputs, input_name))
    if not targets:
        raise LoraChainError("LoRA 无法注入：模型加载器没有下游消费者")

    last: list[Any] = [source_id, 0]
    applied: list[dict[str, Any]] = []
    for lora in loras:
        node_id = next_node_id(next_graph)
        next_graph[node_id] = {
            "class_type": loader_class,
            "inputs": {
                "model": last,
                "lora_name": lora["name"],
                "strength_model": lora["strength"],
            },
            # 界面上看得懂的名字（只取文件名，别把整条子目录糊上去 ✓）
            "_meta": {"title": f"LoRA {_path_name(lora['name'])}"},
        }
        last = [node_id, 0]
        applied.append({"id": node_id, "name": lora["name"], "strength": lora["strength"]})
    for inputs, input_name in targets:
        inputs[input_name] = last
    return next_graph, applied


def _path_name(name: str) -> str:
    """``Krea2/xxx.safetensors`` → ``xxx.safetensors`` ✓（正反斜杠都认 ✓）。"""
    return re.split(r"[\\/]", str(name or ""))[-1] or "LoRA"
