"""ComfyUI 图上「**哪一格是模型**」的清单 + 可用性判定（自研重写 ✓）。

:mod:`app.services.model_folders` 管的是「**路径** → 哪一类模型」✓；
本模块管的是另一半：**节点输入** → 哪一类模型 ✓（``UNETLoader.unet_name`` 是 unet、
``CLIPLoader.clip_name`` 是 text_encoder…）。两者是一张表的两面，**缺哪一面都不行** ✗：

* 没有本表 ⇒ 拿到一张工作流图，说不出「它要哪些模型」✗ ⇒ 也就没法回答
  「这台机器能不能跑」✗（用户只会看到 ComfyUI 甩出来的一句
  ``Prompt outputs failed validation`` ✗）；
* 没有 :mod:`model_folders` ⇒ 知道要 unet，却不知道去哪个目录找 ✓。

⚠️ 口径来源（2026-11-02 逐条核对 ✓）：上游 ``ComfyUI-Local-Infinite-Canvas``（MIT ✓）
的 ``comfyui-gateway/src/model-loaders.js`` ⇒ 只搬**语义与数据表**，代码自己写 ✗。

## 三处保真点（都实测过 ✓）

1. ⭐ **同一个节点可能有两个模型输入，且类别不同** ✗：``llama_cpp_model_loader`` 的
   ``model``（LLM 权重 ✓）与 ``mmproj``（视觉投影 ✓）**不能共用一个选项列表** ✗；
   ``TE_Qwen_Image_2_1_Prompt_Enhancer`` 更极端（4 个模型输入 ✓，其中两个 ``dirs`` 为空 ✓
   = 不落盘、只走选项 ✓）⇒ 规格必须按 ``(class_type, input)`` **二元组**登记 ✗
   （只按 class_type 登记会把 mmproj 当成主模型 ✗）。
2. ⭐ **静态表覆盖不全，必须能动态发现** ✓：用户装的第三方 loader（``LoraLoaderBypassModelOnly``、
   各种 GGUF 加载器 ✗）不在表里 ⇒ :func:`discover_model_loader_specs` 从 ``/object_info``
   里认「选项全是模型文件名」的输入 ✓（**排除** ``image/video/audio/mask/prompt`` 这些 ✗，
   也**排除** ``LoadImage`` / ``VHS_*`` 这类节点 ✗）。
3. ⭐ **模型下拉里混着「无 / None / null」占位项** ✓：它**仍是**模型选择输入 ✗，
   判定「这是不是模型输入」时不能因为它不像是文件名就排除 ✗。

## 可用性的两档口径

* ``missing_node``：图上用了这台机器**没装**的节点（``class_type`` 不在 ``object_info`` ✓）
  —— 装节点才能跑 ✗；
* ``missing_model``：节点有，但**它要的那个权重文件盘上没有** ✓
  —— 匹配时**忽略子目录**（盘上 ``unet/zimage/x.safetensors`` 与图上写的 ``x.safetensors``
  算同一个 ✓，这是 ComfyUI 子目录别名机制的常见情形 ✓）。
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

from .comfyui_match import find_similar_model

__all__ = [
    "MODEL_FILE_PATTERN",
    "MODEL_LOADER_SPECS",
    "MODEL_PLACEHOLDER",
    "NON_MODEL_CLASS",
    "NON_MODEL_INPUT",
    "ModelLoaderSpec",
    "availability_problems",
    "discover_model_loader_specs",
    "model_slots_of",
    "spec_index",
]

#: 模型文件的扩展名（认它是模型输入的关键证据 ✓）
MODEL_FILE_PATTERN = re.compile(r"\.(safetensors|ckpt|pt|pth|bin|gguf|onnx)$", re.IGNORECASE)

#: 下拉里的「不使用」占位项（**仍是**模型输入 ✗，不能被排除 ✓）
MODEL_PLACEHOLDER = re.compile(r"^(无|none|null)$", re.IGNORECASE)

#: 这些输入名**不是**模型（是素材/文本），动态发现时必须跳过 ✓
NON_MODEL_INPUT = re.compile(r"^(image|images|video|videos|audio|audios|file|filename|mask|prompt|text)$", re.IGNORECASE)

#: 这些节点**不是**模型 loader（素材/预览/保存类），动态发现时整体跳过 ✓
NON_MODEL_CLASS = re.compile(r"LoadImage|LoadVideo|LoadAudio|Upload|Preview|Save|VHS_", re.IGNORECASE)


class ModelLoaderSpec(NamedTuple):
    """一个「模型输入格」的口径。

    ``class_type`` + ``input`` 才是**唯一键** ✓（见模块头保真点 1 ✗）；
    ``dirs`` 为空 = 该格不对应磁盘目录（只走 ComfyUI 自己的选项列表 ✓）。
    """

    class_type: str
    input: str
    key: str
    dirs: tuple[str, ...]


#: ComfyUI 里“模型文件名”类输入的统一清单（18 条 ✓）
MODEL_LOADER_SPECS: tuple[ModelLoaderSpec, ...] = (
    ModelLoaderSpec("CheckpointLoaderSimple", "ckpt_name", "checkpoints", ("checkpoints",)),
    ModelLoaderSpec("UNETLoader", "unet_name", "unets", ("diffusion_models",)),
    ModelLoaderSpec("UpscaleModelLoader", "model_name", "upscaleModels", ("upscale_models",)),
    ModelLoaderSpec("SeedVR2LoadDiTModel", "model", "seedvr2Dit", ("upscale_models",)),
    ModelLoaderSpec("SeedVR2LoadVAEModel", "model", "seedvr2Vae", ("upscale_models",)),
    ModelLoaderSpec("CLIPLoader", "clip_name", "clips", ("text_encoders",)),
    ModelLoaderSpec("VAELoader", "vae_name", "vaes", ("vae",)),
    ModelLoaderSpec("LoraLoader", "lora_name", "loras", ("loras",)),
    ModelLoaderSpec("LoraLoaderModelOnly", "lora_name", "lorasModelOnly", ("loras",)),
    ModelLoaderSpec("ApplyVDNH3Advanced", "vdn_checkpoint", "vdn", ("vdn",)),
    # llama-cpp 的 LLM 权重与视觉投影是两个不同输入，不能共用一个选项列表 ✓
    ModelLoaderSpec("llama_cpp_model_loader", "model", "llmModel", ("LLM",)),
    ModelLoaderSpec("llama_cpp_model_loader", "mmproj", "llmMmproj", ("LLM",)),
    # 同一套 LLM 文件在这些节点里输入名不同；静态登记一份，扫描未完成时也能套别名 ✓
    ModelLoaderSpec("QwenTE_ModelLoader", "主模型", "qwenTeModel", ("LLM",)),
    ModelLoaderSpec("QwenTE_ModelLoader", "视觉投影mmproj", "qwenTeMmproj", ("LLM",)),
    ModelLoaderSpec("TE_Qwen_Image_2_1_Prompt_Enhancer", "主模型", "qwenImagePeModel", ("LLM",)),
    ModelLoaderSpec("TE_Qwen_Image_2_1_Prompt_Enhancer", "mmproj", "qwenImagePeMmproj", ("LLM",)),
    ModelLoaderSpec("TE_Qwen_Image_2_1_Prompt_Enhancer", "文生图PE模型", "qwenImagePeT2i", ()),
    ModelLoaderSpec("TE_Qwen_Image_2_1_Prompt_Enhancer", "图生图PE模型", "qwenImagePeI2i", ()),
)


def spec_index(specs: tuple[ModelLoaderSpec, ...] | list[ModelLoaderSpec] | None = None) -> dict[tuple[str, str], ModelLoaderSpec]:
    """把规格表变成 ``{(class_type, input): spec}`` 查表 ✓（查表比遍历快，也避免重复登记 ✗）。"""
    return {(spec.class_type, spec.input): spec for spec in (specs or MODEL_LOADER_SPECS)}


def _options_of(definition: Any) -> list[str] | None:
    """从 ``/object_info`` 的 INPUT_TYPES 定义里取出选项列表 ✓（两种写法都要认 ✓）。"""
    if isinstance(definition, list):
        if definition and isinstance(definition[0], list):
            return [str(item) for item in definition[0]]
        meta = definition[1] if len(definition) > 1 else None
        if isinstance(meta, dict) and isinstance(meta.get("options"), list):
            return [str(item) for item in meta["options"]]
    return None


def discover_model_loader_specs(
    object_info: dict[str, Any] | None,
    base: tuple[ModelLoaderSpec, ...] | None = None,
) -> list[ModelLoaderSpec]:
    """从 ``/object_info`` **动态发现**第三方 loader 的模型输入格 ✓（静态表的**补集** ✓）。

    判定：该输入的选项里**至少有一个**是模型文件名，且**全部**要么像模型文件、
    要么是「无/None/null」占位 ✓ ⇒ 认它是模型输入 ✓。
    """
    specs: list[ModelLoaderSpec] = list(base if base is not None else MODEL_LOADER_SPECS)
    seen = {(spec.class_type, spec.input) for spec in specs}
    for class_type, info in (object_info or {}).items():
        if not isinstance(info, dict) or NON_MODEL_CLASS.search(str(class_type)):
            continue
        groups = [info.get("input", {}).get("required"), info.get("input", {}).get("optional")]
        for group in groups:
            if not isinstance(group, dict):
                continue
            for input_name, definition in group.items():
                if NON_MODEL_INPUT.search(str(input_name)):
                    continue
                options = _options_of(definition)
                if not options:
                    continue
                if not any(MODEL_FILE_PATTERN.search(option) for option in options):
                    continue
                if not all(MODEL_FILE_PATTERN.search(option) or MODEL_PLACEHOLDER.search(option) for option in options):
                    continue
                marker = (str(class_type), str(input_name))
                if marker in seen:
                    continue
                seen.add(marker)
                specs.append(ModelLoaderSpec(str(class_type), str(input_name), f"{class_type}.{input_name}", ()))
    return specs


def model_slots_of(
    graph: dict[str, Any],
    specs: tuple[ModelLoaderSpec, ...] | list[ModelLoaderSpec] | None = None,
) -> list[dict[str, Any]]:
    """列出一张图上**所有**模型输入格及其当前值 ✓（值不是字符串 ⇒ 是连线，跳过 ✓）。"""
    index = spec_index(specs)
    slots: list[dict[str, Any]] = []
    for node_id, node in (graph or {}).items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        inputs = node.get("inputs") or {}
        for input_name, value in inputs.items():
            spec = index.get((class_type, str(input_name)))
            if spec is None or not isinstance(value, str):
                continue
            slots.append(
                {
                    "node": str(node_id),
                    "classType": class_type,
                    "input": str(input_name),
                    "key": spec.key,
                    "dirs": list(spec.dirs),
                    "value": value,
                }
            )
    return slots


def _basename_of(value: str) -> str:
    """忽略子目录 ✓：``unet/zimage/x.safetensors`` ⇒ ``x.safetensors`` ✓。"""
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def availability_problems(
    graph: dict[str, Any],
    *,
    object_info: dict[str, Any] | None = None,
    available: dict[str, Any] | None = None,
    specs: tuple[ModelLoaderSpec, ...] | list[ModelLoaderSpec] | None = None,
) -> list[dict[str, Any]]:
    """判定「这台机器能不能跑这张图」✓，返回**具体问题**（不是一句「校验失败」✗）。

    * ``object_info`` 给了 ⇒ 报 ``missing_node``（图上有这台机器没装的节点 ✓）；
    * ``available`` 给了（``{key: {文件名或 basename 集合}}`` ✓）⇒ 报 ``missing_model`` ✓；
    * 哪个都不给 ⇒ 只能报缺节点/缺模型都查不了，返回空列表 ✓（**没检查 ≠ 通过** ✓，
      调用方要自己保证至少给一个 ✗）。
    """
    problems: list[dict[str, Any]] = []
    known_types = set(object_info or {})
    for node_id, node in (graph or {}).items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        if object_info and class_type not in known_types:
            problems.append({"kind": "missing_node", "node": str(node_id), "classType": class_type})
    if not available:
        return problems
    for slot in model_slots_of(graph, specs):
        value = slot["value"]
        if MODEL_PLACEHOLDER.search(value):
            continue
        pool = available.get(slot["key"])
        if pool is None:
            continue
        names = {str(item) for item in pool} | {_basename_of(str(item)) for item in pool}
        if value in names or _basename_of(value) in names:
            continue
        # 精确没命中 ⇒ 再给一次「同模型另一档」的建议（量化/子目录差异很常见 ✓）
        # ⚠️ 建议**只是建议**：歧义时不替用户选，把候选原样交回去 ✓
        suggestion = find_similar_model(value, sorted(names))
        problem: dict[str, Any] = {
            "kind": "missing_model",
            "node": slot["node"],
            "classType": slot["classType"],
            "input": slot["input"],
            "key": slot["key"],
            "value": value,
            "dirs": slot["dirs"],
        }
        if suggestion.get("file"):
            problem["suggestion"] = suggestion["file"]
            problem["suggestionScore"] = round(float(suggestion["score"]), 4)
        elif suggestion.get("ambiguous"):
            problem["ambiguous"] = True
            problem["candidates"] = suggestion.get("candidates") or []
        problems.append(problem)
    return problems
