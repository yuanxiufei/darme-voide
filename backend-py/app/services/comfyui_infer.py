"""ComfyUI **未登记工作流的参数落点推断**（自 ``comfyui-gateway/src/registry.js`` 的
``inferType`` / ``inferBindings`` 迁入 ✓，按本仓口径重写 ✓）。

**解决什么**：本仓 21 条工作流的参数落点是**手写在 registry 里**的 ✓；从外部拿到的一张新
API 图没有任何声明 ⇒ 前端只能整张提交、**一个参数都改不了** ✗（提示词/尺寸/种子全写死在
图里 ✗）。这里拿图就答「哪些格是给提示词、尺寸、种子、参考图用的」✓。

**只推断、不写盘、不提交、不连上游** ✓ —— 产出的 ``bindings`` 可直接抄进
``configs/comfyui/workflows/registry.json``，也可由调用方拼进 ``POST /runs`` 的 ``entry`` ✓。

**与本仓口径对齐的四处**（源是 JS，字段名不完全同）：

1. ``from`` 走 ``params.*`` ✓ —— 推断出来的工作流**没有** ``presets`` 档位表，写
   ``quality.steps`` 在 ``build_context`` 里解析不到 ✗（看着有参数、填了不生效 ✗）；
2. 多参考图走 ``array`` + ``inputPrefix`` ✓（``apply_bindings`` 的 2c 分支按前缀铺开并重建
   ``LoadImage`` 连线），不是源的 ``image`` / ``image2`` 逐格（那要求图上已有 ``LoadImage`` ✗）；
3. 图上**本来是连线**的格（值形如 ``["76", 0]``）一律不绑 ✓ 并报 warning —— 换成数字/文件名
   等于**拆掉别人接好的线** ✗，图照跑、产物全变样、且**不报错** ✗；
4. 额外推 ``unet`` / ``ckpt`` / ``lora`` / ``lora_strength`` / ``upscale_model`` 这些本仓才有的
   落点 ✓（源只推生成参数，换底模/LoRA 得手改图 ✗）。

**四处刻意与源不同**：源用 ``ensureTypePrefix`` **强制改名**成 ``image-xxx`` ✗，本仓只在
``suggested`` 里给建议、不动调用方交来的 ``name`` ✓；源给负向写死 ``default: ""`` ✗，本仓
**保留图上原有负向**作默认 ✓（抹掉别人写好的负向是静默降质 ✗）；源推断 ``samplerName`` ✗，
本仓**不绑采样器/调度器** ✓ —— 它们的可选值是**节点定义里的枚举**（服务在线才拿得到），
绑成自由文本等于让人填一个不存在的名字：图照提交、ComfyUI 悄悄回落到默认 ✗；源的正负向
共用一个候选表 ✗，本仓**分成两套键** ✓ —— 否则 ``TextEncodeQwenImage21`` 那种
「``prompt`` + ``negative_prompt`` 同节点」的编码器会把用户写的提示词整个灌进**负向格**，
图照出、内容全反、还不报错 ✗。

**推不出来的三类，如实进** ``missing`` **+ warning**（不许瞎填）：
① 未连线的**可选数组槽**在 API 图里根本不出现（``TextEncodeQwenImage21`` 的
``images.image_1…4``、``MiniMaxH3AudioConditioningT8`` 的 ``ref_images.*``）⇒ 只能靠人声明；
② 族专用旋钮（``rho`` / ``w_max`` / ``lowres_scale`` / ``transition_step`` / ``blocks_to_swap`` /
``task_type`` / 各种 cache 开关）值域只有该族自己知道；
③ 本来就是连线的格（尺寸由上游节点算出来的那种）。
"""

from __future__ import annotations

import re
from typing import Any

from . import comfyui_catalog as catalog
from .comfyui_prompts import required_nodes

SEED_MAX = 2147483647

_TYPES = ("image", "video", "audio", "text")
_PREFIX = {"image": "image", "video": "video", "audio": "audio", "text": "text"}

_VIDEO_OUTPUT = re.compile(r"SaveVideo|VHS_VideoCombine|SaveWebM|SaveAnimated", re.I)
_IMAGE_OUTPUT = re.compile(r"SaveImage|PreviewImage|ImageCompare", re.I)
_AUDIO_OUTPUT = re.compile(r"SaveAudio", re.I)

# H3 = 17k+5 帧 @24fps，Wan 类 = 4n+1 帧 @16fps：认错不报错、只是时长不对 ✗
_H3 = re.compile(r"MiniMaxH3|MinimaxH3|SelfLiftH3|VDN.?H3|H3Audio|H3Sigma", re.I)
_VIDEO_NODE = re.compile(r"WanImageToVideo|WanFirstLastFrame|WanVace|H3", re.I)

_FRAME_KEYS = ("length", "num_frames", "frames", "video_length")
_START_KEYS = ("start_image", "start_frame", "first_frame", "init_image")
_END_KEYS = ("end_image", "last_frame", "end_frame")


class InferError(Exception):
    """推断失败 ✓ —— 宁可说「这张图推不出来」，也不给一份**填了不生效**的 bindings ✗。"""

    def __init__(self, message: str, *, code: str = "bad_workflow") -> None:
        super().__init__(message)
        self.code = code


# ── 图结构 ────────────────────────────────────────────────────────────────
def _is_link(value: Any) -> bool:
    """``["76", 0]`` 这种**连线** ✓（连线格不能直接拿文件名/数字覆盖 ✗）。"""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
        and isinstance(value[0], (str, int))
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """把图摊成节点列表 ✓（``id`` 按数字自然序：``"80"`` 排在 ``"110"`` 前 ✓）。"""
    rows: list[dict[str, Any]] = []
    for node_id, node in (graph or {}).items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "").strip()
        if not class_type:
            continue
        inputs = node.get("inputs")
        meta = node.get("_meta")
        rows.append(
            {
                "id": str(node_id),
                "classType": class_type,
                "inputs": dict(inputs) if isinstance(inputs, dict) else {},
                "title": str((meta or {}).get("title") or "") if isinstance(meta, dict) else "",
            }
        )
    rows.sort(key=lambda row: (len(row["id"]), row["id"]))
    return rows


def infer_type(nodes: list[dict[str, Any]], hint: str | None = None) -> str:
    """按**输出节点**判这张图出什么 ✓（``hint`` 只收四类真名，别名不算 ✗）。"""
    if hint is not None:
        if hint not in _TYPES:
            raise InferError(f"type 只支持 {list(_TYPES)}，收到 {hint!r}", code="bad_hint")
        return hint
    classes = " ".join(node["classType"] for node in nodes)
    if _VIDEO_OUTPUT.search(classes):
        return "video"
    if _AUDIO_OUTPUT.search(classes):
        return "audio"
    if _IMAGE_OUTPUT.search(classes):
        return "image"
    raise InferError(
        "这张图里找不到输出节点（``SaveImage`` / ``SaveVideo`` / ``VHS_VideoCombine`` / ``SaveAudio``）"
        "⇒ 它跑完什么也落不了盘，不是一份能跑的工作流 ✗",
        code="no_output",
    )


def _input_of(
    nodes: list[dict[str, Any]], keys: tuple[str, ...], *, numeric: bool
) -> tuple[dict[str, Any], str, Any] | None:
    """找第一个带指定输入名、值类型也对得上的节点 ✓（连线不算 ✓）。"""
    for node in nodes:
        for key in keys:
            value = node["inputs"].get(key)
            if _is_link(value):
                continue
            if numeric and not _is_number(value):
                continue
            if not numeric and not isinstance(value, str):
                continue
            return node, key, value
    return None


# ── 文本 ──────────────────────────────────────────────────────────────────
# ⚠️ 正向与负向**必须是两套键** ✗：共用一个元组时，`TextEncodeQwenImage21` 这种
# 「`prompt` + `negative_prompt` 同节点」的编码器会被读成正向落 `negative_prompt`
# ⇒ 用户写的提示词全灌进负向格，图照出、内容全反、还不报错 ✗
_POS_KEYS = ("prompt", "positive_prompt", "positive", "text", "base_prompt")
_NEG_KEYS = ("negative_prompt", "negative", "text", "prompt")
_NEG_TITLE = re.compile(r"negative|负向|负面|反向", re.I)
_POS_TITLE = re.compile(r"positive|正向|正面", re.I)


def _text_hit(
    by_id: dict[str, dict[str, Any]], link: Any, keys: tuple[str, ...], depth: int = 0
) -> tuple[dict[str, Any], str] | None:
    """顺着 ``positive`` / ``negative`` 的**连线**找到真正吃文本的那一格 ✓。

    支持**往下两跳** ✓：``CFGGuider.positive → TextEncodeQwenImage21.prompt(连线) →
    QwenAnglePromptConcat.base_prompt`` 这种「拼装节点」也是提示词的真落点 ——
    断在第一跳就会报「没找到提示词落点」，而那张图其实可以改文本 ✗。
    """
    if not _is_link(link) or depth > 2:
        return None
    node = by_id.get(str(link[0]))
    if node is None:
        return None
    for key in keys:
        if isinstance(node["inputs"].get(key), str):
            return node, key
    for key in keys:
        hit = _text_hit(by_id, node["inputs"].get(key), keys, depth + 1)
        if hit is not None:
            return hit
    return None


def _apply_text_hits(
    positive_hit: tuple[dict[str, Any], str] | None,
    negative_hit: tuple[dict[str, Any], str] | None,
    bindings: dict[str, Any],
    warnings: list[str],
) -> None:
    if positive_hit is not None:
        # **不给 default**：不传就保留图上原值 ✓（源同口径）
        bindings["prompt"] = {"node": positive_hit[0]["id"], "input": positive_hit[1]}
    if negative_hit is not None:
        original = negative_hit[0]["inputs"][negative_hit[1]]
        bindings["negative"] = {
            "node": negative_hit[0]["id"],
            "input": negative_hit[1],
            "from": "negative",
            # 保留图上原有负向 ✓（抹掉别人写好的负向 = 静默降质 ✗）
            "default": original,
        }
        if original.strip():
            warnings.append(
                f"负向 {negative_hit[0]['id']}.{negative_hit[1]} 图里已有 {len(original)} 字 ⇒ 作为默认保留"
                "（不传就照旧 ✓，传空串才清空）"
            )


def _bind_text(nodes: list[dict[str, Any]], bindings: dict[str, Any], warnings: list[str]) -> None:
    # ① 首选判据：**顺着采样器的 positive / negative 连线反查** ✓
    # 比"按标题猜、按顺序猜"可靠得多 —— Krea2 那种一图两个编码器（「负向编码」+「四视图提示词」）
    # 按顺序猜必然把**负向节点当正向**，正向提示词写进负向格 ✗：图照样出、语义全反、还不报错 ✗
    by_id = {node["id"]: node for node in nodes}
    positive_hit: tuple[dict[str, Any], str] | None = None
    negative_hit: tuple[dict[str, Any], str] | None = None
    for node in nodes:
        if positive_hit is None:
            hit = _text_hit(by_id, node["inputs"].get("positive"), _POS_KEYS)
            if hit is not None and not _NEG_TITLE.search(hit[0]["title"]):
                positive_hit = hit
        if negative_hit is None:
            hit = _text_hit(by_id, node["inputs"].get("negative"), _NEG_KEYS)
            if hit is not None:
                negative_hit = hit
    if (
        positive_hit is not None
        and negative_hit is not None
        and positive_hit[0]["id"] == negative_hit[0]["id"]
        and positive_hit[1] == negative_hit[1]
    ):
        # 同节点**同一格**才去重 ✓ —— 同节点的两格（prompt / negative_prompt）都是真落点 ✗
        negative_hit = None
    if positive_hit is not None or negative_hit is not None:
        _apply_text_hits(positive_hit, negative_hit, bindings, warnings)
        return

    encoders = [node for node in nodes if re.search(r"CLIPTextEncode", node["classType"], re.I)]
    if encoders:
        pool = [node for node in encoders if not _NEG_TITLE.search(node["title"])] or encoders
        positive = next((node for node in pool if _POS_TITLE.search(node["title"])), pool[0])
        negative = next(
            (
                node
                for node in encoders
                if node["id"] != positive["id"] and re.search(r"negative|负向|负面", node["title"], re.I)
            ),
            None,
        )
        if negative is None:
            negative = next((node for node in encoders if node["id"] != positive["id"]), None)

        if "text" not in positive["inputs"] and "text_g" in positive["inputs"]:
            # SDXL 双格文本：一条 prompt 拆两半喂进去只会得到半句话 ✗，一格也装不下 ✗
            warnings.append(
                f"{positive['id']} {positive['classType']} 是 SDXL 双格文本（text_g / text_l）⇒ "
                "没有推断 prompt 落点，需人工指定"
            )
        elif isinstance(positive["inputs"].get("text"), str):
            # **不给 default**：不传就保留图上原值 ✓（源同口径）
            bindings["prompt"] = {"node": positive["id"], "input": "text"}
        elif _is_link(positive["inputs"].get("text")):
            warnings.append(f"{positive['id']}.text 是连线（文本由上游拼）⇒ 没有推断 prompt 落点")

        if negative is not None and isinstance(negative["inputs"].get("text"), str):
            original = negative["inputs"]["text"]
            bindings["negative"] = {
                "node": negative["id"],
                "input": "text",
                "from": "negative",
                # 保留图上原有负向 ✓（抹掉别人写好的负向 = 静默降质 ✗）
                "default": original,
            }
            if original.strip():
                warnings.append(
                    f"负向 {negative['id']} 图里已有 {len(original)} 字 ⇒ 作为默认保留"
                    "（不传就照旧 ✓，传空串才清空）"
                )

    if "prompt" not in bindings:
        # 兜底：非 CLIPTextEncode 的文本口 ✓（如 Krea2 的 `Krea2EditGroundedEncode.prompt`）
        # **跳过标题写着负向的节点** ✗：那种图的负向节点同样有 `prompt` 输入，认错就把正向写进负向格 ✗
        for node in nodes:
            if _NEG_TITLE.search(node["title"]):
                continue
            key = next(
                (
                    item
                    for item in ("positive_prompt", "prompt", "positive", "text", "base_prompt")
                    if isinstance(node["inputs"].get(item), str)
                ),
                None,
            )
            if key is None:
                continue
            bindings["prompt"] = {"node": node["id"], "input": key}
            warnings.append(f"prompt 落到 {node['id']} {node['classType']}.{key}（兜底推断，不是 CLIPTextEncode）")
            break

    if "negative" not in bindings:
        # 负向兜底：`negative_prompt` 是明确命名 ✓；`text` / `prompt` 只认标题写着负向的节点 ✓
        # ⚠️ 只排除**已被 prompt 占用的那一格**，不是排除整个节点 ✗ —— `TextEncodeQwenImage21`
        # 就是 `prompt` 与 `negative_prompt` **同一节点两格**，按节点排除会把负向整个漏掉 ✗
        taken = bindings.get("prompt") or {}
        for node in nodes:
            key = next(
                (
                    item
                    for item in ("negative_prompt", "negative", "text", "prompt")
                    if isinstance(node["inputs"].get(item), str)
                    and (item in ("negative_prompt", "negative") or _NEG_TITLE.search(node["title"]))
                    and not (node["id"] == str(taken.get("node")) and item == taken.get("input"))
                ),
                None,
            )
            if key is None:
                continue
            original = node["inputs"][key]
            bindings["negative"] = {"node": node["id"], "input": key, "from": "negative", "default": original}
            if original.strip():
                warnings.append(
                    f"负向 {node['id']}.{key} 图里已有 {len(original)} 字 ⇒ 作为默认保留"
                    "（不传就照旧 ✓，传空串才清空）"
                )
            break


# ── 尺寸 / 时长 ────────────────────────────────────────────────────────────
def _frame_rule(node: dict[str, Any], frames: int) -> tuple[str, int, str]:
    """帧数换算口径 ✓（H3 是 17k+5、Wan 类是 4n+1，判错不报错只是时长不对 ✗）。"""
    if _H3.search(node["classType"]):
        return "secondsToFrames17", 24, f"{node['classType']} 属 H3 族（17k+5 @24fps）"
    if frames > 5 and (frames - 5) % 17 == 0:
        return "secondsToFrames17", 24, f"当前 {frames} 帧满足 17k+5"
    if frames > 1 and (frames - 1) % 4 == 0:
        return "secondsToFrames", 16, f"当前 {frames} 帧满足 4n+1"
    return "secondsToFrames", 16, "默认按 Wan 口径（4n+1 @16fps）"


def _bind_size(nodes: list[dict[str, Any]], bindings: dict[str, Any], warnings: list[str]) -> None:
    latent = next((node for node in nodes if re.search(r"^Empty.*LatentImage$", node["classType"], re.I)), None)
    if latent is not None:
        for key in ("width", "height"):
            value = latent["inputs"].get(key)
            if _is_number(value):
                bindings[key] = {
                    "node": latent["id"],
                    "input": key,
                    "from": f"size.{key}",
                    "transform": "snap16",
                    "default": int(value),
                }
        batch = latent["inputs"].get("batch_size")
        if _is_number(batch):
            bindings["batch"] = {
                "node": latent["id"],
                "input": "batch_size",
                "from": "count",
                "default": int(batch),
            }

    video_nodes = [
        node
        for node in nodes
        if _VIDEO_NODE.search(node["classType"])
        and any(key in node["inputs"] for key in _FRAME_KEYS + ("width", "height"))
    ]
    for node in video_nodes:
        for key in ("width", "height"):
            value = node["inputs"].get(key)
            if not _is_number(value):
                continue
            # 视频族的宽高是 **32 吸附** ✓（与 registry 那 4 条 H3 同口径）
            bindings[key] = {
                "node": node["id"],
                "input": key,
                "from": f"size.{key}",
                "transform": "snap32",
                "default": int(value),
            }
        frame_key = next((key for key in _FRAME_KEYS if key in node["inputs"]), None)
        if frame_key is None:
            continue
        value = node["inputs"][frame_key]
        if not _is_number(value):
            warnings.append(f"{node['id']}.{frame_key} 不是数字（长度由上游算）⇒ 没有推断时长落点")
            continue
        transform, fps, basis = _frame_rule(node, int(value))
        bindings["length"] = {
            "node": node["id"],
            "input": frame_key,
            "from": "seconds",
            "transform": transform,
            "fps": fps,
            "default": int(value),
        }
        warnings.append(
            f"时长落 {node['id']}.{frame_key}（当前 {int(value)} 帧 ≈ {int(value) / fps:.2f} 秒），"
            f"按 {transform} / {fps}fps 换算 —— 依据：{basis}"
        )

    if "width" not in bindings and "height" not in bindings:
        warnings.append(
            "没找到尺寸落点（没有 Empty*LatentImage、也没有带 width/height 的视频节点）⇒ 只能手改图；"
            "尺寸是连线时更不该绑（绑了就拆线 ✗）"
        )


# ── 采样参数 ───────────────────────────────────────────────────────────────
def _looks_like_sigmas(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) < 3:
        return False
    try:
        [float(item) for item in parts]
    except ValueError:
        return False
    return True


def _steps_from_sigmas(value: str) -> int:
    """sigmas 串里有 N 个数 ⇒ 就是 N−1 步 ✓（首尾是 1 与 0）。"""
    count = len([item for item in value.split(",") if item.strip()])
    return max(1, min(50, count - 1))


def _bind_sampler(nodes: list[dict[str, Any]], bindings: dict[str, Any], warnings: list[str]) -> None:
    # 步数：``sigmas`` 是**字符串**输入（Qwen-Image 2.1 官方轨迹，改旁边的 steps 数字无效 ✗）
    # ⇒ 必须落 sigmas 并走 qwenSigmas ✓；若是**连线**（H3 的 H3SigmaRefiner）⇒ 那不是落点 ✓
    sigmas = _input_of(nodes, ("sigmas",), numeric=False)
    steps = _input_of(nodes, ("steps",), numeric=True)
    if sigmas is not None and _looks_like_sigmas(sigmas[2]):
        node, key, value = sigmas
        bindings["steps"] = {
            "node": node["id"],
            "input": key,
            "from": "params.steps",
            "transform": "qwenSigmas",
            "default": _steps_from_sigmas(value),
        }
        warnings.append(
            f"步数落在 {node['id']}.{key} 的 **sigmas 串**上（改步数生成新轨迹 ✓）—— "
            "该族按字符串喂轨迹，直接改旁边那个 steps 数字不生效 ✗"
        )
    elif steps is not None:
        node, key, value = steps
        bindings["steps"] = {
            "node": node["id"],
            "input": key,
            "from": "params.steps",
            "transform": "int",
            "default": int(value),
        }

    cfg = _input_of(nodes, ("cfg",), numeric=True)
    if cfg is not None:
        node, key, value = cfg
        bindings["cfg"] = {"node": node["id"], "input": key, "from": "params.cfg", "default": value}

    seed = _input_of(nodes, ("seed", "noise_seed"), numeric=True)
    if seed is not None:
        node, key, _value = seed
        bindings["seed"] = {
            "node": node["id"],
            "input": key,
            "from": "params.seed",
            "transform": "int",
            "random": True,
        }
        # 一张图**多个采样器**各有一个 seed ⇒ 只绑第一个会让人以为"改种子了"，
        # 第二段采样其实纹丝不动（图变了但不多 ✗）。其余按 `seed2 / seed3 …` 一并绑到同一个
        # `params.seed`（与 registry 那两条双采样工作流同口径 ✓）
        extras = 0
        for other in nodes:
            if other["id"] == node["id"]:
                continue
            other_key = next((item for item in ("seed", "noise_seed") if _is_number(other["inputs"].get(item))), None)
            if other_key is None:
                continue
            extras += 1
            bindings[f"seed{extras + 1}"] = {
                "node": other["id"],
                "input": other_key,
                "from": "params.seed",
                "transform": "int",
            }
            if extras >= 4:
                break
        if extras:
            warnings.append(
                f"另有 {extras} 处 seed 一并绑到 params.seed（图里 {extras + 1} 个采样器）—— "
                "只绑一个的话，改种子只有第一段会变 ✗"
            )

    # 采样器 / 调度器**不绑** ✗：它们的可选值是**节点定义里的枚举**（服务在线才拿得到），
    # 绑成自由文本等于让人往里填一个不存在的名字 —— 图照提交、ComfyUI 悄悄回落到默认 ✗

    # denoise 只在**图生图**才绑 ✓（纯文生图多一个 denoise=1 的旋钮只会让人白调 ✗，源同口径）
    has_image = any(re.match(r"^LoadImage", node["classType"], re.I) for node in nodes)
    denoise = _input_of(nodes, ("denoise",), numeric=True)
    if has_image and denoise is not None:
        node, key, value = denoise
        bindings["denoise"] = {"node": node["id"], "input": key, "from": "params.denoise", "default": value}


# ── 参考素材 / 模型 ────────────────────────────────────────────────────────
def _is_pure_transform(bindings: dict[str, Any], nodes: list[dict[str, Any]]) -> bool:
    """纯变换图 ✓ —— 既没有提示词、也没有空 latent ⇒ 参考素材是**唯一输入**，不传就该报错 ✓。

    反过来，文生图/文生视频那种图里的 ``LoadImage`` 可能是**配方自带的固定参考**
    （作者烘在里面的），一律标必需会逼调用方凭空造一张图 ✗。
    """
    if "prompt" in bindings:
        return False
    return not any(re.search(r"^Empty.*LatentImage$|^Empty.*Latent$", node["classType"], re.I) for node in nodes)


def _bind_references(nodes: list[dict[str, Any]], bindings: dict[str, Any], warnings: list[str]) -> None:
    required = _is_pure_transform(bindings, nodes)
    load_images = [node for node in nodes if re.match(r"^LoadImage", node["classType"], re.I)]
    for index, node in enumerate(load_images[:4]):
        key = "image" if index == 0 else f"image{index + 1}"
        # `LoadImage.image` 收的是**文件名**✓ ⇒ 直接塞、不用 asLink ✓（与 registry four-view 同形）
        spot = {"node": node["id"], "input": "image", "from": "image", "imageIndex": index}
        if required:
            spot["required"] = True
        bindings[key] = spot
    if len(load_images) > 4:
        warnings.append(f"图里有 {len(load_images)} 个 LoadImage，只推了前 4 个（再多也装不下 ✗）")
    if load_images and not required:
        warnings.append(
            f"参考图落 {load_images[0]['id']}.image ⇒ **不传就跑图里写死的占位文件名**"
            "（该图本身有提示词落点，故未标必填）"
        )

    # 视频素材：`LoadVideo.file` / `VHS_LoadVideo.video` 收的也是**文件名** ✓（apply 2b 直接塞）
    load_videos = [
        node for node in nodes if re.match(r"^(?:LoadVideo|VHS_LoadVideo|LoadVideoUpload)$", node["classType"], re.I)
    ]
    for index, node in enumerate(load_videos[:2]):
        key = "video" if index == 0 else f"video{index + 1}"
        input_name = next((name for name in ("file", "video") if isinstance(node["inputs"].get(name), str)), None)
        if input_name is None:
            continue
        spot = {"node": node["id"], "input": input_name, "from": "video", "videoIndex": index}
        if required:
            spot["required"] = True
        bindings[key] = spot

    for node in nodes:
        if not _VIDEO_NODE.search(node["classType"]):
            continue
        for keys, index in ((_START_KEYS, 0), (_END_KEYS, 1)):
            key = next((item for item in keys if item in node["inputs"]), None)
            if key is None:
                continue
            value = node["inputs"][key]
            if isinstance(value, str):
                bindings["image" if index == 0 else "image2"] = {
                    "node": node["id"],
                    "input": key,
                    "from": "image",
                    "imageIndex": index,
                }
                continue
            # 值是**连线**（IMAGE 插槽）⇒ 必须 asLink 让 apply 重建 LoadImage 再连 ✓
            # 直接塞文件名进去，ComfyUI 当场报类型不符 ✗
            bindings["startFrame" if index == 0 else "endFrame"] = {
                "node": node["id"],
                "input": key,
                "from": "image",
                "imageIndex": index,
                "asLink": True,
            }

    # 多格参考图 ⇒ array 口径 ✓（apply 2c 按前缀铺开再重建连线）
    # 前缀**不写死** `ref_image_` ✗：Qwen-Image 2.1 那族用的是 `images.image_1…4`（indexBase=1），
    # 只认 `ref_image_` 会把它的参考图整片漏掉 —— 图照跑，参考图全空，且**不报错** ✗
    for node in nodes:
        groups: dict[str, list[int]] = {}
        for input_name in node["inputs"]:
            match = re.match(r"^(?P<prefix>.+?image_)(?P<index>\d+)$", input_name, re.I)
            if match:
                groups.setdefault(match.group("prefix"), []).append(int(match.group("index")))
        for prefix, indexes in groups.items():
            base = min(indexes)
            bindings["refs"] = {
                "node": node["id"],
                "input": prefix.rsplit(".", 1)[0] if "." in prefix else prefix,
                "from": "image",
                "array": True,
                "inputPrefix": prefix,
                "indexBase": base,
                "max": len(indexes),
            }
            warnings.append(
                f"参考图落 {node['id']} 的 {prefix}{base}..{prefix}{base + len(indexes) - 1}"
                f"（{len(indexes)} 格，按 array 铺开 ✓）"
            )
        # 音轨/视频轨前缀**不推** ✗：`ref_audio_*` 要 `LoadAudio`（可）但 `ref_video_*` 的插槽语义
            # 是 AUDIO 还是 IMAGE 帧，光看图判不出来 ⇒ 猜错就是接错线、图照跑 ✗ ⇒ 只说一句
        for input_name in node["inputs"]:
            if re.match(r"^(?:.*?\.)?ref_(?:audio|video)_\d+$", input_name, re.I):
                warnings.append(
                    f"{node['id']} 有按槽的音/视频参考（{input_name} …）⇒ **没有推断落点**（插槽类型要从节点定义看，"
                    "光读图猜不出来）"
                )
                break


def _bind_models(nodes: list[dict[str, Any]], bindings: dict[str, Any]) -> None:
    unet = _input_of(nodes, ("unet_name",), numeric=False)
    if unet is not None:
        node, key, value = unet
        bindings["unet"] = {"node": node["id"], "input": key, "from": "params.unet", "default": value}
    ckpt = _input_of(nodes, ("ckpt_name",), numeric=False)
    if ckpt is not None:
        node, key, value = ckpt
        bindings["ckpt"] = {"node": node["id"], "input": key, "from": "params.ckpt", "default": value}
    lora = _input_of(nodes, ("lora_name",), numeric=False)
    if lora is not None:
        node, key, value = lora
        bindings["lora"] = {"node": node["id"], "input": key, "from": "params.lora", "default": value}

    # LoRA 强度与它**成对** ✓：只给换 LoRA 不给强度，换上去的 LoRA 就一直是图里烘死的那个强度 ✗
    for node in nodes:
        if not re.search(r"LoraLoader", node["classType"], re.I):
            continue
        for key in ("strength_model", "strength_clip"):
            value = node["inputs"].get(key)
            if _is_number(value):
                bindings["lora_strength"] = {
                    "node": node["id"],
                    "input": key,
                    "from": "params.lora_strength",
                    "default": value,
                }
                break
        break

    # 放大模型（`UpscaleModelLoader.model_name`）—— 名字自带语义，不是族专用旋钮 ✓
    for node in nodes:
        if not re.search(r"UpscaleModelLoader", node["classType"], re.I):
            continue
        value = node["inputs"].get("model_name")
        if isinstance(value, str):
            bindings["upscale_model"] = {
                "node": node["id"],
                "input": "model_name",
                "from": "params.upscale_model",
                "default": value,
            }
        break


# ── 名字自带语义的通用旋钮 ────────────────────────────────────────────────
# 只收这几个：输入名本身就说明它是什么（`resolution` / `batch_size` / `color_correction`），
# 本仓 registry 里跨族出现（Qwen-Image、SeedVR2 都绑）。
# 其余族专用旋钮（`rho` / `w_max` / `lowres_scale` / `transition_step` / `blocks_to_swap` /
# `task_type` / 各种 cache 开关）**一律不推** ✗ —— 那些值域与含义只有该族自己知道，
# 猜一个默认值填进去 = 图照跑、效果变样、不报错 ✗
_KNOB_INPUTS = (
    ("resolution", "params.resolution", "int"),
    ("batch_size", "params.batch_size", "int"),
    ("color_correction", "params.color_correction", None),
)


def _bind_knobs(nodes: list[dict[str, Any]], bindings: dict[str, Any]) -> None:
    """通用旋钮 ✓ —— 已经绑过的格（如空 latent 的 ``batch_size``）不重复绑 ✗。"""
    taken = {
        (str(spot.get("node")), spot.get("input"))
        for spot in bindings.values()
        if isinstance(spot, dict)
    }
    for node in nodes:
        for input_name, source, transform in _KNOB_INPUTS:
            if input_name in bindings or (node["id"], input_name) in taken:
                continue
            value = node["inputs"].get(input_name)
            if _is_link(value):
                continue
            if transform is None:
                if not isinstance(value, str):
                    continue
                spot: dict[str, Any] = {"node": node["id"], "input": input_name, "from": source, "default": value}
            else:
                if not _is_number(value):
                    continue
                spot = {
                    "node": node["id"],
                    "input": input_name,
                    "from": source,
                    "transform": transform,
                    "default": int(value),
                }
            bindings[input_name] = spot
            taken.add((node["id"], input_name))


# ── 汇总 ──────────────────────────────────────────────────────────────────
def _parameters(bindings: dict[str, Any]) -> list[dict[str, Any]]:
    """给前端的参数表 ✓ —— 只列**能确定上下限**的几个；采样器/调度器不列 ✗
    （全集得由人给，列一个没有 options 的 select 只会让人选不出来 ✗）。
    """
    rows: list[dict[str, Any]] = []
    if "seed" in bindings:
        rows.append(
            {
                "key": "seed",
                "label": "种子",
                "type": "number",
                "min": 1,
                "max": SEED_MAX,
                "step": 1,
                "description": "留空则每次随机",
            }
        )
    if "steps" in bindings:
        via_sigmas = bindings["steps"].get("transform") == "qwenSigmas"
        rows.append(
            {
                "key": "steps",
                "label": "步数（落 sigmas 串）" if via_sigmas else "步数",
                "type": "number",
                "min": 1,
                "max": 50,
                "step": 1,
                "description": "该族按 sigmas 串喂轨迹，改步数会整条重算" if via_sigmas else "",
            }
        )
    if "cfg" in bindings:
        rows.append({"key": "cfg", "label": "CFG", "type": "number", "min": 0, "max": 30, "step": 0.1})
    if "denoise" in bindings:
        rows.append({"key": "denoise", "label": "重绘强度", "type": "number", "min": 0, "max": 1, "step": 0.05})
    return rows


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "workflow"


def infer(graph: Any, *, name: str | None = None, hint: str | None = None) -> dict[str, Any]:
    """推断一张 API 图 ✓ —— 返回 ``bindings`` / ``parameters`` / 落点清单 / warnings ✓。

    推不出来的落点**如实进 ``missing``** ✓（前端据此把那几个键藏起来，而不是给一个填了
    不生效的输入框 ✗）。
    """
    if not catalog.is_api_format(graph):
        raise InferError(
            "这张图不是 **API 格式** ⇒ 推断不了（也提交不了）。请在 ComfyUI 里用 "
            "「Save (API Format)」导出，或走 ``/prompt`` 的 API 图 ✗",
            code="ui_format",
        )
    nodes = _nodes(graph)
    if not nodes:
        raise InferError("图里一个节点都没有 ✗", code="empty")

    wf_type = infer_type(nodes, hint)
    bindings: dict[str, Any] = {}
    warnings: list[str] = []
    _bind_text(nodes, bindings, warnings)
    _bind_size(nodes, bindings, warnings)
    _bind_sampler(nodes, bindings, warnings)
    _bind_references(nodes, bindings, warnings)
    _bind_models(nodes, bindings)
    _bind_knobs(nodes, bindings)

    if "prompt" not in bindings:
        warnings.append("**没找到提示词落点** ⇒ 这张图不能用可编辑文本驱动（只当固定模板提交 ✓）")
    if "seed" not in bindings:
        warnings.append("没找到种子落点 ⇒ 每次跑都是图里写死的 seed（想变只能改图）")

    expected = ["prompt", "negative", "width", "height", "seed", "steps", "cfg"]
    if wf_type == "video":
        expected += ["length", "startFrame"]
    missing = [key for key in expected if key not in bindings]

    return {
        "type": wf_type,
        "bindings": bindings,
        "parameters": _parameters(bindings),
        "requiredNodes": required_nodes(graph),
        "nodeCount": len(nodes),
        "found": sorted(bindings),
        "missing": missing,
        "warnings": warnings,
        "suggested": {
            "id": f"{_PREFIX[wf_type]}-{_slug(name or '')}",
            "name": name or "",
            "type": wf_type,
        },
    }
