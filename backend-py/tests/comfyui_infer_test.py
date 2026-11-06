"""未登记工作流的**参数落点推断**守卫（`comfyui_infer`，2026-11-03 立）。

为什么要这套：推断错**不报错**，只会「图照跑、参数没生效」✗ —— 用户以为改了提示词/种子，
实际那张格根本不是驱动图的那一格。所以每条判据都钉在**会静默走样**的地方：

1. **连线格不许覆盖** ✗：图上 ``positive`` / ``width`` 本来是 ``["5", 0]``，被推断成可填项后
   一填就**拆线**（图照跑、产物全变样、不报错 ✗）⇒ 每类都反着验一次「图里是连线 ⇒ 不出这个键」；
2. **正负向不许混淆** ✗：``TextEncodeQwenImage21`` 是 ``prompt`` + ``negative_prompt`` **同节点两格**，
   而 Krea2 那种是**两个节点**（「负向编码」+「四视图提示词」）⇒ 两半都要验，
   且负向必须**保留图上原值**作默认（抹掉别人写好的负向 = 静默降质 ✗）；
3. **与 21 条已登记工作流逐格对拍** ✓：注册表里的手写落点就是金标准 ——
   只要**图上真有那一格**，推断必须命中同一格（未连线的可选数组槽不会出现在 API 图里，
   那类不作基准，见 ``_cell_exists``）；
4. **推不出来就说推不出来** ✓：族专用旋钮（``rho`` / ``w_max`` / ``task_type`` / cache 开关）
   与采样器枚举**一律不绑** —— 绑成自由文本等于让人填一个不存在的名字 ✗；
5. **纯函数**：不写盘、不连上游、不改原图（跑两遍结果必须逐字段相等）。

运行::

    ./.venv/Scripts/python.exe tests/comfyui_infer_test.py
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
#: ⚠️ cp936 控制台下 ``✓`` 会 ``UnicodeEncodeError`` ⇒ 判据全过也崩在最后一行打印上 ✗
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(BACKEND_PY))

from app.services import comfyui_catalog as cat  # noqa: E402
from app.services import comfyui_infer as ci  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ────────────────────────────── 造图小工具 ──────────────────────────────

def node(class_type: str, inputs: dict, title: str = "") -> dict:
    row: dict = {"class_type": class_type, "inputs": inputs}
    if title:
        row["_meta"] = {"title": title}
    return row


def graph_of(*rows: dict) -> dict:
    return {str(index): row for index, row in enumerate(rows, start=1)}


def cells(result: dict) -> set[tuple[str, str]]:
    return {(spot.get("node"), spot.get("input")) for spot in result["bindings"].values()}


# ────────────────────────────── ① 闸口 ──────────────────────────────

def case_gates() -> None:
    ui = {"nodes": [], "links": []}
    try:
        ci.infer(ui)
        check("① UI 格式必须拒", False, "没抛")
    except ci.InferError as exc:
        check("① UI 格式必须拒", exc.code == "ui_format", exc.code)
    try:
        ci.infer({})
        check("① 空图必须拒", False, "没抛")
    except ci.InferError as exc:
        check("① 空图必须拒", exc.code in {"ui_format", "empty"}, exc.code)
    try:
        ci.infer(graph_of(node("KSampler", {"seed": 1})))
        check("① 没有输出节点必须拒", False, "没抛")
    except ci.InferError as exc:
        check("① 没有输出节点必须拒", exc.code == "no_output", exc.code)
    g = graph_of(node("SaveImage", {"images": ["2", 0]}))
    try:
        ci.infer(g, hint="picture")
        check("① type 别名必须拒", False, "没抛")
    except ci.InferError as exc:
        check("① type 别名必须拒", exc.code == "bad_hint", exc.code)
    check("① hint 覆盖推断结果", ci.infer(g, hint="video")["type"] == "video")


def case_infer_type() -> None:
    check("① SaveImage ⇒ image", ci.infer(graph_of(node("SaveImage", {"images": []})))["type"] == "image")
    check(
        "① VHS_VideoCombine ⇒ video",
        ci.infer(graph_of(node("VHS_VideoCombine", {"images": []})))["type"] == "video",
    )
    check("① SaveAudio ⇒ audio", ci.infer(graph_of(node("SaveAudio", {"audio": []})))["type"] == "audio")
    check(
        "① 视频优先于图片（两类输出都在时）",
        ci.infer(graph_of(node("SaveImage", {"images": []}), node("SaveVideo", {"video": []})))["type"] == "video",
    )


# ────────────────────────────── ② 文本 ──────────────────────────────

def _text_graph(**kwargs) -> dict:
    """采样器 positive/negative 各指一个编码器。"""
    return graph_of(
        node("KSampler", {"positive": ["2", 0], "negative": ["3", 0], "seed": 7, "steps": 8, "cfg": 3.5}),
        node("CLIPTextEncode", {"text": "a cat"}, "正向编码"),
        node("CLIPTextEncode", {"text": "blurry, low quality"}, "负向编码"),
        node("SaveImage", {"images": []}),
    )


def _text_graph_concat() -> dict:
    """真文本在**第二跳**：正向上挂的是编码器，编码器的 `prompt` 又是去拼接节点的连线。"""
    return graph_of(
        node("KSampler", {"positive": ["2", 0], "negative": ["3", 0], "seed": 7, "steps": 8, "cfg": 3.5}),
        node("TextEncodeQwenImage21", {"prompt": ["4", 0], "negative_prompt": "artifacts"}),
        node("CLIPTextEncode", {"text": "blurry, low quality"}, "负向编码"),
        node("QwenAnglePromptConcat", {"base_prompt": "a cat", "camera_text": "45 degree"}, "拼接"),
        node("SaveImage", {"images": []}),
    )


def case_text_links() -> None:
    result = ci.infer(_text_graph())
    check("② 正向落正向编码器", result["bindings"]["prompt"] == {"node": "2", "input": "text"}, result["bindings"].get("prompt"))
    check("② 负向落负向编码器", result["bindings"]["negative"]["node"] == "3", result["bindings"].get("negative"))
    check("② 负向保留图上原值作默认", result["bindings"]["negative"]["default"] == "blurry, low quality")
    check("② 保留负向要给 warning", any("负向" in line for line in result["warnings"]), result["warnings"])

    # 拼接节点（正向连线再往下一跳才是真文本，即真实的 image-qwen21-angle 形状）
    result = ci.infer(_text_graph_concat())
    check(
        "② 第二跳找得到 base_prompt",
        result["bindings"]["prompt"] == {"node": "4", "input": "base_prompt"},
        result["bindings"].get("prompt"),
    )
    check("② 第二跳不影响负向", result["bindings"]["negative"]["node"] == "3", result["bindings"].get("negative"))


def case_text_same_node() -> None:
    """`TextEncodeQwenImage21`：prompt 与 negative_prompt **同节点两格**。"""
    g = graph_of(
        node("CFGGuider", {"positive": ["2", 0], "negative": ["2", 1], "cfg": 1}),
        node("TextEncodeQwenImage21", {"prompt": "", "negative_prompt": "artifacts", "resolution": 1024}),
        node("SaveImage", {"images": []}),
    )
    result = ci.infer(g)
    check("② 同节点：正向落 prompt", result["bindings"]["prompt"] == {"node": "2", "input": "prompt"}, result["bindings"].get("prompt"))
    check(
        "② 同节点：负向落 negative_prompt（不许因同节点被去重掉）",
        result["bindings"].get("negative", {}).get("input") == "negative_prompt",
        result["bindings"].get("negative"),
    )


def case_text_negative_title() -> None:
    """没有连线时按标题兜底：标题写「负向」的节点不许被当成正向。"""
    g = graph_of(
        node("TextEncodeKrea2", {"prompt": "positive text"}, "四视图提示词"),
        node("TextEncodeKrea2", {"prompt": ""}, "负向编码"),
        node("SaveImage", {"images": []}),
    )
    result = ci.infer(g)
    check("② 兜底：正向不含负向节点", result["bindings"]["prompt"]["node"] == "1", result["bindings"].get("prompt"))
    check("② 兜底：负向落负向节点", result["bindings"]["negative"]["node"] == "2", result["bindings"].get("negative"))


def case_text_link_untouched() -> None:
    """正向本身就是连线（不是文本格）⇒ 不许绑，必须报「没找到提示词落点」。"""
    g = graph_of(
        node("CFGGuider", {"positive": ["3", 0], "cfg": 1}),
        node("SomethingWeird", {"text": ["4", 0]}),
        node("TextCombine", {"a": "x"}),
        node("SaveImage", {"images": []}),
    )
    result = ci.infer(g)
    check("② 找不到就如实进 missing", "prompt" in result["missing"], result["missing"])
    check("② 找不到要报 warning", any("提示词" in line for line in result["warnings"]), result["warnings"])


# ────────────────────────────── ③ 尺寸 / 时长 ──────────────────────────────

def case_size() -> None:
    g = graph_of(
        node("EmptySD3LatentImage", {"width": 1024, "height": 768, "batch_size": 1}),
        node("SaveImage", {"images": []}),
    )
    result = ci.infer(g)
    check("③ 宽 snaps16 + default", result["bindings"]["width"]["transform"] == "snap16" and result["bindings"]["width"]["default"] == 1024)
    check("③ 高 snaps16", result["bindings"]["height"]["input"] == "height")
    check("③ batch 走 count", result["bindings"]["batch"]["from"] == "count", result["bindings"].get("batch"))

    g = graph_of(
        node("EmptySD3LatentImage", {"width": ["2", 0], "height": 768, "batch_size": 1}),
        node("MathAdd", {"a": 1}),
        node("SaveImage", {"images": []}),
    )
    result = ci.infer(g)
    check("③ 宽是连线 ⇒ 不绑", "width" not in result["bindings"], result["bindings"].get("width"))
    check("③ 高照旧能绑", result["bindings"]["height"]["input"] == "height")


def case_length() -> None:
    h3 = graph_of(
        node("MiniMaxH3AudioConditioningT8", {"width": 320, "height": 320, "length": 124, "prompt": "x"}),
        node("SaveVideo", {"video": []}),
    )
    result = ci.infer(h3)
    check("③ H3 帧族 17k+5 @24fps", result["bindings"]["length"]["transform"] == "secondsToFrames17", result["bindings"].get("length"))
    check("③ H3 fps=24", result["bindings"]["length"].get("fps") == 24)
    check("③ 视频宽 snaps32", result["bindings"]["width"]["transform"] == "snap32")
    check("③ 时长要报依据", any("17k+5" in line for line in result["warnings"]), result["warnings"])

    wan = graph_of(
        node("WanImageToVideo", {"width": 832, "height": 480, "length": 81}),
        node("SaveVideo", {"video": []}),
    )
    result = ci.infer(wan)
    check("③ Wan 帧族 4n+1 @16fps", result["bindings"]["length"]["transform"] == "secondsToFrames" and result["bindings"]["length"]["fps"] == 16, result["bindings"].get("length"))


# ────────────────────────────── ④ 采样 ──────────────────────────────

def case_sampler() -> None:
    g = graph_of(
        node("KSampler", {"seed": 42, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}),
        node("SaveImage", {"images": []}),
    )
    result = ci.infer(g)
    check("④ seed 随机", result["bindings"]["seed"]["random"] is True and result["bindings"]["seed"]["transform"] == "int")
    check("④ steps / cfg 落同一节点", result["bindings"]["steps"]["input"] == "steps" and result["bindings"]["cfg"]["input"] == "cfg")
    check("④ 采样器不绑（值域要在线才拿得到）", "sampler" not in result["bindings"], sorted(result["bindings"]))
    check("④ 调度器不绑", "scheduler" not in result["bindings"])
    check("④ denoise=1 是纯文生图 ⇒ 不绑", "denoise" not in result["bindings"])

    g2 = graph_of(
        node("LoadImage", {"image": "a.png"}),
        node("KSampler", {"seed": 42, "steps": 20, "cfg": 6.0, "denoise": 0.6}),
        node("SaveImage", {"images": []}),
    )
    check("④ 有参考图的 img2img 才绑 denoise", ci.infer(g2)["bindings"].get("denoise", {}).get("input") == "denoise", sorted(ci.infer(g2)["bindings"]))

    g3 = graph_of(
        node("KSamplerAdvanced", {"noise_seed": 1, "steps": 10}),
        node("KSamplerAdvanced", {"noise_seed": 2, "steps": 10}),
        node("SaveImage", {"images": []}),
    )
    result = ci.infer(g3)
    check("④ 双采样器都要绑（不然改种子只变一段）", {"seed", "seed2"} <= set(result["bindings"]), sorted(result["bindings"]))
    check("④ 双种子同源 params.seed", result["bindings"]["seed2"]["from"] == "params.seed")

    g4 = graph_of(node("KSampler", {"seed": ["9", 0], "steps": 8}), node("RandomNoise", {}), node("SaveImage", {"images": []}))
    result = ci.infer(g4)
    check("④ seed 是连线 ⇒ 不绑", "seed" not in result["bindings"], result["bindings"].get("seed"))
    check("④ 没种子要报 warning", any("种子" in line for line in result["warnings"]), result["warnings"])


def case_qwen_sigmas() -> None:
    """Qwen-Image 2512：步数落在 `sigmas` 串上（`qwenSigmas` 重算轨迹）。"""
    g = graph_of(
        node("ManualSigmas", {"sigmas": "1.0, 0.9, 0.7, 0.4, 0"}),
        node("SaveImage", {"images": []}),
    )
    result = ci.infer(g)
    steps = result["bindings"].get("steps")
    check("④ sigmas 串认得出步数", steps is not None and steps["transform"] == "qwenSigmas", steps)
    check("④ 步数=数个数−1", steps.get("default") == 4, steps)
    row = next((item for item in result["parameters"] if item["key"] == "steps"), None)
    check("④ 参数表要注明落 sigmas", row is not None and "sigmas" in row["label"], row)


# ────────────────────────────── ⑤ 参考素材 ──────────────────────────────

def case_images() -> None:
    g = graph_of(
        node("LoadImage", {"image": "a.png"}),
        node("LoadImage", {"image": "b.png"}),
        node("CLIPTextEncode", {"text": "keep the pose"}, "正向编码"),
        node("SaveImage", {"images": []}),
    )
    result = ci.infer(g)
    check("⑤ 两张图 ⇒ image / image2", result["bindings"]["image"]["input"] == "image" and result["bindings"]["image2"]["imageIndex"] == 1)
    check("⑤ 有提示词落点时不标必填", "required" not in result["bindings"]["image"], result["bindings"].get("image"))
    check("⑤ 不标必填要说清后果", any("占位" in line for line in result["warnings"]), result["warnings"])

    pure = ci.infer(
        graph_of(node("LoadImage", {"image": "a.png"}), node("UpscaleModelLoader", {"model_name": "x.pth"}), node("SaveImage", {"images": []}))
    )
    check("⑤ 纯变换图（无提示词/无空 latent）⇒ 必填", pure["bindings"]["image"].get("required") is True, pure["bindings"].get("image"))


def case_video_input() -> None:
    g = graph_of(
        node("LoadVideo", {"file": "in.mp4"}),
        node("SeedVR2VideoUpscaler", {"resolution": 1080, "batch_size": 5, "color_correction": "lab", "seed": 1}),
        node("SaveVideo", {"video": []}),
    )
    result = ci.infer(g)
    spot = result["bindings"].get("video")
    check("⑤ 视频落 LoadVideo.file", spot is not None and spot["input"] == "file" and spot["from"] == "video", spot)
    check("⑤ 纯变换图里视频必填", spot.get("required") is True)


def case_array_refs() -> None:
    g = graph_of(
        node("TextEncodeQwenImage21", {"prompt": "", "images.image_1": ["3", 0], "images.image_2": ["3", 0]}),
        node("EmptyLatentImage", {"width": 1024, "height": 1024, "batch_size": 1}),
        node("SaveImage", {"images": []}),
    )
    result = ci.infer(g)
    refs = result["bindings"].get("refs")
    check("⑤ 前缀不写死 ref_image_（images.image_N 也要认）", refs is not None, sorted(result["bindings"]))
    check("⑤ 前缀与 indexBase 按图给", refs.get("inputPrefix") == "images.image_" and refs.get("indexBase") == 1, refs)
    check("⑤ array 口径", refs.get("array") is True and refs.get("max") == 2)
    check("⑤ 展开的格要报出来", any("images.image_1" in line for line in result["warnings"]), result["warnings"])

    audio = ci.infer(
        graph_of(
            node("TextEncodeQwenImage21", {"prompt": "", "ref_audio_0": ["2", 0]}),
            node("SaveImage", {"images": []}),
        )
    )
    check("⑤ 音/视频按槽不猜（插槽类型看节点定义）", "refs" not in audio["bindings"] or audio["bindings"]["refs"].get("inputPrefix", "").find("audio") < 0, audio["bindings"].get("refs"))
    check("⑤ 不猜要说一句", any("音/视频" in line for line in audio["warnings"]), audio["warnings"])


# ────────────────────────────── ⑥ 模型 / 旋钮 ──────────────────────────────

def case_models() -> None:
    g = graph_of(
        node("UNETLoader", {"unet_name": "m.safetensors"}),
        node("LoraLoaderModelOnly", {"lora_name": "l.safetensors", "strength_model": 0.8}),
        node("SaveImage", {"images": []}),
    )
    result = ci.infer(g)
    check("⑥ unet 落 unet_name", result["bindings"]["unet"]["default"] == "m.safetensors")
    check("⑥ lora 与 lora_strength 成对", result["bindings"].get("lora", {}).get("input") == "lora_name" and result["bindings"]["lora_strength"]["input"] == "strength_model", sorted(result["bindings"]))
    check("⑥ lora_strength 保留原值", result["bindings"]["lora_strength"]["default"] == 0.8)

    ck = ci.infer(graph_of(node("CheckpointLoaderSimple", {"ckpt_name": "c.safetensors"}), node("SaveImage", {"images": []})))
    check("⑥ ckpt 落 ckpt_name", ck["bindings"]["ckpt"]["default"] == "c.safetensors")


def case_knobs() -> None:
    g = graph_of(
        node("SeedVR2VideoUpscaler", {"resolution": 1080, "batch_size": 5, "color_correction": "lab"}),
        node("SaveVideo", {"video": []}),
    )
    result = ci.infer(g)
    check("⑥ resolution 落 params.resolution", result["bindings"].get("resolution", {}).get("from") == "params.resolution")
    check("⑥ batch_size 落 params.batch_size", result["bindings"].get("batch_size", {}).get("from") == "params.batch_size")
    check("⑥ color_correction 落 params.color_correction", result["bindings"].get("color_correction", {}).get("default") == "lab")

    latent = ci.infer(graph_of(node("EmptyLatentImage", {"width": 512, "height": 512, "batch_size": 4}), node("SaveImage", {"images": []})))
    check("⑥ 空 latent 的 batch_size 归 count，不重复绑", "batch" in latent["bindings"] and "batch_size" not in latent["bindings"], sorted(latent["bindings"]))


# ────────────────────────────── ⑦ 与注册表逐格对拍 ──────────────────────────────

#: 本仓 registry 里**核心**落点的键（族专用旋钮不在内：那些值域只有该族知道）
_CORE_KEYS = {
    "prompt", "negative", "width", "height", "seed", "seed1", "seed2", "seed3", "steps", "cfg",
    "batch", "length", "image", "image2", "video", "unet", "ckpt", "lora", "lora_strength",
    "upscale_model", "resolution", "batch_size", "color_correction", "refs", "frames",
}


def _cell_exists(graph: dict, spot: dict) -> bool:
    """注册表指的格在**图上真有**吗？

    ⚠️ 未连线的可选输入**不会被序列化**（``TextEncodeQwenImage21.images.image_1…4``、
    ``MiniMaxH3AudioConditioningT8.ref_images.*``、H3 的 ``first_frame``）⇒ 图上根本没有这一格，
    任何推断都不可能发现它 ⇒ 这类不作对拍基准（如实记在 missing 里）。
    """
    inputs = (graph.get(str(spot.get("node"))) or {}).get("inputs") or {}
    if spot.get("input") in inputs:
        return True
    prefix = spot.get("inputPrefix")
    return bool(prefix) and any(str(name).startswith(str(prefix)) for name in inputs)


def case_registry_parity() -> None:
    registry = cat.load_registry()
    rows = cat.entries(registry, include_hidden=True)
    mismatched: list[str] = []
    uninferable: list[str] = []
    for entry in rows:
        graph = cat.load_graph(entry)
        result = ci.infer(graph, name=entry.get("id"))
        got = cells(result)
        for key, spot in (entry.get("bindings") or {}).items():
            if key not in _CORE_KEYS:
                continue
            if not _cell_exists(graph, spot):
                uninferable.append(f"{entry.get('id')}:{spot.get('node')}.{spot.get('input')}")
                continue
            want = (str(spot.get("node")), spot.get("input"))
            if want not in got:
                mismatched.append(f"{entry.get('id')}: {key} 应落 {want[0]}.{want[1]}，推断没落")
    check(f"⑦ {len(rows)} 条已登记工作流核心落点逐格命中", not mismatched, mismatched)
    check("⑦ 跳过的都是图上不存在的格（未连线可选槽）", len(uninferable) >= 1, uninferable)


def case_registry_links_never_overwritten() -> None:
    """注册表那条铁律：推断**永不**绑一个图里是连线的格。"""
    bad: list[str] = []
    for entry in cat.entries(cat.load_registry(), include_hidden=True):
        graph = cat.load_graph(entry)
        result = ci.infer(graph, name=entry.get("id"))
        for key, spot in result["bindings"].items():
            inputs = (graph.get(str(spot.get("node"))) or {}).get("inputs") or {}
            if ci._is_link(inputs.get(spot.get("input"))):
                bad.append(f"{entry.get('id')}: {key} -> {spot.get('node')}.{spot.get('input')}")
    check("⑦ 21 条真图里没有一条绑到连线格", not bad, bad)


# ────────────────────────────── ⑧ 纯函数 ──────────────────────────────

def case_pure() -> None:
    entry = cat.get_entry("image-qwen21")
    graph = cat.load_graph(entry)
    before = copy.deepcopy(graph)
    first = ci.infer(graph, name="x")
    second = ci.infer(graph, name="x")
    check("⑧ 不改原图", graph == before)
    check("⑧ 两遍结果一致", first == second)
    check("⑧ suggested 不动调用方名字（只给建议 id）", first["suggested"]["id"] == "image-x" and first["suggested"]["name"] == "x", first["suggested"])
    check("⑧ 回执字段齐", {"type", "bindings", "parameters", "requiredNodes", "nodeCount", "found", "missing", "warnings", "suggested"} <= set(first), sorted(first))


def main() -> None:
    case_gates()
    case_infer_type()
    case_text_links()
    case_text_same_node()
    case_text_negative_title()
    case_text_link_untouched()
    case_size()
    case_length()
    case_sampler()
    case_qwen_sigmas()
    case_images()
    case_video_input()
    case_array_refs()
    case_models()
    case_knobs()
    case_registry_parity()
    case_registry_links_never_overwritten()
    case_pure()

    failures = 0
    for name, ok, detail in _RESULTS:
        if ok:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}\n      {detail!r}")
    print(f"\nSUMMARY: {len(_RESULTS) - failures}/{len(_RESULTS)} passed")


if __name__ == "__main__":
    main()
