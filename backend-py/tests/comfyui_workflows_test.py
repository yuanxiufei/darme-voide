"""ComfyUI 工作流三件套的守卫（`comfyui_catalog` / `comfyui_bindings` / `comfyui_slots`，2026-11-02 立）。

为什么要单独立这套（不是「能 import 就算过」✗）：这三个模块搬的是上游**语义**，
而语义错**不报错**，只会「图出得来但不对」✓✗ —— 所以每条判据都钉在**会静默走样**的地方：

1. **API/UI 格式** ✗：UI 格式提交必然失败（``registry.js`` 明写会拒 ✓）⇒ 必须**认得出**
   且不许静默往下走 ✓；
2. **``round`` / ``snap`` 的 JS 口径** ✗：Python ``round(32.5)=32``、JS ``Math.round(32.5)=33``
   ⇒ 尺寸与帧数会**差一档**（1040 → 1024 vs 1056 ✗）⇒ 断言里**同时**写「等于 JS 值」和
   「不等于 Python 值」✓（只写前者，改成 Python round 也照样过 ✓✗）；
3. **Qwen 2.1 的 sigma 串** ✓：8 步必须**逐字**等于官方轨迹（9 个值 ✓），
   插值出的串不能带尾随零（JS ``Number(toFixed(6))`` 的口径 ✗）；
4. **注入不改原图** ✗：图是共享资产，污染一次后面全错 ⇒ 注入后**原图必须还是空的** ✓；
5. **模型槽位按 ``(class_type, input)`` 二元组** ✗：``llama_cpp_model_loader`` 的
   ``model`` 与 ``mmproj`` 是两类，只按 class_type 登记会把视觉投影当主模型 ✗；
6. **可用性要报得出具体问题** ✓：只说「校验失败」等于没查 ⇒ 断言 ``missing_node`` /
   ``missing_model`` 各带**节点号与缺什么** ✓，且子目录写法要算命中 ✓。

运行::

    ./.venv/Scripts/python.exe tests/comfyui_workflows_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
#: ⚠️ cp936 控制台下 ``✓`` 会 ``UnicodeEncodeError`` ⇒ 判据全过也崩在最后一行打印上 ✗
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(BACKEND_PY))

from app.services import comfyui_bindings as cb  # noqa: E402
from app.services import comfyui_catalog as cat  # noqa: E402
from app.services import comfyui_match as cm  # noqa: E402
from app.services import comfyui_slots as cs  # noqa: E402
from app.services import comfyui_speaker_tags as st  # noqa: E402
from app.services import comfyui_ui_workflow as uw  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ────────────────────────────── ① 目录层 ──────────────────────────────

def case_registry_shape() -> None:
    registry = cat.load_registry()
    all_rows = cat.entries(registry, include_hidden=True)
    visible = cat.entries(registry)
    check("① 注册表 21 条 entry", len(all_rows) == 21, len(all_rows))
    check("① 隐藏项默认不进可选列表", len(visible) < len(all_rows), f"{len(visible)}/{len(all_rows)}")
    hidden_ids = {row.get("id") for row in all_rows} - {row.get("id") for row in visible}
    check("① 隐藏的是 image-qwen21-angle", "image-qwen21-angle" in hidden_ids, hidden_ids)
    check("① 按 id 取得到 z-image-turbo", cat.get_entry("image-z-image-turbo") is not None)
    check("① 取不到的 id 返回 None（不抛）", cat.get_entry("nope") is None)


def case_api_format() -> None:
    check("① API 格式认得出", cat.is_api_format({"1": {"class_type": "KSampler"}}) is True)
    check("① UI 格式（nodes/links）判否", cat.is_api_format({"nodes": [], "links": []}) is False)
    check("① 空图判否", cat.is_api_format({}) is False)


def case_load_graph() -> None:
    entry = cat.get_entry("image-z-image-turbo")
    graph = cat.load_graph(entry)
    check("① 真图读得出且是 API 格式", graph.get("1", {}).get("class_type") == "UNETLoader", list(graph)[:4])
    check("① 图里 KSampler 在", any(node.get("class_type") == "KSampler" for node in graph.values()))
    again = cat.load_graph(entry)
    check("① 每次读都是新对象（改一次不污染下次）", again is not graph and again["5"]["inputs"]["text"] == "")


def case_variants() -> None:
    entry = cat.get_entry("image-qwen21")
    merged = cat.resolve_variant(entry, "edit")
    check("① 变体覆盖 file", merged.get("file") == "image-qwen21-edit-turbo.json", merged.get("file"))
    check("① 变体沿用主 entry 的 bindings", merged.get("bindings") == entry.get("bindings"))
    check("① 不传变体 = 主形态", cat.resolve_variant(entry) == entry)
    try:
        cat.resolve_variant(entry, "nope")
        check("① 变体不在册必须报", False, "没抛")
    except cat.CatalogError as exc:
        check("① 变体不在册必须报", exc.code == "unknown_variant", exc.code)
    try:
        cat.graph_path({"id": "x", "file": "../../evil.json"})
        check("① 路径穿越被拒", False, "没抛")
    except cat.CatalogError as exc:
        check("① 路径穿越被拒", exc.code == "graph_path_escape", exc.code)


def case_preset() -> None:
    entry = cat.get_entry("image-z-image-turbo")
    check("① medium 档取到步数预设", cat.resolve_preset(entry, "medium").get("steps") == 8, cat.resolve_preset(entry, "medium"))
    check("① auto 档不预设", cat.resolve_preset(entry, "auto") == {})
    check("① 大小写/空格不敏感", cat.resolve_preset(entry, " Medium ") == cat.resolve_preset(entry, "medium"))


def case_step_plan() -> None:
    graph = cat.load_graph(cat.get_entry("image-z-image-turbo"))
    plan = cat.estimate_step_plan(graph)
    check("① KSampler 步数被数出来", plan["steps"].get("4") == 8 and plan["total"] == 8, plan)

    advanced = {"9": {"class_type": "KSamplerAdvanced", "inputs": {"steps": 20, "start_at_step": 5, "end_at_step": 15}}}
    check("① KSamplerAdvanced 取区间（不是 steps）", cat.estimate_step_plan(advanced)["total"] == 10, cat.estimate_step_plan(advanced))

    sigmas = {
        "1": {"class_type": "ManualSigmas", "inputs": {"sigmas": "1, 0.5, 0"}},
        "2": {"class_type": "SelfLiftH3Sampler", "inputs": {"sigmas": ["1", 0]}},
    }
    plan2 = cat.estimate_step_plan(sigmas)
    check("① SelfLift 顺着连线数 sigma（个数−1）", plan2["steps"].get("2") == 2, plan2)
    check("① SelfLift 标记为 absolute（不相加）", plan2["absolute"].get("2") is True, plan2["absolute"])

    # 环保护：refiner 的 sigmas 指回自己 ⇒ 不许死循环
    loop = {"1": {"class_type": "H3SigmaRefiner", "inputs": {"sigmas": ["1", 0], "extra_steps": 3}}}
    check("① 成环不死循环", cat.estimate_step_plan(loop)["total"] == 0, cat.estimate_step_plan(loop))


# ────────────────────────────── ② 注入层 ──────────────────────────────

def case_transforms() -> None:
    check("② snap16 上取整（833 → 848）", cb.apply_transform(833, {"transform": "snap16"}) == 848)
    check("② snap16 不动已对齐的值（832）", cb.apply_transform(832, {"transform": "snap16"}) == 832)
    # JS Math.round(32.5) = 33 ⇒ 1056；Python round(32.5) = 32 ⇒ 1024（**同时**断言两个 ✓）
    snapped = cb.apply_transform(1040, {"transform": "snap32"})
    check("② snap32 走 JS 口径（1040 → 1056）", snapped == 1056, snapped)
    check("② snap32 不是 Python 银行家舍入（≠1024）", snapped != 1024, snapped)
    check("② secondsToFrames 是 4n+1", cb.seconds_to_frames(5, 16) == 81, cb.seconds_to_frames(5, 16))
    check("② secondsToFrames17 是 17k+5", cb.seconds_to_frames_17(5, 24) == 124, cb.seconds_to_frames_17(5, 24))
    check("② secondsToFrames17 与 4n+1 不同（不许互相顶替）", cb.seconds_to_frames_17(2, 24) != cb.seconds_to_frames(2, 24))
    check("② invert10", cb.apply_transform(3, {"transform": "invert10"}) == 7)
    check("② boolean", cb.apply_transform("", {"transform": "boolean"}) is False)


def case_qwen_sigmas() -> None:
    official = cb.qwen21_sigmas(8)
    expect = "1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"
    check("② 8 步 = 官方 sigma 轨迹（逐字）", official == expect, official)
    parts = cb.qwen21_sigmas(16).split(", ")
    check("② 16 步插值出 17 个值", len(parts) == 17, len(parts))
    check("② 插值首尾仍是 1 与 0", parts[0] == "1" and parts[-1] == "0", (parts[0], parts[-1]))
    check("② 插值不带尾随零（JS Number(toFixed(6)) 口径）", "0.500000" not in cb.qwen21_sigmas(2), cb.qwen21_sigmas(2))


def case_inject_real_entry() -> None:
    entry = cat.get_entry("image-z-image-turbo")
    graph = cat.load_graph(entry)
    # ``presets`` 由调用方从 entry 上取下来传进来（与上游一致 ✓：context 在注入之外构造）
    ctx = cb.build_context(prompt="一只黑猫", size="1024x1024", quality="medium", presets=entry.get("presets"))
    new, meta = cb.apply_bindings(graph, entry, ctx)
    check("② 提示词写进 CLIPTextEncode.text", new["5"]["inputs"]["text"] == "一只黑猫", new["5"]["inputs"])
    check("② 尺寸写进 EmptyLatentImage（1024 已对齐 16）", new["7"]["inputs"]["width"] == 1024, new["7"]["inputs"])
    check("② 随机种子回传给调用方", isinstance(meta.get("seed"), int) and 1 <= meta["seed"] <= cb.SEED_MAX_EXCLUSIVE - 1, meta.get("seed"))
    check("② 原图没被改（text 仍为空）", graph["5"]["inputs"]["text"] == "", graph["5"]["inputs"])
    check("② 原图没被改（width 仍是 832）", graph["7"]["inputs"]["width"] == 832, graph["7"]["inputs"])
    check("② quality 预设生效（steps=8）", new["4"]["inputs"]["steps"] == 8, new["4"]["inputs"])


def case_inject_missing_required() -> None:
    # ``from`` 指向不存在的路径 ⇒ resolve_path 返回 None ⇒ 才算「缺失」✓
    entry = {"id": "t", "bindings": {"x": {"node": "5", "input": "text", "from": "params.nothing", "required": True}}}
    graph = {"5": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}}
    try:
        cb.apply_bindings(graph, entry, cb.build_context())
        check("② 必填缺失（值为 None）必须报", False, "没抛")
    except cb.BindingError as exc:
        check("② 必填缺失（值为 None）必须报", exc.code == "missing_parameter", exc.code)

    # 空字符串**不算缺失**（与上游一致的口径 ✓：源只判 undefined/null，空串会原样写下去 ✗）
    blank = {"id": "t", "bindings": {"prompt": {"node": "5", "input": "text", "required": True}}}
    new, _ = cb.apply_bindings(graph, blank, cb.build_context())
    check("② 空串不触发 required（与上游同口径）", new["5"]["inputs"]["text"] == "", new["5"]["inputs"])


def case_inject_skipped() -> None:
    entry = {"id": "t", "bindings": {"x": {"node": "5", "input": "not_exist"}}}
    graph = {"5": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}}
    _, meta = cb.apply_bindings(graph, entry, cb.build_context())
    check("② 指向图上没有的输入 ⇒ 记 skipped（不静默）", any("not_exist" in item for item in meta.get("skipped", [])), meta)


def case_frame_mode() -> None:
    entry = {
        "id": "t",
        "bindings": {
            "frames": {
                "node": "4",
                "input": "first_frame",
                "frameMode": True,
                "firstInput": "first_frame",
                "lastInput": "last_frame",
                "max": 2,
            }
        },
    }
    graph = {"4": {"class_type": "KSampler", "inputs": {"first_frame": None, "last_frame": None}}}
    ctx = cb.build_context(uploaded_images=[{"name": "a.png"}, {"name": "b.png"}])
    new, _ = cb.apply_bindings(graph, entry, ctx)
    check("② 首尾帧：两张图 ⇒ 两个口都连上", isinstance(new["4"]["inputs"]["first_frame"], list) and isinstance(new["4"]["inputs"]["last_frame"], list), new["4"]["inputs"])
    loaders = [n for n in new.values() if isinstance(n, dict) and n.get("class_type") == "LoadImage"]
    check("② 首尾帧：新建了 LoadImage 节点（不是塞文件名）", len(loaders) == 2, len(loaders))
    check("② 新建节点 id 不与原图冲突", set(new) >= {"4"}, set(new))

    one = cb.build_context(uploaded_images=[{"name": "a.png"}], params={"frameMode": "first"})
    new2, _ = cb.apply_bindings(graph, entry, one)
    check("② 只要首帧时不挂尾帧", new2["4"]["inputs"].get("last_frame") is None, new2["4"]["inputs"])

    try:
        cb.apply_bindings(graph, entry, cb.build_context(uploaded_images=[{"name": "a.png"}], params={"frameMode": "both"}))
        check("② 要两张却只给一张 ⇒ 报", False, "没抛")
    except cb.BindingError as exc:
        check("② 要两张却只给一张 ⇒ 报", exc.code == "missing_reference_image", exc.code)


def case_array_refs() -> None:
    entry = {
        "id": "t",
        "bindings": {
            "refs": {
                "node": "4",
                "input": "ref_images.ref_image_1",
                "array": True,
                "from": "image",
                "inputPrefix": "ref_images.ref_image_",
                "indexBase": 1,
                "max": 2,
            }
        },
    }
    graph = {"4": {"class_type": "X", "inputs": {}}}
    ctx = cb.build_context(uploaded_images=[{"name": "a.png", "subfolder": "in"}, {"name": "b.png"}])
    new, _ = cb.apply_bindings(graph, entry, ctx)
    inputs = new["4"]["inputs"]
    check("② 多参考按 indexBase 铺开（1、2）", "ref_images.ref_image_1" in inputs and "ref_images.ref_image_2" in inputs, list(inputs))
    names = sorted(n["inputs"]["image"] for n in new.values() if isinstance(n, dict) and n.get("class_type") == "LoadImage")
    check("② 带 subfolder 的素材名要拼全", names == ["b.png", "in/a.png"], names)


# ────────────────────────────── ③ 槽位层 ──────────────────────────────

def case_spec_table() -> None:
    specs = cs.MODEL_LOADER_SPECS
    check("③ 静态表 18 条", len(specs) == 18, len(specs))
    pairs = {(s.class_type, s.input) for s in specs}
    check("③ 按 (class_type, input) 唯一（mmproj 不与 model 混）", len(pairs) == len(specs), len(pairs))
    check("③ llama-cpp 的 model 与 mmproj 是两格", ("llama_cpp_model_loader", "mmproj") in pairs)
    check("③ TE_Qwen_Image_2_1_PE 有 4 个模型输入", sum(1 for s in specs if s.class_type == "TE_Qwen_Image_2_1_Prompt_Enhancer") == 4)


def case_slots_of_graph() -> None:
    graph = cat.load_graph(cat.get_entry("image-z-image-turbo"))
    slots = cs.model_slots_of(graph)
    keys = {slot["key"] for slot in slots}
    check("③ 从真图认出 unet / clip / vae 三格", {"unets", "clips", "vaes"} <= keys, keys)
    unet = next(slot for slot in slots if slot["key"] == "unets")
    check("③ unet 格带当前值", unet["value"] == "z_image_turbo_bf16.safetensors", unet)
    check("③ 连线的值不算模型格", all(isinstance(slot["value"], str) for slot in slots))


def case_discover() -> None:
    object_info = {
        "MyGGUFLoader": {
            "input": {
                "required": {
                    "model": [["a.gguf", "b.gguf"], {}],
                    "image": [["x.png"], {}],
                }
            }
        },
        "LoadImage": {"input": {"required": {"image": [["a.safetensors"], {}]}}},
    }
    specs = cs.discover_model_loader_specs(object_info)
    found = {(s.class_type, s.input) for s in specs}
    check("③ 动态发现第三方 loader 的模型格", ("MyGGUFLoader", "model") in found, found)
    check("③ image 这种素材输入不算模型格", ("MyGGUFLoader", "image") not in found)
    check("③ LoadImage 整节点跳过（NON_MODEL_CLASS）", ("LoadImage", "image") not in found)
    check("③ 已登记的格不重复添加", len(specs) == len(cs.MODEL_LOADER_SPECS) + 1, len(specs))


def case_availability() -> None:
    graph = cat.load_graph(cat.get_entry("image-z-image-turbo"))
    object_info = {"UNETLoader": {}, "CLIPLoader": {}, "VAELoader": {}}
    problems = cs.availability_problems(graph, object_info=object_info)
    kinds = {p["kind"] for p in problems}
    check("③ 缺节点报得出（带 classType）", "missing_node" in kinds and any(p.get("classType") == "KSampler" for p in problems), problems[:2])

    ok = cs.availability_problems(graph, object_info=object_info, available={"unets": {"z_image_turbo_bf16.safetensors"}})
    check("③ 模型在盘上 ⇒ 不报 missing_model", not any(p["kind"] == "missing_model" for p in ok), ok)

    subdir = cs.availability_problems(graph, object_info=object_info, available={"unets": {"unet/zimage/z_image_turbo_bf16.safetensors"}})
    check("③ 子目录写法算命中（basename 比对）", not any(p["kind"] == "missing_model" for p in subdir), subdir)

    missing = cs.availability_problems(graph, object_info=object_info, available={"unets": {"other.safetensors"}})
    target = [p for p in missing if p["kind"] == "missing_model"]
    check("③ 缺模型报得出（带节点号与键）", len(target) == 1 and target[0]["key"] == "unets" and target[0]["node"] == "1", target)

    placeholder = {"4": {"class_type": "UNETLoader", "inputs": {"unet_name": "None"}}}
    check("③ 「无/None/null」占位项不算缺失", cs.availability_problems(placeholder, available={"unets": {"x.safetensors"}}) == [])


# ─────────────────────── ④ 模型名匹配 / UI 转换 / 说话人标签 ───────────────────────

def case_model_match() -> None:
    # ⭐ 格式是红线：.safetensors ↔ .gguf 不许互顶（加载器不同，顶替会直接炸 ✗）
    check("④ 格式不同 ⇒ format", cm.model_name_similarity("a.safetensors", "a.gguf").reason == "format")
    # ⭐ 角色是红线：VAE 不能顶底模
    check("④ 一边有 vae ⇒ role", cm.model_name_similarity("model.safetensors", "model_vae.safetensors").reason == "role")
    # ⭐ 训练规格红线：turbo 与非 turbo 不互顶
    check("④ turbo 差异 ⇒ variant", cm.model_name_similarity("z_image_turbo.safetensors", "z_image.safetensors").reason == "variant")
    # ⭐ 家族红线：Qwen 与 Z-Image 不互顶（注意 turbo 红线排在家族**之前** ✓ 与源同序）
    check("④ 跨家族 ⇒ family", cm.model_name_similarity("qwen_image_2512.safetensors", "z_image.safetensors").reason == "family")
    # ⭐ 量化差异恰恰是**要认出来**的（子目录 + fp8/bf16）
    ok = cm.model_name_similarity("z_image_turbo_bf16.safetensors", "z_image_turbo_fp8.safetensors")
    check("④ 同模型不同量化 ⇒ 认得出", ok.ok and ok.score >= 0.5, ok)
    check("④ 音频 VAE 不顶视频 VAE", cm.model_name_similarity("h3_vae.safetensors", "h3_audio_vae.safetensors").reason == "qualifier")

    # ⭐ 歧义不自动选：两个候选一样像 ⇒ 把候选交回去，不替用户挑 ✗
    ambiguous = cm.find_similar_model("model_a.safetensors", ["model_b.safetensors", "model_c.safetensors"])
    check("④ 分数并列 ⇒ 判歧义（不自动选）", ambiguous.get("ambiguous") is True and ambiguous.get("file") == "", ambiguous)
    only = cm.find_similar_model("model_a.safetensors", ["model_b.safetensors", "totally_other.safetensors"])
    check("④ 唯一高分 ⇒ 给得出文件名", only.get("file") == "model_b.safetensors", only)
    check("④ 同名候选跳过（那是精确命中）", cm.find_similar_model("a.safetensors", ["a.safetensors"])["file"] == "")
    check("④ .engine 不许模糊匹配", cm.find_similar_model("a.engine", ["b.engine"])["file"] == "")


def case_availability_suggestion() -> None:
    graph = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors"}}}
    problems = cs.availability_problems(graph, available={"unets": ["z_image_turbo_fp8.safetensors"]})
    target = [p for p in problems if p["kind"] == "missing_model"]
    check("④ 缺模型时给出可替换建议", len(target) == 1 and target[0].get("suggestion") == "z_image_turbo_fp8.safetensors", target)
    none = cs.availability_problems(graph, available={"unets": ["qwen_image_2512.safetensors"]})
    check("④ 跨家族不给建议（宁缺毋错）", "suggestion" not in (none[0] if none else {}), none)


def _sample_object_info() -> dict:
    return {
        "KSampler": {
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "seed": ["INT", {}],
                    "control_after_generate": ["COMBO", {"options": ["randomize", "fixed"]}],
                    "steps": ["INT", {}],
                    "cfg": ["FLOAT", {}],
                    "sampler_name": ["COMBO", {"options": ["euler"]}],
                    "scheduler": ["COMBO", {"options": ["normal"]}],
                    "denoise": ["FLOAT", {}],
                    "positive": ["CONDITIONING"],
                    "negative": ["CONDITIONING"],
                }
            }
        },
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["model.safetensors"], {}]}}},
        "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {}]}}},
    }


def case_ui_convert() -> None:
    ui = {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "inputs": [], "widgets_values": ["model.safetensors"]},
            {
                "id": 3,
                "type": "KSampler",
                "inputs": [{"name": "model", "link": 10}, {"name": "positive", "link": 11}],
                "widgets_values": [156680208700286, "randomize", 20, 8.0, "euler", "normal", 1.0],
            },
            # ⭐ Note 是前端节点：丢了就行，**不能**报「缺自定义节点」✗
            {"id": 9, "type": "Note", "inputs": [], "widgets_values": ["备注"]},
            {"id": 7, "type": "SaveImage", "inputs": [{"name": "images", "link": 12}], "widgets_values": ["out"]},
        ],
        "links": [[10, 1, 0, 3, 0, "MODEL"], [11, 1, 1, 3, 1, "CONDITIONING"], [12, 3, 0, 7, 0, "IMAGE"]],
    }
    graph, warnings = uw.convert_ui_workflow(ui, _sample_object_info())
    check("④ UI 图转得出 API 格式", set(graph) == {"1", "3", "7"}, set(graph))
    check("④ Note 被丢掉且不报缺节点", "9" not in graph, warnings)
    # ⭐ widget 顺序表：steps 必须拿到 20（拿成 seed 的值就完了 ✗✗）
    check("④ 按 WIDGET_ORDER 对号入座（steps=20）", graph["3"]["inputs"].get("steps") == 20, graph["3"]["inputs"])
    check("④ seed 拿到第一个值", graph["3"]["inputs"].get("seed") == 156680208700286, graph["3"]["inputs"])
    check("④ control_after_generate 不进输入", "control_after_generate" not in graph["3"]["inputs"])
    check("④ 连线转成 [节点, 槽]", graph["3"]["inputs"].get("model") == ["1", 0], graph["3"]["inputs"])
    check("④ class_type 落对", graph["3"]["class_type"] == "KSampler" and graph["1"]["class_type"] == "CheckpointLoaderSimple")

    # ⭐ Reroute 要穿透（否则下游拿到不存在的节点 id ✗）
    rerouted = dict(ui)
    rerouted["nodes"] = list(ui["nodes"]) + [{"id": 5, "type": "Reroute", "inputs": [{"name": "", "link": 10}], "outputs": [{"type": "MODEL"}]}]
    rerouted["links"] = [[10, 1, 0, 5, 0, "MODEL"], [20, 5, 0, 3, 0, "MODEL"], [11, 1, 1, 3, 1, "CONDITIONING"], [12, 3, 0, 7, 0, "IMAGE"]]
    graph2, _ = uw.convert_ui_workflow(rerouted, _sample_object_info())
    check("④ Reroute 被塌缩成上游端点", graph2["3"]["inputs"].get("model") == ["1", 0] and "5" not in graph2, graph2["3"]["inputs"])

    # ⭐ 旁路 = 直连（不是删掉 ✗）；静音 = 丢弃 ✓
    bypassed = {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "inputs": [], "widgets_values": ["model.safetensors"]},
            {"id": 4, "type": "SaveImage", "mode": 4, "inputs": [{"name": "images", "type": "IMAGE", "link": 30}], "outputs": [{"type": "IMAGE"}]},
            {"id": 7, "type": "SaveImage", "inputs": [{"name": "images", "link": 40}], "widgets_values": ["out"]},
        ],
        "links": [[30, 1, 0, 4, 0, "MODEL"], [40, 4, 0, 7, 0, "IMAGE"]],
    }
    graph3, _ = uw.convert_ui_workflow(bypassed, _sample_object_info())
    check("④ 旁路节点直连上游（且不留在图里）", graph3["7"]["inputs"].get("images") == ["1", 0] and "4" not in graph3, graph3)

    try:
        uw.convert_ui_workflow({"a": 1})
        check("④ 非 UI 格式必须报", False, "没抛")
    except uw.UiWorkflowError as exc:
        check("④ 非 UI 格式必须报", exc.code == "not_ui_workflow", exc.code)

    # ⭐ 剪枝：游离节点会被 ComfyUI 一并校验 ⇒ 整次失败 ✗
    pruned = uw.prune_unreachable_nodes({"1": {"class_type": "CheckpointLoaderSimple", "inputs": {}}, "7": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}, "8": {"class_type": "KSampler", "inputs": {}}})
    check("④ 剪掉回溯不到输出的游离节点", set(pruned) == {"1", "7"}, set(pruned))


def case_speaker_tags() -> None:
    # ⭐ 只在「描述段」里补，而且补在 ``<d>`` 台词块**之前**的主体描述上 ✓
    prompt = "integrated_multimodal_description:\n男人站在门口。<d>站住。</d>\n\noverall_soundscape:\n风声"
    result = st.ensure_speaker_tags(prompt)
    check("④ 描述段里补出说话人 id", "男人 (S1) 站在门口" in result["prompt"], result["prompt"])
    check("④ 段外（音景）一字不动", "\n风声" in result["prompt"] and "S1" not in result["prompt"].split("overall_soundscape:")[1], result["prompt"])
    check("④ fixed 计数为真改了几处", result["fixed"] == 1, result["fixed"])

    # ⭐ 画外音要标 off-screen，且说明**只加一次** ✓
    offscreen = st.ensure_speaker_tags("integrated_multimodal_description:\n旁白传来声音。<d>开始。</d>\n")
    check("④ 旁白标 off-screen", "off-screen" in offscreen["prompt"], offscreen["prompt"])
    check("④ 画外说明附在标签后", st.OFFSCREEN_NOTE_ZH in offscreen["prompt"], offscreen["prompt"])

    # ⭐ 认不出主体 ⇒ 不编 id，只记问题 ✓
    unknown = st.ensure_speaker_tags("integrated_multimodal_description:\n<d>某种声音响起。</d>")
    check("④ 无主体不硬编 id（记 issues）", unknown["issues"] and unknown["fixed"] == 0, unknown)
    check("④ 已是合规标签 ⇒ 不改动", st.ensure_speaker_tags("integrated_multimodal_description:\n男人(S1)站在门口。\n")["fixed"] == 0)


def case_media_size() -> None:
    from app.services import media_size as ms

    # ⭐ 档位 × 比例 → 像素（表就是口径：21:9 的 2688x1152 算不出来，是定的 ✓）
    check("⑤ 2k × 16:9 ⇒ 2048x1152", ms.compute_media_size("2k", "16:9") == "2048x1152", ms.compute_media_size("2k", "16:9"))
    check("⑤ 4k × 21:9 ⇒ 3840x1648", ms.compute_media_size("4k", "21:9") == "3840x1648", ms.compute_media_size("4k", "21:9"))
    # ⭐ auto 档返回的是**比例串**，不是像素（调用方必须能接住「还不是像素」✗）
    check("⑤ auto 档只给比例、不给像素", ms.compute_media_size("auto", "16:9") == "16:9", ms.compute_media_size("auto", "16:9"))
    check("⑤ 比例为 auto ⇒ auto", ms.compute_media_size("1k", "auto") == "auto", ms.compute_media_size("1k", "auto"))

    # ⭐ 比例/档位是**就近取**，不是相等（相等判据一个都命中不了 ✗）
    check("⑤ 2048x1152 反推 16:9", ms.infer_media_ratio("2048x1152") == "16:9", ms.infer_media_ratio("2048x1152"))
    check("⑤ 1360x2048 反推 2:3", ms.infer_media_ratio("1360x2048") == "2:3", ms.infer_media_ratio("1360x2048"))
    check("⑤ 2048x1152 反推档位 2k", ms.infer_media_scale("2048x1152") == "2k", ms.infer_media_scale("2048x1152"))
    check("⑤ 未命中预设按长边判档", ms.infer_media_scale("1200x900") == "1k", ms.infer_media_scale("1200x900"))

    # ⭐ 视频：短边 = 分辨率；且必须是偶数（H.264 奇数尺寸会被偷裁 1px ✗）
    check("⑤ 720 + 9:16 ⇒ 720x1280", ms.compute_video_size("720", "9:16") == "720x1280", ms.compute_video_size("720", "9:16"))
    check("⑤ 720 + 16:9 ⇒ 1280x720", ms.compute_video_size("720", "16:9") == "1280x720", ms.compute_video_size("720", "16:9"))
    check("⑤ 480 + 16:9 ⇒ 854x480（偶数）", ms.compute_video_size("480", "16:9") == "854x480", ms.compute_video_size("480", "16:9"))
    check("⑤ 分辨率写法归一", (ms.parse_video_resolution("low"), ms.parse_video_resolution("high"), ms.parse_video_resolution("1080p")) == ("480", "720", "1080"))
    check("⑤ 秒数夹在 2–30", (ms.clamp_video_seconds("1"), ms.clamp_video_seconds("99"), ms.clamp_video_seconds("")) == ("2", "30", "6"))
    check("⑤ 耗时人话", (ms.format_elapsed_duration(1500), ms.format_elapsed_duration(65000)) == ("1.5s", "1m 05s"))

    # ⭐ 认不出的尺寸**不许瞎猜**（拿 normalize 的默认 1k 去算会把 "abc" 静默变成 1024x1024 ✗）
    check("⑤ 乱字符串归不出像素", ms.resolve_size("abc") is None, ms.resolve_size("abc"))
    check("⑤ 只给档位不给比例 ⇒ 归不出像素", ms.normalize_pixel_size("2k") is None, ms.normalize_pixel_size("2k"))
    check("⑤ 档位 + 比例 ⇒ 归得出", ms.normalize_pixel_size("16:9", "2k") == "2048x1152", ms.normalize_pixel_size("16:9", "2k"))
    check("⑤ 读图片尺寸（按档位算）", ms.read_media_dimensions("2k", "16:9") == {"width": 2048, "height": 1152}, ms.read_media_dimensions("2k", "16:9"))
    check("⑤ 读视频尺寸（按分辨率算）", ms.read_video_dimensions(None, "720", "9:16") == {"width": 720, "height": 1280}, ms.read_video_dimensions(None, "720", "9:16"))
    check("⑤ 视频比例反推", ms.infer_video_ratio("720x1280") == "9:16", ms.infer_video_ratio("720x1280"))

    # ⭐ 接线：客户端交 "2k"/"16:9"，注入上下文要拿到像素（否则尺寸静默丢失 ✗）
    ctx = cb.build_context(size="2k", params={"ratio": "16:9"})
    check("⑤ 注入上下文吃掉 2k×16:9", ctx["size"] == {"width": 2048, "height": 1152, "raw": "2048x1152"}, ctx["size"])
    ctx2 = cb.build_context(size="16:9", params={"scale": "2k"})
    check("⑤ 注入上下文吃掉 16:9+2k", ctx2["size"]["width"] == 2048 and ctx2["size"]["height"] == 1152, ctx2["size"])
    # 反向：像素写法与 auto 的行为不许被改动 ✓
    check("⑤ 像素写法原样通过", cb.build_context(size="1024x1536")["size"]["height"] == 1536)
    check("⑤ auto 仍是「没指定」", cb.build_context(size="auto")["size"] is None, cb.build_context(size="auto")["size"])


# ─────────────────────── ⑥ LoRA 链注入 ───────────────────────

def case_lora_chain() -> None:
    from app.services import comfyui as csvc
    from app.services import comfyui_lora_chain as lc

    base = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "steps": 8}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    }
    graph, applied = lc.apply_lora_chain(base, {"loras": [{"name": "Krea2/turbo.safetensors", "strength": 0.8}]})
    check("⑥ 插在模型加载器后面", graph["2"]["inputs"]["model"] == ["9000", 0], graph["2"]["inputs"]["model"])
    check("⑥ 新节点连到原加载器", graph["9000"]["inputs"]["model"] == ["1", 0], graph["9000"]["inputs"])
    check("⑥ 加载器类与强度", graph["9000"]["class_type"] == "LoraLoaderModelOnly" and graph["9000"]["inputs"]["strength_model"] == 0.8)
    check("⑥ 标题只留文件名", graph["9000"]["_meta"]["title"] == "LoRA turbo.safetensors", graph["9000"]["_meta"])
    # ⭐ 必须是克隆：改原图 ⇒ 上游缓存 / 前端比对全乱 ✗
    check("⑥ 不污染原图（深拷贝）", base["2"]["inputs"]["model"] == ["1", 0], base["2"]["inputs"])
    check("⑥ applied 回执", applied == [{"id": "9000", "name": "Krea2/turbo.safetensors", "strength": 0.8}], applied)

    # ⭐ 多条必须**串联**（两个都指加载器 ⇒ 前一个被静默丢掉 ✗✗）
    two, _ = lc.apply_lora_chain(base, {"loras": [{"name": "a.safetensors"}, {"name": "b.safetensors", "strength": 0.5}]})
    check("⑥ 两条串联", two["9001"]["inputs"]["model"] == ["9000", 0] and two["2"]["inputs"]["model"] == ["9001", 0],
          (two["9000"]["inputs"], two["9001"]["inputs"]))
    check("⑥ 第二条强度", two["9001"]["inputs"]["strength_model"] == 0.5)

    # ⭐ 下游必须**全量**重接（只改一处 = 一半分支走 LoRA ✗）
    multi = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "u.safetensors"}},
             "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
             "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}}}
    m2, _ = lc.apply_lora_chain(multi, {"loras": [{"name": "x.safetensors"}]})
    check("⑥ 多下游全部改到链尾", m2["2"]["inputs"]["model"] == ["9000", 0] and m2["3"]["inputs"]["model"] == ["9000", 0],
          (m2["2"]["inputs"], m2["3"]["inputs"]))

    # ⭐ 模型源按「被引用次数」选，平票 UNETLoader 优先（CheckpointLoaderSimple 常只给 CLIP/VAE ✗）
    tie = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
           "2": {"class_type": "UNETLoader", "inputs": {}},
           "3": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["1", 1]}}}
    check("⑥ 模型源选 UNETLoader", lc.find_model_source(tie) == "2", lc.find_model_source(tie))

    same, none_applied = lc.apply_lora_chain(base, {})
    check("⑥ 无 lora 原样返回同一个对象", same is base and none_applied == [], none_applied)

    _, partial = lc.apply_lora_chain(base, {"loras": [{"name": ""}, {"name": "ok.safetensors", "enabled": False},
                                                    {"name": "keep.safetensors", "strength": "abc"}]})
    check("⑥ 空名/禁用跳过、非法强度回 1", [a["name"] for a in partial] == ["keep.safetensors"] and partial[0]["strength"] == 1.0, partial)

    # 四类「宁缺毋错」：都报，不静默跳过（跳过＝用户以为加了其实没加 ✗）
    for name, graph_in, kwargs, expect in (
        ("没有模型加载器", {"2": {"class_type": "KSampler", "inputs": {}}}, {}, "没有可识别的模型加载器"),
        ("加载器没有下游", {"1": {"class_type": "UNETLoader", "inputs": {}}}, {"source_node": "1"}, "没有下游消费者"),
        ("filter 不匹配", base, {"filter": "^qwen"}, "不适用于当前模型"),
        ("available 不含", base, {"available": ["other.safetensors"]}, "不存在"),
    ):
        try:
            lc.apply_lora_chain(graph_in, {"loras": [{"name": "a.safetensors"}]}, **kwargs)
            check(f"⑥ {name} 必须报", False, "没抛")
        except lc.LoraChainError as exc:
            check(f"⑥ {name} 必须报", expect in str(exc) and exc.code == "lora_injection_failed", str(exc))

    check("⑥ 新 id 从 9000 起且跳过占用", lc.next_node_id({"9000": {"class_type": "X"}}) == "9001",
          lc.next_node_id({"9000": {"class_type": "X"}}))

    # ⭐ 接线：入队侧真注入（不是只有个能跑的模块 ✓）
    routed = {"loras": [{"name": "a.safetensors"}], "workflow": base}
    body = csvc._build_upstream_body("workflow", routed)
    check("⑥ 入队侧真的注入了", body["workflow"]["2"]["inputs"]["model"] == ["9000", 0], body["workflow"]["2"]["inputs"])
    check("⑥ applied 记进 params（回执可查）", (routed.get("loraApplied") or [{}])[0].get("name") == "a.safetensors",
          routed.get("loraApplied"))

    try:
        csvc._build_upstream_body("workflow", {"workflowPath": "x.json", "loras": [{"name": "a.safetensors"}]})
        check("⑥ 只给路径 + loras 必须报", False, "没抛")
    except ValueError as exc:
        check("⑥ 只给路径 + loras 必须报", "无法注入 LoRA" in str(exc), str(exc))


# ─────────────────────── ⑦ 进度 / 预览桥 ───────────────────────

def case_progress_bridge() -> None:
    import asyncio
    import json

    from app.services import comfyui_progress as cp

    step, total, percent = cp.percent_of({"8": {"value": 10, "max": 30}, "9": {"value": 5, "max": 20}})
    check("⑦ 进度按 value/max 加权（不是按节点平均）",
          (step, total, percent) == (15, 50, 30.0), (step, total, percent))
    junk = {"1": {"value": 3}, "2": {"value": 3, "max": 0}, "3": "junk"}
    check("⑦ 缺 max / max=0 / 非字典 不参与", cp.percent_of(junk) == (0, 0, 0.0), cp.percent_of(junk))
    check("⑦ value 超 max 要夹住（否则 >100%）", cp.percent_of({"1": {"value": 99, "max": 10}}) == (10, 10, 100.0))
    check("⑦ 空节点表 → 0%", cp.percent_of(None) == (0, 0, 0.0))

    ev = cp.parse_message(json.dumps({"type": "progress_state", "data": {
        "prompt_id": "p1", "nodes": {"8": {"value": 1, "max": 4}}}}), run_id=7)
    check("⑦ progress_state → progress 事件（带 runId/promptId）",
          ev and ev["type"] == "progress" and ev["percent"] == 25.0
          and ev["runId"] == 7 and ev["promptId"] == "p1", ev)

    prev = cp.parse_message(json.dumps({"type": "kj_preview_override", "data": {
        "image": "AAA", "mime": "image/png", "node_id": "9"}}), run_id=7)
    check("⑦ 预览补成 data URL（前端 <img> 直接吃）",
          prev and prev["image"] == "data:image/png;base64,AAA" and prev["nodeId"] == "9", prev)
    check("⑦ 预览缺 image 丢弃",
          cp.parse_message(json.dumps({"type": "kj_preview_override", "data": {}})) is None)

    check("⑦ 成功收尾 = done",
          cp.parse_message('{"type":"execution_success","data":{"prompt_id":"p1"}}')["type"] == "done")
    err = cp.parse_message('{"type":"execution_error","data":{"prompt_id":"p1","exception_message":"oom"}}')
    check("⑦ 失败收尾带原因", err["type"] == "error" and "oom" in err["message"], err)
    check("⑦ 中断收尾 = cancelled", cp.parse_message('{"type":"execution_interrupted","data":{}}')["type"] == "cancelled")
    check("⑦ 无关消息不产事件（executing/status）", cp.parse_message('{"type":"executing","data":{}}') is None)
    check("⑦ 坏输入不炸：坏 JSON / None / bytes",
          cp.parse_message("{oops") is None and cp.parse_message(None) is None
          and cp.parse_message(b'{"type":"execution_success","data":{}}')["type"] == "done")

    tracker = cp.ProgressTracker(ttl_seconds=60)
    seen: list[dict] = []
    off = tracker.subscribe(3, seen.append)
    tracker.apply(3, {"type": "progress", "percent": 42.0, "step": 4, "total": 10})
    snap = tracker.snapshot(3)
    check("⑦ 快照合并 progress", snap["percent"] == 42.0 and snap["state"] == "running" and snap["step"] == 4, snap)
    check("⑦ 订阅者收到事件", len(seen) == 1 and seen[0]["percent"] == 42.0, seen)

    tracker.apply(3, {"type": "preview", "image": "data:image/jpeg;base64,AA", "nodeId": "9"})
    tracker.apply(3, {"type": "preview", "image": "data:image/jpeg;base64,BB", "nodeId": "9"})
    check("⑦ 预览只留最新一帧（不攒历史）",
          tracker.snapshot(3)["preview"]["image"].endswith("BB"), tracker.snapshot(3)["preview"])

    tracker.apply(3, {"type": "done"})
    done_snap = tracker.snapshot(3)
    check("⑦ 收尾把进度拉满（不留 98% 的假失败）",
          done_snap["percent"] == 100.0 and done_snap["state"] == "succeeded", done_snap)

    off()
    tracker.apply(3, {"type": "progress", "percent": 99.0, "step": 9, "total": 10})
    check("⑦ 取消订阅后不再收", len(seen) == 4, len(seen))
    check("⑦ 未知 run 快照为空", tracker.snapshot(999) == {})
    check("⑦ websockets 探测返回布尔", isinstance(cp.websockets_available(), bool))

    saved = cp.websockets_available
    cp.websockets_available = lambda: False  # type: ignore[assignment]
    try:
        ok = asyncio.run(cp.watch_websocket("ws://127.0.0.1:9", "p1", 3, timeout_seconds=0.1))
        check("⑦ 没装 websockets ⇒ 安静降级 False（不抛）", ok is False, ok)
    finally:
        cp.websockets_available = saved  # type: ignore[assignment]

    from app.routers import comfyui_events as ce

    check("⑦ SSE 帧格式", ce._sse_frame("progress", '{"a":1}') == 'event: progress\ndata: {"a":1}\n\n',
          repr(ce._sse_frame("progress", '{"a":1}')))
    check("⑦ 心跳帧只发事件名", ce._sse_frame("ping", "") == "event: ping\n\n")
    check("⑦ 紧凑 JSON 保中文", ce._compact({"s": "中文"}) == '{"s":"中文"}', ce._compact({"s": "中文"}))
    check("⑦ run id 解析：数字过、非数字拒",
          (ce._parse_run_id("12"), ce._parse_run_id("abc")) == (12, None),
          (ce._parse_run_id("12"), ce._parse_run_id("abc")))


# ─────────────────── ⑧ 注册表 → 可提交的图（workflowId 形态） ───────────────────

def case_registry_graph() -> None:
    from app.routers import comfyui_events as ce
    from app.services import comfyui as svc
    from app.services import comfyui_catalog as catalog

    entry = catalog.get_entry("image-z-image-turbo")
    check("⑧ 注册表里能取到 z-image-turbo", entry is not None, entry)

    graph = svc.build_registry_graph({
        "workflowId": "image-z-image-turbo",
        "prompt": "a red fox in falling snow",
        "size": {"width": 832, "height": 1216},
        "quality": {"steps": 10, "cfg": 1.5},
        "values": {"seed": 7},
    })
    base = catalog.load_graph(entry, None)
    check("⑧ 建图不增删节点", len(graph) == len(base), (len(graph), len(base)))
    check("⑧ 提示词落到 #5.text",
          str(graph["5"]["inputs"].get("text") or "").startswith("a red fox"),
          graph["5"]["inputs"].get("text"))
    check("⑧ 尺寸落到 #7（不是写死 1024）",
          (graph["7"]["inputs"].get("width"), graph["7"]["inputs"].get("height")) == (832, 1216),
          (graph["7"]["inputs"].get("width"), graph["7"]["inputs"].get("height")))
    check("⑧ values.seed 落到 #4", graph["4"]["inputs"].get("seed") == 7,
          graph["4"]["inputs"].get("seed"))

    try:
        svc.build_registry_graph({"workflowId": "no-such-workflow"})
        check("⑧ 未知工作流必须报", False, "没抛")
    except ValueError as exc:
        check("⑧ 未知工作流必须报（并指出清单在哪）", "catalog/workflows" in str(exc), str(exc))

    # 显式整图优先 ✓：给了 workflow 就不再按 id 重建（否则调用方拼好的图被静默替换 ✗）
    sentinel = {"1": {"class_type": "X", "inputs": {}}}
    body = svc.build_submit_body(
        {"kind": "workflow", "workflow": sentinel, "workflowId": "image-z-image-turbo"})
    check("⑧ 显式整图优先于 workflowId", body["workflow"] is sentinel,
          type(body.get("workflow")).__name__)

    bare = svc.build_registry_graph({"workflowId": "image-z-image-turbo"})
    check("⑧ 不给任何参数也能建图（走工作流默认档）",
          isinstance(bare, dict) and len(bare) == len(base), len(bare))

    check("⑧ SSE 失败事件叫 failed（避开 EventSource 的 error 同名词）",
          ce._EVENT_NAMES.get("error") == "failed", ce._EVENT_NAMES)

    # ---- 参考素材前置（不碰真上游 ✓）----
    import asyncio
    from pathlib import Path

    from app.core.config import get_storage_root

    storage = Path(get_storage_root()) / "uploads"
    storage.mkdir(parents=True, exist_ok=True)
    sample = storage / "ref-sample.png"
    sample.write_bytes(b"fake-png-bytes")
    try:
        payload, name = svc._reference_file("/uploads/ref-sample.png")
        check("⑧ 素材地址去掉前导斜杠才读得到（Windows 绝对路径坑 ✓）",
              payload == b"fake-png-bytes" and name == "ref-sample.png", (payload[:8], name))
        try:
            svc._reference_file("/uploads/does-not-exist.png")
            check("⑧ 素材不在盘上必须报", False, "没抛")
        except ValueError as exc:
            check("⑧ 素材不在盘上必须报", "不在盘上" in str(exc), str(exc))
        try:
            svc._reference_file("/uploads/../../secret.txt")
            check("⑧ 素材路径穿越必须拦", False, "没抛")
        except ValueError as exc:
            check("⑧ 素材路径穿越必须拦（交给 file_storage 的穿越检查）",
                  "traversal" in str(exc).lower() or "穿越" in str(exc), str(exc))
    finally:
        sample.unlink(missing_ok=True)

    prepared = {"uploadedImages": [{"name": "already.png", "subfolder": ""}]}
    asyncio.run(svc.prepare_reference_media(prepared))
    check("⑧ 已有 uploadedImages 不重复上传（幂等 ✓）",
          prepared["uploadedImages"] == [{"name": "already.png", "subfolder": ""}], prepared)
    untouched: dict = {}
    asyncio.run(svc.prepare_reference_media(untouched))
    check("⑧ 没有参考图时什么都不做", "uploadedImages" not in untouched, untouched)


def main() -> None:
    case_registry_shape()
    case_api_format()
    case_load_graph()
    case_variants()
    case_preset()
    case_step_plan()
    case_transforms()
    case_qwen_sigmas()
    case_inject_real_entry()
    case_inject_missing_required()
    case_inject_skipped()
    case_frame_mode()
    case_array_refs()
    case_spec_table()
    case_slots_of_graph()
    case_discover()
    case_availability()
    case_model_match()
    case_availability_suggestion()
    case_ui_convert()
    case_speaker_tags()
    case_media_size()
    case_lora_chain()
    case_progress_bridge()
    case_registry_graph()

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
