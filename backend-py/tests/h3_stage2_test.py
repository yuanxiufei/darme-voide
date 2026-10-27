"""S7 自检：**H3 薄封装的阶段 2 闭环**（2026-09-16/17 接线；不下载模型、不需要 GPU ✓）。

链路（本自检起一个**真 HTTP stub ComfyUI**，全程真打 ✓）：

    POST /v1/video_generation（后端协议 ✓）
      → 选模板（有首帧用 I2V ✓ 否则 T2V ✓）
      → 注入（提示词 / 时长 / 画幅 / 权重 / 种子 / 产物前缀 ✓）
      → 首尾帧 `POST /upload/image`（**ComfyUI 只认它自己 input 目录里的文件名** ✗)
      → `ui_to_api()`（**UI 图 → API 格式**；模板里被 mute 的尾帧通道按需激活 ✓）
      → 节点预检（缺哪个类直接点名 ✓，**不偷偷下载** ✓）
      → `POST /prompt` → `GET /history/{id}` 轮询 → `GET /view` 取片 → 落盘 → `GET /files/{name}` 回 URL
      → `POST /free` **卸载模型释放显存**（失败路径也要卸 ✓）

覆盖的坑（都是真机上会踩、且不报错的 ✓）：
  ① 模板得选对（T2V vs I2V）② 注入要落在**正确的节点**上（画幅是 COMBO，必须按选项表换 ✓）
  ③ 首尾帧必须先上传成文件名 ✓  ④ 尾帧通道默认 **mute**，要激活 ✓  ⑤ 缺节点/连不上 ComfyUI 时的**报错要指名** ✓
  ⑥ 跑完必须 `/free` ✓（24G 卡否则下一镜 OOM ✗）

运行::

    ./.venv/Scripts/python.exe tests/h3_stage2_test.py
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
H3_DIR = BACKEND_PY / "app" / "local_services" / "h3"
WORKFLOW_DIR = H3_DIR / "workflows"
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="h3stage2_out_"))
os.environ.setdefault("PROXY_TO_NODE", "0")

sys.path.insert(0, str(BACKEND_PY))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

#: 注入点的**图判据** ✓（本套要按它算出来的节点 id 去核图，而不是硬写 id ✗）
from app.local_services.h3 import workflow as wf  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
#: stub 侧记录（**反套套逻辑**：没真打就没有记录 ✓）
SEEN: dict[str, Any] = {"prompts": [], "uploads": [], "free": 0, "view": 0, "history": 0,
                        "models": []}
#: 缺哪个类就返回空（模拟「自定义节点没装」✓）
MISSING_CLASS: dict[str, str | None] = {"name": None}

#: **注入点**的字段表 —— 与真实模板的 node class 对齐 ✓（其余类按图谱的 widget 数量合成 ✓）
SPECS: dict[str, dict[str, Any]] = {
    "MiniMaxH3ImageToVideo": {"input": {"required": {
        "prompt": ["STRING", {}], "width": ["INT", {"default": 1344}],
        "height": ["INT", {"default": 768}], "length": ["INT", {"default": 124}]}}},
    "RandomNoise": {"input": {"required": {
        "noise_seed": ["INT", {"default": 0, "control_after_generate": True}]}}},
    "PrimitiveFloat": {"input": {"required": {"value": ["FLOAT", {"default": 1.0}]}}},
    "ResolutionSelector": {"input": {"required": {
        "ratio": ["COMBO", {"options": ["16:9 (Widescreen)", "9:16 (Portrait)", "1:1 (Square)"]}],
        "megapixels": ["FLOAT", {"default": 0.65}], "multiple": ["INT", {"default": 32}]}}},
    "UNETLoader": {"input": {"required": {
        "unet_name": ["COMBO", {"options": ["MiniMaxH3/fl2va_int8", "MiniMaxH3/ref2va_int8"]}],
        "weight_dtype": ["COMBO", {"options": ["default"]}]}}},
    "VAELoader": {"input": {"required": {
        "vae_name": ["COMBO", {"options": ["MiniMaxH3/video_vae.safetensors",
                                           "MiniMaxH3/audio_vae.safetensors"]}]}}},
    "CLIPLoader": {"input": {"required": {
        "clip_name": ["COMBO", {"options": ["qwen3vl.safetensors"]}],
        "type": ["COMBO", {"options": ["minimax"]}],
        "device": ["COMBO", {"options": ["default"]}]}}},
    "LoadImage": {"input": {"required": {
        # ⚠️ 选项表要写**上传后会出现的文件名** ✓：stub 的 ``/upload/image`` 原样回文件名 ✓，
        #    而真 ComfyUI 的 ``LoadImage.image`` 选项就是「input 目录里的文件名」✓ ⇒ 上传过的名
        #    一定在表里 ✓。表里漏了 ⇒ 预检会报「表外串」✗（那是**假**告警 ✓✗，会掩盖真问题 ✓）。
        "image": ["COMBO", {"options": ["upload_start_frame.png", "first_frame.png", "last_frame.png",
                                        "ref_image_1.png", "ref_image_2.png", "ref_image_3.png"]}],
        "upload": ["COMBO", {"options": ["image"]}]}}},
    # ⭐ 官方 R2V 主节点（``video_minimax_h3_r2v.json`` 的节点 136 ✓）：
    #    ① ``prompt`` 在真模板里是**连线端口**（``inputs[prompt].link = 279`` ✓）⇒ 注册表照真 schema 声明 ✓
    #       （widgets_values 里那个 ``""`` 会被连线覆盖 ✓ —— 见 ``ui_to_api`` 的「连线赢」✓）；
    #    ② **第 5 个 widget 是 ``ref_image_size``** ✓（真值 ``widgets_values_named`` 为
    #       ``{"ref_image_size": "match"}`` ✓）⇒ 漏了它 ⇒ 多出来的非 dict 值会被判「字段错位」拒转 ✗。
    #    出处：核心节点 ``comfy_extras/nodes_minimax_h3.py`` 的 ``MiniMaxH3ReferenceToVideo.define_schema``
    #    （``io.Combo.Input("ref_image_size", options=["match", "max"], default="match")`` ✓）。
    "MiniMaxH3ReferenceToVideo": {"input": {"required": {
        "prompt": ["STRING", {}], "width": ["INT", {}], "height": ["INT", {}], "length": ["INT", {}],
        "ref_image_size": ["COMBO", {"options": ["match", "max"], "default": "match"}]}}},
    "PrimitiveStringMultiline": {"input": {"required": {"value": ["STRING", {}]}}},
    "SaveVideo": {"input": {"required": {
        "filename_prefix": ["STRING", {}], "format": ["COMBO", {"options": ["auto"]}],
        "codec": ["COMBO", {"options": ["auto"]}]}}},
    "CreateVideo": {"input": {"required": {
        "fps": ["FLOAT", {"default": 24}], "quality": ["COMBO", {"options": ["8"]}]}}},
    "KSamplerSelect": {"input": {"required": {
        "sampler_name": ["COMBO", {"options": ["res_multistep"]}]}}},
    "BasicScheduler": {"input": {"required": {
        "scheduler": ["COMBO", {"options": ["simple"]}], "steps": ["INT", {"default": 10}],
        "denoise": ["FLOAT", {"default": 1.0}]}}},
    "ComfyMathExpression": {"input": {"required": {"expression": ["STRING", {}]}}},
    # ⭐ 加速链要用的那一类 ✓：**必须带 ``output``** ✗ —— ``build_chain_nodes`` 靠注册表**数输出槽**
    #    （没给 ⇒ 报「找不到 MODEL 槽」✓，本仓**不猜**槽位 ✗）。这里是 stub 的注册表声明 ✓，
    #    口径与 ComfyUI 核心的 ``LoraLoader`` 一致 ✓（inputs: model/clip/lora_name/strength_* ✓）。
    "LoraLoader": {"input": {"required": {
        "lora_name": ["COMBO", {"options": ["accel.safetensors"]}],
        "strength_model": ["FLOAT", {"default": 1.0}],
        "strength_clip": ["FLOAT", {"default": 1.0}],
        "model": ["MODEL", {}], "clip": ["CLIP", {}]}},
        "output": ["MODEL", "CLIP"]},
}


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def graph_of(name: str) -> dict[str, Any]:
    return json.loads((WORKFLOW_DIR / name).read_text(encoding="utf-8"))


def class_specs(graph: dict[str, Any]) -> dict[str, Any]:
    """按图谱补全 ``object_info``（注入点用 SPECS 的真实字段 ✓，其余按 widget 数量合成 ✓）。"""
    out: dict[str, Any] = {}
    for node in graph.get("nodes") or []:
        cls = node.get("type")
        if not cls or cls in out:
            continue
        if cls in SPECS:
            out[cls] = SPECS[cls]
            continue
        count = len(node.get("widgets_values") or [])
        out[cls] = {"input": {"required": {f"w{i}": ["STRING", {}] for i in range(count)}}}
    return out


# ══════════════════════════════════════════════════════════════════════════
# stub ComfyUI
# ══════════════════════════════════════════════════════════════════════════
def build_stub(catalog: dict[str, Any]) -> FastAPI:
    stub = FastAPI()

    @stub.get("/system_stats")
    def system_stats() -> dict[str, Any]:  # noqa: D103
        return {"system": {"comfyui_version": "stub-stage2"}, "devices": [{"name": "fake-gpu"}]}

    @stub.get("/object_info")
    def object_info_all() -> dict[str, Any]:  # noqa: D103
        # ⭐ 加速链的接缝要**全量注册表**（``GET /object_info`` ✓）—— 真 ComfyUI 两个端点都有 ✓，
        #    stub 少一个 ⇒ 接缝只能判「没查」✗ ⇒ 那一支就**永远验不到** ✓✗（本套第一版正是这样 ✓）。
        return {cls: info for cls, info in catalog.items() if cls != MISSING_CLASS["name"]}

    @stub.get("/object_info/{node_class}")
    def object_info(node_class: str) -> dict[str, Any]:  # noqa: D103
        if MISSING_CLASS["name"] == node_class:
            return {}
        info = catalog.get(node_class)
        return {node_class: info} if info else {}

    @stub.post("/prompt")
    async def post_prompt(payload: dict[str, Any]) -> JSONResponse:  # noqa: D103
        prompt = payload.get("prompt") or {}
        if "nodes" in prompt or not prompt:
            return JSONResponse({"error": {"type": "prompt_no_outputs"}, "node_errors": {}},
                                status_code=400)
        SEEN["prompts"].append(prompt)
        return JSONResponse({"prompt_id": f"p{len(SEEN['prompts'])}", "number": 1, "node_errors": {}})

    @stub.get("/history/{prompt_id}")
    def history(prompt_id: str) -> dict[str, Any]:  # noqa: D103
        SEEN["history"] += 1
        if SEEN["history"] < 2:  # 官方语义：未完成时是空 dict ✓ 逼出轮询 ✓
            return {}
        return {prompt_id: {
            "status": {"status_str": "success", "completed": True},
            "outputs": {"92": {"videos": [{"filename": "stub_00001_.mp4",
                                          "subfolder": "video", "type": "output"}]}},
        }}

    @stub.get("/view")
    def view(filename: str = "", subfolder: str = "", type: str = "output") -> Response:  # noqa: A002
        SEEN["view"] += 1
        return Response(content=b"STUB-MP4-" + filename.encode(), media_type="video/mp4")

    @stub.post("/free")
    async def free(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: D103
        SEEN["free"] += 1
        SEEN["free_body"] = payload
        return {"ok": True}

    @stub.post("/upload/image")
    async def upload_image(image: UploadFile = File(), overwrite: str = Form("true"),
                           subfolder: str = Form("")) -> dict[str, Any]:  # noqa: D103
        raw = await image.read()
        SEEN["uploads"].append({"name": image.filename, "bytes": len(raw), "subfolder": subfolder})
        return {"name": image.filename or "upload.png", "subfolder": subfolder, "type": "input"}

    @stub.get("/models/{folder}")
    def models(folder: str) -> list[str]:  # noqa: D103
        # ⭐ 真 ComfyUI 有这个端点（``GET /models/{folder}`` ✓，是权重的**权威清单** ✓）——
        #    stub 少它 ⇒ ``_pick_checkpoint`` 每次都走「清单取不到 ⇒ 没核」那一支 ✗
        #    ⇒ 「核过清单」这条路径**在本套里永远验不到** ✓✗（正是本仓反复记的那种假覆盖 ✓）。
        SEEN["models"].append(folder)
        if folder == "diffusion_models":
            return list(SPECS["UNETLoader"]["input"]["required"]["unet_name"][1]["options"])
        return []

    return stub


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(app: FastAPI) -> tuple[str, uvicorn.Server, threading.Thread]:
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 25
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/system_stats", timeout=1).status_code == 200:
                return base, server, thread
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise RuntimeError("stub ComfyUI 没能起来")


def load_wrapper(base_url: str) -> Any:
    """按文件加载 8765 薄封装（改 COMFYUI_URL 必须在 import **之前** ✓ —— 它有模块级常量 ✓）。"""
    os.environ["COMFYUI_URL"] = base_url
    os.environ["H3_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ["H3_PUBLIC_BASE_URL"] = "http://127.0.0.1:8765"
    os.environ["H3_WORKFLOW_DIR"] = str(WORKFLOW_DIR)
    os.environ["H3_TIMEOUT"] = "60"
    spec = importlib.util.spec_from_file_location("h3_server_stage2", H3_DIR / "server.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def wait_terminal(client: TestClient, task_id: str, timeout: float = 40) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = client.get(f"/v1/video_generation/task/{task_id}").json()
        if last.get("status") in ("succeeded", "failed"):
            return last
        time.sleep(0.2)
    return last


def find_node(prompt: dict[str, Any], class_type: str, title_contains: str = "") -> dict[str, Any]:
    for node in prompt.values():
        if node.get("class_type") != class_type:
            continue
        if title_contains and title_contains not in str((node.get("_meta") or {}).get("title") or ""):
            continue
        return node
    return {}


def png_data_url(payload: bytes = b"\x89PNG-stub-frame") -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def main() -> int:  # noqa: C901
    t2v = graph_of("MiniMax_H3_Fast_T2V.json")
    i2v = graph_of("MiniMax_H3_Fast_I2V.json")
    off_t2v = graph_of("video_minimax_h3_t2v.json")
    off_i2v = graph_of("video_minimax_h3_i2v.json")
    r2v = graph_of("video_minimax_h3_r2v.json")
    # ⚠️ ``LoraLoader`` **不在模板图里**（是加速链要现挂上去的 ✓）⇒ 得单独并进注册表 ✓
    catalog = {**class_specs(t2v), **class_specs(i2v), **class_specs(off_t2v),
               **class_specs(off_i2v), **class_specs(r2v), "LoraLoader": SPECS["LoraLoader"]}
    base, server, thread = start_server(build_stub(catalog))
    wrapper = load_wrapper(base)
    client = TestClient(wrapper.app)
    try:
        # ── ① T2V：无首帧 ⇒ T2V 模板；注入要落在正确节点上 ✓ ──
        created = client.post("/v1/video_generation", json={
            "model": "hailuo-02", "prompt": "雨夜霓虹街头，主角回头", "aspect_ratio": "9:16",
            "duration": 5, "scene_type": "action",
            "settings": {"checkpoint_map": {"fl2va": "MiniMaxH3/fl2va_int8",
                                            "ref2va": "MiniMaxH3/ref2va_int8"}},
        }).json()
        task = wait_terminal(client, created["task_id"])
        check("① T2V 任务跑通（queued → succeeded ✓）", task.get("status") == "succeeded", task)
        check("①' 走的是 **T2V** 模板（无首帧 ✓）",
              task.get("workflow") == "MiniMax_H3_Fast_T2V.json", task.get("workflow"))
        check("①'' 回给后端的 video_url 指向自己的 /files（产物已落盘 ✓）",
              str(task.get("video_url") or "").startswith("http://127.0.0.1:8765/files/")
              and SEEN["view"] > 0, task.get("video_url"))

        prompt = SEEN["prompts"][-1]
        check("①''' /prompt 收到的是 **API 格式**（class_type/inputs，模板已被转换 ✓）",
              all({"class_type", "inputs"} <= set(n) for n in prompt.values()), list(prompt)[:3])
        main_node = find_node(prompt, "MiniMaxH3ImageToVideo")
        check("② 提示词注入到 MiniMaxH3ImageToVideo.prompt ✓",
              main_node.get("inputs", {}).get("prompt") == "雨夜霓虹街头，主角回头",
              main_node.get("inputs", {}).get("prompt"))
        duration_node = find_node(prompt, "PrimitiveFloat", "DURATION")
        check("③ 时长注入到 «DURATION (seconds)» 节点（帧数由模板表达式算 ✓）",
              duration_node.get("inputs", {}).get("value") == 5, duration_node.get("inputs"))
        check("③' 回给调用方的 expected_frames 与 H3 帧网格一致（5s ⇒ 124 帧 ✓）",
              task.get("expected_frames") == 124, task.get("expected_frames"))
        ratio_node = find_node(prompt, "ResolutionSelector", "RATIO")
        check("④ 画幅按**选项表**换成 9:16 (Portrait)（不是硬塞 '9:16' ✗）",
              "9:16 (Portrait)" in (ratio_node.get("inputs") or {}).values(),
              ratio_node.get("inputs"))
        check("⑤ 权重按 checkpoint_map 路由（scene_type=action ⇒ FL2VA ✓）",
              find_node(prompt, "UNETLoader").get("inputs", {}).get("unet_name") == "MiniMaxH3/fl2va_int8",
              find_node(prompt, "UNETLoader").get("inputs"))
        check("⑥ 种子是整数（control_after_generate=randomize 的语义被实现 ✓）",
              isinstance(find_node(prompt, "RandomNoise").get("inputs", {}).get("noise_seed"), int),
              find_node(prompt, "RandomNoise").get("inputs"))
        check("⑦ 产物前缀带 task_id（便于追溯 ✓）",
              created["task_id"] in str(find_node(prompt, "SaveVideo").get("inputs", {}).get("filename_prefix")),
              find_node(prompt, "SaveVideo").get("inputs"))
        check("⑧ 跑完调了 /free 卸载显存（24G 卡必须 ✓）—— ⚠️ 这条同时是**终态原子性**的判据 ✓："
              "2026-09-24 前是「先写 `succeeded` ✓、再在 `finally` 里 `/free`」✗ ⇒ 这一瞬读到的会是"
              "「成功了但 `freed_vram=null`」✓✗（全量回归里两次**偶发** 19/20 都是它 ✗✗）"
              "⇒ 现在终态与 `freed_vram` 在**同一次 `_update`** 里落地 ✓ ⇒ 读到终态就等于已卸载 ✓",
              SEEN["free"] > 0 and SEEN.get("free_body") == {"unload_models": True, "free_memory": True}
              and task.get("freed_vram") is True, (SEEN.get("free_body"), task.get("freed_vram")))

        # ── ② 产物 URL 真能取到（后端正是这么取片的 ✓）──
        fetched = client.get(str(task["video_url"]).replace("http://127.0.0.1:8765", ""))
        check("⑨ /files/{name} 真能取到产物字节（后端凭 video_url 取片 ✓）",
              fetched.status_code == 200 and fetched.content.startswith(b"STUB-MP4-"), fetched.status_code)

        # ── ③ I2V + 可选尾帧通道：模板选对 + 上传 + 通道激活 ✓ ──
        uploads_before = len(SEEN["uploads"])
        created2 = client.post("/v1/video_generation", json={
            "prompt": "起帧看向镜头，缓慢推近", "duration": 3, "aspect_ratio": "16:9",
            "first_frame_image": png_data_url(),
            "last_frame_image": png_data_url(b"\x89PNG-stub-last-frame"),
        }).json()
        task2 = wait_terminal(client, created2["task_id"])
        check("⑩ 带首帧 ⇒ 走 **I2V** 模板且跑通 ✓",
              task2.get("status") == "succeeded"
              and task2.get("workflow") == "MiniMax_H3_Fast_I2V.json", task2.get("workflow"))
        check("⑪ 首尾帧**都上传**到 ComfyUI（它只认自己 input 目录里的文件名 ✗）",
              len(SEEN["uploads"]) == uploads_before + 2
              and [u["name"] for u in SEEN["uploads"][-2:]] == ["first_frame.png", "last_frame.png"],
              SEEN["uploads"][-2:])
        prompt2 = SEEN["prompts"][-1]
        start_node = find_node(prompt2, "LoadImage", "START FRAME")
        last_node = find_node(prompt2, "LoadImage", "LAST FRAME")
        check("⑫ LoadImage 的 image 被换成**上传后的文件名** ✓",
              start_node.get("inputs", {}).get("image") == "first_frame.png"
              and last_node.get("inputs", {}).get("image") == "last_frame.png",
              (start_node.get("inputs"), last_node.get("inputs")))
        last_link = (find_node(prompt2, "MiniMaxH3ImageToVideo").get("inputs") or {}).get("last_frame")
        src = prompt2.get(str(last_link[0])) if isinstance(last_link, list) and last_link else None
        check("⑬ 尾帧通道被激活并**接回主节点**（接的是该通道末端 = 它的 PreviewImage ✓，镜像首帧接法 ✓）",
              isinstance(last_link, list) and bool(src)
              and src.get("class_type") == "PreviewImage"
              and "LAST FRAME" in str((src.get("_meta") or {}).get("title") or "").upper(),
              (last_link, (src or {}).get("class_type"), (src or {}).get("_meta")))

        # ── ③′ ⭐ H3 **prompt 契约**（提交前 ✓）：文本写法→标签 + **该追加的声明必须追加** ✗✗ ──
        #    那条实测坑：**只写 ``<Audio 1>`` 绑定句不算声明** ✓ ⇒ 照标签跳过会让模型**不复用配音** ✓✗。
        created_c = client.post("/v1/video_generation", json={
            "prompt": "雨夜街头，主角开口 @音1 说出台词",
            "duration": 3, "aspect_ratio": "16:9",
            "reference_audio": ["https://cdn.test/voice.mp3"],
        }).json()
        task_c = wait_terminal(client, created_c["task_id"])
        prompt_c = str(find_node(SEEN["prompts"][-1], "MiniMaxH3ImageToVideo")
                       .get("inputs", {}).get("prompt") or "")
        check("⑰ ⭐ 文本写法 ``@音1`` ⇒ ``<Audio 1>`` 标签（提交前转换 ✓）",
              "<Audio 1>" in prompt_c and "@音1" not in prompt_c, prompt_c[:120])
        check("⑰′ ⭐⭐ 参考音频的**结构化声明被追加** ✗✗（只写标签**不算声明** ✓ ⇒ 不追加等于参考白给 ✓✗）",
              "[reference generation + audio reference]" in prompt_c
              and "retention_analysis" in prompt_c and "reference -" in prompt_c, prompt_c[-200:])
        check("⑰″ 声明是**追加**的 ✓ 不覆盖原提示词 ✓；任务里留下契约报告 ✓",
              prompt_c.startswith("雨夜街头，主角开口 <Audio 1> 说出台词")
              and task_c.get("prompt_contract", {}).get("audioNeeded") is True,
              (prompt_c[:60], task_c.get("prompt_contract")))
        created_n = client.post("/v1/video_generation", json={
            "prompt": "没有参考音频的段", "duration": 3, "aspect_ratio": "16:9",
        }).json()
        wait_terminal(client, created_n["task_id"])
        prompt_n = str(find_node(SEEN["prompts"][-1], "MiniMaxH3ImageToVideo")
                       .get("inputs", {}).get("prompt") or "")
        check("⑰‴ 没有参考音频的段 ⇒ 提示词**一字不改** ✓（不往提示词里塞噪声 ✗）",
              prompt_n == "没有参考音频的段", prompt_n)

        # ── ③″ ⭐ **加速链**（配置驱动 ✓）：接上、且**真的有输入指向它** ✗✗ ──
        created_a = client.post("/v1/video_generation", json={
            "prompt": "加速链用例", "duration": 3, "aspect_ratio": "16:9",
            "settings": {"accel_chain": [{"id": "lora1", "label": "蒸馏 LoRA",
                                          "class_type": "LoraLoader", "kind": "lora",
                                          "inputs": {"lora_name": "accel.safetensors",
                                                     "strength_model": 1.0, "strength_clip": 1.0}}]},
        }).json()
        task_a = wait_terminal(client, created_a["task_id"])
        prompt_a = SEEN["prompts"][-1]
        lora_node = prompt_a.get("accel_lora1")
        check("⑱ ⭐ 加速链被**并进提交的图**里 ✓（节点 id = ``accel_<id>`` ✓）",
              isinstance(lora_node, dict) and lora_node.get("class_type") == "LoraLoader", lora_node)
        check("⑱′ ⭐⭐ 而且**有输入真的指向它** ✗✗（建了节点没人用 = 图看着对、其实没加速 ✓✗）",
              task_a.get("accel_rewired", 0) >= 1
              and any(isinstance(v, list) and v[:1] == ["accel_lora1"]
                      for n in prompt_a.values() if isinstance(n, dict)
                      for v in (n.get("inputs") or {}).values()),
              task_a.get("accel_rewired"))
        check("⑱″ ⭐ LoRA 环的 ``clip`` 流也由链接管 ✓（``kind=lora`` 两条流都要接 ✓）",
              isinstance(lora_node, dict) and isinstance(lora_node.get("inputs", {}).get("clip"), list),
              (lora_node or {}).get("inputs"))

        # ── ③‴ ⭐ 判 error 的环 ⇒ **整条链不上**且**点名** ✗（不静默少一环 ✗✗）──
        created_e = client.post("/v1/video_generation", json={
            "prompt": "坏加速链用例", "duration": 3, "aspect_ratio": "16:9",
            "settings": {"accel_chain": [{"id": "ghost", "class_type": "NotInstalledNode",
                                          "kind": "model_only", "inputs": {}}]},
        }).json()
        task_e = wait_terminal(client, created_e["task_id"])
        check("⑲ ⭐⭐ 环不在注册表 ⇒ **不上链** ✓ 且任务里**点名**（静默少一环 = 图悄悄变样 ✓✗）",
              "accel_ghost" not in SEEN["prompts"][-1]
              and any("ghost" in str(w) for w in (task_e.get("warnings") or [])),
              task_e.get("warnings"))

        # ── ④ 缺节点：报错要**指名**且**不能偷偷下载** ✓ ──
        MISSING_CLASS["name"] = "MiniMaxH3MemoryEfficientSageAttentionPatch"
        prompts_before = len(SEEN["prompts"])
        created3 = client.post("/v1/video_generation",
                               json={"prompt": "缺节点用例", "duration": 3}).json()
        task3 = wait_terminal(client, created3["task_id"])
        check("⑭ 缺节点 ⇒ 失败且**点名**是哪个类 + 给安装指引（不下载 ✗）",
              task3.get("status") == "failed"
              and "MiniMaxH3MemoryEfficientSageAttentionPatch" in str(task3.get("error_msg"))
              and "install-nodes" in str(task3.get("error_msg")), str(task3.get("error_msg"))[:150])
        check("⑮ 缺节点时**根本没有提交**（在 /prompt 之前就拦住了 ✓）",
              len(SEEN["prompts"]) == prompts_before, len(SEEN["prompts"]))
        # ⭐ 失败路径**也要卸** ✓（24G 卡否则下一镜 OOM ✗）—— 此前只有「跑完卸」被断言 ✗，
        #    失败路径这条**没人核** ✓✗（它是 `finally` 的语义 ✓，重构时最容易被顺手改掉 ✓）。
        check("⑭′ ⭐ **失败路径也调了 `/free`** ✓ 且 `freed_vram` 随终态一起落地 ✓（不是事后补 ✗）",
              task3.get("status") == "failed" and task3.get("freed_vram") is True,
              task3.get("freed_vram"))
        MISSING_CLASS["name"] = None

        # ── ⑥ ⭐ 官方模板 + 参考素材（R2VA）：注入点靠**图判据** ✓ 不靠标题 / 不靠节点 id ✗ ──
        #    官方那批与 A 批自制模板「注入点长什么样」页页不同 ✓✗：
        #      · 时长标题三种写法（``DURATION (seconds)`` / ``Float (duration)`` / ``Float (Duration)`` ✗）；
        #      · 官方模板的 ``LoadImage`` **一个标题都没有** ✗✗ ⇒ 按标题找恒找不到 ✓（首帧静默失效 ✓✗）；
        #      · 官方 R2V 的 ``prompt`` 是**连线端口** ✗✗ ⇒ 往主节点硬塞字面量会把连线**覆盖掉** ✓✗。
        #    这一组用例钉的就是「换模板也不瞎」✓，每处都断言**图里那根线真的接上了** ✓ 而不是「没报错」✓。
        r2v_plan = wf.video_plan(graph_of("video_minimax_h3_r2v.json"))
        off_i2v_plan = wf.video_plan(graph_of("video_minimax_h3_i2v.json"))
        uploads_before = len(SEEN["uploads"])
        created5 = client.post("/v1/video_generation", json={
            "prompt": "把 <Picture 1> 的红斗篷穿到 <Picture 2> 的人物身上，站在天台",
            "duration": 6,
            "referenceImageUrls": [png_data_url(b"\x89PNG-ref-1"), png_data_url(b"\x89PNG-ref-2")],
        }).json()
        task5 = wait_terminal(client, created5["task_id"])
        check("⑰ ⭐ 带参考图 ⇒ **自动换官方 R2V 模板**（A 批主节点压根没有参考槽位 ✗）",
              task5.get("status") == "succeeded"
              and task5.get("workflow") == "video_minimax_h3_r2v.json"
              and task5.get("form") == "r2va",
              (task5.get("status"), task5.get("workflow"), task5.get("form")))
        check("⑱ 参考图**真上传**（ComfyUI 只认自己 input 目录里的文件名 ✗）且顺序没乱",
              [u["name"] for u in SEEN["uploads"][uploads_before:]]
              == ["ref_image_1.png", "ref_image_2.png"],
              SEEN["uploads"][uploads_before:])
        prompt5 = SEEN["prompts"][-1]
        main5 = prompt5.get(str(r2v_plan["main_id"])) or {}
        check("⑲ ⭐ R2V 提示词落**上游字符串节点**（主节点那根线**不能被覆盖** ✗✗）",
              isinstance(main5.get("inputs", {}).get("prompt"), list)
              and "<Picture 1>" in str(
                  (prompt5.get(str((r2v_plan["prompt_owner"] or ("", ""))[0])) or {})
                  .get("inputs", {}).get((r2v_plan["prompt_owner"] or ("", ""))[1])),
              (main5.get("inputs", {}).get("prompt"), r2v_plan["prompt_owner"]))
        # ⚠️ ``ref_slot_names`` / ``ref_image_ids`` 是**按槽位对齐**的两张表 ✓（空槽 ⇒ ``None`` ✓，
        #    因为下游 ``_wire_reference_images`` 就是按下标认槽的 ✓）⇒ 这里也要**带槽名**成对地核 ✓：
        #    只比「文件名顺序」的话，「两张图分别落到了哪两个槽」这层没法证 ✓✗（落反了照样过 ✗）。
        landed = [(slot, str((prompt5.get(str(node_id)) or {}).get("inputs", {}).get("image")))
                  for slot, node_id in zip(r2v_plan["ref_slot_names"], r2v_plan["ref_image_ids"])
                  if node_id is not None]
        check("⑳ ⭐ 参考图落到 ``ref_images.ref_image_k`` 指着的那两个 LoadImage 上（图判据找的 ✓，槽序也对 ✓）",
              landed == [("ref_images.ref_image_0", "ref_image_1.png"),
                         ("ref_images.ref_image_1", "ref_image_2.png")], landed)
        check("⑳′ 时长落到官方模板那个标量节点上（它标题是 ``Float (duration)`` ✗ 按标题根本找不到 ✓）",
              str((prompt5.get(str(r2v_plan["duration_id"])) or {}).get("inputs", {}).get("value")) == "6.0",
              (r2v_plan["duration_id"],
               (prompt5.get(str(r2v_plan["duration_id"])) or {}).get("inputs")))

        # ⚠️ 给**少于**槽位数的参考图 ⇒ 多余槽位必须**连同它上游的示例 LoadImage 一起摘掉** ✗✗：
        #    官方模板里那几个 LoadImage 指着模板自带的示例图 ✓（在**我们的** input 目录里不存在 ✓），
        #    而 ``LoadImage.image`` 是 **COMBO** ✓ ⇒ 只要它还在图里，ComfyUI 提交前校验就会因为
        #    「文件名不在可选列表」**当场 400** ✓✗（哪怕这个节点根本不参与出图 ✓）。
        created6 = client.post("/v1/video_generation", json={
            "prompt": "只用 <Picture 1> 这张脸" + "。" * 200,
            "duration": 5, "template": "video_minimax_h3_r2v.json",
            "subject_reference": [png_data_url(b"\x89PNG-only-one")],
        }).json()
        task6 = wait_terminal(client, created6["task_id"])
        prompt6 = SEEN["prompts"][-1]
        spare_id = str(r2v_plan["ref_image_ids"][1])
        check("㉑ ⭐ 参考图只给 1 路 ⇒ 多余槽位**从图上摘掉** ✓ 且它的示例 LoadImage 也**不在图里** ✗",
              task6.get("status") == "succeeded"
              and "ref_images.ref_image_1" not in (prompt6.get(str(r2v_plan["main_id"])) or {}).get("inputs", {})
              and spare_id not in prompt6,
              (task6.get("status"), spare_id in prompt6,
               (prompt6.get(str(r2v_plan["main_id"])) or {}).get("inputs", {}).keys()))
        check("㉒ ⭐ 契约层的 ``<Picture n>`` 按**实际落地路数**算 ✓（``referenceImageUrls`` 这套命名它也认 ✓）",
              task6.get("status") == "succeeded"
              and str((prompt6.get(str(r2v_plan["prompt_owner"][0])) or {})
                      .get("inputs", {}).get(r2v_plan["prompt_owner"][1])).count("<Picture") == 1,
              (prompt6.get(str(r2v_plan["prompt_owner"][0])) or {})
              .get("inputs", {}).get(r2v_plan["prompt_owner"][1], "")[:120])

        created7 = client.post("/v1/video_generation", json={
            "prompt": "官方 I2V 用例" + "。" * 200, "duration": 5,
            # ⚠️ **必须显式指名** ✗：不指名时选模板的规则是「有首帧 ⇒ 本仓那个 Fast I2V」✓
            #    （自动档优先自家加速模板 ✓），而这条用例要钉的是「官方模板的 LoadImage 没标题」✓
            #    ⇒ 得把官方那份 .json 写在 ``template`` 里 ✓。
            "template": "video_minimax_h3_i2v.json",
            "first_frame_image": png_data_url(b"\x89PNG-first"),
        }).json()
        task7 = wait_terminal(client, created7["task_id"])
        prompt7 = SEEN["prompts"][-1]
        check("㉓ ⭐ 显式选**官方 I2V** ⇒ 首帧落到**无标题**的 LoadImage 上（按标题找会找不到 ✓✗）",
              task7.get("status") == "succeeded"
              and task7.get("workflow") == "video_minimax_h3_i2v.json"
              and str((prompt7.get(str(off_i2v_plan["first_frame_id"])) or {})
                      .get("inputs", {}).get("image")) == "first_frame.png",
              (task7.get("workflow"), off_i2v_plan["first_frame_id"],
               (prompt7.get(str(off_i2v_plan["first_frame_id"])) or {}).get("inputs")))

        # ── ⚠️ 首帧 + 参考图**同时给**：现在会自动走 R2V ✓，而 R2V 主节点**没有** ``first_frame`` 端口 ✗
        #    ⇒ 首帧**注定落不下去** ✓（必须**响亮告警** ✗✗ —— 别让用户以为「首帧也生效了」✗）；
        #    并且**连上传都不做** ✓（与参考图那条纪律一致 ✓：落不下去的东西不往人家的 input 目录里塞 ✓）。
        uploads_before11 = len(SEEN["uploads"])
        created11 = client.post("/v1/video_generation", json={
            "prompt": "首帧 + 参考图一起给" + "。" * 200, "duration": 4,
            "first_frame_image": png_data_url(b"\x89PNG-both"),
            "referenceImageUrls": [png_data_url(b"\x89PNG-both-ref")],
        }).json()
        task11 = wait_terminal(client, created11["task_id"])
        picked11 = [u["name"] for u in SEEN["uploads"][uploads_before11:]]
        check("㉗ ⭐ 首帧 + 参考图同时给 ⇒ 走 R2V（参考图优先 ✓）但首帧**落不下去** ⇒ 必须告警 ✗✗",
              task11.get("status") == "succeeded"
              and task11.get("workflow") == "video_minimax_h3_r2v.json"
              and any("START FRAME" in str(w) and "不生效" in str(w) for w in task11.get("warnings") or []),
              (task11.get("status"), task11.get("workflow"), task11.get("warnings")))
        check("㉗′ 落不下去的首帧**不上传** ✗（这一单只传那 1 路参考图 ✓）",
              picked11 == ["ref_image_1.png"], picked11)
        # ⚠️ 官方 I2V 模板**压根没有**尾帧槽位 ✓（连可激活的 mute 通道都没有 ✗）⇒ 尾帧必须告警 ✓；
        #    而这条路径一度会**发两条**说同一件事的告警 ✓✗（「激活不到通道」+「这一帧落不下去」✓）
        #    ⇒ 读的人以为出了两个问题 ✓✗。这里连「只发一条」一起钉住 ✓。
        uploads_before12 = len(SEEN["uploads"])
        created12 = client.post("/v1/video_generation", json={
            "prompt": "官方 I2V + 尾帧" + "。" * 200, "duration": 4,
            "template": "video_minimax_h3_i2v.json",
            "first_frame_image": png_data_url(b"\x89PNG-f"),
            "last_frame_image": png_data_url(b"\x89PNG-l"),
        }).json()
        task12 = wait_terminal(client, created12["task_id"])
        said12 = [str(w) for w in task12.get("warnings") or [] if "LAST FRAME" in str(w)]
        check("㉘ ⭐ 官方 I2V 没有尾帧槽位 ⇒ 尾帧告警**只发一条**（同一件事两条 = 看着像两个问题 ✗）",
              task12.get("status") == "succeeded" and len(said12) == 1 and "不生效" in said12[0],
              (task12.get("status"), task12.get("warnings")))
        check("㉘′ 落不下去的尾帧同样**不上传** ✓（这一单只传首帧 ✓）",
              [u["name"] for u in SEEN["uploads"][uploads_before12:]] == ["first_frame.png"],
              SEEN["uploads"][uploads_before12:])

        prompts_before = len(SEEN["prompts"])
        uploads_before8 = len(SEEN["uploads"])
        created8 = client.post("/v1/video_generation", json={
            "prompt": "硬选不支持参考图的模板" + "。" * 200, "duration": 3,
            "template": "MiniMax_H3_Fast_T2V.json",
            "referenceImageUrls": [png_data_url(b"\x89PNG-nope")],
        }).json()
        task8 = wait_terminal(client, created8["task_id"])
        check("㉔ ⭐ 模板没有参考槽位 ⇒ **响亮告警**（不静默 ✗）且如实说明参考图被忽略",
              task8.get("status") == "succeeded"
              and any("参考图" in str(w) and "忽略" in str(w) for w in task8.get("warnings") or []),
              (task8.get("status"), task8.get("error_msg"), task8.get("warnings")))
        # ⚠️ 判据是「**这一单一个新上传都没有**」✓ —— 别写成「最后一次上传不是 ref_image_1.png」✗：
        #    那种写法在「一个都没传」和「传了别的名」两种情形下都过 ✓✗（后者正是要抓的 bug ✓）。
        check("㉔′ 被忽略时**一个参考图都没上传**（别浪费上传 + 别在 input 目录留垃圾 ✓）",
              len(SEEN["uploads"]) == uploads_before8
              and task8.get("reference_images") == 0,
              (SEEN["uploads"][uploads_before8:], task8.get("reference_images")))

        # ── ⑦ ⭐⭐ 「换了 Ref2VA 权重却白换」那条告警要**分模板**发 ✗✗ ──
        #    它的来历是为 A 批 FL2VA 模板写的 ✓（那些模板没有参考槽位 ⇒ 参考素材确实落不下去 ✓）；
        #    R2V 模板里 ref2va 权重**正是配套的** ✓ ⇒ 同一条告警发出去就是**假消息** ✗
        #    （把「能力接不上」的通知发到「能力已经接上」的路径上 ⇒ 用户会去关掉一个正常的东西 ✓✗）。
        #    ⚠️ 判据要成对 ✓✗：只核「R2V 下没告警」的话，把告警整条删掉也能过 ✗ ⇒ 必须再核
        #    「自家模板下照发」✓，而且两边的 ``unet_name`` 都得真是 ref2va ✓（证明那条分支确实走到了 ✓）。
        weights = {"checkpoint_map": {"fl2va": "MiniMaxH3/fl2va_int8",
                                      "ref2va": "MiniMaxH3/ref2va_int8"}}
        created9 = client.post("/v1/video_generation", json={
            "prompt": "参考权重 + 官方 R2V 模板" + "。" * 200, "duration": 4,
            "model": "MiniMaxH3/ref2va_int8",
            "referenceImageUrls": [png_data_url(b"\x89PNG-w")], "settings": weights,
        }).json()
        task9 = wait_terminal(client, created9["task_id"])
        unet9 = find_node(SEEN["prompts"][-1], "UNETLoader").get("inputs", {}).get("unet_name")
        check("㉖ ⭐ R2V 模板 + ref2va 权重 ⇒ **不发**「参考图不会生效」（权重要**核过清单**才认得 ✓）",
              task9.get("status") == "succeeded" and task9.get("form") == "r2va"
              and unet9 == "MiniMaxH3/ref2va_int8"
              and not any("不会生效" in str(w) for w in task9.get("warnings") or []),
              (unet9, task9.get("warnings")))
        check("㉖″ 顺带核「权重清单真去问了」✓（stub 缺 ``/models`` 时这一整条路径根本验不到 ✗）",
              "diffusion_models" in SEEN["models"], SEEN["models"])

        created10 = client.post("/v1/video_generation", json={
            "prompt": "自家 FL2VA 模板 + 同一个权重表" + "。" * 200, "duration": 4,
            "template": "MiniMax_H3_Fast_T2V.json", "scene_type": "dialogue",
            "settings": weights,
        }).json()
        task10 = wait_terminal(client, created10["task_id"])
        unet10 = find_node(SEEN["prompts"][-1], "UNETLoader").get("inputs", {}).get("unet_name")
        check("㉖′ 同一张权重表 + 自家 FL2VA 模板 ⇒ 那条告警**照发** ✓（不是把告警整条删了 ✗）",
              task10.get("status") == "succeeded" and unet10 == "MiniMaxH3/ref2va_int8"
              and any("不会生效" in str(w) for w in task10.get("warnings") or []),
              (unet10, task10.get("warnings")))

        listed = client.get("/v1/workflows/templates").json()
        by_name = {item["name"]: item for item in listed["templates"]}
        check("㉕ ⭐ 模板清单口子：盘上 9 个模板都在 ✓ 且形态是**图判据**读出来的 ✓",
              len(listed["templates"]) == 9
              and by_name["MiniMax_H3_Fast_T2V.json"]["form"] == "t2va"
              and by_name["video_minimax_h3_r2v.json"]["form"] == "r2va"
              and by_name["video_minimax_h3_i2v.json"]["first_frame_node"] is not None,
              {k: v["form"] for k, v in by_name.items()})
        # ⚠️ 清单要说的是「**各吃哪些入参**」✓✗，只报个形态不够 ✓：调用方拿它来选模板 ✓，
        #    槽位/注入点这几个字段错了 ⇒ 它会挑一个「有参考图但没槽位」的模板 ✓✗。
        check("㉕″ 清单里的**槽位与提示词落点**也是图上读的 ✓（自家那两个：槽位为空 ✓；R2V：三格 ✓）",
              by_name["video_minimax_h3_r2v.json"]["reference_slots"]
              == ["ref_images.ref_image_0", "ref_images.ref_image_1", "ref_images.ref_image_2"]
              and by_name["video_minimax_h3_r2v.json"]["prompt_node"]
              != [r2v_plan["main_id"], "prompt"]
              and by_name["MiniMax_H3_Fast_T2V.json"]["reference_slots"] == [],
              {k: (v["reference_slots"], v["prompt_node"]) for k, v in by_name.items()})
        # ⚠️ 体检（``/v1/system``）里那份「我们会用哪些模板」也得跟上 ✓✗：它原来只报**自制那两个** ✗
        #    ⇒ R2V 接上之后，体检会把「参考素材的通路」整个漏掉 ✓✗（体检缺项比报错更难发现 ✓）。
        system = client.get("/v1/system").json()
        check("㉕‴ 体检报「默认挑哪个模板 + 盘上有哪些」✓（与清单口子**同名同数** ✓）",
              system["workflows"].get("video_minimax_h3_r2v.json") is True
              and system["workflows"].get("MiniMax_H3_Fast_T2V.json") is True
              and sorted(system["templates"]) == sorted(by_name),
              (system["workflows"], system["templates"]))
        check("㉕′ 云端那三个 ``api_*`` 如实标**跑不了本地** ✓（它们走 ComfyUI 云端节点 ✗ 不归本服务管 ✓）",
              all(item["runnable_locally"] is False and item["family"] == "official-cloud-api"
                  for item in listed["templates"] if item["name"].startswith("api_")),
              [item["name"] for item in listed["templates"] if item["name"].startswith("api_")])

        # ── ⑤ ComfyUI 不可达：报错要明确，且**不再是**「阶段 2 未接线」那句 ✗ ──
        server.should_exit = True
        thread.join(timeout=5)
        created4 = client.post("/v1/video_generation",
                               json={"prompt": "不可达用例", "duration": 3}).json()
        task4 = wait_terminal(client, created4["task_id"], timeout=30)
        check("⑯ ComfyUI 不可达 ⇒ 失败且文案是**连接类错误**（不是「not wired yet」✗）",
              task4.get("status") == "failed"
              and "not wired yet" not in str(task4.get("error_msg"))
              and ("不可达" in str(task4.get("error_msg")) or "ComfyUI" in str(task4.get("error_msg"))),
              str(task4.get("error_msg"))[:150])
    finally:
        server.should_exit = True
        try:
            thread.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
