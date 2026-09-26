"""S7 自检：**ComfyUI 执行能力**（客户端 + UI→API 工作流转换；2026-09-16 移植）。

为什么有这份：本项目此前**没有任何代码跟 ComfyUI 说话** ✗（`/prompt` `/history` `/view` `/free`
在产品代码里零命中），而 ComfyUI 的 ``POST /prompt`` **只吃 API 格式**、参考 workflow 全是
**UI 格式** ✗ ⇒ 光有「模型/节点装好」还跑不起来 ✓。这份自检把两件事钉死：

  A. **转换器**（`local_services/h3/workflow.py`）
     A1 拿**真实参考工作流**核对事实（节点类、注入点字段与真实 schema 一致 ✓）；
     A2 用小型合成图逐条验算法：widget 顺序、`forceInput` 不算 widget、
        `control_after_generate` 多吃一槽且种子语义生效、连线解成 ``[id, slot]``、
        非 0 mode 跳过、**没被消费的未知类丢弃 / 被消费的未知类报错**、值多出来要报错 ✓；
     A3 用**真实工作流**整图转换：连线与结构解析（fixture 的 object_info 是按图谱合成的 ✓，
        只为验**结构与连线**；字段语义由 A1/A2 覆盖 ✓）。
  B. **客户端**（`local_services/h3/comfyui_client.py`）—— 起**真 HTTP stub ComfyUI**（uvicorn 线程 ✓，
     不需要 GPU、不需要真装 ComfyUI ✓）走完整闭环：体检 → 节点预检（缺节点**提前**点名 ✓）→
     提交 → 轮询（含「未完成时是空 dict」✓）→ 取产物 → 落盘 → 释放显存 → 上传首帧（multipart ✓）→
     校验失败映射成带 ``node_errors`` 的异常 ✓。

运行::

    ./.venv/Scripts/python.exe tests/h3_comfyui_test.py
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="h3comfy_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402

from app.local_services.h3 import comfyui_client as cc  # noqa: E402
from app.local_services.h3 import workflow as wf  # noqa: E402

#: 真实参考工作流：⚠️ 上游副本在**开发期脚手架** ``reference/`` 下 ✓ —— 用户 2026-09-26 口径 ✓
#: 「等项目完善了之后 reference 这里是要删除的」✗ ⇒ **长期事实以本服务目录里的副本为准** ✓
#: （本常量读的就是它 ✓ 不指脚手架 ✗：删掉 ``reference/`` 后本测试**照样跑** ✓）
REF_WORKFLOW = BACKEND_PY / "app" / "local_services" / "h3" / "workflows" / "MiniMax_H3_Fast_T2V.json"
#: 界面便签：ComfyUI 里**不是执行节点** ✗（本版本核心源码里没有这个类 ✓）
UI_ONLY_CLASSES = {"MarkdownNote", "Note", "NotePlus"}

_RESULTS: list[tuple[str, bool, object]] = []
#: stub 侧计数（**反套套逻辑**用：没发请求 ⇒ 这里是空 ✓）
HITS: dict[str, Any] = {}
COUNTERS: dict[str, int] = {"history_calls": 0, "prompt_calls": 0, "free_calls": 0}


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ══════════════════════════════════════════════════════════════════════════
#  A. 转换器
# ══════════════════════════════════════════════════════════════════════════
def graph_of(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generated_object_info(graph: dict[str, Any], *, skip_classes: set[str]) -> dict[str, Any]:
    """按图谱合成 ``object_info``（**只为验结构与连线** ✓）：每类声明 n 个 STRING widget。

    ``skip_classes`` 模拟「这一类在 ComfyUI 里根本不存在」（如界面便签 ✓）。
    """
    info: dict[str, Any] = {}
    for node in graph.get("nodes") or []:
        cls = node.get("type")
        if not cls or cls in skip_classes or cls in info:
            continue
        count = len(node.get("widgets_values") or [])
        info[cls] = {"input": {"required": {f"w{i}": ["STRING", {}] for i in range(count)}}}
    return info


def case_reference_facts() -> None:
    """A1：真实参考工作流的事实（不靠合成 fixture ✓）。"""
    graph = graph_of(REF_WORKFLOW)
    classes = {n.get("type") for n in graph["nodes"]}
    check("A1① 参考工作流可读：23 个节点 / 20 个类（同类别可以多实例 —— VAELoader×2、PrimitiveFloat×2 ✓）",
          len(graph["nodes"]) == 23 and len(classes) == 20, (len(graph["nodes"]), len(classes)))
    check("A1② 主节点 MiniMaxH3ImageToVideo 在图上（本版本 ComfyUI **核心自带** ✓）",
          "MiniMaxH3ImageToVideo" in classes, sorted(classes)[:3])
    node = next(n for n in graph["nodes"] if n.get("type") == "MiniMaxH3ImageToVideo")
    widgets = node.get("widgets_values") or []
    check("A1③ 注入点与真实 schema 对齐：widgets = [prompt, width, height, length] ✓",
          len(widgets) == 4 and isinstance(widgets[0], str)
          and widgets[1] == 1344 and widgets[2] == 768, widgets)
    check("A1④ length 是**帧数**（24fps 的 124 ≈ 5s）⇒ 时长换算不能按秒直接塞 ✓",
          isinstance(widgets[3], int) and widgets[3] == 124, widgets[3] if len(widgets) > 3 else None)
    check("A1⑤ 界面便签 MarkdownNote 在图上（应当被丢弃的那类 ✓）", "MarkdownNote" in classes, "")


#: A2 用的合成图：一个小 object_info + 一张小图，逐条验算法判据 ✓
SMALL_GRAPH: dict[str, Any] = {
    "nodes": [
        {"id": 1, "type": "Maker", "mode": 0, "title": "源",
         "inputs": [], "widgets_values": ["hi", 12]},
        {"id": 2, "type": "Seed", "mode": 0, "title": "种子",
         "inputs": [], "widgets_values": ["424242424242", "randomize"]},
        {"id": 3, "type": "Legacy", "mode": 0, "title": "老式选项",
         "inputs": [], "widgets_values": ["b"]},
        {"id": 4, "type": "Sink", "mode": 0, "title": "汇聚",
         "inputs": [{"name": "image", "link": 100},
                    {"name": "forced", "link": None},
                    {"name": "tag", "link": None}],
         "widgets_values": ["tail"]},
        # ⚠️ 5 被 bypass（mode=4）⇒ 6 的输入没有活来源 ⇒ 6.image 不该出现 ✓
        {"id": 5, "type": "Muted", "mode": 4, "title": "被旁路",
         "inputs": [], "widgets_values": ["x"]},
        {"id": 6, "type": "Sink", "mode": 0, "title": "吃到旁路上的输入",
         "inputs": [{"name": "image", "link": 102}], "widgets_values": []},
        {"id": 7, "type": "Orphan", "mode": 0, "title": "没人消费的未知类",
         "inputs": [], "widgets_values": []},
        {"id": 8, "type": "Needed", "mode": 0, "title": "被消费的未知类",
         "inputs": [], "widgets_values": []},
        {"id": 9, "type": "Sink", "mode": 0, "title": "值多出来的节点",
         "inputs": [{"name": "image", "link": 103}], "widgets_values": ["x", "y", "z"]},
    ],
    "links": [
        [100, 1, 0, 4, 0, "IMAGE"],
        [102, 5, 0, 6, 0, "IMAGE"],
        [103, 8, 0, 9, 0, "IMAGE"],
    ],
}
SMALL_INFO: dict[str, Any] = {
    "Maker": {"input": {"required": {"text": ["STRING", {}], "steps": ["INT", {"default": 20}]}}},
    "Seed": {"input": {"required": {"seed": ["INT", {"control_after_generate": True}]}}},
    "Legacy": {"input": {"required": {"pick": [["a", "b"], {}]}}},
    "Sink": {"input": {"required": {"image": ["IMAGE"], "forced": ["IMAGE", {"forceInput": True}],
                                    "tag": ["STRING", {}]}}},
    "Muted": {"input": {"required": {"v": ["STRING", {}]}}},
}


def case_converter_rules() -> None:
    """A2：算法逐条（合成图 ✓）。"""
    dropped: list[str] = []
    strict = {k: v for k, v in SMALL_INFO.items() if k != "Needed"}
    # 干净基线：去掉「值多出来」的 9 号节点（连它的连线也一起去掉 ⇒ 8 号不再被消费 ⇒ 走丢弃分支 ✓）
    tolerant = {
        **SMALL_GRAPH,
        "nodes": [n for n in SMALL_GRAPH["nodes"] if n["id"] != 9],
        "links": [link for link in SMALL_GRAPH["links"] if link[0] != 103],
    }
    api = wf.ui_to_api(tolerant, SMALL_INFO, dropped_out=dropped)

    check("A2① widget 值按 object_info 顺序落到正确字段名（text='hi' / steps=12 ✓）",
          api["1"]["inputs"].get("text") == "hi" and api["1"]["inputs"].get("steps") == 12,
          api["1"]["inputs"])
    check("A2② 连线解成 [上游节点id, 输出槽]（image ← ['1', 0] ✓）",
          api["4"]["inputs"].get("image") == ["1", 0], api["4"]["inputs"])
    check("A2③ forceInput=True 不算 widget（forced 未连线 ⇒ 不出现，且不吃掉 tag 的槽位 ✓）",
          "forced" not in api["4"]["inputs"] and api["4"]["inputs"].get("tag") == "tail",
          api["4"]["inputs"])
    check("A2④ control_after_generate 多吃一槽且种子语义生效（'randomize' ⇒ 变成整数且非原值 ✓）",
          isinstance(api["2"]["inputs"].get("seed"), int)
          and api["2"]["inputs"]["seed"] != 424242424242, api["2"]["inputs"])
    check("A2⑤ 老式选项数组（[['a','b'], {}]）也算 widget（pick='b' ✓）",
          api["3"]["inputs"].get("pick") == "b", api["3"]["inputs"])
    check("A2⑥ mode=4（bypass）的节点不转换 ✓", "5" not in api, sorted(api))
    check("A2⑦ 指向被跳过节点的连线不产生输入（6.image 无活来源 ⇒ 不出现 ✓）",
          "6" in api and "image" not in api["6"]["inputs"], api.get("6", {}).get("inputs"))
    check("A2⑧ 未知类但**没人消费**的节点被丢弃（Orphan ✓ 记进 dropped ✓）",
          "7" not in api and "7" in dropped, (sorted(api), dropped))

    # 专用小图：Unknown 类的输出**被活节点消费** ⇒ 必须报错（当作缺的节点包 ✓）
    consumed_missing = {
        "nodes": [
            {"id": 1, "type": "Needed", "mode": 0, "title": "缺的节点包",
             "inputs": [], "widgets_values": []},
            {"id": 2, "type": "Sink", "mode": 0, "title": "消费它",
             "inputs": [{"name": "image", "link": 200}], "widgets_values": ["x"]},
        ],
        "links": [[200, 1, 0, 2, 0, "IMAGE"]],
    }
    try:
        wf.ui_to_api(consumed_missing, strict)
        check("A2⑨ 未知类**有输出被消费** ⇒ 报错（缺的节点包不能静默丢 ✗）", False, "没抛错")
    except wf.WorkflowConversionError as err:
        check("A2⑨ 未知类**有输出被消费** ⇒ 报错（缺的节点包不能静默丢 ✗）",
              "Needed" in str(err), str(err)[:80])

    # 专用小图：Sink 只声明 1 个 widget 却给了 3 个值 ⇒ 字段错位，必须拒绝 ✓
    too_many = {"nodes": [{"id": 1, "type": "Sink", "mode": 0, "title": "值多出来",
                           "inputs": [], "widgets_values": ["a", "b", "c"]}], "links": []}
    try:
        wf.ui_to_api(too_many, SMALL_INFO)
        check("A2⑩ widgets_values 比 object_info 声明多 ⇒ 报错（字段错位必须拒绝 ✗）", False, "没抛错")
    except wf.WorkflowConversionError as err:
        check("A2⑩ widgets_values 比 object_info 声明多 ⇒ 报错（字段错位必须拒绝 ✗）",
              "多出来" in str(err), str(err)[:80])

    # 专用小图：多余的是**纯 dict 伪 widget 标记**（第三方扩展会在 widgets_values 尾部塞这种 ✓）
    pseudo = {"nodes": [{"id": 1, "type": "Sink", "mode": 0, "title": "伪标记",
                         "inputs": [], "widgets_values": ["x", {"type": "HeaderWidget"}]}],
              "links": []}
    try:
        api_pseudo = wf.ui_to_api(pseudo, SMALL_INFO)
        check("A2⑭ 尾部**纯 dict** 的多余值被容忍（伪 widget 标记 ⇒ 不该让整次转换失败 ✗）",
              api_pseudo["1"]["inputs"].get("tag") == "x", api_pseudo["1"]["inputs"])
    except wf.WorkflowConversionError as err:
        check("A2⑭ 尾部**纯 dict** 的多余值被容忍（伪 widget 标记）", False, str(err)[:80])

    changed = wf.set_input(api, "Maker", "text", "新提示词", title_contains="源")
    check("A2⑪ 参数注入：按类名 + 标题片段定位并改写 ✓",
          changed == 1 and api["1"]["inputs"]["text"] == "新提示词", (changed, api["1"]["inputs"]))
    check("A2⑫ 参数注入：字段名不认识时不瞎写（required=True ✓）",
          wf.set_input(api, "Maker", "not_a_field", 1) == 0, api["1"]["inputs"])
    check("A2⑬ node_classes 只统计会执行的节点（bypass 的不算 ✓）",
          "Muted" not in wf.node_classes(SMALL_GRAPH), sorted(wf.node_classes(SMALL_GRAPH)))


def case_reference_graph_conversion() -> None:
    """A3：真实工作流整图转换（结构与连线；object_info 为合成 ✓）。"""
    graph = graph_of(REF_WORKFLOW)
    info = generated_object_info(graph, skip_classes=UI_ONLY_CLASSES)
    dropped: list[str] = []
    # ⚠️ 用 **strict=True**：真实图谱在严格模式下也必须能转（合成 fixture 与图谱同源 ⇒ 数量对齐 ✓）；
    #    若有「值多于声明」会当场抛错 ✓ —— 比 strict=False 强 ✓（A2⑭ 另测伪 widget 的容忍 ✓）。
    api = wf.ui_to_api(graph, info, strict=True, dropped_out=dropped)
    classes = {n.get("type") for n in graph["nodes"]}

    check("A3① 真实工作流 23 节点：**严格模式**下 2 个界面便签被丢、21 个转出 ✓",
          len(api) == 21 and sorted(dropped) == ["142", "143"], (len(api), dropped))
    by_id = {str(n["id"]): n.get("type") for n in graph["nodes"]}
    check("A3①' 被丢的确实是界面便签类（不参与执行的节点 ✓）",
          {by_id.get(d) for d in dropped} <= UI_ONLY_CLASSES, [by_id.get(d) for d in dropped])
    check("A3② 所有 API 节点都有 class_type + inputs ✓",
          all({"class_type", "inputs"} <= set(node) for node in api.values()),
          list(api.values())[0])
    main = next(node for node in api.values() if node["class_type"] == "MiniMaxH3ImageToVideo")
    link_inputs = {k: v for k, v in main["inputs"].items()
                   if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)}
    check("A3③ 主节点的链路输入（clip / vae）都解成 [节点id, 槽位] ✓",
          {"clip", "vae"} <= set(link_inputs), main["inputs"])
    check("A3④ 所有连线目标都真实存在于 api 里（没有悬空连线 ✓）",
          all(str(v[0]) in api for v in link_inputs.values()), link_inputs)
    saver = next(node for node in api.values() if node["class_type"] == "SaveVideo")
    check("A3⑤ SaveVideo 的输入接到上游（整条链没断 ✓）",
          any(isinstance(v, list) for v in saver["inputs"].values()), saver["inputs"])
    check("A3⑥ 转换后类集合 = 图谱类集合 − 被丢的界面便签 ✓",
          classes - {n["class_type"] for n in api.values()} == {"MarkdownNote"},
          classes - {n["class_type"] for n in api.values()})


# ══════════════════════════════════════════════════════════════════════════
#  B. 客户端（真 HTTP stub ComfyUI）
# ══════════════════════════════════════════════════════════════════════════
def build_stub_comfyui() -> FastAPI:
    """按官方 ``server.py`` 的形状造的 stub（**不支持 UI 图** ⇒ 逼出「必须转 API 格式」✓）。"""
    stub = FastAPI()

    @stub.get("/system_stats")
    def system_stats() -> dict[str, Any]:  # noqa: D103
        return {"system": {"comfyui_version": "stub-1.0"}, "devices": [{"name": "fake-gpu"}]}

    @stub.get("/object_info/{node_class}")
    def object_info(node_class: str) -> dict[str, Any]:  # noqa: D103
        if node_class == "MiniMaxH3ImageToVideo":
            return {node_class: {"input": {"required": {
                "prompt": ["STRING", {}], "width": ["INT", {"default": 1344}],
                "height": ["INT", {"default": 768}], "length": ["INT", {"default": 124}],
            }}}}
        if node_class == "SaveVideo":
            return {node_class: {"input": {"required": {"video": ["VIDEO"], "frames": ["INT", {}]}}}}
        return {}

    @stub.post("/prompt")
    async def post_prompt(payload: dict[str, Any]) -> JSONResponse:  # noqa: D103
        COUNTERS["prompt_calls"] += 1
        HITS["prompt"] = COUNTERS["prompt_calls"]
        prompt = payload.get("prompt") or {}
        # 官方行为：UI 图（裸 nodes/links）⇒ 校验不过 ✓
        if "nodes" in prompt or not prompt:
            return JSONResponse({"error": {"type": "prompt_no_outputs"}, "node_errors": {}},
                                status_code=400)
        if any(node.get("class_type") == "MissingNode" for node in prompt.values()):
            return JSONResponse(
                {"error": {"type": "invalid_prompt", "message": "node type MissingNode not found"},
                 "node_errors": {"3": {"class_type": "MissingNode",
                                       "errors": [{"type": "invalid_node"}]}}},
                status_code=400)
        return JSONResponse({"prompt_id": "stub-prompt-1", "number": 1, "node_errors": {}})

    @stub.get("/history/{prompt_id}")
    def history(prompt_id: str) -> dict[str, Any]:  # noqa: D103
        COUNTERS["history_calls"] += 1
        # 官方行为：**未完成时是空 dict**（不是 404 ✓）⇒ 首次返回空，逼出轮询逻辑 ✓
        if COUNTERS["history_calls"] < 2:
            return {}
        return {prompt_id: {
            "status": {"status_str": "success", "completed": True},
            "outputs": {"92": {"videos": [{"filename": "MiniMax_H3_Fast_T2V_00001_.mp4",
                                          "subfolder": "video", "type": "output"}]}},
        }}

    @stub.get("/view")
    def view(filename: str = "", subfolder: str = "", type: str = "output") -> Response:  # noqa: A002
        HITS["view"] = int(HITS.get("view", 0)) + 1
        return Response(content=b"FAKE-MP4-" + filename.encode(), media_type="video/mp4")

    @stub.post("/free")
    async def free(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: D103
        COUNTERS["free_calls"] += 1
        HITS["free"] = COUNTERS["free_calls"]
        HITS["free_body"] = payload
        return {"ok": True}

    @stub.post("/interrupt")
    async def interrupt(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: D103
        HITS["interrupt"] = int(HITS.get("interrupt", 0)) + 1
        return {"ok": True}

    @stub.post("/upload/image")
    async def upload_image(image: UploadFile = File(), overwrite: str = Form("true"),
                           subfolder: str = Form("")) -> dict[str, Any]:  # noqa: D103
        raw = await image.read()
        HITS["upload"] = f"{image.filename}:{len(raw)}"
        HITS["upload_subfolder"] = subfolder
        return {"name": image.filename or "upload.png", "subfolder": subfolder, "type": "input"}

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


def case_client(base: str, workdir: Path) -> None:
    """B：客户端闭环（真 HTTP ✓）。"""
    client = cc.ComfyUIClient(base, timeout=20)
    client.client_id = "test-client"

    stats = client.system_stats()
    check("B① /system_stats 体检可读（版本 + 设备 ✓）",
          stats.get("system", {}).get("comfyui_version") == "stub-1.0", stats)

    missing = client.missing_nodes(["MiniMaxH3ImageToVideo", "SaveVideo", "NoSuchNode"])
    check("B② 节点预检只报**真缺**的类（把「缺哪个节点包」提前点名 ✓）",
          missing == ["NoSuchNode"], missing)

    api_prompt = {"1": {"class_type": "MiniMaxH3ImageToVideo",
                        "inputs": {"prompt": "测试", "width": 768, "height": 432, "length": 124}},
                  "92": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0], "frames": 24}}}
    prompt_id = client.queue_prompt(api_prompt, extra_data={"source": "selftest"})
    check("B③ POST /prompt 拿到 prompt_id ✓", prompt_id == "stub-prompt-1", prompt_id)

    entry = client.wait(prompt_id, timeout=10, poll_interval=0.05)
    check("B④ 轮询能等到完成（stub 首次返回空 dict ⇒ 证明轮询真的在跑 ✓）",
          (entry.get("status") or {}).get("status_str") == "success", entry.get("status"))

    items = cc.extract_output_items(entry)
    check("B⑤ 产物抽取认 videos 键（SaveVideo 走 PreviewVideo ✓）",
          bool(items) and items[0]["filename"].endswith(".mp4") and items[0]["kind"] == "videos",
          items)

    payload = client.fetch_output(items[0])
    check("B⑥ /view 取到产物字节 ✓", payload.startswith(b"FAKE-MP4-"), payload[:16])

    saved = client.save_output(items[0], workdir / "out")
    check("B⑦ 产物落盘且保留原始文件名（便于追溯 ✓）",
          saved.exists() and saved.name == "MiniMax_H3_Fast_T2V_00001_.mp4"
          and saved.stat().st_size > 0, str(saved))

    check("B⑧ POST /free 带 unload_models+free_memory（跑完必须释放显存 ✓）",
          client.free() and HITS.get("free_body") == {"unload_models": True, "free_memory": True},
          HITS.get("free_body"))

    frame = b"\x89PNG-first-frame"
    uploaded = client.upload_image("first.png", frame, subfolder="h3")
    check("B⑨ /upload/image multipart 上传首帧（零依赖手搓 ✓）返回 {name,subfolder,type} ✓",
          uploaded.get("name") == "first.png" and HITS.get("upload") == f"first.png:{len(frame)}"
          and HITS.get("upload_subfolder") == "h3", (uploaded, HITS.get("upload")))

    try:
        client.queue_prompt({"3": {"class_type": "MissingNode", "inputs": {}}})
        check("B⑩ 400 校验失败映射成 WorkflowValidationError（带 node_errors ✓）", False, "没抛错")
    except cc.WorkflowValidationError as err:
        check("B⑩ 400 校验失败映射成 WorkflowValidationError（带 node_errors ✓）",
              "3" in (err.node_errors or {}) and "MissingNode" in str(err), str(err)[:80])

    check("B⑪ 反套套逻辑：stub 真被请求过（prompt / free / view 计数 > 0，且上传留了记录 ✓）",
          all(int(HITS.get(k) or 0) > 0 for k in ("prompt", "free", "view"))
          and str(HITS.get("upload") or "").startswith("first.png:"), HITS)

    before = COUNTERS["prompt_calls"]
    try:
        client.run(api_prompt, required_nodes={"NoSuchNode"}, timeout=5)
        check("B⑫ run() 缺节点时提前抛 NodeMissingError ✓", False, "没抛错")
    except cc.NodeMissingError as err:
        check("B⑫ run() 缺节点时提前抛 NodeMissingError，且**没有提交** ✓",
              err.missing == ["NoSuchNode"] and COUNTERS["prompt_calls"] == before,
              (err.missing, before, COUNTERS["prompt_calls"]))

    check("B⑬ /interrupt 可调用（打断接口在 ✓）", client.interrupt(prompt_id) is True)


def main() -> int:  # noqa: C901
    workdir = Path(tempfile.mkdtemp(prefix="h3comfy_out_"))
    base, server, thread = start_server(build_stub_comfyui())
    try:
        case_reference_facts()
        case_converter_rules()
        case_reference_graph_conversion()
        case_client(base, workdir)
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        shutil.rmtree(workdir, ignore_errors=True)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
