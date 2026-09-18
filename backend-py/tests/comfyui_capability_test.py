"""S7 自检：**ComfyUI 能力门面**（2026-09-17；「功能全都要，但由我们的形式承载」✓）。

ComfyUI 后端（``reference/ComfyUI/server.py``）一共 **26 条公开路由**；本自检盯的是
「**每一条能力在我们这边都有等价物**」✓ —— 起一个**真 HTTP stub ComfyUI**，把门面全部走一遍：

=============================  ==========================================
ComfyUI 原生                   我们的形式（8765）
=============================  ==========================================
``/system_stats`` ``/features``  ``GET  /v1/system``
``/object_info[/{cls}]``         ``GET  /v1/catalog/nodes[/{cls}]``
``/models`` ``/models/{folder}``  ``GET  /v1/catalog/models[/{folder}]``
``/internal/folder_paths``       ``GET  /v1/catalog/folders``
``/internal/files/{type}``       ``GET  /v1/catalog/files/{type}``
``/queue`` ``/prompt``(GET)      ``GET  /v1/queue``
``/queue``(POST clear/delete)    ``POST /v1/queue/clear`` ``DELETE /v1/queue/{id}``
``/history``(GET/POST)           ``GET|DELETE /v1/history``
``/api/jobs*``                   ``GET /v1/jobs`` ``POST /v1/jobs/{id}/cancel``
``/interrupt`` ``/free``         ``POST /v1/interrupt`` ``/v1/free``
``/upload/image`` ``/view``      ``POST /v1/upload/image`` ``GET /v1/view``
``POST /prompt``                 ``POST /v1/workflows/run``（**任意工作流** ✓ UI 或 API 格式 ✓）
（**没有**原生等价物）            ``POST /v1/workflows/validate``（提交**前**校验 ✓ 我们补的 ✓）
=============================  ==========================================

刻意不做的三条（``/`` ``/extensions`` ``/internal/logs*``）在 ``server.py`` 模块头写了理由 ✓。

运行::

    ./.venv/Scripts/python.exe tests/comfyui_capability_test.py
"""
from __future__ import annotations

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
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="comfycap_out_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
#: stub 侧计数（**反套套逻辑**：没真打就没有计数 ✓）
SEEN: dict[str, int] = {}

#: 节点表：两类节点，一类带 widget（STRING）+ 连线输入（IMAGE）✓
NODE_TABLE: dict[str, Any] = {
    "CapLoader": {"input": {"required": {"text": ["STRING", {}], "image": ["IMAGE"]}}},
    "CapSaver": {"input": {"required": {"images": ["IMAGE"], "prefix": ["STRING", {}]}}},
}


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def bump(key: str) -> None:
    SEEN[key] = SEEN.get(key, 0) + 1


# ══════════════════════════════════════════════════════════════════════════
# stub ComfyUI（按官方 server.py 的形状 ✓）
# ══════════════════════════════════════════════════════════════════════════
def build_stub() -> FastAPI:
    stub = FastAPI()

    @stub.get("/system_stats")
    def system_stats() -> dict[str, Any]:  # noqa: D103
        bump("system_stats")
        return {"system": {"comfyui_version": "stub-cap"}, "devices": [{"name": "fake-gpu"}]}

    @stub.get("/features")
    def features() -> dict[str, Any]:  # noqa: D103
        bump("features")
        return {"supports_preview_metadata": True}

    @stub.get("/object_info")
    def object_info_all() -> dict[str, Any]:  # noqa: D103
        bump("object_info_all")
        return NODE_TABLE

    @stub.get("/object_info/{node_class}")
    def object_info(node_class: str) -> dict[str, Any]:  # noqa: D103
        bump("object_info")
        info = NODE_TABLE.get(node_class)
        return {node_class: info} if info else {}

    @stub.get("/models")
    def model_types() -> list[str]:  # noqa: D103
        bump("models")
        return ["checkpoints", "vae", "text_encoders", "diffusion_models"]

    @stub.get("/models/{folder}")
    def models(folder: str) -> list[str]:  # noqa: D103
        bump("models_folder")
        return {"vae": ["vae-a.safetensors", "vae-b.safetensors"]}.get(folder, [])

    @stub.get("/internal/folder_paths")
    def folder_paths() -> dict[str, Any]:  # noqa: D103
        bump("folder_paths")
        return {"vae": ["D:/ComfyUI/models/vae"], "diffusion_models": ["D:/ComfyUI/models/unet"]}

    @stub.get("/internal/files/{directory_type}")
    def internal_files(directory_type: str) -> list[dict[str, Any]]:  # noqa: D103
        bump("internal_files")
        return [{"name": "out_00001_.mp4", "size": 1024, "directory": directory_type}]

    @stub.get("/queue")
    def queue() -> dict[str, Any]:  # noqa: D103
        bump("queue")
        return {"queue_running": [], "queue_pending": [["1", "p1", {}, {}, []]]}

    @stub.post("/queue")
    async def queue_post(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: D103
        bump("queue_post")
        # ⚠️ 记**全部**请求体（只记最后一次会让「清空+删条目」两条断言互相覆盖 ✗ —— 初版就这么错的）
        SEEN.setdefault("queue_post_bodies", []).append(json.dumps(payload))
        return {"ok": True}

    @stub.get("/prompt")
    def prompt_remaining() -> dict[str, Any]:  # noqa: D103
        bump("prompt_get")
        return {"exec_info": {"queue_remaining": 1}}

    @stub.get("/history")
    def history(max_items: int | None = None, offset: int = -1) -> dict[str, Any]:  # noqa: D103
        bump("history")
        return {"p-done": {"status": {"status_str": "success", "completed": True}, "outputs": {}}}

    @stub.post("/history")
    async def history_post(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: D103
        bump("history_post")
        return {"ok": True}

    @stub.get("/api/jobs")
    def jobs() -> dict[str, Any]:  # noqa: D103
        bump("jobs")
        return {"jobs": [{"id": "job-1", "status": "completed"}]}

    @stub.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str) -> dict[str, Any]:  # noqa: D103
        bump("job_cancel")
        return {"cancelled": True, "id": job_id}

    @stub.post("/interrupt")
    async def interrupt(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: D103
        bump("interrupt")
        return {"ok": True}

    @stub.post("/free")
    async def free(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: D103
        bump("free")
        return {"ok": True}

    @stub.post("/upload/image")
    async def upload_image(image: UploadFile = File(), overwrite: str = Form("true"),
                           subfolder: str = Form("")) -> dict[str, Any]:  # noqa: D103
        raw = await image.read()
        bump("upload")
        SEEN["upload_bytes"] = len(raw)
        return {"name": image.filename or "up.png", "subfolder": subfolder, "type": "input"}

    @stub.get("/view")
    def view(filename: str = "", subfolder: str = "", type: str = "output") -> Response:  # noqa: A002
        bump("view")
        return Response(content=b"CAP-" + filename.encode(), media_type="video/mp4")

    @stub.post("/prompt")
    async def post_prompt(payload: dict[str, Any]) -> JSONResponse:  # noqa: D103
        prompt = payload.get("prompt") or {}
        if "nodes" in prompt or not prompt:
            return JSONResponse({"error": {"type": "prompt_no_outputs"}, "node_errors": {}},
                                status_code=400)
        bump("prompt")
        SEEN["last_prompt"] = json.dumps(prompt, ensure_ascii=False)
        return JSONResponse({"prompt_id": "cap-p1", "number": 1, "node_errors": {}})

    @stub.get("/history/{prompt_id}")
    def history_one(prompt_id: str) -> dict[str, Any]:  # noqa: D103
        bump("history_one")
        if SEEN.get("history_one", 0) < 2:
            return {}
        return {prompt_id: {"status": {"status_str": "success", "completed": True},
                            "outputs": {"9": {"videos": [{"filename": "cap_00001_.mp4",
                                                          "subfolder": "video", "type": "output"}]}}}}

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
    os.environ["COMFYUI_URL"] = base_url
    os.environ["H3_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ["H3_PUBLIC_BASE_URL"] = "http://127.0.0.1:8765"
    os.environ["H3_TIMEOUT"] = "60"
    spec = importlib.util.spec_from_file_location("h3_server_cap", H3_DIR / "server.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def wait_terminal(client: TestClient, task_id: str, timeout: float = 40) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = client.get(f"/v1/workflows/task/{task_id}").json()
        if last.get("status") in ("succeeded", "failed"):
            return last
        time.sleep(0.2)
    return last


#: 一张最小 UI 图（两个节点 + 一条连线 ✓）—— 用来验「通用工作流」的 UI→API 路径 ✓
SMALL_UI_GRAPH: dict[str, Any] = {
    "nodes": [
        {"id": 1, "type": "CapLoader", "mode": 0, "title": "装载",
         "inputs": [], "widgets_values": ["原始文本"]},
        {"id": 9, "type": "CapSaver", "mode": 0, "title": "保存",
         "inputs": [{"name": "images", "link": 5}], "widgets_values": ["video/out"]},
    ],
    "links": [[5, 1, 0, 9, 0, "IMAGE"]],
}
#: 同一个工作流的 **API 格式**（验「已经是 API 格式就直接跑」✓）
SMALL_API_PROMPT: dict[str, Any] = {
    "1": {"class_type": "CapLoader", "inputs": {"text": "原始文本"}},
    "9": {"class_type": "CapSaver", "inputs": {"images": ["1", 0], "prefix": "video/out"}},
}


def main() -> int:  # noqa: C901
    base, server, thread = start_server(build_stub())
    wrapper = load_wrapper(base)
    client = TestClient(wrapper.app)
    try:
        # ── ① 系统 / 目录：每一条读能力都要有等价物 ✓ ──
        system = client.get("/v1/system").json()
        check("① /v1/system = /system_stats + /features + 我们自己的状态 ✓",
              system["comfyui"]["reachable"] is True
              and system["comfyui"]["features"].get("supports_preview_metadata") is True
              and system["service"]["install_hint"], list(system))

        nodes = client.get("/v1/catalog/nodes").json()
        check("② /v1/catalog/nodes = /object_info（默认只给类名 ✓ 不给几 MB 全表 ✗）",
              nodes.get("classes") == ["CapLoader", "CapSaver"], nodes)
        one = client.get("/v1/catalog/nodes", params={"class_name": "CapLoader"}).json()
        check("②' ?class_name= 给该类的能力表（字段是**有序**的 required ✓）",
              "CapLoader" in one and "text" in one["CapLoader"]["input"]["required"], one)
        check("②'' 不存在的类 ⇒ 404（不假装有 ✓）",
              client.get("/v1/catalog/nodes", params={"class_name": "NoSuch"}).status_code == 404)

        check("③ /v1/catalog/models = /models（类别）✓",
              client.get("/v1/catalog/models").json().get("folders")
              == ["checkpoints", "vae", "text_encoders", "diffusion_models"],
              client.get("/v1/catalog/models").json())
        check("③' /v1/catalog/models?folder=vae = /models/{folder} ✓",
              client.get("/v1/catalog/models", params={"folder": "vae"}).json().get("files")
              == ["vae-a.safetensors", "vae-b.safetensors"], "")
        check("③'' /v1/catalog/folders = /internal/folder_paths（**磁盘路径** ✓）",
              "D:/ComfyUI/models/vae" in str(client.get("/v1/catalog/folders").json()), "")
        check("③''' /v1/catalog/files/output = /internal/files/{type} ✓",
              client.get("/v1/catalog/files/output").json().get("files"),
              client.get("/v1/catalog/files/output").json())

        # ── ② 控制面：队列 / 历史 / 作业 / 打断 / 释放 ✓ ──
        queue = client.get("/v1/queue").json()
        check("④ /v1/queue = /queue + /prompt(剩余量) ✓",
              queue["queue"]["queue_pending"] and queue["remaining"]["exec_info"]["queue_remaining"] == 1,
              queue)
        # ⚠️ 先调用、再取记录（初版把 `bodies` 取在调用**之前** ⇒ 断言看到空列表 ✗ —— 求值顺序坑）
        cleared_ok = client.post("/v1/queue/clear").json().get("ok")
        deleted_ok = client.delete("/v1/queue/p1").json().get("ok")
        bodies = " ".join(SEEN.get("queue_post_bodies") or [])
        check("④' POST /v1/queue/clear 与 DELETE /v1/queue/{id} 都落到 /queue 的 clear/delete ✓",
              cleared_ok is True and deleted_ok is True and "clear" in bodies and "delete" in bodies,
              (cleared_ok, deleted_ok, SEEN.get("queue_post_bodies")))
        check("⑤ GET/DELETE /v1/history ✓",
              "p-done" in client.get("/v1/history").json().get("history", {})
              and client.delete("/v1/history").json().get("ok") is True, "")
        jobs = client.get("/v1/jobs").json()
        check("⑥ /v1/jobs 支持时给列表、不支持时 supported:false（**不假装** ✓）",
              jobs.get("supported") is True and jobs.get("jobs", {}).get("jobs"),
              jobs)
        check("⑥' POST /v1/jobs/{id}/cancel ✓",
              client.post("/v1/jobs/job-1/cancel").json().get("ok") is True and SEEN.get("job_cancel"))
        check("⑦ POST /v1/interrupt 与 /v1/free ✓",
              client.post("/v1/interrupt", json={}).json().get("ok") is True
              and client.post("/v1/free", json={}).json().get("ok") is True, "")

        # ── ③ 媒体：上传 / 取任意产物 ✓ ──
        uploaded = client.post("/v1/upload/image",
                               files={"image": ("frame.png", b"PNGDATA-12345", "image/png")},
                               data={"subfolder": "h3"}).json()
        check("⑧ POST /v1/upload/image 把 multipart 转交 ComfyUI 并回 {name,subfolder,type} ✓",
              uploaded.get("name") == "frame.png" and SEEN.get("upload_bytes") == 13, (uploaded, SEEN))
        viewed = client.get("/v1/view", params={"filename": "cap_00001_.mp4"})
        check("⑨ GET /v1/view 取任意产物字节 ✓",
              viewed.status_code == 200 and viewed.content.startswith(b"CAP-"), viewed.status_code)

        # ── ④ 通用工作流：**UI 格式**（要转换 ✓）与 **API 格式**（直通 ✓）都能跑 ✓ ──
        created = client.post("/v1/workflows/run", json={
            "workflow": SMALL_UI_GRAPH,
            "params": [{"class_type": "CapLoader", "field": "text", "value": "事件里注入的文本",
                        "title": "装载"}],
        }).json()
        task = wait_terminal(client, created["task_id"])
        check("⑩ UI 格式工作流跑通（内部 UI→API 转换 ✓）且产物有 URL ✓",
              task.get("status") == "succeeded"
              and any(str(f.get("url", "")).startswith("http") for f in task.get("outputs") or []),
              {k: task.get(k) for k in ("status", "outputs", "error_msg")})
        check("⑩' 参数注入落到正确字段（prompt 里 text = 注入值 ✓）",
              "事件里注入的文本" in str(SEEN.get("last_prompt")), str(SEEN.get("last_prompt"))[:160])
        check("⑩'' 产物能被 /files 取到 ✓",
              client.get(str(task["outputs"][0]["url"]).replace("http://127.0.0.1:8765", ""))
              .content.startswith(b"CAP-"), task.get("outputs"))
        check("⑩''' 跑完调了 /free ✓", SEEN.get("free", 0) >= 2, SEEN.get("free"))

        created2 = client.post("/v1/workflows/run", json={"workflow": SMALL_API_PROMPT}).json()
        task2 = wait_terminal(client, created2["task_id"])
        check("⑪ API 格式工作流**直通**（不再转换 ✓）也能跑通 ✓",
              task2.get("status") == "succeeded"
              and json.loads(SEEN["last_prompt"])["1"]["inputs"]["text"] == "原始文本",
              task2.get("status"))

        # ── ⑤ 提交**前**校验（ComfyUI 没有这条能力 ⇒ 我们补的 ✓）──
        good = client.post("/v1/workflows/validate", json={"workflow": SMALL_UI_GRAPH}).json()
        check("⑫ validate: 好工作流 ⇒ ok:true ✓", good.get("ok") is True, good)
        bad_node = client.post("/v1/workflows/validate", json={
            "workflow": {"nodes": [{"id": 1, "type": "NoSuchNode", "mode": 0, "inputs": [],
                                    "widgets_values": []}], "links": []}}).json()
        check("⑬ validate: 缺节点包 ⇒ ok:false 且**点名**类（在提交之前 ✓）",
              bad_node.get("ok") is False and "NoSuchNode" in str(bad_node), bad_node)
        dangling = client.post("/v1/workflows/validate", json={
            "workflow": {"9": {"class_type": "CapSaver",
                               "inputs": {"images": ["404", 0], "prefix": "x"}}}}).json()
        check("⑭ validate: 悬空连线 ⇒ ok:false 且指出是哪个输入（我们补的结构校验 ✓）",
              dangling.get("ok") is False and any("404" in str(p) for p in dangling.get("problems") or []),
              dangling)

        # ── ⑥ 反套套逻辑：stub 侧计数必须真的涨过 ✓ ──
        need = ("system_stats", "features", "object_info_all", "object_info", "models",
                "models_folder", "folder_paths", "internal_files", "queue", "queue_post",
                "prompt_get", "history", "history_post", "jobs", "job_cancel", "interrupt",
                "free", "upload", "view", "prompt")
        check("⑮ 反套套逻辑：stub 的每个端点都真被请求过 ✓（门面不是自说自话 ✓）",
              all(SEEN.get(key) for key in need),
              {key: SEEN.get(key) for key in need if not SEEN.get(key)})
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
