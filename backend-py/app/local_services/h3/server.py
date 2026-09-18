"""MiniMax H3 本地推理薄封装服务（端口 8765）—— **阶段 2 已接线**（2026-09-16/17）
+ **ComfyUI 能力门面**（2026-09-17）✓。

## 两个角色

**① H3 薄封装**（对外协议与 ``backend/src/services/adapters/minimax-video.ts`` 对齐；**后端只见这四条** ✓）：
  POST /v1/video_generation           body={model,prompt,aspect_ratio,duration,...} -> {task_id}
  GET  /v1/video_generation/task/{id} -> {status, video_url, error_msg}
  GET  /files/{name}                  产物下载（后端拿 video_url 来取片 ✓）
  GET  /healthz                       体检（自身 + ComfyUI 可达性 + 工作流是否在盘 ✓）

**② ComfyUI 能力门面**（「ComfyUI 的功能全都要，但由**我们的形式**承载」✓ —— 调用方不必直连 8188 ✓）：

  ================================  ====================================================
  ``POST /v1/workflows/validate``   只校验：转换能不能过 / 缺哪个节点包 / 连线有没有悬空 ✓
  ``POST /v1/workflows/run``        **跑任意工作流**（UI 或 API 格式 ✓）→ 轮询 → 取产物 → URL ✓
  ``GET  /v1/workflows/task/{id}``  通用任务状态（与 H3 同一个任务表 ✓）
  ``GET  /v1/catalog/nodes[/{cls}]``节点目录（默认只给类名 ✓，避免几 MB 全表 ✗）
  ``GET  /v1/catalog/models[/{f}]`` 模型类别 / 某类文件清单 ✓
  ``GET  /v1/catalog/folders``      各类模型的**磁盘路径** ✓
  ``GET  /v1/catalog/files/{type}`` 输入/输出目录清单 ✓
  ``GET  /v1/system``               系统信息 + 能力开关 ✓
  ``GET  /v1/queue`` ``/v1/queue/clear`` ``DELETE /v1/queue/{id}``  队列快照/清空/删条目 ✓
  ``POST /v1/interrupt`` ``/v1/free``  打断 / 卸载显存 ✓
  ``GET|DELETE /v1/history``        历史 / 清空 ✓
  ``GET  /v1/jobs`` ``POST /v1/jobs/{id}/cancel``  作业（老版本不支持时 ``supported:false`` ✓ 不假装 ✓）
  ``POST /v1/upload/image`` ``GET /v1/view``       上传 / 取任意产物 ✓
  ================================  ====================================================

⚠️ **刻意不做**（不是「还没做」✗，是设计决定 ✓）：
* ``GET /`` ``/extensions`` —— 那是 **ComfyUI 前端网页**与它的 js 扩展 ✗（我们自己有前端 ✓）；
* ``/internal/logs*`` 的日志订阅 —— 桌面壳专用 ✗（要日志直接读 ComfyUI 进程输出 ✓）；
* ``POST /prompt`` 的原样透出 —— 通用入口是 ``/v1/workflows/run`` ✓（它替调用方做了
  UI→API 转换、参数注入、节点预检 ✓，比裸透出更难用错 ✓）。

内部链路（**这就是「把 ComfyUI 的能力搬进来」的落点** ✓）：
  body → 选模板（有首帧用 I2V ✓ 否则 T2V ✓）→ 注入（提示词/时长/画幅/种子/权重 ✓）
       → 首尾帧上传 → `workflow.ui_to_api()`（**UI 图 → API 格式** ✓）→ 节点预检（缺哪个包直接点名 ✓）
       → `comfyui_client.queue_prompt()` → `wait()` 轮询 → `/view` 取片 → 落盘 → 回 URL
       → **`/free` 卸载模型释放显存**（24G 卡上这是必须的 ✓）

启动：
  uvicorn server:app --host 0.0.0.0 --port 8765        或  python server.py

环境变量
========  ============================================  ==============================
``COMFYUI_URL``             ComfyUI 地址                    ``http://127.0.0.1:8188``
``H3_TIMEOUT``              单任务总超时（秒）              ``1800``（冷启动/长片要很久 ✓）
``H3_OUTPUT_DIR``           产物落盘目录（`/files` 从这里发） ``<本文件目录>/outputs``
``H3_PUBLIC_BASE_URL``      回给后端的 URL 前缀             ``http://127.0.0.1:8765``
``H3_WORKFLOW_DIR``         工作流模板目录                  ``<本文件目录>/workflows``
``H3_T2V_WORKFLOW``         文生视频模板文件名              ``MiniMax_H3_Fast_T2V.json``
``H3_I2V_WORKFLOW``         图生视频模板文件名              ``MiniMax_H3_Fast_I2V.json``
========  ============================================  ==============================

⚠️ 已知限制（**如实列出**，不假装支持 ✓）：
* 这两个模板是 **FL2VA**（首帧/文生）形态 ⇒ 传了 ``checkpoint_map.ref2va`` 只会换权重名、**参考图/参考音频
  不生效** ✗，此时任务里会带 ``warnings`` 说明 ✓（Ref2VA 需要另一套模板，属后续工作）。
* 采样步数/分辨率档位沿用模板（10 步 / 0.65MP ✓）；``H3_QUALITY`` 可覆盖 megapixels ✓。
* 模板里的 ``length`` 由模板自带的 ``ComfyMathExpression`` 按 **24fps + 17k+5 网格**算 ✓，
  这里只注入「秒」✓（并用同一套数学回一个 ``expected_frames`` 供对账 ✓）。
"""
from __future__ import annotations

import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

# ── 同目录的两个模块：既能作为包导入（uvicorn server:app），也能按文件加载（自检 ✓）──
try:  # pragma: no cover - 取决于加载方式
    from . import comfyui_client as cc
    from . import workflow as wf
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import comfyui_client as cc  # type: ignore[no-redef]
    import workflow as wf  # type: ignore[no-redef]

PORT = int(os.environ.get("H3_PORT") or "8765")
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
TIMEOUT = float(os.environ.get("H3_TIMEOUT") or "1800")
WORKFLOW_DIR = Path(os.environ.get("H3_WORKFLOW_DIR") or (Path(__file__).resolve().parent / "workflows"))
OUTPUT_DIR = Path(os.environ.get("H3_OUTPUT_DIR") or (Path(__file__).resolve().parent / "outputs"))
PUBLIC_BASE_URL = (os.environ.get("H3_PUBLIC_BASE_URL") or f"http://127.0.0.1:{PORT}").rstrip("/")
T2V_WORKFLOW = os.environ.get("H3_T2V_WORKFLOW") or "MiniMax_H3_Fast_T2V.json"
I2V_WORKFLOW = os.environ.get("H3_I2V_WORKFLOW") or "MiniMax_H3_Fast_I2V.json"
#: 覆盖模板里的 megapixels（画质档位）；空 = 用模板值 ✓
QUALITY = os.environ.get("H3_QUALITY") or ""
#: 缺节点时给用户的安装指引（**不在推理路径里偷偷下载** ✗）
INSTALL_HINT = ("python backend-py/app/scripts/model_manager.py install-nodes "
                "（清单在 configs/models.json 的 nodes[]）")

app = FastAPI(title="MiniMax H3 Local Service", version="0.2.0")

# 任务内存存储：task_id -> {status, body, video_url, error_msg, warnings, ...}
_TASKS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════
# 任务状态
# ══════════════════════════════════════════════════════════════════════════
def _update(task: Dict[str, Any], **fields: Any) -> None:
    with _LOCK:
        task.update(fields)


def _warn(task: Dict[str, Any], message: str) -> None:
    with _LOCK:
        task.setdefault("warnings", []).append(message)


# ══════════════════════════════════════════════════════════════════════════
# 输入准备
# ══════════════════════════════════════════════════════════════════════════
def _fetch_media(url: str, timeout: float = 60) -> bytes:
    """把 ``http(s)://`` / ``data:`` / 本地路径 的媒体读成 bytes ✓（首尾帧/参考图都走这里 ✓）。"""
    text = str(url or "").strip()
    if not text:
        raise ValueError("空的媒体地址")
    if text.startswith("data:"):
        import base64

        _, _, payload = text.partition(",")
        return base64.b64decode(payload)
    if text.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(text, timeout=timeout) as response:  # noqa: S310
                return response.read()
        except urllib.error.URLError as err:
            raise ValueError(f"参考媒体取不到：{text}（{err.reason}）") from err
    path = Path(text)
    if path.exists():
        return path.read_bytes()
    raise ValueError(f"参考媒体既不是 URL 也不是存在的本地文件：{text[:120]}")


def _find_load_image(graph: dict[str, Any], keyword: str) -> str | None:
    """找标题含 ``keyword`` 的 ``LoadImage`` 节点 id ✓（如 START / LAST FRAME ✓）。"""
    for node in graph.get("nodes") or []:
        if node.get("type") == "LoadImage" and keyword.upper() in str(node.get("title") or "").upper():
            return str(node.get("id"))
    return None


def _widget_field(api_node: dict[str, Any], fallback: str) -> str:
    """猜某个 API 节点的目标 widget 字段名：优先用调用方给的稳定名，其次取第一个字符串入参 ✓。"""
    inputs = api_node.get("inputs") or {}
    if fallback in inputs:
        return fallback
    for key, value in inputs.items():
        if isinstance(value, str):
            return key
    return fallback


# ══════════════════════════════════════════════════════════════════════════
# 组装：body → API 格式工作流
# ══════════════════════════════════════════════════════════════════════════
def _build_api_prompt(client: "cc.ComfyUIClient", task: Dict[str, Any]
                      ) -> tuple[dict[str, Any], list[str]]:
    """把请求组装成可直接 ``POST /prompt`` 的 **API 格式**工作流；返回 (prompt, dropped 节点 id)。"""
    body: dict[str, Any] = task.get("body") or {}
    prompt_text = str(body.get("prompt") or "").strip()
    if not prompt_text:
        raise ValueError("prompt 为空（后端必须给提示词 ✓）")

    first_frame = str(body.get("first_frame_image") or "").strip()
    last_frame = str(body.get("last_frame_image") or "").strip()
    template_name = I2V_WORKFLOW if first_frame else T2V_WORKFLOW
    template_path = WORKFLOW_DIR / template_name
    if not template_path.exists():
        raise ValueError(f"工作流模板不在盘上：{template_path}（检查 H3_WORKFLOW_DIR ✓）")
    import json

    graph = json.loads(template_path.read_text(encoding="utf-8"))

    # 需要尾帧 ⇒ 把模板里**默认关掉**的可选尾帧通道打开（参考 I2V 模板的 [DISABLED] 节点 ✓）
    activated: list[str] = []
    if last_frame:
        activated = wf.activate_nodes(graph)
        if not activated:
            _warn(task, "模板里没有可激活的尾帧通道 ⇒ 尾帧将被忽略 ✗")

    # 首帧/尾帧上传（ComfyUI 只认它自己 input 目录里的文件名 ✗ ⇒ 必须先 upload ✓）
    uploads: list[tuple[str, str, bytes]] = []
    if first_frame:
        uploads.append(("START FRAME", "first_frame.png", _fetch_media(first_frame)))
    if last_frame:
        uploads.append(("LAST FRAME", "last_frame.png", _fetch_media(last_frame)))

    uploaded: dict[str, str] = {}
    for keyword, filename, payload in uploads:
        result = client.upload_image(filename, payload)
        name = result.get("name") or filename
        subfolder = result.get("subfolder") or ""
        uploaded[keyword] = f"{subfolder}/{name}" if subfolder else str(name)

    classes = wf.node_classes(graph)
    object_info: dict[str, Any] = {}
    missing: list[str] = []
    for cls in sorted(classes):
        info = client.object_info(cls)
        if info:
            object_info.update(info)
        else:
            missing.append(cls)
    if missing:
        raise cc.NodeMissingError(missing)

    dropped: list[str] = []
    api = wf.ui_to_api(graph, object_info, dropped_out=dropped)

    # ── 注入：提示词 ──
    changed = wf.set_input(api, "MiniMaxH3ImageToVideo", "prompt", prompt_text)
    if not changed:
        _warn(task, "没找到 MiniMaxH3ImageToVideo.prompt 注入点 ✗（模板结构变了？）")

    # ── 注入：时长（秒 → 模板自带的 DURATION 节点；帧数由模板的表达式按 24fps/17k+5 算 ✓）──
    duration = float(body.get("duration") or 5)
    expected_frames = wf.frames_for_seconds(duration)
    if not wf.set_input(api, "PrimitiveFloat", "value", duration, title_contains="DURATION"):
        _warn(task, f"没找到 DURATION 节点 ⇒ 时长沿用模板默认值 ✗（请求的是 {duration}s）")

    # ── 注入：画幅（COMBO 必须按**选项表**换，否则会被校验拒 ✓）──
    ratio = str(body.get("aspect_ratio") or "").strip()
    if ratio:
        picked = wf.set_combo(api, "ResolutionSelector", ratio, object_info)
        if not picked:
            _warn(task, f"画幅 {ratio!r} 没匹配到 ResolutionSelector 的选项 ⇒ 用模板默认值 ✗")

    # ── 注入：画质档位（可选）──
    if QUALITY:
        wf.set_input(api, "PrimitiveFloat", "value", float(QUALITY), title_contains="QUALITY")

    # ── 注入：权重（FL2VA / Ref2VA 路由；模板是 FL2VA 形态 ⇒ ref2va 只换名字并告警 ✓）──
    settings = body.get("settings") or {}
    checkpoint_map = settings.get("checkpoint_map") if isinstance(settings, dict) else None
    checkpoint = None
    if isinstance(checkpoint_map, dict) and checkpoint_map:
        scene = str(body.get("scene_type") or settings.get("scene_type") or "").lower()
        checkpoint = (checkpoint_map.get("ref2va") if scene and scene != "action" and scene != "silent"
                      else checkpoint_map.get("fl2va")) or checkpoint_map.get("fl2va")
        if checkpoint and checkpoint_map.get("ref2va") and checkpoint == checkpoint_map.get("ref2va"):
            _warn(task, "模板是 FL2VA 形态：已换 Ref2VA 权重名，但**参考图/参考音频不会生效** ✗")
    if checkpoint:
        wf.set_input(api, "UNETLoader", "unet_name", checkpoint)
    if body.get("reference_audio_urls") or body.get("referenceImageUrls") or body.get("subject_reference"):
        _warn(task, "当前模板不支持参考图/参考音频（Ref2VA）⇒ 这些入参被忽略 ✗")

    # ── 注入：种子（每次不同 ✓）与产物体名（便于追溯 ✓）──
    wf.set_seed(api, "RandomNoise")
    wf.set_input(api, "SaveVideo", "filename_prefix",
                 f"h3/{task.get('task_id') or 'task'}", required=False)

    # ── 首尾帧落到 LoadImage 上（API 里 LoadImage 的入参就是「文件名」✓）──
    first_node_id = _find_load_image(graph, "START FRAME")
    for keyword, node_id in (("START FRAME", first_node_id),
                             ("LAST FRAME", _find_load_image(graph, "LAST FRAME"))):
        if keyword not in uploaded:
            continue
        if not node_id or node_id not in api:
            _warn(task, f"{keyword} 的上传成功但模板里没有对应 LoadImage 节点 ⇒ 该帧不生效 ✗")
            continue
        field = _widget_field(api[node_id], "image")
        api[node_id]["inputs"][field] = uploaded[keyword]

    # ── 尾帧通道**接回主节点**：首帧是模板接好的（first_frame ← 那条链的末端 ✓），
    #    尾帧通道刚被激活时是**没接线**的 ✗ ⇒ 这里镜像首帧的接法，把通道**末端**接上 ✓。
    #    「末端」用图判据找（没有任何同通道节点消费它 ✓），不写死节点 id ✗（模板换了也不怕 ✓）。
    if last_frame and activated:
        referenced = {value[0] for node in api.values()
                      for value in (node.get("inputs") or {}).values()
                      if isinstance(value, list) and len(value) == 2}
        tails = [node_id for node_id in activated if node_id in api and node_id not in referenced]
        if len(tails) == 1:
            wf.set_input(api, "MiniMaxH3ImageToVideo", "last_frame", [tails[0], 0], required=False)
            _update(task, last_frame_tail=tails[0])
        else:
            _warn(task, f"尾帧通道末端不唯一（{tails}）⇒ 没自动接线，尾帧可能不生效 ✗")

    _update(task, workflow=template_name, dropped_nodes=dropped,
            expected_frames=expected_frames, expected_seconds=wf.seconds_for_frames(expected_frames),
            uploaded=uploaded)
    return api, dropped


# ══════════════════════════════════════════════════════════════════════════
# 执行
# ══════════════════════════════════════════════════════════════════════════
def _run_h3(task: Dict[str, Any]) -> None:
    """提交到 ComfyUI 并把产物取回本地（**失败也要 `/free`** ✓）。"""
    client = cc.ComfyUIClient(COMFYUI_URL, timeout=TIMEOUT)
    try:
        _update(task, status="processing")
        api_prompt, _dropped = _build_api_prompt(client, task)
        prompt_id = client.queue_prompt(api_prompt,
                                        extra_data={"source": "voide-darme/h3", "task": task["task_id"]})
        _update(task, comfy_prompt_id=prompt_id)
        entry = client.wait(prompt_id, timeout=TIMEOUT)
        status = (entry.get("status") or {})
        if status.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI 执行失败：{(status.get('messages') or [])!r}"[:600])

        items = cc.extract_output_items(entry)
        videos = [item for item in items if item.get("kind") == "videos"] or items
        if not videos:
            raise RuntimeError("ComfyUI 执行完成但**没有任何产物**（SaveVideo 没接上？检查模板 ✓）")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        saved = client.save_output(videos[0], OUTPUT_DIR,
                                   filename=f"{task['task_id']}{Path(str(videos[0]['filename'])).suffix or '.mp4'}")
        _update(task, status="succeeded",
                video_url=f"{PUBLIC_BASE_URL}/files/{saved.name}",
                outputs=items, saved_path=str(saved), error_msg=None)
    except cc.NodeMissingError as err:
        _update(task, status="failed",
                error_msg=f"{err}｜装节点：{INSTALL_HINT}")
    except Exception as err:  # noqa: BLE001 —— 任何失败都要变成任务失败，不能挂在后台 ✗
        _update(task, status="failed", error_msg=f"{type(err).__name__}: {err}"[:600])
    finally:
        # ⚠️ 24G 卡上跑完必须卸载 ✓（失败路径也要卸，否则下一镜 OOM ✗）
        freed = client.free(unload_models=True, free_memory=True)
        _update(task, freed_vram=freed)


# ══════════════════════════════════════════════════════════════════════════
# 路由
# ══════════════════════════════════════════════════════════════════════════
@app.post("/v1/video_generation")
def create_video(body: Dict[str, Any]) -> JSONResponse:
    task_id = uuid.uuid4().hex
    task: Dict[str, Any] = {
        "task_id": task_id,
        "status": "queued",
        "body": body,
        "video_url": None,
        "error_msg": None,
        "warnings": [],
    }
    with _LOCK:
        _TASKS[task_id] = task
    threading.Thread(target=_run_h3, args=(task,), daemon=True).start()
    return JSONResponse({"task_id": task_id, "status": "queued"})


@app.get("/v1/video_generation/task/{task_id}")
def poll_task(task_id: str) -> JSONResponse:
    with _LOCK:
        task = _TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return JSONResponse({
        "status": task["status"],
        "video_url": task["video_url"],
        "error_msg": task["error_msg"],
        # 以下为**诊断字段**：后端协议只认上面三个 ✓，多给不破坏兼容 ✓
        "warnings": task.get("warnings") or [],
        "comfy_prompt_id": task.get("comfy_prompt_id"),
        "workflow": task.get("workflow"),
        "expected_frames": task.get("expected_frames"),
        "expected_seconds": task.get("expected_seconds"),
        "freed_vram": task.get("freed_vram"),
    })


@app.get("/files/{name}")
def get_file(name: str) -> FileResponse:
    """产物下载：只允许 OUTPUT_DIR 下的**文件名本身**（挡掉路径穿越 ✓）。"""
    safe = Path(name).name
    path = OUTPUT_DIR / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)


def _client() -> "cc.ComfyUIClient":
    return cc.ComfyUIClient(COMFYUI_URL, timeout=TIMEOUT)


# ══════════════════════════════════════════════════════════════════════════
# ① 通用工作流：**任意** ComfyUI 工作流都能在这跑（H3 只是第一个消费者 ✓）
#    —— 这就是「ComfyUI 的功能全都要、但由我们的形式承载」的落点 ✓
# ══════════════════════════════════════════════════════════════════════════
def _prepare_generic(client: "cc.ComfyUIClient", task: Dict[str, Any]
                     ) -> tuple[dict[str, Any], list[str]]:
    """把 ``{workflow, params}`` 准备成 API 提示词：转换（UI→API ✓）+ 参数注入 + 节点预检 ✓。

    * ``workflow``：**UI 格式**（原样给参考工作流即可 ✓）或**已经是 API 格式**（会被识别 ✓）。
    * ``params``：**显式注入清单** ``[{class_type, field, value, title?}]`` ✓ —— 刻意不做
      「猜哪个字段是提示词」的魔法 ✗（猜错一次就是白跑一轮 ✗）。
    """
    body = task.get("body") or {}
    workflow = body.get("workflow")
    if workflow is None:
        raw_path = body.get("workflow_path")
        if not raw_path:
            raise ValueError("缺少 workflow（UI/API JSON）或 workflow_path ✓")
        import json as _json

        workflow = _json.loads(Path(str(raw_path)).read_text(encoding="utf-8"))

    # 已经是 API 格式？（每个值都有 class_type/inputs ⇒ 直接当 API ✓）
    is_api = isinstance(workflow, dict) and workflow and all(
        isinstance(v, dict) and {"class_type", "inputs"} <= set(v) for v in workflow.values())

    if is_api:
        api = {k: {"class_type": v["class_type"],
                   "inputs": dict(v.get("inputs") or {}),
                   "_meta": v.get("_meta") or {}} for k, v in workflow.items()}
        injected = _inject_params(task, api)
        classes = {v["class_type"] for v in api.values()}
        missing = client.missing_nodes(sorted(classes))
        if missing:
            raise cc.NodeMissingError(missing)
        return api, injected

    graph = workflow
    classes = wf.node_classes(graph) if isinstance(graph, dict) else set()
    object_info: dict[str, Any] = {}
    missing: list[str] = []
    for cls in sorted(classes):
        info = client.object_info(cls)
        if info:
            object_info.update(info)
        else:
            missing.append(cls)
    if missing:
        raise cc.NodeMissingError(missing)
    dropped: list[str] = []
    api = wf.ui_to_api(graph, object_info, dropped_out=dropped)
    _update(task, dropped_nodes=dropped, nodes=len(api))
    injected = _inject_params(task, api)
    return api, injected


def _inject_params(task: Dict[str, Any], api: dict[str, Any]) -> list[str]:
    """按显式清单注入参数，返回**注入失败项**的说明（不静默 ✗）。"""
    problems: list[str] = []
    for spec in (task.get("body") or {}).get("params") or []:
        if not isinstance(spec, dict):
            continue
        cls = spec.get("class_type")
        field = spec.get("field")
        if not cls or not field:
            problems.append(f"参数项缺 class_type/field：{spec!r}")
            continue
        changed = wf.set_input(api, str(cls), str(field), spec.get("value"),
                               title_contains=spec.get("title"), required=False)
        if not changed:
            problems.append(f"没找到注入点 {cls}.{field}（title={spec.get('title')!r}）")
    if problems:
        _update(task, injection_problems=problems)
    return problems


@app.post("/v1/workflows/validate")
def validate_workflow(body: Dict[str, Any]) -> JSONResponse:
    """**只校验不提交**：转换能不能过 / 节点包齐不齐 / 连线有没有悬空 ✓。

    （ComfyUI 自己只在 ``POST /prompt`` 那一刻才校验 ✗ ⇒ 这条是我们**补出来的**能力 ✓。）
    """
    task = {"task_id": "validate", "body": body}
    client = _client()
    try:
        api, problems = _prepare_generic(client, task)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"{type(err).__name__}: {err}"[:600],
                             "missing_nodes": getattr(err, "missing", None)},
                            status_code=200)
    known = set((client.object_info_all() or {})) if body.get("check_known_classes") else None
    structural = client.validate(api, known_classes=known)
    return JSONResponse({
        "ok": not (problems or structural),
        "nodes": len(api),
        "injection_problems": problems,
        "problems": structural,
        "missing_nodes": [],
    })


@app.post("/v1/workflows/run")
def run_workflow(body: Dict[str, Any]) -> JSONResponse:
    """**跑任意 ComfyUI 工作流**（UI 或 API 格式 ✓）：转换/注入/预检 → 提交 → 轮询 → 取产物 → 落盘 → URL ✓。"""
    task: Dict[str, Any] = {
        "task_id": uuid.uuid4().hex, "status": "queued", "body": body,
        "video_url": None, "error_msg": None, "warnings": [], "kind": "workflow",
    }
    with _LOCK:
        _TASKS[task["task_id"]] = task
    threading.Thread(target=_run_generic, args=(task,), daemon=True).start()
    return JSONResponse({"task_id": task["task_id"], "status": "queued"})


def _run_generic(task: Dict[str, Any]) -> None:
    """通用工作流的执行体（与 H3 同形：提交 → 轮询 → 取产物 → 落盘 → 释放显存 ✓）。"""
    client = _client()
    try:
        _update(task, status="processing")
        api, _problems = _prepare_generic(client, task)
        prompt_id = client.queue_prompt(api, extra_data={"source": "voide-darme/workflow",
                                                         "task": task["task_id"]})
        _update(task, comfy_prompt_id=prompt_id)
        entry = client.wait(prompt_id, timeout=TIMEOUT)
        status = (entry.get("status") or {})
        if status.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI 执行失败：{(status.get('messages') or [])!r}"[:600])
        items = cc.extract_output_items(entry)
        files = []
        if items:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            for index, item in enumerate(items):
                suffix = Path(str(item.get("filename"))).suffix or ".bin"
                saved = client.save_output(item, OUTPUT_DIR,
                                           filename=f"{task['task_id']}_{index}{suffix}")
                files.append({"kind": item.get("kind"), "source": item.get("filename"),
                              "url": f"{PUBLIC_BASE_URL}/files/{saved.name}"})
        _update(task, status="succeeded", outputs=files,
                video_url=next((f["url"] for f in files if f["kind"] == "videos"), None),
                error_msg=None)
    except cc.NodeMissingError as err:
        _update(task, status="failed", error_msg=f"{err}｜装节点：{INSTALL_HINT}")
    except Exception as err:  # noqa: BLE001
        _update(task, status="failed", error_msg=f"{type(err).__name__}: {err}"[:600])
    finally:
        _update(task, freed_vram=client.free(unload_models=True, free_memory=True))


@app.get("/v1/workflows/task/{task_id}")
def workflow_task(task_id: str) -> JSONResponse:
    with _LOCK:
        task = _TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return JSONResponse({k: v for k, v in task.items() if k != "body"})


# ══════════════════════════════════════════════════════════════════════════
# ② ComfyUI 的其余能力 —— 全部用**我们的 REST 形式**透出（目录/系统/队列/历史/上传/取件 ✓）
#    命名与分组是我们自己的（``/v1/catalog`` ``/v1/system`` …）= 「表现形式不同」✓
# ══════════════════════════════════════════════════════════════════════════
@app.get("/v1/system")
def system() -> Dict[str, Any]:
    """体检：ComfyUI 系统信息 + 能力开关 + 我们自己的状态 ✓（对应 ``/system_stats`` + ``/features`` ✓）。"""
    client = _client()
    try:
        stats = client.system_stats()
        features = client.features()
        reachable = bool(stats)
        detail = str((stats.get("system") or {}).get("comfyui_version") or "ok")
    except Exception as err:  # noqa: BLE001
        stats, features, reachable, detail = {}, {}, False, f"{type(err).__name__}: {err}"[:120]
    return {
        "comfyui": {"url": COMFYUI_URL, "reachable": reachable, "detail": detail,
                    "stats": stats, "features": features},
        "service": {"output_dir": str(OUTPUT_DIR), "public_base_url": PUBLIC_BASE_URL,
                    "timeout_seconds": TIMEOUT, "install_hint": INSTALL_HINT},
        "workflows": {name: (WORKFLOW_DIR / name).is_file() for name in (T2V_WORKFLOW, I2V_WORKFLOW)},
    }


@app.get("/v1/catalog/nodes")
def catalog_nodes(class_name: str | None = None) -> Dict[str, Any]:
    """节点目录：不带参数返回**类名清单**（对应 ``/object_info`` ✓，但默认只给名字 ✗ 不给几 MB 全表 ✓）。"""
    client = _client()
    if class_name:
        info = client.object_info(class_name)
        if not info:
            raise HTTPException(status_code=404, detail=f"节点类不存在：{class_name}")
        return info
    return {"classes": sorted((client.object_info_all() or {}).keys())}


@app.get("/v1/catalog/models")
def catalog_models(folder: str | None = None) -> Dict[str, Any]:
    """模型目录：不带参数返回**类别清单** ✓；带 ``folder`` 返回该类文件清单 ✓（``/models[/{folder}]`` ✓）。"""
    client = _client()
    if folder:
        return {"folder": folder, "files": client.models(folder)}
    return {"folders": client.model_types()}


@app.get("/v1/catalog/folders")
def catalog_folders() -> Dict[str, Any]:
    """各类模型的**磁盘路径**（对应内部端点 ``/internal/folder_paths`` ✓；老版本没有 ⇒ 回退仅给类别 ✓）。"""
    client = _client()
    paths = client.folder_paths()
    if not paths:
        return {"folders": client.model_types(),
                "note": "该 ComfyUI 版本没有 /internal/folder_paths ⇒ 只给类别清单 ✓"}
    return paths


@app.get("/v1/catalog/files/{directory_type}")
def catalog_files(directory_type: str) -> Dict[str, Any]:
    """输入/输出目录的文件清单（对应 ``/internal/files/{type}`` ✓）。"""
    return {"directory": directory_type, "files": _client().internal_files(directory_type)}


@app.get("/v1/queue")
def queue_snapshot() -> Dict[str, Any]:
    """队列快照 + 剩余量（``/queue`` + ``/prompt`` ✓）。"""
    client = _client()
    return {"queue": client.queue(), "remaining": client.queue_remaining()}


@app.post("/v1/queue/clear")
def queue_clear() -> Dict[str, Any]:
    """清空待执行队列（``POST /queue {clear:true}`` ✓）。"""
    return {"ok": _client().clear_queue()}


@app.delete("/v1/queue/{prompt_id}")
def queue_delete(prompt_id: str) -> Dict[str, Any]:
    """从队列里删掉某个待执行任务（``POST /queue {delete:[id]}`` ✓）。"""
    return {"ok": _client().delete_queue_item(prompt_id)}


@app.post("/v1/interrupt")
def interrupt(body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """打断当前执行（可指定 ``prompt_id`` ✓）。"""
    return {"ok": _client().interrupt((body or {}).get("prompt_id"))}


@app.post("/v1/free")
def free_vram(body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """卸载模型 / 释放显存（``POST /free`` ✓）。"""
    payload = body or {}
    return {"ok": _client().free(unload_models=payload.get("unload_models", True),
                                free_memory=payload.get("free_memory", True))}


@app.get("/v1/history")
def history(max_items: int | None = None, offset: int = -1) -> Dict[str, Any]:
    """历史（``/history`` ✓；可 ``max_items`` 限量 ✓）。"""
    return {"history": _client().history_all(max_items=max_items, offset=offset)}


@app.delete("/v1/history")
def history_clear() -> Dict[str, Any]:
    """清空历史（``POST /history {clear:true}`` ✓）。"""
    return {"ok": _client().clear_history()}


@app.get("/v1/jobs")
def jobs() -> Dict[str, Any]:
    """作业列表（新版 ``/api/jobs`` ✓；老版本返回 ``supported: false`` ✓ 不假装成功 ✓）。"""
    data = _client().jobs()
    return {"supported": data is not None, "jobs": data}


@app.post("/v1/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> Dict[str, Any]:
    """取消作业（``/api/jobs/{id}/cancel`` ✓）。"""
    return {"ok": _client().cancel_job(job_id)}


@app.post("/v1/upload/image")
async def upload_image_proxy(image: UploadFile = File(...), subfolder: str = Form(""),
                             overwrite: bool = Form(True)) -> Dict[str, Any]:
    """**同上载能力**（对应 ``/upload/image`` ✓）—— 让调用方不必自己直连 ComfyUI ✓。

    multipart 收（FastAPI 原生 ✓）→ 转交 ComfyUI（客户端手搓 multipart ✓）→ 回
    ``{name, subfolder, type}`` ✓（正是 ``LoadImage.image`` 需要的形状 ✓）。
    """
    return _client().upload_image(image.filename or "upload.png", await image.read(),
                                  subfolder=subfolder, overwrite=overwrite)


@app.get("/v1/view")
def view_proxy(filename: str, subfolder: str = "", type: str = "output") -> Response:  # noqa: A002
    """**同取件能力**（对应 ``/view`` ✓）：取 ComfyUI 输出目录里的任意产物字节流 ✓。"""
    item = {"filename": filename, "subfolder": subfolder, "type": type}
    try:
        payload = _client().fetch_output(item)
    except cc.ComfyUIError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    media = "video/mp4" if str(filename).lower().endswith((".mp4", ".webm")) else "application/octet-stream"
    return Response(content=payload, media_type=media)


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    """体检：本封装 + **ComfyUI 可达性** + 模板是否在盘（排障一次看全 ✓）。"""
    reachable = False
    detail = ""
    try:
        client = cc.ComfyUIClient(COMFYUI_URL, timeout=5)
        stats = client.system_stats()
        reachable = bool(stats)
        detail = str((stats.get("system") or {}).get("comfyui_version") or "ok")
    except Exception as err:  # noqa: BLE001
        detail = f"{type(err).__name__}: {err}"[:120]
    workflows = {name: (WORKFLOW_DIR / name).is_file()
                 for name in (T2V_WORKFLOW, I2V_WORKFLOW)}
    return {
        "ok": True,
        "comfyui_url": COMFYUI_URL,
        "comfyui_reachable": reachable,
        "comfyui_detail": detail,
        "workflows": workflows,
        "output_dir": str(OUTPUT_DIR),
        "public_base_url": PUBLIC_BASE_URL,
        "install_hint": INSTALL_HINT,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
