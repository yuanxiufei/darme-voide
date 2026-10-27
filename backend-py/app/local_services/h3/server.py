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
  ``GET  /v1/workflows/templates``  **盘上所有 H3 模板的形态与注入点** ✓（全部**从图上读** ✓，见下）
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
  body → 选模板（默认自家加速模板 ✓；有参考素材 ⇒ 换官方 R2V ✓；body ``template`` 优先 ✓）
       → ``workflow.video_plan()`` **读图**定注入点（形态/提示词宿主/时长/画幅/首尾帧/参考槽 ✓）
       → 注入（提示词/时长/画幅/种子/权重/参考图 ✓）→ 首尾帧 + **有槽位才传的**参考图上传
       → `workflow.ui_to_api()`（**UI 图 → API 格式** ✓）→ 节点预检（缺哪个包直接点名 ✓）
       → `comfyui_client.queue_prompt()` → `wait()` 轮询 → `/view` 取片 → 落盘 → 回 URL
       → **`/free` 卸载模型释放显存**（24G 卡上这是必须的 ✓）

⚠️ **注入点一律从图上判，不认标题也不认节点 id** ✓✗（2026-09-27 ✓）：三种模板三种标题 ✓、
官方模板的 ``LoadImage`` **根本没标题** ✗、R2V 的 ``prompt`` 还是**连线端口**不是 widget ✗
⇒ 「字段在哪个节点」这件事只能靠**连接关系**问图 ✓（``workflow.py::video_plan`` ✓）。

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
``H3_R2V_WORKFLOW``         参考素材（R2V）模板文件名        ``video_minimax_h3_r2v.json``
``H3_TEMPLATE``             强制指定模板（空=按入参自动挑 ✓） 空（body ``template`` 优先 ✓）
``H3_REF_IMAGE_LIMIT``      最多接几路参考图 ✓              ``3``（盘上 R2V 模板接了 ``ref_image_0..2`` ✓）
========  ============================================  ==============================

⚠️ 已知限制（**如实列出**，不假装支持 ✓）：
* **参考素材（Ref2VA）只有走 R2V 模板才真生效** ✗：传了参考图 + 自家那个 FL2VA/I2VA 模板 ⇒ 主节点上
  压根没有 ``ref_images.*`` 槽位 ⇒ **响亮告警**说明被忽略 ✓、且**连上传都不做** ✓（别在 ComfyUI 的
  ``input`` 目录里留垃圾 ✓）。传参考图时**没显式指定模板 ⇒ 自动换官方 R2V 模板** ✓。
* 参考素材**只接图** ✓：节点**类**其实还支持 ``ref_videos`` / ``ref_video_audios`` / ``ref_audios``
  （各 ≤3 ✓，出处 ``comfy_extras/nodes_minimax_h3.py`` ✓），但**盘上那个 R2V 模板一格都没接** ✗
  ⇒ ``reference_audio*`` 一律**响亮告警 + 忽略** ✓。⚠️ 这条判据是**图上真有的端口** ✓✗
  （``workflow.py::reference_slot_names`` ✓）⇒ 哪天模板把音频槽位接上，这里**不用改代码**就认 ✓。
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

# ── 契约层（提交前那两道关 ✓）：**包内导入**与**独立部署**（``python server.py`` ✓）都要能起来 ✓✗ ──
#    ⚠️ 兄弟模块自己 import ``app.*``（``accel.py`` / ``prompt.py`` ✓）⇒ **仓库根必须先补进 sys.path** ✗
#    （独立部署时 cwd 就在本目录 ✓，``app`` 根本不在路径上 ✓✗）。
CONTRACT_ERROR: str | None = None
_REPO_ROOT = Path(__file__).resolve().parents[3]        # …/backend-py（含 ``app/`` ✓）
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:  # pragma: no cover - 取决于加载方式
    from . import accel as accel_seam                    # noqa: E402
    from . import prompt as prompt_contract              # noqa: E402
except ImportError:  # pragma: no cover
    try:
        import accel as accel_seam                       # type: ignore[no-redef]
        import prompt as prompt_contract                 # type: ignore[no-redef]
    except ImportError as err:
        accel_seam = None                                # type: ignore[assignment]
        prompt_contract = None                           # type: ignore[assignment]
        CONTRACT_ERROR = f"{type(err).__name__}: {err}"

PORT = int(os.environ.get("H3_PORT") or "8765")
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
TIMEOUT = float(os.environ.get("H3_TIMEOUT") or "1800")
WORKFLOW_DIR = Path(os.environ.get("H3_WORKFLOW_DIR") or (Path(__file__).resolve().parent / "workflows"))
OUTPUT_DIR = Path(os.environ.get("H3_OUTPUT_DIR") or (Path(__file__).resolve().parent / "outputs"))
PUBLIC_BASE_URL = (os.environ.get("H3_PUBLIC_BASE_URL") or f"http://127.0.0.1:{PORT}").rstrip("/")
T2V_WORKFLOW = os.environ.get("H3_T2V_WORKFLOW") or "MiniMax_H3_Fast_T2V.json"
I2V_WORKFLOW = os.environ.get("H3_I2V_WORKFLOW") or "MiniMax_H3_Fast_I2V.json"
#: R2V（参考素材）模板：**官方**那份 ✓（主节点 ``MiniMaxH3ReferenceToVideo`` ✓，见该目录 README 出处表 ✓）
R2V_WORKFLOW = os.environ.get("H3_R2V_WORKFLOW") or "video_minimax_h3_r2v.json"
#: 显式指定模板（空 = 按入参自动挑 ✓）；请求体里的 ``template`` 优先于它 ✓
TEMPLATE = os.environ.get("H3_TEMPLATE") or ""
#: 参考图最多几路（官方 R2V 模板的槽位是 ``ref_images.ref_image_0..2`` ⇒ 3 ✓）
REF_IMAGE_LIMIT = int(os.environ.get("H3_REF_IMAGE_LIMIT") or "3")
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
def _as_list(raw: Any) -> list[Any]:
    """**两种形态都认** ✗✗：``reference_audio`` 可能是列表 ✓、也可能是 JSON 字符串 ✓。

    ⚠️ 只认一种 ⇒ 另一种被判成「0 路」⇒ 编号校验/声明全部按空的算 ✓✗（正是「静默错」的形状 ✓）。
    """
    import json

    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _reference_images(body: Dict[str, Any]) -> list[str]:
    """参考图列表 ✓：``referenceImageUrls``（后端协议口径 ✓）与 ``subject_reference``（H3 契约口径 ✓）
    **两套命名都收** ✓，按出现顺序去重 ✓。

    ⚠️ 只认一套 ⇒ 另一套被当成「0 路」⇒ 参考素材**静默不生效** ✓✗（本仓反复踩的同义异名坑 ✓）。
    """
    out: list[str] = []
    for raw in (body.get("referenceImageUrls"), body.get("subject_reference")):
        for item in _media_texts(raw):
            if item not in out:
                out.append(item)
    return out


def _media_texts(raw: Any) -> list[str]:
    """把媒体入参读成非空字符串列表 ✓（列表 ✓ / 单个字符串 ✓ / JSON 字符串列表 ✓ 都认 ✓）。"""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        parsed = _as_list(text)
        return [str(item).strip() for item in parsed if str(item).strip()] or [text]
    return [str(item).strip() for item in _as_list(raw) if str(item).strip()]


def _pick_template(task: Dict[str, Any], refs: list[str]) -> str:
    """选模板 ✓：**显式指定 > 有参考素材 ⇒ R2V > 有首帧 ⇒ I2V > 纯文生 ⇒ T2V** ✓。

    显式指定来自 body ``template`` ✓（优先）或 env ``H3_TEMPLATE`` ✓。

    ⚠️ 有参考素材**必须**换 R2V 模板 ✗✗：A 批那两个模板的主节点是 ``MiniMaxH3ImageToVideo`` ✓，
    **没有参考图槽位** ✗ ⇒ 参考素材只能被丢掉 ✓（旧行为：响告警 + 忽略 ✓）。
    ⚠️ R2V 模板不在盘上 ⇒ **如实告警 + 退回** ✓（不假装支持 ✓；也不因为「可选素材」把整个任务判死 ✓）。
    """
    body = task.get("body") or {}
    explicit = str(body.get("template") or TEMPLATE or "").strip()
    if explicit:
        return explicit
    if refs:
        if (WORKFLOW_DIR / R2V_WORKFLOW).exists():
            return R2V_WORKFLOW
        _warn(task, f"请求带了 {len(refs)} 路参考图，但 R2V 模板不在盘上（{R2V_WORKFLOW} ✗）"
                    f"⇒ 本次**忽略它们** ✗（要用参考素材请把模板放回 {WORKFLOW_DIR} ✓）")
    first_frame = str(body.get("first_frame_image") or "").strip()
    return I2V_WORKFLOW if first_frame else T2V_WORKFLOW


def _wire_reference_images(task: Dict[str, Any], api: Dict[str, Any], plan: Dict[str, Any],
                           uploaded: list[str]) -> int:
    """把上传好的参考图落到主节点的 ``ref_images.ref_image_k`` ✓ ⇒ **落地路数** ✓。

    ⚠️ **没用的槽位必须摘掉** ✗✗：官方 R2V 模板里那几个 ``LoadImage`` 指着模板自带的示例图 ✓
    （文件名在**我们的** input 目录里不存在 ✓）—— 而 ``LoadImage.image`` 是 **COMBO** ✓
    ⇒ 只要它还在图里，ComfyUI 的提交前校验就会因为「文件名不在可选列表」**当场 400** ✓✗
    （即使这个节点根本不参与出图 ✓）。所以：调用方给了几路，就只留几路 ✓，
    多出来的槽位**连同它上游那个 LoadImage 一起删掉** ✓（该 LoadImage 无输入 ✓，不会牵连别的节点 ✓）。
    """
    main_id = plan.get("main_id")
    slots = list(plan.get("ref_slot_names") or [])
    wired = list(plan.get("ref_image_ids") or [])
    landed = 0
    for index, slot_name in enumerate(slots):
        existing = wired[index] if index < len(wired) else None
        if index >= len(uploaded):
            if existing is not None:
                (api.get(str(main_id)) or {}).get("inputs", {}).pop(slot_name, None)
                if not _is_referenced(api, existing):
                    api.pop(str(existing), None)
            continue
        if existing is not None:
            wf.set_node_input(api, existing, "image", uploaded[index])
            landed += 1
            continue
        new_id = f"voide_ref_image_{index + 1}"
        api[new_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded[index]},
            "_meta": {"title": f"REF IMAGE {index + 1} (voide)"},
        }
        wf.set_node_input(api, main_id, slot_name, [new_id, 0])
        landed += 1
    if landed < len(uploaded):
        _warn(task, f"参考图只落地了 {landed} 路（模板槽位 {len(slots)} 个，请求 {len(uploaded)} 路 ✗）"
                    f"⇒ 多出来的被丢掉 ✗")
    return landed


def _is_referenced(api: Dict[str, Any], node_id: Any) -> bool:
    """图里还有没有别的输入指着 ``node_id`` ✓（删节点前先问一句 ✓）。"""
    target = str(node_id)
    for node in api.values():
        if not isinstance(node, dict):
            continue
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, list) and value and str(value[0]) == target:
                return True
    return False


def _contract_prompt(task: Dict[str, Any], body: Dict[str, Any], prompt_text: str, *,
                     pictures: int | None = None) -> str:
    """提示词过 **H3 prompt 契约** ✓（提交前 ✓）：文本写法 → 标签、**核编号**、**该追加的声明必须追加** ✗。

    ⭐ 钉的就是那条实测坑 ✗✗：**只写 ``<Audio 1>`` 绑定句不算声明** ✓ —— 照标签跳过的话，
    自动声明被吞、模型**不复用参考音频**（实测相关性≈0 ✓✗）。所以这一段必须在**提交前**跑 ✓。

    ⚠️ ``pictures``：**实际落地**的参考图路数 ✓ 优先于 body 里的 ``subject_reference`` ✗ ——
    参考图有两种命名来源 ✓（``referenceImageUrls`` 与 ``subject_reference`` ✓），只数后者会在
    前者形态下被判成 0 路 ✓✗ ⇒ ``<Picture n>`` 的编号与自动声明全按空的算 ✓✗（静默错 ✓）。
    所以默认值只在**调用方没给**时才用 ✓。

    ⚠️ 编号错 ⇒ **当场拒** ✗（``ConditioningError`` 冒上去 ⇒ 任务带可行动文案失败 ✓）；
    ⚠️ 契约层没装上 ⇒ **如实告警「没跑」** ✗ 而不是当通过 ✓✗。
    """
    if prompt_contract is None:
        _warn(task, f"H3 提示词契约层没装上（{CONTRACT_ERROR}）⇒ **提交前那道关没跑** ✗（没查不是通过 ✓✗）")
        return prompt_text
    cond_mod = getattr(prompt_contract, "cond", None)
    audios = len(_as_list(body.get("reference_audio")))
    if pictures is None:
        pictures = len(_as_list(body.get("subject_reference")))
    # 参考音频一律是**音色参考** ✓ ⇒ 协议关系 ``reference`` ✓（⚠️ 别用 ``fully_copy`` ✗ ——
    # 那是「逐字复制内容」的意思 ✓，会把参考音频的台词念出来 ✓✗）
    refs = [cond_mod.AudioReference(index=index + 1, relation="reference")
            for index in range(audios)] if cond_mod is not None else []
    settings = body.get("settings") if isinstance(body.get("settings"), dict) else {}
    result = prompt_contract.build_prompt(
        prompt_text, pictures=pictures, audios=audios, videos=0, audio_refs=refs,
        video_count=0, global_prompt=str((settings or {}).get("global_prompt") or ""),
    )
    _update(task, prompt_contract=result["plan"], prompt_markup=result["markup"],
            prompt_prefixed=result["prefixed"])
    return str(result["prompt"])


def _io_ref(api: Dict[str, Any], class_type: str, slot: int = 0) -> list[Any] | None:
    """在 API 图里找某个类节点的输出引用 ✓ ⇒ ``[节点id, 槽位]``（找不到 ⇒ ``None`` ✓）。"""
    for node_id, node in api.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return [str(node_id), slot]
    return None


def _rewire(api: Dict[str, Any], old_ref: list[Any], new_ref: list[Any]) -> int:
    """把图里所有指向 ``old_ref`` 的输入改指 ``new_ref`` ✓ ⇒ 改了几处 ✓。

    ⚠️ 必须**按值逐处改** ✗（漏一处 ⇒ 那条流还在用没加速的模型 ✓✗，图看起来还是对的 ✓）。
    """
    changed = 0
    for node in api.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in list(inputs.items()):
            if isinstance(value, list) and len(value) == 2 and list(value) == list(old_ref):
                inputs[key] = list(new_ref)
                changed += 1
    return changed


def _apply_accel_chain(client: "cc.ComfyUIClient", task: Dict[str, Any],
                       api: Dict[str, Any]) -> None:
    """把**加速链**接到图上 ✓（配置驱动 ✓；口径在 :mod:`app.services.engine.accel_chain` ✓）。

    配置：``body.settings.accel_chain = {"chain": [...]}``（或裸数组 ✓）—— 没配 ⇒ **什么都不做**
    （不告警 ✓，这是默认形态 ✓）。

    三条判据（都是「出问题会静默变样」的形状 ✓✗）：
    1. ⭐ 判 ``error`` 的环 ⇒ ``build=None`` ⇒ **整条链不上** ✓ 并**点名理由** ✗（静默少一环 = 图悄悄变样 ✓✗）；
    2. ⭐ ComfyUI 连不上 ⇒ 判「**没查**」✗（``unchecked`` ✓）⇒ 照样不上链 ✓，但**必须说清是「没查」不是「没问题」** ✓✗；
    3. ⭐ 加速链是**提速件** ✓ 不是正确性 ⇒ 装不上就**退回未加速**跑 ✓（比拒单合理 ✓），但这一定得在日志里看得见 ✓。
    """
    settings = task.get("body", {}).get("settings")
    raw_chain = settings.get("accel_chain") if isinstance(settings, dict) else None
    if not raw_chain:
        return
    if accel_seam is None:
        _warn(task, f"加速链配置给了，但接缝层没装上（{CONTRACT_ERROR}）⇒ **没核也没接** ✗"
                    f"（没查不是通过 ✓✗；这一单按未加速跑 ✓）")
        return
    model_ref = _io_ref(api, "UNETLoader") or _io_ref(api, "CheckpointLoaderSimple")
    clip_ref = _io_ref(api, "CLIPLoader")
    if model_ref is None or clip_ref is None:
        _warn(task, f"模板里没找到 UNETLoader/CLIPLoader（{model_ref} / {clip_ref} ✓）⇒ "
                    f"加速链**没接** ✗（链路要从这两条流上串 ✓）")
        return
    report = accel_seam.check_chain(client, raw_chain, model=model_ref, clip=clip_ref)
    _update(task, accel_chain={"counts": report["counts"], "blocked": report["blocked"],
                               "unreachable": report["unreachable"],
                               "statuses": report["statuses"],
                               "build": None if report["build"] is None else report["build"]["enabled"]})
    if report["unreachable"]:
        _warn(task, f"加速链**没核**（{report['unreachable']}）✗ ⇒ 这一单**不上链** ✓✗"
                    f"（没查不是通过 ✓）")
    if report["build"] is None:
        for status in report["statuses"]:
            for issue in status["issues"]:
                if status["blocksSubmit"]:
                    _warn(task, f"加速项 {status['id']} 判 error ✗ ⇒ 整条链不上：{issue['message']}")
        return
    build = report["build"]
    api.update(build["nodes"])
    rewired = _rewire(api, model_ref, build["model"])
    if list(build["clip"]) != list(clip_ref):
        rewired += _rewire(api, clip_ref, build["clip"])
    skipped = ", ".join(f"{item['id']}（{item['reason']}）" for item in build["skipped"]) or "无"
    # ⚠️ 接上了**不是** warn ✗（"warnings" 是给「出问题了」用的 ✓）；如实记在任务里 ✓ 让日志看得见 ✓
    _update(task, accel_rewired=rewired, accel_enabled=list(build["enabled"]),
            accel_skipped=skipped)
    if rewired == 0:
        _warn(task, "加速链建好了节点，但**没有任何一条输入指向它** ✗ ⇒ 等于没加速 ✓✗"
                    "（模板结构与预期不符？检查 ✓）")


def _pick_checkpoint(client: "cc.ComfyUIClient", task: Dict[str, Any], requested: str,
                     checkpoint_map: Any, main_class: str = "") -> str | None:
    """选 UNET 权重 ✓ ⇒ 文件名字符串（``None`` = 沿用模板默认 ✓）。

    ``main_class``：模板**主节点类名**（来自 ``wf.video_plan`` ✓）—— 用来判「这个模板吃不吃参考素材」✓✗：
    主节点是 ``MiniMaxH3ReferenceToVideo`` ⇒ 参考图**真能落地** ✓ ⇒ 「换了 Ref2VA 权重却白换」那条
    告警**不该发** ✗（发出去就是假消息 ✓）。

    ⭐ **调用方给的 ``body.model`` 优先** ✗✗（2026-09-24 补 ✓）—— 后端已经按 H3 形态规则
    （有参考素材 ⇒ 只能 Ref2VA ✓）选过一次了 ✓；这里**再按场景猜一遍**就是「同一件事两处判断」✓✗，
    两份判据还会**互相覆盖**（后端按参考素材选的权重被这里按 ``scene`` 猜的顶掉 ✓✗）。
    此前本函数的实际行为更糟：``body.model`` **从来没被读过** ✗ ⇒ 后端算出来的权重
    **静默落空**、永远用模板自带的那一个 ✓✗（「能力接不出去不算功能」✓）。

    ⚠️ 用它之前先核**这个文件在不在** ✗（``GET /models/diffusion_models`` ✓ 这是 ComfyUI 的权威目录 ✓）：
    名字对不上就**沿用模板默认并告警** ✓✗ —— 直接塞给 ``UNETLoader`` 会让 ComfyUI 校验报错 ✓，
    而静默换成别的权重更糟 ✓✗（换了模型你还看不出来 ✓）。⚠️ 清单取不到 ⇒ **告警说「没核」** ✗
    而不是默认它存在 ✓。
    """
    chosen: str | None = None
    if requested:
        try:
            available = client.models("diffusion_models")
        except Exception as err:  # noqa: BLE001 —— 连不上/接口变了 ⇒ **没核** ✗ 不算通过 ✓
            _warn(task, f"取 ComfyUI 模型清单失败（{type(err).__name__}: {err}）⇒ "
                        f"**没核**调用方要的权重 {requested!r} ✗（没查不是通过 ✓✗）")
        else:
            if requested in available:
                chosen = requested
            else:
                _warn(task, f"调用方指定的权重 {requested!r} **不在 ComfyUI 的 diffusion_models 里** ✗ ⇒ "
                            f"不用它（该类现有 {len(available)} 个，例如 "
                            f"{', '.join(available[:3]) if available else '（空）'} ✓）")
    # ⚠️ 「没核」与「核了不认」都**继续往下走**✓✗ —— 兜底是 ``settings.checkpoint_map`` ✓，
    #    最后才是模板默认 ✓（**别在这里提前 return** ✗：那会把兜底也跳掉 ✓✗）
    if chosen is None and isinstance(checkpoint_map, dict) and checkpoint_map:
        scene = str(task.get("body", {}).get("scene_type") or checkpoint_map.get("scene_type") or "").lower()
        chosen = (checkpoint_map.get("ref2va") if scene and scene != "action" and scene != "silent"
                  else checkpoint_map.get("fl2va")) or checkpoint_map.get("fl2va")
    if chosen and isinstance(checkpoint_map, dict) and checkpoint_map.get("ref2va") \
            and chosen == checkpoint_map.get("ref2va") \
            and main_class != "MiniMaxH3ReferenceToVideo":
        _warn(task, "模板是 FL2VA 形态：已换 Ref2VA 权重名，但**参考图/参考音频不会生效** ✗"
                    f"（要参考素材请用 R2V 模板：{R2V_WORKFLOW} ✓ —— 那个模板里 ``ref2va`` 权重才对得上 ✓）")
    return chosen if isinstance(chosen, str) and chosen else None



def _build_api_prompt(client: "cc.ComfyUIClient", task: Dict[str, Any]
                      ) -> tuple[dict[str, Any], list[str]]:
    """把请求组装成可直接 ``POST /prompt`` 的 **API 格式**工作流；返回 (prompt, dropped 节点 id)。"""
    body: dict[str, Any] = task.get("body") or {}
    prompt_text = str(body.get("prompt") or "").strip()
    if not prompt_text:
        raise ValueError("prompt 为空（后端必须给提示词 ✓）")

    first_frame = str(body.get("first_frame_image") or "").strip()
    last_frame = str(body.get("last_frame_image") or "").strip()
    all_refs = _reference_images(body)
    refs = all_refs[:REF_IMAGE_LIMIT]
    if len(all_refs) > len(refs):
        _warn(task, f"参考图给多了（{len(all_refs)} 路 > 上限 {REF_IMAGE_LIMIT} 路 ✗）"
                    f"⇒ 只落地前 {len(refs)} 路 ✗（上限口径见官方 R2V 模板的槽位数 ✓）")
    template_name = _pick_template(task, refs)
    template_path = WORKFLOW_DIR / template_name
    if not template_path.exists():
        raise ValueError(f"工作流模板不在盘上：{template_path}（检查 H3_WORKFLOW_DIR / H3_TEMPLATE ✓）")
    import json

    graph = json.loads(template_path.read_text(encoding="utf-8"))

    # ── 注入点**全靠图判据** ✓（不按标题 ✗、不按节点 id ✗）──
    # ⚠️ 为什么不能按标题 ✗✗：实测同一件事在三种模板上三种标题 ✓ —— 时长是 ``DURATION (seconds)`` /
    #    ``Float (duration)`` / ``Float (Duration)`` ✓（连大小写都不统一 ✗），
    #    首帧的 ``LoadImage`` 在官方模板里**完全没标题** ✗；R2V 的 ``prompt`` 还被**拉着线** ✗
    #    （往主节点上塞字面量会把连线覆盖掉 ✓✗）。图判据对三种模板用同一套 ✓。
    plan = wf.video_plan(graph)
    if plan["main_id"] is None:
        raise ValueError(
            f"模板 {template_name} 里没有**唯一**的 H3 主节点"
            f"（{' / '.join(wf.MAIN_VIDEO_CLASSES)}）⇒ 不知道往哪儿注入 ✗"
            f"（云端 api_* 那三个走 ComfyUI 云端节点 ✓ 不归本服务管 ✗；显式指定模板时请挑本地模板 ✓）")

    # 需要尾帧 ⇒ 把模板里**默认关掉**的可选尾帧通道打开（参考 I2V 模板的 [DISABLED] 节点 ✓）
    # ⚠️ 这里**不发告警** ✗：激活不到通道时，下面「这一帧落不下去」那条会**统一**发 ✓
    #    （两处各发一条 ⇒ 同一件事两条告警 ⇒ 读的人以为出了两个问题 ✓✗）。
    activated: list[str] = []
    if last_frame:
        activated = wf.activate_nodes(graph)

    # ⚠️⚠️ **参考图能不能落地，在上传之前就已经知道了** ✓✗：模板主节点上有没有 ``ref_images.*``
    #    槽位是**图上的事实** ✓（``plan`` 已算出来 ✓）⇒ 没有槽位时这几路**注定落不下去** ✓
    #    ⇒ 那就**一个都别传** ✗（传了只是往 ComfyUI 的 ``input`` 目录里留垃圾 ✓✗、
    #    还白等几轮上传 ✓）。自检 ㉔′ 钉的就是这条 ✓。
    ref_slot_names = list(plan["ref_slot_names"])
    if refs and not ref_slot_names:
        _warn(task, f"模板（{template_name}）的主节点**没有参考图槽位** ⇒ {len(refs)} 路参考图被忽略 ✗"
                    f"（**一个都没上传** ✓，别在 ComfyUI 的 input 目录里留垃圾 ✓；要用参考素材请换 R2V "
                    f"模板：{R2V_WORKFLOW} ✓，或 body ``template`` 显式指定 ✓）")

    # ⚠️ 首尾帧**同一条纪律** ✓✗：模板里有没有对应的 ``LoadImage`` 也是**图上的事实** ✓（``plan`` 优先 ✓，
    #    标题只当兜底 ✓ —— A 批模板的尾帧通道是「被 mute 的一整条支路」✓，用标题能一眼认出 ✓）
    #    ⇒ 没有就不传 ✗。⚠️ 这条**不是理论** ✓：R2V 主节点**压根没有** ``first_frame``/``last_frame``
    #    端口 ✗ ⇒ 「带参考图 + 带首帧」现在会自动走 R2V ✓ ⇒ 首帧注定落不下去 ✓（旧写法会白传一张 ✓✗）。
    frame_nodes = {
        "START FRAME": plan["first_frame_id"] or _find_load_image(graph, "START FRAME"),
        "LAST FRAME": plan["last_frame_id"] or _find_load_image(graph, "LAST FRAME"),
    }
    for keyword, requested in (("START FRAME", first_frame), ("LAST FRAME", last_frame)):
        if requested and not frame_nodes[keyword]:
            _warn(task, f"给到了 {keyword}，但模板（{template_name}）里**没有**对应的 LoadImage 端口 "
                        f"⇒ 该帧**不生效** ✗（**也没有上传** ✓：落不下去的东西不往 ComfyUI 的 input "
                        f"目录里塞 ✓）"
                        + ("；要首帧请挑 I2V 模板 ✓ 或 body ``template`` 显式指定 ✓"
                           if keyword == "START FRAME" else
                           "；要首尾帧得用官方的 flf2v 模板 ✓（官方 I2V 模板本来就没有尾帧槽位 ✗；"
                           "本仓那两个的尾帧通道是**被 mute 的一整条支路** ✓）"))

    # 首帧/尾帧/参考图上传（ComfyUI 只认它自己 input 目录里的文件名 ✗ ⇒ 必须先 upload ✓）
    # ⚠️ 参考图那几路**以「有没有槽位」为准** ✗（不是以「请求里给没给」为准 ✗）——
    #    这个列表后面还要拿去算 ``ref_uploads`` ✓，两处口径不同步就会在没槽位时炸 ``KeyError`` ✓✗
    #    （自检实测踩到 ✓）。
    wanted_refs = refs if ref_slot_names else []
    uploads: list[tuple[str, str, bytes]] = []
    if first_frame and frame_nodes["START FRAME"]:
        uploads.append(("START FRAME", "first_frame.png", _fetch_media(first_frame)))
    if last_frame and frame_nodes["LAST FRAME"]:
        uploads.append(("LAST FRAME", "last_frame.png", _fetch_media(last_frame)))
    for index, url in enumerate(wanted_refs):
        uploads.append((f"REF IMAGE {index + 1}", f"ref_image_{index + 1}.png", _fetch_media(url)))

    uploaded: dict[str, str] = {}
    for keyword, filename, payload in uploads:
        result = client.upload_image(filename, payload)
        name = result.get("name") or filename
        subfolder = result.get("subfolder") or ""
        uploaded[keyword] = f"{subfolder}/{name}" if subfolder else str(name)
    ref_uploads = [uploaded[f"REF IMAGE {index + 1}"] for index in range(len(wanted_refs))]

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

    # ── 注入：提示词（**先过 H3 prompt 契约** ✓：文本写法→标签 / 核编号 / 该追加的声明必须追加 ✓）──
    # ⚠️ 契约里的 ``pictures`` 用**实际落地的参考图路数** ✓（不是只数 body 里那个键 ✗）——
    #    而「实际落地」在提交前就唯一确定 ✓：``min(请求路数, 模板槽位数)`` ✓。
    #    ⚠️ 模板没槽位时是 **0** ✓（不是请求路数 ✗）⇒ 提示词里若还引用 ``<Picture 1>`` 就会被
    #    **当场拒** ✓✗（那是对的：它要的素材这单根本交不出去 ✓，比静默出一版「没有那张脸」的片好 ✓）。
    prompt_text = _contract_prompt(task, body, prompt_text,
                                   pictures=min(len(refs), len(ref_slot_names)) if refs else None)
    owner = plan["prompt_owner"]
    if owner is None:
        _warn(task, "没找到提示词注入点 ✗（主节点的 ``prompt`` 既不是自身 widget ✓、"
                    "上游也不是字符串节点 ✓）⇒ 这次提交的提示词是**模板里那串默认文案** ✗")
    elif not wf.set_node_input(api, owner[0], owner[1], prompt_text):
        _warn(task, f"提示词注入点 {owner[0]}.{owner[1]} 不在 API 图里 ✗（转换时被丢了？）")

    # ── 注入：时长（秒 → **喂 ``length`` 的那个标量节点** ✓；帧数由模板表达式按 24fps/17k+5 算 ✓）──
    duration = float(body.get("duration") or 5)
    expected_frames = wf.frames_for_seconds(duration)
    if plan["duration_id"] and wf.set_node_input(api, plan["duration_id"], "value", duration):
        pass
    else:
        _warn(task, f"没找到时长注入点 ✗（喂 ``length`` 的唯一 ``PrimitiveFloat`` 读不出来 ✓）"
                    f"⇒ 时长沿用模板默认值 ✗（请求的是 {duration}s）")

    # ── 注入：画幅（COMBO 必须按**选项表**换，否则会被校验拒 ✓）──
    ratio = str(body.get("aspect_ratio") or "").strip()
    if ratio:
        picked = wf.set_combo(api, "ResolutionSelector", ratio, object_info)
        if not picked:
            _warn(task, f"画幅 {ratio!r} 没匹配到 ResolutionSelector 的选项 ⇒ 用模板默认值 ✗")

    # ── 注入：画质档位（可选；官方模板里**没有**这个档位节点 ⇒ 如实告警 ✓ 不假装设上了 ✗）──
    if QUALITY and not wf.set_input(api, "PrimitiveFloat", "value", float(QUALITY),
                                    title_contains="QUALITY"):
        _warn(task, f"H3_QUALITY={QUALITY} 但模板（{template_name}）里没有画质档位节点 "
                    f"⇒ 画质按模板/画幅算出来的值 ✗")

    # ── 注入：权重（**调用方给的优先** ✓✗；模板是 FL2VA 形态 ⇒ ref2va 只换名字并告警 ✓）──
    settings = body.get("settings") or {}
    checkpoint_map = settings.get("checkpoint_map") if isinstance(settings, dict) else None
    checkpoint = _pick_checkpoint(client, task, str(body.get("model") or "").strip(), checkpoint_map,
                                  main_class=str(plan["main_class"] or ""))
    if checkpoint:
        wf.set_input(api, "UNETLoader", "unet_name", checkpoint)

    # ── 注入：参考素材（图 ✓ / 音频 ✗）──
    #    ⚠️ 「模板没槽位」那条告警**在上传前**就发过了 ✓（那时才决定传不传 ✓）⇒ 这里只管落槽 ✓
    landed_refs = (_wire_reference_images(task, api, plan, ref_uploads)
                   if refs and ref_slot_names else 0)
    if body.get("reference_audio_urls") or body.get("reference_audio"):
        _warn(task, "参考音频（Ref2VA 的音色参考）本地模板还没接上 ⇒ 被忽略 ✗"
                    "（提示词里的 ``<Audio n>`` 也请一并去掉 ✗ 免得声明了却没有对应素材 ✓）")

    # ── 注入：加速链（配置驱动 ✓；没配就是空操作 ✓）──
    _apply_accel_chain(client, task, api)

    # ── 注入：种子（每次不同 ✓）与产物体名（便于追溯 ✓）──
    wf.set_seed(api, "RandomNoise")
    wf.set_input(api, "SaveVideo", "filename_prefix",
                 f"h3/{task.get('task_id') or 'task'}", required=False)

    # ── 首尾帧落到 LoadImage 上（API 里 LoadImage 的入参就是「文件名」✓）──
    # ⚠️ 节点**按图判据找** ✓（官方模板的 ``LoadImage`` 没有标题 ✗ ⇒ 按标题找恒找不到 ✓✗）；
    #    标题只当**兜底** ✓（A 批模板的尾帧通道是「被 mute 的一整条支路」✓，用标题能一眼认出它 ✓）。
    #    ⚠️ 节点表在上传前就算好了 ✓（``frame_nodes`` ✓）⇒ 这里**复用**、别重算 ✗（两处各算一次 ⇒
    #    「传了但没找到」和「找到了但没传」这两种不一致迟早会出现 ✓✗）。
    for keyword, node_id in frame_nodes.items():
        if keyword not in uploaded:
            continue
        if not node_id or str(node_id) not in api:
            # 走到这儿说明「传之前是找得到的 ✓、传完却不在 API 图里 ✗」—— 只可能是转换/激活那步把它丢了 ✓
            _warn(task, f"{keyword} 已上传，但那张 LoadImage 不在最终 API 图里（转换/激活把它丢了 ✗）"
                        f"⇒ 该帧**不生效** ✗")
            continue
        field = _widget_field(api[str(node_id)], "image")
        api[str(node_id)]["inputs"][field] = uploaded[keyword]
        _update(task, **{f"{keyword.lower().replace(' ', '_')}_node": str(node_id)})

    # ── 尾帧通道**接回主节点**：首帧是模板接好的（first_frame ← 那条链的末端 ✓），
    #    尾帧通道刚被激活时是**没接线**的 ✗ ⇒ 这里镜像首帧的接法，把通道**末端**接上 ✓。
    #    「末端」用图判据找（没有任何同通道节点消费它 ✓），不写死节点 id ✗（模板换了也不怕 ✓）。
    form = plan["form"]
    if last_frame and activated:
        referenced = {value[0] for node in api.values()
                      for value in (node.get("inputs") or {}).values()
                      if isinstance(value, list) and len(value) == 2}
        tails = [node_id for node_id in activated if node_id in api and node_id not in referenced]
        if len(tails) == 1 and wf.set_node_input(api, plan["main_id"], "last_frame", [tails[0], 0]):
            _update(task, last_frame_tail=tails[0])
            # 尾帧真接上了 ⇒ 形态就是官方的 **FL2VA** ✓（图上原来是 I2VA 的接线状态 ✓）
            form = "fl2va"
        else:
            _warn(task, f"尾帧通道末端不唯一（{tails}）⇒ 没自动接线，尾帧可能不生效 ✗")

    # ── 预检：**COMBO 白名单** ✓（**只报不改** ✗）。ComfyUI 的 COMBO 是白名单 ✓ ⇒ 表外的串
    #    会被 ``POST /prompt`` 400 拒单 ✗；而「表外的串」最常见的来源就是**抄了别人的展示标签** ✗
    #    （``comfy-org/workflow_templates`` 的 ``mmh3/config.ts`` 写 ``"21:9 (Cinemascope)"`` ✗，
    #    节点真枚举是 ``"21:9 (Ultrawide)"`` ✓ —— 见 ``comfy_extras/nodes_resolution.py`` ✓）。
    #    ⚠️ 放在**所有注入之后** ✓：注入（``set_combo``/``set_input``）本身也可能塞进表外值 ✓。
    for problem in wf.check_combo_choices(api, object_info):
        _warn(task, f"{problem} ⇒ 提交会被 ComfyUI 校验拒 ✗（选项串与节点枚举不一致 ✓）")

    # ── 预检：画幅定下来之后，把「**模板会算出多大**」记进任务 ✓（与 expected_frames 同一类对账口径 ✓；
    #    读不出就 ``None`` ✗ 不猜 ✓）。H3 模板的 ``width``/``height`` 是 **ResolutionSelector 连线算出来的** ✗
    #    ⇒ 节点自带的 ``widgets_values`` 是**旧值** ✗，拿它当时钟会错 ✓（见 ``wf.resolution_selector_size`` ✓）。
    expected_size = wf.expected_resolution(api)
    if expected_size:
        _update(task, expected_width=expected_size[0], expected_height=expected_size[1])

    _update(task, workflow=template_name, form=form, main_node=plan["main_id"],
            dropped_nodes=dropped,
            expected_frames=expected_frames, expected_seconds=wf.seconds_for_frames(expected_frames),
            uploaded=uploaded, reference_images=landed_refs,
            prompt_node=f"{owner[0]}.{owner[1]}" if owner else None,
            duration_node=plan["duration_id"])
    return api, dropped


# ══════════════════════════════════════════════════════════════════════════
# 执行
# ══════════════════════════════════════════════════════════════════════════
def _run_h3(task: Dict[str, Any]) -> None:
    """提交到 ComfyUI 并把产物取回本地（**失败也要 `/free`** ✓）。"""
    client = cc.ComfyUIClient(COMFYUI_URL, timeout=TIMEOUT)
    failure: str | None = None
    success: Dict[str, Any] = {}
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
        success = {"video_url": f"{PUBLIC_BASE_URL}/files/{saved.name}",
                   "outputs": items, "saved_path": str(saved)}
    except cc.NodeMissingError as err:
        failure = f"{err}｜装节点：{INSTALL_HINT}"
    except Exception as err:  # noqa: BLE001 —— 任何失败都要变成任务失败，不能挂在后台 ✗
        failure = f"{type(err).__name__}: {err}"[:600]
    # ⚠️⚠️ **先卸载、再落终态** ✗（2026-09-24 修 ✓）。此前是「先写 `succeeded` ✓，再在 `finally` 里
    #    `/free` + 写 `freed_vram`」✗ ⇒ 中间那一瞬调用方读到的是**自相矛盾**的
    #    「成功了但 `freed_vram=null`」✓✗（真机上是 API 撒谎 ✓；自检侧表现为**偶发** 19/20 ✗✗
    #    —— 2026-09-22 / 09-24 两次全量假红都是它 ✓）。现在终态与 `freed_vram` 在**同一个 `_update`**
    #    里落地 ✓（在锁内 ✓）⇒ 读到的要么是 `processing` ✓、要么是终态 + 已卸载 ✓，没有中间态 ✓。
    #    ⚠️ 失败路径照样卸 ✓（24G 卡否则下一镜 OOM ✗）。
    freed = client.free(unload_models=True, free_memory=True)
    if failure is not None:
        _update(task, status="failed", error_msg=failure, freed_vram=freed)
    else:
        _update(task, status="succeeded", error_msg=None, freed_vram=freed, **success)


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
        # ⭐ 「这一单走的是哪条路、落在图的哪几个节点上」也回给调用方 ✓✗ —— 模板从「只有本仓
        #    自制那两个」扩到「官方那批也在盘上」之后 ✓，**只有服务端知道**它挑了哪个模板、
        #    注入到了哪个节点 ✓ ⇒ 不交出去 ⇒ 出片不对时前端/后端只能猜 ✓✗（与下面两道关同理 ✓）。
        "form": task.get("form"),
        "main_node": task.get("main_node"),
        "prompt_node": task.get("prompt_node"),
        "duration_node": task.get("duration_node"),
        "reference_images": task.get("reference_images"),
        "dropped_nodes": task.get("dropped_nodes"),
        "expected_width": task.get("expected_width"),
        "expected_height": task.get("expected_height"),
        "uploaded": task.get("uploaded"),
        # ⭐ 「提交前那两道关做了什么」也回给调用方 ✓✗ —— 判据要**响亮** ✗：
        #    只写在服务端内存/日志里，出片不对时前端/后端都查不出为什么 ✓✗
        "prompt_contract": task.get("prompt_contract"),
        "prompt_markup": task.get("prompt_markup"),
        "accel_chain": task.get("accel_chain"),
        "accel_enabled": task.get("accel_enabled"),
        "accel_skipped": task.get("accel_skipped"),
        "accel_rewired": task.get("accel_rewired"),
    })


@app.get("/files/{name}")
def get_file(name: str) -> FileResponse:
    """产物下载：只允许 OUTPUT_DIR 下的**文件名本身**（挡掉路径穿越 ✓）。"""
    safe = Path(name).name
    path = OUTPUT_DIR / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)


@app.get("/v1/workflows/templates")
def list_templates() -> JSONResponse:
    """盘上**所有** H3 模板 + 每个模板的**形态 / 注入点** ✓（**全部从图上读** ✓ 不写死名单 ✗）。

    ⚠️ 为什么要开这个口子 ✓：模板从「只有本仓自制那两个」变成「官方那批也在盘上」之后 ✗，
    调用方就**没法知道**盘上有哪些、哪个吃首帧、哪个吃参考图 ✓✗ —— 只能去翻源码或者猜 ✗。
    这里把「盘上有什么 / 各吃哪些入参」直接交出去 ✓，形态与槽位一律来自
    :func:`workflow.video_plan` 的图判据 ✓（与真正注入时用的是**同一套**判据 ✓，不会两套说法 ✓）。

    ⚠️ ``family`` 按**文件名前缀**分 ✓（事实口径：``video_*`` = 官方 ``comfy-org/workflow_templates``
    的**本地推理**版 ✓、``api_*`` = 同仓库的**云端 API** 版 ✓、``MiniMax_H3_Fast_*`` = 本仓自制的
    加速版 ✓ —— 出处见该目录 ``README.md`` ✓）。⚠️ ``runnable_locally=False`` 表示**本服务不做它的注入** ✗
    （那三个用的是 ``MinimaxHailuo03*`` 云端节点 ✓，要 Comfy 账号/额度 ✓，归 ComfyUI 自己管 ✓）。
    """
    import json

    items: list[Dict[str, Any]] = []
    for path in sorted(WORKFLOW_DIR.glob("*.json")):
        plan: Dict[str, Any] = {}
        error: str | None = None
        try:
            plan = wf.video_plan(json.loads(path.read_text(encoding="utf-8")))
        except Exception as err:  # noqa: BLE001 —— 单个模板坏掉不该让整个清单 500 ✗
            error = f"{type(err).__name__}: {err}"
        name = path.name
        items.append({
            "name": name,
            "family": ("official-cloud-api" if name.startswith("api_")
                       else "official-local" if name.startswith("video_")
                       else "voide-fast" if name.startswith("MiniMax_H3_Fast_")
                       else "unknown"),
            "runnable_locally": bool(plan.get("main_id")),
            "form": plan.get("form"),
            "main_class": plan.get("main_class"),
            "prompt_node": plan.get("prompt_owner"),
            "aspect_ratio_node": plan.get("resolution_id"),
            "duration_node": plan.get("duration_id"),
            "first_frame_node": plan.get("first_frame_id"),
            "last_frame_node": plan.get("last_frame_id"),
            "reference_slots": plan.get("ref_slot_names") or [],
            # ⚠️ 键名别叫 ``reference_images`` ✗ —— 任务那些响应里的同名字段是**落地路数（整数）** ✓✗
            #    （一个清单里给「谁接在哪」✓、一个跑单里给「几路落上了」✓，同名字段含义不同最容易读错 ✓）。
            "reference_image_nodes": plan.get("ref_image_ids") or [],
            "error": error,
        })
    return JSONResponse({
        "workflow_dir": str(WORKFLOW_DIR),
        "templates": items,
        "default_t2v": T2V_WORKFLOW,
        "default_i2v": I2V_WORKFLOW,
        "r2v": R2V_WORKFLOW,
        "template_override": TEMPLATE or None,
        "reference_image_limit": REF_IMAGE_LIMIT,
    })


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
    failure: str | None = None
    success: Dict[str, Any] = {}
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
        success = {"outputs": files,
                   "video_url": next((f["url"] for f in files if f["kind"] == "videos"), None)}
    except cc.NodeMissingError as err:
        failure = f"{err}｜装节点：{INSTALL_HINT}"
    except Exception as err:  # noqa: BLE001
        failure = f"{type(err).__name__}: {err}"[:600]
    # ⚠️⚠️ **先卸载、再落终态** ✗（与 `_run_generation` 同一处修法 ✓，2026-09-24 ✓）：
    #    此前「先写 `succeeded` ✓ 再 `finally` 里 `/free`」✗ ⇒ 中间态是「成功了但 `freed_vram=null`」✓✗。
    freed = client.free(unload_models=True, free_memory=True)
    if failure is not None:
        _update(task, status="failed", error_msg=failure, freed_vram=freed)
    else:
        _update(task, status="succeeded", error_msg=None, freed_vram=freed, **success)


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
        # ⚠️ 别只报那两个自制模板 ✗：盘上现在还有**官方那批** ✓（R2V 更是参考素材的**唯一**通路 ✓）
        #    ⇒ 体检要能一眼看出「默认会挑哪个、在不在」✓（``H3_TEMPLATE`` 钉的那个也一并报 ✓）。
        "workflows": {name: (WORKFLOW_DIR / name).is_file()
                      for name in dict.fromkeys([T2V_WORKFLOW, I2V_WORKFLOW, R2V_WORKFLOW, TEMPLATE])
                      if name},
        "templates_dir": str(WORKFLOW_DIR),
        "templates": sorted(p.name for p in WORKFLOW_DIR.glob("*.json")),
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
