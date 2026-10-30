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
``H3_REF_AUDIO_LIMIT``      最多接几路参考音频 ✓            ``3``（模板只声明 1 格 ⇒ 实际落 1 ✓）
``H3_REF_VIDEO_LIMIT``      最多接几路参考视频 ✓            ``3``（模板声明 1 格帧串 + 1 格音轨 ✓）
``H3_REF_VIDEO_NODES``      读视频的节点类候选（逗号分隔）   ``LoadVideo,VHS_LoadVideo``（问引擎取第一个可用的 ✓）
========  ============================================  ==============================

⚠️ 已知限制（**如实列出**，不假装支持 ✓）：
* **参考素材（Ref2VA）只有走 R2V 模板才真生效** ✗：传了参考素材 + 自家那个 FL2VA/I2VA 模板 ⇒ 主节点上
  压根没有 ``ref_images.*`` / ``ref_audios.*`` 槽位 ⇒ **响亮告警**说明被忽略 ✓、且**连上传都不做** ✓
  （别在 ComfyUI 的 ``input`` 目录里留垃圾 ✓）。传参考素材时**没显式指定模板 ⇒ 自动换官方 R2V 模板** ✓。
* 参考素材**图与音都接** ✓（音频是 2026-10-27 补的 ✓）：图落 ``ref_images.ref_image_k`` ✓、
  音落 ``ref_audios.ref_audio_k`` ✓（上游是现建的 ``LoadAudio`` ✓）。===两路各算各的=== ✗✗：
  ``<Picture n>`` 与 ``<Audio n>`` 是**两条独立的下标序列** ✓，混着用会让两边编号都错 ✓。
  ⚠️ 判据一律是**图上真有的端口** ✓（``workflow.py::reference_slot_names`` /
  ``reference_audio_slot_names`` ✓）⇒ 模板加/减槽位都不用改代码 ✓。盘上 R2V 模板：图 3 格（2 格接好、
  1 格空插座）、音 **1** 格（空插座）✓。
* 参考视频 ``ref_videos`` / ``ref_video_audios`` **已接** ✓（2026-10-27 补齐 R2V 第三类素材 ✓✗；
  旧口径写的是「仍不接 ✗（产品侧没有参考视频来源）」✓✗ —— 那是**当时**的事实 ✓，不是判据 ✗：
    模板上**一直就声明着**这两种端口 ✓，缺的只是产品侧这条路 ✓）。
  ⚠️ 它与图/音有一个结构差异 ✓✗：端口类型是 ``IMAGE``（=一串帧 ✓），上游必须是**读视频的加载器** ✓，
  而那个类**不在模板里**（空插座 ✓）⇒ 只能**问引擎**现建 ✓（``_video_loader`` ✓：类名/文件格/
  ``IMAGE``、``AUDIO`` 各在第几槽都从 ``/object_info`` 读 ✓，一个候选都用不了就如实告警且**不上传** ✗）。
  上游候选由 ``H3_REF_VIDEO_NODES`` 覆盖（默认 ``LoadVideo,VHS_LoadVideo`` ✓）。
  ⚠️⚠️ 而且**多半是两跳** ✓✗（2026-10-27 照抄 ``reference/comfy-org`` 后端口径 ✓）：现版核心
  ``LoadVideo`` 只吐 **``VIDEO``** ✗，而 ``ref_videos.ref_video_k`` 要的是 **``IMAGE``** 帧串 ✗
  ⇒ 中间必须再有一个 ``GetVideoComponents``（``VIDEO`` → ``images``/``audio`` ✓，候选由
  ``H3_REF_VIDEO_COMPONENT_NODES`` 覆盖 ✓）。生态节点（``VHS_LoadVideo`` ✓）自带 ``IMAGE``/``AUDIO``
  ⇒ 一跳即可 ✓，故仍优先用一跳 ✓。
* 参考视频的**音轨**会先占掉 ``<Audio j>`` 的编号 ✓✗（后端：每路视频先 append 音轨、
  再 append 视频 ✓；独立音频排在全部视频之后 ✓）⇒ 独立音频编号从 ``声轨数+1`` 起 ✓，
  声轨本身在视频块里声明、且**紧贴它那段视频之前** ✓。
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
#: 参考图最多几路。
#:
#: ⚠️ 默认值**不是**盘上模板的槽位数（那份是 ``ref_images.ref_image_0..2`` ⇒ 3 ✓）✗ ——
#: 取的是**后端节点的真上限** ✓：``nodes_minimax_h3.py`` 里
#: ``io.Autogrow.TemplatePrefix(input=io.Image.Input("ref_image", ...), prefix="ref_image_", min=0, max=9)``
#: ⇒ 9 ✓（旧值 3 会把「模板槽位比盘上那份多」的机器白截断 ✓✗；真正落地几路由模板槽位决定 ✓）。
REF_IMAGE_LIMIT = int(os.environ.get("H3_REF_IMAGE_LIMIT") or "9")
#: 参考音频最多几路（节点 schema ``io.Autogrow.Input("ref_audios", min=0, max=3)`` ⇒ 3 ✓；
#: ⚠️ 盘上 R2V 模板只声明了 **1** 格 ⇒ 真正落地几路仍由**模板槽位数**决定 ✓✗，这个上限只是「别白传」✓）
REF_AUDIO_LIMIT = int(os.environ.get("H3_REF_AUDIO_LIMIT") or "3")
#: 参考**视频**最多几路（节点 schema ``io.Autogrow.Input("ref_videos", min=0, max=3)`` ⇒ 3 ✓；
#: 口径与音频一致：上限只负责「别白传」✓，真正落地几路由**模板槽位数**决定 ✓。
REF_VIDEO_LIMIT = int(os.environ.get("H3_REF_VIDEO_LIMIT") or "3")
#: 读视频的加载器**候选类名**（按顺序问引擎的 ``/object_info`` ✓，谁真装了用谁 ✓）。
#:
#: ⚠️⚠️ 这里**只能给候选、不能写死** ✗✗：ref_videos 的槽位在模板里是**空插座** ✓ ⇒ 上游节点得
#: 现建 ✓，而「读视频」这类节点不是 H3 自带的核心类 ✗ —— 核心有 ``LoadVideo`` ✓、生态里还有
#: ``VHS_LoadVideo`` ✓，widget 名也各不相同（``file`` / ``video`` / ``filename`` ✓）。
#: 写死一个类名 = 换台机器就静默落不了（旧病：见「模板声明了端口但产品侧不接」✓✗）。
#: ⇒ 一律**问引擎**：类在不在、文件格叫什么、``IMAGE``/``AUDIO`` 在第几槽 ✓（见 :func:`_video_loader`）。
REF_VIDEO_LOADERS: tuple[str, ...] = tuple(
    item.strip() for item in (os.environ.get("H3_REF_VIDEO_NODES")
                              or "LoadVideo,VHS_LoadVideo").split(",") if item.strip())
#: 覆盖模板里的 megapixels（画质档位）；空 = 用模板值 ✓
QUALITY = os.environ.get("H3_QUALITY") or ""
#: 参考图缩放档位 ✓ —— 逐字抄后端 ``MiniMaxH3ReferenceToVideo`` 的
#: ``io.Combo.Input("ref_image_size", options=["match", "max"], default="match", ...)`` ✓：
#:
#: * ``match``（默认 ✓）：把参考图缩到**本次出图的像素量** ✓ ⇒ 最快 ✓；
#: * ``max``：缩到**2048 短边**（只缩不放 ✓，docstring：「the best identity fidelity」✓）
#:   ⇒ ⚠️ 每一步采样都带着这些 token ⇒ **可能慢几倍** ✓✗，且要更多显存 ✓。
#:
#: ⚠️ 这是**后端真行为**（不是我们的偏好 ✓）；模板里那格是普通 widget（盘上 R2V 写死 ``"match"`` ✓）
#: ⇒ 想换档必须**显式改图** ✓（见 :func:`_api_widget_owner` ✓），光在提示词里写 ✗ 不管用 ✓。
REF_IMAGE_SIZES: tuple[str, ...] = ("match", "max")
#: 不指定时用后端默认值 ✓（= 模板里的默认 ✓，两边一致 ✓）
REF_IMAGE_SIZE_DEFAULT = "match"
#: 把「视频」拆成「帧串 + 音轨」的那类节点 ✓（候选，按顺序问引擎 ✓）。
#:
#: ⚠️⚠️ 为什么需要**第二跳**（2026-10-27 照抄 ``reference/comfy-org`` 后端口径 ✓✗）：
#: 现版核心 ``LoadVideo``（``comfy_extras/nodes_video.py`` ✓）的输出**只有一个 ``VIDEO``** ★ ——
#: 而 ``ref_videos.ref_video_k`` 要的是 **``IMAGE``**（帧串 ✓，见 ``nodes_minimax_h3.py`` 的
#: ``io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (2-15s)")`` ✓）。
#: ⇒ 两跳：``LoadVideo`` → **``GetVideoComponents``**（``VIDEO`` → ``images``/``audio`` ✓）。
#: 生态节点（``VHS_LoadVideo`` ✓）自带 ``IMAGE``/``AUDIO`` 输出 ⇒ **一跳就够** ✓，仍优先用一跳 ✓。
REF_VIDEO_COMPONENTS: tuple[str, ...] = tuple(
    item.strip() for item in (os.environ.get("H3_REF_VIDEO_COMPONENT_NODES")
                              or "GetVideoComponents").split(",") if item.strip())
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


def _landed_upload(result: Dict[str, Any], filename: str) -> str:
    """把上传回执拼成**节点 widget 要的名字** ✓ ⇒ ``subfolder/name`` 或 ``name`` ✓。

    ⚠️ 上传可能被 ComfyUI 改名 ✓（重名加后缀、``subfolder`` 非空 ✓）⇒ **一律用回执里的名字** ✗✗，
    拿本地文件名当真名会指向一个不存在的文件 ✓（症状：提交 400 或加载到上一单的素材 ✓✗）。
    回执缺 ``name`` ⇒ 退回本地名 ✓（至少不让这里变成第二个未知数 ✓）。
    """
    name = str(result.get("name") or filename)
    subfolder = str(result.get("subfolder") or "")
    return f"{subfolder}/{name}" if subfolder else name


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


def _reference_audios(body: Dict[str, Any]) -> list[str]:
    """参考**音频**列表 ✓：``referenceAudioUrls``（后端协议口径 ✓）、``reference_audio_urls`` ✓ 与
    ``reference_audio``（H3 契约口径 ✓）**三套命名都收** ✓，按出现顺序去重 ✓。

    ⚠️ 只认一套 ⇒ 别的被当成「0 路」⇒ 声线参考**静默不生效** ✓✗。这不是假设 ✓：修之前
    ``_contract_prompt`` 只数 ``reference_audio`` ✓，而后端 ``routers/videos.py`` 发的是
    ``referenceAudioUrls`` ✓ ⇒ 契约层一直按「0 路音频」算 ✓✗（连 ``<Audio 1>`` 的声明都不追加 ✓）。
    """
    out: list[str] = []
    for key in ("referenceAudioUrls", "reference_audio_urls", "reference_audio"):
        for item in _media_texts(body.get(key)):
            if item not in out:
                out.append(item)
    return out


def _reference_videos(body: Dict[str, Any]) -> list[str]:
    """参考**视频**列表 ✓：``referenceVideoUrls``（后端协议口径 ✓）、``reference_video_urls`` ✓ 与
    ``reference_video``（H3 契约口径 ✓）**三套命名都收** ✓，按出现顺序去重 ✓。

    ⚠️ 与图/音**各算各的** ✓✗：这是**第三条**下标序列（提示词里 ``<Video n>`` ✓），
    混进图或音那条 ⇒ 三边编号全错 ✓（同一形状的坑在音频那半边刚踩过一次 ✓）。
    """
    out: list[str] = []
    for key in ("referenceVideoUrls", "reference_video_urls", "reference_video"):
        for item in _media_texts(body.get(key)):
            if item not in out:
                out.append(item)
    return out


def _reference_image_size(body: Dict[str, Any]) -> str:
    """参考图缩放档位 ✓：``referenceImageSize``（协议口径 ✓）/ ``reference_image_size`` / ``ref_image_size``。

    取值只认后端那两个（:data:`REF_IMAGE_SIZES` ✓），**大小写不敏感**（``MAX`` ⇒ ``max`` ✓）；
    空 ⇒ :data:`REF_IMAGE_SIZE_DEFAULT` ✓。**非法值不在这里报错 ✗** —— 交给调用方告警后回落 ✓
    （一个可选画质档位写错，不该把整单判死 ✗）。
    """
    for key in ("referenceImageSize", "reference_image_size", "ref_image_size"):
        text = str(body.get(key) or "").strip().lower()
        if text:
            return text
    return REF_IMAGE_SIZE_DEFAULT


def _api_widget_owner(api: Dict[str, Any], node_id: Any, field: str) -> tuple[str, str] | None:
    """API 图里某个 widget **该往哪儿写** ✓ ⇒ ``(节点id, 字段)``；这图里没有该 widget ⇒ ``None`` ✗。

    ⚠️ 为什么在 **API 图**上判、而不是在 UI 图上判 ✗✗：UI 图的 widget 值只是 ``widgets_values``
    **一串没有名字的数组** ✓，字段名要靠 ``/object_info`` 的 ``INPUT_TYPES()`` **按序**对 ✓
    （见 :func:`workflow.ui_to_api` ✓）—— 也就是说「这个模板有没有 ``ref_image_size``」这件事，
    **只有转完之后才知道** ✓。转完的 API 图里它就是一个**有名字**的输入 ✓ ⇒ 在这判既准又不用额外问引擎 ✓。

    两种合法落点：
    * **字面量**（模板里写死 ``"match"`` ✓）⇒ 写本节点 ✓；
    * **被上游牵着线**（接到 ``PrimitiveString`` ✓）⇒ 写**上游那个**节点 ✓ —— 往本节点写会被连线覆盖 ✗
      （ComfyUI 里连线优先 ✓，写了也不生效，最难查的那种 ✓✗）。
    """
    node = api.get(str(node_id))
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs") or {}
    if field not in inputs:
        return None
    value = inputs[field]
    if isinstance(value, list) and len(value) == 2:  # 连线 ⇒ 找一个唯一的上游原始值节点 ✓
        upstream = api.get(str(value[0])) or {}
        if str(upstream.get("class_type") or "").startswith("PrimitiveString"):
            return (str(value[0]), "value")
        return None
    return (str(node_id), field)


def _wire_reference_image_size(task: Dict[str, Any], api: Dict[str, Any], plan: Dict[str, Any],
                               body: Dict[str, Any], landed: int) -> str | None:
    """把参考图缩放档位落到图上 ✓ ⇒ **实际生效的档位**（落不下去 ⇒ ``None`` ✗）。

    ⚠️ 三件必须如实说的事 ✓：
    ① **没有参考图就不用改** ✗（``landed == 0`` ⇒ 直接返回 ``None`` ✓ —— 那格对纯文生毫无意义 ✓）；
    ② 模板**没有** ``ref_image_size`` 这格 ⇒ 请求了别的档位就**告警**（不假装改成功 ✗）；请求的就是
       默认值 ⇒ 静默放过 ✓（引擎默认就是它 ✓，报 ``None`` 让调用方知道「没显式设过」✓）；
    ③ 值不在后端选项里 ⇒ 告警 + 回落默认值 ✓（不把整单判死 ✗）。
    """
    if not landed:
        return None
    wanted = _reference_image_size(body)
    if wanted not in REF_IMAGE_SIZES:
        _warn(task, f"``referenceImageSize``={wanted!r} 不是后端认的档位 ✗"
                    f"（只有 {' / '.join(REF_IMAGE_SIZES)} ✓）⇒ 本次改用 {REF_IMAGE_SIZE_DEFAULT} ✓")
        wanted = REF_IMAGE_SIZE_DEFAULT
    owner = _api_widget_owner(api, plan.get("main_id"), "ref_image_size")
    if owner and wf.set_node_input(api, owner[0], owner[1], wanted):
        return wanted
    if wanted != REF_IMAGE_SIZE_DEFAULT:
        _warn(task, f"模板主节点上没有 ``ref_image_size`` 这一格 ✗ ⇒ 请求的 **{wanted}** 落不下去 ✓"
                    f"（参考图沿用引擎默认缩放 ✓；要这个档位得换一份带该输入的模板 ✓）")
    return None


def _audio_filename(index: int, url: str) -> str:
    """给要上传到 ComfyUI ``input`` 目录的音频起名 ⇒ ``ref_audio_<k>.<ext>`` ✓。

    ⚠️ 扩展名**必须能过 ``LoadAudio`` 的 COMBO** ✗✗：那个 ``audio`` 是 COMBO ✓，选项表由
    ``folder_paths.filter_files_content_types(os.listdir(input), ["audio","video"])`` **按
    mimetypes 现算** ✓（出处 ``comfy_extras/nodes_audio.py::LoadAudio.define_schema`` ✓）
    ⇒ 存成 ``.bin`` 就是「上传成功、节点里根本选不出来」，提交时 400 且报「文件名不在列表里」✓✗。
    解法：**拿 URL 后缀去问 mimetypes** ✓（与 ComfyUI 同一判据 ✓），认成 ``audio/*`` 就保留 ✓；
    否则（``data:`` 内联、无后缀的地址）回落 ``.wav`` ✓ —— 解码那头 ``load()`` 走的是 av，
    **按内容嗅探** ✓（同文件 344 行 ``af.decode`` ✓）⇒ 容器实际是 mp3 也能解 ✓，扩展名只负责「能选出来」✓。
    """
    import mimetypes

    path = str(url or "").split("?")[0].split("#")[0]
    suffix = Path(path).suffix.lower()
    guess = mimetypes.guess_type(f"probe{suffix}", strict=False)[0] if suffix else None
    if not str(guess or "").startswith("audio/"):
        suffix = ".wav"
    return f"ref_audio_{index + 1}{suffix}"


def _video_filename(index: int, url: str) -> str:
    """给要上传到 ComfyUI ``input`` 目录的参考视频起名 ⇒ ``ref_video_<k>.<ext>`` ✓。

    判据与 :func:`_audio_filename` **逐字同形** ✓：拿 URL 后缀去问 ``mimetypes`` ✓，
    认成 ``video/*`` 才保留 ✓；否则（``data:`` 内联、无后缀）回落 ``.mp4`` ✓。
    ⚠️ 别共用 ``.wav`` / ``.png`` ✗：读视频的加载器那格 COMBO 只列**视频**文件 ✓
    ⇒ 后缀不对 = 「传上去了但节点里选不出来」⇒ 提交 400 ✓✗。
    """
    import mimetypes

    path = str(url or "").split("?")[0].split("#")[0]
    suffix = Path(path).suffix.lower()
    guess = mimetypes.guess_type(f"probe{suffix}", strict=False)[0] if suffix else None
    if not str(guess or "").startswith("video/"):
        suffix = ".mp4"
    return f"ref_video_{index + 1}{suffix}"


def _media_texts(raw: Any) -> list[str]:
    """把媒体入参读成非空字符串列表 ✓（列表 ✓ / 单个字符串 ✓ / JSON 字符串列表 ✓ 都认 ✓）。"""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        parsed = _as_list(text)
        return [str(item).strip() for item in parsed if str(item).strip()] or [text]
    return [str(item).strip() for item in _as_list(raw) if str(item).strip()]


def _pick_template(task: Dict[str, Any], refs: list[str], audios: list[str] | None = None,
                   videos: list[str] | None = None) -> str:
    """选模板 ✓：**显式指定 > 有参考素材（图/音/**视频**）⇒ R2V > 有首帧 ⇒ I2V > 纯文生 ⇒ T2V** ✓。

    显式指定来自 body ``template`` ✓（优先）或 env ``H3_TEMPLATE`` ✓。

    ⚠️ 有参考素材**必须**换 R2V 模板 ✗✗：A 批那两个模板的主节点是 ``MiniMaxH3ImageToVideo`` ✓，
    **没有参考图/参考音频槽位** ✗ ⇒ 参考素材只能被丢掉 ✓（旧行为：响告警 + 忽略 ✓）。
    ⚠️⚠️ **只有参考音频也算参考素材** ✓✗（2026-10-27 改）：旧写法只数图 ``refs`` ✓ ⇒ 「只给声线样本、
    不给图」的请求被塞进 I2V/T2V 模板 ✓ ⇒ 主节点连 ``ref_audios.*`` 都没有 ⇒ 音频被静默丢掉 ✓✗
    （那种请求的产品含义很明确：**学这个音色说话** ✓，不是「没有参考」✗）。
    ⚠️ **参考视频同理** ✓（2026-10-27 一并算）：只有参考视频的请求也必须换 R2V ✓✗。
    ⚠️ R2V 模板不在盘上 ⇒ **如实告警 + 退回** ✓（不假装支持 ✓；也不因为「可选素材」把整个任务判死 ✓）。
    """
    audios = audios or []
    videos = videos or []
    body = task.get("body") or {}
    explicit = str(body.get("template") or TEMPLATE or "").strip()
    if explicit:
        return explicit
    if refs or audios or videos:
        if (WORKFLOW_DIR / R2V_WORKFLOW).exists():
            return R2V_WORKFLOW
        _warn(task, f"请求带了 {len(refs)} 路参考图 / {len(audios)} 路参考音频 / "
                    f"{len(videos)} 路参考视频，但 R2V 模板不在盘上"
                    f"（{R2V_WORKFLOW} ✗）⇒ 本次**忽略它们** ✗"
                    f"（要用参考素材请把模板放回 {WORKFLOW_DIR} ✓）")
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


def _wire_reference_audios(task: Dict[str, Any], api: Dict[str, Any], plan: Dict[str, Any],
                           uploaded: list[str]) -> int:
    """把上传好的参考音频落到主节点的 ``ref_audios.ref_audio_k`` ✓ ⇒ **落地路数** ✓。

    与图那条**同一形状** ✓✗，只有两处不同：
    * 上游节点类名是 ``LoadAudio`` ✓、入参字段是 ``audio`` ✓（就是 ``input`` 目录里的文件名 ✓）；
    * 盘上 R2V 模板声明了 **1 格音频槽位但没有上游节点** ✓（图那边是模板自带 3 个 ``LoadImage`` ✓）
      ⇒ 绝大多数时候走「空槽 ⇒ **现建一个** ``LoadAudio``」那条 ✓。

    ⚠️ 多出来的槽位**必须摘连线** ✗✗：空着不收钱 ✓，但若模板哪天自带 ``LoadAudio`` 且不摘，
    它指着的是上一单的文件名 ✓ ⇒ 空槽会去加载别人的声线 ✓✗（比图那边更隐蔽：图是 400 报错，
    音是**照常出片、只是声音不是这一个** ✓✗）。没别的引用的上游节点连同删掉 ✓。
    """
    main_id = plan.get("main_id")
    slots = list(plan.get("ref_audio_slot_names") or [])
    wired = list(plan.get("ref_audio_ids") or [])
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
            wf.set_node_input(api, existing, "audio", uploaded[index])
            landed += 1
            continue
        new_id = f"voide_ref_audio_{index + 1}"
        api[new_id] = {
            "class_type": "LoadAudio",
            "inputs": {"audio": uploaded[index]},
            "_meta": {"title": f"REF AUDIO {index + 1} (voide)"},
        }
        wf.set_node_input(api, main_id, slot_name, [new_id, 0])
        landed += 1
    if landed < len(uploaded):
        _warn(task, f"参考音频只落地了 {landed} 路（模板槽位 {len(slots)} 个，请求 {len(uploaded)} 路 ✗）"
                    f"⇒ 多出来的被丢掉 ✗（要更多音频槽位得改模板 ✓）")
    return landed


#: 读视频加载器的探测结果缓存 ✓（键＝类名 ✓）—— 一次进程只问引擎一次 ✓，免得每单都多打一轮。
#:
#: ⚠️ **只缓存「探到了」✓，不缓存「没探到」** ✗✗（2026-10-27 改）：本服务是长驻进程 ✓，
#: 而引擎那边**装个节点、重启一下**就能从「没有」变「有」✓ —— 把否定结论也缓存住 ⇒ 用户装完节点
#: 必须连我们这层一起重启才生效 ✓✗（他会以为「装了没用」✓）。否定答案每次现问 ✓（一单多 2~3 个
#: ``GET /object_info/<类>``，只在真带参考视频时才发生 ✓）。
_VIDEO_LOADER_CACHE: Dict[str, Dict[str, Any]] = {}
#: 「视频 → 帧串/音轨」那个拆解节点的探测结果缓存 ✓（键＝类名 ✓；同样只存肯定 ✓）
_VIDEO_SPLITTER_CACHE: Dict[str, Dict[str, Any]] = {}
#: 文件格**可能**叫什么 ✓（候选顺序＝优先级 ✓）：核心 ``LoadVideo`` 是 ``file`` ✓，
#: 生态里还有 ``video`` / ``filename`` 之类的变体 ✓。
_VIDEO_FILE_FIELDS = ("file", "video", "filename", "path")


def _node_spec(client: Any, class_type: str) -> Dict[str, Any]:
    """问引擎要一个类的 schema ✓（探能力失败**不当成错** ✓ —— 拿空 spec 继续试下一个候选 ✓）。"""
    try:
        info = client.object_info(class_type) or {}
    except Exception:  # noqa: BLE001 —— 探能力失败不该把整单判死 ✓
        return {}
    return info.get(class_type) or {}


def _node_inputs(spec: Dict[str, Any]) -> Dict[str, Any]:
    """``required`` + ``optional`` 合起来看 ✓（读视频那格未必是必填 ✓）。"""
    inputs = spec.get("input") or {}
    merged: Dict[str, Any] = {}
    merged.update(inputs.get("required") or {})
    merged.update(inputs.get("optional") or {})
    return merged


def _node_outputs(spec: Dict[str, Any]) -> list[str]:
    return [str(item) for item in spec.get("output") or []]


def _pick_file_field(inputs: Dict[str, Any]) -> str | None:
    """哪个入参是「要读的文件名」✓（先按常见名 ✓，再退到第一个 ``COMBO``/``STRING`` ✓）。"""
    field = next((name for name in _VIDEO_FILE_FIELDS if name in inputs), None)
    if field:
        return field
    for name, spec in inputs.items():
        kind = str((spec or [None])[0]) if isinstance(spec, (list, tuple)) else ""
        if kind in ("COMBO", "STRING"):
            return str(name)
    return None


def _video_splitter(client: Any) -> Dict[str, Any] | None:
    """``VIDEO`` → ``IMAGE``(+``AUDIO``) 的那个拆解节点 ✓ ⇒ ``{"class", "video_field", "image_slot", "audio_slot"}``。

    ⚠️ 为什么需要它 ✗✗（**照抄后端** ✓，不是我们发明的 ✓）：现版核心 ``LoadVideo`` 的输出只有一个
    ``VIDEO`` ★（``comfy_extras/nodes_video.py``：``inputs=[io.Combo.Input("file", ..., upload=io.UploadType.video)]``、
    ``outputs=[io.Video.Output()]`` ✓），而 ``ref_videos.ref_video_k`` 要的是 **``IMAGE``** 帧串
    （``io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (2-15s)")`` ✓）
    ⇒ 必须经 ``GetVideoComponents``（``inputs=[io.Video.Input("video")]``、
    ``outputs=[images(IMAGE), audio(AUDIO), fps, bit_depth, color_space]`` ✓）拆一道 ✓。
    """
    for candidate in REF_VIDEO_COMPONENTS:
        cached = _VIDEO_SPLITTER_CACHE.get(candidate)
        if cached:
            return cached
        spec = _node_spec(client, candidate)
        inputs = _node_inputs(spec)
        outputs = _node_outputs(spec)
        video_field = next((name for name, item in inputs.items()
                            if str((item or [None])[0]).upper() == "VIDEO"), None)
        if video_field and "IMAGE" in outputs:
            found = {
                "class": candidate,
                "video_field": str(video_field),
                "image_slot": outputs.index("IMAGE"),
                "audio_slot": outputs.index("AUDIO") if "AUDIO" in outputs else None,
            }
            _VIDEO_SPLITTER_CACHE[candidate] = found
            return found
    return None


def _video_loader(client: Any) -> tuple[Dict[str, Any] | None, str]:
    """问引擎「哪个类能读视频、文件格叫什么、``IMAGE``/``AUDIO`` 在第几槽」✓ ⇒ ``(loader, 原因)``。

    返回 ``{"class", "field", "image_slot", "audio_slot", "video_slot", "components"}`` ✓：

    * ``video_slot`` 是**加载器自己**吐 ``VIDEO`` 的槽 ✓（``None`` = 它直接吐帧 ✓，不用第二跳 ✓）；
    * ``components`` 是第二跳的拆解节点 ✓（``None`` = 加载器自带 ``IMAGE`` 输出 ✓）；
    * ``image_slot`` / ``audio_slot`` 一律按**接主节点的那个节点**算 ✓ —— 有两跳时指拆解节点的槽 ✓✗
      （这是最容易写错的一处：照加载器的槽接 ⇒ 帧串接上一条 ``VIDEO`` 线 ✓，ComfyUI 提交时类型不符 ✓✗）。

    一个候选都用不了 ⇒ ``(None, 原因)`` ✗（**不猜**、也不临时造一个类名 ✗）。
    原因字符串必须说清是**哪一步**缺的 ✓✗（「引擎里没 LoadVideo」与「LoadVideo 在、但拆解节点没有」
    是两种完全不同的处境：前者要装读视频节点 ✓，后者要装 ``GetVideoComponents`` ✓）。

    ⚠️⚠️ 为什么必须**问引擎**而不是写死 ✗✗：``ref_videos.ref_video_0`` 在盘上模板里是**空插座** ✓
    ⇒ 上游节点要现建 ✓，而「读视频」不是 H3 自带类 ✗（核心 ``LoadVideo`` ✓、生态 ``VHS_LoadVideo`` ✓，
    widget 名还各不相同 ✓）。写死一个 ⇒ 换台机器就「上传成功、图里那个节点提交 400」✓✗，
    而且**报错落在 ComfyUI 侧、看不出是我们的假设错** ✓✗（本仓反复记的那类静默错 ✓）。
    """
    notes: list[str] = []
    for candidate in REF_VIDEO_LOADERS:
        cached = _VIDEO_LOADER_CACHE.get(candidate)
        if cached:
            return cached, ""
        spec = _node_spec(client, candidate)
        if not spec:
            notes.append(f"{candidate}：引擎里没有这个类")
            continue
        inputs = _node_inputs(spec)
        outputs = _node_outputs(spec)
        field = _pick_file_field(inputs)
        if not field:
            notes.append(f"{candidate}：类在，但找不到「要读的文件」那一格")
            continue
        if "IMAGE" in outputs:  # 生态节点：一跳就够 ✓（VHS_LoadVideo 这类 ✓）
            found = {
                "class": candidate, "field": field,
                "image_slot": outputs.index("IMAGE"),
                "audio_slot": outputs.index("AUDIO") if "AUDIO" in outputs else None,
                "video_slot": None, "components": None,
            }
            _VIDEO_LOADER_CACHE[candidate] = found
            return found, ""
        if "VIDEO" in outputs:  # 核心 LoadVideo：要第二跳 ✓
            splitter = _video_splitter(client)
            if splitter:
                found = {
                    "class": candidate, "field": field,
                    "image_slot": splitter["image_slot"],
                    "audio_slot": splitter["audio_slot"],
                    "video_slot": outputs.index("VIDEO"),
                    "components": splitter,
                }
                _VIDEO_LOADER_CACHE[candidate] = found
                return found, ""
            notes.append(f"{candidate}：只吐 {outputs}（没有 IMAGE ✗），而把视频拆成帧串的节点"
                         f"（候选 {'/'.join(REF_VIDEO_COMPONENTS)}）引擎里没有")
            continue
        notes.append(f"{candidate}：输出是 {outputs}，既没有 IMAGE 也没有 VIDEO")
    return None, "；".join(notes) or f"候选表是空的（H3_REF_VIDEO_NODES={','.join(REF_VIDEO_LOADERS)}）"


def _node_wired_from(api: Dict[str, Any], class_type: str, source: list[Any]) -> str | None:
    """图里有没有某个类节点**正接着** ``source`` ✓ ⇒ 节点 id（没有 ⇒ ``None`` ✓）。

    用在一路视频的第二跳上 ✓：重跑/复用同一张图时不该再建一个 ``GetVideoComponents`` ✗
    （两个 ⇒ 同一段视频解码两遍 ✓✗，还白占一份显存 ✓）。
    """
    for node_id, node in api.items():
        if not isinstance(node, dict) or node.get("class_type") != class_type:
            continue
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, list) and len(value) == 2 and list(value) == list(source):
                return str(node_id)
    return None


def _video_source(api: Dict[str, Any], index: int, filename: str, loader: Dict[str, Any],
                  existing: Any = None) -> tuple[list[Any], list[Any] | None]:
    """建/复用**一路**参考视频的上游 ✓ ⇒ ``(帧串引用, 音轨引用或 None)`` ✓。

    两种形状（由 :func:`_video_loader` 问出来的 ``components`` 决定 ✓）：
    * **一跳**（``VHS_LoadVideo`` 这类 ✓）：加载器自己就有 ``IMAGE``/``AUDIO`` 输出 ⇒ 两个引用同一个节点 ✓；
    * **两跳**（核心 ``LoadVideo`` ✓ → ``GetVideoComponents`` ✓）：加载器只吐 ``VIDEO`` ⇒ 再建一个拆解
      节点（``video`` 入参接加载器的 ``VIDEO`` 槽 ✓），帧串/音轨都从**拆解节点**取 ✓✗。

    ⚠️ 一路视频只解码一遍 ✓（两跳也只建一个拆解节点 ✓）；引用一律 ``[节点id, 槽位]`` ✓。
    """
    node_class = str(loader["class"])
    node_id = str(existing) if existing is not None else f"voide_ref_video_{index + 1}"
    if existing is None:
        api[node_id] = {
            "class_type": node_class,
            "inputs": {str(loader["field"]): filename},
            "_meta": {"title": f"REF VIDEO {index + 1} (voide)"},
        }
    else:
        wf.set_node_input(api, node_id, str(loader["field"]), filename)
    components = loader.get("components")
    if not components:
        return ([node_id, int(loader["image_slot"])],
                [node_id, int(loader["audio_slot"])] if loader.get("audio_slot") is not None else None)
    source = [node_id, int(loader["video_slot"])]
    splitter_id = (_node_wired_from(api, str(components["class"]), source)
                   or f"voide_ref_video_split_{index + 1}")
    if splitter_id not in api:
        api[splitter_id] = {
            "class_type": str(components["class"]),
            "inputs": {str(components["video_field"]): source},
            "_meta": {"title": f"REF VIDEO {index + 1} → 帧串/音轨 (voide)"},
        }
    return ([splitter_id, int(components["image_slot"])],
            [splitter_id, int(components["audio_slot"])] if components.get("audio_slot") is not None else None)


def _drop_video_source(api: Dict[str, Any], node_id: Any) -> None:
    """删掉一路参考视频的上游 ✓（连带它后面那个拆解节点 ✓ —— 只删加载器会让图里留下一条断线 ✗）。"""
    target = str(node_id)
    for candidate_id, node in list(api.items()):
        if not isinstance(node, dict):
            continue
        if any(isinstance(value, list) and value and str(value[0]) == target
               for value in (node.get("inputs") or {}).values()):
            api.pop(str(candidate_id), None)
    api.pop(target, None)


def _wire_reference_videos(task: Dict[str, Any], api: Dict[str, Any], plan: Dict[str, Any],
                           landed_files: list[str], loader: Dict[str, Any]) -> int:
    """把上传好的参考视频落到主节点 ⇒ **落地路数** ✓。

    一路视频要接**两根线** ✓✗（这是它和图的唯一结构差异）：
    ``ref_videos.ref_video_k`` ← 帧串（``IMAGE`` ✓）、``ref_video_audios.ref_video_audio_k`` ← 它的音轨
    （``AUDIO`` ✓）。⚠️ 两根线**不一定同源** ✗✗：核心 ``LoadVideo`` → ``GetVideoComponents`` 两跳时，
    帧串是拆解节点的 ``images`` 槽、音轨是它的 ``audio`` 槽 ✓（见 :func:`_video_source` ✓）。

    ⚠️ 三件必须如实说的事 ✓：
    ① 上游没有 ``AUDIO`` 输出（拆解节点也没有 ✓）⇒ 音轨槽**空着**并告警 ✓（不假装接上了 ✗）；
    ② 模板没有 ``ref_video_audios.*`` 槽 ⇒ 同上 ✓；
    ③ 多出来的槽位摘连线、无引用的上游连同删掉 ✓（同图/音那条纪律 ✓）。
    """
    main_id = plan.get("main_id")
    slots = list(plan.get("ref_video_slot_names") or [])
    audio_slots = list(plan.get("ref_video_audio_slot_names") or [])
    wired = list(plan.get("ref_video_ids") or [])
    landed = 0
    for index, slot_name in enumerate(slots):
        existing = wired[index] if index < len(wired) else None
        if index >= len(landed_files):
            if existing is not None:
                (api.get(str(main_id)) or {}).get("inputs", {}).pop(slot_name, None)
                audio_name = audio_slots[index] if index < len(audio_slots) else None
                if audio_name:
                    (api.get(str(main_id)) or {}).get("inputs", {}).pop(audio_name, None)
                _drop_video_source(api, existing)
            continue
        image_ref, audio_ref = _video_source(api, index, landed_files[index], loader, existing)
        wf.set_node_input(api, main_id, slot_name, image_ref)
        landed += 1
        # 音轨：**下标对齐** ✓（后端就是 ``ref_video_audios["ref_video_audio_" + k]`` ✓，见
        # ``nodes_minimax_h3.py`` 的 ``soundtrack = ref_video_audios.get(...)`` ✓）—— 第 k 路视频的音轨落第 k 格 ✓。
        audio_slot_name = audio_slots[index] if index < len(audio_slots) else None
        if audio_slot_name and audio_ref is not None:
            wf.set_node_input(api, main_id, audio_slot_name, audio_ref)
        elif audio_slot_name:
            _warn(task, f"参考视频第 {index + 1} 路的**音轨没接** ✗：上游 {loader['class']}"
                        + (f" → {loader['components']['class']}" if loader.get("components") else "")
                        + f" 没有 ``AUDIO`` 输出 ⇒ ``{audio_slot_name}`` 空着 ✓"
                        f"（想连音轨得换一个能拆出音轨的读视频节点 ✓）")
    if audio_slots and len(audio_slots) < len(slots):
        _warn(task, f"参考视频帧串落了 {len(slots)} 格，但模板只有 {len(audio_slots)} 格音轨槽 "
                    f"⇒ 后面那几路的**音轨被丢掉** ✗")
    if landed < len(landed_files):
        _warn(task, f"参考视频只落地了 {landed} 路（模板槽位 {len(slots)} 个，请求 {len(landed_files)} 路 ✗）"
                    f"⇒ 多出来的被丢掉 ✗（要更多视频槽位得改模板 ✓）")
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
                     pictures: int | None = None, audios: int | None = None,
                     videos: int | None = None, video_soundtracks: int = 0) -> str:
    """提示词过 **H3 prompt 契约** ✓（提交前 ✓）：文本写法 → 标签、**核编号**、**该追加的声明必须追加** ✗。

    ⭐ 钉的就是那条实测坑 ✗✗：**只写 ``<Audio 1>`` 绑定句不算声明** ✓ —— 照标签跳过的话，
    自动声明被吞、模型**不复用参考音频**（实测相关性≈0 ✓✗）。所以这一段必须在**提交前**跑 ✓。

    ⚠️ ``pictures``：**实际落地**的参考图路数 ✓ 优先于 body 里的 ``subject_reference`` ✗ ——
    参考图有两种命名来源 ✓（``referenceImageUrls`` 与 ``subject_reference`` ✓），只数后者会在
    前者形态下被判成 0 路 ✓✗ ⇒ ``<Picture n>`` 的编号与自动声明全按空的算 ✓✗（静默错 ✓）。
    所以默认值只在**调用方没给**时才用 ✓。

    ⚠️ ``audios``：口径与 ``pictures`` **逐字相同** ✓✗ —— 给「实际落地」的音频路数 ✓
    （``min(请求路数, 模板槽位数)`` ✓）。修之前这里**硬按 body 的 ``reference_audio`` 数** ✗ ⇒
    两处会错：① 后端发 ``referenceAudioUrls`` ⇒ 恒判 0 路 ✓✗；② 模板没有音频槽位、音频一路都没落地
    ⇒ 照样按请求路数声明 ✓✗（声明了不存在的素材 ⇒ 模型去「复用」一段根本不存在的音色 ✓）。

    ⚠️ 编号错 ⇒ **当场拒** ✗（``ConditioningError`` 冒上去 ⇒ 任务带可行动文案失败 ✓）；
    ⚠️ 契约层没装上 ⇒ **如实告警「没跑」** ✗ 而不是当通过 ✓✗。
    """
    if prompt_contract is None:
        _warn(task, f"H3 提示词契约层没装上（{CONTRACT_ERROR}）⇒ **提交前那道关没跑** ✗（没查不是通过 ✓✗）")
        return prompt_text
    cond_mod = getattr(prompt_contract, "cond", None)
    if audios is None:
        audios = len(_reference_audios(body))
    if pictures is None:
        pictures = len(_reference_images(body))
    # ⚠️ ``videos`` 与那两个**逐字同口径** ✓✗：给「实际落地」的参考视频路数 ✓ ——
    #    按 body 请求数算的话，槽位不够时会声明出**不存在的** ``<Video k>`` ✓✗
    #    （音频那半边正是这么错的，见上面那段 ✓）。
    if videos is None:
        videos = len(_reference_videos(body))
    # ⚠️ 声轨数必须**夹在**视频路数里 ✓✗（调用方算出来是「真接上音轨的视频」✓；这里再夹一道，
    #    免得 ``<Audio k>`` 的编号跑出 ``<Video k>`` 的范围 ✓）
    soundtracks = max(0, min(int(video_soundtracks or 0), int(videos)))
    # 参考音频一律是**音色参考** ✓ ⇒ 协议关系 ``reference`` ✓（⚠️ 别用 ``fully_copy`` ✗ ——
    # 那是「逐字复制内容」的意思 ✓，会把参考音频的台词念出来 ✓✗）
    # ⚠️ 编号要**从声轨后面起** ✓✗：后端把「视频声轨」与「独立音频」放进同一个 ``<Audio j>`` 序列 ✓
    # （声轨在前 ✓），照 1 起编号 ⇒ 与声轨撞号 ✓✗（模型把音乐当音色参考 ✓）。
    refs = [cond_mod.AudioReference(index=soundtracks + index + 1, relation="reference")
            for index in range(audios)] if cond_mod is not None else []
    settings = body.get("settings") if isinstance(body.get("settings"), dict) else {}
    result = prompt_contract.build_prompt(
        prompt_text, pictures=pictures, audios=audios, videos=videos, audio_refs=refs,
        # ``videos`` 与 ``video_count`` **必须同值** ✓✗：前者是「编号校验的可用数」✓、
        # 后者是「声明要不要追加」✓ ⇒ 只给一个 ⇒ 要么核不过、要么声明被吞 ✓✗（两条都静默）。
        video_count=videos,
        video_soundtracks=soundtracks,
        global_prompt=str((settings or {}).get("global_prompt") or ""),
    )
    _update(task, prompt_contract=result["plan"], prompt_markup=result["markup"],
            prompt_prefixed=result["prefixed"],
            # 声轨占掉的 ``<Audio j>`` 编号也报出来 ✓✗：调用方看到 ``<Audio 1>`` 却以为那是自己的
            # 独立音频时，一眼就能靠这一格分辨 ✓。
            reference_video_soundtracks=soundtracks)
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
    # ⚠️ 音频**同一套纪律** ✓✗：先按节点 schema 的 max 截断 ✓（``REF_AUDIO_LIMIT`` = 3 ✓），
    #    真正落地几路仍由**模板槽位数**决定 ✓（盘上 R2V 只声明 1 格 ✓ ⇒ 第 2、3 路会被丢掉并告警 ✓）。
    all_audios = _reference_audios(body)
    audios = all_audios[:REF_AUDIO_LIMIT]
    if len(all_audios) > len(audios):
        _warn(task, f"参考音频给多了（{len(all_audios)} 路 > 上限 {REF_AUDIO_LIMIT} 路 ✗）"
                    f"⇒ 只落地前 {len(audios)} 路 ✗（H3 节点 schema 的 ``ref_audios`` max=3 ✓；"
                    f"而盘上 R2V 模板只声明了 1 格 ⇒ 实际只落 1 路 ✓）")
    # ⚠️ 参考视频**同一套纪律** ✓✗（第三条独立序列 ⇒ ``<Video n>`` ✓）
    all_videos = _reference_videos(body)
    videos = all_videos[:REF_VIDEO_LIMIT]
    if len(all_videos) > len(videos):
        _warn(task, f"参考视频给多了（{len(all_videos)} 路 > 上限 {REF_VIDEO_LIMIT} 路 ✗）"
                    f"⇒ 只落地前 {len(videos)} 路 ✗（H3 节点 schema 的 ``ref_videos`` max=3 ✓）")
    template_name = _pick_template(task, refs, audios, videos)
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

    # ⚠️ 音频槽位**同样在上传之前确定** ✓✗（同一个道理：落不下去的东西一个都别传 ✓）。
    audio_slot_names = list(plan["ref_audio_slot_names"])
    if audios and not audio_slot_names:
        _warn(task, f"模板（{template_name}）的主节点**没有参考音频槽位** ⇒ {len(audios)} 路参考音频被忽略 ✗"
                    f"（**一个都没上传** ✓；要用音色参考请换 R2V 模板：{R2V_WORKFLOW} ✓，"
                    f"或 body ``template`` 显式指定 ✓）")
    # ⚠️⚠️ 槽位比请求少 ⇒ **在传之前就截断** ✓✗（同一条纪律：落不下去的那几路一个都别传 ✓）——
    #    盘上 R2V 模板只声明 **1 格**音频槽位 ✓ ⇒ 请求 2 路时第 2 路**本来就落不下** ✓；
    #    旧写法照样把它传上去 ✓ ⇒ ComfyUI 的 ``input`` 目录里留一个没人引用的音频 ✓✗
    #    （而且它还是别人的声线样本 —— 白占盘、还误导事后排查 ✓）。告警必须**响亮** ✓：
    #    少落一路声线 = 成片说话的音色跟给的样本对不上 ✓✗。
    if len(audios) > len(audio_slot_names):
        _warn(task, f"参考音频只落地了 {len(audio_slot_names)} 路（模板 {template_name} 只有 "
                    f"{len(audio_slot_names)} 个音频槽位，请求 {len(audios)} 路 ✗）"
                    f"⇒ 多出来的**没有上传** ✓、也不会被声明 ✓（要更多音频槽位得改模板 ✓）")
        audios = audios[:len(audio_slot_names)]

    # ⚠️⚠️ 参考视频**两道闸都在上传之前** ✓✗（视频比图/音多一道：还要先问引擎「有没有读视频的类」✓）。
    #    ① 模板没有 ``ref_videos.*`` 槽 ⇒ 注定落不下去 ⇒ 一个都别传 ✓；
    #    ② 有槽、但引擎里**没有任何可用的读视频节点** ⇒ 同样别传 ✓（传上去只会让现建的那个节点
    #       在提交时 400 ✓✗、而且报错指向「类不存在」而不是「我们挑错了类」✓）；
    #    ③ 槽位比请求少 ⇒ 截断（同音频那条 ✓）。
    video_slot_names = list(plan["ref_video_slot_names"])
    video_audio_slot_names = list(plan["ref_video_audio_slot_names"])
    video_loader: Dict[str, Any] | None = None
    if videos and not video_slot_names:
        _warn(task, f"模板（{template_name}）的主节点**没有参考视频槽位** ⇒ {len(videos)} 路参考视频被忽略 ✗"
                    f"（**一个都没上传** ✓；要用视频参考请换 R2V 模板：{R2V_WORKFLOW} ✓，"
                    f"或 body ``template`` 显式指定 ✓）")
    elif videos:
        video_loader, loader_why = _video_loader(client)
        if video_loader is None:
            _warn(task, f"模板有参考视频槽位，但引擎里**找不到能读视频的节点** ✗"
                        f"（逐个问过引擎：{loader_why} ✓）"
                        f"⇒ {len(videos)} 路参考视频**被忽略** ✗、也**没有上传** ✓"
                        f"（装一个读视频的自定义节点 ✓，或用 env ``H3_REF_VIDEO_NODES=类名`` 指到现成的那个 ✓；"
                        f"核心 ``LoadVideo`` 只吐 ``VIDEO`` ⇒ 还要 ``GetVideoComponents`` 才能在模板里当帧串用 ✓，"
                        f"缺哪个装哪个 ✓）")
            # ⚠️ 必须**清空** ✓✗：不清空 ⇒ 下面契约按「落地路数」算时仍会声明 ``<Video k>`` ✓
            #    ⇒ 模型去复刻一段根本没进去的视频 ✓✗（正是音频那半边刚修过的错 ✓）。
            videos = []
    if videos and video_loader is not None and len(videos) > len(video_slot_names):
        _warn(task, f"参考视频只落地了 {len(video_slot_names)} 路（模板 {template_name} 只有 "
                    f"{len(video_slot_names)} 个视频槽位，请求 {len(videos)} 路 ✗）"
                    f"⇒ 多出来的**没有上传** ✓、也不会被声明 ✓（要更多视频槽位得改模板 ✓）")
        videos = videos[:len(video_slot_names)]

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
    wanted_audios = audios if audio_slot_names else []
    # ⚠️ 视频是**三条件** ✓✗：有槽位 ✓ **且** 引擎里有可用的读视频节点 ✓（``video_loader`` 为空 = 落不下去 ✓）
    wanted_videos = videos if (video_slot_names and video_loader) else []
    uploads: list[tuple[str, str, bytes]] = []
    audio_uploads: list[tuple[str, str, bytes]] = []
    video_uploads: list[tuple[str, str, bytes]] = []
    if first_frame and frame_nodes["START FRAME"]:
        uploads.append(("START FRAME", "first_frame.png", _fetch_media(first_frame)))
    if last_frame and frame_nodes["LAST FRAME"]:
        uploads.append(("LAST FRAME", "last_frame.png", _fetch_media(last_frame)))
    for index, url in enumerate(wanted_refs):
        uploads.append((f"REF IMAGE {index + 1}", f"ref_image_{index + 1}.png", _fetch_media(url)))
    # ⚠️ 音频单独一列 ✓✗：文件名后缀要按**这一路**的地址算 ✓（``_audio_filename`` ✓），
    #    不能跟图共用 ``.png`` ✗ —— 后缀不对 ⇒ ``LoadAudio`` 的 COMBO 里根本没有它 ⇒ 提交 400 ✓✗。
    for index, url in enumerate(wanted_audios):
        audio_uploads.append((f"REF AUDIO {index + 1}", _audio_filename(index, url),
                              _fetch_media(url)))
    # 参考视频**再开一列** ✓✗：后缀必须是视频后缀 ✓（读视频那格 COMBO 只列视频文件 ✓）。
    for index, url in enumerate(wanted_videos):
        video_uploads.append((f"REF VIDEO {index + 1}", _video_filename(index, url),
                              _fetch_media(url)))

    uploaded: dict[str, str] = {}
    for keyword, filename, payload in uploads:
        uploaded[keyword] = _landed_upload(client.upload_image(filename, payload), filename)
    for keyword, filename, payload in audio_uploads:
        uploaded[keyword] = _landed_upload(client.upload_audio(filename, payload), filename)
    for keyword, filename, payload in video_uploads:
        uploaded[keyword] = _landed_upload(client.upload_video(filename, payload), filename)
    ref_uploads = [uploaded[f"REF IMAGE {index + 1}"] for index in range(len(wanted_refs))]
    ref_audio_uploads = [uploaded[f"REF AUDIO {index + 1}"] for index in range(len(wanted_audios))]
    ref_video_uploads = [uploaded[f"REF VIDEO {index + 1}"] for index in range(len(wanted_videos))]

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
    #    ⚠️ ``audios`` **同一口径** ✓✗：也用「实际落地」= ``min(请求路数, 模板音频槽位数)`` ✓
    #    （盘上 R2V 模板 1 格 ⇒ 得 1 ✓；请求带 2 路 ⇒ 契约只声明 ``<Audio 1>`` ✓，与真落地一致 ✓）。
    #    ⚠️⚠️ 三个值**一律显式传** ✓✗ —— 别再写 ``... if refs else None`` ✗✗：``None`` 会让
    #    :func:`_contract_prompt` 回落到「数 body 里的请求路数」✓ ⇒ 一旦某个素材**被拦在前面**
    #    （视频找不到读它的类 ⇒ ``videos`` 清空 ✓）就会按 body 又声明一遍 ✓✗（= 声明了没进去的素材 ✓）。
    #    ⚠️ ``video_soundtracks``（2026-10-27 照抄后端口径 ✓✗）：**带音轨的参考视频会先占掉
    #    ``<Audio j>`` 的编号** ✓ —— ``nodes_minimax_h3.py`` 里对每路视频先 ``ref_items.append({"type": "audio"})``
    #    再 append 视频 ✓（注释原文：「the soundtrack gets its own ``<Audio j>`` label, emitted before
    #    ``<Video k>``」✓），独立音频 ``ref_audios`` 排在**全部视频之后** ✓。
    #    ⇒ 声轨数 = 真能接上音轨的视频路数 ✓（模板有音轨槽 ✓ **且** 上游能拆出 ``AUDIO`` ✓），
    #    独立音频的编号从它后面起 ✓✗（不算 ⇒ 两边都用 ``<Audio 1>`` ⇒ 模型把音乐当音色参考 ✓✗）。
    video_soundtracks = min(len(videos), len(video_slot_names), len(video_audio_slot_names)) \
        if (videos and video_loader and video_loader.get("audio_slot") is not None) else 0
    prompt_text = _contract_prompt(task, body, prompt_text,
                                   pictures=min(len(refs), len(ref_slot_names)),
                                   audios=min(len(audios), len(audio_slot_names)),
                                   videos=min(len(videos), len(video_slot_names)),
                                   video_soundtracks=video_soundtracks)
    owner = plan["prompt_owner"]
    if owner is None:
        _warn(task, "没找到提示词注入点 ✗（主节点的 ``prompt`` 既不是自身 widget ✓、"
                    "上游也不是字符串节点 ✓）⇒ 这次提交的提示词是**模板里那串默认文案** ✗")
    elif not wf.set_node_input(api, owner[0], owner[1], prompt_text):
        _warn(task, f"提示词注入点 {owner[0]}.{owner[1]} 不在 API 图里 ✗（转换时被丢了？）")

    # ── 注入：时长（秒 → **喂 ``length`` 的那个标量节点** ✓；帧数由模板表达式按 24fps/17k+5 算 ✓）──
    #    ⚠️ 两道**后端硬口径**在这里收口 ✓✗（2026-10-27 照抄 ``nodes_minimax_h3.py`` ✓）：
    #    ① ``length`` 的 ``max=3600`` ★ —— 超了引擎 ``validate_inputs`` **直接拒单** ✗（不会替我们截 ✓）；
    #    ② tooltip 声明的训练区间 ``~124-362`` 帧 ✓ —— 出了区间**不报错** ✓，只降稳定性 ✓ ⇒ 只告警 ✓。
    duration = float(body.get("duration") or 5)
    expected_frames = wf.frames_for_seconds(duration)
    if expected_frames > wf.H3_MAX_FRAMES:
        capped = wf.cap_frames(expected_frames)
        _warn(task, f"时长 {duration:g}s ⇒ {expected_frames} 帧，超过后端 ``length`` 的硬上限 "
                    f"{wf.H3_MAX_FRAMES} 帧（{wf.seconds_for_frames(wf.H3_MAX_FRAMES)}s）✗ "
                    f"⇒ 按**上限内最大的合法帧数** {capped} 帧"
                    f"（{wf.seconds_for_frames(capped)}s）出 ✓"
                    f"（再长只能拆成多单接 ✓；不夹住的话这一单会被引擎拒 ✗）")
        expected_frames = capped
        duration = capped / float(wf.H3_FPS)
    #    ⚠️ 下限那格（schema 的 ``min=5``）**不需要**在这里判 ✓✗：:func:`wf.frames_for_seconds`
    #       本身就是 ``raw = max(min_frames, ...)`` ✓ ⇒ ``expected_frames`` **恒 ≥ 5** ✓，
    #       写一个 ``< H3_MIN_FRAMES`` 的分支是**永远进不去**的死代码 ✗（还会让人误以为这里验过下限 ✓✗）。
    low, high = wf.H3_TRAINED_FRAMES
    if not low <= expected_frames <= high:
        _warn(task, f"{expected_frames} 帧（{wf.seconds_for_frames(expected_frames)}s）落在后端声明的"
                    f"**训练区间 {low}–{high} 帧"
                    f"（{wf.seconds_for_frames(low)}–{wf.seconds_for_frames(high)}s）**之外 ⇒ 照做 ✓，"
                    f"但画质／音画稳定性没有保证 ✓（这不是错误 ✓）")
    if plan["duration_id"] and wf.set_node_input(api, plan["duration_id"], "value", duration):
        pass
    else:
        _warn(task, f"没找到时长注入点 ✗（喂 ``length`` 的唯一 ``PrimitiveFloat`` 读不出来 ✓）"
                    f"⇒ 时长沿用模板默认值 ✗（请求的是 {duration:g}s）")

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

    # ── 注入：参考素材（图 ✓ / 音频 ✓ / 视频 ✓ —— 2026-10-27 起三类全接）──
    #    ⚠️ 「模板没槽位」那两条告警**在上传前**就发过了 ✓（那时才决定传不传 ✓）⇒ 这里只管落槽 ✓
    #    ⚠️ 两路**各落各的槽位表** ✓✗（图看 ``ref_images.*``、音看 ``ref_audios.*`` ✓）；
    #    旧代码到这一步就一句「参考音频本地模板还没接上 ⇒ 被忽略」✗ —— 而盘上模板**有**音频槽位 ✓，
    #    于是那句话本身就是假话 ✓✗（即 ⑬ 记的那条：文档/注释与盘上模板不一致 ✓）。
    landed_refs = (_wire_reference_images(task, api, plan, ref_uploads)
                   if refs and ref_slot_names else 0)
    landed_audios = (_wire_reference_audios(task, api, plan, ref_audio_uploads)
                     if audios and audio_slot_names else 0)
    # ⚠️ 视频**额外要 loader** ✓✗：没问出可用的读视频节点 ⇒ 连节点都不该建 ✓（否则提交时才 400 ✓）。
    landed_videos = (_wire_reference_videos(task, api, plan, ref_video_uploads, video_loader)
                     if wanted_videos and video_loader else 0)
    # ⚠️ 参考图缩放档位**必须等图真落了再改** ✓✗：一路图都没进去 ⇒ 那格对本次出图毫无作用 ✓
    #    （改了只会让「回执说 max」与「实际没参考图」互相打脸 ✓）。
    applied_image_size = _wire_reference_image_size(task, api, plan, body, landed_refs)

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
            uploaded=uploaded, reference_images=landed_refs, reference_audios=landed_audios,
            reference_videos=landed_videos,
            video_loader=(video_loader or {}).get("class"),
            # 参考图缩放档位：``None`` = 没显式设过（没有参考图 ⇒ 不需要 ✓ / 模板没这格 ⇒ 落不下去 ✓）
            reference_image_size=applied_image_size,
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
        # ⚠️ **音频单独一格** ✓✗（2026-10-27）：图那格只报图 ✓ —— 只报图 ⇒ 调用方看不出「声线样本到底
        #    落上没有」✓✗（旧行为就是「落不上也不说」✓），而这两条路是**各落各的槽位表** ✓（见
        #    ``_wire_reference_images`` / ``_wire_reference_audios`` ✓）。值都是**落地路数（整数）** ✓，
        #    不是文件名列表 ✗（别和模板清单里的 ``reference_audio_slots`` 混 ✓）。
        "reference_audios": task.get("reference_audios"),
        # ⚠️ **视频也单独一格** ✓✗（同音频那条理由 ✓）：它比图/音多一个坑 —— 「帧串进去了、音轨没进去」
        #    或「模板有槽但引擎里没读视频的节点」都只能靠这一格看出来 ✓。
        "reference_videos": task.get("reference_videos"),
        #: 真用的读视频节点类名 ✓（``None`` = 没问出可用的 ✓；与 ``reference_videos`` 一起读才完整 ✓）
        "video_loader": task.get("video_loader"),
        #: 参考图缩放档位**实际生效值** ✓（``match``/``max`` ✓；``None`` = 没显式设过 ——
        #: 没有参考图 ✓，或模板没有那格 ✓ ⇒ 此时别把这格读成「引擎用了默认值以外的东西」✗）。
        "reference_image_size": task.get("reference_image_size"),
        #: 声轨数 ✓（带音轨的参考视频路数 ✓）—— 它占掉 ``<Audio j>`` 的**前几位** ✓✗：
        #: 调用方写 ``<Audio 1>`` 时，看到这一格非 0 就知道那指的是**视频声轨**、不是独立音频 ✓。
        "reference_video_soundtracks": task.get("reference_video_soundtracks"),
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
            # ⚠️ 音频**单列** ✓✗（2026-10-27）：调用方拿这个清单挑模板 ✓，只报图槽位 ⇒ 它会挑一个
            #    「有语音样本、没音频槽位」的模板 ✓✗（正是本条要防的错 ✓）。
            "reference_audio_slots": plan.get("ref_audio_slot_names") or [],
            "reference_audio_nodes": plan.get("ref_audio_ids") or [],
            # ⚠️ 视频**再单列** ✓✗（2026-10-27 补齐 R2V 的第三类素材 ✓）——尤其要把
            #    ``reference_video_audio_slots`` 报出来 ✗：模板可能只给帧串槽、不给音轨槽 ✓，
            #    调用方据此才知道「这条视频的声音进不进得去」✓。
            "reference_video_slots": plan.get("ref_video_slot_names") or [],
            "reference_video_nodes": plan.get("ref_video_ids") or [],
            "reference_video_audio_slots": plan.get("ref_video_audio_slot_names") or [],
            "reference_video_audio_nodes": plan.get("ref_video_audio_ids") or [],
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
        #: 参考图缩放档位 ✓（后端 ``ref_image_size`` 的两个选项 ✓）——``match`` 快 ✓、``max`` 身份最保真 ✓。
        #: ⚠️ **哪份模板有这一格**这里不报 ✗：模板里它是普通 widget（UI 图上没有字段名 ✓，
        #: 只有转成 API 图后才认得出来 ✓）⇒ 由提交回执的 ``reference_image_size`` 如实回答 ✓
        #: （落不下去时那一格是 ``None`` ✓ 且带告警 ✓）。
        "reference_image_size_options": list(REF_IMAGE_SIZES),
        "reference_image_size_default": REF_IMAGE_SIZE_DEFAULT,
        "reference_audio_limit": REF_AUDIO_LIMIT,
        "reference_video_limit": REF_VIDEO_LIMIT,
        #: 「要读视频，引擎里得有个什么类」也一并公开 ✓✗ —— 它是**运行时探测**的结论 ✓（见
        #: ``_video_loader`` ✓）：调用方据此判断这台机器能不能用参考视频 ✓，不用去猜 ✗。
        "reference_video_loader_candidates": list(REF_VIDEO_LOADERS),
        #: 「视频 → 帧串/音轨」的拆解节点候选 ✓（核心 ``GetVideoComponents`` ✓）—— 现版核心
        #: ``LoadVideo`` 只吐 ``VIDEO`` ⇒ 缺它就没法把参考视频喂进 ``ref_videos`` ✓✗
        "reference_video_component_nodes": list(REF_VIDEO_COMPONENTS),
        #: 参考视频的**后端硬口径** ✓ —— 逐条抄自 ``comfy_extras/nodes_minimax_h3.py`` ✓：
        #: ``io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (2-15s)")`` ✓、
        #: 执行期 ``frames < 5`` 直接抛错（"at least 5 frames (~0.2s at 24 fps)"）✓、
        #: 超长截到 ``frame_count`` ✓、再裁到 ``n % 17 == 5`` ✓、音轨按**同名下标**配对 ✓。
        #: ⚠️ 帧数/时长我们**在本地核不了** ✗（解码在引擎侧 ✓）⇒ 只公开口径、不假装能验 ✓✗。
        "reference_video_rules": {
            "min_frames": 5,
            "fps": 24,
            "seconds": [2, 15],
            "frame_grid": "17k+5",
            "truncated_to_target": True,
            "soundtrack_pairing": "index",
            "audio_order": "soundtracks_then_ref_audios",
        },
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
