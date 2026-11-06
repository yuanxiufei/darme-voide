"""ComfyUI 工作流运行 —— **项目自己的一等公民能力**（2026-09-17）。

## 这不是「透传」，是「实现」

上游（``app/local_services/h3/server.py``，端口 8765）**会**把 ComfyUI 的接口包成我们的形状，
但那只是**能力提供方** ✓。真正让「跑工作流」成为本项目功能的是这一层：

| 能力 | 本项目怎么**自己**实现 |
|---|---|
**记录** | 每次运行写一行 ``comfyui_runs``（状态机 ``queued → processing → succeeded/failed`` ✓）—— 不是把 ComfyUI 的 history 原样转发 ✗ |
**崩溃恢复** | 重启后按 ``remote_task_id`` **续询**（``recover_runs_on_startup`` ✓）；没提交成功的才判 failed ✓ —— 与 ``video_generation`` 同形（**绝不重提交**，避免重复占用 GPU/算力 ✓） |
**串行与排队** | 走 ``gpu_manager`` 的**租约**（本地服务 ✓）⇒ 同一时刻只有一个重负载任务（ComfyUI 自己的队列是它的内部事 ✓ 我们不管 ✗） |
**产物流转** | 产物**下载进我们自己的数据根** ``static/comfyui/<uuid>.<ext>`` ✓（前端可直接播放 ✓）—— 而不是只丢一个指向 8765 的 URL ✗ |
**用量记账** | ``api_usage``（``service_type`` / ``units`` / ``is_local`` ✓），与图片/视频链路同一套口径 ✓ |
**错误归因** | 失败写 ``error_msg``（上游给的原因原样保留 + **可执行建议** ✓），并区分「缺节点 / 缺模型 / 缺 SageAttention / VDN 适配器 / 显存不足 / 服务不可达 / 超时」✓ |
**显存重试** | 上游报显存不足 ⇒ **卸载模型 + 重试一次**（``/v1/free`` 后重提交 ✓；仍不足才判失败，并把「已重试过」写进原因 ✓）——**只重试一次**，绝不无限重刷占 GPU ✗ |

## 与上游的分工（谁负责什么）

* **8765 服务**：懂 ComfyUI（UI→API 转换、节点预检、`/prompt`、轮询、`/free` ✓）；
* **本模块**：懂**项目**（DB、租约、崩溃恢复、产物落地、记账、对外 camelCase ✓）。

⚠️ 本模块**不 import** ``app.local_services.*`` ✗ —— 分层守卫规定 ``services`` 只能依赖 ``core`` ✓；
两者之间是 **HTTP**（``COMFYUI_SERVICE_URL``，默认 ``http://127.0.0.1:8765`` ✓），
和本项目对接 ollama / local-sd 的方式一致 ✓。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.engine import Connection, Row

from ..core.config import get_storage_root
from ..core.db import engine
from ..core.models import comfyui_runs
from ..core.response import now
from . import comfyui_bindings as bindings
from . import comfyui_catalog as catalog
from . import comfyui_progress as progress
from .comfyui_lora_chain import apply_lora_chain
from .comfyui_ui_workflow import convert_ui_workflow, is_ui_workflow
from .file_storage import download_file, get_absolute_path
from .gpu_manager import gpu_manager
from .task_logger import (
    log_task_error,
    log_task_progress,
    log_task_start,
    log_task_success,
    log_task_warn,
)
from .usage_tracking import record_usage

__all__ = [
    "PURGE_PATH_PREFIXES",
    "RETENTION_DAYS",
    "TERMINAL_STATUSES",
    "cancel_run",
    "describe_failure",
    "failure_hints",
    "free_upstream",
    "get_run",
    "interrupt_upstream",
    "is_out_of_memory",
    "list_runs",
    "purge_runs",
    "recover_runs_on_startup",
    "release_run_lease",
    "run_to_dict",
    "service_url",
    "stats",
    "submit_run",
    "system_snapshot",
    "upstream_get",
]

#: 后台任务集合（**不持引用会被 GC 掉** ⇒ 表现为「永远停在 processing」✓ 与 image/video 链路同款坑）
_background_tasks: set[asyncio.Task] = set()

#: 长租约（提交时持有、完成/失败时释放 ✓）
_run_leases: dict[int, Any] = {}


def _spawn(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ══════════════════════════════════════════════════════════════════════════
# 配置（**调用时读**，便于测试改环境变量 ✓）
# ══════════════════════════════════════════════════════════════════════════
def service_url() -> str:
    """上游服务地址（我们的 8765 门面 ✓）—— 调用时读环境，便于自检替换 ✓。"""
    return (os.environ.get("COMFYUI_SERVICE_URL") or "http://127.0.0.1:8765").rstrip("/")


def _run_timeout() -> float:
    return float(os.environ.get("COMFYUI_RUN_TIMEOUT") or "1800")


def _poll_seconds() -> float:
    return float(os.environ.get("COMFYUI_POLL_SECONDS") or "2")


def _dumps(value: Any) -> str:
    """紧凑 JSON（本仓约定：与 JS ``JSON.stringify`` 字节一致 ✓ 守卫在盯 ✓）。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(text: Any, fallback: Any) -> Any:
    if not text:
        return fallback
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return fallback


# ══════════════════════════════════════════════════════════════════════════
# 失败归因（2026-11-03 自 comfyui-gateway ``src/runner.js`` 迁入）
# ══════════════════════════════════════════════════════════════════════════
#: 上游失败**原文**照留 ✓（可追溯 ✓），但要补一句「据此能改什么」✓。
#:
#: 为什么必须有这张表：ComfyUI 的原句是英文节点堆栈（``Value not in list: unet_name:
#: 'xxx' not in [...]`` ✗），对使用者等于没说 —— 他既不知道**是模型名对不上**，
#: 也不知道**该去哪改**（本仓反复踩的「有报错、没结论」✗）。因此：原文一字不改 ✓，
#: 后面追加一条能直接动手的建议 ✓；认不出来就**只留原文** ✓（不硬编原因 ✗）。
_FAILURE_HINTS: tuple[tuple[str, str], ...] = (
    (
        r"allocation on device|out\s*of\s*memory|outofmemoryerror|not enough memory|显存不足",
        "显存不足：把分辨率/帧数/时长调小，或先关掉其它占用显存的程序（画布同档位连跑时最容易撞上）",
    ),
    (
        r"not in list|value_not_in_list",
        "工作流引用的模型文件在本机 ComfyUI 里不存在：到「模型」页核对文件名，或把该档换成本机已有的那一份",
    ),
    (
        r"cannot find node|node type[^\n]*not|class_type[^\n]*not found|node[^\n]*not found|缺少节点包",
        "缺节点插件：ComfyUI 少装了某个节点包（按上面点名的节点去装，装完重启 ComfyUI 再跑）",
    ),
    (
        r"sageattention is not new enough|memory efficient sage attention|sageattention",
        "上游缺/旧 SageAttention 运行库：H3 的加速补丁跑不起来 ⇒ 装好或升级 SageAttention 并重启 ComfyUI 再跑"
        "（本仓 H3 全部工作流都挂着该补丁，没有可绕的变体）",
    ),
    (
        r"stage-?b[^\n]*adapter[^\n]*missing|adapter[^\n]*stage-?b|vdn[^\n]*(missing|不完整)",
        "VDN 适配器不完整：stage-dmd-step-250 缺 adapters/default（Stage-B）⇒ 补齐目录结构后重启，"
        "或直接换 video-h3-*-novdn 变体",
    ),
    (
        r"connection refused|connecterror|服务不可达|econnrefused",
        "上游不可达：先起本地服务（8765），它再去连 ComfyUI（8188）",
    ),
)

#: 显存不足的判定（**与建议表分开** ✓：重试逻辑要一个 True/False，不要一串文案 ✓）
_OUT_OF_MEMORY = re.compile(
    r"allocation on device|out\s*of\s*memory|outofmemoryerror|not enough memory|显存不足",
    re.IGNORECASE,
)


def _oom_retry_wait() -> float:
    """显存重试前的等待（默认 1.5s：给上游 unload 留一拍 ✓；自检置 0 ✓）。"""
    return float(os.environ.get("COMFYUI_OOM_RETRY_WAIT") or "1.5")


def is_out_of_memory(message: Any) -> bool:
    """上游这句失败是**显存不足**吗（唯一判据 ✓ 重试与否只问它 ✓）。"""
    return bool(_OUT_OF_MEMORY.search(str(message or "")))


def failure_hints(message: Any) -> list[str]:
    """按失败原文给出**可执行建议**（认不出 ⇒ 空列表 ✓ 不硬编原因 ✗）。"""
    text = str(message or "")
    hints: list[str] = []
    for pattern, hint in _FAILURE_HINTS:
        if hint in hints:
            continue
        if re.search(pattern, text, re.IGNORECASE):
            hints.append(hint)
    return hints


def describe_failure(message: Any, *, oom_retried: bool = False) -> str:
    """上游原文 + 建议 ⇒ 写进 ``error_msg`` 的一行中文（原文**一字不改** ✓）。"""
    text = str(message or "").strip() or "上游报告失败（无原因）"
    hints = failure_hints(text)
    if oom_retried:
        hints = [hint for hint in hints if "显存不足" not in hint]
        hints.insert(0, "显存不足：已自动卸载上游模型并重试一次仍然不足 ⇒ 把分辨率/帧数/时长调小，"
                        "或先关掉其它占用显存的程序")
    if not hints or "建议：" in text:
        return text
    return f"{text}｜建议：{'；'.join(hints)}"


# ══════════════════════════════════════════════════════════════════════════
# 查询 / 状态
# ══════════════════════════════════════════════════════════════════════════
def _update(run_id: int, **fields: Any) -> None:
    fields["updated_at"] = now()
    with engine.begin() as conn:
        conn.execute(comfyui_runs.update().where(comfyui_runs.c.id == run_id).values(**fields))


def get_run(conn: Connection, run_id: int) -> Row | None:
    return conn.execute(select(comfyui_runs).where(comfyui_runs.c.id == run_id)).first()


def list_runs(conn: Connection, *, status: str | None = None, limit: int = 50) -> list[Row]:
    """按状态过滤（不给就全部 ✓），新的在前 ✓。"""
    statement = select(comfyui_runs).order_by(comfyui_runs.c.id.desc()).limit(max(1, min(limit, 500)))
    if status:
        statement = statement.where(comfyui_runs.c.status == status)
    return list(conn.execute(statement).all())


def run_to_dict(row: Row) -> dict[str, Any]:
    """行 → 字典（camelCase 由路由层统一转 ✓；JSON 字段在这里解开 ✓）。"""
    return {
        "id": row.id,
        "kind": row.kind,
        "workflowName": row.workflow_name,
        "sourcePrompt": row.source_prompt,
        "params": _loads(row.params, {}),
        "status": row.status,
        "step": row.step,
        "remoteTaskId": row.remote_task_id,
        "outputs": _loads(row.outputs, []),
        "primaryUrl": row.primary_url,
        "localPath": row.local_path,
        "errorMsg": row.error_msg,
        "warnings": _loads(row.warnings, []),
        "elapsedMs": row.elapsed_ms,
        "requestedSeconds": row.requested_seconds,
        "usedSeconds": row.used_seconds,
        "isLocal": bool(row.is_local),
        "createdAt": row.created_at,
        "updatedAt": row.updated_at,
    }


# ══════════════════════════════════════════════════════════════════════════
# 参考素材：传进 ComfyUI 的 input 目录
# ══════════════════════════════════════════════════════════════════════════
def _reference_file(url: str) -> tuple[bytes, str]:
    """把本仓素材地址读成 ``(字节, 文件名)`` ✓（只认 ``/uploads`` ``/static`` 下的东西 ✓）。"""
    from .file_storage import get_absolute_path

    # ⚠️ 前导斜杠必须去掉：``Path(base) / "/uploads/x.png"`` 在 Windows 上会把 base **整个丢掉**
    #    （绝对路径优先）⇒ 解析成 ``C:\uploads\x.png`` ⇒ 被判穿越 ✗
    relative = str(url).strip().lstrip("/").replace("\\", "/")
    path = Path(get_absolute_path(relative))
    if not path.is_file():
        raise ValueError(f"参考素材不在盘上：{url}")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"参考素材超过 64MB，拒绝上传：{url}")
    return path.read_bytes(), path.name


async def stage_reference_uploads(params: dict[str, Any]) -> list[dict[str, Any]]:
    """把画布素材**先传进 ComfyUI 的 input 目录** ✓，返回 ``[{name, subfolder}]``。

    为什么必须由本仓做：上游 ``/v1/workflows/run`` 只做「图 + params 注入 + 节点预检」，
    **不碰参考素材**（上传那条是 H3 专用路径 ✓）。而 ``LoadImage.image`` 是 COMBO ——
    往它写一个**不存在的文件名**，ComfyUI 提交前校验会当场 400 ✗（错误还指向别处 ✗）。
    ⇒ 参考图必须先落进 input 目录，节点才能引用它 ✓。

    ``uploadedImages`` 正是 :func:`comfyui_bindings.apply_bindings` 认的形状 ✓
    （``{"name": ..., "subfolder": ...}`` ⇒ 拼成 ``subfolder/name`` ✓ 见 ``_media_value`` ✓）。
    """
    raw = params.get("referenceImageUrls")
    if not raw:
        return []
    urls = raw if isinstance(raw, list) else [raw]
    staged: list[dict[str, Any]] = []
    base = service_url()
    for item in urls:
        text = str(item or "").strip()
        if not text:
            continue
        if text.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.get(text)
            if response.status_code >= 400:
                raise ValueError(f"参考图下载失败（HTTP {response.status_code}）：{text}")
            payload = response.content
            name = text.rsplit("/", 1)[-1].split("?")[0] or "reference.png"
        else:
            payload, name = _reference_file(text)
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"{base}/v1/upload/image",
                files={"image": (name, payload, "application/octet-stream")},
                data={"subfolder": "", "overwrite": "true"},
            )
        if response.status_code >= 400:
            raise ValueError(
                f"参考图上传到 ComfyUI 失败（HTTP {response.status_code}）：{response.text[:160]}")
        info = response.json() if response.content else {}
        if not info.get("name"):
            raise ValueError(f"参考图上传没拿到文件名：{info!r}")
        staged.append({"name": info["name"], "subfolder": info.get("subfolder") or ""})
    return staged


async def prepare_reference_media(params: dict[str, Any]) -> None:
    """提交/校验前的**素材前置** ✓：``referenceImageUrls`` → ``uploadedImages`` ✓。

    失败 **必须抛** ✓：参考图丢了还照跑，等于出一张"没参考到"的图，
    而用户看不出是素材被吞 ✗（本仓反复踩的那类静默错 ✗）。
    """
    # ⚠️ 只服务 ``kind=workflow`` ✓：H3 的参考素材走**另一条协议**（``/v1/video_generation``
    #    收 ``referenceImageUrls`` 自己去取 ✓），在这一层替它下载+上传是**白干一遍** ✗；
    #    更要紧的是远端 URL 取不到时会**在提交前就抛** ✗ ⇒ 一条本来能跑的 H3 参考图请求
    #    被我们自己的前置处理打掉（测试里 ``https://x/a.png`` 就是这么炸的 ✓）。
    if str(params.get("kind") or "workflow").strip() == "h3":
        return
    if params.get("uploadedImages") or not params.get("referenceImageUrls"):
        return
    params["uploadedImages"] = await stage_reference_uploads(params)


# ══════════════════════════════════════════════════════════════════════════
# 注册表 entry → 可提交的图
# ══════════════════════════════════════════════════════════════════════════
def build_registry_graph(params: dict[str, Any]) -> dict[str, Any]:
    """按注册表 entry 生成**可直接提交**的 API 图 ✓（调用方只给 ``workflowId`` + 参数 ✓）。

    为什么必须有这一步：``POST /runs`` 原先只收**整张图**（``workflow``）或**盘上路径**
    （``workflowPath``）。前端要跑「image-krea2-turbo」就得自己下载 20 个 JSON、自己找
    ``#5.text`` 在哪、自己按 ``snap16`` 吸尺寸 —— 那是让前端**复刻一遍 bindings** ✗，
    而且一旦两侧算法漂移，出的图就是"看着成功、其实参数没进去" ✗✗。
    给了 ``workflowId`` 之后，参数注入回到服务层（唯一真源 ✓）。

    ``values`` 是前端表单的原始键值（``seed`` / ``steps`` / ``unet`` …）——
    它们通过 :func:`comfyui_bindings.build_context` 的 ``params`` 块被 ``from: "params.xxx"`` 取到 ✓。
    """
    workflow_id = str(params.get("workflowId") or "").strip()
    entry = catalog.get_entry(workflow_id)
    if entry is None:
        raise ValueError(
            f"没有工作流 {workflow_id!r}（可跑清单见 GET /api/v1/comfyui/catalog/workflows）")
    variant = str(params.get("variant") or "").strip() or None
    # ⚠️ 变体不在册要报错 ✓（catalog.load_graph 自己会抛 CatalogError）
    graph = catalog.load_graph(entry, variant)
    # ⚠️ **注入也必须认变体** ✓：变体自己声明的 ``bindings``（蒙版/留白/sigma 串…）
    #    与 ``parameters`` 是它存在的**全部理由** —— 图读的是变体那份、注入却拿主 entry，
    #    症状是"选了 inpaint 却按 t2i 跑"且**一行都不报错** ✗✗
    #    （``bindings``/``parameters`` 的合并口径以 :func:`catalog.resolve_variant` 为唯一真源 ✓）
    resolved = catalog.resolve_variant(entry, variant)

    values = params.get("values")
    values = dict(values) if isinstance(values, dict) else {}
    context = bindings.build_context(
        prompt=str(params.get("prompt") or values.get("prompt") or ""),
        negative=params.get("negative"),
        size=params.get("size") or params.get("aspectRatio"),
        count=params.get("count") or values.get("count") or 1,
        quality=params.get("quality") or values.get("quality"),
        seconds=params.get("seconds") or params.get("duration"),
        mode=params.get("mode"),
        background=params.get("background"),
        images=params.get("referenceImageUrls"),
        # ⚠️ 参考图必须走 ``uploaded_images`` ✓（``from: "image"`` 的 binding 从这里建 LoadImage 节点 ✓）：
        #    直接塞 URL 进图，ComfyUI 会把它当"input 目录里的文件名"查 ⇒ 400 ✗
        uploaded_images=params.get("uploadedImages"),
        params=values,
        presets=entry.get("presets"),
    )
    graph, meta = bindings.apply_bindings(graph, resolved, context)
    # 回执里能查「到底注入了什么、哪个 binding 配了却没生效」✓（``skipped`` 是真实缺口，不许吞 ✗）
    params["bindingsMeta"] = meta
    params["workflowVariant"] = variant
    params.setdefault("workflowName", entry.get("name") or entry.get("id"))
    return graph


def build_submit_body(params: dict[str, Any]) -> dict[str, Any]:
    """「人给的参数」→ 上游 body 的**唯一入口** ✓（``workflowId`` 在这里展开成图 ✓）。

    ⚠️ ``submit_run`` 与路由的 ``/validate`` **必须都走这里**：各写一遍的话，
    校验的那张图与提交的那张图就不是同一张，校验等于摆设 ✗。
    """
    if params.get("workflowId") and not (params.get("workflow") or params.get("workflowPath")):
        params["workflow"] = build_registry_graph(params)
    return _build_upstream_body(str(params.get("kind") or "workflow"), params)


async def validate_workflow_body(body: dict[str, Any]) -> dict[str, Any]:
    """提交前**只校验不提交** ✓（上游 ``POST /v1/workflows/validate``）。

    不校验的代价：用户点「生成」后要**等队列跑完**，才在 ComfyUI 日志里看到
    ``Prompt outputs failed validation`` ✗ —— 缺哪个节点包、哪根线悬空、哪个注入点没找到，
    全靠人肉翻日志猜 ✗。上游这条接口就是为这个补的，本仓原来没接 ✓。

    ``checked=False`` 表示**没校验**（上游不可达）—— 这是"不知道"，不是"通过" ✓。
    """
    try:
        result = await _post_json(f"{service_url()}/v1/workflows/validate", body, timeout=60)
    except Exception as err:  # noqa: BLE001 —— 上游不在线 ≠ 图不对 ✓
        return {"ok": True, "checked": False, "reason": f"校验服务不可达：{err}"[:240],
                "problems": [], "missingNodes": []}
    problems = [
        *(result.get("problems") or []),
        *(result.get("injection_problems") or []),
    ]
    if result.get("error"):
        problems.append(str(result["error"]))
    return {
        "ok": bool(result.get("ok")) and not problems,
        "checked": True,
        "nodes": result.get("nodes"),
        "problems": [str(item)[:300] for item in problems],
        "missingNodes": result.get("missing_nodes") or [],
    }


# ══════════════════════════════════════════════════════════════════════════
# 入队
# ══════════════════════════════════════════════════════════════════════════
async def submit_run(conn: Connection, params: dict[str, Any]) -> int:
    """入队一次工作流运行，**立即返回 id**（后台跑 ✓ 与 generation 链路同形 ✓）。

    ``params``：
      * ``kind``：``h3``（用 H3 薄封装的视频协议 ✓）或 ``workflow``（任意工作流 ✓，默认）
      * ``workflow`` / ``workflowPath``：UI 或 API 格式的工作流（``kind=workflow`` 时必填 ✓）
      * ``prompt`` / ``duration`` / ``aspectRatio`` / ``firstFrameImage`` / ``lastFrameImage``：
        传给上游（H3 用 ✓；workflow 也可带，由上游决定要不要用 ✓）
      * ``referenceImageUrls`` / ``referenceAudioUrls`` / ``referenceVideoUrls`` / ``referenceImageSize``：
        H3 的参考素材与参考图缩放档位 ✓（原样转给上游 ✓，见 :func:`_build_upstream_body` ✓）
      * ``inject``：显式参数注入清单（``[{class_type, field, value, title?}]`` ✓）
      * ``settings``：交给上游的额外设置（如 ``checkpoint_map`` ✓）
    """
    ts = now()
    kind = str(params.get("kind") or "workflow")
    # 素材前置 ✓：参考图先落进 ComfyUI 的 input（失败就抛 ⇒ **绝不静默丢素材** ✓）
    await prepare_reference_media(params)
    # ``workflowId`` 在这里展开成图（与 /validate 同一个入口 ✓ 见 build_submit_body ✓）
    body = build_submit_body(params)
    result = conn.execute(comfyui_runs.insert().values(
        kind=kind,
        workflow_name=params.get("workflowName") or _workflow_name(params),
        source_prompt=params.get("prompt"),
        params=_dumps(params),
        status="queued",
        step="queued",
        settings=_dumps(params.get("settings") or {}),
        requested_seconds=_as_float(params.get("duration")),
        is_local=True,
        created_at=ts,
        updated_at=ts,
    ))
    run_id = int(result.inserted_primary_key[0])
    log_task_start("ComfyUIRun", "submit", {"id": run_id, "kind": kind,
                                            "seconds": params.get("duration")})
    _spawn(_process_run(run_id, body))
    return run_id


def _workflow_name(params: dict[str, Any]) -> str | None:
    workflow = params.get("workflow")
    if isinstance(workflow, dict):
        return str(workflow.get("name") or workflow.get("id") or "") or None
    path = params.get("workflowPath")
    return os.path.basename(str(path)) if path else None


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _with_lora_chain(workflow: Any, params: dict[str, Any]) -> Any:
    """把 ``params.loras`` 串成节点链插进工作流 ✓（UI 格式先转 API ✓）。

    ⭐ 没有 ``loras`` ⇒ **原对象原样返回** ✓（不克隆、不改一个字节 ✓）。
    ``applied`` 写回 ``params`` ⇒ 随 run 的 ``params`` 落库，回执里能查「到底加了哪几个」✓。
    """
    loras = params.get("loras")
    if not loras or not isinstance(workflow, dict):
        return workflow
    graph = workflow
    if is_ui_workflow(graph):
        graph, _warnings = convert_ui_workflow(graph)
    graph, applied = apply_lora_chain(
        graph,
        {"loras": loras},
        available=params.get("loraFiles"),
        filter=params.get("loraFilter"),
        source_node=params.get("loraSourceNode"),
        loader_class=str(params.get("loraLoaderClass") or "LoraLoaderModelOnly"),
    )
    params["loraApplied"] = applied
    return graph


def _build_upstream_body(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """把项目参数翻成**上游 8765 的两种协议之一** ✓（翻译只在服务层做 ✓）。"""
    if kind == "h3":
        body: dict[str, Any] = {"prompt": params.get("prompt") or ""}
        for key, value in (("duration", params.get("duration")),
                           ("aspect_ratio", params.get("aspectRatio")),
                           ("first_frame_image", params.get("firstFrameImage")),
                           ("last_frame_image", params.get("lastFrameImage")),
                           ("scene_type", params.get("sceneType"))):
            if value is not None:
                body[key] = value
        # ⚠️⚠️ 参考素材**必须原样带过去** ✓✗（2026-10-27 补）：上游 8765 认 ``referenceImageUrls`` /
        #    ``referenceAudioUrls`` / ``referenceVideoUrls`` / ``referenceImageSize``
        #    （见 ``local_services/h3/server.py`` 的 ``_reference_images`` / ``_reference_audios`` /
        #    ``_reference_videos`` / ``_reference_image_size`` ✓），而这里**原来一个都不传** ✗
        #    ⇒ 产品侧带参考素材跑 H3 时，素材**在服务层就被丢掉** ✓✗：收端按纯文生跑完，
        #    回执里 ``reference_*`` 全是 0 ✓，几乎看不出是被谁丢的 ✓（本仓反复记的那类静默错 ✓）。
        #    ⚠️ 键名用 ``camelCase`` ✓：那是上游 body 的**协议口径** ✓（不是我们另起一套 ✗）。
        for key, value in (("referenceImageUrls", params.get("referenceImageUrls")),
                           ("referenceAudioUrls", params.get("referenceAudioUrls")),
                           ("referenceVideoUrls", params.get("referenceVideoUrls")),
                           ("referenceImageSize", params.get("referenceImageSize"))):
            if value not in (None, "", [], {}):
                body[key] = value
        if params.get("settings"):
            body["settings"] = params["settings"]
        return body

    body = {}
    if params.get("workflow") is not None:
        body["workflow"] = _with_lora_chain(params["workflow"], params)
    if params.get("workflowPath"):
        # ⚠️ 只给路径时我们**读不到图** ⇒ 无法找模型源 ⇒ 明确报错，
        #    绝不「收下 loras 却悄悄不注入」（用户会以为 LoRA 生效了 ✗✗）
        if params.get("loras"):
            raise ValueError("给了 loras 但只给了 workflowPath（盘上图）：请改传 workflow（API/UI 图），否则无法注入 LoRA")
        body["workflow_path"] = params["workflowPath"]
    if params.get("inject"):
        body["params"] = params["inject"]
    return body


# ══════════════════════════════════════════════════════════════════════════
# 执行
# ══════════════════════════════════════════════════════════════════════════
def _endpoints(kind: str) -> tuple[str, str]:
    """``(提交路径, 轮询路径模板)`` ✓。"""
    if kind == "h3":
        return "/v1/video_generation", "/v1/video_generation/task/{task_id}"
    return "/v1/workflows/run", "/v1/workflows/task/{task_id}"


async def _post_json(url: str, body: dict[str, Any], *, timeout: float = 60) -> dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=body)
        if response.status_code >= 400:
            raise RuntimeError(f"上游 {url} 返回 HTTP {response.status_code}：{response.text[:200]}")
        return response.json() if response.content else {}


async def _get_json(url: str, *, timeout: float = 30) -> dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        if response.status_code >= 400:
            raise RuntimeError(f"上游 {url} 返回 HTTP {response.status_code}：{response.text[:200]}")
        return response.json() if response.content else {}


async def _process_run(run_id: int, body: dict[str, Any]) -> None:
    """跑完一次运行：租约 → 提交 → 轮询 → 取产物落地 → 记账（失败也要收尾 ✓）。"""
    base = service_url()
    started = time.monotonic()
    kind = ""
    #: 进度预览订阅（可选 ✓ 没装 websockets / 调用方没给 ws 地址 ⇒ 一直是 None ✓）
    ws_task: asyncio.Task[bool] | None = None
    #: 是否为「显存不足」真重试过一次（**失败原因里要写明**，否则使用者会以为是别的原因 ✗）
    oom_retried = False
    try:
        with engine.begin() as conn:
            row = get_run(conn, run_id)
        if row is None:
            return
        kind = row.kind
        submit_path, poll_path = _endpoints(kind)
        request_params = _loads(row.params, {})
        # ⚠️ 预览 WS 地址**只能由调用方给**（``comfyuiWsUrl``）：本仓 base 是 8765 薄封装，
        #    它的 /ws 不是 ComfyUI 的 /ws（连错了会一直等到超时，还让人以为是"上游不支持"✗）
        ws_base = str(
            (request_params.get("comfyuiWsUrl") if isinstance(request_params, dict) else "") or ""
        ).strip()

        # ① 本地服务 ⇒ 申请 GPU 租约（串行化 ✓；失败只告警，不中断 ✓ 与 video 链路一致 ✓）
        try:
            _run_leases[run_id] = await gpu_manager.acquire("video", "comfyui",
                                                            row.workflow_name or "workflow", base)
        except Exception as err:  # noqa: BLE001
            log_task_warn("ComfyUIRun", "gpu-acquire-failed", {"id": run_id, "error": str(err)})

        # ② 提交（**先校验**：缺节点包/悬空连线/注入点找不到 ⇒ 现在就说清，别占着 GPU 等报错 ✓）
        _update(run_id, step="validate")
        checked = await validate_workflow_body(body)
        if checked.get("checked") and not checked.get("ok"):
            detail = "；".join(checked.get("problems") or [])[:400] or "工作流校验未通过"
            raise RuntimeError(f"工作流校验未通过：{detail}")
        if not checked.get("checked"):
            log_task_warn("ComfyUIRun", "validate-skipped",
                          {"id": run_id, "reason": checked.get("reason")})
        _update(run_id, status="processing", step="submit")
        log_task_progress("ComfyUIRun", "submit-request", {"id": run_id, "url": f"{base}{submit_path}"})

        # ②′ 提交 → 轮询（**显存不足自动降档一次** ✓）
        #
        # 为什么要有这一条：画布上同档位连跑、或前一次留下卸载不干净的模型时，ComfyUI
        # 的第一反应就是 ``allocation on device ... out of memory`` ✗ —— 而这一句在**换一次
        # 显存**后往往就跑得过 ✓。上游（8765）会替我们 ``/free`` ✓，本层只负责「重试一次 + 记账」
        # ✓：**绝不无限重刷**（每次重提交都是一次真 GPU 占用 ✗，与「崩溃恢复不重提交」同一原则 ✓）。
        async def _submit_and_poll() -> tuple[str, dict[str, Any]]:
            nonlocal ws_task
            created = await _post_json(f"{base}{submit_path}", body)
            task = str(created.get("task_id") or "")
            if not task:
                raise RuntimeError(f"上游未返回 task_id：{created!r}")
            _update(run_id, remote_task_id=task, step="poll")
            progress.publish(run_id, {"type": "poll", "status": "processing", "step": "poll"})
            deadline = time.monotonic() + _run_timeout()
            current: dict[str, Any] = {}
            while True:
                current = await _get_json(f"{base}{poll_path.format(task_id=task)}")
                state = str(current.get("status") or "")
                # 进度桥：**每轮都进快照** ✓ —— 没装 websockets / 上游不给 prompt_id 时，
                # 这就是唯一的数据源（有真进度时会被后面的 progress 事件覆盖 ✓）
                progress.publish(run_id, {"type": "poll", "status": state or "processing",
                                          "step": "poll"})
                if ws_task is None and ws_base:
                    prompt_id = str(current.get("prompt_id") or "")
                    if prompt_id:
                        ws_task = asyncio.create_task(progress.watch_websocket(
                            ws_base, prompt_id, run_id, timeout_seconds=_run_timeout()))
                if state in ("succeeded", "failed", "completed"):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"等待上游完成超时（{_run_timeout():.0f}s，task={task}）")
                await asyncio.sleep(_poll_seconds())
            if state == "failed":
                raise RuntimeError(str(current.get("error_msg") or "上游报告失败（无原因）"))
            return task, current

        while True:
            try:
                task_id, payload = await _submit_and_poll()
                break
            except Exception as err:  # noqa: BLE001 —— 只拦显存不足，其余原样上抛 ✓
                if oom_retried or not is_out_of_memory(err):
                    raise
                oom_retried = True
                _update(run_id, status="processing", step="retry-oom")
                log_task_warn("ComfyUIRun", "oom-retry", {"id": run_id, "error": str(err)[:160]})
                progress.publish(run_id, {"type": "poll", "status": "processing", "step": "retry-oom"})
                # 上游 /v1/free 会卸载模型 + 空转缓存 ✓（放这层是因为它是一次**真实动作**，
                # 不是文案提示 ✓：只说明不留手，使用者照样跑不过 ✗）
                await free_upstream()
                await asyncio.sleep(_oom_retry_wait())
                continue

        # ④ 取产物**落地到我们数据根** ✓
        warnings = list(payload.get("warnings") or [])
        if oom_retried:
            # 重试成功也要记一笔 ✓：否则「这次为什么慢了一倍」在台账里查不到 ✗
            warnings.append("显存不足：已自动卸载上游模型并重试一次后成功（该档位显存余量偏低，"
                            "建议下调分辨率/帧数/时长）")
        _update(run_id, step="download", warnings=_dumps(warnings))
        primary_url, outputs, local_path = await _adopt_outputs(payload)

        elapsed = int((time.monotonic() - started) * 1000)
        used = _as_float(payload.get("expected_seconds")) or row.requested_seconds
        _update(run_id, status="succeeded", step="done", outputs=_dumps(outputs),
                primary_url=primary_url, local_path=local_path, elapsed_ms=elapsed,
                used_seconds=used, error_msg=None)
        # 收尾必须是显式事件 ✓（只推进度不收尾 ⇒ 前端永远转圈 ✗）
        progress.publish(run_id, {"type": "done"})

        # ⑤ 用量记账（与图片/视频同一口径 ✓：units = 秒 ✓）
        with engine.begin() as conn:
            record_usage(conn, {
                "serviceType": "video", "provider": "comfyui",
                "model": row.workflow_name or "workflow",
                "units": int(used or 0), "isLocal": True, "status": "completed",
                # 显存重试真的多占了一次 GPU ⇒ 记账要如实 +1（「重拍成本」看板靠它 ✓）
                "retryCount": 1 if oom_retried else 0, "settings": _loads(row.settings, {}),
                # ⚠️ `meta` 要传**对象**：`record_usage` 自己会 `json.dumps`（与 TS 一致 ✓）
                #    —— 传字符串会变成**双重编码** ✗（自检 ⑧ 当场抓到：meta 里存着 "{\"comfyuiRunId\":1}" ✓）
                "meta": {"comfyuiRunId": run_id, "remoteTaskId": task_id},
            })
        log_task_success("ComfyUIRun", "done", {"id": run_id, "localPath": local_path,
                                                "elapsedMs": elapsed})
    except Exception as err:  # noqa: BLE001 —— 任何失败都要落到 DB，不能挂在后台 ✗
        elapsed = int((time.monotonic() - started) * 1000)
        # 归因口径：**上游原文一字不改** ✓ + 一条能直接动手的建议 ✓（认不出就不加 ✓）
        described = describe_failure(err, oom_retried=oom_retried)
        _update(run_id, status="failed", step="failed",
                error_msg=f"{type(err).__name__}: {described}"[:600], elapsed_ms=elapsed)
        log_task_error("ComfyUIRun", "failed",
                       {"id": run_id, "error": str(err)[:200], "oomRetried": oom_retried})
        progress.publish(run_id, {"type": "error", "message": described[:200]})
    finally:
        # 收尾先停预览订阅 ✓（不 cancel 的话它会一直挂着收别人的任务广播 ✗）
        if ws_task is not None and not ws_task.done():
            ws_task.cancel()
        release_run_lease(run_id)


async def _adopt_outputs(payload: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]], str | None]:
    """把上游产物**下载进我们数据根** ✓；返回 ``(主 URL, 产物清单, 本地相对路径)``。

    * H3 协议：``video_url`` ✓；通用协议：``outputs[].url`` ✓ —— 两种都收 ✓。
    * 只有指向 **http** 的才下载 ✓（本地路径/空值跳过 ✓）；下载失败**不让整次运行失败** ✗
      —— 保留远端 URL 并记 warning ✓（产物还在上游，可重取 ✓）。
    """
    outputs: list[dict[str, Any]] = []
    for item in payload.get("outputs") or []:
        if isinstance(item, dict):
            outputs.append(dict(item))
    primary_url = str(payload.get("video_url") or "")
    if not primary_url:
        primary_url = next((str(x.get("url")) for x in outputs
                            if x.get("kind") == "videos" and x.get("url")), "")
    local_path: str | None = None
    if primary_url.startswith(("http://", "https://")):
        try:
            local_path = await download_file(primary_url, "comfyui")
        except Exception as err:  # noqa: BLE001
            log_task_warn("ComfyUIRun", "artifact-download-failed",
                          {"url": primary_url, "error": str(err)[:160]})
    return primary_url or None, outputs, local_path


def release_run_lease(run_id: int) -> None:
    """释放该运行的 GPU 租约（幂等 ✓ 与 image/video 的 ``release_*_gpu_lease`` 同形 ✓）。"""
    lease = _run_leases.pop(run_id, None)
    if lease is not None:
        try:
            lease.release()
        except Exception as err:  # noqa: BLE001
            log_task_warn("ComfyUIRun", "lease-release-failed", {"id": run_id, "error": str(err)})


def recover_runs_on_startup() -> None:
    """**崩溃恢复**：把中断的运行收尾（有 ``remote_task_id`` 的说明可续询 ✓，没有的判 failed ✓）。

    ⚠️ 刻意**不重提交** ✗ —— 重提交会再占一次 GPU（与视频链路「绝不重复提交」同一原则 ✓）。
    在途任务（``queued`` 且没提交成功）直接判 failed 并说明「进程重启中断」✓。
    """
    with engine.begin() as conn:
        rows = list(conn.execute(
            select(comfyui_runs).where(comfyui_runs.c.status.in_(("queued", "processing")))).all())
        for row in rows:
            has_remote = bool(row.remote_task_id)
            reason = ("进程重启中断：已提交上游（task=%s），产物可在 8765 服务查，本项目记录标为中断 ✓"
                      % row.remote_task_id) if has_remote else "进程重启中断：提交前即中断，未占用 GPU ✓"
            conn.execute(comfyui_runs.update().where(comfyui_runs.c.id == row.id).values(
                status="failed", step="recovered", error_msg=reason, updated_at=now()))
            log_task_warn("ComfyUIRun", "recover-failed", {"id": row.id, "remoteTaskId": row.remote_task_id})


# ══════════════════════════════════════════════════════════════════════════
# 保留清理（过期任务 + 它们的本地产物 ✓）
# ══════════════════════════════════════════════════════════════════════════
#: 默认保留天数（``COMFYUI_RETENTION_DAYS`` 可改；``<= 0`` = 不清理 ✓）
RETENTION_DAYS = 30

#: **只有这些状态**算终结、可以清 ✓ —— ``queued`` / ``processing`` 一律不动 ✗
TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")

#: 允许删的产物前缀（``download_file(url, "comfyui")`` 的落点 ✓）。
#: 别的路径**整行留下** ✗：路径不认识就说明它不是这条路子落的，宁可不清也不误删 ✓
PURGE_PATH_PREFIXES = ("static/comfyui/", "static\\comfyui\\")


def purge_runs(
    conn: Connection,
    *,
    days: int | None = None,
    statuses: tuple[str, ...] | list[str] = TERMINAL_STATUSES,
    dry_run: bool = True,
    now_ts: datetime | None = None,
) -> dict[str, Any]:
    """清掉**过期**的运行记录与它们的本地产物 ✓，``dry_run=True`` 只算不删 ✓。

    为什么这条必须有守卫（删错是不可逆的 ✗）：

    1. **在跑的绝不碰** ✗：``queued`` / ``processing`` 的记录一删，后台任务还在跑 ⇒
       产物落了盘却**没有任何记录指向它**（孤儿文件，再也没人知道该删）✗；
    2. **行与产物同生共死** ✓：文件删不掉（占用/权限）时**这一行也不删** ✓ ——
       先删行会让「该删哪个文件」这条线索永久丢失 ✗；
       反过来「文件早就不在了」不算错（``missingFiles`` 照实计数 ✓、行照删 ✓）；
    3. **共享产物不删** ✗：同一条相对路径还被**别的行**（比如另一次运行 adopted 了同一个
       本地产物）引用时留着 ✓，报 ``keptShared``；
    4. **不认识的路径不动** ✗：只删 :data:`PURGE_PATH_PREFIXES` 下的东西，
       其余路径**连行一起留下**并报 ``keptOutside``（宁可不清，不许误删 ✗）；
    5. **默认干跑** ✓：``dry_run=False`` 才真删（脚本里对应 ``--apply`` ✓）。

    返回的是**账**，不是布尔：``considered`` 过期行数 / ``runs`` 会删的行数 /
    ``files`` 会删的文件数 / ``bytes`` 释放字节 / ``missingFiles`` / ``keptShared`` /
    ``keptOutside`` / ``keptRunning`` / ``failed``。
    """
    if days is None:
        # 环境变量也认 ✓（脚本/接口都不传时才有意义；传了就以传的为准 ✓，不静默覆盖参数 ✗）
        raw = os.environ.get("COMFYUI_RETENTION_DAYS", "").strip()
        try:
            keep_days = int(raw) if raw else RETENTION_DAYS
        except ValueError:
            keep_days = RETENTION_DAYS
    else:
        keep_days = int(days)
    report: dict[str, Any] = {
        "days": keep_days,
        "dryRun": bool(dry_run),
        "statuses": list(statuses),
    }
    if keep_days <= 0:
        report.update({"cutoff": None, "considered": 0, "runs": 0, "files": 0, "bytes": 0,
                       "missingFiles": 0, "keptShared": [], "keptOutside": [], "keptRunning": [],
                       "failed": [], "note": "保留天数为 0 ⇒ 不清理（要清就给正数）"})
        return report

    stamp = now_ts or datetime.now(timezone.utc)
    cutoff = (stamp - timedelta(days=keep_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    report["cutoff"] = cutoff

    rows = list(conn.execute(select(comfyui_runs)).all())
    expired = [row for row in rows if (row.created_at or "") < cutoff]
    report["considered"] = len(expired)
    report["keptRunning"] = [
        {"id": row.id, "status": row.status, "createdAt": row.created_at}
        for row in expired
        if row.status not in statuses
    ]
    candidates = [row for row in expired if row.status in statuses]

    # 共享核账：留下来的行（含未过期 / 在跑）引用的路径一律不许删 ✓
    doomed_ids = {row.id for row in candidates}
    survivors: dict[str, list[int]] = {}
    for row in rows:
        if row.id in doomed_ids or not row.local_path:
            continue
        survivors.setdefault(str(row.local_path), []).append(row.id)

    plan: list[dict[str, Any]] = []
    kept_shared: list[dict[str, Any]] = []
    kept_outside: list[dict[str, Any]] = []
    for row in candidates:
        path = str(row.local_path) if row.local_path else ""
        if path and not path.startswith(PURGE_PATH_PREFIXES):
            kept_outside.append({"id": row.id, "localPath": path})
            continue
        if path and path in survivors:
            kept_shared.append({"localPath": path, "ids": [row.id, *survivors[path]]})
            continue
        plan.append({"id": row.id, "localPath": path or None})

    files: list[str] = []
    size = 0
    missing = 0
    for item in plan:
        path = item["localPath"]
        if not path:
            continue
        try:
            absolute = Path(get_absolute_path(path))
        except ValueError:
            kept_outside.append({"id": item["id"], "localPath": path, "why": "路径不在数据根内"})
            item["skip"] = True
            continue
        if not absolute.is_file():
            missing += 1
            continue
        files.append(str(absolute))
        size += absolute.stat().st_size

    report.update({
        "runs": len([item for item in plan if not item.get("skip")]),
        "files": len(files),
        "bytes": size,
        "missingFiles": missing,
        "keptShared": kept_shared,
        "keptOutside": kept_outside,
        "failed": [],
        "planned": [item for item in plan if not item.get("skip")],
    })
    if dry_run:
        return report

    deleted_files = 0
    failed: list[dict[str, Any]] = []
    for path in files:
        try:
            os.remove(path)
            deleted_files += 1
        except OSError as err:
            failed.append({"path": path, "error": str(err)[:160]})

    # 只有「它的产物一个都没失败」的行才删 ✓（半删 = 留下一条指向空处的记录 ✗）
    failed_paths = {str(Path(item["path"])) for item in failed}
    removable: list[int] = []
    for item in plan:
        if item.get("skip"):
            continue
        path = item["localPath"]
        absolute = str(Path(get_absolute_path(path))) if path else ""
        if absolute and absolute in failed_paths:
            continue
        removable.append(item["id"])

    removed = 0
    for run_id in removable:
        conn.execute(comfyui_runs.delete().where(comfyui_runs.c.id == run_id))
        removed += 1

    report.update({"runs": removed, "files": deleted_files, "failed": failed,
                   "keptFailed": [item for item in plan if item["id"] not in set(removable)]})
    if removed or deleted_files:
        # 什么都没清就别往日志里写一行「DONE purge」✗ —— 那会让人以为刚清过东西 ✓
        log_task_success("ComfyUIRun", "purge", {"runs": removed, "files": deleted_files,
                                                 "bytes": size, "days": keep_days})
    return report


# ══════════════════════════════════════════════════════════════════════════
# 上游能力（仍然只在服务层做 HTTP ✓ 路由/接口层保持薄壳 ✓）
# ══════════════════════════════════════════════════════════════════════════
async def upstream_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{service_url()}{path}", params=params or {})
        response.raise_for_status()
        return response.json() if response.content else {}


async def upstream_post(path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{service_url()}{path}", json=body or {})
        response.raise_for_status()
        return response.json() if response.content else {}


async def interrupt_upstream() -> bool:
    """让上游打断当前执行（取消运行的一半 ✓；另一半在本地标记 ✓）。"""
    try:
        return bool((await upstream_post("/v1/interrupt", {})).get("ok", True))
    except Exception as err:  # noqa: BLE001
        log_task_warn("ComfyUIRun", "interrupt-failed", {"error": str(err)[:160]})
        return False


async def free_upstream() -> bool:
    """让上游卸载模型/释放显存 ✓。"""
    try:
        return bool((await upstream_post("/v1/free", {"unload_models": True, "free_memory": True})).get("ok", True))
    except Exception as err:  # noqa: BLE001
        log_task_warn("ComfyUIRun", "free-failed", {"error": str(err)[:160]})
        return False


def stats(conn: Connection) -> dict[str, Any]:
    """**我们自己的**运行统计（上游没有这个概念 ✓）—— 给「系统」页做概览用 ✓。"""
    rows = list(conn.execute(select(comfyui_runs.c.status, comfyui_runs.c.elapsed_ms,
                                    comfyui_runs.c.created_at)).all())
    counts: dict[str, int] = {}
    elapsed: list[int] = []
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
        if row.elapsed_ms:
            elapsed.append(int(row.elapsed_ms))
    latest = max((row.created_at for row in rows), default=None)
    return {
        "total": len(rows),
        "byStatus": counts,
        "avgElapsedMs": round(sum(elapsed) / len(elapsed)) if elapsed else None,
        "lastRunAt": latest,
    }


async def system_snapshot(conn: Connection) -> dict[str, Any]:
    """「系统」视图 = **上游体检** + **我们自己的运行统计** ✓（合并过，不是转发 ✗）。"""
    try:
        upstream = await upstream_get("/v1/system")
        reachable = True
        error = None
    except Exception as err:  # noqa: BLE001
        upstream, reachable, error = {}, False, f"{type(err).__name__}: {err}"[:200]
    return {
        "upstream": {"url": service_url(), "reachable": reachable, "error": error, **upstream},
        "runs": stats(conn),
        "hint": ("上游不可达 ⇒ 先起本地服务：python backend-py/app/local_services/h3/server.py "
                 "（端口 8765，它再去连 ComfyUI 8188 ✓）"),
    }


async def cancel_run(conn: Connection, run_id: int) -> Row | None:
    """取消一次运行：**先让上游打断**，再在本地标记（两半都做才算真取消 ✓）。"""
    row = get_run(conn, run_id)
    if row is None:
        return None
    if row.status in ("queued", "processing"):
        interrupted = await interrupt_upstream() if row.remote_task_id else False
        _update(run_id, status="failed", step="cancelled",
                error_msg=("用户取消：已请求上游打断 ✓" if interrupted
                           else "用户取消（尚未提交到上游，未占用 GPU ✓）"))
        # 取消也算收尾 ✓：不推这条，订阅端只能等超时（与源 ``execution_interrupted`` 同口径 ✓）
        progress.publish(run_id, {"type": "cancelled"})
        release_run_lease(run_id)
        log_task_warn("ComfyUIRun", "cancelled", {"id": run_id, "interrupted": interrupted})
    return get_run(conn, run_id)
