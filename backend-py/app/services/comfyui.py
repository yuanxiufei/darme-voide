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
**错误归因** | 失败写 ``error_msg``（上游给的原因原样保留 + 我们的注解 ✓），并区分「缺节点 / 服务不可达 / 超时」✓ |

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
import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection, Row

from ..core.config import get_storage_root
from ..core.db import engine
from ..core.models import comfyui_runs
from ..core.response import now
from .file_storage import download_file
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
    "cancel_run",
    "free_upstream",
    "get_run",
    "interrupt_upstream",
    "list_runs",
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
# 入队
# ══════════════════════════════════════════════════════════════════════════
async def submit_run(conn: Connection, params: dict[str, Any]) -> int:
    """入队一次工作流运行，**立即返回 id**（后台跑 ✓ 与 generation 链路同形 ✓）。

    ``params``：
      * ``kind``：``h3``（用 H3 薄封装的视频协议 ✓）或 ``workflow``（任意工作流 ✓，默认）
      * ``workflow`` / ``workflowPath``：UI 或 API 格式的工作流（``kind=workflow`` 时必填 ✓）
      * ``prompt`` / ``duration`` / ``aspectRatio`` / ``firstFrameImage`` / ``lastFrameImage``：
        传给上游（H3 用 ✓；workflow 也可带，由上游决定要不要用 ✓）
      * ``inject``：显式参数注入清单（``[{class_type, field, value, title?}]`` ✓）
      * ``settings``：交给上游的额外设置（如 ``checkpoint_map`` ✓）
    """
    ts = now()
    kind = str(params.get("kind") or "workflow")
    body = _build_upstream_body(kind, params)
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
        if params.get("settings"):
            body["settings"] = params["settings"]
        return body

    body = {}
    if params.get("workflow") is not None:
        body["workflow"] = params["workflow"]
    if params.get("workflowPath"):
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
    try:
        with engine.begin() as conn:
            row = get_run(conn, run_id)
        if row is None:
            return
        kind = row.kind
        submit_path, poll_path = _endpoints(kind)

        # ① 本地服务 ⇒ 申请 GPU 租约（串行化 ✓；失败只告警，不中断 ✓ 与 video 链路一致 ✓）
        try:
            _run_leases[run_id] = await gpu_manager.acquire("video", "comfyui",
                                                            row.workflow_name or "workflow", base)
        except Exception as err:  # noqa: BLE001
            log_task_warn("ComfyUIRun", "gpu-acquire-failed", {"id": run_id, "error": str(err)})

        # ② 提交
        _update(run_id, status="processing", step="submit")
        log_task_progress("ComfyUIRun", "submit-request", {"id": run_id, "url": f"{base}{submit_path}"})
        created = await _post_json(f"{base}{submit_path}", body)
        task_id = str(created.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"上游未返回 task_id：{created!r}")
        _update(run_id, remote_task_id=task_id, step="poll")

        # ③ 轮询（超时/失败都收尾 ✓）
        deadline = time.monotonic() + _run_timeout()
        payload: dict[str, Any] = {}
        while True:
            payload = await _get_json(f"{base}{poll_path.format(task_id=task_id)}")
            status = str(payload.get("status") or "")
            if status in ("succeeded", "failed", "completed"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"等待上游完成超时（{_run_timeout():.0f}s，task={task_id}）")
            await asyncio.sleep(_poll_seconds())

        if status == "failed":
            raise RuntimeError(str(payload.get("error_msg") or "上游报告失败（无原因）"))

        # ④ 取产物**落地到我们数据根** ✓
        _update(run_id, step="download", warnings=_dumps(payload.get("warnings") or []))
        primary_url, outputs, local_path = await _adopt_outputs(payload)

        elapsed = int((time.monotonic() - started) * 1000)
        used = _as_float(payload.get("expected_seconds")) or row.requested_seconds
        _update(run_id, status="succeeded", step="done", outputs=_dumps(outputs),
                primary_url=primary_url, local_path=local_path, elapsed_ms=elapsed,
                used_seconds=used, error_msg=None)

        # ⑤ 用量记账（与图片/视频同一口径 ✓：units = 秒 ✓）
        with engine.begin() as conn:
            record_usage(conn, {
                "serviceType": "video", "provider": "comfyui",
                "model": row.workflow_name or "workflow",
                "units": int(used or 0), "isLocal": True, "status": "completed",
                "retryCount": 0, "settings": _loads(row.settings, {}),
                # ⚠️ `meta` 要传**对象**：`record_usage` 自己会 `json.dumps`（与 TS 一致 ✓）
                #    —— 传字符串会变成**双重编码** ✗（自检 ⑧ 当场抓到：meta 里存着 "{\"comfyuiRunId\":1}" ✓）
                "meta": {"comfyuiRunId": run_id, "remoteTaskId": task_id},
            })
        log_task_success("ComfyUIRun", "done", {"id": run_id, "localPath": local_path,
                                                "elapsedMs": elapsed})
    except Exception as err:  # noqa: BLE001 —— 任何失败都要落到 DB，不能挂在后台 ✗
        elapsed = int((time.monotonic() - started) * 1000)
        _update(run_id, status="failed", step="failed", error_msg=f"{type(err).__name__}: {err}"[:600],
                elapsed_ms=elapsed)
        log_task_error("ComfyUIRun", "failed", {"id": run_id, "error": str(err)[:200]})
    finally:
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
        release_run_lease(run_id)
        log_task_warn("ComfyUIRun", "cancelled", {"id": run_id, "interrupted": interrupted})
    return get_run(conn, run_id)
