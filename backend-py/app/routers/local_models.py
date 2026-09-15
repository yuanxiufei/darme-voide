"""本地模型路由 —— 与 ``routes/localModels.ts``（688 行）对齐。**11 端点**。

* ``GET  /api/v1/local-models/scan``          扫描模型（``roots/kinds/maxDepth/maxFiles``）
* ``GET  /api/v1/local-models/roots``         默认扫描根目录 + 探测状态
* ``PUT  /api/v1/local-models/roots``         保存模型路径配置（partial 更新）
* ``POST /api/v1/local-models/register``      注册到 ``ai_service_configs``
* ``GET  /api/v1/local-models/drives``        列出磁盘盘符
* ``POST /api/v1/local-models/scan``          启动**异步**扫描任务
* ``GET  /api/v1/local-models/scan/status``   查询任务进度（``?taskId=``）
* ``POST /api/v1/local-models/scan/cancel``   取消任务（``?taskId=``）
* ``POST /api/v1/local-models/hf/files``      列 HF/ModelScope 仓库文件
* ``POST /api/v1/local-models/hf/download``   下载到模型存储目录（**NDJSON 流式**）
* ``POST /api/v1/local-models/delete``        删除存储目录内的模型文件/目录

⚠️ 七处保真点：

1. **`maxDepth` / `maxFiles` 是 camelCase 查询参数**（第 7 道守卫在盯）⇒ 必须
   ``Query(alias=...)``，否则被 FastAPI **静默忽略**（蛇形形参收不到、有默认值不报错）；
2. ``Number(raw) || 默认值`` 的 JS 语义：``NaN``/``0``/空串都落到默认值（``maxDepth=5``、
   ``maxFiles=8000``），**不是**「非法即报错」；
3. ``roots`` 的「空数组但非 undefined」要回 **400**（``roots=[]`` 或 ``,`` 都会解析成空数组）；
4. ``/scan``（异步版）上限比同步版宽松得多（``8 / 50000``），且任务是**内存态**、
   完成后保留 **10 分钟**供前端拉结果；
5. 下载**优先跳过已存在且够大的文件**、支持 ``.part`` **断点续传**（206 续传 / 200 重下）、
   失败**保留 .part** 供下次续传；``files`` 里写了仓库中不存在的文件 ⇒ **400**；
6. ``ModelScope`` 先取**签名 URL** 下载 LFS 大文件，失败回退 ``/resolve/`` 302 直链；
7. 删除一律**限制在模型存储目录内**（防路径穿越），且**不允许删根目录本身**。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from ..core.db import engine
from ..core.models import ai_service_configs
from ..core.request_utils import read_json
from ..core.response import bad_request, now, success
from ..services.ai_configs import map_config_row, parse_settings_object
from ..services.local_model_scan import (
    ScanCancelledError,
    get_default_roots,
    get_extra_roots,
    get_model_paths,
    list_drives,
    save_extra_roots,
    save_model_paths,
    scan_local_models,
    scan_local_models_async,
)
from ..services.task_logger import log_task_error, log_task_success

router = APIRouter(prefix="/api/v1/local-models", tags=["local-models"])

#: 合法大类（对齐 ``VALID_KINDS``）
VALID_KINDS = ["text", "image", "video", "audio", "unknown"]

#: 异步扫描任务的**内存保留时长**（秒；原 TS 是 ``setTimeout(..., 10 * 60 * 1000)``）
SCAN_TASK_TTL_SECONDS = 10 * 60

#: 内存任务表（本地单用户，任务量小，无需持久化）
_scan_tasks: dict[str, dict[str, Any]] = {}


# ============================================================
# 参数解析
# ============================================================

def _js_number_or(raw: str | None, fallback: float) -> float:
    """``Number(raw) || fallback`` 的 JS 语义。

    ⚠️ ``Number('abc')`` 是 ``NaN``、``Number('')`` 是 ``0`` ⇒ 两者都是 falsy ⇒ 回退默认值。
    """
    if raw is None:
        return fallback
    text = raw.strip()
    if not text:
        return fallback
    try:
        value = float(text)
    except ValueError:
        return fallback
    if value != value:  # NaN
        return fallback
    return value if value else fallback


def parse_roots(raw: str | None) -> list[str] | None:
    """解析 ``roots``：JSON 数组字符串，或逗号分隔。**区分 None 与空数组**（见路由的 400）。"""
    if not raw:
        return None
    trimmed = raw.strip()
    if not trimmed:
        return None
    if trimmed.startswith("["):
        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, str) and x]
        except ValueError:
            pass  # 降级为逗号分隔
    return [part.strip() for part in trimmed.split(",") if part.strip()]


def parse_kinds(raw: str | None) -> list[str] | None:
    """解析 ``kinds``（逗号分隔、小写化、只保留合法值；全非法则视为未传）。"""
    if not raw:
        return None
    kinds = [part.strip().lower() for part in raw.split(",")]
    valid = [k for k in kinds if k in VALID_KINDS]
    return valid or None


# ============================================================
# 阶段 1：只读扫描
# ============================================================

@router.get("/scan")
def scan_models(
    roots: str | None = None,
    kinds: str | None = None,
    max_depth_raw: str | None = Query(None, alias="maxDepth"),
    max_files_raw: str | None = Query(None, alias="maxFiles"),
):
    """扫描本地模型（**同步**，跑在线程池里，不阻塞事件循环）。"""
    parsed_roots = parse_roots(roots)
    parsed_kinds = parse_kinds(kinds)
    # ⚠️ 两个参数刻意声明为 **str** 再自己解析：原 TS 是 ``Number(raw) || 默认值``，
    #    非法值（``?maxDepth=abc``）应当**回退到默认值**，而不是让 FastAPI 抛 422。
    depth = _js_number_or(max_depth_raw, 5)
    files = _js_number_or(max_files_raw, 8000)

    if parsed_roots is not None and len(parsed_roots) == 0:
        return bad_request("roots 参数为空或格式不正确")

    result = scan_local_models({"roots": parsed_roots, "kinds": parsed_kinds,
                                "maxDepth": depth, "maxFiles": files})
    return success(result)


@router.get("/roots")
def get_roots():
    """返回默认扫描根目录及探测状态（含 ``extra`` 与 ``paths``）。"""
    roots = get_default_roots()
    detail = []
    for root in roots:
        size_hint = ""
        try:
            if os.path.isdir(root):
                size_hint = os.path.basename(root)  # 仅展示目录名，避免遍历磁盘
        except OSError:
            pass
        detail.append({"path": root, "name": size_hint or os.path.basename(root)})
    return success({"roots": roots, "detail": detail,
                    "extra": get_extra_roots(), "paths": get_model_paths()})


@router.put("/roots")
async def put_roots(request: Request):
    """保存模型路径配置（**partial 更新**：``roots`` 额外扫描目录，``models_dir`` 存储目录）。"""
    try:
        body = await read_json(request)
        extra = get_extra_roots()

        # 额外扫描目录：仅当显式传入 roots 数组时才更新（否则保留现状）
        if isinstance(body.get("roots"), list):
            extra = save_extra_roots([x for x in body["roots"]
                                      if isinstance(x, str) and x.strip()])

        # 模型存储目录：仅当显式传入 models_dir 时才更新
        if body.get("models_dir") is not None:
            save_model_paths({"models_dir": body["models_dir"]})

        log_task_success("LocalModels", "saveRoots", {
            "extra": len(extra),
            "models_dir": body.get("models_dir", "(unchanged)"),
        })
        return success({"extra": extra, "roots": get_default_roots(),
                        "paths": get_model_paths()})
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("LocalModels", "saveRoots", {"error": str(err)})
        return bad_request(str(err))


# ============================================================
# 阶段 2：注册到 ai_service_configs
# ============================================================

SERVICE_LABELS = {"text": "文本", "image": "图像", "video": "视频", "audio": "音频"}


def h3_checkpoint_key(filename: str) -> str | None:
    """从文件名派生 H3 ``checkpoint_map`` 键（fl2va / ref2va）。"""
    name = filename.lower()
    if "fl2va" in name:
        return "fl2va"
    if "ref2va" in name:
        return "ref2va"
    return None


@router.post("/register")
async def register_models(request: Request):
    """注册本地模型。

    * ``standalone`` 模型：按 ``(serviceType|provider|baseUrl)`` 去重合并后 upsert；
    * ``component`` 模型：不单独注册，返回 ``skipped`` 并说明原因；
    * H3 视频 DiT：从文件名派生 ``fl2va/ref2va`` 写进 ``settings.checkpoint_map``。
    """
    try:
        body = await read_json(request)
        models = body.get("models") if isinstance(body.get("models"), list) else []
        override_name = body.get("name").strip() if isinstance(body.get("name"), str) else ""

        if not models:
            return bad_request("models 不能为空")

        ts = now()
        created: list[str] = []
        updated: list[str] = []
        skipped: list[dict[str, str]] = []

        # 按去重键分组 standalone 模型
        groups: dict[str, list[dict[str, Any]]] = {}
        for model in models:
            suggested = (model or {}).get("suggested")
            role = (model or {}).get("role") or (suggested or {}).get("role")
            raw_name = (model or {}).get("filename")
            filename = (raw_name if isinstance(raw_name, str) and raw_name
                        else os.path.basename((model or {}).get("path") or ""))

            if role == "component":
                skipped.append({"filename": filename, "reason": (suggested or {}).get("note")
                                or "组件模型，需在 ComfyUI 工作流中引用，不单独注册"})
                continue
            # ComfyUI 仅作「只读的可选扫描来源」，不注册为可调用后端
            if (suggested or {}).get("runtime") == "comfyui" or (suggested or {}).get("callable") is False:
                skipped.append({"filename": filename, "reason": (suggested or {}).get("note")
                                or "ComfyUI 仅作只读扫描来源，不直接调用"})
                continue
            if not suggested:
                skipped.append({"filename": filename, "reason": "无法识别模型类型，未生成注册建议"})
                continue
            key = f"{suggested.get('serviceType')}|{suggested.get('provider')}|{suggested.get('baseUrl')}"
            groups.setdefault(key, []).append({"suggested": suggested, "filename": filename})

        with engine.begin() as conn:
            for key, items in groups.items():
                service_type, provider, base_url = key.split("|")
                first = items[0]["suggested"]

                # 收集 model 名（去重）与 H3 checkpoint_map
                model_names: list[str] = []
                checkpoint_map: dict[str, str] = {}
                for item in items:
                    model_name = item["suggested"].get("model")
                    if model_name and model_name not in model_names:
                        model_names.append(model_name)
                    if first.get("runtime") == "h3" and service_type == "video":
                        checkpoint = h3_checkpoint_key(item["filename"])
                        if checkpoint:
                            checkpoint_map[checkpoint] = re.sub(r"\.[^.]+$", "", item["filename"])

                name = override_name or f"本地{SERVICE_LABELS.get(service_type, service_type)}服务"

                # 查现有同 serviceType + provider + baseUrl 配置
                existing = [row for row in conn.execute(
                    select(ai_service_configs)
                    .where(ai_service_configs.c.service_type == service_type)
                ).all() if row.provider == provider and row.base_url == base_url]
                row = existing[0] if existing else None
                is_new = not existing

                # 合并 settings（checkpoint_map）
                settings = parse_settings_object(row.settings if row is not None else None)
                if checkpoint_map:
                    settings = {
                        **settings,
                        "checkpoint_map": {**(settings.get("checkpoint_map") or {}), **checkpoint_map},
                    }

                # 合并 model 数组（去重，保序）
                merged_models = model_names
                if row is not None and row.model:
                    try:
                        previous = json.loads(row.model)
                        if isinstance(previous, list):
                            merged_models = list(dict.fromkeys([*previous, *model_names]))
                    except ValueError:
                        pass  # 保留新模型名

                values = {
                    "service_type": service_type,
                    "provider": provider,
                    "name": name,
                    "base_url": base_url,
                    "api_key": (row.api_key if row is not None else None) or "local",
                    "model": json.dumps(merged_models, ensure_ascii=False, separators=(",", ":")),
                    "priority": (row.priority if row is not None and row.priority is not None else 80),
                    "settings": json.dumps(settings, ensure_ascii=False, separators=(",", ":")),
                    "is_active": True,
                    "updated_at": ts,
                }

                if is_new:
                    conn.execute(ai_service_configs.insert().values(**values, created_at=ts))
                    created.append(name)
                else:
                    conn.execute(ai_service_configs.update()
                                 .where(ai_service_configs.c.id == row.id).values(**values))
                    updated.append(name)

        log_task_success("LocalModels", "register", {
            "created": len(created), "updated": len(updated), "skipped": len(skipped),
        })

        with engine.begin() as conn:
            configs = [map_config_row(r)
                       for r in conn.execute(select(ai_service_configs)).all()]

        return success({"created": created, "updated": updated,
                        "skipped": skipped, "configs": configs})
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("LocalModels", "register", {"error": str(err)})
        return bad_request(str(err))


# ============================================================
# 阶段 3：磁盘/全盘异步扫描（后台任务 + 进度轮询 + 取消）
# ============================================================

@router.get("/drives")
def get_drives():
    """列出可用磁盘盘符。"""
    return success({"drives": list_drives()})


async def _run_scan_task(task_id: str, state: dict[str, Any], opts: dict[str, Any]) -> None:
    """后台扫描任务体（成功/取消/失败都收尾到 ``done``，并**延迟 10 分钟**清表）。"""
    loop = asyncio.get_running_loop()

    def _cleanup() -> None:
        # 完成后保留 10 分钟供前端拉取结果，之后清理
        loop.call_later(SCAN_TASK_TTL_SECONDS, _scan_tasks.pop, task_id, None)

    try:
        result = await scan_local_models_async({
            **opts,
            "onProgress": lambda progress: state.update(progress=progress),
            "shouldCancel": lambda: bool(state.get("cancelled_flag")),
        })
        state["progress"] = {**state["progress"], "done": True, "result": result}
    except ScanCancelledError:
        state["progress"] = {**state["progress"], "done": True, "cancelled": True}
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        state["progress"] = {**state["progress"], "done": True, "error": str(err)}
    finally:
        _cleanup()


@router.post("/scan")
async def start_scan(request: Request):
    """启动异步扫描任务（body 可传 ``roots/kinds/maxDepth/maxFiles``）。"""
    try:
        body = await read_json(request)
        raw_roots = body.get("roots")
        roots = ([x for x in raw_roots if isinstance(x, str) and x.strip()]
                 if isinstance(raw_roots, list) else None)
        raw_kinds = body.get("kinds")
        kinds = ([k for k in raw_kinds if k in VALID_KINDS]
                 if isinstance(raw_kinds, list) else None)
        # ⚠️ `Number(body.maxDepth) > 0 ? ... : undefined`（NaN/0/负数都视为未传）
        depth = body.get("maxDepth")
        max_depth = depth if _positive_number(depth) else None
        files = body.get("maxFiles")
        max_files = files if _positive_number(files) else None

        if roots is not None and len(roots) == 0:
            return bad_request("roots 参数为空或格式不正确")

        task_id = str(uuid.uuid4())
        state: dict[str, Any] = {
            "id": task_id,
            "cancelled_flag": False,
            "progress": {
                "scannedFiles": 0,
                "foundModels": 0,
                "currentDir": "",
                "done": False,
                "cancelled": False,
            },
            "cancel": lambda: state.update(cancelled_flag=True),
        }
        _scan_tasks[task_id] = state

        # ⚠️ 用 get_running_loop()（不是 ensure_future）—— 见 auto_pipeline._spawn 的注释
        asyncio.get_running_loop().create_task(_run_scan_task(task_id, state, {
            "roots": roots, "kinds": kinds, "maxDepth": max_depth, "maxFiles": max_files,
        }))

        return success({"taskId": task_id})
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        return bad_request(str(err))


def _positive_number(value: Any) -> bool:
    """``Number(value) > 0``（非数字/NaN/0/负数 → False）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number > 0


@router.get("/scan/status")
def scan_status(task_id: str | None = Query(None, alias="taskId")):
    """查询扫描任务进度/结果。"""
    if not task_id:
        return bad_request("taskId 不能为空")
    state = _scan_tasks.get(task_id)
    if state is None:
        return bad_request("任务不存在或已过期")
    return success({"taskId": task_id, "progress": state["progress"]})


@router.post("/scan/cancel")
def cancel_scan(task_id: str | None = Query(None, alias="taskId")):
    """取消扫描任务（协作式取消：扫描循环检查标志后中止）。"""
    if not task_id:
        return bad_request("taskId 不能为空")
    state = _scan_tasks.get(task_id)
    if state is None:
        return bad_request("任务不存在或已过期")
    state["cancel"]()
    return success({"taskId": task_id, "cancelled": True})


# ============================================================
# 阶段 4：HuggingFace / ModelScope 权重下载
# ============================================================

#: 下载来源：HF 官方 / hf-mirror 国内镜像 / ModelScope 魔搭
SOURCES = {
    "hf": {"base": "https://huggingface.co", "label": "HuggingFace 官方"},
    "hf_mirror": {"base": "https://hf-mirror.com", "label": "hf-mirror 镜像"},
    "modelscope": {"base": "https://modelscope.cn", "label": "ModelScope 魔搭"},
}


def parse_source(raw: Any) -> str:
    """解析下载来源参数，**非法值回退为 hf**。"""
    text = str(raw or "").strip().lower()
    return text if text in ("hf", "hf_mirror", "modelscope") else "hf"


def default_revision(source: str, revision: Any) -> str:
    """默认分支：ModelScope 用 ``master``，HF 系列用 ``main``。"""
    text = str(revision or "").strip()
    if text:
        return text
    return "master" if source == "modelscope" else "main"


def normalize_repo(raw: Any) -> str:
    """校验并归一化仓库名（``owner/name``），**防路径穿越**。"""
    repo = str(raw or "").strip().strip("/")
    if not repo or "/" not in repo:
        raise ValueError("仓库名格式应为 owner/name，例如 Qwen/Qwen3-4B")
    if ".." in repo or "\\" in repo:
        raise ValueError("仓库名不合法")
    return repo


async def fetch_json(url: str, timeout_ms: int = 30_000) -> Any:
    """通用 JSON 请求（失败抛错并**截断**错误信息到 400 字符）。"""
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_ms / 1000) as client:
        response = await client.get(url, headers={"Accept": "application/json"})
        if not response.is_success:
            text = response.text or ""
            raise ValueError((text or f"HTTP {response.status_code}")[:400])
        try:
            return response.json()
        except ValueError:
            return None


async def list_hf_files(repo: str, source: str) -> list[dict[str, Any]]:
    """通过 HF 系列 API 列出仓库文件（含大小）。"""
    base = SOURCES[source]["base"]
    payload = await fetch_json(f"{base}/api/models/{repo}?blobs=true")
    raw_list = []
    if isinstance(payload, dict):
        for key in ("files", "siblings"):
            if isinstance(payload.get(key), list):
                raw_list = payload[key]
                break
    files: list[dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        name = str(item.get("rfilename") or "")
        if name and not name.endswith("/"):
            files.append({"name": name, "size": _number(item.get("size"))})
    return files


async def list_modelscope_files(repo: str, revision: str) -> list[dict[str, Any]]:
    """通过 ModelScope API 列出仓库文件（仅 ``blob``）。"""
    url = (f"https://modelscope.cn/api/v1/models/{repo}/repo/files"
           f"?Revision={_quote(revision)}&Recursive=true")
    payload = await fetch_json(url)
    raw_files = []
    data = payload.get("Data") if isinstance(payload, dict) else None
    if isinstance(data, dict) and isinstance(data.get("Files"), list):
        raw_files = data["Files"]
    files: list[dict[str, Any]] = []
    for item in raw_files:
        if not isinstance(item, dict) or item.get("Type") != "blob":
            continue
        path = str(item.get("Path") or "")
        if path:
            files.append({"name": path, "size": _number(item.get("Size"))})
    return files


def list_remote_files(repo: str, source: str, revision: str) -> Any:
    """按来源列出仓库文件（返回 awaitable）。"""
    if source == "modelscope":
        return list_modelscope_files(repo, revision)
    return list_hf_files(repo, source)


def build_download_url(source: str, repo: str, revision: str, file_path: str) -> str:
    """按来源构造下载直链（modelscope 走 ``/models/.../resolve`` 302 重定向）。"""
    encoded_rev = _quote(revision)
    encoded_path = "/".join(_quote(part) for part in file_path.split("/"))
    if source == "modelscope":
        return f"https://modelscope.cn/models/{repo}/resolve/{encoded_rev}/{encoded_path}"
    return f"{SOURCES[source]['base']}/{repo}/resolve/{encoded_rev}/{encoded_path}"


async def get_modelscope_download_url(repo: str, revision: str,
                                      file_path: str) -> str | None:
    """取 ModelScope 单文件**签名 URL**（用于 LFS 大文件稳定下载）；失败返回 None。"""
    try:
        url = (f"https://modelscope.cn/api/v1/models/{repo}/repo"
               f"?FilePath={_quote(file_path)}&Revision={_quote(revision)}")
        payload = await fetch_json(url, 15_000)
        data = payload.get("Data") if isinstance(payload, dict) else None
        signed = data.get("Url") if isinstance(data, dict) else None
        return signed if isinstance(signed, str) and signed else None
    except Exception:  # noqa: BLE001 —— 与 TS 的裸 catch 等价：任何失败都回退直链
        return None


@router.post("/hf/files")
async def hf_files(request: Request):
    """列出仓库文件（按来源）。"""
    try:
        body = await read_json(request)
        repo = normalize_repo(body.get("repo"))
        source = parse_source(body.get("source"))
        revision = default_revision(source, body.get("revision"))
        files = await list_remote_files(repo, source, revision)
        return success({"repo": repo, "source": source, "revision": revision, "files": files})
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        return bad_request(str(err))


def _ndjson(payload: Any) -> str:
    """NDJSON 帧（⚠️ **紧凑 JSON** 才是对齐 ``JSON.stringify``）。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


@router.post("/hf/download")
async def hf_download(request: Request):
    """下载模型到「模型存储目录」，**NDJSON 流式**返回进度。"""
    try:
        body = await read_json(request)
        repo = normalize_repo(body.get("repo"))
        source = parse_source(body.get("source"))
        revision = default_revision(source, body.get("revision"))
        raw_files = body.get("files") if isinstance(body.get("files"), list) else []
        requested = [x for x in raw_files if isinstance(x, str) and x.strip()]

        models_dir = get_model_paths().get("models_dir")
        if not models_dir:
            return bad_request("请先在「模型存储目录」中设置下载目录")

        all_files = await list_remote_files(repo, source, revision)
        if len(all_files) == 0:
            return bad_request(f"仓库 {repo} 没有可下载的文件")

        if requested:
            targets = []
            for name in requested:
                hit = next((f for f in all_files if f["name"] == name), None)
                if hit is None:
                    raise ValueError(f"仓库中不存在文件：{name}")
                targets.append(hit)
        else:
            targets = all_files

        dest_dir = os.path.join(models_dir, repo.replace("/", "__"))
        os.makedirs(dest_dir, exist_ok=True)

        return StreamingResponse(
            _download_stream(repo, revision, source, targets, dest_dir),
            media_type="application/x-ndjson; charset=utf-8",
        )
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("LocalModels", "hf-download", {"error": str(err)})
        return bad_request(str(err))


async def open_download(url: str, headers: dict[str, str]) -> tuple[Any, Any]:
    """打开流式下载连接，返回 ``(client, response)``。

    ⚠️ 单独抽出来是为了**自检可替换**（自检里换成假响应，不打网络）。
    超时给足 **4 小时**（大权重文件动辄几十 GB）。
    """
    client = httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(4 * 60 * 60))
    request = client.build_request("GET", url, headers=headers)
    return client, await client.send(request, stream=True)


async def _download_stream(repo: str, revision: str, source: str,
                           targets: list[dict[str, Any]],
                           dest_dir: str) -> AsyncIterator[str]:
    """逐个文件下载并逐帧回报（**跳过已完成 / 断点续传 / 失败保留 .part**）。"""
    total_bytes = sum(int(f.get("size") or 0) for f in targets)
    overall = 0
    failed: list[str] = []
    yield _ndjson({"status": "start", "repo": repo, "revision": revision, "source": source,
                   "files": [f["name"] for f in targets], "total": total_bytes})

    for target in targets:
        name = target["name"]
        size = int(target.get("size") or 0)
        file_done = 0
        try:
            url = build_download_url(source, repo, revision, name)
            # ModelScope 优先取签名 URL 下载 LFS 大文件，失败回退 /resolve/ 直链
            if source == "modelscope":
                signed = await get_modelscope_download_url(repo, revision, name)
                if signed:
                    url = signed

            out_path = os.path.join(dest_dir, name)
            part_path = out_path + ".part"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # 已存在目标文件且大小满足要求：跳过，避免重复下载
            existing = _file_size(out_path)
            if existing > 0 and (size == 0 or existing >= size):
                overall += existing
                yield _ndjson({"status": "file_done", "file": name,
                               "downloaded": existing, "skipped": True})
                continue

            # 断点续传：读取 .part 部分文件大小
            part_size = _file_size(part_path)
            if part_size > 0 and size > 0 and part_size >= size:
                overall += part_size
                os.replace(part_path, out_path)
                yield _ndjson({"status": "file_done", "file": name, "downloaded": part_size})
                continue

            headers = {"Accept": "*/*"}
            if part_size > 0:
                headers["Range"] = f"bytes={part_size}-"

            client, response = await open_download(url, headers)
            try:
                try:
                    if not response.is_success:
                        yield _ndjson({"status": "error", "file": name,
                                       "error": f"HTTP {response.status_code}"})
                        failed.append(name)
                        continue

                    # 服务器返回 206 表示续传生效；返回 200 表示不支持 Range，从头重下
                    is_partial = response.status_code == 206
                    file_done = part_size if is_partial else 0
                    if is_partial:
                        overall += part_size

                    content_length = _content_length(response.headers.get("content-length"))
                    if is_partial and content_length > 0:
                        full_size = file_done + content_length
                        if size > 0 and full_size != size:
                            total_bytes += full_size - size
                        size = full_size
                    elif content_length > 0 and content_length != size:
                        total_bytes += content_length - size
                        size = content_length

                    yield _ndjson({"status": "file_start", "file": name, "total": size,
                                   "resumed": is_partial, "downloaded": file_done})

                    # ⚠️ 同步写文件（与原 TS 的 createWriteStream 等价）——本地单用户工具，
                    #    不做线程池搬运，语义优先
                    with open(part_path, "ab" if is_partial else "wb") as handle:
                        async for chunk in response.aiter_bytes():
                            handle.write(chunk)
                            file_done += len(chunk)
                            overall += len(chunk)
                            yield _ndjson({"status": "progress", "file": name,
                                           "downloaded": file_done, "total": size,
                                           "overall": overall, "overallTotal": total_bytes})
                finally:
                    await response.aclose()
            finally:
                await client.aclose()

            os.replace(part_path, out_path)
            yield _ndjson({"status": "file_done", "file": name, "downloaded": file_done})
        except Exception as err:  # noqa: BLE001 —— 保留 .part 供下次续传，不删除
            yield _ndjson({"status": "error", "file": name, "error": str(err)})
            failed.append(name)

    yield _ndjson({"status": "done", "overall": overall,
                   "total": total_bytes, "failed": failed})


def _file_size(path: str) -> int:
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _content_length(raw: str | None) -> int:
    """``Number(header)``；缺失/非数字/非正数 → 0。"""
    try:
        value = float(str(raw))
    except (TypeError, ValueError):
        return 0
    if value != value or value <= 0:
        return 0
    return int(value)


def _quote(value: str) -> str:
    """``encodeURIComponent``（⚠️ 不要用 ``quote_plus``：空格要变 ``%20`` 而不是 ``+``）。"""
    from urllib.parse import quote

    return quote(str(value), safe="")


def _number(raw: Any) -> int:
    """``Number(x) || 0``（非数字 → 0）。"""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0
    return 0 if value != value else int(value)


# ============================================================
# 阶段 5：删除模型存储目录下的文件/目录
# ============================================================

def resolve_model_path(raw: Any) -> str:
    """校验目标路径落在 ``models_dir`` 内，返回规范化绝对路径（**防路径穿越**）。"""
    target = str(raw or "").strip()
    if not target:
        raise ValueError("目标路径不能为空")
    models_dir = get_model_paths().get("models_dir")
    if not models_dir:
        raise ValueError("请先在「模型存储目录」中设置目录")
    base = os.path.abspath(models_dir)
    absolute = os.path.abspath(os.path.join(base, target))
    if absolute == base:
        raise ValueError("不能删除整个模型存储目录")
    if not absolute.startswith(base + os.sep):
        raise ValueError("目标路径不在模型存储目录内")
    return absolute


@router.post("/delete")
async def delete_models(request: Request):
    """删除「模型存储目录」下的本地模型文件/目录（逐个报结果，**不整批失败**）。"""
    try:
        body = await read_json(request)
        raw_paths = body.get("paths") if isinstance(body.get("paths"), list) else [body.get("path")]
        paths = [x for x in raw_paths if isinstance(x, str) and str(x).strip()]
        if not paths:
            return bad_request("请指定要删除的文件或目录")

        deleted: list[str] = []
        failed: list[dict[str, str]] = []
        for raw in paths:
            label = str(raw).strip()
            try:
                absolute = resolve_model_path(label)
                if not os.path.exists(absolute):
                    failed.append({"path": label, "error": "路径不存在"})
                    continue
                if os.path.isdir(absolute) and not os.path.islink(absolute):
                    shutil.rmtree(absolute, ignore_errors=True)
                else:
                    os.remove(absolute)
                deleted.append(label)
            except Exception as err:  # noqa: BLE001 —— 单条失败不阻断整批
                failed.append({"path": label, "error": str(err)})

        log_task_success("LocalModels", "delete", {
            "deleted": len(deleted), "failed": len(failed),
        })
        return success({"deleted": deleted, "failed": failed})
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("LocalModels", "delete", {"error": str(err)})
        return bad_request(str(err))
