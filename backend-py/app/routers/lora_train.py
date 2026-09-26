"""**LoRA 训练**的 HTTP 层 ✓（LoRAMaster 移植 ✓ —— 「能抄的都抄」✓，见 ``services/lora_train`` ✓）。

本仓原本只有推理侧 ✓（LoRA 当加速插件用 ✓），这一块是第一次把**训练**搬到后端 ✓。

## 两组接口 ✓（训练链 ✓ / 素材工具 ✓）

**训练**（``services/lora_train/runtime.py`` 的队列 ✓）：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| ``GET`` | ``/models`` | 5 条训练链的参数表 + 本机环境（解释器 / 工具在不在 / 线程预算 ✓） |
| ``GET`` / ``POST`` / ``DELETE`` | ``/settings`` | 某模型的 ``settings.toml`` 读 / 存 / 重置 ✓ |
| ``POST`` | ``/preview`` | **只拼 argv**（不跑、不占显存 ✓）⇒ 前端先给用户看要跑什么 |
| ``POST`` | ``/train`` | 入队 ✓ ⇒ ``taskId`` ✓（``wait=true`` 同步等 ✓） |
| ``GET`` | ``/runtime`` ``/tasks`` ``/tasks/{id}`` | 忙不忙 / 队列 / 单任务（带增量日志 ✓） |
| ``POST`` | ``/tasks/{id}/cancel`` | 取消 ✓ —— 排队中立刻终结 ✓，跑着的**杀整棵进程树** ✓ |

**素材工具**（``services/lora_train/dataset/runtime.py`` 的队列 ✓）：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| ``GET`` | ``/dataset/catalog`` | 三个工具（改名 / 转格式 / 打标 ✓）的字段表 + 打标取值域 |
| ``POST`` | ``/dataset/run`` | 入队一次素材任务 ✓（``tool`` + ``action`` + ``values`` ✓） |
| ``GET`` | ``/dataset/status`` | 忙不忙 / 队列 / 打标模型加载状态 ✓ |
| ``GET`` | ``/dataset/tasks`` ``/dataset/tasks/{id}`` | 同上 |
| ``POST`` | ``/dataset/tasks/{id}/cancel`` | 取消 ✓（打标 / 转格式是**逐张**停 ✓，改名不响应 ✓ —— 见 ``dataset/runtime.py`` ✓） |

## 四条口径（都是「不说清就会静默出错」的形状 ✓✗）

1. ⭐ **参数在提交时验完** ✗：``merge`` → ``build_plan``（含 ``validate`` ✓ + 工具在不在 ✓）都在
   ``POST /train`` 里**同步**跑完 ✓ ⇒ 参数错是 **400** ✓，而不是"任务跑起来才失败"✓✗。
2. ⭐ **``/preview`` 不写任何任务产物** ✗（除了那个 ``prompt.txt`` ✓ —— 它落在
   ``<数据根>/lora-train/previews/<模型>/`` ✓ 一个固定位置 ✓，不进任务队列 ✓）。
3. ⭐ **打标会抢推理引擎的显存** ✗✗ ⇒ 提交打标时就把话说明白：引擎在忙 ⇒ **拒绝并给两条出路** ✓
   （等它 / 关掉自动腾显存 ✓），**不做 force 强卸** ✗（那会把用户正在跑的视频任务搞失败 ✓）。
4. ⭐ **错误码按 ``LoraTrainError.status_code`` 分派** ✓：参数错 400 ✓ / 任务号不认识 404 ✓ /
   忙 409 ✓ / 命令根本起不来 500 ✓ —— 不一律 400 ✗（前端分不清"我写错了"和"环境坏了"✓✗）。
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from fastapi import APIRouter, Query, Request

from ..core.request_utils import read_json
from ..core.response import bad_request, conflict, not_found, server_error, success
from ..services.lora_train import config as train_config
from ..services.lora_train import dataset as dataset_pkg
from ..services.lora_train.dataset import runtime as dataset_runtime_module
from ..services.lora_train.dataset.runtime import dataset_runtime
from ..services.lora_train.errors import LoraTrainError
from ..services.lora_train.options import DEFAULT_MODEL, catalog, model_keys
from ..services.lora_train.paths import environment_report
from ..services.lora_train.runtime import lora_train_runtime

router = APIRouter(prefix="/api/v1/lora-train", tags=["lora-train"])

#: 预览用的固定落点 ✓（见模块头第 2 条 ✓）
PREVIEW_SUBDIR = "previews"


def _error(err: LoraTrainError) -> Any:
    """按 :attr:`LoraTrainError.status_code` 分派 ✓（见模块头第 4 条 ✓）。"""
    code = int(getattr(err, "status_code", 400) or 400)
    if code == 404:
        return not_found(str(err))
    if code == 409:
        return conflict(str(err))
    if code >= 500:
        return server_error(str(err))
    return bad_request(str(err))


def _guard(handler: Callable[[], Any]) -> Any:
    """跑一段业务 ✓ ⇒ ``success(...)`` ✓；业务错 ⇒ 按码回错 ✓（**不把业务错变成 500** ✗）。"""
    try:
        return success(handler())
    except LoraTrainError as err:
        return _error(err)


def _model_from(body: dict[str, Any]) -> str:
    raw = str(body.get("model") or body.get("modelKey") or body.get("model_key") or "").strip()
    return raw or DEFAULT_MODEL


def _payload_from(body: dict[str, Any]) -> dict[str, Any] | None:
    """请求里的参数覆盖 ✓（``values`` 优先 ✓，其次才是把整个 body 当参数表 ✓）。"""
    values = body.get("values")
    if isinstance(values, dict):
        return dict(values)
    if values is None and body.get("useBodyAsValues"):
        return {key: value for key, value in body.items() if key not in ("model", "modelKey", "wait")}
    return None


# ── 训练：目录 / 参数 ─────────────────────────────────────────────────────


@router.get("/models")
def lora_train_models() -> Any:
    """5 条训练链的**参数表** ✓ + 本机环境 ✓ —— 前端页面整个可以由它渲染出来 ✓。

    ``environment`` 里回答的是"**能不能跑**"✓：解释器在哪 ✓、musubi-tuner / sd-scripts
    在不在 ✓、accelerate 有没有 ✓、CPU 线程预算给几个 ✓（``cpu_budget`` 那套 ✓）。
    ⚠️ 工具目录是**动态探测**的 ✓（``paths.py`` ✓）—— 探不到就如实说"没找到"✓，
    不会编一个机器绝对路径出来 ✗（本仓纪律 ✓）。
    """
    return success({
        "models": catalog(),
        "defaultModel": DEFAULT_MODEL,
        "modelKeys": list(model_keys()),
        "environment": environment_report(),
        "dataDir": str(train_config.data_dir()),
    })


@router.get("/settings")
def get_settings(model: str = DEFAULT_MODEL) -> Any:
    """读某模型的参数 ✓（**默认值 + 盘上文件** ✓）—— 没存过就是一份默认值 ✓。"""
    return _guard(lambda: {"model": model, "values": train_config.load_values(model),
                           "path": str(train_config.settings_path(model))})


@router.post("/settings")
async def save_settings(request: Request) -> Any:
    """存某模型的参数 ✓ —— ⚠️ **先校验再落盘** ✓（参数错 ⇒ 400 且**一个字都不写** ✗）。"""
    body = await read_json(request)
    model = _model_from(body)
    values = _payload_from(body) or {}
    return _guard(lambda: {"model": model, "path": str(train_config.save_values(model, values))})


@router.delete("/settings")
def reset_settings(model: str = DEFAULT_MODEL) -> Any:
    """重置某模型的参数 ✓ ⇒ 是否真删了 ✓（本来就没存过 ⇒ ``removed: false`` ✓，**这不是错误** ✗）。"""
    return _guard(lambda: {"model": model, "removed": train_config.reset_values(model),
                           "values": train_config.load_values(model)})


@router.post("/preview")
async def preview_train(request: Request) -> Any:
    """**只拼 argv** ✓ —— 不跑、不入队、不占显存 ✓（前端"先给我看看要跑什么" ✓）。

    ``values`` 省略 ⇒ 用盘上的 ``settings.toml`` ✓；给了 ⇒ 覆盖 ✓（逐键校验 ✓）。
    ⚠️ 要出样图时会往 ``<数据根>/lora-train/previews/<模型>/prompt.txt`` 写一个文件 ✓
    （**固定位置** ✗ 不随任务变 ✓）—— ``promptFile`` 里会给出来 ✓。
    """
    body = await read_json(request)
    model = _model_from(body)
    values = _payload_from(body)
    preview_dir = train_config.data_dir() / PREVIEW_SUBDIR / model
    preview_dir.mkdir(parents=True, exist_ok=True)
    from ..services.lora_train.commands import build_plan  # noqa: PLC0415 - 只在这个端点用 ✓
    from ..services.lora_train.runtime import resolve_values  # noqa: PLC0415

    def _run() -> dict[str, Any]:
        resolved = resolve_values(model, values)
        plan = build_plan(model, resolved, workspace=preview_dir)
        return {
            "model": model,
            "values": resolved,
            "cacheCommands": [list(argv) for argv in plan.cache],
            "trainCommand": list(plan.train),
            "tools": dict(plan.tools),
            "promptFile": None if plan.prompt_file is None else str(plan.prompt_file),
        }

    return _guard(_run)


# ── 训练：跑 ─────────────────────────────────────────────────────────────


@router.post("/train")
async def start_train(request: Request) -> Any:
    """**开跑** ✓ ⇒ ``taskId`` ✓（默认入队 ✓；``wait=true`` 同步等 ✓）。

    ⚠️ 参数在**这里同步验完** ✓（见模块头第 1 条 ✓）⇒ 参数错是 400 ✓、任务根本没入队 ✓✗。
    ``POST /train`` 与 ``/preview`` 的区别只有一个 ✓：它真起子进程 ✓。
    """
    body = await read_json(request)
    model = _model_from(body)
    values = _payload_from(body)
    task_id = lora_train_runtime.submit(model, values)
    poll = {"taskId": task_id, "pollUrl": f"/api/v1/lora-train/tasks/{task_id}"}
    if not bool(body.get("wait")):
        return success({**poll, "status": "queued", "waited": False})
    try:
        timeout = float(body.get("timeoutSeconds") or body.get("timeout_seconds") or 0) or None
    except (TypeError, ValueError):
        return bad_request(f"timeoutSeconds 不是数字（收到 {body.get('timeoutSeconds')!r} ✗）")
    return _guard(lambda: {**poll, "waited": True,
                           "task": lora_train_runtime.wait(task_id, timeout=timeout)})


@router.get("/runtime")
def train_runtime() -> Any:
    """训练侧现状 ✓：忙不忙 / 在跑谁 / 队列 / 线程预算 / 环境 ✓。"""
    return _guard(lora_train_runtime.status)


@router.get("/tasks")
def list_train_tasks(limit: int = 20) -> Any:
    """最近的任务 ✓（**瘦身版** ✗ 不带日志 ✓ —— 日志走 ``/tasks/{id}`` ✓）。"""
    return success({"tasks": lora_train_runtime.list_tasks(limit=limit)})


@router.get("/tasks/{task_id}")
def get_train_task(task_id: str, since: int = 0,
                   log_limit_raw: int = Query(200, alias="logLimit")) -> Any:
    """单个任务 ✓：状态 / 阶段 / 进度事实 / 计划 argv / **增量日志** ✓。

    ``since`` 是**上一次拿到的最大序号** ✓ ⇒ 只回更新的那些 ✓（``truncated`` 说明有没有追丢 ✓）。
    """
    return _guard(lambda: lora_train_runtime.get_task(task_id, since=since,
                                                      log_limit=max(0, log_limit_raw)))


@router.post("/tasks/{task_id}/cancel")
def cancel_train_task(task_id: str) -> Any:
    """取消 ✓ —— 排队中的立刻终结 ✓；跑着的**杀整棵进程树** ✓（不是只杀父进程 ✓✗）。"""
    return _guard(lambda: lora_train_runtime.cancel(task_id))


# ── 素材工具 ─────────────────────────────────────────────────────────────


@router.get("/dataset/catalog")
def dataset_catalog() -> Any:
    """素材工具的字段表 ✓（改名 / 转格式 / 打标 ✓）+ 打标类型与 27 个补充开关 ✓。"""
    return success({**dataset_pkg.fields.catalog(), "dataDir": str(train_config.data_dir()),
                    "status": dataset_runtime.status()})


@router.post("/dataset/run")
async def dataset_run(request: Request) -> Any:
    """**跑一次素材任务** ✓ ⇒ ``taskId`` ✓。

    ``tool`` = ``rename`` / ``convert`` / ``caption`` ✓；
    ``action`` 省略 ⇒ 该工具的默认动作 ✓（``caption`` 有四个动作 ✓：
    ``caption`` 打标 ✓ / ``trigger`` 只写触发词 ✓ / ``filter`` 过滤标签 ✓ / ``scan`` 只数不改 ✓）。

    ⚠️ 打标**会先卸掉推理引擎腾显存** ✓（见 ``dataset/runtime.py`` 的
    :func:`release_inference_vram` ✓）—— 引擎正忙 ⇒ 这里就 **409** ✓（不是跑起来才 OOM ✓✗）。
    """
    body = await read_json(request)
    tool = str(body.get("tool") or dataset_pkg.fields.DEFAULT_TOOL).strip()
    action = body.get("action")
    values = _payload_from(body)
    task_id = dataset_runtime.submit(tool, None if action in (None, "") else str(action), values)
    return success({"taskId": task_id, "tool": tool, "status": "queued",
                    "pollUrl": f"/api/v1/lora-train/dataset/tasks/{task_id}"})


@router.get("/dataset/status")
def dataset_status() -> Any:
    """素材侧现状 ✓：忙不忙 / 队列 / **打标模型装没装** ✓（``captionModel.loaded`` ✓）。"""
    return _guard(dataset_runtime.status)


@router.get("/dataset/tasks")
def list_dataset_tasks(limit: int = 20) -> Any:
    """最近跑过的素材任务 ✓（带结果事实 ✓ —— 改了多少张 / 转了多少张 ✓）。"""
    return success({"tasks": dataset_runtime.list_tasks(limit=limit)})


@router.get("/dataset/tasks/{task_id}")
def get_dataset_task(task_id: str, since: int = 0,
                     log_limit_raw: int = Query(300, alias="logLimit")) -> Any:
    """单个素材任务 ✓（增量日志 ✓ —— 打标是**逐张**打一行 ✓，前端照它画进度 ✓）。"""
    return _guard(lambda: dataset_runtime.get_task(task_id, since=since,
                                                   log_limit=max(0, log_limit_raw)))


@router.post("/dataset/tasks/{task_id}/cancel")
def cancel_dataset_task(task_id: str) -> Any:
    """取消素材任务 ✓（⚠️ 改名**不响应**取消 ✓ —— 理由见 ``dataset/runtime.py`` 模块头 ✓）。"""
    return _guard(lambda: dataset_runtime.cancel(task_id))


@router.get("/dataset/caption-model")
def caption_model_status() -> Any:
    """打标模型现状 ✓（装没装 / 什么档 / 在哪张卡 ✓）—— 前端据此显示"要不要等它下载" ✓。"""
    return success(dataset_runtime_module.caption.caption_engine.describe())


@router.post("/dataset/caption-model/unload")
async def unload_caption_model() -> Any:
    """把打标模型卸掉 ✓（打标跑完想立刻把显存还给推理引擎时用 ✓）。

    ⚠️ ``asyncio.to_thread`` ✗：卸模型会等 CUDA 同步 ✓，扔在事件循环里会把整个后端卡住 ✓✗
    （``routers/engine.py`` 的装载也是这么办的 ✓）。
    """
    released = await asyncio.to_thread(dataset_runtime_module.caption.caption_engine.unload)
    return success({"released": released,
                    "model": dataset_runtime_module.caption.caption_engine.describe()})
