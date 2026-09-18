"""自研推理引擎的**体检路由**（2026-09-17）。

为什么需要它（这是用户最痛的两件事 ✓）：

1. **「为什么跑不起来」**——本机现实是「8 个组件、主权重 19.53 GiB」✓，缺一个就跑不动 ✗；
   这个接口**逐条**告诉你：哪个组件缺、哪个坏了（截断/头部矛盾）、还差多少 GB ✓。
2. **「这个文件是不是我要的」**——几十 GB 的 safetensors，**只读头部**几 KB 就能报出
   张量清单/形状/精度 ✓，不用等到加载时才炸 ✗。

* ``GET /api/v1/engine/readiness``  某阶段的就绪报告（``?stage=h3|image|audio|text`` ✓）
* ``POST /api/v1/engine/inspect``   体检**任意** safetensors 文件（body: ``{"path": "..."}`` ✓）
* ``POST /api/v1/engine/plan``      生成请求 → **具体数字**（尺寸/帧数/σ 序列 ✓ 不跑推理 ✓）
* ``POST /api/v1/engine/dry-run``   用**干跑后端**把整条管线走一遍（验编排 ✓ 输出标 ``synthetic`` ✗）

⚠️ 前两个**不加载模型、不占显存** ✓（只读文件头，毫秒级 ✓）。
⚠️ ``dry-run`` **不生成真画面** ✗（零依赖小向量 ✓）⇒ 返回值里 ``synthetic: true`` ✓，
前端必须据此打标，**不许**当成生成结果展示 ✗（等 ``torch`` 后端落地后再加 ``/generate`` ✓）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request

from ..core.response import bad_request, success
from ..core.request_utils import read_json
from ..services.engine import conditioning
from ..services.engine import guidance
from ..services.engine import inventory as inv
from ..services.engine import loader
from ..services.engine import pipeline as pipe
from ..services.engine import safetensors as st
from ..services.engine.dryrun import DryRunBackend
from ..services.engine.torch_backend import TorchBackend

router = APIRouter(prefix="/api/v1/engine", tags=["engine"])

#: 前端字段名（camelCase ✓）→ 请求字段（snake_case ✓）—— 两边都收，取第一个非空 ✓
_REQUEST_KEYS: dict[str, str] = {
    "prompt": "prompt", "negative": "negative", "seed": "seed",
    "seconds": "seconds", "duration": "seconds", "fps": "fps",
    "ratio": "ratio", "aspectRatio": "ratio",
    "megapixels": "megapixels", "steps": "steps",
    "sampler": "sampler", "schedule": "schedule",
    "temporalCompression": "temporal_compression",
    "firstFrame": "first_frame", "referenceFrames": "reference_frames",
    "outputsDir": "outputs_dir",
}


#: 引导字段：前端叫 ``cfg`` / ``guidanceScale`` 都收 ✓（界面上常直接写 CFG ✓）
_GUIDANCE_KEYS: dict[str, str] = {
    "cfg": "scale", "guidanceScale": "scale", "guidanceRescale": "rescale",
    "guidanceStart": "start", "guidanceEnd": "end", "guidanceRamp": "ramp_steps",
}


#: 首帧条件字段（图生视频 ✓）
_CONDITION_KEYS: dict[str, str] = {
    "conditionKeep": "keep", "conditionFade": "fade", "conditionStrength": "strength",
}


def _generation_request(body: dict[str, Any]) -> pipe.GenerationRequest:
    """JSON → :class:`GenerationRequest` ✓（按前端 camelCase 收，缺项走默认 ✓）。"""
    values: dict[str, Any] = {}
    for source, target in _REQUEST_KEYS.items():
        if target in values or source not in body or body[source] in (None, ""):
            continue
        values[target] = body[source]
    if "ratio" in values and isinstance(values["ratio"], list):
        values["ratio"] = tuple(float(item) for item in values["ratio"] if item is not None)
    if "reference_frames" in values and isinstance(values["reference_frames"], list):
        values["reference_frames"] = tuple(str(item) for item in values["reference_frames"])

    guidance_values: dict[str, Any] = {}
    for source, target in _GUIDANCE_KEYS.items():
        if source not in body or body[source] in (None, ""):
            continue
        try:
            guidance_values[target] = (int(body[source]) if target == "ramp_steps"
                                       else float(body[source]))
        except (TypeError, ValueError) as err:
            # ⚠️ 坏输入必须变成**明确的 400** ✓（而不是让 float() 抛出去变 500 ✗）
            raise pipe.StageError("plan", f"引导参数 {source} 不是数字：{body[source]!r}") from err
    if guidance_values:
        values["guidance"] = guidance.GuidanceConfig(**guidance_values)

    conditioning_values: dict[str, Any] = {}
    for source, target in _CONDITION_KEYS.items():
        if source not in body or body[source] in (None, ""):
            continue
        try:
            conditioning_values[target] = (int(body[source]) if target != "strength"
                                           else float(body[source]))
        except (TypeError, ValueError) as err:
            raise pipe.StageError("plan", f"首帧条件参数 {source} 不是数字：{body[source]!r}") from err
    if conditioning_values:
        values["conditioning"] = conditioning.ConditioningConfig(**conditioning_values)
    return pipe.GenerationRequest(**values)


@router.get("/readiness")
def readiness(
    stage: str = "h3",
    models_dir: str | None = None,
    capacity_gib_raw: str | None = Query(None, alias="capacityGiB"),
) -> Any:
    """就绪报告：``ready`` + 缺哪些（``missingRequired``）+ 坏哪些（``brokenRequired``）✓。

    ``capacityGiB`` 用于算显存余量（默认 24 = 本机 A5000 ✓，可覆盖 ✓）。
    ``modelsDir`` 不给就用清单里配置的模型根目录 ✓（便于「先看看别的盘上有没有」✓）。
    """
    if not inv.STAGE_FILTERS.get(stage):
        return bad_request(f"未知阶段 {stage!r}；可用：{sorted(inv.STAGE_FILTERS)}")
    # ⚠️ 声明成 str 再自己解析：非法值应当**回退到默认**而不是让 FastAPI 抛 422 ✓（本仓一致做法 ✓）
    try:
        capacity = float(capacity_gib_raw) if capacity_gib_raw else inv.DEFAULT_CAPACITY_GIB
    except ValueError:
        capacity = inv.DEFAULT_CAPACITY_GIB
    root = Path(models_dir) if models_dir else None
    return success(inv.readiness(stage, root=root, capacity_gib=max(1.0, capacity)))


@router.post("/inspect")
async def inspect_file(request: Request) -> Any:
    """体检一个 safetensors 文件（**只读头部** ✓）：张量数、精度分布、最大的几个、是否截断 ✓。

    ``filePath`` 也可直接给**组件 key**（如 ``dit_fl2va_int8`` ✓）⇒ 按清单解析真实路径 ✓，
    这样前端不用自己拼 ``kind/filename`` ✓。
    """
    body = await read_json(request)
    # ⚠️ 这里**不再**写 ``isinstance(body, dict)`` ✗ —— ``read_json`` 已经把非法 JSON / 非对象
    #    归一成 ``{}`` ✓（对齐 TS：由**业务校验**给出 400 ✓），那行是死代码 ✗。
    raw_path = str(body.get("path") or body.get("filePath") or "").strip()
    key = str(body.get("key") or "").strip()
    if not raw_path and key:
        entry = next((item for item in inv.load_catalog()["models"] if item.get("key") == key), None)
        if entry is None:
            return bad_request(f"清单里没有 key={key!r} ✓")
        resolved = inv.component_path(entry)
        if resolved is None:
            return bad_request("清单里该条目没有可解析路径（且 models_dir 未配置 ✗）")
        raw_path = str(resolved)
    if not raw_path:
        return bad_request("需要 path / filePath / key 之一 ✓")
    info = st.inspect(raw_path)
    return success(info.to_dict())


@router.get("/backends")
def backends() -> Any:
    """**推理后端现状** ✓ —— 现在能跑什么、要装什么、装完还差什么 ✓。

    这条接口专治一种含糊：只说「后端不可用」✗ ⇒ 分不清是**依赖没装**（去装包 ✓）
    还是**实现待写**（去写代码 ✓）。两者治法完全不同 ✗（见 `torch_backend` 的 ``reason`` ✓）。
    """
    return success({
        "dryrun": {
            "name": "dryrun", "synthetic": True, "available": True, "canGenerate": False,
            "note": "零依赖干跑 ✓：只验编排（阶段/事件/取消/归因/引导/首帧 ✓），**不产生画面** ✗",
        },
        "torch": TorchBackend().describe(),
        "guidance": {"supportsCfg": True, "note": "引导由引擎按步计算 ✓ 与后端无关 ✓"},
        "conditioning": {"supportsFirstFrame": True,
                         "note": "引擎算掩码 ✓、后端套到自己的布局上 ✓（不猜布局 ✗）"},
    })


@router.get("/load-plan")
def load_plan(
    stage: str = "h3",
    models_dir: str | None = None,
    capacity_gib_raw: str | None = Query(None, alias="capacityGiB"),
) -> Any:
    """**加载计划** ✓ —— 「装齐之后怎么跑」：每个文件什么角色/精度、要不要反量化、
    以及最关键的**显存排班**（谁先进、谁先退 ✓）。

    ⚠️ 比 ``/readiness`` 多出来的那层价值：``readiness`` 只说「齐没齐」✓，
    这里回答**H3 主权重 19.53 GiB + 24GB 卡能不能同时待着** ✗（答：不能 ✗ ⇒
    必须「文本编码器用完就退 → 上 DiT → 最后 VAE」✓，这个顺序由 :func:`loader.residency_plan` 给出 ✓）。
    """
    if not inv.STAGE_FILTERS.get(stage):
        return bad_request(f"未知阶段 {stage!r}；可用：{sorted(inv.STAGE_FILTERS)}")
    try:
        capacity = float(capacity_gib_raw) if capacity_gib_raw else inv.DEFAULT_CAPACITY_GIB
    except ValueError:
        capacity = inv.DEFAULT_CAPACITY_GIB
    root = Path(models_dir) if models_dir else None
    return success(loader.plan_stage(stage, root=root, capacity_gib=max(1.0, capacity)))


@router.post("/plan")
async def plan_prediction(request: Request) -> Any:
    """生成请求 → **具体数字**（尺寸 / 帧数 / σ 序列 / 警告 ✓），**不跑推理** ✓。

    用途：前端在用户按下生成前就能显示「会出 1088×608 / 124 帧 / 30 步」✓，
    ``steps``/采样器写错也**这里就报 400**（而不是等跑一半才炸 ✗）。
    ``weightsBytes`` 可选 ⇒ 顺带给出显存估算 ✓（口径与 ``/readiness`` 同一个 ✓）。
    """
    body = await read_json(request)
    # ⚠️ 这里**不再**写 ``isinstance(body, dict)`` ✗ —— ``read_json`` 已经把非法 JSON / 非对象
    #    归一成 ``{}`` ✓（对齐 TS：由**业务校验**给出 400 ✓），那行是死代码 ✗。
    try:
        weights = int(body.get("weightsBytes") or 0) or None
    except (TypeError, ValueError):
        weights = None
    try:
        plan = pipe.build_plan(_generation_request(body), weights_bytes=weights)
    except pipe.StageError as err:
        return bad_request(str(err))
    return success(plan.to_dict())


@router.post("/dry-run")
async def dry_run(request: Request) -> Any:
    """用**干跑后端**把整条管线走一遍 ✓ —— 验的是**编排**（阶段/事件/取消/归因 ✓），不是画面 ✗。

    返回值永远是 ``synthetic: true`` ✓ 且 ``outputs.videoPath == null`` ✓ ——
    前端据此明确打「试跑」标 ✓，**不许**当生成结果展示 ✗。
    """
    body = await read_json(request)
    # ⚠️ 这里**不再**写 ``isinstance(body, dict)`` ✗ —— ``read_json`` 已经把非法 JSON / 非对象
    #    归一成 ``{}`` ✓（对齐 TS：由**业务校验**给出 400 ✓），那行是死代码 ✗。
    events: list[dict[str, Any]] = []
    body_events = bool(body.get("includeEvents"))
    result = pipe.run_sync(_generation_request(body), DryRunBackend(),
                           on_event=events.append if body_events else None)
    payload = result.to_dict()
    if body_events:
        payload["events"] = events
    return success(payload)
