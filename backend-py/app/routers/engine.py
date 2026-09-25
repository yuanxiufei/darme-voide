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
from ..services.engine import hybrid_load
from ..services.engine import inventory as inv
from ..services.engine import loader
from ..services.engine import safetensors as st
from ..services.engine import upscale as upscaler
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


#: 超清计划的合法 ``mode`` ✓（口径在 :mod:`app.services.engine.upscale` ✓）——
#: ⚠️ 不认识 ⇒ **报错并列出合法的** ✗ 且**不静默回落普通模式** ✗✗（那会让用户以为超清开着 ✓）。
_UPSCALE_MODES = ("normal", "ai-2x", "ai-2x-resize", "tiled-ai-2x")


def _upscale_from(body: dict[str, Any]) -> Any:
    """请求里的 ``upscale``（``/engine/upscale-plan`` 的产物 ✓）⇒ :class:`upscale.UpscalePlan` ✓。

    ⚠️ 管线**只执行计划** ✗：倍率/分块/能不能做都由计划层定 ✓ —— 请求里再判一遍就会出现
    「两处口径」（本仓在 H3 权重那件事上刚踩过 ✓✗）。
    """
    raw = body.get("upscale") if body.get("upscale") not in (None, "", {}, False) else body.get("upscalePlan")
    if raw in (None, "", {}, False):
        return None
    if not isinstance(raw, dict):
        raise pipe.StageError("plan", f"upscale 必须是对象（收到 {type(raw).__name__} ✓）")
    mode = str(raw.get("mode") or "").strip()
    if mode not in _UPSCALE_MODES:
        raise pipe.StageError(
            "plan", f"未知超清 mode {mode!r} ✗ ⇒ 只能是 {' / '.join(_UPSCALE_MODES)} ✓"
                    f"（先调 ``/engine/upscale-plan`` 拿计划 ✓，别自己编 ✗）")
    tiles: list[tuple[int, int, int, int]] = []
    for item in raw.get("tiles") or ():
        try:
            tiles.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
        except (TypeError, ValueError, IndexError) as err:
            raise pipe.StageError("plan", f"分块格式不对 ✗（每块要 4 个整数 ✓）：{item!r}") from err
    try:
        scale = float(raw.get("scale") or 1.0)
    except (TypeError, ValueError) as err:
        raise pipe.StageError("plan", f"超清倍率不是数字 ✗（收到 {raw.get('scale')!r} ✓）") from err
    # ⚠️ 二采的两个数**照收** ✓（管线**只执行计划** ✗ —— 缺了就等于「不做二采」✓ 并在 notes 里说清 ✓）
    try:
        refine_steps = int(raw.get("refineSteps") or 0)
        refine_denoise = float(raw.get("refineDenoise") or 0.0)
    except (TypeError, ValueError) as err:
        raise pipe.StageError(
            "plan", f"超清二采参数不是数字 ✗（refineSteps/refineDenoise 收到 "
                    f"{raw.get('refineSteps')!r}/{raw.get('refineDenoise')!r} ✓）") from err
    return upscaler.UpscalePlan(
        mode=mode, scale=scale, steps=tuple(str(item) for item in raw.get("steps") or ()),
        notes=tuple(str(item) for item in raw.get("notes") or ()), tiles=tuple(tiles),
        fallback_reason=raw.get("fallbackReason"),
        refine_steps=refine_steps, refine_denoise=refine_denoise)


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

    upscale = _upscale_from(body)
    if upscale is not None:
        values["upscale"] = upscale
    return pipe.GenerationRequest(**values)


@router.get("/readiness")
def readiness(
    stage: str = "h3",
    models_dir: str | None = None,
    capacity_gib_raw: str | None = Query(None, alias="capacityGiB"),
    upscale_key: str | None = Query(None, alias="upscaleKey"),
    upscale_path: str | None = Query(None, alias="upscalePath"),
) -> Any:
    """就绪报告：``ready`` + 缺哪些（``missingRequired``）+ 坏哪些（``brokenRequired``）✓。

    ``capacityGiB`` 用于算显存余量（默认 24 = 本机 A5000 ✓，可覆盖 ✓）。
    ``modelsDir`` 不给就用清单里配置的模型根目录 ✓（便于「先看看别的盘上有没有」✓）。
    ``upscaleKey`` / ``upscalePath`` 可选 ⇒ 顺带回答**超清放大器能不能用** ✓
    （⚠️ 不给就是 ``checked=false`` **「没查」** ✗ —— **不是**「超清可用」✗✗）。
    """
    if not inv.STAGE_FILTERS.get(stage):
        return bad_request(f"未知阶段 {stage!r}；可用：{sorted(inv.STAGE_FILTERS)}")
    # ⚠️ 声明成 str 再自己解析：非法值应当**回退到默认**而不是让 FastAPI 抛 422 ✓（本仓一致做法 ✓）
    try:
        capacity = float(capacity_gib_raw) if capacity_gib_raw else inv.DEFAULT_CAPACITY_GIB
    except ValueError:
        capacity = inv.DEFAULT_CAPACITY_GIB
    root = Path(models_dir) if models_dir else None
    return success(inv.readiness(stage, root=root, capacity_gib=max(1.0, capacity),
                                 upscale_key=upscale_key or "", upscale_path=upscale_path or ""))


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


def _resolve_component_path(raw: str, key: str) -> str | None:
    """``path`` 或**组件 key** ⇒ 真实路径 ✓（与 ``/inspect`` 同一口径 ✓ ⇒ 前端不用自己拼 ``kind/filename`` ✓）。"""
    if raw:
        return raw
    if not key:
        return None
    entry = next((item for item in inv.load_catalog()["models"] if item.get("key") == key), None)
    if entry is None:
        return None
    resolved = inv.component_path(entry)
    return None if resolved is None else str(resolved)


@router.post("/hybrid-merge")
async def hybrid_merge(request: Request) -> Any:
    """**混合加载**（``fl2va`` 基底 + ``ref2va`` 的 adaLN 覆盖层 ✓）⇒ 一个两用权重 ✓。

    默认**只做计划** ✓（不写盘 ✗）：取哪些键 / 跳哪些 + 产物多大 ✓；``apply=true`` 才**真合并落盘** ✓。

    四条口径（每条都是「出问题会静默」的形状 ✓✗）：
    1. ⭐ **合并不改键集** ✗✗（产物键集 == 基底键集 ✓ ⇒ 落盘前核一遍 ✓）；
    2. ⭐ **继承基底的内嵌元数据** ✗✗（``load_weights`` 靠它读结构 ✓；丢了它 ⇒ 产物装不上 ✓✗）；
    3. ⭐ 复用要过**产物自证 + sidecar 源指纹**两道 ✓（坏缓存一律拒 ⇒ 重合并 ✓）；
    4. ⭐ 磁盘余量不足 ⇒ **拒且不写一个字节** ✗；写 sidecar 失败 ⇒ **不阻塞** ✓ 但报告里带着错因 ✓✗。
    """
    body = await read_json(request)
    base = _resolve_component_path(str(body.get("base") or body.get("basePath") or "").strip(),
                                   str(body.get("baseKey") or "").strip())
    overlay = _resolve_component_path(str(body.get("overlay") or body.get("overlayPath") or "").strip(),
                                      str(body.get("overlayKey") or "").strip())
    if not base or not overlay:
        return bad_request("需要 base/baseKey 与 overlay/overlayKey（两份权重 ✓），"
                           "或清单里能解析到的组件 key ✓")
    include_final = bool(body.get("includeFinalAdaLn"))
    cache_path = str(body.get("cachePath") or "").strip() or None
    try:
        if not body.get("apply"):
            return success(hybrid_load.plan_hybrid_load(base, overlay,
                                                        include_final_adaln=include_final))
        report = hybrid_load.run_hybrid_load(base, overlay, cache_path,
                                             force=bool(body.get("force")),
                                             include_final_adaln=include_final)
    except hybrid_load.HybridLoadError as err:
        return bad_request(str(err))
    except Exception as err:  # noqa: BLE001 —— 任何意外都变成**明确的 400** ✓（不是 500 ✗）
        return bad_request(f"{type(err).__name__}: {err}")
    return success(report.to_dict())


@router.post("/upscale-plan")
async def upscale_plan(request: Request) -> Any:
    """**超清计划** ✓：这个倍率能不能做、怎么做、要什么分块 ✓（纯规划 ✓ **不跑推理、不加载权重** ✗）。

    ``targetScale``（默认 2 ✓）、``frame``（``[宽, 高]`` ✓ 4× 时必给 ✓）、``maxTile``（>2 倍率必给 ✓）、
    ``path`` / ``filePath`` / ``key``（放大器权重 ✓ ⇒ 从它的 ``safetensors`` 头部读内嵌契约 ✓）。

    三条口径（都是「静默降级/静默被拒」的形状 ✓✗）：
    1. ⭐ **没给权重 或 契约读不出来 ⇒ 回退普通模式** ✓ 且 ``fallbackReason`` **必须给** ✗✗
       （静默降级 = 你以为超清开着 ✓）；
    2. ⭐ **1.5× 不是直接提交给放大器** ✗ —— 它只做空间 2× ⇒ 走「**先合法的 2× 再缩放**」✓✗；
    3. ⭐ **>2 倍率不给 ``maxTile`` ⇒ 报 400** ✗（本仓**不猜**安全块大小 ✓：大块解码出 NaN/Inf ✓✗）。
    """
    body = await read_json(request)
    raw_path = str(body.get("path") or body.get("filePath") or "").strip()
    key = str(body.get("key") or "").strip()
    if not raw_path and key:
        entry = next((item for item in inv.load_catalog()["models"] if item.get("key") == key), None)
        if entry is None:
            return bad_request(f"清单里没有 key={key!r} ✓")
        resolved = inv.component_path(entry)
        if resolved is None:
            return bad_request("清单里该条目没有可解析路径（且 models_dir 未配置 ✗）")

    contract: dict[str, Any] | None = None
    contract_error: str | None = None
    if raw_path:
        try:
            metadata, _header = st.read_header(raw_path)
            contract = upscaler.read_upscaler_contract(metadata)
        except (FileNotFoundError, IsADirectoryError) as err:
            contract_error = f"放大器权重读不到（{type(err).__name__}: {err} ✓）"
        except Exception as err:  # noqa: BLE001 —— 格式/契约不合法 ⇒ **回退普通 + 给理由** ✓ 不是 500 ✗
            contract_error = f"{type(err).__name__}: {err}"

    raw_frame = body.get("frame")
    frame: tuple[int, int] | None = None
    if isinstance(raw_frame, (list, tuple)) and len(raw_frame) == 2:
        try:
            frame = (int(raw_frame[0]), int(raw_frame[1]))
        except (TypeError, ValueError):
            return bad_request(f"frame 必须是两个整数（收到 {raw_frame!r} ✗）")
    raw_max_tile = body.get("maxTile") or body.get("max_tile")
    try:
        max_tile = int(raw_max_tile) if raw_max_tile not in (None, "") else None
    except (TypeError, ValueError):
        return bad_request(f"maxTile 必须是整数（收到 {raw_max_tile!r} ✗）")
    try:
        target = float(body.get("targetScale") or 2.0)
    except (TypeError, ValueError):
        return bad_request(f"targetScale 必须是数字（收到 {body.get('targetScale')!r} ✗）")
    # 二采的两个参数 ✓（⚠️ **默认值在编译层** ✗ ⇒ 不给就"不做二采" ✓ 并写进 notes ✓ 绝不猜 ✗）
    raw_refine_steps = body.get("refineSteps") if body.get("refineSteps") is not None \
        else body.get("refine_steps")
    raw_denoise = body.get("refineDenoise") if body.get("refineDenoise") is not None \
        else body.get("refine_denoise")
    try:
        refine_steps = int(raw_refine_steps) if raw_refine_steps not in (None, "") else None
    except (TypeError, ValueError):
        return bad_request(f"refineSteps 必须是整数（收到 {raw_refine_steps!r} ✗）")
    try:
        refine_denoise = float(raw_denoise) if raw_denoise not in (None, "") else None
    except (TypeError, ValueError):
        return bad_request(f"refineDenoise 必须是数字（收到 {raw_denoise!r} ✗）")
    try:
        plan = upscaler.plan_upscale(target_scale=target, contract=contract, frame=frame,
                                     max_tile=max_tile, refine_steps=refine_steps,
                                     refine_denoise=refine_denoise)
    except ValueError as err:
        return bad_request(str(err))
    return success({**plan.to_dict(), "contractError": contract_error,
                    "contractFormat": upscaler.CHECKPOINT_FORMAT,
                    "weightsPath": raw_path or None})


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
