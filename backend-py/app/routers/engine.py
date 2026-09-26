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

**运行时**（2026-09-26 起 ✓ —— 这一组才是「真跑」那条路 ✓）：

* ``GET  /api/v1/engine/runtime``       运行时现状（依赖 / 装了没 / 忙不忙 / 队列 / 最近任务 ✓）
* ``POST /api/v1/engine/load``          装齐引擎（主 DiT ⇒ TE ⇒ VAE ✓ 幂等 ✓）
* ``POST /api/v1/engine/unload``        丢掉张量 + 清缓存 ✓（**这就是"不依赖外部"的卸载** ✗✗ 不通知任何人）
* ``POST /api/v1/engine/generate``      **真生成** ✓（默认入队 ⇒ 拿 ``taskId`` 轮询 ✓；``wait=true`` 同步等 ✓）
* ``GET  /api/v1/engine/tasks`` / ``/tasks/{id}`` / ``POST /tasks/{id}/cancel``   队列与进度 ✓

⚠️ 前两个**不加载模型、不占显存** ✓（只读文件头，毫秒级 ✓）。
⚠️ ``dry-run`` **不生成真画面** ✗（零依赖小向量 ✓）⇒ 返回值里 ``synthetic: true`` ✓，
前端必须据此打标，**不许**当成生成结果展示 ✗。
⚠️ ``generate`` 走**进程内**运行时（:mod:`app.services.engine.runtime` ✓）—— **不用 ComfyUI /
SD WebUI / Ollama** ✓、不起任何子进程 ✓；但它**同样**标 ``synthetic`` ✓✗：主 DiT 权重是真的 ✓，
参考 TE / VAE **未经训练** ✗ ⇒ 产物是真文件 + 噪声画面 ✓（真权重到位前不许改口 ✗）。
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request

from ..core.config import get_storage_root
from ..core.response import bad_request, success
from ..core.request_utils import read_json
from ..services.engine import conditioning
from ..services.engine import guidance
from ..services.engine import hybrid_load
from ..services.engine import inventory as inv
from ..services.engine import loader
from ..services.engine import runtime
from ..services.engine import upscale as upscaler
from ..services.engine import pipeline as pipe
from ..services.engine import safetensors as st
from ..services.engine.dryrun import DryRunBackend
from ..services.engine.runtime import engine_runtime
from ..services.engine.torch_backend import TorchBackend, TorchBackendUnavailable

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
    ``models_dir`` 不给 ⇒ 搜索域 = 清单落点（``models_dir`` ✓）+ **动态探测根** ✓（= 加载时用的那条 ✓）；
    给了 ⇒ **只认那个根** ✓（便于「只按这个目录对账」✓）。
    ⚠️ 查询参数名就是 ``models_dir`` ✗（**没有** ``modelsDir`` 这个别名 ✓ —— 文档里曾写成 camelCase ✓✗，
    而 FastAPI 不别名就按 Python 形参名匹配 ✓ ⇒ 照文档写会**静默按默认域**算 ✓✗）。
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
        # ⚠️ 走**同一条解析** ✗（2026-09-27 修 ✓）：清单落点优先 → 动态探测根 ✓ —— 与加载 /
        #    体检 / :func:`inventory.resolve_component` 完全一致 ✓。以前只看 ``models_dir`` ✗
        #    ⇒ 盘上那份在探测根里时，这里会报"没有可解析路径" ✓✗（同一件事两个说法 ✓）。
        resolved, _source = inv.resolve_component(entry)
        if resolved is None:
            # ⚠️ 盘上没有**不报 400** ✗：这条路的语义是"体检这个文件" ✓ —— 落点算得出来就照检 ✓
            #    （`st.inspect` 会回「文件不存在」✓，前端照旧能显示"没下"✓、也能看到 ``expectedPath`` ✓）；
            #    连**落点**都算不出来（清单没 filename / models_dir 没配 ✓）才 400 ✓。
            resolved = inv.component_path(entry)
            if resolved is None:
                return bad_request("清单里该条目没有可解析路径（filename 为空，或 models_dir 未配置 ✗）")
        raw_path = str(resolved)
    if not raw_path:
        return bad_request("需要 path / filePath / key 之一 ✓")
    info = st.inspect(raw_path)
    return success(info.to_dict())


def _resolve_component_path(raw: str, key: str) -> str | None:
    """``path`` 或**组件 key** ⇒ 真实路径 ✓（与 ``/inspect`` 同一口径 ✓ ⇒ 前端不用自己拼 ``kind/filename`` ✓）。

    ⚠️ 解析走 :func:`inventory.resolve_component` ✓（清单落点优先 → 动态探测根 ✓），
    与加载 / 体检**同一条** ✗ —— 2026-09-27 修 ✓（以前只看 ``models_dir`` ✓✗）。
    """
    if raw:
        return raw
    if not key:
        return None
    entry = next((item for item in inv.load_catalog()["models"] if item.get("key") == key), None)
    if entry is None:
        return None
    # ⚠️ 盘上没有时**回落清单落点** ✗（不是返回 None ✓）：这样加载会带着**真实期望路径**报错 ✓
    #    （「文件不存在：…/diffusion_models/xxx.safetensors」比「没找到路径」可行动得多 ✓）。
    resolved, _source = inv.resolve_component(entry)
    if resolved is None:
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
        # ⚠️ 与 ``/inspect`` / 加载同一条解析 ✓（2026-09-27 修 ✓：清单落点优先 → 动态探测根 ✓）；
        #    盘上没有时回落清单落点（下面读契约那步会给出"读不到"的具名原因 ✓ —— 不是 500 ✓）。
        resolved, _source = inv.resolve_component(entry)
        if resolved is None:
            resolved = inv.component_path(entry)
            if resolved is None:
                return bad_request("清单里该条目没有可解析路径（filename 为空，或 models_dir 未配置 ✗）")
        raw_path = str(resolved)

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


# ── 运行时：真跑那条路（**进程内** ✓ 不依赖任何外部服务 ✓）──────────────────────
#: 装配字段（前端 camelCase ✓ / 后端 snake_case ✓ 都收 ✓ 取第一个非空的 ✓）
_ASSEMBLY_KEYS: dict[str, str] = {
    "stage": "stage",
    "ditPath": "dit_path", "dit_path": "dit_path",
    "tokenizerPath": "tokenizer_path", "tokenizer_path": "tokenizer_path",
    "device": "device",
    "audioLatentMode": "audio_latent_mode", "audio_latent_mode": "audio_latent_mode",
    "attachVae": "attach_video_vae", "attachVideoVae": "attach_video_vae",
    "attach_video_vae": "attach_video_vae",
    "attachAudioVae": "attach_audio_vae", "attach_audio_vae": "attach_audio_vae",
    "force": "force",
}

#: 取整口径的合法值 ✓（口径在 `torch_backend.dualStream` ✓）——
#: ⚠️ 不认识 ⇒ **报错并列出合法的** ✗ 且**不给默认值** ✗✗（本仓老账：好默认值会把"未核实"伪装成"已实现" ✓）
_AUDIO_LATENT_MODES = ("round", "ceil", "floor")


def _bool_from(body: dict[str, Any], *names: str) -> bool | None:
    """取布尔 ✓（``None`` = 没给 ✓ —— 与"给了 false"分得开 ✓✗）。"""
    for name in names:
        if name in body and body[name] not in (None, ""):
            return bool(body[name])
    return None


def _assembly_from(body: dict[str, Any]) -> dict[str, Any]:
    """JSON → :meth:`runtime.EngineRuntime.ensure_loaded` 的**关键字参数** ✓（不给的项**不出现** ✗ ⇒ 走默认 ✓）。"""
    values: dict[str, Any] = {}
    for source, target in _ASSEMBLY_KEYS.items():
        if target in values or source not in body or body[source] in (None, ""):
            continue
        values[target] = body[source]
    for key in ("dit_path", "tokenizer_path", "device", "stage"):
        if key in values:
            values[key] = str(values[key])
    if "audio_latent_mode" in values:
        mode = str(values["audio_latent_mode"]).strip()
        # ⚠️ 不认的口径**必须报错** ✗（静默当成单流 = 用户以为双流开着 ✓✗）
        if mode not in _AUDIO_LATENT_MODES:
            raise pipe.StageError(
                "plan", f"未知 audioLatentMode {mode!r} ✗ ⇒ 只能是 "
                        f"{' / '.join(_AUDIO_LATENT_MODES)} ✓（不给 ⇒ 双流不启用 ✓）")
        values["audio_latent_mode"] = mode
    for key in ("attach_video_vae", "attach_audio_vae", "force"):
        if key in values:
            values[key] = bool(values[key])
    return values


def _default_outputs_dir() -> Path:
    """产物默认落 ``<数据根>/static/engine`` ✓ —— 这样前端能直接用 ``/static`` 取 ✓。"""
    return Path(get_storage_root()) / "engine"


@router.get("/runtime")
def runtime_status() -> Any:
    """运行时现状 ✓：依赖齐不齐 / 装了哪几个组件 / 忙不忙 / 队列 + 最近任务 ✓。

    ⚠️ ``available=false`` 与 ``loaded.dit=false`` 是**两件事** ✗：前者是 ``torch`` 没装 ✓，
    后者是权重没装 ✓（治法完全不同 ✓ —— 见 ``reason`` ✓）。
    """
    return success(engine_runtime.status())


@router.post("/load")
async def load_runtime(request: Request) -> Any:
    """**装齐引擎** ✓（主 DiT ⇒ TE ⇒ VAE ✓ 幂等 ✓）。

    ⚠️ 装载是**阻塞**的 ✗（19.53 GiB 光读盘就很久 ✓）⇒ 用 ``to_thread`` 扔出事件循环 ✓✗
    （**不**在事件循环里直接调 ✓：那会把整个后端卡死 ✓✗ —— 本模块第一条纪律 ✓）。
    装不上 ⇒ **400 + 缺什么** ✓（``reason``: ``deps`` 依赖没装 / ``pending`` 权重或结构没到 ✓）。
    """
    body = await read_json(request)
    try:
        assembly = _assembly_from(body)
    except pipe.StageError as err:
        return bad_request(str(err))
    try:
        report = await asyncio.to_thread(engine_runtime.ensure_loaded, **assembly)
    except runtime.EngineBusy as err:
        return bad_request(str(err))
    except TorchBackendUnavailable as err:
        return bad_request(f"{err}（reason={getattr(err, 'reason', 'pending')} ✓）")
    return success({"loadReport": report, "runtime": engine_runtime.status()})


@router.post("/unload")
async def unload_runtime(request: Request) -> Any:
    """**丢掉张量 + 清缓存** ✓ —— 与外部服务那套（发 HTTP 让人家卸载 ✓）**不是一回事** ✗✗。

    ``force=true`` 可在有任务在跑时强卸 ✓（⚠️ 那个任务会跟着失败 ✗ —— 所以默认**拒** ✓）。
    """
    body = await read_json(request)
    try:
        return success(await asyncio.to_thread(engine_runtime.unload, force=bool(body.get("force"))))
    except runtime.EngineBusy as err:
        return bad_request(str(err))


@router.post("/generate")
async def generate(request: Request) -> Any:
    """**真生成** ✓ —— 送上请求、入队、拿 ``taskId`` 轮询 ✓（``wait=true`` 则同步等 ✓）。

    三种用法（都能用 ✓，按需要挑 ✓）：

    * ``dryRun=true`` —— 用**干跑后端**过一遍运行时自身的队列/装载/事件 ✓（**不装权重** ✗ ⇒
      没权重时也能验这条管道 ✓）；
    * 默认 —— ``assembly`` 给装配口径 ✓（``ditPath`` / ``tokenizerPath`` / ``audioLatentMode`` ✓），
      装不上就把**原因写进任务** ✓（不拒绝请求 ✓：``/readiness`` 已经答过"能不能跑" ✓）；
    * ``wait=true`` + ``timeoutSeconds`` —— 同步等结果 ✓（产物路径在 ``result.outputs`` ✓）。

    ⚠️ 产物仍是 ``synthetic`` ✓✗（参考 TE / VAE 未训练 ✗）—— 前端必须打标 ✗。
    """
    body = await read_json(request)
    if not str(body.get("outputsDir") or body.get("outputs_dir") or "").strip():
        # ⚠️ 默认给**数据根下的 engine 目录** ✓：不给 ⇒ 管线落到临时目录 ✓✗（前端取不到 ✓）
        body = {**body, "outputsDir": str(_default_outputs_dir())}
    try:
        generation = _generation_request(body)
        assembly = _assembly_from(body)
    except pipe.StageError as err:
        return bad_request(str(err))
    assembly["dryRun"] = bool(body.get("dryRun"))
    task_id = engine_runtime.submit(generation, assembly=assembly, stage=str(assembly.get("stage") or "h3"))
    poll = {"taskId": task_id, "pollUrl": f"/api/v1/engine/tasks/{task_id}"}
    if not bool(body.get("wait")):
        return success({**poll, "status": "queued", "waited": False, "synthetic": True})
    try:
        timeout = float(body.get("timeoutSeconds") or body.get("timeout_seconds") or 0) or None
    except (TypeError, ValueError):
        return bad_request(f"timeoutSeconds 不是数字（收到 {body.get('timeoutSeconds')!r} ✗）")
    task = engine_runtime.wait(task_id, timeout=timeout)
    return success({**poll, "waited": True, "timeoutSeconds": timeout, "task": task})


@router.get("/tasks")
def list_engine_tasks(limit: int = 20) -> Any:
    """最近的任务 ✓（**瘦身版** ✗ 不带事件 ✓ —— 详情走 ``/tasks/{id}`` ✓）。"""
    return success({"tasks": engine_runtime.list_tasks(limit=limit)})


@router.get("/tasks/{task_id}")
def get_engine_task(task_id: str, events_limit_raw: int = Query(200, alias="eventsLimit")) -> Any:
    """单个任务 ✓（带事件 ✓ —— 前端按它画进度 ✓；截断时 ``eventsTruncated`` 会说明 ✓）。"""
    task = engine_runtime.get_task(task_id, events_limit=max(0, int(events_limit_raw)))
    if task is None:
        return bad_request(f"没有这个任务 {task_id!r} ✓（列表见 /api/v1/engine/tasks ✓）")
    return success(task)


@router.post("/tasks/{task_id}/cancel")
def cancel_engine_task(task_id: str) -> Any:
    """请求取消 ✓ —— **排队中的**立刻终结 ✓ / **跑着的**由管线在下一步停 ✗（不是瞬停 ✓）。"""
    task = engine_runtime.cancel(task_id)
    if task is None:
        return bad_request(f"没有这个任务 {task_id!r} ✓（列表见 /api/v1/engine/tasks ✓）")
    return success(task)


# ⚠️ 这里**没有** "同步端点读 body" 的辅助函数 ✗ —— 读不了 ✓✗（`read_json` 是 async ✗）
#   ⇒ 需要读体的端点一律 ``async def`` + ``await read_json`` + ``asyncio.to_thread`` ✓。
