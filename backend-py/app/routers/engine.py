"""自研推理引擎的**体检路由**（2026-09-17）。

为什么需要它（这是用户最痛的两件事 ✓）：

1. **「为什么跑不起来」**——本机现实是「8 个组件、主权重 19.53 GiB」✓，缺一个就跑不动 ✗；
   这个接口**逐条**告诉你：哪个组件缺、哪个坏了（截断/头部矛盾）、还差多少 GB ✓。
2. **「这个文件是不是我要的」**——几十 GB 的 safetensors，**只读头部**几 KB 就能报出
   张量清单/形状/精度 ✓，不用等到加载时才炸 ✗。

* ``GET /api/v1/engine/capabilities`` 引擎能力表（认哪些值 / 几何纪律 / 量化族 / 设备策略 ✓ 纯静态 ✓）
* ``GET /api/v1/engine/readiness``  某阶段的就绪报告（``?stage=h3|image|audio|text`` ✓）
* ``POST /api/v1/engine/inspect``   体检**任意** safetensors 文件（body: ``{"path": "..."}`` ✓）
* ``POST /api/v1/engine/plan``      生成请求 → **具体数字**（尺寸/帧数/σ 序列 ✓ 不跑推理 ✓）
* ``POST /api/v1/engine/canvas-plan``  **画布计划**（多大/为什么/改过哪边 ✓ 含用途档位 ✓ 不跑推理 ✓）
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
SD WebUI / Ollama** ✓、不起任何子进程 ✓；⚠️ 产物**真不真按「已装件」算** ✗✗（2026-11-01 ✓，
旧文档写死「同样标 synthetic」**已失真** ✗）：装的是参考 TE / VAE（未训练 ✓）⇒ ``synthetic: true``
＋ 噪声画面 ✓；装的是**真权重** VAE（``torch_backend.REAL_VAE_KINDS`` ✓）⇒ ``synthetic: false``
＋ 真画面 ✓。两个方向都不许说谎 ✗：把「参考件跑通」说成「真模型出片」✗，把真出片说成噪声 ✗。
⚠️ 三格**分开报**（混成一格就看不出该去装权重还是改代码 ✓✗）：``canGenerate``（此刻能不能出片 ✓）、
``pendingParts``（**这台**还没装什么 ✓）、``codeGaps``（**代码**里还没写什么 ✓）——
``/engine/backends`` 的 ``torch`` 与 ``/engine/runtime`` 两处同源（:data:`torch_backend.CODE_GAPS` ✓）。
"""
from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request

from ..core.config import get_storage_root
from ..core.response import bad_request, success
from ..core.request_utils import read_json
from ..services.engine import bridge
from ..services.engine import canvas as canvas_mod
from ..services.engine import conditioning
from ..services.engine import device_policy as policy_mod
from ..services.engine import facade
from ..services.engine import guidance
from ..services.engine import hybrid_load
from ..services.engine import inventory as inv
from ..services.engine import loader
from ..services.engine import residency as residency_mod
from ..services.engine import runtime
from ..services.engine import upscale as upscaler
from ..services.engine import pipeline as pipe
from ..services.engine import profiles as profiles_mod
from ..services.engine import safetensors as st
from ..services.engine.dryrun import DryRunBackend
from ..services.engine.runtime import engine_runtime
from ..services.engine.torch_backend import (
    TorchBackend,
    TorchBackendUnavailable,
    normalize_vae_tiling,
)

router = APIRouter(prefix="/api/v1/engine", tags=["engine"])


def _engine_error(err: facade.EngineError) -> Any:
    """门面异常 ⇒ 400 + **引擎错误码**（E1.5：错误码只有一个真源 `facade.CODES`）。

    ⚠️ 码与文案**缺一不可** ✗：`message` 给人看（照着能修 ✓），`engineCode` 给机器分流
    （「缺包 / 缺权重 / 量化族不支持」是三种完全不同的处置 ✓）。只在 router 里另写一套字符串
    ⇒ 引擎改了码这里不改，前端就永远收不到新码 ✓✗。
    """
    text = err.message + (f"（{err.hint}）" if err.hint else "")
    return bad_request(text, engine_code=err.code)

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
    # ⭐ 2026-11-01：这三格**以前不在表里** ⇒ 前端送了也被**静默丢掉** ✓✗（不报错 ✓）
    # ``target`` 是用途名 ✓（引擎自己查 :data:`engine.profiles.PROFILES` 算画布 ✓）；
    # ``denoise`` 是起始噪声比例 ✓（<1 ⇒ σ 梯子截尾 + 从 ``initImage`` 起跑 ✓，
    # 两样**必须同时给** ✓ —— 缺一样当场拒 ✓，见 ``pipeline.build_plan`` 的理由 ✓）；
    # ``initImage`` 是图生图的**起点图** ✓（≠ ``referenceFrames`` ✓ 那份是条件 ✓ 这份是起点 ✓）；
    # ``canvas`` 走 :func:`_canvas_from` ✓（对象 ✓ 不是标量 ✓ ⇒ 不在这张标量映射表里 ✗）。
    "target": "target", "purpose": "target", "denoise": "denoise",
    "initImage": "init_image", "init_image": "init_image",
    # ⭐ ``stage`` 以前也**不在**这张表里 ✓✗（``_ASSEMBLY_KEYS`` 有 ✓ 但那是真跑那条路 ✓）——
    # 后果：``/engine/plan`` 传 ``"stage": "qwen_image"`` 会被**静默忽略** ✓✗，永远按 ``h3`` 算
    # σ/帧数 ✓（前端拿着图片请求却看到 124 帧 ✓）。画布档位也按 stage 查 ✓ ⇒ 必须收 ✓。
    "stage": "stage",
    # ⭐ ``lyrics`` 以前也**不在**这张表里 ✓✗（2026-11-06 接 Music3 时发现 ✓）：
    # 后果与上面 ``stage`` 那条**完全同形** ✓ —— 前端送 ``"lyrics": "……"`` ⇒ 被**静默丢掉** ✓✗，
    # ``request.lyrics`` 恒空字符串 ✓ ⇒ 管线以为"用户要纯器乐" ✓、`plan.warnings` 里那句
    # 「没给歌词 ⇒ 按纯器乐出 ✓」还理直气壮 ✓✗。**不报错、只是没唱** ✓ —— 最坏的一类 ✓。
    # ⚠️ 别把它并进 ``prompt`` ✗：参考侧是**两条支路** ✓（caption 走 `<|caption_start|>` ✓、
    # 歌词走自己的段落 ✓），拼成一句 ⇒ 模型把歌词当描述念一遍 ✓✗（见 `te_music3.build_prompt` ✓）。
    "lyrics": "lyrics",
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


def _canvas_from(body: dict[str, Any]) -> Any:
    """请求里的 ``canvas``（``/engine/canvas-plan`` 的产物 ✓）⇒ :class:`canvas.CanvasPlan` ✓。

    ⚠️ 与 ``upscale`` **同一条纪律** ✗：画布**由计划层定** ✓，管线只执行 ✓ ——
    在请求侧再判一次"该吸到几"就又是「两处口径」✓✗。

    ⚠️ 收到的是**像素** ✓ 但我**不直接信** ✗：按 ``stage`` 的吸附步长再过一遍
    :func:`canvas.plan_canvas` ✓（幂等 ✓ —— 已合法的宽高过一遍不变 ✓；
    被人手改坏的 1343×769 会被吸回 1344×768 并**如实写进 notes** ✓）。
    """
    raw = body.get("canvas") or body.get("canvasPlan")
    if raw in (None, "", {}, False):
        return None
    if not isinstance(raw, dict):
        raise pipe.StageError(
            "plan", f"canvas 必须是对象（收到 {type(raw).__name__} ✓）—— "
                    "先调 ``/engine/canvas-plan`` 拿计划 ✓，别自己拼 ✗")
    try:
        width = int(raw.get("width"))
        height = int(raw.get("height"))
    except (TypeError, ValueError) as err:
        raise pipe.StageError(
            "plan", f"canvas 的 width/height 必须是整数 ✗（收到 "
                    f"{raw.get('width')!r}/{raw.get('height')!r} ✓）") from err
    stage = str(body.get("stage") or pipe.VIDEO_STAGE)
    try:
        planned = canvas_mod.plan_canvas(stage, width=width, height=height)
    except ValueError as err:
        raise pipe.StageError("plan", str(err), cause=err) from err
    extra = tuple(str(item) for item in raw.get("notes") or ())
    profile = raw.get("profile")
    if not extra and not profile:
        return planned
    return canvas_mod.CanvasPlan(
        width=planned.width, height=planned.height, megapixels=planned.megapixels,
        ratio=planned.ratio, grid=planned.grid, scheme=planned.scheme,
        source=str(raw.get("source") or planned.source),
        profile=profile or planned.profile, notes=planned.notes + extra)


def _canvas_from_pixels(body: dict[str, Any], stage: str) -> Any:
    """顶层 ``width`` + ``height`` ⇒ :class:`canvas.CanvasPlan` ✓（与 ``canvas`` 同一条纪律 ✓）。

    ⚠️ 这两格原来**根本不在** ``_REQUEST_KEYS`` 里 ✓✗ ⇒ 直接给像素的调用方（脚本 / CLI / 别的客户端 ✓）
    送了也被**静默丢掉** ✓，产物按 ``megapixels × ratio`` 的**默认画布**出 ✓
    —— 症状是"尺寸不对"而不是报错 ✓✗（实测：要 768×768，拿到 1360×768 ✓）。

    吸附与合法性仍只问 :func:`canvas.plan_canvas` ✓（这里**不**再算一遍步长 ✓✗）。
    """
    raw_w, raw_h = body.get("width"), body.get("height")
    has_w, has_h = raw_w not in (None, ""), raw_h not in (None, "")
    if not has_w and not has_h:
        return None
    if has_w != has_h:
        raise pipe.StageError(
            "plan",
            f"尺寸只给了 {'width' if has_w else 'height'} ✗ —— 另一边**不猜** ✓"
            f"（猜出来的比例必然是别的用途 ✓）⇒ 两个都给，或改用 canvas ✓")
    try:
        width, height = int(raw_w), int(raw_h)
    except (TypeError, ValueError) as err:
        raise pipe.StageError(
            "plan", f"width/height 必须是整数 ✗（收到 {raw_w!r}/{raw_h!r} ✓）") from err
    try:
        return canvas_mod.plan_canvas(stage, width=width, height=height)
    except ValueError as err:
        raise pipe.StageError("plan", str(err), cause=err) from err


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

    # ⚠️ ``dryRun`` 原来**不在** ``_REQUEST_KEYS`` 里 ✓✗ ⇒ 回执 ``request.dry_run`` 恒 ``false``
    #    （实测复现 ✓），而真 dry 靠 ``assembly["dryRun"]`` ✓ —— 同一个开关两个说法 ✓✗。
    #    布尔必须**严格解析** ✗：``"false"`` 在 Python 里是 truthy ✓，``bool()`` 会把关读成开 ✓✗。
    for flag_key in ("dryRun", "dry_run"):
        if flag_key in body and body[flag_key] not in (None, ""):
            values["dry_run"] = _as_bool(flag_key, body[flag_key])
            break

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
    planned_canvas = _canvas_from(body)
    if planned_canvas is None:
        # ⚠️ ``canvas`` 与顶层 ``width/height`` 是**同一件事的两种写法** ✓ ⇒ 只认一个 ✓
        planned_canvas = _canvas_from_pixels(
            body, str(values.get("stage") or body.get("stage") or pipe.VIDEO_STAGE))
    if planned_canvas is not None:
        values["canvas"] = planned_canvas
    # ⭐ 画布素材是**本仓 URL** ✓，引擎只读**本机文件** ✗ ⇒ 在这里换成绝对路径 ✓。
    #    不做这步的后果：``/uploads/x.png`` 被当相对路径喂给 `Image.open` ⇒
    #    报 "No such file"，而用户会以为"参考图没上传成功" ✓✗（其实在盘上 ✓）。
    if values.get("first_frame"):
        values["first_frame"] = _local_media(str(values["first_frame"]), what="首帧")
    if values.get("init_image"):
        values["init_image"] = _local_media(str(values["init_image"]), what="起点图")
    if values.get("reference_frames"):
        values["reference_frames"] = tuple(
            _local_media(str(item), what="参考帧") for item in values["reference_frames"])
    return pipe.GenerationRequest(**values)


def _local_media(value: str, *, what: str = "素材") -> str:
    """画布素材地址 → **本机路径** ✓（引擎只读本地文件 ✗ 不吃 URL ✓）。

    三种写法：
    * 绝对路径 ⇒ **原样**（存不存在由管线读文件时说 ✓ —— 这里再判一次只会多一种报错口径 ✗）；
    * ``/uploads/…`` / ``static/…`` ⇒ 解析成绝对路径 ✓，**不在盘上就拒** ✗✗
      （这是本仓发出来的地址 ✓，解析不到 = 画布那张卡被清掉了 ✓，早说是帮忙 ✓）；
    * 裸文件名（``首帧.png`` 这种）⇒ **原样放行** ✓（那是调用方自己的口径 ✓，
      在这里改语义会把既有调用打成 400 ✗）；
    * 远程 URL ⇒ **当场拒** ✗✗（下载是 `comfyui.stage_reference_uploads` 那条路的活 ✓，
      在这里悄悄 `httpx.get` 会把一次外部依赖藏进生成调用里 ✓✗）。
    """
    text = str(value or "").strip()
    if not text:
        return text
    if text.startswith(("http://", "https://")):
        raise pipe.StageError(
            "plan", f"{what}只认**本机文件** ✗，不接受远程地址：{text[:80]} ✓"
                    "（先上传到画布素材再引用 ✓）")
    path = Path(text)
    if path.is_absolute():
        return str(path)
    relative = text.lstrip("/").replace("\\", "/")
    if not relative.startswith(("uploads/", "static/")):
        return text
    from ..services.file_storage import get_absolute_path  # 局部 import ✓（避免循环 ✓）

    resolved = Path(get_absolute_path(relative))
    if not resolved.is_file():
        raise pipe.StageError("plan", f"{what}不在盘上：{text} ✓（画布上那张卡是不是被清掉了？）")
    return str(resolved)


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
    cache_path = str(body.get("cachePath") or "").strip() or None
    try:
        include_final = _flag(body, "includeFinalAdaLn")
        if not body.get("apply"):
            return success(hybrid_load.plan_hybrid_load(base, overlay,
                                                        include_final_adaln=include_final))
        report = hybrid_load.run_hybrid_load(base, overlay, cache_path,
                                             force=_flag(body, "force"),
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
            # ⚠️ 用 ``load_upscaler_contract``（读 + **结构核** ✓）而不是 ``read_upscaler_contract`` ✗：
            #    后者只解出契约 ⇒ 缺字段的契约会被这条接口当成「2× 可做」**静默绿灯** ✓✗
            #    （真跑时张量层按默认值建网络 ⇒ 形状自洽、与真权重不是一套 ✓✗）。
            #    结构不合法 ⇒ 与"读不出来"**同一档**处理 ✓：``contract=None`` ⇒ 下面按
            #    ``plan_upscale`` 的规则**回退普通模式** ✓ 并把真因写进 ``contractError`` ✓。
            contract = upscaler.load_upscaler_contract(metadata)
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


@router.post("/canvas-plan")
async def canvas_plan(request: Request) -> Any:
    """**画布计划** ✓：这次出多大、为什么是这个数、吸附时改了哪边 ✓（纯几何 ✓ **不跑推理** ✗、**不读权重** ✗）。

    三种入口（二选一 ✓，同时给 ⇒ 400 ✗ 不猜优先级 ✗）：

    * ``target``（``"character-sheet"`` ✓）⇒ 查 :data:`engine.profiles.PROFILES` ✓ 并回执 ``sampling``
      （该档的步数/cfg/采样器 ✓ —— 前端可以直接照抄 ✓）；
    * ``megapixels`` + ``ratio`` ⇒ 按该 ``stage`` 的算法算 ✓（``zimage``/``krea2`` 走 ``ResolutionSelector`` ✓，
      ``qwen_image``/``qwen_image21``/``h3`` 走 ``megapixels`` ✓ —— 见 ``canvas.STAGE_CANVAS`` ✓）；
    * ``width`` + ``height`` ⇒ 只做吸附 ✓（两个都给才收 ✓）。

    三条口径（与 ``/upscale-plan`` 同风格 ✓）：
    1. ⭐ **吸附步长按族给** ✓：``h3`` 32 ✓、``sdxl`` 8 ✓、图族 16 ✓ —— 全局一个数是错的 ✓✗；
    2. ⭐ **``1344×768`` 与 ``16:9`` 不闭合**（1.75 vs 1.778 ✓）⇒ 档位命中时**像素优先** ✓，
       比例差 >3% 才按档位面积重算 ✓ 且 ``notes`` 写明 ✓（不悄悄换画幅 ✗）；
    3. ⭐ **表里没定画布的档 ⇒ 400** ✗（H3 由导演台出画布 ✓）—— 本仓**不猜**一个"看着合理"的数 ✓✗。
    """
    body = await read_json(request)
    stage = str(body.get("stage") or pipe.VIDEO_STAGE)
    # ⚠️ ``purpose`` / ``profile`` 都当 ``target`` 收 ✓（前端三种叫法都见过 ✓）；空串视为没给 ✓。
    target = next((str(body[key]) for key in ("target", "purpose", "profile")
                   if body.get(key) not in (None, "")), None)

    raw_mp = body.get("megapixels") if body.get("megapixels") not in (None, "") else body.get("megapixelsBudget")
    try:
        megapixels = float(raw_mp) if raw_mp not in (None, "") else None
    except (TypeError, ValueError):
        return bad_request(f"megapixels 必须是数字（收到 {raw_mp!r} ✗）")

    raw_ratio: Any = body.get("ratio") if body.get("ratio") not in (None, "") else body.get("aspectRatio")
    if isinstance(raw_ratio, (list, tuple)):
        try:
            raw_ratio = tuple(float(item) for item in raw_ratio if item is not None)
        except (TypeError, ValueError):
            return bad_request(f"ratio 列表必须是数字（收到 {raw_ratio!r} ✗）")
    elif isinstance(raw_ratio, str) and not raw_ratio.strip():
        raw_ratio = None

    try:
        width = int(body["width"]) if body.get("width") not in (None, "") else None
        height = int(body["height"]) if body.get("height") not in (None, "") else None
    except (TypeError, ValueError):
        return bad_request(
            f"width/height 必须是整数（收到 {body.get('width')!r}/{body.get('height')!r} ✗）")

    try:
        planned = canvas_mod.plan_canvas(stage, target=target, megapixels=megapixels,
                                         ratio=raw_ratio, width=width, height=height)
    except ValueError as err:
        return bad_request(str(err))

    payload: dict[str, Any] = planned.to_dict()
    payload["availableTargets"] = [item.to_dict() for item in profiles_mod.for_stage(stage)
                                   ] if stage in canvas_mod.STAGE_CANVAS else []
    if planned.profile:
        entry = profiles_mod.PROFILES[planned.profile]
        payload["label"] = entry.label
        # ⚠️ 这几格是**该档的采样档** ✓（用不用由调用方定 ✓）—— 回执里给出来是为了
        #    「照工作流跑」不必再手抄四个数 ✓（抄漏一个就是糊图且不报错 ✓✗）。
        payload["sampling"] = {"steps": entry.steps, "cfg": entry.cfg,
                              "sampler": entry.sampler, "schedule": entry.schedule,
                              "denoise": entry.denoise,
                              # ⚠️ ``denoise<1`` 的档（``storyboard-last`` ✓ 0.55）**必须**
                              #    连 ``initImage`` 一起传 ✓✗ —— 提前说清 ✓，否则前端照抄这份
                              #    ``sampling`` 会撞一个看不懂的 400 ✓。
                              "requiresInitImage": entry.requires_init_image}
    payload["stages"] = sorted(canvas_mod.STAGE_CANVAS)
    return success(payload)


@router.get("/stages")
def engine_stages() -> Any:
    """**可跑档位** ✓（画布"选哪一档"就读它）：每档挑到哪些件、缺什么、要多少 GiB ✓。

    ⚠️ 纯读清单 + ``stat`` 文件 ✓（**不装权重** ✗、**不占显存** ✗）⇒ 画布一打开就问一次，
    毫秒级 ✓。与 :func:`inventory.assembly_for_stage` 走**同一条**解析 ✓ ——
    这里显示"会挑到哪几件"，提交时挑到的就是那几件 ✓（两处不一致 = 显示一套、跑另一套 ✗）。
    """
    plan = [
        (pipe.ZIMAGE_STAGE, "Z-Image（文生图）"),
        (pipe.KREA2_STAGE, "Krea2（文生图）"),
        (pipe.QWEN21_STAGE, "Qwen-Image 2.1（文生图）"),
        (pipe.QWEN_IMAGE_STAGE, "Qwen-Image 2512（文生图）"),
        (pipe.WAN22_STAGE, "Wan 2.2 I2V（首帧 → 视频）"),
        (pipe.VIDEO_STAGE, "H3 双流（文生视频 + 音频）"),
        #: ⭐ **Music3**（2026-11-06 加 ✓）：⚠️ 标签**必须**写明产物是音频 ✗✗ ——
        #: 上面五档全是画面 ✓，前端把「有 stage 就有画面」当默认 ✓✗（同一张卡片渲染 ✓、
        #: 预览拿 `primaryUrl` 当 video 用 ✓）⇒ 它拿到的是 `.wav` ✓，症状是「预览黑了」✓✗
        #: （不是"没跑出来"✓）。所以标签里带「**音频**」✓，让用户看一眼就知道该拿什么 ✓。
        #: ⚠️ 同理 ``fits`` 也算得出来：13.34 GiB < 22.49 ✓ ⇒ 本档是**唯一**一条
        #: 「件齐 且 装得下 且 不用分时」的音乐线 ✓。
        (pipe.MUSIC3_STAGE, "Music3（文生音乐 · **产物是音频**，不是画面）"),
    ]
    rows: list[dict[str, Any]] = []
    for stage, label in plan:
        assembly, note = _auto_assembly(stage, {})
        #: 只回**文件名** ✓：完整路径是本机事实，前端拿它没用 ✓（排障走 ``/readiness`` ✓）
        components = {key: Path(str(value)).name for key, value in assembly.items()
                      if isinstance(value, str) and key.endswith(("_path", "weights"))}
        ready = inv.readiness(stage)
        vram = ready.get("vram") or {}
        rows.append({
            "stage": stage,
            "label": label,
            "components": components,
            #: ``ready`` ＝ **件齐**（清单层面 ✓）；``fits`` ＝ **装得下**（显存层面 ✓）——
            #: 两者分开报 ✗✗：``qwen_image`` 件齐但 47 GiB 装不下 24 GB 卡 ✓，
            #: 合成一个布尔值就会让画布以为"能跑"，点下去装到一半 OOM ✓。
            "ready": bool(ready.get("ready")),
            "fits": vram.get("fits"),
            "estimatedGiB": vram.get("estimatedGiB"),
            "requiredGiB": ready.get("requiredWeightsGiB"),
            "missing": ready.get("missingRequired") or [],
            "note": note or None,
        })
    return success({"stages": rows})


@router.get("/capabilities")
def engine_capabilities() -> Any:
    """**引擎能力表** ✓ —— 这台引擎认哪些值、每族的几何纪律、量化族、设备策略 ✓（**纯静态** ✗ 不读卡 ✗）。

    ⚠️ 这条路由以前**没有** ✗✗（``facade.capabilities()`` 只被内部用 ✓）：后果是前端想知道
    「支持哪些采样器 / 调度器 / 用途档 / 吸附步长」只能**读代码猜** ✓✗ —— 猜错的代价在
    **没校验**的路径上就是「静默按默认跑」✓✗（采样器拼错不报错，出图却和以为的不是一回事 ✓）。

    三条口径：
    1. ⭐ **零副作用** ✗✗：不 import torch ✓、不读权重 ✓、**不探显存** ✓（卡况在 ``GET /engine/runtime``
       的 ``devicePolicy`` ✓ —— 混进来这张表就随机器负载变 ✓✗，落盘事实也没法与它对账 ✓）；
    2. ⭐ 值**从注册表取** ✓ 不手抄 ✗（``samplers``/``schedules``/``profiles`` 各只有一个真源 ✓，
       手抄一份必漂移 ✓）；
    3. ⭐ **能力 ≠ 已验收** ✗：``unverified`` 单列 ✓（能装能跑一次，和有性能基线是两件事 ✓）。
    """
    return success(facade.capabilities())


@router.get("/backends")
def backends() -> Any:
    """**推理后端现状** ✓ —— 现在能跑什么、要装什么、装完还差什么 ✓。

    这条接口专治一种含糊：只说「后端不可用」✗ ⇒ 分不清是**依赖没装**（去装包 ✓）
    还是**实现待写**（去写代码 ✓）。两者治法完全不同 ✗（见 `torch_backend` 的 ``reason`` ✓）。

    ⚠️ ``torch`` 那一份里的三格**各答一件事** ✗✗（2026-11-01 ✓）：``canGenerate``（此刻能不能出片 ✓，
    按**已装件**算 ⇒ 装齐就是 ``true``）、``pendingParts``（**这台**还缺哪几件 ✓）、
    ``codeGaps``（**代码**里还没写的能力 ✓ —— 它不是 ``knownGaps`` ✗，后者是 ``capabilities()``
    的**环境**缺口 ✓）。曾经 ``canGenerate`` **写死** ``false``、``pendingParts`` 塞的是一堆**沿革散文**
    ⇒ 装了整链、真出过 mp4 之后本接口仍报「不能出片」✓✗（UI 唯一的自述口 ✓ ⇒ 整条引擎看着都不能用 ✗）。
    """
    return success({
        "dryrun": {
            "name": "dryrun", "synthetic": True, "available": True, "canGenerate": False,
            "note": "零依赖干跑 ✓：只验编排（阶段/事件/取消/归因/引导/首帧 ✓），**不产生画面** ✗",
        },
        "torch": TorchBackend().describe(),
        "guidance": {"supportsCfg": True, "note": "引导由引擎按步计算 ✓ 与后端无关 ✓"},
        "conditioning": {"supportsFirstFrame": True,
                         # 尾帧（fl2v ✓）：双流走 `cond` 段（下标 frames−1 ✓）；
                         # **单流**（只有 condition_first_frame）做不到 ⇒ 管线会明确拒 ✗ 不静默丢 ✓
                         "supportsLastFrame": True,
                         "lastFrameBackends": ["torch（双流形态 ✓）"],
                         "note": "引擎算掩码 ✓、后端套到自己的布局上 ✓（不猜布局 ✗）；"
                                 "尾帧只有双流能接 ✓，单流给 lastFrame 会当场失败 ✓"},
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
    #: ⚠️ 走**门面**而不是直接调 `loader.plan_stage` ✗（E1.3 唯一入口）：CLI 与 HTTP 必须同一份
    #: 结论 ✓ —— 两处各算一遍，版本号与错误码迟早只在一处更新 ✓✗。
    try:
        return success(facade.plan(stage, root=root, capacity_gib=max(1.0, capacity)))
    except facade.EngineError as err:
        return _engine_error(err)


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
    try:
        body_events = _flag(body, "includeEvents")
        result = pipe.run_sync(_generation_request(body), DryRunBackend(),
                               on_event=events.append if body_events else None)
    except pipe.StageError as err:
        # ⚠️ ``_generation_request`` 验参也是抛 ``StageError`` ✓ —— 老代码没接住 ⇒ 500 ✓✗
        #    （参数写错本该是 400 + 说清哪个字段 ✗）
        return bad_request(str(err))
    payload = result.to_dict()
    if body_events:
        payload["events"] = events
    return success(payload)


# ── 运行时：真跑那条路（**进程内** ✓ 不依赖任何外部服务 ✓）──────────────────────
#: 装配字段（前端 camelCase ✓ / 后端 snake_case ✓ 都收 ✓ 取第一个非空的 ✓）
_ASSEMBLY_KEYS: dict[str, str] = {
    "stage": "stage",
    "ditPath": "dit_path", "dit_path": "dit_path",
    # ⭐ **第二份主干** ✓（2026-11-02 加 ✓）：Wan2.2 I2V 是**双专家** ✓ —— `ditPath` 高噪、
    #    `ditPath2` 低噪 ✓。⚠️ 不进这张表 ⇒ 传了被**静默丢掉** ✓✗，采样中途换装时"低噪那份"仍是
    #    高噪 ✓（症状＝后段细节照样糊，且一条报错都没有 ✓✗）。
    "ditPath2": "dit_path2", "dit_path2": "dit_path2",
    "tokenizerPath": "tokenizer_path", "tokenizer_path": "tokenizer_path",
    "device": "device",
    "audioLatentMode": "audio_latent_mode", "audio_latent_mode": "audio_latent_mode",
    "attachVae": "attach_video_vae", "attachVideoVae": "attach_video_vae",
    "attach_video_vae": "attach_video_vae",
    "attachAudioVae": "attach_audio_vae", "attach_audio_vae": "attach_audio_vae",
    # ⚠️ 真 VAE 三个口 ✓（2026-10-29 ✓）：**默认不出现** ⇒ 走参考族 ✓（不给 ≠ 给 false ✓ 但两者
    #    在装出来的东西上没差别 ✓，差别只在报告里"没要求"这件事没有记录 ✗ —— 保持"不出现"更诚实 ✓）
    "realVideoVae": "real_video_vae", "real_video_vae": "real_video_vae",
    "videoVaeDevice": "video_vae_device", "video_vae_device": "video_vae_device",
    "videoVaeWeights": "video_vae_weights", "video_vae_weights": "video_vae_weights",
    # ⭐ **音频** VAE 两格 ✓（2026-11-04 ✓）：H3 是双流 ✓，音轨走的是**另一族** VAE ✓✗
    #    （DAC 血统编码 ＋ BigVGAN 解码 ✓），视频那份顶不了 ✓ —— 上面三条只管画面 ✓。
    #    ⚠️ 不加进这张表 ⇒ 传了被**静默丢掉** ✓✗，然后 `_attach_reference_audio_vae` 照旧挂
    #    参考实现 ⇒ **音轨是噪声而回执不报** ✓✗（正是上面 `int8Offload` 那条同款漂移 ✓）。
    "realAudioVae": "real_audio_vae", "real_audio_vae": "real_audio_vae",
    "audioVaeWeights": "audio_vae_weights", "audio_vae_weights": "audio_vae_weights",
    # ⭐ Z-Image 那两格 ✓（2026-10-31 ✓）：`stage=zimage` 要**三份权重** ✓ —— DiT 走 `ditPath` ✓，
    #    另两份在这里 ✓。⚠️ 不加进这张表 ⇒ 传了也被**静默丢掉** ✓✗，然后装配报「缺 textWeights」✓✗
    #    —— 用户明明给了，报错却说没给（正是 2026-10-29 `int8_offload` 那次漂移的同款 ✓）。
    "textWeights": "text_weights", "text_weights": "text_weights",
    "imageVaeWeights": "image_vae_weights", "image_vae_weights": "image_vae_weights",
    # ⚠️ `int8_offload` **必须在这张表里** ✗✗（2026-10-29 补 ✓）：`bridge.assembly_from_config`
    #    早就认它 ✓，这张表却没有 ⇒ 同一个开关在两条路上**行为不同** ✓✗ ——
    #    走出图服务的 config ⇒ 生效 ✓；走 `POST /engine/load` ⇒ 被白名单**静默丢掉** ✓✗
    #    （用户以为开了，报告里也不说没收到 ⇒ 在 720×1280 上照样 OOM 且找不到原因 ✓✗）。
    #    白名单是**逐项手抄**的 ✓ ⇒ 这种漂移只能靠判据盯（见 `engine_router_contract_test` ⑪′ ✓）。
    "int8Offload": "int8_offload", "int8_offload": "int8_offload",
    # ⭐ **卡 + CPU 联合跑** ✓（2026-10-31 加 ✓ —— 上一条注释说的"漂移"正是这次要避的 ✓）：
    #    取值清单在 `residency.MODES` ✓，这里**只搬不改** ✗，校验在下面对 `residency` 那段 ✓。
    "residency": "residency",
    "residencyBudgetMib": "residency_budget_mib", "residency_budget_mib": "residency_budget_mib",
    # ⭐ **设备策略** ✓（2026-10-31 加 ✓）：`auto`（默认，GPU 优先、压力大才转移）/ `gpu` / `cpu`
    #    / `balanced` ✓ —— 合法清单在 `device_policy.POLICIES` ✓，归一与校验都交给
    #    `normalize_policy` ✓（在下面 `_assembly_from` 里**调一次** ✓✗，白名单**只搬不改** ✗
    #    —— 否则同一份合法值又在这里手抄一遍 ✓✗）。
    "policy": "policy", "devicePolicy": "policy", "device_policy": "policy",
    # ⚠️ `loras`（2026-10-29 收 ✓）：以前**故意不收** ✗（裸透传会跳过归一 ✗），改成
    #    **收下来再走同一份归一** ✓ —— `bridge.lora_specs_from_settings` ✓（跟 config 那条路
    #    是**同一个函数** ✓，不在这里手抄 ✗✗）。两种写法都收：`loras`（清单）/ `lora`（单条）。
    #    ⚠️ 值这里先原样搬 ✓，真正归一在 `_assembly_from` 后面那段（要读 `modelsDir` ✓）。
    "loras": "loras", "lora": "loras",
    # ⭐ **VAE 解码省显存旋钮** ✓（2026-11-02 收 ✓）：对象形
    #    `{"tile": 512, "halo": 16, "attnChunk": 1024, "fallback": true}` ✓ ——
    #    合法键与取值全归 `torch_backend.normalize_vae_tiling` ✓（下面 `_assembly_from`
    #    里**调一次** ✓✗，白名单**只搬不改** ✗ —— 否则同一份规则又在这里手抄一遍 ✓✗）。
    #    ⚠️ 不进这张表 ⇒ 传了被**静默丢掉** ✓✗，用户以为"已经分块了"、其实全程整张解 ✓✗，
    #    3072×2304 上照样 OOM 且回执里查不到任何线索 ✓✗。
    "vaeTiling": "vae_tiling", "vae_tiling": "vae_tiling",
    "force": "force",
}

#: 取整口径的合法值 ✓（口径在 `torch_backend.dualStream` ✓）——
#: ⚠️ 不认识 ⇒ **报错并列出合法的** ✗ 且**不给默认值** ✗✗（本仓老账：好默认值会把"未核实"伪装成"已实现" ✓）
_AUDIO_LATENT_MODES = ("round", "ceil", "floor")


def _bool_from(body: dict[str, Any], *names: str) -> bool | None:
    """取布尔 ✓（``None`` = 没给 ✓ —— 与"给了 false"分得开 ✓✗）。"""
    for name in names:
        if name in body and body[name] not in (None, ""):
            return _as_bool(name, body[name])
    return None


def _auto_assembly(stage: str, explicit: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """**显式优先、缺的从清单补** ✓ ⇒ ``(装配参数, 说明)``。

    为什么需要它：画布/前端只该报「跑哪一档」✓ —— 让它填
    ``ditPath`` / ``textWeights`` / ``imageVaeWeights`` / ``tokenizerPath`` 四格路径，
    等于把清单解析搬到前端 ✗（一份清单两处解析，迟早漂 ✓✗）。

    ⚠️ ``note`` **不进装配** ✗✗：``ensure_loaded`` 只认白名单键 ✓，多一个键就是
    ``TypeError`` ⇒ 整个请求 500 ✓✗（这个坑本仓踩过：白名单漏一项就静默丢参数 ✓）。
    所以说明单独返回，由调用方放进回执 ✓。
    """
    auto = inv.assembly_for_stage(stage)
    note = str(auto.pop("note", "") or "")
    merged: dict[str, Any] = {**auto, **explicit}
    merged["stage"] = stage
    return merged, note


def _stage_of(body: dict[str, Any], assembly: dict[str, Any]) -> str:
    """这次要跑哪一档 ✓（显式 ``stage`` > 装配里的 > 默认视频 ✓）。"""
    raw = assembly.get("stage") or body.get("stage") or "h3"
    return str(raw).strip().lower()


def _flag(body: dict[str, Any], name: str, default: bool = False) -> bool:
    """取一个**必有值**的开关 ✓（没给 ⇒ ``default`` ✓）。"""
    value = _bool_from(body, name)
    return default if value is None else value


def _as_bool(key: str, value: Any) -> bool:
    """严格布尔 ✓ —— 不认就**报错** ✗✗（与 ``audio_latent_mode`` 那条同一个口径 ✓）。

    ⚠️ 为什么不能用 ``bool()`` ✗✗：``bool("false")`` 是 **True** ✓✗ —— 前端把开关序列化成
    字符串（``"false"`` / ``"0"`` / ``"no"`` 都见过 ✓）时，``bool()`` 会把**关**读成**开** ✓✗，
    而且一声不响：``int8Offload: "false"`` 会让人以为「我没开」，实际每步多一趟 PCIe ✓✗。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise pipe.StageError(
        "load", f"{key} 只认 true / false ✗，收到 {value!r} ✗"
                "（不认的值**不许猜** ✗ —— 猜错就是「以为关着、其实开着」✓✗）")


def _assembly_from(body: dict[str, Any]) -> dict[str, Any]:
    """JSON → :meth:`runtime.EngineRuntime.ensure_loaded` 的**关键字参数** ✓（不给的项**不出现** ✗ ⇒ 走默认 ✓）。"""
    values: dict[str, Any] = {}
    for source, target in _ASSEMBLY_KEYS.items():
        if target in values or source not in body or body[source] in (None, ""):
            continue
        values[target] = body[source]
    for key in ("dit_path", "dit_path2", "tokenizer_path", "device", "stage",
                "video_vae_device", "video_vae_weights",
                "text_weights", "image_vae_weights"):
        if key in values:
            values[key] = str(values[key])
    if "residency" in values:
        mode = str(values["residency"]).strip().lower()
        # ⚠️ 不认的取值**当场报** ✗✗（静默当成 ``off`` ⇒ 用户以为开了联合驻留、实际整份在内存里爬 ✓✗）
        if mode not in residency_mod.MODES:
            raise pipe.StageError(
                "plan", f"未知 residency {mode!r} ✗ ⇒ 只能是 {' / '.join(residency_mod.MODES)} ✓"
                        "（`split` 卡上算前缀 / `park` 权重留内存逐层搬 / `off` 不切 ✓）")
        values["residency"] = mode
    if "residency_budget_mib" in values:
        raw = values["residency_budget_mib"]
        try:
            mib = float(raw)
        except (TypeError, ValueError) as exc:
            raise pipe.StageError(
                "plan", f"residencyBudgetMib 只认数字（MiB）✗，收到 {raw!r} ✗") from exc
        # ⚠️ 负预算 = 让整份退回 CPU 的**假配置** ✗（报告里会被判"没联合"✓✗）⇒ 直接报 ✗
        if not math.isfinite(mib) or mib < 0:
            raise pipe.StageError(
                "plan", f"residencyBudgetMib 只认 ≥0 的有限数 ✗，收到 {raw!r} ✗"
                        "（不想设预算就**别写这一项** ✓ —— 装载时按当时空闲显存现算 ✓）")
        values["residency_budget_mib"] = mib
        # ⚠️ **只给预算、不给模式** ⇒ 当场报 ✗✗：预算只在 `residency` 开着时才有意义 ✓ ——
        #    收下它再让 `off` 吞掉，用户会以为"预算都给了、肯定在切" ✓✗（正是"配了没生效"那种坑 ✓）。
        if str(values.get("residency") or "off") == "off":
            raise pipe.StageError(
                "plan", "给了 `residencyBudgetMib` 但 **`residency` 没开** ✗ ⇒ 预算无处可用 ✓"
                        "（要联合跑就同时给 `residency: split` 或 `park` ✓）")
    if "policy" in values:
        # ⚠️ 在这里就归一 ✗✗（**不**留到 `ensure_loaded` ✓）：拼错要 **400**、不是 **500** ✓ ——
        #    `normalize_policy` 抛的是 `ValueError` ✓，而 `/load` 只接 `EngineBusy` /
        #    `TorchBackendUnavailable` ⇒ 漏出去就是 500 ✓✗，用户看到「服务器错误」，
        #    根本不知道自己只是把 `balanced` 拼成了一个别的词 ✓✗（`residency` /
        #    `audioLatentMode` 那两处早就是当场 400 ✓ —— 同一份口径不能只做一半 ✗）。
        #    ⚠️ 归一仍调**同一个函数** ✓（`device_policy.normalize_policy` ✓）⇒ 别名表
        #    （`gpu-first` / `cuda` / `joint` … ✓）只有一份 ✓，不在这里手抄 ✗。
        try:
            values["policy"] = policy_mod.normalize_policy(values["policy"])
        except ValueError as err:
            raise pipe.StageError("load", str(err)) from err
    if "vae_tiling" in values:
        # ⚠️ 同上 ✗✗：**在路由层就归一**（不是留到 `ensure_loaded` ✓）—— 拼错的键 / 非数字
        #    要 **400**、不是 **500** ✓；而 `normalize_vae_tiling` 抛的是 `ValueError` ✓，
        #    `/load` 不接它 ⇒ 漏出去就是「服务器错误」✓✗（用户只是把 `attnChunk` 拼成了
        #    `attn_chunks` ✓，该被告知合法的键有哪些 ✓）。
        #    ⚠️ 归一仍调**同一个函数** ✓（`torch_backend.normalize_vae_tiling` ✓）⇒
        #    「哪个键算数」只有一份 ✓，不在这里手抄 ✗（白名单**只搬不改** ✓）。
        try:
            values["vae_tiling"] = normalize_vae_tiling(values["vae_tiling"])
        except ValueError as err:
            raise pipe.StageError("load", str(err)) from err
        #: 全关就等于没配 ✓（装配默认本来就是全关 ✓）⇒ 不往 `values` 里放一个"假配置" ✗
        #: （放了会让装载键多一格、白装一遍 ✓✗）。
        if values["vae_tiling"] == normalize_vae_tiling(None):
            values.pop("vae_tiling")
    if "audio_latent_mode" in values:
        mode = str(values["audio_latent_mode"]).strip()
        # ⚠️ 不认的口径**必须报错** ✗（静默当成单流 = 用户以为双流开着 ✓✗）
        if mode not in _AUDIO_LATENT_MODES:
            raise pipe.StageError(
                "plan", f"未知 audioLatentMode {mode!r} ✗ ⇒ 只能是 "
                        f"{' / '.join(_AUDIO_LATENT_MODES)} ✓（不给 ⇒ 双流不启用 ✓）")
        values["audio_latent_mode"] = mode
    for key in ("attach_video_vae", "attach_audio_vae", "real_video_vae", "real_audio_vae",
                "int8_offload", "force"):
        if key in values:
            values[key] = _as_bool(key, values[key])
    # ⚠️ ``loras`` **收下来、但必须走归一** ✗✗（2026-10-29 ✓）：白名单只是把它搬进 ``values`` ✓，
    #    裸用等于跳过三件必做的事 —— 相对路径按 ``modelsDir`` 解析 ✓、**文件不存在当场报** ✓、
    #    strength/name 归一 ✓（相对路径会按**进程当前目录**解析 ⇒ 指到别处 ✗）。
    #    ⚠️ 归一**调 config 那条路的同一份函数** ✓（``bridge.lora_specs_from_settings`` ✓）——
    #    在这里手抄一份 = 埋一次漂移 ✗✗（``int8_offload`` 就是在两条路上行为不同漏过一次 ✓）。
    #    ⚠️ 两个键**一起递进去** ✓：``loras`` 非空时它压过 ``lora`` ✓ —— 那条优先级本来就在
    #    归一里写着 ✓，在这里先挑就等于又抄了一遍 ✓✗。
    #    ⚠️ 报错要翻成 ``StageError`` ✗：``ValueError`` 漏出去就是 **500** ✓✗，而这明明是
    #    请求写错 ⇒ 该 400 + 说清是哪个文件 ✓。
    if "loras" in values:
        try:
            values["loras"] = bridge.lora_assembly(bridge.lora_specs_from_settings({
                "loras": body.get("loras"), "lora": body.get("lora"),
                "models_dir": body.get("modelsDir") or body.get("models_dir")}))
        except (ValueError, TypeError) as err:
            raise pipe.StageError("load", str(err)) from err
    return values


def _default_outputs_dir() -> Path:
    """产物默认落 ``<数据根>/static/engine`` ✓ —— 这样前端能直接用 ``/static`` 取 ✓。"""
    return Path(get_storage_root()) / "engine"


def _with_urls(task: dict[str, Any] | None) -> dict[str, Any] | None:
    """任务回执补 ``primaryUrl`` ✓ —— ``primaryPath`` 是**本机绝对路径** ✗，前端拿它没用 ✓。

    只转 ``<数据根>/static`` 下的产物 ✓：别的目录（调用方自给的 ``outputsDir`` ✓）不在
    静态挂载里 ✗，硬转 URL 只会 404 ✓✗（比不给更误导 ✓）。
    """
    if not isinstance(task, dict) or not task.get("primaryPath"):
        return task
    root = Path(get_storage_root()).resolve()
    try:
        rel = Path(str(task["primaryPath"])).resolve().relative_to(root)
    except ValueError:
        return task
    return {**task, "primaryUrl": f"/static/{rel.as_posix()}"}


@router.get("/runtime")
def runtime_status() -> Any:
    """运行时现状 ✓：依赖齐不齐 / 装了哪几个组件 / 忙不忙 / 队列 + 最近任务 ✓。

    ⚠️ ``available=false`` 与 ``loaded.dit=false`` 是**两件事** ✗：前者是 ``torch`` 没装 ✓，
    后者是权重没装 ✓（治法完全不同 ✓ —— 见 ``reason`` ✓）。

    ⚠️ 走**门面** `status()`（E1.3 唯一入口）：CLI 的 `status` 与本路返回**同一份事实** ✓，
    并多带 `schemaVersion` / `apiVersion` ✓（旧字段一个不少 ✓）。
    """
    try:
        return success(facade.status())
    except facade.EngineError as err:
        return _engine_error(err)


@router.post("/load")
async def load_runtime(request: Request) -> Any:
    """**装齐引擎** ✓（主 DiT ⇒ TE ⇒ VAE ✓ 幂等 ✓）。

    ⚠️ 装载是**阻塞**的 ✗（19.53 GiB 光读盘就很久 ✓）⇒ 用 ``to_thread`` 扔出事件循环 ✓✗
    （**不**在事件循环里直接调 ✓：那会把整个后端卡死 ✓✗ —— 本模块第一条纪律 ✓）。
    装不上 ⇒ **400 + 缺什么** ✓（``reason``: ``deps`` 依赖没装 / ``pending`` 权重或结构没到 ✓）。

    ## 设备策略 ``policy`` ✓（2026-10-31 ✓，不写默认走 ``auto`` ✓）

    「谁跑在哪台设备上」**每次装载前现测** ✓，不是构造时定死的开关 ✓✗。四种取值：

    * ``auto``（**默认** ✓）—— GPU 优先 ✓，装了才看卡上有多满：够 ⇒ 上卡 ✓；
      差一点 ⇒ 上卡 + ``residency=split``（GPU/CPU **联合** ✓）；差太多 ⇒ 整件转 CPU ✓（慢，但不失败 ✓）；
    * ``gpu`` —— 钉住上卡 ✓（装不下就 OOM ✓✗，**不替你降级** ✗：这是调用方的决定 ✓）；
    * ``cpu`` —— 钉住内存 ✓（fp32 ✓，慢但稳 ✓）；
    * ``balanced`` —— 余量足时**主动**摊开 ✓（省显存换一点速度 ✓）。

    三种写法都收 ✓：``policy`` / ``devicePolicy`` / ``device_policy``（取第一个非空的 ✓）。
    别名也认（``gpu-first`` ⇒ ``auto`` ✓、``cuda`` ⇒ ``gpu`` ✓、``joint`` ⇒ ``balanced`` ✓）。
    ⚠️ 拼错**当场 400** ✓✗（不静默回落 ``auto`` ✗ —— 回落的后果是「你以为钉了 gpu、其实被挪去 CPU」✓✗）。
    ⚠️ 同时给了显式 ``device`` ⇒ **以 device 为准、策略不动它** ✗（钉死的设备是调用方的决定 ✓）；
    此时 ``policy`` 只影响**后续阶段**（``enter_phase`` 换装时）✓。

    ## 回执里怎么读「为什么在这台设备上」✓

    * ``loadReport.devicePolicy`` —— 这台后端**现在**的策略实况 ✓（``policy`` / ``pressure``
      当时的卡况 ✓ / ``partNeedGiB`` 分时账 ✓ / ``lastDecision`` 最近一次判定 ✓）；
    * ``loadReport.chainDevicePolicy`` —— **这次装配之前**按整链账判的那一次 ✓（``null`` ＝ 没判 ✓，
      比如显式给了 ``device`` ✓）。两者答的是不同问题 ✓✗：前者答"现在什么状态"，
      后者答"这次为什么整条链落在 CPU 上" ✓（只看 ``device`` 会把一次降级读成能力缺失 ✓✗）；
    * ``runtime.devicePolicy`` —— 同上那份实况 ✓（前端轮询 ``GET /engine/runtime`` 时读它 ✓）。

    ## VAE 解码省显存 ``vaeTiling`` ✓（2026-11-02 ✓，不写就是**全关** ✓）

    对象形，五个键都可省 ✓：``{"tile": 512, "halo": 16, "attnChunk": 1024, "temporalChunk": 8,
    "fallback": true}``（``attn_chunk`` / ``temporal_chunk`` 这种 snake 写法也认 ✓）。

    * ``tile`` —— **像素**口径（与 ComfyUI ``VAEDecodeTiled`` 同 ✓），切「全图注意力**之后**」
      的 up 链 ✓；``halo`` 是不用的重叠边（给 0 也会被抬到感受野下限 ✓ —— 这条**不信调用方** ✗）；
    * ``attnChunk`` —— 注意力按 **query** 分块 ✓（数学上无近似 ✓）；
    * ``temporalChunk`` —— **只对 SEEDVR2 那族**有效 ✓，单位是**潜帧** ✓：沿时间维切块解码 ✓，
      每块带**块外真实上文** ⇒ 结果与整段**相等** ✓✗（`engine_seedvr2_vae_test` ⑫ 兜住 ✓）；
      给别的族就是闲着 ✓（不报错 ✗）；
    * ``fallback``（**默认 true** ✓）—— 只在**真 OOM** 时降档重试 ✓，回执里 ``reason="oom"`` ✓。
      ⚠️ **降的是哪一档，按族不同** ✓✗：Wan 那族降 ``tile``/``attnChunk`` ⇒ **字节会变** ✗；
      SEEDVR2 那族降的是 ``temporalChunk`` 阶梯（``4→2→1`` ✓，整条试遍仍 OOM 才报人话 ✓）⇒
      **不动数值** ✓，回执里多一格 ``"exact": true`` ✓（两种降档要分开读，别拿 "reason=oom"
      一个词推断画质变没变 ✗✗）。
      ⭐ SEEDVR2 那族还有一条**不靠 OOM** 的兜底 ✓✗：这份卡实测 ``chunk=4`` **不报 OOM 而是溢出到
      主机内存**（``reserved`` 86.67 GiB ≫ 卡 22.49，只慢 28 倍）⇒ 解码后量 ``max_memory_reserved()``，
      超过物理显存就按 ``shrink_for_peak()`` **一次跳到该降的档**并记进后续调用 ✓，回执里是
      ``"spilled": true, "peakReservedGiB": 86.67, "nextTemporalChunk": 1, "applied": true`` ✓；
      自动降的那次下一轮会写 ``reason="auto"`` ✓（不是你配的 ⇒ 别当成 ``configured`` ✗）；
      你**显式**配的档位我们只回报不擅改 ✓（此时 ``applied: false`` ✓）。

    ⚠️ **前两个一开就换字节** ✗✗（``temporalChunk`` **不在**这一列 ✓：它与整段逐位相等 ✓）：
    2048×1536 fp32 实测 ``tile`` 相对差 **5.8e-03**（halo 加到 24
    也消不掉 ✓：cuDNN 按窗口尺寸选算法 ✓）、``attnChunk`` 只有与 SDPA 内部分块对齐时才逐位 ✓
    （小画布逐位 ✓、2048×1536 差 8.55e-03 ✓）⇒ **想逐像素复现就别开** ✓，或开完把回执里的
    ``vaeTiling`` 一起记下来 ✓✗（不然下一次复现不出同一张图时无从判断原因 ✓）。
    省多少：8× 那份 2048×1536 从 **20.96 GiB** ⇒ ``attnChunk=1024`` **3.94** ⇒ 再叠
    ``tile=512`` **1.03 GiB** ✓（3072×2304 整张解会 OOM ✓）。

    ⚠️ 它**进装载键** ✓：改旋钮 ⇒ 重装一次（``reused=false`` ✓）。理由是一份后端只存一套设置 ✗
    —— 不进键就是"给了新旋钮却悄悄还是旧值" ✓✗。全关（或不写）等于没配 ✓、白装一遍是不会发生的 ✗。
    """
    body = await read_json(request)
    try:
        assembly = _assembly_from(body)
        # ⭐ 与 ``/generate`` 同一条口 ✓：只报档位就能装 ✓（缺的件从清单解析 ✓，
        #    解析不到 ⇒ runtime 当场点名缺哪格 ✓ —— 不在这里猜路径 ✗）
        assembly, auto_note = _auto_assembly(_stage_of(body, assembly), assembly)
    except pipe.StageError as err:
        return bad_request(str(err))
    try:
        report = await asyncio.to_thread(engine_runtime.ensure_loaded, **assembly)
    except runtime.EngineBusy as err:
        return _engine_error(facade.translate(err))
    except TorchBackendUnavailable as err:
        return bad_request(f"{err}（reason={getattr(err, 'reason', 'pending')} ✓）")
    return success({"loadReport": report, "runtime": engine_runtime.status(),
                    "autoNote": auto_note or None})


@router.post("/unload")
async def unload_runtime(request: Request) -> Any:
    """**丢掉张量 + 清缓存** ✓ —— 与外部服务那套（发 HTTP 让人家卸载 ✓）**不是一回事** ✗✗。

    ``force=true`` 可在有任务在跑时强卸 ✓（⚠️ 那个任务会跟着失败 ✗ —— 所以默认**拒** ✓）。
    """
    body = await read_json(request)
    try:
        return success(await asyncio.to_thread(engine_runtime.unload, force=_flag(body, "force")))
    except runtime.EngineBusy as err:
        return _engine_error(facade.translate(err))
    except pipe.StageError as err:
        # ⚠️ ``force: "false"`` 被 ``bool()`` 读成 **True** ⇒ **强卸** ⇒ 在跑的任务跟着死 ✓✗
        #    （用户的本意正好相反 ✓✗）⇒ 不认的值一律 400 ✓，不许猜 ✓。
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
    stage_key = "h3"
    auto_note = ""
    try:
        generation = _generation_request(body)
        assembly = _assembly_from(body)
        # ⭐ 只报「哪一档」也能跑 ✓：显式路径优先 ✓，缺的从清单解析 ✓（解析不到就不塞 ✓）
        stage_key = _stage_of(body, assembly)
        assembly, auto_note = _auto_assembly(stage_key, assembly)
        # ⚠️ ``dryRun: "false"`` 被 ``bool()`` 读成 **True** ⇒ 出一段**假画面** ✓✗，
        #    而用户以为自己关着 ✓✗（产物还标 synthetic ✓ 但没人看标签 ✓）⇒ 严格解析 ✓。
        assembly["dryRun"] = _flag(body, "dryRun")
        wait = _flag(body, "wait")
        # ⭐ **H3 缺取整口径 ⇒ 提交时就拒** ✗✗（2026-11-02 真跑抓的 ✓）：
        # H3 形态**没有单流前向** ✓（`denoise` 只喂得起一路 ✗）⇒ `audioLatentMode` 不给
        # ⇒ 装载照过（**20 秒白装** ✓✗）、采样第 0 步才炸 ✓，报错还落在「主干少参数」上 ✓✗
        # （读起来像模型坏了 ✗）。stage-only 那条路（`assembly_for_stage("h3")`）拿的是
        # **空设置** ⇒ 永远不带这格 ⇒ 画布点 H3 必撞 ✓✗。⇒ 在**提交前**点名缺什么 ✓。
        if (stage_key == pipe.VIDEO_STAGE and not assembly.get("dryRun")
                and not str(assembly.get("audio_latent_mode") or "").strip()):
            return bad_request(
                "H3 双流**必须给音频潜帧的取整口径** ✗：请求里加 ``\"audioLatentMode\": "
                "\"round\"``（或 ceil / floor ✓）——「秒数 → 音频潜帧数」的取整方式**没有默认值** ✗✗"
                "（`geometry.audio_latent_frames` 的纪律 ✓：没核过的口径不许猜 ✓）。"
                "不给这格的后果是：装载照过（约 20 秒 ✓）、**采样第 0 步才失败** ✓✗，"
                "报错还读起来像「主干坏了」✗ —— 所以在这里提前拦 ✓")
        # ⭐ 同一条纪律的**第二格** ✗✗（2026-11-02 同一次真跑抓的 ✓）：双流的 `init_dual_latents`
        #    要把像素帧数换算成**视频潜帧数** ✓，`temporalCompression` 不给 ⇒ init 阶段才炸 ✓✗
        #    （H3 的视频 VAE 是 **4 倍时间压缩** ✓ —— `pipeline` / `h3_edit` 全线都是 4 ✓）。
        if (stage_key == pipe.VIDEO_STAGE and not assembly.get("dryRun")
                and str(assembly.get("audio_latent_mode") or "").strip()
                and not body.get("temporalCompression") and not body.get("temporal_compression")):
            return bad_request(
                "H3 双流**必须给视频的时间压缩比** ✗：请求里加 ``\"temporalCompression\": 4``"
                "（H3 的视频 VAE 是 4 倍时间压缩 ✓ ✓✗）—— 本仓不猜压缩比 ✗✗"
                "（`geometry.latent_frames` 的纪律 ✓）。不给 ⇒ 装载照过、init 阶段才失败 ✓✗")
        if (stage_key == pipe.VIDEO_STAGE and not assembly.get("dryRun")
                and assembly.get("attach_audio_vae") is False):
            return bad_request(
                "H3 双流**必须挂音频 VAE** ✗（``attachAudioVae: false`` 是自断双流 ✓✗）："
                "H3 形态没有单流前向 ✓ ⇒ 关掉音频 VAE ⇒ 装载照过、采样必炸 ✓✗ —— "
                "要只出画面请走 `wan22` / `seedvr2` 那两档 ✓，别关这一格 ✗")
    except pipe.StageError as err:
        return bad_request(str(err))
    try:
        task_id = engine_runtime.submit(generation, assembly=assembly,
                                        stage=str(assembly.get("stage") or "h3"))
    except runtime.EngineBusy as err:
        #: ⚠️ 队满**必须当场说**（E3.1 背压）✗：不接住就是 500 ✓✗ —— 前端看到「服务器错误」，
        #: 既不知道「没入队」也不知道「该等一会儿再试」✓。这里翻成 400 + `E_ENGINE_BUSY` ✓。
        return _engine_error(facade.translate(err))
    poll = {"taskId": task_id, "pollUrl": f"/api/v1/engine/tasks/{task_id}",
            "stage": stage_key, "autoNote": auto_note or None}
    if not wait:
        return success({**poll, "status": "queued", "waited": False, "synthetic": True})
    try:
        timeout = float(body.get("timeoutSeconds") or body.get("timeout_seconds") or 0) or None
    except (TypeError, ValueError):
        return bad_request(f"timeoutSeconds 不是数字（收到 {body.get('timeoutSeconds')!r} ✗）")
    task = engine_runtime.wait(task_id, timeout=timeout)
    return success({**poll, "waited": True, "timeoutSeconds": timeout, "task": _with_urls(task)})


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
    return success(_with_urls(task))


@router.post("/tasks/{task_id}/cancel")
def cancel_engine_task(task_id: str) -> Any:
    """请求取消 ✓ ⇒ 任务快照 + ``cancel.verdict`` ✓。

    ⚠️ 裁决**由 ``runtime.cancel`` 在同一瞬间给出** ✓，这里**原样透传** ✗✗ 不许自己拿
    ``status`` 再判一次 —— 那读的是另一个瞬间的事实 ✓✗（worker 可能刚好收尾 ✓）：
    ``accepted``（排队中 ⇒ 已当场终结 ✓，保证停了）/ ``too_late``（已经在跑 ⇒ 只置了旗标，
    **不保证**停 ✗）/ ``completed``（早已终结 ⇒ 取消无从生效 ✓）。
    """
    task = engine_runtime.cancel(task_id)
    if task is None:
        return bad_request(f"没有这个任务 {task_id!r} ✓（列表见 /api/v1/engine/tasks ✓）")
    return success(task)


# ⚠️ 这里**没有** "同步端点读 body" 的辅助函数 ✗ —— 读不了 ✓✗（`read_json` 是 async ✗）
#   ⇒ 需要读体的端点一律 ``async def`` + ``await read_json`` + ``asyncio.to_thread`` ✓。
