"""业务链路 ⇒ **进程内自研引擎** 的桥 ✓（``provider=engine`` 那条路 ✓ —— 2026-09-26 ✓）。

## 它解决什么真问题

``services/engine/runtime.py`` 已经能「装 → 排队 → 跑 → 卸」✓，但它只认
:class:`~app.services.engine.pipeline.GenerationRequest` ✓（**引擎自己的语言** ✓）。
业务链路（``image_generation`` / ``video_generation``）手里是**数据库记录 + ``config.settings``** ✓
—— 两边的话不一样 ✓。中间缺的这一层就是本模块：把业务那侧翻成引擎那侧 ✓，
再把引擎的**事件 / 产物**搬回业务这侧 ✓。

## 三条纪律

1. **不依赖外部** ✓：全程**进程内**（``engine_runtime`` ✓）—— **不 HTTP 调 ComfyUI / SD WebUI ✗**，
   **不起外部推理服务 ✗**。只调本机 ``ffmpeg`` CLI 落盘 ✓（那是**工具**不是**服务** ✓，
   本仓色校正 / 合成早就在用 ✓）。
2. **缺什么报什么** ✗：``settings.ditPath`` 没给 ⇒ 先按 ``configs/models.json`` 清单自动找 ✓
   （``dit_fl2va_int8`` ⇒ ``dit_ref2va_int8`` ✓）；清单里也不在盘上 ⇒ **当场明确报错** ✗
   （不猜一个"差不多"的路径 ✓✗，见 :func:`resolve_dit_path` ✓）。
3. **如实自述** ✓：参考 TE / VAE **未训练** ✗ ⇒ ``synthetic`` 恒 True ✓。由此推出一条**必须说清**的事：
   引擎跑出来的是**视频** ✓ ⇒ 图片链路拿的是它的**首帧** ✓（不是"引擎直接画了张图"✗，
   见 :func:`extract_still` ✓）。

## ``settings`` 里认哪些键（camelCase 是契约 ✓，snake_case 也收 ✓）

===================  ==========================================================
``ditPath``          **必需**（不给则按清单自动找 ✓）
``tokenizerPath``    可选 —— 不给则挂**桩**分词器 ✓（分词无语义 ✓，运行时会如实标注 ✓）
``device``           可选 —— ``cuda`` / ``cpu``；不给则由引擎自己探 ✓
``audioLatentMode``  可选 —— ``round`` / ``ceil`` / ``floor``；**不给就不开双流** ✗（见下）
``temporalCompression`` 可选 —— 双流/潜帧要对齐时**必须显式给** ✓（本仓不猜 ✗）
``megapixels`` / ``steps`` / ``sampler`` / ``schedule`` / ``seed``  可选 —— 覆盖默认 ✓
===================  ==========================================================

⚠️ 另外把服务的 ``model`` 写成 ``h3`` ✓：``gpu_manager`` 的显存台账按 ``provider:model`` 查表 ✓，
``engine:h3`` 那条**登记了 19.5 GB + ``engine-unload`` 策略** ✓（写别的名字会落到 ``default`` 档 ⇒
抢显存时不会被安排卸载 ✗）。``provider`` = ``engine``（大小写随意 ✓）+ ``model`` = ``h3`` = 正解 ✓。

⚠️ **不猜的那两条**（与 ``geometry`` / ``pipeline`` 同一纪律 ✓）：``audioLatentMode`` 与
``temporalCompression`` 都**没有默认值** ✓ —— 不给 ⇒ 引擎照常出片，但**没有音轨** ✓；
这不是失败，是**如实的功能边界** ✓（日志里明说 ✓，见 :func:`run_job` 的 ``notes`` ✓）。
"""
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...core.config import get_storage_root
from ..file_storage import get_absolute_path, parse_data_url
from ..task_logger import log_task_error, log_task_progress, log_task_warn

__all__ = [
    "AUDIO_LATENT_MODES",
    "PROVIDER",
    "artifact_paths",
    "assembly_from_config",
    "extract_still",
    "extract_still_to_storage",
    "is_engine_provider",
    "materialize_reference",
    "materialize_references",
    "normalize_settings",
    "outputs_dir_for",
    "request_from_image_record",
    "request_from_video_record",
    "run_job",
    "static_relative",
]

#: ``provider`` 的取值 ✓ —— 大小写不敏感 ✓（配置里写 ``Engine`` 也认 ✓）
PROVIDER = "engine"

#: 音频潜帧的取整口径 ✓ —— **必须显式给** ✗（见 ``geometry.audio_latent_frames`` ✓）
AUDIO_LATENT_MODES = ("round", "ceil", "floor")

#: 轮询间隔（秒）✓ —— 事件就在**内存里** ✓ ⇒ 这一步只是把进度搬给 ``task_logger`` ✓
_POLL_SECONDS = 1.0

#: 图片链路要的**最短**时长 ✓：H3 帧数是 ``17k+5`` ✓ ⇒ ``0.2s @24fps`` 正好吸附到 **5 帧** ✓（最小 ✓）
_STILL_SECONDS = 0.2

#: 双流关掉时要说的那句实话 ✓（**不是**失败 ✗ —— 是功能边界 ✓）
_DUAL_STREAM_NOTE = (
    "⚠️ 本次**没有音轨** ✗：``audioLatentMode`` / ``temporalCompression`` 未给 ⇒ 双流未启用 ✓"
    "（本仓不猜取整口径与时间压缩比 ✗ —— 要原生音频就在服务 settings 里显式配上 ✓）"
)

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def is_engine_provider(provider: Any) -> bool:
    """``provider`` 是不是自研引擎 ✓（``None`` / ``""`` / 别的 ⇒ False ✓）。"""
    return str(provider or "").strip().lower() == PROVIDER


def normalize_settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """``config["settings"]`` ⇒ **snake_case 字典** ✓（``ditPath`` ⇒ ``dit_path`` ✓）。

    ``settings`` 可能是 dict ✓、也可能是 JSON 字符串 ✓（跨语言配置里两种都出现过 ✓）；
    读不动就当**空**（⇒ 后面按"没配"处理 ✓ —— 缺 DiT 会**明确报错** ✗ 而不是静默用个默认 ✓）。
    """
    raw: Any = (config or {}).get("settings")
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = {}
    if not isinstance(raw, Mapping):
        return {}
    return {_CAMEL_BOUNDARY.sub("_", str(key)).lower(): value for key, value in raw.items()}


def resolve_dit_path(settings: Mapping[str, Any] | None) -> tuple[str, str]:
    """定 DiT 主权重路径 ✓ ⇒ ``(路径, 来源说明)`` ✓ —— **查不到就报** ✗，绝不猜 ✓。

    来源**两级**（顺序固定 ✓）：
    1. ``settings.ditPath``（调用方**明确**指的 ✓ —— 永远优先 ✓）；
    2. ``configs/models.json`` 清单 ⇒ ``models_dir``（``dit_fl2va_int8`` ⇒ ``dit_ref2va_int8`` ✓，
       且要求**文件真在盘上** ✓）。

    ⚠️ 第 2 级**只认清单**✗，不做"扫目录猜文件名" ✓ —— 猜中的代价是加载到**别的**权重 ✓✗
    （那会「装得进去、跑得出来、结果不对」✓✗）。
    """
    values = settings or {}
    given = str(values.get("dit_path") or "").strip()
    if given:
        return given, "settings.ditPath"

    from . import inventory  # 局部 import ✓：桥不该把清单层的可选依赖拖进来 ✓

    catalog = inventory.load_catalog()
    by_key = {item.get("key"): item for item in catalog.get("models") or []}
    for key in ("dit_fl2va_int8", "dit_ref2va_int8"):
        entry = by_key.get(key)
        if entry is None:
            continue
        path = inventory.component_path(entry)
        if path is not None and Path(path).exists():
            return str(path), f"清单 configs/models.json ⇒ {key}"

    raise ValueError(
        "自研引擎缺 DiT 主权重 ✗：``settings.ditPath`` 没给 ✗，清单里也找不到在盘上的 "
        "``dit_fl2va_int8`` / ``dit_ref2va_int8`` ✗。\n"
        f"  · 清单：{inventory.catalog_path()}（模型根 = {inventory.models_dir() or '未配置 ✗'}）\n"
        "  · 怎么修：在图像/视频服务的 settings 里配 ``ditPath`` ✓，"
        "或按清单把权重放进 ``models_dir`` ✓。\n"
        "⚠️ 这里**不猜路径** ✗：猜错会「装得进去、跑得出来、结果不对」✓✗。"
    )


def assembly_from_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """业务 ``config`` ⇒ 运行时装配参数 ✓（``engine_runtime.submit(assembly=…)`` ✓）。

    只产出运行时装得下的键 ✓ —— 多给一个键会被 ``_ensure_loaded_internal`` 当成
    ``TypeError`` 炸掉 ✓✗，所以这里**逐项白名单** ✓ 而不是 ``**settings`` 整包铺开 ✓。
    ⚠️ 因此"DiT 是从哪来的"（:func:`resolve_dit_path` 的第 2 个返回值 ✓）**不进**装配 ✓
    —— 那是**日志事实** ✓ 不是装载参数 ✗，调用方自己拿它记一条 ✓。
    """
    settings = normalize_settings(config)
    dit_path, _source = resolve_dit_path(settings)
    assembly: dict[str, Any] = {"dit_path": dit_path}

    tokenizer = str(settings.get("tokenizer_path") or "").strip()
    if tokenizer:
        assembly["tokenizer_path"] = tokenizer

    device = str(settings.get("device") or "").strip()
    if device:
        assembly["device"] = device

    mode = str(settings.get("audio_latent_mode") or "").strip().lower()
    if mode:
        if mode not in AUDIO_LATENT_MODES:
            raise ValueError(
                f"``audioLatentMode`` 只能是 {AUDIO_LATENT_MODES} ✓，收到 {mode!r} ✗"
                "（取整口径**不许猜** ✗ —— 猜错是「看着对的错音频」 ✓✗）"
            )
        assembly["audio_latent_mode"] = mode

    if settings.get("attach_video_vae") is False:
        assembly["attach_video_vae"] = False
    if settings.get("attach_audio_vae") is False:
        assembly["attach_audio_vae"] = False
    return assembly


def outputs_dir_for(kind: str, record_id: Any) -> Path:
    """引擎产物落点 ✓：``<storageRoot>/engine/<kind>-<id>/`` ✓（**在数据目录里** ✓ 好被 /static 服务 ✓）。"""
    return Path(get_storage_root()) / "engine" / f"{kind}-{record_id}"


def static_relative(path: str | Path) -> str:
    """绝对路径 ⇒ **相对数据根**的 ``static/...`` ✓（与 ``download_file`` 的返回值同一口径 ✓）。

    ⚠️ 不在数据目录里 ⇒ **原样返回** ✓（调用方仍能用绝对路径打开 ✓ —— 假装它是个 ``static/`` URL
    才是真坑 ✓✗）。
    """
    target = Path(path)
    storage = Path(get_storage_root())
    try:
        return f"static/{target.resolve().relative_to(storage.resolve()).as_posix()}"
    except (ValueError, OSError):
        return str(target)


def materialize_reference(value: Any, *, sub_dir: str = "engine/refs") -> str | None:
    """参考素材（data URL / ``static/...`` / 绝对路径）⇒ **本机文件路径** ✓。

    引擎读的是**文件** ✓（不是 URL ✗ —— 它不联网 ✓）：data URL 落成临时图 ✓、
    ``static/...`` 用 ``get_absolute_path`` 解 ✓；解不出来 ⇒ ``None`` ✓（调用方记一条 warn ✓，
    而不是把一段 base64 当路径丢给引擎 ✓✗）。
    """
    text = str(value or "").strip()
    if not text:
        return None
    parsed = parse_data_url(text)
    if parsed is not None:
        directory = Path(get_storage_root()) / sub_dir
        directory.mkdir(parents=True, exist_ok=True)
        extension = mimetypes.guess_extension(parsed["mimeType"]) or ".bin"
        target = directory / f"{uuid.uuid4().hex}{extension}"
        target.write_bytes(base64.b64decode(parsed["data"]))
        return str(target)
    if text.startswith(("static/", "/static/")):
        relative = text[1:] if text.startswith("/static/") else text
        try:
            return get_absolute_path(relative)
        except ValueError:
            return None
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        return None  # 远程 URL ⇒ 引擎**不联网** ✗ ⇒ 由调用方先落地 ✓
    return text


def materialize_references(values: Any, *, sub_dir: str = "engine/refs") -> tuple[str, ...]:
    """一串参考素材 ⇒ 本机路径元组 ✓（解不出来的**逐条记 warn** ✓ 不静默丢 ✗）。"""
    if isinstance(values, str):
        try:
            decoded = json.loads(values)
        except (ValueError, TypeError):
            decoded = [values]
        values = decoded if isinstance(decoded, list) else [values]
    if not isinstance(values, Sequence):
        return ()
    out: list[str] = []
    for index, item in enumerate(values):
        resolved = materialize_reference(item, sub_dir=sub_dir)
        if resolved is None:
            log_task_warn("EngineBridge", "reference-unusable", {
                "index": index, "value": str(item)[:120],
                "reason": "data URL 解不出 / 是远程地址（引擎不联网）",
            })
            continue
        out.append(resolved)
    return tuple(out)


def _parse_size(size: Any) -> tuple[int, int] | None:
    """``"1920x1080"`` ⇒ ``(1920, 1080)`` ✓（解析不了 ⇒ ``None`` ✓ 交给 megapixels 兜底 ✓）。"""
    match = re.match(r"^\s*(\d+)\s*[x×X*]\s*(\d+)\s*$", str(size or ""))
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return width, height


def _visual_kwargs(settings: Mapping[str, Any]) -> dict[str, Any]:
    """分辨率 / 步数 / 采样器这类**可覆盖**的视觉参数 ✓（给了才传 ✓ —— 不给就用引擎默认 ✓）。"""
    kwargs: dict[str, Any] = {}
    for key, caster in (("steps", int), ("seed", int), ("megapixels", float)):
        value = settings.get(key)
        if value is not None and str(value) != "":
            kwargs[key] = caster(value)
    for key in ("sampler", "schedule"):
        value = str(settings.get(key) or "").strip()
        if value:
            kwargs[key] = value
    compression = settings.get("temporal_compression")
    if compression is not None and str(compression) != "":
        kwargs["temporal_compression"] = int(compression)
    return kwargs


def _materialize_extras(record: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    """条件素材：首帧 ✓ / 参考图 ✓ / 参考音频 ✓ / 参考视频 ✓（**只搬得动本机文件** ✓）。"""
    extras: dict[str, Any] = {}
    first_frame = materialize_reference(record.get("firstFrame"))
    if first_frame is None:
        for key in ("firstFramePath", "firstFrameUrl", "imagePath", "imageUrl"):
            first_frame = materialize_reference(record.get(key))
            if first_frame is not None:
                break
    if first_frame is not None:
        extras["first_frame"] = first_frame

    frames = materialize_references(record.get("referenceImages"))
    if frames:
        extras["reference_frames"] = frames
    audio = materialize_references(record.get("referenceAudios"), sub_dir="engine/refs-audio")
    if audio:
        extras["reference_audio"] = audio
    videos = materialize_references(record.get("referenceVideos"), sub_dir="engine/refs-video")
    if videos:
        extras["reference_videos"] = videos
    return extras


def request_from_image_record(record: Mapping[str, Any], *,
                              outputs_dir: str | Path,
                              settings: Mapping[str, Any] | None = None) -> Any:
    """图片生成记录 ⇒ 引擎请求 ✓。

    ⚠️ **必须说清的一件事**：H3 是**视频**模型 ✓ —— 引擎这里跑的是**最短那段视频** ✓
    （5 帧 ✓），调用方再取**首帧**当图片 ✓（见 :func:`extract_still` ✓）。
    所以"图片链路走引擎" ≠ "引擎会画单张图" ✓✗。
    """
    from . import geometry, pipeline  # 局部 import ✓：桥不把引擎层拉进业务模块的 import 期 ✓

    values = settings or {}
    ratio = _parse_size(record.get("size"))
    megapixels = values.get("megapixels")
    kwargs: dict[str, Any] = {
        "prompt": str(record.get("prompt") or ""),
        "negative": str(record.get("negativePrompt") or ""),
        "seconds": float(values.get("seconds") or _STILL_SECONDS),
        "outputs_dir": str(outputs_dir),
    }
    if ratio is not None:
        kwargs["ratio"] = ratio
        if megapixels is None:
            kwargs["megapixels"] = geometry.megapixels_for_size(*ratio)
    kwargs.update(_visual_kwargs(values))
    kwargs.update(_materialize_extras(record, values))
    return pipeline.GenerationRequest(**kwargs)


def request_from_video_record(record: Mapping[str, Any], *,
                              outputs_dir: str | Path,
                              settings: Mapping[str, Any] | None = None) -> Any:
    """视频生成记录 ⇒ 引擎请求 ✓（时长 / 画幅 / 首尾帧 / 参考素材 ✓）。"""
    from . import pipeline  # 局部 import ✓

    values = settings or {}
    kwargs: dict[str, Any] = {
        "prompt": str(record.get("prompt") or ""),
        "negative": str(record.get("negativePrompt") or ""),
        "outputs_dir": str(outputs_dir),
    }
    duration = record.get("duration")
    if duration is not None and str(duration) != "":
        kwargs["seconds"] = float(duration)
    ratio = str(record.get("aspectRatio") or "").strip()
    if ratio:
        kwargs["ratio"] = ratio
    kwargs.update(_visual_kwargs(values))
    kwargs.update(_materialize_extras(record, values))
    return pipeline.GenerationRequest(**kwargs)


def artifact_paths(task: Mapping[str, Any] | None) -> dict[str, str | None]:
    """引擎任务 ⇒ 产物路径 ✓（``videoPath`` / ``audioPath`` / ``primaryPath`` ✓）。"""
    outputs = ((task or {}).get("result") or {}).get("outputs") or {}
    return {
        "videoPath": outputs.get("videoPath"),
        "audioPath": outputs.get("audioPath"),
        "primaryPath": outputs.get("primaryPath"),
    }


async def run_job(request: Any, assembly: Mapping[str, Any], *,
                  task_type: str, record_id: Any, label: str = "") -> dict[str, Any]:
    """提交给**进程内**引擎 ✓ ⇒ 等它跑完 ✓ ⇒ 返回任务事实 ✓；失败**抛错** ✗（交给业务侧的收口 ✓）。

    进度搬运 ✓：引擎每一步都会留一条事件 ✓ ⇒ 这里把**新增的**那几条转给 ``task_logger`` ✓
    （``eventsTruncated`` 也算进去 ✓ —— 事件被截断时下标会错位 ✓✗，那会让进度看着"卡住" ✓）。

    ⚠️ 失败抛的是 :class:`RuntimeError` ✓（消息里带**引擎任务号** ✓）—— 业务侧已有的
    fallback / 失败收口逻辑照常生效 ✓，不需要它认识引擎 ✓。
    """
    from .runtime import engine_runtime  # 局部 import ✓：`provider=engine` 才需要引擎 ✓

    task_id = engine_runtime.submit(request, assembly=dict(assembly))
    prefix = f"{label + ' ' if label else ''}"
    log_task_progress(task_type, "engine-submit",
                      {"id": record_id, "engineTaskId": task_id, "assembly": dict(assembly)})

    emitted = 0
    while True:
        task = await asyncio.to_thread(engine_runtime.wait, task_id, timeout=_POLL_SECONDS)
        if task is None:
            raise RuntimeError(f"引擎任务 {task_id} 在队列里消失了 ✗（自述里查不到它 ✓）")
        events = list(task.get("events") or [])
        # ⚠️ 事件是**尾部截断**给出来的 ✓ ⇒ 第 i 条的真实序号要加上被截掉的条数 ✓
        #    （不加就会把老事件当成新的重发一遍 ✓✗，进度看着像在打转 ✓）
        truncated = int(task.get("eventsTruncated") or 0)
        start = truncated
        for offset, event in enumerate(events):
            if start + offset < emitted:
                continue
            log_task_progress(task_type, f"engine-{event.get('stage') or event.get('kind')}", {
                "id": record_id, "engineTaskId": task_id,
                "label": event.get("label"), "step": event.get("step"),
                "total": event.get("total"), "elapsedMs": event.get("elapsedMs"),
                "note": event.get("note"), "kind": event.get("kind"),
            })
        emitted = start + len(events)
        if task.get("terminal"):
            break

    status = task.get("status")
    if status != "done":
        log_task_error(task_type, "engine-failed", {
            "id": record_id, "engineTaskId": task_id, "status": status,
            "error": task.get("error"),
        })
        raise RuntimeError(
            f"{prefix}自研引擎任务 {task_id} 未成功（status={status}）："
            f"{task.get('error') or '看引擎任务详情 ✓'}"
        )

    result = task.get("result") or {}
    if result.get("synthetic"):
        log_task_warn(task_type, "engine-synthetic", {
            "id": record_id, "engineTaskId": task_id,
            "note": "参考 TE / VAE **未训练** ✗ ⇒ 产物是真文件 + 噪声画面 ✓（如实标注 ✓）",
        })
    if not (result.get("outputs") or {}).get("audioPath"):
        log_task_warn(task_type, "engine-no-audio", {
            "id": record_id, "engineTaskId": task_id, "note": _DUAL_STREAM_NOTE,
        })
    log_task_progress(task_type, "engine-done", {
        "id": record_id, "engineTaskId": task_id,
        "synthetic": result.get("synthetic"),
        "stageMs": result.get("stageMs"), "sampleSteps": result.get("sampleSteps"),
        **artifact_paths(task),
    })
    return dict(task)


async def extract_still(video_path: str | Path, target: str | Path, *, index: int = 0) -> dict[str, Any]:
    """视频 ⇒ **首帧 PNG** ✓（图片链路用 ✓ —— 引擎出的是视频 ✓ 不是单图 ✓✗）。

    全在本机做 ✓：``media.load_video_tensor``（ffmpeg 解码 ✓）⇒ ``media.write_image``（真 PNG ✓）。
    **不联网、不起服务** ✓。
    """
    from . import media as media_mod  # 局部 import ✓

    def _run() -> dict[str, Any]:
        frames, facts = media_mod.load_video_tensor(video_path)
        written = media_mod.write_image(frames, target, index=index)
        written["sourceFrames"] = int(facts.get("frames") or 0)
        return written

    return await asyncio.to_thread(_run)


async def extract_still_to_storage(video_path: str | Path, *, record_id: Any,
                                   sub_dir: str = "images") -> dict[str, Any]:
    """视频 ⇒ **落进数据目录**的首帧 PNG ✓ ⇒ 返回带 ``localPath``（``static/...`` ✓）的事实 ✓。

    放在这里（而不是业务模块里 ✓）的理由：落点口径只该有**一个**来源 ✓ —— 与
    ``download_file`` / ``save_base64_image`` 同一条「相对数据根」的约定 ✓✗。
    """
    directory = Path(get_storage_root()) / sub_dir
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"engine-{record_id}-{uuid.uuid4().hex[:8]}.png"
    written = await extract_still(video_path, target)
    return {**written, "localPath": static_relative(target), "absolutePath": str(target)}
