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
``tokenizerPath``    **视频**可选 —— 不给则挂**桩**分词器 ✓（分词无语义 ✓，运行时会如实标注 ✓）；
                     **出图必需** ✗ —— 不给则按上游约定探测 ✓，探测不到**当场报** ✗（见下）
``device``           可选 —— ``cuda`` / ``cpu``；不给则由引擎自己探 ✓
``audioLatentMode``  可选 —— ``round`` / ``ceil`` / ``floor``；**不给就不开双流** ✗（见下）
``temporalCompression`` 可选 —— 双流/潜帧要对齐时**必须显式给** ✓（本仓不猜 ✗）
``megapixels`` / ``steps`` / ``sampler`` / ``schedule`` / ``seed``  可选 —— 覆盖默认 ✓
===================  ==========================================================

⚠️ **出图那条路（``stage="sdxl"``）另外说** ✓：它要的是 **SDXL 主权重 + 一张 CLIP 词表** ✓✗。
``sdxlPath`` 不给 ⇒ 按清单 ``sdxl_base`` 自动找 ✓（先在 ``models_dir`` ✓，再到**动态探测**出来的
根下找**同名**文件 ✓）；``tokenizerPath`` 不给 ⇒ 按**上游约定**在 ComfyUI 类安装的
``comfy/sd1_tokenizer/`` 里探测 ✓ —— **两处都没有就当场报** ✗（视频那边缺词表只是"挂桩" ✓，
出图缺词表是**真跑不了** ✗，见 :func:`resolve_tokenizer_path` ✓）。

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


#: 两条**阶段**名 ✓（= `pipeline.VIDEO_STAGE` / `pipeline.IMAGE_STAGE` ✓ = `runtime` 装配的 ``stage=`` ✓）。
#: ⚠️ 这里用**字面量** ✗ 而不是 ``from .pipeline import …`` —— 桥刻意**只做懒加载** ✓
#: （见各函数里的局部 import ✓），模块级引入会把 pipeline 那串依赖一起拖进来 ✓✗。
#: ⚠️ 代价是**可能悄悄分叉** ✗ ⇒ 自检里有一条**钉住三者相等**的判据 ✓
#: （`bridge.IMAGE_STAGE == pipeline.IMAGE_STAGE == SdxlBackend.name` ✓），分叉必红 ✓。
VIDEO_STAGE = "h3"
IMAGE_STAGE = "sdxl"


def resolve_dit_path(settings: Mapping[str, Any] | None) -> tuple[str, str]:
    """定 DiT 主权重路径 ✓ ⇒ ``(路径, 来源说明)`` ✓ —— **查不到就报** ✗，绝不猜 ✓。

    三级来源（顺序固定 ✓，口径与 :func:`resolve_sdxl_path` **一模一样** ✓）：
    1. ``settings.ditPath``（调用方**明确**指的 ✓ —— 永远优先 ✓）；
    2. ``configs/models.json`` 清单 ⇒ ``models_dir/<kind>/<文件名>`` ✓（``dit_fl2va_int8`` ⇒
       ``dit_ref2va_int8`` ✓，且要求**文件真在盘上** ✓）；
    3. **同一份**清单条目在 :func:`app.services.engine.inventory.candidate_roots` 那些**动态探测**
       出来的根里 ✓（``<根>/diffusion_models/…`` 这类 ComfyUI 惯例布局 ✓ —— 本机的 DiT 就在这种
       目录里 ✓✗，以前只认 ``models_dir`` ⇒ **扫描能看见、视频看不见** ✓✗，2026-09-26 真机实测 ✓）。

    ⚠️ 第 2/3 级**只认清单**✗，不做"扫目录猜文件名" ✓ —— 猜中的代价是加载到**别的**权重 ✓✗
    （那会「装得进去、跑得出来、结果不对」✓✗）。
    """
    values = settings or {}
    given = str(values.get("dit_path") or "").strip()
    if given:
        return given, "settings.ditPath"

    from . import inventory  # 局部 import ✓：桥不该把清单层的可选依赖拖进来 ✓

    catalog = inventory.load_catalog()
    by_key = {item.get("key"): item for item in catalog.get("models") or []}
    searched: list[str] = []
    for key in ("dit_fl2va_int8", "dit_ref2va_int8"):
        entry = by_key.get(key)
        if entry is None:
            continue
        path = inventory.component_path(entry)
        if path is not None and Path(path).exists():
            return str(path), f"清单 configs/models.json ⇒ {key}"
        searched.append(key)
        # ③ 换根 ✓：同一份清单条目，去动态探测出来的根里找**同名**文件 ✓（文件名仍然只来自清单 ✓）
        for found, source in inventory.component_candidates(entry):
            if path is not None and found == path:
                continue
            if found.exists():
                return str(found), f"清单 {key}（文件名取自清单 ✓）⇒ {source}"

    raise ValueError(
        "自研引擎缺 DiT 主权重 ✗：``settings.ditPath`` 没给 ✗，清单里也找不到在盘上的 "
        f"``{'`` / ``'.join(searched) or 'dit_fl2va_int8 / dit_ref2va_int8'}``"
        "（``models_dir`` 与所有**动态探测**到的根都找过了 ✓）。\n"
        f"  · 清单：{inventory.catalog_path()}（模型根 = {inventory.models_dir() or '未配置 ✗'}）\n"
        "  · 怎么修：在图像/视频服务的 settings 里配 ``ditPath`` ✓；或按清单把权重放进 "
        "``models_dir`` ✓；或放进任一被探测到的 ComfyUI 的 ``models/diffusion_models/`` ✓"
        "（探测根表见 ``local_model_scan.get_default_roots`` ✓）。\n"
        "⚠️ 这里**不猜路径** ✗：猜错会「装得进去、跑得出来、结果不对」✓✗。"
    )


def resolve_sdxl_path(settings: Mapping[str, Any] | None) -> tuple[str, str]:
    """定 **SDXL 主权重**路径 ✓ ⇒ ``(路径, 来源说明)`` ✓（口径与 :func:`resolve_dit_path` **一模一样** ✓）。

    三级来源（顺序固定 ✓）：1. ``settings.sdxlPath`` ✓；2. 清单 ``sdxl_base`` 在 ``models_dir`` 里 ✓；
    3. **同一份**清单条目在 :func:`app.services.engine.inventory.candidate_roots` 那些
    **动态探测**出来的根里 ✓（``<根>/checkpoints/…`` 这类 ComfyUI 惯例布局 ✓ —— 那台机器上的
    SDXL 就在这种目录里 ✓✗，以前只认 ``models_dir`` ⇒ **扫描能看见、出图看不见** ✓✗）。

    ⚠️ 出图与视频是**两份不同的权重** ✗✗：拿 H3 的 DiT 去装 SDXL ⇒ 四个前缀一个都对不上 ✓
    ⇒ `load_sdxl_components` 当场报「这不是 SDXL 主权重」✓（反过来同理 ✓）。
    ⚠️ 第 3 级**换根不换名** ✗：文件名仍然只来自清单 ✓（见 ``inventory.component_candidates`` ✓）。
    """
    values = settings or {}
    given = str(values.get("sdxl_path") or "").strip()
    if given:
        return given, "settings.sdxlPath"

    from . import inventory  # 局部 import ✓：桥不该把清单层的可选依赖拖进来 ✓

    entry = inventory.catalog_entry("sdxl_base")
    if entry is not None:
        # ② 清单声明的落点（``models_dir`` ✓）—— 口径与消息**一字不改** ✓（既有自检钉着它 ✓）
        path = inventory.component_path(entry)
        if path is not None and Path(path).exists():
            return str(path), "清单 configs/models.json ⇒ sdxl_base"
        # ③ 换根 ✓：同一份清单条目，去动态探测出来的根里找**同名**文件 ✓
        for found, source in inventory.component_candidates(entry):
            if path is not None and found == path:
                continue
            if found.exists():
                return str(found), f"清单 sdxl_base（文件名取自清单 ✓）⇒ {source}"

    raise ValueError(
        "自研引擎出图缺 **SDXL 主权重** ✗：``settings.sdxlPath`` 没给 ✗，清单里也找不到在盘上的 "
        "``sdxl_base`` ✗（``models_dir`` 与所有**动态探测**到的根都找过了 ✓）。\n"
        f"  · 清单：{inventory.catalog_path()}（模型根 = {inventory.models_dir() or '未配置 ✗'}）\n"
        "  · 怎么修：在出图服务的 settings 里配 ``sdxlPath`` ✓；或按清单把 "
        "``sd_xl_base_1.0.safetensors`` 放进 ``models_dir`` ✓；或放进任一被探测到的 ComfyUI 的 "
        "``models/checkpoints/`` ✓（探测根表见 ``local_model_scan.get_default_roots`` ✓）。\n"
        "⚠️ 这里**不猜路径** ✗：猜错会「装得进去、跑得出来、结果不对」✓✗。"
    )


#: CLIP 词表目录里可以长什么样 ✓（口径来自 :mod:`app.services.engine.tokenizer_bpe` 的**形态嗅探** ✓）：
#: 单文件 ``tokenizer.json`` ✓，或 ``vocab.json`` + ``merges.txt`` 这一对 ✓。
_TOKENIZER_FILES = ("tokenizer.json",)
_TOKENIZER_BPE_FILES = ("vocab.json", "merges.txt")

#: ComfyUI 装完之后**它自己**那份 CLIP 词表的落点 ✓ —— **相对**于 ComfyUI 安装根 ✓
#: （⚠️ **不是**本机盘符 ✗）：上游把 SD1/SDXL 共用的那份 CLIP-BPE 词表放在 ``comfy/sd1_tokenizer/`` ✓
#: （SDXL 的 CLIP-L 与 OpenCLIP-bigG **共用**这一份 ✓ —— 两个塔的 BPE 表是同一张 ✓）。
_COMFYUI_TOKENIZER_SUBDIR = ("comfy", "sd1_tokenizer")


def _looks_like_tokenizer(directory: Path) -> bool:
    """这个目录是不是一份**能直接喂给** ``load_tokenizer`` 的词表 ✓（判据 = 文件**真在** ✓ ✗ 不是"目录存在"）。"""
    if any((directory / name).is_file() for name in _TOKENIZER_FILES):
        return True
    return all((directory / name).is_file() for name in _TOKENIZER_BPE_FILES)


def resolve_tokenizer_path(settings: Mapping[str, Any] | None) -> tuple[str, str]:
    """定 **CLIP 词表目录** ✓ ⇒ ``(目录, 来源说明)`` ✓ —— 定不下来就**报** ✗。

    两级来源（顺序固定 ✓）：1. ``settings.tokenizerPath`` ✓；
    2. ComfyUI 类安装根（``local_model_scan.get_comfyui_roots`` ✓）下的 ``comfy/sd1_tokenizer/`` ✓。

    ⚠️ 词表**不是**权重 ✗：它是**一张 BPE 表** ✓ ⇒ 靠"有没有 ``.safetensors``"是找不着它的 ✓✗；
    但也**不瞎找** ✗ —— 只在**上游约定**的那个落点上找 ✓（``comfy/sd1_tokenizer`` 是 ComfyUI
    自己的目录约定 ✓，跟本机盘符无关 ✓ ⇒ 与"扫描路径不写死"的口径**不冲突** ✓）。
    """
    values = settings or {}
    given = str(values.get("tokenizer_path") or "").strip()
    if given:
        return given, "settings.tokenizerPath"

    from ..local_model_scan import get_comfyui_roots  # 局部 import ✓

    seen: list[str] = []
    for root, source in get_comfyui_roots():
        directory = Path(root).joinpath(*_COMFYUI_TOKENIZER_SUBDIR)
        if str(directory) in seen:
            continue
        seen.append(str(directory))
        if _looks_like_tokenizer(directory):
            return str(directory), f"{source} ⇒ {'/'.join(_COMFYUI_TOKENIZER_SUBDIR)}"

    raise ValueError(
        "自研引擎出图缺 **CLIP 词表** ✗：``settings.tokenizerPath`` 没给 ✗，ComfyUI 类安装里也找不到 "
        f"``{'/'.join(_COMFYUI_TOKENIZER_SUBDIR)}/`` ✗。\n"
        "  · 词表长这样：``tokenizer.json`` ✓ 或 ``vocab.json`` + ``merges.txt`` 一对 ✓"
        "（SDXL 双塔**共用**这一张 CLIP-BPE 表 ✓）。\n"
        "  · 怎么修：在出图服务的 settings 里配 ``tokenizerPath`` ✓；或设 ``COMFYUI_PATH`` 指向装了 "
        "ComfyUI 的那份 ✓。\n"
        "⚠️ 词表**不是权重** ✗：去 ``.safetensors`` 里找是找不着它的 ✓✗（本仓**不猜**它放哪 ✓）。"
    )


def assembly_from_config(config: Mapping[str, Any] | None, *,
                         stage: str = VIDEO_STAGE) -> dict[str, Any]:
    """业务 ``config`` ⇒ 运行时装配参数 ✓（``engine_runtime.submit(assembly=…)`` ✓）。

    只产出运行时装得下的键 ✓ —— 多给一个键会被 ``_ensure_loaded_internal`` 当成
    ``TypeError`` 炸掉 ✓✗，所以这里**逐项白名单** ✓ 而不是 ``**settings`` 整包铺开 ✓。
    ⚠️ 因此"DiT 是从哪来的"（:func:`resolve_dit_path` 的第 2 个返回值 ✓）**不进**装配 ✓
    —— 那是**日志事实** ✓ 不是装载参数 ✗，调用方自己拿它记一条 ✓。

    ⚠️ ``stage`` **必须由调用方明说** ✗（默认视频 ✓）：它决定装**哪一套**组件 ✓ ——
    ``"h3"`` ⇒ H3 的 DiT + 参考 TE/VAE ✓；``"sdxl"`` ⇒ SDXL 的 **UNet + VAE + 双文本塔** ✓
    （两者**根本不是一份权重** ✓✗）。⚠️ 猜错的后果是「装得进去、跑得出来、结果全错」✓✗
    ⇒ 这里**不按 config 里的字段自己推** ✗（图片服务知道自己是图片 ✓，就由它明说 ✓）。
    """
    settings = normalize_settings(config)
    wanted = str(stage or VIDEO_STAGE).strip().lower()
    if wanted == IMAGE_STAGE:
        image_path, _image_source = resolve_sdxl_path(settings)
        assembly: dict[str, Any] = {"stage": IMAGE_STAGE, "dit_path": image_path}
        # ⭐ 2026-09-26 补 ✓：出图**另外**还要一份 CLIP 词表 ✓✗ —— 视频那边缺词表是**挂桩** ✓
        #    （分词无语义、运行时如实标注 ✓），出图**缺词表就是跑不了** ✗ ⇒ 这里就定下来 ✓：
        #    settings 给了用给的 ✓，没给按上游约定探测 ✓，两处都没有 ⇒ **当场报** ✗
        #    （口径同 :func:`resolve_sdxl_path` ✓：早报早好，别装作能跑 ✓）。
        tokenizer_path, _tokenizer_source = resolve_tokenizer_path(settings)
        assembly["tokenizer_path"] = tokenizer_path
    else:
        dit_path, _source = resolve_dit_path(settings)
        assembly = {"stage": VIDEO_STAGE, "dit_path": dit_path}

    tokenizer = str(settings.get("tokenizer_path") or "").strip()
    if tokenizer and not assembly.get("tokenizer_path"):
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
                              settings: Mapping[str, Any] | None = None,
                              stage: str = VIDEO_STAGE) -> Any:
    """图片生成记录 ⇒ 引擎请求 ✓（``stage`` 决定走**哪条**路 ✓，见下 ✓）。

    ⚠️ 默认取 **``h3``** ✗ 而不是 ``sdxl`` ✓ —— 与 :func:`assembly_from_config` 的默认**逐字对齐** ✓：
    调用方两个都不传 ⇒ 请求与装配落到**同一条**路 ✓（两处默认各写各的 ⇒ 会出现"请求按图建 ✓、
    权重按视频装 ✓"这种**错配** ✓✗ —— 实测撞到过 ✓，现在有 :func:`run_job` 的闸门兜住 ✓）。

    ⚠️ **两条路产物不同** ✗✗，别混为一谈 ✓：

    * ``stage="sdxl"``：SDXL **真·文生图** ✓ ⇒ 产物是**单张 PNG** ✓。⚠️ 出图服务**显式**传它 ✓。
      ⚠️ σ 调度要**离散**的 ✓（``normal`` / ``simple`` ✓）：**settings 没明说 ⇒ 给 ``normal``** ✓
      （SDXL 的常规默认 ✓）；**明说了别的**（如视频那套 ``karras`` ✓）⇒ 交给
      :func:`pipeline.build_plan` **当场报错** ✗ —— 这里**不悄悄替换** ✓✗
      （静默换调度 = 出现"你要的和你拿到的不是一回事" ✓）。
    * ``stage="h3"``（默认 ✓）：H3 是**视频**模型 ✓ ⇒ 只能跑**最短那段视频** ✓（5 帧 ✓）再取**首帧** ✓
      （见 :func:`extract_still` ✓）⇒ 「图片链路走引擎」**≠**「引擎会画单张图」✓✗。
    """
    from . import geometry, pipeline  # 局部 import ✓：桥不把引擎层拉进业务模块的 import 期 ✓

    values = settings or {}
    ratio = _parse_size(record.get("size"))
    megapixels = values.get("megapixels")
    wanted = str(stage or VIDEO_STAGE).strip().lower()
    kwargs: dict[str, Any] = {
        "prompt": str(record.get("prompt") or ""),
        "negative": str(record.get("negativePrompt") or ""),
        "seconds": float(values.get("seconds") or _STILL_SECONDS),
        "outputs_dir": str(outputs_dir),
        "stage": wanted,
    }
    if ratio is not None:
        kwargs["ratio"] = ratio
        if megapixels is None:
            kwargs["megapixels"] = geometry.megapixels_for_size(*ratio)
    kwargs.update(_visual_kwargs(values))
    if wanted == pipeline.IMAGE_STAGE and "schedule" not in kwargs:
        # ⚠️ 只是**补默认** ✓ 不是覆盖 ✓：`_visual_kwargs` 是"给了才传" ✓ ⇒ 走到这里就是"没配" ✓。
        kwargs["schedule"] = "normal"
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

    # ⚠️ **两个 stage 必须是同一份** ✗✗（实测撞到过 ✓）：请求那边的 ``stage`` 决定 **σ 调度与帧数** ✓
    #    （图片 1 帧 / 视频 17k+5 帧 ✓、离散 σ 表 / karras ✓），装配那边的 ``stage`` 决定**装哪套权重** ✓。
    #    对不上 ⇒ H3 的 DiT 去解 SDXL 的潜变量那种事 ✓✗ —— 后果是「装得进去、跑得出来、
    #    结果全错」✓✗（**最坏情况甚至不报错** ✓）。⇒ 这儿当场拦下 ✓，报错里写明**怎么修** ✓。
    declared = str((assembly or {}).get("stage") or VIDEO_STAGE).strip().lower()
    wanted = str(getattr(request, "stage", None) or VIDEO_STAGE).strip().lower()
    if declared != wanted:
        raise ValueError(
            f"请求与装配的 **stage 对不上** ✗✗：请求 ``stage={wanted}`` / 装配 ``stage={declared}`` ✓。\n"
            "  · 这两个必须是**同一个**值 ✓：一个决定 σ 调度与帧数 ✓、一个决定装哪套权重 ✓ —— "
            "对不上就是「装得进去、跑得出来、结果全错」✓✗。\n"
            "  · 怎么修：``assembly_from_config(config, stage=…)`` 与 "
            "``request_from_image_record(…, stage=…)`` 传**同一个**值 ✓"
            f"（视频 ``{VIDEO_STAGE}`` / 出图 ``{IMAGE_STAGE}`` ✓）。"
        )

    # ⚠️ 2026-09-26 修 ✓：``submit`` 的 ``stage`` **必须跟着传** ✗✗ —— 不传就按它自己的默认 ``h3`` ✓
    #    ⇒ 出图任务在**任务事实**里被标成 ``h3`` ✓✗（前面那两个 stage 我明明已经对齐过了 ✓，
    #    唯独没把同一个值交给 ``submit`` ✓）⇒ 进度/摘要/排障全看着像视频任务 ✓✗。
    #    这里传的是**刚刚对齐过的那个** ``wanted`` ✓（不另起一份取值 ✓）。
    task_id = engine_runtime.submit(request, assembly=dict(assembly), stage=wanted)
    prefix = f"{label + ' ' if label else ''}"
    log_task_progress(task_type, "engine-submit",
                      {"id": record_id, "engineTaskId": task_id,
                       "stage": wanted, "assembly": dict(assembly)})

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


async def store_image_to_storage(source: str | Path, *, record_id: Any,
                                 sub_dir: str = "images") -> dict[str, Any]:
    """**真·图片文件** ⇒ **落进数据目录** ✓ ⇒ 返回带 ``localPath``（``static/...`` ✓）的事实 ✓。

    ⚠️ 与 :func:`extract_still_to_storage` 共用**同一条**「相对数据根」的约定 ✓✗（落点口径只该有
    一个来源 ✓）：SDXL 后端把 PNG 写在 ``outputs_dir`` ✓（``<storageRoot>/engine/image-<id>/`` ✓），
    那是**引擎产物区** ✓、不是图片资产区 ✗ ⇒ 这里**复制**一份进 ``images/`` ✓，与
    ``download_file`` / ``save_base64_image`` 的落点一致 ✓。
    ⚠️ 是**复制**不是移动 ✗：引擎产物留着可复查 ✓（排障时要看原始 PNG ✓）。
    """
    import shutil  # noqa: PLC0415

    origin = Path(source)
    if not origin.exists():
        raise RuntimeError(f"图片产物不在盘上 ✗：{origin}（引擎报的路径对不上 ✓？）")
    directory = Path(get_storage_root()) / sub_dir
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"engine-{record_id}-{uuid.uuid4().hex[:8]}{origin.suffix or '.png'}"
    await asyncio.to_thread(shutil.copyfile, origin, target)
    return {"sourcePath": str(origin), "absolutePath": str(target),
            "localPath": static_relative(target), "bytes": int(target.stat().st_size)}
