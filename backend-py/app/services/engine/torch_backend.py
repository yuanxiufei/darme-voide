"""**torch 后端** —— 真张量实现（2026-09-17；本机已装 CPU 版 torch 2.14 ✓）。

## 它现在是**真**的哪一半（务必分清 ✗）

| 维度 | 现状 |
|---|---|
**张量本身** | ✅ **真 torch 张量** ✓（`realTensors=True` ✓）—— 不再用 `dryrun` 那个 12 行 `list[float]` 假张量 ✓ |
**随机与复现** | ✅ 真 `torch.Generator().manual_seed(seed)` + `torch.randn` ✓ ⇒ 同种子**逐位可复现** ✓ |
**采样/引导/首帧数学** | ✅ 全部跑在真张量上 ✓（与 `dryrun` 共用同一份 `sampler`/`guidance`/`conditioning` ✓） |
**模型前向（DiT/TE/VAE）** | ❌ **未接** ✗ —— H3 权重（主 DiT **19.53 GiB** ✗）没下载 ✓，架构装载也还没写 ✓ ⇒ 前向是**占位实现** ✗ |
**产物** | ❌ **不是生成画面** ✗ ⇒ `synthetic` **保持 True** ✓（`write` 落的是张量清单 ✓，不是 mp4 ✗） |

⇒ 两个标志**同时**给出，不许混 ✗：`realTensors=True`（张量真 ✓）+ `synthetic=True`（画面不真 ✗）。
UI/接口据此可以显示「真张量 ✓ / 真模型 ✗」而不至于误导 ✓。

## 为什么本机是 CPU 版

本机核实**没有 NVIDIA 显卡** ✗（Iris Xe 集显 ✓、无 `nvidia-smi` ✓）⇒ 装 **CPU 轮子**（124 MB ✓）才有意义 ✓
（2.5 GB CUDA 轮子在这里白装 ✗）。代码**一行都不用改** ✓：到 A5000 那台机器换 CUDA 轮子即可 ✓
（`device` 自动探测 ✓）。

## 依赖闸门（保留 ✓）

模块**不 import torch** ✓（懒加载 ✓）⇒ 没装也能 :meth:`TorchBackend.describe` ✓。
不可用时 ``reason`` 分两类 ✗：``deps``（去装包 ✓）／``pending``（依赖齐了但**这部分实现待写** ✓）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import audio_vae as audio_vae_mod
from . import dit as dit_mod
from . import geometry as geometry_mod
from . import inventory as inv
from . import loader
from . import media as media_mod
from . import safetensors as st
from . import text_encoder as te_mod
from . import vae as vae_mod
from . import weights as weights_mod

__all__ = [
    "DEPENDENCIES",
    "PENDING_PARTS",
    "TorchBackend",
    "TorchBackendUnavailable",
    "dependency_status",
    "torch_available",
]


class TorchBackendUnavailable(RuntimeError):
    """torch 后端暂时不可用 ✓ —— ``reason`` 区分「缺依赖」与「实现待写」✓。"""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class _Dependency:
    module: str
    package: str
    purpose: str
    approximate_mb: int


#: 真后端所需依赖 ✓（``module`` 用于探测 ✓，``package`` 用于给出安装命令 ✓）
DEPENDENCIES: tuple[_Dependency, ...] = (
    _Dependency("torch", "torch", "张量与设备运行时 ✓", 124),
    _Dependency("numpy", "numpy", "数组互转 / 数值工具", 20),
    _Dependency("safetensors", "safetensors", "权重读取（本仓另有纯 Python 读取器 ✓ 体检不必装 ✓）", 1),
    _Dependency("PIL", "pillow", "首帧/参考图解码 ✓", 3),
)

#: ⚠️ **仍未实现**的部分 ✓（依赖齐了、DiT 也装好了，也还是这些 ✗）—— 分开报，别让人去查环境 ✗
#: ⚠️ 2026-09-20 **更正**：上一版清单**过期了** ✗ —— 它写着「文本编码器前向 ✗」「VAE 解码与出片落盘 ✗」，
#: 但这些**机制都已经实现** ✓（`encode_text` ✓ / `attach_text_encoder` ✓ / `decode` ✓ /
#: `attach_vae` ✓ / `write` ✓，且 `engine_io_test` / `engine_text_test` 已跑通
#: 「TE → DiT → VAE → **真 mp4**」✓）。⇒ **真正缺的只有「真权重 + 真配置」** ✓：
#: 机制齐了、权重没到 ✓ ⇒ `canGenerate=False` **依然正确** ✓（只是别把原因归到机制上 ✗）。
PENDING_PARTS: tuple[str, ...] = (
    "H3 真权重未下载（主 DiT 19.53 GiB ✗ ⇒ 用 `loader.plan_stage('h3')` 看还差多少 ✓）",
    "⚠️ **H3 形态已建到「行级主干」并已接进管线** ✓（`engine/h3_form.py` ✓）："
    "积木（RMSNorm / SwiGLU / 显式注意力 / 18 路 adaLN / 正弦时间嵌入 / RoPE ✓）+ 打包层"
    "（段表 / 坐标 / 跨度 ✓）+ **主干 `H3FormTrunk`** ✓（模块名与参考 `__init__` 逐字对齐 ✓、"
    "含 `video_patch_proj`/`audio_patch_proj`（fp32 ✓）/ `rope.inv_freq`（**缓冲区** ✓）/ `token_refiner` ✓）"
    "⇒ 行级前向出 **(视频行, 音频行)** ✓ 且**按参考取负** ✓ —— ✅ **双流已接进管线** ✓"
    "（`init_dual_latents` / `sample_dual` / `_decode_dual` / `write` ✓ + `pipeline` 按后端自述分流 ✓"
    "⇒ 一次 `run_sync` 出 **真 mp4 + 真 wav** ✓，见 `engine_dual_stream_test` ✓）；"
    "参考块四类全有入口 ✓（图 ✓ / 音 ✓ / 视频 ✓ / 带音轨视频 ✓ —— `H3_FORM_TODO` 已清零 ✓）",
    "⭐ **DiT 还有 7 处结构差异** ✗（2026-09-20 读全参考实现后从 5 条补到 7 条 ✓ —— "
    "逐条见 `engine/dit.py::H3_STRUCTURAL_GAPS` ✓：adaLN 18 路 ✓ / 2D 打包序列 ✓ / `q_norm`·`k_norm` ✓ / "
    "RoPE ✓ / 正弦时间嵌入 + 每 token 独立 timestep ✓ / RMSNorm + SwiGLU ✓ / fp32 头 + denoise mask ✓）"
    "⇒ ⚠️ **这不是「小改」**✗：要**单开一个 H3 形态 forward** ✓（规格已固化：`H3_SHAPE_FACTS` ✓ + "
    "`H3_PACK_FACTS` ✓ + `schedules.time_shift_sigma` ✓）；✅ 已关 3 条见 `dit.H3_GAPS_CLOSED` ✓"
    "（⚠️「关掉 ≠ 核过真权重」✗：音频头与 refiner 内部仍是近似 ✓）",
    "H3 的 DiTConfig（**事实表已抄好** ✓ `dit.H3_SHAPE_FACTS`：hidden 5376 / depth 50 / heads 56 / "
    "headDim 128 / ffn 14336 / patch (1,2,2) / vae_scale 16 ✓ —— 拿到真权重后仍要按元数据复核 ✓）",
    "文本编码器 / VAE 的**权重**（机制已实现 ✓；tokenizer 是**注入式** ✓ —— "
    "本仓不内置词表 ✗，得自己给 ✓）",
    "端到端 `generate()` 拿真权重跑一次（现在 `canGenerate=False` ✓ = 「机制齐、权重没到」✓）",
)


def _spec(name: str) -> Any:
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ValueError):  # pragma: no cover - 极少数损坏的安装
        return None


def dependency_status() -> dict[str, Any]:
    """逐项依赖现状 ✓（**不导入**它们 ✓ ⇒ 毫秒级、无副作用 ✓）。"""
    items: list[dict[str, Any]] = []
    missing: list[str] = []
    for dependency in DEPENDENCIES:
        found = _spec(dependency.module) is not None
        items.append({
            "module": dependency.module, "package": dependency.package,
            "present": found, "purpose": dependency.purpose,
            "approximateMB": dependency.approximate_mb,
        })
        if not found:
            missing.append(dependency.package)
    return {
        "items": items,
        "missing": missing,
        "ready": not missing,
        "install": ([f"pip install {' '.join(missing)}"] if missing else []),
        "note": "有 NVIDIA 卡就用 CUDA 轮子（索引源按驱动选 ✓）；没有就得用 CPU 轮子 ✓"
                "（本项目**刻意不写死版本** ✗）",
    }


def torch_available() -> tuple[bool, str]:
    """``(可用?, 原因)`` ✓ —— 不可用时原因**可行动** ✓。"""
    status = dependency_status()
    if status["ready"]:
        return True, "依赖齐备 ✓"
    return False, (f"缺少依赖：{', '.join(status['missing'])}（共约 "
                   f"{sum(item['approximateMB'] for item in status['items'] if not item['present'])} MB ✓）"
                   f" ⇒ 先跑 {status['install'][0]} ✓")


class TorchBackend:
    """真张量后端 ✓（``realTensors=True`` ✓ / ``synthetic=True`` ✗ —— 见模块头那张表 ✓）。"""

    name = "torch"
    #: ⚠️ **产物不是真画面** ✗（前向是占位 ✓）—— 刻意保持 True ✓，不许因为"用上 torch 了"就改成 False ✗
    synthetic = True
    #: 张量是真的 ✓（与 :class:`app.services.engine.dryrun.DryRunBackend` 的本质区别 ✓）
    realTensors = True
    #: 条件/噪声的向量长度（占位前向用 ✓ —— 真模型接入后由架构决定 ✓）
    width = 512

    def __init__(self, device: str | None = None, *,
                 audio_latent_mode: str | None = None) -> None:
        available, reason = torch_available()
        self._available = available
        self._reason = reason
        self._device = device or self._detect_device()
        #: 装载成功后这里就是**真模型** ✓（None ⇒ 前向走占位 ✓，`describe()` 会如实说 ✓）
        self._model: Any = None
        self._config: Any = None
        self._loadReport: dict[str, Any] | None = None
        #: 参考 VAE ✓（挂了才解码出真帧 ✓）；未训练 ⇒ 画面是噪声 ✓（如实标 ✓）
        self._vae: Any = None
        self._vaeConfig: Any = None
        #: 参考**音频** VAE ✓（**与视频 VAE 是两族** ✗ —— 见 `vae.H3_AUDIO_VAE_FACTS` ✓）
        self._audioVae: Any = None
        self._audioVaeConfig: Any = None
        #: ⚠️ 「秒数 → 音频潜帧数」的**取整口径** ✓（``round`` / ``ceil`` / ``floor`` ✓）。
        #: 这一步**没核过** ✗（参考里 ``audio_t`` 是**从调用方张量读**的 ✓ ⇒ 怎么从秒数算，
        #: 没核到 ✓）⇒ 按 `geometry.audio_latent_frames` 的纪律**逼成必填** ✓：不给 ⇒
        #: 双流**不启用** ✓ —— 好默认值会把「未核实」伪装成「已实现」✗✓（本仓老账 ✓）。
        self._audioLatentMode = audio_latent_mode
        #: 参考文本编码器 ✓（挂了才出真条件 ✓）；同样未训练 ⇒ 无数值语义 ✓（如实标 ✓）
        self._textEncoder: Any = None
        self._textConfig: Any = None
        self._tokenizer: Any = None
        #: 首帧条件**实际走了哪条路** ✓（真 VAE 编码 / 占位 ✓ —— 由管线回给调用方 ✓）
        self._conditioningNote: dict[str, Any] | None = None

    @property
    def conditioningNote(self) -> dict[str, Any] | None:  # noqa: N802 —— 与前端 camelCase 对齐 ✓
        """首帧条件的实况自述 ✓（管线会把它并进 `conditioning` 结果里 ✓）。"""
        return self._conditioningNote

    #: H3 形态的**名称** ✓（`load_weights` 判出形态后写进 `_form` ✓；只有一处 ✓）
    H3_FORM_NAME = "h3-form"

    @property
    def dualStream(self) -> dict[str, Any]:  # noqa: N802 —— 与前端 camelCase 对齐 ✓
        """**双流（视频 + 音频）现状** ✓ —— 三条都满足才 ``enabled=True`` ✓，否则**逐条报缺什么** ✓。

        ⚠️ 刻意**不返回一个布尔** ✗：只说 ``False`` 的话，调用方还得自己猜"是没挂音频 VAE、
        还是模型形态不对、还是没给取整口径"✓✗（本仓判据：**缺什么就报什么** ✓，见
        `describe()` 的四个层次分答 ✓）。
        """
        blocked: list[str] = []
        if getattr(self, "_form", None) != self.H3_FORM_NAME:
            blocked.append("模型不是 H3 形态 ✗（`_form` 为空或不是 h3-form ✓ —— 装了 H3 权重才有 ✓）")
        if self._audioVae is None:
            blocked.append("未挂音频 VAE ✗（`attach_audio_vae()` ✓）")
        if not self._audioLatentMode:
            blocked.append("未给音频潜帧的取整口径 ✗（构造参数 `audio_latent_mode` ✓ —— "
                           "「秒数 → 音频潜帧数」这一步**没核过** ✗，不许给默认值 ✓）")
        elif self._audioLatentMode not in ("round", "ceil", "floor"):
            blocked.append(f"取整口径 {self._audioLatentMode!r} 不认 ✓（可用 round / ceil / floor ✓）")
        return {
            "enabled": not blocked, "audioVaeLoaded": self._audioVae is not None,
            "audioLatentMode": self._audioLatentMode, "blockedBy": blocked,
            # ⚠️ **能力声明** ✓（不是"看着像支持"✗）：本后端确实会在 `init_dual_latents` 里把首帧
            #    编成 `cond` 段 ✓（见 `_dual_keyframes` ✓）—— 管线据此决定"要不要拒绝"✓
            #    （对**不做这一步**的后端 ✓ 管线会**明确拒绝**首帧 ✗，而不是静默按文生视频跑 ✗✗）。
            "firstFrame": True,
            # ⚠️ 参考块**分类自述** ✓（别合成一个 bool ✗ —— 那样管线分不清"缺哪一类"✓✗）：
            #    * 图片 → ``ref_img`` 块 ✓（用**图片自己的**空间网格 ✓）；
            #    * 音频 → ``ref_audio`` 块 ✓（走**另一族** VAE ✓、不重采样 ✓）；
            #    * 视频 → ``video`` / ``video_audio`` 块 ✓（有无音轨决定 ✓ 两条流各走各的 VAE ✓）。
            "referencesImage": True,
            "referencesAudio": True,
            "referencesVideo": True,
        }

    def _detect_device(self) -> str:
        """有 CUDA 就用 CUDA ✓，否则 CPU ✓（**实测探测**，不猜 ✗）。"""
        if not self._available:
            return "cpu"
        torch = self._torch()
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _torch(self) -> Any:
        """懒导入 ✓（模块级不 import ✓ ⇒ 没装 torch 也能 import 本模块 ✓）。"""
        import torch  # noqa: PLC0415

        return torch

    def _gate(self) -> None:
        if not self._available:
            raise TorchBackendUnavailable(
                f"torch 后端不可用：{self._reason}"
                f"　⇒ 在此之前可用干跑后端验编排 ✓（`engine/dryrun.py` ✓）",
                reason="deps")

    @property
    def device(self) -> str:
        return self._device

    def load_weights(self, *, path: str | None = None,
                     config: Any = None) -> dict[str, Any]:
        """**装载真模型** ✓ ⇒ 之后 :meth:`denoise` 走**真前向** ✓（不再占位 ✓）。

        * ``path`` 省略 ⇒ 按 ``configs/models.json`` 里的主 DiT 解析 ✓（同一份清单 ✓）；
        * ``config`` 省略 ⇒ 先从权重 ``__metadata__`` 读 ✓（读到就用 ✓）；读不到 ⇒ **报错** ✗
          （**不猜结构** ✓；道理同 :mod:`dit` 的模块注释 ✓）；
        * 权重没下载 ⇒ ``reason="pending"`` ✓ + 给出还缺多少 ✓（`loader` 的口径 ✓）。
        """
        self._gate()
        target = Path(path) if path else self._default_dit_path()
        if target is None or not Path(target).exists():
            plan = self._plan_for_default_dit()
            raise TorchBackendUnavailable(
                f"主 DiT 权重未就绪（{target or '清单里没有可解析路径'} ✗）"
                f"　⇒ 先按加载计划把权重装上 ✓：{plan}",
                reason="pending")

        info = st.inspect(Path(target))
        # ⚠️ **先判形态** ✓（按**招牌键**判 ✓ 不按文件名猜 ✗）：H3 形态**不是** DiT ✗ ——
        #    少了这一步，H3 权重会被**静默**按 DiT 建 ✓✗（表面只表现为"装载报告里缺一堆键" ✓，
        #    人很容易读成"权重没下全" ✗）。判形态是纯函数、只看**头部键名** ✓ ⇒ 毫秒级 ✓ 不读大文件 ✓。
        from app.services.engine import h3_form  # noqa: PLC0415 —— 局部 import ✓：它模块级不碰 torch ✓ 无循环 ✓
        if h3_form.looks_like_h3_form(info.tensors.keys()):
            form_defaults = dict(h3_form.H3_TRUNK_DEFAULTS)
            if isinstance(config, dict):
                form_defaults.update(config)   # 自检用**缩小版**配置走**同一条**路 ✓
            # ⭐ **头库大小从权重形状推断** ✓（PDD ✓）—— 真权重**不会**给这个数 ✓ ⇒ 既不猜也不硬编 ✓：
            #    * 拿默认 1 去装 banks>1 的权重 ⇒ **形状不符** ✗；
            #    * 写死一个数 ⇒ 换检查点就错 ✓✗。
            entry = info.tensors.get(h3_form.H3_VIDEO_OUT_KEY)
            if entry is None:
                raise TorchBackendUnavailable(
                    f"H3 权重里没有 {h3_form.H3_VIDEO_OUT_KEY} ✗ ⇒ 推断不出头库大小 ✓"
                    f"（不静默按 1 装 ✗ —— 那会让 PDD 头库白装而**不报错** ✓✗）", reason="pending")
            inferred = h3_form.head_banks_from_shape(
                entry.shape,
                h3_form.video_patch_dim(int(form_defaults["latents_dim"]),
                                        tuple(form_defaults["patch_size"])))
            explicit = form_defaults.get("head_banks")
            if explicit is not None and int(explicit) != inferred:
                raise TorchBackendUnavailable(
                    f"配置里显式 `head_banks={explicit}` 与权重推断出的 {inferred} **不一致** ✗"
                    f"（两者必须一致 ✓ —— 不静默取其中一个 ✗）", reason="pending")
            form_defaults["head_banks"] = inferred
            form_report = weights_mod.load_module_weights(
                h3_form.H3FormTrunk(**form_defaults), target, device=self._device)
            if not form_report.complete:
                raise TorchBackendUnavailable(
                    f"H3 形态权重与结构对不上 ✗（complete=False ⇒ **不是**完整模型）"
                    f"　：{form_report.to_dict()}",
                    reason="pending")
            self._model = h3_form.H3FormTrunk(**form_defaults).to(self._device)
            weights_mod.load_module_weights(self._model, target, device=self._device)
            self._config = form_defaults
            self._form = self.H3_FORM_NAME
            self._loadReport = form_report.to_dict()
            return self._loadReport
        if config is None:
            config = dit_mod.DiTConfig.from_metadata(info.metadata)   # 读不到会**明确报错** ✓
        report = weights_mod.load_module_weights(
            dit_mod.build_dit(config), target, device=self._device)
        if not report.complete:
            raise TorchBackendUnavailable(
                f"权重与模型结构对不上 ✗（complete=False ⇒ **不是**完整模型）"
                f"　：{report.to_dict()}",
                reason="pending")
        self._model = dit_mod.build_dit(config).to(self._device)
        weights_mod.load_module_weights(self._model, target, device=self._device)
        self._config = config
        # ⚠️ 形态必须**每次装载都显式写** ✓：同一实例"先装 H3、再装 DiT"时，不写就会把上一次的
        #    `_form` 留着 ✓✗ ⇒ 双流自述会说"可用"而模型其实是 DiT ✗✗（静默 ✓）。
        self._form = "dit"
        self._loadReport = report.to_dict()
        return self._loadReport

    def _default_dit_path(self) -> str | None:
        """按清单找主 DiT ✓（``category=video`` 且 ``required`` 的第一个 ✓ —— 与体检同一份清单 ✓）。"""
        for entry in inv.load_catalog()["models"]:
            if entry.get("category") == "video" and entry.get("required") \
                    and str(entry.get("kind")) == "diffusion_models":
                resolved = inv.component_path(entry)
                return str(resolved) if resolved else None
        return None

    def _plan_for_default_dit(self) -> str:
        """把"还差什么"用 :mod:`loader` 的口径说清 ✓（而不是只说一句"缺权重" ✗）。"""
        try:
            stage = loader.plan_stage("h3")
            residency = stage.get("residency") or {}
            return (f"缺 {len(residency.get('missing') or [])} 项 ✓"
                    f"（合计约 {stage.get('requiredWeightsGiB', 0)} GiB ✓，"
                    f"峰值驻留 {residency.get('peakResidentGiB', 0)} GiB ✓）")
        except Exception:  # noqa: BLE001 - 诊断信息不该反过来把主流程搞崩 ✓
            return "（加载计划不可用 ✓）"

    def _latent_mode(self) -> str:
        """潜变量现在是什么形态 ✓ —— **四态**（别混成一个 bool ✗）：

        ``h3-dual`` ✓（H3 形态 + 音频 VAE + 取整口径齐 ⇒ 视频与音频两条 ✓）/
        ``h3-trunk-only`` ✓（H3 形态但双流不齐 ⇒ 只有主干，**不能走管线** ✗）/
        ``real-shape`` ✓（DiT + vae_scale ⇒ 真形状单流 ✓）/ ``placeholder-1d`` ✓（一维占位 ✓）。
        """
        if getattr(self, "_form", None) == self.H3_FORM_NAME:
            return "h3-dual" if self.dualStream["enabled"] else "h3-trunk-only"
        if self._model is not None and self._config is not None \
                and getattr(self._config, "vae_scale", 0):
            return "real-shape"
        return "placeholder-1d"

    def describe(self) -> dict[str, Any]:
        """自述 ✓（**只探依赖 + 报设备**，不做张量操作 ✓ ⇒ 任何时候都能调 ✓）。"""
        torch_version = ""
        if self._available:
            try:
                torch_version = str(self._torch().__version__)
            except Exception:  # noqa: BLE001 - pragma: no cover
                torch_version = ""
        return {
            "name": self.name, "synthetic": self.synthetic, "realTensors": self.realTensors,
            "available": self._available, "reason": self._reason,
            "device": self._device, "torchVersion": torch_version,
            "cudaAvailable": self._device == "cuda",
            # ⚠️ 四个层次**分开答** ✓（混在一起就会变成"看着像能做、其实不行" ✗）：
            #    依赖齐了 ✓ → 张量是真的 ✓ → DiT 真前向（装了权重才是 ✓）→ 能不能出片 ✗
            "modelLoaded": self._model is not None,
            "vaeLoaded": self._vae is not None,
            "textEncoderLoaded": self._textEncoder is not None,
            "tokenizer": (getattr(self._tokenizer, "name", None)
                          if self._tokenizer is not None else None),
            # 落盘能力其实来自 ffmpeg ✓ ⇒ 一并报出来（缺 ffmpeg 时前端能提前说清 ✓）
            "ffmpeg": {"available": media_mod.have_ffmpeg(), "version": media_mod.ffmpeg_version()},
            # 「装了模型 + 给了 vae_scale」⇒ 拿真形状的潜变量 ✓；否则一维占位 ✓（如实标 ✓）
            "latentMode": self._latent_mode(),
            # ⚠️ 音频 VAE 与**双流现状**分开报 ✓（挂了音频 VAE ≠ 双流可用 ✓ —— 还缺形态与取整口径 ✓）
            "audioVaeLoaded": self._audioVae is not None,
            "dualStream": self.dualStream,
            "loadReport": self._loadReport,
            # ⚠️ config **两种形态**：DiT 是 dataclass（有 `to_dict` ✓）、H3 形态是 **dict** ✓
            #    —— 2026-09-20 自检当场抓到：H3 装载后调 `describe()` 直接 `AttributeError` ✗
            #    （而 `/engine/backends` 端点正是调它 ✓ ⇒ 装了 H3 权重后端就 500 ✓✗）。
            "config": (dict(self._config) if isinstance(self._config, dict)
                       else (self._config.to_dict() if self._config is not None else None)),
            # ⚠️ 2026-09-20 更正：TE / VAE / 落盘的**机制都已实现** ✓（见 `PENDING_PARTS` 的注释 ✓）
            #    ⇒ 现在「不能出片」的**唯一**原因是**真权重没到** ✗（如实 ✓；别归因错到机制上 ✗）
            "canGenerate": False,
            "pendingParts": list(PENDING_PARTS),
            "dependencies": dependency_status(),
        }

    # ── GenerationBackend 协议 ──────────────────────────────────────────
    def condition_width(self) -> int:
        """条件向量宽度 ✓ —— **装了模型就按它的 ``text_dim``** ✓（否则类默认 ✓）。

        ⚠️ 这里踩过一次：初版一律用类默认 512 ✗，而测试配置的 ``text_dim=16`` ✓
        ⇒ 真前向报 `mat1 and mat2 shapes cannot be multiplied (1x512 and 16x32)` ✗
        （自检 ㉗ 整链跑到才暴露 ✓ —— 又一条"必须端到端跑"的证据 ✓）。
        """
        return self._configured_text_dim(self.width)

    def _configured_text_dim(self, fallback: int) -> int:
        """配置里的 ``text_dim`` ✓ —— **两种 config 形态都认** ✓（H3 形态是 **dict** ✓、DiT 是 dataclass ✓）。

        ⚠️ 踩过：H3 分支只写 `getattr(config, "text_dim", …)` ✗ ⇒ 在 dict 上取不到 ⇒ **悄悄回落**
        到默认值 ✓✗（TE 的 output_dim 与主干 `condition_proj` 对不上 ✓）。那次会**报错**✓（形状不符 ✓），
        但别指望每次都这么走运 ✓ —— 同一份配置两种形态，就得两处都认 ✓。
        """
        config = self._config
        if isinstance(config, dict):
            return int(config.get("text_dim") or fallback)
        if config is not None:
            return int(getattr(config, "text_dim", fallback) or fallback)
        return fallback

    def _condition(self, text: str, width: int | None = None) -> Any:
        """文本 → **真张量**条件 ✓（占位：TE 权重未下载 ✗ ⇒ 用 sha256 造确定性向量 ✓ 可复现 ✓）。"""
        torch = self._torch()
        count = int(width or self.condition_width())
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        values = [((digest[index % len(digest)] / 255.0) * 2.0 - 1.0)
                  for index in range(count)]
        return torch.tensor(values, dtype=torch.float32, device=self._device)

    def _encode_text_states(self, request: Any) -> dict[str, Any]:
        """**H3 形态**要的是**文本状态** ``[L, text_dim]`` ✓ —— 不是 ``{positive, negative}`` 条件对 ✗。

        ⚠️ 为什么必须挂了 TE 才能走：``L``（token 数）**只有分词器知道** ✓ —— 拿 sha256 占位
        造一条 ``[1, text_dim]`` 就是**编**一个不存在的序列长度 ✗ ⇒ 明确报错 ✓（宁可当场失败 ✓）。
        ⚠️ 返回值里**刻意没有 negative** ✗：H3 参考实现**不做 CFG** ✓（第 111 步核到的事实 ✓）——
        给了 negative 会让管线以为要引导 ✓✗，而双流路径**明确拒绝引导** ✓（见 `pipeline` ✓）。
        """
        if self._textEncoder is None:
            raise TorchBackendUnavailable(
                "H3 形态需要**文本状态** `[L, text_dim]` ✗，但没挂文本编码器 ✓ ⇒ 先 "
                "attach_text_encoder() ✓（token 数只有分词器知道 ✓，占位造不出来 ✓）",
                reason="pending")
        torch = self._torch()
        config = self._textConfig
        ids, original, truncated = te_mod.tokenize_prompt(
            self._tokenizer, request.prompt, max_length=config.max_length)
        tensor = torch.tensor([ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            states = self._textEncoder(tensor)
        shape = tuple(states.shape)
        if len(shape) != 3 or int(shape[0]) != 1:
            raise TorchBackendUnavailable(
                f"文本编码器输出应为 (1, L, text_dim) ✓，收到 {shape} ✗", reason="pending")
        return {"positive": states[0], "ids": list(ids), "tokens": len(ids),
                "originalTokens": original, "truncated": truncated,
                "textEncoder": "reference-untrained"}

    def encode_text(self, request: Any) -> dict[str, Any]:
        """提示词 → 条件 ✓：挂了文本编码器就走**真编码器** ✓，否则用 sha256 占位 ✓（都如实标 ✓）。"""
        self._gate()
        if getattr(self, "_form", None) == self.H3_FORM_NAME:
            return self._encode_text_states(request)
        if self._textEncoder is not None:
            torch = self._torch()
            config = self._textConfig
            results: dict[str, Any] = {}
            meta: dict[str, Any] = {}
            positive_ids: list[int] = []
            for name, text in (("positive", request.prompt), ("negative", request.negative)):
                ids, original, truncated = te_mod.tokenize_prompt(
                    self._tokenizer, text, max_length=config.max_length)
                tensor = torch.tensor([ids], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    results[name] = self._textEncoder(tensor)
                if name == "positive":
                    positive_ids = list(ids)
                meta[name] = {"tokens": len(ids), "originalTokens": original,
                              "truncated": truncated}
            # ⚠️ `ids` 必须是**真的 id 列表** ✓（第一版把 meta 字典塞进了 `ids` ✗ ⇒ 调用方
            #    `len(...)` 拿到的是"字典键数"而不是 token 数 ✓ —— 自检 ⑯ 当场红 ✓ 已改 ✓）；
            #    逐分支明细放 `byBranch` ✓。
            return {**results, "ids": positive_ids, "tokens": len(positive_ids),
                    "originalTokens": meta["positive"]["originalTokens"],
                    "truncated": meta["positive"]["truncated"],
                    "byBranch": meta, "textEncoder": "reference-untrained"}
        return {"positive": self._condition(request.prompt),
                "negative": self._condition(request.negative),
                "textEncoder": "sha256-placeholder"}

    def init_latents(self, plan: Any, request: Any) -> Any:
        """初始噪声 ✓ —— **真** `torch.randn` + **真**种子 ✓ ⇒ 同种子逐位可复现 ✓。

        * **装了真模型**（且配置给了 ``vae_scale`` ✓）⇒ 造**真形状**的潜变量 ✓
          ``(1, C, 潜帧, H/vae, W/vae)`` ✓ ⇒ 真前向才吃得下 ✓；
        * 否则 ⇒ 退回**一维占位** ✓（只为验编排 ✓，`describe()["latentMode"]` 会如实说 ✓）。
        """
        self._gate()
        torch = self._torch()
        shape = self.latent_shape(plan)
        generator = torch.Generator(device="cpu").manual_seed(int(request.seed))
        if shape is None:
            noise = torch.randn(self.width, generator=generator, dtype=torch.float32)
            return noise.to(self._device)
        noise = torch.randn(*shape, generator=generator, dtype=torch.float32)
        return noise.to(self._device)

    def latent_shape(self, plan: Any) -> tuple[int, ...] | None:
        """潜变量形状 ✓（``None`` = 信息不够 ⇒ **用占位** ✓ 并如实标注 ✓）。

        ⚠️ 需要 ``vae_scale``（像素↔潜空间边长比 ✓）与潜帧数 ✓；**任一缺就返回 None** ✗
        —— 猜一个"看起来合理"的 8 只会让真机上错得莫名其妙 ✗。
        """
        config = self._config
        if self._model is None or config is None or not getattr(config, "vae_scale", 0):
            return None
        latent_frames = getattr(plan, "latent_frames", None)
        if not latent_frames:
            return None
        scale = int(config.vae_scale)
        return (1, int(config.in_channels), int(latent_frames),
                max(1, int(plan.height) // scale), max(1, int(plan.width) // scale))

    # ── H3 双流（视频 + 音频 ✓ 2026-09-20 接进管线 ✓）──────────────────────
    def init_dual_latents(self, plan: Any, request: Any) -> dict[str, Any]:
        """H3 **双流**初始噪声 ✓ ⇒ ``{"video": [C,T,H,W], "audio": [C,ch,T_a]}`` ✓（**无 batch 维** ✓）。

        * 视频潜尺寸 = ``plan`` 像素尺寸 ÷ **vae_scale(16)** ✓（H3 事实 ✓）—— **不整除就报错** ✗
          （吸附网格是**调用方**的事 ✓，这里不悄悄取整 ✗）；
        * 音频潜帧数走 :func:`geometry.audio_latent_frames`（``mode`` 来自构造参数 ✓ —— **没核过**
          的那一步**必填** ✓，见 :attr:`dualStream` ✓）；
        * 两条流共用一个 **generator** ✓（同一 ``seed`` ⇒ 顺序固定 ⇒ **逐位可复现** ✓）。
        """
        self._gate()
        state = self.dualStream
        if not state["enabled"]:
            raise TorchBackendUnavailable("双流不可用 ✗：" + "；".join(state["blockedBy"]),
                                          reason="pending")
        config = self._config if isinstance(self._config, dict) else {}
        video_dim = int(config.get("latents_dim") or 0)
        audio_dim = int(config.get("audio_latents_dim") or 0)
        if not video_dim or not audio_dim:
            raise TorchBackendUnavailable(
                "H3 形态配置里缺 `latents_dim` / `audio_latents_dim` ✗ ⇒ 造不出潜变量形状 ✓",
                reason="pending")
        latent_frames = getattr(plan, "latent_frames", None)
        if not latent_frames:
            raise TorchBackendUnavailable(
                "双流的视频潜帧数未知 ✗ ⇒ 请求里要**显式给** `temporal_compression` ✓"
                "（本仓不猜压缩比 ✓ —— 见 `geometry.latent_frames` ✓）", reason="pending")
        scale = int(vae_mod.H3_VIDEO_VAE_FACTS["vaeScale"])
        height, width = int(plan.height), int(plan.width)
        if height % scale or width % scale:
            raise TorchBackendUnavailable(
                f"像素尺寸 {width}×{height} 不能被 H3 的 vae_scale={scale} 整除 ✗"
                f"（要吸附到网格 ✓ —— 这里**不悄悄取整** ✗）", reason="pending")
        torch = self._torch()
        audio_frames = geometry_mod.audio_latent_frames(
            int(plan.frames) / max(1, int(plan.fps)), mode=str(self._audioLatentMode))
        generator = torch.Generator(device="cpu").manual_seed(int(request.seed))
        video = torch.randn(1, video_dim, int(latent_frames), height // scale, width // scale,
                            generator=generator, dtype=torch.float32)[0]
        audio = torch.randn(1, audio_dim, geometry_mod.AUDIO_LATENT_CHANNELS, int(audio_frames),
                            generator=generator, dtype=torch.float32)[0]
        keyframes = self._dual_keyframes(plan, request, scale)
        return {"video": video.to(self._device), "audio": audio.to(self._device),
                "audioLatentFrames": int(audio_frames), "vaeScale": scale,
                "keyframes": keyframes,
                "references": self._dual_references(plan, request)}

    def _dual_keyframes(self, plan: Any, request: Any, scale: int) -> list[dict[str, Any]]:
        """首帧 → **关键帧条件块** ✓（H3 的 ``cond`` 段 ✓）—— 没给首帧 ⇒ 空列表 ✓（正常 ✓）。

        ⚠️ 事实（核自参考 ✓）：``cond`` 行用的是**目标的空间网格** ✓ ⇒ 首帧**必须**编码成目标潜尺寸 ✓
        （图片按 ``plan`` 的宽高缩放 ✓ 再编码 ✓）；``t`` 起点 = ``cursor + FRAME_RESCALE × 帧号`` ✓，
        首帧就是**第 0 帧** ✓。
        ⚠️ 编码用的是本仓**参考（未训练）视频 VAE** ✓ ⇒ 这条条件**数值上没有语义** ✓（验的是管道 ✓）；
        真权重到位后调用点不变 ✓。
        """
        image = getattr(request, "first_frame", None)
        if not image:
            return []
        if self._vae is None:
            raise TorchBackendUnavailable(
                "双流首帧条件需要**视频 VAE**（把首帧编码成潜变量 ✓）✗ ⇒ 先 attach_vae() ✓",
                reason="pending")
        torch = self._torch()
        frames, info = media_mod.load_image_tensor(
            image, width=int(plan.width), height=int(plan.height))
        if int(plan.width) % scale or int(plan.height) % scale:
            raise TorchBackendUnavailable(
                f"首帧尺寸 {plan.width}×{plan.height} 不能被 vae_scale={scale} 整除 ✗", reason="pending")
        with torch.no_grad():
            latent = self._vae.encode(frames.to(self._device))[0]
        return [{"resolved_frame_index": 0, "latent": latent, "audio_latent": None,
                 "imagePath": str(info.get("path") or image)}]

    def _dual_references(self, plan: Any, request: Any) -> list[dict[str, Any]]:
        """参考图 → ``ref_img`` 块 ✓（⚠️ 用**图片自己的**网格 ✓ —— 与 ``cond`` 用**目标**网格**不同** ✓ 事实 ✓）。

        参考块现有三类入口 ✓：图片（``reference_frames`` ✓）、音频（``reference_audio`` ✓）、
        视频（``reference_videos`` ✓ —— 有音轨则升为 ``video_audio`` ✓）。
        ⚠️ 与本项目的其它参考实现一样：VAE **未训练** ✗ ⇒ 参考值**没有语义** ✓（验的是管道 ✓）。
        """
        images = tuple(getattr(request, "reference_frames", ()) or ())
        clips = tuple(getattr(request, "reference_audio", ()) or ())
        videos = tuple(getattr(request, "reference_videos", ()) or ())
        if not images and not clips and not videos:
            return []
        torch = self._torch()
        blocks: list[dict[str, Any]] = []
        if images:
            if self._vae is None:
                raise TorchBackendUnavailable(
                    "参考图需要**视频 VAE**（把它编码成潜变量 ✓）✗ ⇒ 先 attach_vae() ✓",
                    reason="pending")
            for image in images:
                frames, info = media_mod.load_image_tensor(
                    image, width=int(plan.width), height=int(plan.height))
                with torch.no_grad():
                    latent = self._vae.encode(frames.to(self._device))[0]      # [C, 1, h, w] ✓
                blocks.append({"kind": "image", "latent": latent,
                               "latent_h": int(latent.shape[2]), "latent_w": int(latent.shape[3]),
                               "imagePath": str(info.get("path") or image)})
        for clip in clips:
            # ⚠️ 音频参考走**另一族** VAE ✗（视频 VAE 顶替不了 ✓✗）；且**不重采样、不混声道** ✓
            #    —— 采样率/声道不符就报错 ✓（见 `media.load_wav_tensor` ✓）。
            if self._audioVae is None:
                raise TorchBackendUnavailable(
                    "参考音频需要**音频 VAE** ✗ ⇒ 先 attach_audio_vae() ✓（视频 VAE 顶替不了 ✓）",
                    reason="pending")
            waveform, info = media_mod.load_wav_tensor(
                clip, sample_rate=int(self._audioVaeConfig.sample_rate),
                channels=int(self._audioVaeConfig.stereo_channels))
            with torch.no_grad():
                latent = self._audioVae.encode(
                    waveform.unsqueeze(0).to(self._device))[0]             # [C, 2, T] ✓
            blocks.append({"kind": "audio", "latent": latent,
                           "ref_audio_t": int(latent.shape[-1]),
                           "audioPath": str(info.get("path") or clip)})
        for clip in videos:
            # ⚠️ 参考**视频**块 ✓（``video`` / ``video_audio``）：视频流走视频 VAE ✓；
            #    文件里**有音轨** ⇒ 块类型升为 ``video_audio`` ✓ 且音轨走**另一族**音频 VAE ✓
            #    （不静默丢音轨 ✗ —— 丢了对用户不可见 ✓✗；要无声参考就给无声文件 ✓）。
            # ⚠️ 原生尺寸解码 ✓（不缩放、不抽帧 ✗）；宽高不能整除 vae_scale ⇒ **报错** ✗。
            if self._vae is None:
                raise TorchBackendUnavailable(
                    "参考视频需要**视频 VAE**（把帧编码成潜变量 ✓）✗ ⇒ 先 attach_vae() ✓",
                    reason="pending")
            frames, info = media_mod.load_video_tensor(clip)
            scale = int(self._vaeConfig.spatial_scale)
            if int(info["width"]) % scale or int(info["height"]) % scale:
                raise TorchBackendUnavailable(
                    f"参考视频尺寸 {info['width']}×{info['height']} 不能被 vae_scale={scale} 整除 ✗"
                    f"（吸附网格是调用方的事 ✓ —— 这里**不悄悄取整** ✗）", reason="pending")
            with torch.no_grad():
                latent = self._vae.encode(frames.to(self._device))[0]   # [C, t, h, w] ✓
            block: dict[str, Any] = {
                "kind": "video", "latent": latent,
                "latent_t": int(latent.shape[1]), "latent_h": int(latent.shape[2]),
                "latent_w": int(latent.shape[3]), "ref_audio_t": 0,
                "videoPath": str(info.get("path") or clip)}
            if info.get("hasAudio"):
                if self._audioVae is None:
                    raise TorchBackendUnavailable(
                        "参考视频**带音轨** ✗ ⇒ 需要**音频 VAE** 把它编进 ``video_audio`` 块 ✓"
                        "（要么先 attach_audio_vae() ✓，要么给无声的视频文件 ✓ —— "
                        "**不静默丢音轨** ✗）", reason="pending")
                with tempfile.TemporaryDirectory(prefix="ref_video_audio_") as folder:
                    extracted = media_mod.extract_wav(clip, Path(folder) / "ref_audio.wav")
                    waveform, _ainfo = media_mod.load_wav_tensor(
                        extracted["path"],
                        sample_rate=int(self._audioVaeConfig.sample_rate),
                        channels=int(self._audioVaeConfig.stereo_channels))
                with torch.no_grad():
                    audio_latent = self._audioVae.encode(
                        waveform.unsqueeze(0).to(self._device))[0]       # [C, 2, T] ✓
                block.update({"kind": "video_audio", "audio_latent": audio_latent,
                              "ref_audio_t": int(audio_latent.shape[-1])})
            blocks.append(block)
        return blocks

    def _dual_extra_rows(self, latents: Any, condition: Any) -> tuple[Any, Any]:
        """条件/参考块 → **行** ✓（⚠️ 行序**必须**与 `h3_form.packed_layout` 的段序一致 ✗✗
        —— 同数不同序的话总行数校验**拦不住** ✓✗，只会"画面不对" ✓）。

        段序（事实 ✓）：``[cond* / cond_audio*] → [ref_img* / ref_audio*]`` ✓ ⇒ 这里就照它拼 ✓：
        **先关键帧、后参考块** ✓。⚠️ 两条流（视频行 / 音频行）是**独立**的两条 ✓ ⇒ 各自内部顺序对就行 ✓；
        块**内部**顺序由 `patchify_video` / `pack_audio` 保证 ✓（与 layout 同一套外壳 ✓）。
        """
        from app.services.engine import h3_form  # noqa: PLC0415 —— 同 load_weights 的理由 ✓
        keyframes = list(latents.get("keyframes") or [])
        refs = list(latents.get("references") or [])
        if not keyframes and not refs:
            return None, None
        # ⚠️ 这里**自己算一遍布局** ✓（`forward` 里面还会再算一次 ✓）—— 两次都是**同一个纯函数** ✓
        #    ⇒ 只要参数一致就一致 ✓（这与"两处各写一套规则"是两回事 ✓）。
        # ⭐ 关键是：**行序不再由本方法决定** ✓ —— 只提供「块 → 潜变量」✓，顺序由布局的块清单说了算 ✓
        #    （2026-09-20 之前是"先关键帧、后参考块"的人工对齐 ✗ —— 同数不同序时谁都拦不住 ✗✗）。
        target = latents["video"]
        layout = h3_form.packed_layout(int(condition.shape[0]), int(target.shape[1]),
                                       int(target.shape[2]), int(target.shape[3]),
                                       int(latents["audio"].shape[-1]),
                                       keyframes=keyframes or None, refs=refs or None)
        video_sources: dict[Any, Any] = {}
        audio_sources: dict[Any, Any] = {}
        for index, keyframe in enumerate(keyframes):
            if keyframe.get("latent") is not None:
                video_sources[("keyframe", index)] = keyframe["latent"]
            if keyframe.get("audio_latent") is not None:
                audio_sources[("keyframe", index)] = keyframe["audio_latent"]
        for index, block in enumerate(refs):
            if block.get("latent") is None:
                continue
            # ⚠️ **两条流分开** ✓：图片 → 视频行 ✓、音频 → 音频行 ✓（同名的 `latent` 字段 ✓ 别混 ✗）
            if block.get("kind") == "image":
                video_sources[("ref", index)] = block["latent"]
            elif block.get("kind") == "audio":
                audio_sources[("ref", index)] = block["latent"]
            elif block.get("kind") in ("video", "video_audio"):
                # ⚠️ 参考视频块**两条流都可能有行** ✓（``video_audio`` ✓）：视频行 + 音频行 ✓
                #    （布局里音频行排在视频行**之前** ✓ —— 顺序由 `packed_layout` 说了算 ✓）。
                video_sources[("ref", index)] = block["latent"]
                if block.get("audio_latent") is not None:
                    audio_sources[("ref", index)] = block["audio_latent"]
        return h3_form.assemble_blocks(layout, video_sources, audio_sources,
                                       self._model.patch_size)

    def sample_dual(self, latents: Any, sigmas: Any, condition: Any, request: Any,
                    on_step: Any = None) -> dict[str, Any]:
        """双流采样 ✓ —— **判据只有一份** ✓：直接调 :func:`h3_form.sample_dual_stream` ✓
        （本方法**不重写**采样循环 ✗，只把"主干 + 两条流 + shift" 递过去 ✓）。

        ⚠️ ``shift`` 取自 `h3_form.H3_SIGMA_SHIFTS` ✓（**唯一一处** ✓ 事实 = 12.0 / 3.0 ✓）。
        """
        self._gate()
        if not isinstance(latents, dict) or "video" not in latents or "audio" not in latents:
            raise TorchBackendUnavailable(
                "双流采样需要 :meth:`init_dual_latents` 的产出（含 video/audio 两条 ✓）✗",
                reason="pending")
        from app.services.engine import h3_form  # noqa: PLC0415 —— 同 load_weights 的理由 ✓
        keyframes = list(latents.get("keyframes") or [])
        refs = list(latents.get("references") or [])
        # ⚠️ 行序责任**收敛到一处** ✓（`_dual_extra_rows` ✓ 照 `packed_layout` 的段序拼 ✓）——
        #    上一轮这里还是"按类型分组"的写法 ✓ 且**只敢收一块** ✗（多块时行序会错而**不报错** ✓✗）；
        #    现在逐块按段序拼 ⇒ **多关键帧也支持** ✓（不再是"明确拒绝" ✓）。
        extra_video, extra_audio = self._dual_extra_rows(latents, condition)
        shifts = h3_form.H3_SIGMA_SHIFTS
        return h3_form.sample_dual_stream(
            self._model, latents["video"], latents["audio"], condition, sigmas,
            shift_v=float(shifts["sigma_shift_video"]), shift_a=float(shifts["sigma_shift_audio"]),
            callback=on_step, extra_video_rows=extra_video, extra_audio_rows=extra_audio,
            keyframes=keyframes or None, refs=refs or None)

    def denoise(self, latents: Any, sigma: float, condition: Any, request: Any) -> Any:
        """**真前向**（装了模型 ✓）或**占位**（没装 ✓）—— 由 :meth:`describe` 如实标注 ✓。

        * 真前向：``x0 = DiT(x, σ, context)``（流匹配 velocity → x0 ✓ 见 :func:`dit.flow_match_x0` ✓）；
        * 占位：``(1−g)·cond + g·x``（``g→0`` 当 ``σ→0`` ✓ 保证采样按时收敛 ✓）。
        """
        self._gate()
        torch = self._torch()
        sigma = float(sigma)
        if self._model is not None:
            with torch.no_grad():
                prediction = self._model(latents, sigma, condition)
            return dit_mod.flow_match_x0(prediction, latents, sigma)
        with torch.no_grad():
            g = min(0.5, sigma / (sigma + 1.0)) if sigma > 0 else 0.0
            return condition * (1.0 - g) + latents * g

    def condition_first_frame(self, latents: Any, image_path: str, mask: list[float],
                              plan: Any, request: Any) -> Any:
        """首帧条件 ✓ —— **真张量按掩码混合** ✓（掩码由引擎算出 ✓ 见 `conditioning` ✓）。"""
        self._gate()
        torch = self._torch()
        # ⚠️ 两条路都要支持（初版只按 5 维写 ⇒ 占位一维时 `reshape` 直接崩 ✗，自检 ㊾ 抓到 ✓）：
        #   * **真形状潜变量** (B,C,T,h,w) ⇒ 按掩码**逐潜帧**混 ✓（引擎算的 mask ✓ 长度 = T ✓）；
        #   * **占位一维潜变量** ⇒ 按「首帧权重」整体混 ✓（占位下谈逐帧没意义 ✓，如实标 ✓）。
        # ⚠️ 仍是**占位数值**：真做法是用 VAE `encode` 把首帧图片编码成潜变量 ✓（`vae` 已就绪 ✓，
        #    缺"读图 + 缩放 + 组 batch"的胶水 ✓）—— 数学是真的 ✓、数值是占位的 ✓。
        if latents.ndim == 5:
            frames = int(latents.shape[2])
            weight0 = float(mask[0]) if mask else 0.0
            if self._vae is not None and weight0 > 0:
                # ── **真路径** ✓：图片 → VAE 编码 → 按掩码混进**第 0 个潜帧** ✓ ────────────
                scale = int(self._vaeConfig.spatial_scale)
                pixel_w, pixel_h = int(latents.shape[4]) * scale, int(latents.shape[3]) * scale
                image, info = media_mod.load_image_tensor(image_path, width=pixel_w,
                                                          height=pixel_h)
                torch = self._torch()
                with torch.no_grad():
                    encoded = self._vae.encode(
                        image.to(device=latents.device, dtype=latents.dtype))
                # ⚠️ **只比 C/h/w，不比 T** ✓ —— 图片天然是**单帧** ⇒ 编码出 ``T=1`` ✓
                #    而潜变量有 T 帧 ✓（初版连 T 一起比 ⇒ 自己把自己拦住了 ✗，自检㉟ 当场红 ✓）。
                if tuple(encoded.shape[1::2]) != tuple(latents.shape[1::2]):
                    raise TorchBackendUnavailable(
                        f"首帧编码后形状 {tuple(encoded.shape)} 与潜变量 {tuple(latents.shape)} "
                        f"的「通道/高/宽」对不上 ✗（VAE 配置要与 DiT 的 vae_scale、通道数一致 ✓）",
                        reason="pending")
                blended = latents.clone()
                blended[:, :, :1] = latents[:, :, :1] * (1.0 - weight0) + encoded * weight0
                self._conditioningNote = {
                    "mode": "vae-encode", "imageInfo": info, "weight": weight0,
                    "note": "图片经 **VAE 编码**后按掩码混入第 0 个潜帧 ✓"
                            + ("（⚠️ keep>1 时只有第 0 潜帧能来自图片 ✓ 其余按掩码保持生成 ✓）"
                               if len([w for w in mask if w > 0]) > 1 else ""),
                }
                return blended
            # ── 占位路径：没有 VAE（或掩码首权重为 0）⇒ 按元素数造确定性向量 ✓ 并如实标 ✓ ──
            count = int(latents.shape[1] * latents.shape[2] * latents.shape[3] * latents.shape[4])
            image = self._condition(f"first-frame:{image_path}", width=count).reshape(latents.shape)
            weights = torch.tensor(list(mask) + [0.0] * max(0, frames - len(mask)),
                                   dtype=latents.dtype, device=latents.device)[:frames]
            weights = weights.reshape(1, 1, -1, 1, 1)
            self._conditioningNote = {
                "mode": "placeholder", "weight": weight0,
                "note": "未挂 VAE ⇒ 首帧条件是**占位向量** ✗（挂 `attach_vae()` 后走真编码 ✓）",
            }
        else:
            image = self._condition(f"first-frame:{image_path}", width=int(latents.numel()))
            first = float(mask[0]) if mask else 0.0
            weights = torch.full_like(latents, first, dtype=latents.dtype)
            self._conditioningNote = {
                "mode": "placeholder-1d", "weight": first,
                "note": "潜变量是**一维占位**（未装模型）⇒ 首帧条件无从谈起 ✓（如实标 ✓）",
            }
        with torch.no_grad():
            return latents * (1.0 - weights) + image * weights

    def attach_text_encoder(self, config: Any = None, tokenizer: Any = None) -> dict[str, Any]:
        """挂上 :mod:`app.services.engine.text_encoder` ✓ ⇒ :meth:`encode_text` 出**真条件** ✓。

        ⚠️ 它同样**未经训练** ✗ ⇒ 条件数值没有语义 ✓（验的是管道 ✓）。真权重到位后换实现即可 ✓
        （调用点不变 ✓）。``tokenizer`` 走**注入** ✓（本仓不内置词表 ✗ 见那里的模块注释 ✓）。
        """
        self._gate()
        config = config or te_mod.TextEncoderConfig(
            output_dim=self._configured_text_dim(64))
        self._tokenizer = tokenizer or te_mod.StubTokenizer(
            config.vocab_size, max_length=config.max_length)
        self._textEncoder = te_mod.build_text_encoder(config).to(self._device).eval()
        self._textConfig = config
        return {
            "config": config.to_dict(),
            "tokenizer": getattr(self._tokenizer, "name", type(self._tokenizer).__name__),
            "note": "TE 与本仓库其它模型一样是**参考实现（未训练）** ✗ ⇒ 条件无数值语义 ✓",
        }

    def attach_vae(self, config: Any = None) -> dict[str, Any]:
        """挂上 :mod:`app.services.engine.vae` 的参考 VAE ✓ ⇒ :meth:`decode` 能出**真帧** ✓。

        ⚠️ 它是**未经训练**的 ✓ ⇒ 画面是噪声 ✓（验的是**管道** ✓）。真 VAE 权重到位后换 `vae.py`
        的实现即可 ✓（`decode` 的调用点不变 ✓）。
        """
        self._gate()
        config = config or vae_mod.VideoVAEConfig()
        self._vae = vae_mod.build_vae(config).to(self._device).eval()
        self._vaeConfig = config
        return config.to_dict()

    def attach_audio_vae(self, config: Any = None) -> dict[str, Any]:
        """挂上 :mod:`app.services.engine.audio_vae` 的参考音频 VAE ✓ ⇒ 双流的音频侧能解出波形 ✓。

        ⚠️ 它**与视频 VAE 毫无关系** ✗（DAC 血统编码 + BigVGAN 解码 ✓ —— 见
        `vae.H3_AUDIO_VAE_FACTS` ✓）：H3 的音频侧**不能**用视频 VAE 顶替 ✓✗（会得到"看着接上了"的
        假接线 ✗）。
        ⚠️ 参考实现**未经训练** ✗ ⇒ 出来是噪声 ✓（验的是**管道** ✓）；真权重到位后换实现即可 ✓
        （调用点不变 ✓）。
        """
        self._gate()
        config = config or audio_vae_mod.AudioVAEConfig()
        self._audioVae = audio_vae_mod.build_audio_vae(config).to(self._device).eval()
        self._audioVaeConfig = config
        return config.to_dict()

    def _decode_dual(self, latents: dict[str, Any]) -> dict[str, Any]:
        """**双流解码** ✓：视频走视频 VAE ✓、音频走**另一族**音频 VAE ✓ —— 缺哪个报哪个 ✓。

        ⚠️ 形状：H3 潜变量**没有 batch 维** ✓（``[C,T,H,W]`` / ``[C,ch,T]`` ✓），而本仓两个参考 VAE
        都吃 ``(B, …)`` ✓ ⇒ 这里**加一维、解完取第 0 条** ✓（不是"静默丢维度" ✗ —— 那一维就是
        我们自己加的 ✓）。
        """
        self._gate()
        if self._vae is None or self._audioVae is None:
            missing = "视频 VAE" if self._vae is None else "音频 VAE"
            raise TorchBackendUnavailable(
                f"双流解码缺 {missing} ✗ ⇒ `attach_vae()` / `attach_audio_vae()` ✓", reason="pending")
        torch = self._torch()
        with torch.no_grad():
            frames = self._vae.decode(latents["video"].unsqueeze(0))
            waveform = self._audioVae.decode(latents["audio"].unsqueeze(0))
        sample_rate = int(getattr(self._audioVaeConfig, "sample_rate", 0) or 32000)
        return {
            "synthetic": True,      # 真张量 ✓ + 两个真 VAE ✓ ⇒ 但**都没训练** ✗ ⇒ 仍是合成 ✓
            "note": "真 torch 张量 + 两个参考 VAE ✓（视频 / 音频 ✓），但都**未经训练** ✗ "
                    "⇒ 画面与声音都是噪声 ✓（验的是管道 ✓；真权重到位后换实现 ✓）",
            # ⚠️ 两条流的 batch 处理**刻意不同** ✗✓ —— 各自按**落盘工具的契约**来 ✓：
            #    `media.write_video` 要 **5 维** ``(B,3,T,H,W)`` ✓（与单流 decode 同口径 ✓）
            #    ⇒ 视频**保留**那一维 ✓；`media.write_wav` 要 ``(channels, N)`` ✓ ⇒ 音频**去掉** ✓。
            #    2026-09-20 自检抓到过：视频也去掉 ⇒ `write` 认不出帧张量 ⇒ **默默走成"张量清单 JSON"**
            #    那条路 ✗✗（错在 `json.dumps(Tensor)` 才炸 ✓ —— 属于"响亮但指不到真因" ✓）。
            "frames": frames, "frameCount": int(frames.shape[2]),
            "shape": list(frames.shape), "dtype": str(frames.dtype), "device": str(frames.device),
            "audioWaveform": waveform[0], "audioShape": list(waveform.shape),
            "sampleRate": sample_rate,
            "vaeConfig": self._vaeConfig.to_dict() if self._vaeConfig is not None else None,
            "audioVaeConfig": (self._audioVaeConfig.to_dict()
                               if self._audioVaeConfig is not None else None),
        }

    def decode(self, latents: Any, plan: Any, request: Any) -> dict[str, Any]:
        """**VAE 解码**（挂了 VAE ✓）或只回真张量事实（没挂 ✓）—— 都如实标注 ✓。

        ⚠️ **双流**（``init_dual_latents`` 的产出 ✓）走另一条 ✓：视频与音频**各用各的 VAE** ✗
        （见 :meth:`_decode_dual` ✓）。
        """
        self._gate()
        if isinstance(latents, dict) and "video" in latents and "audio" in latents:
            return self._decode_dual(latents)
        if self._vae is not None:
            torch = self._torch()
            with torch.no_grad():
                frames = self._vae.decode(latents)
            return {
                "synthetic": True,          # 真张量 ✓ 真 VAE ✓ 但**未训练** ✗ ⇒ 仍是合成 ✓
                "note": "真 torch 张量 + 参考 VAE ✓，但 VAE **未经训练** ✗ ⇒ 画面是噪声 ✓"
                        "（验的是管道 ✓；接真权重后这里换成训练好的 VAE ✓）",
                "frames": frames, "frameCount": int(frames.shape[2]),
                "shape": list(frames.shape), "dtype": str(frames.dtype),
                "device": str(frames.device), "vaeConfig": self._vaeConfig.to_dict()
                if self._vaeConfig is not None else None,
            }
        return {
            "synthetic": True,
            "note": "真 torch 张量 ✓ 但**未经 VAE** ✗（未挂 VAE ✓）⇒ 这不是生成画面 ✗",
            "frames": int(plan.frames), "frameWidth": int(plan.width), "frameHeight": int(plan.height),
            "shape": list(latents.shape), "dtype": str(latents.dtype), "device": str(latents.device),
            "norm": round(float(latents.norm()), 6),
            "preview": [round(float(value), 4) for value in latents[:8].tolist()],
        }

    def write(self, outputs: dict[str, Any], plan: Any, request: Any) -> dict[str, Any]:
        """有**真帧张量**就落**真 mp4** ✓（ffmpeg ✓）；否则落张量清单 JSON ✓ —— 两种情况都如实标 ✓。"""
        self._gate()
        where = Path(request.outputs_dir) if request.outputs_dir else Path(
            tempfile.mkdtemp(prefix="engine_torch_"))
        where.mkdir(parents=True, exist_ok=True)
        frames = outputs.get("frames")
        waveform = outputs.get("audioWaveform")
        if hasattr(frames, "shape") and len(tuple(frames.shape)) == 5:
            target = where / f"video_seed{int(request.seed)}.mp4"
            report = media_mod.write_video(frames, target, fps=int(plan.fps or 24),
                                           value_range="-1..1")
            result: dict[str, Any] = {
                # ⚠️ 文件是真的 ✓ 但内容仍是**未训练 VAE** 的产物 ✗ ⇒ `synthetic` 保持 True ✓
                "synthetic": True, "realTensors": True, "realFile": True,
                "videoPath": str(target), "primaryPath": str(target),
                "video": report,
                "artifacts": [{"kind": "video", "path": str(target), "bytes": report["bytes"]}],
                "note": "**真 mp4** ✓（ffprobe 可复核 ✓），但画面来自**未训练**的参考 VAE ✗",
            }
            if waveform is not None:
                # ⚠️ 双流：音频**单独落 wav** ✓（真 16-bit PCM ✓ 标准库 ✓）—— 不塞进 mp4 ✗：
                #    塞进去要重编码 ✓，而"两条产物各是各的"更好独立核对 ✓（本仓一惯做法 ✓）。
                audio_target = where / f"audio_seed{int(request.seed)}.wav"
                rate = int(outputs.get("sampleRate")
                           or audio_vae_mod.AudioVAEConfig().sample_rate)
                audio_report = media_mod.write_wav(waveform, audio_target, sample_rate=rate)
                result["artifacts"] = list(result["artifacts"]) + [
                    {"kind": "audio", "path": str(audio_target), "bytes": audio_report["bytes"]}]
                result["audioPath"] = str(audio_target)
                result["audio"] = audio_report
                # ⚠️ 这是"本次真的产出了音频"的**铁证** ✓（不是"后端支持音频"那种能力声明 ✗）
                result["dualStream"] = True
                result["note"] = (str(result["note"])
                                  + "＋ **真 wav** ✓（同一份音频的两条各自落盘 ✓）")
            return result
        target = where / f"torch_stub_seed{int(request.seed)}.json"
        target.write_text(json.dumps({
            "synthetic": True,
            "note": "torch 后端占位产物：真张量统计 ✓，**不是生成的画面** ✗",
            "request": request.to_dict(), "plan": plan.to_dict(), "decode": outputs,
        }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return {
            "synthetic": True, "realTensors": True, "realFile": True, "videoPath": None,
            "artifacts": [{"kind": "torch-stub-manifest", "path": str(target),
                           "bytes": target.stat().st_size}],
            "primaryPath": str(target),
        }
