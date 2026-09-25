"""**torch 后端** —— 真张量实现（2026-09-17；本机已装 CPU 版 torch 2.14 ✓）。

## 它现在是**真**的哪一半（务必分清 ✗）

| 维度 | 现状 |
|---|---|
**张量本身** | ✅ **真 torch 张量** ✓（`realTensors=True` ✓）—— 不再用 `dryrun` 那个 12 行 `list[float]` 假张量 ✓ |
**随机与复现** | ✅ 真 `torch.Generator().manual_seed(seed)` + `torch.randn` ✓ ⇒ 同种子**逐位可复现** ✓ |
**采样/引导/首帧数学** | ✅ 全部跑在真张量上 ✓（与 `dryrun` 共用同一份 `sampler`/`guidance`/`conditioning` ✓） |
**模型前向（DiT/TE/VAE）** | ✅ **机制已接** ✓（`load_weights` 真装载 DiT/H3 形态 ✓ + `denoise`/`encode_text`/`decode` 真前向 ✓ + H3 双流 `sample_dual` 真前向 ✓）—— ⚠️ **真权重（主 DiT 19.53 GiB ✗）没下载** ⇒ 没装模型时前向是**占位** ✓（如实标 ✓ 不冒充 ✗） |
**产物** | ✅ **H3 双流已出真 mp4 + 真 wav** ✓（`engine_dual_stream_test` ✓）—— ⚠️ 没装真权重/单流时仍是合成产物 ✗（`synthetic=True` ✓） |

⇒ 两个标志**同时**给出，不许混 ✗：`realTensors=True`（张量真 ✓）+ `synthetic=True`（没装真权重时画面不真 ✗）。
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
import os
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
    #: ⭐ ``True`` ⇒ **可选**（缺了也**不阻断** ✓）。判据：本仓有等价的自研实现 ✓（如自研 BPE ✓）
    optional: bool = False


#: 真后端所需依赖 ✓（``module`` 用于探测 ✓，``package`` 用于给出安装命令 ✓）
DEPENDENCIES: tuple[_Dependency, ...] = (
    _Dependency("torch", "torch", "张量与设备运行时 ✓", 124),
    _Dependency("numpy", "numpy", "数组互转 / 数值工具", 20),
    _Dependency("safetensors", "safetensors", "权重读取（本仓另有纯 Python 读取器 ✓ 体检不必装 ✓）", 1),
    _Dependency("PIL", "pillow", "首帧/参考图解码 ✓", 3),
    # ⚠️ **可选** ✓：``transformers`` 只是「HF tokenizer 适配」这一条通道 ✓ ——
    #    本仓**自己实现了字节级 BPE** ✓（`engine/tokenizer_bpe.py` ✓ 零依赖 ✓ 离线 ✓）
    #    ⇒ 缺它**不该**把后端判成不可用 ✗（`torch_available()` 走的是 `ready` ✓ = **必需**项齐不齐 ✓）。
    _Dependency("transformers", "transformers",
                "HF tokenizer 适配（本仓另有**自研 BPE** ✓ ⇒ 不装也能离线分词 ✓）", 120,
                optional=True),
)

#: ⚠️ **仍未实现**的部分 ✓（依赖齐了、DiT 也装好了，也还是这些 ✗）—— 分开报，别让人去查环境 ✗
#: ⚠️ 2026-09-20 **更正**：上一版清单**过期了** ✗ —— 它写着「文本编码器前向 ✗」「VAE 解码与出片落盘 ✗」，
#: 但这些**机制都已经实现** ✓（`encode_text` ✓ / `attach_text_encoder` ✓ / `decode` ✓ /
#: `attach_vae` ✓ / `write` ✓，且 `engine_io_test` / `engine_text_test` 已跑通
#: 「TE → DiT → VAE → **真 mp4**」✓）。⇒ **真正缺的只有「真权重 + 真配置」** ✓：
#: 机制齐了、权重没到 ✓ ⇒ `canGenerate=False` **依然正确** ✓（只是别把原因归到机制上 ✗）。
PENDING_PARTS: tuple[str, ...] = (
    "✅ **低精度权重能真装了** ✓（2026-09-22 ✓ `engine/quant.py` ✓）：fp8/int8 先按配套 scale **反量化** ✓"
    "（布局**按形状**判 ✓ per-tensor / per-row / per-col / 逐元素 ✓；`*_scale_inv` 走除 ✓）"
    "⇒ 再归一 dtype ✓ —— ⚠️ 此前是**直接 `to(bf16)`** ✗ ⇒ 尺度丢掉但**不报错** ✓✗（19.53 GiB 的主权重"
    "就是 fp8 ✓）。⚠️ 判不出来（块量化 / 分组量化 / 缺 scale / scale 含 0 ✓）⇒ **中止装载** ✓ 不按猜的算 ✗。",
    "H3 真权重未下载（主 DiT 19.53 GiB ✗ ⇒ **上机前先跑一次** "
    "`python app/scripts/h3_readiness.py` ✓ —— 它把依赖 / 权重就绪 / 加载计划 / "
    "**真权重预检（键名核对 + 结构推导）+ 词表** 串成一次调用 ✓ "
    "（拿到权重后加 `--weights <路径> [--tokenizer <词表目录>]` ✓ 判据见 `engine_readiness_script_test` ✓））",
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
    "✅ **H3 结构推导已实现** ✓（2026-09-20 ✓ `h3_keys.infer_h3_trunk_config` ✓）：尺寸 / 层数 / "
    "head_dim / ffn / text_dim / modalities / `inv_freq_len` / **PDD 头库** 全部**从权重读** ✓"
    "（出厂常量只作回落 ✗ ⇒ 社区重导出 / 蒸馏 / 自检缩小版走**同一条**路 ✓）；"
    "每个字段的来源随装载报告给出 ✓（`configSources` ✓ —— 免得「回落」被读成「权重事实」 ✗）。"
    "⚠️ 仍留给真权重的一步：按元数据复核 `patch_size` ✗（**权重里没有 patch 事实** ⇒ 只能显式给 ✓）"
    "与 eps 类（**形状验不出 eps** ✗ ⇒ 用参考默认 1e-5 ✓）。"
    "⭐ 2026-09-22 收紧一半 ✓：`patch_size` 现在有**一条独立约束** ✓ —— 主干推的 `latents_dim` 与"
    "**VAE 自述的潜通道数**必须一致 ✓（`vae.py` 自己写着这两者是同一个事实 ✓）⇒ 给错的 `patch_size`"
    "（**能整除**那种 ✓✗：形状全自洽、装得进去 ✓✗）会在**挂载期**被拒 ✓，不再拖到解码时报形状错 ✗"
    "（见 :meth:`TorchBackend._check_attached_vaes` ✓ / 自检 55~56′ ✓）。"
    "⚠️ 但这仍是**两条自述相互印证** ✗（两边一起错会漏 ✗）**不是权重事实** ✗ ⇒ 真权重到手后照样要核 ✓",
    "✅ **tokenizer 已自研** ✓（2026-09-20 ✓ `engine/tokenizer_bpe.py` ✓ 零依赖 ✓ 离线 ✓）："
    "字节级 BPE ✓（GPT-2 映射算法生成 ✓ + 自写预分词扫描器 ✓ + 按 merges 排名合并 ✓ + 往返恒等 ✓），"
    "读**随权重来的**词表文件 ✓（`tokenizer.json` ✓ 或 `vocab.json`+`merges.txt` ✓，"
    "`load_tokenizer` 嗅探 ✓）⇒ **不需要外部 `transformers`** ✓"
    "（`transformers` 已登记为**可选**依赖 ✓ ⇒ 它只是另一条通道 ✓，且能当**核对用**的第二实现 ✓"
    "—— 自检里已逐例同 id 比过 ✓）。⚠️ 仍缺的只是**词表文件本身** ✗（那是权重的一部分 ✗ "
    "本仓不内置任何词表 ✓）",
    "✅ **分词器总入口也升级了** ✓（2026-09-20 ✓ `engine/tokenizer_hub.py` ✓）：形态嗅探 ✓ → "
    "自研 BPE 优先 ✓ → `Unigram`/`WordPiece`/`Metaspace`（Llama/Qwen 系 ✓）**回退参考实现** ✓ "
    "+ 批量 ✓ LRU 缓存 ✓ 离线/关遥测/缓存目录 ✓ 运行期互校 ✓ "
    "⇒ 词表形态**不再限制能力** ✓（`attach_text_encoder(tokenizer_path=…)` 直接给路径 ✓）",
    "文本编码器 / VAE 的**权重**（机制已实现 ✓；tokenizer 走**注入** ✓ 本仓已有自研 BPE ✓）",
    "端到端 `generate()` 拿真权重跑一次（现在 `canGenerate=False` ✓ = 「机制齐、权重没到」✓）",
)


def _same_setting(left: Any, right: Any) -> bool:
    """两个「配置值」是不是**同一个** ✓（``tuple`` ↔ ``list`` 算相同 ✓、数是数 ✓）。

    ⚠️ 为什么要专门写它 ✗：`patch_size` / 形状的写法在配置里可能是 ``[1,2,2]`` ✓、
    推断出来的是 ``(1,2,2)`` ✓ —— 直接用 `!=` 比会**假报"不一致"** ✓✗（把对的配置拒掉 ✓）。
    """
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return tuple(left) == tuple(right)
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _spec(name: str) -> Any:
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ValueError):  # pragma: no cover - 极少数损坏的安装
        return None


def dependency_status() -> dict[str, Any]:
    """逐项依赖现状 ✓（**不导入**它们 ✓ ⇒ 毫秒级、无副作用 ✓）。

    ⚠️⚠️ 两个清单**分开答** ✓（2026-09-20 起 ✓）：``missing`` = **必需**项（缺 ⇒ 后端不可用 ✓）；
    ``optionalMissing`` = **可选**项（缺 ⇒ 只少一条通道 ✓ —— 例如缺 ``transformers`` 时
    仍有**自研 BPE** 能分词 ✓）。⇒ 混在一起会让「可选没装」把整个后端判死 ✓✗。
    """
    items: list[dict[str, Any]] = []
    missing: list[str] = []
    optional_missing: list[str] = []
    for dependency in DEPENDENCIES:
        found = _spec(dependency.module) is not None
        items.append({
            "module": dependency.module, "package": dependency.package,
            "present": found, "purpose": dependency.purpose,
            "approximateMB": dependency.approximate_mb, "optional": dependency.optional,
        })
        if not found:
            (optional_missing if dependency.optional else missing).append(dependency.package)
    return {
        "items": items,
        "missing": missing,
        "optionalMissing": optional_missing,
        "ready": not missing,
        "install": ([f"pip install {' '.join(missing)}"] if missing else []),
        "installOptional": ([f"pip install {' '.join(optional_missing)}"]
                            if optional_missing else []),
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
        #: 超清放大器 ✓（按**需求装载** ✓ —— 没要超清就不装 ✓ 省显存）；装载报告与本次二采的实况 ✓
        self._upscaler: Any = None
        self._upscalerReport: dict[str, Any] | None = None
        self.refineDetails: dict[str, Any] | None = None

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

    # ── ⚠️ 调用方给的张量：**只点破，不搬运** ✗（2026-09-25 定的语义 ✓）─────────────────────
    def _device_torch(self) -> Any:
        """``self._device`` 的**具体** :class:`torch.device` ✓（``"cuda"`` ⇒ ``cuda:0`` ✓）。

        ⚠️⚠️ **必须规范化，不许比字符串** ✗✗（这是**真踩过的坑** ✓）：后端写的是 ``"cuda"`` ✓，
        而张量的 ``.device`` 永远带卡号 —— ``cuda:0`` ✓ ⇒ ``str()`` 一比就成了"不是同一设备" ✗。
        2026-09-25 本校验器**初版正是这么写的** ✗，当场把 `engine_pipeline_test` 的真链路
        （管线自己的张量 ✓，本来就该放行 ✓）误判成混用 ✗✗ —— 好在报错文案里
        ``cuda:0`` vs ``cuda`` 并排摆着 ✓，一眼看得出是**判据**错而不是链路错 ✓。

        ⚠️ 规范化**不等于放水** ✗：多卡机器上 ``cuda:1`` 与 ``cuda:0`` 是**真不同** ✓
        ⇒ 补完卡号后照旧要报错 ✓（本仓判据：宁可当场说清 ✓，不许"看着能跑就放过" ✗）。
        ⚠️ 没卡时 ``cuda.is_available()`` 为假 ⇒ 就按 ``torch.device("cuda")`` 比 ✓，
        那种情况下张量也不可能是 ``cuda:0`` ✓ ⇒ 仍然会如实报错 ✓。
        """
        torch = self._torch()
        device = torch.device(self._device)
        if device.type == "cuda" and device.index is None and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    def _require_own_device(self, where: str, **tensors: Any) -> None:
        """校验调用方给的张量**已经在** :attr:`device` 上 ✓✗ —— 不在就**明确报错** ✗（**不替它搬** ✗）。

        ⚠️⚠️ **为什么不替调用方搬**（这是**有意留着**的语义 ✓，不是没做完 ✗）：
        唯一的搬法是拷一份 ``.to(...)`` ✓ ⇒ 调用方手里那个张量**不再被就地改写** ✗ ——
        而 :meth:`condition_first_frame` 的语义正是「按掩码混进第 0 个潜帧、**回新的 latents**」✓；
        悄悄改其中任何一半都是**静默换语义** ✗✗（比报错坏得多 ✓）。再者：搬走要多占一份显存 ✗
        （真权重下潜变量几百 MB ✓），而「该在哪个设备上」**只有调用方知道** ✓。

        ⚠️⚠️ **为什么非要点破**：混用时 torch 原生报错是
        ``Expected all tensors to be on the same device`` / ``aten::slow_conv3d_forward`` ✓✗ ——
        它**不说是哪个参数** ✗、也不说后端在哪个设备 ✗ ⇒ 在**有卡那台**上排查成本极高 ✓✗
        （本仓 2026-09-25 正为此烧掉一轮 ✓）。

        ⚠️ 只认**带 `device` 的对象**（真 torch 张量 ✓、挂上的模块 ✓）：``None`` / 数字 / 字符串 /
        干跑后端的 ``TinyTensor``（**没有** `device` 属性 ✓）/ 引擎给的 ``plan.sigmas``（纯 float 列表 ✓）
        一律**原样放行** ✓ —— 宁可放过 ✓，绝不许误伤真链路 ✗。

        ## ⭐ 入口**逐个查过**的清点表 ✓（2026-09-25 ✓ —— 「全」是这么个全法 ✓：**写明结论** ✗✗）

        | 入口 | 吃调用方张量？ | 处置 |
        |---|---|---|
        | :meth:`denoise` | ✓ `latents` / `condition` | **校验** ✓ |
        | :meth:`condition_first_frame` | ✓ `latents` | **校验** ✓ |
        | :meth:`sample_dual` | ✓ `latents` / `condition`（``sigmas`` 是纯 float ✓ 自动放行 ✓） | **校验** ✓ |
        | :meth:`refine_latents` | ✓ `latents` / `condition`（放大器在 ``self._device`` ✓） | **校验** ✓ |
        | :meth:`decode` | ✓ `latents` | **校验** ✓ |
        | :meth:`write` | ✓ `outputs["frames"]` | ⚠️ **不校验** ✗ —— 落盘边界自己收口 ✓✗：``media.write_video`` / ``write_wav`` 结尾都 ``.to("cpu", …)`` ✓ ⇒ **cpu 上解出来的帧照样能落盘** ✓（加了校验反而把这条正当用法拒掉 ✗✗，见该方法的注释 ✓） |
        | :meth:`init_latents` / :meth:`init_dual_latents` | ✗ **出**张量（自己造 ✓ 已在 ``self._device`` ✓） | 无需 ✓ |
        | :meth:`encode_text` / :meth:`latent_shape` / :meth:`condition_width` / :meth:`describe` | ✗（只吃 ``request`` / ``plan`` ✓） | 无需 ✓ |
        | :meth:`load_weights` / :meth:`attach_*` | ✗（吃**模块/配置** ✓ 不是张量 ✓） | 无需 ✓ |
        | 私有 ``_decode_dual`` / ``_dual_extra_rows`` / ``_second_pass`` / ``_upscale_keyframes`` | ✗ 内部 ✓（张量都来自上面已校验的入口 ✓） | 无需 ✓ |

        ⇒ 真链路（管线 ✓）**一个校验都碰不到** ✓：它拿到手的每个张量都是本后端自己造的 ✓。
        """
        for name, value in tensors.items():
            self._require_one_device(where, name, value)

    def _require_one_device(self, where: str, name: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                self._require_one_device(where, f"{name}[{key!r}]", item)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self._require_one_device(where, f"{name}[{index}]", item)
            return
        found = getattr(value, "device", None)
        if found is None:
            return
        expected = self._device_torch()
        if found == expected:
            return
        raise TorchBackendUnavailable(
            f"{where}：调用方给的 ``{name}`` 在 **{found}** ✗，而本后端（含已装的权重/放大器）在 "
            f"**{expected}** ✓（``self._device = {self._device!r}`` ✓）⇒ 混用必炸 ✗。"
            f"⚠️ 本仓**不替调用方搬张量** ✗ ⇒ 该由调用方自己 ``.to(\"{expected}\")`` ✓。"
            f"⚠️ **管线自己**的张量全在 {expected} 上 ✓ ⇒ 真链路永远走不到这里 ✓；"
            f"走到这儿说明有人**拿自己造的张量**直接调了后端 ✓（正是自检的用法 ✓ ⇒ "
            f"自检要像 engine_io / engine_refine 那样**显式钉同一个设备** ✓✗）。"
            f"⚠️ 点破它只为让报错**指得到原因** ✓：torch 原生只会说 "
            f"``Expected all tensors to be on the same device`` ✗、不说是哪个参数 ✗。",
            reason="pending")

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
        from app.services.engine import h3_keys  # noqa: PLC0415 —— 同上 ✓（结构推导/键名核对都**不需要 torch** ✓）
        if h3_form.looks_like_h3_form(info.tensors.keys()):
            explicit = dict(config) if isinstance(config, dict) else {}
            # ⭐⭐ **结构从权重推** ✓（2026-09-20 起）—— 出厂常量 `H3_TRUNK_DEFAULTS` 只作**回落** ✗：
            #    尺寸 / 层数 / 头数 / head_dim / ffn / text_dim / modalities / `inv_freq_len` /
            #    **PDD 头库** 全部读权重 ✓ ⇒ 自检的**缩小版**与工作站上的**真权重**走**同一条**路 ✓。
            #    ⚠️ 老写法（`form_defaults = H3_TRUNK_DEFAULTS` + 只推 banks ✗）对**非出厂**检查点
            #    （社区重导出 / 蒸馏 / 缩小版 ✓）会「名字对、形状错」✓✗ —— 那正是 `PENDING_PARTS`
            #    第 4 条「H3 的 DiTConfig」剩下的机制缺口 ✓（现在补上了 ✓）。
            shapes = {name: tensor.shape for name, tensor in info.tensors.items()}
            inferred = h3_keys.infer_h3_trunk_config(shapes, patch_size=explicit.get("patch_size"))
            if not inferred.ok:
                reasons = list(inferred.problems)
                if inferred.audit.missing:
                    reasons.append(f"权重缺 {len(inferred.audit.missing)} 个键 ✗"
                                   f"（如 {inferred.audit.missing[:4]} ✓）")
                if inferred.audit.shape_mismatch:
                    reasons.append(f"{len(inferred.audit.shape_mismatch)} 处形状不符 ✗"
                                   f"（如 {inferred.audit.shape_mismatch[:2]} ✓）")
                raise TorchBackendUnavailable(
                    "H3 权重推不出可装的结构 ✗（**不拿出厂常量硬装** ✗ —— 那会「名字对、形状错」✓✗）："
                    + "；".join(reasons), reason="pending")
            form_defaults = {**h3_form.H3_TRUNK_DEFAULTS, **inferred.config}
            # ⚠️ 显式配置**只做交叉校验** ✓：与权重推断不一致 ⇒ **报错** ✗（不静默取其中一个 ✓
            #    —— 老版只在 `head_banks` 上守这条 ✓，现在推广到**每个**可推字段 ✓）。
            conflicts = {key: [value, inferred.config[key]] for key, value in explicit.items()
                         if key in inferred.config and not _same_setting(value, inferred.config[key])}
            if conflicts:
                raise TorchBackendUnavailable(
                    f"配置与权重推断出的结构**不一致** ✗：{conflicts}"
                    f"（两者必须一致 ✓ —— 不静默取其中一个 ✗）", reason="pending")
            form_defaults.update({key: value for key, value in explicit.items()
                                  if key not in inferred.config})   # 推不出的字段（如 eps ✓）由显式配置补 ✓
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
            try:
                vae_check = self._check_attached_vaes()
                text_check = self._check_text_encoder()   # ⭐ 已挂的 TE 也在这一刻再核一次 ✓
            except TorchBackendUnavailable:
                # ⚠️ 校验失败 ⇒ **中止装载 + 不污染模块** ✓（本仓在反量化那条上定过这条规矩 ✓）：
                #    留着半装状态会让 `dualStream` / `latentMode` 按"装好了"自述 ✓✗。
                self._model = None
                self._config = None
                self._form = None
                raise
            self._loadReport = {
                **form_report.to_dict(),
                # ⭐ 「哪些字段是**权重里读出来的** ✓、哪些是**不可推的回落** ✗」随报告一起给 ✓
                #    —— 装真权重时，光看 `complete=True` 分不清这两种 ✓✗（本仓那条纪律 ✓）。
                "configSources": dict(inferred.sources),
                "derivedFields": inferred.derived_count,
                "fieldCount": len(inferred.config),
                # ⭐ 已挂的 VAE 在这里**再查一次** ✓（挂载顺序反过来也要守 ✓ ——
                #    先挂 VAE 的那次调用时形态还没定 ⇒ 只能等这一刻 ✓）；空 = 还没有 VAE 可比 ✓
                "vaeCheck": vae_check,
                "textEncoderCheck": text_check,
            }
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
            # ⭐ **已挂 VAE 的跨来源校验结果** ✓（核过要能自证 ✗；⚠️ 这里**不抛** ✗ ——
            #    `describe()` 任何时候都得能调 ✓，所以对不上只记为 `agrees=False` ✓）
            "vaeCheck": self._check_attached_vaes(raise_on_mismatch=False),
            # ⭐ 文本编码器的 `output_dim` ↔ 主干 `text_dim` ✓（同上：这里**不抛** ✗）
            "textEncoderCheck": self._check_text_encoder(raise_on_mismatch=False),
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
        # ⚠️ `sigmas` 是引擎给的**纯 float 列表** ✓（没 `device` ⇒ 校验器原样放行 ✓）。
        self._require_own_device("sample_dual", latents=latents, sigmas=sigmas, condition=condition)
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

    def _upscaler_path(self) -> str | None:
        """放大器权重的解析顺序 ✓：显式 ``H3_UPSCALER`` ✓ > 清单里 ``kind=upscale_models`` 的第一条 ✓。

        ⚠️ 与 :meth:`_default_dit_path` **同一份清单** ✓（不另开一处配置 ✗）；⚠️ 解析不到 ⇒ ``None`` ✓
        （由调用方给**明确拒绝**的文案 ✓ 不猜路径 ✗）。
        """
        explicit = str(os.environ.get("H3_UPSCALER") or "").strip()
        if explicit:
            return explicit
        for entry in inv.load_catalog()["models"]:
            if str(entry.get("kind")) == "upscale_models":
                resolved = inv.component_path(entry)
                return str(resolved) if resolved else None
        return None

    def refine_latents(self, latents: Any, plan: Any, request: Any, *,
                       condition: Any = None) -> Any:
        """**超清二采（张量层 ✓ 2026-09-25 补 ✓）**：空间 2× ✓ ⇒（可选）**带掩码的低噪声重去噪** ✓✗。

        ⚠️⚠️ 四条口径（都是「看着像超分、其实坏了」的形状 ✓✗）：
        1. ⭐ **只重采视频流** ✗✗：``audio`` **原对象放回** ✓（重采音频会把音轨弄坏而画面看着正常 ✓）——
           二采那条路靠的是**掩码**（video **1** / audio **0** ✓，口径来自上游 ``_h3_build_denoise_mask`` ✓✗）；
        2. ⭐ **时间维不变** ✗✗（``T`` 前后必须相等 ⇒ 不等就**拒** ✓ —— 时间插值会让动作速率变错 ✓✗）；
        3. ⭐ 放大器**严格装载** ✓：契约与张量不符（缺键/形状不一 ✓）⇒ **拒** ✗
           （``strict=False`` 会留下随机初始化的层 ⇒ 输出是「像超分」的噪声 ✓✗）；
        4. ⭐ **二采的两个参数必填** ✗（``plan.refine_steps`` / ``refine_denoise`` ✓）—— 默认值在
           **编译层** ✗ ⇒ 不给就**只做上采样** ✓ 并如实写进报告 ✓；给了一半 ⇒ ``plan_upscale`` 就报错 ✓。
        ⚠️ **要二采就必须给 ``condition``** ✗（二采要重跑主干 ✓ —— 没条件就**拒** ✓，不猜 ✓✗）。
        """
        self._gate()
        # ⚠️ 放大器在 `self._device` 上 ✓ ⇒ 拿 CPU 潜变量进来必炸 ✗（2026-09-25 真机踩过 ✓）⇒ 先点破 ✓。
        self._require_own_device("refine_latents", latents=latents, condition=condition)
        torch = self._torch()
        from app.services.engine import h3_form, upscale as upscale_mod, upscale_net  # noqa: PLC0415
        from app.services.engine import latent_container as lc  # noqa: PLC0415

        if not isinstance(latents, dict) or "video" not in latents or "audio" not in latents:
            raise TorchBackendUnavailable(
                "二采需要 :meth:`init_dual_latents` 的产出（含 video/audio 两条 ✓）✗"
                "　—— ⚠️ 别拿单流潜变量来二采 ✓✗", reason="pending")
        path = self._upscaler_path()
        if path is None or not Path(path).exists():
            raise TorchBackendUnavailable(
                "放大器权重未就绪（" + (path or "清单里 ``kind=upscale_models`` 一条都没有 ✗") + " ✓）"
                "　⇒ 先放权重（或用 ``H3_UPSCALER`` 指路 ✓），或让计划层按 ``contract=None`` "
                "**回退普通模式** ✓（⚠️ 本仓不静默降级出「看着像超清」的片子 ✗）", reason="pending")
        metadata, _header = st.read_header(path)
        contract = upscale_mod.read_upscaler_contract(metadata)
        channels = int(contract["base_config"]["in_channels"])
        # ⭐ 跨来源**同一个事实** ✓✗：契约的 ``in_channels`` ↔ **本次装的主干**的视频流通道 ✓。
        #    ⚠️ 通道数**从已装主干推** ✗（不写死 24 ✗）—— 自检的**缩小版**与真权重要走**同一条路** ✓✗
        #    （真权重推出来就是 24 ✓，缩小版推出来是它自己的 latents_dim ✓）。
        expected = int((self._config or {}).get("latents_dim") or lc.H3_VIDEO_CHANNELS)
        if channels != expected:
            raise TorchBackendUnavailable(
                f"放大器契约的 ``in_channels={channels}`` ✗，而本次主干/视频流是 {expected} 通道 ✓ "
                f"（真权重下应当都是 {lc.H3_VIDEO_CHANNELS} ✓）⇒ 不是同一套 ✓✗（拒，不硬套 ✗）",
                reason="pending")
        if self._upscaler is None:
            # ⚠️ 用 ``weights.load_module_weights`` 而不是 ``upscale_net.build_upscaler`` ✗：
            #    后者要**内存里的 state_dict** ✓，而本仓的装载入口是**文件 + 严格报告** ✓
            #    （``complete=False`` ⇒ 拒 ✓ —— 与 ``strict=True`` **同一条纪律** ✓✗）。
            net = upscale_net.H3LatentUpscalerV3(contract["base_config"], contract["config"])
            report = weights_mod.load_module_weights(net, path, device=self._device)
            if not report.complete:
                raise TorchBackendUnavailable(
                    f"放大器的张量与声明的架构**不符** ✗（缺 {len(report.missing)} 键 / "
                    f"形状不符 {len(report.shapeMismatch)} 处 ✓，如 {report.missing[:4]} ✓）"
                    f"　⇒ 拒 ✗（**不许** ``strict=False`` 糊过去 ✓✗）", reason="pending")
            self._upscaler = net.to(self._device).eval().requires_grad_(False)
            self._upscalerReport = {**report.to_dict(), "path": path, "contract": dict(contract)}

        video = latents["video"]
        shape = tuple(int(value) for value in video.shape)
        if len(shape) != 4 or shape[0] != channels:
            raise TorchBackendUnavailable(
                f"视频流形状 {shape} ✗：本后端的双流是 ``[C,T,H,W]``（C 必须是 {channels} ✓）— "
                f"⚠️ 拿别的流当视频解会得到一堆噪声 ✓✗", reason="pending")
        with torch.no_grad():
            refined = self._upscaler(video.unsqueeze(0)).squeeze(0)
        after = tuple(int(value) for value in refined.shape)
        if after[1] != shape[1]:
            raise TorchBackendUnavailable(
                f"放大器**改了时间维** ✗：{shape[1]} → {after[1]} ✓（时间插值会让动作速率变错 ✓✗）"
                f" ⇒ 拒，不用这个产物 ✓", reason="pending")
        tail_steps = int(getattr(plan, "refine_steps", 0) or 0)
        denoise = float(getattr(plan, "refine_denoise", 0.0) or 0.0)
        details: dict[str, Any] = {
            "videoShapeBefore": list(shape), "videoShapeAfter": list(after),
            "temporalUnchanged": after[1] == shape[1],
            # ⭐ 真话：音频那条流**不进**任何"会改它"的路 ✓ 原对象放回 ✓
            "audioIdentical": True, "audioTouched": False,
            "denoiseSteps": 0, "mask": None,
            "upscalerPath": path,
            "notes": [],
        }
        out_video = refined
        keyframes = self._upscale_keyframes(latents, refined, shape)
        if tail_steps > 0:
            out_video, denoise_report = self._second_pass(
                refined, latents, plan, request, condition=condition, keyframes=keyframes)
            details.update(denoise_report)
        else:
            details["notes"].append(
                "⚠️ **二采未跑** ✗：``plan.refine_steps`` = 0 ✓（这两个参数的默认值在**编译层** ✗ ⇒ "
                "本仓**不猜** ✓）⇒ 本次只做了放大器上采样 ✓；要二采请把 ``refineSteps`` / "
                "``refineDenoise`` 一起给上 ✓✗")
        if keyframes is not None:
            details["keyframesScaled"] = len(keyframes)
        self.refineDetails = details
        merged = {**latents, "video": out_video}
        if keyframes is not None:
            merged["keyframes"] = keyframes
        # ⚠️ 音频流**原对象**放回 ✓（不是 copy ✗ —— copy 也会让「是同一份」这条判据失效 ✓✗）
        return merged

    def _upscale_keyframes(self, latents: Any, refined: Any, shape: tuple[int, ...]) -> Any:
        """⭐ 关键帧/参考行的潜变量**也要 2×** ✗✗（口径来自上游 ``_h3_scale_cond_refs`` ✓）。

        ⚠️ 不同步就会「目标域 2× 而关键帧还是 1×」✓✗ —— 形状对不上一般会**报错** ✓，
        但**擦边对得上**的组合会**静默**错 ✓✗（所以这里也顺手核一遍）。
        ⚠️ ``references`` 要不要跟着 2× **没核到** ✗ ⇒ **拒** ✗（不猜 ✓✗）。
        """
        if latents.get("references"):
            raise TorchBackendUnavailable(
                "二采遇到 ``references`` ✗：它们在超清二采里**要不要跟着 2× 我们没核到** ✓ ⇒ "
                "**不猜** ✗（要么先去核上游、要么这次别带参考 ✓✗）", reason="pending")
        from app.services.engine import upscale_net  # noqa: PLC0415

        items = latents.get("keyframes")
        if not items:
            return None
        scaled: list[Any] = []
        for item in items:
            if not isinstance(item, dict) or item.get("latent") is None:
                scaled.append(item)          # 不是"带 latent 的帧"⇒ 原样带过去 ✓（不猜它是什么 ✗）
                continue
            latent = item["latent"]
            if len(tuple(int(value) for value in latent.shape)) != 5:
                raise TorchBackendUnavailable(
                    f"关键帧潜变量应当是 5 维（收到 {tuple(int(v) for v in latent.shape)} ✓）✗",
                    reason="pending")
            # ⚠️ 用**同一个** 2× 口径 ✓（`upscale_net.spatial_bilinear_2x` ✓ —— 与放大器内部一致 ✓✗）
            upscaled = upscale_net.spatial_bilinear_2x(latent.float())
            scaled.append({**item, "latent": upscaled.to(latent.dtype)})
        return scaled

    def _second_pass(self, refined: Any, latents: Any, plan: Any, request: Any, *,
                     condition: Any, keyframes: Any) -> tuple[Any, dict[str, Any]]:
        """**带掩码的低噪声二采** ✓✗（口径来自上游可读源码 ✓；σ 曲线用**本仓自研**调度 ✓ 已注明 ✓）。

        上游 ``studio_node.py`` 的可核事实 ✓：① 掩码 = video **1** / audio **0** ✓；
        ② ``total = max(steps+1, round(steps/clamp(denoise,0.15,1)))`` ⇒ 取**尾部** ``steps+1`` 个 σ ✓；
        ③ 噪声种子 = 段种子 **+1000001** ✓；④ cond 的 keyframes 也要 2× ✓；⑤ 失败 ⇒ **回退一采结果** ✓。
        """
        from app.services.engine import h3_form, schedules  # noqa: PLC0415
        from app.services.engine import upscale as upscale_mod  # noqa: PLC0415

        torch = self._torch()
        if condition is None:
            raise TorchBackendUnavailable(
                "要跑二采就必须给 ``condition`` ✗（二采要重跑主干 ✓ —— 没条件就**拒** ✓ 不猜 ✓✗）",
                reason="pending")
        if self._model is None:
            raise TorchBackendUnavailable(
                "二采要重跑主干，而**主 DiT 还没装载** ✗ ⇒ 先 load_weights ✓（不静默跳过二采 ✗✗）",
                reason="pending")
        tail_steps = int(getattr(plan, "refine_steps", 0))
        denoise = float(getattr(plan, "refine_denoise", 0.0))
        total = upscale_mod.refine_tail_steps(tail_steps, denoise)
        try:
            full = schedules.sigmas_for(total, str(getattr(request, "schedule", "karras") or "karras"))
        except (ValueError, TypeError) as err:
            raise TorchBackendUnavailable(f"二采取不到 σ 调度 ✗：{err}", reason="pending") from err
        tail_sigmas = [float(value) for value in list(full)[-(tail_steps + 1):]]
        tail_sigmas[-1] = 0.0
        states = condition
        if isinstance(condition, dict):
            if condition.get("negative") is not None:
                raise TorchBackendUnavailable(
                    "二采**不做 CFG** ✗（参考实现无引导 ✓）⇒ 条件里不该有 negative ✓ "
                    "（给了就说明按可引导的方式准备了条件 ✓ 不静默忽略 ✓）", reason="pending")
            states = condition.get("positive")
            if states is None:
                raise TorchBackendUnavailable("条件 dict 里没有 ``positive`` ✗ ⇒ 二采没法跑 ✓",
                                               reason="pending")
        seed = int(getattr(request, "seed", 0) or 0) + 1000001
        generator = torch.Generator(device=getattr(refined, "device", self._device)).manual_seed(seed)
        # ⚠️ 噪声**只加在视频流** ✗（音频那条锁住 ✓ —— 给它加噪再锁住就等于"把干净音轨换成噪声"✓✗）
        noisy = refined + torch.randn(refined.shape, generator=generator, device=refined.device,
                                      dtype=refined.dtype) * float(tail_sigmas[0])
        ones = torch.ones_like(noisy)
        zeros = torch.zeros_like(latents["audio"])
        try:
            sampled = h3_form.sample_dual_stream(
                self._model, noisy, latents["audio"], states, tail_sigmas,
                keyframes=keyframes or None, denoise_mask=(ones, zeros))
        except Exception as err:  # noqa: BLE001 —— 上游口径：二采失败 ⇒ **回退一采结果** ✓ + 如实记 ✗
            return refined, {"denoiseSteps": tail_steps, "mask": "video=1,audio=0",
                             "secondPass": False, "seed": seed, "tailSigmas": len(tail_sigmas),
                             "denoiseError": f"{type(err).__name__}: {err}",
                             "notes": [f"⚠️ 二采**失败** ⇒ **回退到上采样结果** ✓（上游口径 ✓）；"
                                       f"原因：{type(err).__name__}: {err} ✓✗"]}
        after = tuple(int(value) for value in sampled["video"].shape)
        if after != tuple(int(value) for value in refined.shape):
            raise TorchBackendUnavailable(
                f"二采改了形状 ✗：{tuple(int(v) for v in refined.shape)} → {after} ✓ ⇒ 拒 ✗",
                reason="pending")
        return sampled["video"], {
            "denoiseSteps": tail_steps, "secondPass": True, "seed": seed,
            "mask": "video=1,audio=0", "tailSigmas": len(tail_sigmas),
            # ⭐⭐ 音频**逐位不变**的证明（掩码 0 ⇒ 该流不更新 ✓）—— 真跑出来的事实 ✓✗
            "audioIdentical": torch.equal(sampled["audio"], latents["audio"]),
            "temporalUnchanged": after[1] == int(refined.shape[1]),
            "notes": [f"二采：尾部 {tail_steps} 步（denoise={denoise:g} ⇒ total={total} ✓）、"
                      f"种子 = 段种子 + 1000001 ✓、掩码 video=1/audio=0 ✓",
                      "⚠️ σ **曲线**用本仓自研调度 ✓（上游那套在 ComfyUI 里 ✗）—— 口径是「取尾部」✓，"
                      "曲线本身**未逐点对照** ✗"],
        }

    #: ⭐ **自述「二采锁定音频流」** ✗✗ —— 没这条，管线会**明确拒绝**跑二采 ✓（见 pipeline 的 refine 阶段 ✓）
    @property
    def refineNote(self) -> dict[str, Any]:  # noqa: N802 —— 与前端 camelCase 对齐 ✓
        # ⚠️ 2026-09-25 更正 ✗：这里早先写 ``denoise: False`` +「带掩码二采**未实现**」✗，但
        #    :meth:`refine_latents` 的 ``_second_pass`` **早就实现了**带掩码二采 ✓✗（自述与实现相反 ✓ ——
        #    调用方读到 ``denoise=False`` 会以为二采没做 ✗）。现已对齐 ✓：``denoise=True`` 是**能力自述** ✓，
        #    「这一次做没做」由 ``plan.refine_steps`` 决定（不给 ⇒ 只上采样 ✓ 见 ``refine_latents`` ✓）。
        return {"locksAudio": True,
                "note": "只把**视频流**送进放大器 ✓（音频流原对象放回 ✓）；"
                        "带掩码二采**已实现** ✓（要 ``refineSteps``/``refineDenoise`` + ``condition`` + 主干 ✓；"
                        "σ 曲线用本仓自研调度 ✓ 未逐点对照上游 ✗）",
                "denoise": True,
                "upscalerReady": self._upscaler is not None,
                "upscalerPath": self._upscaler_path()}

    def denoise(self, latents: Any, sigma: float, condition: Any, request: Any) -> Any:
        """**真前向**（装了模型 ✓）或**占位**（没装 ✓）—— 由 :meth:`describe` 如实标注 ✓。

        * 真前向：``x0 = DiT(x, σ, context)``（流匹配 velocity → x0 ✓ 见 :func:`dit.flow_match_x0` ✓）；
        * 占位：``(1−g)·cond + g·x``（``g→0`` 当 ``σ→0`` ✓ 保证采样按时收敛 ✓）。
        """
        self._gate()
        # ⚠️ 调用方给的张量（自检常用 ✓）必须已在后端设备上 ✓✗ —— 不搬，只点破 ✓（见校验器 ✓）。
        self._require_own_device("denoise", latents=latents, condition=condition)
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
        # ⚠️ 这条最有必要点破 ✓：这里的 latents 常是**自检自己造的**（真链路里来自 `init_latents` ✓）⇒
        #    在**有卡那台**上最容易"自检 cpu / 后端 cuda" ⇒ 原报错指向 `aten::slow_conv3d_forward` ✗。
        self._require_own_device("condition_first_frame", latents=latents)
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

    def attach_text_encoder(self, config: Any = None, tokenizer: Any = None,
                            tokenizer_path: str | None = None) -> dict[str, Any]:
        """挂上 :mod:`app.services.engine.text_encoder` ✓ ⇒ :meth:`encode_text` 出**真条件** ✓。

        ⚠️ 它同样**未经训练** ✗ ⇒ 条件数值没有语义 ✓（验的是管道 ✓）。真权重到位后换实现即可 ✓
        （调用点不变 ✓）。tokenizer 三种给法（**都是注入** ✓ 本仓不内置词表 ✗）：

        * ``tokenizer=<Tokenizer>`` —— 任意实现 ✓（协议只要 `vocab_size` + `encode` ✓）；
        * ⭐ ``tokenizer_path=<目录或文件>`` —— **本仓自研 BPE** ✓（2026-09-20 ✓
          `engine/tokenizer_bpe.py` ✓ 零依赖 ✓ 离线 ✓）：自动嗅探 ``tokenizer.json`` ✓
          或 ``vocab.json`` + ``merges.txt`` ✓ ⇒ 词表随权重到手就能直接分词 ✓；
        * 都不给 ⇒ :class:`~app.services.engine.text_encoder.StubTokenizer`（**假桩** ✓
          只验管道 ✗ —— `describe().tokenizer` 会报 `stub` ✓ 一眼看得出 ✓）。
        """
        self._gate()
        if tokenizer is None and tokenizer_path:
            # ⭐ 走**总入口** ✓（2026-09-20 ✓）：形态嗅探 → 自研 BPE 优先 ✓ → 参考实现回退 ✓
            #    ⇒ 词表是 `Unigram`/`Metaspace`（Llama/Qwen 系 ✓）这类本仓自研**未覆盖**的形态时
            #    也能直接挂上 ✓（此前会**直接报错** ✗ ⇒ 能力到此为止 ✓✗）。
            from app.services.engine import tokenizer_hub  # noqa: PLC0415 —— 局部引 ✓ 顶层不碰 torch ✓
            tokenizer = tokenizer_hub.load(tokenizer_hub.HubConfig(path=str(tokenizer_path)))
            if config is None:
                # ⚠️⚠️ 嵌入表要**装得下**这个分词器 ✓ —— 必须用 `required_vocab_size`（含特殊符 ✓）
                #    而**不是** `vocab_size` ✗（后者不含 added tokens ⇒ 特殊符 id 会**越界** ✓✗；
                #    实测：合成词表里 `<|endoftext|>`=999 ⇒ 用 262 建表，下一行的守卫**当场拦下** ✓）。
                required = int(getattr(tokenizer, "required_vocab_size",
                                       getattr(tokenizer, "vocab_size", 0)) or 0)
                config = te_mod.TextEncoderConfig(
                    vocab_size=max(required, int(getattr(tokenizer, "vocab_size", 0)), 2),
                    output_dim=self._configured_text_dim(64))
        config = config or te_mod.TextEncoderConfig(
            output_dim=self._configured_text_dim(64))
        # ⚠️ **先核后改** ✗（2026-09-22 ✓）：核不过是**这份候选配置**的问题 ✓ ⇒ 此刻模块一个字没动 ✓
        #    —— 若改成"先挂上再核" ✗，失败时要么留着坏的 ✓✗、要么把**先前挂好的** TE 也清掉 ✓✗
        #    （⚠️ 后一种更坏 ✗：调用方只是想换一个，结果连原来那个也没了 ✗✗）。
        self._check_text_encoder(candidate=config)
        if tokenizer is not None:
            required = int(getattr(tokenizer, "required_vocab_size",
                                   getattr(tokenizer, "vocab_size", 0)) or 0)
            if required > int(config.vocab_size):
                raise TorchBackendUnavailable(
                    f"分词器需要嵌入表 ≥ {required} 个 id ✗，但 TE 的 vocab_size={config.vocab_size} ✓"
                    f" ⇒ 特殊符 id 会**越界** ✓✗（不静默截断 ✗ —— 请显式给足够大的 vocab_size ✓）",
                    reason="pending")
        self._tokenizer = tokenizer or te_mod.StubTokenizer(
            config.vocab_size, max_length=config.max_length)
        self._textEncoder = te_mod.build_text_encoder(config).to(self._device).eval()
        self._textConfig = config
        report = {
            "config": config.to_dict(),
            "tokenizer": getattr(self._tokenizer, "name", type(self._tokenizer).__name__),
            "note": "TE 与本仓库其它模型一样是**参考实现（未训练）** ✗ ⇒ 条件无数值语义 ✓",
        }
        if tokenizer_path:
            report["tokenizerPath"] = str(tokenizer_path)
        describe = getattr(self._tokenizer, "describe", None)
        if callable(describe):
            report["tokenizerDetail"] = describe()
        return report

    def attach_vae(self, config: Any = None) -> dict[str, Any]:
        """挂上 :mod:`app.services.engine.vae` 的参考 VAE ✓ ⇒ :meth:`decode` 能出**真帧** ✓。

        ⚠️ 它是**未经训练**的 ✓ ⇒ 画面是噪声 ✓（验的是**管道** ✓）。真 VAE 权重到位后换 `vae.py`
        的实现即可 ✓（`decode` 的调用点不变 ✓）。
        """
        self._gate()
        config = config or vae_mod.VideoVAEConfig()
        # ⚠️ **先核后改** ✗（2026-09-22 ✓）：核不过是**候选配置**的问题 ✓ ⇒ 此刻模块一个字没动 ✓
        #    —— 若改成"先挂上再核" ✗，失败时要么留着坏的 ✓✗、要么把**先前挂好的**那个也清掉 ✓✗。
        self._check_attached_vaes(video=config)
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
        self._check_attached_vaes(audio=config)   # ⭐ 同上：**先核后改** ✓（音频那一档 ✓）
        self._audioVae = audio_vae_mod.build_audio_vae(config).to(self._device).eval()
        self._audioVaeConfig = config
        return config.to_dict()

    def _check_text_encoder(self, *, candidate: Any = None,
                            raise_on_mismatch: bool = True) -> dict[str, Any] | None:
        """⭐⭐ **跨来源校验**：文本编码器的 `output_dim` ↔ 主干要的 `text_dim` ✓（**必须一致** ✗）。

        为什么要专门守 ✗（2026-09-22 补 ✓）：`text_encoder.TextEncoderConfig` 的注释**本来就写着**
        「应与 `DiTConfig.text_dim` 一致 ✓」✓ —— 但此前**只在编码之后**由形状错兜住 ✗（报的是
        「文本编码器输出应为 (1,L,5376) ✓，收到 (1,L,64) ✗」，**指不到**"是把 TE 配错了" ✓✗）。
        两个来源：主干那侧是**权重里的事实** ✓（`condition_proj` / `token_refiner` 的形状推出来 ✓）；
        TE 那侧是**配置** ✓。
        ⚠️ **不给** TE 配置时本仓会**自动对齐** ✓（`_configured_text_dim` ✓ —— 连"两种 config 形态都认"
        这条都被踩出来过 ✓）；但**显式给**的时候没人核 ✗✗，而真权重到手时那正是常见用法 ✓。
        ⚠️ 挂载**顺序**两向都守 ✓：先挂 TE 后装权重（`load_weights` 里再查一次 ✓）/ 先装权重后挂 ✓。
        """
        expected = self._configured_text_dim(0)
        config = candidate if candidate is not None else self._textConfig   # 候选优先 ✓（先核后改用 ✓）
        actual = int(getattr(config, "output_dim", 0) or 0) if config is not None else 0
        if not expected or not actual:
            return None                     # 主干还没装 / 没挂 TE ⇒ 没有可比的事实 ✓（不是漏 ✓）
        record = {"field": "output_dim", "expected": expected, "textEncoder": actual,
                  "agrees": actual == expected}
        if record["agrees"] or not raise_on_mismatch:
            return record
        raise TorchBackendUnavailable(
            f"文本编码器的 `output_dim` 与主干要的 `text_dim` 对不上 ✗：主干要 {expected} ✓"
            f"（来自**权重**：`condition_proj` / `token_refiner` 的形状 ✓），"
            f"而 TE 配置自述 {actual} ✗　⇒ ⚠️ 大概率是**显式给的 TE 配置**没跟上权重 ✓"
            f"（本仓**不显式给**时会自动按主干对齐 ✓ —— `_configured_text_dim` ✓）。"
            f"⚠️ **不静默取其中一个** ✗ —— 取错的一方会让 `condition_proj` 收到错的宽度 ✓✗"
            f"（那要到**编码之后**才报形状错 ✗，而且看不出是把 TE 配错了 ✓✗）。", reason="pending")

    def _trunk_latent_dims(self) -> tuple[int, int] | None:
        """主干自述的潜通道数 ✓ ``(视频, 音频)`` —— 不是 H3 形态 / 还没推出来 ⇒ ``None`` ✓。"""
        if getattr(self, "_form", None) != self.H3_FORM_NAME:
            return None
        config = self._config if isinstance(self._config, dict) else {}
        video = int(config.get("latents_dim") or 0)
        audio = int(config.get("audio_latents_dim") or 0)
        return (video, audio) if video and audio else None

    def _check_attached_vaes(self, *, video: Any = None, audio: Any = None,
                             raise_on_mismatch: bool = True) -> list[dict[str, Any]]:
        """⭐⭐ **跨来源校验**：**挂上来的 VAE 必须与后端算潜尺寸时用的那一套事实一致** ✓。

        为什么要专门守 ✗（2026-09-22 补 ✓）：下面每一条**本来是同一个事实** ✓，但各自有**两个来源** ✗✗，
        而且**只在解码时才由形状错兜住** ✗（报出来是「潜变量形状应为 (B,24,…)」✓✗ —— 指不到真因 ✓）：

        * ``视频 latent_channels`` ↔ 主干推的 ``latents_dim`` ✓ —— 真因往往是**显式 `patch_size` 与权重
          不符** ✗✗：`latents_dim` 是从「`video_patch_proj.weight` 列数 ÷ **patch 乘积**」反推的 ✓
          ⇒ patch 给错但**能整除**时得出的 `latents_dim` 也是错的 ✓✗，而两边形状**都还自洽**
          ⇒ 装得进去 ✓✗（**画面不对且不报错** ✗✗）。
          （依据就在本仓：`vae.py` 的事实表写着「24 个均值/标准差 ⇒ 潜通道 24 ✓（与
          `dit.H3_SHAPE_FACTS` 的 `latents_dim` 一致 ✓）」✓）
        * ``视频 spatial_scale`` ↔ **H3 事实** `vaeScale` ✓（双流造潜变量 / 首帧编码都按它算 ✓）
          —— 挂错族的 VAE ⇒ **解码出来的画面尺寸与请求的不是一回事** ✓✗（不报错 ✗）。
        * ``音频 latent_channels / stereo_channels / latents_per_second`` ↔ 主干推的 `audio_latents_dim`
          ✓ / `geometry.AUDIO_LATENT_CHANNELS` ✓ / `geometry.AUDIO_LATENT_HZ` ✓ —— ⚠️ 帧率那条最阴 ✗✗：
          潜帧数按 **40 Hz** 算 ✓、解码却按 **VAE 自己的 hop** 展开 ✓ ⇒ 出来的 wav **时长就是错的** ✓✗。
        * DiT 形态（非 H3）只核 ``视频 spatial_scale`` ↔ `config.vae_scale` ✓（H3 那几条不适用 ✗）。

        ⚠️ 仍是**两个来源相互印证** ✗，**不是权重事实** ✗：两边一起错仍会漏 ✓ ⇒ 真权重到手后要按元数据
        再核一次 ✓（见 `PENDING_PARTS` ✓）。
        ⚠️ 挂载**顺序**两向都守 ✓：先挂 VAE 后装权重（`load_weights` 里再查一次 ✓）/ 先装权重后挂 ✓。
        """
        h3_dims = self._trunk_latent_dims()
        video_channels, audio_channels = h3_dims if h3_dims else (0, 0)
        video_scale = int(vae_mod.H3_VIDEO_VAE_FACTS["vaeScale"])
        #: `(流, 自述字段, 期望值, 期望值的来源, 对不上时的「大概率是」✗)`
        pairs: list[tuple[str, str, int, str, str]] = []
        # ⚠️ 判据要连**候选**一起看 ✗（2026-09-22 自检当场抓到 ✓✗：改成"先核后改"后只按 `self._vaeConfig`
        #    判 ⇒ 从没挂过 VAE 的后端**根本不核候选** ✓✗ ⇒ 错的候选照样挂上去 ✗✗）。
        if video is not None or self._vaeConfig is not None:
            if h3_dims:
                pairs.append((
                    "视频", "latent_channels", video_channels,
                    "主干推断：`video_patch_proj.weight` 列数 ÷ patch 乘积 ✓（patch 由供方自述 ✗）",
                    f"**显式 `patch_size` 与权重不符** ✗（当前 patch_size="
                    f"{(self._config or {}).get('patch_size')} ✓ —— 给错但**能整除**时会推出错的 "
                    f"`latents_dim` ✓✗）；也可能是视频 VAE 自己的配置给错 ✓"))
                pairs.append((
                    "视频", "spatial_scale", video_scale,
                    f"**H3 事实** `vae.H3_VIDEO_VAE_FACTS.vaeScale={video_scale}` ✓"
                    f"（`init_dual_latents` 与首帧编码都按它换算 ✓）",
                    "**挂上来的不是 H3 那一族 VAE** ✗✗（倍率不同 ⇒ 潜尺寸与像素尺寸两边各算各的 ✓✗"
                    "⇒ 解码出的画面尺寸与请求的**不是一回事** ✓✗，而且不报错 ✓✗）"))
            else:
                dit_scale = int(getattr(self._config, "vae_scale", 0) or 0)
                if dit_scale:
                    pairs.append((
                        "视频", "spatial_scale", dit_scale,
                        "DiT 配置里的 `vae_scale` ✓（装权重时读出来的 ✓ —— 0 = 未给 ⇒ 不核 ✓）",
                        "**VAE 与 DiT 配置的倍率对不上** ✗（潜变量 ↔ 像素的换算两边不同 ✓✗）"))
        if (audio is not None or self._audioVaeConfig is not None) and h3_dims:
            pairs.append((
                "音频", "latent_channels", audio_channels,
                "主干推断：`audio_patch_proj.weight` 列数 ✓（音频那侧**不乘 patch** ✓）",
                "**音频 VAE 的配置给错** ✓（音频那侧不乘 patch ✗ ⇒ 与 `patch_size` 无关 ✓）"))
            pairs.append((
                "音频", "stereo_channels", int(geometry_mod.AUDIO_LATENT_CHANNELS),
                f"`geometry.AUDIO_LATENT_CHANNELS={geometry_mod.AUDIO_LATENT_CHANNELS}` ✓"
                f"（`init_dual_latents` 按它造立体声那一维 ✓）",
                "**音频 VAE 的声道数与几何事实不符** ✗（造出的潜变量与解码端要的对不上 ✓✗）"))
            pairs.append((
                "音频", "latents_per_second", int(geometry_mod.AUDIO_LATENT_HZ),
                f"`geometry.AUDIO_LATENT_HZ={geometry_mod.AUDIO_LATENT_HZ}` ✓（潜帧数按它算 ✓）",
                "**音频 VAE 的帧率与几何事实不符** ✗✗（潜帧数按 40 Hz 算 ✓ 而解码按它自己的 hop 展开 ✓"
                "⇒ 出来的 wav **时长就是错的** ✓✗，而且不报错 ✓✗）"))
        # ⚠️ 允许传**候选** ✓（`attach_vae(video=…)` 先核后改用 ✓）：候选优先 ✓，没传就用**已挂的** ✓
        configs: dict[str, Any] = {"视频": video if video is not None else self._vaeConfig,
                                   "音频": audio if audio is not None else self._audioVaeConfig}
        records: list[dict[str, Any]] = []
        for label, field, want, source, likely in pairs:
            raw = getattr(configs[label], field, None)
            got = int(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else raw
            record = {"stream": label, "field": field, "expected": want, "vae": got}
            if got == want:
                records.append({**record, "agrees": True})
                continue
            if not raise_on_mismatch:      # `describe()` 用 ✓ —— 它**任何时候都得能调** ✗（不许抛 ✗）
                records.append({**record, "agrees": False, "likely": likely})
                continue
            raise TorchBackendUnavailable(
                f"{label} VAE 的 `{field}` 与**后端用的那一套事实对不上** ✗：期望 {want} ✓"
                f"（来自 {source}），它自述 {got} ✗　⇒ ⚠️ 大概率是{likely}。"
                f"⚠️ **不静默取其中一个** ✗ —— 取错的一方会让潜尺寸 / 行打包 / 解码各按各的走 ✓✗"
                f"（形状自洽、装得进去、**出来不对且不报错** ✗✗）。", reason="pending")
        return records

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
        # ⚠️ 解码同样吃**调用方给的**潜变量（真链路里来自 `init_latents` ✓，自检里常是自己造的 ✓）
        #    —— VAE 在 `self._device` 上 ✓ ⇒ 混设备必炸 ✗（报错又是那串 `aten::` ✗）⇒ 先点破 ✓。
        self._require_own_device("decode", latents=latents)
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
        # ⚠️ 这条入口**刻意不做**设备校验 ✗（与上面五个入口**不同** ✓ —— 是查清后的判断 ✓ 不是漏 ✓）：
        #    `frames` 是 :meth:`decode` 的产物 ✓，而**落盘边界自己在收口** ✓✗ ——
        #    `media.write_video` / `write_wav` 结尾都 ``.to("cpu", …)`` ✓
        #    （见 `media.py` 的 ``scaled * 255`` 与 ``pcm`` 两处 ✓）⇒ **cpu 上解出来的帧照样能落盘** ✓。
        #    在这里加校验会把那条正当用法**拒掉** ✗✗（自检就常在 cpu 上解码 ✓），
        #    而它本来完全跑得通 ✓ ⇒ 「功能要全」= **逐个入口查清并写下结论** ✓，不是一律加 ✗。
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
