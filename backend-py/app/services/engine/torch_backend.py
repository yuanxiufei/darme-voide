"""**torch 后端** —— 真张量实现（2026-09-17；2026-10-29 落定：``.venv`` = torch 2.14.1+cu130、CUDA 可用 ✓）。

## 它现在是**真**的哪一半（务必分清 ✗）

| 维度 | 现状 |
|---|---|
**张量本身** | ✅ **真 torch 张量** ✓（`realTensors=True` ✓）—— 不再用 `dryrun` 那个 12 行 `list[float]` 假张量 ✓ |
**随机与复现** | ✅ 真 `torch.Generator().manual_seed(seed)` + `torch.randn` ✓ ⇒ 同种子**逐位可复现** ✓ |
**采样/引导/首帧数学** | ✅ 全部跑在真张量上 ✓（与 `dryrun` 共用同一份 `sampler`/`guidance`/`conditioning` ✓） |
**模型前向（DiT/TE/VAE）** | ✅ **五族真权重都真跑通过** ✓（2026-11-01 ✓：Z-Image / Krea2 / Qwen-Image 2.1（含 2512）/ Flux / H3 双流 ✓，各带端到端自检 ✓）—— ⚠️ **没装权重**时前向仍是**占位** ✓（如实标 ✓ 不冒充 ✗） |
**产物** | ✅ **真 PNG ✓ / 真 mp4 + wav ✓ 都已实测落盘** ✓（`engine_dual_stream_test` 等 ✓）—— ⚠️ 没装真权重时仍是合成产物 ✗ |

⇒ 两个标志**同时**给出，不许混 ✗：`realTensors=True`（张量真 ✓）+ `synthetic`（**按已装件算** ✓ ——
没装权重 / VAE 是参考实现 ⇒ `True` ✗；DiT ＋ 真权重 VAE 都在 ⇒ `False` ✓，见 :meth:`_picture_is_real` ✓）。
UI/接口据此可以显示「真张量 ✓ / 真模型 ✗」而不至于误导 ✓。

## 关于 CPU / CUDA（⚠️ 2026-10-29 实测落定）

**别在代码或文档里写死"本机没有 N 卡"** ✗✗ —— 本仓换过机器 ✓：笔记本时代是 Iris Xe 集显 ✓（CPU 轮子 ✓），
现在跑在 **A5000 工作站**上 ✓。2026-10-29 实测：``backend-py/.venv`` 装 **torch 2.14.1+cu130** ✓
（``cuda_built=True`` ✓、``torch.version.cuda=13.0`` ✓、``cuda.is_available()=True`` ✓、bf16 可用 ✓、
4096³ bf16 矩阵乘实测 91.7 TFLOP/s ✓、可用显存 21.28 GiB ✓、跑 5790 的后端已在该环境上 ✓）。
上游事实：**torch 2.15.0 在 cu126/cu128/cu129/cu130 全平台（含 Linux）都没有轮子** ✓，最高仅 2.14.1 ✓
⇒ 要用 CUDA 就停在 2.14.1，不是本机特例 ✗。旧的 ``2.15.0+cpu``（影子 .venv）与全局 Python 的
``2.13.0+cpu`` 都是 CPU-only ✗（前者从未被生产加载 ✓）。
⇒ 设备一律**实测** ✓（``_detect_device()`` / ``engine_readiness.device_facts()`` ✓，
`/engine/runtime` 的 ``device`` + ``deviceSource`` 也是它 ✓）；代码**一行都不用改** ✓。

## 依赖闸门（保留 ✓）

模块**不 import torch** ✓（懒加载 ✓）⇒ 没装也能 :meth:`TorchBackend.describe` ✓。
不可用时 ``reason`` 分两类 ✗：``deps``（去装包 ✓）／``pending``（依赖齐了但**这部分实现待写** ✓）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, NoReturn

from app.core import cpu_budget

from . import audio_vae as audio_vae_mod
from . import device_policy
from . import dit as dit_mod
from . import dit_build
from . import foresight as foresight_mod
from . import geometry as geometry_mod
from . import host_cache as host_cache_mod
from . import inventory as inv
from . import keyframes as keyframes_mod
from . import loader
from . import media as media_mod
from . import output_paths
from . import quant as quant_mod
from . import quant_nvfp4 as nvfp4_mod
from . import residency as residency_mod
from . import scheduler as scheduler_mod
from . import safetensors as st
from . import text_encoder as te_mod
from . import vae as vae_mod
from . import weights as weights_mod

__all__ = [
    "CODE_GAPS",
    "DEPENDENCIES",
    "IMPLEMENTATION_HISTORY",
    "TorchBackend",
    "TorchBackendUnavailable",
    "VAE_TILING_DEFAULTS",
    "dependency_status",
    "int4_scale_plan",
    "normalize_vae_tiling",
    "part_need_gib",
    "resident_bytes_of",
    "resident_need_gib",
    "torch_available",
    "torch_facts",
]


def _residency_mode(value: Any) -> str:
    """``residency`` 归一 ✓（``off`` / ``split`` / ``park`` ✓ —— 清单只有 `residency.MODES` 一份 ✓）。

    ⚠️ 不认的取值**当场炸** ✗✗：静默回落到 ``off`` 的后果是「用户以为开了联合驻留、实际整份在内存里爬」✓✗
    —— 与 `int8_offload` 那条"配置写了但没生效"是同一个病 ✓（不生效可以 ✓，不说不行 ✗）。
    """
    mode = str(value if value is not None else "off").strip().lower()
    if mode not in residency_mod.MODES:
        raise ValueError(f"residency 只认 {' / '.join(residency_mod.MODES)}，拿到 {value!r}")
    return mode


def _residency_kind(what: Any) -> str | None:
    """``what``（**报告名** ✓）⇒ 余量档的**件类型**（``"dit"`` / ``"te"`` / ``"vae"`` ✓ / ``None`` = 认不出 ✓）。

    ## 为什么这一层不能省 ✗✗（2026-11-01 实测）

    `_place_with_residency` 的 ``what`` 是**给报告读者看的** ✓：DiT 那两族各写各的名字
    （``"dit"`` / ``"h3"`` ✓），文本塔写 ``"te"`` / ``"h3-te"`` ✓。而余量只认三档 ✓
    （`residency.RESERVE_BY_KIND` ✓）。**直接拿 ``what`` 去查表** ⇒ ``"h3-te"`` 查不到 ⇒
    退回 25% ✓✗ ⇒ 改了半天，恰恰是 H3 那件 **25.44 GiB 的文字塔**（全仓最需要这条改动的一件 ✓）
    一点没吃到 ✓✗。这不是猜的：仓库里 grep ``_place_int8_module`` 的 ``what=`` 只有
    ``"h3"`` / ``"h3-te"`` 两个值 ✓。

    ## 归一的取舍

    * ``*-te`` / ``te`` / ``text*`` ⇒ ``te`` ✓（**先判** ✗：``"h3-te"`` 也含 ``h3`` ✓，
      顺序颠倒会把文本塔判成 DiT ✓✗ —— 余量从 5% 变 25%，白扔 4 GiB ✓）
    * 含 ``vae`` ⇒ ``vae`` ✓（``"h3-audio-vae"`` 这类 ✓）
    * ``dit`` / ``h3*`` ⇒ ``dit`` ✓
    * 其余 ⇒ ``None`` ✓（认不出就往保守侧倒 ✓，见 `reserve_for` ✓）
    """
    key = str(what if what is not None else "").strip().lower()
    if not key:
        return None
    if key.endswith("-te") or key in ("te", "text") or key.startswith("text"):
        return "te"
    if "vae" in key:
        return "vae"
    if key in ("dit", "h3") or key.startswith("h3"):
        return "dit"
    return None


#: ⭐ ``what``（**报告名**）⇒ ``part``（**分时部件名**）✓✗ —— 两套名字**从来不是一套** ✗✗。
#:
#: ## 为什么必须有一张表（2026-11-07 真机实测 ✓）
#:
#: 摆位置时用的名字是"给报告读者看的"（``"dit"`` / ``"h3"`` / ``"te"`` / ``"h3-te"`` ✓，见
#: :func:`_residency_kind` 的说明 ✓），而 `release_part` 的入参是**分时部件**
#: （``"dit"`` / ``"text"`` / ``"vae"`` / ``"vision"`` ✓）。原先 `release_part` 只做**同一性**判断
#: ``holder.module is self._textEncoder`` ✓✗ —— 于是：
#:
#: * **文本塔**（`attach_krea2_te` ✓）摆的是 ``module.tower`` ✓、存的是外壳 ``module`` ✓ ⇒
#:   两者**不是同一个对象** ⇒ 把手永不摘 ✓✗ ⇒ `residency.Handle` 的钩子环攥着每一层 ✓
#:   ⇒ `gc.collect()` 收不掉 ⇒ 卡上那份**一字节不退** ✓✗（假 8 GiB 卡实测：退 text 前后
#:   ``torch.cuda.memory_allocated`` 都是 6.41 GiB ✓）⇒ 下一拍 `decide_for` 读到"没余量" ✓
#:   ⇒ 整件转 CPU ⇒ 宿主不够 ⇒ 报「要整份放进内存」✓✗（**看着像内存不够，其实是卡没退** ✓✗）。
#: * 记账那本也一样 ✓：``reserve`` 用的是 ``what="te"`` ✓，而 ``release("text")`` 找不到这个键 ✓✗
#:   ⇒ ``residencyTotal`` 只涨不退 ✓。
#:
#: ⇒ 摆放时**记下用的是哪个名字** ✓，退件时按**部件**认 ✓，两边都退 ✓。
_WHAT_TO_PART: dict[str, str] = {
    "dit": "dit", "h3": "dit",            #: 两族 DiT ✓
    "te": "text", "h3-te": "text",        #: 各座文本塔 ✓（含 Krea2 / H3 / Music3 ✓）
}

#: 层预取上卡的槽数开关 ✓✗（2026-11-02 加 ✓）。**默认 0 ＝ 不存在这一级** ✓
#: （与 `Strata` 上游的 ``STRATA_FS_SLOTS > 0`` 同一条纪律 ✓）：它花的是一整层显存 ✓，
#: 而"该不该多占这一层"是**策略**不是正确性 ✓ ⇒ 显式开关 ✓，不设时引擎一字不变 ✓。
FORESIGHT_SLOTS_ENV = "DARME_FORESIGHT_SLOTS"

#: 默认槽数 ✓（``0`` ＝ 关 ✓）。**为什么"往上一层看"要 2 个槽** ✗✗：一个槽正在被这一层用 ✓，
#: 另一层要拷就得另有一块地 ✓ —— 只有 1 个槽也能跑 ✓，但每层都是"先拷完再算" ✓
#: （``busy`` 数涨满 ✓，正是"没有重叠"的证据 ✓）。
FORESIGHT_SLOTS_DEFAULT = 0


def _foresight_slots(raw: Any = None, *, default: int = FORESIGHT_SLOTS_DEFAULT) -> int:
    """槽数归一 ✓（读不懂就用默认 ✓、负数当 0 ✓）—— 不为一个限额把引擎起不来 ✓✗。"""
    if raw is None:
        raw = os.environ.get(FORESIGHT_SLOTS_ENV)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return max(0, int(default))
    return max(0, value)


def _foresight_of(mode: str) -> Any:
    """按环境变量给出**层预取**的配置 ✓ ⇒ :class:`foresight.Foresight` 或 ``None`` ✓。

    ⚠️ 它**不吃机器档** ✗✗（2026-11-07 明确 ✓）：引擎侧默认槽数是
    :data:`FORESIGHT_SLOTS_DEFAULT`＝**0（关 ✓，不预取不花显存 ✓）** ⇒ 低配档**无需**再压 ✓；
    而唯一能把它打开的是显式环境变量 ✓ —— 那条指令必须赢过档位 ✓✗，否则用户在 8 GB 机上设的
    ``DARME_FORESIGHT_SLOTS=0`` 会被"档位说 1"静默顶掉 ✓✗（多占的那一层显存正是他要避开的 ✓）。
    详见 :data:`profile.NOT_WIRED` ✓。

    ``None``（**没开** ✓）与"开了但这次没等到落地"（``hits=0`` ✓）在报告里必须分得开 ✗✗ ——
    前者整个 ``foresight`` 字段是 ``None`` ✓，后者字段在、``yieldLayers=0`` ✓。
    """
    slots = _foresight_slots()
    if slots <= 0:
        return None
    conf = foresight_mod.Foresight(slots=slots)
    if mode != "split":
        conf.slots = 0
        conf.disabledReason = (f"residency={mode} ⇒ 层预取不启用 ✓"
                              "（park 本来就逐层搬权重 ✓、off 没有卡 ✓）")
    return conf


def _residency_budget_mib(value: Any) -> float | None:
    """显存预算归一 ✓（MiB ✓；None ⇒ 按当时空闲显存现算 ✓；负数/NaN **当场炸** ✗✗）。"""
    if value is None:
        return None
    try:
        mib = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"residency_budget_mib 要是数字（MiB），拿到 {value!r}") from exc
    if not math.isfinite(mib) or mib < 0:
        raise ValueError(f"residency_budget_mib 要是 ≥0 的有限数，拿到 {value!r}")
    return mib


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
    #: ⭐ ``True`` ⇒ **可选**（缺了也**不阻断** ✓）。判据两条任一即可 ✓：
    #: ① 本仓有等价的自研实现 ✓（如自研 BPE ✓）；② 该包只是**增强/互校**通道 ✓（缺了主干照跑 ✓）。
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
    # ⚠️ 下面三项是 2026-11-07 补登记的 ✓ —— 此前**一条都不在册** ✗：换一台机器装引擎时，没人知道
    #    还差它们 ✓✗（症状是「某个族的提示词/分词悄悄走另一条路」，不报错 ✓）。三条都是**可选** ✓。
    _Dependency("jinja2", "jinja2",
                "渲染权重自带的 chat 骨架（缺它 ⇒ 提示词按裸拼接走 ✓，不阻断 ✓）", 5,
                optional=True),
    _Dependency("sentencepiece", "sentencepiece",
                "UMT5 分词的真值路（本仓另有纯 Python proto 解析 ✓ `parse_sp_model` ✓）", 5,
                optional=True),
    _Dependency("tokenizers", "tokenizers",
                "词表**互校**的参考侧（缺它 ⇒ 只剩自研 BPE ✓，主干不受影响 ✓）", 10,
                optional=True),
)

#: ⚠️ **已闭合缺口的沿革** ✓（人读 ✓）—— ⚠️ **不再进接口** ✗✗（2026-11-01 起 `describe()` 只报
#: :data:`CODE_GAPS` ＋ 运行期实况 ✓）：这 75 行是"当时缺什么、后来怎么补的" ✓，**不是**今天的缺口 ✗。
#: 把它当 `pendingParts` 报出去的后果实证过 ✗：装了整链照样说「还不能出片」✓✗（见 `describe()` 那段 ✓）。
#: ⚠️ 2026-09-20 **更正**：上一版清单**过期了** ✗ —— 它写着「文本编码器前向 ✗」「VAE 解码与出片落盘 ✗」，
#: 但这些**机制都已经实现** ✓（`encode_text` ✓ / `attach_text_encoder` ✓ / `decode` ✓ /
#: `attach_vae` ✓ / `write` ✓，且 `engine_io_test` / `engine_text_test` 已跑通
#: 「TE → DiT → VAE → **真 mp4**」✓）。⇒ **真正缺的只有「真权重 + 真配置」** ✓：
#: 机制齐了、权重没到 ✓ ⇒ `canGenerate=False` **依然正确** ✓（只是别把原因归到机制上 ✗）。
IMPLEMENTATION_HISTORY: tuple[str, ...] = (
    "✅ **低精度权重能真装了** ✓（2026-09-22 ✓ `engine/quant.py` ✓）：fp8/int8 先按配套 scale **反量化** ✓"
    "（布局**按形状**判 ✓ per-tensor / per-row / per-col / 逐元素 ✓；`*_scale_inv` 走除 ✓）"
    "⇒ 再归一 dtype ✓ —— ⚠️ 此前是**直接 `to(bf16)`** ✗ ⇒ 尺度丢掉但**不报错** ✓✗（19.53 GiB 的主权重"
    "就是 fp8 ✓）。⚠️ 判不出来（块量化 / 分组量化 / 缺 scale / scale 含 0 ✓）⇒ **中止装载** ✓ 不按猜的算 ✗。",
    "⚠️ **2026-10-30 实测更正**：H3 真权重**已在盘上** ✓ —— 原句「H3 真权重未下载（19.53 GiB）✗」"
    "**已失真** ✗（会让人去下 3 份共 39.55 GiB 里已经齐了的文件 ✓✗）。实测位置 "
    "共享池里的 `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` ✓"
    "（19.53 GiB ✓，同目录另有 ref2va / Singularity 变体 ✓），且 "
    "`h3_readiness.py --weights <该文件>` 的**真权重预检已通过** ✓：932 张量 ✓ 键名核对 ✓ "
    "结构推导 hidden=5376 / layers=50 / heads=56 / modalities=3 / banks=1 ✓（变体 curve ✓ refiner=2 ✓）。"
    "⇒ 现在**真正还差的只有一步：真权重端到端出片** ✗（合成权重那条 `engine_dual_stream_test` 已出真 mp4+wav ✓，"
    "真权重只走到「装得上、结构对」✓，**没跑过 `run_sync`** ✗ ⇒ 出片效果/显存/耗时全是未知 ✓✗）。"
    "⚠️ 别把「预检通过」读成「能出片」✗：预检只看键名与形状，不看数值是否 sane ✓。"
    "词表也齐 ✓ 但**只有一份对**：H3 的 TE 是 Qwen3-VL-32B ⇒ 必须用 `qwen25_tokenizer`（151643 ✓ "
    "与 ComfyUI `qwen3vl.py` 的 `Qwen3VLSDTokenizer` 同一份 ✓，现落在**模型池** "
    "`<模型根>/tokenizers/qwen3vl_32b` ✓ —— 原先只在开发期脚手架里 ✓✗，已搬出来 ✓）；"
    "⚠️ `llama_tokenizer`（vocab=128000）**是 Llama-3 的词表** ✗ —— 它装得上、`describe()` 也不报错 ✗，"
    "但 token id 与 Qwen 词表**整体错位** ⇒ 出片与提示词无关 ✓✗（2026-10-30 实测踩过 ✓）。",
    "✅ **词表切法已互校** ✓（2026-10-30 ✓）：自研 BPE 与 `Qwen2Tokenizer` **5 例逐 token 一致** ✓"
    "（含中英混排 ✓）—— `h3_readiness` 现在直接报「互校：比了 5 例，逐 token 一致 ✓」。"
    "⚠️ 此前长期报「没比成」**不是自研切错了** ✗，而是参考侧**造不出来** ✓✗："
    "`tokenizer_hub._build_reference_encoder_from_path` 当时只认 `tokenizer.json` ✗ ⇒ "
    "只有 `vocab.json`+`merges.txt` 的 Qwen 系发布包根本进不来 ✓（且 `Qwen2Tokenizer` 在"
    "transformers 5.x 的形参已从 `vocab_file=` 改成 `vocab=` ✓，传旧名会被 `**kwargs` 吞掉 ⇒ "
    "`vocab_size=1` 的假象 ✓）。两条都已修 ✓。",
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
    "—— 自检里已逐例同 id 比过 ✓）。⚠️ 原写的「仍缺的只是**词表文件本身** ✗、本仓不内置任何词表」"
    "**已部分失真** ✗（2026-10-30 ✓）：Qwen3-VL 该用的那一份（151643 词条 ✓ 与 ComfyUI `qwen3vl.py` "
    "同源 ✓）现落在**模型池** `<模型根>/tokenizers/qwen3vl_32b` ✓，"
    "且 `tokenizer_hub.find_h3_tokenizer()` 会**自动探测**它 ✓ ⇒ `h3_readiness.py --weights …` "
    "**不必再手传 `--tokenizer`** ✓（实测自动装上并互校 5 例逐 token 一致 ✓）。"
    "⚠️ 探测**只认 Qwen 系** ✗（看词表自带的 `tokenizer_class` ✓）：`llama_tokenizer`（128000 ✓ "
    "Llama-3 的词表 ✗）装得上、也不报错 ✗，但 token id 整体错位 ⇒ 出片与提示词无关 ✓✗。",
    "✅ **分词器总入口也升级了** ✓（2026-09-20 ✓ `engine/tokenizer_hub.py` ✓）：形态嗅探 ✓ → "
    "自研 BPE 优先 ✓ → `Unigram`/`WordPiece`/`Metaspace`（Llama/Qwen 系 ✓）**回退参考实现** ✓ "
    "+ 批量 ✓ LRU 缓存 ✓ 离线/关遥测/缓存目录 ✓ 运行期互校 ✓ "
    "⇒ 词表形态**不再限制能力** ✓（`attach_text_encoder(tokenizer_path=…)` 直接给路径 ✓）",
    "✅ **文本编码器 / VAE 的权重也都在盘上** ✓（2026-10-30 实测 ✓）："
    "TE `qwen3vl_32b_nvfp4_awq` ✓（351 层，能自动还原 ✓）、`h3_video_vae_fp16` 4.9 GiB ✓、"
    "`h3_audio_vae_fp32` 577 MiB ✓ —— 原写的「缺文本编码器/VAE 权重」**已失真** ✗。",
    "⭐ **真出片只剩「端到端 `run_sync` 跑一次」** ✗（2026-10-30 ✓）：机制 ✓ 权重 ✓ 词表 ✓ "
    "（自动探测 ✓ 互校通过 ✓）显存计划 ✓ 全齐、`ready=True` ✓，但**没跑过真出片** ✗ ⇒ "
    "画面效果 / 真显存峰值 / 耗时**全是未知** ✓✗。⚠️ 原写的「`canGenerate=False` = 机制齐、权重没到」"
    "**已失真** ✗（权重早到了 ✓）；合成权重那条 `engine_dual_stream_test` 出过真 mp4+wav ✓，"
    "但它验的是**管道** ✓、不是真权重 ✓。",
)

#: ⭐ **代码里现在还没写的**能力 ✓（`describe()["codeGaps"]` 报的就是它 ✓）—— 与 `pendingParts`
#: （"这台后端还没装" ✓，随 `attach_*` / `load_weights` 变 ✓）**必须分开** ✗✗：
#: 混成一格就看不出「该去修代码 ✓」还是「该去装权重 ✓」✓✗ —— 这正是 2026-11-01 之前那 75 行沿革
#: 被当成 `pendingParts` 报出去时的病根 ✓（装齐了也照样说"不能出片"✗）。
#: ⚠️ 事实口径：`python -m app.scripts.dit_families` 的「通路」栏 ✓（`有` / `无` ✓，2026-11-01 实测 ✓）。
#: ⚠️⚠️ **为什么不叫 `knownGaps`** ✗✗：`facade.capabilities()["knownGaps"]` 已经占了那个名字 ✓，
#: 而它是**环境**缺口（`[{id, severity, summary, facts, fix}]` ✓，见 `facade._known_gaps` ✓）——
#: 同名不同形、同在 `/engine/*` 名下 ✓✗：前端按 `gaps.map(g => g.summary)` 读环境缺口 ✓，
#: 拿到这里的一串字符串只会读出 `undefined` ✓；反过来按"有没有 codeGaps"判「要不要修代码」
#: 也会被环境缺口顶掉 ✓✗。⇒ 两个面各自用**自解释**的名字 ✓：环境缺口 `knownGaps`、
#: 代码缺口 `codeGaps` ✓（`engine_runtime_test` 的 ①⁵ 钉这一条 ✓）。
CODE_GAPS: tuple[str, ...] = (
    #: ⚠️ 2026-11-02 **已删**「Wan2.2 I2V 的 36 通道拼接未接 / umt5 无实现」✓ —— 三段都接上了 ✓：
    #: 首帧 20 通道条件在 `encode` 段用 VAE 编好 ✓（`_wan_i2v_condition`）、`i2v_concat` 拼 36 ✓、
    #: UMT5 文本塔 ✓（`umt5_te` ＋ `umt5_tokenizer` ✓，`_encode_wan22_text` 顺手把 pad 位置清零 ✓）、
    #: 双专家按 **σ 阈值**接力 ✓（`register_wan_experts` ✓）。
    #: ⚠️ 剩下的**不是代码缺口** ✗：单份 bf16 **26.62 GiB > 22.49 GiB 卡** 是硬件边界 ✓ ⇒ 走
    #: `split`/`cpu` 并按件分时 ✓（写进 `readiness` / 回执，不要退回 `codeGaps` 里报"没实现"✗）。
    "**Wan Animate 2** 判族落到 `wan`，但它是 **Wan2.1 式参考图通路**（`cross_attn.k_img/v_img` ＋ "
    "`img_emb.proj.*`，2263 键）⇒ `dit_wan22` 见到这些键**当场拒** ✓✗（别当成「能跑」✗：`dit_families` "
    "的「可建 3/3」只说明拓扑读得全 ✓，不说明建得出 ✓）",
    #: ⚠️ 2026-11-02 **已删**「SEEDVR2 无前向」✓ —— `dit_seedvr2` ＋ `vae_seedvr2` 已进 `dit_build` 与
    #: `stage="seedvr2"` 装配线 ✓ 且真 3B 权重出过片 ✓（唯一短板是解码慢，那是已知的性能事实 ✓
    #: 不是代码缺口 ✗）。
    #: ⭐ 2026-11-02 同日把「Music3 无前向」**改写成真话** ✓✗：骨架（`dit_music3` ✓）与音频 VAE
    #: 解码（`vae_music3` ✓）**都写完了** ✓，缺的是**把它们连起来** ✗：没有 `stage="music3"`
    #: 装配线、没有歌词/文本塔接线、没有滑窗采样循环 ✓ —— 写「无前向」会让人以为一行代码都不用写 ✓✗。
    "**Music3** 音频生成：DiT 骨架 ＋ 前向口径（σ 翻转取负 / RoPE split-half / 潜帧率 86.1328 Hz）"
    " ＋ DAV 解码都已写出并自检 ✓，但**没接进 `pipeline`** ✗：无 `stage=\"music3\"` 装配线、"
    "无歌词-文本塔接线、无 689/344 滑窗采样循环（TE 与音频 VAE 都在盘 ✓）",
    #: ⚠️ 2026-11-01 **已删**「Edit 2511 编辑通路未接」那条 ✓ —— 视觉塔 ⇒ `<|image_pad|>` 注入 ✓、
    #: 参考图 ⇒ `ref_latents` ✓ 两路都接上 ✓（守卫见 `engine_qwen_image_edit_test` ✓）。
    #: ⚠️ 同轮**也删了**「模板只有一个图槽 ⇒ 多参考图仍拒」✓ —— 多图走 Plus 口径 ✓
    #: （`Picture i: <视觉块>` 逐图一槽 ✓、`ref_latents` 一列 ✓，守卫见同一测试的 ④′/⑦′ ✓）。
    "GGUF **Q2_K / Q3_K** 反量化未实现 ✓（盘上无该档文件 ⇒ **有意不补** ✗，见 `gguf_dequant` ✓）",
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


def int8_scale_plan(info: Any, *, strip_prefix: str | None = None,
                    int4_stems: Any = None) -> dict[str, tuple[int, ...]]:
    """从权重头部挑出「该走 **int8 常驻**」的线性层 ✓ ⇒ ``{子模块名: scale 形状}`` ✓。

    判据只两条 ✓：键以 ``.weight`` 结尾 ✓、dtype 是 ``I8`` ✓；且**必须有配套 scale** ✓。

    ⚠️ **``int4_stems`` 必须给** ✗✗（2026-11-02 加 ✓）：4bit 打包权重**存成** ``I8`` 且**也带**
    per-row ``weight_scale`` ⇒ 两条判据**逐字同形** ✓✗ ⇒ 只看头部就会把「每字节两个 4bit」
    当成「per-row int8 常驻」✓，乘出来的数值全错、形状全对、一声不响 ✓✗（这正是 2026-10-30
    抓到的那个假绿 ✓）。4bit 与 int8 只能靠**文件自述**分（:func:`quant.int4_stems_of` ✓）⇒
    调用方把那份集合递进来，这里只负责跳过 ✓（不在这里读盘 ✗ —— 本函数的入参只有头部 ✓）。
    ⚠️ 缺 scale 的**不在这里补** ✗ —— 留给装载闸口报「缺 scale」✓（这里塞个假尺度只是把
    真问题藏起来 ✓✗）。``weight_scale`` 与 ``.scale`` 两种写法盘上都有 ✓。

    ⭐ ``strip_prefix`` ✓（2026-11-01 加 ✓，接 Krea2 时逼出来的 ✓）：有的导出把全部键套一层
    ``model.diffusion_model.`` ✓（见 :data:`_KREA2_PREFIXES` ✓），而 :func:`h3_form.upgrade_linear_to_int8`
    是**按模块名**查表 ✓ ⇒ 表里若还带着那层前缀，**一个名字都对不上** ✓✗：结果不是报错 ✗，
    而是「计划点名了、模块里还是普通 ``nn.Linear``」⇒ 权重按 bf16 反量化落盘 ✓✗
    （12.82B 参数 ⇒ 23.9 GiB 挤 22.49 GiB 卡 ⇒ 采样第 1 步 OOM ✓，报错指不到前缀 ✓）。
    ⚠️ **不在这里猜前缀** ✗：剥什么由调用方按 ``_krea2_stripped`` 的结果显式给 ✓。
    """
    plan: dict[str, tuple[int, ...]] = {}
    skip: frozenset[str] = frozenset(str(item) for item in (int4_stems or ()))
    for logical, tensor in info.tensors.items():
        if tensor.dtype != "I8" or not logical.endswith(".weight"):
            continue
        name = logical[: -len(".weight")]
        if name in skip:
            continue                       # 4bit 打包 ⇒ 不是 int8 常驻层 ✓（见 docstring ✓）
        scale = info.tensors.get(f"{name}.weight_scale") or info.tensors.get(f"{name}.scale")
        if scale is None:
            continue
        if strip_prefix and name.startswith(strip_prefix):
            name = name[len(strip_prefix):]
        plan[name] = tuple(int(dim) for dim in scale.shape)
    return plan


def int4_scale_plan(info: Any, *, strip_prefix: str | None = None,
                    int4_stems: Any = None) -> dict[str, tuple[int, ...]]:
    """从权重头部挑出「该走 **4bit 打包常驻**」的线性层 ✓ ⇒ ``{子模块名: scale 形状}`` ✓。

    判据**三条一起**（缺一条就拒绝，不猜 ✗）：

    1. 这一层**文件自述**是 4bit 打包 ✓（``int4_stems`` ＝ :func:`quant.int4_stems_of` 的结果 ✓）——
       ⚠️ 只看 dtype/scale 是不行的 ✗✗：打包权重存成 ``I8`` 且带 per-row ``weight_scale``，
       与 per-row int8 **逐字同形** ✓（2026-10-30 那个假绿 ✓）；
    2. 头部写得出 ``(rows, cols)`` 且 dtype 是单字节整型 ✓（:func:`quant.int4_shape_problem` ✓）
       —— 逻辑列数（打包列 ×2 ✓）必然是偶数 ✓，这条无需另判 ✓；
    3. 有配套 scale ✓（``weight_scale`` / ``scale`` 两种写法盘上都有 ✓ 缺失不在这里补 ✗）。

    ⚠️ 与 :func:`int8_scale_plan` **互斥** ✓：调用方必须把 ``int4_stems`` 同时递给两边
    （那边是"跳过" ✓、这边是"才要" ✓），否则同一层会被两个计划同时点名 ✓✗。
    """
    skip_free: frozenset[str] = frozenset(str(item) for item in (int4_stems or ()))
    plan: dict[str, tuple[int, ...]] = {}
    if not skip_free:
        return plan
    for logical, tensor in info.tensors.items():
        if not logical.endswith(".weight"):
            continue
        name = logical[: -len(".weight")]
        if name not in skip_free:
            continue
        if quant_mod.int4_shape_problem(name, tensor.dtype, tensor.shape) is not None:
            continue                       # 自称打包但存不了打包 ⇒ 不换壳 ✓（装载侧会拒 ✓）
        scale = info.tensors.get(f"{name}.weight_scale") or info.tensors.get(f"{name}.scale")
        if scale is None:
            continue
        if strip_prefix and name.startswith(strip_prefix):
            name = name[len(strip_prefix):]
        plan[name] = tuple(int(dim) for dim in scale.shape)
    return plan


def nvfp4_scale_plan(info: Any, *, strip_prefix: str | None = None,
                     packed_stems: Any = None) -> dict[str, tuple[tuple[int, ...], bool]]:
    """从权重头部挑出「该走 **nvfp4 打包常驻**」的线性层 ✓ ⇒ ``{子模块名: (块 scale 形状, 有没有 AWQ)}`` ✓。

    判据只认**文件自述 ＋ 三个配套张量齐全** ✓（缺一条就不点名 ✓，装载闸口会当场拒 ✓）：

    1. ``.weight`` 是 ``U8`` ✓ 且文件自述**每字节两个元素** ✓ ——
       ``packed_stems`` 递 :func:`quant.packed_weight_stems_of` ✓（⚠️ 不能只看 dtype ✗：同一份文件里
       ``embed_tokens`` 是 ``I8`` ✓、int4 文件也可能存成 ``U8`` ✓，只看 dtype 会误认 ✓✗）；
    2. ``weight_scale``（``F8_E4M3`` 块尺度 ✓，每 16 元素一个 ✓）在 ✓；
    3. ``weight_scale_2``（f32 **0 维**标量 ✓）在 ✓ —— ⭐ 这一条同时把 int4/w4a8 排除掉 ✓
       （它们只有 per-row/per-group scale、**没有**二级标量 ✓）；
    4. 可选 ``pre_quant_scale``（AWQ ✓，乘在**输入**上 ⇒ 前向里折 ✓，见 :func:`quant_nvfp4.fold_pre_quant_scale` ✓）。

    ⭐ 为什么单独一支、不塞进 :func:`int8_scale_plan` ✗✗（2026-11-09 ✓）：nvfp4 的**块尺度是一张
    带 padding 的二维表** ✓（``[rows, ceil4(in/16)]`` ✓，cuBLAS 128×4 swizzle 布局 ✓），而 int8 是
    **每行一个标量** ✓ —— 同一个 ``weight_scale`` 名字、两种形状、两种语义 ✓✗。混着认会把块尺度当成
    per-row 乘下去：形状全对、数值全错、一声不响 ✓✗。

    ⚠️ ``strip_prefix`` 同 :func:`int8_scale_plan` ✓（按**模块名**查表 ✓，前缀不对一个都点不上 ✓）。
    """
    plan: dict[str, tuple[tuple[int, ...], bool]] = {}
    stems = None if packed_stems is None else frozenset(str(item) for item in packed_stems)
    for logical, tensor in info.tensors.items():
        if not logical.endswith(".weight") or str(tensor.dtype) != "U8":
            continue
        stem = logical[: -len(".weight")]
        if stems is not None and stem not in stems:
            continue                       # 文件没自述是打包权重 ⇒ 不猜 ✓（装载侧会拒 ✓）
        scale = info.tensors.get(f"{stem}{nvfp4_mod.COMFY_SCALE_SUFFIX}")
        scale_2 = info.tensors.get(f"{stem}{nvfp4_mod.COMFY_SCALE_2_SUFFIX}")
        if scale is None or scale_2 is None:
            continue                       # 配套缺 ⇒ 不点名 ✓（让装载闸口报"缺哪一件" ✓）
        awq = f"{stem}{nvfp4_mod.PRE_QUANT_SCALE_SUFFIX}" in info.tensors
        name = stem[len(strip_prefix):] if strip_prefix and stem.startswith(strip_prefix) else stem
        plan[name] = (tuple(int(dim) for dim in scale.shape), awq)
    return plan


def resident_need_gib(path: str | Path, *, fallback: float,
                      strip_prefix: str | None = None) -> tuple[float, dict[str, Any]]:
    """**这一件权重**装起来会占多少显存 ✓ ⇒ ``(GiB, 凭据)`` ✓。

    ## 为什么一张表不够 ✗

    Qwen-Image 2512 那一族盘上**同时有两种文件** ✓：

    * ``qwen_image_2512_bf16.safetensors`` ＝ 38.05 GiB / **bf16** ✓ ⇒ 装起来就是 38.05 GiB ✓
      （:attr:`TorchBackend.QWEN_IMAGE_PART_NEED_GIB` 里那个数是**量出来的** ✓）；
    * ``qwen_image_edit_2511_int8_convrot.safetensors`` ＝ 19.09 GiB / **840 层 int8** ✓ ⇒
      走 `Int8Linear` 常驻后**仍然是 19.09 GiB** ✓（**不再**反量化成 bf16 ✓）。

    拿一张写死的 38.05 去判后者 ⇒ `decide()` 判 `split`（38.05 > 22.49 卡 ✓）或 `cpu` ✓✗ ——
    **实测就是这么发生的** ✓：`gpu 15.9 / cpu 22.15 GiB` ⇒ 58% 的权重在内存里逐层爬 ✓✗，
    而卡其实**整份装得下**（19.09 < 19.71 可用 ✓）。判据本身没错 ✗，错在**需求数** ✓。

    ## 判据（只认文件头部，毫秒级 ✓）

    文件里 `.weight` 是 ``I8`` 且有配套 scale ⇒ 这一层会**原样常驻** ✓ ⇒ 该件的驻留字节
    ＝**数据区字节** ✓（int8 权重 1 字节/参数 ＋ scale 自己也在文件里 ✓，一个 tensor 不多不少 ✓）。
    否则退回 ``fallback``（表里的**实测**值 ✓ —— 不在这里编一个 bf16 倍率 ✗：fp8_scaled 是 2×、
    bf16 是 1×，两者都叫"低精度文件" ✓✗）。

    ⚠️ 读不出头部 / 头部自述有问题 ⇒ **退回 fallback** ✓ 并把原因写进凭据 ✓（不猜 ✓）。
    """
    table = float(fallback)
    try:
        info = st.inspect(path)
    except Exception as err:  # noqa: BLE001 —— 读不到就按表，别把装配挡住
        return table, {"int8": None, "why": f"读不出头部 ⇒ 按表 {table:.2f} GiB（{err}）"}
    if not info.ok:
        return table, {"int8": None,
                       "why": f"头部自述有问题 ⇒ 按表 {table:.2f} GiB（{info.problems[:2]}）"}
    plan = int8_scale_plan(info, strip_prefix=strip_prefix)
    if not plan:
        return table, {"int8": False,
                       "why": f"文件侧没有 int8 层 ⇒ 按表 {table:.2f} GiB（反量化后的实测值 ✓）"}
    resident = info.total_tensor_bytes / device_policy.GIB
    return resident, {
        "int8": True, "layers": len(plan),
        "fileGiB": round(info.file_bytes / device_policy.GIB, 2),
        "why": (f"文件侧 {len(plan)} 层 int8 ⇒ 走 `Int8Linear` 常驻 ⇒ 驻留 ＝ 数据区 "
                f"{resident:.2f} GiB ✓（不再按 bf16 反量化；表里那 {table:.2f} GiB 是 bf16 文件的账 ✓）"),
    }


#: 走 **int8 常驻** 的入口 ✓（`dit_build._ADAPTERS` 的两个 ＋ H3 那条自带换壳的路 ✓）——
#: 只有这些入口才会把 ``I8`` 权重**原样**留在卡上 ✓；其余族一律反量化成 bf16 ⇒ 1 B/参 变 2 B/参 ✓。
_INT8_RESIDENT_ADAPTERS: frozenset[str] = frozenset({"int8", "krea2", "h3"})

#: 走 **4bit 常驻** 的入口 ✓（2026-11-02 补壳 ✓）—— 判据是「这条入口的换壳器会不会把 4bit 层
#: 换成 ``Int8Linear(packed4=True)``」✓：:func:`dit_build._shell_int8` 现在**同时**换 int8 与
#: 4bit 两种壳 ✓ ⇒ 用它做 shell 的两个 adapter（``int8`` / ``krea2``）都算 ✓。
#: ⚠️ ``h3`` **不在**集合里 ✗：H3 那条路（:func:`_prepare_h3_module` ✓）**没传** ``int4_stems``
#: ⇒ 4bit 文件在它手里仍按 int8 常驻（数值全错 ✗✗）—— 所以宁可让它按 2 倍记（保守 ✓），
#: 也不许在这里假装能跑 ✓（盘上 H3 没有 4bit 文件 ✓，这条只是不许写成通用假绿 ✗）。
_INT4_RESIDENT_ADAPTERS: frozenset[str] = frozenset({"int8", "krea2"})

#: 本仓**没有** int8 的**嵌入壳** ✓✗（常驻壳只有 `Int8Linear` / `NvFp4Linear` ✓，都是 ``nn.Linear``
#: 的替身 ✓；``nn.Embedding`` 没有对应物 ✓）⇒ 名字落在这些尾巴上的 int8 权重**一定**在装载时按
#: bf16 反量化 ✓（2 B/参 ✓）。
#: ⭐ 为什么必须单列（2026-11-09 实测 ✓）：nvfp4 那份 H3 文本塔里 ``model.embed_tokens`` 是
#: ``I8``＋``F32`` per-row scale ✓ ⇒ 被 :func:`int8_scale_plan` 认成"能常驻的 int8 层" ✓✗
#: ⇒ 账算 **0.72 GiB**、实际物化 **1.45 GiB** ✓ ⇒ 整份报 14.61 而卡上真占 **15.35 GiB** ✓✗
#: （0.74 GiB 的假余量，正是"判得下、装不上"的那一档 ✓）。
#: ⚠️ 这是**关于本仓壳集合的事实** ✓，不是按名字猜量化格式 ✗（格式仍只认文件自述 ✓）。
_INT8_EMBED_STEMS: tuple[str, ...] = ("embed_tokens", "lm_head", "token_embedding")

#: 存储侧**每参 1 字节**的 dtype ✓ ⇒ 物化成 bf16 之后每参数变 2 字节 ✓
#: ⚠️ 这是**按字面换算**（1 B → 2 B ✓），不是「低精度文件一律乘 2」那种猜 ✗：bf16/f16 文件是 1× ✓。
_ONE_BYTE_STORAGE: frozenset[str] = frozenset({
    "I8", "U8", "F8_E4M3", "F8_E5M2", "F8_E4M3FN", "F8_E5M2FN", "I4", "U4"})

#: 各 dtype 的**元素字节数** ✓（认不出 ⇒ 退回该张量的文件字节 ✓，不猜 ✗）
_DTYPE_BYTES: dict[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E4M3FN": 1, "F8_E5M2FN": 1, "I4": 1, "U4": 1,
}


def _materialized_bytes(entry: Any) -> int:
    """一个**非量化权重**张量物化之后占几字节 ✓（物化＝换成 bf16 ✓）。

    ⚠️ 只对「存储 1 B/参」的那几类乘 2 ✗（``I8``/``U8``/``F8_*``/``I4``/``U4`` ✓）——
    bf16/f16/f32 原样 ✓，``BOOL``（掩码）原样 ✓。认不出的 dtype 或形状与字节对不上 ⇒
    **退回该张量的文件字节** ✓（宁可保守，不编一个倍率 ✗）。
    """
    dtype = str(getattr(entry, "dtype", "") or "").upper()
    nbytes = int(getattr(entry, "nbytes", 0) or 0)
    per = _DTYPE_BYTES.get(dtype)
    if per is None:
        return nbytes
    count = int(getattr(entry, "element_count", 0) or 0)
    if count <= 0 or count * per != nbytes:
        return nbytes                     # 形状与字节对不上 ⇒ 按文件字节 ✓（不猜 ✗）
    if dtype in _ONE_BYTE_STORAGE:
        return count * 2
    return nbytes


def resident_bytes_of(info: Any, *, adapter: str | None,
                      specs: Any = None) -> tuple[int, dict[str, Any]]:
    """**这一份权重**装起来真占多少字节 ✓ ⇒ ``(字节, 逐类账)`` ✓。

    ## 三种层的算法（每一条都有据可查 ✓）

    1. **4bit 打包层**（文件自述 ``convrot_w4a4``/``int4`` ✓、存成 1 B/字节×2 参 ✓）：
       有 int4 常驻壳 ⇒ 打包字节 ✓；**现在没有** ⇒ 物化成 bf16 ＝ **逻辑参数 × 2 B** ✓
       （:func:`quant.logical_weight_shape` 把打包维 ×2 ✓ —— 不能拿打包形状去乘 ✗，那样少一半 ✓）。
    2. **int8 + 配套 scale 的 ``.weight``**：该装配线有常驻壳（``adapter`` ∈
       :data:`_INT8_RESIDENT_ADAPTERS` ✓）⇒ 文件字节（1 B/参 ✓）；没有 ⇒ **×2**（反量化 bf16 ✓）。
       例外：名字落在 :data:`_INT8_EMBED_STEMS` 上的（``embed_tokens`` / ``lm_head`` ✓）**一律 ×2**
       ✓✗ —— 本仓没有 int8 嵌入壳 ✓，量再"能常驻"也只能物化 ✓（14.61 ⇒ 15.33 GiB 那 0.72 GiB ✓）。
    3. **其余张量**（norm/embedding/scale/bias ✓）：按 dtype 的**物化**字节 ✓
       （见 :func:`_materialized_bytes` ✓）。

    4. **nvfp4 打包层**（``U8`` ＋ ``F8_E4M3`` 块尺度 ＋ f32 标量 ✓，2026-11-09 ✓）：
      **打包原样常驻** ✓（1 B/2参 ✓）—— 反量化挪到**前向逐层** ✓（见
      :class:`h3_form.NvFp4Linear` ✓），所以这里**不 ×2** ✓；它的三个配套张量
      （块尺度 / 标量 / AWQ ✓）也按文件字节算 ✓（块尺度是 ``F8`` ⇒ 走通用口径会被 ×2 ✗✗，
      14.61 GiB 那份会白多报 1.42 GiB ✓）。
      ⚠️ 这一条**不按 adapter 门控** ✗：能接住 ``uint8`` 常驻壳的只有 H3 文本塔那条路
      （:func:`attach_h3_text` ✓，按文件换壳 ✓）；别的装配线拿到这种文件会被装载闸口
      **当场拒**（目标参数是 bf16 ✓ ⇒ "不能原样写入" ✓），不会静默按 2 倍装 ✓ ⇒ 按 1 倍记账
      不会把哪次真跑算成"装得下" ✓。

    ⚠️ 与 :func:`resident_need_gib` 的分工 ✗：那个函数只认 int8（历史口径 ✓、Qwen-Image 那条线
    在用 ✓）；这个是**逐层分类**的精算 ✓，服务「同一族不同量化档」的其余五条线 ✓。
    """
    int4_stems = quant_mod.int4_stems_of(specs) if specs else frozenset()
    plan = int8_scale_plan(info, int4_stems=int4_stems)
    #: ⚠️ 4bit 的判据必须问**换壳器真正会换的那几张** ✗✗（:func:`int4_scale_plan` ✓）：
    #: 只问「文件自称 4bit 吗」⇒ 自称了却缺 scale / 存成 2D 以外的层会被当成能常驻 ✓✗
    #: （那种层换不了壳 ⇒ 装载时按 bf16 反量化 ⇒ 实际是 2 倍 ✓，账却按 1 B/2参 报 ⇒ 装到一半 OOM ✓）。
    int4_plan = int4_scale_plan(info, int4_stems=int4_stems)
    nv_stems = quant_mod.packed_weight_stems_of(specs) if specs else frozenset()
    nv_plan = nvfp4_scale_plan(info, packed_stems=nv_stems)
    #: 打包层的**配套张量**（块尺度 F8 ✓ / 标量 f32 ✓ / AWQ bf16 ✓）—— 它们跟着壳原样留着 ✓，
    #: 不能按"物化"算 ✓✗（块尺度是 ``F8`` ⇒ 通用口径会 ×2 ✓，那 1.42 GiB 是凭空多出来的 ✓）。
    nv_aux = {f"{stem}{suffix}" for stem in nv_plan
              for suffix in (nvfp4_mod.COMFY_SCALE_SUFFIX, nvfp4_mod.COMFY_SCALE_2_SUFFIX,
                             nvfp4_mod.PRE_QUANT_SCALE_SUFFIX)}
    keep_int8 = adapter in _INT8_RESIDENT_ADAPTERS
    keep_int4 = adapter in _INT4_RESIDENT_ADAPTERS
    total = 0
    int8_hit = 0
    int8_embed = 0
    int4_hit = 0
    nv_hit = 0
    for name, entry in info.tensors.items():
        if name.endswith(quant_mod.QUANT_META_SUFFIX):
            continue                       # 量化说明本身不是模型参数 ✓
        stem = name[: -len(".weight")] if name.endswith(".weight") else None
        dtype = str(getattr(entry, "dtype", "") or "").upper()
        if name in nv_aux:
            total += int(entry.nbytes)     # 打包层的配套张量原样常留 ✓
            continue
        if stem is not None and stem in nv_plan:
            nv_hit += 1
            total += int(entry.nbytes)     # 1 B/2参（打包维已经是半宽 ✓）—— 原样常驻 ✓
            continue
        if stem is not None and stem in int4_stems and quant_mod.is_int4_storage(dtype):
            #: ``packed_ok`` ＝ 这一层**真能**换成 4bit 常驻壳（有 scale ✓ 头部自洽 ✓）——
            #: 换不了 ⇒ 只能按 bf16 反量化 ⇒ **逻辑**参数 × 2 B ✓（拿打包形状乘会少一半 ✓✗）
            packed_ok = keep_int4 and (stem in int4_plan)
            int4_hit += 1
            logical = quant_mod.logical_weight_shape(entry.shape, packed=True)
            count = 1
            for dim in logical:
                count *= int(dim)
            total += int(entry.nbytes) if packed_ok else count * 2
            continue
        if stem is not None and keep_int8 and stem in plan:
            #: ⚠️ 嵌入/输出头**没有 int8 壳** ✓✗（2026-11-09 ✓）⇒ 必按 bf16 物化 ✓（见
            #: :data:`_INT8_EMBED_STEMS` ✓ 与 :func:`_materialized_bytes` 的 2 B/参 ✓）。
            if any(stem.endswith(token) for token in _INT8_EMBED_STEMS):
                int8_embed += 1
                total += _materialized_bytes(entry)
                continue
            int8_hit += 1
            total += int(entry.nbytes)     # 1 B/参 原样留 + scale 在别的张量里原样计 ✓
            continue
        total += _materialized_bytes(entry)
    return total, {"int8Layers": len(plan), "int8ResidentLayers": int8_hit,
                   #: int8 里**没有壳**的那几个（``embed_tokens`` ✓）—— 它们按 bf16 记 ✓，
                   #: 单列一格是为了让回执能解释「为什么帧数比文件大 0.7 GiB」✓
                   "int8EmbedLayers": int8_embed,
                   "int4Layers": int4_hit,                    "int4Resident": keep_int4,
                   #: nvfp4：``Layers`` ＝ 文件里**能接住**的打包层数 ✓（三件配套齐全 ✓）、
                   #: ``Resident`` ＝ 账上按"原样常驻"算的层数 ✓（当前两者恒等 ✓，分开是为了让
                   #: 将来"某族没有壳"的情形**算得出 2 倍** ✓✗）
                   "nvfp4Layers": len(nv_plan), "nvfp4Resident": nv_hit,
                   #: 真正会被换壳的层数 ✓ —— 没给 adapter / 该 adapter 没有 4bit 壳 ⇒ **0** ✓
                   #: （算成 len(int4_plan) 会让回执说「256 层走 4bit 壳」而实际按 bf16 装 ✓✗）
                   "int4ShellLayers": len(int4_plan) if keep_int4 else 0,
                   "adapter": adapter}


@dataclass(frozen=True)
class PartSpec:
    """一条装配线上**一件权重**的取数口径 ✓✗（2026-11-07 收口 ✓）。

    三格各答一件事，混在一起就会算错而不报错 ✗：

    * ``assume`` —— 表里那个数是按**哪一档文件**量的 ✓（:func:`part_need_gib` 的口径 ✓）；
    * ``adapter`` —— 这族有没有**常驻壳** ✓（`"krea2"` / `"h3"` / `"int8"` ⇒ int8 层原样留 ✓）；
    * ``measured`` —— 这件的数**按不按文件精算** ✓✗（``False`` ⇒ 只信表 ✓）。
    """
    assume: str | None = None
    adapter: str | None = None
    measured: bool = True


#: ⭐⭐ 「某档某件要多少 GiB」的**唯一出处** ✓✗（2026-11-07 ✓）。
#:
#: 为什么必须收成一张表 ✗✗：这 9 处口径原先**内联在 8 条装配线里** ✓ ——
#: 装配线要判"装不装得下" ✓、预检（`feasibility`）要**不装载就答同一个数** ✓✗。
#: 两处各写一份 ⇒ 迟早一处改了另一处没改 ✓：预检说「装得下」、真装却 OOM ✓✗，
#: 而报错落在**后装的那件**上，读起来像"VAE 坏了" ✗（这个坑本仓已经踩过两次 ✓：
#: Krea2 表值按 int8 量、Qwen-Image 2511 按 38.05 判 ✓）。
#:
#: ⚠️ 表值本身**不在这里** ✗：各族的数仍挂在 :class:`TorchBackend` 的 ``*_PART_NEED_GIB`` ✓
#: （与 2.1 的 :data:`PART_NEED_GIB` 并列、两族必须分开 ✓）。这里只登记**怎么取** ✓。
#:
#: ⚠️ 键是**stage**（不是族名 ✗）：``qwen_image``（2512/Edit）与 ``qwen_image21``（2.1）
#: 是两条线、两本账 ✓✗；``image`` 是 SDXL 那条（单 checkpoint、无表 ⇒ 退回文件字节 ✓）。
STAGE_PART_SPECS: dict[str, dict[str, PartSpec]] = {
    "h3": {"dit": PartSpec("int8", "h3"),          # 19.62 是按 int8 档量的 ✓
           "text": PartSpec("int8"),              # 25.44 是按 int8 档量的 ✓
           #: H3 的真 VAE 是 `keep_parts` 常驻件、由后端自己找文件 ⇒ 没有路径可算 ✓，只信表 5.2 ✓
           "vae": PartSpec(measured=False)},
    "zimage": {"dit": PartSpec("plain"), "text": PartSpec("plain"), "vae": PartSpec("plain")},
    "qwen_image21": {"dit": PartSpec("plain"),    # 13.26 是 bf16 那份量的 ✓
                     "text": PartSpec("int8"),    # 14.79 是 int8 语言塔反量化后量的 ✓
                     "vae": PartSpec("plain")},
    "krea2": {"dit": PartSpec("int8", "krea2"),   # 12.57 是按 224 层 Int8Linear 常驻量的 ✓
              "text": PartSpec("int8"),           # 7.5 是按 int8 量的 ✓
              "vae": PartSpec("plain")},
    "seedvr2": {"dit": PartSpec("plain"), "vae": PartSpec("plain")},
    "music3": {"dit": PartSpec("plain"),          # fp16 原样驻留 ✓
               "text": PartSpec("int8"),          # 表值是按 int8_convrot 常驻量的 ✓
               "vae": PartSpec("plain")},
    "wan22": {"dit": PartSpec("plain"), "text": PartSpec("plain"), "vae": PartSpec("plain")},
    "qwen_image": {"dit": PartSpec("plain", "int8"),   # 38.05 是 bf16 那份量的 ✓（别调小 ✗）
                   #: 这条线只有 dit 按文件精算 ✓；text/vae 仍按表 ✓（改了会让预检与装配线对不上 ✗）
                   "text": PartSpec(measured=False),
                   "vae": PartSpec(measured=False)},
    "image": {"dit": PartSpec("plain")},          # SDXL：单 checkpoint，调用方退回文件字节 ✓
}


def stage_table_need(stage: str, part: str) -> float:
    """**某档某件**的表值 ✓（GiB ✓）—— 不许查不到还编一个数 ✓✗。

    ⚠️ 查不到就**抛** ✓（档名不在 :attr:`TorchBackend.STAGE_NEED_TABLES`、或表里没这一件）——
    预检拿到异常应当**如实报"没有这本账"** ✓，而不是退回 0 让结论看着能跑 ✓✗。
    """
    stage_key = str(stage or "")
    table = TorchBackend.STAGE_NEED_TABLES.get(stage_key)
    if table is None:
        raise ValueError(f"档 {stage_key!r} 没有逐件驻留表"
                         f"（有表的档：{sorted(TorchBackend.STAGE_NEED_TABLES)}）"
                         " ⇒ 这条线必须显式给 fallback")
    if part not in table:
        raise ValueError(f"档 {stage_key!r} 的表里没有 {part!r}（表里只有 {sorted(table)}）")
    return float(table[part])


def stage_part_need(stage: str, part: str, path: str | Path | None, *,
                    fallback: float | None = None) -> tuple[float, dict[str, Any]]:
    """**某档某件**要多少 GiB ✓ ⇒ ``(GiB, 凭据)`` ✓ —— 装配线与预检的**唯一出口** ✓✗。

    ``fallback=None`` ⇒ 按 :attr:`TorchBackend.STAGE_NEED_TABLES` 现查 ✓（预检走这条：
    它**没有**装配线里那份局部变量，也不该自己抄一份 ✓✗）；装配线仍显式传值 ✓（行为不变 ✓）。
    """
    spec = STAGE_PART_SPECS.get(str(stage or ""), {}).get(str(part or "")) or PartSpec()
    if fallback is None:
        fallback = stage_table_need(stage, part)
    if not spec.measured:
        return float(fallback), {"source": "table", "mode": "table-only",
                                 "table": round(float(fallback), 2),
                                 "why": "这条线不为该件按文件精算 ⇒ 按表 ✓（见 STAGE_PART_SPECS ✓）"}
    return part_need_gib(path, fallback=fallback, adapter=spec.adapter, assume=spec.assume)


def part_need_gib(path: str | Path | None, *, fallback: float,
                  adapter: str | None = None,
                  assume: str | None = None) -> tuple[float, dict[str, Any]]:
    """**这一件权重**在这条装配线上装起来占多少 GiB ✓ ⇒ ``(GiB, 凭据)`` ✓。

    ## 为什么不能只留一张表 ✗

    表里的数是「**某个文件**在某台机器上量出来的驻留」✓ —— 同一族换一个量化档就**不再成立** ✓✗：

    * Krea2 的 ``12.57 GiB`` 是 **int8** 档（224 层 ``Int8Linear`` 常驻 ✓）；
      换成 **int4** 档 ⇒ 反量化成 bf16 是 **23.9 GiB** ✓✗ —— 拿 12.57 判「装得下」⇒ 整份上卡
      ⇒ 采样第 1 步 OOM ✓，而报错指不到「表用错了档」✗。
    * 反过来，拿 bf16 档的数去判 int8 档 ⇒ 白白切一半到内存里逐层爬 ✓✗（Qwen-Image 那次的症状 ✓）。

    ## 判据（`assume` 就是表值的**出处** ✓）

    ``assume`` ＝「表里这个数是按**哪种档**的文件量的」✓（``"int8"`` / ``"int4"`` / ``"plain"``）：

    * 实测档 == ``assume`` ⇒ **用表值** ✓（表值含**骨架自建参数** ✓，比只算文件数据区更准 ✓）；
    * 否则 ⇒ 用 :func:`resident_bytes_of` 的**精算** ✓（按 dtype 与常驻壳逐层算 ✓）；
    * 读不出头部 / 头部自述有问题 ⇒ 用表值 ✓ 并把原因写进凭据（不猜 ✗）。

    凭据里 ``source`` 说明这次用的是 ``"table"`` 还是 ``"file"`` ✓ —— 回执里要能答
    「这个需求数是量出来的还是算出来的」✓。
    """
    table = float(fallback)
    if not str(path or "").strip():
        return table, {"source": "table", "mode": "unknown", "table": table,
                       "why": "没给权重路径 ⇒ 按表（无从精算 ✓）"}
    try:
        info = st.inspect(str(path))
    except Exception as err:  # noqa: BLE001 —— 读不到就按表，别把装配挡住
        return table, {"source": "table", "mode": "unknown", "table": table,
                       "why": f"读不出头部 ⇒ 按表 {table:.2f} GiB（{err}）"}
    if not info.ok:
        return table, {"source": "table", "mode": "unknown", "table": table,
                       "why": f"头部自述有问题 ⇒ 按表 {table:.2f} GiB（{info.problems[:2]}）"}
    try:
        specs = loader.quant_specs_of(Path(str(path)), info)
    except Exception as err:  # noqa: BLE001 —— 补证据失败不该把装配搞崩
        specs = {}
        specs_error = str(err)
    else:
        specs_error = ""
    resident, ledger = resident_bytes_of(info, adapter=adapter, specs=specs)
    #: ⭐ 档位名必须**先认 nvfp4** ✗✗（2026-11-09 ✓）：nvfp4 那份文件里 ``embed_tokens`` 是 ``I8`` ✓
    #: ⇒ ``int8Layers=1`` ✓ ⇒ 老算法会把整份判成 **int8 档** ✓✗，于是"assume=int8 且 mode=int8"
    #: 成立 ⇒ 返回表值 **25.44 GiB** ✓✗（那份 bf16 口径的账 ✓），而打包常驻真值 ≈15.3 GiB ✓
    #: ⇒ 凭空多要 10 GiB、把能整份上卡的档判成必须分时换装 ✓（不报错，只是慢与账假 ✓✗）。
    mode = ("nvfp4" if ledger["nvfp4Layers"]
            else ("int4" if ledger["int4Layers"] else ("int8" if ledger["int8Layers"] else "plain")))
    if assume is not None and str(assume) == mode:
        return table, {
            "source": "table", "mode": mode, "table": table,
            "fileGiB": round(info.file_bytes / device_policy.GIB, 2),
            "why": (f"文件是 **{mode}** 档 ✓ ⇒ 表里 {table:.2f} GiB 正是按这种文件量的 ✓"
                    f"（含骨架自建参数 ✓；文件数据区 "
                    f"{resident / device_policy.GIB:.2f} GiB 只作旁证 ✓）"),
        }
    est = resident / device_policy.GIB
    return est, {
        "source": "file", "mode": mode, "table": table,
        "fileGiB": round(info.file_bytes / device_policy.GIB, 2),
        "residentGiB": round(est, 2),
        "int8ResidentLayers": ledger["int8ResidentLayers"],
        "int8EmbedLayers": ledger.get("int8EmbedLayers", 0),
        "int4Layers": ledger["int4Layers"],
        "int4ShellLayers": ledger.get("int4ShellLayers", 0),
        "nvfp4Layers": ledger.get("nvfp4Layers", 0),
        "why": (f"文件是 **{mode}** 档 ✓，而表里 {table:.2f} GiB 是按 **{assume or '未登记'}** 档量的"
                f" ⇒ 这次**不能**用表 ✓ ⇒ 按文件精算 {est:.2f} GiB ✓"
                f"（int8 常驻 {ledger['int8ResidentLayers']} 层 ✓、4bit 打包 {ledger['int4Layers']} 层 ✓"
                f"，其中 {ledger.get('int4ShellLayers', 0)} 层走 4bit 常驻壳 ✓"
                f"{'（其余按 bf16 反量化 ✓）' if ledger.get('int4Layers', 0) > ledger.get('int4ShellLayers', 0) else ''}"
                + (f"；nvfp4 打包常驻 {ledger.get('nvfp4Layers', 0)} 层 ✓"
                   f"（打包原样 ✓、前向逐层反量化 ✓；块尺度/标量按文件字节算 ✓）"
                   if ledger.get("nvfp4Layers", 0) else "")
                #: ⚠️ 嵌入层**没有 int8 壳** ⇒ 它按 bf16 物化 ✓ —— 不写这一句，
                #: 精算值会比文件数据区大 0.7 GiB 而**看不出原因** ✓✗
                + (f"；另有 {ledger.get('int8EmbedLayers', 0)} 个 int8 嵌入/输出头 ✓"
                   f"（本仓没有 int8 嵌入壳 ⇒ 按 bf16 物化 ✓）"
                   if ledger.get("int8EmbedLayers", 0) else "") + "）"
                + (f"；⚠️ 量化说明读取失败（{specs_error}）⇒ 4bit 层可能按 int8 算 ✓"
                   if specs_error else "")),
    }


def _int8_bytes_of(module: Any, torch_mod: Any) -> int:
    """模块里 int8 **与 uint8** 参数总字节 ✓（＝「钉到 CPU 会占多少内存」✓）。

    ⚠️ ``uint8`` 必须一起数 ✗✗（2026-11-09 ✓）：nvfp4 打包壳（:class:`~.h3_form.NvFp4Linear` ✓）
    的权重是 ``uint8`` ✓ —— 只数 ``int8`` 会让「钉 CPU 前先问内存够不够」这一拍**漏掉**
    11.35 GiB ✓✗，而 :func:`~.h3_form.place_module` 照样把它们钉过去 ✓ ⇒ 内存被悄悄吃掉 ✓。
    """
    total = 0
    for param in module.parameters():
        if param.dtype in (torch_mod.int8, torch_mod.uint8):
            total += int(param.numel())
    return total


def _materialize_meta(module: Any, *, torch_mod: Any, device: Any = "cpu") -> int:
    """把 meta 上的参数/缓冲区换成未初始化实存 ⇒ 换了几个（**实现已下沉**，见 :mod:`dit_build`）。

    保留这个名字是因为 H3 那条路（:func:`_prepare_h3_module`）与本文件另外两处直接调它，
    以及 ``tests/engine_*`` 里有用例按名字取 —— 行为与实现逐字一致 ✓。
    """
    return dit_build.materialize_meta(module, torch_mod=torch_mod, device=device)


def _int8_residency_problems(module: Any, report: Any) -> tuple[dict[str, Any], str | None]:
    """核对「int8 权重是不是**全都**落成了常驻层」✓ ⇒ ``(账目, 致命问题或 None)`` ✓。

    ## 为什么必须有这一条 ✗

    :func:`h3_form.upgrade_linear_to_int8` 返回「换掉了哪些层」✓，但装载侧一直是把它**丢掉**的 ✗
    —— 换失败（某层名解析错 / 不是 ``nn.Linear`` / 将来权重多出一类新键）时最可能的结局**不是报错** ✗，
    而是：「计划里点名了、模块里还是普通 ``nn.Linear``」⇒ 装载时 ``copy_`` 走**反量化**那条路 ✓
    ⇒ 该层按 **bf16 落盘（2 倍显存）** ✓✗ ⇒ 22.49 GiB 的卡上直接 OOM ✓✗，
    而错误会出现在**后面的前向**里（「CUDA out of memory」），指不到「哪层没接管」✗。

    两个数字原本并排在报告里（``int8Resident`` 文件侧 ✓、``parked`` 模块侧 ✓），
    但它们**从来没人对过账** ✗ —— 各自都"正常"、合起来才说明问题 ✓✗。

    ## 判据（两条，都是**不猜**的等式 ✓）

    1. ``Int8Linear`` 层数 == int8 参数个数 ✓（一层恰好一个 int8 权重 ✓）；
    2. 文件侧 ``int8Resident`` **不少于**层数 ✓ —— 少了就是「常驻层没被填」✓
       （那几层权重要么全 0 ✓✗，要么被反量化进了别的形状 ✓✗，两种都必错 ✗）。

    ⚠️ 只查 ``<``（漏装）不查 ``>``：将来若出现**不是** Linear 的 int8 张量（如量化后的
    embedding ✓），``int8Resident`` 会比层数**大** ✓ —— 那是合法的新品类，不该被这条判死 ✗。

    ## 第 2 条的分母：**写进去的 ＋ 卸下去的** ✓✗（2026-11-07 ✓）

    「内存这一级」会**故意**把几层 int8 卸到内存/盘上（`weights` 的 ``skip_prefixes`` ✓，`report.skipped`
    ✓）—— 那些层**根本没被写进去** ✓ 是设计 ✓。原先分母只取 ``int8Resident`` ⇒ 卸得越多、越判成
    「差的层会按 bf16 落盘」⇒ **真机实测**：假 8 GiB 卡上装 Krea2，224 层里 88 层写进去、136 层卸下去
    ⇒ 被拒装 ✓✗（`residency="off"` 同机同文件 5.7 s 装得进去 ⇒ 是判据错、不是权重错 ✓）。
    所以分母 ＝ ``int8Resident + int8Skipped`` ✓；但**卸下去就必须填得回来** ✗✗：``restorer is None``
    而 ``int8Skipped > 0`` ⇒ 那几层是**填不回去的洞** ✗ ⇒ 照样拒装（并说明是洞、不是算术 ✓）。
    """
    import torch  # noqa: PLC0415

    layers = sum(1 for sub in module.modules() if type(sub).__name__ == "Int8Linear")
    int8_params = [param for param in module.parameters() if param.dtype == torch.int8]
    resident = int(getattr(report, "int8Resident", 0) or 0)
    #: ⭐ 卸到内存/盘上、因而**故意没写**的 int8 层 ✓（2026-11-07 ✓）—— 它们也**交代清了** ✓✗。
    skipped = int(getattr(report, "int8Skipped", 0) or 0)
    ledger = {
        "layers": layers,
        "int8Params": len(int8_params),
        #: 留在 CPU 的那份 ✓ —— 这就是 `int8_offload` 要省下的显存 ✓（单位见 `parkedBytes` ✓）
        "int8OnCpu": sum(1 for param in int8_params if param.device.type == "cpu"),
        "int8OnGpu": sum(1 for param in int8_params if param.device.type != "cpu"),
        "int8Resident": resident,
        #: ⭐ 其中**卸到内存/盘上**的层数 ✓（2026-11-07 ✓）：它们没被写进去是**策略** ✓，
        #: 但 `restorer` 必须能把它们填回来 ✓✗（否则就是洞 ✗）。
        "int8Skipped": skipped,
        #: ⭐ 其中**4bit 打包**的层数 ✓（2026-11-02 ✓）：与 ``int8Resident`` 是**子集关系** ✓
        #: —— 它们的参数同样 ``int8`` ✓，只是前向多一步解包 ✓。单列出来才看得出
        #: 「说好 4bit、实际按 int8 乘」✓✗（头部逐字同形 ⇒ 合计数看不出来 ✓）。
        "int4Resident": int(getattr(report, "int4Resident", 0) or 0),
    }
    if layers != len(int8_params):
        return ledger, (
            f"int8 常驻层 {layers} 个、int8 参数 {len(int8_params)} 个 ✗ ⇒ 结构自相矛盾"
            f"（一层应当恰好一个 int8 权重 ✓）")
    if skipped > 0 and getattr(report, "restorer", None) is None:
        return ledger, (
            f"{skipped} 层 int8 被卸到内存/盘上，但这次装载**没留下还原器** ✗ ⇒ 那几层填不回来"
            f"（前向到那层就是一个空壳 ✓✗）—— 卸层省内存的前提是 **`report.restorer` 能重读** ✓✗")
    accounted = resident + skipped
    if accounted < layers:
        return ledger, (
            f"文件里只有 {resident} 个 int8 权重被**原样**装进常驻层"
            f"（另有 {skipped} 层卸到内存/盘上 ✓），而计划换掉的是 {layers} 层 ✗ ⇒ "
            f"差的 {layers - accounted} 层会按 bf16 落盘（显存 2 倍 ✓✗），"
            f"在 22.49 GiB 的卡上就是 OOM ✓✗ —— 而报告里 `parked` / `int8Resident` 两个数各自都\"正常\"✓✗")
    return ledger, None


def _prepare_h3_module(h3_form_mod: Any, form_defaults: dict[str, Any], target: Path,
                       info: Any, *, dtype: Any = None) -> tuple[Any, int]:
    """建 H3 主干、按**权重自己说的精度**摆好 ✓ ⇒ ``(模块, 物化了多少个参数)`` ✓。

    ## 四件事，顺序不能换 ✗

    1. **在 ``meta`` 上建壳** ✓ —— 19.53 GiB 的主干在 fp32 下是 **78 GiB** ✓✗；先真建一遍再降精度
       ⇒ 峰值就是 78 GiB（96 GB 机器上进程 WS 实测 49~65 GiB ✓）、**显存一字节不动** ✓✗。
       早先「建两次」也栽在这条上 ✓（建两次 = 156 GiB ✓✗）；
    2. **降到计算精度** ✓（有 CUDA 时 bf16 ✓）—— 但**文件里写明 fp32 的键必须留在 fp32** ✓
       （H3 的 ``video_patch_proj`` / 输出头是 **fp32 岛** ✓，降了就是数值退化 ✓✗）；
       ⚠️ 在 meta 上做**不分配** ✓（只记 dtype ✓）；
    3. **int8 层换成常驻层** ✓（:func:`h3_form.upgrade_linear_to_int8` ✓）—— 必须在降精度
       **之后** ✗：否则 ``weight_scale`` 被一起降成 bf16 ✗（尺度精度比 int8 的量化误差还大 ✗）；
    4. **到这时才物化** ✓（:func:`_materialize_meta` ✓）—— 按**换完壳的最终形状与精度**一次分配 ✓
       ⇒ 峰值 = 19.5 GiB 上下 ✓。⚠️ 换成 ``module.to_empty(...)`` 会**不管 3 步的壳**、按 bf16 整体
       分一遍（38.5 GiB ✓✗）—— 见 :func:`_materialize_meta` 的注释 ✓。

    ⚠️ 第 2 步**不能**改成「干脆不降」✗：fp32 主干 78 GiB ⇒ 装不下 ✓（这正是
    「22.49 GiB 显存装 19.53 GiB 权重」那条账的另一半 ✓）。
    """
    import torch  # noqa: PLC0415

    from app.services.engine import quant as quant_mod  # noqa: PLC0415 —— 局部 ✓ 不拖 torch ✓

    #: ⭐⭐⭐ **先在 ``meta`` 上建壳** ✓✗（2026-11-01 实测的真病根 ✓）—— 这一行原来是
    #: ``H3FormTrunk(**form_defaults)`` ✓，也就是**先按 fp32 真建一遍** ✓：19.53 GiB 的 int8 主干
    #: 在 fp32 下是 **78 GiB** ✓✗ ⇒ 进程 WS 冲到 **49 ~ 65 GiB**、宿主只剩 30 GiB 甚至 9 GiB ✓，
    #: **而这段时间显存一字节没动** ✓✗（用户看到的就是「说了 GPU 优先，卡上却什么都没有」✓✗）。
    #: 更坏的是它**不是一瞬间**：建一份 78 GiB 的 fp32 主干在本机要几分钟 ✓，全程都在页错误里爬 ✓。
    #: ⇒ 与 :meth:`attach_h3_text` 同一条处置 ✓：meta 上建（0 字节 ✓）→ 降精度（meta 上免费 ✓）
    #: → 扳 fp32 岛（免费 ✓）→ 换 int8 常驻层（免费 ✓）→ :func:`_materialize_meta` **按最终精度
    #: 一次分配** ✓ ⇒ 峰值 = 最终大小（约 19.5 GiB ✓）而不是 78 GiB ✓。
    with torch.device("meta"):
        module = h3_form_mod.H3FormTrunk(**form_defaults)
    # ② 计算精度：CUDA 上 bf16 ✓（A5000 原生支持 ✓、权重里非 int8 的张量本来就是 bf16 ✓）；
    #    CPU 上留 fp32 ✓（bf16 在 CPU 上既慢又没有收益 ✓）。
    #    ⚠️ 此刻还在 meta 上 ✓ ⇒ 这行**不分配** ✓，只是把 dtype 记在壳上 ✓。
    module.to(dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32))
    #: ⚠️ 逐键把文件里是 **F32** 的键扳回 fp32 ✓（不能只认名字 ✗：fp32 岛的具体清单属于
    #:    「权重自己说了算」那一类 ✓，写死名单就会与将来的重导出权重脱钩 ✓✗）。
    #    ⚠️ 必须遍历 **`named_parameters`/`named_buffers`** ✗ —— `state_dict()` 里给的是
    #    **detached 张量** ✓✗：改它们的 `.data` 只是改了那份 view ✓，`module` 里的参数纹丝不动 ✓✗
    #    （2026-10-29 实测踩到 ✓：fp32 岛静默留成 bf16 ⇒ 前向报
    #    「mat1 and mat2 must have the same dtype, but got Float and BFloat16」✗，
    #    而报错位置在 `video_patch_proj` ✓ 与本行只隔一次 `.to()` ✓）。
    holders = (dict(module.named_parameters()), dict(module.named_buffers()))
    for logical, tensor in info.tensors.items():
        if tensor.dtype != "F32":
            continue
        for holder in holders:
            target = holder.get(logical)
            if target is not None and target.is_floating_point():
                target.data = target.data.to(torch.float32)

    #: ③ int8 常驻层 ✓ —— convrot 必须走**逐层**说明（`loader.quant_specs_of` ✓ 与体检、装载侧
    #:    同一个入口 ✓✗）：`quant_formats_of` 只读前 8 个**去重**的族级说明 ✓ ⇒ 两个坑都在这里：
    #:    ① H3 全量版逐层分组不同（64/256 并存）⇒ 拿族级判会把正品拒掉 ✓；
    #:    ② Krea2 int4 那族一个 `*.comfy_quant` 都没有、说明只在 metadata 里 ⇒ 拿族级判等于
    #:       「没看见旋转」⇒ 常驻层**少一步反旋转** ⇒ 能装、能跑、画面全错 ✓✗（假绿 ✓）。
    convrot, convrot_gap = quant_mod.convrot_group_of(loader.quant_specs_of(target, info))
    if convrot_gap:
        raise TorchBackendUnavailable(
            f"convrot 说明自相矛盾 ✗ ⇒ 不装载（不按猜的 group 算 ✗）：{convrot_gap}", reason="pending")
    plan = int8_scale_plan(info)
    if plan:
        h3_form_mod.upgrade_linear_to_int8(module, plan, convrot=convrot)
    #: ④ **到这一步才真分配** ✓✗ —— 并且是按"换完壳之后的**最终**形状与精度"分配 ✓：
    #:    int8 常驻层各占 1 字节/元素 ✓、bf16 岛 2 字节 ✓、fp32 岛 4 字节 ✓ ⇒ 合计就是文件那
    #:    19.53 GiB 上下 ✓（不是 78 GiB ✓✗）。⚠️ **顺序不能换** ✗：`upgrade_linear_to_int8`
    #:    必须在物化**之前** ✓，否则先按 bf16 物化一份（38.5 GiB ✓✗）再换壳 ⇒ 白分配 38 GiB ✓✗
    #:    （这就是 `to_empty` 那条老坑 ✓ 见 :func:`_materialize_meta` 的注释 ✓）。
    filled = _materialize_meta(module, torch_mod=torch, device="cpu")
    return module, filled


#: 能按**专用骨架**装的族 ✓ —— 目前只有 Z-Image（`NextDiT` 同构 ✓ 见 :mod:`dit_lumina` ✓）。
#: 其余族仍走通用 :func:`dit.build_dit` ✓（那是「一条 dit 骨架」的旧路 ✓，不是这里要改的事 ✗）。
ZIMAGE_FAMILY = "zimage"

#: Qwen-Image **2.1**（单流 + SwiGLU ✓ 见 :mod:`dit_qwen21` ✓）。
#: ⚠️ 与 ``qwen_image``（2512/Edit 的 MMDiT ✓ 有 ``attn.add_q_proj`` ✓）**不是一族** ✗：
#: 2.1 没有 ``add_q/add_k/add_v``、调制是全局 ``modulation.1``、MLP 是 ``gate_up`` ✓
#: （判族靠 :mod:`dit_family` 的两条 sentinels 分开 ✓）。
QWEN21_FAMILY = "qwen_image21"

#: Qwen-Image **2512**（t2i）/ **Edit 2511**（双流 MMDiT ✓ 见 :mod:`dit_qwen_image` ✓）。
#: ⚠️ 与 :data:`QWEN21_FAMILY` 只差一个 `21` ✗✗ —— 但两者**结构、潜通道数、时间口径全不同** ✓
#: （双流 6 组调制 / 16ch / `1000·σ` 不翻转 vs 单流 / 64ch / lumina 式翻转 ✓）。
#: 这两格**不能靠字符串前缀糊在一起** ✗：`qwen_image21` 虽然以 `qwen_image` 开头 ✓，
#: 族名比较一律用 `==` ✗ 不要用 `startswith` ✓（用它 ⇒ 2.1 会被 2512 的骨架接走 ✓✗，
#: 4 组调制对 6 组调制的键对账会报「缺 4 组」✓，读起来像权重没下全 ✗）。
QWEN_IMAGE_FAMILY = "qwen_image"

#: ⭐ **Krea2**（单流 ✓ ``first/last/blocks/txtfusion`` ✓ 见 :mod:`dit_generic` 的 ``Krea2DiT`` ✓）。
#: 与上面三族都**不是**一套键 ✗；条件也不是「一根向量」✗ 而是 **Qwen3-VL-4B 的 12 层栈**
#: （``3krea2`` 的 ``txtlayers=12 × txtdim=2560 ⇒ 30720`` 宽 ✓ 盘上逐数如此 ✓）。
#: 潜变量 / VAE 走 **Wan21**：16ch、8×、**有** mean/std 反缩放 ✓（``latent_formats.Wan21`` ✓），
#: 采样 ``shift = 1.15``（``supported_models.Krea2.sampling_settings`` ✓ —— 不是 Flux 的 3 ✗）。
KREA2_FAMILY = "krea2"

#: ⭐ **SEEDVR2**（视频**修复/超分** ✓ 见 :mod:`dit_seedvr2` ✓ / :mod:`vae_seedvr2` ✓）。
#:
#: 与前六族的差别是**任务换了一类** ✗：它不是「从噪声生成」✓，而是「给一段退化视频 ⇒ 出修复版」✓
#: ⇒ 主干每一步都要额外吃 **conditioning 17 通道** ✓（退化视频的 16 通道潜变量 ＋ 1 通道 loss mask ✓），
#: 所以 ``vid_in`` 是 **33** 通道 ✓（噪声 16 ＋ 条件 17 ✓）而**潜变量仍是 16** ✓✗
#: （:meth:`TorchBackend.latent_shape` 必须读 ``latent_channels`` 那一格 ✓，见那里的坑 ✓）。
#: 潜空间口径 ``latent_formats.SeedVR2`` ✓（反缩放 ×**0.9152** ✓ 不是 mean/std 组 ✗）；
#: 而且它**没有文本塔** ✓ —— 提示词是 ``positive_conditioning`` / ``negative_conditioning``
#: 烤在权重里的固定上下文 ✓（58 / 64 条 × 5120 宽 ✓）。
SEEDVR2_FAMILY = "seedvr2"

#: Krea2 权重在盘上带**三层**前缀（不同导出工具各写一层 ✓）：先长后短地试 ✓，
#: 一起交给 :func:`weights.load_module_weights` 的 ``key_map`` ✓（正则 ⇒ 点要转义 ✓）。
#: ⚠️ 判据是「**全部**键都带这一层」✓ 不是「有一些带」✗ —— 后者会把两族混在一起认 ✓✗。
#: 单一真源：表在 :mod:`dit_build`（判前缀与建网现在是同一份声明），这里只留名字给文档引用。
_KREA2_PREFIXES: tuple[str, ...] = dit_build.KREA2_PREFIXES

#: ⭐ **前向契约**（潜变量几维 / σ 喂什么形态）—— 真源在 :attr:`dit_build.BUILDERS` 的
#: ``contract`` 字段 ✓，这里只是**派生视图**（保留旧名字给文档与测试引用 ✓）。
#:
#: 2026-11-01 收口前：这个元组是手写的 ✓✗，``latent_shape`` / ``_latent_mode`` / ``denoise``
#: 三处各读一遍同名族名 ⇒ 加族时「改了两处、漏一处」**不报错** ✗（形状对、数是噪声 ✓✗）。
#: 现在三处都读 :func:`dit_build.contract_of` ✓，族表加一行三处自动跟上 ✓✓。
IMAGE_ONLY_FAMILIES: tuple[str, ...] = dit_build.image_only_keys()

#: ⭐ **Flux.1**（双流 19 + 单流 38 ✓ 见 :mod:`dit_flux` ✓）。
#: 与前四族的**接口**差别只有一处 ✓✗：条件不止一根 —— T5 的 4096 宽走 ``condition`` ✓，
#: 另有 CLIP-L 的 pooled 768 宽 ✓ 和一个 guidance **标量** ✓。后两根不进 ``denoise`` 的
#: 位置参数（那会让所有族都多两个空位 ✗），而是**挂在模块上** ✓（``default_vector`` /
#: ``default_guidance`` ✓，见 :meth:`TorchBackend.attach_flux_conditioning` ✓）。
FLUX_FAMILY = "flux"


def _dit_family_of(info: Any) -> str | None:
    """按**权重键名/形状**判族 ⇒ 族名 ✓；认不出 ⇒ ``None`` ✓（**不猜** ✗ —— 认不出就走老路 ✓）。

    ⚠️ 为什么非要判族 ✗：Z-Image 那一族（``x_embedder``/``cap_embedder``/``noise_refiner``/
    ``context_refiner``/``layers`` ✓）与 Krea2 形态（``first``/``last``/``blocks``/``txtfusion`` ✓）
    是**两种键布局** ✓✗ —— 硬把 Z-Image 权重塞进 Krea2 骨架，结果是「装载报告缺一堆键」✓，
    人很容易读成「权重没下全」✗（同一个坑 H3 那条已经踩过 ✓ 见上面判形态那段注释 ✓）。

    判族是**纯函数、只看头部键名/形状** ✓ ⇒ 毫秒级、不读大文件 ✓（:mod:`dit_family` 只报告
    权重里写着的事实 ✓ 头数/eps 这类读不到的一律进 ``unknown`` ✗ 不猜 ✓）。
    """
    from app.services.engine import dit_family as dit_family_mod  # noqa: PLC0415 —— 局部 ✓ 不拖 torch ✓
    try:
        spec = dit_family_mod.detect(info)
    except Exception:  # noqa: BLE001 —— 判族失败只该「按老路装」✓，不该把装载整个搞崩 ✗
        return None
    return getattr(spec, "key", None)


#: ── 以下是**过渡薄壳**（2026-11-01）────────────────────────────────────────────
#: 实现已收口到 :mod:`dit_build` 的装配表（一族一行声明）。保留这些名字有两个原因：
#: ``tests/engine_dit_lumina_test.py`` / ``engine_dit_qwen_image_test.py`` 直接调它们，
#: 运维脚本也可能这么调 —— 行为逐字不变 ✓。
#: **新代码别照这个抄** ✗：加族请改 ``dit_family._RULES`` + ``dit_build.BUILDERS``，
#: 不要再往本文件加 ``_build_*``（那正是本表要消掉的重复样板 ✗）。
def _build_via_table(key: str | None, info: Any, explicit: dict[str, Any] | None,
                     *, dtype: Any = None, path: Any = None) -> tuple[Any, dict[str, Any]]:
    """查表建网 ⇒ ``(module, config)``；表里没有这族 ⇒ 抛不可用（**不静默回老路**）。"""
    try:
        built = dit_build.build(key, info, explicit or {}, dtype=dtype, path=path,
                                int8_plan=int8_scale_plan, int4_plan=int4_scale_plan)
    except dit_build.BuildError as err:
        raise TorchBackendUnavailable(str(err), reason=getattr(err, "reason", "pending")) from err
    if built is None:
        raise TorchBackendUnavailable(
            f"族 {key} 没有装配声明 ✗（`dit_build.BUILDERS` 里缺这一行："
            f"加族要同时改 `dit_family._RULES` 与它）", reason="pending")
    return built


def _build_zimage_module(info: Any, explicit: dict[str, Any],
                         *, dtype: Any = None) -> tuple[Any, dict[str, Any]]:
    return _build_via_table(ZIMAGE_FAMILY, info, explicit, dtype=dtype)


def _build_qwen21_module(info: Any, explicit: dict[str, Any],
                         *, dtype: Any = None) -> tuple[Any, dict[str, Any]]:
    return _build_via_table(QWEN21_FAMILY, info, explicit, dtype=dtype)


def _build_qwen_image_module(info: Any, explicit: dict[str, Any],
                             *, dtype: Any = None) -> tuple[Any, dict[str, Any]]:
    return _build_via_table(QWEN_IMAGE_FAMILY, info, explicit, dtype=dtype)


def _build_krea2_module(info: Any, explicit: dict[str, Any], *, dtype: Any = None,
                        path: Any = None) -> tuple[Any, dict[str, Any]]:
    return _build_via_table(KREA2_FAMILY, info, explicit, dtype=dtype, path=path)


def _krea2_stripped(shapes: dict[str, tuple[int, ...]]
                   ) -> tuple[dict[str, tuple[int, ...]], dict[str, str] | None, str | None]:
    return dit_build.krea2_strip(shapes)


def _krea2_specs(specs: Any, prefix: str | None) -> Any:
    return dit_build.krea2_specs(specs, prefix)


def _timesteps_of(torch_mod: Any, latents: Any, sigma: float) -> Any:
    """σ ⇒ ``(b,)`` 的 timestep **张量** ✓（`LuminaDiT.forward` 要逐 batch 的张量 ✗ 不是标量 ✓）。

    ⚠️ 口径**不在这里换** ✗✗：``LuminaDiT`` 内部自己做 ``t = 1 - timesteps`` 再乘
    ``time_scale`` ✓（见 :mod:`dit_lumina` 模块头第 1 条 ✓）⇒ 这里必须**原样传 σ** ✓；
    先取反就是错两次 ✓✗（形状全正常、数值全错 ✓ —— 这一族最容易写错的一行 ✓）。

    ``dtype`` 跟 latent 走 ✓：bf16 latent 配 fp32 时间向量 ⇒ 第一层调制 MatMul 就撞
    ``mat1 and mat2 must have the same dtype`` ✗（H3 的 fp32 岛那条同一个形状的报错 ✓）。
    """
    batch = int(latents.shape[0]) if getattr(latents, "ndim", 0) else 1
    return torch_mod.full((batch,), float(sigma), dtype=latents.dtype, device=latents.device)


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


def torch_facts() -> dict[str, Any]:
    """``torch`` **本体**装没装、是不是 **CUDA 轮子** ✓ —— 与「后端能不能用」是**两件事** ✗。

    ⚠️ 为什么单列（2026-10-29 ✓）：``torch_available()`` 是「**必需依赖齐不齐**」✓ —— 缺
    ``safetensors`` 时它也返回 ``False`` ✗，于是报告只写「torch ✗」，读起来像「torch 没装」
    ✗✗。2026-10-29 实测本机：``torch 2.14.1+cu130`` **装了** ✓ 且**是 CUDA 轮子** ✓
    （``cudaAvailable=True`` ✓）；此前一度是 ``2.15.0+cpu`` ✗（上游 2.15.0 尚无任何 CUDA 轮子 ✓）
    —— 那种情形比缺两个小包严重得多（19.53 GiB 主权重在 CPU 上跑不出片），却完全不会出现在报告里 ✗。

    ⭐ **单一真源** ✓：本函数是唯一实现 ✓；`app.services.engine_readiness.torch_facts` 只是
    转发 ✓ —— 同一个事实两处各写一遍，迟早给出两个说法 ✗✗（2026-10-29 的重复实现已合并 ✓）。
    """
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        #: ⚠️ `sys.modules["torch"] = None` 时 `find_spec` 会抛 **ValueError** ✗（不是返回 None ✓）
        #: ⇒ 不接住它，「无 torch 也能算」这条判据会在探测这一步就崩 ✓✗（契约测试 ⑥ 正是这么打的 ✓）。
        spec = None
    if spec is None:
        return {"installed": False, "version": None, "cudaVersion": None,
                "cudaAvailable": False, "cpuWheel": False}
    try:
        import torch  # noqa: PLC0415 —— 只有体检才走这里 ✓（命令行/手动端点 ✓ 不是热路径 ✓）
    except Exception as err:  # noqa: BLE001 —— 装坏了也是**结论** ✓ 不抛 ✗
        return {"installed": False, "version": None, "cudaVersion": None,
                "cudaAvailable": False, "cpuWheel": False,
                "error": f"{type(err).__name__}: {err}"}
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_available = False
    return {"installed": True, "version": str(getattr(torch, "__version__", "") or ""),
            # ⚠️ `version.cuda is None` ＝ **CPU 轮子** ✓（编译期事实 ✓ 与「有没有卡」无关 ✗）
            "cudaVersion": cuda_version, "cudaAvailable": cuda_available,
            "cpuWheel": cuda_version is None}


def torch_available() -> tuple[bool, str]:
    """``(可用?, 原因)`` ✓ —— 不可用时原因**可行动** ✓。"""
    status = dependency_status()
    if status["ready"]:
        return True, "依赖齐备 ✓"
    return False, (f"缺少依赖：{', '.join(status['missing'])}（共约 "
                   f"{sum(item['approximateMB'] for item in status['items'] if not item['present'])} MB ✓）"
                   f" ⇒ 先跑 {status['install'][0]} ✓")


def _freeze_inference(module: Any) -> int:
    """把装好的模块摆成**纯推理**形态 ✓ ⇒ 返回这回被标成「不取梯度」的参数个数 ✓✗。

    ## 为什么这一句不是可有可无的洁癖（2026-11-01 端到端实测 ✓）

    骨架里凡是 ``nn.Parameter(torch.ones(dim))`` 建的 norm 权重（``h3_form.RMSNorm`` / 
    ``dit_lumina.RMSNorm`` / ``qwen35_te`` 的 ``_RmsNorm`` …）**默认 ``requires_grad=True``** ✓ ——
    本仓是推理引擎 ✓、**从不 backward** ✗，于是这些参数的唯一作用是：让整条前向
    **建起 autograd 图** ✓✗。代价不是"多算一点"✗，而是三层实打实的灾难 ✓：

    1. **每个中间激活都被 `saved_tensors` 钉住** ✓✗ —— 每步的整条计算图都不会在步末释放 ✓
       ⇒ H3 一次 4 步采样实测：DiT 图上 ``allocated`` 一路涨到 **83.7 GiB** ✓（卡只有 22.49 GiB ✓）；
    2. **超出的部分由 WDDM 静默挪进系统内存** ✓✗（"Shared GPU memory"）—— 于是宿主私有提交
       从 38 GiB 涨到 **99.5 GiB**、占用率 99% ✓，`nvidia-smi` 却报"卡用了 22.6 GiB 正常" ✓；
    3. **慢到一步都跑不完** ✓✗（实测 100 秒还停在采样第 0 步 ✓，看起来像"卡死"✓）。

    反量化那条路上还会多留一份：``Int8Linear.forward`` 每层现反量化的 bf16 权重是**局部变量** ✓，
    正常情况下 `forward` 一返回就释放 ✓；一旦建图，它就被 ``AddmmBackward`` 当 saved tensor 吊住 ✓✗
    （探针实测：378 次反量化里 28 个副本一直活着 ✓）。

    ## 为什么用「冻结参数」而不是逐个入口包 ``no_grad()`` ✗

    入口有 ``sample`` / ``sample_dual`` / ``retake_latents`` / ``encode`` / ``decode`` … 一串 ✓，
    逐个包就是「一句话手抄若干份」✓✗（漏一个就静默复发、且症状一模一样 ✓）。冻结参数是**根因**：
    参数不取梯度 ⇒ 任何入口、任何新加的入口，图都建不起来 ✓（单一真源 ✓）。

    ⚠️ 挂 LoRA 之后要**再冻一次** ✓：LoRA 的 A/B 是装载之后才 ``nn.Parameter`` 出来的 ✓，
    它们默认 ``requires_grad=True`` ✓；只冻底模 = LoRA 那几层照样建图 ✓✗。
    """
    if module is None:
        return 0
    try:
        module.eval()
    except Exception:  # noqa: BLE001 —— 不是 nn.Module 的东西没有 train/eval ✓ 不拦 ✗
        pass
    frozen = 0
    try:
        for parameter in module.parameters():
            if bool(getattr(parameter, "requires_grad", False)):
                parameter.requires_grad_(False)
                frozen += 1
    except Exception:  # noqa: BLE001
        return frozen
    return frozen


def _is_oom(exc: BaseException) -> bool:
    """这个异常是不是**显存不够** ✓✗（用于「降档再来一次」，别把真 bug 也吞掉 ✗）。

    判据取**最宽**的两种写法 ✓：``torch.cuda.OutOfMemoryError``（torch ≥ 1.13 ✓）与
    ``RuntimeError`` 里的 ``out of memory`` 字串 ✓（旧版 / ``torch.OutOfMemoryError`` 未导出时 ✓）。
    ⚠️ 刻意**不**在这里调 ``empty_cache`` ✗：那是「腾地方」不是「判类型」✓，混在一起会让
    "判定函数" 有副作用 ✓✗（调用点看不出卡上被清过 ✓）。
    """
    if exc.__class__.__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


#: VAE 解码省显存旋钮的**装配默认** ✓✗（2026-11-02 ✓）：两个**改数值**的关 ✓、
#: 只在 OOM 才生效的兜底开 ✓ —— 理由见 :meth:`TorchBackend.set_vae_tiling` ✓。
VAE_TILING_DEFAULTS: dict[str, Any] = {"tile": None, "halo": None, "attnChunk": None,
                                       "temporalChunk": None, "fallback": True}

#: 键的两个写法都认 ✓（``attn_chunk`` 是 Python 侧、``attnChunk`` 是回执/HTTP 侧 ✓）
#: ⇒ 调用方写哪个都不会「给了没生效」✓✗。
_VAE_TILING_ALIASES: dict[str, str] = {"tile": "tile", "halo": "halo",
                                       "attn_chunk": "attnChunk", "attnchunk": "attnChunk",
                                       "attnChunk": "attnChunk",
                                       "temporal_chunk": "temporalChunk",
                                       "temporalchunk": "temporalChunk",
                                       "temporalChunk": "temporalChunk",
                                       "fallback": "fallback"}

#: 报错时把「这个数是什么单位」说清楚 ✓✗ —— 一律写「像素」会把 ``temporalChunk``（**潜帧** ✓）
#: 也读成像素 ✓✗：``5`` 帧 vs ``5`` 像素 差着两个数量级 ✓。
_VAE_TILING_UNITS: dict[str, str] = {"tile": "像素（宽/高 ✓）", "halo": "像素（宽/高 ✓）",
                                     "attnChunk": "元素数（query 行 ✓）",
                                     "temporalChunk": "潜帧数（时间维 ✓）"}

#: 布尔的字面写法 ✓（**嵌套值到不了路由层的 ``_as_bool``** ✗ —— 那是给顶层开关用的 ✓，
#: 这里的值已经是 JSON 里掏出来的了 ⇒ 词表只能落在这儿 ✓，两处**语义必须一致** ✓）。
_TRUE_WORDS = ("1", "true", "yes", "on", "y", "t")


def normalize_vae_tiling(raw: Any) -> dict[str, Any]:
    """``None`` / 部分 dict → **规范化**的 VAE 解码旋钮 ✓✗（非法值当场 ``ValueError`` ✓）。

    这是**唯一**的归一入口 ✓：装配线（进装载键 ✓）、HTTP 白名单（转 400 ✓）、
    回执（回吐实值 ✓）都调它 ⇒ 三处的口径不可能分叉 ✓✗。

    规则：

    * ``None`` ⇒ :data:`VAE_TILING_DEFAULTS` 的副本 ✓（＝``tile`` / ``halo`` / ``attnChunk`` /
      ``temporalChunk`` 全关、OOM 兜底开 ✓）；
    * 数字 ``<= 0`` ⇒ **关** ✓（``0`` 不是「0 像素一块」✗ —— 与 ``vae_wan`` 的
      ``chunk <= 0`` 同一个意思 ✓）；
    * 认 ``attn_chunk`` 与 ``attnChunk`` 两种写法 ✓（别让「换了个写法 ⇒ 没生效」发生 ✗）；
      ``temporalChunk`` 同理认 snake / camel 三种写法 ✓；
    * **不认识的键 ⇒ 抛** ✗（``tile_size`` / ``attn_chunks`` 这类拼错的键静默忽略 ⇒
      调用方以为开了、其实全程逐位 ✓✗ —— 那是最难查的一种「没生效」✗）。
    """
    if raw is None:
        return dict(VAE_TILING_DEFAULTS)
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, (list, tuple)):
        #: 也收 ``[("tile", 512), …]`` ✓（``_vae_tiling_key`` 的快照形直接回灌 ✓，幂等 ✓）
        if not all(isinstance(x, (list, tuple)) and len(x) == 2 for x in raw):
            raise ValueError(f"`vaeTiling` 给成序列时必须形如 [[键, 值], …] ✗ 实收 {raw!r}")
        items = [tuple(x) for x in raw]  # type: ignore[misc]
    else:
        raise ValueError(
            f"`vaeTiling` 只收对象（形如 {{\"tile\": 512, \"attnChunk\": 1024}}）或不写 ✓，"
            f"实收 {type(raw).__name__} {raw!r} ✗ —— 想全关就写 {{}} 或显式给 null ✓"
            f"（**没有** ``true``/``false`` 这种写法 ✗：那会把「两个旋钮」和「OOM 兜底」"
            f"混成一件事 ✓✗）")

    out = dict(VAE_TILING_DEFAULTS)
    for key, value in items:
        canonical = _VAE_TILING_ALIASES.get(str(key))
        if canonical is None:
            raise ValueError(
                f"`vaeTiling` 里有不认识的键 {key!r} ✗ —— 只认 "
                f"{sorted(set(_VAE_TILING_ALIASES.values()))} ✓（拼错的键静默忽略 ⇒ "
                f"你以为开了、其实全程逐位 ✓✗）")
        if canonical == "fallback":
            if isinstance(value, bool):
                out[canonical] = value
            elif isinstance(value, (int, float)):
                out[canonical] = bool(value)
            elif isinstance(value, str) and value.strip():
                out[canonical] = value.strip().lower() in _TRUE_WORDS
            else:
                raise ValueError(f"`vaeTiling.fallback` 要布尔 ✓，实收 {value!r} ✗")
            continue
        if value is None:
            out[canonical] = None
            continue
        if isinstance(value, str):
            #: 字符串数字也收 ✓✗（同 ``residencyBudgetMib`` 那条口径 ✓：表单/查询串送来的是
            #: ``"768"`` ✓；而 ``"big"`` 仍然要**当场报** ✓ —— 不是"收下再说" ✗）。
            try:
                number = int(float(value.strip()))
            except (TypeError, ValueError):
                raise ValueError(
                    f"`vaeTiling.{canonical}` 要整数（{_VAE_TILING_UNITS.get(canonical, '整数')}），"
                    f"实收 {value!r} ✗") from None
            out[canonical] = number if number > 0 else None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"`vaeTiling.{canonical}` 要整数"
                             f"（{_VAE_TILING_UNITS.get(canonical, '整数')}），"
                             f"实收 {type(value).__name__} {value!r} ✗")
        number = int(value)
        out[canonical] = number if number > 0 else None
    return out


class TorchBackend:
    """真张量后端 ✓（``realTensors=True`` 恒真 ✓ / ``synthetic`` **按已装件算** ✗ —— 见模块头那张表 ✓）。"""

    name = "torch"
    #: 装到这些 ``vaeKind`` ⇒ 解出来的是**真画面** ✓（``reference-untrained`` / ``None`` ⇒ 合成 ✗）
    #: ⚠️ ``music3-real`` 解出来的是**真波形** ✓（不是画面 ✗）—— 但它确实是"真解码器" ✓，
    #: 所以照样入册 ✓；落盘侧靠 ``audioPath`` 区分 ✓（``write`` 的纯音频分支 ✓）。
    REAL_VAE_KINDS = ("h3-real", "flux-real", "wan21-real", "seedvr2-real", "music3-real")
    #: 张量是真的 ✓（与 :class:`app.services.engine.dryrun.DryRunBackend` 的本质区别 ✓）
    realTensors = True
    #: 条件/噪声的向量长度（占位前向用 ✓ —— 真模型接入后由架构决定 ✓）
    width = 512

    def __init__(self, device: str | None = None, *,
                 audio_latent_mode: str | None = None,
                 int8_offload: bool = False,
                 residency: str = "off",
                 residency_budget_mib: float | None = None,
                 gpu_layers: Any = None,
                 policy: str | None = None,
                 vae_tiling: Any = None,
                 profile_name: str = "auto",
                 profile_applied: dict[str, Any] | None = None) -> None:
        available, reason = torch_available()
        self._available = available
        self._reason = reason
        #: ⭐ **设备策略** ✓（``auto`` 默认 ✓ = GPU 优先、压力大才转移/联合 ✓ —— 见 `device_policy`）。
        #: ⚠️ 与 ``device`` 是**两层** ✗：``device`` 是"这次要谁"（可以是 ``"auto"`` ✓ 也可以显式钉住 ✓），
        #: ``policy`` 是"**允许**在压力下改主意吗" ✓✗。两个都显式给时 ``device`` 优先 ✓（见 `decide` ①）。
        self._policy = device_policy.normalize_policy(policy)
        #: 构造时**原始的**请求 ✓（``"auto"`` / ``None`` / ``"cpu"`` / ``"cuda"`` ✓）—— 决策要靠它区分
        #: 「用户钉住了 CPU ✓」与「默认优先 GPU 但压力大可以走 ✓」✗✗。
        self._requestedDevice = device
        self._device = (self._detect_device() if str(device or "").strip().lower() in ("", "auto")
                        else str(device))
        #: 最近一次决策 ✓ 与**逐阶段**的决策史 ✓（`describe()` 报出去 ✓ —— 静默改设备 = 用户以为在上卡 ✗✗）
        self._lastDecision: dict[str, Any] | None = None
        self._phaseDecisions: dict[str, dict[str, Any]] = {}
        #: **int8 常驻权重留在 CPU** ✓（``int8_offload`` ✓ —— 2026-10-29 加 ✓）。
        #:
        #: ⚠️ 为什么要有这一档 ✗：17.94 GiB int8 全常驻显卡后**只剩 2.87 GiB** 给激活 ✓✗，
        #: 而 720×1280×5s 的激活要十几 GiB ✓✗（三档实测见 `h3_form.place_module` 的注释 ✓）。
        #: ⇒ 这一档把 int8 钉在 **CPU 锁页内存** ✓、前向逐层搬到卡上 ✓（峰值 = **单层** ✓）。
        #: ⚠️ 默认 **False** ✓：CPU 那条（自检 / 无卡机）与"装得下"的大卡都该走老路 ✓，
        #: 只有真的 OOM 才值得付这趟 PCIe ✓ —— 默认打开会让**每台机器**都慢一截 ✓✗。
        self._int8Offload = bool(int8_offload)
        #: **卡 + CPU 联合跑** ✓（``residency`` ✓ —— 2026-10-31 加 ✓，创作者原话：
        #: 「以后不能只走 CPU，也要可以 GPU 和 CPU 联合跑模型，单个 CPU 很容易超载」✓）。
        #:
        #: ⚠️ 这一档补的是老路**唯一的那个洞** ✗：以前只有「全上卡」✓ 或「全在内存」✗ 两档 ✓✗，
        #: 而 24 GiB 卡装不下 17.94 GiB int8 + 十几 GiB 激活 ✓、纯 CPU 又要几十分钟/步 ✓✗
        #: ⇒ 按显存预算**切一刀** ✓：卡上放连续前缀、剩下的留在内存 ✓（两种搬法见 `residency` ✓）。
        #: ⚠️ 默认 **off** ✓：装得下的机器走老路 ✓（**不挂任何 hook、不多一次判断** ✓）；
        #: 联合驻留必然多付跨界搬运 ✓ ⇒ 只有"真装不下/被 CPU 拖死"时才值得开 ✓。
        #: 取值单一真源在 `residency.MODES` ✓（``off`` / ``split`` / ``park`` ✓）。
        self._residency = _residency_mode(residency)
        #: ⭐ **``gpuLayers`` 旋钮** ✓✗（2026-11-07 加 ✓）：``None`` ⇒ 没提，按预算自算 ✓；
        #: ``0`` ⇒ 整份留内存 ✓；``n`` ⇒ 前 n 层上卡 ✓（口径单一真源 = `residency.normalize_gpu_layers` ✓）。
        #:
        #: ⚠️ **给了数就等于要求联合驻留** ✗✗（``off`` 时自动开成 ``split`` ✓）：``residency=off``
        #: 那条路**根本不调 `residency.place`** ✓ ⇒ 旋钮会被静默吞掉 ✓✗ —— 而"给了参数不生效、
        #: 回执里又看不出来"是本仓最恨的一类坑 ✓。**开成 split 而不是 park** 的理由可算 ✓：
        #: 0 层时一样是"整份留内存" ✓（稳态相同 ✓），非 0 层时 park 每步都要整层来回搬 ✓（几十 GiB × 2 ✓）。
        self._gpuLayers: int | None = residency_mod.normalize_gpu_layers(gpu_layers)
        self._gpuLayersOpenedResidency = (self._gpuLayers is not None
                                         and self._residency == "off")
        if self._gpuLayersOpenedResidency:
            self._residency = "split"
        #: 显式显存预算 ✓（MiB ✓；None ⇒ 按**当时**空闲显存 ×(1−reserve) 现算 ✓ —— 见 `residency.budget_from_cuda`）
        self._residencyBudgetMib = _residency_budget_mib(residency_budget_mib)
        #: 联合驻留的把手 ✓（``place()`` 的返回值 ✓ —— 报告与 `detach()` 都靠它 ✓；off ⇒ None ✓）
        self._residencyHandle: Any = None
        #: ⭐ 这个把手是**用哪个 `what` 名摆的** ✓（2026-11-07 ✓）—— 见 `_WHAT_TO_PART` ✓：
        #: `release_part("text")` 要认得出「以 `what="te"` 摆的那一把」✓✗，
        #: 否则 `holder.module is self._textEncoder` 恒 False ⇒ **把手永远不摘** ✗✗。
        self._residencyOwnerWhat: str | None = None
        #: ⭐ **内存这一级**（宿主配额表 ✓ —— 2026-11-02 加 ✓）：**每件一本** ✓✗
        #: （``dit`` 与 ``te`` 的组名都叫 ``blocks.0`` ✓ —— 合成一本账会把一件的层当成另一件的
        #: 同一层卸掉 ✓，前向到那里就会撞上 0 元素占位 ✓）。值 ``None`` ⇒ 这台机器的内存读数
        #: 取不到 ⇒ 这一级**不启用** ✓（如实走老路，不假装有上限 ✓）。
        self._hostCaches: dict[str, Any] = {}
        #: ⭐ **机器档**（2026-11-07 加 ✓，见 :mod:`profile`）：档位展开出来的那几格**已经生效的值** ✓。
        #: ⚠️ 空 dict ＝ ``auto`` 档（＝一切照默认 ✓，**零变化** ✓✗）—— 不给每一项都填一个"默认值"
        #: （那会让报告和装载键都显得像配过 ✓✗，见 `profile.resolve` 的 ⚠️）。
        #: ⚠️ **在构造期定死** ✓（不提供运行时 `apply_profile` ✗✗）：档位进装载键 ✓ ⇒ 换档必重装 ✓，
        #: 一个实例不会中途被改档 ✓ —— 也就不存在「配额账本写着旧比例」那段坑 ✓。
        self._profileName: str = str(profile_name or "auto")
        self._profileApplied: dict[str, Any] = dict(profile_applied or {})
        #: ⭐ **任务账本** ✓（"这些字节现在是谁的" ✓）＋ **运行期压力状态机** ✓（带滞回 ✓）——
        #: 见 :mod:`scheduler` ✓。⚠️ 账本**不是判据** ✗：判据永远是**现测**读数（`device_policy.probe` ✓），
        #: 账本只回答归属 ✓（换装泄漏 / 忘记回收 / 两件重叠占卡，都从这看出来 ✓）。
        self._ledger = scheduler_mod.Ledger()
        self._watch = scheduler_mod.LoadWatch()
        #: 运行期压力事件 ✓（`tick()` 记 ✓，`describe()` 报 ✓）—— **静默降级**是这套最坏的失败面 ✗✗：
        #: 用户以为一路全速 ✓，实际早被挤到 CPU ✓。
        self._loadEvents: list[dict[str, Any]] = []
        #: 最近一次「整份转 CPU 前核内存」的回执 ✓（`decide_for` 写 ✓，`_apply_decision` 报 ✓）
        self._lastHostCheck: dict[str, Any] | None = None
        #: 装载成功后这里就是**真模型** ✓（None ⇒ 前向走占位 ✓，`describe()` 会如实说 ✓）
        self._model: Any = None
        self._config: Any = None
        #: 装的**是哪一族骨架** ✓（``h3`` / ``zimage`` / ``None`` ✓ —— 前向口径跟着它变 ✗，
        #: 光有 ``_form`` 不够：Z-Image 与通用 DiT 都是 ``form="dit"`` 但 σ 的形状不同 ✓✗）
        self._ditFamily: str | None = None
        self._loadReport: dict[str, Any] | None = None
        #: 参考 VAE ✓（挂了才解码出真帧 ✓）；未训练 ⇒ 画面是噪声 ✓（如实标 ✓）
        self._vae: Any = None
        self._vaeConfig: Any = None
        #: 这个 VAE 是**哪一族** ✓（``reference-untrained`` / ``h3-real`` ✓ —— 报告与 note 都读它 ✓；
        #: ⚠️ 不许从"挂上了"推"真权重"✗：两族的调用点**完全一样** ✓✗）
        self._vaeKind: str | None = None
        #: 反缩放事实 ✓（``(mean, std, family)`` ✓ —— 只有 ``wan21-real`` 用 ✓）。
        #: ⚠️ 归 **VAE 侧**而不是 DiT 侧 ✗：它是**解码**这一步的常数 ✓，DiT 吃的是归一化后的值 ✓。
        self._vaeLatents: tuple[tuple[float, ...], tuple[float, ...], str] | None = None
        #: ⭐ VAE 解码的**两个**省显存旋钮 ✓✗（2026-11-02 实测落地 ✓）：``tile``（切注意力
        #: **之后**的 up 链 ✓）＋ ``attn_chunk``（注意力按 query 分块 ✓）。**默认全关** ✓ ——
        #: 开任一都**改数值** ✓（tile 实测相对差 5.8e-03 ✓、attn 见 `AttentionBlock` 的表 ✓），
        #: 而本仓对「同 seed 同文本逐像素复现」有硬要求 ✓ ⇒ 由调用方 :meth:`set_vae_tiling`
        #: 明确开口 ✓，回执里如实报「开了哪个、为什么、不是逐位」✓。
        #: ⚠️ **整段按潜变量切块**这一条永远错 ✗：``middle.1`` 是全图单头注意力 ✓ ⇒ 切块后
        #: 注意力只看得到自己那块 ✓、拼回来**整张**都变（真 2.1 权重实测相对差 1.35 ✓，
        #: 不是接缝 ✓）。``fallback=True`` 只在 OOM 时才临时降档 ✓（那是"总比崩强" ✓ 不是优化 ✗）。
        #: ⚠️ 为什么 ``fallback`` **默认开**、而两个旋钮默认关 ✗✗：前者只在**本来要崩**的那条路上
        #: 生效 ✓（装得下时逐位不变 ✓ ⇒ 对「逐像素复现」零影响 ✓），后者一动就换字节 ✓。
        #: 归一在 :func:`normalize_vae_tiling` **一处** ✓✗（装配线、HTTP、回执三处共用 ✓）。
        self._vaeTiling: dict[str, Any] = normalize_vae_tiling(vae_tiling)
        #: 「这一档是**溢出自动降下来的**」✓ != 创作者点名要的 ✓✗ —— 两者必须分开 ✓：
        #: 自动降下来的还可以继续降 ✓，点名要的只回报不擅动 ✓（见 :meth:`TorchBackend.decode` ✓）。
        self._vaeAutoChunk: int | None = None
        #: 上一轮**溢出**过（reserved 超物理显存）⇒ 下一轮开跑前先把缓存块还回去 ✓✗：
        #: 那些块散在主机内存里 ✓，不还的话新的一轮还在原地颠簸 ✓（实测已降到 1 档却仍花 200 秒 ✗，
        #: 还回去之后是 8~10 秒 ✓）。
        self._vaeSpilled = False
        #: 音频那侧的同一件事 ✓（现在只有参考一族 ✓；写在这里是为了 `_decode_dual` 的 ``synthetic`` ✓
        #: 能**按事实算** ✓ 而不是写死 True ✗）
        self._audioVaeKind: str | None = None
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
        #: 文本塔**是哪一档** ✓：``None``（没挂 ✓）/ ``"reference-untrained"``（参考 ✓ 无数值语义 ✗）/
        #: ``"qwen3-te"``（**真** Qwen3-4B ✓）。⭐ 2026-10-31 补 ✗：原先 `encode_text` 把
        #: ``"reference-untrained"`` **写死**在返回里 ✓✗ ⇒ 挂上真文本塔之后仍然自述"未训练"✓，
        #: 调用方据此会以为条件没有语义 ✓✗（而它其实是真的 ✓）。
        self._textKind: str | None = None
        #: Qwen2.5-VL 的**视觉塔** ✓（Edit 2511 参考图的语义通路 ✓；挂在文本塔旁边 ✓，见
        #: :meth:`attach_qwen25vl_vision` ✓）。它与文本塔**同文件不同前缀** ✓（``visual.*`` ✓）。
        self._visionTower: Any = None
        self._visionShape: Any = None
        self._visionReport: dict[str, Any] | None = None
        #: ⭐ Edit 2511 的**参考图潜变量** ✓（由 :meth:`encode_text` 算出 ✓、:meth:`denoise`
        #: 每一步带进 DiT ✓）。放后端字段而不是回执里 ✓，是因为 `pipeline` 只把
        #: ``condition["positive"]`` 递给 `denoise` ✓ ⇒ 走回执就**只有第一步能拿到** ✓✗。
        #: ⚠️ 类型是**一列** ✓（单图 = 长度 1 ✓、多图 = N ✓）：统一成列表 ⇒ 两条路不用分叉 ✓。
        self._refLatents: Any = None
        #: 首帧条件**实际走了哪条路** ✓（真 VAE 编码 / 占位 ✓ —— 由管线回给调用方 ✓）
        self._conditioningNote: dict[str, Any] | None = None
        #: 超清放大器 ✓（按**需求装载** ✓ —— 没要超清就不装 ✓ 省显存）；装载报告与本次二采的实况 ✓
        self._upscaler: Any = None
        self._upscalerReport: dict[str, Any] | None = None
        self.refineDetails: dict[str, Any] | None = None
        #: 本实例**借来还没还**的 LoRA ✓（E2.5 ✓：借的是进程级 `lora_cache` 里那份共享张量 ✓，
        #: 同一个内容挂两次只占**一份**内存 ✓）；模型被换掉 / 卸载时必须 :meth:`release_loras` ✗✗
        self._loraLeases: list[Any] = []
        #: ⭐ 挂上去的**补丁钩子** ✓（2026-11-01 补 ✗✗）：LoRA 走 bypass 路时是**挂载**（
        #: ``register_forward_hook`` ✓）而不是折进权重 ✓ —— 归还租约只把**张量**还了 ✓，
        #: 钩子还赖在模块上 ✗ ⇒ 再挂一次就是「同一份增量叠两遍」✓✗（不报错、只是越来越"重" ✓）。
        #: 所以卸载要**两件一起还**：先摘钩子 ✓、再还租约 ✓。
        self._loraHooks: list[Any] = []
        #: 分时驻留的**链登记** ✓（``declare_chain`` 写 ✓、``enter_phase`` 读 ✓；``None`` ⇒ 不换装 ✓）。
        #: ⚠️ 只有「三件之和装不下」的族才登记 ✗（Qwen-Image 2.1 ✓ —— 账见 :attr:`PHASE_PARTS` ✓）。
        self._chain: dict[str, Any] | None = None
        #: ⭐ **装配期也走一次策略** ✓✗（2026-10-31 ✓）：老路的 `_detect_device()` 只问「有没有卡」✓，
        #: 卡被别人占满时照样判 `cuda` ✓✗ ⇒ 装配到一半 `CUDA out of memory` ✓，报错还落在很远的地方 ✓✗。
        #: 这里补的是"**有卡但卡上没地方**"这一档 ✓ —— 只在**没钉死设备**时做 ✓
        #: （显式 `device="cuda"` 是调用方的决定 ✓ 本模块不擅自改 ✗）。
        if str(self._requestedDevice or "").strip().lower() in ("", "auto"):
            self._apply_decision(self.decide_for(()))

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
            # 尾帧（fl2v ✓）同理 ✓ —— 同一个 ``cond`` 段机制 ✓，只是下标取 ``frames − 1`` ✓
            # （见 `_dual_keyframes` ✓）。管线对**没声明**它的后端会**明确拒绝**尾帧 ✗。
            "lastFrame": True,
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

    def _compute_dtype(self) -> Any:
        """计算精度**跟着 `self._device` 走** ✓✗ —— 不跟「这台机器有没有卡」走 ✗✗。

        ⚠️ 这两者在一台**有卡但显式选了 ``device="cpu"``** 的机器上会分叉 ✓✗：
        旧写法四处都按 ``torch.cuda.is_available()`` 选 bf16 ✓ ⇒ 明明跑在 CPU 上却把
        权重建成 bf16 ✓ —— CPU 上 bf16 的 ``conv2d``/``matmul`` 既慢又有算子未实现 ✓✗
        （VAE 解码那一步最容易撞上 ✓，报错还指向算子名、指不到"精度选错了" ✗）。
        反过来「没卡的机器」这一路本来就是 fp32 ✓ 不变 ✓。

        ⭐ 设备与精度必须是**一对** ✓：GPU / CPU **两条都要能跑** ✓（创作者 2026-10-31 的口径 ✓），
        所以凡是选 dtype 的地方一律走这个口 ✓（模块级构建函数接 `dtype=` 参数 ✓，见
        :func:`_build_zimage_module` / :func:`_build_qwen21_module` ✓）。
        """
        torch = self._torch()
        device = torch.device(self._device)
        return torch.bfloat16 if device.type == "cuda" else torch.float32

    def _torch(self) -> Any:
        """懒导入 ✓（模块级不 import ✓ ⇒ 没装 torch 也能 import 本模块 ✓）。"""
        import torch  # noqa: PLC0415

        return torch

    # ── 设备策略：GPU 优先 + 压力自适应（2026-10-31 ✓）────────────────────────────
    #: ⭐ **H3**（MiniMax 双流）三件套的驻留账 ✓ —— 与下面 :attr:`PART_NEED_GIB`（2.1 的）
    #: **必须分开** ✗✗（两张表差 11 GiB ✓✗）。逐格依据（哪份文件、多大、为什么 ×2 或被钉在
    #: int8 上）写在 ``runtime.H3_PART_NEED_GIB`` 名旁 ≈20 行表里 ✓ —— 那份名字现在是本表的**别名**
    #: （2026-11-07 收口 ✓：表值统一挂在类上，``STAGE_NEED_TABLES`` 才引用得到 ✓）。
    H3_PART_NEED_GIB: dict[str, float] = {"dit": 19.62, "text": 25.44, "vae": 5.2}

    #: 每个**分时部件**在 bf16 下的驻留量 ✓（GiB ✓）—— 这是 Qwen-Image 2.1 那三件的**实测账** ✓
    #: （DiT 13.26 ＋ TE 14.79 ＋ VAE 0.63 ＝ 28.68 ✓，见 :attr:`PHASE_PARTS` ✓）。
    #: ⚠️ 别的族**没核过** ✗ ⇒ 不在表里的部件按"需求未知"处理 ✓（`device_policy.decide` ⑤ 那条
    #: 用**占用率**粗判并更保守 ✓）—— 编一个数字进去会让决策看着精确、其实是猜 ✗✗。
    PART_NEED_GIB: dict[str, float] = {"dit": 13.26, "text": 14.79, "vae": 0.63}

    #: ⭐ **Qwen-Image 2512 / Edit 2511** 那一族的分时账 ✓（2026-10-31 加 ✓）——
    #: 与 :attr:`PART_NEED_GIB`（2.1 的 ✓）**必须分开** ✗✗：两张表差 **24.8 GiB** ✓✗。
    #:
    #: 三个数都是**盘上文件大小直接读出来的** ✓（不是估的 ✓）：
    #:
    #: | 部件 | 文件 | 盘上 | 驻留 | 为什么 |
    #: |---|---|---|---|---|
    #: | DiT | ``qwen_image_2512_bf16`` ✓ | **38.05 GiB** ✓ | **38.05** ✓ | 本来就是 bf16 ⇒ 1:1 ✓ |
    #: | DiT（Edit）| ``qwen_image_edit_2511_int8`` ✓ | **19.09 GiB** ✓ | **19.09** ✓ | 840 层 int8 **原样常驻** ✓ ⇒ 1:1（旧账写 ≈35.5 是"一律反量化成 bf16"时代的 ✗，已作废 ✓） |
    #: | 文本塔 | ``qwen_2.5_vl_7b_fp8_scaled`` ✓ | 8.74 GiB ✓ | ≈**16.1** ✓ | fp8 ⇒ ×2 ✓；且**只装语言塔** ✓（视觉塔 714 张量不建 ✗ ⇒ 略小于 ×2 ✓） |
    #: | VAE | ``qwen_image_vae`` ✓ | 0.24 GiB ✓ | **0.24** ✓ | 本体就是 bf16/fp32 权重 ✓ |
    #: | 视觉塔 | 同 ``qwen_2.5_vl_7b_fp8_scaled`` 的 ``visual.*`` ✓ | 8.74 GiB 里的 714 张量 ✓ | ≈**1.35** ✓ | **只在 Edit 链**的 encode 段在场 ✓（fp8 ⇒ ×2 ✓；装配后实测 390 真权重 ✓） |
    #:
    #: ⚠️ **只在 bf16 的 2512 这一份上装不满** ✗✗：DiT 单件 38.05 GiB > 卡 **22.49** ✓ ⇒ `auto` 会判
    #: `split`（按层切给卡＋内存 ✓，跨界前向慢 ✓）或 `cpu`（整件在内存 ✓，更慢 ✓）。**这是必付
    #: 成本** ✓，不是可省的优化 ✗ —— 报告里如实说 ✓，不许写"已优化" ✗。
    #: ⚠️ 拿 2.1 的 13.26 判它 ⇒ 结论"装得下" ✓✗、`residency` 留 `full` ✓、装到 20 GiB 才 OOM ✓。
    #: ⭐⭐ 表里的 ``dit`` 是 **bf16 那一份**的账 ✓✗（2026-11-01 ✓）：装配线不直接用这一格 ✗，
    #: 而是把它当 ``fallback`` 交给 :func:`resident_need_gib` ✓ —— 文件侧是 int8（Edit 2511 ✓）
    #: ⇒ 返回**数据区字节 19.09** ✓ ⇒ `off` 可用（19.71 ✓）⇒ 整份上卡 ✓。
    #: 这一格**必须留在 38.05** ✗✗：它是 bf16 那份的真驻留 ✓，调小 ⇒ 2512 装到 20 GiB 才 OOM ✓✗。
    QWEN_IMAGE_PART_NEED_GIB: dict[str, float] = {"dit": 38.05, "text": 16.1, "vae": 0.24,
                                                  "vision": 1.35}

    #: ⭐ **Krea2** 那一族的分时账 ✓（2026-11-01 加 ✓）—— 与上面两张表都**必须分开** ✗✗。
    #:
    #: | 部件 | 文件 | 盘上 | 驻留 | 为什么 |
    #: |---|---|---|---|---|
    #: | DiT | ``krea2_turbo_int8_convrot`` ✓ | 12.57 GiB ✓ | **12.57** ✓ | I8 12.155B ＋ BF16 0.665B ⇒ **常驻 int8** ⇒ 1:1 ✓ |
    #: | 文本塔 | ``Krea2-Engineer-V1-int8-convrot`` ✓ | 5.08 GiB ✓ | ≈**7.5** ✓ | int8 ⇒ 反量化 ×2 ✓，但**只装语言塔** ✓（视觉塔 0.415B 不建 ✗） |
    #: | VAE | ``qwen_image_vae`` ✓ | 0.24 GiB ✓ | **0.24** ✓ | 本体 bf16 ✓ |
    #:
    #: ⚠️ **DiT 这一格是这一族唯一的活路** ✓✗：12.82B 参数按 bf16 物化是 **23.9 GiB** ✓✗ >
    #: 卡 22.49 ✓ ⇒ 「照另三族那样 `to_empty` 按 bf16 建」**必装不下** ✓。所以这里记的是
    #: **int8 常驻之后**的 12.57 ✓ —— 而它成立的前提是 `_build_krea2_module` 真把层换成了
    #: ``Int8Linear`` ✓（少一层 ⇒ 那层 bf16 ⇒ 实际驻留 14.8 GiB ✓，账就虚了 ✓✗）。
    #: ⚠️ 拿 2.1 的 ``13.26`` 去判 Krea2 ⇒ 结论"差不多装得下" ✓，看着没错 ✓✗ ——
    #: 但拿 2512 的 ``38.05`` 判 ⇒ 结论"必然切 CPU" ✓，白扔掉 12 GiB 余量 ✓✗。
    KREA2_PART_NEED_GIB: dict[str, float] = {"dit": 12.57, "text": 7.5, "vae": 0.24}

    #: ⭐ **SEEDVR2**（视频修复 ✓）两件套的驻留账 ✓ —— 本仓**唯一一条两件同驻**的线 ✓。
    #:
    #: | 部件 | 文件 | 盘上 | 驻留 | 为什么 |
    #: |---|---|---|---|---|
    #: | DiT | ``seedvr2_3b_int8_convrot`` ✓ | 3.39 GiB ✓ | ≈**6.8** ✓ | int8 ⇒ **不挂常驻壳** ✗ ⇒ 反量化成 bf16 ⇒ ×2 ✓ |
    #: | VAE | ``ema_vae_fp16`` ✓ | 0.47 GiB ✓ | **0.47** ✓ | 本体 fp16 ⇒ 装成 bf16 同宽 ✓ |
    #:
    #: ⚠️ **DiT 这一格与 Krea2 相反** ✗✗：Krea2 的 12.82B 按 bf16 物化 23.9 GiB **装不下** ✓ ⇒
    #: 必须常驻 int8 ✓；而 SEEDVR2 是 **3.39B** ✓ ⇒ bf16 物化只要 6.8 ✓ ⇒ **不换壳更简单也更准** ✓
    #: （`dit_build` 里 seedvr2 那行**没有** ``adapter`` ✓ 就是这个意思 ✓）。所以这里记 **6.8** ✓、
    #: 文件字节按 3.39 记会少一半 ✓✗（症状＝判了 `off` 结果装到一半 OOM ✓）。
    #: ⚠️ 表值是**int8 档**量的 ✓；换成 ``w4a8`` / ``int4_convrot`` ⇒ 文件更小但**驻留不变** ✓（都反量化
    #: 成 bf16 ✓）⇒ `part_need_gib` 的精算会给出同一个 6.8 ✓（不信文件字节 ✓）。
    SEEDVR2_PART_NEED_GIB: dict[str, float] = {"dit": 6.8, "vae": 0.47}

    #: ⭐ **Wan 2.2 I2V**（第八条线 ✓）的**分时**驻留账 ✓ —— 与上面几张表都**必须分开** ✗✗。
    #:
    #: | 部件 | 文件 | 盘上 | 驻留 | 为什么 |
    #: |---|---|---|---|---|
    #: | DiT | ``wan2.2_i2v_high_noise_14B`` ✓ | 13.31 GiB ✓ | **26.62** ✓ | fp8_scaled 只省盘 ✓ ⇒ 反量化成 bf16 ×2 ✓ |
    #: | 文本塔 | ``umt5_xxl_fp8_e4m3fn_scaled`` ✓ | 6.27 GiB ✓ | **10.58** ✓ | 同上 ×2（数据区 10.59 ✓）⇒ 只建**编码器** ✓ |
    #: | VAE | ``wan_2.1_vae`` ✓ | 0.24 GiB ✓ | **0.24** ✓ | 本体 bf16 ✓（16ch / 8× / 4× 时间 ✓） |
    #:
    #: ⚠️⚠️ ``dit`` 记的是**一份专家** ✓✗：本线的两份（高噪 ＋ 低噪）**永不并存** ✓
    #: （`:meth:`register_wan_experts` 只登记路径 ✓，采样中途**原地重装** ✓）⇒ 把两格相加
    #: （53.24 ✓）当需求 ⇒ 结论"整卡都装不下、连 split 也没戏" ✓✗ —— 那不是保守 ✓，
    #: 是错误的悲观 ✓（会让 `decide_for` 选更大范围的 offload ✓、白慢一倍 ✓）。
    #: ⚠️ 但**单份 26.62 ＞ 卡 22.49** ✓✗ ⇒ 这一格的意义正相反 ✓：它保证 ``decide_for``
    #: **必须**给出 ``split`` ✓（给 ``off`` ⇔ 走到 ``sample`` 当场 OOM ✓，报在 ``module.to`` 上 ✓，
    #: 读起来像"卡被别的程序占了" ✗）。
    #: ⚠️ 表值是**fp8_scaled 文件 ＋ bf16 档**量的 ✓；换 ``int8_convrot``（Animate 2 ✓）或 bf16 本体
    #: ⇒ 文件更小/更大但**驻留不变** ✓（都反量化成 bf16 ✓）⇒ :func:`part_need_gib` 的精算会给出同样的数 ✓。
    WAN22_PART_NEED_GIB: dict[str, float] = {"dit": 26.62, "text": 10.58, "vae": 0.24}

    #: ⭐ **Music3**（音频族 ✓）三件套的驻留账 ✓ —— 2026-11-06 按**盘上真实文件字节**记 ✓。
    #:
    #: | 部件 | 文件 | 盘上 | 驻留 | 为什么 |
    #: |---|---|---|---|---|
    #: | DiT | ``minimax_music3_dit_fp16`` ✓ | 4.58 GiB ✓ | **4.58** ✓ | 本体 fp16 ⇒ 装成 bf16 **同宽** ✓（不 ×2 ✓✗） |
    #: | 语义塔 | ``minimax_music3_text_encoder_int8_convrot`` ✓ | 8.57 GiB ✓ | **8.57** ✓ | int8 ⇒ `load_tower(require_int8=True)` **常驻 int8** ✓ |
    #: | DAV | ``minimax_music3_dav`` ✓ | 0.20 GiB ✓ | **0.20** ✓ | 本体 fp16 ✓（512× 上采样 ＋ 44.1 kHz ✓） |
    #:
    #: 三件相加 **13.34 GiB** ✓ ＜ 卡 22.49 ⇒ 这一族**不需要分时链** ✓（与 Qwen-Image 2.1 相反 ✗✗）。
    #: ⚠️ 语义塔按 int8 记是对的 ✓，但**前提是**真装成了 int8 常驻 ✓：若哪天有人把 `require_int8=False`
    #: 打开 ⇒ 反量化成 bf16 ⇒ 那份变 **17.1 GiB** ✓，这张表就虚了 ✓✗（症状＝判了 `off`、
    #: 装到语义塔才 OOM ✓，报错落在文本塔上看着像"文本塔坏了" ✓）。
    #: ⚠️ DiT 这一格**不能**按"fp16 物化 ×2"记 ✗：fp16 与 bf16 都是 2 字节 ✓（只有 fp8/int8 才需要 ×2 ✓）。
    MUSIC3_PART_NEED_GIB: dict[str, float] = {"dit": 4.58, "text": 8.57, "vae": 0.20}

    #: **整条链同时驻留**的实测总账 ✓（GiB ✓）—— 与 :attr:`PART_NEED_GIB`（**分时**账 ✓）是两回事 ✗✗。
    #: ``zimage`` 那条线三件**同时在卡上** ✓✗（不像 2.1 那样 encode/sample/decode 各占一段 ✓）：
    #: DiT 11.46 ＋ Qwen3-4B TE 7.49 ＋ Flux VAE 0.33 ＝ **19.28** ✓，装配后 ``reserved`` 实测
    #: **19.39 GiB** ✓（差的 0.11 是 allocator 保留块 ✓）。
    #: ⚠️ 为什么需要这张表 ✗✗：光靠「有没有卡」判，卡上剩 10 GiB 也照样敢装 ✓✗ —— DiT 装得进去 ✓、
    #: 装 TE 时才 OOM ✓，报错落在 TE 上，看着像"文本塔坏了" ✓✗（实测过一次这种误导 ✓）。
    #: ⚠️ 没核过的族**不进表** ✗：查不到实测就传 ``0``（走占用率粗判 ✓），不编数字 ✗✗。
    CHAIN_NEED_GIB: dict[str, float] = {"zimage": 19.28}

    #: **整链账的分件明细** ✓（GiB ✓）—— 与 :attr:`CHAIN_NEED_GIB` 是同一套实测的两种粒度 ✓。
    #: ⚠️ 为什么需要拆开 ✗✗（2026-11-02 ✓）：整链 19.28 只在「三份文件都是 bf16」时成立 ✓ ——
    #: 三件里任意一件换了量化档（例如 DiT 换 int8 ⇒ 反量化后 11.46 变不了 ✓，但换 fp8 ⇒ ×2 ✓）
    #: 整链数就不成立 ✓✗，而只拿总和判又会把差额记到"最后装的那件"头上 ✓（报错误导 ✓）。
    #: 分件表的每一格都必须与 :attr:`CHAIN_NEED_GIB` **对得上**（11.46+7.49+0.33=19.28 ✓）。
    CHAIN_NEED_PARTS_GIB: dict[str, dict[str, float]] = {
        "zimage": {"dit": 11.46, "text": 7.49, "vae": 0.33},
    }

    #: ⭐ 档 ⇒ 该档的**逐件表值** ✓（GiB ✓）—— **引用**上面几本账、不复制数字 ✓✗（2026-11-07 ✓）。
    #:
    #: 为什么需要这张索引 ✗✗：上面每一本账都叫 ``*_PART_NEED_GIB``，但**档名不一样**
    #: （``qwen_image21`` 用 2.1 那本 ✓、``qwen_image`` 用 2512/Edit 那本 ✓、``image`` 干脆没表 ✓）。
    #: 预检（``feasibility``）要「不装载就答同一个数」✓，就得知道**这个档该翻哪一本** ✓——
    #: 让预检自己写一份 ``if stage == ...`` 就是第二本账 ✓✗（本仓踩过两次：Krea2 按 int8 量、
    #: Qwen-Image 2511 按 38.05 判 ✓）。
    #:
    #: ⚠️ 值是**同一批 dict 对象** ✓（``is`` 相等 ✓，见 ``engine_feasibility_test`` ③）——
    #: 也就是说改一本账，预检立刻跟着变 ✓；库里没有第二份数字 ✓✗。
    #: ⚠️ ``image``（SDXL）**不在表里** ✓：单 checkpoint、驻留＝文件字节 ✓ ⇒ 调用方显式给
    #: ``fallback=os.stat(...)`` ✓（不是"漏了" ✗）。
    STAGE_NEED_TABLES: dict[str, dict[str, float]] = {
        "h3": H3_PART_NEED_GIB,
        "zimage": CHAIN_NEED_PARTS_GIB["zimage"],
        "qwen_image21": PART_NEED_GIB,
        "krea2": KREA2_PART_NEED_GIB,
        "seedvr2": SEEDVR2_PART_NEED_GIB,
        "music3": MUSIC3_PART_NEED_GIB,
        "wan22": WAN22_PART_NEED_GIB,
        "qwen_image": QWEN_IMAGE_PART_NEED_GIB,
    }

    def gpu_pressure(self) -> dict[str, Any] | None:
        """现测一次卡况 ✓（``None`` ⇒ 没卡/取不到 ✓ —— 不猜 ✗）。

        ⚠️ 每次装载/换阶段**都要重新测** ✗✗：卡上有多少余量是**会变的** ✓（别的进程、本进程上一件
        还没退干净 ✓）⇒ 拿构造时那次读数做决定，就是"用旧账判新事" ✓✗。
        """
        #: ⚠️ 判「在不在卡上」**只许**走 `device_policy.is_cuda_device` ✓✗（此前这里写
        #: `startswith("cuda")`、`_place_with_residency` 写 `split(":")[0]`、`describe()` 写
        #: `== "cuda"` ✓ —— 三份手抄，第三份在 `cuda:0` 上答错 ✓✗）。
        if not device_policy.is_cuda_device(self._device):
            return None
        read = device_policy.probe(self._torch() if self._available else None)
        return None if read is None else read.to_dict()

    def decide_for(self, parts: tuple[str, ...], *, phase: str | None = None,
                   need_gib: dict[str, float] | None = None
            ) -> device_policy.Decision:
        """这**一批部件**该放哪 ✓（``parts`` = 同一阶段要装的那几件 ✓）。

        需求按 :attr:`PART_NEED_GIB` 相加 ✓；任一件不在表里 ⇒ 整批按"需求未知"处理 ✓（保守 ✓）。

        ⚠️ ``need_gib`` 是**按族覆盖**这一格 ✓✗（2026-10-31 加 ✓）：:attr:`PART_NEED_GIB` 是
        **2.1 那一族的账** ✓（DiT 13.26 / TE 14.79 / VAE 0.63 ✓）。2512 的 DiT 是 **20B / bf16**
        ⇒ 驻留 **35.4 GiB** ✓✗ —— 拿 2.1 的 13.26 去判，结论会是"装得下" ✓✗，`residency` 留在
        `full` ✓，真装上时 OOM 才炸 ✓（这次恰好在 DiT 上炸，看着像"权重坏了" ✗）。
        ``declare_chain`` 把本族的账一起登记 ✓、这里按它判 ✓。
        """
        table = need_gib if need_gib else self.PART_NEED_GIB
        need_bytes = 0
        for part in parts:
            gib = table.get(part)
            if gib is None:
                need_bytes = 0
                break
            need_bytes += int(float(gib) * device_policy.GIB)
        #: ⚠️ **无论当前在哪个设备上都探** ✓✗：上一阶段被降级到 CPU 之后，卡可能已经空出来了 ✓
        #: ⇒ 这一阶段就该**回到 GPU** ✓（只在上卡时才探的话，一旦降级就再也回不去 ✓✗）。
        #: ⚠️⚠️ 探之前**先把 allocator 的空闲块还给驱动** ✓✗（2026-11-01 端到端实证 ✓）：
        #: 不还，量到的是「上一件留下的缓存块仍占着卡」⇒ 假的紧 ✓ ⇒ 这一件被白切掉一大半 ✓✗。
        #: 实测差多少：`enter_phase("encode")` 那一步 `reserved` 比 `allocated` 多 **14.3 GiB** ✓，
        #: 于是 probe 报「余 4.92 GiB」✓（真实可用 19.2 GiB ✓）⇒ 25.28 GiB 的塔只上卡 3.5 GiB ✓✗。
        self._trim_cuda()
        read = device_policy.probe(self._torch()) if self._available else None
        decision = device_policy.decide(
            policy=self._policy, requested_device=self._requestedDevice,
            need_bytes=need_bytes, pressure=read,
            gpu_available=bool(self._available and self._torch().cuda.is_available()),
            phase=phase, part=",".join(parts) or None)
        #: ⭐ **CPU 侧兜底** ✓✗（2026-11-01 加 ✓ —— 见 :mod:`scheduler` 模块头的洞 ①）：
        #: `decide()` 只看**显存** ✓，判「整份转 CPU」时不知道内存还剩多少 ✗。
        #: 32B 文本塔那条实测：宿主吃到 **74/93 GiB** ✓、系统只剩 6 GiB ✓、进程在页错误里停
        #: 十几分钟 ✓、连权重文件都没开始读 ✓✗。⇒ 真转 CPU 之前核一次 ✓，放不下就当场拒 ✓✗
        #: （交给交换文件硬跑，只会在十几分钟后于别处报一个看不出根因的错 ✓）。
        #: ⚠️ 只管 ``device == "cpu"`` 这一档 ✗：``split`` 的内存那部分由 `residency.place(guard=)`
        #: 在**切完之后、搬之前**核 ✓（那时才知道要留多少在内存里 ✓）。
        self._lastHostCheck = None
        if need_bytes and decision.device == "cpu" and decision.residency == "off":
            self._lastHostCheck = scheduler_mod.ensure_host_room(
                need_bytes, what=f"{','.join(parts) or '?'}（phase={phase or '-'}）")
        return decision

    def _apply_decision(self, decision: device_policy.Decision) -> None:
        """把决策**落到这台后端上** ✓（换设备 / 换驻留方式 ✓）—— 并留痕 ✓。

        ⚠️ 只动这两样 ✗：步数、分辨率、精度以外的任何创作参数**一律不碰** ✓✗
        （降级=换设备与换驻留方式 ✓，见 `device_policy` 模块头 ✓）。
        """
        previous_device, previous_residency = self._device, self._residency
        if decision.device != self._device:
            #: ⚠️ 切换必须在**该件已全部退掉之后** ✓（`enter_phase` 的调用顺序保证 ✓）——
            #: 两件跨设备共存时显存账说不清，还会让旧的 `residencyHandle` 指向已换设备的参数 ✗。
            #: ⚠️ 置 None **必须**走 `_drop_residency_handle()` ✓✗：直接赋 None 会留下 hook 环 ✓
            #: ⇒ 旧模块收不掉 ✓ ⇒ 换设备那一步的显存账直接是假的 ✓。
            self._drop_residency_handle()
            self._device = decision.device
        if decision.residency in residency_mod.MODES and decision.residency != self._residency:
            self._residency = decision.residency
        payload = decision.to_dict()
        payload["changed"] = {
            "device": [previous_device, self._device],
            "residency": [previous_residency, self._residency],
        }
        #: 「转 CPU 之前核过内存没有」✓ —— 核过就报读数 ✓、取不到就如实说没核 ✓（不假装 ✓✗）
        if self._lastHostCheck is not None:
            payload["hostRoom"] = self._lastHostCheck
        self._lastDecision = payload
        if decision.phase:
            self._phaseDecisions[decision.phase] = payload

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

    def _place_with_residency(self, module: Any, *, what: str) -> dict[str, Any] | None:
        """按 ``self._residency`` 把 ``module`` 的权重**摆到卡 + CPU 两处** ✓ ⇒ 报告 ✓。

        * ``off`` ⇒ **返回 None、一根 hook 都不挂** ✓（老路一字未变 ✓，零开销 ✓）；
        * ``split`` / ``park`` ⇒ 调 `residency.place()` ✓（切分数学与 hook 全在那边 ✓，
          本方法只管「什么时候摆 / 报告怎么写」✓）。

        ⚠️ **必须在 `load_module_weights` 之前调** ✗✗：装载走 ``param.copy_(...)`` ✓ ⇒ 权重会落在
        **参数当前所在的设备** ✓ —— 先摆好再装，装载期的峰值显存就只是"卡上那一刀" ✓；
        反过来（全装完再摆）峰值仍是整份 ✗✗ —— 那正好是这次要治的 OOM ✓。

        ⚠️ **设备本来就不是 CUDA ⇒ 不做联合** ✓：没有第二台设备可"联" ✓，整份在内存里就是答案 ✓
        （**如实说，不假装省了显存** ✗ —— 与 `int8Offload` 在 CPU 上的那条口径一致 ✓）。

        ⚠️ 切完之后**退化成纯 CPU** 的情况也要报 ✗：预算比"非层参数"还小时 `plan.fits=False` ✓
        ⇒ 装载照样成功 ✓、但前向全在内存里爬 ✓✗ —— 报告里 ``fits=False`` + ``reason`` 必须看得见 ✓
        （静默退化 = 用户以为开了联合、实际比不切还慢 ✓✗）。

        ⭐ ``what`` 有两个用途，**别再合成一个** ✗✗（2026-11-01 ✓）：进报告的名字是它本身 ✓
        （``"dit"`` / ``"h3"`` / ``"h3-te"`` ✓ —— 读者要能分清是哪一族哪一件 ✓），而**余量**只认
        件类型三档 ✓ ⇒ 中间过一次 :func:`_residency_kind` ✓（见那里的归一表 ✓）。
        """
        if self._residency == "off":
            return None
        if not device_policy.is_cuda_device(self._device):
            return {
                "mode": self._residency, "applied": False, "what": what,
                "device": str(self._device), "budgetMib": self._residencyBudgetMib,
                "note": f"后端设备是 **{self._device}** ⇒ 没有第二台设备可联合 ✓（整份在内存 ✓，"
                        "`residency` 不生效 ✗ —— **不是失败**，但也没省到显存 ✓）",
            }
        #: ⚠️ ``place()`` 的默认预算口径是「**此刻**空闲显存」✓✗（见 `budgetSource="cuda-free"` ✓）——
        #: 那一刻 allocator 里若还压着上一件留下的空闲块 ✓，算出来的预算就是假的 ✓
        #: ⇒ 先还再算 ✓（与 `decide_for` 是同一句 `_trim_cuda()` ✓，不是第二套口径 ✓）。
        self._trim_cuda()
        budget = None if self._residencyBudgetMib is None else int(self._residencyBudgetMib * 2**20)
        #: ⭐ 余量**按件**给 ✓✗（2026-11-01 ✓，见 `residency.RESERVE_BY_KIND` ✓）：``what``
        #: 就是 ``"dit"`` / ``"te"`` ✓（调用点走 `_place_before_load`，只有这两个值 ✓）。
        #: ⚠️ 这里**只**在"没显式给预算"时才生效 ✗：``residencyBudgetMib`` 是调用方的决定 ✓，
        #: 按件猜的余量不许覆盖它 ✓✗。
        #: ⭐ 内存这一级 ✓✗（2026-11-02 ✓）：落到 CPU 侧的层要过**宿主配额** ✓，
        #: 装不下的卸到盘上 ✓（`:mod:`host_cache`` 的模块头讲了为什么必须补这一级 ✓）。
        #: ⚠️ **每件一个** HostCache ✗：``dit`` 与 ``te`` 的组名都叫 ``blocks.0`` ✓✗ ——
        #: 合成一本账会把两件的层当成同一层互相卸掉 ✓（配额是每件按总内存算的 ✓，
        #: 真装不下时由 `_host_guard` 那道闸兜底 ✓）。
        handle = residency_mod.place(module, main_device=self._device, budget=budget,
                                     mode=self._residency, kind=_residency_kind(what),
                                     gpu_layers=self._gpuLayers,
                                     guard=self._host_guard(what),
                                     host_cache=self._host_cache_of(what),
                                     reload=residency_mod.ReloadSlot(),
                                     foresight=_foresight_of(self._residency))
        self._residencyHandle = handle
        #: ⭐ 记下这一把是**用哪个名字**摆的 ✓（`release_part` 靠它认件 ✓，见 `_WHAT_TO_PART` ✓）
        self._residencyOwnerWhat = str(what)
        #: 记账 ✓（**切完才记** ✓ —— 记的是"实际留在内存的那份" ✓，不是估算 ✓）
        self._ledger.reserve(what, bytes_=int(handle.plan.gpu_bytes), device=str(self._device),
                             residency=self._residency,
                             note=f"卡上 {handle.plan.layers_on_gpu} 层 / 内存放 "
                                  f"{handle.plan.cpu_bytes / device_policy.GIB:.2f} GiB")
        self._ledger.reserve(f"{what}#host", bytes_=int(handle.plan.cpu_bytes), device="cpu",
                             residency=self._residency, note="联合驻留的 CPU 那一半")
        report = dict(handle.report())
        #: ⚠️ ``budget <= 0`` 是 `place()` 里"没有可用显存"那一档 ✓ ⇒ **没联合** ✗（与 `fits` 是两回事 ✓）：
        #: 显式写个 0、或现算出来是 0（卡被占满 ✓）时，报 ``applied=True`` 会让人以为切过了 ✓✗。
        applied = handle.plan.budget > 0
        report.update({
            "mode": self._residency, "applied": applied, "what": what,
            "device": str(self._device),
            "budgetMib": round(handle.plan.budget / 2**20, 1),
            "budgetSource": "explicit" if budget is not None else "cuda-free",
            #: ⭐ 这次按多少余量算的 ✓✗（``None`` ⇒ 显式预算，没走这条 ✓）—— 2026-11-01 加 ✓：
            #: 25.44 GiB 的塔只上卡 61% 时，第一个要看的数就是它 ✓（不是预算、不是层数 ✓）。
            "reserve": None if budget is not None else residency_mod.reserve_for(
                _residency_kind(what)),
            #: 余量档认的是哪一类件 ✓（``None`` ⇒ 没归一出件类型 ⇒ 退回 25% ✓，见 `_residency_kind`）
            "reserveKind": _residency_kind(what),
            # ⭐ 「切完还剩多少在卡上」是**判断有没有意义**的那个数 ✓（0 层 ⇒ 等于纯 CPU ✗）
            "fits": handle.plan.fits,
            #: ⭐ 给了 ``gpuLayers`` 就得在回执里**看见它** ✓✗（2026-11-07 ✓）——
            #: 尤其是"给了数、但这台机器没有第二台设备"那一路 ✓（见上面的 ``return`` ✓）。
            "gpuLayers": handle.plan.gpuLayers,
            #: 这一跑的联合驻留是**被 gpuLayers 顶开的** ✓（``residency=off`` ＋ 给了数 ✓）——
            #: 不写这一句，用户会以为自己填的 ``residency`` 生效了 ✓✗。
            "residencyOpenedBy": "gpuLayers" if self._gpuLayersOpenedResidency else None,
        })
        #: ⭐ **GPU 承担了多少**（按字节 ✓）—— 2026-11-01 加 ✓：用户问"为什么大部分计算在 CPU 上"时，
        #: 唯一能直接回答的数 ✓（层数占比会骗人 ✗：卡上 41% 的层也可能只占 41% 的字节、
        #: 却吃掉 95% 的时间 ✗）。分母为 0 ⇒ 不给（不编一个 1.0 ✗）。
        total = int(handle.plan.gpu_bytes) + int(handle.plan.cpu_bytes)
        report["gpuShare"] = (round(int(handle.plan.gpu_bytes) / total, 4) if total > 0 else None)
        layers_note = self._gpu_layers_note(handle.plan)
        if layers_note:
            report["note"] = layers_note
        elif not applied:
            report["note"] = (
                f"预算 **{report['budgetMib']} MiB** ⇒ 没有可切的显存 ✓（`{handle.plan.reason}`）"
                "⇒ 整份留在内存里 ✓✗（**等于纯 CPU** ✓ —— 装载成功、但这一档没联合 ✗；"
                "卡被别人占着就等它空出来 ✓，或显式给 `residencyBudgetMib` ✓）")
        elif not handle.plan.fits:
            #: ⭐ 2026-11-02：**这笔预算被谁花掉了** ✓✗ —— 只说"预算 16.9 GiB 不够"没法行动 ✗
            detail = report.get("budget") or {}
            spent = ""
            if detail.get("marginGiB") is not None:
                spent = (f"（可用 {detail['usableGiB']} GiB − 余量 {detail['marginGiB']} GiB"
                         f" − 钉死 {detail['fixedGiB']} GiB ＝ **{detail['effectiveGiB']} GiB** ✓）")
            report["note"] = (
                f"⚠️ 预算（{report['budgetMib']} MiB{spent}）连**非层参数**都放不下 ✗"
                f" ⇒ 整份回落到 CPU ✓✗"
                f"（`plan.reason`：{handle.plan.reason}）—— 装载**成功** ✓，但这一步**没有联合** ✗，"
                "前向会全在内存里跑 ✓（余量占大头 ⇒ 换 `kind` 档 ✓；"
                "钉死那块占大头 ⇒ 少钉一点 ✓；两者都不大 ⇒ 这件真比卡大 ✓，换更小的档位 ✓）")
        elif handle.plan.layers_on_cpu and self._residency == "split":
            report["note"] = (
                f"卡上 {handle.plan.layers_on_gpu} 层 / CPU {handle.plan.layers_on_cpu} 层 ✓ ⇒ "
                "跨界只搬**激活** ✓（每步一次，代价小 ✓）；CPU 那几层在计算核上 ✓ 会比全卡慢 ✓")
        return report

    def _gpu_layers_note(self, plan_: Any) -> str:
        """``gpuLayers`` 那一路的**人话** ✓✗（2026-11-07 ✓）—— 没给过数就返回 ``""`` ✓。

        ⚠️ 为什么优先级**高于**其余几条 ✗✗：给过数的那一跑，"卡上几层"是**用户的决定** ✓
        （不是预算算出来的 ✓）—— 还去解释"预算怎么花的"会把人引向错误的旋钮 ✓。
        """
        if self._gpuLayers is None:
            return ""
        got = plan_.grantedLayers
        opened = "（`residency` 由 `gpuLayers` 自动打开为 `split` ✓ —— 不然这个旋钮无处生效 ✗）" \
            if self._gpuLayersOpenedResidency else ""
        if self._gpuLayers == 0:
            return (f"`gpuLayers=0` ⇒ **整份留内存** ✓（一层都不上卡 ✓，等同 `park` 的稳态 ✓；"
                    f"**不是**装载失败 ✗ —— 想上卡就给个正整数 ✓）{opened}")
        if got is None:
            return f"`gpuLayers={self._gpuLayers}` 没生效（回执没给 granted ✓）—— 这条不该出现，请报 bug ✗"
        if plan_.capped:
            return (f"⚠️ `gpuLayers={self._gpuLayers}` **没给满** ✗：预算只装得下 {got} 层 ✓"
                    f"（卡上 {plan_.gpu_bytes / 2**30:.2f} GiB ✓）⇒ 按 {got} 层切 ✓、"
                    "剩余层在内存里 ✓（**不是**退回 auto ✗ —— 切法仍是「从前往后按你给的数取」✓；"
                    "要真装满 ⇒ 加 `residencyBudgetMib` ✓ 或换更小的件 ✓）" + opened)
        return (f"`gpuLayers={self._gpuLayers}` ⇒ 卡上 {got} 层 ✓（"
                f"{plan_.gpu_bytes / 2**30:.2f} GiB ✓）"
                + ("，其余在内存里 ✓（跨界只搬激活 ✓）" if plan_.layers_on_cpu else "，全上卡 ✓")
                + opened)

    def _host_guard(self, what: str) -> Any:
        """给 ``residency.place(guard=)`` 的最后一道闸 ✓：**切完、搬之前**核内存放不放得下 ✓✗。

        ⚠️ 为什么卡在这一步 ✗：切分结论（卡上几层 / 内存几层 ✓）到 ``plan`` 出来才知道 ✓ ——
        在 `place()` 之前估只能瞎猜 ✓✗。放不下 ⇒ 抛 :class:`scheduler.InsufficientResources`
        ⇒ **装载中止** ✓（不许"先搬进去再说" ✗：搬完就撞 OOM，报错还指向别处 ✓）。
        """
        def _guard(plan_: Any) -> None:
            need = int(getattr(plan_, "cpu_bytes", 0) or 0)
            if need <= 0:
                return
            result = scheduler_mod.ensure_host_room(need, what=f"{what} 的 CPU 那一半")
            self._record_event("host-guard", {"what": what, **result})
        return _guard

    def profile(self) -> dict[str, Any]:
        """机器档的**实际生效面** ✓（``readout()`` / 回执用 ✓）。

        ``applied`` 为空 ⇒ ``auto`` 档 ✓（``source="default"`` ✓ —— 别在报告里写成"已按 L1 压过配额" ✗）。
        """
        return {"profile": self._profileName, "applied": dict(self._profileApplied),
                "source": "profile" if self._profileApplied else "default",
                "hostRatio": self._profileApplied.get("hostRatio"),
                "note": ("机器档只改**配额上限**与**解码分时** ✓；不换权重、不改像素 ✓"
                         "（开 `tile` 才换字节 ✗，档位不碰它 ✓）")}

    def _host_cache_of(self, what: str) -> Any:
        """按件取**宿主层配额表** ✓ ⇒ :class:`host_cache.HostCache` 或 ``None``（这一级不启用 ✓）。

        ⚠️ **每件一本** ✗✗：``dit`` 与 ``te`` 的组名都是 ``blocks.0`` ✓ —— 合成一本账时
        "装不下就卸 LRU"会去卸**另一件**的层 ✓，而那件的 hook 还会去 `ensure` 它 ✓✗
        ⇒ 症状是采样第 1 步报「0 元素占位、形状对不上」✓，查起来像权重坏了 ✓。

        ⚠️ 配额取 ``min(quota_of(总内存), host_room_bytes())`` ✓✗ —— **两个数各管一头**：
        :func:`host_cache.quota_of` 说"这台机器最多能拿多少给权重"（总内存 × 75% ✓，
        留着给激活 / 碎片 / Python 本体 ✓）；:func:`scheduler.host_room_bytes` 说"**此刻**
        还剩多少"（已经被别的进程或别件吃掉的不能算 ✓）。只取前者 ⇒ 在一台还跑着别的东西的
        机器上照样往内存里塞 ✓✗；只取后者 ⇒ 上限随一次采样抖动 ✓✗（同一件两次装载两套账 ✓）。
        ★真到了那一刻的兜底仍是 :meth:`_host_guard`（装载前**现测** ✓）—— 这里只是把上限
        钉低一点 ✓，**不是**第二道判据 ✗。

        ⚠️ 读数取不到 ⇒ ``None`` ✓（`:func:`scheduler.host_memory` 只说 ``None`` 表示测不到，
        不猜一个数 ✗）—— 没有配额就没有"该卸谁"的判据 ✓，宁可不启用也不拿假数卸层 ✓✗。
        """
        if what in self._hostCaches:
            return self._hostCaches[what]
        cache: Any = None
        host = scheduler_mod.host_memory()
        if host is not None:
            total = int(host[1])
            room = scheduler_mod.host_room_bytes(host)
            #: ⚠️ 比例**只有**机器档能改 ✓✗（``l1`` → 0.60、``l2`` → 0.50 ✓；``auto``/``gpu`` ⇒
            #: 落回 :data:`host_cache.HOST_USE_RATIO` ✓，老路**零变化** ✓）。档位表里不抄这个数 ✓。
            ratio = self._profileApplied.get("hostRatio")
            quota = host_cache_mod.quota_of(total, **({} if ratio is None else {"ratio": ratio}))
            if room is not None:
                quota = min(quota, max(0, int(room)))
            cache = host_cache_mod.HostCache(quota=quota, name=what)
            self._record_event("host-cache", {
                "what": what, "quotaGiB": round(quota / device_policy.GIB, 2),
                "totalGiB": round(total / device_policy.GIB, 2),
                #: ⚠️ 比例也要报 ✓✗：配额是个 GiB 数 ✓，只看数字分不清"是这台机器内存小"还是
                #: "档位把比例压下来了" ✓✗（两者对用户的动作完全不同 ✓）。
                "ratio": ratio if ratio is not None else host_cache_mod.HOST_USE_RATIO,
                "ratioSource": "profile" if ratio is not None else "default",
                "roomGiB": (None if room is None
                            else round(int(room) / device_policy.GIB, 2)),
                "note": ("内存这一级启用了 ✓：CPU 那几层超出配额就卸到盘上、"
                         "前向到那一层再读回来 ✓（不再因为装不下而中止装载 ✓）")})
        self._hostCaches[what] = cache
        return cache

    def _record_event(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """记一条调度事件 ✓（**限长** ✓：运行期每步都可能探 ✓，无上限会自己把内存吃光 ✗✗）。"""
        event = {"kind": kind, "n": len(self._loadEvents) + 1, **payload}
        self._loadEvents.append(event)
        if len(self._loadEvents) > 200:
            del self._loadEvents[:100]
        return event

    def tick(self, phase: str = "run") -> dict[str, Any] | None:
        """运行期**一次探压** ✓（带滞回 ✓）⇒ 到硬阈值才**回收**本阶段用不着的件 ✓。

        ⚠️ 只做**回收**（退掉不该在场的件 ✓），**不做**"搬来搬去"的优化 ✗✗：
        搬一次 14 GiB 的代价远大于它省下的显存 ✓，而且一步一搬就是抖动的定义 ✓▸
        真正换装该发生在阶段边界（`enter_phase` 的「退 → 测 → 定 → 装」✓）。

        ⚠️ 返回 ``None`` ＝ **这一拍什么都没发生** ✓（不要把 None 当"没压力" ✗：读数取不到时
        `LoadWatch` 也返回 None ✓，两者在报告里由 ``watch.state`` / ``source`` 分开 ✓✗）。
        """
        if not self._available:
            return None
        pressure = None
        if device_policy.is_cuda_device(self._device):
            try:
                pressure = device_policy.probe(self._torch())
            except Exception:  # noqa: BLE001 - 探压失败不该让任务挂 ✓（如实说测不到 ✓）
                pressure = None
        previous = self._watch.state
        state = self._watch.sample(pressure)
        if state != "hard":
            if state != previous:
                self._record_event("pressure", {"phase": phase, **self._watch.to_dict()})
            return None
        #: ⚠️ 阶段表没登记这一拍 ⇒ **不乱退** ✓✗（猜"哪件用不着"会把正在用的那件卸掉 ✓✗；
        #: 宁可只报压力 ✓ —— 用户能看到 `watch.state == "hard"` ✓，而不会被一个静默卸载坑到 ✓）。
        wanted: tuple[str, ...] | None = None
        if self._chain is not None:
            wanted = (self._chain.get("phaseParts") or self.PHASE_PARTS).get(phase)
            #: ⚠️ **常驻件不许退** ✓✗（同 `enter_phase` ✓）：把它们并进 `wanted` 交给
            #: :func:`scheduler.reclaim_for` ✓ —— 那个纯函数的语义正是「不在 wanted 里的才退」✓，
            #: 并进来就等于"这一阶段也用得着"✓（**不动** `decide_for` 的账 ✗：它们不占卡 ✓）。
            kept = tuple(self._chain.get("keepParts") or ())
            if wanted is not None and kept:
                wanted = tuple(dict.fromkeys((*wanted, *kept)))
        present = {"dit": self._model is not None, "text": self._textEncoder is not None,
                   "vae": self._vae is not None}
        #: 判据落在 :func:`scheduler.reclaim_for` ✓（纯函数 ⇒ 能单测 ✓；写在这里就只能靠实跑验 ✗）。
        released = [self.release_part(part)
                    for part in scheduler_mod.reclaim_for(phase, present, wanted)]
        return self._record_event("pressure", {
            "phase": phase, **self._watch.to_dict(), "released": released,
            "note": ("到硬阈值 ⇒ 已退掉本阶段用不着的件 ✓" if released
                     else "到硬阈值 ⇒ 本阶段没有多余件可退 ✓（不搬动在场的权重 ✗）")})

    def _place_before_load(self, module: Any, *, what: str) -> dict[str, Any] | None:
        """**先摆位置、再装权重** ✓ ⇒ 驻留报告（``off`` ⇒ None ✓ 且照老路整份上卡 ✓）。

        ⚠️⚠️ 为什么值得单独一个方法 ✗✗（2026-10-31 收口 ✓）：这个"先切后装"的**顺序**已经在
        **四处**各写了一遍 ✓（`load_weights` 的 H3 线、`attach_dit`、`attach_qwen3_te`、
        `attach_qwen35_te` ✓）—— 每处都是同一段：

        ```python
        report = self._place_with_residency(module, what=...)
        if report is None:
            module.to(self._device)          # ⚠️ 这一行漏掉 ⇒ 整份还在 CPU，前向才炸
        ```

        ⚠️ 两个方向都会出事，且都**不报错** ✗：
        * **漏了 `if None 才 to()`** ⇒ ``split`` 时权重全留内存、``self._model`` 自以为在卡上 ✓✗
          （前向在第一步炸，报错却指向算子 ✓）；
        * **无条件 `.to()`**（切完又整份搬回卡 ✓✗）⇒ 切分当场作废、峰值仍是整份 ✓✗ ——
          正是这次要治的 OOM ✓。

        ⇒ 收成一处：新加一个 `attach_*` 时**不可能**只抄对一半 ✓✗。

        ⚠️ 剩下没接的件（`attach_vae` / `attach_flux_vae` / `attach_wan_vae` / 音频 VAE /
        upscaler / `attach_text_encoder` 的 stub ✓）是**有意**不接 ✗：它们 0.33–0.63 GiB
        （stub TE 更小 ✓），切开只多付每步搬激活的钱 ✓、省不到什么 ✓；而"装不下"这一档
        在它们身上根本走不到 ✓。**不许**照着这三处给它们补 —— 那是抄形状、不是抄理由 ✓✗。
        """
        report = self._place_with_residency(module, what=what)
        if report is None:
            module.to(self._device)
        return report

    def _place_int8_module(self, module: Any, *, what: str
                           ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """**int8 常驻**模块的放置 ✓ ⇒ ``(residency 报告, int8 放置报告)`` ✓（老路 ⇒ 两个都 None ✓）。

        ⚠️ 为什么单独一个方法 ✗✗（2026-11-01 收口 ✓）：这段「``residency`` 与 ``int8_offload``
        **二选一**」原先只写在 `load_weights` 的 H3 分支里 ✓✗，而 H3 的**文本塔**
        （同一个 int8 文件里的另一半 ✓，见 :meth:`attach_h3_text` ✓）要走**同一条** ✓ ——
        抄一份的话，「两条策略不能同时干活」那句必然只留在一边 ✓✗（症状：两档都开着 ⇒ 按品类
        钉 CPU 的与按层切卡的同时改同一批参数 ✓，两边报告里却各自写"已生效" ✓✗）。

        三个分支与 `int8_offload` 的口径**逐字**继承自 H3 那条线 ✓：
        * ``residency`` 生效 ⇒ 整份交给它 ✓（int8 跟着它所在的层走 ✓，但要说清 ``int8_offload``
          没**单独**生效 ✗）；
        * ``residency`` 没生效 ＋ 有卡 ＋ ``int8_offload`` ⇒ :func:`~.h3_form.place_module`
          把 int8 钉在**锁页内存** ✓、其它参数上卡 ✓；
        * 其余 ⇒ 整份 ``.to(self._device)`` ✓（CPU 后端上 ``int8_offload`` **无意义** ✓，如实说没生效 ✗）。
        """
        from app.services.engine import device_policy  # noqa: PLC0415
        from app.services.engine import h3_form  # noqa: PLC0415

        residency_report = self._place_with_residency(module, what=what)
        if residency_report is not None and residency_report.get("applied"):
            if not self._int8Offload:
                return residency_report, None
            return residency_report, {
                "device": str(self._device), "parkedOn": None,
                "applied": False, "moved": 0, "parked": 0,
                "parkedBytes": 0, "pinned": False,
                "note": f"`int8_offload` **没单独生效** ✗ —— 整份放置已交给 "
                        f"`residency={self._residency}` ✓"
                        "（两条策略一条按品类、一条按层，同时开会互相掀桌 ✗；"
                        "int8 权重跟着它所在的层走 ✓，不需要再单独钉一次 ✓）"}
        if self._int8Offload and not device_policy.is_cpu_device(self._device):
            #: ⚠️ 「钉 CPU」也是**往内存里塞字节** ✗ —— 同 `scheduler` 模块头的洞 ① ✓：
            #: int8 常驻层动辄 22.7 GiB ✓，内存不够时钉进去就是把机器拖进 swap ✓✗。
            scheduler_mod.ensure_host_room(
                _int8_bytes_of(module, self._torch()),
                what=f"{what} 的 int8 常驻（`int8_offload` 钉 CPU）")
            return residency_report, h3_form.place_module(module, self._device,
                                                          park_int8_on="cpu")
        module.to(self._device)
        if not self._int8Offload:
            return residency_report, None
        return residency_report, {
            "device": str(self._device), "parkedOn": None,
            "applied": False, "moved": 0, "parked": 0,
            "parkedBytes": 0, "pinned": False,
            "note": "CPU 后端本来就在内存里 ⇒ `int8_offload` 无需生效 ✓"
                    "（**不是失败** ✗，但也没省到任何显存 ✓）"}

    def _te_tokenizer(self, tokenizer_path: Any, shape: Any, *, id_hint: str = "") -> Any:
        """加载分词器并**核嵌入表装得下** ✓✗（``shape.vocab`` ≥ 分词器的 ``required_vocab_size``）。

        ⚠️ 为什么单独一个方法 ✗✗（2026-10-31 收口）：这一整段（``HubConfig`` / ``required`` /
        ``required > shape.vocab`` ⇒ 抛）在两条塔里**逐字重复** ✓✗，两处消息只差括号里的 id ✓
        —— 将来加一种分词器（或者修 `required_vocab_size` 的取法 ✓）必然只改一处 ✓✗。

        ⚠️ 漏了这条的真实症状（Z-Image 踩过 ✓）：``embed_tokens`` 越界 ⇒ **前向第 1 步**才炸 ✓✗，
        报错指向 ``embedding`` 索引越界 ✓，看着像"权重坏了" ✓。
        """
        if not tokenizer_path:
            return None
        from app.services.engine import tokenizer_hub  # noqa: PLC0415
        tokenizer = tokenizer_hub.load(tokenizer_hub.HubConfig(path=str(tokenizer_path)))
        required = int(getattr(tokenizer, "required_vocab_size",
                               getattr(tokenizer, "vocab_size", 0)) or 0)
        if required > shape.vocab:
            raise TorchBackendUnavailable(
                f"分词器需要嵌入表 ≥ {required} ✓，但文本塔的 vocab={shape.vocab} ✗"
                f" ⇒ 特殊符 id{id_hint}会**越界** ✓✗（词表与权重不是同一份 ✓？）",
                reason="pending")
        return tokenizer

    def _raise_te_mismatch(self, report: Any, what: str = "文本塔") -> NoReturn:
        """权重与结构**不符 ⇒ 拒** ✓✗（留一层随机权重比报错坏得多 ✓）。

        ⚠️ 为什么单独一个方法 ✗✗（2026-10-31 收口）：这条消息原先在 `attach_qwen3_te` 与
        `attach_qwen35_te` 里**逐字抄了两遍** ✓✗ —— 改措辞、或想多报一点（比如把前几个
        缺的键名列出来 ✓）时，必然只改到一处 ✓✗。

        ⚠️ 返回类型是 ``NoReturn`` ✓✗（**不是** ``None``）：调用方写完这句就**不再往下走** ✓，
        而它后面紧跟着 ``self._textEncoder = module.eval()`` ✓ —— 若标成 ``None``，静态检查
        会认为「`del module` 之后还可能用到 `module`」✓✗（第一版就这样，自己给自己挖了个坑 ✓）。

        ⚠️ `del module` **留在调用方** ✗：18.5 GiB 的壳得在抛之前放开 ✓（本方法拿不到那个
        局部变量 ✓ —— 传进来再 `del` 也删不掉调用方的名字 ✓）。
        """
        raise TorchBackendUnavailable(
            f"{what}权重与结构**不符** ✗（缺 {len(report.missing)} 键 ✓ / "
            f"形状不符 {len(report.shapeMismatch)} 处 ✓）⇒ 拒 ✗（不留随机层 ✓✗）",
            reason="pending")

    def _te_report(self, *, kind: str, runtime: Any, path: Path, want: Any,
                   residency_report: dict[str, Any] | None, report_weights: Any,
                   note: str, tokenizer: Any = None, tokenizer_path: Any = None,
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """两条文本塔的**回执**（同一套键 ✓✗）。

        ⚠️ 为什么单独一个方法 ✗✗：`attach_qwen3_te` 与 `attach_qwen35_te` 的回执原先各写一份 ✓
        —— 十格里九格逐字相同 ✓✗。这不是美观问题 ✗：``residency`` 这一格是**这轮才加的** ✓，
        当时就是手动往两份里各插一遍 ✓；下一格（比如 ``firstLayerDevice`` ✓）还会重复一次 ✓✗。
        第三座塔（Qwen3-VL-32B 64 GiB ✓）接进来时，也该**直接复用** ✓。

        ⚠️ 各塔的差异只有 ``kind`` / ``note`` / ``extra`` ✓（``unexpectedCount`` 与
        ``ignoredTowerKeys`` 是 Qwen3.5 独有的 ✓ —— 它那份文件里带视觉塔与 lm_head ✓）。
        """
        report = {
            "kind": kind, "config": runtime.to_dict(), "complete": True,
            "path": str(path), "dtype": str(want), "device": str(self._device),
            #: ⭐ 「卡上几层 / CPU 几层 / 每步搬多少」✓ —— `off` ⇒ None（老路 ✓）。
            #: 不报这一格 ⇒ 装着十几 GiB 的塔、读者只能从 `device` 猜它在哪 ✓✗。
            "residency": residency_report,
            "loadedKeys": int(report_weights.loadedKeys),
            "sourceKeys": int(report_weights.sourceKeys),
            "tokenizer": (getattr(tokenizer, "name", type(tokenizer).__name__)
                          if tokenizer is not None else None),
            "tokenizerPath": str(tokenizer_path) if tokenizer_path else None,
            "note": note,
        }
        if extra:
            report.update(extra)
        return report

    def load_weights(self, *, path: str | None = None,
                     config: Any = None, loras: Any = None) -> dict[str, Any]:
        """**装载真模型** ✓ ⇒ 之后 :meth:`denoise` 走**真前向** ✓（不再占位 ✓）。

        * ``path`` 省略 ⇒ 按 ``configs/models.json`` 里的主 DiT 解析 ✓（同一份清单 ✓）；
        * ``config`` 省略 ⇒ 先从权重 ``__metadata__`` 读 ✓（读到就用 ✓）；读不到 ⇒ **报错** ✗
          （**不猜结构** ✓；道理同 :mod:`dit` 的模块注释 ✓）；
        * 权重没下载 ⇒ ``reason="pending"`` ✓ + 给出还缺多少 ✓（`loader` 的口径 ✓）。
        """
        self._gate()
        # ⚠️ 上一次装载的**驻留把手先丢掉** ✓：换模型/换开关时留着它 ⇒ `describe()` 会把旧切法
        #    当成现状自述 ✓✗。⚠️ 原注释说「旧模块与其 hook 已经跟着 `_model` 一起被替换掉了」✗✗
        #    —— 那句话是**错的** ✓：handle 自己攥着 `module` ✓、hook 闭包又攥着 handle ✓，
        #    直接置 None 只丢掉了「我们这一侧的引用」✓，环还在 ⇒ 旧模块连同卡上那份权重一直活着 ✓✗。
        self._drop_residency_handle()
        # ⚠️ **CPU 线程预算**（用户口径 ✓ 2026-09-26）：本仓重张量运算走 GPU ✓，CPU 只负责加载/IO/搬张量 ✓
        #    ⇒ 必须**钳住** torch 的线程池 ✗（不钳时它的默认并行度 = **逻辑核数** ✓✗ ⇒ 装权重这一步的
        #    反量化 + `to(dtype)` 就能把**整机**拉满 ✓✗ —— 实测本机 32 逻辑核 ✓）。
        #    口径**只有一处** ✓：`app/core/cpu_budget.py` ✓（自检驱动与引擎共用 ✓）。
        self._threadBudget = cpu_budget.apply_torch(self._torch())
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
            #    （社区重导出 / 蒸馏 / 缩小版 ✓）会「名字对、形状错」✓✗ —— 那正是 `IMPLEMENTATION_HISTORY`
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
            # ⚠️ **只建一次** ✓（老写法建两次 ✓✗ —— fp32 主干 78 GiB，建两次直接炸 ✓✗）；
            #    顺序见 :func:`_prepare_h3_module` ✓（meta 建壳 → 降精度 → 扳回 fp32 岛 →
            #    int8 常驻层 → **最后**才物化 ✓✗ —— 物化按最终精度，峰值 ≈ 19.5 GiB 而非 78 GiB ✓）。
            module, materialized = _prepare_h3_module(h3_form, form_defaults, target, info,
                                                      dtype=self._compute_dtype())
            # ⭐ **联合驻留**（卡 + CPU 按预算切一刀 ✓）—— 摆位置必须在 `load_module_weights` **之前** ✓✗：
            #    装载走 ``param.copy_`` ⇒ 权重落在参数**当前**的设备上 ✓ ⇒ 先切再装，装的时候卡上就只压那一刀 ✓
            #    （反过来的话峰值仍是整份 ✗✗，正好是这次要治的 OOM ✓）。`off` ⇒ None，老路一字未变 ✓。
            # ⚠️ 两条放置策略**不能同时干活** ✗✗：`int8_offload` 按**品类**搬（int8 一律钉 CPU ✓）、
            #    `residency` 按**层**搬（连续前缀上卡 ✓）⇒ 同时开会互相掀桌 ✓✗。
            #    整段（含"没单独生效"的三种如实说法 ✓）已收进 `_place_int8_module` ✓ ——
            #    H3 的**文本塔**（同一个 int8 文件里的另一半 ✓）走的是**同一个**方法 ✓✗。
            residency_report, parked = self._place_int8_module(module, what="h3")
            form_report = weights_mod.load_module_weights(module, target)
            if not form_report.complete:
                raise TorchBackendUnavailable(
                    f"H3 形态权重与结构对不上 ✗（complete=False ⇒ **不是**完整模型）"
                    f"　：{form_report.to_dict()}",
                    reason="pending")
            # ⚠️ int8 的账**当场对** ✓✗（见 :func:`_int8_residency_problems` ✓）：层没升级成功
            #    不会在这行报错 ✗，而是几十分钟后在前向里「CUDA out of memory」✗✗ —— 指不到真原因 ✓。
            #    ⚠️ **与 offload 无关地查** ✓：老路（全在卡上）同样会因为漏升级而爆 ✓✗。
            int8_ledger, int8_problem = _int8_residency_problems(module, form_report)
            if int8_problem:
                raise TorchBackendUnavailable(
                    int8_problem + "　⇒ **不装载** ✓（装进去也跑不起来 ✗；"
                    "而报告里那几个数各自都\"正常\"✓✗）", reason="pending")
            #: ⭐⭐ **摆成纯推理形态** ✓✗（2026-11-01 实测的根因 ✓，见 :func:`_freeze_inference` ✓）：
            #: H3 的 norm 权重默认 ``requires_grad=True`` ⇒ 不冻就整条采样在**建图** ⇒ 每步的中间
            #: 激活与被现反量化的 bf16 权重都被钉住 ✓ ⇒ 实测涨到 83.7 GiB 后被 WDDM 挪进系统内存 ✓✗。
            frozen = _freeze_inference(module)
            self._model = module
            self._config = form_defaults
            self._form = self.H3_FORM_NAME
            self._ditFamily = "h3"           # 同 `_form`：**每次装载都显式写** ✓（换模型时不许留旧的 ✗）
            try:
                vae_check = self._check_attached_vaes()
                text_check = self._check_text_encoder()   # ⭐ 已挂的 TE 也在这一刻再核一次 ✓
                lora_report = self._apply_loras(loras)    # ⭐ LoRA 在**权重装齐之后**挂 ✓
                #: ⚠️ **挂完 LoRA 再冻一次** ✓✗：LoRA 的 A/B 参数是挂载那一刻才建出来的 ✓，
                #: 默认取梯度 ✓ ⇒ 只冻底模的话，LoRA 那几层照样把整张图吊起来 ✓✗
                #: （症状与完全不冻**一模一样** ⇒ 会被误判成"冻结没生效"✓）。
                frozen += _freeze_inference(self._model)
            except TorchBackendUnavailable:
                # ⚠️ 校验失败 ⇒ **中止装载 + 不污染模块** ✓（本仓在反量化那条上定过这条规矩 ✓）：
                #    留着半装状态会让 `dualStream` / `latentMode` 按"装好了"自述 ✓✗。
                self._model = None
                self._config = None
                self._form = None
                self._ditFamily = None
                self._drop_residency_handle()   # 半装模块不留把手 ✓（hook 环必须真摘 ✓）
                self.release_loras(reason="装载中止 ⇒ 旧模型的 LoRA 也还回去 ✓")
                raise
            self._loadReport = {
                **form_report.to_dict(),
                # ⭐ 「哪些字段是**权重里读出来的** ✓、哪些是**不可推的回落** ✗」随报告一起给 ✓
                #    —— 装真权重时，光看 `complete=True` 分不清这两种 ✓✗（本仓那条纪律 ✓）。
                "configSources": dict(inferred.sources),
                # ⭐ 「有多少参数是从 meta 壳上**真分配**出来的」✓ —— 与 TE 那份同名同义 ✓（
                #    `attach_h3_text` 也报这一格 ✓）：这份数字就是「没有先建 78 GiB fp32 主干」的凭据 ✓。
                "materializedParams": materialized,
                # ⭐ 「int8 到底在卡上还是在 CPU」是**显存能不能跑起来**的那件事 ✓ ⇒ 进报告 ✓
                #    （`parked=None` ⇒ 老路：全在卡上 ✓；`parked.parkedBytes` 是留在 CPU 的那份 ✓）
                "int8Residency": parked,
                # ⭐ 「卡上几层 / CPU 几层 / 每步要搬多少字节」✓ —— `residency=off` ⇒ None（老路 ✓）
                "residency": residency_report,
                # ⭐ 四个数放一处 ✓：层数 / int8 参数数 / 在 CPU 的 / 文件侧常驻数 ✓
                #    —— 前两个与第四个必须相等（不等已经在上面**拒绝装载**了 ✓）
                "int8Ledger": int8_ledger,
                "derivedFields": inferred.derived_count,
                "fieldCount": len(inferred.config),
                # ⭐ 已挂的 VAE 在这里**再查一次** ✓（挂载顺序反过来也要守 ✓ ——
                #    先挂 VAE 的那次调用时形态还没定 ⇒ 只能等这一刻 ✓）；空 = 还没有 VAE 可比 ✓
                "vaeCheck": vae_check,
                "textEncoderCheck": text_check,
                "loras": lora_report,
                #: 「这回冻掉了几个参数」✓ —— 大于 0 是**正常**的（norm 权重默认取梯度 ✓），
                #: 恒为 0 反而该查（说明骨架自己已经设了 ``requires_grad=False`` ✓ 或装载走了别的路 ✓）。
                "frozenParams": frozen,
            }
            return self._loadReport
        # ⭐⭐ **按族选骨架**（2026-10-31 ✓）：H3 之外原先是「一条通用 dit」✗ —— 而 Z-Image 那族
        #    的键布局与 Krea2 形态**不同** ✗（见 :mod:`dit_lumina` 模块头 ✓），硬塞进
        #    ``dit.build_dit`` 的结果是「报告缺一堆键」✓，读起来像「权重没下全」✗✗。
        #    ⇒ 判族（纯函数 ✓ 只看头部键名 ✓）→ 有专用骨架的走专用 ✓ → 其余仍走老路 ✓（一字未改 ✓）。
        family = _dit_family_of(info)
        explicit = dict(config) if isinstance(config, dict) else {}
        #: ⭐ 精度**跟着 `self._device`** 走 ✓✗（有卡但 `device="cpu"` ⇒ fp32 ✓，见 `_compute_dtype` ✓）
        want_dtype = self._compute_dtype()
        #: ⭐ 一个族一条装配声明（2026-11-01 收口 ✓）：原先这里是 `if/elif` 四支、每支各抄
        #:    一遍「抽形状 → infer → meta → 物化 → 搬字段 → 写口径」，少改一支就静默回落默认值 ✗。
        #:    表在 :mod:`dit_build`；这里只**查一次**。加族不需要再动本文件 ✓（有测试守着 ✓）。
        #:    ⚠️ ``path`` 必须传 ✗✗：Krea2 要按**文件里的** convrot 说明把 int8 层换成
        #:    ``Int8Linear`` 常驻（12.82B 参数 ⇒ 按 bf16 物化 23.9 GiB 装不进 22.49 GiB 卡 ✓）。
        #:    不给 ⇒ 换壳一个都不做 ⇒ 权重按 bf16 落盘 ⇒ 采样第 1 步 OOM ✓✗（别的族忽略它 ✓）。
        built = _build_via_table(family, info, explicit, dtype=want_dtype, path=target) \
            if family in dit_build.BUILDERS else None
        if built is not None:
            module, config = built
        else:
            if config is None:
                config = dit_mod.DiTConfig.from_metadata(info.metadata)   # 读不到会**明确报错** ✓
            # ⚠️ **只建一次、只装一次** ✓✗ —— 老写法是「先建一个验一下、再建一个真装」✗，
            #    而 :func:`weights.load_module_weights` 的 ``device`` **不参与读取** ✓（字节先进 CPU ✓，
            #    再由 ``copy_`` 落到 param 自己所在的 device ✓）⇒ 那个"验一下"的 model 从没 ``.to()`` ✓
            #    ⇒ 整份权重被**装进一份马上要丢弃的 CPU 模型** ✓✗：白建一遍 + 白读一遍盘 ✓✗，
            #    峰值上多压一份模型 ✓（与 :func:`_prepare_h3_module` 里记的那条**同一个**病根 ✓✗，
            #    那边已修 ✓、这里漏着 ✗ —— 19.53 GiB 的权重上，这一份就是能不能装得下的差别 ✓）。
            #    ⚠️ 复用同一份时**放置必须和原来"真装"那一次完全一样** ✓：先 ``.to(device)`` 再装 ✓，
            #    否则权重会落在 CPU 而 ``self._model`` 自以为在卡上 ✓✗（静默，前向才炸 ✗）。
            module = dit_mod.build_dit(config)
        # ⭐ **联合驻留**（卡 + CPU ✓）先切再装 ✓✗ —— 同上：权重落在参数当前所在的设备上 ✓
        #    ⇒ 装载期的峰值显存就只是"卡上那一刀" ✓（先装后切 ⇒ 峰值仍是整份 ✗✗）。
        #    `residency=off` ⇒ 原样 ``.to(device)`` ✓（老路一字未变 ✓）。
        residency_report = self._place_before_load(module, what="dit")
        #: ⭐ Krea2 那层**导出前缀**（``model.diffusion_model.`` 之类 ✓）在装载时剥掉 ✓ ——
        #: ``key_map`` 由 :func:`_build_krea2_module` 按「全部键都带哪一层」实测给出 ✓，
        #: 其余族恒为 ``None`` ✓（老路一字未改 ✓）。
        key_map = config.get("weightKeyMap") if isinstance(config, dict) else None
        report = weights_mod.load_module_weights(module, target, device=self._device,
                                                 key_map=key_map)
        if not report.complete:
            del module                        # 别让半装的模型在 raise 期间继续被引用 ✓
            raise TorchBackendUnavailable(
                f"权重与模型结构对不上 ✗（complete=False ⇒ **不是**完整模型）"
                f"　：{report.to_dict()}",
                reason="pending")
        #: ⭐ **int8 常驻层的入册对账** ✓✗（目前只有 Krea2 这一族真走常驻 ✓）：
        #: 计划点名了 N 层、模块里就得真有 N 层 ``Int8Linear`` ✓；少一层 ⇒ 那层按 bf16 落盘 ⇒
        #: 显存按 2 倍算 ⇒ **装得下、跑不动**（OOM 落在采样第 1 步，报错指不到"哪层没接管" ✗✗）。
        #: ⚠️ 这条必须**装完之后**查 ✓：装载前查只能看见"壳换没换"✓，看不见"权重真落进 int8 参数了没"✓。
        int8_intent = config.get("int8Plan") if isinstance(config, dict) else None
        int8_ledger: dict[str, Any] | None = None
        if int8_intent:
            int8_ledger, leftover = _int8_residency_problems(module, report)
            if leftover:
                del module
                raise TorchBackendUnavailable(
                    f"int8 常驻层没接管齐 ✗ ⇒ 不装载（少一层就多 2 倍显存 ✓✗）："
                    f"{leftover}　〔计划 {int8_intent.get('layers')} 层 / "
                    f"账目 {int8_ledger}〕", reason="pending")
        #: ⭐⭐ **纯推理形态** ✓✗（同 H3 那一条 ✓，见 :func:`_freeze_inference` 里 83.7 GiB 那份实测 ✓）：
        #: 这条路的骨架（`dit_lumina` / `dit_qwen_image` / `dit_generic` ✓）的 norm 权重同样是
        #: ``nn.Parameter(torch.ones(...))`` ✓（默认取梯度 ✓）⇒ 不冻就白建图 ✓（Z-Image 8 步的图上，
        #: 表现为「显存占用比账面高 2~3 GiB、每步比预期慢」✓ —— 不会炸 ✗，但一直在白花 ✓）。
        frozen = _freeze_inference(module)
        self._model = module
        self._config = config
        # ⚠️ 形态必须**每次装载都显式写** ✓：同一实例"先装 H3、再装 DiT"时，不写就会把上一次的
        #    `_form` 留着 ✓✗ ⇒ 双流自述会说"可用"而模型其实是 DiT ✗✗（静默 ✓）。
        self._form = "dit"
        # ⭐ 装了**哪一族的骨架** ✓ —— 前向口径跟着它变 ✗（`denoise` 里 σ 是标量还是 ``(b,)`` 张量 ✓），
        #    所以它不能只活在注释里 ✓；`None` = 认不出族 ⇒ 走老路（通用 dit ✓）。
        self._ditFamily = family
        self._loadReport = report.to_dict()
        self._loadReport["ditFamily"] = family
        self._loadReport["frozenParams"] = frozen
        if isinstance(config, dict) and config.get("configSources"):
            # ⭐ 哪些字段是**权重里读出来的** ✓、哪些是**默认值** ✗ —— 与 H3 那条同一个口径 ✓
            self._loadReport["configSources"] = dict(config["configSources"])
        # ⭐ 「卡上几层 / CPU 几层 / 每步搬多少字节」✓ —— `off` ⇒ None（老路 ✓）
        self._loadReport["residency"] = residency_report
        # ⭐ 「int8 常驻层到底接管了几层」✓ —— 只有走常驻的族（Krea2 ✓）才有这一格；
        #    其余族仍报 ``int8Ledger=None`` ✓（老路 ✓，不凭空多一个字段 ✗）。
        if int8_ledger is not None:
            self._loadReport["int8Ledger"] = int8_ledger
        if self._int8Offload:
            # ⚠️ **必须如实说"没生效"** ✗✗：`int8_offload` 只认 H3 形态的 `Int8Linear` 常驻层 ✓
            #    （见 :func:`_prepare_h3_module` ✓）；这条路把 int8 **反量化进计算精度** ✓✗
            #    ⇒ 一个 int8 参数都不会被 park ✓。静默忽略的后果是「以为省了 18 GiB 显存」✓✗，
            #    然后在 720×1280 上照样 OOM、且找不到原因 ✓✗
            #    （与 CPU 后端那条"无需生效"**同一个口径** ✓：不生效可以 ✓，不说不行 ✗）。
            #: ⚠️ 但 🚨**别把这句话写死成"没有常驻层"** ✗✗（2026-11-01 修 ✓）：Krea2 之后这条路上
            #: **有**常驻层了 ✓（12.155B 个 int8 参数 ✓）⇒ 老句子变成假话 ✓✗（"有 224 层却说没有" ✓）。
            #: 判据必须是**实测**：数一数模块里真有几个 ``Int8Linear`` ✓。
            self._loadReport["int8Residency"] = {
                "device": str(self._device), "parkedOn": None, "applied": False,
                "moved": 0, "parked": 0, "parkedBytes": 0, "pinned": False,
                "note": (f"有 {int8_ledger['layers']} 层 `Int8Linear` 常驻 ✓，但 `int8_offload` "
                         "只接了 H3 形态那条路 ⇒ 这次**不生效** ✓（常驻权重仍全在卡上 ✓）"
                         if int8_ledger and int8_ledger.get("layers") else
                         "DiT 形态没有 `Int8Linear` 常驻层 ⇒ `int8_offload` **不生效** ✓"
                         "（它只对 H3 形态的 int8 权重有意义 ✓；"
                         "这条路上的 int8 会被反量化成计算精度 ⇒ 显存按 2 倍算 ✓）")}
        try:
            self._loadReport["loras"] = self._apply_loras(loras)
            #: ⚠️ 同 H3：LoRA 是挂载那刻才建参数 ✓ ⇒ 挂完再冻一次 ✓（否则那几层照样建图 ✓✗）。
            frozen += _freeze_inference(self._model)
            self._loadReport["frozenParams"] = frozen
        except TorchBackendUnavailable:
            self._model = None                       # 不半装 ✓（同 H3 那条 ✓）
            self._config = None
            self._form = None
            self._ditFamily = None
            self._drop_residency_handle()            # 同 H3：半装不留把手 ✓（hook 环必须真摘 ✓）
            self.release_loras(reason="装载中止 ⇒ 旧模型的 LoRA 也还回去 ✓")
            raise
        #: ⚠️ 装完**立刻**把装载临时（逐键搬权重的中转 ✓）还给驱动 ✓✗：
        #: 不还，下一步 `enter_phase` 的探测会把「我们的缓存块」算成「卡被别人占了」✓✗
        #: ⇒ 下一件被白切（实测：DiT 装完后 ``reserved`` 比 ``allocated`` 多 3.8 GiB ✓，
        #: 那次 25.28 GiB 的塔因此只上卡 3.5 GiB ✓）。
        self._trim_cuda()
        return self._loadReport

    def _apply_loras(self, loras: Any) -> dict[str, Any] | None:
        """按 :class:`engine.lora.LoraSpec` 挂 LoRA ✓（**权重装齐之后** ✓）⇒ 报告 ✓。

        ⚠️ **装不上必须炸** ✗✗（本仓纪律 ✓）：少挂一层 LoRA 的后果是「跑得动、出得来、
        但不是你要的那个档」✓✗ —— 4 步档尤其（turbo LoRA 就是那 4 步的全部依据 ✓）。
        三种拒装理由各说各的 ✓：① 全部层都对不上（拿错文件 / 拿错底模 ✗）；
        ② 有 adaln 层（本仓还不支持运行期注入 ✗）；③ 一层都没生效 ✓。
        """
        from app.services.engine import lora as lora_mod  # noqa: PLC0415 —— 局部 import ✓ 不拖 torch ✓

        specs: list[Any] = []
        for item in (loras or ()):
            #: ⚠️ 写法与 ``strength`` 口径交给 `lora.coerce_spec` ✓（单一真源 ✓）——
            #:    这里以前写的是 ``float(item.get("strength", 1.0) or 1.0)`` ✗✗：显式 ``0``
            #:    会被抬成 **1.0** ✓✗，而装载键按 ``0.0`` 记 ✓ ⇒ 「键说 0.0、实际按 1.0 挂」✓✗。
            try:
                specs.append(lora_mod.coerce_spec(item))
            except lora_mod.LoraError as err:
                raise TorchBackendUnavailable(f"LoRA 参数 ✗：{err}", reason="pending") from err
        if not specs:
            return None

        # ⚠️ 换模型先还上一批 ✗✗：本实例再来一次 `load_weights` ⇒ 旧模型的 LoRA 张量没人用了 ✓；
        #    不还就变成"每换一次装多吊一份" ✓✗（E2.5 之前正是这个形状：压根没有账本 ✓）。
        self.release_loras()

        from app.services.engine import lora_cache as cache_mod  # noqa: PLC0415 —— 零依赖 ✓

        cache = cache_mod.shared()
        leases: list[Any] = []
        loaded: list[str] = []

        def reader(path: str) -> Mapping[str, Any]:
            """借一份张量 ✓ —— **同内容不读第二遍** ✓（E2.5 ✓）。"""
            lease = cache.acquire(path, reader=lora_mod.read_tensors,
                                  reason=f"挂 LoRA ✓：{Path(path).name}")
            leases.append(lease)
            loaded.append(lease.digest)
            return lease.tensors

        try:
            result = lora_mod.apply_specs(self._model, specs, ops=lora_mod.torch_ops(),
                                          reader=reader)
        except Exception:
            # ⚠️ 挂到一半炸了 ⇒ 借出去的全还回去 ✗✗（这里不还 ⇒ 张量永远吊在账本里 ✓✗，
            #    而症状要等到下一次装载才发现"怎么又多占了一份"✓✗）
            self._release_leases(leases, reason="挂 LoRA 失败 ⇒ 还回去 ✓")
            raise
        report = {"specs": [spec.to_dict() for spec in specs], **result.to_dict(),
                  "files": loaded}
        if result.error or result.applied == 0:
            #: ⚠️ 半装的钩子也要摘 ✗✗（"没装全"里可能有几十层已经挂上了 ✓）——
            #:    不摘 ⇒ 抛异常后模块上还留着半个 LoRA ✓✗，调用方重试一次就变成两份 ✓。
            self._detach_lora_hooks(list(result.hooks))
            self._release_leases(leases, reason="LoRA 没装全 ⇒ 不占用内存 ✓")
            if result.error:
                raise TorchBackendUnavailable(f"LoRA 没装全 ✗：{result.error}", reason="pending")
            raise TorchBackendUnavailable(
                f"LoRA 一层都没装上 ✗（键名与底模对不上 ⇒ 多半拿错了文件 ✓）：{report}",
                reason="pending")
        #: ⚠️ 借来的**一律留到模型卸载** ✓✗ —— 报告里分不出"哪条 spec 折进了权重、哪条走的是
        #:    hook"，而 hook 那批**必须**一直活着 ✓（张量被回收 ⇒ 前向读到已释放的内存 ✗✗）；
        #:    折进去的那批其实可以立刻还 ✓，但为多省那点 LoRA 内存去猜"哪条是 merge"不值得 ✓
        #:    （猜错 = 跑起来像对、数值悄悄不对 ✓✗）。要优化得先让 `apply_specs` 逐条回报 ✓。
        self._loraLeases.extend(leases)
        #: ⭐ 钩子入账 ✓（`release_loras` 摘的就是这批 ✓）—— 与租约**同生同灭** ✓。
        self._loraHooks.extend(result.hooks)
        facts = cache.facts()
        report["cache"] = {"entries": facts["entryCount"], "refTotal": facts["refTotal"],
                           "tensorBytes": facts["tensorBytes"],
                           "deduped": facts["deduped"], "misses": facts["misses"],
                           "note": "同内容只留一份 ✓（`lora_cache` ✓）；这些张量**留到模型卸载** ✓"}
        return report

    def _release_leases(self, leases: list[Any], *, reason: str) -> dict[str, Any]:
        """把一批租约还掉 ✓（**成对** ✓）—— 已经不在账本里的跳过 ✗（`release_all` 之后就会这样 ✓）。"""
        from app.services.engine import lora_cache as cache_mod  # noqa: PLC0415

        cache = cache_mod.shared()
        released = 0
        freed_bytes = 0
        for lease in leases:
            try:
                fact = cache.release(lease, reason=reason)
            except cache_mod.LoraCacheError:
                continue                      # 已被 `release_all` 清掉 ⇒ 不算错 ✓
            if fact.get("released"):
                released += 1
                freed_bytes += int(fact.get("tensorBytes") or 0)
        return {"leases": len(leases), "released": released, "freedBytes": freed_bytes}

    def _detach_lora_hooks(self, handles: list[Any]) -> int:
        """把 bypass 钩子**真摘掉** ✓ ⇒ 摘掉几个 ✓（重复摘同一个 handle 是空操作 ✓ 不炸 ✗）。"""
        removed = 0
        for handle in handles:
            try:
                handle.remove()
                removed += 1
            except Exception:                 # noqa: BLE001 —— 句柄已失效 ⇒ 目的已达成 ✓
                continue
        del handles[:]
        return removed

    def release_loras(self, *, reason: str = "") -> dict[str, Any]:
        """还掉本实例借的全部 LoRA ✓ ⇒ 报告 ✓（**模型卸载 / 换装配时必须调** ✗✗）。

        ⚠️ **两件一起还** ✗✗：钩子（补丁）＋ 租约（张量）✓ —— 只还租约 ⇒ 模块上还挂着
        指向已释放张量的 hook ✓✗（前向读到释放内存 ✓），而掩掉那类崩溃的更糟情形是：钩子活着、
        张量被 cache 复用给**另一条** LoRA ⇒ 增量静静变成别人的 ✓✗（画面"像但不对" ✓）。
        """
        hooks = self._detach_lora_hooks(self._loraHooks)
        leases, self._loraLeases = self._loraLeases, []
        fact = self._release_leases(leases, reason=reason or "模型卸载 ⇒ 还 LoRA ✓")
        return {**fact, "hooks": hooks}

    def _default_dit_path(self) -> str | None:
        """按清单找主 DiT ✓（``category=video`` 且 ``required`` 的第一个 ✓ —— 与体检同一份清单 ✓）。

        ⚠️ **同一条解析** ✗（2026-09-27 修 ✓）：走 :func:`inventory.resolve_component` ✓ ——
        清单落点（``models_dir`` ✓）优先 → 动态探测根 ✓。以前这里只看 ``models_dir`` ✗✗ ⇒
        「体检/桥接看得见那份权重、装载这条路看不见」✓✗ —— 同一件事两个说法 ✓（用户当场指出 ✓）。
        """
        for entry in inv.load_catalog()["models"]:
            if entry.get("category") == "video" and entry.get("required") \
                    and str(entry.get("kind")) == "diffusion_models":
                found, _source = inv.resolve_component(entry)
                return str(found) if found else None
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

    #: ⚠️ 原来这里是**类属性写死** ``synthetic = True`` ✗✗（注释写着「刻意保持 True」✓）。在真权重
    #: 还没接上时它是对的 ✓；2026-11-01 起 Z-Image / Krea2 / Qwen-Image 2.1 / Flux / H3 都端到端出片 ✓
    #: ⇒ 写死 True 就成了「明明出的是真图、接口却说自己出的是假图」✗✗，而且与本仓 `sdxl_backend`
    #: （那一格早就按装载状态算 ✓）自相矛盾 ✗。⇒ 改按**已装件的事实**算 ✓。
    @property
    def synthetic(self) -> bool:
        """产物**是不是合成图** ✓ —— 按**现在装了什么**算 ✗，不写死 ✗（见 :meth:`_picture_is_real` ✓）。"""
        return not self._picture_is_real()

    def _picture_is_real(self) -> bool:
        """此刻解出来的画面**是不是真**的 ✓ —— DiT 装了 ＋ VAE 是**真权重**那三种之一 ✓。

        ⚠️ 两个条件缺一不可 ✗：只装 VAE ⇒ 前向仍走占位（DiT 没装 ✓）⇒ 画面不是真的 ✓；
        只装 DiT ⇒ 解不了（`decode` 自己会拒 ✓）。H3 的音频轨另有判据，见 :meth:`_decode_dual` ✓。
        """
        return (self._model is not None
                and self._vae is not None
                and self._vaeKind in self.REAL_VAE_KINDS)

    def _generation_facts(self) -> dict[str, Any]:
        """⭐「**此刻**能不能出片 ✓ / 还缺什么 ✓ / 画面真不真 ✓」—— 全按已装件算 ✗（不写死 ✗）。

        ⚠️ 为什么必须按状态算 ✗✗（2026-11-01 修 ✓）：`describe()` 原来把 ``canGenerate`` **写死 `False`**、
        ``pendingParts`` 塞了 75 行沿革散文 ✓✗ ⇒ 装完整链、真出过 mp4 之后接口仍报
        「不能出片 ＋ 一堆『没实现』」✓✗（`/engine/backends` 是 UI 唯一的自述口 ✓，
        读起来像整个引擎都不能用 ✗）。本仓 `llm_backend` 的 ③″ 与 `sdxl_backend` 早就是这套口径 ✓，
        这里只是补齐 ✗。

        ⚠️ 缺件按**件**列 ✗，不按"机制"列 ✓（机制层缺口在 :data:`CODE_GAPS` ✓）：
        依赖没装 ⇒ **只报依赖这一件** ✓（别把 DiT/TE/VAE 一起报成"缺"✗ —— 那是"还没轮到装"✓）。
        """
        if not self._available:
            return {"canGenerate": False,
                    "pendingParts": [f"torch 本身（{self._reason or '缺依赖'} ✓ —— 见 `dependencies` ✓）"],
                    "synthetic": True}
        missing: list[str] = []
        if self._model is None:
            missing.append("DiT 权重（`load_weights(path)` 或 `ensure_loaded(...)` ✓）")
        if self._textEncoder is None:
            #: ⚠️ **SEEDVR2 没有文本塔** ✗（条件＝源视频编码 ＋ mask 的 17 通道 ✓，
            #: 正向/负向条件都烤在权重里 ✓）—— 拿「缺文本编码器」判它，会让这条**唯一装得下的**
            #: 修复线永远自述"不能出片" ✓✗（2026-11-01 接线时踩到 ✓）。
            #: 它真正缺的一件不是常驻的 ✓ 而是**每单的源视频** ✓ ⇒ 不列进 ``pendingParts`` ✗
            #: （那是「还缺哪个权重文件」的地方 ✓）；请求里没给 `referenceVideos` 由
            #: :meth:`_encode_seedvr2_conditioning` 当场拒并说清 ✓。
            if getattr(self, "_ditFamily", None) != SEEDVR2_FAMILY:
                missing.append("文本编码器（`attach_text_encoder(...)` ✓）")
        if self._vae is None:
            missing.append("画面 VAE（`attach_vae(...)` ✓）")
        #: ⚠️ 只有 H3 形态**才**要音频 VAE ✗（单流族没有音频轨 ⇒ 拿它判会让 Z-Image 永远"缺一件"✓✗）
        if getattr(self, "_form", None) == self.H3_FORM_NAME and self._audioVae is None:
            missing.append("音频 VAE（H3 双流要它 ✓ —— `attach_h3_audio_vae(...)` ✓）")
        return {"canGenerate": not missing, "pendingParts": missing,
                "synthetic": not self._picture_is_real()}

    def _latent_mode(self) -> str:
        """潜变量现在是什么形态 ✓ —— **四态**（别混成一个 bool ✗）：

        ``h3-dual`` ✓（H3 形态 + 音频 VAE + 取整口径齐 ⇒ 视频与音频两条 ✓）/
        ``h3-trunk-only`` ✓（H3 形态但双流不齐 ⇒ 只有主干，**不能走管线** ✗）/
        ``real-shape`` ✓（DiT + vae_scale ⇒ 真形状单流 ✓）/ ``placeholder-1d`` ✓（一维占位 ✓）。

        ⭐ Z-Image 也报 ``real-shape`` ✓ 但**形状不同** ✗：它的潜变量是 **4 维** ``(1,c,h,w)`` ✓
        （图 ✓ 没有时间维 ✗ —— 见 :meth:`latent_shape` ✓）。⚠️ 2026-10-31 更正 ✗：此前这里写
        「Z-Image 不需要 ``vae_scale`` ✓」⇒ 配合 ``latent_shape`` 的 ``getattr`` 一起，等于**永远造不出
        真形状** ✓✗。现在 ``_build_zimage_module`` 会给 config 补 ``vae_scale=8``（同族事实 ✓）⇒ 这条
        判据也按同一事实判 ✓，读不到就老实报 ``placeholder-1d`` ✓（不自述成 real-shape ✗）。
        """
        if getattr(self, "_form", None) == self.H3_FORM_NAME:
            return "h3-dual" if self.dualStream["enabled"] else "h3-trunk-only"
        #: ⭐ **音频族（3 维）** ✓✗（2026-11-06 ✓）：它同样**没有** `vae_scale` ✓，若走下面的
        #: 「有 vae_scale 才算真形状」判据 ⇒ 自述成 `placeholder-1d` ✓✗（说的与做的相反 ✓：
        #: :meth:`latent_shape` 已经能给出 ``(1,128,T)`` ✓）。判据与那边**同一口径** ✓：
        #: 有 config ＋ 有潜通道 ⇒ 真形状 ✓（潜帧数由 plan 给 ✓，见 `_fill_music3_plan` ✓）。
        if dit_build.contract_of(getattr(self, "_ditFamily", None)).latent_dims == 3:
            channels = int(self._config_value("latent_channels", 0) or 0) \
                if self._config is not None else 0
            return "real-shape" if channels else "placeholder-1d"
        if dit_build.contract_of(getattr(self, "_ditFamily", None)).latent_dims == 4:
            #: ⚠️ 判据与 :meth:`latent_shape` **同一口径** ✓✗ —— 形状靠 `_config` ✓ 不靠模块 ✓：
            #: 分时驻留下「TE 在场、DiT 已退」是真正常态 ✓，那时 `init_latents` 造的**是真形状** ✓，
            #: 这里若还看 `_model` 就会自述成 `placeholder-1d` ✓✗（说的与做的相反 ✓）。
            return "real-shape" if (self._config is not None
                                    and self._config_value("vae_scale", 0)) else "placeholder-1d"
        if self._model is not None and self._config is not None \
                and self._config_value("vae_scale", 0):
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
        #: ⭐ 「此刻能不能出片」的事实 ✓ —— 一次算完再分发到三格 ✓（别在同一份自述里各算各的 ✗）
        facts = self._generation_facts()
        return {
            "name": self.name, "synthetic": self.synthetic, "realTensors": self.realTensors,
            "available": self._available, "reason": self._reason,
            "device": self._device, "torchVersion": torch_version,
            # ⚠️ 「int8 在卡上还是 CPU」默认**看不见** ✗ ⇒ 显存账对不上时只能猜 ✓✗（所以报出来 ✓）
            "int8Offload": self._int8Offload,
            # ⭐ 「卡 + CPU 联合跑」的开关与**实况** ✓（``off`` ⇒ None ✓；摆过 ⇒ 层数/字节/hook 数 ✓）
            #    ⚠️ 只报开关不够 ✗：`split` 也可能切完一层都没上卡（`fits=False` ✓）⇒ 实况一并给 ✓
            "residency": {
                "mode": self._residency,
                "budgetMib": self._residencyBudgetMib,
                "report": (self._residencyHandle.report()
                           if self._residencyHandle is not None else None),
            },
            #: ⚠️⚠️ 原来写的是 ``self._device == "cuda"`` ✗✗ —— 这一格在 ``cuda:0`` 上会答
            #: **False** ✓✗（调用方显式写卡号、或多卡机在 ``cuda:1`` ✓），而注释自己写着
            #: 「这格是**这台后端现在**在不在卡上」✓ ⇒ 语义就是判类型、不是比字面量 ✓。
            #: 后果链：`engine_readiness.device_facts` 读它 ⇒ 后端明明在卡上、能力表说不在 ✓✗。
            "cudaAvailable": device_policy.is_cuda_device(self._device),
            # ⚠️ 上面那格是「**这台后端现在**在不在卡上」✗，不是「本机有没有卡」✓✗ ——
            #    策略把某个阶段转移到 CPU 之后 `cudaAvailable` 会变 False ✓，拿它当"本机无卡"
            #    会读出「集显 / CPU ⇒ 很慢」这类**错误**结论 ✗✗（`engine_readiness.device_facts` 就踩这个口 ✓）。
            "gpuAvailable": bool(self._available and self._torch().cuda.is_available()),
            # ⭐ **设备策略的实况** ✓（2026-10-31 ✓）—— 光看 `device` 分不清「钉在 CPU」与
            #    「压力大被转移过来」✗✗，所以把策略名、逐阶段决策、当前读数一起报 ✓
            "devicePolicy": {
                "policy": self._policy,
                "requestedDevice": self._requestedDevice,
                "lastDecision": self._lastDecision,
                "phases": dict(self._phaseDecisions),
                #: ⚠️ 现测读数 ✓（``None`` ⇒ 测不到 ✓ 不是"没压力" ✗ —— 两者必须能分开 ✓）
                "pressure": self.gpu_pressure(),
                #: ⚠️⚠️ 这一格必须报**生效中的那本账** ✗✗（2026-10-31 修 ✓）：原来一律报类属性
                #: ``PART_NEED_GIB``（2.1 的 DiT 13.26 ✓），而 2512 声明链后用的是
                #: ``_chain["partNeedGiB"]``（**38.05** ✓）⇒ 读者按 13.26 估余量、结论"还剩 9 GiB"
                #: ✓✗，与决策真用的表差 24.8 GiB ✓（正是 §7 那条静默 OOM 的读数版 ✓）。
                #: 两格并存：``partNeedGiB`` = 现在按什么判 ✓、``defaultPartNeedGiB`` = 类缺省 ✓。
                #: ⚠️ `_chain` 在 `__init__` 里就置 `None` 了 ✓（见那里 ✓），所以这里能直接读 ✓。
                "partNeedGiB": (dict(self._chain["partNeedGiB"]) if self._chain
                                else dict(self.PART_NEED_GIB)),
                "defaultPartNeedGiB": dict(self.PART_NEED_GIB),
                #: ⚠️ 分时账（上面那格 ✓）与**整链同驻**账（这格 ✓）必须分开报 ✗✗：
                #: Z-Image 是 19.28 GiB **一次占满** ✓，2.1 是 14.79 GiB **一段一段** ✓ ——
                #: 混在一起报会让读者按"最重的单件"去估，差 4.5 GiB 就 OOM ✓✗。
                "chainNeedGiB": dict(self.CHAIN_NEED_GIB),
            },
            #: ⭐ **任务级调度** ✓（2026-11-01 ✓）—— `devicePolicy` 答的是"这一件放哪" ✓，
            #: 这里答的是"**现在这些字节是谁的**" ✓ ＋ "运行期压力到了哪一档" ✓：
            #: * ``ledger`` ✓：每个件占多少、在卡上还是内存 ✓（换装泄漏 / 忘回收 / 两件重叠占卡 ✓）；
            #: * ``watch`` ✓：滞回状态机 ✓（``hard`` 才动手 ✓、读数取不到 ⇒ ``source="unknown"`` ✓✗
            #:   —— "测不到"**不是**"没压力" ✓）；
            #: * ``events`` ✓：最近几条调度事件 ✓（**静默降级**是这套最坏的失败面 ✗✗）。
            "scheduling": {
                "policy": scheduler_mod.POLICY_NAME,
                "ledger": self._ledger.to_dict(),
                "watch": self._watch.to_dict(),
                "events": list(self._loadEvents[-8:]),
            },
            # ⚠️ 四个层次**分开答** ✓（混在一起就会变成"看着像能做、其实不行" ✗）：
            #    依赖齐了 ✓ → 张量是真的 ✓ → DiT 真前向（装了权重才是 ✓）→ 能不能出片 ✗
            "modelLoaded": self._model is not None,
            # ⭐ 装了**哪一族** ✓（``h3`` / ``zimage`` / ``None``=认不出 ⇒ 老路 ✓）—— 光看
            #    ``modelLoaded`` 分不出「H3 双流」与「Z-Image 单流」✗，而两者的前向口径不同 ✓✗
            "ditFamily": getattr(self, "_ditFamily", None),
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
            # ⚠️⚠️ 2026-11-01 修（原来是**写死**的两格 ✗✗）：`canGenerate` 恒 `False` ✓✗ —— 注释
            #    写着「唯一原因是真权重没到」✗，而 H3 / Z-Image / Krea2 / Qwen-Image 2.1 / Flux 早已
            #    端到端出片 ✓ ⇒ 装完整链也照样报「不能出片」✓✗；`pendingParts` 塞的是 75 行**沿革散文** ✓✗
            #    （读起来像"机制都没写"，实际早就闭合 ✓）。
            #    现在两格按**已装件的事实**算 ✓，与 `llm_backend` ③″ / `sdxl_backend` 同一口径 ✓。
            "canGenerate": facts["canGenerate"],
            "pendingParts": list(facts["pendingParts"]),
            # ⭐ **代码里还没写的**缺口 ✓ —— 与「这台后端还没装」**分开报** ✗✗（混一格就看不出
            #    该去修代码还是去装权重 ✓✗）。事实口径 = `python -m app.scripts.dit_families` 的「通路」栏 ✓
            #    ⚠️ 键名是 `codeGaps` 而**不是** `knownGaps` ✗✗（后者被 `capabilities()` 占了，
            #    语义是**环境**缺口、形状是 dict ✓ —— 见 `CODE_GAPS` 上方那段 ✓）。
            "codeGaps": list(CODE_GAPS),
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
        if self._textKind == "h3-te":
            return self._encode_h3_condition(request)
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

    def _h3_pieces(self, request: Any) -> dict[str, Any]:
        """请求 ⇒ **呈现格式的段序列** ✓ ＋ 视觉塔要吃的像素 ✓（H3 的三条口径 ✓）。

        ## 三条口径按「有没有参考素材」分 ✓（与 `conditioning.select_h3_task` 的判据同一套 ✓）
        * 有 ``reference_frames`` / ``reference_audio`` / ``reference_videos`` ⇒ ``ref2va`` ✓；
        * 否则有 ``first_frame`` ⇒ ``fl2va`` ✓（首帧那张图走 ``<Picture 1>`` ✓）；
        * 否则 ⇒ ``t2va`` ✓（纯文 ✓，段序列只有 `system + prompt` ✓）。

        ## 参考素材的**先后**照参考实现 ✓（2026-11-01 补 ✓ —— 此前是「两类以上就拒」✗）
        ``ref_items`` 的顺序＝参考 ``nodes_minimax_h3.py::MiniMaxH3ReferenceToVideo.execute`` ✓：

        1. **参考图** ✓（按请求顺序 ✓）；
        2. **每一路参考视频**：带音轨 ⇒ **先**一条 ``<Audio j>`` ✓ **再** ``<Video k>`` ✓；
        3. 全部视频之后才是**独立音频** ✓（``reference_audio`` ✓）。

        ⚠️ 为什么「独立音频排在全部视频之后」不是排版口味 ✗✗：``<Audio j>`` 的 ``j`` 是**跨全部
        音频参考**数的 ✓，而参考实现让「每路视频的声轨」占**前**几位 ✓、独立音频从 ``S+1`` 起 ✓
        （:func:`~.conditioning.video_declaration` 的 S+1 口径 ✓）—— 反过来排 ⇒ 声轨拿到 ``S+1`` ✓✗、
        文本说的号与采样侧 ``_dual_references`` 的块序**对不上** ✓，而形状全对 ✓、不报错 ✓。

        ⚠️ 声轨**不从请求里另给** ✗：容器自带音轨 ⇒ ``media.load_video_tensor`` 的 ``hasAudio``
        就是那一路有没有声轨 ✓（本仓的请求契约里 ``reference_audio`` 只有**独立音频** ✓）。
        这样「第 k 路的声轨」不会与「第 j 条独立音频」编号错位 ✓。

        ## 视频的**帧怎么取**（与采样侧同一份素材 ✓）
        按 **2 fps** 抽帧 ✓（``h3_prompt.VIDEO_FPS`` ✓ —— 参考 ``sample_idx = range(0, n, FPS // 2)`` ✓），
        奇数帧**重复末帧**补齐 ✓、时间戳同步补末值 ✓（:func:`~.qwen3vl_pixels.pad_frames_even` 的
        配套口径 ✓）。⚠️ **不按成片时长截断** ✗✗（参考实现会 ``frames[:length]`` ✓ 再对齐 ``17k+5`` ✓）：
        本仓采样侧的参考视频块用的是**整段**解码 ✓（``_dual_references`` 不截 ✗）—— 只在文本侧截
        ⇒ 文本说"5 秒里这几帧" ✓、潜变量却是 12 秒的整段 ✓，两边**描述的不是同一段素材** ✓✗。
        要么两边都截 ✓、要么都不截 ✓；本仓选**都不截** ✓（素材原样 ✓，规则单一 ✓）。
        """
        from PIL import Image  # noqa: PLC0415

        from app.services.engine import h3_prompt as h3_prompt_mod  # noqa: PLC0415
        from app.services.engine import media as media_mod  # noqa: PLC0415
        from app.services.engine import qwen3vl_pixels as pixels_mod  # noqa: PLC0415

        torch = self._torch()
        frames = tuple(str(item) for item in (getattr(request, "reference_frames", None) or ()))
        audios = tuple(str(item) for item in (getattr(request, "reference_audio", None) or ()))
        videos = tuple(str(item) for item in (getattr(request, "reference_videos", None) or ()))
        first = getattr(request, "first_frame", None)

        def load(path: str) -> Any:
            import numpy as np  # noqa: PLC0415

            with Image.open(path) as handle:
                array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            return torch.from_numpy(array.copy())

        #: ⚠️ **硬首帧 ＋ 参考素材**同时要 ⇒ **拒** ✓✗（与 `conditioning.select_h3_task` 的第 2 条同一
        #: 口径 ✓）：FL2VA 通道不支持参考 ✓、Ref2VA 通道不锁首帧 ✓ —— 随便挑一个就是**静默丢**另一个 ✓✗。
        if first and (frames or audios or videos):
            raise TorchBackendUnavailable(
                "本段**同时**给了「硬首帧」与「参考素材」✗ ⇒ 不可能同时满足 ✓（FL2VA 锁首帧但不吃参考 ✓、"
                "Ref2VA 吃参考但不锁首帧 ✓）—— 二选一 ✓：要么去掉参考素材 ✓，要么把首帧当"
                "**参考图**放进 `reference_frames` ✓（那时它走 `<Picture 1>` ✓、不锁尾帧 ✓）",
                reason="pending")

        images: list[Any] = []
        refs: list[dict[str, Any]] = []
        video_frames: list[Any] = []
        if frames:
            for path in frames:
                images.append(load(path))
                refs.append({"type": "image", "path": path})

        for path in videos:
            decoded, facts = media_mod.load_video_tensor(path)      # (1, 3, T, H, W) ∈ [-1, 1]
            full = decoded[0].permute(1, 2, 3, 0)                   # (T, H, W, 3)
            #: ⚠️ 帧率读不出来 ⇒ **当场拒** ✓✗（`float(None)` 会抛 TypeError ✓ —— 报错指向
            #: 「float() argument」，看不出是「这个容器没有帧率」✓；而随便猜一个 ⇒ 抽帧与时间戳
            #: 不同源 ✓、形状全对 ✓✗）
            rate = float(facts.get("fps") or 0.0)
            if rate <= 0:
                raise TorchBackendUnavailable(
                    f"参考视频读不出帧率 ✗：{path}（抽帧口径是 {h3_prompt_mod.VIDEO_FPS:g} fps ✓ —— "
                    f"没有帧率就没法照它抽 ✓，猜一个 ⇒ 时间戳与素材不同源 ✓ 且不报错 ✗）",
                    reason="pending")
            step = max(1, int(round(rate / h3_prompt_mod.VIDEO_FPS)))
            sampled = full[::step]
            if not int(sampled.shape[0]):
                #: ⚠️ 抽帧后一帧不剩 ⇒ **当场拒** ✓✗（空序列会让下面 `stamps[-1]` 抛 IndexError ✓，
                #: 报错指向"列表下标越界" ✓、看不出是"这个视频没有可用帧" ✓）
                raise TorchBackendUnavailable(
                    f"参考视频**一帧都没抽出来** ✗：{path}（按 {h3_prompt_mod.VIDEO_FPS:g} fps 抽 ⇒ "
                    f"空序列 ✓）—— 换一个能正常解码的文件 ✓", reason="pending")
            #: 值域换回 **uint8** ✓：`process_image` 的口径是"先 ÷255 再减均值" ✓
            #: （喂 [-1,1] 的浮点进去 ⇒ 特征整体偏移 ✓、形状全对 ✓✗）。
            sampled = ((sampled + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
            stamps = list(h3_prompt_mod.default_timestamps(int(sampled.shape[0])))
            padded = pixels_mod.pad_frames_even(sampled)
            if int(padded.shape[0]) != int(sampled.shape[0]):
                stamps.append(stamps[-1])       #: 补的那帧时间戳**重复末值** ✓（不是再 +1/2 秒 ✗）
            if facts.get("hasAudio"):
                #: ⭐ 声轨标签**排在它自己那路视频之前** ✓（见方法头第 2 条 ✓）
                refs.append({"type": "audio", "path": path, "soundtrack": True})
            refs.append({"type": "video", "path": path, "frames": int(padded.shape[0]),
                         "timestamps": stamps})
            video_frames.append(padded)

        for path in audios:
            refs.append({"type": "audio", "path": path})

        #: ⚠️ **两条口径互斥** ✓✗（`presentation` 同时收到 `refs` 与 `images` 会当场 `ValueError` ✓）：
        #: ``refs`` = ref2va ✓（要标签＋视觉块按请求顺序交错 ✓）、``images`` = fl2va ✓（首帧那张 ✓）。
        #: 把两者一起传 ⇒ 直接炸 ✓；而不传 ``images`` 又会丢视觉块 ✓✗ ⇒ 只能**二选一** ✓。
        prompt = str(request.prompt or "")
        if refs:
            pieces = h3_prompt_mod.presentation(prompt, refs=refs)
        elif first:
            images.append(load(str(first)))
            pieces = h3_prompt_mod.presentation(prompt, images=images)
        else:
            pieces = h3_prompt_mod.presentation(prompt)
        return {"pieces": pieces, "images": images, "videos": video_frames, "refs": refs,
                "task": "ref2va" if refs else ("fl2va" if first else "t2va")}

    def _encode_music3_condition(self, request: Any) -> dict[str, Any]:
        """提示串 → **音频语义帧** ``(F, 32768)`` ✓（Music3 的"条件"不是一次编码 ✓✗）。

        ## 与另几座塔的三处根本不同（每一处写错都**不报错** ✓✗）

        ① **条件有时间维度** ✓：``F = 25 × seconds`` ✓（``AUDIO_FRAMES_PER_SECOND`` ✓）。它是**量**，
        不是一个向量 ⇒ 必须与 ``plan.frames`` **逐帧一致** ✓✗ —— 多一帧少一帧都不炸 ✓，
        DiT 拿到的是"比歌长/短"的条件 ✓（症状＝尾巴越来越空 ✓）。所以这里**照 plan 的同一算式
        重算** ✓（``round(seconds × 25)`` ✓），对不上就**当场拒** ✗（不许"按自己的数跑" ✓）。

        ② **词表焊在权重里** ✓：盘上没有配套 ``tokenizer.json`` ✓ ⇒ 用 ``self._tokenizer``
        （装配时从 safetensors 的 ``tokenizer_json`` 建的 ✓，见 :meth:`attach_music3_te` ✓）。

        ③ **CFG 在自回归内部** ✓✗：``generate`` 一次吃**两条支路**（正文 ＋ 用 ``<|audio_cfg|>``
        顶掉正文段的那条 ✓）并按官方 ``CFG_SCALE`` / ``CFG_TOP_K`` 做截断式引导 ✓
        ⇒ 这里返回的条件**已经是引导过的** ✓。

        ⚠️ 因此 ``negative`` 一律 **None** ✓✗：DiT 层再要一条负向条件就得**再跑一遍完整自回归** ✓
        （几分钟量级 ✓、且要与正向严格同长 ✓）。这一格留空是**如实的**"没接" ✓，不是"不用负向" ✓ ——
        回执 ``byBranch`` 里写清楚 ✓。
        """
        self._gate()
        torch = self._torch()
        from app.services.engine import te_music3 as te_mod  # noqa: PLC0415

        if self._textEncoder is None or self._tokenizer is None:
            raise TorchBackendUnavailable(
                "Music3 要**语义塔 ＋ 内嵌词表**两样 ✓✗（现在 "
                f"塔={'有' if self._textEncoder is not None else '没有'} ✓、"
                f"词表={'有' if self._tokenizer is not None else '没有'} ✓）"
                " ⇒ 先 ``attach_music3_te()`` ✓（它是生成器不是编码器 ✓，缺了连条件都造不出来 ✓）",
                reason="pending")
        caption = str(request.prompt or "").strip()
        lyrics = str(getattr(request, "lyrics", "") or "").strip()
        try:
            seconds = float(request.seconds)
        except (TypeError, ValueError):
            seconds = 0.0
        cond = int(round(seconds * int(te_mod.AUDIO_FRAMES_PER_SECOND)))
        if cond <= 0:
            raise TorchBackendUnavailable(
                f"Music3 的条件帧算成 {cond} 帧 ✗（``seconds``={request.seconds!r} ✓）"
                " —— 时长只认 ``seconds`` ✓（音频没有「像素帧」可数 ✓）", reason="pending")
        if cond > int(te_mod.MAX_AUDIO_FRAMES):
            raise TorchBackendUnavailable(
                f"Music3 要 {cond} 条件帧 ✗ > 上限 {int(te_mod.MAX_AUDIO_FRAMES)} ✓"
                f"（＝ {int(te_mod.MAX_AUDIO_FRAMES) / int(te_mod.AUDIO_FRAMES_PER_SECOND):g} 秒 ✓）"
                " —— 这是塔的硬上限 ✓，超了不是慢而是编不出 ✓", reason="pending")
        text = te_mod.build_prompt(caption, lyrics)
        ids = list(self._tokenizer.encode(text, add_special_tokens=False))
        if len(ids) > int(te_mod.MAX_PROMPT_TOKENS):
            raise TorchBackendUnavailable(
                f"Music3 提示串 {len(ids)} token ✗ > 上限 {int(te_mod.MAX_PROMPT_TOKENS)} ✓"
                " —— （超了要**截提示词** ✓，不能截音频 ✓；本仓不自动截 ✗：那会静默少一句歌词 ✓✗）",
                reason="pending")
        if len(ids) < 3:
            raise TorchBackendUnavailable(
                f"Music3 提示串只有 {len(ids)} token ✗ —— ``build_prompt`` 的正常形状是"
                " ``<|im_start|>…<|im_end|><|audio_start|>`` ✓，至少 3 个 ✓（少于此 ⇒ 词表对不上 ✓）",
                reason="pending")
        tensor = torch.tensor([ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            frames = self._textEncoder.generate(
                tensor, seed=int(request.seed), max_audio_frames=cond,
                cfg_scale=float(te_mod.CFG_SCALE), top_k=int(te_mod.CFG_TOP_K))
        got = int(frames.shape[0]) if hasattr(frames, "shape") and len(frames.shape) >= 1 else 0
        if got != cond:
            #: ⚠️ 参考侧是"生成到停止符就停" ✓ ⇒ 短了**是可能的** ✓（不是 bug ✓✗）。但**不许**
            #: 拿短条件去喂一个按 cond 定长的 latent ✗：DiT 会拿零填充补齐 ✓、尾部变成越来越空的
            #: 拖音 ✓✗。两种处理都可（补零 or 缩 latent ✓），本仓选**报错**：让调用方显式选 ✓。
            raise TorchBackendUnavailable(
                f"Music3 自回归只生成 {got} 帧 ✗（要 {cond} ✓）—— 早停 ✓（遇到停止符 ✓）"
                f"或时长估算偏大 ✓；pipeline 已按 ``seconds`` 定 latent 长度 ✓，"
                f"这里**不替它补零** ✗（补零 ⇒ 尾部一段越来越空的拖音 ✓✗）", reason="pending")
        #: ⚠️⚠️ **必须在边界这里搬一次卡** ✗✗（2026-11-06 首次真跑栽的第二处 ✓）：
        #: `te_music3.generate` 是**有意** `return ….to(device="cpu")` 的 ✓（生成过程要腾卡 ✓、
        #: 条件只有 32 MB/20 秒 ✓），而 `denoise` 一进门就核「condition 与后端同设备」✓✗ ⇒
        #: 原样往下传 ⇒ `调用方给的 condition 在 cpu，而后端在 cuda:0` ✓。
        #: 读起来像"调用方忘了搬" ✓，其实是**上一环故意留在 CPU** ✓ —— 交接点只有这里一处 ✓。
        frames = frames.to(device=self._device_torch(), dtype=self._compute_dtype())
        return {
            "positive": frames.unsqueeze(0),
            "negative": None,
            "ids": ids,
            "tokens": len(ids),
            "truncated": False,
            "textEncoder": "music3-ar",
            "conditioning": {
                "kind": "music3-audio-frames",
                "frames": int(got),
                "frameRate": int(te_mod.AUDIO_FRAMES_PER_SECOND),
                "dim": int(frames.shape[-1]),
                "seconds": float(got) / float(te_mod.AUDIO_FRAMES_PER_SECOND),
            },
            "byBranch": {
                "positive": {"kind": "music3-ar", "frames": int(got), "tokens": len(ids),
                             "cfg": float(te_mod.CFG_SCALE), "topK": int(te_mod.CFG_TOP_K),
                             "lyrics": len(lyrics)},
                "negative": None,
            },
        }

    def _encode_h3_condition(self, request: Any) -> dict[str, Any]:
        """**真**两塔串联 ✓：段序列 → 视觉塔 → 拼 embeds → 语言塔 ⇒ ``hidden`` ＋ ``tags`` ✓。

        ⚠️ ``tags`` 是这一环的**另一半产物** ✓✗：``h3_prompt.token_tags`` 只把视觉占位 token
        那几个位置标 0 ✓（走 video 行 ✓），其余标 1 ✓（走 text 行 ✓）—— 丢了它 ⇒ 视觉块
        **整段**用错 adaLN 行 ✓、形状与数值都没报错 ✓✗（见 `h3_form.mod_segments_for` ✓）。
        """        
        prepared = self._h3_pieces(request)
        #: ⚠️ ``merge`` **从视觉塔本体现取** ✓✗：它决定 ``<|image_pad|>`` 的个数（``(h/m)·(w/m)`` ✓）
        #: —— 拿配置文件里的字面值或写死 2 都行不通 ✓✗（重导出改了 ``spatial_merge_size`` ⇒
        #: 占位个数与特征个数**双双不同** ✓、形状全对 ✓✗，见 :func:`~.h3_encode.plan_sequence` ✓）。
        #: 塔不在场 / 没这个属性 ⇒ **当场拒** ✓（不许回落成 2 ✗）。
        visual = getattr(self._textEncoder, "visual", None)
        merge = int(getattr(visual, "spatial_merge_size", 0) or 0)
        if merge < 1:
            raise TorchBackendUnavailable(
                "文本编码器上没有视觉塔的 `spatial_merge_size` ✗ ⇒ 不知道 "
                "`<|image_pad|>` 该占多少个 ✓（猜一个 ⇒ 特征与位置错位 ✓ 且不报错 ✗；"
                "真正该走的是 `attach_h3_text` ✓，它把两半塔合成一个编码器 ✓）", reason="pending")
        with self._torch().no_grad():
            condition = self._textEncoder.encode(prepared["pieces"], tokenizer=self._tokenizer,
                                                 images=prepared["images"] or None,
                                                 #: 多路视频传**列表** ✓（一路一个 ✓，按 `<Video k>` 顺序 ✓）；
                                                 #: 单路传张量也行 ✓ —— 但这里**统一传列表** ✓，少一个分叉 ✓。
                                                 video=prepared["videos"] or None,
                                                 merge_size=merge)
        shape = tuple(condition.hidden.shape)
        if len(shape) != 3 or int(shape[0]) != 1:
            raise TorchBackendUnavailable(
                f"语言塔输出应为 (1, L, text_dim) ✓，收到 {shape} ✗", reason="pending")
        tags = [int(value) for value in condition.tags.tolist()]
        return {"positive": condition.hidden[0], "ids": list(condition.ids),
                "tokens": condition.seq_len, "originalTokens": condition.seq_len,
                "truncated": False, "textEncoder": "h3-qwen3vl",
                #: ⭐ 文本段的 adaLN 行号 ✓（`0` = video 行 ✓、`1` = text 行 ✓）
                "tags": tags,
                "spans": list(condition.spans), "grids": [list(g) for g in condition.grids],
                "h3Task": prepared["task"],
                "h3Refs": [str(item.get("type")) for item in (prepared.get("refs") or ())],
                "visionBlocks": len(condition.spans)}

    def _encode_wan22_text(self, request: Any) -> dict[str, Any]:
        """Wan 2.1/2.2 的文本条件 ✓：**UMT5-XXL**（4096 宽 ✓）＋ 参考图条件（I2V 那 20 通道 ✓）。

        ## 为什么不能退回占位

        出图族（`qwen_image`/`krea2`/`zimage`）没挂文本塔时用 sha256 造状态位**能跑**（只影响画风 ✓），
        但 Wan 的 cross-attn **只吃文本状态** ✓：宽度 / 层数不对就当场 shape 报错 ✓；即使宽度凑对 ✓，
        占位状态没有任何语义 ✓ ⇒ 出片是一片**会动的糊**✓✗，比报错难查得多 ✓。所以这里**硬拒** ✓。

        ## 三条与出图族不同的口径

        1. **padding 位置要置 0** ✓：Wan 的 cross-attn 不吃 attention mask ✓（`:class:`WanDiT`
           只收 ``context`` ✓），不置 0 就是拿 pad 行的**真数值**当语义 ✓✗；
        2. **序列长 512** ✓：官方 Wan 口径 ✓（`Umt5TeRuntime.max_length` ✓），短提示词补 pad ✓、
           长提示词截尾但仍以 ``</s>`` 收尾 ✓（该收尾由分词器保证 ✓）；
        3. **I2V 的两段条件在这里一起做** ✓：参考图 → VAE → 16 通道 latent（**整段 T 帧** ✓，
           首帧是参考图、其余是 0.5 灰 ✓ —— 与 ``WanImageToVideo`` 逐字同构 ✓），掩码 4 通道
           首拍为 1 ✓。VAE 只在 encode 段挂着 ✓（`:meth:`enter_phase` ✓），所以**必须**在这一拍做完 ✗
           留到 sample 段就没 VAE 了 ✓✗。
        """
        torch = self._torch()
        runtime = getattr(self, "_textConfig", None)
        encoder = getattr(self, "_textEncoder", None)
        tokenizer = getattr(self, "_tokenizer", None)
        if encoder is None or tokenizer is None or int(getattr(runtime, "output_dim", 0) or 0) <= 0:
            raise TorchBackendUnavailable(
                "Wan 2.1/2.2 的文本塔必须是 **UMT5-XXL** ✗（`umt5_xxl_fp8_e4m3fn_scaled.safetensors` ✓）"
                " —— 现在没挂 ⇒ 拒绝拿 sha256 占位顶替 ✓（Wan 的 cross-attn 只吃真文本状态 ✓，"
                "占位状态出片是一片会动的糊 ✓✗）",
                reason="attach")
        #: ⚠️⚠️ 2026-11-02 **真跑抓到的第四处** ✓✗：这里原先写 ``self._device()`` ✓，
        #: 而 ``_device`` 是**字符串属性** ✓（``"cuda"`` ✓，见 ``__init__`` 与 :attr:`device` ✓）
        #: ⇒ ``TypeError: 'str' object is not callable`` ✓，报在「文本编码失败」那句里 ✓
        #: （读起来像分词器/塔坏了 ✓✗）。具体 :class:`torch.device` 的正确取法是
        #: :meth:`_device_torch` ✓（它顺手补卡号 ✓ —— ``"cuda"`` vs ``cuda:0`` ✓）。
        device = self._device_torch()
        limit = int(getattr(runtime, "max_length", 0) or 512)
        want = next(encoder.parameters()).dtype

        def states(text: str) -> tuple[Any, int, int]:
            ids, mask = tokenizer.encode(runtime.prompt(text), max_length=limit)
            ids_t = torch.as_tensor([ids], dtype=torch.long, device=device)
            mask_t = torch.as_tensor([mask], dtype=torch.long, device=device)
            with torch.no_grad():
                hidden = encoder(input_ids=ids_t, attention_mask=mask_t)
            #: ⭐ padding 位置置 0 ✓（Wan cross-attn 不吃 mask ✓）—— 只清 pad ✓，`</s>` 那行留着 ✓
            hidden = hidden * mask_t.unsqueeze(-1).to(hidden.dtype)
            return hidden.to(want), int(mask_t.sum().item()), int(ids_t.shape[1])

        prompt = str(getattr(request, "prompt", "") or "")
        negative = str(getattr(request, "negative", "") or "")
        positive, pos_tokens, pos_seq = states(prompt)
        negative_states, neg_tokens, _ = states(negative)
        plan = getattr(self, "_plan", None)
        condition = self._wan_i2v_condition(request, plan, want=want)
        note = (
            f"真 UMT5-XXL 编码器 ✓（{int(getattr(runtime, 'output_dim', 0))} 宽 ✓、"
            f"序列 {pos_seq} ✓、有效 token {pos_tokens} ✓、padding 位置置 0 ✓）")
        if condition is None:
            note += "；**没给首帧** ⇒ 走的是纯文本路线 ✓（I2V 权重没条件图时仅 16 通道可用 ✗，"
            note += "所以本份权重会当场拒 ✓）"
        else:
            note += (f"；首帧条件已拼 ✓（{condition['shape']} ⇒ 送 DiT 的是 36 通道 ✓，"
                     f"掩码首拍 1 ✓、其余 0 ✓，参考 latent 首帧来自 {condition['source']} ✓）")
        return {
            "positive": positive,
            "negative": negative_states,
            "note": note,
            "tokens": pos_tokens,
            "originalTokens": pos_tokens,
            "truncated": False,
            "textEncoder": "umt5-xxl",
            "wan22": {
                "textDim": int(getattr(runtime, "output_dim", 0) or 0),
                "seqLen": pos_seq,
                "maxLength": limit,
                "i2v": None if condition is None else {
                    "concat": int(condition["concat"]),
                    "maskFrames": int(condition["mask_frames"]),
                    "layout": list(condition["layout"]),
                    "source": condition["source"],
                },
            },
        }

    def _wan_i2v_condition(self, request: Any, plan: Any, *, want: Any) -> dict[str, Any] | None:
        """把首帧变成 I2V 的 **20 通道**条件 ✓（掩码 4 ＋ 参考 latent 16 ✓，顺序按 `I2V_LAYOUT` ✓）。

        参考实现（``comfy_extras/nodes_wan.py::WanImageToVideo`` ✓）是**整段像素**一起编码 ✓：
        第 0 帧是首帧 ✓、其余补 0.5 灰 ✓（``[-1,1]`` 里就是 0 ✓）。**不要**只编码一张再零填充 ✗ ——
        灰帧的 latent 不是 0 ✓✗，少一段会让"没条件的那几拍"整体偏暗 ✓。
        """
        torch = self._torch()
        from app.services.engine import dit_wan22 as wan22_mod  # noqa: PLC0415
        from app.services.engine import media as media_mod  # noqa: PLC0415

        path = getattr(request, "first_frame", None)
        if not path:
            return None
        if getattr(self, "_vae", None) is None:
            raise TorchBackendUnavailable(
                "Wan I2V 的首帧条件要过 VAE ✗，而这一拍手上没有 VAE ✓ —— "
                "`encode` 段必须同时留着 vae ✓（`declare_chain(phase_parts={'encode': ('text', 'vae')})` ✓）",
                reason="attach")
        shape = self.latent_shape(plan) if plan is not None else None
        if not shape or len(shape) != 5:
            raise TorchBackendUnavailable(
                "Wan I2V 要按潜变量尺寸编码首帧 ✗，但这一拍算不出 5 维潜变量 ✗"
                f"（拿到的是 {shape} ✓；`bind_plan` 没跑或 vae_scale / latent_frames 缺 ✓）",
                reason="pending")
        b, _c, t, h, w = (int(v) for v in shape)
        scale = int(self._config_value("vae_scale", 0) or 0) or 8
        frames = 4 * (t - 1) + 1
        height, width = h * scale, w * scale
        image, info = media_mod.load_image_tensor(path, width=width, height=height)
        image = image.to(device=self._device_torch(), dtype=want)
        #: ⚠️ **整段像素**一起编码 ✓（首帧＝参考图 ✓，其余 0.5 灰 ⇒ `[-1,1]` 里是 0 ✓）
        video = torch.zeros((1, 3, frames, height, width), dtype=want, device=self._device_torch())
        video[:, :, 0] = image[:, :, 0]
        with torch.no_grad():
            image_latents = self._encode_pixels(video)
        if image_latents.ndim == 4:            # ⚠️ 4 维是单帧 `(b,c,h,w)` ✓ ⇒ 补时间轴 ✓
            image_latents = image_latents.unsqueeze(2)
        got_c = int(image_latents.shape[1])
        if got_c != 16:
            raise TorchBackendUnavailable(
                f"Wan 的参考 latent 要 16 通道 ✗，VAE 实际给 {got_c} ✓"
                f"（这份 VAE 不是 Wan 2.1 那套 ⇒ 16 通道/8× ✗）", reason="pending")
        got_t = int(image_latents.shape[2])
        if got_t != t:
            raise TorchBackendUnavailable(
                f"参考 latent 时间维 {got_t} ✗ ≠ 潜变量 {t} ✓ "
                f"（像素喂了 {frames} 帧 ✓；Wan 的 VAE 是 4 倍时间压缩 ⇒ 帧数必须 4k+1 ✓）",
                reason="pending")
        mask = wan22_mod.i2v_condition_mask(
            frames=t, height=h, width=w, dtype=want, device=self._device_torch(), batch=b)
        #: ⭐ 只存**两段**，拼接交给 :func:`dit_wan22.i2v_concat` ✓（顺序只有一份真源 ✓）
        self._i2vImage = image_latents.to(want)
        self._i2vMask = mask
        return {
            "concat": 36,
            "mask_frames": t,
            "shape": (b, got_c + 4, t, h, w),
            "source": str(getattr(info, "name", "") or Path(str(path)).name),
            "layout": [f"{name}:{width_}" for name, width_ in wan22_mod.I2V_LAYOUT],
        }

    def encode_text(self, request: Any) -> dict[str, Any]:
        """提示词 → 条件 ✓：挂了文本编码器就走**真编码器** ✓，否则用 sha256 占位 ✓（都如实标 ✓）。

        ⭐ **Edit 2511 的参考图**（2026-11-01 接上 ✓，**多图**同轮接 ✓）：装了 Edit 权重
        （`ref_timestep_zero` ✓）⇒ 这里一次办完三件事 ✓：

        1. 参考图过**视觉塔** ⇒ ``image_embeds`` ✓（语义那一路 ✓；**一张一块** ✓，多图就是一列 ✓）；
        2. 正向提示词走 :meth:`Qwen25TeRuntime.prompt_edit` 模板 ✓（`<|vision_start|>` 在 **user 段**、
           正文**之前** ✓✗ 写成 system 段就静默变成"描述一张不存在的图" ✓）；多图时按张数选
           Plus 那套模板 ✓（``Picture i: <视觉块>`` 逐图一个槽 ✓）；
        3. 参考图过 **VAE** ⇒ ``ref_latents`` ✓，存进 ``self._refLatents``（**一列** ✓）供
           :meth:`denoise` **每一步**带进 DiT ✓。

        ⚠️ 反向提示词**不带图** ✓：照 comfy 的做法负向走普通模板 ✓（负向那句是"不要什么" ✓，
        跟参考图没有语义关系 ✓；硬塞同一串图像 token 只会让负向也不再是纯文本 ✓）。
        """
        self._gate()
        if getattr(self, "_form", None) == self.H3_FORM_NAME:
            return self._encode_text_states(request)
        #: ⭐ **SEEDVR2 修复线没有文本塔** ✓（2026-11-01 接上 ✓）：这一步在出图族里是
        #: 「提示词 → 条件」✓，在这一族是「**源视频 → 17 通道条件**」✓（见
        #: :meth:`_encode_seedvr2_conditioning` ✓）。⚠️ 它的 ``positive`` 是**张量**不是文本状态 ✓
        #: —— `pipeline._branches` 会把它当正向条件原样交给 `denoise` ✓
        #: （`negative=None` ⇒ 单次前向 ✓ 不做 CFG ✓，这一族官方就是 cfg=1 ✓）。
        if getattr(self, "_ditFamily", None) == "seedvr2":
            return self._encode_seedvr2_conditioning(request)
        #: ⭐ **Wan 2.1/2.2**（2026-11-02 接上 ✓）：文本塔是 UMT5-XXL ✓，且 I2V 的首帧条件
        #: **必须在这一拍**做完 ✓（VAE 只在 encode 段挂着 ✓，到 sample 段就退掉了 ✓✗）。
        if getattr(self, "_ditFamily", None) == "wan":
            return self._encode_wan22_text(request)
        #: ⭐ **Music3**（2026-11-06 接上 ✓）：这一族的"文本塔"是**自回归生成器** ✓✗ ——
        #: 它不吃"编码一次"就完事 ✓，要按**时长**逐帧生成 `25/s` 的音频语义帧 ✓。
        #: ⚠️ 判据用 ``_textKind``（装配时就钉死 ✓）而不是 ``family ==`` ✓：与 `dit_build` 的
        #: 「族只认表」同口径 ✓；也保证"装错了塔"（比如塞了 Qwen）在这一拍立刻现形 ✓。
        if getattr(self, "_textKind", None) == "music3-ar":
            return self._encode_music3_condition(request)
        if self._textEncoder is not None:
            torch = self._torch()
            config = self._textConfig
            results: dict[str, Any] = {}
            meta: dict[str, Any] = {}
            positive_ids: list[int] = []
            #: ⭐ Edit 2511：参考图的那两样 ✓（非 Edit 族 ⇒ ``None`` ✓；Edit 族没给图 ⇒ **当场报错** ✗✗）
            #: ⚠️ ``latents`` 是**一列** ✓（单图也是长度 1 ✓）—— 统一成列表 ⇒ `denoise` 不用分叉 ✓；
            #: 单图存裸张量、多图存列表那种写法会让"换回 t2i 没清空"更难发现 ✗。
            edit = self._edit_reference(request)
            self._refLatents = list(edit["latents"]) if edit is not None else None
            # ⭐ 模板**只在配置里加**一次 ✓（`Qwen3TeRuntime.prompt` ✓）；参考 TE 没这个方法 ✓ ⇒ 原样送 ✓。
            wrap = getattr(config, "prompt", None)
            #: Edit 2511 用的是**另一套**模板 ✓（同一个类里的 `prompt_edit` ✓）；没这个方法 ⇒ 说明
            #: 挂的不是 Qwen2.5-VL 那座塔 ✓ ⇒ 下面注入时当场报错 ✓（不静默退回纯文本 ✗）。
            wrap_edit = getattr(config, "prompt_edit", None)
            for name, text in (("positive", request.prompt), ("negative", request.negative)):
                raw = "" if text is None else str(text)
                use_edit = edit is not None and name == "positive"
                if use_edit and callable(wrap_edit):
                    #: ⚠️ 张数必须与嵌入块数**一致** ✓：模板少一个视觉块 ⇒ 多出来的那块没处放 ✓✗
                    #: （`_with_image_blocks` 会当场报「块数 ≠ 槽数」✓）；多一个块 ⇒ 提示词里
                    #: 没有对应锚点 ✓✗ ⇒ 两条都由注入点兜 ✓。
                    prepared = wrap_edit(raw, image_count=int(edit["count"]))
                else:
                    prepared = wrap(raw) if callable(wrap) else raw
                #: ⚠️ ``max_length`` 是**文本**预算 ✓✗：图是排在文本**前面**的 ✓ ⇒ 一张 384² 图
                #: 约 196 token ✓、三张就 588 ✓ > 512 ⇒ 截断正好切掉**尾部那句指令** ✓✗
                #: （截断的是"把背景换成海边"这类话 ✓，图像条件留着 ✓）—— ``truncated=True`` 只说明
                #: "截了" ✓，不说明"编辑能力没了" ✓✗。参考实现那座塔干脆不设上限 ✓；
                #: 这里折中：把图的 token 数**加到文本预算之上** ✓（文本仍保 512 ✓）。
                limit = int(config.max_length)
                if use_edit:
                    limit += int(edit["report"]["vision"]["tokens"])
                ids, original, truncated = te_mod.tokenize_prompt(
                    self._tokenizer, prepared, max_length=limit)
                tensor = torch.tensor([ids], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    if use_edit:
                        #: ⚠️ `image_embeds` 走**关键字** ✓；参考 TE（`te_mod` 那几座）没这个参数 ✗
                        #: ⇒ 只在 Edit 这一支传 ✓（无条件传会给所有族一个 `TypeError` ✓✗）。
                        results[name] = self._textEncoder(tensor, image_embeds=edit["embeds"])
                    else:
                        results[name] = self._textEncoder(tensor)
                if name == "positive":
                    positive_ids = list(ids)
                meta[name] = {"tokens": len(ids), "originalTokens": original,
                              "truncated": truncated, "templated": callable(wrap) or use_edit}
                if use_edit:
                    meta[name]["imageTokens"] = int(edit["report"]["vision"]["tokens"])
                    meta[name]["images"] = int(edit["count"])
                    meta[name]["maxLength"] = limit
                    meta[name]["template"] = "edit" if int(edit["count"]) == 1 else "edit-plus"
            # ⚠️ `ids` 必须是**真的 id 列表** ✓（第一版把 meta 字典塞进了 `ids` ✗ ⇒ 调用方
            #    `len(...)` 拿到的是"字典键数"而不是 token 数 ✓ —— 自检 ⑯ 当场红 ✓ 已改 ✓）；
            #    逐分支明细放 `byBranch` ✓。
            payload = {**results, "ids": positive_ids, "tokens": len(positive_ids),
                       "originalTokens": meta["positive"]["originalTokens"],
                       "truncated": meta["positive"]["truncated"],
                       "byBranch": meta,
                       "textEncoder": self._textKind or "reference-untrained"}
            if edit is not None:
                #: 回执里**如实写**两路口径 ✓（382² / 1024² 这类数字只有这里能查 ✓）；
                #: `ids`/`tokens` 是**注入后**的序列事实 ✓（`<|image_pad|>` 那一格已换成 n 条 ✓）。
                payload["edit"] = edit["report"]
            return payload
        return {"positive": self._condition(request.prompt),
                "negative": self._condition(request.negative),
                "textEncoder": "sha256-placeholder"}

    def _edit_reference(self, request: Any) -> dict[str, Any] | None:
        """Edit 2511 的参考图在哪 ✓ —— 非 Edit 族 ⇒ ``None`` ✓；Edit 族没给图 ⇒ **当场报错** ✗✗。

        ⚠️ 「没给图就退回纯 t2i」是**必须拒**的 ✓✗：那样出来的图与参考图毫无关系 ✓、
        却一条报错都没有 ✓✗ —— 正是本仓最不能接受的那类错 ✓。

        ⭐ **多图**（2026-11-01 已接 ✓）：``referenceFrames`` 有几张就准备几份 ✓
        （第 i 张 ⇒ 模板第 i 个视觉块 ✓ ＋ ``ref_latents[i]`` ✓）；顺序**即**语义 ✓✗。
        """
        is_edit = bool(self._config_value("ref_timestep_zero", False))
        if not is_edit:
            return None
        frames = [str(item) for item in (getattr(request, "reference_frames", ()) or ())
                  if str(item).strip()]
        if not frames:
            raise TorchBackendUnavailable(
                "装的是 **Edit 2511** 的权重 ✓（权重里有 ``__index_timestep_zero__`` 哨兵 ✓）"
                "却**没给参考图** ✗ ⇒ 编辑无从谈起 ✓（`referenceFrames` 至少一张 ✓）",
                reason="input")
        return self.prepare_edit_references(frames)

    def bind_plan(self, plan: Any) -> None:
        """把**本轮 plan** 递给后端 ✓（可选钩子 ✓ —— 不实现的后端零影响 ✓）。

        为什么需要它 ✗：``encode_text(request)`` 的契约里**没有 plan** ✓，出图族确实不需要 ✓
        （提示词编码与画布/帧数无关 ✓✗）；**修复族不成立** ✗✗ —— 它的"条件"**就是**源视频编码
        出的 16 通道 ✓，而"编多少帧、编成多大"正是 plan 说了算 ✓ ⇒ 两边必须看**同一份** ✓
        （两处各算一遍的症状：plan 说 96 帧、实际编出 121 帧 ✓，采样第 1 步 33 通道拼不上 ✓，
        报错指不到"算了两遍"✓）。

        ⚠️ 这里**只登记不判断** ✗ —— 尺寸/帧数的裁定在
        :meth:`_encode_seedvr2_conditioning` 里 ✓（它才是唯一读源视频的地方 ✓）。
        """
        self._plan = plan
        #: ⭐ Wan 2.2 的**双专家分界**在这里落定 ✓（`plan.wan_high_steps` ✓，非本族恒 0 ✓）。
        #:
        #: ⚠️⚠️ 判据是**σ 阈值**，不是"数第几次调用" ✗✗：`denoise` 每步的调用次数**不是 1** ✓ ——
        #: 引导开时每步 **2** 次（正向＋负向 ✓ 见 `pipeline.model_fn` ✓）、二阶采样器（heun）
        #: 再翻倍 ⇒ 每步最多 **4** 次 ✓。按调用次数数 ⇒ 在第 1/2 步就换 ✓✗（还换在负向那一趟 ✓，
        #: 同一步里正负两趟用的**不是同一个专家** ✓ ⇒ 引导被两份权重各算一次 ✓✗，不报错 ✓）。
        #: σ 则是**单调下降**的 ✓ ⇒ 同一步的全部子调用落在分界的同一侧 ✓；
        #: 阈值取 ``plan.sigmas[wan_high_steps]`` ✓（**跟着 schedule 走** ✓，不是写死的数 ✓✗）。
        _high = int(getattr(plan, "wan_high_steps", 0) or 0)
        _sig = list(getattr(plan, "sigmas", None) or [])
        self._wanHighSteps = _high
        self._wanSwitchSigma = (float(_sig[_high]) if (_high > 0 and len(_sig) > _high) else None)
        #: 每次新 run 都从「高噪」重来 ✓✗（上一轮换到低噪后 `_wanActive` 会一直是 ``low`` ✓，
        #: 不清 ⇒ 第二次跑从第一步就用低噪专家 ✓、画面糊且**没有任何提示** ✓✗）。
        self._wanActive = "high"

    # ------------------------------------------------------------------ Wan 2.2 双专家
    def register_wan_experts(self, high: str, low: str | None = None) -> dict[str, Any]:
        """登记 Wan 2.2 的**两份主干** ✓（``high`` = 高噪 ✓、``low`` = 低噪 ✓）。

        ## 为什么非要有这一格

        Wan2.2 A14B 把噪声区间拆成**两个专家** ✓：官方 UI 模板 ``video_wan2_2_14B_i2v.json``
        是**两个** ``KSamplerAdvanced`` 接力 ✓（高噪跑 ``[0, 2)`` ✓、低噪跑 ``[2, 4)`` ✓，
        切换在 **50%** ✓）。只用其中一份跑满全程也能出画面 ✓✗ —— 前段发灰 ✓、后段细节糊 ✓，
        而且**一条报错都没有** ✓（盘上两份文件名字只差 ``HighNoise`` / ``LowNoise`` 两个词 ✓）。

        ## 26.62 GiB 装不进 22.49 GiB ⇒ **换装**（不并存 ✗）

        单份 14B 反量化到 bf16 ≈ **26.62 GiB** ✓（实测本仓 ``WAN22_PART_NEED_GIB["dit"]`` ✓），
        两份并存 ≈ **53 GiB** ✗✗ —— 卡装不下 ✓、也没必要 ✓：同一时刻只有一份在算 ✓。
        所以这里**只登记路径** ✓、真换发生在 :meth:`_wan_switch_expert` ✓（走到分界那一步 ✓），
        换法是 ``load_module_weights`` **原地重装** ✓（流式 ✓ 峰值≈单张量 ✓，见 `weights` 的说明 ✓）
        —— 模块骨架、驻留切分（``split`` ✓）、设备全都不动 ✓✗。

        ⚠️ 换装**不是免费** ✓：要重读 13.31 GiB 的 fp8 文件 ✓（本机约 20~40 秒 ✓）⇒
        4 步里有 2 步是"第一步就要换" ✓；只在**真跑长视频**时才值得 ✓。
        """
        path_high = str(high or "").strip()
        if not path_high:
            raise TorchBackendUnavailable("Wan 双专家的「高噪」路径为空 ✗", reason="input")
        experts = {"high": path_high}
        path_low = str(low or "").strip()
        if path_low:
            if Path(path_low) == Path(path_high):
                raise TorchBackendUnavailable(
                    f"Wan 2.2 的**低噪**权重与高噪是同一个文件 ✗（{path_high} ✓）—— "
                    "这两个专家必须给两份不同文件 ✓（同名就是「只有一份 ⇒ 不接力」✓，"
                    "请**别登记**低噪 ✓ 而不是登记成同一份 ✓）", reason="input")
            experts["low"] = path_low
        self._wanExperts = experts
        self._wanActive = "high"
        return {"experts": dict(experts), "active": "high",
                "note": ("两份专家已登记 ✓（高噪 → 低噪，分界由 `plan.wanHighSteps` 说 ✓）；"
                         "⚠️ 只登记、**没装**第二份 ✓ —— 换装发生在采样走到分界那一步 ✓"
                         if len(experts) > 1 else
                         "只登记了**一份**主干 ✓ ⇒ 不接力 ✓（后段质量会掉 ✓，如实标 ✓）")}

    def _wan_switch_expert(self, want: str) -> dict[str, Any]:
        """把主干**原地换成**另一个专家 ✓（骨架/驻留切分不动 ✓，只重装权重 ✓）。"""
        experts = getattr(self, "_wanExperts", None) or {}
        path = experts.get(want)
        if not path:
            raise TorchBackendUnavailable(
                f"要换成 Wan 的「{want}」专家 ✗，但没登记那份权重 ✓ "
                f"（已登记 {sorted(experts)} ✓）", reason="attach")
        if self._model is None:
            raise TorchBackendUnavailable(
                "Wan 双专家换装要一份**已装好的骨架** ✗（`attach_dit` 没跑 ✓）", reason="pending")
        started = time.perf_counter()
        report = weights_mod.load_module_weights(
            self._model, Path(path), dtype=self._compute_dtype())
        if not report.complete:
            raise TorchBackendUnavailable(
                f"Wan「{want}」专家权重装不全 ✗：缺"
                f"{len(report.missing)} / 多{len(report.unexpected)} / "
                f"形状不符{len(report.mismatched)} ✓（两份专家的骨架必须完全同构 ✓）",
                reason="pending")
        self._wanActive = want
        elapsed = time.perf_counter() - started
        note = {"expert": want, "path": str(path), "seconds": round(elapsed, 2),
                "note": f"采样中途**换装**到「{want}」专家 ✓（原地重装 ✓，"
                        f"{elapsed:.1f} 秒 ✓；骨架与驻留切分未动 ✓）"}
        self._wanLastSwitch = note
        return note

    def _wan_expert_tick(self, sigma: float) -> None:
        """按 **σ 阈值**决定这一拍用哪个专家 ✓（`:meth:`denoise` 每次前向调一次 ✓）。

        ⚠️ **不能按"第几次调用"** ✗✗：每步的调用次数随配置变 ✓（引导 2 次 / heun 再翻倍 ✓），
        而阈值只有一个 ✓ ⇒ 数次数会在**半路**换 ✓、还会让同一步的正负两趟分属两个专家 ✓✗。
        σ 单调下降 ✓ ⇒ 阈值判据天然"每步一致"✓。**换了就不回头** ✓（`_wanActive` 只是记账 ✓）。

        ⚠️ 边界必须写 **``<=``** ✗✗：阈值取的是 ``plan.sigmas[wanHighSteps]`` ✓，而采样第
        ``index`` 步喂进来的正是 ``sigmas[index]`` ✓ ⇒ 第 ``wanHighSteps`` 步（0 基）那一次的 σ
        **恰好等于**阈值 ✓。写成严格 ``<`` ⇒ 那一步仍用高噪 ✓，于是 4 步里变成「高噪 3 步 ＋
        低噪 1 步」✓✗（官方是 2/2 ✓）—— 这条**不报错** ✓、只是后段细节被高噪专家重画一遍 ✓✗。
        """
        experts = getattr(self, "_wanExperts", None) or {}
        if len(experts) < 2:
            return
        if getattr(self, "_wanActive", "high") == "low":   # 已换过 ⇒ 不回头 ✓（σ 本就单调 ✓）
            return
        switch = getattr(self, "_wanSwitchSigma", None)
        if switch is None:                                  # plan 没给分界 ⇒ 全程高噪 ✓并如实标 ✓
            return
        if float(sigma) <= float(switch) + 1e-12:
            self._wan_switch_expert("low")

    def _encode_seedvr2_conditioning(self, request: Any) -> dict[str, Any]:
        """**源视频 → 17 通道 conditioning** ✓ —— SEEDVR2 修复线的"条件"那一路 ✓。

        ``17 = 退化视频 16 ＋ loss mask 1`` ✓；主干再把它与噪声的 16 拼成 ``vid_in`` 的 **33** ✓
        （拼接在 :meth:`dit_seedvr2.SeedVR2DiT.forward` 里做 ✓ —— 这里**不要**先拼成 33 ✓✗，
        那会变成「噪声 16 ＋ 条件 17 ＋ 条件 17 = 50」✓）。

        四条硬口径（都属于「不报错、只是结果不对」那一类 ✗✗）：

        ① **帧数 ``4n+1`` 向下吸** ✓（:func:`pipeline.seedvr2_target` 是唯一真源 ✓），且用
           **真解码出来的帧数**再算一遍 ✗：ffprobe 的 ``nbFrames`` 常比真能解出来的多 ✓
           （封装 / 尾部丢帧 ✓）—— 按它编会「少一帧就差一个潜帧」✓；
        ② **``mask`` 全 1** ✓（整段重绘 ✓）：SEEDVR2 官方推理就是全 1 ✓，局部修复要 per-frame
           mask 得先有"训练时怎么给"的证据 ✓，本仓没有 ⇒ **不编** ✗（写 0.5 之类的"半透明
           修复"没有任何依据 ✓✗）；
        ③ **必须走真 VAE** ✓（``_vaeKind == "seedvr2-real"`` ✓）：Wan / Flux 的 VAE 同为
           16ch / 8× ✓、同样装得下 ✓，只差 ``×0.9152`` 与「编码器出 32 通道取前 16」✓
           ⇒ 跑得下去、修出来**整片发灰** ✓✗；
        ④ **``positive`` 是张量** ✓：`pipeline._branches` 会把它当正向条件原样交给
           :meth:`denoise` ✓（后者见 ``condition`` 带 ``shape`` 就直接当 conditioning ✓）；
           ``negative=None`` ⇒ **单次前向** ✓ 不做 CFG ✓（这一族官方就是 cfg=1 ✓）。
        """
        self._gate()
        if self._vaeKind != "seedvr2-real" or self._vae is None:
            raise TorchBackendUnavailable(
                f"SEEDVR2 修复线的条件要用**它自己的 VAE** ✗（当前挂的是 "
                f"``{self._vaeKind or '未挂任何 VAE'}`` ✓）—— 先 "
                f"``attach_seedvr2_vae('vae/SEEDVR2/ema_vae_fp16.safetensors')`` ✓。\n"
                "  ⚠️ 别拿 Wan / Flux 的 VAE 顶替 ✗✗：形状（16ch / 8×）**一样** ✓、也装得下 ✓，"
                "只差 `×0.9152` 与「编码器出 32 通道、取前 16」✓ ⇒ 跑得下去、修出来**整片发灰** ✓"
                "（不报错 ✓，正是本仓最忌讳的那类 ✗）。", reason="attach")
        sources = [str(item) for item in (getattr(request, "reference_videos", ()) or ())
                   if str(item).strip()]
        if not sources:
            raise TorchBackendUnavailable(
                "SEEDVR2 是**修视频**的线 ✓ —— 它**没有**「凭空生成」这一路 ✗（主干必填 17 通道 "
                "conditioning ✓，缺了 33 通道的 ``vid_in`` 就拼不出来 ✓）。请给 "
                "``referenceVideos[0]``（要修的那条源视频 ✓）；按文字出片请走 ``h3`` ✓。",
                reason="input")
        if len(sources) > 1:
            raise TorchBackendUnavailable(
                f"SEEDVR2 一次只修**一条**源视频 ✗（收到 {len(sources)} 条 ✓）—— 多段请走 chain"
                "（每段一条请求 ✓），不静默只取第一路 ✗（那会让你以为其余也修好了 ✓✗）",
                reason="input")

        torch = self._torch()
        from . import media as media_mod  # noqa: PLC0415 —— 延迟 import：只这一族用到 ✓
        from .pipeline import seedvr2_target  # noqa: PLC0415 —— 尺寸/帧数吸附的**唯一真源** ✓

        video, facts = media_mod.load_video_tensor(sources[0])
        decoded = int(video.shape[2])
        source_h, source_w = int(video.shape[3]), int(video.shape[4])
        try:
            target = seedvr2_target({**facts, "nbFrames": decoded,
                                     "width": source_w, "height": source_h})
        except ValueError as err:
            raise TorchBackendUnavailable(
                f"源视频没法按 SEEDVR2 的口径取尺寸/帧数 ✗：{err}", reason="input") from err

        frames = int(target["frames"])
        out_w, out_h = int(target["width"]), int(target["height"])
        plan = getattr(self, "_plan", None)
        work = video[:, :, :frames]
        if (out_w, out_h) != (source_w, source_h):
            #: 5D → 4D 逐帧插值 ✓（``interpolate`` 只吃 4 维 ✓）；就近吸附差几个像素 ✓，
            #: 属"不改构图的轻微缩放" ✓（与 letterbox / crop 不同 ✗），如实进回执 ✓。
            flat = work.permute(0, 2, 1, 3, 4).reshape(-1, 3, source_h, source_w)
            flat = torch.nn.functional.interpolate(
                flat, size=(out_h, out_w), mode="bilinear", align_corners=False)
            work = flat.reshape(1, frames, 3, out_h, out_w).permute(0, 2, 1, 3, 4)
        work = work.to(self._device).contiguous()
        packed = self.encode_pixels(work, plan)
        latents = packed["latents"]
        #: ``mask`` 通道**只能 1** ✓（17−16 ✓）：写成 3 或 16 会得 19 / 32 通道 ✓
        #: ⇒ 主干 ``vid_in`` 的 Conv3d 当场抛「期望 33、实际 N」✓（这处报错还算清楚 ✓）。
        mask = torch.ones((1, 1) + tuple(latents.shape[2:]),
                          dtype=latents.dtype, device=latents.device)
        condition = torch.cat([latents, mask], dim=1)

        corrections: dict[str, list[Any]] = {}
        if plan is not None:
            for key, value in (("width", out_w), ("height", out_h), ("frames", frames),
                               ("latent_frames", int(latents.shape[2])),
                               ("fps", int(target["fps"]))):
                current = getattr(plan, key, None)
                if current != value:
                    corrections[key] = [current, value]
                    try:
                        setattr(plan, key, value)
                    except (AttributeError, TypeError):
                        pass
            warnings = getattr(plan, "warnings", None)
            if corrections and isinstance(warnings, list):
                pairs = "、".join(f"{key} {old!r} → {new!r}"
                                  for key, (old, new) in corrections.items())
                warnings.append(
                    f"plan 已按**真解码**的源视频校正 ✓：{pairs} ✓"
                    "（ffprobe 的 nbFrames 常比真能解出来的多 ✓ —— 按它编会「少一帧就差一个潜帧」✓；"
                    "源视频是唯一事实 ✓，这里只报**不改素材** ✓）")

        report: dict[str, Any] = {
            "source": str(sources[0]),
            "sourceSize": f"{source_w}×{source_h}",
            "sourceFrames": int(facts.get("nbFrames") or decoded),
            "decodedFrames": decoded,
            "frames": frames,
            "droppedFrames": max(0, decoded - frames),
            "outputSize": f"{out_w}×{out_h}",
            "fps": int(target["fps"]),
            "latentFrames": int(latents.shape[2]),
            "latentShape": [int(value) for value in latents.shape],
            "conditioningShape": [int(value) for value in condition.shape],
            "channels": {"degraded": 16, "mask": 1, "vidIn": 33},
            "mask": "全 1（整段重绘 ✓ —— SEEDVR2 官方推理就是全 1 ✓）",
            "vaeScale": 0.9152,
            "planCorrections": corrections,
            "note": (f"源 {decoded} 帧 → 修 **{frames}** 帧（``4n+1`` ✓，截尾 "
                     f"{max(0, decoded - frames)} 帧 ✓）；条件 = 退化视频 **16** ＋ mask **1** = "
                     f"**17** 通道 ✓，主干拼成 ``vid_in`` **33** ✓；本族**没有文本塔** ✓（cfg 无效 ✓）"),
        }
        self._repairCondition = condition
        self._repairSource = report
        return {"positive": condition, "negative": None,
                "textEncoder": "none（SEEDVR2 没有文本塔 ✓：条件＝源视频编码 ＋ mask ✓）",
                "repair": report, "note": report["note"]}

    def init_latents(self, plan: Any, request: Any) -> Any:
        """初始噪声 ✓ —— **真** `torch.randn` + **真**种子 ✓ ⇒ 同种子逐位可复现 ✓。

        * **装了真模型**（且配置给了 ``vae_scale`` ✓）⇒ 造**真形状**的潜变量 ✓
          ``(1, C, 潜帧, H/vae, W/vae)`` ✓ ⇒ 真前向才吃得下 ✓；
        * 否则 ⇒ 退回**一维占位** ✓（只为验编排 ✓，`describe()["latentMode"]` 会如实说 ✓）。
        """
        self._gate()
        torch = self._torch()
        if getattr(self, "_ditFamily", None) == SEEDVR2_FAMILY:
            #: ⭐ **修复族**：噪声形状**跟着条件走** ✓✗（2026-11-01 接线 ✓）。
            #: 出图族的形状由 ``plan`` 决定 ✓；这一族真正的形状事实是**条件的形状** ✓
            #: （条件又来自**真解码**的源视频 ✓ ⇒ 源视频是唯一事实 ✓）。
            #: ``plan`` 到这一步应已被 :meth:`_encode_seedvr2_conditioning` 校正过 ✓；
            #: 这里再核一遍是**幂等的防御** ✓（绕过 ``encode_text`` 直接调 init 的调用方 ✓）——
            #: 不核的症状是「采样第 1 步 33 通道拼不上」✓✗，那种报错看不出是**帧数**错 ✓。
            condition = getattr(self, "_repairCondition", None)
            if condition is None:
                raise TorchBackendUnavailable(
                    "SEEDVR2 没有 17 通道条件 ✗ ⇒ 造不出噪声形状 ✓（这一族的形状由**条件**决定 ✓，"
                    "不是由 plan ✓）。先跑 `encode_text` ✓"
                    "（= :meth:`_encode_seedvr2_conditioning` ✓）再 `init_latents` ✓。", reason="input")
            channels = int(self._config_value("latent_channels", 0) or 0)
            if not channels:
                raise TorchBackendUnavailable(
                    "SEEDVR2 主干的 ``latent_channels`` 读不出来 ✗ ⇒ 不猜通道数 ✓", reason="pending")
            shape = (1, channels, int(condition.shape[2]), int(condition.shape[3]),
                     int(condition.shape[4]))
            #: 像素帧数取**编码阶段实测**的那份 ✓（`_repairSource.frames` ✓）；缺了才按 4n+1 反推 ✓。
            source = getattr(self, "_repairSource", None) or {}
            frames = int(source.get("frames") or ((shape[2] - 1) * 4 + 1))
            stale: dict[str, list[Any]] = {}
            for key, value in (("latent_frames", int(shape[2])), ("frames", frames)):
                old = getattr(plan, key, None)
                if old != value:
                    stale[key] = [old, value]
                    try:
                        setattr(plan, key, value)
                    except (AttributeError, TypeError):
                        pass
            warnings = getattr(plan, "warnings", None)
            if stale and isinstance(warnings, list):
                pairs = "、".join(f"{key} {old!r} → {new!r}" for key, (old, new) in stale.items())
                warnings.append(
                    f"噪声形状按**条件**校正 plan ✓：{pairs} ✓（条件的形状来自真解码 ✓，"
                    "不是 ffprobe 估算 ✓ —— 不校正就是采样第 1 步 33 通道拼不上 ✓✗）")
            generator = torch.Generator(device="cpu").manual_seed(int(request.seed))
            noise = torch.randn(*shape, generator=generator, dtype=torch.float32)
            return noise.to(self._device)
        shape = self.latent_shape(plan)
        generator = torch.Generator(device="cpu").manual_seed(int(request.seed))
        if shape is None:
            noise = torch.randn(self.width, generator=generator, dtype=torch.float32)
            return noise.to(self._device)
        noise = torch.randn(*shape, generator=generator, dtype=torch.float32)
        return self._mix_init_image(noise.to(self._device), plan, request)

    def init_image_latents(self, plan: Any, request: Any) -> Any:
        """``initImage`` ⇒ **归一化**潜变量 ✓（图生图的**起点** ✓，不是条件 ✗）。

        形状跟着 :meth:`latent_shape` 走 ✓ —— 它可能是 **4 维**（Z-Image / Qwen-Image 2.1 ✓
        ``IMAGE_ONLY_FAMILIES`` ✓ 没有时间维 ✗）也可能是 **5 维** ✓，这里**照它的维数**交 ✓✗
        （一律造 5 维 ⇒ 图族 `patchify` 上多一维 ⇒ 报「expected 4D」✓，指不到"起点给错了"✓）。

        ⚠️ 编码**只能**走 :meth:`_encode_pixels` ✗✗：各家 ``encode`` 有三种返回口径 ✓
        （H3 自带归一化 ✓、Flux 返二元组 ✓、Wan 返未归一化 ✓）—— 裸调 ``self._vae.encode``
        在 Flux 族上拿到的是 ``(mean, logvar)`` 的**第一个** ✓ 形状对、数值是半份统计量 ✓✗。
        """
        self._gate()
        path = str(getattr(request, "init_image", "") or "").strip()
        if not path:
            raise TorchBackendUnavailable(
                "denoise<1 要一张**起点图** ✗ 但 ``initImage`` 是空的 ✓ —— "
                "起点不是「可选的美化项」 ✓：不给图就只剩纯噪声起跑 ✓✗（图与参考无关、且不报错 ✓）。",
                reason="input")
        target = self.latent_shape(plan)
        if target is None:
            raise TorchBackendUnavailable(
                "读不出**真形状**（``vae_scale`` / 通道数缺 ✓）⇒ 没法把参考图编码成起点 ✗。"
                "占位路径下图生图无从谈起 ✓（先 `ensure_loaded` 配好 vae_scale ✓）", reason="pending")
        torch = self._torch()
        image, info = media_mod.load_image_tensor(path, width=int(plan.width),
                                                  height=int(plan.height))
        with torch.no_grad():
            encoded = self._encode_pixels(image.to(device=self._device, dtype=torch.float32))
        # ⚠️ **只比 C/h/w、不比 T** ✓（图天然单帧 ⇒ 编码出 T=1 ✓；比较下标逐位点名 ✗ 不用滑窗 ✓）。
        if encoded.ndim == 5:
            got = tuple(encoded.shape[i] for i in (1, 3, 4))
            ref = encoded[:, :, :1]
        else:
            got = tuple(encoded.shape[1:])
            ref = encoded
        want = (tuple(target[i] for i in (1, 3, 4)) if len(target) == 5
                else tuple(target[i] for i in (1, 2, 3)))
        if got != want:
            raise TorchBackendUnavailable(
                f"参考图编码后「通道/高/宽」{got} 与潜变量要的 {want} 对不上 ✗ —— "
                f"VAE 与 DiT 不是一套 ✓（``vae_scale`` / 通道数必须同源 ✓；"
                f"整条原形状 {tuple(encoded.shape)} vs {tuple(target)} ✓）", reason="pending")
        if len(target) == 4:
            ref = ref.reshape(1, ref.shape[1], ref.shape[3], ref.shape[4])
        self._initImageNote = {"imageInfo": info, "shape": tuple(ref.shape)}
        return ref.to(device=self._device, dtype=torch.float32)

    def _mix_init_image(self, noise: Any, plan: Any, request: Any) -> Any:
        """按 ``σ0`` 把参考图 latent 混进噪声 ✓ = :meth:`init_latents` 的**后半步** ✓✗。

        口径逐字对齐参考实现（``comfy/sample.py::noise_scaling`` ✓）::

            x0 = σ0 · noise + (1 − σ0) · image_latent

        ⚠️ 这一半**不能省** ✗✗：``denoise<1`` 只把 σ 梯子截短 ✓ 而起点仍是纯噪声 ⇒
        跑完是「一张与参考图无关、对比度偏低的图」✓ —— **形状全对、一条报错都没有** ✓✗
        （这正是 `plan` 层强制 ``denoise<1 ⇒ initImage`` 的理由 ✓）。
        """
        if str(getattr(plan, "init_from", "noise") or "noise") != "image":
            return noise
        sigmas = getattr(plan, "sigmas", None) or [1.0]
        sigma0 = float(sigmas[0])
        torch = self._torch()
        ref = self.init_image_latents(plan, request).to(device=noise.device, dtype=noise.dtype)
        if tuple(ref.shape) != tuple(noise.shape):
            raise TorchBackendUnavailable(
                f"起点图 latent {tuple(ref.shape)} 与噪声 {tuple(noise.shape)} 形状不一致 ✗ —— "
                "``initImage`` 的像素尺寸必须是 ``plan.width/height`` ✓（不是任意尺寸 ✓）",
                reason="input")
        with torch.no_grad():
            return noise * sigma0 + ref * (1.0 - sigma0)

    def noise_like(self, sample: Any, request: Any, *, nonce: int = 0) -> Any:
        """随机采样器（``sa_solver`` / ``euler_ancestral``）每步回注的噪声 ✓。

        ⚠️ 两件都**不是**随手抄 :meth:`init_latents` ✓：

        * **不乘 σ_max** ✗✗：:meth:`init_latents` 是 ``n·σ_max`` ✓，而回注量由采样器自己乘
          ``σ·renoise`` ✓ ⇒ 这里再乘一次就是**平方** ✓✗（症状：随机档画面比确定性档明显更脏、
          且**步数越多越脏** ✓ —— 不报错 ✓）；
        * **``nonce`` 必须进种子** ✓✗：同一条噪声反复注入 ⇒ 画面出现**结构性条纹** ✓✗
          （同样不报错 ✓，只在真机上看得见 ✓）。
        """
        self._gate()
        torch = self._torch()
        generator = torch.Generator(device="cpu").manual_seed(
            (int(getattr(request, "seed", 0) or 0) * 2654435761 + int(nonce) * 7919) & 0x7FFFFFFF)
        noise = torch.randn(tuple(sample.shape), generator=generator, dtype=torch.float32)
        return noise.to(device=sample.device, dtype=sample.dtype)

    def _config_value(self, name: str, default: Any = None) -> Any:
        """从 ``self._config`` 取一个字段 ✓ —— **两种形态都认** ✓（H3/DiT 是 dataclass ✓、Z-Image 是 dict ✓）。

        ⚠️ 2026-10-31 踩到 ✓：Z-Image 的 config 是 **dict** ✓（``_build_zimage_module`` 返回 ✓），
        而 :meth:`latent_shape` / ``_check_attached_vaes`` 原来都写 ``getattr(config, "vae_scale", 0)`` ✗
        ⇒ 在 dict 上**永远取不到** ✓ ⇒ 潜变量静默回落**一维占位** ✓✗（而 ``describe()`` 还写着
        ``latentMode="real-shape"`` ✓✗ —— 正好是"看不出来"那一类错 ✓）。凡读配置一律走这个口 ✓。
        """
        config = self._config
        if isinstance(config, dict):
            return config.get(name, default)
        if config is None:
            return default
        return getattr(config, name, default)

    def latent_shape(self, plan: Any) -> tuple[int, ...] | None:
        """潜变量形状 ✓（``None`` = 信息不够 ⇒ **用占位** ✓ 并如实标注 ✓）。

        ⚠️ 需要 ``vae_scale``（像素↔潜空间边长比 ✓）与潜帧数 ✓；**任一缺就返回 None** ✗
        —— 猜一个"看起来合理"的 8 只会让真机上错得莫名其妙 ✗。

        ⭐ **Z-Image / Qwen-Image 2.1 是图，不是片** ✓（2026-10-31 补 ✓）：它们的潜变量是
        **4 维** ``(1, C, H/scale, W/scale)`` ✓，**没有时间维** ✗✗（``LuminaDiT.forward`` /
        ``Qwen21DiT.forward`` 都只吃 4 维 ✓）。若沿用 5 维那条（``latent_frames`` = 1 ✓）⇒
        ``patchify`` / ``flatten(2)`` 上会多出一维 ⇒ 报的是「expected 4D」✓✗，指不到"这是图模型"✓。
        判据读 :data:`IMAGE_ONLY_FAMILIES` ✓（不是这里再抄一遍族名 ✗）。

        ⭐ **音频族是 3 维** ✓（2026-11-02 接 Music3 ✓）：``(1, 128, T)`` ✓ —— ``T`` 是
        **潜帧**（44.1 kHz ÷ 512 = 86.13 Hz ✓），既不是像素边也不是视频帧数 ✓✗。
        """
        #: ⚠️ 守门读 **`_config`** 不读 `_model` ✗✗（2026-10-31 实测踩到 ✓）：形状只需要
        #: 「权重反推出的结构事实」✓（几 KB ✓），**不需要 13 GiB 的模块在场** ✓。
        #: 分时驻留时 `init` 阶段跑在 **encode 之后** ✓ —— 那时 TE 在场、DiT 已被退掉 ✓，
        #: 读 `_model is None` ⇒ 静默退回**一维占位** ✓✗ ⇒ 采样第 1 步炸
        #: ``latents 要 (B,C,H,W)，实际 (512,)`` ✓✗（报错指向 DiT ✓，根因在换装 ✓）。
        if self._config is None:
            return None
        #: ⚠️ **音频族必须排在 `vae_scale` 之前** ✗✗（2026-11-06 接 Music3 装配线时踩到 ✓）：
        #: Music3 的结构里**没有 `vae_scale`**（它的潜帧来自条件帧 ✓，不是"像素边 ÷ 缩放" ✓）
        #: ⇒ 先判 `vae_scale` 会让它整条掉进 `None` ✓✗，而 `None` 的语义是「信息不够 ⇒ 一维占位」✓
        #: ⇒ 采样第 1 步炸 ``latents 要 (B,C,L)，实际 (512,)`` ✓（报错指向 DiT ✓，根因在**这里** ✓）。
        #: 所以顺序改成：先取通道 ⇒ 3 维族直接回形状 ⇒ 剩下的才要 `vae_scale` ✓。
        #: ⚠️ **潜通道 ≠ 输入通道** ✗✗（2026-11-01 接 SEEDVR2 时踩到 ✓）：多数族两者同值 ✓
        #: （``in_channels`` 就是 ``x_embedder`` 的列数 ÷ patch² ✓），但**带条件的族**把
        #: 「潜变量 + 条件」在入口拼起来 ✓ —— SEEDVR2 是 ``conditioning`` 17 通道（退化视频 16 ＋
        #: loss mask 1 ✓）＋ 噪声 16 ⇒ ``vid_in`` 吃 **33** ✓✗。照 ``in_channels`` 造潜变量
        #: 会造出 ``(1, 33, T, H, W)`` ✓ ⇒ 形状自洽 ✓、采样第 1 步在 ``vid_in`` 上炸 ✓
        #: （报「expected 16 got 33」✓，指不到"读错了那一格"✓）。⇒ 有 ``latent_channels`` 就用它 ✓。
        channels = int(self._config_value("latent_channels", 0)
                       or self._config_value("in_channels", 0) or 0)
        #: ⭐ **音频族（3 维）在这里分出去** ✓✗（2026-11-02 接 Music3 ✓）：它没有空间维 ✓，
        #: 潜帧来自**条件帧**（``latent_length(condition_frames)`` ✓ 25 Hz → 86.13 Hz ✓）
        #: 而不是 ``plan.latent_frames`` 的"像素帧 ÷ 4k+1" ✓。若照视频那条算 ⇒
        #: ``(1, 128, T, H, W)`` 五维 ✓ 而 ``Music3DiT.forward`` 第一句就判 ``dim() == 3`` ✗
        #: （报错倒是明确 ✓，但形状本该在这里就分开 ✓）。
        if dit_build.contract_of(getattr(self, "_ditFamily", None)).latent_dims == 3:
            latent_frames = getattr(plan, "latent_frames", None)
            if not latent_frames or not channels:
                return None
            return (1, channels, int(latent_frames))
        scale = int(self._config_value("vae_scale", 0) or 0)
        if not scale:
            return None
        height = max(1, int(plan.height) // scale)
        width = max(1, int(plan.width) // scale)
        if dit_build.contract_of(getattr(self, "_ditFamily", None)).latent_dims == 4:
            if not channels:
                return None
            return (1, channels, height, width)
        latent_frames = getattr(plan, "latent_frames", None)
        if not latent_frames or not channels:
            return None
        return (1, channels, int(latent_frames), height, width)

    # ── H3 双流（视频 + 音频 ✓ 2026-09-20 接进管线 ✓）──────────────────────
    def init_dual_latents(self, plan: Any, request: Any) -> dict[str, Any]:
        """H3 **双流**初始噪声 ✓ ⇒ ``{"video": [C,T,H,W], "audio": [C,ch,T_a]}`` ✓（**无 batch 维** ✓）。

        * 视频潜尺寸 = ``plan`` 像素尺寸 ÷ **vae_scale(16)** ✓（H3 事实 ✓）—— **不整除就报错** ✗
          （吸附网格是**调用方**的事 ✓，这里不悄悄取整 ✗）；
          ⚠️ 与 ``vae_scale`` **无关**的另一处对齐：主干 patch 是 ``(1, 2, 2)`` ✓ ⇒ 潜帧的 H/W
          还要能被 **2** 整除 ✓ —— 但**不需要调用方管** ✓：:func:`h3_form.pad_to_patch_size`
          会 circular 补到整数倍、算完再裁回原点 ✓（与参考 `model.py` 606 / 781 行同义 ✓）
          ⇒ **别在这里补 32 的倍数** ✗✗（720×1280 ⇒ 潜帧 45×80 ✓ 45 是奇数 ✓ 照样跑 ✓）；
          ⚠️ 也**别把这条当成"可以给任意尺寸"** ✗：``vae_scale`` 那条仍然是硬的 ✓。
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
        video, audio = self._dual_noise(int(latent_frames), int(audio_frames), height // scale,
                                        width // scale, int(request.seed))
        keyframes = self._dual_keyframes(plan, request, scale)
        payload: dict[str, Any] = {"video": video, "audio": audio,
                                   "audioLatentFrames": int(audio_frames), "vaeScale": scale,
                                   "keyframes": keyframes,
                                   "references": self._dual_references(plan, request)}
        # ⭐⭐ **续拍**（2026-09-26 ✓）：上一段的**尾部潜变量**当作本段**头部**的冻结锚 ✓。
        #    上游 ``h3_extend`` 的 context_pin 就是这个口径 ✓（钉的是潜变量 ✓ 不是帧 ✓ ——
        #    颜色与速度都逐位接得上 ✓）。⚠️ 与「拼接」的单帧 PNG 锚（``first_frame`` ✓）**是两回事** ✗：
        #    那个要过 VAE 一趟 ✓ 只能管住一帧的颜色 ✓；这个管住整段尾巴 ✓。
        carry = getattr(request, "carry_latents", None)
        if carry is not None:
            pin_video, pin_audio, (pin_v, pin_a) = self._dual_carry_pin(
                carry, video, audio,
                temporal_compression=getattr(request, "temporal_compression", None))
            sigma_v0, sigma_a0 = self._dual_sigma0(plan)
            slots_v, slots_a = int(video.shape[1]), int(audio.shape[2])
            mask_v = torch.ones((1, slots_v, 1, 1), device=self._device, dtype=torch.float32)
            mask_a = torch.ones((1, 1, slots_a), device=self._device, dtype=torch.float32)
            mask_v[:, :pin_v] = 0.0        # 头部**冻结** ✓（0 = 冻结 ✓ 见 sample_dual_stream ✓）
            mask_a[:, :, :pin_a] = 0.0
            payload["video"] = pin_video + sigma_v0 * video
            payload["audio"] = pin_audio + sigma_a0 * audio
            payload["pin"] = (pin_video, pin_audio)
            payload["pinMask"] = (mask_v, mask_a)
            payload["continuation"] = {"pinFrames": int(carry.get("frames") or 0),
                                       "videoSlots": pin_v, "audioSlots": pin_a,
                                       "sigmaVideo0": sigma_v0, "sigmaAudio0": sigma_a0,
                                       "source": "carry_latents"}
        return payload

    def _dual_sigma0(self, plan: Any) -> tuple[float, float]:
        """续拍要用的**首 σ** ✓：视频取 ``plan.sigmas[0]`` ✓、音频走 ``time_shift_sigma`` ✓。

        ⚠️ **必须能读到 ``plan.sigmas``** ✗：续拍的 ``x₀ = 尾巴 + σ₀·noise`` ✓ ⇒ 没有 σ₀ 就
        钉不出那条含噪轨迹 ✓（凭空写一个 1.0 ✗ 会让钉进去的尾巴与噪声水平**对不上** ✓✗）。
        口径与 :func:`h3_form.sample_dual_stream` 的入口校验**同一份** ✓（≥2 个、末位 0 ✓）。
        """
        from app.services.engine import h3_form, schedules  # noqa: PLC0415
        steps = [float(value) for value in (getattr(plan, "sigmas", None) or [])]
        if len(steps) < 2 or steps[-1] != 0.0:
            raise TorchBackendUnavailable(
                f"续拍要钉尾部潜变量 ⇒ 必须能从 ``plan.sigmas`` 读到 σ 日程 ✗（读到 {len(steps)} 个、"
                f"末位 {steps[-1] if steps else '无'} ✓）—— 编译层没算 sigmas ✓", reason="pending")
        if steps[0] <= 0.0:
            raise TorchBackendUnavailable(
                f"续拍的首 σ 必须 > 0 ✗（收到 {steps[0]} ✓ —— 钉住的尾巴要按 ``x₀ = 尾巴 + σ₀·noise`` "
                "铺进采样轨迹 ✓）", reason="pending")
        shifts = h3_form.H3_SIGMA_SHIFTS
        return steps[0], schedules.time_shift_sigma(steps[0], float(shifts["sigma_shift_video"]),
                                                    float(shifts["sigma_shift_audio"]))

    def _dual_carry_pin(self, carry: Any, video: Any, audio: Any, *,
                        temporal_compression: int | None) -> tuple[Any, Any, tuple[int, int]]:
        """``carry_latents`` → **补齐到整条长度**的 pin ✓（头部放尾巴 ✓ 其余位置是**没用的填充** ✓）。

        ``carry`` 要有 ``video`` / ``audio``（**上一段的尾部潜变量** ✓ 与 ``[C,T,h,w]`` /
        ``[C,ch,Ta]`` 同族 ✓）与 ``frames``（那段尾巴覆盖多少**像素帧** ✓ —— 本仓**不猜** ✗：
        没有它就说不清"钉了多少" ✓）。

        ⚠️ ``temporal_compression`` **必须**给 ✓：``frames`` 是像素帧 ✓、尾巴是潜槽 ✓ ⇒ 两者之间
        只有 :func:`geometry.latent_frames` 这一条换算 ✓，而压缩比**不在** ``plan`` 上（它是请求的
        字段 ✓）⇒ 由调用方显式递进来 ✓，缺了就报错 ✓（**不**拿「读不到」当「没问题」✗）。

        ⚠️ 未钉到的位置填 **0** ✓：那儿的 ``pin_mask`` 是 1（要重画 ✓）⇒ pin 的取值**用不上** ✓
        （``noise_dir = (x₀ − pin)/σ₀`` 只在 mask=0 处生效 ✓）—— 填 0 而不是复制噪声 ✗ 是为了
        **一眼看得出**那是填充 ✓（复制噪声反而像"有语义" ✓✗）。
        """
        from app.services.engine import geometry as geometry_mod  # noqa: PLC0415
        self._require_own_device("续拍尾巴", carry=carry)
        torch = self._torch()
        if not isinstance(carry, dict) or "video" not in carry or "audio" not in carry:
            raise TorchBackendUnavailable(
                f"``carry_latents`` 要给 ``video`` / ``audio`` 两路（**上一段的尾部潜变量** ✓）✗"
                f"（收到 {sorted(carry.keys()) if isinstance(carry, dict) else type(carry).__name__} ✓）"
                "—— 别拿解码后的帧来钉 ✓（那是「拼接」的单帧锚 ✗ 两回事 ✓）", reason="pending")
        if not carry.get("frames"):
            raise TorchBackendUnavailable(
                "``carry_latents`` 缺 ``frames``（那段尾巴覆盖多少**像素帧** ✓）✗ —— "
                "本仓**不猜**钉了多少 ✓（要按秒对齐音频 ✓ 见 `h3_edit.continuation_pin` ✓）",
                reason="pending")
        tail_v, tail_a = carry["video"], carry["audio"]
        shape_v = tuple(int(value) for value in getattr(tail_v, "shape", ()) or ())
        shape_a = tuple(int(value) for value in getattr(tail_a, "shape", ()) or ())
        want_v = tuple(int(value) for value in video.shape)
        want_a = tuple(int(value) for value in audio.shape)
        if (len(shape_v) != 4 or len(shape_a) != 3 or shape_v[0] != want_v[0]
                or shape_v[2:] != want_v[2:] or shape_a[0] != want_a[0]
                or shape_a[1] != want_a[1]):
            raise TorchBackendUnavailable(
                f"续拍尾巴与本次窗口**不是同一套潜空间** ✗：尾巴 {shape_v} / {shape_a} ✓，"
                f"本次 {want_v} / {want_a} ✓ —— 通道数与空间尺寸必须**逐项相等** ✓（换了模型/尺寸就钉不上 ✓）",
                reason="pending")
        if shape_v[1] > want_v[1] or shape_a[2] > want_a[2]:
            raise TorchBackendUnavailable(
                f"续拍尾巴比本次窗口**还长** ✗：尾巴 {shape_v[1]} / {shape_a[2]} 槽 ✓，"
                f"窗口 {want_v[1]} / {want_a[2]} 槽 ✓（钉不下 ✓ —— 缩短钉子或加长窗口 ✓）",
                reason="pending")
        # ⭐ 第二份事实：``frames`` 必须与**视频槽数**对得上 ✓（``latent_frames`` 是**唯一**换算 ✓）
        #    ⚠️ 2026-09-26 纠：原来读 ``getattr(plan, "temporal_compression", None)`` —— 而
        #    :class:`~app.services.engine.pipeline.GenerationPlan` 上**没有**这个字段 ✗（压缩比是
        #    :class:`GenerationRequest` 的 ✓，plan 只存换算结果 ``latent_frames`` ✓）⇒ 这条守卫在
        #    真管线里**永远不触发** ✓✗：拿「读不到」当「没问题」就是**静默兜底** ✓。压缩比**只能**
        #    从请求来 ✓ ⇒ 显式传入 ✓；缺了 ⇒ 响亮报错 ✓（口径同 ``latent_frames`` 未知那一条 ✓、
        #    也同 :meth:`_dual_sigma0` 对 ``plan.sigmas`` 的态度 ✓ —— 本仓**不猜**压缩比 ✓）。
        #    ⚠️ 另一份佐证：原来那行连 :func:`geometry.latent_frames` 的**调用签名**都是错的 ✗
        #    （压缩比是**关键字**参数 ✓）⇒ 它要是真跑过一次，早就 ``TypeError`` 了 ✓。
        #    ℹ️ 走 :meth:`init_dual_latents` 这条真路时，上面 ``plan.latent_frames`` 那道守卫已经先
        #    拦过一次 ✓（两者同源：都来自请求的 ``temporalCompression`` ✓）⇒ 这里再要一次压缩比
        #    是**给直接调用兜底** ✓ 并把不变量写进签名 ✓（自检 ㊶′ 专钉它 ✓）。
        if not temporal_compression:
            raise TorchBackendUnavailable(
                "钉续拍尾巴要按**压缩比**把 ``frames`` 折成视频潜槽 ✓ 而这次请求没给 "
                "``temporalCompression`` ✗ ⇒ 「``frames`` 与槽数」这条**同一件事写两处**的就"
                "**没得对** ✓（本仓不猜压缩比 ✓ —— 见 `geometry.latent_frames` ✓）", reason="pending")
        expected = geometry_mod.latent_frames(int(carry["frames"]),
                                               temporal_compression=int(temporal_compression))
        if int(expected) != shape_v[1]:
            raise TorchBackendUnavailable(
                f"``carry_latents['frames']={carry['frames']}`` 按压缩比 {temporal_compression} 算是 "
                f"{expected} 个视频潜槽 ✓，而尾巴实际有 {shape_v[1]} 个 ✗"
                "（同一件事写两处 ✓ 不等 ⇒ 必有一处错 ✓）", reason="pending")
        pin_video = torch.zeros(want_v, device=self._device, dtype=tail_v.dtype)
        pin_audio = torch.zeros(want_a, device=self._device, dtype=tail_a.dtype)
        pin_video[:, :shape_v[1]] = tail_v
        pin_audio[:, :, :shape_a[2]] = tail_a
        return pin_video, pin_audio, (shape_v[1], shape_a[2])

    def _dual_noise(self, latent_frames: int, audio_frames: int, latent_height: int,
                    latent_width: int, seed: int) -> tuple[Any, Any]:
        """双流的**初始噪声** ✓（``[C,T,H,W]`` 与 ``[C,ch,T_a]`` ✓ 无 batch 维 ✓）。

        ⚠️ 抽成一处 ✓ **只为让重拍能复用同一条噪声路径** ✓（2026-09-26 ✓）—— 重拍要的是
        ``x₀ = 原始潜变量 + σ₀·noise`` ✓ ⇒ 它必须拿到**与首采同一条**噪声 ✓（另起一处生成 ✗
        就会悄悄换掉重拍窗口里的随机性 ✓✗）。两条流共用一个 generator ✓ ⇒ 同 seed 逐位可复现 ✓。
        """
        torch = self._torch()
        video_dim = int((self._config or {}).get("latents_dim") or 0)
        audio_dim = int((self._config or {}).get("audio_latents_dim") or 0)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        video = torch.randn(1, video_dim, int(latent_frames), int(latent_height),
                            int(latent_width), generator=generator, dtype=torch.float32)[0]
        audio = torch.randn(1, audio_dim, geometry_mod.AUDIO_LATENT_CHANNELS, int(audio_frames),
                            generator=generator, dtype=torch.float32)[0]
        return video.to(self._device), audio.to(self._device)

    def _dual_keyframes(self, plan: Any, request: Any, scale: int) -> list[dict[str, Any]]:
        """首帧 / 尾帧 → **关键帧条件块** ✓（H3 的 ``cond`` 段 ✓）—— 两个都没给 ⇒ 空列表 ✓。

        ⚠️ 事实（核自参考 ✓）：``cond`` 行用的是**目标的空间网格** ✓ ⇒ 关键帧**必须**编码成目标潜尺寸 ✓
        （图片按 ``plan`` 的宽高缩放 ✓ 再编码 ✓）；``t`` 起点 = ``cursor + FRAME_RESCALE × 帧号`` ✓。
        首帧 = **第 0 帧** ✓、尾帧 = **第 ``frames − 1`` 帧** ✓ —— 两者共用同一套 ``cond`` 机制 ✓，
        只是下标不同 ✓（参考实现的 fl2v 也是这么落的 ✓）。

        ⚠️ 尾帧要 ``frames ≥ 2`` ✓：只有 1 帧时首尾是同一个位置 ✓✗ ⇒ 当场报错 ✗，不悄悄退化成首帧 ✓。

        ⚠️ 编码用的是本仓**参考（未训练）视频 VAE** ✓ ⇒ 这条条件**数值上没有语义** ✓（验的是管道 ✓）；
        真权重到位后调用点不变 ✓。
        """
        first = getattr(request, "first_frame", None)
        last = getattr(request, "last_frame", None)
        if not first and not last:
            return []
        # 下标规划 → `keyframes.plan_keyframes` ✓（纯函数 ✓ 无 GPU 也能验 ✓）
        try:
            wanted = keyframes_mod.plan_keyframes(plan.frames, first, last)
        except keyframes_mod.KeyframePlanError as error:
            raise TorchBackendUnavailable(str(error), reason="pending") from error
        if self._vae is None:
            raise TorchBackendUnavailable(
                "双流首/尾帧条件需要**视频 VAE**（把关键帧编码成潜变量 ✓）✗ ⇒ 先 attach_vae() ✓",
                reason="pending")
        if int(plan.width) % scale or int(plan.height) % scale:
            raise TorchBackendUnavailable(
                f"关键帧尺寸 {plan.width}×{plan.height} 不能被 vae_scale={scale} 整除 ✗",
                reason="pending")
        torch = self._torch()
        blocks: list[dict[str, Any]] = []
        for index, image in wanted:
            frames, info = media_mod.load_image_tensor(
                image, width=int(plan.width), height=int(plan.height))
            with torch.no_grad():
                #: ⚠️ 走统一编码口子 ✓✗（这三条虽是 H3 专用 ✓，但裸调 ``encode`` 会把
                #: 「H3 自带归一化 ✓ / Flux 返二元组 ✗ / Wan 没 encode ✗」三种口径混进同一行 ✓）。
                latent = self._encode_pixels(frames.to(self._device))[0]
            blocks.append({"resolved_frame_index": int(index), "latent": latent,
                           "audio_latent": None,
                           "imagePath": str(info.get("path") or image)})
        return blocks

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
                    latent = self._encode_pixels(frames.to(self._device))[0]      # [C, 1, h, w] ✓
                blocks.append({"kind": "image", "latent": latent,
                               "latent_h": int(latent.shape[2]), "latent_w": int(latent.shape[3]),
                               "imagePath": str(info.get("path") or image)})
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
                latent = self._encode_pixels(frames.to(self._device))[0]   # [C, t, h, w] ✓
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
        #: ⚠️⭐ **独立音频排在全部视频之后** ✗✗（2026-11-01 改序 ✓）：照参考
        #: ``nodes_minimax_h3.py::MiniMaxH3ReferenceToVideo.execute`` 的 ``ref_blocks`` 顺序 ✓
        #: ＝ ``[image…] → [video / video_audio…] → [audio…]`` ✓，与文本侧 ``<Audio j>`` 的
        #: **编号同源** ✓（每路视频的声轨占前几位 ✓、独立音频从 ``S+1`` 起 ✓）。
        #: 反过来（独立音频在前 ✓✗）⇒ 文本说 ``<Audio 1>`` 是第 1 路视频的声轨 ✓、
        #: 而块序里第 1 个音频块却是那条独立音频 ✓ —— 号与素材**整体错位** ✓、形状全对 ✓✗。
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
        return blocks

    def _dual_extra_rows(self, latents: Any, condition: Any, *, seed: int = 0,
                         visual_cond_aug: Any = None,
                         audio_cond_aug: Any = None) -> tuple[Any, Any]:
        """条件/参考块 → **行** ✓（⚠️ 行序**必须**与 `h3_form.packed_layout` 的段序一致 ✗✗
        —— 同数不同序的话总行数校验**拦不住** ✓✗，只会"画面不对" ✓）。

        段序（事实 ✓）：``[cond* / cond_audio*] → [ref_img* / ref_audio*]`` ✓ ⇒ 这里就照它拼 ✓：
        **先关键帧、后参考块** ✓。⚠️ 两条流（视频行 / 音频行）是**独立**的两条 ✓ ⇒ 各自内部顺序对就行 ✓；
        块**内部**顺序由 `patchify_video` / `pack_audio` 保证 ✓（与 layout 同一套外壳 ✓）。

        ⭐ ``seed`` / ``*_cond_aug`` 是给 :func:`h3_form.assemble_blocks` 的**条件行噪声增广** ✓
        （2026-11-01 补齐 ✓）：``None`` ⇒ 用参考的默认档 ✓（0.999 / 1.0 ✓）——**不给"关掉"的默认值** ✗✗
        （默认关 ⇒ 「旋钮忘了接线」永远不会被发现 ✓✗；要关必须显式写 1.0 ✓）。
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
        return h3_form.assemble_blocks(
            layout, video_sources, audio_sources, self._model.patch_size,
            seed=int(seed),
            visual_cond_aug=(h3_form.VISUAL_COND_TIMESTEP if visual_cond_aug is None
                             else float(visual_cond_aug)),
            audio_cond_aug=(h3_form.AUDIO_COND_TIMESTEP if audio_cond_aug is None
                            else float(audio_cond_aug)))

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
        # ⚠️ ``seed`` 必须**递给**增广那一步 ✓（条件行的噪声由 seed 决定 ✓ —— 不递的话条件行永远
        #    用 seed=0 的噪声 ✓✗：同一条提示词换 seed，条件行那一丁点噪声**不变** ✓ 不报错 ✓）。
        extra_video, extra_audio = self._dual_extra_rows(
            latents, condition,
            seed=int(getattr(request, "seed", 0) or 0),
            visual_cond_aug=getattr(request, "visual_cond_noise_aug", None),
            audio_cond_aug=getattr(request, "audio_cond_noise_aug", None))
        shifts = h3_form.H3_SIGMA_SHIFTS
        # ⭐ **冻结区**（续拍 / 重拍 ✓ 2026-09-26 ✓）：构造方是 :meth:`init_dual_latents`（续拍 ✓）
        #    或 :meth:`retake_latents`（重拍 ✓）—— 两者都把 ``pin`` / ``pinMask`` 放进同一个 dict ✓
        #    ⇒ 采样这条路**只有一份** ✓（本方法不自己造 pin ✗ 也不猜 ✗）。
        pin = latents.get("pin")
        pin_mask = latents.get("pinMask")
        if (pin is None) != (pin_mask is None):
            raise TorchBackendUnavailable(
                f"`latents` 里 ``pin`` 与 ``pinMask`` 必须**成对**出现 ✗（收到 pin="
                f"{'有' if pin is not None else '无'} / pinMask="
                f"{'有' if pin_mask is not None else '无'} ✓）—— 构造方见 :meth:`retake_latents` ✓",
                reason="pending")
        return h3_form.sample_dual_stream(
            self._model, latents["video"], latents["audio"], condition, sigmas,
            shift_v=float(shifts["sigma_shift_video"]), shift_a=float(shifts["sigma_shift_audio"]),
            callback=on_step, extra_video_rows=extra_video, extra_audio_rows=extra_audio,
            keyframes=keyframes or None, refs=refs or None,
            pin=pin, pin_mask=pin_mask,
            # ⚠️⭐ **文本标签**（`h3_encode.encode_condition` 产的 ``H3Condition.tags`` ✓）——
            #    走 ``request`` 而不是 ``condition`` 是因为它是**逐 token 的整数行号档** ✓ 不是特征 ✗
            #    （塞进 tensor 里就只能靠"多一通道"这种约定 ✓✗，一改文本长度就对不上 ✓）。
            #    ⇒ 已是**请求契约**里的正式字段 ✓：`GenerationRequest.text_token_tags` ✓，
            #    由 `pipeline` 在 encode 之后从 `condition["tags"]` 填进去 ✓（见那儿的两行注释 ✓）。
            text_token_tags=getattr(request, "text_token_tags", None),
            # ⚠️ 采样器名走**请求字段** ✓（白名单在 `h3_form.DUAL_SAMPLERS` ✓ 唯一真源 ✓）——
            #    不在则按 `euler` ✓（= 2026-11-01 之前的行为 ✓，逐位不变 ✓）
            sampler=str(getattr(request, "sampler", "") or "euler"))

    def retake_latents(self, latents: Any, plan: Any, request: Any, *, condition: Any,
                       sigmas: Any, retake: Any, on_step: Any = None) -> dict[str, Any]:
        """**重拍**（retake ✓ 2026-09-26 移植 ✓）：把**成片**的双流潜变量按窗口重画一遍 ✓，
        窗外**逐位保留** ✓（自检钉住 ✓）。

        事实来源：``h3_retake.py``（MIT ✓）—— 上游是
        ``VAEEncode(成片)`` → ``noise_mask`` 划窗 → ``SamplerCustomAdvanced`` ✓，窗外那片由上游
        ``KSamplerX0Inpaint`` **逐位写回原始潜变量** ✓。本仓对应 :func:`h3_form.sample_dual_stream`
        的 ``pin`` / ``pin_mask`` ✓——⚠️ **不是** ``denoise_mask`` ✗✗（那个冻结在"第 0 步的含噪值"上 ✓，
        见那儿的 docstring ✓）。

        ⚠️⚠️ 四条口径（都是「看着像重拍、其实坏了」的形状 ✓✗）：
        1. **``latents`` 是原始潜变量** ✗ 不是噪声 ✗（形状与首采同一套 ``[C,T,H,W]`` / ``[C,ch,T_a]`` ✓）
           —— 给错就是"拿噪声重拍噪声" ✓✗；
        2. **槽位数三方一致** ✓✗：``retake.video_slots`` / ``retake.audio_slots``（**计划层按秒算的** ✓）
           必须等于张量的**实际时间维** ✓ ⇒ 不等就**拒** ✗（一路按秒 ✓ 一路按张量 ✗ ⇒ 一定有一处错 ✓；
           悄悄按张量改窗口 = 静默换产物 ✗）；
        3. **首 σ 必须 > 0** ✗（冻结区的噪声方向靠 ``x₀ = pin + σ₀·noise`` 反推 ✓）；
        4. **锚只有 pin** ✓：首帧/参考块的处理见下面注释 ✓（随机性由 ``request.seed`` 交代 ✓
           —— 本方法**不自己摇随机数** ✗）。
        """
        self._gate()
        self._require_own_device("retake_latents", latents=latents, condition=condition)
        torch = self._torch()
        from app.services.engine import h3_edit, h3_form, schedules  # noqa: PLC0415
        if not isinstance(latents, dict) or "video" not in latents or "audio" not in latents:
            raise TorchBackendUnavailable(
                "重拍要**成片的双流潜变量**（含 video/audio ✓）✗ —— 别拿噪声/单流来重拍 ✓✗",
                reason="pending")
        if not isinstance(retake, h3_edit.RetakePlan):
            raise TorchBackendUnavailable(
                "重拍必须给计划层的 :class:`h3_edit.RetakePlan` ✗（窗口与槽位是**算出来的** ✓ "
                "⇒ 后端**不自己猜窗口** ✗：猜错就是「重拍了一整段别的」✓✗）", reason="pending")
        steps = [float(value) for value in sigmas]
        if len(steps) < 2 or steps[-1] != 0.0:
            raise TorchBackendUnavailable(
                f"重拍的 σ 日程必须 ≥2 个且**末位为 0** ✗（收到 {len(steps)} 个、末位 "
                f"{steps[-1] if steps else '无'} ✓）—— 与 :func:`h3_form.sample_dual_stream` 同口径 ✓",
                reason="pending")
        video, audio = latents["video"], latents["audio"]
        shape_v = tuple(int(value) for value in getattr(video, "shape", ()) or ())
        shape_a = tuple(int(value) for value in getattr(audio, "shape", ()) or ())
        if len(shape_v) != 4 or len(shape_a) != 3:
            raise TorchBackendUnavailable(
                f"重拍的潜变量形状要 ``[C,T,H,W]`` / ``[C,ch,T_a]`` ✗（收到 {shape_v} / {shape_a} ✓）"
                "—— ⚠️ 别把别的流当视频解 ✓✗", reason="pending")
        v_slots, a_slots = shape_v[1], shape_a[2]
        if v_slots != int(retake.video_slots) or a_slots != int(retake.audio_slots):
            raise TorchBackendUnavailable(
                f"计划层算的槽位与**张量实际时间维**不一致 ✗：视频 {retake.video_slots} vs {v_slots} ✓、"
                f"音频 {retake.audio_slots} vs {a_slots} ✓ —— 窗口是**按秒映射到槽位**算的 ✓ ⇒ 两处必须"
                "是**同一条片子**的同一个事实 ✓（不等 ⇒ 必有一处错 ✓ 拒 ✗，不按张量改窗口 ✗）",
                reason="pending")
        # ⭐ 与 ``plan`` 的**第二份事实**对一遍 ✓✗（同一件事写两处 ⇒ 不等就是有一处错了 ✓）：
        #    潜帧数由 ``plan.latent_frames`` 算 ✓、音频潜帧由**秒数**算 ✓ —— 两边都必须落回同一批槽位 ✓。
        planned_slots = getattr(plan, "latent_frames", None)
        if planned_slots and int(planned_slots) != v_slots:
            raise TorchBackendUnavailable(
                f"``plan.latent_frames={planned_slots}`` 与成片潜变量的 T={v_slots} 不一致 ✗"
                "（同一件事写两处 ✓ 不等 ⇒ 必有一处错 ✓）", reason="pending")
        planned_audio = geometry_mod.audio_latent_frames(
            int(getattr(plan, "frames", 0)) / max(1, int(getattr(plan, "fps", 24))),
            mode=str(self._audioLatentMode))
        if int(planned_audio) != a_slots:
            raise TorchBackendUnavailable(
                f"按 ``plan`` 的秒数算出来是 {planned_audio} 个音频潜帧 ✓，而成片是 {a_slots} 个 ✗"
                "（重拍的音频窗口要按**秒**对齐 ✓ ⇒ 两边必须是同一条片子 ✓）", reason="pending")
        if steps[0] <= 0.0:
            raise TorchBackendUnavailable(
                f"重拍的首 σ 必须 > 0 ✗（收到 {steps[0]} ✓）—— 冻结区的噪声方向是反推出来的 ✓",
                reason="pending")
        noise_v, noise_a = self._dual_noise(v_slots, a_slots, shape_v[2], shape_v[3],
                                            int(request.seed))
        shifts = h3_form.H3_SIGMA_SHIFTS
        sigma_v0 = steps[0]
        sigma_a0 = schedules.time_shift_sigma(sigma_v0, float(shifts["sigma_shift_video"]),
                                              float(shifts["sigma_shift_audio"]))
        # 掩码：**1 = 重画 / 0 = 冻结** ✓ —— 整条先置 0（全冻结 ✓）再把窗口开成 1 ✓
        # ⚠️ 某条流"整条冻结"（video only / audio only ✓）时掩码**全 0** ✓：主干是**联合**前向 ✓
        #    ⇒ 那条流仍要逐步喂进去 ✓（velocity 换成它自己的噪声方向 ⇒ 净效果 = 原样 ✓ 不白算 ✓）。
        mask_v = torch.zeros((1, v_slots, 1, 1), device=self._device, dtype=torch.float32)
        if retake.video is not None:
            mask_v[:, int(retake.video[0]):int(retake.video[1])] = 1.0
        mask_a = torch.zeros((1, 1, a_slots), device=self._device, dtype=torch.float32)
        if retake.audio is not None:
            mask_a[:, :, int(retake.audio[0]):int(retake.audio[1])] = 1.0
        noisy = {"video": video + sigma_v0 * noise_v,
                 "audio": audio + sigma_a0 * noise_a,
                 "pin": (video, audio), "pinMask": (mask_v, mask_a),
                 # ⚠️⚠️ **锚只有 pin** ✓：首帧（``cond`` 段）**清掉** ✗ —— 它是个**时间锚** ✓，
                 #    会跟"窗外冻结"打架 ✓✗（重拍要的是"这段用同样的提示词重画" ✓，不是"接在首帧后面" ✓）；
                 #    参考块（``ref_img``）**保留** ✓ —— 那是"这条片子里是谁/什么风格" ✓ 重拍**不该换人** ✓。
                 "keyframes": [], "references": list(latents.get("references") or []),
                 "audioLatentFrames": a_slots,
                 "vaeScale": int(vae_mod.H3_VIDEO_VAE_FACTS["vaeScale"])}
        result = self.sample_dual(noisy, steps, condition, request, on_step=on_step)
        result["retake"] = {**retake.to_dict(), "sigmaVideo0": sigma_v0, "sigmaAudio0": sigma_a0}
        return result

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
                # ⚠️ 与 :meth:`_default_dit_path` 同一条解析 ✓（清单落点优先 → 动态探测根 ✓）
                resolved, _source = inv.resolve_component(entry)
                return str(resolved) if resolved else None
        return None

    def refine_latents(self, latents: Any, plan: Any, request: Any, *,
                       condition: Any = None) -> Any:
        """**超清二采（张量层 ✓ 2026-09-25 补 ✓）**：空间 2× ✓ ⇒（可选）**带掩码的低噪声重去噪** ✓✗。

        ⚠️⚠️ 四条口径（都是「看着像超分、其实坏了」的形状 ✓✗）：
        1. ⭐ **只重采视频流** ✗✗：``audio`` **原对象放回** ✓（重采音频会把音轨弄坏而画面看着正常 ✓）——
           二采那条路靠的是**掩码**（video **1** / audio **0** ✓，口径来自上游 ``_h3_build_denoise_mask`` ✓✗）；
           ⚠️ 这张掩码有**两半**作用 ✓（2026-11-01 补齐 ✓）：① **缩放增量**（采样器 ✓）② **改逐行时间戳**
           （主干 ✓ —— ``audio`` 那片的档位从 ``1−σ_a`` 换成条件档 ``1.0`` ✓ = 「已定稿的干净音轨」✓）。
           只给 ① 时音频的**数值**照样冻结 ✓（看着对 ✓）但主干仍把它当**含噪**样本 ✓✗ ⇒ 口型条件变软 ✓；
        2. ⭐ **时间维不变** ✗✗（``T`` 前后必须相等 ⇒ 不等就**拒** ✓ —— 时间插值会让动作速率变错 ✓✗）；
        3. ⭐ 放大器**严格装载** ✓：契约与张量不符（缺键/形状不一 ✓）⇒ **拒** ✗
           （``strict=False`` 会留下随机初始化的层 ⇒ 输出是「像超分」的噪声 ✓✗）；
        4. ⭐ **二采的两个参数必填** ✗（``plan.refine_steps`` / ``refine_denoise`` ✓）—— 默认值在
           **编译层** ✗ ⇒ 不给就**只做上采样** ✓ 并如实写进报告 ✓；给了一半 ⇒ ``plan_upscale`` 就报错 ✓。
        ⚠️ **要二采就必须给 ``condition``** ✗（二采要重跑主干 ✓ —— 没条件就**拒** ✓，不猜 ✓✗）。
        """
        self._gate()
        # ⭐ **每次入口先清空上这次的实况** ✗✗（2026-11-08 补 ✗）：``refineDetails`` 是**实例属性** ✓，
        #    上一次二采留下的值会活到这一次 ✓✗ —— 本次若走「只上采样」或中途在设值**之前**返回 ✓，
        #    管线读到的就是**上一次**的实测倍率/尺寸 ✓（比如上次 1.5×、这次 2×，回执写 1.5 而画面是 2 ✓）。
        #    症状属于本仓最怕那类：**数字自洽、只与画面不符** ✓。清空后读不到 ⇒ 调用方当场知道"没自述" ✓。
        self.refineDetails = None
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
        # ⚠️ 读 + **结构核** ✓（``load_upscaler_contract``）：只读不核时 ✗，契约缺一个字段会走
        #    ``H3LatentUpscalerV2`` 的**默认值**建网络 ✓✗（形状自洽 ⇒ 报错会落在"张量不符"上，
        #    看起来像权重坏了 ✓✗）；这里当场把真因摆出来 ✓。
        # ⭐ 通道数**从已装主干推** ✗（不写死 24 ✗）—— 自检的**缩小版**与真权重要走**同一条路** ✓✗
        #    （真权重推出来就是 24 ✓，缩小版推出来是它自己的 latents_dim ✓）。
        #    ⚠️ 必须在**读契约之前**算出来并传下去 ✗✗：契约校验那一层原先写死 24 ⇒ 缩小版**在进主干
        #    推导之前**就被当成"不是这套检查点"拒了 ✓✗（症状＝真权重全绿、自检整条崩在契约校验上 ✓）。
        expected = int((self._config or {}).get("latents_dim") or lc.H3_VIDEO_CHANNELS)
        try:
            contract = upscale_mod.load_upscaler_contract(metadata, expect_channels=expected)
        except ValueError as err:  # ⚠️ ``ContractError`` 就是 ``ValueError`` ✓（读不出 / 格式对不上 / 结构不合法 ✓）
            raise TorchBackendUnavailable(
                f"放大器内嵌契约**不可用** ✗（读不出 / ``format`` 对不上 / 结构不合法 ✓）：{err} ✓"
                f" ⇒ 拒，不拿默认值糊过去 ✓✗"
                f"（要么换配套的检查点 ✓，要么让计划层按 ``contract=None`` 回退普通模式 ✓）",
                reason="pending") from err
        if contract is None:
            raise TorchBackendUnavailable(
                f"放大器权重 {path} 里**没有内嵌契约** ✗（``strict_latent_only`` / ``base_config`` / "
                f"``config`` 三样缺 ✓）⇒ 不知道网络怎么建 ✓ 不猜 ✗（计划层按 ``contract=None`` 回退 ✓）",
                reason="pending")
        channels = int(contract["base_config"]["in_channels"])
        # ⭐ 跨来源**同一个事实** ✓✗：契约的 ``in_channels`` ↔ **本次装的主干**的视频流通道 ✓。
        #    （``expected`` 已在读契约之前算好并透给校验层 ✓ —— 这里再核一次是**有意的**双保险 ✗✗：
        #    万一将来有人只调 :func:`upscale.check_upscaler_contract` 而忘了传 ``expect_channels`` ✓，
        #    这条仍会拦住 ✓。）
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
        # ── ④c 缩到**计划的**倍率 ✓（2026-11-08 补 ✗✗）────────────────────────────────
        # 放大器只会做**空间 2×** ✓（`upscale.plan_upscale` 的 notes 早就写明这一条 ✓）；
        # 而 `ai-2x-resize` 的第二步「高质量缩放 2× → 1.5×」此前**一步都没执行** ✗✗：
        # 计划要 1.5× ⇒ 实际出的是 2× ⇒ 而 `result.refine.scale` 照写 1.5 ✓✗
        # （**尺寸与回执对不上、回执自己却"自洽"** —— 正是本项目最怕的那类错 ✓）。
        # ⚠️ 顺序是有意的：**先缩到目标域、再二采** ✓ 与计划层 notes 的
        #    「带掩码的二次去噪（先 1.5× 的目标域 ✓）」逐字一致 ✓✗（反过来就是在 2× 域精修）。
        target, scale_notes, method = self._plan_target_size(plan, shape, after)
        details["notes"].extend(scale_notes)
        out_video = refined
        if target is not None:
            out_video = self._scale_video_latents(refined, target, method)
            scaled = tuple(int(value) for value in out_video.shape)
            if scaled[1] != shape[1]:
                raise TorchBackendUnavailable(
                    f"缩放**改了时间维** ✗：{shape[1]} → {scaled[1]} ✓"
                    f"（时间插值会让动作速率变错 ✓✗） ⇒ 拒，不用这个产物 ✓", reason="pending")
            details["videoShapeScaled"] = list(scaled)
        # ⭐ **实测倍率对每一次二采都要有** ✗✗（2026-11-08 补 ✗）：原先只在「有第二步缩放」时写 ✓，
        #    ``ai-2x`` 那条（放大器原样 2×、target 为 None ✓）回执里就没有实测倍率 ✓✗ ——
        #    管线只能拿**计划值**顶 ✓，正好回到「计划层 vs 执行层」那个老毛病 ✓。
        scaled_shape = details.get("videoShapeScaled") or list(out_video.shape)
        scale_h = round(int(scaled_shape[2]) / int(shape[2]), 6)
        scale_w = round(int(scaled_shape[3]) / int(shape[3]), 6)
        details["targetScale"] = float(getattr(plan, "scale", 0.0) or 0.0)
        #: ⚠️ 以**高度**比为准（与 ``videoShapeScaled`` 同一口径 ✓）；宽度比另存 ✓
        details["spatialScale"] = scale_h
        details["spatialScaleW"] = scale_w
        if abs(scale_h - scale_w) > 1e-6:
            details["notes"].append(
                f"⚠️ 宽高实际倍率不同（高 {scale_h:g}× / 宽 {scale_w:g}× ✓）—— 目标尺寸按整数取整所致 ✓")
        if target is not None:
            details["resizeMethod"] = method
        keyframes = self._upscale_keyframes(latents, refined, shape, target=target)
        if tail_steps > 0:
            # ⚠️ 二采吃**缩放后**的 `out_video` ✓✗：传 `refined` 就是在 2× 域精修、
            #    出来再缩 —— 与计划层的口径相反（细节生成在错误的分辨率上 ✓）。
            out_video, denoise_report = self._second_pass(
                out_video, latents, plan, request, condition=condition, keyframes=keyframes)
            # ⚠️⚠️ **notes 要并进去、不能 ``update`` 覆盖** ✗✗（2026-11-08 补 ✗）：
            #    ``denoise_report`` 自己也带 ``"notes"`` ✓ ⇒ 直接 ``details.update(...)`` 会把
            #    上面这段攒的**缩放假话**（「放大器 2× → 计划 1.5× ✓」「宽高倍率不同」✓）
            #    整段冲掉 ✓✗ —— 而**只要二采真跑了就必然冲掉** ✓（成功/失败两条路径都带 notes ✓）。
            #    症状：回执里只剩「二采：尾部 N 步」✓，读者**根本不知道画面被缩过** ✓✗
            #    （倍率与尺寸那两格还在 ✓，所以不会有任何一处报错 ✓）。
            tail_notes = list(denoise_report.pop("notes", ()) or ())
            details.update(denoise_report)
            details["notes"].extend(tail_notes)
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

    #: `ai-2x-resize` 第二步「高质量缩放」用的重采样方法 ✓ ——
    #: 2× → 1.5× 是**缩小** ✓：`bilinear` 偏糊 ✗、`lanczos` 在潜空间容易振铃 ✗、
    #: `bicubic` 居中 ✓（可换 ✓；回执里如实报方法名 ✓，不让人去猜用的哪一档 ✓）。
    SCALE_RESIZE_METHOD = "bicubic"

    @staticmethod
    def _target_latent(value: float) -> int:
        """目标潜空间边长：至少 2 ✓ —— 但**不强制偶数** ✗✗。

        潜变量 ×8 就是像素 ✓ ⇒ 只要边长是整数，像素必是 8 的倍数 ⇒ h264 要的"偶数宽高"
        **天然满足** ✓；反过来硬取偶数会把 2 → 3 顶成 4 ✓✗ ⇒ 1.5× 悄悄变成 2×
        （正是本函数要防的那个错 ✓；实测踩过 ✓）。
        """
        return max(2, int(round(value)))

    def _plan_target_size(self, plan: Any, before: tuple[int, ...], after: tuple[int, ...]
                          ) -> tuple[tuple[int, int] | None, list[str], str]:
        """按超清计划算**目标潜空间尺寸** ``(H, W)``；``None`` ⇒ 不缩（保持放大器的 2× ✓）。

        三种模式与 :func:`app.services.engine.upscale.plan_upscale` 逐条对齐 ✓：

        * ``ai-2x`` ⇒ 目标就是 2× ✓（放大器给足了 ⇒ ``None``；给不足 ⇒ 按计划缩回去 ✓）；
        * ``ai-2x-resize`` ⇒ 2× 再缩到 ``scale`` ✓（**这一步此前完全缺失** ✗✗）；
          ⚠️ 只接受 ``1 < scale < 2`` ✓（计划层产出这个模式的唯一区间 ✓）—— 区间外**拒** ✗✗，
          因为 ``≤1`` 是「放大再缩回原尺寸」（尺寸不变、耗时 6× ✓✗）、``≥2`` 是「靠插值冒充分块放大」
          （2× 以上不会长细节 ✓✗），两者都会交付一张**自洽的假超清** ✓；
        * ``tiled-ai-2x`` ⇒ ``scale > 2`` 要**多次分块放大 + 缝合** ✓ —— 执行层没做 ✗
          ⇒ **拒** ✗（静默出 2× 会让人以为 4× 开了 ✓✗）。

        ⚠️ ``mode`` 为空（``GenerationPlan`` 直传 / 老调用）⇒ ``None``：行为**零变化** ✓。
        """
        mode = str(getattr(plan, "mode", "") or "")
        if not mode:
            return None, [], self.SCALE_RESIZE_METHOD
        scale = float(getattr(plan, "scale", 0.0) or 0.0)
        if mode == "ai-2x":
            want = (before[2] * 2, before[3] * 2)
            if (after[2], after[3]) == want:
                return None, [f"倍率 2× 由放大器直接给出 ✓（{after[2]}×{after[3]} 潜变量 ✓）"], \
                    self.SCALE_RESIZE_METHOD
            return want, [f"⚠️ 放大器给出 {after[2]}×{after[3]} ✗，而计划是 2×"
                          f"（{want[0]}×{want[1]} ✓）⇒ 按计划缩回去 ✓"], self.SCALE_RESIZE_METHOD
        if mode == "ai-2x-resize":
            # ⚠️⚠️ 区间必须核 ✗✗（2026-11-08 补 ✗）：计划层 ``plan_upscale`` 只在 ``1 < scale < 2``
            #    才产这个模式 ✓（``scale ≤ 1`` ⇒ normal ✓、``scale > 2`` ⇒ tiled ✓）；而请求体是
            #    **人/前端手写的 dict** ✓ ⇒ ``mode`` 与 ``scale`` 可以配出计划层永远不会产出的组合 ✓✗。
            #    两种组合都会**静默出一张"假超清"** ✓：
            #      · ``scale ≤ 1`` ⇒ 先放大 2× 再缩回原尺寸 ✓ ⇒ 尺寸与普通模式**一模一样**、
            #        耗时却是 ≈6× ✓✗，回执 ``applied=true / scale=1.0`` 自洽得没有任何破绽 ✓；
            #      · ``scale ≥ 2`` ⇒ 放大器只给 2× ✓ ⇒ 再从 2× **插值**放大到 3×/4× ✓✗
            #        （bicubic 放大不会长出细节 ✓，却会被当成"AI 4×"交付 ✓）。
            if scale <= 1.0:
                raise TorchBackendUnavailable(
                    f"``ai-2x-resize`` 要 {scale:g}× ✗：**≤1× 不需要放大器** ✓ ⇒ 这次等同普通模式"
                    f"（先 2× 再缩回原尺寸 = 尺寸不变、耗时 ≈6× ✓✗）⇒ **拒** ✗ "
                    f"—— 真的要原尺寸请把超清关掉（``mode=normal``）✓，要放大请给 >1× ✓",
                    reason="pending")
            if scale >= 2.0:
                raise TorchBackendUnavailable(
                    f"``ai-2x-resize`` 要 {scale:g}× ✗✗：这个模式的名字就写着**缩到 2× 以下** ✓"
                    f"（放大器只会做**空间 2×** ✓）；再往上**只能靠插值** ✓✗（不会长出新细节 ✓）"
                    f"⇒ **拒** ✗ —— {scale:g}× 若真要 AI 放大得走 ``tiled-ai-2x``（多次分块放大 + 缝合 ✓，"
                    f"本后端**没实现** ✗ ⇒ 也会拒 ✓）；恰好 2× 请用 ``ai-2x`` ✓（放大器直接给 ✓）",
                    reason="pending")
            height = self._target_latent(before[2] * scale)
            width = self._target_latent(before[3] * scale)
            note = (f"缩放：放大器 2× → 计划 {scale:g}× ✓（潜变量 {height}×{width} ✓、"
                    f"方法 {self.SCALE_RESIZE_METHOD} ✓）")
            actual = height / before[2]
            if abs(actual - scale) > 0.005:
                note += f"；⚠️ 取整后实际 {actual:.4g}×"
            return (height, width), [note], self.SCALE_RESIZE_METHOD
        if mode == "tiled-ai-2x":
            raise TorchBackendUnavailable(
                f"``tiled-ai-2x``（{scale:g}× 分块）**执行层没实现** ✗✗：本后端只做一次整张 2× ✓，"
                f"而 {scale:g}× 要**多次分块放大 + 缝合** ✓（大块解码会出 NaN/Inf ✓✗）⇒ **拒** ✗ "
                f"—— 静默出 2× 会让你以为 {scale:g}× 开着 ✓✗；只要 1.5× 请用 ``ai-2x-resize`` ✓",
                reason="pending")
        raise TorchBackendUnavailable(
            f"不认识的超清模式 {mode!r} ✗ ⇒ **不猜**（已知：ai-2x / ai-2x-resize / tiled-ai-2x ✓）",
            reason="pending")

    def _scale_video_latents(self, video: Any, target: tuple[int, int], method: str) -> Any:
        """把 ``[C,T,H,W]`` 缩到目标 ``(H, W)`` ✓ —— **只动空间维**（逐帧 ✓、时间不插值 ✗✗）。

        ⭐ 这里就是 :mod:`image_ops` 的真接线点 ✓：它吃 5 维 ``(B,C,T,H,W)`` 并把 ``T`` 折进
        batch **逐帧**缩 ✓ —— 不该自己再写一份双线性 ✗（两份口径迟早对不上 ✓✗）。
        """
        from app.services.engine import image_ops  # noqa: PLC0415

        scaled = image_ops.common_upscale(video.unsqueeze(0), int(target[1]), int(target[0]),
                                          method, "disabled")
        return scaled.squeeze(0).to(video.dtype)

    def _upscale_keyframes(self, latents: Any, refined: Any, shape: tuple[int, ...], *,
                           target: tuple[int, int] | None = None) -> Any:
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
            if target is None:
                upscaled = upscale_net.spatial_bilinear_2x(latent.float())
            else:
                # 主视频要缩到**目标域** ✓ ⇒ 关键帧必须同尺寸 ✓✗（两边不同分辨率擦边对上时
                # 是**静默**错 ✓✗）。直接一次缩到位 ✓ 不必"先 2× 再缩"✓（多插一次反而更糊 ✓）。
                from app.services.engine import image_ops  # noqa: PLC0415
                upscaled = image_ops.common_upscale(
                    latent.float(), int(target[1]), int(target[0]),
                    self.SCALE_RESIZE_METHOD, "disabled")
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
        # ⚠️ σ 口径必须与**一采**同源 ✓✗（`pipeline.video_flow_shift_for` 是 flow 字符串的唯一解析处 ✓）：
        #    二采若自己按 `karras` 重算 ⇒ 掩码比例对、**σ 域却换了一族** ✓✗ ⇒ 尾部那几步
        #    σ 会从 flow 的 [0,1] 跳到 EDM 的十几 ✓ ⇒ `t = 1−σ` 为负 ✓、不报错、糊片 ✓✗。
        from app.services.engine import pipeline as pipeline_mod  # noqa: PLC0415

        spec = str(getattr(request, "schedule", "") or "flow")
        try:
            if pipeline_mod.is_flow_schedule(spec):
                full = schedules.flow_sigmas_for_steps(total,
                                                       shift=pipeline_mod.video_flow_shift_for(request))
            else:
                full = schedules.sigmas_for(total, spec)
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
                keyframes=keyframes or None, denoise_mask=(ones, zeros),
                # ⚠️ 二采**也要**同一份文本标签 ✓✗：这里漏了的话二采会把视觉占位 token
                #    按 text 行调制 ⇒ 一采对了、二采又偏 ✓ —— 表现是"超清一遍反而差一点" ✓✗
                text_token_tags=getattr(request, "text_token_tags", None))
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

    def _wan_concat_latents(self, latents: Any) -> Any:
        """把 16 通道噪声拼成 Wan I2V 的 **36 通道** 入口 ✓（顺序由 :data:`dit_wan22.I2V_LAYOUT` 说 ✓）。

        ## 两个必须拒跑的场合（都不是"少一点质量"，是**结果完全无关** ✗✗）

        * **没条件**（`:meth:`_wan_i2v_condition` 没跑 / 没给首帧 ✓）⇒ I2V 权重没有 20 通道可拼 ✓，
          零填充"凑够 36"不会有任何报错 ✓，只是出片跟首帧毫无关系 ✓✗；
        * **条件与噪声帧数/尺寸不一致** ✓ ⇒ 拼出来像 36 通道 ✓、``in_dim`` 校验也过 ✓，
          但每一拍对应的位置错位 ✓✗。

        为什么不在 ``denoise`` 里自己算条件 ✗：那时 VAE 已经被 `enter_phase("sample")` 退掉了 ✓
        （DiT 26.62 GiB ＋ VAE 0.24 GiB 与文本塔三件同时驻留装不进 22.49 GiB ✓）⇒ 条件必须
        在 encode 段做好存起来 ✓。
        """
        torch = self._torch()
        from app.services.engine import dit_wan22 as wan22_mod  # noqa: PLC0415

        image = getattr(self, "_i2vImage", None)
        mask = getattr(self, "_i2vMask", None)
        if image is None or mask is None:
            raise TorchBackendUnavailable(
                "装的是 **Wan I2V** 权重（in_dim=36 ✓）但**没有首帧条件** ✗ "
                "⇒ 这条路线**必须有 `firstFrame`** ✓（T2V 那份 16 通道权重才是纯文本 ✓）。\n"
                "  · 真因几乎总在上游：`pipeline.run_sync(stage=\"wan22\", firstFrame=<图>)` 没给 ✓ / "
                "`encode_text` 没跑过 ✓ / 中间有人清了 `_i2vImage` ✓。\n"
                "  · 零填充「凑够 36」不会有报错 ✓，只是出片跟首帧**毫无关系** ✓✗ ⇒ 这里拒跑 ✓。",
                reason="input")
        if getattr(latents, "ndim", 0) != 5:
            raise TorchBackendUnavailable(
                f"Wan 的潜变量要 5 维 (b,16,t,h,w) ✗，实际 {tuple(getattr(latents, 'shape', ()))} ✓",
                reason="input")
        return wan22_mod.i2v_concat(
            latents,
            image.to(device=latents.device, dtype=latents.dtype),
            mask.to(device=latents.device, dtype=latents.dtype))

    def denoise(self, latents: Any, sigma: float, condition: Any, request: Any) -> Any:
        """**真前向**（装了模型 ✓）或**占位**（没装 ✓）—— 由 :meth:`describe` 如实标注 ✓。

        * 真前向：``x0 = DiT(x, σ, context)``（流匹配 velocity → x0 ✓ 见 :func:`dit.flow_match_x0` ✓）；
        * 占位：``(1−g)·cond + g·x``（``g→0`` 当 ``σ→0`` ✓ 保证采样按时收敛 ✓）。

        ⚠️ **σ 的形态按族分** ✗：通用 DiT 收标量 ✓；Z-Image（:class:`dit_lumina.LuminaDiT`）与
        Qwen-Image 2.1（:class:`dit_qwen21.Qwen21DiT`）都收 ``(b,)`` 的**逐 batch 张量** ✓
        （见 :func:`_timesteps_of` ✓）—— 三者都 ``form="dit"`` ✓✗，所以判据是 ``_ditFamily`` ✗
        不是 ``_form`` ✓（喂错形态不会报错，只会把 batch 里的各条当成同一个 σ ✓✗）。
        """
        self._gate()
        #: ⚠️⚠️ **H3 形态没有单流前向** ✗✗（2026-11-01 真权重实跑当场抓到 ✓）：
        #: `H3FormTrunk.forward` 的头三个位置口是 **(视频行, 音频行, 文本行)** ✓，紧接着还要
        #: `σ_v` / `latent_h` / `latent_w` ✓ —— 而本方法的契约是 `(latents, σ, context)` ✓✗。
        #: 硬套下去只会得到「`forward() missing 3 required positional arguments:
        #: 'sigma_v', 'latent_h' and 'latent_w'`」✓，**一个字都不指向真因** ✗（真因窗口是
        #: 「这一档双流没开 ⇒ `pipeline` 走了单流分支 ✓✗」—— 见 `dualStream.blockedBy` ✓）。
        #: ⇒ 这里**当场拒** ✓，并把缺的几格逐条摊开 ✓（缺什么就报什么 ✓，不静默降级 ✗）。
        if self._model is not None and getattr(self, "_form", None) == self.H3_FORM_NAME:
            state = self.dualStream
            missing = state.get("blockedBy") or []
            raise TorchBackendUnavailable(
                "装的是 **H3 形态**（`H3FormTrunk` ✓ 双流 ✓）✗ —— 它**没有单流前向** ✗："
                "`forward` 要的是「视频行 + 音频行 + 文本行」三路行张量 ✓，"
                "而 `denoise(latents, σ, context)` 只喂得起一路 ✓✗。\n"
                "  · 真因窗口：这一档双流没开 ⇒ `pipeline` 的 `dual=False` ⇒ 走了单流采样 ✓✗"
                "（报错却落在 `forward` 少参数上 ✓，看着像「主干坏了」 ✗）。\n"
                "  · 现在缺的：" + ("、".join(missing) if missing else "（`dualStream` 说齐了 ✓ —— "
                "那是本方法用错场景了 ✗）") + "\n"
                "  · 怎么修：把上面几格补齐 ✓（``ensure_loaded(..., audio_latent_mode=\"round\")`` ✓ "
                "+ ``attach_audio_vae()`` ✓）⇒ `pipeline` 自动改走 `sample_dual` ✓",
                reason="pending")
        # ⚠️ 调用方给的张量（自检常用 ✓）必须已在后端设备上 ✓✗ —— 不搬，只点破 ✓（见校验器 ✓）。
        self._require_own_device("denoise", latents=latents, condition=condition)
        torch = self._torch()
        sigma = float(sigma)
        if self._model is not None:
            #: ⭐ **Edit 2511 的参考图**（2026-11-01 接上 ✓）：那份权重靠 `__index_timestep_zero__`
            #: 哨兵把"参考图 token"与"目标 token"拼在一条序列里 ✓、并给参考图行喂 σ=0 的调制行 ✓
            #: ⇒ ``ref_latents`` **每一步**都得给 ✓✗（第一步给、后面不给 ⇒ 中途"忘了参考图" ✓，
            #: 图会从"照着改"漂成"按提示词重画" ✓，且**一条报错都没有** ✓）。
            edit_refs = None
            if getattr(getattr(self._model, "shape", None), "ref_timestep_zero", False):
                stored = getattr(self, "_refLatents", None)
                if not stored:
                    raise TorchBackendUnavailable(
                        "装的是 **Qwen-Image Edit 2511** 的权重 ✓，但 `ref_latents` 不在 ✗ "
                        "⇒ 参考图那一路整段缺席 ✓✗。"
                        "真因几乎总在上游：`encode_text` 没跑过 ✓ / 跑过但没给 `referenceFrames` ✓ / "
                        "中间有人把 `_refLatents` 清了 ✓（拒跑 > 静默按 t2i 出图 ✓）。",
                        reason="pending")
                #: ⚠️ 逐张搬（**多图** ✓）：`_refLatents` 是列表 ✓，漏掉 `list()` 直接
                #: `.to()` 会 AttributeError ✓（至少是报错 ✓，不会静默 ✗）；而只搬第 0 张
                #: ⇒ 第 2 张图的条件凭空消失 ✓✗（不报错 ✓）。
                edit_refs = [item.to(device=latents.device, dtype=latents.dtype) for item in stored]
            with torch.no_grad():
                #: σ 的**形态**也是契约（:class:`dit_build.ForwardContract`）✓：图族收 ``(b,)``
                #: 张量 ✓、老路通用 DiT 收标量 ✓ —— 喂错不会报错 ✗，只会让 batch 里各条共用同一个 σ ✓✗。
                timesteps = (_timesteps_of(torch, latents, sigma)
                             if dit_build.sigma_form_of(getattr(self, "_ditFamily", None)) == "batch"
                             else sigma)
                #: ⚠️ `ref_latents` 是**关键字** ✓、且**只有 Edit 那一族**收 ✓ ⇒ 无条件传会给
                #: 2512 t2i 主干一个 ``TypeError`` ✓✗（它的 forward 没有这个参数 ✓）。
                #: ⚠️⚠️ 它的类型是 **``Sequence[Tensor]``** ✓✗（``dit_qwen_image.forward`` ✓）——
                #: 传单个 4 维张量**不会**报「类型不对」✗：``if ref_latents:`` 为真 ✓、
                #: 然后 `for ref in ref_latents` 会**按 batch 维遍历** ✓ ⇒ 每个 ``ref`` 变 3 维 ✓
                #: ⇒ `_patch` 里 `b, c, h, w = x.shape` 直接 ``ValueError`` ✓（batch=1 时才炸得这么早 ✓；
                #: 这种"少包一层列表"读起来完全像对的 ✓✗）。
                if getattr(self, "_ditFamily", None) == "seedvr2":
                    #: ⚠️⚠️ **SEEDVR2 的 forward 与出图族不同位置** ✗✗（2026-11-01 接线 ✓）：
                    #: 它是 ``forward(latents, σ, context, conditioning=)`` ✓ —— 第 **4** 位那个
                    #: ``conditioning``（17 通道 ✓）是**必填** ✓，喂在第 3 位（context 位）只会
                    #: 得到「缺 ``conditioning``」或静默把条件当文本用 ✓✗。
                    #: 本族**没有文本塔** ✓ ⇒ ``context=None`` 是**正解** ✓（不是"漏了提示词" ✓）。
                    conditioning = condition if hasattr(condition, "shape") else getattr(
                        self, "_repairCondition", None)
                    if conditioning is None:
                        raise TorchBackendUnavailable(
                            "装的是 **SEEDVR2 修复主干** ✓ 但**没有 17 通道 conditioning** ✗ "
                            "⇒ 这一族**不能凭空生成** ✓（它的输入是「退化视频 16ch ＋ loss mask 1ch」✓）。\n"
                            "  · 真因几乎总在上游：`encode_text` 没跑过 ✓ / 跑过但没给 "
                            "`referenceVideos[0]` ✓ / 中间有人把 `_repairCondition` 清了 ✓。\n"
                            "  · 怎么修：`pipeline.run_sync(stage=\"seedvr2\", referenceVideos=[<源视频>])` ✓"
                            "（拒跑 > 静默按噪声出一片灰 ✓）。", reason="input")
                    prediction = self._model(latents, timesteps, None,
                                             conditioning=conditioning.to(
                                                 device=latents.device, dtype=latents.dtype))
                elif getattr(self, "_ditFamily", None) == "wan":
                    #: ⚠️⚠️ **Wan I2V 的入口是 36 通道** ✗✗（2026-11-02 接线 ✓）：主干只吃
                    #: ``[噪声 16, 掩码 4, 条件图 16]`` ✓，直接喂 16 通道的潜变量会当场报
                    #: 「Wan 的 in_dim 是 36」✓；而**只改 in_dim 不改拼接**更糟 ✗ ——
                    #: 通道数凑对 ✓、条件图与掩码错位 ✓ ⇒ 出片完全不受首帧约束 ✓且一条报错都没有 ✓✗。
                    #:
                    #: ⭐ 双专家接力 ✓：σ 跌破 ``plan.sigmas[wanHighSteps]`` 就**换装**（原地重装 ✓，
                    #: 骨架/驻留切分不动 ✓）—— 判据是 σ 而不是调用次数 ✓（每步可能 2~4 次 ✓✗）。
                    self._wan_expert_tick(sigma)
                    prediction = self._model(self._wan_concat_latents(latents), timesteps, condition)
                elif edit_refs is None:
                    prediction = self._model(latents, timesteps, condition)
                else:
                    #: ⚠️ 传的是**列表** ✓：一张图一个元素 ✓（单图也要包一层 ✓✗）。
                    prediction = self._model(latents, timesteps, condition,
                                             ref_latents=list(edit_refs))
            return dit_mod.flow_match_x0(prediction, latents, sigma)
        with torch.no_grad():
            g = min(0.5, sigma / (sigma + 1.0)) if sigma > 0 else 0.0
            return condition * (1.0 - g) + latents * g

    def _encode_pixels(self, frames: Any) -> Any:
        """像素 ⇒ **归一化潜变量** ✓（``[B,C,T,h,w]`` ✓）；按**已挂 VAE 的口径**做正向缩放 ✓。

        ## 为什么必须走这一个口子（**别再裸调 `self._vae.encode`** ✗）

        各家的 ``encode`` 返回值**三种口径**混在一起 ✓✗：

        | VAE | 返回 | 归一化 |
        |---|---|---|
        | H3 视频 VAE | ``[B,24,T,h,w]`` 张量 ✓ | **自带** mean/std ✓ |
        | Flux / SDXL（:mod:`sdxl_vae` 那套） | ``(mean, logvar)`` **二元组** ✗ | **没有**（差 ``0.3611/0.1159`` ✓） |
        | Wan（:class:`vae_wan.WanVAE`） | ``[B,z,T,h,w]`` ✓ | **没有**（差 16 组 mean/std ✓） |
        | 参考未训 VAE（:mod:`vae`） | ``[B,C,T,h,w]`` ✓ | 交给 :meth:`normalize_latents` ✓ |

        裸调的结果是：Wan 族 ``AttributeError`` ✓（这个方法以前**不存在** ✓✗ —— 2026-11-01 补 ✓）、
        Flux 族把二元组当张量 ⇒ ``AttributeError: 'tuple' object has no attribute 'shape'`` ✓✗、
        参考 VAE 族拿到**未归一化**的潜变量 ⇒ 首帧条件差一个量级 ✓、**形状全对、不报错** ✓✗。

        ⚠️ 这里**不吞异常** ✗：认不出的 VAE 种类 ⇒ 人话指出"哪种 VAE 没有编码器" ✓（别让调用方
        去猜 ``AttributeError`` ✓）。
        """
        if self._vae is None:
            raise TorchBackendUnavailable(
                "没挂 VAE ✗ ⇒ 没法把图片编码成潜变量 ✓（先 `attach_vae()` ✓，"
                "或让这条条件走占位路径 ✓）", reason="attach")
        kind = self._vaeKind
        if kind == "seedvr2-real":
            #: ⚠️ SEEDVR2 的编码器**不是**解码器的镜像 ✓（出 32 通道＝16 均值 + 16 logvar ✓、
            #: 无 ``quant_conv`` ✓）⇒ 取前 16 的活由 :func:`vae_seedvr2.encode_latents` 做 ✓，
            #: 且它已经把 ``× 0.9152`` 算进去 ✓（裸调 ``self._vae.encode`` 会漏这一步 ✓✗）。
            from . import vae_seedvr2 as vae_seedvr2_mod  # noqa: PLC0415
            return vae_seedvr2_mod.encode_latents(self._vae, frames, family="seedvr2")
        if kind == "h3-real":
            #: H3 视频 VAE 的 ``encode`` **自带** mean/std 归一化 ✓（见 :mod:`vae_h3` ✓）⇒ 原样返 ✓。
            return self._vae.encode(frames)
        if kind == "wan21-real":
            from . import vae_wan as vae_wan_mod  # noqa: PLC0415 —— 只在真编码这条路上用 ✓
            family = self._vaeLatents[2] if self._vaeLatents else None
            if not family:
                raise TorchBackendUnavailable(
                    "Wan 族 VAE 已挂但**登记口径丢了** ✗（``_vaeLatents`` 为空 ✓）"
                    "⇒ 不知道用哪组 mean/std 归一化 ✓，别猜 ✓", reason="attach")
            return vae_wan_mod.encode_latents(self._vae, frames, family=family)
        if kind == "flux-real":
            from . import vae_flux as vae_flux_mod  # noqa: PLC0415
            latents = vae_flux_mod.encode_latents(self._vae, frames, config=self._vaeConfig)
            #: ⚠️ Flux 的编解码器是 **2D** ✓（没有时间维 ✗）⇒ 单帧图补一个 ``T=1`` ✓，
            #: 下游（掩码混合 ✓）一律按 5 维处理 ✓。
            return latents if latents.ndim == 5 else latents.unsqueeze(2)
        if kind == "reference-untrained":
            latents = self._vae.encode(frames)
            normalize = getattr(self._vae, "normalize_latents", None)
            return normalize(latents) if callable(normalize) else latents
        raise TorchBackendUnavailable(
            f"认不出的 VAE 种类 {kind!r} ✗ ⇒ 不知道该怎么把像素编码成**这个族**的潜变量 ✓"
            f"（H3/Wan/Flux/参考各一种口径 ✓；新族要先在 `_encode_pixels` 里登记 ✓）",
            reason="attach")

    def condition_first_frame(self, latents: Any, image_path: str, mask: list[float],
                             plan: Any, request: Any) -> Any:
        """首帧条件 ✓ —— **真张量按掩码混合** ✓（掩码由引擎算出 ✓ 见 `conditioning` ✓）。"""
        self._gate()
        # ⚠️ 这条最有必要点破 ✓：这里的 latents 常是**自检自己造的**（真链路里来自 `init_latents` ✓）⇒
        #    在**有卡那台**上最容易"自检 cpu / 后端 cuda" ⇒ 原报错指向 `aten::slow_conv3d_forward` ✗。
        self._require_own_device("condition_first_frame", latents=latents)
        #: ⭐ **Wan 2.2 I2V**：条件不走"混进第 0 潜帧" ✗，而是**每一步都把 20 通道并进主干入口** ✓
        #: （`:func:`dit_wan22.i2v_concat` ✓）。⇒ 这里**不动 latents** ✓，只把条件做出来存好 ✓：
        #: 主干看到的 ``[噪声 16, 掩码 4, 条件图 16]`` 才是 Wan 的官方条件通路 ✓✗
        #: （混进 latents 只会得到"第 0 帧像是对的、后面全靠自由发挥" ✓且不报错 ✓）。
        if getattr(self, "_ditFamily", None) == "wan":
            report = self._wan_i2v_condition(request, plan, want=latents.dtype)
            if report is None:
                raise TorchBackendUnavailable(
                    "Wan I2V 要首帧条件 ✗，但这一拍拿不到 `firstFrame` ✓ —— "
                    "它必须由 `pipeline` 从请求里传下来 ✓（`firstFrame` 与 `initImage` 是两个词 ✓✗）",
                    reason="input")
            self._conditioningNote = {
                "mode": "i2v-concat",
                "maskFrames": int(report["mask_frames"]),
                "concat": int(report["concat"]),
                "imageInfo": {"name": report["source"], "shape": list(report["shape"])},
                "note": ("首帧**不进 latents** ✓ —— 走 Wan 的 **36 通道**条件入口 ✓"
                         f"（{report['concat']} = 噪声 16 ＋ 掩码 4 ＋ 条件图 16 ✓，"
                         f"掩码首拍为 1、其余为 0 ✓，布局 {'、'.join(report['layout'])} ✓）；"
                         "潜变量原样返回 ✓（每步在 `denoise` 里拼 ✓）"),
            }
            return latents
        torch = self._torch()
        # ⚠️ 两条路都要支持（初版只按 5 维写 ⇒ 占位一维时 `reshape` 直接崩 ✗，自检 ㊾ 抓到 ✓）：
        #   * **真形状潜变量** (B,C,T,h,w) ⇒ 按掩码**逐潜帧**混 ✓（引擎算的 mask ✓ 长度 = T ✓）；
        #   * **占位一维潜变量** ⇒ 按「首帧权重」整体混 ✓（占位下谈逐帧没意义 ✓，如实标 ✓）。
        # ⚠️ **哪条是真、哪条是占位，别混着说** ✗✗（2026-11-01 更正 ✓：这段旧注释写着「仍是占位数值
        #    —— 缺读图/缩放/组 batch 的胶水」，而它正下方那条 5 维分支**已经在做真编码** ✓✗）：
        #    * 5 维 + 挂了 VAE + 首权重 > 0 ⇒ **真数值** ✓（读图 ✓ → 缩放到潜变量尺寸 ✓ →
        #      `_encode_pixels` 按已挂 VAE 的口径归一化 ✓ → 按掩码混进第 0 个潜帧 ✓）；
        #    * 5 维但没挂 VAE / 首权重为 0 ⇒ **占位向量** ✓（如实标 `mode="placeholder"` ✗）；
        #    * 一维占位潜变量 ⇒ **无从谈起** ✓（如实标 `mode="placeholder-1d"` ✗）。
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
                    #: ⚠️ 走统一编码口子 ✓✗（裸调 `self._vae.encode` ⇒ Wan 族 AttributeError ✓、
                    #: Flux 族拿到 `(mean, logvar)` 二元组 ✓、参考 VAE 族拿到**未归一化**潜变量 ✓）。
                    encoded = self._encode_pixels(
                        image.to(device=latents.device, dtype=latents.dtype))
                # ⚠️ **只比 C/h/w，不比 T** ✓ —— 图片天然是**单帧** ⇒ 编码出 ``T=1`` ✓
                #    而潜变量有 T 帧 ✓（初版连 T 一起比 ⇒ 自己把自己拦住了 ✗，自检㉟ 当场红 ✓）。
                #: ⚠️ 比较下标**别用滑窗** ✗✗：``shape[1::2]`` 取的是「第 1、3、5…位」✓，
                #: 对 ``[B,C,T,h,w]`` 得到 ``(C,h,w)`` 里**夹着 T** ✓✗（写注释说"不比 T" ✓、
                #: 代码却在比 T ✓ → 注释对、判据错 ✓）。这里逐位点名 ✓。
                if tuple(encoded.shape[i] for i in (1, 3, 4)) != tuple(latents.shape[i] for i in (1, 3, 4)):
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

    def attach_flux_conditioning(self, vector: Any = None, guidance: float | None = None
                                 ) -> dict[str, Any]:
        """给 **Flux 族**注入外部条件：CLIP-L 的 pooled 向量 ✓ + guidance 标量 ✓。

        ⚠️ 为什么需要这个专门的挂载口：Flux 的条件是**两路 + 一根标量**（T5-XXL 的 4096 宽走
        ``condition`` ✓、CLIP-L 的 768 宽 pooled 走 ``vector`` ✓、蒸馏 guidance 是**输入标量** ✓），
        而这两座文本塔盘上都没有 ✗（只有 ``vae_flux_ae`` ✓）⇒ 只能从外面给。
        不挂也能跑 ✓（零向量 + 默认 guidance ✓，见 :data:`dit_flux.GUIDANCE_DEFAULT` ✓）——
        但那是**结构验证** ✗，不是语义 ✓，所以本方法把「挂没挂」如实记进返回值 ✓。

        返回：``{"vector": …, "guidance": …, "note": …}`` ✓（给 `describe` / 报告用 ✓）。
        """
        self._gate()
        model = getattr(self, "_model", None)
        if model is None:
            raise TorchBackendUnavailable(
                "还没装模型 ⇒ 没有地方挂 Flux 条件 ✗（先 ensure_loaded ✓）", reason="pending")
        if not hasattr(model, "default_vector") or not hasattr(model, "shape"):
            raise TorchBackendUnavailable(
                f"当前装的不是 Flux 族（family={getattr(self, '_ditFamily', None) or '未知'}）⇒ "
                f"没有 pooled / guidance 这两根输管 ✗（这两根是 Flux 的接口事实 ✓）",
                reason="pending")
        torch = self._torch()
        shape = model.shape
        param = next(model.parameters())
        # ⚠️ 回执必须报**这一拍真正生效**的值 ✗✗：挂载是**写进模型属性**的 ✓ ⇒ 第二次只传 vector 时，
        #    上一次挂的 guidance（和 vector）**还在生效** ✗。原先 else 分支一律报 `guidance_default`、
        #    vector 一律报「零向量」⇒ 回执与实际跑的东西对不上（最坏情况：以为退回默认 3.5、实际还在 7.0）。
        live_vec = getattr(model, "default_vector", None)
        note: dict[str, Any] = {
            "vector": ("沿用上次注入的 " + f"{int(live_vec.numel())} 维" if live_vec is not None
                       else "零向量（未注入 ⇒ 全局语义缺席 ✗）"),
            "guidance": None}
        if vector is not None:
            vec = torch.as_tensor(vector, dtype=param.dtype, device=param.device).reshape(-1)
            if vec.numel() < shape.vec_dim:
                raise TorchBackendUnavailable(
                    f"pooled 向量宽 {vec.numel()} ✗ < 权重要求的 {shape.vec_dim} ✓"
                    f"（CLIP-L 的 pooled 就是 {shape.vec_dim} 维 ✓）", reason="pending")
            model.default_vector = vec
            note["vector"] = f"已注入 {int(vec.numel())} 维（取前 {shape.vec_dim} ✓）"
        if guidance is not None:
            model.default_guidance = float(guidance)
            note["guidance"] = float(guidance)
        else:
            live = getattr(model, "default_guidance", None)
            note["guidance"] = float(shape.guidance_default if live is None else live)
            if live is None:
                note["note"] = ("guidance 用默认值 ⇒ **不是**权重事实 ✗（权重里只有 `guidance_in` 投影 ✓，"
                                "数值来自推理配置 ✓）")
            else:
                note["note"] = (f"沿用上次挂的 guidance {float(live):g}（本次没给 ⇒ **不**回落默认 ✗；"
                                f"要回落就显式传 {float(shape.guidance_default):g} 或重装模型）")
        return note

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
        self._textEncoder = (te_mod.build_text_encoder(config).to(self._device).eval()
                             .requires_grad_(False))
        self._textConfig = config
        self._textKind = "reference-untrained"   # ⚠️ 每次挂载**显式写** ✓（不许留上一次的 ✗）
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

    def attach_qwen3_te(self, weights: str, *, tokenizer_path: str | None = None,
                        max_length: int = 512, dtype: Any = None) -> dict[str, Any]:
        """挂**真**文本塔 ✓：Qwen3-4B dense（``qwen_3_4b.safetensors`` ✓ **398** 张量 ✓ —— 别和
        DiT 的 **453** 混 ✓✗，两个数差 55，混了会让"键数对不对"这条自检失去意义 ✓）
        ⇒ :meth:`encode_text` 出的是**有语义的条件** ✓（Z-Image 全链的第一环 ✓）。

        ## 与 :meth:`attach_text_encoder` 的分工（**不许混用** ✗）

        * ``attach_text_encoder``：**参考实现** ✓，权重随机 ✓ ⇒ 只验管道 ✓（报告 ``reference-untrained`` ✓）；
        * 本方法：**真权重** ✓ ⇒ 报告 ``kind="qwen3-te"`` ✓ + 结构逐项 ✓ + 装载键数 ✓。

        ## 五条硬口径（每一条都对着参考实现核过 ✓）

        1. **取倒数第 2 层** ✓（``layer_idx=-2`` ✓）⇒ 0-based 第 34 层**之后**的 hidden ✓；
        2. **不过 final norm** ✗（``layer_norm_hidden_state=False`` ✓）—— ``model.norm`` 建了但**不调** ✓；
        3. **pad id = 151643** ✓（``<|endoftext|>`` ✓，见 :data:`qwen3_te.PAD_TOKEN_ID` ✓）；
        4. **套模板** ✓（``<|im_start|>user\\n{}\\n<|im_end|>\\n<|im_start|>assistant\\n`` ✓）——
           由 :class:`qwen3_te.Qwen3TeRuntime.prompt` 加 ✓，本方法**不再重复拼** ✗；
        5. **词表要装得下特殊符** ✓：``required_vocab_size`` = 151646 ✓ > ``vocab.json`` 的 151643 ✓
           —— 这一条是 ``tokenizer_hub.added_tokens_of`` 才修好的 ✓（漏了 ⇒ ``embed_tokens`` 越界 ✓✗）。

        ⚠️ **设备**：7.49 GiB bf16 ✓ 上 ``self._device`` ✓ —— 与 11.46 GiB 的 Z-Image DiT **同卡装不下** ✓✗
        （22.49 GiB 可用 ✓）⇒ 24 GiB 卡上必须**分时**：先 TE 编码 ✓ → 让 TE 下卡 ✓ → 再装 DiT ✓
        （本方法**不替调用方决定** ✗，但报告里 ``device`` 写清楚 ✓）。
        """
        self._gate()
        from app.services.engine import qwen3_te as qwen3_mod  # noqa: PLC0415 —— 局部引 ✓ 顶层不碰 torch ✓

        path = Path(str(weights))
        if not path.exists():
            raise TorchBackendUnavailable(
                f"文本塔权重不存在 ✗：{path} ✓（Z-Image 要的是 `qwen_3_4b.safetensors` ✓，"
                f"**不是** Qwen2.5-VL / Qwen3-VL ✗ —— 见 z_image.py 的 `Qwen3_4BModel` ✓）",
                reason="pending")
        torch = self._torch()
        # ⚠️ 只读**头部**拿形状 ✓（11.46 GiB 不落到内存 ✓ —— `st.inspect` 只解析 header ✓）。
        info = st.inspect(path)
        shapes = {name: tuple(record.shape) for name, record in info.tensors.items()}
        try:
            shape = qwen3_mod.infer_shape(shapes)
        except qwen3_mod.ShapeError as error:
            raise TorchBackendUnavailable(
                f"这份权重推不出 Qwen3-4B 文本塔 ✗：{error} ✓", reason="pending") from error
        runtime = qwen3_mod.Qwen3TeRuntime(shape=shape, max_length=int(max_length))
        # ⚠️ **先核后改** ✓：`output_dim` 必须与主干的 `text_dim` 一致 ✓（Z-Image 两个数都是 2560 ✓）。
        self._check_text_encoder(candidate=runtime)
        tokenizer = self._te_tokenizer(tokenizer_path, shape)
        #: 跟着 `self._device` 走 ✓（有卡但显式选 CPU 时不能建成 bf16 ✗，见 `_compute_dtype` ✓）
        want = dtype or self._compute_dtype()
        # ⚠️ **只建一次** ✗（`build_from_shapes` 自己用 `set_default_dtype`+`finally` 复原 ✓ ——
        #    在外面再包一层等于建两遍 ✓✗：7.49 GiB 的壳建两份，峰值直接翻倍 ✓✗ 与 `load_weights`
        #    里记的那条"白建一遍"同一个病根 ✓）。
        module = qwen3_mod.build_from_shapes(shapes, layer_idx=shape.layer_idx, dtype=want)
        #: ⭐ **联合驻留**（卡 + CPU ✓）先切再装 ✓✗ —— 与 `load_weights` 那条**同一个顺序** ✓：
        #: 权重落在参数**当前所在**的设备上 ✓ ⇒ 装载期峰值只是"卡上那一刀" ✓；反过来
        #: （先整份上卡再切）峰值仍是整份 ✗✗，那正好是要治的 OOM ✓。
        #: ⚠️ 没有这一处时的真实症状 ✗✗：`enter_phase("text")` 判成 ``split`` ✓（卡上剩 12 GiB、
        #: 这份要 14.79 ✓）、报告也写 `residency=split` ✓，然后这里一句 ``.to("cuda")`` 把 14.79 GiB
        #: **整份**塞进去 ⇒ OOM ✓✗ —— 报告说联合、实际硬塞，比不报策略更坏 ✓✗。
        residency_report = self._place_before_load(module, what="te")
        report_weights = weights_mod.load_module_weights(module, path, dtype=want)
        if not report_weights.complete:
            del module
            self._raise_te_mismatch(report_weights)
        self._textEncoder = module.eval().requires_grad_(False)
        self._textConfig = runtime
        self._textKind = "qwen3-te"
        self._tokenizer = tokenizer
        return self._te_report(
            kind="qwen3-te", runtime=runtime, path=path, want=want,
            residency_report=residency_report, report_weights=report_weights,
            tokenizer=tokenizer, tokenizer_path=tokenizer_path,
            note="**真** Qwen3-4B 文本塔 ✓（倒数第 2 层 hidden ✓、不过 final norm ✓、套模板 ✓）")

    def attach_qwen35_te(self, weights: str, *, tokenizer_path: str | None = None,
                         max_length: int = 512, dtype: Any = None) -> dict[str, Any]:
        """挂**真**文本塔 ✓：Qwen3.5-9B 混合线性注意力（Qwen-Image **2.1** 的 TE ✓）
        ⇒ :meth:`encode_text` 出 **4096** 维的真条件 ✓（2.1 全链的第一环 ✓）。

        ## 与 :meth:`attach_qwen3_te` 的差别（**别互相顶替** ✗）

        | | Qwen3-4B（Z-Image ✓） | Qwen3.5-9B（Qwen-Image 2.1 ✓） |
        |---|---|---|
        | 结构 | 36 层 dense attention ✓ | **32 层混合**：24 层 Gated DeltaNet ＋ 8 层 gated attention ✓ |
        | 取层 | 第 34 层之后 ✓ | 第 **30** 层之后 ✓（``layer_idx=-2`` ✓） |
        | 宽度 | 2560 ✓ | **4096** ✓ |

        两族的 ``layer_idx=-2`` 都是「倒数第 2 层之后」✓，但**层号完全不同** ✗ —— 换族时这一格必须跟着换 ✓。

        ## 五条硬口径（每一条都对着 comfy 的 ``qwen35.py`` 逐位核过 ✓）

        1. **取倒数第 2 层** ✓ ⇒ 0-based 第 30 层**之后**的 hidden ✓；
        2. **不过 final norm** ✗（``model.norm`` 建了但**不调** ✓）—— 与 Qwen3 那条同一个口径 ✓；
        3. **pad id = 248044** ✓、``<|im_start|>=248045`` ✓、``<|im_end|>=248046`` ✓、
           思考段的 ``248068/248069`` ✓（见 :mod:`qwen35_te` 顶部 ✓）；
        4. **套模板** ✓（``user`` 段 ＋ ``assistant`` 段里**先关思考** ✓）—— 由
           :class:`qwen35_te.Qwen35TeRuntime.prompt` 加 ✓，本方法**不再重复拼** ✗；
        5. **词表要装得下特殊符** ✓（词表 248320 ✓ ≫ 248069 ✓）。

        ## 与 comfy 的对拍结论（2026-10-31 实测 ✓）

        真权重 fp32 下，本实现与 ``comfy.text_encoders.qwen35`` 的 ``Qwen35TransformerBlock``
        **逐位相同**（最大绝对差 **0.000e+00** ✓，线性层与完整注意力层各验 ✓，长度 23 / 130 各验 ✓）
        ⇒ :meth:`encode_text` 出的条件与 comfy 出的**是同一批数** ✓。

        ⚠️ **显存**：9B 上 bf16 约 **18.5 GiB** ✓ —— 比 2.1 的 DiT **还大** ✓✗（22.49 GiB 可用 ✓）
        ⇒ 24 GiB 卡上必须**分时**：先 TE 编码 ✓ → 让 TE 下卡 ✓ → 再装 DiT ✓（本方法**不替调用方
        决定** ✗，报告里 ``device`` 写清楚 ✓）。
        ⚠️ **同一文件里还有视觉塔** ✓（549 键 ✓）：``key_map`` 只把 ``model.language_model.*`` 搬到
        ``model.*`` ✓，视觉塔与 ``lm_head`` 于是落进 ``unexpected`` ✓ —— **这是预期的** ✓，
        报告里单列 ``ignoredTowerKeys`` ✓，别让人以为是装错了 ✓✗。
        """
        self._gate()
        from app.services.engine import qwen35_te as qwen35_mod  # noqa: PLC0415 —— 局部引 ✓

        path = Path(str(weights))
        if not path.exists():
            raise TorchBackendUnavailable(
                f"文本塔权重不存在 ✗：{path} ✓（Qwen-Image 2.1 要的是 "
                f"`qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors` ✓，"
                f"**不是** Qwen3-4B / Qwen3-VL ✗ —— 见 qwen_image21.py 的 TE 定义 ✓）",
                reason="pending")
        torch = self._torch()
        info = st.inspect(path)                       # ⚠️ 只读头部 ✓（18.5 GiB 不落内存 ✓）
        shapes = {name: tuple(record.shape) for name, record in info.tensors.items()}
        try:
            shape = qwen35_mod.infer_shape(shapes)
        except qwen35_mod.ShapeError as error:
            raise TorchBackendUnavailable(
                f"这份权重推不出 Qwen3.5-9B 文本塔 ✗：{error} ✓", reason="pending") from error
        runtime = qwen35_mod.Qwen35TeRuntime(shape=shape, max_length=int(max_length))
        # ⚠️ **先核后改** ✓：`output_dim` 必须等于主干的 `text_dim`（2.1 两边都是 4096 ✓）
        self._check_text_encoder(candidate=runtime)
        tokenizer = self._te_tokenizer(tokenizer_path, shape, id_hint="（248045/248046 ✓）")
        #: 跟着 `self._device` 走 ✓（有卡但显式选 CPU 时不能建成 bf16 ✗，见 `_compute_dtype` ✓）
        want = dtype or self._compute_dtype()
        # ⚠️ **只建一次** ✗（`build_from_shapes` 自己用 `set_default_dtype`+`finally` 复原 ✓）
        module = qwen35_mod.build_from_shapes(shapes, layer_idx=shape.layer_idx, dtype=want)
        #: ⭐ **联合驻留**先切再装 ✓✗（与 `load_weights` / `attach_qwen3_te` 同一顺序 ✓）。
        #: ⚠️ 这一件是全仓**最大**的单件 ✓✗（9B ⇒ 18.5 GiB bf16 ✓，比 2.1 的 DiT 还大 ✓）：
        #: 不给它切，则 `enter_phase("text")` 判成 ``split`` 也只是一句 ``.to("cuda")`` ✓✗
        #: ⇒ 报告写联合、实际硬塞 18.5 GiB ⇒ OOM ✓✗。
        #: ⚠️ 能不能切得动取决于 `residency.survey` 认不认这棵树 ✗✗：这条塔的外壳是
        #: ``self.model = nn.Module()`` ✓、真正的层表在 ``model.layers`` ✓ —— 穿透那一层
        #: 是 `residency._expand` 的职责 ✓（不穿透 ⇒ 整份 18.5 GiB 被当成「非层参数」⇒
        #: 静默退化成纯 CPU ✓✗）。
        residency_report = self._place_before_load(module, what="te")
        # ⚠️ 同文件里的视觉塔与 lm_head 在这里**必然**落进 unexpected ✓（不是装错 ✓ 见 docstring）
        report_weights = weights_mod.load_module_weights(
            module, path, dtype=want, key_map={r"^model\.language_model\.": "model."})
        if not report_weights.complete:
            del module
            self._raise_te_mismatch(report_weights)
        ignored = [name for name in report_weights.unexpected
                   if name.startswith(("model.visual.", "lm_head.", "model.lm_head."))]
        self._textEncoder = module.eval().requires_grad_(False)
        self._textConfig = runtime
        self._textKind = "qwen35-te"
        self._tokenizer = tokenizer
        return self._te_report(
            kind="qwen35-te", runtime=runtime, path=path, want=want,
            residency_report=residency_report, report_weights=report_weights,
            tokenizer=tokenizer, tokenizer_path=tokenizer_path,
            #: 这两格是 Qwen3.5 **独有**的 ✓：它那份文件里带视觉塔与 lm_head ✓（有意不装 ✓）。
            extra={"unexpectedCount": len(report_weights.unexpected),
                   "ignoredTowerKeys": len(ignored)},
            note=("**真** Qwen3.5-9B 混合线性注意力文本塔 ✓（第 30 层之后 hidden ✓、"
                  "不过 final norm ✓、套模板 ✓）；与 comfy 逐位对拍 max abs diff = 0 ✓；"
                  "unexpected 里的视觉塔/lm_head 是**有意不装** ✓"))

    def attach_qwen25_te(self, weights: str, *, tokenizer_path: str | None = None,
                         max_length: int = 512, dtype: Any = None) -> dict[str, Any]:
        """挂**真**文本塔 ✓：Qwen2.5-VL-7B 的**语言部分**（Qwen-Image **2512 / Edit 2511** 的 TE ✓）
        ⇒ :meth:`encode_text` 出 **3584** 维的真条件 ✓（Qwen-Image 全链的第一环 ✓）。

        ## 与另两座塔的差别（**三座并列，谁也顶不了谁** ✗）

        | | Qwen3-4B（Z-Image ✓） | Qwen3.5-9B（2.1 ✓） | **Qwen2.5-VL-7B（2512/Edit ✓）** |
        |---|---|---|---|
        | 层数 / 宽度 | 36 / 2560 ✓ | 32 混合 / 4096 ✓ | **28 / 3584** ✓ |
        | 取层 | 倒数第 2 层 ✓ | 倒数第 2 层 ✓ | **最后一层**（``-1`` ✓） |
        | q/k/v bias | 无 ✗ | 无 ✗ | **有**（三根 ✓） |
        | QK-norm | 有 ✓ | 有 ✓ | **无** ✗ |
        | 模板 | ``user`` 单段 ✓ | ``user`` ＋ 关思考 ✓ | **``system`` ＋ ``user``** 两段 ✓ |

        ## 五条硬口径（每条都对着 ``comfy/text_encoders/qwen_image.py`` 核过 ✓）

        1. **取最后一层** ✓（``layer_idx=-1`` ✓）⇒ 第 27 层**之后**的 hidden ✓；
        2. **不过 final norm** ✗（``layer_norm_hidden_state=False`` ✓ —— ``model.norm`` 建了不调 ✓）；
        3. **切掉 system 段** ✓（``trim_start``：找到**第 2 个** ``<|im_start|>`` ✓、连 ``user``+``\\n``
           一起丢 ✓）—— 这是在**模块前向里**做的 ✓（不是分词时 ✗）：位置编码必须算在**完整序列**上 ✓✗，
           先切后编码会让位置整体前移 ✓（不报错、语义偏移 ✓）；
        4. **q/k/v 带 bias** ✓（``qkv_bias=True`` ✓ 由 ``llm.LlmConfig`` 一路传到 ``GQAAttention`` ✓）——
           漏掉不会报错 ✗，只是每层少一个常数偏置、提示词跟随明显变弱 ✓✗；
        5. **分词器要装得下特殊符** ✓（``<|im_start|>=151644`` ✓ / vocab 152064 ✓）。

        ⚠️ **显存**：7B 上 bf16 约 **14.79 GiB** ✓，2512 的 DiT（~20B ⇒ bf16 约 40 GiB ✗✗）
        **同卡绝对装不下** ✓ ⇒ 只能分时（编码完让 TE 下卡再装主干 ✓，与 2.1 那条同一套
        ``enter_phase`` ✓）。本方法**不替调用方决定** ✗，报告里 ``device``/``residency`` 写清楚 ✓。

        ⚠️ 同一份文件里**还有视觉塔**（``visual.*`` 714 张量 ✓）与 ``lm_head`` ✓ ⇒ 它们必然落进
        ``unexpected`` ✓，这是**预期的** ✓（本方法只挂语言塔 ✗）；报告单列 ``ignoredTowerKeys`` ✓。
        """
        self._gate()
        from app.services.engine import qwen25_te as qwen25_mod  # noqa: PLC0415 —— 局部引 ✓

        path = Path(str(weights))
        if not path.exists():
            raise TorchBackendUnavailable(
                f"文本塔权重不存在 ✗：{path} ✓（Qwen-Image 2512 / Edit 要的是 "
                f"`qwen_2.5_vl_7b_fp8_scaled.safetensors` ✓，**不是** Qwen3-4B / Qwen3.5 ✗ —— "
                f"见 qwen_image.py 的 `Qwen25_7BVLIModel` ✓）",
                reason="pending")
        torch = self._torch()
        info = st.inspect(path)                       # ⚠️ 只读头部 ✓（13 GiB 不落内存 ✓）
        shapes = {name: tuple(record.shape) for name, record in info.tensors.items()}
        try:
            shape = qwen25_mod.infer_shape(shapes)
        except qwen25_mod.ShapeError as error:
            raise TorchBackendUnavailable(
                f"这份权重推不出 Qwen2.5-VL-7B 文本塔 ✗：{error} ✓", reason="pending") from error
        runtime = qwen25_mod.Qwen25TeRuntime(shape=shape, max_length=int(max_length))
        # ⚠️ **先核后改** ✓：`output_dim` 必须等于主干的 `text_dim`（2512 两边都是 3584 ✓）
        self._check_text_encoder(candidate=runtime)
        tokenizer = self._te_tokenizer(tokenizer_path, shape,
                                       id_hint="（151644/151648 ✓）")
        #: 跟着 `self._device` 走 ✓（有卡但显式选 CPU 时不能建成 bf16 ✗，见 `_compute_dtype` ✓）
        want = dtype or self._compute_dtype()
        # ⚠️ **只建一次** ✗（`build_from_shapes` 自己用 `set_default_dtype`+`finally` 复原 ✓）
        module = qwen25_mod.build_from_shapes(shapes, layer_idx=shape.layer_idx, dtype=want)
        #: ⭐ **联合驻留**先切再装 ✓✗（与 `load_weights` / 另两座塔同一顺序 ✓）：
        #: 先整份上卡再切 ⇒ 装载期峰值仍是整份 14.79 GiB ✓✗，那正是要治的 OOM ✓。
        residency_report = self._place_before_load(module, what="te")
        report_weights = weights_mod.load_module_weights(module, path, dtype=want)
        if not report_weights.complete:
            del module
            self._raise_te_mismatch(report_weights)
        ignored = [name for name in report_weights.unexpected
                   if name.startswith(("visual.", "lm_head.", "model.visual."))]
        self._textEncoder = module.eval().requires_grad_(False)
        self._textConfig = runtime
        self._textKind = "qwen25-te"
        self._tokenizer = tokenizer
        return self._te_report(
            kind="qwen25-te", runtime=runtime, path=path, want=want,
            residency_report=residency_report, report_weights=report_weights,
            tokenizer=tokenizer, tokenizer_path=tokenizer_path,
            extra={"unexpectedCount": len(report_weights.unexpected),
                   "ignoredTowerKeys": len(ignored)},
            note=("**真** Qwen2.5-VL-7B 语言塔 ✓（**最后一层** hidden ✓、不过 final norm ✓、"
                  "切掉 system 段 ✓、q/k/v 带 bias ✓）；unexpected 里的视觉塔/lm_head 是"
                  "**有意不装** ✓"))

    def attach_umt5_te(self, weights: str, *, tokenizer_path: str | None = None,
                       max_length: int = 512, dtype: Any = None) -> dict[str, Any]:
        """挂 **UMT5-XXL 编码器** ✓（Wan2.1/2.2 的文本塔 ✓）⇒ :meth:`encode_text` 出 4096 维条件 ✓。

        ## 与另外三座塔的差别（**别互相顶替** ✗）

        | | Qwen3-4B | Qwen3.5-9B | Qwen2.5-VL-7B | **UMT5-XXL** |
        |---|---|---|---|---|
        | 结构 | 36 层 decoder | 32 层混合 | 28 层 decoder | **24 层纯 encoder** ✓ |
        | 宽度 | 2560 | 4096 | 3584 | **4096** ✓ |
        | 取层 | 倒数第 2 层 | 倒数第 2 层 | **最后一层** | **最后一层的 final norm 之后** ✓ |
        | 模板 | 套 chat 模板 | 套模板＋关思考 | 切 system 段 | **不套模板**（裸文本 ＋ ``</s>``）✓ |

        ## 四条硬口径（对着 comfy 的 ``wan`` 文本通路核过 ✓）

        1. ``shared.weight`` 是与输出层**共享**的那份嵌入 ✓（文件里只有一份 ✓）；
        2. **相对位置偏置**必须按桶算 ✓（``relative_attention_bias`` ✓，32 桶 ✓）——
           :class:`umt5_te.Umt5Encoder` 已实现 ✓，这里只负责接线 ✓；
        3. **padding 侧必须置 0** ✓（``zeroOutMasked`` ✓）：Wan 的 cross-attn **不吃 mask** ✗ ⇒
           不置 0 就是拿 padding 行的**真数值**去做注意力 ✓✗（形状全对、出片发灰 ✓）；
        4. ``max_length`` **512** ✓（官方 Wan 模板口径 ✓；不是 Qwen 那套 1024/2048 ✓）。

        ## 显存

        UMT5-XXL 编码器 bf16 约 **10.58 GiB** ✓（实测：242 个 ``.weight`` ＋ ``spiece_model`` ✓）。
        与 Wan2.2 I2V 的 DiT（26.62 GiB ✓）**同时驻留装不进 22.49 GiB** ✗ ⇒ 必须分时 ✓
        （`:meth:`declare_chain` ＋ :meth:`enter_phase` ✓，由 ``runtime`` 那条装配线负责 ✓）。

        ⚠️ 分词器**优先用文件内嵌的** ``spiece_model`` ✓（4.5 MB 单张量 ✓）：Wan 的
        ``umt5_xxl_fp8_e4m3fn_scaled.safetensors`` 与 comfy 的 ``t5_pile_tokenizer`` 是同一份词表 ✓，
        但内嵌那份**跟着权重走** ✓ ⇒ 换文件不会静默换词表 ✓。``tokenizer_path`` 只作回落 ✓。
        """
        self._gate()
        from app.services.engine import umt5_te as umt5_mod  # noqa: PLC0415 —— 局部引 ✓
        from app.services.engine import umt5_tokenizer as umt5_tok_mod  # noqa: PLC0415

        path = Path(str(weights))
        if not path.exists():
            raise TorchBackendUnavailable(
                f"文本塔权重不存在 ✗：{path} ✓（Wan 2.1/2.2 要的是 "
                f"`umt5_xxl_fp8_e4m3fn_scaled.safetensors` ✓，**不是** Qwen/CLIP ✗）",
                reason="pending")
        info = st.inspect(path)
        shapes = {name: tuple(record.shape) for name, record in info.tensors.items()}
        try:
            shape = umt5_mod.infer_shape(shapes)
        except umt5_mod.ShapeError as error:
            raise TorchBackendUnavailable(
                f"这份权重推不出 UMT5 编码器 ✗：{error} ✓", reason="pending") from error
        runtime = umt5_mod.Umt5TeRuntime(shape=shape, max_length=int(max_length))
        #: **先核后改** ✓：``output_dim`` 必须等于主干 ``text_dim``（Wan 主干 4096 ✓）
        self._check_text_encoder(candidate=runtime)
        #: 词表：**内嵌 spiece 优先** ✓（跟着权重走 ✓），读不到才用 ``tokenizer_path`` ✓
        spiece: bytes | None = None
        if "spiece_model" in info.tensors:
            try:
                spiece = st.read_tensor_bytes(path, "spiece_model")
            except Exception as error:  # noqa: BLE001 —— 读不到就走回落 ✓，但要说出来 ✓
                spiece = None
                self._umt5SpieceNote = f"内嵌 spiece 读不出（{type(error).__name__}）⇒ 回落外部词表"
        tokenizer = umt5_tok_mod.load_tokenizer(
            spiece_bytes=spiece, path=tokenizer_path, max_length=int(max_length))
        want = dtype or self._compute_dtype()
        module = umt5_mod.build_from_shapes(shapes, dtype=want)
        residency_report = self._place_before_load(module, what="te")
        report_weights = weights_mod.load_module_weights(module, path, dtype=want)
        if not report_weights.complete:
            del module
            self._raise_te_mismatch(report_weights)
        self._textEncoder = module.eval().requires_grad_(False)
        self._textConfig = runtime
        self._textKind = "umt5-te"
        self._tokenizer = tokenizer
        return self._te_report(
            kind="umt5-te", runtime=runtime, path=path, want=want,
            residency_report=residency_report, report_weights=report_weights,
            tokenizer=tokenizer, tokenizer_path=tokenizer_path or f"{path}#spiece_model",
            extra={"unexpectedCount": len(report_weights.unexpected),
                   "spieceEmbedded": bool(spiece)},
            note=("**真** UMT5-XXL **编码器** ✓（24 层 ✓、最后一层 final norm 之后 ✓、"
                  "不套模板 ✓、padding 位置置 0 ✓）；词表用文件内嵌 ``spiece_model`` ✓"))

    def attach_qwen25vl_vision(self, weights: str, *, dtype: Any = None) -> dict[str, Any]:
        """挂 **Qwen2.5-VL 的视觉塔** ✓：Edit 2511 参考图的**语义**通路 ✓。

        ## 它解决什么 ✗

        Edit 的模板里有 ``<|vision_start|><|image_pad|><|vision_end|>`` ✓ —— 那个
        ``<|image_pad|>`` 是**真 token**（id 151655 ✓）⇒ 不填就退化成词表嵌入 ✓✗：
        模型装得上、跑得动、**Edit 能力一点没有** ✓（本仓最贵的一类静默错 ✗）。
        填进去的正是本方法挂起来的塔 ✓ 的输出（``(grid_h/2 × grid_w/2, 3584)`` ✓）。

        ## 三条硬口径

        1. **从同一份 TE 文件里取 ``visual.*``** ✓：Qwen2.5-VL 的语言塔与视觉塔在**一个文件**里 ✓
           （``qwen_2.5_vl_7b_fp8_scaled.safetensors`` ✓）⇒ 不需要第二个路径参数 ✓，
           但 **必须** `key_map` 剥掉 ``visual.`` ✓（不剥 ⇒ 390 个键全 unexpected ⇒ 一个都没装上 ✓✗）；
        2. **不参与联合驻留** ✗：塔本体 1.35 GiB ✓，而 ``_residencyHandle`` 是**单槽** ✓✗
           —— 给它切一刀会把**文本塔（14.79 GiB）**的 handle 顶掉 ✓✗（那才是该切的那件 ✓）
           ⇒ 这里整份 ``.to(device)`` ✓，报告里 ``residency=None`` ✓ 如实说「没切」✓；
        3. **族名必须当场核** ✓：盘上文件里 ``visual.`` 前缀同时属于 2.5-VL 与 3-VL 两族 ✗，
           尺子差 patch 14 vs 16 ✓ ⇒ 形状推不出来就拒 ✓，不猜 ✓。
        """
        self._gate()
        from app.services.engine import qwen25vl_vision as vl_mod  # noqa: PLC0415 —— 局部引 ✓

        path = Path(str(weights))
        if not path.exists():
            raise TorchBackendUnavailable(
                f"视觉塔权重不存在 ✗：{path} ✓（Edit 2511 的视觉塔就在 **Qwen2.5-VL-7B** 那份文件里 ✓，"
                f"前缀 ``visual.*`` ✓ —— 就是 `attach_qwen25_te` 用的那一份 ✓）", reason="pending")
        info = st.inspect(path)                        # ⚠️ 只读头部 ✓
        shapes = {name: tuple(record.shape) for name, record in info.tensors.items()
                  if name.startswith("visual.")}
        if not shapes:
            raise TorchBackendUnavailable(
                f"这份文件里一个 ``visual.*`` 都没有 ✗：{path.name} ✓ ⇒ 不是 Qwen2.5-VL 的全量文件 ✓"
                f"（截断过的纯语言塔 / 别的族 ✓）。缺视觉塔 ⇒ ``<|image_pad|>`` 那一槽没有东西可填 ✓✗",
                reason="pending")
        try:
            shape = vl_mod.infer_shape(shapes)
        except Exception as error:                    # noqa: BLE001 —— 族的自述就是 ValueError ✓
            raise TorchBackendUnavailable(
                f"视觉塔结构推不出来 ✗：{error} ✓（2.5-VL 与 3-VL 都有 ``visual.*`` ✓，"
                f"尺子差 patch 14 vs 16 ⇒ 本方法只认 2.5-VL ✓）", reason="pending") from error
        problems = vl_mod.shape_problems(shape)
        if problems:
            raise TorchBackendUnavailable(
                "视觉塔形状自相矛盾 ✗：" + "；".join(problems[:3]) + " ✓", reason="pending")
        torch = self._torch()
        want = dtype or self._compute_dtype()
        module = vl_mod.build_vision_tower(shape, dtype=want, device="meta")
        filled = _materialize_meta(module, torch_mod=torch, device="cpu")
        module = module.to(self._device)
        report_weights = weights_mod.load_module_weights(
            module, path, dtype=want, key_map={"^visual\\.": ""})
        if not report_weights.complete:
            del module
            self._trim_cuda()
            raise TorchBackendUnavailable(
                f"视觉塔权重没装全 ✗：缺 {len(report_weights.missing)} 个 / 形状不符 "
                f"{len(report_weights.shapeMismatch)} 个 ✓（首个："
                f"{(report_weights.missing or report_weights.shapeMismatch or ['—'])[0]} ✓）",
                reason="broken")
        self._visionTower = module.eval().requires_grad_(False)
        self._visionShape = shape
        report = {
            "kind": "qwen25vl-vision", "path": str(path),
            "device": str(self._device), "dtype": str(want),
            "layers": int(shape["depth"]), "hidden": int(shape["hidden_size"]),
            "heads": int(shape["num_heads"]), "patchSize": int(shape["patch_size"]),
            "mergeSize": int(shape["spatial_merge_size"]),
            "windowSize": int(shape["window_size"]),
            "outHidden": int(shape["out_hidden_size"]),
            "fullAttBlocks": list(shape["fullatt_block_indexes"]),
            "loadedKeys": int(report_weights.loadedKeys),
            "consumedScales": int(report_weights.consumedScales),
            "materialized": int(filled),
            "residency": None,
            "residencyNote": ("塔本体 1.35 GiB ⇒ 不切 ✓：``_residencyHandle`` 单槽 ✓✗，"
                              "切它会把文本塔（14.79 GiB）的 handle 顶掉 ✓"),
            "note": ("真 Qwen2.5-VL 视觉塔 ✓；输出 ``(n, 3584)`` 由 "
                     "`Qwen25TextTowerModule.forward(image_embeds=…)` 按 ``<|image_pad|>`` 位置整段替换 ✓"),
        }
        self._visionReport = report
        #: 账本按名字登记 ✓（`release_part("vision")` / `_pinned_gib("vision")` 都读这一条 ✓；
        #: 这一族**不切** ✓ ⇒ device 恒为 ``self._device`` ✓，账上不会被当成"可搬"的件 ✓）。
        self._ledger.reserve(
            "vision",
            bytes_=int(sum(item.numel() * item.element_size()
                           for item in self._visionTower.parameters())),
            device=str(self._device),
            note="Edit 2511 参考图视觉塔（只在 encode 段在场 ✓）")
        self._trim_cuda()
        return report

    # ── Edit 2511 的参考图：**两条路都得走** ✓（2026-11-01 接上 ✓）────────────────────
    #
    #: 参考图给**视觉塔**看的面积 ✓✗ —— 两路**不是同一个数** ✗✗（逐字照 `nodes_qwen.py` ✓）：
    #:
    #: | 节点 | 视觉塔（语义 ✓）| VAE（像素 ✓）| 一张图约 |
    #: |---|---|---|---|
    #: | ``TextEncodeQwenImageEditPlus`` ✓（2509/2511 走它 ✓）| **384²** ✓ | 1024² ✓ | **196** token |
    #: | ``TextEncodeQwenImageEdit``（单图老节点 ✓）| 1024² ✓ | 1024² ✓ | 1338 token |
    #:
    #: ⚠️ 本仓默认按 **Plus** ✓；要换老节点口径就改这一个常量 ✓ —— 但**不许「两种都试、挑看着顺眼的」** ✗✗：
    #: 换的是「模型能看到多少细节」✓，差的是模型行为、不是 bug ✓。
    #:
    #: ⭐ **多图**（2026-11-01 接上 ✓）：Plus 那个节点有 **3 个**输入口 ✓，本仓**不设上限** ✓ ——
    #: 一张图一个``(视觉块 + ref_latent)`` ✓，真正的边界是文本塔的 ``max_length`` ✓
    #: （16:9 的图 384² 面积约 **196** token ✓，三张就 588 ✓ ⇒ 512 的默认上限会被截断 ✓✗）。
    EDIT_VISION_AREA = 384 * 384
    #: 参考图给 **VAE** 编码的面积 ✓（两个节点都 1024² ✓，且边长必须是 **8 的倍数** ✓）。
    EDIT_VAE_AREA = 1024 * 1024
    #: 老节点 ``TextEncodeQwenImageEdit`` 的视觉口径 ✓（备查 ✓，默认不生效 ✓）。
    EDIT_VISION_AREA_CLASSIC = 1024 * 1024

    @staticmethod
    def area_size(width: int, height: int, area: int, *, multiple: int = 1) -> tuple[int, int]:
        """``(宽,高)`` **等比**缩到「面积≈``area``」✓，再按 ``multiple`` 取整 ✓（下限一个 ``multiple`` ✓）。

        ⚠️ 是**面积**不是边长 ✓✗：``1024*1024`` 对 16:9 的图是 1344×756 ✓，不是 1024×576 ✓。
        """
        width, height = max(int(width), 1), max(int(height), 1)
        step = max(int(multiple), 1)
        scale = math.sqrt(float(area) / float(width * height))
        out_w = max(step, int(round(width * scale / step)) * step)
        out_h = max(step, int(round(height * scale / step)) * step)
        return out_w, out_h

    def _pixels_of_area(self, image: str, area: int, *, multiple: int = 1,
                        value_range: str = "0..1") -> tuple[Any, dict[str, Any]]:
        """参考图 ⇒ 指定面积下的像素张量 ✓ ``(1, 3, 1, H, W)`` ✓。

        ⚠️ 先读**原图尺寸**再算目标宽高 ✓（`load_image_tensor` **按给定宽高拉伸** ✗ ⇒
        直接把 1024²当宽高喂进去，构图会被**纵向压扁** ✓✗ —— 不报错，只是越看越不对 ✓）。
        """
        path = Path(str(image))
        if not path.is_file():
            raise TorchBackendUnavailable(f"参考图不存在 ✗：{path}", reason="input")
        from PIL import Image  # noqa: PLC0415  —— 只为读原图尺寸 ✓
        with Image.open(path) as handle:
            native_w, native_h = handle.size
        out_w, out_h = self.area_size(native_w, native_h, area, multiple=multiple)
        pixels, info = media_mod.load_image_tensor(
            str(path), width=out_w, height=out_h, value_range=value_range)
        return pixels, {**info, "native": [int(native_w), int(native_h)],
                        "area": int(area), "scaled": [int(out_w), int(out_h)]}

    def reference_embeds(self, image: str, *, area: int | None = None) -> dict[str, Any]:
        """参考图 ⇒ **视觉塔嵌入** ✓ ``(n, out_hidden)`` ✓（Edit 2511 的**语义**那一路 ✓）。

        ``n`` 是**合并后**的 token 数 ✓（= 网格格数 ÷ ``spatial_merge_size``² ✓）——
        384² 面积约 **196** ✓、1024² 约 1338 ✓；**别拿面积除 28² 手算** ✗（取整会漂 ✓，
        一律读返回值里的 ``tokens`` ✓）。
        """
        self._gate()
        if self._visionTower is None or self._visionShape is None:
            raise TorchBackendUnavailable(
                "没挂 Qwen2.5-VL **视觉塔** ✗ ⇒ 参考图看不懂 ✓（先 `attach_qwen25vl_vision(路径)` ✓）",
                reason="attach")
        from app.services.engine import qwen25vl_vision as vl_mod  # noqa: PLC0415
        torch = self._torch()
        target = int(area or self.EDIT_VISION_AREA)
        pixels, info = self._pixels_of_area(image, target, multiple=1, value_range="0..1")
        #: ⚠️ 视觉塔那一路吃的是 **``(H, W, C)`` 三维** ✓✗ —— `load_image_tensor` 统一给
        #: ``(1, 3, 1, H, W)`` ✓（那是 VAE 的口 ✓）⇒ 这里必须**挤成单图** ✓，
        #: 直接喂 5 维会在 `process_image_with` 里报「必须是三维」（至少是报错 ✓，不是静默 ✗）。
        frame = pixels[0, :, 0].permute(1, 2, 0).contiguous()      # (H, W, 3) ✓
        #: ⚠️ 网格按 **patch 14 × merge 2 = 28** 取整 ✓（`process_image_with` 内部 ✓）——
        #: 于是 384² 面积实际走 **392²** 之类 ✓ ⇒ token 数是取整后的 ✓。
        patches, grid = vl_mod.patches_of(frame, self._visionShape)
        grid_thw = torch.tensor([list(grid)], dtype=torch.long, device=self._device)
        with torch.no_grad():
            embeds = self._visionTower(patches.to(self._device), grid_thw)
        embeds = embeds.reshape(-1, int(embeds.shape[-1]))
        merge = int(self._visionShape["spatial_merge_size"])
        report = {
            "path": str(image), "area": target,
            "native": info["native"], "scaled": info["scaled"],
            "grid": [int(grid[1]), int(grid[2])],
            "gridAfterMerge": [int(grid[1]) // merge, int(grid[2]) // merge],
            "tokens": int(embeds.shape[0]), "hidden": int(embeds.shape[1]),
            "note": ("视觉塔 **合并后**的 token ✓（每 ``merge²`` 个 patch 合成 1 个 ✓）；"
                     "按 ``<|image_pad|>`` 位置**整段替换**进文本序列 ✓"),
        }
        return {"embeds": embeds, "report": report}

    def reference_latents(self, image: str, *, area: int | None = None) -> dict[str, Any]:
        """参考图 ⇒ **VAE 参考潜变量** ✓ ``(1, C, h, w)`` ✓（Edit 2511 的**像素**那一路 ✓）。

        ⚠️ 三件容易搞错的地方 ✓✗：

        * 面积是 **1024²**（不是视觉塔那个 384² ✗✗），边长还要是 **8 的倍数** ✓；
        * 归一化走 :meth:`_encode_pixels` ✓（裸调 ``self._vae.encode`` 拿到的是**未归一化**潜变量 ✓✗，
          症状是"整张图发灰/偏色"，且**不报错** ✓）；
        * Wan 族编码出 **5 维** ``(1, C, 1, h, w)`` ✓ ⇒ 参考图是**单帧** ⇒ 必须挤成 **4 维** ✓✗
          （``QwenImageDiT.forward`` 的 ``ref_latents`` 只吃 4 维 ✓，5 维会在 patchify 上炸 ✓）。
        """
        self._gate()
        if self._vae is None:
            raise TorchBackendUnavailable(
                "没挂 **VAE** ✗ ⇒ 参考图潜变量无从谈起 ✓"
                "（`attach_wan_vae(…, family=…)` ✓ 或走装配链的 encode 段 ✓）", reason="attach")
        torch = self._torch()
        target = int(area or self.EDIT_VAE_AREA)
        pixels, info = self._pixels_of_area(image, target, multiple=8, value_range="-1..1")
        with torch.no_grad():
            latent = self._encode_pixels(pixels.to(device=self._device))
        if latent.ndim == 5:
            if int(latent.shape[2]) != 1:
                raise TorchBackendUnavailable(
                    f"参考图编码出 {int(latent.shape[2])} 个潜帧 ✗（参考图是**单帧** ✓）", reason="pending")
            latent = latent[:, :, 0]
        if latent.ndim != 4:
            raise TorchBackendUnavailable(
                f"参考图潜变量是 {int(latent.ndim)} 维 ✗（`ref_latents` 只吃 4 维 ✓）", reason="pending")
        report = {
            "path": str(image), "area": target,
            "native": info["native"], "scaled": info["scaled"],
            "shape": [int(v) for v in latent.shape],
            "channels": int(latent.shape[1]),
            "grid": [int(latent.shape[2]), int(latent.shape[3])],
            "note": ("参考图潜变量 ✓：进 DiT 的 ``ref_latents`` ✓，**每一步采样都要带** ✓"
                     "（漏带 = 编辑能力凭空消失、图与参考图无关且不报错 ✓✗）"),
        }
        return {"latents": latent, "report": report}

    def prepare_edit_reference(self, image: str, *, vision_area: int | None = None,
                               vae_area: int | None = None) -> dict[str, Any]:
        """**一张**参考图 ⇒ Edit 2511 要的**两样** ✓：视觉嵌入 ✓ ＋ VAE 参考潜变量 ✓（缺一不可 ✓）。"""
        embeds = self.reference_embeds(image, area=vision_area)
        latents = self.reference_latents(image, area=vae_area)
        return {
            "embeds": embeds["embeds"],
            "latents": latents["latents"],
            "report": {
                "image": str(image),
                "vision": embeds["report"],
                "vae": latents["report"],
                "note": ("两路口径分开 ✓：视觉塔 384² 面积（token 数）✓、"
                         "VAE 1024² 面积（像素）✓ —— 混用不报错，只让整图偏色/细节掉 ✓✗"),
            },
        }

    def prepare_edit_references(self, images: Sequence[str], *, vision_area: int | None = None,
                                vae_area: int | None = None) -> dict[str, Any]:
        """**N 张**参考图 ⇒ ``embeds`` / ``latents`` **各 N 份** ✓（Edit 2511 多图 ✓）。

        ⚠️ **顺序就是语义** ✗✗：提示词里 ``Picture 1: …`` 与 ``embeds[0]`` / ``latents[0]``
        必须指**同一张**图 ✓。DiT 把 ``ref_latents`` **按列表顺序**接到序列头 ✓、
        文本侧按 ``<|image_pad|>`` 的**位置顺序**整段换 ✓ —— 两条顺序一致才对得上 ✓；
        乱序**不报错** ✓，只让「把第一张的头换掉」作用到第二张上 ✓✗。

        ⚠️ 每张图的两路口径**各自独立** ✓（不同尺寸的参考图 token 数不同 ✓）：视觉塔 384² 面积 ✓、
        VAE 1024² 面积且边长 8 的倍数 ✓（见 :meth:`prepare_edit_reference` ✓）。

        ⚠️ 本仓**不设**张数上限 ✓（参考节点给了 3 个输入口 ✓，模型侧只是序列变长 ✓）；
        真正的边界是文本塔的 ``max_length`` ✓（一张 384² 图约 196 token ✓，越界由分词器拦 ✓）。
        """
        paths = [str(item) for item in (images or ())]
        if not paths:
            raise TorchBackendUnavailable(
                "参考图列表是空的 ✗ ⇒ Edit 2511 至少要一张 ✓（不给图就是纯 t2i，请关掉编辑口径 ✓）",
                reason="input")
        pieces = [self.prepare_edit_reference(path, vision_area=vision_area, vae_area=vae_area)
                  for path in paths]
        visions = [piece["report"]["vision"] for piece in pieces]
        vaes = [piece["report"]["vae"] for piece in pieces]
        return {
            "count": len(pieces),
            "embeds": [piece["embeds"] for piece in pieces],
            "latents": [piece["latents"] for piece in pieces],
            "report": {
                "images": len(pieces),
                "paths": paths,
                "vision": {
                    "tokens": sum(int(item["tokens"]) for item in visions),
                    "perImage": visions,
                },
                "vae": {
                    "channels": int(vaes[0]["channels"]),
                    "shapes": [item["shape"] for item in vaes],
                    "perImage": vaes,
                },
                "note": ("多图 ✓：第 i 张 ⇒ 模板第 i 个视觉块 ✓ ＋ ``ref_latents[i]`` ✓ —— "
                         "两路**同序**才对得上 ✓（乱序不报错，只让编辑作用错图 ✓✗）；"
                         "``Picture i: `` 那 8 个 token 由 :func:`qwen25_te.edit_prefix` 拼 ✓"),
            },
        }

    def attach_krea2_te(self, weights: str, *, tokenizer_path: str | None = None,
                        max_length: int = 512, dtype: Any = None,
                        taps: tuple[int, ...] | None = None) -> dict[str, Any]:
        """挂 **Krea2 的文本塔** ✓：Qwen3-VL-4B 语言塔 ＋ **12 层栈条件** ✓。

        ## 与另四座塔的差别（**这是第五条路** ✗ 谁都不顶替谁 ✓）

        | | Qwen3-4B | Qwen3.5-9B | Qwen2.5-VL-7B | Qwen3-VL-32B | **Qwen3-VL-4B（Krea2）** |
        |---|---|---|---|---|---|
        | 层/宽 | 36/2560 | 32/4096 | 28/3584 | 64/5120 | **36/2560** |
        | 取层 | 倒数第 2 | 第 30 层后 | 最后一层 | 最后一层 | **12 层栈（2,5,…,35）** |
        | 输出宽 | 2560 | 4096 | 3584 | 5120 | **12 × 2560 = 30720** |

        ## 三条硬口径

        1. **取 12 层、不取最后一层** ✗✗：主干吃 ``txtlayers × txtdim`` 的**栈** ✓
           （``Krea2DiT._unpack`` ⇒ ``reshape(b, seq, 12, 2560)`` ✓）—— 只送最后一层 ⇒
           宽度 2560 ≠ 30720 ⇒ 当场形状错 ✓（这条会报 ✓）；
        2. **裁掉模板前缀** ✓：条件从 ``<|im_start|>user\\n`` **之后**起算 ✓（参考逐字 ✓）——
           裁剪在**模块前向里**做 ✓（必须算完位置编码再裁 ✓，先裁后编码 ⇒ 位置整体前移 ✓✗，
           不报错、只是画面偏移 ✓）；
        3. **不过 final norm** ✗（盘上截断模型根本没有 ``model.norm`` ✓，建了就永远缺一个键 ✓）。

        ⚠️ 盘上三份文件里另两份（``Qwen3_VL_4B/*`` 的 int8 全量）带 ``model.language_model.``
        前缀与视觉塔 ✓ ⇒ 这里统一用 ``key_map`` 搬到 ``model.`` ✓、视觉塔落 ``unexpected`` ✓（预期 ✓）。
        """
        self._gate()
        from app.services.engine import qwen3vl_te as vl_mod  # noqa: PLC0415 —— 局部引 ✓

        path = Path(str(weights))
        if not path.exists():
            raise TorchBackendUnavailable(
                f"Krea2 文本塔权重不存在 ✗：{path} ✓（Krea2 要的是 **Qwen3-VL-4B** ✓ 如 "
                f"`Krea2-Engineer-V1-int8-convrot.safetensors` ✓；**不是** Qwen3-4B / 3.5 / 2.5-VL ✗）",
                reason="pending")
        torch = self._torch()
        info = st.inspect(path)                       # ⚠️ 只读头部 ✓
        raw = {name: tuple(record.shape) for name, record in info.tensors.items()}
        #: ⚠️ 两套前缀都得认 ✓：截断版（Krea2-Engineer ✓）直接就是 ``model.layers.N`` ✓；
        #: 全量版写成 ``model.language_model.layers.N`` ✓ ⇒ 搬到 ``model.`` ✓（视觉塔随之落 unexpected ✓）。
        prefixed = any(name.startswith("model.language_model.") for name in raw)
        key_map = {r"^model\.language_model\.": "model."} if prefixed else None
        shapes = ({name.replace("model.language_model.", "model.", 1): value
                   for name, value in raw.items() if name.startswith("model.language_model.")}
                  if prefixed else raw)
        try:
            shape = vl_mod.infer_shape(shapes)
        except ValueError as error:
            raise TorchBackendUnavailable(
                f"这份权重推不出 Qwen3-VL-4B 语言塔 ✗：{error} ✓", reason="pending") from error
        #: ⚠️ ``infer_shape`` 对**缺键**是「跳过那一项」✓ ⇒ 缺了关键键它会**默默给出 32B 的默认尺子** ✗
        #: （层数 64 / 宽 5120 ✓）—— 于是壳建得巨大、装载报「缺 几千个键」✓
        #: 读起来像「权重不全」✗，其实是拿错了文件 ✓。这两格必须显式在 ✓。
        missing = [key for key in ("model.embed_tokens.weight",
                                   "model.layers.0.self_attn.q_proj.weight")
                   if key not in shapes]
        if missing:
            raise TorchBackendUnavailable(
                f"这份权重不像 Qwen3-VL 语言塔 ✗：缺 {missing} ✓"
                f"（Krea2 要 `Qwen3_VL_4B/Krea2-Engineer-V1-*.safetensors` ✓）", reason="pending")
        want_taps = tuple(int(value) for value in (taps or vl_mod.KREA2_TAPS))
        depth = int(shape["layers"])
        if max(want_taps) >= depth:
            raise TorchBackendUnavailable(
                f"Krea2 要取的层号 {want_taps} 超出这份塔的层数 {depth} ✗ "
                f"（条件宽会少一段 ⇒ 主干报 30720 ≠ 实数 ✓✗）", reason="pending")
        runtime = vl_mod.Krea2TeRuntime(shape=shape, taps=want_taps,
                                       max_length=int(max_length))
        #: ⚠️ **先核后改** ✓：``output_dim`` 必须等于主干 ``text_dim``（30720 ✓）
        self._check_text_encoder(candidate=runtime)
        tokenizer = self._te_tokenizer(tokenizer_path, runtime,
                                       id_hint="（151644/151645 ✓）")
        want = dtype or self._compute_dtype()
        module = vl_mod.build_krea2_stack(shape, dtype=want, taps=want_taps)
        #: ⚠️ 摆位置要摆**语言塔本体** ✓✗（不是外壳 ✓，2026-11-02 改 ✓）—— 两件事都指同一个对象：
        #: ① `load_module_weights` 装进 ``module.tower`` ✓ ⇒ 内存这一级的两个约定
        #: （``_residency_reload`` / ``_residency_spilled`` ✓）与"卸了哪几层"都得写在同一棵树上 ✓，
        #: 写在外壳上 ⇒ 装载时读不到 ⇒ **卸了的层再也不会被读回来** ✓✗；
        #: ② `residency.survey` 只在裸 ``nn.Module`` 容器上穿透 ✓（见 `_expand` ✓），而外壳的类型是
        #: ``Krea2TextStack`` ✓ ⇒ 摆外壳会把**整座 8 GiB 的塔当成一个非层组** ✓（切不动 ⇒ 静默退成纯 CPU ✗）。
        #: 摆本体之后 `model.embed_tokens` ＋ `model.layers.N` 才认得出 ✓，按层切才有意义 ✓。
        #: ⚠️ 入口 hook 挂在本体上也够 ✓：外壳的 ``forward`` 只是 ``self.tower(ids=..)`` ✓（无自己的参数 ✓）。
        residency_report = self._place_before_load(module.tower, what="te")
        #: 权重装进**语言塔本体** ✓（外层那层只做"取 12 层 + 裁前缀"✓，参数全在 ``tower`` 里 ✓
        #: —— 装到外壳上的话键名会多一层 ``tower.`` ✓ 全落 missing ✓✗）。
        report_weights = weights_mod.load_module_weights(module.tower, path, dtype=want,
                                                        key_map=key_map)
        if not report_weights.complete:
            del module
            self._raise_te_mismatch(report_weights)
        ignored = [name for name in report_weights.unexpected
                   if name.startswith(("model.visual.", "visual.", "lm_head.", "model.lm_head."))]
        self._textEncoder = module.eval().requires_grad_(False)
        self._textConfig = runtime
        self._textKind = "krea2-te"
        self._tokenizer = tokenizer
        return self._te_report(
            kind="krea2-te", runtime=runtime, path=path, want=want,
            residency_report=residency_report, report_weights=report_weights,
            tokenizer=tokenizer, tokenizer_path=tokenizer_path,
            extra={"unexpectedCount": len(report_weights.unexpected),
                   "ignoredTowerKeys": len(ignored), "taps": list(want_taps),
                   "conditionDim": runtime.output_dim},
            note=("**真** Qwen3-VL-4B 语言塔 ✓（取第 2,5,…,35 层叠成 12 层栈 ✓、不过 final norm ✓、"
                  "裁掉模板前缀 ✓）；unexpected 里的视觉塔/lm_head 是**有意不装** ✓"))

    def attach_h3_text(self, weights: str, *, tokenizer_path: str | None = None,
                       max_length: int = 512, dtype: Any = None) -> dict[str, Any]:
        """挂 **H3 的真文本塔** ✓：同一个文件里的**语言塔 ＋ 视觉塔**（Qwen3-VL-32B ✓）
        ⇒ :meth:`encode_text` 出真 ``hidden`` ＋ ``tags`` ✓（H3 全链的第一环 ✓）。

        ## 为什么"一个方法挂两半"，与另三座塔**正好相反** ✗
        盘上那份 ``qwen3vl_32b_minimax_h3_int8_convrot.safetensors``（**25.28 GiB** ✓）一半是
        语言塔（``model.*`` ✓ 1251 键）、一半是视觉塔（``visual.*`` ✓ 351 键）✓ —— 两半**必须
        同时在场** ✓：视觉特征由视觉塔算出、再按 ``<|image_pad|>`` 的位置塞进语言塔的 embeds ✓
        （见 :func:`~.h3_encode.encode_condition` ✓）。只挂语言塔 ⇒ fl2va/ref2va 的参考图块
        **没有特征** ✓，而位置已经占了 ✓、形状全对 ✓✗（不报错）。
        ⚠️ 另三座塔（Qwen3-4B / Qwen3.5-9B / Qwen2.5-VL-7B）是**只挂语言塔** ✓、视觉塔落
        ``unexpected`` ✓ —— 那是**有意**不装 ✗，别把这套"忽略视觉塔"的口径搬过来 ✗。

        ## 键名空间（这里最容易静默错 ✓）
        :func:`~.h3_encode.combine` 把语言塔挂在 ``self.model`` ✓、视觉塔挂在 ``self.visual`` ✓
        ⇒ ``state_dict()`` 与盘上**逐字对应** ✓ ⇒ **一次** ``load_module_weights`` 装完 ✓
        （不是两次装载 ✓、也不需要 ``key_map`` 搬前缀 ✓）。

        ## 两条硬口径
        1. **int8 常驻** ✓：这份文件是 ``int8_convrot`` ✓（350 个线性层 ＝ int8 权重 ＋
           ``weight_scale`` ＋ ``comfy_quant`` 说明 ✓）⇒ 必须走
           :func:`~.h3_form.upgrade_linear_to_int8` 把 ``nn.Linear`` 换成
           :class:`~.h3_form.Int8Linear` ✓，否则每个 int8 权重都按"反量化成 bf16"装进来 ⇒
           50 GiB ✓✗（卡上 22.49 GiB ✓）；漏升级**不会当场报错** ✗，几十分钟后才在
           前向里 OOM ✓✗；
        2. **放置要在装载之前** ✓（:meth:`_place_int8_module` ✓）：int8 部分是 **22.7 GiB** ✓，
           比整块卡（22.49 GiB ✓）还大 ⇒ 必须要么按层切（``residency`` ✓）、要么把 int8
           钉在锁页内存（``int8_offload`` ✓）—— 两种都不开 ⇒ 装载期就 OOM ✓✗。
        """
        self._gate()
        from app.services.engine import h3_encode as h3_encode_mod  # noqa: PLC0415 —— 不拖 torch ✓
        from app.services.engine import h3_form  # noqa: PLC0415
        from app.services.engine import quant as quant_mod  # noqa: PLC0415
        from app.services.engine import qwen3vl_te as vl_te_mod  # noqa: PLC0415
        from app.services.engine import qwen3vl_vision as vl_vision_mod  # noqa: PLC0415

        path = Path(str(weights))
        if not path.exists():
            raise TorchBackendUnavailable(
                f"H3 文本塔权重不存在 ✗：{path} ✓（要的是 Qwen3-**VL**-32B 那份"
                f"「语言塔＋视觉塔同文件」✓，如 `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` ✓；"
                f"**不是** Qwen3-4B / Qwen3.5 / Qwen2.5-VL ✗ —— 那几份只有语言塔 ✓）",
                reason="pending")
        info = st.inspect(path)                       # ⚠️ 只读头部 ✓（25 GiB 不落内存 ✓）
        shapes = {name: tuple(record.shape) for name, record in info.tensors.items()}
        #: ⭐ 先读**逐层量化说明** ✓✗（2026-11-09 ✓）：① ``packed`` 决定**反推尺子**时列数要不要 ×2 ✓
        #: （nvfp4 的 ``q_proj`` 是 ``[8192, 2560]`` ✓，按形状直读会把 hidden 读成 2560 ✓✗ ⇒
        #: heads 变 32、40 层前向全错、**一声不响** ✓）；② ``convrot`` 给 int8 那一支换壳用 ✓。
        specs = loader.quant_specs_of(path, info)
        packed = quant_mod.packed_weight_stems_of(specs)
        #: ⚠️ **先看两半在不在** ✗（依据是**键名**，不是文件名 ✓）：只有一半 ⇒ 形状照样全对 ✓、
        #: 装完 `complete=True` ✓、前向里视觉块拿不到特征 ✓✗ —— 这会一路静默到出片才发现 ✓。
        halves = {"model.embed_tokens.weight": "语言塔（model.* ✓）",
                  "visual.patch_embed.proj.weight": "视觉塔（visual.* ✓）"}
        absent = [text for key, text in halves.items() if key not in shapes]
        if absent:
            raise TorchBackendUnavailable(
                f"这份权重缺了 {'、'.join(absent)} ✗ ⇒ 语言塔与视觉塔**必须同文件** ✓"
                f"（H3 的 `0/1` 类标签与 ``<|image_pad|>`` 注入都要求两半同时在场 ✓；"
                f"缺视觉塔时形状照样全对 ✓、前向里参考图块**静静地**没有特征 ✓✗）",
                reason="pending")
        try:
            text_shape = vl_te_mod.infer_shape(shapes, packed_stems=packed)
            vision_shape = vl_vision_mod.infer_shape(shapes, packed_stems=packed)
        except Exception as error:  # noqa: BLE001 —— 两族的 ShapeError 都不是同一类 ✓
            raise TorchBackendUnavailable(
                f"这份权重推不出 Qwen3-VL-32B 的 H3 文本塔 ✗：{error} ✓",
                reason="pending") from error
        runtime = h3_encode_mod.H3TeRuntime(text=text_shape, vision=vision_shape,
                                            max_length=int(max_length))
        # ⚠️ **先核后改** ✓：`output_dim`（5120 ✓）必须等于主干的 `text_dim`（H3 也是 5120 ✓）
        self._check_text_encoder(candidate=runtime)
        tokenizer = self._te_tokenizer(tokenizer_path, runtime,
                                       id_hint="（`<|image_pad|>` / `<|vision_start|>` ✓）")
        #: 跟着 `self._device` 走 ✓（有卡但显式选 CPU 时不能建成 bf16 ✗，见 `_compute_dtype` ✓）
        want = dtype or self._compute_dtype()
        #: ⚠️ 两半**只建一次** ✓（各自 set_default_dtype 后复原 ✓）；``combine`` 把语言塔挂
        #: ``model`` ✓、视觉塔挂 ``visual`` ✓ ⇒ 键名与盘上逐字对齐 ✓。
        #: ⚠️⚠️ 骨架**在 meta 上建** ✓✗：32B 的 bf16 骨架是 **62 GiB** ✓ —— 按 CPU 真建会先
        #: `uniform_` 填满这 62 GiB ✓（实测宿主内存冲到 **80/93 GiB** ✓、进程随后停滞十几分钟 ✓、
        #: 连权重文件一个字节都没开始读 ✓✗）。meta 建壳 ⇒ 一块实存都不占 ✓。
        module = h3_encode_mod.combine(
            vl_te_mod.build_text_tower(text_shape, dtype=want, device="meta"),
            vl_vision_mod.build_vision_tower(vision_shape, dtype=want, device="meta"))
        #: ⭐ int8 常驻层 ✓ —— convrot 走**逐层**说明 ✓（同一份文件里逐层分组可以不同 ✓，
        #: 拿族级说明判会给常驻层少一步反旋转 ⇒ 能装能跑、画面全错 ✓✗）。
        #: ⚠️⚠️ **换壳必须排在物化之前** ✗✗：那 350 个 ``nn.Linear`` 一旦被物化成 bf16 实存
        #: （62 GiB ✓），换掉它们**不会**把内存还给系统 ✓（实测 WS 停在 74 GiB、系统只剩 6 GiB ✓✗）。
        #: 反过来先在 meta 上换壳 ⇒ 物化时只按 **int8 22.7 GiB ＋ 其余 3 GiB** 分配 ✓。
        convrot, convrot_gap = quant_mod.convrot_group_of(specs)
        if convrot_gap:
            raise TorchBackendUnavailable(
                f"convrot 说明自相矛盾 ✗ ⇒ 不装载（不按猜的 group 算 ✗）：{convrot_gap}",
                reason="pending")
        #: ⭐ **两支常驻壳，按文件自述分派** ✓✗（2026-11-09 ✓）：
        #: ``int8_convrot`` 那份 ⇒ :func:`h3_form.upgrade_linear_to_int8` ✓（``Int8Linear`` ✓）；
        #: ``nvfp4`` 那份 ⇒ :func:`h3_form.upgrade_linear_to_nvfp4` ✓（``NvFp4Linear`` ✓，权重
        #: ``uint8 [out, in/2]`` ✓ 打包原样上卡 ✓、反量化在前向里逐层 ✓）。
        #: ⚠️ 两支**互斥但都要在物化之前** ✗✗：同一份文件里 350 个 ``nn.Linear`` 若先被物化成 bf16
        #: 实存（62 GiB ✓），换壳**不会**把内存还给系统 ✓（实测 WS 停在 74 GiB ✓✗）。
        nv_plan = nvfp4_scale_plan(info, packed_stems=packed)
        plan = int8_scale_plan(info)
        if nv_plan:
            replaced = h3_form.upgrade_linear_to_nvfp4(module, nv_plan)
            quant_kind = "nvfp4"
            #: ⚠️ 同一份文件里若**还有** int8 线性层（混合量化 ✓，盘上目前没有 ✓）：它们会被
            #: `load_module_weights` 按 bf16 反量化 ✓（数值对 ✓、只是占 2 倍 ✓）⇒ 如实记账，
            #: 不假装"都常驻了" ✗（``nvfp4LeftoverInt8`` 一并进回执 ✓）。
            linear_names = {name for name, holder in module.named_modules()
                            if isinstance(holder, self._torch().nn.Linear)}
            leftover_int8 = sorted(set(plan) & linear_names)
        elif plan:
            replaced = h3_form.upgrade_linear_to_int8(module, plan, convrot=convrot)
            quant_kind = "int8"
            leftover_int8 = []
            #: ⚠️ nvfp4 那一支里 ``embed_tokens`` 也是 ``I8`` ＋ per-row scale ✓ —— 它不是
            #: ``nn.Linear`` 而是 ``nn.Embedding`` ✓ ⇒ **不换壳**（没有 int8 嵌入壳 ✓）、装载时
            #: 按 bf16 反量化 ✓（151936×5120 ⇒ 1.45 GiB ✓，这份账已算在"其余张量"里 ✓）。
        else:
            raise TorchBackendUnavailable(
                f"这份 H3 文本塔里**一个 int8/nvfp4 权重都没有** ✗（{len(info.tensors)} 键 ✓）⇒ "
                f"不像 `int8_convrot` / `nvfp4_awq` 那两份 ✓；按 bf16 整份装会要 50 GiB ✗ ⇒ **拒装** ✓"
                f"（真要用 bf16 版请显式换成 nvfp4/bf16 的那条路 ✓）", reason="pending")
        #: ⚠️ 先钉**文件里是 F32** 的键 ✓（与 `_prepare_h3_module` 的同一条口径与顺序 ✓）：
        #: 这份文件里 F32 只有 ``*.weight_scale`` ✓（int8 常驻层自己就按 float32 建 ✓）——
        #: 但**不写死名单** ✗：将来的重导出可能带别的 fp32 岛 ✓。
        #: ⚠️ 这一步在 **meta 上免费** ✓（只改 dtype ✓ 不分配 ✓；顺序也必须在物化之前 ✗，
        #: 否则 weight_scale 会先按 fp32 物化一遍再被换掉 ✓）。
        holders = (dict(module.named_parameters()), dict(module.named_buffers()))
        for logical, tensor in info.tensors.items():
            if tensor.dtype != "F32":
                continue
            for holder in holders:
                holder_target = holder.get(logical)
                if holder_target is not None and holder_target.is_floating_point():
                    holder_target.data = holder_target.data.to(self._torch().float32)
        #: 到这里才**真的分配内存** ✓（只给还停在 meta 上的那些 ✓；int8 层的 22.7 GiB 是这一份 ✓）。
        #: ⚠️ 安全网：`load_module_weights` 的 ``missing`` = "模块里有、文件里没有" ✓
        #: ⇒ 空壳里少写一个参数就 `complete=False` ⇒ 当场拒 ✓（不会把没填的壳当模型跑 ✓）。
        filled = _materialize_meta(module, torch_mod=self._torch(), device="cpu")
        #: ⭐ **联合驻留／int8 钉 CPU 先切再装** ✓✗（与 `load_weights` 的 H3 线同一个方法 ✓）：
        #: 25.28 GiB 的塔在 22.49 GiB 的卡上，不切就是装载期 OOM ✓。
        residency_report, parked = self._place_int8_module(module, what="h3-te")
        report_weights = weights_mod.load_module_weights(module, path, dtype=want)
        if not report_weights.complete:
            del module
            self._raise_te_mismatch(report_weights, "H3 文本塔")
        # ⚠️ int8 的账**当场对** ✓✗（层没接管 ⇒ 报告里两个数各自"正常" ✓✗，见 `_int8_residency_problems` ✓）
        int8_ledger, int8_problem = _int8_residency_problems(module, report_weights)
        if int8_problem:
            del module
            raise TorchBackendUnavailable(
                int8_problem + "　⇒ **不装载** ✓（装进去也跑不起来 ✗）", reason="pending")
        self._textEncoder = module.eval().requires_grad_(False)
        self._textConfig = runtime
        self._textKind = "h3-te"
        self._tokenizer = tokenizer
        #: ⚠️ 装完**立刻**把装载临时还给驱动 ✓✗（与 `load_weights` 末尾同一句 ✓）：
        #: 剩下不还的那部分，下一步 `enter_phase` 的探测会把它算成"卡被占了" ✓ ⇒ 下一件被白切 ✓✗。
        self._trim_cuda()
        return self._te_report(
            kind="h3-te", runtime=runtime, path=path, want=want,
            residency_report=residency_report, report_weights=report_weights,
            tokenizer=tokenizer, tokenizer_path=tokenizer_path,
            extra={"int8Residency": parked, "int8Ledger": int8_ledger,
                   #: meta 建壳那条路的账 ✓：实际被物化的参数/缓冲区个数 ✓（0 ⇒ 一个 meta 都没剩 ✓）
                   "int8Upgraded": len(replaced), "materializedParams": filled,
                   #: ⭐ 量化档位**如实报** ✓（2026-11-09 ✓）：``int8`` / ``nvfp4`` 两条路的
                   #: 常驻形态、反量化时机、AWQ 层数都不一样 ✓ ⇒ 回执只写 ``int8Upgraded`` 会让
                   #: 人把 nvfp4 那份当成 int8 ✓✗（``nvfp4Resident`` 与 ``weightLoad`` 的
                   #: ``dequant.nvfp4Resident`` 是**同一件事的两个观测点** ✓）。
                   "quantKind": quant_kind, "nvfp4Upgraded": len(replaced) if nv_plan else 0,
                   "nvfp4AwqLayers": sum(1 for _, awq in nv_plan.values() if awq),
                   "nvfp4PackedGiB": round(sum(int(info.tensors[f"{stem}.weight"].nbytes)
                                               for stem in nv_plan) / device_policy.GIB, 2),
                   "leftoverInt8Layers": list(leftover_int8),
                   "textDim": runtime.output_dim,
                   "visionShape": dict(vision_shape),
                   "unexpectedCount": len(report_weights.unexpected)},
            note=(("**真** Qwen3-VL-32B 双塔 ✓（语言塔 `model.*` ＋ 视觉塔 `visual.*` ✓，"
                   "一个文件一次装 ✓）；" + ("nvfp4 打包常驻层 " + f"{len(replaced)} 个 ✓"
                                             "（打包原样上卡 ✓、反量化在**前向逐层** ✓，"
                                             f"其中 AWQ {sum(1 for _, awq in nv_plan.values() if awq)} 层 ✓）"
                                             if nv_plan else
                                             f"int8 常驻层 {int8_ledger['layers']} 个 ✓")
                   + "；`tags` 由 `h3_encode.encode_condition` 按 `<|image_pad|>` 位置算 ✓")))

    def attach_vae(self, config: Any = None) -> dict[str, Any]:
        """挂视频 VAE ✓ ⇒ :meth:`decode` 能出**真帧** ✓ —— **两族都认** ✓（2026-10-29 扩 ✓）。

        * 不给 / 给 :class:`vae.VideoVAEConfig` ⇒ **参考** VAE ✓（⚠️ **未经训练** ✗ ⇒ 画面是噪声 ✓，
          验的是**管道** ✓）；
        * 给 :class:`vae_h3.H3VideoVAEConfig`（**必须带 ``weights``** ✓）⇒ **真** VAE ✓
          （H3 那份 fp16 权重 ✓，562 个键逐个对上 ✓）⇒ :meth:`decode` 出的是**真画面** ✓。

        ⚠️ 两族的 ``encode`` / ``decode`` 调用点**完全一样** ✓（当初才敢写"换实现即可" ✓）；
        差别只在**权重从哪来** ✗ 与**报告里怎么说** ✓ ⇒ 报告多两个字段 ``kind`` / ``complete`` ✓：
        调用方据此才知道这条链路到底是不是真画面 ✓✗（**不许沉默** ✗）。
        ⚠️ 真 VAE 的**设备另算** ✗：``H3VideoVAEConfig.device`` 说了算 ✓（默认 **cpu** ✓）——
        它与"主 DiT **19.62 GiB int8 常驻**"抢同一块显存 ✓✗（``loader.residency_plan`` 早说过：
        两族生命周期**不重叠** ✓ ⇒ 要上卡得先让 DiT 下卡 ✓，那是分时驻留的事 ✓，本方法不替它决定 ✗）。
        """
        self._gate()
        from . import vae_h3 as vae_h3_mod  # noqa: PLC0415 —— 只有真 VAE 那条路要它 ✓

        if isinstance(config, vae_h3_mod.H3VideoVAEConfig):
            # ⚠️ **先核后改** ✓（同一套跨来源校验 ✓）：候选不合格 ⇒ 此刻模块一个字没动 ✓
            self._check_attached_vaes(video=config)
            # ⚠️ 内部 strict=True ✓：装不满**当场抛** ✗（不返回"半个模型" ✓✗）
            module = vae_h3_mod.build_h3_video_vae(config)
            load_report = getattr(module, "loadReport", None)
            report = config.to_dict()
            report["kind"] = "h3-real"
            report["complete"] = bool(getattr(load_report, "complete", False))
            report["weightDevice"] = str(module.weight_device)
            report["missing"] = list(getattr(load_report, "missing", []) or [])
            #: ⭐ 与下面两条真 VAE（flux-real / wan21-real ✓）**同一个形态** ✓✗：原来这一格只赋值、
            #: 没 ``eval()`` 也没冻 ✓ ⇒ H3 的 VAE 解码同样在**建图** ✓（decoder 的 norm 权重默认取梯度 ✓）
            #: ⇒ 一步解码就把整份解码器中间激活钉在显存里 ✓✗（`decode` 阶段恰好是"DiT 已退、卡上空着" ✓，
            #: 所以症状不是 OOM ✗ 而是"解码比预期慢、且退完不还"✓）。
            self._vae = module.eval().requires_grad_(False)
            self._vaeConfig = config
            self._vaeKind = "h3-real"
            return report

        config = config or vae_mod.VideoVAEConfig()
        # ⚠️ **先核后改** ✗（2026-09-22 ✓）：核不过是**候选配置**的问题 ✓ ⇒ 此刻模块一个字没动 ✓
        #    —— 若改成"先挂上再核" ✗，失败时要么留着坏的 ✓✗、要么把**先前挂好的**那个也清掉 ✓✗。
        self._check_attached_vaes(video=config)
        self._vae = vae_mod.build_vae(config).to(self._device).eval().requires_grad_(False)
        self._vaeConfig = config
        self._vaeKind = "reference-untrained"
        report = config.to_dict()
        report["kind"] = "reference-untrained"
        return report

    def attach_flux_vae(self, weights: str, *, dtype: Any = None,
                        config: Any = None) -> dict[str, Any]:
        """挂 **Flux VAE** ✓（``vae/ae.safetensors`` ✓ **244** 键 ✓）⇒ Z-Image 的 **16 通道**潜变量
        能解成**真图** ✓（见 :mod:`vae_flux` ✓）。

        为什么单独一个方法 ✗（不塞进 :meth:`attach_vae` 的两个分支 ✓）：那两族都是**视频 3D/5 维** ✓
        （H3 真权重 ✓ / 参考未训练 ✓），而 Flux VAE 是**图 4 维** ✓、缩放系数还是**另一套**
        （``0.3611 / 0.1159`` ✗ 不是 ``0.13025`` ✓✗ 也不是 H3 的 ✓）。三族混进一个 isinstance 链
        ⇒ 迟早有人拿 H3 的配置去解 Z-Image 的潜变量 ✓✗（形状会先炸 ✓，但**缩放错**那条**根本不炸** ✓✗）。

        ⚠️ 四条纪律同 :meth:`attach_vae` ✓：
        1. **先核后改** ✓（候选不合格 ⇒ 模块一个字没动 ✓）；
        2. **严格装载** ✓（键集/形状对不上**当场抛** ✗ —— 不留随机初始化的层 ✓✗）；
        3. **报告说真话** ✓（``kind="flux-real"`` + ``complete`` ✓ ⇒ 调用方据此知道这是不是真图 ✓）；
        4. **设备**：VAE 上 ``self._device`` ✓ —— 与 DiT 抢显存 ✓✗ ⇒ 24 GiB 卡上要先让 DiT 下卡 ✓
           （分时驻留 ✓，本方法不替它决定 ✗）。
        """
        self._gate()
        from app.services.engine import vae_flux as vae_flux_mod  # noqa: PLC0415
        from app.services.engine import weights as weights_mod  # noqa: PLC0415

        path = Path(str(weights))
        if not path.exists():
            raise TorchBackendUnavailable(
                f"Flux VAE 权重不存在 ✗：{path} ✓（Z-Image 要的是 **Flux** 的 ae ✓，"
                f"不是 qwen_image_vae ✗ —— 见 vae_flux 模块头 ✓）", reason="pending")
        spec = config or vae_flux_mod.flux_vae_config()
        # ⚠️ **先核后改** ✓：`_check_attached_vaes` 会核 `spatial_scale` ↔ 主干 `vae_scale` ✓
        #    与 `latent_channels` ↔ 主干 `in_channels` ✓（16 ✓ —— 挂成 SDXL 的 4 通道会当场拒 ✓）。
        self._check_attached_vaes(video=spec)
        torch = self._torch()
        #: 跟着 `self._device` 走 ✓（有卡但显式选 CPU 时不能建成 bf16 ✗，见 `_compute_dtype` ✓）
        want = dtype or self._compute_dtype()
        module = vae_flux_mod.build_flux_vae(spec, dtype=want)
        report = weights_mod.load_module_weights(module, path, dtype=want)
        if not report.complete:
            raise TorchBackendUnavailable(
                f"Flux VAE 权重与架构**不符** ✗（缺 {len(report.missing)} 键 ✓ / "
                f"形状不符 {len(report.shapeMismatch)} 处 ✓，如 {report.missing[:4]} ✓）"
                f"　⇒ 拒 ✗（不许 `strict=False` 糊过去 ✓✗）", reason="pending")
        self._vae = module.to(self._device).eval().requires_grad_(False)
        self._vaeConfig = spec
        self._vaeKind = "flux-real"
        detail = spec.to_dict()
        detail.update({"kind": "flux-real", "complete": True, "path": str(path),
                       "dtype": str(want), "loadedKeys": int(report.loadedKeys),
                       "sourceKeys": int(report.sourceKeys),
                       "note": "**真** Flux VAE ✓（244 键逐键对上 ✓）⇒ 解码出的是真图 ✓"})
        return detail

    def attach_wan_vae(self, weights: str, *, dtype: Any = None,
                       family: str = "qwen_image21") -> dict[str, Any]:
        """挂 **Wan layout 的 VAE** ✓ ⇒ Qwen-Image **2.1**（64ch ✓）／**Krea2**（16ch ✓）的
        潜变量都能解成真图 ✓（见 :mod:`vae_wan` ✓）。

        ⭐ 两族**共用这一条**、但``family`` 给的不一样 ✓✗（2026-11-01 接 Krea2 时确认 ✓）：
        2.1 给 ``"qwen_image21"`` ✓（64ch，mean/std 恒等 ✓）；**Krea2 给 ``"wan21"``** ✓ ——
        它的潜空是 ``latent_formats.Wan21`` ✓（comfy ``supported_models.Krea2`` 里写死 ✓），
        而 VAE 本体用 ``vae/qwen_image_vae.safetensors`` ✓（16ch / 8× ✓）。
        ⚠️ 别把这两格弄反 ✗✗：给错 mean/std ⇒ 解码照跑 ✓、形状全对 ✓、**图整体偏色/发灰** ✓✗；
        给错 VAE 文件 ⇒ 尺寸或通道对不上 ⇒ `_check_attached_vaes` 当场拒 ✓（这条会报 ✓）。

        为什么单开一个方法 ✗（不扩 :meth:`attach_flux_vae` ✓）：族不同 ⇒ **反缩放不同** ✓✗
        —— Flux 是 ``(x − 0.1159) / 0.3611`` ✓，2.1 是 ``x * std + mean``（64 个数 ✓）。两族混一处
        ⇒ 迟早拿 Flux 的常数解 2.1 的潜变量 ✓✗：形状全对 ✓、图发灰 ✓、一个错都不报 ✓✗。

        ⚠️ 四条纪律同 :meth:`attach_vae` ✓：**先核后改** / **严格装载** / **报告说真话** /
        **设备**（VAE 上 ``self._device`` ✓，与 DiT 抢显存 ⇒ 分时驻留由调用方决定 ✓）。

        ``family`` 只用来取 ``latents_mean/std`` ✓；取不到 ⇒ 当场抛 ✗（不回退成 ``(0,1)`` ✗✗）。
        """
        self._gate()
        from app.services.engine import vae_wan as vae_wan_mod  # noqa: PLC0415
        from app.services.engine import weights as weights_mod  # noqa: PLC0415

        path = Path(str(weights))
        if not path.exists():
            #: ⚠️ 别把两族的文件名写死成一句 ✗✗（2026-11-01 修 ✓）：2.1 用
            #: ``qwen_image_2.1_vae_bf16`` ✓、**Krea2 用 ``qwen_image_vae``** ✓ —— 老句子
            #: 「不是 qwen_image_vae ✗」对 Krea2 恰好说反 ✓✗（照它找 ⇒ 拿到 64ch 那份 ⇒ 装不上 ✓）。
            raise TorchBackendUnavailable(
                f"Wan 系 VAE 权重不存在 ✗：{path} ✓　〔2.1 要 ``vae/qwen_image_2.1_vae_bf16`` ✓、"
                f"Krea2 要 ``vae/qwen_image_vae`` ✓，两族**不同文件** ✗〕",
                reason="pending")
        info = st.inspect(path)
        spec = vae_wan_mod.infer_shape({name: tuple(t.shape) for name, t in info.tensors.items()})
        # ⚠️ **先核后改** ✓：`spatial_scale` ↔ 主干 `vae_scale`（2.1 = 16 ✓）、
        #    `latent_channels` ↔ 主干 `in_channels`（64 ✓）—— 挂成 16 通道那族会当场拒 ✓。
        self._check_attached_vaes(video=spec)
        try:
            mean, std = vae_wan_mod.latents_stats(family, z_dim=spec.z_dim)
        except ValueError as err:
            raise TorchBackendUnavailable(
                f"这族的**反缩放事实**取不到 ✗：{err}　⇒ ⚠️ 不拿 ``(0, 1)`` 顶上 ✗✗"
                f"（那会解码照跑、形状全对、图发灰、不报错 ✓✗）", reason="pending") from err
        torch = self._torch()
        #: 跟着 `self._device` 走 ✓（有卡但显式选 CPU 时不能建成 bf16 ✗，见 `_compute_dtype` ✓）
        want = dtype or self._compute_dtype()
        module = vae_wan_mod.build(spec, dtype=want)
        report = weights_mod.load_module_weights(module, path, dtype=want)
        if not report.complete:
            raise TorchBackendUnavailable(
                f"Wan VAE 权重与架构**不符** ✗（缺 {len(report.missing)} 键 ✓ / "
                f"形状不符 {len(report.shapeMismatch)} 处 ✓，如 {report.missing[:4]} ✓）"
                f"　⇒ 拒 ✗（不许 `strict=False` 糊过去 ✓✗）", reason="pending")
        self._vae = module.to(self._device).eval().requires_grad_(False)
        self._vaeConfig = spec
        self._vaeKind = "wan21-real"
        self._vaeLatents = (tuple(mean), tuple(std), str(family))
        detail = spec.to_dict()
        #: ⚠️ 回执要点名**人话族名** ✓✗（老文案写死「否则 H3」⇒ 挂 2.1 也报 H3 ✓✗）。
        #: 族名 → 标题走 :func:`vae_wan.family_title` ✓（单一真源 ✓，不在这里另抄一张表 ✗）。
        title = vae_wan_mod.family_title(family)
        detail.update({"kind": "wan21-real", "complete": True, "path": str(path),
                       "dtype": str(want), "loadedKeys": int(report.loadedKeys),
                       "sourceKeys": int(report.sourceKeys), "latentsFamily": str(family),
                       "latentsTitle": title,
                       "latentsMean": len(mean), "latentsStd": len(std),
                       #: ⚠️ "238 键"是 2.1 那份的事实 ✓，别写死 ✗（Krea2 的 ``qwen_image_vae`` 键数
                       #: 不同 ✓）：这里报**实测**装了几个键 ✓。
                       "note": f"**真** Wan layout VAE ✓（{int(report.loadedKeys)} 键逐键对上 ✓）"
                               f"＋ 反缩放按 **{title}** 的 {len(mean)} 个 mean/std ✓"
                               f"（族名 ``{family}`` ✓）⇒ 解码出的是真图 ✓"})
        return detail

    def set_vae_tiling(self, *, tile: int | None = None, halo: int | None = None,
                       attn_chunk: int | None = None, temporal_chunk: int | None = None,
                       fallback: bool | None = None) -> dict[str, Any]:
        """开/关 VAE 解码的省显存旋钮 ✓✗（``tile``/``halo``/``attn_chunk`` 只对 Wan layout 那族 ✓；
        ``temporal_chunk`` 只对 SEEDVR2 那族 ✓ —— 按族分派 ✓，给错族就是闲着 ✓，不报错 ✗）。

        传 ``None`` ⇒ 保持原值 ✓（``0`` / 负数是**关** ✓：``vae_wan`` 里 ``chunk<=0`` 与
        ``tile=None`` 都是一个意思 ✓）；显式复位用 ``reset_vae_tiling()``（全 ``None`` ✓）。

        ⚠️ **开了就改数值** ✓✗，这是它必须由调用方开口、而不是我做默认的原因 ✓：

        * ``tile``（**像素**单位 ✓，与 ComfyUI ``VAEDecodeTiled`` 同口径 ✓）—— 2048×1536 fp32
          实测相对差 **5.8e-03** ✓（不是接缝 ✓，halo 加到 24 也还是 5e-2 量级 ✓：cuDNN 对不同
          窗口尺寸选不同卷积算法 ✓）。
        * ``attn_chunk`` —— 只切 query 维 ✓ ⇒ softmax 每行独立 ✓，但 SDPA **内部**本来也按
          query 分块 ✓ ⇒ 只有 chunk 与它自己对齐时才逐位 ✓（小画布实测逐位 ✓、2048×1536 实测
          差 8.55e-03 ✓）。
        * ``fallback``（**默认 True** ✓✗）⇒ 只在**抛 OOM** 时临时降档重试 ✓（``tile`` /
          ``attn_chunk`` 取 :data:`vae_wan.SUGGESTED_TILE` / :data:`vae_wan.SUGGESTED_ATTN_CHUNK` ✓），
          回执里 ``reason="oom"`` ✓。**为什么默认开** ✗✗：它只在「本来就要崩溃/出不了图」
          那条路上生效 ✓ —— 装得下时**逐位不变** ✓（对逐像素复现零影响 ✓），
          而两条显式旋钮一动就换字节 ✓ ⇒ 那两个按纪律默认关 ✓、这个按事实默认开 ✓。
          那是「总比崩强」✓，**不是优化** ✗（报告里说清是 OOM 降的档 ✓）。

        * ``temporal_chunk``（**潜帧**单位 ✓，只对 SEEDVR2 那族有效 ✓）—— 沿时间维切块解码 ✓，
          每块带**块外真实上文** ✓ ⇒ **结果与整段相等** ✓✗（测试 ⑫ 的「帧数 ``4T-3`` ＋
          ``max|Δ| ≤ 1e-5``」两条判据兜住 ✓）。它和上面两个**不是一类** ✓：那两个是「拿数值换显存」✗，
          它只降**峰值**、不动数值 ✓（峰值只与每块帧数有关 ✓，与整片多长无关 ✓）。
          默认仍关 ✓（= 与整段同一条代码路径 ⇒ 复现性零风险 ✓）：长片/高分辨率按需开 ✓。
          ⭐ 因此 ``fallback`` 在**这一族**降的是**它**（:func:`vae_seedvr2.chunk_ladder` 的阶梯 ✓），
          不是 ``tile`` ✓✗ —— 那一族的峰值来自「时间 ×2 上采样后的重排张量」✓，空间切块救不了 ✓；
          而且这条兜底**不换画质** ✓（回执里多一格 ``"exact": true`` ✓，与 Wan 那族降档必须标
          ``reason="oom"`` 且字节变了 ✓✗ 要分开读 ✓）。
          ⚠️ 阶梯**不是**「降一档就收工」✗✗：峰值 ≈ 与每块像素帧数成正比 ✓，`4` 在 720×480 下
          一块 17 像素帧 ≈ 19 GiB ✓ 自己就接近上限 ✓ ⇒ 4→2→1 逐档减半 ✓，回执 ``tried`` 记下
          试过哪几档 ✓（真到 1 仍 OOM 才是「分辨率/帧数超了」✓）。

        返回**当前生效的**设置 ✓（回执里原样带上 ✓，别让这件事只活在我的脑子里 ✓）。
        """
        patch: dict[str, Any] = {}
        if temporal_chunk is not None:
            patch["temporalChunk"] = temporal_chunk
        if tile is not None:
            patch["tile"] = tile
        if halo is not None:
            patch["halo"] = halo
        if attn_chunk is not None:
            patch["attnChunk"] = attn_chunk
        if fallback is not None:
            patch["fallback"] = fallback
        #: ⚠️ 归一**只走** :func:`normalize_vae_tiling` ✗✗（``0`` / 负数 ⇒ 关 ✓、拼错的键 ⇒ 抛 ✓）
        #: —— 这里以前自己写 ``int(x) if int(x) > 0 else None`` ✓，和装配线那份是**两套口径** ✓✗。
        self._vaeTiling.update(normalize_vae_tiling({**self._vaeTiling, **patch}))
        if temporal_chunk is not None:
            #: 点名要的档位 ⇒ 忘掉「上一轮是自动降的」✓ —— 否则会被当成还能继续降 ✓✗。
            self._vaeAutoChunk = None
        return dict(self._vaeTiling)

    def vae_tiling(self) -> dict[str, Any]:
        """**当前**的 VAE 解码旋钮 ✓（回执/报告读它 ✓，别让调用方去摸私有字段 ✗）。"""
        return dict(self._vaeTiling)

    def read_ahead(self) -> dict[str, Any]:
        """这一趟装配**实际用的预取深度** ✓（§4.3 ✓，2026-11-07 ✓；回执 ``readAhead`` 一格读它 ✓）。

        ⚠️ 深度是**逐文件**定的 ✗（同一趟里 DiT 与 TE 可能不在同一块盘上 ✓）⇒ 这里报的是
        **最后一件**的 plan ✓，带 ``disk`` / ``mbps`` / ``source`` 凭据 ✓ —— 「为什么是 2」要能读出来 ✓✗。
        """
        return dict(weights_mod.last_read_ahead())

    def reset_vae_tiling(self) -> dict[str, Any]:
        """回到**装配默认** ✓：两个旋钮全关 ✓（逐位 ✓）＋ OOM 兜底仍开 ✓（见 :meth:`set_vae_tiling` ✓）。"""
        self._vaeTiling = normalize_vae_tiling(None)
        self._vaeAutoChunk = None
        self._vaeSpilled = False
        return dict(self._vaeTiling)

    def attach_seedvr2_vae(self, weights: str, *, dtype: Any = None) -> dict[str, Any]:
        """挂 **SEEDVR2 VAE** ✓（``vae/SEEDVR2/ema_vae_fp16.safetensors`` ✓ **250** 键 ✓）⇒
        视频修复链的潜变量能解成真画面 ✓（见 :mod:`vae_seedvr2` ✓）。

        为什么第五条独立方法 ✗✗（不并进 ``wan21`` 那条 ✓）：这一族**没有** ``quant_conv`` /
        ``post_quant_conv`` ✓（编码器直出 **32** 通道 ✓、取前 16 均值即 latent ✓）、反缩放是
        ``× 0.9152`` ✓（不是 mean/std 组 ✓、不是 Flux 的 ``0.3611/0.1159`` ✓）；且**空间 8× / 时间 4×** ✓
        —— 拿 Wan 那条解会走 ``latents_mean`` 分支 ✓✗：形状对得上（都是 16ch / 8× ✓）⇒ **不报错** ✓、
        只是整片发灰发糊 ✓✗（这正是本仓反复踩的"不报错只是图不对" ✓）。

        ⚠️ 形状**从权重反推** ✗（不像 flux / wan 那样静态配置 ✓）：块数、通道、``spatial_downscale_ratio``
        全由 250 键的键名与形状决定 ✓ —— 3b / 7b 两个档共用这一条 ✓（``infer_shape`` ✓）。
        反推不出 ⇒ **当场拒** ✗（不许拿默认结构糊 ✓）。

        ⭐ 三条例外/配套事实（2026-11-01 接这条通路时逐条查清 ✓，与 :mod:`vae_seedvr2` 的
        对拍测试 ⑧ 互证 ✓）：

        * **帧号是 4n+1** ✓✗（时间 ÷4 ＋ 因果卷积「复制首帧」补边 ✓）—— :meth:`decode` 之所以
          吃 **5D** ``(B,16,T,H,W)`` 而不是 Wan 的 4D ✓，就是为了让 `4n+1` 这条口径在类型上可见 ✓；
          T 不合规 ⇒ 当场拒并给出最近两个合法值 ✓（别补帧糊 ✗）。
        * **通道对账有一条例外** ✗✗：这一族 DiT ``in_channels`` 是 **33 = 16 潜变量 + 17
          conditioning** ✓（参考实现把两份在**主干内部** concat ✓）⇒ 「DiT ``in_channels`` ==
          VAE ``latent_channels``」这条通则**对它不成立** ✗。`:meth:`_check_attached_vaes` 靠
          ``latent_channels`` 属性（16 ✓）而不是 ``in_channels``（33 ✗）判 ✓ —— 两处必须同步改 ✗：
          只改一处 ⇒ 要么永远挂不上 ✓✗、要么挂上了却不核 ✓✗。
        * **编码侧也在这条通路上** ✓：修链的第一步就是「降质帧 → 潜变量」✓，走的是
          :meth:`encode_pixels` ✓（**不是**把 ``decode`` 反着调 ✗：编码器出 32 通道 ✓ 且要取前 16 ✓）。

        四条纪律同 :meth:`attach_vae` ✓：**先核后改** / **严格装载** / **报告说真话** / **设备**。
        """
        self._gate()
        import safetensors as st  # noqa: PLC0415

        from app.services.engine import vae_seedvr2 as vae_seedvr2_mod  # noqa: PLC0415
        from app.services.engine import weights as weights_mod  # noqa: PLC0415

        path = Path(str(weights))
        if not path.exists():
            raise TorchBackendUnavailable(
                f"SEEDVR2 VAE 权重不存在 ✗：{path} ✓（这一族要的是 ``vae/SEEDVR2/"
                f"ema_vae_fp16.safetensors`` ✓ 0.47 GiB ✓，不是 wan / qwen 的任何一支 ✗）",
                reason="pending")
        with st.safe_open(str(path), framework="pt", device="cpu") as handle:
            shapes = {key: tuple(int(dim) for dim in handle.get_slice(key).get_shape())
                      for key in handle.keys()}
        try:
            spec = vae_seedvr2_mod.infer_shape(shapes)
        except vae_seedvr2_mod.SeedVR2VaeError as exc:
            raise TorchBackendUnavailable(
                f"SEEDVR2 VAE 权重**反推不出结构** ✗：{exc} ✓　⇒ 拒 ✗", reason="pending") from exc
        # ⚠️ **先核后改** ✓：`spatial_scale` ↔ 主干 `vae_scale`（8 ✓）、`latent_channels` ↔ 潜通道
        #    （16 ✓ —— 别用主干 ``in_channels`` ✗，SEEDVR2 那格是 33 ✓）。
        self._check_attached_vaes(video=spec)
        want = dtype or self._compute_dtype()
        module = vae_seedvr2_mod.build(spec, dtype=want)
        report = weights_mod.load_module_weights(module, path, dtype=want)
        if not report.complete:
            raise TorchBackendUnavailable(
                f"SEEDVR2 VAE 权重与架构**不符** ✗（缺 {len(report.missing)} 键 ✓ / "
                f"形状不符 {len(report.shapeMismatch)} 处 ✓，如 {report.missing[:4]} ✓）"
                f"　⇒ 拒 ✗（不许 `strict=False` 糊过去 ✓✗）", reason="pending")
        self._vae = module.to(self._device).eval().requires_grad_(False)
        self._vaeConfig = spec
        self._vaeKind = "seedvr2-real"
        #: ⚠️ 反缩放不走 mean/std 组 ✗ ⇒ 清掉上一支留下的账 ✓✗（否则 `_vaeLatents` 还是旧族的
        #: 64 个数 ✓、`decode` 里按它算会静默套错常数 ✓✗）。
        self._vaeLatents = None
        detail = spec.to_dict()
        detail.update({"kind": "seedvr2-real", "complete": True, "path": str(path),
                       "dtype": str(want), "loadedKeys": int(report.loadedKeys),
                       "sourceKeys": int(report.sourceKeys),
                       "scaleFactor": float(spec.scale_factor),
                       "note": f"**真** SEEDVR2 VAE ✓（{int(report.loadedKeys)} 键逐键对上 ✓）"
                               f"＋ 反缩放 ×{float(spec.scale_factor):.4f} ✓"
                               f"（不是 mean/std 组、也不是 Flux 常数 ✗）、"
                               f"帧号 4n+1 ✓（时间 ÷4 ✓）⇒ 解出的是真画面 ✓、"
                               f"``encode_pixels`` 出的也是真潜变量 ✓"})
        return detail

    def attach_music3_te(self, weights: str, *, dtype: Any = None) -> dict[str, Any]:
        """挂 **Music3 语义塔** ✓（``text_encoder/minimax_music3_text_encoder_int8_convrot`` ✓
        648 键 ✓）⇒ :meth:`encode_text` 出**真条件帧** ``(F, 32768)`` ✓。

        ## 这一座塔与另五座的根本差别（别按"文本塔"的直觉想它 ✗✗）

        它**不编码文本** ✓✗ —— 它是一台**自回归生成器** ✓：吃 ``build_prompt(caption, lyrics)``
        的 token 串 ✓、按 **25 Hz** 逐帧吐**音频语义帧** ✓（每帧 8 codebook × 4096 = **32768** 维 ✓，
        ``CONTEXT_DIM`` ✓）。所以：

        * 它**不会被"编码一次"** ✓：:meth:`encode_text` 对它走 :meth:`_encode_music3_condition` ✓，
          一次调用要跑几百上千步自回归 ✓（**秒级到分钟级** ✓，与 DiT 采样是两段时间 ✓）；
        * 条件帧数**由时长定**（``25 × 秒`` ✓），不是由词数定 ✓ —— 所以它必须先于潜变量尺寸确定 ✓
          （pipeline 的 ``plan.latent_frames`` 与 ``plan.frames`` 都从它来 ✓）。
        * **词表焊在这份 safetensors 里** ✓（键名 ``tokenizer_json`` ✓，一个 uint8 字节块 ✓）：
          盘上**没有** ``tokenizer.json`` 文件 ✓ ⇒ 走 :meth:`tokenizer_bpe.BpeTokenizer.from_tokenizer_payload`
          ✓（2026-11-06 为此加的 ✓；不许往盘上写临时文件 ✗、也不为它引 HF ``tokenizers`` ✗）。
          ⚠️ 建完必须用 :func:`te_music3.validate_tokenizer` 核**四个特殊符** ✓ —— 少一个 ⇒
          ``tokens[1:-2]`` 那段区间整个错位 ✓✗（不报错 ✓、出来的是"人声念提示词"那种噪声 ✓✗）。

        ⚠️ ``require_int8=True`` **不给放松** ✗：这份文件是 int8_convrot ✓，按 bf16 整份装要 13.8 GiB ✓
        与基卡余量打架 ✓、且与 int8 版**逐位不同** ✓（出的是另一段音乐 ✓✗）。
        """
        self._gate()
        from app.services.engine import te_music3 as te_music3_mod  # noqa: PLC0415
        from app.services.engine import tokenizer_bpe as tok_mod  # noqa: PLC0415

        path = Path(str(weights))
        if not path.exists():
            raise TorchBackendUnavailable(
                f"Music3 语义塔权重不存在 ✗：{path} ✓（这一族要的是 ``text_encoder/"
                f"minimax_music3_text_encoder_int8_convrot.safetensors`` ✓ 8.57 GiB ✓；"
                f"**不是**任何 Qwen 文本塔 ✗ —— 它不编码文本、它生成音频语义帧 ✓）",
                reason="pending")
        #: ⚠️ **先词表**（纯字节读 ✓，不占显存 ✓）：词表坏了要能在装 8.57 GiB 之前就拒 ✗✗。
        try:
            raw = st.read_tensor_bytes(str(path), "tokenizer_json", max_bytes=64 << 20)
        except Exception as exc:                                  # noqa: BLE001 —— 归一成人话 ✓
            raise TorchBackendUnavailable(
                f"这份塔里读不到内嵌词表 ✗（键 ``tokenizer_json`` ✓）：{exc} ✓　⇒ 拒 ✗"
                f"（盘上**没有**配套 tokenizer.json 文件 ✓，词表只在这一块字节里 ✓）",
                reason="pending") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:                                  # noqa: BLE001
            raise TorchBackendUnavailable(
                f"内嵌词表**不是合法 JSON** ✗（{len(raw)} 字节 ✓）：{exc} ✓　⇒ 拒 ✗",
                reason="pending") from exc
        tokenizer = tok_mod.BpeTokenizer.from_tokenizer_payload(
            payload, source=f"{path.name}:tokenizer_json")
        tk_problems = te_music3_mod.validate_tokenizer(tokenizer)
        if tk_problems:
            raise TorchBackendUnavailable(
                f"内嵌词表**缺特殊符** ✗：{tk_problems[:3]} ✓（共 {len(tk_problems)} 条 ✓）"
                f"　⇒ 拒 ✗（``build_prompt`` 的 ``tokens[1:-2]`` 会整段错位 ✓✗ —— "
                f"不报错、出来的是人声念提示词那种噪声 ✓）", reason="pending")
        runtime = SimpleNamespace(output_dim=int(te_music3_mod.CONTEXT_DIM),
                                  kind="music3-ar")
        self._check_text_encoder(candidate=runtime)
        want = dtype or self._compute_dtype()
        try:
            tower, tower_report = te_music3_mod.load_tower(
                str(path), dtype=want, place=lambda module: self._place_before_load(
                    module, what="te"))
        except ValueError as exc:
            raise TorchBackendUnavailable(
                f"这份权重装不成 Music3 语义塔 ✗：{exc} ✓　⇒ 拒 ✗", reason="pending") from exc
        self._textEncoder = tower
        self._textConfig = runtime
        self._textKind = "music3-ar"
        self._tokenizer = tokenizer
        return {
            "kind": "music3-ar",
            "config": {"outputDim": int(te_music3_mod.CONTEXT_DIM),
                       "audioFramesPerSecond": int(te_music3_mod.AUDIO_FRAMES_PER_SECOND),
                       "maxAudioFrames": int(te_music3_mod.MAX_AUDIO_FRAMES),
                       "codebooks": int(te_music3_mod.NUM_CODEBOOKS),
                       "codebookWidth": int(te_music3_mod.CODEBOOK_WIDTH)},
            "complete": True,
            "path": str(path),
            "dtype": str(want),
            "device": str(self._device),
            "residency": tower_report.get("placement"),
            "loadedKeys": int(tower_report["keys"]),
            "sourceKeys": int(tower_report["keys"]),
            "int8Layers": int(tower_report["int8Layers"]),
            "tokenizer": getattr(tokenizer, "name", type(tokenizer).__name__),
            "tokenizerPath": "embedded:tokenizer_json",
            "tokenizerBytes": int(len(raw)),
            "note": (f"**真** Music3 语义塔 ✓（{int(tower_report['keys'])} 键 ✓、"
                     f"{int(tower_report['int8Layers'])} 层 int8 常驻 ✓）；⚠️ 它**不编码文本** ✗ ——"
                     f" 吃 ``build_prompt(caption, lyrics)`` 并按 {int(te_music3_mod.AUDIO_FRAMES_PER_SECOND)} Hz"
                     f" 逐帧生成 {int(te_music3_mod.CONTEXT_DIM)} 维音频语义帧 ✓；"
                     f"词表内嵌在这份文件里 ✓（{len(raw)} 字节 ✓，四个特殊符齐全 ✓）"),
        }

    def attach_music3_dav(self, weights: str, *, dtype: Any = None) -> dict[str, Any]:
        """挂 **Music3 DAV** ✓（``vae/minimax_music3_dav.safetensors`` ✓ **121** 键 ✓）⇒
        音频潜变量 ``(B,128,T)`` 能解成真波形 ✓（见 :mod:`vae_music3` ✓）。

        为什么第五条独立 VAE 方法 ✗✗：这一族的潜空间**是 3 维的** ✓（``128 × T`` ✓，没有 H/W ✓）、
        反缩放**恒等** ✓（``scale_factor = 1.0`` ✓ —— 既不是 Flux 的 ``0.3611/0.1159`` ✓、
        也不是 Wan 的 mean/std 组 ✓）；拿视频那条解会走 4D reshape ✓✗：形状对得上（128 = 2×64 ✓）
        ⇒ **不报错** ✓、出来的是一段"有音量没内容"的噪声 ✓✗。

        ⚠️ **输出不是帧是波形** ✗✗：所以 :meth:`decode` 对这一族**提前 return** ✓
        （见 :meth:`_decode_music3` ✓），不走 ``frames`` 那条通用路 ✓。
        ⚠️ **不核 ``_check_attached_vaes`` 的通道对账** ✗：那条通则是「DiT ``in_channels`` == VAE
        ``latent_channels``」✓ —— Music3 的 DiT 入口是 **128** ✓、DAV 也是 **128** ✓，本来就相等 ✓，
        所以照常核 ✓（不像 SEEDVR2 的 33 vs 16 ✓✗）。
        """
        self._gate()
        import safetensors as st  # noqa: PLC0415

        from app.services.engine import vae_music3 as vae_music3_mod  # noqa: PLC0415
        from app.services.engine import weights as weights_mod  # noqa: PLC0415

        path = Path(str(weights))
        if not path.exists():
            raise TorchBackendUnavailable(
                f"Music3 DAV 权重不存在 ✗：{path} ✓（这一族要的是 ``vae/"
                f"minimax_music3_dav.safetensors`` ✓ 0.20 GiB ✓，不是 H3 的音频 VAE ✗、"
                f"也不是任何视频 VAE ✗）", reason="pending")
        with st.safe_open(str(path), framework="pt", device="cpu") as handle:
            shapes = {key: tuple(int(dim) for dim in handle.get_slice(key).get_shape())
                      for key in handle.keys()}
        problems = vae_music3_mod.shape_problems(shapes)
        if problems:
            raise TorchBackendUnavailable(
                f"Music3 DAV 权重与结构**不符** ✗（{problems[:3]} ✓，共 {len(problems)} 处 ✓）"
                f"　⇒ 拒 ✗（121 键这一族是固定的 ✓，对不上就是拿错了文件 ✓）", reason="pending")
        spec = vae_music3_mod.MUSIC3_DAV_FACTS
        #: ⚠️ ``sample_rate`` 必须**记进配置** ✓✗（2026-11-06 首次真跑栽的坑 ✓）：
        #: :meth:`_decode_music3` 要拿它算时长 ✓，而 ``MUSIC3_DAV_FACTS`` 是**模块常量** ✓ ——
        #: 那一路读的是 ``self._vaeConfig`` ✓，配置里不记就只剩"猜一个 32000" ✗。
        candidate = SimpleNamespace(latent_channels=int(spec["latentChannels"]),
                                    spatial_scale=1,
                                    sample_rate=int(spec["sampleRate"]),
                                    channels=int(spec["stereo"]))
        self._check_attached_vaes(video=candidate)
        want = dtype or self._compute_dtype()
        module = vae_music3_mod.build_music3_dav(dtype=want)
        #: ⚠️ **必须**给 ``key_map`` ✗✗（2026-11-06 首次真跑栽的坑 ✓）：检查点里 weight_norm 是
        #: **旧式** ``…weight_g / …weight_v`` ✓，而 ``build_music3_dav`` 造出来的模块
        #: ``state_dict`` 是**新式** ``…parametrizations.weight.original0/1`` ✓ ⇒ 通用装载器按原键
        #: 硬对 ⇒ 报「缺 60 键 / 形状不符 0 处」✓✗（读起来像拿错了文件 ✓，其实是**同一份文件**
        #: 两套写法 ✓；`shape_problems` 会归一所以它全绿 ✓ —— 两个口径不一致才是真病根 ✓）。
        report = weights_mod.load_module_weights(module, path, dtype=want,
                                                 key_map=vae_music3_mod.KEY_MAP)
        if not report.complete:
            raise TorchBackendUnavailable(
                f"Music3 DAV 权重与架构**不符** ✗（缺 {len(report.missing)} 键 ✓ / "
                f"形状不符 {len(report.shapeMismatch)} 处 ✓，如 {report.missing[:4]} ✓）"
                f"　⇒ 拒 ✗（不许 `strict=False` 糊过去 ✓✗）", reason="pending")
        self._vae = module.to(self._device).eval().requires_grad_(False)
        self._vaeConfig = candidate
        self._vaeKind = "music3-real"
        #: ⚠️ 反缩放恒等 ✓ ⇒ 清掉上一支留下的 mean/std 账 ✓✗（否则 `decode` 里按它算会
        #: 给波形套上一个毫不相干的常数 ✓✗）。
        self._vaeLatents = None
        return {
            "kind": "music3-real",
            "complete": True,
            "path": str(path),
            "dtype": str(want),
            "loadedKeys": int(report.loadedKeys),
            "sourceKeys": int(report.sourceKeys),
            "latentChannels": int(spec["latentChannels"]),
            "sampleRate": int(spec["sampleRate"]),
            "upscaleRatio": int(spec["upscaleRatio"]),
            "scaleFactor": float(spec["scaleFactor"]),
            "note": (f"**真** Music3 DAV ✓（{int(report.loadedKeys)} 键逐键对上 ✓）＋ 反缩放**恒等** ✓"
                     f"（既不是 mean/std 组、也不是 Flux 常数 ✗）、{int(spec['upscaleRatio'])}× 上采样 ⇒"
                     f" {int(spec['sampleRate'])} Hz 立体声 ✓（T 潜帧 ⇒ {int(spec['upscaleRatio'])}·T 样本 ✓）"),
        }

    def attach_audio_vae(self, config: Any = None, *,
                         weights: str | None = None) -> dict[str, Any]:
        """挂上 :mod:`app.services.engine.audio_vae` 的音频 VAE ✓ ⇒ 双流的音频侧能解出波形 ✓。

        ⚠️ 它**与视频 VAE 毫无关系** ✗（DAC 血统编码 + BigVGAN 解码 ✓ —— 见
        `vae.H3_AUDIO_VAE_FACTS` ✓）：H3 的音频侧**不能**用视频 VAE 顶替 ✓✗（会得到"看着接上了"的
        假接线 ✗）。

        ``weights``（2026-11-04 新增 ✓，**默认不给** = 参考实现 ✗）：
        给 **H3 真音频 VAE**（``minimax_h3_audio_vae_fp32.safetensors`` ✓ **917 键** ✓）⇒ 音频侧
        出的是**真波形** ✓、``kind="h3-real"`` ✓（:meth:`_decode_dual` 的 ``audio_real`` 因此为真 ✓、
        ``synthetic`` 才有机会为 False ✓）。

        ⚠️ 三条纪律与 :meth:`attach_vae` 的真分支**逐条同形** ✓✗（当初真视频 VAE 踩过、这里一次写对 ✓）：
        1. **先核后改** ✓（``_check_attached_vaes`` 不合格 ⇒ 模块一个字没动 ✓）；
        2. **严格装载** ✓（:func:`weights.load_module_weights` 的 ``complete`` 为假**当场抛** ✗ ——
           917 键里少一个就是随机初始化的层，出的是"半真半噪"的音，而**一声不响** ✓✗）；
        3. **报告说真话** ✓（``kind`` 由**权重来源**决定 ✓，不按"参数看着像真的"猜 ✗）。
        ⚠️ dtype **跟着文件走** ✗（不给 ``dtype`` ✓）：音频 VAE 总共 0.56 GiB fp32 ✓，
        转 bf16 省不下什么却会改变 BigVGAN 的输出刻度 ✓✗。
        """
        self._gate()
        config = config or audio_vae_mod.AudioVAEConfig()
        self._check_attached_vaes(audio=config)   # ⭐ 同上：**先核后改** ✓（音频那一档 ✓）
        if str(weights or "").strip():
            path = Path(str(weights))
            if not path.exists():
                raise TorchBackendUnavailable(
                    f"H3 音频 VAE 权重不存在 ✗：{path} ✓（音频侧要的是 **BigVGAN 解码**那份 ✓，"
                    f"视频 VAE 顶替不了 ✓✗）", reason="pending")
            module = audio_vae_mod.build_audio_vae(config)
            load_report = weights_mod.load_module_weights(module, path)
            if not load_report.complete:
                raise TorchBackendUnavailable(
                    f"H3 音频 VAE 权重与架构**不符** ✗（缺 {len(load_report.missing)} 键 ✓ / "
                    f"形状不符 {len(load_report.shapeMismatch)} 处 ✓，"
                    f"如 {list(load_report.missing)[:4]} ✓）⇒ 拒 ✗（不许 ``strict=False`` 糊过去 ✓✗）",
                    reason="pending")
            #: ⚠️ 与视频侧同一条坑 ✓✗：不 ``eval()`` 也不冻 ⇒ 解码在建图 ⇒ BigVGAN 的 norm 权重
            #: 默认取梯度 ⇒ 整段上采样中间激活被钉住（`decode` 那一拍正好是"卡空着"✓，
            #: 所以症状不是 OOM ✗ 而是"解码变慢且退完不还"✓）。
            self._audioVae = module.to(self._device).eval().requires_grad_(False)
            self._audioVaeConfig = config
            self._audioVaeKind = "h3-real"
            report = config.to_dict()
            report.update({"kind": "h3-real", "complete": True,
                           "weights": str(path), "keys": int(load_report.loadedKeys)})
            return report
        self._audioVae = audio_vae_mod.build_audio_vae(config).to(self._device).eval()
        self._audioVaeConfig = config
        self._audioVaeKind = "reference-untrained"
        report = config.to_dict()
        report["kind"] = "reference-untrained"
        return report

    # ── 阶段互斥装载（Qwen-Image 2.1 ✓：三件同时驻留**装不下** ✓）──────────────
    #: 三件权重**反量化到 bf16 之后**的驻留量 ✓（按真文件 shape 乘积算 ✓ = 参数量 × 2 字节 ✓，
    #: **不是**盘上的 int8 字节数 ✗✗ —— 那只有它的一半 ✓）：
    #: DiT **13.26 GiB** ＋ TE **14.79 GiB** ＋ VAE **0.63 GiB** ＝ **28.68 GiB** ✗✗，
    #: 而 A5000 可用只有 **22.49 GiB**（连 TE+DiT 的 28.05 都过不去 ✗）。
    #: ⇒ 「四格齐全就一次装完」这条路在 24GB 卡上**必然 OOM** ✓✗，且 OOM 落在**最后一件**
    #: ⇒ 报错指向 VAE ✓✗（读起来像"VAE 装不上" ✓，实际是三件之和 ✓）。
    #: 好在这三段**本来就不交叠** ✓（encode 只用 TE ✓、sample 只用 DiT ✓、decode 只用 VAE ✓）
    #: ⇒ 按阶段**互斥换装** ✓，峰值 = max(14.79, 13.26, 0.63) ≈ **14.8 GiB** ✓（余 7.7 GiB ✓）。
    #: ⚠️ 不走 `residency` 的 split/park ✗：那是给「**单件**都装不下」的分层搬运 ✓ ——
    #: 逐层跨 CPU↔GPU 搬比换装慢一个量级 ✓，而这里换装**零跨界** ✓。
    PHASE_PARTS: dict[str, tuple[str, ...]] = {
        "encode": ("text",),
        #: ⭐ **图生图的起点**也要 VAE ✓✗（2026-11-02 真跑逼出来的 ✓）：`init_latents` 要把
        #: `initImage` 编成起点 latent ✓ ⇒ 那一拍手上必须有 VAE ✓。上面三段**漏了它** ✗✗ ——
        #: 文生图永远走不到 ✓（请求里没图 ✓），**图生图一走就报**「没挂 VAE ✗ ⇒ 没法把图片
        #: 编码成潜变量」✓（实测：2512 storyboard-last / denoise=0.55 ✓，装载与采样都正常 ✓，
        #: 只在 init 这一拍断 ✓）。单元测试里 VAE 是**常驻替身** ✓ ⇒ 永远试不出来 ✓✗。
        #: ⚠️ 只在**真要编码图片**时申请 ✓（`pipeline` 判 `request.initImage` ✓）：无条件加会
        #: 让纯文生图每次多一趟 VAE 换装 ✓✗（VAE 0.24 GiB 从盘读 ✓ 白花几秒 ✓）。
        #: ⚠️ **text 也写进来** ✓✗：`enter_phase` 会退掉「不在 wanted 里的件」✓ ⇒ 只写 vae
        #: 就会把刚 encode 完的 TE（16.1 GiB）退掉 ✓✗，紧接着 `sample` 又得从盘重读一遍 ✓。
        #: 两者同驻 = 16.1 ＋ 0.24 ＝ **16.34 GiB** ✓ < 22.49 ✓ 装得下 ✓ ⇒ 留着更省。
        "init": ("text", "vae"),
        "sample": ("dit",),
        "decode": ("vae",),
    }

    def declare_chain(self, *, dit_path: str, text_weights: str,
                      tokenizer_path: str | None = None, vae_path: str,
                      vae_family: str = "qwen_image21",
                      text_kind: str = "qwen35",
                      phase_parts: dict[str, tuple[str, ...]] | None = None,
                      part_need_gib: dict[str, float] | None = None,
                      keep_parts: tuple[str, ...] | None = None) -> dict[str, Any]:
        """登记一条**三件不同时驻留**的完整链 ✓ ⇒ 之后 :meth:`enter_phase` 按阶段换装 ✓。

        为什么不在这里一次装完 ✗✗：见 :attr:`PHASE_PARTS` 上面的账（28.68 GiB > 22.49 GiB ✓）。

        不登记链时（其它族 ✓），:meth:`enter_phase` **一律不做事** ✓ —— 零影响 ✓。

        ⭐ **两个按族参数**（2026-10-31 加 ✓，接 2512 时逼出来的 ✓）：

        * ``text_kind`` ✓ —— ``"qwen35"``（2.1 ✓ Qwen3.5-9B ⇒ :meth:`attach_qwen35_te` ✓）、
          ``"qwen25"``（2512 ✓ Qwen2.5-VL-7B ⇒ :meth:`attach_qwen25_te` ✓）、
          ``"h3"``（双流 ✓ 语言＋视觉两半 ⇒ :meth:`attach_h3_text` ✓）、
          ``"krea2"``（Krea2 ✓ Qwen3-VL-4B **只挂语言塔** ＋ 12 层隐状态栈 ⇒ :meth:`attach_krea2_te` ✓）、
          ``"umt5"``（Wan2.1/2.2 ✓ UMT5-XXL **纯 encoder** ⇒ :meth:`attach_umt5_te` ✓）。
          ⚠️ 这几个**互不可代替** ✗✗：骨架不同 ✓、取层口径不同 ✓（末层隐状态 vs 12 层栈 ✓
          vs encoder 末层过 final norm ✓）、模板不同 ✓（Krea2 要 ``<|im_start|>system…<|im_start|>assistant\\n``
          并把前缀裁掉 ✓；UMT5 **不套模板** ✓）。
          ⚠️⚠️ **漏写一格就落到 ``qwen35`` 兜底** ✓✗（``umt5`` 2026-11-02 才补上 ✓）：真跑报的是
          「缺 ``model.embed_tokens.weight`` ⇒ 这不是 Qwen3.5 的文本塔」✓ —— 措辞对 ✓、
          但指不到真因（分派表少一格 ✗），读起来像"权重下坏了" ✓✗。
          ⚠️ 原来 ``enter_phase`` 里**写死** ``attach_qwen35_te`` ✓✗ ⇒ 2512 的
          ``qwen_2.5_vl_7b_fp8_scaled`` 会被 Qwen3.5 的骨架吃 ✓✗：报的是「缺 `linear_attn.*`」
          ✓（读起来像"权重下错了" ✗，实际是把另一族的塔接了上来 ✓✗）。
        * ``phase_parts`` ✓ —— 本族的**分段表** ✓（缺省 = :attr:`PHASE_PARTS` ✓，即
          ``encode→text`` / ``sample→dit`` / ``decode→vae`` ✓）。H3 用
          ``{"encode": ("text",), "sample": ("dit",), "decode": ("vae",)}`` ✓：它的真 VAE 在
          **装配期**就挂在 cpu 上 ✓（5.2 GiB 不占卡 ✓）⇒ ``decode`` 那一段**不重读** ✓，只把
          常驻的它搬上卡 ✓（`keep_parts` 在 `present` 里命中 ⇒ 跳过装载 ✓，见 `_relocate_keep_parts` ✓）。
          ⚠️ 表里**没有**的阶段 ⇒ :meth:`enter_phase` 返回 ``None`` ✓（不换装 ✓，不是报错 ✗）。
        * ``part_need_gib`` ✓ —— 本族的驻留账 ✓（缺省 = :attr:`PART_NEED_GIB` ✓）。
          2512 的 DiT 驻留 **38.05 GiB** ✓（20.4B × 2 字节 ＋ 非参缓冲 ✓；2.1 是 13.26 ✓，
          差 24.8 GiB ✓✗）：拿错表判 ⇒ 决策说"装得下" ✓、`residency` 留 ``off`` ✓
          （= 不用切 ✓）、装到一半 OOM ✓。
        * ``keep_parts`` ✓ —— **常驻件** ✓✗（2026-11-01 加 ✓，H3 端到端逼出来的 ✓）：在整个
          生命周期里**既不装也不卸**的那些件 ✓。它们**不进** ``phase_parts`` ✗✗ —— 一进就坏两件事：
          ① ``decide_for`` 会把它们算进"这一阶段要占多少"（H3 的 VAE 5.2 GiB 是 **cpu 常驻** ✓，
          算进卡上需求 ⇒ encode 阶段凭空多要 5.2 GiB ⇒ 判"装不下" ⇒ 整座 25.44 GiB 的 TE
          被推去 CPU ✓✗）；② 它们在场的阶段会被当成"装好了"去 `reused` 判定 ✓ 混乱 ✓。
          ⚠️ 但**不声明**的话，:meth:`enter_phase` 退"不在 wanted 里的件"时就会把常驻件退掉 ✓✗ ——
          H3 实测正是如此 ✓：采样跑完 4 步 ✓，`decode` 时 `_vae is None` ✓，报
          「双流解码缺 视频 VAE ⇒ attach_vae()」✓✗（读起来像没装配 ✓，其实是被 `sample`
          那一拍自己卸的 ✓✗）。⚠️ 常驻件只许是**不占卡**的件 ✗（本机制不动设备 ✓）。
        """
        table = dict(part_need_gib) if part_need_gib else dict(self.PART_NEED_GIB)
        self._chain = {
            "ditPath": dit_path,
            "textWeights": text_weights,
            "tokenizerPath": tokenizer_path,
            "vaePath": vae_path,
            "vaeFamily": vae_family,
            "textKind": text_kind,
            "phaseParts": ({key: tuple(value) for key, value in phase_parts.items()}
                           if phase_parts else None),
            #: ⚠️ **常驻件** ✓✗（只许是**不占卡**的件 ✓）：`enter_phase` 退件时绕开它们 ✓、
            #: `release_for_pressure` 也不许动它们 ✓；它们**不进** `phaseParts` ✗✗（理由见 docstring ✓）。
            "keepParts": tuple(keep_parts or ()),
            "partNeedGiB": table,
        }
        #: ⚠️ `peakGiB` **由本族的账算** ✓✗（原来写死 ``14.79`` ✓✗）：峰值就是
        #: 「最重的那一件」✓，而三件的账只该有**一处**真源 ✓ —— 写死的话，将来某件权重变大、
        #: 常量跟着改了，这一格却还报旧值 ✓✗（读者会按它估余量 ⇒ 估错 ✓✗）。
        peak = round(max(table.values()), 2) if table else 0.0
        return {"phased": True, "peakGiB": peak, "chain": dict(self._chain),
                #: ⭐ 决策口径 ✓：默认 `auto` = GPU 优先、每阶段现测现定 ✓（见 `device_policy` ✓）。
                #: ⚠️ 把「按什么判的」一起报出去 ✗：只说"在 CPU 上跑"会让人以为是不支持 GPU ✓✗。
                "policy": self._policy, "partNeedGiB": table}

    def _trim_cuda(self) -> int:
        """把 allocator 里**空闲**的块还给驱动 ✓ ⇒ 返回还掉多少字节 ✓✗。

        为什么必须单独有一条 ✗✗（2026-11-01 端到端实测 ✓）：``torch.cuda.empty_cache()`` 只还
        **空闲**块 ✓，而装载/切分过程中会顺手分配大量**临时**（逐键搬权重时的中转 ✓）✓ ——
        搬完临时就死了 ✓、块却留在 allocator 里 ✓ ⇒ ``reserved`` 比 ``allocated`` 高出十几 GiB ✓✗。
        而 :func:`device_policy.probe` 量的是 ``mem_get_info`` 的 **free**（驱动视角 ✓）⇒
        那些"看着空闲、其实没还"的块会让它报出**假的紧** ✓✗ ⇒
        `decide_for` 判"余 4.92 GiB" ⇒ 下一件只切 3.5 GiB 上卡 ✓、其余挤内存 ✓
        ⇒ 越换越挤 ✓、最后卡还被 WDDM 超分（``reserved`` 一度 24252 MiB > 卡 23027 MiB ✓✗）。

        实测（同一台机、同一份权重 ✓）：`enter_phase("encode")` 退掉 DiT 之后
        ``allocated`` 16377 → 3550 MiB ✓，``reserved`` 却只从 20174 → 17936 ✓✗ ——
        多出来的 14.3 GiB 就是没还的缓存块 ✓，全被下一轮的探测当成"卡上已经没了" ✗。
        """
        try:
            import torch  # noqa: PLC0415
        except Exception:  # noqa: BLE001 —— 没装 torch 就是没显存可还 ✓
            return 0
        try:
            if not torch.cuda.is_available():
                return 0
            before = int(torch.cuda.memory_reserved())
            #: ⚠️ **必须先 synchronize** ✓✗（2026-11-01 实测 ✓）：allocator 只肯还「所属 stream
            #: 已经没有在途工作」的块 ✓；装载刚做完时搬权重的 copy 还挂在 stream 上 ✓ ⇒
            #: 直接 `empty_cache()` **一个字节都不还** ✓✗（实测：装完 DiT 后活着的 tensor 只有
            #: 4.02 GiB ✓、`reserved` 却 36402 MiB ✓ —— 那 32 GiB 全是"空闲但还在途" ✓）。
            #: ⚠️ **一轮也不够** ✓✗：`synchronize()` 只同步当前 stream ✓，别的 stream（pin/拷贝
            #: 用的）要等它自己的 event 结算 ✓ ⇒ 一次 `empty_cache()` 放一半、下一次才放另一半 ✓。
            #: 实测同一时刻连清两轮：20390 → **4332** MiB（中间那轮只放掉了 16.2 GiB 里的一部分 ✓）。
            #: 换装是**低频**动作（每阶段一次 ✓），多清两轮的毫秒级代价换真显存，值 ✓。
            current = before
            for _ in range(3):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                latest = int(torch.cuda.memory_reserved())
                if latest >= current:      # 这一轮没再放出东西 ⇒ 到此为止 ✓（不空转 ✓）
                    break
                current = latest
            after = current
            return max(0, before - after)
        except BaseException:  # noqa: BLE001 —— 清理动作**绝不许**把主流程带崩 ✓
            return 0

    def _drop_residency_handle(self) -> None:
        """摘掉旧的切分把手 ✓✗ —— **必须 `detach()`，光置 None 不算数** ✓。

        ``residency.Handle`` 攥着 ``module`` ✓、它的 hook 闭包又攥着 node 与 handle ✓；
        ``detach()`` 把 hook 摘掉之后 ✓，模块才可能被 `gc.collect()` 收掉 ✓，
        卡上那份权重才会随 `torch.cuda.empty_cache()` 真还回去 ✓。
        只把 ``self._residencyHandle`` 置 None ⇒ 环还在 ✓、模块活着 ✓、显存一点不退 ✓✗
        （2026-11-01 端到端实证：`text` 卸完卡上仍占 14.8 GiB ✓，DiT 因此被迫整份落内存 ✓）。
        """
        holder, self._residencyHandle = self._residencyHandle, None
        #: ⚠️ 名字也要跟着清 ✓✗：留着的话，下一个把手会被误判成「上一件摆的」✓（见 `release_part` ✓）
        self._residencyOwnerWhat = None
        if holder is not None:
            try:
                holder.detach()
            except BaseException:  # noqa: BLE001
                pass

    def release_part(self, part: str) -> dict[str, Any]:
        """卸掉**一件** ✓（``dit`` / ``text`` / ``vae`` ✓）—— 真的放引用并清缓存 ✓。

        ⚠️ **四步**缺一不可 ✗：``_residencyHandle.detach()`` ＋ **置 None** ＋ ``gc.collect()`` ＋
        ``torch.cuda.empty_cache()`` ✓。只置 None 时显存不还 ✗（`empty_cache` 只还**空闲**块 ✓）；
        只做到「置 None + gc」也还是**不还** ✗✗ —— 因为 `residency.Handle` 自己就攥着
        ``self.module`` ✓、hook 闭包又攥着 handle 与 node ✓ ⇒ 形成环 ✓，
        `gc.collect()` 收不掉一个**被外部活引用**的 handle ✓✗（`self._residencyHandle` 就是那个活引用 ✓）。
        2026-11-01 真权重端到端实测：`text` 卸掉后卡上那 14.8 GiB **一字节没退** ✓✗
        ⇒ 紧接着装 DiT 时预算≈0 ⇒ 整份 17.94 GiB 落到内存 ✓ ⇒ 宿主 71 GiB / 卡 22.5 GiB 卡死 ✓。
        """
        target = {"dit": self._model, "text": self._textEncoder,
                  "vae": self._vae, "vision": self._visionTower}.get(part)
        if part not in ("dit", "text", "vae", "vision"):
            raise ValueError(f"不认识的分时部件 {part!r} ✗（只认 dit / text / vae / vision ✓）")
        #: ⚠️ **只退「这一件」的把手** ✓：`_residencyHandle` 是单槽 ✓，它对应**最后装的那件** ✓；
        #: 若正好就是要退的这件 ✓（`module is target` ✓）⇒ 必须先 detach 才有资格谈释放 ✓✗。
        #: ⭐ **两条判据，缺一不可** ✓✗（2026-11-07 ✓）：
        #: ① 同一性（``holder.module is target`` ✓）—— DiT 那种"摆的就是它本身" ✓；
        #: ② **名字**（`self._residencyOwnerWhat` 归一到 `part` ✓）—— 文本塔那种"摆的是**本体**、
        #:    存的是**外壳**"✓✗（`attach_krea2_te` 摆 ``module.tower`` ✓、``_textEncoder`` 是 ``module`` ✓）。
        #:    只有 ① 时 ② 恒不成立 ✓ ⇒ 把手不摘 ⇒ 钩子环吊着每一层 ⇒ **卡上一字节不退** ✓✗。
        #: ⚠️ 用 `getattr` 兜底 ✓✗：一堆测试替身是 `__new__` 出来的（不跑 `__init__` ✓），
        #: 写 `self._residencyOwnerWhat` 会让它们 `AttributeError` ⇒ 判据没跑就先炸 ✓（§9 教训）。
        owner = getattr(self, "_residencyOwnerWhat", None)
        same_part = owner is not None and _WHAT_TO_PART.get(owner, owner) == part
        holder = self._residencyHandle
        if holder is not None and target is not None \
                and (getattr(holder, "module", None) is target or same_part):
            self._drop_residency_handle()
        #: ⚠️⚠️ **两个局部引用必须当场放掉** ✓✗（2026-11-01 实测 ✓，这一句是本轮的关键 ✓）：
        #: ``target`` 就是那座要退的塔 ✓、``holder.module`` 又指回它 ✓ —— 只要这两个名字还在帧里 ✓，
        #: 下面的 `gc.collect()` 收不掉 ✓、`empty_cache()` 就认定那些块「有人在用」✓ ⇒ **一字节不还** ✓✗。
        #: 实测症状：`release_part("text")` 明明 `released: True` ✓、`_textEncoder` 已置 None ✓，
        #: 可下一次 `decide_for` 仍读到「余 5.44 GiB」（真实可用 21 GiB ✓）⇒ 17.94 GiB 的 DiT
        #: 只被允许上卡 4 GiB ✓、其余挤进内存 ✓ ⇒ 宿主 71 GiB 卡死 ✓。
        del holder, target
        if part == "dit":
            existed = self._model is not None
            self._model = None
            #: ⚠️ ``_config`` / ``_ditFamily`` **故意留着** ✓✗：它们是「权重反推出的结构事实」
            #: （``vae_scale`` / ``in_channels`` / 族名 ✓，几 KB ✓）⇒ `latent_shape` 靠它们 ✓。
            #: 一起清掉的话，`init_latents` 会在「DiT 已退、TE 在场」那一刻静默造占位 ✓✗
            #: （症状见 `latent_shape` 的注释 ✓）。显存全在 ``_model`` 里 ✓，留着这两样不退一字节 ✓。
        elif part == "text":
            existed = self._textEncoder is not None
            self._textEncoder = None
            self._tokenizer = None
            self._textConfig = None
            self._textKind = None
        elif part == "vae":
            existed = self._vae is not None
            self._vae = None
            self._vaeConfig = None
            self._vaeKind = None
            self._vaeLatents = None
        elif part == "vision":
            #: ⭐ Edit 2511 的**视觉塔** ✓：只在 encode 段在场 ✓ ⇒ 采样前后各装一次 ✓✗
            #: （1.35 GiB 不贵 ✓，但它和 38 GiB 的 DiT 抢卡时会直接把 DiT 挤去内存 ✓）。
            existed = self._visionTower is not None
            self._visionTower = None
            self._visionShape = None
            self._visionReport = None
        #: 账本同步还回去 ✓（**无论有没有真退**都清 ✓：两处不一致比不记更糟 ✓✗）
        #: ⚠️ 键名**不只一个** ✗✗：`reserve` 那侧写的是 ``what``（``"dit"`` / ``"h3"`` / ``"te"`` /
        #: ``"h3-te"`` ✓），只退 ``part`` 会退了个空 ✓✗ —— 账本只涨不退，`residencyTotal` 里那件
        #: 永远还在 ✓（假 8 GiB 卡实测：退 text 之后仍报「已占 6.41 GiB」✓✗）。按 `_WHAT_TO_PART`
        #: 的**同名项**全退一遍 ✓（不存在的键 `release` 是空操作 ✓）。
        for alias in {part, *(name for name, owner_part in _WHAT_TO_PART.items()
                              if owner_part == part)}:
            self._ledger.release(alias)
            self._ledger.release(f"{alias}#host")
        if existed:
            import gc  # noqa: PLC0415

            gc.collect()
            #: ⚠️ 走 `_trim_cuda()` 而不是裸 ``empty_cache()`` ✓：同一个动作只许有一份实现 ✓✗
            #: （顺带把「还了多少」留成返回值 ✓，报告里能看出这一步到底退干净没有 ✓）。
            self._trim_cuda()
        return {"part": part, "released": existed}

    @staticmethod
    def _module_device(module: Any) -> str | None:
        """模块**现在躺在哪台** ✓ ⇒ 认不出（无参无缓冲 ✓ 或不是 Module ✓）返回 ``None`` ✓。

        为什么不用 ``module.device`` ✗：``nn.Module`` **没有**这个属性 ✓✗（只有张量有 ✓），
        于是"想当然写 `module.device`"的代码会在第一次运行时 `AttributeError` ✓，
        而不是"读到一个错设备" ✓ —— 后者才是真正难查的那种 ✗。
        """
        if module is None:
            return None
        try:
            for parameter in module.parameters():
                return str(parameter.device)
        except Exception:  # noqa: BLE001
            return None
        try:
            for buffer in module.buffers():
                return str(buffer.device)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _pinned_gib(self, part: str) -> int:
        """这件**占了多少字节** ✓ ⇒ 读账本里现成的那一条 ✓（读不到就算 0 ✓）。

        ⚠️ 账本的键是**摆放时的 `what` 名** ✓✗（见 :attr:`_WHAT_TO_PART` ✓）：文本塔摆的时候叫
        ``te`` ✓，而这里查的是部件名 ``text`` ✓ —— 只按一个名字查 ⇒ 读到 0 ✓ ⇒ 搬完记账记成
        **0 字节** ✓✗（账本说"没占"、人却看着 7.5 GiB 躺在内存里 ✓）。所以按**别名**都试一遍 ✓。
        """
        names = [part, *(alias for alias, target in _WHAT_TO_PART.items() if target == part)]
        for name in names:
            entry = self._ledger.items.get(name) or self._ledger.items.get(f"{name}#host")
            bytes_ = int(getattr(entry, "bytes", 0) or 0)
            if bytes_:
                return bytes_
        return 0

    def _relocate_keep_parts(self, kept: tuple[str, ...], wanted: tuple[str, ...],
                             *, to_host_only: bool = False) -> list[dict[str, Any]]:
        """常驻件的**阶段定位** ✓✗ —— 这一阶段要它 ⇒ **上卡** ✓；不要 ⇒ **回 cpu** ✓。

        ## 为什么必须"能上去"（2026-11-01 端到端实测 ✓）

        H3 的**真视频 VAE 装配期只能落 cpu** ✓（5.2 GiB 与 DiT 的 19.62 GiB 抢同一块 22.49 GiB 卡 ✓✗），
        而它又是**全程常驻**（`keep_parts` ✓：不重读 5.2 GiB ✓）。只做到"不被卸掉"就够跑了 ✗，
        但代价是 **`decode` 整段都在 cpu 上算** ✓✗：实测 320×320 × 34 帧解了 **94.6 秒** ✓
        （同一份 VAE 在卡上只要个位数秒 ✓）—— 换成真业务尺寸（720p × 上百帧 ✓）就是"跑一夜" ✓✗。

        而 `decode` 这个阶段恰好是**卡全空**的 ✓（DiT 已退 ✓）⇒ 5.2 GiB 上卡毫无压力 ✓；
        解完再回 cpu ✓（`sample` 阶段要 19.62 GiB 的 DiT ✓，那时它若还赖在卡上就是抢位 ✓✗）。

        ⚠️ **不重读、只搬** ✓：``module.to(device)`` 是原地换参数张量 ✓，权重从
        `attach_vae` 那次装载的张量里直接过 PCIe ✓（5.2 GiB ≈ 半秒 ✓），不是再读盘一遍 ✓。

        ⚠️ 账本要跟着搬 ✓✗：本仓约定「卡上那部分记 ``<件>``、内存那部分记 ``<件>#host``」✓ ——
        搬完不记账，`schedulingAfter.ledger` 就会说"vae 在 cpu" ✓ 而它正躺在卡上 ✓✗。
        """
        modules = {"vae": self._vae, "text": self._textEncoder, "dit": self._model}
        moved: list[dict[str, Any]] = []
        for part in kept:
            module = modules.get(part)
            if module is None:
                continue
            current = self._module_device(module)
            if current is None:
                continue
            target = str(self._device) if part in wanted else "cpu"
            #: ⭐ ``to_host_only`` ＝ **只做"让开"那一半** ✓✗（2026-11-07 ✓）：调用方要**先腾卡
            #: 再决策** ✓ ⇒ 这一步只把"这一拍用不着的"往 cpu 挪 ✓，上卡留到装完之后那次调用 ✓
            #: （反了就成"往满的卡里塞" ✓✗，见本方法末尾那条顺序说明 ✓）。
            if to_host_only and device_policy.is_cuda_device(target):
                continue
            if current == target:
                continue
            if (device_policy.is_cuda_device(current)
                    and device_policy.is_cuda_device(target)):
                continue          # 同族不同号（cuda / cuda:0 ✓）—— 不值得为这点差异搬 5 GiB ✗
            gib = self._pinned_gib(part)
            try:
                module.to(target)
            except Exception as err:  # noqa: BLE001 —— OOM 之类要**响亮**地报成"换装失败" ✓
                raise TorchBackendUnavailable(
                    f"常驻件 `{part}` 从 {current} 搬到 {target} 失败 ✗："
                    f"{type(err).__name__}: {err}\n"
                    f"⚠️ 这一阶段{'要' if part in wanted else '不要'}它 ✓ ⇒ 不搬的后果是"
                    f"『该在卡上算的留在 cpu』✓（慢一个量级 ✓ 不报错 ✓✗）",
                    reason="pending") from err
            self._ledger.release(part)
            self._ledger.release(f"{part}#host")
            if device_policy.is_cuda_device(target):
                self._ledger.reserve(part, bytes_=gib, device=target,
                                     note="常驻件按阶段上卡（decode 用）✓")
            else:
                self._ledger.reserve(f"{part}#host", bytes_=gib, device="cpu",
                                     note="常驻件的 cpu 那一半 ✓")
            moved.append({"part": part, "from": current, "to": target,
                          "GiB": round(gib / device_policy.GIB, 2)})
            #: 搬完要**立刻**把旧位置那几 GiB 还掉 ✓（``module.to`` 已就地换了参数张量 ✓，
            #: 旧块只是 allocator 手里的缓存 ✓ ⇒ `empty_cache` 才真还 ✓）。
            self._trim_cuda()
        return moved

    def enter_phase(self, phase: str) -> dict[str, Any] | None:
        """进某个阶段前**把不用的那件让出去、把要用的那件装进来** ✓（没登记链 ⇒ 返回 None ✓）。

        ``phase`` 取表里的名字（:attr:`PHASE_PARTS` ✓＝ ``encode`` / ``init`` / ``sample`` /
        ``decode`` ✓）；**不在表里的**名字（``plan`` / ``refine`` ✓）不换装 ✓ —— 2.1 的超清
        二采仍用同一个 DiT ✓（它在 ``sample`` 阶段装着呢 ✓）。
        ⚠️ ``init`` 是**图生图专属**那一段 ✓（`pipeline` 只在给了 ``initImage`` 时才调 ✓，
        见 :attr:`PHASE_PARTS` 里那段账 ✓）。

        ⚠️ **失败要炸在这里** ✓（管线按阶段记错 ✓）：换装不成 ⇒ 后面的 encode/sample/decode
        没有可用权重 ✓，让它"跑下去"只会得到一句更远的报错 ✓✗。
        """
        if self._chain is None:
            return None
        #: ⚠️ **按链覆盖** ✓✗（H3 ✓）：它的 VAE 在**装配期**就挂好了 ✓（真权重落 cpu ✓，
        #: 5.2 GiB 不占卡 ✓）⇒ 「decode 阶段换装 VAE」对它是**白读一遍 5.2 GiB** ✗ ⇒
        #: 这条链只声明 ``encode`` / ``sample`` 两段 ✓；``decode`` **不在表里** ⇒ 不换装 ✓。
        wanted = (self._chain.get("phaseParts") or self.PHASE_PARTS).get(phase)
        if wanted is None:
            return None
        #: ⚠️ **常驻件** ✓✗（`declare_chain(keep_parts=…)` ✓）：H3 的真 VAE 装配期就挂在 cpu 上 ✓、
        #: 全程不换装 ✓ —— 不在这里绕开它，`sample` 那一拍就会把它当"用不着"卸掉 ✓✗，
        #: 采样跑完到 `decode` 才报「缺 视频 VAE」✓✗（2026-11-01 实测 ✓）。
        kept = tuple(self._chain.get("keepParts") or ())
        present = {"dit": self._model is not None,
                   "text": self._textEncoder is not None,
                   "vae": self._vae is not None,
                   #: ⭐ Edit 2511 的**视觉塔** ✓（只有 Edit 链的 encode 段要它 ✓）
                   "vision": self._visionTower is not None}
        reused = [part for part in wanted if present[part]]
        #: ⭐⭐ **常驻件里这一拍用不着的，先"停到内存"** ✓✗（2026-11-07 ✓）—— 必须排在
        #: 「测卡况 → 决策 → 装下一件」**之前** ✗✗：它还压在卡上时测出来的余量是"假的宽" ✓
        #: ⇒ 会把下一件判成"整份装得下" ✓ ⇒ 真装的时候撞上它 ✓✗（Krea2 实测口径：text 7.5 GiB
        #: 还在卡上 ⇒ dit 12.57 GiB 被判 `off` ⇒ 装到 20 GiB 时 OOM ✓）。
        #: 「上卡」那一半**留到装完之后** ✓（`_relocate_keep_parts` 末尾那次 ✓）—— 顺序反了就是
        #: 往满的卡里塞 ✓✗。⚠️ 两个分支都要有 ✗✗：`reused` 走的是**早退**那条 ✓，
        #: 而它同样会"边搬上去、边被占着的件堵着" ✓✗。
        parked = self._relocate_keep_parts(kept, wanted, to_host_only=True)
        if len(reused) == len(wanted):
            #: 要的已在场 ✓ ⇒ 只把**不用的**退掉 ✓（能退多少退多少 ✓；常驻件不算"用不着" ✓）
            released = [self.release_part(part) for part in present
                        if present[part] and part not in wanted and part not in kept]
            #: ⚠️ 已在场就**别再决策** ✗✗：重装一遍只为了换设备，等于把 14 GiB 白搬一次 ✓✗
            #: （`decode` 之后回 `sample` 那种回头路是少数 ✓，那时该由调用方决定要不要重装 ✓）。
            #: ⚠️ 但**要搬设备** ✓✗：H3 的 `decode` 正是"VAE 已在场（cpu ✓）而这一拍要它上卡"✓ ——
            #: 不搬就整个解码在 cpu 上跑 ✓（实测 94.6 秒 ✓，见 `_relocate_keep_parts` ✓）。
            relocated = parked + self._relocate_keep_parts(kept, wanted)
            return {"phase": phase, "reused": reused, "released": released,
                    "relocated": relocated, "decision": None, "device": self._device,
                    "residency": self._residency}
        #: ⚠️ 退的是「**既不在 wanted、也不是常驻件**」✓✗ —— 不是「present 里全部」✗（2026-11-01 修 ✓）：
        #: `wanted` 里**已在场**的那件正是这一阶段要用的 ✓，退掉它等于"装完自己又卸了"✓✗；
        #: 常驻件更是不许碰 ✓（H3 的 VAE ✓）。原写法 ``[... for part in present if present[part]]``
        #: 在 2.1 这种"每次缺件时 wanted 里都空着"的族上**看不出差别** ✓✗（wanted 里的件恰好
        #: 都是缺的那件 ✓）—— 差别只在 H3 上露头 ✓：`sample` 一拍把常驻的 VAE 也退了 ✓✗。
        released = [self.release_part(part) for part in present
                    if present[part] and part not in wanted and part not in kept]
        # ⭐⭐ **退干净之后**才测卡况、才决策 ✓✗：退之前卡上还压着上一件 ✓，那时测出来的
        #     余量是"假的紧" ✓✗ ⇒ 会把自己越推越往 CPU（一旦降级就回不来 ✓✗）。
        #     顺序就是「退 → 测 → 定 → 装」✓，四步一步都不能挪 ✗。
        #: ⚠️⚠️ 决策只针对「**还没装的那几件**」✓✗（2026-11-07 ✓）—— 已在场的**不许再算一遍** ✗✗：
        #: `decide_for` 的 `need` 是"这几件加起来多大"✓，而 `free` 是**此刻**卡上余量 ✓ ——
        #: 已在场那件**本身就占着** `free` 里被扣掉的那几 GiB ✓ ⇒ 按整份 `wanted` 算就是把它
        #: 数了两遍 ✓✗ ⇒ 需求虚高 ⇒ 判成 `cpu/off` ⇒ 接着 `ensure_host_room` 要"整份进内存" ⇒
        #: **宿主不够 ⇒ 整条链拒装** ✓✗。实测（假 8 GiB 卡 ✓）：`init` 那拍 `text` 已在场（6.41 GiB ✓）、
        #: 只差 0.24 GiB 的 `vae` ✓，却因为按「text+vae=7.74 GiB」判 ⇒ 卡上只剩 0.38 GiB
        #: ⇒ 转 CPU ⇒ 要 7.74 GiB 内存（宿主只剩 3.59 GiB）⇒ 报「要整份放进内存」✓✗ ——
        #: **看着像内存不够，其实只差一个 0.24 GiB 的 VAE** ✓✗。
        missing = tuple(part for part in wanted if not present[part])
        decision = self.decide_for(missing, phase=phase,
                                   need_gib=self._chain.get("partNeedGiB"))
        self._apply_decision(decision)
        chain = self._chain
        for part in wanted:
            #: ⚠️ **已在场的不重装** ✓✗（同 `reused` 那条口径 ✓）：H3 的 VAE 全程常驻 ✓，
            #: 出现在 `wanted` 里只是"声明别卸我" ✓；照装一遍会白读 5.2 GiB 并把 cpu 那份顶掉 ✓✗。
            if present[part]:
                continue
            if part == "dit":
                self.load_weights(path=chain["ditPath"])
            elif part == "text":
                #: ⚠️ **按族分派** ✓✗（见 `declare_chain` 的 `text_kind` ✓）：两座塔的骨架、
                #: 取层、模板都不同 ✓，接错族报的是「缺 `linear_attn.*`」—— 读起来像权重坏了 ✗。
                kind = str(chain.get("textKind") or "qwen35")
                if kind in ("h3", "h3-te"):
                    #: H3 ✓：一个文件里**两半塔** ✓（语言 ＋ 视觉 ✓）⇒ 必须走 `attach_h3_text`
                    #: ✗✗（另两座是"只挂语言塔" ✓ 的口径 ✓，拿来接 32B 那份会把视觉塔当
                    #: unexpected 丢掉 ✓⇒ 参考图块没有特征 ✓、形状全对 ✓✗）。
                    self.attach_h3_text(chain["textWeights"],
                                        tokenizer_path=chain["tokenizerPath"])
                elif kind == "qwen25":
                    self.attach_qwen25_te(chain["textWeights"],
                                          tokenizer_path=chain["tokenizerPath"])
                elif kind in ("krea2", "krea2-te"):
                    #: Krea2 ✓：Qwen3-VL-4B **只挂语言塔** ✓ + 取 12 层隐状态栈 ✓ + 套
                    #: ``<|im_start|>…`` 模板并在编码后裁掉前缀 ✓（全在 `attach_krea2_te` 里 ✓）。
                    #: ⚠️ 不能落到下面那条 ``qwen35`` 兜底 ✗✗：那座塔按「最后一层隐状态」取 ✓，
                    #: 宽度 2560 ≠ 主干要的 30720 ⇒ 报「文本宽度对不上」✓，读起来像拿错塔 ✗，
                    #: 真因是「没取层栈」✓✗。
                    self.attach_krea2_te(chain["textWeights"],
                                         tokenizer_path=chain["tokenizerPath"])
                elif kind in ("umt5", "umt5-te"):
                    #: ⭐ **Wan 2.1/2.2 的 UMT5-XXL** ✓✗（2026-11-02 真跑抓到的第三处 ✓）：
                    #: 这一支原先**根本没写** ✓ ⇒ 落到下面的 ``qwen35`` 兜底 ✓ ⇒
                    #: 拿 Qwen3.5-9B 的骨架去读 UMT5 的文件 ✓ ⇒ 报「缺 ``model.embed_tokens.weight`` ✓
                    #: ⇒ 这不是 Qwen3.5 的文本塔」✓ —— 措辞完全正确 ✓、但**指不到真因** ✓
                    #: （真因是"分派表里少一格"✗，不是"权重坏了"✓）。
                    #: ⚠️ 别改成 `try: umt5; except: qwen35` ✗✗：两座塔宽度都可能是 4096 ✓，
                    #: 试错能"跑起来"✓ 而隐状态来源完全不同 ✓（UMT5 是 encoder 末层过 final norm 且
                    #: **不套 chat 模板** ✓）⇒ 出图照样有 ✓、只是条件全错 ✓✗。
                    self.attach_umt5_te(chain["textWeights"],
                                        tokenizer_path=chain["tokenizerPath"])
                else:
                    self.attach_qwen35_te(chain["textWeights"],
                                          tokenizer_path=chain["tokenizerPath"])
            elif part == "vae":
                if str(chain.get("vaeFamily") or "") == "h3":
                    self._attach_h3_vae_weights(chain["vaePath"])
                else:
                    self.attach_wan_vae(chain["vaePath"], family=chain["vaeFamily"])
            else:
                #: ⭐ **视觉塔** ✓✗（Edit 2511 ✓）：和语言塔**同一个文件** ✓（``visual.*`` 714 张量 ✓）
                #: ⇒ 路径直接复用 ``textWeights`` ✓，不另开一个链字段 ✗（那样两处路径会各自漂 ✓）。
                self.attach_qwen25vl_vision(chain["textWeights"])
        #: ⭐⭐ **装完、且装载栈退干净之后**再还一次显存 ✓✗（这一句是 2026-11-01 端到端的关键 ✓）。
        #: 为什么装载函数自己末尾那句不够 ✗✗：那时它的**局部变量还活着** ✓（物化中转 ✓、
        #: `WeightLoadReport` 握着的引用 ✓）⇒ allocator 看那些块「还在用」⇒ `empty_cache()` **一个字节不还** ✓✗。
        #: 实测：`load_weights` 末尾 trim 之后 `reserved` 仍 36402 MiB ✓（活张量只有 4.02 GiB ✓）；
        #: 等栈退干净（探针里过了一秒 ✓）再 synchronize+empty_cache ⇒ 立刻 36402 → **4332** ✓、
        #: `free` 0 → **17457 MiB** ✓。所以这一句必须落在**调用方**、装完之后 ✓。
        #: ⚠️ 常驻件的设备迁移排在**装完之后** ✓✗：先把卡让空（上一件退掉 ✓、这一件装好 ✓），
        #: 再搬那 5 GiB 上去 ✓ —— 顺序反了就是"往满的卡里塞" ✓✗（OOM 报在 `module.to` 上 ✓）。
        relocated = parked + self._relocate_keep_parts(kept, wanted)
        self._trim_cuda()
        return {"phase": phase, "loaded": list(wanted), "released": released,
                "relocated": relocated,
                "device": self._device, "residency": self._residency,
                "decision": decision.to_dict()}

    def _attach_h3_vae_weights(self, weights: str, *, device: str = "cpu") -> dict[str, Any]:
        """挂 **H3 口径的真视频 VAE** ✓（``decode`` 阶段专用 ✓）—— 与参考族**同名**但两回事 ✗。

        ⚠️ 为什么不能借用 `attach_wan_vae` ✗✗：那一条只认 ``wan``／``qwen_image21`` 一类的族名 ✓，
        H3 的 VAE 是 ``H3VideoVAEConfig`` ✓（64ch 潜 ＋ ``spatialScale=16`` ✓）。用错族的壳装
        H3 权重 ⇒ 报「大段 missing」✓（读起来像权重不全 ✗，实际是壳不对 ✓✗）。

        ⚠️ 默认 **cpu** ✓：它 5.2 GiB 与 DiT 的 17.94 GiB int8 抢同一块 22.49 GiB 卡 ✓✗；
        而 ``decode`` 阶段 DiT 已经退了 ✓ —— 想上卡须由调用方显式要 ✓（本方法不擅自决定 ✗）。
        """
        from app.services.engine import vae_h3 as vae_h3_mod  # noqa: PLC0415

        report = self.attach_vae(vae_h3_mod.H3VideoVAEConfig(weights=str(weights),
                                                             device=str(device), strict=True))
        if not report.get("complete"):
            raise TorchBackendUnavailable(
                f"真 H3 视频 VAE 装不齐 ✗（缺 {len(report.get('missing') or [])} 个键 ✓）"
                f"⇒ **拒用** ✓：{weights}", reason="pending")
        return report

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
        再核一次 ✓（见 `IMPLEMENTATION_HISTORY` ✓）。
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
                dit_scale = int(self._config_value("vae_scale", 0) or 0)
                if dit_scale:
                    pairs.append((
                        "视频", "spatial_scale", dit_scale,
                        "DiT 配置里的 `vae_scale` ✓（装权重时读出来的 ✓ —— 0 = 未给 ⇒ 不核 ✓）",
                        "**VAE 与 DiT 配置的倍率对不上** ✗（潜变量 ↔ 像素的换算两边不同 ✓✗）"))
                # ⭐ 2026-10-31 补 ✓：**通道数**也要核 ✗ —— 此前 DiT 形态只核倍率 ✓，
                #    于是一张 16 通道的 Z-Image 潜变量可以挂上 **4 通道的 SDXL VAE** ✓✗：
                #    `decoder.conv_in` 会在解码那一刻炸 ✓（报 「given groups=1, weight [512,4,…]」✓ ——
                #    指不到「挂错族」✓），更阴的是**同是 16 通道**却缩放系数不同的两族 ✓✗：
                #    形状全对 ✓、图发灰 ✓、不报错 ✓✗。通道这一条先拦掉错族的绝大多数 ✓。
                #: ⚠️ 与 :meth:`latent_shape` **同一格** ✓✗：带条件的族（SEEDVR2 ✓）`in_channels`
                #: 是「潜变量 ＋ 条件」拼起来的宽度（33 ✓），不是潜通道（16 ✓）⇒ 照它核会把
                #: **正确的** VAE 判成挂错族 ✓✗（16 ≠ 33 ⇒ 当场拒 ✓）。
                dit_channels = int(self._config_value("latent_channels", 0)
                                   or self._config_value("in_channels", 0) or 0)
                if dit_channels:
                    pairs.append((
                        "视频", "latent_channels", dit_channels,
                        "DiT 配置里的 `latent_channels` ✓（没有这格才退回 `in_channels` ✓"
                        "—— 后者是 `x_embedder` 入维 ÷ patch² ✓，带条件的族会多出条件通道 ✗；"
                        "Z-Image = 16 ✓ / SEEDVR2 潜 16 ✓ 而入口 33 ✓）",
                        "**挂错族的 VAE** ✗（SDXL 4 通道 ✗ / Flux 16 通道 ✓ / H3 走上面那条 ✗）"
                        "—— 解码当场在 `conv_in` 上炸 ✓，且报错不指向挂载代码 ✓✗"))
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
        # ⚠️ 双流只认 **H3 系** 视频 VAE ✗（2026-10-31 补 ✓）：2.1 的 Wan 式 VAE 是**单帧图**
        #    口径 ✓✗，拿它解 5 维视频潜变量时 `decode` 会「接受」并吐一个形状看着合理的张量 ✓✗
        #    —— 时间维被当空间维吃掉 ✓、不报错 ✓、画面是糊的 ✓✗。先拦在门口 ✓。
        if self._vaeKind not in ("h3-real", "reference-untrained"):
            raise TorchBackendUnavailable(
                f"双流解码只认 H3 系视频 VAE ✗（当前挂的是 {self._vaeKind!r} ✓）—— "
                f"``attach_vae()`` / ``attach_vae(video=…)`` 换成 H3 那族 ✓✗", reason="pending")
        torch = self._torch()
        with torch.no_grad():
            frames = self._vae.decode(latents["video"].unsqueeze(0))
            waveform = self._audioVae.decode(latents["audio"].unsqueeze(0))
        sample_rate = int(getattr(self._audioVaeConfig, "sample_rate", 0) or 32000)
        # ⚠️ ``synthetic`` **按事实算** ✓（2026-10-29 ✓）：视频 VAE 可以是**真权重** ✓、音频那族还只有
        #    参考实现 ✗ ⇒ 写死 True 会在"真视频 VAE + 假音频"时**骗人** ✗（两者都在场时画面确实是
        #    噪声 ✓，但理由得说对 ✓ —— 是音频侧拉了后腿 ✓，不是视频没接 ✓）。
        video_real = self._vaeKind == "h3-real"
        audio_real = self._audioVaeKind == "h3-real"
        return {
            "synthetic": not (video_real and audio_real),
            "note": ("真 torch 张量 + 视频 VAE="
                     + ("**H3 真权重** ✓" if video_real else "参考（**未经训练** ✗）")
                     + " / 音频 VAE="
                     + ("**H3 真权重** ✓" if audio_real else "参考（**未经训练** ✗）")
                     + (" ⇒ 两族都真 ✓，这是真画面 ✓"
                        if video_real and audio_real
                        else " ⇒ 音频侧仍是噪声 ✓（视频侧的接法已换真 ✓）"
                             if video_real else
                             " ⇒ 画面与声音都是噪声 ✓（验的是管道 ✓；真权重到位后换实现 ✓）")),
            # ⚠️ 两条流的 batch 处理**刻意不同** ✗✓ —— 各自按**落盘工具的契约**来 ✓：
            #    `media.write_video` 要 **5 维** ``(B,3,T,H,W)`` ✓（与单流 decode 同口径 ✓）
            #    ⇒ 视频**保留**那一维 ✓；`media.write_wav` 要 ``(channels, N)`` ✓ ⇒ 音频**去掉** ✓。
            #    2026-09-20 自检抓到过：视频也去掉 ⇒ `write` 认不出帧张量 ⇒ **默默走成"张量清单 JSON"**
            #    那条路 ✗✗（错在 `json.dumps(Tensor)` 才炸 ✓ —— 属于"响亮但指不到真因" ✓）。
            "frames": frames, "frameCount": int(frames.shape[2]),
            # ⭐ **顺手把潜变量交回** ✓（2026-09-26 ✓）：**续拍**要拿上一段的**尾部潜变量**当下一段的
            #    头部锚 ✓（`chain.run_chain` 的 ``continuation`` ✓）—— 上游 `h3_extend` 的 context_pin
            #    正是这么做的 ✓（钉的是**潜变量** ✓ 不是解码后的帧 ✓：省一趟 VAE ✓ 且逐位干净 ✓）。
            #    ⚠️ 这里交的是**引用** ✓（不多占显存 ✗）：调用方自己决定什么时候丢掉 ✓（链上只留尾巴 ✓）。
            "latents": {"video": latents["video"], "audio": latents["audio"]},
            "shape": list(frames.shape), "dtype": str(frames.dtype), "device": str(frames.device),
            "audioWaveform": waveform[0], "audioShape": list(waveform.shape),
            "sampleRate": sample_rate,
            "vaeKind": self._vaeKind, "audioVaeKind": self._audioVaeKind,
            "vaeConfig": self._vaeConfig.to_dict() if self._vaeConfig is not None else None,
            "audioVaeConfig": (self._audioVaeConfig.to_dict()
                               if self._audioVaeConfig is not None else None),
        }

    def _decode_music3(self, latents: Any) -> dict[str, Any]:
        """音频潜变量 ``(B,128,T)`` → **波形** ``(B,2,512·T)`` ✓（不是帧 ✓✗）。

        ## 为什么这一族在 :meth:`decode` 里**提前 return** ✗✗

        通用那条路后面全是**视频**口径 ✓：``frames`` → ``(B,T,3,H,W)`` ✓、``frameCount`` ✓、
        ``fps`` 由 plan 来 ✓…… 一样都对不上音频 ✓。硬走通也不是"顺手兼容" ✓，
        而是把"音频被当成 1 帧高的视频"这种**形状全对语义全错**的东西放出去 ✓✗。

        三处必须核 ✓：
        ① **通道 = 128** ✓（DAV 的 ``latentChannels`` ✓）—— 128 也是 **2×64** ✓，所以一条
           ``(B,64,8,8,H,W)`` 式的错 reshape 完全塞得进同一个数 ✓（这不是假想 ✗）；
        ② **必须是 3 维** ✓：采样侧给它的是 ``(B,128,T)`` ✓（``latent_shape`` 已认 3 维 ✓）；
           万一上游塞了 ``(B,1,128,T)`` ✓，这里**只**允许这一种 4 维形态并当场压平 ✓；
        ③ **反缩放 = 恒等** ✓：交给 :func:`vae_music3.decode_latents` ✓（它会自己核
           ``latent_formats.minimax_music3.scale_factor`` 还是不是 1.0 ✓ —— 有人改成 0.3611
           那里就当场报错 ✓，不静默出一段音量偏差 3 倍的声音 ✓✗）。
        """
        self._gate()
        torch = self._torch()
        from app.services.engine import vae_music3 as vae_music3_mod  # noqa: PLC0415

        spec = self._vaeConfig
        want_ch = int(getattr(spec, "latent_channels", 0) or 0)
        if not want_ch:
            raise TorchBackendUnavailable(
                "``music3-real`` 缺**通道事实** ✗（``_vaeConfig.latent_channels`` 没记下来 ✓）"
                "　⇒ 重挂一次 ``attach_music3_dav()`` ✓", reason="pending")
        shape_note = ""
        if latents.dim() == 4:
            if int(latents.shape[1]) != 1:
                raise TorchBackendUnavailable(
                    f"Music3 的潜变量是 3 维 ✗（``(B,{want_ch},T)`` ✓），现在是 {tuple(latents.shape)} ✓"
                    " —— 4 维只放过 ``(B,1,C,T)`` 这一种空维形态 ✓", reason="pending")
            latents = latents[:, 0]
            shape_note = "（4 维 `(B,1,C,T)` 已压平成 3 维 ✓）"
        if latents.dim() != 3:
            raise TorchBackendUnavailable(
                f"Music3 的潜变量要 3 维 ✗（``(B,{want_ch},T)`` ✓），现在是 "
                f"{tuple(latents.shape)} ✓", reason="pending")
        if int(latents.shape[1]) != want_ch:
            raise TorchBackendUnavailable(
                f"Music3 潜变量通道 {int(latents.shape[1])} ✗ ≠ DAV 的 {want_ch} ✓"
                f"（{tuple(latents.shape)} ✓）—— 这一族是**音频**潜变量 ✓，拿视频那套解出来的"
                "是 64 通道的空间图 ✓✗", reason="pending")
        wave = vae_music3_mod.decode_latents(self._vae, latents)
        #: ⚠️ 回落值**只能**取 ``MUSIC3_DAV_FACTS["sampleRate"]`` ✓✗（2026-11-06 首次真跑：
        #: 这里写过 ``vae_music3_mod.SAMPLE_RATE`` ✓ —— 那个名字**从来不存在** ✓，
        #: 于是整条解码在最后一步炸成 AttributeError ✓，前面四环全绿也救不回来 ✓）。
        rate = int(getattr(spec, "sample_rate", 0)
                   or vae_music3_mod.MUSIC3_DAV_FACTS["sampleRate"])
        channels = int(wave.shape[1])
        samples = int(wave.shape[-1])
        seconds = float(samples) / float(rate)
        return {
            "frames": None,
            "audioWaveform": wave[0] if int(wave.shape[0]) == 1 else wave,
            "sampleRate": rate,
            "audioChannels": channels,
            "audioSamples": samples,
            "audioSeconds": seconds,
            "real": True,
            "vae": "music3-real",
            "note": (
                f"**真** Music3 DAV ✓：{int(latents.shape[1])} 通道 × {int(latents.shape[-1])} 潜帧 ✓"
                f" ⇒ {channels} 声道 × {samples} 样本 ✓（{seconds:.2f} 秒 @ {rate} Hz ✓、"
                f"反缩放**恒等** ✓）{shape_note}"),
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
            #: ⚠️ **解码前的口径对账** ✓✗（2026-11-01 ✓）：DiT 族的潜空间口径 与 已挂 VAE 的
            #: 反缩放口径 必须同族 ✓ —— 16ch 的 ``wan21`` 与 16ch 的 ``qwen_image`` **形状完全相同** ✓、
            #: 只差 mean/std ✓⇒ 挂错了 ⇒ 图整体偏色/发灰 ✓、形状全对 ✓、一个错都不报 ✓✗。
            #: 未知（族没表态 / 老路 VAE / 名字不在表里）⇒ 不拦 ✓，但那是"没检查"✓✗ 不是绿灯 ✓。
            _vae_format: str | None = None
            if self._vaeKind == "flux-real":
                _vae_format = dit_build.contract_of("flux").latent_format
            elif self._vaeKind == "wan21-real" and self._vaeLatents:
                _vae_format = self._vaeLatents[2]
            elif self._vaeKind == "h3-real":
                #: ⚠️ H3 的骨架不走装配表 ✓，但它的口径**是知道的** ✓（`latent_formats.h3_video` ✓）
                #: ⇒ 从 :func:`dit_build.latent_format_of` 取 ✓，别因为"不在 BUILDERS"就整条跳过 ✗✗。
                _vae_format = dit_build.latent_format_of("h3")
            elif self._vaeKind == "seedvr2-real":
                #: ⚠️ 这一族**必须在表里** ✓（它是 `BUILDERS["seedvr2"]` 的正牌装配族 ✓）——
                #: 与 Wan 的 16ch 形状完全相同、只差 ``× 0.9152`` 那一格 ✓ ⇒ 挂错族**不报错**、
                #: 只是整片偏亮发糊 ✓✗。对账走契约里的规范名 ✓（`latent_formats.SeedVR2` ✓）。
                _vae_format = dit_build.contract_of("seedvr2").latent_format
            elif self._vaeKind == "music3-real":
                #: ⚠️ 音频族也要对账 ✓：它只有 **128 通道 × T** ✓，形状上与任何视频族都不像 ✓，
                #: 但对账是**便宜且不需要额外知识**的一步 ✓ —— 留着它，才有"挂错族当场拒" ✓。
                _vae_format = dit_build.contract_of("music3").latent_format
            _mismatch = dit_build.latent_format_problem(getattr(self, "_ditFamily", None), _vae_format)
            if _mismatch:
                raise TorchBackendUnavailable(_mismatch, reason="attach")
            #: ⭐ **Music3 在口径对账之后立刻分出去** ✓（2026-11-06 ✓）：下面整段是**视频**口径
            #: （``frames``/``frameCount``/``fps`` ✓），音频没有一样对得上 ✓✗。
            if self._vaeKind == "music3-real":
                return self._decode_music3(latents)
            torch = self._torch()
            alpha_dropped = False
            #: 省显存旋钮这一拍**实际**用了什么 ✓✗（``None`` ⇒ 全关、逐位那条路 ✓）。
            #: 只有 ``wan21-real``（空间 tile / attn chunk ✓）与 ``seedvr2-real``（时间分块 ✓）
            #: 会写它 ✓（别的族没有这些旋钮 ✓ —— 不在这里假装有 ✗）。
            tiling_note: dict[str, Any] | None = None
            if self._vaeKind == "flux-real":
                # ⚠️ 真 Flux VAE **必须**走 ``decode_latents`` ✗（内部先 `(latents − shift)/scale` ✓）
                #    —— 直接 ``self._vae.decode`` 会跳过 0.3611/0.1159 ✓✗：形状全对 ✓、图发灰 ✓、
                #    一个错都不报 ✓✗（见 `vae_flux` 模块头 ✓）。
                from app.services.engine import vae_flux as vae_flux_mod  # noqa: PLC0415
                with torch.no_grad():
                    frames = vae_flux_mod.decode_latents(
                        self._vae, latents, config=self._vaeConfig, to_unit=False)
                real = True
            elif self._vaeKind == "wan21-real":
                # ⚠️ 同理 ✗：真 Wan 式 VAE **必须**走 ``decode_latents`` ✓（内部先 ``x*std+mean`` ✓）
                #    —— 直接 ``self._vae.decode`` 会把**归一化**潜变量当像素尺度解 ✓✗：
                #    形状全对 ✓、图发灰/糊 ✓、一个错都不报 ✓✗（见 `vae_wan` 模块头 ✓）。
                from app.services.engine import vae_wan as vae_wan_mod  # noqa: PLC0415
                if not self._vaeLatents:
                    raise TorchBackendUnavailable(
                        "``wan21-real`` VAE 缺**反缩放事实** ✗（``latents_mean/std`` 没记下来 ✓）"
                        "　⇒ 重挂一次 ``attach_wan_vae()`` ✓；⚠️ 别拿恒等换算顶上 ✗✗"
                        "（解码照跑、图发灰、不报错 ✓✗）", reason="pending")
                latents_mean, latents_std, family = self._vaeLatents
                #: ⭐ 两个省显存旋钮 ✓✗（2026-11-02 ✓）：**默认全关** ✓ ⇒ ``tile=None`` 走旧路径 ✓、
                #: 逐位不变 ✓。开了才改数值 ✓（见 :meth:`set_vae_tiling` 的两条实测差 ✓）。
                #: ⚠️ ``fallback`` 只在**真 OOM** 时降档重试 ✓ —— 不预先降档 ✗：那个画布明明
                #: 装得下的时候（实测 2048×1536 要 20.96 GiB / 卡 22.49 ✓）悄悄分块就是白掉质量 ✓✗。
                _knobs = dict(self._vaeTiling)
                _tile, _halo, _chunk = _knobs["tile"], _knobs["halo"], _knobs["attnChunk"]

                def _decode(tile, halo, chunk):
                    with torch.no_grad():
                        return vae_wan_mod.decode_latents(
                            self._vae, latents, family=family,
                            mean=latents_mean, std=latents_std, to_unit=False,
                            tile=tile, halo=halo, attn_chunk=chunk)

                try:
                    frames = _decode(_tile, _halo, _chunk)
                    if _tile is not None or _chunk is not None:
                        tiling_note = {"tile": _tile, "halo": _halo, "attnChunk": _chunk,
                                       "reason": "configured"}
                except RuntimeError as exc:
                    #: ⚠️ 只接**显存**这一种 ✗（:func:`_is_oom` ✓）：别的错降档也救不了 ✓，
                    #: 盖上"已降档"反而把真 bug 埋掉 ✓✗。
                    if not (_knobs["fallback"] and _is_oom(exc)
                            and (_tile is None or _chunk is None)):
                        raise
                    #: 先腾地方再重试 ✓ —— 这一拍 ``latents`` 还活着 ✓，不腾会**同一处**再 OOM ✓✗。
                    frames = None
                    self._trim_cuda()
                    _tile = _tile or vae_wan_mod.SUGGESTED_TILE
                    _chunk = _chunk or vae_wan_mod.SUGGESTED_ATTN_CHUNK
                    try:
                        frames = _decode(_tile, None, _chunk)
                    except RuntimeError as retry_exc:
                        if not _is_oom(retry_exc):
                            raise
                        raise TorchBackendUnavailable(
                            f"VAE 解码**显存仍不够** ✗：已降到 ``tile={_tile}`` + "
                            f"``attn_chunk={_chunk}`` 仍 OOM ✓（原报错：{exc} ✓）"
                            f"　⇒ 这次画布是真的超了 ✓ —— 降分辨率，别去降精度/压 cfg ✗✗",
                            reason="resource") from retry_exc
                    tiling_note = {"tile": _tile, "halo": None, "attnChunk": _chunk,
                                   "reason": "oom", "error": str(exc)[:200]}
                # ⚠️ 这族的 ``head`` 出 **4 通道 RGBA** ✓✗（2026-10-31 实测：`media.write_image`
                #    只认 ``(B,3,H,W)`` ⇒ 落盘那一步才炸 ✓，报得还像"形状给错了" ✓✗）。
                #    第 4 通道是 alpha ✓，出图**必须丢** ✓ —— `decode_latents` 保留它只是为了把
                #    「4 通道」这个族事实留在张量上 ✓，投影到 RGB 是**这一步**的职责 ✓。
                #: ⚠️ 但**不是每个 wan21-real 的 head 都是 4 通道** ✗（2026-11-01 ✓）：Krea2 用的
                #: ``qwen_image_vae`` 是 **3 通道** ✓ ⇒ 判据只能是**实测通道数** ✓（写死族名 ⇒
                #: 把 3 通道当 4 通道丢第一个通道 ⇒ 图变红蓝错位 ✓✗、不报错 ✓）。
                alpha_dropped = int(frames.shape[1]) == 4
                if alpha_dropped:
                    frames = frames[:, :3]
                real = True
            elif self._vaeKind == "seedvr2-real":
                # ⚠️ 同理 ✗：这一族**必须**走 ``decode_latents`` ✓（内部先 ``÷ 0.9152`` 回 VAE 空间 ✓）
                #    —— 直接 ``self._vae.decode`` 会把**模型空间**潜变量当 VAE 空间解 ✓✗：
                #    形状全对 ✓、整片偏亮发糊 ✓、一个错都不报 ✓✗（见 `vae_seedvr2` 模块头 ✓）。
                #    输出是 **[-1, 1]** ✓（与 Wan 同口径 ✓，落盘那一步再投影 ✓）。
                from app.services.engine import vae_seedvr2 as vae_seedvr2_mod  # noqa: PLC0415
                #: ``temporalChunk`` 只在这一族有效 ✓（它是**解码器调用级**的时间切块 ✓，
                #: 每块带块外真实上文 ⇒ 结果与整段相等 ✓，不是拿数值换显存 ✓）。
                _knobs = dict(self._vaeTiling)
                _temporal_chunk = _knobs["temporalChunk"]
                #: 显式配过就不自降 ✓（创作者点名要的档位，慢也该照办 ✓，只如实回报 ✓）；
                #: 但**上一轮自己降下来的**不算显式 ✓✗ —— 那条还得能继续降 ✓。
                _explicit = (_temporal_chunk is not None
                             and _temporal_chunk != self._vaeAutoChunk)
                #: 溢出探测：``reserved > 物理显存`` ＝ 驱动已经把一部分落到**主机内存** ✓✗
                #: （Windows 的 sysmem fallback ✓）—— 这一步**不会**报 OOM ✓，只会慢到 28 倍 ✓。
                _can_probe = bool(torch.cuda.is_available()) and (
                    self._device_torch().type == "cuda")
                _card_bytes = (torch.cuda.get_device_properties(
                    torch.cuda.current_device()).total_memory if _can_probe else 0)
                if _can_probe:
                    if self._vaeSpilled:
                        #: 上一轮把块散到主机内存了 ✓ ⇒ 开跑前先还 ✓（否则这一轮照样颠簸 ✓✗）。
                        torch.cuda.synchronize()
                        self._trim_cuda()
                        self._vaeSpilled = False
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

                def _decode_temporal(chunk: int | None) -> Any:
                    with torch.no_grad():
                        return vae_seedvr2_mod.decode_latents(
                            self._vae, latents, family="seedvr2", to_unit=False,
                            temporal_chunk=chunk)

                try:
                    frames = _decode_temporal(_temporal_chunk)
                    if _temporal_chunk:
                        tiling_note = {"tile": None, "halo": None, "attnChunk": None,
                                       "temporalChunk": _temporal_chunk,
                                       "reason": "configured" if _explicit else "auto",
                                       "exact": True}
                except RuntimeError as exc:
                    #: ⚠️ 这一族的兜底**与 wan21 那族不是一回事** ✓✗：那边降 tile 是**换字节** ✓✗，
                    #: 这边降的是**时间分块** ⇒ 只降峰值、**不动数值** ✓ ⇒ 兜底不换画质 ✓（
                    #: 实测 `engine_seedvr2_vae_test` ⑫ 与整段 `≤1e-5` ✓）。这条正是 336 秒 / 47.60 GiB
                    #: 那次「全程在 OOM 重试里打滚」的**唯一**无损出路 ✓✗。
                    #: 判据仍只接**真 OOM** ✓（:func:`_is_oom` ✓）：别的错降档救不了 ✓，盖上"已降档"
                    #: 反而把真 bug 埋掉 ✓✗。
                    if not (_knobs["fallback"] and _is_oom(exc) and _temporal_chunk is None):
                        raise
                    #: ⭐ 不是「降一档就收工」✗✗：峰值 ≈ 与**每块像素帧数**（``4L``）成正比 ✓，
                    #: 而 ``SUGGESTED_TEMPORAL_CHUNK=4`` 在 720×480 这种真分辨率下**自己就可能不够** ✓
                    #: （一块 17 像素帧 ≈ 19 GiB ✓）。所以按 :func:`vae_seedvr2.chunk_ladder` 一档档减半 ✓，
                    #: 每档都先腾地方 ✓ —— 这一拍 ``latents`` 还活着 ✓，不腾会**同一处**再 OOM ✓✗。
                    frames = None
                    _trail: list[int] = []
                    for _step in vae_seedvr2_mod.chunk_ladder():
                        self._trim_cuda()
                        try:
                            frames = _decode_temporal(_step)
                        except RuntimeError as retry_exc:
                            if not _is_oom(retry_exc):
                                raise
                            _trail.append(_step)
                            continue
                        _temporal_chunk = _step
                        break
                    if frames is None:
                        raise TorchBackendUnavailable(
                            f"SEEDVR2 解码**显存仍不够** ✗：``temporalChunk`` 已试遍 "
                            f"{_trail} 潜帧（最后是 1，一块只 1 像素帧 ✓）仍 OOM ✓"
                            f"（原报错：{exc} ✓）"
                            f"　⇒ 这次是真的超了 ✓ —— 降像素帧数/分辨率，别去降精度 ✗✗"
                            f"（修复族的输出分辨率就是交付物 ✓）",
                            reason="resource") from exc
                    tiling_note = {"tile": None, "halo": None, "attnChunk": None,
                                   "temporalChunk": _temporal_chunk, "reason": "oom",
                                   "exact": True, "error": str(exc)[:200],
                                   "tried": [*_trail, _temporal_chunk]}
                #: ⭐ 溢出（没报 OOM 但 reserved 超过物理显存 ⇒ 已落到主机内存）⇒ **自降一档并记住** ✓：
                #: 这一次的 231 秒已经花掉了 ✓，不必下次再花一遍 ✓。判据与估算见
                #: :func:`vae_seedvr2.shrink_for_peak` ✓；显式配置的档位不擅自动 ✓，只回报 ✓。
                if _can_probe:
                    _peak = torch.cuda.max_memory_reserved()
                    _shrunk = vae_seedvr2_mod.shrink_for_peak(
                        _temporal_chunk or vae_seedvr2_mod.SUGGESTED_TEMPORAL_CHUNK,
                        _peak, _card_bytes)
                    if _shrunk is not None:
                        if not _explicit:
                            self._vaeTiling["temporalChunk"] = _shrunk
                            self._vaeAutoChunk = _shrunk
                        self._vaeSpilled = True
                        if tiling_note is None:
                            tiling_note = {"tile": None, "halo": None, "attnChunk": None,
                                           "temporalChunk": _temporal_chunk,
                                           "reason": "spill",
                                           "exact": bool(_temporal_chunk)}
                        tiling_note["spilled"] = True
                        tiling_note["peakReservedGiB"] = round(_peak / 2**30, 2)
                        tiling_note["nextTemporalChunk"] = _shrunk
                        #: ``applied=False`` ＝ 他点名要的档位，我们只是**建议**降到这个数 ✓✗
                        #: （没替他改 ✓）—— 读回执的人据此决定要不要改 ✓。
                        tiling_note["applied"] = not _explicit
                real = True
            else:
                with torch.no_grad():
                    frames = self._vae.decode(latents)
                real = self._vaeKind == "h3-real"
            # ⚠️ **图**（4 维 ✓）没有时间维 ✗ ⇒ 不能写死 ``shape[2]`` 当帧数 ✓✗（那是高度 ✓）。
            frame_count = int(frames.shape[2]) if len(tuple(frames.shape)) == 5 else 1
            # ⚠️ 族名要**逐族点名** ✗（此前是二选一 "Flux / 否则 H3" ✓✗ ⇒ 挂了 2.1 的 VAE 也会
            #    被写成 H3 ✓✗，报告里看不出到底哪族解的 ✓）。
            #: ⚠️ ``wan21-real`` 不再是「2.1 那一个文件」✗（2026-11-01 ✓）：2.1 是
            #: ``qwen_image_2.1_vae``（64ch ✓）、Krea2 是 ``qwen_image_vae``（16ch ✓）⇒ 标签按
            #: **实测通道数 + 记下来的 family** 拼 ✓（写死「64 通道」会把 Krea2 的解码说成 2.1 ✓✗）。
            _wan_label = "Wan layout ✓"
            if self._vaeLatents:
                from app.services.engine import vae_wan as vae_wan_mod  # noqa: PLC0415
                _fam = self._vaeLatents[2]
                _dim = (self._vaeConfig.z_dim if self._vaeConfig is not None else 0)
                #: 族名 → 人话标题 ✓（``qwen_image21`` ⇒ 「Qwen-Image 2.1」✓）；
                #: 没登记的族原样回显 ✓（不猜 ✗）。
                _wan_label = (f"**{vae_wan_mod.family_title(_fam)}**（Wan layout，{_dim} 通道，"
                              f"族名 ``{_fam}``）✓")
            kind_label = {"flux-real": "Flux ✓", "wan21-real": _wan_label,
                          "seedvr2-real": "SEEDVR2 视频修复 VAE ✓"}.get(
                self._vaeKind, "H3 ✓")
            return {
                "synthetic": not real,      # 真张量 ✓ + **真 VAE** ✓ ⇒ 这就是真画面 ✓
                "note": ("真 torch 张量 + **真 VAE**（"
                         + kind_label
                         + "）⇒ 这是真画面 ✓（潜变量若是 DiT 出的 ✓ 就是成品 ✓）"
                         + ("　〔第 4 通道 alpha 已丢 ✓ —— 落盘只认 RGB ✓〕" if alpha_dropped else "")
                         if real else
                         "真 torch 张量 + 参考 VAE ✓，但 VAE **未经训练** ✗ ⇒ 画面是噪声 ✓"
                         "（验的是管道 ✓；接真权重后这里换成训练好的 VAE ✓）"),
                "vaeKind": self._vaeKind,
                #: 省显存旋钮的**实际**用法 ✓✗（``None`` ⇒ 全关、逐位 ✓；非空 ⇒ 产物字节会变 ✓）。
                #: ⚠️ 必须跟着回执走 ✗：开了却说"没开" ⇒ 复现不出同一张图时无从判断原因 ✓✗。
                "vaeTiling": tiling_note,
                "frames": frames, "frameCount": frame_count,
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

    def encode_pixels(self, pixels: Any, plan: Any = None, *,
                      pixel_range: str = "minus_one_to_one") -> dict[str, Any]:
        """**像素 → 潜变量** ✓ —— SEEDVR2 修复链的**入口一步** ✓（见 :mod:`vae_seedvr2` ✓）。

        为什么需要它 ✗（不能拿 :meth:`decode` 反着调 ✓）：编码器**不是**解码器的镜像入口 ✓ ——
        出 **32** 通道（16 均值 + 16 logvar ✓）、要**取前 16** ✓、且没有 ``quant_conv`` ✓。
        直接 ``self._vae.encode`` ✗ 会漏掉 ``× 0.9152`` ✓✗（形状全对 ✓、潜变量整体小一圈 ✓、
        修完画面发灰 ✓、一个错都不报 ✓✗）。

        ⚠️ **只**给 ``seedvr2-real`` ✓（本仓五族真 VAE 里只有它在链路上需要「像素进」✓）：
        - ``flux-real`` / ``wan21-real`` 是**出图**那一侧 ✓（像素由 DiT 直接给 ✓）；
        - ``h3-real`` 双流要音频侧（:meth:`_decode_dual` ✓）；
        - 其余族的「编码」在管线里由「首帧条件图」另走一条 ✓。

        ⚠️ **像素口径**（``pixel_range`` ✓，写错同样"不报错只是整片偏灰" ✓✗）：
        ``"minus_one_to_one"``（默认 ✓，与 :meth:`decode` 的输出**同口径** ✓ ⇒ 往返自洽 ✓）
        / ``"zero_to_one"``（8bit 帧的 [0,1] 归一 ✓，这里替你 ``×2−1`` ✓）。**别**自己先乘一半 ✗。

        ⚠️ **帧数必须 4n+1** ✓（时间 ÷4 ＋ 因果补边 ✓）：不合规 ⇒ :mod:`vae_seedvr2` 当场拒
        并给最近两个合法值 ✓（**别**在这里替调用方补帧 ✗ —— 补边会不会进 mask、动的是哪条时间轴，
        是**修链**该决定的事 ✓）。
        """
        self._gate()
        if self._vaeKind != "seedvr2-real" or self._vae is None:
            raise TorchBackendUnavailable(
                f"``encode_pixels`` **只**给 ``seedvr2-real`` 用 ✗（当前 VAE："
                f"``{self._vaeKind or '未挂'}`` ✓）—— 修复链的入口一步要 SEEDVR2 的 VAE ✓"
                f"（``vae/SEEDVR2/ema_vae_fp16.safetensors`` ✓）：先 ``attach_seedvr2_vae(path)`` ✓；"
                f"⚠️ 别拿 Wan/Flux 的 VAE 顶替 ✗✗：形状（16ch / 8×）**一样** ✓、"
                f"只差 ``× 0.9152`` 与 32→16 取均值 ✓ ⇒ 跑得下去、潜变量整体不对 ✓✗",
                reason="attach")
        #: ⚠️ 与 :meth:`decode` 同一条设备纪律 ✓：[0,1] 帧常在 CPU 上拼 ✓、
        #: VAE 在 `self._device` 上 ✓ ⇒ 混设备必炸 ✗，报错还是那串 `aten::` ✗。
        self._require_own_device("encode_pixels", pixels=pixels)
        if pixel_range not in ("minus_one_to_one", "zero_to_one"):
            raise TorchBackendUnavailable(
                f"``pixel_range`` 只认 ``minus_one_to_one`` / ``zero_to_one`` ✗"
                f"（收到 {pixel_range!r} ✓）—— 别自己先做半程归一 ✗", reason="pending")
        torch = self._torch()
        from app.services.engine import vae_seedvr2 as vae_seedvr2_mod  # noqa: PLC0415

        if not hasattr(pixels, "dim") or int(pixels.dim()) != 5:
            raise TorchBackendUnavailable(
                f"``encode_pixels`` 要 **(B,3,T,H,W)** 的 5D 帧批 ✗（收到形状 "
                f"{tuple(getattr(pixels, 'shape', ()))} ✓）—— 单帧 4D 也要先补 T=1 ✓"
                f"（但 T=1 合法 ✓：1 = 4·0+1 ✓）", reason="pending")
        frames = int(pixels.shape[2])
        #: 帧号铁律在这儿先问一遍 ✓（不合规 ⇒ 人话拒绝并给最近合法值 ✓，见 `latent_frames` ✓）。
        latent_frames = vae_seedvr2_mod.latent_frames(frames, self._vaeConfig)
        work = pixels.to(self._device)
        if pixel_range == "zero_to_one":
            work = work.mul(2).sub(1)
        with torch.no_grad():
            latents = vae_seedvr2_mod.encode_latents(self._vae, work, family="seedvr2").float()
        #: ⚠️ 回的是**模型空间**潜变量 ✓（已 ×0.9152 ✓，与 DiT 的输入同口径 ✓）——
        #: 调用方**别**再乘一次 ✗（`:meth:`decode` 会再 ÷ 回来 ✓✗）。
        return {
            "latents": latents,
            "modelSpace": True,
            "vaeKind": self._vaeKind,
            "scaleFactor": float(self._vaeConfig.scale_factor)
            if self._vaeConfig is not None else None,
            "shape": list(latents.shape),
            "dtype": str(latents.dtype),
            "device": str(latents.device),
            "frames": frames,
            "latentFrames": latent_frames,
            "spatialScale": int(getattr(self._vaeConfig, "spatial_scale", 0)),
            "temporalScale": int(getattr(self._vaeConfig, "temporal_scale", 0)),
            "pixelRange": pixel_range,
            "note": (f"像素 → 潜变量 ✓（真 SEEDVR2 VAE ✓，{frames} 帧 ⇒ {latent_frames} 潜帧 ✓ "
                     f"＝ 4n+1 ✓；已按 ×{float(self._vaeConfig.scale_factor):.4f} 进模型空间 ✓ "
                     f"—— 调用方**别**再乘一次 ✗；解回像素走 ``decode`` ✓ 会自动 ÷ 回来 ✓）")
            if self._vaeConfig is not None else "像素 → 潜变量 ✓（真 SEEDVR2 VAE ✓）",
        }

    # ---------- 产物取证（2026-11-07 补 ✓：产物自己带"怎么来的" ✓）----------

    @staticmethod
    def _artifact_metadata(request: Any, plan: Any, kind: str) -> dict[str, Any]:
        """落进 PNG iTXt / mp4 容器的**参数快照** ✓ —— 图脱离作业单后仍能自证 ✓。

        ⚠️ 只放**短**字段 ✓（提示词截断 ✓）：容器元数据块是产物的一部分 ✓，
        长提示词整段塞进去会让 PNG 体积不可控 ✓✗，而完整提示词本来就有 ``manifest.json`` 兜 ✓。
        """
        def _int(name: str) -> int:
            try:
                return int(getattr(plan, name, 0) or 0)
            except (TypeError, ValueError):
                return 0

        prompt = str(getattr(request, "prompt", "") or "")
        negative = str(getattr(request, "negative", "") or "")
        return {
            "engine": "darme-voide/engine",
            "kind": str(kind),
            "stage": str(getattr(request, "stage", "") or ""),
            "seed": int(getattr(request, "seed", 0) or 0),
            "width": _int("width"), "height": _int("height"),
            "frames": _int("frames"), "fps": _int("fps"), "steps": _int("steps"),
            "sampler": str(getattr(request, "sampler", "") or ""),
            "schedule": str(getattr(request, "schedule", "") or ""),
            "prompt": prompt[:2000],
            "negative": negative[:500],
        }

    def _load_summary(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        """装配摘要 ✓（manifest 用 ✓）—— **顺手失败不算失败** ✓（清单不该拖垮一次成功的生成 ✓）。"""
        try:
            desc = self.describe()
        except Exception:  # noqa: BLE001 - 自述本该总能调 ✓，但清单不是关键路径 ✓
            desc = {}
        residency = desc.get("residency") if isinstance(desc.get("residency"), dict) else {}
        return {
            "backend": desc.get("name"),
            "device": desc.get("device"),
            "available": desc.get("available"),
            "residencyMode": residency.get("mode"),
            "residency": residency.get("report"),
            "vaeKind": outputs.get("vaeKind"),
            "audioVaeKind": outputs.get("audioVaeKind"),
        }

    def _write_manifest(self, where: Path, request: Any, plan: Any,
                        result: dict[str, Any], outputs: Mapping[str, Any]) -> str | None:
        """产物目录的 ``manifest.json`` ✓（request / plan / 装配摘要 / ffprobe 事实 / sha256 ✓）。

        ⚠️ 只在**真写出产物之后**调用 ✓（拒绝/失败路径一个文件都不该多 ✗）；
        写不动 **不改判**这次生成 ✓，但会把原因记进 ``result["manifestError"]`` ✓（不静默 ✗）。
        """
        payload: dict[str, Any] = {
            "schemaVersion": 1,
            "writtenAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "request": request.to_dict() if hasattr(request, "to_dict") else None,
            "plan": plan.to_dict() if hasattr(plan, "to_dict") else None,
            "load": self._load_summary(outputs),
            "artifacts": [],
        }
        facts_by_kind = {"image": result.get("image"), "video": result.get("video"),
                         "audio": result.get("audio")}
        for item in result.get("artifacts") or []:
            entry = dict(item)
            path = Path(str(entry.get("path") or ""))
            if path.is_file():
                entry["bytes"] = path.stat().st_size
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                facts = facts_by_kind.get(str(entry.get("kind")))
                if facts:
                    entry["facts"] = facts
            payload["artifacts"].append(entry)
        target = where / "manifest.json"
        try:
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as err:
            result["manifestError"] = f"{type(err).__name__}: {err}"
            return None
        result["manifestPath"] = str(target)
        return str(target)

    @staticmethod
    def _clip_alert(report: dict[str, Any]) -> str | None:
        """「波形被削顶」那一句 ✓（**唯一句源** ✓ —— 5 维那支与纯音频那支都调它 ✓）。

        ⚠️ 为什么必须喊 ✗✗（2026-11-10 补 ✗）：:func:`media.write_wav` 早就把 ``clipped``
        （**越界样本数** ✓）算出来交出去了 ✓ —— 可它只躺在 ``audio`` 子字典里 ✓✗，
        ``note`` / ``alerts`` 一个字不提 ✓。于是「整段削顶、听着一片爆音 ✗」与
        「增益刚好 ✓」在回执里**同形** ✓（都是"真 wav ✓"），而削顶**绝不抛异常** ✓、
        也不会被判成 ``silent`` ✓（峰值恒为 1.0 ✓ 甚至更"响" ✓）⇒
        这是本族里最隐蔽的一格：**测量到了，但没人说** ✓✗。
        ⚠️ 人话要**先排除**解码/混流 ✗：削顶的锅在**波形生成那一段的增益/归一化** ✓。
        """
        count = int(report.get("clipped") or 0)
        if count <= 0:
            return None
        total = int(report.get("samples") or 0) * int(report.get("channels") or 0)
        share = f"，占 {count / total * 100:.2f}%" if total else ""
        return ("；⚠️ 波形有 %d 个样本**超出 [-1, 1] 被削顶** ✗%s —— "
                "这一段会听成**爆音/失真** ✓，**不是**解码或混流的锅 ✓"
                "（该查的是波形生成那一段的增益/归一化 ✓）" % (count, share))

    def write(self, outputs: dict[str, Any], plan: Any, request: Any) -> dict[str, Any]:
        """有**真帧张量**就落**真 mp4** ✓（ffmpeg ✓）；否则落张量清单 JSON ✓ —— 两种情况都如实标 ✓。"""
        self._gate()
        # ⚠️ 这条入口**刻意不做**设备校验 ✗（与上面五个入口**不同** ✓ —— 是查清后的判断 ✓ 不是漏 ✓）：
        #    `frames` 是 :meth:`decode` 的产物 ✓，而**落盘边界自己在收口** ✓✗ ——
        #    `media.write_video` / `write_wav` 结尾都 ``.to("cpu", …)`` ✓
        #    （见 `media.py` 的 ``scaled * 255`` 与 ``pcm`` 两处 ✓）⇒ **cpu 上解出来的帧照样能落盘** ✓。
        #    在这里加校验会把那条正当用法**拒掉** ✗✗（自检就常在 cpu 上解码 ✓），
        #    而它本来完全跑得通 ✓ ⇒ 「功能要全」= **逐个入口查清并写下结论** ✓，不是一律加 ✗。
        #: ⚠️ 兜底落点**不再** `tempfile.mkdtemp` ✗✗（2026-11-07 收口 ✓）：旧写法每次跑都新建
        #: 一个 `%TEMP%\engine_torch_xxxx` ✓✗ ⇒ 图散在临时目录里、清理时连凭据一起丢 ✓、
        #: 想留就得人工搬一遍 ✓✗。现在统一走 :mod:`output_paths` ✓ ⇒
        #: `<storageRoot>/engine/<image|video|audio>/` ✓（`/static` 服务得到 ✓，路径稳定 ✓）。
        #: kind 由**手里真有的东西**决定 ✓✗（先看帧维数、再看音频 ✓ ——
        #: H3 双流同时有 5 维帧与波形 ✓ ⇒ 按 video 归 ✓，音轨本来就在那个 mp4 里 ✓）。
        if request.outputs_dir:
            where = Path(request.outputs_dir)
        else:
            frames_shape = getattr(outputs.get("frames"), "shape", None)
            dims = len(tuple(frames_shape)) if frames_shape is not None else 0
            where = output_paths.fallback_outputs_dir(
                {4: "image", 5: "video"}.get(dims)
                or ("audio" if outputs.get("audioWaveform") is not None else output_paths.DEFAULT_KIND))
        where.mkdir(parents=True, exist_ok=True)
        frames = outputs.get("frames")
        waveform = outputs.get("audioWaveform")
        if hasattr(frames, "shape") and len(tuple(frames.shape)) == 4:
            # ⭐ **图**（4 维 ✓）落 PNG ✓ —— 2026-10-31 补 ✗：此前这里只认 5 维 ✓✗ ⇒ Z-Image 走完
            #    「真 TE ✓ → 真 DiT ✓ → 真 VAE ✓」之后，**最后一步落成了 JSON 清单** ✓✗
            #    （`synthetic: True` 还写着 ✓ —— 明明手里握着成品像素 ✓✗）。
            target = where / f"image_seed{int(request.seed)}.png"
            #: ⭐ 2026-11-07 补 ✓：把参数快照写进 PNG 的 iTXt ✓（盘上那批图此前**零元数据** ✓✗ ——
            #: 图一旦离开作业单就再也说不清是哪个模型/多少步/什么提示词出的 ✓）。
            report = media_mod.write_image(frames, target, value_range="-1..1",
                                          metadata=self._artifact_metadata(request, plan, "image"))
            synthetic = bool(outputs.get("synthetic", True))
            result: dict[str, Any] = {
                "synthetic": synthetic, "realTensors": True, "realFile": True,
                "imagePath": str(target), "primaryPath": str(target), "image": report,
                "artifacts": [{"kind": "image", "path": str(target), "bytes": report["bytes"]}],
                "note": ("**真 PNG** ✓（"
                         + ("真 VAE ✓ ⇒ 这就是成品图 ✓" if not synthetic
                            else "VAE 未训练 ✗ ⇒ 画面是噪声 ✓")
                         + "）"),
            }
            if report.get("degraded"):
                # ⚠️ 纯色/全黑/全白**不抛异常** ✓✗（画布上看着就是"一张图" ✓）⇒ 必须由回执喊出来 ✓
                result["degraded"] = True
                result["degradedReason"] = (
                    f"画面几乎是纯色（像素标准差 {report.get('std')} < 阈值 ✓）"
                    f"—— 检查值域口径（-1..1 / 0..1 ✓）与 VAE 是否真解出了内容 ✓")
                result["note"] = str(result["note"]) + "；⚠️ 但画面**几乎是纯色** ✗（见 `degradedReason` ✓）"
                #: ⭐⭐ 同一句还得进 ``alerts`` ✓✗（2026-11-10 补 ✗）—— 与音频那条**对称** ✓：
                #:   纯色图**不抛异常** ✓、画布上就是"一张图" ✓，而前端只显示 ``warnings`` ✓✗
                #:   （``result.note`` 与 ``degraded`` 它一个都不读 ✗）⇒ 用户看到一张灰片、
                #:   回执里没一个字到他眼前 ✓✗。上一轮只给音轨补了这一步、图这一支漏了 ✗
                #:   （同一个病根在同一个函数里换分支再犯一次 ✓）。
                #:   ⚠️ ``_alert`` 定义在 5 维那支里面 ✓（闭包绑那个 ``result`` ✓）⇒ 这里不能复用 ✗，
                #:   但这句与 ``note`` 里那句**逐字相同** ✓（不许各写一版 ✗）。
                result.setdefault("alerts", []).append(
                    "；⚠️ 但画面**几乎是纯色** ✗（见 `degradedReason` ✓）")
            #: ⭐⭐ 「PNG 里**真有**那份元数据吗」也要单独表态 ✓✗（2026-11-10 补 ✗）：
            #:   :func:`media.write_image` 现在写完后**从盘上读回来**逐键核过 ✓（``metadataKeys``
            #:   只列真读到的 ✓、``metadataMissing`` 列送了却没进去的 ✓、``metadataReadback`` 三档 ✓）。
            #:   这里只负责把「送了却没写进去」喊出来 ✓ —— 图一旦离开作业单，**元数据就是它唯一的出处** ✓，
            #:   静默丢掉 = 这张图**永久说不清是哪个模型/多少步/什么提示词** ✗。
            #:   ⚠️ 与 mp4 那侧 ``metadataDropped`` 同口径 ✓（2026-11-07 就为 mp4 补过 ✓，
            #:   图这支当时只核了键名在不在**送出清单**里 ✓✗ —— "打算写"被当成"写成了" ✓）。
            if report.get("metadataMissing"):
                result["metadataMissing"] = list(report["metadataMissing"])
                _missing_sentence = (
                    "；⚠️ 元数据有 %d 个键**没写进 PNG** ✗（%s ✓）—— "
                    "这张图脱离作业单后就说不清出处了 ✗"
                    % (len(report["metadataMissing"]),
                       "、".join(str(k) for k in report["metadataMissing"])))
                result["note"] = str(result["note"]) + _missing_sentence
                result.setdefault("alerts", []).append(_missing_sentence)
            result["metadataReadback"] = report.get("metadataReadback")
            self._write_manifest(where, request, plan, result, outputs)
            return result
        if hasattr(frames, "shape") and len(tuple(frames.shape)) == 5:
            #: ⚠️ 2026-11-07 补 ✓：**落盘前先核帧数** ✓✗ —— 盘上三个 ``video_seed777.mp4``
            #: 分别是 10/34/39 帧 ✓，就是同一个名字被"帧数不同"的两次生成**静默覆盖** ✓✗。
            #: 帧数是 plan 说了算的（``plan.frames`` ✓），解出来的 T 与它不符 ⇒ 当场失败 ✓；
            #: 不静默落一条"看着正常、其实少了几帧"的片子 ✓（那会一路流到成片里 ✗）。
            expected_frames = int(getattr(plan, "frames", 0) or 0)
            actual_frames = int(tuple(frames.shape)[2])
            if expected_frames and actual_frames != expected_frames:
                raise media_mod.MediaError(
                    f"帧数不符 ✗：plan.frames = {expected_frames} ✓，手里有 {actual_frames} 帧 ✗ "
                    f"—— 解码出的帧数与计划不一致（先查时间压缩比/潜帧口径 ✓），"
                    f"**不落一条少帧的片子** ✓")
            target = where / f"video_seed{int(request.seed)}.mp4"
            #: ⭐ 双流：**先把音轨落成 wav** ✓ 再混进 mp4 ✓ —— 顺序不能反 ✓
            #: （混流要读那个文件 ✓）。wav 仍**原样保留** ✓：要干净音频（转写/配音对齐 ✓）读它 ✓。
            audio_target: Path | None = None
            audio_report: dict[str, Any] | None = None
            if waveform is not None:
                audio_target = where / f"audio_seed{int(request.seed)}.wav"
                rate = int(outputs.get("sampleRate")
                           or audio_vae_mod.AudioVAEConfig().sample_rate)
                audio_report = media_mod.write_wav(waveform, audio_target, sample_rate=rate)
            report = media_mod.write_video(frames, target, fps=int(plan.fps or 24),
                                           value_range="-1..1", audio=audio_target,
                                           metadata=self._artifact_metadata(request, plan, "video"))
            #: ⚠️ 这两道闸门在 ``write_video`` **之后**才判 ✓ ⇒ 一旦失败，盘上已经留了一条**帧数不对**
            #:   的 mp4（＋可能还有 wav ✓）。而清单还没写 ✓、上游拿到的是异常 ✓ ⇒ 若把文件留在
            #:   渲染目录里，它就成了一件"看着正常、其实作废"的产物 ✓✗（㉕ 那条不变量：「失败路径
            #:   一个文件都没写」✓）。所以先把这次刚落的文件删掉再抛 ✓。
            def _discard_written() -> None:
                for item in (target, audio_target):
                    try:
                        if item is not None and Path(item).exists():
                            Path(item).unlink()
                    except OSError:
                        pass              # 删不掉也不能把真正的错误信息盖掉 ✓
            #: ⭐⭐ **「读数缺失」不许被读成「核对通过」** ✗✗（2026-11-08 补 ✗）：
            #:   ``nb_frames`` 只在封装**写了 index** 时才有 ✓（分片 mp4 / 某些编码器 / 重封装中间态
            #:   全都没有 ✓），``probe`` 如实给 ``None`` ✓。而旧的闸门写的是
            #:   ``encoded_frames is not None`` ✓✗ —— 读不到就**整条跳过** ✓，回执里却照样写着
            #:   「**真 mp4** ✓（ffprobe 可复核 ✓）」✓✗ ⇒ 「帧数逐帧核过了」与「根本没核」**同形** ✓。
            #:   三层表态：``nb_frames`` 直读 ✓ → ``时长 × fps`` 估（±1 帧容差 ✓）→ 都没有则
            #:   **明说没核过** ✓（``framesVerified: None`` ＋ note 里点名 ✗）。
            #: ⭐⭐ **判定只许写一处** ✗✗（2026-11-08 收口 ✗）：上面那三层原本在**这里**又实现了一遍 ✓，
            #:   而 ``chain.run_chain`` 落成片走的是 ``media.write_video`` ✓ ⇒ 那条路**一次都没核过
            #:   帧数** ✓✗（它只在张量层守恒 ✓、不问编码器有没有丢帧 ✓）。同一件事两个地方各写一半 ⇒
            #:   最典型的下场是一处改了另一处不改 ✓。现在唯一真源是 :func:`media.frame_verdict` ✓
            #:   （``write_video`` 内已 ``report.update(...)`` ✓），这里**只读字段、不重算** ✓：
            #:   ``framesVerified``（``"nb_frames"`` 直读 ✓ / ``"duration×fps"`` **估的** ✓ /
            #:   ``None`` **没核过** ✗）、``framesMatch``（``False`` 才拦 ✓；``None`` ＝**没有可比读数**
            #:   ✗ ⇒ 既不拦 ✗、也**不许读成通过** ✗ ⇒ 只写 note ✓）、``framesTolerance``（直读 0 / 估 ±1 ✓）。
            probed_verified = report.get("framesVerified")
            probed_encoded = report.get("encodedFrames")
            encoded_frames = probed_encoded
            #: ⚠️ ``expected_frames`` 为 0 时（plan 没声明帧数 ✓）``framesMatch`` 仍可能是 True/False ✓，
            #:   但那条不是"与计划不符"（没有计划 ✓）⇒ 只在有计划时才表态 ✓。
            frames_checked: str | None = probed_verified if expected_frames else None
            if expected_frames and report.get("framesMatch") is False:
                guessed = probed_verified == "duration×fps"
                _discard_written()
                if guessed:
                    #: ⚠️ 估出来的值**只容 ±1 帧** ✓（容器时长是 float 秒 ✓，round 会上下抖一帧 ✓）：
                    #:   超出就不是"读数抖了"，是**真丢/真补了帧** ✗ ⇒ 照样不许静默落盘 ✓（容差在
                    #:   ``frame_verdict`` 里判 ✓，这里只报 ✓）。
                    raise media_mod.MediaError(
                        f"落盘后的帧数**估出来**与计划不符 ✗：时长×帧率 ≈ {int(probed_encoded)} 帧 ✓，"
                        f"plan.frames = {expected_frames} ✓（{target} ✗；已删掉这条废片 ✓；"
                        f"ffprobe 没给 nb_frames ✓ 才用时长效估 ✓ —— 差这么多不是舍入 ✓）")
                # ffprobe 读得到 nb_frames 时必须与 plan 一致 ✓（编码器丢/补帧同样是静默错 ✓✗）
                raise media_mod.MediaError(
                    f"落盘后的帧数与计划不符 ✗：mp4 里 {int(probed_encoded)} 帧 ✓，"
                    f"plan.frames = {expected_frames} ✓（{target} ✗；已删掉这条废片 ✓）")
            # ⚠️ 2026-11-01 修 ✗✗（接 SEEDVR2 修复线时踩到 ✓）：这里原先把 ``synthetic=True``
            #    与「画面来自**未训练**的参考 VAE ✗」**写死** ✓ —— 5 维帧此前只可能来自参考 VAE ✓，
            #    接上真视频 VAE（SEEDVR2 ``seedvr2-real`` ✓）之后这句话就成了假证 ✓✗：
            #    手里握着**真 VAE 解码**的成品像素 ✓，落盘回执却说"噪声" ✓（教训与 4 维 PNG 那次
            #    同形 ✓：`synthetic` 必须由**产出方**（`decode`）说了算 ✓，不在这里替它表态 ✓）。
            synthetic = bool(outputs.get("synthetic", True))
            #: ⚠️ 2026-11-04 修 ✗✗（H3 端到端实测踩到 ✓）：``synthetic`` 是**画面 ∧ 音频**的合取 ✓
            #    （见 `_decode_dual` ✓），而这里原先把 ``synthetic`` 直接当成"画面的成色"来写话 ✓✗ ——
            #    于是 H3 **视频侧已经是真 VAE**（``vaeKind="h3-real"`` ✓、24ch/16×/4× ✓）、
            #    只有**音频侧**还没真件（``audioVaeKind="reference-untrained"`` ✓）时，
            #    落盘回执照样写「画面来自未训练的参考 VAE」✗ —— **画面这半句是假证** ✓✗。
            #    口径：各自按**各自的 kind** 表态 ✓（都是 REAL_VAE_KINDS 才算真 ✓），
            #    不替任何一侧代言 ✓；`synthetic` 本身仍原样透传 ✓（它是合取、不会被这句改 ✓）。
            vae_kind = str(outputs.get("vaeKind") or "")
            audio_kind = str(outputs.get("audioVaeKind") or "")
            video_real = vae_kind in self.REAL_VAE_KINDS
            audio_real = audio_kind in self.REAL_VAE_KINDS
            picture_note = (
                (f"，画面来自**真 VAE 解码** ✓（`{vae_kind}` ✓）" if vae_kind else
                 "，画面来自**真 VAE 解码** ✓") if video_real else
                ("，画面来自**未训练**的参考 VAE ✗" if vae_kind else
                 "，画面来自**没挂 VAE** 的占位解码 ✗"))
            if waveform is not None:
                sound_note = (
                    (f"；音频来自**真音频 VAE** ✓（`{audio_kind}` ✓）" if audio_kind else
                     "；音频来自**真音频 VAE** ✓") if audio_real else
                    "；音频仍是**参考实现**的噪声 ✗（`synthetic` 因此为 True ✗ —— 与画面无关 ✓）")
            else:
                sound_note = ""
            result: dict[str, Any] = {
                "synthetic": synthetic, "realTensors": True, "realFile": True,
                "videoPath": str(target), "primaryPath": str(target),
                "video": report,
                #: ⭐ "**核过帧数**"与"**读不到所以没核**"必须分开 ✓✗（见上面 ``frames_checked`` 那段 ✓）：
                #:   ``nb_frames``（直读 ✓）/ ``duration×fps``（估的 ✓）/ ``None``（**没核过** ✗）。
                #:   原先这里什么都没有 ⇒ 回执只有「真 mp4 ✓（ffprobe 可复核 ✓）」✓✗ —— 读者会把它
                #:   读成"帧数也核过了" ✓，而真相是"ffprobe 没给 nb_frames、闸门整条跳过" ✓。
                "framesVerified": frames_checked,
                "encodedFrames": (int(encoded_frames) if encoded_frames is not None else None),
                #: ⭐⭐ 帧率同样三档 ✓✗（2026-11-10 补 ✗，判据在 :func:`media.compare_fps` ✓）：
                #:   ``"exact"`` ✓ / ``"drift"`` ✗（帧率被改了 ✓）/ ``None`` ✗（**盘上没读到** ⇒
                #:   「有没有被改」**不知道** ✓，不是"一致" ✗）。缺了这两格 ⇒ 「24 fps 封好 ✓」
                #:   与「按 12 fps 封了 ✗」在同一条回执里同形 ✓（每格都"有值"、都 ffprobe 可复核 ✓）。
                "fpsVerified": report.get("fpsVerified"),
                "fpsRequested": report.get("fpsRequested"),
                "artifacts": [{"kind": "video", "path": str(target), "bytes": report["bytes"]}],
                "note": "**真 mp4** ✓（ffprobe 可复核 ✓）" + picture_note + sound_note,
            }
            #: ⭐⭐ 「跑完才知道的结论」还要能**被看见** ✗✗（2026-11-09 补 ✗）：``note`` 是一整段
            #:   中文长文 ✓（用户很少读完 ✓），而画布上真正显示的是 ``warnings`` ✓ —— 那是
            #:   ``result.plan.warnings`` ✓（**提交前**的计划警告 ✓），跑完才知道的话此前**只**进
            #:   ``note`` ✓✗ ⇒ 「音轨几乎是静音 ✗」「mp4 里音轨被截掉 0.6 秒 ✗」这些在界面上
            #:   **一句都看不到** ✓✗（后端说了、没人看见 ＝ 白说 ✓）。
            #:   ⇒ 凡是要人**拔腿去查**的结论，同时 append 进 ``result.alerts`` ✓（前端把它并进
            #:   warnings ✓；REST 用户也能直接读 ✓）。返回原句，好让 ``note`` 那句与它**同源** ✓
            #:   （两处各写一遍必然哪天漂移 ✗）。
            def _alert(sentence: str) -> str:
                result.setdefault("alerts", []).append(sentence)
                return sentence

            if expected_frames and frames_checked == "duration×fps":
                result["note"] = (str(result["note"])
                                  + f"；帧数按时长×帧率**估**得 {int(encoded_frames)} 帧 ✓"
                                    f"（ffprobe 没给 ``nbFrames`` ✓ ⇒ 与计划的 ±1 帧内算对上 ✓；"
                                    "要逐帧实证得重封装修 index ✓）")
            elif expected_frames and frames_checked is None:
                result["note"] = (str(result["note"])
                                  + _alert("；⚠️ 落盘帧数**没核对过** ✗（ffprobe 既没给 ``nbFrames`` ✓、"
                                           "也读不出可用的时长/帧率 ✓）—— 「真 mp4 ✓」不等于"
                                           "「帧数对得上」✗"))
            if report.get("fpsVerified") == "drift":
                #: ⚠️ 这一句要人**拔腿去查**（不是"多看一眼"✓）⇒ 同时进 ``alerts`` ✓：
                #:   帧率被改 ⇒ **音画不同步** ✗、成片时长也不对 ✗，而画面上完全看不出来 ✓
                #:   （它就是"正常播" ✓），回执里此前每格都"有值" ✓。
                result["fpsOnDisk"] = report.get("fps")
                result["fpsDrift"] = report.get("fpsDrift")
                result["note"] = (str(result["note"])
                                  + _alert(str(report.get("fpsNote")
                                               or "；⚠️ 盘上帧率与请求的**不一致** ✗")))
            elif report.get("fpsVerified") is None:
                #: ⚠️ 「**没读到**帧率」同样不许安静放过 ✗✗（2026-11-10 补 ✗，与上面那条
                #:   「落盘帧数没核对过」**同一规矩** ✓）：``None`` 不是"一致" ✗ —— 它是
                #:   「这次**没核**」✓。不喊的话，回执里 ``fps`` 那格是 ``None`` ✓、
                #:   而 ``durationSeconds`` / ``videoSeconds`` 都"有值" ✓ ⇒ 读起来像一切正常 ✓✗。
                result["note"] = (str(result["note"])
                                  + _alert("；⚠️ 盘上帧率**没读到** ✗（``r_frame_rate`` 缺失 ✓）"
                                           "⇒「帧率是不是请求的那个」**不知道** ✗"
                                           "（这不是「帧率一致」✓✗）"))
            if report.get("metadataDropped"):
                # ⚠️ "写了元数据"与"元数据在文件里"是两件事 ✓✗（mp4 muxer 会静默丢掉不认识的键 ✓）
                #: ⭐⭐ 这句也**必须进 ``alerts``** ✗✗（2026-11-10 补 ✗，与图像那侧 ``metadataMissing``
                #:   同口径 ✓）：2026-11-07 为 mp4 补了 ``metadataDropped`` ✓，可只写进 ``note`` ✓✗
                #:   ⇒ 前端一个 ``note`` 字都不读 ✓、画布上照样播得正常 ✓，用户永远不知道
                #:   "这条片子说不清出处" ✓。元数据是产物离开作业单后**唯一的自证** ✓。
                result["metadataDropped"] = list(report["metadataDropped"])
                result["note"] = (str(result["note"])
                                  + _alert("；⚠️ 容器元数据有键**没写进去** ✗（"
                                           + ", ".join(str(k) for k in report["metadataDropped"])
                                           + "）—— 这条片子脱离作业单后说不清出处 ✗"))
            if waveform is not None and audio_target is not None and audio_report is not None:
                # ⚠️ 2026-11-07 改 ✓：音轨**同时**进 mp4 ✓（``-af apad`` ＋ ``-shortest`` 以画面收尾 ✓）
                #    —— 此前只有"静音 mp4 ＋ 单独 wav" ✓✗：播放器里没声音、成片还得再合一次 ✓。
                #    wav **照旧另存** ✓：转写/配音对齐/换轨都要干净音频 ✓，且两条产物各能独立核对 ✓
                #    （``hasAudio`` 为真的 mp4 ＋ 原样 PCM 的 wav ✓，互相印证 ✓）。
                result["artifacts"] = list(result["artifacts"]) + [
                    {"kind": "audio", "path": str(audio_target), "bytes": audio_report["bytes"]}]
                result["audioPath"] = str(audio_target)
                result["audio"] = audio_report
                # ⚠️ 这是"本次真的产出了音频"的**铁证** ✓（不是"后端支持音频"那种能力声明 ✗）
                result["dualStream"] = True
                #: 回执里写明"mp4 自带音轨" ✓ —— 只剩一条 wav 时前端会**只看画面** ✓✗
                result["videoHasAudio"] = bool(report.get("hasAudio"))
                #: ⭐⭐ **「源音轨多长」读没读到，也是三档** ✗✗（2026-11-08 补 ✗，与帧数那条
                #:   同一个病根 ✓）：「音轨被截掉 1 秒」与「音轨有没有被截**不知道**」此前在回执里
                #:   **完全同形** ✓✗ —— ``write_video`` 里 ``if audio_seconds is not None:`` 之后没有
                #:   else ✓，读不到就整段跳过 ✓，而下面这句人话照旧写着「音轨补静音到画面长度 ✓」
                #:   ⇒ 一句**肯定**的话盖住了「这次没核」✓。现在 ``audioLengthVerified`` 四档
                #:   ＝ ``"wave"``（wav 头直读 ✓）/ ``"ffprobe"``（流时长兜底 ✓）/ ``"decoded"``
                #:   （全解码量到 ✓）/ ``None``（**三路都没读到** ✗）⇒ ``None`` 时**不许**复用那句
                #:   「补静音到画面长度」✓✗。
                #: ⭐⭐ 且 ``None`` 内部还分两档 ✓✗（2026-11-09 补 ✗）：``audioLengthMiss``
                #:   ＝ ``"too_long"``（源超过全解码上限 ⇒ **这次没量** ✓）/ ``"unreadable"``
                #:   （三路都失败 ✓）。这句人话此前**写死**成「读不出来」✗ ⇒ ``media.write_video``
                #:   刚分好的两档到这里被**拍平** ✓✗（下游把上游的表态吃掉，正是这条线的老病根 ✓）；
                #:   现在按那个机器码分两句 ✓ —— 不靠中文长句做字符串匹配 ✗。
                audio_how = report.get("audioLengthVerified")
                audio_miss = report.get("audioLengthMiss")
                result["audioLengthVerified"] = audio_how
                #: ⭐ 原因也**原样上去** ✓（``too_long`` / ``unreadable`` / ``None`` ✓）：回执外面
                #:   那句中文人话有两处（``audio_tail`` 与 ``audioTrimUnverified`` ✓），只靠中文
                #:   分不开"哪一句写了什么" ✗ —— 判据与前端都该读机器码 ✓（与 media 同一口径 ✓）。
                result["audioLengthMiss"] = audio_miss
                if not result["videoHasAudio"]:
                    audio_tail = "但**没混进 mp4** ✗（见 `videoHasAudio` ✓）"
                elif audio_how is None and audio_miss == "too_long":
                    audio_tail = ("并已混进 mp4 ✓ —— 音轨按 ``-t`` 收尾到**画面长度** ✓，"
                                  "但源音轨**没有量到底** ✗（超过全解码上限 ⇒ 这次**没量** ✓，"
                                  "不是读不出来 ✓）⇒ 有没有被截**不知道** ✗（这不是「没截断」✓✗）")
                elif audio_how is None:
                    audio_tail = ("并已混进 mp4 ✓ —— 音轨按 ``-t`` 收尾到**画面长度** ✓，"
                                  "但源音轨到底多长**读不出来** ✗ ⇒ 有没有被截**不知道** ✗"
                                  "（这不是「没截断」✓✗）")
                else:
                    audio_tail = "并已混进 mp4 ✓ —— 音轨补静音到画面长度、以画面收尾 ✓"
                result["note"] = (str(result["note"])
                                  + "＋ **真 wav** ✓（16-bit PCM 原样另存 ✓；"
                                  + audio_tail
                                  + "）")
                #: ⭐⭐ 音轨**在** ≠ 音轨**有声音** ✗✗（2026-11-09 补 ✗）：上面每一格（``videoHasAudio``
                #:   ✓、``audioLengthVerified`` ✓、``audioTruncated`` ✓）都只回答"这条流**存在**吗"✓，
                #:   没有一格回答"这段波形**有没有振幅**"✗ —— 于是**全零波形**与正常音频在回执里
                #:   同形 ✓✗：wav 照写 ✓、混进 mp4 照成 ✓、回执照说「真 wav ✓ 已混进 mp4 ✓」✓，
                #:   用户点开是一段**无声片**，而回执里一个提示都没有 ✗。
                #:   ⚠️ 人话必须把两件事**分开说** ✓：不是"音频没混进去"✗（那会让人去查 ffmpeg 参数 ✗），
                #:   而是"音轨在、但内容是静的"✓ —— 该查的是波形生成那一段 ✓。
                if audio_report.get("silent"):
                    _dbfs = audio_report.get("peakDbfs")
                    result["note"] = (str(result["note"])
                                      + _alert("；⚠️ 这条音轨**几乎是静音** ✗（"
                                               + (f"峰值 {_dbfs:g} dBFS" if _dbfs is not None
                                                  else "峰值就是 0：**每个样本都是零**")
                                               + " ⇒ 波形很可能**没生成出来** ✓；**不是**"
                                                 "「音频没混进去」✗ —— 音轨确实在 mp4 里 ✓，"
                                                 "只是内容是静的 ✓✗）"))
                #: ⭐⭐ **削顶**也得喊 ✓✗（2026-11-10 补 ✗，句源 :meth:`_clip_alert` ✓）：
                #:   ``clipped`` 早就算出来了 ✓ 但只躺在 ``audio`` 子字典里 ✓✗ ⇒ 整段爆音与
                #:   增益刚好在回执里同形 ✓；而且削顶**不会被判成 silent** ✗（峰值恒 1.0 ✓ 更"响" ✓）
                #:   ⇒ 这是最容易被当成"正常"的那一格 ✓。
                _clip_sentence = self._clip_alert(audio_report)
                if _clip_sentence:
                    result["note"] = str(result["note"]) + _alert(_clip_sentence)
                #: ⭐⭐ **音轨被截这件事必须喊出来** ✗✗（2026-11-08 补 ✗）：``write_video`` 的 ``-t``
                #:   是单向的 ✓ —— 音轨比画面**短**是补静音（无损 ✓），比画面**长**是**截掉**（内容真丢 ✗）。
                #:   此前只有 ``audioMuxed: True`` ✓✗ ⇒ 「用户那段音频被剪掉 1 秒」与「音频本来就这么长」
                #:   在回执里**完全同形** ✓；而 ``probe`` 读的又是混完之后那条（被收过尾 ✓ ⇒ 永远自洽 ✓✗）。
                #:   现在单列一格 ✓：给两侧真实秒数 ✓、指明 wav 未动 ✓、并写明"mp4 里听不到后半段" ✗。
                if report.get("audioTruncated"):
                    result["audioTruncated"] = {
                        "videoSeconds": report.get("videoSeconds"),
                        "sourceSeconds": report.get("audioSourceSeconds"),
                        "trimmedSeconds": report.get("audioTrimSeconds"),
                        "wavPath": result.get("audioPath"),
                        "why": "音轨比画面长 ⇒ 混流用 ``-t``（画面长度）收尾 ✓，"
                               "**多出来的那截没进 mp4** ✗（内容丢失 ✗ 不是口径问题 ✓）；"
                               "源 wav 原样未动 ✓ —— 要全长读 ``wavPath`` ✓",
                    }
                    result["note"] = (str(result["note"])
                                      + _alert(f"；⚠️ 音轨比画面长 {report.get('audioTrimSeconds')} 秒 ⇒ "
                                               "**mp4 里被截掉** ✗（wav 是全长 ✓，见 `audioPath` ✓）"))
                elif result["videoHasAudio"] and audio_how is None:
                    #: ⚠️ 反过来那一半 ✗✗：**读不到**就单列一格 ＋ 一句人话 ✓ —— 不许安静放过 ✓
                    #:   （安静放过等于替用户担保「你的音频整段都在」✗，而这次根本没量 ✓）。
                    #: ⭐ ``why`` 也得分两档 ✓✗（2026-11-09 补 ✗）：原先写死「**两路**都没读到
                    #:   ＋ ``wave`` 只认 PCM、ffprobe 的流时长也缺」✓✗ —— 那句话在今天的实现上
                    #:   有**两处**是假话 ✓：① 早就有**第四级全解码** ✓（不是两路 ✓）；② 源过长
                    #:   那次是**没量** ✗（不是"读不出来" ✓）。机器码单列 ``reason`` ✓，人和前端
                    #:   都能直接读 ✓，不必猜中文 ✓。
                    result["audioTrimUnverified"] = {
                        "videoSeconds": report.get("videoSeconds"),
                        "wavPath": result.get("audioPath"),
                        "reason": audio_miss or "unreadable",
                        "why": (
                            "源音轨**超过全解码上限** ⇒ 这次**没量** ✗（不是读不出来 ✓）："
                            "标准库 ``wave`` 只认 PCM ✓、ffprobe 的音频流时长也缺 ✓、"
                            "容器时长又不能当音轨长度用 ✗，而全解码被成本闸门拦下 ✓ "
                            "⇒ 「有没有被 ``-t`` 截掉」**不知道** ✗；这不是「没截断」✓✗ —— "
                            "源 wav 在 ``wavPath`` ✓，要确认就单独量它 ✓"
                            if audio_miss == "too_long" else
                            "源音轨时长**三路都没读到** ✗（标准库 ``wave`` 只认 PCM ✓；"
                            "ffprobe 的音频流时长也缺 ✓；连全解码都没给出可用时长 ✓）"
                            "⇒ 「有没有被 ``-t`` 截掉」**不知道** ✗；"
                            "这不是「没截断」✓✗ —— 源 wav 在 ``wavPath`` ✓，要确认就单独量它 ✓"),
                        "note": report.get("audioLengthNote"),
                    }
                    result["note"] = (str(result["note"])
                                      + _alert("；⚠️ 源音轨**没有量到底**（超过全解码上限 ⇒ 这次没量 ✓）"
                                               "⇒ mp4 里那截**有没有被截不知道** ✗"
                                               "（要确认就单独量 `audioPath` 那条 wav ✓）"
                                               if audio_miss == "too_long" else
                                               "；⚠️ 源音轨时长**读不出来** ⇒ mp4 里那截"
                                               "**有没有被截不知道** ✗"
                                               "（要确认就单独量 `audioPath` 那条 wav ✓）"))
            self._write_manifest(where, request, plan, result, outputs)
            return result
        if waveform is not None and hasattr(waveform, "shape"):
            #: ⭐ **纯音频**（Music3 ✓，2026-11-06 补 ✗✗）：上面两条都是"有画面"的分支 ✓ ——
            #: 音频族**没有帧** ✓（``decode`` 的 ``frames`` 是 ``None`` ✓），于是此前会一路掉到
            #: 最后的 JSON 清单 ✓✗：VAE 真解出了波形 ✓、回执照样写"占位产物 ✗" ✓、盘上只有一个
            #: ``torch_stub_seed*.json`` ✓ —— 与 4 维 PNG 那次（Z-Image ✓）是**同一个坑** ✓✗。
            #: ⚠️ 顺序必须**在** 5 维之后 ✓：H3 双流是 5 维 ＋ 音频 ✓，它要的是 mp4 ＋ 另落 wav ✓。
            target = where / f"audio_seed{int(request.seed)}.wav"
            rate = int(outputs.get("sampleRate") or 44100)
            report = media_mod.write_wav(waveform, target, sample_rate=rate)
            synthetic = bool(outputs.get("synthetic", True))
            result = {
                "synthetic": synthetic, "realTensors": True, "realFile": True,
                "audioPath": str(target), "primaryPath": str(target), "audio": report,
                "artifacts": [{"kind": "audio", "path": str(target), "bytes": report["bytes"]}],
                "note": ("**真 wav** ✓（" + ("真 DAV ✓ ⇒ 这就是成品音频 ✓" if not synthetic
                                             else "DAV 未接 ✗ ⇒ 只是占位波形 ✓")
                         + f"；{int(outputs.get('audioSamples') or 0)} 样本 @ {rate} Hz ✓"
                         + "；这一族**没有画面** ✓ —— 不要拿 ``primaryUrl`` 当视频用 ✗✗）"),
            }
            #: ⭐⭐ 出歌这条分支**更需要**这一格 ✓✗（2026-11-09 补 ✗）：纯音频**没有画面可佐证** ✓
            #:   ⇒ 一段全静音的 wav 若回执照说「真 wav ✓ 这就是成品音频 ✓」✓✗，用户只会怀疑
            #:   "我哪设错了"（歌词？步数？）✓，而真相是波形那一段没出东西 ✓。
            #:   ⚠️ 所以进 ``alerts`` ✓（会被前端显示 ✓），不只写在 ``note`` 长文里 ✓。
            if report.get("silent"):
                _dbfs = report.get("peakDbfs")
                _sentence = ("；⚠️ 这条音频**几乎是静音** ✗（"
                             + (f"峰值 {_dbfs:g} dBFS" if _dbfs is not None
                                else "峰值就是 0：**每个样本都是零**")
                             + " ⇒ 波形很可能**没生成出来** ✓；**不是**「文件没写成」✗ —— "
                               "wav 在 ``audioPath`` ✓，只是内容是静的 ✓✗）")
                result["note"] = str(result["note"]) + _sentence
                result.setdefault("alerts", []).append(_sentence)
            #: ⭐⭐ 出歌这条同样要报**削顶** ✓✗（2026-11-10 补 ✗）：音乐一整段超 [-1,1] ⇒
            #:   成品是一段破音 ✓，而它比"静音"更**像正常产物** ✗（``silent`` 恒假 ✓、
            #:   ``peakDbfs`` 还写着 0.0 dBFS ✓ 或更好看 ✓）⇒ 不喊就永远没人查 ✓。
            _clip_sentence = self._clip_alert(report)
            if _clip_sentence:
                result["note"] = str(result["note"]) + _clip_sentence
                result.setdefault("alerts", []).append(_clip_sentence)
            self._write_manifest(where, request, plan, result, outputs)
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
