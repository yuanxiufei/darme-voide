"""**torch 后端** —— 真张量实现（2026-09-17；2026-10-29 落定：``.venv`` = torch 2.14.1+cu130、CUDA 可用 ✓）。

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn

from app.core import cpu_budget

from . import audio_vae as audio_vae_mod
from . import device_policy
from . import dit as dit_mod
from . import geometry as geometry_mod
from . import inventory as inv
from . import keyframes as keyframes_mod
from . import loader
from . import media as media_mod
from . import residency as residency_mod
from . import scheduler as scheduler_mod
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


def int8_scale_plan(info: Any) -> dict[str, tuple[int, ...]]:
    """从权重头部挑出「该走 **int8 常驻**」的线性层 ✓ ⇒ ``{子模块名: scale 形状}`` ✓。

    判据只两条 ✓：键以 ``.weight`` 结尾 ✓、dtype 是 ``I8`` ✓；且**必须有配套 scale** ✓。
    ⚠️ 缺 scale 的**不在这里补** ✗ —— 留给装载闸口报「缺 scale」✓（这里塞个假尺度只是把
    真问题藏起来 ✓✗）。``weight_scale`` 与 ``.scale`` 两种写法盘上都有 ✓。
    """
    plan: dict[str, tuple[int, ...]] = {}
    for logical, tensor in info.tensors.items():
        if tensor.dtype != "I8" or not logical.endswith(".weight"):
            continue
        name = logical[: -len(".weight")]
        scale = info.tensors.get(f"{name}.weight_scale") or info.tensors.get(f"{name}.scale")
        if scale is None:
            continue
        plan[name] = tuple(int(dim) for dim in scale.shape)
    return plan


def _int8_bytes_of(module: Any, torch_mod: Any) -> int:
    """模块里 int8 参数总字节 ✓（＝「钉到 CPU 会占多少内存」✓）。"""
    total = 0
    for param in module.parameters():
        if param.dtype == torch_mod.int8:
            total += int(param.numel())
    return total


def _materialize_meta(module: Any, *, torch_mod: Any, device: Any = "cpu") -> int:
    """把还停在 **meta** 上的参数/缓冲区换成同形状同 dtype 的**未初始化**实存 ✓ ⇒ 返回换了几个 ✓。

    ## 为什么不能用 ``module.to_empty()`` ✗✗（2026-11-01 实测 ✓）

    32B 的 H3 文本塔在 meta 上建壳之后，``to_empty`` 会把**整份 62 GiB bf16** 一次性分配出来 ✓✗
    —— 而那 350 个 ``nn.Linear`` 紧接着就被 :func:`h3_form.upgrade_linear_to_int8` 换成
    int8 常驻层（22.7 GiB ✓）✓。**换掉不等于还给系统** ✗：实测进程 WS 停在 **74 GiB**、
    系统物理内存只剩 **6 GiB**（共 93 GiB ✓）⇒ 后面的装载全在页错误里爬 ✓✗。

    ⇒ 这里只碰 ``device.type == "meta"`` 的那些 ✓：换过壳的 int8 层**已经不在 meta 上** ✓，
    一根 byte 都不会被重复分配 ✓。调用点必须排在 ``upgrade_linear_to_int8`` **之后** ✗。

    ⚠️ 分配出来是**未初始化**的 ✓（``torch.empty`` ✓）—— 这是**故意**的：值一律由
    :func:`weights.load_module_weights` 逐张量写入 ✓，而它报 ``complete=False`` 时装载方当场拒 ✓
    （少写一个键 ⇒ ``missing`` 非空 ✓），所以"没填的壳"不可能被当成模型跑 ✓✗。
    """
    moved = 0
    for sub in module.modules():
        for name, param in list(sub._parameters.items()):
            if param is None or param.device.type != "meta":
                continue
            sub._parameters[name] = torch_mod.nn.Parameter(
                torch_mod.empty(param.shape, dtype=param.dtype, device=device),
                requires_grad=bool(param.requires_grad))
            moved += 1
        for name, buffer in list(sub._buffers.items()):
            if buffer is None or buffer.device.type != "meta":
                continue
            sub._buffers[name] = torch_mod.empty(buffer.shape, dtype=buffer.dtype, device=device)
            moved += 1
    return moved


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
    """
    import torch  # noqa: PLC0415

    layers = sum(1 for sub in module.modules() if type(sub).__name__ == "Int8Linear")
    int8_params = [param for param in module.parameters() if param.dtype == torch.int8]
    resident = int(getattr(report, "int8Resident", 0) or 0)
    ledger = {
        "layers": layers,
        "int8Params": len(int8_params),
        #: 留在 CPU 的那份 ✓ —— 这就是 `int8_offload` 要省下的显存 ✓（单位见 `parkedBytes` ✓）
        "int8OnCpu": sum(1 for param in int8_params if param.device.type == "cpu"),
        "int8OnGpu": sum(1 for param in int8_params if param.device.type != "cpu"),
        "int8Resident": resident,
    }
    if layers != len(int8_params):
        return ledger, (
            f"int8 常驻层 {layers} 个、int8 参数 {len(int8_params)} 个 ✗ ⇒ 结构自相矛盾"
            f"（一层应当恰好一个 int8 权重 ✓）")
    if resident < layers:
        return ledger, (
            f"文件里只有 {resident} 个 int8 权重被**原样**装进常驻层，而计划换掉的是 {layers} 层 ✗ ⇒ "
            f"差的 {layers - resident} 层会按 bf16 落盘（显存 2 倍 ✓✗），"
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

#: ⭐ **只出图、潜变量 4 维** 的族 ✓（``(1, C, H/scale, W/scale)`` ✓ **没有时间维** ✗）。
#: 单一真源 ✓：``latent_shape`` / ``_latent_mode`` / ``denoise`` **三处都读它** ✓ ——
#: 手抄成三份的下场是"加了新族只改了两处"，而漏掉的那处**不会报错** ✗（形状对、数是噪声 ✓✗）。
IMAGE_ONLY_FAMILIES: tuple[str, ...] = (ZIMAGE_FAMILY, QWEN21_FAMILY, QWEN_IMAGE_FAMILY)


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


def _build_zimage_module(info: Any, explicit: dict[str, Any],
                         *, dtype: Any = None) -> tuple[Any, dict[str, Any]]:
    """Z-Image 族 ⇒ 按**权重形状**建 :class:`dit_lumina.LuminaDiT` ✓ ⇒ ``(module, config)`` ✓。

    ``config`` 返回 **dict** ✓（与 H3 形态同一种形态 ✓ ⇒ :meth:`TorchBackend.describe` 认 dict ✓；
    旧 DiT 那个 dataclass 形态是另一条路 ✓ 两边都得认 ✓ 见 ``_configured_text_dim`` ✓）。

    ⚠️ ``patch`` **推不出来** ✗：``x_embedder`` 的入维是 ``channels × patch²`` ✓ —— 只知道乘积、
    分不出「16 通道 patch=2」与「64 通道 patch=1」✓✗，两者都能建成功、前向都算得出数 ✓✗
    （错了只会让出片与潜变量错位 ✗）。⇒ 默认 2（ComfyUI ``z_image`` 的构造口径 ✓）但**必须
    在报告里写明"这是默认值、不是从权重读出来的"** ✗（``configSources`` 那套口径 ✓）。

    ⚠️ 建的时候直接给**计算精度** ✗（有 CUDA ⇒ bf16 ✓）：Z-Image bf16 权重约 8 GiB ✓，
    fp32 建一遍就是 16 GiB ✓✗ —— H3 那条 78 GiB 的教训同一个 ✓（见 :func:`_prepare_h3_module` ✓）。
    """
    import torch  # noqa: PLC0415

    from app.services.engine import dit_lumina as lumina_mod  # noqa: PLC0415

    shapes = {name: tuple(entry.shape) for name, entry in info.tensors.items()}
    patch = explicit.get("patch", explicit.get("patch_size"))
    source = "explicit" if patch is not None else "default"
    patch = int(patch or 2)
    #: ⚠️ ``dtype`` 由调用方按**实际设备**给 ✓✗（``self._compute_dtype()`` ✓）—— 这里若按
    #: ``cuda.is_available()`` 自己判 ✓，在有卡而 `device="cpu"` 时会建成 bf16 ✓✗（CPU 上慢/缺算子 ✓）。
    dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    try:
        module = lumina_mod.build_from_shapes(shapes, patch=patch, dtype=dtype)
    except lumina_mod.ShapeError as err:
        raise TorchBackendUnavailable(
            f"Z-Image 权重推不出可装的结构 ✗（**不拿出厂形状硬装** ✗ —— 那会「名字对、形状错」✓✗）："
            f"{err}", reason="pending") from err
    shape = lumina_mod.infer_shape(shapes, patch=patch)
    config: dict[str, Any] = dict(shape.__dict__)          # dataclass ⇒ 字段名即权重事实 ✓
    config["family"] = ZIMAGE_FAMILY
    # ⚠️ 文本特征宽度就叫 ``cap_dim`` ✓（Lumina 的命名 ✓）—— 但后端协议问的是 ``text_dim`` ✗
    #    ⇒ 这里显式映射一次 ✓，否则 `_configured_text_dim` 会**悄悄回落**到默认值 ✓✗
    #    （TE 的 output_dim 与主干对不上 ✓，那条路 2026-09-20 踩过 ✓）。
    config["text_dim"] = shape.cap_dim
    # ⚠️ 后端协议问的是 ``in_channels`` / ``vae_scale`` ✗，而 Lumina 的字段叫 ``channels`` ✓ 且
    #    **根本没有 vae_scale** ✗ ⇒ 不补这两个，`latent_shape` 会在 dict 上 ``getattr`` 取不到 ✓✗
    #    ⇒ 静默回 ``None`` ⇒ 造**一维占位**潜变量 ⇒ 真 DiT 前向当场形状错 ✓✗（而报告会写"已装模型"✓）。
    config["in_channels"] = int(shape.channels)
    #: Z-Image 的 VAE 是 **Flux** 那一档 ✓（``latent_formats.Flux`` ✓ ``spatialScale=8`` ✓ 见
    #  ``vae_flux`` 模块头 ✓）—— 这不是从 DiT 权重读出来的 ✗，是**同族事实** ✓ ⇒ 记进 ``configSources`` ✓。
    config["vae_scale"] = 8
    config["configSources"] = {"patch": source, "结构": "权重形状反推 ✓",
                               "vae_scale": "同族事实（Flux VAE，8× ✓）"}
    return module, config


def _build_qwen21_module(info: Any, explicit: dict[str, Any],
                         *, dtype: Any = None) -> tuple[Any, dict[str, Any]]:
    """Qwen-Image 2.1 族 ⇒ 按**权重形状**建 :class:`dit_qwen21.Qwen21DiT` ✓ ⇒ ``(module, config)`` ✓。

    ⚠️ 三件"权重里写着、但后端协议按**别的名字**问"的事实，必须在这里显式搬一次 ✓
    （少搬一个都是**静默**的 ✗ —— 后端 ``getattr`` 取不到就回落默认值 ✓✗）：

    * ``context_dim`` ⇒ ``text_dim``（TE 输出宽度 4096 ✓；不搬 ⇒ ``_configured_text_dim`` 回落 ⇒
      TE 与主干对不上，报错指不到这里 ✗）；
    * ``channels`` ⇒ ``in_channels``（**64** ✓ 不是 16 ✗ —— 2.1 的 VAE 直出 64ch ✓，
      ``img_in`` 就是 ``Linear(64 → dim)`` ✓，**没有 conv patchify** ✗）；
    * ``vae_scale = 16``（**同族事实** ✓ 见 ``latent_formats.QwenImage21.spacial_downscale_ratio`` ✓）
      —— 这一档**权重里没有** ✗，也不是 8 ✗（用 8 ⇒ 潜变量边长翻倍 ⇒ 真前向当场形状错 ✓✗）。

    ⚠️ **meta 建、再 to_empty** ✗✗：2.1 是 32 层 × 4096 ⇒ bf16 约 14 GiB ✓，fp32 建一遍就是 28 GiB ✓
    （与 H3 那 78 GiB 的同一条教训 ✓）。``with torch.device("meta")`` 只建元数据 ✓，``.to(dtype)``
    在 meta 上**只改 dtype、不分配** ✓，``to_empty`` 才按目标 dtype 真正分配 ✓ ⇒ 峰值只剩"真正那一份" ✓。
    """
    import torch  # noqa: PLC0415

    from app.services.engine import dit_qwen21 as q21_mod  # noqa: PLC0415

    shapes = {name: tuple(entry.shape) for name, entry in info.tensors.items()}
    heads = explicit.get("heads", explicit.get("num_heads"))
    #: ``axes_dim`` 权重里没有 ✗ ⇒ 默认走族事实 (16,56,56) ✓；只给自检缩结构时覆盖 ✓
    axes = explicit.get("axes_dim")
    try:
        shape = q21_mod.infer_shape(shapes,
                                    heads=int(heads) if heads else None,
                                    axes_dim=tuple(int(x) for x in axes) if axes else None)
    except q21_mod.ShapeError as err:
        raise TorchBackendUnavailable(
            f"Qwen-Image 2.1 权重推不出可装的结构 ✗（**不拿出厂形状硬装** ✗ —— 那会「名字对、形状错」✓✗）："
            f"{err}", reason="pending") from err
    #: 同上：精度跟**实际设备**走 ✓（`self._compute_dtype()` ✓ 传进来 ✓）。
    dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    with torch.device("meta"):
        module = q21_mod.build(shape)
    module.to(dtype)                     # meta 上只改 dtype ✓（不分配 ✓）
    module.to_empty(device="cpu")        # 到这里才按目标 dtype 分配 ✓
    config: dict[str, Any] = dict(shape.__dict__)      # dataclass ⇒ 字段名即权重事实 ✓
    config["family"] = QWEN21_FAMILY
    config["text_dim"] = int(shape.context_dim)
    config["in_channels"] = int(shape.channels)
    config["vae_scale"] = 16
    config["configSources"] = {
        "结构": "权重形状反推 ✓",
        "heads": "权重算出（to_q 输出宽 ÷ norm_q 宽 ✓）" if heads is None else "explicit",
        "text_dim": "权重事实（`txt_in.in_layer` 输入宽 ✓）",
        "vae_scale": "同族事实（`latent_formats.QwenImage21`：64ch / 16× ✓）",
    }
    return module, config


def _build_qwen_image_module(info: Any, explicit: dict[str, Any],
                             *, dtype: Any = None) -> tuple[Any, dict[str, Any]]:
    """Qwen-Image **2512**（t2i）/ **Edit 2511** 族 ⇒ 按形状建 :class:`dit_qwen_image.QwenImageDiT` ✓。

    与 :func:`_build_qwen21_module` 是**两条不同的路** ✗（族名 ``qwen_image`` vs ``qwen_image21`` ✓）：

    * 2512 是**双流 MMDiT** ✓（``attn.to_q`` ＋ ``attn.add_q_proj`` ✓、每层 6 组调制 ✓）；
    * 潜变量 16 通道 ✓、VAE 8× ✓ —— **不是** 2.1 的 64 通道 / 16× ✗（用错 ⇒ 潜变量边长翻倍 ✓✗）；
    * 时间口径 **``1000·σ`` 不翻转** ✓（2.1 是 lumina 式 ``1000·(1−σ)`` ✗）—— 这一格错了两族都
      装得上、都算得出，只是画面与提示词脱钩 ✓✗（见 :mod:`dit_qwen_image` 模块头 ✓）。

    三件"权重里写着、但后端协议按别的名字问"的事实，显式搬一次 ✓（与 2.1 那条同一个理由 ✓）：

    * ``context_dim`` ⇒ ``text_dim``（Qwen2.5-VL-7B 出 **3584** ✓ —— 不是 2.1 的 4096 ✗）；
    * ``channels`` ⇒ ``in_channels``（**16** ✓：``img_in`` 吃的是拍平后的 16×2×2 = 64 ✓）；
    * ``vae_scale = 8``（同族事实：``latent_formats.Wan21`` ✓ 见参考实现
      ``supported_models.QwenImage.latent_format`` ✓ —— 权重里没有这一格 ✗）。

    ⚠️ **Edit 2511（``__index_timestep_zero__`` ✓）的参考图通路尚未接** ✗✗：它要走 Qwen2.5-VL
    的视觉塔把参考图编成 token ✓ 再由 :meth:`TorchBackend.denoise` 传 ``ref_latents`` ✓ ——
    两段都还没有 ✓。⇒ 这里**照装**（键对账与形状必须现在就能核 ✓），但把事实写进
    ``configSources`` ✓，并让 :meth:`TorchBackend.denoise` 在 Edit 权重上**明确报错** ✓✗
    （静默按 t2i 出图 = 编辑能力凭空消失 ✓✗）。
    """
    import torch  # noqa: PLC0415

    from app.services.engine import dit_qwen_image as qimg_mod  # noqa: PLC0415

    shapes = {name: tuple(entry.shape) for name, entry in info.tensors.items()}
    heads = explicit.get("heads", explicit.get("num_heads"))
    axes = explicit.get("axes_dim")
    try:
        shape = qimg_mod.infer_shape(
            shapes,
            heads=int(heads) if heads else None,
            axes_dim=tuple(int(x) for x in axes) if axes else None,
            ref_method=explicit.get("ref_method"))
    except qimg_mod.ShapeError as err:
        raise TorchBackendUnavailable(
            f"Qwen-Image 权重推不出可装的结构 ✗（**不拿出厂形状硬装** ✗ —— 那会「名字对、形状错」✓✗）："
            f"{err}", reason="pending") from err
    #: 精度跟**实际设备**走 ✓（`self._compute_dtype()` ✓ 传进来 ✓）—— 20B 的族 fp32 建一遍是
    #: 两倍显存 ✓✗，所以这里与 2.1 一样走「meta 建 ⇒ 只改 dtype ⇒ to_empty 才分配」✓。
    dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    with torch.device("meta"):
        module = qimg_mod.build(shape)
    module.to(dtype)                     # meta 上只改 dtype ✓（不分配 ✓）
    module.to_empty(device="cpu")        # 到这里才按目标 dtype 分配 ✓
    config: dict[str, Any] = dict(shape.__dict__)      # dataclass ⇒ 字段名即权重事实 ✓
    config["family"] = QWEN_IMAGE_FAMILY
    config["text_dim"] = int(shape.context_dim)
    config["in_channels"] = int(shape.channels)
    config["vae_scale"] = 8
    config["configSources"] = {
        "结构": "权重形状反推 ✓",
        "heads": "权重算出（`attn.norm_q` 宽 = head_dim ✓）" if heads is None else "explicit",
        "text_dim": "权重事实（`txt_norm` 宽 / `txt_in` 出宽 ✓ = Qwen2.5-VL-7B 的 3584 ✓）",
        "vae_scale": "同族事实（`latent_formats.Wan21`：16ch / 8× ✓；权重里没有 ✗）",
        "ref_method": f"{shape.ref_method} ✓（Edit 哨兵 = `{qimg_mod.EDIT_SENTINEL}` ✓）",
        "editRefWired": "False —— 参考图通路（视觉塔 + 参考图 VAE 编码）未接 ✓【未接事实】"
                        if shape.ref_timestep_zero else "n/a（t2i 权重 ✓）",
    }
    return module, config


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
                 audio_latent_mode: str | None = None,
                 int8_offload: bool = False,
                 residency: str = "off",
                 residency_budget_mib: float | None = None,
                 policy: str | None = None) -> None:
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
        #: 显式显存预算 ✓（MiB ✓；None ⇒ 按**当时**空闲显存 ×(1−reserve) 现算 ✓ —— 见 `residency.budget_from_cuda`）
        self._residencyBudgetMib = _residency_budget_mib(residency_budget_mib)
        #: 联合驻留的把手 ✓（``place()`` 的返回值 ✓ —— 报告与 `detach()` 都靠它 ✓；off ⇒ None ✓）
        self._residencyHandle: Any = None
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
        #: 首帧条件**实际走了哪条路** ✓（真 VAE 编码 / 占位 ✓ —— 由管线回给调用方 ✓）
        self._conditioningNote: dict[str, Any] | None = None
        #: 超清放大器 ✓（按**需求装载** ✓ —— 没要超清就不装 ✓ 省显存）；装载报告与本次二采的实况 ✓
        self._upscaler: Any = None
        self._upscalerReport: dict[str, Any] | None = None
        self.refineDetails: dict[str, Any] | None = None
        #: 本实例**借来还没还**的 LoRA ✓（E2.5 ✓：借的是进程级 `lora_cache` 里那份共享张量 ✓，
        #: 同一个内容挂两次只占**一份**内存 ✓）；模型被换掉 / 卸载时必须 :meth:`release_loras` ✗✗
        self._loraLeases: list[Any] = []
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
    #: | DiT（Edit）| ``qwen_image_edit_2511_int8`` ✓ | 17.78 GiB ✓ | ≈**35.5** ✓ | int8 ⇒ ×2 ✓ |
    #: | 文本塔 | ``qwen_2.5_vl_7b_fp8_scaled`` ✓ | 8.74 GiB ✓ | ≈**16.1** ✓ | fp8 ⇒ ×2 ✓；且**只装语言塔** ✓（视觉塔 714 张量不建 ✗ ⇒ 略小于 ×2 ✓） |
    #: | VAE | ``qwen_image_vae`` ✓ | 0.24 GiB ✓ | **0.24** ✓ | 本体就是 bf16/fp32 权重 ✓ |
    #:
    #: ⚠️ **这一族在这张卡上注定装不满** ✗✗：DiT 单件 38.05 GiB > 卡 **22.49** ✓ ⇒ `auto` 会判
    #: `split`（按层切给卡＋内存 ✓，跨界前向慢 ✓）或 `cpu`（整件在内存 ✓，更慢 ✓）。**这是必付
    #: 成本** ✓，不是可省的优化 ✗ —— 报告里如实说 ✓，不许写"已优化" ✗。
    #: ⚠️ 拿 2.1 的 13.26 判它 ⇒ 结论"装得下" ✓✗、`residency` 留 `full` ✓、装到 20 GiB 才 OOM ✓。
    QWEN_IMAGE_PART_NEED_GIB: dict[str, float] = {"dit": 38.05, "text": 16.1, "vae": 0.24}

    #: **整条链同时驻留**的实测总账 ✓（GiB ✓）—— 与 :attr:`PART_NEED_GIB`（**分时**账 ✓）是两回事 ✗✗。
    #: ``zimage`` 那条线三件**同时在卡上** ✓✗（不像 2.1 那样 encode/sample/decode 各占一段 ✓）：
    #: DiT 11.46 ＋ Qwen3-4B TE 7.49 ＋ Flux VAE 0.33 ＝ **19.28** ✓，装配后 ``reserved`` 实测
    #: **19.39 GiB** ✓（差的 0.11 是 allocator 保留块 ✓）。
    #: ⚠️ 为什么需要这张表 ✗✗：光靠「有没有卡」判，卡上剩 10 GiB 也照样敢装 ✓✗ —— DiT 装得进去 ✓、
    #: 装 TE 时才 OOM ✓，报错落在 TE 上，看着像"文本塔坏了" ✓✗（实测过一次这种误导 ✓）。
    #: ⚠️ 没核过的族**不进表** ✗：查不到实测就传 ``0``（走占用率粗判 ✓），不编数字 ✗✗。
    CHAIN_NEED_GIB: dict[str, float] = {"zimage": 19.28}

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
        handle = residency_mod.place(module, main_device=self._device, budget=budget,
                                     mode=self._residency, guard=self._host_guard(what))
        self._residencyHandle = handle
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
            # ⭐ 「切完还剩多少在卡上」是**判断有没有意义**的那个数 ✓（0 层 ⇒ 等于纯 CPU ✗）
            "fits": handle.plan.fits,
        })
        if not applied:
            report["note"] = (
                f"预算 **{report['budgetMib']} MiB** ⇒ 没有可切的显存 ✓（`{handle.plan.reason}`）"
                "⇒ 整份留在内存里 ✓✗（**等于纯 CPU** ✓ —— 装载成功、但这一档没联合 ✗；"
                "卡被别人占着就等它空出来 ✓，或显式给 `residencyBudgetMib` ✓）")
        elif not handle.plan.fits:
            report["note"] = (
                f"⚠️ 预算（{report['budgetMib']} MiB）连**非层参数**都放不下 ✗ ⇒ 整份回落到 CPU ✓✗"
                f"（`plan.reason`：{handle.plan.reason}）—— 装载**成功** ✓，但这一步**没有联合** ✗，"
                "前向会全在内存里跑 ✓（把预算调大 / 换 `split`→`park` 再试 ✓）")
        elif handle.plan.layers_on_cpu and self._residency == "split":
            report["note"] = (
                f"卡上 {handle.plan.layers_on_gpu} 层 / CPU {handle.plan.layers_on_cpu} 层 ✓ ⇒ "
                "跨界只搬**激活** ✓（每步一次，代价小 ✓）；CPU 那几层在计算核上 ✓ 会比全卡慢 ✓")
        return report

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
        if family == ZIMAGE_FAMILY:
            module, config = _build_zimage_module(info, explicit, dtype=want_dtype)
        elif family == QWEN21_FAMILY:
            module, config = _build_qwen21_module(info, explicit, dtype=want_dtype)
        elif family == QWEN_IMAGE_FAMILY:
            module, config = _build_qwen_image_module(info, explicit, dtype=want_dtype)
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
        report = weights_mod.load_module_weights(module, target, device=self._device)
        if not report.complete:
            del module                        # 别让半装的模型在 raise 期间继续被引用 ✓
            raise TorchBackendUnavailable(
                f"权重与模型结构对不上 ✗（complete=False ⇒ **不是**完整模型）"
                f"　：{report.to_dict()}",
                reason="pending")
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
        if self._int8Offload:
            # ⚠️ **必须如实说"没生效"** ✗✗：`int8_offload` 只认 H3 形态的 `Int8Linear` 常驻层 ✓
            #    （见 :func:`_prepare_h3_module` ✓）；这条路把 int8 **反量化进计算精度** ✓✗
            #    ⇒ 一个 int8 参数都不会被 park ✓。静默忽略的后果是「以为省了 18 GiB 显存」✓✗，
            #    然后在 720×1280 上照样 OOM、且找不到原因 ✓✗
            #    （与 CPU 后端那条"无需生效"**同一个口径** ✓：不生效可以 ✓，不说不行 ✗）。
            self._loadReport["int8Residency"] = {
                "device": str(self._device), "parkedOn": None, "applied": False,
                "moved": 0, "parked": 0, "parkedBytes": 0, "pinned": False,
                "note": "DiT 形态没有 `Int8Linear` 常驻层 ⇒ `int8_offload` **不生效** ✓"
                        "（它只对 H3 形态的 int8 权重有意义 ✓；"
                        "这条路上的 int8 会被反量化成计算精度 ⇒ 显存按 2 倍算 ✓）"}
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

    def release_loras(self, *, reason: str = "") -> dict[str, Any]:
        """还掉本实例借的全部 LoRA ✓ ⇒ 报告 ✓（**模型卸载 / 换装配时必须调** ✗✗）。"""
        leases, self._loraLeases = self._loraLeases, []
        return self._release_leases(leases, reason=reason or "模型卸载 ⇒ 还 LoRA ✓")

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
        if getattr(self, "_ditFamily", None) in IMAGE_ONLY_FAMILIES:
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
            # ⭐ 模板**只在配置里加**一次 ✓（`Qwen3TeRuntime.prompt` ✓）；参考 TE 没这个方法 ✓ ⇒ 原样送 ✓。
            wrap = getattr(config, "prompt", None)
            for name, text in (("positive", request.prompt), ("negative", request.negative)):
                raw = "" if text is None else str(text)
                prepared = wrap(raw) if callable(wrap) else raw
                ids, original, truncated = te_mod.tokenize_prompt(
                    self._tokenizer, prepared, max_length=config.max_length)
                tensor = torch.tensor([ids], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    results[name] = self._textEncoder(tensor)
                if name == "positive":
                    positive_ids = list(ids)
                meta[name] = {"tokens": len(ids), "originalTokens": original,
                              "truncated": truncated, "templated": callable(wrap)}
            # ⚠️ `ids` 必须是**真的 id 列表** ✓（第一版把 meta 字典塞进了 `ids` ✗ ⇒ 调用方
            #    `len(...)` 拿到的是"字典键数"而不是 token 数 ✓ —— 自检 ⑯ 当场红 ✓ 已改 ✓）；
            #    逐分支明细放 `byBranch` ✓。
            return {**results, "ids": positive_ids, "tokens": len(positive_ids),
                    "originalTokens": meta["positive"]["originalTokens"],
                    "truncated": meta["positive"]["truncated"],
                    "byBranch": meta,
                    "textEncoder": self._textKind or "reference-untrained"}
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
        """
        #: ⚠️ 守门读 **`_config`** 不读 `_model` ✗✗（2026-10-31 实测踩到 ✓）：形状只需要
        #: 「权重反推出的结构事实」✓（几 KB ✓），**不需要 13 GiB 的模块在场** ✓。
        #: 分时驻留时 `init` 阶段跑在 **encode 之后** ✓ —— 那时 TE 在场、DiT 已被退掉 ✓，
        #: 读 `_model is None` ⇒ 静默退回**一维占位** ✓✗ ⇒ 采样第 1 步炸
        #: ``latents 要 (B,C,H,W)，实际 (512,)`` ✓✗（报错指向 DiT ✓，根因在换装 ✓）。
        if self._config is None:
            return None
        scale = int(self._config_value("vae_scale", 0) or 0)
        if not scale:
            return None
        channels = int(self._config_value("in_channels", 0) or 0)
        height = max(1, int(plan.height) // scale)
        width = max(1, int(plan.width) // scale)
        if getattr(self, "_ditFamily", None) in IMAGE_ONLY_FAMILIES:
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
                latent = self._vae.encode(frames.to(self._device))[0]
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
                    latent = self._vae.encode(frames.to(self._device))[0]      # [C, 1, h, w] ✓
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
            text_token_tags=getattr(request, "text_token_tags", None))

    def retake_latents(self, latents: Any, plan: Any, request: Any, *, condition: Any,
                       sigmas: Any, retake: Any, on_step: Any = None) -> dict[str, Any]:
        """**重拍**（retake ✓ 2026-09-26 移植 ✓）：把**成片**的双流潜变量按窗口重画一遍 ✓，
        窗外**逐位保留** ✓（自检钉住 ✓）。

        事实来源：``reference/ComfyUI-H3-Multishot/h3_retake.py``（MIT ✓）—— 上游是
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
            #: ⚠️ **Edit 2511 的参考图通路未接 ⇒ 拒绝按 t2i 出图** ✗✗（2026-10-31 显式拦 ✓）：
            #: 那份权重靠 `__index_timestep_zero__` 哨兵把"参考图 token"与"目标 token"拼在一条
            #: 序列里 ✓、并给参考图行喂 σ=0 的调制行 ✓；本方法**没有** `ref_latents` 这个口 ✗
            #: ⇒ 若照装照跑，参考图那一路**整段缺席** ✓✗，模型仍会画出"像提示词"的图 ✓
            #: —— 报 0 错、能力凭空消失 ✓✗（这正是 `dit_qwen_image` 模块头点破的那个坑 ✓）。
            if getattr(getattr(self._model, "shape", None), "ref_timestep_zero", False):
                raise TorchBackendUnavailable(
                    "装的是 **Qwen-Image Edit 2511** 的权重（带 `__index_timestep_zero__` 哨兵 ✓）✗，"
                    "但参考图通路（Qwen2.5-VL 视觉塔 + 参考图 VAE 编码 + `ref_latents` 入前向 ✓）"
                    "**尚未接** ✗ ⇒ 拒绝按纯 t2i 静默出图 ✓✗（那样编辑能力凭空消失、却一条报错都没有 ✓）。"
                    "要用编辑：先接视觉塔与 `ref_latents` ✓；要用纯文生图：换 2512 的 t2i 权重 ✓",
                    reason="pending")
            with torch.no_grad():
                if getattr(self, "_ditFamily", None) in IMAGE_ONLY_FAMILIES:
                    prediction = self._model(latents, _timesteps_of(torch, latents, sigma), condition)
                else:
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
            text_shape = vl_te_mod.infer_shape(shapes)
            vision_shape = vl_vision_mod.infer_shape(shapes)
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
        convrot, convrot_gap = quant_mod.convrot_group_of(loader.quant_specs_of(path, info))
        if convrot_gap:
            raise TorchBackendUnavailable(
                f"convrot 说明自相矛盾 ✗ ⇒ 不装载（不按猜的 group 算 ✗）：{convrot_gap}",
                reason="pending")
        plan = int8_scale_plan(info)
        if not plan:
            raise TorchBackendUnavailable(
                f"这份 H3 文本塔里**一个 int8 权重都没有** ✗（{len(info.tensors)} 键 ✓）⇒ "
                f"不像 `int8_convrot` 那份 ✓；按 bf16 整份装会要 50 GiB ✗ ⇒ **拒装** ✓"
                f"（真要用 bf16 版请显式换成 `nvfp4`/bf16 的那条路 ✓）", reason="pending")
        replaced = h3_form.upgrade_linear_to_int8(module, plan, convrot=convrot)
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
                   "textDim": runtime.output_dim,
                   "visionShape": dict(vision_shape),
                   "unexpectedCount": len(report_weights.unexpected)},
            note=("**真** Qwen3-VL-32B 双塔 ✓（语言塔 `model.*` ＋ 视觉塔 `visual.*` ✓，"
                  "一个文件一次装 ✓）；int8 常驻层 "
                  f"{int8_ledger['layers']} 个 ✓；`tags` 由 "
                  "`h3_encode.encode_condition` 按 `<|image_pad|>` 位置算 ✓"))

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
        """挂 **Wan 2.2 layout 的 64 通道 VAE** ✓ ⇒ Qwen-Image **2.1** 的潜变量能解成真图 ✓
        （见 :mod:`vae_wan` ✓；盘上 ``vae/qwen_image_2.1_vae_bf16.safetensors`` **238** 键 ✓）。

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
            raise TorchBackendUnavailable(
                f"Wan 系 VAE 权重不存在 ✗：{path} ✓（2.1 要的是 "
                f"``vae/qwen_image_2.1_vae_bf16.safetensors`` ✓，不是 qwen_image_vae ✗）",
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
        detail.update({"kind": "wan21-real", "complete": True, "path": str(path),
                       "dtype": str(want), "loadedKeys": int(report.loadedKeys),
                       "sourceKeys": int(report.sourceKeys), "latentsFamily": str(family),
                       "latentsMean": len(mean), "latentsStd": len(std),
                       "note": "**真** Wan 2.2 layout VAE ✓（238 键逐键对上 ✓）＋ 反缩放 "
                               f"``{family}`` 的 {len(mean)} 个 mean/std ✓ ⇒ 解码出的是真图 ✓"})
        return detail

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

        * ``text_kind`` ✓ —— ``"qwen35"``（2.1 ✓ Qwen3.5-9B ⇒ :meth:`attach_qwen35_te` ✓）
          或 ``"qwen25"``（2512 ✓ Qwen2.5-VL-7B ⇒ :meth:`attach_qwen25_te` ✓）。
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
                  "vae": self._vae}.get(part)
        if part not in ("dit", "text", "vae"):
            raise ValueError(f"不认识的分时部件 {part!r} ✗（只认 dit / text / vae ✓）")
        #: ⚠️ **只退「这一件」的把手** ✓：`_residencyHandle` 是单槽 ✓，它对应**最后装的那件** ✓；
        #: 若正好就是要退的这件 ✓（`module is target` ✓）⇒ 必须先 detach 才有资格谈释放 ✓✗。
        holder = self._residencyHandle
        if holder is not None and target is not None \
                and getattr(holder, "module", None) is target:
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
        #: 账本同步还回去 ✓（**无论有没有真退**都清 ✓：两处不一致比不记更糟 ✓✗）
        self._ledger.release(part)
        self._ledger.release(f"{part}#host")
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
        """这件**占了多少字节** ✓ ⇒ 读账本里现成的那一条 ✓（读不到就算 0 ✓）。"""
        entry = self._ledger.items.get(part) or self._ledger.items.get(f"{part}#host")
        return int(getattr(entry, "bytes", 0) or 0)

    def _relocate_keep_parts(self, kept: tuple[str, ...],
                             wanted: tuple[str, ...]) -> list[dict[str, Any]]:
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

        ``phase`` 取 ``encode`` / ``sample`` / ``decode`` ✓；别的名字（``plan`` / ``init`` /
        ``refine`` ✓）不换装 ✓ —— 2.1 的超清二采仍用同一个 DiT ✓（它在 ``sample`` 阶段装着呢 ✓）。

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
                   "vae": self._vae is not None}
        reused = [part for part in wanted if present[part]]
        if len(reused) == len(wanted):
            #: 要的已在场 ✓ ⇒ 只把**不用的**退掉 ✓（能退多少退多少 ✓；常驻件不算"用不着" ✓）
            released = [self.release_part(part) for part in present
                        if present[part] and part not in wanted and part not in kept]
            #: ⚠️ 已在场就**别再决策** ✗✗：重装一遍只为了换设备，等于把 14 GiB 白搬一次 ✓✗
            #: （`decode` 之后回 `sample` 那种回头路是少数 ✓，那时该由调用方决定要不要重装 ✓）。
            #: ⚠️ 但**要搬设备** ✓✗：H3 的 `decode` 正是"VAE 已在场（cpu ✓）而这一拍要它上卡"✓ ——
            #: 不搬就整个解码在 cpu 上跑 ✓（实测 94.6 秒 ✓，见 `_relocate_keep_parts` ✓）。
            relocated = self._relocate_keep_parts(kept, wanted)
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
        decision = self.decide_for(tuple(wanted), phase=phase,
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
                else:
                    self.attach_qwen35_te(chain["textWeights"],
                                          tokenizer_path=chain["tokenizerPath"])
            else:
                if str(chain.get("vaeFamily") or "") == "h3":
                    self._attach_h3_vae_weights(chain["vaePath"])
                else:
                    self.attach_wan_vae(chain["vaePath"], family=chain["vaeFamily"])
        #: ⭐⭐ **装完、且装载栈退干净之后**再还一次显存 ✓✗（这一句是 2026-11-01 端到端的关键 ✓）。
        #: 为什么装载函数自己末尾那句不够 ✗✗：那时它的**局部变量还活着** ✓（物化中转 ✓、
        #: `WeightLoadReport` 握着的引用 ✓）⇒ allocator 看那些块「还在用」⇒ `empty_cache()` **一个字节不还** ✓✗。
        #: 实测：`load_weights` 末尾 trim 之后 `reserved` 仍 36402 MiB ✓（活张量只有 4.02 GiB ✓）；
        #: 等栈退干净（探针里过了一秒 ✓）再 synchronize+empty_cache ⇒ 立刻 36402 → **4332** ✓、
        #: `free` 0 → **17457 MiB** ✓。所以这一句必须落在**调用方**、装完之后 ✓。
        #: ⚠️ 常驻件的设备迁移排在**装完之后** ✓✗：先把卡让空（上一件退掉 ✓、这一件装好 ✓），
        #: 再搬那 5 GiB 上去 ✓ —— 顺序反了就是"往满的卡里塞" ✓✗（OOM 报在 `module.to` 上 ✓）。
        relocated = self._relocate_keep_parts(kept, wanted)
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
                dit_channels = int(self._config_value("in_channels", 0) or 0)
                if dit_channels:
                    pairs.append((
                        "视频", "latent_channels", dit_channels,
                        "DiT 配置里的 `in_channels` ✓（= `x_embedder` 入维 ÷ patch² ✓；"
                        "Z-Image = 16 ✓）",
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
            alpha_dropped = False
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
                with torch.no_grad():
                    frames = vae_wan_mod.decode_latents(
                        self._vae, latents, family=family,
                        mean=latents_mean, std=latents_std, to_unit=False)
                # ⚠️ 这族的 ``head`` 出 **4 通道 RGBA** ✓✗（2026-10-31 实测：`media.write_image`
                #    只认 ``(B,3,H,W)`` ⇒ 落盘那一步才炸 ✓，报得还像"形状给错了" ✓✗）。
                #    第 4 通道是 alpha ✓，出图**必须丢** ✓ —— `decode_latents` 保留它只是为了把
                #    「4 通道」这个族事实留在张量上 ✓，投影到 RGB 是**这一步**的职责 ✓。
                alpha_dropped = int(frames.shape[1]) == 4
                if alpha_dropped:
                    frames = frames[:, :3]
                real = True
            else:
                with torch.no_grad():
                    frames = self._vae.decode(latents)
                real = self._vaeKind == "h3-real"
            # ⚠️ **图**（4 维 ✓）没有时间维 ✗ ⇒ 不能写死 ``shape[2]`` 当帧数 ✓✗（那是高度 ✓）。
            frame_count = int(frames.shape[2]) if len(tuple(frames.shape)) == 5 else 1
            # ⚠️ 族名要**逐族点名** ✗（此前是二选一 "Flux / 否则 H3" ✓✗ ⇒ 挂了 2.1 的 VAE 也会
            #    被写成 H3 ✓✗，报告里看不出到底哪族解的 ✓）。
            kind_label = {"flux-real": "Flux ✓",
                          "wan21-real": "Qwen-Image 2.1（Wan 2.2 layout，64 通道）✓"}.get(
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
        if hasattr(frames, "shape") and len(tuple(frames.shape)) == 4:
            # ⭐ **图**（4 维 ✓）落 PNG ✓ —— 2026-10-31 补 ✗：此前这里只认 5 维 ✓✗ ⇒ Z-Image 走完
            #    「真 TE ✓ → 真 DiT ✓ → 真 VAE ✓」之后，**最后一步落成了 JSON 清单** ✓✗
            #    （`synthetic: True` 还写着 ✓ —— 明明手里握着成品像素 ✓✗）。
            target = where / f"image_seed{int(request.seed)}.png"
            report = media_mod.write_image(frames, target, value_range="-1..1")
            synthetic = bool(outputs.get("synthetic", True))
            return {
                "synthetic": synthetic, "realTensors": True, "realFile": True,
                "imagePath": str(target), "primaryPath": str(target), "image": report,
                "artifacts": [{"kind": "image", "path": str(target), "bytes": report["bytes"]}],
                "note": ("**真 PNG** ✓（"
                         + ("真 VAE ✓ ⇒ 这就是成品图 ✓" if not synthetic
                            else "VAE 未训练 ✗ ⇒ 画面是噪声 ✓")
                         + "）"),
            }
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
