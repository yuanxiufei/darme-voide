"""**DiT（Diffusion Transformer）** —— 本项目自己的实现 ✓（真 `torch.nn.Module` ✓ 可独立验证 ✓）。

## 职责边界（与 :mod:`app.services.engine.torch_backend` 的分工 ✓）

这里只做**模型本身** ✓：张量进、预测出 ✓。**采样循环**在 :mod:`sampler` ✓、**引导**在 :mod:`guidance` ✓、
**首帧掩码**在 :mod:`conditioning` ✓、**权重体检/加载计划**在 :mod:`safetensors` / `loader` ✓ ——
彼此不重复 ✓（这也是"换后端不动算法"的前提 ✓）。

## 为什么能脱离 H3 真权重就写出来（✗ 不猜架构）

实现的是**公开的标准结构** ✓（patch embed + 逐块 adaLN-Zero 调制的注意力/MLP + 最终 adaLN 输出 ✓，
出自 DiT / PixArt / SD3 这一系公开论文 ✓ —— **不是**抄某个仓库的源码 ✗）：

* 结构由 :class:`DiTConfig` **显式给出** ✓（隐层/深度/头数/patch 尺寸 ✓）⇒ 本模块**不猜** H3 的配置 ✗；
* 配置可以从权重文件的 ``__metadata__`` 里读 ✓（:meth:`DiTConfig.from_metadata` ✓）——
  读到就用 ✓，读不到就要求调用方给 ✓（**缺就报错**，不编一个"看起来合理"的 ✗）；
* **流匹配的参数化转换**（velocity → x0 ✓）单独成函数 ✓ ⇒ 换预测目标不用改模型 ✓。

⚠️ 因此：**"能跑通"不等于"H3 能出片"** ✗ —— 中间还差
「H3 的张量命名 → 本模块模块名」的**映射表** ✓ 与**权重的该模型配置** ✓（两者都要等真权重 ✓）。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any

from . import safetensors as st

__all__ = ["DiTConfig", "DiTConfigError", "H3_SHAPE_FACTS", "H3_PACK_FACTS", "H3_STRUCTURAL_GAPS",
           "H3_GAPS_CLOSED", "build_dit", "flow_match_x0", "infer_config_from_info"]

#: ⭐ **H3 的结构事实**（2026-09-20 抄自 `reference/ComfyUI` ✓，逐条带出处 ✓ —— 不是猜的 ✗）。
#: 用途：① 让"真权重到手"那一刻**不必重新翻参考项目** ✓；② 让下面的缺口清单**可被机械核对** ✓。
#: 出处：`comfy/ldm/minimax/model.py:474-481`（默认值）、`comfy/model_detection.py:390-418`（从权重反推）、
#: `comfy/sd.py:1011-1017`（VAE 压缩比）、`comfy/text_encoders/llama.py:289-297`（TE ✓）。
H3_SHAPE_FACTS: dict[str, object] = {
    "hidden": 5376,             # 主干隐层 ✓
    "depth": 50,                # `blocks.N` 个数 ✓
    "heads": 56,                # 注意力头数 ✓
    "headDim": 128,             # ⚠️ 注意：56 × 128 = 7168 **≠** hidden 5376 ✓（见下面缺口 ①）
    "attnInnerDim": 7168,       # heads × headDim ⇒ H3 走 qkv_proj → out_proj 模式 ✓
    "ffnHidden": 14336,         # MLP fc1 输出 ÷2（SwiGLU ✓）
    "patchSize": (1, 2, 2),     # ✓
    "videoLatents": 24,         # 视频潜通道 ✓
    "audioLatents": 32,         # 音频潜通道 ✓
    "textDim": 5120,            # Qwen3-VL-32B 输出维 ✓
    "timeEmbedIn": 256,         # 时间步输入维 ✓
    "timeEmbedHidden": 5376,
    "timeEmbedOut": 2688,
    "eps": 1e-5,                # norm_eps / qk_norm_eps / final_norm_eps ✓
    "ropeInvFreqLen": 16,       # 3 轴 × 16 = 48 ⇒ 复制成 96 ✓
    "tokenRefinerLayers": 2,    # ✓
    "vaeScale": 16,             # 空间压缩比 ✓（时间：帧 17k+5 ↔ 潜 5k+2 ✓）
    "sigmaShiftVideo": 12.0,    # 采样 shift ✓（audio 3.0 ⇒ audio_scale = 12/3 = 4.0 ✓）
    "sigmaShiftAudio": 3.0,
}

#: ⚠️ **H3 打包序列的规格**（2026-09-20 读全参考实现后补 ✓ —— 与 `H3_SHAPE_FACTS` 分开存 ✓：
#: 那张表是「形状数字」✓，这张是「**行怎么排**」✓；混在一张表里会让两类断言互相牵连 ✗）。
#: ⚠️ 全部是**结构事实** ✓（本仓要自己实现 ✗ 不抄代码 ✓）；本仓目前**没有**打包层 ✗ ——
#: ⚠️ 且**故意不先建** ✗：按本仓判据「**没人调用的库不算功能**」✓，等 H3 形态 forward 落地时再建 ✓。
H3_PACK_FACTS: tuple[str, ...] = (
    "顺序：text | cond(关键帧) 或 refs | **audio** | **video** ✓（目标音频在目标视频之前 ✓ 且为最后两段 ✓）",
    "视频行数 = `latent_t × (lat_h//2)×(lat_w//2)` ✓（每个 2×2 patch 一行 ✓）",
    "音频行数 = `audio_t × 2` ✓（**立体声 channel-major** ⇒ 每行 = 一个声道的一帧 ✓）",
    "视频时间轴：每 token 跨度表 (1, 4, 4, 4, 4) 循环 ✓ × 5/3 ✓（`FRAME_PER_TOKEN`/`FRAME_RESCALE` ✓）",
    "三段模态 tag（modalities=3 ✓）：text/cond 同一档 ✓、audio、video 各自一档 ✓",
    "**无 attention mask**（整条序列互相可见 ✓）；去噪掩码是**另一回事**（per-2×2 行 ∈[0,1] ✓）✓",
    "采样：只喂**视频 σ** ✓，音频按 `time_shift_sigma(σ, 12.0, 3.0)` **逐帧换算** ✓（`schedules.py` ✓）",
)

#: ⚠️⚠️⚠️ **别在本模块补这 7 条** ✗✗（2026-09-20 加注 ✓）：H3 那条路**已经**由
#: `h3_form.H3FormTrunk` 独立实现 ✓（18 路 adaLN ✓ / 2D 行级序列 ✓ / `q_norm`·`k_norm` ✓ /
#: RoPE ✓ / 正弦时间嵌入 ✓ / RMSNorm+SwiGLU ✓ / fp32 头 + PDD ✓ —— 且后端按**招牌键**分流 ✓）。
#: 下面这张表是「**通用 DiT 相对 H3 差在哪**」的事实记录 ✓，用途是**防止**把 H3 数字填进
#: `DiTConfig` ✓（那会得到「名字对、形状错」的模型 ✓✗）—— **不是**待办清单 ✗。
#:
#: ⚠️⚠️ **本仓 DiT 与 H3 的结构缺口**（2026-09-20 交叉核对时发现 ✓ —— **必须显式记着** ✗，
#: 否则"把 H3 数字填进 DiTConfig"会得到**名字对、形状错**的模型 ✓✗，那比不填更坏 ✓）。
#: ⚠️ **本清单是"还剩几件"** ✓ —— 关掉的条目**移进** `H3_GAPS_CLOSED` ✓（列表要**只减不骗** ✗）。
#: ⚠️⚠️ **它在第 114 步被"补全"过一次** ✗✓：第 113 步只记了 5 条 ✓，直到把参考实现
#: **读全**（源码 + 单元测试 ✓）才发现还漏着 **RoPE / 正弦时间嵌入 / RMSNorm / SwiGLU /
#: fp32 输出头 / denoise mask** 等 ✗ —— 「**没记 ≠ 没有**」✓：清单**没写全**时，
#: 下一步会以为"只剩 3 件"而开工 ✗ ⇒ 结果又是「名字对、装不上」✗。
H3_STRUCTURAL_GAPS: tuple[str, ...] = (
    "adaLN：H3 `AdalnProj(expand=6, modalities=3)` = 18 路 ✓ **且按 per-token mod-row 分段应用** ✓ "
    "（`mod_segments` = [(start, stop, row)] ✓，row 可为逐 token 索引 ✓）；本仓只 6 路单模态 ✗",
    "打包序列：H3 是 **2D**（`[总行, 通道×patch]` ✓ **无 batch 维** ✗）的 "
    "`text | cond/refs | audio | video` ✓（目标音频在视频**之前** ✓ 且为最后两段 ✓）"
    "+ per-token 模态 tag + **无 attention mask** ✓；本仓是 (B,N,D) + 「先算文本 → pooled 调制」✗",
    "`q_norm` / `k_norm`：per-head RMSNorm（eps 1e-5 ✓）—— 本仓显式 qkv 形态**未实现** ✗",
    "RoPE：3 轴 ✓ partial split-half ✓ 坐标按**面积归一化**（`_axis_from_sqrt_area` ✓ ×32 ✓ endpoint=False ✓）"
    "—— 本仓是**学习式** `pos_embed` ✗（H3 无位置表 ✓）",
    "时间条件：H3 用**正弦** `TimeEmbedder(freq_dim→hidden→out)` ✓（cos 在前 ✓）且"
    "**每 token 独立 timestep**（`t = 1 − σ` ✓ 音频另按 shift 3.0 换算 ✓）；本仓是 `Linear(1,hidden)` ✗ 单标量 ✗",
    "归一化与 MLP：H3 全 **RMSNorm** ✓ + **SwiGLU** MLP（fc1 `hidden→2×ffn` ✓ 无 bias ✓）"
    "—— 本仓 LayerNorm + 非 SwiGLU ✗",
    "输出与掩码：H3 两个头存 **fp32** ✓（bias=True ✓）+ 支持**部分去噪**（per-2×2 行 ∈[0,1] ✓）"
    "+ 音频按**立体声 channel-major** 打包（`audio_t×2` 行 ✓）；本仓单头 + 无 mask ✗",
)

#: ✅ **已经关掉的缺口**（2026-09-20 ✓）—— 每条注明**靠什么关的** ✓：
#: * ① 注意力维度 ⇒ `DiTConfig.attn_dim` + 显式 `qkv_proj`/`out_proj` ✓（5376→3×7168 ✓、7168→5376 ✓）；
#: * ② 双输出 ⇒ `DiTConfig.audio_latents` + `final_layer.video_out`/`audio_out` ✓；
#: * ④ `condition_proj` + `token_refiner` ⇒ `DiTConfig.text_refiner_layers` ✓（键名逐字对齐 ✓）；
#: * ④′ **refiner 内部结构**（2026-09-22 ✓）⇒ **直接复用 `h3_form.TokenRefiner`** ✓（见下 ✓）：
#:   旧实现拿本仓 `DiTBlock`（带 adaLN ✗）充当 refiner ✗ ⇒ 键是 `token_refiner.blocks.N.adaln_proj.*` ✗
#:   （**多出参考没有的键** ✓✗）且缺 `attn.` 前缀 ✗ ⇒ 真权重装不上 ✓✗。
#: ⚠️ 关掉 ≠ **核过真权重** ✗：② 里的"音频第二个头"是**可装载的近似** ✓（H3 真形态是拼序列出 ✓，
#: 见上面第 2 条 ✓）；④′ 现在是**与参考源码逐字对齐** ✓（不是"待核"了 ✓）—— 但**仍未核过真权重** ✗
#: （「源码对齐」与「权重核过」是两件事 ✓，别混 ✓）。
H3_GAPS_CLOSED: tuple[str, ...] = (
    "① 注意力维度：attn_dim + 显式 qkv_proj/out_proj（键名/形状对齐 H3 ✓）",
    "② 双输出：final_layer.video_out + audio_out（近似：H3 真形态是拼序列 ✓）",
    "④ condition_proj + token_refiner：层数/键名对齐 ✓ + **内部结构已按参考对齐** ✓"
    "（2026-09-22 复用 `h3_form.TokenRefiner` ✓：RMSNorm ✓ **无 adaLN** ✓ "
    "`attn.{qkv_proj,q_norm,k_norm,out_proj}` ✓ SwiGLU `mlp.{fc1,fc2}` ✓ `final_norm` ✓；"
    "⚠️ 真权重仍未核 ✗）",
)


class DiTConfigError(ValueError):
    """配置不合法 / 推断不出来 ✓（**明确报错**，不编一个默认值糊过去 ✗）。"""


@dataclass(frozen=True)
class DiTConfig:
    """模型结构参数 ✓（**全部显式** —— 本模块不猜 ✗）。"""

    hidden: int = 256
    depth: int = 4
    heads: int = 4
    #: patch 尺寸 ``(T, H, W)`` ✓（视频潜变量一般 1×2×2 ✓）
    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 4
    text_dim: int = 512
    mlp_ratio: float = 4.0
    #: 文本条件以 **cross-attention** 进入（True ✓）还是只做 pooled 调制（False ✓）
    cross_attention: bool = False
    #: **像素 → 潜空间**的边长比 ✓（8 = 每 8×8 像素一个潜token ✓）。
    #: ⚠️ ``0`` = **未给** ✓ ⇒ 这时**不能**从 plan 推潜变量形状 ✓（本该如此：压缩比是 VAE 的知识 ✗，
    #: 本模块**不猜** ✓ —— 与 `geometry` 不猜时间压缩比是同一条纪律 ✓）。
    vae_scale: int = 0
    #: **注意力内部维度** ✓（``hidden → attn_dim`` 的 qkv 投影 + ``attn_dim → hidden`` 的输出投影 ✓）。
    #: ⚠️ **默认 ``0`` = 与 ``hidden`` 相同** ✓（= 旧行为 ✓，走 ``nn.MultiheadAttention`` ✓，
    #: 键名 ``attn.in_proj_weight`` ✓ 不变 ✓）。非 0 且 ≠ ``hidden`` 时改用**显式**
    #: ``qkv_proj`` / ``out_proj`` ✓ —— **这正是 H3 的形态** ✓：见 `H3_SHAPE_FACTS`，
    #: heads **56** × headDim **128** = **7168** ≠ hidden **5376** ✓，而 ``MultiheadAttention``
    #: 只会推 ``hidden // heads`` = **96** ✗ ⇒ **根本装不上真权重** ✗
    #: （2026-09-20 交叉核对数字时发现 ✓ 见 `H3_STRUCTURAL_GAPS` 第 ① 条 ✓）。
    attn_dim: int = 0
    #: **音频潜通道数** ✓（``0`` = 单输出 = 旧行为 ✓）。
    #: 非 0 ⇒ 走 **H3 的双输出形态** ✓：``final_layer.video_out``（hidden→pT·pH·pW·in_channels ✓）
    #: + ``final_layer.audio_out``（hidden→``audio_latents`` ✓）—— **键名与 H3 逐字一致** ✓
    #: （它的检测器就是按 ``final_layer.audio_out.weight.shape[0]`` 读音频潜通道的 ✓）。
    #: ⚠️ H3 真形态里音频是**与视频 token 拼在同一条序列**上出的 ✓（见 `H3_STRUCTURAL_GAPS` ⑤ ✓）；
    #: 这里的"第二个头"是**可装载的近似** ✓，语义差异写在缺口清单里 ✓（不假装等价 ✗）。
    audio_latents: int = 0
    #: **文本 refiner 层数** ✓（``0`` = 无 = 旧行为 ✓）。
    #: 非 0 ⇒ 走 H3 的文本侧形态 ✓：``condition_proj``(text_dim→hidden ✓) + ``token_refiner.blocks.N``
    #: （每层自注意力 ✓）—— 同样**键名与 H3 逐字一致** ✓（H3 是 2 层 ✓）。
    #: ⭐ 2026-09-22：refiner **整个复用 `h3_form.TokenRefiner`** ✓（那边已与参考逐字对齐 ✓：
    #: RMSNorm ✓ **无 adaLN** ✓ `attn.{qkv_proj,q_norm,k_norm,out_proj}` ✓ SwiGLU ✓ `final_norm` ✓）
    #: ⇒ 内部结构**不再是"近似"** ✓；⚠️ 与全仓同一条纪律：**源码对齐 ≠ 权重核过** ✗。
    text_refiner_layers: int = 0

    def __post_init__(self) -> None:
        if self.hidden % self.heads:
            raise DiTConfigError(f"hidden={self.hidden} 不能被 heads={self.heads} 整除 ✗")
        if self.hidden <= 0 or self.depth <= 0:
            raise DiTConfigError("hidden/depth 必须为正 ✗")
        if self.attn_dim < 0:
            raise DiTConfigError(f"attn_dim 不能为负（收到 {self.attn_dim} ✗）")
        if self.attn_dim and self.attn_dim % self.heads:
            raise DiTConfigError(
                f"attn_dim={self.attn_dim} 不能被 heads={self.heads} 整除 ✗"
                f"（H3 是 56 头 × 128 维 = 7168 ✓）")
        if self.audio_latents < 0 or self.text_refiner_layers < 0:
            raise DiTConfigError(
                f"audio_latents / text_refiner_layers 不能为负 ✗"
                f"（收到 {self.audio_latents} / {self.text_refiner_layers} ✓）")
        if self.audio_latents and not self.attn_dim:
            # 双输出是 **H3 形态**的一部分 ✓ ⇒ 单独打开它只会得到"半套结构"✗（装不上真权重 ✓）
            raise DiTConfigError(
                "`audio_latents>0` 属 H3 双输出形态 ✓ ⇒ 必须同时给 `attn_dim` ✗"
                "（否则是「半套 H3」，装不上真权重 ✓）")
        if any(size <= 0 for size in self.patch_size):
            raise DiTConfigError(f"patch_size 必须为正（收到 {self.patch_size} ✗）")
        #: ⚠️ **两个自相矛盾的开关**：显式 qkv 形态 = 自注意力 ✓（H3 形态 ✓）⇒ 不能要 cross-attention ✗
        if self.attn_dim and self.attn_dim != self.hidden and self.cross_attention:
            raise DiTConfigError(
                "「显式 qkv_proj 形态」只做**自注意力** ✓ ⇒ 不能与 `cross_attention=True` 同时给 ✗"
                "（H3 的文本是**拼进同一条序列**的 ✓，见 `H3_STRUCTURAL_GAPS` ⑤ ✓）"
                "—— 宁可**构造时就报错** ✓，也不做出「能跑但装不上真权重」的假模型 ✗")

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def inner_dim(self) -> int:
        """注意力内部维度 ✓（``attn_dim`` 未给 ⇒ 等于 ``hidden`` ✓ = 旧行为 ✓）。"""
        return self.attn_dim or self.hidden

    @property
    def attn_head_dim(self) -> int:
        """每头维度 ✓（H3：``inner_dim // heads`` = 7168 // 56 = **128** ✓）。"""
        return self.inner_dim // self.heads

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden, "depth": self.depth, "heads": self.heads,
            "patchSize": list(self.patch_size), "inChannels": self.in_channels,
            "textDim": self.text_dim, "mlpRatio": self.mlp_ratio,
            "crossAttention": self.cross_attention, "vaeScale": self.vae_scale,
            "attnDim": self.attn_dim, "attnInnerDim": self.inner_dim,
            "attnHeadDim": self.attn_head_dim, "audioLatents": self.audio_latents,
            "textRefinerLayers": self.text_refiner_layers,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> "DiTConfig":
        """从 ``__metadata__`` 读配置 ✓（读到几个用几个 ✓；一个都没读到就**报错** ✗）。

        键名按各家习惯**多写法**匹配 ✓（``hidden`` / ``hidden_size`` / ``dim`` ✓）——
        这是"宽容读入" ✓，但**不猜值** ✗：没有的项必须由调用方显式给 ✓。
        """
        def pick(*names: str) -> Any:
            for name in names:
                for key, value in metadata.items():
                    if str(key).lower() == name:
                        return value
            return None

        def as_int(value: Any) -> int | None:
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                return None

        lowered = {str(key).lower(): value for key, value in metadata.items()}
        blob = lowered.get("dit_config") or lowered.get("config") or lowered.get("ditconfig")
        if isinstance(blob, str) and blob.strip().startswith("{"):
            try:
                parsed = json.loads(blob)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                merged = {**parsed, **metadata}
                return cls.from_metadata({str(k): str(v) for k, v in merged.items()})

        hidden = as_int(pick("hidden", "hidden_size", "dim", "model_dim"))
        depth = as_int(pick("depth", "num_layers", "n_layers", "num_blocks"))
        heads = as_int(pick("heads", "num_heads", "n_heads"))
        if not all((hidden, depth, heads)):
            raise DiTConfigError(
                "权重元数据里读不到完整结构（需要 hidden/depth/heads ✓）⇒ "
                "请在调用方**显式给 DiTConfig** ✓ —— 本模块**不猜**结构 ✗"
                f"（现有元数据键：{sorted(metadata)[:8]} ✓）")
        patch = pick("patch_size", "patch")
        patch_size = (1, 2, 2)
        if isinstance(patch, str):
            numbers = [as_int(part) for part in patch.replace("x", ",").split(",")]
            if len(numbers) == 3 and all(numbers):
                patch_size = (numbers[0], numbers[1], numbers[2])  # type: ignore[assignment]
        return cls(hidden=hidden, depth=depth, heads=heads, patch_size=patch_size,
                   in_channels=as_int(pick("in_channels", "latent_channels")) or 4,
                   text_dim=as_int(pick("text_dim", "context_dim")) or 512,
                   # ⚠️ `vae_scale` 也要读（自检 ⑲ 抓到漏读 ✗）—— 读不到就是 0 ✓（= 未给 ✓ 不猜 ✓）
                   vae_scale=(as_int(pick("vae_scale", "vae_scale_factor",
                                          "spatial_compression")) or 0),
                   # 注意力内部维度 ✓（H3 是 7168 = 56×128 ✓ ≠ hidden 5376 ✓）；读不到就是 0 ✓ = 未给 ✓
                   attn_dim=(as_int(pick("attn_dim", "attention_dim", "inner_dim")) or 0),
                   # H3 双输出与文本 refiner ✓；读不到就是 0 ✓（= 旧形态 ✓ 不猜 ✓）
                   audio_latents=(as_int(pick("audio_latents", "audio_latent_channels")) or 0),
                   text_refiner_layers=(as_int(pick("text_refiner_layers",
                                                    "token_refiner_layers")) or 0))


def infer_config_from_info(info: st.SafetensorsInfo, *,
                           base: DiTConfig | None = None) -> DiTConfig:
    """从**权重头部**能**机械确定**的那一项：**层数**（``blocks.N`` 的个数 ✓）。

    其余（hidden/heads ✓）**要读张量形状**才能定 ✗ ⇒ 那属于 :mod:`loader` 的层块报告 ✓，
    或者由调用方给 ✓。这里**只填空缺** ✓（``base`` 给的就是"调用方已知的部分" ✓）。
    """
    groups: dict[str, set[int]] = {}
    for name in info.tensors:
        head, _, tail = str(name).partition(".")
        if tail and tail.split(".")[0].isdigit():
            groups.setdefault(head, set()).add(int(tail.split(".")[0]))
    if not groups:
        raise DiTConfigError("权重头部里找不到 ``<blocks>.<N>`` 这样的层块命名 ⇒ "
                             "无法确定层数 ✓（请显式给 DiTConfig ✓）")
    head = max(groups, key=lambda name: len(groups[name]))
    slots = sorted(groups[head])
    expected = list(range(len(slots)))
    if slots != expected:
        raise DiTConfigError(
            f"层号不连续（{head}.N = {slots[:8]}… ✓）⇒ 权重很可能**被截断或拼错** ✗"
            f"（用 `loader.plan_component` 能看到同一结论 ✓）")
    depth = len(slots)
    if base is None:
        return DiTConfig(depth=depth)
    return replace(base, depth=depth)


def flow_match_x0(model_output: Any, latents: Any, sigma: float, *,
                  prediction: str = "velocity") -> Any:
    """流匹配下**从模型输出还原 x0** ✓（公开的标准转换 ✓）。

    * ``velocity``（流匹配/rectified flow ✓）：``x0 = x − σ·v`` ✓
    * ``epsilon``（σ 是噪声强度 ✓）：``x0 = x − σ·ε`` ✓
    * ``sample``（模型直接给 x0 ✓）：原样 ✓
    """
    sigma = float(sigma)
    if sigma <= 0:
        return latents
    kind = str(prediction or "velocity").lower()
    if kind in ("velocity", "v", "flow"):
        return latents - model_output * sigma
    if kind in ("epsilon", "eps", "noise"):
        return latents - model_output * sigma
    if kind in ("sample", "x0"):
        return model_output
    raise DiTConfigError(f"未知预测类型 {prediction!r}；可用：velocity / epsilon / sample ✓")


def _modules() -> tuple[Any, Any]:
    """懒导入 torch ✓（模块能 import 不依赖 torch ✓）。"""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    return torch, nn


def _build() -> tuple[type, type, type]:
    """构造 :class:`DiT` 用到的三个类 ✓（**模块级**定义会强制 import torch ✗ ⇒ 放进工厂 ✓）。"""
    torch, nn = _modules()

    class PatchEmbed3D(nn.Module):
        """视频潜变量 → token 序列 ✓（3D 卷积，kernel=stride=patch ✓）。"""

        def __init__(self, config: DiTConfig) -> None:
            super().__init__()
            self.proj = nn.Conv3d(config.in_channels, config.hidden,
                                  kernel_size=config.patch_size, stride=config.patch_size)

        def forward(self, latent: Any) -> Any:
            """``(B, C, T, H, W)`` → ``(B, N, D)`` ✓。"""
            out = self.proj(latent)
            return out.flatten(2).transpose(1, 2)

    class DiTBlock(nn.Module):
        """adaLN-Zero 调制的注意力 + MLP ✓（DiT 系的公开做法 ✓）。

        ⚠️ **两种注意力形态**（2026-09-20 为"能装 H3 权重"而开 ✓，判据见 `DiTConfig.attn_dim` ✓）：
        * ``attn_dim == hidden``（默认 ✓）⇒ ``nn.MultiheadAttention`` ✓ —— **旧行为一字不改** ✓
          （键名 ``attn.in_proj_weight`` ✓ 与既有预设/自检/往返装载全都照旧 ✓）；
        * ``attn_dim ≠ hidden`` ⇒ **显式** ``qkv_proj``(hidden→3·attn_dim ✓) + ``out_proj``(attn_dim→hidden ✓)
          —— **H3 的形态** ✓（5376 → 3×7168 ✓、7168 → 5376 ✓）。
        """

        def __init__(self, config: DiTConfig) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(config.hidden, elementwise_affine=False)
            self.heads = config.heads
            self.inner = config.inner_dim
            self.head_dim = config.attn_head_dim
            #: 是否需要**显式投影** ✓（= 是否走 H3 那种 attention 形态 ✓）
            self.explicit_attention = config.inner_dim != config.hidden
            if self.explicit_attention:
                self.qkv_proj = nn.Linear(config.hidden, 3 * self.inner)
                self.out_proj = nn.Linear(self.inner, config.hidden)
            else:
                self.attn = nn.MultiheadAttention(config.hidden, config.heads, batch_first=True)
            self.norm2 = nn.LayerNorm(config.hidden, elementwise_affine=False)
            inner = int(config.hidden * config.mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(config.hidden, inner), nn.GELU(),
                                     nn.Linear(inner, config.hidden))
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden, 6 * config.hidden))
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)   # adaLN-**Zero** ✓：起步是恒等 ✓

        def _attend(self, tokens: Any) -> Any:
            """显式 ``qkv_proj`` → 注意力 → ``out_proj`` ✓（**自注意力** ✓）。

            * ``qkv_proj`` 的输出是 **[q | k | v] 三段**（与 H3 一致 ✓ ——
              它的检测器就是按 ``qkv_proj.weight.shape[0] // (3·head_dim)`` 反推头数的 ✓）；
            * ⚠️ **本形态不做 cross-attention** ✗ ⇒ 给了 ``context`` 就**明确报错** ✓
              （H3 的文本是**拼进同一条序列**的 ✓，见 `H3_STRUCTURAL_GAPS` ⑤ ✓；
              硬装作支持 = 得到一个"能跑但装不上权重"的假模型 ✗）；
            * ⚠️ H3 的 ``q_norm``/``k_norm``（eps 1e-5 ✓）**暂未实现** ✗ —— 等真权重到手、
              确认归一化位置再补 ✓（**先不假装有** ✗）。
            """
            batch, length, _ = tokens.shape
            qkv = self.qkv_proj(tokens)
            query, key, value = qkv.chunk(3, dim=-1)
            shape = (batch, length, self.heads, self.head_dim)
            attended = torch.nn.functional.scaled_dot_product_attention(
                query.view(shape).transpose(1, 2),
                key.view(shape).transpose(1, 2),
                value.view(shape).transpose(1, 2))
            attended = attended.transpose(1, 2).reshape(batch, length, self.inner)
            return self.out_proj(attended)

        def forward(self, tokens: Any, cond: Any, context: Any = None) -> Any:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.modulation(cond).chunk(6, dim=-1)
            normed = self.norm1(tokens) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            if self.explicit_attention:
                if context is not None:
                    raise DiTConfigError(
                        "显式 qkv_proj 形态只做**自注意力** ✓（H3 无 cross-attention ✗）——"
                        "要跨注意力请用 ``attn_dim == hidden`` 的形态 ✓")
                attended = self._attend(normed)
            elif context is not None:
                attended, _ = self.attn(normed, context, context, need_weights=False)
            else:
                attended, _ = self.attn(normed, normed, normed, need_weights=False)
            tokens = tokens + gate_msa.unsqueeze(1) * attended
            normed = self.norm2(tokens) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            return tokens + gate_mlp.unsqueeze(1) * self.mlp(normed)

    # ⭐⭐ 2026-09-22：**不再自带一份 refiner** ✗ ⇒ 直接复用 `h3_form.TokenRefiner` ✓
    #      （**不抄第二份** ✗ —— 那份已与参考实现逐字对齐 ✓：RMSNorm ✓ **无 adaLN** ✓
    #      `attn.{qkv_proj,q_norm,k_norm,out_proj}` ✓ SwiGLU `mlp.{fc1,fc2}` ✓ + 收尾 `final_norm` ✓）。
    #
    #      ⚠️⚠️ 旧实现用的是本仓 :class:`DiTBlock`（**带 adaLN** ✗ ⇒ 键变成
    #      `token_refiner.blocks.N.adaln_proj.*` ✗）⇒ 既**多出参考没有的键** ✓✗、又**缺**
    #      `attn.` 这层前缀 ✓✗ ⇒ 真权重不但装不上，报错还是"缺一堆键 + 多一堆键"那种最难查的 ✓。
    #      ⚠️ `h3_keys` 早把「refiner 多出 adaLN」当 unexpected 报出来了 ✓（自检 `engine_h3_keys_test` ⑫ ✓）
    #      —— dit 这份**一直没跟上** ✗；这次对齐 ✓（自检 `engine_dit_test` ④¹¹ 原来钉的是**错的键名** ✗，
    #      也一并改正 ✓：「测试钉着的名字」不等于「参考的名字」✗）。
    from . import h3_form  # noqa: PLC0415 —— 懒导入 ✓（h3_form 模块级不 import torch ✓ 无环 ✓）
    TokenRefiner = h3_form.TokenRefiner

    class FinalLayerHeads(nn.Module):
        """H3 的 `final_layer` 输出头 ✓（``video_out`` + ``audio_out`` ✓ 键名逐字对齐 ✓）。

        * ``video_out``：``hidden → pT·pH·pW·in_channels`` ✓（H3 是 5376→96 = 2×2×24 ✓）
          —— 与它的检测器口径一致（``video_out.shape[0] // 4`` ⇒ 24 潜通道 ✓）；
        * ``audio_out``：``hidden → audio_latents`` ✓（H3 按 ``audio_out.shape[0]`` 读 32 ✓）；
        * ⚠️ H3 真形态里音频 token 与视频 token **在同一条序列**上出 ✓，且这一层还带
          ``adaln_proj.linear`` ✗ ⇒ 本头是**可装载的近似** ✓（缺口清单第 ③ 条 ✓）。
        """

        def __init__(self, config: DiTConfig) -> None:
            super().__init__()
            self.video_out = nn.Linear(
                config.hidden, math.prod(config.patch_size) * config.in_channels)
            self.audio_out = nn.Linear(config.hidden, config.audio_latents)

    class DiT(nn.Module):
        """**本项目自己的 DiT** ✓（结构显式来自 :class:`DiTConfig` ✓ 不猜 ✗）。"""

        def __init__(self, config: DiTConfig) -> None:
            super().__init__()
            self.config = config
            self.patch = PatchEmbed3D(config)
            # 位置编码：**按需插值** ✓（标准做法 ✓）。
            # ⚠️ 这里踩过一次：初版写死 ``(1, 4096, hidden)`` ✗ ⇒ 真实分辨率下 token 数
            #    是 **12920**（1088×608 潜变量 ✓）⇒ 直接 `size mismatch` 崩在采样里 ✗
            #    （自检 ㉗ 跑到整链才暴露 ✓ —— 这也是"必须端到端跑一次"的理由 ✓）。
            #    真权重若自带同长度的位置表 ⇒ 加载时形状一致就直接用 ✓；不同长度则插值 ✓。
            self.pos_embed = nn.Parameter(torch.zeros(1, 4096, config.hidden))
            self.sigma_embed = nn.Sequential(
                nn.Linear(1, config.hidden), nn.SiLU(), nn.Linear(config.hidden, config.hidden))
            # 文本侧：旧形态 = `text_proj` ✓；H3 形态 = `condition_proj` + `token_refiner.blocks.N` ✓
            # ⚠️ **键名逐字对齐 H3** ✓ ⇒ 装真权重时不用再改名字 ✓（见 DiTConfig.text_refiner_layers ✓）
            self.uses_refiner = config.text_refiner_layers > 0
            if self.uses_refiner:
                self.condition_proj = nn.Linear(config.text_dim, config.hidden)
                # ⚠️ eps 用参考默认 1e-5 ✓（`DiTConfig` 里**没有** eps 字段 ✗ —— 那是 H3 事实表
                #    里的 `norm_eps` / `qk_norm_eps` / `final_norm_eps` ✓；要改就在 `h3_form` 侧改 ✓）。
                self.token_refiner = TokenRefiner(
                    config.text_refiner_layers, config.hidden, config.heads,
                    config.attn_head_dim, int(config.hidden * config.mlp_ratio))
            else:
                self.text_proj = (nn.Identity() if config.text_dim == config.hidden
                                  else nn.Linear(config.text_dim, config.hidden))
            self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.depth)])
            self.final_norm = nn.LayerNorm(config.hidden, elementwise_affine=False)
            self.final_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(config.hidden, 2 * config.hidden))
            nn.init.zeros_(self.final_modulation[-1].weight)
            nn.init.zeros_(self.final_modulation[-1].bias)
            out_channels = math.prod(config.patch_size) * config.in_channels
            if config.audio_latents:
                # H3 双输出：`final_layer.video_out` + `final_layer.audio_out` ✓（键名对齐 ✓）
                self.final_layer = FinalLayerHeads(config)
            else:
                self.out = nn.Linear(config.hidden, out_channels)   # 旧形态 ✓（键名 `out` 不变 ✓）

        def forward(self, latent: Any, sigma: Any, context: Any = None) -> Any:
            """``latent (B,C,T,H,W)`` + ``sigma`` + 文本上下文 → **与 latent 同形状的预测** ✓。

            ``sigma`` 是标量或 ``(B,)`` 张量 ✓（每步变化 ⇒ 条件**真的**进网络了 ✓，
            自检会验证"改 sigma 输出就变" ✓ —— 防"条件没接进去"的假绿 ✗）。
            """
            batch = latent.shape[0]
            f_t, f_h, f_w = self.grid_sizes(latent)
            tokens = self.patch(latent) + self._positions(f_t * f_h * f_w, latent)
            sigma_tensor = torch.as_tensor(sigma, dtype=tokens.dtype, device=tokens.device)
            if sigma_tensor.ndim == 0:
                sigma_tensor = sigma_tensor.repeat(batch)
            cond = self.sigma_embed(sigma_tensor.reshape(batch, 1).to(tokens.dtype))
            text = None
            if context is not None:
                # ⚠️ 维度要**归一化**再进网络 ✓：调用方可能给一维向量（我们把条件当 pooled 用 ✓）、
                #    二维 (B,D) 或三维 (B,L,D) ✓。初版只处理 2/3 维 ⇒ 一维时 `mean(dim=1)` 直接
                #    `IndexError` ✗（自检 ㉗ 整链才暴露 ✓）。
                raw = torch.as_tensor(context, dtype=tokens.dtype, device=tokens.device)
                if raw.ndim == 1:
                    raw = raw.unsqueeze(0)              # (D,) → (1, D) ✓
                if raw.ndim == 2:
                    raw = raw.unsqueeze(1)              # (B, D) → (B, 1, D) ✓
                text = self.condition_proj(raw) if self.uses_refiner else self.text_proj(raw)
                if int(text.shape[0]) != batch:         # 单条条件广播到整个 batch ✓
                    text = text.expand(batch, *text.shape[1:])
                if self.uses_refiner:
                    # H3 的 refiner 是**文本侧自注意力** ✓（只处理文本 token ✓ 不碰潜变量 ✓，
                    # 也**不看 cond** ✗ —— 参考的 RefinerBlock 只收 x ✓ 无 adaLN 调制 ✓）。
                    # ⚠️⚠️ **形状口径不同** ✗：`h3_form` 的积木按 H3 的 **2D 打包行**（`(rows, hidden)` ✓
                    #    **无 batch 维** ✗）写 ✓；本模块走 `(B, L, D)` ✓ ⇒ **逐条**按 2D 喂进去 ✓。
                    #    ⚠️ 不能图省事整批 `reshape(-1, hidden)` ✗ —— 那会把 batch 之间也拉进同一条
                    #    序列里互相注意 ✓✗（`B=1` 的自检**发现不了** ✓✗，典型的"测试通过但语义错"✓）。
                    text = (torch.stack([self.token_refiner(row) for row in text], dim=0)
                            if text.ndim == 3 else self.token_refiner(text))
                cond = cond + text.mean(dim=1)          # pooled 调制 ✓
            for block in self.blocks:
                tokens = block(tokens, cond, text if self.config.cross_attention else None)
            shift, scale = self.final_modulation(cond).chunk(2, dim=-1)
            tokens = self.final_norm(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            if self.config.audio_latents:
                # 双输出 ⇒ **(视频, 音频)** ✓（音频是每 token 一个 `audio_latents` 向量 ✓）
                # ⚠️ 旧形态仍返回**单张量** ✓ ⇒ 调用方按 config 判形态 ✓（不静默换形状 ✗）
                return (self.unpatchify(self.final_layer.video_out(tokens), latent),
                        self.final_layer.audio_out(tokens))
            return self.unpatchify(self.out(tokens), latent)

        def unpatchify(self, tokens: Any, reference: Any) -> Any:
            """``(B, N, pT·pH·pW·C)`` → ``(B, C, T, H, W)`` ✓（**与输入同形** ✓）。

            维度命名（免得看错 ✓）：``pT/pH/pW`` = patch 尺寸 ✓；``fT/fH/fW`` = 网格数 ✓。
            顺序必须与 :meth:`patch_tokens` 一致（``fT·fH·fW`` ✓），否则画面会**错位但不报错** ✗。
            """
            patch_t, patch_h, patch_w = self.config.patch_size
            channels = self.config.in_channels
            batch = int(tokens.shape[0])
            f_t, f_h, f_w = self.grid_sizes(reference)
            grid = tokens.reshape(batch, f_t, f_h, f_w, patch_t, patch_h, patch_w, channels)
            return grid.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
                batch, channels, f_t * patch_t, f_h * patch_h, f_w * patch_w)

        def _positions(self, count: int, latents: Any) -> Any:
            """位置编码**插值到需要的长度** ✓（表长正好 ⇒ 直接用 ✓）。

            ``interpolate`` 沿 token 维做线性插值 ✓ —— 这是公开的常规做法 ✓（换分辨率不必重训位置表 ✓）。
            """
            table = self.pos_embed
            if int(table.shape[1]) == count:
                return table
            flat = table.transpose(1, 2)                    # (1, D, P) ✓
            resized = torch.nn.functional.interpolate(
                flat, size=int(count), mode="linear", align_corners=False)
            return resized.transpose(1, 2).to(dtype=latents.dtype, device=latents.device)

        def grid_sizes(self, latent: Any) -> tuple[int, int, int]:
            """潜变量 → 网格数 ``(fT, fH, fW)`` ✓（整除不了就报错 ✗，不悄悄向下取整 ✓）。"""
            sizes = []
            for index in range(3):
                size = int(latent.shape[index + 2])
                patch = self.config.patch_size[index]
                if size % patch:
                    raise DiTConfigError(
                        f"潜变量第 {index} 维 {size} 不能被 patch {patch} 整除 ✗"
                        f"（尺寸要吸附到网格 ✓）")
                sizes.append(size // patch)
            return sizes[0], sizes[1], sizes[2]

    return DiT, DiTBlock, PatchEmbed3D


_DIT_CLASS: Any = None


def build_dit(config: DiTConfig) -> Any:
    """构造 DiT 实例 ✓（**懒导入 torch** ✓ ⇒ 没装 torch 时 import 本模块也不会炸 ✓）。

    ⚠️ 刻意**不**把这个工厂命名成 ``DiT`` ✗ —— 那会与里面真正的类同名 ✓，
    调用方（和读代码的人）会分不清"类"和"工厂函数" ✗（第一版就是这样 ✓ 已改 ✓）。
    """
    global _DIT_CLASS
    if _DIT_CLASS is None:
        _DIT_CLASS = _build()[0]
    return _DIT_CLASS(config)
