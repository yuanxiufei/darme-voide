"""**SDXL 图片出图后端** —— 自研的「图」闭环（`GenerationBackend` 的一个实现 ✓）。

## 为什么有这一层

在此之前 `provider=engine` 出图只能拿 **H3（视频模型）** 凑静态图 ✗（`engine-still-from-video` ✓）。
四件套早就各自就位 ✓：`sdxl.py`（UNet 主干 ✓）、`clip_text.py`（双塔文本条件 ✓）、
`sdxl_vae.py`（潜空间解码 ✓ 2026-09-26 ✓）、`sampler.py` / `schedules.py`（σ 几何 ✓）
—— **缺的就是把它们按 SDXL 的口径串起来的那一层** ✓，本模块补它 ✓。

## σ↔t 口径（⚠️ 本模块最要紧的一条事实 ✓）

SDXL 的 UNet **吃的是整数时间步 ``t ∈ [0, 999]``** ✗，**不是**连续 σ ✓✗。把 σ 直接当 t 喂进去
**不会报错** ✓ —— 出来的是「结构像图、但整体是噪声/糊」的东西 ✓✗。所以：先把 σ 映射到 t ✓：

- **表** ✓：``betas`` 由 ``sqrt(linear_start) … sqrt(linear_end)`` 线性取平方 ✓（1000 点 ✓），
  ``alphas_cumprod = cumprod(1 − betas)`` ✓，``σ_i = √((1 − a_i) / a_i)`` ✓（**单调增** ✓：
  ``i=0`` 最干净 ✓、``i=999`` 最噪 ✓）；
- **映射** ✓：``t = argmin |ln σ − ln σ_i|`` ✓（**对数域**最近邻 ✓ —— σ 跨数量级，线性域会对不准 ✓）；
- **前向预条件（EPS ✓）**：`x_in = x / √(σ² + 1)` ✓、`x0 = x − ε·σ` ✓。

⚠️ 这套算式**照的是参考实现**（`ComfyUI/comfy/model_sampling.py` 的 ``EPS`` /
``ModelSamplingDiscrete`` ✓）—— **只当规格书读 ✓，代码是自己写的** ✗（GPL 不搬进本仓 ✗）。

## 装配（四个前缀，一份检查点 ✓）

官方 `sd_xl_base_1.0.safetensors` 一个文件里就有全部四件 ✓（前缀均是**真权重实测** ✓）：

| 部件 | 前缀 | 装载函数 |
|---|---|---|
| UNet | ``model.diffusion_model.`` | :func:`app.services.engine.sdxl.load_sdxl_unet_state_dict` |
| VAE | ``first_stage_model.`` | :func:`app.services.engine.sdxl_vae.load_sdxl_vae_state_dict` |
| CLIP-L | ``conditioner.embedders.0.`` | :func:`app.services.engine.clip_text.load_clip_text_state_dict` |
| CLIP-G | ``conditioner.embedders.1.`` | 同上（prefix 不同 ✓） |

⚠️ 四个**全部 `strict` 装载** ✓，任何一个缺键/形状不符 ⇒ **报错** ✗（不装半个模型 ✗）。
⇒ 四件都是**真权重** ✓ ⇒ :attr:`SdxlBackend.synthetic` 为 **False** ✓ —— 这是本模块与
`TorchBackend`（H3 的参考组件未训练 ⇒ ``synthetic=True`` ✓）**最重要的差别** ✓。

## 验证口径

`tests/engine_sdxl_vae_test.py`（VAE 层 ✓）与 `tests/engine_sdxl_backend_test.py`（本模块 ✓）：
σ↔t 映射的单调性/边界 ✓、EPS 前向的算术 ✓（拿恒等桩钉死 ✓）、尺寸吸附 ✓、
四前缀装配 ✓（真权重在盘上时**逐部件**核对键数 ✓）。⚠️ 真权重不在 ⇒ **显式 SKIP** ✓（没跑 ≠ 绿 ✗）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # ⚠️ torch 是**可选**依赖 ✓（纯数学部分没它也能跑能自检 ✓，同 `sdxl.py` / `sdxl_vae.py` ✓）
    import torch
except ImportError:  # pragma: no cover - 无 torch 环境只走纯数学自检 ✓
    torch = None  # type: ignore[assignment]

from app.core import cpu_budget

__all__ = [
    "SDXL_CLIP_G_KEY_PREFIX",
    "SDXL_CLIP_L_KEY_PREFIX",
    "SDXL_COMPUTE_DTYPE",
    "SDXL_DTYPE_NAMES",
    "SDXL_IMAGE_FACTS",
    "SDXL_LATENT_SCALE",
    "SDXL_SCHEDULES",
    "SDXL_SIGMA_TIMESTEPS",
    "SDXL_TRAINED_MEGAPIXELS",
    "SDXL_TRAINED_RESOLUTION",
    "SDXL_VAE_DTYPE",
    "SDXL_VAE_KEY_PREFIX",
    "SdxlBackend",
    "SdxlBackendError",
    "denoised_from_eps",
    "eps_input",
    "load_sdxl_components",
    "resolve_sdxl_dtype_names",
    "sigma_at_timestep",
    "sigma_to_timestep",
    "sdxl_sigmas_for_steps",
    "sdxl_sigmas_table",
]

#: SDXL 检查点里四件套的前缀（**真权重实测** ✓，不是猜的 ✓）。
SDXL_UNET_KEY_PREFIX = "model.diffusion_model."
SDXL_VAE_KEY_PREFIX = "first_stage_model."
SDXL_CLIP_L_KEY_PREFIX = "conditioner.embedders.0."
SDXL_CLIP_G_KEY_PREFIX = "conditioner.embedders.1."

#: 潜空间与像素的倍数 ✓（VAE 三次下采样 ⇒ 8 ✓；尺寸必须是 8 的倍数 ✓）。
SDXL_LATENT_SCALE = 8

#: SDXL 的**训练分辨率** ✓ —— **真权重头实测** ✓（不是查文档猜的 ✗）：同一份 ``__metadata__`` 里
#: ``modelspec.resolution = "1024x1024"`` ✓，且 ``prediction_type = "epsilon"`` ✓（后者正是
#: :func:`denoised_from_eps` 用的那套口径 ✓ ⇒ 两个事实互相印证 ✓）。
#: ⚠️ 它**不是**"建议值" ✗：**低于**这个像素预算会**不报错**地出废图 ✓✗ —— 实拍证据（同 seed / 同提示词 ✓）：
#: 512×512 ⇒ **物体重复 + 霓虹过饱和**（一长串苹果 + 荧光绿 + 紫黑深影 ✓），256×256 ⇒ **退化成纯色块** ✓；
#: 而这两种症状跟"实现写错了"长得**一模一样** ✓✗（属"看不出来"那类 ✓）⇒ 计划阶段**两个方向都收到**
#: 训练预算 ✓（见 :func:`pipeline._fill_image_plan` ✓）。顺带：它也是估显存的**基准点** ✓。
#: **高于**预算同样坏 ✓✗（实测 ✓）：1440×1440（2.07 MP）⇒ VAE 解码的 ``reserved`` 冲到
#: **30.66 GiB**，而物理显存只有 **22.49 GiB** ✗ ⇒ WDDM 共享显存**换页** ⇒ 光解码 **11.0 s**
#: （同条件 1024² 只要 **0.66 s** ✓）；把分配器换成 ``expandable_segments:True`` 后同一个 1440²
#: 只要 **1.23 s**、``reserved`` 回到 **20.03 GiB** ✓ ⇒ 说明那 11 s 是**分配器碎片**换页、不是算不动 ✓。
#: ⚠️ 本机环境里**预设**着 ``PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024`` ✓（就是制造碎片的那个 ✓，
#: 不是本仓设的 ✗）⇒ 换成 ``expandable_segments:True`` 是**环境**的事 ✓，代码**不替用户动环境** ✗。
#: 本仓"百万像素"口径 = ``宽·高/1e6`` ✓（见 :func:`geometry.megapixels_for_size` ✓）⇒ 1024×1024 = 1.0486 MP ✓。
SDXL_TRAINED_RESOLUTION = 1024
SDXL_TRAINED_MEGAPIXELS = SDXL_TRAINED_RESOLUTION * SDXL_TRAINED_RESOLUTION / 1_000_000.0

#: σ 表长度 ✓（SDXL 用 1000 步离散表 ✓）。
SDXL_SIGMA_TIMESTEPS = 1000

#: 可用的 SDXL 采样调度 ✓（口径 = 官方的**离散**调度器 ✓；新增请**同时**加自检 ✓）。
#: ⚠️ 刻意**不**复用视频那边的 `schedules.SCHEDULES` ✗（karras/linear 那些是**连续**调度 ✓，
#: 套到离散格上会算出落不到格的 σ ✓ ⇒ 见 :func:`sdxl_sigmas_for_steps` 的说明 ✓）。
SDXL_SCHEDULES: tuple[str, ...] = ("normal", "simple")

#: 官方 SDXL base 的 β 调度端点 ✓（LDM ``make_beta_schedule("linear")`` ✓）。
_SDXL_BETA_LINEAR_START = 0.00085
_SDXL_BETA_LINEAR_END = 0.012

#: 计算组件（UNet + 两座文本塔）在 CUDA 上的**默认精度** ✓ —— ⚠️ 这行是**实测**逼出来的 ✗，不是偏好 ✓。
#: **实测事实**（RTX A5000 / 物理 22.49 GiB ✓，真权重 `sd_xl_base_1.0.safetensors` ✓）：
#: 装配里**不传 dtype** ⇒ 三个 ``build_*`` 都走 ``model.to(device=…, dtype=None)`` ✓ ⇒ 模块**保持 fp32** ✗
#: ⇒ 权重 **12.92 GiB**（unet 9.56 + clipG 2.59 + clipL 0.46 + vae 0.31 ✓），而检查点**本身是 fp16** ✓
#: （`modelspec` 头 ✓ + 真权重读出的 dtype ✓）⇒ **白占一倍显存**（6.46 GiB ✓✗）。后果**实测**：
#: 1376×768 单张峰值 ``reserved=26.62 GiB > 22.49 GiB 物理`` ✓ ⇒ 走 WDDM 共享显存**换页** ⇒
#: 光解码 **13.78 s**（同量级 1024² 只要 **0.66 s** ✓）、同进程里第二张从 **30.0 s** 掉到 **82.7 s** ✓✗
#: —— 正撞用户「不许逼近显存/别长时间独占」的口径 ✗。⇒ 默认取 **fp16** ✓。
#: ⚠️ 从 fp16 权重读出的名字需与 ``torch`` 上的属性同名 ✓（见 :func:`resolve_sdxl_dtype_names` ✓）。
SDXL_COMPUTE_DTYPE = "float16"

#: VAE 的精度**钉死 fp32** ✗✗ —— 与上面**刻意不同** ✓：VAE 压到 fp16 会**静默出黑图/发灰** ✓✗
#: （见 `sdxl_vae` 模块头 ✓），属本仓最忌的「不报错但结果坏」✓ ⇒ 宁可多占 0.31 GiB ✓。
#: ⚠️ 这也正是 `decode` 里那句"潜变量在采样侧一套 dtype、VAE 可以是另一套"的由来 ✓
#: （它按 :func:`_model_dtype` 把潜变量对齐到 **VAE 权重** ✓）⇒ 混合精度本来就设计好了 ✓。
SDXL_VAE_DTYPE = "float32"

#: 认的精度名 ✓（别的一律**报错** ✗ —— 不静默当 fp32 蒙过去 ✓）。
SDXL_DTYPE_NAMES: tuple[str, ...] = ("float16", "bfloat16", "float32")

#: 事实表 ✓（供日志/自检取用 ✓；**数字只有一处来源** ✓）。
SDXL_IMAGE_FACTS: dict[str, Any] = {
    "keyPrefixes": {
        "unet": SDXL_UNET_KEY_PREFIX,
        "vae": SDXL_VAE_KEY_PREFIX,
        "clipL": SDXL_CLIP_L_KEY_PREFIX,
        "clipG": SDXL_CLIP_G_KEY_PREFIX,
    },
    "latentScale": SDXL_LATENT_SCALE,
    "sigmaTimesteps": SDXL_SIGMA_TIMESTEPS,
    #: 训练分辨率 ✓（出处见常量注释 ✓）—— 计划阶段按它抬底 ✓。
    "trainedResolution": SDXL_TRAINED_RESOLUTION,
    "trainedMegapixels": SDXL_TRAINED_MEGAPIXELS,
    #: 默认精度 ✓（CUDA 口径 ✓）—— 「不传 dtype ⇒ fp32 ⇒ 换页」的那条实测见 :data:`SDXL_COMPUTE_DTYPE` ✓。
    "dtypes": {"compute": SDXL_COMPUTE_DTYPE, "vae": SDXL_VAE_DTYPE},
    "betaLinear": (_SDXL_BETA_LINEAR_START, _SDXL_BETA_LINEAR_END),
    #: 官方权重的张量数 ✓（**只读头部实测** ✓）：四件套 1680 + 248 + 197 + 390 = **2515** ✓，
    #: 而文件**总键数也是 2515** ✓✗ ⇒ 四个前缀**恰好完整划分**了这份检查点 ✓
    #: （少一个前缀 / 前缀写错 ⇒ 键数对不上、且会有大量剩余键 ✓ 自检据此必红 ✓）。
    "componentKeys": {"unet": 1680, "vae": 248, "clipL": 197, "clipG": 390},
    "totalKeys": 2515,
}


class SdxlBackendError(RuntimeError):
    """SDXL 后端的装配/装载/前向出错（不静默兜底 ✗）。"""


# ══════════════════════════════════════════════════════════════════════════
# σ ⇄ t 口径（纯数学 ✓ —— 不依赖 torch ✓ ⇒ 没装 torch 也能自检 ✓）
# ══════════════════════════════════════════════════════════════════════════
def sdxl_sigmas_table(timesteps: int = SDXL_SIGMA_TIMESTEPS, *,
                      linear_start: float = _SDXL_BETA_LINEAR_START,
                      linear_end: float = _SDXL_BETA_LINEAR_END) -> tuple[float, ...]:
    """SDXL 的 **σ 表** ✓（``timesteps`` 个 ✓，**随下标单调增** ✓）。

    ⚠️ 这是**离散格**（下标就是 UNet 要的 ``t`` ✓），**不是**采样序列 ✗ —— 采样要按格取点 ✓，
    见 :func:`sdxl_sigmas_for_steps` ✓（两个名字只差一个词 ✓，别用混 ✗）。
    """
    steps = int(timesteps)
    if steps < 2:
        raise SdxlBackendError(f"σ 表至少要 2 点 ✗（拿到 {steps} ✓）")
    if not 0.0 < float(linear_start) < float(linear_end) < 1.0:
        raise SdxlBackendError(
            f"β 端点要满足 0 < start < end < 1 ✗（拿到 {linear_start} / {linear_end} ✓）"
            "—— ⚠️ 顺序反了表会**单调减** ✓，采样整条就废了 ✓")
    # ⚠️ LDM 的 "linear" 是**在 √β 上线性**、再平方 ✓ —— 直接在 β 上线性是**另一条**调度 ✗。
    low, high = math.sqrt(float(linear_start)), math.sqrt(float(linear_end))
    alphas_cumprod = 1.0
    table: list[float] = []
    for index in range(steps):
        beta = (low + (high - low) * index / (steps - 1)) ** 2
        alphas_cumprod *= (1.0 - beta)
        if alphas_cumprod <= 0.0:  # pragma: no cover - 只在端点离谱时发生
            raise SdxlBackendError(f"第 {index} 步的 a 累积到了非正数 ✗（β 端点有问题 ✓）")
        table.append(math.sqrt((1.0 - alphas_cumprod) / alphas_cumprod))
    return tuple(table)


def sigma_to_timestep(sigma: float, *, table: tuple[float, ...] | None = None) -> int:
    """连续 σ ⇒ 整数时间步 ✓（**对数域最近邻** ✓）。

    ⚠️ 这是 SDXL 的**硬口径** ✗：UNet 只认 ``t ∈ [0, 999]`` ✓。把 σ 直接当 t 喂进去**不会报错** ✓，
    出来的是「结构像图但是噪声」的东西 ✓✗ —— 属本仓最忌的「看不出来」那类错 ✓。
    ``σ = 0`` ⇒ ``0`` ✓（最后一个时间步 ✓，不取对数 ✓）。
    """
    values = table if table is not None else sdxl_sigmas_table()
    target = float(sigma)
    if not math.isfinite(target):
        raise SdxlBackendError(f"σ 不是有限数 ✗（拿到 {sigma!r} ✓）")
    if target <= 0.0:
        return 0
    probe = math.log(target)
    best_index, best_distance = 0, float("inf")
    for index, value in enumerate(values):
        distance = abs(math.log(value) - probe)
        if distance < best_distance:
            best_index, best_distance = index, distance
    return best_index


def sigma_at_timestep(step: float, *, table: tuple[float, ...] | None = None) -> float:
    """时间步 ⇒ σ ✓（**对数域线性插值** ✓ —— 与官方 `ModelSamplingDiscrete.sigma` 同一口径 ✓）。

    ⚠️ 是**插值** ✗ 不是查表 ✓：官方在 ``log_sigmas[floor(t)]`` 到 ``[ceil(t)]`` 之间按小数部分
    线性插值 ✓（只有整数 ``t`` 才正好落在表上 ✓）。⇒ 调度器取非整数 t 是合法的 ✓，
    而**喂给 UNet 的 t** 仍由 :func:`sigma_to_timestep` 反算成整数 ✓；两侧都按「log 空间最近邻」✓
    ⇒ 自洽 ✓。
    """
    values = table if table is not None else sdxl_sigmas_table()
    target = float(step)
    if not math.isfinite(target):
        raise SdxlBackendError(f"时间步不是有限数 ✗（拿到 {step!r} ✓）")
    clamped = min(max(target, 0.0), float(len(values) - 1))
    low = int(math.floor(clamped))
    high = int(math.ceil(clamped))
    weight = clamped - low
    log_sigma = (1.0 - weight) * math.log(values[low]) + weight * math.log(values[high])
    return math.exp(log_sigma)


def _ramp(start: float, stop: float, count: int) -> list[float]:
    """闭区间等分 ✓（``count == 1`` ⇒ ``[start]`` ✓ —— 与 ``torch.linspace`` 的边界一致 ✓✗）。"""
    if count <= 1:
        return [float(start)]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def sdxl_sigmas_for_steps(steps: int, *, schedule: str = "normal",
                          table: tuple[float, ...] | None = None) -> list[float]:
    """SDXL 的采样 σ 序列 ✓：``steps`` 个区间 ⇒ ``len == steps + 1`` ✓、**单调递减** ✓、
    末尾必然是 ``0.0`` ✓（与本仓 `schedules` 的约定一致 ✓）。

    ⚠️ 这一条**必须与视频那边分开** ✗✗：SDXL 是**离散**调度 ✓ —— ``t`` 只在 ``0..999`` 的**整数格**
    上 ✓ ⇒ σ 序列要**按格取** ✓（``normal`` 在格之间按 log 插值 ✓、``simple`` 直接取整格 ✓）。
    拿视频那边的 karras ρ-ramp 硬套 ⇒ σ 落不到格上 ✓ ⇒ :func:`sigma_to_timestep` 会把**相邻两步映到
    同一个 t** ✓ ⇒ 白跑一遍同样的噪声水平（**不报错、图还能出** ✓ 属"看不出来"那类 ✓✗）。
    """
    count = max(1, int(steps))
    values = table if table is not None else sdxl_sigmas_table()
    name = str(schedule or "normal").strip().lower()
    if name == "normal":
        # 官方 `normal_scheduler` 口径 ✓：t 在 [999, 0] 上等分 ✓（不是 σ 等分 ✗），再映射回 σ ✓。
        sigmas = [sigma_at_timestep(step, table=values)
                  for step in _ramp(float(len(values) - 1), 0.0, count)]
    elif name == "simple":
        # 官方 `simple_scheduler` 口径 ✓：按 ``1000/steps`` 的**步长取整格** ✓（从最噪那头往下 ✓）。
        span = len(values) / count
        sigmas = [values[-(1 + min(int(index * span), len(values) - 1))] for index in range(count)]
    else:
        raise SdxlBackendError(
            f"未知 SDXL 调度 {schedule!r} ✗；可用：{list(SDXL_SCHEDULES)} ✓"
            "（⚠️ **别**拿视频那边的 karras/linear 名字 ✓ —— 离散调度要按格取 ✓）")
    return [float(value) for value in sigmas] + [0.0]


def eps_input(noise: Any, sigma: float) -> Any:
    """EPS 预条件 ✓：``x_in = x / √(σ² + 1)`` ✓（σ 是 Python float ✓，广播到整个张量 ✓）。"""
    scale = math.sqrt(float(sigma) ** 2 + 1.0)
    if scale == 0.0:  # pragma: no cover - √(σ²+1) ≥ 1 恒成立
        raise SdxlBackendError("√(σ²+1) 为 0 ✗（不可能，除非 σ 是 NaN ✓）")
    return noise / scale


def denoised_from_eps(noise: Any, sigma: float, eps: Any) -> Any:
    """EPS 去噪 ✓：``x0 = x − ε·σ`` ✓（``ε`` 是 UNet 的**原始**输出 ✓，它没做预条件 ✓）。"""
    return noise - eps * float(sigma)


def snap_image_size(width: int, height: int, *,
                    scale: int = SDXL_LATENT_SCALE) -> tuple[int, int, bool]:
    """把宽高吸附到 ``scale`` 的**倍数** ✓（返回 ``(宽, 高, 是否改过)`` ✓）。

    ⚠️ **向下**吸附（``//`` ✓）而不是四舍五入 ✗ —— 宁可少几个像素 ✓，也不能超出手册给的范围 ✓
    （超了 ⇒ 显存/尺寸双输 ✓）。⚠️ 改过就**如实返回 True** ✗（调用方据此写日志 ✓，不静默改输入 ✓）。
    """
    step = int(scale)
    if step <= 0:
        raise SdxlBackendError(f"吸附倍数要为正 ✗（拿到 {scale} ✓）")
    want_w, want_h = int(width), int(height)
    if want_w <= 0 or want_h <= 0:
        raise SdxlBackendError(f"宽高要为正 ✗（拿到 {want_w}×{want_h} ✓）")
    fixed_w = max(step, want_w // step * step)
    fixed_h = max(step, want_h // step * step)
    return fixed_w, fixed_h, (fixed_w != want_w or fixed_h != want_h)


# ══════════════════════════════════════════════════════════════════════════
# 装配：一份检查点 ⇒ 四件套（⚠️ **只读一遍** ✗：6.9 GiB 读四遍是浪费 ✓）
# ══════════════════════════════════════════════════════════════════════════
def _split_by_prefix(state: dict[str, Any], prefix: str, *,
                     what: str) -> dict[str, Any]:
    """切出 ``prefix`` 那一半并**剥掉前缀** ✓；一个键都没有 ⇒ **报错** ✗（不静默给空字典 ✓）。"""
    size = len(prefix)
    scoped = {key[size:]: value for key, value in state.items() if key.startswith(prefix)}
    if not scoped:
        raise SdxlBackendError(
            f"检查点里没有以 {prefix!r} 开头的键 ⇒ 拿不到{what} ✗（这不是 SDXL 主权重 ✓）")
    return scoped


def _dtype_name(value: Any) -> str | None:
    """``None`` / torch dtype / 字符串 ⇒ **统一的小写名字** ✓（``torch.float16`` ⇒ ``"float16"`` ✓）。

    ⚠️ 认不出就**报错** ✗（不静默落回 fp32 ✓ —— 那正是「以为给了、其实没生效」✓✗）。
    """
    if value is None:
        return None
    if isinstance(value, str):
        name = value.strip().lower()
    else:
        name = str(getattr(value, "__name__", value)).strip().lower().rsplit(".", 1)[-1]
    if name not in SDXL_DTYPE_NAMES:
        raise SdxlBackendError(
            f"认不出的精度 {value!r} ✗ ⇒ 只认 {SDXL_DTYPE_NAMES} ✓（不静默当 fp32 蒙过去 ✗）")
    return name


def _is_cuda_device(device: Any) -> bool:
    """``device`` 是否**明确**指向 CUDA ✓（``None`` ⇒ 假 ✓：那意味着权重留在 CPU ✓，见 ``build_*`` ✓）。"""
    return str(device or "").strip().lower().startswith("cuda")


def resolve_sdxl_dtype_names(device: Any = None, *, dtype: Any = None,
                             vae_dtype: Any = None) -> tuple[str, str]:
    """算出四件套该用哪套精度 ✓ ⇒ ``(计算组件名, VAE 名)`` ✓。

    ⚠️ 刻意返回**字符串** ✗ 而不是 torch dtype ✓：这样这条口径**没装 torch 也判得了** ✓
    （同 :func:`sdxl_sigmas_table` 的取舍 ✓）；由 :func:`load_sdxl_components` 再转成 torch dtype ✓。

    口径（出处与实测数字见 :data:`SDXL_COMPUTE_DTYPE` / :data:`SDXL_VAE_DTYPE` ✓）：

    ① 调用方**显式**给了 ``dtype`` ⇒ 就用它 ✓（显式优先，绝不偷偷改写 ✓）；
    ② 没给 + ``device`` 明确是 CUDA ⇒ **fp16** ✓（检查点本身就是 fp16 ✓；不给就变 fp32 ⇒ 白占一倍 ✓✗）；
    ③ 没给 + 其它（含 ``device=None`` ⇒ 权重留 CPU ✓）⇒ **fp32** ✓（CPU 上 fp16 又慢又不省 ✓）；
    ④ VAE **恒 fp32** ✗✗（除调用方显式 ``vae_dtype=`` ✓）：压 fp16 会**静默出黑图** ✓✗。
    """
    compute = _dtype_name(dtype)
    if compute is None:
        compute = SDXL_COMPUTE_DTYPE if _is_cuda_device(device) else "float32"
    vae = _dtype_name(vae_dtype) or SDXL_VAE_DTYPE
    return compute, vae


def _torch_dtype(name: str) -> Any:
    """精度名 ⇒ torch dtype ✓（只在**真要建模型**时才需要 torch ✓，见 :func:`resolve_sdxl_dtype_names` ✓）。"""
    return getattr(torch, name)


def load_sdxl_components(
    path: str | Path, *, device: Any = None, dtype: Any = None, vae_dtype: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """把官方检查点赞成四件套 ✓，返回 ``(部件字典, 报告)`` ✓。

    ⚠️ 四件**全部严格装载** ✓：任一件缺键 / 形状不符 / 多出认不得的键 ⇒ **报错** ✗
    （不装半个模型 ✗ —— 半装出来的图会「有形状、有颜色、但语义全错」✓✗，本仓最忌这类 ✓）。

    ⚠️ 精度**按组件分开** ✗✗：``dtype`` 只管 UNet + 两座文本塔 ✓；VAE 另有 ``vae_dtype``、
    默认**恒 fp32** ✓ —— 口径与实测出处见 :func:`resolve_sdxl_dtype_names` ✓。
    报告里如实回报实际用的两套精度 ✓（日志/`engine-status` 要能看出"这轮到底跑什么精度"✓）。
    """
    from . import clip_text as clip_mod  # noqa: PLC0415
    from . import sdxl as unet_mod  # noqa: PLC0415
    from . import sdxl_vae as vae_mod  # noqa: PLC0415

    target = Path(path)
    if not target.exists():
        raise SdxlBackendError(f"检查点不存在 ✗：{target}（先按 `loader.plan_stage` 的报告去装 ✓）")
    try:
        from safetensors.torch import load_file  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover - 依赖缺失时给出可行动信息 ✓
        raise SdxlBackendError(
            f"官方 safetensors 未安装（{err} ✓）⇒ 先装它 ✓；"
            f"注意本仓自带的纯 Python 读取器**只**能体检、不能装载 ✗") from err
    try:
        state = dict(load_file(str(target), device=str(device or "cpu")))
    except Exception as err:  # noqa: BLE001 - 加载期异常类型不稳定，统一转成中文错 ✓
        raise SdxlBackendError(f"读取权重失败 ✗：{type(err).__name__}: {err}") from err

    compute_name, vae_name = resolve_sdxl_dtype_names(device, dtype=dtype, vae_dtype=vae_dtype)
    compute_dtype = _torch_dtype(compute_name)
    report: dict[str, Any] = {"path": str(target), "sourceKeys": len(state), "components": {}}

    unet = unet_mod.build_sdxl_unet(device=device, dtype=compute_dtype)
    vae = vae_mod.build_sdxl_vae(device=device, dtype=_torch_dtype(vae_name))
    clip_l = clip_mod.build_clip_text_model(clip_mod.CLIP_L, device=device, dtype=compute_dtype)
    clip_g = clip_mod.build_clip_text_model(clip_mod.CLIP_G, device=device, dtype=compute_dtype)

    for name, model, loader, prefix in (
        ("unet", unet, unet_mod.load_sdxl_unet_state_dict, SDXL_UNET_KEY_PREFIX),
        ("vae", vae, vae_mod.load_sdxl_vae_state_dict, SDXL_VAE_KEY_PREFIX),
        ("clipL", clip_l, None, SDXL_CLIP_L_KEY_PREFIX),
        ("clipG", clip_g, None, SDXL_CLIP_G_KEY_PREFIX),
    ):
        scoped = _split_by_prefix(state, prefix, what=name)   # 计数 + 缺前缀时**当场报错** ✓
        # ⚠️ **别在这里剥前缀再交出去** ✗✗：四个装载函数**都收整份** ✓（各自按前缀切 ✓）——
        #    提前剥一层 ⇒ UNet/VAE 会报「这不是 SDXL 检查点」✓✗（实测撞到过 ✓）；
        #    CLIP 侧更绕 ✗：它的**前缀探测**只认 `transformer.*`/`text_model.*`/`text_projection*`
        #    这几种「已剥」形态 ✓，而 **G 塔剥完是 OpenCLIP 原生的 `model.*`** ✓（真权重实测 ✓）
        #    ⇒ 不显式给 `prefix=` 就会被判成「认不出」✗。⇒ **两侧一律显式传 `prefix=`** ✓。
        if loader is None:
            leftover = clip_mod.load_clip_text_state_dict(model, state, prefix=prefix)
            declined = len(leftover)
        else:
            leftover = loader(model, state)      # ← 传**整份** ✓，它自己按前缀切 ✓
            declined = None                      # 它回的「外人键」= 另外三件套 ✓ 不是异常 ✓
        report["components"][name] = {"prefix": prefix, "keys": len(scoped),
                                     "declined": declined}
    #: 实际用的两套精度 ✓（**如实** ✓ —— 别让日志里出现"以为给了 fp16"✓✗）。
    report["dtypes"] = {"compute": compute_name, "vae": vae_name}
    return {"unet": unet, "vae": vae, "clipL": clip_l, "clipG": clip_g}, report


@dataclass(frozen=True)
class _Condition:
    """一路文本条件 ✓（``context`` 给交叉注意力 ✓、``pooled`` 给 ``adm`` ✓）。"""

    context: Any
    pooled: Any
    tokens: int
    truncated: bool


class SdxlBackend:
    """SDXL 出图后端 ✓（`GenerationBackend` 的一个实现 ✓ —— 引擎里**唯一**碰 SDXL 张量的地方 ✓）。

    ⚠️ ``name`` 是 ``"sdxl"`` ✗ 而不是 ``"torch"`` ✓ —— 出图链路的**分流依据**就是它 ✓
    （`runtime` 按它选后端 ✓、`image_generation` 按它写 ``engine-backend`` ✓）。
    """

    name = "sdxl"

    def __init__(self, *, device: Any = None, dtype: Any = None,
                 vae_dtype: Any = None, tokenizer: Any = None) -> None:
        """``dtype`` 只管计算组件 ✓、``vae_dtype`` 管 VAE ✓（默认各自见 :func:`resolve_sdxl_dtype_names` ✓）。

        ⚠️ 两个都留 ``None`` 也**不是**"随便" ✓：``device="cuda"`` ⇒ 计算组件 **fp16**、VAE **fp32** ✓
        （出处与实测见 :data:`SDXL_COMPUTE_DTYPE` ✓——生产装配 ``runtime`` 正是这么调的 ✓）。
        """
        self.device = device
        self.dtype = dtype
        self.vae_dtype = vae_dtype
        self._tokenizer = tokenizer
        self._unet: Any = None
        self._vae: Any = None
        self._clip_l: Any = None
        self._clip_g: Any = None
        self.load_report: dict[str, Any] = {}

    # ── 状态（⚠️ **如实** ✓：没装齐就是 synthetic ✓，不许"看起来像真图"✗）──────────
    @property
    def loaded(self) -> bool:
        """四件套是否**全都装上了** ✓。"""
        return all(part is not None
                   for part in (self._unet, self._vae, self._clip_l, self._clip_g))

    @property
    def synthetic(self) -> bool:
        """⚠️ 只有四件**真权重**都装上才为 ``False`` ✓ —— 否则如实标**合成** ✗。

        这是与 `TorchBackend`（H3 参考组件未训练 ⇒ 恒 ``True`` ✓）**最关键的区别** ✓。
        """
        return not self.loaded

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise SdxlBackendError(
                "没给 tokenizer ✗ ⇒ 编不了文本条件 ✓（词表是**权重的一部分** ✓，引擎不内置 ✗；"
                "请用 `tokenizer_bpe.load_tokenizer(clip 词表目录)` ✓ 再传进来 ✓）")
        return self._tokenizer

    # ── 装配 ─────────────────────────────────────────────────────────────
    def load_weights(self, path: str | Path, *, tokenizer: Any = None) -> dict[str, Any]:
        """装四件套 ✓ 并返回报告 ✓。⚠️ 任一件不齐 ⇒ **报错** ✗（不会留下半个模型 ✓）。"""
        # ⚠️ **CPU 线程预算**（用户口径 ✓ 2026-09-26）：重张量运算走 GPU ✓，CPU 只做加载/IO/搬张量 ✓
        #    ⇒ 钳住 torch 线程池 ✗（默认并行度 = **逻辑核数** ✓✗ ⇒ 四件套的 `to(device, dtype)`
        #    反量化 + 文本编码前后处理就能把整机拉满 ✓✗）。口径只有一处 ✓：`app/core/cpu_budget.py` ✓。
        self._threadBudget = cpu_budget.apply_torch(torch)
        parts, report = load_sdxl_components(path, device=self.device, dtype=self.dtype,
                                            vae_dtype=self.vae_dtype)
        self._unet = parts["unet"]
        self._vae = parts["vae"]
        self._clip_l = parts["clipL"]
        self._clip_g = parts["clipG"]
        if tokenizer is not None:
            self._tokenizer = tokenizer
        for model in (self._unet, self._vae, self._clip_l, self._clip_g):
            model.eval()
        report["synthetic"] = self.synthetic
        report["device"] = str(self.device or "cpu")
        self.load_report = report
        return report

    # ── 协议 ①：文本条件 ─────────────────────────────────────────────────
    def _encode_one(self, text: str) -> _Condition:
        """一句话 ⇒ 一路条件 ✓（双塔各跑 ``layer_idx=-2`` 且**不做**末层归一化 ✓ —— SDXL 口径 ✓）。"""
        from . import clip_text as clip_mod  # noqa: PLC0415

        tokenizer = self.tokenizer
        ids_l, cut_l = clip_mod.sdxl_token_ids(tokenizer, text, clip_mod.CLIP_L)
        ids_g, cut_g = clip_mod.sdxl_token_ids(tokenizer, text, clip_mod.CLIP_G)
        with torch.no_grad():
            out_l = self._clip_l(_as_ids(ids_l, _model_device(self._clip_l)),
                                 layer_idx=-2, layer_norm_hidden_state=False)
            out_g = self._clip_g(_as_ids(ids_g, _model_device(self._clip_g)),
                                 layer_idx=-2, layer_norm_hidden_state=False)
        context, pooled = clip_mod.encode_sdxl_conditioning(out_l, out_g)
        return _Condition(context=context, pooled=pooled,
                          tokens=int(len(ids_g)), truncated=bool(cut_l or cut_g))

    def encode_text(self, request: Any) -> dict[str, Any]:
        """出 ``{"positive": 条件, "negative": 条件}`` ✓（**两路都算** ✓ —— CFG 要用 ✓）。"""
        if not self.loaded:
            raise SdxlBackendError("四件套没装齐 ✗ ⇒ 编不了文本 ✓（先 `load_weights` ✓）")
        return {
            "positive": self._encode_one(request.prompt),
            "negative": self._encode_one(request.negative or ""),
        }

    # ── 协议 ②：初始潜变量 ───────────────────────────────────────────────
    def init_latents(self, plan: Any, request: Any) -> Any:
        """标准正态噪声 **× σ_max** ✓（``x = n·σ`` ✓ —— 与 `EPS` 的加噪口径一致 ✓）。"""
        if not self.loaded:
            raise SdxlBackendError("四件套没装齐 ✗ ⇒ 起不了潜变量 ✓")
        sigmas = list(plan.sigmas or [])
        if not sigmas:
            raise SdxlBackendError("调度里一个 σ 都没有 ✗（steps 是不是 0 ✓？）")
        shape = (1, int(self._vae.config.latent_channels),
                 int(plan.height) // SDXL_LATENT_SCALE, int(plan.width) // SDXL_LATENT_SCALE)
        generator = torch.Generator(device="cpu").manual_seed(int(request.seed))
        noise = torch.randn(shape, generator=generator, dtype=torch.float32)
        # ⚠️ 落位**照权重问** ✗（见 :func:`_model_device` ✓）—— 信 `self.device` 会在它为 ``None`` 时
        #    把潜变量丢在 CPU 上 ✓，而下一步 UNet 在 CUDA ⇒ 又是一次「换设备才炸」✓✗。
        return (noise * float(sigmas[0])).to(device=_model_device(self._unet),
                                             dtype=_model_dtype(self._unet))

    # ── 协议 ③：单步去噪（**返回 x0** ✓ —— 采样器要的就是这个 ✓）─────────────
    def denoise(self, latents: Any, sigma: float, condition: Any, request: Any) -> Any:
        """一步 ✓ ⇒ 回 **x₀** ✓。

        ⚠️ 两个「喂错也不报错」的地方都在这 ✗✗：
        ① **σ 要先映射成整数 t** ✓（SDXL 的 UNet 只认 0..999 ✓）；
        ② UNet 吃的是**预条件过**的输入 ``x/√(σ²+1)`` ✓，但 ``x₀`` 要**用原始 x** 算 ✓ ——
        把预条件后的值当代入 ⇒ 结果**系统性偏亮** ✓✗（不报错、图还在 ✓ 属"看不出来"那类 ✓）。

        ⚠️ ``request`` 在这里**用不上** ✓（尺寸从潜变量形状推 ✓，见 :func:`_time_ids` ✓）——
        它留着是因为 :class:`~app.services.engine.pipeline.GenerationBackend` 的采样协议就是这么传的 ✓
        （引擎那边 ``model_fn`` 只递得出 ``request`` ✗ ⇒ 别删这个形参 ✓）。
        """
        from . import sdxl as unet_mod  # noqa: PLC0415

        if not isinstance(condition, _Condition):
            raise SdxlBackendError(
                f"条件不是本后端给的 `_Condition` ✗（拿到 {type(condition).__name__} ✓）"
                "—— 多半是把整个 `encode_text` 的字典直接传进来了 ✓（要传**单路** ✓）")
        step = sigma_to_timestep(float(sigma))
        batch = latents.shape[0]
        timesteps = torch.full((batch,), float(step), device=latents.device, dtype=latents.dtype)
        adm = unet_mod.build_adm(condition.pooled, _time_ids(latents))
        # ⚠️ **这里必须关梯度** ✗✗（以下是**实测事实** ✓，不是估计 ✓）：UNet 权重默认
        #    ``requires_grad=True`` ✓ ⇒ 不关就会**每步建一张计算图** ✓，而采样循环逐步累积 ✓ ⇒
        #    1024×1024 / 20 步跑到**第 6 步**时 PyTorch 已分配 **94.92 GiB**（``free: 0`` ✓）⇒ 直接
        #    OOM ✓✗（1024 恰是 SDXL 唯一正确的分辨率 ✓ ⇒ 这条等于把唯一能出好图的档位堵死 ✓）；
        #    512×512 虽能侥幸跑完 ✓，但 20 步要 **50.4 s** ✓（正常 4~6 s ✓）。
        #    ``x₀`` 在 no_grad 里算出 ⇒ ``requires_grad=False`` ✓ ⇒ 采样器 ``x + d·Δσ`` 也不带图 ✓
        #    ⇒ **循环不再累积** ✓（堵这一处就够 ✓ 采样器不必改 ✓）。
        with torch.no_grad():
            eps = self._unet(eps_input(latents, float(sigma)), timesteps, condition.context, adm)
            x0 = denoised_from_eps(latents, float(sigma), eps)
        # ⚠️ **非有限值当场报** ✗✗（不顺着往下采样出"看着像图的垃圾" ✓✗）：
        #    fp16 权重（默认口径 ✓ 见 :data:`SDXL_COMPUTE_DTYPE` ✓）**存在溢出成 inf/NaN 的可能** ✓，
        #    而它不会让任何一行报错 ✓ —— 会一路被采样器放大成**黑图/糊图** ✓✗（本仓最忌的
        #    「不报错但结果坏」✓）。代价是每步一次小同步 ✓（(1,4,96,172) 上 ≈0.1 ms ✓，
        #    相对单步 ~1 s 可忽略 ✓），换来的是"坏在明处" ✓。
        if not bool(torch.isfinite(x0).all()):
            raise SdxlBackendError(
                f"σ={float(sigma):.4f} 这一步的 x₀ 出了非有限值（inf/NaN）✗ ⇒ 停在这里报 ✓\n"
                f"  · 多半是 fp16 溢出 ✓：改用 fp32 试（`SdxlBackend(device=…, dtype=\"float32\")` ✓）\n"
                f"  · ⚠️ 别把它当成「图糊」放过 ✗ —— 继续采下去只会得到一张**不报错的废图** ✓✗")
        return x0

    # ── 协议 ④：潜变量 ⇒ 像素 ────────────────────────────────────────────
    def decode(self, latents: Any, plan: Any, request: Any) -> dict[str, Any]:
        """⚠️ **必须走 `sdxl_vae.decode_latents`** ✗（它替我们除了缩放系数 0.13025 ✓）——
        直接调 `model.decode` ⇒ 图**发灰/过曝**且**不报错** ✓✗（见 `sdxl_vae` 模块头 ✓）。
        产物是 ``(B, 3, 1, H, W)`` ✓（补一个 T 轴 ✓ —— `media.write_image` 要 5 维 ✓）。
        """
        from . import sdxl_vae as vae_mod  # noqa: PLC0415

        if not self.loaded:
            raise SdxlBackendError("四件套没装齐 ✗ ⇒ 解不了码 ✓")
        # ⚠️ 潜变量在**采样侧**那一套 dtype/设备上 ✓（常见 fp16 + cuda ✓），而 VAE 允许是**另一套** ✓
        #    ⇒ 这里必须**对齐到 VAE 权重** ✗（见 :func:`_model_dtype` ✓）：不对齐 ⇒ 轻则 conv2d 报
        #    dtype/device 不符 ✓，重则**静默出黑图** ✓✗（VAE 被压成 fp16 的老毛病 ✓）。
        # ⚠️ 解码同样是**纯推理** ✓ ⇒ 一并关梯度 ✗：VAE 在 fp32 下解 1024×1024（128×128×4 潜变量 ✓）
        #    带计算图会白吃一大块显存 ✓，而且 ``.cpu()`` 还会把图一起搬过去 ✓✗（见 `denoise` 的实测 ✓）。
        with torch.no_grad():
            pixels = vae_mod.decode_latents(
                self._vae,
                latents.to(device=_model_device(self._vae), dtype=_model_dtype(self._vae)),
                config=self._vae.config)
        return {"frames": pixels.unsqueeze(2).to(torch.float32).cpu()}

    # ── 协议 ⑤：落盘 ─────────────────────────────────────────────────────
    def write(self, outputs: dict[str, Any], plan: Any, request: Any) -> dict[str, Any]:
        """写一张 PNG ✓（**只有一帧** ✓ —— 出图链路不是"视频取首帧" ✗）。"""
        from . import media  # noqa: PLC0415

        frames = outputs.get("frames")
        if frames is None:
            raise SdxlBackendError("产物里没有 `frames` ✗（decode 是不是没跑 ✓？）")
        target = self.output_path(plan, request)
        target.parent.mkdir(parents=True, exist_ok=True)
        written = media.write_image(frames, str(target), index=0, value_range="0..1")
        return {
            "backend": self.name,
            "synthetic": self.synthetic,
            "primaryPath": str(target),
            "images": [str(target)],
            "width": int(plan.width),
            "height": int(plan.height),
            "frames": 1,
            "bytes": int(written.get("bytes") or 0),
        }

    def output_path(self, plan: Any, request: Any) -> Path:
        """输出路径 ✓（**确定性** ✓：同 seed + 同尺寸 ⇒ 同文件名 ✓，便于复现与自检 ✓）。"""
        base = Path(request.outputs_dir) if getattr(request, "outputs_dir", None) else \
            Path.cwd() / "outputs"
        return base / f"sdxl-{int(request.seed)}-{int(plan.width)}x{int(plan.height)}.png"

    def describe(self) -> dict[str, Any]:
        """如实描述当前状态 ✓（日志/`engine-status` 用 ✓，**绝不**把没装的写成装了 ✗）。

        ⚠️ 带 ``dtypes`` ✓：装完后是**实测**用的两套精度 ✓；没装时给**将要**用的口径 ✓
        （取值同 :func:`resolve_sdxl_dtype_names` ✓）—— 精度是显存与"静默黑图"的开关 ✓，
        日志里必须能一眼看出来 ✓✗。
        """
        return {
            "backend": self.name,
            "synthetic": self.synthetic,
            "loaded": self.loaded,
            "device": str(self.device or "cpu"),
            "dtypes": dict(self.load_report.get("dtypes") or {}) or self._planned_dtypes(),
            "facts": SDXL_IMAGE_FACTS,
            "load": self.load_report,
        }

    def _planned_dtypes(self) -> dict[str, str]:
        """还没装权重时，如实预告这轮**将要**用的精度 ✓（口径只有一处来源 ✓）。"""
        compute, vae = resolve_sdxl_dtype_names(self.device, dtype=self.dtype,
                                                vae_dtype=self.vae_dtype)
        return {"compute": compute, "vae": vae}


def _time_ids(latents: Any) -> Any:
    """SDXL 的 ``time_ids`` ✓：``[高, 宽, 裁上, 裁左, 目标高, 目标宽]`` ✓（**顺序别换** ✗）。

    ⚠️ 是 ``(B, 6)`` 的**张量** ✗ 不是元组 ✓（``build_adm`` 会查形状 ✓）；
    顺序错了**不报错** ✓ —— 只是把高宽当成裁切量 ✓ ⇒ 图上表现为「构图莫名偏/裁切」✓✗。

    ⚠️ 尺寸**从 latents 形状推**（``× SDXL_LATENT_SCALE`` ✓）—— **不是**从 ``request`` 取 ✗：
    ① 采样协议给 `denoise` 的只有 ``request``，而 ``GenerationRequest`` 上**没有** ``width``/``height``
       （尺寸是 :class:`GenerationPlan` 的事 ✓ —— 实测取 ``request.width`` 直接 ``AttributeError`` ✓✗）；
    ② latents 正是 ``init_latents`` 按 ``plan.height/width`` 建的 ✓ ⇒ 反过来乘回去就是**真正喂进网络**
       的那个尺寸 ✓（plan 的尺寸已被吸附到 8 的整数倍 ✓，两者必然一致 ✓）。
    裁切量取 0 ✓、目标尺寸取原尺寸 ✓ —— 与官方默认（无裁切、不放大）一致 ✓。
    """
    height = int(latents.shape[-2]) * SDXL_LATENT_SCALE
    width = int(latents.shape[-1]) * SDXL_LATENT_SCALE
    batch = max(1, int(latents.shape[0]))
    return torch.tensor([[height, width, 0, 0, height, width]] * batch,
                        device=latents.device, dtype=latents.dtype)


def _model_device(model: Any) -> Any:
    """取模型权重**实际所在**的设备 ✓。

    ⚠️ **从权重问** ✗，而不是信 ``self.device`` ✓ —— 后者允许是 ``None``（那时模型按框架默认落位 ✓），
    拿它去建张量会落错设备 ✓。权重一定在某个真实设备上 ✓，这是唯一可信的来源 ✓。
    """
    parameter = next(iter(model.parameters()), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _model_dtype(model: Any) -> Any:
    """取模型权重**实际所在**的 dtype ✓（同 :func:`_model_device` ✓：唯一可信的来源是权重本身 ✓）。"""
    parameter = next(iter(model.parameters()), None)
    return parameter.dtype if parameter is not None else torch.float32


def _as_ids(ids: Any, device: Any) -> Any:
    """token id 列表 ⇒ ``(1, N)`` 的长整型张量 ✓（``forward`` 要这个形状 ✓），**建在 ``device`` 上** ✗✗。

    ⚠️ ⚠️ **这一条漏了只在换设备时才炸** ✗✗：token 嵌入是**查找表** ✓ ⇒ 索引张量与权重**必须同设备** ✓。
    默认 ``torch.tensor`` 造在 **CPU** 上 ✓ ⇒ 在 CPU 上跑时正好两边都在 CPU、**一声不响地正常出图** ✓，
    一搬到 CUDA 就 ``Expected all tensors to be on the same device`` ✗（实测撞到过 ✓）。
    结论：**不能靠默认值蒙** ✗，设备要显式给 ✓。
    """
    return torch.tensor([list(ids)], dtype=torch.long, device=device)
