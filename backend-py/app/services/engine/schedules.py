"""噪声调度（sigma 序列）—— **自己实现**（零依赖，纯 Python ✓）。

来源是**公开论文里的公式**，不是任何项目的源码 ✗：

* ``karras``      —— Karras et al. 2022《Elucidating the Design Space of Diffusion-Based
  Generative Models》的 ρ-ramp 采样（ρ=7 ✓）；
* ``exponential`` —— 在对数域线性插值（σ 从 σ_max 指数衰减到 σ_min ✓）；
* ``linear_log_snr`` —— DPM-Solver++ 一类的「log-SNR 均匀」思路 ✓；
* ``normal``      —— 几何（对数）插值 ✓；
* ``ddim_uniform``—— DDIM 的均匀 t 采样再映射回 σ ✓。

⚠️ 与任何具体实现的**数值**都**不承诺逐位一致** ✗ —— 这是我们自己的引擎 ✓，
名字只描述**数学**；要复现别人某个 scheduler 的确切行为，请把它当**参数**（``sigma_min`` /
``sigma_max`` / 步数 / ρ ✓）而不是当暗号 ✗。

约定（与扩散模型的通用做法一致 ✓）：

* 返回的列表**末尾必然有一个 0.0**（最后一步去噪到 x0 ✓，即 ``len(sigmas) == steps + 1`` ✓）；
* σ 序列**单调递减**（从大到小 ✓）；
* σ 是**噪声标准差**（不是 t、不是 index ✓）。
"""
from __future__ import annotations

import math
from typing import Any

__all__ = ["SCHEDULES", "interpolate_sigmas", "sigmas_for", "time_shift_sigma", "timesteps_for"]

#: 常用锚点（SD 系模型的经验值；**可覆盖** ✓）
DEFAULT_SIGMA_MIN = 0.0292
DEFAULT_SIGMA_MAX = 14.6146


def _linspace(start: float, stop: float, count: int) -> list[float]:
    """闭区间等分（``count>=1`` ✓；纯 Python，避免依赖 numpy ✓）。"""
    if count <= 1:
        return [float(stop)]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    """把 σ 从一条 flow-shift 网格**换算到**另一条 ✓（闭式 ✓ 纯数学 ✓ 零依赖 ✓）。

    推导（自己做的 ✓，不是抄来的 ✗）：shift 网格与 base 网格的关系是
    ``σ = s·b / (1 + (s−1)·b)`` ✓ ⇒ 代进去解 ``b``：

    ``σ + σ(s−1)b = s·b`` ⇒ ``b = σ / (s + σ(1−s))`` ✓

    再把它套到目标 shift 上 ✓，即得下面的两行 ✓。

    ⚠️ **不换算就会犯的错**（多流模型都踩 ✓）：视频与音频跑在**各自的 shift** 上
    （本项目自研引擎按 H3 的取值 = 视频 **12.0** / 音频 **3.0** ✓ 见 `dit.H3_SHAPE_FACTS` ✓），
    但采样器只喂**视频的 σ** ✓ ⇒ 音频若直接吃同一个数 ✓，等于把「**同一个 σ**」当成
    「**两种不同的噪声水平**」✗ ⇒ 音频会**系统性偏噪或偏净** ✗（而且不会报错 ✓✗）。

    不变量（自检钉住 ✓）：
    ``time_shift_sigma(s, k, k) == s`` ✓（同一 shift ⇒ 恒等 ✓）；
    ``σ=0 ⇒ 0`` ✓、``σ=1 ⇒ 1`` ✓（**任意 shift 的不动点** ✓）；
    ``to_shift < from_shift`` 时把 ``0<σ<1`` **单调抬向 1** ✓；往返（12→3→12）回到原值 ✓。
    """
    if from_shift <= 0 or to_shift <= 0:
        raise ValueError(f"shift 必须为正（收到 {from_shift} / {to_shift} ✗）")
    denominator = from_shift + sigma * (1.0 - from_shift)
    if denominator == 0:
        raise ValueError(f"σ={sigma} 落在 shift={from_shift} 的极点上 ✗（分母为 0）")
    base = sigma / denominator
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def karras_sigmas(steps: int, *, sigma_min: float = DEFAULT_SIGMA_MIN,
                  sigma_max: float = DEFAULT_SIGMA_MAX, rho: float = 7.0) -> list[float]:
    """Karras ρ-ramp：``σ_i = (σ_max^(1/ρ) + i/(n-1)·(σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ`` ✓。

    ρ 越小越偏向「多花步数在高噪声段」（细节/结构）；ρ 越大越均匀 ✓。
    """
    steps = max(1, int(steps))
    inv_min = sigma_min ** (1.0 / rho)
    inv_max = sigma_max ** (1.0 / rho)
    ramp = _linspace(0.0, 1.0, steps)
    sigmas = [(inv_max + t * (inv_min - inv_max)) ** rho for t in ramp]
    return sigmas + [0.0]


def exponential_sigmas(steps: int, *, sigma_min: float = DEFAULT_SIGMA_MIN,
                       sigma_max: float = DEFAULT_SIGMA_MAX) -> list[float]:
    """对数域线性插值：``σ_i = σ_max^(1−i/(n−1)) · σ_min^(i/(n−1))`` ✓。"""
    steps = max(1, int(steps))
    sigmas = [sigma_max ** (1.0 - i / (steps - 1 if steps > 1 else 1))
              * sigma_min ** (i / (steps - 1 if steps > 1 else 1))
              for i in range(steps)]
    return sigmas + [0.0]


def normal_sigmas(steps: int, *, sigma_min: float = DEFAULT_SIGMA_MIN,
                  sigma_max: float = DEFAULT_SIGMA_MAX) -> list[float]:
    """几何插值（对 lnσ 等分 ✓）。"""
    steps = max(1, int(steps))
    log_min, log_max = math.log(sigma_min), math.log(sigma_max)
    return [math.exp(value) for value in _linspace(log_max, log_min, steps)] + [0.0]


def ddim_uniform_sigmas(steps: int, *, sigma_min: float = DEFAULT_SIGMA_MIN,
                        sigma_max: float = DEFAULT_SIGMA_MAX) -> list[float]:
    """DDIM 风格：t 在 ``[0,1]`` 均匀取点，再线性映射到 σ ✓（步数少时更「粗」但稳定 ✓）。"""
    steps = max(1, int(steps))
    return [sigma_max + (sigma_min - sigma_max) * t for t in _linspace(0.0, 1.0, steps)] + [0.0]


#: 可用调度名 → 生成函数 ✓（新增调度请**同时**加自检 ✓）
#:
#: ⚠️ 2026-09-17 自检抓到并**删掉**了一个 ``linear_log_snr``：本仓扩散 σ 的常见范围到 **14.6（>1）**✗，
#: 而 log-SNR 的 ``α = √(1−σ²)`` 在 σ>1 时无定义 ⇒ 那个调度**在标准 σ 段上数学不成立** ✗。
#: 规则：**只留能被自检验证的数学** ✓ —— 没验过的调度不放进列表（宁缺毋滥 ✓，不搞"看起来像"的模仿 ✗）。
SCHEDULES: dict[str, Any] = {
    "karras": karras_sigmas,
    "exponential": exponential_sigmas,
    "normal": normal_sigmas,
    "ddim_uniform": ddim_uniform_sigmas,
}


def sigmas_for(steps: int, kind: str = "karras", **options: Any) -> list[float]:
    """按名字取 sigma 序列（``steps`` 是**去噪步数** ⇒ 返回 ``steps+1`` 个 ✓）。"""
    fn = SCHEDULES.get(str(kind or "karras").lower())
    if fn is None:
        raise ValueError(f"未知调度 {kind!r}；可用：{sorted(SCHEDULES)}")
    sigmas = fn(steps, **options)
    return [float(value) for value in sigmas]


def timesteps_for(sigmas: list[float], *, sigma_data: float = 1.0) -> list[float]:
    """σ → t（``t = σ / (σ + sigma_data)`` 的单调映射 ✓；只用于日志/断点，不影响采样 ✓）。"""
    return [value / (value + sigma_data) if (value + sigma_data) else 0.0 for value in sigmas]


def interpolate_sigmas(sigmas: list[float], factor: float) -> list[float]:
    """把 sigma 序列**等比例**拉伸/压缩（``factor=2`` ⇒ 步数翻倍 ✓；线性插值 ✓）。

    用途：模型自带一个基准步数（例如 10 步 ✓），我们要跑 20 步时按几何插值细分 ✓
    （噪声段本就接近对数分布 ⇒ 对 σ 取对数再插值 ✓）。
    """
    factor = float(factor or 1.0)
    if factor <= 1.0 or len(sigmas) < 2:
        return [float(value) for value in sigmas]
    # ⚠️ 语义是**步数**×factor（``len(sigmas) = 步数 + 1`` ✓）：
    #    初版拿 ``len`` 去乘 ⇒ 多出一步 ✗；改成步数后又手滑 ``+1`` ⇒ 仍多一步 ✗
    #    （自检 ⑥ 两次都抓到：期望 11 → 21 ✓）。`target` 是**零点之前的点数** = 步数 ✓。
    steps = len(sigmas) - 1
    target = max(2, int(round(steps * factor)))
    out: list[float] = []
    last = len(sigmas) - 2  #: 末尾的 0 不参与插值 ✓
    for index in range(target):
        position = index * last / (target - 1)
        left, right = int(math.floor(position)), min(last, int(math.ceil(position)))
        if left == right or sigmas[left] <= 0 or sigmas[right] <= 0:
            out.append(float(sigmas[left]))
            continue
        #: 对数域插值（σ 跨度是数量级的 ✓ 线性插值会把中间段压扁 ✗）
        weight = position - left
        value = math.exp(math.log(sigmas[left]) * (1 - weight) + math.log(sigmas[right]) * weight)
        out.append(float(value))
    return out + [0.0]
