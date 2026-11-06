"""噪声调度（sigma 序列）—— **自己实现**（零依赖，纯 Python ✓）。

来源是**公开论文里的公式**，不是任何项目的源码 ✗：

* ``karras``      —— Karras et al. 2022《Elucidating the Design Space of Diffusion-Based
  Generative Models》的 ρ-ramp 采样（ρ=7 ✓）；
* ``exponential`` —— 在对数域线性插值（σ 从 σ_max 指数衰减到 σ_min ✓）；
* ``linear_log_snr`` —— DPM-Solver++ 一类的「log-SNR 均匀」思路 ✓；
* ``normal``      —— **timestep 等距**（离散/EDM 域里 timestep↔σ 是 log 关系 ⇒ 落到 σ 上就是几何插值 ✓；
  flow 域里 timestep **就是** σ ⇒ 落到 σ 上就是**σ 等距** ✓ —— 见 :func:`normal_sigmas` ✓）；
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

__all__ = ["FLOW_LADDER_STEPS", "FLOW_SHIFT_DEFAULT", "SCHEDULES", "beta_ppf", "beta_sigmas",
           "flow_shift_for_sequence", "flow_sigma_bounds", "flow_sigmas_for_steps",
           "flow_sigmas_for_video", "interpolate_sigmas", "sigmas_for",
           "simple_sigmas", "time_shift_sigma", "timesteps_for"]

#: flow 族自带的 σ 细梯子**格数** ✓（``ModelSamplingDiscreteFlow.set_parameters(timesteps=1000)`` ✓
#: ⇒ ``sigmas = [σ(i/1000) for i in 1..1000]`` ✓）。:func:`flow_sigma_bounds` 与
#: :func:`simple_sigmas` 用的是**同一条**梯子 ✓ —— 别各写一个 1000 ✗。
FLOW_LADDER_STEPS = 1000

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


def flow_sigma_bounds(shift: float, *, base_steps: int = FLOW_LADDER_STEPS
                      ) -> tuple[float, float]:
    """flow 族**自己的** σ 定义域端点 ``(σ_min, σ_max)`` ✓ —— 不是 EDM 那对 ✗✗。

    出处：``ModelSamplingDiscreteFlow.sigmas = [σ(i/1000) for i in 1..1000]`` ✓ ⇒
    上端 ``σ(1) = 1``（任意 shift 的不动点 ✓）、下端 ``σ(1/1000)`` ✓（shift=3 ⇒ ``0.002994`` ✓）。

    ⚠️ 为什么必须问模型要这对端点 ✗✗：``karras`` / ``normal`` 这些名字在**参考实现**里都是拿
    ``model_sampling.sigma_min/sigma_max`` 当端点算的 ✓（``comfy/samplers.py::calculate_sigmas`` ✓
    逐字：``sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=model_sampling.sigma_min,
    sigma_max=model_sampling.sigma_max, ...)`` ✓）—— 也就是说它们**不是**「EDM 专用调度」✓，
    是「在**模型自己的** σ 区间上画曲线」✓。拿 EDM 的 ``0.0292 / 14.6146`` 套到 flow 模型上 ✓✗
    ⇒ 首值 14.6 ⇒ DiT 收到 ``t = 1 − σ = −13.6`` ✓ ⇒ **不报错、出糊图** ✓✗（本仓踩过 ✓）。
    """
    ladder = _ladder(max(1, int(base_steps)), shift=float(shift),
                     sigma_min=DEFAULT_SIGMA_MIN, sigma_max=DEFAULT_SIGMA_MAX)
    return float(ladder[0]), float(ladder[-1])


def normal_sigmas(steps: int, *, sigma_min: float = DEFAULT_SIGMA_MIN,
                  sigma_max: float = DEFAULT_SIGMA_MAX,
                  shift: float | None = None) -> list[float]:
    """``normal`` 调度 ✓ —— 语义是「**timestep 等距**」，两种域下落到 σ 上是**两条不同的曲线** ✓✗。

    出处（逐字对齐 ✓）：``comfy/samplers.py::normal_scheduler`` ✓

    .. code-block:: python

        start = s.timestep(s.sigma_max); end = s.timestep(s.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = [s.sigma(t) for t in timesteps] + [0.0]

    ⇒ ① **离散/EDM 域**（``shift=None`` ✓）：``timestep ↔ σ`` 是 log 关系 ✓ ⇒ timestep 等距
    = **对数等距** ⇒ 几何插值 ✓（本函数的老行为 ✓，逐位不变 ✓）；
    ② **flow 域**（给了 ``shift`` ✓）：``multiplier = 1`` ⇒ ``timestep(sigma) = sigma`` **恒等** ✓
    ⇒ timestep 等距 = **σ 等距** ✓，再各点过一遍 shift ✓（``s.sigma(t)`` 那一步 ✓）⇒ 与
    :func:`flow_sigmas_for_steps`（**t** 等距）**不是同一条** ✓✗：shift=3 / 8 步时第 2 点
    ``0.9475`` vs ``0.9545`` ✓、末段密度也不同 ✓ —— 写成 flow 的 t 等分不是「换个名字」✗，
    是真的换了日程 ✓（不报错、只是与参考实现差一截 ✓✗）。

    ⚠️ flow 域**必须**用 :func:`flow_sigma_bounds` 那对端点 ✗✗（不是 EDM 的 14.6 ✓）——
    见该函数的说明 ✓。
    """
    steps = max(1, int(steps))
    if shift is not None:
        shift = float(shift)
        if shift <= 0:
            raise ValueError(f"shift 必须为正（收到 {shift} ✗）")
        floor, top = flow_sigma_bounds(shift)
        #: ⚠️ 端点**先取模型自己那对** ✓，再逐点过 shift ✓（= ``s.sigma(timesteps)`` 那一步 ✓）。
        grid = _linspace(top, floor, steps)
        return [shift * value / (1.0 + (shift - 1.0) * value) for value in grid] + [0.0]
    log_min, log_max = math.log(sigma_min), math.log(sigma_max)
    return [math.exp(value) for value in _linspace(log_max, log_min, steps)] + [0.0]


def ddim_uniform_sigmas(steps: int, *, sigma_min: float = DEFAULT_SIGMA_MIN,
                        sigma_max: float = DEFAULT_SIGMA_MAX) -> list[float]:
    """DDIM 风格：t 在 ``[0,1]`` 均匀取点，再线性映射到 σ ✓（步数少时更「粗」但稳定 ✓）。"""
    steps = max(1, int(steps))
    return [sigma_max + (sigma_min - sigma_max) * t for t in _linspace(0.0, 1.0, steps)] + [0.0]


#: flow matching 的**默认 shift** ✓（Z-Image ✓：`supported_models.ZImage.sampling_settings` 里
#: ``multiplier=1.0, shift=3.0`` ✓；官方 turbo 模板 `image_z_image_turbo.json` 的
#: `ModelSamplingAuraFlow(shift=3)` 是同一条 ✓ —— 两处独立来源对得上 ✓）。
FLOW_SHIFT_DEFAULT = 3.0


def flow_sigmas_for_steps(steps: int, *, shift: float = FLOW_SHIFT_DEFAULT) -> list[float]:
    """**flow matching** 的 σ 序列 ✓（``t`` 均匀取点 + ``shift`` 平移 ✓；末尾 0.0 ✓）。

    公式（闭式 ✓ 纯数学 ✓ 零依赖 ✓）：

    ``σ(t) = shift·t / (1 + (shift−1)·t)`` ✓，``t`` 从 **1 递减到 0** ✓；取 ``steps+1`` 个点
    ⇒ ``σ_0 = 1`` ✓、``σ_steps = 0`` ✓、中间单调递减 ✓。

    ⚠️ 这不是「karras 换个系数」✗ —— 两者**自变量域都不同** ✗：karras 的 σ 能到 **14.6** ✓，而
    flow 的 σ **恒在 [0,1]** ✓（它本来就是「t=1 是纯噪声 ✓」的约定 ✓）。拿 karras 那条喂 Z-Image ✓✗
    ⇒ DiT 收到 σ=14.6 ⇒ ``t = 1 − 14.6 = −13.6`` ✓✗（**不报错** ✓，图像是一团糊 ✓✗）。

    ⚠️ ``shift`` 的语义也要对齐 ✗：Z-Image 的 ``multiplier = 1.0`` ✓ ⇒ **喂给模型的 timestep 就是 σ 本身** ✓
    （不是 ``σ×1000`` ✗ —— 那是 multiplier=1000 的老档 ✓，`ModelSamplingAuraFlow.patch_aura` 明写
    ``multiplier=1.0`` ✓ 见 `comfy_extras/nodes_model_advanced.py` ✓）。
    DiT 内部再自己 ``(1 − σ) × time_scale(1000)`` ✓（`ldm/lumina/model.py` 的 ``t = 1.0 - timesteps`` ✓）
    ⇒ 这一层**不要**提前乘 1000 ✓✗，否则 ``t`` 落到 −1e6 量级 ✓。
    """
    steps = max(1, int(steps))
    shift = float(shift)
    if shift <= 0:
        raise ValueError(f"shift 必须为正（收到 {shift} ✗）")
    sigmas = [1.0]
    for index in range(1, steps):
        t = 1.0 - index / steps
        sigmas.append(shift * t / (1.0 + (shift - 1.0) * t))
    sigmas.append(0.0)
    return sigmas


def flow_shift_for_sequence(seq_len: int, *, frames: int = 1, base_shift: float = 1.0,
                           max_shift: float = 3.0, seq_min: int = 256,
                           seq_max: int = 4096) -> float:
    """按**序列长度**与**帧数**推一个 flow shift ✓（视频专用 ✓；闭式 ✓ 纯数学 ✓ 零依赖 ✓）。

    公式（自己写的 ✓，不是抄代码 ✗）：``α = lin(seq_len) · √frames`` ✓ ——
    ``lin`` 是过 ``(seq_min, base_shift)`` 与 ``(seq_max, max_shift)`` 的**直线** ✓，
    **两端之外也照直走** ✗（刻意不夹断：夹断会让"很短"与"很长"两头退化成同一个常数 ✓，
    于是调分辨率/帧数**看不出任何变化** ✓✗ —— 那种失败一声不响 ✓）。

    ⚠️ 为什么视频需要一个**不是常数**的 shift ✗（写死一个 3 或 12 不是更省事吗 ✓）：
    同一条 shift 在不同 token 数下**不是同一种噪声日程** ✗ —— 序列越长，每步要改的 token 越多，
    固定 shift 会让**噪声占比偏小** ⇒ 运动弱、画面糊 ✓（**不报错** ✓，症状是"怎么调 cfg 都不动" ✓）。
    ``√frames`` 那一项同理 ✓：帧数翻倍 ⇒ 时间轴 token 翻倍 ⇒ 要更大的 shift 才落回同一段分布 ✓。

    不变量（`tests/engine_video_schedule_test.py` 逐条钉住 ✓）：
    ``α(seq_min, frames=1) == base_shift`` ✓、``α(seq_max, frames=1) == max_shift`` ✓、
    ``α(frames=4) == 2·α(frames=1)`` ✓（⇒ ``√`` 写错成线性，判据当场为假 ✓）、
    ``α`` 恒为正 ✓（seq_len 很小时会 < ``base_shift`` ✓ 但不会 ≤ 0 ✓）。

    ⚠️ ``seq_len`` 是**潜空间的 token 数** ✓（``(h_lat // patch) · (w_lat // patch)`` ✓）——
    不是像素数、也不是潜像素数 ✗（拿像素数进来 ⇒ α 大一个量级 ⇒ σ 前几步就冲到 1 附近 ⇒ 白跑 ✓✗）。
    """
    seq_min, seq_max = int(seq_min), int(seq_max)
    if seq_max == seq_min:
        raise ValueError(f"seq_min 与 seq_max 不能相同（收到 {seq_min} ✗ —— 那样直线画不出来 ✓）")
    if float(frames) < 1:
        raise ValueError(f"frames 必须 ≥1（收到 {frames} ✗ —— 它只进 √ 号 ✓）")
    slope = (float(max_shift) - float(base_shift)) / float(seq_max - seq_min)
    alpha = float(base_shift) + slope * (float(seq_len) - float(seq_min))
    alpha *= math.sqrt(float(frames))
    if alpha <= 0:
        raise ValueError(
            f"算出的 flow shift = {alpha:.6g} ≤ 0 ✗（seq_len={seq_len} / frames={frames} ✓）"
            "—— shift 必须为正，否则 σ(t) 的分母会翻号 ✓")
    return alpha


def flow_sigmas_for_video(steps: int, *, shift: float | None = None, seq_len: int | None = None,
                          latent_frames: int = 1, base_shift: float = 1.0,
                          max_shift: float = 3.0, seq_min: int = 256,
                          seq_max: int = 4096) -> list[float]:
    """视频的 flow σ 日程 ✓ —— **要么给 ``shift``**（事实值 ✓）、**要么给 ``seq_len``**（自适应 ✓）。

    ⚠️ 两个都不给就**报错** ✗（不拿 ``FLOW_SHIFT_DEFAULT`` 硬凑 ✓）：图族的默认 3.0 是
    Z-Image 的口径 ✓，视频族的 shift 是**另一个数**（本项目 H3 = 12.0 ✓，见
    ``pipeline.H3_FLOW_SHIFT`` ✓）—— 拿错不会报错 ✓，只是运动量与清晰度一起掉 ✓✗。
    """
    if shift is None:
        if seq_len is None:
            raise ValueError(
                "flow 视频日程要么给 shift、要么给 seq_len ✗（两个都没给 ✓ —— 不猜一个默认 shift ✓）")
        shift = flow_shift_for_sequence(seq_len, frames=latent_frames, base_shift=base_shift,
                                        max_shift=max_shift, seq_min=seq_min, seq_max=seq_max)
    return flow_sigmas_for_steps(steps, shift=float(shift))


def _ladder(base_steps: int, *, shift: float | None, sigma_min: float,
            sigma_max: float) -> list[float]:
    """**模型自带的那条细梯子** ✓（`simple` / `beta` 都从它上面取点 ✓）。

    ``shift`` 给了 ⇒ **flow 域** ✓（``σ(t) = shift·t/(1+(shift−1)t)`` ✓，``t = (i+1)/base_steps`` ⇒
    ``σ_last = 1`` ✓，与 ``ModelSamplingDiscreteFlow.sigmas`` **逐位同构** ✓）；
    没给 ⇒ 几何梯子（``sigma_max`` ⇒ ``sigma_min`` ✓，离散模型的 ``sigmas`` 在 log 域上就是这个形状 ✓）。
    """
    count = max(1, int(base_steps))
    if shift is None:
        log_min, log_max = math.log(sigma_min), math.log(sigma_max)
        # ⚠️ 必须是**升序** ✓（0 号最小 ✓、末号 = ``sigma_max`` ✓）—— 与 comfy 的
        #    ``model_sampling.sigmas`` 同向 ✓（它从"最干净"排到"最噪" ✓，取点靠 ``-(1+idx)`` ✓）。
        #    写成降序**不报错** ✓✗：``simple`` 会原样倒过来用 ✓，σ 从 0.03 起步 ⇒ 一轮全是细修 ✓。
        return [math.exp(log_min + (log_max - log_min) * index / max(1, count - 1))
                for index in range(count)]
    shift = float(shift)
    if shift <= 0:
        raise ValueError(f"shift 必须为正（收到 {shift} ✗）")
    sigmas = []
    for index in range(count):
        t = (index + 1) / count                 # ← 与 comfy 的 `i/1000, i=1..1000` 同一式子 ✓
        sigmas.append(shift * t / (1.0 + (shift - 1.0) * t))
    return sigmas


def _betacf(a: float, b: float, x: float) -> float:
    """正则化不完全 beta 的**连分式**（Lentz 法 ✓；纯 math ✓ 零依赖 ✓）。"""
    tiny, eps = 1e-300, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for step in range(1, 201):
        twice = 2 * step
        for numerator in (step * (b - step) * x / ((qam + twice) * (a + twice)),
                          -(a + step) * (qab + step) * x / ((a + twice) * (qap + twice))):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """正则化不完全 beta ``I_x(a,b)`` ✓（x 单调增 ✓ ⇒ 可二分求逆 ✓）。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(probability: float, alpha: float, beta: float) -> float:
    """beta 分布的**分位函数** ✓（二分求逆 ✓；``alpha=beta=1`` ⇒ 恒等 ✓）。

    为什么自己写 ✗：``scipy.stats.beta.ppf`` 要 scipy ✓，本仓零依赖 ✓。
    精度到 1e-12 量级就够 ✓ —— 下游还要 ``rint`` 到 1/1000 的格子上 ✓。
    """
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if _betainc(alpha, beta, middle) < probability:
            low = middle
        else:
            high = middle
        if high - low < 1e-13:
            break
    return (low + high) / 2.0


def simple_sigmas(steps: int, *, shift: float | None = None,
                  sigma_min: float = DEFAULT_SIGMA_MIN,
                  sigma_max: float = DEFAULT_SIGMA_MAX,
                  base_steps: int = FLOW_LADDER_STEPS) -> list[float]:
    """``simple`` 调度 ✓ —— **在模型自带的那条细梯子上等距取点** ✓（不是另造一条曲线 ✗）。

    出处（逐字对齐 ✓）：``comfy/samplers.py::simple_scheduler`` ✓ ——

    .. code-block:: python

        ss = len(s.sigmas) / steps
        for x in range(steps):
            sigs += [float(s.sigmas[-(1 + int(x * ss))])]
        sigs += [0.0]

    ⚠️ 两个必须照抄的细节（差一点只是在**第 4 位小数**上不同 ✓，肉眼看不出来 ✓✗）：
    ① 取点是 ``int(x·1000/steps)`` **向下**取整 ✓（**不是** ``round`` ✗）；
    ② 梯子本身是 ``1000`` 格 ✓ —— 于是 ``steps`` 整除 1000 时（8/10/20/25/40/50 ✓）结果与
       :func:`flow_sigmas_for_steps` **完全一致** ✓，除不尽时（30/50 以外 ✓）会略有偏差 ✓。

    ``shift`` 给了 ⇒ 走 **flow 域**（H3/图族 ✓，与 ComfyUI 的 flow 模型同一把尺子 ✓）。
    """
    steps = max(1, int(steps))
    ladder = _ladder(base_steps, shift=shift, sigma_min=sigma_min, sigma_max=sigma_max)
    grid = len(ladder)
    sigmas = []
    for index in range(steps):
        pick = grid - 1 - int(index * grid / steps)
        sigmas.append(float(ladder[max(0, min(grid - 1, pick))]))
    return sigmas + [0.0]


def beta_sigmas(steps: int, *, alpha: float = 0.6, beta: float = 0.6,
                shift: float | None = None, sigma_min: float = DEFAULT_SIGMA_MIN,
                sigma_max: float = DEFAULT_SIGMA_MAX,
                base_steps: int = FLOW_LADDER_STEPS) -> list[float]:
    """``beta`` 调度 ✓（出处 ``comfy/samplers.py::beta_scheduler`` ✓，论文 arXiv:2407.12173 ✓）。

    .. code-block:: python

        ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
        ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * (len(sigmas) - 1))
        # 相邻重复值只留一个，末尾补 0.0

    ``alpha`` / ``beta`` 越小 ⇒ 取点越挤向**高噪声**端 ✓（前几步改动大、后几步细修 ✓）；
    ``1.0/1.0`` ⇒ 与 ``simple`` 几乎重合 ✓（差别只在 ``rint`` 与 ``int`` 那一格 ✓）。

    ⚠️ 相邻重复值会**合并** ✓ ⇒ 返回长度可能 **小于** ``steps + 1`` ✓（这是参考实现的语义 ✓，
    不是 bug ✗）；调用方拿 ``len(sigmas) - 1`` 当步数即可 ✓。
    """
    steps = max(1, int(steps))
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"alpha/beta 必须为正（收到 {alpha}/{beta} ✗）")
    ladder = _ladder(base_steps, shift=shift, sigma_min=sigma_min, sigma_max=sigma_max)
    total = len(ladder) - 1
    sigmas: list[float] = []
    last: int | None = None
    for index in range(steps):
        probability = 1.0 - index / steps            # ← `linspace(0,1,steps,endpoint=False)` 的 1−t ✓
        picked = int(round(beta_ppf(max(1e-12, min(1.0 - 1e-12, probability)),
                                    alpha, beta) * total))
        picked = max(0, min(total, picked))
        if picked != last:
            sigmas.append(float(ladder[picked]))
            last = picked
    return sigmas + [0.0]


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
    # 2026-11-01 补两档 ✓：**工作流侧 8 个工作流全用 `simple`** ✓（`16/17/18/19/21` 与
    # `01/02/03/20` 的 KSampler 里写的都是 `scheduler="simple"` ✓）—— 引擎不认这个名字
    # 就只能报「未知调度」✗，于是「按工作流参数原样跑」这条路断在这里 ✓✗。
    "simple": simple_sigmas,
    "beta": beta_sigmas,
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
