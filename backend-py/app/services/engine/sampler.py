"""采样器（**自己实现**，零依赖 ✓）—— 扩散/流匹配的去噪循环。

## 为什么零依赖

采样循环本质只是「标量 × 张量 + 张量」的**代数**（`x ← x + d·(σ_next − σ)` ✓），
与框架无关 ✓。把它写成**张量无关**（只要求对象支持 ``+ − * float`` ✓）：

* 真机上 ``x`` 就是 ``torch.Tensor`` ✓；
* 自检里用一个 **12 行的假张量** 就能验算法（步数、方向、收敛、回调顺序 ✓）——**不需要 GPU、不需要权重、不需要 torch** ✓。

## 模型接口（鸭子类型 ✓）

``model(x, sigma) -> denoised``：给定当前样本与噪声强度，返回**去噪后的估计** ✓
（也就是通常说的 x0/denoised ✓）。三种常见预测（eps / v / x0）由**调用方**在闭包里转换 ✓ ——
采样器只管几何 ✓（这样换模型不用改采样器 ✓）。

## 算法（公开数学，不是照抄任何项目的源码 ✗）

* ``euler``：``d = (x − denoised)/σ``；``x ← x + d·(σ_next − σ)``（一阶、稳、步数够就很好 ✓）
* ``heun``：先 Euler 预测一步，再用**两端斜率平均**校正（二阶 ✓，同sigma数下更精细但**每步两次模型调用** ✓）
* ``multistep``：缓存上一步的 ``d``，用线性外推做二阶（少步场景比 euler 好 ✓，模型调用数与 euler 相同 ✓）
* ``dpmpp_2m``：DPM-Solver++(2M) ✓ —— 在**对数 σ 空间**做二阶多步，用上一轮 denoised 做线性修正
  （少步质量的性价比最高的一档，Flux/Wan 社区默认 ✓）
* ``lms``：线性多步 Adams-Bashforth（4 阶 ✓），系数由多项式积分给出
* ``euler_ancestral``：祖先采样（每步按 η 回注噪声 ✓，同 seed 可复现但结果与 euler 不同族 ✓）
* ``sa_solver``：随机 Adams 求解器（NeurIPS 2023 ✓）—— 高阶 + 可控随机性，**10–15 步达到 25 步观感** ✓

## 与参考实现的差别（写清楚 ✗，免得读成"等价"✓）

参考实现（ComfyUI `k_diffusion/sampling.py` ✓）把**标量系数也用张量算**（`torch.tensor` 参与
``vander`` / ``linalg.solve`` ✓）⇒ 每步都要在卡上算几个 4×4 矩阵再同步回 CPU ✓✗。
本模块是**张量无关**的，系数一律用 Python ``float`` 算（:func:`_sa_b_coeffs` 等 ✓）：
数学一致 ✓、少掉每步的 GPU↔CPU 往返 ✓、且自检里不需要 torch ✓✓。
唯一需要张量的只有「噪声回注」——那一步用调用方传进来的 ``noise_sampler`` ✓（默认 ``None`` ⇒ 纯 ODE）
——真管线在 :func:`app.services.engine.pipeline._noise_sampler_for` 里用后端的 ``noise_like`` 把它接上 ✓
（后端没实现就**明确报错** ✗，不静默退化成 ODE ✗）。

## 两张名单（都是**唯一真源** ✓）

* :data:`SAMPLERS` —— 可用的名字 ✓（:func:`sample` 与管线 plan 阶段都查它 ✓）；
* :data:`STOCHASTIC` —— 哪些**需要** ``noise_sampler`` 才真有随机性 ✓（管线据此决定接不接噪声 ✓）；
* :data:`MODEL_CALLS` —— 每步调几次模型 ✓（耗时系数读它 ✓，见 :func:`model_calls_per_step` ✓）。
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable, Sequence

__all__ = [
    "SAMPLERS", "STOCHASTIC", "MODEL_CALLS", "sample", "model_calls_per_step",
    "sample_euler", "sample_heun", "sample_multistep",
    "sample_dpmpp_2m", "sample_lms", "sample_euler_ancestral", "sample_sa_solver",
]

ModelFn = Callable[[Any, float], Any]
ProgressFn = Callable[[int, float, Any], None]
#: ``noise_sampler(sigma_from, sigma_to) -> 与 x 同形的标准正态噪声`` ✓
#: （随机类采样器用；``None`` ⇒ 退化成确定性 ODE ✓，不认识噪声的采样器直接忽略它 ✓）
NoiseFn = Callable[[float, float], Any]


def _sigma_float(sigma: Any) -> float:
    """σ 取成 Python float（兼容 torch 标量张量 ✓）。"""
    try:
        return float(sigma)
    except (TypeError, ValueError):  # pragma: no cover - 张量标量的兜底
        return float(sigma.item())  # type: ignore[attr-defined]


def _denoised(model: ModelFn, x: Any, sigma: Any) -> Any:
    """调模型拿去噪估计（σ=0 时没有噪声可去 ⇒ 原样返回 ✓，避免除零 ✗）。"""
    if _sigma_float(sigma) <= 0:
        return x
    value = model(x, sigma)
    if value is None:
        raise ValueError("模型返回 None（denoised 估计不能为空 ✓）")
    return value


def _direction(x: Any, denoised: Any, sigma: Any) -> Any:
    """``d = (x − denoised)/σ``（扩散参数化的通用方向 ✓）。"""
    sigma_value = _sigma_float(sigma)
    if sigma_value <= 0:
        return x - denoised  # 理论上不会走到（末位 σ=0 直接结束 ✓）
    return (x - denoised) * (1.0 / sigma_value)


def _combine(coeffs: Sequence[float], values: Sequence[Any]) -> Any:
    """``Σ coeff_i · value_i`` ✓（第一项用乘法起头 ⇒ 不需要 ``0 · 张量`` ✓）。"""
    total = values[0] * float(coeffs[0])
    for coeff, value in zip(coeffs[1:], values[1:]):
        total = total + value * float(coeff)
    return total


# ── 标量系数工具（**纯 Python** ✓ 不碰 torch ✓，见文件头「与参考实现的差别」）──────

def _half_log_snr(sigmas: Sequence[float], mode: str = "flow") -> list[float]:
    """σ 序列 ⇒ 半对数信噪比 ``λ`` ✓（**两个参数化各一式，别混用** ✗✗）。

    * ``flow``（流匹配 ✓ 主力）：``λ = log((1−σ)/σ) = −logit(σ)`` ✓
    * ``snr``（离散 / karras ✓ SDXL 那类）：``λ = −log σ`` ✓

    ⚠️⚠️ 口径来源：ComfyUI ``k_diffusion/sampling.py::sigma_to_half_log_snr`` 按
    ``model_sampling`` 是不是 ``CONST``（＝flow）分这两支 ✓ —— **不是可选装饰** ✗：
    拿错一支 ⇒ 高阶项系数全错 ⇒ 出图发灰/糊 ✓（**不报错** ✗，最难查的那种）。
    """
    kind = str(mode or "flow").lower()
    if kind not in ("flow", "snr"):
        raise ValueError(f"未知 logsnr 口径 {mode!r}；可用：['flow', 'snr']")
    out: list[float] = []
    for sigma in sigmas:
        value = _sigma_float(sigma)
        if value <= 0:
            out.append(math.inf)          # 末位 σ=0 ⇒ λ=+inf ✓（不参与运算就无害 ✓）
        elif kind == "flow":
            out.append(-math.inf if value >= 1.0 else math.log((1.0 - value) / value))
        else:
            out.append(-math.log(value))
    return out


def _offset_first_sigma(sigmas: Sequence[float], mode: str = "flow",
                        percent_offset: float = 1e-4) -> list[float]:
    """flow 口径下把首 σ **压出 1.0** ✓✗。

    否则 ``λ₀ = log(0/1) = −inf`` ⇒ 第一个区间 ``h = λ₁ − λ₀ = +inf`` ⇒ **一步炸** ✓✗
    （症状：首步就出纯色 / NaN ✓，而 euler 用得明明很好 ✓ —— 因为 euler 根本不算 λ ✓）。
    ComfyUI ``offset_first_sigma_for_snr`` 只对 flow 且 ``σ₀ ≥ 1`` 时换成
    ``percent_to_sigma(1e-4) = 1 − 1e-4`` ✓，这里照此办理 ✓。
    """
    values = [_sigma_float(sigma) for sigma in sigmas]
    if str(mode or "flow").lower() == "flow" and len(values) > 1 and values[0] >= 1.0:
        values[0] = 1.0 - float(percent_offset)
    return values


#: 8 点 Gauss–Legendre 节点/权重 ✓（对 ≤15 次多项式**精确** ⇒ LMS 的 order≤4 绰绰有余 ✓✓）
#: 为什么自己带表：参考实现用 ``scipy.integrate.quad`` ✓，而本机 venv **没有 scipy** ✗
#: （`importlib.util.find_spec("scipy")` 实测 False）⇒ 不加依赖、改用固定阶高斯积分 ✓
_GLEG8_X = (-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
            0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363)
_GLEG8_W = (0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
            0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763)


def _integrate(fn: Callable[[float], float], start: float, stop: float) -> float:
    """``[start, stop]`` 上的数值积分 ✓（8 点高斯 ⇒ 被积函数是多项式时**无误差** ✓）。"""
    half = 0.5 * (stop - start)
    mid = 0.5 * (stop + start)
    return half * sum(weight * fn(mid + half * node) for node, weight in zip(_GLEG8_X, _GLEG8_W))


def _linear_multistep_coeff(order: int, ts: Sequence[float], index: int, offset: int) -> float:
    """LMS 的 Lagrange 积分权重 ✓（等价于 ComfyUI 同名函数，但那版要 scipy ✓✗）。"""
    if order - 1 > index:
        raise ValueError(f"阶数 {order} 对第 {index} 步太高")
    def fn(tau: float) -> float:
        product = 1.0
        for k in range(order):
            if offset == k:
                continue
            product *= (tau - ts[index - k]) / (ts[index - offset] - ts[index - k])
        return product
    return _integrate(fn, ts[index], ts[index + 1])


def _ancestral_step(sigma_from: float, sigma_to: float, eta: float) -> tuple[float, float]:
    """祖先采样一步：返回 ``(sigma_down, sigma_up)`` ✓（``sigma_up`` ＝ 要回注的噪声量 ✓）。"""
    if not eta or sigma_from <= 0:
        return sigma_to, 0.0
    sigma_up = min(sigma_to,
                   eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = max(sigma_to ** 2 - sigma_up ** 2, 0.0) ** 0.5
    return sigma_down, sigma_up


def _solve_linear(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """解 N×N 线性系统 ✓（部分主元高斯消元；N ≤ 8 ✓）。"""
    size = len(rhs)
    grid = [list(row) + [float(rhs[index])] for index, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(grid[row][col]))
        if abs(grid[pivot][col]) < 1e-300:
            raise ValueError("SA-Solver 系数矩阵奇异（λ 点有重复？✗）")
        if pivot != col:
            grid[col], grid[pivot] = grid[pivot], grid[col]
        for row in range(col + 1, size):
            factor = grid[row][col] / grid[col][col]
            if factor:
                for k in range(col, size + 1):
                    grid[row][k] -= factor * grid[col][k]
    out = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = grid[row][size] - sum(grid[row][k] * out[k] for k in range(row + 1, size))
        out[row] = total / grid[row][row]
    return out


def _sa_integral_coeffs(lambda_s: float, lambda_t: float, order: int, tau: float) -> list[float]:
    """SA-Solver 的指数积分系数 ✓（``(1+τ²)·∫exp((1+τ²)x)x^p dx``，已提出 ``exp((1+τ²)t)``）。

    与参考实现 `compute_exponential_coeffs` 逐项对应（含 ``tril`` 与符号交替 ✓），
    只是把 ``torch`` 的 ``lgamma``／``exp`` 换成 ``math`` ✓。
    """
    tau_mul = 1.0 + tau ** 2
    span = lambda_t - lambda_s
    log_factorial = [math.lgamma(p + 1) for p in range(order)]
    product = [lambda_t ** p - (lambda_s ** p) * math.exp(-tau_mul * span) for p in range(order)]
    coeffs: list[float] = []
    for p in range(order):
        total = 0.0
        for q in range(p + 1):            # 下三角 ⇒ 只保 p ≥ q ✓
            depth = p - q
            value = math.exp(log_factorial[p] - log_factorial[q])
            if tau > 0:
                value *= math.exp(-depth * math.log(tau_mul))
            if depth % 2:
                value = -value
            total += value * product[q]
        coeffs.append(total)
    return coeffs


def _sa_b_coeffs(sigma_next: float, lambdas: Sequence[float], lambda_s: float, lambda_t: float,
                 tau: float, simple_order_2: bool = False,
                 corrector: bool = False) -> list[float]:
    """SA-Solver 的 ``b_i`` 系数 ✓（对应参考 `compute_stochastic_adams_b_coeffs`）。"""
    count = len(lambdas)
    if simple_order_2 and count == 2:
        tau_mul = 1.0 + tau ** 2
        span = lambda_t - lambda_s
        alpha_t = sigma_next * math.exp(lambda_t)
        if corrector:
            b_one = alpha_t * (0.5 * tau_mul * span)
            b_two = alpha_t * (-math.expm1(-span * tau_mul)) - b_one
        else:
            b_two = alpha_t * (0.5 * tau_mul * span ** 2) / (lambdas[-2] - lambda_s)
            b_one = alpha_t * (-math.expm1(-span * tau_mul)) - b_two
        return [b_two, b_one]
    integral = _sa_integral_coeffs(lambda_s, lambda_t, count, tau)
    # 参考实现：vander(lambdas, count, increasing=True).T 后 solve ⇒ A[j][k] = lambdas[k] ** j ✓
    matrix = [[lambdas[k] ** j for k in range(count)] for j in range(count)]
    weights = _solve_linear(matrix, integral)
    alpha_t = sigma_next * math.exp(lambda_t)
    return [alpha_t * weight for weight in weights]


def _stochastic_strength(noise_sampler: NoiseFn | None, eta: float,
                         s_noise: float) -> float:
    """随机强度 ✓：**只有真能回注噪声时**才允许 η > 0 ✓✗（否则强制 0 ⇒ 纯 ODE ✓）。

    ⚠️⚠️ 这条是**唯一的真源** ✓✗，两个随机采样器（``euler_ancestral`` / ``sa_solver``）
    都必须走它 ✓。首版只给 ``sa_solver`` 写了、``euler_ancestral`` 漏了 ✓✗ ——
    后果：无 ``noise_sampler`` 时它按 η=1 **缩短了步长**（``σ_down = σ_next²/σ``）却**不注入噪声** ⇒
    既没走到该到的位置、也没补上随机量 ⇒ **两边坏事** ✓✗，而它不报错、只是图偏灰 ✓。

    三个闸门都要有 ✗：``noise_sampler`` 为 ``None``（没通路 ／ 显式要 ODE ✓）、
    ``eta <= 0``（用户要确定性 ✓）、``s_noise <= 0``（要确定性但没把 eta 归零 ✓ ——
    参考实现用 ``if tau > 0 and s_noise > 0`` 门控噪声，但**系数仍按 τ 算** ✗，
    于是"系数随机、实际不随机"，数学上不自洽 ✓✗）。
    """
    if noise_sampler is None or float(eta) <= 0 or float(s_noise) <= 0:
        return 0.0
    return float(eta)


def _tau_interval(sigmas: Sequence[float], eta: float) -> Callable[[float], float]:
    """默认的随机强度区间 ✓：σ 落在「首 σ 的 20%~80%」时 ``τ = η``，其余为 0（纯 ODE ✓）。

    ⚠️ 参考实现用 ``percent_to_sigma(0.2)``／``percent_to_sigma(0.8)``（按**时间百分比**换算 ✓），
    这里改用**相对 σ 位置** ✓ —— 对 flow（首 σ≈1）两者数值一致（[0.8, 0.2] ✓）；
    对 karras 这类首 σ 远大于 1 的调度，相对位置是更稳的通用化 ✓（**不声称等价** ✗）。
    """
    top = max(sigmas) if sigmas else 0.0
    start, end = top * 0.8, top * 0.2
    def tau_of(sigma: float) -> float:
        return eta if eta > 0 and start >= sigma >= end else 0.0
    return tau_of


def sample_euler(model: ModelFn, x: Any, sigmas: list[float], *,
                 callback: ProgressFn | None = None) -> tuple[Any, int]:
    """一阶 Euler ✓（每个区间一次模型调用 ✓）。"""
    steps = 0
    for index in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[index], sigmas[index + 1]
        denoised = _denoised(model, x, sigma)
        x = x + _direction(x, denoised, sigma) * (_sigma_float(sigma_next) - _sigma_float(sigma))
        steps += 1
        if callback:
            callback(index + 1, sigma_next, x)
    return x, steps


def sample_heun(model: ModelFn, x: Any, sigmas: list[float], *,
                callback: ProgressFn | None = None) -> tuple[Any, int]:
    """二阶 Heun（预测 + 校正 ✓；区间末尾（σ_next=0）退化为一次调用 ✓）。"""
    steps = 0
    for index in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[index], sigmas[index + 1]
        denoised = _denoised(model, x, sigma)
        d0 = _direction(x, denoised, sigma)
        span = _sigma_float(sigma_next) - _sigma_float(sigma)
        x_predicted = x + d0 * span
        if _sigma_float(sigma_next) > 0:
            denoised_next = _denoised(model, x_predicted, sigma_next)
            d1 = _direction(x_predicted, denoised_next, sigma_next)
            x = x + (d0 + d1) * (0.5 * span)
        else:
            x = x_predicted
        steps += 1
        if callback:
            callback(index + 1, sigma_next, x)
    return x, steps


def sample_multistep(model: ModelFn, x: Any, sigmas: list[float], *,
                     callback: ProgressFn | None = None) -> tuple[Any, int]:
    """多阶（二阶）采样：用**上一步的方向**做线性外推 ✓（调用次数与 Euler 相同 ✓）。

    外推式（我们自己的写法 ✓）：``d_used = 1.5·d_now − 0.5·d_prev``（Adam/中点外推同款思想 ✓）；
    首步没有历史 ⇒ 退化为 Euler ✓。
    """
    previous: Any = None
    steps = 0
    for index in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[index], sigmas[index + 1]
        denoised = _denoised(model, x, sigma)
        current = _direction(x, denoised, sigma)
        direction = current if previous is None else (current * 1.5 - previous * 0.5)
        previous = copy.copy(current)
        x = x + direction * (_sigma_float(sigma_next) - _sigma_float(sigma))
        steps += 1
        if callback:
            callback(index + 1, sigma_next, x)
    return x, steps


def sample_dpmpp_2m(model: ModelFn, x: Any, sigmas: list[float], *,
                    callback: ProgressFn | None = None,
                    noise_sampler: NoiseFn | None = None) -> tuple[Any, int]:
    """DPM-Solver++(2M) ✓ —— 对数 σ 空间里的二阶多步（**每步一次模型调用** ✓）。

    为什么值得当默认档 ✓：``euler`` 在 σ 空间走**直线**，而真实轨迹在对数空间才是直的 ⇒
    同步数下 dpmpp_2m 明显更「敢走」✓；又不像 ``heun`` 要两次调用 ✓ ⇒
    8–20 步的少步场景是性价比最高的一档 ✓（Flux/Wan 社区默认就是它 ✓）。
    """
    times = [_sigma_float(sigma) for sigma in sigmas]
    def time_of(sigma: float) -> float:
        return math.inf if sigma <= 0 else -math.log(sigma)
    def sigma_of(time: float) -> float:
        return 0.0 if time == math.inf else math.exp(-time)

    previous: Any = None
    steps = 0
    for index in range(len(times) - 1):
        sigma, sigma_next = times[index], times[index + 1]
        denoised = _denoised(model, x, sigma)
        now, later = time_of(sigma), time_of(sigma_next)
        span = later - now
        if previous is None or sigma_next == 0 or span == 0:
            # 首步 / 末步（σ→0 ⇒ 系数正好化成 x = denoised ✓）/ 间隔退化 ⇒ 一阶
            scale = sigma_of(later) / sigma_of(now) if now != later else 1.0
            x = x * scale - denoised * math.expm1(-span)
        else:
            span_last = now - time_of(times[index - 1])
            ratio = span_last / span
            guided = denoised * (1.0 + 1.0 / (2.0 * ratio)) - previous * (1.0 / (2.0 * ratio))
            x = x * (sigma_of(later) / sigma_of(now)) - guided * math.expm1(-span)
        previous = denoised
        steps += 1
        if callback:
            callback(index + 1, sigma_next, x)
    return x, steps


def sample_lms(model: ModelFn, x: Any, sigmas: list[float], *,
               callback: ProgressFn | None = None,
               noise_sampler: NoiseFn | None = None,
               order: int = 4) -> tuple[Any, int]:
    """线性多步（Adams–Bashforth，默认 4 阶 ✓）—— 复用历史方向，模型调用数与 euler 相同 ✓。

    ⚠️ 与参考实现的唯一实质差别：``linear_multistep_coeff`` 那一步参考用 `scipy.quad` ✓，
    本实现用 8 点高斯积分 ✓（被积函数是 ≤order−1 次多项式 ⇒ **数值精确** ✓，且不加依赖 ✓）。
    """
    times = [_sigma_float(sigma) for sigma in sigmas]
    history: list[Any] = []
    steps = 0
    for index in range(len(times) - 1):
        sigma, sigma_next = times[index], times[index + 1]
        denoised = _denoised(model, x, sigma)
        history.append(_direction(x, denoised, sigma))
        if len(history) > order:
            del history[0]
        if sigma_next == 0:
            x = denoised
        else:
            used = min(index + 1, order)
            coeffs = [_linear_multistep_coeff(used, times, index, offset) for offset in range(used)]
            x = x + _combine(coeffs, list(reversed(history)))
        steps += 1
        if callback:
            callback(index + 1, sigma_next, x)
    return x, steps


def sample_euler_ancestral(model: ModelFn, x: Any, sigmas: list[float], *,
                           callback: ProgressFn | None = None,
                           noise_sampler: NoiseFn | None = None,
                           eta: float = 1.0, s_noise: float = 1.0,
                           parameterization: str = "flow") -> tuple[Any, int]:
    """祖先 Euler ✓（每步按 ``eta`` 回注噪声 —— 随机类采样器 ✓）。

    ⚠️ **两种参数化的式子不同** ✗，别混用（参考实现按 ``CONST`` 分两支 ✓）：

    * ``flow``：按 ``α = 1 − σ`` 换算，步长按 ``sigma_down/sigma`` 比值混合 ✓；
    * ``discrete``：Karras 式祖先步，``d·(sigma_down − σ) + 噪声·sigma_up`` ✓。

    ⚠️ ``noise_sampler=None`` 时**自动退化成确定性 euler** ✓（不是"随机但没加噪声"✗ ——
    那种会在同一 seed 下每次都给不同图、却看起来像 ODE ✓✗ 最难查）。
    """
    kind = str(parameterization or "flow").lower()
    if kind not in ("flow", "discrete"):
        raise ValueError(f"未知参数化 {parameterization!r}；可用：['flow', 'discrete']")
    strength = _stochastic_strength(noise_sampler, eta, s_noise)
    times = [_sigma_float(sigma) for sigma in sigmas]
    steps = 0
    for index in range(len(times) - 1):
        sigma, sigma_next = times[index], times[index + 1]
        denoised = _denoised(model, x, sigma)
        if sigma_next == 0:
            x = denoised
        elif kind == "flow":
            ratio = 1.0 + (sigma_next / sigma - 1.0) * strength if sigma > 0 else 1.0
            sigma_down = sigma_next * ratio
            alpha_next, alpha_down = 1.0 - sigma_next, 1.0 - sigma_down
            if alpha_down == 0:
                x = denoised
            else:
                mix = sigma_down / sigma
                x = x * mix + denoised * (1.0 - mix)
                if strength > 0:
                    renoise = max(sigma_next ** 2 - sigma_down ** 2 * alpha_next ** 2
                                  / alpha_down ** 2, 0.0) ** 0.5
                    x = x * (alpha_next / alpha_down) + noise_sampler(sigma, sigma_next) * (renoise * s_noise)
        else:
            sigma_down, sigma_up = _ancestral_step(sigma, sigma_next, strength)
            if sigma_down == 0:
                x = denoised
            else:
                x = x + _direction(x, denoised, sigma) * (sigma_down - sigma)
                if strength > 0 and sigma_up > 0:
                    x = x + noise_sampler(sigma, sigma_next) * (sigma_up * s_noise)
        steps += 1
        if callback:
            callback(index + 1, sigma_next, x)
    return x, steps


def sample_sa_solver(model: ModelFn, x: Any, sigmas: list[float], *,
                     callback: ProgressFn | None = None,
                     noise_sampler: NoiseFn | None = None,
                     tau_func: Callable[[float], float] | None = None,
                     s_noise: float = 1.0, eta: float = 1.0,
                     predictor_order: int = 3, corrector_order: int = 4,
                     simple_order_2: bool = False,
                     logsnr: str = "flow") -> tuple[Any, int]:
    """随机 Adams 求解器（SA-Solver，NeurIPS 2023 ✓）—— 预测–校正双段，高阶 + 可控随机性 ✓。

    实测价值：**10–15 步出 25 步的观感** ✓（这是它相对 euler/dpmpp 的全部意义）。

    ⚠️ 三处与参考实现的口径差异（都写清 ✗，别读成等价 ✓）：

    1. **系数用 Python float 算** ✓（参考用张量算 ⇒ 每步 4×4 的 ``linalg.solve`` 要过一遍卡 ✓✗）；
    2. **随机强度区间用相对 σ 位置** ✓（参考用 ``percent_to_sigma`` ✓；对 flow 数值一致 ✓）；
    3. ``noise_sampler=None`` ⇒ **强制 η=0 走纯 ODE** ✓（不是"τ 照算但不加噪声"✗ ——
       那样系数按有噪声推导、实际没加 ⇒ 数学不自洽 ✓✗）。

    ``simple_order_2`` 打开时走论文附录 D 的二阶简化式 ✓（更快更稳，高阶收益变小 ✓）。
    """
    if len(sigmas) <= 1:
        return x, 0
    values = _offset_first_sigma(sigmas, logsnr)
    lambdas = _half_log_snr(values, logsnr)
    strength = _stochastic_strength(noise_sampler, eta, s_noise)
    if tau_func is None:
        tau_func = _tau_interval(values, strength)
    max_order = max(int(predictor_order), int(corrector_order), 1)
    lower_to_end = values[-1] == 0.0

    x_pred = x
    span = 0.0
    tau = 0.0
    noise: Any = None
    history: list[Any] = []
    steps = 0
    for index in range(len(values) - 1):
        sigma, sigma_next = values[index], values[index + 1]
        denoised = _denoised(model, x_pred, sigma)
        history.append(denoised)
        if len(history) > max_order:
            del history[0]

        predicted = min(int(predictor_order), len(history))
        corrected = 0 if (index == 0 or sigma_next == 0) else min(int(corrector_order), len(history))
        if lower_to_end:
            predicted = min(predicted, len(values) - 2 - index)
            corrected = min(corrected, len(values) - 1 - index)

        # ── 校正段（在**上一步预测**的状态上做一次高阶修正 ✓）
        if corrected <= 0:
            x = x_pred
        else:
            coeffs = _sa_b_coeffs(sigma, lambdas[index - corrected + 1:index + 1],
                                 lambdas[index - 1], lambdas[index], tau,
                                 simple_order_2, corrector=True)
            x = x * (sigma / values[index - 1] * math.exp(-(tau ** 2) * span))
            x = x + _combine(coeffs, history[-corrected:])
            if tau > 0 and s_noise > 0 and noise is not None:
                x = x + noise

        # ── 预测段
        if sigma_next == 0:
            x_pred = denoised
        else:
            tau = float(tau_func(sigma_next))
            coeffs = _sa_b_coeffs(sigma_next, lambdas[index - predicted + 1:index + 1],
                                  lambdas[index], lambdas[index + 1], tau,
                                  simple_order_2, corrector=False)
            span = lambdas[index + 1] - lambdas[index]
            x_pred = x * (sigma_next / sigma * math.exp(-(tau ** 2) * span))
            x_pred = x_pred + _combine(coeffs, history[-predicted:])
            if tau > 0 and s_noise > 0 and noise_sampler is not None:
                renoise = max(-math.expm1(-2.0 * tau ** 2 * span), 0.0) ** 0.5
                noise = noise_sampler(sigma, sigma_next) * (sigma_next * renoise * s_noise)
                x_pred = x_pred + noise
            else:
                noise = None
        steps += 1
        if callback:
            callback(index + 1, sigma_next, x_pred)
    return x_pred, steps


#: 可用采样器 ✓（新增请同时加自检 ✓）
SAMPLERS: dict[str, Callable[..., tuple[Any, int]]] = {
    "euler": sample_euler,
    "heun": sample_heun,
    "multistep": sample_multistep,
    "dpmpp_2m": sample_dpmpp_2m,
    "lms": sample_lms,
    "euler_ancestral": sample_euler_ancestral,
    "sa_solver": sample_sa_solver,
}

#: 需要 ``noise_sampler`` 才**真有随机性**的那些 ✓（其余传了也会被忽略 ✓）
STOCHASTIC: frozenset[str] = frozenset({"euler_ancestral", "sa_solver"})

#: 每步**模型调用次数** ✓ —— 相对耗时系数（:func:`app.services.engine.pipeline._finish_plan` ✓）
#: 的**唯一真源** ✓✗。
#:
#: ⚠️ 只有 ``heun`` 是 2 ✓（预测一步 + 校正一步 ✓）。``sa_solver`` **是 1 不是 2** ✗：
#: 它的预测段与校正段**吃同一份** ``denoised``（``history[-1]`` ✓），一阶都只调模型一次 ✓ ——
#: 顺手写成 2 会让耗时系数凭空虚高 ✓（用户按它排期 ⇒ 白等 ✓）。
MODEL_CALLS: dict[str, float] = {"heun": 2.0}


def model_calls_per_step(sampler: str) -> float:
    """这个采样器**每步**要调几次模型 ✓（名单与 :func:`sample` 同一份 ✓ —— 未知名字照旧抛 ✓）。

    单独的出口是为了**只有一份账** ✓：原先 ``pipeline`` 里手写 ``== "heun"`` ✓✗，
    新增采样器时那里不会跟着变 ✓（本次加 ``sa_solver``/``dpmpp_2m``/``lms`` 就正是这样漏的 ✓）。
    """
    name = str(sampler or "euler").lower()
    if name not in SAMPLERS:
        raise ValueError(f"未知采样器 {sampler!r}；可用：{sorted(SAMPLERS)}")
    return float(MODEL_CALLS.get(name, 1.0))


def sample(model: ModelFn, x: Any, sigmas: list[float], sampler: str = "euler", *,
           callback: ProgressFn | None = None,
           noise_sampler: NoiseFn | None = None,
           **options: Any) -> tuple[Any, int]:
    """按名字跑采样循环；返回 ``(最终样本, 实际步数)`` ✓。

    ``sigmas`` 必须**单调递减且末位为 0** ✓（由 :mod:`schedules` 保证 ✓）；
    这里再校验一次 —— 序列反了会**静默出噪声** ✗（最难查的那种 ✗）。

    ``options`` 只转给认它的采样器（``eta``/``s_noise``/``order``/``logsnr`` … ✓）；
    传给不认的采样器会抛 ``TypeError`` ✓ —— 这是**有意**的：静默吞掉参数正是
    「设了 ``eta`` 却没生效」这类假绿的来源 ✓✗。
    """
    name = str(sampler or "euler").lower()
    fn = SAMPLERS.get(name)
    if fn is None:
        raise ValueError(f"未知采样器 {sampler!r}；可用：{sorted(SAMPLERS)}")
    values = [_sigma_float(value) for value in sigmas]
    if not values or values[-1] != 0.0:
        raise ValueError("sigmas 末位必须是 0.0（否则没去噪到 x0 ✓）")
    for index in range(len(values) - 1):
        if values[index] < values[index + 1]:
            raise ValueError(f"sigmas 必须单调递减（第 {index} 位 {values[index]} < 后一位 "
                             f"{values[index + 1]} ⇒ 序列反了 ✗）")
    kwargs: dict[str, Any] = {"callback": callback, **options}
    if noise_sampler is not None:
        kwargs["noise_sampler"] = noise_sampler
    return fn(model, x, values, **kwargs)
