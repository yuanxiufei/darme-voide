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
"""
from __future__ import annotations

import copy
from typing import Any, Callable

__all__ = ["SAMPLERS", "sample", "sample_euler", "sample_heun", "sample_multistep"]

ModelFn = Callable[[Any, float], Any]
ProgressFn = Callable[[int, float, Any], None]


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


#: 可用采样器 ✓（新增请同时加自检 ✓）
SAMPLERS: dict[str, Callable[..., tuple[Any, int]]] = {
    "euler": sample_euler,
    "heun": sample_heun,
    "multistep": sample_multistep,
}


def sample(model: ModelFn, x: Any, sigmas: list[float], sampler: str = "euler", *,
           callback: ProgressFn | None = None) -> tuple[Any, int]:
    """按名字跑采样循环；返回 ``(最终样本, 实际步数)`` ✓。

    ``sigmas`` 必须**单调递减且末位为 0** ✓（由 :mod:`schedules` 保证 ✓）；
    这里再校验一次 —— 序列反了会**静默出噪声** ✗（最难查的那种 ✗）。
    """
    fn = SAMPLERS.get(str(sampler or "euler").lower())
    if fn is None:
        raise ValueError(f"未知采样器 {sampler!r}；可用：{sorted(SAMPLERS)}")
    values = [_sigma_float(value) for value in sigmas]
    if not values or values[-1] != 0.0:
        raise ValueError("sigmas 末位必须是 0.0（否则没去噪到 x0 ✓）")
    for index in range(len(values) - 1):
        if values[index] < values[index + 1]:
            raise ValueError(f"sigmas 必须单调递减（第 {index} 位 {values[index]} < 后一位 "
                             f"{values[index + 1]} ⇒ 序列反了 ✗）")
    return fn(model, x, values, callback=callback)
