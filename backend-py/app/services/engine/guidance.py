"""**引导（Classifier-Free Guidance）** + 重标定 + 步区间（零依赖 ✓ 纯数学）。

## 为什么它必须在引擎里（而不是留给后端 ✓）

扩散/流匹配的**采样语义**就是「往哪个方向走」✗ —— 而"往条件更近的方向多走一点"这件事
**跟具体模型无关** ✓：只要求张量支持 ``+ − * float``（与 :mod:`app.services.engine.sampler`
同一套鸭子类型约定 ✓）。所以它属于**引擎**：换后端不该换引导语义 ✓。

## 三条能力（都是公开做法 ✓ 不是照抄某项目源码 ✗）

1. **CFG**：``guided = neg + scale · (pos − neg)`` ✓。``scale = 1`` ⇒ 就是 ``pos``
   （**不打两次模型** ✓ —— 见 :func:`combine` 的短路 ✓）；``scale > 1`` ⇒ 更强地靠提示词
   （常见 3~9 ✓），代价是**每步两次模型调用** ✗。
2. **重标定（rescale）**：CFG 会让结果**方差变大**（常见过曝/过饱和 ✗）⇒
   按 ``guided ← guided · (rms(pos)/rms(guided))`` 把尺度拉回 ✓（``0`` = 不重标定 ✓）。
   ⚠️ 这需要张量能报尺度 ⇒ 走 :func:`stat_scale` 的鸭子类型探测，**探测不到就明确跳过 + 警告** ✗
   （不装作做过 ✓）。
3. **步区间（range）**：只在 ``[start, end)`` 这段**施加引导** ✓（前段留自由度、后段收敛细节 ✓）；
   外加**斜坡（ramp）**：前 ``rampSteps`` 步把 scale 从 1 线性升到目标值 ✓（避免开局被强引导锁死 ✗）。

## ⚠️ 边界（不猜 ✓）

本模块**不知道**目标模型该用多少 scale ✗ —— 那是模型/权重的知识 ✓。
默认 ``scale = 1.0``（等价于不引导 ✓）⇒ **不做任何没被要求的干预** ✓。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = ["GuidanceConfig", "combine", "scale_for_step", "stat_scale"]


@dataclass(frozen=True)
class GuidanceConfig:
    """引导配置（不可变 ✓）。"""

    #: 目标 CFG 强度 ✓（``1.0`` = 不引导 ✓ ⇒ 不做多余干预 ✓）
    scale: float = 1.0
    #: 重标定强度 ✓（``0`` = 不重标定 ✓；``1`` = 完全把尺度拉回正条件 ✓）
    rescale: float = 0.0
    #: 只在这个归一化区间内施加引导 ✓（``[start, end)``，``0..1`` ✓）
    start: float = 0.0
    end: float = 1.0
    #: 前 N 步从 1 线性升到 ``scale`` ✓（``0`` = 恒定 ✓）
    ramp_steps: int = 0

    @property
    def enabled(self) -> bool:
        return abs(float(self.scale) - 1.0) > 1e-9 or float(self.rescale) > 0

    def to_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "rescale": self.rescale, "start": self.start,
                "end": self.end, "rampSteps": self.ramp_steps, "enabled": self.enabled}


def scale_for_step(index: int, total: int, config: GuidanceConfig) -> float:
    """第 ``index`` 步（0 基 ✓）实际用的 scale ✓（区间外返回 1.0 = 不引导 ✓）。

    ``total`` 为 0 时按"全程施加"处理 ✓（不除零 ✗）。
    """
    if not config.enabled:
        return 1.0
    total = max(1, int(total))
    position = (int(index) + 1) / total          # 1..total → 0..1 的进度 ✓
    if position <= float(config.start) or position > float(config.end):
        return 1.0
    target = float(config.scale)
    ramp = int(config.ramp_steps or 0)
    if ramp > 0 and index < ramp:
        return 1.0 + (target - 1.0) * ((int(index) + 1) / ramp)   # 线性升 ✓
    return target


def stat_scale(value: Any) -> float | None:
    """取张量的**尺度**（RMS ✓）—— 鸭子类型探测 ✓，取不到返回 ``None`` ✓（不猜 ✗）。

    * ``torch.Tensor``：有 ``.std()`` ✓（用总体标准差 ✓，与重标定惯例一致 ✓）；
    * 只要支持 ``len()`` 且能算平方和 ✓ ⇒ 用 RMS 兜底（引擎自检的 ``TinyTensor`` 就是这种 ✓）；
    * 都没有 ⇒ ``None`` ⇒ 调用方**跳过重标定并警告** ✗（绝不假装做过 ✓）。
    """
    std = getattr(value, "std", None)
    if callable(std):
        try:
            return abs(float(std()))
        except (TypeError, ValueError, RuntimeError):  # pragma: no cover - 形状异常的张量
            pass
    norm = getattr(value, "norm", None)
    try:
        length = len(value)
    except TypeError:
        return None
    if callable(norm) and length:
        try:
            return float(norm()) / math.sqrt(float(length))
        except (TypeError, ValueError, RuntimeError):  # pragma: no cover
            return None
    return None


def combine(positive: Any, negative: Any, scale: float, *,
            rescale: float = 0.0) -> tuple[Any, dict[str, Any]]:
    """算出这一步的**引导估计** ✓ ⇒ ``(结果, 报告)``。

    * ``scale == 1`` 或 ``negative is None`` ⇒ **短路返回 positive** ✓（不打多余算力 ✗ ✓）；
    * ``rescale > 0`` 时按 :func:`stat_scale` 拉回尺度 ✓；取不到尺度 ⇒ ``rescaled: false`` ✓ + ``note`` ✓。
    """
    if negative is None or abs(float(scale) - 1.0) <= 1e-9:
        return positive, {"guided": False, "scale": float(scale), "rescaled": False}

    guided = negative + (positive - negative) * float(scale)
    report: dict[str, Any] = {"guided": True, "scale": float(scale), "rescaled": False}
    if float(rescale) > 0:
        positive_scale = stat_scale(positive)
        guided_scale = stat_scale(guided)
        if positive_scale and guided_scale and guided_scale > 1e-12:
            factor = (positive_scale / guided_scale) ** float(rescale)
            guided = guided * factor
            report["rescaled"] = True
            report["rescaleFactor"] = round(factor, 4)
        else:
            report["note"] = "后端张量取不到尺度 ⇒ 重标定**已跳过** ✗（不假装做过 ✓）"
    return guided, report
