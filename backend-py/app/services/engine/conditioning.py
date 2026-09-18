"""**条件生成**（图生视频的"首帧/参考图"那一半 ✓）—— 零依赖 ✓ 只算**掩码**。

## 分工（这是本模块唯一的设计决定 ✓）

图生视频要「首帧来自图片、后面交给模型」✗ —— 但**"怎么把图片潜变量和噪声拼起来"是实现相关的**
✗：有的在通道维拼接 ✓、有的在时间维替换 ✓、有的走 mask 输入 ✓，形状布局各家不同 ✗。
⇒ **引擎算掩码（与模型无关 ✓），后端把掩码套到自己的布局上** ✓。
引擎若去猜布局，只会在真机上错得莫名其妙 ✗（与 :mod:`app.services.engine.geometry`
「不猜时间压缩比」是同一条纪律 ✓）。

## 掩码语义（纯数学 ✓ 公开做法 ✓）

设潜空间帧数 ``L``（由 ``(帧数-1)//压缩比 + 1`` 得到 ✓ 见 :func:`geometry.latent_frames` ✓），
每帧一个权重 ✓：

```
权重 = strength ×  clamp(1 − (i − keep + 1)/fade)
```

* ``keep`` 帧完全来自首帧（权重 1 ✓，默认 1 ✓）；
* 随后 ``fade`` 帧线性过渡到 0 ✓（``fade = 0`` ⇒ 硬切 ✓）；
* 其余为 0 ✓（纯模型生成 ✓）。

⚠️ **不猜的部分**：``keep``/``fade``/``strength`` 的**好取值取决于模型与权重** ✗ ⇒
本模块只提供默认值（``keep=1, fade=0, strength=1`` ✓ = 标准首帧条件 ✓），
别的一律由调用方显式给 ✓。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ConditioningConfig", "first_frame_mask", "latent_frames_for"]


@dataclass(frozen=True)
class ConditioningConfig:
    """首帧条件配置（不可变 ✓）。"""

    #: 完全取自首帧的**潜帧数** ✓（默认 1 ✓ —— 标准首帧条件 ✓）
    keep: int = 1
    #: 过渡潜帧数 ✓（``0`` = 硬切 ✓）
    fade: int = 0
    #: 整体强度 ✓（``0..1``；``1`` = 首帧完全生效 ✓）
    strength: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"keep": self.keep, "fade": self.fade, "strength": self.strength}


class ConditioningError(ValueError):
    """条件无法构造 ✓（**明确报错**比悄悄退化成"文生视频"好得多 ✗ —— 见 :func:`latent_frames_for` ✓）。"""


def latent_frames_for(frames: int, temporal_compression: int | None) -> int:
    """帧数 → 潜帧数 ✓；**没给压缩比就报错** ✗（不猜 ✓ —— 猜错会让首帧落在错误的帧上 ✗）。"""
    if not temporal_compression:
        raise ConditioningError(
            "首帧条件需要 temporalCompression（潜空间时间压缩比 ✓）—— 本引擎**不猜**这个值 ✗："
            "猜错会把首帧放到错误的潜帧上 ✗。请在请求里显式给出 ✓。")
    from . import geometry

    return geometry.latent_frames(int(frames), temporal_compression=int(temporal_compression))


def first_frame_mask(latent_frames: int, config: ConditioningConfig | None = None) -> list[float]:
    """算出**每个潜帧的权重** ✓（长度 = ``latent_frames`` ✓；越界取值会**报错**而不是截断 ✗）。"""
    config = config or ConditioningConfig()
    total = max(1, int(latent_frames))
    keep = int(config.keep)
    fade = max(0, int(config.fade))
    strength = float(config.strength)
    if keep < 0 or fade < 0:
        raise ConditioningError(f"keep/fade 不能为负（收到 keep={keep}, fade={fade} ✗）")
    if not 0.0 <= strength <= 1.0:
        raise ConditioningError(f"strength 必须落在 0..1（收到 {strength} ✗）")
    if keep > total:
        raise ConditioningError(
            f"keep={keep} 超过潜帧数 {total} ✗ ⇒ 这等于「整段都给首帧」✗，几乎肯定是参数写错了 ✓")
    if keep + fade > total:
        raise ConditioningError(
            f"keep+fade = {keep + fade} 超过潜帧数 {total} ✗ ⇒ 过渡区放不下 ✓")

    mask: list[float] = []
    for index in range(total):
        if index < keep:
            weight = 1.0
        elif fade and index < keep + fade:
            # 第 keep 帧起线性淡出到 0 ✓（在 keep+fade 处正好为 0 ✓）
            weight = max(0.0, 1.0 - (index - keep + 1) / fade)
        else:
            weight = 0.0
        mask.append(round(weight * strength, 6))
    return mask
