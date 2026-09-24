"""**混合加载：fl2va 基底 + ref2va 的 adaLN 覆盖层** ✓（2026-09-24 补 ✓，口径来自逆向 ✓ 零依赖 ✓）。

## 它是什么（口径 ✓）
H3 的两个扩散权重（``fl2va`` / ``ref2va`` ✓）**主要差在 adaLN** ✓ ⇒ 一台机器可以拿
「**基底 + 覆盖层**」拼出**一个**两用模型 ✓（参考实现就是这么干的 ✓ —— 它还带缓存、磁盘余量检查
与 fp8 dtype 选项 ✓）。本模块只做**计划层** ✓：给定两边的**张量名与形状** ⇒ 决定
**取哪些 / 跳哪些 / 为什么** ✓（真正拼张量、落盘、缓存校验在加载层 ✓ —— ⚠️ 还没做 ✗，见模块尾 ✓）。

## 判据（每条都要能说清 ✓，不许静默 ✗✗）
* ``overlay`` 里**认成 adaLN 的键** ⇒ 必须**在 base 里也有** ✓（没有 ⇒ **拒** ✗✗ —— 说明两边不是同族 ✓）；
* 认出来的键**形状必须一致** ✓（不一致 ⇒ **拒** ✗ —— 硬塞只会换来形状错 ✓✗）；
* ``overlay`` 里**没认成 adaLN** 的键 ⇒ **跳过并给理由** ✓（不是静默忽略 ✗✗）；
* ⚠️ **匹配器可传** ✓：adaLN 的**具体命名本仓只核到「名字里有 adaln」这一层** ✗ ⇒ 默认匹配器
  就是这个 ✓，但调用方可以换 ✓（⚠️ 真权重到手后第一件事就是拿它核一遍 ✓）。

## 不猜
* 不解释 ``include_final_adaln`` 的**模型语义** ✗ —— 只按「名字里同时有 final 与 adaln」筛 ✓
  （口径：参考实现有这么个开关 ✓，默认**不带** final ✓）；
* 不做**张量拷贝 / 落盘 / 缓存** ✗（那是加载层 ✓）。⚠️ 而缓存**必须校验** ✓：截断 / 偏移错 /
  非法编码一律拒 ✓（这条已列进加载层待办 ✓）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

__all__ = ["DEFAULT_ADALN_HINT", "MergePlan", "apply_merge", "is_final_adaln", "matches_adaln",
           "plan_hybrid_merge"]

#: 默认的 adaLN 线索词 ✓（⚠️ 只到「名字里有它」这一层 ✓ —— 更细的命名**未核** ✗）。
DEFAULT_ADALN_HINT = "adaln"

#: 匹配器签名 ✓：``(张量名) -> 是不是 adaLN`` ✓（调用方可以换成自己那套 ✓）。
Matcher = Callable[[str], bool]


class HybridMergeError(ValueError):
    """两边**不是同族 / 形状对不上** ✓ ⇒ 当场拒 ✗（硬拼只会换来形状错 ✓✗）。"""


def is_final_adaln(name: str) -> bool:
    """名字里**同时**有 ``final`` 与 adaLN 线索 ✓（口径：参考实现那个开关就是这么分的 ✓）。"""
    lowered = str(name or "").lower()
    return "final" in lowered and DEFAULT_ADALN_HINT in lowered


def matches_adaln(name: str, *, hint: str = DEFAULT_ADALN_HINT,
                  matcher: Matcher | None = None) -> bool:
    """这个张量名算不算 adaLN ✓（``matcher`` 给了就用它 ✓ —— 覆盖默认线索 ✓）。"""
    if matcher is not None:
        return bool(matcher(str(name)))
    return str(hint or "").lower() in str(name or "").lower()


@dataclass(frozen=True)
class MergePlan:
    """合并计划 ✓：``take`` = 从覆盖层取的键 ✓；``skip`` = 跳过的键 + **理由** ✓（不许静默 ✗✗）。"""

    take: tuple[str, ...]
    skip: tuple[tuple[str, str], ...]
    base_count: int = 0
    overlay_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"take": list(self.take), "takeCount": len(self.take),
                "skip": [{"name": name, "reason": reason} for name, reason in self.skip],
                "skipCount": len(self.skip), "baseCount": self.base_count,
                "overlayCount": self.overlay_count}


def _shapes(items: Mapping[str, Sequence[int]]) -> dict[str, tuple[int, ...]]:
    return {str(name): tuple(int(value) for value in shape) for name, shape in items.items()}


def plan_hybrid_merge(base: Mapping[str, Sequence[int]], overlay: Mapping[str, Sequence[int]],
                      *, matcher: Matcher | None = None, include_final_adaln: bool = False,
                      hint: str = DEFAULT_ADALN_HINT) -> MergePlan:
    """计划「覆盖层哪些键该盖到基底上」 ✓ ⇒ :class:`MergePlan` ✓。

    ⚠️ 四类情况**分得很清** ✓：能盖 ✓ / **不是 adaLN 所以跳过**（给理由 ✓）/ **基底里没有**
    （**拒** ✗✗）/ **形状不一致**（**拒** ✗✗）。
    """
    base_shapes = _shapes(base)
    overlay_shapes = _shapes(overlay)
    take: list[str] = []
    skip: list[tuple[str, str]] = []
    for name in sorted(overlay_shapes):
        if not matches_adaln(name, hint=hint, matcher=matcher):
            skip.append((name, f"不是 adaLN（线索词 {hint!r} ✓）⇒ 覆盖层只负责 adaLN ✓"))
            continue
        if is_final_adaln(name) and not include_final_adaln:
            skip.append((name, "final 层的 adaLN ✓ ⇒ 本次没开 ``include_final_adaln`` ✓"))
            continue
        if name not in base_shapes:
            raise HybridMergeError(
                f"覆盖层有 ``{name}`` ✗，而基底里**没有这个键** ✓ ⇒ 两边**不是同族权重** ✗✗"
                f"（基底共 {len(base_shapes)} 个键 ✓，比如 `fl2va` 拼 `ref2va` ✓ —— "
                f"⚠️ 别把别的模型/别的版本混进来 ✓）")
        if base_shapes[name] != overlay_shapes[name]:
            raise HybridMergeError(
                f"``{name}`` 的形状对不上 ✗：基底 {base_shapes[name]} ✓、覆盖层 "
                f"{overlay_shapes[name]} ✓ ⇒ **拒** ✗（硬塞只会换来形状错 ✓✗）")
        take.append(name)
    return MergePlan(take=tuple(take), skip=tuple(skip), base_count=len(base_shapes),
                     overlay_count=len(overlay_shapes))


def apply_merge(base: Mapping[str, Any], overlay: Mapping[str, Any],
                plan: MergePlan) -> dict[str, Any]:
    """按计划合并**映射** ✓ ⇒ 新 dict ✓（⚠️ **不改原映射** ✗ —— 纯函数 ✓）。

    ``base`` / ``overlay`` 的值可以是形状、也可以是真张量 ✓（本模块只搬运 ✓ 不看内容 ✗）。
    """
    merged = dict(base)
    missing = [name for name in plan.take if name not in overlay]
    if missing:
        raise HybridMergeError(
            f"计划里要取 {missing[0]!r} ✗，但覆盖层里没有这个键 ✓ ⇒ 计划与数据**对不上** ✗"
            f"（别拿过期的 MergePlan 来合并 ✓✗）")
    for name in plan.take:
        merged[name] = overlay[name]
    return merged
