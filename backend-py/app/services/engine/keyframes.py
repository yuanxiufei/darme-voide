"""关键帧条件（首帧 / 尾帧）的**下标规划** ✓ —— 纯函数 ✓、零依赖 ✓。

## 为什么单独一个模块

首帧与尾帧走的是**同一套** `cond` 段机制 ✓（见 `h3_form.init_dual_latents` 里
``cond_t = cursor + FRAME_RESCALE × resolved_frame_index`` ✓），唯一区别就是**下标**：

* 首帧 ⇒ ``0`` ✓；
* 尾帧 ⇒ ``frames − 1`` ✓（参考实现的 fl2v 也是这么落的 ✓）。

下标算错**不会报错** ✗：把尾帧记成 0，产物会变成「开头是尾帧、末端随便」✓✗ —— 只能靠人眼
逐帧比对才发现 ✓。所以这段数学必须**离开 torch 单独可验** ✓（本仓纪律：算法先有零依赖自检 ✓）。

## 不静默的两条

* 给了尾帧但 ``frames < 2`` ⇒ **报错** ✗：只有 1 帧时首尾是同一个位置 ✓✗，悄悄退化成首帧
  就等于丢了尾帧 ✓。
* 两个都没给 ⇒ 空列表 ✓（正常 ✓，纯文本 / 单图以外的情形 ✓）。
"""
from __future__ import annotations

__all__ = ["KeyframePlanError", "plan_keyframes"]


class KeyframePlanError(ValueError):
    """关键帧参数本身不成立 ✗（不是后端能力问题 ✓ —— 那种由调用方另报 ✓）。"""


def plan_keyframes(frames: int, first: str | None, last: str | None) -> list[tuple[int, str]]:
    """``(总帧数, 首帧图, 尾帧图)`` ⇒ ``[(下标, 图路径), …]`` ✓（**按下标升序** ✓）。

    顺序很重要 ✓：下游按顺序把各块 ``cond`` 段拼进同一条时间轴 ✓，乱序会让段与段重叠 ✓✗。
    """
    total = int(frames)
    if total < 1:
        raise KeyframePlanError(f"总帧数必须 ≥ 1 ✗（收到 {frames} ✓）")
    wanted: list[tuple[int, str]] = []
    first_path = str(first) if first else ""
    last_path = str(last) if last else ""
    if first_path:
        wanted.append((0, first_path))
    if last_path:
        if total < 2:
            raise KeyframePlanError(
                f"尾帧条件至少要 2 帧 ✗（本次 frames={total} ✓）—— "
                f"只有 1 帧时首尾是**同一个位置** ✓✗，会悄悄退化成首帧 ✗")
        wanted.append((total - 1, last_path))
    wanted.sort(key=lambda item: item[0])
    return wanted
