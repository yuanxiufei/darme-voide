"""**长视频分段**（纯计划数学 ✓ 零依赖 ✓）—— 把"想要的总帧数"拆成模型能吃的若干段 ✓。

## 为什么需要它（不是可选项 ✗）

模型单次能生成的帧数**有上限** ✓（H3 的帧网格是 ``17k+5`` ✓ 见 :mod:`geometry` —— 网格本身就是
"离散可取长度"的事实 ✓）。想要更长 ⇒ 只能**分段生成再拼** ✓。

## 三条不变量（本模块的全部价值 ✓ 自检逐条钉住 ✓）

1. **长度合法**：每段帧数落在 ``grid·k + min_frames`` 上 ✓（否则模型要么拒绝要么静默改 ✗）；
2. **无缝覆盖**：相邻段**重叠** ``overlap_frames`` 帧 ✓，生成区间并集**恰好覆盖** ``[0, total)`` ✓；
3. **保留帧数守恒**：各段 ``keep`` 之和 **== 总帧数** ✓✓（重叠在拼接时**只留一份** ✓ ——
   这是拼接器的硬契约 ✓，自检直接断言守恒 ✓）。

## 与首帧条件的联动（"无缝"的关键 ✓）

第 k 段（k>0）的首帧**就是**上一段的最后一帧 ✓ ⇒ ``carryFrame`` 记下**全局帧号** ✓；
:func:`build_segment_requests` 把调用方给的**边界帧张量**落成 PNG ✓ 并塞进
``GenerationRequest.first_frame`` ✓ ⇒ 走 :mod:`torch_backend` 的**真 VAE 编码**首帧条件 ✓（已通 ✓）。

⚠️ 本模块**不生成画面** ✗、也**不拼接** ✗ —— 它只产**计划** ✓ 与**每段的输入** ✓（各司其职 ✓）。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import geometry

__all__ = ["SegmentError", "SegmentPlan", "build_segment_requests", "grid_length",
           "max_grid_length", "plan_segments"]


class SegmentError(ValueError):
    """分段参数不合法 / 覆盖不出来 ✓（**明确报错**并给出可调的量 ✓，不悄悄改时长 ✗）。"""


@dataclass(frozen=True)
class SegmentPlan:
    """一段的**全部事实** ✓（生成哪些帧 ✓、拼接保留哪些帧 ✓ 分得清清楚楚 ✓）。"""

    index: int
    #: 本段生成覆盖的全局帧区间 ``[startFrame, endFrame)`` ✓（``endFrame`` 可超总数 ⇒ 用 keep 裁 ✓）
    startFrame: int
    endFrame: int
    #: 本段**实际生成**的帧数 ✓（网格对齐 ✓）
    frames: int
    #: 首帧取自**全局**哪一帧 ✓（``None`` = 第一段，无首帧条件 ✓）
    carryFrame: int | None
    #: 拼接保留**本段内**的 ``[keepFrom, keepTo)`` ✓
    keepFrom: int
    keepTo: int

    @property
    def keptFrames(self) -> int:  # noqa: N802 —— 与前端 camelCase 对齐 ✓
        return max(0, int(self.keepTo) - int(self.keepFrom))

    @property
    def overlapTail(self) -> int:  # noqa: N802
        """本段尾部被下一段重叠掉、拼接时要丢的帧数 ✓。"""
        return max(0, int(self.frames) - int(self.keepTo))

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "startFrame": self.startFrame, "endFrame": self.endFrame,
            "frames": self.frames, "carryFrame": self.carryFrame,
            "keepFrom": self.keepFrom, "keepTo": self.keepTo,
            "keptFrames": self.keptFrames, "overlapTail": self.overlapTail,
        }


def grid_length(value: int, *, grid: int = geometry.H3_FRAME_GRID,
                min_frames: int = geometry.H3_MIN_FRAMES) -> int:
    """把长度**向上**吸附到 ``grid·k + min_frames`` ✓（与 `geometry.snap_frames` 同一套网格 ✓）。"""
    value = max(int(min_frames), int(value))
    steps = -(-(value - int(min_frames)) // max(1, int(grid)))
    return int(min_frames) + steps * max(1, int(grid))


def max_grid_length(limit: int, *, grid: int = geometry.H3_FRAME_GRID,
                    min_frames: int = geometry.H3_MIN_FRAMES) -> int:
    """**不超过** ``limit`` 的最大合法长度 ✓（**向下**取 ✓ —— 宁可少生成也别越界 ✗）。"""
    if int(limit) < int(min_frames):
        raise SegmentError(
            f"单段上限 {limit} 小于最小合法帧数 {min_frames} ✗（模型吃不下这么短 ✓）")
    steps = max(0, (int(limit) - int(min_frames)) // max(1, int(grid)))
    return int(min_frames) + steps * max(1, int(grid))


def plan_segments(total_frames: int, *, max_frames: int, overlap_frames: int = 0,
                  grid: int = geometry.H3_FRAME_GRID,
                  min_frames: int = geometry.H3_MIN_FRAMES) -> list[SegmentPlan]:
    """总帧数 → 分段计划 ✓（三条不变量保证 ✓；覆盖不出来就**报错并说清可调的量** ✓）。"""
    total = int(total_frames)
    if total <= 0:
        raise SegmentError(f"总帧数必须为正（收到 {total_frames} ✗）")
    if total < int(min_frames):
        raise SegmentError(f"总帧数 {total} 小于最小合法帧数 {min_frames} ✗")
    overlap = int(overlap_frames)
    if overlap < 0:
        raise SegmentError(f"重叠帧数不能为负（收到 {overlap_frames} ✗）")
    per_segment = max_grid_length(int(max_frames), grid=grid, min_frames=min_frames)
    if overlap >= per_segment:
        raise SegmentError(
            f"重叠 {overlap} 帧 ≥ 单段长度 {per_segment} 帧 ✗ ⇒ 分段会**原地打转**（步进 ≤ 0 ✗）")

    plans: list[SegmentPlan] = []
    start = 0
    index = 0
    while start < total:
        need = total - start
        # 多要 overlap 帧 ✓：那是给**下一段的接缝**用的 ✓（所以本段 keep 时要把它们交出去 ✓）；
        # 但不超过单段上限 ✓。
        want = min(need + (overlap if need > overlap else 0), per_segment)
        length = min(per_segment, grid_length(want, grid=grid, min_frames=min_frames))
        is_final = start + length >= total
        # ⚠️ **非末段**必须把重叠的那 ``overlap`` 帧**交出去**（下一段会重新生成它们 ✓）
        #    —— 初版一律 `keep_to = min(length, total - start)` ✗ ⇒ 每段都多留一截 ⇒
        #    守恒断言当场炸（保留 592 ≠ 总 496 ✓）。这条正是"守恒是硬契约"的价值 ✓。
        keep_to = min(length, total - start) if is_final else max(1, length - overlap)
        plans.append(SegmentPlan(index=index, startFrame=start, endFrame=start + length,
                                 frames=length,
                                 carryFrame=(start - 1) if index else None,
                                 keepFrom=0, keepTo=max(1, keep_to)))
        if is_final:
            break
        start += length - overlap
        index += 1

    # ⚠️ 末段太短 ⇒ 覆盖不出来 ✓：**报错并给出可调的量** ✓（不悄悄改时长 ✗、也不生成非法长度 ✗）
    last = plans[-1]
    if last.keptFrames < int(min_frames) and len(plans) > 1:
        raise SegmentError(
            f"末段只剩 {last.keptFrames} 帧（最小合法 {min_frames} ✓）⇒ 覆盖不出来 ✗："
            f"把总帧数调到 {last.startFrame + min_frames} 的邻近合法值 ✓、或调大/调小 overlap "
            f"（当前 {overlap} ✓）、或调大 max_frames（当前 {per_segment} ✓）✓")

    # ⚠️ **保留帧数守恒** ✓（这条是拼接器的硬契约 ✓）：不满足就报错，**不**去偷偷改数字 ✗
    kept = sum(plan.keptFrames for plan in plans)
    if kept != total:
        raise SegmentError(
            f"保留帧数 {kept} ≠ 总帧数 {total} ✗ ⇒ 分段参数（max_frames={per_segment}、"
            f"overlap={overlap}）与总长不兼容 ✓ 请调整 ✓")
    return plans


def build_segment_requests(request: Any, plans: list[SegmentPlan], *,
                           root: str | Path,
                           carryFrames: dict[int, Any] | None = None,
                           carryLatents: dict[int, Any] | None = None) -> list[Any]:
    """分段计划 → **每段一个** ``GenerationRequest`` ✓（第 k>0 段自动带**首帧** ✓）。

    ``carryFrames[段号]`` = 调用方从**上一段产物**里取出的**边界帧张量** ✓
    （形状 ``(3,H,W)`` 或 ``(1,3,T,H,W)`` ✓ —— 与 :func:`media.write_image` 同口径 ✓）：
    本函数把它落成 PNG ✓ 再塞进 ``first_frame`` ✓（于是后端的**真 VAE 编码**首帧条件就生效 ✓）。

    ⚠️ 没给 ``carryFrames[k]`` ⇒ 该段**不带首帧** ✓，并记进 ``reference_frames`` 之外的
    ``_segmentNotes`` ✓（不假装有 ✓）。**本函数不生成画面** ✗。

    ``carryLatents[段号]`` = **续拍**的头部锚 ✓（上一段的**尾部潜变量** ✓ 2026-09-26 ✓）——
    ⚠️ 与 ``carryFrames`` **不是一回事** ✗：那个是**一帧像素**（要过 VAE 一趟 ✓ 只压住颜色 ✓），
    这个是**整段尾巴的潜变量**（逐位 ✓ 见 :meth:`torch_backend.init_dual_latents` ✓）。
    本函数只做**交接** ✓：用 ``object.__setattr__`` **旁挂**成 ``carry_latents`` ✓
    —— 与 ``_with_note`` 同一个理由 ✓（``GenerationRequest`` 加字段会牵动序列化 ✓✗）。
    ⚠️ 两套锚**同时给同一段** ⇒ 本函数**报错** ✗（一个说"接在这一帧后面"✓、一个说"开头这段照抄"✓
    ⇒ 语义重叠而口径不同 ✓ 叠着用只会让人分不清是哪条在起作用 ✓✗）。
    """
    from . import media as media_mod  # noqa: PLC0415

    where = Path(root)
    where.mkdir(parents=True, exist_ok=True)
    carry = carryFrames or {}
    latents = carryLatents or {}
    built: list[Any] = []
    for plan in plans:
        seconds = plan.frames / max(1, int(getattr(request, "fps", 24) or 24))
        first_frame: str | None = None
        note = ""
        frame = carry.get(plan.index)
        pinned = latents.get(plan.index)
        if pinned is not None and frame is not None:
            raise ValueError(
                f"第 {plan.index} 段同时给了 ``carryFrames``（单帧锚 ✓）与 ``carryLatents``"
                f"（尾部潜变量锚 ✓）✗ —— 续拍与拼接是**两套锚** ✓（各管一段连续性的写法不同 ✓）"
                "⇒ 一次只用一套 ✓✗")
        if plan.carryFrame is not None and frame is not None:
            target = where / f"carry_{plan.index:03d}_from{plan.carryFrame}.png"
            media_mod.write_image(_as_frame_batch(frame), target)
            first_frame = str(target)
            note = f"首帧取自全局第 {plan.carryFrame} 帧 ✓（落成 {target.name} ✓）"
        elif plan.carryFrame is not None:
            note = (f"未提供第 {plan.carryFrame} 帧的张量 ⇒ 本段**不带首帧** ✓"
                    f"（接缝连续性会变差 ✗ —— 如实标注，不假装有 ✓）")
        updated = replace(request, seconds=seconds, first_frame=first_frame)
        if pinned is not None:
            # ⚠️ 必须在 ``replace`` **之后**旁挂 ✗：``replace`` 造的是**新对象** ✓
            #    ⇒ 挂在旧对象上的属性会**悄悄丢掉** ✓✗（同一个坑 `_with_note` 也踩得到 ✓）。
            object.__setattr__(updated, "carry_latents", pinned)
            note = (note + " | " if note else "") + _carry_latents_note(pinned)
        built.append(updated if not note else _with_note(updated, plan, note))
    return built


def _carry_latents_note(latents: Any) -> str:
    """续拍交接的一句话事实 ✓（**如实**报出钉了多少 ✓ —— 报不出来就说"形状未知" ✓ 不编 ✗）。"""
    shape_v = tuple(getattr(latents.get("video") if isinstance(latents, dict) else None,
                            "shape", ()) or ())
    shape_a = tuple(getattr(latents.get("audio") if isinstance(latents, dict) else None,
                            "shape", ()) or ())
    frames = int((latents or {}).get("frames") or 0) if isinstance(latents, dict) else 0
    return (f"续拍头部锚 = 上一段尾部潜变量 ✓（视频槽 {shape_v[1] if len(shape_v) == 4 else '?'} ✓ / "
            f"音频槽 {shape_a[2] if len(shape_a) == 3 else '?'} ✓ / 覆盖 {frames} 像素帧 ✓）")


def _as_frame_batch(frame: Any) -> Any:
    """单帧/短序列 → ``(B,3,T,H,W)`` ✓（``write_image`` 只认这个口径 ✓ —— 初版没归一化 ✗
    ⇒ 传 ``(3,H,W)`` 直接报"形状应为 (B,3,T,H,W)" ✓，自检当场红 ✓）。
    """
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) == 3:                     # (3,H,W) ⇒ (1,3,1,H,W) ✓
        return frame.unsqueeze(0).unsqueeze(2)
    if len(shape) == 4:                     # (3,T,H,W) ⇒ (1,3,T,H,W) ✓
        return frame.unsqueeze(0)
    return frame


def _with_note(request: Any, plan: SegmentPlan, note: str) -> Any:
    """把分段说明挂到请求上 ✓（``dataclass(frozen=True)`` ⇒ 用 ``object.__setattr__`` ✓）。

    ``GenerationRequest`` 里加字段会牵动序列化 ✓ ⇒ 这里用**旁挂属性** ✓，
    并保持"没有就不存在"的语义 ✓（调用方可自行 ``getattr`` ✓）。
    """
    object.__setattr__(request, "segmentNote", {"index": plan.index, "note": note,
                                                "plan": plan.to_dict()})
    return request
