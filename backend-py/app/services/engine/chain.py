"""**多段成片**（分段生成 → 拼接 → 链式规范化 ✓）—— 把 :mod:`segments` 的计划兑现成**一条片子** ✓。

## 为什么要有这一层（缺口是查过的 ✓ 不是猜的 ✗）

* :mod:`segments` 只产**计划**与**每段的输入** ✓（模块头自己写着「本模块**不拼接** ✗」✓）；
* ``media.write_video`` 对 ``batch != 1`` **明确拒绝** ✓，并注明「多段拼接见后续计划 ✗」✓。

⇒ 中间那层（**拼接 + 接缝治理**）此前是空的 ✗ —— 本模块就是它 ✓。
没有它，``seconds`` 超过单段上限就只能一路放大到显存爆掉 ✗（见 :func:`plan_chain` ✓）。

## 事实来源（不猜 ✗）

* **单段原生上限 = 362 帧**：``ComfyUI-H3-Multishot`` 的 ``README.md`` 那条 verified recipe
  ``1280x736 / 362 / 14`` ✓ —— 而 ``362 = 17×21 + 5`` ✓ 正好落在 H3 的帧网格上 ✓
  （网格见 :mod:`geometry` ✓）：即 ``362/24 ≈ 15.08s`` ✓ 与它「每段约 10-15 秒」一致 ✓。
* **链式规范化的机制与数值口径**：``ComfyUI-H3-Multishot`` 的 ``h3_chain_normalize.py`` 的
  ``H3ChainNormalize.run``（**MIT ✓ Copyright (c) 2026 RiftCast ✓**）——
  * 每帧**直方图匹配**到首段的参考帧 ⇒ 治**色彩/曝光漂移** ✓；
  * 修正只加在 **~5px 减 ~17px 的结构带**上 ⇒ 颗粒/噪声原样通过 ✓（整帧模糊同样能压漂移，
    但会吃掉约两成细部 ✓ —— 那是"摄像机质感"✓）；
  * 修正量走 **EMA 渐入** ⇒ 不在段边界"跳一下" ✓；
  * 基线取**首段曝光淡入之后**的中位数 ✓（H3 开头约 1.7s 是淡入 ✓ 算进去会把基线拖暗
    ⇒ 整片被误软化 ✓）；
  * **deadband** ⇒ 首段不会拿自己的均值削自己 ✓。
  **实现按本仓风格重写** ✓（纯函数 ✓ 显式校验 ✓ 报告不静默 ✓）。
  ⚠️ 它同仓的 ``writer_pack/`` 是**另一套许可**（仅学术/非商用 ✗）⇒ **本模块一行都没碰它** ✓。

## 与 :mod:`segments` 的硬契约：保留帧数守恒 ✓

``plan_segments`` 已保证「各段 ``keep`` 之和 == 总帧数」✓ —— 但上游断言过不是下游免检的理由 ✗：
:func:`stitch_frames` 拼完**再断言一次** ✓（不符就报错并说清差在哪 ✓，绝不静默补/裁 ✗）。
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from . import geometry, media as media_mod, segments as segments_mod

__all__ = ["ChainError", "ChainPlan", "DEFAULT_OVERLAP_FRAMES", "H3_NATIVE_MAX_FRAMES",
           "NORM_DEFAULTS", "assemble_chain", "needs_chain", "normalize_chain",
           "plan_chain", "run_chain", "stitch_frames"]

#: 单段原生上限（帧 ✓）—— ``362 = 17×21 + 5`` ✓ 见模块头的事实来源 ✓
#: （``max_grid_length(362) == 362`` ✓ 正好是网格点 ✓ ⇒ 这个数不用再往网格上吸 ✓）
H3_NATIVE_MAX_FRAMES = 362

#: 相邻段的重叠帧数（默认 ✓）—— 只重叠 **1 帧** ✓：那正是下一段要当首帧的**锚点** ✓
#: （``segments`` 的口径：重叠帧由下一段**重新生成**再丢掉 ✓ ⇒ 重叠越多越费 ✓ ——
#: 链式续接要的也只是"上一段的最后一帧" ✓，1 帧足够 ✓）。
DEFAULT_OVERLAP_FRAMES = 1

#: 链式规范化的默认口径 ✓（**照抄参考实现的实测值** ✓ 不自己调 ✓ 见模块头 ✓）
NORM_DEFAULTS: dict[str, Any] = {
    "baseline_seconds": 10.0,
    "skip_seconds": 2.0,
    "strength": 1.0,
    "deadband": 1.06,
    "ema": 0.10,
    "colour_match": True,
}


class ChainError(ValueError):
    """多段成片的组合不成立 ✓（**明确报错**并说清是第几段 / 差多少 ✓，不静默兜底 ✗）。"""


@dataclass(frozen=True)
class ChainPlan:
    """一次成片的**全部计划事实** ✓（总帧数 ✓ 分几段 ✓ 每段怎么裁 ✓ 一眼看全 ✓）。"""

    totalFrames: int
    fps: int
    maxFrames: int
    overlapFrames: int
    segments: tuple[Any, ...]

    @property
    def segmentCount(self) -> int:  # noqa: N802 —— 与前端 camelCase 对齐 ✓
        return len(self.segments)

    @property
    def seconds(self) -> float:
        return self.totalFrames / max(1, int(self.fps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "totalFrames": self.totalFrames, "fps": self.fps,
            "seconds": round(self.seconds, 4),
            "maxFrames": self.maxFrames, "overlapFrames": self.overlapFrames,
            "segmentCount": self.segmentCount,
            "segments": [plan.to_dict() for plan in self.segments],
        }


# ══════════════════════════════════════════════════════════════════════════
# ① 计划：这个请求要不要分成几段
# ══════════════════════════════════════════════════════════════════════════
def plan_chain(request: Any, *, max_frames: int = H3_NATIVE_MAX_FRAMES,
               overlap_frames: int = DEFAULT_OVERLAP_FRAMES) -> ChainPlan | None:
    """请求 → 分段计划 ✓；**单段就够** ⇒ ``None`` ✓（调用方走原路 ✓ 行为一字不改 ✓）。

    ⚠️ 只对**视频**阶段（``stage="h3"``）分段 ✗：图片阶段（``stage="sdxl"`` ✓）只有一帧 ✓，
    "多段"在它那里没有意义 ✓（给了也只能忽略 ⇒ 那才是静默 ✗ ⇒ 这里明确回 ``None`` ✓）。

    ⚠️ 帧数与 :func:`app.services.engine.pipeline.build_plan` **同一套算法** ✓
    （:func:`geometry.snap_frames` ✓）—— 两处口径必须一致 ✗，否则会出现
    "计划说 124 帧、分段按 125 拆"这种**互相矛盾**的事实 ✓✗。
    """
    stage = str(getattr(request, "stage", "h3") or "h3")
    if stage != "h3":
        return None
    fps = int(getattr(request, "fps", geometry.H3_FPS) or geometry.H3_FPS)
    seconds = float(getattr(request, "seconds", 0.0) or 0.0)
    total = geometry.snap_frames(seconds, fps=fps)
    # ⚠️ 判据与 `plan_segments` 内部**同一口径**（网格下取 ✓）：`<= 上限` 才算单段 ✓
    if total <= segments_mod.max_grid_length(int(max_frames)):
        return None
    plans = segments_mod.plan_segments(total, max_frames=int(max_frames),
                                       overlap_frames=int(overlap_frames))
    return ChainPlan(totalFrames=total, fps=fps, maxFrames=int(max_frames),
                     overlapFrames=int(overlap_frames), segments=tuple(plans))


def needs_chain(request: Any, *, max_frames: int = H3_NATIVE_MAX_FRAMES,
                overlap_frames: int = DEFAULT_OVERLAP_FRAMES) -> bool:
    """这个请求要不要走多段 ✓（``runtime`` 的接线判据 ✓ —— 与 :func:`plan_chain` 同源 ✓）。"""
    return plan_chain(request, max_frames=max_frames,
                      overlap_frames=overlap_frames) is not None


# ══════════════════════════════════════════════════════════════════════════
# ② 拼接：按每段的 keep 裁好再首尾相接
# ══════════════════════════════════════════════════════════════════════════
def stitch_frames(parts: Sequence[Any], plans: Sequence[Any], *,
                  expected_frames: int | None = None) -> tuple[Any, dict[str, Any]]:
    """各段帧（``(1,3,T,H,W)`` ✓）→ **一条**序列 ✓ ⇒ ``(张量, 报告)`` ✓。

    三条把关（都不静默 ✗）：

    1. 段数与计划数**必须相等** ✓（少一段 = 产物短一截 ✓ 而用户看不出来 ✓✗）；
    2. 每段**解出的帧数 ≥ 它要保留的 ``keepTo``** ✓（少了就是解码没给够 ✓ 报错说清差几帧 ✓）；
    3. 拼完的帧数 **== 各段 keep 之和 == ``expected_frames``** ✓（守恒 ✓ 不符就报 ✓）。
    """
    import torch  # noqa: PLC0415 —— 张量层懒加载 ✓（见 engine/__init__ 的分层说明 ✓）

    plans = list(plans)
    parts = list(parts)
    if len(parts) != len(plans):
        raise ChainError(f"段数对不上 ✗：给了 {len(parts)} 段帧、计划是 {len(plans)} 段 ✓")
    if not plans:
        raise ChainError("没有段可以拼 ✗（计划为空 ✓）")

    kept: list[Any] = []
    detail: list[dict[str, Any]] = []
    for plan, part in zip(plans, parts):
        shape = tuple(getattr(part, "shape", ()))
        if len(shape) != 5 or int(shape[1]) != 3:
            raise ChainError(f"第 {plan.index} 段帧张量形状应为 (1,3,T,H,W) ✓，收到 {shape} ✗")
        if int(shape[2]) < int(plan.keepTo):
            raise ChainError(
                f"第 {plan.index} 段只解出 {int(shape[2])} 帧，而计划要保留到第 {plan.keepTo} 帧 ✗"
                f" ⇒ 短了 {int(plan.keepTo) - int(shape[2])} 帧 ✓（不静默补帧 ✗）")
        piece = part[:, :, int(plan.keepFrom):int(plan.keepTo)]
        kept.append(piece)
        detail.append({"index": plan.index, "from": int(plan.keepFrom), "to": int(plan.keepTo),
                       "frames": int(piece.shape[2])})
    sizes = {tuple(piece.shape[3:]) for piece in kept}
    if len(sizes) > 1:
        raise ChainError(f"各段画幅不一致 ✗：{sorted(sizes)} ✓ —— 拼不出一条片子 ✓")

    out = torch.cat(kept, dim=2)
    total = sum(int(plan.keptFrames) for plan in plans)
    if int(out.shape[2]) != total:
        raise ChainError(f"拼出来的帧数 {int(out.shape[2])} ≠ 各段 keep 之和 {total} ✗（内部矛盾 ✓）")
    if expected_frames is not None and int(out.shape[2]) != int(expected_frames):
        raise ChainError(
            f"拼出来的帧数 {int(out.shape[2])} ≠ 总帧数 {int(expected_frames)} ✗"
            f"（守恒是硬契约 ✓ —— 差 {int(out.shape[2]) - int(expected_frames)} 帧 ✓ 不静默补/裁 ✗）")
    report = {"frames": int(out.shape[2]), "segments": detail, "keptTotal": total,
              "expectedFrames": None if expected_frames is None else int(expected_frames),
              "conserved": expected_frames is None or total == int(expected_frames)}
    return out, report


# ══════════════════════════════════════════════════════════════════════════
# ③ 链式规范化：把跨段的**色彩/曝光漂移**与**结构能量爬升**拉平
# ══════════════════════════════════════════════════════════════════════════
def _torch() -> Any:
    """张量层懒加载 ✓（``import`` 本模块**不**要求装 torch ✓ —— 与 :mod:`pipeline` 同一口径 ✓）。"""
    import torch  # noqa: PLC0415

    return torch


def _box(x: Any, r: int) -> Any:
    """盒式模糊 ✓（``[B,H,W,C]`` ✓ 可分离累积和 ✓ 边缘 replicate ✓）—— 口径照抄参考实现 ✓。"""
    torch = _torch()
    pad = torch.nn.functional.pad(x.permute(0, 3, 1, 2), (r, r, r, r),
                                  mode="replicate").permute(0, 2, 3, 1)
    cs = pad.cumsum(1)
    cs = torch.cat([torch.zeros_like(cs[:, :1]), cs], 1)
    x = (cs[:, 2 * r + 1:] - cs[:, :-2 * r - 1]) / float(2 * r + 1)
    cs = x.cumsum(2)
    cs = torch.cat([torch.zeros_like(cs[:, :, :1]), cs], 2)
    return (cs[:, :, 2 * r + 1:] - cs[:, :, :-2 * r - 1]) / float(2 * r + 1)


def _norm_lap(gray: Any) -> float:
    """单帧 ``[H,W]`` 的**对比度归一化** Laplacian 能量 ✓（照抄参考实现 ✓）。"""
    torch = _torch()
    k = (gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1]
         - gray[1:-1, :-2] - gray[1:-1, 2:])
    return float((k ** 2).mean() / torch.clamp(gray.std() ** 2, min=1e-9))


def _cdf(channel_u8: Any) -> Any:
    """单通道 uint8 的**累积分布** ✓（256 级 ✓ 照抄参考实现 ✓）。"""
    torch = _torch()
    hist = torch.bincount(channel_u8.reshape(-1), minlength=256).float()
    return torch.cumsum(hist, 0) / torch.clamp(hist.sum(), min=1.0)


def _hist_match(frame: Any, ref_cdfs: list[Any]) -> Any:
    """把一帧 ``[H,W,3]``（0..1 ✓）**匹配**到首段参考帧的分布上 ✓（逐通道查表 ✓ 照抄 ✓）。"""
    torch = _torch()
    out = torch.empty_like(frame)
    idx = torch.arange(256, device=frame.device, dtype=torch.float32)
    for c in range(3):
        src = torch.clamp(frame[..., c] * 255.0, 0, 255).to(torch.uint8)
        s_cdf = _cdf(src)
        pos = torch.searchsorted(ref_cdfs[c].contiguous(), s_cdf.contiguous())
        lut = torch.clamp(pos.float(), 0, 255)
        lut = torch.where(torch.isfinite(lut), lut, idx)
        out[..., c] = lut[src.long()] / 255.0
    return out


def normalize_chain(frames: Any, *, fps: int, baseline_seconds: float = 10.0,
                    skip_seconds: float = 2.0, strength: float = 1.0, deadband: float = 1.06,
                    ema: float = 0.10, colour_match: bool = True, value_range: str = "-1..1",
                    segment_starts: Sequence[int] | None = None) -> tuple[Any, dict[str, Any]]:
    """跨段**拉平**（色彩/曝光 ✓ + 结构能量 ✓）⇒ ``(张量, 报告)`` ✓。

    默认值与 :data:`NORM_DEFAULTS` **必须一致** ✓（那是参考实现的实测值 ✓ —— 自检直接钉住 ✓）。

    ⚠️ 帧数 ``< 8`` ⇒ **原样返回** ✓ 并在报告里写清为什么没动 ✓（不静默 ✓ 参考实现同口径 ✓）。
    ⚠️ 值域：本仓帧张量是 ``-1..1`` ✓（见 :mod:`media` ✓），而机制本身量在 ``0..1`` 上 ✓
    ⇒ 这里自己换算 ✓ 并且 ``value_range`` 认不出就**报错** ✗（不猜 ✗）。
    ⚠️ 逐帧走 Python 循环 ✓（与参考实现一致 ✓）：几千帧量级没问题 ✓，
    真正超长的片子要提速的话，**换的是这一处** ✓。
    """
    torch = _torch()
    if int(fps) <= 0:
        raise ChainError(f"fps 必须为正（收到 {fps} ✗）—— 基线的秒→帧换算全指望它 ✓")
    if float(deadband) < 1.0:
        raise ChainError(f"deadband 必须 ≥ 1.0（收到 {deadband} ✗）—— 它是「不动」的比例 ✓")
    if not 0.0 < float(ema) <= 1.0:
        raise ChainError(f"ema 要落在 (0,1]（收到 {ema} ✗）—— 修正的渐入速度 ✓")
    if float(strength) < 0.0:
        raise ChainError(f"strength 不能为负（收到 {strength} ✗）")
    kind = str(value_range or "-1..1").replace(" ", "")
    if kind not in ("-1..1", "[-1,1]", "0..1", "[0,1]"):
        raise ChainError(f"未知值域 {value_range!r} ✓ 可用：-1..1 / 0..1 ✓")
    shape = tuple(frames.shape)
    if len(shape) != 5 or int(shape[1]) != 3:
        raise ChainError(f"帧张量形状应为 (B,3,T,H,W) ✓，收到 {shape} ✗")

    n = int(shape[2])
    report: dict[str, Any] = {
        "frames": n, "fps": int(fps), "valueRange": kind,
        "baselineSeconds": float(baseline_seconds), "skipSeconds": float(skip_seconds),
        "strength": float(strength), "deadband": float(deadband), "ema": float(ema),
        "colourMatch": bool(colour_match),
        #: 机制与数值口径的出处 ✓（只写上游项目+文件+符号 ✓ 见模块头 ✓）
        "source": "ComfyUI-H3-Multishot/h3_chain_normalize.py 的 H3ChainNormalize.run"
                  "（MIT / RiftCast 2026）—— 机制照抄，实现按本仓风格重写",
    }
    if n < 8:
        report.update({"applied": False,
                       "reason": f"只有 {n} 帧（< 8 ✓）⇒ 没什么可拉平的 ✓ 原样返回 ✓"})
        return frames, report

    with torch.no_grad():
        work = frames[0].permute(1, 2, 3, 0).to(torch.float32)  # (C,T,H,W) ⇒ (T,H,W,3) ✓
        if kind in ("-1..1", "[-1,1]"):
            work = (work + 1.0) * 0.5
        work = work.clamp(0.0, 1.0).contiguous()

        skip = min(n - 1, int(float(skip_seconds) * int(fps)))
        take = min(n, skip + max(8, int(float(baseline_seconds) * int(fps))))
        laps = sorted(_norm_lap(work[i].mean(-1)) for i in range(skip, take))
        baseline = laps[len(laps) // 2]                        # 中位数 ✓
        ref_idx = skip + (take - skip) // 2
        ref_cdfs = None
        if colour_match:
            ref_u8 = torch.clamp(work[ref_idx] * 255.0, 0, 255).to(torch.uint8)
            ref_cdfs = [_cdf(ref_u8[..., c]) for c in range(3)]

        out = torch.empty_like(work)
        sigma = 0.0
        applied: list[float] = []
        for i in range(n):
            frame = work[i]
            if ref_cdfs is not None:
                frame = _hist_match(frame, ref_cdfs)
            cur = _norm_lap(frame.mean(-1))
            target = max(0.0, (cur / max(baseline, 1e-9)) - float(deadband)) * float(strength)
            sigma = float(ema) * target + (1.0 - float(ema)) * sigma
            amount = min(0.85, sigma)                          # 上限照抄参考实现 ✓
            if amount > 0.02:
                batch = frame.unsqueeze(0)
                band = _box(batch, 2) - _box(batch, 8)         # ~5px 减 ~17px ✓ 只动结构带 ✓
                frame = torch.clamp(batch - amount * band, 0.0, 1.0).squeeze(0)
            applied.append(amount)
            out[i] = frame

        tail = applied[-int(2 * int(fps)):] or [0.0]
        report.update({
            "applied": True, "skipFrames": skip, "baselineFrames": [skip, take],
            "baseline": round(baseline, 6), "referenceFrame": ref_idx,
            "correctionMax": round(max(applied), 4),
            "correctionTailMean": round(sum(tail) / len(tail), 4),
            "correctionsBySegment": _segment_means(applied, segment_starts, n),
        })
        result = out.permute(3, 0, 1, 2).unsqueeze(0)          # (1,3,T,H,W) ✓
        if kind in ("-1..1", "[-1,1]"):
            result = result * 2.0 - 1.0
        return result.to(frames.dtype).contiguous(), report


def _segment_means(applied: Sequence[float], starts: Sequence[int] | None,
                   n: int) -> list[dict[str, Any]] | None:
    """每段的**平均修正量** ✓ —— 段边界"跳没跳"就靠它看 ✓（没给段起点就不报 ✓ 不编 ✗）。"""
    if not starts:
        return None
    bounds = [int(value) for value in starts] + [int(n)]
    rows: list[dict[str, Any]] = []
    for index in range(len(bounds) - 1):
        low, high = bounds[index], bounds[index + 1]
        chunk = list(applied[low:high]) or [0.0]
        rows.append({"start": low, "frames": high - low,
                     "meanCorrection": round(sum(chunk) / len(chunk), 4)})
    return rows


def assemble_chain(parts: Sequence[Any], plans: Sequence[Any], *, fps: int,
                   normalize: bool = True, expected_frames: int | None = None,
                   norm_kwargs: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """**拼接 + 规范化**一步到位 ✓ ⇒ ``(张量, 报告)`` ✓（报告里两段事实都在 ✓ 不合并掉 ✗）。"""
    stitched, stitch_report = stitch_frames(parts, plans, expected_frames=expected_frames)
    if not normalize:
        # ⭐ 明确关掉也要**说清** ✓（静默跳过 = 用户以为治过了 ✓✗）
        return stitched, {"stitch": stitch_report,
                          "normalize": {"applied": False, "reason": "调用方明确关掉了链式规范化 ✓"}}
    starts = [int(plan.startFrame) for plan in plans]
    kwargs = dict(norm_kwargs or {})
    normalized, norm_report = normalize_chain(stitched, fps=int(fps), segment_starts=starts,
                                              **kwargs)
    return normalized, {"stitch": stitch_report, "normalize": norm_report}


def _continuation_carry(decoded: Any, segment: Any, continuation: Any) -> dict[str, Any]:
    """上一段产物 → **续拍的头部锚**（尾部潜变量 ✓ 2026-09-26 补 ✓）。

    事实来源：上游 ``h3_extend`` / context_pin —— 把上一窗口的**尾段潜变量**钉进新窗口的**开头** ✓
    （**不用**解码后的帧 ✗：省一趟 VAE ✓ 且颜色与速度都逐位接得上 ✓）。

    ⚠️ 三条都要**当场**点破 ✗（否则就成了"看着像续拍、其实是拼接"✓✗）：
    1. 本段的重叠尾巴必须**正好**是钉长 ✗（不等 ⇒ 钉进去的内容与拼接时丢掉的帧**对不上** ✓）；
    2. ``decoded`` 必须带 ``latents`` ✗（拿不出来就**报错** ✓ —— **不退回**单帧锚 ✗，
       那会静默换成另一种产物 ✓✗）；
    3. 潜变量时间维要**够切** ✗（尾巴比成片还长 ⇒ 切不出来 ✓）。
    """
    if int(getattr(segment, "overlapTail", 0)) != int(continuation.pin_frames):
        raise ChainError(
            f"续拍的钉长与分段的**重叠尾巴**对不上 ✗：第 {segment.index} 段重叠 "
            f"{int(getattr(segment, 'overlapTail', 0))} 帧 ✓，而钉长是 {int(continuation.pin_frames)} 帧 ✓"
            "　—— 钉进下一段头部的那段**就是**「重叠再生」的那段 ✓ ⇒ 两者必须相等 ✓"
            "（``run_chain(overlap_frames=…)`` 与 ``h3_edit.continuation_pin(pin_frames=…)`` 同源 ✓）")
    latents = (decoded or {}).get("latents") if isinstance(decoded, dict) else None
    if not isinstance(latents, dict) or "video" not in latents or "audio" not in latents:
        keys = sorted(decoded.keys()) if isinstance(decoded, dict) else type(decoded).__name__
        raise ChainError(
            f"第 {segment.index} 段的 decode **没交回潜变量** ✗ ⇒ 续拍拿不到要钉的尾巴 ✓"
            f"（该段 decode 的键：{keys} ✓）—— 后端 `_decode_dual` 会顺手给 ``latents`` ✓"
            "；⚠️ 本函数**不会**退回单帧 PNG 锚 ✗（那是另一条产物 ✓✗）")
    video, audio = latents["video"], latents["audio"]
    shape_v = tuple(int(value) for value in getattr(video, "shape", ()) or ())
    shape_a = tuple(int(value) for value in getattr(audio, "shape", ()) or ())
    want_v, want_a = int(continuation.video_slots), int(continuation.audio_slots)
    if len(shape_v) != 4 or len(shape_a) != 3:
        raise ChainError(
            f"续拍要 ``[C,T,H,W]`` / ``[C,ch,Ta]`` 两条潜变量 ✗（收到 {shape_v} / {shape_a} ✓）")
    if shape_v[1] < want_v or shape_a[2] < want_a:
        raise ChainError(
            f"这一段的潜变量**不够切** ✗：要尾巴 {want_v} / {want_a} 槽 ✓，本段只有 "
            f"{shape_v[1]} / {shape_a[2]} 槽 ✓（钉长 {int(continuation.pin_frames)} 帧 "
            f"= {continuation.seconds:.3f} s ✓ 比成片还长 ✓ ⇒ 调小钉长 ✓ 或加长窗口 ✓）")
    return {"video": video[:, shape_v[1] - want_v:, :, :],
            "audio": audio[:, :, shape_a[2] - want_a:],
            "frames": int(continuation.pin_frames)}


# ══════════════════════════════════════════════════════════════════════════
# ④ 跑起来：逐段生成 → 拼接 → 规范化 → 落**一条**成片
# ══════════════════════════════════════════════════════════════════════════
def run_chain(request: Any, backend: Any, *, run_segment: Callable[..., Any],
              on_event: Callable[[dict[str, Any]], None] | None = None,
              cancel: Callable[[], bool] | None = None,
              max_frames: int = H3_NATIVE_MAX_FRAMES,
              overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
              normalize: bool = True, norm_kwargs: dict[str, Any] | None = None,
              frame_key: str = "frames",
              continuation: Any = None) -> Any:
    """多段跑完 ⇒ 返回 ``PipelineResult``（**同形** ✓ 调用方无感 ✓）。

    ⚠️ 阶段编排**不在这里重写** ✗：``run_segment`` 由调用方给（生产上是
    :func:`app.services.engine.pipeline.run_sync` ✓）—— 本模块只管**分段之间的编排** ✓
    （造下一段的请求 ✓ 传接缝锚点 ✓ 收帧 ✓ 拼 ✓ 治 ✓ 落 ✓）；也因此自检不必碰真权重 ✓。

    ⚠️ 每段**一个子目录** ✓（``…/segments/segNNN/``）：否则各段都写 ``video_seed{seed}.mp4`` ✓✗
    同目录下**互相覆盖** ✗ —— 只剩最后一段能核对 ✓（那正是"静默丢事实"✗）。

    ## ⭐ **续拍**（``continuation`` ✓ 2026-09-26 补 ✓）

    给 :class:`h3_edit.ContinuationPin` ⇒ 走**续拍** ✓：段与段之间交的是**尾部潜变量** ✓
    （下一段的头部锚 ✓ 逐位 ✓ 见 :meth:`torch_backend.init_dual_latents` ✓）而不是**一帧 PNG** ✓。
    事实来源：``reference/ComfyUI-H3-Multishot/h3_extend.py`` 的 context_pin 口径 ✓。

    ⚠️ 三条同步口径（不说清就会"看着像续拍、其实是拼接"✓✗）：
    1. ``overlap_frames`` 要**等于** ``continuation.pin_frames`` ✗ —— 钉进下一段头部的那段
       正是"重叠再生"的那段 ✓（两者不等 ⇒ 钉的内容与拼掉的内容对不上 ✓ ⇒ 本函数**报错** ✗）；
    2. 续拍模式下**不用** ``carryFrames``（单帧锚 ✓）✗：两套锚语义重叠 ✓（见
       :func:`segments.build_segment_requests` ✓）；
    3. 上一段必须交回 ``decoded["latents"]`` ✓（后端 `_decode_dual` 会顺手给 ✓）——
       给不出来就**报错** ✗，**不退回**单帧锚 ✗（那会静默换成另一种产物 ✓✗）。
    """
    from dataclasses import replace  # noqa: PLC0415

    from . import pipeline as pipeline_mod  # noqa: PLC0415 —— 局部导入：避免模块级互相 import ✓

    if continuation is not None:
        # ⚠️ 续拍的计划**只认** `h3_edit.ContinuationPin` ✓（钉多少槽是**算出来的** ✓）——
        #    真给个 True/"yes" 就只能猜 ✓ ⇒ 当场报错 ✓ 不猜 ✗。
        from . import h3_edit as h3_edit_mod  # noqa: PLC0415
        if not isinstance(continuation, h3_edit_mod.ContinuationPin):
            raise ChainError(
                "续拍要的是 :class:`h3_edit.ContinuationPin` ✓（由 "
                "`h3_edit.continuation_pin(pin_frames, temporal_compression=…, "
                "audio_latent_mode=…)` 算出来 ✓），收到 "
                f"{type(continuation).__name__} ✗ —— 钉多少**槽**取决于压缩比与音频潜帧率 ✓ "
                "⇒ 这里不猜 ✗")
    plan = plan_chain(request, max_frames=max_frames, overlap_frames=overlap_frames)
    if plan is None:
        raise ChainError("这个请求单段就够 ✓ ⇒ 不该走 run_chain ✓"
                         "（调用方先看 needs_chain ✓ —— 两处同源 ✓）")
    fps = int(plan.fps)
    root = Path(request.outputs_dir) if getattr(request, "outputs_dir", None) else Path(
        tempfile.mkdtemp(prefix="engine_chain_"))
    root.mkdir(parents=True, exist_ok=True)
    parts: list[Any] = []
    runs: list[dict[str, Any]] = []
    stage_ms: dict[str, int] = {}
    steps = 0
    guided = 0
    carry: dict[int, Any] = {}
    #: 续拍：下一段的**头部锚**（上一段尾部潜变量 ✓）—— 与 ``carry``（单帧锚 ✓）**互斥** ✓
    carry_latents: dict[int, Any] = {}

    def emit(payload: dict[str, Any]) -> None:
        if on_event:
            on_event(payload)

    def stopped() -> bool:
        return bool(cancel and cancel())

    for index, segment in enumerate(plan.segments):
        if stopped():
            return _stopped(pipeline_mod, backend, stage_ms, runs, segment.index)
        seg_dir = root / "segments" / f"seg{segment.index:03d}"
        base = replace(request, outputs_dir=str(seg_dir))
        # ⚠️ 续拍模式**不交**单帧锚 ✗（只交尾部潜变量 ✓）：两套锚语义重叠 ✓
        #    ⇒ `build_segment_requests` 同段同时收到两者会**报错** ✓（这里就不给它机会 ✓）。
        seg_request = segments_mod.build_segment_requests(
            base, [segment], root=seg_dir,
            carryFrames={} if continuation is not None else carry,
            carryLatents=carry_latents)[0]
        holder: list[Any] = []

        def grab(decoded: Any, _holder: list[Any] = holder) -> None:
            _holder.append(decoded)

        def relay(payload: dict[str, Any], _index: int = index,
                  _segment: Any = segment) -> None:
            # ⚠️ 段号进**结构化字段** ✓（不能只写在文案里 ✗ —— 前端要画"第 2/4 段" ✓）
            emit({**payload, "segment": int(_index), "segmentTotal": plan.segmentCount,
                  "segmentFrames": int(_segment.frames)})

        result = run_segment(seg_request, backend, on_event=relay, cancel=cancel,
                             on_decoded=grab)
        for key, value in dict(result.stageMs).items():
            # ⚠️ 各段耗时**逐项累加** ✓（取最后一段 = 少报 ✗）
            stage_ms[key] = stage_ms.get(key, 0) + int(value)
        steps += int(result.sampleSteps)
        guided += int(result.guidanceSteps)
        runs.append({"index": int(segment.index), "ok": bool(result.ok),
                     "cancelled": bool(result.cancelled),
                     "videoPath": (result.outputs or {}).get("videoPath"),
                     "error": result.error})
        if result.cancelled:
            return _stopped(pipeline_mod, backend, stage_ms, runs, segment.index)
        if not result.ok:
            # ⚠️ 一段失败 ⇒ **整条失败** ✓（绝不交一条"少一段"的片子 ✗✗ —— 用户看不出来 ✓）
            return _segment_failed(pipeline_mod, backend, result, stage_ms, runs, segment.index)
        decoded = holder[0] if holder else {}
        frames = (decoded or {}).get(frame_key)
        if not hasattr(frames, "shape") or len(tuple(frames.shape)) != 5:
            raise ChainError(
                f"第 {segment.index} 段没交回可用的帧张量 ✗（decoded[{frame_key!r}] = "
                f"{type(frames).__name__} ✓；该段 decode 的键：{sorted(decoded.keys())} ✓）"
                f"—— 拼接要的是**无损**张量 ✓（不回去读刚落的 mp4 ✗：那要多一次编解码 ✓）")
        parts.append(frames)
        if index + 1 < plan.segmentCount:
            nxt = plan.segments[index + 1]
            expected_at = int(segment.startFrame) + int(segment.keepTo) - 1
            if nxt.carryFrame != expected_at:
                raise ChainError(
                    f"接缝对不上 ✗：第 {nxt.index} 段的首帧取自全局第 {nxt.carryFrame} 帧 ✓，"
                    f"而第 {segment.index} 段保留到第 {expected_at} 帧 ✓（内部矛盾 ✓）")
            if continuation is not None:
                # ⭐ **续拍**：交**尾部潜变量** ✓（不是那一帧像素 ✗）
                carry_latents = {int(nxt.index): _continuation_carry(
                    decoded, frame_key, segment, continuation)}
            else:
                # 锚点 = 本段**保留区间**的最后一帧 ✓（不是整段最后一帧 ✗ —— 那还在重叠区里 ✓✗）
                # ⚠️ 交的是**单帧 3 维** `(3,H,W)` ✓：`segments._as_frame_batch` 把 4 维当
                #    `(T,C,H,W)` 的**序列**看 ✓ ⇒ 给 4 维会被再包一层成 `(1,1,3,H,W)` ✗
                #    （2026-09-26 实测踩到 ✓ `media.write_image` 当场报形状 ✓）。
                carry = {int(nxt.index): frames[0, :, int(segment.keepTo) - 1]}
        emit({"kind": "stage", "stage": "segment", "segment": int(segment.index),
              "segmentTotal": plan.segmentCount,
              "note": f"第 {segment.index + 1}/{plan.segmentCount} 段完成 ✓"
                      f"（生成 {int(segment.frames)} 帧 ⇒ 保留 {int(segment.keptFrames)} 帧 ✓）"})

    chain_started = time.perf_counter()
    stitched, chain_report = assemble_chain(parts, plan.segments, fps=fps, normalize=normalize,
                                            expected_frames=plan.totalFrames,
                                            norm_kwargs=norm_kwargs)
    target = root / f"video_chain_seed{int(getattr(request, 'seed', 0) or 0)}.mp4"
    video_report = media_mod.write_video(stitched, target, fps=fps, value_range="-1..1")
    stage_ms["chain"] = int((time.perf_counter() - chain_started) * 1000)

    outputs: dict[str, Any] = {
        "synthetic": bool(getattr(backend, "synthetic", False)),
        "realTensors": True, "realFile": True,
        "videoPath": str(target), "primaryPath": str(target),
        "video": video_report,
        "artifacts": ([{"kind": "video", "path": str(target),
                        "bytes": video_report.get("bytes")}]
                      + [{"kind": "segment", "index": row["index"], "path": row["videoPath"]}
                         for row in runs if row.get("videoPath")]),
        "chain": {
            "plan": plan.to_dict(), "report": chain_report, "segmentRuns": runs,
            "audio": {"note": "各段音频（双流时 ✓）落在**各自那一节**的目录里 ✓；"
                              "本仓口径是音轨不塞进 mp4 ✓（见 media.write_video 的 ``-an`` ✓）"
                              "⇒ 成片暂不带音轨 ✓ —— 如实说明 ✓ 不假装有 ✗"},
        },
        "note": "**多段成片**：真 mp4 ✓（ffprobe 可复核 ✓）；各段产物**同时保留** ✓ 便于逐段核对 ✓",
    }
    emit({"kind": "stage", "stage": "chain",
          "note": f"多段成片 ✓：{plan.segmentCount} 段 ⇒ {plan.totalFrames} 帧"
                  f"（{plan.seconds:.2f}s ✓）"})
    return pipeline_mod.PipelineResult(
        ok=True, cancelled=False, backend=_backend_name(backend),
        synthetic=bool(getattr(backend, "synthetic", False)), plan=None,
        outputs=outputs, stageMs=stage_ms, sampleSteps=steps, guidanceSteps=guided)


def _backend_name(backend: Any) -> str:
    return str(getattr(backend, "name", type(backend).__name__))


def _stopped(pipeline_mod: Any, backend: Any, stage_ms: dict[str, int],
             runs: list[dict[str, Any]], index: int) -> Any:
    """段间被叫停 ✓ ⇒ **取消**（不是失败 ✗ —— 前端要分开显示 ✓）；已跑完的段产物**留着** ✓。"""
    return pipeline_mod.PipelineResult(
        ok=False, cancelled=True, backend=_backend_name(backend),
        synthetic=bool(getattr(backend, "synthetic", False)), stageMs=dict(stage_ms),
        outputs={"chain": {"segmentRuns": runs, "stoppedBefore": int(index)}},
        error={"stage": "chain", "cancelled": True,
               "message": f"已取消：停在第 {index} 段之前 ✓（前面几段的产物保留 ✓）"})


def _segment_failed(pipeline_mod: Any, backend: Any, result: Any, stage_ms: dict[str, int],
                    runs: list[dict[str, Any]], index: int) -> Any:
    """第 ``index`` 段失败 ⇒ 整条失败 ✓（阶段名带上段号 ✓ 不然只看到一个 decode 失败 ✓）。"""
    error = dict(result.error or {})
    error["stage"] = f"segment[{index}]/{error.get('stage', '?')}"
    error["segment"] = int(index)
    return pipeline_mod.PipelineResult(
        ok=False, cancelled=False, backend=_backend_name(backend),
        synthetic=bool(getattr(backend, "synthetic", False)), stageMs=dict(stage_ms),
        outputs={"chain": {"segmentRuns": runs, "failedSegment": int(index)}},
        error=error)

