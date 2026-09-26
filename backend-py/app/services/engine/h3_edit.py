"""**续拍**（extend）与**重拍**（retake）—— 与「拼接」正交的两件事（2026-09-26 移植 ✓）。

## 三者不是一回事（先分清，别混 ✗）

* **拼接**（:func:`app.services.engine.chain.run_chain` ✓）：多段**不同提示词**的镜头接成一条片子；
  段与段之间只靠**一帧 PNG** 当锚（``carryFrames`` ✓ —— 过了 VAE 一趟 ✓）。
* **续拍**：一条**连续**的戏，N 个窗口同一提示词；上一段的**尾部潜变量**（不是解码后的帧 ✗）
  钉进下一段的**开头**（``pin`` ✓ 掩码 0 ⇒ 逐位保留 ✓），所以速度与颜色都接得上 ✓。
* **重拍**：把**成片**的某一段重画一遍（画面 / 声音 / 两者 ✓ 互相独立 ✓），窗外**冻结** ✓。

三者共用同一个底层原语：**掩码冻结的局部重绘**（:func:`app.services.engine.h3_form.sample_dual_stream`
的 ``pin`` / ``pin_mask`` ✓）。本模块只管**计划**：算窗口、算槽位、算掩码形状（零依赖 ✓ 可离线钉死 ✓）。

## 事实来源（不猜 ✗）

* ``reference/ComfyUI-H3-Multishot/h3_extend.py`` 的 :func:`_snap_grid` / :func:`plan_take`（MIT ✓）；
* ``reference/ComfyUI-H3-Multishot/h3_retake.py`` 的 ``H3Retake._window`` 与两处网格换算（MIT ✓）；
* 帧网格 ``17k+5`` / 24fps / 音频潜 40Hz：本仓 :mod:`app.services.engine.geometry` 的既有事实 ✓。

## ⚠️ 与上游**刻意不同**的三处（都是「本仓不猜」✓ 不是漏 ✗）

1. 上游 ``plan_take(window="auto")`` 用**实测的显卡速率**（``_GGUF_RATE_GB_PER_MCELL`` 等 ✓）估算
   显存池 ⇒ 本仓**没有**那套实测标定 ✗ ⇒ ``window="auto"`` **必须**给 ``fits`` 判据（调用方自己
   拿本仓的显存估算 ✓），不给就**报错** ✓（编一个"看起来合理"的窗口只会静默改产物 ✗）。
2. 上游 ``_window`` 的 ``b`` 用 ``round(x + 0.5)``（**银行家舍入**的后果与"四舍五入"不同 ✓）——
   本仓**逐位照抄** ✓（换了就等于换窗口边界 ✓ 而窗口边界决定"重画哪一段"✗）；自检拿真值钉住 ✓。
3. 上游对 ``pin_frames`` **不做**任何合法性校验 ✗（widget 只写 ``min``/``max`` ✓）⇒ 钉长 ≥ 窗口
   时它照跑：``step = max(1, f - pin)`` 把每窗新帧压成 1 帧 ✓ 一个 10 s 的片子要跑 219 个窗口 ✓
   ✗。本仓**响亮报错** ✓，判据落在**本窗口**上（``pin ≥ frames`` ✓ 不是网格的最小窗口 22 帧 ✗
   —— 上游钉长可到 56 帧 ✓ 配 141/192/243 帧窗口是合法且有实测的配置 ✓，拿 22 当上限会把它们
   连同本模块的默认值 22 一起拒掉 ✗）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "CONTINUATION_MODES",
    "CANDIDATE_WINDOWS",
    "DEFAULT_PIN_FRAMES",
    "FRAME_GRID",
    "RETAKE_MODES",
    "ContinuationPin",
    "EditPlanError",
    "RetakePlan",
    "TakePlan",
    "continuation_pin",
    "grid_frames",
    "latent_window",
    "plan_retake",
    "plan_take",
    "snap_grid",
]

#: H3 的合法片段长度：``(F - 5) % 17 == 0`` ✓（事实来源：核心节点 schema ✓，本仓
#: :mod:`app.services.engine.geometry` 的 ``H3_MIN_FRAMES`` / ``H3_FRAME_GRID`` 同值 ✓）
FRAME_GRID: tuple[int, int] = (5, 17)
#: 上游 ``h3_extend._WINDOWS`` ✓（**从大到小** ✓ —— 装得下时**接缝最少**的窗口赢 ✓）。
CANDIDATE_WINDOWS: tuple[int, ...] = (243, 226, 209, 192, 175, 158, 141, 124, 107, 90)
#: 上游采样器的图像钉长度（每次接缝裁掉的头 ✓）；``22`` 是它的默认值 ✓（可显式覆盖 ✓）。
DEFAULT_PIN_FRAMES = 22

#: 重拍的三种模式 ✓（文案照抄上游 ``H3Retake.MODES`` ✓ —— 前端要用同一串字 ✓）
RETAKE_MODES: tuple[str, ...] = (
    "video + audio (redo the moment)",
    "video only (keep the performance)",
    "audio only (keep the picture)",
)
CONTINUATION_MODES: tuple[str, ...] = ("video + audio", "video only", "audio only")


class EditPlanError(ValueError):
    """续拍/重拍的计划无法成立（**宁可当场报错** ✓ 不悄悄换个窗口跑 ✗）。"""


def snap_grid(frames: float, *, grid: tuple[int, int] = FRAME_GRID) -> int:
    """帧数 → **四舍五入**到 ``5 + 17k`` 网格 ✓（上游 ``h3_extend._snap_grid`` 逐位照抄 ✓）。

    ⚠️ 与 :func:`app.services.engine.geometry.snap_frames` **不同** ✗：那个是**向上取** ✓（请求 →
    合法长度 ✓），这个是**就近取** ✓（给一个窗口候选时不要白涨一档 ✓）。两者都留着，各有各的用处 ✓。

    校验（与上游同值 ✓）：``snap_grid(243) == 243``（243 = 5 + 14×17 ✓）、``snap_grid(200) == 192`` ✓
    （到 192 差 8 ✓、到 209 差 9 ✓ ⇒ 就近取前者 ✓；要举「就近 ≠ 向上」的例子得用 ``snap_grid(201)``
    ⇒ **209** ✓ —— 2026-09-26 纠：原文把 200 写成 209 ✗，拿它当例子正好把结论举反了 ✓）。
    """
    low, step = grid
    value = max(low + step, int(frames))
    return low + step * max(1, int(round((value - low) / float(step))))


@dataclass(frozen=True)
class TakePlan:
    """续拍的窗口计划 ✓（``num_windows`` 个窗口、每窗 ``frames_per_window`` 帧 ✓）。"""

    num_windows: int
    frames_per_window: int
    step_frames: int          #: 后续每个窗口**新交付**的帧数 = 窗口 − 钉长 ✓
    total_frames: int         #: 全片交付帧数（≈ 请求时长 ✓ 只会**长一点**不会短 ✓）
    pin_frames: int
    fps: int
    requested_frames: int
    summary: str

    @property
    def seconds(self) -> float:
        return self.total_frames / float(max(1, self.fps))

    def to_dict(self) -> dict[str, Any]:
        return {"numWindows": self.num_windows, "framesPerWindow": self.frames_per_window,
                "stepFrames": self.step_frames, "totalFrames": self.total_frames,
                "pinFrames": self.pin_frames, "fps": self.fps,
                "requestedFrames": self.requested_frames,
                "seconds": round(self.seconds, 3), "summary": self.summary}


def plan_take(take_seconds: float, window: int | str, *,
              fps: int = 24, pin_frames: int = DEFAULT_PIN_FRAMES,
              grid: tuple[int, int] = FRAME_GRID,
              windows: tuple[int, ...] = CANDIDATE_WINDOWS,
              fits: Callable[[int], bool] | None = None) -> TakePlan:
    """**续拍**：整条时长 → 窗口长度 + 窗口个数 ✓（上游 ``h3_extend.plan_take`` 的算术部分 ✓）。

    ``window``：给帧数（会 :func:`snap_grid` 就近吸附 ✓）或 ``"auto"`` ✓ —— 走 ``fits`` 判据
    **从大到小**挑第一个装得下的 ✓（``fits`` 由调用方给：本仓的显存估算是**另一处**的事实 ✓，
    这里不复制一份 ✗）；``"auto"`` 而没给 ``fits`` ⇒ :class:`EditPlanError` ✓（见模块头第 1 条 ✓）。

    ⚠️ 窗口个数是**解出来的** ✓ 不是除出来的 ✗：``step = 窗口 − 钉长`` ✓（每个后续窗口只交付
    ``step`` 帧新的 ✓，头部的钉长是**重放**上一段的尾巴 ✓）⇒ ``f + (n-1)·step ≥ 想要`` ✓。

    ``pin ≥ 本窗口`` ⇒ :class:`EditPlanError` ✓（模块头第 3 条 ✓ —— 上游不校验 ✗）。
    """
    fps = int(fps)
    if fps <= 0:
        raise EditPlanError(f"fps 必须是正数（收到 {fps} ✗）")
    wanted = int(round(float(take_seconds) * fps))
    if wanted <= 0:
        raise EditPlanError(f"take_seconds 太小（{take_seconds} s @{fps}fps ⇒ {wanted} 帧 ✗）")
    pin = int(pin_frames)
    if pin < 0:
        raise EditPlanError(f"pin_frames 不能是负数（收到 {pin} ✗）")
    note = ""
    if str(window).strip().lower() == "auto":
        if fits is None:
            raise EditPlanError(
                "window='auto' 必须给 fits 判据 ✗（本仓**没有**上游那套按显卡实测标定的显存速率表 ✓"
                "—— 编一个窗口只会静默改产物 ✓ ⇒ 要么给 fits ✓、要么显式给窗口帧数 ✓）")
        chosen = None
        for candidate in windows:
            if fits(int(candidate)):
                chosen = int(candidate)
                break
        if chosen is None:
            chosen = int(windows[-1])
            note = f" | WARNING: 连 {chosen} 帧的窗口都判作装不下 ⇒ 仍取最小的那个 ✓（调用方自担 ✓）"
        frames = snap_grid(chosen, grid=grid)
        note = f" | auto: fits 判据挑出 {frames} 帧窗口{note}"
    else:
        frames = snap_grid(int(window), grid=grid)
    # 判据落在**本窗口**上 ✓（不是网格的最小窗口 22 帧 ✗ —— 2026-09-26 纠 ✓）：
    #   上游 ``pin_frames`` 的上限是 56 帧 ✓ 而候选窗口最小也有 90 帧 ✓ ⇒ 钉长 39 / 56 配
    #   141 / 192 / 243 帧窗口是上游**合法且有实测**的配置 ✓（见模块头）⇒ 拿「网格最小合法
    #   长度 22」当上限会把它们连同本模块自己的默认值 22 ✗ 一起拒掉 ✓✗。
    #   会「永远交付不了新帧」的只有一种情形：**钉长 ≥ 本窗口** ✓（那时 ``max(1, …)`` 把
    #   每窗新帧压成 1 帧 ⇒ 10 s 的片子要 219 个窗口 ✓ —— 静默跑完比报错更难收拾 ✓）。
    if pin >= frames:
        raise EditPlanError(
            f"pin_frames={pin} 不小于本窗口的帧数 {frames} ✗ ⇒ 每个窗口都会被钉满 ⇒ 永远"
            f"交付不了新帧 ✓（要么调小钉长 ✓、要么加长窗口 ✓）")
    delivered = max(1, frames - pin)
    count = 1
    while frames + (count - 1) * delivered < wanted:
        count += 1
    total = frames + (count - 1) * delivered
    summary = ("EXTEND TAKE: %.1f s requested -> %d windows x %d frames (pin %d) "
               "= %d delivered frames = %.1f s" % (float(take_seconds), count, frames, pin,
                                                   total, total / float(fps)))
    return TakePlan(num_windows=count, frames_per_window=frames, step_frames=delivered,
                    total_frames=total, pin_frames=pin, fps=fps, requested_frames=wanted,
                    summary=summary + note)


def grid_frames(frame_count: int, *, grid: tuple[int, int] = FRAME_GRID,
                min_for_snap: int = 22) -> int:
    """成片帧数 → **网格内**的帧数（**向下**取 ✓；剩下的尾巴原样接回 ✓）。

    上游 ``h3_retake`` 那段 ✓：``keep = 5 + ((n - 5) // 17) * 17``（``n >= 22`` 时 ✓），否则**原样**
    ✓；⚠️ 是**向下** ✗ 不是向上 ✗ —— 重拍只能吃网格内的素材 ✓，多出来的尾巴在**解码之后**原样
    ``torch.cat`` 回去 ✓（与上游同序 ✓）。

    校验：``grid_frames(124) == 124`` ✓、``grid_frames(130) == 124`` ✓（尾巴 6 帧原样送回 ✓）。
    """
    low, step = grid
    count = int(frame_count)
    if count < min_for_snap:
        return max(0, count)
    return low + ((count - low) // step) * step


def latent_window(n_latent: int, total_seconds: float, start_seconds: float,
                  end_seconds: float) -> tuple[int, int]:
    """秒区间 → **潜槽位**区间 ``[a, b)`` ✓（上游 ``H3Retake._window`` 逐位照抄 ✓）。

    ⚠️ 这里的取整**刻意照抄** ✓（``b = int(round(x + 0.5))`` ✓）：窗口边界决定「重画哪一段」✗，
    换个写法就是**换产物** ✓。自检拿真值钉住 ✓（见 ``tests/engine_h3_edit_test.py`` ✓）。
    """
    slots = max(1, int(n_latent))
    total = float(total_seconds)
    if total <= 0:
        raise EditPlanError(f"total_seconds 必须是正数（收到 {total_seconds} ✗）")
    per = total / float(slots)                       # 每个潜槽位多少秒 ✓
    a = int(max(0.0, float(start_seconds)) / per)
    b = int(round(min(float(end_seconds), total) / per + 0.5))
    a = max(0, min(a, slots - 1))
    b = max(a + 1, min(b, slots))
    return a, b


@dataclass(frozen=True)
class RetakePlan:
    """重拍的计划 ✓：两条流**各算各的**窗口（``None`` = 该流整体冻结 ✓）。"""

    mode: str
    total_seconds: float
    start_seconds: float
    end_seconds: float
    video: tuple[int, int] | None     #: ``(i, j)`` 视频潜槽位窗口（``None`` ⇒ 画面整条冻结 ✓）
    audio: tuple[int, int] | None     #: ``(i, j)`` 音频潜槽位窗口（``None`` ⇒ 声音整条冻结 ✓）
    video_slots: int
    audio_slots: int

    @property
    def redo_video(self) -> bool:
        return self.video is not None

    @property
    def redo_audio(self) -> bool:
        return self.audio is not None

    def describe(self, *, clip_seconds: float | None = None) -> str:
        """一行事实 ✓（照上游 ``H3Retake`` 的 info 口径 ✓ —— 前端直接显示 ✓）。

        ``clip_seconds`` 给**成片**的时长 ✓（不是重拍窗口的时长 ✗ —— 上游也是这么标的 ✓）。
        """
        video = "%d-%d of %d" % (*self.video, self.video_slots) if self.video else "frozen"
        audio = "%d-%d of %d" % (*self.audio, self.audio_slots) if self.audio else "frozen"
        return ("retake %.1f-%.1f s of a %.1f s clip | %s | video slots %s | audio slots %s"
                % (self.start_seconds, self.end_seconds,
                   float(clip_seconds) if clip_seconds is not None else self.total_seconds,
                   self.mode, video, audio))

    def to_dict(self, *, clip_seconds: float | None = None) -> dict[str, Any]:
        return {"mode": self.mode, "totalSeconds": self.total_seconds,
                "startSeconds": self.start_seconds, "endSeconds": self.end_seconds,
                "video": list(self.video) if self.video else None,
                "audio": list(self.audio) if self.audio else None,
                "videoSlots": self.video_slots, "audioSlots": self.audio_slots,
                "redoVideo": self.redo_video, "redoAudio": self.redo_audio,
                "info": self.describe(clip_seconds=clip_seconds)}


def plan_retake(video_slots: int, audio_slots: int, *,
                total_seconds: float, start_seconds: float, end_seconds: float,
                mode: str = RETAKE_MODES[0]) -> RetakePlan:
    """重拍计划 ✓：时间窗 → 两条流各自的潜槽位窗口（**各自独立** ✓ —— 这是上游的关键口径 ✓）。

    ``mode`` 必须是 :data:`RETAKE_MODES` 里那三条之一 ✓（**按前缀判** ✓ 与上游同一个写法 ✓）：
    ``"audio only"`` ⇒ 画面冻结 ✓、``"video only"`` ⇒ 声音冻结 ✓（保住表演 ✓）。

    ⚠️ ``end_seconds <= start_seconds`` ⇒ **报错** ✓（上游也是这么做的 ✓ —— 静默把窗口翻过来
    等于重画了一整段别的 ✗）。
    """
    if mode not in RETAKE_MODES:
        raise EditPlanError(f"mode 必须是 {list(RETAKE_MODES)} 之一（收到 {mode!r} ✗）")
    if float(end_seconds) <= float(start_seconds):
        raise EditPlanError(
            f"end_seconds({end_seconds}) 必须在 start_seconds({start_seconds}) 之后 ✗"
            "（上游 H3Retake 同口径 ✓）")
    total = float(total_seconds)
    redo_video = not mode.startswith("audio only")
    redo_audio = not mode.startswith("video only")
    video = latent_window(video_slots, total, start_seconds, end_seconds) if redo_video else None
    audio = latent_window(audio_slots, total, start_seconds, end_seconds) if redo_audio else None
    return RetakePlan(mode=mode, total_seconds=total, start_seconds=float(start_seconds),
                      end_seconds=float(end_seconds), video=video, audio=audio,
                      video_slots=max(1, int(video_slots)), audio_slots=max(1, int(audio_slots)))


@dataclass(frozen=True)
class ContinuationPin:
    """续拍要**钉住**的头部槽位数 ✓（上一段的尾巴潜变量落在新窗口的开头 ✓）。"""

    pin_frames: int
    video_slots: int
    audio_slots: int
    #: 被钉住的秒数（音频槽数就是按它算的 ✓）
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {"pinFrames": self.pin_frames, "videoSlots": self.video_slots,
                "audioSlots": self.audio_slots, "seconds": round(self.seconds, 4)}


def continuation_pin(pin_frames: int, *, fps: int = 24, temporal_compression: int,
                     audio_latent_mode: str, audio_hz: int = 40) -> ContinuationPin:
    """钉住的**像素帧数** → 两条流的**头部槽数** ✓（续拍唯一的换算 ✓ 只此一处 ✓）。

    ⚠️ ``temporal_compression``（视频 VAE 的时间压缩比）**必须显式给** ✗（本仓不猜压缩比 ✓ ——
    见 :mod:`app.services.engine.geometry` 的模块头 ✓）；音频侧走 :func:`geometry.audio_latent_frames`
    的 ``mode``（同样必填 ✓ 理由见那儿 ✓），槽数由**秒数**算（不是帧数 ✗ —— 音频那侧只认秒 ✓）。
    """
    from app.services.engine import geometry  # noqa: PLC0415 —— 只用一个换算 ✓

    frames = int(pin_frames)
    if frames < 0:
        raise EditPlanError(f"pin_frames 不能是负数（收到 {frames} ✗）")
    if fps <= 0:
        raise EditPlanError(f"fps 必须是正数（收到 {fps} ✗）")
    compression = int(temporal_compression)
    if compression <= 0:
        raise EditPlanError(f"temporal_compression 必须是正数（收到 {temporal_compression} ✗）")
    slots = int(math.ceil(frames / float(compression))) if frames else 0
    seconds = frames / float(fps)
    return ContinuationPin(pin_frames=frames, video_slots=slots,
                           audio_slots=geometry.audio_latent_frames(seconds,
                                                                    mode=str(audio_latent_mode),
                                                                    hz=int(audio_hz)),
                           seconds=seconds)
