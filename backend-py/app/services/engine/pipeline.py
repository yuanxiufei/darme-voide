"""生成管线（**编排层**，2026-09-17）—— 把一个生成请求拆成**可观测、可取消、可归因**的阶段。

## 它解决什么真问题

生成很慢（视频动辄几十秒到几分钟 ✓）而失败很贵 ✗：如果只写「一个大函数」，用户只能看到
**转圈**（不知道卡在哪 ✗）、失败只看到**一句异常**（不知道是编码、采样还是解码炸的 ✗）。

所以这里把流程钉成六个**有名字的阶段** ✓，每阶段记耗时、抛错时**带上阶段与步号** ✓：

```
plan → encode → init → sample → decode → write
```

## 后端可插拔（这是本模块最重要的设计 ✓）

管线本身**不碰张量** ✗ —— 所有张量操作都经 :class:`GenerationBackend` 协议交给后端 ✓：

* **真后端**：``torch`` + 权重（装好依赖后由 ``torch_backend`` 实现 ✓）；
* **干跑后端**：:mod:`app.services.engine.dryrun`（**零依赖** ✓，用纯 Python 小向量跑完整管线 ✓）
  —— 用来验**编排本身**（阶段顺序、事件、取消、错误归因 ✓），也是前端「不装权重先试流程」的底座 ✓。

⚠️ 干跑**不产生真图/真视频** ✗（输出会显式标注 ``synthetic: true`` ✓）—— 不许拿它冒充生成结果 ✗。

## 算法层是真用的（不是摆设 ✓）

``sample`` 阶段**真的**调 :func:`app.services.engine.sampler.sample` ✓（Euler/Heun/多阶 ✓），
σ 序列**真的**来自 :mod:`app.services.engine.schedules` ✓（Karras 等 ✓），
尺寸/帧数**真的**来自 :mod:`app.services.engine.geometry` ✓ ⇒ 换后端不换算法 ✓。

## 取消（协同式 ✓）

长任务必须能停 ✓：``cancel`` 是个可调用对象（返回 ``True`` 表示要停 ✓），
在**阶段边界**与**每个采样步**检查 ✓ ⇒ 取消后抛 :class:`GenerationCancelled` ✓
（已完成阶段的信息仍会带出来 ✓，便于前端显示「停在第几步」✓）。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from . import conditioning as conditioning_mod
from . import geometry, guidance as guidance_mod, sampler as sampler_mod, schedules

__all__ = [
    "IMAGE_STAGE",
    "STAGE_ORDER",
    "VIDEO_STAGE",
    "GenerationBackend",
    "GenerationCancelled",
    "GenerationPlan",
    "GenerationRequest",
    "PipelineResult",
    "StageError",
    "build_plan",
    "run_sync",
]

#: 两条**阶段**名 ✓（= `runtime.ensure_loaded(stage=…)` 认的那两个 ✓ —— 一个概念贯穿三层 ✓）。
#: * :data:`VIDEO_STAGE` ✓：H3 双流视频（默认 ✓）；
#: * :data:`IMAGE_STAGE` ✓：SDXL **单张图片** ✓。
#: ⚠️ 名字与 `SdxlBackend.name`（``"sdxl"`` ✓）**刻意相同** ✓ ——
#: `runtime` 按 stage 选后端 ✓、`bridge` 按 stage 拼装配 ✓、出图链路按后端名分流 ✓，
#: 三处对不上就会「装的是视频后端、跑的是图片 plan」✓✗（那种错**不报错** ✓ 只是结果全错 ✓）。
VIDEO_STAGE = "h3"
IMAGE_STAGE = "sdxl"

#: 阶段顺序（也是前端进度条的依据 ✓）
#: ⚠️ 这里**只列必经阶段** ✗ —— 可选阶段（``condition`` ✓、``refine`` ✓）靠 :data:`STAGE_LABELS`
#: 与耗时表体现 ✓（加进来会让"六个阶段都有耗时"这类判据失效 ✓✗）。
STAGE_ORDER: tuple[str, ...] = ("plan", "encode", "init", "sample", "decode", "write")

#: 各阶段的中文名（报错/事件里给人看 ✓）
STAGE_LABELS: dict[str, str] = {
    "plan": "规划（尺寸/帧数/σ 序列）",
    "encode": "文本编码",
    "init": "初始化潜变量",
    #: 可选阶段 ✓（只有图生视频会给首帧 ⇒ 不给就不出现在耗时表里 ✓）
    "condition": "首帧条件（图生视频）",
    "sample": "去噪采样",
    #: 可选阶段 ✓（只有计划说"用放大器"才跑 ✓；**音频流不重采** ✗）
    "refine": "二采精修（超清）",
    "decode": "解码（潜变量 → 帧）",
    "write": "落盘",
}


class GenerationError(RuntimeError):
    """引擎的基类错误 ✓。"""


class GenerationCancelled(GenerationError):
    """用户中途取消 ✓（**不是失败** ✗ —— 前端应当按"已取消"显示 ✓）。"""


class StageError(GenerationError):
    """某个阶段失败 ✓ —— 带**阶段名 + 步号 + 原异常**，便于归因 ✓。"""

    def __init__(self, stage: str, message: str, *, step: int | None = None,
                 cause: BaseException | None = None) -> None:
        label = STAGE_LABELS.get(stage, stage)
        where = f"（第 {step} 步）" if step else ""
        super().__init__(f"{label}{where}失败：{message}")
        self.stage = stage
        self.step = step
        self.__cause__ = cause


# ══════════════════════════════════════════════════════════════════════════
# 请求与计划
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class GenerationRequest:
    """一次生成请求（不可变 ✓ —— 便于重放与缓存键计算 ✓）。"""

    prompt: str
    negative: str = ""
    seed: int = 0
    seconds: float = 5.0
    fps: int = geometry.H3_FPS
    ratio: Any = "16:9"          # "16:9" / 数字 / (w, h) ✓
    megapixels: float = 0.65
    steps: int = 30
    sampler: str = "euler"
    schedule: str = "karras"
    temporal_compression: int | None = None   # ⚠️ 必须显式给（见 geometry 的原则 ✓）
    first_frame: str | None = None            # 图生视频的首帧 ✓
    reference_frames: tuple[str, ...] = ()
    #: 参考**音频**（wav 路径 ✓）—— H3 的 ``ref_audio`` 块 ✓（本仓库自己实现读写 ✓）。
    #: ⚠️ 采样率/声道要与音频 VAE 一致 ✓（**不重采样** ✗ —— 见 `media.load_wav_tensor` ✓）。
    reference_audio: tuple[str, ...] = ()
    #: 参考**视频**（视频文件路径 ✓）—— H3 的 ``video`` / ``video_audio`` 块 ✓
    #:（有无音轨决定块类型 ✓ —— 有音轨则两条流都编码 ✓）。
    #: ⚠️ 用**原生尺寸/帧率** ✓（**不缩放、不抽帧** ✗ —— 见 `media.load_video_tensor` ✓）；
    #: 宽高要能被 vae_scale 整除 ✗（不整除**报错** ✓ 不悄悄取整 ✗）。
    reference_videos: tuple[str, ...] = ()
    outputs_dir: str | None = None
    dry_run: bool = False
    #: 引导（CFG ✓）—— 默认 ``scale=1.0`` ⇒ **不引导** ✓（不被要求的干预一律不做 ✓）
    guidance: guidance_mod.GuidanceConfig = field(default_factory=guidance_mod.GuidanceConfig)
    #: 首帧条件（图生视频 ✓）—— 只在给了 ``first_frame`` 时生效 ✓
    conditioning: conditioning_mod.ConditioningConfig = field(
        default_factory=conditioning_mod.ConditioningConfig)
    #: 超清计划 ✓ —— 由 :func:`app.services.engine.upscale.plan_upscale` 算出 ✓（调用方先走
    #: ``/engine/upscale-plan`` ✓）；``None`` ⇒ 普通模式 ✓。⚠️ 用 **frozen dataclass** 而非 dict ✗：
    #: 本请求要能当**缓存键**（不可变 ✓ 可哈希 ✓），dict 会让它不可哈希 ✓✗。
    upscale: Any = None
    #: 走哪条**阶段** ✓：``"h3"`` = 视频（默认 ✓，H3 双流）/ ``"sdxl"`` = **单张图片** ✓。
    #: ⚠️ 与 `runtime` 装配的 ``stage=`` 是**同一个概念** ✓（一个名字贯穿三层 ✓）——
    #: ``runtime`` 按它选后端 ✓、:func:`build_plan` 按它选 plan 口径 ✓、
    #: `bridge` 按它拼装配 ✓、`image_generation` 按它选产物路径 ✓。
    #: ⚠️ 放在**末尾** ✗ 不是随手 ✗：插在中间会移动位置参数 ✓✗。
    stage: str = "h3"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ratio"] = list(self.ratio) if isinstance(self.ratio, tuple) else self.ratio
        return data


@dataclass
class GenerationPlan:
    """请求 → **具体数字**（给用户看、给后端用 ✓）。"""

    width: int = 0
    height: int = 0
    frames: int = 0
    fps: int = 0
    latent_frames: int | None = None
    sigmas: list[float] = field(default_factory=list)
    timesteps: list[float] = field(default_factory=list)
    steps: int = 0
    warnings: list[str] = field(default_factory=list)
    #: 超清计划（``None`` = 本次没要超清 ✓；有值就说明**会跑二采阶段** ✓）
    upscale: dict[str, Any] | None = None
    #: **相对**耗时系数 ✓（1 步 euler 无引导 = 1 ✓）—— 见 :meth:`to_dict` 里为什么不报秒数 ✗
    cost_factor: float = 0.0

    @property
    def costFactor(self) -> float:  # noqa: N802 —— 与前端 camelCase 对齐的只读别名 ✓
        return self.cost_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width, "height": self.height, "frames": self.frames, "fps": self.fps,
            "latentFrames": self.latent_frames, "steps": self.steps,
            "sigmas": [round(value, 4) for value in self.sigmas],
            "timesteps": [round(value, 4) for value in self.timesteps],
            "durationSeconds": round(self.frames / self.fps, 3) if self.fps else 0,
            # ⚠️ **相对**耗时系数（步数 × 采样器 × 引导 ✓ 三者都可精确算 ✓）——
            #    刻意不给"秒" ✗：那需要每步的算力模型（我们还没有 ✓），
            #    硬报一个秒数就是**编** ✗。前端要秒数就等真后端跑一次后按实测标定 ✓。
            "costFactor": round(self.cost_factor, 4),
            "upscale": self.upscale,
            "warnings": self.warnings,
        }


def build_plan(request: GenerationRequest, *, weights_bytes: int | None = None) -> GenerationPlan:
    """算出这一次要用的**尺寸/帧数/σ 序列** ✓（纯计算、毫秒级、可先给用户看 ✓）。

    ⚠️ 这里**提前**校验采样器名与参数 ✓ —— 让错误在「还没开始跑」时就以明确文案返回 ✓，
    而不是跑到一半才炸 ✗。
    """
    plan = GenerationPlan(fps=int(request.fps or geometry.H3_FPS))
    if not str(request.prompt or "").strip():
        raise StageError("plan", "提示词不能为空 ✓")
    if request.steps < 1:
        raise StageError("plan", f"steps 必须 ≥1（收到 {request.steps} ✗）")
    if request.seconds <= 0:
        raise StageError("plan", f"seconds 必须 >0（收到 {request.seconds} ✗）")
    if str(request.sampler or "").lower() not in sampler_mod.SAMPLERS:
        raise StageError("plan", f"未知采样器 {request.sampler!r}；可用：{sorted(sampler_mod.SAMPLERS)}")

    stage = str(request.stage or VIDEO_STAGE).strip().lower()
    if stage == IMAGE_STAGE:
        _fill_image_plan(request, plan)
        return _finish_plan(request, plan, weights_bytes)
    if stage != VIDEO_STAGE:
        raise StageError(
            "plan", f"未知 stage {request.stage!r} ✗；可用：{sorted((VIDEO_STAGE, IMAGE_STAGE))} ✓"
            "（⚠️ 别随手编一个名字 ✓ —— 它决定选哪个后端 ✓✗）")

    plan.width, plan.height = geometry.size_for_megapixels(
        max(0.05, float(request.megapixels)), request.ratio)
    plan.frames = geometry.snap_frames(request.seconds, fps=plan.fps)
    plan.steps = int(request.steps)

    try:
        plan.sigmas = schedules.sigmas_for(plan.steps, str(request.schedule or "karras"))
    except (ValueError, TypeError) as err:
        raise StageError("plan", f"σ 调度不可用：{err}", cause=err) from err
    plan.timesteps = schedules.timesteps_for(plan.sigmas)

    if request.temporal_compression:
        plan.latent_frames = geometry.latent_frames(plan.frames,
                                                    temporal_compression=int(request.temporal_compression))
    else:
        # 诚实标注：不猜压缩比（见 geometry 的模块注释 ✓）—— 后端可以自己算 ✓
        plan.warnings.append("未给 temporalCompression ⇒ 潜空间帧数未知（不影响采样，解码时后端自行决定 ✓）")
    plan.warnings.append(f"{plan.frames} 帧 @{plan.fps}fps ≈ {plan.frames / max(1, plan.fps):.2f}s"
                         f"（已吸附到 {geometry.H3_FRAME_GRID}k+{geometry.H3_MIN_FRAMES} 网格 ✓）")
    return _finish_plan(request, plan, weights_bytes)


def _fill_image_plan(request: GenerationRequest, plan: GenerationPlan) -> None:
    """SDXL **单张图片**的 plan ✓。

    ⚠️ 与视频那条路**刻意分开** ✗✗（不是"顺手复用" ✓）：四处口径根本不同 ✓ ——
    ① **尺寸**要吸附到 **8 的倍数** ✓（SDXL 潜空间 1/8 ✓；不整除会在下采样时报错 ✗）；
    ② **帧数恒为 1** ✓、**没有帧率**概念 ✓（``fps = 0`` ✓ —— 不拿 24/30 冒充 ✗）；
    ③ **σ 序列必须按 0..999 的离散格取** ✓✗（见 :func:`sdxl_backend.sdxl_sigmas_for_steps` ✓）——
       套视频那边的 karras ρ-ramp ⇒ σ 落不到格上 ✓ ⇒ 相邻两步被 :func:`sigma_to_timestep`
       映到**同一个 t** ✓ ⇒ 白跑一遍同样的噪声水平（**不报错、图还能出** ✓ 属"看不出来"那类 ✓✗）；
    ④ **画布恒按训练预算收** ✓（= :data:`sdxl_backend.SDXL_TRAINED_MEGAPIXELS` ✓，只保留
       **比例** ✗）—— 视频默认那 0.65 MP 是**视频模型**的实测口径 ✓，而生产线上图片记录又默认
       **1920×1080 = 2.07 MP** ✓ ⇒ 一个偏低、一个偏高，**两个方向都坏、且都不报错** ✓✗
       （症状与实测数据见 :data:`sdxl_backend.SDXL_TRAINED_RESOLUTION` 的注释 ✓）。
    """
    from . import sdxl_backend as image_mod  # 局部 import ✓：这条路的可选依赖别拖累视频 ✓

    # ⚠️ **画布只能是训练预算** ✗✗（不是"用户要多大就给多大" ✓）—— **两个方向**都会**不报错地**坏掉 ✓✗，
    #    而且都在本仓"看不出来"那一类里 ✓：
    #    * **低于**预算：实测（同 seed / 同提示词 ✓）512×512 ⇒ **物体重复 + 霓虹过饱和**（一长串苹果 +
    #      荧光绿 ✓），256×256 ⇒ **纯色块** ✓ —— 症状与**代码写错**长得一模一样 ✓✗；
    #    * **高于**预算：实测 1440×1440（2.07 MP ✓）⇒ VAE 解码的 ``reserved`` 冲到 **30.66 GiB**，
    #      而物理显存只有 **22.49 GiB** ✗✗ ⇒ 走 WDDM **共享显存换页** ⇒ **光解码就 11.0 s**
    #      （1024² 同条件 **0.66 s** ✓）—— 这正是用户口径里明令不许的"逼近/超过显存上限"✓✗。
    #      顺带把更大的那个误算也钉住 ✗：**生产默认的 1920×1080 ⇒ 2.07 MP** ✓（见
    #      `image_generation.py` 里 ``size=params.get("size") or "1920x1080"`` ✓）⇒ 默认就落在换页区 ✓✗。
    #    ⇒ 两个方向都**按训练预算收** ✓ 并**如实报改过** ✗（不静默 ✓）；**要更大像素走超清那条** ✓
    #      （放大是常规做法 ✓；把扩散画布硬撑大 = 拿"没训过的档位"换画质与显存 ✗）。
    wanted_mp = max(0.05, float(request.megapixels))
    budget_mp = image_mod.SDXL_TRAINED_MEGAPIXELS
    # ⚠️ 容差 1% ✗（不拿"完全相等"当判据 ✓）：UI 给的 1024×1024 经 `geometry.megapixels_for_size`
    #    会被 round 成 1.0486（真值 1.048576 ✓）⇒ 差 0.0024% ✓ ⇒ 这种**不能被判成"改了用户的"** ✓✗。
    if abs(wanted_mp - budget_mp) > 0.01 * budget_mp:
        direction = "低于" if wanted_mp < budget_mp else "高于"
        # ⭐ **如实报改过** ✓ 不静默改 ✗（同下面"吸附到 8 的倍数"那条纪律 ✓）
        plan.warnings.append(
            f"像素预算{direction} SDXL 的训练分辨率 ⇒ **已按训练预算收到 {budget_mp:g} MP** ✓"
            f"（原 {wanted_mp:g} MP ✓；训练分辨率 "
            f"{image_mod.SDXL_TRAINED_RESOLUTION}×{image_mod.SDXL_TRAINED_RESOLUTION} ✓，"
            "出处见 `sdxl_backend.SDXL_TRAINED_RESOLUTION` 注释 ✓）。"
            "⚠️ 两个方向都**不报错**但都会坏 ✓✗：低 ⇒ 物体重复 + 霓虹过饱和（实测 512×512 ✓）/ "
            "纯色块（实测 256×256 ✓）；高 ⇒ 解码 reserved 超物理显存 ⇒ 共享显存换页 ⇒ 光解码 11 s"
            "（实测 1440×1440 ✓）。要更大像素请走**超清**那条 ✓"
            "（把扩散画布撑大是拿没训过的档位换画质 ✗）")

    want_w, want_h = geometry.size_for_megapixels(budget_mp, request.ratio)
    plan.width, plan.height, snapped = image_mod.snap_image_size(want_w, want_h)
    if snapped:
        # ⭐ **如实报改过** ✓ 不静默改 ✗（用户要 1023 却拿到 1016 而不被告知 ⇒ "看不出来" ✓✗）
        plan.warnings.append(
            f"尺寸已吸附到 {image_mod.SDXL_LATENT_SCALE} 的倍数：{want_w}×{want_h} ⇒ "
            f"**{plan.width}×{plan.height}** ✓（SDXL 潜空间按 1/{image_mod.SDXL_LATENT_SCALE} 采样 ✓，"
            "不整除会在下采样时报错 ✗）")
    plan.frames = 1
    plan.fps = 0                       # 单张图没有帧率 ✓（`to_dict` 据此报 durationSeconds=0 ✓）
    plan.steps = int(request.steps)
    schedule = str(request.schedule or "").strip().lower()
    try:
        plan.sigmas = image_mod.sdxl_sigmas_for_steps(plan.steps, schedule=schedule)
    except image_mod.SdxlBackendError as err:
        raise StageError(
            "plan",
            f"SDXL 的 σ 调度不可用：{err}\n"
            f"  · 当前 schedule={request.schedule!r} ✓；可用：{list(image_mod.SDXL_SCHEDULES)} ✓\n"
            "  · ⚠️ 视频那套 karras/linear 是**连续**调度 ✗，套到离散格上算出的 σ 会落不到格 ✓✗",
            cause=err) from err
    # ⚠️ 这里的 `timesteps` 是**真·UNet 时间步** ✓（整数 ✓）—— 视频那边的 `timesteps` 不是这个含义 ✗。
    plan.timesteps = [float(image_mod.sigma_to_timestep(value)) for value in plan.sigmas]
    plan.warnings.append(
        f"{plan.width}×{plan.height} 单张图片 ✓、{plan.steps} 步 {schedule or 'normal'} ✓"
        f"（σ {plan.sigmas[0]:.4g} ⇒ 0 ✓；UNet 时间步 {plan.timesteps[0]:.0f} ⇒ {plan.timesteps[-1]:.0f} ✓）")


def _finish_plan(request: GenerationRequest, plan: GenerationPlan,
                 weights_bytes: int | None) -> GenerationPlan:
    """两条路**共用**的收尾 ✓（相对耗时系数 ✓ / 超清计划 ✓ / 显存估算 ✓）。

    ⚠️ 刻意**不含尺寸与 σ** ✗ —— 那两样两条路口径不同 ✓，各自在前面算好了 ✓。
    """
    # ── 相对耗时系数（**可精确算** ✓：步数 × 采样器 × 引导 ✓）────────────────
    # heun 每个区间调模型 2 次 ✓；CFG 让"施加引导的那些步"再翻倍 ✓ ⇒ 引导只需要按**生效比例**算 ✓：
    # cost = (步数 + 生效步数) × 采样器倍数 ✓（全程引导时正好 = 步数 × 2 × 倍数 ✓）。
    sampler_multiplier = 2 if str(request.sampler or "").lower() == "heun" else 1
    config = request.guidance
    guided_steps = sum(1 for index in range(plan.steps)
                       if guidance_mod.scale_for_step(index, plan.steps, config) != 1.0)
    plan.cost_factor = float((plan.steps + guided_steps) * sampler_multiplier)
    if config.enabled:
        # ⚠️ 引导的**代价**必须提前说 ✓（CFG 每步两次模型调用 ⇒ 二阶采样器就是每步四次 ✗）
        plan.warnings.append(
            f"引导 cfg={config.scale:g}：{plan.steps} 步里有 {guided_steps} 步生效，"
            f"相对耗时约 ×{plan.cost_factor / max(1.0, float(plan.steps)):.2f}"
            f"（heun 每步还要 ×2 ✗）" +
            (f"（重标定 {config.rescale:g} ✓）" if config.rescale else ""))
    # ── 超清计划（**照调用方给的** ✓ 不自己猜倍率/分块 ✗）────────────────────────
    if request.upscale is not None:
        plan.upscale = request.upscale.to_dict()
        if plan.upscale.get("usesUpscaler"):
            plan.warnings.append(
                f"超清二采：{plan.upscale['mode']} ×{float(plan.upscale.get('scale') or 0):g}"
                f"（⚠️ 只重采**视频流** ✓ 音频流锁定 ✗；耗时按逆向口径 ≈**6×** ✓）")
        else:
            # ⭐ **不启用也必须说清为什么** ✗（静默降级 = 用户以为超清开着 ✓✗）
            plan.warnings.append(
                f"超清**未启用** ⇒ 普通模式出片：{plan.upscale.get('fallbackReason') or '（没给理由 ✗）'}")
    if weights_bytes:
        from .inventory import estimate_vram

        vram = estimate_vram(int(weights_bytes))
        plan.warnings.append(
            f"权重约 {vram['weightsGiB']} GiB ⇒ 估算占用 {vram['estimatedGiB']} GiB"
            f"（{vram['disclaimer']}）")
    return plan


# ══════════════════════════════════════════════════════════════════════════
# 后端协议
# ══════════════════════════════════════════════════════════════════════════
class GenerationBackend(Protocol):
    """推理后端（**唯一**碰张量的地方 ✓）。"""

    name: str
    #: 干跑后端标 ``True``（输出必须被标注为合成 ✓）
    synthetic: bool

    def encode_text(self, request: GenerationRequest) -> Any:
        """提示词 → 条件 ✓。

        **推荐**回 ``{"positive": ..., "negative": ...}`` ✓ —— 管线据此做 CFG
        （见 :mod:`app.services.engine.guidance` ✓）；只回单个条件也**兼容** ✓
        （那时视为 positive-only ⇒ 不引导 ✓，不报错 ✗）。
        """

    def init_latents(self, plan: GenerationPlan, request: GenerationRequest) -> Any:
        """按 plan 造初始噪声（种子由 ``request.seed`` 决定 ✓）。"""

    # 可选能力（不给也行 ✓，但给了 ``request.first_frame`` 却不实现 ⇒ 会**明确失败** ✗，
    # 见 pipeline 的 condition 阶段 ✓）：
    #
    # def condition_first_frame(self, latents, image_path, mask, plan, request) -> Any:
    #     """首帧条件 ✓：后端**自己**把图片编码成潜变量（VAE 是它的事 ✓），再按 ``mask``
    #     （每潜帧一个权重 ✓ 由 :mod:`app.services.engine.conditioning` 算出 ✓）套到 latents 上 ✓。
    #
    #     怎么套（通道维拼接 / 时间维替换 / mask 输入）是**后端自己的事** ✓ —— 引擎不猜布局 ✗。
    #     必须**返回**新的 latents ✓（返回 None ⇒ 引擎判为错误 ✓，不装作成功 ✗）。"""

    def denoise(self, latents: Any, sigma: float, condition: Any,
                request: GenerationRequest) -> Any:
        """给 ``sampler`` 用的**模型调用** ✓：返回去噪估计（x0 ✓）。"""

    def decode(self, latents: Any, plan: GenerationPlan, request: GenerationRequest) -> dict[str, Any]:
        """潜变量 → 帧（可含音频 ✓）。"""

    def write(self, outputs: dict[str, Any], plan: GenerationPlan,
              request: GenerationRequest) -> dict[str, Any]:
        """落盘 ✓（返回产物描述：路径/尺寸/时长 ✓）。"""


# ══════════════════════════════════════════════════════════════════════════
# 结果
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class PipelineResult:
    """一次运行的全部事实 ✓（成功与否都返回它 —— 便于调用方统一落库/展示 ✓）。"""

    ok: bool = False
    cancelled: bool = False
    backend: str = ""
    synthetic: bool = False
    plan: GenerationPlan | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    stageMs: dict[str, int] = field(default_factory=dict)
    sampleSteps: int = 0
    #: 实际施加了引导的步数 ✓（0 = 全程未引导 ✓ —— 告诉用户"真的做了"还是"跳过了" ✓）
    guidanceSteps: int = 0
    #: 首帧条件的实况 ✓（``None`` = 本次没要求图生视频 ✓；有值就说明**真的套上去了** ✓）
    conditioning: dict[str, Any] | None = None
    #: 二采精修的实况 ✓（``None`` = 本次没要超清 / 计划说不启用放大器 ✓）
    refine: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    @property
    def totalMs(self) -> int:
        return sum(self.stageMs.values())

    def event(self, stage: str, *, step: int | None = None, total: int | None = None,
              note: str = "") -> dict[str, Any]:
        """给前端的事件载荷（字段名与前端 camelCase 约定一致 ✓）。"""
        payload: dict[str, Any] = {
            "kind": "stage", "stage": stage, "label": STAGE_LABELS.get(stage, stage),
            "elapsedMs": self.totalMs, "stageMs": self.stageMs.get(stage, 0),
        }
        if step is not None:
            payload["step"] = step
        if total is not None:
            payload["total"] = total
        if note:
            payload["note"] = note
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "cancelled": self.cancelled, "backend": self.backend,
            "synthetic": self.synthetic, "sampleSteps": self.sampleSteps,
            "guidanceSteps": self.guidanceSteps, "conditioning": self.conditioning,
            "totalMs": self.totalMs, "stageMs": dict(self.stageMs),
            "plan": self.plan.to_dict() if self.plan else None,
            "refine": self.refine,
            "outputs": self.outputs, "error": self.error,
        }


CancelFn = Callable[[], bool]


def run_sync(request: GenerationRequest, backend: GenerationBackend, *,
             on_event: Callable[[dict[str, Any]], None] | None = None,
             cancel: CancelFn | None = None) -> PipelineResult:
    """**同步**跑完一次生成（FastAPI 的同步端点跑在线程池里 ✓ ⇒ 不会卡事件循环 ✓）。

    返回 :class:`PipelineResult` ✓（**不抛异常** ✗ —— 失败也体现在结果里 ✓，
    因为「阶段 + 步号 + 文案」比异常栈对用户有用得多 ✓）。
    """
    result = PipelineResult(backend=getattr(backend, "name", type(backend).__name__),
                            synthetic=bool(getattr(backend, "synthetic", False)))
    condition: Any = None
    latents: Any = None
    #: ⭐ 本次是否走**双流**（视频 + 音频 ✓ H3 形态）—— 由**后端自述**决定 ✓（管线不猜 ✗）
    dual = False

    def emit(payload: dict[str, Any]) -> None:
        if on_event:
            on_event(payload)

    def ensure_not_cancelled(stage: str, step: int | None = None) -> None:
        if cancel and cancel():
            raise GenerationCancelled(f"已取消（停在{STAGE_LABELS.get(stage, stage)}"
                                      f"{f' 第 {step} 步' if step else ''} ✓）")

    # ── ① plan ────────────────────────────────────────────────────────────
    started = time.perf_counter()
    try:
        result.plan = build_plan(request)
    except StageError as err:
        result.stageMs["plan"] = int((time.perf_counter() - started) * 1000)
        result.error = {"stage": err.stage, "message": str(err)}
        return result
    result.stageMs["plan"] = int((time.perf_counter() - started) * 1000)
    emit(result.event("plan", note=f"{result.plan.width}×{result.plan.height} / "
                                   f"{result.plan.frames} 帧 / {result.plan.steps} 步"))
    plan = result.plan

    # ── ② encode ──────────────────────────────────────────────────────────
    started = time.perf_counter()
    try:
        ensure_not_cancelled("encode")
        condition = backend.encode_text(request)
    except BaseException as err:  # noqa: BLE001 —— 见下：取消与失败都归到这里 ✓
        result.stageMs["encode"] = int((time.perf_counter() - started) * 1000)
        return _fail(result, "encode", err)
    result.stageMs["encode"] = int((time.perf_counter() - started) * 1000)
    emit(result.event("encode"))

    # ── ③ init ────────────────────────────────────────────────────────────
    started = time.perf_counter()
    try:
        ensure_not_cancelled("init")
        init_dual = getattr(backend, "init_dual_latents", None)
        note = getattr(backend, "dualStream", None)
        # ⚠️ 只有后端**自述可用**（``enabled=True`` ✓）才走双流 ✓ —— 光"有这个方法"不算 ✗。
        #    `dualStream` 会**逐条报缺什么** ✓（没挂音频 VAE / 形态不对 / 没给取整口径 ✓）——
        #    别把"双流不齐"顺手当成"那就照单流跑" ✗（那会做出**与请求不符**的产物 ✓✗）。
        dual = bool(callable(init_dual) and isinstance(note, dict) and note.get("enabled"))
        # ⚠️ 参考图同理（见下面首帧那段 ✓）：后端没自述就别让它跑 ✗ ——
        #    `init_dual_latents` 会**静默忽略** `referenceFrames` ✓✗（产物与参考无关 ✓ 而用户看不出来 ✓）。
        # ⚠️ 判定必须在**调用 init 之前** ✓：等它跑完再判就已经"悄悄忽略过了"✗。
        if dual and tuple(request.reference_frames or ()) and not note.get("referencesImage"):
            raise conditioning_mod.ConditioningError(
                "双流后端**没有自述支持参考图** ✗ ⇒ 不静默忽略 `referenceFrames` ✗"
                "（要么后端把 `ref_img` 段接上 ✓，要么这次别给参考图 ✓）")
        if dual and tuple(request.reference_audio or ()) and not note.get("referencesAudio"):
            raise conditioning_mod.ConditioningError(
                "双流后端**没有自述支持参考音频** ✗ ⇒ 不静默忽略 `referenceAudio` ✗"
                "（要么后端把 `ref_audio` 块接上 ✓，要么这次别给参考音频 ✓）")
        if dual and tuple(request.reference_videos or ()) and not note.get("referencesVideo"):
            raise conditioning_mod.ConditioningError(
                "双流后端**没有自述支持参考视频** ✗ ⇒ 不静默忽略 `referenceVideos` ✗"
                "（要么后端把 `video` / `video_audio` 块接上 ✓，要么这次别给参考视频 ✓）")
        latents = init_dual(plan, request) if dual else backend.init_latents(plan, request)
    except BaseException as err:  # noqa: BLE001
        result.stageMs["init"] = int((time.perf_counter() - started) * 1000)
        return _fail(result, "init", err)
    result.stageMs["init"] = int((time.perf_counter() - started) * 1000)
    emit(result.event("init", step=0, total=plan.steps,
                      note="双流（视频 + 音频 ✓）" if dual else ""))

    # ── ③b condition（**可选阶段** ✓：只有给了首帧才跑 ✓）────────────────────
    # ⚠️ 这里的设计取舍值得写明：**后端不支持就报错** ✗，而不是**悄悄按文生视频跑** ✗。
    #    后者看着"更宽容"，实际是最坏的：用户要的是图生视频 ✓，拿到的却是无关产物 ✗
    #    而且**看不出来** ✗ ⇒ 宁可当场失败并把原因说清 ✓。
    if dual and request.first_frame:
        # ⚠️ 双流的首帧走**另一条路** ✓：不是这里的 `condition_first_frame`（那是单流的 ✓），
        #    而是 `init_dual_latents` 里把它编成 H3 的 ``cond`` 段 ✓（**已经做完了** ✓）。
        #    ⚠️ 但**先看后端自述** ✓：没声明支持却照跑 = 静默忽略首帧 ⇒ 产出与首帧无关的画面 ✓✗
        #    ⇒ 那种情况**明确拒绝** ✗（本仓判据：宁可当场失败并把原因说清 ✓）。
        state = getattr(backend, "dualStream", None) or {}
        if not state.get("firstFrame"):
            started = time.perf_counter()
            result.stageMs["condition"] = int((time.perf_counter() - started) * 1000)
            return _fail(result, "condition", conditioning_mod.ConditioningError(
                "双流后端**没有自述支持首帧条件** ✗ ⇒ 不静默按文生视频跑 ✗"
                "（要么后端把 `cond` 段接上 ✓，要么这次别给 firstFrame ✓）"))
        started = time.perf_counter()
        result.conditioning = {
            "applied": True, "mode": "h3-keyframe-cond", "imagePath": request.first_frame,
            "via": "init_dual_latents（首帧在 init 阶段编成 `cond` 段 ✓）",
        }
        note = getattr(backend, "conditioningNote", None)
        if isinstance(note, dict):
            result.conditioning["backendNote"] = note
        result.stageMs["condition"] = int((time.perf_counter() - started) * 1000)
        emit(result.event("condition", note="首帧 → `cond` 段 ✓（H3 形态 ✓）"))

    if request.first_frame and not dual:
        started = time.perf_counter()
        try:
            ensure_not_cancelled("condition")
            hook = getattr(backend, "condition_first_frame", None)
            if not callable(hook):
                raise conditioning_mod.ConditioningError(
                    f"后端 {getattr(backend, 'name', type(backend).__name__)} 未实现 "
                    f"condition_first_frame ✗ ⇒ 无法做图生视频（刻意报错，不退回文生视频 ✗）")
            latent_count = conditioning_mod.latent_frames_for(plan.frames,
                                                              request.temporal_compression)
            mask = conditioning_mod.first_frame_mask(latent_count, request.conditioning)
            conditioned = hook(latents, request.first_frame, mask, plan, request)
            if conditioned is None:
                raise conditioning_mod.ConditioningError(
                    "condition_first_frame 返回 None ✗（必须回新的 latents ✓）")
            latents = conditioned
            result.conditioning = {
                "applied": True, "mode": "first-frame", "imagePath": request.first_frame,
                "latentFrames": latent_count, "mask": mask,
                "config": request.conditioning.to_dict(),
            }
            # 后端愿意自述"实际走了哪条路"（真 VAE 编码 / 占位 ✓）就透出来 ✓ —— 不猜 ✗：
            # 不同后端实现不同（`torch` 有、`dryrun` 没有 ✓），没有就不加这个键 ✓。
            note = getattr(backend, "conditioningNote", None)
            if isinstance(note, dict):
                result.conditioning["backendNote"] = note
        except BaseException as err:  # noqa: BLE001
            result.stageMs["condition"] = int((time.perf_counter() - started) * 1000)
            return _fail(result, "condition", err)
        result.stageMs["condition"] = int((time.perf_counter() - started) * 1000)
        emit(result.event("condition", note=f"首帧条件 ✓（{latent_count} 潜帧，"
                                            f"keep={request.conditioning.keep} ✓）"))

    # ── ④ sample（**真调采样器** ✓）────────────────────────────────────────
    started = time.perf_counter()
    holder: dict[str, Any] = {"step": 0, "scale": 1.0}
    positive, negative = _branches(condition)
    guided_counter = {"steps": 0, "calls": 0}

    def on_step(step: int, sigma: float, _x: Any) -> None:
        holder["step"] = step
        note = f"σ={sigma:.4g}"
        if holder["scale"] != 1.0:
            note += f" / cfg={holder['scale']:.2f}"
        emit(result.event("sample", step=step, total=plan.steps, note=note))
        ensure_not_cancelled("sample", step)  # 每步都能停 ✓

    try:
        if dual:
            # ── 双流（H3 形态 ✓）：两条流各走各的 σ ✓，采样循环**只有一份** ✓（`h3_form` ✓）──
            # ⚠️ H3 参考实现**不做 CFG** ✓（第 111 步核到的事实 ✓）⇒ 要求引导就**明确报错** ✗：
            #    静默忽略 = 用户以为按 cfg=X 跑了、其实没有 ✓✗（本仓最忌讳那种"看不出来"✗）。
            guided = [index for index in range(plan.steps)
                      if guidance_mod.scale_for_step(index, plan.steps, request.guidance) != 1.0]
            if guided or request.guidance.enabled:
                raise StageError(
                    "sample", "双流（H3 形态）**不支持引导** ✗ —— 参考实现无 CFG ✓"
                              f"（收到 cfg={request.guidance.scale:g} ✓）")
            # ⚠️ 同理：双流的循环是**一阶欧拉** ✓（`h3_form.sample_dual_stream` ✓）——
            #    收到 heun 还照跑就是**静默换算法** ✗✗（调度器名字被无视 ✓ 而用户看不出来 ✓）。
            if str(request.sampler or "euler").lower() != "euler":
                raise StageError(
                    "sample", "双流（H3 形态）**只做一阶欧拉** ✗（参考实现如此 ✓）"
                              f"（收到 sampler={request.sampler!r} ✓ —— 不静默忽略 ✗）")
            sample_dual = getattr(backend, "sample_dual", None)
            if not callable(sample_dual):
                raise StageError("sample", "后端自述双流可用 ✓ 却没实现 `sample_dual` ✗")

            # ⚠️ 双流要的是**文本状态张量** ✓ —— 不是 `encode_text` 那个 dict ✗。
            #    2026-09-20 自检抓到：直接把 `condition`（dict ✓）递下去 ⇒ 主干把它当行张量 ⇒
            #    `AttributeError: 'dict' object has no attribute 'shape'` ✓（好在它**响亮** ✓）。
            states = condition
            if isinstance(condition, dict):
                if condition.get("negative") is not None:
                    raise StageError(
                        "sample", "双流（H3 形态）不该有 negative 条件 ✗ —— 参考实现无 CFG ✓"
                                  "（给了就说明后端按可引导的方式准备了条件 ✓ 不静默忽略 ✓）")
                states = condition.get("positive")
                if states is None:
                    raise StageError(
                        "sample", "后端回的 dict 里没有 `positive`（文本状态 ✓）✗ ⇒ 双流没法跑 ✓")

            def on_dual_step(index: int, sigma_v: float, sigma_a: float) -> None:
                emit(result.event("sample", step=index, total=plan.steps,
                                  note=f"σv={sigma_v:.4g} / σa={sigma_a:.4g}"))
                ensure_not_cancelled("sample", index)   # 每步都能停 ✓

            sampled = sample_dual(latents, plan.sigmas, states, request, on_dual_step)
            latents = {"video": sampled["video"], "audio": sampled["audio"]}
            result.sampleSteps = int(sampled["steps"])
            result.guidanceSteps = 0      # 双流=无引导 ✓（上面已拒绝非 1.0 的引导 ✓）
        else:
            def model_fn(x: Any, sigma: float) -> Any:
                """**引导就在这儿生效** ✓（引擎的采样语义 ✓，与后端无关 ✓）。

                ⚠️ 二阶采样器（heun）每步会调**两次** ⇒ 引导时每步共 **4** 次后端调用 ✓
                —— 这是 CFG + 二阶的固有代价 ✓（如实计数 ✓，见 ``guidanceCalls`` ✓）。
                """
                index = int(holder["step"])      # 0 基；on_step 在每步结束后 +1 ⇒ 正是当前步 ✓
                scale = guidance_mod.scale_for_step(index, plan.steps, request.guidance)
                holder["scale"] = scale
                guided_counter["calls"] += 1
                if scale == 1.0 or negative is None:
                    return backend.denoise(x, float(sigma), positive, request)
                positive_estimate = backend.denoise(x, float(sigma), positive, request)
                negative_estimate = backend.denoise(x, float(sigma), negative, request)
                combined, _report = guidance_mod.combine(
                    positive_estimate, negative_estimate, scale, rescale=request.guidance.rescale)
                return combined

            # 统计「实际施加了引导的步数」✓（区间外/斜坡外不算 ✓）
            for index in range(plan.steps):
                if guidance_mod.scale_for_step(index, plan.steps, request.guidance) != 1.0:
                    guided_counter["steps"] += 1
            result.guidanceSteps = int(guided_counter["steps"])

            latents, result.sampleSteps = sampler_mod.sample(
                model_fn, latents, plan.sigmas, str(request.sampler or "euler"),
                callback=on_step)
    except BaseException as err:  # noqa: BLE001
        result.stageMs["sample"] = int((time.perf_counter() - started) * 1000)
        return _fail(result, "sample", err, step=holder["step"] or None)
    result.stageMs["sample"] = int((time.perf_counter() - started) * 1000)
    emit(result.event("sample", step=result.sampleSteps, total=plan.steps))

    # ── ④b refine（**可选阶段** ✓：只有计划说"用放大器"才跑 ✓ 2026-09-24 接 ✓）────────
    # 口径（逆向 ✓）：低清一采 ⇒ latent **空间 2×** ⇒ **低噪声二采精修**（⚠️ **音频流不重采** ✗✗ ——
    # 重采音频会把音轨弄坏/丢掉 ✓✗）。计划由 :func:`app.services.engine.upscale.plan_upscale` 给出 ✓
    # （调用方先走 ``/engine/upscale-plan`` ✓ 把计划放进请求 ✓ —— 管线**不自己猜**倍率/分块 ✗）。
    if request.upscale is not None and getattr(request.upscale, "uses_upscaler", False):
        started = time.perf_counter()
        try:
            ensure_not_cancelled("refine")
            hook = getattr(backend, "refine_latents", None)
            if not callable(hook):
                raise StageError(
                    "refine", f"后端 {getattr(backend, 'name', type(backend).__name__)} 未实现 "
                              f"refine_latents ✗ ⇒ 做不了二采（**不静默按普通模式出片** ✗✗ —— "
                              f"那会让用户以为超清开着 ✓）。要么后端把二采接上 ✓，要么这次别要超清 ✓")
            # ⭐ 音频流锁定**必须是后端自述的** ✓✗：二采只重采视频流 ✓ —— 后端没声明就拒 ✗
            #    （重采了音频 ⇒ 音轨坏掉而画面看着正常 ✓，属于最坏的"看不出来"✗）。
            note = getattr(backend, "refineNote", None)
            if not isinstance(note, dict) or note.get("locksAudio") is not True:
                raise StageError(
                    "refine", "后端**没有自述「二采锁定音频流」** ✗ ⇒ 不跑二采 ✗✗"
                              "（重采音频会把音轨弄坏 ✓ 而画面上看不出来 ✓）；"
                              "请让后端声明 refineNote={'locksAudio': True, ...} ✓")
            # ⚠️ 二采要**重跑主干** ⇒ 必须把条件递下去 ✗（不然后端只能拒绝 ✓ —— 见 refine 的注释 ✓）
            latents = hook(latents, plan, request, condition=condition)
            result.refine = {"applied": True, "mode": request.upscale.mode,
                             "scale": request.upscale.scale,
                             "tiles": len(request.upscale.tiles or ()),
                             "backendNote": dict(note)}
            details = getattr(backend, "refineDetails", None)
            if isinstance(details, dict):
                result.refine["details"] = dict(details)
        except BaseException as err:  # noqa: BLE001
            result.stageMs["refine"] = int((time.perf_counter() - started) * 1000)
            return _fail(result, "refine", err)
        result.stageMs["refine"] = int((time.perf_counter() - started) * 1000)
        emit(result.event("refine", note=f"二采精修（{request.upscale.mode} ✓，音频流锁定 ✓）"))

    # ── ⑤ decode ──────────────────────────────────────────────────────────
    started = time.perf_counter()
    decoded: dict[str, Any] = {}
    try:
        ensure_not_cancelled("decode")
        decoded = backend.decode(latents, plan, request)
    except BaseException as err:  # noqa: BLE001
        result.stageMs["decode"] = int((time.perf_counter() - started) * 1000)
        return _fail(result, "decode", err)
    result.stageMs["decode"] = int((time.perf_counter() - started) * 1000)
    emit(result.event("decode"))

    # ── ⑥ write ───────────────────────────────────────────────────────────
    started = time.perf_counter()
    try:
        ensure_not_cancelled("write")
        result.outputs = backend.write(decoded, plan, request)
    except BaseException as err:  # noqa: BLE001
        result.stageMs["write"] = int((time.perf_counter() - started) * 1000)
        return _fail(result, "write", err)
    result.stageMs["write"] = int((time.perf_counter() - started) * 1000)
    result.ok = True
    emit(result.event("write", note="完成 ✓"))
    return result


def _branches(condition: Any) -> tuple[Any, Any]:
    """把 ``encode_text`` 的产出拆成 ``(正, 负)`` ✓。

    兼容两种后端写法 ✓：``{"positive":…, "negative":…}`` ✓（推荐 ✓）；
    或**直接回一个条件对象** ✓ ⇒ 视为 positive-only、负为空 ⇒ 管线**不引导** ✓（而不是报错 ✗）。
    """
    if isinstance(condition, dict):
        positive = condition.get("positive")
        if positive is None:            # dict 但没给 positive ⇒ 当成"单个条件" ✓ 不硬拆 ✗
            return condition, None
        return positive, condition.get("negative")
    return condition, None


def _fail(result: PipelineResult, stage: str, err: BaseException,
          *, step: int | None = None) -> PipelineResult:
    """把异常归一成结果 ✓ —— 取消**不是失败** ✗（前端要分开显示 ✓）。"""
    if isinstance(err, GenerationCancelled):
        result.cancelled = True
        # ⚠️ 步号必须**进结构化字段**（不能只写在文案里 ✓）：自检 ㉗ 抓到初版只把「第 3 步」
        #    放进了 message ⇒ 前端想画「停在第 3/20 步」还得去**解析中文** ✗。
        result.error = {"stage": stage, "step": step, "message": str(err), "cancelled": True}
        return result
    stage_error = err if isinstance(err, StageError) else StageError(
        stage, f"{type(err).__name__}: {err}"[:300], step=step, cause=err)
    result.error = {"stage": stage_error.stage, "step": stage_error.step,
                    "message": str(stage_error),
                    "type": type(err).__name__}
    return result
