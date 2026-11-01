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
σ 序列**真的**来自 :mod:`app.services.engine.schedules` ✓（视频默认 **flow** ✓ ——
H3 的 σ 是 flow 域 ✓，`karras` 那套 14.6 起步只留给显式要它的调用方 ✓，见 :func:`_video_sigmas` ✓），
尺寸/帧数**真的**来自 :mod:`app.services.engine.geometry` ✓ ⇒ 换后端不换算法 ✓。

## 取消（协同式 ✓）

长任务必须能停 ✓：``cancel`` 是个可调用对象（返回 ``True`` 表示要停 ✓），
在**阶段边界**与**每个采样步**检查 ✓ ⇒ 取消后抛 :class:`GenerationCancelled` ✓
（已完成阶段的信息仍会带出来 ✓，便于前端显示「停在第几步」✓）。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace as dataclass_replace
from typing import Any, Callable, Protocol

from . import conditioning as conditioning_mod
from . import geometry, guidance as guidance_mod, latent_formats, sampler as sampler_mod, schedules

__all__ = [
    "H3_FLOW_SHIFT",
    "IMAGE_STAGE",
    "QWEN21_FLOW_SHIFT",
    "QWEN21_STAGE",
    "QWEN_IMAGE_FLOW_SHIFT",
    "QWEN_IMAGE_STAGE",
    "STAGE_ORDER",
    "VIDEO_STAGE",
    "ZIMAGE_STAGE",
    "GenerationBackend",
    "GenerationCancelled",
    "GenerationPlan",
    "GenerationRequest",
    "PipelineResult",
    "StageError",
    "build_plan",
    "is_flow_schedule",
    "run_sync",
    "video_flow_shift_for",
]

#: 两条**阶段**名 ✓（= `runtime.ensure_loaded(stage=…)` 认的那两个 ✓ —— 一个概念贯穿三层 ✓）。
#: * :data:`VIDEO_STAGE` ✓：H3 双流视频（默认 ✓）；
#: * :data:`IMAGE_STAGE` ✓：SDXL **单张图片** ✓。
#: ⚠️ 名字与 `SdxlBackend.name`（``"sdxl"`` ✓）**刻意相同** ✓ ——
#: `runtime` 按 stage 选后端 ✓、`bridge` 按 stage 拼装配 ✓、出图链路按后端名分流 ✓，
#: 三处对不上就会「装的是视频后端、跑的是图片 plan」✓✗（那种错**不报错** ✓ 只是结果全错 ✓）。
VIDEO_STAGE = "h3"
IMAGE_STAGE = "sdxl"
#: ⭐ :data:`ZIMAGE_STAGE` ✓：Z-Image **单张图片** ✓（2026-10-31 新增 ✓）。
#:
#: ⚠️ 为什么**不能**并进 :data:`IMAGE_STAGE` ✗（明明都是"出一张图" ✓）：
#: 两者的 **σ 语义根本不同** ✗✗ —— SDXL 是离散扩散 ✓（σ 到 14.6 ✓、调度器是 karras 一族 ✓），
#: Z-Image 是 **flow matching** ✓（σ 恒在 [0,1] ✓、shift=3 ✓、timestep 就是 σ 本身 ✓）。
#: 合流 ⇒ 谁都可能在对方那套 σ 上跑 ✓✗：**形状全对、不报错、结果是噪声** ✓✗
#: （这正是本仓记忆里「`vae_scale` 取不到 ⇒ 静默占位」同一类错误的翻版 ✓）。
#: 而且它们的**后端也不是同一个** ✓：SDXL 走 ``SdxlBackend`` ✓，Z-Image 走 ``TorchBackend`` ✓。
ZIMAGE_STAGE = "zimage"
#: ⭐ :data:`QWEN21_STAGE` ✓：Qwen-Image **2.1** 单张图片 ✓（2026-10-31 新增 ✓）。
#:
#: ⚠️ 为什么**又**要一个新名字 ✗（已经是第三套"出一张图"了 ✓）：
#: 三套的**采样口径与组件全不同** ✗✗ ——
#: * SDXL：离散扩散 ✓ σ 到 14.6 ✓；四件套（UNet + 两个 CLIP + VAE）✓；
#: * Z-Image：flow matching ✓ shift=**3** ✓；DiT 453 键 + Qwen3-4B + Flux VAE ✓；
#: * 2.1：flow matching ✓ 但 shift=**0.69** ✓（`supported_models.QwenImage` 的
#:   ``sampling_settings = {shift: 0.69, multiplier: 1.0}`` ✓）＋ **64 通道**潜变量 ✓（其余两套 4/16 ✓）
#:   ＋ 混合文本塔（Qwen3.5-9B ✓ 24 层线性注意力 + 8 层 full attention ✓）。
#: 合流任何一对 ⇒ 在对方那套 σ / 通道上跑 ✓✗：**形状可能都对、图是噪声、一声不响** ✓✗。
QWEN21_STAGE = "qwen_image21"
#: Qwen-Image 2.1 的 flow shift ✓（**不是** :data:`schedules.FLOW_SHIFT_DEFAULT` 的 3 ✓✗）。
#: 出处：comfy ``supported_models.QwenImage`` 的 ``sampling_settings`` ✓
#: （``{"multiplier": 1.0, "shift": 0.69}`` ✓）—— 拿 3 去跑同一个 seed ⇒ 前几步就把图定死 ✓、
#: 后几步白跑 ✓✗（**不报错** ✓，只是"怎么调都不像" ✓）。
QWEN21_FLOW_SHIFT = 0.69
#: ⭐ :data:`QWEN_IMAGE_STAGE` ✓：Qwen-Image **2512**（t2i）/ **Edit 2511** 单张图片 ✓（2026-10-31 新增 ✓）。
#:
#: ⚠️ 与 :data:`QWEN21_STAGE`（``"qwen_image21"`` ✓）只差一个 `21` ✗✗ —— 但**不是**同一族 ✓：
#: * 结构：**双流 MMDiT**（6 组调制 ✓） vs 单流（2 行调制 ✓）；
#: * 潜通道：**16** ✓ vs 64 ✗；
#: * 时间口径：``1000·σ`` **不翻转** ✓ vs lumina 式翻转 ✓；
#: * flow shift：**3.1** ✓ vs 0.69 ✗；
#: * 文本塔：**Qwen2.5-VL-7B（3584 ✓、qkv 带 bias ✓、取最后一层 ✓）** vs Qwen3.5-9B（4096 ✓、取 -2 ✓）。
#:
#: ⚠️ 两族**族名比较一律 `==`** ✗ 不许 `startswith` ✓：``"qwen_image21"`` 以 ``"qwen_image"`` 开头 ✓，
#: 前缀判族会让 2.1 被 2512 的骨架接走 ✓✗（报「缺 4 组调制」，读起来像权重没下全 ✗）。
QWEN_IMAGE_STAGE = "qwen_image"
#: Qwen-Image **2512** 的 flow shift ✓。
#: 出处：官方 UI 模板 ``image_qwen_Image_2512.json`` 的 ``ModelSamplingAuraFlow`` 节点
#: ``widgets_values`` = **3.1** ✓（不是 2.1 的 0.69 ✗，也不是 Z-Image 的 3 ✓）。
#: ⚠️ **两处数字不一致，取模板那一格** ✗✓：comfy ``supported_models.QwenImage`` 的
#: ``sampling_settings`` 写的是 ``{"multiplier": 1.0, "shift": 1.15}`` ✓ —— 那是**代码默认** ✓；
#: 而官方模板在工作流里**显式挂** ``ModelSamplingAuraFlow`` 把 shift 覆盖成 **3.1** ✓
#: ⇒ 创作者点开"官方 Qwen-Image 2512"跑的就是 3.1 ✓。取 1.15 不会报错 ✓，只是 σ 曲线不同 ✓✗。
#: ⚠️ 拿 2.1 的 0.69 去跑 2512 ⇒ σ 曲线被压在前段 ✓✗：**能出图、不报错** ✓、
#: 但 50 步里有一大半几乎不动 ✓，表现为"细节糊、提示词不跟" ✓✗。
QWEN_IMAGE_FLOW_SHIFT = 3.1

#: 视频族（H3 ✓）的 flow shift ✓。出处：``h3_form.H3_SIGMA_SHIFTS["sigma_shift_video"]`` ✓
#: （σ 与音频那一路 ``sigma_shift_audio=3`` 分开 ✓ —— 两路**各在自己的噪声日程上** ✓✗）。
#: 这里存一份是为了让 :mod:`pipeline` 保持**零 torch**（plan 阶段不 import 主干 ✓）；
#: 两处一致由 ``tests/engine_video_schedule_test.py`` 的接线守卫钉住 ✓（只改一处 ⇒ 判据为假 ✓）。
H3_FLOW_SHIFT = 12.0

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
    #: σ 日程 ✓ —— 默认 **``flow``** ✓（视频族是 flow 域 ✓；四形见 :func:`_video_sigmas` ✓）。
    #: ⚠️ 2026-10-31 之前默认是 ``karras`` ✗：它的首值 14.61 会被 H3 当成 flow 的 σ 用 ✓✗
    #: ⇒ ``t = 1−σ`` 落到 −13.6 ✓、**不报错、出噪声** ✓✗（依据见 :func:`_video_sigmas` ✓）。
    #: 注：图片三族（sdxl / zimage / qwen_image21）**不读这个字段** ✓ —— 它们的 shift 各自写死在
    #: ``_fill_*_plan`` 里 ✓（同一个字段喂四套不同 σ 语义 = 上一次"合流"事故的翻版 ✗）。
    schedule: str = "flow"
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
    #: 图生视频的**尾帧** ✓（fl2v ✓）—— 与 ``first_frame`` 同一套 ``cond`` 段机制 ✓，
    #: 区别只是关键帧下标取 **``frames − 1``** 而不是 ``0`` ✓（见 `torch_backend._dual_keyframes` ✓）。
    #: ⚠️ 只对**双流**后端有意义 ✓：单流只有 ``condition_first_frame``（掩码混第 0 潜帧 ✓）⇒
    #: 给单流后端尾帧会**明确失败** ✗，不静默忽略 ✓（忽略 = 产出与尾帧无关的画面且看不出来 ✗）。
    #: ⚠️ 也**追加在末尾** ✓（同 ``stage`` 的理由 ✓）。
    last_frame: str | None = None
    #: 文本段**逐 token 的 adaLN 行号档** ✓（``0`` = video 行 ✓、``1`` = text 行 ✓）。
    #: 由 :meth:`TorchBackend.encode_text` 里 `h3_encode.encode_condition` 算出的
    #: ``H3Condition.tags`` 转手填进来 ✓（`pipeline` 在 encode 之后、采样之前 ✓）——
    #: ⚠️ 为什么落**请求**而不是落 condition ✗：它是**逐 token 的整数行号** ✓ 不是特征 ✗，
    #: 塞进特征张量就只剩"多一通道"这种暗约定 ✓✗（一改文本长度就对不上 ✓）；
    #: ⚠️ 为什么不落 ``GenerationPlan`` ✗：plan 是"请求 → 具体数字"的**纯函数产物** ✓，
    #: 而这里要的是**编码的副产品** ✓（依赖分词器与视觉塔有没有装 ✓）。
    #: ``None`` ⇒ 不传 ✓（与"全 1"逐位相同 ✓ —— 纯文本提示词没有视觉占位 ✓）。
    #: ⚠️ 追加在**末尾** ✓（同 ``stage`` / ``last_frame`` 的理由 ✓）。
    text_token_tags: tuple[int, ...] | None = None

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
    if stage == ZIMAGE_STAGE:
        _fill_zimage_plan(request, plan)
        return _finish_plan(request, plan, weights_bytes)
    if stage == QWEN21_STAGE:
        _fill_qwen21_plan(request, plan)
        return _finish_plan(request, plan, weights_bytes)
    if stage == QWEN_IMAGE_STAGE:
        _fill_qwen_image_plan(request, plan)
        return _finish_plan(request, plan, weights_bytes)
    if stage != VIDEO_STAGE:
        raise StageError(
            "plan", f"未知 stage {request.stage!r} ✗；可用："
            f"{sorted((VIDEO_STAGE, IMAGE_STAGE, ZIMAGE_STAGE, QWEN21_STAGE, QWEN_IMAGE_STAGE))} ✓"
            "（⚠️ 别随手编一个名字 ✓ —— 它决定选哪个后端 ✓✗）")

    plan.width, plan.height = geometry.size_for_megapixels(
        max(0.05, float(request.megapixels)), request.ratio)
    plan.frames = geometry.snap_frames(request.seconds, fps=plan.fps)
    plan.steps = int(request.steps)

    # ⚠️ 潜帧数必须**先**算 ✗（``schedule="flow:auto"`` 的 shift 按它自适应 ✓）——
    #    这个先后不是排版问题：反过来的话 `flow:auto` 只能拿到 ``None`` 而报错 ✓✗。
    if request.temporal_compression:
        plan.latent_frames = geometry.latent_frames(plan.frames,
                                                    temporal_compression=int(request.temporal_compression))
    else:
        # 诚实标注：不猜压缩比（见 geometry 的模块注释 ✓）—— 后端可以自己算 ✓
        plan.warnings.append("未给 temporalCompression ⇒ 潜空间帧数未知（不影响采样，解码时后端自行决定 ✓）")

    try:
        plan.sigmas = _video_sigmas(request, plan)
    except (ValueError, TypeError) as err:
        raise StageError("plan", f"σ 调度不可用：{err}", cause=err) from err
    plan.timesteps = schedules.timesteps_for(plan.sigmas)
    plan.warnings.append(f"{plan.frames} 帧 @{plan.fps}fps ≈ {plan.frames / max(1, plan.fps):.2f}s"
                         f"（已吸附到 {geometry.H3_FRAME_GRID}k+{geometry.H3_MIN_FRAMES} 网格 ✓）")
    return _finish_plan(request, plan, weights_bytes)


def is_flow_schedule(spec: object) -> bool:
    """这个 ``schedule`` 字符串是不是 **flow 族** ✓（``flow`` / ``flow:<数>`` / ``flow:auto`` ✓）。

    单独一个判定函数是因为**两个**消费点要问同一句话 ✓ —— video plan（:func:`_video_sigmas` ✓）
    与二采（:func:`torch_backend._second_pass` ✓）；任何一处自己写 `startswith("flow")` ✓
    都会在"`flow` 三个字母不带头"那几形上分叉 ✓✗。
    """
    text = str(spec or "").strip().lower()
    return text in ("", "flow", "flow:", "flow:default") or text.startswith("flow:")


def video_flow_shift_for(request: GenerationRequest, *, width: int | None = None,
                         height: int | None = None, latent_frames: int | None = None) -> float:
    """按 ``request`` 算视频的 flow shift ✓ —— **flow 字符串的唯一解析处** ✓。

    ``schedule`` 三形（大小写不敏感 ✓）：
    ① ``"flow"`` / ``"flow:default"`` / 空 ⇒ :data:`H3_FLOW_SHIFT`（12.0 ✓，**默认** ✓）；
    ② ``"flow:<数>"`` ⇒ 该数 ✓（扫 shift 直接写 `flow:6` ✓）；
    ③ ``"flow:auto"`` ⇒ 按**序列长**与**潜帧数**自适应 ✓（``width/height/latent_frames`` 给了就用 ✓，
       没给就从 ``request`` 自己推 ✓；推不出 ⇒ **报错** ✗ —— 不猜压缩比 ✗）。
    非 flow 族 ⇒ 抛 ``ValueError`` ✓（要不要报警由调用方决定 ✓ —— 这里**不**静默回落 ✗）。
    """
    spec = str(request.schedule or "flow").strip().lower()
    if spec in ("", "flow", "flow:", "flow:default"):
        return H3_FLOW_SHIFT
    if not spec.startswith("flow:"):
        raise ValueError(f"{spec!r} 不是 flow 族 ✓（可用：flow / flow:<数> / flow:auto ✓）")
    tail = spec[len("flow:"):].strip()
    if tail not in ("auto", "adaptive"):
        try:
            return float(tail)
        except ValueError:
            raise ValueError(
                f"schedule={request.schedule!r} 的 shift 读不出来 ✗；可用形："
                "flow / flow:<数> / flow:auto / karras / exponential / normal / ddim_uniform ✓") from None

    if not request.temporal_compression:
        raise ValueError(
            "schedule=flow:auto 要按潜帧数自适应 ✓ ⇒ 必须给 temporalCompression ✓"
            "（H3 视频是 4 ✓）；不给就说明「不知道压缩比」✓，那也没法定 shift ✓")
    if width is None or height is None:
        width, height = geometry.size_for_megapixels(max(0.05, float(request.megapixels)),
                                                    request.ratio)
    if not latent_frames:
        fps = int(request.fps or geometry.H3_FPS)
        latent_frames = geometry.latent_frames(geometry.snap_frames(request.seconds, fps=fps),
                                               temporal_compression=int(request.temporal_compression))
    seq_len = _video_token_count(int(width), int(height))
    return schedules.flow_shift_for_sequence(seq_len, frames=int(latent_frames))


def _video_sigmas(request: GenerationRequest, plan: GenerationPlan) -> list[float]:
    """视频阶段的 σ 日程 ✓ —— flow 族走 :func:`video_flow_shift_for` ✓，其余走旧路（带报警 ✓）。

    ⚠️ **为什么默认从 ``karras`` 改成 ``flow``**（本轮改 ✓，依据都在本仓 ✓✗）：
    H3 采样器把 σ 当 **flow 域**用 ✓ —— ``h3_form`` 里是 ``t = 1 − σ`` ✓、欧拉步是
    ``x += (σ_next − σ) · v`` ✓、音频那一路还要 ``time_shift_sigma(σ_v, 12, 3)`` ✓；
    而 ``karras`` 的**首值是 14.6146** ✓（EDM 域的 σ 上界 ✓）⇒ ``t = 1 − 14.6 = −13.6`` ✗
    ⇒ 起始噪声比模型见过的任何一档都高 ✓、音频 shift 的分母还会翻号 ✓（``12 + 14.6·(1−12) < 0`` ✓）
    ⇒ **不报错、出噪声/糊片** ✓✗。旧判据没抓到它，是因为 ``tests`` 里各处都**手动**喂
    ``[1.0, …, 0.0]`` ✓（``h3_edit`` 那些脚本口径 ✓）—— **plan 那一层的默认值从没被真跑过** ✓✗。

    旧路（``karras`` / ``exponential`` / ``normal`` / ``ddim_uniform``）**照跑** ✓ 但会往
    ``plan.warnings`` 写一句"域不重叠" ✓ —— 想复现旧行为就显式传，别让它静默 ✓✗。
    """
    spec = str(request.schedule or "flow").strip().lower()
    if is_flow_schedule(spec):
        return schedules.flow_sigmas_for_steps(plan.steps, shift=video_flow_shift_for(
            request, width=plan.width, height=plan.height, latent_frames=plan.latent_frames))

    sigmas = schedules.sigmas_for(plan.steps, spec)
    if max(sigmas) > 1.0:
        plan.warnings.append(
            f"σ 调度 {spec!r} 的首值 {max(sigmas):.4g} 超出 flow 的 [0,1] ✓ ⇒ H3 主干会收到 "
            f"t = 1−σ = {1.0 - max(sigmas):.4g} ✓（**不报错**但起点比训练见过的都高 ✗）。"
            "视频族的默认已改成 flow（shift 12 ✓，见 `pipeline.H3_FLOW_SHIFT` ✓）——"
            "这里照你要的跑，只是如实标出域不重叠 ✓")
    return sigmas


def _video_token_count(width: int, height: int, *, patch: int = 2) -> int:
    """``flow:auto`` 要的**序列长**（token 数 ✓）—— ``(h_lat // patch) · (w_lat // patch)`` ✓。

    ⚠️ 是 **token** 数 ✗ 不是像素数、也不是潜像素数 ✗（拿错 ⇒ α 大一个量级 ⇒ 前几步就冲到
    σ≈1 ⇒ 等于没采 ✓✗）。两个常数各有出处 ✓：空间 16× 见 :func:`latent_formats.get_latent_format`
    的 ``h3_video`` ✓；patch 2 见 ``h3_form.H3_TRUNK_DEFAULTS["patch_size"]``（= (1,2,2) ✓，
    时间维不 patch ✓ ⇒ 每帧一组 token ✓）。
    """
    side = int(latent_formats.get_latent_format("h3_video").spatial_downscale_ratio)
    latent_h, latent_w = height // side, width // side
    if latent_h < patch or latent_w < patch:
        raise ValueError(f"{width}×{height} 太小 ✓（潜空间 {latent_w}×{latent_h} 装不下一个 "
                         f"{patch}×{patch} token ✗）")
    return (latent_h // patch) * (latent_w // patch)


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


def _fill_zimage_plan(request: GenerationRequest, plan: GenerationPlan) -> None:
    """Z-Image **单张图片**的 plan ✓（flow matching ✓ 不是 SDXL 的离散格 ✗）。

    五条口径都与**官方模板**对齐 ✓（``comfyui_workflow_templates_json/templates/image_z_image_turbo.json`` ✓）：

    ① 尺寸 **1024×1024** 起步 ✓（模板 ``EmptySD3LatentImage(1024, 1024, 1)`` ✓）并吸附到 **16 的倍数** ✓
       （VAE 下采样 8 ✓ × DiT patch 2 ✓ ⇒ 边长不整除 16 会在 patchify 前就被切掉一块 ✓✗）；
    ② **帧数恒 1** ✓、``fps = 0`` ✓（图没有帧率概念 ✓）；
    ③ σ **必须是 flow 那套** ✓（``schedules.flow_sigmas_for_steps`` ✓ ⇒ σ∈[0,1] ✓、shift=3 ✓）；
    ④ ``timesteps = sigmas`` ✓（Z-Image 的 ``multiplier = 1.0`` ✓ ⇒ timestep 就是 σ ✓，
       **不要**乘 1000 ✗ —— DiT 内部自己 ``(1−σ)×1000`` ✓）；
    ⑤ **步数/引导**：模板是 ``steps=8, cfg=1.0`` ✓（turbo 8 步蒸馏 ✓）；这里**按请求走** ✓，
       只把「与模板不同」如实写进 ``warnings`` ✓（不静默替用户改 ✗）。
    """
    # ⚠️ Z-Image turbo 的**官方画布就是 1024²** ✓（不是"越大越好" ✗ —— 训练档位外的尺寸是拿
    #    没训过的区域换画质 ✓✗，症状是构图崩/多主体重复 ✓ 且**不报错** ✓）。请求没给就按 1024² ✓。
    wanted_mp = float(request.megapixels)
    # ⚠️ `GenerationRequest.megapixels` 的默认是 **0.65** ✓（视频档位 ✓）—— 那个数对 Z-Image
    #    没有意义 ✓✗（它是 H3 的口径 ✓）。这里不能拿 `>0.05` 判"没给"✗（默认永远 >0.05 ✓✗），
    #    只能拿"**等于默认值**"判 ✓，并且**如实报改** ✓（不静默换用户没要的画布 ✓）。
    if abs(wanted_mp - 0.65) <= 0.01 * 0.65:
        wanted_mp = 1.0
        plan.warnings.append(
            "像素预算仍是 `GenerationRequest` 的默认 0.65 MP ✓（视频口径 ✗）⇒ 已按 Z-Image turbo "
            "官方模板换成 **1024×1024（1.0 MP）** ✓"
            "（出处：`image_z_image_turbo.json` 的 `EmptySD3LatentImage(1024, 1024, 1)` ✓）；"
            "要保持视频那档请显式传 megapixels ✓")
    want_w, want_h = geometry.size_for_megapixels(wanted_mp, request.ratio)
    grid = 16                       # 8（VAE）× 2（patch）✓
    plan.width = max(grid, int(round(want_w / grid)) * grid)
    plan.height = max(grid, int(round(want_h / grid)) * grid)
    if (plan.width, plan.height) != (int(want_w), int(want_h)):
        plan.warnings.append(
            f"尺寸已吸附到 {grid} 的倍数（VAE 1/8 ✓ × patch 2 ✓）：{int(want_w)}×{int(want_h)} ⇒ "
            f"**{plan.width}×{plan.height}** ✓（不整除会在 patchify 前被裁掉一条 ✓✗）")

    plan.frames = 1
    plan.fps = 0                    # 单张图 ✓ 没有帧率 ✓
    plan.steps = int(request.steps)
    shift = schedules.FLOW_SHIFT_DEFAULT
    plan.sigmas = schedules.flow_sigmas_for_steps(plan.steps, shift=shift)
    # ⚠️ 这里**不是** `schedules.timesteps_for` ✗（那是 σ/(σ+1) 的日志映射 ✓）——
    #    Z-Image 的 multiplier=1.0 ✓ ⇒ 喂给 DiT 的 timestep **就是 σ** ✓。
    plan.timesteps = [float(value) for value in plan.sigmas]
    plan.warnings.append(
        f"{plan.width}×{plan.height} 单张图片 ✓、{plan.steps} 步 **flow matching** ✓"
        f"（shift={shift:g} ✓；σ {plan.sigmas[0]:.4g} ⇒ 0 ✓；timestep 即 σ ✓ 不乘 1000 ✗）")
    # ⚠️ CFG 在**引导配置**里 ✓（`request.guidance.scale` ✓），没有 `request.cfg` ✗。
    cfg_scale = float(getattr(request.guidance, "scale", 1.0) or 1.0)
    if plan.steps != 8 or abs(cfg_scale - 1.0) > 1e-6:
        plan.warnings.append(
            f"与官方 turbo 模板的默认档不同（模板：steps=8 / cfg=1.0 ✓；本次 steps={plan.steps} / "
            f"cfg={cfg_scale:g} ✓）—— 蒸馏模型在别的档位上**能跑但未必更好** ✓，"
            "这里只报不改 ✗")


def _fill_qwen21_plan(request: GenerationRequest, plan: GenerationPlan) -> None:
    """Qwen-Image **2.1** 单张图片的 plan ✓（flow matching ✓ 但 shift 与 Z-Image **不同** ✗✗）。

    六条口径：

    ① 尺寸吸附到 **16 的倍数** ✓ —— VAE 下采样 16× ✓（`latent_formats.qwen_image21`
       ``spatial_downscale_ratio=16`` ✓）；而 2.1 的 DiT **没有 patchify** ✗
       （``img_in = Linear(64 → dim)`` ✓，见 `dit_qwen21` 模块头 ✓）⇒ **不再乘 2** ✗✗
       （Z-Image 那条要 16 = 8×2 ✓，这一条只要 16 就够 ✓，多乘一次只是白裁边 ✗）；
    ② 帧数恒 1 ✓、``fps = 0`` ✓；
    ③ σ 走 flow ✓ 但 **shift=0.69** ✓（:data:`QWEN21_FLOW_SHIFT` ✓）—— 套 3.0 ⇒
       σ 曲线整体被推向 1 ✓✗：前几步就把构图定死 ✓、后几步几乎不动 ✓，**不报错** ✓；
    ④ ``timesteps = sigmas`` ✓（``multiplier = 1.0`` ✓ ⇒ 喂给 DiT 的 timestep 就是 σ ✓、
       **不乘 1000** ✗；DiT 内部自己 ``(1−σ)×1000`` ✓）；
    ⑤ **潜变量 64 通道** ✓ —— 这一条不在 plan 里（后端按权重事实填 ✓），但记在这：
       4/16 通道的 VAE 接上去**形状当场不对** ✓（能早报 ✓），而通道对了、**反缩放没做** ⇒
       图发灰且不报错 ✓✗（见 `vae_wan` 模块头第 ④ 条 ✓）；
    ⑥ 步数/引导**按请求走** ✓，只把「与官方默认档不同」如实写进 ``warnings`` ✓（不静默改 ✗）。
    """
    wanted_mp = float(request.megapixels)
    # ⚠️ 与 Z-Image 同一条纪律 ✗：`GenerationRequest.megapixels` 的默认是 **0.65** ✓（视频档位 ✓），
    #    对 2.1 没有意义 ✗ ⇒ 只能拿"**等于默认值**"判 ✓ 并**如实报改** ✓（不静默换画布 ✗）。
    default_mp = 0.65
    if abs(wanted_mp - default_mp) <= 0.01 * default_mp:
        #: 2.1 官方示例档 ≈ **1.76 MP** ✓（1:1 时正是 **1328×1328** ✓；`image_qwen_image.json` 一族模板同档 ✓）。
        #: ⚠️ 长宽比仍**跟着 `request.ratio`** 走 ✓（16:9 的 1.76 MP ⇒ 1792×1024 ✓，不是硬塞成方的 ✗）。
        wanted_mp = 1.76
        plan.warnings.append(
            "像素预算仍是 `GenerationRequest` 的默认 0.65 MP ✓（视频口径 ✗）⇒ 已按 Qwen-Image 官方"
            "示例档换成 **1.76 MP** ✓（1:1 时即官方 1328×1328 ✓；长宽比仍跟 ratio ✓）；"
            "要保持视频那档请显式传 megapixels ✓")
    grid = 16                       # VAE 16× ✓（**没有** patch ×2 ✗✗）
    #: ⚠️ `size_for_megapixels` 的**默认吸附步长是 32** ✗（全局 `RESOLUTION_MULTIPLE` ✓，
    #: 那是给有 patchify 的族用的 ✓）⇒ 这里必须**显式传 16** ✗✗：走默认会让 2.1 也被吸到 32 ✓
    #: —— 不报错 ✓、只是每张白裁一条边 ✓（"按 16 算"就成了空话 ✓✗）。
    want_w, want_h = geometry.size_for_megapixels(wanted_mp, request.ratio, multiple=grid)
    plan.width = max(grid, int(round(want_w / grid)) * grid)
    plan.height = max(grid, int(round(want_h / grid)) * grid)
    if (plan.width, plan.height) != (int(want_w), int(want_h)):
        plan.warnings.append(
            f"尺寸已吸附到 {grid} 的倍数（VAE 1/16 ✓；这一族**没有** patchify ⇒ 不乘 2 ✓）："
            f"{int(want_w)}×{int(want_h)} ⇒ **{plan.width}×{plan.height}** ✓")

    plan.frames = 1
    plan.fps = 0                    # 单张图 ✓ 没有帧率 ✓
    plan.steps = int(request.steps)
    shift = QWEN21_FLOW_SHIFT
    plan.sigmas = schedules.flow_sigmas_for_steps(plan.steps, shift=shift)
    plan.timesteps = [float(value) for value in plan.sigmas]
    plan.warnings.append(
        f"{plan.width}×{plan.height} 单张图片 ✓、{plan.steps} 步 **flow matching** ✓"
        f"（shift={shift:g} ✓ **不是** Z-Image 的 3 ✓✗；σ {plan.sigmas[0]:.4g} ⇒ 0 ✓；"
        "timestep 即 σ ✓ 不乘 1000 ✗）")
    cfg_scale = float(getattr(request.guidance, "scale", 1.0) or 1.0)
    if cfg_scale == 1.0:
        # ⚠️ 2.1 是**非蒸馏**模型 ✓ ⇒ cfg=1.0（不引导）会出「糊、构图散」的图 ✓✗ 且不报错 ✓；
        #    官方档 2.5~4.0 ✓。这里**只报不改** ✗（用户的引导配置是用户的 ✓）。
        plan.warnings.append(
            "引导 cfg=1.0 ⇒ **等效不引导** ✓：2.1 不是在 cfg=1 上蒸馏的 ✗，官方档 2.5~4.0 ✓ —— "
            "不引导**能跑** ✓、但构图与细节会明显软 ✓（这里只报不改 ✗）")


def _fill_qwen_image_plan(request: GenerationRequest, plan: GenerationPlan) -> None:
    """Qwen-Image **2512**（t2i）/ Edit **2511** 单张图片的 plan ✓（flow matching ✓ shift=**3.1** ✓）。

    与 :func:`_fill_qwen21_plan` 的**四格差别**（都对着 2512 官方模板
    ``comfyui_workflow_templates_json/templates/image_qwen_Image_2512.json`` 核过 ✓）：

    | | 2.1（2509） ✓ | **2512** ✓ |
    |---|---|---|
    | flow shift | **0.69** ✓ | **3.1** ✓（``ModelSamplingAuraFlow`` 的 ``shift`` 格 = 3.1 ✓） |
    | 分辨率吸附 | 16 ✓ | **16** ✓（同一格数，但**理由不同** ✗：2.1 是 VAE 16×；2512 是 VAE 8× × DiT patch 2×2 ✓） |
    | 潜通道 | 64 ✓ | **16** ✓（16 = 8×8 拍平前的通道 ✓，见 `dit_qwen_image` ✓） |
    | 官方步/引导 | 2.5~4 档 ✓ | **50 步 / cfg 4.0 / euler / simple** ✓（模板 ``KSampler`` 的 ``widgets_values`` ✓） |

    ⚠️ **shift 抄错这一格是"能出图、但怎么调都不像"** ✓✗：σ 曲线整体被压/被推，
    前几步就把构图定死或后几步几乎不动 ✓；**不报错** ✓，只是提示词跟随变差 ✓✗。
    2.1 的 0.69 与这里的 3.1 **互相不能顶替**（两族连时间口径都相反 ✓✗）。

    ⚠️ 尺寸为什么是 **16 的倍数**（不是 8 ✗）：VAE 下采样 **8×** ✓ ⇒ 1328/8 = 166 ✓；
    而 DiT 内部还要按 **2×2** 拍平（``img_in`` 吃 16×2×2 = 64 ✓，见 `dit_qwen_image` 模块头 ✓）
    ⇒ **潜变量边长必须是偶数** ✓ ⇒ 像素必须是 16 的倍数 ✓✗。只吸到 8 ⇒ 潜边长 83（奇数）✓✗
    ⇒ DiT 内部拍平当场形状错（这个会报 ✓），但报的是 ``img_in`` 的宽度，指不到尺寸 ✗。
    """
    wanted_mp = float(request.megapixels)
    default_mp = 0.65                    # `GenerationRequest` 的默认是视频档 ✓（同 Z-Image / 2.1 的理由 ✓）
    if abs(wanted_mp - default_mp) <= 0.01 * default_mp:
        #: 2512 官方模板 ``EmptySD3LatentImage`` = **1328×1328** ✓ ⇒ 1.76 MP ✓（与 2.1 同一档位 ✓）。
        wanted_mp = 1.76
        plan.warnings.append(
            "像素预算仍是 `GenerationRequest` 的默认 0.65 MP ✓（视频口径 ✗）⇒ 已按 Qwen-Image 2512 "
            "官方模板换成 **1.76 MP** ✓（1:1 时即 1328×1328 ✓；长宽比仍跟 ratio ✓）")
    grid = 16                            # VAE 8× ✓ × DiT patch 2× ✓（理由与 2.1 不同，见 docstring ✓）
    want_w, want_h = geometry.size_for_megapixels(wanted_mp, request.ratio, multiple=grid)
    plan.width = max(grid, int(round(want_w / grid)) * grid)
    plan.height = max(grid, int(round(want_h / grid)) * grid)
    if (plan.width, plan.height) != (int(want_w), int(want_h)):
        plan.warnings.append(
            f"尺寸已吸附到 {grid} 的倍数（VAE 1/8 ✓ × DiT patch 2 ⇒ 潜边长必须是偶数 ✓）："
            f"{int(want_w)}×{int(want_h)} ⇒ **{plan.width}×{plan.height}** ✓")

    plan.frames = 1
    plan.fps = 0                         # 单张图 ✓
    plan.steps = int(request.steps)
    shift = QWEN_IMAGE_FLOW_SHIFT
    plan.sigmas = schedules.flow_sigmas_for_steps(plan.steps, shift=shift)
    plan.timesteps = [float(value) for value in plan.sigmas]
    plan.warnings.append(
        f"{plan.width}×{plan.height} 单张图片 ✓、{plan.steps} 步 **flow matching** ✓"
        f"（shift={shift:g} ✓ **不是** 2.1 的 0.69、也不是 Z-Image 的 3 ✓✗；σ {plan.sigmas[0]:.4g} ⇒ 0 ✓；"
        "timestep 即 σ ✓ 不乘 1000 ✗）")
    cfg_scale = float(getattr(request.guidance, "scale", 1.0) or 1.0)
    if cfg_scale == 1.0:
        plan.warnings.append(
            "引导 cfg=1.0 ⇒ **等效不引导** ✓：2512 不是在 cfg=1 上蒸馏的 ✗，官方模板档是 **cfg 4.0 / 50 步** ✓ "
            "—— 不引导**能跑** ✓、但构图与细节会明显软 ✓（这里只报不改 ✗）")


def _finish_plan(request: GenerationRequest, plan: GenerationPlan,
                 weights_bytes: int | None) -> GenerationPlan:
    """两条路**共用**的收尾 ✓（相对耗时系数 ✓ / 超清计划 ✓ / 显存估算 ✓）。

    ⚠️ 刻意**不含尺寸与 σ** ✗ —— 那两样两条路口径不同 ✓，各自在前面算好了 ✓。
    """
    # ── 相对耗时系数（**可精确算** ✓：步数 × 采样器 × 引导 ✓）────────────────
    # heun 每个区间调模型 2 次 ✓；CFG 让"施加引导的那些步"再翻倍 ✓ ⇒ 引导只需要按**生效比例**算 ✓：
    # cost = (步数 + 生效步数) × 采样器倍数 ✓（全程引导时正好 = 步数 × 2 × 倍数 ✓）。
    # ⚠️ 采样器倍数**归 `sampler` 管** ✓（唯一真源 :data:`sampler_mod.MODEL_CALLS` ✓）——
    #    这里再手写一次 `== "heun"` 就是第二份账 ✓✗：新增 `sa_solver`/`dpmpp_2m`/`lms` 时它不会跟着变 ✓。
    sampler_multiplier = sampler_mod.model_calls_per_step(str(request.sampler or "euler"))
    config = request.guidance
    guided_steps = sum(1 for index in range(plan.steps)
                       if guidance_mod.scale_for_step(index, plan.steps, config) != 1.0)
    plan.cost_factor = float((plan.steps + guided_steps) * sampler_multiplier)
    if config.enabled:
        # ⚠️ 引导的**代价**必须提前说 ✓（CFG 每步两次模型调用 ⇒ 二阶采样器就是每步四次 ✗）
        plan.warnings.append(
            f"引导 cfg={config.scale:g}：{plan.steps} 步里有 {guided_steps} 步生效，"
            f"相对耗时约 ×{plan.cost_factor / max(1.0, float(plan.steps)):.2f}"
            + (f"（{str(request.sampler or 'euler')} 每步 ×{sampler_multiplier:g} 次模型调用 ✗）"
               if sampler_multiplier != 1.0 else "")
            + (f"（重标定 {config.rescale:g} ✓）" if config.rescale else ""))
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


def _backend_phase(backend: GenerationBackend, phase: str) -> Any:
    """按阶段**换装** ✓（只有登记过链的后端才动 ✓ —— 别的族没有这个方法 ⇒ 返回 None ✓）。

    为什么这件事归管线 ✗（不归后端自己 ✓）：**阶段边界只有管线知道** ✓ —— 后端不知道自己
    何时被 encode 用、何时被 sample 用 ✓✗。后端只提供「进这个阶段要哪几件在场」这一条事实 ✓
    （`TorchBackend.PHASE_PARTS` ✓）。

    ⚠️ 3 件之和装得下的机器**不受影响** ✓：`TorchBackend` 没登记链 ⇒ `enter_phase` 返回 None ✓，
    这里连一次判断都不多付 ✓。
    """
    hook = getattr(backend, "enter_phase", None)
    if not callable(hook):
        return None
    step = hook(phase)
    #: ⚠️ **换装之后、回到管线这一层再清一次显存** ✓✗（2026-11-01 端到端实测 ✓）。
    #: 为什么后端自己末尾那次不够 ✗✗：`enter_phase` 调 `load_weights` 时，**装载栈里的中转张量
    #: 还没退** ✓ ⇒ allocator 认定「有人在用」⇒ `empty_cache()` 一个字节不还 ✓✗。
    #: 实测：`enter_phase` 末尾自己清过一轮 ✓（已经还掉了 `text` 那份 16.2 GiB ✓），
    #: 可 `reserved` 仍比 `allocated` 多 **16.2 GiB** ✓；等栈完全退干净、由这一层再清一次 ⇒
    #: 20390 → **4332 MiB** ✓、`free` 1146 → **17457 MiB** ✓。
    #: 差这一步的后果很实在 ✓：下一次 `decide_for` 量到的余量是"假的紧"✓ ⇒ 越换越保守 ✓✗。
    trim = getattr(backend, "_trim_cuda", None)
    if callable(trim):
        try:
            trim()
        except BaseException:  # noqa: BLE001 —— 收尾清理**绝不许**把采样带崩 ✓
            pass
    return step


def _backend_tick(backend: GenerationBackend, phase: str, *, step: int = 0,
                  every: int = 4) -> None:
    """运行期**探压** ✓（每 ``every`` 步一次 ✓）—— 没有 ``tick`` 的后端直接跳过 ✓。

    ⚠️ 为什么归管线 ✗：**只有管线知道"跑到第几步了"** ✓（同 `_backend_phase` 的理由 ✓）。
    后端只负责"读到压力之后怎么办"（`TorchBackend.tick` ✓）。

    ⚠️ 为什么**吞掉异常** ✗✗：这是**观测** ✓，不是业务 ✓ —— 探压失败（驱动抽风 / 别的进程把卡
    弄没 ✓）绝不许把一次已经跑到第 20 步的采样弄死 ✓。真实故障会在真正用卡时以它自己的样子
    报出来 ✓，不该由观测代它发火 ✓✗。

    ⚠️ 为什么每 4 步而不是每步 ✗：`cudaMemGetInfo` 本身很便宜 ✓，但 `tick` 到硬阈值时会
    `gc.collect()` ＋ `empty_cache()` ✓ —— 那是**毫秒级到百毫秒级**的停顿 ✓，每步都做就是在给
    采样加税 ✓✗。4 步的粒度足够在任何一步 OOM 之前看到趋势 ✓。
    """
    tick = getattr(backend, "tick", None)
    if not callable(tick):
        return
    if every > 0 and step % every:
        return
    try:
        tick(phase)
    except Exception:  # noqa: BLE001 - 观测不许拖垮任务 ✓
        return


def _noise_sampler_for(backend: GenerationBackend, request: GenerationRequest,
                       holder: dict[str, Any]) -> Callable[[float, float], Any] | None:
    """随机采样器（``euler_ancestral`` / ``sa_solver`` ✓）每步回注的噪声 ✓。
    （确定性采样器 ⇒ 回 ``None`` ✓ —— 它们压根不认识这个参数 ✓，给了也白给 ✗。）

    ⚠️ 三条口径，都是"不报错但结果不对"那一类 ✗✗：

    ① **后端没实现 ⇒ 明确报错** ✓✗：静默按 ODE 跑 = 用户选了「随机档」却拿到确定性轨迹 ✓
       —— 同一 seed 每次一样、画面还挺正常 ✓，正是本仓最忌讳的"看不出来"✗；
    ② **每步的噪声必须不同** ✓（``nonce`` 递增 ✓）：同一条噪声反复注入 ⇒ 画面出现**结构性条纹** ✓✗，
       不报错、只在真机上看得见 ✓（:func:`dryrun.noise_like` 的 ``nonce`` 就为这个 ✓）；
    ③ **形状/设备/精度一律问当前的 x** ✓（采样器只递得出 σ 两格 ✗ ⇒ 这里从 ``holder["x"]`` 取 ✓
       —— 每步 :func:`run_sync` 的 ``on_step`` 会刷新它 ✓）。
    """
    name = str(request.sampler or "euler").lower()
    if name not in sampler_mod.STOCHASTIC:
        return None
    hook = getattr(backend, "noise_like", None)
    if not callable(hook):
        raise StageError(
            "sample",
            f"采样器 {name} 是**随机采样器** ✗，但后端 "
            f"{getattr(backend, 'name', type(backend).__name__)} 没实现 `noise_like` ✗ "
            f"⇒ 退化成纯 ODE 只是「看着能跑」✓✗（刻意不静默退化 ✗）")
    counter = {"nonce": 0}

    def noise_sampler(sigma_from: float, sigma_to: float) -> Any:
        counter["nonce"] += 1
        return hook(holder["x"], request, nonce=int(counter["nonce"]))

    return noise_sampler


def run_sync(request: GenerationRequest, backend: GenerationBackend, *,
             on_event: Callable[[dict[str, Any]], None] | None = None,
             cancel: CancelFn | None = None,
             on_decoded: Callable[[dict[str, Any]], None] | None = None) -> PipelineResult:
    """**同步**跑完一次生成（FastAPI 的同步端点跑在线程池里 ✓ ⇒ 不会卡事件循环 ✓）。

    ``on_decoded``：⑤ decode 之后回调一次 ✓（收到 decoded ✓）—— 多段成片
    （``app.services.engine.chain`` ✓）靠它拿**无损**帧张量 ✓；不给就完全没这回事 ✓。

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
        _backend_phase(backend, "encode")      # 分时驻留：此时只该有 TE 在场 ✓（见 PHASE_PARTS ✓）
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

    if dual and request.last_frame:
        # 尾帧（fl2v ✓）走**同一套** `cond` 机制 ✓，下标 = `frames − 1` ✓。
        # ⚠️ 同样**先看后端自述** ✗：没声明却照跑 = 尾帧被静默丢掉 ⇒ 产物末端根本不像尾帧 ✓✗
        state = getattr(backend, "dualStream", None) or {}
        if not state.get("lastFrame"):
            return _fail(result, "condition", conditioning_mod.ConditioningError(
                "双流后端**没有自述支持尾帧条件** ✗ ⇒ 不静默丢掉尾帧 ✗"
                "（要么后端把尾帧 `cond` 段接上 ✓，要么这次别给 lastFrame ✓）"))
        last_index = max(int(plan.frames) - 1, 0)
        started = time.perf_counter()
        info = dict(result.conditioning or {})
        info.update({
            "applied": True,
            "mode": info.get("mode") or "h3-keyframe-cond",
            "lastImagePath": request.last_frame,
            "lastFrameIndex": last_index,
            "viaLast": f"init_dual_latents（尾帧在第 {last_index} 帧编成 `cond` 段 ✓）",
        })
        info.setdefault("imagePath", request.last_frame)
        result.conditioning = info
        result.stageMs["condition"] = (int(result.stageMs.get("condition", 0))
                                       + int((time.perf_counter() - started) * 1000))
        emit(result.event("condition", note=f"尾帧 → 第 {last_index} 帧 `cond` 段 ✓"))

    if request.last_frame and not dual:
        # 单流的 `condition_first_frame` 只会把掩码混到**第 0 潜帧** ✓ ⇒ 做不了尾帧 ✗。
        # 硬做（比如硬塞成首帧）会让产物**开头**是尾帧 ✓✗ —— 比报错难查得多 ✓。
        return _fail(result, "condition", conditioning_mod.ConditioningError(
            f"后端 {getattr(backend, 'name', type(backend).__name__)} 未实现尾帧条件 ✗"
            f"（单流只有 conditioning_first_frame ⇒ 掩码只能混第 0 潜帧 ✓）"
            f"；刻意报错，不静默忽略尾帧 ✗"))

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
    holder: dict[str, Any] = {"step": 0, "scale": 1.0, "x": latents}
    positive, negative = _branches(condition)
    guided_counter = {"steps": 0, "calls": 0}

    def on_step(step: int, sigma: float, _x: Any) -> None:
        holder["step"] = step
        holder["x"] = _x           # 随机采样器要按**当前** x 问形状/设备/精度 ✓（见 `_noise_sampler_for` ✓）
        note = f"σ={sigma:.4g}"
        if holder["scale"] != 1.0:
            note += f" / cfg={holder['scale']:.2f}"
        emit(result.event("sample", step=step, total=plan.steps, note=note))
        ensure_not_cancelled("sample", step)  # 每步都能停 ✓
        #: ⭐ 运行期探压 ✓（GPU 优先那条规则的**运行期**一半 ✓ —— 见 `_backend_tick` ✓）
        _backend_tick(backend, "sample", step=step)

    try:
        #: ⭐ 换装点 ✓：到此 TE 已经用完 ✓（`condition` 是**张量** ✓，不占它）⇒ 让位给 DiT ✓。
        #: 1.0 版那条「TE 与 DiT 同时驻留 28 GiB」在这里必然 OOM ✓✗（见 `PHASE_PARTS` 的账 ✓）。
        _backend_phase(backend, "sample")
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
                # ⚠️⭐ **文本标签转手**（`H3Condition.tags` ✓）：编码器按 ``<|image_pad|>`` 的位置
                #    算出"哪个 token 走 video 行 ✓"，采样时 `h3_form.mod_segments_for` 靠它把文本段
                #    切成若干 run ✓✗。**丢了它** ⇒ 视觉块整段用错 adaLN 行 ✓、形状全对 ✓✗
                #    （fl2va/ref2va 只要有图就必错 ✓；纯文本时标签恒全 1 ✓ 与"不给"逐位相同 ✓）。
                tags = condition.get("tags")
                if tags is not None:
                    request = dataclass_replace(request, text_token_tags=tuple(
                        int(value) for value in tags))

            def on_dual_step(index: int, sigma_v: float, sigma_a: float) -> None:
                emit(result.event("sample", step=index, total=plan.steps,
                                  note=f"σv={sigma_v:.4g} / σa={sigma_a:.4g}"))
                _backend_tick(backend, "sample", step=index)
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

            #: ⚠️ 随机采样器（`euler_ancestral` / `sa_solver`）要**真的**能把噪声注回去 ✓✗ ——
            #: 不接这根线时它们按 η=1 缩短步长却不加噪 ✓（数学上不自洽 ⇒ 图偏灰 ✓、不报错 ✓）。
            latents, result.sampleSteps = sampler_mod.sample(
                model_fn, latents, plan.sigmas, str(request.sampler or "euler"),
                callback=on_step,
                noise_sampler=_noise_sampler_for(backend, request, holder))
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
        _backend_phase(backend, "decode")      # 分时驻留：DiT 让位给 VAE ✓（2.1 的 64 通道解码 ✓）
        decoded = backend.decode(latents, plan, request)
        # ⭐ ⑤b 出片前的**交棒点** ✓：多段成片（见 app.services.engine.chain ✓）要的正是这里的
        #    **无损**张量 ✓ —— 不让它去读刚落的 mp4 ✗（那要多一次编解码 ✓，还会把 yuv420p 的
        #    色度损失带进拼接件 ✓✗）。⚠️ 刻意放在 decode 的 try **里面** ✓：回调自己炸了也得算
        #    「decode 阶段失败」✓ —— run_sync「不抛异常」的契约不能被一个可选回调破掉 ✗。
        if on_decoded is not None:
            on_decoded(decoded)
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
    """把异常归一成结果 ✓ —— 取消**不是失败** ✗（前端要分开显示 ✓）。

    ⚠️ ``OutOfMemoryError`` **单独点名** ✗✗（2026-10-29 加 ✓）：显存不够是本仓**唯一**一个
    「仓库自己知道答案、而原生报错里一个字都不提」的失败 ✓✗ —— ``CUDA out of memory. Tried to
    allocate 2.00 GiB`` 只说明分配失败 ✓，**不会**告诉你「720×1280 只有 ``int8_offload``
    这一档装得下」✓✗，而那句话就写在本仓 ``h3_form.place_module`` 与
    ``TorchBackend.__init__`` 的注释里 ✓（= 三处实测：int8 全在卡上只剩 2.87 GiB 给激活 ✓）。
    ⇒ 这里把三条**可行动**的路补进 message ✓，并把 ``vramExhausted`` 写进结构化字段 ✓
    （前端要能**直接判**、而不是解析中文 ✓ —— 与「取消」那条同一个口径 ✓）。

    ⚠️ 判据用**类名**、不解析英文报错 ✗：原生文案一改（torch 大版本升级就会改 ✓）映射就失效 ✓✗
    —— 与 ``facade._friendly`` 里 ``name == "EngineBusy"`` 的做法一致 ✓。
    """
    if isinstance(err, GenerationCancelled):
        result.cancelled = True
        # ⚠️ 步号必须**进结构化字段**（不能只写在文案里 ✓）：自检 ㉗ 抓到初版只把「第 3 步」
        #    放进了 message ⇒ 前端想画「停在第 3/20 步」还得去**解析中文** ✗。
        result.error = {"stage": stage, "step": step, "message": str(err), "cancelled": True}
        return result
    stage_error = err if isinstance(err, StageError) else StageError(
        stage, f"{type(err).__name__}: {err}"[:300], step=step, cause=err)
    message = str(stage_error)
    # ⚠️ 用 ``in`` 不用 ``==`` ✓：真类是 ``torch.cuda.OutOfMemoryError`` ✓（名字恰好等于
    #    ``OutOfMemoryError`` ✓），但将来若出现子类（``CudaOutOfMemoryError`` 之类 ✓）也要认 ✓✗。
    vram = "OutOfMemory" in type(err).__name__
    if vram:
        # ⚠️ 这是**给用户看的**文案 ⇒ 不带 ✓✗ 记号 ✓（那是本仓注释里的写法 ✗ ——
        #    `facade.CODES` 的每一条都是干净中文 ✓，这里对齐 ✓）。
        message += (
            "\n  · 这是显存不够（不是权重坏，也不是实现缺）。按代价从低到高有三条路："
            "\n    ① 开 int8Offload: true（装配键 int8_offload）：int8 常驻权重钉在 CPU 锁页内存、"
            "前向逐层搬上卡，显存立省 17.94 GiB（本仓 dit_fl2va_int8 的实测值，别的量化档按它自己的"
            "体积算；代价：每步多一趟 PCIe）。在本仓实测的 22.49 GiB（A5000、ECC 开启）上，"
            "720×1280 只有这一档装得下；先看 status 里的 int8Offload 是不是本来就没开。"
            "\n    ② 降这一次的分辨率 / 帧数（megapixels、ratio、seconds）—— 激活是按它们长的。"
            "\n    ③ 换更大的卡（22.49 GiB 的 A5000 在 720×1280 上就是这么紧）。"
            "\n  · 已经开着 int8Offload 还是爆，就只剩 ②③：offload 已经把权重那份省掉了，"
            "剩下的就是激活，它跟分辨率走。")
    result.error = {"stage": stage_error.stage, "step": stage_error.step,
                    "message": message,
                    "type": type(err).__name__}
    if vram:
        result.error["vramExhausted"] = True
    return result
