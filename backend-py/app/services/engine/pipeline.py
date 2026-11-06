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

import math
import time
from dataclasses import asdict, dataclass, field, replace as dataclass_replace
from typing import Any, Callable, Protocol

from . import canvas as canvas_mod
from . import conditioning as conditioning_mod
from . import geometry, guidance as guidance_mod, latent_formats, sampler as sampler_mod, schedules
from . import profiles as profiles_mod
from . import tuning as tuning_mod
#: ⚠️ 只为 :func:`h3_latent_frames` 的**分段算式**（纯数学 ✓ 顶层不 import torch ✓）——
#: 拿通用式 ``(n-1)//4+1`` 算 H3 的潜帧会**少 2 格**（见该函数 docstring ✓）。
from . import vae_h3 as vae_h3_mod
#: ⚠️ 这两个**顶部零 torch** ✓（torch 都在函数里懒加载 ✓）⇒ 可以直接 import 取真源 ✓。
#: ``dit_music3`` 不行 ✗（第 30 行就是 ``import torch`` ✓）⇒ 那三个数只能抄一份 ✓，
#: 由 ``tests/engine_music3_wiring_test.py`` 的 A7／A8 两组逐值比对 ✓（1..9000 帧对拍 ✓）。
from . import te_music3 as te_music3_mod  # noqa: E402
from . import vae_music3 as vae_music3_mod  # noqa: E402

__all__ = [
    "H3_FLOW_SHIFT",
    "IMAGE_STAGE",
    "KREA2_FLOW_SHIFT",
    "KREA2_STAGE",
    "QWEN21_FLOW_SHIFT",
    "QWEN21_STAGE",
    "QWEN_IMAGE_FLOW_SHIFT",
    "QWEN_IMAGE_STAGE",
    "SEEDVR2_FLOW_SHIFT",
    "SEEDVR2_STAGE",
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
    "krea2_base_shift",
    "krea2_shift_mu",
    "run_sync",
    "seedvr2_target",
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

#: ⭐ :data:`KREA2_STAGE` ✓：**Krea2** 单张图片 ✓（2026-11-01 新增 ✓）。
#:
#: ⚠️ 这已经是**第五套**"出一张图"了 ✗（SDXL ✓ / Z-Image ✓ / 2.1 ✓ / 2512 ✓），为什么还要新名字 ✓：
#: * σ：flow ✓ 但 shift=**μ 1.15 ⇒ 底数 3.158** ✓（与 3 ✓ / 0.69 ✓ / 3.1 ✓ 全不同 ✓，
#:   且**只有这一族是 μ 形** ✗ —— 见 :func:`krea2_base_shift` ✓）；
#: * 潜变量：**16 通道 / 8×** ✓ 与 2512 同为 16 ✓，但**反缩放取 Wan21 的 mean/std** ✓，
#:   而 VAE 本体却是 ``qwen_image_vae`` ✓ —— 这类**跨族组合**一旦被"就近复用"就会静默错 ✓✗；
#: * 文本塔：**Qwen3-VL-4B 只挂语言塔** ✓ ＋ 取 **12 层隐状态栈** ✓（宽 30720 ✓）；
#: * 主干：单流 DiT ✓ **12.82B 参数** ✓（必须 int8 常驻 ✓，见 `_build_krea2_module` ✓）。
#: 合进任何一条 ⇒ 尺寸/σ/通道/文本宽度四项里至少一项按别族的规矩走 ✓✗，
#: 而这类错的共同症状是「形状自洽、能出图、不报错、结果不对」✓✗。
KREA2_STAGE = "krea2"
#: Krea2 的 flow shift **μ** ✓（指数形 ✓，不是底数 ✗）。出处：comfy ``supported_models.Krea2`` 的
#: ``sampling_settings = {"multiplier": 1.0, "shift": 1.15}`` ✓ ⇒ ``model_base.Krea2`` 是
#: ``ModelType.FLUX`` ✓ ⇒ ``ModelSamplingFlux`` ✓ ⇒ ``sigma(t) = flux_time_shift(1.15, 1, t)`` ✓
#: = ``e^1.15 / (e^1.15 + 1/t − 1)`` ✓。
#:
#: ⚠️ **这条是 μ，不是底数** ✗✗（2026-11-01 实测抓的 ✓）：``ModelSamplingFlux`` 走的是
#: ``flux_time_shift`` ✓（``comfy/model_sampling.py`` ✓），底数其实是 ``e^μ ≈ 3.158`` ✓；
#: 而 2512 / Z-Image 走 ``ModelSamplingAuraFlow`` ✓，那里的 ``shift`` 就是**底数本身** ✓
#: （``shift·t / (1 + (shift−1)·t)`` ✓）。两族同名不同义 ✗ —— 把 1.15 当底数喂下去
#: ⇒ σ 曲线几乎不弯 ✓ ⇒ **能出图、不报错** ✓，只是"怎么调都不像" ✓✗（与前两条坑同形 ✓）。
#: 用的时候一律 :func:`krea2_base_shift` ✓，别直接把它塞进 ``flow_sigmas_for_steps`` ✗。
KREA2_FLOW_SHIFT = 1.15
#: ⭐ :data:`SEEDVR2_STAGE` ✓：SEEDVR2 **视频修复/超分**（2026-11-01 新增 ✓）—— 本仓第一条
#: **不生成、只修**的线 ✓。
#:
#: ⚠️ 为什么不并进 :data:`VIDEO_STAGE`（H3 ✓ 明明也都是"出一段视频" ✓）✗✗：
#: * **输入完全不同** ✗：H3 是「文本/提示词 ⇒ 视频」✓；这一族是「**一段退化视频** ＋ loss mask
#:   ⇒ 修复版」✓ —— 请求里**必须**有源视频 ✓（``referenceVideos[0]`` ✓），没有就当场报错 ✓
#:   （不许退化成"文生视频" ✗ —— 那会出一段**内容无关**的片子 ✓✗，还看不出错 ✓）；
#: * **主干入参不同** ✗：每一步的 ``conditioning`` 是 **17 通道** ✓（退化 latent 16 ＋ mask 1 ✓），
#:   ``vid_in`` 因此是 **33** 通道 ✓（噪声 16 ＋ 条件 17 ✓）⇒ 潜变量**不能**按 ``in_channels`` 造 ✓；
#: * **没有文本塔** ✗（条件烤在权重里 ✓），⇒ 它与 H3 的组件表、装载线、回执照都不一样 ✓；
#: * **潜空间口径不同** ✗：反缩放 ×**0.9152** ✓（H3 自己那一档 ✓ 也不是 mean/std 组 ✓✗）。
#: 合流 ⇒ 形状可能都对 ✓、跑完也不报错 ✓、出来的片子"像但不对"✓✗ —— 与本仓其余五条坑同形 ✓。
SEEDVR2_STAGE = "seedvr2"
#: SEEDVR2 的 flow shift ✓ —— 与 2.1 / 2512 / Krea2 都**不是**一个数 ✗。
#:
#: 出处：官方的 ``NaDiT`` 采样走 ``flow`` 且 **shift = 1.0** ✓（即不做时间重排 ✓）；实测
#: ComfyUI 侧的 seedvr2 节点也是"线性 flow、不插值" ✓。⇒ 这里**不**调 :func:`krea2_shift_mu` ✓、
#: 也**不**用 ``FLOW_SHIFT_DEFAULT=3`` ✗（用 3 会让 σ 曲线整体偏陡 ✓ ⇒ 修出来的片子细节被"洗掉"
#: ✓、且**不报错** ✓✗）。
SEEDVR2_FLOW_SHIFT = 1.0
#: SEEDVR2 的**帧网格** ✓：像素帧必须是 ``4n+1`` ✓（视频 VAE 时间压缩 4× ✓；``T=1`` 也走 4× 的
#: 「截核」那一路 ✓ —— 见 `vae_seedvr2.CAUSAL_TAIL` ✓）；最少 5 帧 ✓（``n ≥ 1`` ✓）。
SEEDVR2_FRAME_GRID = 4
SEEDVR2_MIN_FRAMES = 5
#: 修复目标的**边长网格** ✓ = VAE 8× ✓ × 主干 patch 2× ✓ ⇒ **16** ✓（与 Krea2 同一条算术 ✓；
#: 写 8 会让潜边长出奇数 ✓ ⇒ `_PatchIn` 当场抛「不能整除 patch」✓，指不到"吸附写小了"✓）。
SEEDVR2_DIM_GRID = 16
#: 官方/实测的**步数档** ✓：这一族是「**一步**修复」的蒸馏档 ✓（``NaDiT`` 采样 1 步 ✓）。
SEEDVR2_OFFICIAL_STEPS = 1
#: 单次修复的帧数**上限** ✓（超过就只修前 N 帧并如实报 ✓ —— 不静默截断 ✗）。
#: 依据：17 通道条件 + 3B 主干在 22.49 GiB 卡上，121 帧 @736×416 已是 9 GiB 级激活 ✓；
#: 长片应当**分段**调用本阶段 ✓（每段一次 ``/engine/...`` 请求 ✓）。
SEEDVR2_MAX_FRAMES = 121

#: ⭐ :data:`WAN22_STAGE` ✓：**Wan 2.2 I2V**（首帧 ⇒ 视频 ✓，2026-11-02 新增 ✓）—— 本仓第一条
#: **真·图生视频**的线 ✓（H3 是文生视频 ✓、SEEDVR2 是修复 ✓）。
#:
#: ⚠️ 为什么不能并进 :data:`VIDEO_STAGE`（H3 ✓ 也是"出一段视频" ✓）✗✗：
#: * **主干入口 36 通道** ✗（噪声 16 ＋ 掩码 4 ＋ 条件图 16 ✓，见 `dit_wan22.I2V_LAYOUT` ✓）
#:   —— H3 是 16 通道直入 ✓；只改 ``in_dim`` 不改拼接 ⇒ 通道数对 ✓、条件错位 ✓，出片完全
#:   不受首帧约束 ✓且一条报错都没有 ✓✗；
#: * **文本塔是 UMT5-XXL** ✗（24 层纯 encoder ✓、4096 宽 ✓、不套模板 ✓、**padding 位置要置 0** ✓）
#:   —— H3 用 Qwen3-VL ✓、另外五套图族用 Qwen3-4B / Qwen3.5-9B / Qwen2.5-VL ✓；四座塔**不可互换** ✓✗；
#: * **两个专家** ✗：Wan2.2 A14B 把「高噪」与「低噪」拆成**两份权重** ✓（各 13.31 GiB ✓），
#:   官方模板是**两个 KSamplerAdvanced 接力** ✓（前一半高噪、后一半低噪 ✓）—— H3 只有一份 ✓；
#: * **潜空间口径**：Wan2.1 VAE ✓ 16 通道 / 8× ✓ ＋ **4 倍时间压缩** ✓（帧数必须 ``4n+1`` ✓）。
#: 合流 ⇒ 尺寸/通道/文本宽度/σ 四项里必有一项按别族走 ✓✗，症状与其余各族的同类坑同形 ✓。
WAN22_STAGE = "wan22"
#: Wan 2.2 的 flow shift ✓。出处：官方 UI 模板 ``video_wan2_2_14B_i2v.json`` 的两处
#: ``ModelSamplingSD3`` 节点 ``widgets_values`` = **5.0** ✓（⚠️ 不是 Wan2.1 的 8.0 ✗）。
#:
#: ``ModelSamplingSD3`` 走 ``ModelSamplingDiscreteFlow`` ✓ ⇒ ``σ = shift·t / (1 + (shift−1)·t)`` ✓
#: （**底数形** ✓，与 Krea2 的 μ 形不同 ✓✗ —— 别把 1.15 那套搬过来 ✓）。
WAN22_FLOW_SHIFT = 5.0
#: 官方模板的帧率 ✓（``CreateVideo`` / ``SaveVideo`` 都是 **16** ✓）。
WAN22_FPS = 16
#: 像素帧网格 ✓：Wan2.1 VAE 时间压缩 **4×** ✓ ⇒ 帧数必须 ``4n+1`` ✓（81 帧 ⇒ 21 潜帧 ✓）。
WAN22_FRAME_GRID = 4
#: 最少帧数 ✓（``n ≥ 1`` ⇒ 5 帧 ✓；再少 VAE 时间核就只剩"截核"那一路 ✓）。
WAN22_MIN_FRAMES = 5
#: 边长网格 ✓ = VAE 8× ✓ × 主干 patch 2× ✓ ⇒ **16** ✓（与 Krea2 / SEEDVR2 同一条算术 ✓）。
WAN22_DIM_GRID = 16
#: 官方模板的**双段步数** ✓：``KSamplerAdvanced`` 两个都是 ``steps=4`` ✓，
#: 高噪跑 ``[0, 2)`` ✓、低噪跑 ``[2, 4)`` ✓ ⇒ 整条梯子 **4 段** ✓、切换点在 **50%** ✓。
WAN22_OFFICIAL_STEPS = 4
#: 高噪专家跑前多少比例的步 ✓（官方 2/4 = **0.5** ✓）。剩余步交给低噪专家 ✓。
WAN22_SWITCH_RATIO = 0.5
#: 官方模板的画布 ✓（``WanImageToVideo`` 的 ``width=640 / height=640 / length=81`` ✓）。
#: ⚠️ 只作**默认值**报出来 ✓，不覆盖调用方给的尺寸 ✗。
WAN22_OFFICIAL_SIZE = (640, 640)
WAN22_OFFICIAL_FRAMES = 81


# ══════════════════════════════════════════════════════════════════════════
# Music3（音频族 ✓ —— 第三段：context ⇒ latent ⇒ 44.1k 立体声 ✓）
# ══════════════════════════════════════════════════════════════════════════
MUSIC3_STAGE = "music3"
#: ⚠️ 下面四个数**抄自 ``dit_music3``** ✓ —— 它第 30 行就是 ``import torch`` ✓✗，
#: 在 plan 阶段 import 它会把这族的 torch 依赖拖进来 ✓（本模块「不碰张量」是硬约束 ✓）。
#: 「抄一份」是有代价的 ✓✗：两边分叉**不报错** ✓、只出**长度不对的歌** ✓ ⇒ 一致性由
#: ``tests/engine_music3_wiring_test.py`` 的 A7／A8 两组逐值比对钉住 ✓（不许只靠注释 ✓）。
MUSIC3_MAX_COND_FRAMES = 200     #: ``dit_music3.MAX_CONDITION_FRAMES`` ✓
MUSIC3_COND_HOP_FRAMES = 100     #: ``dit_music3.CONDITION_HOP_FRAMES`` ✓
#: 条件帧 → latent 帧的倍率 ✓ ＝ ``LATENT_SAMPLE_RATE/CONDITION_SAMPLE_RATE ×
#: LATENT_HOP/CONDITION_HOP`` ＝ ``44100/24000 × 512/960`` 的倒数 ＝ **3.4453125** ✓。
#: ⚠️ 「两边都当 1:1」是本族最先撞上的一处 ✓✗：条件 **25 Hz** / latent **86.13 Hz** ✓，
#: 长度不换算就被 pad/crop **静默**吃掉 ✓（歌少掉后半段 ✓、一步都不报错 ✓✗）。
MUSIC3_LATENT_RATIO = 44100.0 / 24000.0 * 960.0 / 512.0
#: σ 口径 ✓ —— **有出处** ✓（2026-11-06 翻到参考实现，原先那句「没有出处」作废 ✓✗）：
#:
#: 1. ``comfy/model_base.py::MiniMaxMusic3.__init__`` 传 ``model_type=ModelType.FLOW`` ✓；
#: 2. ``ModelType.FLOW`` ⇒ ``model_sampling.ModelSamplingDiscreteFlow(model_config)`` ✓，
#:    而它取 ``shift=sampling_settings.get("shift", 1.0)`` ✓、``multiplier=get("multiplier", 1000)`` ✓；
#: 3. ``MiniMaxMusic3.sampling_settings = {"multiplier": 1.0}`` ✓ ⇒ **没有 shift 键** ⇒ shift=**1.0** ✓、
#:    multiplier=**1.0**（不是 1000 ✓ ⇒ 喂给主干的 timestep 就是 σ 本身 ✓）；
#: 4. ``time_snr_shift(alpha, t)`` 头一行就是 ``if alpha == 1.0: return t`` ✓ ⇒ σ 网格**不加弯** ✓，
#:    就是 ``i/1000``（i=1..1000 ⇒ σ∈[0.001, 1.0] ✓）。
#:
#: ⚠️ **不许**照抄邻族 ✓（2.1 的 0.69 / Krea2 的 μ1.15 / Wan22 的 5.0 / Z-Image 的 3.0 ✓ 各一套 ✓）：
#: 抄错了不报错 ✓、只是歌越来越糊 ✓✗。
MUSIC3_FLOW_SHIFT = 1.0
#: 条件帧率与上限 ✓ —— 直接从 ``te_music3`` 取 ✓（它顶部零 torch ✓，不用抄 ✓）。
MUSIC3_COND_FPS = te_music3_mod.AUDIO_FRAMES_PER_SECOND
MUSIC3_MAX_AUDIO_FRAMES = te_music3_mod.MAX_AUDIO_FRAMES


def music3_latent_frames(cond_frames: int) -> int:
    """条件帧数 ⇒ latent 帧数 ✓（**向下取整** ✓ ＝ 参考侧 ``frames // 512`` 的同一口径 ✓）。

    ⚠️ 取整方向要**与参考侧一致** ✓✗：向上取整会让 latent 多出一截 ✓ ⇒ 解码回来的音频
    比请求的秒数**长** ✓（播放器看不出异样 ✓，但 ``durationSeconds`` 与 file 时长对不上 ✓）。
    """
    return int(math.floor(max(0, int(cond_frames)) * MUSIC3_LATENT_RATIO))


def music3_window_plan(latent_frames: int) -> dict[str, Any]:
    """全长 latent 要几个窗 ✓（返回窗数／窗长／步长／最后一窗覆盖到哪 ✓）。

    ⚠️ 这一族的「长音频」**不是**靠 pipeline 切窗 ✓ —— 切窗已经写在
    :meth:`dit_music3.Music3DiT.velocity` 里 ✓（按 latent 689／跳 344 ✓、重叠段按权重叠加 ✓）。
    所以这里**只诊断、不切** ✓：算出「会切几个窗」写进 plan 让人看得见 ✓，
    绝不在这里再切一遍 ✓✗（切两遍 ⇒ 每段都被自己的重叠区抹一次 ✓、糊得很均匀 ✓）。
    """
    window = music3_latent_frames(MUSIC3_MAX_COND_FRAMES)
    hop = music3_latent_frames(MUSIC3_COND_HOP_FRAMES)
    total = max(1, int(latent_frames))
    count = 1 if total <= window else 1 + -(-(total - window) // hop)
    last_end = min(total, window + (count - 1) * hop)
    return {"count": count, "window": window, "hop": hop,
            "total": total, "lastEnd": last_end,
            "covered": last_end >= total}


def seedvr2_target(facts: dict[str, Any]) -> dict[str, Any]:
    """ffprobe facts ⇒ 修复目标的（宽 ✓ 高 ✓ 帧数 ✓ 潜帧数 ✓ 帧率 ✓）—— **唯一口径** ✓✗。

    ⚠️ 为什么抽成函数 ✗✗：``plan`` 阶段（:func:`_fill_seedvr2_plan` ✓）与**后端 encode 阶段**
    （``torch_backend._encode_seedvr2_conditioning`` ✓）读的是**同一个源文件** ✓，
    两处各写一遍吸附规律 ⇒ 「plan 说 704×448 ✓、实际编码出 720×448 ✓」这种**不报错**的错位 ✓✗
    （症状是采样第 1 步形状对不上 ✓，报错指不到"吸附写了两份" ✓）。

    三条口径（都是硬的 ✓）：
    * **边长**按 :data:`SEEDVR2_DIM_GRID`（16 ✓）**就近**吸 ✓ —— **不缩放内容** ✗（拉伸会改构图 ✓）；
    * **帧数**按 ``4n+1`` **向下**吸 ✓（截掉尾部几帧 ✓）—— **绝不住前补帧** ✗（补的帧会让
      「最后一帧停住」写进修复结果 ✓✗，而画面上完全看不出来是补的 ✓）；
    * **帧率**用源视频的 ✓（修复不改时间轴 ✓），读不到才退 24 ✓ 并在 ``warnings`` 里点名 ✓。
    """
    width = int(facts.get("width") or 0)
    height = int(facts.get("height") or 0)
    grid = SEEDVR2_DIM_GRID
    if width < grid or height < grid:
        raise ValueError(
            f"源视频尺寸读不出来 / 太小 ✗：{width}×{height} ✓（至少要 {grid}×{grid} ✓）")
    target_w = max(grid, int(round(width / grid)) * grid)
    target_h = max(grid, int(round(height / grid)) * grid)
    warnings: list[str] = []
    if (target_w, target_h) != (width, height):
        warnings.append(
            f"源尺寸 {width}×{height} 不是 {grid} 的整数倍 ✓ ⇒ 修复目标取 **{target_w}×{target_h}** ✓"
            f"（**就近吸附** ✓ 不做 letterbox/crop ✗ —— 那会改构图 ✓✗；"
            f"要精确按源尺寸出片得先把素材裁成 {grid} 的倍数 ✓）")
    fps_raw = facts.get("fps")
    fps = int(round(float(fps_raw))) if fps_raw else 24
    if not fps_raw:
        warnings.append("源视频帧率读不出来 ⇒ 按 24fps 记 ✓（修复不改时间轴 ✓，这里只影响时长标注 ✓）")
    raw_frames = facts.get("nbFrames")
    if not raw_frames:
        duration = facts.get("durationSeconds")
        if not duration:
            raise ValueError(
                "源视频的帧数与时长**都**读不出来 ✗ ⇒ 不知道该修多少帧 ✓"
                "（`nbFrames` / `durationSeconds` 同时为空 ✓ —— 换一份带完整 index 的封装 ✓，"
                "或先用 ffmpeg 重封装 ✓）")
        raw_frames = int(round(float(duration) * max(1, fps)))
        warnings.append(f"`nbFrames` 缺失 ⇒ 按时长 {float(duration):.3f}s × {fps}fps 估 **{raw_frames} 帧** ✓")
    raw_frames = int(raw_frames)
    if raw_frames < SEEDVR2_MIN_FRAMES:
        raise ValueError(
            f"源视频只有 {raw_frames} 帧 ✗ ⇒ SEEDVR2 至少要 {SEEDVR2_MIN_FRAMES} 帧 ✓"
            f"（像素帧必须是 ``4n+1`` ✓）")
    if raw_frames > SEEDVR2_MAX_FRAMES:
        warnings.append(
            f"源视频 {raw_frames} 帧 > 单次上限 {SEEDVR2_MAX_FRAMES} 帧 ✓ ⇒ **只修前 "
            f"{SEEDVR2_MAX_FRAMES} 帧** ✓（如实截断 ✓ 不假装修了整片 ✗；"
            "长片请**分段**调本阶段 ✓）")
        raw_frames = SEEDVR2_MAX_FRAMES
    #: 向下吸到 ``4n+1`` ✓（:func:`geometry.latent_frames` 的逆运算 ✓ —— 两边必须同源 ✓）。
    frames = ((raw_frames - 1) // SEEDVR2_FRAME_GRID) * SEEDVR2_FRAME_GRID + 1
    frames = max(SEEDVR2_MIN_FRAMES, frames)
    if frames != raw_frames:
        warnings.append(
            f"帧数 {raw_frames} 不是 ``4n+1`` ✓ ⇒ 修 **{frames}** 帧 ✓"
            f"（**向下**截掉尾部 {raw_frames - frames} 帧 ✓ —— 补帧会把『最后一帧停住』写进结果 ✗）")
    return {"width": target_w, "height": target_h, "frames": frames,
            "latent_frames": geometry.latent_frames(frames,
                                                    temporal_compression=SEEDVR2_FRAME_GRID),
            "fps": fps, "sourceWidth": width, "sourceHeight": height,
            "sourceFrames": raw_frames, "warnings": warnings}


#: ``ModelSamplingFlux`` 的分辨率插值端点 ✓（出处 ``comfy/model_sampling.py`` ✓）：
#: 序列长度 ``256`` ⇒ ``base_shift`` ✓，``4096`` ⇒ ``max_shift`` ✓，中间**线性** ✓，
#: 两端**夹住** ✓。序列长度 = ``(宽/16)·(高/16)`` ✓（VAE 8× ✓ × patch 2×2 ✓ —— 与 Krea2
#: 的吸附步长 16 同源 ✓）。
KREA2_SHIFT_SEQ_RANGE: tuple[float, float] = (256.0, 4096.0)
KREA2_SHIFT_MU_RANGE: tuple[float, float] = (0.5, 1.15)


def krea2_shift_mu(width: int | None = None, height: int | None = None) -> float:
    """Krea2 的 **μ** —— **按实际分辨率插值** ✓（不是恒 1.15 ✗）。

    出处（逐字对齐 ✓）：``comfy/model_sampling.py::ModelSamplingFlux.get_sigmas`` ✓ ——

    .. code-block:: python

        x1, x2 = 256, 4096
        mm = (max_shift - base_shift) / (x2 - x1)      # (1.15 - 0.5) / 3840
        b = base_shift - mm * x1
        shift = (width * height / (8 * 8 * 2 * 2)) * mm + b     # ← 序列长度 × mm + b

    ⚠️ 为什么必须插值 ✓✗：工作流里 ``ModelSamplingFlux`` 的 ``width``/``height`` **是连线** ✓
    （接的正是出图尺寸 ✓）⇒ 分辨率一变，ComfyUI 给的 μ 就变 ✓。写死 1.15 只在
    **1024×1024（seq=4096 ✓ 恰在上端点 ✓）** 上等价 ✓；512×512（seq=1024 ✓）真实值是
    **0.6266** ✓，写死 1.15 会让低分辨率出图的 σ 曲线整体**偏陡** ✓（不报错 ✓、细节发糊 ✓✗）。
    默认 1024² ⇒ **1.15** ✓（老口径的值 ✓，逐位不变 ✓）。
    """
    width = 1024 if width is None else int(width)
    height = 1024 if height is None else int(height)
    low, high = KREA2_SHIFT_SEQ_RANGE
    floor, ceiling = KREA2_SHIFT_MU_RANGE
    sequence = (max(1, width) / 16.0) * (max(1, height) / 16.0)
    if sequence <= low:
        return floor
    if sequence >= high:
        return ceiling
    slope = (ceiling - floor) / (high - low)
    return floor + slope * (sequence - low)


def krea2_base_shift(mu: float | None = None, *, width: int | None = None,
                     height: int | None = None) -> float:
    """把 Krea2 的 **μ** 换成 ``flow_sigmas_for_steps`` 要的**底数** ✓（``e^μ`` ✓）。

    单独抽成函数是为了让"这条是 μ"这件事**写在名字上** ✓ —— 直接写 ``shift=1.15`` 也能跑 ✓、
    数字也"看着对" ✓，只是 σ 曲线错 ✓✗（本轮真栽在这上面：出图是一团纹理 ✓、提示词几乎不跟 ✓）。

    ``mu`` 不给 ⇒ **按分辨率算** ✓（:func:`krea2_shift_mu` ✓；不给宽高 ⇒ 1024² ⇒ 1.15 ✓，
    与旧口径**逐位相同** ✓）。
    """
    if mu is None:
        mu = krea2_shift_mu(width, height)
    return math.exp(float(mu))

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
    #: ⭐ **画布计划** ✓ —— :class:`app.services.engine.canvas.CanvasPlan` 的实例 ✓（调用方先走
    #: ``/engine/canvas-plan`` ✓，口径与超清那条同构 ✓：「计划层定 ✓、管线只执行」✗）。
    #: ``None`` ⇒ **完全按旧路径算** ✓（逐位不变 ✓ —— 这是零回归的前提 ✗ 别在填充函数里改默认 ✓）。
    #: ⚠️ 追加在**末尾** ✓（同 ``stage`` 的理由 ✓）。
    canvas: Any = None
    #: ⭐ **用途名** ✓（``"scene-card"`` / ``"character-sheet"`` ✓ 见 :data:`profiles.PROFILES`）——
    #: 引擎自己按 :data:`~app.services.engine.canvas.STAGE_CANVAS` 查表算画布 ✓。
    #: ⚠️ 与 ``canvas`` **二选一** ✓✗（给了计划就按计划 ✓；两个都给 ⇒ 报错 ✗ 不猜优先级 ✗）。
    target: str | None = None
    #: ⭐ **起始噪声比例** ✓（图生图/尾帧精修 ✓）—— 工作流 ``KSampler.denoise`` ✓
    #: （``20_分镜尾帧_Qwen2512参考图`` = **0.55** ✓）。
    #: ``1.0`` ⇒ 从 σ 上界起跑 ✓（旧行为 ✓ 逐位不变 ✓）；``<1`` ⇒ **砍掉前段 σ** ✓
    #: ⇒ 实际步数变少 ✓ 且 ``plan.steps`` 报**实际跑的那几步** ✗（报名义步数是在骗人 ✓✗）。
    denoise: float = 1.0
    #: ⭐ 图生图的**起点图** ✓（本地路径 ✓）—— 与 ``reference_frames`` **不是一件事** ✗✗：
    #: 那份是**条件**（Edit 的 ``ref_latents`` 拼进上下文 ✓、改的是"参考什么" ✓）；
    #: 这份是**起点**（VAE 编码后按 ``σ0`` 与噪声混 ✓、改的是"从哪一格起跑" ✓）。
    #: ``denoise<1`` **必须**给 ✓（不给 ⇒ 当场拒 ✗ —— 只砍 σ 不换起点 = 与参考图无关的灰图 ✓✗）。
    #: ⚠️ 追加在**末尾** ✓（同 ``stage`` 的理由 ✓）。
    init_image: str | None = None
    #: ⭐ **歌词** ✓（Music3 那一档 ✓）—— 与 ``prompt``（caption／曲风描述 ✓）**分开** ✓✗：
    #: 参考侧 ``build_prompt`` 把两者拼成两条支路 ✓（曲风在前 ✓、歌词进 ``[verse]`` 段 ✓），
    #: 合成一句话会让模型分不清「这段是风格还是唱词」✓（症状：把提示词当歌词念 ✓✗）。
    #: 其余各族**不读**这个字段 ✓（给了也只在 ``stage="music3"`` 生效 ✓）——
    #: ⚠️ 这里**不写具体条数** ✗：本句原写「其余八条线」✓，接完 Music3 后已是过期散文 ✓✗
    #: （与 :func:`build_plan` 空提示词那条注释**同一个病根** ✓：手抄的计数必过期 ✓）。
    #: ⚠️ 追加在**末尾** ✓（同 ``stage`` 的理由 ✓ —— 插中间会移动位置参数 ✓✗）。
    lyrics: str = ""

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
    #: 画布来源 ✓（``None`` = 走的是**旧路径**算的 ✓ 不是"没画布" ✗）——
    #: 有值 ⇒ 内容是 :meth:`canvas.CanvasPlan.to_dict` ✓（含 ``source`` / ``profile`` / ``notes`` ✓）
    #: ⇒ 前端能显示「这张 1344×768 取自 scenecard 档」而不是只看到一个数 ✓。
    canvas: dict[str, Any] | None = None
    #: **相对**耗时系数 ✓（1 步 euler 无引导 = 1 ✓）—— 见 :meth:`to_dict` 里为什么不报秒数 ✗
    cost_factor: float = 0.0
    #: ⭐ 图生图那组数字 ✓（``denoise=1`` 时恒为默认值 ✓ 逐位不变 ✓）。
    #: ``denoise`` 原样回显 ✓、``total_steps`` = 名义步数 ``int(steps/denoise)`` ✓
    #: （**σ 梯子按它算** ✓、随后只取尾巴 ✓）、``start_sigma`` = 真正起跑的 σ ✓、
    #: ``init_from`` = ``"noise"``（纯噪声起跑 ✓）/ ``"image"``（参考图编码起跑 ✓）。
    #: ⚠️ 为什么四个都要报 ✗：``steps=20 / denoise=0.55`` 时梯子其实是 36 步的尾巴 ✓ ——
    #: 只报 ``steps=20`` 会让人以为"20 步从 σ=1 起跑" ✓✗（那就等于没精修 ✓）。
    denoise: float = 1.0
    total_steps: int = 0
    start_sigma: float = 0.0
    init_from: str = "noise"
    #: ⭐ Wan 2.2 的**双专家接力**分界 ✓（前多少步走「高噪」权重 ✓）——
    #: ``0`` = 本份 plan 不分段 ✓（其余各族恒为 0 ✓）。见 :data:`WAN22_SWITCH_RATIO` ✓。
    wan_high_steps: int = 0

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
            "canvas": self.canvas,
            # ⚠️ 图生图四格如实报 ✓（``denoise=1`` 时就是 1.0 / steps / 1.0 / "noise" ✓）——
            #    ``startSigma`` 是**真正起跑的那一格** ✓，不是"我打算跑多少步" ✗。
            "denoise": round(float(self.denoise), 4),
            "totalSteps": int(self.total_steps or self.steps),
            "startSigma": round(float(self.start_sigma), 4),
            "initFrom": self.init_from,
            # ⚠️ Wan 2.2 的**双专家分界** ✓（其余各族恒为 0 ✓）—— 报出来才看得出
            #    「前 2 步高噪、后 2 步低噪」是**照官方模板**分的 ✓，不是随手砍一半 ✓。
            "wanHighSteps": int(self.wan_high_steps or 0),
            "warnings": self.warnings,
        }


def build_plan(request: GenerationRequest, *, weights_bytes: int | None = None) -> GenerationPlan:
    """算出这一次要用的**尺寸/帧数/σ 序列** ✓（纯计算、毫秒级、可先给用户看 ✓）。

    ⚠️ 这里**提前**校验采样器名与参数 ✓ —— 让错误在「还没开始跑」时就以明确文案返回 ✓，
    而不是跑到一半才炸 ✗。
    """
    plan = GenerationPlan(fps=int(request.fps or geometry.H3_FPS))
    #: ⚠️ **空提示词只在修复族合法** ✓（2026-11-01 接 SEEDVR2 时补 ✓）：这一族**没有文本塔** ✗
    #: —— 条件是「源视频编码 ＋ loss mask」✓，逼调用方填一句"反正进不了前向"的提示词 ✓✗
    #: 只会教出「随便写点什么」的用法 ✓（与本仓"说真话"直接冲突 ✗）。其余各族**仍然必须给** ✓。
    #: ⚠️ 这里**不写具体条数** ✗：本句原来写「其余六条线」✓，接完 Wan22 与 Music3 后已成过期散文 ✓✗
    #: （阶段数只会继续加 ✓，任何手抄的计数都会过期 ✓）。
    _stage_key = str(request.stage or VIDEO_STAGE).strip().lower()
    if not str(request.prompt or "").strip() and _stage_key != SEEDVR2_STAGE:
        raise StageError("plan", "提示词不能为空 ✓（修复族 SEEDVR2 例外 ✓：它没有文本塔 ✓）")
    if request.steps < 1:
        raise StageError("plan", f"steps 必须 ≥1（收到 {request.steps} ✗）")
    if request.seconds <= 0:
        raise StageError("plan", f"seconds 必须 >0（收到 {request.seconds} ✗）")
    if str(request.sampler or "").lower() not in sampler_mod.SAMPLERS:
        raise StageError("plan", f"未知采样器 {request.sampler!r}；可用：{sorted(sampler_mod.SAMPLERS)}")

    # ── 参数覆盖层（P3 ✓）────────────────────────────────────────────────────
    #: 只改**调用方没显式表态**的白名单字段（见 `engine/tuning.py::TUNABLES` ✓），
    #: 且默认只有 ``steps`` ✓。没有 ``configs/engine/tuning.json`` ⇒ 请求原样返回 ✓、
    #: ``notes`` 为空 ⇒ **行为逐字不变** ✓（所以这一层不需要开关 ✓）。
    #: ⚠️ 必须在**分派**之前改 ``request`` ✗✗：plan 里的 σ / timesteps / costFactor
    #: 全是从 ``request.steps`` 推的 ✓ —— 事后改 ``plan.steps`` 会出现「30 步的 σ 配 4 步的
    #: steps」✓，形状对、数不对、**不报错** ✗。
    request, _tuning_notes = tuning_mod.apply_to_request(_stage_key, request)
    plan.warnings.extend(_tuning_notes)

    # ── denoise（起始噪声比例 ✓）──────────────────────────────────────────────
    # 语义**逐字对齐参考实现** ✓：``comfy`` 的 ``KSampler.set_steps`` ✓
    #   ``denoise<1`` ⇒ ``new_steps = int(steps / denoise)`` 算一条**更长**的 σ 梯子 ✓，
    #   再取它的**尾巴** ``sigmas[-(steps + 1):]`` ✓ ⇒ 起点落在 **σ_mid** 而不是 σ_max ✓；
    # 配套的**另一半**是起点 latent ✗✗：``noise_scaling(σ0, noise, image_latent)`` ✓
    #   = ``σ0·noise + (1−σ0)·image_latent`` ✓（``comfy/sample.py`` ✓）——
    #   少这一半就是「只砍 σ、起点仍是纯噪声」✓✗ ⇒ 出「低对比、与参考图无关」的图且**不报错** ✗
    #   （本仓栽过一整类"不报错只是图不对" ✓）⇒ 所以 ``denoise<1`` **必须**配 ``initImage`` ✓✗。
    # ⚠️ 视频侧**不接** ✗：视频"接着上一段跑"走 ``pin`` / ``lastFrame``（尾帧当关键帧 ✓）那条 ✓；
    #   单张参考图喂不进 N 帧的起点 ✗（``(1,C,1,h,w)`` vs ``(1,C,T,h,w)`` ⇒ 只能广播成同一张 ✓✗
    #   = 死帧 ✓ 且不报错 ✓）。
    try:
        denoise = float(request.denoise)
    except (TypeError, ValueError) as err:
        raise StageError("plan", f"denoise 必须是 0~1 的数 ✗（收到 {request.denoise!r} ✓）",
                         cause=err) from err
    if not 0.0 < denoise <= 1.0:
        raise StageError("plan", f"denoise 必须落在 (0, 1] ✗（收到 {denoise:g} ✓）")
    init_image = str(request.init_image or "").strip()
    if denoise < 1.0:
        if _stage_key == WAN22_STAGE:
            raise StageError(
                "plan",
                f"Wan 2.2 I2V 侧 denoise={denoise:g} < 1 不支持 ✗ —— 它的起点图是 ``firstFrame`` ✓"
                "（走 **36 通道条件** ✓，由主干在每一步看着它 ✓），不是「从参考 latent 起跑」✓✗；"
                "这两个词只差一个字母、效果却是「逐帧锚在图上」vs「只在第 0 步掺一次」✓✗")
        if _stage_key == VIDEO_STAGE:
            raise StageError(
                "plan",
                f"视频侧 denoise={denoise:g} < 1 不支持 ✗ —— 视频的「接着上一段跑」走 ``pin`` / "
                "``lastFrame``（尾帧当关键帧 ✓）那条 ✓；单张参考图广播成 N 帧起点只会出**死帧** ✓✗")
        if not init_image:
            raise StageError(
                "plan",
                f"denoise={denoise:g} < 1 必须同时给 **initImage** ✗✗ —— 它的语义是"
                "「从参考图编码出的 latent 起跑」✓（``noise_scaling(σ0, noise, latent)`` ✓）；"
                "只砍前段 σ 会让起点仍是**纯噪声** ✓✗ ⇒ 出「低对比、与参考图无关」的图且**不报错** ✗"
                "（本仓最恨的那类坑 ✓）。要给图就传 ``initImage`` ✓；"
                "没有参考图又想起低噪声起点，请用超清的**二采**（``upscale.refine_denoise`` ✓）")
    elif init_image:
        if _stage_key == WAN22_STAGE:
            #: ⚠️ **拒**，不是"静默忽略"✗（2026-11-02 ✓）：这两格在别的族里都能收 ✓，
            #: 于是「两个都传」看起来像"从这张图起跑、再用首帧约束"✓ —— 语义上根本不存在 ✓✗
            #: （``firstFrame`` 走 36 通道条件 ✓、每一步都看着它 ✓；``initImage`` 是"VAE 编码后
            #: 按 σ0 掺一次噪声当起点"✓✗，两者互斥 ✓）。静默丢掉 ``initImage`` 的后果是用户
            #: 以为"起点是他那张图"✓，实际起点是纯噪声 ✓、画面与那张图无关 ✓✗。
            raise StageError(
                "plan",
                "Wan 2.2 I2V **不收** ``initImage`` ✗✗ —— 本阶段的起点图是 ``firstFrame`` ✓"
                "（走 **36 通道条件** ✓，主干每一步都看着它 ✓）；``initImage`` 的语义是"
                "「VAE 编码后按 σ0 掺一次噪声当起点」✓✗ —— 两个都传＝用户以为「先用这张图起步、"
                "再被首帧约束」✓，而实际只能二选一 ✓ ⇒ 这里**拒绝**而不是静默丢掉 ✓。"
                "给起点请用 ``firstFrame`` ✓")
        raise StageError(
            "plan",
            f"给了 initImage 但 denoise=1 ✗ —— 从 σ 上界起跑时起点是**纯噪声** ✓✗，"
            "参考图会被**整个丢掉**（不报错、图与参考无关 ✗）。"
            "要图生图请把 ``denoise`` 调到 (0,1) ✓；要纯文生图请别传 ``initImage`` ✓")
    if abs(denoise - 1.0) <= 1e-9 and request.canvas is not None \
            and not isinstance(request.canvas, canvas_mod.CanvasPlan):
        raise StageError(
            "plan", f"canvas 类型不对 ✗（收到 {type(request.canvas).__name__} ✓）—— "
                    "先调 ``POST /api/v1/engine/canvas-plan`` 拿计划对象 ✓ 再原样传 ✓；"
                    "别把 dict 直接塞进来 ✗（那就等于在请求侧又算一遍画布 ✓✗）")

    stage = _stage_key
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
    if stage == KREA2_STAGE:
        _fill_krea2_plan(request, plan)
        return _finish_plan(request, plan, weights_bytes)
    if stage == SEEDVR2_STAGE:
        _fill_seedvr2_plan(request, plan)
        return _finish_plan(request, plan, weights_bytes)
    if stage == WAN22_STAGE:
        _fill_wan22_plan(request, plan)
        return _finish_plan(request, plan, weights_bytes)
    if stage == MUSIC3_STAGE:
        _fill_music3_plan(request, plan)
        return _finish_plan(request, plan, weights_bytes)
    if stage != VIDEO_STAGE:
        raise StageError(
            "plan", f"未知 stage {request.stage!r} ✗；可用："
            f"{sorted((VIDEO_STAGE, IMAGE_STAGE, ZIMAGE_STAGE, QWEN21_STAGE, QWEN_IMAGE_STAGE, KREA2_STAGE, SEEDVR2_STAGE, WAN22_STAGE, MUSIC3_STAGE))} ✓"
            "（⚠️ 别随手编一个名字 ✓ —— 它决定选哪个后端 ✓✗）")

    override = _canvas_override(request)
    if override is None:
        plan.width, plan.height = geometry.size_for_megapixels(
            max(0.05, float(request.megapixels)), request.ratio)
    else:
        # ⚠️ 覆盖只换**画布** ✗ —— σ 的 shift 仍按 `plan.width/height` 走 ✓（`_video_sigmas` 读的
        #    就是它 ✓）⇒ `flow:auto` 的序列长会跟着新画布变 ✓，这是对的 ✓（shift 本来就与分辨率绑 ✓）。
        plan.width, plan.height = override.width, override.height
        plan.canvas = override.to_dict()
        plan.warnings.extend(override.notes)
    plan.frames = geometry.snap_frames(request.seconds, fps=plan.fps)
    plan.steps = int(request.steps)

    # ⚠️ 潜帧数必须**先**算 ✗（``schedule="flow:auto"`` 的 shift 按它自适应 ✓）——
    #    这个先后不是排版问题：反过来的话 `flow:auto` 只能拿到 ``None`` 而报错 ✓✗。
    if request.temporal_compression:
        plan.latent_frames = h3_latent_frames(
            plan.frames, temporal_compression=int(request.temporal_compression))
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


def h3_latent_frames(frames: int, *, temporal_compression: int | None = None) -> int:
    """H3 的**像素帧 ⇒ 潜帧**换算 ✓ —— 与真视频 VAE **同一套分段算式** ✓。

    ⚠️ 通用式 ``(frames - 1) // 4 + 1`` **不适用 H3** ✗✗（2026-11-04 端到端实测钉死 ✓）：
    H3 的视频 VAE 编码器是 **17 帧一段**（末段重复末帧补齐 ✓）、**末尾整段丢 3 个潜格**
    ⇒ 真实口径是 ``5·ceil(T/17) − 3``（＝ :func:`vae_h3.latents_for_frames` ✓），
    反向 :func:`vae_h3.frames_for_latents` 在 ``17k+5`` 上**严格闭环** ✓（`geometry` 那个网格
    之所以是 ``17k+5`` 正是因为它 ✓）。

    拿通用式算 39 帧 ⇒ 得 **10** 潜帧 ✗，而真 VAE 要 **12** 格才解回 39 帧 ✓ ⇒ 少 2 格
    ⇒ 解出来只有 **34 帧**（1.4167 s）而音频仍是 1.625 s ✓✗ —— 症状是「视频比音频短 0.2 秒」，
    而且**一条报错都没有** ✓✗（`plan.frames` 说 39、`write_video` 回执说 34，两处都在回执里 ✓）。

    ``temporal_compression`` 只为**如实**表态：它不等于 VAE 的 ``temporalScale``（4 ✓）时，
    这套分段算式无从套用 ⇒ **退回通用式** ✓（保持调用方原来"不猜压缩比"的口径 ✓，不假装闭环 ✗）。
    """
    tc = vae_h3_mod.H3_VIDEO_VAE_ARCH_FACTS["temporalScale"] if temporal_compression is None \
        else int(temporal_compression)
    if int(tc) == int(vae_h3_mod.H3_VIDEO_VAE_ARCH_FACTS["temporalScale"]):
        return int(vae_h3_mod.latents_for_frames(int(frames)))
    return int(geometry.latent_frames(int(frames), temporal_compression=int(tc)))


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
        #: ⚠️ 走 :func:`h3_latent_frames` ✗✗（不是通用式 ✓）：`flow:auto` 的 shift 必须按**同一个**
        #: 潜帧数算 ✓ —— 这里若用 `(n-1)//4+1` 会与 plan 里的 12 格差 2，同一个请求两条路算出两个 shift ✓✗。
        latent_frames = h3_latent_frames(geometry.snap_frames(request.seconds, fps=fps),
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

    旧路（``karras`` / ``exponential`` / ``ddim_uniform``）**照跑** ✓ 但会往
    ``plan.warnings`` 写一句"域不重叠" ✓ —— 想复现旧行为就显式传，别让它静默 ✓✗。
    （``normal`` 已从这一桶里拿出来 ✓：它是「timestep 等距」✓、而 flow 域里 timestep 就是 σ ✓
    ⇒ 属于正排程 ✓，见 :func:`schedules.normal_sigmas` ✓。）
    """
    spec = str(request.schedule or "flow").strip().lower()
    if is_flow_schedule(spec):
        return schedules.flow_sigmas_for_steps(plan.steps, shift=video_flow_shift_for(
            request, width=plan.width, height=plan.height, latent_frames=plan.latent_frames))

    # ⚠️ `simple` / `beta` 不是"另造一条曲线" ✓ 而是**在模型自带的梯子上取点** ✓
    #    （见 `schedules.simple_sigmas` ✓）⇒ 必须把 flow 的 shift 一起给下去 ✓，
    #    否则会落到几何梯子（σ 上界 14.6 ✗）⇒ 又变成"域不重叠" ✓✗。工作流侧 H3 模板的
    #    `scheduler` 就是 `simple` ✓，这条通路是"按工作流原参数跑"的必要条件 ✓。
    #    ⚠️ shift 只能取**视频默认** ✗：`simple` / `beta` 这形里没有放 shift 的位置 ✓，
    #    而 `video_flow_shift_for` 对非 flow 名会**当场抛错** ✓✗（本轮就栽在这里：`simple`
    #    本该走取点路 ✓，却因为多问了它一句 shift 而整单被拒 ✓）。要扫 shift 就用
    #    `flow:<数>` ＋ 别的曲线用 `flow:` 名表达不了 ⇒ 视频侧 shift 保持 12.0 ✓。
    if spec in ("simple", "beta"):
        return schedules.sigmas_for(plan.steps, spec, shift=H3_FLOW_SHIFT)
    # ⭐ ``normal`` **不进 EDM 桶** ✓（本轮改 ✓）：参考实现的 ``normal`` 是「timestep 等距」✓，
    #    而 flow 域里 timestep 就是 σ ✓ ⇒ 它是**正经的 flow 排程**（σ 等距 ✓），不是"EDM 曲线
    #    硬喂给 flow 模型" ✗。旧版让它在下面回落成 flow ＋ 告警 ✓✗ ⇒ 传了 normal 拿到 flow ✓。
    if spec == "normal":
        return schedules.normal_sigmas(plan.steps, shift=H3_FLOW_SHIFT)

    sigmas = schedules.sigmas_for(plan.steps, spec)
    if max(sigmas) > 1.0:
        plan.warnings.append(
            f"σ 调度 {spec!r} 的首值 {max(sigmas):.4g} 超出 flow 的 [0,1] ✓ ⇒ H3 主干会收到 "
            f"t = 1−σ = {1.0 - max(sigmas):.4g} ✓（**不报错**但起点比训练见过的都高 ✗）。"
            "视频族的默认已改成 flow（shift 12 ✓，见 `pipeline.H3_FLOW_SHIFT` ✓）——"
            "这里照你要的跑，只是如实标出域不重叠 ✓")
    return sigmas


def _image_flow_sigmas(request: GenerationRequest, plan: GenerationPlan, *,
                       shift: float) -> list[float]:
    """**图族**的 flow σ 日程 ✓ —— 默认就是「t 等分」那条 ✓，但**认得工作流的调度名** ✓。

    工作流侧 8 张图（01/02/03/20 + 16/17/18/19/21 ✓）的 KSampler 都写 ``scheduler="simple"`` ✓；
    而 ``simple`` 在 flow 模型上 = **同一条梯子上等距取点** ✓ ⇒ 与默认结果几乎逐位相同 ✓
    （``steps`` 整除 1000 时完全相同 ✓），所以这里接 ``simple`` / ``beta`` **不改默认数值** ✓。

    ⭐ ``normal`` **在这里是真排程** ✓（本轮改 ✓）：参考实现的 ``normal_scheduler`` 语义是
    「**timestep 等距**」✓，而 flow 族的 ``multiplier = 1`` ⇒ ``timestep`` 就是 σ 本身 ✓ ⇒
    落到 σ 上是 **σ 等距** ✓（**不是** t 等距 ✓✗，也不是 EDM 域那条 log 等距 ✓✗）——
    见 :func:`schedules.normal_sigmas` 的推导 ✓。旧版把它和 ``karras`` 一起当"EDM 专用"回落 ✓✗，
    于是「传了 normal」拿到的是「flow」✓ 且只留一句告警 ✓（不报错、只是不是你要的那条 ✓✗）。

    ⚠️ 剩下 ``karras`` / ``exponential`` / ``ddim_uniform`` **仍是回落** ✓：它们的 EDM 默认端点
    （``0.0292 / 14.6146``）套到 flow 上会给出 ``t = 1−σ < 0`` ✓✗；真要那条曲线得把端点换成
    :func:`schedules.flow_sigma_bounds` 那一对 ✓ —— 那是**换数值** ✗，不能顺手做 ✓（会静默改掉
    已有出图 ✓✗）⇒ 这里**按 flow 跑并如实报** ✓（不静默 ✓、也不因为一个调度名就拒掉整单 ✓）。

    ⚠️ ``denoise<1`` 的**σ 截断也只在这里做** ✓✗（五个图族的唯一出口 ✓）：梯子按名义步数
    ``int(steps/denoise)`` 算 ✓、再取尾巴 ``[-(steps+1):]`` ✓ —— 参考实现
    ``KSampler.set_steps`` 逐字如此 ✓。``plan.total_steps`` 一并落盘 ✓（后端与回执都读它 ✓）。
    """
    spec = str(request.schedule or "flow").strip().lower()
    denoise, steps, total = _denoise_split(request, plan)
    plan.denoise, plan.total_steps = denoise, total
    if spec in ("", "flow") or spec.startswith("flow:"):
        ladder = schedules.flow_sigmas_for_steps(total, shift=shift)
    elif spec in ("simple", "beta"):
        ladder = schedules.sigmas_for(total, spec, shift=shift)
    elif spec == "normal":
        ladder = schedules.normal_sigmas(total, shift=shift)
    else:
        plan.warnings.append(
            f"σ 调度 {spec!r} 属 EDM 域（σ 上界 ~14.6 ✓），图族走的是 **flow** 域（σ∈[0,1] ✓）⇒ "
            f"已按 flow(shift={shift:g}) 跑 ✓（要换曲线请用 simple / beta / normal ✓）")
        ladder = schedules.flow_sigmas_for_steps(total, shift=shift)
    return _denoise_slice(ladder, steps)


def _denoise_split(request: GenerationRequest, plan: GenerationPlan) -> tuple[float, int, int]:
    """``(denoise, 实跑步数, 名义步数)`` ✓ —— 全仓**唯一**做 ``int(steps/denoise)`` 算术处 ✗✗。

    参考实现逐字（``comfy/samplers.py::KSampler.set_steps`` ✓）::

        new_steps = int(steps / denoise)
        sigmas = self.calculate_sigmas(new_steps)[-(steps + 1):]

    ⚠️ 用 ``int`` **截断**而不是四舍五入 ✗：0.55/20 步时两者都是 36 ✓，但 0.3/20 步是 66 vs 67 ✓ ——
    与工作流对拍时差一步就不同图 ✓，所以照抄截断 ✓。
    """
    try:
        denoise = float(request.denoise)
    except (TypeError, ValueError):
        denoise = 1.0
    steps = max(1, int(plan.steps or request.steps or 1))
    if denoise >= 1.0:
        return 1.0, steps, steps
    return denoise, steps, max(steps, int(steps / denoise))


def _denoise_slice(sigmas: list[float], steps: int) -> list[float]:
    """取 σ 梯子的**尾巴** ✓ = ``sigmas[-(steps+1):]`` ✓（参考实现逐字 ✓；末位 0 恒保住 ✓）。

    ⚠️ 梯子比要的步数短时（``denoise`` 贴近 1 ✓）**原样返回** ✓ 而不是硬凑 ✗ —— 硬凑会造出
    重复的 σ ✓✗（相邻两步一个噪声水平 ⇒ 白跑一步且不报错 ✓）。
    """
    values = list(sigmas)
    if steps <= 0 or steps + 1 >= len(values):
        return values
    return values[-(steps + 1):]


def _canvas_override(request: GenerationRequest) -> canvas_mod.CanvasPlan | None:
    """请求里的画布覆盖 ⇒ :class:`canvas.CanvasPlan` ✓；**没点名 ⇒ ``None``** ✓（走旧路径 ✓）。

    ⚠️ 这是**唯一**读 ``request.canvas`` / ``request.target`` 的地方 ✗✗ ——
    五个图族 + 视频各自去读一次，就会出现「六处口径」（正是本轮要收掉的东西 ✓）。

    ⚠️ ``target`` 命中档位时**不传 request.ratio** ✗✗：那样默认的 ``"16:9"`` 会把
    ``2048×1536``（4:3 ✓）的定妆板**按 16:9 重算成别的尺寸** ✓✗（而调用方其实只是没改过 ratio ✓）。
    ⇒ 档位像素优先 ✓、比例不一致**如实报** ✓；真要换画幅就走 ``/engine/canvas-plan?ratio=…`` ✓。
    """
    planned = request.canvas
    if planned is not None and request.target not in (None, ""):
        raise StageError(
            "plan",
            f"canvas 与 target **同时**给了 ✗（target={request.target!r} ✓）⇒ 两处口径 ✓✗；"
            "二选一 ✓：已完成计划就只给 canvas ✓、要引擎查表就只给 target ✓")
    if planned is not None:
        if not isinstance(planned, canvas_mod.CanvasPlan):
            raise StageError(
                "plan",
                f"canvas 必须是 :class:`canvas.CanvasPlan` ✗（收到 {type(planned).__name__} ✓）—— "
                "先走 ``POST /api/v1/engine/canvas-plan`` 拿到计划 ✓ 再原样传回来 ✓")
        return planned
    if request.target not in (None, ""):
        try:
            resolved = canvas_mod.plan_canvas(request.stage, target=request.target, ratio=None)
        except ValueError as err:
            raise StageError("plan", str(err), cause=err) from err
        wanted = geometry.parse_ratio(request.ratio)
        base = resolved.width / resolved.height
        if abs(wanted - base) > canvas_mod.RATIO_TOLERANCE * base:
            resolved = dataclass_replace(resolved, notes=resolved.notes + (
                f"⚠️ 档位画幅是 {resolved.ratio}（{base:.4f} ✓），而请求 ratio="
                f"{request.ratio!r}（{wanted:.4f} ✓）差 "
                f"{abs(wanted - base) / base * 100:.2f}% ⇒ **档位像素优先** ✓（target 的语义就是"
                "「精确复现该用途档」✓）；要按别的画幅出图请走 ``/engine/canvas-plan?ratio=…`` ✓",))
        return resolved
    return None


def _resolution_selector_size(megapixels: float, ratio: Any, *, multiple: int,
                              plan: GenerationPlan) -> tuple[int, int]:
    """尺寸口径 = **ComfyUI ``ResolutionSelector`` 节点** ✓（见 :func:`geometry.size_for_resolution_selector` ✓）。

    为什么要有（2026-11-01 ✓）：本仓的 :func:`geometry.size_for_megapixels` 是**按实测反推**的
    近似（``ceil`` + ``1e6`` 基数 ✓），``1.0MP@1:1`` 会给 **1008×1008** ✓；ComfyUI 的节点给
    **1024×1024** ✓（``round`` + ``1024²`` 基数 ✓）。

    ⚠️ 但它**不是"图族统一换口径"** ✗✗ —— 只有**官方画布正好等于节点输出**的那几档才该用 ✓：
    ``zimage``（官方 1024² ✓）、``krea2``（官方 1024² ✓，旧的近似口径给 1040 ✓✗）。
    而 ``qwen_image`` / ``qwen_image21`` 的官方画布是 **1328×1328** ✓（``EmptySD3LatentImage`` ✓），
    节点口径给的是 **1360×1360** ✓ —— 换过去就是**把工作流模板的画布改掉 2.4%** ✓✗
    （多裁一条边、且与模板 `#7.width/height` 对不上 ⇒ 逐位对拍永远差一截 ✓）。
    ⇒ 这两族**仍走** :func:`geometry.size_for_megapixels` ✓；SDXL 那条也不动 ✗
    （训练预算与实测显存/耗时绑死 ✓）。
    """
    width, height = geometry.size_for_resolution_selector(megapixels, ratio, multiple=multiple)
    # ⚠️ 只在**两家算出来的不一样**时才写一条 ✓（一样的档位就别刷噪音 ✗）——
    #    1:1 的 1.76 MP 就是这种：两边都给 1328 ✓。
    old = geometry.size_for_megapixels(megapixels, ratio, multiple=multiple)
    if (int(old[0]), int(old[1])) != (width, height):
        plan.warnings.append(
            f"尺寸口径已按 **ComfyUI ResolutionSelector**（round + 1024² 基数 ✓）："
            f"{float(megapixels):.3g} MP @ {ratio} ⇒ **{width}×{height}** ✓"
            f"（旧的近似口径给 {int(old[0])}×{int(old[1])} ✓）")
    return width, height


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

    # ⚠️ SDXL 的画布**不能被覆盖** ✗✗（不是"忘了接" ✓）：它的尺寸由**训练预算**收口 ✓，
    #    外面再塞一个 1344×768 就是把它推回"低于/高于训练档"那两个不报错的坏区 ✓✗。
    #    ⇒ 明确拒 ✓ 并说清替代路径 ✗（静默忽略 = 用户以为生效了 ✓✗）。
    if _canvas_override(request) is not None:
        raise StageError(
            "plan",
            f"SDXL 的画布**不接受** canvas / target ✗✗ —— 它只能按训练预算收口："
            f"{image_mod.SDXL_TRAINED_RESOLUTION}² = {image_mod.SDXL_TRAINED_MEGAPIXELS:g} MP ✓"
            "（低 ⇒ 物体重复 + 霓虹过饱和 ✓；高 ⇒ 解码 reserved 超物理显存 ⇒ 共享显存换页 ✓ —— "
            "**两个方向都不报错** ✓✗）。要更大像素走**超清** ✓；要按工作流档位出图请换 stage"
            "（zimage / qwen_image / krea2 ✓）")

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
    # ⚠️ ``denoise<1`` ⇒ 梯子按**名义步数**算 ✓ 再取尾巴 ✓（见 :func:`_denoise_split` ✓）——
    #    SDXL 这条同样要切 ✓✗：不切就是"从 σ 上界起跑"但回执说 0.55 ✓（自欺 ✓）。
    denoise, steps, total = _denoise_split(request, plan)
    plan.denoise, plan.total_steps = denoise, total
    try:
        plan.sigmas = _denoise_slice(image_mod.sdxl_sigmas_for_steps(total, schedule=schedule), steps)
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


def _warn_off_official_canvas(plan: GenerationPlan, official_mp: float,
                              title: str) -> None:
    """画布**明显偏离**这族的官方训练档 ⇒ 如实说一句 ✓（**不拦** ✗：显式给的像素预算照跑 ✓）。

    ⚠️ 为什么必须有一条 ✗✗：各族的 `_fill_*_plan` 只在「等于 `GenerationRequest` 的默认 0.65」
    时才换成官方档 ✓ —— 用户**明写** 0.2 / 0.1 MP 就原样照跑 ✓✗（实测 zimage 0.2 MP @16:9
    ⇒ **608×336**，那是 1024² 的 1/5 面积 ✓）。训练档位外的尺寸是拿**没训过**的区域换画质 ✓，
    症状是构图崩 / 多主体重复 / 细节糊成一片 —— 而它**不报错** ✓✗，
    不显式标出来，用户只会以为"只是小一点/大一点" ✓✗。

    ⚠️ 容差写宽（±30~60%）✓：ResolutionSelector 的 round 会让 1.0 MP @16:9 落到 1360×768
    （=1.02 MP ✓）、@9:16 落到 768×1360 ✓ —— 这些都是**同一档**，不该报警 ✗（见 ×0.7 / ×1.6）。
    """
    actual_mp = (int(plan.width) * int(plan.height)) / (1024 * 1024)
    if official_mp * 0.7 <= actual_mp <= official_mp * 1.6:
        return
    direction = "低于" if actual_mp < official_mp else "高于"
    plan.warnings.append(
        f"画布 **{plan.width}×{plan.height}**（{actual_mp:.2f} MP）{direction} {title} 的"
        f"训练档位 **{official_mp:g} MP** ✓ —— 训练档位外的尺寸是拿**没训过**的区域换画质 ✓✗，"
        "症状是构图崩 / 多主体重复 / 细节糊，而且**不报错** ✓✗；这里照你要的跑 ✓，只如实标出来 ✗"
        "（要小图请按官方档生成后再缩 ✓ —— 反过来缩不回来 ✓✗）")


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
    override = _canvas_override(request)
    if override is not None:
        plan.width, plan.height = override.width, override.height
        plan.canvas = override.to_dict()
        plan.warnings.extend(override.notes)
    else:
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
        grid = 16                       # 8（VAE）× 2（patch）✓
        plan.width, plan.height = _resolution_selector_size(wanted_mp, request.ratio,
                                                           multiple=grid, plan=plan)
        # ⚠️ **显式**给了个远离官方档位的像素预算 ⇒ 必须如实说 ✗✗（0.2 MP @16:9 实测落 608×336，
        #    那是训练档位 1024² 的 1/5 面积 ✓，而它**不报错** ✓✗）
        _warn_off_official_canvas(plan, 1.0, "Z-Image turbo")

    plan.frames = 1
    plan.fps = 0                    # 单张图 ✓ 没有帧率 ✓
    plan.steps = int(request.steps)
    shift = schedules.FLOW_SHIFT_DEFAULT
    plan.sigmas = _image_flow_sigmas(request, plan, shift=shift)
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
    override = _canvas_override(request)
    if override is not None:
        plan.width, plan.height = override.width, override.height
        plan.canvas = override.to_dict()
        plan.warnings.extend(override.notes)
    else:
        wanted_mp = float(request.megapixels)
        # ⚠️ 与 Z-Image 同一条纪律 ✗：`GenerationRequest.megapixels` 的默认是 **0.65** ✓（视频档位 ✓），
        #    对 2.1 没有意义 ✗ ⇒ 只能拿"**等于默认值**"判 ✓ 并**如实报改** ✓（不静默换画布 ✗）。
        default_mp = 0.65
        turned = abs(wanted_mp - default_mp) <= 0.01 * default_mp
        if turned:
            #: 2.1 官方示例档 ≈ **1.76 MP** ✓（`image_qwen_image.json` 一族模板同档 ✓）。
            #: ⚠️ 长宽比仍**跟着 `request.ratio`** 走 ✓（16:9 的 1.76 MP ⇒ 1792×1024 ✓，不是硬塞成方的 ✗）。
            wanted_mp = 1.76
        grid = 16                       # VAE 16× ✓（**没有** patch ×2 ✗✗）
        #: ⚠️ `size_for_megapixels` 的**默认吸附步长是 32** ✗（全局 `RESOLUTION_MULTIPLE` ✓，
        #: 那是给有 patchify 的族用的 ✓）⇒ 这里必须**显式传 16** ✗✗：走默认会让 2.1 也被吸到 32 ✓
        #: —— 不报错 ✓、只是每张白裁一条边 ✓（"按 16 算"就成了空话 ✓✗）。
        plan.width, plan.height = geometry.size_for_megapixels(
            wanted_mp, request.ratio, multiple=grid)
        if turned:
            # ⚠️ 回执里的数字必须是**算出来的那个** ✗✗：以前写死"1:1 时即官方 1328×1328"，
            #    一旦尺寸口径换了（比如 `ResolutionSelector` 的 1024² 基数 ⇒ 1360 ✓）这句就成了
            #    "同一张回执里一句说 1328、下一句说 1360" ✓✗（读的人只能猜哪个是画的尺寸 ✓）。
            plan.warnings.append(
                "像素预算仍是 `GenerationRequest` 的默认 0.65 MP ✓（视频口径 ✗）⇒ 已按 Qwen-Image 官方"
                f"示例档换成 **1.76 MP** ✓（1:1 实际落 **{plan.width}×{plan.height}** ✓，"
                "官方示例图是 1328×1328 ✓；长宽比仍跟 ratio ✓）；"
                "要保持视频那档请显式传 megapixels ✓")
        # 显式给了远离官方示例档的预算 ⇒ 如实说（同 zimage / krea2 的理由 ✓；这族官方 ≈1.76 MP ✓）
        _warn_off_official_canvas(plan, 1.76, "Qwen-Image 2.1")

    plan.frames = 1
    plan.fps = 0                    # 单张图 ✓ 没有帧率 ✓
    plan.steps = int(request.steps)
    shift = QWEN21_FLOW_SHIFT
    plan.sigmas = _image_flow_sigmas(request, plan, shift=shift)
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
    override = _canvas_override(request)
    if override is not None:
        plan.width, plan.height = override.width, override.height
        plan.canvas = override.to_dict()
        plan.warnings.extend(override.notes)
    else:
        wanted_mp = float(request.megapixels)
        default_mp = 0.65                # `GenerationRequest` 的默认是视频档 ✓（同 Z-Image / 2.1 的理由 ✓）
        turned = abs(wanted_mp - default_mp) <= 0.01 * default_mp
        if turned:
            #: 2512 官方模板 ``EmptySD3LatentImage`` = **1328×1328** ✓ ⇒ 1.76 MP ✓（与 2.1 同一档位 ✓）。
            wanted_mp = 1.76
        grid = 16                        # VAE 8× ✓ × DiT patch 2× ✓（理由与 2.1 不同，见 docstring ✓）
        plan.width, plan.height = geometry.size_for_megapixels(
            wanted_mp, request.ratio, multiple=grid)
        if turned:
            # ⚠️ 同 2.1：回执里点名的画布必须是**本次算出来的** ✗✗（写死 1328 ⇒ 口径一换就成了
            #    "回执与产物两个数" ✓✗）。
            plan.warnings.append(
                "像素预算仍是 `GenerationRequest` 的默认 0.65 MP ✓（视频口径 ✗）⇒ 已按 Qwen-Image 2512 "
                f"官方模板换成 **1.76 MP** ✓（1:1 实际落 **{plan.width}×{plan.height}** ✓，"
                "官方模板是 1328×1328 ✓；长宽比仍跟 ratio ✓）")
            # ⚠️ 这一条是**本轮加的** ✓：1.76 MP 只是"官方模板档"✓，而**生产线实测**用的是两档
            #    别的画布 ✓✗（定妆板 2048×1536 ✓ / 场景卡 1344×768 ✓，见 `profiles.PROFILES` ✓）
            #    —— 不说明的话，「照官方档跑」和「照生产线跑」出了两张不一样的图，谁都看不出是谁定的 ✓✗。
            plan.warnings.append(
                "⚠️ 本族**生产线实测档**不是这一张画布 ✓：" + " / ".join(
                    f"{item.label} {item.width}×{item.height}" for item in profiles_mod.for_stage(
                        QWEN_IMAGE_STAGE) if item.width and item.height)
                + " ✓ ⇒ 要对齐工作流请传 ``target``（如 ``target=\"character-sheet\"`` ✓）"
                "或先走 ``/engine/canvas-plan`` ✓（引擎**不替**你选用途 ✓✗）")

    plan.frames = 1
    plan.fps = 0                         # 单张图 ✓
    plan.steps = int(request.steps)
    shift = QWEN_IMAGE_FLOW_SHIFT
    plan.sigmas = _image_flow_sigmas(request, plan, shift=shift)
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


def _fill_krea2_plan(request: GenerationRequest, plan: GenerationPlan) -> None:
    """**Krea2** 单张图片的 plan ✓（flow matching ✓ μ=**1.15** ⇒ 底数 **e^1.15≈3.158** ✓）。

    与另四套图片族的差别（逐条对着盘上的东西核过 ✓）：

    | | SDXL ✓ | Z-Image ✓ | 2.1 ✓ | 2512 ✓ | **Krea2** ✓ |
    |---|---|---|---|---|---|
    | flow shift | 离散扩散 ✗ | 3 ✓ | 0.69 ✓ | 3.1 ✓ | **μ 按分辨率插值 ✓**（1024² ⇒ 1.15 ⇒ 3.158 ✓） |
    | 吸附步长 | 8 ✓ | 16 ✓ | 16 ✓ | 16 ✓ | **16** ✓ |
    | 潜通道 | 4 ✓ | 16 ✓ | 64 ✓ | 16 ✓ | **16** ✓ |
    | 官方步/引导 | — | 8 / 1 ✓ | 2.5~4 ✓ | 50 / 4.0 ✓ | **8 / 1.0** ✓ |

    ⚠️ **数字写法是 μ、不是底数** ✗✗（见 :func:`krea2_base_shift` ✓）：其余四族的 shift
    都是 ``ModelSamplingAuraFlow`` 的**底数** ✓，只有这一族走 ``ModelSamplingFlux`` ✓
    ⇒ 1.15 要过 ``exp`` 才是底数 ✓。当底数喂下去**不报错** ✓、σ 曲线就是不弯 ✓✗。

    ⚠️ 三处**别族能套、这一族会错**的地方 ✓✗：

    ① **吸附 16 = 「VAE 8× × DiT patch 2」** ✓ —— 与 **2512 同构** ✓、与 **2.1 不同** ✗
       （2.1 没有 patchify ✓，VAE 就是 8× ✓ ⇒ 它吸 16 是保守 ✓，不是必需 ✓）：
       ``SingleStreamDiT.patch_size = 2`` ✓ ⇒ 潜边长必须是**偶数** ✓。
       ⚠️ 两个方向都别写错 ✗✗：吸到 **32** 只是白裁边 ✓（不报错 ✓、白算 ✓）；
       反过来把 8× 当 16× 算 ⇒ 潜变量小一半 ⇒ 解码出**缩小一圈又糊**的图 ✓✗；
    ② **``cfg = 1.0`` 在 Krea2 上是「对」的** ✓✗ —— 与 2.1 / 2512 恰好相反 ✓：
       盘上这一档是 **turbo 蒸馏版** ✓（模板 ``image_krea2_turbo_t2i`` = 8 步 / cfg 1 ✓）
       ⇒ 这里**不该**报"不引导会糊" ✓；该报的是反过来的「cfg>1 白花一倍算力」✓；
    ③ **步数别按视频那 30 步走** ✓：蒸馏档 8 步 ✓，30 步不会报错 ✓、只是慢 3.75 倍且不会更好 ✓。
    """
    override = _canvas_override(request)
    if override is not None:
        plan.width, plan.height = override.width, override.height
        plan.canvas = override.to_dict()
        plan.warnings.extend(override.notes)
    else:
        wanted_mp = float(request.megapixels)
        default_mp = 0.65                # `GenerationRequest` 的默认是视频档 ✓（同另四族的理由 ✓）
        if abs(wanted_mp - default_mp) <= 0.01 * default_mp:
            #: 官方模板 ``image_krea2_turbo_t2i`` 的 ``EmptyLatentImage`` = **1024×1024** ✓ = **1.00 MP** ✓
            #: （``ResolutionSelector`` 口径下 1.00 MP @1:1 **正好** 1024 ✓；旧值 1.05 是按"向上吸 16"
            #: 反推出来的 1040 档 ✓✗，换口径后旧值反而会多出 1056 = 白算 6% 像素 ✗）。
            #: ⚠️ 比 2.1/2512 那 1.76 MP **低** ✓✗（别顺手抄成 1.76 ✓：那会把 1328 档画布塞给 1024 档的模型 ✓）。
            wanted_mp = 1.0
            plan.warnings.append(
                "像素预算仍是 `GenerationRequest` 的默认 0.65 MP ✓（视频口径 ✗）⇒ 已按 Krea2 turbo "
                "官方模板换成 **1.00 MP** ✓（2026-11-01 起按 ``ResolutionSelector`` 口径算 ⇒ 1:1 就是"
                "**1024×1024** ✓，与官方画布**逐像素相同** ✓ —— 旧的 ``1.05 MP + 向上吸 16`` 那套会落到 "
                "1040×1040 ✓✗，已废 ✗）；长宽比仍跟 ratio ✓；要保持视频那档请显式传 megapixels ✓")
            # ⚠️ 1024² 是**模板默认** ✓，而生产线上这一族实际用的是 2048×1536 / 1344×768 / 720×1280 ✓✗
            #    —— 差 3 倍的画布不会报错，只会"风格不像" ✓✗（见 `profiles.PROFILES` 的出处 ✓）。
            plan.warnings.append(
                "⚠️ 本族**生产线实测档**不是这一张画布 ✓：" + " / ".join(
                    f"{item.label} {item.width}×{item.height}" for item in profiles_mod.for_stage(
                        KREA2_STAGE) if item.width and item.height)
                + " ✓ ⇒ 要对齐工作流请传 ``target``（如 ``target=\"scene-card\"`` ✓）"
                "或先走 ``/engine/canvas-plan`` ✓（引擎**不替**你选用途 ✓✗）")
        #: 16 = VAE 8× ✓ × DiT patch 2× ✓（潜边长必须是偶数 ✓，同 2512 的算术 ✓）。
        grid = 16
        plan.width, plan.height = _resolution_selector_size(wanted_mp, request.ratio,
                                                            multiple=grid, plan=plan)
        # 显式给了远离 1024² 的预算 ⇒ 如实说（同 zimage 的理由 ✓；这一族的训练档也是 1.0 MP ✓）
        _warn_off_official_canvas(plan, 1.0, "Krea2 turbo")

    plan.frames = 1
    plan.fps = 0                         # 单张图 ✓ 没有帧率 ✓
    plan.steps = int(request.steps)
    #: ⚠️ 喂的是**底数** ``e^μ`` ✓（见 :func:`krea2_base_shift` 的说明 ✗）—— 别写 ``shift=1.15`` ✗。
    #: ⚠️ μ **按实际分辨率插值** ✓（工作流里 `ModelSamplingFlux` 的 width/height 是**连线** ✓）：
    #: 1024² ⇒ 1.15 ✓（老值，逐位不变 ✓）；512² ⇒ 0.6266 ✓；≥4096 序列长一律 1.15 ✓。
    mu = krea2_shift_mu(plan.width, plan.height)
    shift = krea2_base_shift(mu)
    plan.sigmas = _image_flow_sigmas(request, plan, shift=shift)
    plan.timesteps = [float(value) for value in plan.sigmas]
    plan.warnings.append(
        f"{plan.width}×{plan.height} 单张图片 ✓、{plan.steps} 步 **flow matching** ✓"
        f"（μ={mu:.4f} ⇒ 底数 {shift:.4f} ✓ **不是** 2.1 的 0.69 / 2512 的 3.1 / "
        f"Z-Image 的 3 ✓✗；σ {plan.sigmas[0]:.4g} ⇒ 0 ✓；timestep 即 σ ✓ 不乘 1000 ✗）")
    if plan.steps != 8:
        plan.warnings.append(
            f"步数 {plan.steps} ≠ 官方 turbo 模板的 **8 步** ✓（``image_krea2_turbo_t2i`` 的 "
            f"``KSampler`` = 8 步 / cfg 1.0 / euler simple ✓）—— 蒸馏档跑更多步**不会更好** ✓，"
            "只是更慢 ✓（这里只报不改 ✗）")
    cfg_scale = float(getattr(request.guidance, "scale", 1.0) or 1.0)
    if cfg_scale > 1.0:
        # ⚠️ 与 2.1 / 2512 **正好相反** ✗✗：那两族是「cfg=1 ⇒ 糊」，这一族是「cfg>1 ⇒ 白花一倍算力」✓。
        plan.warnings.append(
            f"引导 cfg={cfg_scale:g} > 1.0 ⇒ Krea2 这一档是 **turbo 蒸馏版** ✓（官方模板 cfg **1.0** ✓）"
            "⇒ 引导**能跑** ✓、但每步两次模型调用 ✗（12.82B 的主干 ⇒ 单个 1024² 出图**直接翻倍** ✓✗），"
            "且蒸馏档上引导常常只让画面更「硬」✓（这里只报不改 ✗）")


def _fill_seedvr2_plan(request: GenerationRequest, plan: GenerationPlan) -> None:
    """**SEEDVR2 视频修复**的 plan ✓（flow matching ✓ shift=**1.0** ✓ 不插值 ✓）。

    与其他各族的差别（这一族**不生成、只修** ✓）：

    | | H3 ✓ | Z-Image ✓ | 2.1 ✓ | 2512 ✓ | Krea2 ✓ | **SEEDVR2** ✓ |
    |---|---|---|---|---|---|---|
    | 输入 | 文本 ✓ | 文本 ✓ | 文本 ✓ | 文本/图 ✓ | 文本 ✓ | **一段退化视频 ✓**（没有就拒 ✓） |
    | 文本塔 | Qwen3-VL ✓ | Qwen3-4B ✓ | Qwen3.5-9B ✓ | Qwen2.5-VL ✓ | Qwen3-VL-4B ✓ | **没有** ✗（条件烤在权重里 ✓） |
    | 主干入参 | 两流 ✓ | 4 维 ✓ | 4 维 ✓ | 4 维 ✓ | 4 维 ✓ | **潜 16 ＋ 条件 17 ⇒ vid_in 33** ✓ |
    | flow shift | 12 ✓ | 3 ✓ | 0.69 ✓ | 3.1 ✓ | μ 1.15 ✓ | **1.0** ✓ |
    | 官方步数 | — | 8 ✓ | 2.5~4 ✓ | 50 ✓ | 8 ✓ | **1** ✓ |

    ⚠️ 三处**别族能套、这一族会错**的地方 ✓✗：

    ① **尺寸/帧数都得从源视频推** ✗（不是从 ``megapixels``/``ratio``/``seconds`` ✓）：
       修复的语义是「把这些帧修好」✓ ⇒ 改尺寸就是改素材 ✗ ⇒ 一律走 :func:`seedvr2_target` ✓；
    ② **步数别按 ``GenerationRequest`` 默认的 30 步** ✗✗：这一档是**一步蒸馏** ✓
       （``NaDiT`` 采样 1 步 ✓）⇒ 30 步不报错 ✓、只是慢 30 倍且**更糊** ✓✗。
       未显式改过（仍是默认 30 ✓）就换成 1 并**如实报** ✓；显式传了就照跑 ✓（只报不改 ✗）；
    ③ **不允许 CFG** ✗：没有文本塔 ⇒ 没有负向 ✓ ⇒ ``cfg > 1`` 只能是"两次相同前向"
       （精确地白花一倍算力 ✓）或读一份不存在的负向 ✓✗ ⇒ 这里**明确报**得很轻、但不拒 ✓
       （拒会让"沿用图片那套请求体"的调用方整单失败 ✓，而多花一倍算力是可接受的降级 ✓）。
    """
    from . import media as media_mod   # noqa: PLC0415 —— 纯 ffprobe ✓ 不引 torch ✓（plan 期零 torch ✓）

    videos = [str(item) for item in (getattr(request, "reference_videos", ()) or ())
              if str(item).strip()]
    if not videos:
        raise StageError(
            "plan",
            "SEEDVR2 是**修复**族 ✗ —— 它修的是**你给的那段视频** ✓，"
            "请求里必须带 `referenceVideos`（第一路即被修素材 ✓）。\n"
            "  · 没有源视频 ⇒ 本阶段**没有**任何可修的东西 ✓ ⇒ 当场拒 ✓"
            "（**不**退化成文生视频 ✗ —— 那会出一段内容无关的片子 ✓、还看不出错 ✗）；\n"
            "  · 要按文字生成视频请用 `stage=\"h3\"` ✓（H3 双流 ✓）或对应图族 ✓。")
    if len(videos) > 1:
        raise StageError(
            "plan",
            f"SEEDVR2 一次只修**一条**视频 ✗（收到 {len(videos)} 路 ✓）—— 多路不受支持 ✓，"
            "**不**静默只取第一路 ✗（那会让你以为其余几段也修好了 ✓✗）；请逐条调用 ✓ "
            "或改用 `stage=\"h3\"` 的参考视频 ✓")
    try:
        facts = media_mod.probe(videos[0])
    except Exception as err:  # noqa: BLE001 —— ffprobe 的所有失败都归「读不了这份素材」✓
        raise StageError("plan", f"读不了源视频 ✗：{type(err).__name__}: {err}"
                                 f"（`referenceVideos[0]` = {videos[0]!r} ✓）", cause=err) from err
    try:
        target = seedvr2_target(facts)
    except ValueError as err:
        raise StageError("plan", str(err), cause=err) from err

    plan.width, plan.height = int(target["width"]), int(target["height"])
    plan.frames = int(target["frames"])
    plan.fps = int(target["fps"])
    plan.latent_frames = int(target["latent_frames"])
    plan.warnings.extend(target["warnings"])
    #: ⚠️ 步数：默认 30 是**视频/图片族的**默认 ✓（``GenerationRequest.steps`` ✓）⇒ 这一族要换 ✓。
    default_steps = 30
    plan.steps = int(request.steps)
    if plan.steps == default_steps:
        plan.steps = SEEDVR2_OFFICIAL_STEPS
        plan.warnings.append(
            f"步数仍是 ``GenerationRequest`` 的默认 **{default_steps} 步** ✓（别的族的口径 ✗）⇒ "
            f"已按本族**一步蒸馏**换成 **{SEEDVR2_OFFICIAL_STEPS} 步** ✓（``NaDiT`` 1 步 ✓）；"
            "要走多步请显式传 `steps` ✓（多步**不会更好** ✓，只是更慢 ✓）")
    elif plan.steps != SEEDVR2_OFFICIAL_STEPS:
        plan.warnings.append(
            f"步数 {plan.steps} ≠ 官方 **{SEEDVR2_OFFICIAL_STEPS} 步** ✓ —— 蒸馏档跑多步**不会更好** ✓，"
            "只会成倍变慢 ✓（这里只报不改 ✗）")
    plan.sigmas = _image_flow_sigmas(request, plan, shift=SEEDVR2_FLOW_SHIFT)
    plan.timesteps = [float(value) for value in plan.sigmas]
    plan.warnings.append(
        f"修 {plan.frames} 帧 @{plan.fps}fps ≈ {plan.frames / max(1, plan.fps):.2f}s ✓ ⇒ "
        f"**{plan.width}×{plan.height}** ✓（潜变量 {plan.latent_frames} 帧 × "
        f"{plan.height // 8}×{plan.width // 8} ✓、主干另吃 **17 通道**条件 ✓ ⇒ `vid_in` 33 ✓）；"
        f"{plan.steps} 步 **flow matching** ✓（shift={SEEDVR2_FLOW_SHIFT:g} ✓ **不是** H3 的 12 / "
        f"2.1 的 0.69 / 2512 的 3.1 / Krea2 的 μ1.15 ✓✗；σ {plan.sigmas[0]:.4g} ⇒ 0 ✓；"
        "timestep 即 σ ✓ 不乘 1000 ✗）")
    cfg_scale = float(getattr(request.guidance, "scale", 1.0) or 1.0)
    if cfg_scale > 1.0:
        plan.warnings.append(
            f"引导 cfg={cfg_scale:g} > 1.0 ⇒ 本族**没有文本塔** ✗（条件烤在权重里 ✓，"
            "``positive_conditioning`` / ``negative_conditioning`` 都是权重自带的固定张量 ✓）"
            "⇒ 这次引导**不改变任何结果** ✓、只让每步多跑一次前向 ✗（精确地白花一倍算力 ✓）")
    if request.negative:
        plan.warnings.append(
            "负向提示词被**忽略** ✓ —— 本族没有文本塔 ✗（``negative`` 进不了任何前向 ✓）；"
            "要它生效只能用带文本塔的族 ✓")
    if request.first_frame or request.last_frame:
        plan.warnings.append(
            "`firstFrame` / `lastFrame` 被**忽略** ✓ —— 修复的起点是**源视频自己的潜变量** ✓，"
            "不是一张图 ✓✗（给它也不会报错 ✓，只是不生效 ✓）")


def _fill_music3_plan(request: GenerationRequest, plan: GenerationPlan) -> None:
    """**Music3** 的 plan ✓（caption ＋ 歌词 ⇒ 一段 44.1k 立体声 ✓）。

    这一族与其余各族有三处**结构性**不同 ✓，三条都不是「少写一行」而是「口径不同」✓：

    | | 图族四套 ✓ | 视频两族 ✓ | **Music3** ✓ |
    |---|---|---|---|
    | 产物 | 一张图 ✓ | 一段片子 ✓ | **一段音频** ✓ |
    | 宽高 | 必填 ✓ | 必填 ✓ | **恒 0** ✓（没有画布 ✓） |
    | ``frames`` 是什么 | 1 ✓ | 像素帧 @fps ✓ | **条件帧 @25Hz** ✓（latent 是 86.13Hz ✓） |

    ⚠️ 三条硬口径（每条都有具体的坏状态 ✓）：

    1. **必须有 caption** ✓：空 ``prompt`` 在这一族**照样跑完** ✓、出来的是一段没有语义的噪声 ✓
       —— 挡住它的是 ``build_plan`` 开头那条**共用**校验 ✓（唯一例外 SEEDVR2 ✓）；
       ⚠️ **不要在这里再写一遍** ✗✗：本轮第一版写了 ✓，实测**一次都进不去** ✓（共用的先拦 ✓）
       ⇒ 那是一段永不执行的分支 ✓（同一个判据写两处 ✓、其中一处永远是死码 ✓）；
    2. **``width``/``height`` 恒 0** ✓：别拿 1 填 ✓✗ —— ``to_dict`` 会按 ``frames/fps`` 算时长 ✓、
       但前端拿到 ``1×1`` 会以为是画布算错 ✓（症状：进度条旁写着「1 × 1」✓）；
    3. **σ 曲线有出处** ✓（2026-11-06 对完参考实现 ✓）：``MiniMaxMusic3`` 是 ``ModelType.FLOW``
       ⇒ ``ModelSamplingDiscreteFlow``、``shift=1.0``（``sampling_settings`` 无该键 ✓）、
       ``multiplier=1.0`` ⇒ ``time_snr_shift(1.0, t)=t`` 恒等 ⇒ ``normal`` 就是 **σ 等距**
       ＋ 端点 ``(1.0, 0.001)`` ✓；**步数不在模型里** ✗（在 ``KSampler`` 节点上 ✓）。
       照抄邻族的值不会报错 ✓、只会让歌越来越糊 ✓✗。
    """
    #: ⚠️ 空 caption 的拦截**不在这里** ✗ —— 见 :func:`build_plan` 开头的共用校验 ✓
    #: （唯一例外 SEEDVR2 ✓）。本轮第一版在这里又写了一遍 ✓，实测**一次都进不去** ✓ ⇒ 已删 ✓。
    try:
        seconds = float(request.seconds)
    except (TypeError, ValueError):
        seconds = 0.0
    if not seconds > 0:
        raise StageError(
            "plan", f"Music3 要 ``seconds`` > 0 ✗（收到 {request.seconds!r} ✓）—— "
                    f"这一族的时长**只有这一个入口** ✓（没有 ``frames`` ✗："
                    f"音频的「帧」是条件帧 @{MUSIC3_COND_FPS}Hz ✓，由秒数算出来 ✓）")

    cond = int(round(seconds * MUSIC3_COND_FPS))
    if cond > MUSIC3_MAX_AUDIO_FRAMES:
        raise StageError(
            "plan", f"太长 ✗（{seconds:g} 秒 ⇒ 条件 {cond} 帧 ✓ > 上限 "
                    f"{MUSIC3_MAX_AUDIO_FRAMES} ✓ ＝ {MUSIC3_MAX_AUDIO_FRAMES / MUSIC3_COND_FPS:g} 秒 ✓）"
                    "—— 文本塔的条件长度是**硬上限** ✓✗，超了不是慢而是编不出条件 ✓")
    if cond <= 0:
        raise StageError("plan", f"太短 ✗（{seconds:g} 秒 ⇒ 条件 {cond} 帧 ✓）")

    latent = music3_latent_frames(cond)
    #: 两式互核 ✓：``dit_music3.latent_length``（骨架 ✓）与 ``vae_music3.latent_frames_for_seconds``
    #: （解码侧 ✓）**各算一遍** ✓ —— 分叉时不报错、只出长度不对的歌 ✓✗，所以必须显式比 ✓。
    from_vae = int(vae_music3_mod.latent_frames_for_seconds(seconds))
    if from_vae != latent:
        plan.warnings.append(
            f"⚠️ 两条长度算式**分叉** ✓✗：plan 算 {latent} 帧 ✓、解码侧算 {from_vae} 帧 ✓"
            f"（差 {from_vae - latent:+d} ✓）—— 出片时长会与 ``seconds`` 对不上 ✓")

    plan.width = 0
    plan.height = 0
    plan.frames = cond
    plan.fps = float(MUSIC3_COND_FPS)
    plan.latent_frames = latent

    win = music3_window_plan(latent)
    plan.warnings.append(
        f"{seconds:g}s ⇒ 条件 {cond} 帧 @{MUSIC3_COND_FPS}Hz ⇒ latent {latent} 帧 @86.13Hz ✓；"
        f"主干按窗长 {win['window']} / 跳 {win['hop']} 走 **{win['count']} 个窗** ✓"
        f"（重叠 {win['window'] - win['hop']} 帧 ✓ 按权重叠加 ✓ —— 切窗在主干内部 ✓，"
        f"pipeline 不切 ✓✗）")
    if not win["covered"]:
        #: 这个分支理论上进不来 ✓（算式保证末窗到末端 ✓）；留着是因为「保证了」与「验过了」不同 ✓。
        raise StageError(
            "plan", f"窗覆盖算出来不闭合 ✗（latent {latent} ✓、末窗止于 {win['lastEnd']} ✓）"
                    "—— 尾部会被静默丢掉 ✓✗")

    lyrics = str(getattr(request, "lyrics", "") or "").strip()
    if not lyrics:
        plan.warnings.append(
            "没给歌词 ✓ ⇒ 按**纯器乐**出 ✓（``lyrics`` 为空时参考侧只喂 caption 支路 ✓）；"
            "要人声就传 ``lyrics`` ✓（与 caption 分开 ✓ 别混成一句 ✓✗）")
    else:
        plan.warnings.append(
            f"歌词 {len(lyrics)} 字 ✓ 走独立支路 ✓（与 caption 拼成两条 ✓ 不是一句话 ✓）")

    #: ⚠️ **引导这一格在本族是"记账"不是"旋钮"** ✗✗（2026-11-06 改 ✓，原文说「给 1.0 等于只留一条
    #: 文本支路、会明显更平」✗ —— 那是**错的** ✓✗，把两层 CFG 混成一层了 ✓）：
    #: · 本族的截断式引导发生在**自回归内部** ✓（``te_music3.generate`` 固定
    #:   ``cfg {CFG_SCALE}`` / top-``{CFG_TOP_K}`` ✓，一次生成里两条支路 ✓）；
    #: · DiT 那一层的引导**要负向条件** ✓，而本仓的 ``_encode_music3_condition`` 恒回
    #:   ``negative=None`` ✓（要负向就得**再跑一遍**整段自回归 ✓）⇒ ``pipeline`` 见
    #:   ``negative is None`` 就**跳过 CFG** ✓ ⇒ 这一格**调多少都不变** ✓✗。
    #: ⇒ 提醒的重点不是"平" ✗，而是「**以为调高了更强，其实一点没变**」✓✗（这才是会误导人的地方 ✓）。
    cfg = float(getattr(request.guidance, "scale", 1.0) or 1.0)
    plan.warnings.append(
        f"引导 {cfg:g} ✓ **对出歌没有作用** ✓✗ —— 本族的两层引导要分清："
        f"① 真正的引导在**自回归内部** ✓（固定 ``cfg {te_music3_mod.CFG_SCALE:g}`` / "
        f"top-``{te_music3_mod.CFG_TOP_K}`` ✓，不受这一格影响 ✗）；"
        f"② DiT 那层的 CFG 要**负向条件** ✓，而本仓 ``negative`` 恒空 ✗"
        f"（要它就得再跑一遍整段自回归 ✓，几分钟量级 ✓）⇒ ``cfg`` 只被记在方案里 ✓，"
        f"调高**不会**更「听话」 ✓✗")

    #: 步数**照抄请求** ✓（``GenerationRequest.steps`` 默认 30 ✓）—— 官方**没有内置步数** ✓：
    #: 那个数在 ``KSampler`` 节点上（`nodes_ksampler.py` 的默认 20 ✓ 由用户改 ✓），
    #: 所以「官方步数」这个说法本身不成立 ✓（编一个数＝把选择权说成标定值 ✓✗）。
    plan.steps = int(request.steps)
    #: ⭐ σ **必须走 ``normal``（σ 等距 ＋ 端点 ``(1.0, 0.001)`` ＋ 末尾 0）** ✓✗ —— 这一格是本轮
    #: 对参考实现时抓到的**真差异** ✓：
    #:
    #: * 官方 ``KSampler`` 的默认 ``scheduler`` 就是 ``normal`` ✓，落到 flow 族（``multiplier=1``）＝
    #:   ``linspace(timestep(σ_max)=1.0, timestep(σ_min)=0.001, steps) + [0.0]`` ✓（见
    #:   :func:`schedules.normal_sigmas` 的逐字转写 ✓）；
    #: * 旧写法走 :func:`flow_sigmas_for_steps` ✓（**t** 等分、端点 ``(1.0, 0.0)`` ✓）⇒ 30 步时
    #:   末段是 ``0.0333 ⇒ 0`` ✓✗，而参考实现是 ``0.001 ⇒ 0`` ✓ —— 最后一步的噪声跳幅差 **33 倍** ✓，
    #:   症状是「尾巴那点残噪没清掉」✓（不报错、只是与参考差一截 ✓✗）。
    plan.sigmas = schedules.normal_sigmas(plan.steps, shift=MUSIC3_FLOW_SHIFT)
    plan.timesteps = [float(value) for value in plan.sigmas]
    plan.warnings.append(
        f"σ 口径 ✓ 有出处：参考实现 ``MiniMaxMusic3`` 是 ``ModelType.FLOW`` ⇒ "
        f"``ModelSamplingDiscreteFlow`` 取 ``shift={MUSIC3_FLOW_SHIFT:g}``（``sampling_settings`` 没写 "
        f"shift ✓）＋ ``multiplier=1.0`` ⇒ ``time_snr_shift(1.0, t) = t`` **恒等** ✓；"
        f"步数由 KSampler 节点给（官方无内置值 ✓），``normal`` ⇒ σ 等距 "
        f"{plan.sigmas[0]:.4g} ⇒ {plan.sigmas[-2]:.4g} ⇒ 0 ✓")


def _fill_wan22_plan(request: GenerationRequest, plan: GenerationPlan) -> None:
    """**Wan 2.2 I2V** 的 plan ✓（首帧 ⇒ 视频 ✓；flow shift=**5.0** ✓、16fps ✓、``4n+1`` 帧 ✓）。

    与其他各族的差别（这一族是**唯一**吃「首帧条件」的生成族 ✓）：

    | | H3 ✓ | SEEDVR2 ✓ | 图族五套 ✓ | **Wan22 I2V** ✓ |
    |---|---|---|---|---|
    | 输入 | 文本 | **源视频** ✓ | 文本/图 | **首帧图 ＋ 文本** ✓ |
    | 主干入口 | 16 | 33 | 4/16/64 | **36** ✓（16＋4＋16 ✓） |
    | 文本塔 | Qwen3-VL | **无** ✗ | Qwen3/3.5/2.5 | **UMT5-XXL** ✓ |
    | σ | flow 12 | flow 1.0 | flow 3/0.69/3.1/μ1.15 | **flow 5.0** ✓ |
    | 帧数 | ``4k+1`` | 源视频``4n+1`` | — | ``4k+1`` ✓（81 官方 ✓） |

    ⚠️ 三条硬口径：

    1. **必须有首帧** ✗（``firstFrame`` ✓）：I2V 权重入口 36 通道 ✓，没有条件图就拼不出 ✓✗
       ⇒ 当场拒 ✓（**不**退化成文生视频 ✗ —— 那会出一段与图无关的片子 ✓且不报错 ✓✗）；
    2. **步数与切换点** ✓：官方模板两个 ``KSamplerAdvanced`` 各 4 步 ✓、切换在 2/4 = **50%** ✓
       ⇒ ``plan.wan_high_steps`` 是**前多少步走"高噪专家"** ✓（见 `torch_backend._wan_expert_for` ✓）；
    3. **尺寸吸附 16** ✓（VAE 8× × patch 2× ✓）、**帧数吸附 ``4k+1``** ✓ —— 两处都不四舍五入 ✗
       （向上吸 ✓：少一帧会被 VAE 时间核当"残帧"吃掉 ✓，症状是时长比要的短且不报错 ✓✗）。
    """
    first = str(request.first_frame or "").strip()
    if not first:
        raise StageError(
            "plan",
            "Wan 2.2 I2V 的请求**必须**带 `firstFrame`（首帧图 ✓）✗✗ —— "
            "本阶段装的是 **I2V** 权重 ✓（主干入口 36 = 噪声 16 ＋ 掩码 4 ＋ 条件图 16 ✓），"
            "没有条件图就拼不出这 36 通道 ✓。\n"
            "  · 要**纯文本**出视频请用 `stage=\"h3\"` ✓（那是 16 通道直入的 T2V ✓）；\n"
            "  · **不**退化成文生视频 ✗ —— 那会出一段与首帧无关的片子 ✓、且一条报错都没有 ✓✗。")
    #: 尺寸：**先吸到 16 网格** ✓（VAE 8× ✓ × 主干 patch 2× ✓）——
    #: ⚠️ 吸附**向上**（``ceil`` ✓）：向下会掉到官方 640 以下 ✓✗；且必须保持 16 的倍数 ✓，
    #: 否则 ``_PatchIn`` 当场抛「不能整除 patch」✓（指不到"吸附写错了" ✓）。
    override = _canvas_override(request)
    if override is not None:
        width, height = int(override.width), int(override.height)
        plan.canvas = override.to_dict()
        plan.warnings.extend(override.notes)
    else:
        width, height = geometry.size_for_megapixels(
            max(0.05, float(request.megapixels)), request.ratio)
    plan.width = int(-(-width // WAN22_DIM_GRID) * WAN22_DIM_GRID)
    plan.height = int(-(-height // WAN22_DIM_GRID) * WAN22_DIM_GRID)
    #: 帧率：官方 **16** ✓。``request.fps`` 的默认值是别的族的 **24** ✗ ⇒ 与步数同一套口径：
    #: 取到默认值就换成 16 并**说出来** ✓。
    #: ⚠️ 诚实说清一处**取不到的信息** ✗✗：``fps: int = 24`` 是**值**不是"没填" ✓ ⇒
    #: 「没填」与「显式填了 24」在这一层**分不出来** ✓✗（不像 ``canvas`` 有 ``None`` 哨兵 ✓）。
    #: 所以这一档现在**给不了 24fps** ✓（要走 24 得先把 ``fps`` 改成 ``int | None`` ✓ 那是 API 改动 ✓）。
    #: ⚠️ 别把这条写成"显式传 24 会保留"✗✗ —— 那是假的 ✓（值为 24 一律换 ✓），
    #: 而假说明比没说明更糟 ✓：用户会照它传 24 然后拿到 16fps ✓、还以为自己控制住了 ✓✗。
    plan.fps = int(request.fps or WAN22_FPS)
    if plan.fps == geometry.H3_FPS:
        plan.fps = WAN22_FPS
        plan.warnings.append(
            f"帧率收到 **{geometry.H3_FPS}fps** ✓（恰是 ``GenerationRequest`` 的默认值 ✗、H3 的口径 ✗）"
            f"⇒ 已按官方 Wan2.2 模板换成 **{WAN22_FPS}fps** ✓。⚠️ **这一档现在给不了 {geometry.H3_FPS}fps** ✓✗："
            f"``fps`` 是**值**不是「没填」✓ ⇒ 分不出「没写」与「写了 {geometry.H3_FPS}」✓"
            "（要支持得把字段改成可选 ✓，属 API 改动 ✓）")
    elif plan.fps != WAN22_FPS:
        plan.warnings.append(
            f"帧率 {plan.fps} ≠ 官方 **{WAN22_FPS}fps** ✓ —— 能跑 ✓（只是出片节奏与官方不同 ✓）")
    #: 帧数：官方 **81** ✓（``4k+1`` ✓、16fps ⇒ 5.06 秒 ✓）。按 ``seconds`` 算再**向上**吸 ✓
    #: （向下吸会让出片比要的短 ✓✗；且 ``4k+1`` 缺一帧就丢掉最后 4 帧的残差 ✓）。
    want_frames = max(WAN22_MIN_FRAMES, int(round(float(request.seconds) * plan.fps)) + 1)
    plan.frames = int(-(-(want_frames - 1) // WAN22_FRAME_GRID) * WAN22_FRAME_GRID + 1)
    plan.latent_frames = geometry.latent_frames(plan.frames,
                                               temporal_compression=WAN22_FRAME_GRID)
    #: 步数：官方 4 ✓（**两个专家各 4 步、切换在 50%** ✓）。默认 30 是别的族的 ✓ ⇒ 换掉并说明 ✓。
    default_steps = 30
    plan.steps = int(request.steps)
    if plan.steps == default_steps:
        plan.steps = WAN22_OFFICIAL_STEPS
        plan.warnings.append(
            f"步数仍是 ``GenerationRequest`` 的默认 **{default_steps} 步** ✓（别的族的口径 ✗）⇒ "
            f"已按官方 Wan2.2 I2V 模板换成 **{WAN22_OFFICIAL_STEPS} 步** ✓"
            f"（两个专家各 4 步、切换在 {WAN22_SWITCH_RATIO:g} ✓）—— 要更多步请显式传 `steps` ✓")
    elif plan.steps != WAN22_OFFICIAL_STEPS:
        plan.warnings.append(
            f"步数 {plan.steps} ≠ 官方 **{WAN22_OFFICIAL_STEPS} 步** ✓ —— 能跑 ✓，"
            "只是 σ 梯子与官方模板不同 ✓（这里只报不改 ✗）")
    high_steps = int(round(plan.steps * WAN22_SWITCH_RATIO))
    plan.wan_high_steps = max(1, min(plan.steps - 1, high_steps)) if plan.steps > 1 else 1
    plan.sigmas = _image_flow_sigmas(request, plan, shift=WAN22_FLOW_SHIFT)
    plan.timesteps = [float(value) for value in plan.sigmas]
    plan.warnings.append(
        f"**{plan.width}×{plan.height}** ✓、{plan.frames} 帧 @{plan.fps}fps ≈ "
        f"{plan.frames / max(1, plan.fps):.2f}s ✓（吸附到 {WAN22_FRAME_GRID}k+1 ✓）⇒ 潜变量 "
        f"{plan.latent_frames} 帧 × {plan.height // 8}×{plan.width // 8} ✓ ×16 通道 ✓；"
        f"主干另吃 **20 通道**条件 ✓（掩码 4 ＋ 条件图 16 ✓）⇒ 入口 36 ✓；"
        f"{plan.steps} 步 **flow matching** ✓（shift={WAN22_FLOW_SHIFT:g} ✓ **不是** H3 的 12 / "
        f"2.1 的 0.69 / 2512 的 3.1 / Krea2 的 μ1.15 / SEEDVR2 的 1.0 ✓✗；"
        f"σ {plan.sigmas[0]:.4g} ⇒ 0 ✓；timestep 即 σ ✓ 不乘 1000 ✗）")
    if plan.steps > 1:
        plan.warnings.append(
            f"**两个专家接力** ✓：前 **{plan.wan_high_steps}** 步走「高噪」权重 ✓、"
            f"后 **{plan.steps - plan.wan_high_steps}** 步换成「低噪」权重 ✓"
            f"（官方模板两个 ``KSamplerAdvanced`` 各 4 步、切换在 {WAN22_SWITCH_RATIO:g} ✓）"
            "—— 只给一份权重 ⇒ **不接力** ✓，后段质量会明显掉 ✓（这里只报不改 ✗）")
    #: ⚠️ ``initImage`` 这里**没有**警告 ✓✗ —— 不是漏 ✓：`build_plan` 在上游就**当场拒**它 ✓
    #: （两格语义互斥 ✓，见 ``elif init_image`` 那段 ✓），走到这一步说明它根本没给 ✓。
    #: 曾经写过一条「被忽略」的警告 ✓，那是**死代码** ✓✗：它永远进不来 ✓，
    #: 而且与上游的「拒绝」互相矛盾 ✓（读代码的人会以为能传 ✓）。
    if request.last_frame:
        plan.warnings.append(
            "`lastFrame` 被**忽略** ✓ —— Wan 2.2 I2V 只有**首帧**条件 ✗（尾帧要 FLF2V 那套权重 ✓，"
            "本阶段不收 ✓；给也不会报错 ✓，只是不生效 ✓）")
    #: ⚠️ ``referenceFrames`` 的警告**不在这里** ✓ —— 统一收进 `_finish_plan` 的
    #: :func:`_warn_unused_references` ✓（每条线同一句口径 ✓；各写一遍必然有的写有的忘 ✗✗，
    #: 忘了那条就是「静默忽略参考图」✓✗）。


#: 真吃 ``referenceFrames`` 的档 ✓（**只有 H3 双流** ✗✗：它走 ``ref2va``、一张一个
#: ``<|image_pad|>`` 槽 ✓）。其余各族拿到参考图**一个字节都不用** ✓✗。
_REFERENCE_FRAME_STAGES = (VIDEO_STAGE,)


def _warn_unused_references(request: GenerationRequest, plan: GenerationPlan) -> None:
    """参考图被**静默忽略**时必须说一句 ✓✗ —— 这是画布上最容易踩的一格。

    ⚠️ 为什么必须在这里统一 ✗✗：画布上最自然的动作就是「连一张图 → 点生成」✓，
    而图片四档（zimage / krea2 / qwen_image / qwen_image21）与 wan22 都**不读**
    ``reference_frames`` ✓✗ ⇒ 用户以为在按图改 ✓、其实出的是一张与它无关的新图 ✓✗，
    而且**一条日志都不会有** ✓✗。原先只有 wan22 那条线写了这句 ✓（其余各族漏 ✗），
    每条线各抄一遍必然继续漏 ⇒ 收到收尾处一处判 ✓。
    """
    frames = tuple(request.reference_frames or ())
    if not frames:
        return
    if str(request.stage or "").strip().lower() in _REFERENCE_FRAME_STAGES:
        return
    plan.warnings.append(
        f"`referenceFrames`（{len(frames)} 路）被**忽略** ✓ —— "
        f"「{request.stage or '本档'}」不吃参考图 ✗（只有 H3 双流真用 ✓：它走 ref2va、"
        "一张一个图像槽 ✓）。要按图改请走 ComfyUI 的 Edit/参考图工作流 ✓，"
        "或改用 `initImage` 那一格（起点图 ✓，denoise<1 才生效 ✓）")


def _finish_plan(request: GenerationRequest, plan: GenerationPlan,
                 weights_bytes: int | None) -> GenerationPlan:
    """两条路**共用**的收尾 ✓（相对耗时系数 ✓ / 超清计划 ✓ / 显存估算 ✓）。

    ⚠️ 刻意**不含尺寸与 σ** ✗ —— 那两样两条路口径不同 ✓，各自在前面算好了 ✓。
    """
    _warn_unused_references(request, plan)
    # ── 图生图那四格（``denoise=1`` ⇒ 1.0 / steps / 1.0 / "noise" ✓ 与旧回执只多不少 ✓）──
    # ⚠️ ``start_sigma`` 读的是**切完的**梯子首值 ✓（`_image_flow_sigmas` / SDXL 那条已经切过 ✓），
    #    不是 `σ_max` ✗ —— 报 σ_max 就等于说"从 1.0 起跑"，而那正是 denoise 要躲开的那一格 ✓✗。
    plan.denoise = float(plan.denoise or 1.0)
    plan.total_steps = int(plan.total_steps or plan.steps)
    plan.start_sigma = float(plan.sigmas[0]) if plan.sigmas else 0.0
    plan.init_from = "image" if plan.denoise < 1.0 else "noise"
    if plan.denoise < 1.0:
        plan.warnings.append(
            f"图生图 denoise={plan.denoise:g} ✓：σ 梯子按名义 {plan.total_steps} 步算再取尾巴 ⇒ "
            f"**实跑 {plan.steps} 步** ✓、起点 σ=**{plan.start_sigma:.4g}** ✓（不是 1.0 ✓）；"
            "起点 latent = ``initImage`` 的 VAE 编码 ✓ 按 ``σ0·noise + (1−σ0)·latent`` 混入 ✓"
            "（参考实现 ``noise_scaling`` ✓）—— 少这一半就只剩「低对比、与参考图无关」 ✓✗")
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
        #: ⭐ **可选钩子 `bind_plan`** ✓（2026-11-01 接 SEEDVR2 时加 ✓）——
        #: ``encode_text(request)`` 的契约里**没有 plan** ✓，出图族确实不需要 ✓（提示词与画布无关 ✓✗）；
        #: **修复族**不成立 ✗✗：它的"条件"就是**源视频编码出的 16 通道** ✓，而"编多少帧 / 编成多大"
        #: 正是 plan 说了算 ✓ ⇒ 两边必须看**同一份**（否则症状：plan 说 96 帧、实际编了 121 帧 ✓，
        #: 采样第 1 步形状对不上 ✓，报错指不到 plan ✓）。不实现该钩子的后端**零影响** ✓。
        bind_plan = getattr(backend, "bind_plan", None)
        if callable(bind_plan):
            bind_plan(plan)
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
        # ⚠️ **分时驻留的族**：`init_latents` 若要把参考图编成起点 ⇒ 这一拍必须有 VAE 在场 ✓✗
        #   （2512 文生图的 `phaseParts` 是 encode→text ✓ / sample→dit ✓ / decode→vae ✓ ⇒
        #    init 这一拍手上只有 TE ✓、VAE 要到 decode 才装 ✓ ⇒ 图生图必报
        #    「没挂 VAE ✗ ⇒ 没法把图片编码成潜变量」✓ —— 2026-11-02 真跑实测 ✓）。
        #   只在**真要编码图片**时申请 ✓：无条件申请会让纯文生图每次多一趟 VAE 换装 ✓✗。
        if request.init_image:
            _backend_phase(backend, "init")
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
    #: ⭐ **修复族（SEEDVR2）的条件在 encode 期就编好了** ✓（源视频 → 17 通道 conditioning ✓）。
    #: 它不是"首帧/尾帧"那类**加在潜变量上的掩码** ✓ ⇒ 下面三条 keyframe 分支**必须绕开** ✓✗
    #: （否则一次合法的修复请求会因为"单流不支持尾帧"被拒 ✓✗ —— 那是把另一个族的规矩套上来 ✓）。
    repair_condition = (condition.get("repair") if isinstance(condition, dict) else None)
    if isinstance(repair_condition, dict):
        result.conditioning = {
            "applied": True,
            "mode": "video-repair",
            "sourceVideo": repair_condition.get("source"),
            "via": "encode_text（源视频 → 17 通道 conditioning ✓：退化 16 ＋ mask 1 ✓）",
            "backendNote": repair_condition,
        }
        emit(result.event(
            "condition",
            note=f"修复条件 ✓（源 {repair_condition.get('sourceFrames')} 帧 → "
                 f"条件 {repair_condition.get('conditioningShape')} ✓）"))
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

    if request.last_frame and not dual and not isinstance(repair_condition, dict):
        # 单流的 `condition_first_frame` 只会把掩码混到**第 0 潜帧** ✓ ⇒ 做不了尾帧 ✗。
        # 硬做（比如硬塞成首帧）会让产物**开头**是尾帧 ✓✗ —— 比报错难查得多 ✓。
        return _fail(result, "condition", conditioning_mod.ConditioningError(
            f"后端 {getattr(backend, 'name', type(backend).__name__)} 未实现尾帧条件 ✗"
            f"（单流只有 conditioning_first_frame ⇒ 掩码只能混第 0 潜帧 ✓）"
            f"；刻意报错，不静默忽略尾帧 ✗"))

    if request.first_frame and not dual and not isinstance(repair_condition, dict):
        started = time.perf_counter()
        try:
            ensure_not_cancelled("condition")
            #: ⚠️ Wan2.2 的首帧条件**要 VAE** ✓✗（首帧图得先过 VAE 编成 **20 通道**条件 ✓，
            #: 见 `torch_backend._wan_i2v_condition` ✓）。而这一拍在管线里**没有自己的阶段名** ✓
            #: （紧跟 init ✓、在 sample 之前 ✓）⇒ 当前在场的件取决于**上一拍是谁**✓：
            #: wan22 的分时链是 ``encode=(text,vae)`` ✓，init 那拍又**不换装** ✓ ⇒ 此刻 VAE 恰好还在 ✓。
            #: **但「恰好」不算保证** ✗✗：将来若 init 也给 wan22 换了装 ✓、或者把首帧条件挪个位置 ✓，
            #: VAE 就静默不在场 ✓ ⇒ 报错长成「没挂 VAE ✗ ⇒ 没法把图片编码成潜变量」✓，
            #: 指不到「是换装换早了」✓✗。⇒ 这里**显式**要一次 encode 那一拍 ✓
            #: （两件都在场 ⇒ 一件都不装 ✓ 零代价 ✓；不在场 ⇒ 就地装上 ✓）。
            if str(request.stage or "").strip().lower() == WAN22_STAGE:
                _backend_phase(backend, "encode")
            hook = getattr(backend, "condition_first_frame", None)
            if not callable(hook):
                raise conditioning_mod.ConditioningError(
                    f"后端 {getattr(backend, 'name', type(backend).__name__)} 未实现 "
                    f"condition_first_frame ✗ ⇒ 无法做图生视频（刻意报错，不退回文生视频 ✗）")
            #: ⚠️ 潜帧数的**第一来源是 plan** ✓✗ —— 各族的 ``_fill_*_plan`` 早就按**自己那张网格**
            #: 算好了 ✓（wan22 把像素帧吸成 ``4k+1`` ⇒ ``plan.latent_frames`` 就是 4 倍压缩的结果 ✓）。
            #: 只有 plan 上真没有时才退回「按请求里的压缩比现算」✓。顺序反过来的症状（2026-11-02
            #: 真跑撞的 ✓）：Wan 2.2 的请求不带 ``temporalCompression`` ✓（它是族常量、不该逼调用方
            #: 抄一遍 ✓）⇒ 当场报「首帧条件需要 temporalCompression ✗」，读起来像**用户少填了字段**
            #: ✓✗，其实是**族自己知道网格却没被问** ✓。
            planned = getattr(plan, "latent_frames", None)
            latent_count = (int(planned) if planned
                            else conditioning_mod.latent_frames_for(plan.frames,
                                                                    request.temporal_compression))
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
            # ⚠️ 同理：双流能跑的采样器是**白名单** ✓（`h3_form.DUAL_SAMPLERS` ✓ 唯一真源 ✓）——
            #    收到 heun/dpmpp 还照跑就是**静默换算法** ✗✗（名字被无视 ✓ 而用户看不出来 ✓）。
            #    2026-11-01 放开到 `euler` + `res_multistep` ✓（后者是 H3 官方模板的默认档 ✓，
            #    每步一次模型调用 ✓ ⇒ 双流能直接套 ✓）；不在名单里仍**响亮报错** ✗。
            from . import h3_form as h3_form_mod  # noqa: PLC0415 —— 零 torch 模块 ✓（顶层不 import 只为 plan 期干净 ✓）
            dual_samplers = tuple(h3_form_mod.DUAL_SAMPLERS)
            if str(request.sampler or "euler").strip().lower() not in dual_samplers:
                raise StageError(
                    "sample", f"双流（H3 形态）只支持 {list(dual_samplers)} ✗ —— 双流每步只有"
                              "**一次**共享前向 ✓ ⇒ 每步两次调用的采样器（heun/dpmpp ✗）套不上 ✓"
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
