"""自检：**Krea2 的出图口径与装配门禁**（2026-11-01 ✓）。

为什么单独一份 ✗（`engine_qwen21_test` 已经钉过「σ 要走 flow」✓、`engine_zimage_test` 钉过
「shift 是底数」✓）：Krea2 在**同一个形状上**犯的是**第三种错** ✗✗ ——

==========================  ==============================  ==============================
                           Qwen-Image 2.1                  **Krea2**
==========================  ==============================  ==============================
shift 写法                 ``0.69`` = **底数** ✓            **``1.15`` = μ** ✓✗（要过 ``exp`` ✓）
官方步 / 引导              30 / 2.5~4.0 ✓                  **8 / 1.0** ✓（turbo 蒸馏 ✓）
潜变量                     64 通道 / 16× ✓                 **16 通道 / 8×** ✓
反缩放族                   ``qwen_image21`` ✓              **``wan21``** ✓✗（VAE 文件却是 2512 那份 ✓）
文本条件                   末层 4096 ✓                     **12 层隐状态栈 30720** ✓
==========================  ==============================  ==============================

⇒ 两处「形状都对、图不对、也不报错」的换法：
① 把 **μ 当底数** 喂下去 ✓✗ —— σ 曲线几乎不弯 ✓，前几步没把噪声压下去 ✓，出图是一团纹理 ✓
   （本轮真栽在这上面：提示词几乎不跟 ✓、换提示词输出相关 0.9998 ✓）；
② 反缩放取 **2512** 那套 mean/std ✓✗ —— 解码照跑 ✓、形状全对 ✓，只是整体偏色发灰 ✓
   （两套数差一个数量级 ✓，且两条路互不报错 ✗）。

本文件钉**口径与门禁** ✓：μ ⇒ ``e^μ`` ✓、尺寸吸附 16 ✓、默认像素预算被如实换掉 ✓、
四格缺项点名 ✓、换 TE/VAE 必须换装载键 ✓、就绪档只列这一条链的三件 ✓。
⚠️ **不跑真装** ✗（三件 ≈17.9 GiB ✓）：真装与出图是人为验收那一步 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_krea2_test.py
"""
from __future__ import annotations

import ast
import math
import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import pipeline, schedules  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call) -> str | None:  # noqa: ANN001
    try:
        call()
    except Exception as error:  # noqa: BLE001
        return str(error)
    return None


def _error_of(call) -> str:  # noqa: ANN001
    text = _raises(call)
    return text if text is not None else ""


def _underlying_type(call) -> type | None:  # noqa: ANN001
    try:
        call()
    except Exception as error:  # noqa: BLE001
        return type(error)
    return None


def _plan(**kw):  # noqa: ANN001, ANN202
    base = dict(prompt="a cat", stage=pipeline.KREA2_STAGE)
    base.update(kw)
    return pipeline.build_plan(pipeline.GenerationRequest(**base))


# --------------------------------------------------------------- ① 阶段名


def case_stage_names() -> None:
    """① 五个「出一张图」的 stage 必须**互不相同** ✗✗（合流 ⇒ σ / 通道 / 后端三处一起串 ✓）。"""
    names = (pipeline.IMAGE_STAGE, pipeline.ZIMAGE_STAGE, pipeline.QWEN21_STAGE,
             pipeline.QWEN_IMAGE_STAGE, pipeline.KREA2_STAGE)
    check("① `image` / `zimage` / `qwen_image21` / `qwen_image` / `krea2` 五名互不相同 ✓✗"
          "（合一个 ⇒ 后端也只有一条 ✓）",
          len(set(names)) == 5, names)
    message = _error_of(lambda: pipeline.build_plan(
        pipeline.GenerationRequest(prompt="x", stage="nope")))
    check("①′ 未知 stage ⇒ 报错把**五个名字**都列出来 ✓（别让调用方猜 ✓）",
          all(name in message for name in names), message)


# --------------------------------------------------------------- ② σ 口径


def case_shift() -> None:
    """② **1.15 是 μ、不是底数** ✗✗（``ModelSamplingFlux`` 走 ``flux_time_shift`` ✓）。

    参考：``comfy/supported_models.py::Krea2`` 的 ``sampling_settings={"shift": 1.15}`` ✓ ⇒
    ``model_base.Krea2`` 是 ``ModelType.FLUX`` ✓ ⇒ ``ModelSamplingFlux`` ✓ ⇒
    ``sigma(t) = e^1.15 / (e^1.15 + 1/t − 1)`` ✓。

    ⚠️ 这条与 2512 / Z-Image 的差别**只在写法** ✓✗：那两族走 ``ModelSamplingAuraFlow`` ✓，
    那里的 ``shift`` 就是``shift·t / (1 + (shift−1)·t)`` 里的**底数本身** ✓ ⇒ 同一个数字
    ``1.15`` 在两族里代表**两条曲线** ✓✗ —— 它不是「图会崩」级的错 ✓，是「怎么调都不像」✓✗。
    """
    check("② `KREA2_FLOW_SHIFT` 记的是 **μ = 1.15** ✓（不是底数 ✗）",
          abs(pipeline.KREA2_FLOW_SHIFT - 1.15) < 1e-12, pipeline.KREA2_FLOW_SHIFT)
    base = pipeline.krea2_base_shift()
    check("②′ `krea2_base_shift()` = **e^1.15 ≈ 3.1582** ✓（漏了 `exp` ⇒ 曲线不弯 ✓✗）",
          abs(base - math.exp(1.15)) < 1e-12 and round(base, 4) == 3.1582, base)

    steps = 8
    sigmas = schedules.flow_sigmas_for_steps(steps, shift=base)
    check("②″ 长度 = 步数 + 1 ✓、σ₀ = 1 ✓、σ_last = 0 ✓、严格单调递减 ✓",
          len(sigmas) == steps + 1 and sigmas[0] == 1.0 and sigmas[-1] == 0.0
          and all(a > b for a, b in zip(sigmas[:-1], sigmas[1:])), sigmas)
    #: 手算：``σ(t) = shift·t / (1 + (shift − 1)·t)``，``t = 1 − 1/steps``，``shift = e^1.15`` ✓。
    t = 1 - 1 / steps
    want = base * t / (1 + (base - 1) * t)
    check(f"②‴ 第 2 个点 = {want:.6f} ✓（公式/常量写错必红 ✓）",
          abs(sigmas[1] - want) < 1e-12, (sigmas[1], want))

    #: ⚠️ 反验：**把 μ 当底数**喂下去不会报错 ✓，只是逐点更小 ✓（＝噪声压得更慢 ✓ ⇒ 出图糊 ✓）。
    wrong = schedules.flow_sigmas_for_steps(steps, shift=pipeline.KREA2_FLOW_SHIFT)
    check("②⁴ 把 1.15 当底数 ⇒ σ **逐点更小** ✓✗（这就是「出图一团纹理」的数值形状 ✓）",
          all(a < b for a, b in zip(wrong[1:-1], sigmas[1:-1])),
          list(zip(wrong[:4], sigmas[:4])))

    # ------------------------------------------------------------------ ②⁵ μ 随画布走
    # ⚠️ 2026-11-01 ✗：官方 ``shift`` 是**按模板画布 1024²** 定的 ✓（``Krea2`` 的
    # ``sampling_settings`` 只有这一个数 ✓，而模板画布也是 1024² ✓）。低分辨率（512²）
    # 沿用 1.15 时 σ 曲线**一点不弯**在 t>0.5 那段就压完了 ✓ ⇒ 出图只剩大结构 ✓✗；
    # comfy 的 ``ModelSamplingFlux`` 里 ``shift`` 与 ``max_shift`` 是**按 seq_len 插值**的 ✓
    # （``time_shift`` 的默认行为 ✓）⇒ 引擎按 ``seq=(w/16)·(h/16)`` 线性插 ✓。
    check("②⁵ μ 按分辨率插值：256² ⇒ **0.5**（下限 ✓）、1024² ⇒ **1.15**（上限 ✓）、"
          "512² ⇒ **0.63**（线性 ✓）",
          abs(pipeline.krea2_shift_mu(256, 256) - 0.5) < 1e-12
          and abs(pipeline.krea2_shift_mu(1024, 1024) - 1.15) < 1e-12
          and abs(pipeline.krea2_shift_mu(512, 512) - 0.63) < 1e-9,
          [round(pipeline.krea2_shift_mu(n, n), 4) for n in (256, 512, 768, 1024, 1536)])

    check("②⁵′ 分辨率越大 μ **越大**（不回头 ✓）、且两端**夹紧** ✓（外推会给出 >1.15 的 μ ✗）",
          pipeline.krea2_shift_mu(256, 256) < pipeline.krea2_shift_mu(512, 512)
          < pipeline.krea2_shift_mu(768, 768) < pipeline.krea2_shift_mu(2048, 2048)
          and abs(pipeline.krea2_shift_mu(2048, 2048) - 1.15) < 1e-12,
          [pipeline.krea2_shift_mu(2048, 2048), pipeline.krea2_shift_mu(3072, 3072)])

    low = schedules.flow_sigmas_for_steps(steps, shift=pipeline.krea2_base_shift(width=512, height=512))
    check("②⁵″ 512² 的 σ₀ 之后第一个点**比 1024² 的小** ✓（＝低分辨率那档曲线更缓 ✓）",
          low[1] < sigmas[1], (low[1], sigmas[1]))

    check("②⁵‴ 显式给 μ 时**不再**插值 ✓（`krea2_base_shift(mu=…)` 是逃生口 ✓）",
          abs(pipeline.krea2_base_shift(1.0, width=512, height=512) - math.e) < 1e-12,
          pipeline.krea2_base_shift(1.0, width=512, height=512))


# --------------------------------------------------------------- ③ plan


def case_plan() -> None:
    """③ 默认 0.65 MP ⇒ **1.0 MP 的官方画布** ✓、单帧 ✓、``timesteps`` 就是 σ ✓、cfg=1 是对的 ✓。

    ⚠️ 2026-11-01 改口径 ✗：默认预算 ``1.05 → 1.0`` MP ✓，且尺寸改走 ComfyUI
    ``ResolutionSelector`` ✓（``round`` + 1024² 基数 ✓）⇒ 1:1 就是 **1024×1024** ✓，
    与官方画布**逐像素相同** ✓。旧的 ``1.05 MP ＋ 向上吸 16`` 会落到 **1040×1040** ✓✗ ——
    多算 3% 像素、且回执里得专门解释"不是 1024"，读的人只会困惑 ✓。
    """
    plan = _plan(ratio="1:1")
    check("③ 默认像素预算（0.65 MP 的**视频口径** ✗）⇒ 换 Krea2 turbo 官方 **1.0 MP / 1024×1024** ✓ "
          "且**如实告警** ✓",
          (plan.width, plan.height) == (1024, 1024)
          and any("0.65" in text and "1.00 MP" in text for text in plan.warnings),
          ((plan.width, plan.height), plan.warnings[:2]))
    #: ⚠️ 官方画布是 **1024×1024** ✓ ⇒ 新口径**正好**落上去 ✓；回执必须与像素数一致 ✓✗。
    check("③′ 1:1 @ 1.0 MP ⇒ **1024×1024** ✓（`ResolutionSelector` 口径 ✓；"
          "旧的 1040 那套已废 ✗）",
          (plan.width, plan.height) == (1024, 1024)
          and any("ResolutionSelector" in text for text in plan.warnings), plan.warnings[:2])
    check("③″ 单张图 ⇒ `frames=1` ✓、`fps=0` ✓（不是视频 ✓）",
          plan.frames == 1 and plan.fps == 0, (plan.frames, plan.fps))

    #: ⚠️ 反向：**16 的倍数、但不必是 32 的倍数** ✗✗ —— Krea2 是 ``VAE 8× × patch 2`` ✓，
    #: 多吸到 32 只是白裁边 ✓、不报错 ✓。16:9 @ 1.0 MP ⇒ 1360×768 ✓（1360 = 85×16 ✓、不是 32 的倍数 ✓）。
    #: （3:2 那档恰好 1248/832 都是 32 的倍数 ⇒ 判不出"有没有被多吸" ✗，别用它 ✓。）
    odd = _plan(ratio="16:9")
    check("③‴ 宽高都吸附到 **16** 的倍数 ✓、且**不**被额外吸到 32 ✓✗",
          odd.width % 16 == 0 and odd.height % 16 == 0
          and (odd.width % 32 != 0 or odd.height % 32 != 0), (odd.width, odd.height))

    steps = _plan(steps=8, ratio="1:1")
    check("③⁴ `timesteps` **就是 σ** ✓✗（``multiplier=1.0`` ✓ ⇒ 不乘 1000 ✗、也不翻成 0..999 ✗）",
          [round(float(v), 9) for v in steps.timesteps]
          == [round(float(v), 9) for v in steps.sigmas], steps.timesteps[:3])
    check("③⁵ 官方 8 步 ⇒ **不报**「步数不对」✓（蒸馏档就是 8 步 ✓）",
          not any("8 步" in text and "≠" in text for text in steps.warnings),
          steps.warnings[-2:])

    more = _plan(steps=30, ratio="1:1")
    check("③⁶ 30 步 ⇒ **只报不改** ✓（``image_krea2_turbo_t2i`` 的 KSampler 是 8 步 ✓；"
          "改用户的配置越界 ✗）",
          more.steps == 30 and any("8 步" in text for text in more.warnings), more.warnings[-1:])
    #: ⚠️ 与 2.1 / 2512 **相反** ✓✗：这两族 cfg=1 会警告「等效不引导」✓，而 Krea2 这一档
    #: 本身就是 turbo 蒸馏（模板 cfg 1.0 ✓）⇒ 这里**不该**报那句 ✓。
    check("③⁷ cfg=1.0 在 Krea2 上是**对**的 ✓✗（别照抄 2.1 那句「不引导会糊」✗）",
          not any("不引导" in text or "cfg=1.0" in text for text in more.warnings),
          more.warnings)


# --------------------------------------------------------------- ④ 装配门禁


def case_runtime_refuses() -> None:
    """④ 四格缺项**当场拒** ✓、只点名真缺的 ✓、换组件必须换装载键 ✓。

    ⚠️ 这条**不跑真装载** ✗（≈17.9 GiB ✓）：验的是门口 ✓。
    """
    from app.services.engine import runtime as rt  # noqa: PLC0415

    call = lambda **kw: rt.EngineRuntime()._ensure_loaded_internal(  # noqa: E731
        stage=pipeline.KREA2_STAGE, **kw)

    message = _raises(call)
    check("④ `stage=krea2` 什么都不给 ⇒ **当场报** ✓✗（不静默当 2512 / 2.1 装 ✓、不半装 ✓）",
          _underlying_type(call) is rt.TorchBackendUnavailable, message)
    check("④′ 拒的理由把**四格**都点名 ✓"
          "（`ditPath` / `textWeights` / `imageVaeWeights` / `tokenizerPath`）",
          all(key in (message or "") for key in
              ("ditPath", "textWeights", "imageVaeWeights", "tokenizerPath")), message)
    check("④″ 报错里点了**本机真文件长什么样** ✓（12.82B 参数 / 16 通道 / 反缩放取 **Wan21** ✓"
          "—— 照着报错就能补齐 ✓）",
          all(text in (message or "") for text in ("12.82B", "16 通道", "Wan21")), message)

    partial = _raises(lambda: call(dit_path="d.safetensors", text_weights="t.safetensors",
                                   tokenizer_path="tok/"))
    #: ⚠️ 只看**首行** ✗✗：第二行起是 `_KREA2_STAGE_NOTE` 的四格说明 ✓（有意列的 ✓）。
    head = (partial or "").split("\n", 1)[0]
    check("④‴ 给了三格、只缺 `imageVaeWeights` ⇒ **缺项清单里只有它** ✓（说明段列四格不算 ✗）",
          "imageVaeWeights" in head and "textWeights" not in head, head)

    base = dict(stage=pipeline.KREA2_STAGE, dit_path="d.safetensors", tokenizer_path="tok/")
    k_a = rt._load_key(**base, text_weights="te_a.safetensors", image_vae_weights="vae.safetensors")
    k_te = rt._load_key(**base, text_weights="te_b.safetensors", image_vae_weights="vae.safetensors")
    k_vae = rt._load_key(**base, text_weights="te_a.safetensors", image_vae_weights="vae_b.safetensors")
    check("④⁴ 换 `textWeights` ⇒ 装载键变 ✓（不复用上一套 Qwen3-VL ✓）", k_a != k_te, (k_a, k_te))
    check("④⁵ 换 `imageVaeWeights` ⇒ 装载键变 ✓（VAE 文件与 2512 同名也得看路径 ✓）",
          k_a != k_vae, (k_a, k_vae))


# --------------------------------------------------------------- ⑤ 就绪与登记


def case_stage_registered() -> None:
    """⑤ 就绪档**只列这条链的三件** ✗、三件全算必需 ✓、与 `plan_stage` 同一份条目 ✓。

    ⚠️ **不许断言 `ready`** ✗✗：那是"本机三份权重在不在"，换机必红 ✓。
    """
    from app.services.engine import inventory as inv  # noqa: PLC0415
    from app.services.engine import loader, registry  # noqa: PLC0415

    check("⑤ `STAGE_FILTERS` 与 `registry.STAGES` 双向对齐 ✓（少一边就别处会红 ✓）",
          pipeline.KREA2_STAGE in inv.STAGE_FILTERS and pipeline.KREA2_STAGE in registry.STAGES
          and inv.STAGE_FILTERS[pipeline.KREA2_STAGE]["category"]
          == registry.STAGES[pipeline.KREA2_STAGE].category,
          {"filter": inv.STAGE_FILTERS.get(pipeline.KREA2_STAGE),
           "registry": getattr(registry.STAGES.get(pipeline.KREA2_STAGE), "category", None)})

    report = inv.readiness(pipeline.KREA2_STAGE)
    keys = sorted(row["key"] for row in report["components"])
    check("⑤′ `readiness(\"krea2\")` **只列这条链的三件** ✓✗"
          "（把 SDXL / Z-Image / 2512 的组件混进来 ⇒ 24 GB 卡会凭空报装不下 ✓✗）",
          keys == ["dit_krea2_turbo", "te_krea2_qwen3vl4b", "vae_krea2"], keys)
    check("⑤″ 三件在这个档里**全部算必需** ✓（`requireAll` 生效 ✓；`required: false` 不是「不需要」✗）",
          all(row["required"] for row in report["components"]),
          [(row["key"], row["required"]) for row in report["components"]])
    #: 12.57 + 5.08 + 0.24 = 17.89 GiB ✓ —— 三件**同驻**才 17.9 ✓（22.49 的卡挤得进 ✓），
    #: 但仍按阶段换装 ✓：1024² 采样还要算激活 ✓、余量留给工作区 ✓。
    check("⑤‴ 必需权重合计 ≈ **17.88 GiB** ✓（只算这三件 ✓）",
          abs(float(report["requiredWeightsGiB"]) - 17.88) < 0.05,
          report["requiredWeightsGiB"])

    plan = loader.plan_stage(pipeline.KREA2_STAGE)
    plan_keys = sorted(row["key"] for row in plan["components"])
    check("⑤⁴ `loader.plan_stage` 与就绪档**同一份条目** ✓✗",
          plan_keys == keys, {"plan": plan_keys, "readiness": keys})
    required_keys = sorted(row["key"] for row in plan["components"] if row.get("required"))
    check("⑤⁵ `plan_stage` 的必需集合**非空** ✓✗（空集合 ⇒ `ready = all([]) = True`，文件全缺也说就绪 ✓）",
          required_keys == keys, required_keys)

    #: ⚠️ 反向：`image` 档（SDXL）**与 Krea2 互斥** ✓ ⇒ 不许把这三件算进它的必需集合 ✗✗
    #: （加一起会在 24 GB 卡上凭空报「装不下」✓✗）。
    image_report = inv.readiness("image")
    check("⑤⁶ `readiness(\"image\")` 的必需集合**不含** Krea2 三件 ✓✗（互斥备选不许相加 ✓）",
          not any(row["key"] in keys and row["required"]
                  for row in image_report["components"]),
          [(row["key"], row["required"]) for row in image_report["components"]])


# --------------------------------------------------- ⑥ 分时驻留（三件走同一条机制 ✓）


def case_phased() -> None:
    """⑥ Krea2 走**分时驻留** ✓：装 DiT（12.57 GiB）＋ `declare_chain` 登记三件 ✓。

    ⚠️ 这条**不装真权重** ✗：验的是「登记了什么、三件是否互斥」✓ —— 真装与出图是人为验收 ✓。
    """
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    parts = tb.TorchBackend.PHASE_PARTS
    #: ⚠️ 2026-11-02：不再断言「一件**只许**出现在一个阶段」✓✗ —— 管线多了可选的 `init` 一拍
    #: （`"init": ("text", "vae")` ✓），一件出现在两拍是**合法**的 ✓。这里只判「三件都登记了」✓；
    #: 真正要守的「同一拍装得下」由 2.1 那套按 GiB 的判据管 ✓（本族 dit 12.57 ＋ text 7.5 ＝ 20.07
    #: < 22.49 ⇒ 就算同拍也装得下 ✓，与 2.1 的 28.05 不同 ✗）。
    check("⑥ `PHASE_PARTS` 里 dit / text / vae 三件都登记了 ✓",
          sorted({part for group in parts.values() for part in group}) == ["dit", "text", "vae"],
          parts)
    check("⑥′ Krea2 有一条**自己的**驻留账 ✓✗（DiT 12.57 / TE 7.5 / VAE 0.24 ✓ ——"
          "拿 2.1 的 13.26 / 14.79 判会误判 ✓）",
          "dit" in tb.TorchBackend.KREA2_PART_NEED_GIB
          and abs(float(tb.TorchBackend.KREA2_PART_NEED_GIB["dit"]) - 12.57) < 0.05,
          dict(getattr(tb.TorchBackend, "KREA2_PART_NEED_GIB", {})))

    #: ⚠️ 反向：换装钩子必须**真挂进管线** ✗✗ —— 机制写好了但 `run_sync` 从不调用，
    #: 「装不下」照样 OOM ✓✗。三个族共用同一条钩子 ✓（`enter_phase` 分派 ✓）。
    #: ⚠️ 判据走 **AST** ✗✗（旧版是 `inspect.getsource` ＋ 搜字面量 `_backend_phase(backend, "encode")`）：
    #: 那种写法钉的是**源码排版** ✓ —— 把 `backend` 换个形参名、换单引号、或拆成两行写 ✓，
    #: 换装一次都没少 ✓、判据却当场红 ✓✗（提示的是"少了一处换装"✗，读起来像真缺陷 ✓）。
    #: 判据该盯的是**没少、顺序对** ✓：AST 里取调用点的**常量实参** ＋ `lineno` ✓。
    tree = ast.parse((Path(__file__).resolve().parents[1] / "app" / "services" / "engine"
                      / "pipeline.py").read_text(encoding="utf-8"), filename="pipeline.py")
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "run_sync":
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call) or not isinstance(inner.func, ast.Name):
                continue
            if inner.func.id != "_backend_phase":
                continue
            #: 位置参数第 2 个是阶段名 ✓；也认 `phase="encode"` 这种写法 ✓（两者都合法 ✓）
            arg = inner.args[1] if len(inner.args) >= 2 else next(
                (kw.value for kw in inner.keywords if kw.arg == "phase"), None)
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                sites.append((inner.lineno, arg.value))
    #: ⚠️ `ast.walk` 是**广度优先**、不是源码顺序 ✓✗ ⇒ 按 `lineno` 排 ✓（拿遍历下标比大小会随机绿红 ✗）
    sites.sort()
    phases = [phase for _, phase in sites]
    #: ⚠️ 2026-11-02：`run_sync` 多了一拍**可选**的 `init`（`request.init_image` 为真时才申请
    #: `"init": ("text", "vae")` ⇒ 图生图要在这一拍把参考图编成起点 ✓）。判据仍该盯
    #: 「**必经三拍按序都在** ＋ 没有表外的拍」✓✗ —— 写成 `phases == ["encode","sample","decode"]`
    #: 会把"多了一拍**合法**换装"误报成"少了一拍换装"✗（提示读起来还像真缺陷 ✓）。
    required = ["encode", "sample", "decode"]
    unknown = [phase for phase in phases if phase not in {*required, "init"}]
    kept = [phase for phase in phases if phase in required]
    #: ⚠️ 2026-11-07：`run_sync` 又有**一处**可选的 `encode` ✓✗ —— Wan2.2 的首帧条件前面
    #: **显式**要一次那一拍 ✓（见 `pipeline.py` 里「首帧条件**要 VAE**」那段长注释 ✓：两件已在场
    #: ⇒ 一件都不装、零代价 ✓；不在场 ⇒ 就地装上 ✓；不显式要 ⇒ 将来 init 真换了装就会静默缺 VAE ✓✗）。
    #: ⇒ 同 2026-11-02 那条 `init` 的坑 ✓：写成 `kept == required` 会把「多了一拍**合法**换装」
    #: 误报成「少了一拍换装」✓✗（提示读起来还像真缺陷 ✓）。判据改成三句话 ✓：
    #: ① 三拍**按序**都在（去掉重复后逐字等于 required ✓）；② 重复**只许** `encode` ✓✗
    #: （`sample`/`decode` 重复＝真换了两次装 ✓，那是缺陷 ✓）；③ 那次重复的 `encode` 必须落在
    #: `sample` **之前** ✓（首帧条件在采样前 ✓ —— 跑到采样后补编码等于白换 ✓）。
    deduped = [phase for index, phase in enumerate(kept) if phase not in kept[:index]]
    repeats = sorted({phase for phase in kept if kept.count(phase) > 1})
    ok_repeat = all(phase == "encode" for phase in repeats)
    ok_place = True
    if repeats:
        encode_sites = [index for index, phase in enumerate(phases) if phase == "encode"]
        ok_place = (len(encode_sites) == 2 and "sample" in phases
                    and encode_sites[1] < phases.index("sample"))
    #: `init` 若在，只许**紧跟 encode**（它换的是 text＋vae 那一组 ✓，跑到 sample 之后再换就白换 ✓✗）
    #: ⚠️ 先判 `encode` 在不在再取下标 ✗ —— 少了 `encode` 时 `index` 会抛 `ValueError`，
    #: 判据就从「报红」变成「测试自己崩」✓✗（那种崩还不带可读理由 ✓）。
    ok_init = True
    if "init" in phases and "encode" in phases:
        ok_init = phases.index("init") == phases.index("encode") + 1
    check("⑥″ `run_sync` 三处换装都在 ✓、顺序是 encode → sample → decode ✓✗"
          "（按调用点的阶段实参 ＋ 行号判，不按源码排版 ✗；`init` 是可选拍、只许紧跟 encode ✓；"
          "`encode` 允许重复一次且必须在 sample 之前 ✓，`sample`/`decode` 不许重复 ✗）",
          deduped == required and not unknown and ok_repeat and ok_place and ok_init, sites)


# ------------------------------------------- ⑦ 装得下就**不分时**（2026-11-07 ✓）


def case_all_resident() -> None:
    """⑦ 三件装得下 ⇒ **不分时** ✓✗（否则每张图都要重读一次文本塔 ✓）。

    这条守的是"**换装税**"这一级 ✓ —— 不验真装（真装是人为验收 ✓），验三件事：
    ① 表本身：四段都点名三件 ✓（少点名一件 ⇒ 那一拍就把它退了 ✓✗）；
    ② 门闩走**设备策略**（1 GiB 余量 ＋ 压力比 ✓）而不是手写阈值 ✗；
    ③ 数值两档对拍：int4 判 `off`（全驻留 ✓）、int8 判**不是** `off`（老实分时 ✓）。
    """
    from app.services.engine import device_policy as policy_mod  # noqa: PLC0415
    from app.services.engine import runtime as runtime_mod      # noqa: PLC0415

    table = runtime_mod.KREA2_ALL_RESIDENT_PHASES
    check("⑦ `KREA2_ALL_RESIDENT_PHASES` 四段（encode/init/sample/decode）**每段都点名三件** ✓✗"
          "（少点名一件 ⇒ `enter_phase` 会在那一拍把它退掉 ✓✗，换装税就回来了 ✓）",
          bool(table) and all(set(parts) == {"dit", "text", "vae"} for parts in table.values())
          and {"encode", "init", "sample", "decode"} <= set(table), table)

    #: ② 门闩不许自己写 GiB 阈值 ✗ —— 走 `decide_for`，只读它的 `residency` ✓。
    source = (Path(__file__).resolve().parents[1] / "app" / "services" / "engine"
              / "runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="runtime.py")
    #: ⚠️ 只扫 **这三个函数** ✗✗：`runtime.py` 里另有 2.1 / 2512 两条装配线也在调
    #: `declare_chain(phase_parts=…)` 与 `decide_for(("dit","text","vae"))` ✓（模组按族分时 ✓）——
    #: 全树扫会把它们一起数进来 ⇒ 判据报"收到 4 个" ✗（假红 ✓✗）。
    #: ⚠️ `decide_for` 现在住在 `_krea2_decision` 里 ✓（`_krea2_all_resident` 与 `_krea2_keep_parts`
    #: 都走它 ✓）—— 只扫前两个 ⇒ 这条会报"0 个" ✓✗（改成单一出口后必须跟着改扫描面 ✓）。
    scopes = [node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and node.name in {"_ensure_krea2_loaded", "_krea2_decision"}]
    declares: list[ast.Call] = []
    gates: list[ast.Call] = []
    for node in (ast.walk(scope) for scope in scopes):
        for sub in node:
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "declare_chain":
                declares.append(sub)
            elif name == "decide_for":
                gates.append(sub)
    passed = [(kw.arg, kw.value) for call in declares for kw in call.keywords
              if kw.arg == "phase_parts"]
    kept = [(kw.arg, kw.value) for call in declares for kw in call.keywords
            if kw.arg == "keep_parts"]
    check("⑦′ `declare_chain` 真收到 `phase_parts=` ✓，且值**按门闩**给（`KREA2_ALL_RESIDENT_PHASES`"
          " 或 `None` ✓ —— 直接写死表 ⇒ 装不下的档也会全驻留 ⇒ OOM 落在采样第 1 步 ✓✗）",
          len(passed) == 1 and "KREA2_ALL_RESIDENT_PHASES" in ast.dump(passed[0][1])
          and any(isinstance(node, ast.Constant) and node.value is None
                  for node in ast.walk(passed[0][1])), passed)
    #: `runtime.py` 这条装配线里 `decide_for` 有**两个**调用点 ✓：一个是 `_apply_decision`
    #: 那条（管 DiT 自己装法 ✓），另一个是 `_krea2_decision` 这个**唯一问"装得下吗"的出口** ✓
    #: （`_krea2_all_resident` / `_krea2_keep_parts` 都走它 ✓）。判据盯的正是后者 **只有一处** ✓✗。
    single = [call for call in gates
              if "tuple(parts)" in ast.unparse(call.args[0])[:40]
              and any(kw.arg == "need_gib" for kw in call.keywords)]
    check("⑦″ 门闩读的是设备策略（`decide_for(tuple(parts), …, need_gib=part_need)` ✓）而不是"
          "自己算 GiB ✗，且\"装得下吗\"**只有一处出口** ✓✗（各写一份 ⇒ 口径就漂 ✓）",
          len(single) == 1 and len(gates) == 2,
          [ast.unparse(call)[:90] for call in gates])
    check("⑦⁵ `declare_chain` 也真收到 `keep_parts=` ✓✗（漏给 ⇒ 分时档每张图都要把文本塔"
          "**从盘上重读一遍** ✓：真卡实测 19.4 秒／张 ✓，且与提示词无关 ✓）",
          len(kept) == 1 and "keep_parts" in ast.unparse(kept[0][1]), kept)

    #: ⑦⁗ ⚠️ 这条是**真跑抓到的回归** ✓✗：`decide_for` 返的是 `Decision`（dataclass ✓），
    #: 写成 `verdict.get(...)` 会抛 `AttributeError` ✓，再被 `except` 一兜 ⇒ **静默变分时** ✓✗
    #: （表现＝"全驻留永不生效，回执一切正常"✓✗，最贵的那种 bug ✓）。两种形状都必须认 ✓。
    class _Stub:  # noqa: D401 - 一次性桩件
        def __init__(self, verdict: object) -> None:
            self._verdict = verdict

        def decide_for(self, parts, **_kwargs):  # noqa: ANN001, ANN201
            return self._verdict

    def verdict(text: str):  # noqa: ANN202
        return policy_mod.Decision(device="cuda", residency=text, degraded=False,
                                   reason="桩件", policy="auto")

    class _Boom:  # noqa: D401 - 一次性桩件
        def decide_for(self, parts, **_kwargs):  # noqa: ANN001, ANN201
            raise RuntimeError("探不到卡 ✓")

    helper = runtime_mod._krea2_all_resident  # noqa: SLF001
    got = {
        "Decision(off)": helper(_Stub(verdict("off")), {}),
        "dict(off)": helper(_Stub({"residency": "off"}), {}),
        "Decision(split)": helper(_Stub(verdict("split")), {}),
        "抛异常": helper(_Boom(), {}),
    }
    check("⑦⁗ 门闩认 **`Decision` 与 `dict` 两种形状** ✓✗（只认 dict ⇒ `AttributeError` 被 `except`"
          " 兜成\"永远分时\"✓✗，实测就是这么漏的 ✓）＋ 抛异常时退回分时 ✓",
          got == {"Decision(off)": True, "dict(off)": True,
                  "Decision(split)": False, "抛异常": False}, got)

    #: ⑦⁶ ⚠️ 只有**整份装得下（`off`）**的件才许停内存 ✓✗：`split`（分层搬运 ✓）的件一旦混进
    #: `keepParts`，`_relocate_keep_parts` 会把它**整份**搬去 cpu 再**整份**搬回卡上 ✓✗ ⇒
    #: 分层计划当场作废 ✓、搬回时直接 OOM ✓（8 GiB 低配档正是这种 ✓）。
    keeper = runtime_mod._krea2_keep_parts  # noqa: SLF001

    class _Pick:  # noqa: D401 - 一次性桩件：只认 `("text",)` 判 off
        def decide_for(self, parts, **_kwargs):  # noqa: ANN001, ANN201
            return verdict("off" if tuple(parts) == ("text",) else "split")

    keeper_got = {
        "together": keeper(_Stub(verdict("off")), {}, together=True),
        "两件都off": keeper(_Stub(verdict("off")), {}, together=False),
        "只text": keeper(_Pick(), {}, together=False),
        "全split": keeper(_Stub(verdict("split")), {}, together=False),
        "判不出来": keeper(_Boom(), {}, together=False),
    }
    check("⑦⁶ 停内存的件**只许整份装得下的**（`off` ✓）：`split` 的件混进去 ⇒ 整份搬来搬去 ⇒"
          " 分层计划作废 ＋ 搬回时 OOM ✓✗（8 GiB 档正是这种 ✓）；同驻档一件都不搬 ✓",
          keeper_got == {"together": (),
                         "两件都off": ("dit", "text", "vae"), "只text": ("text",),
                         "全split": (), "判不出来": ()}, keeper_got)

    #: ⑦⁷ ⭐ **"停到内存"不许改数** ✓✗（创作者红线："不能质量掉" ✓）—— 搬的是**同一份权重字节** ✓
    #: （`nn.Module.to()` 同 dtype 只做设备拷贝 ✓、不转精度 ✗），所以 `cpu → cuda → cpu` 走一圈
    #: 必须**逐位相同** ✓。没有 CUDA 就跳过 ✓（这是设备事实、不是判据放宽 ✓）。
    try:
        import torch  # noqa: PLC0415 - 只为这一条判据
    except Exception:  # pragma: no cover - 引擎装不上就别在这儿报 ✓
        torch = None                                            # type: ignore[assignment]
    if torch is not None and torch.cuda.is_available():
        tower = torch.nn.Module()
        #: ⚠️ int8 参数必须 `requires_grad=False` ✓（`Int8Linear.weight` 就是这么建的 ✓）——
        #: 默认 True 时 `Parameter.__new__` 当场抛「Only Tensors of floating point …」✗。
        tower.w = torch.nn.Parameter(torch.randint(-128, 127, (97, 33), dtype=torch.int8),
                                    requires_grad=False)
        tower.register_buffer("scale", torch.rand(33, dtype=torch.float32))
        tower.register_buffer("bf", torch.randn(5, 5, dtype=torch.bfloat16))
        tower.register_buffer("idx", torch.arange(7, dtype=torch.int64))
        snapshot = {name: tensor.detach().cpu().clone() for name, tensor
                    in [*tower.named_parameters(), *tower.named_buffers()]}
        tower.to("cuda").to("cpu")
        moved = {name: tensor.detach().cpu() for name, tensor
                 in [*tower.named_parameters(), *tower.named_buffers()]}
        same = {name: (snapshot[name].dtype == moved[name].dtype
                       and torch.equal(snapshot[name], moved[name])) for name in snapshot}
        check("⑦⁷ 「停到内存」搬的是**同一份字节** ✓（`cpu→cuda→cpu` 逐位相同 ✓、dtype 不变 ✓）"
              "⇒ 步数 / 精度 / 采样参数一律不动 ✓（红线：不能质量掉 ✓）",
              bool(same) and all(same.values()), same)

    #: ③ 数值对拍：同一张 22.49 GiB 卡、同样 20.93 GiB 可用 ✓。
    free = int(20.93 * policy_mod.GIB)
    pressure = policy_mod.Pressure(totalBytes=int(22.49 * policy_mod.GIB), freeBytes=free)

    def residency_of(need: dict[str, float]) -> str:
        decision = policy_mod.decide(policy="auto", need_bytes=int(sum(need.values())
                                                                     * policy_mod.GIB),
                                     pressure=pressure)
        return str(getattr(decision, "residency", ""))

    int4 = residency_of({"dit": 5.98, "text": 7.5, "vae": 0.24})     # 13.72 GiB ✓
    int8 = residency_of({"dit": 12.57, "text": 7.5, "vae": 0.24})    # 20.31 GiB ✓
    check("⑦‴ int4 档（5.98 ＋ 7.5 ＋ 0.24 ＝ 13.72 GiB）判 `off` ⇒ 全驻留 ✓；"
          "int8 档（20.31 GiB）**不**判 `off` ⇒ 老实分时 ✓（挤得进但没余量给 1024² 的激活 ✓✗）",
          int4 == "off" and int8 != "off", {"int4": int4, "int8": int8, "freeGiB": 20.93})


def main() -> int:
    case_stage_names()
    case_shift()
    case_plan()
    case_runtime_refuses()
    case_stage_registered()
    case_phased()
    case_all_resident()

    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      实测: {detail}")
    for reason in _SKIPS:
        print(f"SKIP  {reason}")
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
