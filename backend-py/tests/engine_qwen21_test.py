"""自检：**Qwen-Image 2.1 的出图口径与装配门禁**（2026-10-31）。

为什么单独一份 ✗（`engine_zimage_test` 已经钉过"σ 要走 flow" ✓）：2.1 与 Z-Image 在**同一个形状上**
犯的是**不同的错** ✗✗ —— 两者都是 flow ✓、都出 4 维潜变量 ✓、都不报错 ✓，但：

==========================  ==============================  ===============================
                           Z-Image                         Qwen-Image **2.1**
==========================  ==============================  ===============================
flow shift                 ``3.0`` ✓                        **``0.69``** ✓
VAE 下采样                  8× ✓（尺寸吸附 16 = 8 × patch 2 ✓）  **16×** ✓ 且 **没有 patchify** ✗（只吸附 16）
潜变量通道                  **16** ✓                          **64** ✓
文本塔                     Qwen3-4B（`text_dim` 2560 ✓）      Qwen3.5-9B（**4096** ✓ / 混合线性注意力 ✓）
VAE 键数                   244 ✓（Flux ae ✓）                **238** ✓（Wan 式 3D 因果 ✓）
==========================  ==============================  ===============================

⇒ 四处「形状都对、图不对」的换法：σ 用 3 ✓✗（前几步定死构图 ✓）、尺寸多乘一次 patch 2 ✓✗（白裁边 ✓）、
VAE 拿 16 通道那份 ✓✗（通道数会当场拒 ✓，这条能拦）、**同族不同代**的 DiT（2.1 649 键 vs 2512 1933 键 ✓✗
—— `text_dim` 与 VAE 通道完全一样 ✓，三道互核全过 ✓，只有族名拦得住 ✓）。

本文件钉的是**口径与门禁** ✓：σ 的 shift ✓、尺寸吸附 ✓、四格缺项点名 ✓、换 TE/VAE 必须换装载键 ✓、
就绪档只列这一条链的三件 ✓。⚠️ **不跑真装** ✗（三份权重 ≈16.2 GiB ✓，自检里不装 ✓）——
真装与整链出图是人为验收的那一步 ✓（见 `app/scripts/dit_families.py::ENGINE_STATE` ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_qwen21_test.py
"""
from __future__ import annotations

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
    base = dict(prompt="a cat", stage=pipeline.QWEN21_STAGE)
    base.update(kw)
    return pipeline.build_plan(pipeline.GenerationRequest(**base))


# --------------------------------------------------------------- ① 阶段名


def case_stage_names() -> None:
    """① 三个「出一张图」的 stage 必须**互不相同** ✗✗（合流 ⇒ σ / 通道 / 后端三处一起串 ✓）。"""
    names = (pipeline.IMAGE_STAGE, pipeline.ZIMAGE_STAGE, pipeline.QWEN21_STAGE)
    check("① `image` / `zimage` / `qwen_image21` 三名互不相同 ✓✗（合一个 ⇒ 后端也只有一个 ✓）",
          len(set(names)) == 3, names)
    message = _error_of(lambda: pipeline.build_plan(
        pipeline.GenerationRequest(prompt="x", stage="nope")))
    check("①′ 未知 stage ⇒ 报错把**四个名字**都列出来 ✓（别让调用方猜 ✓）",
          all(name in message for name in names), message)


# --------------------------------------------------------------- ② σ 口径


def case_sigmas() -> None:
    """② **shift=0.69** ✓（不是 3 ✗）：端点 / 手算点 / 与 3 的差。

    ⚠️ 这条是 2.1 最容易"看着能跑"的坑 ✗✗：拿 3.0 跑 ⇒ 形状全对 ✓、步数全对 ✓、
    日志全对 ✓，只是**前几步就把构图定死、后几步几乎不动** ✓✗。
    """
    steps = 20
    sigmas = schedules.flow_sigmas_for_steps(steps, shift=pipeline.QWEN21_FLOW_SHIFT)
    check("② 长度 = 步数 + 1 ✓、σ₀ = 1 ✓、σ_last = 0 ✓",
          len(sigmas) == steps + 1 and sigmas[0] == 1.0 and sigmas[-1] == 0.0,
          (len(sigmas), sigmas[0], sigmas[-1]))
    check("②′ 严格单调递减 ✓", all(a > b for a, b in zip(sigmas, sigmas[1:-1])), sigmas)
    # 手算：σ(t) = shift·t / (1 + (shift − 1)·t)，t = 1 − 1/steps；shift=0.69
    t = 1 - 1 / steps
    want = 0.69 * t / (1 + (0.69 - 1) * t)
    check(f"②″ 第 2 个点 = {want:.6f} ✓（公式/常量写错必红 ✓）",
          abs(sigmas[1] - want) < 1e-9, (sigmas[1], want))
    shifted = schedules.flow_sigmas_for_steps(steps, shift=3.0)
    check("②‴ 0.69 与 3.0 **逐点不同** ✓，且 0.69 在中间档一律更**偏干净** ✓"
          "（这就是「套错 shift 图会僵」的形状 ✓）",
          all(a < b for a, b in zip(sigmas[1:-1], shifted[1:-1])),
          list(zip(sigmas[:4], shifted[:4])))


# --------------------------------------------------------------- ③ plan


def case_plan() -> None:
    """③ 尺寸只吸附到 **16** ✓（这一族**没有** patchify ✗）＋ 单帧 ＋ 默认像素预算被如实换掉 ✓。"""
    plan = _plan(ratio="1:1")
    check("③ 默认像素预算（0.65 MP 的**视频口径** ✗）⇒ 换 Qwen-Image 官方示例档 ✓ 且**如实告警** ✓"
          "（1:1 时正是官方 **1328×1328** ✓）",
          (plan.width, plan.height) == (1328, 1328)
          and any("0.65" in text and "1.76" in text for text in plan.warnings),
          ((plan.width, plan.height), plan.warnings[:2]))
    check("③′ 单张图 ⇒ `frames=1` ✓、`fps=0` ✓（不是视频 ✓）",
          plan.frames == 1 and plan.fps == 0, (plan.frames, plan.fps))

    #: ⚠️ 反向：**16 的倍数、但不必是 32 的倍数** ✗✗ —— 2.1 的 `img_in` 是 `Linear` ✓
    #: （`dit_qwen21` 模块头 ✓），多乘一次 patch 2 只是白裁边 ✓、不报错 ✓。
    #: 3:2 的 1.0 MP ⇒ √1.5M = **1225** ⇒ 吸 16 ⇒ **1232**（77×16 ✓、**不是** 32 的倍数 ✓）——
    #: 走默认 `RESOLUTION_MULTIPLE=32` 会得到 1248 ✓✗，这一档才分得出来 ✓。
    odd = _plan(megapixels=1.0, ratio="3:2")
    check("③″ 宽高都吸附到 **16** 的倍数 ✓", odd.width % 16 == 0 and odd.height % 16 == 0,
          (odd.width, odd.height))
    #: ⚠️ 判据是「**至少一边**不是 32 的倍数」✗：高这边 832 恰好也是 32 的倍数 ✓（巧合 ✓），
    #: 要求"两边都不是"会把正确实现判红 ✓✗。
    check("③‴ ⭐ **不**被额外吸到 32 ✓✗（有 patchify 的族才需要 16 = 8×2 ✓；2.1 多裁一刀会白掉边 ✓）",
          odd.width % 32 != 0 or odd.height % 32 != 0, (odd.width, odd.height))

    steps = _plan(steps=30, megapixels=1.0, ratio="1:1")
    check("③⁗ `timesteps` **就是 σ** ✓（multiplier=1.0 ✓，不乘 1000 ✗）",
          [round(float(v), 9) for v in steps.timesteps]
          == [round(float(v), 9) for v in steps.sigmas]
          and abs(steps.sigmas[1] - 0.69 * (1 - 1 / 30) / (1 + (0.69 - 1) * (1 - 1 / 30))) < 1e-9,
          steps.sigmas[:3])
    #: 默认 ``guidance`` 就是 ``scale=1.0`` ✓（⇒ 等效不引导 ✓）—— 2.1 不是 cfg=1 上蒸馏的 ✗，
    #: 官方档 2.5~4.0 ✓ ⇒ 这条**只报不改** ✓（改用户的引导配置越界 ✗）。
    quiet = _plan(steps=30, megapixels=1.0, ratio="1:1")
    check("③⁵ cfg=1.0 ⇒ **只报不改** ✓（2.1 不是 cfg=1 上蒸馏的 ✗；改用户的引导配置越界 ✗）",
          any("cfg=1.0" in text for text in quiet.warnings),
          quiet.warnings[-1:])


# --------------------------------------------------------------- ④ 装配门禁


def case_runtime_refuses() -> None:
    """④ 四格缺项**当场拒** ✓、只点名真缺的 ✓、换组件必须换装载键 ✓。

    ⚠️ 这条**不跑真装载** ✗（≈16.2 GiB ✓）：验的是门口 ✓。
    """
    from app.services.engine import runtime as rt  # noqa: PLC0415

    call = lambda **kw: rt.EngineRuntime()._ensure_loaded_internal(  # noqa: E731
        stage=pipeline.QWEN21_STAGE, **kw)

    message = _raises(call)
    check("④ `stage=qwen_image21` 什么都不给 ⇒ **当场报** ✓✗（不静默当别的档装 ✓、不半装 ✓）",
          _underlying_type(call) is rt.TorchBackendUnavailable, message)
    check("④′ 拒的理由把**四格**都点名 ✓（`ditPath` / `textWeights` / `imageVaeWeights` / `tokenizerPath`）",
          all(key in (message or "") for key in
              ("ditPath", "textWeights", "imageVaeWeights", "tokenizerPath")), message)
    check("④″ 报错里点了**本机真文件长什么样** ✓（649 键 / 238 键 / 别拿 2512 ✗）——"
          "照着报错就能补齐 ✓",
          all(text in (message or "") for text in ("649", "238", "2512")), message)

    partial = _raises(lambda: call(dit_path="d.safetensors", text_weights="t.safetensors",
                                   tokenizer_path="tok/"))
    #: ⚠️ 只看**首行** ✗✗：第二行起是 `_QWEN21_STAGE_NOTE` 的四格说明 ✓（有意列的 ✓）。
    head = (partial or "").split("\n", 1)[0]
    check("④‴ 给了三格、只缺 `imageVaeWeights` ⇒ **缺项清单里只有它** ✓（说明段列四格不算 ✗）",
          "imageVaeWeights" in head and "textWeights" not in head, head)

    base = dict(stage=pipeline.QWEN21_STAGE, dit_path="d.safetensors", tokenizer_path="tok/")
    k_a = rt._load_key(**base, text_weights="te_a.safetensors", image_vae_weights="vae_a.safetensors")
    k_te = rt._load_key(**base, text_weights="te_b.safetensors", image_vae_weights="vae_a.safetensors")
    k_vae = rt._load_key(**base, text_weights="te_a.safetensors", image_vae_weights="vae_b.safetensors")
    check("④⁴ 换 `textWeights` ⇒ 装载键变 ✓（不复用上一套 9B ✗）", k_a != k_te, (k_a, k_te))
    check("④⁵ 换 `imageVaeWeights` ⇒ 装载键变 ✓（不复用上一套 64 通道 VAE ✗）", k_a != k_vae, (k_a, k_vae))


# --------------------------------------------------------------- ⑤ 就绪与登记


def case_stage_registered() -> None:
    """⑤ 就绪档**只列这条链的三件** ✗、三件全算必需 ✓、与 `plan_stage` 同一份条目 ✓。

    ⚠️ **不许断言 `ready`** ✗✗：那是"本机三份权重在不在"，换机必红 ✓。
    """
    from app.services.engine import inventory as inv  # noqa: PLC0415
    from app.services.engine import loader, registry  # noqa: PLC0415

    check("⑤ `STAGE_FILTERS` 与 `registry.STAGES` 双向对齐 ✓（少一边就别处会红 ✓）",
          pipeline.QWEN21_STAGE in inv.STAGE_FILTERS and pipeline.QWEN21_STAGE in registry.STAGES
          and inv.STAGE_FILTERS[pipeline.QWEN21_STAGE]["category"]
          == registry.STAGES[pipeline.QWEN21_STAGE].category,
          {"filter": inv.STAGE_FILTERS.get(pipeline.QWEN21_STAGE),
           "registry": getattr(registry.STAGES.get(pipeline.QWEN21_STAGE), "category", None)})

    report = inv.readiness(pipeline.QWEN21_STAGE)
    keys = sorted(row["key"] for row in report["components"])
    check("⑤′ `readiness(\"qwen_image21\")` **只列这条链的三件** ✓✗"
          "（把 SDXL / Z-Image 的组件混进来 ⇒ 24 GB 卡会凭空报装不下 ✓✗）",
          keys == ["dit_qwen_image21", "te_qwen35_image21", "vae_qwen_image21"], keys)
    check("⑤″ 三件在这个档里**全部算必需** ✓（`requireAll` 生效 ✓；`required: false` 不是「不需要」✗）",
          all(row["required"] for row in report["components"]),
          [(row["key"], row["required"]) for row in report["components"]])
    #: 6.76 + 8.82 + 0.63 = 16.21 GiB ✓ —— 与卡上 22.49 的余量账对得上 ✓（真装实测那一步会再核 ✓）。
    check("⑤‴ 必需权重合计 ≈ **16.21 GiB** ✓（只算这三件 ✓）",
          abs(float(report["requiredWeightsGiB"]) - 16.21) < 0.05,
          report["requiredWeightsGiB"])

    #: ⚠️ `plan_stage` 与 `readiness` 必须走**同一份筛选** ✗✗（2026-10-31 抽 `stage_entries` 之前
    #: `plan_stage` 自己写了一遍比较 ⇒ 列出 SDXL 且必需集合为空 ⇒ `ready` 恒 True ✓✗）。
    plan = loader.plan_stage(pipeline.QWEN21_STAGE)
    plan_keys = sorted(row["key"] for row in plan["components"])
    check("⑤⁴ `loader.plan_stage` 与就绪档**同一份条目** ✓✗",
          plan_keys == keys, {"plan": plan_keys, "readiness": keys})
    required_keys = sorted(row["key"] for row in plan["components"] if row.get("required"))
    check("⑤⁵ `plan_stage` 的必需集合**非空** ✓✗（空集合 ⇒ `ready = all([]) = True`，文件全缺也说就绪 ✓）",
          required_keys == keys, required_keys)

    #: ⚠️ 反向：`image` 档（SDXL）**不许**把这 16.21 GiB 算进必需 ✗✗ —— 两套是**互斥备选** ✓，
    #: 加一起会在 24 GB 卡上凭空报「装不下」✓✗。
    image_report = inv.readiness("image")
    check("⑤⁶ `readiness(\"image\")` 的必需集合**不含** 2.1 三件 ✓✗（互斥备选不许相加 ✓）",
          not any(row["key"] in keys and row["required"]
                  for row in image_report["components"]),
          [(row["key"], row["required"]) for row in image_report["components"]])


# ------------------------------------------------- ⑥ 分时驻留（三件装不下 ✓）


class _PhaseBackend:
    """只记「谁被装、谁被退」的假后端 ✓（**不读盘** ✓）—— 验的是换装**机制** ✓。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []


def case_phased() -> None:
    """⑥ 三件 bf16 驻留 **28.68 GiB > 22.49 GiB** ✗✗ ⇒ 必须按阶段互斥换装 ✓。

    ⚠️ 这条**不装真权重** ✗：把三个装载方法换成记账的 ✓，验「进哪个阶段只剩哪一件在场」✓
    —— 真装与出图是人为验收那一步 ✓。
    """
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    #: ⑥ 三件**互不重叠** ✓✗ —— 只要有一件出现在两个阶段，"峰值 = max(单件)"这条账就假了 ✓✗
    #: （那件会跟另一件同住 ✓，28 GiB 又回来了 ✓）。
    all_parts = [part for parts in tb.TorchBackend.PHASE_PARTS.values() for part in parts]
    check("⑥ `PHASE_PARTS` 三件**互斥** ✓✗（一件出现在两个阶段 ⇒ 峰值不再是单件 ✓）",
          sorted(all_parts) == ["dit", "text", "vae"], tb.TorchBackend.PHASE_PARTS)
    check("⑥′ 三个阶段名就是管线的三段 ✓（encode / sample / decode ✓）",
          sorted(tb.TorchBackend.PHASE_PARTS) == ["decode", "encode", "sample"],
          sorted(tb.TorchBackend.PHASE_PARTS))

    class Fake(tb.TorchBackend):        # noqa: D401 —— 只覆盖装载，不碰盘 ✓
        def __init__(self) -> None:     # noqa: D107
            super().__init__(device="cpu")
            self.events = []
            self._chain = None

        def load_weights(self, *, path=None, **_kw):     # noqa: ANN001, ANN003, D102
            self.events.append(("load", "dit"))
            self._model = object()
            return {"path": path}

        def attach_qwen35_te(self, weights, *, tokenizer_path=None, **_kw):  # noqa: ANN001, ANN003, D102
            self.events.append(("load", "text"))
            self._textEncoder = object()
            return {"weights": weights}

        def attach_wan_vae(self, weights, *, family="qwen_image21", **_kw):  # noqa: ANN001, ANN003, D102
            self.events.append(("load", "vae"))
            self._vae = object()
            self._vaeKind = "wan21-real"
            return {"weights": weights, "family": family}

    backend = Fake()
    check("⑥″ 没登记链 ⇒ `enter_phase` **一动不动** ✓（Z-Image / SDXL / H3 零影响 ✓）",
          backend.enter_phase("encode") is None and backend.events == [], backend.events)

    backend.declare_chain(dit_path="d.safetensors", text_weights="t.safetensors",
                          tokenizer_path="tok/", vae_path="v.safetensors")
    backend.enter_phase("encode")
    check("⑥‴ `encode` ⇒ **只装 TE** ✓（不顺手把 DiT/VAE 也装上 ✓✗）",
          backend.events == [("load", "text")]
          and backend._textEncoder is not None
          and backend._model is None and backend._vae is None, backend.events)

    backend.enter_phase("sample")
    check("⑥⁴ `sample` ⇒ **退掉 TE、只装 DiT** ✓（TE 还在就是 28 GiB 那条路 ✓✗）",
          backend._textEncoder is None and backend._model is not None and backend._vae is None,
          backend.events)

    again = backend.enter_phase("sample")
    check("⑥⁵ 同一阶段进两次 ⇒ **不重装** ✓（`reused` 报 dit ✓；重复装会白付一次 13 GiB 读盘 ✓）",
          isinstance(again, dict) and again.get("reused") == ["dit"], again)

    backend.enter_phase("decode")
    check("⑥⁶ `decode` ⇒ **退掉 DiT、只装 VAE** ✓（DiT 13.26 还占着就装不下 64 通道 VAE ✓✗）",
          backend._model is None and backend._vae is not None
          and backend._vaeKind == "wan21-real", backend.events)

    backend.enter_phase("refine")
    check("⑥⁷ 非三段的阶段名（如 `refine` ✓）⇒ **不换装** ✓（超清二采还用同一个 DiT ✓）",
          backend._vae is not None and backend._model is None, backend.events)

    bad = _error_of(lambda: backend.release_part("audio"))
    check("⑥⁸ 不认识的部件 ⇒ **当场抛** ✓✗（静默不动会让调用方以为卸过了 ✓）",
          "audio" in bad, bad)

    #: ⑥⁹ ⚠️ 反向：钩子必须**真挂进管线** ✗✗ —— 机制写好了但 `run_sync` 从不调用，
    #: 就是「装不下」照样 OOM ✓✗（这条断言专防"删掉调用/写错顺序"✓）。
    import inspect  # noqa: PLC0415

    source = inspect.getsource(pipeline.run_sync)
    positions = [source.find(f'_backend_phase(backend, "{phase}")')
                 for phase in ("encode", "sample", "decode")]
    check("⑥⁹ `run_sync` 里三处换装**都在** ✓、顺序是 encode → sample → decode ✓✗",
          all(position > 0 for position in positions)
          and positions == sorted(positions), positions)

    #: ⑥¹⁰ 没有 `enter_phase` 的后端（SDXL / H3 / 参考实现 ✓）⇒ 钩子返回 None ✓、不抛 ✓
    check("⑥¹⁰ 后端没有 `enter_phase` ⇒ 管线钩子**直接放行** ✓（不为别的族加分支 ✓）",
          pipeline._backend_phase(object(), "encode") is None, "")


# ------------------------------------------------- ⑦ 两种设备都能用（GPU / CPU ✓）


def case_both_devices() -> None:
    """⑦ **GPU 与 CPU 两条都要能跑** ✓（创作者 2026-10-31 口径 ✓）。

    ⚠️ 这条只验「精度跟着**设备**走」✓✗ —— 不验「能装」而验「选对了精度」✓：
    旧写法四处都按 ``torch.cuda.is_available()`` 选 bf16 ✓ ⇒ 在**有卡但显式 ``device="cpu"``**
    的机器上，权重会被建成 bf16 ✓✗，而 CPU 上 bf16 的 ``conv2d`` 既慢又有算子未实现 ✓✗
    （VAE 解码最先撞上 ✓，报错只给算子名、指不到"精度选错了" ✗）。
    """
    import torch  # noqa: PLC0415

    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    cpu = tb.TorchBackend(device="cpu")
    check("⑦ CPU 后端 ⇒ 计算精度 **fp32** ✓✗（跟着设备走；跟着\"机器有没有卡\"走会选错 ✓）",
          cpu._compute_dtype() is torch.float32, cpu._compute_dtype())
    check("⑦′ CPU 后端自述的设备就是 `cpu` ✓（不跟着机器有没有卡变 ✓）",
          cpu.device == "cpu" and cpu.describe().get("device") == "cpu",
          (cpu.device, cpu.describe().get("device")))

    if not torch.cuda.is_available():
        _SKIPS.append("⑦″ 本机没有 CUDA ⇒ CUDA 侧精度断言跳过（CPU 那两条已验 ✓）")
        return
    gpu = tb.TorchBackend(device="cuda")
    check("⑦″ CUDA 后端 ⇒ **bf16** ✓（A5000 原生支持 ✓；本机有卡时必须仍是 bf16 ✓✗ 别为 CPU 让步 ✓）",
          gpu._compute_dtype() is torch.bfloat16, gpu._compute_dtype())
    check("⑦‴ 同一台机器上两个后端**各说各的设备** ✓✗（不许共用一份探测结果 ✓）",
          gpu.device == "cuda" and cpu.device == "cpu", (gpu.device, cpu.device))
    check("⑦⁴ 分时驻留**与设备无关** ✓：CPU 后端登记的链同样能换装 ✓（两个后端同一套 `PHASE_PARTS` ✓）",
          cpu.PHASE_PARTS is tb.TorchBackend.PHASE_PARTS
          and gpu.PHASE_PARTS is tb.TorchBackend.PHASE_PARTS, "")


def main() -> int:
    case_stage_names()
    case_sigmas()
    case_plan()
    case_runtime_refuses()
    case_stage_registered()
    case_phased()
    case_both_devices()

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
