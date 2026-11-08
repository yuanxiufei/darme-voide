"""自检：**Qwen-Image 2512 / Edit 2511 的出图口径与装配门禁**（2026-10-31）。

为什么单独一份 ✗（`engine_qwen21_test` 已经钉过"2.1 的 shift 是 0.69" ✓）：2512 与 2.1 的族名
**只差一个 `21`** ✗✗（``qwen_image`` vs ``qwen_image21`` ✓），而两者的关键事实件件不同 ✓：

==========================  ==============================  ===============================
                           2.1（2509 ✓）                    **2512 / Edit 2511**
==========================  ==============================  ===============================
DiT 键数 / 结构            649 / 单流 ✓                     **1933**（Edit 3614）/ 双流 ✓
潜通道                     64 ✓                            **16** ✓（`img_in` 吃 16×2×2 ✓）
flow shift                 **0.69** ✓                      **3.1** ✓
时间口径                   lumina 式翻转 ✓                  **`1000·σ` 不翻转** ✓
文本塔                     Qwen3.5-9B / 4096 / 取 −2 ✓      **Qwen2.5-VL-7B / 3584 / 取 −1** ✓
==========================  ==============================  ===============================

⇒ 三种「形状都对、图不对」的换法：**shift 抄成 0.69** ✓✗（σ 曲线整体被推 ⇒ 提示词跟随变差 ✓）、
尺寸只吸到 **8** ✓✗（潜边长 83 奇数 ⇒ `img_in` 拍平当场形状错 ✓）、
`text_kind` 留缺省 `qwen35` ✓✗（报的是「缺 `linear_attn.*`」，像权重下错了 ✓）。

本文件钉的是**口径与门禁** ✓：σ 的 shift ✓、尺寸吸附 16 且**潜边长为偶** ✓、四格缺项点名 ✓、
`text_kind` / `vae_family` / `part_need_gib` 三个按族参数真被用上 ✓、就绪档只列这条链的三件 ✓、
**两本账不许混** ✓✗（拿 2.1 的 13.26 判 38.05 ⇒ 结论"装得下" ⇒ 装到一半 OOM ✓✗）。
⚠️ **不跑真装** ✗（三份权重 ≈54.4 GiB ✓，自检里绝不动它 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_qwen_image_test.py
"""
from __future__ import annotations

import inspect
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
    base = dict(prompt="a cat", stage=pipeline.QWEN_IMAGE_STAGE)
    base.update(kw)
    return pipeline.build_plan(pipeline.GenerationRequest(**base))


# --------------------------------------------------------------- ① 阶段名


def case_stage_names() -> None:
    """① 四张图的阶段名**互不相同** ✗✗（合流 ⇒ σ / 通道 / 文本塔一起串 ✓）。

    ⚠️ `qwen_image21` 以 `qwen_image` 开头 ✓✗ ⇒ 这条断言是「前缀判族」的守卫 ✓：
    谁把 ``stage == QWEN_IMAGE_STAGE`` 改成 ``startswith``，2.1 立刻被 2512 的 plan 接走 ✓
    （shift 3.1 套在 649 键的单流 DiT 上 ⇒ 不报错、图僵 ✓✗）。
    """
    names = (pipeline.IMAGE_STAGE, pipeline.ZIMAGE_STAGE, pipeline.QWEN21_STAGE,
             pipeline.QWEN_IMAGE_STAGE)
    check("① `image` / `zimage` / `qwen_image21` / `qwen_image` 四名互不相同 ✓✗",
          len(set(names)) == 4, names)
    check("①′ `qwen_image21` **以** `qwen_image` 开头 ✓ —— 所以族名比较只能 `==` ✗✗",
          pipeline.QWEN21_STAGE.startswith(pipeline.QWEN_IMAGE_STAGE)
          and pipeline.QWEN21_STAGE != pipeline.QWEN_IMAGE_STAGE, names)

    message = _error_of(lambda: pipeline.build_plan(
        pipeline.GenerationRequest(prompt="x", stage="nope")))
    check("①″ 未知 stage ⇒ 报错把**四个名字**都列出来 ✓（别让调用方猜 ✓）",
          all(name in message for name in names), message)


# --------------------------------------------------------------- ② σ 口径


def case_sigmas() -> None:
    """② **shift=3.1** ✓（不是 2.1 的 0.69 ✗、也不是 Z-Image 的 3 ✗）。

    ⚠️ 这条是 2512 最容易"看着能跑"的坑 ✗✗：拿 0.69 跑 ⇒ 形状全对 ✓、步数全对 ✓、
    日志全对 ✓，只是**前几步几乎不动、后几步突然定死** ✓✗（与 3.1 恰好相反 ✓）。
    """
    steps = 20
    sigmas = schedules.flow_sigmas_for_steps(steps, shift=pipeline.QWEN_IMAGE_FLOW_SHIFT)
    check("② 长度 = 步数 + 1 ✓、σ₀ = 1 ✓、σ_last = 0 ✓",
          len(sigmas) == steps + 1 and sigmas[0] == 1.0 and sigmas[-1] == 0.0,
          (len(sigmas), sigmas[0], sigmas[-1]))
    check("②′ 严格单调递减 ✓", all(a > b for a, b in zip(sigmas, sigmas[1:-1])), sigmas)
    t = 1 - 1 / steps
    want = 3.1 * t / (1 + (3.1 - 1) * t)
    check(f"②″ 第 2 个点 = {want:.6f} ✓（shift 常量写错必红 ✓）",
          abs(sigmas[1] - want) < 1e-9, (sigmas[1], want))

    other = schedules.flow_sigmas_for_steps(steps, shift=0.69)
    check("②‴ 3.1 与 2.1 的 0.69 **逐点不同** ✓，且 3.1 在中间档一律更**偏噪** ✓"
          "（这就是「套错 shift 图会僵」的形状 ✓）",
          all(a > b for a, b in zip(sigmas[1:-1], other[1:-1])),
          list(zip(sigmas[:4], other[:4])))
    shifted = schedules.flow_sigmas_for_steps(steps, shift=3.0)
    check("②⁴ 3.1 与 Z-Image 的 3.0 也**不同** ✓（差得小但仍不是同一格 ✓）",
          abs(sigmas[1] - shifted[1]) > 1e-6, (sigmas[1], shifted[1]))


# --------------------------------------------------------------- ③ plan


def case_plan() -> None:
    """③ 尺寸吸附 **16** ✓（VAE 8× × DiT patch 2×2 ✓）＋ 单帧 ＋ 默认像素预算被如实换掉 ✓。

    ⚠️ 画布数字跟着**尺寸口径**走 ✗✗：`zimage` / `krea2` 用 ComfyUI ``ResolutionSelector``
    （``√(MP × 1024²)`` ✓），而 **2512 / 2.1 仍走** :func:`geometry.size_for_megapixels`
    （≈``1e6`` 基数 + 向上吸 ✓）—— 官方模板 ``EmptySD3LatentImage`` 就是 **1328×1328** ✓，
    换 selector 会给 1360 ✓、与工作流 `#7.width/height` 对不上 ✓✗。判据认本族口径 ✓，
    并要求回执**写出真数** ✓✗（不许一边写 1328 一边画别的 ✓）。
    """
    plan = _plan(ratio="1:1")
    check("③ 默认像素预算（0.65 MP 的**视频口径** ✗）⇒ 换 2512 官方模板档 ✓ 且**如实告警** ✓"
          "（1:1 正是官方 **1328×1328** ✓，回执点名这个数 ✓）",
          (plan.width, plan.height) == (1328, 1328)
          and any("0.65" in text and "1.76" in text for text in plan.warnings)
          and any("1328×1328" in text for text in plan.warnings),
          ((plan.width, plan.height), plan.warnings[:2]))
    check("③′ 单张图 ⇒ `frames=1` ✓、`fps=0` ✓（不是视频 ✓）",
          plan.frames == 1 and plan.fps == 0, (plan.frames, plan.fps))
    check("③″ `latent_frames` 不被这条线填 ✓（图没有时间维 ✗；填了会诱使后端造 5 维潜变量 ✓✗）",
          plan.latent_frames is None, plan.latent_frames)

    #: ⚠️ 反向：**8 的倍数不够** ✗✗ —— VAE 8× ✓ × DiT patch 2×2 ✓ ⇒ 潜边长必须是**偶数** ✓。
    #: 3:2 的 1.0 MP ⇒ √1.5M = **1225** ⇒ 吸 16 ⇒ **1232**（= 77×16 ✓，潜边 154 = 偶数 ✓）——
    #: 只吸到 8 会得到 1224（潜边 153 = 奇数 ✗）⇒ `img_in` 拍平当场形状错 ✓。
    odd = _plan(megapixels=1.0, ratio="3:2")
    check("③‴ 宽高都是 **16** 的倍数 ✓ 且**潜边长是偶数** ✓✗（只吸 8 ⇒ 奇数潜边 ⇒ `img_in` 拍平爆 ✓）",
          odd.width % 16 == 0 and odd.height % 16 == 0
          and (odd.width // 8) % 2 == 0 and (odd.height // 8) % 2 == 0,
          (odd.width, odd.height, odd.width // 8, odd.height // 8))

    steps = _plan(steps=50, megapixels=1.0, ratio="1:1")
    check("③⁴ `timesteps` **就是 σ** ✓（2512 的 multiplier=1.0 ✓，不乘 1000 ✗、也不翻转 ✗）",
          [round(float(v), 9) for v in steps.timesteps]
          == [round(float(v), 9) for v in steps.sigmas]
          and abs(steps.sigmas[1] - 3.1 * (1 - 1 / 50) / (1 + (3.1 - 1) * (1 - 1 / 50))) < 1e-9,
          steps.sigmas[:3])
    check("③⁵ 计划里的警告写着 `shift=3.1` 并点名**不是** 0.69 ✓（口误要看得见 ✓）",
          any("shift=3.1" in text and "0.69" in text for text in steps.warnings),
          [text for text in steps.warnings if "shift" in text])

    quiet = _plan(steps=50, megapixels=1.0, ratio="1:1")
    check("③⁶ cfg=1.0 ⇒ **只报不改** ✓ 且报出官方档（2512 不是 cfg=1 蒸馏的 ✗）",
          any("cfg=1.0" in text and "4.0" in text and "50" in text for text in quiet.warnings),
          quiet.warnings[-1:])


# --------------------------------------------------------------- ④ 装配门禁


def case_runtime_refuses() -> None:
    """④ 四格缺项**当场拒** ✓、只点名真缺的 ✓、换组件必须换装载键 ✓。

    ⚠️ 这条**不跑真装载** ✗（≈54.4 GiB ✓）：验的是门口 ✓。
    """
    from app.services.engine import runtime as rt  # noqa: PLC0415

    call = lambda **kw: rt.EngineRuntime()._ensure_loaded_internal(  # noqa: E731
        stage=pipeline.QWEN_IMAGE_STAGE, **kw)

    message = _raises(call)
    check("④ `stage=qwen_image` 什么都不给 ⇒ **当场报** ✓✗（不静默当 2.1 装 ✓、不半装 ✓）",
          _underlying_type(call) is rt.TorchBackendUnavailable, message)
    check("④′ 拒的理由把**四格**都点名 ✓（`ditPath` / `textWeights` / `imageVaeWeights` / `tokenizerPath`）",
          all(key in (message or "") for key in
              ("ditPath", "textWeights", "imageVaeWeights", "tokenizerPath")), message)
    check("④″ 报错里点了**本机真文件长什么样** ✓（1933 键 / Qwen2.5 / 16 通道 ✓）——"
          "照着报错就能补齐 ✓",
          all(text in (message or "") for text in ("1933", "Qwen2.5", "16")), message)
    check("④‴ 报错顺手点破**别拿 2.1 的件** ✓（两族件件不同 ✓，配错是「装得进去、图是糊的」✗）",
          "2.1" in (message or ""), message)

    partial = _raises(lambda: call(dit_path="d.safetensors", text_weights="t.safetensors",
                                   tokenizer_path="tok/"))
    #: ⚠️ 只看**首行** ✗✗：第二行起是 `_QWEN_IMAGE_STAGE_NOTE` 的五份说明 ✓（有意列的 ✓）。
    head = (partial or "").split("\n", 1)[0]
    check("④⁴ 给了三格、只缺 `imageVaeWeights` ⇒ **缺项清单里只有它** ✓（说明段列四格不算 ✗）",
          "imageVaeWeights" in head and "textWeights" not in head, head)

    base = dict(stage=pipeline.QWEN_IMAGE_STAGE, dit_path="d.safetensors", tokenizer_path="tok/")
    k_a = rt._load_key(**base, text_weights="te_a.safetensors", image_vae_weights="vae_a.safetensors")
    k_te = rt._load_key(**base, text_weights="te_b.safetensors", image_vae_weights="vae_a.safetensors")
    k_vae = rt._load_key(**base, text_weights="te_a.safetensors", image_vae_weights="vae_b.safetensors")
    check("④⁵ 换 `textWeights` ⇒ 装载键变 ✓（不复用上一套 2.1 的 9B ✗）", k_a != k_te, (k_a, k_te))
    check("④⁶ 换 `imageVaeWeights` ⇒ 装载键变 ✓（不复用上一套 64 通道 VAE ✗）",
          k_a != k_vae, (k_a, k_vae))


# --------------------------------------------------------------- ⑤ 就绪与登记


def case_stage_registered() -> None:
    """⑤ 就绪档**只列这条链的三件** ✗、三件全算必需 ✓、与 `plan_stage` 同一份条目 ✓。

    ⚠️ **不许断言 `ready`** ✗✗：那是"本机三份权重在不在"，换机必红 ✓。
    """
    from app.services.engine import inventory as inv  # noqa: PLC0415
    from app.services.engine import loader, registry  # noqa: PLC0415

    check("⑤ `STAGE_FILTERS` 与 `registry.STAGES` 双向对齐 ✓（少一边就别处会红 ✓）",
          pipeline.QWEN_IMAGE_STAGE in inv.STAGE_FILTERS
          and pipeline.QWEN_IMAGE_STAGE in registry.STAGES
          and inv.STAGE_FILTERS[pipeline.QWEN_IMAGE_STAGE]["category"]
          == registry.STAGES[pipeline.QWEN_IMAGE_STAGE].category,
          {"filter": inv.STAGE_FILTERS.get(pipeline.QWEN_IMAGE_STAGE),
           "registry": getattr(registry.STAGES.get(pipeline.QWEN_IMAGE_STAGE), "category", None)})

    report = inv.readiness(pipeline.QWEN_IMAGE_STAGE)
    keys = sorted(row["key"] for row in report["components"])
    check("⑤′ `readiness(\"qwen_image\")` **只列这条链的三件** ✓✗"
          "（把 SDXL / Z-Image / 2.1 的组件混进来 ⇒ 24 GB 卡会凭空报装不下 ✓✗）",
          keys == ["dit_qwen_image_2512", "te_qwen25_vl_7b", "vae_qwen_image"], keys)
    check("⑤″ 三件在这个档里**全部算必需** ✓（`requireAll` 生效 ✓；`required: false` 不是「不需要」✗）",
          all(row["required"] for row in report["components"]),
          [(row["key"], row["required"]) for row in report["components"]])
    #: 38.05 + 8.74 + 0.24 = 47.03 GiB ✓ —— **单是 DiT 就超卡** ✓✗（22.49 ✓）。
    check("⑤‴ 必需权重合计 ≈ **47.03 GiB** ✓（这一档注定装不满卡 ✓ 是必付成本、不是「已优化」✗）",
          abs(float(report["requiredWeightsGiB"]) - 47.03) < 0.05,
          report["requiredWeightsGiB"])

    #: ⚠️ **Edit 2511 故意不在这一档** ✗✗：它与 t2i 互斥 ✓，进 `qwen2512` tag 会让 `requireAll`
    #: 把两代都算成必需 ⇒ 就绪档凭空假红（还多报 17.78 GiB ✓）。
    check("⑤⁴ Edit 2511 **不在**必需集合里 ✓✗（两代互斥；进 tag 会让就绪档假红 ✓）",
          not any("edit" in key for key in keys), keys)

    plan = loader.plan_stage(pipeline.QWEN_IMAGE_STAGE)
    plan_keys = sorted(row["key"] for row in plan["components"])
    check("⑤⁵ `loader.plan_stage` 与就绪档**同一份条目** ✓✗",
          plan_keys == keys, {"plan": plan_keys, "readiness": keys})
    required_keys = sorted(row["key"] for row in plan["components"] if row.get("required"))
    check("⑤⁶ `plan_stage` 的必需集合**非空** ✓✗（空集合 ⇒ `ready = all([]) = True`，文件全缺也说就绪 ✓）",
          required_keys == keys, required_keys)

    #: ⚠️ 反向：`qwen_image21` 档**不许**把这 47.03 GiB 算进必需 ✗✗ —— 两套是**互斥备选** ✓。
    other = inv.readiness(pipeline.QWEN21_STAGE)
    check("⑤⁷ `readiness(\"qwen_image21\")` 的必需集合**不含** 2512 三件 ✓✗（互斥备选不许相加 ✓）",
          not any(row["key"] in keys and row["required"] for row in other["components"]),
          [(row["key"], row["required"]) for row in other["components"]])


# --------------------------------------------- ⑥ 分时驻留（三个按族参数 ✓）


def case_phased() -> None:
    """⑥ 三件 bf16 驻留 **≈54.4 GiB > 22.49 GiB** ✗✗ ⇒ 必须按阶段互斥换装 ✓，
    且**三个按族参数**（`text_kind` / `vae_family` / `part_need_gib`）真被用上 ✓✗。

    ⚠️ 这条**不装真权重** ✗：把装载方法换成记账的 ✓，验「装哪件、按哪张账、接哪座塔」✓
    —— 真装与出图是人为验收那一步 ✓。
    """
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    class Fake(tb.TorchBackend):        # noqa: D401 —— 只覆盖装载，不碰盘 ✓
        def __init__(self) -> None:     # noqa: D107
            super().__init__(device="cpu")
            self.events: list[tuple[str, str]] = []
            self._chain = None

        def load_weights(self, *, path=None, **_kw):     # noqa: ANN001, ANN003, D102
            self.events.append(("load", "dit"))
            self._model = object()
            return {"path": path}

        def attach_qwen25_te(self, weights, *, tokenizer_path=None, **_kw):  # noqa: ANN001, ANN003, D102
            self.events.append(("load", "text-qwen25"))
            self._textEncoder = object()
            return {"weights": weights, "tokenizer": tokenizer_path}

        def attach_qwen35_te(self, weights, *, tokenizer_path=None, **_kw):  # noqa: ANN001, ANN003, D102
            #: ⚠️ 两座塔必须记成**可辨的名字** ✗✗：同名记账的话，「接错族」这条
            #: （真机上报「缺 `linear_attn.*`」）在自检里永远是绿的 ✓✗。
            self.events.append(("load", "text-qwen35"))
            self._textEncoder = object()
            return {"weights": weights, "tokenizer": tokenizer_path}

        def attach_wan_vae(self, weights, *, family="qwen_image21", **_kw):  # noqa: ANN001, ANN003, D102
            self.events.append(("load", f"vae-{family}"))
            self._vae = object()
            self._vaeKind = "wan21-real"
            return {"weights": weights, "family": family}

    backend = Fake()
    chain = backend.declare_chain(
        dit_path="d.safetensors", text_weights="t.safetensors",
        tokenizer_path="tok/", vae_path="v.safetensors",
        vae_family="qwen_image", text_kind="qwen25",
        part_need_gib=tb.TorchBackend.QWEN_IMAGE_PART_NEED_GIB)
    #: ⭐ 2026-11-01 起这本账多一格 ``vision`` ✓（Edit 2511 的视觉塔 1.35 GiB ✓）——
    #: 断言只比**三件主件** ✓✗：写成整字典等值会把"以后再加一格"变成假红 ✓。
    check("⑥ 登记回执带**本族的账** ✓✗（缺省是 2.1 的 13.26 ⇒ 判「装得下」⇒ 装到一半 OOM ✓）",
          {key: chain["partNeedGiB"].get(key) for key in ("dit", "text", "vae")}
          == {"dit": 38.05, "text": 16.1, "vae": 0.24}
          and abs(float(chain["partNeedGiB"]["vision"]) - 1.35) < 1e-6,
          chain["partNeedGiB"])
    check("⑥′ `peakGiB` 由**本族的账**算 ✓（写死 14.79 会让读者按 2.1 的峰值估余量 ✗）",
          abs(float(chain["peakGiB"]) - 38.05) < 1e-6, chain["peakGiB"])
    check("⑥″ 链里记着**两座塔怎么分** ✓（`textKind` 缺省 qwen35 ⇒ 2512 会被 Qwen3.5 骨架吃 ✓✗）",
          chain["chain"]["textKind"] == "qwen25"
          and chain["chain"]["vaeFamily"] == "qwen_image", chain["chain"])

    backend.enter_phase("encode")
    check("⑥‴ `encode` ⇒ **只装 TE、且接的是 Qwen2.5 塔** ✓✗（接成 Qwen3.5 ⇒ 报「缺 `linear_attn.*`」✓）",
          backend.events == [("load", "text-qwen25")]
          and backend._textEncoder is not None and backend._model is None, backend.events)

    backend.enter_phase("sample")
    check("⑥⁴ `sample` ⇒ 退掉 TE、只装 DiT ✓（TE 还在就是 54 GiB 那条路 ✓✗）",
          backend._textEncoder is None and backend._model is not None
          and backend.events[-1] == ("load", "dit"), backend.events)

    backend.enter_phase("decode")
    check("⑥⁵ `decode` ⇒ 只装 VAE，且 family 是 **qwen_image** ✓✗（给 `wan21` 会吃 Wan 的 "
          "16 个 mean/std ⇒ 图偏色 ✓✗；给 2.1 的 `qwen_image21` ⇒ 按 64 通道查 16 通道潜变量 ⇒ 当场报 ✓）",
          backend._model is None and backend._vae is not None
          and backend.events[-1] == ("load", "vae-qwen_image"), backend.events)


# --------------------------------------------- ⑦ 两本账不许混（38.05 vs 13.26）


def case_two_accounts() -> None:
    """⑦ **按族的账**必须真进决策 ✓✗ —— 拿 2.1 的 13.26 判 2512 的 38.05 是「静默 OOM」✓✗。

    ⚠️ 用**纯函数** `device_policy.decide` + 假体注入判（不读真卡 ✓ ⇒ 换机不红 ✓）：
    同一份卡况（22.49 GiB 总量、余 22.0 ✓）下，13.26 判「上卡不降级」✓、
    38.05 必须判「转 CPU」✓✗ —— 这正是两张表差 24.8 GiB 的**后果** ✓。
    """
    from app.services.engine import device_policy as dp  # noqa: PLC0415
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    small = tb.TorchBackend.PART_NEED_GIB["dit"]
    big = tb.TorchBackend.QWEN_IMAGE_PART_NEED_GIB["dit"]
    check("⑦ 两张账**必须分开** ✓✗（DiT 13.26 vs **38.05** ⇒ 差 24.8 GiB ✓）",
          abs(big - small) > 20 and big > 22.49, {"2.1": small, "2512": big})

    def _fake(total_gib: float, free_gib: float) -> dp.Pressure:  # noqa: ANN202
        return dp.Pressure(totalBytes=int(total_gib * dp.GIB), freeBytes=int(free_gib * dp.GIB))

    pressure = _fake(22.49, 22.0)
    by_21 = dp.decide(need_bytes=int(small * dp.GIB), pressure=pressure)
    by_2512 = dp.decide(need_bytes=int(big * dp.GIB), pressure=pressure)
    check("⑦′ 用 2.1 的 13.26 判 ⇒ **整件上卡、不降级**（`residency=off` ✓ 即「不用切」✓；"
          "对 2512 就是这个假结论 ✓✗）",
          by_21.device == "cuda" and by_21.residency == "off" and not by_21.degraded,
          by_21.to_dict())
    #: ⚠️ 38.05 GiB 在余 22.0 GiB 上**放不下整件** ✓，但余量 ≫ 0.5 GiB ⇒ 按三档规则走
    #: **联合**（卡上放能放的层、剩下的在 CPU ✓，`degraded=True` ✓）。
    #: ⇒ 判据是「**不许报 `off`**」✗✗：报 off 就等于相信"装得下"⇒ 装到一半 OOM ✓✗。
    check("⑦″ 用 2512 的 38.05 判 ⇒ **不许整件上卡**（必须 `split` 且 `degraded` ✓✗）",
          by_2512.residency == "split" and by_2512.degraded,
          by_2512.to_dict())
    tight = dp.decide(need_bytes=int(big * dp.GIB), pressure=_fake(22.49, 0.36))
    check("⑦‴ 余量连 0.5 GiB 都没有 ⇒ **整份转 CPU** ✓（连联合都拼不起来时只有这一档 ✓）",
          tight.device == "cpu" and tight.degraded, tight.to_dict())

    #: ⚠️ 接线：`decide_for(need_gib=)` 必须**真把这张表读进去** ✗✗ —— 只在常量里写对、
    #: 决策时却用缺省表，等于白写 ✓（38.05 变 13.26，结论又成「装得下」✓✗）。
    backend = tb.TorchBackend(device="cpu")
    tuned = backend.decide_for(("dit",), phase="assemble",
                               need_gib=tb.TorchBackend.QWEN_IMAGE_PART_NEED_GIB).to_dict()
    default = backend.decide_for(("dit",), phase="assemble").to_dict()
    check("⑦⁴ `decide_for(need_gib=…)` **真按本族账算** ✓✗（缺省表仍报 13.26 ✓ 两格都不许 None ✓）",
          tuned.get("needGiB") == 38.05 and default.get("needGiB") == 13.26,
          {"tuned": tuned.get("needGiB"), "default": default.get("needGiB")})

    #: ⚠️ **可见面也得跟着换** ✗✗（2026-10-31 修 ✓）：`describe().devicePolicy.partNeedGiB`
    #: 原来一律报类属性 13.26 ✓ ⇒ 2512 装到一半 OOM 时，回执上写着"每件最多 13.26 GiB、
    #: 22.49 的卡还剩 9 GiB"✓✗（读数与决策用的表差 24.8 GiB ✓ 正是静默 OOM 的读数版 ✓）。
    plain = tb.TorchBackend(device="cpu").describe()["devicePolicy"]
    check("⑦⁵ 没声明链 ⇒ `partNeedGiB` 就是类缺省（13.26 ✓，且 `_chain` 为 None 也不炸 ✓）",
          plain.get("partNeedGiB", {}).get("dit") == 13.26
          and plain.get("defaultPartNeedGiB", {}).get("dit") == 13.26, plain.get("partNeedGiB"))
    phased = tb.TorchBackend(device="cpu")
    phased.declare_chain(dit_path="d", text_weights="t", tokenizer_path="k", vae_path="v",
                         vae_family="qwen_image", text_kind="qwen25",
                         part_need_gib=tb.TorchBackend.QWEN_IMAGE_PART_NEED_GIB)
    seen = phased.describe()["devicePolicy"]
    #: ⚠️ 能力表（`capabilities()` ✓，零副作用 ✓）也得有这一本 ✗✗：并进 `part` 不行 ✓
    #: （两族键名相同 ⇒ 互相覆盖 ✓），所以单开 `qwenImagePart` ✓。
    from app.services.engine import facade  # noqa: PLC0415
    cap = facade.capabilities()["devicePolicy"]["needGiB"]
    check("⑦⁵′ 能力表里 `needGiB.qwenImagePart.dit` = **38.05** ✓✗，且与 `part.dit` 13.26 分开报 ✓",
          abs(float(cap.get("qwenImagePart", {}).get("dit") or 0) - 38.05) < 0.01
          and abs(float(cap.get("part", {}).get("dit") or 0) - 13.26) < 0.01, cap)
    check("⑦⁶ 声明链后 `partNeedGiB` 报**生效账 38.05** ✓✗、`defaultPartNeedGiB` 仍留 13.26 ✓"
          "（两格分开 ⇒ 读者不会拿 2.1 的数估余量 ✓）",
          seen.get("partNeedGiB", {}).get("dit") == 38.05
          and seen.get("defaultPartNeedGiB", {}).get("dit") == 13.26,
          (seen.get("partNeedGiB"), seen.get("defaultPartNeedGiB")))


# --------------------------------------------- ⑧ VAE 反缩放与 Edit 拒跑


def case_vae_and_edit() -> None:
    """⑧ 2512 的 VAE 反缩放是**恒等**（族事实 ✓）、Edit 权重在**缺 `ref_latents` 时拒跑** ✓✗。"""
    from app.services.engine import torch_backend as tb  # noqa: PLC0415
    from app.services.engine import vae_wan  # noqa: PLC0415

    mean, std = vae_wan.latents_stats("qwen_image", z_dim=16)
    check("⑧ 2512 的反缩放是**恒等** ✓（16 个 0 / 16 个 1 ✓ —— `LatentFormat` 基类事实 ✓）",
          len(mean) == 16 and len(std) == 16
          and all(float(v) == 0.0 for v in mean) and all(float(v) == 1.0 for v in std),
          (mean[:3], std[:3]))
    #: ⚠️ 这一条是 2026-10-31 修的**真缺陷** ✗✗：``wan21`` 原先登记成 0/1，理由是"参考里
    #: ``Wan21`` 没覆写 ``scale_factor``"—— 那句把 ``scale_factor``(1.0 ✓) 与 ``latents_mean/std``
    #: （**另一组数** ✓）混成了一件事 ✓✗。改真值之后，2512 若还借 ``wan21`` 这个名字，
    #: 就会吃到 Wan 的 mean/std ⇒ 图整体偏绿、不报错 ✓✗（故 2512 已改用 ``qwen_image`` ✓）。
    wan_mean, wan_std = vae_wan.latents_stats("wan21", z_dim=16)
    check("⑧ᵃ `wan21` 有**真** mean/std ✓✗（不是 16 个 0/1：参考 ``Wan21.__init__`` 逐字 -0.7571… ✓）",
          abs(wan_mean[0] + 0.7571) < 1e-6 and abs(wan_std[0] - 2.8184) < 1e-6
          and wan_mean != mean, (wan_mean[:3], wan_std[:3]))
    check("⑧′ 2512 用的 `qwen_image` 与 `wan21` **不是同一组数** ✓✗（借名 ⇒ 图偏色、不报错 ✓✗）",
          vae_wan.latents_stats("qwen_image", z_dim=16) != (wan_mean, wan_std)
          and vae_wan.latents_stats("qwen_image_2512", z_dim=16) == (mean, std), "")
    wrong_dim = _error_of(lambda: vae_wan.latents_stats("qwen_image", z_dim=64))
    check("⑧″ 通道对不上 ⇒ **当场拒** ✓✗（拿 2.1 的 64 通道口径套 16 通道 VAE ⇒ 不是同一族 ✓）",
          "64" in wrong_dim and "16" in wrong_dim, wrong_dim)
    no_kit = _error_of(lambda: vae_wan.latents_stats("no_such_family"))
    check("⑧‴ 没登记的族 ⇒ **报错** ✓✗（回退成 (0, 1) ⇒ 解码照跑、图发灰、不报错 ✓✗）",
          "没有" in no_kit, no_kit)

    check("⑧⁴ `qwen_image` 在**只出图**族里 ✓✗（漏了它 ⇒ 潜变量按 5 维造、`denoise` 送错形态 ✓）",
          tb.QWEN_IMAGE_FAMILY in tb.IMAGE_ONLY_FAMILIES, tb.IMAGE_ONLY_FAMILIES)

    #: ⚠️ 结构性断言：Edit 的**拒跑必须在向前之前** ✗✗ —— 写在 ``prediction = self._model(``
    #: 之后就等于「先按 t2i 画完再报错」✓✗（能力凭空消失、还白算一遍 ✓）。
    #: ⭐ 2026-11-01 起拒跑的门槛从「参考图通路没接」变成「`ref_latents` 不在」✓
    #: （视觉塔与 VAE 都已接上 ✓，见 `engine_qwen_image_edit_test` ✓）—— 但**必须在向前之前**不变 ✓✗。
    source = inspect.getsource(tb.TorchBackend.denoise)
    guard = source.find("ref_timestep_zero")
    forward = source.find("prediction = self._model(")
    check("⑧⁵ Edit 的拒跑**在 `self._model(` 之前** ✓✗（放到后面 = 白算一遍再报 ✓）",
          guard > 0 and forward > 0 and guard < forward, {"guard": guard, "forward": forward})
    check("⑧⁶ 拒跑的文案点名**上游三处** ✓（`encode_text` / `referenceFrames` / `_refLatents` ✓ "
          "—— 光说「没参考图」读者仍不知道去哪修 ✓）",
          all(word in source for word in ("encode_text", "referenceFrames", "_refLatents")), "")

    check("⑧⁷ 采样时 `ref_latents` 走**关键字**、且由 `list(...)` 包成**列表** ✓✗"
          "（`Sequence[Tensor]` ⇒ 传裸张量会按 batch 遍历 ✓）；**多图**逐条搬 ✓",
          "ref_latents=list(edit_refs)" in source
          and "for item in stored" in source, "")
    check("⑧⁸ `_refLatents` 统一成**列表** ✓✗（单图存裸张量、多图存列表那种写法 ⇒ "
          "`denoise` 要分叉、且「换回 t2i 没清空」更难发现 ✓）",
          "list(edit[\"latents\"])" in inspect.getsource(tb.TorchBackend.encode_text), "")


def _tiny_shape(**kwargs) -> Any:
    """一个**只过 dataclass 校验**的最小形状 ✓（不建模型、不跑前向 ✓）。"""
    from app.services.engine import dit_qwen_image as qi  # noqa: PLC0415
    base = dict(features=128, layers=1, heads=2, head_dim=64, hidden=256, channels=16,
                context_dim=32, axes_dim=(16, 24, 24))
    base.update(kwargs)
    return qi.QwenImageShape(**base)


def case_ref_methods() -> None:
    """⑨ 参考图口径：三种齐全 ✓、第四种显式拒 ✓、`ref_index_scale` 不再只存不用 ✓（2026-10-31）。"""
    from app.services.engine import dit_qwen_image as qi  # noqa: PLC0415

    check("⑨ `negative_index` 进口径表 ✓（参考实现四种之一；缺它 ⇒ 这类权重只能靠报错挡 ✗）",
          qi.REF_NEGATIVE in qi.REF_METHODS and len(qi.REF_METHODS) == 3, qi.REF_METHODS)
    shape = _tiny_shape(ref_method=qi.REF_NEGATIVE)
    check("⑨′ `negative_index` **不**走 t=0 那一行调制 ✓（只有 `index_timestep_zero` 走 ✓✗）",
          shape.ref_timestep_zero is False, shape.ref_method)
    offset = _error_of(lambda: _tiny_shape(ref_method="offset"))
    check("⑨″ 参考实现的第四种口径 `offset` ⇒ **当场拒** ✓✗（按 index 猜 = 出图像但不对 ✗）",
          "ref_method" in offset, offset)
    source = inspect.getsource(qi.QwenImageDiT.forward)
    check("⑨‴ `ref_index_scale` **真的乘进位置号** ✓（只存不用 ⇒ 特殊权重悄悄错 ✗✗）",
          "ref_index_scale" in source, "")
    check("⑨⁴ 序号**方向**随口径走 ✓（`negative_index` ⇒ -1、-2… ✓）",
          "REF_NEGATIVE" in source, "")


def case_per_file_need() -> None:
    """⑩ 需求数按**文件精度**取 ✓✗（38.05 的 bf16 与 19.09 的 int8 是两份账 ✓）。

    ## 这一条抓的那个真缺陷（2026-11-01 ✓）

    装配线原先把 ``QWEN_IMAGE_PART_NEED_GIB["dit"] = 38.05`` **无条件**当需求 ✓✗ ——
    对 bf16 的 2512 完全正确 ✓，但对 **int8 的 Edit 2511** 多报 19 GiB ✓✗ ⇒ `decide()` 判
    `split` ⇒ 实测 ``gpu 15.9 / cpu 22.15 GiB`` ⇒ **58% 的权重在内存里逐层爬** ✓✗，
    而卡其实整份装得下（19.09 < 19.71 可用 ✓）。判据没错 ✗，错在**需求数** ✓。
    """
    from app.services.engine import device_policy as dp  # noqa: PLC0415
    from app.services.engine import runtime as rt  # noqa: PLC0415
    from app.services.engine import safetensors as st  # noqa: PLC0415
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    fallback = tb.TorchBackend.QWEN_IMAGE_PART_NEED_GIB["dit"]
    gib, why = tb.resident_need_gib("no_such_file.safetensors", fallback=fallback)
    check("⑩ 读不到文件 ⇒ **退回表值 38.05** ✓（不猜、不炸 ✓；且 `int8` 报 None 而不是 False ✓✗"
          "—— `False` 会被读成「查过了，没有 int8」✓）",
          abs(gib - fallback) < 1e-9 and why.get("int8") is None, why)

    class _Entry:
        def __init__(self, dtype: str, shape: tuple[int, ...]) -> None:
            self.dtype, self.shape, self.begin, self.end = dtype, shape, 0, 0

        @property
        def nbytes(self) -> int:
            return 0

        @property
        def element_count(self) -> int:
            total = 1
            for dim in self.shape:
                total *= dim
            return total

    class _Info:
        """一份**假头部** ✓（只喂 `resident_need_gib` 读的那几格 ✓）。"""

        def __init__(self, tensors: dict[str, Any]) -> None:
            self.tensors = tensors
            self.ok = True
            self.problems: list[str] = []
            self.total_tensor_bytes = int(19.09 * dp.GIB)
            self.file_bytes = self.total_tensor_bytes + 2048

    int8_info = _Info({
        "blocks.0.attn.to_q.weight": _Entry("I8", (4, 8)),
        "blocks.0.attn.to_q.weight_scale": _Entry("F16", (4, 1)),
        "blocks.0.norm.weight": _Entry("BF16", (8,)),
    })
    plain_info = _Info({"blocks.0.attn.to_q.weight": _Entry("BF16", (4, 8))})
    real = st.inspect
    try:
        st.inspect = lambda path: int8_info   # noqa: ARG005
        by_int8, why_int8 = tb.resident_need_gib("x.safetensors", fallback=fallback)
        st.inspect = lambda path: plain_info   # noqa: ARG005
        by_plain, why_plain = tb.resident_need_gib("x.safetensors", fallback=fallback)
    finally:
        st.inspect = real

    check("⑩′ 文件侧有 int8 层 ⇒ 驻留取**数据区字节**（19.09 ✓ 不是 38.05 ✓✗），且报出层数 ✓",
          abs(by_int8 - 19.09) < 0.01 and why_int8.get("int8") is True
          and why_int8.get("layers") == 1, why_int8)
    check("⑩″ 没有 int8 层的文件 ⇒ **照旧用表值** ✓✗（`fp8_scaled` 是 2×、bf16 是 1× ✓ —— "
          "在这里编一个「低精度文件统一乘 1」会把 fp8 那份少算一半 ✓✗）",
          abs(by_plain - fallback) < 1e-9 and why_plain.get("int8") is False, why_plain)
    #: ⚠️ 判据连着看才成立 ✓：19.09 那份该判 `off`（整份上卡 ✓）、38.05 那份该判 `split` ✓✗
    pressure = dp.Pressure(totalBytes=int(22.49 * dp.GIB), freeBytes=int(20.71 * dp.GIB))
    small = dp.decide(need_bytes=int(by_int8 * dp.GIB), pressure=pressure)
    big = dp.decide(need_bytes=int(fallback * dp.GIB), pressure=pressure)
    check("⑩‴ 同一张卡：19.09 ⇒ `off`（整份上卡 ✓）/ 38.05 ⇒ `split`（按层切 ✓✗）——"
          "这就是「需求数错 19 GiB」的全部后果 ✓",
          small.residency == "off" and not small.degraded and big.residency == "split", 
          {"int8": small.to_dict(), "bf16": big.to_dict()})

    #: ⚠️ 接线 ✗✗：算对了却只在回执里写、决策仍读原表 ⇒ 白改 ✓（两条调用口都得吃校正后的表 ✓）。
    source = inspect.getsource(rt.EngineRuntime._ensure_qwen_image_loaded)
    check("⑩⁴ 装配线**真的**按文件精度取需求 ✓✗（走**单一出口** `stage_part_need(\"qwen_image\", \"dit\", …)` ✓"
          "—— 2026-11-07 起「哪件按哪档量、挂不挂常驻壳、按不按文件精算」收在 "
          "`torch_backend.STAGE_PART_SPECS` ✓，它再转 `part_need_gib` 涵盖 int8 / int4 / plain 三档 ✓，"
          "旧 `resident_need_gib` 只认 int8 ⇒ 装配线不许再用它 ✗；"
          "**也不许在这条线里手写 assume/adapter** ✗ —— 那等于又开第二本账 ✓✗），"
          "且 `decide_for` 与 `declare_chain` **两处都吃校正后的表** ✓✗"
          "（只改一处 ⇒ 装配报整份上卡、`sample` 段又切 ✓）",
          "stage_part_need(" in source and "resident_need_gib(" not in source
          and '"qwen_image", "dit"' in source
          and "part_need_gib(" not in source
          and "need_gib=part_need)" in source
          and "part_need_gib=part_need)" in source, "")
    check("⑩⁵ 校正只动 `dit` 一格 ✓（TE 16.1 / VAE 0.24 / 视觉塔 1.35 三项一个都不许被顺手改 ✗）",
          "part_need = dict(TorchBackend.QWEN_IMAGE_PART_NEED_GIB)" in source
          and 'part_need["dit"] = float(dit_need_gib)' in source, "")
    check("⑩⁶ 回执里写清「这次按什么数判的」✓✗（只报 `devicePolicy` ⇒ 读者看不出 38.05 还是 19.09 ✓）",
          '"ditNeed"' in source and "dit_need_why" in source, "")


def main() -> int:
    case_stage_names()
    case_sigmas()
    case_plan()
    case_runtime_refuses()
    case_stage_registered()
    case_phased()
    case_two_accounts()
    case_vae_and_edit()
    case_ref_methods()
    case_per_file_need()

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

