"""自检：**Z-Image 的采样口径**（flow matching ✓ 2026-10-31）。

为什么值得单独立一份自检 ✗（明明"只是另一个图模型" ✓）：Z-Image 的 σ 与 SDXL 的 **不是一回事** ✗✗，
而两族在本仓长得很像（都 `stage=*image*` ✓ 都出 4 维潜变量 ✓ 都不报错 ✓）：

============================  ==========================  ==========================
                             SDXL（离散扩散 ✓）            Z-Image（flow matching ✓）
============================  ==========================  ==========================
σ 取值范围                    ``0.0292 … 14.6146`` ✓        **恒在 [0, 1]** ✓
喂给主干的 timestep          ``σ → 0..999`` 的整数格 ✓      **就是 σ 本身** ✓（multiplier=1 ✓）
主干内部的 t                  ——                            ``(1 − σ) × time_scale(1000)`` ✓
============================  ==========================  ==========================

⇒ 拿 karras 那条 σ 喂 Z-Image：``1 − 14.6`` 是个大负数 ✓、**形状全对、不报错** ✓、出来是噪声 ✓✗。
本文件就是钉死这张表 ✓（含**反向验证** ✓：故意混用，必须能被判据抓到 ✓）。

⚠️ 公式来源是**闭式数学** + 参考实现的口径（``supported_models.ZImage`` 的
``multiplier=1.0, shift=3.0`` ✓、``ldm/lumina/model.py`` 的 ``t = 1.0 - timesteps`` ✓）；
**一行代码都不抄** ✗（ComfyUI 是 GPL-3.0 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_zimage_test.py
"""
from __future__ import annotations

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


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001
        return False
    return True


def case_sigmas() -> None:
    """① σ 序列：端点 / 单调 / 长度 / 定值点（手算的 ✓ 不是抄的 ✗）。"""
    sigmas = schedules.flow_sigmas_for_steps(8, shift=3.0)
    check("① 长度 = 步数 + 1 ✓（末尾那个 0 是「去噪到 x0」✓）", len(sigmas) == 9, len(sigmas))
    check("①′ σ₀ = 1 ✓（flow 的 t=1 就是纯噪声 ✓）", sigmas[0] == 1.0, sigmas[0])
    check("①″ σ_last = 0 ✓", sigmas[-1] == 0.0, sigmas[-1])
    check("①‴ 严格单调递减 ✓", all(a > b for a, b in zip(sigmas, sigmas[1:-1])),
          sigmas)
    # 手算：σ(t) = 3t/(1+2t) ⇒ t=0.875 ⇒ 2.625/2.75 = 0.954545…
    want = 3 * 0.875 / (1 + 2 * 0.875)
    check("①⁗ 第 2 个点 = 3·0.875/(1+2·0.875) ≈ 0.9545 ✓（公式写错必红 ✓）",
          abs(sigmas[1] - want) < 1e-9, (sigmas[1], want))
    check("①⁵ shift=1 ⇒ 退化成线性 ✓（`σ = t` ✓ 不动点 ✓）",
          all(abs(s - (1 - i / 8)) < 1e-12 for i, s in
              enumerate(schedules.flow_sigmas_for_steps(8, shift=1.0))),
          schedules.flow_sigmas_for_steps(8, shift=1.0))
    check("①⁶ shift 越大 ⇒ 中间档越**偏噪** ✓（3 > 1 逐点比 ✓）",
          all(a > b for a, b in zip(schedules.flow_sigmas_for_steps(16, shift=3.0)[1:-1],
                                    schedules.flow_sigmas_for_steps(16, shift=1.0)[1:-1])))
    check("①⁷ shift ≤ 0 ⇒ **报错** ✗（不是静默出 NaN ✓）",
          _raises(lambda: schedules.flow_sigmas_for_steps(8, shift=0.0)) is not None)


def case_family_boundary() -> None:
    """② ⭐ 两族不可混用 ✓ —— 这条是**反向验证** ✓（能抓到才算判据 ✓）。"""
    flow = schedules.flow_sigmas_for_steps(8, shift=3.0)
    karras = schedules.sigmas_for(8, "karras")
    check("② ⭐ flow 的 σ **全部 ≤ 1** ✓✗（这是它的定义域 ✓ 不是巧合 ✓）",
          all(0.0 <= s <= 1.0 for s in flow), max(flow))
    check("②′ karras 的 σ **超过 1** ✓（最大 14.6 那档 ✓ ⇒ 两族定义域不重叠 ✓）",
          max(karras) > 1.0, max(karras))
    # ⚠️ 这就是"混用会怎样"的**算术**：Z-Image 主干算 `(1 − timestep) × 1000` ✓
    wrong = (1.0 - max(karras)) * 1000.0
    right = (1.0 - max(flow)) * 1000.0
    check("②″ 拿 karras 喂 Z-Image ⇒ 主干收到一个**大负数** ✓✗（形状对、不报错、出噪声 ✓）",
          wrong < -1000.0 < 0.0 <= right <= 1000.0, (wrong, right))
    # ⚠️ 主干是 `t = (1 − timestep) × 1000` ✓ ⇒ timestep 若误乘 1000 ⇒ `(1 − σ·1000)·1000` ✓。
    #    取 `min` 而不是只比最后一项 ✓：σ₀=1 ⇒ −999000 ✓（**全体**都在很负那一侧 ✓），
    #    只有末尾 σ=0 那一项是 +1000 ✓（无噪步，本来就该是 1.0 倍 ✓ 所以不看它 ✓）。
    off_by_1000 = [(1.0 - s * 1000.0) * 1000.0 for s in flow]
    check("②‴ 顺手钉住 timestep 的**另一处**误解：不是 `σ × 1000` ✗"
          "（multiplier=1 ✓ ⇒ 就是 σ ✓；乘 1000 后主干算出的 t 会到 −1e6 量级 ✓）",
          min(off_by_1000) < -1e5, off_by_1000[:3])


def case_plan() -> None:
    """③ plan 口径：16 对齐 / 单帧 / 无 fps / timestep 就是 σ / 与模板差异如实报 ✓。"""
    request = pipeline.GenerationRequest(prompt="a red fox in snow", seed=42, steps=8,
                                         ratio=(1, 1), megapixels=1.0,
                                         stage=pipeline.ZIMAGE_STAGE)
    plan = pipeline.build_plan(request)
    check("③ 1024×1024 ✓（官方 turbo 模板的画布 ✓）", (plan.width, plan.height) == (1024, 1024),
          (plan.width, plan.height))
    check("③′ 单帧 ✓ / 没有帧率 ✓（图不是片 ✓ 不拿 24/30 冒充 ✗）",
          plan.frames == 1 and plan.fps == 0, (plan.frames, plan.fps))
    check("③″ ⭐ `timesteps` **就是 σ** ✓✗（不是 `σ→0..999` 的整数格 ✓、也不乘 1000 ✓）",
          list(plan.timesteps) == list(plan.sigmas), (plan.timesteps[:3], plan.sigmas[:3]))
    check("③‴ σ 是 flow 那套 ✓（首 1 / 末 0 / 全 ≤1 ✓）",
          plan.sigmas[0] == 1.0 and plan.sigmas[-1] == 0.0
          and all(0.0 <= s <= 1.0 for s in plan.sigmas), plan.sigmas)

    odd = pipeline.GenerationRequest(prompt="x", seed=1, steps=8, ratio=(1, 1),
                                     megapixels=1.0, stage=pipeline.ZIMAGE_STAGE)
    odd_plan = pipeline.build_plan(odd)
    check("③⁗ 边长恒是 **16** 的倍数 ✓（VAE 8 × patch 2 ✓ 不整除会在 patchify 前被裁 ✓✗）",
          odd_plan.width % 16 == 0 and odd_plan.height % 16 == 0,
          (odd_plan.width, odd_plan.height))

    default_mp = pipeline.GenerationRequest(prompt="x", seed=1, steps=8, ratio=(1, 1),
                                            stage=pipeline.ZIMAGE_STAGE)
    default_plan = pipeline.build_plan(default_mp)
    check("③⁵ 默认 0.65 MP（视频口径 ✗）⇒ 已换成 1024² 且**如实报改** ✓（不静默 ✓）",
          (default_plan.width, default_plan.height) == (1024, 1024)
          and any("0.65" in text for text in default_plan.warnings),
          default_plan.warnings)

    off = pipeline.GenerationRequest(prompt="x", seed=1, steps=20, ratio=(1, 1), megapixels=1.0,
                                     stage=pipeline.ZIMAGE_STAGE)
    check("③⁶ steps≠8 / cfg≠1 ⇒ **只报不改** ✓（蒸馏模型别的档位能跑 ✓ 但不静默替用户改 ✗）",
          pipeline.build_plan(off).steps == 20
          and any("turbo" in text for text in pipeline.build_plan(off).warnings))


def case_stage_names() -> None:
    """④ 阶段名 ✓：Z-Image 与 SDXL **必须**是两个名字（否则后端/σ/VAE 三处都会串 ✗✗）。"""
    check("④ `ZIMAGE_STAGE != IMAGE_STAGE` ✓✗（并成一个 ⇒ σ 语义串 ✓ 且后端也不是同一个 ✓）",
          pipeline.ZIMAGE_STAGE != pipeline.IMAGE_STAGE,
          (pipeline.ZIMAGE_STAGE, pipeline.IMAGE_STAGE))
    check("④′ 未知 stage ⇒ 报错里**三个名字都列出来** ✓（别让调用方猜 ✓）",
          pipeline.ZIMAGE_STAGE in str(_error_of(lambda: pipeline.build_plan(
              pipeline.GenerationRequest(prompt="x", stage="nope")))),
          _error_of(lambda: pipeline.build_plan(
              pipeline.GenerationRequest(prompt="x", stage="nope"))))


def case_runtime_refuses() -> None:
    """⑥ 装配口 ✓：缺格必拒 ✗、说清缺哪个 ✓、装载键必须认这两格 ✓。

    ⚠️ 这条**不跑真装载** ✗（三份权重 ≈19.3 GiB ✓，自检里不装 ✓）—— 它验的是**门口**：
    缺格拒得干脆 ✓、拒的话能照着补 ✓、以及"换 TE 就是换模型"进不进键 ✓。
    """
    from app.services.engine import runtime as rt  # noqa: PLC0415

    call = lambda **kw: rt.EngineRuntime()._ensure_loaded_internal(  # noqa: E731
        stage=pipeline.ZIMAGE_STAGE, **kw)

    message = _raises(call)
    check("⑥ `stage=zimage` 什么都不给 ⇒ **当场报** ✓✗（不静默当 H3 装 ✓、不半装 ✓）",
          _underlying_type(call) is rt.TorchBackendUnavailable, message)
    check("⑥′ 拒的理由把**四格**都点名 ✓（`ditPath` / `textWeights` / `imageVaeWeights` / `tokenizerPath`）"
          "—— 用户照着报错就能补齐 ✓，不用回来问 ✓",
          all(k in (message or "") for k in ("ditPath", "textWeights", "imageVaeWeights", "tokenizerPath")),
          message)
    #: ⚠️ 「缺一格」与「什么都没给」必须**分得开** ✗✗：只看"有没有报错"会把
    #: 「给了三格、少一格」也当成"没实现"✓✗ —— 那正是 2026-10-31 之前那句
    #: 「stage=zimage 走不了装配口」的来源 ✓，实际只是**少一格** ✓。
    partial = _raises(lambda: call(dit_path="d.safetensors", text_weights="t.safetensors",
                                   tokenizer_path="tok/"))
    #: ⚠️ 只看**首行** ✗✗：第二行起是 `_ZIMAGE_STAGE_NOTE` 的**四格说明** ✓（那是有意列的 ✓）——
    #: 拿整段找 "textWeights" 会把这条说明误判成"缺项"✓✗（实测踩过一次 ✓）。
    head = (partial or "").split("\n", 1)[0]
    check("⑥″ 给了三格、只缺 `imageVaeWeights` ⇒ **缺项清单里只有它** ✓（说明段仍列四格 ✓ 那不算缺项 ✗）",
          "imageVaeWeights" in head and "textWeights" not in head, head)

    #: ⚠️ 装载键：**换 TE / 换 VAE 就是换了一个模型** ✓ —— 不进键 ⇒ 给了新的 qwen_3_4b 却复用上一套 ✓✗，
    #: 而且一声不响 ✓✗（同 2026-10-29 `int8_offload` 那次漂移 ✓）。
    base = dict(stage=pipeline.ZIMAGE_STAGE, dit_path="d.safetensors", tokenizer_path="tok/")
    k_a = rt._load_key(**base, text_weights="te_a.safetensors", image_vae_weights="vae_a.safetensors")
    k_te = rt._load_key(**base, text_weights="te_b.safetensors", image_vae_weights="vae_a.safetensors")
    k_vae = rt._load_key(**base, text_weights="te_a.safetensors", image_vae_weights="vae_b.safetensors")
    check("⑥‴ 换 `textWeights` ⇒ 装载键变 ✓（不复用上一套 TE ✗）", k_a != k_te, (k_a, k_te))
    check("⑥⁗ 换 `imageVaeWeights` ⇒ 装载键变 ✓（不复用上一套 VAE ✗）", k_a != k_vae, (k_a, k_vae))


def case_stage_registered() -> None:
    """⑦ 能力自述与就绪报告 ✓：`zimage` 得一档 ✓、**只列这条链的三件** ✗、三件都算必需 ✓。

    ⚠️ 这条**不碰盘上的权重文件** ✗（`components[].required` 与 `requiredWeightsGiB` 都只来自
    清单的 `size_gib` ＋ 规则覆盖 ✓）—— 所以换台机器照样给同一结论 ✓。
    ⚠️ 但**不许断言 `ready`** ✗✗：那是"本机三份权重在不在"，换机必红 ✓（就绪报告由
    `engine_inventory_test` 按真实清单核 ✓）。
    """
    from app.services.engine import inventory as inv  # noqa: PLC0415
    from app.services.engine import registry  # noqa: PLC0415

    check("⑦ `STAGE_FILTERS` 有 `zimage` 一档 ✓，且与 `registry.STAGES` 双向对齐 ✓（少一边就别处会红 ✓）",
          "zimage" in inv.STAGE_FILTERS and "zimage" in registry.STAGES
          and inv.STAGE_FILTERS["zimage"]["category"] == registry.STAGES["zimage"].category,
          {"filter": inv.STAGE_FILTERS.get("zimage"),
           "registry": getattr(registry.STAGES.get("zimage"), "category", None)})

    report = inv.readiness("zimage")
    keys = sorted(c["key"] for c in report["components"])
    check("⑦′ `readiness(\"zimage\")` **只列这条链的三件** ✓✗（不把 `sdxl_base` 混进来 ✓ —— "
          "两套链的组件根本不通 ✓）",
          keys == ["dit_zimage_turbo", "te_zimage_qwen3", "vae_flux_ae"], keys)
    check("⑦″ 三件在这个档里**全部算必需** ✓（`requireAll` 生效 ✓）—— 它不是「有一条链能跑就行」✗",
          all(c["required"] for c in report["components"]),
          [(c["key"], c["required"]) for c in report["components"]])
    #: ⚠️ **反向**：`image` 档（SDXL）**不许**把这 19.27 GiB 算进必需 ✗✗ —— 两套是**互斥备选** ✓，
    #: 加一起 25.73 GiB 在 24 GB 卡上会凭空报「装不下」✓✗（没有任何一条链用得上这个数 ✓）。
    image_report = inv.readiness("image")
    check("⑦‴ `readiness(\"image\")` 的必需权重**不含** Z-Image 那 19.27 GiB ✓✗"
          "（含了就是「互斥备选被加在一起」那个坑 ✓）",
          image_report["requiredWeightsGiB"] < 18.0,
          {"image": image_report["requiredWeightsGiB"],
           "zimage": report["requiredWeightsGiB"]})
    check("⑦⁗ `zimage` 档的必需权重 ≈19.27 GiB ✓（DiT 11.46 ＋ TE 7.49 ＋ VAE 0.31 ✓ —— "
          "这个数与实测 `reserved 19.39 GiB` 对得上 ✓）",
          abs(report["requiredWeightsGiB"] - 19.26) < 0.5, report["requiredWeightsGiB"])

    #: ⚠️ `plan_stage` 与 `readiness` **必须走同一份筛选** ✗✗：2026-10-31 抽 `stage_entries` 之前
    #: `loader.plan_stage` 自己写了一遍 `category` 比较 ⇒ `plan_stage("zimage")` 列出 SDXL、
    #: 且三条 `required: false` 让 ``ready`` **恒为 True** ✓✗（三份权重一份都没有时说"就绪" ✓✗）。
    from app.services.engine import loader  # noqa: PLC0415

    plan = loader.plan_stage("zimage")
    plan_keys = sorted(c["key"] for c in plan["components"])
    check("⑦⁵ `loader.plan_stage(\"zimage\")` 与就绪档**同一份条目** ✓✗（各写一遍筛选会漂 ✗）",
          plan_keys == keys, {"plan": plan_keys, "readiness": keys})
    #: 反验精神：这里不核 `ready` 的真假（依赖盘上有没有 ✓），只核「必需集合非空」——
    #: 必需集合为空时 `ready` 会退化成 `all([]) = True` ✓✗，那才是那个 bug 的形状 ✓。
    required_keys = sorted(c["key"] for c in plan["components"] if c.get("required"))
    check("⑦⁶ `plan_stage` 的必需集合**非空** ✓✗（空集合 ⇒ `ready` 恒 True，文件全缺也说就绪 ✓）",
          required_keys == keys, required_keys)


def case_timesteps_tensor() -> None:
    """⑤ `_timesteps_of`：**原样传 σ** ✓（不取反 ✗）、dtype 跟 latent 走 ✓。"""
    if not _has_torch():
        skip("没装 torch ⇒ timestep 张量那几条不跑 ✓（**没跑 ≠ 绿** ✗）")
        return
    import torch  # noqa: PLC0415

    from app.services.engine.torch_backend import _timesteps_of  # noqa: PLC0415

    latents = torch.zeros((2, 16, 4, 4), dtype=torch.bfloat16)
    out = _timesteps_of(torch, latents, 0.75)
    check("⑤ 形状 `(b,)` ✓（`LuminaDiT` 要逐 batch 的张量 ✗ 不是标量 ✓）",
          tuple(out.shape) == (2,), tuple(out.shape))
    check("⑤′ ⭐ 值**就是 σ 本身** ✓✗（这里先取反 = 主干里再取反 ⇒ 错两次 ✓ 形状全正常 ✓）",
          all(abs(float(v) - 0.75) < 1e-6 for v in out), out.tolist())
    check("⑤″ dtype 跟 latent 走 ✓（bf16 latent + fp32 时间向量 ⇒ 第一层调制就撞 dtype ✗）",
          out.dtype == torch.bfloat16, out.dtype)


def case_te_runtime() -> None:
    """⑥ 文本塔配置壳 ✓：模板只在这里拼一次 ✓、`output_dim` 与主干 `text_dim` 同口径 ✓。"""
    from app.services.engine import qwen3_te  # noqa: PLC0415

    shape = qwen3_te.Qwen3TeShape(vocab=151936, hidden=2560, depth=36, heads=32,
                                  kv_heads=8, head_dim=128, ffn=9728, layer_idx=-2)
    runtime = qwen3_te.Qwen3TeRuntime(shape=shape, max_length=512)
    check("⑥ `output_dim` = hidden ✓（2560 ✓ 与 Z-Image 的 `text_dim` 同口径 ✓）",
          runtime.output_dim == 2560, runtime.output_dim)
    templated = runtime.prompt("a red fox")
    check("⑥′ 模板 `<|im_start|>user … <|im_start|>assistant` ✓（ChatML ✓ 参考实现口径 ✓）",
          templated.startswith("<|im_start|>user") and templated.rstrip().endswith("<|im_start|>assistant"),
          templated)
    check("⑥″ 原文**原样**嵌进去 ✓（不 trim / 不转义 ✓）", "a red fox" in templated, templated)
    check("⑥‴ `to_dict()` 自述 `kind=qwen3-te` ✓（`_textKind` 靠它如实回报 ✓）",
          runtime.to_dict().get("kind") == "qwen3-te", runtime.to_dict())


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


def main() -> int:
    case_sigmas()
    case_family_boundary()
    case_plan()
    case_stage_names()
    case_runtime_refuses()
    case_stage_registered()
    case_timesteps_tensor()
    case_te_runtime()

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
