"""自检：**超清二采的张量层**（``TorchBackend.refine_latents`` ✓ 2026-09-25 补 ✓）。

补的是什么 ✗：上一轮把**计划层**（:mod:`upscale`）+ **网络本体**（:mod:`upscale_net`）都实现了 ✓，
但**没有任何调用方** ✗✗ —— 真机上带 ``upscale`` 的请求只会得到一句
「后端未实现 refine_latents」（**响亮**是好事 ✓，但能力确实接不出去 ✓✗）。
本套验的是接上之后那半边：**真装载放大器 + 真前向 + 只换视频流** ✓。

三条判据（都是「看着像超分、其实坏了」的形状 ✓✗）：
1. ⭐⭐ **音频流原对象放回** ✗✗（``is`` 判定 ✓ —— 重采音频会把音轨弄坏而画面看着正常 ✓）；
2. ⭐⭐ **时间维不变** ✗✗（``T`` 变了 ⇒ **拒** ✓ —— 时间插值会让动作速率变错 ✓）；
3. ⭐⭐ **严格装载** ✗（缺键/形状不符 ⇒ 拒 ✓：``strict=False`` 会留下随机初始化的层 ⇒
   输出是「像超分」的噪声 ✓✗）。

⚠️ 本套**不宣称**画质 ✗：验的是**管道与不变量** ✓；⚠️ 逆向口径里的「带掩码的二次去噪」
**未实现** ✗（步数/掩码语义未核 ⇒ 不猜 ✓）——自检里**钉住这条如实报告** ✓✗。

运行::

    ./.venv/Scripts/python.exe tests/engine_refine_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PROXY_TO_NODE", "0")
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="refine_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import torch_backend as tb  # noqa: E402
from app.services.engine import upscale as upscale_mod  # noqa: E402
from app.services.engine import upscale_net  # noqa: E402
from app.services.engine import weights as weights_mod  # noqa: E402
from app.services.engine.pipeline import GenerationPlan, GenerationRequest  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    print(f"SKIP  {reason}")


CONTRACT = {
    "format": "minimax_h3_clean_latent_upscaler_v3_factorized_attention",
    "base_config": {"in_channels": 24, "hidden_channels": 8, "num_blocks": 2,
                    "refine_channels": 8, "refine_blocks": 1, "temporal_kernel": 3},
    "config": {"width": 8, "blocks": 1, "heads": 2, "window": 2, "mlp_ratio": 2},
    "strict_latent_only": True,
}


def write_upscaler(path: Path, *, drop_keys: int = 0, contract: dict | None = None) -> Path:
    """造一个**真**放大器权重 ✓（小配置 ✓ CPU 秒级 ✓）⇒ 带内嵌契约的 safetensors ✓。"""
    import torch  # noqa: PLC0415

    spec = dict(contract or CONTRACT)
    torch.manual_seed(0)
    net = upscale_net.H3LatentUpscalerV3(spec["base_config"], spec["config"])
    state = net.state_dict()
    if drop_keys:
        for key in list(state)[-drop_keys:]:
            state.pop(key)
    from safetensors.torch import save_file  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().to("cpu").contiguous() for k, v in state.items()}, str(path),
              metadata={"metadata": json.dumps(spec)})
    return path


def dual_latents(temporal: int = 2, side: int = 2, width: int | None = None):
    import torch  # noqa: PLC0415

    # ⚠️ `width` 是给**非方形**用的 ✓：缩放那段最容易出的静默错是**宽高传反** ✓✗
    #    （`common_upscale(x, width, height, ...)` 显式长名，传反了形状依然合法 ✓ ——
    #    只有非方形才照得出 ✓，方形怎么传都对 ✓）。
    return {"video": torch.zeros(24, temporal, side, width or side),
            "audio": torch.zeros(2, temporal, 3)}


def backend_or_skip() -> tb.TorchBackend | None:
    """⚠️ **显式 CPU** ✓（2026-09-25 在 A5000 上实测）：默认 `device=None` ⇒ **自动探测** ✓
    ⇒ 本机（有卡 ✓）选 `cuda` ✓，而本套的 latents 全是 :func:`dual_latents` 造的 **CPU 张量** ✓
    ⇒ 混用当场炸 ✓✗（且报的是 `aten::slow_conv3d_forward` 这种指不到原因的文案 ✗）。
    本套验的是**二采的张量层数学** ✓ ⇒ 设备无关 ✓ ⇒ 钉 CPU ✓；整条管线的真 CUDA 跑
    由 `engine_dual_stream_test` 覆盖 ✓（真机实测 ✓）。
    """
    available, reason = tb.torch_available()
    if not available:
        skip(f"没装 torch（{reason} ✓）⇒ 二采张量层验不了 ✓（⚠️ 这不是通过 ✓✗）")
        return None
    return tb.TorchBackend(device="cpu")


def case_happy(root: Path) -> None:
    backend = backend_or_skip()
    if backend is None:
        return
    weights = write_upscaler(root / "up_happy.safetensors")
    os.environ["H3_UPSCALER"] = str(weights)
    try:
        latents = dual_latents()
        audio_before = latents["audio"]
        out = backend.refine_latents(latents, GenerationPlan(), GenerationRequest(prompt="x"))
        details = backend.refineDetails or {}
        check("① ⭐⭐ 视频流**空间 2×**（H/W 翻倍 ✓ 通道仍是 24 ✓）",
              tuple(out["video"].shape)[2:] == (4, 4) and tuple(out["video"].shape)[0] == 24,
              tuple(out["video"].shape))
        check("①′ ⭐⭐ **时间维不变** ✗✗（时间插值会让动作速率变错 ✓）",
              tuple(out["video"].shape)[1] == 2 and details.get("temporalUnchanged") is True,
              (tuple(out["video"].shape), details.get("temporalUnchanged")))
        check("①″ ⭐⭐⭐ **音频流是原对象** ✗✗（``is`` ✓ —— 只换视频那一槽 ✓）",
              out["audio"] is audio_before and details.get("audioIdentical") is True,
              (out["audio"] is audio_before, details.get("audioTouched")))
        check("①‴ 键集没变 ✓（video/audio 两条都在 ✓；外加的 keyframes/references 不被吃掉 ✗）",
              set(out) == {"video", "audio"}, sorted(out))
        check("①⁴ 报告里带上**装载路径**与形状 ✓（出问题时查得动 ✓）",
              details.get("upscalerPath") == str(weights)
              and details.get("videoShapeBefore") == [24, 2, 2, 2]
              and details.get("videoShapeAfter") == [24, 2, 4, 4], details)
        check("①⁵ ⭐ **如实报告「二采没跑 + 为什么」** ✗✗（不准把没做的说成做了 ✓✗）",
              details.get("denoiseSteps") == 0 and details.get("mask") is None
              and any("二采未跑" in note for note in (details.get("notes") or [])),
              (details.get("denoiseSteps"), details.get("notes")))
        check("①⁶ 自述**与实现对齐** ✓（2026-09-25 更正 ✗：早先 ``denoise=False`` 与已实现的二采相反 ✓✗）",
              backend.refineNote.get("locksAudio") is True
              and backend.refineNote.get("denoise") is True
              and "已实现" in backend.refineNote.get("note", ""), backend.refineNote)
        check("①⁷ 放大器**按需装载** ✓（没要超清就不该占显存 ✓）",
              backend._upscaler is not None and backend._upscalerReport is not None,
              backend._upscalerReport and backend._upscalerReport.get("complete"))
    finally:
        os.environ.pop("H3_UPSCALER", None)


def case_refusals(root: Path) -> None:
    backend = backend_or_skip()
    if backend is None:
        return

    os.environ["H3_UPSCALER"] = str(root / "nope.safetensors")
    try:
        refused = refusal(backend, dual_latents())
        check("② ⭐ 权重不在盘上 ⇒ **明确拒绝** ✗ 且文案点明「不静默降级」✗✗",
              bool(refused) and "不静默降级" in str(refused) and "回退普通模式" in str(refused),
              refused)
    finally:
        os.environ.pop("H3_UPSCALER", None)

    bad_contract = dict(CONTRACT)
    bad_contract["base_config"] = {**CONTRACT["base_config"], "in_channels": 16}
    os.environ["H3_UPSCALER"] = str(write_upscaler(root / "up_ch16.safetensors", contract=bad_contract))
    try:
        refused = refusal(backend, dual_latents())
        check("②′ ⭐ 契约 ``in_channels != 24`` ⇒ 拒 ✗（**跨来源同一个事实** ✓：契约 ↔ 视频流通道 ✓）",
              bool(refused) and "24" in str(refused), refused)
    finally:
        os.environ.pop("H3_UPSCALER", None)

    os.environ["H3_UPSCALER"] = str(write_upscaler(root / "up_drop.safetensors", drop_keys=2))
    try:
        refused = refusal(backend, dual_latents())
        check("②″ ⭐⭐ 张量比契约**少键** ⇒ 拒 ✗ 且**点名缺几个** ✓"
              "（``strict=False`` 会留下随机初始化的层 ⇒ 输出是「像超分」的噪声 ✓✗）",
              bool(refused) and "不符" in str(refused) and "strict=False" in str(refused), refused)
    finally:
        os.environ.pop("H3_UPSCALER", None)

    # ⭐⭐ 「时间维被改」这条口径要能**真触发** ✗：拿一个故意把 T 加倍的假放大器 ✓
    os.environ["H3_UPSCALER"] = str(write_upscaler(root / "up_ok.safetensors"))
    try:
        backend._upscaler = _TemporalDoubler()
        refused = refusal(backend, dual_latents())
        check("②‴ ⭐⭐ 放大器**改了时间维** ⇒ 拒 ✗✗（这条不报错的话，成片动作速率就是错的 ✓✗）",
              bool(refused) and "时间维" in str(refused), refused)
    finally:
        backend._upscaler = None
        os.environ.pop("H3_UPSCALER", None)

    check("②⁴ 单流潜变量（不是双流 dict）⇒ 拒 ✗（别拿单流来二采 ✓✗）",
          "含 video/audio" in str(refusal(backend, object()) or ""))


def case_second_pass(root: Path) -> None:
    """⭐⭐ **带掩码的低噪声二采**（口径来自上游可读源码 ✓）：真跑一次，钉住「音频逐位不变」✗✗。

    ⚠️ 用**缩小版主干**跑**同一条路** ✓（`latents_dim=2` ✓ ⇒ 契约也写成 2 ✓）——
    这正是本仓的规矩：自检的缩小版与真权重走同一条路 ✓✗（通道数从**已装主干**推 ✓ 不写死 24 ✗）。
    """
    backend = backend_or_skip()
    if backend is None:
        return
    import torch  # noqa: PLC0415

    from app.services.engine import h3_form  # noqa: PLC0415

    tiny = dict(hidden=8, layers=1, heads=2, head_dim=12, ffn=12, text_dim=5, latents_dim=2,
                audio_latents_dim=3, patch_size=(1, 2, 2), time_input_dim=4, time_hidden=8,
                time_dim=6, inv_freq_len=2, refiner_layers=1)
    backend._model = h3_form.H3FormTrunk(**tiny).eval()
    backend._config = {**tiny}

    tiny_contract = dict(CONTRACT)
    tiny_contract["base_config"] = {**CONTRACT["base_config"], "in_channels": 2}
    weights = write_upscaler(root / "up_tiny.safetensors", contract=tiny_contract)
    os.environ["H3_UPSCALER"] = str(weights)
    plan = upscale_mod.UpscalePlan(mode="ai-2x", scale=2.0, refine_steps=2, refine_denoise=0.5)
    latents = {"video": torch.randn(2, 3, 4, 4), "audio": torch.randn(3, 2, 5)}
    text = torch.randn(7, 5)
    try:
        out = backend.refine_latents(latents, plan, GenerationRequest(prompt="x", seed=7),
                                     condition=text)
        details = backend.refineDetails or {}
        check("⑤ ⭐⭐ 二采**真跑了** ✓（``secondPass=True`` ✓ 尾部 σ 数 = 步数+1 ✓、种子 = 段种子+1000001 ✓）",
              details.get("secondPass") is True and details.get("denoiseSteps") == 2
              and details.get("tailSigmas") == 3 and details.get("seed") == 7 + 1000001, details)
        check("⑤′ ⭐⭐⭐ **音频流逐位不变** ✗✗（掩码 0 ⇒ 该流不更新 ✓ —— 真跑出来的事实 ✓，不是自述 ✓）",
              details.get("audioIdentical") is True
              and torch.equal(out["audio"], latents["audio"]), details.get("audioIdentical"))
        check("⑤″ 视频流被二采**改过** ✓（反套套逻辑：没改的话就是「掩码把整段都锁死了」✗✗）",
              not torch.equal(out["video"], latents["video"]), "video 没变")
        check("⑤‴ 形状与时间维都不变 ✓（二采只改数值 ✓ 不改尺寸/帧数 ✓✗）",
              tuple(out["video"].shape) == (2, 3, 8, 8)
              and tuple(out["audio"].shape) == tuple(latents["audio"].shape)
              and details.get("temporalUnchanged") is True, tuple(out["video"].shape))

        refused = refusal(backend, latents, plan=plan, condition=None)
        check("⑤⁴ ⭐ 要二采却**没给 condition** ⇒ **拒** ✗（二采要重跑主干 ✓ 没条件就拒 ✓ 不猜 ✓✗）",
              bool(refused) and "condition" in str(refused), refused)

        backend._model = None
        refused_no_model = refusal(backend, latents, plan=plan, condition=text)
        check("⑤⁵ 二采要重跑主干而**主干没装载** ⇒ **拒** ✗（不静默跳过二采 ✗✗ —— "
              "那会让用户以为二采跑了 ✓）",
              bool(refused_no_model) and "主 DiT" in str(refused_no_model), refused_no_model)
    finally:
        backend._upscaler = None
        backend._model = None
        os.environ.pop("H3_UPSCALER", None)


def case_keyframes(root: Path) -> None:
    """⭐ 二采时**关键帧潜变量也要 2×**（口径来自上游 `_h3_scale_cond_refs` ✓✗）。"""
    backend = backend_or_skip()
    if backend is None:
        return
    import torch  # noqa: PLC0415

    os.environ["H3_UPSCALER"] = str(write_upscaler(root / "up_kf.safetensors"))
    latents = {
        "video": torch.randn(24, 2, 4, 4),
        "audio": torch.randn(2, 2, 3),
        "keyframes": [{"latent": torch.randn(1, 24, 2, 4, 4), "frame_index": 0}],
    }
    try:
        out = backend.refine_latents(latents, GenerationPlan(), GenerationRequest(prompt="x"))
        check("⑥ ⭐ 关键帧潜变量被**同一个 2× 口径**处理 ✓（不同步就会「目标域 2×、关键帧还是 1×」✓✗）",
              tuple(out["keyframes"][0]["latent"].shape)[-2:] == (8, 8)
              and (backend.refineDetails or {}).get("keyframesScaled") == 1,
              tuple(out["keyframes"][0]["latent"].shape))
        check("⑥′ 关键帧条目的**其它字段原样保留** ✓（别把 frame_index 之类吃掉 ✗）",
              out["keyframes"][0].get("frame_index") == 0)
        with_refs = {**latents, "references": [{"latent": torch.randn(1, 24, 2, 4, 4)}]}
        refused = refusal(backend, with_refs)
        check("⑥″ ⭐ **带 ``references`` ⇒ 拒** ✗（要不要跟着 2× **没核到** ✓ ⇒ 不猜 ✓✗）",
              bool(refused) and "references" in str(refused), refused)
    finally:
        backend._upscaler = None
        os.environ.pop("H3_UPSCALER", None)


def case_scale_to_target(root: Path) -> None:
    """⭐⭐ 计划要 1.5× ⇒ 产物必须**真的是 1.5×** ✗✗（2026-11-08 补 ✗）。

    此前 `ai-2x-resize` 的第二步「高质量缩放 2× → 1.5×」**一步都没执行** ✗✗：
    放大器只做**空间 2×** ✓ ⇒ 要 1.5× 的请求拿到的是 2× ✓✗，而 `result.refine.scale`
    照写 1.5 ✓✗ —— **尺寸与回执对不上、回执自己却"自洽"** ✓（本项目最怕的那类错 ✓）。
    """
    backend = backend_or_skip()
    if backend is None:
        return
    import torch  # noqa: PLC0415

    os.environ["H3_UPSCALER"] = str(write_upscaler(root / "up_scale.safetensors"))
    try:
        # 24×2×2×2 ⇒ 放大器 2× 得 4×4 ⇒ 计划 1.5× 应是 **3×3**（4×4 就是没缩 ✗）
        latents = dual_latents(temporal=2, side=2)
        audio_before = latents["audio"]
        plan = upscale_mod.UpscalePlan(mode="ai-2x-resize", scale=1.5)
        out = backend.refine_latents(latents, plan, GenerationRequest(prompt="x"))
        details = backend.refineDetails or {}
        check("⑦ ⭐⭐ `ai-2x-resize` 1.5× ⇒ 潜变量**缩到 3×3** ✓（此前一律是 4×4 ✓✗）",
              tuple(out["video"].shape)[2:] == (3, 3)
              and details.get("videoShapeScaled") == [24, 2, 3, 3],
              (tuple(out["video"].shape), details.get("videoShapeScaled")))
        check("⑦′ ⭐⭐ 时间维**一点没动** ✗✗（只走空间 ✓ —— 时间插值会让动作速率变错 ✓）",
              tuple(out["video"].shape)[1] == 2 and details.get("temporalUnchanged") is True,
              tuple(out["video"].shape))
        check("⑦″ 回执**写实际值与方法** ✓（不是照抄计划 ✓✗；换过方法也看得见 ✓）",
              details.get("targetScale") == 1.5
              and abs(float(details.get("spatialScale") or 0.0) - 1.5) <= 1e-6
              and details.get("resizeMethod") == backend.SCALE_RESIZE_METHOD, details)
        check("⑦‴ 音频流仍是**原对象** ✓（缩放不该碰它 ✗）",
              out["audio"] is audio_before and details.get("audioTouched") is False,
              details.get("audioTouched"))

        # ⭐ 关键帧必须**同一尺寸** ✗✗：主视频 1.5× 而关键帧 2× 是"擦边能跑"的静默错 ✓✗
        with_kf = {**latents, "keyframes": [{"latent": torch.randn(1, 24, 2, 2, 2),
                                            "frame_index": 0}]}
        out_kf = backend.refine_latents(with_kf, plan, GenerationRequest(prompt="x"))
        check("⑦⁴ ⭐⭐ 关键帧**跟着缩到 3×3** ✓（不同步就是静默错 ✓✗）",
              tuple(out_kf["keyframes"][0]["latent"].shape)[-2:] == (3, 3),
              tuple(out_kf["keyframes"][0]["latent"].shape))

        # ⭐⭐ **非方形**必须单独判 ✗✗（2026-11-08 补 ✗）：`common_upscale(x, width, height, ...)`
        #    是**显式长名** ✓，宽高传反了**形状依然合法** ✓✗ ⇒ 方形（上面 ⑦ 那几条全用它 ✓）
        #    怎么传都对 ✓、照不出这个错 ✗。非方形下「传反」= 把 2×4 缩成 4×3 ✓
        #    ⇒ 画面被各向异性拉斜 ✓ 而任何一步都不会报错 ✓。
        #    2×4（H=2 W=4）计划 1.5× ⇒ 应得 **H=3、W=6** ✓（传反则是 4×3 ✓）。
        rect = dual_latents(temporal=2, side=2, width=4)
        out_rect = backend.refine_latents(rect, plan, GenerationRequest(prompt="x"))
        rect_details = backend.refineDetails or {}
        check("⑦¹³ ⭐⭐ **非方形** 2×4 缩到 **3×6**（H=3/W=6 ✓）—— 宽高传反会得 4×3 且"
              "**不报任何错** ✓✗（方形用例照不出来 ✓）",
              tuple(out_rect["video"].shape)[2:] == (3, 6)
              and rect_details.get("videoShapeScaled") == [24, 2, 3, 6],
              (tuple(out_rect["video"].shape), rect_details.get("videoShapeScaled")))
        with_kf_rect = {**rect, "keyframes": [{"latent": torch.randn(1, 24, 2, 2, 4),
                                              "frame_index": 0}]}
        out_kf_rect = backend.refine_latents(with_kf_rect, plan, GenerationRequest(prompt="x"))
        check("⑦¹⁴ 关键帧在非方形下也是 **3×6** ✓（宽高同样不许传反 ✓）",
              tuple(out_kf_rect["keyframes"][0]["latent"].shape)[-2:] == (3, 6),
              tuple(out_kf_rect["keyframes"][0]["latent"].shape))
    finally:
        backend._upscaler = None
        os.environ.pop("H3_UPSCALER", None)

    # ⭐⭐ `tiled-ai-2x`（>2×）要**多次分块放大 + 缝合** ✓ —— 执行层没做 ⇒ 必须**响亮拒绝** ✗✗
    #     （静默出 2× 会让人以为 4× 开着 ✓✗ —— 比直接失败更坏 ✓）
    os.environ["H3_UPSCALER"] = str(write_upscaler(root / "up_4x.safetensors"))
    try:
        plan4 = upscale_mod.UpscalePlan(mode="tiled-ai-2x", scale=4.0)
        refused = refusal(backend, dual_latents(), plan=plan4)
        check("⑦⁵ ⭐⭐ `tiled-ai-2x`（4×）执行层没实现 ⇒ **拒** ✗ 且点明「没实现」✗✗",
              bool(refused) and "没实现" in str(refused), refused)
        unknown = upscale_mod.UpscalePlan(mode="ai-8x-magic", scale=8.0)
        refused_unknown = refusal(backend, dual_latents(), plan=unknown)
        check("⑦⁶ 不认识的模式 ⇒ **拒** ✓（不猜它要几倍 ✓✗）",
              bool(refused_unknown) and "不认识的超清模式" in str(refused_unknown), refused_unknown)

        # ⭐⭐ `ai-2x-resize` 的**区间**也必须核 ✗✗（2026-11-08 补 ✗）：
        #    计划层只在 ``1 < scale < 2`` 才产这个模式 ✓，而请求体是**手写 dict** ⇒ 能配出计划层
        #    永远不产的组合 ✓✗。两种都会**静默出假超清**（尺寸/回执双双自洽 ✓）。
        shrink = refusal(backend, dual_latents(),
                        plan=upscale_mod.UpscalePlan(mode="ai-2x-resize", scale=1.0))
        check("⑦¹⁰ ⭐⭐ `ai-2x-resize` + ``scale=1.0`` ⇒ **拒** ✗✗（= 先放大 2× 再缩回原尺寸："
              "**尺寸一模一样、耗时 ≈6×** ✓✗ —— 回执 applied/scale 全自洽，最像「成功」✓）",
              bool(shrink) and "不需要放大器" in str(shrink), shrink)
        fake_up = refusal(backend, dual_latents(),
                          plan=upscale_mod.UpscalePlan(mode="ai-2x-resize", scale=3.0))
        check("⑦¹¹ ⭐⭐ `ai-2x-resize` + ``scale=3.0`` ⇒ **拒** ✗✗（放大器只给 2× ⇒ 2→3 是"
              "**插值**放大、不长细节 ✓✗ —— 会被当成「AI 3×」交付 ✓）且指向 ``tiled-ai-2x`` ✓",
              bool(fake_up) and "tiled-ai-2x" in str(fake_up), fake_up)
        two = refusal(backend, dual_latents(),
                      plan=upscale_mod.UpscalePlan(mode="ai-2x-resize", scale=2.0))
        check("⑦¹² ``scale=2.0`` 配这个模式 ⇒ **拒** ✓ 并指向 ``ai-2x`` ✗（同名 2× 但走的是"
              "「缩到同尺寸」= 白多一次 bicubic 重采样 ✓✗）",
              bool(two) and "恰好 2× 请用" in str(two), two)
    finally:
        backend._upscaler = None
        os.environ.pop("H3_UPSCALER", None)


def case_scale_with_second_pass(root: Path) -> None:
    """⭐⭐ 缩放与二采**同一轮**：二采必须在**目标域**跑 ✓✗（2026-11-08 补）。

    两条路各自对、拼起来错的情况最容易漏 ✓✗：只测 1.5×（不带二采）看不出二采在哪个域跑 ✓。
    """
    backend = backend_or_skip()
    if backend is None:
        return
    import torch  # noqa: PLC0415

    from app.services.engine import h3_form  # noqa: PLC0415

    tiny = dict(hidden=8, layers=1, heads=2, head_dim=12, ffn=12, text_dim=5, latents_dim=2,
                audio_latents_dim=3, patch_size=(1, 2, 2), time_input_dim=4, time_hidden=8,
                time_dim=6, inv_freq_len=2, refiner_layers=1)
    backend._model = h3_form.H3FormTrunk(**tiny).eval()
    backend._config = {**tiny}
    tiny_contract = dict(CONTRACT)
    tiny_contract["base_config"] = {**CONTRACT["base_config"], "in_channels": 2}
    os.environ["H3_UPSCALER"] = str(
        write_upscaler(root / "up_mix.safetensors", contract=tiny_contract))
    try:
        plan = upscale_mod.UpscalePlan(mode="ai-2x-resize", scale=1.5,
                                       refine_steps=2, refine_denoise=0.5)
        latents = {"video": torch.randn(2, 3, 4, 4), "audio": torch.randn(3, 2, 5)}
        out = backend.refine_latents(latents, plan, GenerationRequest(prompt="x", seed=7),
                                     condition=torch.randn(7, 5))
        details = backend.refineDetails or {}
        check("⑦⁷ ⭐⭐ 1.5× 与二采**同一轮** ⇒ 最终 6×6 ✓（拿到 8×8 就是「没缩」✗✗）",
              tuple(out["video"].shape) == (2, 3, 6, 6), tuple(out["video"].shape))
        check("⑦⁸ ⭐⭐ 二采真跑了 ✓（secondPass=True、尾 σ 数 = 步数+1 ✓）",
              details.get("secondPass") is True and details.get("tailSigmas") == 3, details)
        check("⑦⁹ 音频**逐位不变** ✓（缩放与二采都不该碰它 ✗✗）",
              torch.equal(out["audio"], latents["audio"]), details.get("audioIdentical"))
        # ⭐⭐ **两段须知都要在** ✗✗（2026-11-08 补 ✗）：``denoise_report`` 自己带 ``notes`` ✓ ⇒
        #    用 ``details.update(...)`` 会把上面那段缩放假话**整段冲掉** ✓✗，而**只要二采真跑了
        #    就必然冲掉** ✓（成功/失败两条路径都带 notes ✓）⇒ 回执只剩「二采：尾部 N 步」✓，
        #    读者**根本不知道画面被缩过** ✓（倍率/尺寸那两格还在 ✓ 所以不报错 ✓）。
        check("⑦¹⁵ ⭐⭐ 二采跑过之后，**缩放假话还在** ✓（``notes`` 要并、不能 ``update`` 覆盖 ✗✗）；"
              "两段（缩放＋二采）都在才叫如实 ✗",
              any("缩放" in note for note in (details.get("notes") or ()))
              and any("二采" in note for note in (details.get("notes") or ())),
              details.get("notes"))
    finally:
        backend._model = None
        backend._config = None
        backend._upscaler = None
        os.environ.pop("H3_UPSCALER", None)


def case_wiring() -> None:
    """⭐⭐ **生产链路**必须把「超清计划」交给后端 ✗✗（2026-11-08 补 ✗）。

    这是一处**测试与生产不同构**的真错 ✓✗：本套一直传 ``UpscalePlan`` ⇒ 绿 ✓，
    而 `pipeline` 传的是 ``GenerationPlan`` ✓✗ ⇒ 后端
    ``getattr(plan, 'refine_steps', 0)`` **恒为 0** ⇒ 二采从未跑过 ✓✗
    （回执还写"⚠️ 二采未跑：plan.refine_steps = 0"✓ 而用户明明给了 refineSteps ✓✗）。
    """
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "app/services/engine/pipeline.py").read_text(encoding="utf-8")
    check("⑧ ⭐⭐ 管线传的是 `request.upscale`（超清计划）而**不是** `plan` ✗✗"
          "（后者没有 refine_steps/mode/scale ⇒ 二采与缩放双双失效 ✓✗）",
          "hook(latents, request.upscale, request" in source
          and "hook(latents, plan, request" not in source)
    backend_src = (repo / "app/services/engine/torch_backend.py").read_text(encoding="utf-8")
    check("⑧′ ⭐ `image_ops` 真被接上了 ✓（欠账表已删 ⇒ 生产代码必须找得到引用 ✗）",
          "image_ops.common_upscale" in backend_src)


def case_gate() -> None:
    """⚠️ 没装 torch ⇒ ``refine_latents`` 必须**响亮** ✗（不是静默当没这回事 ✓）。"""
    backend = tb.TorchBackend()
    backend._available = False
    backend._reason = "（自检故意 ✓）"
    refused = refusal(backend, {"video": None, "audio": None})
    check("③ 后端不可用 ⇒ 走 ``_gate()`` 的**明确拒绝** ✓（不是 AttributeError 之类的天书 ✓✗）",
          bool(refused) and "（自检故意 ✓）" in str(refused), refused)


class _TemporalDoubler:
    """假放大器：**故意**把时间维翻倍 ✓（用来证明「时间维变了就拒」这条判据**能触发** ✓✗）。"""

    def __call__(self, value):
        import torch  # noqa: PLC0415

        return torch.cat([value, value], dim=2)

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    def requires_grad_(self, *_args, **_kwargs):
        return self


def case_details_truthful(root: Path) -> None:
    """⭐⭐ 实测值必须**每次都算、每次都有**（2026-11-08 补 ✗）。

    两处此前都缺 ✓✗：
    ① ``spatialScale`` 只在「有第二步缩放」时写 ⇒ ``ai-2x`` 那条（放大器原样 2×）的回执里
       根本没有实测倍率 ⇒ 管线只好拿**计划值**顶 ✓（计划层 vs 执行层的老毛病 ✓）；
    ② ``refineDetails`` 是**实例属性** ⇒ 上一次二采的值会活到这一次 ✓：本次若在设值**之前**
       失败 ✓，回执写的就是**上一次**的倍率与尺寸 ✓ —— 数字自洽、只与画面不符 ✓。
    """
    backend = backend_or_skip()
    if backend is None:
        return

    os.environ["H3_UPSCALER"] = str(write_upscaler(root / "up_truth.safetensors"))
    try:
        # ① 放大器原样 2×（target 为 None）也必须报实测倍率 ✓
        plan2 = upscale_mod.UpscalePlan(mode="ai-2x", scale=2.0)
        out2 = backend.refine_latents(dual_latents(temporal=2, side=2), plan2,
                                      GenerationRequest(prompt="x"))
        d2 = backend.refineDetails or {}
        check("⑨ ⭐⭐ ``ai-2x``（放大器原样 2×、没有第二步缩放）也报**实测倍率 2×** ✓"
              "（此前这一格整天缺席 ⇒ 管线只能拿计划值顶 ✓✗）",
              d2.get("spatialScale") == 2.0 and d2.get("spatialScaleW") == 2.0
              and "videoShapeScaled" not in d2, d2)
        check("⑨′ 倍率的凭据是**尺寸比** ✓（2×2 → 4×4 ⇒ 4/2 ✓，不是照抄计划 ✓）",
              tuple(out2["video"].shape)[2:] == (4, 4)
              and d2.get("spatialScale") == round(4 / 2, 6), tuple(out2["video"].shape))

        # ② 先跑一次 1.5× 留下 details ⇒ 再让它**在设值之前**失败 ⇒ 旧值不许残留 ✗✗
        backend.refine_latents(dual_latents(temporal=2, side=2),
                               upscale_mod.UpscalePlan(mode="ai-2x-resize", scale=1.5),
                               GenerationRequest(prompt="x"))
        stale = backend.refineDetails or {}
        plan_ref = upscale_mod.UpscalePlan(mode="ai-2x", scale=2.0,
                                          refine_steps=4, refine_denoise=0.4)
        # 缺 ``condition`` ⇒ 二采前就被拒 ✓（拒绝点在 ``self.refineDetails = details`` 之前 ✓）
        err = refusal(backend, dual_latents(), plan=plan_ref)
        check("⑨″ ⭐⭐ 失败那次**不留上一次的实测值** ✗✗（残留 ⇒ 回执写 1.5× 与 3×3 尺寸 ✓，"
              "而这一跑一张画面都没有 ✓）",
              stale.get("spatialScale") == 1.5 and "condition" in str(err)
              and backend.refineDetails is None,
              (stale.get("spatialScale"), err, backend.refineDetails))
    finally:
        backend._upscaler = None
        os.environ.pop("H3_UPSCALER", None)


def refusal(backend: tb.TorchBackend, latents: object, *,
            plan: object | None = None, condition: object = None) -> str | None:
    try:
        backend.refine_latents(latents, plan or GenerationPlan(), GenerationRequest(prompt="x"),
                               condition=condition)
    except tb.TorchBackendUnavailable as err:
        return str(err)
    except Exception as err:  # noqa: BLE001 —— 别让别的异常类型漏过去 ✓✗
        return f"__WRONG__{type(err).__name__}: {err}"
    return None


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="refine_weights_"))
    case_happy(root)
    case_refusals(root)
    case_second_pass(root)
    case_keyframes(root)
    case_scale_to_target(root)
    case_details_truthful(root)
    case_scale_with_second_pass(root)
    case_wiring()
    case_gate()
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
