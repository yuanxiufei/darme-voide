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


def dual_latents(temporal: int = 2, side: int = 2):
    import torch  # noqa: PLC0415

    return {"video": torch.zeros(24, temporal, side, side),
            "audio": torch.zeros(2, temporal, 3)}


def backend_or_skip() -> tb.TorchBackend | None:
    available, reason = tb.torch_available()
    if not available:
        skip(f"没装 torch（{reason} ✓）⇒ 二采张量层验不了 ✓（⚠️ 这不是通过 ✓✗）")
        return None
    return tb.TorchBackend()


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
