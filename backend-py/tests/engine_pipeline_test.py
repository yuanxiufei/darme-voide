"""S7 自检：引擎**管线编排**（零依赖 ✓ 2026-09-17）。

为什么这套值得写：管线是本引擎的"总装线" ✓ —— 它错了的话，**每个后端都错** ✗，
而且错法很难看：卡在某阶段（用户只看到转圈 ✗）、失败只给一句异常（不知道是编码还是解码 ✗）、
取消要等几分钟才生效 ✗。这些**都与 GPU 无关** ✓ ⇒ 用干跑后端（纯 Python 小向量）就能钉死 ✓。

覆盖：阶段顺序 / 进度事件 / **算法层真被调用**（收敛 + 换采样器结果真的不同 ✓）/ 取消（阶段边界 + 步中 ✓）/
错误归因（阶段 + 步号 ✓）/ 同种子可复现 / 干跑输出**明确标注合成** ✗（不许冒充真结果 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_pipeline_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
#: ⚠️ 干跑报告**必须**隔离到临时树 ✓✗（`output_paths.ENV_DRYRUN_OVERRIDE`）：
#: 本套的 `outputs_dir` 是 `mkdtemp(prefix="engine_pipe_")` 随机目录 ✓，以前没人设这个环境
#: 变量 ⇒ 每次跑完都往**真数据根** `data/engine-dryrun/` 里留 29 个 `engine_pipe_xxxx/` ✓✗
#: （2026-11-07 实测积了 29 个 ✓，全部创建在同一秒 ✓）。用 `setdefault` 让外层脚本能覆盖 ✓。
os.environ.setdefault("DARME_ENGINE_DRYRUN", tempfile.mkdtemp(prefix="dryrun_pipe_"))
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import keyframes as keyframes_mod  # noqa: E402
from app.services.engine import pipeline as pipe  # noqa: E402
from app.services.engine import upscale as upscale_mod  # noqa: E402
from app.services.engine import vae_h3  # noqa: E402
from app.services.engine.dryrun import DryRunBackend, TinyTensor  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    """**显式**跳过 ✓（因缺依赖而跳过时不许默默变绿 ✗ —— 汇总会印出 skip 数 ✓）。"""
    _SKIPS.append(reason)


class _CountingProxy:
    """给**任意**后端套一层数调用次数 ✓（引导那条硬证据要证「真打了两次」✓）。"""

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "calls", 0)

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)

    def denoise(self, latents: Any, sigma: float, condition: Any, request: Any) -> Any:
        object.__setattr__(self, "calls", self.calls + 1)
        return self.inner.denoise(latents, sigma, condition, request)


def req(**overrides: Any) -> pipe.GenerationRequest:
    base: dict[str, Any] = {"prompt": "雨夜霓虹街头，主角回头", "seed": 7, "seconds": 5.0,
                            "steps": 6, "outputs_dir": tempfile.mkdtemp(prefix="engine_pipe_")}
    base.update(overrides)
    return pipe.GenerationRequest(**base)


# ══════════════════════════════════════════════════════════════════════════
# ① build_plan：请求 → 具体数字（跑之前就能给用户看 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_plan() -> None:
    plan = pipe.build_plan(req())
    check("① 0.65MP @16:9 ⇒ 1088×608（与 geometry 的实测锚点一致 ✓）",
          (plan.width, plan.height) == (1088, 608), (plan.width, plan.height))
    check("② 5s@24fps ⇒ 124 帧（吸附到 17k+5 网格 ✓）", plan.frames == 124, plan.frames)
    check("③ σ 序列：长度 = steps+1、末位 0、单调递减（采样器硬要求 ✓）",
          len(plan.sigmas) == plan.steps + 1 and plan.sigmas[-1] == 0.0
          and all(plan.sigmas[i] >= plan.sigmas[i + 1] for i in range(len(plan.sigmas) - 1)),
          plan.sigmas[:3])
    check("④ timesteps 由 σ 派生（同一份数学 ✓）", len(plan.timesteps) == len(plan.sigmas),
          plan.timesteps[:3])
    check("⑤ 计划里带**人话警告**（帧数/时长已吸附 ✓）",
          any("网格" in item for item in plan.warnings), plan.warnings)

    check("⑥ 空提示词 ⇒ plan 阶段就失败（不是跑到一半才炸 ✗）",
          _plan_error(req(prompt="   ")) is not None)
    check("⑦ steps=0 ⇒ plan 失败", _plan_error(req(steps=0)) is not None)
    check("⑧ seconds=0 ⇒ plan 失败", _plan_error(req(seconds=0)) is not None)
    unknown = _plan_error(req(sampler="no-such-sampler"))
    check("⑨ 未知采样器 ⇒ 报错里列出可用项（能据此行动 ✓）",
          unknown is not None and "可用" in str(unknown), unknown)

    with_compression = pipe.build_plan(req(temporal_compression=4))
    #: 期望值**不许写死**。2026-11-04 起 h3 走「17 帧一组、组内 5 倍压缩」的分段算式
    #: `5*ceil(T/17)-3`，此前用的通用式 `(T-1)/4+1` 对 h3 是错的：124 帧算出 31，
    #: 交给 H3 的 VAE 只能解回 103 帧 —— 计划 5.17 秒、成片 4.29 秒，而且**一步都不报错**。
    #: 这里原先写死 `== 31`，公式改对之后它就一直红着（168 次静默丢帧的活证）。
    #: 改成「拿 VAE 自己的算式当期望值」＋「必须闭环」，公式再改这条也不会过期。
    _h3_lat = vae_h3.latents_for_frames(with_compression.frames)
    check("⑩ 给了 temporalCompression ⇒ 算出潜空间帧数 ✓",
          with_compression.latent_frames == _h3_lat,
          (with_compression.latent_frames, _h3_lat, with_compression.frames))
    check("⑩' 潜帧数必须能被 H3 VAE 解回原帧数（解不回来＝静默丢帧）",
          vae_h3.frames_for_latents(with_compression.latent_frames) == with_compression.frames,
          (with_compression.latent_frames,
           vae_h3.frames_for_latents(with_compression.latent_frames),
           with_compression.frames))
    without = pipe.build_plan(req())
    check("⑩' 没给 ⇒ **不猜**，如实警告（潜空间帧数未知 ✓）",
          without.latent_frames is None and any("temporalCompression" in item
                                                for item in without.warnings), without.warnings)
    with_weights = pipe.build_plan(req(), weights_bytes=int(19 * 1024 ** 3))
    check("⑩'' 带上权重体积 ⇒ 计划里附带显存估算（含免责声明 ✓）",
          any("GiB" in item and "估算" in item for item in with_weights.warnings),
          with_weights.warnings[-1:])


def _plan_error(request: pipe.GenerationRequest) -> Exception | None:
    try:
        pipe.build_plan(request)
    except pipe.StageError as err:
        return err
    return None


# ══════════════════════════════════════════════════════════════════════════
# ② 干跑整条管线
# ══════════════════════════════════════════════════════════════════════════
def case_schedule_names() -> None:
    """㉕ `schedule=` 收工作流的档名 ✓（``simple`` / ``beta`` ✓），EDM 档名**不拒单**但会说清楚 ✓。

    ⚠️ 为什么"不拒单" ✗：工作流 JSON 里 ``KSampler.scheduler`` 可能是 ``karras``/``normal``
    这类 EDM 档名 ✓，而本引擎的各族 σ 都是 **flow** 域 ✓（恒在 [0,1] ✓）—— 直接拒单会让
    「照工作流参数原样跑」在这条路上断死 ✓✗；静默按 EDM 跑又会让 σ 越过 1 ✓✗。
    ⇒ 按 flow 跑 **＋ 回执写明"这个档名没接、按 flow 算"** ✓。
    """
    plain = pipe.build_plan(req(steps=8, schedule="simple"))
    check("㉕ `schedule=simple` ⇒ 收下 ✓、σ 长度 = 步数+1 ✓、σ₀ = 1 ✓、末位 0 ✓、严格递减 ✓",
          len(plain.sigmas) == 9 and plain.sigmas[0] == 1.0 and plain.sigmas[-1] == 0.0
          and all(a > b for a, b in zip(plain.sigmas[:-1], plain.sigmas[1:])), plain.sigmas)

    bent = pipe.build_plan(req(steps=8, schedule="beta"))
    check("㉕′ `schedule=beta` ⇒ 收下 ✓（长度可 < 步数+1 ✓：`rint` 撞车合并 ✓）、末位 0 ✓",
          len(bent.sigmas) <= 9 and bent.sigmas[-1] == 0.0
          and all(a > b for a, b in zip(bent.sigmas[:-1], bent.sigmas[1:])), bent.sigmas)

    flow = pipe.build_plan(req(steps=8))
    outlier = pipe.build_plan(req(steps=8, schedule="karras"))
    check("㉕″ 视频侧没接的档名（`karras`）⇒ **不拒单** ✓、照跑 ✓、且回执**如实报「域不重叠」** ✓✗"
          "（静默 ⇒ 用户以为 karras 生效了 ✓✗；口径见 `engine_video_schedule_test` ③⁵③⁶ ✓）",
          outlier.sigmas != flow.sigmas and max(outlier.sigmas) > 1.0
          and any("域不重叠" in text or "不在 flow" in text or "超出 flow" in text
                  for text in outlier.warnings), outlier.warnings[:2])

    # ⚠️ **图族**相反：EDM 名**按 flow 跑** ✓（图族整条链路只认得 flow 域 ✓，照跑会出糊图 ✗）
    image_flow = pipe.build_plan(req(prompt="a cat", stage=pipe.QWEN_IMAGE_STAGE, steps=8))
    image_edm = pipe.build_plan(req(prompt="a cat", stage=pipe.QWEN_IMAGE_STAGE, steps=8,
                                    schedule="karras"))
    check("㉕‴ ⭐ 图族碰到 EDM 名 ⇒ **按 flow 跑**（σ 与默认逐位相同 ✓）＋ 回执写明已换 ✓✗"
          "（照 EDM 跑 ⇒ σ>1 ⇒ `t<0` ⇒ 出糊图 ✗✗，而日志全绿 ✓）",
          image_edm.sigmas == image_flow.sigmas
          and any("karras" in text for text in image_edm.warnings), image_edm.warnings[:2])


def case_end_to_end() -> None:
    backend = DryRunBackend()
    request = req(steps=6)
    events: list[dict[str, Any]] = []
    result = pipe.run_sync(request, backend, on_event=events.append)

    check("⑪ 干跑能跑完整条管线（ok=True ✓）", result.ok is True, result.error)
    check("⑫ 输出**显式标注合成**（不许被当成真结果 ✗）",
          result.synthetic is True and result.outputs.get("synthetic") is True
          and result.outputs.get("videoPath") is None, result.outputs.get("artifacts"))
    check("⑬ 采样步数 = 计划步数（σ 序列被真用上了 ✓）",
          result.sampleSteps == request.steps == 6, result.sampleSteps)
    check("⑭ 六个阶段都有耗时 ✓ 且 totalMs = 各阶段之和 ✓",
          all(stage in result.stageMs for stage in pipe.STAGE_ORDER)
          and result.totalMs == sum(result.stageMs.values()), result.stageMs)

    kinds: list[str] = [event["stage"] for event in events]
    check("⑮ 事件顺序：plan → encode → init → sample×N → decode → write ✓",
          kinds[0] == "plan" and kinds[1] == "encode" and kinds[2] == "init"
          and kinds[-1] == "write" and kinds.count("sample") >= request.steps
          and kinds.index("decode") < kinds.index("write"), kinds)
    sampled = [event for event in events if event["stage"] == "sample" and "step" in event]
    check("⑯ 采样事件带**步号/总数**（前端进度条要的就是它 ✓）",
          sampled and sampled[0]["step"] == 1 and sampled[0]["total"] == 6,
          (sampled[0] if sampled else None))
    check("⑰ 事件带人可读标签 ✓", isinstance(sampled[0].get("label"), str) and sampled[0]["label"])

    artifacts = result.outputs.get("artifacts") or []
    payload = json.loads(Path(artifacts[0]["path"]).read_text(encoding="utf-8")) if artifacts else {}
    check("⑱ 落盘的是**干跑报告**（不是 mp4 ✗），里面写明「不是画面」✓",
          bool(artifacts) and payload.get("synthetic") is True
          and "不是生成的画面" in str(payload.get("note")), payload.get("note"))
    check("⑲ 报告里同时记下请求与计划（可复现/可审计 ✓）",
          payload.get("request", {}).get("seed") == 7 and payload.get("plan", {}).get("steps") == 6)


def case_algorithm_really_runs() -> None:
    backend = DryRunBackend()
    request = req(steps=8)

    # ⭐ 收敛：采样**真的**把噪声推向了条件（否则算法层只是被「路过」✗）
    finals = _final_latents(backend, request)
    check("⑳ ⭐ 采样**真收敛**：末态与条件的距离 << 初始噪声到条件的距离（算法层真在跑 ✓）",
          finals["finalDistance"] < finals["initialDistance"] / 2, finals)

    # ⭐ 换采样器真的换算法（若模型与 x 无关，三种会一模一样 ⇒ 那种自检是假绿 ✗）
    outs = {name: _preview(req(steps=8, sampler=name))
            for name in ("euler", "heun", "multistep")}
    check("㉑ ⭐ 三种采样器结果**确实不同**（证明管线的 sample 阶段真调了采样器 ✓）",
          len(set(outs.values())) == 3, {k: v[:2] for k, v in outs.items()})

    check("㉒ 同种子 ⇒ 完全可复现（sha256 条件 + Random(seed) ✓）",
          _preview(req(seed=11)) == _preview(req(seed=11)), "")
    check("㉓ 换种子 ⇒ 结果不同（种子真的生效 ✓）",
          _preview(req(seed=11)) != _preview(req(seed=12)), "")
    check("㉔ 提示词变了 ⇒ 结果也变（否则「同种子就同结果」会掩盖条件没生效 ✗）",
          _preview(req(prompt="另一个场景")) != _preview(req()), "")

    # ⭐⭐ 2026-11-01 新补的三档（名字直接照抄工作流 ✓）——判据同 ㉑：**必须彼此不同** ✓✗。
    # 「注册了」不等于「接上了」✓✗：`SAMPLERS` 里加个名字、`sample()` 里忘记分派时，
    # 全仓测试都绿、只有出图会静默退回 euler ✓（同名不同结果才是真接线 ✓）。
    fresh = {name: _preview(req(steps=8, sampler=name))
             for name in ("res_multistep", "er_sde", "lcm")}
    check("㉔′ ⭐ `res_multistep` / `er_sde` / `lcm` 三档结果**两两不同** ✓ 且都与 `euler` 不同 ✓",
          len({*fresh.values(), outs["euler"]}) == 4, {k: v[:2] for k, v in fresh.items()})


class _CountingNoise(DryRunBackend):
    """数「每步真的来要噪声了吗」✓ 并记下 ``nonce`` ✓（要证的是**通路**＋**每步不同** ✓）。

    ⚠️ 判据不能写成「调用次数 = 步数」✗：祖先档最后一步 ``σ_next == 0`` 直接回 x₀ ✓，
    ``sa_solver`` 的随机强度只在 σ 中段（20%~80% ✓）打开 ✓ ⇒ 两者都**少于**步数 ✓。
    写成等式会把正确行为判成缺陷 ✓✗（同族：把"没有 ``layers.N``"当缺陷那次 ✓）。
    """

    def __init__(self) -> None:
        self.nonces: list[int] = []
        self.sizes: list[int] = []

    def noise_like(self, sample, request, *, nonce=0):  # noqa: ANN001
        self.nonces.append(int(nonce))
        self.sizes.append(len(sample.data))
        return super().noise_like(sample, request, nonce=nonce)


class _NoNoiseBackend(DryRunBackend):
    """**没有** ``noise_like`` 的后端 ✓（模拟只做了确定性那条路的实现 ✓）。"""

    noise_like = None


def case_stochastic_samplers() -> None:
    """随机采样器（``sa_solver`` / ``euler_ancestral``）在**真管线**里的噪声通路 ✓。"""
    for name in ("sa_solver", "euler_ancestral"):
        backend = _CountingNoise()
        request = req(steps=8, sampler=name)
        result = pipe.run_sync(request, backend)
        want = len(DryRunBackend().init_latents(pipe.build_plan(request), request).data)
        check(f"㉕ᴿ {name}：管线**真把噪声注回去**了 ✓（{name} 是随机档 ⇒ 没人来要就说明接错了 ✗）",
              result.ok and 1 <= len(backend.nonces) < 8, (result.error, backend.nonces))
        check(f"㉖ᴿ {name}：每次的 ``nonce`` **都不同** ✓✗（同一条噪声反复注入 ⇒ 结构性条纹 ✓）",
              len(set(backend.nonces)) == len(backend.nonces) and len(backend.nonces) >= 1,
              backend.nonces)
        check(f"㉗ᴿ {name}：噪声**形状跟着当前 x** ✓（不是随手造一个 ✓）",
              bool(backend.sizes) and set(backend.sizes) == {want}, (backend.sizes, want))

    missing = pipe.run_sync(req(steps=4, sampler="sa_solver"), _NoNoiseBackend())
    check("㉘ᴿ ⭐ 后端没实现 ``noise_like`` ⇒ **sample 阶段明确报错** ✓✗"
          "（静默按 ODE 跑 = 用户以为在跑随机档 ✓✗ 本仓最忌讳「看不出来」✗）",
          missing.error is not None and missing.error.get("stage") == "sample"
          and "noise_like" in str(missing.error.get("message")), missing.error)
    check("㉙ᴿ 确定性采样器（euler）不碰这条通路 ✓（后端没实现也照跑 ✓ 不该被牵连 ✗）",
          pipe.run_sync(req(steps=4, sampler="euler"), _NoNoiseBackend()).ok is True)
    check("㉚ᴿ 随机档**同 seed 可复现** ✓（nonce 序列是确定的 ✓）",
          _preview(req(steps=6, sampler="euler_ancestral"))
          == _preview(req(steps=6, sampler="euler_ancestral")))


def _preview(request: pipe.GenerationRequest) -> tuple[float, ...]:
    result = pipe.run_sync(request, DryRunBackend())
    path = result.outputs["artifacts"][0]["path"]
    return tuple(json.loads(Path(path).read_text(encoding="utf-8"))["decode"]["preview"])


def _final_latents(backend: DryRunBackend, request: pipe.GenerationRequest) -> dict[str, float]:
    """直接跑「几何 + 采样」两段，量出收敛程度（复用管线自己的计划 ✓）。"""
    plan = pipe.build_plan(request)
    condition = backend._condition(request)  # noqa: SLF001
    latents = backend.init_latents(plan, request)
    initial = latents.distance(condition)
    final, _steps = __import__("app.services.engine.sampler", fromlist=["x"]).sample(
        lambda x, sigma: backend.denoise(x, float(sigma), condition, request),
        latents, plan.sigmas, "euler")
    return {"initialDistance": round(initial, 4),
            "finalDistance": round(final.distance(condition), 4),
            "startNorm": round(latents.norm(), 4), "endNorm": round(final.norm(), 4)}


# ══════════════════════════════════════════════════════════════════════════
# ⑦ 引导（CFG）—— 引擎的**采样语义** ✓（不是后端的私事 ✓）
# ══════════════════════════════════════════════════════════════════════════
class _Counting(DryRunBackend):
    """数**后端被调了几次** ✓ —— 用来证明「CFG 真的把模型调用翻倍」✓（而不是只改了个参数 ✗）。"""

    def __init__(self) -> None:
        self.calls = 0

    def denoise(self, latents, sigma, condition, request):  # noqa: ANN001
        self.calls += 1
        return super().denoise(latents, sigma, condition, request)


class _SingleCondition(DryRunBackend):
    """只回**单个条件**的后端 ✓ ⇒ 管线应当**兼容**（不引导、不报错 ✓）。"""

    def encode_text(self, request):  # noqa: ANN001
        return self._condition(request)


class _NoScale:
    """能算术、但**报不出尺度**的张量 ✓（模拟不给 ``std``/``norm``/``len`` 的后端 ✓）。"""

    def __init__(self, values: list[float]) -> None:
        self.values = [float(value) for value in values]

    def __add__(self, other: "_NoScale") -> "_NoScale":
        return _NoScale([a + b for a, b in zip(self.values, other.values)])

    def __sub__(self, other: "_NoScale") -> "_NoScale":
        return _NoScale([a - b for a, b in zip(self.values, other.values)])

    def __mul__(self, factor: float) -> "_NoScale":
        return _NoScale([value * float(factor) for value in self.values])

    __rmul__ = __mul__


def case_guidance() -> None:
    from app.services.engine import guidance as g

    positive, negative = TinyTensor([1.0, 2.0]), TinyTensor([0.0, 0.0])
    guided, report = g.combine(positive, negative, 3.0)
    check("㊽ CFG 数学：`neg + s·(pos−neg)`（s=3 ⇒ 条件的 3 倍 ✓）",
          guided.head() == [3.0, 6.0] and report["guided"] is True, (guided.head(), report))
    short_circuit, report_one = g.combine(positive, negative, 1.0)
    check("㊾ s=1 ⇒ **短路返回正条件本身**（不打多余算力 ✗）且如实标「未引导」✓",
          short_circuit is positive and report_one["guided"] is False, report_one)
    no_negative, report_none = g.combine(positive, None, 7.0)
    check("㊿ 没有负条件 ⇒ 也不引导（不硬造一个 ✗）", no_negative is positive
          and report_none["guided"] is False, report_none)

    rescaled, report_rescale = g.combine(positive, negative, 4.0, rescale=1.0)
    rms = lambda value: value.norm() / (len(value) ** 0.5)  # noqa: E731
    check("①' 重标定：把尺度拉回正条件的 RMS（防 CFG 过曝 ✓）",
          report_rescale["rescaled"] is True and abs(rms(rescaled) - rms(positive)) < 1e-6,
          (round(rms(rescaled), 6), round(rms(positive), 6)))
    opaque, report_opaque = g.combine(_NoScale([1.0, 2.0]), _NoScale([0.0, 0.0]), 4.0, rescale=1.0)
    check("②' 张量**报不出尺度**（没有 std/norm/len）⇒ 明确跳过重标定并说明 ✓（不假装做过 ✗）",
          g.stat_scale(_NoScale([1.0])) is None and report_opaque["rescaled"] is False
          and "已跳过" in str(report_opaque.get("note")), report_opaque)

    config = g.GuidanceConfig(scale=7.0)
    check("③' 默认全区间施加：每一步都是 7.0 ✓",
          all(g.scale_for_step(index, 10, config) == 7.0 for index in range(10)))
    check("④' 步区间：`start=0.5` ⇒ 前半程不引导、后半程引导 ✓",
          [g.scale_for_step(index, 10, g.GuidanceConfig(scale=7.0, start=0.5)) for index in range(10)]
          == [1.0] * 5 + [7.0] * 5, "")
    ramp = g.GuidanceConfig(scale=7.0, ramp_steps=4)
    check("⑤' 斜坡：前 4 步从 1 线性升到 7（开局不被强引导锁死 ✓）",
          [round(g.scale_for_step(index, 10, ramp), 3) for index in range(5)] == [2.5, 4.0, 5.5, 7.0, 7.0],
          [round(g.scale_for_step(index, 10, ramp), 3) for index in range(5)])
    check("⑥' 默认配置（scale=1）⇒ 全程 1.0 且 `enabled=False`（不做没被要求的干预 ✓）",
          g.GuidanceConfig().enabled is False
          and all(g.scale_for_step(index, 5, g.GuidanceConfig()) == 1.0 for index in range(5)))

    # ── 管线集成 ─────────────────────────────────────────────────────────
    plain = _Counting()
    pipe.run_sync(req(steps=6), plain)
    check("⑦' 不引导（cfg=1）⇒ 每步**只调一次**模型 ✓", plain.calls == 6, plain.calls)

    guided_backend = _Counting()
    guided = pipe.run_sync(req(steps=6, guidance=g.GuidanceConfig(scale=3.0)), guided_backend)
    check("⑧' ⭐ CFG 开启 ⇒ 每步**调两次**模型（`calls == 2×步数` ✓：真打了两次，不是只改参数 ✗）",
          guided_backend.calls == 12, guided_backend.calls)
    check("⑨' 结果里如实报「生效了几步」✓", guided.guidanceSteps == 6, guided.guidanceSteps)
    check("⑩' 引导**真的改变了结果**（cfg=3 与 cfg=1 的产物不同 ✓）",
          _preview(req(steps=6, guidance=g.GuidanceConfig(scale=3.0)))
          != _preview(req(steps=6)), "")
    check("⑪' 进度事件里带 cfg 值（用户能看见斜坡在爬 ✓）",
          any("cfg=" in str(event.get("note", "")) for event in
              _events(req(steps=6, guidance=g.GuidanceConfig(scale=3.0)))), "")

    single = pipe.run_sync(req(steps=5), _SingleCondition())
    check("⑫' 后端只回**单个条件** ⇒ 兼容（不引导、不报错 ✓、guidanceSteps=0 ✓）",
          single.ok is True and single.guidanceSteps == 0, single.error)
    check("⑬' 计划里提前说清**引导的代价**（给相对耗时倍数，而不是含糊的「会很慢」✗）",
          any("引导" in item and "相对耗时" in item
              for item in pipe.build_plan(req(guidance=g.GuidanceConfig(scale=7.0))).warnings),
          pipe.build_plan(req(guidance=g.GuidanceConfig(scale=7.0))).warnings[-1:])


def _events(request: pipe.GenerationRequest) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    pipe.run_sync(request, DryRunBackend(), on_event=collected.append)
    return collected


# ══════════════════════════════════════════════════════════════════════════
# ⑧ 首帧条件（图生视频）—— 引擎算**掩码** ✓，后端套到自己的布局上 ✓
# ══════════════════════════════════════════════════════════════════════════
class _NoConditionHook:
    """只实现必需协议、**没有** ``condition_first_frame`` 的后端 ✓（用组合包一层 ✓）。

    ⚠️ 刻意**不用** ``__getattr__`` 全转发 ✗ —— 那会把干跑实现里的同名方法也转过来 ✓，
    于是这条用例就永远绿 ✗（假绿 ✗）。显式置 ``None`` 才能真的模拟"不支持" ✓。
    """

    name = "no-condition"
    synthetic = True
    condition_first_frame = None

    def __init__(self) -> None:
        self.inner = DryRunBackend()

    def __getattr__(self, item):  # noqa: ANN401
        return getattr(self.inner, item)


def case_conditioning() -> None:
    from app.services.engine import conditioning as cond

    check("①'' 掩码数学：`keep=1, fade=0` ⇒ 只有第一个潜帧来自首帧（硬切 ✓）",
          cond.first_frame_mask(5) == [1.0, 0.0, 0.0, 0.0, 0.0], cond.first_frame_mask(5))
    check("②'' 淡出：`fade=2` ⇒ 权重 1 → 0.5 → 0（线性 ✓）",
          cond.first_frame_mask(5, cond.ConditioningConfig(keep=1, fade=2)) == [1.0, 0.5, 0.0, 0.0, 0.0],
          cond.first_frame_mask(5, cond.ConditioningConfig(keep=1, fade=2)))
    check("③'' strength 整体缩放（0.5 ⇒ 全部减半 ✓）",
          cond.first_frame_mask(3, cond.ConditioningConfig(strength=0.5)) == [0.5, 0.0, 0.0])
    check("④'' 潜帧数由 `(帧数−1)//压缩比+1` 推得（124 帧 / 4 ⇒ 31 ✓）",
          cond.latent_frames_for(124, 4) == 31, cond.latent_frames_for(124, 4))
    check("⑤'' **没给压缩比 ⇒ 明确报错**（不猜 ✗ —— 猜错会把首帧放到错误的潜帧上 ✗）",
          "temporalCompression" in str(_cond_error(124, None)), _cond_error(124, None))
    check("⑥'' `keep` 超过潜帧数 ⇒ 报错（那等于整段都给首帧 ✗，几乎肯定是写错了 ✓）",
          "超过潜帧数" in str(_cond_error(5, 4, keep=9)))
    check("⑦'' `keep+fade` 放不下 ⇒ 报错 ✓", "过渡区放不下" in str(_cond_error(5, 4, keep=1, fade=9)))
    check("⑧'' strength 越界 ⇒ 报错（0..1 ✓）",
          "0..1" in str(_cond_error(5, 4, strength=1.5)))

    # ── 管线集成 ─────────────────────────────────────────────────────────
    plain = pipe.run_sync(req(steps=4), DryRunBackend())
    check("⑨'' 没给首帧 ⇒ **不出现** condition 阶段、`conditioning` 为 None（不做多余的事 ✓）",
          "condition" not in plain.stageMs and plain.conditioning is None, plain.stageMs)

    img = pipe.run_sync(req(steps=4, seconds=2.5, temporal_compression=4, first_frame="首帧.png"),
                        DryRunBackend())
    check("⑩'' ⭐ 图生视频：condition 阶段真跑 ✓ 并把掩码记录进结果（可审计 ✓）",
          img.ok is True and "condition" in img.stageMs
          and (img.conditioning or {}).get("applied") is True
          and (img.conditioning or {}).get("mask", [])[:1] == [1.0],
          (img.conditioning or {}).get("mask"))
    check("⑪'' 掩码长度 = 潜帧数 ✓（不是帧数 ✗ —— 差一点就全错 ✓）",
          len((img.conditioning or {}).get("mask") or []) == (img.conditioning or {}).get("latentFrames"),
          (len((img.conditioning or {}).get("mask") or []),
           (img.conditioning or {}).get("latentFrames")))
    check("⑫'' 首帧**真的改变了结果**（与无首帧对比 ✓）",
          _preview(req(steps=4, seconds=2.5, temporal_compression=4, first_frame="首帧.png"))
          != _preview(req(steps=4, seconds=2.5, temporal_compression=4)), "")

    unsupported = pipe.run_sync(req(steps=4, seconds=2.5, temporal_compression=4,
                                    first_frame="首帧.png"), _NoConditionHook())
    check("⑬'' ⭐⭐ 后端不支持首帧 ⇒ **明确失败**（stage=condition ✓ 且文案说清原因 ✓）"
          "—— **绝不悄悄按文生视频跑** ✗（那样用户拿到的产物与预期不符且看不出来 ✗）",
          unsupported.ok is False and (unsupported.error or {}).get("stage") == "condition"
          and "未实现" in str((unsupported.error or {}).get("message")),
          unsupported.error)

    # ── 关键帧**下标规划**（纯函数 ✓ 无 torch 也能验 ✓）──
    check("⑬⁗ ⭐ 首帧 ⇒ 下标 **0**；尾帧 ⇒ 下标 **frames−1** ✓（同一套 `cond` 段 ✓ 只换下标 ✓）",
          keyframes_mod.plan_keyframes(21, "首.png", None) == [(0, "首.png")]
          and keyframes_mod.plan_keyframes(21, None, "尾.png") == [(20, "尾.png")]
          and keyframes_mod.plan_keyframes(21, "首.png", "尾.png") == [(0, "首.png"), (20, "尾.png")],
          keyframes_mod.plan_keyframes(21, "首.png", "尾.png"))
    check("⑬⁗′ 两个都不给 ⇒ 空（纯文本情形 ✓，不算错 ✓）",
          keyframes_mod.plan_keyframes(21, None, None) == [])
    try:
        keyframes_mod.plan_keyframes(1, None, "尾.png")
    except keyframes_mod.KeyframePlanError as error:
        one_frame_error = str(error)
    else:
        one_frame_error = ""
    check("⑬⁗″ ⭐ frames=1 给尾帧 ⇒ **报错**（首尾同一位置 ✓✗ —— 悄悄退化成首帧就是丢尾帧 ✗）",
          "2 帧" in one_frame_error, one_frame_error)

    last_unsupported = pipe.run_sync(
        req(steps=4, seconds=2.5, temporal_compression=4, last_frame="尾帧.png"), DryRunBackend())
    check("⑬''' ⭐⭐ 单流后端给**尾帧** ⇒ **明确失败**（stage=condition、文案点名尾帧 ✓）"
          "—— 单流的掩码只能混第 0 潜帧 ✓ ⇒ 硬做会把**尾帧**放到**开头** ✓✗，比报错难查得多 ✗",
          last_unsupported.ok is False
          and (last_unsupported.error or {}).get("stage") == "condition"
          and "尾帧" in str((last_unsupported.error or {}).get("message")),
          last_unsupported.error)
    check("⑭'' 没给压缩比就要求首帧 ⇒ 也归到 condition 阶段失败（而不是 500 ✗）",
          (pipe.run_sync(req(steps=4, seconds=2.5, first_frame="首帧.png"),
                         DryRunBackend()).error or {}).get("stage") == "condition", "")
    check("⑮'' 计划里给出**相对耗时系数**（步数×采样器×引导 ✓ 可精确算 ⇒ 给系数不给秒数 ✗）",
          pipe.build_plan(req(steps=6, sampler="heun",
                              guidance=__import__("app.services.engine.guidance",
                                                  fromlist=["g"]).GuidanceConfig(scale=5))).cost_factor
          == 24.0,
          pipe.build_plan(req(steps=6, sampler="heun")).cost_factor)


def _cond_error(frames: int, compression: int | None, **config: Any) -> Exception | None:
    from app.services.engine import conditioning as cond

    try:
        cond.first_frame_mask(cond.latent_frames_for(frames, compression),
                              cond.ConditioningConfig(**config))
    except (cond.ConditioningError, ValueError) as err:
        return err
    return None


# ══════════════════════════════════════════════════════════════════════════
# ③ 取消（长任务必须能停 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_cancel() -> None:
    immediate = pipe.run_sync(req(), DryRunBackend(), cancel=lambda: True)
    check("㉕ 一开始就取消 ⇒ cancelled=True（**不是** failed ✗，前端要分开显示 ✓）",
          immediate.cancelled is True and immediate.ok is False
          and (immediate.error or {}).get("cancelled") is True, immediate.error)
    check("㉖ 取消发生在 encode 阶段（还没白跑任何重活 ✓）",
          (immediate.error or {}).get("stage") == "encode", immediate.error)

    state = {"steps": 0}

    def on_event(event: dict[str, Any]) -> None:
        if event["stage"] == "sample" and "step" in event:
            state["steps"] = max(state["steps"], int(event["step"]))

    mid = pipe.run_sync(req(steps=20), DryRunBackend(), on_event=on_event,
                        cancel=lambda: state["steps"] >= 3)
    check("㉗ 采样中取消 ⇒ 报出**停在第几步**（用户最想知道的就是这个 ✓）",
          mid.cancelled is True and (mid.error or {}).get("stage") == "sample"
          and (mid.error or {}).get("step") == 3, mid.error)
    check("㉘ 取消也保留已完成阶段的耗时 ✓（plan/encode/init 都记了）",
          all(stage in mid.stageMs for stage in ("plan", "encode", "init", "sample")),
          mid.stageMs)
    check("㉙ 取消后**没有** decode / write 耗时（真停了，不是只标记 ✗）",
          "decode" not in mid.stageMs and "write" not in mid.stageMs, mid.stageMs)
    #: ⭐⭐ **取消时「真跑过几步」必须进 ``sampleSteps``** ✗✗（2026-11-08 补 ✗）：原先它只在
    #:    采样器**正常返回**那一处赋值 ✓ ⇒ 中断一律停在 **0** ✓✗ —— 而同一份回执里
    #:    ``stageMs.sample`` 是真实秒数 ✓、㉗ 又刚断言 ``error.step == 3`` ✓（``on_step`` 在每步
    #:    **算完之后**才回调 ✓ ⇒ 步号＝已跑完步数 ✓）⇒ 「跑了 3 步」与「0 步」并存且无人解释 ✓。
    #:    消费方照 0 走：``chain`` 逐段累加（被取消那段记 0 ✓✗）、``bridge`` 把 0 写进日志 ✓。
    cancel_card = mid.to_dict()
    check("㉙¹ ⭐⭐ 采样中取消 ⇒ ``sampleSteps`` 记**真跑过的 3 步** ✓ 不是 0 ✗"
          "（与 ``error.step`` 同一个数 ✓、``stepsNote`` 写明 3/20 ✓）"
          "—— 「跑了 3 步被取消」和「一步没跑」在记录里不该同形 ✓",
          mid.sampleSteps == 3
          and (mid.error or {}).get("step") == 3
          and "3 / 20" in str(cancel_card.get("stepsNote") or "")
          and "取消" in str(cancel_card.get("stepsNote") or ""),
          {"sampleSteps": mid.sampleSteps,
           "stepsNote": cancel_card.get("stepsNote"), "error": mid.error})
    check("㉙² 一开始就取消（`error.stage == encode`）⇒ **不**伪造步数 ✓"
          "（``sampleSteps`` 仍 0 ✓、没有 ``stepsNote`` ✓ —— 采样真的一步都没开始 ✓）",
          immediate.sampleSteps == 0 and not immediate.to_dict().get("stepsNote"),
          {"sampleSteps": immediate.sampleSteps,
           "stepsNote": immediate.to_dict().get("stepsNote")})


# ══════════════════════════════════════════════════════════════════════════
# ④ 错误归因（失败要能一眼看出是哪个阶段哪一步 ✓）
# ══════════════════════════════════════════════════════════════════════════
class _FailingDecode(DryRunBackend):
    def decode(self, latents, plan, request):  # noqa: ANN001
        raise RuntimeError("显存不足（模拟 ✓）")


class _FailingAtStep(DryRunBackend):
    def denoise(self, latents, sigma, condition, request):  # noqa: ANN001
        if getattr(self, "calls", 0) >= 5:
            raise ValueError("模型调用失败（模拟第 5 步 ✓）")
        self.calls = getattr(self, "calls", 0) + 1
        return super().denoise(latents, sigma, condition, request)


class OutOfMemoryError(Exception):  # noqa: A001 —— 刻意与 torch 的类名一致 ✓（见 docstring ✓）
    """模拟 ``torch.cuda.OutOfMemoryError`` ✓（**类名必须一致** ✓ —— `_fail` 是按类名判的 ✓✗）。

    ⚠️ 不用真 torch 类 ✗：`pipeline` 是**零依赖**的 ✓（torch 没装也要能跑这条 ✓），
    而判据要验的是「这个类名会不会被单独点名」✓，不是 torch 的行为 ✓。
    ⚠️ 它遮蔽了 3.11+ 的内建同名异常 ✓ —— 本模块不用那个 ✓（内建 ``MemoryError`` 才是常用的 ✓）。
    """


class _VramExhausted(DryRunBackend):
    def denoise(self, latents, sigma, condition, request):  # noqa: ANN001
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


def case_error_attribution() -> None:
    failed = pipe.run_sync(req(), _FailingDecode())
    check("㉚ decode 抛错 ⇒ 归因到「解码」阶段 + 保留原类型（能据此排障 ✓）",
          failed.ok is False and (failed.error or {}).get("stage") == "decode"
          and "解码" in str((failed.error or {}).get("message"))
          and (failed.error or {}).get("type") == "RuntimeError", failed.error)
    check("㉛ 失败前的阶段耗时仍在（知道跑到哪儿炸的 ✓）",
          all(stage in failed.stageMs for stage in ("plan", "encode", "init", "sample")),
          failed.stageMs)

    mid = pipe.run_sync(req(steps=10), _FailingAtStep())
    check("㉜ 采样到第 5 步炸 ⇒ 报出**步号**（不是只说「采样失败」✗）",
          (mid.error or {}).get("stage") == "sample" and (mid.error or {}).get("step") == 5,
          mid.error)
    check("㉝ 采样失败的文案带「去噪采样」字样（人话，不是异常类名 ✓）",
          "去噪采样" in str((mid.error or {}).get("message")), mid.error)
    #: ⭐⭐ 失败那条**同样是**「读数缺失被当成读数说没有」✗✗（2026-11-08 补 ✗）：OOM/模型报错
    #:    多半发生在跑到一半 ✓ —— 而 ``sampleSteps`` 原先只在采样器正常返回时赋值 ✓ ⇒ 记 0 ✓✗，
    #:    于是「第 5 步炸的（前面 5 步的白工是真花了钱 ✓）」在回执里与「采样没开始」同形 ✓。
    broken_card = mid.to_dict()
    check("㉜¹ ⭐⭐ 采样第 5 步失败 ⇒ ``sampleSteps`` 记**真跑过的 5 步** ✓ 不是 0 ✗"
          "（``stepsNote`` 写明 5/10 ✓ 且带「失败」字样 ✓）"
          "—— 前面那 5 步是**真花掉的算力** ✓，记 0 会让人以为白跑的是 0 步 ✓",
          mid.sampleSteps == 5
          and "5 / 10" in str(broken_card.get("stepsNote") or "")
          and "失败" in str(broken_card.get("stepsNote") or ""),
          {"sampleSteps": mid.sampleSteps, "stepsNote": broken_card.get("stepsNote")})

    bad_plan = pipe.run_sync(req(prompt=""), DryRunBackend())
    check("㉞ plan 阶段就失败 ⇒ **没有**后续阶段耗时（根本没开始跑 ✓）",
          (bad_plan.error or {}).get("stage") == "plan"
          and set(bad_plan.stageMs) == {"plan"}, (bad_plan.error, bad_plan.stageMs))

    # ⭐ OOM **必须点名可行动的路** ✗✗（2026-10-29 加 ✓）
    #    ⚠️ 原生报错是 ``CUDA out of memory. Tried to allocate 2.00 GiB`` ✓ —— 它只说明分配失败 ✓，
    #    不会告诉你「720×1280 只有 `int8_offload` 这一档装得下」✓✗，而本仓**自己知道**那句话 ✓
    #    （三处实测写在 `h3_form.place_module` 与 `TorchBackend.__init__` 的注释里 ✓）。
    #    这是本仓唯一一个「答案在注释里、用户看不到」的失败 ✓✗ ⇒ 必须进 message + 结构化字段 ✓。
    oom = pipe.run_sync(req(steps=4), _VramExhausted())
    oom_error = oom.error or {}
    check("㉞′ ⭐ OOM ⇒ 错误里**点名 `int8Offload`** ✓✗（老代码只有英文那句 ⇒ "
          "用户照着排障只能瞎试：换卡 / 降分辨率都是猜 ✓✗）",
          "int8Offload" in str(oom_error.get("message")),
          str(oom_error.get("message"))[:160])
    check("㉞″ ⭐ OOM ⇒ 结构化字段 ``vramExhausted=True`` ✓（前端要能**直接判**，"
          "不许解析中文 ✓ —— 与「取消」那条同一个口径 ✓）",
          oom_error.get("vramExhausted") is True
          and oom_error.get("type") == "OutOfMemoryError", oom_error)
    check("㉞‴ 三条路都写全了（① 开 offload 省 17.94 GiB ／ ② 降分辨率帧数 ／ ③ 换大卡 ✓）"
          "且说明「已开还爆就只剩 ②③」✓（否则会让人反复去开一个已经开着的开关 ✗）",
          all(word in str(oom_error.get("message"))
              for word in ("17.94 GiB", "megapixels", "更大的卡", "已经开着")),
          str(oom_error.get("message"))[:400])
    check("㉞⁗ ⚠️ **非** OOM 的失败不许被当成显存问题 ✗✗（否则真因会被这三条建议盖住 ✓✗）",
          "vramExhausted" not in (failed.error or {})
          and "int8Offload" not in str((failed.error or {}).get("message")),
          failed.error)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 对外的字典形状（路由/落库要用 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_serialization() -> None:
    result = pipe.run_sync(req(steps=4), DryRunBackend())
    data = result.to_dict()
    check("㉟ to_dict 用 camelCase（与前端约定一致 ✓）且字段齐",
          {"ok", "cancelled", "backend", "synthetic", "sampleSteps", "totalMs",
           "stageMs", "plan", "outputs", "error"} <= set(data), sorted(data))
    check("㊱ plan 字典含用户要看的关键数字 ✓",
          {"width", "height", "frames", "fps", "steps", "sigmas", "durationSeconds"}
          <= set(data["plan"] or {}), sorted((data["plan"] or {})))
    check("㊲ 干跑结果里 synthetic=True 一路传到对外字典 ✓（前端可据此打标 ✓）",
          data["synthetic"] is True and data["plan"]["frames"] == 124)


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 路由：管线要**真能被调用**（干跑也走服务 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_api() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    planned = client.post("/api/v1/engine/plan", json={
        "prompt": "雨夜霓虹街头", "seconds": 5, "steps": 8, "aspectRatio": "16:9",
        "weightsBytes": 19 * 1024 ** 3})
    data = planned.json().get("data") or {}
    check("㊳ POST /engine/plan 收**前端字段名**（aspectRatio ✓）并给出具体数字",
          planned.status_code == 200 and (data.get("width"), data.get("height")) == (1088, 608)
          and data.get("frames") == 124 and len(data.get("sigmas") or []) == 9,
          (planned.status_code, data.get("width"), data.get("frames")))
    check("㊴ 计划里带显存估算警告（weightsBytes 生效 ✓）",
          any("GiB" in item for item in data.get("warnings") or []), data.get("warnings"))

    rejected = client.post("/api/v1/engine/plan", json={"prompt": "x", "sampler": "nope"})
    check("㊵ 采样器写错 ⇒ **HTTP 400**（不是 500 ✗，也不是等到跑一半 ✗）",
          rejected.status_code == 400 and "可用" in str(rejected.json().get("message")),
          rejected.json())

    run = client.post("/api/v1/engine/dry-run",
                      json={"prompt": "雨夜霓虹街头", "steps": 4, "includeEvents": True})
    payload = run.json().get("data") or {}
    check("㊶ POST /engine/dry-run 走完整管线（200 + ok ✓）", run.status_code == 200
          and payload.get("ok") is True, (run.status_code, payload.get("error")))
    check("㊷ ⭐ 干跑结果对外**必须**标 synthetic 且 videoPath 为空（前端不许当真结果展示 ✗）",
          payload.get("synthetic") is True and (payload.get("outputs") or {}).get("videoPath") is None,
          payload.get("outputs"))
    check("㊸ includeEvents ⇒ 回进度事件（前端进度条直接用 ✓）",
          isinstance(payload.get("events"), list) and len(payload["events"]) > 4
          and payload["events"][0]["stage"] == "plan", len(payload.get("events") or []))
    check("㊹ 不带 includeEvents ⇒ 不回事件（默认响应体别太大 ✓）",
          "events" not in (client.post("/api/v1/engine/dry-run",
                                       json={"prompt": "x", "steps": 2}).json().get("data") or {}))

    # ── 引导经 API 生效 ✓（前端直接写 `cfg` ✓）────────────────────────────
    guided_plan = client.post("/api/v1/engine/plan", json={
        "prompt": "雨夜霓虹街头", "steps": 8, "cfg": 7})
    guided_data = guided_plan.json().get("data") or {}
    check("㊺ 前端字段 `cfg` 直接生效 ✓ 且计划里**提前警告代价**（每步多次调用 ✗）",
          guided_plan.status_code == 200
          and any("引导" in item for item in guided_data.get("warnings") or []),
          guided_data.get("warnings"))
    guidance_payload = client.post("/api/v1/engine/dry-run",
                                   json={"prompt": "x", "steps": 4, "cfg": 3}).json().get("data") or {}
    check("㊻ 干跑里 `guidanceSteps` 如实回报（4 步全生效 ✓）",
          guidance_payload.get("guidanceSteps") == 4, guidance_payload.get("guidanceSteps"))
    check("㊼ 坏引导参数 ⇒ **400**（不是 500 ✗）",
          client.post("/api/v1/engine/plan",
                      json={"prompt": "x", "cfg": "很大"}).status_code == 400)

    # ── 首帧条件经 API ✓（前端给 firstFrame + temporalCompression ✓）──────────
    img2vid = client.post("/api/v1/engine/dry-run", json={
        "prompt": "雨夜霓虹街头", "steps": 3, "seconds": 2.5, "temporalCompression": 4,
        "firstFrame": "首帧.png", "conditionFade": 1}).json().get("data") or {}
    check("㊽ 图生视频经 API 生效 ✓（conditioning.applied + 掩码长度 = 潜帧数 ✓）",
          (img2vid.get("conditioning") or {}).get("applied") is True
          and len((img2vid.get("conditioning") or {}).get("mask") or [])
          == (img2vid.get("conditioning") or {}).get("latentFrames"),
          img2vid.get("conditioning"))
    missing_compression = client.post("/api/v1/engine/dry-run", json={
        "prompt": "x", "steps": 2, "seconds": 2.5, "firstFrame": "首帧.png"}).json().get("data") or {}

    # ── ㊾ 超清计划（2026-09-24 接线 ✓）：规划层此前**只有自检在调** ✗✗ ⇒ 现在业务面拿得到 ✓ ──
    no_weights = client.post("/api/v1/engine/upscale-plan", json={"targetScale": 2})
    nw = no_weights.json().get("data") or {}
    check("㊾ 没给放大器权重 ⇒ 回退普通模式 ✓ 且 ``fallbackReason`` **必须给** ✗✗"
          "（静默降级 = 你以为超清开着 ✓✗）",
          no_weights.status_code == 200 and nw.get("mode") == "normal"
          and nw.get("usesUpscaler") is False and bool(nw.get("fallbackReason")), nw)

    weights = Path(tempfile.mkdtemp(prefix="upscaler_")) / "h3_upscaler.safetensors"
    contract = {
        "format": "minimax_h3_clean_latent_upscaler_v3_factorized_attention",
        "base_config": {"in_channels": 24, "hidden_channels": 64, "num_blocks": 4,
                        "refine_channels": 32, "refine_blocks": 2, "temporal_kernel": 3},
        "config": {"width": 64, "blocks": 2, "heads": 4, "window": 8, "mlp_ratio": 2},
        "strict_latent_only": True,
    }
    header = json.dumps({"__metadata__": {"metadata": json.dumps(contract)}}).encode()
    weights.write_bytes(len(header).to_bytes(8, "little") + header)

    check("㊿ ⭐有放大器却 >2 倍率**不给 maxTile ⇒ 400** ✗✗（本仓**不猜**安全块大小 ✓："
          "大块解码出 NaN/Inf ✓✗；⚠️ 没权重那条走的是「回退普通」✓，**不是**这条 ✗）",
          client.post("/api/v1/engine/upscale-plan",
                      json={"targetScale": 4, "filePath": str(weights)}).status_code == 400)

    tiled = client.post("/api/v1/engine/upscale-plan", json={
        "targetScale": 4, "maxTile": 128, "frame": [1088, 608], "filePath": str(weights)})
    td = tiled.json().get("data") or {}
    tiles = td.get("tiles") or []
    check("㊿′ 给了权重（真 safetensors 头 ✓）+ 倍率/分块 ⇒ ``tiled-ai-2x`` ✓（内嵌契约被真读 ✓）",
          tiled.status_code == 200 and td.get("mode") == "tiled-ai-2x" and td.get("contractError") is None
          and all(step for step in (td.get("steps") or [])), (tiled.status_code, td.get("contractError")))
    check("㊿″ 分块是画面的**精确划分** ✓（各块宽之和 == 1088 ✓、高之和 == 608 ✓ —— "
          "面积对得上**不算** ✗：错位/重叠的组合太多 ✓✗）",
          sum(int(t[2]) for t in tiles if int(t[1]) == 0) == 1088
          and sum(int(t[3]) for t in tiles if int(t[0]) == 0) == 608
          and len(tiles) > 1, tiles)

    half = client.post("/api/v1/engine/upscale-plan",
                       json={"targetScale": 1.5, "filePath": str(weights)}).json().get("data") or {}
    check("㊿‴ ⭐ 1.5× ⇒ 「**先合法的 2× 再缩放**」✗✗（直接提交 1.5× 会被放大器拒 ✓）",
          half.get("mode") == "ai-2x-resize" and len(half.get("steps") or []) == 2, half)

    # ── 二采参数（2026-09-25 ✓）：口径已核 ✓ ⇒ 走真链路验一遍「API → 计划 → 请求」✗✗ ──
    refine_plan = client.post("/api/v1/engine/upscale-plan", json={
        "targetScale": 2, "filePath": str(weights), "refineSteps": 4, "refineDenoise": 0.4})
    rp = refine_plan.json().get("data") or {}
    check("⑤¹⁵ ⭐ 给了二采参数 ⇒ ``secondPass=true`` ✓ + notes 写清**掩码口径**（video 1 / audio 0 ✓✗）；"
          "⚠️ ``refineSteps`` 必须**保留原值 4** ✗✗ 而派生量单列（``refineTailSteps=10`` ✓）"
          "—— 曾把派生值写回原值 ⇒ 下游再算一次就**越滚越长**（4→10→25 ✓✗）",
          refine_plan.status_code == 200 and rp.get("secondPass") is True
          and rp.get("refineSteps") == 4 and rp.get("refineTailSteps") == 10
          and any("video 1 / audio 0" in note for note in (rp.get("notes") or [])), rp)
    check("⑤¹⁶ 只给一半（有 steps 没 denoise）⇒ **400 带理由** ✗（默认值在编译层 ⇒ 不猜 ✓✗）",
          client.post("/api/v1/engine/upscale-plan",
                      json={"targetScale": 2, "filePath": str(weights), "refineSteps": 4}
                      ).status_code == 400)
    planned_refine = client.post("/api/v1/engine/plan", json={
        "prompt": "x", "steps": 4, "upscale": {**rp, "usesUpscaler": True}})
    refine_warn = ((planned_refine.json().get("data") or {}).get("warnings") or [])
    check("⑤¹⁷ ⭐ 计划经请求体送到管线 ✓（``upscale.refineSteps`` 到得了 ✓）且**提前说清**"
          "二采与音频锁定 ✓（不是跑起来才知道 ✓✗）",
          planned_refine.status_code == 200
          and ((planned_refine.json().get("data") or {}).get("upscale") or {}).get("refineSteps") == 4
          and any("音频流锁定" in note or "6×" in note for note in refine_warn),
          (planned_refine.status_code, refine_warn))

    # ── 派生量跟着搬（2026-11-08 补 ✗）─────────────────────────────────────────
    # ⚠️ `plan.upscale` 是**重建出来的**（HTTP dict → UpscalePlan → to_dict ✓）。早先只搬
    #    refineSteps/refineDenoise ⇒ 派生量 `refineTailSteps` 掉成 0 ⇒ 回执写「尾部 0 步」
    #    而张量层真跑了 10 步（它自己重算 ✓）—— **回执与实跑自相矛盾** ✓✗。
    planned_upscale = (planned_refine.json().get("data") or {}).get("upscale") or {}
    check("⑤¹⁸ ⭐⭐ 派生量 ``refineTailSteps`` 经 HTTP 重建**仍在**（=10 ✓；掉成 0 就是"
          "「回执写 0 步、实跑 10 步」✓✗）",
          planned_upscale.get("refineTailSteps") == 10
          and planned_upscale.get("refineSteps") == 4, planned_upscale)
    mismatch = client.post("/api/v1/engine/plan", json={
        "prompt": "x", "steps": 4, "upscale": {**rp, "refineTailSteps": 3}})
    check("⑤¹⁹ 手改的 ``refineTailSteps`` 与公式**不符** ⇒ **400** ✗✗（两处口径必有一处错 ✓；"
          "不静默采纳 ✓、也不静默改成公式值 ✓）",
          mismatch.status_code == 400
          and "对不上" in json.dumps(mismatch.json(), ensure_ascii=False),
          (mismatch.status_code, mismatch.json()))
    half_refine = client.post("/api/v1/engine/plan", json={
        "prompt": "x", "steps": 4,
        "upscale": {"mode": "ai-2x", "scale": 2.0, "refineSteps": 4}})
    check("⑤²⁰ ⭐ 只给 ``refineSteps`` 不给 ``refineDenoise`` ⇒ **在 plan 阶段就 400** ✗✗"
          "（早先要等**采样+放大器全跑完**、进二采才 ValueError ✓✗ —— 计划层明明早就拦 ✓）",
          half_refine.status_code == 400
          and "refineDenoise" in json.dumps(half_refine.json(), ensure_ascii=False),
          (half_refine.status_code, half_refine.json()))
    # ⭐⭐ `mode` 与 `scale` 的相容性也在**装模型之前**核 ✗✗（同判据的双保险在
    #    `torch_backend._plan_target_size` ✓ —— 这里只是让它早一步、并挡住手写的计划 ✓）。
    for bad_scale in (1.0, 3.0):
        bad = client.post("/api/v1/engine/plan", json={
            "prompt": "x", "steps": 4,
            "upscale": {"mode": "ai-2x-resize", "scale": bad_scale}})
        check(f"⑤²¹ `ai-2x-resize` + {bad_scale:g}× ⇒ **plan 阶段 400** ✗✗"
              f"（1× = 放大再缩回原尺寸：尺寸不变、耗时 ≈6× ✓✗；>2× = 插值冒充分块放大 ✓✗）",
              bad.status_code == 400
              and "自相矛盾" in json.dumps(bad.json(), ensure_ascii=False),
              (bad_scale, bad.status_code, bad.json()))
    ok_half = client.post("/api/v1/engine/plan", json={
        "prompt": "x", "steps": 4, "upscale": {**half}})
    check("⑤²² 1.5×（计划层真会产出的那档）⇒ **不误拦** ✓（判据不许见谁都喊 ✗）",
          ok_half.status_code == 200
          and ((ok_half.json().get("data") or {}).get("upscale") or {}).get("mode") == "ai-2x-resize",
          (ok_half.status_code, ok_half.json().get("data")))

    ghost = client.post("/api/v1/engine/upscale-plan", json={
        "targetScale": 2, "filePath": str(weights.with_name("nope.safetensors"))})
    check("㊿⁗ 权重文件不存在 ⇒ **200 + 回退 + 带理由** ✓（不是 500 ✗、也不是静默当没配 ✗）",
          ghost.status_code == 200 and (ghost.json().get("data") or {}).get("mode") == "normal"
          and bool((ghost.json().get("data") or {}).get("contractError")),
          (ghost.status_code, (ghost.json().get("data") or {}).get("contractError")))
    # ── ⑤ 超清二采接进管线（2026-09-24 接 ✓）：计划在请求里 ✓、阶段真跑 ✓、音频流锁定 ✓ ──
    ai2x = {"mode": "ai-2x", "scale": 2.0, "steps": ["AI 2×（潜空间放大器 ✓）", "带掩码的二次去噪 ✓"],
            "notes": ["输出分辨率翻倍 ✓、耗时 ≈6× ✓"], "tiles": [], "fallbackReason": None}
    planned_up = client.post("/api/v1/engine/plan",
                             json={"prompt": "x", "steps": 4, "upscale": ai2x})
    up_plan = (planned_up.json().get("data") or {}).get("upscale") or {}
    check("⑤⁵ 请求带超清计划 ⇒ 计划里**如实带上** ✓ 且**提前说代价**（≈6× ✓ 与音频流锁定 ✗）",
          planned_up.status_code == 200 and up_plan.get("mode") == "ai-2x"
          and any("6×" in item or "音频流锁定" in item
                  for item in ((planned_up.json().get("data") or {}).get("warnings") or [])),
          (planned_up.status_code, up_plan))

    # ⭐⭐ **耗时口径不许抄到别的模式上** ✗✗（2026-11-08 补 ✗）：≈6× 是 ``ai-2x``
    #    （放大器**原样 2×**）的逆向读数 ✓，只写在 ``upscale.plan_upscale`` 给该模式的
    #    ``notes`` 里 ✓；``ai-2x-resize``（先 2× 再缩）与 ``tiled-ai-2x`` **都没有**这个读数 ✓✗。
    #    原先 ``_finish_plan`` 对所有 ``usesUpscaler`` 一律印「≈6×」✓✗ ⇒ 1.5× 的用户被告知
    #    一个没有出处的倍率 ✓，而它读起来正像实测 ✓。
    up_warn = ((planned_up.json().get("data") or {}).get("warnings") or [])
    check("⑤²⁷ 倍率与耗时**逐字来自计划层** ✓（``ai-2x`` 那句 ≈6× 原样引用，不在这层另编 ✗）",
          any("≈6×" in item and "输出分辨率翻倍" in item for item in up_warn), up_warn)
    resize_up = client.post("/api/v1/engine/plan", json={
        "prompt": "x", "steps": 4,
        "upscale": {"mode": "ai-2x-resize", "scale": 1.5,
                    "steps": ["AI 2×（合法的放大器倍率 ✓）", "高质量缩放 2× → 1.5× ✓"],
                    "notes": ["⚠️ 放大器只会做**空间 2×** ✗ ⇒ 所以是「先 2× 再缩」✓"],
                    "tiles": [], "fallbackReason": None}})
    resize_warn = ((resize_up.json().get("data") or {}).get("warnings") or [])
    check("⑤²⁸ ⭐⭐ ``ai-2x-resize``（1.5×）**不许**被印上 ≈6× ✗✗（那是 2× 那条的读数 ✓）"
          "—— 而要说清**本模式没有耗时读数** ✗（不猜一个像实测的数 ✓）",
          resize_up.status_code == 200
          and not any("6×" in item for item in resize_warn)
          and any("没有逆向耗时读数" in item for item in resize_warn), resize_warn)

    # ⭐⭐ **``costFactor`` 只算采样** ✗✗（2026-11-08 补 ✗）：它与 ``upscale`` 并排躺在同一份
    #    回执里 ✓ —— 开 2× 超清时「×1.00」读起来像整跑耗时 ✓✗，而放大器那一段 ≈6× 压根没进这个数 ✓。
    #    现在用两格说清覆盖范围 ✓（不编一个"总耗时"✗：放大器没有可精确算的系数 ✓）。
    up_cost = (planned_up.json().get("data") or {})
    check("⑤²⁹ ⭐⭐ ``costFactor`` 点明**只算采样** ✓ 且开超清时列进 ``costExcludes`` ✗✗"
          "（否则「×1.00」与「≈6×」并排、读的人不知道哪个是整跑代价 ✓）",
          up_cost.get("costCovers") == ["sample"]
          and up_cost.get("costExcludes") == ["upscale"],
          (up_cost.get("costCovers"), up_cost.get("costExcludes")))
    plain_plan = client.post("/api/v1/engine/plan", json={"prompt": "x", "steps": 4})
    check("⑤²⁹′ 没有超清 ⇒ ``costExcludes`` 为空 ✓（判据不许见谁都喊 ✗）",
          (plain_plan.json().get("data") or {}).get("costExcludes") == [],
          (plain_plan.json().get("data") or {}).get("costExcludes"))
    guided_up = client.post("/api/v1/engine/plan",
                            json={"prompt": "x", "steps": 4, "cfg": 5, "upscale": ai2x})
    guided_warn = ((guided_up.json().get("data") or {}).get("warnings") or [])
    check("⑤³⁰ 引导那条「相对耗时 ×N」必须点明**只算采样** ✗✗"
          "（同一份 ``warnings`` 里下面还有超清 ≈6× ✓ —— 不点明就是把两个口径并排放 ✓）",
          any("只算采样" in item and "不含下面那段超清" in item for item in guided_warn),
          guided_warn)

    normal = client.post("/api/v1/engine/plan", json={
        "prompt": "x", "steps": 4,
        "upscale": {"mode": "normal", "scale": 1.0, "fallbackReason": "放大器权重缺失 ✓"}})
    check("⑤⁶ ⭐ 计划说**不启用** ⇒ 必须**说清为什么** ✗✗（静默降级 = 用户以为超清开着 ✓✗）",
          normal.status_code == 200
          and any("未启用" in item and "放大器权重缺失" in item
                  for item in ((normal.json().get("data") or {}).get("warnings") or [])),
          (normal.json().get("data") or {}).get("warnings"))

    check("⑤⁷ 编一个不存在的 mode ⇒ **400 并列出合法的** ✗（不静默回落普通模式 ✓✗）",
          client.post("/api/v1/engine/plan",
                      json={"prompt": "x", "steps": 2, "upscale": {"mode": "8x-magic"}}
                      ).status_code == 400)

    refined = client.post("/api/v1/engine/dry-run",
                          json={"prompt": "超清用例", "steps": 3, "seconds": 2.5,
                                "temporalCompression": 4, "upscale": ai2x}).json().get("data") or {}
    details = (refined.get("refine") or {}).get("details") or {}
    check("⑤⁸ ⭐⭐ 干跑真跑**二采阶段** ✓（``stageMs`` 里有 refine ✓、结果里带实况 ✓）",
          refined.get("ok") is True and (refined.get("refine") or {}).get("applied") is True
          and "refine" in (refined.get("stageMs") or {}), refined.get("refine"))
    check("⑤⁹ ⭐⭐ **音频流逐位不变** ✗✗（只换视频那一槽 ✓；重采音频会把音轨弄坏而画面看着正常 ✓）",
          details.get("audioIdentical") is True and details.get("isContainer") is True,
          details)
    check("⑤¹⁰ ⭐ 视频流空间 2× ✓ 而**时间维不变** ✗✗（时间插值会让动作速率变错 ✓）",
          details.get("videoShapeAfter", [0] * 5)[3] == details.get("videoShapeBefore", [0] * 5)[3] * 2
          and details.get("videoShapeAfter", [0] * 5)[4] == details.get("videoShapeBefore", [0] * 5)[4] * 2
          and details.get("temporalUnchanged") is True,
          (details.get("videoShapeBefore"), details.get("videoShapeAfter")))
    check("⑤¹¹ 容器类型**保住了** ✓（不许退化成裸 list ✗✗ —— 那会让下游炸得莫名其妙 ✓）",
          details.get("containerType") == "TinyAVLatents", details.get("containerType"))

    # ⭐⭐ **实测倍率必须进回执** ✗✗（2026-11-08 补 ✗）：``scale`` 一直是**计划值** ✓，
    #    而没有任何一格写「实跑出来几倍」✓✗ ⇒ 又回到「计划层有真源、执行结果不自证」✓。
    #    现在两数并排：``scale``（计划）+ ``scaleActual``（实测，取自后端自述的尺寸 ✓）。
    refine_block = refined.get("refine") or {}
    check("⑤²³ ⭐⭐ 回执**并排**给计划倍率与实测倍率 ✓（实测是尺寸比算出来的 ✓，不是照抄计划 ✓）；"
          "两者一致时**不**报 ``scaleMismatch`` ✓",
          refine_block.get("scale") == 2.0 and refine_block.get("scaleActual") == 2.0
          and "scaleMismatch" not in refine_block, refine_block)

    plain = client.post("/api/v1/engine/dry-run",
                        json={"prompt": "普通模式", "steps": 2}).json().get("data") or {}
    check("⑤¹² 没要超清 ⇒ **不跑二采阶段** ✓（可选阶段的正面情形要保住 ✓✗）",
          plain.get("refine") is None and "refine" not in (plain.get("stageMs") or {}),
          (plain.get("refine"), list((plain.get("stageMs") or {}))))

    class _NoLock(DryRunBackend):
        refineNote = {"locksAudio": False, "note": "（故意的 ✓）"}

    no_lock = pipe.run_sync(pipe.GenerationRequest(prompt="x", steps=2,
                                                   upscale=upscale_mod.UpscalePlan(
                                                       mode="ai-2x", scale=2.0)),
                            _NoLock())
    check("⑤¹³ ⭐⭐ 后端**没自述锁定音频流** ⇒ **明确拒绝**跑二采 ✗✗（不许重采音频还说成功 ✓）",
          no_lock.ok is False and (no_lock.error or {}).get("stage") == "refine"
          and "锁定音频流" in str((no_lock.error or {}).get("message")),
          no_lock.error)

    class _NoRefine(DryRunBackend):
        refine_latents = None      # type: ignore[assignment]

    no_hook = pipe.run_sync(pipe.GenerationRequest(prompt="x", steps=2,
                                                   upscale=upscale_mod.UpscalePlan(
                                                       mode="ai-2x", scale=2.0)),
                            _NoRefine())
    class _NoDetails(DryRunBackend):
        """⚠️ 跑得完、也自述「锁定音频流」✓，唯独**不报实况** ✗ —— 正是要被拦的那种。"""

        refineDetails = None

        def refine_latents(self, latents: Any, plan: Any, request: Any, *,
                           condition: Any = None) -> Any:
            return latents      # 不写 refineDetails ⇒ 回执只剩计划倍率

    no_details = pipe.run_sync(pipe.GenerationRequest(prompt="x", steps=2,
                                                     upscale=upscale_mod.UpscalePlan(
                                                         mode="ai-2x", scale=2.0)),
                               _NoDetails())
    check("⑤¹⁴′ ⭐⭐ 后端跑完二采却**不自述实况** ⇒ 明确失败 ✗✗（不许只写计划倍率就当成超清 ✓ —— "
          "那又是「回执自洽、与画面无关」✓）",
          no_details.ok is False and (no_details.error or {}).get("stage") == "refine"
          and "自述" in str((no_details.error or {}).get("message")),
          no_details.error)

    check("⑤¹⁴ 后端**没实现** ``refine_latents`` ⇒ 明确拒绝 ✗（**不静默按普通模式出片** ✗✗ —— "
          "那会让用户以为超清开着 ✓）",
          no_hook.ok is False and (no_hook.error or {}).get("stage") == "refine"
          and "refine_latents" in str((no_hook.error or {}).get("message")),
          no_hook.error)

    # ⭐⭐ **``totalMs`` 不是总耗时** ✗✗（2026-11-08 补 ✗）：它只是 ``stageMs`` 那几个键之和 ✓，
    #    不含装载（在 ``runtime._execute`` 里、``run_sync`` 之外 ✓）、阶段间隙与收尾清理 ✓✗；
    #    而 ``task.elapsedMs`` 是真墙钟 ✓ —— 两个数就躺在**同一份**回执里 ✓，谁也没说差在哪 ✓。
    #    现在 ``totalCovers`` 逐字给出实际入账的阶段 ✓（动态取键 ✗，不手抄清单 ✓），
    #    ``wallMs`` 给 ``run_sync`` 的真墙钟 ✓，``unaccountedMs`` 给差额 ✓。
    timing = pipe.run_sync(pipe.GenerationRequest(prompt="x", steps=2),
                           DryRunBackend()).to_dict()
    check("⑤³¹ ⭐⭐ ``totalMs`` 旁边必须写清**它覆盖哪几段** ✗✗"
          "（``totalCovers`` 逐字等于 ``stageMs`` 的键 ✓ 不是手抄清单 ✓），"
          "且 ``wallMs`` ≥ ``totalMs``、``unaccountedMs`` ＝ 两者之差 ✓ —— "
          "一个叫 total 的数**不含装载** ✗✗",
          timing.get("totalCovers") == list(timing.get("stageMs") or {})
          and "plan" in (timing.get("totalCovers") or [])      # 不许空表蒙过去 ✓
          and isinstance(timing.get("wallMs"), int)
          and timing["wallMs"] >= timing["totalMs"]
          and timing.get("unaccountedMs") == max(0, timing["wallMs"] - timing["totalMs"]),
          {key: timing.get(key)
           for key in ("totalMs", "totalCovers", "wallMs", "unaccountedMs")})

    # ⭐⭐ **计划说要二采、实况说没跑** ✗✗（2026-11-08 补 ✗）：张量层对「二采失败」的口径是
    #    **回退到上采样结果**（不炸 ✓），于是 ``applied=true`` 照样成立 ✓、``secondPass: false``
    #    只在 ``details`` 里 ✓✗ —— 而计划层那份 ``upscale.secondPass`` 写的是 true ✓ ⇒
    #    拿「applied + 计划」来读会以为二采跑了 ✓。现在顶层单列 ``secondPass`` 与跳过理由 ✓。
    class _SecondPassFails(DryRunBackend):
        """⚠️ 二采**失败** ⇒ 回退到上采样结果（不炸 ✓）—— 正是要在回执顶层说清的那种。"""

        def refine_latents(self, latents: Any, plan: Any, request: Any, *,
                           condition: Any = None) -> Any:
            super().refine_latents(latents, plan, request, condition=condition)
            self.refineDetails = {**self.refineDetails, "secondPass": False,
                                  "denoiseError": "RuntimeError: 假的二采故障 ✓",
                                  "notes": ["⚠️ 二采失败 ⇒ 回退到上采样结果 ✓"]}
            return latents

    failed_second = pipe.run_sync(
        pipe.GenerationRequest(prompt="x", steps=2,
                               upscale=upscale_mod.UpscalePlan(
                                   mode="ai-2x", scale=2.0, refine_steps=4, refine_denoise=0.4)),
        _SecondPassFails())
    failed_refine = failed_second.refine or {}
    check("⑤²⁴ ⭐⭐ 二采失败（回退）⇒ 回执**顶层** ``secondPass=False`` ✓ 且单列"
          "``secondPassSkipped``（带计划参数与原因 ✓）—— 不许让人拿「applied+计划」误以为跑过 ✗✗",
          failed_second.ok is True
          and failed_refine.get("secondPass") is False
          and "假的二采故障" in str((failed_refine.get("secondPassSkipped") or {}).get("why")),
          failed_refine.get("secondPassSkipped"))

    class _SecondPassOk(DryRunBackend):
        def refine_latents(self, latents: Any, plan: Any, request: Any, *,
                           condition: Any = None) -> Any:
            super().refine_latents(latents, plan, request, condition=condition)
            self.refineDetails = {**self.refineDetails, "secondPass": True,
                                  "denoiseSteps": 2, "tailSigmas": 3,
                                  "notes": ["二采：尾部 2 步 ✓"]}
            return latents

    ok_second = pipe.run_sync(
        pipe.GenerationRequest(prompt="x", steps=2,
                               upscale=upscale_mod.UpscalePlan(
                                   mode="ai-2x", scale=2.0, refine_steps=4, refine_denoise=0.4)),
        _SecondPassOk())
    ok_refine = ok_second.refine or {}
    check("⑤²⁵ 二采**真跑了** ⇒ ``secondPass=True`` ✓ 且**不**报跳过 ✓（判据不许见谁都喊 ✗）",
          ok_refine.get("secondPass") is True and "secondPassSkipped" not in ok_refine,
          ok_refine.get("secondPass"))

    unplanned = pipe.run_sync(
        pipe.GenerationRequest(prompt="x", steps=2,
                               upscale=upscale_mod.UpscalePlan(mode="ai-2x", scale=2.0)),
        _SecondPassOk())
    unplanned_refine = unplanned.refine or {}
    check("⑤²⁶ 计划**没要**二采、后端**真跑了** ⇒ 单独点名 ``secondPassUnplanned`` ✗"
          "（多花的耗时与改掉的细节都不在计划里 ✓）",
          unplanned_refine.get("secondPass") is True
          and "secondPassUnplanned" in unplanned_refine,
          unplanned_refine.get("secondPassUnplanned"))

    # 后端现状接口 ✓：分得清「缺依赖」与「实现待写」✓
    backends = client.get("/api/v1/engine/backends").json().get("data") or {}
    # ⚠️ 这里的断言**不随 torch 装没装而变** ✓（第一版写成"必须缺依赖"✗ ⇒ 装上就红 ✓）：
    #    稳定成立的是「真张量 ✓ / 画面仍不真 ✗ / 还不能出片 ✗」这三件事 ✓。
    check("⑮''' GET /engine/backends 如实报三件事：干跑可用 ✓、torch **真张量** ✓、"
          "但**真模型未接 ⇒ 还不能出片** ✗",
          backends.get("dryrun", {}).get("canGenerate") is False
          and backends.get("torch", {}).get("realTensors") is True
          and backends.get("torch", {}).get("synthetic") is True
          and backends.get("torch", {}).get("canGenerate") is False
          and bool(backends.get("torch", {}).get("pendingParts")),
          backends.get("torch", {}).get("pendingParts"))
    check("⑯''' 该接口还说明引导与首帧的支持面（前端据此决定显示什么 ✓）",
          backends.get("guidance", {}).get("supportsCfg") is True
          and backends.get("conditioning", {}).get("supportsFirstFrame") is True,
          sorted(backends))

    check("㊾ 给了首帧但**没给压缩比** ⇒ 失败归因到 condition 阶段，且文案要求补参数 ✓"
          "（不猜、不悄悄退回文生视频 ✗）",
          (missing_compression.get("error") or {}).get("stage") == "condition"
          and "temporalCompression" in str((missing_compression.get("error") or {}).get("message")),
          missing_compression.get("error"))


# ══════════════════════════════════════════════════════════════════════════
# ⑨ torch 后端：**依赖闸门**（本机没装 torch ⇒ 正好能验"报错可行动" ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_torch_backend() -> None:
    """torch 后端：**两种世界都成立** ✓ —— 没装依赖时验闸门 ✓，装了之后验**真张量** ✓。"""
    from app.services.engine import torch_backend as tb

    status = tb.dependency_status()
    check("①''' 依赖探测**不导入**这些包也能给结论（毫秒级、无副作用 ✓）",
          isinstance(status["items"], list) and len(status["items"]) >= 4
          and isinstance(status["ready"], bool), len(status["items"]))
    check("②''' 缺依赖时给出**可执行的安装命令**（不是只说「依赖缺失」✗）",
          status["ready"] or (bool(status["missing"])
                              and str(status["install"][0]).startswith("pip install ")),
          status["install"])
    # ⭐ 必需 / 可选**分开答** ✓（2026-09-20 ✓）：可选项缺了**不能**把后端判死 ✗ ——
    #    判据：本仓有等价的自研实现 ✓（如 `transformers` ↔ 自研 BPE `tokenizer_bpe.py` ✓）。
    optional = [item["package"] for item in status["items"] if item.get("optional")]
    check("②′''' ⭐ 依赖分**必需 / 可选**两栏 ✓：`transformers` 属可选 ✓ ⇒ 缺它只少一条通道 ✓"
          "（混在一起会让「可选没装」把后端判死 ✗✗）",
          "transformers" in optional
          and set(status["missing"]).isdisjoint(optional)      # 可选缺 ⇒ **不进**必需缺失 ✓
          and status["ready"] == (not status["missing"])       # `ready` 只看必需项 ✓
          and isinstance(status.get("optionalMissing"), list)
          and bool(status.get("installOptional")) == bool(status["optionalMissing"]),
          (optional, status.get("optionalMissing"), status["missing"]))
    transformers_entry = next(item for item in status["items"] if item["module"] == "transformers")
    check("②″''' 可选项的 `purpose` 要写清**自研替代**在哪 ✓（免得后人以为少了它就不能分词 ✗）",
          "自研 BPE" in transformers_entry["purpose"], transformers_entry["purpose"])

    # ── 闸门一：缺依赖（**模拟** ⇒ 与真机装没装无关 ✓ 永远可验 ✓）──────────
    original = tb.torch_available
    tb.torch_available = lambda: (False, "缺少依赖：torch（约 124 MB ✓） ⇒ 先跑 pip install torch ✓")
    try:
        gated = tb.TorchBackend()
        try:
            gated.init_latents(object(), object())
            deps_error: Exception | None = None
        except tb.TorchBackendUnavailable as err:
            deps_error = err
        gated_describe = gated.describe()
    finally:
        tb.torch_available = original  # type: ignore[assignment]
    check("③''' 缺依赖 ⇒ `reason='deps'` + 文案含 pip + 说明「现在还能干什么」（不是死胡同 ✗）",
          isinstance(deps_error, tb.TorchBackendUnavailable) and deps_error.reason == "deps"
          and "pip install" in str(deps_error) and "干跑" in str(deps_error), str(deps_error)[:70])
    check("④''' `describe()` **不做张量操作** ⇒ 缺依赖也能调 ✓（接口不会一起挂 ✗）",
          gated_describe["available"] is False and gated_describe["canGenerate"] is False
          and gated_describe["name"] == "torch", gated_describe["reason"])

    # ── 闸门二：依赖齐但**真模型未接**（真机装了 torch 也仍然成立 ✓）──────
    backend = tb.TorchBackend()
    described = backend.describe()
    check("⑤''' ⭐ 分清两件事：`realTensors=True`（张量真 ✓）但 `synthetic=True`（画面**不真** ✗）"
          "—— 不许因为「用上 torch 了」就把 synthetic 翻成 False ✗",
          described["realTensors"] is True and described["synthetic"] is True
          and described["canGenerate"] is False, described["pendingParts"])
    # ⚠️ **钉空域** ✗（2026-09-26 全量实测抓出来的 ✓）：本条要验的是「**权重没下载** ⇒ pending」✓，
    #    而 `_default_dit_path()` 走的是「解析只有一条」的口径 ✓（清单落点 → **动态探测根** ✓）
    #    ⇒ 在这台机器上它会**解析到盘上那份真文件** ✓✗（落在 ComfyUI 共享目录 ✓），于是
    #    `load_weights()` 不再停在「未下载」✗、而是往下走到**形态自检** ✓ 报「结构对不上」✓✗
    #    ⇒ 断言**随机器装了什么都变** ✗✗（正撞本仓「自检不得随机器翻脸」✗）。
    #    钉法：把落点钉到**空的临时目录** ✓ ⇒ 「缺权重 ⇒ pending」在任何机器上都成立 ✓。
    pinned = Path(tempfile.mkdtemp(prefix="voide-no-weights-")) / "diffusion_models" / "没有这个文件.safetensors"
    original_default_dit = backend._default_dit_path  # type: ignore[attr-defined]
    backend._default_dit_path = lambda: str(pinned)  # type: ignore[method-assign, assignment]
    try:
        backend.load_weights()
        weights_error: Exception | None = None
    except tb.TorchBackendUnavailable as err:
        weights_error = err
    finally:
        backend._default_dit_path = original_default_dit  # type: ignore[method-assign, assignment]
    if described["available"]:
        # ⚠️ 这条断言随实现推进**改过含义** ✓（原来断言"实现未写 ⇒ pending" ✗）：
        #    现在 `load_weights` **真的实现了** ✓ ⇒ `pending` 的含义变成「**权重没下载**」✓
        #    —— 而且报错要指向**加载计划**（还差多少 ✓），不是干巴巴一句"没有权重" ✗。
        check("⑥''' ⭐ 依赖齐、实现也在 ⇒ `load_weights` 因**权重未下载**报 `pending`，"
              "且**指向加载计划**（还差多少 ✓ 可行动 ✓）",
              isinstance(weights_error, tb.TorchBackendUnavailable)
              and weights_error.reason == "pending"
              and ("加载计划" in str(weights_error) or "权重未就绪" in str(weights_error)),
              str(weights_error)[:90])
    else:
        # ⚠️ 依赖缺失时，闸门**先**拦在 `deps` ✓ —— pending 只有在依赖齐了才谈得上 ✓
        #    （第一版这里不分世界 ⇒ 没装 torch 时断言 pending ✗ 必然红 ✓）
        check("⑥''' 依赖缺失时闸门先报 `deps` ✓（`pending` 只在依赖齐备后才可能 ✓）",
              isinstance(weights_error, tb.TorchBackendUnavailable)
              and weights_error.reason == "deps", str(weights_error)[:70])
        skip("torch 未安装 ⇒ 真张量部分跳过（`pip install torch --index-url "
             "https://download.pytorch.org/whl/cpu` 后重跑即可 ✓）")
        return

    # ── 装了 torch ⇒ 验**真张量**（本机 CPU 版也够 ✓：数学与真机一致 ✓）───
    plan = pipe.build_plan(req(steps=6))
    first = backend.init_latents(plan, req(seed=5))
    check("⑦''' 初始噪声是**真 torch 张量** ✓（有 dtype/shape/device ✓，不是 list ✓）",
          hasattr(first, "dtype") and hasattr(first, "norm")
          and str(first.dtype) == "torch.float32", (type(first), getattr(first, "dtype", None)))
    check("⑧''' 同种子**逐位可复现** ✓（真 `torch.Generator` + 真 `randn` ✓）",
          bool(backend.init_latents(plan, req(seed=5)).equal(first)), "")
    check("⑨''' 换种子结果不同 ✓（种子真的驱动了随机数 ✓）",
          not bool(backend.init_latents(plan, req(seed=6)).equal(first)), "")

    result = pipe.run_sync(req(steps=8), backend)
    check("⑩''' ⭐ 整条管线在**真张量**上跑通（同一份 sampler/guidance/conditioning ✓）",
          result.ok is True and result.sampleSteps == 8, result.error)

    from app.services.engine import sampler as real_sampler

    positive = backend.encode_text(req())["positive"]
    raw = backend.init_latents(plan, req(seed=7))
    before = float((raw - positive).norm())
    final, _steps = real_sampler.sample(
        lambda x, sigma: backend.denoise(x, float(sigma), positive, req()),
        raw, plan.sigmas, "euler")
    after = float((final - positive).norm())
    check("⑪''' 真张量上采样**真收敛**（末态到条件的距离 << 初始 ✓ —— 算法在真张量上成立 ✓）",
          after < before / 2, {"before": round(before, 4), "after": round(after, 4)})

    conditioned = backend.condition_first_frame(first, "首帧.png", [1.0, 0.0], plan, req())
    check("⑪'''b 首帧按掩码**真的混合**了（第 0 位取自图片 ✓）",
          float(conditioned[0]) != float(first[0]), (float(first[0]), float(conditioned[0])))

    guided = _CountingProxy(backend)
    pipe.run_sync(req(steps=6, guidance=__import__("app.services.engine.guidance",
                                                   fromlist=["g"]).GuidanceConfig(scale=3.0)), guided)
    check("⑫''' CFG 在真张量后端上同样**每步两次前向** ✓（calls == 2×步数 ✓）",
          guided.calls == 12, guided.calls)
    check("⑬''' 设备是**实测探测**出来的（本机无 NVIDIA ⇒ cpu ✓ 不写死 ✗）",
          described["device"] in ("cpu", "cuda") and described["cudaAvailable"] == (described["device"] == "cuda"),
          described["device"])


def case_img2img_mix() -> None:
    """图生图那**另一半** ✓：起点 latent 必须由参考图编码而来 ✓✗。

    为什么单独钉 ✗：``denoise<1`` 的错法**全程不报错** ✓ —— σ 梯子截短了 ✓、步数也对 ✓、
    图也出得来 ✓，只是那张图与参考图**毫无关系**（起点是纯噪声 ✓）✓✗。所以这里钉的是
    **公式** ``σ0·noise + (1−σ0)·ref``（参考实现 ``noise_scaling`` ✓）而不是"跑通了" ✓。
    """
    from app.services.engine import torch_backend as tb

    backend = tb.TorchBackend()
    if not backend.describe()["available"]:
        skip("torch 未安装 ⇒ 图生图混合（真张量）跳过 ✓")
        return
    import torch  # noqa: PLC0415  （已确认 ready ✓）

    plan = pipe.build_plan(req(stage="zimage", steps=8))
    # ⚠️ σ0 从 **``sigmas[0]``** 读 ✓（不是 ``start_sigma`` ✗）：起跑那一格的真源是梯子首值 ✓，
    #    两处都设成 0.55 才与「denoise=0.55 截尾后」的真实计划一致 ✓。
    plan.denoise, plan.start_sigma, plan.init_from = 0.55, 0.55, "image"
    plan.sigmas = [0.55, 0.2, 0.05, 0.0]

    class _MixStub(tb.TorchBackend):
        """只替换「参考图 ⇒ latent」那一步 ✓（公式与形状守卫走**真代码** ✓）。"""

        def __init__(self, ref: float, shape: tuple[int, ...]) -> None:
            self._device = "cpu"
            self._ref = float(ref)
            self._shape = shape

        def _torch(self) -> Any:
            return torch

        def latent_shape(self, _plan: Any) -> tuple[int, ...]:
            return self._shape

        def init_image_latents(self, _plan: Any, _request: Any) -> Any:
            return torch.full(self._shape, self._ref)

    stub = _MixStub(3.0, (1, 4, 8, 8))
    noise = torch.ones(1, 4, 8, 8)
    mixed = stub._mix_init_image(noise, plan, req())
    check("⑭''' 起点按 ``σ0·noise + (1−σ0)·ref`` 混 ✓（σ0=0.55、ref=3 ⇒ 恰好 1.9 ✓ "
          "少混一项 / 用 1−σ0 反了都到不了这个数 ✗✗）",
          abs(float(mixed[0, 0, 0, 0]) - 1.9) < 1e-6, float(mixed[0, 0, 0, 0]))

    plan.init_from = "noise"
    check("⑮''' ``init_from='noise'`` ⇒ **原样返回** ✓（纯文生图路径逐位不变 ✓ 不白跑一遍 VAE ✗）",
          bool(stub._mix_init_image(noise, plan, req()).equal(noise)), "")

    plan.init_from = "image"
    wrong = _MixStub(3.0, (1, 4, 9, 9))
    check("⑯''' 起点 latent 与噪声形状不一致 ⇒ **当场拒** ✗（广播成同一张 = 死图且不报错 ✗✗）",
          "形状不一致" in str(_err(lambda: wrong._mix_init_image(noise, plan, req()))),
          str(_err(lambda: wrong._mix_init_image(noise, plan, req())))[:70])

    real = tb.TorchBackend()
    check("⑰''' ``initImage`` 空 ⇒ 拒 ✗（起点不是可选美化项 ✓✗）",
          "空的" in str(_err(lambda: real.init_image_latents(plan, req(init_image=None)))),
          str(_err(lambda: real.init_image_latents(plan, req(init_image=None))))[:70])

    # ── ⭐ `init` 段：图生图的起点要 VAE ✓✗（2026-11-02 真跑逼出来的 ✓）─────────────
    #   `PHASE_PARTS` 原本只有 encode→text / sample→dit / decode→vae ✓ ⇒ 分时驻留的族
    #   （2512 / 2.1 / krea2 ✓）在 init 这一拍手上**只有 TE** ✓、VAE 要到 decode 才装 ✓
    #   ⇒ 图生图必报「没挂 VAE ✗ ⇒ 没法把图片编码成潜变量」✓（真跑实测 ✓）。
    #   ⚠️ 单测里 VAE 是**常驻替身** ✓ ⇒ 老判据永远试不出来 ✓✗，只能钉「表里有这一段」✓。
    import ast as _ast

    _parts = dict(getattr(tb.TorchBackend, "PHASE_PARTS", {}) or {})
    check("⑰'''' `PHASE_PARTS` 有 `init` 段且含 text（不重读 16 GiB 的 TE ✓）",
          tuple(_parts.get("init") or ()) == ("text", "vae"), _parts)
    _tree = _ast.parse((BACKEND_PY / "app/services/engine/pipeline.py").read_text(encoding="utf-8"))
    _guarded = ""
    for _node in _ast.walk(_tree):
        if not isinstance(_node, _ast.If) or "init_image" not in _ast.unparse(_node.test):
            continue
        for _sub in _ast.walk(_node):
            if (isinstance(_sub, _ast.Call) and getattr(_sub.func, "id", "") == "_backend_phase"
                    and len(_sub.args) >= 2 and _ast.unparse(_sub.args[1]) == "'init'"):
                _guarded = _ast.unparse(_node.test)
    check("⑰''''' 只在 `initImage` 非空时申请 `init` 段（纯文生图不多跑一趟换装 ✓）",
          _guarded == "request.init_image",
          _guarded or "pipeline 里找不到 `if request.init_image:` 包住 `_backend_phase(..., \"init\")`")

    # ── 4 维族（Z-Image / Qwen-Image 2.1 ✓ 没有时间维）必须**跟着族交维数** ✓✗ ─────
    original_load = tb.media_mod.load_image_tensor
    seen: dict[str, Any] = {}

    def _fake_load(path: str, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        seen.update(kwargs)
        return torch.zeros(1, 3, 1, 32, 32), {"width": kwargs.get("width"), "height": kwargs.get("height")}

    class _EncStub(tb.TorchBackend):
        """``latent_shape``（DiT 要的）与 ``_encode_pixels``（VAE 给的）**分开给** ✓✗ ——
        混成一个数就永远对得上 ✓（那样「通道/高/宽 对不上」这条守卫根本试不出来 ✓）。"""

        def __init__(self, shape: tuple[int, ...], encoded: tuple[int, int, int] | None = None) -> None:
            self._device = "cpu"
            self._shape = shape
            self._encoded = encoded or (shape[1], shape[-2], shape[-1])

        def _torch(self) -> Any:
            return torch

        def _gate(self) -> None:
            return None

        def latent_shape(self, _plan: Any) -> tuple[int, ...]:
            return self._shape

        def _encode_pixels(self, _frames: Any) -> Any:
            channels, height, width = self._encoded
            return torch.zeros(1, channels, 1, height, width)

    tb.media_mod.load_image_tensor = _fake_load  # type: ignore[assignment]
    try:
        four = _EncStub((1, 16, 32, 32)).init_image_latents(plan, req(init_image="x.png"))
        five = _EncStub((1, 16, 1, 32, 32)).init_image_latents(plan, req(init_image="x.png"))
        mismatch = _err(lambda: _EncStub((1, 16, 16, 16), encoded=(16, 32, 32))
                        .init_image_latents(plan, req(init_image="x.png")))
    finally:
        tb.media_mod.load_image_tensor = original_load  # type: ignore[assignment]
    check("⑱''' 4 维族 ⇒ 起点 latent 也**是 4 维** ✓、5 维族 ⇒ 5 维 ✓（一律造 5 维会让图族 "
          "patchify 报 expected 4D ✓ 指不到「起点给错」 ✗✗）",
          tuple(four.shape) == (1, 16, 32, 32) and tuple(five.shape) == (1, 16, 1, 32, 32),
          (tuple(four.shape), tuple(five.shape)))
    check("⑲''' 读参考图按**画布尺寸**取 ✓（``plan.width/height`` ✓ 不是原图尺寸 ✗ "
          "—— 尺寸不一致会在混合那步才炸 ✓ 且指向噪声 ✗）",
          (seen.get("width"), seen.get("height")) == (plan.width, plan.height), seen)
    check("⑳''' 编码出的「通道/高/宽」对不上 ⇒ **点名三个维度**报 ✗（不是一句「形状错」✗）",
          "通道/高/宽" in str(mismatch), str(mismatch)[:80])


def case_unused_references() -> None:
    """参考图**只有 H3 双流真吃** ✓ —— 其余档必须**说出来**（静默忽略＝用户以为在按图改 ✗✗）。

    为什么值得单列：画布上最自然的动作就是「连一张图 → 点生成」✓，而图片四档
    （zimage / krea2 / qwen_image / qwen_image21）与 wan22 都**一个字节都不读**
    ``reference_frames`` ✓✗ ⇒ 不说的话，用户拿到一张与参考图无关的新图，
    且**没有任何日志**能看出这一点 ✓✗（本仓最恨的一类）。
    """
    frames = ("参考一.png", "参考二.png")
    for stage in (pipe.ZIMAGE_STAGE, pipe.KREA2_STAGE, pipe.QWEN21_STAGE, pipe.QWEN_IMAGE_STAGE):
        plan = pipe.build_plan(pipe.GenerationRequest(
            prompt="p", stage=stage, reference_frames=frames), weights_bytes=None)
        joined = "\n".join(plan.warnings)
        check(f"⑳〰 {stage} 收到 2 路参考图 ⇒ **如实说被忽略**（并指路 Edit 工作流 / initImage）",
              "referenceFrames" in joined and "忽略" in joined and "initImage" in joined,
              joined[-260:])

    # ⭐ 反向：H3 双流**真吃**参考图 ⇒ 不许把它也报成"忽略" ✗（那会让人以为这档废了）
    h3_plan = pipe.build_plan(pipe.GenerationRequest(
        prompt="p", stage=pipe.VIDEO_STAGE, reference_frames=frames), weights_bytes=None)
    check("⑳〰 H3 双流**不报**「参考图被忽略」（它走 ref2va、真用得上 ✓）",
          "referenceFrames" not in "\n".join(h3_plan.warnings),
          "\n".join(h3_plan.warnings)[-260:])

    # ⭐ 没给参考图时不许无中生有（一条警告都不该多）
    bare = pipe.build_plan(pipe.GenerationRequest(prompt="p", stage=pipe.ZIMAGE_STAGE), weights_bytes=None)
    check("⑳〰 没给参考图 ⇒ 不出这条警告（不做无用的提示）",
          "referenceFrames" not in "\n".join(bare.warnings), bare.warnings)


def case_off_official_canvas() -> None:
    """**显式**给一个远离官方档位的像素预算 ⇒ 必须如实说（不拦 ✓ 但不许沉默 ✓✗）。

    为什么值得单列 ✗：各族的 `_fill_*_plan` 只在「像素预算**等于** `GenerationRequest` 的默认
    0.65」时才换成官方档 ✓ —— 用户明写 0.2 MP 就原样照跑 ✓。实测 zimage 0.2 MP @16:9
    落 **608×336**（官方 1024² 的 1/5 面积 ✓），而训练档位外的尺寸是拿**没训过**的区域换画质 ✓：
    症状是构图崩 / 多主体重复 / 细节糊，**一声不响** ✓✗。
    """
    def warnings_of(stage: str, mp: float) -> str:
        plan = pipe.build_plan(pipe.GenerationRequest(
            prompt="p", stage=stage, megapixels=mp), weights_bytes=None)
        return "\n".join(plan.warnings)

    low = warnings_of(pipe.ZIMAGE_STAGE, 0.2)
    check("⑳〸 zimage 0.2 MP ⇒ 报「低于训练档位 + 没训过的区域」（不报错但结果会崩）",
          "低于" in low and "没训过" in low, low[-300:])

    # ⭐ 反向（不瞎报）：1.0 MP 就是官方档 ⇒ 一条都不该出
    ok = warnings_of(pipe.ZIMAGE_STAGE, 1.0)
    check("⑳〸 zimage 1.0 MP = 官方档 ⇒ **不报**（判据不许见谁都喊）",
          "低于" not in ok and "高于" not in ok, ok[-300:])

    high = warnings_of(pipe.ZIMAGE_STAGE, 3.0)
    check("⑳〸 zimage 3 MP ⇒ 报「高于训练档位」（两个方向都要管，不只管小图）",
          "高于" in high, high[-300:])

    for stage, title in ((pipe.KREA2_STAGE, "Krea2"), (pipe.QWEN21_STAGE, "Qwen-Image 2.1")):
        joined = warnings_of(stage, 0.2)
        check(f"⑳〸 {title} 0.2 MP ⇒ 同样如实报（三族共用 `_warn_off_official_canvas`）",
              "低于" in joined and "没训过" in joined, joined[-300:])

    # ⭐ `target`（用途档位）是**有出处**的画布 ⇒ 不许再拿"偏离官方"去质疑它 ✗
    framed = pipe.build_plan(pipe.GenerationRequest(
        prompt="p", stage=pipe.KREA2_STAGE, target="character-sheet"), weights_bytes=None)
    check("⑳〸 走 `target` 档位时**不报**偏离官方（那是生产线实测画布，不是用户乱填）",
          "没训过" not in "\n".join(framed.warnings),
          "\n".join(framed.warnings)[-300:])


def _err(fn: Any) -> Exception | None:
    try:
        fn()
    except Exception as err:  # noqa: BLE001
        return err
    return None


def case_alerts_channel() -> None:
    """``alerts`` 必须**抬到回执顶层** ✓✗（2026-11-10 补 ✗）。

    ⚠️ 这不是"再加一个结论" ✗ —— 这是**通路** ✓：各后端把跑完才知道的结论写进
    ``backend.write()`` 的返回 ✓，被 :attr:`PipelineResult.outputs` 原样收下 ✓
    ⇒ 落在 ``result.outputs.alerts`` ✓✗；而前端读的是 ``task.result.alerts``（**顶层** ✓）
    ⇒ 2026-11-09 起那一整批「音轨几乎是静音 ✗ / 音轨被截 ✗ / 落盘帧数没核对过 ✗」
    **一个都没到用户眼前** ✓（后端说得再清楚，通道断了就是白说 ✓）。
    """
    loud = pipe.PipelineResult(ok=True, outputs={"videoPath": "a.mp4",
                                                 "alerts": ["⚠️ 音轨几乎是静音 ✗",
                                                            "⚠️ 落盘帧数没核对过 ✗"]})
    quiet = pipe.PipelineResult(ok=True, outputs={"videoPath": "b.mp4"})
    topless = pipe.PipelineResult(ok=True, outputs=None)
    loud_d, quiet_d, topless_d = loud.to_dict(), quiet.to_dict(), topless.to_dict()
    check("㊱¹² ⭐⭐ ``alerts`` 从 ``outputs`` **抬到回执顶层** ✓✗（前端只读 ``result.alerts`` ✓）："
          "有 alerts ⇒ 顶层**逐字**等于 ``outputs.alerts`` ✓（抬层不篡改 ✓）、"
          "``outputs`` 里**仍保留** ✓（唯一真源还在后端 ✓）；"
          "没有 alerts ⇒ 顶层**根本没有这个键** ✓（不是空列表 ✗ —— 「一条都没有」与「没说」"
          "要分得开 ✓）；``outputs=None`` 也不许崩 ✓",
          loud_d.get("alerts") == ["⚠️ 音轨几乎是静音 ✗", "⚠️ 落盘帧数没核对过 ✗"]
          and loud_d["outputs"].get("alerts") == loud_d["alerts"]
          and "alerts" not in quiet_d and "alerts" not in topless_d,
          {"loud": loud_d.get("alerts"), "quietHas": "alerts" in quiet_d,
           "noneHas": "alerts" in topless_d})


def main() -> int:
    case_plan()
    case_schedule_names()
    case_end_to_end()
    case_algorithm_really_runs()
    case_stochastic_samplers()
    case_cancel()
    case_error_attribution()
    case_serialization()
    case_guidance()
    case_conditioning()
    case_torch_backend()
    case_img2img_mix()
    case_unused_references()
    case_off_official_canvas()
    case_api()
    case_alerts_channel()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
