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
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import pipeline as pipe  # noqa: E402
from app.services.engine import upscale as upscale_mod  # noqa: E402
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
    check("⑩ 给了 temporalCompression ⇒ 算出潜空间帧数 ✓",
          with_compression.latent_frames == 31, with_compression.latent_frames)
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

    bad_plan = pipe.run_sync(req(prompt=""), DryRunBackend())
    check("㉞ plan 阶段就失败 ⇒ **没有**后续阶段耗时（根本没开始跑 ✓）",
          (bad_plan.error or {}).get("stage") == "plan"
          and set(bad_plan.stageMs) == {"plan"}, (bad_plan.error, bad_plan.stageMs))


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
    check("⑤¹⁴ 后端**没实现** ``refine_latents`` ⇒ 明确拒绝 ✗（**不静默按普通模式出片** ✗✗ —— "
          "那会让用户以为超清开着 ✓）",
          no_hook.ok is False and (no_hook.error or {}).get("stage") == "refine"
          and "refine_latents" in str((no_hook.error or {}).get("message")),
          no_hook.error)

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
    try:
        backend.load_weights()
        weights_error: Exception | None = None
    except tb.TorchBackendUnavailable as err:
        weights_error = err
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


def main() -> int:
    case_plan()
    case_end_to_end()
    case_algorithm_really_runs()
    case_cancel()
    case_error_attribution()
    case_serialization()
    case_guidance()
    case_conditioning()
    case_torch_backend()
    case_api()

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
