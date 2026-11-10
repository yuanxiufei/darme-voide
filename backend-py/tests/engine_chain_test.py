"""S? 自检：**多段成片**（分段 → 拼接 → 链式规范化 ✓ 2026-09-26）。

判据只写**不变量**与**可复核的事实** ✓（换网格 / 换上限 / 换重叠都还站得住 ✓），不记死数字 ✗：

* 单段请求**原路** ✓（``plan_chain`` 回 ``None`` ⇒ 行为一字不改 ✓）；
* 拼完帧数 == 各段 keep 之和 == 总帧数 ✓（守恒是硬契约 ✓）；
* 规范化要**真的把跨段漂移拉小** ✓（判「拉平了没有」✓ 不判「等于某个数」✗）；
* 逐段产物**各在自己的子目录** ✓（同目录互相覆盖 ⇒ 只剩最后一段能核对 ✗）；
* 接线**是真的** ✓：静态扫源码确认 ``runtime`` 走链子 ✓、``pipeline`` 真的**调**那个回调 ✓。

⚠️ 全程**不打真权重** ✓：段执行器是注入的**假后端** ✓（真编排 ✓ 真张量 ✓ 真 mp4 ✓ 假画面 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_chain_test.py
"""
from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import chain  # noqa: E402
from app.services.engine import geometry  # noqa: E402
from app.services.engine import pipeline as pipe  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _error(fn) -> Exception:  # noqa: ANN001
    try:
        fn()
    except Exception as err:  # noqa: BLE001
        return err
    return Exception("（没有抛错 ✗）")


def case_plan() -> None:
    """分段判据：什么时候**该**分 ✓、什么时候**一定不许**分 ✓。"""
    short = pipe.GenerationRequest(prompt="短", seed=1, seconds=5.0)
    check("① 短请求（5s 量级）**不分段** ✓（回 None ⇒ 原路一字不改 ✓）",
          chain.plan_chain(short) is None and chain.needs_chain(short) is False,
          chain.plan_chain(short))

    still = pipe.GenerationRequest(prompt="图", seed=1, seconds=600.0, stage="sdxl")
    check("② 图片阶段（`stage=sdxl`）**永不分段** ✓（它只有一帧 ✓ 硬分就是改语义 ✗）",
          chain.plan_chain(still) is None, chain.plan_chain(still))

    plan = chain.plan_chain(pipe.GenerationRequest(prompt="长", seed=7, seconds=60.0))
    check("③ 长请求（60s）**要分段** ✓ 且段数 > 1 ✓",
          plan is not None and plan.segmentCount > 1, None if plan is None else plan.to_dict())
    if plan is None:
        return
    check("④ 总帧数与 `geometry.snap_frames` **同源** ✓（两处口径打架 ⇒ 计划自相矛盾 ✗）",
          plan.totalFrames == geometry.snap_frames(60.0, fps=plan.fps), plan.totalFrames)
    check("⑤ 每段都 ≤ 单段原生上限 ✓（362 帧 ✓ 见 chain 模块头的事实来源 ✓）",
          all(item.frames <= chain.H3_NATIVE_MAX_FRAMES for item in plan.segments),
          [item.frames for item in plan.segments])
    check("⑥ ⭐ 各段 keep 之和 == 总帧数 ✓（拼接器的硬契约 ✓）",
          sum(item.keptFrames for item in plan.segments) == plan.totalFrames,
          [(item.keptFrames, item.frames) for item in plan.segments])
    check("⑦ 第 0 段无首帧 ✓、其后每段首帧 == **上一段保留区间最后一帧** ✓（接缝才真连续 ✓）",
          plan.segments[0].carryFrame is None
          and all(plan.segments[i].carryFrame == plan.segments[i - 1].startFrame
                  + plan.segments[i - 1].keepTo - 1 for i in range(1, plan.segmentCount)),
          [item.carryFrame for item in plan.segments])
    check("⑧ `to_dict` 带完整段表 ✓（可审计 ✓）",
          len(plan.to_dict().get("segments") or []) == plan.segmentCount, plan.to_dict().keys())

    edge = pipe.GenerationRequest(prompt="边界", seed=1,
                                  seconds=chain.H3_NATIVE_MAX_FRAMES / 24.0)
    check("⑨ 正好等于单段上限（362 帧 ≈ 15.08s）⇒ **仍不分段** ✓（判据不许差一格 ✗）",
          chain.plan_chain(edge) is None, chain.plan_chain(edge))
    over = pipe.GenerationRequest(prompt="边界+", seed=1,
                                  seconds=(chain.H3_NATIVE_MAX_FRAMES + 1) / 24.0)
    got = chain.plan_chain(over)
    check("⑩ 再长一点点 ⇒ **立刻**分段 ✓（不是「忍到翻倍才分」✗）",
          got is not None and got.segmentCount == 2, None if got is None else got.to_dict())


def case_stitch() -> None:
    """拼接：守恒 ✓、越界报错 ✓、画幅不一致报错 ✓。"""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        skip("torch 未安装 ⇒ 拼接 / 规范化 / 造段那几节全部跳过 ✓（未跑 ≠ 绿 ✗）")
        return

    plan = chain.plan_chain(pipe.GenerationRequest(prompt="x", seed=1, seconds=40.0))
    if plan is None:
        check("⑪ 40s 竟然没分段 ⇒ 前置条件不成立 ✗", False, "plan_chain 回 None")
        return
    parts = [torch.rand(1, 3, item.frames, 8, 8) for item in plan.segments]
    out, report = chain.stitch_frames(parts, plan.segments, expected_frames=plan.totalFrames)
    check("⑪ 拼出的帧数 == 总帧数 ✓ 且报告说守恒 ✓",
          int(out.shape[2]) == plan.totalFrames and report["conserved"] is True, report["frames"])
    manual = torch.cat([part[:, :, item.keepFrom:item.keepTo]
                        for part, item in zip(parts, plan.segments)], dim=2)
    check("⑫ 每段只留 `[keepFrom, keepTo)` ✓ ⇒ 重叠区**只出现一次** ✓（多了会顿 ✓ 少了会缺 ✓）",
          bool(torch.equal(out, manual)), (tuple(out.shape), tuple(manual.shape)))
    short = [torch.rand(1, 3, max(1, item.keepTo - 1), 8, 8) for item in plan.segments]
    check("⑬ 某段解出的帧数不够 ⇒ **报错** ✓（不静默补帧 ✗）",
          "只解出" in str(_error(lambda: chain.stitch_frames(short, plan.segments))))
    check("⑭ 段数对不上 ⇒ **报错** ✓（少一段 = 产物短一截而用户看不出来 ✗）",
          "段数对不上" in str(_error(lambda: chain.stitch_frames(parts[:1], plan.segments))))
    mixed = [parts[0], *[torch.rand(1, 3, item.frames, 16, 16) for item in plan.segments[1:]]]
    check("⑮ 各段画幅不一致 ⇒ **报错** ✓（拼不出一条片子 ✓）",
          "画幅不一致" in str(_error(lambda: chain.stitch_frames(mixed, plan.segments))))
    check("⑯ 与 `expected_frames` 不符 ⇒ **报错** ✓（守恒是硬契约 ✓）",
          "总帧数" in str(_error(
              lambda: chain.stitch_frames(parts, plan.segments, expected_frames=1))))
    check("⑰ 形状不对（4 维）⇒ **报错** ✓",
          "形状应为" in str(_error(lambda: chain.stitch_frames(
              [torch.rand(3, 5, 8, 8)] * len(plan.segments), plan.segments))))


def case_normalize() -> None:
    """链式规范化：**真的拉平了** ✓、报告不静默 ✓、参数不猜 ✓。"""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    import inspect  # noqa: PLC0415

    frames = torch.rand(1, 3, 240, 64, 64)
    frames[:, :, 120:] = (frames[:, :, 120:] * 0.6 + 0.8).clamp(-1, 1)  # 后半过曝 ⇒ 漂移最小复现 ✓
    before = abs(float(frames[0, :, :120].mean()) - float(frames[0, :, 120:].mean()))
    out, report = chain.normalize_chain(frames, fps=24, segment_starts=[0, 120])
    after = abs(float(out[0, :, :120].mean()) - float(out[0, :, 120:].mean()))
    check("㊱ 跨段**亮度漂移被真的拉小** ✓（判「拉平了没有」✓ 不判死数字 ✗）",
          after < before * 0.5, {"before": round(before, 4), "after": round(after, 4)})
    check("㊲ 报告不静默 ✓：基线 ✓ 参考帧 ✓ 修正峰值 ✓ 分段均值 ✓ 出处 ✓ 全在 ✓",
          report["applied"] is True and report["baseline"] > 0 and report["referenceFrame"] >= 0
          and report["correctionMax"] >= 0 and isinstance(report["correctionsBySegment"], list)
          and "ComfyUI-H3-Multishot" in report["source"], sorted(report))
    check("㊳ 首段受 **deadband** 保护 ✓（不拿自己的均值削自己 ✓）",
          report["correctionsBySegment"][0]["meanCorrection"] <= 0.05,
          report["correctionsBySegment"][0])

    few = torch.rand(1, 3, 4, 64, 64)
    same, tiny = chain.normalize_chain(few, fps=24)
    check("㊴ 帧数 < 8 ⇒ **原样返回** ✓ 且写清为什么 ✓（不静默改画面 ✓）",
          bool(torch.equal(same, few)) and tiny["applied"] is False
          and "没什么可拉平" in tiny["reason"], tiny.get("reason"))
    signature = inspect.signature(chain.normalize_chain).parameters
    mismatch = {key: (signature[key].default, value)
                for key, value in chain.NORM_DEFAULTS.items() if signature[key].default != value}
    check("㊵ 函数默认值与 `NORM_DEFAULTS` **逐项一致** ✓（两条独立来源互证 ✓ 谁改漏谁红 ✓）",
          not mismatch, mismatch)
    for label, kwargs in {"fps=0": {"fps": 0}, "deadband<1": {"fps": 24, "deadband": 0.9},
                          "ema=0": {"fps": 24, "ema": 0.0},
                          "值域认不出": {"fps": 24, "value_range": "0-255"}}.items():
        err = _error(lambda kw=kwargs: chain.normalize_chain(frames, **kw))
        check(f"㊶ 参数 {label} ⇒ **报错** ✓（不猜不静默 ✓）",
              isinstance(err, chain.ChainError), str(err)[:70])
    check("㊷ 形状不对（3 维）⇒ **报错** ✓",
          isinstance(_error(lambda: chain.normalize_chain(
              torch.rand(64, 64, 3), fps=24)), chain.ChainError))


class _FakeBackend:
    """假后端 ✓（`run_chain` 只用它的名字与 `synthetic` ✓）。"""

    name = "fake"
    synthetic = False


def _fake_runner(size: int = 48, decode_ms: int = 7) -> dict:
    """假**段执行器** ✓：真编排 ✓ 真张量 ✓ 真 mp4 ✓ 假画面 ✓（不碰真权重 ✓）。"""
    import torch  # noqa: PLC0415

    from app.services.engine import media as media_mod

    state: dict = {"calls": [], "requests": [], "fail_at": None, "cancel_at": None,
                   "no_frames": False, "steps": 12}

    def runner(request, backend, *, on_event=None, cancel=None, on_decoded=None):  # noqa: ANN001
        index = len(state["calls"])
        state["calls"].append(index)
        state["requests"].append(request)
        if state["cancel_at"] is not None and index >= state["cancel_at"]:
            return pipe.PipelineResult(ok=False, cancelled=True, backend="fake")
        count = geometry.snap_frames(request.seconds, fps=int(request.fps or 24))
        frames = None if state["no_frames"] else torch.rand(1, 3, count, size, size) * 2 - 1
        if on_decoded is not None:
            on_decoded({} if frames is None else {"frames": frames, "frameCount": count})
        outputs: dict = {}
        if frames is not None:
            target = Path(request.outputs_dir or tempfile.mkdtemp()) / f"video_seed{request.seed}.mp4"
            outputs = {"videoPath": str(target),
                       "video": media_mod.write_video(frames, target, fps=int(request.fps or 24))}
        #: 每段**真跑过的步数** ✓（假后端也给一个非零值 ✓ —— 否则「累加漏传」这条判据永远看不见 ✓✗）
        result = pipe.PipelineResult(ok=True, backend="fake", outputs=outputs,
                                     sampleSteps=int(state["steps"]))
        result.stageMs["decode"] = decode_ms
        if on_event:
            on_event({"kind": "stage", "stage": "decode"})
        if state["fail_at"] is not None and index == state["fail_at"]:
            result.ok = False
            result.error = {"stage": "sample", "message": "假后端故意失败 ✓"}
        return result

    state["runner"] = runner
    return state


def case_run_chain(root: Path) -> None:
    """整链：逐段跑 ✓ 接缝接力 ✓ 拼成真 mp4 ✓ 失败/取消/缺帧都不静默 ✓。"""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    from app.services.engine import media as media_mod

    request = pipe.GenerationRequest(prompt="链", seed=5, seconds=30.0, outputs_dir=str(root))
    plan = chain.plan_chain(request)
    check("㊸ 30s 要分段 ✓（本组用例的前提 ✓）",
          plan is not None and plan.segmentCount >= 2, None if plan is None else plan.to_dict())
    if plan is None:
        return
    state = _fake_runner()
    events: list[dict] = []
    result = chain.run_chain(request, _FakeBackend(), run_segment=state["runner"],
                             on_event=events.append, normalize=False)
    check("㊹ 整链跑通 ✓ 段数一致 ✓",
          len(state["calls"]) == plan.segmentCount and bool(result.ok), len(state["calls"]))
    dirs = [str(item.outputs_dir) for item in state["requests"]]
    check("㊺ ⭐ 逐段产物**各在自己的子目录** ✓（同目录互相覆盖 ⇒ 只剩最后一段能核对 ✗）",
          len(set(dirs)) == plan.segmentCount and all("segments" in item for item in dirs), dirs)
    check("㊻ 第 0 段无首帧 ✓、其后每段都带**真 PNG** 当首帧 ✓（接缝靠它 ✓）",
          state["requests"][0].first_frame is None
          and all(Path(item.first_frame).is_file() for item in state["requests"][1:]),
          [item.first_frame for item in state["requests"]])
    check("㊼ 段号进**结构化字段** ✓（前端要画「第 2/4 段」✓ 只写文案不行 ✗）",
          any(ev.get("segment") == plan.segmentCount - 1
              and ev.get("segmentTotal") == plan.segmentCount for ev in events),
          [ev.get("stage") for ev in events][:8])
    check("㊽ 各段耗时**逐项累加** ✓（取最后一段 = 少报 ✗）",
          int(result.stageMs.get("decode", 0)) == 7 * plan.segmentCount,
          result.stageMs.get("decode"))
    check("㊾ 报告说清「拼接守恒」✓ 且「规范化被**明确**关掉」✓（静默跳过 = 用户以为治过了 ✗）",
          result.outputs["chain"]["report"]["stitch"]["conserved"] is True
          and result.outputs["chain"]["report"]["normalize"]["applied"] is False,
          result.outputs["chain"]["report"]["normalize"])
    back, _info = media_mod.load_video_tensor(Path(result.outputs["videoPath"]))
    check("㊿ ⭐ 成片是**真 mp4** ✓ 且读回帧数 == 总帧数 ✓（硬事实 ✓）",
          int(back.shape[2]) == plan.totalFrames,
          (int(back.shape[2]), plan.totalFrames))

    bad = _fake_runner()
    bad["fail_at"] = 1
    failed = chain.run_chain(request, _FakeBackend(), run_segment=bad["runner"], normalize=False)
    check("① 一段失败 ⇒ **整条失败** ✓ 且阶段名带段号 ✓（否则只看到「sample 失败」✗）",
          failed.ok is False and str(failed.error.get("stage", "")).startswith("segment[1]/")
          and failed.outputs["chain"]["failedSegment"] == 1, failed.error)
    gone = _fake_runner()
    gone["cancel_at"] = 1
    stopped = chain.run_chain(request, _FakeBackend(), run_segment=gone["runner"],
                              normalize=False)
    check("② 段间被叫停 ⇒ **取消**（不是失败 ✗ —— 前端要分开显示 ✓）且已完成段留住 ✓",
           stopped.cancelled is True and stopped.outputs["chain"]["stoppedBefore"] == 1,
           stopped.outputs["chain"])
    #: ⭐⭐ **``stageMs`` 累加了、步数却没带上** ✗✗（2026-11-08 补 ✗）：``_stopped`` / ``_segment_failed``
    #:    都是**新建**一个 ``PipelineResult`` ✓（早退 ✓），``stageMs`` 有人逐项累加着交过去 ✓，
    #:    而 ``sampleSteps`` 没人传 ⇒ 用**默认 0** ✓✗ —— 于是「跑完 3 段（每段 12 步、真花了钱 ✓）
    #:    后在第 4 段被取消」在回执里写成 ``sampleSteps: 0`` ✓，与**同一份**回执的 ``stageMs``
    #:    互相矛盾 ✓；而成功路径上**同名**字段是真步数之和 ✓（同名字段两种含义 ✓）。
    check("②¹ ⭐⭐ 段间取消 ⇒ 已完成那些段的步数**必须跟着出来** ✓ 不是 0 ✗"
          "（1 段 × 12 步 ⇒ ``sampleSteps`` 12 ✓）—— 「跑完 1 段后取消」与「一步没跑」不该同形 ✓",
          int(stopped.sampleSteps) == 12, int(stopped.sampleSteps))
    #: ⚠️ 期望值是 **24** 不是 12（第一版写成 12 被这条判据抓出来 ✓）：失败那一段**自己**也跑了
    #:    12 步才失败 ✓（假后端在置 ``ok=False`` **之前**就报了 ``sampleSteps`` ✓，与真实采样
    #:    「跑到第 5 步炸、前 5 步是真花掉」同理 ✓）⇒ 累计 24 步 ✓。这个数恰好说明「失败段
    #:    的成本也要算进去」✓ —— 只报成功段的 12 步会把失败段的算力悄悄抹掉 ✓。
    check("①¹ ⭐⭐ 某段失败 ⇒ 前面跑完那些段的步数**同样要交出来** ✓ 不是 0 ✗"
          "（2 段 × 12 步 ＝ 24 ✓：**失败那段也跑了 12 步** ✓ —— 已花掉的算力记 0 会让人以为"
          "「没跑起来、所以不贵」✗）",
          int(failed.sampleSteps) == 24, int(failed.sampleSteps))
    loop_top = _fake_runner()
    loop_stopped = chain.run_chain(
        request, _FakeBackend(), run_segment=loop_top["runner"], normalize=False,
        cancel=lambda: len(loop_top["calls"]) >= 1)
    check("②² ⭐ 取消发生在**循环顶上**（上一段刚跑完、下一段还没开 ✓）⇒ 步数照样带出来 ✓"
          "（``stoppedBefore == 1`` ✓、``sampleSteps`` 12 ✓ —— 这一支和 ② 是**两个**早退口 ✓）",
          loop_stopped.cancelled is True
          and loop_stopped.outputs["chain"]["stoppedBefore"] == 1
          and int(loop_stopped.sampleSteps) == 12,
          {"stoppedBefore": loop_stopped.outputs["chain"].get("stoppedBefore"),
           "sampleSteps": int(loop_stopped.sampleSteps)})
    mute = _fake_runner()
    mute["no_frames"] = True
    err = _error(lambda: chain.run_chain(request, _FakeBackend(), run_segment=mute["runner"],
                                         normalize=False))
    check("③ 某段没交回帧张量 ⇒ **报错**并列出该段 decode 的键 ✓（不静默丢一段 ✗）",
          isinstance(err, chain.ChainError) and "没交回可用的帧张量" in str(err), str(err)[:110])
    err2 = _error(lambda: chain.run_chain(
        pipe.GenerationRequest(prompt="短", seed=1, seconds=4.0), _FakeBackend(),
        run_segment=state["runner"]))
    check("④ 单段请求误调 `run_chain` ⇒ **报错** ✓（调用方判错要立刻知道 ✓）",
          isinstance(err2, chain.ChainError) and "不该走 run_chain" in str(err2), str(err2)[:90])

    small = pipe.GenerationRequest(prompt="小链", seed=6, seconds=16.0,
                                   outputs_dir=str(root / "small"))
    state2 = _fake_runner()
    res2 = chain.run_chain(small, _FakeBackend(), run_segment=state2["runner"])
    norms = res2.outputs["chain"]["report"]["normalize"]
    check("⑤ 带**链式规范化**的整链也跑通 ✓ 且报告有基线/参考帧/分段修正 ✓",
          bool(res2.ok) and norms["applied"] is True and norms["baseline"] > 0
          and len(norms["correctionsBySegment"]) == len(state2["calls"]), norms)


def case_chain_frames(root: Path) -> None:
    """⭐⭐ **成片也要核帧数** ✗✗（2026-11-08 补 ✗）。

    :func:`chain.stitch_frames` 核的是**张量层**守恒 ✓（拼出来的帧数 == 各段 keep 之和 ==
    总帧数 ✓），但它**管不到编码器** ✓✗ —— libx264 会不会丢/补帧 ✓、封装有没有写 index ✓、
    重封装中间态读不读得到 ``nb_frames`` ✓。而 ``torch_backend.write`` 那条单段路**早就在核** ✓
    ⇒ 同一件事实两条路一半核一半不核 ✓。这里把 ``probe`` 包一层做**读数缺失**的模拟
    （真 ffprobe 仍在底下跑 ✓，只把某一格涂掉 ✓），与 ``engine_artifact_test`` 同一手法 ✓。
    """
    try:
        import torch  # noqa: PLC0415,F401
    except ImportError:
        return
    from app.services.engine import media as media_mod

    real_probe = media_mod.probe

    def probe_patch(drop: tuple[str, ...] = (), **overrides: object):
        def patched(path):
            facts = dict(real_probe(path))
            for key in drop:
                facts[key] = None
            facts.update(overrides)
            return facts

        return patched

    def run(tag: str, request: Any, drop: tuple[str, ...] = (), **overrides: object):
        out = root / f"chain_frames_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        #: ⚠️ ``GenerationRequest`` 是 **frozen** 的 ✓ ⇒ 落点只能另造一份 ✓（直接赋值会抛 ✓）。
        request = dataclasses.replace(request, outputs_dir=str(out))
        state = _fake_runner()
        media_mod.probe = probe_patch(drop, **overrides)
        try:
            return chain.run_chain(request, _FakeBackend(), run_segment=state["runner"],
                                   normalize=False), out, state
        finally:
            media_mod.probe = real_probe

    ok_request = pipe.GenerationRequest(prompt="链", seed=7, seconds=16.0)
    ok_plan = chain.plan_chain(ok_request)
    ok_card, _ok_dir, _ok_state = run("ok", ok_request)
    check("⑤¹ ⭐ 成片回执**带着**帧数核对结论 ✓（此前这三格一个都没有 ✗ ⇒ 只能从「真 mp4 ✓」"
          "推断「大概也核过了」✗）：``framesVerified == 'nb_frames'`` ✓ "
          "``encodedFrames == plan.totalFrames`` ✓ ``framesMatch is True`` ✓",
          ok_card.ok is True and ok_card.outputs.get("framesVerified") == "nb_frames"
          and int(ok_card.outputs.get("encodedFrames") or -1) == int(ok_plan.totalFrames)
          and ok_card.outputs.get("framesMatch") is True,
          {"framesVerified": ok_card.outputs.get("framesVerified"),
           "encodedFrames": ok_card.outputs.get("encodedFrames"),
           "totalFrames": None if ok_plan is None else int(ok_plan.totalFrames)})

    guessed_request = pipe.GenerationRequest(prompt="链", seed=8, seconds=16.0)
    guessed_plan = chain.plan_chain(guessed_request)
    guessed_card, _g_dir, _g_state = run("guessed", guessed_request, ("nbFrames",))
    check("⑤² ⭐ ``nbFrames`` 读不到但读得到时长/帧率 ⇒ 按**时长×帧率估** ✓ 并**标明是估的** ✓"
          "（``framesVerified == 'duration×fps'`` ✓、``framesMatch is True`` ✓、note 里有「估」✓）"
          "—— 与单段路（``torch_backend.write``）逐字同口径 ✓",
          guessed_card.outputs.get("framesVerified") == "duration×fps"
          and guessed_card.outputs.get("framesMatch") is True
          and int(guessed_card.outputs.get("encodedFrames") or -1) == int(guessed_plan.totalFrames)
          and "估" in str(guessed_card.outputs.get("note") or ""),
          {"framesVerified": guessed_card.outputs.get("framesVerified"),
           "note": guessed_card.outputs.get("note")})

    blind_request = pipe.GenerationRequest(prompt="链", seed=9, seconds=16.0)
    blind_card, blind_dir, _ = run("blind", blind_request, ("nbFrames", "durationSeconds"))
    video_left = list(blind_dir.glob("video_chain_*.mp4"))
    check("⑤³ ⭐⭐ 两格都读不到 ⇒ 如实写 ``framesVerified: None`` ＋ ``framesMatch: None`` ＋ note "
          "**点名没核过** ✗（**不许**把 ``None`` 读成「通过」✗）；成片仍落盘 ✓"
          "（没读数既不该拦 ✗，也不许装作核过 ✓✗）",
          blind_card.ok is True and blind_card.outputs.get("framesVerified") is None
          and blind_card.outputs.get("framesMatch") is None
          and "没核对过" in str(blind_card.outputs.get("note") or "")
          and len(video_left) == 1,
          {"framesVerified": blind_card.outputs.get("framesVerified"),
           "framesMatch": blind_card.outputs.get("framesMatch"),
           "note": blind_card.outputs.get("note")})

    short_request = pipe.GenerationRequest(prompt="链", seed=10, seconds=16.0)
    short_plan = chain.plan_chain(short_request)
    short_seconds = (int(short_plan.totalFrames) - 10) / float(short_plan.fps or 24)
    short_err = None
    short_dir = root / "chain_frames_short"
    short_dir.mkdir(parents=True, exist_ok=True)
    short_request = dataclasses.replace(short_request, outputs_dir=str(short_dir))
    short_state = _fake_runner()
    media_mod.probe = probe_patch(("nbFrames",), durationSeconds=short_seconds)
    try:
        chain.run_chain(short_request, _FakeBackend(), run_segment=short_state["runner"],
                        normalize=False)
    except Exception as err:  # noqa: BLE001
        short_err = err
    finally:
        media_mod.probe = real_probe
    leftovers = sorted(item.name for item in short_dir.glob("video_chain_*.mp4"))
    check("⑤⁴ ⭐⭐ 估出来的帧数**差得多**（不是舍入）⇒ 照样当场失败 ✓ 且那条**废片不留在盘上** ✓"
          "（㉕ 同一口径：失败路径一个文件都没写 ✓）"
          f"（时长写着 {round(short_seconds, 3)} 秒 ⇒ ≈ "
          f"{int(round(short_seconds * float(short_plan.fps or 24)))} 帧 vs "
          f"plan {int(short_plan.totalFrames)} ✓）",
          isinstance(short_err, chain.ChainError) and not leftovers,
          {"error": str(short_err)[:110], "leftovers": leftovers})

    #: ⭐⭐ 成片路的**帧率三档** ✓✗（2026-11-10 补 ✗）：单段路（``torch_backend.write``）早就有
    #:   ``fpsVerified`` ✓，成片这条**另一条**路此前只有 ``framesVerified`` ✓✗ —— 「帧率被改了
    #:   （音画不同步 ✓）」与「帧率没读到（不知道 ✓）」在这条路上**一声不响** ✓，而两格读数
    #:   （``durationSeconds`` ✓ / ``videoSeconds`` ✓）都"有值、ffprobe 可复核" ✓。
    #:   ⚠️ 三条与 ``engine_artifact_test`` ㊱¹³/㊱¹⁵ **同口径** ✓ —— 这组判据存在的理由就是
    #:   「一处改了另一处不改」那个老病根 ✓：链层不补，帧率这件事在成片路上仍然半核半不核 ✓。
    ok_payload = ok_card.to_dict()
    check("⑤⁵ ⭐ 成片路也交**帧率第一档** ✓：``fpsVerified == 'exact'`` ✓ ＋ ``fpsRequested`` ＝"
          "计划帧率 ✓ ＋ ``fpsOnDisk`` 读得上 ✓；而**没有问题**时顶层**不出现** ``alerts`` 键 ✓"
          "（缺键 ＝「一条都没有」✓ —— 不拿空数组去混淆「没说」✗）",
          ok_card.outputs.get("fpsVerified") == "exact"
          and int(ok_card.outputs.get("fpsRequested") or -1) == int(ok_plan.fps)
          and ok_card.outputs.get("fpsOnDisk") is not None
          and "alerts" not in ok_card.outputs
          and "alerts" not in ok_payload,
          {"fpsVerified": ok_card.outputs.get("fpsVerified"),
           "fpsRequested": ok_card.outputs.get("fpsRequested"),
           "fpsOnDisk": ok_card.outputs.get("fpsOnDisk"),
           "payloadAlerts": ok_payload.get("alerts", "<缺键>")})

    drift_request = pipe.GenerationRequest(prompt="链", seed=11, seconds=16.0)
    drift_plan = chain.plan_chain(drift_request)
    drift_fps = float(drift_plan.fps) / 2.0
    drift_card, _dr_dir, _dr_state = run("fps_drift", drift_request, fps=drift_fps)
    drift_payload = drift_card.to_dict()
    drift_alerts = drift_payload.get("alerts")
    check("⑤⁶ ⭐⭐ 盘上帧率**真被改了** ⇒ ``fpsVerified == 'drift'`` ✓ ＋ 一句人话进 ``alerts`` ✓"
          "**并且抬到回执顶层** ✓（前端只读 ``result.alerts`` ✓ —— 只写 ``note`` 等于没写 ✗）："
          f"顶层那句与 ``outputs['alerts']`` **逐字一致** ✓（请求 {int(drift_plan.fps)} fps vs "
          f"盘上 {drift_fps:g} fps ✓）",
          drift_card.outputs.get("fpsVerified") == "drift"
          and drift_card.outputs.get("fpsOnDisk") is not None
          and float(drift_card.outputs.get("fpsOnDisk") or 0) == drift_fps
          and isinstance(drift_alerts, list) and len(drift_alerts) == 1
          and "不一致" in drift_alerts[0]
          and list(drift_alerts) == list(drift_card.outputs.get("alerts") or [])
          and "不一致" in str(drift_card.outputs.get("note") or ""),
          {"fpsVerified": drift_card.outputs.get("fpsVerified"),
           "fpsOnDisk": drift_card.outputs.get("fpsOnDisk"),
           "payloadAlerts": drift_alerts,
           "outputsAlerts": drift_card.outputs.get("alerts")})

    blind_fps_request = pipe.GenerationRequest(prompt="链", seed=12, seconds=16.0)
    blind_fps_card, _bf_dir, _bf_state = run("fps_blind", blind_fps_request, fps=None)
    blind_fps_payload = blind_fps_card.to_dict()
    blind_fps_alerts = blind_fps_payload.get("alerts")
    check("⑤⁷ ⭐⭐ 帧率**读不到** ⇒ ``fpsVerified`` 是 ``None``（**不是** ``'exact'`` ✗）＋ 顶层一句"
          "「没读到 ⇒ 不知道」✓ —— 「没核过」与「一致」**不许同形** ✗（同一族：读数缺失被当成"
          "读数说没问题 ✓✗）",
          blind_fps_card.outputs.get("fpsVerified") is None
          and blind_fps_card.outputs.get("fpsOnDisk") is None
          and isinstance(blind_fps_alerts, list) and bool(blind_fps_alerts)
          and any("没读到" in item for item in blind_fps_alerts)
          and list(blind_fps_alerts) == list(blind_fps_card.outputs.get("alerts") or []),
          {"fpsVerified": blind_fps_card.outputs.get("fpsVerified"),
           "fpsOnDisk": blind_fps_card.outputs.get("fpsOnDisk"),
           "payloadAlerts": blind_fps_alerts})


def case_wiring() -> None:
    """接线守卫：扫源码 ✓（谁把接线删了谁红 ✓ —— 静态守卫见 cpu_budget_test 的先例 ✓）。"""
    engine = BACKEND_PY / "app" / "services" / "engine"
    runtime_src = (engine / "runtime.py").read_text(encoding="utf-8")
    pipeline_src = (engine / "pipeline.py").read_text(encoding="utf-8")
    check("⑥ 接线①：`runtime` 真的走链子 ✓（删了 `chain.run_chain` ⇒ 红 ✓）",
          "chain.plan_chain(" in runtime_src and "chain.run_chain(" in runtime_src)
    check("⑦ 接线②：单段那条分支**仍在** ✓（不许为了分段把原路删掉 ✗）",
          "if chain_plan is None:" in runtime_src
          and "pipe.run_sync(task.request, backend" in runtime_src)
    check("⑧ 接线③：`pipeline` 真的**调用**回调 ✓（只加参数不调 = 假接线 ✗）",
          "if on_decoded is not None:" in pipeline_src and "on_decoded(decoded)" in pipeline_src)
    inside = pipeline_src.split("decoded = backend.decode(latents, plan, request)", 1)[-1]
    inside = inside.split("except BaseException", 1)[0]
    check("⑨ 接线④：回调在 decode 的 try **里面** ✓（它炸了也算 decode 失败 ✓ 不破「不抛」契约 ✗）",
          "on_decoded(decoded)" in inside, inside[-60:])
    from app.services import engine as engine_pkg  # noqa: PLC0415

    check("⑩ 接线⑤：新模块登记进 `engine.__all__` ✓（模块清单是权威 ✓）",
          "chain" in engine_pkg.__all__)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="engine_chain_"))
    case_plan()
    case_stitch()
    case_normalize()
    case_run_chain(root)
    case_chain_frames(root)
    case_wiring()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name
              + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
