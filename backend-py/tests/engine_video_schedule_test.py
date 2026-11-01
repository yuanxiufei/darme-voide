"""自检：**视频族的 σ 日程**（flow 域 ✓ 2026-10-31）。

为什么值得单独立一份自检 ✗（`flow_sigmas_for_steps` 早被别的自检盖过 ✓ 按说不用再测 ✓）：

真正没被测过的是 **plan 那一层的默认值** ✓✗ —— ``GenerationRequest.schedule`` 的旧默认是 ``karras`` ✓，
它的首值是 **14.6146** ✓（EDM 域的上界），而 H3 采样器把 σ 当 **flow 域** 用 ✓：

==================  ====================================  ==========================
                   SDXL / karras 一族 ✓                    H3 视频（flow ✓）
==================  ====================================  ==========================
σ 取值范围          ``0.0292 … 14.6146`` ✓                 **恒在 [0, 1]** ✓
主干里的 t          ``σ → 0..999`` 整数格 ✓                **``t = 1 − σ``** ✓
欧拉步              ——                                    ``x += (σ_next − σ)·v`` ✓
音频那一路          ——                                    ``time_shift_sigma(σ, 12, 3)`` ✓
==================  ====================================  ==========================

⇒ 拿 karras 的 σ 喂 H3：``t = 1 − 14.6 = −13.6`` ✓、音频 shift 的**分母翻号** ✓
⇒ **形状全对、不报错、出来是噪声/糊片** ✓✗（正是本仓最忌的那一类 ✓）。

本文件钉三件事 ✓：
① `flow_shift_for_sequence` 的**闭式**（两端定值 ✓、`√frames` ✓、两端外不夹断 ✓）；
② plan 侧的**默认就是 flow** ✓，且与 `h3_form` 的出厂 shift（12.0 ✓）**逐点相等** ✓；
③ **反向证据** ✓：karras 的 σ 进来 ⇒ `t<0` 且音频 shift 算成负数 ⇒ 判据能证明"域不重叠"✓。

⚠️ 公式是**闭式数学** + 参考口径（``arXiv:2501.12375`` 的 ``time_shift`` / 分辨率-帧数自适应 α ✓）；
**一行代码都不抄** ✗（两套参考仓分别是 Apache-2.0 与 GPL-3.0 ✓，抄进来就说不清了 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_video_schedule_test.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import geometry, latent_formats, pipeline, schedules  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _error_of(call) -> str:  # noqa: ANN001
    try:
        call()
    except Exception as error:  # noqa: BLE001
        return str(error)
    return ""


def case_shift_formula() -> None:
    """① 自适应 shift 的闭式：两端定值 / 中点 / √frames / 外推不夹断 / 非法参数报错。"""
    line = schedules.flow_shift_for_sequence

    check("① α(seq_min=256, frames=1) = 1.0 ✓（直线左端 ✓）",
          abs(line(256) - 1.0) < 1e-12, line(256))
    check("①′ α(seq_max=4096, frames=1) = 3.0 ✓（直线右端 ✓）",
          abs(line(4096) - 3.0) < 1e-12, line(4096))
    # 手算：中点 (256+4096)/2 = 2176 ⇒ 1 + 2·(2176−256)/3840 = 2 ✓
    check("①″ α(2176, frames=1) = 2.0 ✓（中点 ✓ 公式写错必红 ✓）",
          abs(line(2176) - 2.0) < 1e-12, line(2176))

    # √frames：9 帧 = 3 倍、4 帧 = 2 倍 ✓。写成线性（×frames）⇒ 这两条当场为假 ✓。
    check("①‴ α(frames=4) = 2·α(frames=1) ✓（√ 不是线性 ✓）",
          abs(line(1024, frames=4) - 2 * line(1024)) < 1e-12, line(1024, frames=4))
    check("①⁗ α(frames=9) = 3·α(frames=1) ✓", abs(line(1024, frames=9) - 3 * line(1024)) < 1e-12,
          line(1024, frames=9))

    # 外推不夹断：seq=64 落在 256 左边 ⇒ 1 − 2·192/3840 = 0.9 ✓（**小于** base_shift ✓ 但仍正 ✓）
    want_left = 1.0 + (3.0 - 1.0) * (64 - 256) / (4096 - 256)
    check("①⁵ 短序列外推 ⇒ 0.9 ✓（不夹到 1.0 ✓ —— 夹断会让「更短」与「256 长」完全同档 ✓✗）",
          abs(line(64) - want_left) < 1e-12 and line(64) < 1.0, line(64))
    want_right = 1.0 + (3.0 - 1.0) * (8192 - 256) / (4096 - 256)
    check("①⁶ 长序列外推 ⇒ 5.133… ✓（不夹到 3.0 ✓）",
          abs(line(8192) - want_right) < 1e-12 and line(8192) > 3.0, line(8192))

    check("①⁷ α 恒为正 ✓（seq→0 也不翻号 ✓）", line(1) > 0, line(1))

    text = _error_of(lambda: schedules.flow_shift_for_sequence(1024, seq_min=512, seq_max=512))
    check("①⁸ seq_min == seq_max ⇒ 报错 ✓（直线画不出来 ✓ 不静默除零 ✗）",
          "不能相同" in text, text)
    text = _error_of(lambda: schedules.flow_shift_for_sequence(1024, frames=0))
    check("①⁹ frames=0 ⇒ 报错 ✓", "frames" in text, text)


def case_video_sigmas() -> None:
    """② 视频 σ 序列：端点 / 长度 / 手算 / 单调 / shift 越大越偏噪 / 缺参数报错。"""
    sigmas = schedules.flow_sigmas_for_video(8, shift=12.0)

    check("② 长度 = 步数 + 1 ✓", len(sigmas) == 9, len(sigmas))
    check("②′ σ₀ = 1 ✓（flow 的不动点 ✓ 与 shift 无关 ✓）", sigmas[0] == 1.0, sigmas[0])
    check("②″ σ_last = 0 ✓", sigmas[-1] == 0.0, sigmas[-1])
    check("②‴ 严格单调递减 ✓", all(a > b for a, b in zip(sigmas, sigmas[1:-1])), sigmas)

    # 手算：σ(t) = 12t/(1+11t) ⇒ t=0.875 ⇒ 10.5/10.625 = 0.988235…
    want = 12 * 0.875 / (1 + 11 * 0.875)
    check("②⁗ 第 2 点 = 12·0.875/(1+11·0.875) ≈ 0.9882 ✓", abs(sigmas[1] - want) < 1e-9,
          (sigmas[1], want))
    check("②⁵ shift=12 比 shift=3 每一步都更偏噪 ✓（视频口径不是图族那个 3 ✓✗）",
          all(a > b for a, b in zip(sigmas[1:-1],
                                    schedules.flow_sigmas_for_steps(8, shift=3.0)[1:-1])),
          sigmas)

    # 自适应形与显式 shift 形**必须同源** ✓（一个手算 α、一个让它自己算 ⇒ 两处写错必红 ✓）
    seq, frames = 1024, 33
    alpha = schedules.flow_shift_for_sequence(seq, frames=frames)
    auto = schedules.flow_sigmas_for_video(8, seq_len=seq, latent_frames=frames)
    check("②⁶ seq_len 形 == 手算 α 的 shift 形 ✓（逐点 ✓）",
          all(abs(a - b) < 1e-12 for a, b in zip(auto, schedules.flow_sigmas_for_steps(8, shift=alpha))),
          (auto, alpha))

    text = _error_of(lambda: schedules.flow_sigmas_for_video(8))
    check("②⁷ 既不给 shift 也不给 seq_len ⇒ 报错 ✓（不偷偷拿图族的 3 ✓✗）",
          "shift" in text, text)


def _request(**kwargs):  # noqa: ANN001, ANN202
    base = {"prompt": "a cat walking", "stage": pipeline.VIDEO_STAGE, "seconds": 5.0,
            "steps": 8, "megapixels": 0.65, "temporal_compression": 4}
    base.update(kwargs)
    return pipeline.GenerationRequest(**base)


def case_plan_default_flow() -> None:
    """③ plan 侧默认：flow ✓ 且与出厂 shift（12.0）逐点相等 ✓；karras 显式要走 ⇒ 如实报警 ✓。"""
    plan = pipeline.build_plan(_request())

    check("③ 默认 schedule 就是 flow ⇒ σ₀ = 1 ✓（旧默认 karras 会给 14.6146 ✓✗）",
          plan.sigmas[0] == 1.0, plan.sigmas[0])
    check("③′ σ 全程不出 [0,1] ✓（flow 域 ✓）",
          max(plan.sigmas) == 1.0 and min(plan.sigmas) == 0.0, (max(plan.sigmas), min(plan.sigmas)))
    check("③″ 长度 = 步数 + 1 ✓", len(plan.sigmas) == plan.steps + 1, len(plan.sigmas))

    want = schedules.flow_sigmas_for_steps(plan.steps, shift=pipeline.H3_FLOW_SHIFT)
    check("③‴ 与 H3_FLOW_SHIFT 那条网格**逐点相等** ✓（换默认值必须动这里的判据 ✓）",
          all(abs(a - b) < 1e-12 for a, b in zip(plan.sigmas, want)),
          (plan.sigmas[:3], want[:3]))

    plan6 = pipeline.build_plan(_request(schedule="flow:6"))
    check("③⁗ schedule='flow:6' ⇒ 就是 shift=6 ✓（扫 shift 的直接口 ✓）",
          all(abs(a - b) < 1e-12 for a, b in
              zip(plan6.sigmas, schedules.flow_sigmas_for_steps(plan6.steps, shift=6.0))),
          plan6.sigmas[:3])

    plan_k = pipeline.build_plan(_request(schedule="karras"))
    check("③⁵ 显式 karras 照跑 ✓（旧路不删 ✓ 但 max σ > 1 ✓ 说明是 EDM 域 ✓）",
          max(plan_k.sigmas) > 1.0, max(plan_k.sigmas))
    check("③⁶ 且**如实报警** ✓（域不重叠不能一声不响 ✓✗）",
          any("超出 flow" in text for text in plan_k.warnings), plan_k.warnings)
    # 反向证据：karras 首值喂 H3 ⇒ t = 1 − σ < 0 ✓（这就是"不报错、出噪声"的物理原因 ✓）
    check("③⁷ 反向证据：max(karras) = 14.61… ⇒ t = 1−σ < 0 ✓（H3 会当真 ✓✗）",
          abs(max(plan_k.sigmas) - 14.6146) < 1e-3 and 1.0 - max(plan_k.sigmas) < 0,
          1.0 - max(plan_k.sigmas))


def case_flow_auto() -> None:
    """④ `flow:auto`：按序列长与潜帧数自适应 ✓；缺件 / 坏数字都当场拒 ✓。"""
    width, height = geometry.size_for_megapixels(0.65, "16:9")
    side = latent_formats.get_latent_format("h3_video").spatial_downscale_ratio
    seq = ((height // side) // 2) * ((width // side) // 2)
    frames = geometry.latent_frames(geometry.snap_frames(5.0, fps=geometry.H3_FPS),
                                    temporal_compression=4)
    alpha = schedules.flow_shift_for_sequence(seq, frames=frames)

    plan = pipeline.build_plan(_request(schedule="flow:auto"))
    want = schedules.flow_sigmas_for_steps(plan.steps, shift=alpha)
    check("④ flow:auto ⇒ 与本文件手算的 α 逐点相等 ✓（plan 里的 token 数没算错 ✓）",
          all(abs(a - b) < 1e-12 for a, b in zip(plan.sigmas, want)),
          (plan.sigmas[:3], want[:3], seq, frames))
    check("④′ 算出的 α 落在一个**正常人**的范围 ✓（<50 ✓ —— 拿像素数当 seq 会去到几百 ✓✗）",
          0 < alpha < 50, alpha)

    text = _error_of(lambda: pipeline.build_plan(_request(schedule="flow:auto",
                                                          temporal_compression=None)))
    check("④″ 不给 temporalCompression ⇒ 报错 ✓ 且点名 flow:auto ✓（不猜压缩比 ✓）",
          "flow:auto" in text, text)

    text = _error_of(lambda: pipeline.build_plan(_request(schedule="flow:abc")))
    check("④‴ 坏数字 ⇒ 报错 ✓ 并列出可用形 ✓（不回落默认 ✓✗）",
          "可用形" in text, text)


def case_wiring() -> None:
    """⑤ 接线守卫：`pipeline.H3_FLOW_SHIFT` 与 `h3_form` 的出厂事实**必须同值** ✓。"""
    try:
        from app.services.engine import h3_form  # noqa: PLC0415
    except Exception as error:  # noqa: BLE001
        skip(f"h3_form 读不到（{error} ✓）⇒ 接线守卫本环境跳过 ✓")
        return

    fact = h3_form.H3_SIGMA_SHIFTS["sigma_shift_video"]
    check("⑤ pipeline.H3_FLOW_SHIFT == h3_form 的 sigma_shift_video ✓（只改一处 ⇒ 红 ✓）",
          float(pipeline.H3_FLOW_SHIFT) == float(fact), (pipeline.H3_FLOW_SHIFT, fact))
    check("⑤′ 音频那一路是**另一个数** ✓（12 / 3 ✓ 两路不共用 ✓）",
          float(h3_form.H3_SIGMA_SHIFTS["sigma_shift_audio"]) != float(fact),
          h3_form.H3_SIGMA_SHIFTS)

    patch = tuple(h3_form.H3_TRUNK_DEFAULTS["patch_size"])
    check("⑤″ patch 的时间维**不 patch** ✓（`(1, 2, 2)` ✓ ⇒ 每帧一组 token ✓）",
          patch[0] == 1 and patch[1:] == (2, 2), patch)
    got = latent_formats.get_latent_format("h3_video").spatial_downscale_ratio
    check("⑤‴ 空间下采样 16 ✓（`_video_token_count` 的常数与它同源 ✓）", int(got) == 16, got)


def case_audio_shift_domain() -> None:
    """⑥ 域不重叠的**第二份证据**：karras 的 σ 过音频 shift 换算 ⇒ 负数 ✓✗。"""
    karras_first = max(schedules.sigmas_for(8, "karras"))
    shifted = schedules.time_shift_sigma(karras_first, 12.0, 3.0)
    check("⑥ karras 首值过 time_shift_sigma(·, 12, 3) ⇒ **负数** ✓（分母翻号 ✓ 这就是「不报错但坏」✓）",
          shifted < 0, (karras_first, shifted))
    check("⑥′ 而 flow 的 σ ∈ [0,1] 过同一个换算 ⇒ 仍落在 [0,1] ✓（域对了就不会翻号 ✓）",
          all(0.0 <= schedules.time_shift_sigma(s, 12.0, 3.0) <= 1.0
              for s in schedules.flow_sigmas_for_steps(8, shift=12.0)),
          [round(schedules.time_shift_sigma(s, 12.0, 3.0), 4)
           for s in schedules.flow_sigmas_for_steps(8, shift=12.0)])
    check("⑥″ σ=1 是任意 shift 的不动点 ✓（⇒ 视频/音频的**起点**本来就是同一档 ✓）",
          schedules.time_shift_sigma(1.0, 12.0, 3.0) == 1.0,
          schedules.time_shift_sigma(1.0, 12.0, 3.0))


def case_flow_resolver() -> None:
    """⑦ flow 字符串的**唯一解析处**：`is_flow_schedule` 判定 + `video_flow_shift_for` 三形 ✓。

    为什么要单独测解析 ✗（σ 序列本身上面已经测过了 ✓）：**两个**消费点要问同一句话 ✓ ——
    video plan 与二采（`torch_backend._second_pass` ✓）；任何一处自己写
    `spec.startswith("flow")` ✓ 都会在裸 `flow` 上分叉 ✓✗（裸形不带冒号 ✓ ⇒ 判定为假 ⇒
    静默走 karras ⇒ 又回到 14.61 ✓✗）。
    """
    for text in ("flow", "FLOW", " flow ", "", "flow:", "flow:auto", "flow:6", "flow:default"):
        check(f"⑦ is_flow_schedule({text!r}) = True ✓", pipeline.is_flow_schedule(text), text)
    for text in ("karras", "exponential", "normal", "ddim_uniform", "flo", "flows"):
        check(f"⑦ is_flow_schedule({text!r}) = False ✓", not pipeline.is_flow_schedule(text), text)

    request = _request()
    check("⑦′ 裸 flow ⇒ 就是出厂 12.0 ✓（不是 3、也不是 0.69 ✓）",
          pipeline.video_flow_shift_for(request) == pipeline.H3_FLOW_SHIFT,
          pipeline.video_flow_shift_for(request))
    check("⑦″ flow:6 ⇒ 6.0 ✓", pipeline.video_flow_shift_for(_request(schedule="flow:6")) == 6.0,
          pipeline.video_flow_shift_for(_request(schedule="flow:6")))

    width, height = geometry.size_for_megapixels(0.65, "16:9")
    side = latent_formats.get_latent_format("h3_video").spatial_downscale_ratio
    seq = ((height // side) // 2) * ((width // side) // 2)
    frames = geometry.latent_frames(geometry.snap_frames(5.0, fps=geometry.H3_FPS),
                                    temporal_compression=4)
    want = schedules.flow_shift_for_sequence(seq, frames=frames)
    got = pipeline.video_flow_shift_for(_request(schedule="flow:auto"))
    check("⑦‴ flow:auto 只给 request 也能算 ✓（二采没有 plan 可借 ✓ 这条就是它的通路 ✓）",
          abs(got - want) < 1e-12, (got, want))

    text = _error_of(lambda: pipeline.video_flow_shift_for(_request(schedule="karras")))
    check("⑦⁗ 非 flow 族 ⇒ 抛 ValueError ✓（**不**静默回落 12 ✓✗ 由调用方决定报警 ✓）",
          "不是 flow 族" in text, text)


def main() -> int:
    case_shift_formula()
    case_video_sigmas()
    case_plan_default_flow()
    case_flow_auto()
    case_wiring()
    case_audio_shift_domain()
    case_flow_resolver()

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
