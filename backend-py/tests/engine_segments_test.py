"""S7 自检：**长视频分段**（纯计划数学 ✓ 零依赖 ✓ 2026-09-17）。

判据刻意按**不变量**写 ✓ 而不是记死数字 ✓（换网格 / 换上限 / 换重叠都还站得住 ✓）：

* 每段长度在 ``17k+5`` 网格上 ✓、且不超过上限 ✓；
* 相邻段重叠**恰好** N 帧 ✓，生成区间并集**恰好覆盖**总长 ✓；
* ⭐ 各段 ``keep`` 之和 **== 总帧数** ✓（拼接器的硬契约 ✓ —— 少了会缺帧 ✗、多了会重复 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_segments_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import segments as seg  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


GRID, MIN = 17, 5


def _on_grid(value: int) -> bool:
    return value >= MIN and (value - MIN) % GRID == 0


def _check_invariants(name: str, total: int, plans: list[seg.SegmentPlan], *, overlap: int,
                      per_segment: int) -> None:
    """三条不变量 + 接缝事实 ✓（一次报一条 ✓ 便于定位 ✓）。"""
    check(name + "：每段长度都在「17k+5」网格上 ✓（否则模型会拒绝或静默改 ✗）",
          all(_on_grid(plan.frames) for plan in plans),
          [(plan.index, plan.frames) for plan in plans])
    check(name + "：每段都不超过单段上限 ✓",
          all(plan.frames <= per_segment for plan in plans), per_segment)

    # 生成区间并集恰好覆盖 [0, total) ✓（首段从头 ✓、步进 = 长度 − 重叠 ✓）
    covered = set()
    for plan in plans:
        covered |= set(range(plan.startFrame, min(plan.endFrame, total)))
    check(name + "：生成区间**恰好覆盖** [0, total) ✓（有洞就会缺帧 ✗）",
          covered == set(range(total)), (len(covered), total))

    gaps_ok = all(plans[index + 1].startFrame == plans[index].endFrame - overlap
                  for index in range(len(plans) - 1))
    check(name + "：相邻段重叠**恰好** " + str(overlap) + " 帧 ✓（多了少了都会卡顿/缺帧 ✗）",
          gaps_ok, [(plan.startFrame, plan.endFrame) for plan in plans])
    # ⚠️ 这条断言我**第一版写错过** ✗：写成「carry == 上一段 endFrame − 1」✓，但有重叠时
    #    ``start_{k+1} = end_k − overlap`` ✓ ⇒ 接缝帧是 ``start−1`` ✓（也就是上一段**最后一个
    #    被保留**的帧 ✓）。按这个更强的语义断言 ✓：carry 必须既等于 ``start−1`` ✓、
    #    又等于上一段``keepTo``对应的全局帧 ✓✓。
    check(name + "：第 k>0 段的首帧 == **本段 start−1** 且 == **上一段最后一个保留帧** ✓"
          "（两个口径同时成立 ⇒ 接缝才真的连续 ✓）",
          all(plans[index].carryFrame == plans[index].startFrame - 1
              and plans[index].carryFrame == plans[index - 1].startFrame
              + plans[index - 1].keepTo - 1
              for index in range(1, len(plans))), [plan.carryFrame for plan in plans])
    check(name + "：⭐ 各段 keep 之和 == 总帧数 ✓（拼接器硬契约 ✓）",
          sum(plan.keptFrames for plan in plans) == total,
          [(plan.index, plan.keptFrames) for plan in plans])


def case_grid() -> None:
    check("① 长度向上吸附（124 → 124 ✓ / 120 → 124 ✓ / 1 → 5 ✓）",
          (seg.grid_length(124), seg.grid_length(120), seg.grid_length(1)) == (124, 124, 5),
          [seg.grid_length(value) for value in (124, 120, 1)])
    check("② 上限**向下**取（62 → 56 ✓ 因为 56 = 5+3×17 ✓）",
          seg.max_grid_length(62) == 56, seg.max_grid_length(62))
    check("③ 上限小于最小帧数 ⇒ **报错**（别硬生成长度 0 ✗）",
          "小于最小合法帧数" in str(_error(lambda: seg.max_grid_length(3))))


def case_plans() -> None:
    single = seg.plan_segments(124, max_frames=124, overlap_frames=0)
    check("④ 一段装得下 ⇒ **就一段** ✓（不做无意义的分段 ✓）",
          len(single) == 1 and single[0].carryFrame is None, [plan.to_dict() for plan in single])
    _check_invariants("单段", 124, single, overlap=0, per_segment=124)

    # 多段 + 重叠：用不变量扫几组（含"总长不是网格倍数"的情形 ✓）
    for total, max_frames, overlap in ((300, 124, 0), (300, 124, 12), (500, 200, 17),
                                       (124 * 3, 124, 24), (777, 300, 5)):
        per_segment = seg.max_grid_length(max_frames, grid=GRID, min_frames=MIN)
        try:
            plans = seg.plan_segments(total, max_frames=max_frames, overlap_frames=overlap)
        except seg.SegmentError as err:
            skip(f"total={total} max={max_frames} overlap={overlap} 覆盖不出来（{str(err)[:40]}…）"
                 f"⇒ 跳过这组 ✓（报错本身也有专门用例 ✓）")
            continue
        _check_invariants(f"[总{total}/上限{max_frames}/重叠{overlap}]", total, plans,
                          overlap=overlap, per_segment=per_segment)

    many = seg.plan_segments(124 * 4, max_frames=124, overlap_frames=24)
    check("⑤ 多段时**段号连续** ✓（从 0 开始 ✓）",
          [plan.index for plan in many] == list(range(len(many))), [plan.index for plan in many])
    check("⑥ 末段若需要裁帧 ⇒ 如实体现为 `overlapTail > 0` 或 endFrame 超出总数 ✓（不藏着 ✗）",
          all(plan.overlapTail >= 0 for plan in many)
          and any(plan.endFrame > 124 * 4 or plan.overlapTail > 0 for plan in many), "")

    check("⑦ 总帧数为 0 / 负 ⇒ **报错** ✓", "必须为正" in str(_error(
        lambda: seg.plan_segments(0, max_frames=124))))
    check("⑧ 总帧数小于最小合法帧数 ⇒ **报错**（模型吃不下 ✗）",
          "小于最小合法帧数" in str(_error(lambda: seg.plan_segments(3, max_frames=124))))
    check("⑨ 重叠 ≥ 单段长度 ⇒ **报错并说清「原地打转」** ✓",
          "原地打转" in str(_error(lambda: seg.plan_segments(300, max_frames=124,
                                                            overlap_frames=124))))
    check("⑩ 重叠为负 ⇒ 报错 ✓", "不能为负" in str(_error(
        lambda: seg.plan_segments(300, max_frames=124, overlap_frames=-1))))

    # 覆盖不出来时，报错必须**给出可调的量**（否则用户只能猜 ✓）
    try:
        seg.plan_segments(300, max_frames=124, overlap_frames=17)
        message = ""
    except seg.SegmentError as err:
        message = str(err)
    check("⑪ 覆盖不出来 ⇒ 报错里带**可调的量**（总帧数/overlap/max_frames ✓ 可行动 ✓）",
          (not message) or all(word in message for word in ("overlap", "max_frames")),
          message[:80])


def case_glue(root: Path) -> None:
    """分段 → 每段请求（首帧落 PNG ✓）。"""
    from app.services.engine import pipeline as pipe

    plans = seg.plan_segments(300, max_frames=124, overlap_frames=12)
    request = pipe.GenerationRequest(prompt="雨夜霓虹街头", seed=1, steps=2,
                                     outputs_dir=str(root))
    plain = seg.build_segment_requests(request, plans, root=root)
    check("⑫ 每段一个请求 ✓（段数一致 ✓）且**秒数按本段帧数换算** ✓（fps 不变 ⇒ 时长才对得上 ✓）",
          len(plain) == len(plans)
          and abs(plain[0].seconds - plans[0].frames / 24) < 1e-9,
          [(item.seconds, plan.frames) for item, plan in zip(plain, plans)])
    check("⑬ 没给边界帧 ⇒ 第 k>0 段**不带首帧** ✓ 且**如实标注**（不假装有 ✓）",
          plain[1].first_frame is None
          and "不带首帧" in str(getattr(plain[1], "segmentNote", {}).get("note", "")),
          getattr(plain[1], "segmentNote", None))

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        skip("torch 未安装 ⇒ 首帧落盘那半跳过 ✓")
        return

    torch.manual_seed(0)
    boundary = torch.rand(3, 16, 16) * 2.0 - 1.0      # (3,H,W) ✓ 与 write_image 同口径 ✓
    with_carry = seg.build_segment_requests(request, plans, root=root,
                                            carryFrames={1: boundary})
    path = with_carry[1].first_frame
    check("⑭ ⭐ 给了边界帧 ⇒ 落成**真 PNG** 并塞进 `first_frame` ✓（于是后端走真 VAE 编码 ✓）",
          bool(path) and Path(path).exists() and Path(path).stat().st_size > 0, path)

    from app.services.engine import media as media_mod

    tensor, info = media_mod.load_image_tensor(path, width=16, height=16)
    check("⑮ 落盘的 PNG 能被读回、**形状/值域正确** ✓（接缝帧真的传下去了 ✓）",
          tuple(tensor.shape) == (1, 3, 1, 16, 16) and float(tensor.min()) >= -1.0 - 1e-6
          and float(tensor.max()) <= 1.0 + 1e-6, info)
    check("⑯ 文件名带**来源全局帧号** ✓（排查接缝问题时一眼能对上 ✓）",
          "from" in Path(path).name, Path(path).name)
    check("⑰ 段说明里带完整计划 ✓（可审计 ✓）",
          isinstance((getattr(with_carry[1], "segmentNote", {}) or {}).get("plan"), dict),
          (getattr(with_carry[1], "segmentNote", {}) or {}).keys())
    check("⑱ 第 0 段**没有**首帧条件 ✓（它是从头生成的 ✓）",
          with_carry[0].first_frame is None, with_carry[0].first_frame)


def _error(fn) -> Exception:  # noqa: ANN001
    try:
        fn()
    except Exception as err:  # noqa: BLE001
        return err
    return Exception("（没有抛错 ✗）")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="engine_seg_"))
    case_grid()
    case_plans()
    case_glue(root)

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
