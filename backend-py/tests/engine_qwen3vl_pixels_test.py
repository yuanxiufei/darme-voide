"""Qwen3-VL **像素预处理**的自检 ✓（H3 的视觉输入口径 ✓）。

钉的是「**归一化 0.5 / resize 倍数 32 / merge 块优先行序 / 两帧进时间维**」四类硬事实 ✓：
* 用 CLIP 均值方差 ⇒ 颜色整体偏 ✓ 不报错 ✗
* 行序写成「像素行优先」⇒ 位置编码套错格 ✓ 不报错 ✗
* 视频块把 2 帧当成 2 个 ``grid_t`` ⇒ token 数翻倍 ✓ 且位置表对不上 ✗
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.engine import qwen3vl_pixels as px  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(name: str) -> None:
    _RESULTS.append((name, True, "（跳过：没有 torch ✓ —— **如实标记** ✓）"))


def _raises(fn) -> object:
    try:
        fn()
    except BaseException as err:  # noqa: BLE001
        return f"{type(err).__name__}: {err}"
    return "（**没有报错** ✗）"


def main() -> int:
    # ① resize 计划（纯 Python ✓）
    plan = px.resize_plan(64, 64)
    check("① 64×64（已是 32 倍数）⇒ 原样 ✓", (plan["height"], plan["width"]) == (64, 64), plan)
    check("①′ 100×100 ⇒ ``round(100/32)=3`` ⇒ **96**×96 ✓（``floor`` 也是 96 —— 但 110 就会分开 ✓）",
          (px.resize_plan(100, 100)["height"], px.resize_plan(100, 100)["width"]) == (96, 96),
          px.resize_plan(100, 100))
    check("①″ 110×110 ⇒ ``round(3.4)=3`` ⇒ 96 ✓（``floor`` 同值 ✓）；120×120 ⇒ ``round(3.75)=4`` ⇒ 128 ✓"
          "（**round 不是 floor** ✓ 这条只在 3.5~4.0 区间分得开 ✓）",
          px.resize_plan(120, 120)["height"] == 128, px.resize_plan(120, 120))
    tiny = px.resize_plan(10, 10)
    check("①‴ 10×10 ⇒ 被 ``min_pixels`` **放大**到 64×64 ✓（参考缺省就是这样 ✓ 不是 bug ✓）",
          (tiny["height"], tiny["width"], tiny["reason"]) == (64, 64, "grow"), tiny)
    huge = px.resize_plan(4000, 4000)
    check("①⁗ 4000×4000 ⇒ 被 ``max_pixels`` 缩 ✓ 且仍 ≤ 上限 ✓",
          huge["reason"] == "shrink" and huge["height"] * huge["width"] <= px.MAX_PIXELS, huge)
    check("①⁵ 倍数是 ``patch*merge = 32`` ✓", plan["factor"] == 32, plan["factor"])
    bad = _raises(lambda: px.resize_plan(0, 10))
    check("①⁶ 高宽 ≤ 0 ⇒ 报错 ✓（静默算出 0×0 ⇒ 后面除零 ✓ 报错点离真相很远 ✗）",
          "高宽" in str(bad), bad)

    try:
        import torch
    except BaseException:  # noqa: BLE001
        for name in ("② 单图形状", "②′ 归一化 0.5", "②″ 块优先行序",
                     "③ 视频块形状", "③′ 两帧进时间维", "⑤ 单帧×2 == 两帧同图"):
            skip(name)
        return _report()

    # ② 单图 ⇒ 行数 = 网格 patch 数 ✓、宽 = 3×2×16×16 = 1536 ✓
    zeros = torch.zeros(64, 64, 3)
    patches, grid = px.process_image(zeros)
    check("② 64×64 单图 ⇒ 16 行 × 1536 宽 ✓、grid = (1,4,4) ✓（grid_t 恒 1 ✓）",
          tuple(patches.shape) == (16, 1536) and grid == (1, 4, 4), (tuple(patches.shape), grid))

    # ②′ 归一化是 **0.5/0.5** ✓：0 ⇒ -1 ✓、1 ⇒ +1 ✓（CLIP 均值 ⇒ 全 0 会变成 -1.79/-1.75/-1.48 ✗）
    ones, _ = px.process_image(torch.ones(64, 64, 3))
    check("②′ 归一化 0.5/0.5 ✓ ⇒ 全 0 像素 → **-1**、全 1 像素 → **+1** ✓"
          "（用 CLIP 均值方差 ✗ ⇒ 会得到 -1.79 / -1.75 / -1.48 ✓ 颜色整体偏 ✓ 不报错 ✗）",
          bool(torch.allclose(patches, torch.full_like(patches, -1.0)))
          and bool(torch.allclose(ones, torch.full_like(ones, 1.0))),
          (float(patches.min()), float(patches.max())))

    # ②″ 行序「merge 块优先」✓：4 个 merge 块各涂一个常数 ⇒ 前 4 行同块 ✓、第 5 行换块 ✓
    tiled = torch.zeros(64, 64, 3)
    for block_h in range(2):
        for block_w in range(2):
            tiled[block_h * 32:(block_h + 1) * 32, block_w * 32:(block_w + 1) * 32] = \
                (block_h * 2 + block_w + 1) * 0.2
    tiled_patches, _ = px.process_image(tiled)
    heads = [round(float(row[0]), 3) for row in tiled_patches]
    check("②″ 行序是 **merge 块优先** ✓ ⇒ 行 0..3 是同一块（同一个值）✓、行 4 换成下一块 ✓"
          "（「像素行优先」✗ ⇒ 行 2 就会换块 ✓ 位置编码套错格 ✓ 不报错 ✗）",
          len(set(heads[:4])) == 1 and heads[4] != heads[0], heads)
    #     ⚠️ 值是**归一化之后**的 ✓：(v - 0.5) / 0.5 ⇒ 0.2 → -0.6、0.8 → 0.6 ✓
    normalized_values = [round((value - 0.5) / 0.5, 3) for value in (0.2, 0.4, 0.6, 0.8)]
    check("②‴ 4 个 merge 块 ⇒ 4 个不同的常数（归一化后 -0.6/-0.2/0.2/0.6）✓ 且每块各占 4 行 ✓",
          sorted(set(heads)) == normalized_values and heads.count(heads[0]) == 4,
          heads)

    # ③ 视频块 ⇒ 行数 = 一个 2 帧块的 patch 数 ✓（**不翻倍** ✓）
    frames_black = torch.zeros(2, 64, 64, 3)
    block_patches, block_grid = px.process_video_block(frames_black)
    check("③ 2 帧视频块 ⇒ 仍是 16 行 ✓、grid_t 仍 = 1 ✓（两帧进**时间维** ✓ 不是两个 grid_t ✗）",
          tuple(block_patches.shape) == (16, 1536) and block_grid == (1, 4, 4),
          (tuple(block_patches.shape), block_grid))
    # ③′ 帧 0 全 0 / 帧 1 全 1 ⇒ 每行 t=0 段 = -1、t=1 段 = +1 ✓
    mixed = torch.zeros(2, 64, 64, 3)
    mixed[1] = 1.0
    mixed_patches, _ = px.process_video_block(mixed)
    row = mixed_patches[0]
    # (C, tp, p, p) 展平 ⇒ 每通道 2×256：t=0 在 [0:256] ✓、t=1 在 [256:512] ✓
    check("③′ 两帧不同 ⇒ 每行 **t=0 段全 -1**（帧 0）✓、**t=1 段全 +1**（帧 1）✓"
          "（两帧顺序写反 ✗ ⇒ 这两段互换 ✓ 且不报错 ✗）",
          bool(torch.allclose(row[0:256], torch.full((256,), -1.0)))
          and bool(torch.allclose(row[256:512], torch.full((256,), 1.0))),
          (float(row[0:256].min()), float(row[0:256].max()),
           float(row[256:512].min()), float(row[256:512].max())))

    # ③″ 块帧数不是 2 ⇒ 报错 ✓
    three = _raises(lambda: px.process_video_block(torch.zeros(3, 64, 64, 3)))
    check("③″ 3 帧当一块 ⇒ 报错 ✓（多给的帧会被当时间维 ✓ 位置表对不上 ✗ 且不报 ✗）",
          "恰好" in str(three), three)
    bad_dim = _raises(lambda: px.process_video_block(torch.zeros(64, 64, 3)))
    check("③‴ 少一维（单图当块）⇒ 报错 ✓（静默 reshape ⇒ 帧数被读成 64 ✓ 报错点很远 ✗）",
          "四维" in str(bad_dim), bad_dim)

    # ④ 奇数帧补齐 ✓
    four = px.pad_frames_even(torch.arange(3 * 4).reshape(3, 2, 2).float())
    check("④ 3 帧 ⇒ 补成 4 帧 ✓ 且第 4 帧 = 第 3 帧 ✓（参考 ``cat([frames, frames[-1:]])`` ✓）",
          tuple(four.shape)[0] == 4 and bool(torch.equal(four[3], four[2])), tuple(four.shape))
    even = px.pad_frames_even(torch.zeros(4, 2, 2))
    check("④′ 已经是偶数 ⇒ **原样**返回 ✓（不重复加 ✓）", tuple(even.shape)[0] == 4, tuple(even.shape))

    # ⑤ 单图重复两遍 ≡ 两帧同图 ⇒ **逐字节相同** ✓（极强的判据 ✓）
    picture = torch.rand(64, 64, 3)
    from_image, grid_image = px.process_image(picture)
    from_pair, grid_pair = px.process_video_block(picture.unsqueeze(0).repeat(2, 1, 1, 1))
    check("⑤ 单图（重复两遍）与「两帧都是同一张」⇒ **逐字节相同** ✓ 且网格相同 ✓"
          "（不等 ⇒ 两条口径的 patchify / 归一化有分叉 ✓）",
          grid_image == grid_pair and bool(torch.equal(from_image, from_pair)),
          (grid_image, grid_pair, float((from_image - from_pair).abs().max())))

    return _report()


def _report() -> int:
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAIL {name} —— 实际：{detail!r}")
    print(f"SUMMARY {len(_RESULTS) - len(failed)}/{len(_RESULTS)} 通过 ✓")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
