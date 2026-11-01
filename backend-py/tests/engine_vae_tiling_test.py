"""VAE 分块缩放自检 ✓（`vae_tiling` 落户 ✓）。

**判据不能是"跑通就行"** ✗：分块最典型的坏法是「**不报错，只是图上有接缝/边角变暗**」✓✗，
所以这里全用**可精确比较**的标尺 ✓：

* 恒等函数 ``f(x)=x``（upscale=1 ✓）⇒ 分块结果必须与输入**逐位相同** ✓
  —— 抓「权重没归一 / 漏了边缘块 / 末块没贴边」✓✗；
* 逐元素非线性 ``f(x)=sin(3x)``（局部算子 ✓）⇒ 分块结果必须与整张运算**逐位相同** ✓
  —— 抓「重叠带没做平均 / 拼错位置」✓✗；
* 常数张量 ⇒ 输出仍是**同一个常数** ✓ —— 抓「除法归一漏了 / 边角被压暗」✓✗（那条最像"光照"）；
* 步数统计（进度条总数）必须等于实际调用次数 ✓ —— 抓「进度条数与真实块数对不上」✓✗。

末尾几条专打**参数非法**档（tile 太小 / overlap 不足 1 个潜像素 / overlap ≥ tile ✓）：
真实症状全是"能跑完、只是接缝难看" ✓✗ ⇒ 必须**当场抛** ✓。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import vae_tiling as vt  # noqa: E402

try:
    import torch
except Exception:  # noqa: BLE001  （没有 torch ⇒ 这一套确实做不了 ✓，明说跳过 ✓）
    torch = None  # type: ignore[assignment]

_FAILS: list[str] = []
_PASSED = 0


def check(label: str, ok: bool, detail: object = "") -> None:
    global _PASSED
    if ok:
        _PASSED += 1
        print(f"PASS  {label}")
        return
    _FAILS.append(label)
    print(f"FAIL  {label}  ->  {detail!r}")


def _rand(*shape: int) -> "torch.Tensor":
    generator = torch.Generator().manual_seed(20261031)
    return torch.randn(*shape, generator=generator)


def case_identity() -> None:
    """① 恒等 + upscale=1 ⇒ **逐位相同** ✓（含非整除的怪尺寸 ✓）。"""
    for shape, tile in (((1, 4, 37, 53), (16, 16)),
                        ((1, 4, 64, 64), (64, 64)),      # 单块 ✓
                        ((2, 3, 9, 11), (4, 4))):        # 批 > 1 ✓、tile 比图大 ✓
        source = _rand(*shape)
        got = vt.tiled_scale(source, lambda t: t, tile, overlap=max(1, tile[0] // 4), upscale=1)
        check(f"① 恒等 tile={tile} shape={shape} ⇒ 与输入逐位相同 ✓", torch.equal(got, source),
              float((got - source).abs().max()))


def case_pointwise() -> None:
    """② 逐元素非线性 ⇒ 与整张运算**逐位相同** ✓（重叠带平均是对的 ✓）。"""
    source = _rand(1, 3, 50, 67)
    whole = torch.sin(3.0 * source)
    got = vt.tiled_scale(source, lambda t: torch.sin(3.0 * t), (16, 16), 4, upscale=1)
    check("② 逐元素 sin(3x)：分块 = 整张 ✓（重叠带没做平均就会在这里露 ✓✗）",
          torch.equal(got, whole), float((got - whole).abs().max()))


def case_constant_and_upscale() -> None:
    """③ 常数张量 + upscale ⇒ 常数不变 ✓（除归一漏了 ⇒ 边角被压暗 ✗）。"""
    source = torch.full((1, 4, 12, 20), 0.75)
    got = vt.tiled_scale(source, lambda t: t.repeat_interleave(4, -1).repeat_interleave(4, -2),
                         (6, 6), 2, upscale=4)
    check("③ 常数 0.75 × 4 倍放大 ⇒ 整张仍是 0.75 ✓✗（不除 ``div`` 的实现边角会变暗 ✓）",
          tuple(got.shape) == (1, 4, 48, 80) and float((got - 0.75).abs().max()) < 1e-6,
          (tuple(got.shape), float((got - 0.75).abs().max())))


def case_steps() -> None:
    """④ 步数统计 = 实际调用次数 ✓（进度条总数对不上就会卡在 99% ✓✗）。"""
    calls = {"n": 0}
    seen: list[tuple[int, int]] = []

    def counted(piece: object) -> object:
        calls["n"] += 1
        return piece

    source = _rand(1, 2, 40, 55)
    vt.tiled_scale(source, counted, (16, 16), 8, upscale=1,
                   progress=lambda done, total: seen.append((done, total)))
    want = vt.tiled_scale_steps((40, 55), (16, 16), 8)
    check("④ 进度条总数 = 实际切块数 ✓", calls["n"] == want and seen == [(i, want) for i in
                                                                   range(1, want + 1)],
          (calls["n"], want, seen[:3]))
    check("④′ 同一张起点表：`tile_starts` 的块数 × 各轴 = 总步数 ✓",
          want == len(vt.tile_starts(40, 16, 8)) * len(vt.tile_starts(55, 16, 8)), want)
    check("④″ 末块**贴住右下角** ✓（否则最边上一条永远没解码 ✓✗）",
          vt.tile_starts(40, 16, 8)[-1] == 40 - 16 and vt.tile_starts(55, 16, 8)[-1] == 55 - 16,
          (vt.tile_starts(40, 16, 8), vt.tile_starts(55, 16, 8)))


def case_video() -> None:
    """⑤ 5 维（视频）只切**空间** ⇒ 与逐帧 2 维结果一致 ✓。"""
    source = _rand(1, 2, 3, 24, 24)
    per_frame = torch.stack([vt.tiled_scale(source[:, :, i], lambda t: t, (8, 8), 2)
                            for i in range(source.shape[2])], dim=2)
    got = vt.tiled_scale(source, lambda t: t, (8, 8), 2, upscale=1)
    check("⑤ 视频按帧切（tile 只用 2 维）⇒ 与逐帧结果逐位相同 ✓", torch.equal(got, per_frame),
          float((got - per_frame).abs().max()))


def case_guards() -> None:
    """⑥ 非法参数**当场抛** ✓✗（症状都是"能跑完、接缝难看"✓）。"""
    cases = {
        "overlap ≥ tile": lambda: vt.tile_starts(100, 8, 8),
        "tile ≤ 0": lambda: vt.tile_starts(100, 0, 0),
        "overlap < 0": lambda: vt.tile_starts(100, 8, -1),
        "tile 维数不对": lambda: vt.tiled_scale(_rand(1, 2, 8, 8), lambda t: t, (4, 4, 4)),
        "样本维数对不上": lambda: vt.tiled_scale(_rand(1, 2, 8), lambda t: t, (4, 4)),
        "upscale < 1": lambda: vt.tiled_scale(_rand(1, 2, 8, 8), lambda t: t, (4, 4), 1,
                                              upscale=0),
    }
    for label, call in cases.items():
        try:
            call()
        except vt.TileError:
            check(f"⑥ {label} ⇒ 抛 TileError ✓", True)
        except BaseException as error:  # noqa: BLE001
            check(f"⑥ {label} ⇒ 抛 TileError ✓", False, f"{type(error).__name__}: {error}")
        else:
            check(f"⑥ {label} ⇒ 抛 TileError ✓", False, "没抛 ✗")

    #: ⚠️ 形状不符**不会**自己报错 ✗（拼回来就是错位的图 ✓✗）⇒ 必须点名 ✓。
    try:
        vt.tiled_scale(_rand(1, 2, 16, 16), lambda t: t, (8, 8), 2, upscale=2)
    except vt.TileError as error:
        check("⑥′ function 输出形状不对 ⇒ 抛错并点名 upscale/通道 ✓",
              "upscale" in str(error) or "形状" in str(error), str(error))
    else:
        check("⑥′ function 输出形状不对 ⇒ 抛错并点名 upscale/通道 ✓", False, "没抛 ✗")


def case_latent_tile() -> None:
    """⑦ 像素 ⇒ 潜像素的换算 ✓（用户按潜像素填是这里最容易踩的错 ✓）。"""
    check("⑦ 512 px / 64 px @8× ⇒ (64, 8) 个潜像素 ✓", vt.latent_tile(512, 64, 8) == (64, 8),
          vt.latent_tile(512, 64, 8))
    for label, args in (("tile 只够 1 个潜像素", (8, 64, 8)),
                        ("overlap 不足 1 个潜像素", (512, 4, 8)),
                        ("overlap 超过半个 tile", (64, 32, 8))):
        try:
            vt.latent_tile(*args)
        except vt.TileError as error:
            check(f"⑦ {label} ⇒ 抛错 ✓✗（否则接缝静默出现 ✓）", True)
        else:
            check(f"⑦ {label} ⇒ 抛错 ✓✗（否则接缝静默出现 ✓）", False, args)


def case_decode_wrapper() -> None:
    """⑧ `decode_latents_tiled` 的 8× 放大链 ✓（潜变量 ⇒ 像素的整条口径 ✓）。"""
    latents = _rand(1, 4, 16, 20)
    got = vt.decode_latents_tiled(lambda t: t.repeat_interleave(8, -1).repeat_interleave(8, -2),
                                 latents, spatial_scale=8, tile=64, overlap=16)
    want = latents.repeat_interleave(8, -1).repeat_interleave(8, -2)
    check("⑧ 8× 解码链：tile=64 px ⇒ 8 个潜像素 ✓，拼回与整张逐位相同 ✓",
          tuple(got.shape) == (1, 4, 128, 160) and torch.equal(got, want),
          (tuple(got.shape), float((got - want).abs().max())))
    check("⑧′ 默认档就是 ComfyUI 的 512/64 ✓",
          vt.TILE_DEFAULTS == {"tile": 512, "overlap": 64}, vt.TILE_DEFAULTS)


def main() -> int:
    if torch is None:
        print("SKIP  没装 torch ✗ ⇒ 分块缩放整套做不了 ✓（本仓不静默降级 ✓）")
        return 0
    case_identity()
    case_pointwise()
    case_constant_and_upscale()
    case_steps()
    case_video()
    case_guards()
    case_latent_tile()
    case_decode_wrapper()

    total = _PASSED + len(_FAILS)
    print(f"\nSUMMARY: {_PASSED}/{total} passed")
    for label in _FAILS:
        print(f"  FAILED: {label}")
    return 0 if not _FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
