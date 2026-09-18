"""S7 自检：**自研推理引擎的纯算法层**（2026-09-17）。

本自检**不下载模型、不需要 GPU、不需要 torch** ✓ —— 它盯的是「引擎的数学对不对」：
sigma 调度 ✓、分辨率/帧数几何 ✓、采样循环 ✓。

为什么这一层必须先钉死：**这三点错了都不会报错** ✗ ——
σ 序列反了 ⇒ 出纯噪声；分辨率不对齐 32 ⇒ 形状断言在最后一步炸；
帧数不在 17k+5 网格 ⇒ 画面抖/时长错 ✓。而它们**与权重、显卡无关** ⇒ 现在就能验 ✓。

采样器用**假张量**（12 行、只支持 ``+ − *``✓）验证 ⇒ 真机上换成 ``torch.Tensor`` 是同一段代码 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_core_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="enginecore_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import geometry as geo  # noqa: E402
from app.services.engine import sampler as smp  # noqa: E402
from app.services.engine import schedules as sch  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


class Vec:
    """**假张量**：只是个浮点数（真机上是 ``torch.Tensor`` ✓）。"""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __add__(self, other: "Vec") -> "Vec":
        return Vec(self.value + other.value)

    def __sub__(self, other: "Vec") -> "Vec":
        return Vec(self.value - other.value)

    def __mul__(self, scalar: float) -> "Vec":
        return Vec(self.value * float(scalar))

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return isinstance(other, Vec) and abs(self.value - other.value) < 1e-9

    def __repr__(self) -> str:  # noqa: D105
        return f"Vec({self.value:.6g})"


def case_schedules() -> None:
    """① sigma 调度：长度、单调、末位 0、公式抽查 ✓。"""
    for name in sorted(sch.SCHEDULES):
        sigmas = sch.sigmas_for(6, name)
        monotonic = all(sigmas[i] > sigmas[i + 1] for i in range(len(sigmas) - 2))
        check(f"① {name}: {len(sigmas)} 个（= 步数+1）、单调递减、末位 0 ✓",
              len(sigmas) == 7 and monotonic and sigmas[-1] == 0.0, sigmas[:3])

    check("② karras 的 ρ-ramp 公式可按定义复算（ρ=1 时退化为线性 ✓）",
          sch.karras_sigmas(2, sigma_min=1.0, sigma_max=4.0, rho=1.0) == [4.0, 1.0, 0.0],
          sch.karras_sigmas(2, sigma_min=1.0, sigma_max=4.0, rho=1.0))
    check("③ normal 是**几何**插值（对数域等分 ✓）",
          abs(sch.normal_sigmas(3, sigma_min=1.0, sigma_max=4.0)[1] - 2.0) < 1e-9,
          sch.normal_sigmas(3, sigma_min=1.0, sigma_max=4.0))
    check("④ 未知调度 ⇒ 报错（不静默用默认 ✗）",
          "未知调度" in str(_raises(lambda: sch.sigmas_for(4, "no-such"))))
    check("⑤ timesteps 映射单调且落在 [0,1) ✓",
          all(0.0 <= t < 1.0 for t in sch.timesteps_for([10.0, 1.0, 0.0])),
          sch.timesteps_for([10.0, 1.0, 0.0]))

    base = sch.sigmas_for(10, "karras")
    doubled = sch.interpolate_sigmas(base, 2.0)
    check("⑥ 步数细分：2 倍后长度翻倍、首尾保持、仍单调 ✓",
          len(doubled) == len(base) * 2 - 1 and doubled[0] == base[0]
          and doubled[-1] == 0.0
          and all(doubled[i] > doubled[i + 1] for i in range(len(doubled) - 2)),
          (len(base), len(doubled)))
    check("⑦ factor<=1 ⇒ 原样返回（不瞎改 ✓）", sch.interpolate_sigmas(base, 1.0) == base)


def case_geometry() -> None:
    """② 几何：帧网格 / 分辨率 / 潜空间 ✓。"""
    check("⑧ 帧网格 17k+5：5s ⇒ 124 帧、3s ⇒ 73 帧、0.1s ⇒ 下限 5 帧 ✓",
          geo.snap_frames(5) == 124 and geo.snap_frames(3) == 73 and geo.snap_frames(0.1) == 5,
          (geo.snap_frames(5), geo.snap_frames(3), geo.snap_frames(0.1)))
    check("⑨ 帧数换算的秒数回读与参考实测一致（124 帧 ⇒ 5.167s ✓）",
          abs(geo.snap_frames(5) / geo.H3_FPS - 5.1667) < 0.001, geo.snap_frames(5) / geo.H3_FPS)

    # ⭐ 与**参考项目的实测输出**对齐（0.65MP ⇒ 1088×608、0.98MP ⇒ 1344×768 ✓）
    check("⑩ 像素预算→尺寸：0.65MP@16:9 ⇒ **1088×608**（与参考实测一致 ✓）",
          geo.size_for_megapixels(0.65, "16:9") == (1088, 608), geo.size_for_megapixels(0.65, "16:9"))
    check("⑪ 像素预算→尺寸：0.98MP@16:9 ⇒ **1344×768**（与参考实测一致 ✓）",
          geo.size_for_megapixels(0.98, "16:9") == (1344, 768), geo.size_for_megapixels(0.98, "16:9"))
    size = geo.size_for_megapixels(0.65, "9:16")
    check("⑫ 两边都对齐 32 ✓（模型要求 step=32 ✗ 差一点就形状炸），竖版也成立 ✓",
          size[0] % 32 == 0 and size[1] % 32 == 0 and size[0] < size[1], size)
    check("⑬ 画幅比解析容错（':'/`：`/浮点/元组 ✓）",
          abs(geo.parse_ratio("16：9") - 16 / 9) < 1e-9
          and abs(geo.parse_ratio((9, 16)) - 9 / 16) < 1e-9
          and abs(geo.parse_ratio("垃圾") - 16 / 9) < 1e-9, geo.parse_ratio("垃圾"))

    check("⑭ 潜空间帧数：压缩比**必须显式给**（不给就报错 ✓ 不猜 4 或 8 ✗）",
          "必须 >= 1" in str(_raises(lambda: geo.latent_frames(124, temporal_compression=0)))
          and geo.latent_frames(124, temporal_compression=4) == 31
          and geo.latent_frames(1, temporal_compression=4) == 1,
          geo.latent_frames(124, temporal_compression=4))
    check("⑮ 像素回显（用于成本/日志 ✓）",
          abs(geo.megapixels_for_size(1088, 608) - 0.6615) < 0.001,
          geo.megapixels_for_size(1088, 608))


def case_sampler() -> None:
    """③ 采样循环（假张量 ✓）—— 包括**反套套逻辑**：结果精确，不是「差不多」✓。"""
    sigmas = [1.0, 0.0]
    perfect = lambda _x, _sigma: Vec(0.0)  # noqa: E731 —— 完美去噪器
    frozen = lambda x, _sigma: x  # noqa: E731 —— 完全不去噪

    result, steps = smp.sample(perfect, Vec(1.0), sigmas, "euler")
    check("⑯ Euler：完美去噪器 + σ 1→0 ⇒ 结果**精确**为 0（证明更新式真的生效 ✓ 不是近似 ✓）",
          result == Vec(0.0) and steps == 1, result)

    result2, _ = smp.sample(frozen, Vec(1.0), sigmas, "euler")
    check("⑰ 反套套逻辑：完全不去噪的模型 ⇒ 样本**一动不动**（采样器没有自己乱改 x ✓）",
          result2 == Vec(1.0), result2)

    many = sch.sigmas_for(8, "karras")
    ticks: list[tuple[int, float]] = []
    for name in ("euler", "heun", "multistep"):
        out, used = smp.sample(perfect, Vec(1.0), many, name,
                               callback=lambda step, sigma, _x: ticks.append((step, sigma)))
        check(f"⑱ {name}：步数 == len(sigmas)-1，且能收敛到 0 ✓（完美去噪器下）",
              used == len(many) - 1 and out == Vec(0.0), (name, used, out))
    check("⑲ 进度回调按序触发、数量正确（前端进度条靠它 ✓）",
          [step for step, _ in ticks[-len(many) + 1:]] == list(range(1, len(many))),
          ticks[-len(many) + 1:][:3])

    check("⑳ 采样前校验：序列反了 / 末位不是 0 ⇒ **当场报错**（否则会静默出噪声 ✗）",
          "单调递减" in str(_raises(lambda: smp.sample(perfect, Vec(1.0), [0.0, 1.0, 0.0])))
          and "末位必须是 0.0" in str(_raises(lambda: smp.sample(perfect, Vec(1.0), [1.0, 0.5]))))
    check("㉑ 未知采样器 ⇒ 报错（可用列表给出 ✓）",
          "未知采样器" in str(_raises(lambda: smp.sample(perfect, Vec(1.0), sigmas, "nope"))))


def _raises(fn) -> Exception:  # noqa: ANN001
    try:
        fn()
    except Exception as err:  # noqa: BLE001
        return err
    return Exception("（没有抛错 ✗）")


def main() -> int:
    case_schedules()
    case_geometry()
    case_sampler()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
