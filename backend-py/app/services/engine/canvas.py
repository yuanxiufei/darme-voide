"""**画布计划层** ✓ —— 「这次出多大」的**唯一计算处**，与 :mod:`app.services.engine.upscale` 同构。

## 为什么单独一层（照抄已踩过的教训 ✓）

超清那条路的口径是「**计划层定能不能/怎么做 ✓、管线只执行计划** ✗」——
理由写在本仓 ``routers/engine.py::_upscale_from`` 的注释里：**在请求侧再判一遍就会出现"两处口径"** ✓✗
（H3 权重那件事刚踩过 ✓）。画布现在是**六处**各自算（五个图片族 + 视频 ✓，每处一段
``0.65 ⇒ 1.76/1.0`` 的替换 ✓）⇒ 同一张画布的口径分散在四套常量里 ✓✗。

⇒ 这一层只做三件事 ✓：
1. **几何**：像素 ↔ 比例 ↔ 吸附步长（**复用** :mod:`app.services.engine.geometry` ✓ 不写第二套算术 ✗）；
2. **查档**：用途名 → :class:`app.services.engine.profiles.CanvasProfile` ✓（数在表里 ✓ 不在代码里 ✗）；
3. **如实报**：来源、改过什么、以及"该 stage 有哪些档位" ✓。

## 三条纪律

* **吸附步长按族给** ✗ 不全局一个数：SDXL 8 ✓（VAE 1/8）、H3 32 ✓（schema ``step=32``）、
  图族 16 ✓（VAE 8× × patch 2× / 2.1 的 VAE 16×）。
* **给了宽高就不再算 MP** ✗✗：``1344×768`` 与 ``16:9`` **不闭合**（1.75 vs 1.778 ✓），
  用 MP 重算会落到 ``1360×768`` ✓✗ —— 真值是像素 ✓。
* **没有数就说没有** ✗：表里 ``width=None`` 的档（H3 导演台）⇒ 明确报错 ✓，不填一个"看着合理"的 ✗。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from . import geometry, profiles

__all__ = ["CanvasPlan", "STAGE_CANVAS", "stage_canvas_spec", "plan_canvas"]

#: 每个 stage 的**几何口径** ✓：``(吸附步长, 尺寸算法)``。
#:
#: * ``"resolution-selector"`` ⇒ :func:`geometry.size_for_resolution_selector` ✓
#:   （``round`` + ``1024²`` 基数 ✓，与 ComfyUI 节点逐字一致 ✓）——
#:   只有**官方画布正好等于节点输出**的族能用 ✓（``zimage`` / ``krea2`` ✓）；
#: * ``"megapixels"`` ⇒ :func:`geometry.size_for_megapixels` ✓（``ceil`` + ``1e6`` ✓）——
#:   ``qwen_image`` / ``qwen_image21`` 的官方画布是 ``1328×1328`` ✓，节点口径会给 ``1360×1360`` ✓✗
#:   ⇒ 这两族**不能**换口径 ✓（换了就与模板 ``#7.width/height`` 对不上 ✓）。
STAGE_CANVAS: dict[str, tuple[int, str]] = {
    "h3": (32, "megapixels"),
    "sdxl": (8, "megapixels"),
    "zimage": (16, "resolution-selector"),
    "qwen_image21": (16, "megapixels"),
    "qwen_image": (16, "megapixels"),
    "krea2": (16, "resolution-selector"),
}

#: 比例不闭合的容差 ✓（``1344/768=1.75`` vs ``16/9=1.7778`` 差 **1.59%** ✓ ⇒
#: 3% 内视为"同一个画幅"✓，超过就是用户真的要换比例 ✓✗）。
RATIO_TOLERANCE = 0.03


@dataclass(frozen=True)
class CanvasPlan:
    """一次画布决策的结果 ✓（**纯几何** + 来源 ✓，可直接当缓存键 ✓）。"""

    width: int
    height: int
    megapixels: float
    ratio: str
    grid: int
    scheme: str
    #: ``"request"`` / ``"profile:<id>"`` / ``"profile:<id>+ratio"`` ✓
    source: str = "request"
    #: 档位 id（没说用途就是 ``None`` ✓）
    profile: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width, "height": self.height, "megapixels": self.megapixels,
            "ratio": self.ratio, "grid": self.grid, "scheme": self.scheme,
            "source": self.source, "profile": self.profile, "notes": list(self.notes),
        }


def stage_canvas_spec(stage: str) -> tuple[int, str]:
    """``stage`` → ``(grid, scheme)`` ✓；不认识的 stage ⇒ **报错并列出合法的** ✗✗。"""
    key = str(stage or "").strip().lower()
    spec = STAGE_CANVAS.get(key)
    if spec is None:
        raise ValueError(
            f"未知 stage {stage!r} ✗ ⇒ 不知道该用哪个吸附步长 ✓；可用：{sorted(STAGE_CANVAS)} ✓")
    return spec


def _ratio_text(width: int, height: int) -> str:
    """像素 → 可读比例 ✓（能约成整数就约 ✓，否则原样 ✓）。"""
    divisor = math.gcd(int(width), int(height))
    return f"{int(width) // divisor}:{int(height) // divisor}"


def _snap(width: int, height: int, grid: int) -> tuple[int, int, bool]:
    """两边**就近**吸到 ``grid`` 的倍数 ✓（就近不向上：向上会白算像素 ✓✗）。"""
    snapped_w = geometry.snap_to_multiple(width, grid)
    snapped_h = geometry.snap_to_multiple(height, grid)
    return snapped_w, snapped_h, (snapped_w, snapped_h) != (int(width), int(height))


def plan_canvas(stage: str, *, target: str | None = None, megapixels: float | None = None,
                ratio: str | float | tuple[float, float] | None = None,
                width: int | None = None, height: int | None = None) -> CanvasPlan:
    """决定「这次画布多大」✓ ⇒ :class:`CanvasPlan` ✓。

    三种入口（**优先级从高到低** ✓，同时给两种 ⇒ 报错 ✗ 不猜哪个优先 ✓✗）：

    1. ``width`` + ``height``：显式像素 ✓（只做吸附 ✓）；只给一个 ⇒ 报错 ✗（另一个不猜 ✗）；
    2. ``target``：查 :data:`profiles.PROFILES` ✓ —— 给了 ``ratio`` 且与档位比例差 >3% ⇒
       **按档位面积重算**并写明 ✓（不是悄悄改用户的画幅 ✗）；
    3. ``megapixels`` + ``ratio``：按该 stage 的算法算 ✓（口径见 :data:`STAGE_CANVAS` ✓）。

    都没有 ⇒ 报错 ✓（"不给就明说"✓ 不编一个默认 ✗）。
    """
    grid, scheme = stage_canvas_spec(stage)
    notes: list[str] = []

    has_pixels = width is not None or height is not None
    if has_pixels:
        if width is None or height is None:
            raise ValueError(
                f"画布只给了 {'width' if width is not None else 'height'} ✗ —— "
                "另一边**不猜** ✓✗（猜出来的比例必然是别的用途 ✓）⇒ 两个都给，或用 target / megapixels ✓")
        if target not in (None, ""):
            raise ValueError(
                f"同时给了像素与用途档（{target!r}）✗ ⇒ 两处口径 ✓✗ —— 二选一 ✓："
                "要档位就只给 target ✓、要指定像素就只给 width/height ✓")
        if megapixels is not None:
            raise ValueError(
                "同时给了像素与 megapixels ✗ ⇒ 两处口径 ✓✗ —— 二选一 ✓（像素更准 ✓）")
        final_w, final_h, snapped = _snap(int(width), int(height), grid)
        if snapped:
            notes.append(
                f"尺寸已吸附到 {grid} 的倍数：{int(width)}×{int(height)} ⇒ "
                f"**{final_w}×{final_h}** ✓（该 stage 的潜空间按 1/{grid} 采样 ✓）")
        return CanvasPlan(width=final_w, height=final_h,
                          megapixels=geometry.megapixels_for_size(final_w, final_h),
                          ratio=str(ratio) if ratio is not None else _ratio_text(final_w, final_h),
                          grid=grid, scheme=scheme, source="request", notes=tuple(notes))

    if target not in (None, ""):
        profile = profiles.resolve(stage, target)
        if profile.width is None or profile.height is None:
            raise ValueError(
                f"档位 {profile.id!r} **没有定画布** ✗（{profile.label or '该用途'}："
                f"{profile.source or '出处未记'} ✓）—— ⚠️ 这不是缺省值 ✗：该用途的画布由"
                "工作流里的节点内部生成 ✓（例如 H3 导演台 ✓）⇒ 请显式给 width/height 或 megapixels ✓，"
                "本表**不猜** ✗")
        base_w, base_h = int(profile.width), int(profile.height)
        base_aspect = base_w / base_h
        if ratio is None:
            return CanvasPlan(width=base_w, height=base_h,
                              megapixels=geometry.megapixels_for_size(base_w, base_h),
                              ratio=_ratio_text(base_w, base_h), grid=grid, scheme=scheme,
                              source=f"profile:{profile.id}", profile=profile.id,
                              notes=(f"画布取自档位 **{profile.id}** ✓（{profile.label} ✓，"
                                     f"出处 {profile.source} ✓）",))
        wanted = geometry.parse_ratio(ratio)
        if abs(wanted - base_aspect) <= RATIO_TOLERANCE * base_aspect:
            notes.append(
                f"画布取自档位 **{profile.id}** ✓（{base_w}×{base_h} ✓）—— 请求比例 "
                f"{wanted:.4f} 与档位比例 {base_aspect:.4f} 差 "
                f"{abs(wanted - base_aspect) / base_aspect * 100:.2f}% ✓ ≤ "
                f"{RATIO_TOLERANCE * 100:.0f}% ⇒ **保留档位像素** ✓（拿 MP 重算会落到别的格 ✓✗）")
            return CanvasPlan(width=base_w, height=base_h,
                              megapixels=geometry.megapixels_for_size(base_w, base_h),
                              ratio=_ratio_text(base_w, base_h), grid=grid, scheme=scheme,
                              source=f"profile:{profile.id}", profile=profile.id, notes=tuple(notes))
        area_mp = geometry.megapixels_for_size(base_w, base_h)
        cost = max(0.05, area_mp)
        new_w, new_h = _size_for(scheme, cost, ratio, grid)
        notes.append(
            f"档位 **{profile.id}** 的画布是 {base_w}×{base_h} ✓（比例 {base_aspect:.4f} ✓），"
            f"而请求要 {wanted:.4f} ✓ ⇒ 差 {abs(wanted - base_aspect) / base_aspect * 100:.2f}% > "
            f"{RATIO_TOLERANCE * 100:.0f}% ⇒ **按档位面积（{area_mp:g} MP）与请求比例重算** ⇒ "
            f"{new_w}×{new_h} ✓（步数/cfg 仍按该档 ✓）")
        return CanvasPlan(width=new_w, height=new_h,
                          megapixels=geometry.megapixels_for_size(new_w, new_h),
                          ratio=_ratio_text(new_w, new_h), grid=grid, scheme=scheme,
                          source=f"profile:{profile.id}+ratio", profile=profile.id,
                          notes=tuple(notes))

    if megapixels is None:
        raise ValueError(
            "画布三样都没给 ✗（target / megapixels / width+height ✓）⇒ **不猜** ✗✗；"
            f"{stage} 的可选档位：{[p.target for p in profiles.for_stage(stage)]} ✓")

    cost = max(0.05, float(megapixels))
    final_w, final_h = _size_for(scheme, cost, ratio if ratio is not None else "16:9", grid)
    notes.append(
        f"按 **{scheme}** 口径算 ✓：{cost:g} MP ⇒ **{final_w}×{final_h}** ✓"
        f"（吸附步长 {grid} ✓）")
    return CanvasPlan(width=final_w, height=final_h,
                      megapixels=geometry.megapixels_for_size(final_w, final_h),
                      ratio=str(ratio) if ratio is not None else "16:9",
                      grid=grid, scheme=scheme, source="request", notes=tuple(notes))


def _size_for(scheme: str, megapixels: float, ratio: Any, grid: int) -> tuple[int, int]:
    """按 stage 的算法算尺寸 ✓（**两条口径唯一的汇合点** ✓ —— 别在别处再写一遍 ✗）。"""
    if scheme == "resolution-selector":
        return geometry.size_for_resolution_selector(megapixels, ratio, multiple=grid)
    return geometry.size_for_megapixels(megapixels, ratio, multiple=grid)
