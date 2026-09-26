"""S7 自检：**超清放大器的网络本体**（V2 主干 + V3 因子化注意力 ✓ 2026-09-24）。

⚠️ 判据全是**不变量** ✓（换任何配置都站得住 ✓），并且**只谈结构与形状** ✗ ——
「画质」要真权重 + 真采样才谈得上 ✓✗（本仓纪律：不把没验的说成验过 ✗）。

钉的几条：

* ⭐ **`T` 不变** ✓✗：2× 放大**只动空间**（H/W 翻倍 ✓）—— 时间维插值会让动作速率变错 ✓✗；
* ⭐ **残差恒等式** ✓：`forward(x) == bilinear2x(x) + correction(x)`（V2 ✓）、
  `forward(x) == base(x) + delta(x)`（V3 ✓）—— 说明「网络只学修正量」✓；
* ⭐ **严格装载** ✗✗：键少一个 / 多一个 ⇒ **报错**（写明「架构不符」✓）⇒ 不许 `strict=False` 糊过去 ✗；
* ⚠️ **没装 torch ⇒ 显式 SKIP** ✓（**没跑 ≠ 绿** ✗）；`_HAS_TORCH=False` 时 `build_upscaler` 要**明确报错** ✗
  （用 monkeypatch 造这个场景 ✓ —— 本仓纪律：缺依赖那类路径用 monkeypatch ✓ 别写成「必然缺」✓✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_upscale_net_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import upscale_net as net  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: Any, needle: str = "") -> str | None:
    try:
        call()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        return text if needle in text else None
    return None


TOY_V2 = {"in_channels": 4, "hidden_channels": 8, "num_blocks": 2, "refine_channels": 6,
          "refine_blocks": 1, "temporal_kernel": 3}
TOY_V3 = {"width": 8, "blocks": 2, "heads": 2, "window": 2, "mlp_ratio": 2}


def case_constants() -> None:
    """① 常量与分组数 ✓（口径来自上游 ✓）。"""
    check("① 低清段的膨胀循环固定为 ``(1, 2, 1, 3)`` ✓（口径 ✓ 按 ``i % 4`` 取 ✓）",
          net.RESIDUAL_DILATIONS == (1, 2, 1, 3), net.RESIDUAL_DILATIONS)
    check("①′ ``groups_for`` 从 ``min(16, C)`` **往下**找能整除的 ✓（24 ⇒ 12 ✓、128 ⇒ 16 ✓、"
          "7 ⇒ 7 ✓；⚠️ 不是「贪心取大」也不是「必须 16」✗）",
          net.groups_for(24) == 12 and net.groups_for(128) == 16 and net.groups_for(7) == 7,
          (net.groups_for(24), net.groups_for(128), net.groups_for(7)))


def case_space_ops() -> None:
    """② 两个空间算子 ✓：**只动空间、不动时间** ✗。"""
    import torch  # noqa: PLC0415

    x = torch.zeros(1, 4, 3, 4, 4)
    up = net.spatial_bilinear_2x(x)
    check("② ``spatial_bilinear_2x``：H/W ×2 ✓、**通道不变** ✓、⭐ **``T`` 不变** ✗✗",
          tuple(up.shape) == (1, 4, 3, 8, 8), tuple(up.shape))
    shuffled = net.spatial_pixel_shuffle_2x(torch.zeros(1, 24, 3, 4, 4), 6)
    check("②′ ``spatial_pixel_shuffle_2x``：``C_入 = out×4`` ⇒ H/W ×2 ✓、通道 = out ✓、T 不变 ✓",
          tuple(shuffled.shape) == (1, 6, 3, 8, 8), tuple(shuffled.shape))
    check("②″ 通道数不是 ``out×4`` ⇒ **报错** ✗（不许硬凑 ✗ —— 那说明结构/权重装错了 ✓✗）",
          _raises(lambda: net.spatial_pixel_shuffle_2x(torch.zeros(1, 5, 3, 4, 4), 6),
                  "× 4") is not None)


def case_v2() -> None:
    """③ V2 主干 ✓：形状 + ⭐ **残差恒等式**。"""
    import torch  # noqa: PLC0415

    v2 = net.H3LatentUpscalerV2(**TOY_V2).eval()
    x = torch.randn(1, 4, 3, 4, 4)
    with torch.no_grad():
        out = v2(x)
    check("③ V2 前向：输出 `(B, 24→in, T, 2H, 2W)` ✓（**T 不变** ✗、通道回到 ``in_channels`` ✓）",
          tuple(out.shape) == (1, 4, 3, 8, 8), tuple(out.shape))
    with torch.no_grad():
        manual = net.spatial_bilinear_2x(x) + v2.correction(x)
    check("③′ ⭐ **残差恒等式** ✓：``forward(x) == bilinear2x(x) + correction(x)``"
          "（⇒ 网络只学**修正量** ✓、基底天然在输出里 ✓）",
          bool(torch.equal(out, manual)), float((out - manual).abs().max()))
    check("③″ ``refine_stem`` 的输入通道 = ``refine_channels + in_channels`` ✓"
          "（⚠️ 口径：**把双线性基底拼进通道** ✓）",
          v2.refine_stem.in_channels == TOY_V2["refine_channels"] + TOY_V2["in_channels"],
          v2.refine_stem.in_channels)


def case_v3() -> None:
    """④ V3 ✓：内含 V2 当 base ✓、shifted 交替 ✓、形状 ✓。"""
    import torch  # noqa: PLC0415

    v3 = net.H3LatentUpscalerV3(TOY_V2, TOY_V3).eval()
    x = torch.randn(1, 4, 3, 4, 4)
    with torch.no_grad():
        out = v3(x)
    check("④ V3 前向：形状与 V2 一致 ✓（T 不变 ✓ H/W 翻倍 ✓）",
          tuple(out.shape) == (1, 4, 3, 8, 8), tuple(out.shape))
    shifted = [bool(block.shift) for block in v3.blocks]
    check("④′ ⭐ **shifted 按 ``index % 2`` 交替** ✓（第 0 块不挪 ✓、第 1 块挪半窗 ✓ —— "
          "不交替的话窗口边界**永远落在同一处** ✗✗）",
          shifted == [False, True], shifted)
    check("④″ V3 **内含** V2（``v3.base`` ✓ ⇒ 两层都是残差 ✓：内 `bilinear+correction` ✓、外 `base+delta` ✓）",
          isinstance(v3.base, net.H3LatentUpscalerV2)
          and v3.to_delta.out_channels == TOY_V2["in_channels"] * 4
          and v3.to_delta.in_channels == TOY_V3["width"], v3.to_delta.out_channels)


def case_build() -> None:
    """⑤ 装配 ✓：往返能过 ✓、键不符**必拒** ✗、没 torch **必报** ✗。"""
    import torch  # noqa: PLC0415

    model = net.H3LatentUpscalerV3(TOY_V2, TOY_V3)
    state = dict(model.state_dict())
    built = net.build_upscaler(state, base_config=TOY_V2, config=TOY_V3)
    check("⑤ 用自己刚建出来的 state_dict **往返装载**成功 ✓、且 ``eval()`` + 不训练梯度 ✓",
          built.training is False
          and all(not parameter.requires_grad for parameter in built.parameters()), built.training)
    missing = {key: value for key, value in state.items() if not key.endswith("mlp.2.weight")}
    check("⑤′ ⭐ 少一个键 ⇒ **报错**并写明「张量与声明的架构**不符**」✓✗（不许 ``strict=False`` 糊 ✗）",
          _raises(lambda: net.build_upscaler(missing, base_config=TOY_V2, config=TOY_V3),
                  "不符") is not None)
    extra = {**state, "不存在的键": torch.zeros(1)}
    check("⑤″ 多一个键 ⇒ 也**报错** ✓（多余张量说明检查点不是这套架构 ✓✗）",
          _raises(lambda: net.build_upscaler(extra, base_config=TOY_V2, config=TOY_V3),
                  "不符") is not None)

    original = net._HAS_TORCH          # noqa: SLF001
    net._HAS_TORCH = False             # noqa: SLF001 —— monkeypatch 造「没装 torch」✓
    try:
        msg = _raises(lambda: net.build_upscaler(state, base_config=TOY_V2, config=TOY_V3), "不静默降级")
        check("⑤‴ ⚠️ ``_HAS_TORCH=False``（monkeypatch ✓）⇒ **明确报错** ✗ 且指向**回退普通模式**那条路 ✓"
              "（缺依赖那类路径**必须用 monkeypatch 造** ✓ —— 别写成「本机必然缺」✗）",
              msg is not None, msg)
    finally:
        net._HAS_TORCH = original      # noqa: SLF001


def main() -> int:
    if not net.has_torch():
        skip("本机没装 torch ⇒ 网络本体**没跑** ✓（不是通过 ✗）")
    else:
        case_constants()
        case_space_ops()
        case_v2()
        case_v3()
        case_build()
    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
