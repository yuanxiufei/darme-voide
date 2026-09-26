"""S7 自检：**档位表**（步数 / 分辨率 / 加速件 / 显存建议 / 必备模型 ✓ 零依赖 ✓ 2026-09-24）。

判据按**不变量**写 ✓（换一档数字也站得住 ✓）：

* 各档的必备模型**都以基础模型清单开头** ✓（漏了 CLIP/音频 VAE 这种错，部署时最贵 ✓✗）；
* **零第三方**那档的模型清单里**不许出现 LoRA** ✗（要么「零依赖」要么「带加速件」，不能两头都要 ✓✗）；
* **步数越少 ⇒ 倍率越大** ✓（反了就是表写错了 ✓）；
* ⭐ 不认识档位 ⇒ **报错并列出合法的** ✗，**不静默回落默认档** ✗✗；
* ⭐ **跨分辨率不估耗时** ✗✗（倍率是在各自分辨率下量的 ✓）；
* ⭐ 显存是**建议**不是门槛 ✓：没给显存要报「**没比**」✗ 而不是通过 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_tiers_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import geometry as geo  # noqa: E402
from app.services.engine import tiers as tiers_mod  # noqa: E402

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


def case_table() -> None:
    """① 三档表本身 ✓（口径来自参考实现的实测表 ✓ 见 `TOPICS.md` ✓）。"""
    order = [preset.key for preset in tiers_mod.presets()]
    check("① 三档齐 ✓ 且按**步数从多到少**排 ✓（= 从稳到快 ✓）",
          order == ["official", "std8", "fast4"], order)
    official = tiers_mod.resolve("official")
    check("①′ ``official`` = **零第三方依赖** ✓ 且 25 步 / 832×480 ✓；"
          "⚠️ 它的模型清单里**不许出现 LoRA** ✗（要么零依赖要么带加速件 ✓）",
          official.third_party_free and official.steps == 25 and official.size == (832, 480)
          and not any("LoRA" in model for model in official.models), official.to_dict())
    fast4 = tiers_mod.resolve("fast4")
    check("①″ ``fast4`` 带**实锤文件名**的 4 步 LoRA ✓（`minimax_h3_ref2v_turbo_4step` ✓）、4 步 ✓",
          fast4.steps == 4
          and any("minimax_h3_ref2v_turbo_4step" in model for model in fast4.models),
          fast4.to_dict())
    std8 = tiers_mod.resolve("std8")
    check("①‴ ``std8`` 只记**品类**（LightX2V 系 ✓）并**自己说明文件名未核** ✗ "
          "（⚠️ 把没核到的写成实锤就是造假 ✓✗）",
          std8.steps == 8 and any("LoRA" in model for model in std8.models)
          and any("未核" in model for model in std8.models), std8.to_dict())
    check("①⁴ ⭐ 每档的模型清单都以**基础模型清单**开头 ✓（漏 CLIP / 音频 VAE 这类错最贵 ✓✗）",
          all(preset.models[:len(tiers_mod.BASE_MODELS)] == tiers_mod.BASE_MODELS
              for preset in tiers_mod.presets()),
          [len(preset.models) for preset in tiers_mod.presets()])
    check("①⁵ 各档分辨率都落在 H3 的**分辨率步长**上 ✓（跨来源：`geometry.RESOLUTION_MULTIPLE` ✓）",
          all(preset.size[0] % geo.RESOLUTION_MULTIPLE == 0
              and preset.size[1] % geo.RESOLUTION_MULTIPLE == 0
              for preset in tiers_mod.presets()),
          [(preset.key, preset.size) for preset in tiers_mod.presets()])
    check("①⁶ 不变量：**步数越少 ⇒ 倍率越大** ✓（反了就是表写错 ✓）",
          all(a.steps > b.steps and a.speedup < b.speedup
              for a, b in zip(tiers_mod.presets(), tiers_mod.presets()[1:])),
          [(preset.key, preset.steps, preset.speedup) for preset in tiers_mod.presets()])


def case_lookup() -> None:
    """② 查档 ✓：反查得到 ✓、查不出 ⇒ 自定义 ✓、不认识 ⇒ **报错并列出合法的** ✗。"""
    check("② ``for_steps``：25/8/4 各对上 ✓；**20 步 ⇒ ``None``** ✓（= 自定义档 ✓ 不硬塞 ✗）",
          [tiers_mod.for_steps(steps).key if tiers_mod.for_steps(steps) else None
           for steps in (25, 8, 4, 20)] == ["official", "std8", "fast4", None])
    msg = _raises(lambda: tiers_mod.resolve("turbo"), "official")
    check("②′ ⭐ 档位不认识 ⇒ 报错且**列出合法的三个** ✓ 并写明**不静默回落默认档** ✗✗"
          "（回落会让「选了极速」变成「跑了稳妥」且不报错 ✓✗）",
          msg is not None and "不静默回落" in msg, msg)


def case_estimate() -> None:
    """③ 耗时估算 ✓：基准档 10~15 分钟 ✓、加速档更快 ✓、**跨分辨率一律拒** ✗✗。"""
    base = tiers_mod.estimate_minutes(tiers_mod.resolve("official"), 5.0)
    check("③ ``official`` 5 秒段 ⇒ **10~15 分钟** ✓（口径就是实测那一句 ✓）",
          base["low"] == 10.0 and base["high"] == 15.0, base)
    fast = tiers_mod.estimate_minutes(tiers_mod.resolve("fast4"), 5.0)
    mid = tiers_mod.estimate_minutes(tiers_mod.resolve("std8"), 5.0)
    check("③′ 加速档更快 ✓ 且 4 步比 8 步更快 ✓（倍率真的用上了 ✓ 不是摆设 ✗）",
          fast["high"] < mid["high"] < base["high"], (fast, mid))
    cross = _raises(lambda: tiers_mod.estimate_minutes(
        tiers_mod.resolve("std8"), 5.0, size=(832, 480)), "不套")
    check("③″ ⭐⭐ **跨分辨率不估** ✗✗：改分辨率还拿那个倍率 ⇒ 报错 ✓"
          "（倍率是在各自分辨率下量的 ✓ —— 套数字等于编 ✓）",
          cross is not None, cross)
    check("③‴ 同分辨率就放行 ✓（正向对照：上面那条不是「永远报错」的套套逻辑 ✓）",
          tiers_mod.estimate_minutes(tiers_mod.resolve("std8"), 5.0, size=(1024, 576))["low"] > 0
          and _raises(lambda: tiers_mod.estimate_minutes(
              tiers_mod.resolve("official"), 0)) is not None)


def case_vram() -> None:
    """④ 显存 ✓：**建议**不是门槛 ✓；没给显存 ⇒ 「**没比**」✗ 不是通过 ✓。"""
    official = tiers_mod.resolve("official")
    none = tiers_mod.advised(official, gpu_gib=None)
    check("④ 没给显存 ⇒ ``enough is None`` ✓ 且写明「**没比**（不是通过 ✓）」（本仓纪律 ✓）",
          none["enough"] is None and "没比" in none["note"], none)
    check("④′ 8G 对 ``official`` ⇒ 够 ✓；4G ⇒ 不够 ✓；两边都带上「不是硬门槛」的说明 ✓",
          tiers_mod.advised(official, gpu_gib=8)["enough"] is True
          and tiers_mod.advised(official, gpu_gib=4)["enough"] is False
          and "硬门槛" in tiers_mod.advised(official, gpu_gib=4)["note"])
    check("④″ 表里的建议值：official **8G** ✓、std8 **16G+** ✓（分辨率高一档 ⇒ 要求更高 ✓）、"
          "fast4 **8G 甜点** ✓（四个档的数字都来自实测表 ✓）",
          tiers_mod.resolve("official").vram_min_gib == 8.0
          and tiers_mod.resolve("std8").vram_min_gib == 16.0
          and tiers_mod.resolve("fast4").vram_min_gib == 8.0,
          [(preset.key, preset.vram_min_gib) for preset in tiers_mod.presets()])


def case_wired_into_readiness() -> None:
    """⑤ ⭐ **接线**：档位表必须出现在**就绪报告**里 ✓（本仓纪律：接不出去不算功能 ✗）。"""
    from app.services.engine import inventory as inv  # noqa: PLC0415 —— 只在接线这一条用 ✓

    report = inv.readiness("h3")
    block = report.get("tiers") or {}
    check("⑤ ⭐ 就绪报告里**真的带上了档位表** ✓（三档齐 ✓、每档都带**必备模型** ✓ —— "
          "免得部署时漏装 LoRA / 多装 LoRA ✓✗）",
          [item["key"] for item in block.get("presets", [])] == ["official", "std8", "fast4"]
          and all(item["models"] for item in block.get("presets", [])),
          block.get("presets", [{}])[:1])
    check("⑤′ ⚠️ 档位块里写明口径**边界** ✓：显存是建议 ✓、换机器/换分辨率要**重新量** ✗"
          "（别让人把倍率当承诺 ✓）",
          "建议" in str(block.get("note")) and "重新量" in str(block.get("note")),
          block.get("note"))


def main() -> int:
    case_table()
    case_lookup()
    case_estimate()
    case_vram()
    case_wired_into_readiness()
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
