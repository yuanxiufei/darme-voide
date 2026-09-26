"""S7 自检：**H3 prompt 接缝**（提交前那一道关 ✓ 零依赖 ✓ 2026-09-24）。

钉的是「**没这道关就会静默出错**」的两条 ✗✗：

* ⭐ **编号越界提交前就拒** ✗：引用了 ``<Picture 3>`` 却只给 2 张 ⇒ 当场报 ✓
  （编号最常见的错就是**续接尾帧占了 1 号** ⇒ 换场景后全体**前移** ✓✗）；
* ⭐ **只写 ``<Audio 1>`` 不算声明** ✗✗：必须**照样追加**三行结构化声明 ✓
  —— 照标签跳过 ⇒ 模型自由发挥、成片音轨与配音相关性≈0 ✓✗；
* ⭐ **幂等** ✓：对产出再跑一次 ⇒ 一字不变 ✓（否则每轮重试都会堆一份声明 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/h3_prompt_wiring_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.local_services.h3 import prompt as prompt_mod  # noqa: E402
from app.services.engine import conditioning as cond  # noqa: E402

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


REF = cond.AudioReference(1, "fully_copy", 1)


def case_markup_and_validate() -> None:
    """① 标签转换 + **提交前核编号** ✓（越界 ⇒ 拒 ✗）。"""
    got = prompt_mod.build_prompt("看着 @图1 说话", pictures=1)
    check("① ``@图1`` 转换成 ``<Picture 1>`` ✓ 且最终 prompt 里就是标签 ✓",
          "<Picture 1>" in got["prompt"] and got["markup"] == {"Picture": [1]}, got["markup"])
    bad = _raises(lambda: prompt_mod.build_prompt("看着 <Picture 3> 说话", pictures=2), "续接尾帧")
    check("①′ ⭐ 编号越界 ⇒ **提交前**就报错 ✓ 且提示「续接尾帧排最前 ⇒ 编号会前移」✗",
          bad is not None and "最前" in bad, bad)
    check("①″ 给足编号 ⇒ 通过 ✓（正向对照：上面那条不是「凡引用必报」的套套逻辑 ✓）",
          prompt_mod.build_prompt("看着 <Picture 2> 说话", pictures=2)["prompt"].count("<Picture 2>")
          == 1)


def case_declarations() -> None:
    """② 声明追加 ✓（**标签不算声明** ✗✗ —— 这条是全模块最贵的判据 ✓）。"""
    plain = prompt_mod.build_prompt("黄昏屋顶，少年转身")
    check("② 没有参考素材 ⇒ **不加任何声明** ✓（不往提示词里塞噪声 ✗）",
          "subject_definitions" not in plain["prompt"] and plain["plan"]["audioNeeded"] is False)
    tagged = prompt_mod.build_prompt("<Audio 1> 是 <Picture 1> 的音色参考", pictures=1, audios=1,
                                     audio_refs=[REF])
    check("②′ ⭐⭐ 只写了 ``<Audio 1>`` 绑定句 ⇒ **照样追加**三行声明 ✓✗"
          "（照标签跳过 ⇒ 模型忽略参考音频 ✓✗）",
          all(key in tagged["prompt"] for key in cond.DECLARATION_KEYS)
          and "<Audio 1>: fully_copy" in tagged["prompt"], tagged["prompt"][:60])
    mine = prompt_mod.build_prompt("retention_analysis: <Audio 1>: fully_copy", audios=1,
                                   audio_refs=[REF])
    check("②″ 用户**自己写了** ``retention_analysis`` ⇒ 不重复追加 ✓ 且计划里**给了理由** ✓",
          mine["prompt"].count("retention_analysis") == 1
          and mine["plan"]["audioNeeded"] is False
          and "保留声明" in (mine["plan"]["audioSkipped"] or ""), mine["plan"])
    video = prompt_mod.build_prompt("跟着 @视频1 的动作", videos=1, video_count=1)
    check("②‴ 参考视频 ⇒ 追加 ``reference`` 关系声明 ✓（1 路 ✓）；4 路 ⇒ 报错（点名上限 3 ✓）",
          "retention_analysis" in video["prompt"]
          and _raises(lambda: prompt_mod.build_prompt("x", video_count=4), "3 路") is not None,
          video["plan"])


def case_global_and_idempotent() -> None:
    """③ 全局提示词在前 ✓ + ⭐ **幂等** ✓。"""
    got = prompt_mod.build_prompt("少年转身", global_prompt="Pixar 风，暖光，24fps")
    check("③ 全局提示词拼在**最前** ✓（单段提示词在后 ✓）、并标出拼过 ✓",
          got["prompt"].startswith("Pixar 风") and got["prefixed"] is True
          and "少年转身" in got["prompt"] and got["plan"]["audioNeeded"] is False, got["prompt"])
    check("③′ 只有全局 / 只有正文 ⇒ 都不崩 ✓（空的那半不留多余空行 ✓）",
          prompt_mod.build_prompt("", global_prompt="风格句")["prompt"] == "风格句"
          and prompt_mod.build_prompt("正文")["prefixed"] is False)
    one = prompt_mod.build_prompt("黄昏屋顶", pictures=1, audio_refs=[REF])
    again = prompt_mod.build_prompt(one["prompt"], pictures=1, audio_refs=[REF])
    check("③″ ⭐⭐ **幂等**：对产出再跑一次 ⇒ prompt **一字不变** ✓（否则每轮重试都会堆声明 ✗）"
          "且第二次的计划里 ``audioNeeded=False`` ✓",
          again["prompt"] == one["prompt"] and again["plan"]["audioNeeded"] is False,
          (len(one["prompt"]), len(again["prompt"])))


def main() -> int:
    case_markup_and_validate()
    case_declarations()
    case_global_and_idempotent()
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
