"""S7 自检：**段级音频合成**（纯样本运算 ✓ 零依赖 ✓ 2026-09-24）。

钉的四条都是**会静默出错**的口径 ✗✗：

* ⭐ **长度守恒** ✓：输出长度恒等于 ``round(duration × sr)`` ✓（配音偏长不能把视频拖长 ✓✗）；
* ⭐ ``enabled=False`` ⇒ **没有音轨**（``None`` ✓）而不是**全零音轨** ✗；
* ⭐ 三支的**削波口径不一样** ✗：``mix`` 是 ``模型声×0.6 + 配音×音量`` 且 clamp ±1 ✓、
  环境音那支 clamp **±0.95** ✓✗（留 0.05 防爆音 ✓）、**纯模型声那支不削波** ✗；
* ⭐ **不静默重采样 / 不静默转声道** ✗：不一致 ⇒ **报错** ✓（静默处理会让音高、时长悄悄变 ✓✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_audio_mix_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import audio_mix as am  # noqa: E402

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


SR = 8          # 小采样率 ⇒ 样本数好数 ✓（1 秒 = 8 个样本 ✓）
STEREO = [[0.2] * 8, [0.2] * 8]


def case_length() -> None:
    """① 长度与「没有音轨」✓。"""
    check("① ``samples_for_seconds``：``round(秒 × 采样率)`` ✓、且**至少 1** ✓（0 长度音轨不许有 ✗）",
          am.samples_for_seconds(2, SR) == 16 and am.samples_for_seconds(0, SR) == 1
          and _raises(lambda: am.samples_for_seconds(1, 0), "为正") is not None)
    got = am.mix_segment_audio(duration=2.0, sample_rate=SR, model_audio=[[0.5] * 16, [0.5] * 16])
    check("①′ ⭐ 长度守恒 ✓：输出样本数 == ``round(时长 × 采样率)`` ✓（长截短补 ✓✗）",
          got is not None and got.frames == 16 and got.channels == 2, got and got.to_dict())
    short = am.mix_segment_audio(duration=2.0, sample_rate=SR, model_audio=[[0.5] * 3, [0.5] * 3])
    long_ = am.mix_segment_audio(duration=1.0, sample_rate=SR, model_audio=[[0.5] * 99, [0.5] * 99])
    check("①″ 短了**补零** ✓（尾部是 0 ✓、长度仍然对 ✓）、长了**截断** ✓（不拖长 ✓✗）",
          short is not None and short.frames == 16 and short.samples[0][-1] == 0.0
          and long_ is not None and long_.frames == 8, (short and short.frames, long_ and long_.frames))
    check("①‴ ⭐ ``enabled=False`` ⇒ ``None`` ✓（**没有音轨** ✗ —— 不是全零音轨 ✓✗）",
          am.mix_segment_audio(duration=2.0, sample_rate=SR, model_audio=STEREO,
                               enabled=False) is None)


def case_fit_trim_offset() -> None:
    """② 三个零件 ✓：对齐 / 裁剪 / 偏移。"""
    looped = am.fit([[1.0, 2.0]], 5, loop=True)
    check("② ``fit(loop=True)`` ⇒ **循环填充** ✓（环境音比段短时用 ✓ —— 补零会听出「断一下」✗）",
          looped == [[1.0, 2.0, 1.0, 2.0, 1.0]], looped)
    check("②′ ``trim`` ``keep``：保留 ``[start, end)`` ✓；``end <= start`` ⇒ **取到结尾** ✓（不是空区间 ✗）",
          am.trim([[0, 1, 2, 3, 4, 5, 6, 7]], sample_rate=SR, start=0.25, end=0.75,
                  mode="keep") == [[2, 3, 4, 5]]
          and am.trim([[0, 1, 2, 3, 4, 5, 6, 7]], sample_rate=SR, start=0.25, mode="keep")
          == [[2, 3, 4, 5, 6, 7]])
    check("②″ ``trim`` ``cut``：**把中间挖掉再接起来** ✓（上游 UI 的「删除选中区」✓）",
          am.trim([[0, 1, 2, 3, 4, 5, 6, 7]], sample_rate=SR, start=0.25, end=0.5,
                  mode="cut") == [[0, 1, 4, 5, 6, 7]])
    check("②‴ 裁剪模式不认识 ⇒ 报错并列出合法的两个 ✓",
          _raises(lambda: am.trim([[1, 2]], sample_rate=SR, mode="中间"), "keep") is not None)
    check("②⁴ ``offset_by`` ⇒ **前置静音** ✓（= 从段内第 X 秒开始播 ✓ —— 不是往后接 ✗）",
          am.offset_by([[1.0, 2.0]], seconds=0.25, sample_rate=SR) == [[0.0, 0.0, 1.0, 2.0]],
          am.offset_by([[1.0, 2.0]], seconds=0.25, sample_rate=SR))


def case_modes() -> None:
    """③ 三支模式 ✓（⚠️ **削波口径不一样** ✗✗ —— 这条最容易「统一」掉 ✓✗）。"""
    model = [[0.2] * 8, [0.2] * 8]
    voice = [[0.4] * 8, [0.4] * 8]
    replaced = am.mix_segment_audio(duration=1.0, sample_rate=SR, model_audio=model,
                                    voice_audio=voice, voice_rate=SR, voice_mode="replace")
    check("③ ``replace`` ⇒ 输出**就是配音**（模型声完全不在 ✓✗ —— 「顶替」✓）",
          replaced is not None and replaced.samples[0][0] == 0.4, replaced and replaced.samples[0][:2])
    mixed = am.mix_segment_audio(duration=1.0, sample_rate=SR, model_audio=model, voice_audio=voice,
                                 voice_rate=SR, voice_mode="mix", voice_volume=0.5)
    expected = 0.2 * am.MIX_MODEL_GAIN + 0.4 * 0.5
    check(f"③′ ⭐ ``mix`` ⇒ ``模型声×{am.MIX_MODEL_GAIN}（写死 ✓）+ 配音×音量`` ✓（精确到系数 ✓）",
          mixed is not None and abs(mixed.samples[0][0] - expected) < 1e-9,
          (mixed and mixed.samples[0][0], expected))
    # ⚠️ 环境音也要**同声道数** ✗（本模块**不静默转声道** ✓ —— 本套第一版喂了单声道环境音
    #    配立体声模型，被自己的判据拦下 ✓✗：拦得对 ✓，是**夹具**错了 ✓）。
    amb = am.mix_segment_audio(duration=1.0, sample_rate=SR, model_audio=[[0.9] * 8, [0.9] * 8],
                               ambient_audio=[[1.0] * 4, [1.0] * 4], ambient_rate=SR)
    check("③″ ⭐⭐ 环境音那支：``模型声 + 环境音×0.25`` ⇒ 0.9+0.25=1.15 ⇒ **clamp 到 0.95** ✓✗"
          "（**不是 1.0** ✓ —— 留 0.05 防爆音 ✓）",
          amb is not None and abs(amb.samples[0][0] - am.AMBIENT_CEILING) < 1e-9,
          amb and amb.samples[0][0])
    ambient_mismatch = _raises(lambda: am.mix_segment_audio(
        duration=1.0, sample_rate=SR, model_audio=[[0.9] * 8, [0.9] * 8],
        ambient_audio=[[1.0] * 8], ambient_rate=SR), "不静默转声道")
    check("③‴′ ⭐ 单声道环境音配立体声模型 ⇒ **报错** ✓（不静默转声道 ✗ —— 先在上游统一 ✓）",
          ambient_mismatch is not None, ambient_mismatch)
    plain = am.mix_segment_audio(duration=1.0, sample_rate=SR, model_audio=[[1.5] * 8, [1.5] * 8])
    check("③‴ ⭐ **纯模型声那支不削波** ✗（1.5 还是 1.5 ✓ —— 别把这支也 clamp 掉 ✗✗）",
          plain is not None and abs(plain.samples[0][0] - 1.5) < 1e-9, plain and plain.samples[0][0])
    check("③⁴ 环境音比段短 ⇒ **循环**填满 ✓（`fit(loop=True)` 生效 ✓）",
          amb is not None and len(amb.samples[0]) == 8 and len(amb.samples) == 2, amb and amb.to_dict())


def case_refusals() -> None:
    """④ 不猜 ✓：采样率 / 声道数 / 模式 / 空输入。"""
    same = _raises(lambda: am.mix_segment_audio(
        duration=1.0, sample_rate=SR, voice_audio=[[0.1] * 8, [0.1] * 8], voice_rate=SR * 2),
        "不静默重采样")
    check("④ ⭐ 采样率不一致 ⇒ **报错** ✓ 且点名「**不静默重采样**」✗（静默处理会让音高/时长悄悄变 ✓✗）"
          "并给出正确做法（**加载时**就按段采样率解码 ✓）",
          same is not None and "加载时" in same, same)
    check("④′ 声道数不一致 ⇒ **报错** ✓（不静默转声道 ✗）",
          _raises(lambda: am.mix_segment_audio(duration=1.0, sample_rate=SR,
                                               model_audio=[[0.1] * 8, [0.1] * 8],
                                               voice_audio=[[0.1] * 8], voice_rate=SR), "声道数")
          is not None)
    check("④″ 给了配音却**不给采样率** ⇒ 报错 ✓（本仓不猜 ✗ —— 猜错会让音高/时长悄悄变 ✓✗）",
          _raises(lambda: am.mix_segment_audio(duration=1.0, sample_rate=SR,
                                               voice_audio=[[0.1] * 8]), "采样率") is not None)
    check("④‴ 配音模式不认识 ⇒ 报错且说明**与对口型无关** ✓（换音轨不会让嘴动对上 ✓✗）",
          _raises(lambda: am.mix_segment_audio(duration=1.0, sample_rate=SR, model_audio=STEREO,
                                               voice_audio=[[0.1] * 8], voice_rate=SR,
                                               voice_mode="dub"), "对口型") is not None)
    check("④⁴ 什么都没给但 ``enabled=True`` ⇒ 报错 ✓（没东西可拼 ✓ —— 要静音请显式关掉 ✓）",
          _raises(lambda: am.mix_segment_audio(duration=1.0, sample_rate=SR), "enabled") is not None)
    # ⚠️ 空列表是**假值** ✓ ⇒ 走的是「没东西可拼 ⇒ 要静音请显式 enabled=False」那条 ✓✗
    #    （本套第一版去匹配「空的」✗ —— needle 写错，模块行为本来就对 ✓）。
    check("④⁵ 空轨 ⇒ 报错 ✓（走的是「**没东西可拼**」那条 ✓）；"
          "长度不一致的声道 ⇒ 报错点名「长度不一致」✓",
          _raises(lambda: am.mix_segment_audio(duration=1.0, sample_rate=SR, model_audio=[]),
                  "enabled") is not None
          and _raises(lambda: am.fit([[1.0], [1.0, 2.0]], 4), "长度不一致") is not None)
    check("④⁶ ``to_dict`` 只报**形状与时长** ✓（⚠️ 不塞样本 ✗ —— 日志里几十万个浮点数没价值 ✓✗）",
          am.SegmentAudio(((0.1, 0.2), (0.1, 0.2)), SR).to_dict()
          == {"channels": 2, "frames": 2, "sampleRate": SR, "seconds": 0.25},
          am.SegmentAudio(((0.1, 0.2), (0.1, 0.2)), SR).to_dict())


def main() -> int:
    case_length()
    case_fit_trim_offset()
    case_modes()
    case_refusals()
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
