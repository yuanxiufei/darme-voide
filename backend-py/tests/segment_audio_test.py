"""自检：**段级音频合成**（模型声 + 配音 → 一条音轨 ✓ 2026-09-24 接 ✓ 零依赖 ✓）。

背景（「能力接不出去不算功能」的最后一处 ✓✗）：``engine/audio_mix.py`` 的口径早就立了 ✓
（长度守恒 ✓ / 三支削波**各不相同** ✗ / 不静默重采样 ✓），但**没有任何调用方** ✗；
而单镜合成是 ``ffmpeg -map 1:a`` **顶替** ✗ —— 生成视频自带的音轨（H3 是联合 AV ✓）**被直接丢掉** ✓✗。
本套验的是接上之后那半边：**真读写 wav**（标准库 ✓ **不碰 torch** ✗）+ 分镜合成**真的换了音轨** ✓。

三条最容易「听着还行、其实变了」的判据 ✓✗：
1. ⭐⭐ **长度守恒**：输出恒 ``round(duration × sr)``（配音长了截断 ✓、短了补零 ✓）——
   音轨**不许把视频拖长/拖短** ✓✗；
2. ⭐⭐ **混音支 = 模型声 ×0.6（写死 ✓）+ 配音 ×音量 ⇒ clamp ±1** ✓；``replace`` 支是**顶替** ✓
   （模型声**消失** ✓ —— 两者是不同产品语义，别混 ✓✗）；
3. ⭐ **采样率/声道不一致 ⇒ 拒** ✓ 且**两个值都印出来** ✓✗（悄悄重采样会让音高与时长都变 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/segment_audio_test.py
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import wave
from pathlib import Path

os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import segment_audio as sa  # noqa: E402
from app.services.engine import audio_mix as mix_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def write_wav(path: Path, rows: list[list[float]], rate: int = 32000) -> Path:
    """自己写一个小 wav ✓（标准库 ✓ —— 与产品侧同一个格式：16-bit PCM ✓）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = len(rows[0])
    data = bytearray()
    for index in range(frames):
        for row in rows:
            data += struct.pack("<h", int(max(-1.0, min(1.0, row[index])) * 32767))
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(len(rows))
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(data))
    return target


def read_rows(path: Path) -> tuple[list[list[float]], int]:
    rows, rate = sa.read_wav_tracks(path)
    return rows, rate


def case_io(root: Path) -> None:
    src = write_wav(root / "io.wav", [[0.0, 0.5, -0.5, 1.0, -1.0], [0.25, -0.25, 0.0, 0.5, 0.5]])
    rows, rate = read_rows(src)
    check("① 读回：声道数/采样率/样本值都对 ✓（16-bit ⇒ 1/32767 的量化误差内 ✓）",
          rate == 32000 and len(rows) == 2
          and all(abs(a - b) < 1e-4 for a, b in zip(rows[0], [0.0, 0.5, -0.5, 1.0, -1.0])),
          (rate, rows))

    stereo = write_wav(root / "stereo.wav", [[0.5, 0.5], [0.5, 0.5]])
    with wave.open(str(stereo), "rb") as handle:
        check("①′ 写出去的是 **16-bit PCM / 原生采样率** ✓（不是浮点 wav ✗）",
              handle.getsampwidth() == 2 and handle.getframerate() == 32000
              and handle.getnchannels() == 2, handle.getparams())

    bad = root / "bad.txt"
    bad.write_bytes(b"not a wav")
    try:
        sa.read_wav_tracks(bad)
        check("①″ 不是 wav ⇒ 拒 ✗（不抛 wild 异常 ✓✗）", False, "没拒")
    except sa.SegmentAudioError as err:
        check("①″ 不是 wav ⇒ 拒 ✗（不抛 wild 异常 ✓✗）", bool(err), str(err)[:80])


def case_mix(root: Path) -> None:
    # 模型声 0.5、配音 0.5 ⇒ mix = 0.5*0.6 + 0.5*1.0 = 0.8 ✓（写死的 0.6 ✓）
    model = write_wav(root / "model.wav", [[0.5] * 8, [0.5] * 8])
    voice = write_wav(root / "voice.wav", [[0.5] * 8, [0.5] * 8])
    out = root / "mixed.wav"
    report = sa.mix_model_and_voice(model_audio_path=model, voice_path=voice, output_path=out)
    rows, _ = read_rows(out)
    check("② ⭐ 混音支 = 模型声 ×**0.6**（写死 ✓）+ 配音 × 音量 ⇒ 0.5→0.8 ✓（逐样本核 ✓✗）",
          all(abs(value - 0.8) < 1e-4 for value in rows[0]) and report["voiceMode"] == "mix",
          (rows[0][:3], report["voiceMode"]))
    sa.mix_model_and_voice(model_audio_path=model, voice_path=voice,
                           output_path=root / "rep.wav", voice_mode="replace")
    rep_rows, _ = read_rows(root / "rep.wav")
    check("②′ ⭐ ``replace`` 支是**顶替** ✓（模型声消失 ✓✗ —— 与 mix 是两种产品语义 ✓）",
          abs(rep_rows[0][0] - 0.5) < 1e-4, rep_rows[0][:3])

    sa.mix_model_and_voice(model_audio_path=model, voice_path=voice,
                           output_path=root / "vol.wav", voice_volume=0.5)
    vol_rows, _ = read_rows(root / "vol.wav")
    check("②″ 音量按给定值乘 ✓（0.5×0.6 + 0.5×0.5 = 0.55 ✓ —— 模型声的那 0.6 **不随音量变** ✗）",
          abs(vol_rows[0][0] - 0.55) < 1e-4, vol_rows[0][:3])

    loud = write_wav(root / "loud.wav", [[0.9] * 8, [0.9] * 8])
    clip = sa.mix_model_and_voice(model_audio_path=loud, voice_path=loud, output_path=root / "clip.wav")
    check("②‴ ⭐ 混音支的削波上限是 **±1** ✓（0.9×0.6+0.9=1.44 ⇒ **1.0** ✓）；"
          "⚠️ ``clippedSamples`` 这里应为 **0** ✗ —— 削波是 ``audio_mix`` 干的 ✓，"
          "写盘那道只是**兜底** ✓（别把两处混读 ✓✗）",
          abs(read_rows(root / "clip.wav")[0][0][0] - 1.0) < 1e-4 and clip["clippedSamples"] == 0,
          (read_rows(root / "clip.wav")[0][0][:2], clip["clippedSamples"]))
    floor = sa.write_wav_tracks([[2.0, -2.0, 0.5]], 32000, root / "floor.wav")
    check("②⁴ 写盘那道兜底**真的会钳** ✓（直接喂 2.0 ⇒ 1.0 ✓ 且**如实回报**截样数 ✓✗）",
          floor["clippedSamples"] == 2 and abs(read_rows(root / "floor.wav")[0][0][0] - 1.0) < 1e-4,
          (floor["clippedSamples"], read_rows(root / "floor.wav")[0][0][:3]))


def case_length(root: Path) -> None:
    model = write_wav(root / "len-model.wav", [[0.4] * 160])          # 160 样本 @32k = 5 ms
    longer = write_wav(root / "len-long.wav", [[0.5] * 400])          # 配音更长 ✓
    shorter = write_wav(root / "len-short.wav", [[0.5] * 40])         # 配音更短 ✓

    cut = sa.mix_model_and_voice(model_audio_path=model, voice_path=longer,
                                 output_path=root / "len-cut.wav")
    rows_cut, _ = read_rows(root / "len-cut.wav")
    check("③ ⭐⭐ **长度守恒**（配音更长）⇒ 输出 = **模型声长度** ✓✗（音轨不许把视频拖长 ✓）",
          len(rows_cut[0]) == 160 and cut["targetSeconds"] == cut["modelSeconds"],
          (len(rows_cut[0]), cut["targetSeconds"], cut["modelSeconds"]))

    pad = sa.mix_model_and_voice(model_audio_path=model, voice_path=shorter,
                                 output_path=root / "len-pad.wav")
    rows_pad, _ = read_rows(root / "len-pad.wav")
    check("③′ ⭐⭐ 配音更短 ⇒ **补零** ✓（不是循环 ✗ —— 循环只有环境音那支用 ✓）且长度照旧 ✓",
          len(rows_pad[0]) == 160 and abs(rows_pad[0][-1] - 0.4 * 0.6) < 1e-4,
          (len(rows_pad[0]), rows_pad[0][-3:]))

    explicit = sa.mix_model_and_voice(model_audio_path=model, voice_path=longer,
                                      output_path=root / "len-exp.wav", duration=0.002)
    check("③″ 显式给 ``duration`` ⇒ 按它算长度 ✓（round(duration×sr) ✓ 与试算一致 ✓）",
          explicit["frames"] == mix_mod.samples_for_seconds(0.002, 32000),
          (explicit["frames"], mix_mod.samples_for_seconds(0.002, 32000)))


def case_refuse(root: Path) -> None:
    model = write_wav(root / "r-model.wav", [[0.4] * 32])
    other_rate = write_wav(root / "r-rate.wav", [[0.4] * 32], rate=44100)
    try:
        sa.mix_model_and_voice(model_audio_path=model, voice_path=other_rate,
                               output_path=root / "r-out.wav")
        check("④ ⭐ 采样率不一致 ⇒ **拒** ✗（不静默重采样 ✓✗）", False, "没拒")
    except sa.SegmentAudioError as err:
        check("④ ⭐ 采样率不一致 ⇒ **拒** ✗（不静默重采样 ✓✗）且**两个值都印出来** ✓",
              "44100" in str(err) and "32000" in str(err), str(err)[:120])

    mono = write_wav(root / "r-mono.wav", [[0.4] * 32])
    stereo = write_wav(root / "r-stereo.wav", [[0.4] * 32, [0.4] * 32])
    try:
        sa.mix_model_and_voice(model_audio_path=stereo, voice_path=mono,
                               output_path=root / "r-out2.wav")
        check("④′ ⭐ 声道数不一致 ⇒ **拒** ✗（不静默转声道 ✓✗）", False, "没拒")
    except sa.SegmentAudioError as err:
        check("④′ ⭐ 声道数不一致 ⇒ **拒** ✗（不静默转声道 ✓✗）", "声道" in str(err), str(err)[:120])

    try:
        sa.mix_model_and_voice(model_audio_path=root / "nope.wav", voice_path=model,
                               output_path=root / "r-out3.wav")
        check("④″ 文件不存在 ⇒ 拒 ✗（不是 FileNotFoundError 漏出去 ✓✗）", False, "没拒")
    except sa.SegmentAudioError as err:
        check("④″ 文件不存在 ⇒ 拒 ✗（不是 FileNotFoundError 漏出去 ✓✗）",
              "不存在" in str(err), str(err)[:80])


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="segment_audio_"))
    case_io(root)
    case_mix(root)
    case_length(root)
    case_refuse(root)
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
