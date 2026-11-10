"""S7 自检：引擎产物的**交付形态与可追溯性**（2026-11-07 补 ✓）。

这套盯的是三个"不报错但产物不可信"的缺口 ✓：

① **双流只交静音片 ＋ 单独 wav** ✓✗ —— 播放器里没声音、成片还得再合一次音频；
② **落盘帧数没人核** ✓✗ —— 盘上三个 ``video_seed777.mp4`` 分别 10/34/39 帧，
   同名互相覆盖，谁也说不清哪个是这次的；
③ **PNG 零元数据 ＋ 无一页清单** ✓✗ —— 产物离开作业单就说不清"哪个模型/多少步/什么提示词出的"。

⚠️ 判据一律**把文件读回来核** ✓（``ffprobe`` / 标准库 ``wave`` / ``PIL.Image.text`` / ``hashlib`` ✓），
**不看函数返回值自证** ✗ —— 返回值说"写了元数据"而 muxer 静默丢掉，正是本轮要消灭的那类失败 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_artifact_test.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import media as media_mod  # noqa: E402
from app.services.engine import pipeline as pipe  # noqa: E402
from app.services.engine import runtime as runtime_mod  # noqa: E402
from app.services.engine import torch_backend as torch_backend_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []

SAMPLE_RATE = 32000
FPS = 24


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _torch():
    import torch  # noqa: PLC0415

    return torch


def _frames(count: int, size: int = 64):
    torch = _torch()
    gen = torch.Generator().manual_seed(7)
    return torch.rand(1, 3, count, size, size, generator=gen) * 2 - 1


def _tone(seconds: float):
    """``(2, N)`` 的 440 Hz 正弦 ✓ —— 只用标准库/pytorch ✓（不引第三方音频库 ✓）。"""
    torch = _torch()
    n = int(SAMPLE_RATE * seconds)
    wave = torch.sin(torch.arange(n, dtype=torch.float32) / SAMPLE_RATE * 2 * 3.14159265 * 440)
    return (wave * 0.2).repeat(2, 1)


def _ffprobe(path: Path) -> dict:
    """**独立** ffprobe 复核 ✓（绕开 :func:`media.probe` ✓ —— 用它取数就是自己证自己 ✗）。"""
    done = subprocess.run(
        [media_mod._tool("ffprobe"), "-v", "error", "-show_streams", "-show_format",  # noqa: SLF001
         "-of", "json", str(path)],
        capture_output=True, timeout=120)
    if done.returncode:
        raise RuntimeError(done.stderr.decode("utf-8", "replace")[-400:])
    return json.loads(done.stdout.decode("utf-8", "replace"))


def _have_ffmpeg() -> bool:
    try:
        media_mod._tool("ffmpeg")  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return False
    return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════
# ① 双流：音轨真进 mp4（且**画面一帧不少** ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_mux(root: Path) -> None:
    frames = _frames(12)                       # 12 帧 @24 ⇒ 画面 0.5 秒 ✓
    video = root / "mux.mp4"
    cue = root / "cue_short.wav"
    media_mod.write_wav(_tone(0.3), cue, sample_rate=SAMPLE_RATE)   # 音轨比画面**短** ✓

    facts = media_mod.write_video(
        frames, video, fps=FPS, audio=cue,
        metadata={"engineTask": "eng-artifact-test", "stage": "h3"})

    check("① 混流后产物里有音轨 ✓（`hasAudio` 由 ffprobe 给出 ✓）",
          facts.get("hasAudio") is True, facts.get("hasAudio"))
    check("② 音轨真被**编码**成 aac ✓（不是把 wav 原样塞进去 ✓）",
          str(facts.get("audioCodec") or "") == "aac", facts.get("audioCodec"))
    check("③ 帧数没被音轨影响 ✓（12 帧 ✓）",
          int(facts.get("nbFrames") or 0) == 12, facts.get("nbFrames"))
    check("④ ⭐ 音轨比画面**短**时画面也没被剪 ✓（成片 0.5 秒 ✓ —— `apad` 补静音 ＋ `-t` 截断 ✓）",
          abs(float(facts.get("durationSeconds") or 0) - 0.5) < 0.06, facts.get("durationSeconds"))
    check("⑤ 音轨时长如实报 ✓（补静音后 = 画面长度 ✓，不是原始 0.3 秒 ✓）",
          abs(float(facts.get("audioDurationSeconds") or 0) - 0.5) < 0.12,
          facts.get("audioDurationSeconds"))
    #: ⭐⭐ 「补了静音」和「截掉了内容」必须分开说 ✗✗（2026-11-08 补 ✗）：⑤ 那个数（``audioDurationSeconds``）
    #:    读的是**混完之后**的音轨 ✓ ⇒ 它在两种情况下**都**等于 0.5 ✓（本来就同形 ✓✗）。
    #:    现在按源 wav（0.3 秒 ✓）与画面（0.5 秒 ✓）两侧对账：短 ⇒ ``audioPadSeconds`` ✓、零截断 ✓。
    check("⑤′ ⭐⭐ 音轨比画面**短** ⇒ 报「补了静音 0.2 秒」✓ 且**零截断** ✓"
          "（``audioPadSeconds`` 有值 ＋ ``audioTruncated`` 不出现 —— 不许把「补」与「截」说成同一件事 ✗）",
          abs(float(facts.get("audioPadSeconds") or 0) - 0.2) < 0.02
          and abs(float(facts.get("audioTrimSeconds") or 0)) < 1e-9
          and not facts.get("audioTruncated")
          and abs(float(facts.get("audioSourceSeconds") or 0) - 0.3) < 0.02,
          {k: facts.get(k) for k in ("audioSourceSeconds", "audioPadSeconds",
                                     "audioTrimSeconds", "audioTruncated")})

    probed = _ffprobe(video)
    streams = {str(item.get("codec_type")): item for item in probed.get("streams") or []}
    check("⑥ 独立 ffprobe 复核：mp4 里**同时**有 video 与 audio 两路 ✓",
          {"video", "audio"} <= set(streams), sorted(streams))
    check("⑦ 独立复核：视频轨仍是 12 帧 / 24fps ✓（`-t` 没吃掉最后一帧 ✓）",
          str(streams.get("video", {}).get("nb_frames")) == "12"
          and str(streams.get("video", {}).get("r_frame_rate")) == "24/1",
          {k: streams.get("video", {}).get(k) for k in ("nb_frames", "r_frame_rate")})
    tags = {str(key).lower(): value for key, value in (probed.get("format", {}).get("tags") or {}).items()}
    check("⑧ 容器元数据**事后读得回** ✓（`use_metadata_tags` 让自定义键落进 mdta ✓）",
          tags.get("enginetask") == "eng-artifact-test" and tags.get("stage") == "h3", tags)
    check("⑨ 元数据对账：送了几个键就落地几个 ✓（`metadataDropped` 为空 ✓）",
          list(facts.get("metadataDropped") or []) == [], facts.get("metadataDropped"))


def case_mux_long_audio(root: Path) -> None:
    """音轨比画面**长** ⇒ 必须以画面收尾 ✓（不然成片尾部多出一段"定格" ✓✗）。"""
    frames = _frames(12)
    video = root / "mux_long.mp4"
    cue = root / "cue_long.wav"
    media_mod.write_wav(_tone(1.0), cue, sample_rate=SAMPLE_RATE)
    facts = media_mod.write_video(frames, video, fps=FPS, audio=cue)
    check("⑩ 音轨比画面**长**时仍以画面收尾 ✓（成片 0.5 秒 ✓）",
          abs(float(facts.get("durationSeconds") or 0) - 0.5) < 0.06, facts.get("durationSeconds"))
    probed = _ffprobe(video)
    check("⑪ 独立复核：容器时长 = 画面时长 ✓（不是音轨那 1.0 秒 ✓）",
          abs(float(probed.get("format", {}).get("duration") or 0) - 0.5) < 0.06,
          probed.get("format", {}).get("duration"))
    #: ⭐⭐ 上面两条**正是这个缺陷的背书** ✗✗（2026-11-08 补 ✗）：⑩⑪ 只断言"成片 0.5 秒 ✓" ✓，
    #:    而 1.0 秒的音轨有 **0.5 秒被 ``-t`` 截掉** ✓✗ —— 那是内容丢失 ✓，不是格式问题 ✓，
    #:    此前回执里零痕迹 ✓（``audioMuxed: True`` ＋ ``audioDurationSeconds: 0.5`` 与"音频本来 0.5 秒"
    #:    完全同形 ✓；``probe`` 读的又是混完之后那条 ✓ ⇒ 永远自洽 ✓✗）。
    check("⑩′ ⭐⭐ 音轨比画面**长** ⇒ 必须报出**被截掉多少秒** ✗✗"
          "（1.0 秒音轨 ＋ 0.5 秒画面 ⇒ ``audioTrimSeconds`` 0.5 ✓、``audioTruncated`` 为真 ✓、"
          "并指明 wav 是全长 ✓）—— 只报「容器 0.5 秒」等于把内容丢失藏进格式里 ✓",
          abs(float(facts.get("audioTrimSeconds") or 0) - 0.5) < 0.02
          and facts.get("audioTruncated") is True
          and "截" in str(facts.get("audioTruncatedNote") or "")
          and abs(float(facts.get("audioSourceSeconds") or 0) - 1.0) < 0.02,
          {k: facts.get(k) for k in ("audioSourceSeconds", "videoSeconds",
                                     "audioTrimSeconds", "audioTruncated")})
    check("⑩″ 独立复核：mp4 里那条音轨**确实只有**画面长度 ✓（0.5 秒 ✓ —— 0.5 秒真没了 ✗）",
          abs(float(facts.get("audioDurationSeconds") or 0) - 0.5) < 0.12,
          facts.get("audioDurationSeconds"))


def case_mux_guards(root: Path) -> None:
    frames = _frames(6)
    missing = root / "not_here.wav"
    target = root / "guard.mp4"
    try:
        media_mod.write_video(frames, target, fps=FPS, audio=missing)
        raised = False
    except media_mod.MediaError:
        raised = True
    check("⑫ 音轨文件不存在 ⇒ 当场报错 ✓（不静默出一条无声片 ✓）", raised)
    check("⑬ 报错路径**一个文件都没写** ✓", not target.exists())

    plain = root / "plain.mp4"
    facts = media_mod.write_video(frames, plain, fps=FPS)
    check("⑭ 不给 ``audio`` ⇒ 仍是**纯视频轨** ✓（默认路径一字未变 ✓）",
          facts.get("hasAudio") is False, facts.get("hasAudio"))
    check("⑮ 无声片如实报「没有音轨」 ✓（`audioDurationSeconds` 必须是 None ✓，"
          "不许回落成容器时长 ✓✗）", facts.get("audioDurationSeconds") is None,
          facts.get("audioDurationSeconds"))


# ══════════════════════════════════════════════════════════════════════════
# ② 图：元数据写进 PNG ＋ 空白图判据
# ══════════════════════════════════════════════════════════════════════════
def case_image(root: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    torch = _torch()
    good = root / "image_ok.png"
    picture = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(1)) * 2 - 1
    report = media_mod.write_image(
        picture, good, metadata={"seed": 42, "stage": "zimage", "prompt": "雨夜霓虹下的街口"})
    check("⑯ 正常画面不标 ``degraded`` ✓（标准差 %s ✓）" % report.get("std"),
          report.get("degraded") is False, report)
    check("⑰ 元数据键如实回执 ✓", "prompt" in (report.get("metadataKeys") or []),
          report.get("metadataKeys"))
    with Image.open(good) as opened:
        texts = dict(opened.text or {})
    check("⑱ ⭐ PNG 里的元数据**真能读回来** ✓（`PngInfo` ⇒ iTXt 块 ✓）",
          texts.get("seed") == "42" and texts.get("stage") == "zimage"
          and "雨夜霓虹" in str(texts.get("prompt")), texts)
    check("⑲ 存图尺寸仍是 32×32 ✓（写元数据没改像素 ✓）",
          report.get("width") == 32 and report.get("height") == 32,
          (report.get("width"), report.get("height")))

    blank = root / "image_blank.png"
    flat = torch.full((1, 3, 32, 32), -1.0)          # 值域下限 ⇒ 纯黑 ✓
    blank_report = media_mod.write_image(flat, blank, metadata={"seed": 1})
    check("⑳ ⭐ 纯色图被标 ``degraded`` ✓（标准差 0 ✓ —— 这类失败**不会抛异常** ✓✗，只能靠它喊 ✓）",
          blank_report.get("degraded") is True and float(blank_report.get("std") or 0) == 0.0,
          blank_report)
    with Image.open(blank) as opened:
        dark = opened.convert("RGB").getpixel((0, 0))
    check("㉑ 纯色图本身仍是可打开的合法 PNG ✓（我们只报不拦 ✓）", dark == (0, 0, 0), dark)


def case_metadata_helper() -> None:
    from app.services.engine.torch_backend import TorchBackend  # noqa: PLC0415

    request = pipe.GenerationRequest(prompt="长" * 3000, negative="负" * 900, seed=5, stage="h3")
    plan = pipe.build_plan(request)
    meta = TorchBackend._artifact_metadata(request, plan, "image")  # noqa: SLF001
    check("㉒ 落进产物的提示词被**截断** ✓（2000 字 ✓）—— 容器元数据不该装整段长提示词 ✓",
          len(meta["prompt"]) == 2000 and len(meta["negative"]) == 500,
          (len(meta["prompt"]), len(meta["negative"])))
    check("㉓ 参数快照带齐 seed/stage/尺寸 ✓",
          meta["seed"] == 5 and meta["stage"] == "h3" and meta["width"] == plan.width,
          {k: meta[k] for k in ("seed", "stage", "width", "height", "fps", "frames")})


# ══════════════════════════════════════════════════════════════════════════
# ③ 落盘边界：帧数断言 ＋ 产物目录清单
# ══════════════════════════════════════════════════════════════════════════
def case_frame_guard(root: Path) -> None:
    from app.services.engine.torch_backend import TorchBackend  # noqa: PLC0415

    backend = TorchBackend(device="cpu")
    if not backend.describe()["available"]:
        skip("torch 不可用 ⇒ 落盘边界整段跳过 ✓")
        return

    out = root / "frames"
    out.mkdir(parents=True, exist_ok=True)
    request = pipe.GenerationRequest(prompt="x", steps=2, seed=11, outputs_dir=str(out))
    plan = pipe.build_plan(request)
    expected = int(plan.frames)
    wrong = expected + 3

    try:
        backend.write({"frames": _frames(wrong, 32), "synthetic": True}, plan, request)
        raised = False
    except media_mod.MediaError:
        raised = True
    check(f"㉔ ⭐ 解出来的帧数（{wrong}）≠ ``plan.frames``（{expected}）⇒ **当场失败** ✓"
          "（不落一条少帧的片子 ✓）", raised)
    check("㉕ 失败路径**一个文件都没写** ✓（含清单 ✓ —— 拒绝/失败不该留下任何产物 ✓）",
          not any(out.iterdir()), sorted(item.name for item in out.iterdir()))


def case_frames_unverifiable(root: Path) -> None:
    """⭐⭐ ``nbFrames`` 读不到 ≠ 帧数核过了 ✗✗（2026-11-08 补 ✗）。

    ``nb_frames`` 只在封装**写了 index** 时才有（分片 mp4 / 某些编码器 / 重封装中间态全没有），
    ``probe`` 如实给 ``None``。旧闸门写的是 ``encoded_frames is not None`` ⇒ 读不到就**整条跳过**，
    而回执照样写「**真 mp4** ✓（ffprobe 可复核 ✓）」⇒ 「帧数逐帧核过了」与「根本没核」**同形**。
    这里把 ``probe`` 包一层做**读数缺失**的模拟（真 ffprobe 仍在底下跑 ✓，只把某一格涂掉 ✓）。
    """
    from app.services.engine.torch_backend import TorchBackend  # noqa: PLC0415

    backend = TorchBackend(device="cpu")
    if not backend.describe()["available"]:
        skip("torch 不可用 ⇒ 帧数核对那几条跳过 ✓")
        return

    real_probe = media_mod.probe

    def probe_hiding(*drop: str):
        def patched(path):
            facts = dict(real_probe(path))
            for key in drop:
                facts[key] = None
            return facts

        return patched

    def run(seed: int, drop: tuple[str, ...]):
        out = root / f"frames_probe_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        request = pipe.GenerationRequest(prompt="x", steps=2, seed=seed, outputs_dir=str(out))
        plan = pipe.build_plan(request)
        frames = _frames(int(plan.frames), 32)
        media_mod.probe = probe_hiding(*drop)
        try:
            return backend.write({"frames": frames, "synthetic": True}, plan, request), plan, out
        finally:
            media_mod.probe = real_probe

    card, plan, _ = run(21, ("nbFrames",))
    check("㉔′ ⭐⭐ ``nbFrames`` 缺失但读得到时长/帧率 ⇒ 按**时长×帧率估**并**标明是估的** ✓"
          "（``framesVerified == 'duration×fps'`` ✓、``encodedFrames`` 与实际帧数一致 ✓）"
          "—— 不许再出现「闸门静默跳过、回执却像核过」的写法 ✗",
          card.get("framesVerified") == "duration×fps"
          and int(card.get("encodedFrames") or -1) == int(plan.frames),
          {"framesVerified": card.get("framesVerified"), "encodedFrames": card.get("encodedFrames")})
    check("㉔″ ⭐ 估出来的那句必须写进 ``note`` ✓（「估」字 ＋ 说明 ffprobe 没给 ``nbFrames`` ✓）",
          "估" in str(card.get("note") or "") and "nbFrames" in str(card.get("note") or ""),
          card.get("note"))

    unverified, plan2, _ = run(22, ("nbFrames", "durationSeconds"))
    check(f"㉔‴ ⭐⭐ 两格都读不到 ⇒ 如实写 ``framesVerified: None`` ＋ note **点名没核过** ✗"
          f"（但仍然落盘 ✓ —— 没读数既不该拦 ✗，也不许装作核过 ✓✗）"
          f"（plan.frames = {int(plan2.frames)} ✓）",
          unverified.get("framesVerified") is None
          and "没核对过" in str(unverified.get("note") or "")
          and Path(str(unverified.get("videoPath"))).exists(),
          {"framesVerified": unverified.get("framesVerified"), "note": unverified.get("note")})

    bad = root / "frames_probe_23"
    bad.mkdir(parents=True, exist_ok=True)
    request = pipe.GenerationRequest(prompt="x", steps=2, seed=23, outputs_dir=str(bad))
    plan3 = pipe.build_plan(request)
    short_seconds = (int(plan3.frames) - 10) / float(plan3.fps or 24)
    real_probe3 = media_mod.probe

    def probe_short(path):
        facts = dict(real_probe3(path))
        facts["nbFrames"] = None
        facts["durationSeconds"] = short_seconds
        return facts

    media_mod.probe = probe_short
    try:
        backend.write({"frames": _frames(int(plan3.frames), 32), "synthetic": True}, plan3, request)
        raised = False
    except media_mod.MediaError:
        raised = True
    finally:
        media_mod.probe = real_probe3
    check(f"㉔⁗ ⭐⭐ 估出来的帧数**差得多**（不是舍入）⇒ 照样当场失败 ✓"
          f"（时长写着 {round(short_seconds, 3)} 秒 ⇒ ≈ "
          f"{int(round(short_seconds * float(plan3.fps or 24)))} 帧 vs plan {int(plan3.frames)} ✓）"
          "—— 读数缺失不是「放过」的理由 ✓", raised)
    check("㉕′ 判失败后那条**帧数不对的 mp4 不留在渲染目录里** ✓"
          "（清单都没写 ✓、盘上却躺着一件「看着正常」的废片最误导 ✓）",
          not any(bad.iterdir()), sorted(item.name for item in bad.iterdir()))


def case_manifest(root: Path) -> None:
    from app.services.engine.torch_backend import TorchBackend  # noqa: PLC0415

    backend = TorchBackend(device="cpu")
    if not backend.describe()["available"]:
        skip("torch 不可用 ⇒ 清单判据跳过 ✓")
        return

    out = root / "manifest"
    out.mkdir(parents=True, exist_ok=True)
    request = pipe.GenerationRequest(prompt="雨夜", steps=2, seed=777, outputs_dir=str(out),
                                     stage="h3")
    plan = pipe.build_plan(request)
    wave = _tone(0.2)
    result = backend.write(
        {"frames": _frames(int(plan.frames), 32), "audioWaveform": wave,
         "sampleRate": SAMPLE_RATE, "synthetic": True}, plan, request)

    kinds = sorted(str(item.get("kind")) for item in result.get("artifacts") or [])
    check("㉖ 双流落盘仍是**两条产物** ✓（video ＋ audio ✓）", kinds == ["audio", "video"], kinds)
    check("㉗ ⭐ mp4 **自带音轨** ✓（`videoHasAudio` ✓）—— 此前只有静音 mp4 ✓✗",
          result.get("videoHasAudio") is True, result.get("video"))
    check("㉘ ``dualStream`` 仍如实标 ✓（与「后端支持音频」是两回事 ✓）",
          result.get("dualStream") is True)

    manifest_path = Path(str(result.get("manifestPath") or ""))
    check("㉙ 产物目录里有 ``manifest.json`` ✓", manifest_path.is_file(), manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    check("㉚ 清单四块齐 ✓（request / plan / 装配摘要 / 产物 ✓）",
          {"schemaVersion", "writtenAt", "request", "plan", "load", "artifacts"} <= set(manifest),
          sorted(manifest))
    check("㉛ 清单里的 request 真带落点与种子 ✓",
          str((manifest.get("request") or {}).get("outputs_dir") or "").endswith("manifest")
          and (manifest.get("request") or {}).get("seed") == 777,
          (manifest.get("request") or {}).get("seed"))
    entries = manifest.get("artifacts") or []
    check("㉜ 清单逐条**真算**了 sha256 / bytes ✓（不是抄回执 ✓）",
          len(entries) == 2 and all(len(str(item.get("sha256") or "")) == 64 for item in entries),
          [item.get("kind") for item in entries])
    check("㉝ 清单里的 sha256 与盘上文件一致 ✓",
          all(_sha256(Path(str(item.get("path")))) == item.get("sha256") for item in entries),
          [str(item.get("path")) for item in entries])
    check("㉞ 清单里带 ffprobe 事实 ✓（`hasAudio` / 帧数 ✓）",
          bool((entries[0].get("facts") or {}).get("nbFrames")) and
          any((item.get("facts") or {}).get("hasAudio") is True
              for item in entries if item.get("kind") == "video"),
          [item.get("kind") for item in entries])
    check("㉟ 清单自己**不混进** ``artifacts`` ✓（它是一页账，不是产物 ✓）",
          all(str(item.get("kind")) != "manifest" for item in result.get("artifacts") or []))


def case_audio_truncation_surfaces(root: Path) -> None:
    """⭐⭐ 音轨**比画面长**时那一截是真丢内容 ✗✗ —— 回执必须点名，不能只报「容器 0.5 秒」。

    （2026-11-08 补 ✗）``media.write_video`` 的 ``-t`` 是**单向**的：短了补静音（无损 ✓）、
    长了**截掉**（内容真丢 ✗）。此前两种情况在回执里**完全同形** ✓✗ —— 因为 ``probe`` 读的是
    **混完之后**那条音轨 ✓（被收过尾 ⇒ 天然等于画面时长 ✓）。此处走**整条**装配线
    （``TorchBackend.write`` ✓）判「上游到底有没有把它说出来」✓。
    """
    from app.services.engine.torch_backend import TorchBackend  # noqa: PLC0415

    backend = TorchBackend(device="cpu")
    if not backend.describe()["available"]:
        skip("torch 不可用 ⇒ 音轨截断判据跳过 ✓")
        return

    out = root / "audio_trim"
    out.mkdir(parents=True, exist_ok=True)
    request = pipe.GenerationRequest(prompt="雨夜", steps=1, seed=11, seconds=1.0,
                                     outputs_dir=str(out), stage="h3")
    plan = pipe.build_plan(request)
    video_seconds = float(plan.frames) / float(plan.fps)
    #: ⭐ 音轨故意给**画面两倍多** ✓（3.0 秒 vs ≈1.0 秒 ✓）—— 让它真的被截 ✓，别挑个刚好等长的
    #:   用例"证明"没事 ✗（那正是旧 ⑩⑪ 两条判据在做的事 ✓✗）。
    wave = _tone(3.0)
    result = backend.write(
        {"frames": _frames(int(plan.frames), 32), "audioWaveform": wave,
         "sampleRate": SAMPLE_RATE, "synthetic": True}, plan, request)

    expected_trim = 3.0 - video_seconds
    trimmed = result.get("audioTruncated") or {}
    check("㉟′ ⭐⭐ 音轨比画面长 ⇒ 回执**单独一格**给出被截掉的秒数 ✗✗"
          "（整条装配线口径：``audioTruncated.trimmedSeconds`` ✓ 与画面/音轨两侧秒数自洽 ✓）"
          "—— 只报「成片 1.0 秒」等于把内容丢失藏进格式里 ✓",
          abs(float(trimmed.get("trimmedSeconds") or 0) - expected_trim) < 0.06
          and abs(float(trimmed.get("videoSeconds") or 0) - video_seconds) < 0.06
          and abs(float(trimmed.get("sourceSeconds") or 0) - 3.0) < 0.02,
          trimmed)
    check("㉟″ ⭐ 同一件事**也要写进人话** ``note`` ✓ 并指明全长音频在 ``audioPath`` ✓"
          "（``why`` 里点名「内容丢失」✗ 不是口径问题 ✓）",
          "截" in str(result.get("note") or "")
          and "audioPath" in str(result.get("note") or "")
          and "wav" in str(result.get("note") or "")
          and "内容丢失" in str(trimmed.get("why") or ""),
          result.get("note"))
    video_card = result.get("video") or {}
    check("㉟‴ 产物卡的 ``audioDurationSeconds`` 仍是**混完之后**那条 ✓（≈画面长度 ✓）"
          "—— 这个数与「源音轨 3 秒」**必须能并排读** ✓，别让人以为音轨整段进了 mp4 ✗",
          abs(float(video_card.get("audioDurationSeconds") or 0) - video_seconds) < 0.12
          and abs(float(video_card.get("audioSourceSeconds") or 0) - 3.0) < 0.02,
          {k: video_card.get(k) for k in ("audioDurationSeconds", "audioSourceSeconds",
                                          "audioTrimSeconds")})


def case_audio_length_unknown(root: Path) -> None:
    """⭐⭐ 「源音轨多长」**读不出来** ≠ 「没截断」✗✗（2026-11-08 补 ✗）。

    与帧数那个是**同一个病根**（读数缺失被当成读数说没问题 ✓）：``write_video`` 里原本
    ``if audio_seconds is not None:`` 之后**没有 else** ✓ ⇒ 读不到就整段跳过 ✓，
    而 ``torch_backend`` 那句人话照旧写着「音轨补静音到画面长度 ✓」✓✗ ——
    「音轨被截掉 1 秒」与「有没有被截**不知道**」在回执里完全同形 ✓。

    这里同时验**两件不同的事** ✓：
    ① 读不到**先别急着说读不到** ✗✗：标准库 ``wave`` 只认 PCM ✓（float32 / EXTENSIBLE 这些
       **完全合法**的 wav 它会 ``unknown format`` ✗），而 ffmpeg 照混不误 ⇒ 必须 **ffprobe 兜底** ✓
       （实拍：``pcm_f32le`` 的 wav 走 ``wave`` 必失败 ✓）；
    ② 两路都真的没有 ⇒ 回执**明说不知道** ✓，并**收回**那句「补静音到画面长度」✗。
    """
    from app.services.engine.torch_backend import TorchBackend  # noqa: PLC0415

    f32 = root / "tone_f32.wav"
    done = subprocess.run(
        [media_mod._tool("ffmpeg"), "-y", "-v", "error",  # noqa: SLF001
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1.0",
         "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(f32)],
        capture_output=True, timeout=120)
    if done.returncode:
        skip(f"造不出 float32 wav ⇒ 音轨时长兜底判据跳过 ✓（{done.stderr[-200:]!r}）")
        return
    check("㊱¹ 前提：``pcm_f32le`` 的 wav 是**合法** wav ✓ 而标准库 ``wave`` 读不了它 ✓"
          "（真 ffprobe 读得到 ⇒ 这条前提不成立的话后面三条都是空转 ✓）",
          media_mod._wav_seconds(f32) is None  # noqa: SLF001
          and abs(float(media_mod._probe_audio_seconds(f32) or 0) - 1.0) < 0.05,
          {"wavSeconds": media_mod._wav_seconds(f32),  # noqa: SLF001
           "ffprobeSeconds": media_mod._probe_audio_seconds(f32)})  # noqa: SLF001

    frames = _frames(12)                       # 12 帧 @24 ⇒ 画面 0.5 秒 ✓
    video_a = root / "audio_len_a.mp4"
    facts_a = media_mod.write_video(frames, video_a, fps=FPS, audio=f32, value_range="-1..1")
    check("㊱² ⭐⭐ ``wave`` 读不了 ⇒ 走 **ffprobe 兜底** ✓ 拿到真时长 1.0 秒 ✓ 并标明来源 ✓"
          "（``audioLengthVerified == 'ffprobe'`` ✓、``audioSourceSeconds`` ≈1.0 ✓、"
          "截断照旧对账出 0.5 秒 ✗）—— 只认 ``wave`` 会把「能读到的」说成「读不出来」✗",
          facts_a.get("audioLengthVerified") == "ffprobe"
          and abs(float(facts_a.get("audioSourceSeconds") or 0) - 1.0) < 0.05
          and abs(float(facts_a.get("audioTrimSeconds") or 0) - 0.5) < 0.05
          and facts_a.get("audioTruncated") is True,
          {k: facts_a.get(k) for k in ("audioLengthVerified", "audioSourceSeconds",
                                       "audioTrimSeconds", "audioTruncated")})

    real_probe = media_mod.probe

    def _hiding_probe(path):
        """只把 **wav** 的时长涂掉 ✓（真 ffprobe 仍在底下跑 ✓，手法同 ㉔ 那组 ✓）。

        ⚠️ 判据按**后缀**不按文件名 ✓✗：整条装配线那份源 wav 的名字是 ``audio_seed12.wav`` ✓
        （由 ``write_wav`` 起名 ✓），拿 ``tone_f32.wav`` 去比会**永远不命中** ✓ ⇒ 判据看着是绿的、
        其实一帧都没涂 ✗（这正是"判据自己编码错"的那类假绿 ✓）。
        """
        facts = dict(real_probe(path))
        if Path(path).suffix.lower() == ".wav":
            facts["audioDurationSeconds"] = None
            facts["durationSeconds"] = None
        return facts

    video_b = root / "audio_len_b.mp4"
    media_mod.probe = _hiding_probe
    real_decode = media_mod._decode_audio_seconds  # noqa: SLF001
    #: ⚠️ 第四级也要涂掉 ✓✗：不涂它的话这一级会**真解码**那份 wav 拿到 1.0 秒 ✓ ⇒
    #:   「不知道」这一档根本不触发（实测：只在去掉这条 patch 时 ㊱³/㊱⁴ 一起红 ✓）——
    #:   这正是「能力提升后，旧判据的模拟面要跟着扩」✓，否则测的还是上一版的世界 ✓。
    media_mod._decode_audio_seconds = lambda *a, **k: None  # noqa: SLF001
    try:
        facts_b = media_mod.write_video(frames, video_b, fps=FPS, audio=f32, value_range="-1..1")
    finally:
        media_mod.probe = real_probe
        media_mod._decode_audio_seconds = real_decode  # noqa: SLF001
    check("㊱³ ⭐⭐ 三路都没读到 ⇒ 必须**明说不知道** ✓✗ 且**一格假数都不编** ✗"
          "（``audioLengthVerified is None`` ✓、``audioSourceSeconds``/``audioTrimSeconds``/"
          "``audioPadSeconds`` 三格**全不出现** ✓、``audioLengthNote`` 里点名「不知道」✓）"
          "—— 旧代码在这里整段跳过 ⇒ 与「没截断」同形 ✓✗",
          facts_b.get("audioLengthVerified") is None
          and "audioSourceSeconds" not in facts_b
          and "audioTrimSeconds" not in facts_b
          and "audioPadSeconds" not in facts_b
          and "不知道" in str(facts_b.get("audioLengthNote") or ""),
          {k: facts_b.get(k) for k in ("audioLengthVerified", "audioSourceSeconds",
                                       "audioTrimSeconds", "audioPadSeconds")})

    #: ⭐⭐ ④「不知道」还要分**两句**说 ✓✗（2026-11-09 补 ✗）：源超过全解码上限 ⇒ 回执必须说
    #:   「**没有量到底**」✓，不能笼统说「读不出来」✗ —— 前者是"这次没试" ✓（我们自己的闸门 ✓），
    #:   后者会让用户去查编码、换封装 ✓，白查一场 ✗。
    #:   ⚠️ 注入方式：只把**流时长**涂掉 ✓（``_hiding_audio_only`` 保留 ``durationSeconds`` ✓，
    #:   那是闸门唯一要看的预估值 ✓）＋ 把上限压到 0.001 秒 ✓。
    #:   ⚠️⚠️ 这里同时是「默认值不许在 ``def`` 时绑死」的判据 ✓✗：``_decode_audio_seconds`` 的
    #:   ``max_seconds`` 若写成 ``= _MAX_DECODE_SECONDS`` ✗，patch 模块常量对它**毫无作用** ✓
    #:   ⇒ 真解码照样成功 ⇒ 这条会红在"``audioLengthVerified`` 不是 None"上 ✓（不是空转 ✗）。
    def _hiding_audio_only(path):
        facts = dict(real_probe(path))
        facts["audioDurationSeconds"] = None        # 流时长没了 ⇒ ffprobe 那路读不到 ✓
        return facts                                # 容器时长**留着** ⇒ 闸门有预估 ✓

    video_d = root / "audio_len_d.mp4"
    real_cap = media_mod._MAX_DECODE_SECONDS  # noqa: SLF001
    media_mod.probe = _hiding_audio_only
    media_mod._MAX_DECODE_SECONDS = 0.001  # noqa: SLF001
    try:
        facts_d = media_mod.write_video(frames, video_d, fps=FPS, audio=f32, value_range="-1..1")
    finally:
        media_mod.probe = real_probe
        media_mod._MAX_DECODE_SECONDS = real_cap  # noqa: SLF001
    note_d = str(facts_d.get("audioLengthNote") or "")
    check("㊱⁶ ⭐⭐ 源超全解码上限 ⇒ 「不知道」也要说清**是哪一档**不知道 ✓✗"
          "（``audioLengthNote`` 点名「**没有量到底**」✓、**不许**说「读不出来」✗；"
          "``audioLengthVerified is None`` ✓、``audioSourceSeconds`` **不出现** ✓ 一格假数都不编 ✗）"
          "—— 两种原因混成一句 ⇒ 用户会去查一个并不存在的编码问题 ✗",
          facts_d.get("audioLengthVerified") is None
          and "没有量到底" in note_d
          #: ⚠️ 判据必须盯**那一句的字面形** ✓✗：这份人话里本来就有一句「（不是读不出来 ✓）」✓
          #:   ⇒ 只判「不含『读不出来』」会把正确输出判红 ✓（首跑实测正是如此 ✗）。
          #:   要禁的是**另一档的开头**：「源音轨时长**读不出来**」✓。
          and "源音轨时长**读不出来**" not in note_d
          and "audioSourceSeconds" not in facts_d
          and "audioTrimSeconds" not in facts_d,
          {"verified": facts_d.get("audioLengthVerified"), "note": note_d})

    #: ⭐⭐ ③ **容器时长顶替流时长**这一档（2026-11-09 补 ✗）：把那份真 wav 封进 **mkv** ⇒
    #:   封装**不给**流的 ``duration`` ✗、只给容器一条**带余量**的时长 ✗（实测 1.0 秒 ⇒ 1.045 ✓）。
    #:   旧代码 ``(audio.get("duration") or duration)`` 会静默拿容器数当**源音轨长度** ✓✗ ⇒
    #:   与 0.5 秒画面一比就报「截掉 0.545 秒」✗✗ —— 一个**纯属封装余量**的假截断 ✓。
    mkv = root / "tone_f32.mkv"
    mkv_done = subprocess.run(
        [media_mod._tool("ffmpeg"), "-y", "-v", "error", "-i", str(f32),  # noqa: SLF001
         "-c:a", "libmp3lame", str(mkv)], capture_output=True, timeout=120)
    if mkv_done.returncode == 0:
        video_c = root / "audio_len_c.mp4"
        facts_c = media_mod.write_video(frames, video_c, fps=FPS, audio=mkv, value_range="-1..1")
        declared = float(media_mod.probe(mkv).get("audioDurationSeconds") or 0)   # 容器声明（带余量 ✗）
        decoded = media_mod._decode_audio_seconds(mkv)  # noqa: SLF001  —— 全解码（内容长度 ✓）
        #: 前提式分开（同 ⑭⁷ 的口径 ✓）：只差 0.02~0.045 秒 ⇒ 先确认本次封装真拉开了 ✓。
        #: ⚠️ 锚必须是**源内容 1.0 秒**这个独立事实 ✓✗，不能是 ``|declared − decoded|`` ✗ ——
        #:   那是自洽陷阱：解码那支偷懒返回 declared 时它算成 0、条件判放行（反验二实测：改前仍绿 ✗）。
        gap_c = abs(declared - 1.0)
        got_c = float(facts_c.get("audioSourceSeconds") or 0)
        check("㊱⁵ ⭐⭐ mkv 源（流不给 ``duration`` ✗、容器给 1.045 ✗）⇒ **走全解码**拿真时长 ✓✗"
              "（``audioLengthVerified == 'decoded'`` ✓、``audioSourceSeconds`` 等于**解码值** ✓、"
              "两者明显不同时**不等于**容器声明值 ✗、``audioTrimSeconds`` ≈0.5 秒照旧对账出来 ✓）"
              "—— 上一版到这里只能认「不知道」✓；而拿容器的 1.045 顶替会报**带封装余量**的假截断 ✗✗",
              media_mod.probe(mkv).get("audioDurationSource") == "container"
              and facts_c.get("audioLengthVerified") == "decoded"
              and decoded is not None
              and abs(got_c - decoded) < 0.005
              and (gap_c <= 0.01 or abs(got_c - declared) > 0.01)
              and abs(float(facts_c.get("audioTrimSeconds") or 0) - 0.5) < 0.06,
              {"mkvSource": media_mod.probe(mkv).get("audioDurationSource"),
               "declared": declared, "decoded": decoded, "gap": round(gap_c, 4),
               **{k: facts_c.get(k) for k in ("audioLengthVerified", "audioSourceSeconds",
                                              "audioTrimSeconds")}})
    else:
        skip(f"封不出 mkv ⇒ 容器回落那条判据跳过 ✓（{mkv_done.stderr[-160:]!r}）")

    backend = TorchBackend(device="cpu")
    if not backend.describe()["available"]:
        skip("torch 不可用 ⇒ 整条装配线那条判据跳过 ✓")
        return

    out = root / "audio_len_unknown"
    out.mkdir(parents=True, exist_ok=True)
    request = pipe.GenerationRequest(prompt="雨夜", steps=1, seed=12, seconds=1.0,
                                     outputs_dir=str(out), stage="h3")
    plan = pipe.build_plan(request)
    real_wav_seconds = media_mod._wav_seconds  # noqa: SLF001
    media_mod.probe = _hiding_probe           # ⚠️ 只对**那份 wav** 生效 ✓（成片 mp4 照旧读 ✓）
    media_mod._wav_seconds = lambda path: None  # noqa: SLF001  —— 模拟 `wave` 也读不了 ✓
    media_mod._decode_audio_seconds = lambda *a, **k: None  # noqa: SLF001  —— 连解码也读不到 ✓
    try:
        result = backend.write(
            {"frames": _frames(int(plan.frames), 32), "audioWaveform": _tone(1.0),
             "sampleRate": SAMPLE_RATE, "synthetic": True}, plan, request)
    finally:
        media_mod.probe = real_probe
        media_mod._wav_seconds = real_wav_seconds  # noqa: SLF001
        media_mod._decode_audio_seconds = real_decode  # noqa: SLF001
    unverified = result.get("audioTrimUnverified") or {}
    check("㊱⁴ ⭐⭐ 整条装配线：三路都读不到 ⇒ 回执**单独一格** ``audioTrimUnverified`` ✓ ＋ "
          "人话里说「**不知道**」✓ 且**不许**再写「补静音到画面长度」✗✗"
          "（那句话是**肯定**句 ✓，读不到时它等于替用户担保「音频整段都在」✗）",
          result.get("audioLengthVerified") is None
          and bool(unverified)
          and "不知道" in str(result.get("note") or "")
          and "补静音到画面长度" not in str(result.get("note") or ""),
          {"audioLengthVerified": result.get("audioLengthVerified"),
           "unverified": unverified, "note": result.get("note")})

    #: ⭐⭐ ⑤ 两档「不知道」必须**一路走到回执** ✓✗（2026-11-09 补 ✗）：``media.write_video`` 里
    #:   分好的 ``too_long`` / ``unreadable`` 落到 ``torch_backend`` 时，那句人话原本是**写死**的
    #:   「源音轨到底多长读不出来」✗、``audioTrimUnverified.why`` 也写死「**两路**都没读到」✗
    #:   ⇒ 上游刚分好的两档在半路被**拍平** ✓✗：前端看到的还是同一句，"你没让它解"和"文件坏了"
    #:   又变回同形 ✓（这正是这条线的老病根，只是换到下一层犯 ✓）。判据同时钉住两档 ✓：
    #:   ``reason`` 单列成机器码 ✓、两档人话不许互相串 ✓。
    out_t = root / "audio_len_toolong"
    out_t.mkdir(parents=True, exist_ok=True)
    request_t = pipe.GenerationRequest(prompt="雨夜", steps=1, seed=12, seconds=1.0,
                                       outputs_dir=str(out_t), stage="h3")
    plan_t = pipe.build_plan(request_t)
    real_cap_t = media_mod._MAX_DECODE_SECONDS  # noqa: SLF001
    media_mod.probe = _hiding_audio_only        # 只涂流时长 ⇒ 闸门仍有预估 ✓（同 ㊱⁶ ✓）
    media_mod._wav_seconds = lambda path: None  # noqa: SLF001
    media_mod._MAX_DECODE_SECONDS = 0.001  # noqa: SLF001
    try:
        result_t = backend.write(
            {"frames": _frames(int(plan_t.frames), 32), "audioWaveform": _tone(1.0),
             "sampleRate": SAMPLE_RATE, "synthetic": True}, plan_t, request_t)
    finally:
        media_mod.probe = real_probe
        media_mod._wav_seconds = real_wav_seconds  # noqa: SLF001
        media_mod._MAX_DECODE_SECONDS = real_cap_t  # noqa: SLF001
    unverified_t = result_t.get("audioTrimUnverified") or {}
    note_t = str(result_t.get("note") or "")
    check("㊱⁷ ⭐⭐ 两档「不知道」必须**一路走到回执** ✓✗：源过长 ⇒ 回执级 "
          "``audioLengthMiss == 'too_long'`` ✓ ＋ ``audioTrimUnverified.reason == 'too_long'`` ✓ ＋ "
          "人话「没有量到底」✓；而 ㊱⁴ 那次三路全失败的这两格都是 ``'unreadable'`` ✓。"
          "⚠️ **两处**中文人话（``audio_tail`` 与 ``audioTrimUnverified``）各自都钉一句字面形 ✓✗ —— "
          "只钉一处时反验关掉另一处仍然绿 ✓（首跑实测：拍平 ``audio_tail`` 后判据照样 PASS ✗）",
          result_t.get("audioLengthMiss") == "too_long"
          and unverified_t.get("reason") == "too_long"
          and "没有量到底" in note_t
          #: ``audio_tail`` 那半若回退成写死的「读不出来」，note 里就会出现这一句 ✓
          and "源音轨到底多长**读不出来**" not in note_t
          #: ``audioTrimUnverified.why`` 那半若回退成旧文案 ✓
          and "两路都没读到" not in str(unverified_t.get("why") or "")
          and result.get("audioLengthMiss") == "unreadable"
          and (unverified or {}).get("reason") == "unreadable",
          {"tooLongMiss": result_t.get("audioLengthMiss"),
           "tooLongReason": unverified_t.get("reason"), "tooLongNote": note_t,
           "tooLongWhy": unverified_t.get("why"),
           "unreadableMiss": result.get("audioLengthMiss"),
           "unreadableReason": (unverified or {}).get("reason")})

    #: ⭐⭐ ⑥「音轨**在**」≠「音轨**有声音**」 ✗✗（2026-11-09 补 ✗）：``videoHasAudio`` /
    #:   ``audioLengthVerified`` / ``audioTruncated`` 全都只回答"这条流**存在**吗" ✓，
    #:   **没有一格**回答"这段波形**有没有振幅**"✗ ⇒ 全零波形与正常音频在回执里同形 ✓✗：
    #:   wav 照写 ✓、混进 mp4 照成 ✓、回执照说「真 wav ✓ 已混进 mp4 ✓」✓ ⇒ 用户点开是无声片，
    #:   而回执里一个提示都没有 ✗（``write_image`` 那边早有 ``blank_std`` ✓，音频侧没有对应物 ✗）。
    #:   ⚠️ 人话必须把两件事**分开** ✓：要说的不是"音频没混进去"✗（那会让人去查 ffmpeg ✗），
    #:   而是"音轨在、内容是静的"✓ —— 该查的是波形生成那一段 ✓。
    #:   对照组用 ㊱⁷ 那次真 tone ✓（同一段代码路径 ✓，差别只在波形有没有振幅 ✓）。
    out_z = root / "audio_silent"
    out_z.mkdir(parents=True, exist_ok=True)
    request_z = pipe.GenerationRequest(prompt="雨夜", steps=1, seed=13, seconds=1.0,
                                       outputs_dir=str(out_z), stage="h3")
    plan_z = pipe.build_plan(request_z)
    result_z = backend.write(
        {"frames": _frames(int(plan_z.frames), 32),
         "audioWaveform": _torch().zeros(1, SAMPLE_RATE), "sampleRate": SAMPLE_RATE,
         "synthetic": True}, plan_z, request_z)
    audio_z = result_z.get("audio") or {}
    note_z = str(result_z.get("note") or "")
    check("㊱⁸ ⭐⭐ 全零波形 ⇒ 回执必须点名「**几乎是静音**」✓✗：``audio.silent is True`` ✓、"
          "``audio.peakDbfs is None`` ✓（不是 ``-inf`` ✗）、人话同时写清**不是**「音频没混进去」✗ "
          "—— 音轨在、内容是静的 ✓；同一句还要进 ``alerts`` ✓（前端只显示 ``warnings`` ✓，"
          "只写 ``note`` ＝ 用户看不到 ✗）；对照组（㊱⁷ 那次真 tone）``silent is False`` ✓ 且不提静音 ✗",
          audio_z.get("silent") is True and audio_z.get("peakDbfs") is None
          #: ⚠️ 字面形带加粗标记 ✓（人话原文是「**不是**「音频没混进去」」✓ —— 只判
          #:   「不是「音频没混进去」」会**恒假** ✗，这条首跑就是这么红的 ✓）。
          and "几乎是静音" in note_z and "**不是**「音频没混进去」" in note_z
          and result_z.get("videoHasAudio") is True
          #: ⭐ ``alerts`` 里的句子必须是 ``note`` 里那一句**同一个字符串** ✓✗（同源 ✓）：
          #:   两处各写一遍必然哪天只改一处 ✓ ⇒ 界面上显示与回执长文说的是两件事 ✗。
          and any(str(s) in note_z for s in (result_z.get("alerts") or []))
          and (result_t.get("audio") or {}).get("silent") is False
          #: ⚠️ 对照组**可以有**别的 alerts ✓（㊱⁷ 那次本来就有「源音轨没有量到底」✓）——
          #:   这里只判"静音那条不许出现"✗，别把整个列表判空 ✗（首跑正是这么红的 ✓）。
          and not any("几乎是静音" in str(s) for s in (result_t.get("alerts") or []))
          and "几乎是静音" not in note_t,
          {"silent": audio_z.get("silent"), "peakDbfs": audio_z.get("peakDbfs"),
           "hasAudio": result_z.get("videoHasAudio"), "alerts": result_z.get("alerts"),
           "note": note_z,
           "controlSilent": (result_t.get("audio") or {}).get("silent"),
           "controlAlerts": result_t.get("alerts")})

    #: ⭐⭐ ⑦ 纯音频那一支**更需要**这句 ✓✗（2026-11-09 补 ✗）：出歌**没有画面可佐证** ✓，
    #:   用户只能靠耳朵 ✓ ⇒ 一段全静音的 wav 若回执照说「真 wav ✓ 这就是成品音频 ✓」✓✗，
    #:   用户会先怀疑"我哪设错了"（歌词？步数？种子？）✓，而真相是波形那一段没出东西 ✓。
    #:   ⚠️ 分支顺序：这条走 ``waveform is not None`` ✓（没有 5 维帧 ✓ ⇒ 不会落成 mp4 ✓）。
    out_o = root / "audio_only_silent"
    out_o.mkdir(parents=True, exist_ok=True)
    request_o = pipe.GenerationRequest(prompt="雨夜", steps=1, seed=14, seconds=1.0,
                                       outputs_dir=str(out_o), stage="h3")
    plan_o = pipe.build_plan(request_o)
    result_o = backend.write(
        {"audioWaveform": _torch().zeros(2, 8000), "sampleRate": 8000,
         "audioSamples": 16000, "synthetic": True}, plan_o, request_o)
    note_o = str(result_o.get("note") or "")
    check("㊱⁹ ⭐⭐ 纯音频分支（没有任何画面）也必须把「几乎是静音」抬进 ``alerts`` ✓✗："
          "``audio.silent is True`` ✓、``alerts`` 那句与 ``note`` 逐字同源 ✓、"
          "``primaryPath`` 是 ``.wav`` ✓（这条分支**没有** mp4 ✗ —— 不许把 wav 当视频用 ✗）",
          (result_o.get("audio") or {}).get("silent") is True
          and any("几乎是静音" in str(s) and str(s) in note_o
                  for s in (result_o.get("alerts") or []))
          and "**不是**「文件没写成」" in note_o
          and str(result_o.get("primaryPath") or "").endswith(".wav"),
          {"silent": (result_o.get("audio") or {}).get("silent"),
           "alerts": result_o.get("alerts"), "note": note_o,
           "primaryPath": result_o.get("primaryPath")})

    #: ⭐⭐ ⑧ 图那一支也要「被看见」 ✓✗（2026-11-10 补 ✗）：``write_image`` 早有 ``blank_std``
    #:   ⇒ 纯色/全黑/全白**不抛异常** ✓✗（画布上看着就是"一张图" ✓），``result["degraded"]``
    #:   与 ``note`` 里那句都写了 ✓ —— 可前端只显示 ``warnings`` ✓（``note``/``degraded`` 一个
    #:   都不读 ✗）⇒ 用户拿到一张灰片、界面上零提示 ✓✗。上一轮只给音轨补了 ``alerts`` ✓✗，
    #:   图这一支漏着 ✓ —— 同一个病根在**同一个函数**里换个分支再犯一次 ✓。
    #:   对照组用同一函数里的随机噪声图 ✓（同一段代码路径 ✓，差别只在像素有没有内容 ✓）。
    out_i = root / "image_flat"
    out_i.mkdir(parents=True, exist_ok=True)
    request_i = pipe.GenerationRequest(prompt="雨夜", steps=1, seed=15, seconds=1.0,
                                       outputs_dir=str(out_i), stage="zimage")
    plan_i = pipe.build_plan(request_i)
    flat_i = _torch().full((1, 3, 32, 32), -1.0)          # 值域下限 ⇒ 纯黑 ✓
    result_i = backend.write({"frames": flat_i, "synthetic": False}, plan_i, request_i)
    noise_i = _torch().rand(1, 3, 32, 32, generator=_torch().Generator().manual_seed(2)) * 2 - 1
    out_n = root / "image_noise"
    out_n.mkdir(parents=True, exist_ok=True)
    request_n = pipe.GenerationRequest(prompt="雨夜", steps=1, seed=16, seconds=1.0,
                                       outputs_dir=str(out_n), stage="zimage")
    result_n = backend.write({"frames": noise_i, "synthetic": False},
                             pipe.build_plan(request_n), request_n)
    note_i = str(result_i.get("note") or "")
    check("㊱¹⁰ ⭐⭐ 纯色图 ⇒ ``degraded is True`` ✓ 且那句「**几乎是纯色**」必须进 ``alerts`` ✓✗"
          "（且与 ``note`` 逐字同源 ✓）；对照组随机噪声图 ``degraded`` 为假 ✓ 且 ``alerts`` 空 ✓ "
          "—— 只写 ``note`` ＝ 界面上一句看不到 ✗",
          result_i.get("degraded") is True
          and bool(result_i.get("degradedReason"))
          and any("几乎是纯色" in str(s) and str(s) in note_i
                  for s in (result_i.get("alerts") or []))
          #: ⚠️ 对照组**不能判** ``is False`` ✗：正常图那一支**根本不写** ``degraded`` 这个键 ✓
          #:   （只在真标了才写 ``True`` ✓）⇒ 读到的是 ``None`` ✓，判 ``is False`` 会**恒假** ✗
          #:   （首跑正是这么红的：``controlDegraded: None`` ✓）。
          and not result_n.get("degraded")
          and not (result_n.get("alerts") or []),
          {"degraded": result_i.get("degraded"), "alerts": result_i.get("alerts"),
           "reason": result_i.get("degradedReason"),
           "controlDegraded": result_n.get("degraded"),
           "controlAlerts": result_n.get("alerts")})

    #: ⭐⭐ 图像这一支的元数据也要**读回来核** ✓✗（2026-11-10 补 ✗）：
    #:   此前 ``metadataKeys`` 列的是**送出去的键名** ✓✗ —— 而 ``_png_info`` 会跳过
    #:   ``value is None`` 与空键名 ✓ ⇒ 「打算写 N 个」被当成「写成了 N 个」✓（甚至交出
    #:   一个**根本不在 PNG 里**的键 ✗）。mp4 那侧 2026-11-07 就有 ``metadataDropped`` ✓、
    #:   图这侧一直没有 ✓ ⇒ 同口径：``metadataReadback`` 三档 ＋ ``metadataMissing`` 点名列不出 ✓。
    image_i = result_i.get("image") or {}
    check("㊱¹³ ⭐⭐ 图像元数据**从盘上读回来**才算数 ✓✗：``image.metadataReadback is True`` ✓、"
          "``metadataKeys`` 非空 ✓、``metadataMissing`` 为空 ✓；"
          "正常 plan 的元数据不该有丢的 ⇒ ``alerts`` 里**不许**出现「没写进 PNG」✗（喊狼也是错 ✓）",
          image_i.get("metadataReadback") is True
          and bool(image_i.get("metadataKeys"))
          and not image_i.get("metadataMissing")
          and result_i.get("metadataReadback") is True
          and not any("没写进 PNG" in str(s) for s in (result_i.get("alerts") or [])),
          {"readback": image_i.get("metadataReadback"),
           "keys": image_i.get("metadataKeys"),
           "missing": image_i.get("metadataMissing"),
           "alerts": result_i.get("alerts")})

    #: ⭐⭐ 帧率也得在**回执顶层**分得清 ✓✗（2026-11-10 补 ✗，判据在 ``media.compare_fps`` ✓）：
    #:   ``video.fps`` 是**盘上**读数 ✓、请求值另在 ``fpsRequested`` ✓ —— 此前 ``torch_backend``
    #:   的回执里**一格都没有** ✓✗（要自己钻进 ``video`` 里拿两个数比 ✓，而别人不会去比 ✓）。
    #:   帧率一漂 ⇒ **音画不同步** ✗，可 ``durationSeconds``（真实 ✓）与 ``videoSeconds``
    #:   （按请求 fps 算 ✓）都"有值" ✓ ⇒ 看起来一切正常 ✓✗。
    #:   ⚠️ 一致性时**也不许**喊 ✗：老喊狼的告警，用户第三次就学会无视它了 ✓✗。
    video_z = result_z.get("video") or {}
    want_fps = int(plan_z.fps or 24)
    check("㊱¹¹ ⭐⭐ 帧率对账**一路走到回执顶层** ✓✗：``fpsVerified == 'exact'`` ✓、"
          "``fpsRequested == plan.fps`` ✓、盘上 ``video.fps`` 与请求一致 ✓；"
          "一致时 ``alerts`` 里**不许**出现帧率告警 ✗（喊狼也是错 ✓）",
          result_z.get("fpsVerified") == "exact"
          and result_z.get("fpsRequested") == want_fps
          and float(video_z.get("fps") or 0) == float(want_fps)
          and not any("帧率" in str(s) for s in (result_z.get("alerts") or []))
          and "fpsDrift" not in video_z,
          {"fpsVerified": result_z.get("fpsVerified"),
           "fpsRequested": result_z.get("fpsRequested"), "fpsOnDisk": video_z.get("fps"),
           "alerts": result_z.get("alerts")})

    #: ⭐⭐ 「削顶」必须**说出来** ✗（2026-11-10 补 ✗）：``write_wav`` 早把 ``clipped``
    #:   （越界样本数 ✓）算出来了 ✓，可它只躺在 ``audio`` 子字典里 ✓✗、``note``/``alerts``
    #:   一个字不提 ✓。削顶**不抛异常** ✗、也**不会**被判成 ``silent`` ✗（峰值恒 1.0 ✓ 更"响" ✓）
    #:   ⇒ 整段爆音与"增益刚好"在回执里同形 ✓ —— 本族最容易被当成"正常"的那一格 ✓。
    clip_none = torch_backend_mod.TorchBackend._clip_alert({"clipped": 0, "samples": 100, "channels": 2})
    clip_some = torch_backend_mod.TorchBackend._clip_alert(
        {"clipped": 100, "samples": 1000, "channels": 2})
    clip_empty = torch_backend_mod.TorchBackend._clip_alert({})
    check("㊱¹⁴ ⭐⭐ 「削顶」有**唯一句源** ✓ 且三档分得开 ✓✗：``clipped == 0`` ⇒ ``None`` ✓"
          "（不喊狼 ✓）；``clipped > 0`` ⇒ 一句人话**点名爆音/失真** ✓、给出占比 ✓、"
          "并**先排除**解码/混流（锅在波形生成的增益/归一化 ✓）；"
          "键缺失 ⇒ ``None`` ✓（没测过 ≠ 测了没问题 ✗）",
          clip_none is None and clip_empty is None
          and bool(clip_some) and "削顶" in str(clip_some)
          and "爆音" in str(clip_some) and "5.00%" in str(clip_some),
          {"zero": clip_none, "some": clip_some, "empty": clip_empty})

    #: ⭐⭐ 通道端到端 ✓✗（2026-11-10 补 ✗）：后端把结论写在 ``backend.write()`` 的返回里 ✓，
    #:   而它就是 :attr:`PipelineResult.outputs` ✓ ⇒ 前端读的 ``task.result.alerts`` 顶层
    #:   **此前什么都没有** ✓✗（``task.result = PipelineResult.to_dict()`` ✓）。这里用**真** result
    #:   走一遍 ``to_dict()``，判的就是用户真拿到的那份对象 ✓。
    wire_z = pipe.PipelineResult(ok=True, outputs=result_z).to_dict()
    want_alerts = [str(s) for s in (result_z.get("alerts") or [])]
    check("㊱¹⁵ ⭐⭐ ``alerts`` 真的走到**用户拿到的那份回执**里 ✓✗：``to_dict()['alerts']`` 与 "
          "``outputs.alerts`` **逐字**一致 ✓；有就抬、没有就**不出现这个键** ✓"
          "（区分「一条都没有」与「没说」✓）",
          [str(s) for s in (wire_z.get("alerts") or [])] == want_alerts
          and ("alerts" in wire_z) == bool(want_alerts),
          {"topLevel": wire_z.get("alerts"), "outputs": want_alerts})

    #: ⭐⭐ ``alerts`` 的写入口只许**追加** ✓✗（2026-11-10 补 ✗）：同一个 ``result`` 上有**四处**
    #:   写 alerts ✓（纯色图 ✓ / 元数据丢失 ✓ / 静音 ✓ / 削顶 ✓），此前是**两种写法混用** ——
    #:   「``result["alerts"] = [...]``」**覆盖** ✓✗ 与「``list(result.get("alerts") or []) + [...]``」
    #:   追加 ✓。今天恰好覆盖写法都排在追加之前 ✓（所以没错 ✓），可只要有人以后往上面再加一句 ✓，
    #:   前面那条结论就**静默消失** ✓✗ —— 不报错 ✗、判据也看不出来 ✗，正是本族最坏的那种隐患 ✓。
    #:   ⇒ 立成硬规矩：一律 ``result.setdefault("alerts", []).append(...)`` ✓（单一写法 ✓），
    #:   覆盖写法一处都不许留 ✓。（扫描对象是**源码字面量** ✓，与 ⑤/⑨ 那种"真跑一遍"互补 ✓：
    #:   行为判据只能看见今天跑到的分支 ✓，写法判据才不会随分支漏掉将来那句 ✓。）
    engine_dir = Path(torch_backend_mod.__file__).resolve().parent
    alert_offenders = []
    lift_sites = []
    for source in sorted(engine_dir.glob("*.py")):
        for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            overwrite = '["alerts"] = [' in line or '["alerts"] = list(' in line
            if not overwrite:
                continue
            #: ``payload["alerts"] = [...]`` 是**消费侧**的唯一抬层口 ✓（``to_dict()`` 里，
            #:   它写的是**交给用户的那份新字典** ✓，天然只能整份给出 ✓、不存在"抹掉前一句" ✓）
            #:   ⇒ 它是允许的那一处 ✓；除此之外任何「``X["alerts"] = [...]``」都是**生产侧**
            #:   的错误写法 ✗（生产侧四处在同一个 ``result`` 上跑 ✓）。
            if 'payload["alerts"]' in line:
                lift_sites.append(f"{source.name}:{lineno}")
            else:
                alert_offenders.append(f"{source.name}:{lineno}")
    check("㊱¹⁶ ⭐⭐ ``alerts`` 生产侧只许**追加**、不许覆盖 ✓✗：``app/services/engine/*.py`` 里除 "
          "``to_dict()`` 那一处抬层口 ✓ 之外，没有任何「``X[\"alerts\"] = [...]``」/「``= list(…)``」" "✓ —— 四处写入点必须**同一写法**（``setdefault(...).append(...)`` ✓）；"
          "混着写＝将来在上面再加一句就把前面的结论**静默抹掉** ✗",
          not alert_offenders, alert_offenders)
    check("㊱¹⁶′ 抬层口**只有一处** ✓（``payload[\"alerts\"] = [...]`` 在 ``pipeline.py`` 里出现 1 次 ✓）",
          len(lift_sites) == 1 and lift_sites[0].startswith("pipeline.py:"), lift_sites)


# ══════════════════════════════════════════════════════════════════════════
# ④ 落点：`/engine/generate` 走 `<kind>-<taskId>/`（不再全落一个平铺目录 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_task_outputs_dir(root: Path) -> None:
    engine = runtime_mod.EngineRuntime()
    request = pipe.GenerationRequest(prompt="雨夜", seed=3, steps=1, stage="h3",
                                     outputs_dir="")
    task_id = engine.submit(request, assembly={})
    snapshot = engine.get_task(task_id, events_limit=0) or {}
    assigned = str((snapshot.get("request") or {}).get("outputs_dir") or "")
    check("㊱ ⭐ 没给落点 ⇒ 按任务建子目录 ✓（`<kind>-<taskId>/` ✓）",
          assigned.endswith(f"video-{task_id}"), assigned)
    check("㊲ 真跑的落点在存储根 ``static/engine/`` 之下 ✓（前端能用 ``/static`` 取 ✓）",
          "engine" in Path(assigned).parts and "static" in Path(assigned).parts, assigned)

    #: ⚠️ 干跑报告是**凭据**不是产物 ✓✗ ⇒ 不许落进 ``static``（2026-11-07 补 ✓）
    task_dry = engine.submit(request, assembly={"dryRun": True})
    snapshot_dry = engine.get_task(task_dry, events_limit=0) or {}
    assigned_dry = str((snapshot_dry.get("request") or {}).get("outputs_dir") or "")
    check("㊳ ⭐ 干跑的落点**不在** ``static`` 之下 ✓（`<数据根>/engine-dryrun/` ✓）",
          "static" not in Path(assigned_dry).parts and "engine-dryrun" in Path(assigned_dry).parts,
          assigned_dry)
    check("㊴ 干跑子目录同样带任务号 ✓（`<kind>-<taskId>/` ✓，与真跑同口径 ✓）",
          assigned_dry.endswith(f"video-{task_dry}"), assigned_dry)

    explicit = root / "given"
    task2 = engine.submit(replace(request, outputs_dir=str(explicit)), assembly={"dryRun": True})
    snapshot2 = engine.get_task(task2, events_limit=0) or {}
    given = str((snapshot2.get("request") or {}).get("outputs_dir") or "")
    check("㊵ 调用方**给了**落点就照用 ✓（不替它改道 ✓）", given == str(explicit), given)
    for item in (task_id, task_dry, task2):
        engine.cancel(item)


def case_primary_url() -> None:
    """回执的 ``primaryUrl`` ✓：只发**媒体** ✗ —— 干跑报告/张量清单是 ``.json`` ✓，绝不能当成品链接 ✓。"""
    from app.core.config import get_storage_root  # noqa: PLC0415
    from app.routers.engine import _with_urls  # noqa: PLC0415

    root = Path(get_storage_root())
    video = {"primaryPath": str(root / "engine" / "video-eng-x" / "video_seed777.mp4")}
    check("㊶ 媒体产物照旧拿到 ``/static/...`` ✓",
          str(_with_urls(dict(video)).get("primaryUrl") or "").startswith("/static/engine/"),
          _with_urls(dict(video)).get("primaryUrl"))
    for suffix in (".json", ".txt"):
        report = {"primaryPath": str(root / "engine" / "video-eng-x" / f"dryrun_seed777{suffix}")}
        check(f"㊷ ⭐ ``{suffix}`` 报告**不发** ``primaryUrl`` ✓（它是凭据不是产物 ✓）",
              "primaryUrl" not in _with_urls(dict(report)), _with_urls(dict(report)))
    outside = {"primaryPath": str(root.parent / "engine-dryrun" / "video-eng-x" / "a.mp4")}
    check("㊸ 存储根之外的产物也不发 URL ✓（硬转只会 404 ✓）",
          "primaryUrl" not in _with_urls(dict(outside)), _with_urls(dict(outside)))
    check("㊹ 后缀→kind 的判据只有一份 ✓（``.json`` ⇒ None ✓、``.png`` ⇒ image ✓）",
          runtime_mod.EngineRuntime.kind_of_path("a/b.json") is None
          and runtime_mod.EngineRuntime.kind_of_path("a/b.png") == "image"
          and runtime_mod.EngineRuntime.kind_of_path("a/b.mp4") == "video")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="engine_artifact_"))
    if not _have_ffmpeg():
        skip("ffmpeg/ffprobe 不可用 ⇒ 视频相关判据跳过 ✓")
    else:
        try:
            _torch()
        except ImportError:
            skip("torch 未安装 ⇒ 混流判据跳过 ✓")
        else:
            case_mux(root)
            case_mux_long_audio(root)
            case_mux_guards(root)
            case_image(root)
        case_metadata_helper()
        case_frame_guard(root)
        case_frames_unverifiable(root)
        case_manifest(root)
        case_audio_truncation_surfaces(root)
        case_audio_length_unknown(root)
    case_task_outputs_dir(root)
    case_primary_url()

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
