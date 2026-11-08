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
        case_manifest(root)
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
