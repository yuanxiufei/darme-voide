"""S7 自检：引擎的**解码与落盘**（真 VAE ✓ + 真 mp4/wav ✓ 2026-09-17）。

这套的判据特别强调**独立复核** ✓：写完 mp4 **不是**"函数没报错就算过" ✗ ——
而是用 **ffprobe** 把文件读回来，核对**宽/高/fps/帧数/时长** ✓✓；wav 用标准库 `wave` 读回来核对 ✓。

⚠️ 画面内容是**未经训练的参考 VAE** 的输出 ✓ ⇒ 看起来是噪声 ✓（**验的是管道** ✓，不是"H3 能出片" ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_io_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import wave
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import media as media_mod  # noqa: E402
from app.services.engine import vae as vae_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


# 4 层 ⇒ spatial_scale = 2^3 = 8 ✓（与 DiT 自检里的 `vae_scale=8` 同一口径 ✓）；通道小 ⇒ CPU 秒出 ✓
VAE_CONFIG = vae_mod.VideoVAEConfig(base_channels=8, latent_channels=4,
                                    channel_multipliers=(1, 1, 1, 1))


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════
# ① VAE：形状自洽 + 校验报错（**不悄悄裁剪** ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_vae(root: Path) -> None:
    check("① spatial_scale = 2^层数（3 层 ⇒ 8 ✓，与 `DiTConfig.vae_scale` 同一口径 ✓）",
          VAE_CONFIG.spatial_scale == 8, VAE_CONFIG.spatial_scale)
    check("② 时间压缩显式标 1（本参考实现**不做**时间压缩 ✓ 不假装有 ✗）",
          VAE_CONFIG.to_dict()["temporalCompression"] == 1)
    try:
        vae_mod.VideoVAEConfig(channel_multipliers=())
        empty_ok = False
    except vae_mod.VAEError:
        empty_ok = True
    check("③ 空 channel_multipliers ⇒ 报错 ✓", empty_ok)

    if not _have_torch():
        skip("torch 未安装 ⇒ VAE 前向与落盘跳过 ✓")
        return
    import torch  # noqa: PLC0415

    torch.manual_seed(0)
    model = vae_mod.build_vae(VAE_CONFIG)
    frames = torch.rand(1, 3, 4, 16, 16)          # (B,3,T,H,W) ✓ 16/8 = 2 ✓
    latents = model.encode(frames)
    check("④ 编码：``(B,3,T,H,W)`` → ``(B,C_lat,T,H/S,W/S)``（**时间维不变** ✓）",
          tuple(latents.shape) == (1, 4, 4, 2, 2), tuple(latents.shape))
    restored = model.decode(latents)
    check("⑤ 解码：回到与输入**同形** ✓（机制可逆 ✓，不是「还原画面」✗）",
          tuple(restored.shape) == tuple(frames.shape), tuple(restored.shape))
    check("⑥ 同输入逐位可复现 ✓（无隐藏随机 ✓）",
          bool(torch.allclose(model.decode(model.encode(frames)), restored)), "")

    try:
        model.encode(torch.rand(1, 3, 4, 15, 16))
        odd_ok = False
    except vae_mod.VAEError as err:
        odd_ok = "整除" in str(err)
    check("⑦ 像素尺寸不整除 spatial_scale ⇒ **明确报错** ✓（悄悄裁剪会改变产物却不报 ✗）", odd_ok)
    try:
        model.decode(torch.rand(1, 3, 4, 2, 2))
        channel_ok = False
    except vae_mod.VAEError as err:
        channel_ok = "潜变量形状" in str(err)
    check("⑧ 潜变量通道数不符 ⇒ 报错（不是硬跑出错 ✗）", channel_ok)


# ══════════════════════════════════════════════════════════════════════════
# ② 落盘：**真文件 + 独立复核**（本套的重点 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_media(root: Path) -> None:
    check("⑨ 本机 ffmpeg/ffprobe 可用 ✓（落盘不需要新依赖 ✓ 与色校正功能共用同一件事 ✓）",
          media_mod.have_ffmpeg(), media_mod.ffmpeg_version())
    if not media_mod.have_ffmpeg() or not _have_torch():
        skip("缺 ffmpeg 或 torch ⇒ 落盘部分跳过 ✓")
        return
    import torch  # noqa: PLC0415

    torch.manual_seed(0)
    frames = torch.rand(1, 3, 6, 64, 64) * 2.0 - 1.0      # [-1,1] ✓
    path = root / "out.mp4"
    report = media_mod.write_video(frames, path, fps=24)
    check("⑩ 写出**真 mp4**（文件存在且非空 ✓）",
          path.exists() and path.stat().st_size > 0, report.get("bytes"))
    check("⑪ ⭐ ffprobe 复核：宽/高 = 64×64 ✓（与输入一致 ✓ 不是「随便生成了个文件」✗）",
          (report["width"], report["height"]) == (64, 64), (report["width"], report["height"]))
    check("⑫ ⭐ ffprobe 复核：fps = 24 ✓、编码 = h264 ✓",
          report["fps"] == 24.0 and report["codec"] == "h264", (report["fps"], report["codec"]))
    check("⑬ ⭐ ffprobe 复核：时长 ≈ 6/24 = 0.25s ✓（帧数确实写进去了 ✓）",
          report["durationSeconds"] is not None and abs(report["durationSeconds"] - 0.25) < 0.06,
          report["durationSeconds"])
    check("⑭ 回报里带**请求帧数**（便于交叉核对 ✓）且与时长口径一致 ✓",
          report["requestedFrames"] == 6, report["requestedFrames"])

    try:
        media_mod.write_video(torch.rand(1, 3, 2, 65, 64), root / "odd.mp4")
        odd_ok = False
    except media_mod.MediaError as err:
        odd_ok = "偶数" in str(err) and "yuv420p" in str(err)
    check("⑮ 奇数尺寸 ⇒ **明确报错并解释 yuv420p 要求** ✓（不偷偷裁一行像素 ✗）", odd_ok)

    try:
        media_mod.write_video(torch.rand(1, 3, 2, 8, 8), root / "vr.mp4", value_range="0..5")
        vr_ok = False
    except media_mod.MediaError as err:
        vr_ok = "值域" in str(err)
    check("⑯ 未知值域 ⇒ 报错（猜错值域 = 全白/全黑**且不报错** ✗ —— 最难查 ✓）", vr_ok)

    try:
        media_mod.write_video(torch.rand(2, 3, 2, 8, 8), root / "b2.mp4")
        batch_ok = False
    except media_mod.MediaError as err:
        batch_ok = "batch" in str(err)
    check("⑰ batch>1 ⇒ **明确拒绝** ✓（只取第一段那种静默丢弃最坏 ✗）", batch_ok)

    # ── wav：用标准库读回来核对 ✓ ───────────────────────────────────────
    samples = torch.sin(torch.linspace(0, 40, 16000)) * 0.8      # 0.5s @32k ✓
    wav_path = root / "out.wav"
    wav_report = media_mod.write_wav(samples, wav_path, sample_rate=32000)
    with wave.open(str(wav_path), "rb") as handle:
        read_channels = handle.getnchannels()
        read_rate = handle.getframerate()
        read_frames = handle.getnframes()
    check("⑱ 写出**真 wav**，并用标准库读回来核对（声道/采样率/帧数 ✓）",
          read_channels == 1 and read_rate == 32000 and read_frames == 16000,
          (read_channels, read_rate, read_frames))
    check("⑲ 时长口径一致（16000/32000 = 0.5s ✓）",
          abs(float(wav_report["durationSeconds"]) - 0.5) < 1e-6, wav_report["durationSeconds"])

    loud = torch.tensor([0.0, 1.5, -2.0, 0.5])
    loud_report = media_mod.write_wav(loud, root / "loud.wav", sample_rate=8000)
    check("⑳ 越界样本被**钳位并如实回报** `clipped` ✓（不绕回成大噪声 ✗）",
          loud_report["clipped"] == 2, loud_report["clipped"])
    with wave.open(str(root / "loud.wav"), "rb") as handle:
        import numpy as np  # noqa: PLC0415

        peak = int(np.frombuffer(handle.readframes(4), dtype=np.int16).max())
    check("㉑ 钳位后峰值正好是 int16 上限 ✓（32767 ✓）", peak == 32767, peak)


# ══════════════════════════════════════════════════════════════════════════
# ③ 端到端：潜变量 → VAE 解码 → **真 mp4**（"出片"这条管道 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_end_to_end(root: Path) -> None:
    if not media_mod.have_ffmpeg() or not _have_torch():
        skip("缺 ffmpeg 或 torch ⇒ 出片管道跳过 ✓")
        return
    import torch  # noqa: PLC0415

    torch.manual_seed(3)
    model = vae_mod.build_vae(VAE_CONFIG)
    latents = torch.randn(1, VAE_CONFIG.latent_channels, 8, 8, 8)   # (B,C,T,h,w) ✓
    with torch.no_grad():
        frames = model.decode(latents)
    check("㉒ 潜变量 → VAE 解码 ⇒ 帧张量形状正确（8 倍放大 ✓）",
          tuple(frames.shape) == (1, 3, 8, 64, 64), tuple(frames.shape))

    path = root / "pipeline.mp4"
    report = media_mod.write_video(frames, path, fps=24, value_range="-1..1")
    check("㉓ ⭐⭐ 端到端：潜变量 → 解码 → **真 mp4**，ffprobe 复核尺寸/帧数/时长 ✓",
          path.exists() and (report["width"], report["height"]) == (64, 64)
          and report["fps"] == 24.0 and report["requestedFrames"] == 8
          and report["durationSeconds"] is not None
          and abs(report["durationSeconds"] - 8 / 24) < 0.08,
          {k: report[k] for k in ("width", "height", "fps", "requestedFrames",
                                  "durationSeconds")})
    check("㉔ 产物路径可被外部拿到（同一份事实回给调用方 ✓ 便于落库/前端播放 ✓）",
          report["path"] == str(path) and report["bytes"] > 0, report["path"])


# ══════════════════════════════════════════════════════════════════════════
# ④ 走后端 + 管线：合成权重 → 装载 DiT → 挂 VAE → **真 mp4**（全链集成 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_backend_pipeline(root: Path) -> None:
    if not media_mod.have_ffmpeg() or not _have_torch():
        skip("缺 ffmpeg 或 torch ⇒ 后端整链跳过 ✓")
        return
    from app.services.engine import dit as dit_mod
    from app.services.engine import pipeline as pipe
    from app.services.engine import weights as weights_mod
    from app.services.engine.torch_backend import TorchBackend

    backend = TorchBackend()
    if not backend.describe()["available"]:
        skip("torch 不可用 ⇒ 后端整链跳过 ✓")
        return

    dit_config = dit_mod.DiTConfig(hidden=32, depth=2, heads=4, patch_size=(1, 2, 2),
                                   in_channels=4, text_dim=16, mlp_ratio=2.0, vae_scale=8)
    checkpoint = weights_mod.save_module_weights(
        dit_mod.build_dit(dit_config), root / "dit_synth.safetensors",
        metadata={"hidden": "32", "depth": "2", "heads": "4", "vae_scale": "8"})
    loaded = backend.load_weights(path=checkpoint, config=dit_config)
    check("㉕ 装载合成权重 ⇒ `modelLoaded` ✓（`denoise` 从占位切到**真前向** ✓）",
          backend.describe()["modelLoaded"] is True and loaded["complete"] is True, loaded)
    check("㉖ 挂参考 VAE ⇒ `vaeLoaded` ✓", bool(backend.attach_vae(VAE_CONFIG)["spatialScale"] == 8))

    request = pipe.GenerationRequest(prompt="雨夜霓虹街头", seed=4, steps=2, seconds=0.2,
                                     temporal_compression=1, outputs_dir=str(root))
    result = pipe.run_sync(request, backend)
    video = (result.outputs or {}).get("videoPath")
    check("㉗ ⭐⭐ 管线跑通并写出**真 mp4**（真 DiT 前向 + 真 VAE 解码 + 真 ffmpeg ✓）",
          result.ok is True and bool(video) and Path(video).exists()
          and Path(video).stat().st_size > 0, (result.error, video))
    probed = media_mod.probe(video) if video else {}
    check("㉘ ffprobe 复核这条管线产物（宽/高 = plan 的 1/8 × 8 = plan 尺寸 ✓、fps = plan ✓）",
          bool(probed) and probed["width"] == result.plan.width
          and probed["height"] == result.plan.height and probed["fps"] == float(result.plan.fps),
          {k: probed.get(k) for k in ("width", "height", "fps")} if probed else None)
    check("㉙ 仍然如实标 `synthetic=True` ✓（**未训练**的 VAE ⇒ 画面不是真结果 ✗ —— "
          "不许因为「写得出 mp4」就翻成 False ✗）",
          result.synthetic is True and result.outputs.get("synthetic") is True, result.synthetic)
    check("㉚ 但 `realFile=True` ✓（文件确实是真的 ✓ —— 两个标志各说各的事 ✓）",
          result.outputs.get("realFile") is True, result.outputs)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 首帧条件：图片 → **真 VAE 编码**（把条件路径最后一个占位换掉 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_first_frame(root: Path) -> None:
    if not _have_torch():
        skip("torch 未安装 ⇒ 首帧条件跳过 ✓")
        return
    import torch  # noqa: PLC0415

    from app.services.engine import dit as dit_mod
    from app.services.engine import pipeline as pipe
    from app.services.engine import weights as weights_mod
    from app.services.engine.torch_backend import TorchBackend

    # ① 图片 I/O：写一张真 PNG 再读回来 ✓
    frames = torch.rand(1, 3, 2, 32, 32) * 2.0 - 1.0
    written = media_mod.write_image(frames, root / "frame.png", index=1)
    check("㉛ 帧张量 → **真 PNG**（文件存在且尺寸正确 ✓）",
          Path(written["path"]).exists() and (written["width"], written["height"]) == (32, 32),
          written)
    tensor, info = media_mod.load_image_tensor(written["path"], width=16, height=16)
    check("㉜ 图片 → 张量：形状 ``(1,3,1,H,W)``（**单帧** ✓ 首帧条件要的就是它 ✓）",
          tuple(tensor.shape) == (1, 3, 1, 16, 16), tuple(tensor.shape))
    check("㉝ 值域口径与 `write_video` 一致（默认 -1..1 ✓）且**如实回报被拉伸** ✓",
          float(tensor.min()) >= -1.0 - 1e-6 and float(tensor.max()) <= 1.0 + 1e-6
          and info["resized"] is True and info["originalWidth"] == 32, info)
    check("㉞ 图片不存在 ⇒ **可行动报错** ✓", "不存在" in str(_media_error(root / "nope.png")))

    backend = TorchBackend()
    if not backend.describe()["available"]:
        skip("torch 不可用 ⇒ 首帧条件集成跳过 ✓")
        return

    dit_config = dit_mod.DiTConfig(hidden=32, depth=2, heads=4, patch_size=(1, 2, 2),
                                   in_channels=4, text_dim=32, mlp_ratio=2.0, vae_scale=8)
    checkpoint = weights_mod.save_module_weights(
        dit_mod.build_dit(dit_config), root / "dit_ff.safetensors",
        metadata={"hidden": "32", "depth": "2", "heads": "4", "vae_scale": "8"})
    backend.load_weights(path=checkpoint, config=dit_config)
    backend.attach_vae(VAE_CONFIG)

    torch.manual_seed(9)
    latents = torch.randn(1, 4, 3, 4, 4)        # (B,C,T,h,w) ✓ T=3 ✓
    before = latents.clone()
    mask = [1.0, 0.0, 0.0]
    blended = backend.condition_first_frame(latents, written["path"], mask, pipe.build_plan(
        pipe.GenerationRequest(prompt="x", steps=2)), pipe.GenerationRequest(prompt="x", steps=2))

    expected = backend._vae.encode(                      # noqa: SLF001 —— 自检直接比真值 ✓
        media_mod.load_image_tensor(written["path"], width=32, height=32)[0]
        .to(dtype=latents.dtype))
    check("㉟ ⭐⭐ 首帧真的走了 **VAE 编码**：第 0 个潜帧 == 图片编码结果（逐位 ✓）",
          bool(torch.allclose(blended[:, :, :1], expected, atol=1e-5)),
          float((blended[:, :, :1] - expected).abs().max()))
    check("㊱ ⭐ 掩码为 0 的潜帧**完全没被动**（不是整段都被首帧污染 ✓）",
          bool(torch.equal(blended[:, :, 1:], before[:, :, 1:])), "")
    check("㊲ 后端如实自述走的是哪条路 ✓（`mode=vae-encode` ✓ 而不是含糊不提 ✗）",
          (backend.conditioningNote or {}).get("mode") == "vae-encode",
          backend.conditioningNote)

    result = pipe.run_sync(pipe.GenerationRequest(
        prompt="雨夜霓虹街头", seed=5, steps=2, seconds=0.2, temporal_compression=1,
        first_frame=written["path"], outputs_dir=str(root)), backend)
    note = (result.conditioning or {}).get("backendNote") or {}
    check("㊳ ⭐ 经**管线**也如实带出后端自述（`conditioning.backendNote.mode` ✓ 前端可显示 ✓）",
          note.get("mode") == "vae-encode", note)
    check("㊴ 图生视频整链仍能出片 ✓（首帧混入 ⇒ 采样 ⇒ VAE 解码 ⇒ 真 mp4 ✓）",
          result.ok is True and bool((result.outputs or {}).get("videoPath")), result.error)


def _media_error(path: Path) -> Exception | None:
    try:
        media_mod.load_image_tensor(path, width=8, height=8)
    except media_mod.MediaError as err:
        return err
    return None


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="engine_io_"))
    case_vae(root)
    case_media(root)
    case_end_to_end(root)
    case_backend_pipeline(root)
    case_first_frame(root)

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
