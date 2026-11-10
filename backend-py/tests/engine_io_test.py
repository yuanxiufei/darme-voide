"""S7 自检：引擎的**解码与落盘**（真 VAE ✓ + 真 mp4/wav ✓ 2026-09-17）。

这套的判据特别强调**独立复核** ✓：写完 mp4 **不是**"函数没报错就算过" ✗ ——
而是用 **ffprobe** 把文件读回来，核对**宽/高/fps/帧数/时长** ✓✓；wav 用标准库 `wave` 读回来核对 ✓。

⚠️ 画面内容是**未经训练的参考 VAE** 的输出 ✓ ⇒ 看起来是噪声 ✓（**验的是管道** ✓，不是"H3 能出片" ✗）。

⚠️ **设备口径**（2026-09-25 在 A5000 上实测后加的 ✓；⭐ 同日晚**已收口** ✓ 见 `case_device_guard` ✓）：
这套的 latents / mask 是**测试里造的 CPU 张量** ✓，而 `TorchBackend()` 默认**自动探测设备** ✓ ⇒
在有卡的机器上会选 `cuda` ✓✗ —— 两者一混用就当场炸 ✓，而且报的是 `aten::slow_conv3d_forward` /
`Expected all tensors to be on the same device` 这类**指不到原因**的文案 ✗
（**在没卡的机器上一切正常** ✓✗ ⇒ 这台机器上才现形 ✓）。
⇒ ① 这套**显式钉 CPU** ✓（验的是 IO / 接缝的数学 ✓ 与设备无关 ✓）；**真 CUDA 的整条管线**由
`engine_dual_stream_test` 覆盖 ✓（它在 A5000 上真跑 ✓）。
⇒ ② 产品侧同时收口 ✓：五个吃调用方张量的入口（`denoise` / `condition_first_frame` / `sample_dual` /
`refine_latents` / `decode` ✓）现在会**当场明确报错** ✓（说清**哪个参数**、**哪个设备**、**该怎么办** ✓），
而**刻意不替调用方搬张量** ✗（搬 = 拷一份 ⇒ 破坏 `condition_first_frame` 的"就地改写"语义 ✗✗）——
逐入口清点表见 `TorchBackend._require_own_device` ✓；`write` 那条**查过后判定不校验** ✓
（落盘边界自己 `.to("cpu")` ✓）。这套的 `case_device_guard` 把上面每一条都钉成了判据 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_io_test.py
"""
from __future__ import annotations

import os
import subprocess
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
# ①′ 潜变量归一化：**H3 真权重必须开** ✓（第 123 步交付；落盘自检在此补齐 ✓）
#   ⚠️ 编号口径（2026-11-10 统一 ✓）：本段用 ``㊵′–㊽′`` **带撇** ✓ —— 圈码 ㊵–㊿ 已归
#      ⑦ 设备口径那一段用 ✓✗，两段各起一套同号会让汇总里分不清哪条红 ✗（``main()`` 里
#      有「编号唯一」的夹具自检兜着 ✓）；带撇＝「同批补进来的旁支」✓，与 ``㉑′`` 同惯例 ✓。
# ══════════════════════════════════════════════════════════════════════════
def case_latent_stats(root: Path) -> None:
    """⚠️ 那组 **mean/std 猜不出来** ✗ —— 漏了它，画面「看着还行、数值全错」✗✗ 而且**不报错** ✗。

    ⇒ 这里钉的是「**真的在用那组数**」✓（不是"函数没抛异常"✗）：
    逐通道**手算对照** ✓、往返恒等 ✓、常量不动点 ✓、**改数必变**（反套套逻辑 ✓）、数不对必报错 ✓。
    """
    facts = vae_mod.H3_VIDEO_VAE_FACTS
    width = len(facts["latentsMean"])
    check("㊵′ 归一化统计数与 `dit.H3_SHAPE_FACTS` 的潜通道**同源** ✓（24 ✓，不是各写各的 ✗）",
          width == len(facts["latentsStd"]) == 24, (width, len(facts["latentsStd"])))

    off = vae_mod.VideoVAEConfig(base_channels=8, latent_channels=4,
                                 channel_multipliers=(1, 1, 1, 1))
    on = vae_mod.VideoVAEConfig(base_channels=8, latent_channels=width,
                                channel_multipliers=(1, 1, 1, 1), apply_latent_stats=True)
    if not _have_torch():
        skip("torch 未安装 ⇒ 潜变量归一化用例跳过 ✓")
        return
    import torch  # noqa: PLC0415

    sample = torch.randn(1, 4, 2, 2, 2)
    model_off = vae_mod.build_vae(off)
    check("㊶′ 未开归一化 ⇒ `normalize_latents` **原样返回**（`is` 同一对象 ✓）",
          model_off.normalize_latents(sample) is sample)
    check("㊷′ 未开 ⇒ `denormalize_latents` 也原样 ✓（两个方向都透明 ✓）",
          model_off.denormalize_latents(sample) is sample)

    model = vae_mod.build_vae(on)
    latents = torch.randn(1, width, 2, 2, 2)
    view = (1, width, 1, 1, 1)
    mean = torch.tensor(facts["latentsMean"]).reshape(view)
    std = torch.tensor(facts["latentsStd"]).reshape(view)
    normalized = model.normalize_latents(latents)
    check("㊸′ ⭐ 归一化 = 逐通道 `(x−mean)/std` ✓（与 facts 里那组数**手算对照** ✓ —— "
          "空转或硬编码都过不了 ✓）",
          bool(torch.allclose(normalized, (latents - mean) / std, atol=1e-5)),
          float((normalized - (latents - mean) / std).abs().max()))
    check("㊹′ ⭐ 往返恒等：`denormalize(normalize(x)) == x` ✓（最强那类不变量 ✓）",
          bool(torch.allclose(model.denormalize_latents(normalized), latents, atol=1e-5)))

    at_mean = mean.repeat(1, 1, 2, 2, 2)
    at_mean_plus = (mean + std).repeat(1, 1, 2, 2, 2)
    check("㊺′ 常量不动点：`x = mean ⇒ 0` ✓、`x = mean + std ⇒ 1` ✓（两个点都能自己算出来 ✓）",
          bool(torch.allclose(model.normalize_latents(at_mean),
                              torch.zeros_like(at_mean), atol=1e-5))
          and bool(torch.allclose(model.normalize_latents(at_mean_plus),
                                  torch.ones_like(at_mean_plus), atol=1e-5)))

    original = list(facts["latentsStd"])
    try:
        facts["latentsStd"] = [value * 2.0 for value in original]
        changed = model.normalize_latents(latents)
    finally:
        facts["latentsStd"] = original
    check("㊻′ ⭐⭐ 反套套逻辑：把 std 翻倍 ⇒ 归一化结果**必须变** ✓"
          "（不变 ⇒ 那组数根本没被读进去 ✗✗ —— 「条件没接进去」的假绿 ✓）",
          not bool(torch.allclose(changed, normalized, atol=1e-6)))
    check("㊼′ 恢复后与改前**逐位相同** ✓（自检不留全局副作用 ✓）",
          bool(torch.allclose(model.normalize_latents(latents), normalized, atol=1e-7)))

    mismatch = vae_mod.VideoVAEConfig(base_channels=8, latent_channels=4,
                                      channel_multipliers=(1, 1, 1, 1), apply_latent_stats=True)
    try:
        vae_mod.build_vae(mismatch).normalize_latents(torch.randn(1, 4, 2, 2, 2))
        mismatch_ok = False
    except vae_mod.VAEError as err:
        mismatch_ok = "24" in str(err) and "4" in str(err)
    check("㊽′ 潜通道与统计数不符 ⇒ **明确报错** ✓（4 vs 24 ✓ —— 不猜、不截断 ✗："
          "凑一组「差不多」的只会得到看着对的错图 ✗✗）", mismatch_ok)


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

    #: ⭐⭐ 「落盘帧数到底核没核」的判定**只有一处** ✓✗（``media.frame_verdict`` ✓，2026-11-08 补 ✗）：
    #:   ``torch_backend.write`` 与 ``chain.run_chain`` 都读它 ✓ —— 此前各写一半 ✓（单段路核 ✓、
    #:   成片路一次都没核 ✗），最典型的下场是一处改了另一处不改 ✓。
    check("⑭¹ ⭐ ``write_video`` 的回执**自带**帧数判定 ✓（``nb_frames`` 直读 ⇒ "
          "``framesVerified == 'nb_frames'`` ✓、容差 0 ✓、``framesMatch is True`` ✓）"
          "—— 调用方不必各自重算 ✓",
          report.get("framesVerified") == "nb_frames"
          and report.get("framesTolerance") == 0
          and report.get("framesMatch") is True
          and int(report.get("encodedFrames") or -1) == 6,
          {k: report.get(k) for k in ("framesVerified", "framesTolerance",
                                      "framesMatch", "encodedFrames")})

    direct = media_mod.frame_verdict({"requestedFrames": 24, "nbFrames": 24,
                                      "durationSeconds": 1.0, "fps": 24})
    off_by_one = media_mod.frame_verdict({"requestedFrames": 24, "nbFrames": 23,
                                          "durationSeconds": 1.0, "fps": 24})
    check("⑭² 直读这一档**容差是 0** ✓（差一帧就 ``framesMatch: False`` ✗ —— 直读没有舍入误差 ✓）",
          direct["framesMatch"] is True and off_by_one["framesMatch"] is False,
          (direct, off_by_one))

    guessed = media_mod.frame_verdict({"requestedFrames": 24, "nbFrames": None,
                                       "durationSeconds": 1.0, "fps": 24})
    guessed_bad = media_mod.frame_verdict({"requestedFrames": 24, "nbFrames": None,
                                           "durationSeconds": 0.5, "fps": 24})
    check("⑭³ ⭐ ``nbFrames`` 缺失 ⇒ 按 ``时长×帧率`` **估** ✓ 且容差 ±1 ✓"
          "（1.0s×24 = 24 ⇒ 对上 ✓；0.5s×24 = 12 ⇒ 差 12 帧仍然 False ✗ —— "
          "容差不是「随便放过」✗）",
          guessed["framesVerified"] == "duration×fps" and guessed["framesTolerance"] == 1
          and guessed["framesMatch"] is True and guessed["encodedFrames"] == 24
          and guessed_bad["framesMatch"] is False,
          (guessed, guessed_bad))

    blind = media_mod.frame_verdict({"requestedFrames": 24, "nbFrames": None,
                                     "durationSeconds": None, "fps": None})
    zero = media_mod.frame_verdict({"requestedFrames": 24, "nbFrames": None,
                                    "durationSeconds": 0, "fps": 24})
    check("⑭⁴ ⭐⭐ 两格都读不到 ⇒ ``framesVerified: None`` ＋ ``framesMatch: None`` ✓✗"
          "（**不是 True 也不是 False** ✓ —— ``None`` 表示「没有可比读数」✓，"
          "任何调用方把它读成「通过」都是同一个病根 ✗）；且估法**不猜 0** ✗"
          "（时长 0 ⇒ ``frames_from_duration`` 回 ``None`` ✓ 而不是 0 帧 ✗）",
          blind["framesVerified"] is None and blind["framesMatch"] is None
          and blind["encodedFrames"] is None
          and zero["framesVerified"] is None and zero["framesMatch"] is None,
          (blind, zero))

    #: ⭐⭐ 「帧率是不是**我请求的那个**」也是必须单独表态的一格 ✓✗（2026-11-10 补 ✗）：
    #:   ⚠️ 编号口径：⑭⁹/⑭¹⁰ **不按行序升序** ✓ —— 它们与 ⑭¹¹（元数据读回核）都是
    #:   `docs/feature-map.md` **按号引用**的锚点 ✓，所以按补入批次编号、不重排 ✓（重排会让文档
    #:   指错判据 ✗）。想看唯一性看 `main()` 里那条夹具自检 ✓，别看行序 ✓。
    #:   ``report["fps"]`` 是 ``probe`` 从 ``r_frame_rate`` 读回来的**盘上**读数 ✓，请求值是
    #:   ``write_video(fps=…)`` 参数 ✓ —— 两者此前**一格都没对过账** ✓✗，于是「按 24 fps 封好 ✓」
    #:   与「它按 12 fps 封了 ✗（画面慢一倍 ✓）」在回执里同形 ✓（每格都"有值、可复核" ✓）。
    check("⑭⁹ ⭐⭐ ``write_video`` 的回执**自带**帧率对账 ✓（``fpsVerified == 'exact'`` ✓、"
          "``fpsRequested == 24`` ✓、盘上 ``fps == 24.0`` ✓）—— 请求值与盘上读数**分开**交出去 ✓",
          report.get("fpsVerified") == "exact" and report.get("fpsRequested") == 24
          and report.get("fps") == 24.0 and not report.get("fpsDrift"),
          {k: report.get(k) for k in ("fps", "fpsRequested", "fpsVerified", "fpsDrift")})
    same_fps, same_note = media_mod.compare_fps(24, 24.0)
    drift_fps, drift_note = media_mod.compare_fps(24, 12.0)
    blind_fps, blind_note = media_mod.compare_fps(24, None)
    check("⑭¹⁰ ⭐⭐ 帧率对账**三档必须分得开** ✓✗：一致 ⇒ ``exact`` ✓；真被改 ⇒ ``drift`` ✗ ＋ "
          "一句人话说清**音画不同步** ✓；盘上读不到 ⇒ ``None`` ✗ —— **不许**塌成 ``exact`` ✗"
          "（那正是「读数缺失被当成读数说没问题」✗）",
          same_fps == "exact" and same_note is None
          and drift_fps == "drift" and bool(drift_note) and "音画" in str(drift_note)
          and blind_fps is None and blind_note is None,
          {"same": same_fps, "drift": drift_fps, "blind": blind_fps})

    #: ⭐⭐ 「音频时长」这一格的**来源**也必须能分辨 ✓✗（2026-11-09 补 ✗）：``probe`` 在流时长
    #:   缺失时会**静默**回落到容器时长 ✓（最长那条流 ＋ 封装余量 ✗），而
    #:   ``_probe_audio_seconds`` 会拿它当**源音轨长度**去和画面做截断对账 ⇒ 报一个**假的**
    #:   「音轨被截掉 0.045 秒」✗✗。实测：1.0 秒的 mp3 原样封进 mkv ⇒ 容器 1.045 ✓、
    #:   webm ⇒ 1.008 ✓，两种封装都**不给**流的 ``duration`` 也**不给** ``duration_ts`` ✗。
    tone = (torch.sin(torch.linspace(0.0, 880.0 * 6.283, 32000)) * 0.2)
    wav_path = root / "probe_src.wav"
    media_mod.write_wav(tone, wav_path, sample_rate=32000)
    mkv_path = root / "probe_src.mkv"
    subprocess.run([media_mod._tool("ffmpeg"), "-y", "-v", "error", "-i", str(wav_path),
                    "-c:a", "libmp3lame", str(mkv_path)],
                   capture_output=True, timeout=120)
    facts_wav = media_mod.probe(wav_path) if mkv_path.exists() else {}
    facts_mkv = media_mod.probe(mkv_path) if mkv_path.exists() else {}
    check("⑭⁵ ⭐⭐ 「音频时长**从哪来**」三档必须能分辨 ✓✗（``audioDurationSource``）："
          "纯 wav ⇒ ``'stream'`` ✓（流自己给的 ✓）/ 封进 **mkv** ⇒ ``'container'`` ✗"
          "（流不给时长 ⇒ 容器顶替 ✓，**带封装余量** ✗）/ 无音轨的 mp4 ⇒ ``None`` ✗。"
          "少了这一格 ⇒「源音轨 1.0 秒」与「容器 1.045 秒顶替」在回执里**同形** ✓",
          facts_wav.get("audioDurationSource") == "stream"
          and facts_mkv.get("audioDurationSource") == "container"
          and report.get("audioDurationSource") is None,
          {k: v.get(k) for k, v in (("wav", facts_wav), ("mkv", facts_mkv))} )
    check("⑭⁶ ⭐⭐ 拿**容器回落值**当源音轨长度 ⇒ 一律按**没读到**处理 ✓✗"
          "（``_probe_audio_seconds(mkv) is None`` ✓）—— 容器的 1.045 秒不是音轨的 1.0 秒 ✗，"
          "这一路宁可交回 ``None`` ✓，也不许拿它喊假的截断 ✗",
          facts_mkv.get("audioDurationSource") == "container"
          and media_mod._probe_audio_seconds(mkv_path) is None
          and media_mod._probe_audio_seconds(wav_path) is not None,
          {"mkvSource": facts_mkv.get("audioDurationSource"),
           "mkvSeconds": facts_mkv.get("audioDurationSeconds"),
           "wavSeconds": facts_wav.get("audioDurationSeconds"),
           "mkvAccepted": media_mod._probe_audio_seconds(mkv_path)})
    #: ⭐⭐ 但「ffprobe 读不到」**不等于**「只能认不知道」✗✗（2026-11-09 补 ✗）：
    #:   mkv / webm 不给流的 ``duration`` ✗，而**包级 span** 量出来**恰好就是容器声明**
    #:   （1.045 ✗ —— 数包这条路也是错的 ✓），只有**全解码**才对得上内容 ✓。
    decoded = media_mod._decode_audio_seconds(mkv_path)
    length, how, miss = media_mod._audio_length(mkv_path)
    declared = float(facts_mkv.get("audioDurationSeconds") or 0)     # 容器声明（带余量 ✗）
    #: ⚠️ 「解码值」与「容器声明」只差 0.02~0.045 秒 ✓✗ ⇒ 光判「≈1.0 秒」分不开谁是谁 ✗
    #:   （反验二证实：让解码那支偷懒返回容器值，这条仍然会绿 ✗）。
    #:   ⇒ 前提必须锚在**独立事实**上：源内容是**我们造出来的 1.0 秒** ✓，不是被测函数的返回值 ✗ ——
    #:   拿 ``|declared − decoded|`` 当前提是**自洽陷阱** ✗✗：解码那支偷懒返回 declared 时，
    #:   这个差会算成 0、条件判直接放行 ✓，判据与被测对象**共享同一个错误源** ✓。
    gap = abs(declared - 1.0)
    check("⑭⁷ ⭐⭐ mkv 源必须走**全解码**兜底拿到真时长 ✓✗（不许停在「不知道」✗、"
          "更不许拿容器的 1.045 顶替 ✗）：``_audio_length(mkv)`` ⇒ 来源 ``'decoded'`` ✓、"
          "数值 ≈1.0 秒 ✓、且在两者明显不同时**取解码值**而非容器声明值 ✓",
          how == "decoded"
          and length is not None and abs(length - 1.0) < 0.05
          and facts_mkv.get("audioDurationSource") == "container"
          and (gap <= 0.01 or abs(length - declared) > 0.01),
          {"how": how, "decoded": decoded, "length": length,
           "container": facts_mkv.get("audioDurationSeconds"), "gap": round(gap, 4)})

    #: ⭐⭐ 「不知道」还得分两句说 ✓✗（2026-11-09 补 ✗）：源长到不值得全解码时，``_audio_length``
    #:   返 ```(None, None, "too_long")`` ✓ —— 那是**这次没量** ✓，不是"读不出来" ✗。
    #:   此前两者都塌成 ``(None, None)`` ⇒ 回执只能说"读不出来" ✗，用户照着去查编码/换封装 ✓，
    #:   而真正的原因是"上限不让解" ✓✗ —— 又一次「读数缺失被当成读数说没有」的变体 ✓。
    #:   ⚠️ 判据用 ``max_decode_seconds`` 注入一个不可能达到的上限 ✓（0.001 秒），不真造 2 小时文件 ✓；
    #:   同一文件在**默认上限**下必须仍是 ``"decoded"`` ✓（对照组 ⇒ 证明差别只来自闸门 ✓）。
    capped, capped_how, capped_miss = media_mod._audio_length(mkv_path, max_decode_seconds=0.001)
    check("⑭⁸ ⭐⭐ 源超全解码上限 ⇒ 交的是 **``too_long``**（这次没量 ✓）而不是 ``unreadable``"
          "（读不出来 ✗）：``_audio_length(mkv, max_decode_seconds=0.001)`` ⇒ "
          "``(None, None, 'too_long')`` ✓；同一文件默认上限下仍是 ``'decoded'`` ✓（只差闸门 ✓）",
          capped is None and capped_how is None and capped_miss == "too_long"
          and how == "decoded" and miss is None,
          {"capped": capped, "cappedHow": capped_how, "cappedMiss": capped_miss,
           "defaultHow": how, "defaultMiss": miss})

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

    #: ⭐⭐ 「这段音频**有没有声音**」也必须是回执里的一格 ✓✗（2026-11-09 补 ✗）：``write_image``
    #:   早有 ``blank_std`` 抓纯色图 ✓，音频侧此前只有 ``bytes``/``durationSeconds`` ✓ ⇒
    #:   **全零波形**与一段正常音频在回执里**完全同形** ✓✗（wav 照写 ✓、混进 mp4 也照成 ✓、
    #:   回执照说「真 wav ✓」✓）⇒ 用户点开是一段无声片，而没有任何一格提示过 ✗
    #:   （"没量振幅"就等于宣布"音频没问题"✗ —— 同一族病根 ✓）。
    #:   这条同时钉四件 ✓：全零被抓到 ✓、**0 帧**边界不崩 ✓、正常音频**不许**被误判 ✗、
    #:   ``peakDbfs`` 在全零时是 ``None`` 而**不是** ``-inf`` ✓✗（``-inf`` 序列化成
    #:   ``-Infinity`` ⇒ **非法 JSON** ✗ ⇒ 前端 ``JSON.parse`` 直接炸 ✗）。
    silent_report = media_mod.write_wav(torch.zeros(2, 8000), root / "silent.wav", sample_rate=8000)
    empty_report = media_mod.write_wav(torch.zeros(0), root / "empty.wav", sample_rate=8000)
    check("㉑′ ⭐⭐ 全零波形 ⇒ ``silent is True`` ✓（``peak == 0.0`` ✓、``peakDbfs is None`` ✓ "
          "**不是** ``-inf`` ✗）；0 帧也不崩 ✓；正常 0.8 正弦**不许**被判静音 ✗",
          silent_report["silent"] is True and silent_report["peak"] == 0.0
          and silent_report["peakDbfs"] is None
          and empty_report["silent"] is True and empty_report["samples"] == 0
          and wav_report["silent"] is False and wav_report["peakDbfs"] is not None,
          {"silent": silent_report, "empty": empty_report,
           "normal": {"silent": wav_report["silent"], "peakDbfs": wav_report["peakDbfs"]}})

    #: ⭐⭐ 「元数据**真写进 PNG 了**」必须**从盘上读回来核** ✗✗（2026-11-10 补 ✗）：
    #:   ``write_image`` 原先交的 ``metadataKeys`` 是**送出去的键名** ✓✗ —— 而 ``_png_info`` 会
    #:   **跳过** ``value is None`` 的项 ✓、键名 strip 后为空也跳过 ✓ ⇒ 「打算写 3 个」被当成
    #:   「写成了 3 个」✓，甚至交出一个**根本不在文件里**的键名 ✗（本族"读数缺失被当成读数"
    #:   换了个位置：这里的"读数"是**我们自己打算写的清单** ✓）。
    #:   mp4 那侧 2026-11-07 就有 ``metadataDropped`` 对账 ✓，图这侧一直没有 ✓。
    meta_ok = media_mod.write_image(torch.rand(1, 3, 8, 8), root / "meta_ok.png",
                                    metadata={"prompt": "a test", "steps": 8})
    meta_bad = media_mod.write_image(torch.rand(1, 3, 8, 8), root / "meta_bad.png",
                                     metadata={"prompt": "a test", "dropped": None, " ": "x"})
    meta_none = media_mod.write_image(torch.rand(1, 3, 8, 8), root / "meta_none.png")
    check("⑭¹¹ ⭐⭐ 元数据要**读回来**才算数 ✓✗：送了 prompt+steps ⇒ ``metadataKeys`` 两键都在 ✓、"
          "``metadataMissing`` 空 ✓、``metadataReadback is True`` ✓；``value is None`` 与**空白键名** ⇒ "
          "必须落进 ``metadataMissing`` ✗（**不许**出现在 ``metadataKeys`` 里 ✗）；"
          "没送 metadata ⇒ ``metadataReadback is None`` ✓（**不是** True ✗ —— 没送过 ≠ 核过了）",
          meta_ok.get("metadataReadback") is True
          and meta_ok.get("metadataKeys") == ["prompt", "steps"]
          and meta_ok.get("metadataMissing") == []
          and meta_bad.get("metadataReadback") is True
          and meta_bad.get("metadataKeys") == ["prompt"]
          and sorted(meta_bad.get("metadataMissing") or []) == [" ", "dropped"]
          and meta_none.get("metadataReadback") is None
          and meta_none.get("metadataKeys") == [],
          {"ok": meta_ok, "bad": meta_bad, "none": meta_none})


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

    # ⚠️ **显式 CPU** ✓（2026-09-25 在 A5000 上实测）：`TorchBackend()` 默认**自动探测** ✓
    #    ⇒ 本机（有卡 ✓）会选 `cuda` ✓，而本用例的 latents / mask **是在测试里造的 CPU 张量** ✓
    #    ⇒ 混用当场炸 ✓✗，报的还是 `aten::slow_conv3d_forward` 这种**指不到原因**的文案 ✗（见测试头注 ✓）。
    #    这几个判据验的是**盘上 IO / 首帧条件化 / VAE 接缝的数学** ✓ ⇒ 设备无关 ✓ ⇒ 钉 CPU ✓。
    #    GPU 路径**不是没人测** ✓：`engine_dual_stream_test` 整条管线在真 CUDA 上跑 ✓（真机实测 ✓）。
    backend = TorchBackend(device="cpu")
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

    # ⚠️ 同前一处 ✓：本用例的 latents 是测试里造的 **CPU 张量** ✓ ⇒ 钉 CPU ✓（别让"本机有卡"改判据 ✓✗）。
    backend = TorchBackend(device="cpu")
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


# ══════════════════════════════════════════════════════════════════════════
# ⑦ 设备口径：调用方给的张量不在后端设备上 ⇒ **可行动报错**（2026-09-25 收 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_device_guard(root: Path) -> None:
    """⭐⭐ 设备口径**逐个入口**收一遍 ✓✗（2026-09-25 ✓ —— 「全」= 连**不校验的**入口也写明结论 ✓）。

    ⚠️⚠️ 本仓**不替调用方搬张量** ✗（这是**有意留**的语义 ✓，不是没做完 ✗）：唯一的搬法是拷一份 ✓
    ⇒ 调用方手里那个张量**不再被就地改写** ✗ —— 而 `condition_first_frame` 的语义正是
    「按掩码混进第 0 个潜帧、**回新的** latents」✓；悄悄改其中一半就是**静默换语义** ✗✗。
    再者：搬走要多占一份显存 ✓（真权重下潜变量几百 MB ✓），而「该在哪个设备上」只有调用方知道 ✓。

    ⚠️ 这套**不需要显卡** ✓：把后端的 `_device` 改成 ``"cuda"`` ✓、张量留在 CPU ✓ ⇒ 错配就造出来了 ✓；
    **有卡那台**上这些判据**一模一样地成立** ✓（``cuda:0`` ≠ ``cpu`` 更不等 ✓）。真正只有有卡才验得了的
    是「``"cuda"`` ⇒ ``cuda:0`` 的具体化」✓ —— 那条**两支都写成判据** ✓（哪台机器都有一条绿 ✓，不 skip ✓）。

    ⚠️ 真链路（管线 ✓）**一个校验都碰不到** ✓：它拿到的每个张量都是本后端自己造的 ✓
    —— 上面 `case_first_frame` 里那条 `pipe.run_sync` 就是活证 ✓。
    """
    if not _have_torch():
        skip("torch 未安装 ⇒ 设备口径跳过 ✓")
        return
    import torch  # noqa: PLC0415

    from app.services.engine import pipeline as pipe
    from app.services.engine.torch_backend import TorchBackend, TorchBackendUnavailable

    backend = TorchBackend(device="cpu")
    if not backend.describe()["available"]:
        skip("torch 不可用 ⇒ 设备口径跳过 ✓")
        return

    plan = pipe.build_plan(pipe.GenerationRequest(prompt="x", steps=2))
    request = pipe.GenerationRequest(prompt="x", steps=2)
    latents = torch.randn(1, 4, 3, 4, 4)                     # (B,C,T,h,w) ✓ CPU ✓
    condition = torch.randn(7, 5)
    dual = {"video": torch.randn(24, 3, 4, 4), "audio": torch.randn(2, 5)}
    before = latents.clone()
    # ⚠️ 图片要**真写一张** ✓：设备校验在最前面 ✓ 现在拦得住 ✓ —— 但万一将来校验顺序被挪到读图之后，
    #    这条判据就会变成"图片不存在"⇒ **判据变味** ✗（那种假绿最难查 ✓）⇒ 干脆给真的 ✓。
    image = media_mod.write_image(torch.rand(1, 3, 2, 32, 32) * 2.0 - 1.0,
                                  root / "guard.png", index=1)["path"]

    def _is_device_error(err: BaseException) -> bool:
        """是不是**这条**设备错 ✓ —— 认它独有的标记 ✓（比比对整段文案稳 ✓）。"""
        return isinstance(err, TorchBackendUnavailable) and "混用必炸" in str(err)

    calls = {
        "denoise": lambda: backend.denoise(latents, 0.5, condition, request),
        "condition_first_frame": lambda: backend.condition_first_frame(
            latents, image, [1.0, 0.0, 0.0], plan, request),
        "sample_dual": lambda: backend.sample_dual(dual, plan.sigmas, condition, request),
        "refine_latents": lambda: backend.refine_latents(dual, plan, request, condition=condition),
        "decode": lambda: backend.decode(latents, plan, request),
    }

    # ── ① 错设备：**五个入口逐个**都要拦下来 ✓（伪装成 cuda 是造错配最省的办法 ✓）────────
    backend._device = "cuda"                                 # noqa: SLF001 —— 有意伪装 ✓
    errors: list[BaseException | None] = []
    for call in calls.values():
        try:
            call()
            errors.append(None)
        except BaseException as err:                         # noqa: BLE001 —— 要的就是"它报了什么" ✓
            errors.append(err)
    for number, (name, err) in zip("㊵㊶㊷㊸㊹", zip(calls, errors)):
        check(f"{number} {name}：调用方给 **CPU 张量** ⇒ **明确报错** ✓（不是 `aten::slow_conv3d_forward` "
              f"那串指不到原因的 ✗）、类型 `TorchBackendUnavailable` ✓",
              _is_device_error(err) if err is not None else False,
              f"{type(err).__name__}: {str(err)[:70]}" if err is not None else "没报错（静默跑了 ✗）")

    # ── ② 文案**指得到原因** ✓（这是"点破"的全部价值 ✓）──────────────────────────────────
    text = str(errors[0] or "")
    check("㊺ 报错**指得到原因** ✓：含**参数名**（`latents` ✓）+ **两个设备**（`cpu` 与 `cuda` ✓）"
          "+ 「不替调用方搬张量」与 ``.to(…)`` 的**行动指引** ✓",
          all(mark in text for mark in ("latents", "cpu", "cuda", "不替调用方搬", ".to(")), text[:140])

    # ── ③ **真的没搬** ✓（搬了就破坏"就地改写"语义 ✗✗）──────────────────────────────────
    check("㊻ **真没搬**：报错之后原张量仍在 CPU ✓、且**一字未改** ✓（搬 = 拷一份 ⇒ 调用方那份就"
          "不再被就地改写了 ✗✗，而首帧条件正是靠就地改写语义 ✓✗）",
          str(latents.device) == "cpu" and bool(torch.equal(latents, before)), str(latents.device))

    backend._device = "cpu"                                  # noqa: SLF001 —— 装回真设备 ✓

    # ── ④ 只认**带 `device` 的对象** ✓（干跑后端 / `plan.sigmas` 靠这条不受影响 ✓）────────
    class _NoDevice:                                         # 干跑后端的 TinyTensor 就是这形态 ✓
        pass

    try:
        backend._require_own_device("自检", a=None, b=0.5, c="static/x.png",   # noqa: SLF001
                                    d=[1.0, 2.0], e={"k": 3}, f=_NoDevice())
        relaxed = True
    except BaseException as err:                             # noqa: BLE001
        relaxed = False
        text = f"{type(err).__name__}: {err}"
    check("㊼ 只认带 `device` 的对象 ✓：``None`` / 数字 / 字符串 / float 列表 / 无 `device` 的假张量"
          "**一律放行** ✓（宁可放过 ✓，绝不许误伤真链路 ✗）", relaxed, "" if relaxed else text)

    # ── ⑤ ⚠️ **不许比字符串** ✗✗（初版就是这么错的 ✓ —— 把真链路误判成混用 ✗）────────────
    backend._device = "cuda"                                 # noqa: SLF001
    fake_cuda = backend._device_torch()                      # noqa: SLF001
    backend._device = "cpu"                                  # noqa: SLF001
    check("㊽ ⚠️ ``torch.device('cuda') != torch.device('cuda:0')`` ✓ 而**两者是同一设备** ✓ ⇒ "
          "判据必须比**规范化后的具名设备** ✓（比字符串就是 2026-09-25 那个把 `engine_pipeline_test` "
          "真链路误判成混用的错 ✗✗）+ CPU 规范化后仍是 `cpu` ✓",
          torch.device("cuda") != torch.device("cuda:0")
          and backend._device_torch() == torch.device("cpu")   # noqa: SLF001
          and fake_cuda.type == "cuda", f"{fake_cuda!r}")

    # ── ⑥ ⭐⭐ **不许误伤**：同设备张量在五个入口上都要照旧能用 ✓（真链路就是这么用的 ✓）────
    hurt: list[str] = []
    for name, call in calls.items():
        try:
            call()
        except BaseException as err:                         # noqa: BLE001 —— 别的错都不算这条红 ✓
            if _is_device_error(err):
                hurt.append(name)
    check("㊾ ⭐⭐ **不许误伤**：同设备（CPU）张量在**五个入口**上都不报设备错 ✓（别的错照旧允许 ✓"
          " —— 没装模型/放大器没就绪本来就该报 ✓）；⚠️ 这正是把真链路判错过的那个坑 ✗",
          not hurt, hurt)

    # ── ⑦ 具体化：``"cuda"`` ⇒ ``cuda:0`` ✓（**两支都判** ✓ —— 哪台机器都不 skip ✓）────────
    named = TorchBackend(device="cuda")._device_torch()      # noqa: SLF001
    if torch.cuda.is_available():
        check("㊿ ⭐ 有卡时 ``\"cuda\"`` **具体化**成 ``cuda:0`` ✓（张量的 `.device` 永远带卡号 ✓ ⇒ "
              "不比字符串就对了 ✓）；⚠️ 多卡时 ``cuda:1`` 与 ``cuda:0`` 是**真不同** ✓ 照旧报错 ✓",
              named == torch.device("cuda:0"), repr(named))
    else:
        check("㊿′ 无卡时 ``\"cuda\"`` 保持 ``device('cuda')`` ✓（此时真张量也上不去 cuda ✓ ⇒ 仍然照实报错 ✓）",
              named == torch.device("cuda"), repr(named))


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="engine_io_"))
    case_vae(root)
    case_latent_stats(root)
    case_media(root)
    case_end_to_end(root)
    case_backend_pipeline(root)
    case_first_frame(root)
    case_device_guard(root)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    #: ⚠️ **判据编号在本文件内必须唯一** ✓✗（2026-11-10 立 ✓）：本套口径是「编号 + 空格 + 说明」✓，
    #:   而圈码只有 ①–㊿ 五十个 ✓ —— 两段各起一套同号（`case_latent_stats` 曾与 `case_device_guard`
    #:   都用 ㊵–㊽ ✓）汇总里就**读不出哪条红** ✓✗，且这种撞号**不会自己报错** ✗。这里按**夹具自身的
    #:   错**处理 ✓（不进 SUMMARY 计数 ✓，它是编号问题不是被测对象的问题 ✓），退出码照样 1 ✓。
    #:   真撞了就给后一段加 ``′`` ✓（见 ①′ 段抬头 ✓），别改成两段共号 ✗。
    labels = [name.split(" ")[0] for name, _passed, _detail in _RESULTS]
    repeated = sorted({label for label in labels if labels.count(label) > 1})
    if repeated:
        print("FAIL  ⚠️ 判据编号重复：" + "、".join(repeated)
              + "（同号两条 ⇒ 汇总分不清哪条红 ✗；给后一段加 ``′`` ✓）")
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures or repeated else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
