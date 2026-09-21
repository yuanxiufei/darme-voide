"""S7 自检：**H3 双流（视频 + 音频）接进管线** ✓（2026-09-20 ✓）。

这一套钉的是「**真功能**」✓，不是"编排能跑"✗：真 H3 形态主干（**缩小版** ✓）+ 真视频 VAE +
真音频 VAE ⇒ 一次 `run_sync` 出 **真 mp4 + 真 wav** ✓，两条都用**外部工具 / 标准库**读回来核对 ✓
（ffprobe ✓ / `wave` ✓）。

⚠️ 缩小配置**只改尺寸** ✓ —— 走的还是 `load_weights` **判形态**那条真路径 ✓（与工作站上装真权重
走的是同一条 ✓）。所有模型都是**参考实现（未训练）** ✓ ⇒ 画面与声音是噪声 ✓（验的是管道 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_dual_stream_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import audio_vae as audio_vae_mod  # noqa: E402
from app.services.engine import guidance as guidance_mod  # noqa: E402
from app.services.engine import h3_form  # noqa: E402
from app.services.engine import media as media_mod  # noqa: E402
from app.services.engine import pipeline as pipe  # noqa: E402
from app.services.engine import vae as vae_mod  # noqa: E402
from app.services.engine import weights as weights_mod  # noqa: E402
from app.services.engine.dryrun import DryRunBackend  # noqa: E402
from app.services.engine.torch_backend import (  # noqa: E402
    TorchBackend, TorchBackendUnavailable)

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


def _raises(call: Any, needle: str) -> str | None:
    """**能触发**的反向证明 ✓：调它、看报错里有没有那个词 ✓（没报错 ⇒ 返回 None ⇒ 断言红 ✓）。"""
    try:
        call()
    except Exception as err:  # noqa: BLE001 —— 就是来看它报什么的 ✓
        return str(err) if needle in str(err) else None
    return None


# ⚠️ **缩小版** H3 主干 ✓（只改尺寸 ✓ —— 模块名 / 键名与真权重一致 ✓ ⇒ 形态判别照样认它 ✓）。
# 注意 `heads × head_dim = 64 ≠ hidden 32` ✓ **不是笔误** ✓：H3 本来就不等（56×128=7168 vs 5376 ✓）。
TRUNK: dict[str, Any] = {
    "hidden": 32, "layers": 2, "heads": 4, "head_dim": 16, "ffn": 64,
    "text_dim": 16, "latents_dim": 4, "audio_latents_dim": 8, "patch_size": (1, 2, 2),
    "time_input_dim": 16, "time_hidden": 32, "time_dim": 32, "inv_freq_len": 2,
    "refiner_layers": 1,
}
#: 视频 VAE：**5 层 ⇒ spatial_scale 16** ✓（与 H3 的 `vaeScale` 同口径 ✓ ⇒ 出片尺寸 = plan 尺寸 ✓）
VIDEO_VAE = vae_mod.VideoVAEConfig(base_channels=4, latent_channels=4,
                                   channel_multipliers=(1, 1, 1, 1, 1))
#: 音频 VAE：`latent_channels` **必须** = 主干的 `audio_latents_dim` ✓（对接不上解码端就报形状 ✓）；
#: ⚠️ `decoder_dim` 必须 ≥ `2^级数` = 128 ✓ —— 解码器每级**通道减半** ✓，给小了末级宽度就是 0 ✗
#: （2026-09-20 踩到过 ✓：报错是**一句 oneDNN 反卷积消息** ✗✗，指不到真因 ✓）
AUDIO_VAE = audio_vae_mod.AudioVAEConfig(latent_channels=8, encoder_dim=8,
                                         latent_dim=32, decoder_dim=128)
#: 请求：1:1 ⇒ 224×224 ✓（/16 = 14×14 ✓ 行数 245 ✓ CPU 秒级 ✓）；3 步 ✓
REQUEST: dict[str, Any] = {"prompt": "雨夜霓虹街头", "seed": 7, "steps": 3, "seconds": 0.2,
                           "ratio": "1:1", "megapixels": 0.05, "temporal_compression": 1}


def _h3_backend(root: Path, *, name: str = "h3_synth", mode: str | None = "round",
                with_audio: bool = True, with_text: bool = True) -> TorchBackend:
    """按**真路径**建后端 ✓：造 H3 形态权重 ⇒ `load_weights` 判形态 ⇒ 挂 TE / 两个 VAE ✓。"""
    backend = TorchBackend(audio_latent_mode=mode)
    checkpoint = weights_mod.save_module_weights(
        h3_form.H3FormTrunk(**TRUNK), root / f"{name}.safetensors")
    backend.load_weights(path=checkpoint, config=dict(TRUNK))
    if with_text:
        backend.attach_text_encoder()
    backend.attach_vae(VIDEO_VAE)
    if with_audio:
        backend.attach_audio_vae(AUDIO_VAE)
    return backend


# ══════════════════════════════════════════════════════════════════════════
# ① 能力自述：**缺什么就报什么** ✓（不许只回一个光秃秃的 False ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_gating() -> None:
    blank = TorchBackend()
    state = blank.dualStream
    check("① 什么都没装 ⇒ 双流不可用 ✓（`enabled=False` ✓）", state["enabled"] is False, state)
    check("② ⭐ 三条**逐条报** ✓（形态 ✗ + 音频 VAE ✗ + 取整口径 ✗ —— 都在 ✓）",
          len(state["blockedBy"]) == 3
          and any("H3 形态" in item for item in state["blockedBy"])
          and any("音频 VAE" in item for item in state["blockedBy"])
          and any("取整口径" in item for item in state["blockedBy"]), state["blockedBy"])
    check("③ 给了取整口径 ⇒ 「口径」那条**真的从清单里消失** ✓（清单是活的 ✓ 不是套话 ✓）",
          all("取整口径" not in item
              for item in TorchBackend(audio_latent_mode="ceil").dualStream["blockedBy"]))
    bad = TorchBackend(audio_latent_mode="nearest")
    check("④ 口径写错 ⇒ **明确报不认** ✓（不悄悄按某个默认跑 ✗）",
          any("不认" in item for item in bad.dualStream["blockedBy"]), bad.dualStream["blockedBy"])
    check("⑤ `latentMode` 分**四态**自述 ✓（没模型 ⇒ placeholder-1d ✓，不是含糊的布尔 ✓）",
          blank.describe()["latentMode"] == "placeholder-1d", blank.describe()["latentMode"])


# ══════════════════════════════════════════════════════════════════════════
# ② 双流初始噪声：形状 / 可复现 / 口径真的在用 / 两类拒绝
# ══════════════════════════════════════════════════════════════════════════
def case_dual_latents(root: Path) -> None:
    import torch  # noqa: PLC0415

    backend = _h3_backend(root, name="h3_dir", with_audio=False)
    check("⑥ 形态判别 ✓：H3 招牌键 ⇒ `latentMode=h3-trunk-only` ✓（按**键名**判 ✓ 不按文件名 ✗）",
          backend.describe()["latentMode"] == "h3-trunk-only", backend.describe()["dualStream"])
    check("⑦ H3 权重 + 没挂音频 VAE ⇒ 仍不可用 ✓（清单只剩那一条 ✓）",
          backend.dualStream["enabled"] is False
          and len(backend.dualStream["blockedBy"]) == 1, backend.dualStream["blockedBy"])

    backend = _h3_backend(root, name="h3_dir2")
    check("⑧ 挂齐（形态 ✓ + 音频 VAE ✓ + 口径 ✓）⇒ `enabled=True` ✓ + `h3-dual` ✓",
          backend.dualStream["enabled"] is True
          and backend.describe()["latentMode"] == "h3-dual", backend.dualStream)

    request = pipe.GenerationRequest(**REQUEST)
    plan = pipe.build_plan(request)
    check("⑨ plan 尺寸能被 **H3 的 vae_scale=16** 整除 ✓（分辨率吸附 32 网格的后果 ✓）",
          plan.width % 16 == 0 and plan.height % 16 == 0, (plan.width, plan.height))

    latents = backend.init_dual_latents(plan, request)
    check("⑩ 视频潜 = ``[C,T,H/16,W/16]`` ✓ **无 batch 维** ✓（与行级表示的约定一致 ✓）",
          tuple(latents["video"].shape) == (TRUNK["latents_dim"], int(plan.latent_frames),
                                            plan.height // 16, plan.width // 16),
          tuple(latents["video"].shape))
    check("⑪ 音频潜 = ``[audio_dim, 2, T_a]`` ✓（立体声在**中间**那维 ✓；32 是特征维 ✗ 不是声道 ✗）",
          tuple(latents["audio"].shape)[0] == TRUNK["audio_latents_dim"]
          and tuple(latents["audio"].shape)[1] == 2, tuple(latents["audio"].shape))
    check("⑫ 音频潜帧 = round(5/24 × 40) = **8** ✓（口径是**显式给的** ✓ 不是内置默认 ✗）",
          latents["audioLatentFrames"] == 8, latents["audioLatentFrames"])

    again = backend.init_dual_latents(plan, request)
    check("⑬ ⭐ 同 seed ⇒ 两条流**逐位可复现** ✓（两条共用一个 generator ✓）",
          bool(torch.equal(latents["video"], again["video"]))
          and bool(torch.equal(latents["audio"], again["audio"])))
    other = backend.init_dual_latents(plan, pipe.GenerationRequest(**{**REQUEST, "seed": 8}))
    check("⑭ 换 seed ⇒ **必变** ✓（否则就是「反正都随机」的假绿 ✓）",
          not bool(torch.equal(latents["video"], other["video"])))

    ceil_backend = TorchBackend(audio_latent_mode="ceil")
    ceil_backend.load_weights(path=root / "h3_dir2.safetensors", config=dict(TRUNK))
    ceil_backend.attach_audio_vae(AUDIO_VAE)
    ceil_latents = ceil_backend.init_dual_latents(plan, request)
    check("⑮ ⭐ 换口径 ⇒ 音频潜帧**必变** ✓（`ceil` ⇒ 9 ✓ —— 证明那个参数真在用 ✓ 不是摆设 ✗）",
          ceil_latents["audioLatentFrames"] == 9
          and ceil_latents["audioLatentFrames"] != latents["audioLatentFrames"],
          ceil_latents["audioLatentFrames"])

    no_compression = pipe.GenerationRequest(**{**REQUEST, "temporal_compression": None})
    check("⑯ 没给 `temporal_compression` ⇒ **明确报错** ✓（潜帧数不猜 ✓）",
          _raises(lambda: backend.init_dual_latents(pipe.build_plan(no_compression),
                                                    no_compression), "潜帧数") is not None)
    odd = pipe.GenerationPlan(width=200, height=200, frames=5, fps=24, latent_frames=5,
                              steps=3, sigmas=[1.0, 0.5, 0.0])
    check("⑰ 像素不能整除 16 ⇒ **明确报错** ✓（不悄悄取整 ✗ —— 那会改变产物却不报 ✓）",
          _raises(lambda: backend.init_dual_latents(odd, request), "整除") is not None)


# ══════════════════════════════════════════════════════════════════════════
# ③ 条件：H3 要的是**文本状态** `[L, text_dim]` ✓（不是条件对 ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_text_states(root: Path) -> None:
    backend = _h3_backend(root, name="h3_te")
    states = backend.encode_text(pipe.GenerationRequest(**REQUEST))
    check("⑱ H3 的条件**只有 positive** ✓（无 negative ⇒ 管线不会做 CFG ✓ 与「参考无 CFG」一致 ✓）",
          "negative" not in states, sorted(states))
    check("⑲ 文本状态形状 ``[L, text_dim]`` ✓（**无 batch 维** ✓ —— 与主干 forward 约定一致 ✓）",
          len(tuple(states["positive"].shape)) == 2
          and int(states["positive"].shape[1]) == TRUNK["text_dim"],
          tuple(states["positive"].shape))
    check("⑳ token 数如实回报 ✓（`L` 就是它 ✓ —— 不编一个不存在的序列长度 ✓）",
          int(states["positive"].shape[0]) == states["tokens"] and states["tokens"] > 0,
          (tuple(states["positive"].shape), states["tokens"]))

    bare = _h3_backend(root, name="h3_note", with_text=False)
    check("㉑ 没挂 TE ⇒ **明确报错** ✓（`L` 只有分词器知道 ✓ ⇒ 不许拿占位编一个 ✓）",
          _raises(lambda: bare.encode_text(pipe.GenerationRequest(**REQUEST)),
                  "文本编码器") is not None)


# ══════════════════════════════════════════════════════════════════════════
# ④ ⭐⭐ 整链：一次 `run_sync` ⇒ **真 mp4 + 真 wav**（本套的重点 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_pipeline(root: Path) -> None:
    request = pipe.GenerationRequest(**REQUEST, outputs_dir=str(root / "run"))
    result = pipe.run_sync(request, _h3_backend(root, name="h3_pipe"))
    check("㉒ ⭐⭐ 双流整链 `ok=True` ✓（真主干前向 + 两个真 VAE + 真落盘 ✓）",
          result.ok is True, (result.error, result.stageMs))
    if not result.ok:
        return

    outputs = result.outputs or {}
    video, audio = outputs.get("videoPath"), outputs.get("audioPath")
    check("㉓ ⭐ 真 mp4 与 真 wav **都在盘上** ✓（两个文件都非空 ✓）",
          bool(video) and Path(video).exists() and Path(video).stat().st_size > 0
          and bool(audio) and Path(audio).exists() and Path(audio).stat().st_size > 0,
          (video, audio))
    check("㉔ `outputs.dualStream=True` ✓（「本次真出了音频」的**铁证** ✓ —— 不是能力声明 ✗）",
          outputs.get("dualStream") is True, outputs.get("dualStream"))
    check("㉕ 产物清单里**两条都在** ✓（video + audio ✓ 各自可独立核对 ✓）",
          {item.get("kind") for item in outputs.get("artifacts") or []} == {"video", "audio"},
          [item.get("kind") for item in outputs.get("artifacts") or []])

    probed = media_mod.probe(video) if video else {}
    check("㉖ ⭐ ffprobe 复核：出片尺寸 = plan 尺寸 ✓（潜 × vae_scale 16 ✓ 与 plan 对得上 ✓）",
          bool(probed) and probed["width"] == result.plan.width
          and probed["height"] == result.plan.height
          and probed["fps"] == float(result.plan.fps),
          {key: probed.get(key) for key in ("width", "height", "fps")} if probed else None)

    with wave.open(str(audio), "rb") as handle:
        channels, rate, frames = (handle.getnchannels(), handle.getframerate(),
                                  handle.getnframes())
    reported = int((outputs.get("audio") or {}).get("samples") or -1)
    check("㉗ ⭐ 标准库读回 wav：立体声 2 ✓ / 32 kHz ✓ / 帧数与**写盘回执逐位相等** ✓"
          "（一个来自写入函数自报 ✓、一个来自独立读盘 ✓ ⇒ 这才叫「落盘忠实」✓）",
          (channels, rate) == (2, 32000) and frames == reported,
          (channels, rate, frames, reported))
    check("㉘ 音频时长**自己成立** ✓（≈0.19s ✓ —— 不假装与视频逐帧对齐 ✗）",
          abs(frames / rate - 0.2) < 0.02, frames / rate)
    check("㉘′ ⚠️ 别把「理想值 T×hop = 6400」当事实 ✗：实测 **6208** ✓ —— `ConvTranspose1d` 带 "
          "padding ✓ 每级少 1 个样本 ✓ ⇒ **结构使然** ✓（我第一版断言就是照理想值写的 ✗ ⇒ 当场红 ✓）",
          frames < 8 * AUDIO_VAE.hop_length and frames > 8 * AUDIO_VAE.hop_length * 0.9,
          (frames, 8 * AUDIO_VAE.hop_length))

    # ⭐ 反套套逻辑：换提示词 ⇒ 音频必须变（同一次前向出的两条流 ✓ 条件没接进去就当不了绿 ✓）
    changed = pipe.run_sync(pipe.GenerationRequest(**{**REQUEST, "prompt": "晴日海边",
                                                      "outputs_dir": str(root / "p2")}),
                            _h3_backend(root, name="h3_p2"))
    check("㉙ ⭐ 反套套逻辑：换提示词 ⇒ **wav 字节必变** ✓（相同 ⇒ 条件压根没接进去 ✗✗）",
          changed.ok is True
          and Path(changed.outputs["audioPath"]).read_bytes() != Path(audio).read_bytes(),
          changed.error)

    # ── 三条"必须当场拒绝"的路（拒绝 = 什么都不做 ✓ 不是"照跑但忽略那个参数"✗）──
    cfg_dir = root / "cfg"
    refused = pipe.run_sync(
        pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(cfg_dir),
                                  "guidance": guidance_mod.GuidanceConfig(scale=7.0)}),
        _h3_backend(root, name="h3_cfg"))
    check("㉚ ⭐ 双流 + cfg=7 ⇒ **当场报错** ✓（不静默忽略引导 ✓）",
          refused.ok is False and "不支持引导" in str((refused.error or {}).get("message")),
          refused.error)
    check("㉛ ⭐ 而且**一个文件都没写** ✓（「拒绝」必须是「什么都不做」✓ 半套产物最坏 ✓）",
          not cfg_dir.exists() or not list(cfg_dir.iterdir()),
          list(cfg_dir.iterdir()) if cfg_dir.exists() else None)

    # ── 首帧条件（H3 的 `cond` 段 ✓ 2026-09-20 接线 ✓）──
    import torch  # noqa: PLC0415
    torch.manual_seed(11)
    frame_png = media_mod.write_image(torch.rand(1, 3, 1, 64, 64) * 2.0 - 1.0,
                                      root / "first_frame.png", index=0)["path"]
    with_frame = pipe.run_sync(
        pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(root / "first"),
                                  "first_frame": frame_png}),
        _h3_backend(root, name="h3_ff"))
    check("㉜ ⭐⭐ 双流 + 首帧 ⇒ **跑通** ✓ 且如实标注 `conditioning.mode = h3-keyframe-cond` ✓"
          "（首帧在 **init 阶段**编成 ``cond`` 段 ✓ —— 不是「忽略掉照样跑」✗）",
          with_frame.ok is True
          and (with_frame.conditioning or {}).get("mode") == "h3-keyframe-cond",
          (with_frame.error, with_frame.conditioning))
    check("㉜′ ⭐⭐ **能失败的反向证明**：给首帧 vs 不给 ⇒ **wav 字节必不同** ✓"
          "（相同 ⇒ 首帧根本没进模型 ✗✗ —— 这是「条件真接进去了」唯一算得上的证明 ✓）",
          with_frame.ok is True
          and Path(with_frame.outputs["audioPath"]).read_bytes() != Path(audio).read_bytes(),
          with_frame.error)

    # ── 参考图（H3 的 `ref_img` 段 ✓ 2026-09-20 接线 ✓）──
    with_ref = pipe.run_sync(
        pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(root / "ref"),
                                  "reference_frames": (frame_png,)}),
        _h3_backend(root, name="h3_ref"))
    check("㊴ ⭐⭐ 双流 + 参考图 ⇒ **跑通** ✓（``ref_img`` 块 ✓ 事实：用**图片自己的**网格 ✓ —— "
          "与 ``cond`` 用**目标**网格**不同** ✓ 两种块别混 ✗）",
          with_ref.ok is True, with_ref.error)
    check("㊵ ⭐ **反套套**：给参考图 vs 不给 ⇒ **wav 字节必不同** ✓（相同 ⇒ 参考压根没进模型 ✗✗）",
          with_ref.ok is True
          and Path(with_ref.outputs["audioPath"]).read_bytes() != Path(audio).read_bytes(),
          with_ref.error)

    heun_dir = root / "heun"
    refused_heun = pipe.run_sync(
        pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(heun_dir), "sampler": "heun"}),
        _h3_backend(root, name="h3_heun"))
    check("㉝ ⭐ 双流 + heun ⇒ **明确报错** ✓（循环是一阶欧拉 ✓ —— 静默换算法比报错坏得多 ✗）",
          refused_heun.ok is False
          and "欧拉" in str((refused_heun.error or {}).get("message")), refused_heun.error)


# ══════════════════════════════════════════════════════════════════════════
# ④′ 条件形状守卫：双流要的是**文本状态张量** ✓，不是带 negative 的条件对 ✗
# ══════════════════════════════════════════════════════════════════════════
def case_head_banks_inference(root: Path) -> None:
    """⭐ **头库大小从权重形状推断** ✓（PDD ✓）—— 真权重**不给**这个数 ✓ ⇒ 不猜也不硬编 ✓。"""
    checkpoint = weights_mod.save_module_weights(
        h3_form.H3FormTrunk(**TRUNK, head_banks=3), root / "h3_banked.safetensors")
    backend = TorchBackend(audio_latent_mode="round")
    backend.load_weights(path=checkpoint, config=dict(TRUNK))
    check("㊾ ⭐⭐ 装 `head_banks=3` 的权重 ⇒ 后端**自动推断出 3** ✓"
          "（`describe().config.head_banks` ✓）—— 拿默认 1 去装会**形状不符** ✗；"
          "写死一个数则换检查点就错 ✓✗",
          backend.describe()["config"]["head_banks"] == 3, backend.describe()["config"])
    check("㊾′ ⭐ 显式给**不一致**的 `head_banks` ⇒ **报错** ✓（两者必须一致 ✓ 不静默取一个 ✗）",
          _raises(lambda: TorchBackend(audio_latent_mode="round").load_weights(
              path=checkpoint, config=dict(TRUNK) | {"head_banks": 2}), "不一致") is not None)
    check("㊾″ ⭐ 行数**不是整数倍** ⇒ 报错 ✓（单元级 ✓ —— 不静默回落到 1 ✗，"
          "回落会让 PDD **静默失效** ✓✗ —— 本文件已经在 `out_features` 上栽过一次 ✓）",
          _raises(lambda: h3_form.head_banks_from_shape((99, 8), 4), "整数倍") is not None)
    check("㊾‴ ⭐ 单头宽度 = `latents_dim × pT·pH·pW` ✓（与 `FinalLayer` 里那个 `video_dim` 同一算法 ✓）",
          h3_form.video_patch_dim(24, (1, 2, 2)) == 96
          and h3_form.video_patch_dim(TRUNK["latents_dim"], TRUNK["patch_size"])
          == TRUNK["latents_dim"] * 4)


def case_reference_audio(root: Path) -> None:
    """⭐ 参考**音频**（H3 的 ``ref_audio`` 块 ✓）：wav → 音频 VAE 编码 → 行 ✓ —— wav 读取是**自己写的** ✓。"""
    import torch  # noqa: PLC0415

    rate, channels, seconds = 32000, 2, 0.1
    torch.manual_seed(21)
    samples = (torch.rand(channels, int(rate * seconds)) * 2.0 - 1.0) * 0.5
    wav_path = media_mod.write_wav(samples, root / "ref_clip.wav", sample_rate=rate)["path"]
    clip, info = media_mod.load_wav_tensor(wav_path, sample_rate=rate, channels=channels)
    check("㊿ ⭐ **自己写的 wav 读** ✓（与 `write_wav` **成对** ✓）：形状 ``(channels, N)`` ✓、"
          "值域 ⊆ [-1, 1] ✓、采样率/帧数如实回报 ✓",
          tuple(clip.shape) == (channels, int(rate * seconds))
          and float(clip.abs().max()) <= 1.0 and info["sampleRate"] == rate,
          (tuple(clip.shape), info))
    check("㊿′ ⭐ 采样率/声道**不符** ⇒ 报错 ✓（**不重采样、不混声道** ✗ —— 那会悄悄改掉参考音频的内容 ✓✗）",
          _raises(lambda: media_mod.load_wav_tensor(wav_path, sample_rate=16000,
                                                    channels=channels), "不重采样") is not None)
    backend = _h3_backend(root, name="h3_refaudio")
    result = pipe.run_sync(
        pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(root / "refaudio"),
                                  "reference_audio": (wav_path,)}), backend)
    check("㊿″ ⭐⭐ 双流 + 参考音频 ⇒ **跑通** ✓（wav → 音频 VAE 编码 → ``ref_audio`` 行 ✓）",
          result.ok is True, result.error)
    plain = pipe.run_sync(
        pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(root / "noaudio")}),
        _h3_backend(root, name="h3_noaudio"))
    check("㊿‴ ⭐ **反套套**：给参考音频 vs 不给 ⇒ **wav 字节必不同** ✓（相同 ⇒ 参考压根没进模型 ✗✗）",
          result.ok is True and plain.ok is True
          and Path(result.outputs["audioPath"]).read_bytes()
          != Path(plain.outputs["audioPath"]).read_bytes(), result.error)


def case_reference_video(root: Path) -> None:
    """⭐ 参考**视频**（H3 的 ``video`` / ``video_audio`` 块 ✓）—— 参考块四类的**最后一条入口** ✓。"""
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import torch  # noqa: PLC0415

    # ── 自己造素材 ✓：write_video 出真 mp4（无音轨 ⇒ ``video`` 块 ✓）──
    # ⚠️ 用**平滑渐变**而不用随机噪声 ✗：噪声是有损编码的**最坏情况** ✓（实测 crf=12 下
    #    平均误差 0.35 ✓✗ —— 那是编码器的结构使然 ✗ 不是"读错了"✓ ⇒ 拿它断言"往返忠实"会假红 ✓）。
    source = (torch.linspace(-1.0, 0.5, 64).view(1, 1, 1, 64)
              .expand(1, 3, 3, 64, 64) + torch.tensor([0.0, 0.15, 0.3]).view(1, 1, 3, 1, 1))
    mp4 = media_mod.write_video(source, root / "ref_clip.mp4", fps=12, crf=12)["path"]
    clip, info = media_mod.load_video_tensor(mp4)
    check("51 ⭐ **自己写的视频读** ✓（与 `write_video` **成对** ✓）：形状 ``(1,3,T,H,W)`` ✓、"
          "帧数如实 ✓、``hasAudio=False`` ✓、值域 ⊆ [-1, 1] ✓",
          tuple(clip.shape) == (1, 3, 3, 64, 64) and info["frames"] == 3
          and info["hasAudio"] is False and float(clip.abs().max()) <= 1.0,
          (tuple(clip.shape), {k: info[k] for k in ("frames", "hasAudio")}))
    check("51′ ⭐ **往返忠实** ✓（读回 ≈ 写入 ✓ —— 差只来自有损编码 ✓ 且在上界内 ✓）",
          float((clip - source).abs().mean()) < 0.05, float((clip - source).abs().mean()))

    backend = _h3_backend(root, name="h3_refvideo")
    result = pipe.run_sync(
        pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(root / "refvideo"),
                                  "reference_videos": (mp4,)}), backend)
    check("51″ ⭐⭐ 双流 + 参考视频（无音轨 ⇒ ``video`` 块 ✓）⇒ **跑通** ✓"
          "（帧 → 视频 VAE → ``ref_img`` 行 ✓ 用**它自己的**网格 ✓）",
          result.ok is True, result.error)
    plain = pipe.run_sync(
        pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(root / "novideo")}),
        _h3_backend(root, name="h3_novideo"))
    check("51‴ ⭐ **反套套**：给参考视频 vs 不给 ⇒ **wav 字节必不同** ✓（相同 ⇒ 参考压根没进模型 ✗✗）",
          result.ok is True and plain.ok is True
          and Path(result.outputs["audioPath"]).read_bytes()
          != Path(plain.outputs["audioPath"]).read_bytes(), result.error)

    # ── 带音轨 ⇒ ``video_audio`` 块 ✓：wav + ffmpeg 混流进 mkv（pcm_s16le 可**无损**进 mkv ✓）──
    rate = 32000
    torch.manual_seed(32)
    samples = (torch.rand(2, int(rate * 0.1)) * 2.0 - 1.0) * 0.5
    wav_path = media_mod.write_wav(samples, root / "ref_va_audio.wav", sample_rate=rate)["path"]
    mkv = root / "ref_clip_audio.mkv"
    ffmpeg = shutil.which("ffmpeg")
    muxed = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                            "-i", mp4, "-i", wav_path, "-c", "copy", str(mkv)],
                           capture_output=True, timeout=120)
    if mkv.exists() and mkv.stat().st_size > 0 and muxed.returncode == 0:
        with_audio = pipe.run_sync(
            pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(root / "refva"),
                                      "reference_videos": (str(mkv),)}),
            _h3_backend(root, name="h3_refva"))
        check("52 ⭐⭐ 双流 + **带音轨**参考视频 ⇒ ``video_audio`` 块 ✓ **跑通** ✓"
              "（抽轨 → 音频 VAE → 音频行 ✓ —— 音频行排在视频行**之前** ✓ 事实 ✓）",
              with_audio.ok is True, with_audio.error)
        check("52′ ⭐ **反套套**：带音轨 vs 无音轨参考 ⇒ **wav 字节必不同** ✓"
              "（相同 ⇒ 音轨压根没进模型 ✗✗）",
              with_audio.ok is True
              and Path(with_audio.outputs["audioPath"]).read_bytes()
              != Path(result.outputs["audioPath"]).read_bytes(), with_audio.error)
    else:
        skip("ffmpeg 混流 mkv 失败 ⇒ ``video_audio`` 块用例跳过 ✓")

    # ── 两条"必须当场拒绝"的路 ✓ ──
    class NoVideoRefs:
        """包一层：把能力自述里的 ``referencesVideo`` 摘掉 ✓（模拟"没接参考视频"的后端 ✓）。"""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        @property
        def dualStream(self) -> dict[str, Any]:
            return {**self._inner.dualStream, "referencesVideo": False}

    refused = pipe.run_sync(
        pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(root / "norefvideo"),
                                  "reference_videos": (mp4,)}),
        NoVideoRefs(_h3_backend(root, name="h3_guardvideo")))
    check("53 ⭐ 后端没自述支持参考视频 ⇒ **当场报错** ✓（不静默忽略 `referenceVideos` ✗）",
          refused.ok is False
          and "参考视频" in str((refused.error or {}).get("message")), refused.error)

    odd = media_mod.write_video(torch.rand(1, 3, 2, 100, 100) * 2.0 - 1.0,
                                root / "ref_odd.mp4", fps=12)["path"]
    refused_odd = pipe.run_sync(
        pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(root / "oddref"),
                                  "reference_videos": (odd,)}),
        _h3_backend(root, name="h3_oddref"))
    check("54 ⭐ 参考视频尺寸不能整除 vae_scale ⇒ **明确报错** ✓（不悄悄取整 ✗ —— "
          "那会改变参考素材的网格却没人知道 ✓✗）",
          refused_odd.ok is False
          and "整除" in str((refused_odd.error or {}).get("message")), refused_odd.error)


def case_condition_guard(root: Path) -> None:
    """⚠️ 2026-09-20 自检抓到过真错 ✓：管线把 `encode_text` 的 **dict** 直接当成文本状态递下去 ✗
    ⇒ 主干把它当行张量 ⇒ `AttributeError: ... 'dict' object has no attribute 'shape'` ✓（它**响亮** ✓
    算是走运 ✓）。⇒ 除了走通 ✓，还要钉住"**形状不对就当场拒绝**" ✓。
    """
    inner = _h3_backend(root, name="h3_guard")

    class WithNegative:
        """包一层：让条件变成**带 negative 的 dict** ✓（模拟"按可引导方式准备"的后端 ✓）。"""

        def __init__(self, backend: Any) -> None:
            self._inner = backend

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def encode_text(self, request: Any) -> dict[str, Any]:
            states = self._inner.encode_text(request)
            return {**states, "negative": states["positive"]}

    result = pipe.run_sync(
        pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(root / "guard")}),
        WithNegative(inner))
    check("㊳ 双流 + 条件里带 negative ⇒ **明确报错** ✓（不给「反正不引导」就无视它 ✗ —— "
          "那正是「静默忽略」✓✗）",
          result.ok is False and "negative" in str((result.error or {}).get("message")),
          result.error)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ ⭐ **默认路径一个字没动**：没有双流能力的后端照旧跑单流 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_single_stream_untouched(root: Path) -> None:
    result = pipe.run_sync(pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(root / "one")}),
                           DryRunBackend())
    outputs = result.outputs or {}
    check("㉞ 干跑后端（**没有**双流能力）⇒ 照旧 `ok=True` ✓ 单流 ✓",
          result.ok is True and "audioPath" not in outputs, result.error)
    check("㉟ `outputs.dualStream` **不存在** ✓（没有就是没有 ✓ 不冒充 ✓）",
          "dualStream" not in outputs, sorted(outputs))
    check("㊱ 干跑后端也没有 `dualStream` 属性 ⇒ 管线按单流走 ✓（能力探测 ✓ 不是猜 ✓）",
          not hasattr(DryRunBackend(), "dualStream"))
    check("㊲ `TorchBackendUnavailable` 仍是**明确报错**类型 ✓（双流不齐时报的是它 ✓ 不是裸 assert ✗）",
          issubclass(TorchBackendUnavailable, RuntimeError))


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="engine_dual_"))
    case_gating()
    case_single_stream_untouched(root)
    if not (_have_torch() and media_mod.have_ffmpeg()):
        skip("缺 torch 或 ffmpeg ⇒ 双流整链（含 mp4/wav 落盘）跳过 ✓")
    else:
        case_dual_latents(root)
        case_text_states(root)
        case_pipeline(root)
        case_condition_guard(root)
        case_head_banks_inference(root)
        case_reference_audio(root)
        case_reference_video(root)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name
              + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
