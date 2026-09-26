"""自检：**续拍**（extend ✓）与**重拍**（retake ✓）—— 与「拼接」正交的两件事 ✓（2026-09-26 ✓）。

钉的都是**能证伪**的东西 ✓：
* 计划层拿**上游真值**钉死 ✓（帧网格 / 窗口边界 / 银行家舍入的后果 ✓）——
  事实来源 ``reference/ComfyUI-H3-Multishot/h3_extend.py`` 与 ``h3_retake.py`` ✓（MIT ✓）；
* **冻结区逐位等于原始潜变量** ✓✓（这是"重拍/续拍"与"整条锁死"的分水岭 ✓，见
  :func:`app.services.engine.h3_form.sample_dual_stream` 的 docstring ✓）；
* **掩码全开 == 不 pin** 逐位相同 ✓（证明这条新路径**不影响**原来的采样 ✓）；
* 反向证明：槽位对不上 / 尾巴比窗口长 / 缺 ``frames`` / σ₀=0 ⇒ **都当场报错** ✓（不猜 ✗）。

⚠️ 用**缩小版真主干**（与 ``tests/engine_dual_stream_test.py`` 同一套配置 ✓ ⇒ 走的还是
``load_weights`` 判形态那条真路径 ✓）；权重是**参考实现（未训练）** ✓ ⇒ 数值没有语义 ✓，
本自检要的正是「**结构 + 语义**」✓ 不是画质 ✓。

运行::

    python tests/engine_h3_edit_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import audio_vae as audio_vae_mod  # noqa: E402
from app.services.engine import geometry as geometry_mod  # noqa: E402
from app.services.engine import h3_edit as edit_mod  # noqa: E402
from app.services.engine import h3_form  # noqa: E402
from app.services.engine import pipeline as pipe  # noqa: E402
from app.services.engine import vae as vae_mod  # noqa: E402
from app.services.engine import weights as weights_mod  # noqa: E402
from app.services.engine.torch_backend import TorchBackend  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: Any, needle: str) -> str | None:
    """**能触发**的反向证明 ✓：调它、看报错里有没有那个词 ✓（没报错 ⇒ 返回 None ⇒ 断言红 ✓）。"""
    try:
        call()
    except Exception as err:  # noqa: BLE001 —— 就是来看它报什么的 ✓
        return str(err) if needle in str(err) else None
    return None


def _torch() -> Any:
    import torch  # noqa: PLC0415
    return torch


TRUNK: dict[str, Any] = {
    "hidden": 32, "layers": 2, "heads": 4, "head_dim": 16, "ffn": 64,
    "text_dim": 16, "latents_dim": 4, "audio_latents_dim": 8, "patch_size": (1, 2, 2),
    "time_input_dim": 16, "time_hidden": 32, "time_dim": 32, "inv_freq_len": 2,
    "refiner_layers": 1,
}
VIDEO_VAE = vae_mod.VideoVAEConfig(base_channels=4, latent_channels=4,
                                   channel_multipliers=(1, 1, 1, 1, 1))
AUDIO_VAE = audio_vae_mod.AudioVAEConfig(latent_channels=8, encoder_dim=8,
                                         latent_dim=32, decoder_dim=128)
#: 1:1 ⇒ 224×224（/16 = 14×14 ✓）/ 5 帧 / 3 步 ✓ —— 与双流自检同口径 ✓ CPU 秒级 ✓
REQUEST: dict[str, Any] = {"prompt": "雨夜霓虹街头", "seed": 7, "steps": 3, "seconds": 0.2,
                           "ratio": "1:1", "megapixels": 0.05, "temporal_compression": 1}


def _h3_backend(root: Path, *, name: str = "h3_edit") -> TorchBackend:
    """按**真路径**建后端 ✓：造 H3 形态权重 ⇒ `load_weights` 判形态 ⇒ 挂 TE / 两个 VAE ✓。

    ⚠️ **显式 CPU** ✓（2026-09-26 纠 ✓，口径同 `engine_refine_test.backend_or_skip` ✓）：
    本套验的是**冻结区/钉头的张量层语义** ✓ ⇒ 设备无关 ✓；而尾巴这类张量是自检**自己造**的
    （``torch.randn`` 默认 CPU ✓）⇒ 后端若自动探测到 ``cuda`` 就会设备混用当场炸 ✓✗
    （`_require_own_device` 的文案自己就点名了这条 ✓）。真 CUDA 上的整条双流由
    `engine_dual_stream_test` 覆盖 ✓。
    """
    backend = TorchBackend(device="cpu", audio_latent_mode="round")
    checkpoint = weights_mod.save_module_weights(
        h3_form.H3FormTrunk(**TRUNK), root / f"{name}.safetensors")
    backend.load_weights(path=checkpoint, config=dict(TRUNK))
    backend.attach_text_encoder()
    backend.attach_vae(VIDEO_VAE)
    backend.attach_audio_vae(AUDIO_VAE)
    return backend


# ══════════════════════════════════════════════════════════════════════════
# ① 计划层（**零依赖** ✓ 可离线钉死 ✓）—— 真值全部来自上游源码 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_plan_arithmetic() -> None:
    check("① `snap_grid` 就近吸附到 5+17k ✓（243 = 5+14×17 ✓ 是**合法**窗口 ✓）",
          edit_mod.snap_grid(243) == 243, edit_mod.snap_grid(243))
    check("② ⭐ 就近**不是**向上 ✗：`snap_grid(201)` = **209** ✓（到 192 差 9 ✓、到 209 差 8 ✓）",
          edit_mod.snap_grid(201) == 209, edit_mod.snap_grid(201))
    check("②′ 而 `snap_grid(200)` = **192** ✓（到 192 差 8 ✓、到 209 差 9 ✓ ⇒ 就近取下面那档 ✓ —— "
          "2026-09-26 纠：原来这里断言 209 ✗，恰好把「就近不是向上」举成了它的**反例** ✓）",
          edit_mod.snap_grid(200) == 192, edit_mod.snap_grid(200))
    check("③ 太小的窗口**兜到最小合法长度** 22 帧 ✓（不返回 5 ✗）",
          edit_mod.snap_grid(6) == 22 and edit_mod.snap_grid(1) == 22, edit_mod.snap_grid(6))

    plan = edit_mod.plan_take(10.0, 90, fps=24, pin_frames=22)
    check("④ 续拍窗口数**是解出来的** ✓：10 s@24 = 240 帧 ✓、窗口 90 ✓、钉 22 ⇒ "
          "step 68 ✓ ⇒ 4 个窗口 ✓（90+3×68 = 294 ✓）",
          (plan.num_windows, plan.frames_per_window, plan.step_frames, plan.total_frames)
          == (4, 90, 68, 294), plan.to_dict())
    check("⑤ 交付帧数只会**长一点**不会短 ✓（294 ≥ 240 ✓）且 summary 带上窗口事实 ✓",
          plan.total_frames >= plan.requested_frames and "4 windows x 90 frames" in plan.summary,
          plan.summary)
    check("⑥ ⚠️ `window='auto'` **没给 fits 判据** ⇒ 明确报错 ✓（本仓没有上游那套实测速率表 ✗）",
          _raises(lambda: edit_mod.plan_take(10.0, "auto"), "fits") is not None)
    auto = edit_mod.plan_take(10.0, "auto", fits=lambda frames: frames <= 100)
    check("⑦ 给了 fits ⇒ 从大到小挑第一个装得下的 ✓（≤100 ⇒ 90 ✓ 不是 243 ✗）",
          auto.frames_per_window == 90 and "auto" in auto.summary, auto.summary)
    # 判据落在**本窗口**上 ✓（2026-09-26 纠：原来这条喂 pin=30 / 窗口 90 —— 30 < 90 ✓ 根本没
    # 触发「钉满」✗，名字说的「钉长 ≥ 一个窗口」跟喂进去的数对不上 ✗）。上游钉长可到 56 帧 ✓
    # 配 141/192/243 帧窗口是**合法且有实测**的配置 ✓ ⇒ 拿网格最小窗口 22 帧当上限会把上游合法
    # 配置（连同本模块默认值 22 ✗）一起拒掉 ✗。会「永远交付不了新帧」的只有 pin ≥ 本窗口 ✓。
    check("⑧ 钉长 ≥ 本窗口 ⇒ 明确报错 ✓（每窗新帧被 max(1, …) 压成 1 帧 ⇒ 永远交付不了新帧 ✓）",
          _raises(lambda: edit_mod.plan_take(10.0, 90, pin_frames=90), "钉满") is not None)
    check("⑧′ 上游合法上限 56 帧配 141 帧窗口**照样能跑** ✓（22 帧那条界限是错的 ✗）",
          edit_mod.plan_take(10.0, 141, pin_frames=56).step_frames == 85,
          edit_mod.plan_take(10.0, 141, pin_frames=56).to_dict())

    check("⑨ `grid_frames` 是**向下**吃网格 ✓：130 帧 ⇒ 124 ✓（124 = 5+7×17 ✓ 尾巴 6 帧原样送回 ✓）",
          edit_mod.grid_frames(130) == 124, edit_mod.grid_frames(130))
    check("⑩ <22 帧 ⇒ **原样** ✓（不硬塞进网格 ✗）",
          edit_mod.grid_frames(10) == 10 and edit_mod.grid_frames(21) == 21,
          (edit_mod.grid_frames(10), edit_mod.grid_frames(21)))

    check("⑪ `latent_window` 按秒映射到槽 ✓：40 槽 / 4 s / 1.0→2.0 s ⇒ (10, 20) ✓",
          edit_mod.latent_window(40, 4.0, 1.0, 2.0) == (10, 20),
          edit_mod.latent_window(40, 4.0, 1.0, 2.0))
    check("⑫ ⭐ 上游那个 `round(x + 0.5)` 的**银行家舍入后果**照抄 ✓：40 槽 / 4 s / 0→4.0 s ⇒ "
          "**b=40** ✓（round(40.5)=40 ✓ 不是 41 ✗）",
          edit_mod.latent_window(40, 4.0, 0.0, 4.0) == (0, 40),
          edit_mod.latent_window(40, 4.0, 0.0, 4.0))
    check("⑬ 窗口至少 1 个槽、且绝不越界 ✓（b ≤ 槽数 ✓ a ≤ 槽数-1 ✓）",
          edit_mod.latent_window(5, 1.0, 0.99, 9.0) == (4, 5),
          edit_mod.latent_window(5, 1.0, 0.99, 9.0))
    # ⚠️ 2026-09-26 纠：`latent_window` 那一层是**纯映射** ✓（上下界各夹一次 ✓，不校验区间顺序 ✓
    #    —— 见它的 docstring ✓）；「end ≤ start 要报错」的守卫在 `plan_retake` 上 ✓（docstring 第
    #    271 行那条 ✓）⇒ 原来拿底层原语去验上层守卫 ⇒ 它当然不报 ✓✗。
    check("⑭ `end <= start` ⇒ 明确报错 ✓（静默把窗口翻过来 = 重画了一整段别的 ✗）",
          _raises(lambda: edit_mod.plan_retake(5, 8, total_seconds=0.2, start_seconds=0.12,
                                               end_seconds=0.12), "之后") is not None)

    both = edit_mod.plan_retake(5, 8, total_seconds=0.2, start_seconds=0.04, end_seconds=0.12)
    check("⑮ 重拍 `video + audio` ⇒ **两条流各算各的**窗口 ✓（视频 5 槽 / 音频 8 槽 ✓ —— "
          "各自按自己的帧率映射 ✓ 不是同一个数 ✗）",
          both.video is not None and both.audio is not None and both.video != both.audio,
          (both.video, both.audio))
    video_only = edit_mod.plan_retake(5, 8, total_seconds=0.2, start_seconds=0.04,
                                      end_seconds=0.12, mode="video only (keep the performance)")
    check("⑯ `video only` ⇒ **声音整条冻结** ✓（audio 窗口 None ✓ 保住表演 ✓）",
          video_only.redo_video and not video_only.redo_audio and video_only.audio is None,
          video_only.to_dict())
    audio_only = edit_mod.plan_retake(5, 8, total_seconds=0.2, start_seconds=0.04,
                                      end_seconds=0.12, mode="audio only (keep the picture)")
    check("⑰ `audio only` ⇒ **画面整条冻结** ✓（保住画面 ✓）",
          audio_only.redo_audio and not audio_only.redo_video and audio_only.video is None,
          audio_only.describe(clip_seconds=0.2))
    check("⑱ 模式是**白名单** ✓（编一个 `both` 不算 ✓ ⇒ 报错 ✓）",
          _raises(lambda: edit_mod.plan_retake(5, 8, total_seconds=0.2, start_seconds=0.0,
                                               end_seconds=0.1, mode="both"), "mode") is not None)

    pin = edit_mod.continuation_pin(2, fps=24, temporal_compression=1, audio_latent_mode="round")
    check("⑲ 续拍钉 2 帧（压缩比 1 ✓）⇒ 视频 2 槽 ✓、音频 round(2/24×40) = **3** 槽 ✓"
          "（按**秒**算 ✓ 不是按帧 ✗）",
          (pin.video_slots, pin.audio_slots) == (2, 3), pin.to_dict())
    check("⑳ 音频槽数**必须**与 `geometry.audio_latent_frames` 同一份事实 ✓（换口径就会变 ✓）",
          pin.audio_slots == geometry_mod.audio_latent_frames(pin.seconds, mode="round")
          and edit_mod.continuation_pin(2, fps=24, temporal_compression=1,
                                        audio_latent_mode="ceil").audio_slots
          != pin.audio_slots)
    pin4 = edit_mod.continuation_pin(22, fps=24, temporal_compression=4, audio_latent_mode="round")
    check("㉑ 压缩比 4 ⇒ 22 帧 = **6 个视频槽** ✓（ceil(22/4) ✓ 不是 5 ✗）",
          pin4.video_slots == 6, pin4.to_dict())


# ══════════════════════════════════════════════════════════════════════════
# ② 冻结区原语：**逐位**语义（真张量 ✓ 缩小版真主干 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_freeze_semantics(root: Path) -> None:
    torch = _torch()
    backend = _h3_backend(root, name="h3_edit")
    request = pipe.GenerationRequest(**REQUEST)
    plan = pipe.build_plan(request)
    condition = backend.encode_text(request)
    # ⚠️ 双流要的是**文本状态张量** ✓ —— 不是 `encode_text` 那个 dict ✗（口径与
    #    `pipeline.py` 同 ✓：它也是 `condition["positive"]` 再递下去 ✓）。
    #    2026-09-26 纠：原来直接把 dict 递进去 ⇒ 主干拿它当行张量 ⇒
    #    `AttributeError: 'dict' object has no attribute 'shape'` ✓。
    states = condition["positive"]
    # ⚠️ σ 日程**只有一份** ✓：喂给采样器的与 `plan` 上读得到的必须是同一份 ✓（2026-09-26 纠 ✓）。
    #    `_dual_sigma0` 取 `plan.sigmas[0]` 当钉头的 σ₀ ✓，而采样器吃的是**显式递进去**的那份 ✓
    #    （口径同 `pipeline.py`：它也是 `sample_dual(…, plan.sigmas, …)` ✓）。两份不一致 ⇒ 钉头按
    #    karras 首值 14.61 铺 ✓、采样器按 1.0 走 ✓ ⇒ 下面几条张量断言必然对不上 ✓。
    # ⚠️ 本组要**两条** σ 日程，各验各的 ✓（每组都只有一份 ✓ —— 换日程时 `plan.sigmas` 一起换 ✓）：
    #    ① σ₀ = **1** ⇒ ㉓ 那个恒等式才成立 ✓（``x₀ = 0 + 1·noise`` 正好是那条**不 pin** 的起点 ✓）；
    #       但 1 是任意 shift 的**不动点** ✓ ⇒ 那里两条流的 σ 都是 1 ✗ ⇒ 折算**看不出来** ✓；
    #    ② σ₀ = 0.5 ⇒ 才验得出「两条流各算各的」✓（⇒ 音频 **0.2** ✓ 见 ㉖′ ✓）。
    sigmas = [1.0, 0.5, 0.0]
    plan.sigmas = list(sigmas)
    noise = backend.init_dual_latents(plan, request)
    slots_v, slots_a = int(noise["video"].shape[1]), int(noise["audio"].shape[2])
    check("㉒ 素材：视频 5 槽 / 音频 8 槽 ✓（与 plan 同源 ✓）",
          (slots_v, slots_a) == (int(plan.latent_frames), 8), (slots_v, slots_a))

    # ① 掩码**全开** + z0=0 ⇒ 与"不 pin"**逐位相同** ✓（新路径不影响原来的采样 ✓）
    plain = backend.sample_dual({"video": noise["video"], "audio": noise["audio"]},
                                sigmas, states, request)
    zero = {"video": torch.zeros_like(noise["video"]), "audio": torch.zeros_like(noise["audio"])}
    whole = edit_mod.plan_retake(slots_v, slots_a, total_seconds=0.2, start_seconds=0.0,
                                 end_seconds=0.2)
    opened = backend.retake_latents(zero, plan, request, condition=states, sigmas=sigmas,
                                    retake=whole)
    check("㉓ ⭐ 掩码**全开** + z0=0 ⇒ 与不 pin **逐位相同** ✓（重画区不被扰动 ✓）",
          bool(torch.equal(opened["video"], plain["video"]))
          and bool(torch.equal(opened["audio"], plain["audio"])),
          (float((opened["video"] - plain["video"]).abs().max()),
           float((opened["audio"] - plain["audio"]).abs().max())))

    # ② 中间开窗 ⇒ 窗外**逐位**等于原始潜变量 ✓✓（这就是重拍的承诺 ✓）
    z0 = {"video": torch.randn_like(noise["video"]) * 0.5,
          "audio": torch.randn_like(noise["audio"]) * 0.5}
    middle = edit_mod.plan_retake(slots_v, slots_a, total_seconds=0.2, start_seconds=0.04,
                                  end_seconds=0.08)
    got = backend.retake_latents(z0, plan, request, condition=states, sigmas=sigmas,
                                 retake=middle)
    v_i, v_j = middle.video
    a_i, a_j = middle.audio
    frozen_v = torch.cat([got["video"][:, :v_i], got["video"][:, v_j:]], dim=1)
    want_v = torch.cat([z0["video"][:, :v_i], z0["video"][:, v_j:]], dim=1)
    frozen_a = torch.cat([got["audio"][:, :, :a_i], got["audio"][:, :, a_j:]], dim=2)
    want_a = torch.cat([z0["audio"][:, :, :a_i], z0["audio"][:, :, a_j:]], dim=2)
    check(f"㉔ ⭐⭐ 窗外**逐位**等于原始潜变量 ✓✓（视频槽 [{v_i},{v_j}) / 音频槽 [{a_i},{a_j}) 之外 ✓）",
          bool(torch.equal(frozen_v, want_v)) and bool(torch.equal(frozen_a, want_a)),
          (float((frozen_v - want_v).abs().max()), float((frozen_a - want_a).abs().max())))
    check("㉕ 窗内**真的变了** ✓（否则就是「什么都没重画」的假绿 ✗）",
          not bool(torch.equal(got["video"][:, v_i:v_j], z0["video"][:, v_i:v_j]))
          and not bool(torch.equal(got["audio"][:, :, a_i:a_j], z0["audio"][:, :, a_i:a_j])))
    # ⚠️ 2026-09-26 纠：原来这条断言 `sigmaVideo0 == 1.0` ✓ **并且**要求两条流的 σ **不等** ✗ ——
    #    而 σ₀ 恰好就是 1.0 ✓，1 是任意 shift 的**不动点** ✓ ⇒ 那里两条流的 σ 必然**相等** ✓
    #    ⇒ 原断言自相矛盾 ✓✗（产品没错 ✓ 是测试写错了 ✓）。本组 σ₀ = 1 是 ㉓ 那个恒等式的前提 ✓
    #    ⇒ 这里就**照实**钉「两条都 = 1」✓，折算另开一条（㉖′ ✓ σ₀ = 0.5 ✓）。
    check("㉖ 报告如实带上窗口与两条流各自的**首 σ** ✓（σ₀ = 1 ⇒ 两条都 = 1 ✓ —— σ=1 是任意"
          " shift 的**不动点** ✓ 这里相等是数学 ✓ 不是 bug ✓）",
          isinstance(got.get("retake"), dict)
          and got["retake"]["sigmaVideo0"] == sigmas[0]
          and got["retake"]["sigmaAudio0"] == sigmas[0], got.get("retake"))
    # ⚠️ 折算**必须另开一条** ✓：σ₀ = 1 上根本看不出两条流有什么区别 ✓（不动点 ✓）—— 换成 0.5 ⇒
    #    视频 0.5 ✓、音频 ``time_shift_sigma(0.5, 12, 3)`` = **0.2** ✓（手算：base = 0.5/(12−11×0.5)
    #    = 1/13 ✓ ⇒ 3·(1/13)/(1+2/13) = 3/15 = 0.2 ✓）。张量极小 ⇒ 多跑一趟采样可忽略 ✓。
    shifted_sigmas = [0.5, 0.25, 0.0]
    plan.sigmas = list(shifted_sigmas)
    shifted = backend.retake_latents(z0, plan, request, condition=states, sigmas=shifted_sigmas,
                                     retake=middle)
    check("㉖′ ⭐ σ₀ = 0.5 ⇒ **音频那条按 shift 折算** ✓（视频 0.5 ✓ / 音频 **0.2** ✓ —— 两条流"
          "各算各的 ✓ 不是同一个数 ✗）",
          abs(shifted["retake"]["sigmaVideo0"] - 0.5) < 1e-9
          and abs(shifted["retake"]["sigmaAudio0"] - 0.2) < 1e-9, shifted.get("retake"))
    check("㉗ `pinned` 标志如实 ✓（走了冻结这条路的自陈 ✓）",
          got.get("pinned") is True and plain.get("pinned") is False,
          (got.get("pinned"), plain.get("pinned")))

    # ③ 整条冻结（audio only）⇒ 那条流**逐位**等于原始潜变量 ✓
    only_audio = edit_mod.plan_retake(
        slots_v, slots_a, total_seconds=0.2, start_seconds=0.04, end_seconds=0.08,
        mode="audio only (keep the picture)")
    picture = backend.retake_latents(z0, plan, request, condition=states, sigmas=sigmas,
                                     retake=only_audio)
    check("㉘ ⭐ `audio only` ⇒ 画面**整条逐位**保留 ✓（掩码全 0 ✓ 但主干是**联合**前向 ⇒ "
          "仍要逐步喂进去 ✓ 不白算 ✓）",
          bool(torch.equal(picture["video"], z0["video"])),
          float((picture["video"] - z0["video"]).abs().max()))

    # ④ 反向证明：槽位 / sigma 对不上 ⇒ 都当场报错 ✓
    bad = edit_mod.plan_retake(slots_v + 2, slots_a, total_seconds=0.2, start_seconds=0.04,
                               end_seconds=0.08)
    check("㉙ 计划层的槽位数与**张量实际时间维**不等 ⇒ 报错 ✓（不按张量改窗口 ✗）",
          _raises(lambda: backend.retake_latents(z0, plan, request, condition=states,
                                                 sigmas=sigmas, retake=bad), "不一致") is not None)
    check("㉚ 不给 `RetakePlan`（给个 True ✓）⇒ 报错 ✓（窗口必须**算出来** ✓ 不猜 ✗）",
          _raises(lambda: backend.retake_latents(z0, plan, request, condition=states,
                                                 sigmas=sigmas, retake=True), "RetakePlan")
          is not None)
    check("㉛ σ₀=0 ⇒ 报错 ✓（冻结区的噪声方向反推不出来 ✓ 不猜 ✗）",
          _raises(lambda: backend.retake_latents(z0, plan, request, condition=states,
                                                 sigmas=[0.0, 0.0], retake=middle), "首 σ")
          is not None)
    check("㉜ σ 末位不是 0 ⇒ 报错 ✓（与 `sample_dual_stream` 同口径 ✓）",
          _raises(lambda: backend.retake_latents(z0, plan, request, condition=states,
                                                 sigmas=[1.0, 0.5], retake=middle), "末位")
          is not None)


# ══════════════════════════════════════════════════════════════════════════
# ③ 续拍：上一段的**尾部潜变量**钉进下一段的**开头** ✓（逐位 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_continuation(root: Path) -> None:
    torch = _torch()
    backend = _h3_backend(root, name="h3_edit_pin")
    request = pipe.GenerationRequest(**REQUEST)
    plan = pipe.build_plan(request)
    condition = backend.encode_text(request)
    # ⚠️ 双流要的是**文本状态张量** ✓ —— 不是 `encode_text` 那个 dict ✗（口径与
    #    `pipeline.py` 同 ✓：它也是 `condition["positive"]` 再递下去 ✓）。
    #    2026-09-26 纠：原来直接把 dict 递进去 ⇒ 主干拿它当行张量 ⇒
    #    `AttributeError: 'dict' object has no attribute 'shape'` ✓。
    states = condition["positive"]
    # ⚠️ 同 `case_freeze_semantics` ✓：σ 日程**只有一份** —— 采样器拿到的与 `plan.sigmas` 必须是
    #    同一份 ✓（续拍的钉头 σ₀ 就取 ``plan.sigmas[0]`` ✓，见 `_dual_sigma0` ✓）。
    sigmas = [0.5, 0.25, 0.0]
    plan.sigmas = list(sigmas)
    plain = backend.init_dual_latents(plan, request)

    pin = edit_mod.continuation_pin(2, fps=24, temporal_compression=1, audio_latent_mode="round")
    # ⚠️ 通道数与空间尺寸要**照抄**本次窗口 ✓（2026-09-26 纠：原来写死 1 通道 ⇒
    #    `_dual_carry_pin` 的「同一套潜空间」守卫当场红 ✓ —— 那是守卫对 ✓ 是这里的形状错 ✗）。
    tail_v = torch.randn(int(plain["video"].shape[0]), pin.video_slots,
                         int(plain["video"].shape[2]), int(plain["video"].shape[3]))
    tail_a = torch.randn(int(plain["audio"].shape[0]), int(plain["audio"].shape[1]),
                         pin.audio_slots)
    object.__setattr__(request, "carry_latents",
                       {"video": tail_v, "audio": tail_a, "frames": pin.pin_frames})
    carried = backend.init_dual_latents(plan, request)
    head_v, head_a = pin.video_slots, pin.audio_slots
    check("㉝ ⭐ 续拍把尾巴铺成 ``x₀ = 尾巴 + σ₀·noise`` ✓（头部 2 槽**逐位**相等 ✓；头部之后 = "
          "``σ₀·noise`` ✓ —— 那儿 pin 是**填充的 0** ✓ 不是上一段的尾巴 ✗）",
          bool(torch.allclose(carried["video"][:, :head_v],
                              tail_v + float(sigmas[0]) * plain["video"][:, :head_v]))
          # ⚠️ 2026-09-26 纠：原来这里拿 ``carried[…, head_v:]`` 去比**未折算**的噪声 ✗ —— 那隐含
          #    σ₀ = 1 ✓（原来那条 σ 恰好是 1 ✓ 于是碰巧为真 ✓✗）；σ₀ ≠ 1 时头部之后是 ``σ₀·noise`` ✓
          #    （掩码 = 1 ⇒ 要重画 ✓ 而 pin 在那儿是**填充 0** ✓）。本用例里 `plain` = 那条**没钉**
          #    的路 ⇒ 它的初值就是噪声本身 ✓。
          and bool(torch.allclose(carried["video"][:, head_v:],
                                  float(sigmas[0]) * plain["video"][:, head_v:], atol=1e-6)),
          float((carried["video"][:, :head_v] - tail_v).abs().max()))
    check("㉞ 音频头用的是**音频自己的 σ₀** ✓（`time_shift_sigma(0.5, 12, 3)` = **0.2** ✓ 不是 0.5 ✗ "
          "—— 手算：base = 0.5/(12−11×0.5) = 1/13 ✓ ⇒ 3·(1/13)/(1+2/13) = 3/15 = 0.2 ✓。"
          "2026-09-26 纠：原文写 0.25 ✗ —— `time_shift_sigma(1.0, 12, 3)` 其实是 **1.0** ✓"
          "（σ=1 是任意 shift 的**不动点** ✓），0.25 不知道从哪来 ✓）",
          bool(torch.allclose(carried["audio"][:, :, :head_a],
                              tail_a + 0.2 * plain["audio"][:, :, :head_a], atol=1e-6)),
          float((carried["audio"][:, :, :head_a] - tail_a).abs().max()))
    mask_v, mask_a = carried["pinMask"]
    check("㉟ 掩码：**头部 0（冻结 ✓）**、其余 1（重画 ✓）—— 两条流各切在**自己的**槽位上 ✓",
          float(mask_v[:, :head_v].max()) == 0.0 and float(mask_v[:, head_v:].min()) == 1.0
          and float(mask_a[:, :, :head_a].max()) == 0.0
          and float(mask_a[:, :, head_a:].min()) == 1.0,
          (mask_v.flatten().tolist(), mask_a.flatten().tolist()))
    check("㊱ 交接的事实如实回报 ✓（`continuation` 段带帧数 / 槽数 / 两条流的首 σ ✓）",
          (carried.get("continuation") or {}).get("pinFrames") == pin.pin_frames
          and (carried.get("continuation") or {}).get("videoSlots") == head_v
          and (carried.get("continuation") or {}).get("source") == "carry_latents",
          carried.get("continuation"))

    ran = backend.sample_dual(carried, sigmas, states, request)
    check("㊲ ⭐⭐ 采完样之后，**头部逐位等于上一段的尾巴** ✓✓（续拍的核心承诺 ✓ "
          "—— 不是“差不多” ✓ 是**逐位** ✓）",
          bool(torch.equal(ran["video"][:, :head_v], tail_v))
          and bool(torch.equal(ran["audio"][:, :, :head_a], tail_a)),
          (float((ran["video"][:, :head_v] - tail_v).abs().max()),
           float((ran["audio"][:, :, :head_a] - tail_a).abs().max())))
    check("㊳ 头部之后**是新画的** ✓（其余槽位与噪声初值不同 ⇒ 真跑了 ✓ 不是原样退回 ✗）",
          not bool(torch.equal(ran["video"][:, head_v:], carried["video"][:, head_v:])))

    # 反向证明：三条口径都有话直说 ✓
    # ⚠️ 2026-09-26 纠：这里原来写死 **1 通道** ✗ ⇒ 先撞上「不是同一套潜空间」那条守卫（它更靠前 ✓）
    #    ⇒ 「还长」这条根本轮不到 ✓✗（实跑的红就是这个 ✓）。通道/空间照抄本次窗口 ✓（同 `tail_v` ✓）。
    long_tail = {"video": torch.randn(int(plain["video"].shape[0]), int(plan.latent_frames) + 1,
                                      int(plain["video"].shape[2]), int(plain["video"].shape[3])),
                 "audio": tail_a, "frames": int(plan.latent_frames) + 1}
    object.__setattr__(request, "carry_latents", long_tail)
    check("㊴ 尾巴比本次窗口**还长** ⇒ 报错 ✓（钉不下 ✓ 不静默截断 ✗）",
          _raises(lambda: backend.init_dual_latents(plan, request), "还长") is not None)
    object.__setattr__(request, "carry_latents",
                       {"video": tail_v, "audio": tail_a, "frames": pin.pin_frames})
    check("㊵ 缺 `frames` ⇒ 报错 ✓（「钉了多少帧」本仓**不猜** ✓）",
          _raises(lambda: backend._dual_carry_pin(
              {"video": tail_v, "audio": tail_a}, plain["video"], plain["audio"],
              temporal_compression=int(request.temporal_compression)),
              "frames") is not None)
    # ⚠️ 2026-09-26 纠两处：
    #   1. 这条守卫原来读 ``getattr(plan, "temporal_compression", None)`` ✗ —— 而 `GenerationPlan`
    #      上**没有**这个字段 ✓（压缩比是 `GenerationRequest` 的 ✓，plan 只存换算结果 ✓）⇒ 它恒为
    #      None ⇒ 守卫**在真管线里永不触发** ✓✗（拿「读不到」当「没问题」＝静默兜底 ✓）。现在压缩比
    #      由调用方显式递进来 ✓、缺了直接报错 ✓ ⇒ 多了下面 ㊶′ 那条专钉它 ✓。
    #   2. 原来的第一条断言用 `is None` 去配「对不上」这个词 ✗ —— 产品的文案里**没有**这个词 ✓
    #      ⇒ 它靠「针不对 ⇒ `_raises` 返回 None」**碰巧**为真 ✓✗（假绿 ✓）。改成钉产品真有的那句 ✓。
    check("㊶ `frames` 与**视频槽数**对不上（同一件事写两处 ✓）⇒ 报错 ✓（7 ✓ / 9 ✓ 都拦 ✓；"
          "对得上的 2 ✓ 放行 ✓）",
          _raises(lambda: backend._dual_carry_pin(
              {"video": tail_v, "audio": tail_a, "frames": 7}, plain["video"], plain["audio"],
              temporal_compression=int(request.temporal_compression)),
              "必有一处错") is not None
          and _raises(lambda: backend._dual_carry_pin(
              {"video": tail_v, "audio": tail_a, "frames": 9}, plain["video"], plain["audio"],
              temporal_compression=int(request.temporal_compression)),
              "必有一处错") is not None
          and _raises(lambda: backend._dual_carry_pin(
              {"video": tail_v, "audio": tail_a, "frames": pin.pin_frames}, plain["video"],
              plain["audio"], temporal_compression=int(request.temporal_compression)), "") is None)
    check("㊶′ ⭐ 没给压缩比 ⇒ 报错 ✓（压缩比**不在** `plan` 上 ✓ ⇒ 只有**请求**能提供 ✓ "
          "—— 缺了就**没得对** ✓ 不拿「读不到」当「没问题」✗）",
          _raises(lambda: backend._dual_carry_pin(
              {"video": tail_v, "audio": tail_a, "frames": 2}, plain["video"], plain["audio"],
              temporal_compression=None), "不猜压缩比") is not None)
    check("㊷ 拿**解码后的帧**来当尾巴（形状不是潜变量 ✗）⇒ 报错 ✓（两回事 ✓）",
          _raises(lambda: backend._dual_carry_pin(
              {"video": torch.randn(3, 224, 224), "audio": tail_a, "frames": 2},
              plain["video"], plain["audio"],
              temporal_compression=int(request.temporal_compression)),
              "不是同一套潜空间") is not None)
    check("㊸ 没算 σ 日程（`plan.sigmas` 空）⇒ 报错 ✓（钉不出那条含噪轨迹 ✓ 不猜 σ₀ ✗）",
          _raises(lambda: backend._dual_sigma0(
              pipe.GenerationPlan(width=224, height=224, frames=5, fps=24, latent_frames=5,
                                  steps=3, sigmas=[])), "sigmas") is not None)

    # ④ 链层交接：`carry_latents` **旁挂**在替换后的请求上 ✓（挂错对象会**悄悄丢掉** ✓✗）
    from app.services.engine import chain as chain_mod  # noqa: PLC0415
    from app.services.engine import segments as segments_mod  # noqa: PLC0415
    # ⚠️ 2026-09-26 纠：`plan_segments` 吃的是**总帧数**（int ✓ —— 口径见 `engine_segments_test` ✓）
    #    不是 `GenerationPlan` ✗；且总数必须**比单段上限长**才切得出第 1 段 ✓（原来喂计划对象 ⇒
    #    连两段都切不出来 ✓✗）。取「2×单段 − 重叠」= 两段肩并肩 ✓，且第 0 段的重叠尾巴**正好**
    #    等于 2 ✓ —— 下面链层那条要拿它当钉长 ✓（对不上会被链层的守卫拦 ✓）。
    per_segment = segments_mod.max_grid_length(22)
    segments = segments_mod.plan_segments(2 * per_segment - 2, max_frames=22, overlap_frames=2)
    check("㊸′ 前提：这组参数真的切出**两段** ✓ 且第 0 段重叠尾巴 = 2 ✓（不然下面几条验不到 ✓）",
          len(segments) == 2 and int(segments[0].overlapTail) == 2,
          [plan.to_dict() for plan in segments])
    built = segments_mod.build_segment_requests(
        request, segments, root=root / "segs",
        carryLatents={int(segments[1].index): {"video": tail_v, "audio": tail_a, "frames": 2}})
    check("㊹ ⭐ 续拍锚**旁挂成功了** ✓（``build_segment_requests`` 内部 ``replace`` 造的是**新对象** "
          "✓ —— 挂错对象上就会悄悄丢 ✓✗ ⇒ 这条专钉它 ✓）",
          getattr(built[1], "carry_latents", None) is not None
          and getattr(built[0], "carry_latents", None) is None
          and "续拍头部锚" in str((getattr(built[1], "segmentNote", None) or {}).get("note")),
          getattr(built[1], "carry_latents", None) is not None)
    check("㊺ 同一段**同时**给单帧锚与潜变量锚 ⇒ 报错 ✓（两套锚语义重叠 ✓ 一次只用一套 ✓）",
          _raises(lambda: segments_mod.build_segment_requests(
              request, segments, root=root / "segs2",
              carryFrames={int(segments[1].index): torch.zeros(3, 224, 224)},
              carryLatents={int(segments[1].index): {"video": tail_v, "audio": tail_a,
                                                     "frames": 2}}), "两套锚") is not None)
    fake = {"frames": "x", "latents": {"video": tail_v, "audio": tail_a}}
    nxt = segments          # 同一份计划 ✓（第 0 段的重叠尾巴 = 钉长 2 ✓ 见上面那条前提 ✓）
    carry = chain_mod._continuation_carry(
        fake, nxt[0], edit_mod.continuation_pin(2, fps=24, temporal_compression=1,
                                                audio_latent_mode="round"))
    check("㊻ 链层切出来的尾巴 = 上一段潜变量的**最后**几个槽 ✓（且带上是多少**像素帧** ✓）",
          tuple(carry["video"].shape) == tuple(tail_v.shape)
          and tuple(carry["audio"].shape) == tuple(tail_a.shape)
          and carry["frames"] == 2, {k: tuple(v.shape) if hasattr(v, "shape") else v
                                     for k, v in carry.items()})
    check("㊼ 钉长与分段**重叠尾巴**对不上 ⇒ 报错 ✓（钉的内容与拼掉的内容必须是同一段 ✓）",
          _raises(lambda: chain_mod._continuation_carry(
              fake, nxt[0], edit_mod.continuation_pin(3, fps=24, temporal_compression=1,
                                                      audio_latent_mode="round")), "对不上")
          is not None)
    check("㊽ 上一段 decode **没交回潜变量** ⇒ 报错 ✓（**不退回**单帧锚 ✗ —— 那是另一条产物 ✓）",
          _raises(lambda: chain_mod._continuation_carry(
              {"frames": "x"}, nxt[0],
              edit_mod.continuation_pin(2, fps=24, temporal_compression=1,
                                        audio_latent_mode="round")), "没交回潜变量") is not None)


def case_wiring() -> None:
    """接线守卫：**续拍这条真路**是不是真连着 ✓（静态扫源码 ✓ —— 先例见 `engine_chain_test` 的
    ``case_wiring`` ✓ 与 `cpu_budget_test` 的静态守卫 ✓）。

    ⚠️ 为什么非要有这一条：本轮修掉的那个缺陷**恰恰是"接线看着全在、真路里永不触发"** ✓ ——
    `_dual_carry_pin` 里读的是 ``plan.temporal_compression`` ✗（而 `GenerationPlan` 上**没有**这个
    字段 ✓）⇒ 函数本身写得再对也没用 ✗：**没人把该给的东西递给它** ✓。所以这里**逐环**钉住：
    ``chain`` 切尾巴 ✓ → ``segments`` 旁挂 ✓ → ``pipeline`` 递**同一个 request** ✓ → 后端从 request 读 ✓。
    """
    engine = BACKEND_PY / "app" / "services" / "engine"
    pipeline_src = (engine / "pipeline.py").read_text(encoding="utf-8")
    backend_src = (engine / "torch_backend.py").read_text(encoding="utf-8")
    segments_src = (engine / "segments.py").read_text(encoding="utf-8")
    chain_src = (engine / "chain.py").read_text(encoding="utf-8")
    # ⚠️ 「不许再退回读 plan」那条只扫**代码行** ✓：注释里引用旧写法是**有意留的** ✓（说清改了什么 ✓）
    #    ⇒ 拿全文去比会把它当成违规 ✗（本轮刚写下的那条注释立刻就会被误伤 ✓✗）。
    backend_code = "\n".join(ln for ln in backend_src.splitlines()
                             if not ln.lstrip().startswith("#"))

    def at(src: str, needle: str) -> int:
        """找不到 ⇒ -1 ✓（**不抛** ✓ —— 守卫要的是"红" ✓ 不是"崩" ✓）。"""
        return src.find(needle)

    def reads_plan(code: str) -> bool:
        """那段**代码**里是不是还从 `plan` 上读压缩比 ✓（＝本轮修掉的那个写法 ✓）。"""
        return at(code, 'getattr(plan, "temporal_compression"') >= 0

    check("㊾ 接线①：`pipeline` 把**请求本身**递给双流初始化 ✓（递个剥过的副本 ⇒ `carry_latents` "
          "当场就没了 ✓✗）", "init_dual(plan, request)" in pipeline_src)
    check("㊿ 接线②：后端从 **`request`** 上读 `carry_latents` ✓（从 `plan` 上读 ⇒ 永远读不到 ✗）",
          'getattr(request, "carry_latents", None)' in backend_src)
    check("51 接线③：⭐ 压缩比也从 **`request`** 读 ✓ 且**代码里不许**再退回读 `plan` ✗"
          "（那正是本轮修掉的「永不触发的守卫」✓；注释里留旧写法是**有意**的 ✓ 不算违规 ✓）",
          'temporal_compression=getattr(request, "temporal_compression", None)' in backend_code
          and not reads_plan(backend_code))
    check("52 接线④：`chain` 真的把切出来的尾巴**传下去** ✓（切了不传 = 白切 ✗）",
          at(chain_src, "_continuation_carry(") >= 0
          and at(chain_src, "carryLatents=carry_latents)") >= 0)
    check("53 接线⑤：`segments` 的旁挂**在 `replace` 之后** ✓（挂在前一个对象上 ⇒ 悄悄丢 ✓✗）",
          -1 < at(segments_src, "updated = replace(request, seconds=seconds")
          < at(segments_src, 'object.__setattr__(updated, "carry_latents", pinned)'))
    check("54 接线⑥：续拍模式**不**同时交单帧锚 ✓（两套锚叠着用 ⇒ 产品当场报错 ✓ —— 见 ㊺）",
          "carryFrames={} if continuation is not None else carry," in chain_src)
    from app.services import engine as engine_pkg  # noqa: PLC0415

    check("55 接线⑦：新模块登记进 `engine.__all__` ✓（模块清单是权威 ✓ —— 本轮实测 `h3_edit` "
          "**漏登记**过 ✗ 已补 ✓）", "h3_edit" in engine_pkg.__all__)
    # 负控 ✓ 证上面 51 那条**不是空守卫** ✗（`cpu_budget_test` 那条守卫同款做法 ✓）：
    # 把**旧写法**原样摆回去 ⇒ 同一判据必须**当场为假** ✓（否则"永远为真"的判据等于没判 ✓✗）。
    old_line = '        compression = getattr(plan, "temporal_compression", None)'
    check("56 负控 ✓：51 的判据对**旧写法**必须为假 ✓（旧行摆回去 ⇒ 同一判据当场红 ✓ "
          "—— 不是「怎么改都通过」的空守卫 ✗）",
          reads_plan(old_line) and not reads_plan(backend_code))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="h3_edit_") as tmp:
        root = Path(tmp)
        case_plan_arithmetic()
        case_freeze_semantics(root)
        case_continuation(root)
        case_wiring()
    failed = [name for name, ok, _ in _RESULTS if not ok]
    for name, ok, detail in _RESULTS:
        print(("  OK  " if ok else "  RED ") + name + (f"   <{detail}>" if not ok else ""))
    print(f"\nengine_h3_edit: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} 项通过 ✓")
    if _SKIPS:
        print("跳过：" + "; ".join(_SKIPS))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
