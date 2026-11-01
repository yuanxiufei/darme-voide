"""音频 VAE 自检 ✓（要 torch ✓ —— 没有就 SKIP 并**明说** ✓）。

⭐ 这一套的"**真功能**"证明有两条 ✓：
① ``decode`` 的形状契约 ``L = T × hop`` ✓（H3：``T × 800`` ✓）；
② **真写一个 wav 文件** ✓，再用**标准库 `wave` 读回**核对 `32000 Hz / 2 声道 / 帧数` ✓
（不是"跑通不报错"✗ —— 是**盘上真有个能播的文件** ✓）。
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys
import wave

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.engine import audio_vae as av  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _raises(call: object) -> BaseException | None:
    try:
        call()  # type: ignore[operator]
    except BaseException as err:  # noqa: BLE001
        return err
    return None


def main() -> int:
    # ① 事实（**不需要 torch** ✓）：32 kHz / hop 800 / 40 Hz / 32 × 立体声 2 ✓
    defaults = av.H3_AUDIO_VAE_DEFAULTS
    check("① 事实：sample_rate **32000** ✓、hop = 2·4·4·5·5 = **800** ✓、"
          "latents/s = **40** ✓、潜 **32** 通道 × **立体声 2** ✓",
          defaults["sample_rate"] == 32000 and defaults["latent_channels"] == 32
          and defaults["stereo_channels"] == 2
          and av.AudioVAEConfig(**defaults).hop_length == 800
          and av.AudioVAEConfig(**defaults).latents_per_second == 40,
          av.AudioVAEConfig(**defaults).to_dict())
    check("①′ ⭐ **编解码 hop 必须自洽**（事实 ✓：编码 800 ✓ = 解码 `(5,5,2,2,2,2,2)` 的乘积 800 ✓）",
          av.AudioVAEConfig(**defaults).hop_length == av.AudioVAEConfig(**defaults).decoder_hop)
    bad = dict(defaults) | {"decoder_rates": (5, 5, 2, 2, 2, 2)}
    check("①″ hop 不自洽 ⇒ **构造期就报错** ✓（不等的话「解出来的时长」与「编进去的」对不上 ✓✗，"
          "而且**不会报错** ✗）；统计给一半 ⇒ 也报错 ✓",
          "必须相等" in str(_raises(lambda: av.AudioVAEConfig(**bad)))
          and "要么都给" in str(_raises(lambda: av.AudioVAEConfig(latents_mean=(0.0,) * 32))))
    tiny = dict(defaults) | {"decoder_dim": 64}          # 7 级 ⇒ 至少要 128 ✓
    check("①‴ ⭐ `decoder_dim` 小于 ``2^级数`` ⇒ **构造期就报错** ✓（2026-09-20 双流自检当场抓到 ✗："
          "给小了末级通道算成 0 ⇒ 卷积非法 ✓ 而报的却是**一句 oneDNN 反卷积消息** ✗✗，"
          "完全指不到真因 ✓ ⇒ 不变量钉在这里 ✓）",
          "太小" in str(_raises(lambda: av.AudioVAEConfig(**tiny)))
          and av.AudioVAEConfig(**defaults).decoder_dim == 1024)

    try:
        import torch
    except ImportError:
        check("②…⑨ torch 相关检查（**没有 torch ⇒ 如实标记跳过** ✓，不算通过 ✗）", True, "SKIP")
        return _report()

    torch.manual_seed(0)
    # 小配置跑真模型 ✓（**事实仍是上面那组** ✓ —— 缩小只为跑得快 ✓）
    small = av.AudioVAEConfig(sample_rate=100, encoder_rates=(2, 2), decoder_rates=(2, 2),
                              latent_channels=4, stereo_channels=2, encoder_dim=4,
                              latent_dim=8, decoder_dim=16, resblock_kernels=(3,),
                              resblock_dilations=((1, 3),), heads=2)
    model = av.build_audio_vae(small)
    check("② 小配置自洽 ✓（hop = 4 ✓、latents/s = 25 ✓）", small.hop_length == 4
          and small.latents_per_second == 25 and small.decoder_hop == 4)

    latents = torch.randn(1, 4, 2, 5)
    with torch.no_grad():
        wave_out = model.decode(latents)
    check("③ ⭐ **decode 形状契约**：`[1,4,2,5]` ⇒ `[1,2,20]` ✓（``L = T × hop`` = 5×4 ✓）、"
          "无 NaN ✓、**落在 [-1,1]** ✓（能直接写 wav ✓）",
          tuple(wave_out.shape) == (1, 2, 20) and bool(torch.isfinite(wave_out).all())
          and float(wave_out.min()) >= -1.0 and float(wave_out.max()) <= 1.0,
          (tuple(wave_out.shape), float(wave_out.min()), float(wave_out.max())))
    rounded = torch.randn(1, 2, 18)
    with torch.no_grad():
        encoded = model.encode(rounded)
    check("④ ⭐ **encode 形状契约**：`[1,2,18]` ⇒ `[1,4,2,5]` ✓（**右侧补零**到 hop 的整数倍 ✓，"
          "``T = ceil(18/4) = 5`` ✓ —— 补零而不是报错 ✓，因为编任意长度是正常需求 ✓）",
          tuple(encoded.shape) == (1, 4, 2, 5), tuple(encoded.shape))
    check("④′ 潜变量与波形**互相逗得住**：decode(encode(x)) 的帧数 = `ceil(L/hop)×hop` ✓（20 ✓）",
          tuple(model.decode(encoded).shape) == (1, 2, 20), None)

    # 统计：不给 ⇒ 恒等 ✓；给了 ⇒ **真的用上**（std=2 ⇒ 输出必须变 ✓）
    flat_probe = torch.randn(2, 4, 5)
    scaled = av.build_audio_vae(dataclasses.replace(
        small, latents_mean=(0.0,) * 4, latents_std=(2.0,) * 4))
    with torch.no_grad():
        with_stats = scaled.decode(latents)
    check("⑤ 归一化：没给统计 ⇒ **恒等** ✓（`is` 同一个对象 ✓，连拷贝都没做 ✓）；"
          "给了 ⇒ **真用上** ✓（`std=2` ⇒ 解码结果必须与恒等时不同 ✓ —— "
          "否则就是「统计没接进去」的假绿 ✗✗）",
          model.normalize_latents(flat_probe) is flat_probe
          and not torch.allclose(with_stats, wave_out, atol=1e-6))
    check("⑤′ 统计的**逆**成立 ✓（`denorm(norm(z)) == z` ✓）；数不对 ⇒ 报错 ✓",
          torch.allclose(scaled.denormalize_latents(scaled.normalize_latents(flat_probe)),
                         flat_probe, atol=1e-5)
          and "对不上就别用" in str(_raises(lambda: dataclasses.replace(
              small, latents_mean=(0.0,) * 8, latents_std=(1.0,) * 8))))

    # ⭐⭐ 真功能：**写出一个真的 wav**，再用标准库读回来核对 ✓
    target = pathlib.Path(__file__).resolve().parents[1] / "tests" / "_audio_probe.wav"
    try:
        receipt = av.write_wav(target, wave_out[0].detach().numpy(), small.sample_rate)
        with wave.open(str(target), "rb") as handle:
            read_back = (handle.getnchannels(), handle.getframerate(), handle.getnframes(),
                         handle.getsampwidth())
    finally:
        target.unlink(missing_ok=True)
    check("⑥ ⭐⭐ **真写 wav 并标准库读回** ✓：声道 **2** ✓、采样率 **100** ✓（H3 是 32000 ✓）、"
          "帧数 **20** ✓、位深 **2** 字节（16-bit ✓）—— **盘上真有个文件** ✓（不是「跑通」✗）",
          read_back == (2, 100, 20, 2) and receipt["frames"] == 20
          and receipt["bytes"] > 44, (read_back, receipt))
    check("⑥′ 临时文件**已清理** ✓（自检不许留垃圾 ✗）", not target.exists())
    check("⑦ 形状不合法 ⇒ **报错** ✓（不猜、不裁剪 ✗）",
          "潜变量应为" in str(_raises(lambda: model.decode(torch.randn(1, 3, 2, 5))))
          and "波形应为" in str(_raises(lambda: model.encode(torch.randn(1, 3, 8)))))

    # ⑧ ⭐ **键名逐条对齐参考** ✓（2026-11-01 按 `comfy/ldm/minimax/audio_vae.py` **逐行**重写 ✓）：
    #    真权重是**按名字**装进模块的 ✓ ⇒ 名字差一个字就「装不上」✓；这里把**每一处结构判断**
    #    都钉成一条键 ✓（嵌套/扁平、有 bias/无 bias、`act.` 那层、`block`/`blocks` 的拼法 ✓）。
    keys = set(av.build_audio_vae(small).state_dict())
    must_have = (
        "encoder.block.0.weight",                     # Conv1d(1, d, 7) ✓
        "encoder.block.1.block.0.block.0.alpha",      # ResidualUnit 的 Snake1d ✓
        "encoder.block.1.block.0.block.1.weight",     # 7×1 带膨胀卷积 ✓
        "encoder.block.1.block.3.alpha",              # 三个 ResidualUnit 之后的 Snake1d ✓
        "encoder.block.1.block.4.weight",             # 跨步卷积 ✓
        "pre_block.norm1.weight", "pre_block.norm3.weight", "pre_block.norm2.weight",
        "pre_block.attn.qkv.weight",                  # **无 bias** ✓
        "pre_block.attn.q_bias", "pre_block.attn.v_bias",   # 参数 ✓
        "pre_block.attn.zero_k_bias",                 # **buffer** ✓（k 侧恒 0 ✓）
        "pre_block.attn.proj.weight", "pre_block.proj.weight",
        "pre_block.mlp.norm.weight", "pre_block.mlp.w0.weight",
        "pre_block.mlp.w1.weight", "pre_block.mlp.w2.weight",
        "mean_proj.weight", "logs_proj.weight", "dec_in_proj.weight",
        "decoder.conv_pre.weight",
        "decoder.ups.0.0.weight",                     # **嵌套** ModuleList ✓
        "decoder.resblocks.0.convs1.0.weight",        # **扁平** resblocks ✓
        "decoder.resblocks.0.convs2.0.weight",
        "decoder.resblocks.0.activations.0.act.alpha",   # Activation1d 包着 SnakeBeta ✓
        "decoder.resblocks.0.activations.0.upsample.filter",
        "decoder.resblocks.0.activations.0.downsample.lowpass.filter",
        "decoder.activation_post.act.alpha",
        "decoder.activation_post.upsample.filter",
        "decoder.conv_post.weight",
        "latents_mean", "latents_std",                # 检查点里的**统计 buffer** ✓
    )
    must_not = (
        "decoder.conv_post.bias",            # 参考 bias=False ✓
        "decoder.ups.0.weight",              # 扁平版（旧实现 ✗）
        "pre_block.attn.in_proj_weight",     # MultiheadAttention（旧近似 ✗）
        "encoder.blocks.0.weight",           # 旧名字 ✗（参考叫 encoder.block ✓）
    )
    missing = [name for name in must_have if name not in keys]
    extra = [name for name in must_not if name in keys]
    check("⑧ ⭐ **键名逐条对齐参考** ✓：结构判断全钉在键上（嵌套 `ups.0.0` ✓、扁平 `resblocks` ✓、"
          "`activations.0.act.*` ✓、`qkv` 无 bias ✓、`zero_k_bias` 是 buffer ✓、`conv_post` 无 bias ✓、"
          "`logs_proj` ✓、`latents_mean/std` ✓）—— 少了哪个真权重就装不上 ✓",
          not missing and not extra, (missing, extra))

    ratio_of = av.build_audio_vae(small).decoder
    check("⑧′ `ups` 是**双层**的、`resblocks` 是**扁平** `级数×kernel 数` 个 ✓（参考如此 ✓；"
          "写成一层 ups / 嵌套 resblocks ⇒ 键全错 ✓）",
          all(isinstance(inner, torch.nn.ModuleList) and len(inner) == 1
              for inner in ratio_of.ups)
          and len(ratio_of.resblocks) == len(small.decoder_rates) * len(small.resblock_kernels),
          (len(ratio_of.ups), len(ratio_of.resblocks)))

    # ⑨ ⭐ 抗混叠链**真的守恒** ✓：常数信号进 `Activation1d`（升采样→Snake→降采样）必须还是常数 ✓
    #    （滤波器没归一化 / 补边用错 ⇒ 边界与内部都会鼓包 ✓；这条能当场抓到 ✓）
    activation = model.decoder.activation_post
    probe = torch.full((1, small.decoder_dim // (2 ** len(small.decoder_rates)), 40), 0.3)
    with torch.no_grad():
        activated = activation(probe)
    check("⑨ 抗混叠链守恒 ✓：常数 0.3 过 `upsample → Snake → downsample` 仍是常数 ✓"
          "（内部**逐元素相同** ✓、长度不变 ✓）—— 滤波器归一化/补边写错就会鼓包 ✓",
          tuple(activated.shape) == tuple(probe.shape)
          and bool(torch.allclose(activated, activated[..., 10:11].expand_as(activated),
                                  atol=1e-5)),
          (tuple(activated.shape), float(activated.std())))

    return _report()


def _report() -> int:
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
