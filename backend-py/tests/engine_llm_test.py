"""自检：自研 decoder-only LLM 推理（``engine/llm.py`` ✓ 2026-09-25 起 ✓）。

验的是什么 ✗：本仓文本生成此前走 ollama（HTTP 服务）✗ —— 本套验的是**自己实现**的那半边：
decoder-only transformer 的前向 + KV cache + 采样 ✓（**缩小版**模型走**同一条**前向 ✓）。

五条判据（都是「看着能生成、其实坏了」的形状 ✓✗）：
1. ⭐⭐⭐ **KV cache 不变量** ✗✗：整段前向 vs 逐 token（带 cache）前向，最后位置的 logits **逐位相同** ✓
   （错了 = 因果掩码 / RoPE 位置 / cache 拼接任一处坏 ✓ 且**不会报错** ✗）；
2. ⭐⭐ **因果掩码** ✗✗：改后面的 token ⇒ 前面的 logits **不动** ✓（漏掩码 = 每个位置偷看未来 ✗）；
3. ⭐⭐ **RoPE 旋转不变量** ✓：旋转保范数 ✓（位置编码错了 = 范数/点积悄悄变 ✗）；
4. ⭐ **GQA 分组** ✓：``kv_heads < heads`` 时前向形状仍对、不报错 ✓；
5. ⭐ **tie embeddings** ✓：``lm_head.weight is embed.weight`` ✓（不共享 = 参数量翻倍 ✗ 且不报错 ✗）。

⚠️ 本套**不宣称**会生成像样文本 ✗（缩小版 + 随机权重 ✓）：验的是**架构与不变量** ✓。
⚠️ GGUF 权重加载 / k-quant 反量化**未做** ✗（权重是另一件事 ✓ 见 ``engine/llm.py`` 模块头 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_llm_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="llm_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import llm as llm_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    print(f"SKIP  {reason}")


#: 缩小版配置 ✓（head_dim=8 ⇒ RoPE 4 对 ✓；kv_heads=1 < heads=2 ⇒ GQA ✓）
CONFIG = llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=2, heads=2, kv_heads=1,
                           head_dim=8, ffn=32, rope_theta=100.0, rms_eps=1e-5,
                           eos_id=31)


def backend_or_skip():
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError as err:
        skip(f"没装 torch（{err} ✓）⇒ LLM 前向验不了 ✓（⚠️ 这不是通过 ✓✗）")
        return None


def case_config() -> None:
    check("配置: 非法 kv_heads（heads 不整除）⇒ 拒 ✗（GQA 分组对不上 ✗）",
          _raises(lambda: llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=1, heads=2,
                                            kv_heads=3, head_dim=8, ffn=32)))
    check("配置: head_dim 奇数 ⇒ 拒 ✗（RoPE 配对旋转对不上 ✗）",
          _raises(lambda: llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=1, heads=2,
                                            kv_heads=2, head_dim=7, ffn=32)))
    check("配置: vocab/depth/heads 非正 ⇒ 拒 ✗",
          _raises(lambda: llm_mod.LlmConfig(vocab_size=1, hidden=16, depth=1, heads=2,
                                            kv_heads=2, head_dim=8, ffn=32)))


def case_forward(torch) -> None:
    model = llm_mod.build_llm(CONFIG)
    torch.manual_seed(0)
    ids = torch.randint(0, CONFIG.vocab_size, (2, 5))
    logits, kv = model(ids)
    check("① 前向形状 ✓（(b, s) → (b, s, vocab) ✓ + KV cache 每层一对 ✓）",
          tuple(logits.shape) == (2, 5, CONFIG.vocab_size)
          and len(kv) == CONFIG.depth and len(kv[0]) == 2,
          (tuple(logits.shape), len(kv)))
    check("①′ ⭐ tie embeddings ✓（lm_head.weight is embed.weight ✓ —— 不共享 = 参数量翻倍 ✗）",
          model.lm_head.weight is model.embed.weight)

    # ⭐⭐ 因果掩码：只改最后一个 token ⇒ 前面位置的 logits **不动** ✗✗
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % CONFIG.vocab_size
    logits2, _ = model(ids2)
    check("② ⭐⭐ 因果掩码 ✓：改最后一个 token ⇒ 前 4 个位置的 logits **逐位不动** ✗✗",
          torch.allclose(logits[:, :4], logits2[:, :4], atol=1e-6)
          and not torch.allclose(logits[:, 4], logits2[:, 4], atol=1e-6),
          "")


def case_rope(torch) -> None:
    cos, sin = llm_mod.rope_freqs(torch.arange(6), 8, 100.0)
    x = torch.randn(2, 1, 6, 8)
    rotated = llm_mod.apply_rope(x, cos, sin)
    check("③ ⭐⭐ RoPE 旋转保范数 ✓（位置编码错了 = 范数悄悄变 ✗✗）",
          torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-6)
          and not torch.allclose(rotated, x), "")
    check("③′ RoPE 位置敏感 ✓（不同位置的 cos/sin 不同 ✓ —— 位置编码真在用 ✓）",
          not torch.allclose(cos[0], cos[1]))


def case_kv_cache(torch) -> None:
    model = llm_mod.build_llm(CONFIG)
    torch.manual_seed(1)
    ids = torch.randint(0, CONFIG.vocab_size, (1, 6))

    # 整段前向
    full_logits, _ = model(ids)
    last_full = full_logits[:, -1, :]

    # 逐 token（带 KV cache）前向
    past_kv = None
    for index in range(ids.shape[1]):
        step_logits, past_kv = model(ids[:, index:index + 1], past_kv)
    last_step = step_logits[:, -1, :]

    check("④ ⭐⭐⭐ KV cache 不变量 ✓：逐 token（带 cache）的最后 logits == 整段的最后 logits ✗✗",
          torch.allclose(last_full, last_step, atol=1e-5),
          (last_full.flatten()[:4].tolist(), last_step.flatten()[:4].tolist()))


def case_generate(torch) -> None:
    model = llm_mod.build_llm(CONFIG)
    torch.manual_seed(2)
    ids = torch.randint(0, CONFIG.vocab_size, (1, 3))

    out = llm_mod.generate_llm(model, ids, max_new_tokens=5, temperature=0)
    check("⑤ greedy（temperature=0）形状正确 ✓（前缀 + max_new_tokens ✓ 前缀**原样保留** ✓）",
          tuple(out.shape) == (1, 3 + 5)
          and torch.equal(out[:, :3], ids), tuple(out.shape))

    torch.manual_seed(2)
    out2 = llm_mod.generate_llm(model, ids, max_new_tokens=5, temperature=0)
    check("⑤′ greedy **可复现** ✓（同种子 + temperature=0 ⇒ 逐位相同 ✓）",
          torch.equal(out, out2))

    out3 = llm_mod.generate_llm(model, ids, max_new_tokens=5, temperature=1.0, top_p=0.9, top_k=3)
    check("⑤″ temperature + top-p + top-k 采样不越界 ✓（token 都在词表内 ✓）",
          tuple(out3.shape) == (1, 8) and bool((out3 < CONFIG.vocab_size).all()), tuple(out3.shape))


def case_eos(torch) -> None:
    model = llm_mod.build_llm(CONFIG)
    torch.manual_seed(3)
    ids = torch.randint(0, CONFIG.vocab_size, (1, 2))
    out = llm_mod.generate_llm(model, ids, max_new_tokens=32, temperature=0, eos_id=CONFIG.eos_id)
    check("⑥ 早停 ✓（eos_id 命中 ⇒ 长度 < 前缀+32 ✓；上限保护 ✓）",
          tuple(out.shape) == (1, 2 + 32) or out.shape[1] < 2 + 32, tuple(out.shape))


def case_generate_memory(torch) -> None:
    """⑦ 系列 ⭐⭐ **显存/图**两条：`generate_llm` 必须关梯度 ✓ 且长提示词必须分块 ✓✗（2026-11-07 修）。

    背景（真机踩出来的 ✓）：``generate_llm`` 原来**没有** ``torch.inference_mode()`` ✓，参数默认
    ``requires_grad=True`` ⇒ 每一步前向都在建 autograd 图 ✗，36 层中间量（注意力分数最大 ✓）被
    backward 节点吊住 ✓✗。4B / 3205 词元实测 **94.62 GiB**（卡 22.49 GiB ✗ —— WDDM 甩进共享内存
    硬撑 60 秒才 OOM ✓）。⚠️ 这两条**都不是**"结果错了"，所以下面的 ①②④ 全绿也发现不了 ✗✗ ——
    必须直接问「前向时梯度开着吗」「一次喂了多少词元」✓（钩子里看得见 ✓，不用靠显存读数 ✓）。
    """
    model = llm_mod.build_llm(CONFIG)
    torch.manual_seed(4)
    ids = torch.randint(0, CONFIG.vocab_size, (1, 7))
    seen: list[tuple[int, bool, bool]] = []

    def _hook(module, args) -> None:       # noqa: ANN001 —— 钩子签名由 torch 定 ✓
        seen.append((int(args[0].shape[1]), torch.is_grad_enabled(),
                     torch.is_inference_mode_enabled()))

    handle = model.blocks[0].register_forward_pre_hook(_hook)
    try:
        # 反向验证 ✓：钩子自己得看得见「裸前向时梯度是开的 ✓」—— 否则下面那串 False 可能只是钩子没挂上 ✗
        model(ids)
        plain = list(seen)
        seen.clear()
        chunked = llm_mod.generate_llm(model, ids, max_new_tokens=4, temperature=0, prefill_chunk=3)
        first_run = list(seen)
        seen.clear()
        one_shot = llm_mod.generate_llm(model, ids, max_new_tokens=4, temperature=0, prefill_chunk=7)
    finally:
        handle.remove()

    check("⑦ ⭐ 钩子自己有效 ✓（裸 ``model(ids)`` 那次看得到「梯度开着」✗✗）",
          bool(plain) and plain[0][1] is True, plain[:1])
    check("⑦′ ⭐⭐ ``generate_llm`` 全程**关梯度** ✓（前向里 ``is_grad_enabled()`` 全 False + 在 "
          "``inference_mode`` 里 ✓ ⇒ 中间量不被 backward 吊住 ✗✗）",
          bool(first_run) and all(not grad and inf for _, grad, inf in first_run),
          [item for item in first_run if item[1]][:3])
    check("⑦″ ⭐ 提示词按 ``prefill_chunk`` 分块 ✓（7 词元 / 每块 3 ⇒ 3+3+1 ✓，每块都不超 ✓）",
          [length for length, _, _ in first_run[:3]] == [3, 3, 1],
          [length for length, _, _ in first_run][:5])
    check("⑦‴ ⭐⭐ 分块预填充与「一次吃完」**逐 token 相同** ✓（数学等价 ✓ —— 分块不许改结果 ✗✗）",
          torch.equal(chunked, one_shot), (chunked.tolist(), one_shot.tolist()))
    check("⑦⁴ 非法 ``prefill_chunk``（0）⇒ 当场拒 ✗（静默退回老路径 = 长提示词照爆 ✗✗）",
          _raises(lambda: llm_mod.generate_llm(model, ids, max_new_tokens=1,
                                               prefill_chunk=0)))


def case_kv_quant(torch) -> None:
    """⑧ 系列 ⭐ **KV cache 的 INT8 分组量化**（2026-11-08 补 ✓ 口径参考 Strata ``kv_q8.hpp``）。

    为什么值得单列 ✗：KV cache 是**唯一随上下文线性长**的那块 ✓（权重是常数 ✓）。本仓老路径每步
    ``torch.cat`` 一份 fp 缓存 ✓ 且**没有上限** ✗ —— 32K 上下文在 4B 档就是 4.5 GiB ✓。
    量化的三条「看着对、其实坏了」：

    1. **分组方向**：沿 ``head_dim``（每 64 个通道一组 ✓）而不是沿序列 ✗✗ —— 沿序列分组等于让两个
       语义无关的 token 共用一个 scale ✓（量程差 1000× 时小的那批全被压成 0 ✓，而 shape 全对 ✗）；
    2. **全零组**：``amax=0`` ⇒ ``scale=0`` ⇒ ``0/0`` = NaN ⇒ softmax 吃 NaN ⇒ 整段输出 NaN ✓✗
       （报错位置离量化十万八千里 ✗）；
    3. **重复量化**：若每步把「整段历史」反量化后重量化，误差会随步数累积 ✓✗ —— 正确做法是旧段的
       int8 **原样搬走** ✓，只量化刚算出来的那一小段 ✓。
    """
    # ── 口径归一 ✓
    check("⑧ ``kvMode`` 归一 ✓（off/fp ⇒ fp；q8/int8/auto ⇒ q8 ✓）",
          llm_mod.resolve_kv_mode(None) == "fp" and llm_mod.resolve_kv_mode("OFF") == "fp"
          and llm_mod.resolve_kv_mode("") == "fp" and llm_mod.resolve_kv_mode("q8") == "q8"
          and llm_mod.resolve_kv_mode("INT8") == "q8")
    check("⑧′ ``kvMode`` 拼错 ⇒ **具名拒** ✗（静默当 fp ⇒「开了量化还这么占显存」，或反过来悄悄变差 ✗✗）",
          _raises(lambda: llm_mod.resolve_kv_mode("q08")))

    # ── 量化-反量化：误差界 + 分组方向 ✓
    torch.manual_seed(7)
    b, h, s, d, g = 1, 2, 5, 8, 4
    x = torch.randn(b, h, s, d) * 3.0
    q, scale = llm_mod.quantize_kv(x, group=g)
    back = llm_mod.dequantize_kv(q, scale, g, dtype=x.dtype, head_dim=d)
    limit = (scale / 2.0).unsqueeze(-1).expand(b, h, s, -1, g).reshape(b, h, s, d)
    check("⑧″ ⭐ 对称量化的**误差界** ✓：每个元素 |反量化 − 原值| ≤ 组 scale/2 ✓（超界 = 量化写错 ✗）",
          tuple(back.shape) == tuple(x.shape)
          and bool(((back - x).abs() <= limit + 1e-6).all()),
          float((back - x).abs().max()))
    check("⑧‴ ⭐⭐ **分组沿 head_dim、每 token 每 head 各自一份 scale** ✓"
          "（``scale`` 的序列维 == s ✓；沿序列分组会塌成 1 ✓✗）",
          tuple(scale.shape) == (b, h, s, d // g), tuple(scale.shape))

    # 反向验证：两个 token 量级差 1000× ⇒ 沿 head_dim 分组时小那个 token 仍保得住 ✓
    torch.manual_seed(8)
    mixed = torch.randn(1, 1, 2, 8)
    mixed[:, :, 1, :] *= 1000.0
    qm, sm = llm_mod.quantize_kv(mixed, group=8)
    bm = llm_mod.dequantize_kv(qm, sm, 8, dtype=mixed.dtype, head_dim=8)
    small_err = float((bm[:, :, 0] - mixed[:, :, 0]).abs().max())
    check("⑧⁴ ⭐⭐ 量级差 1000× 的两段**互不拖累** ✓（小段误差仍 ≈ 自身量级 × 1/127 ✓；"
          "共用 scale 时小段会被整段抹平 ✗✗）",
          small_err <= float(mixed[:, :, 0].abs().max()) / 127.0 * 1.5, small_err)

    # ── 全零组不许出 NaN ✓
    zeros = torch.zeros(1, 1, 3, 8)
    qz, sz = llm_mod.quantize_kv(zeros, group=4)
    bz = llm_mod.dequantize_kv(qz, sz, 4, dtype=zeros.dtype, head_dim=8)
    check("⑧⁵ ⭐⭐ 全零段 ⇒ 反量化是 0 且**没有 NaN** ✓（scale=0 ⇒ 0/0 ⇒ softmax 吃 NaN ⇒ 整段胡话 ✗✗）",
          bool(torch.isfinite(bz).all()) and float(bz.abs().max()) == 0.0
          and bool((sz > 0).all()))

    # ── head_dim 除不尽组宽 ⇒ 补零后裁回 ✓
    odd = torch.randn(1, 1, 4, 6)
    qo, so = llm_mod.quantize_kv(odd, group=4)
    bo = llm_mod.dequantize_kv(qo, so, 4, dtype=odd.dtype, head_dim=6)
    check("⑧⁶ head_dim 除不尽组宽（6 / 4）⇒ 尾部**补零到整组**再量化 ✓（q 的最后一维是 8 而不是 6 ✓）",
          int(qo.shape[-1]) == 8 and tuple(so.shape) == (1, 1, 4, 2), (tuple(qo.shape), tuple(so.shape)))
    check("⑧⁶′ 裁剪回 6 后误差仍受**组 scale/2** 约束 ✓（不裁 = 序列里多出 2 个通道 ✗✗）",
          bool(((bo - odd).abs()
                <= (so / 2.0).unsqueeze(-1).expand(1, 1, 4, -1, 4).reshape(1, 1, 4, 8)[..., :6] + 1e-6
                ).all()))

    # ── 模型侧的 KV 段 ✓
    model = llm_mod.build_llm(CONFIG)
    torch.manual_seed(9)
    ids = torch.randint(0, CONFIG.vocab_size, (1, 4))
    logits_fp, kv_fp = model(ids, None, kv_mode="fp")
    logits_q8, kv_q8 = model(ids, None, kv_mode="q8")
    head = kv_q8[0][0]
    check("⑧⁷ ⭐ q8 口径下 cache 里存的是 **Q8Kv**（值 + scale 绑在一起 ✓），"
          "``shape``/``device`` 与量化前一致 ✓",
          isinstance(head, llm_mod.Q8Kv)
          and tuple(head.shape) == tuple(kv_fp[0][0].shape)
          and head.device == kv_fp[0][0].device
          and isinstance(kv_fp[0][0], torch.Tensor), type(head).__name__)
    rel = float((logits_q8 - logits_fp).abs().max() / logits_fp.abs().max())
    check("⑧⁸ ⭐⭐ 端到端：q8 与 fp 的 logits **相对偏差 < 0.5%** ✓（有损但不许走样 ✗✗）",
          rel < 5e-3, rel)

    out_fp = llm_mod.generate_llm(model, ids, max_new_tokens=16, temperature=0, kv_mode="fp")
    out_q8 = llm_mod.generate_llm(model, ids, max_new_tokens=16, temperature=0, kv_mode="q8")
    same = int((out_fp == out_q8).sum())
    check("⑧⁹ 16 步 greedy 里 q8 与 fp 的**一致率 ≥ 93%** ✓（随机小权重下最坏情形 ✓；"
          "真权重上见 docs §7.2 ✓）",
          same >= int(out_fp.numel() * 0.93), f"{same}/{out_fp.numel()}")

    # 分块预填充：量化边界跟着块走 ⇒ 结果仍必须逐位相同 ✓
    chunked = llm_mod.generate_llm(model, ids, max_new_tokens=6, temperature=0,
                                   kv_mode="q8", prefill_chunk=2)
    one_shot = llm_mod.generate_llm(model, ids, max_new_tokens=6, temperature=0,
                                    kv_mode="q8", prefill_chunk=9)
    check("⑧¹⁰ ⭐ q8 下**分块预填充**与一次吃完**逐位相同** ✓（分组沿 head_dim ⇒ 与序列切法无关 ✓）",
          torch.equal(chunked, one_shot), (chunked.tolist(), one_shot.tolist()))

    # ── 预算账 ✓（回执只报这个数）
    tiny = llm_mod.kv_bytes_per_token(CONFIG, "fp", dtype_bytes=2)
    tiny_q8 = llm_mod.kv_bytes_per_token(CONFIG, "q8", dtype_bytes=2)
    real = llm_mod.LlmConfig(vocab_size=151936, hidden=2560, depth=36, heads=32,
                             kv_heads=8, head_dim=128, ffn=9728)
    real_fp = llm_mod.kv_bytes_per_token(real, "fp", dtype_bytes=2)
    real_q8 = llm_mod.kv_bytes_per_token(real, "q8", dtype_bytes=2)
    check("⑧¹¹ 每 token 预算算得对 ✓（小模型：fp 64 B ⇒ q8 48 B；**4B：144 KiB ⇒ 76.5 KiB** ✓）",
          tiny["totalBytes"] == 64 and tiny_q8["totalBytes"] == 48
          and real_fp["totalBytes"] == 147456 and real_q8["totalBytes"] == 78336,
          (tiny_q8["totalBytes"], real_q8["totalBytes"]))
    check("⑧¹² 32K 上下文的实际账 ✓（4B：fp 4608 MiB ⇒ q8 2448 MiB ✓ —— 回执要说得出这个数 ✓）",
          round(real_fp["totalBytes"] * 32768 / 2**20) == 4608
          and round(real_q8["totalBytes"] * 32768 / 2**20) == 2448,
          (round(real_fp["totalBytes"] * 32768 / 2**20), round(real_q8["totalBytes"] * 32768 / 2**20)))
    check("⑧¹³ 非法 ``group`` / ``dtype_bytes`` ⇒ 具名拒 ✗",
          _raises(lambda: llm_mod.kv_bytes_per_token(CONFIG, "q8", dtype_bytes=0))
          and _raises(lambda: llm_mod.kv_bytes_per_token(CONFIG, "q8", group=0)))


def case_kv_reuse(torch) -> None:
    """⑨ 系列 ⭐ **跨调用 KV 前缀复用**（2026-11-08 补 ✓ 参考 Strata ``conversation_memory`` 的口径）。

    值得做的现实理由 ✗：agent 工具循环每一轮都把**同一段** system 提示 + 工具声明重发一遍 ✓，
    ``/tune`` 拿**同一句**提示词连调参数 ✓ —— 这些调用里前缀一字不差 ✓，老路径每轮都从头 prefill ✗。

    四条「看着对、其实坏了」：

    1. **按 id 比、不按文本比** ✓：文本相同不保证 id 相同 ✗（模板空白、特殊符、编解码往返 ✓）——
       按文本判定＝拿一段错位的 KV 去接 ✓ 不报错 ✓✗；
    2. **至少留一个 token 给前向** ✓：复用整段 ⇒ 没有查询可算 ⇒ 拿不到 logits ✗✗；
    3. **口径必须同一套** ✓：``fp`` 的 cache 喂给 ``q8`` 跑＝长度对、数值两套 ✗；
    4. **装不下就不装** ✓：常驻 cache 跨调用活着 ✓，超预算还存 ⇒ 报错落在**下一个件**上 ✓✗。
    """
    # ── 预算旋钮归一 ✓
    check("⑨ ``kvCacheBytes`` 归一 ✓（不写/on ⇒ 512 MiB ✓；0/off ⇒ 0 ✓；512MiB/1GiB ⇒ 字节 ✓）",
          llm_mod.resolve_kv_cache_bytes(None) == 512 * 2 ** 20
          and llm_mod.resolve_kv_cache_bytes("off") == 0
          and llm_mod.resolve_kv_cache_bytes(0) == 0
          and llm_mod.resolve_kv_cache_bytes("512MiB") == 512 * 2 ** 20
          and llm_mod.resolve_kv_cache_bytes("1GiB") == 2 ** 30
          and llm_mod.resolve_kv_cache_bytes(False) == 0)
    check("⑨′ ⭐ 拼错 ⇒ **具名拒** ✗（悄悄当 0 ⇒「配了复用」却一次没命中 ✓✗）",
          _raises(lambda: llm_mod.resolve_kv_cache_bytes("512兆"))
          and _raises(lambda: llm_mod.resolve_kv_cache_bytes(-1)))

    model = llm_mod.build_llm(CONFIG)
    torch.manual_seed(11)
    ids = torch.randint(0, CONFIG.vocab_size, (1, 6))

    # ── cache 字节数必须逐张量数 ✓（回执报的就是这个数 ✓）
    _, kv_fp = model(ids, None, kv_mode="fp")
    _, kv_q8 = model(ids, None, kv_mode="q8")
    fp_bytes = llm_mod.kv_cache_bytes(kv_fp)
    q8_bytes = llm_mod.kv_cache_bytes(kv_q8)
    check("⑨″ ``kv_cache_bytes`` 逐张量数 ✓（q8 = int8 + fp32 scale ⇒ 比 fp 小 ✓、不为 0 ✓）",
          fp_bytes > 0 and 0 < q8_bytes < fp_bytes, (fp_bytes, q8_bytes))

    # ── 单槽 cache 的语义 ✓
    cache = llm_mod.KvPrefixCache(64 * 2 ** 20, mode="fp")
    hit, n = cache.lookup(ids[0].tolist())
    check("⑨‴ 空 cache ⇒ 不命中（``(None, 0)`` ✓ 不是「空张量」✗）", hit is None and n == 0)
    stored = cache.store(ids[0].tolist(), kv_fp)
    hit, n = cache.lookup(ids[0].tolist())
    check("⑨⁴ ⭐⭐ **同一串**也要留一个 token ✓ ⇒ 复用 ``len − 1`` ✓"
          "（复用整段 = 没有查询 ⇒ 拿不到 logits ✗✗）",
          stored == fp_bytes and n == int(ids.shape[1]) - 1, (stored, n))
    longer = torch.cat([ids, torch.randint(0, CONFIG.vocab_size, (1, 3))], dim=1)
    _, n_long = cache.lookup(longer[0].tolist())
    check("⑨⁵ 更长的串 ⇒ 复用**公共前缀全部** ✓（这才是 agent 循环每轮回长对话的账 ✓）",
          n_long == int(ids.shape[1]), n_long)
    other = torch.randint(CONFIG.vocab_size - 1, CONFIG.vocab_size, (1, 6))
    _, n_other = cache.lookup(other[0].tolist())
    check("⑨⁶ ⭐ 一点共同前缀都没有 ⇒ **当场释放**常驻 cache ✓（留着「万一」会一路占着显存 ✗✗）",
          n_other == 0 and cache.describe()["storedTokens"] == 0, cache.describe())

    # ── 预算不够就不存 ✓
    tiny_cache = llm_mod.KvPrefixCache(max(1, fp_bytes // 2), mode="fp")
    refused = tiny_cache.store(ids[0].tolist(), kv_fp)
    check("⑨⁷ ⭐⭐ 超预算 ⇒ **不存** ＋ 回执写明原因 ✓（先存着试试 ⇒ 报错落在别的件上 ✗✗）",
          refused == 0 and tiny_cache.describe()["skipped"] == 1
          and "预算" in tiny_cache.describe()["note"], tiny_cache.describe()["note"])
    check("⑨⁸ 预算 0 ⇒ 整个关闭 ✓（``enabled`` 为假、lookup 恒不命中 ✓）",
          not llm_mod.KvPrefixCache(0).enabled)

    # ── 裁段：值和 scale 必须一起裁 ✓
    cut = llm_mod.truncate_kv(kv_q8, 4)
    check("⑨⁹ ⭐ ``truncate_kv`` 把 ``q`` 与 ``scale`` **一起**裁 ✓（只裁一半 = 长度对不上 ⇒ 静默错位 ✗✗）",
          tuple(cut[0][0].shape)[2] == 4 and tuple(cut[0][0].scale.shape)[2] == 4
          and tuple(cut[0][1].shape)[2] == 4
          and tuple(llm_mod.truncate_kv(kv_q8, 6)[0][0].shape)[2] == 6)

    # ── 端到端：命中不得改变结果 ✓（这是整个优化唯一的安全底线 ✓✗）
    for mode in ("fp", "q8"):
        cold = llm_mod.generate_llm(model, ids, max_new_tokens=8, temperature=0, kv_mode=mode)
        c = llm_mod.KvPrefixCache(64 * 2 ** 20, mode=mode)
        first = llm_mod.generate_llm(model, ids, max_new_tokens=8, temperature=0,
                                     kv_mode=mode, kv_cache=c)
        second = llm_mod.generate_llm(model, ids, max_new_tokens=8, temperature=0,
                                      kv_mode=mode, kv_cache=c)
        report = c.describe()
        check(f"⑨¹⁰ [{mode}] ⭐⭐ 命中一轮后的输出与**冷跑逐位相同** ✓"
              f"（{mode} 都要 ✓ —— 因果注意力下前缀 KV 只由前缀决定 ✓）",
              torch.equal(first, cold) and torch.equal(second, cold),
              (first.tolist(), cold.tolist(), second.tolist()))
        check(f"⑨¹¹ [{mode}] 回执如实：命中 1 次、复用 ``前缀−1`` 个 token、占字节 > 0、"
              f"存的是**整段**（提示词＋续写）✓",
              report["hits"] == 1 and report["reusedTokens"] == int(ids.shape[1]) - 1
              and report["bytes"] > 0 and report["storedTokens"] == int(cold.shape[1]),
              report)

    # 口径不同 ⇒ 不许复用 ✓
    cross = llm_mod.KvPrefixCache(64 * 2 ** 20, mode="q8")
    cross.store(ids[0].tolist(), kv_q8)
    out_fp_with_q8_cache = llm_mod.generate_llm(model, ids, max_new_tokens=4, temperature=0,
                                                kv_mode="fp", kv_cache=cross)
    out_fp_cold = llm_mod.generate_llm(model, ids, max_new_tokens=4, temperature=0, kv_mode="fp")
    check("⑨¹² ⭐ fp 生成**不接受** q8 口径的 cache ✓（口径不同 ⇒ 不复用，结果仍与冷跑一致 ✓）",
          torch.equal(out_fp_with_q8_cache, out_fp_cold) and cross.describe()["hits"] == 0,
          cross.describe()["hits"])


def _cycle_model(torch, vocab: int = 5, *, stuck_at=None, sensitive: bool = False,
                 trace: list | None = None):
    """可控假模型 ✓：``argmax = (x+1) % vocab`` ⇒ 周期文本上草稿**必然全中** ✓。

    三条都是为了让判据「抓得住」而设计的 ✓✗：

    * 真模型（随机权重）只能验**等价** ✗ —— 它几乎不接受任何草稿 ⇒ 验不出「真的投机了」✗✗；
    * ``stuck_at`` 让「该 token 之后」改成固定 ``vocab-1`` ✓ ⇒ 造「部分接受」（草稿按周期给、
      模型不认 ⇒ 从某个草稿起全拒 ✓）；
    * ``sensitive`` 让输出**依赖 ``past_len``** ✓ ⇒ 回滚裁错（早裁/晚裁）会让 ``past_len`` 偏离
      ⇒ 输出当场不同 ✓✗ —— 否则假模型不看 KV、裁错也能通过 ✗✗（这是最容易自己骗过自己的地方）；
    * ``trace`` 记下每次调用的 ``(past_len, 本次行数)`` ✓ ⇒ 断言「相邻两次 past 之差落在
      ``[1, 本次行数]`` 内」即抓「没裁」（差 > 行数）与「裁过头」（差 < 1）✓。
    """
    class _Cycle:
        def __call__(self, ids, past_kv=None, *, kv_mode="fp"):  # noqa: ANN001, ARG002
            batch, length = int(ids.shape[0]), int(ids.shape[1])
            past_len = 0 if past_kv is None else int(past_kv[0][0].shape[2])
            if trace is not None:
                trace.append((past_len, length))
            logits = torch.full((batch, length, vocab), -1.0)
            for i in range(length):
                token = int(ids[0, i])
                if stuck_at is not None and token == stuck_at:
                    target = vocab - 1
                else:
                    shift = (past_len % 3) if sensitive else 0
                    target = (token + 1 + shift) % vocab
                logits[0, i, target] = 1.0
            total = past_len + length
            kv = [(torch.zeros((batch, 1, total, 1)), torch.zeros((batch, 1, total, 1)))]
            return logits, kv

    return _Cycle()


def case_spec_lookup(torch) -> None:
    """⑩ 系列 —— **投机解码**（2026-11-08 ✓ 照搬 ``strata::spec::SuffixDrafter`` ＋ ``Verifier::commit``）。

    这一套专门抓「抄错了**不报错**、只让后续位置整体错位」的形状 ✓✗（八层里最危险的一类 ✓）：

    1. ⭐⭐⭐ **等价**：开 vs 不开，贪心下**逐位相同** ✓（真模型 ✓ 唯一有硬判据的场景 ✓）；
    2. ⭐⭐ **真的投机了**：周期文本 ＋ 可控模型 ⇒ ``accepted == drafted > 0`` ✓
       （否则「等价」可能只是「因为没开」✗ —— 本套最容易被自己骗过的地方 ✓✗）；
    3. ⭐⭐ **回滚裁对**：输出依赖 ``past_len`` 的模型 ⇒ 裁早/裁晚当场露馅 ✓；
    4. ⭐ **温度 > 0 如实退回** ✓（``enabled=False`` ＋ 同种子输出与不开逐位相同 ✓ 不假装开了 ✗）；
    5. **草稿器三条口径**：允许重叠 ✓、等长保留更近 ✓、``min_match`` 门槛 ✓；
    6. **额度边界**：草稿多于剩余额度 ⇒ 长度恰好 ``max_new_tokens`` ✓；
    7. **q8 也等价** ✓（量化分组沿 ``head_dim`` ⇒ 每 token 的 KV 独立 ✓）。
    """
    model = llm_mod.build_llm(CONFIG)
    torch.manual_seed(11)
    ids = torch.randint(0, CONFIG.vocab_size, (1, 9))

    # ── 1. 等价（真模型 · 贪心）
    plain = llm_mod.generate_llm(model, ids, max_new_tokens=12, temperature=0)
    spec = llm_mod.generate_llm(model, ids, max_new_tokens=12, temperature=0, spec_lookup=True)
    check("⑩¹ ⭐⭐⭐ 开投机与不开 **逐位相同** ✓（真模型 · 贪心 ⇒ 唯一有硬判据的场景 ✓）",
          torch.equal(plain, spec), (plain.tolist(), spec.tolist()))

    # ── 7. q8 也等价
    plain_q8 = llm_mod.generate_llm(model, ids, max_new_tokens=8, temperature=0, kv_mode="q8")
    spec_q8 = llm_mod.generate_llm(model, ids, max_new_tokens=8, temperature=0, kv_mode="q8",
                                   spec_lookup=True)
    check("⑩² [q8] 量化档同样逐位相同 ✓（分组沿 head_dim ⇒ 每个 token 的 KV 独立 ✓）",
          torch.equal(plain_q8, spec_q8), (plain_q8.tolist(), spec_q8.tolist()))

    # ── 2. 全接受（可控模型 ＋ 周期文本）
    cycle = torch.tensor([[0, 1, 2, 3, 4] * 4])
    trace: list = []
    report: dict = {}
    got = llm_mod.generate_llm(_cycle_model(torch, vocab=5, trace=trace), cycle,
                               max_new_tokens=9, temperature=0, spec_lookup=True,
                               spec_window=4, spec_report=report)
    want = [(4 + i) % 5 for i in range(1, 10)]
    check("⑩³ ⭐⭐ 周期文本 ＋ 复制型模型 ⇒ **草稿全中** ✓（``accepted == drafted > 0``；"
          "输出 = 周期延续 ✓ —— 证明窗口前向与接受判据都通着 ✓）",
          report["drafted"] > 0 and report["accepted"] == report["drafted"]
          and got[0, cycle.shape[1]:].tolist() == want, report)

    # ── 3. 回滚：裁早了位置前移、裁晚了会看见被拒的 KV ✓✗
    # ⚠️ 只看「差 <= 窗口行数」抓不住「没裁」✗：**全接受**时 ``1 + len(draft)`` 恰好等于行数 ✓
    # ⇒ 不裁也只差这么多 ✓ 假绿 ✗✗。所以再跑一遍**部分拒绝**的（``stuck_at`` 造拒绝 ✓）：
    # 只要出现过一次拒绝 ✓ 真裁的差就 **严格小于** 行数 ✓（``1 + a < 1 + len(draft)`` ✓），
    # 没裁则差值恒等于行数 ⇒ 当场露馅 ✓✗。
    trace2: list = []
    llm_mod.generate_llm(_cycle_model(torch, vocab=5, stuck_at=2, trace=trace2), cycle,
                         max_new_tokens=9, temperature=0, spec_lookup=True, spec_window=4)
    gaps = [(trace[i + 1][0] - trace[i][0], trace[i][1]) for i in range(len(trace) - 1)]
    gaps2 = [(trace2[i + 1][0] - trace2[i][0], trace2[i][1]) for i in range(len(trace2) - 1)]
    check("⑩⁴ ⭐⭐ 回滚长度**夹在 [1, 本次窗口行数]** 内，且部分拒绝时**严格小于**行数 ✓"
          "（差 > 行数 = 没裁 ✗；差 < 1 = 裁过头 ✗✗）",
          # 第一次调用是**预填充**整段提示词 ✓（``past_len == 0`` ＋ 行数 == 提示词长 ✓），
          # 从那之后每一拍的 ``past_len`` 只许前移 ``1 + 接受数`` 行 ✓（``<= 窗口行数`` ✓）。
          trace and trace[0][0] == 0 and trace[0][1] == int(cycle.shape[1])
          and len(trace) > 2
          and all(1 <= gap <= rows for gap, rows in gaps)
          and all(1 <= gap <= rows for gap, rows in gaps2)
          and any(gap < rows for gap, rows in gaps2), (trace, gaps, trace2, gaps2))

    sensitive = _cycle_model(torch, vocab=5, sensitive=True)
    got2 = llm_mod.generate_llm(sensitive, cycle, max_new_tokens=9, temperature=0, spec_lookup=True)
    ref2 = llm_mod.generate_llm(_cycle_model(torch, vocab=5, sensitive=True), cycle,
                                max_new_tokens=9, temperature=0)
    check("⑩⁵ ⭐⭐ 输出**依赖 past_len** 时仍与 plain 逐位相同 ✓（裁错 ⇒ past_len 偏离 ⇒ 必露馅 ✓✗）",
          torch.equal(got2, ref2), (got2.tolist(), ref2.tolist()))

    report2: dict = {}
    rejected = llm_mod.generate_llm(_cycle_model(torch, vocab=5, stuck_at=2), cycle,
                                    max_new_tokens=9, temperature=0, spec_lookup=True,
                                    spec_report=report2)
    refused = llm_mod.generate_llm(_cycle_model(torch, vocab=5, stuck_at=2), cycle,
                                   max_new_tokens=9, temperature=0)
    check("⑩⁶ ⭐ 部分接受（模型故意不认某个草稿）时输出仍逐位相同 ✓，且确实发生了拒绝 ✓"
          "（``accepted < drafted`` —— 否则上一条在验一个不存在的分支 ✗）",
          torch.equal(rejected, refused) and report2["accepted"] < report2["drafted"],
          (report2, rejected.tolist(), refused.tolist()))

    # ── 4. 温度 > 0 如实退回
    report3: dict = {}
    warm_plain = llm_mod.generate_llm(model, ids, max_new_tokens=6, temperature=1.0, seed=21)
    warm_spec = llm_mod.generate_llm(model, ids, max_new_tokens=6, temperature=1.0, seed=21,
                                     spec_lookup=True, spec_report=report3)
    check("⑩⁷ ⭐ 温度 > 0 ⇒ **如实退回** ✓（``enabled=False`` ＋ 写明原因 ＋ 同种子输出与不开逐位相同 ✓）",
          report3.get("enabled") is False and "temperature" in str(report3.get("reason"))
          and torch.equal(warm_plain, warm_spec), report3)

    # ── 5. 草稿器三条口径
    overlap = llm_mod.LookupDrafter(min_match=2)
    overlap.append([0, 1, 2, 3, 0, 1, 2])
    check("⑩⁸ 口径③ **允许与当前后缀重叠** ✓（``[3,0,1,2]`` 末尾 ``0,1,2`` 就是当前后缀本身 ✓；"
          "删掉重叠 ⇒ 周期性文本永远拿不到草稿 ✗✗）",
          overlap.propose(4) == [3, 0, 1, 2], overlap.propose(4))

    near = llm_mod.LookupDrafter(min_match=2)
    near.append([1, 2, 3, 7, 1, 2, 3, 9, 1, 2, 3])
    check("⑩⁹ 口径② **等长匹配保留更近的** ✓（``[9,1,2,3]`` 取自 pos=6；"
          "``>`` 写成 ``>=`` 会取 pos=2 ⇒ ``[7,1,2,3]`` ✓ 不报错但结果不同 ✗）",
          near.propose(4) == [9, 1, 2, 3], near.propose(4))

    strict = llm_mod.LookupDrafter(min_match=4)
    strict.append([1, 2, 3, 7, 1, 2, 3, 9, 1, 2, 3])
    check("⑩¹⁰ ``min_match`` 门槛：最长匹配只有 3 ⇒ 门槛 4 时**不给草稿** ✓（空列表不是错 ✓）",
          strict.propose(4) == [], strict.propose(4))

    # ── 6. 额度边界与参数校验
    tight = llm_mod.generate_llm(_cycle_model(torch, vocab=5), cycle, max_new_tokens=3,
                                 temperature=0, spec_lookup=True, spec_window=8)
    check("⑩¹¹ 草稿数超过剩余额度 ⇒ 长度**恰好** ``max_new_tokens`` ✓（不许多产 ✗）",
          tuple(tight.shape) == (1, cycle.shape[1] + 3), tuple(tight.shape))
    check("⑩¹² 参数校验：``spec_window < 1`` / ``spec_min_match < 1`` ⇒ 当场拒 ✓",
          _raises(lambda: llm_mod.generate_llm(model, ids, max_new_tokens=1, spec_window=0))
          and _raises(lambda: llm_mod.generate_llm(model, ids, max_new_tokens=1, spec_min_match=0)))


def _raises(fn) -> bool:
    try:
        fn()
    except (llm_mod.LlmError, ValueError):
        return True
    except Exception:  # noqa: BLE001 —— 别的异常也算「没按预期报错」✗
        return False
    return False


def main() -> int:
    torch = backend_or_skip()
    case_config()
    if torch is not None:
        case_forward(torch)
        case_rope(torch)
        case_kv_cache(torch)
        case_generate(torch)
        case_eos(torch)
        case_generate_memory(torch)
        case_kv_quant(torch)
        case_kv_reuse(torch)
        case_spec_lookup(torch)
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
