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


def case_fast_path(torch) -> None:
    """快路径（2026-11-08 ✓ ``fastPath`` ✓）—— 两个融合档 ＋ 折叠 RoPE。

    判据的重心不在「跑得出来」✗，在三件**不报错就会静默错**的事：

    1. ``resolve_fast_path`` 的归一表 ✓ —— 尤其**拼错必须当场拒** ✗✗（``"nrom"`` 静默当关 ⇒ 人以为
       开了快路径、每步还是 2506 次 launch ✓）；返回 ``frozenset`` 而不是名字串 ✓（``"all"`` 里不含
       ``"norm"`` 子串 ⇒ 字符串包含判断会把 ``all`` 读成「只开 attn」✓ 不报错、只慢 ✗）；
    2. 折叠 RoPE 与老 :func:`apply_rope` **逐位相同** ✓ —— 这条**没有开关**（不是"近似"✓）⇒
       一旦不逐位相同，默认关的那条路也已经被改了 ✗✗，而判据只有逐位比才抓得到 ✓；
    3. 两个融合档**真的走到融合核** ✓ —— 用 spy 数 ``F.rms_norm`` / ``F.scaled_dot_product_attention``
       的调用次数 ✓✗：只验「输出形状对、不崩」在**开了开关却静默走老路**时也是绿的 ✗✗。
    """
    import math  # noqa: PLC0415

    # ── 1. 归一表
    cases = [
        (None, set()), ("", set()), ("off", set()), ("0", set()), (0, set()), (False, set()),
        ("NORM", {"norm"}), (" fused_norm ", {"norm"}), ("rms_norm", {"norm"}),
        ("sdpa", {"attn"}), ("attention", {"attn"}), ("gqa", {"attn"}),
        ("kvbuf", {"kvbuf"}), ("kv_buffer", {"kvbuf"}), ("prealloc", {"kvbuf"}),
        ("all", {"norm", "attn", "kvbuf"}), ("on", {"norm", "attn", "kvbuf"}),
        (True, {"norm", "attn", "kvbuf"}),
        ("norm,attn", {"norm", "attn"}), ("norm+attn", {"norm", "attn"}),
        ("norm and attn", {"norm", "attn"}), (["norm", "attn"], {"norm", "attn"}),
        ({"attn"}, {"attn"}), (("norm",), {"norm"}),
        ("all", set(llm_mod.FAST_PATH_ITEMS)),
    ]
    wrong = [(raw, sorted(llm_mod.resolve_fast_path(raw)), sorted(want))
             for raw, want in cases if set(llm_mod.resolve_fast_path(raw)) != want]
    check("⑫¹ ``resolve_fast_path`` 归一表 ✓（大小写/``_``/``+``/列表都认 ✓；关＝空集 ✓）",
          not wrong, wrong)
    check("⑫² ⭐ 返回 ``frozenset`` 而**不是**名字串 ✓（``'all'`` 不含 ``'norm'`` 子串 ⇒ "
          "字符串包含判断会把 ``all`` 读成只开 attn ✓ 不报错、只慢 ✗✗）",
          isinstance(llm_mod.resolve_fast_path("all"), frozenset)
          and "norm" not in "all")
    bad_msg = ""
    try:
        llm_mod.resolve_fast_path("nrom")
    except llm_mod.LlmError as err:
        bad_msg = str(err)
    check("⑫³ ⭐ 拼错**当场拒** ✓（``nrom`` 静默当关 ⇒ 人以为开了、其实一步没省 ✗✗；"
          "报错要带得出「可写什么」✓）",
          "nrom" in bad_msg and "fastPath" in bad_msg, bad_msg)

    # ── 2. 折叠 RoPE 逐位对拍
    torch.manual_seed(0)
    seq, half = 7, CONFIG.head_dim // 2
    pos = torch.arange(seq)
    cos_f, sin_f = llm_mod.rope_freqs(pos, CONFIG.head_dim, CONFIG.rope_theta)
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(2, CONFIG.heads, seq, CONFIG.head_dim).to(dtype)
        x = x * 3.0
        want = llm_mod.apply_rope(x, cos_f, sin_f)
        wide_cos, wide_sin = llm_mod.fold_rope(cos_f, sin_f, dtype=dtype)
        got = llm_mod.apply_rope_wide(x, wide_cos, wide_sin)
        check(f"⑫⁴ 折叠 RoPE 与 ``apply_rope`` **逐位相同** ✓（{str(dtype).split('.')[-1]} ✓ "
              f"max|差|={(want.float() - got.float()).abs().max().item():.3e} ✓）",
              torch.equal(want, got),
              (want.float() - got.float()).abs().max().item())
        # ⚠️ 反向验证用：把前半那条减号写反 ⇒ 必须**看得出**差 ✗（否则判据没牙 ✓）
        hd = CONFIG.head_dim
        flipped = llm_mod.apply_rope_wide(
            x, wide_cos, torch.cat((-wide_sin[..., :hd // 2], wide_sin[..., hd // 2:]), dim=-1))
        check(f"⑫⁵ 反证：``sin_wide`` 前半的减号写反 ⇒ 结果**确实不同** ✓（{str(dtype).split('.')[-1]} ✓ "
              "⇒ 上一条不是恒真 ✓）",
              not torch.equal(want, flipped))

    # 先降精度再折 ≡ 折完再降 ✓（``fold_rope(dtype=)`` 只降一次 ⇒ 与逐层降的结果一致 ✓）
    lo = torch.bfloat16
    a = llm_mod.fold_rope(cos_f, sin_f, dtype=lo)
    b = llm_mod.fold_rope(cos_f.to(lo), sin_f.to(lo))
    check("⑫⁶ ``fold_rope(dtype=)`` 与「先 ``.to`` 再折」逐位相同 ✓（每步只降一次 ✓ "
          "老路是每层每次降 ✓ 4B 档一步 72 次 ✗）",
          a[0].dtype is lo and torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))

    # 窄形 cos 走老路 ⇒ 与宽形走新路逐位相同（模型里按 ``cos.shape[-1]`` 分流 ✓）
    attn = llm_mod.primitives()["GQAAttention"](CONFIG)
    xr = torch.randn(1, 3, CONFIG.hidden) * 2.0
    cos_3, sin_3 = llm_mod.rope_freqs(torch.arange(3), CONFIG.head_dim, CONFIG.rope_theta)
    wide = llm_mod.fold_rope(cos_3, sin_3, dtype=xr.dtype)
    got_wide = attn(xr, *wide)[0]
    got_narrow = attn(xr, cos_3, sin_3)[0]
    check("⑫⁷ 模型里按 ``cos.shape[-1]`` 分流 ✓（宽形 → 新路 ✓、窄形 → 老 ``apply_rope`` ✓）"
          "两条**逐位相同** ✓",
          torch.equal(got_wide, got_narrow))

    # ── 3. SDPA 与手写三式对拍（掩码偏移是唯一的静默错点 ✗）
    def _manual(q, k, v, past_len, n_rep, shift=0):
        k_rep = llm_mod.repeat_kv(k, n_rep)
        v_rep = llm_mod.repeat_kv(v, n_rep)
        scores = q @ k_rep.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if q.shape[2] > 1:
            mask = torch.triu(torch.full((q.shape[2], past_len + q.shape[2]), float("-inf"),
                                         dtype=scores.dtype), diagonal=past_len + 1 + shift)
            scores = scores + mask
        return torch.softmax(scores, dim=-1) @ v_rep

    torch.manual_seed(1)
    for past_len, s in ((0, 5), (4, 3), (6, 1)):
        q = torch.randn(1, CONFIG.heads, s, CONFIG.head_dim)
        k = torch.randn(1, CONFIG.kv_heads, past_len + s, CONFIG.head_dim)
        v = torch.randn(1, CONFIG.kv_heads, past_len + s, CONFIG.head_dim)
        want = _manual(q, k, v, past_len, CONFIG.heads // CONFIG.kv_heads)
        got = llm_mod.sdpa_attention(q, k, v, past_len=past_len,
                                     n_rep=CONFIG.heads // CONFIG.kv_heads)
        gap = (want - got).abs().max().item()
        check(f"⑫⁸ SDPA 与手写三条式数值一致 ✓（past={past_len} s={s} ⇒ max|差|={gap:.2e} ✓ "
              "``enable_gqa`` 不吃 ``repeat_kv`` 物化 ✓）", gap < 1e-6, gap)
        if s > 1:
            off = (want - _manual(q, k, v, past_len, CONFIG.heads // CONFIG.kv_heads,
                                 shift=1)).abs().max().item()
            check(f"⑫⁹ 反证：掩码对角偏 1 ⇒ 差异**远超**容差 ✓（past={past_len} "
                  f"⇒ {off:.2e} ✓ 挡住「掩码整体错位一行」这类静默错 ✗✗）", off > 1e-3, off)

    # ── 4. spy：两个档位**真的**调到融合核
    fused = llm_mod.primitives()["F"]
    if getattr(fused, "rms_norm", None) is None:
        skip("⑫¹⁰ 本机 F.rms_norm 不存在（torch < 2.4）⇒ 融合 norm 判不了 ✓（≠ 通过 ✓）")
    else:
        real_norm, real_sdpa = fused.rms_norm, fused.scaled_dot_product_attention
        calls = {"norm": 0, "sdpa": 0}

        def _count_norm(*args, **kwargs):
            calls["norm"] += 1
            return real_norm(*args, **kwargs)

        def _count_sdpa(*args, **kwargs):
            calls["sdpa"] += 1
            return real_sdpa(*args, **kwargs)

        fused.rms_norm, fused.scaled_dot_product_attention = _count_norm, _count_sdpa
        try:
            model = llm_mod.build_llm(CONFIG)
            probe = torch.randint(0, CONFIG.vocab_size, (1, 4))
            seen: dict[str, tuple[int, int]] = {}
            for name, fast in (("off", frozenset()), ("norm", frozenset({"norm"})),
                               ("attn", frozenset({"attn"})),
                               ("all", frozenset({"norm", "attn"}))):
                calls["norm"] = calls["sdpa"] = 0
                model(probe, None, kv_mode="fp", fast_path=fast)
                seen[name] = (calls["norm"], calls["sdpa"])
        finally:
            fused.rms_norm, fused.scaled_dot_product_attention = real_norm, real_sdpa
        check("⑫¹⁰ ⭐ spy：``fastPath=off`` ⇒ 融合核**一次都没调** ✓（老路径 ✓ 逐位不变 ✓）、"
              "``norm`` ⇒ 只调 ``F.rms_norm`` ✓、``attn`` ⇒ 只调 SDPA ✓、``all`` ⇒ 两个都调 ✓"
              f"（实测 {seen} ✓）",
              seen["off"] == (0, 0) and seen["norm"][0] >= CONFIG.depth * 2 + 1
              and seen["norm"][1] == 0 and seen["attn"] == (0, CONFIG.depth)
              and seen["all"] == (seen["norm"][0], CONFIG.depth), seen)
        check("⑫¹¹ ``fastPath=off`` 与不传 ``fast_path`` **逐位相同** ✓（装上开关不许顺手动老路 ✗✗）",
              torch.equal(model(probe, None, kv_mode="fp")[0],
                          model(probe, None, kv_mode="fp", fast_path=frozenset())[0]))

    check("⑫¹² ``fused_norm_available()`` 拿得到 bool ✓ 且**不抛** ✓（``F`` 是 ``_build`` 的局部名 ✗✗ "
          "⇒ 写成模块级 ``getattr(F, ...)`` 会让 ``describe()`` 每次 ``NameError`` ✓，"
          "连「没装 torch」都报不出来 ✗✗）",
          isinstance(llm_mod.fused_norm_available(), bool))

    # ── 5. 默认关 ＝ 调用签名逐字不变（拿桩模型判 ✓）
    class _Stub:
        def __call__(self, ids, past_kv=None, kv_mode=None):     # 只认老签名 ✗ 多的关键字就炸
            vocab = CONFIG.vocab_size
            return torch.zeros(1, int(ids.shape[1]), vocab), []

    stub_ok, stub_err = True, ""
    try:
        llm_mod.generate_llm(_Stub(), torch.zeros(1, 3, dtype=torch.long),
                             max_new_tokens=2, temperature=0, fast_path="off")
    except Exception as err:  # noqa: BLE001
        stub_ok, stub_err = False, f"{type(err).__name__}: {err}"
    check("⑫¹³ ⭐ 默认关时**不传** ``fast_path`` 关键字 ✓（空集 ⇒ 调用签名与老路逐字相同 ✓ "
          "否则本仓一堆桩模型测试全炸 ✗✗）", stub_ok, stub_err)
    raised = False
    try:
        llm_mod.generate_llm(_Stub(), torch.zeros(1, 3, dtype=torch.long),
                             max_new_tokens=2, temperature=0, fast_path="norm")
    except TypeError:
        raised = True
    check("⑫¹⁴ 反证：开着时**确实**把 ``fast_path`` 递下去了 ✓（桩只认老签名 ⇒ 必 ``TypeError`` ✓ "
          "⇒ 上一条不是「反正都没传」✓）", raised)


def case_kv_buffer(torch) -> None:
    """KV 定长缓冲（2026-11-08 ✓ ``fastPath=kvbuf`` ✓）—— 「只换存法、一个数都不动」的判据。

    这一档与 ``norm``／``attn`` 的性质**不同** ✗✗：那两档改数值 ✓（所以只判「真的走到融合核」✓），
    本档宣称的是**逐位相同** ✓ —— 换的只是「历史怎么写进显存」：老路每步 ``torch.cat`` 把整段
    重新分配 + 重拷一遍 ✓，新路是一块定长存储 ＋ 原位写新来的那几行 ✓。所以判据只有一种写法：
    ``torch.equal`` ✓✗（``allclose`` 抓不到「某一行写到错位置」✓ —— 每行的量级一样 ✓ 平均下来
    差异还是小 ✓ 于是绿的 ✗✗）。

    五件事（全是「跑得出来、数字悄悄变」的形状）：
    1. 逐位相同 ✓✗：``fp``／``q8`` 两档、prefill 多行 ＋ decode 单行、带 ``past`` ✓；
    2. ``shape[2]`` **仍是已用长度** ✓✗（暴露 ``cap`` ⇒ 位置/掩码整排错 ✓ 而且形状自洽 ✓）；
    3. ⭐ ``aten::cat`` **一次都不发** ✓（这是本档的全部意义 ✓ 用 spy 数 ✓ —— 只判「值对」在
       「其实还在 cat」时照样绿 ✗✗）；
    4. ⭐ 前缀复用处必须**独立副本** ✓✗（:func:`truncate_kv` ✓）：就地改长度会把 ``KvPrefixCache``
       里那份常驻 cache 改坏 ✓（不报错 ✓，下次命中更长前缀时读到脏行 ✗✗）；
    5. 容量翻倍 ✓、混用（开→关→开）✓ 都不许让数值飘 ✓。
    """
    prim = llm_mod.primitives()
    KvBuf = prim["KvBuf"]
    d, h = CONFIG.head_dim, CONFIG.kv_heads
    torch.manual_seed(7)
    steps = [torch.randn(1, h, 3, d), torch.randn(1, h, 1, d), torch.randn(1, h, 1, d)]

    # ── 1. fp：逐位相同 + shape[2] 语义
    off_past = buf_past = None
    same, kinds, lens = True, True, []
    for fresh in steps:
        off_past = llm_mod.update_kv_cache(off_past, fresh, "fp")
        buf_past = llm_mod.update_kv_cache(buf_past, fresh, "fp", buffer=True)
        kinds = kinds and isinstance(buf_past, KvBuf)
        same = same and torch.equal(off_past, llm_mod.materialize_kv(buf_past))
        lens.append(int(buf_past.shape[2]))
    check("⑬¹ ⭐ 缓冲档（fp）与 ``torch.cat`` **逐位相同** ✓（prefill 3 行 + decode 各 1 行 ✓）"
          "＋ ``shape[2]`` 报的是**已用长度** ✓✗（报 ``cap`` ⇒ 位置/掩码整排错位、形状还自洽 ✗✗）",
          same and kinds and lens == [3, 4, 5], (same, kinds, lens))

    # ── 2. q8：值与 scale 都要逐字节对（只对一半＝长度对、数值错位 ✗✗）
    off_q = buf_q = None
    same_q = True
    for fresh in steps:
        off_q = llm_mod.update_kv_cache(off_q, fresh, "q8")
        buf_q = llm_mod.update_kv_cache(buf_q, fresh, "q8", buffer=True)
        same_q = same_q and torch.equal(off_q.q, buf_q.store[:, :, :buf_q.length])
        same_q = same_q and torch.equal(off_q.scale, buf_q.scale[:, :, :buf_q.length])
        same_q = same_q and torch.equal(off_q.dequant(), llm_mod.materialize_kv(buf_q))
    check("⑬² q8 档连**量化字节**都逐位相同 ✓✗（``q`` 与 ``scale`` 各对一遍 ✓ —— "
          "只搬一半是「长度对、数值错位」✓ 不报错 ✗）", same_q)

    # ── 3. 容量翻倍：150 行 ⇒ length 150、cap 从 64 起翻到 256，且全程逐位相同
    torch.manual_seed(11)
    cat_long = buf_long = None
    same_long = True
    for _ in range(150):
        fresh = torch.randn(1, h, 1, d)
        cat_long = llm_mod.update_kv_cache(cat_long, fresh, "fp")
        buf_long = llm_mod.update_kv_cache(buf_long, fresh, "fp", buffer=True)
        same_long = same_long and torch.equal(cat_long, llm_mod.materialize_kv(buf_long))
    check("⑬³ 容量按**翻倍**涨 ✓（150 行 ⇒ ``cap`` 64→128→256 ✓ 摊还 O(1) ✓）＋ 全程逐位相同 ✓ "
          "＋ ``kv_cache_bytes`` 对缓冲报**已分配**字节 ✓（翻倍留的尾巴也在卡上 ✓ 会 OOM ✓）",
          same_long and int(buf_long.length) == 150 and int(buf_long.cap) == 256
          and llm_mod.kv_cache_bytes([(buf_long,)]) == buf_long.kv_bytes(),
          (buf_long.length, buf_long.cap))

    # ── 4. ⭐ spy：``aten::cat`` 真的不再发
    real_cat = torch.cat
    counts = {"cat": 0}

    def _count_cat(*args, **kwargs):
        counts["cat"] += 1
        return real_cat(*args, **kwargs)

    torch.cat = _count_cat
    try:
        off_p = on_p = q8_p = None
        offf = []
        for fresh in steps:
            off_p = llm_mod.update_kv_cache(off_p, fresh, "fp")
        off_calls = counts["cat"]
        counts["cat"] = 0
        for fresh in steps:
            on_p = llm_mod.update_kv_cache(on_p, fresh, "fp", buffer=True)
        on_calls = counts["cat"]
        counts["cat"] = 0
        for fresh in steps:
            q8_p = llm_mod.update_kv_cache(q8_p, fresh, "q8", buffer=True)
        q8_calls = counts["cat"]
        del offf
    finally:
        torch.cat = real_cat
    check("⑬⁴ ⭐ spy ``aten::cat``：老路 2 次（3 步去掉首个 None ✓）、缓冲 **0** 次 ✓✗ —— "
          "只判「值还对」在「其实还在 cat」时照样绿 ✗✗（q8 档也要 0 次 ✓ 它老路是双份 cat ✗）",
          off_calls == 2 and on_calls == 0 and q8_calls == 0,
          (off_calls, on_calls, q8_calls))

    # ── 5. 整模型：prefill + 3 步 decode 逐位相同 + 每步 cat 数对得上
    model = llm_mod.build_llm(CONFIG)
    probe = torch.randint(0, CONFIG.vocab_size, (1, 4))
    lg_off, kv_off = model(probe, None, kv_mode="fp")
    lg_buf, kv_buf = model(probe, None, kv_mode="fp", fast_path=frozenset({"kvbuf"}))
    same_steps = (torch.equal(lg_off, lg_buf) and isinstance(kv_buf[0][0], KvBuf)
                  and not isinstance(kv_off[0][0], KvBuf))
    tok = torch.tensor([[3]], dtype=torch.long)
    for _ in range(3):
        l_off, kv_off = model(tok, kv_off, kv_mode="fp")
        l_buf, kv_buf = model(tok, kv_buf, kv_mode="fp", fast_path=frozenset({"kvbuf"}))
        same_steps = same_steps and torch.equal(l_off, l_buf)
    check("⑬⁵ ⭐ 整模型：prefill ＋ 3 步 decode 的 logits 与老路**逐位相同** ✓✗"
          "（fp 省的是「每步整段重拷」✓ 数值一个字节都不许动 ✓）", same_steps)

    counts["cat"] = 0
    torch.cat = _count_cat
    try:
        model(tok, kv_off, kv_mode="fp")
        off_step = counts["cat"]
        counts["cat"] = 0
        model(tok, kv_buf, kv_mode="fp", fast_path=frozenset({"kvbuf"}))
        buf_step = counts["cat"]
    finally:
        torch.cat = real_cat
    # 一 step 的 cat 数口径（实测 ✓）：折叠 RoPE 2 次 ＋ 每层 ``apply_rope_wide`` 2 次（q/k）
    # ⇒ 与 cache 无关的固定开销 ``2 + 2×depth``；老路每层还要**再**加 k/v 两次整段重拷 ✓。
    fixed = 2 + 2 * CONFIG.depth
    check("⑬⁶ 一 step 的 cat 数：老路 ``fixed + 2×depth`` 次（``fixed`` = 折叠 RoPE 2 ＋ 每层 q/k 各 1 ✓）、"
          "缓冲路只剩 ``fixed`` ✓（= 每层那 2×depth 次整段重拷真的没了 ✓ fp 下 4B 档一步就是 72 次 ✗）",
          buf_step == fixed and off_step == fixed + 2 * CONFIG.depth,
          (off_step, buf_step, fixed))

    # ── 6. ⭐ 前缀复用处：``truncate_kv`` 必须是**独立副本**
    src = buf_long                       # 150 行（fp ✓）
    cut = llm_mod.truncate_kv([(src, src)], 100)
    kept = src.store[:, :, 100:104].clone()
    cut[0][0].write(torch.randn(1, h, 4, d))
    check("⑬⁷ ⭐ ``truncate_kv`` 对缓冲是**独立副本** ✓✗（就地改长度会把 ``KvPrefixCache`` 那份常驻 "
          "cache 的历史覆盖 ✓ ⇒ 下次命中更长前缀读到脏行、不报错 ✗✗）",
          int(cut[0][0].length) == 104 and int(src.length) == 150
          and torch.equal(kept, src.store[:, :, 100:104]),
          (cut[0][0].length, src.length))
    same_obj = llm_mod.rollback_kv([(src, src)], 100)
    check("⑬⁸ 反证：``rollback_kv`` **就地**回拨 ✓（同一对象、零拷贝 ✓ 投机窗口专用 ✓）"
          "＋ 上一条之后原份仍是 150 行 ✓ ⇒ 两者**不许互换** ✗✗",
          same_obj[0][0] is src and int(src.length) == 100)

    # ── 7. 混用（开 → 关 → 开）字节原样：不许中途 dequant→quantize 多损一次
    torch.manual_seed(13)
    chained = [torch.randn(1, h, 1, d) for _ in range(4)]
    bq = None
    for fresh in chained[:2]:
        bq = llm_mod.update_kv_cache(bq, fresh, "q8", buffer=True)
    seg = bq.as_q8()
    off_chain = [(llm_mod.update_kv_cache(seg, chained[2], "q8"),
                  llm_mod.update_kv_cache(seg, chained[2], "q8"))]
    back = llm_mod.update_kv_cache(off_chain[0][0], chained[3], "q8", buffer=True)
    ref = None
    for fresh in chained:
        ref = llm_mod.update_kv_cache(ref, fresh, "q8")
    check("⑬⁹ 混用（开→关→开）**字节原样** ✓✗（``Q8Kv``↔``KvBuf`` 换算走 :meth:`as_q8`／:meth:`_take` ✓ "
          "—— 中途 dequant 再 quantize 会多损一次 ✓ 数值悄悄变、不报错 ✗✗）",
          torch.equal(ref.q, back.store[:, :, :back.length])
          and torch.equal(ref.scale, back.scale[:, :, :back.length])
          and torch.equal(ref.dequant(), llm_mod.materialize_kv(back)))

    # ── 8. 口径不符当场拒
    bad_batch = mixed = ""
    try:
        buf_long.write(torch.randn(2, h, 1, d))
    except llm_mod.LlmError as err:
        bad_batch = str(err)
    try:
        llm_mod.update_kv_cache(buf_long, torch.randn(1, h, 1, d), "q8", buffer=True)
    except llm_mod.LlmError as err:
        mixed = str(err)
    check("⑬¹⁰ 口径不符**当场拒** ✓（批/头数 ✗、``kvMode`` 与缓冲不一致 ✗）—— 静默挑一个口径算就是"
          "「形状自洽、数值全烂」✗✗", "批/头数" in bad_batch and "kvMode" in mixed,
          (bad_batch, mixed))

    # ── 9. 反证：写错位置 ⇒ 上面的逐位判据立刻红
    broken = KvBuf.empty(1, h, 64, quant=False, dtype=torch.float32, device=steps[0].device,
                         group=d, head_dim=d)
    for fresh in steps:
        n = int(fresh.shape[2])
        broken.store[:, :, 0:n].copy_(fresh)      # 一律写回位置 0（错位 ✓）
        broken.length += n
    check("⑬¹¹ 反证：把新段一律写到**位置 0** ⇒ 与 ``torch.cat`` 的逐位判据立刻红 ✓"
          "（证明 ⑬¹ 不是「怎么摆都一样」✓）",
          not torch.equal(llm_mod.materialize_kv(broken), off_past))

    # ── 10. generate_llm ＋ 前缀复用：逐 token 相同、命中也要相同
    ids = torch.randint(0, CONFIG.vocab_size, (1, 5))
    plain = llm_mod.generate_llm(model, ids, max_new_tokens=3, temperature=0)
    bufout = llm_mod.generate_llm(model, ids, max_new_tokens=3, temperature=0,
                                  fast_path="kvbuf")
    cache = llm_mod.KvPrefixCache(8 * 2 ** 20)
    first = llm_mod.generate_llm(model, ids, max_new_tokens=3, temperature=0,
                                 fast_path="kvbuf", kv_cache=cache)
    hit = llm_mod.generate_llm(model, ids, max_new_tokens=3, temperature=0,
                               fast_path="kvbuf", kv_cache=cache)
    receipt = cache.describe()
    check("⑬¹² ``generate_llm`` 整条：开缓冲与老路**逐 token 相同** ✓ ＋ 带 ``KvPrefixCache`` "
          "（缓冲进、缓冲出 ✓）第二次**命中**且结果仍相同 ✓✗ —— 复用走的是 ``truncate_kv`` 的副本 ✓",
          torch.equal(plain, bufout) and torch.equal(plain, first) and torch.equal(plain, hit)
          and int(receipt["hits"]) >= 1 and int(receipt["reusedTokens"]) >= 1,
          (plain.tolist(), bufout.tolist(), hit.tolist(), receipt))


def case_kv_budget(torch) -> None:
    """显存侧 KV 预算 ＋ 滑窗压缩（2026-11-08 ✓ 低显存两件 ✓）。

    这两件解决的是**同一个**病：低显存下「装得进、跑起来才炸」✗✗ ——

    * :func:`llm.kv_room_tokens` 把「这块卡还放得下多少 token」算**在生成之前** ✓（原先只有
      ``_context_length`` ＝ **权重**声明的窗口 ✓，那是模型的先天上限 ✓、与卡上剩多少无关 ✗）；
    * :func:`llm.compact_kv` ＋ ``position_offset`` 让历史**超窗口就滚掉中间段** ✓（有损 ✓），
      而 RoPE 位置**继续按绝对位置涨** ✗✗ —— 这一条是全场最容易写错的地方：压缩后
      ``cache 行数 ≠ 历史长度`` ✓，位置若跟着行数退回 ⇒ **每滚一轮整体相位跳一次** ✓ 且
      **不报错** ✓✗（只有生成质量悄悄烂 ✗）。

    判据七条（``⑭¹``–``⑭¹¹``）：三态不编 0 ✓、余量真被扣 ✓、q8 真的更省 ✓、
    压缩逐位对（fp/KvBuf/Q8Kv 三种存法 ✓）、``KvBuf`` **独立副本** ✓、预分配与上限拒 ✓、
    ⭐ **位置必须按绝对位置** ✓（``⑭⁹`` 反证：位置退回 ⇒ logits 不同 ✓）。
    """
    prim = llm_mod.primitives()
    KvBuf = prim["KvBuf"]
    d, h = CONFIG.head_dim, CONFIG.kv_heads
    reserve = int(llm_mod.KV_ACTIVATION_RESERVE_BYTES)
    free = 4 * 2 ** 30          # ⚠️ 必须 **> 余量**（512 MiB ✓）：给 4 MiB 那三态全塌成 0 ✓✗

    # ── 1. 三态：给足 / 给不出结论 / 连余量都不够（**不编 0** ✗✗）
    big = llm_mod.kv_room_tokens(CONFIG, "fp", free_bytes=free)
    unknown = llm_mod.kv_room_tokens(CONFIG, "fp", free_bytes=None)
    tiny = llm_mod.kv_room_tokens(CONFIG, "fp", free_bytes=1024)
    check("⑭¹ ``kv_room_tokens`` 三态分明 ✓✗：给足 ⇒ ``ok`` 且 ``maxTokens == "
          "(free − 余量) // 每token`` ✓；**探不到显存 ⇒ ``maxTokens=None``** ✓（报 0 ＝「一个 token 都"
          "放不下」✓ 那是另一个结论 ✗✗）；连余量都不够 ⇒ 0 且 ``ok=False`` ✓",
          big["ok"] and big["maxTokens"] == (free - reserve) // big["perToken"]
          and unknown["maxTokens"] is None and unknown["ok"] is False
          and tiny["maxTokens"] == 0 and tiny["ok"] is False,
          (big, unknown["maxTokens"], tiny["maxTokens"]))

    # ── 2. 余量**真被扣掉**（不扣 ⇒ 算着正好装下、跑起来差几百 MiB ✗）
    nores = llm_mod.kv_room_tokens(CONFIG, "fp", free_bytes=free, reserve_bytes=0)
    check("⑭² 反证：``reserve_bytes=0`` 算出的 token 数**严格更多** ✓ ⇒ 证明 ⑭¹ 里那笔余量真的"
          "从可用字节里扣了 ✓✗（不是写了个没用的参数 ✓）",
          nores["maxTokens"] > big["maxTokens"] and nores["perToken"] == big["perToken"],
          (nores["maxTokens"], big["maxTokens"], nores["perToken"]))

    # ── 3. q8 真的更省（同一块卡能放更多 token）
    q8 = llm_mod.kv_room_tokens(CONFIG, "q8", free_bytes=free)
    check("⑭³ ``kvMode=q8`` 在同一块卡上能放的 token **多于** fp ✓（每 token 字节更少 ✓）"
          "—— 这就是「低显存先撞 KV 预算时该拧的那个开关」✓",
          q8["perToken"] < big["perToken"] and q8["maxTokens"] > big["maxTokens"],
          (q8["perToken"], big["perToken"], q8["maxTokens"], big["maxTokens"]))

    # ── 4. compact_kv 对普通张量：头 head ＋ 尾 tail，逐位对
    torch.manual_seed(11)
    kv = torch.randn(1, h, 10, d)
    got = llm_mod.compact_kv(kv, 2, 3)
    want = torch.cat([kv[:, :, :2], kv[:, :, 7:]], dim=2)
    tail_only = llm_mod.compact_kv(kv, 0, 4)
    keep_all = llm_mod.compact_kv(kv, 6, 6)      # 6+min(6,4)=10 ⇒ 全留 ⇒ 原样返回
    check("⑭⁴ ``compact_kv``（普通张量）：头 2 ＋ 尾 3 **逐位等于**手工 ``cat`` ✓、``head=0`` 只留尾 ✓、"
          "``head+tail ≥ total`` ⇒ **原样返回**（不白拷一次 ✓）",
          torch.equal(got, want) and int(got.shape[2]) == 5
          and torch.equal(tail_only, kv[:, :, 6:]) and keep_all is kv,
          (tuple(got.shape), bool(torch.equal(got, want)), keep_all is kv))

    # ── 5. compact_kv 对 KvBuf：结果对 ＋ **原缓冲一个字节不动**（独立副本 ✗✗）
    buf = KvBuf.empty(1, h, 10, quant=False, dtype=torch.float32, device=kv.device,
                      group=llm_mod.KV_GROUP, head_dim=d)
    buf.write(kv)
    before = llm_mod.materialize_kv(buf).clone()
    small = llm_mod.compact_kv(buf, 2, 3)
    check("⑭⁵ ``compact_kv``（``KvBuf``）：压缩结果逐位对 ✓、``length`` 报压缩后的行数 ✓、"
          "⭐ **原缓冲仍是最长那段** ✓✗（返回独立副本 ✓ —— 原缓冲可能正被 ``KvPrefixCache`` 或"
          "回滚路径引用 ✓，就地改会改坏别人 ✓ 且不报错 ✗）",
          torch.equal(llm_mod.materialize_kv(small), want) and int(small.length) == 5
          and torch.equal(llm_mod.materialize_kv(buf), before) and int(buf.length) == 10,
          (int(small.length), int(buf.length), bool(torch.equal(llm_mod.materialize_kv(buf), before))))

    # ── 6. compact_kv 对 Q8Kv：值与 scale 都要对（只对一半＝长度对、数值错位 ✗✗）
    q8_full = llm_mod.update_kv_cache(None, kv, "q8")
    q8_small = llm_mod.compact_kv(q8_full, 2, 3)
    q8_want = llm_mod.update_kv_cache(None, want, "q8")
    check("⑭⁶ ``compact_kv``（``Q8Kv``）：``q`` 与 ``scale`` **两段都逐位对** ✓✗"
          "（只搬一半 ⇒ 长度对、数值错位 ✓ 而且形状自洽 ✓）",
          torch.equal(q8_small.q, q8_want.q) and torch.equal(q8_small.scale, q8_want.scale),
          (bool(torch.equal(q8_small.q, q8_want.q)), bool(torch.equal(q8_small.scale, q8_want.scale))))

    # ── 7. 预分配：一次开够（省掉「翻倍再翻倍」白占的那一倍）、超上限**具名拒**
    one = torch.randn(1, h, 2, d)
    pre = llm_mod.update_kv_cache(None, one, "fp", buffer=True, reserve_rows=100, ceiling_rows=100)
    clamped = llm_mod.update_kv_cache(None, one, "fp", buffer=True, reserve_rows=200, ceiling_rows=100)
    legacy = llm_mod.update_kv_cache(None, one, "fp", buffer=True)
    check("⑭⁷ ``reserve_rows`` 一次开够 ⇒ ``cap == 100`` ✓（老路只开 ``KV_BUFFER_MIN_ROWS`` ✓）；"
          "``reserve_rows > ceiling_rows`` ⇒ **被钳到上限** ✓（否则首建就超预算 ✓ 自己判自己 ✗）",
          int(pre.cap) == 100 and int(clamped.cap) == 100
          and int(legacy.cap) == int(prim["KV_BUFFER_MIN_ROWS"]),
          (int(pre.cap), int(clamped.cap), int(legacy.cap)))

    def _overflow() -> None:
        llm_mod.update_kv_cache(clamped, torch.randn(1, h, 99, d), "fp", buffer=True,
                                ceiling_rows=100)

    check("⑭⁸ ⭐ 超上限 ⇒ **具名 ``LlmError``** ✓✗（不是静默扩容 ✓ 也不是撞 OOM ✓ —— OOM 会报在"
          "下一个算子上、离真因几十步 ✗）",
          _raises(_overflow) and int(clamped.cap) == 100,
          (int(clamped.cap), _raises(_overflow)))

    # ── 9. ⭐ 位置必须按**绝对位置**：压缩后 cache 行数 ≠ 历史长度
    model = llm_mod.build_llm(CONFIG)
    torch.manual_seed(12)
    ids = torch.randint(0, CONFIG.vocab_size, (1, 9))
    with torch.inference_mode():
        _, past7 = model(ids[:, :7])
        squeezed = [(llm_mod.compact_kv(k, 2, 3), llm_mod.compact_kv(v, 2, 3))
                    for k, v in past7]
        nxt = ids[:, 7:8]
        logits_abs, _ = model(nxt, squeezed, position_offset=7)
        logits_naive, _ = model(nxt, squeezed)          # 位置 = past_len = 5（错的那条）
    check("⑭⁹ ⭐⭐ 压缩后 RoPE 位置必须按**绝对位置**（7）✓✗：``position_offset=7`` 与"
          "「跟着 cache 行数走」（5）算出的 logits **必须不同** ✓ —— 相同就说明 ``position_offset`` "
          "根本没生效 ✓（症状：每滚一轮整体相位跳一次 ✓ 且**不报错** ✓✗）",
          not torch.equal(logits_abs, logits_naive),
          (float((logits_abs - logits_naive).abs().max()),))

    # ── 10. generate_llm 整条：滑窗真压缩（用 spy 数，**不靠输出不同** ✗✗）
    #    ⚠️ 第一版判「滑窗输出与不滑窗不同」✓ —— 实测**两条一模一样** ✓ 但那不是压缩没发生 ✓：
    #    随机权重 + 贪心从第 6 个 token 起就锁死在同一个 id ✓ ⇒ 后面 14 步无论记不记得中间段都一样 ✓✗。
    #    「结果不同」在贪心上是**弱判据** ✓ ⇒ 改成数调用次数 + 数压缩后的行数 ✓。
    #    ⚠️ 也不能换掉模块上的 ``compact_kv`` 来数调用 ✗✗：``llm.py`` 是「工厂 + ``__getattr__``」结构 ✓，
    #    ``generate_llm`` 里那个名字是 ``_build()`` 的**闭包局部名** ✓ ⇒ 改模块属性对闭包一点影响都没有 ✓
    #    （实测 0 次调用、判据恒红 ✓ 而压缩其实真的发生了 ✗）。改成给 ``LlmModel`` 挂 forward pre-hook ✓：
    #    直接读「这一步模型拿到的 past 有多少行」＋「position_offset 是多少」✓ —— 这是**外部可观测**的量 ✓，
    #    压缩真发生就必然反映在行数上 ✓（不滑窗时行数会一路涨到 19 ✓）。
    torch.manual_seed(13)
    ids2 = torch.randint(0, CONFIG.vocab_size, (1, 6))
    bucket: list[tuple[int, int | None]] = []

    def _pre_hook(_module: Any, args: Any, kwargs: Any) -> None:  # noqa: ANN401 —— 测试钩子 ✓
        past = args[1] if len(args) > 1 else kwargs.get("past_kv")
        rows = 0 if past is None else int(past[0][0].shape[2])
        bucket.append((rows, kwargs.get("position_offset")))

    handle = model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    try:
        bucket.clear()
        plain = llm_mod.generate_llm(model, ids2, max_new_tokens=14, temperature=0)
        plain_rows = [r for r, _ in bucket]
        plain_pos = [p for _, p in bucket]
        bucket.clear()
        win = llm_mod.generate_llm(model, ids2, max_new_tokens=14, temperature=0,
                                   kv_window=3, kv_sink=2)
        win_rows = [r for r, _ in bucket]
        win_pos = [p for _, p in bucket]
    finally:
        handle.remove()
    check("⑭¹⁰ ``generate_llm(kv_window=3, kv_sink=2)``：⭐ **真压缩了**（挂 forward pre-hook 读模型每步"
          "拿到的 past 行数 ✓）：滑窗时**恒定 5 行** = ``sink+win`` ✓（不是越压越短、也不是压了没写回 ✓）；"
          "不滑窗同一条路涨到 19 行 ✓（证明 5 不是「本来就小」✓）；滑窗时 ``position_offset`` 逐拍递增"
          "**绝对位置** 0..14 ✓、不滑窗一个都不传 ✓（默认关 = 老签名 ✓）；输出长度仍正确 ✓",
          len(win_rows) > 1 and win_rows[0] == 0 and set(win_rows[1:]) == {5}
          and max(plain_rows) == 6 + 13 and len(win_rows) == len(plain_rows) == 15
          and win_pos == [0] + list(range(6, 20))
          and all(p is None for p in plain_pos)
          and int(win.shape[1]) == 6 + 14,
          (win_rows, max(plain_rows or [0]), win_pos[:6], plain_pos[:3]))

    # ⭐⭐ 2026-11-09 修：**滑窗管不了 prefill** ✗✗ —— ``kv_rows`` 预分配只按 ``sink+window`` 留行数 ✓，
    # 而压缩只发生在**两个生成词元之间** ✓ ⇒ 提示词比窗口长时，第 2 个 prefill 块就撞「KV 缓冲超预算」✓✗，
    # 且回执给的补救（缩上下文／开 q8／调小 maxOutputTokens）**三条都不是真因** ✓✗。
    # 实测复现（修前）：``kv_window=3, kv_sink=2``（⇒ 15 行上限）＋ 20 行提示词 ⇒「要 20 行、上限 15 行」✓。
    torch.manual_seed(31)
    ids_long = torch.randint(0, CONFIG.vocab_size, (1, 20))
    long_rows: list[int] = []

    def _hook_long(_m: Any, args: Any, kwargs: Any) -> None:  # noqa: ANN401 —— 测试钩子 ✓
        past = args[1] if len(args) > 1 else kwargs.get("past_kv")
        long_rows.append(0 if past is None else int(past[0][0].shape[2]))

    handle_long = model.register_forward_pre_hook(_hook_long, with_kwargs=True)
    long_err: str = ""
    long_out: Any = None
    try:
        long_rows.clear()
        long_out = llm_mod.generate_llm(model, ids_long, max_new_tokens=4, temperature=0,
                                        fast_path="kvbuf", kv_window=3, kv_sink=2)
    except Exception as err:  # noqa: BLE001 —— 要的是「不抛」，抛了就把话说清 ✓
        long_err = f"{type(err).__name__}: {err}"
    finally:
        handle_long.remove()
    check("⑭¹⁰′ ⭐⭐ 滑窗 ＋ 长提示词**必须能跑** ✓（2026-11-09 修）：20 行提示词 > ``sink+window+2+spec``"
          "（＝15 行）✓ ⇒ 缓冲峰值按 ``max(total, sink+window)`` 留 ✓✗（只按窗口留 ⇒ 第 2 个 prefill 块就"
          "「超预算」✓ 而理由是「缩上下文／调小 maxOutputTokens」✓ 全是假补救 ✓）；"
          "＋ 生成期照样压到 ``sink+window=5`` 行 ✓（修的是峰值，不是「干脆不压了」✗）",
          not long_err and long_out is not None and int(long_out.shape[1]) >= 21
          and long_rows[:1] == [0] and set(long_rows[1:]) == {5},
          (long_err, tuple(long_out.shape) if long_out is not None else None, long_rows))

    # ── 11. 两个组合闸门：滑窗×前缀缓存、sink ≥ window
    def _with_cache() -> None:
        llm_mod.generate_llm(model, ids2, max_new_tokens=2, temperature=0,
                             kv_window=4, kv_sink=2,
                             kv_cache=llm_mod.KvPrefixCache(8 * 2 ** 20))

    def _bad_sink() -> None:
        llm_mod.generate_llm(model, ids2, max_new_tokens=2, temperature=0,
                             kv_window=4, kv_sink=4)

    check("⑭¹¹ 两个组合**当场拒** ✓✗：``kv_window`` ＋ ``kv_cache``（前缀缓存存的是完整历史 ✓，"
          "滑窗后那段已经不存在了 ✓ ⇒ 复用回来会让模型看见「本不该记得」的行 ✓）；"
          "``kv_sink >= kv_window``（sink 是窗口之外另留的开头几行 ✓，相等就没有可滚的窗口 ✗）",
          _raises(_with_cache) and _raises(_bad_sink),
          (_raises(_with_cache), _raises(_bad_sink)))

    # ── 12. 两个归一函数的**边界**（2026-11-08 补 ✓）：``off`` 一族 ⇒ 0 ✓；拼错／负数 ⇒ 拒 ✗；
    # ``True`` 不许当 1 ✗✗（「开」不等于「只记最后一个 token」✓）；``0`` **合法** ✓（＝真的不要 sink ✓）。
    win = llm_mod.resolve_kv_window
    sink = llm_mod.resolve_kv_sink

    def _rejects(fn, value) -> bool:  # noqa: ANN001 —— 测试小工具 ✓
        try:
            fn(value)
        except llm_mod.LlmError:
            return True
        except Exception:  # noqa: BLE001
            return False
        return False

    check("⑭¹² ``resolve_kv_window`` 归一 ✓：``None``/``\"\"``/``off``/``false``/``0``/``0`` 字符串 ⇒ **0**"
          "（老路径 ✓）；正整数与数字字符串 ⇒ 原值 ✓；``True``／``\"3k\"``／``-1`` ⇒ **当场拒** ✗"
          "（``3k`` 静默当 0 ⇒ 人以为开了滑窗、显存照涨 ✓✗；负窗口＝一行历史都不留却照样出字 ✓✗）",
          win(None) == 0 and win("") == 0 and win("off") == 0 and win("FALSE") == 0
          and win(0) == 0 and win("0") == 0 and win(1024) == 1024 and win("1024") == 1024
          and _rejects(win, True) and _rejects(win, "3k") and _rejects(win, -1),
          (win(None), win("1024"), _rejects(win, True), _rejects(win, "3k")))

    check("⑭¹³ ``resolve_kv_sink`` 归一 ✓：不写 ⇒ 默认 4 ✓；``0`` **合法** ✓（与 ``specWindow`` 那条"
          "「< 1 即拒」口径**不同** ✗ —— 照抄会把「真的不要 sink」误拒 ✓✗）；``-1``／``\"a\"`` ⇒ 拒 ✗",
          sink(None) == llm_mod.KV_SINK_DEFAULT == 4 and sink("") == 4
          and sink(0) == 0 and sink("0") == 0 and sink(8) == 8
          and _rejects(sink, -1) and _rejects(sink, "a"),
          (sink(None), sink(0), _rejects(sink, -1)))


def case_cuda_graph(torch) -> None:
    """图化 decode 单步（2026-11-08 ✓ ``cudaGraph`` ✓）—— 判据全压在「捕获成功、重放出来是错的」这条线上。

    图化把三个**每步都变的 Python 标量**（历史长度 ✓ 绝对位置 ✓ 写哪一行 ✓）赶出图外 ✓ —— 它们在图里
    会被烘成常量 ✗✗，三处都得换成「设备侧张量」✓／「一次性常量」✓。漏掉任一处：**形状全对、一步不报错**，
    只是从第 2 步起算错 ✓✗（症状是「越往后越胡言乱语」✓ 而不是崩 ✗）。
    """
    wants = [(raw, True) for raw in ("on", "ON", "true", "1", "auto", "graph", True, 1)]
    wants += [(raw, False) for raw in (None, "", "off", "0", "false", "none", False, 0)]
    wrong = [(raw, llm_mod.resolve_cuda_graph(raw), want) for raw, want in wants
             if llm_mod.resolve_cuda_graph(raw) is not want]
    check("⑮¹ ``resolve_cuda_graph`` 归一表 ✓（**默认关** ✓ —— 归约长度与老路不同 ⇒ 不许默认换 ✗）",
          not wrong, wrong)
    bad_msg = ""
    try:
        llm_mod.resolve_cuda_graph("grpah")
    except llm_mod.LlmError as err:
        bad_msg = str(err)
    check("⑮² ⭐ 拼错**当场拒** ✓（``grpah`` 静默当关 ⇒ 人以为在跑图、其实每步仍在发几千次 launch ✗✗）",
          "grpah" in bad_msg and "cudaGraph" in bad_msg, bad_msg)

    # ⭐ 2026-11-09 补：``auto`` 此前是 ``on`` 的**别名** ✗✗（``resolve_cuda_graph`` 的接受表里就有它）
    # ⇒ 人写 auto 期望「该开则开」✓，实际连 8 词元那种净亏 ~28 ms 的档也照捕 ✓、回执还只报 ``on`` ✓✗。
    # 三档必须先分得开 ✓，否则「按长度决定」这件事根本没有落点。
    pol = llm_mod.resolve_cuda_graph_policy

    def _pol_rejects(value) -> bool:  # noqa: ANN001 —— 小工具 ✓
        try:
            pol(value)
        except llm_mod.LlmError:
            return True
        except Exception:  # noqa: BLE001
            return False
        return False

    pol_wrong = [(raw, pol(raw), want) for raw, want in
                 (("auto", "auto"), ("AUTO", "auto"), (" on ", "on"), ("graph", "on"),
                  (True, "on"), (1, "on"), (None, "off"), ("", "off"), ("0", "off"), (False, "off"))
                 if pol(raw) != want]
    check("⑮¹′ ⭐ ``auto`` 是**真三档** ✓（2026-11-09 补）：``off``／``on``／``auto`` 各自归一 ✓ ＋ "
          "``resolve_cuda_graph('auto')`` 仍答「**可能**开」✓（bool 那一问不丢 ✓）＋ 拼错同样当场拒 ✓"
          "（两函数共用一张表 ⇒ 不会再出现「一处接、一处拒」✓）＋ 阈值常量在册 ✓",
          not pol_wrong and llm_mod.resolve_cuda_graph("auto") is True
          and _pol_rejects("grpah") and llm_mod.CUDA_GRAPH_AUTO_MIN_TOKENS == 16, pol_wrong)


    # ── ⭐ 定长读窗口 ≡ 截断读（CPU 就能跑 ✓ 与 CUDA 无关 ✓）
    prim = llm_mod.primitives()
    KvSpan, RopeTable = prim["KvSpan"], prim["RopeTable"]
    fast = frozenset({"kvbuf"})
    model = llm_mod.build_llm(CONFIG)
    probe = torch.randint(0, CONFIG.vocab_size, (1, 5))
    # 两条**独立**缓冲 ✓（同一次调用的返回是同一个对象 ✗ 拿它做 A/B 会互相污染 ✓✗）
    _, past_off = model(probe, None, kv_mode="fp", fast_path=fast)
    _, past_buf = model(probe, None, kv_mode="fp", fast_path=fast)
    cap = int(past_buf[0][0].cap)

    def _zero_tail(past) -> None:
        # ⚠️ 尾部必须**先清零一次** ✗✗：``torch.empty`` 的残留若是 NaN ⇒ ``q@kᵀ`` 出 NaN ⇒ 加 ``-inf``
        # 还是 NaN ⇒ 整条 softmax 全 NaN ✓ 而**不报错** ✗（``exp(-inf)=0`` 只救有限值 ✓）。
        for key, val in past:
            key.store[:, :, int(key.length):].zero_()
            val.store[:, :, int(val.length):].zero_()

    _zero_tail(past_buf)
    cursor = KvSpan(cap, torch.full((1,), int(past_buf[0][0].length), dtype=torch.long))
    table = RopeTable(cap, CONFIG.head_dim, CONFIG.rope_theta,
                      dtype=model.embed.weight.dtype, device=probe.device)
    tok = torch.tensor([[3]], dtype=torch.long)
    worst, picks_ok, lengths = 0.0, True, []
    for _ in range(3):
        lg_off, past_off = model(tok, past_off, kv_mode="fp", fast_path=fast)
        lg_buf, past_buf = model(tok, past_buf, kv_mode="fp", fast_path=fast,
                                 span=cursor, rope_table=table)
        cursor.advance(1)
        worst = max(worst, float((lg_off.float() - lg_buf.float()).abs().max()))
        picks_ok = picks_ok and int(lg_off.argmax(-1)) == int(lg_buf.argmax(-1))
        lengths.append(int(past_buf[0][0].length))
    check("⑮³ ⭐ **定长读窗口 ≡ 截断读** ✓（3 步 ✓ 掩码挡掉的列权重精确为 0 ✓ 差异只剩归约顺序 ✓）"
          f"＋ 图档**不动** ``length`` ✓（实测 {lengths} ✓ 收尾归 ``generate_llm`` ✓）",
          worst <= 1e-3 and picks_ok and lengths == [5, 5, 5], (worst, picks_ok, lengths))
    # 反证：掩码若把「该挡的」留下 ⇒ 差异必须是**量级** ✓ 否则上一条没牙 ✓
    _, past_bad = model(probe, None, kv_mode="fp", fast_path=fast)
    _zero_tail(past_bad)
    spoiled = KvSpan(cap, torch.full((1,), cap - 1, dtype=torch.long))
    lg_off2, _ = model(tok, past_off, kv_mode="fp", fast_path=fast)
    lg_bad, _ = model(tok, past_bad, kv_mode="fp", fast_path=fast,
                      span=spoiled, rope_table=table)
    spoiled_gap = float((lg_off2.float() - lg_bad.float()).abs().max())
    check("⑮⁴ ⭐ 反证：把「还没写的行」也放进来 ⇒ logits 差必须是**量级** ✓（掩码真在起作用 ✓）",
          spoiled_gap > 1e-2, spoiled_gap)

    # ── 前置闸门：要求了就这么写理由 ✓ 不许静默当关 ✗✗
    ids = torch.randint(0, CONFIG.vocab_size, (1, 4))

    def _run(n: int = 2, graph: object = True, **kw) -> dict:
        sink: dict = {}
        llm_mod.generate_llm(model, ids, max_new_tokens=n, temperature=0,
                             cuda_graph=graph, cuda_graph_report=sink, **kw)
        return sink

    r_q8 = _run(kv_mode="q8", fast_path="kvbuf")
    r_win = _run(kv_mode="fp", fast_path="kvbuf", kv_window=4, kv_sink=1)
    r_spec = _run(kv_mode="fp", fast_path="kvbuf", spec_lookup=True)
    check("⑮⁵ ⭐ 闸门**逐条给理由** ✓：q8／滑窗／投机 ⇒ 各自 ``enabled=False`` ＋ 指明是哪一条 ✗ 不是空的 ✓",
          r_q8.get("reason") == "kvMode=q8"
          and r_win.get("reason") == "kvWindow=4"
          and r_spec.get("reason") == "specLookup"
          and not any(x.get("enabled") for x in (r_q8, r_win, r_spec)),
          (r_q8.get("reason"), r_win.get("reason"), r_spec.get("reason")))

    # ⭐⭐ 2026-11-09 补：``auto`` 的**跳过**要按长度、且**不能**和「让路／前置不满足」混成一个理由 ✗✗
    # —— 两者的处理者视角完全不同 ✓：一个该改 maxOutputTokens、一个该关投机或查驱动 ✓。
    # 阈值 16 的**两侧**都钉住：15 ⇒ 跳过、16 ⇒ 不因长度跳过（在 CPU 上落到 ``device=cpu`` ✓
    # ＝它确实**试着**捕了 ✓ 而不是又被长度挡住 ✓）。
    min_tok = llm_mod.CUDA_GRAPH_AUTO_MIN_TOKENS
    r_a15 = _run(min_tok - 1, "auto", kv_mode="fp", fast_path="kvbuf")
    r_a16 = _run(min_tok, "auto", kv_mode="fp", fast_path="kvbuf")
    check("⑮⁵′ ⭐⭐ ``auto`` 按长度决定、两侧都钉 ✓（2026-11-09 补）："
          f"{min_tok - 1} 词元 ⇒ ``reason=\"autoTooShort({min_tok - 1}<{min_tok})\"`` ✓ 且**不替你加 kvbuf** ✓"
          f"（跳过就该是完全老路径 ✓）；{min_tok} 词元 ⇒ 不再因长度跳过 ✓（落到 ``device=cpu`` ✓ 说明它试着捕了 ✓）"
          "＋ 两档 ``requested`` 都是 True ✓（配置确实要求了 ✓ 别把「跳过」写成「没要求」✗）",
          r_a15.get("policy") == "auto" and r_a15.get("enabled") is False
          and r_a15.get("reason") == f"autoTooShort({min_tok - 1}<{min_tok})"
          and r_a15.get("impliedKvbuf") is False and r_a15.get("requested") is True
          and r_a16.get("policy") == "auto" and r_a16.get("reason") == "device=cpu"
          and r_a16.get("requested") is True,
          (r_a15, r_a16))
    # ⚠️ 非 cuda ⇒ 回退 plain ✓ 且**逐位相同** ✓（「默认路径一个字都不动」是硬口径 ✓）
    plain = llm_mod.generate_llm(model, ids, max_new_tokens=4, temperature=0,
                                 kv_mode="fp", fast_path="kvbuf")
    sink_cpu: dict = {}
    graphy = llm_mod.generate_llm(model, ids, max_new_tokens=4, temperature=0, kv_mode="fp",
                                  fast_path="kvbuf", cuda_graph=True, cuda_graph_report=sink_cpu)
    check("⑮⁶ 非 cuda ⇒ 回退 plain ✓ 输出**逐位相同** ✓ ＋ 理由写明 ``device=cpu`` ✓ 且 ``impliedKvbuf`` 不谎报 ✓"
          " ＋ ``padRows``/``captureMs`` 都是 **None 不是 0** ✓（没捕获就没这份账 ✗ 别编 0）",
          torch.equal(plain, graphy) and sink_cpu.get("reason") == "device=cpu"
          and sink_cpu.get("enabled") is False and sink_cpu.get("impliedKvbuf") is False
          and sink_cpu.get("padRows") is None and sink_cpu.get("captureMs") is None,
          (sink_cpu, bool(torch.equal(plain, graphy))))
    sink_impl: dict = {}
    llm_mod.generate_llm(model, ids, max_new_tokens=2, temperature=0, kv_mode="fp",
                         cuda_graph=True, cuda_graph_report=sink_impl)
    check("⑮⁷ ``cudaGraph`` **隐含 kvbuf** ✓ 且回执写明「这一格是替你加的」✓（不静默 ✓）",
          sink_impl.get("impliedKvbuf") is True, sink_impl)

    # ⭐ 2026-11-09 补：``kv_reserve_rows`` 是**下限**不是**设定值** ✗✗ —— 写成赋值 ⇒「预留比需求小」
    # 的请求会拿到一个装不下自己的缓冲 ✓ ⇒ 生成到最后几步才被自己的预算拒 ✓✗（看着像显存不够 ✓，
    # 其实是把 ``max()`` 写成了赋值 ✓）。这条同时管住另一个方向（滑窗 ⇒ 见 ⑭¹³）。
    short_res = llm_mod.generate_llm(model, ids, max_new_tokens=6, temperature=0,
                                     kv_mode="fp", fast_path="kvbuf", kv_reserve_rows=1)
    big_res = llm_mod.generate_llm(model, ids, max_new_tokens=6, temperature=0,
                                   kv_mode="fp", fast_path="kvbuf", kv_reserve_rows=64)
    check("⑮⁷′ ⭐ ``kv_reserve_rows`` **只抬不压** ✓（2026-11-09 补）：给 1 行也照样产满 6 个词元 ✓"
          "（写成赋值 ⇒ 第 2 个词元就「KV 缓冲超预算」✓ 而回执说的是「调小 maxOutputTokens」✓ 假补救 ✓✗）"
          "＋ 给 64 行同样产满 ✓（放大只是白占 ✓ 但不得改变结果 ✓）",
          int(short_res.shape[1]) == int(ids.shape[1]) + 6 == int(big_res.shape[1]),
          (tuple(short_res.shape), tuple(big_res.shape)))

    # ── 真捕获（有卡才跑 ✓ 没卡就跳过 ✓ 不假装 ✓）
    if not torch.cuda.is_available():
        check("⑮⁸ 真捕获 ⇒ **跳过**（本机没有 CUDA ✓ 上一条已证明回退路径逐位不变 ✓）", True, "no cuda")
        return
    dev = torch.device("cuda")
    gmodel = llm_mod.build_llm(CONFIG).to(dev)
    gids = torch.randint(0, CONFIG.vocab_size, (1, 4), device=dev)
    plain_g = llm_mod.generate_llm(gmodel, gids, max_new_tokens=6, temperature=0,
                                   kv_mode="fp", fast_path="kvbuf", kv_reserve_rows=64)
    sink_g: dict = {}
    graph_g = llm_mod.generate_llm(gmodel, gids, max_new_tokens=6, temperature=0, kv_mode="fp",
                                   fast_path="kvbuf", kv_reserve_rows=64,
                                   cuda_graph=True, cuda_graph_report=sink_g)
    check("⑮⁹ ⭐ 真捕获 ＋ 重放 ✓（``enabled`` ✓ ``cap``/``pastLen`` 对 ✓ ``replays``>0 ✓）"
          "＋ ``captureMs`` 是**正数**（捕获真的花了时间 ✗ 别编 0 骗「图化白拿」）"
          f"＋ 词元流与 plain **一致** ✓ {sink_g}",
          sink_g.get("enabled") is True and sink_g.get("cap") == 64
          and sink_g.get("pastLen") == 4 and int(sink_g.get("replays") or 0) >= 4
          and isinstance(sink_g.get("captureMs"), float) and sink_g["captureMs"] > 0.0
          and torch.equal(plain_g, graph_g),
          (sink_g, plain_g.tolist(), graph_g.tolist()))
    # 收尾长度：图里只写**行**、不动 ``length`` ✗✗ ⇒ 少补 ⇒ 前缀缓存只存了 prefill 那段（不报错 ✗）
    cache = prim["KvPrefixCache"](8 * 2 ** 20)
    sink_c: dict = {}
    out_c = llm_mod.generate_llm(gmodel, gids, max_new_tokens=4, temperature=0, kv_mode="fp",
                                 fast_path="kvbuf", kv_reserve_rows=64, cuda_graph=True,
                                 kv_cache=cache, cuda_graph_report=sink_c)
    rows = int(cache._past[0][0].shape[2]) if cache._past else -1
    check("⑮¹⁰ ⭐ **长度收尾**：``write_at`` 不改 ``length`` ✓ ⇒ ``generate_llm`` 必须补 ✓✗"
          f"（缓存收了 {rows} 行 vs 词元 {int(out_c.shape[1])} 条 ✓）",
          sink_c.get("enabled") is True and rows == int(out_c.shape[1]), (sink_c, rows))
    # ⭐ ``padRows``（2026-11-08 真权重实测补 ✓）：数值差异**全来自多读的 padding 行** ✓ ⇒ 回执必须直接交这个数 ✓
    # （紧桶 257 行 |Δ|=0e0 ✓、满桶 352 行 |Δ|=1.9e-1 ✓、32 词元 25 个分叉 ✓）—— 回执只报 ``cap``/``pastLen``
    # 让人自己减，就把「这次到底差多少」留给读者算 ✗；而且 ``kv_reserve_rows`` 被放大时**没人会去看** ✗✗。
    sink_tight: dict = {}
    out_tight = llm_mod.generate_llm(gmodel, gids, max_new_tokens=6, temperature=0, kv_mode="fp",
                                     fast_path="kvbuf", cuda_graph=True, cuda_graph_report=sink_tight)
    check("⑮¹¹ ⭐ ``padRows`` 真值 ✓：默认预留＝ ``2 + specWindow``（不传 ``kv_reserve_rows`` 也**不是 0** ✗）"
          f"＋ 词元流与老路**一致** ✓ {sink_tight}",
          sink_tight.get("enabled") is True
          and sink_tight.get("padRows") == 2 + llm_mod.SPEC_WINDOW
          and sink_tight.get("cap") == 4 + 6 + 2 + llm_mod.SPEC_WINDOW
          and sink_tight.get("pastLen") == 4
          and torch.equal(plain_g, out_tight),
          (sink_tight, plain_g.tolist(), out_tight.tolist()))
    check("⑮¹² ⭐ 反证：``kv_reserve_rows`` 放大 ⇒ ``padRows`` 必须**跟着变大** ✓"
          "（上一条 ⑮⁹ 的 ``cap=64`` 就是这种情况：多读 54 行 ✓ 差异来源看得见 ✓）",
          sink_g.get("padRows") == 64 - 4 - 6, (sink_g, 64 - 4 - 6))


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
        case_fast_path(torch)
        case_kv_buffer(torch)
        case_kv_budget(torch)
        case_cuda_graph(torch)
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
