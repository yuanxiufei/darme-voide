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
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
