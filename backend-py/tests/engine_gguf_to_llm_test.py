"""自检：GGUF 权重 → LlmModel 的**装载**（``engine/gguf_to_llm.py`` ✓ 2026-09-25 起 ✓）。

验的是什么 ✗：把 llama.cpp 的 ggml 命名（``blk.N.attn_q.weight`` ✓）映射成 LlmModel 命名
（``blocks.N.attn.q_proj.weight`` ✓）+ 线性层**转置** ✓（GGUF (in,out) → PyTorch (out,in) ✓）。

判据（都是「看着装上了、其实权重接错了」的形状 ✓✗）：
1. ⭐⭐⭐ **往返恒等** ✗✗：LlmModel 的 ``state_dict`` → 转回 GGUF 命名（反向映射 + 反向转置）→
   再正向映射回来 ⇒ 与原始 ``state_dict`` **逐张量相等** ✓ —— 映射或转置错一处，就有一处对不上 ✗✗；
2. ⭐ **tie embeddings** ✓：tie 时 GGUF 的 ``output.weight`` **跳过** ✓（LlmModel 已 tie ✓）；
3. ⭐ **未知名字返回 None** ✓（不抛错 ✓ —— 真「缺不漏」由 ``load_state_dict(strict=True)`` 兜底 ✓）。

⚠️ 本套**不宣称**能装真 Qwen3 GGUF ✗（真文件的张量名可能带 ``.bias``/``.scale`` 等后缀 ✓ 未测 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_gguf_to_llm_test.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="g2l_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import gguf_to_llm as g2l  # noqa: E402
from app.services.engine import llm as llm_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


#: LlmModel 参数名 → GGUF 张量名（反向 ✓ 独立于实现的正向映射 ✓）
def _reverse(name: str) -> str:
    if name == "embed.weight":
        return "token_embd.weight"
    if name == "norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"
    m = re.match(r"blocks\.(\d+)\.attn\.(q_proj|k_proj|v_proj|o_proj)\.weight", name)
    if m:
        suffix = {"q_proj": "attn_q", "k_proj": "attn_k",
                  "v_proj": "attn_v", "o_proj": "attn_output"}[m.group(2)]
        return f"blk.{m.group(1)}.{suffix}.weight"
    m = re.match(r"blocks\.(\d+)\.ffn\.(gate|up|down)\.weight", name)
    if m:
        return f"blk.{m.group(1)}.ffn_{m.group(2)}.weight"
    m = re.match(r"blocks\.(\d+)\.norm([12])\.weight", name)
    if m:
        return f"blk.{m.group(1)}.{'attn_norm' if m.group(2) == '1' else 'ffn_norm'}.weight"
    raise AssertionError(f"未知 LlmModel 参数名 {name}")


def case_map() -> None:
    check("① 命名映射：token_embd → embed ✓ / output_norm → norm ✓",
          g2l.map_name("token_embd.weight", tie_embeddings=False) == "embed.weight"
          and g2l.map_name("output_norm.weight", tie_embeddings=False) == "norm.weight")
    check("①′ 命名映射：blk.N.attn_q → blocks.N.attn.q_proj ✓（层号透传 ✓）",
          g2l.map_name("blk.7.attn_q.weight", tie_embeddings=False)
          == "blocks.7.attn.q_proj.weight")
    check("①″ 命名映射：output → lm_head ✓（非 tie ✓）",
          g2l.map_name("output.weight", tie_embeddings=False) == "lm_head.weight")
    check("①‴ ⭐ 未知名字 → None ✓（不抛错 ✓）",
          g2l.map_name("blk.0.some_unknown.weight", tie_embeddings=False) is None)


def case_roundtrip() -> None:
    config = llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=2, heads=2, kv_heads=1,
                               head_dim=8, ffn=32, tie_embeddings=False)
    model = llm_mod.build_llm(config)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # 反向：LlmModel → GGUF（独立实现 ✓）
    gguf_weights = {}
    for name, tensor in state.items():
        data = tensor.numpy()
        if name.endswith((".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight",
                          ".gate.weight", ".up.weight", ".down.weight")):
            data = data.T  # 模拟 GGUF 的 (in, out)
        gguf_weights[_reverse(name)] = data

    # 正向：GGUF → LlmModel
    state2 = g2l.gguf_state_dict(gguf_weights, config)

    import torch  # noqa: PLC0415
    ok = set(state2) == set(state) and all(
        tuple(state2[k].shape) == tuple(state[k].shape) and torch.equal(state2[k], state[k])
        for k in state)
    check("② ⭐⭐⭐ 往返恒等 ✓：state_dict → GGUF → 映射回 ⇒ **逐张量相等** ✗✗",
          ok, sorted(set(state) - set(state2)) or sorted(set(state2) - set(state)))


def case_tie() -> None:
    config = llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=1, heads=2, kv_heads=2,
                               head_dim=8, ffn=32, tie_embeddings=True)
    model = llm_mod.build_llm(config)
    check("③ ⭐ tie embeddings ✓（lm_head.weight is embed.weight ✓）",
          model.lm_head.weight is model.embed.weight)
    check("③′ tie 时 GGUF output.weight **跳过** ✓",
          g2l.map_name("output.weight", tie_embeddings=True) is None)


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001 —— GgufDequantError / LlmError 都算「按预期拒」✓
        return True
    return False


def case_infer() -> None:
    meta = {
        "general.architecture": "llama",
        "llama.vocab_size": 32, "llama.embedding_length": 16, "llama.block_count": 2,
        "llama.attention.head_count": 2, "llama.attention.head_count_kv": 1,
        "llama.feed_forward_length": 32, "llama.rope.freq_base": 100.0,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
        "tokenizer.ggml.eos_token_id": 31,
    }
    config = g2l.infer_llm_config(meta)
    check("④ infer: 完整元数据 → LlmConfig ✓（GQA kv_heads=1 ✓ head_dim=hidden/heads=8 ✓ eos ✓）",
          config.vocab_size == 32 and config.hidden == 16 and config.depth == 2
          and config.heads == 2 and config.kv_heads == 1 and config.head_dim == 8
          and config.ffn == 32 and config.eos_id == 31 and config.rms_eps == 1e-5,
          config)

    meta3 = {
        "general.architecture": "qwen3",
        "qwen3.vocab_size": 64, "qwen3.embedding_length": 32, "qwen3.block_count": 4,
        "qwen3.attention.head_count": 4, "qwen3.feed_forward_length": 64,
    }
    config3 = g2l.infer_llm_config(meta3)
    check("④′ infer: qwen3 前缀 ✓（kv_heads 缺省=head ✓ head_dim 推导=8 ✓）",
          config3.vocab_size == 64 and config3.hidden == 32
          and config3.kv_heads == 4 and config3.head_dim == 8, config3)

    meta_fb = {"general.architecture": "qwen3", "llama.vocab_size": 32,
               "llama.embedding_length": 16, "llama.block_count": 1,
               "llama.attention.head_count": 2, "llama.feed_forward_length": 32}
    config_fb = g2l.infer_llm_config(meta_fb)
    check("④″ infer: qwen3 缺字段 ⇒ **回退 llama 前缀** ✓", config_fb.vocab_size == 32)

    check("④‴ infer: 缺必需字段 ⇒ 具名拒绝 ✗（不猜默认 ✗）",
          _raises(lambda: g2l.infer_llm_config({"general.architecture": "llama"})))
    check("④⁴ infer: head_dim 推不出（hidden 不被 heads 整除）⇒ 拒 ✗",
          _raises(lambda: g2l.infer_llm_config({
              "general.architecture": "llama", "llama.vocab_size": 32,
              "llama.embedding_length": 15, "llama.block_count": 1,
              "llama.attention.head_count": 2, "llama.feed_forward_length": 32})))


def main() -> int:
    case_map()
    case_roundtrip()
    case_tie()
    case_infer()
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
