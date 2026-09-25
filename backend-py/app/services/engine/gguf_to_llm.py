"""GGUF 权重 → :class:`llm.LlmModel` 的**装载**（2026-09-25 起）。

## 为什么单独一层

:mod:`gguf_dequant` 反量化出 ``{张量名: fp32 数组}``，但**没映射到 LlmModel 的参数名**，也没做
GGUF → PyTorch 的**转置** ✗。本模块补上这一步 ✓：把 llama.cpp 的 ggml 命名（``blk.N.attn_q.weight`` ✓）
映射成 LlmModel 的命名（``blocks.N.attn.q_proj.weight`` ✓），并把线性层权重**转置** ✓。

## 两条口径（不猜 ✗，事实来源：llama.cpp 的 ggml 约定 ✓）

1. **命名**：GGUF 用 ggml 的 ``blk.N.*`` ✓ → LlmModel 用 ``blocks.N.*`` ✓（映射表见下 ✓）；
2. **转置**：``attn_q/k/v``、``attn_output``、``ffn_gate/up/down`` 这 6 类线性层要 **transpose** ✗
   （GGUF 存 ``(in, out)`` ✓，PyTorch ``nn.Linear`` 要 ``(out, in)`` ✓）；norm / embedding / lm_head **不转** ✓；
3. **tie embeddings**：``config.tie_embeddings=True`` 时 LlmModel 的 ``lm_head.weight is embed.weight`` ✓
   ⇒ GGUF 里的 ``output.weight`` **跳过** ✗（已经 tie ✓ —— 强制装会多出一个形状一样的 key ✗）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import gguf as gguf_mod
from . import gguf_dequant
from . import llm as llm_mod

__all__ = ["gguf_state_dict", "infer_llm_config", "load_llm_from_gguf", "map_name"]

#: GGUF 层名 → LlmModel 层名 ✓（``{}`` 是层号 ✓）
_LAYER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"blk\.(\d+)\.attn_q\.weight", "blocks.{}.attn.q_proj.weight"),
    (r"blk\.(\d+)\.attn_k\.weight", "blocks.{}.attn.k_proj.weight"),
    (r"blk\.(\d+)\.attn_v\.weight", "blocks.{}.attn.v_proj.weight"),
    (r"blk\.(\d+)\.attn_output\.weight", "blocks.{}.attn.o_proj.weight"),
    (r"blk\.(\d+)\.ffn_gate\.weight", "blocks.{}.ffn.gate.weight"),
    (r"blk\.(\d+)\.ffn_up\.weight", "blocks.{}.ffn.up.weight"),
    (r"blk\.(\d+)\.ffn_down\.weight", "blocks.{}.ffn.down.weight"),
    (r"blk\.(\d+)\.attn_norm\.weight", "blocks.{}.norm1.weight"),
    (r"blk\.(\d+)\.ffn_norm\.weight", "blocks.{}.norm2.weight"),
)

#: 需要**转置**的层（GGUF (in,out) → PyTorch (out,in) ✓）
_TRANSPOSE_SUFFIXES = (".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight",
                       ".gate.weight", ".up.weight", ".down.weight")


def map_name(gguf_name: str, *, tie_embeddings: bool) -> str | None:
    """GGUF 张量名 → LlmModel 参数名 ✓（``None`` = 跳过：tie 时的 ``output.weight`` ✓）。

    ⚠️ 未知名字也返回 ``None`` ✗（不是抛错 ✓）—— 真正的「缺不漏」由 ``load_state_dict(strict=True)``
    兜底 ✓（漏一个 key 它就会报「missing key」✗）。
    """
    if gguf_name == "token_embd.weight":
        return "embed.weight"
    if gguf_name == "output_norm.weight":
        return "norm.weight"
    if gguf_name == "output.weight":
        return None if tie_embeddings else "lm_head.weight"
    for pattern, template in _LAYER_PATTERNS:
        match = re.match(pattern, gguf_name)
        if match:
            return template.format(match.group(1))
    return None


def gguf_state_dict(weights: dict[str, Any], config: llm_mod.LlmConfig) -> dict[str, Any]:
    """反量化后的 ``{GGUF 名: numpy 数组}`` → LlmModel 的 ``state_dict``（torch 张量 ✓ 含转置 ✓）。"""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    state: dict[str, Any] = {}
    for gguf_name, arr in weights.items():
        target = map_name(gguf_name, tie_embeddings=config.tie_embeddings)
        if target is None:
            continue
        if target.endswith(_TRANSPOSE_SUFFIXES):
            arr = np.ascontiguousarray(arr.T)
        state[target] = torch.from_numpy(np.ascontiguousarray(arr))
    return state


def load_llm_from_gguf(path: str | Path, config: llm_mod.LlmConfig) -> Any:
    """GGUF 文件 → 装载好权重的 :class:`llm.LlmModel` ✓。

    ⚠️ **逐张量**读 + 反量化 + 转置 ✓（大文件不整份提进内存 ✗）；``strict=True`` ⇒
    映射漏一个 / 形状不对一个都**当场报错** ✓✗（不静默装半套 ✗）。
    """
    info = gguf_mod.inspect(path)
    model = llm_mod.build_llm(config)
    state = gguf_state_dict(
        {name: _read_tensor(path, info, tensor)
         for name, tensor in info.tensors.items()},
        config,
    )
    model.load_state_dict(state, strict=True)
    return model


def _read_tensor(path: str | Path, info: Any, tensor: Any) -> Any:
    with open(path, "rb") as handle:
        handle.seek(info.data_start + tensor.offset)
        blob = handle.read(tensor.nbytes)
    arr = gguf_dequant.dequantize_tensor(blob, tensor.type_id, tensor.numel)
    return arr.reshape(tensor.shape)


def infer_llm_config(metadata: dict[str, Any], *, arch: str | None = None) -> llm_mod.LlmConfig:
    """从 GGUF 元数据推导 :class:`llm.LlmConfig` ✓（架构参数**全从权重读** ✓ 不猜 ✗）。

    ⚠️ GGUF 的架构参数前缀是 ``general.architecture``（``llama`` / ``qwen2`` / ``qwen3`` ✓），
    字段名同族：``{arch}.embedding_length`` / ``{arch}.block_count`` / ``{arch}.attention.head_count``
    / ``{arch}.attention.head_count_kv``（GQA ✓）/ ``{arch}.feed_forward_length`` / ``{arch}.rope.freq_base``
    / ``{arch}.attention.layer_norm_rms_epsilon`` 等 ✓（Qwen3 GGUF 偶尔沿用 ``llama.`` 前缀 ⇒ 回退读 ✓）。

    ⚠️ 缺必需字段 ⇒ **具名拒绝** ✗（不猜默认值 ✗ —— 猜了会装出「名字对、形状全错」的模型 ✗）。
    """
    if arch is None:
        arch = str(metadata.get("general.architecture") or "llama")

    def _get(name: str) -> Any:
        for prefix in (arch, "llama"):
            value = metadata.get(f"{prefix}.{name}")
            if value is not None:
                return value
        return None

    vocab_size = _get("vocab_size")
    hidden = _get("embedding_length")
    depth = _get("block_count")
    heads = _get("attention.head_count")
    kv_heads = _get("attention.head_count_kv")
    ffn = _get("feed_forward_length")
    head_dim = _get("attention.head_dim")
    rope_theta = _get("rope.freq_base")
    rms_eps = _get("attention.layer_norm_rms_epsilon")

    required = {"vocab_size": vocab_size, "embedding_length": hidden, "block_count": depth,
                "attention.head_count": heads, "feed_forward_length": ffn}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise GgufDequantError(
            f"GGUF 元数据缺架构参数：{missing} ✗（arch={arch} ✓）⇒ 推不出 LlmConfig ✗ "
            f"（不猜默认值 ✗）")

    if kv_heads is None:
        kv_heads = heads
    if head_dim is None:
        if int(hidden) % int(heads) != 0:
            raise GgufDequantError(
                f"embedding_length（{hidden}）不被 head_count（{heads}）整除 ✗ ⇒ head_dim 推不出 ✗")
        head_dim = int(hidden) // int(heads)

    eos = metadata.get("tokenizer.ggml.eos_token_id")
    pad = metadata.get("tokenizer.ggml.pad_token_id")
    return llm_mod.LlmConfig(
        vocab_size=int(vocab_size), hidden=int(hidden), depth=int(depth),
        heads=int(heads), kv_heads=int(kv_heads), head_dim=int(head_dim),
        ffn=int(ffn), rope_theta=float(rope_theta if rope_theta is not None else 10000.0),
        rms_eps=float(rms_eps if rms_eps is not None else 1e-6),
        tie_embeddings=bool(metadata.get(f"{arch}.tie_embeddings", False)),
        pad_id=int(pad) if pad is not None else 0,
        eos_id=int(eos) if eos is not None else None,
    )
