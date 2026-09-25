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

__all__ = ["gguf_state_dict", "load_llm_from_gguf", "map_name"]

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
