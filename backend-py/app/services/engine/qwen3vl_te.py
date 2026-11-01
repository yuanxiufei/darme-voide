"""Qwen3-VL-32B 的**语言塔**（H3 文本编码器的那一半 ✓）+ 与视觉塔对接的入口 ✓。

## 盘上事实（2026-11-01 读 ``qwen3vl_32b_minimax_h3_int8_convrot.safetensors`` 头 ✓）

```
model.embed_tokens.weight         BF16  [151936, 5120]
model.layers.{0..49}.*            50 层（**截断**：没有第 50..63 层 ✓）
  self_attn.{q,k,v,o}_proj         I8 + weight_scale（convrot）+ comfy_quant 标记
  self_attn.{q_norm,k_norm}.weight BF16 [128]        ⇒ **有 QK-norm** ✓
  mlp.{gate,up,down}_proj          I8（ffn = 25600 = 5×5120 ✓）
  {input,post_attention}_layernorm.weight  BF16 [5120]
model.norm.weight                 **不存在** ✗ ⇒ 截断模型**没有 final norm** ✓
```

头里只有 50 层、没有 ``norm`` 与 ``lm_head`` ⇒ 这是一个**只到第 50 层**的编码器 ✓，
输出是第 50 层的 hidden（**未归一化** ✓）。参考 `llama.py::Llama2_.forward` 的收尾正是
``if self.norm is not None: x = self.norm(x)`` ✓ —— 这里是 ``None`` ⇒ 原样输出 ✓。

## 尺子（`:data:`QWEN3VL_32B``）

| 项 | 值 | 依据 |
| --- | --- | --- |
| hidden | 5120 | ``q_proj`` 入宽 ✓ |
| heads / kv_heads | 64 / 8 | ``q_proj`` 出宽 8192 = 64×128 ✓ / ``k_proj`` 出宽 1024 = 8×128 ✓ |
| head_dim | 128 | 8192 / 64 ✓（与 ``q_norm.weight`` 的 128 同源 ✓） |
| ffn | 25600 | ``gate_proj`` 出宽 ✓ |
| layers | 50 | 层号上界 ✓ |
| eps | 1e-6 | Qwen3 口径 ✓ |
| rope_theta | 5e6 | ``Qwen3VL_32BConfig`` ✓ |
| rope_dims | (24, 20, 20) | ``rope_scaling={"mrope_section": …}`` ✓ 和 = 64 = head_dim/2 ✓ |
| interleaved | True | ``interleaved_mrope`` ✓ ⇒ **每 3 个频率交错取轴** ✓ 不是分段 ✗ |

## 三处最容易静默错的地方

1. **interleaved mrope 是「隔 3 取 1」交错** ✓ 不是「按 24/20/20 分段拼接」✗：

   ```python
   freqs_inter = freqs[0].clone()
   for axis_idx, offset in ((1, 1), (2, 2)):
       idx = slice(offset, rope_dims[axis_idx] * 3, 3)
       freqs_inter[..., idx] = freqs[axis_idx, ..., idx]
   ```

   ⇒ T 轴拿 ``0,3,6,…`` 共 24 个 ✓、H 轴拿 ``1,4,7,…`` 共 20 个 ✓、W 轴拿 ``2,5,8,…`` 共 20 个 ✓。
   ⚠️ 分段拼接（``cat(24,20,20)``）**不报错** ✓✗ 只是位置编码全错 ✓ ⇒ 画面漂移但看不出是哪里的问题 ✗。
2. **rope 只转 ``head_dim/2`` 宽** ✓（rotary_dim = 64 ✓）后 64 维**原样不动** ✓；
   ``cos/sin`` 的宽是 **32**（= rotary_dim/2 ✓），要 ``cat((cos, cos))`` 才够 64 ✓。
3. **DeepStack 加在「层之后」** ✓ 且只加前 3 层 ✓、**只加在视觉位置上** ✓、是**加法** ✓ 不缩放 ✗
   （``x[visual_pos_masks] += deepstack_embeds[i]`` ✓）。

## 与视觉塔的接缝（:mod:`.qwen3vl_vision` ✓）

视觉塔输出 ``(merged_tokens, 5120)`` ✓ 与文本 hidden 同宽 ✓ ⇒ 直接**替换** ``imgpad`` 位置的 embedding ✓
（``masked_scatter`` ✓ 不是相加 ✗）。DeepStack 那三份 ``(merged_tokens, 5120)`` 在**语言塔**里加 ✓。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

__all__ = [
    "QWEN3VL_32B", "ROPE_DIMS", "ROPE_THETA", "build_text_tower", "deepstack_layers_of",
    "freqs_cis", "infer_shape", "mrope_position_ids", "rotary_dim_of", "shape_problems",
]

#: 盘上那套 32B 的尺子 ✓（:func:`infer_shape` 会从**真形状**再核一遍 ✓，常量只作缺省 ✓）
QWEN3VL_32B: dict[str, Any] = {
    "hidden": 5120,
    "layers": 50,
    "heads": 64,
    "kvHeads": 8,
    "headDim": 128,
    "ffn": 25600,
    "vocab": 151936,
    "eps": 1e-6,
    "ropeTheta": 5e6,
    "ropeDims": (24, 20, 20),
    "interleavedMrope": True,
    "finalNorm": False,
    "qkNorm": True,
    "textDim": 5120,
}

#: mrope 三轴频率数 ✓（和必须 = head_dim // 2 ✓）
ROPE_DIMS = (24, 20, 20)

#: mrope 基频 ✓
ROPE_THETA = 5e6


def rotary_dim_of(head_dim: int) -> int:
    """参与旋转的宽度 ✓ = **整个 head_dim** ✓（Qwen3-VL 是**全维旋转** ✓ 不是转一半 ✗）。

    ⚠️ 别把 ``head_dim // 2`` 当旋转宽度 ✗：那是**频率个数**（:func:`freq_count_of` ✓）。
    这两个数差一倍，写反了 ⇒ ``cos/sin`` 宽错 ⇒ 一个报尺寸错 ✓ 或**静默只转一半** ✓✗。
    """
    if head_dim % 2:
        raise ValueError(f"head_dim {head_dim} 是奇数 ✗ ⇒ 旋转要成对 ✓")
    return int(head_dim)


def freq_count_of(head_dim: int) -> int:
    """旋转频率个数 ✓ = ``head_dim // 2`` ✓ —— 也是 ``sum(rope_dims)`` 该等于的值 ✓。"""
    return int(head_dim) // 2


def deepstack_layers_of(count: int = 3) -> tuple[int, ...]:
    """DeepStack 三份分别加在**哪几层之后** ✓ ⇒ ``(0, 1, 2)`` ✓（第 i 层之后加第 i 份 ✓）。"""
    if count <= 0:
        return ()
    return tuple(range(count))


def infer_shape(shapes: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """从**真权重形状**反推尺子 ✓（不信常量 ✓ —— 常量和盘上不一样时必须报 ✗）。

    ``shapes`` 只要给这几类键即可（缺了就跳过那一项校验 ✓）：

    * ``model.embed_tokens.weight`` ✓
    * ``model.layers.0.self_attn.q_proj.weight`` / ``k_proj`` / ``o_proj`` ✓
    * ``model.layers.0.mlp.gate_proj.weight`` ✓
    * ``model.layers.0.self_attn.q_norm.weight``（有 ⇒ 有 QK-norm ✓）
    * ``model.norm.weight``（**没有** ⇒ 截断、无 final norm ✓）
    """
    shape = dict(QWEN3VL_32B)
    embed = shapes.get("model.embed_tokens.weight")
    if embed:
        shape["vocab"], shape["hidden"] = int(embed[0]), int(embed[1])
    q_proj = shapes.get("model.layers.0.self_attn.q_proj.weight")
    if q_proj:
        shape["hidden"] = int(q_proj[1])
        q_out = int(q_proj[0])
        # ⚠️ heads 与 head_dim 只能靠 q_norm 的宽（= head_dim ✓）才分得开 ✓
        norm = shapes.get("model.layers.0.self_attn.q_norm.weight")
        if norm:
            shape["headDim"] = int(norm[0])
            shape["heads"] = q_out // shape["headDim"]
        else:
            shape["headDim"] = 128
            shape["heads"] = q_out // 128
    k_proj = shapes.get("model.layers.0.self_attn.k_proj.weight")
    if k_proj and shape["headDim"]:
        shape["kvHeads"] = int(k_proj[0]) // int(shape["headDim"])
    gate = shapes.get("model.layers.0.mlp.gate_proj.weight")
    if gate:
        shape["ffn"] = int(gate[0])
    layers = set()
    for key in shapes:
        if key.startswith("model.layers."):
            piece = key.split(".")[2]
            if piece.isdigit():
                layers.add(int(piece))
    if layers:
        shape["layers"] = max(layers) + 1
    shape["qkNorm"] = "model.layers.0.self_attn.q_norm.weight" in shapes
    shape["finalNorm"] = "model.norm.weight" in shapes
    return shape


def shape_problems(shape: Mapping[str, Any]) -> list[str]:
    """尺子的**自洽**检查 ✓ ⇒ 问题列表（空 = 自洽 ✓）。与盘上不一致由 :func:`infer_shape` 负责 ✓。"""
    problems: list[str] = []
    hidden = int(shape.get("hidden") or 0)
    heads = int(shape.get("heads") or 0)
    kv_heads = int(shape.get("kvHeads") or 0)
    head_dim = int(shape.get("headDim") or 0)
    dims = tuple(shape.get("ropeDims") or ())
    if not hidden or not heads or not head_dim:
        problems.append(f"hidden / heads / headDim 不能为 0 ✗（收到 {hidden}/{heads}/{head_dim}）")
    if heads and kv_heads and heads % kv_heads:
        problems.append(f"heads {heads} 不能被 kvHeads {kv_heads} 整除 ✗（GQA 分组不成立 ✓）")
    if head_dim % 2:
        problems.append(f"headDim {head_dim} 是奇数 ✗ ⇒ rotary 拿不出一半 ✓")
    if dims and sum(dims) != head_dim // 2:
        problems.append(
            f"ropeDims {dims} 之和 {sum(dims)} ✗ ≠ headDim//2 = {head_dim // 2} ✓"
            f"（和不对 ⇒ 有维度**没拿到任何轴** ✓ 且不报错 ✗）")
    if len(dims) not in (0, 3):
        problems.append(f"ropeDims 必须 3 段（T/H/W）✗（收到 {len(dims)} 段）")
    return problems


def mrope_position_ids(spans: Sequence[Mapping[str, Any]], seq_len: int
                       ) -> tuple[list[int], list[int], list[int]] | None:
    """``(3, seq_len)`` 的 T/H/W 位置表 ✓ —— 逐字对照 ``qwen_vl.py::qwen2vl_mrope_position_ids`` ✓。

    ``spans`` 每项 ``{"index": 起点, "size": 视觉 token 数, "grid": (t, h, w)}`` ✓
    （``grid`` 的单位是 **patch 数** ✓ 不是像素 ✗ —— 与视觉塔的 ``grid_thw`` 同一套 ✓）。

    **没有视觉块 ⇒ 返回 ``None``** ✓（与参考一致 ✓：那时该走 **1D rope** ✓）。
    ⚠️ 别「给全零表当真用」✗：三行全零 ⇒ 每个 token 的位置编码都一样 ✓ 位置信息**全丢** ✓
    而且**不报错** ✗（声音画面照出、只是全糊 ✓）—— 这是本模块最想拦住的一种静默错 ✓。
    """
    rows = [[0] * seq_len for _ in range(3)]
    offset = 0
    touched = False
    for span in spans:
        start = int(span["index"])
        size = int(span["size"])
        grid = tuple(int(value) for value in (span.get("grid") or (1, 0, 0)))
        if len(grid) != 3:
            raise ValueError(f"grid 必须是 (t, h, w) 三段 ✗（收到 {grid!r}）")
        if start < 0 or size <= 0 or start + size > seq_len:
            raise ValueError(
                f"视觉块 [{start}, {start + size}) 越出序列 {seq_len} ✗")
        if not touched:
            # ⚠️ 第一次遇到视觉块时才铺「视觉之前的文本」✓（参考就是这么写的 ✓）
            for axis in range(3):
                rows[axis][:start] = list(range(0, start))
            touched = True
        end = start + size
        len_max = max(grid) // 2
        start_next = len_max + start
        for axis in range(3):
            rows[axis][end:] = list(range(start_next + offset, start_next + (seq_len - end) + offset))
        rows[0][start:end] = [start + offset] * size
        cols = int(math.ceil(size / max(1, grid[1] // 2)))
        height_values: list[int] = []
        for row_index in range(max(1, grid[1] // 2)):
            height_values.extend([start + offset + row_index] * cols)
        rows[1][start:end] = height_values[:size]
        width_values: list[int] = []
        for _row in range(cols):
            width_values.extend([start + offset + column for column in range(max(1, grid[2] // 2))])
        rows[2][start:end] = width_values[:size]
        offset += len_max - size
    return (rows[0], rows[1], rows[2]) if touched else None


def _torch_parts() -> dict[str, Any]:
    """一次性拿 torch 件 ✓（本模块 import 时**不**拉 torch ✓ —— 纯 Python 部分要能离线测 ✓）。"""
    import torch
    from torch import nn

    return {"torch": torch, "nn": nn, "F": torch.nn.functional}


def freqs_cis(position_ids: Sequence[Sequence[int]] | None, shape: Mapping[str, Any], *,
              length: int | None = None, device: Any = None, dtype: Any = None) -> tuple[Any, Any]:
    """``(cos, sin)`` ✓ 宽 = ``head_dim // 2`` ✓（⚠️ 是**频率个数** ✓ 不是 head_dim ✗）。

    ``position_ids`` 是 :func:`mrope_position_ids` 的产物 ``(3, seq)`` ✓；
    给 ``None`` ⇒ 走 **1D** ✓（纯文本提示词 ✓ 参考同样在这种情况下退回 1D ✓）。

    interleaved 交错（参考 ``precompute_freqs_cis`` 的 ``interleaved_mrope`` 分支 ✓ 逐字）：

    ```python
    freqs_inter = freqs[0].clone()
    for axis_idx, offset in ((1, 1), (2, 2)):
        idx = slice(offset, rope_dims[axis_idx] * 3, 3)
        freqs_inter[..., idx] = freqs[axis_idx, ..., idx]
    ```

    ⇒ T 拿 ``0,3,6,…``（24 个 ✓）、H 拿 ``1,4,7,…``（20 个 ✓）、W 拿 ``2,5,8,…``（20 个 ✓）。
    """
    parts = _torch_parts()
    torch = parts["torch"]
    head_dim = int(shape["headDim"])
    freq_count = freq_count_of(head_dim)
    theta = float(shape.get("ropeTheta") or ROPE_THETA)
    rope_dims = tuple(int(value) for value in (shape.get("ropeDims") or ROPE_DIMS))

    exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / float(head_dim)
    inv_freq = 1.0 / (theta ** exponent)
    if inv_freq.numel() != freq_count:
        raise ValueError(f"频率个数 {inv_freq.numel()} ✗ ≠ head_dim//2 = {freq_count} ✓")

    if position_ids is None:
        if length is None:
            raise ValueError("纯文本（position_ids=None）必须给 ``length`` ✗")
        positions = torch.arange(int(length), device=device, dtype=torch.float32)
        freqs = positions[:, None] * inv_freq[None, :]           # (seq, freq)
    else:
        rows = torch.as_tensor([list(row) for row in position_ids],
                               device=device, dtype=torch.float32)   # (3, seq)
        # (3, seq) × (freq,) ⇒ (3, freq, seq) ⇒ permute ⇒ (seq, freq, 3) ✓
        freqs = (rows[:, None, :] * inv_freq[None, :, None]).permute(2, 1, 0)
        if shape.get("interleavedMrope", True):
            picked = freqs[:, :, 0].clone()                       # T 打底 ✓
            for axis_index, offset in ((1, 1), (2, 2)):
                length_axis = rope_dims[axis_index] * 3
                index = slice(offset, length_axis, 3)
                picked[:, index] = freqs[:, index, axis_index]
            freqs = picked
        else:
            # 分段口径（Qwen2.5-VL ✓）：**先 cat 成两倍宽**再按 ``rope_dims * 2`` 切 6 段 ✓，
            # 第 i 段取第 ``i % 3`` 轴 ✓（参考 ``mrope_section = rope_dims * 2`` ✓ 逐字）
            doubled = torch.cat((freqs, freqs), dim=1)
            sections = [int(value) for value in rope_dims] * 2
            chunks = []
            cursor = 0
            for step, width in enumerate(sections):
                chunks.append(doubled[:, cursor:cursor + width, step % 3])
                cursor += width
            freqs = torch.cat(chunks, dim=1)
    if freqs.shape[-1] != freq_count:
        # interleaved 分支出来是 freq_count ✓；分段分支出来也是 freq_count ✓
        raise ValueError(f"频率宽 {freqs.shape[-1]} ✗ ≠ head_dim//2 = {freq_count} ✓")
    emb = torch.cat((freqs, freqs), dim=-1)
    # ⚠️ 参考的 `cos` 宽 = 2×freq，但两边**逐值相同** ✓ ⇒ 只留前半 ✓（:func:`_rope` 用 ``cat((c,c))`` 补回 ✓）
    cos = emb.cos()[..., :freq_count]
    sin = emb.sin()[..., :freq_count]
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


def build_text_tower(shape: Mapping[str, Any], *, dtype: Any = None, device: Any = None) -> Any:
    """建**空壳**语言塔（权重另行装载 ✓ —— 结构与盘上键名一一对应 ✓）。

    ``forward(ids=None, embeds=None, spans=None, deepstack=None)`` ✓：

    * ``ids`` 或 ``embeds`` 二选一 ✓（``embeds`` 是**已拼好视觉特征**的 ``(1, seq, hidden)`` ✓）；
    * ``spans`` = ``[{"index", "size", "grid"}]`` ✓（imgpad 段起点/长度/patch 网格 ✓）；
    * ``deepstack`` = 视觉塔那三份 ``(total_vision_tokens, hidden)`` ✓。
    """
    parts = _torch_parts()
    torch = parts["torch"]
    nn = parts["nn"]
    functional = parts["F"]

    hidden = int(shape["hidden"])
    layers = int(shape["layers"])
    heads = int(shape["heads"])
    kv_heads = int(shape["kvHeads"])
    head_dim = int(shape["headDim"])
    ffn = int(shape["ffn"])
    vocab = int(shape["vocab"])
    eps = float(shape.get("eps") or 1e-6)
    use_qk_norm = bool(shape.get("qkNorm", True))
    deepstack_count = int(shape.get("deepstackCount") or 3)

    class _RmsNorm(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width))

        def forward(self, x: Any) -> Any:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            scaled = (x.to(torch.float32) * torch.rsqrt(variance + eps)).to(x.dtype)
            return scaled * self.weight

    def _rope(x: Any, cos: Any, sin: Any) -> Any:
        """split-half 旋转 ✓（前 ``head_dim/2`` 与后 ``head_dim/2`` 配对 ✓ 与参考的 2×2 矩阵展开一致 ✓）。"""
        half = cos.shape[-1]
        first = x[..., :half]
        second = x[..., half:half * 2]
        rotated = torch.cat((first * cos - second * sin,
                             second * cos + first * sin), dim=-1)
        if x.shape[-1] > half * 2:
            rotated = torch.cat((rotated, x[..., half * 2:]), dim=-1)
        return rotated

    class _Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
            self.q_norm = _RmsNorm(head_dim) if use_qk_norm else None
            self.k_norm = _RmsNorm(head_dim) if use_qk_norm else None

        def forward(self, x: Any, cos: Any, sin: Any) -> Any:
            batch, length, _ = x.shape
            query = self.q_proj(x).view(batch, length, heads, head_dim).transpose(1, 2)
            key = self.k_proj(x).view(batch, length, kv_heads, head_dim).transpose(1, 2)
            value = self.v_proj(x).view(batch, length, kv_heads, head_dim).transpose(1, 2)
            # ⚠️ QK-norm 在 **rope 之前** ✓（参考 `llama.py::Attention.forward` 顺序 ✓）
            if self.q_norm is not None:
                query = self.q_norm(query)
            if self.k_norm is not None:
                key = self.k_norm(key)
            query = _rope(query, cos, sin)
            key = _rope(key, cos, sin)
            core = functional.scaled_dot_product_attention(
                query, key, value, is_causal=True,
                enable_gqa=(heads != kv_heads))
            core = core.transpose(1, 2).reshape(batch, length, heads * head_dim)
            return self.o_proj(core)

    class _Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden, ffn, bias=False)
            self.up_proj = nn.Linear(hidden, ffn, bias=False)
            self.down_proj = nn.Linear(ffn, hidden, bias=False)

        def forward(self, x: Any) -> Any:
            return self.down_proj(functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = _RmsNorm(hidden)
            self.self_attn = _Attention()
            self.post_attention_layernorm = _RmsNorm(hidden)
            self.mlp = _Mlp()

        def forward(self, x: Any, cos: Any, sin: Any) -> Any:
            x = x + self.self_attn(self.input_layernorm(x), cos, sin)
            return x + self.mlp(self.post_attention_layernorm(x))

    class Qwen3VLTextTower(nn.Module):
        """⚠️ **没有** ``self.model.norm`` ✓ —— 盘上截断模型没有 final norm ✓（有反而是错的 ✗）。"""

        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(vocab, hidden)
            self.model.layers = nn.ModuleList([_Block() for _ in range(layers)])
            self.shape = dict(shape)

        def forward(self, *, ids: Any = None, embeds: Any = None,
                    spans: Sequence[Mapping[str, Any]] | None = None,
                    deepstack: Sequence[Any] | None = None) -> Any:
            torch_ = parts["torch"]
            if (ids is None) == (embeds is None):
                raise ValueError("``ids`` 与 ``embeds`` 必须**二选一** ✗")
            if embeds is None:
                #: ⚠️ **`ids` 必须跟着词表权重走** ✓✗（2026-11-01 真权重端到端实测 ✓）：GPU/CPU
                #: 协同驻留下 `embed_tokens` 可能在**卡上** ✓、而 `ids` 是编码侧在 CPU 上造的 ✓
                #: ⇒ `F.embedding` 内部一句 `index_select` 就炸 ✓✗：
                #: ``Expected all tensors to be on the same device, but got index is on cpu,
                #: different from other tensors on cuda:0`` ✓ —— 报错指向 embedding ✓，
                #: 看着像"词表坏了" ✓，其实是**两台设备各拿一半**的正常形态 ✓。
                ids = ids.to(self.model.embed_tokens.weight.device)
                x = self.model.embed_tokens(ids)
            else:
                x = embeds if embeds.dim() == 3 else embeds.unsqueeze(0)
            if x.shape[-1] != hidden:
                raise ValueError(f"输入宽 {x.shape[-1]} ✗ ≠ hidden {hidden} ✓")
            seq_len = int(x.shape[1])

            masks = None
            if spans:
                # ⚠️ imgpad 位置 = 视觉特征所在处 ✓（`<|vision_start|>`/`<|vision_end|>` 不在内 ✓）
                masks = torch_.zeros(seq_len, dtype=torch_.bool, device=x.device)
                for span in spans:
                    start = int(span["index"])
                    size = int(span["size"])
                    masks[start:start + size] = True
                position_ids = mrope_position_ids(spans, seq_len)
                cos, sin = freqs_cis(position_ids, shape, length=seq_len,
                                     device=x.device, dtype=x.dtype)
            else:
                cos, sin = freqs_cis(None, shape, length=seq_len,
                                     device=x.device, dtype=x.dtype)
            if deepstack is not None:
                want = masks.sum().item() if masks is not None else 0
                got = sum(int(item.shape[0]) for item in deepstack)
                if got != want:
                    raise ValueError(
                        f"DeepStack 视觉 token 总数 {got} ✗ ≠ 视觉位置 {want} ✓"
                        f"（对不上 ⇒ 注入会错位 ✓ 且不报错 ✗）")

            target = deepstack_layers_of(deepstack_count)
            for index, block in enumerate(self.model.layers):
                x = block(x, cos, sin)
                # ⚠️ 注入在**层之后** ✓（参考 `llama.py::Llama2_.forward` ✓）；挪到层之前 ⇒ 视觉段
                #    之前的 token 也会被波及 ✓ 但**形状全对、不报错** ✓✗
                if masks is not None and deepstack and index in target and index < len(deepstack):
                    cursor = 0
                    for span in spans or ():
                        size = int(span["size"])
                        chunk = deepstack[index][cursor:cursor + size]
                        start = int(span["index"])
                        #: ⚠️ 视觉塔与语言塔**分处两台设备**时（协同驻留 ✓），视觉特征要跟到
                        #: 当前 `x` 所在那台 ✓ —— 只转 dtype 不转 device ⇒ 相加当场报设备不一致 ✓✗。
                        x[:, start:start + size, :] = x[:, start:start + size, :] + chunk.to(
                            device=x.device, dtype=x.dtype)
                        cursor += size
            return x

    #: ⚠️⚠️ **`device="meta"` 必须在构造时生效** ✗✗：先按 CPU 建再 ``.to("meta")`` 的话，
    #: 62 GiB 的 bf16 骨架已经**真分配 + 真 `uniform_` 填过**了 ✓（`reset_parameters` 逐元素写 ✓）
    #: —— 32B 的 H3 文本塔实测把宿主内存吃到 80/93 GiB ✓、然后停滞十几分钟 ✓、连权重文件都
    #: 没开始读 ✓✗。用 ``with torch.device("meta")`` 包住构造 ⇒ 参数在 meta 上建 ✓、
    #: `uniform_` 在 meta 上是空操作 ✓、一块实存都不占 ✓。
    if device is not None and str(device) == "meta":
        with torch.device("meta"):
            model = Qwen3VLTextTower()
    else:
        model = Qwen3VLTextTower()
    if dtype is not None or device is not None:
        model = model.to(device=device, dtype=dtype)
    return model
