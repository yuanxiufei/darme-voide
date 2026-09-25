"""自研 decoder-only LLM 推理（2026-09-25 起）。

## 为什么有这一层

本仓文本生成此前走 **ollama**（本地 11434 的 HTTP 服务）✗ —— 那是**调用**，不是**实现** ✗。
本模块把它换成**自己实现**的 decoder-only transformer ✓：RMSNorm + GQA（分组查询注意力）+
RoPE + SwiGLU + **因果掩码** + **KV cache** ✓ —— Qwen3 / Llama 这类 LLM 的架构 ✓。

## 与其它模块的分工（不重叠 ✓）

* :mod:`tokenizer_bpe` / :mod:`tokenizer_hub` —— 文本 → token id ✓（词表是**权重的一部分** ✓）；
* :mod:`gguf` —— 读 GGUF 元数据（张量表 / 量化方案 ✓；**张量反量化见后续** ✓）；
* 本模块 —— token id → **逐 token 生成**（前向 + KV cache + 采样 ✓）；
* :mod:`quant` —— 低精度权重反量化（GGUF 的 k-quant 尚未接 ✓ 已写进缺口 ✓）。

## 两条硬约束（本仓纪律 ✓）

1. **架构参数全部显式** ✗（:class:`LlmConfig` ✓）—— 不写死 Qwen3 的具体值 ✓，从权重/配置读 ✓；
2. **懒导入 torch** ✓（模块级不 import ✓ ⇒ 没装也能读配置/常量 ✓）。

## 验证

自检（``engine_llm_test`` ✓）用**缩小版**模型走**同一条**前向 ✓（本仓纪律 ✓）：
因果掩码 ✓ / RoPE 旋转不变量 ✓ / GQA 分组 ✓ / KV cache 逐 token 不变量 ✓ / 采样形状 ✓。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "LlmConfig",
    "apply_rope",
    "build_llm",
    "generate_llm",
    "repeat_kv",
    "rope_freqs",
]


class LlmError(ValueError):
    """LLM 配置/输入不合法 ✓（**明确报错** ✗ 不静默猜 ✓）。"""


@dataclass(frozen=True)
class LlmConfig:
    """decoder-only LLM 的**显式**结构参数 ✓（不猜具体模型 ✗）。

    ⚠️ ``kv_heads`` 是 **GQA** 的关键 ✓：``kv_heads < heads`` 表示分组查询注意力
    （若干 query 头共享一组 KV ✓）；``kv_heads == heads`` 退化为标准 MHA ✓。
    """

    vocab_size: int
    hidden: int
    depth: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = True
    pad_id: int = 0
    eos_id: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise LlmError(f"vocab_size 必须 > 1（收到 {self.vocab_size} ✗）")
        for name, value in (("hidden", self.hidden), ("depth", self.depth),
                            ("heads", self.heads), ("head_dim", self.head_dim),
                            ("ffn", self.ffn), ("kv_heads", self.kv_heads)):
            if value <= 0:
                raise LlmError(f"{name} 必须 > 0（收到 {value} ✗）")
        if self.heads % self.kv_heads != 0:
            raise LlmError(f"GQA：heads（{self.heads}）必须被 kv_heads（{self.kv_heads}）整除 ✗")
        if self.head_dim % 2 != 0:
            raise LlmError(f"head_dim 必须是偶数（RoPE 配对旋转 ✗ 收到 {self.head_dim} ✗）")
        if self.rope_theta <= 0 or self.rms_eps <= 0:
            raise LlmError("rope_theta / rms_eps 必须 > 0 ✗")


def build_llm(config: LlmConfig) -> Any:
    """构造 :class:`LlmModel` ✓（**懒导入 torch** ✓）。"""
    return _build()["LlmModel"](config)


def _build() -> dict[str, Any]:
    """造出需要 torch 的类/函数 ✓（模块级定义会强制 import torch ✗ ⇒ 放进工厂 ✓）。"""
    import math  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    class RMSNorm(nn.Module):
        """**RMSNorm** ✓：``x / sqrt(mean(x²) + eps) * weight`` ✓（LLM 用这个，不是 LayerNorm ✗）。"""

        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: Any) -> Any:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    def rope_freqs(positions: Any, head_dim: int, theta: float) -> tuple[Any, Any]:
        """预计算 RoPE 的 cos/sin ✓（``(seq, head_dim//2)`` ✓ —— **GPT-NeoX / Qwen 风格** ✓）。"""
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device).float()
                                    / head_dim))
        freqs = torch.outer(positions.float(), inv_freq)
        return torch.cos(freqs), torch.sin(freqs)

    def apply_rope(x: Any, cos: Any, sin: Any) -> Any:
        """对最后两维做旋转 ✓（**前半/后半配对** ✓ —— GPT-NeoX / Qwen 风格 ✓）。"""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def repeat_kv(kv: Any, n_rep: int) -> Any:
        """GQA：``(b, kv_heads, s, d)`` → ``(b, kv_heads*n_rep, s, d)`` ✓（query 头对齐 KV 头 ✓）。"""
        if n_rep == 1:
            return kv
        b, kv_h, s, d = kv.shape
        return kv[:, :, None, :, :].expand(b, kv_h, n_rep, s, d).reshape(b, kv_h * n_rep, s, d)

    class GQAAttention(nn.Module):
        """分组查询注意力 ✓（GQA ✓ + RoPE ✓ + **因果掩码** ✓ + KV cache ✓）。"""

        def __init__(self, config: LlmConfig) -> None:
            super().__init__()
            self.heads = config.heads
            self.kv_heads = config.kv_heads
            self.head_dim = config.head_dim
            self.n_rep = config.heads // config.kv_heads
            self.q_proj = nn.Linear(config.hidden, config.heads * config.head_dim, bias=False)
            self.k_proj = nn.Linear(config.hidden, config.kv_heads * config.head_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden, config.kv_heads * config.head_dim, bias=False)
            self.o_proj = nn.Linear(config.heads * config.head_dim, config.hidden, bias=False)

        def forward(self, x: Any, cos: Any, sin: Any,
                    past_kv: tuple[Any, Any] | None = None) -> tuple[Any, tuple[Any, Any]]:
            b, s, _ = x.shape
            q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            past_len = 0
            if past_kv is not None:
                pk, pv = past_kv
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
                past_len = int(pk.shape[2])
            new_kv = (k, v)
            k_rep = repeat_kv(k, self.n_rep)
            v_rep = repeat_kv(v, self.n_rep)
            scores = q @ k_rep.transpose(-2, -1) / math.sqrt(self.head_dim)
            # ⭐ 因果掩码 ✓：当前位置（past_len+i）只许看 ≤ 它自己的位置 ✓✗
            mask = torch.triu(torch.full((s, past_len + s), float("-inf"), device=x.device),
                              diagonal=past_len + 1)
            attn = torch.softmax(scores + mask, dim=-1)
            out = attn @ v_rep
            out = out.transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
            return self.o_proj(out), new_kv

    class SwiGLU(nn.Module):
        """**SwiGLU** MLP ✓：``gate`` 与 ``up`` 两条 ✓ ``silu(gate) * up`` ✓（无 bias ✓）。"""

        def __init__(self, config: LlmConfig) -> None:
            super().__init__()
            self.gate = nn.Linear(config.hidden, config.ffn, bias=False)
            self.up = nn.Linear(config.hidden, config.ffn, bias=False)
            self.down = nn.Linear(config.ffn, config.hidden, bias=False)

        def forward(self, x: Any) -> Any:
            return self.down(F.silu(self.gate(x)) * self.up(x))

    class TransformerBlock(nn.Module):
        """pre-norm 一层 ✓：RMSNorm → 注意力 → 残差 → RMSNorm → SwiGLU → 残差 ✓。"""

        def __init__(self, config: LlmConfig) -> None:
            super().__init__()
            self.norm1 = RMSNorm(config.hidden, config.rms_eps)
            self.attn = GQAAttention(config)
            self.norm2 = RMSNorm(config.hidden, config.rms_eps)
            self.ffn = SwiGLU(config)

        def forward(self, x: Any, cos: Any, sin: Any,
                    past_kv: tuple[Any, Any] | None = None) -> tuple[Any, tuple[Any, Any]]:
            attended, new_kv = self.attn(self.norm1(x), cos, sin, past_kv)
            x = x + attended
            x = x + self.ffn(self.norm2(x))
            return x, new_kv

    class LlmModel(nn.Module):
        """token id → logits ✓（``(b, s)`` → ``(b, s, vocab)`` ✓ + KV cache ✓）。"""

        def __init__(self, config: LlmConfig) -> None:
            super().__init__()
            self.config = config
            self.head_dim = config.head_dim
            self.rope_theta = config.rope_theta
            self.embed = nn.Embedding(config.vocab_size, config.hidden, padding_idx=config.pad_id)
            self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.depth)])
            self.norm = RMSNorm(config.hidden, config.rms_eps)
            self.lm_head = nn.Linear(config.hidden, config.vocab_size, bias=False)
            if config.tie_embeddings:
                self.lm_head.weight = self.embed.weight

        def forward(self, ids: Any, past_kv: Any = None) -> tuple[Any, Any]:
            batch, seq = ids.shape
            past_len = 0 if past_kv is None else int(past_kv[0][0].shape[2])
            positions = torch.arange(past_len, past_len + seq, device=ids.device)
            cos, sin = rope_freqs(positions, self.head_dim, self.rope_theta)
            x = self.embed(ids)
            new_kv: list[tuple[Any, Any]] = []
            for index, block in enumerate(self.blocks):
                layer_kv = None if past_kv is None else past_kv[index]
                x, kv = block(x, cos, sin, layer_kv)
                new_kv.append(kv)
            x = self.norm(x)
            return self.lm_head(x), new_kv

    def generate_llm(model: Any, ids: Any, *, max_new_tokens: int = 64,
                     temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0,
                     eos_id: int | None = None) -> Any:
        """逐 token 生成 ✓（greedy / temperature / top-p / top-k ✓ + 早停 ✓）。

        ⚠️ ``temperature=0`` ⇒ greedy ✓（直接 argmax ✓ —— 保证可复现 ✓）。
        """
        if max_new_tokens < 1:
            raise LlmError(f"max_new_tokens 必须 ≥1（收到 {max_new_tokens} ✗）")
        if temperature < 0:
            raise LlmError(f"temperature 不能为负（收到 {temperature} ✗）")
        past_kv = None
        current = ids
        generated: list[Any] = []
        for _ in range(max_new_tokens):
            logits, past_kv = model(current, past_kv)
            next_logits = logits[:, -1, :]
            if temperature == 0:
                token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                if top_k > 0:
                    topk = torch.topk(probs, min(top_k, probs.shape[-1]))
                    mask = torch.full_like(probs, float("-inf"))
                    mask.scatter_(1, topk.indices, topk.values)
                    probs = torch.softmax(mask, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    remove = cum > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_probs[remove] = 0.0
                    probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                token = torch.multinomial(probs, 1)
            current = token
            generated.append(token)
            if eos_id is not None and bool((token == eos_id).all()):
                break
        return torch.cat([ids] + generated, dim=1)

    return {
        "torch": torch, "nn": nn, "F": F, "RMSNorm": RMSNorm, "GQAAttention": GQAAttention,
        "SwiGLU": SwiGLU, "TransformerBlock": TransformerBlock, "LlmModel": LlmModel,
        "rope_freqs": rope_freqs, "apply_rope": apply_rope, "repeat_kv": repeat_kv,
        "generate_llm": generate_llm,
    }


def __getattr__(name: str) -> Any:
    """惰性导出 ✓（**先查 ``__all__``** ✓ 再建积木 ✓ —— 与 h3_form 同一纪律 ✓）。"""
    if name.startswith("_") or name not in __all__:
        raise AttributeError(f"llm 没有 {name} ✓")
    return _build()[name]
