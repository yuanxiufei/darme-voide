"""MiniMax Music3 的**语义塔**（caption/lyrics ⇒ 条件序列 ``(b, F, 8 × 4096)`` ✓）。

一句话：Music3 的「文本塔」不是把句子编码成若干个 token 向量 ✓，而是**逐帧生成一段音频符号**
（25 帧/秒 ✓），再把 **LM 隐状态 ＋ 7 层残差码本隐状态** 拼成宽度 32768 的条件序列 ✓✗
（``8 × 4096`` ✓，与 :mod:`dit_music3` 的 ``context_dim`` 同一个数 ✓）。

⚠️ 这一族最容易「形状全对、歌全是噪声」的四件事 ✓✗（都靠 :class:`Music3TextTower` 钉住 ✓）：

① **条件帧率是 25 Hz** ✓，而潜帧率是 ``44100/512 = 86.1328 Hz`` ✓（见 :mod:`dit_music3` ✓）——
   两者差 3.445 倍 ✓，把 25 当 86 会得到一个「时长只有三分之一」的歌 ✓✗。
② **CFG 的两条支路不是同一句** ✓：无条件支路把 ``tokens[1:-2]`` 整段换成 ``<|audio_cfg|>`` ✓
   （保留首尾的 ``<|im_start|>`` / ``<|audio_end|>``-前一格 ✓），不是「空提示词」✗。
③ **c0 的 top-k 门限只从条件支路取** ✓（``_guided_c0`` ✓）：门限若取自引导后的分数 ✓，
   会把「无条件支路更喜欢的 token」放进候选集 ✓ —— 听着像「偶尔跑调」✓✗，其实是自造退化 ✓。
④ **深度解码是 7 步残差** ✓（不是一次出 7 个头 ✓）：第 ``k`` 步的输入要拼上第 ``k−1`` 步的
   码本嵌入 ✓，漏拼 ⇒ 每层都对、合起来是另一首歌 ✓✗。

参考实现：``comfy/text_encoders/minimax_music.py`` ＋ ``comfy/ldm/minimax_music/{ar,prompt}.py`` ✓
（本模块逐段转写 ✓，语义按**逐字对齐**而非「看着像」✓）。

盘上三件套 ✓：``text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors``（648 张量 ✓
＝ 36 层 Qwen3-8B 主干 ＋ 4 层深度解码器 ＋ 剪枝后的嵌入/输出头 ＋ **内嵌 tokenizer.json** ✓）／
``diffusion_models/minimax_music3_dit_fp16.safetensors``（见 :mod:`dit_music3` ✓）／
``vae/minimax_music3_dav.safetensors``（见 :mod:`vae_music3` ✓）。
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

try:                                                          # 与同族模块同一口径 ✓
    import torch as _TORCH
    _HAS_TORCH = True
except Exception:                                             # pragma: no cover —— 没装 torch 也要能看常量 ✓
    _TORCH = None
    _HAS_TORCH = False


def has_torch() -> bool:
    return _HAS_TORCH


def _torch() -> Any:
    if _TORCH is None:                                       # pragma: no cover
        raise RuntimeError("这条线要 torch ✓（venv 里那份 3.12 ＋ 2.14 ✓）")
    return _TORCH


__all__ = [
    "AUDIO_CODE_OFFSET",
    "AUDIO_FRAMES_PER_SECOND",
    "CFG_SCALE",
    "CFG_TOP_K",
    "CONTEXT_DIM",
    "MUSIC3_AR_FACTS",
    "Music3TextTower",
    "SPECIAL_TOKEN_IDS",
    "build_prompt",
    "build_tower",
    "clean_caption",
    "derive_seed",
    "generate_hidden_frames",
    "load_tower",
    "normalize_lyrics",
    "sample_topk",
    "validate_tokenizer",
]


#: 特殊 token ✓（**逐字抄**参考实现 ✓ —— 151644 与 151645 是 Qwen 的 ``im_start``/``im_end`` ✓，
#: 而 151669/151670 才是音频段起止 ✓；两组差 25 ⇒ 抄错一个就是「提示词被当成音频码」✓✗）。
SPECIAL_TOKEN_IDS: dict[str, int] = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|audio_cfg|>": 151654,
    "<|audio_start|>": 151669,
    "<|audio_end|>": 151670,
    "<|caption_start|>": 151671,
    "<|caption_end|>": 151672,
    "<|lyrics_start|>": 151673,
    "<|lyrics_end|>": 151674,
}

#: 音频码在**完整词表**里的起点 ✓（剪枝版不用它 ✓，留着是为了认清「151675 = 0 号码本的第 0 个码」✓）。
AUDIO_CODE_OFFSET = 151675

#: 0 号（语义最粗那本）码本的词表大小 ✓ = 剪枝后 ``embed_tokens_audio`` 的行数 ✓。
C0_VOCAB_SIZE = 16384

#: **停止码**与「剪枝后码 ⇒ 码本下标」的偏移 ✓（``code_of(0) = 0`` ✓、``code_of(1) = 0`` ✓）。
STOP_TOKEN = 0
CODE_OFFSET = 1

MAX_PROMPT_TOKENS = 5000
MAX_AUDIO_FRAMES = 9000
AUDIO_FRAMES_PER_SECOND = 25
CFG_SCALE = 1.5
CFG_TOP_K = 50

#: 主干几何 ✓（Qwen3-8B ✓）：36 层／hidden 4096／q 32 头／kv 8 头／head_dim 128／rope θ = 1e6 ✓
#: ⇒ ``qkv_proj`` 的 6144 = 4096 + 1024 + 1024 ✓（**GQA** ✓，不是 3 × 4096 ✓✗）。
HIDDEN = 4096
LAYERS = 36
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0
RMS_EPS = 1e-6
INTERMEDIATE = 12288

#: 深度解码器几何 ✓：4 层／16 头（head_dim 256 ⇒ q/k/v 各 4096 ✓ 是 **MHA 不是 GQA** ✓）／
#: 中间宽 6144（``gate_up_proj`` 12288 = 2 × 6144 ✓）／位置表 16 格 ✓（实际最多用到 9 格 ✓）。
DEPTH_LAYERS = 4
DEPTH_HEADS = 16
DEPTH_HEAD_DIM = 256
DEPTH_INTERMEDIATE = 6144
DEPTH_POSITIONS = 16

NUM_CODEBOOKS = 8
CODEBOOK_WIDTH = 1024

#: 喂给 DiT 的宽度 ✓ = ``NUM_CODEBOOKS × HIDDEN`` ✓（8 组：基座隐状态 1 组 ＋ 残差 7 组 ✓）。
CONTEXT_DIM = NUM_CODEBOOKS * HIDDEN

MUSIC3_AR_FACTS: dict[str, Any] = {
    "keyCount": 648,
    "textEncoderFile": "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
    "layers": LAYERS,
    "hidden": HIDDEN,
    "qHeads": Q_HEADS,
    "kvHeads": KV_HEADS,
    "headDim": HEAD_DIM,
    "ropeTheta": ROPE_THETA,
    "depthLayers": DEPTH_LAYERS,
    "depthHeads": DEPTH_HEADS,
    "depthPositions": DEPTH_POSITIONS,
    "numCodebooks": NUM_CODEBOOKS,
    "codebookWidth": CODEBOOK_WIDTH,
    "c0Vocab": C0_VOCAB_SIZE,
    "stopToken": STOP_TOKEN,
    "audioFramesPerSecond": AUDIO_FRAMES_PER_SECOND,
    "maxPromptTokens": MAX_PROMPT_TOKENS,
    "maxAudioFrames": MAX_AUDIO_FRAMES,
    "cfgScale": CFG_SCALE,
    "cfgTopK": CFG_TOP_K,
    "contextDim": CONTEXT_DIM,
    "quant": "int8 常驻（per-row scale ＋ convrot ✓，9.2 GB 文件 ✓；反量化成 bf16 要 16.4 GiB ⇒ 挤不下卡 ✓✗）",
    "prunedEmbedding": "embed_tokens_prefill 只有 151675 行 ✓（缺的尾部是音频码区 ✓，音频码改走 embed_tokens_audio ✓）",
    "prunedHead": "lm_head_pruned 只有 16385 行 ✓（= 16384 个音频码 ＋ 1 个停止码 ✓，字面上没有别的 token ✓）",
}


_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LYRIC_TAG_RE = re.compile(r"\s*(\[[^\]]+\])\s*")


def _remove_markdown_format(text: str) -> str:
    """去掉 markdown 的「格式壳」✓ —— 保留**字面词** ✓（caption 里的 ``**`` / ``#`` / ``- `` 会
    被当成歌词结构 ✓✗：参考实现连这段一起去，逐字对齐 ✓）。
    """
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", raw_line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text.replace("• ", "").replace("    ", "")


def clean_caption(caption: str) -> str:
    """caption 归一 ✓：``<|bpm 120|>`` 这类标签换成 ``bpm is 120`` ✓（**标签名当主体** ✓）。"""

    def replace_special(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(replace_special, caption)
    text = _remove_markdown_format(text)
    return re.sub(r"\n{2,}", "\n", text)


def normalize_lyrics(lyrics: str) -> str:
    """歌词归一 ✓：**段落标签小写** ✓、`` ^ `` 当换行 ✓、开头补 ``[start]`` ✓。"""
    parts = _LYRIC_TAG_RE.split(lyrics)
    text = "\n".join(part.lower() if part.startswith("[") else part for part in parts if part)
    text = text.replace(" ^ ", "\n")
    return f"[start]\n{text}"


def build_prompt(caption: str, lyrics: str) -> str:
    """拼出**字面**提示串 ✓（顺序是 ``caption`` 在前、``lyrics`` 在后 ✓✗ —— 反过来照样能生成 ✓，
    只是**唱错词** ✓）。
    """
    return (
        "<|im_start|><|caption_start|>"
        f"{clean_caption(caption)}"
        "<|caption_end|><|lyrics_start|>"
        f"{normalize_lyrics(lyrics)}"
        "<|lyrics_end|><|im_end|><|audio_start|>"
    )


def validate_tokenizer(tokenizer: Any) -> list[str]:
    """核特殊 token 的 id ✓✗（**词表对不上时不会报错** ✓：整条提示会被切成普通文本 ✓，
    歌照样出 ✓，只是内容跟提示词无关 ✓）。
    """
    problems: list[str] = []
    for token, expected in SPECIAL_TOKEN_IDS.items():
        got = tokenizer.token_id(token) if hasattr(tokenizer, "token_id") else None
        if got is None:
            got = _token_id_via_encode(tokenizer, token)
        if got != expected:
            problems.append(f"{token} 期望 {expected} ✗ 实得 {got} ✓")
    return problems


def _token_id_via_encode(tokenizer: Any, token: str) -> Any:
    try:
        ids = list(tokenizer.encode(token, add_special_tokens=False))
    except TypeError:                                    # 有的实现不吃关键字 ✓
        ids = list(tokenizer.encode(token))
    return ids[0] if len(ids) == 1 else (None if not ids else ids[0])


def derive_seed(seed: int, *parts: Any) -> int:
    """``blake2b`` 派生 ✓（**逐字抄**参考实现的 ``person=b"minimax-ttm"`` ✓）；同 seed 同提示词
    ⇒ 同一首歌 ✓✗（差一个字节 ⇒ 完全另一首 ✓）。
    """
    digest = hashlib.blake2b(digest_size=8, person=b"minimax-ttm")
    digest.update(int(seed).to_bytes(8, "little", signed=False))
    for part in parts:
        value = str(part).encode("utf-8")
        digest.update(len(value).to_bytes(4, "little"))
        digest.update(value)
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def sample_topk(logits: Any, top_k: int, generator: Any) -> Any:
    """top-k 采样 ✓：先把 ``< top_k`` 的名次打成 ``-inf`` ✓，再归一化后 ``multinomial`` ✓
    （参考实现先归一化再抽 ✓ —— 直接扔未归一的分数给 ``multinomial`` 也对 ✓，但那样
    「分数里有 inf」时会静默变成均匀分布 ✓✗ ✓）。
    """
    torch = _torch()
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_k = min(int(top_k), int(values.shape[-1]))
    threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probabilities = torch.nan_to_num(torch.softmax(values, dim=-1), nan=0.0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


@dataclass(frozen=True)
class KvCache:
    """**预分配**的 KV 缓存 ✓（``(b, kv 头, 容量, head_dim)`` ✓）。

    ✗ 不用「每步 ``cat`` 一个新张量」✓：1500 帧就是 1500 次整段拷贝 ✓，36 层 × 每次几 MB ⇒
    O(n²) 的显存带宽 ✓，出歌时间从分钟级掉到小时级 ✓。容量按 ``提示词长度 + 帧数 + 1`` 一次给足 ✓。
    """

    keys: Any
    values: Any
    index: int = 0

    @property
    def capacity(self) -> int:
        return int(self.keys.shape[2])

    def length(self) -> int:
        return int(self.index)

    def advance(self, count: int) -> "KvCache":
        return KvCache(self.keys, self.values, int(self.index) + int(count))


def _split_half_rope(x: Any, cos: Any, sin: Any) -> Any:
    """**split-half** 旋转位置编码 ✓（配对是 ``i ↔ i + d/2`` ✓，不是相邻两维 ✓✗）。

    ⚠️ 与 :mod:`dit_music3` 的 ``_apply_rope_positions`` 同一口径 ✓（那边逐位对拍过
    ``comfy_kitchen.apply_rope_split_half`` ✓）；这里只是位置来自 KV 计数、不是外挂坐标 ✓。
    """
    torch = _torch()
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    #: ⚠️ 位置表要补成 ``(1, 1, L, half)`` ✓✗：``x`` 是 ``(批, 头, L, half)`` ✓，
    #: 只 ``unsqueeze(1)`` 得 ``(L, 1, half)`` ✓ 会按「头的维」去对位 ✓✗
    #: （实数 56 个 token、32 个头的场合当场报 ``32 != 56`` ✓；凑巧数值相等的场合**静默算错** ✓）。
    cos = cos.unsqueeze(0).unsqueeze(0) if cos.dim() == 2 else cos
    sin = sin.unsqueeze(0).unsqueeze(0) if sin.dim() == 2 else sin
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


@functools.lru_cache(maxsize=1)
def _make_modules() -> Any:
    """惰性造类 ✓（torch 没装时本模块**仍可 import** ✓：只做形状与口径查询 ✓）。"""
    if not has_torch():
        raise RuntimeError("没装 torch ✗ ⇒ 建不了 Music3 语义塔 ✓")
    torch = _torch()
    nn = torch.nn
    fn = torch.nn.functional

    class RmsNorm(nn.Module):
        """Qwen3 的 RMSNorm ✓：**带权** ✓、eps 1e-6 ✓、内部升 f32 算再降回 ✓。"""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: Any) -> Any:
            return fn.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), RMS_EPS)

    class Attention(nn.Module):
        """Qwen3 注意力 ✓：**合并** ``qkv_proj`` ✓、GQA 32:8 ✓、q/k 各带一个 128 维 RMSNorm ✓、
        split-half RoPE ✓、**v 不转** ✓、cache 就地写 ✓。
        """

        def __init__(self, hidden: int = HIDDEN, heads: int = Q_HEADS,
                     kv_heads: int = KV_HEADS, head_dim: int = HEAD_DIM) -> None:
            super().__init__()
            self.heads = heads
            self.kv_heads = kv_heads
            self.head_dim = head_dim
            self.qkv_proj = nn.Linear(hidden, head_dim * (heads + 2 * kv_heads), bias=False)
            self.q_norm = RmsNorm(head_dim)
            self.k_norm = RmsNorm(head_dim)
            self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)

        def forward(self, x: Any, cache: KvCache, start: int) -> Any:
            batch, length = int(x.shape[0]), int(x.shape[1])
            qkv = self.qkv_proj(x)
            q, k, v = qkv.split([self.heads * self.head_dim,
                                 self.kv_heads * self.head_dim,
                                 self.kv_heads * self.head_dim], dim=-1)
            q = self.q_norm(q.reshape(batch, length, self.heads, self.head_dim)).transpose(1, 2)
            k = self.k_norm(k.reshape(batch, length, self.kv_heads, self.head_dim)).transpose(1, 2)
            v = v.reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)

            positions = torch.arange(start, start + length, device=x.device)
            inv = 1.0 / (ROPE_THETA ** (torch.arange(0, self.head_dim, 2,
                                                     device=x.device, dtype=torch.float32)
                                        / self.head_dim))
            freqs = torch.outer(positions.float(), inv)
            cos, sin = freqs.cos().to(q.dtype), freqs.sin().to(q.dtype)
            q, k = _split_half_rope(q, cos, sin), _split_half_rope(k, cos, sin)

            stop = start + length
            cache.keys[:, :, start:stop] = k
            cache.values[:, :, start:stop] = v
            k_all, v_all = cache.keys[:, :, :stop], cache.values[:, :, :stop]
            if self.heads != self.kv_heads:
                repeat = self.heads // self.kv_heads
                k_all = k_all.repeat_interleave(repeat, dim=1)
                v_all = v_all.repeat_interleave(repeat, dim=1)
            out = fn.scaled_dot_product_attention(q, k_all, v_all, is_causal=length > 1,
                                                  scale=1.0 / math.sqrt(self.head_dim))
            return self.o_proj(out.transpose(1, 2).reshape(batch, length, self.heads * self.head_dim))

    class Mlp(nn.Module):
        """**合并** ``gate_up_proj`` ✓：前半 gate ✓、后半 up ✓（顺序反了 ⇒ 每层都能跑 ✓、歌全是噪声 ✓✗）。"""

        def __init__(self, hidden: int = HIDDEN, intermediate: int = INTERMEDIATE) -> None:
            super().__init__()
            self.gate_up_proj = nn.Linear(hidden, intermediate * 2, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)

        def forward(self, x: Any) -> Any:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            return self.down_proj(fn.silu(gate) * up)

    class Block(nn.Module):
        def __init__(self, hidden: int = HIDDEN, intermediate: int = INTERMEDIATE,
                     heads: int = Q_HEADS, kv_heads: int = KV_HEADS,
                     head_dim: int = HEAD_DIM) -> None:
            super().__init__()
            self.input_layernorm = RmsNorm(hidden)
            self.self_attn = Attention(hidden, heads, kv_heads, head_dim)
            self.post_attention_layernorm = RmsNorm(hidden)
            self.mlp = Mlp(hidden, intermediate)

        def forward(self, x: Any, cache: KvCache, start: int) -> Any:
            x = x + self.self_attn(self.input_layernorm(x), cache, start)
            return x + self.mlp(self.post_attention_layernorm(x))

    class DepthAttention(nn.Module):
        """深度解码器的注意力 ✓：**MHA**（16 × 256 ✓）、**没有 RoPE** ✓✗（加了也能跑 ✓，
        但「第几步」的信息会跟绝对位置打架 ✓ —— 序列短（≤ 9 格）时症状最轻、最难查 ✓）。
        """

        def __init__(self, hidden: int = HIDDEN, heads: int = DEPTH_HEADS,
                     head_dim: int = DEPTH_HEAD_DIM) -> None:
            super().__init__()
            self.heads = heads
            self.head_dim = head_dim
            self.qkv_proj = nn.Linear(hidden, hidden * 3, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)

        def forward(self, x: Any) -> Any:
            batch, length = int(x.shape[0]), int(x.shape[1])
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
            shape = (batch, length, self.heads, self.head_dim)
            q, k, v = (part.reshape(shape).transpose(1, 2) for part in (q, k, v))
            out = fn.scaled_dot_product_attention(q, k, v, is_causal=length > 1,
                                                  scale=1.0 / math.sqrt(self.head_dim))
            return self.o_proj(out.transpose(1, 2).reshape(batch, length, self.heads * self.head_dim))

    class DepthMlp(nn.Module):
        def __init__(self, hidden: int = HIDDEN, intermediate: int = DEPTH_INTERMEDIATE) -> None:
            super().__init__()
            self.gate_up_proj = nn.Linear(hidden, intermediate * 2, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)

        def forward(self, x: Any) -> Any:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            return self.down_proj(fn.silu(gate) * up)

    class DepthBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = RmsNorm(HIDDEN)
            self.self_attn = DepthAttention()
            self.post_attention_layernorm = RmsNorm(HIDDEN)
            self.mlp = DepthMlp()

        def forward(self, x: Any) -> Any:
            x = x + self.self_attn(self.input_layernorm(x))
            return x + self.mlp(self.post_attention_layernorm(x))

    class DepthDecoder(nn.Module):
        """从「基座隐状态 ＋ c0 嵌入」推出**另外 7 本码本** ✓。

        ⚠️ 位置表只有 16 格 ✓（第 0 格给基座隐状态 ✓，第 1–8 格给 c0 与 7 步残差 ✓）。
        """

        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(HIDDEN, HIDDEN, bias=False)
            self.pos_embedding = nn.Embedding(DEPTH_POSITIONS, HIDDEN)
            self.audio_heads = nn.ModuleList([nn.Linear(HIDDEN, CODEBOOK_WIDTH, bias=False)
                                              for _ in range(NUM_CODEBOOKS - 1)])
            self.layers = nn.ModuleList([DepthBlock() for _ in range(DEPTH_LAYERS)])
            self.norm = RmsNorm(HIDDEN)

        def forward(self, sequence: Any) -> Any:
            positions = torch.arange(int(sequence.shape[1]), device=sequence.device)
            x = sequence + self.pos_embedding(positions).unsqueeze(0).to(sequence.dtype)
            for layer in self.layers:
                x = layer(x)
            return self.norm(x)

    class Backbone(nn.Module):
        """Qwen3-8B 主干 ＋ 剪枝后的两端 ✓（键名与权重文件**逐字同名** ✓，装载才可能 strict ✓）。"""

        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens_prefill = nn.Embedding(AUDIO_CODE_OFFSET, HIDDEN)
            self.embed_tokens_audio = nn.Embedding(C0_VOCAB_SIZE, HIDDEN)
            self.audio_extra_embedding = nn.Embedding(CODEBOOK_WIDTH * (NUM_CODEBOOKS - 1), HIDDEN)
            self.layers = nn.ModuleList([Block() for _ in range(LAYERS)])
            self.norm = RmsNorm(HIDDEN)
            self.lm_head_pruned = nn.Linear(HIDDEN, C0_VOCAB_SIZE + 1, bias=False)
            self.audio_decoder = DepthDecoder()

        def forward(self, embeds: Any, cache: KvCache) -> Any:
            """``(b, L, 4096)`` ＋ 缓存 ⇒ ``(末层隐状态, 推进后的缓存)`` ✓。

            ⚠️ 三层口径都要**逐字**对齐参考实现 ✓✗：
            ① 末尾 **要**过 ``norm`` ✓（``llama.py`` 里那句 ``if self.norm is not None: x = self.norm(x)`` ✓）——
               漏了 ⇒ 形状全对 ✓、``lm_head_pruned`` 吃的是未归一化的隐状态 ⇒ 采样温度虚高、歌在乱唱 ✓✗；
            ② 位置 = **缓存游标** ✓（``arange(start, start+length)`` ✓），不是「每次都从 0 数」✗；
            ③ 缓存游标由**调用方**推进 ✓（:meth:`KvCache.advance` ✓）—— 本函数只写 ``[start:start+length]`` ✓，
               自己加会被调用方再加一次 ✓✗ ⇒ 位置整体翻倍、RoPE 从第二帧起就错 ✓。
            """
            start = cache.length()
            length = int(embeds.shape[1])
            x = embeds
            for layer in self.layers:
                x = layer(x, cache, start)
            return self.norm(x), cache.advance(length)

    class TextTower(nn.Module):
        """整塔 ✓：``self.model`` 这一层不是装饰 ✓ —— 权重文件里的键就是 ``model.xxx`` ✓✗。"""

        def __init__(self) -> None:
            super().__init__()
            self.model = Backbone()

    return {"RmsNorm": RmsNorm, "Attention": Attention, "Mlp": Mlp, "Block": Block,
            "DepthDecoder": DepthDecoder, "Backbone": Backbone, "TextTower": TextTower}


def expected_shapes() -> dict[str, tuple[int, ...]]:
    """**该有的 327 个权重**与形状 ✓（盘上那份 648 键 = 327 权重 ＋ 160 ``weight_scale``
    ＋ 160 ``comfy_quant`` ＋ 1 ``tokenizer_json`` ✓ —— 后三者不是模块里的参数 ✓）。

    ⚠️ 这张表不是「抄一遍形状」✓：它是 :func:`tower_problems` 的判据 ✓ —— 少了哪一格、
    哪一格宽度错，装载前就能点名 ✓✗（宽度错到能被广播时**装得上、跑得出噪声** ✓）。
    """
    out: dict[str, tuple[int, ...]] = {
        "model.embed_tokens_prefill.weight": (AUDIO_CODE_OFFSET, HIDDEN),
        "model.embed_tokens_audio.weight": (C0_VOCAB_SIZE, HIDDEN),
        "model.audio_extra_embedding.weight": (CODEBOOK_WIDTH * (NUM_CODEBOOKS - 1), HIDDEN),
        "model.norm.weight": (HIDDEN,),
        "model.lm_head_pruned.weight": (C0_VOCAB_SIZE + 1, HIDDEN),
        "model.audio_decoder.projection.weight": (HIDDEN, HIDDEN),
        "model.audio_decoder.pos_embedding.weight": (DEPTH_POSITIONS, HIDDEN),
        "model.audio_decoder.norm.weight": (HIDDEN,),
    }
    for index in range(NUM_CODEBOOKS - 1):
        out[f"model.audio_decoder.audio_heads.{index}.weight"] = (CODEBOOK_WIDTH, HIDDEN)
    for index in range(LAYERS):
        prefix = f"model.layers.{index}"
        out[f"{prefix}.input_layernorm.weight"] = (HIDDEN,)
        out[f"{prefix}.post_attention_layernorm.weight"] = (HIDDEN,)
        out[f"{prefix}.self_attn.q_norm.weight"] = (HEAD_DIM,)
        out[f"{prefix}.self_attn.k_norm.weight"] = (HEAD_DIM,)
        out[f"{prefix}.self_attn.qkv_proj.weight"] = (HEAD_DIM * (Q_HEADS + 2 * KV_HEADS), HIDDEN)
        out[f"{prefix}.self_attn.o_proj.weight"] = (HIDDEN, HIDDEN)
        out[f"{prefix}.mlp.gate_up_proj.weight"] = (INTERMEDIATE * 2, HIDDEN)
        out[f"{prefix}.mlp.down_proj.weight"] = (HIDDEN, INTERMEDIATE)
    for index in range(DEPTH_LAYERS):
        prefix = f"model.audio_decoder.layers.{index}"
        out[f"{prefix}.input_layernorm.weight"] = (HIDDEN,)
        out[f"{prefix}.post_attention_layernorm.weight"] = (HIDDEN,)
        #: ⚠️ 深度解码器是 **MHA** ✓ ⇒ ``qkv_proj`` 是 ``3 × 4096`` ✓（不是 ``4096 + 2 × 1024`` ✗）
        out[f"{prefix}.self_attn.qkv_proj.weight"] = (HIDDEN * 3, HIDDEN)
        out[f"{prefix}.self_attn.o_proj.weight"] = (HIDDEN, HIDDEN)
        out[f"{prefix}.mlp.gate_up_proj.weight"] = (DEPTH_INTERMEDIATE * 2, HIDDEN)
        out[f"{prefix}.mlp.down_proj.weight"] = (HIDDEN, DEPTH_INTERMEDIATE)
    return out


#: 量化**旁路**键 ✓（文件里有、模块里没有 ✓）：``weight_scale`` 由 ``Int8Linear`` 的 buffer 接 ✓、
#: ``comfy_quant`` 是**逐层量化说明**（convrot 分组 ✓）✓、``tokenizer_json`` 是内嵌分词器 ✓。
_NOT_WEIGHT_SUFFIXES = (".weight_scale", ".comfy_quant", ".scale")


def tower_problems(shapes: dict[str, Any]) -> list[str]:
    """文件头部 vs 本模块 ⇒ **人话**问题清单 ✓（空 ⇒ 结构自洽 ✓）。"""
    expected = expected_shapes()
    problems: list[str] = []
    missing = [key for key in expected if key not in shapes]
    if missing:
        problems.append(f"缺 {len(missing)} 个权重 ✗：{missing[:4]}{' …' if len(missing) > 4 else ''}")
    for key, want in expected.items():
        got = shapes.get(key)
        if got is None:
            continue
        if tuple(int(dim) for dim in got) != want:
            problems.append(f"{key} 形状 {tuple(int(dim) for dim in got)} ✗ ≠ 期望 {want} ✓")
    extra = [key for key in shapes
             if key not in expected
             and key != "tokenizer_json"
             and not key.endswith(_NOT_WEIGHT_SUFFIXES)]
    if extra:
        problems.append(f"多出 {len(extra)} 个不认识的权重 ✗：{extra[:4]}{' …' if len(extra) > 4 else ''}")
    total = len(shapes)
    if not problems and total != int(MUSIC3_AR_FACTS["keyCount"]):
        problems.append(
            f"键总数 {total} ✗ ≠ 本族自述的 {MUSIC3_AR_FACTS['keyCount']} ✓"
            f"（= {len(expected)} 权重 ＋ 每层一条 ``weight_scale`` / ``comfy_quant`` ＋ 1 内嵌分词器 ✓）"
            f" ⇒ 很可能混进了另一族的文件 ✓")
    return problems


def build_tower(*, dtype: Any = None, device: Any = "meta") -> Any:
    """建**空壳** ✓（默认 ``device="meta"`` ✓ —— 一块实存都不占 ✓）。

    ⚠️ 必须能在 meta 上建 ✗✗：这份塔的 bf16 骨架是 **13.8 GiB** ✓，而 int8 文件只有 9.2 GB ✓ ——
    按真设备建 ⇒ 先 ``uniform_`` 填满 13.8 GiB ✓ 再被 :class:`Int8Linear` 逐层换掉 ✓✗
    （换掉的块**不会**立刻还给系统 ✓，与 H3 那条 62 GiB 的坑同一个形状 ✓）。
    """
    torch = _torch()
    mods = _make_modules()
    previous_dtype = torch.get_default_dtype()
    try:
        previous_device = torch.get_default_device()
    except AttributeError:                                    # 老 torch 没有这层 ✓
        previous_device = "cpu"
    try:
        if dtype is not None:
            torch.set_default_dtype(dtype)
        if device is not None:
            torch.set_default_device(device)
        return mods["TextTower"]()
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_default_device(previous_device or "cpu")


def _as_ids(torch: Any, input_ids: Any, device: Any) -> Any:
    """提示词 ⇒ ``(1, L)`` long ✓（吃 list / tuple / 一维／二维张量 ✓）。"""
    if hasattr(input_ids, "shape"):
        ids = input_ids
        if int(ids.dim()) == 1:
            ids = ids.unsqueeze(0)
        if int(ids.dim()) != 2 or int(ids.shape[0]) != 1:
            raise ValueError(f"提示词 token 要是 (1, L) ✗，收到 {tuple(ids.shape)} ✓")
        return ids.to(device=device, dtype=torch.long)
    return torch.tensor([list(input_ids)], dtype=torch.long, device=device)


class Music3TextTower:
    """把 :class:`TextTower` 包成「能**生成**的塔」✓（与 ``h3_encode.H3TeRuntime`` 同一个角色 ✓）。

    ## 为什么这层不是装饰 ✗

    装配侧只问两件事 ✓：``output_dim``（喂给 DiT 的条件宽度 ✓ ＝ 32768 ✓）与 ``max_length``
    （提示词上限 ✓）。**生成**这件事（逐帧 AR 解码 ✓）才是这一族的「编码」✓ —— 它产出的是
    ``(F, 32768)`` 的条件**序列** ✓，不是「一个句向量」✗。把生成逻辑挂在这里，装配侧的
    ``text_kind="music3"`` 才有东西可调 ✓。

    ## 与参考实现**逐字**对齐的四处 ✓
    ``_guided_c0`` / ``_depth_codes`` / ``_embed_audio_frame`` / ``_sample_c0`` 是
    ``comfy/ldm/minimax_music/ar.py`` 的同名方法 ✓（连 `topk` 门限只取条件支路这种细节也照抄 ✓）。
    """

    #: 条件宽度 ✓（DiT 的 ``context_dim`` ✓）—— 装配侧按这个名字问 ✓。
    output_dim = CONTEXT_DIM
    #: 提示词 token 上限 ✓（超了直接拒 ✓，不静默截断 ✗）。
    max_length = MAX_PROMPT_TOKENS

    def __init__(self, module: Any, *, audio_vocab_size: int = CODEBOOK_WIDTH,
                 num_codebooks: int = NUM_CODEBOOKS, dtype: Any = None) -> None:
        self.module = module
        self.audio_vocab_size = int(audio_vocab_size)
        self.num_codebooks = int(num_codebooks)
        #: ``8 ** -0.5`` ✓ —— 8 本码本的嵌入**相加**后要除一次 ✓✗（漏了 ⇒ 量级大 2.83 倍 ✓）。
        self.embedding_scale = float(self.num_codebooks) ** -0.5
        backbone = module.model
        self.pruned_embedding = hasattr(backbone, "embed_tokens_prefill")
        self.pruned_lm_head = hasattr(backbone, "lm_head_pruned")
        self._dtype = dtype
        #: 上一次生成的事实账 ✓（帧数／是否撞停止码／容量 ✓）—— 回执里要用 ✓。
        self.last_stats: dict[str, Any] = {}

    # ---- 基础设施 ------------------------------------------------------

    @property
    def backbone(self) -> Any:
        return self.module.model

    def device(self) -> Any:
        weight = getattr(self.backbone, "embed_tokens_prefill", None)
        weight = getattr(weight, "weight", None)
        return weight.device if weight is not None else _torch().device("cpu")

    def execution_dtype(self) -> Any:
        """算的精度 ✓ ＝ **权重**的精度 ✓（与 ``dit_qwen21`` 同一条口径 ✓）。"""
        torch = _torch()
        if self._dtype is not None:
            return self._dtype
        weight = getattr(self.backbone.embed_tokens_prefill, "weight", None)
        dtype = getattr(weight, "dtype", None)
        return dtype if dtype is not None and dtype.is_floating_point else torch.float32

    def new_cache(self, capacity: int) -> KvCache:
        """按 ``(b, kv 头, 容量, head_dim)`` **一次**开足 ✓（见 :class:`KvCache` 的 docstring ✓）。"""
        torch = _torch()
        device = self.device()
        dtype = self.execution_dtype()
        shape = (2, KV_HEADS, int(capacity), HEAD_DIM)
        return KvCache(torch.empty(shape, device=device, dtype=dtype),
                       torch.empty(shape, device=device, dtype=dtype), 0)

    # ---- 参考实现的四个同名方法 ----------------------------------------

    def _guided_c0(self, logits: Any, cfg_scale: float, top_k: int) -> Any:
        """CFG 引导 ＋ **门限只取条件支路** ✓✗（参考 :meth:`MiniMaxMusic3AR._guided_c0` ✓）。"""
        torch = _torch()
        conditioned = logits[0:1].float()
        unconditioned = logits[1:2].float()
        guided = unconditioned + (conditioned - unconditioned) * float(cfg_scale)
        k = min(int(top_k), int(conditioned.shape[-1]))
        threshold = torch.topk(conditioned, k, dim=-1).values[..., -1, None]
        return guided.masked_fill(conditioned < threshold, -float("inf"))

    def _depth_codes(self, hidden: Any, c0: Any, c0_embed: Any, generator: Any,
                     dtype: Any, cfg_scale: float, top_k: int) -> tuple[Any, Any]:
        """基座隐状态 ＋ c0 ⇒ **另外 7 本码本** ✓ 与**它们的隐状态** ✓（宽度 7 × 4096 ✓）。"""
        torch = _torch()
        backbone = self.backbone
        decoder = backbone.audio_decoder
        sequence = [decoder.projection(hidden).unsqueeze(1)]
        sequence.append(decoder.projection(c0_embed).unsqueeze(1))
        codes = [c0]
        hidden_parts = []
        for index in range(1, self.num_codebooks):
            out = decoder(torch.cat(sequence, dim=1))[:, -1]
            #: ⚠️ **只留条件支路** ✓：两条支路的残差隐状态混进来 ⇒ 条件序列里一半是无条件支路 ✓✗
            hidden_parts.append(out[:1].detach())
            logits = decoder.audio_heads[index - 1](out)
            conditioned = logits[:1].float()
            unconditioned = logits[1:2].float()
            #: ⚠️ 深度这几步**不套** top-k 门限 ✓（参考实现就是这么写的 ✓ —— 门限只服务 c0 ✓）
            code = sample_topk(unconditioned + (conditioned - unconditioned) * float(cfg_scale),
                               top_k, generator).repeat(2)
            codes.append(code)
            if index < self.num_codebooks - 1:
                embedding = backbone.audio_extra_embedding(
                    code + (index - 1) * self.audio_vocab_size).to(dtype)
                sequence.append(decoder.projection(embedding).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    def _embed_c0(self, codes: Any, dtype: Any) -> Any:
        if self.pruned_embedding:
            return self.backbone.embed_tokens_audio(codes).to(dtype)
        return self.backbone.embed_tokens(codes + AUDIO_CODE_OFFSET).to(dtype)

    def _embed_audio_frame(self, codes: Any, dtype: Any) -> Any:
        """一帧 8 个码 ⇒ 一个 ``(2, 1, 4096)`` 的**反馈嵌入** ✓（8 份相加再乘 ``8**-0.5`` ✓）。"""
        torch = _torch()
        c0 = self._embed_c0(codes[:, 0], dtype)
        offsets = torch.arange(self.num_codebooks - 1, device=codes.device) * self.audio_vocab_size
        extra = self.backbone.audio_extra_embedding(
            codes[:, 1:] + offsets.unsqueeze(0)).to(dtype).sum(dim=1)
        return ((c0 + extra) * self.embedding_scale).unsqueeze(1)

    def _sample_c0(self, hidden: Any, cfg_scale: float, top_k: int,
                   generator: Any) -> tuple[Any, Any, int]:
        """从隐状态采 **c0** ✓ ⇒ ``(码本下标, 原始码, 停止码)`` ✓。"""
        torch = _torch()
        backbone = self.backbone
        if self.pruned_lm_head:
            guided = self._guided_c0(backbone.lm_head_pruned(hidden).float(), cfg_scale, top_k)
            code = sample_topk(guided, top_k, generator)
            stop_token, offset = STOP_TOKEN, CODE_OFFSET
        else:
            logits = backbone.lm_head(hidden).float()
            stop_token = SPECIAL_TOKEN_IDS["<|audio_end|>"]
            mask = torch.ones(int(logits.shape[-1]), dtype=torch.bool, device=logits.device)
            mask[AUDIO_CODE_OFFSET:AUDIO_CODE_OFFSET + C0_VOCAB_SIZE] = False
            mask[stop_token] = False
            logits = logits.masked_fill(mask, -float("inf"))
            guided = self._guided_c0(logits, cfg_scale, top_k).masked_fill(mask, -float("inf"))
            code = sample_topk(guided, top_k, generator)
            offset = AUDIO_CODE_OFFSET
        return torch.where(code == stop_token, torch.zeros_like(code), code - offset), code, int(stop_token)

    # ---- 主循环 --------------------------------------------------------

    def generate(self, input_ids: Any, *, seed: int, max_audio_frames: int,
                 cfg_scale: float = CFG_SCALE, top_k: int = CFG_TOP_K,
                 progress: Any = None) -> Any:
        """提示词 ⇒ **条件序列** ``(F, 32768)`` ✓（F ＝ 生成的音频帧数 ✓，25 帧/秒 ✓）。

        ⚠️ 三个「不报错只是歌不对」的坑都在这个函数里 ✓✗：
        ① 无条件支路是 ``tokens[1:-2]`` **整段换成** ``<|audio_cfg|>`` ✓（不是空提示词 ✗）；
        ② 反馈嵌入要乘 ``8**-0.5`` ✓；
        ③ 条件帧**从第 1 帧起**才采集 ✓（第 0 帧的隐状态是「还没听过任何音频」的状态 ✓，
           参考实现有意不采 ✓ —— 采集它会让 DiT 看到一帧与音频不同步的条件 ✓✗）。
        """
        torch = _torch()
        backbone = self.backbone
        ids = _as_ids(torch, input_ids, self.device())
        prompt_tokens = int(ids.shape[1])
        if prompt_tokens > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"Music3 提示词有 {prompt_tokens} 个 token ✗ ⇒ 上限 {MAX_PROMPT_TOKENS} ✓"
                "（超了要截提示词 ✓，不能截音频 ✗）")
        if prompt_tokens < 3:
            raise ValueError(
                f"提示词只有 {prompt_tokens} 个 token ✗ ⇒ 无条件支路要替换 ``tokens[1:-2]`` ✓，"
                "至少 3 个（正常提示串带 ``<|im_start|>…<|audio_start|>`` 结构 ✓）")
        decode_limit = min(int(max_audio_frames), MAX_AUDIO_FRAMES)
        if decode_limit < 1:
            raise ValueError(f"要生成的音频帧数 {max_audio_frames} ✗ ⇒ 至少 1 帧 ✓")

        dtype = self.execution_dtype()
        device = self.device()
        unconditioned = ids.clone()
        unconditioned[:, 1:-2] = SPECIAL_TOKEN_IDS["<|audio_cfg|>"]
        text_ids = torch.cat((ids, unconditioned), dim=0)
        text_embeds = backbone.embed_tokens_prefill(text_ids).to(dtype)

        cache = self.new_cache(prompt_tokens + decode_limit + 1)
        generator = torch.Generator(device=device).manual_seed(derive_seed(seed, "ar"))
        hidden_frames: list[Any] = []
        pending_hidden = None
        pending_valid = False
        pending_code = None
        stop_token = STOP_TOKEN
        frame_index = -1

        with torch.no_grad():
            hidden, cache = backbone(text_embeds, cache)
            last_hidden = hidden[:, -1].clone()
            del hidden
            for frame_index in range(decode_limit + 1):
                if pending_code is not None:
                    if int(pending_code.item()) == stop_token:
                        pending_valid = False
                        break
                    if pending_valid:
                        hidden_frames.append(pending_hidden)
                        if progress is not None:
                            progress(len(hidden_frames), decode_limit)
                        if len(hidden_frames) >= decode_limit:
                            break

                c0, code_or_stop, stop_token = self._sample_c0(
                    last_hidden, cfg_scale, top_k, generator)
                pending_code = code_or_stop
                c0 = c0.repeat(2)
                c0_embed = self._embed_c0(c0, dtype)
                codes, depth_hidden = self._depth_codes(
                    last_hidden, c0, c0_embed, generator, dtype, cfg_scale, top_k)
                frame_hidden = torch.cat((last_hidden[:1].detach(), depth_hidden), dim=-1)
                if frame_index > 0:
                    pending_hidden = frame_hidden[0].clone()
                    pending_valid = True
                else:
                    pending_valid = False
                feedback = self._embed_audio_frame(codes, dtype)
                hidden, cache = backbone(feedback, cache)
                last_hidden = hidden[:, -1].clone()
                del hidden, feedback, frame_hidden, depth_hidden, codes, c0_embed, c0

            if pending_valid and len(hidden_frames) < decode_limit:
                if pending_code is not None and int(pending_code.item()) != stop_token:
                    hidden_frames.append(pending_hidden)

        if not hidden_frames:
            raise ValueError(
                "Music3 一帧音频都没生成 ✗（第 1 帧就撞到停止码 ✓）—— 提示词结构或词表对不上 ✓")
        self.last_stats = {
            "frames": len(hidden_frames),
            "promptTokens": prompt_tokens,
            "decodeLimit": decode_limit,
            "stopped": pending_code is not None and int(pending_code.item()) == stop_token,
            "iterations": frame_index + 1,
            "cacheCapacity": cache.capacity,
            "contextDim": CONTEXT_DIM,
        }
        return torch.stack(hidden_frames).to(device="cpu")


def generate_hidden_frames(tower: Music3TextTower, input_ids: Any, **kwargs: Any) -> Any:
    """:meth:`Music3TextTower.generate` 的函数式别名 ✓（回执/脚本里少写一层 ✓）。"""
    return tower.generate(input_ids, **kwargs)


def load_tower(path: str | Any, *, dtype: Any = None, device: Any = "cpu",
               place: Any = None, require_int8: bool = True
               ) -> tuple[Music3TextTower, dict[str, Any]]:
    """把盘上那份 ``pruned_int8_convrot`` 装成能**生成**的塔 ✓。

    ## 五步顺序一个都不能换 ✗

    ① 只读头部核结构 ✓ → ② **meta** 建壳 ✓ → ③ 换 ``Int8Linear`` 壳 ✓ → ④ 物化 ✓ →
    ⑤ （可选）排布 ✓ → ⑥ 填权重 ✓。

    ③ 必须在 ④ 之前 ✓✗：那 144 个 ``nn.Linear`` 一旦物化成 bf16 实存（13.8 GiB ✓），
    换掉它们**不会**把内存还给系统 ✓（与 H3 那条 62 GiB 的坑同形 ✓）。
    ⑤ 必须在 ⑥ 之前 ✓（int8 那 9.2 GB 要么按层切、要么钉锁页内存 ✓，否则装载期就 OOM ✓）。

    ``place`` 是给装配侧（:mod:`torch_backend`）留的挂点 ✓：签名 ``place(module) -> dict`` ✓，
    在**物化之后、填权重之前**被调用 ✓（与 ``_place_before_load`` 同一条顺序 ✓）。
    """
    from . import h3_form, loader as loader_mod, quant as quant_mod  # noqa: PLC0415
    from . import safetensors as st, weights as weights_mod  # noqa: PLC0415
    from .torch_backend import _materialize_meta, int8_scale_plan  # noqa: PLC0415 —— 同包私有，仅为复用

    torch = _torch()
    target = path
    info = st.inspect(target)
    shapes = {name: tuple(record.shape) for name, record in info.tensors.items()}
    problems = tower_problems(shapes)
    if problems:
        raise ValueError(
            "这份权重不是 Music3 的语义塔 ✗：" + "；".join(problems[:3])
            + f"（共 {len(problems)} 条 ✓，{len(shapes)} 个键 ✓）")
    want = dtype or torch.bfloat16
    module = build_tower(dtype=want, device="meta")
    convrot, convrot_gap = quant_mod.convrot_group_of(loader_mod.quant_specs_of(target, info))
    if convrot_gap:
        raise ValueError(f"convrot 说明自相矛盾 ✗ ⇒ 不装载（不按猜的 group 算 ✗）：{convrot_gap}")
    plan = int8_scale_plan(info)
    if not plan and require_int8:
        raise ValueError(
            f"这份塔里一个 int8 权重都没有 ✗（{len(info.tensors)} 键 ✓）—— "
            "按 bf16 整份装要 13.8 GiB 且与 int8 版逐位不同 ✓；真要用 bf16 版请显式 "
            "``require_int8=False`` ✓")
    replaced = h3_form.upgrade_linear_to_int8(module, plan, convrot=convrot) if plan else 0
    #: 文件里是 **F32** 的键要钉回 fp32 ✓（这一步在 meta 上免费 ✓，顺序必须在物化之前 ✓）
    holders = (dict(module.named_parameters()), dict(module.named_buffers()))
    for logical, tensor in info.tensors.items():
        if tensor.dtype != "F32":
            continue
        for holder in holders:
            slot = holder.get(logical)
            if slot is not None and slot.is_floating_point():
                slot.data = slot.data.to(torch.float32)
    filled = _materialize_meta(module, torch_mod=torch, device=device)
    placement = place(module) if place is not None else None
    report = weights_mod.load_module_weights(module, path, dtype=want)
    if not report.complete:
        raise ValueError(
            f"Music3 语义塔没装满 ✗ ⇒ 缺 {len(report.missing)} 个 ✗：{list(report.missing)[:4]}"
            f"（多出 {len(report.unexpected)} 个 ✓）")
    module.eval().requires_grad_(False)
    tower = Music3TextTower(module, dtype=want)
    return tower, {
        "keys": len(shapes),
        "int8Layers": int(replaced if isinstance(replaced, int) else len(replaced)),
        "convrot": convrot,
        "materialized": int(filled),
        "placement": placement,
        "unexpected": len(report.unexpected),
        "outputDim": CONTEXT_DIM,
        "dtype": str(want),
    }
