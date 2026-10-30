"""**GGUF 自带词表** → 分词器（2026-10-29 起 ✓）—— Ollama 库里的模型**只有 GGUF** ✗，
词表在元数据里 ✓，不在 ``tokenizer.json`` ✗。

## 为什么非做不可

:mod:`.tokenizer_hub` / :mod:`.tokenizer_own` 只认 **HF 词表文件**（``tokenizer.json`` ✓、
``vocab.json``+``merges.txt`` ✓）⇒ 真机 qwen3:8b（Ollama blob ✓）**一条都装不上** ✗✗，
于是 :class:`~.llm_backend.LlmBackend` 卡在「词表未就绪」✗：权重装好了、能算 logits ✓，
却**一个 token 都编不出、解不回** ✗（分不清是模型错还是词表缺 ✓✗）。

## 事实来源（GGUF 的 tokenizer 元数据键，llama.cpp ``llm_load_vocab`` ✓）

* ``tokenizer.ggml.model``：``gpt2``（BPE ✓）/ ``llama``（SPM ✗ 本模块**不做** ✗）；
* ``tokenizer.ggml.tokens``：``list[str]`` ✓ —— **id 就是下标** ✓（这是词表的全部真相 ✓）；
* ``tokenizer.ggml.merges``：``list[str]`` ✓，每条形如 ``"a b"`` ✓（= HF 的新格式 ✓）；
* ``tokenizer.ggml.token_type``：``list[int]`` ✓ —— 1 NORMAL / 2 UNKNOWN / 3 **CONTROL** /
  4 **USER_DEFINED** / 5 UNUSED / 6 BYTE ✓。⚠️ **控制符（如 ``<|im_start|>``）靠它识别** ✗：
  不把它当「整体 token」 ⇒ 会被字节级 BPE 拆成一串 id ✗✗（聊天模板立刻编错 ✓）；
* ``tokenizer.ggml.pre``：预分词规则名 ✓（``qwen2`` ✓ / ``default`` ✓ / ``llama-bpe`` ✗ …）；
* ``tokenizer.ggml.{bos,eos}_token_id`` ＋ ``add_bos_token``/``add_eos_token`` ✓；
* ``tokenizer.chat_template``：由 :mod:`.chat_template` 单独读 ✓（不在这里重复 ✗）。

⚠️ **``inspect`` 拿不到词表正文** ✗（它对数组只存前 16 项 ✓✗）⇒ 本模块走
:func:`~.gguf.read_arrays`（全量读 ✓，见那里的说明 ✓）。

## 预分词：GPT-2 ≠ Qwen2（**硬套会静默编出错 id** ✗✗）

:func:`.tokenizer_bpe.pretokenize` 实现的是 GPT-2 正则
``'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`` ✓——
Qwen2 的那条**有两处实质不同** ✗：

1. ``\p{N}`` 是**单个数字** ✓（GPT-2 是 ``\p{N}+`` 连续数字 ✓）⇒ ``"1234"`` 在 Qwen2 里
   是 **4 个 token 起点** ✗、在 GPT-2 里是 **1 个** ✗；
2. 多一条 ``\s*[\r\n]+`` 分支 ✓（换行前的空格**并入**换行段 ✓），且字母前的可选字符是
   ``[^\r\n\p{L}\p{N}]?`` ✓（GPT-2 是 `` ?`` ✓ —— 只有空格 ✓）。

⇒ 本模块按**规则名**分发预分词器 ✓（:data:`PRETOKENIZERS` ✓），不认识的规则名**报错** ✓
（不猜、不回落 GPT-2 ✗✗）。
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Callable

from . import gguf as gguf_mod
from . import tokenizer_bpe
from . import tokenizer_hub

__all__ = ["GgufTokenizerError", "PRETOKENIZERS", "load_gguf_tokenizer", "pretokenize_qwen2"]

#: 要读的元数据键 ✓（⚠️ 全是**数组**或标量 ✓：词表十万级 ⇒ 必须走全量读 ✓）
_KEYS: tuple[str, ...] = (
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
)

#: ``token_type`` 里算「整体 token」的类型 ✓（3 CONTROL / 4 USER_DEFINED ✓）—— 见模块注释 ✓
_SPECIAL_TYPES = (3, 4)


class GgufTokenizerError(ValueError):
    """这份 GGUF 的词表**装不成** ✓ —— 理由具名 ✓（不回落、不猜 ✗）。"""


# ── Qwen2 预分词 ✓（自写扫描器 ✓ —— 与 :func:`.tokenizer_bpe.pretokenize` 同款理由：
#    标准库 ``re`` 不支持 ``\p{L}``/``\p{N}`` ✗，而本仓不引 ``regex`` 依赖 ✗）

def _cat(char: str) -> str:
    """单字符归类 ✓（顺序要紧 ✗：``\\r\\n`` 必须先判 ✓，否则会被当成普通空白 ✓✗）。"""
    if char in "\r\n":
        return "newline"
    kind = unicodedata.category(char)[0]
    if kind == "L":
        return "letter"
    if kind == "N":
        return "number"
    if char.isspace():
        return "space"
    return "other"


_CONTRACTIONS: tuple[str, ...] = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")


def _contraction_at(text: str, index: int) -> str:
    """``(?i:'s|'t|'re|'ve|'m|'ll|'d)`` ✓ —— **大小写不敏感** ✓（GPT-2 那条是敏感的 ✗，别照抄 ✗）。"""
    window = text[index:index + 3].lower()
    for suffix in _CONTRACTIONS:                      # 先 3 字符再 2 字符 ✓（长匹配优先 ✓）
        if window.startswith(suffix):
            return text[index:index + len(suffix)]
    return ""


def pretokenize_qwen2(text: str) -> list[str]:
    """Qwen2 预分词 ✓（规则字面形，见模块注释 ✓；分支顺序与正则**同序** ✓）。

    ``(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+``
    """
    pieces: list[str] = []
    size = len(text)
    index = 0
    while index < size:
        hit = _contraction_at(text, index)                      # ① 收缩词 ✓
        if hit:
            pieces.append(hit)
            index += len(hit)
            continue

        kind = _cat(text[index])
        if kind == "letter":                                    # ② 纯字母串 ✓
            end = index
            while end < size and _cat(text[end]) == "letter":
                end += 1
            pieces.append(text[index:end])
            index = end
            continue
        if kind in ("space", "other") and index + 1 < size and _cat(text[index + 1]) == "letter":
            # ② 的 ``[^\r\n\p{L}\p{N}]?`` 前缀 ✓ —— **可以**是空格 ✓（GPT-2 的 `` ?`` 只能是空格 ✗）
            end = index + 1
            while end < size and _cat(text[end]) == "letter":
                end += 1
            pieces.append(text[index:end])
            index = end
            continue

        if kind == "number":                                    # ③ ``\p{N}`` **单个数字** ✓
            pieces.append(text[index])
            index += 1
            continue

        start = index                                            # ④ `` ?`` + 非空白/非字母/非数字 + 尾随换行 ✓
        probe = index
        if text[probe] == " " and probe + 1 < size and _cat(text[probe + 1]) == "other":
            probe += 1
        if _cat(text[probe]) == "other":
            end = probe
            while end < size and _cat(text[end]) == "other":
                end += 1
            while end < size and _cat(text[end]) == "newline":
                end += 1
            pieces.append(text[start:end])
            index = end
            continue

        probe = index                                            # ⑤ ``\s*[\r\n]+`` ✓
        while probe < size and _cat(text[probe]) == "space":
            probe += 1
        if probe < size and _cat(text[probe]) == "newline":
            end = probe
            while end < size and _cat(text[end]) == "newline":
                end += 1
            pieces.append(text[index:end])
            index = end
            continue

        end = index                                              # ⑥⑦ ``\s+(?!\S)`` / ``\s+`` ✓
        while end < size and _cat(text[end]) in ("space", "newline"):
            end += 1
        if end == index:                                        # 防御：不该发生 ⇒ 别死循环 ✗
            pieces.append(text[index])
            index += 1
            continue
        if end < size and end - index > 1:
            # 多字符空白且后面还有东西 ⇒ **末尾那个留给下一段** ✓（= ``(?!\S)`` 的回溯效果 ✓）
            pieces.append(text[index:end - 1])
            index = end - 1
        else:
            pieces.append(text[index:end])
            index = end
    return [piece for piece in pieces if piece]


#: 预分词规则名 → ``(实现 ✓, 说明 ✓)``（llama.cpp 的 ``tokenizer.ggml.pre`` 取值 ✓）
PRETOKENIZERS: dict[str, tuple[Callable[[str], list[str]], str]] = {
    "qwen2": (pretokenize_qwen2, "Qwen2 规则（``\\p{N}`` 单数字 ＋ ``\\s*[\\r\\n]+``）"),
    "default": (tokenizer_bpe.pretokenize, "llama.cpp 的 default（= GPT-2 ByteLevel 那套）"),
    "gpt-2": (tokenizer_bpe.pretokenize, "GPT-2 ByteLevel"),
    "": (tokenizer_bpe.pretokenize, "没写 ``pre`` ⇒ llama.cpp 按 default 走（GPT-2 那套）"),
}


def load_gguf_tokenizer(path: str | Path) -> tokenizer_hub.TokenizerHub:
    """GGUF 文件 → 可直接 ``encode``/``decode`` 的分词器 ✓（包在 :class:`TokenizerHub` 里 ✓
    以复用它的 LRU 与自述 ✓）。

    ⚠️ 三条**具名拒绝**（都不猜 ✗）：词表模型不是 ``gpt2`` 血统 ✗、没有 ``tokens`` ✗、
    没有 ``token_type``（⇒ 认不出控制符 ✗✗）✗、``pre`` 是没实现的规则名 ✗。
    """
    target = Path(path)
    if not target.exists():
        raise GgufTokenizerError(f"GGUF 文件不存在：{target} ✗")
    meta = gguf_mod.read_arrays(target, _KEYS)

    model_name = str(meta.get("tokenizer.ggml.model") or "")
    if model_name and model_name != "gpt2":
        raise GgufTokenizerError(
            f"词表模型是 `{model_name}` ✗ —— 本模块只实现 **gpt2 血统的 BPE** ✓"
            f"（``llama``/SPM 的切分规则完全不同 ✓ ⇒ 硬套会**静默编出错 id** ✗✗）")

    tokens = meta.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or not tokens:
        raise GgufTokenizerError(
            "GGUF 里没有 ``tokenizer.ggml.tokens`` ✗（有些导出把它剥掉了 ✗）"
            "⇒ 没有词表就编不出 id ✓（不猜、不用哈希假桩 ✗）")
    if not all(isinstance(token, str) for token in tokens):
        raise GgufTokenizerError("``tokenizer.ggml.tokens`` 里有非字符串元素 ✗")

    types = meta.get("tokenizer.ggml.token_type")
    if not isinstance(types, list) or len(types) != len(tokens):
        raise GgufTokenizerError(
            f"``tokenizer.ggml.token_type`` 缺失或长度不符（{len(types) if isinstance(types, list) else '没有'}"
            f" vs 词表 {len(tokens)}）✗ ⇒ **认不出哪些是控制符** ✗（``<|im_start|>`` 会被字节级拆散 ✗✗）")

    pre = str(meta.get("tokenizer.ggml.pre") or "")
    if pre not in PRETOKENIZERS:
        raise GgufTokenizerError(
            f"``tokenizer.ggml.pre = {pre!r}`` ✗ —— 这个预分词规则本模块没实现 ✓"
            f"（已有的：{sorted(name for name in PRETOKENIZERS if name)} ✓）"
            f"⇒ 不回落 GPT-2 ✗（两套规则差一个字符就会编出错 id ✗✗）")
    splitter, pre_note = PRETOKENIZERS[pre]

    merges = meta.get("tokenizer.ggml.merges")
    if not isinstance(merges, list):
        merges = []
    vocab = {token: index for index, token in enumerate(tokens)}
    # 控制符 / 用户自定义符 ⇒ **整体 token** ✓（聊天模板里 ``<|im_start|>`` 必须一个 id ✓）
    added = {token: index for index, token in enumerate(tokens)
             if types[index] in _SPECIAL_TYPES}
    bos_id = meta.get("tokenizer.ggml.bos_token_id")
    eos_id = meta.get("tokenizer.ggml.eos_token_id")
    impl = tokenizer_bpe.BpeTokenizer(
        vocab, merges, added_tokens=added, splitter=splitter, pre_tokenizer_name=pre or "default",
        notes=[f"词表来自 GGUF 元数据 ✓（{len(tokens)} 项 ＋ {len(merges)} 条 merge ＋ "
               f"{len(added)} 个控制符 ✓）",
               f"预分词 = `{pre or 'default'}` ✓：{pre_note}"])

    hub = tokenizer_hub.TokenizerHub(
        "own-bpe", impl, config=tokenizer_hub.HubConfig(path=str(target)),
        vocab_size=impl.vocab_size, required_vocab_size=impl.required_vocab_size,
        detail={"form": {"modelType": "BPE", "preTokenizer": pre or "default",
                         "source": "gguf-metadata"},
                "ggufPath": str(target), "merges": impl.merge_count,
                "specialTokens": len(added),
                "bosTokenId": bos_id, "eosTokenId": eos_id,
                "addBosToken": meta.get("tokenizer.ggml.add_bos_token"),
                "addEosToken": meta.get("tokenizer.ggml.add_eos_token")},
        notes=[f"{len(tokens)} 项词表读自 GGUF 本体 ✓ ⇒ **不需要 `tokenizer.json`** ✗",
               # ⚠️ 生成时**不自动**加 BOS/EOS ✗：特殊符由 chat 模板写在文本里 ✓（见 chat_template ✓）
               "``encode(add_special_tokens=True)`` **不加** BOS/EOS ✗ —— 它们由对话骨架负责 ✓"
               "（生成侧把 EOS 拼在提示词尾部会直接算错 ✓✗）"])
    hub._source_path = str(target)                       # noqa: SLF001
    hub._install_cache()                                 # noqa: SLF001
    return hub
