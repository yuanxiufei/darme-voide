r"""**自研分词算法扩充**：`Unigram`（Viterbi ✓）/ `WordPiece`（贪心最长匹配 ✓）/ `Metaspace` ✓。

## 这一步在「升级」什么

上一轮 `tokenizer_hub` 的形态覆盖是：**自研 BPE**（ByteLevel-BPE ✓）+ **回退参考实现**
（其余形态 ✗）。⇒ 算力上"能用"✓，但**能力仍被外部限制** ✗：`Unigram`（Llama/Qwen/T5 系 ✓）
与 `WordPiece`（BERT 系 ✓）这两大血统**一旦没装 `transformers` 就编不出 token** ✗。
本模块把它们**自己实现** ✓ ⇒ 参考实现降级为**核对用的第二实现** ✓（`tokenizer_hub` 里
「回退」只剩极少数形态 ✓）。

## 事实来源（都是公开算法 / 公开格式 ✓；代码自己写 ✓）

* **Unigram**：词表给每个 token 一个 log 概率 ✓，切分取**总得分最大**的那条路 ✓ ——
  用 **Viterbi 动态规划**在**前缀树**上跑 ✓；未知字符按 `unk` 处理 ✓，
  `fuse_unk`（默认真 ✓）把**连续未知**并成一个 `unk` ✓，其得分用
  ``min_score − 10.0`` ✓（SentencePiece 血统的常见口径 ✓）；
* **WordPiece**：**贪心最长匹配** ✓ + 续接前缀 ``##`` ✓；⚠️ **整词失败 ⇒ 整个词给 `[UNK]`** ✓
  （不是"能切多少切多少" ✗ —— 这条最容易写错 ✓）；预分词是 ``BertPreTokenizer`` ✓
  （**空白 + 标点**各切一刀 ✓，标点判据与 HF 一致：ASCII 四段 + ``unicodedata`` 的 ``P*`` ✓）；
* **Metaspace**：把空格换成 ``▁`` 并在**首部补一个** ✓（SentencePiece 的 ``prepend_scheme`` ✓）。

## normalizer（2026-09-21 补 ✓）

此前 ``normalizer`` 非空就**一律拒绝** ✗ ⇒ BERT 系（带 `BertNormalizer` ✓）这类真实词表只能回退 ✓。
现在实现了 ✓：``BertNormalizer``（`clean_text` ✓ / `handle_chinese_chars` ✓ / `lowercase` ✓ /
`strip_accents` ✓ —— **四步顺序照参考** ✓）、``NFC``·``NFD``·``NFKC``·``NFKD`` ✓、``Lowercase`` ✓、
``StripAccents`` ✓、``Strip`` ✓、``Prepend`` ✓、``Replace`` ✓、``Sequence`` ✓
（统一用 :class:`NormalizedTokenizer` 套住 ✓ 三种模型共用一个包装 ✓）。

⚠️ 两处**实测出来的语义差**（都靠逐例比对抓到的 ✓，不靠文档 ✗）：

* ``BertNormalizer`` 的 ``clean_text``：``\\t``/``\\n`` ⇒ **一个空格** ✓、
  **不 trim ✓ 不合并连续空格** ✗（``'  spaced\\t tabs\\n'`` ⇒ ``'  spaced  tabs '`` ✓）；
* ``StripAccents`` **不先做 NFD** ✓✗：预组合的 ``é`` **原样保留** ✓，只丢**独立**组合字符 ✓
  —— 而 ``BertNormalizer(strip_accents=True)`` **会**先 NFD ✓（两处语义**不一样** ✓）。

## 明确的边界（不认识就**拒绝**，不硬套 ✗）

* ``normalizer`` 里含**未实现**项 ⇒ **不走自研** ✓（``Precompiled`` ✗ 要 SentencePiece 的
  **charsmap 表** ✓ 不是 NFKC 换个写法 ✗；``Nmt`` ✗）；⚠️ 判定要按 ``Sequence`` **展平后的每一项** ✓
  （只看顶层类型会把 ``Sequence`` 读成"能接" ✓✗）；
* ⭐ ``byte_fallback: true`` 的 **Unigram 已实现** ✓（2026-09-21 第二轮 ✓）：判据是 **段级**的 ✓
  （见 :meth:`UnigramTokenizer.emit_unknown` ✓）；⚠️ **别的模型**的 ``byte_fallback`` **不走自研** ✓
  （BPE 的触发条件与它不同 ✓ 未核清 ✓ 不按猜的实现 ✓）。
  ⚠️ 教训 ✗：第一轮曾判成「参考实现没走字节回退 ⇒ 拒绝」✓✗ —— 那次实测**少了前置条件** ✗
  （词表里**没有** ``<0xNN>`` 字节 token ✓）。**「实测过」必须写明前置条件** ✗；
* 预分词器不认识（``UnicodeScripts`` —— 要 Unicode script 表 ✓ 标准库没有 ✗；
  ``Split`` 的空匹配正则 / ``\p{…}`` ✓）⇒ **不走自研** ✓。

⇒ 走不了自研时由 :mod:`.tokenizer_hub` 决定**回退参考实现**还是报错 ✓（本模块只回答「我行不行」✓）。
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import tokenizer_bpe

__all__ = ["BEHAVIORS", "BertNormalizer", "BertPreTokenizer", "FixedLengthPretokenizer",
           "MetaspacePretokenizer", "NormalizedTokenizer", "OwningSupport", "SequenceNormalizer",
           "SimplePreTokenizer", "SplitPretokenizer", "UnicodeNormalizer", "UnigramTokenizer",
           "WordPieceTokenizer", "build_normalizer", "fixed_length_problem",
           "normalizer_spec_types", "own_support", "load_own_tokenizer", "split_spec_problem",
           "SUPPORTED_NORMALIZERS"]

#: ⚠️ 自研**实现了**的 normalizer ✓；不在表里的 ⇒ **拒绝** ✓（由 hub 决定回退还是报错 ✓）。
#: 明确不做 ✗：`Precompiled`（要 SentencePiece 的 **charsmap 表** ✓ —— 实测
#: ``Precompiled.__new__() missing 1 required positional argument`` ✓✗，
#: 那张表不是"NFKC 换个写法" ✓ 不能假装 ✓）、`Nmt`、以及一切认不出的 ✓。
SUPPORTED_NORMALIZERS: tuple[str, ...] = (
    "BertNormalizer", "NFC", "NFD", "NFKC", "NFKD",
    "Lowercase", "StripAccents", "Strip", "Prepend", "Replace", "Sequence",
)

#: **片段级缓存上限** ✓（超了**整体清空** ✓ 不做 LRU ✗ —— 换一篇就换工作集 ✓ 简单规则够用且不抖动 ✓）。
#: ⚠️ 放在**模块级** ✗：两个模型（Unigram / WordPiece）都用它 ✓ —— 写在某个类里会让另一个类
#: `AttributeError` ✓✗（2026-09-21 实测踩到 ✓）。
PIECE_CACHE_LIMIT = 32768


class OwningSupport:
    """「自研能不能覆盖这份词表」的结论 ✓（``ok`` + ``reason`` ✓ —— 走不了也要**说清为什么** ✓）。"""

    def __init__(self, ok: bool, reason: str, form: dict[str, Any] | None = None) -> None:
        self.ok = bool(ok)
        self.reason = str(reason)
        self.form = dict(form or {})

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, "form": dict(self.form)}


def _is_punctuation(char: str) -> bool:
    """标点判据 ✓（与 HF 的 `BertPreTokenizer` 同口径 ✓：ASCII 四段 + Unicode ``P*`` ✓）。"""
    code = ord(char)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126:
        return True
    return unicodedata.category(char).startswith("P")


# ══════════════════════════════════════════════════════════════════════════
# normalizer ✓（2026-09-21 补：此前一律**拒绝** ✗ ⇒ 很多真实词表（BERT 系 ✓）只能回退参考实现 ✗）
# ══════════════════════════════════════════════════════════════════════════
class _Normalizer:
    """normalizer 基类 ✓：``normalize(text) -> text`` ✓ + ``name`` ✓。"""

    name = "normalizer"

    def normalize(self, text: str) -> str:            # pragma: no cover - 抽象 ✓
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}

    def __call__(self, text: str) -> str:
        return self.normalize(text)


def _is_control(char: str) -> bool:
    """控制字符 ✓（与 HF 同口径：``\\t`` / ``\\n`` / ``\\r`` **不算** ✗ —— 它们会被转成空格 ✓）。"""
    if char in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(char).startswith("C")


def _is_whitespace(char: str) -> bool:
    """空白 ✓（HF 口径：``' '`` / ``\\t`` / ``\\n`` / ``\\r`` 或 Unicode ``Zs`` ✓）。"""
    return char in (" ", "\t", "\n", "\r") or unicodedata.category(char) == "Zs"


def _is_chinese_char(code: int) -> bool:
    """CJK 判据 ✓（HF 的四段区间 + 扩展 A 等 ✓ —— 与 `BertNormalizer` 同口径 ✓）。"""
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF or 0x2F800 <= code <= 0x2FA1F)


class BertNormalizer(_Normalizer):
    """``BertNormalizer`` ✓ —— 四步，**顺序照参考实现** ✓（实测逐例对齐 ✓ 2026-09-21）：

    1. ``clean_text`` ✓：**去掉控制字符** ✓（``\\t``/``\\n``/``\\r`` 除外 ✓）、
       ``\\t``/``\\n``/``\\r``/``Zs`` ⇒ **一个空格** ✓（⚠️ **不 trim ✓ 不合并连续空格** ✗
       —— 实测 ``'  spaced\\t tabs\\n'`` ⇒ ``'  spaced  tabs '`` ✓）；
    2. ``handle_chinese_chars`` ✓：每个 CJK 字符**两侧各加一个空格** ✓
       （实测 ``'中文abc'`` ⇒ ``' 中  文 abc'`` ✓）；
    3. ``lowercase`` ✓；
    4. ``strip_accents`` ✓：NFD 后**丢掉 Mn** ✓（⚠️ 该字段为 ``null`` 时**跟随 lowercase** ✓
       —— 参考实现的默认耦合 ✓）。

    ⚠️ 顺序不能换 ✗：先小写再去重音 vs 反着来，在组合字符上会出**不同的串** ✓✗（不报错的那种）。
    """

    name = "BertNormalizer"

    def __init__(self, clean_text: bool = True, handle_chinese_chars: bool = True,
                 strip_accents: bool | None = None, lowercase: bool = False) -> None:
        self.clean_text = bool(clean_text)
        self.handle_chinese_chars = bool(handle_chinese_chars)
        self.lowercase = bool(lowercase)
        self.strip_accents = self.lowercase if strip_accents is None else bool(strip_accents)

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "BertNormalizer":
        return cls(clean_text=bool(spec.get("clean_text", True)),
                   handle_chinese_chars=bool(spec.get("handle_chinese_chars", True)),
                   strip_accents=spec.get("strip_accents"),
                   lowercase=bool(spec.get("lowercase", False)))

    def normalize(self, text: str) -> str:
        out = text
        if self.clean_text:
            buffer: list[str] = []
            for char in out:
                if ord(char) == 0 or ord(char) == 0xFFFD or _is_control(char):
                    continue
                buffer.append(" " if _is_whitespace(char) else char)
            out = "".join(buffer)
        if self.handle_chinese_chars:
            spaced: list[str] = []
            for char in out:
                if _is_chinese_char(ord(char)):
                    spaced.append(" " + char + " ")
                else:
                    spaced.append(char)
            out = "".join(spaced)
        if self.lowercase:
            out = out.lower()
        if self.strip_accents:
            decomposed = unicodedata.normalize("NFD", out)
            out = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
        return out

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "cleanText": self.clean_text,
                "handleChineseChars": self.handle_chinese_chars,
                "lowercase": self.lowercase, "stripAccents": self.strip_accents}


class UnicodeNormalizer(_Normalizer):
    """Unicode 规范化 ✓（``NFC`` / ``NFD`` / ``NFKC`` / ``NFKD`` ✓ —— 直接落标准库 ✓）。"""

    def __init__(self, form: str) -> None:
        if form not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError(f"Unicode 规范化只认 NFC/NFD/NFKC/NFKD（收到 {form!r} ✗）")
        self.name = form
        self.form = form

    def normalize(self, text: str) -> str:
        return unicodedata.normalize(self.form, text)


class SequenceNormalizer(_Normalizer):
    """``Sequence`` ✓：**按顺序**串起来 ✓（顺序有意义 ✓ —— 前后调换结果可能不同 ✓）。"""

    name = "Sequence"

    def __init__(self, parts: Sequence[_Normalizer]) -> None:
        if not parts:
            raise ValueError("`Sequence` normalizer 里没有子项 ✗")
        self.parts = list(parts)

    def normalize(self, text: str) -> str:
        out = text
        for part in self.parts:
            out = part.normalize(out)
        return out

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "parts": [part.describe() for part in self.parts]}


class SimpleNormalizer(_Normalizer):
    """几个一行就能说清的 ✓：``Lowercase`` / ``StripAccents`` / ``Strip`` / ``Prepend`` / ``Replace`` ✓。"""

    def __init__(self, kind: str, **options: Any) -> None:
        self.name = kind
        self.options = dict(options)
        if kind == "Lowercase":
            self._apply = lambda text: text.lower()
        elif kind == "StripAccents":
            # ⚠️⚠️ **不先做 NFD** ✗ —— 实测（2026-09-21）：参考实现的 `StripAccents` 对
            #    **预组合**字符（如 `é` = U+00E9 ✓）**原样保留** ✓✗（``'café naïve'`` ⇒ 不变 ✓），
            #    只丢**独立的组合字符**（``e`` + U+0301 ⇒ ``e`` ✓）。第一版加了 NFD ⇒ 与参考
            #    不一致 ✓✗（自检 ⑬ 当场红 ✓）。
            #    ⚠️ 与 `BertNormalizer(strip_accents=True)` **不同** ✗：那个**会**先 NFD ✓
            #    （同一次自检里两个用例分别对齐 ✓ ⇒ 两处语义确实不一样 ✓）。
            self._apply = lambda text: "".join(
                char for char in text if unicodedata.category(char) != "Mn")
        elif kind == "Strip":
            left = bool(options.get("strip_left", True))
            right = bool(options.get("strip_right", True))
            self._apply = lambda text: (text.lstrip() if left else text).rstrip() if right else \
                (text.lstrip() if left else text)
        elif kind == "Prepend":
            prefix = str(options.get("prepend") or "")
            self._apply = lambda text: prefix + text
        elif kind == "Replace":
            pattern = options.get("pattern") or ""
            content = str(options.get("content") or "")
            if isinstance(pattern, dict):
                if "Regex" in pattern:
                    compiled = re.compile(str(pattern["Regex"]))
                else:
                    compiled = re.compile(re.escape(str(pattern.get("String", ""))))
            else:                                     # pragma: no cover - 旧格式兜底 ✓
                compiled = re.compile(re.escape(str(pattern)))
            self._apply = lambda text: compiled.sub(content, text)
        else:                                          # pragma: no cover - build_normalizer 已挡 ✓
            raise ValueError(f"不认识的简单 normalizer：{kind} ✗")

    def normalize(self, text: str) -> str:
        return str(self._apply(text))

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, **self.options}


def _unsupported_hint(kind: str) -> str:
    """给「未实现的 normalizer」补一句**为什么** ✓（判定与装载两处**同一句话** ✗ 别各写一份 ✓）。"""
    if kind == "Precompiled":
        return (" —— 它要 SentencePiece 的 **charsmap 表** ✓ 不是 NFKC 换个写法 ✗"
                "（实测 `Precompiled.__new__()` 就要求那张表 ✓）")
    if kind == "Nmt":
        return " —— Moses/NMT 的那套替换规则 ✗（表很大 ✓ 且本仓用不到 ✓）"
    return ""


def normalizer_spec_types(spec: Any) -> list[str]:
    """把 normalizer 规格**展平**成类型名列表 ✓（``Sequence`` 递归 ✓ —— 判定要按**每一项** ✓）。"""
    if not spec or not isinstance(spec, dict):
        return []
    kind = str(spec.get("type") or "")
    if kind == "Sequence":
        types: list[str] = []
        for part in spec.get("normalizers") or []:
            types.extend(normalizer_spec_types(part))
        return types
    return [kind] if kind else []


def build_normalizer(spec: Any) -> tuple[_Normalizer | None, str]:
    """造 normalizer ✓ `(impl, reason)` ✓ —— **没 spec ⇒ ``(None, "")``** ✓（不需要规范化 ✓）。

    ⚠️ 认不出/没实现 ⇒ 回 ``(None, 理由)`` ✓ 由 :func:`own_support` 变成"拒绝" ✓
    （**不抛** ✗、也不静默跳过 ✗ —— 静默跳过 = 用未规范化的文本去查词表 ⇒ 静默算错 id ✓✗）。
    """
    if not spec:
        return None, ""
    if not isinstance(spec, dict):
        return None, f"normalizer 规格不是对象 ✗（{type(spec).__name__} ✓）"
    kind = str(spec.get("type") or "")
    if kind not in SUPPORTED_NORMALIZERS:
        return None, f"normalizer `{kind or '空'}` 未实现 ✗{_unsupported_hint(kind)}"
    try:
        if kind == "BertNormalizer":
            return BertNormalizer.from_spec(spec), ""
        if kind in ("NFC", "NFD", "NFKC", "NFKD"):
            return UnicodeNormalizer(kind), ""
        if kind == "Sequence":
            parts: list[_Normalizer] = []
            for index, part in enumerate(spec.get("normalizers") or []):
                impl, reason = build_normalizer(part)
                if impl is None:
                    return None, f"`Sequence` 第 {index} 项不可用 ✗：{reason}"
                parts.append(impl)
            return SequenceNormalizer(parts), ""
        return SimpleNormalizer(kind, **{key: value for key, value in spec.items()
                                         if key != "type"}), ""
    except Exception as err:  # noqa: BLE001 —— 构造失败也走"拒绝"✓ 与"没实现"同一出口 ✓
        return None, f"normalizer `{kind}` 构造失败 ✗：{type(err).__name__}: {err} ✓"


class NormalizedTokenizer:
    """**给任意自研实现套一层 normalizer** ✓（三种模型共用一个 ✓ ⇒ 不必各自改 ✗）。

    ``encode`` 先规范化再切分 ✓；``decode`` **不加**规范化 ✗（还原本来就不该改文本 ✓）。
    """

    def __init__(self, impl: Any, normalizer: _Normalizer) -> None:
        self._impl = impl
        self.normalizer = normalizer

    @property
    def name(self) -> str:
        return str(getattr(self._impl, "name", "own"))

    @property
    def vocab_size(self) -> int:
        return int(self._impl.vocab_size)

    @property
    def required_vocab_size(self) -> int:
        return int(self._impl.required_vocab_size)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self._impl.encode(self.normalizer.normalize(str(text or "")),
                                 add_special_tokens=add_special_tokens)

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self._impl.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def describe(self) -> dict[str, Any]:
        return {**self._impl.describe(), "normalizer": self.normalizer.describe()}

    def fingerprint(self) -> dict[str, Any]:
        base = dict(self._impl.fingerprint()) if hasattr(self._impl, "fingerprint") else {}
        return {**base, "normalizer": self.normalizer.name}


class BertPreTokenizer:
    """``BertPreTokenizer`` ✓：**空白**切一刀 ✓、**标点**各切一刀 ✓（标点独立成片 ✓）。"""

    @staticmethod
    def split(text: str) -> list[str]:
        pieces: list[str] = []
        buffer: list[str] = []
        for char in text:
            if char.isspace():
                if buffer:
                    pieces.append("".join(buffer))
                    buffer = []
                continue
            if _is_punctuation(char):
                if buffer:
                    pieces.append("".join(buffer))
                    buffer = []
                pieces.append(char)
                continue
            buffer.append(char)
        if buffer:
            pieces.append("".join(buffer))
        return pieces


class SimplePreTokenizer:
    """另外几种**规则型**预分词器 ✓（``Digits`` / ``Punctuation`` / ``Whitespace`` /
    ``WhitespaceSplit`` / ``CharDelimiterSplit`` ✓）—— ⚠️ 规则全部**逐例实测**得来 ✓（2026-09-21 ✓）：

    * ``WhitespaceSplit`` ✓：**只按空白**切 ✓（空白丢掉 ✓ 其余原样 ✓）；
    * ``Whitespace`` ✓：先按空白切（丢掉空白 ✓），再对每一片按**字母数字 ↔ 非字母数字的边界**切 ✓
      （⚠️ 连续非字母数字**留在一起** ✓ —— 实测 ``'a|b||c'`` ⇒ ``['a','|','b','||','c']`` ✓，
      不是"每个标点一刀" ✗）；
    * ``Punctuation`` ✓：**每个标点各自成片** ✓；``is_whitespace_prefix``（默认真 ✓）
      把前导空白**并给后一片** ✓（实测 ``'a, b!'`` ⇒ ``['a', ',', ' b', '!']`` ✓）；
    * ``Digits`` ✓：数字按 ``individual_digits`` 决定是否**逐位**切 ✓（实测
      ``'ab12cd'`` ⇒ ``['ab','1','2','cd']`` ✓）；
    * ``CharDelimiterSplit`` ✓：按给的分隔符切 ✓ 分隔符丢掉 ✓ **空片丢弃** ✓
      （实测 ``'a|b||c'`` ⇒ ``['a','b','c']`` ✓）。
    """

    def __init__(self, kind: str, **options: Any) -> None:
        self.name = kind
        self.options = dict(options)
        self.kind = kind

    def split(self, text: str) -> list[str]:
        if self.kind == "WhitespaceSplit":
            return text.split()
        if self.kind == "Whitespace":
            return [group for piece in text.split()
                    for group in _alnum_groups(piece)]
        if self.kind == "Punctuation":
            behavior = self.options.get("behavior")
            if behavior is None:
                if "is_whitespace_prefix" in self.options:
                    # ⚠️ 老字段（bool ✓）与新版 `behavior` **不是一一对应** ✗ ⇒ **拒绝** ✓
                    #    （映射关系没核过 ✓ 不猜 ✗ —— 猜错就是静默算错 id ✓✗）
                    raise ValueError("`Punctuation` 用了老的 `is_whitespace_prefix` ✗"
                                     " ⇒ 语义映射未核过 ✓ 请改用 `behavior` ✓")
                behavior = "isolated"
            return _punctuation_pieces(text, str(behavior))
        if self.kind == "Digits":
            return _digit_pieces(text, bool(self.options.get("individual_digits", True)))
        if self.kind == "CharDelimiterSplit":
            delimiter = str(self.options.get("delimiter") or "")
            if not delimiter:
                raise ValueError("`CharDelimiterSplit` 没给 `delimiter` ✗")
            return [piece for piece in text.split(delimiter) if piece]
        raise ValueError(f"不认识的简单预分词器：{self.kind} ✗")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, **self.options}


def _alnum_groups(piece: str) -> list[str]:
    """按**字母数字 / 非字母数字**的边界分组 ✓（同类**连续**字符留在一起 ✓ —— 见 `Whitespace` 注释 ✓）。"""
    groups: list[str] = []
    buffer: list[str] = []
    kind: bool | None = None
    for char in piece:
        current = char.isalnum()
        if kind is not None and current != kind and buffer:
            groups.append("".join(buffer))
            buffer = []
        kind = current
        buffer.append(char)
    if buffer:
        groups.append("".join(buffer))
    return groups


def _punctuation_pieces(text: str, behavior: str) -> list[str]:
    """``Punctuation`` ✓：**按"标点原子"组装** ✓（模型是从参考实现**逐例实测**反推的 ✓ 2026-09-21 ✓）。

    先把文本切成原子：``run``（不含标点的极大段 ✓，**可以含空白** ✓）与 ``punct``（单个标点 ✓）✓
    ⇒ 五种 behavior 的差别**只在"标点怎么贴"** ✓：

    * ``removed`` ✓：只留**非空 run** ✓（`'a, b!!  c'` ⇒ ``['a',' b','  c']`` ✓）；
    * ``isolated`` ✓：每个标点自成一片 ✓ + **run 整段保留** ✓
      （⚠️ **不按空白切** ✗ —— 实测 ``'hi  there'`` ⇒ ``['hi  there']`` ✓、``'  a'`` ⇒ ``['  a']`` ✓；
      我先前据一次**被截断**的输出以为要按空白切 ✓✗ ⇒ 白写了一个 `_split_ws_keep_one` 又删掉 ✓）；
    * ``merged_with_previous`` ✓：标点贴**前一个** run ✓；若前一片本身就是纯标点 ⇒ **另起一片** ✓
      （`'a, b!!  c'` ⇒ ``['a,',' b!','!','  c']`` ✓）；
    * ``merged_with_next`` ✓：标点贴**后一个** run ✓（`⇒ ['a',', b','!','!  c']` ✓）；
    * ``contiguous`` ✓：**连续标点合成一片** ✓，其余按 run ✓（`'a|b||c'` ⇒ ``['a','|','b','||','c']`` ✓；
      且**空白不切** ✗：`'hi  there'` 是**一片** ✓）。

    ⚠️ 前两版都不对 ✗：第一版把空白当分隔（`isolated` 对、`contiguous`/`removed` **全错** ✓✗）；
    第二版状态机也没对齐 ✓ ⇒ 现在是**从数据反推的原子模型** ✓（探针里逐 behavior 零差异 ✓）。
    """
    atoms = _punct_atoms(text)
    if behavior == "removed":
        return [item for kind, item in atoms if kind == "run"]
    pieces: list[str] = []
    for index, (kind, item) in enumerate(atoms):
        if kind == "run":
            if behavior == "merged_with_next" and pieces and pieces[-1] and \
                    index and atoms[index - 1][0] == "punct":
                pieces[-1] = pieces[-1] + item
                continue
            if item:
                pieces.append(item)                     # ⚠️ run **整段保留** ✓（不按空白切 ✗）
            continue
        if behavior in ("merged_with_previous", "merged_with_next"):
            if behavior == "merged_with_previous" and pieces and \
                    not _is_punctuation(pieces[-1][-1]):   # ⚠️ 判**结尾**是不是标点 ✓
                pieces[-1] = pieces[-1] + item             # （不是"整片是不是纯标点" ✗
            else:                                          #   —— 实测 ' b!' 后面那个 '!' 要独立 ✓）
                pieces.append(item)
            continue
        if behavior == "contiguous":
            if pieces and _is_punct_only(pieces[-1]):
                pieces[-1] = pieces[-1] + item
            else:
                pieces.append(item)
            continue
        pieces.append(item)                             # isolated ✓
    return [piece for piece in pieces if piece]


def _is_punct_only(piece: str) -> bool:
    return bool(piece) and all(_is_punctuation(char) for char in piece)


def _punct_atoms(text: str) -> list[tuple[str, str]]:
    """切成原子 ✓：``("run", 不含标点的极大段)`` 与 ``("punct", 单个标点)`` ✓。"""
    atoms: list[tuple[str, str]] = []
    buffer: list[str] = []
    for char in text:
        if _is_punctuation(char):
            if buffer:
                atoms.append(("run", "".join(buffer)))
                buffer = []
            atoms.append(("punct", char))
        else:
            buffer.append(char)
    if buffer:
        atoms.append(("run", "".join(buffer)))
    return atoms


def _digit_pieces(text: str, individual: bool) -> list[str]:
    """``Digits`` ✓：数字逐位切（``individual_digits`` ✓）或整段留一起 ✓。"""
    pieces: list[str] = []
    buffer: list[str] = []
    digits: list[str] = []
    for char in text:
        if char.isdigit():
            if buffer:
                pieces.append("".join(buffer))
                buffer = []
            if individual:
                pieces.append(char)
            else:
                digits.append(char)
            continue
        if digits:
            pieces.append("".join(digits))
            digits = []
        buffer.append(char)
    if digits:
        pieces.append("".join(digits))
    if buffer:
        pieces.append("".join(buffer))
    return pieces


class MetaspacePretokenizer:
    """``Metaspace`` ✓：空格 → ``replacement``（默认 ``▁`` ✓）且**首部补一个** ✓。"""

    def __init__(self, replacement: str = "▁", *, prepend: bool = True) -> None:
        if not replacement:
            raise ValueError("`replacement` 不能为空 ✗")
        self.replacement = replacement
        self.prepend = bool(prepend)

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "MetaspacePretokenizer":
        return cls(str(spec.get("replacement") or "▁"),
                   prepend=str(spec.get("prepend_scheme") or "always") != "never")

    def split(self, text: str) -> list[str]:
        """按**参考实现实测出来的规则**切 ✓（2026-09-20 用 `Metaspace.pre_tokenize_str` 逐例探清 ✓，
        不是照文档猜 ✗ —— 第一版按「空白归一」写 ✓✗，双空格当场与参考不一致 ✓）：

        1. 只对**半角空格**动手 ✓（``\\t``/``\\n`` **不切** ✓：实测 ``"hello\\tworld"`` 是**一片** ✓）；
        2. 首字符不是空格 ⇒ 先在**整串**前面补一个 ``replacement`` ✓（``prepend_scheme="always"`` ✓）；
        3. ⚠️ **首字符判断要连 `replacement` 一起看** ✗（2026-09-21 实测补 ✓）：既不是空格、
           **也不是 `replacement`** 才补一个 ✓ —— 输入本来就带 `▁`（如 ``"▁a"`` ✓）⇒ **不再补** ✓
           （第一版只看空格 ✗ ⇒ ``"▁"`` 被补成 `▁▁` ⇒ 出两个 token ✓✗，与参考不一致 ✓）。
        4. 空格 → ``replacement`` ✓，按 ``replacement`` 切 ✓，**丢掉第一个空块** ✓（它是上一步补出来的 ✓），
           每块再补回一个 ``replacement`` ✓。

        实测对照（``replacement="▁"`` ✓）：``"hello  world"`` ⇒ ``['▁hello','▁','▁world']`` ✓、
        ``" hello"`` ⇒ ``['▁hello']`` ✓（前导空格被"补的那一个"顶掉 ✓）、``"  "`` ⇒ ``['▁','▁']`` ✓、
        ``"▁"`` ⇒ ``['▁']`` ✓、``"▁▁a"`` ⇒ ``['▁','▁a']`` ✓。
        """
        if not text:
            return []
        body = text if text[:1] in (" ", self.replacement) else self.replacement + text
        body = body.replace(" ", self.replacement)
        return [self.replacement + chunk for chunk in body.split(self.replacement)[1:]]


class FixedLengthPretokenizer:
    """``FixedLength`` ✓：按**固定字符数**切 ✓（最后一片可短 ✓ 空串 ⇒ 无片 ✓）。

    实测（2026-09-21 ✓）：``'abcdefg'`` + length 3 ⇒ ``['abc','def','g']`` ✓；
    ``'中文测试x'`` ⇒ ``['中文测','试x']`` ✓ —— 按**码元/字符**数切 ✓（不是字节 ✗）。
    """

    def __init__(self, length: int) -> None:
        size = int(length)
        if size < 1:
            raise ValueError(f"`FixedLength` 的 `length` 必须 ≥ 1 ✗（收到 {length!r} ✓）")
        self.name = "FixedLength"
        self.length = size

    def split(self, text: str) -> list[str]:
        return [text[index:index + self.length]
                for index in range(0, len(text), self.length)]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "length": self.length}


class SplitPretokenizer:
    """``Split`` ✓：按**正则/字面串**匹配切 ✓ —— **五种 behavior 逐例实测**得来 ✓（2026-09-21 ✓）。

    语料 ``pattern = r'\\d'``（单数字 ✓）与 ``r'\\d+'`` ✓；观测（参考实现 ✓）：

    * ``isolated`` ✓：``'ab12cd'`` ⇒ ``['ab','1','2','cd']`` ✓（**非匹配段原样保留** ✓ ——
      ⚠️ **空白不特殊处理** ✗：``'ab 12 cd'`` ⇒ ``['ab ','12',' cd']`` ✓，与 `Punctuation` 那套不同 ✓）；
    * ``contiguous`` ✓：``'ab12cd'`` ⇒ ``['ab','12','cd']`` ✓（**相邻匹配合成一片** ✓）；
    * ``removed`` ✓：``'ab12cd'`` ⇒ ``['ab','cd']`` ✓（匹配段丢掉 ✓ 只留非空段 ✓）；
    * ``merged_with_previous`` ✓：``'ab12cd'`` ⇒ ``['ab1','2','cd']`` ✓
      （匹配并进**前一片** ✓；⚠️ 前一片**已经以匹配结尾**时 ⇒ **另起一片** ✓ —— 第二个 ``'2'`` 独立 ✓）；
    * ``merged_with_next`` ✓：``'ab12cd'`` ⇒ ``['ab','1','2cd']`` ✓（匹配并**后一段** ✓；
      ⚠️ 匹配后面**紧跟另一个匹配**（或到串尾）⇒ 它**自成一片** ✓ —— ``'1'`` 独立 ✓）。

    ``invert`` ✓：把**补集**当匹配 ✓（实测 ``[a-z]+`` + invert ⇒ 匹配的是数字段 ✓）；
    ``pattern`` ✓：``{"Regex": …}`` / ``{"String": …}`` ✓（后者按**字面**切 ✓）。

    ⚠️ 两类**明确拒绝**（构造即报错 ✓ 由 :func:`own_support` 提前说清 ✓）：
    ① **能匹配空串**的正则（如 ``x*`` ✓）—— 参考实现会退化成逐字符切 ✓✗，语义**没核清** ✗ 不模仿；
    ② Python `re` **不支持的语法**（如 ``\\p{N}`` ✓ —— 标准库**没有** ``\\p{...}`` ✗，
       装 `regex` 才能用 ✓ 但本仓不引它 ✗）⇒ 宁可回退参考实现 ✓ 也不静默按错的语义切 ✓✗。
    """

    def __init__(self, pattern: Any, behavior: str = "isolated", *, invert: bool = False) -> None:
        self.name = "Split"
        self.behavior = str(behavior or "isolated")
        if self.behavior not in BEHAVIORS:
            raise ValueError(f"`Split` 不认识的 behavior：{self.behavior!r} ✗"
                             f"（认得：{list(BEHAVIORS)} ✓）")
        self.invert = bool(invert)
        self.literal = self.pattern_text(pattern)
        self._compiled = self._compile(pattern)

    @staticmethod
    def pattern_text(pattern: Any) -> str | None:
        """字面串模式 ✓ ⇒ 返回那个串；正则会返回其源串或 ``None`` ✓（由调用方区分 ✓）。"""
        if isinstance(pattern, dict):
            if "String" in pattern:
                return str(pattern["String"])
            if "Regex" in pattern:
                return str(pattern["Regex"])
            return None
        return str(pattern) if isinstance(pattern, str) else None

    @staticmethod
    def _compile(pattern: Any) -> Any:
        literal = None
        source: str | None = None
        if isinstance(pattern, dict):
            if "String" in pattern:
                literal = str(pattern["String"])
            elif "Regex" in pattern:
                source = str(pattern["Regex"])
        elif isinstance(pattern, str):
            literal = pattern
        if literal is not None:
            if not literal:
                raise ValueError("`Split` 的 `String` 模式为空 ✗")
            return re.compile(re.escape(literal))
        if not source:
            raise ValueError(f"`Split` 的 `pattern` 认不出来 ✗（{pattern!r} ✓）")
        try:
            compiled = re.compile(source)
        except re.error as err:
            raise ValueError(f"`Split` 的正则 Python `re` 编译不了 ✗（{source!r}：{err} ✓）"
                             f"（⚠️ 标准库**没有** `\\p{{...}}` 这类语法 ✓ ⇒ 不静默按错的语义切 ✓）") from err
        if compiled.match("") is not None:
            raise ValueError(f"`Split` 的正则能匹配**空串** ✗（{source!r} ✓）⇒ 语义没核清 ✓"
                             f"（参考实现在这种模式上会退化成逐字符切 ✓✗）不模仿 ✓")
        return compiled

    def _spans(self, text: str) -> list[tuple[int, int]]:
        spans = [match.span() for match in self._compiled.finditer(text)]
        if not self.invert:
            return spans
        # 补集 ✓：把「没被匹配到」的区间当成匹配 ✓
        complement: list[tuple[int, int]] = []
        cursor = 0
        for start, end in spans:
            if start > cursor:
                complement.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < len(text):
            complement.append((cursor, len(text)))
        return complement

    def _atoms(self, text: str) -> list[tuple[bool, str]]:
        atoms: list[tuple[bool, str]] = []
        cursor = 0
        for start, end in self._spans(text):
            if start > cursor:
                atoms.append((False, text[cursor:start]))
            atoms.append((True, text[start:end]))
            cursor = end
        if cursor < len(text):
            atoms.append((False, text[cursor:]))
        return atoms

    def split(self, text: str) -> list[str]:
        behavior = self.behavior
        atoms = self._atoms(text)
        if behavior == "removed":
            return [item for is_match, item in atoms if not is_match and item]
        pieces: list[str] = []
        for index, (is_match, item) in enumerate(atoms):
            if not item:
                continue
            if not is_match:
                pieces.append(item)
                continue
            if behavior == "merged_with_previous":
                # ⚠️ 判据是「**前一片是不是以匹配结尾**」✗ 不是「前一片存不存在」✓（实测 ✓）
                if pieces and atoms[index - 1][0] is False:
                    pieces[-1] = pieces[-1] + item
                else:
                    pieces.append(item)
                continue
            if behavior == "merged_with_next":
                following = atoms[index + 1] if index + 1 < len(atoms) else (True, "")
                if following[0] is False and following[1]:
                    pieces.append(item + following[1])
                    atoms[index + 1] = (True, "")      # 已被吸收 ✓ 免得再出一片 ✓
                else:
                    pieces.append(item)
                continue
            if behavior == "contiguous" and pieces and atoms[index - 1][0] is True:
                pieces[-1] = pieces[-1] + item
                continue
            pieces.append(item)                        # isolated ✓
        return [piece for piece in pieces if piece]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "behavior": self.behavior, "invert": self.invert,
                "pattern": self.literal}


#: ``Split`` 认得的 behavior ✓（与参考实现同一组 ✓）
BEHAVIORS: tuple[str, ...] = ("isolated", "removed", "merged_with_previous", "merged_with_next",
                              "contiguous")


def split_spec_problem(spec: Any) -> str:
    """``Split`` 规格能不能用 ✓（``""`` = 没问题 ✓；否则是**拒绝理由** ✓）。

    ⚠️ 单独抽出来 ✗ 是因为**判定**（`own_support` ✓）与**构造**（`_build_pre_tokenizer` ✓）
    必须同一句话 ✓ —— 否则拒绝理由会说成别的 ✓✗（本仓在 `Precompiled` 上踩过同类坑 ✓）。
    """
    if not isinstance(spec, dict):
        return f"`Split` 规格不是对象 ✗（{type(spec).__name__} ✓）"
    try:
        SplitPretokenizer(spec.get("pattern"), str(spec.get("behavior") or "isolated"),
                          invert=bool(spec.get("invert")))
    except ValueError as err:
        return str(err)
    return ""


def fixed_length_problem(spec: Any) -> str:
    """``FixedLength`` 规格能不能用 ✓（同上 ✓）。"""
    if not isinstance(spec, dict):
        return f"`FixedLength` 规格不是对象 ✗（{type(spec).__name__} ✓）"
    try:
        FixedLengthPretokenizer(spec.get("length") or 0)
    except (ValueError, TypeError) as err:
        return str(err)
    return ""


#: 字节回退的 token 名 ✓（实测：只认 ``<0x`` + **两位十六进制** ✓ —— ``<0xZZ>`` 之类不算 ✗）
BYTE_TOKEN_PATTERN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


class UnigramTokenizer:
    """**Unigram + Viterbi** ✓（词表 = ``[(token, score), …]`` ✓，id = 下标 ✓）。"""

    name = "own-unigram"

    def __init__(self, vocab: Sequence[Sequence[Any]], *, unk_id: int | None = None,
                 pre_tokenizer: Any = None, fuse_unk: bool = True,
                 byte_fallback: bool = False) -> None:
        entries = [(str(item[0]), float(item[1])) for item in vocab]
        if not entries:
            raise ValueError("Unigram 词表是空的 ✗")
        self.vocab = {token: index for index, (token, _) in enumerate(entries)}
        self.scores = {token: score for token, score in entries}
        self.unk_id = 0 if unk_id is None else int(unk_id)
        self.fuse_unk = bool(fuse_unk)
        # ⭐ **字节回退** ✓（2026-09-21 核清参考语义后实现 ✓）：见 :meth:`_byte_ids` ✓
        self.byte_fallback = bool(byte_fallback)
        #: ``字节值 → token id`` ✓（只认 ``<0xNN>`` ✓ —— 实测「怪名字」（如 ``<0xZZ>`` ✓）**不算** ✗）
        self.byte_tokens = {int(match.group(1), 16): self.vocab[token]
                            for token in self.vocab
                            for match in [BYTE_TOKEN_PATTERN.fullmatch(token)] if match}
        #: 未知字符的得分 ✓（SentencePiece 血统的常见口径：最低分 − 10 ✓）
        self.unk_score = min(self.scores.values()) - 10.0 if self.scores else -10.0
        self._trie = self._build_trie(entries)
        self.pre_tokenizer = pre_tokenizer
        self.max_token_len = max(len(token) for token, _ in entries)
        #: ⚠️ **前缀树扫描次数** ✓（可观测 ✗）—— 「优化有没有效」不能靠感觉 ✓：
        #: 每段文本的扫描次数应当 ≈ 字符数（每个起点扫一次 ✓）✓；若随长度**平方增长** ✓✗
        #: 就是第一版那种退化 ✓（基准里会把它打出来 ✓ 自检里也会断言它的量级 ✓）。
        self.trieScans = 0
        #: ⭐ **片段级缓存** ✓（2026-09-21 ✓）：跨文本复用的**预分词片段**很多 ✓
        #: （提示词里反复出现的词 / 标点 / 空格组合 ✓）⇒ 命中就省掉整段 Viterbi ✓。
        #: ⚠️ 上限后**整体清空** ✓（不做 LRU ✗ —— 换一篇就换工作集 ✓ 简单规则够用 ✓ 且不抖动 ✓）
        #: ⚠️ 返回值**必须是拷贝** ✗（否则调用方 `extend` 会把缓存里的列表改掉 ✓✗）
        self._pieceCache: dict[str, tuple[int, ...]] = {}
        self.pieceCacheHits = 0
        self.pieceCacheMisses = 0

    @staticmethod
    def _build_trie(entries: Iterable[tuple[str, float]]) -> dict[str, Any]:
        """前缀树 ✓（词表可能十万级 ✗ ⇒ 逐位置暴力扫会因为"最短匹配"错、暴力枚举又太慢 ✓）。"""
        root: dict[str, Any] = {}
        for token, _ in entries:
            node = root
            for char in token:
                node = node.setdefault(char, {})
            node["\0"] = True
        return root

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def required_vocab_size(self) -> int:
        return len(self.vocab)

    def _matches(self, piece: str, start: int) -> list[tuple[str, float]]:
        """从前缀树里取出**以 start 开头**的全部词表项 ✓（长到短都留 ✓ Viterbi 才会对 ✓）。"""
        self.trieScans += 1
        node = self._trie
        found: list[tuple[str, float]] = []
        index = start
        while index < len(piece):
            node = node.get(piece[index])
            if node is None:
                break
            index += 1
            if "\0" in node:
                token = piece[start:index]
                found.append((token, self.scores[token]))
        return found

    def _encode_piece(self, piece: str) -> list[int]:
        size = len(piece)
        # ⚠️⚠️ **每个起点只扫一次前缀树** ✓（2026-09-21 优化 ✓）：
        #    第一版是「对每个 `end`，把 `end-maxTokenLen…end` 的起点**重新扫一遍**」✗
        #    ⇒ 同一个起点被扫约 `maxTokenLen` 遍 ✓✗（基准：约千字符扫 5456 次 ⇒ 就是它 ✓）。
        #    改成「先按起点算好匹配表 ✓ 再做**前向** DP」✓ ⇒ 扫描次数降到 ≈ 字符数 ✓。
        #    ⚠️ 语义**必须一模一样** ✗：DP 的更新顺序变了 ✓，但取的是**严格大于**才替换 ✓
        #    ⇒ 平局时的选择与第一版一致 ✓（对拍靠自检里那批「与参考逐例同 id」✓）。
        matches = [self._matches(piece, start) for start in range(size)]
        #: `best[i]` = 前 i 个字符的**最优总分** ✓；`back[i]` = (起点, token 或 None=未知 ✓)
        best = [float("-inf")] * (size + 1)
        back: list[tuple[int, str | None]] = [(0, None)] * (size + 1)
        best[0] = 0.0
        unk_score = self.unk_score
        for start in range(size):
            base = best[start]
            if base == float("-inf"):                    # pragma: no cover - 未知兜底保证不会发生 ✓
                continue
            for token, score in matches[start]:
                end = start + len(token)
                candidate = base + score
                if candidate > best[end]:
                    best[end] = candidate
                    back[end] = (start, token)
            # 未知单字符兜底 ✓（**只有一个字符** ✓ —— 连续未知由 `fuse_unk` 在下游合并 ✓）
            candidate = base + unk_score
            if candidate > best[start + 1]:
                best[start + 1] = candidate
                # ⭐ 开了字节回退 ⇒ 这个未知字符**先记成它的字节序列** ✓（真的展开在 :meth:`_byte_ids` ✓）
                back[start + 1] = (start, (piece[start],))
        tokens: list[str | tuple[str, ...] | None] = []
        cursor = size
        while cursor > 0:
            start, token = back[cursor]
            tokens.append(token)
            cursor = start
        tokens.reverse()
        ids: list[int] = []

        def emit_unknown(chars: list[str]) -> None:
            """把一段**连续未知字符**落成 id ✓（⭐ 2026-09-21 实测出来的**段级**判据 ✓）。

            ⚠️⚠️ **判据是「段」不是「字符」** ✗（第一版按字符判 ⇒ 与参考不一致 ✓✗）：
            实测（词表只有 ``▁``/``a`` + 「中」的三个字节 token ✓）：

            * ``'中中'`` ⇒ 段内**每个**字符都能展开 ⇒ 逐字符展开 ✓（6 个字节 token ✓）；
            * ``'中x'`` ⇒ 段内 ``x`` 展开不了 ⇒ **整段合成一个 `unk`** ✓（``'中'`` **也不**展开 ✓✗）；
            * ``'a中x'`` ⇒ 词表能匹配的 ``a`` **照常保留** ✓（``[▁, a, unk('中x')]`` ✓）⇒ 段 ≠ 整片 ✓；
            * ``'中中x'`` ⇒ 整段一个 unk ✓（前两个本可展开也不展开 ✓）。
            """
            expanded = [self._byte_ids(char) for char in chars]
            if all(item is not None for item in expanded):
                for item in expanded:
                    ids.extend(item or [])
                return
            if self.fuse_unk:
                ids.append(self.unk_id)                  # 整段一个 ✓
            else:
                ids.extend([self.unk_id] * len(chars))   # 逐字符各一个 ✓（与不回流时的口径一致 ✓）

        pending: list[str] = []
        for token in tokens:
            if token is None:                            # pragma: no cover - 新逻辑不再产出 ✓
                continue
            if isinstance(token, tuple):                 # ⭐ 未知字符位 ✓ ⇒ 先攒着 ✓（段级判据 ✓）
                pending.append(token[0])
                continue
            if pending:
                emit_unknown(pending)
                pending = []
            ids.append(self.vocab[token])
        if pending:
            emit_unknown(pending)
        return ids

    def _byte_ids(self, char: str) -> list[int] | None:
        """把**一个字符**展开成它的 UTF-8 字节 token id ✓（回退不了 ⇒ ``None`` ✓）。

        ⚠️ 参考语义**逐条实测**（2026-09-21 ✓ 第一轮探错了 ✗ —— 见 `own_support` 的注释 ✓）：

        * 判据是**该字符的每一个字节**都得有 ``<0xNN>`` token ✓ —— 缺**任何一个**就**回退不了** ✓
          （实测：词表只有 ``<0xE4>`` 时，``'中'``（``E4 B8 AD``）⇒ 回退不了 ✓ 而不是只出 ``<0xE4>`` ✓✗）；
        * 字符本身在词表里 ⇒ 走**正常词表 token** ✓（字节只是兜底 ✓）；展开**不与 `unk` 合并** ✓；
        * ⚠️ 但「一个字符能否回退」**只是零件** ✗ —— 最终要不要展开由 :func:`emit_unknown` 按**段**定 ✓。
        """
        if not self.byte_fallback:
            return None
        ids = [self.byte_tokens.get(byte) for byte in char.encode("utf-8")]
        if any(item is None for item in ids):
            return None
        return [int(item) for item in ids]

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        pieces = (self.pre_tokenizer.split(str(text or "")) if self.pre_tokenizer is not None
                  else [str(text or "")])
        ids: list[int] = []
        cache = self._pieceCache
        for piece in pieces:
            if not piece:
                continue
            hit = cache.get(piece)
            if hit is not None:
                self.pieceCacheHits += 1
                ids.extend(hit)
                continue
            self.pieceCacheMisses += 1
            encoded = self._encode_piece(piece)
            if len(cache) >= PIECE_CACHE_LIMIT:
                cache.clear()                       # 换工作集 ⇒ 整体清空 ✓（见上面的理由 ✓）
            cache[piece] = tuple(encoded)
            ids.extend(encoded)
        return ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        index = {value: key for key, value in self.vocab.items()}
        pieces = [index.get(int(item), "") for item in ids]
        text = "".join(pieces)
        if isinstance(self.pre_tokenizer, MetaspacePretokenizer):
            text = text.replace(self.pre_tokenizer.replacement, " ").strip()
        return text

    def cache_stats(self) -> dict[str, int]:
        """片段缓存现状 ✓（命中率要能读出来 ✗ —— 否则"加了缓存"无法自证 ✓）。"""
        total = self.pieceCacheHits + self.pieceCacheMisses
        return {"entries": len(self._pieceCache), "hits": self.pieceCacheHits,
                "misses": self.pieceCacheMisses, "limit": PIECE_CACHE_LIMIT,
                "hitRate": round(self.pieceCacheHits / total, 4) if total else 0.0}

    def clear_cache(self) -> None:
        self._pieceCache.clear()

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "vocabSize": self.vocab_size, "unkId": self.unk_id,
                "fuseUnk": self.fuse_unk, "maxTokenLen": self.max_token_len,
                # ⭐ 字节回退**要能读出来** ✗（否则「开没开」无法自证 ✓）—— 连**词表里有几个字节 token** 一起报 ✓
                "byteFallback": self.byte_fallback, "byteTokens": len(self.byte_tokens)}

    def fingerprint(self) -> dict[str, Any]:
        return {"vocabSize": self.vocab_size, "unkId": self.unk_id}


class WordPieceTokenizer:
    """**WordPiece** ✓：贪心最长匹配 + ``##`` 续接 ✓；⚠️ **整词失败 ⇒ 整个词 `[UNK]`** ✓。"""

    name = "own-wordpiece"

    def __init__(self, vocab: dict[str, int], *, unk_token: str = "[UNK]",
                 prefix: str = "##", max_chars: int = 100,
                 pre_tokenizer: Any = None) -> None:
        if not vocab:
            raise ValueError("WordPiece 词表是空的 ✗")
        self.vocab = {str(token): int(index) for token, index in vocab.items()}
        self.unk_token = str(unk_token)
        if self.unk_token not in self.vocab:
            raise ValueError(f"词表里没有 `{self.unk_token}` ✗（WordPiece 必须有 unk ✓）")
        self.unk_id = self.vocab[self.unk_token]
        self.prefix = str(prefix)
        self.max_chars = int(max_chars)
        self.pre_tokenizer = pre_tokenizer
        #: ⭐ 片段级缓存 ✓（与 `UnigramTokenizer` 同一套规则与理由 ✓ 见那里的注释 ✓）
        self._pieceCache: dict[str, tuple[int, ...]] = {}
        self.pieceCacheHits = 0
        self.pieceCacheMisses = 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def required_vocab_size(self) -> int:
        return max(self.vocab.values()) + 1

    def _encode_word(self, word: str) -> list[int]:
        if len(word) > self.max_chars:
            return [self.unk_id]
        ids: list[int] = []
        start = 0
        while start < len(word):
            end = len(word)
            piece_id = None
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = self.prefix + piece
                if piece in self.vocab:
                    piece_id = self.vocab[piece]
                    break
                end -= 1
            if piece_id is None:
                return [self.unk_id]        # ⚠️ 整词 UNK ✓（不是部分切 ✓）
            ids.append(piece_id)
            start = end
        return ids

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        pieces = (self.pre_tokenizer.split(str(text or "")) if self.pre_tokenizer is not None
                  else [str(text or "")])
        ids: list[int] = []
        cache = self._pieceCache
        for piece in pieces:
            if not piece:
                continue
            hit = cache.get(piece)
            if hit is not None:
                self.pieceCacheHits += 1
                ids.extend(hit)
                continue
            self.pieceCacheMisses += 1
            encoded = self._encode_word(piece)
            if len(cache) >= PIECE_CACHE_LIMIT:
                cache.clear()
            cache[piece] = tuple(encoded)
            ids.extend(encoded)
        return ids

    def cache_stats(self) -> dict[str, int]:
        """片段缓存现状 ✓（与 `UnigramTokenizer` 同一口径 ✓）。"""
        total = self.pieceCacheHits + self.pieceCacheMisses
        return {"entries": len(self._pieceCache), "hits": self.pieceCacheHits,
                "misses": self.pieceCacheMisses, "limit": PIECE_CACHE_LIMIT,
                "hitRate": round(self.pieceCacheHits / total, 4) if total else 0.0}

    def clear_cache(self) -> None:
        self._pieceCache.clear()

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        index = {value: key for key, value in self.vocab.items()}
        out: list[str] = []
        for item in ids:
            token = index.get(int(item), "")
            out.append(token[len(self.prefix):] if token.startswith(self.prefix) else
                       (" " + token if out else token))
        return "".join(out)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "vocabSize": self.vocab_size, "unkToken": self.unk_token,
                "continuingPrefix": self.prefix, "maxChars": self.max_chars}

    def fingerprint(self) -> dict[str, Any]:
        return {"vocabSize": self.vocab_size, "unkToken": self.unk_token}


#: 自研覆盖的**模型** ✓（三种血统 ✓）
_OWN_MODELS: tuple[str, ...] = ("BPE", "Unigram", "WordPiece")

#: 自研覆盖的**预分词器** ✓（⚠️ 每一条的规则都**逐例实测**过 ✓ —— 见各实现的注释 ✓）。
#: 明确**不做** ✗：`UnicodeScripts`（要 Unicode script 表 ✓ 标准库**没有** ✗）、
#: `Split` 里**能匹配空串**的正则 ✓ 与 `\p{...}` 语法 ✓（见 `SplitPretokenizer` 注释 ✓）
#: —— 都由 :func:`own_support` **带理由**拒绝 ✓ 交给回退 ✓。
_OWN_PRE_TOKENIZERS: tuple[str, ...] = (
    "ByteLevel", "Metaspace", "BertPreTokenizer",
    "Whitespace", "WhitespaceSplit", "Punctuation", "Digits", "CharDelimiterSplit",
    "Split", "FixedLength",
)


def own_support(form: dict[str, Any]) -> OwningSupport:
    """自研**能不能**覆盖这份词表 ✓ + 理由 ✓（再决定要不要参考实现 ✓）。

    ``form`` 里的字段：``modelType`` / ``preTokenizer`` / ``normalizerTypes``（**展平** ✓）/
    ``byteFallback`` ✓ —— 由 :func:`.tokenizer_hub.detect_form` 提供 ✓（同一口径 ✓）。
    """
    model_type = str(form.get("modelType") or "")
    pre_type = str(form.get("preTokenizer") or "")
    normalizer_types = [str(item) for item in (form.get("normalizerTypes") or [])]
    unsupported = [item for item in normalizer_types if item not in SUPPORTED_NORMALIZERS]
    if unsupported:
        hints = "".join(_unsupported_hint(item) for item in unsupported)
        return OwningSupport(False, f"`normalizer` 含未实现项 `{unsupported}` ✗{hints}"
                                    f"（已实现：{list(SUPPORTED_NORMALIZERS)} ✓）"
                                    f"⇒ 自研不硬套 ✓（跳过规范化 = 用错的文本查词表 ✓✗）", form)
    if form.get("byteFallback") and model_type != "Unigram":
        # ⚠️⚠️ 这里**改过口径** ✗（2026-09-21 第二轮 ✓）：第一轮实测「`Unigram(byte_fallback=True)` 对未知字符
        #      仍出 `unk`」⇒ 判成「语义未核清 ✓ 拒绝 ✓」✗ —— ⚠️ 那个实测**少了前置条件** ✗：
        #      词表里**没有** `<0xNN>` 字节 token ✓✗（没有字节 token 当然回退不出去 ✓）。
        #      补上字节 token 再测 ⇒ 真走回退 ✓（`'中'` ⇒ `<0xE4><0xB8><0xAD>` ✓）⇒ 现在 **Unigram 已实现** ✓。
        #    ⚠️ 但 **BPE 的 `byte_fallback` 仍未核** ✗（触发条件与 Unigram 不同 ✓ —— BPE 是字节级词表 ✓
        #      回退只在「该字节不在 byte↔unicode 映射里」时才可能发生 ✓）⇒ 继续**带理由拒绝** ✓ 不硬套 ✓。
        return OwningSupport(False, f"`byte_fallback: true` + 模型 `{model_type}` ✗ ⇒ 实测与实现的"
                                    f"判据只在 **Unigram** 上核清过 ✓（逐字符字节展开 ✓ 缺一字节则整体 `unk` ✓）"
                                    f"；该模型的触发条件未核 ✓ 不硬套 ✓", form)
    if model_type not in _OWN_MODELS:
        return OwningSupport(False, f"模型 `{model_type}` ✗ 不在自研覆盖表里 ✓"
                                    f"（覆盖：{list(_OWN_MODELS)} ✓）", form)
    if pre_type not in _OWN_PRE_TOKENIZERS:
        return OwningSupport(False, f"预分词器 `{pre_type}` ✗ 未实现 ✓"
                                    f"（覆盖：{list(_OWN_PRE_TOKENIZERS)} ✓）", form)
    # ⚠️ `Split` / `FixedLength` 的**参数**也可能不合格 ✗（空匹配正则 ✓ `\p{...}` ✓ length≤0 ✓）
    #    ⇒ 判定与构造**用同一句话** ✓（否则拒绝理由会说成别的 ✓✗）。
    pre_spec = form.get("preTokenizerSpec")
    if pre_type == "Split":
        problem = split_spec_problem(pre_spec)
        if problem:
            return OwningSupport(False, problem, form)
    if pre_type == "FixedLength":
        problem = fixed_length_problem(pre_spec)
        if problem:
            return OwningSupport(False, problem, form)
    suffix = f" + normalizer `{normalizer_types}` ✓" if normalizer_types else ""
    return OwningSupport(True, f"自研实现覆盖 `{model_type}` + `{pre_type}`{suffix} ✓", form)


def load_own_tokenizer(path: str | Path) -> Any | None:
    """按形态装载**自研**实现 ✓（覆盖不了 ⇒ ``None`` ✓ —— 由调用方决定回退还是报错 ✓）。"""
    target = Path(path)
    json_path = target if target.is_file() and target.name.endswith(".json") \
        else target / "tokenizer.json"
    if not json_path.exists():
        # 只有 `vocab.json`+`merges.txt` ⇒ 就是 ByteLevel-BPE ✓（自研 BPE 覆盖 ✓）
        try:
            return tokenizer_bpe.load_tokenizer(path)
        except Exception:  # noqa: BLE001 —— 找不到就是找不到 ✓ 由调用方报 ✓
            return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    model = payload.get("model") or {}
    normalizer_spec = payload.get("normalizer")
    form = {"modelType": str(model.get("type") or ""),
            "preTokenizer": _pre_tokenizer_type(payload.get("pre_tokenizer")),
            # ⚠️⚠️ **原始规格必须带上** ✗ —— 少了它，`Split` 的**正则** / `FixedLength` 的
            #    **length** 就无从校验 ⇒ `own_support` 会把这类型一律判成"参数不合格" ✓✗
            #    （2026-09-21 实测：`Split` 词表被**误判回退** ✓✗，自检 ㉘ 当场红 ✓）。
            #    ⚠️ 这份 `form` 与 :func:`.tokenizer_hub.detect_form` 是**同口径的两处** ✓
            #    ⇒ 字段要一起加 ✗（改一处记得改另一处 ✓）。
            "preTokenizerSpec": payload.get("pre_tokenizer"),
            "normalizerType": _normalizer_type(normalizer_spec),
            "normalizerTypes": normalizer_spec_types(normalizer_spec),
            "byteFallback": bool(model.get("byte_fallback"))}
    support = own_support(form)
    if not support.ok:
        return None
    pre = _build_pre_tokenizer(payload.get("pre_tokenizer"))
    if form["modelType"] == "BPE":
        impl: Any = tokenizer_bpe.load_tokenizer(json_path)
    elif form["modelType"] == "Unigram":
        impl = UnigramTokenizer(model.get("vocab") or [], unk_id=model.get("unk_id"),
                                pre_tokenizer=pre,
                                fuse_unk=bool(model.get("fuse_unk", True)),
                                byte_fallback=bool(model.get("byte_fallback")))
    else:
        impl = WordPieceTokenizer(model.get("vocab") or {},
                                  unk_token=str(model.get("unk_token") or "[UNK]"),
                                  prefix=str(model.get("continuing_subword_prefix") or "##"),
                                  max_chars=int(model.get("max_input_chars_per_word") or 100),
                                  pre_tokenizer=pre)
    # ⭐ normalizer 统一在这一层套 ✓（三种模型共用一个包装 ✓ ⇒ 不必各自改 ✗）
    normalizer, reason = build_normalizer(normalizer_spec)
    if normalizer is None and normalizer_spec:
        return None                       # ⚠️ 有 spec 却造不出来 ⇒ **拒绝** ✓（不静默跳过 ✗）
    if normalizer is not None:
        return NormalizedTokenizer(impl, normalizer)
    return impl


def _pre_tokenizer_type(spec: Any) -> str:
    node = spec
    if isinstance(node, dict) and str(node.get("type")) == "Sequence":
        inner = node.get("pretokenizers") or []
        if len(inner) == 1:
            node = inner[0]
    if node is None:
        return "ByteLevel"
    return str((node or {}).get("type") or "") if isinstance(node, dict) else ""


def _normalizer_type(spec: Any) -> str:
    if not spec:
        return ""
    return str((spec or {}).get("type") or "") if isinstance(spec, dict) else ""


def _build_pre_tokenizer(spec: Any) -> Any:
    node = spec
    if isinstance(node, dict) and str(node.get("type")) == "Sequence":
        inner = node.get("pretokenizers") or []
        node = inner[0] if len(inner) == 1 else node
    if not isinstance(node, dict):
        return None
    node_type = str(node.get("type") or "")
    if node_type == "Metaspace":
        return MetaspacePretokenizer.from_spec(node)
    if node_type == "BertPreTokenizer":
        return BertPreTokenizer()
    if node_type == "Split":
        return SplitPretokenizer(node.get("pattern"), str(node.get("behavior") or "isolated"),
                                 invert=bool(node.get("invert")))
    if node_type == "FixedLength":
        return FixedLengthPretokenizer(node.get("length") or 0)
    if node_type in ("Whitespace", "WhitespaceSplit", "Punctuation", "Digits", "CharDelimiterSplit"):
        return SimplePreTokenizer(node_type, **{key: value for key, value in node.items()
                                                if key != "type"})
    return None
