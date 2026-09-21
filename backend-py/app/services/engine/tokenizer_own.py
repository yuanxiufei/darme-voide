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
* ``byte_fallback: true`` 的 Unigram ⇒ **不走自研** ✓ —— 实测（2026-09-21）参考实现**该配置下并没有**
  走字节回退 ✗（未知字符仍出 ``unk`` ✓）⇒ **语义没核清** ✓ 不按猜的实现 ✓；
* 预分词器不认识（``CharDelimiterSplit`` / ``Digits`` / ``Split`` …）⇒ **不走自研** ✓。

⇒ 走不了自研时由 :mod:`.tokenizer_hub` 决定**回退参考实现**还是报错 ✓（本模块只回答「我行不行」✓）。
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import tokenizer_bpe

__all__ = ["BertNormalizer", "BertPreTokenizer", "MetaspacePretokenizer", "NormalizedTokenizer",
           "OwningSupport", "SequenceNormalizer", "UnicodeNormalizer", "UnigramTokenizer",
           "WordPieceTokenizer", "build_normalizer", "normalizer_spec_types", "own_support",
           "load_own_tokenizer", "SUPPORTED_NORMALIZERS"]

#: ⚠️ 自研**实现了**的 normalizer ✓；不在表里的 ⇒ **拒绝** ✓（由 hub 决定回退还是报错 ✓）。
#: 明确不做 ✗：`Precompiled`（要 SentencePiece 的 **charsmap 表** ✓ —— 实测
#: ``Precompiled.__new__() missing 1 required positional argument`` ✓✗，
#: 那张表不是"NFKC 换个写法" ✓ 不能假装 ✓）、`Nmt`、以及一切认不出的 ✓。
SUPPORTED_NORMALIZERS: tuple[str, ...] = (
    "BertNormalizer", "NFC", "NFD", "NFKC", "NFKD",
    "Lowercase", "StripAccents", "Strip", "Prepend", "Replace", "Sequence",
)


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
        3. 空格 → ``replacement`` ✓，按 ``replacement`` 切 ✓，**丢掉第一个空块** ✓（它是上一步补出来的 ✓），
           每块再补回一个 ``replacement`` ✓。

        实测对照（``replacement="▁"`` ✓）：``"hello  world"`` ⇒ ``['▁hello','▁','▁world']`` ✓、
        ``" hello"`` ⇒ ``['▁hello']`` ✓（前导空格被"补的那一个"顶掉 ✓）、``"  "`` ⇒ ``['▁','▁']`` ✓。
        """
        if not text:
            return []
        body = text if text.startswith(" ") else self.replacement + text
        body = body.replace(" ", self.replacement)
        return [self.replacement + chunk for chunk in body.split(self.replacement)[1:]]


class UnigramTokenizer:
    """**Unigram + Viterbi** ✓（词表 = ``[(token, score), …]`` ✓，id = 下标 ✓）。"""

    name = "own-unigram"

    def __init__(self, vocab: Sequence[Sequence[Any]], *, unk_id: int | None = None,
                 pre_tokenizer: Any = None, fuse_unk: bool = True) -> None:
        entries = [(str(item[0]), float(item[1])) for item in vocab]
        if not entries:
            raise ValueError("Unigram 词表是空的 ✗")
        self.vocab = {token: index for index, (token, _) in enumerate(entries)}
        self.scores = {token: score for token, score in entries}
        self.unk_id = 0 if unk_id is None else int(unk_id)
        self.fuse_unk = bool(fuse_unk)
        #: 未知字符的得分 ✓（SentencePiece 血统的常见口径：最低分 − 10 ✓）
        self.unk_score = min(self.scores.values()) - 10.0 if self.scores else -10.0
        self._trie = self._build_trie(entries)
        self.pre_tokenizer = pre_tokenizer
        self.max_token_len = max(len(token) for token, _ in entries)

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
        #: `best[i]` = 前 i 个字符的**最优总分** ✓；`back[i]` = (起点, token 或 None=未知 ✓)
        best = [float("-inf")] * (size + 1)
        back: list[tuple[int, str | None]] = [(0, None)] * (size + 1)
        best[0] = 0.0
        for end in range(1, size + 1):
            for start in range(max(0, end - self.max_token_len), end):
                if best[start] == float("-inf"):
                    continue
                for token, score in self._matches(piece, start):
                    if start + len(token) != end:
                        continue
                    candidate = best[start] + score
                    if candidate > best[end]:
                        best[end] = candidate
                        back[end] = (start, token)
            # 未知单字符兜底 ✓（**只有一个字符** ✓ —— 连续未知由 `fuse_unk` 在下游合并 ✓）
            single = best[end - 1]
            if single != float("-inf") and single + self.unk_score > best[end]:
                best[end] = single + self.unk_score
                back[end] = (end - 1, None)
        tokens: list[str | None] = []
        cursor = size
        while cursor > 0:
            start, token = back[cursor]
            tokens.append(token)
            cursor = start
        tokens.reverse()
        ids: list[int] = []
        pending_unknown = False
        for token in tokens:
            if token is None:
                pending_unknown = True
                continue
            if pending_unknown and self.fuse_unk:
                ids.append(self.unk_id)     # 连续未知**合并成一个** ✓（参考实现默认如此 ✓）
            elif pending_unknown:
                ids.append(self.unk_id)
            pending_unknown = False
            ids.append(self.vocab[token])
        if pending_unknown:
            ids.append(self.unk_id)
        return ids

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        pieces = (self.pre_tokenizer.split(str(text or "")) if self.pre_tokenizer is not None
                  else [str(text or "")])
        ids: list[int] = []
        for piece in pieces:
            if piece:
                ids.extend(self._encode_piece(piece))
        return ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        index = {value: key for key, value in self.vocab.items()}
        pieces = [index.get(int(item), "") for item in ids]
        text = "".join(pieces)
        if isinstance(self.pre_tokenizer, MetaspacePretokenizer):
            text = text.replace(self.pre_tokenizer.replacement, " ").strip()
        return text

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "vocabSize": self.vocab_size, "unkId": self.unk_id,
                "fuseUnk": self.fuse_unk, "maxTokenLen": self.max_token_len}

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
        for piece in pieces:
            if piece:
                ids.extend(self._encode_word(piece))
        return ids

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
#: 明确**不做** ✗：`Split`（behavior 组合多 ✗ 未逐条核过 ✓）、`FixedLength`、`UnicodeScripts`、
#: `WhitespaceSplit` 之外的组合等 ✓ ⇒ 由 :func:`own_support` 拒绝 ✓ 交给回退 ✓。
_OWN_PRE_TOKENIZERS: tuple[str, ...] = (
    "ByteLevel", "Metaspace", "BertPreTokenizer",
    "Whitespace", "WhitespaceSplit", "Punctuation", "Digits", "CharDelimiterSplit",
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
    if form.get("byteFallback"):
        # ⚠️ 实测（2026-09-21）：参考实现在**该配置下并没有**走字节回退 ✓✗
        #    （`Unigram(byte_fallback=True)` 对未知字符仍出 `unk` ✓）⇒ 语义**没核清** ✗
        #    ⇒ 继续拒绝 ✓（不按猜的语义实现 ✓）。
        return OwningSupport(False, "`byte_fallback: true` ✗ ⇒ 实测参考实现该配置下**没走**字节回退 ✓"
                                    "（未知字符仍出 `unk` ✓）⇒ 语义未核清 ✓ 不硬套 ✓", form)
    if model_type not in _OWN_MODELS:
        return OwningSupport(False, f"模型 `{model_type}` ✗ 不在自研覆盖表里 ✓"
                                    f"（覆盖：{list(_OWN_MODELS)} ✓）", form)
    if pre_type not in _OWN_PRE_TOKENIZERS:
        return OwningSupport(False, f"预分词器 `{pre_type}` ✗ 未实现 ✓"
                                    f"（覆盖：{list(_OWN_PRE_TOKENIZERS)} ✓）", form)
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
                                fuse_unk=bool(model.get("fuse_unk", True)))
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
    if node_type in ("Whitespace", "WhitespaceSplit", "Punctuation", "Digits", "CharDelimiterSplit"):
        return SimplePreTokenizer(node_type, **{key: value for key, value in node.items()
                                                if key != "type"})
    return None
