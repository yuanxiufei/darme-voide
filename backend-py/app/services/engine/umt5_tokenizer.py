"""UMT5 的**分词器** ✓（Wan2.1/2.2 的提示词入口 ✓）。

⚠️ 为什么不能拿"差不多"的分词器顶 ✓✗：
UMT5 用的是 sentencepiece **unigram** ＋ ``nmt_nfkc`` 归一化 ＋ ``add_dummy_prefix`` ✓
（词表 256384、``<pad>=0``、``</s>=1`` ✓）—— 与 Qwen 系的 BPE 完全两套 ✗。
拿错一个，提示词会**逐字变成别的词** ✓✗，症状是"图/视频跟提示词关系很淡"，却**不报任何错** ✓。

两条路 ✓（真实通路优先 ✓）：

1. **sentencepiece 直读** ✓（首选 ✓）：它带 ``precompiled_charsmap`` ✓ —— 全角/兼容字符归一化
   只有它做得对 ✓；
2. **自研回落** ✓（没有 sentencepiece 库时 ✓）：自己解 ``ModelProto`` 的 ``pieces`` ✓ 交给
   :class:`tokenizer_own.UnigramTokenizer` 走 Viterbi ✓。**如实承认缺口** ✗：这一路**没有**
   ``nmt_nfkc`` 字符归一化 ✓（只对 ASCII 提示词等价 ✓），回执里必须写明 ✓（见 ``to_dict()``）。

⚠️ 权重文件自带分词器 ✓：``umt5_xxl_fp8_e4m3fn_scaled.safetensors`` 里有一张 ``spiece_model``
（``U8``、4 548 313 字节 ✓）—— 它是**唯一权威**来源 ✓，本模块 :func:`load_tokenizer`
会先试它 ✓，其次才是盘上的 ``tokenizer.model`` ✓。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from .tokenizer_own import UnigramTokenizer

__all__ = [
    "Umt5Tokenizer",
    "available",
    "load_tokenizer",
    "parse_sp_model",
]

#: UMT5 的填空 / 句末 id ✓（与 :mod:`.umt5_te` 的常量同源 ✓）
PAD_ID = 0
EOS_ID = 1

#: ``SentencePiece.ModelProto`` 的字段号 ✓
_F_PIECES = 1
_F_NORMALIZER = 3


def available() -> bool:
    """有没有 sentencepiece 库 ✓（真值路 ✓）。"""
    try:
        import sentencepiece  # noqa: F401
    except Exception:  # pragma: no cover - 环境决定 ✓
        return False
    return True


# --------------------------------------------------------------------------- proto 解析


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("varint 越界 ✗（不是 sentencepiece ModelProto）")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint 过长 ✗")


def _fields(buf: bytes) -> list[tuple[int, int, bytes | int]]:
    """把一段 protobuf 拆成 ``(字段号, 线型, 值)`` ✓（只认 varint 与定长 ✓）。"""
    out: list[tuple[int, int, bytes | int]] = []
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 0x7
        if wire == 0:
            value, pos = _varint(buf, pos)
            out.append((field, wire, value))
        elif wire == 2:
            length, pos = _varint(buf, pos)
            out.append((field, wire, buf[pos:pos + length]))
            pos += length
        elif wire == 5:
            out.append((field, wire, buf[pos:pos + 4]))
            pos += 4
        elif wire == 1:
            out.append((field, wire, buf[pos:pos + 8]))
            pos += 8
        else:  # pragma: no cover - sentencepiece 不会用 3/4 ✓
            raise ValueError(f"不认识的 protobuf 线型 {wire} ✗")
    return out


def parse_sp_model(data: bytes) -> list[tuple[str, float, int]]:
    """解 ``ModelProto.pieces`` ⇒ ``[(piece, score, type)]`` ✓（id 就是列表下标 ✓）。"""
    pieces: list[tuple[str, float, int]] = []
    for field, wire, value in _fields(data):
        if field != _F_PIECES or wire != 2 or not isinstance(value, bytes):
            continue
        piece, score, kind = b"", 0.0, 1
        for sub, sub_wire, sub_value in _fields(value):
            if sub == 1 and sub_wire == 2 and isinstance(sub_value, bytes):
                piece = sub_value
            elif sub == 2 and sub_wire == 5 and isinstance(sub_value, bytes):
                score = struct.unpack("<f", sub_value)[0]
            elif sub == 3 and sub_wire == 0 and isinstance(sub_value, int):
                kind = sub_value
        pieces.append((piece.decode("utf-8", "replace"), score, kind))
    if not pieces:
        raise ValueError("这份 sentencepiece 模型里一条 piece 都没有 ✗")
    return pieces


def _normalizer_add_dummy_prefix(data: bytes) -> bool:
    """``NormalizerSpec.add_dummy_prefix`` ✓（umt5 是 True ✓ —— 句首补一个 ``▁``）。"""
    for field, wire, value in _fields(data):
        if field != _F_NORMALIZER or wire != 2 or not isinstance(value, bytes):
            continue
        for sub, sub_wire, sub_value in _fields(value):
            if sub == 3 and sub_wire == 0 and isinstance(sub_value, int):
                return bool(sub_value)
    return True


# --------------------------------------------------------------------------- 分词器


class Umt5Tokenizer:
    """UMT5 分词 ✓：``encode(text)`` ⇒ (ids, mask)，末尾**必带** ``</s>`` ✓、补到 ``max_length`` ✓。"""

    def __init__(self, backends: Any = None, *, vocab: list[tuple[str, float, int]] | None = None,
                 add_dummy_prefix: bool = True, max_length: int = 512) -> None:
        self._sp = backends
        self._vocab = vocab
        self._own: UnigramTokenizer | None = None
        if backends is None and vocab is not None:
            # 只收 NORMAL(1) 与 USER_DEFINED(4) ✓（UNKNOWN/CONTROL/BYTE/UNUSED 不进分词表 ✗）
            self._own = UnigramTokenizer(
                vocab=[(p, s) for (p, s, kind) in vocab if kind in (1, 4)],
                unk_id=PAD_ID,
            )
        self.add_dummy_prefix = add_dummy_prefix
        self.max_length = int(max_length)

    # -- 构造 -----------------------------------------------------------------

    @classmethod
    def from_bytes(cls, data: bytes, *, max_length: int = 512) -> "Umt5Tokenizer":
        """从 ``spiece_model`` 字节建分词器 ✓（有 sentencepiece 就用它 ✓）。"""
        if available():
            import sentencepiece as sp

            processor = sp.SentencePieceProcessor()
            # ⚠️ 用 ``LoadFromSerializedProto`` ✓（``load`` 只吃**文件路径** ✗ —— 权限/临时文件都不必要 ✓）
            processor.load_from_serialized_proto(data)
            return cls(processor, add_dummy_prefix=_normalizer_add_dummy_prefix(data),
                       max_length=max_length)
        return cls(None, vocab=parse_sp_model(data),
                   add_dummy_prefix=_normalizer_add_dummy_prefix(data), max_length=max_length)

    @classmethod
    def from_file(cls, path: str | Path, *, max_length: int = 512) -> "Umt5Tokenizer":
        return cls.from_bytes(Path(path).read_bytes(), max_length=max_length)

    # -- 分词 -----------------------------------------------------------------

    @property
    def backend(self) -> str:
        return "sentencepiece" if self._sp is not None else "自研 unigram（无 nmt_nfkc 归一化）"

    def encode(self, text: str, *, max_length: int | None = None) -> tuple[list[int], list[int]]:
        """文本 ⇒ ``(ids, mask)`` ✓（``mask`` 1=有效 0=pad ✓，长度固定 ``max_length`` ✓）。"""
        limit = int(max_length or self.max_length)
        text = (text or "").strip()
        if self._sp is not None:
            ids = list(self._sp.encode(text, out_type=int))
        else:
            assert self._own is not None
            # ⚠️ 空格 → ``▁`` ＋ 句首补 ``▁`` ✓ —— sentencepiece 主路由 ``normalizer`` 自己做 ✓，
            # 自研路只能**手工复现**这两步 ✓（其余归一化确实没有 ✗，见 ``to_dict()`` 的 ``knownGap``）
            normalized = text.replace(" ", "▁")
            if self.add_dummy_prefix:
                normalized = "▁" + normalized
            ids = list(self._own.encode(normalized))
        ids = ids[: max(0, limit - 1)] + [EOS_ID]          # 尾部**一定**是 </s> ✓
        mask = [1] * len(ids) + [0] * (limit - len(ids))
        ids = ids + [PAD_ID] * (limit - len(ids))
        return ids, mask

    def decode(self, ids: list[int]) -> str:
        if self._sp is not None:
            return self._sp.decode([int(i) for i in ids])
        assert self._own is not None
        return self._own.decode([int(i) for i in ids]).replace("▁", " ").strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "umt5",
            "backend": self.backend,
            # ⚠️ 这是 **spiece 里的 piece 数** ✓（umt5 是 256000 ✓）—— 权重 ``shared.weight`` 有
            # **256384** 行 ✓（尾部 384 行是 padding ✓），两个数**本来就不同** ✓ 别当缺陷 ✗
            "spiecePieces": (self._sp.get_piece_size() if self._sp is not None else len(self._vocab or [])),
            "padTokenId": PAD_ID,
            "endTokenId": EOS_ID,
            "addDummyPrefix": self.add_dummy_prefix,
            "maxLength": self.max_length,
            "knownGap": ("" if self._sp is not None else
                         "自研路没有 nmt_nfkc 字符归一化（全角/兼容字符不等价），"
                         "ASCII 提示词等价；要紧的出图请装 sentencepiece"),
        }


def load_tokenizer(*, spiece_bytes: bytes | None = None, path: str | Path | None = None,
                   max_length: int = 512) -> Umt5Tokenizer:
    """按「权重内嵌 → 盘上文件」顺序取分词器 ✓（两处都没有就明说 ✗）。"""
    if spiece_bytes:
        return Umt5Tokenizer.from_bytes(spiece_bytes, max_length=max_length)
    if path is not None and Path(path).exists():
        return Umt5Tokenizer.from_file(path, max_length=max_length)
    raise FileNotFoundError(
        "找不到 UMT5 的分词器 ✗：权重里没有 `spiece_model` ✓、也没给 `--tokenizer` 路径 ✗ "
        "（Comfy 内置那份在 `comfy/text_encoders/t5_pile_tokenizer/tokenizer.model` ✓）")
