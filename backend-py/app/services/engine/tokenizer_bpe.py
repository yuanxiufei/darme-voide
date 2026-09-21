r"""**自研 BPE 分词器**（纯 Python ✓ 零依赖 ✓ 离线 ✓）—— 文本 → token id，**本仓自己实现** ✗。

## 为什么要有它（2026-09-20；用户红线：所有功能自己实现，不调外部 ✗）

在此之前，:mod:`.text_encoder` 的 tokenizer 只有两条路 ✗：

* :class:`~app.services.engine.text_encoder.StubTokenizer` —— **确定性哈希**，与任何真词表无关 ✗
  （只验管道通不通 ✓ 语义结论一个都不能下 ✗）；
* :class:`~app.services.engine.text_encoder.HFTokenizer` —— 要装**外部** `transformers` ✗
  （违反「不对外依赖」✓✗）。

⇒ 本模块补上**第三条路** ✓：直接读**随权重一起来的词表文件**（HF ``tokenizer.json`` ✓
或经典的 ``vocab.json`` + ``merges.txt`` ✓），在**本仓**把 BPE 跑完 ✓ ——
零依赖 ✓ 离线 ✓ 不需要 ``transformers`` ✗。

## 事实来源（格式是公开事实 ✓，代码是自己写的 ✓）

* **byte ↔ unicode 映射**：GPT-2 的那张 256 项表 ✓ —— 本模块**按算法生成** ✓
  （``_bytes_to_unicode`` ✓）而不是抄 256 行常量 ✗（抄常量迟早漏一项且不报错 ✓✗）；
* **预分词规则**：GPT-2/`ByteLevel` 的那条正则
  ``'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`` ✓
  —— ⚠️ **标准库 `re` 不支持 `\p{L}`/`\p{N}`** ✗ 而本仓**不引入 `regex` 依赖** ✗
  ⇒ 这条规则用**自己写的扫描器**实现 ✓（字符分类走 ``unicodedata.category`` ✓，
  ``L`` = 字母 ✓、``N`` = 数字 ✓），并在自检里对 **ASCII 子集**与**等价正则**交叉核对 ✓；
* **合并顺序**：按 ``merges`` 里的**排名**取最小的那对 ✓，并**一次合并该对的全部出现** ✓
  （与 HF 的 BPE 一致 ✓），不是"贪心从左到右"✗。

## 明确的边界（不支持的形态**报错**，不装作能跑 ✗）

* ``model.type`` 必须是 ``BPE`` ✓（``Unigram``/``WordPiece`` 等 ⇒ 报错 ✓）；
* ``pre_tokenizer`` 必须是 ``ByteLevel``（含 ``Sequence`` 里含它的情形 ✓）——
  ⚠️ **Metaspace**（SentencePiece 风格，如 Llama 系 ✗）/ ``Whitespace`` / ``BertPreTokenizer``
  ⇒ **明确报错** ✓（这些形态的切分规则不同，硬套会**静默算出错的 id** ✓✗）；
* ``post_processor`` 只认 ``TemplateProcessing`` ✓（不认识时**记进 `notes`** ✓ 并**不加**特殊符 ✗
  —— 不静默假装加过 ✓）。

## 不变量（自检钉住的）

``decode(encode(text)) == text`` ✓（**往返恒等** ✓ —— 这是本模块最强的自检 ✓；因为 256 个字节
都被映射 ✓，所以**任何** UTF-8 文本都编得出、解不回就是 bug ✗，不存在 "UNK 吃掉" 的借口 ✓）。
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

__all__ = ["BpeTokenizer", "bytes_to_unicode", "load_tokenizer", "pretokenize"]

#: GPT-2 收缩词后缀 ✓（**大小写敏感** ✓ —— 与原正则一致 ✓：``DON'T`` 不会被切成 ``'T`` ✓）
_CONTRACTIONS: tuple[str, ...] = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 的 **byte → unicode** 表 ✓（256 项、双射 ✓）。

    构造规则（算法 ✓ 不抄表 ✗）：可打印区间原样映射 ✓，其余字节按顺序补到 256+ ✓
    ⇒ 保证映射结果都是「可见、不含空白」的单字符 ✓（这样字节级 BPE 才能在文本层做 ✓）。
    """
    printable = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("¡"), ord("¬") + 1))
                 + list(range(ord("®"), ord("ÿ") + 1)))
    table = list(printable)
    extra = list(printable)
    counter = 0
    for byte in range(2 ** 8):
        if byte not in table:
            table.append(byte)
            extra.append(2 ** 8 + counter)
            counter += 1
    return {byte: chr(code) for byte, code in zip(table, extra)}


def _classify(char: str) -> str:
    """字符属于字母 / 数字 / 其它 ✓（对应 ``\\p{L}`` / ``\\p{N}`` / ``[^\\s\\p{L}\\p{N}]`` ✓）。"""
    if char.isspace():
        return "space"
    kind = unicodedata.category(char)[0]
    if kind == "L":
        return "letter"
    if kind == "N":
        return "number"
    return "other"


def pretokenize(text: str) -> list[str]:
    """按 GPT-2/``ByteLevel`` 规则**预分词** ✓（自写扫描器 ✓ —— 见模块注释里为什么不用正则 ✗）。

    规则（按优先级 ✓，与原正则同序 ✓）：

    1. 收缩词后缀 ``'s 't 're 've 'm 'll 'd``（**大小写敏感** ✓）；
    2. `` ?`` + **字母**串 / **数字**串 / **其它非空白**串 ✓（那个可选空格**并进**本 token ✓）；
    3. 空白串 ✓（⚠️ 末尾留一个空格给**下一段** ✓ —— 这就是正则里 ``\\s+(?!\\S)`` 的回溯效果 ✓）。
    """
    pieces: list[str] = []
    size = len(text)
    index = 0
    while index < size:
        matched = None
        for suffix in _CONTRACTIONS:                      # ① 收缩词 ✓（先 3 字符再 2 字符 ✓）
            if text.startswith(suffix, index):
                matched = suffix
                break
        if matched is not None:
            pieces.append(matched)
            index += len(matched)
            continue

        cursor = index
        if text[cursor] == " " and cursor + 1 < size:     # ② ` ?` 前缀 ✓
            cursor += 1
        if cursor < size:
            kind = _classify(text[cursor])
            if kind in ("letter", "number", "other"):
                end = cursor
                while end < size and _classify(text[end]) == kind:
                    end += 1
                pieces.append(text[index:end])
                index = end
                continue

        end = index                                     # ③ 空白串 ✓
        while end < size and text[end].isspace():
            end += 1
        # ⚠️⚠️ 两条分支的差别就是正则里 `\s+(?!\S)` 与 `\s+` 的顺序 ✓：
        #    * 空白串**长度 > 1** 且后面还有非空白 ⇒ 末尾那个空白**留给下一段**当 ` ?` 前缀 ✓；
        #    * 长度 == 1 时 `\s+(?!\S)` 没法通过（缩短到 0 不行 ✗）⇒ 落到最后的 `\s+` ⇒ **整段吃掉** ✓。
        #    ⚠️ 第一版把"留给下一段"写成无条件 ✓✗ ⇒ 单字符空白（如 `a\tb` 里的 `\t`）
        #    会 append 空串且 `index = end - 1 == index` ⇒ **死循环** ✓✗（自检当场挂住 ✓）。
        if end < size and end - index > 1:
            pieces.append(text[index:end - 1])
            index = end - 1
        else:
            pieces.append(text[index:end])
            index = end
    return [piece for piece in pieces if piece]


class BpeTokenizer:
    """**字节级 BPE** ✓（`Tokenizer` 协议 ✓：只要有 ``vocab_size`` 与 ``encode`` ✓ 就能被注入 ✓）。

    ``name = "bpe"`` ✓（后端 ``describe().tokenizer`` 会把它报出来 ✓ —— 一眼看得出
    用的是**真词表** ✓ 还是 `stub` 假桩 ✗）。
    """

    name = "bpe"

    def __init__(self, vocab: dict[str, int], merges: Iterable[Sequence[str]], *,
                 added_tokens: dict[str, int] | None = None,
                 bos_ids: Sequence[int] = (), eos_ids: Sequence[int] = (),
                 notes: Sequence[str] = ()) -> None:
        if not vocab:
            raise ValueError("词表是空的 ✗")
        self.vocab: dict[str, int] = {str(token): int(index) for token, index in vocab.items()}
        self._byte_encoder = bytes_to_unicode()
        self._byte_decoder = {char: byte for byte, char in self._byte_encoder.items()}
        self._rank: dict[tuple[str, str], int] = {}
        merged_pairs = 0
        for order, pair in enumerate(merges):
            if isinstance(pair, str):                       # `tokenizer.json` 里是 "a b" ✓
                left, _, right = pair.partition(" ")
            else:                                           # 新格式是 [a, b] ✓
                left, right = str(pair[0]), str(pair[1])
            if left and right:                              # 表头/空行 ⇒ 跳过 ✓（`#version` 那行 ✓）
                self._rank[(left, right)] = order
                merged_pairs += 1
        self._added: dict[str, int] = {str(k): int(v) for k, v in (added_tokens or {}).items()}
        self._bos = tuple(int(item) for item in bos_ids)
        self._eos = tuple(int(item) for item in eos_ids)
        self.merge_count = merged_pairs
        self.notes: list[str] = list(notes)

    # ── 协议面 ✓
    @property
    def vocab_size(self) -> int:
        """词表条目数 ✓（与 HF 的 `tokenizer.vocab_size` 同口径 ✗ —— **不含** added tokens ✓）。"""
        return len(self.vocab)

    @property
    def required_vocab_size(self) -> int:
        """**要能装下这个分词器，嵌入表至少得多大** ✓（= 最大 id + 1 ✓，含特殊符 ✓）。

        ⚠️ 为什么要单独给 ✗：`vocab_size` 不含 added tokens ✓ ⇒ 拿它去建
        ``nn.Embedding`` 会在**特殊符 id 上越界** ✓✗（而且往往到真跑才炸 ✓）。
        """
        ids = list(self.vocab.values()) + list(self._added.values())
        return (max(ids) + 1) if ids else 0

    def describe(self) -> dict[str, Any]:
        """自述 ✓（词表从哪来、多少 token、多少条 merge、有什么保留说明 ✓）。"""
        return {"name": self.name, "vocabSize": self.vocab_size, "merges": self.merge_count,
                "addedTokens": len(self._added), "byteLevel": True,
                "bosIds": list(self._bos), "eosIds": list(self._eos), "notes": list(self.notes)}

    # ── 结构对比 ✓（自检/装载报告要用它回答「换词表了吗」✓）
    def fingerprint(self) -> dict[str, Any]:
        return {"vocabSize": self.vocab_size, "merges": self.merge_count,
                "addedTokens": sorted(self._added)}

    # ── 核心：文本 → id ✓
    def _split_specials(self, text: str) -> Iterator[tuple[str, str]]:
        """把文本切成「普通段」与「特殊 token」✓（**最长匹配优先** ✓ —— 否则 ``<|a|>`` 会被 ``<|`` 抢走 ✓✗）。"""
        if not self._added:
            yield "text", text
            return
        wanted = sorted(self._added, key=len, reverse=True)   # ⚠️ 长匹配优先 ✓
        buffer: list[str] = []
        index = 0
        while index < len(text):
            hit = None
            for token in wanted:
                if text.startswith(token, index):
                    hit = token
                    break
            if hit is None:
                buffer.append(text[index])
                index += 1
                continue
            if buffer:
                yield "text", "".join(buffer)
                buffer = []
            yield "special", hit
            index += len(hit)
        if buffer:
            yield "text", "".join(buffer)

    def _bpe(self, piece: str) -> list[str]:
        """一段预分词 → token 串 ✓（按 ``merges`` 的**排名**逐轮合并**全部出现** ✓）。"""
        word = [self._byte_encoder[byte] for byte in piece.encode("utf-8")]
        if len(word) > 1:
            while True:
                best: tuple[str, str] | None = None
                best_rank = None
                for left, right in zip(word, word[1:]):
                    rank = self._rank.get((left, right))
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best, best_rank = (left, right), rank
                if best is None:
                    break
                merged: list[str] = []
                index = 0
                while index < len(word):
                    if (index + 1 < len(word) and word[index] == best[0]
                            and word[index + 1] == best[1]):
                        merged.append(best[0] + best[1])
                        index += 2
                    else:
                        merged.append(word[index])
                        index += 1
                word = merged
                if len(word) == 1:
                    break
        for token in word:
            if token not in self.vocab:
                raise ValueError(
                    f"合并结果 `{token}` 不在词表里 ✗ ⇒ 词表与 `merges` 不是同一份 ✓"
                    f"（不静默丢 token ✗ —— 那会算出一串**看着正常**的错 id ✓✗）")
        return word

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """文本 → id ✓（**特殊 token 原子化** ✓；普通段走预分词 + BPE ✓）。"""
        ids: list[int] = list(self._bos) if add_special_tokens else []
        for kind, chunk in self._split_specials(str(text or "")):
            if kind == "special":
                ids.append(self._added[chunk])
                continue
            for piece in pretokenize(chunk):
                ids.extend(self.vocab[token] for token in self._bpe(piece))
        if add_special_tokens:
            ids.extend(self._eos)
        return ids

    # ── 反向：id → 文本 ✓（往返恒等是自检用的强不变量 ✓）
    def _id_index(self) -> dict[int, str]:
        index: dict[int, str] = {}
        for token, token_id in self.vocab.items():
            index.setdefault(token_id, token)
        for token, token_id in self._added.items():
            index.setdefault(token_id, token)
        return index

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """id → 文本 ✓（字节反查后按 UTF-8 还原 ✓）。"""
        index = self._id_index()
        specials = set(self._added)
        chars: list[str] = []
        for token_id in ids:
            token = index.get(int(token_id))
            if token is None:
                raise ValueError(f"id {token_id} 不在词表里 ✗（不静默跳过 ✗）")
            if skip_special_tokens and token in specials:
                continue
            chars.append(token)
        raw = bytearray()
        # ⚠️⚠️ 要按**映射后的字符**逐个反查 ✓ —— **不能按 token 反查** ✗：
        #    合并出来的 token（如 `hello` ✓）本身就是**多个映射字符的拼接** ✓
        #    （第一版按 token 查 ⇒ 报「字符 `he` 不是 byte-level 映射的结果」✓✗，往返恒等当场红 ✓）。
        for char in "".join(chars):
            byte = self._byte_decoder.get(char)
            if byte is None:
                raise ValueError(f"字符 `{char}` 不是 byte-level 映射的结果 ✗ ⇒ 这份词表不是 byte-BPE ✓")
            raw.append(byte)
        return bytes(raw).decode("utf-8", errors="replace")

    # ── 三种装载方式 ✓
    @classmethod
    def from_vocab_and_merges(cls, vocab_path: str | Path, merges_path: str | Path) -> "BpeTokenizer":
        """经典格式 ✓：``vocab.json``（token → id ✓）+ ``merges.txt``（**首行是版本注释** ✓ 要跳过 ✓）。"""
        vocab_file, merges_file = Path(vocab_path), Path(merges_path)
        for path in (vocab_file, merges_file):
            if not path.exists():
                raise FileNotFoundError(f"词表文件不存在：{path} ✗")
        vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
        if not isinstance(vocab, dict):
            raise ValueError(f"{vocab_file} 不是 token→id 的字典 ✗（收到 {type(vocab).__name__} ✗）")
        lines = merges_file.read_text(encoding="utf-8").splitlines()
        merges = [line for line in lines if line and not line.startswith("#version")]
        return cls(vocab, merges, notes=[f"来自经典格式：{vocab_file.name} + {merges_file.name} ✓"])

    @classmethod
    def from_tokenizer_json(cls, path: str | Path) -> "BpeTokenizer":
        """HF ``tokenizer.json`` ✓（只认 ``BPE`` + ``ByteLevel`` ✓ —— 别的形态**明确报错** ✓）。"""
        json_path = Path(path)
        if not json_path.exists():
            raise FileNotFoundError(f"tokenizer.json 不存在：{json_path} ✗")
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            raise ValueError(f"{json_path} 不是合法 JSON ✗（{err} ✓）") from err
        model = payload.get("model") or {}
        model_type = str(model.get("type") or "")
        if model_type != "BPE":
            raise ValueError(
                f"`model.type` 是 `{model_type or '空'}` ✗ —— 本模块只实现了 **BPE** ✓"
                f"（`Unigram` / `WordPiece` 的切分规则不同 ✓ ⇒ 硬套会**静默算出错的 id** ✓✗）")
        vocab = model.get("vocab") or {}
        merges = model.get("merges") or []
        added: dict[str, int] = {}
        for item in payload.get("added_tokens") or []:
            content = item.get("content")
            if content is not None and item.get("id") is not None:
                added[str(content)] = int(item["id"])
        notes: list[str] = [f"来自 `{json_path.name}` ✓"]
        notes.extend(_check_pre_tokenizer(payload.get("pre_tokenizer")))
        bos, eos, post_note = _post_processor_specials(payload.get("post_processor"), added)
        if post_note:
            notes.append(post_note)
        tokenizer = cls(vocab, merges, added_tokens=added, bos_ids=bos, eos_ids=eos, notes=notes)
        tokenizer.source_path = str(json_path)
        return tokenizer


def _check_pre_tokenizer(spec: Any) -> list[str]:
    """``pre_tokenizer`` 必须是 ``ByteLevel`` ✓（``Sequence`` 里含它也行 ✓）—— 否则**报错** ✓。"""
    node = spec
    if isinstance(node, dict) and str(node.get("type")) == "Sequence":
        inner = node.get("pretokenizers") or []
        if len(inner) == 1:
            node = inner[0]
    if node is None:
        return ["没有 `pre_tokenizer` 段 ✓ ⇒ 按 ByteLevel 规则切分 ✓（BPE 字节模型的默认 ✓）"]
    node_type = str((node or {}).get("type") or "")
    if node_type == "ByteLevel":
        return ["`pre_tokenizer` = ByteLevel ✓（本模块实现的正是它 ✓）"]
    raise ValueError(
        f"`pre_tokenizer.type` 是 `{node_type or '空'}` ✗ —— 本模块只实现了 **ByteLevel** ✓；"
        f"`Metaspace`（SentencePiece 风格 ✗）/ `Whitespace` / `BertPreTokenizer` 等的切分规则不同 ✓ "
        f"⇒ **明确报错** ✓（不硬套 ✓ —— 硬套会静默算出错的 id ✓✗）")


def _post_processor_specials(spec: Any, added: dict[str, int]
                             ) -> tuple[tuple[int, ...], tuple[int, ...], str | None]:
    """从 ``post_processor`` 里取 ``$A`` **前面/后面**的特殊 token ✓（只认 ``TemplateProcessing`` ✓）。"""
    if spec is None:
        return (), (), "没有 `post_processor` ✓ ⇒ 不加 BOS/EOS ✓（显式调用可按需加 ✓）"
    spec_type = str((spec or {}).get("type") or "")
    if spec_type != "TemplateProcessing":
        return (), (), (f"`post_processor.type` = `{spec_type}` ✗ ⇒ **不加**特殊符 ✓"
                        f"（记录在此 ✓ —— 不静默假装加过 ✓）")
    single = (spec or {}).get("single") or []
    bos: list[int] = []
    eos: list[int] = []
    seen_sequence = False
    for item in single:
        if "Sequence" in item:
            seen_sequence = True
            continue
        special = (item.get("SpecialToken") or {})
        token_id = special.get("id")
        if token_id is None:
            continue
        (eos if seen_sequence else bos).append(int(token_id))
    return tuple(bos), tuple(eos), None


def load_tokenizer(path: str | Path) -> BpeTokenizer:
    """按**目录或文件**装载 ✓（自动嗅探 ✓：``tokenizer.json`` ✓ 或 ``vocab.json``+``merges.txt`` ✓）。

    ⚠️ 嗅探不到 ⇒ **报错并说清找过什么** ✗（不静默回落到 `StubTokenizer` ✓✗ ——
    那会让"用了假词表"这件事**看不出来** ✓）。
    """
    target = Path(path)
    if target.is_file():
        if target.name.endswith(".json"):
            return BpeTokenizer.from_tokenizer_json(target)
        raise ValueError(f"看不懂的文件：{target.name} ✗（要 `tokenizer.json` 或走 `vocab.json`+`merges.txt` ✓）")
    if not target.exists():
        raise FileNotFoundError(f"路径不存在：{target} ✗")
    candidates = [target / "tokenizer.json", target / "vocab.json", target / "merges.txt"]
    tokenizer_json = target / "tokenizer.json"
    if tokenizer_json.exists():
        return BpeTokenizer.from_tokenizer_json(tokenizer_json)
    if (target / "vocab.json").exists() and (target / "merges.txt").exists():
        return BpeTokenizer.from_vocab_and_merges(target / "vocab.json", target / "merges.txt")
    raise FileNotFoundError(
        f"{target} 里没有词表文件 ✗（找过：{[item.name for item in candidates]} ✓）")
