"""S25 自检：**自研 BPE 分词器**（零依赖 ✓ 离线 ✓ 2026-09-20）。

背景：tokenizer 此前只有「哈希假桩 `StubTokenizer`」✗ 与「要装外部 `transformers` 的
`HFTokenizer`」✗ 两条路 —— 按用户红线（所有功能自己实现、不调外部 ✓）这正是**能力外包** ✗。
本套验证新 `engine/tokenizer_bpe.py`：真词表文件（`tokenizer.json` / `vocab.json`+`merges.txt` ✓）
在本仓跑完字节级 BPE ✓。

强不变量（**能自己算出来的** ✓）：``decode(encode(text)) == text`` ✓ —— 256 个字节全在映射里 ✓
⇒ **任何** UTF-8 文本都编得出 ✓，解不回就是 bug ✗（没有「UNK 吃掉」的借口 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_tokenizer_test.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import text_encoder as te  # noqa: E402
from app.services.engine import tokenizer_bpe as tb  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _raises(call: Any, needle: str | None = None) -> str | None:
    """**能触发**的反向证明 ✓：调它、看报错里有没有那个词 ✓（没报错 ⇒ None ⇒ 断言红 ✓）。"""
    try:
        call()
    except Exception as err:  # noqa: BLE001 —— 就是来看它报什么的 ✓
        return str(err) if needle is None or needle in str(err) else None
    return None


# ══════════════════════════════════════════════════════════════════════════
# 小词表夹具：**自己造** ✓（真词表随权重来 ✓ —— 本机没有 ✗ ⇒ 不依赖它 ✓）
# ══════════════════════════════════════════════════════════════════════════
def _byte_vocab() -> dict[str, int]:
    """256 个单字节 token ✓（真词表**必须**含全 256 ✓ ⇒ 任何文本都编得出 ✓）。"""
    return {char: index for index, char in enumerate(sorted(tb.bytes_to_unicode().values()))}


#: ⚠️ **空格在 byte 映射里不是 `" "`** ✓ ⇒ 词表/merges 里要写映射后的那个字符 ✓
#: （GPT-2 是 `Ġ` ✓、本模块是 `chr(256+0)` ✓）—— 真词表文件里也是这么写的 ✓（别手写成空格 ✗）。
SPACE = tb.bytes_to_unicode()[32]


def tiny_tokenizer(*, with_specials: bool = False) -> tb.BpeTokenizer:
    """小词表 ✓：单字节 + 几条**故意能合并**的规则 ✓（这样「合并真的发生了」可验 ✓）。

    ⚠️ 词表必须含**每一条合并的产物** ✓（`ll` 也在内 ✓）—— 否则 `_bpe` 会按设计**报错** ✓
    （本套第 ⑧ 条正是拿这个当反向证明 ✓）。
    """
    vocab = _byte_vocab()
    extra = 300
    for token in ("he", "ll", "hel", "hell", "hello",
                  f"{SPACE}w", f"{SPACE}wo", f"{SPACE}wor", f"{SPACE}worl", f"{SPACE}world",
                  "he he"):
        vocab[token] = extra
        extra += 1
    # ⚠️ 合并链**必须自洽** ✗：`he l`（要求词中间存在 `l` ✓）在 `l l` 已经先把两个 l 并成 `ll` 之后就
    #    **再也合不上** ✓✗ ⇒ `hello` 会停在 `he|ll|o`（第一版就踩了 ✓，第 ④ 条当场红 ✓）。
    #    正确写法是按**字面相邻**排：`h e` → `l l` → `he ll` → `hell o` ✓。
    merges = ["h e", "l l", "he ll", "hell o",
              f"{SPACE} w", f"{SPACE}w o", f"{SPACE}wo r", f"{SPACE}wor l", f"{SPACE}worl d"]
    added = {"<|endoftext|>": 999, "<|pad|>": 998} if with_specials else None
    return tb.BpeTokenizer(vocab, merges, added_tokens=added,
                           bos_ids=(999,) if with_specials else (),
                           eos_ids=(998,) if with_specials else ())


def write_tokenizer_json(root: Path, *, pre_tokenizer: Any = None,
                         post_processor: Any = None, model_type: str = "BPE") -> Path:
    payload = {
        "model": {"type": model_type, "vocab": _byte_vocab(),
                  "merges": ["h e", "l l", "he l", "hell o"]},
        "added_tokens": [{"id": 999, "content": "<|endoftext|>", "special": True}],
    }
    if pre_tokenizer is not None:
        payload["pre_tokenizer"] = pre_tokenizer
    if post_processor is not None:
        payload["post_processor"] = post_processor
    root.mkdir(parents=True, exist_ok=True)
    path = root / "tokenizer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ══════════════════════════════════════════════════════════════════════════
# ① 字节映射与预分词（规则事实 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_byte_map() -> None:
    table = tb.bytes_to_unicode()
    check("① byte↔unicode：256 项 ✓ **双射** ✓ 且都可见（不含空白 ✓）",
          len(table) == 256 and len(set(table.values())) == 256
          and all(not char.isspace() for char in table.values()),
          (len(table), len(set(table.values()))))

    cases = {
        "Hello world": ["Hello", " world"],
        "don't": ["don", "'t"],
        "we're": ["we", "'re"],
        "DON'T": ["DON", "'", "T"],          # ⚠️ 收缩词**大小写敏感** ✓（与 GPT-2 原正则一致 ✓）
        "12.5": ["12", ".", "5"],
        "a  b": ["a", " ", " b"],
        "  b": [" ", " b"],
        "a ": ["a", " "],
        "a\tb": ["a", "\t", "b"],
        "中文测试": ["中文测试"],
    }
    wrong = {text: (tb.pretokenize(text), want) for text, want in cases.items()
             if tb.pretokenize(text) != want}
    check("② 预分词：收缩词 / 可选空格 / 字母·数字·其它分档 / 空白回溯 ✓（10 例全中 ✓）",
          not wrong, wrong)

    # ⭐ 交叉核对：ASCII 子集上，自写扫描器 == **等价正则** ✓（独立第二实现 ✓）
    pattern = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
    samples = ["Hello, world!", "don't stop  me", "12.5kg", "a  b   ", "  x", "A-1_B",
               "I'll be there", "3 4 5", "!!??", "x",
               # ⚠️ 下面这几例是**冲着手写扫描器的坑**去的 ✓（2026-09-20 实测踩过 ✓）：
               #    单字符空白后面还有非空白（上一版这里**死循环** ✓✗ —— 正则那边是 `\s+` 兜底 ✓）
               "a\tb", "a b", " \t x", "a\u00a0b"]
    mismatched = {sample: (tb.pretokenize(sample), pattern.findall(sample)) for sample in samples
                  if tb.pretokenize(sample) != pattern.findall(sample)}
    check("③ ⭐ **独立第二实现**交叉核对：自写扫描器 == 等价正则 ✓（ASCII 子集 ✓ 14 例 ✓"
          "含单字符空白/制表符/不换行空格 ✓）",
          not mismatched, mismatched)


# ══════════════════════════════════════════════════════════════════════════
# ② 编码：合并真的发生 ✓、特殊符原子化 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_encode() -> None:
    tokenizer = tiny_tokenizer()
    ids = tokenizer.encode("hello world", add_special_tokens=False)
    check("④ BPE 合并**真的发生**了 ✓：`hello world` ⇒ 2 个 id（不是 11 个字符 id ✗）"
          "且空格 token 是**映射后**的字符 ✓（不是半角空格 ✗）",
          len(ids) == 2
          and [tokenizer._id_index()[item] for item in ids] == ["hello", f"{SPACE}world"],  # noqa: SLF001
          [tokenizer._id_index()[item] for item in ids])                                    # noqa: SLF001
    check("④′ 空格在 byte 映射里**不是半角空格** ✓（`chr(256+0)` ✓ —— 真词表也用映射字符 ✓）",
          SPACE != " " and tb.bytes_to_unicode()[65] == "A", (repr(SPACE),))

    # ⚠️ 反向证明（能触发 ✓）：拿掉 merges ⇒ 同一个文本必然变长 ✓
    bare = tb.BpeTokenizer(_byte_vocab(), [])
    check("⑤ 反向证明：**没有 merges** 的同一份词表 ⇒ 同一文本 id 数**必变多** ✓"
          "（否则第 ④ 条就是恒真 ✓）",
          len(bare.encode("hello world", add_special_tokens=False)) > len(ids),
          (len(bare.encode("hello world", add_special_tokens=False)), len(ids)))

    specials = tiny_tokenizer(with_specials=True)
    with_special = specials.encode("hi<|endoftext|>there", add_special_tokens=False)
    check("⑥ 特殊 token **原子化** ✓（不被预分词切开 ✓）且 BOS/EOS 按 post_processor 加上 ✓",
          "<|endoftext|>" in [specials._id_index()[item] for item in with_special]      # noqa: SLF001
          and specials.encode("x")[0] == 999 and specials.encode("x")[-1] == 998,
          ([specials._id_index()[item] for item in with_special], specials.encode("x")))  # noqa: SLF001

    long_match = tiny_tokenizer(with_specials=True)
    long_match._added["<|endoftext|>x"] = 997                                              # noqa: SLF001
    ids_long = long_match.encode("<|endoftext|>x", add_special_tokens=False)
    check("⑦ 特殊 token **最长匹配优先** ✓（`<|endoftext|>x` 不会被 `<|endoftext|>` 抢走 ✓✗）",
          997 in ids_long, ids_long)

    unknown = tb.BpeTokenizer(_byte_vocab(), ["a b"])   # merges 推出了词表里没有的 token ⇒ 报错 ✓
    check("⑧ merges 与词表**不是同一份** ⇒ 报错 ✓（不静默丢 token ✗）",
          _raises(lambda: unknown.encode("ab"), "不在词表里") is not None, None)


# ══════════════════════════════════════════════════════════════════════════
# ③ ⭐ 往返恒等（本模块最强的不变量 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_roundtrip() -> None:
    tokenizer = tiny_tokenizer()
    samples = ["", "a", "hello world", "中文 · テスト · 한국어", "emoji 🎬🎞️ 混排",
               "  多空格   与\t制表\n换行  ", "don't we're I'll", "１２３ ①②③ ½",
               "mixed中英123 4.5%", "\u200b零宽字符", "ß", "😀😀"]
    broken = {text: tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))
              for text in samples
              if tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) != text}
    check("⑨ ⭐⭐ **往返恒等**：`decode(encode(text)) == text` ✓（12 例：ASCII ✓ CJK ✓ "
          "emoji ✓ 空白 ✓ 零宽 ✓ —— 256 字节全映射 ⇒ 不存在 UNK 借口 ✓）",
          not broken, broken)

    with_specials = tiny_tokenizer(with_specials=True)
    ids = with_specials.encode("hi")
    check("⑩ 特殊符跳过 ⇒ 文本恒等 ✓（BOS/EOS 不污染 decode ✓）",
          with_specials.decode(ids) == "hi"
          and with_specials.decode(ids, skip_special_tokens=False) == "<|endoftext|>hi<|pad|>",
          (with_specials.decode(ids), with_specials.decode(ids, skip_special_tokens=False)))

    check("⑪ 词表里**没有的 id** ⇒ 报错 ✓（不静默跳过 ✗）",
          _raises(lambda: tokenizer.decode([123456]), "不在词表里") is not None, None)


# ══════════════════════════════════════════════════════════════════════════
# ④ 装载：嗅探 / 报错 / 不支持的形态
# ══════════════════════════════════════════════════════════════════════════
def case_loading(root: Path) -> None:
    json_path = write_tokenizer_json(
        root, pre_tokenizer={"type": "ByteLevel"},
        post_processor={"type": "TemplateProcessing", "single": [
            {"SpecialToken": {"id": 999, "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
        ]})
    loaded = tb.load_tokenizer(json_path)
    check("⑫ `tokenizer.json` 装载 ✓：词表口径 ✓ + ByteLevel 记录 ✓ + post_processor 的 "
          "BOS 被识别 ✓（`<|endoftext|>` 在 `$A` 之前 ✓）且 `required_vocab_size` 覆盖特殊符 ✓",
          loaded.vocab_size == len(_byte_vocab())          # ⚠️ 与 HF 同口径：**不含** added tokens ✓
          and loaded.required_vocab_size == 1000           # 但嵌入表要覆盖 999 ⇒ 1000 ✓
          and loaded.describe()["byteLevel"]
          and loaded.describe()["bosIds"] == [999]
          and any("ByteLevel" in note for note in loaded.notes),
          loaded.describe())

    directory = root / "from_dir"
    directory.mkdir()
    (directory / "tokenizer.json").write_text(json_path.read_text(encoding="utf-8"),
                                              encoding="utf-8")
    check("⑬ 目录嗅探：优先 `tokenizer.json` ✓",
          tb.load_tokenizer(directory).vocab_size == loaded.vocab_size, None)

    classic = root / "classic"
    classic.mkdir()
    (classic / "vocab.json").write_text(json.dumps({**_byte_vocab(), "he": 300}),
                                        encoding="utf-8")
    (classic / "merges.txt").write_text("#version: 0.2\nh e\nl l\n", encoding="utf-8")
    from_classic = tb.load_tokenizer(classic)
    check("⑭ 经典格式 `vocab.json`+`merges.txt` ✓（**首行 `#version` 要跳过** ✓ 否则会多一条假 merge ✗）",
          from_classic.merge_count == 2
          and from_classic.encode("he", add_special_tokens=False) == [300],
          (from_classic.merge_count, from_classic.encode("he", add_special_tokens=False)))

    empty = root / "empty"
    empty.mkdir()
    check("⑮ 找不到词表 ⇒ **说清找过什么** ✓（不静默回落假桩 ✗）",
          _raises(lambda: tb.load_tokenizer(empty), "没有词表文件") is not None, None)

    metaspace = write_tokenizer_json(
        root / "meta", pre_tokenizer={"type": "Metaspace", "replacement": "▁"})
    check("⑯ `pre_tokenizer` 不是 ByteLevel（如 Metaspace ✗）⇒ **明确报错** ✓"
          "（硬套会静默算出错的 id ✓✗）",
          _raises(lambda: tb.load_tokenizer(metaspace), "只实现了 **ByteLevel**") is not None, None)

    unigram = write_tokenizer_json(root / "uni", model_type="Unigram")
    check("⑰ `model.type` 不是 BPE ⇒ 报错 ✓",
          _raises(lambda: tb.load_tokenizer(unigram), "只实现了 **BPE**") is not None, None)

    weird_post = write_tokenizer_json(
        root / "post", pre_tokenizer={"type": "ByteLevel"},
        post_processor={"type": "RobertaProcessing"})
    weird = tb.load_tokenizer(weird_post)
    check("⑱ 不认识的 `post_processor` ⇒ **记进 notes** ✓ 且**不加**特殊符 ✓"
          "（不静默假装加过 ✓）",
          weird.describe()["bosIds"] == [] and weird.describe()["eosIds"] == []
          and any("RobertaProcessing" in note for note in weird.notes), weird.notes)

    bad_json = root / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    check("⑲ 坏 JSON ⇒ 报错 ✓（不当作空词表继续 ✗）",
          _raises(lambda: tb.load_tokenizer(bad_json), "JSON") is not None, None)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 接进既有管道：本模块**满足 `Tokenizer` 协议** ✓（可注入 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_protocol() -> None:
    tokenizer = tiny_tokenizer()
    ids, original, truncated = te.tokenize_prompt(tokenizer, "hello world", max_length=16)
    check("⑳ 满足 `Tokenizer` 协议 ✓：`tokenize_prompt` 直接用 ✓（`vocab_size` + `encode` ✓）",
          ids and original == len(ids) and truncated is False, (ids, original, truncated))

    cut, original, truncated = te.tokenize_prompt(tokenizer, "hello world hello world",
                                                  max_length=3)
    check("㉑ 截断**如实回报** ✓（`originalTokens` + `truncated` 给调用方 ✓ 不静默切 ✗）",
          len(cut) == 3 and original > 3 and truncated is True, (cut, original, truncated))

    check("㉒ `name = bpe` ✓ ⇒ 后端 `describe().tokenizer` 一眼看出用的是**真词表** ✓ 不是 `stub` ✗",
          tokenizer.name == "bpe" and tokenizer.describe()["vocabSize"] == tokenizer.vocab_size,
          tokenizer.describe())


# ══════════════════════════════════════════════════════════════════════════
# ⑥ ⭐ 与**参考实现**（HF `tokenizers` ✓ Rust 那份 ✓）可执行核对
# ══════════════════════════════════════════════════════════════════════════
def case_reference(root: Path) -> None:
    """⭐ 装了 `tokenizers` ⇒ 用**同一份词表**逐例比 id ✓；没装 ⇒ SKIP ✓（不是失败 ✗）。

    ⚠️ 这条 = 「**独立第二实现**」级判据 ✓：自写扫描器 + 自写 BPE 只要有一处与
    权威实现不一致（预分词边界 ✓ / 合并顺序 ✓ / 空格映射 ✓）就会**逐例显出差异** ✓。
    """
    try:
        from tokenizers import Tokenizer as ReferenceTokenizer  # noqa: PLC0415
        from tokenizers import models as ref_models  # noqa: PLC0415
        from tokenizers import pre_tokenizers as ref_pre  # noqa: PLC0415
    except ImportError:
        check("㉓ 参考实现核对：本机没装 `tokenizers` ⇒ **跳过** ✓（不是失败 ✗；"
              "自研 BPE 不需要它 ✓）", True, "skipped")
        return

    ours = tiny_tokenizer()
    # ⚠️ `merges` 必须按**排名顺序**交给参考实现 ✓（`_rank` 是「对 → 名次」的字典 ✓ ⇒ 要排序 ✓）
    ordered = sorted(ours._rank.items(), key=lambda item: item[1])                 # noqa: SLF001
    merges = [(pair[0], pair[1]) for pair, _ in ordered]

    def build(*, prefix_space: bool) -> Any:
        tokenizer = ReferenceTokenizer(ref_models.BPE(vocab=dict(ours.vocab), merges=list(merges)))
        tokenizer.pre_tokenizer = ref_pre.ByteLevel(add_prefix_space=prefix_space)
        return tokenizer

    reference = build(prefix_space=False)
    samples = ["", "hello world", "hello", "don't stop", "Hello, World! 12.5",
               "中文测试 emoji 🎬", "a\tb  c", "  leading", "trailing  ", "!!??"]
    mismatched = {}
    for text in samples:
        theirs = list(reference.encode(text).ids)
        mine = ours.encode(text, add_special_tokens=False)
        if theirs != mine:
            mismatched[text] = (mine, theirs)
    check("㉓ ⭐⭐ 与参考实现 `tokenizers`（Rust ✓）**逐例同 id** ✓"
          "（10 例含 CJK ✓ emoji ✓ 制表符 ✓ 前后空白 ✓ 标点 ✓ —— 预分词 / 合并顺序 / 空格映射"
          "**任何一处不一致都会显形** ✓）",
          not mismatched, mismatched)

    # ⚠️ 反向证明（**能触发** ✓ —— 不是拿同一配置自比 ✗）：参考实现换一种**合法**配置 ⇒ id 必变 ✓
    spaced = build(prefix_space=True)
    probe = "hello"
    check("㉔ 反向证明：同一份参考实现换 `add_prefix_space=True` ⇒ 同一文本 id **必不同** ✓"
          "（⇒ 第 ㉓ 条比的是真行为 ✓ 不是恒真 ✓）",
          list(spaced.encode(probe).ids) != ours.encode(probe, add_special_tokens=False),
          (list(spaced.encode(probe).ids), ours.encode(probe, add_special_tokens=False)))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_byte_map()
        case_encode()
        case_roundtrip()
        case_loading(root)
        case_protocol()
        case_reference(root)
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      ↳ {detail}"))
    # ⚠️ 汇总行**必须是 `SUMMARY: n/m passed`** ✓ —— `run_all.py` 按这个前缀收敛项数 ✓✗
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed"
          + (" ✗✗✗" if failed else " ✓"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
