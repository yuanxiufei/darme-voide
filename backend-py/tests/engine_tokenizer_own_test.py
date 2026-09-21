"""S28 自检：**自研 `Unigram` / `WordPiece` / `Metaspace`**（2026-09-20）。

动机：`tokenizer_hub` 的「形态覆盖」此前是 **自研 BPE + 回退参考实现** ✗ ⇒ 没装
`transformers` 时 `Unigram`（Llama/Qwen/T5 系 ✓）与 `WordPiece`（BERT 系 ✓）**编不出 token** ✗。
本套验证 `engine/tokenizer_own.py` 把这两大血统**自己实现** ✓。

⚠️ 判据只有一条够用：**与参考实现逐例同 id** ✓（`tokenizers` Rust 那份 ✓ 本机已装 ✓）。
缺参考实现时按「两种世界都成立」降级为**结构断言** ✓（不写"真机必然有"✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_tokenizer_own_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import tokenizer_own as own  # noqa: E402

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


def reference_module() -> Any | None:
    try:
        import tokenizers  # noqa: PLC0415

        return tokenizers
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════════
# 夹具：用**参考实现**程序化造词表 ✓（不需要联网 ✓）+ 存成 `tokenizer.json` ✓
# ══════════════════════════════════════════════════════════════════════════
def write_unigram_metaspace(root: Path) -> Path | None:
    tokenizers = reference_module()
    if tokenizers is None:
        return None
    from tokenizers import Tokenizer  # noqa: PLC0415
    from tokenizers import models, pre_tokenizers  # noqa: PLC0415

    target = root / "uni_meta"
    target.mkdir(parents=True, exist_ok=True)
    vocab = [("<unk>", 0.0), ("<s>", 0.0), ("</s>", 0.0),
             ("▁hello", -1.0), ("▁world", -1.2), ("▁he", -2.0), ("llo", -2.5),
             ("▁wor", -2.2), ("ld", -2.4), ("▁h", -3.0), ("e", -3.1), ("l", -3.2),
             ("o", -3.3), ("▁", -3.4), ("中", -4.0), ("文", -4.1)]
    tokenizer = Tokenizer(models.Unigram(vocab, unk_id=0))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁")
    tokenizer.save(str(target / "tokenizer.json"))
    return target


def write_wordpiece(root: Path) -> Path | None:
    tokenizers = reference_module()
    if tokenizers is None:
        return None
    from tokenizers import Tokenizer  # noqa: PLC0415
    from tokenizers import models, pre_tokenizers  # noqa: PLC0415

    target = root / "wordpiece"
    target.mkdir(parents=True, exist_ok=True)
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3, "world": 4,
             "he": 5, "##llo": 6, "wor": 7, "##ld": 8, "a": 9, "b": 10, "##c": 11,
             "中": 12, "文": 13}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.save(str(target / "tokenizer.json"))
    return target


def write_with_normalizer(root: Path) -> Path:
    """带**已实现** `normalizer` 的词表 ✓（手写 JSON ✓ 不需要参考实现 ✓）。"""
    target = root / "with_norm"
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {"type": "WordPiece", "vocab": {"[UNK]": 0, "hello": 1},
                  "unk_token": "[UNK]"},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "normalizer": {"type": "BertNormalizer", "lowercase": True},
    }
    (target / "tokenizer.json").write_text(json.dumps(payload), encoding="utf-8")
    return target


# ══════════════════════════════════════════════════════════════════════════
# ① 覆盖判定：什么时候走自研 / 什么时候必须拒绝
# ══════════════════════════════════════════════════════════════════════════
def case_support(root: Path) -> None:
    cases = {
        "BPE+ByteLevel": {"modelType": "BPE", "preTokenizer": "ByteLevel"},
        "Unigram+Metaspace": {"modelType": "Unigram", "preTokenizer": "Metaspace"},
        "WordPiece+BERT": {"modelType": "WordPiece", "preTokenizer": "BertPreTokenizer"},
        # ⚠️ 规则型预分词器也已经自研 ✓（2026-09-21 ✓）⇒ 别再拿 `Digits` 当"不支持"样本 ✗
        #    （「判据跟着能力走」✓ —— 同一个坑这轮已经踩过两次 ✓✗）
        "WordPiece+Digits": {"modelType": "WordPiece", "preTokenizer": "Digits"},
        "Unigram+Whitespace": {"modelType": "Unigram", "preTokenizer": "Whitespace"},
        "还没做的预设分词": {"modelType": "WordPiece", "preTokenizer": "Split"},
        "不认识的模型": {"modelType": "UnigramFake", "preTokenizer": "Metaspace"},
    }
    verdicts = {name: own.own_support(form).ok for name, form in cases.items()}
    check("① 覆盖表 ✓：**三种模型 × 八种预分词器**都自研覆盖 ✓；"
          "还没做的（`Split` ✗）/ 不认识的模型 ⇒ **拒绝** ✓",
          verdicts == {"BPE+ByteLevel": True, "Unigram+Metaspace": True,
                       "WordPiece+BERT": True, "WordPiece+Digits": True,
                       "Unigram+Whitespace": True, "还没做的预设分词": False,
                       "不认识的模型": False}, verdicts)

    # ⚠️ 2026-09-21 起语义变了 ✗：`normalizer` **不再是"一律拒绝"** ✓ —— 实现了的（BertNormalizer
    #    / NFKC / Sequence … ✓）**能接** ✓，没实现的（Precompiled ✓）才拒绝 ✓。本条改判"能接" ✓。
    supported = own.own_support({"modelType": "WordPiece", "preTokenizer": "BertPreTokenizer",
                                 "normalizerTypes": ["BertNormalizer"]})
    rejected = own.own_support({"modelType": "WordPiece", "preTokenizer": "BertPreTokenizer",
                                "normalizerTypes": ["Nmt"]})
    check("② `normalizer` 分两档 ✓：实现了的**能接** ✓（`BertNormalizer` ✓）；"
          "没实现的**拒绝并说明** ✓（`Nmt` ✓）",
          supported.ok and "BertNormalizer" in supported.reason
          and not rejected.ok and "Nmt" in rejected.reason,
          (supported.reason, rejected.reason))
    fallback = own.own_support({"modelType": "Unigram", "preTokenizer": "Metaspace",
                                "byteFallback": True}).reason
    check("③ `byte_fallback` ⇒ 拒绝并说明 ✓（按字节兜底表尚未实现 ✓ 不假装能跑 ✗）",
          "byte_fallback" in fallback, fallback)


# ══════════════════════════════════════════════════════════════════════════
# ② WordPiece：贪心最长匹配 + 整词 UNK（**最容易写错的一条** ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_wordpiece(root: Path) -> None:
    path = write_wordpiece(root)
    if path is None:
        check("④ 参考实现没装 ⇒ WordPiece 对照跳过 ✓", True, "skipped")
        return
    impl = own.load_own_tokenizer(path)
    check("④ 装载：`WordPiece` + `BertPreTokenizer` ⇒ 走**自研** ✓（不是回退 ✓）",
          isinstance(impl, own.WordPieceTokenizer), type(impl).__name__)

    from tokenizers import Tokenizer  # noqa: PLC0415
    reference = Tokenizer.from_file(str(path / "tokenizer.json"))
    samples = ["hello world", "hello", "helloworld", "he llo", "hello, world!", "a b c",
               "中文 hello", "xyzzy", "hello-world", "HELLO"]
    mismatched = {}
    for text in samples:
        mine = impl.encode(text)
        theirs = list(reference.encode(text).ids)
        if mine != theirs:
            mismatched[text] = (mine, theirs)
    check("⑤ ⭐⭐ **与参考实现逐例同 id** ✓（`hello`→`he`+`##llo` ✓；`xyzzy` **整词 `[UNK]`** ✓"
          "；标点各切一刀 ✓；大小写敏感 ✓ —— 10 例 ✓）",
          not mismatched, mismatched)

    check("⑥ ⚠️ **整词失败 ⇒ 整个词 UNK** ✓（不是「能切多少切多少」 ✓✗ —— 这条最容易写错 ✓）",
          impl.encode("hellox") == [impl.unk_id], impl.encode("hellox"))


# ══════════════════════════════════════════════════════════════════════════
# ③ Unigram + Metaspace：Viterbi 最优路径 + `fuse_unk`
# ══════════════════════════════════════════════════════════════════════════
def case_unigram(root: Path) -> None:
    path = write_unigram_metaspace(root)
    if path is None:
        check("⑦ 参考实现没装 ⇒ Unigram 对照跳过 ✓", True, "skipped")
        return
    impl = own.load_own_tokenizer(path)
    check("⑦ 装载：`Unigram` + `Metaspace` ⇒ 走**自研** ✓",
          isinstance(impl, own.UnigramTokenizer), type(impl).__name__)

    from tokenizers import Tokenizer  # noqa: PLC0415
    reference = Tokenizer.from_file(str(path / "tokenizer.json"))
    # ⚠️ 空白用例是**冲着手写 Metaspace 的坑**去的 ✓（双空格 ✓ 前导 ✓ 尾随 ✓ 制表符**不切** ✓）：
    samples = ["hello world", "hello", "helloworld", "he llo world", "中文 hello",
               "hello  world", "world hello", "x", "hello!",
               " hello", "hello ", "hello\tworld", "a  b"]
    mismatched = {}
    for text in samples:
        mine = impl.encode(text)
        theirs = list(reference.encode(text).ids)
        if mine != theirs:
            mismatched[text] = (mine, theirs)
    check("⑧ ⭐⭐ **与参考实现逐例同 id** ✓（Viterbi 走**最优路径** ✓：`▁hello` 优先于 "
          "`▁he`+`llo` ✓；空格→`▁` ✓；首部补 `▁` ✓；**双空格不折叠** ✓；`\\t` **不切** ✓ —— 13 例 ✓）",
          not mismatched, mismatched)

    fused = impl.encode("xyz")            # 连续未知 ⇒ 合成一个 unk ✓（前面那个 `▁` 是 Metaspace 补的 ✓）
    theirs = list(reference.encode("xyz").ids)
    check("⑨ ⚠️ `fuse_unk`：**连续未知合成一个** `unk` ✓（不是三个 ✓✗；且与参考同 id ✓ —— "
          "`▁` 是 Metaspace 补的前导 ✓ 不是「多出来的 token」✓）",
          impl.encode("xyz").count(impl.unk_id) == 1 and fused == theirs, (fused, theirs))


# ══════════════════════════════════════════════════════════════════════════
# ④ 拒绝路径：不认识就**不走自研** ✓（由 hub 决定回退还是报错 ✓）
# ══════════════════════════════════════════════════════════════════════════
def write_normalized_wordpiece(root: Path) -> Path | None:
    """`WordPiece` + `BertNormalizer(lowercase=True)` ✓（**自研现在能接** ✓ 2026-09-21 ✓）。"""
    tokenizers = reference_module()
    if tokenizers is None:
        return None
    from tokenizers import Tokenizer  # noqa: PLC0415
    from tokenizers import models, normalizers, pre_tokenizers  # noqa: PLC0415

    target = root / "norm_wordpiece"
    target.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.WordPiece(
        {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3, "world": 4,
         "he": 5, "##llo": 6, "wor": 7, "##ld": 8, "cafe": 9, "中": 10, "文": 11},
        unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.save(str(target / "tokenizer.json"))
    return target


def case_normalizer(root: Path) -> None:
    """⭐ `normalizer` 自研（2026-09-21 补 ✓）：此前一律**拒绝** ✗ ⇒ BERT 系词表只能回退 ✗。"""
    # ── ① 覆盖判定 ✓
    ok_form = {"modelType": "WordPiece", "preTokenizer": "BertPreTokenizer",
               "normalizerTypes": ["BertNormalizer"]}
    seq_form = {"modelType": "Unigram", "preTokenizer": "Metaspace",
                "normalizerTypes": ["NFKC", "Lowercase"]}
    bad_form = {"modelType": "WordPiece", "preTokenizer": "BertPreTokenizer",
                "normalizerTypes": ["Precompiled"]}
    check("⑫ 覆盖判定：`BertNormalizer` ✓ / `Sequence`(NFKC+Lowercase) ✓ 都**能接** ✓；"
          "`Precompiled` ✗ ⇒ **拒绝并说明要 charsmap 表** ✓（不假装是 NFKC 换个写法 ✓）",
          own.own_support(ok_form).ok and own.own_support(seq_form).ok
          and not own.own_support(bad_form).ok
          and "charsmap" in own.own_support(bad_form).reason,
          own.own_support(bad_form).reason)

    # ── ② 逐例对齐参考实现的 `normalize_str` ✓（**先对规范化，再对 id** ✓ 定位快 ✓）
    if reference_module() is None:
        check("⑬ 参考实现没装 ⇒ normalizer 对照跳过 ✓", True, "skipped")
        return
    from tokenizers import Regex  # noqa: PLC0415
    from tokenizers import normalizers  # noqa: PLC0415

    samples = ["Hello WORLD", "  spaced\t tabs\n", "中文abc", "café naïve", "a\x00b",
               "① ℌ ½ ａ", "e\u0301", "\u3000x", "ＡBＣ", "  a\t\tb \n"]
    built = {
        "BertNormalizer(lowercase=True)": (own.BertNormalizer(lowercase=True),
                                           normalizers.BertNormalizer(lowercase=True)),
        "BertNormalizer(全关)": (own.BertNormalizer(lowercase=False, strip_accents=False,
                                                    handle_chinese_chars=False, clean_text=False),
                                 normalizers.BertNormalizer(lowercase=False, strip_accents=False,
                                                            handle_chinese_chars=False,
                                                            clean_text=False)),
        "NFKC": (own.UnicodeNormalizer("NFKC"), normalizers.NFKC()),
        "NFD": (own.UnicodeNormalizer("NFD"), normalizers.NFD()),
        "Lowercase": (own.SimpleNormalizer("Lowercase"), normalizers.Lowercase()),
        "StripAccents": (own.SimpleNormalizer("StripAccents"), normalizers.StripAccents()),
        "Strip": (own.SimpleNormalizer("Strip"), normalizers.Strip()),
        "Prepend": (own.SimpleNormalizer("Prepend", prepend="P:"), normalizers.Prepend("P:")),
        "Replace": (own.SimpleNormalizer("Replace", pattern={"Regex": r"\s+"}, content="_"),
                    normalizers.Replace(Regex(r"\s+"), "_")),
        "Sequence": (own.SequenceNormalizer([own.UnicodeNormalizer("NFKC"),
                                             own.BertNormalizer(lowercase=True)]),
                     normalizers.Sequence([normalizers.NFKC(),
                                           normalizers.BertNormalizer(lowercase=True)])),
    }
    wrong: dict[str, Any] = {}
    for label, (mine, theirs) in built.items():
        for text in samples:
            got, want = mine.normalize(text), theirs.normalize_str(text)
            if got != want:
                wrong[f"{label} | {text!r}"] = (got, want)
    check("⑬ ⭐⭐ **10 种 normalizer × 10 例 `normalize` 逐例与参考一致** ✓"
          "（含 `\\t`/`\\n`→空格 ✓ 控制字符丢弃 ✓ CJK 两侧加空格 ✓ 去重音 ✓ NFKC ½→1⁄2 ✓）",
          not wrong, wrong)

    # ── ③ 端到端：带 normalizer 的词表 ⇒ **走自研** ✓ 且逐例同 id ✓
    path = write_normalized_wordpiece(root)
    impl = own.load_own_tokenizer(path)
    check("⑭ 带 `BertNormalizer` 的词表 ⇒ **走自研**（包装层 ✓）且报告里能看到 normalizer ✓",
          isinstance(impl, own.NormalizedTokenizer)
          and impl.describe()["normalizer"]["name"] == "BertNormalizer", type(impl).__name__)

    from tokenizers import Tokenizer  # noqa: PLC0415
    reference = Tokenizer.from_file(str(path / "tokenizer.json"))
    texts = ["Hello WORLD", "hello, world!", "CAFÉ", "中文 hello", "  HELLO  "]
    mismatched = {text: (impl.encode(text), list(reference.encode(text).ids))
                  for text in texts if impl.encode(text) != list(reference.encode(text).ids)}
    check("⑮ ⭐⭐ 端到端**逐例同 id** ✓（规范化 + 预分词 + 贪心匹配三层一起对 ✓ —— 5 例含 CJK ✓）",
          not mismatched, mismatched)

    # ⚠️ 反向证明（**能触发** ✓）：把 normalizer 摘掉再编，结果**必不同** ✓（否则第 ⑮ 条恒真 ✓）
    bare = own.WordPieceTokenizer(impl._impl.vocab, unk_token="[UNK]",  # noqa: SLF001
                                 pre_tokenizer=own.BertPreTokenizer())
    check("⑯ 反向证明：**不套 normalizer** 的同一份权重 ⇒ `CAFÉ` 必然编不出来 ✓"
          "（⇒ 规范化的确是**在用**的 ✓ 不是摆设 ✓）",
          bare.encode("CAFÉ") != impl.encode("CAFÉ"), (bare.encode("CAFÉ"), impl.encode("CAFÉ")))


def case_pretokenizers(root: Path) -> None:
    """⭐ 另外几种**规则型**预分词器（2026-09-21 补 ✓）：规则全部**逐例实测**得来 ✓。"""
    if reference_module() is None:
        check("⑰ 参考实现没装 ⇒ 预分词器对照跳过 ✓", True, "skipped")
        return
    from tokenizers import pre_tokenizers as ref_pre  # noqa: PLC0415

    texts = ["ab12cd", "a, b!", "a|b||c", "hi  there", "a  b", "!!!", ",x", "x,",
             "中,文", "  a", "a ", "", "a,,b", "..."]
    suites: list[tuple[str, Any, Any]] = [
        ("WhitespaceSplit", own.SimplePreTokenizer("WhitespaceSplit"), ref_pre.WhitespaceSplit()),
        ("Whitespace", own.SimplePreTokenizer("Whitespace"), ref_pre.Whitespace()),
        ("Digits(逐位)", own.SimplePreTokenizer("Digits", individual_digits=True),
         ref_pre.Digits(individual_digits=True)),
        ("Digits(整段)", own.SimplePreTokenizer("Digits", individual_digits=False),
         ref_pre.Digits(individual_digits=False)),
        ("CharDelimiterSplit", own.SimplePreTokenizer("CharDelimiterSplit", delimiter="|"),
         ref_pre.CharDelimiterSplit("|")),
    ]
    for behavior in ("isolated", "removed", "merged_with_previous", "merged_with_next",
                     "contiguous"):
        suites.append((f"Punctuation({behavior})",
                       own.SimplePreTokenizer("Punctuation", behavior=behavior),
                       ref_pre.Punctuation(behavior=behavior)))

    wrong: dict[str, Any] = {}
    for label, mine, theirs in suites:
        for text in texts:
            got = [piece for piece in mine.split(text) if piece]
            want = [piece for piece, _ in theirs.pre_tokenize_str(text)]
            if got != want:
                wrong[f"{label} | {text!r}"] = (got, want)
    check("⑰ ⭐⭐ **10 种预分词配置 × 14 例**与参考逐例一致 ✓（`Whitespace` 按"
          "字母数字/非字母数字边界 ✓、`Punctuation` **五种 behavior** ✓、"
          "`Digits` 两种口径 ✓、`CharDelimiterSplit` 丢掉分隔符与空片 ✓）",
          not wrong, wrong)

    legacy = own.SimplePreTokenizer("Punctuation", is_whitespace_prefix=True)
    check("⑱ 老字段 `is_whitespace_prefix`（bool ✓ 与新版 `behavior` **不是一一对应** ✗）"
          "⇒ **拒绝** ✓（映射没核过就不猜 ✓）",
          _raises(lambda: legacy.split("a, b"), "is_whitespace_prefix") is not None, None)


def case_reject(root: Path) -> None:
    # ⚠️ 覆盖面变了 ✗：带 `normalizer` 的词表**现在能走自研** ✓ ⇒ 拒绝用例换成**没实现**的
    #    `Precompiled` ✓（要 SentencePiece charsmap 表 ✓ 手写 JSON 即可 ✓ 不需要参考实现 ✓）。
    path = write_with_normalizer(root)
    check("⑩ 带**已实现** normalizer 的词表 ⇒ 走自研包装 ✓（不再是「一律拒绝」 ✗）",
          isinstance(own.load_own_tokenizer(path), own.NormalizedTokenizer),
          type(own.load_own_tokenizer(path)).__name__)

    unsupported = root / "unsupported"
    unsupported.mkdir(parents=True, exist_ok=True)
    (unsupported / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "Unigram", "vocab": [("<unk>", 0.0), ("a", -1.0)], "unk_id": 0},
        "pre_tokenizer": {"type": "Metaspace"},
        "normalizer": {"type": "Precompiled", "precompiled_charsmap": "AAA"},
    }), encoding="utf-8")
    check("⑪ 含**未实现** normalizer（`Precompiled` ✗ 要 charsmap 表 ✓）⇒ 返回 `None` ✓"
          "（明确不走自研 ✓ 由上层回退参考实现 ✓）",
          own.load_own_tokenizer(unsupported) is None, None)

    empty = root / "nothing"
    empty.mkdir()
    check("⑪′ 找不到词表 ⇒ `None` ✓（不抛 ✗ —— 「行不行」是结论 ✓ 不是异常 ✓）",
          own.load_own_tokenizer(empty) is None, None)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_support(root)
        case_wordpiece(root)
        case_unigram(root)
        case_normalizer(root)
        case_pretokenizers(root)
        case_reject(root)
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      ↳ {detail}"))
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed"
          + (" ✗✗✗" if failed else " ✓"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
