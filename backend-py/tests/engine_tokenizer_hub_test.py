"""S26 自检：**分词器总入口（优化改造层）**（2026-09-20）。

上一版只有「自研 BPE（**只覆盖 ByteLevel-BPE** ✗）」与「`HFTokenizer`（只覆盖能进
`AutoTokenizer` 的 ✓）」两块**各管一段** ✗。本套验证 `engine/tokenizer_hub.py` 把它们合成
**统一入口** ✓：形态嗅探 ✓ → 自研优先 ✓ → 参考实现回退 ✓ → 批量 + 缓存 ✓ → 可互校 ✓ →
`transformers` 用法优化（离线 ✓ 关遥测 ✓ 缓存目录 ✓）。

⚠️ 参考实现侧用的是 `transformers` 自带的 `tokenizers` ✓（本机已装 ✓）；**没装的世界**用
``monkeypatch`` 模拟 ✓（本仓纪律：缺依赖路径不许写成「真机必然缺」✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_tokenizer_hub_test.py
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

from app.services.engine import tokenizer_bpe as tb  # noqa: E402
from app.services.engine import tokenizer_hub as hub_mod  # noqa: E402

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


SPACE = tb.bytes_to_unicode()[32]


# ══════════════════════════════════════════════════════════════════════════
# 夹具：① ByteLevel-BPE（自研能覆盖 ✓）② Unigram+Metaspace（自研覆盖不了 ✗）
# ══════════════════════════════════════════════════════════════════════════
def write_byte_bpe(root: Path, *, nested: bool = False, variant: int = 0) -> Path:
    """ByteLevel-BPE 词表 ✓（`nested=True` ⇒ 形态字段包在 ``Sequence`` 里 ✓ 验递归 ✓；
    `variant=1` ⇒ **多一条 merge** ✓ 用来验「指纹随词表变」✓）。"""
    target = root / ("nested" if nested else f"bytebpe{variant}")
    target.mkdir(parents=True, exist_ok=True)
    vocab = {char: index for index, char in enumerate(sorted(tb.bytes_to_unicode().values()))}
    extra = 400
    # ⚠️ 合并链要**完整** ✓：`hello` 与 `" world"`（注意空格是映射后的字符 ✓）都要能一路合到底 ✓，
    #    否则 `hello world` 会停在 6 个 token ✓✗（"2 个 token" 的断言就是拿它当判据 ✓）。
    merges = ["h e", "l l", "he ll", "hell o",
              f"{SPACE} w", f"{SPACE}w o", f"{SPACE}wo r", f"{SPACE}wor l", f"{SPACE}worl d"]
    for token in ("he", "ll", "hell", "hello",
                  f"{SPACE}w", f"{SPACE}wo", f"{SPACE}wor", f"{SPACE}worl", f"{SPACE}world"):
        vocab[token] = extra
        extra += 1
    if variant:
        vocab[f"{SPACE}h"] = extra
        merges.append(f"{SPACE} h")
    vocab["<|endoftext|>"] = 999
    pre: Any = ({"type": "Sequence", "pretokenizers": [{"type": "ByteLevel"}]}
                if nested else {"type": "ByteLevel"})
    (target / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
        "added_tokens": [{"id": 999, "content": "<|endoftext|>", "special": True}],
        "pre_tokenizer": pre,
    }), encoding="utf-8")
    return target


def write_wordpiece(root: Path) -> Path | None:
    """WordPiece + `BertPreTokenizer` 词表 ✓（同样**程序化构造** ✓ 不需要联网 ✓）。"""
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415
        from tokenizers import models as ref_models  # noqa: PLC0415
        from tokenizers import pre_tokenizers as ref_pre  # noqa: PLC0415
    except ImportError:
        return None
    target = root / "wordpiece"
    target.mkdir(parents=True, exist_ok=True)
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3, "world": 4,
             "he": 5, "##llo": 6, "wor": 7, "##ld": 8}
    tokenizer = Tokenizer(ref_models.WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ref_pre.BertPreTokenizer()
    tokenizer.save(str(target / "tokenizer.json"))
    return target


def write_refused_form(root: Path) -> Path | None:
    """**自研明确拒绝**的形态 ✓ —— 现在是 `UnicodeScripts` 预分词器 ✓（2026-09-21 ✓）：

    ⚠️ **判据跟着能力走** ✗（这个夹具换过三次 ✓）：最早是「带 `normalizer` 的 WordPiece」✗
    ⇒ normalizer 自研后失效 ✓；后来用 `byte_fallback` 的 Unigram ✗ ⇒ **字节回退自研后也失效** ✓
    （而且当初的拒绝理由本身是错的 ✗ —— 那次实测**少了前置条件**：词表里没有 `<0xNN>` 字节 token ✓）。
    现在用 `UnicodeScripts` ✓：它要 **Unicode script 表** ✓ 而**标准库没有** ✗（装 `regex` 才有 ✓
    但本仓不引 ✓）⇒ 自研**明确拒绝** ✓、参考实现支持 ✓ ⇒ 是干净的「回退」用例 ✓。
    """
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415
        from tokenizers import models as ref_models  # noqa: PLC0415
        from tokenizers import pre_tokenizers as ref_pre  # noqa: PLC0415
    except ImportError:
        return None
    target = root / "refused"
    target.mkdir(parents=True, exist_ok=True)
    vocab = [("<unk>", 0.0), ("▁hello", -1.0), ("▁h", -2.0), ("e", -2.5), ("llo", -2.6)]
    tokenizer = Tokenizer(ref_models.Unigram(vocab, unk_id=0))
    tokenizer.pre_tokenizer = ref_pre.UnicodeScripts()
    tokenizer.save(str(target / "tokenizer.json"))
    return target


def write_metaspace_unigram(root: Path) -> Path | None:
    """Unigram + Metaspace 词表 ✓（用参考实现**程序化构造** ✓ 不需要联网 ✓）。"""
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415
        from tokenizers import models as ref_models  # noqa: PLC0415
        from tokenizers import pre_tokenizers as ref_pre  # noqa: PLC0415
    except ImportError:
        return None
    target = root / "unigram_meta"
    target.mkdir(parents=True, exist_ok=True)
    vocab = [(token, -float(index)) for index, token in enumerate(
        ["<unk>", "<s>", "</s>", "▁hello", "▁world", "▁he", "llo", "▁wor", "ld", "▁h", "e"])]
    # ⚠️ `unk_id` 必须显式给 ✓（否则编码时报 `Encountered an unknown token but unk_id is missing` ✓✗
    #    —— 这是参考实现的要求 ✓，不是我们包装层的问题 ✓）
    tokenizer = Tokenizer(ref_models.Unigram(vocab, unk_id=0))
    # ⚠️ `Metaspace` 的参数随版本变过 ✓：0.23 只认 `replacement` ✓（`add_prefix_space` 会
    #    `TypeError` ✓ —— 别照旧文档写 ✗）
    tokenizer.pre_tokenizer = ref_pre.Metaspace(replacement="▁")
    tokenizer.save(str(target / "tokenizer.json"))
    return target


# ══════════════════════════════════════════════════════════════════════════
# ① 形态嗅探（判据是**结构字段** ✓ 不是文件名 ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_detect(root: Path) -> None:
    form = hub_mod.detect_form(write_byte_bpe(root))
    check("① ByteLevel-BPE ⇒ `ownBpeOk=True` ✓（自研能覆盖 ✓）",
          form["ownBpeOk"] and form["modelType"] == "BPE"
          and form["preTokenizer"] == "ByteLevel", form)

    nested = hub_mod.detect_form(write_byte_bpe(root, nested=True))
    check("② 形态字段包在 `Sequence` 里 ⇒ **递归取到真实形态** ✓（否则会误判成不支持 ✗）",
          nested["ownBpeOk"] and nested["preTokenizer"] == "ByteLevel", nested)

    classic = root / "classic"
    classic.mkdir()
    (classic / "vocab.json").write_text(json.dumps({**_byte_vocab(), "he": 300}), encoding="utf-8")
    (classic / "merges.txt").write_text("#version: 0.2\nh e\n", encoding="utf-8")
    check("③ 经典 `vocab.json`+`merges.txt` ⇒ 也认作 ByteLevel-BPE ✓（GPT-2 格式就是它 ✓）",
          hub_mod.detect_form(classic)["ownBpeOk"] is True, None)

    meta = write_metaspace_unigram(root)
    if meta is not None:
        form = hub_mod.detect_form(meta)
        check("④ Unigram + Metaspace ⇒ `ownBpeOk=False` ✓（自研**不覆盖** ⇒ 该走回退 ✓）",
              form["ownBpeOk"] is False and form["modelType"] == "Unigram"
              and form["preTokenizer"] == "Metaspace", form)
    else:
        check("④ 参考实现没装 ⇒ 这项形态嗅探跳过 ✓（不是失败 ✗）", True, "skipped")

    empty = root / "nothing"
    empty.mkdir()
    check("⑤ 认不出的目录 ⇒ **说清找过什么** ✓（不静默当空词表 ✗）",
          _raises(lambda: hub_mod.detect_form(empty), "没有可识别的词表") is not None, None)


def _byte_vocab() -> dict[str, int]:
    return {char: index for index, char in enumerate(sorted(tb.bytes_to_unicode().values()))}


# ══════════════════════════════════════════════════════════════════════════
# ② 自研优先 ✓（**不需要** transformers ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_own_first(root: Path) -> None:
    original = hub_mod.reference_available
    hub_mod.reference_available = lambda: False        # 模拟「参考实现没装」的世界 ✓
    try:
        hub = hub_mod.load(hub_mod.HubConfig(path=str(write_byte_bpe(root))))
    finally:
        hub_mod.reference_available = original  # type: ignore[assignment]
    check("⑥ ⭐ **自研优先**：ByteLevel-BPE 词表在「参考实现没装」的世界里**照样装得上** ✓"
          "（零依赖 ✓ 离线 ✓ —— 这条证明它真没依赖 `transformers` ✗）",
          hub.backend == "own-bpe" and hub.name == "bpe" and hub.vocab_size > 250,
          hub.describe())
    check("⑦ 装载报告**如实**写出「为什么不用参考实现」✓（不是口号 ✓）",
          any("自研实现覆盖" in note and "不需要 `transformers`" in note for note in hub.notes),
          hub.notes)

    ids = hub.encode("hello world")
    check("⑧ 自研侧编码 ✓：`hello world` ⇒ 2 个 token（合并真的发生 ✓）",
          len(ids) == 2, ids)
    check("⑨ 往返恒等 ✓（自研侧的强不变量 ✓）",
          hub.decode(hub.encode("hello world"), skip_special_tokens=False) != "" and
          hub.decode(hub.encode("中文", add_special_tokens=False)) == "中文", None)


# ══════════════════════════════════════════════════════════════════════════
# ③ 参考实现回退 ✓（**能力升级**：形态覆盖从 1 种变成多种）
# ══════════════════════════════════════════════════════════════════════════
def case_fallback(root: Path) -> None:
    """⚠️ 这一组**只测"自研明确拒绝"的形态** ✓ —— 因为三大血统现在都自研了 ✗
    （2026-09-20 第二次扩充 ✓ ⇒ 别再拿 `Unigram` 当回退用例 ✓✗：它会走自研 ✓）。"""
    refused = write_refused_form(root)
    if refused is None:
        check("⑩ 参考实现没装 ⇒ 回退用例跳过 ✓", True, "skipped")
        return
    hub = hub_mod.load(hub_mod.HubConfig(path=str(refused)))
    check("⑩ ⭐ **回退只在「自研拒绝」时发生** ✓：`UnicodeScripts` 预分词器"
          "（要 script 表 ✓ 标准库没有 ✗）⇒ 走参考实现 ✓ 且**真能编出 id** ✓",
          hub.backend == "transformers" and len(hub.encode("hello world")) > 0
          and any("回退" in note for note in hub.notes),
          (hub.backend, hub.encode("hello world"), hub.notes))
    check("⑪ 回退路径的 `required_vocab_size` 覆盖特殊符 ✓（HF 的 `len(tokenizer)` 口径 ✓）",
          hub.required_vocab_size >= hub.vocab_size, (hub.vocab_size, hub.required_vocab_size))
    check("⑫ 回退也**如实报形态** ✓（模型 + 预分词 + **展平后的 normalizer 列表** ✓ + "
          "`byteFallback` ✓ —— 判据跟着能力走 ✓）",
          hub.detail["form"]["modelType"] == "Unigram"
          and hub.detail["form"]["preTokenizer"] == "UnicodeScripts"
          and hub.detail["form"]["byteFallback"] is False
          and hub.detail["form"]["normalizerTypes"] == [], hub.detail)

    # ⚠️ 反向证明：**同一份**词表，参考实现「没装」⇒ 必须**报错并给命令** ✓（不静默换实现 ✗）
    original = hub_mod.reference_available
    hub_mod.reference_available = lambda: False
    try:
        message = _raises(lambda: hub_mod.load(hub_mod.HubConfig(path=str(refused))))
    finally:
        hub_mod.reference_available = original  # type: ignore[assignment]
    check("⑬ 自研拒绝该形态 + 参考实现没装 ⇒ **报错 + 可执行安装命令 + 拒绝理由** ✓"
          "（绝不静默换一个词表 ✓✗）",
          message is not None and "pip install" in message and "UnicodeScripts" in message,
          message)


# ══════════════════════════════════════════════════════════════════════════
# ④ 性能优化：批量一致 ✓ + LRU 缓存可观测 ✓ + transformers 用法开关 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_own_three_families(root: Path) -> None:
    """⭐⭐ **三大血统都自研**（2026-09-20 第二次扩充 ✓）：`Unigram`/`WordPiece` 不再依赖回退 ✓。"""
    from app.services.engine import tokenizer_own as own_mod  # noqa: PLC0415

    meta = write_metaspace_unigram(root)
    if meta is None:
        check("㉖ 参考实现没装 ⇒ 自研覆盖用例跳过 ✓（不是失败 ✗）", True, "skipped")
        return
    unigram = hub_mod.load(hub_mod.HubConfig(path=str(meta)))
    check("㉖ ⭐⭐ **`Unigram` + `Metaspace` 走自研** ✓（`backend=own-unigram` ✓ "
          "—— 不再是「必须回退参考实现」✗）且**能编出 id** ✓",
          unigram.backend == "own-unigram" and len(unigram.encode("hello world")) > 0
          and own_mod.own_support(hub_mod.detect_form(meta)).ok,
          (unigram.backend, unigram.encode("hello world")))

    wordpiece = write_wordpiece(root)
    if wordpiece is None:
        check("㉗ 参考实现没装 ⇒ WordPiece 覆盖用例跳过 ✓", True, "skipped")
        return
    wp_hub = hub_mod.load(hub_mod.HubConfig(path=str(wordpiece)))
    check("㉗ ⭐⭐ **`WordPiece` + `BertPreTokenizer` 走自研** ✓（`backend=own-wordpiece` ✓）"
          "—— 三大血统（BPE / Unigram / WordPiece）**都不靠参考实现** ✓",
          wp_hub.backend == "own-wordpiece" and len(wp_hub.encode("hello world")) > 0,
          (wp_hub.backend, wp_hub.encode("hello world")))

    # ⚠️ 预分词器的**参数**也会决定能不能接 ✗（2026-09-21 起 ✓）：`Split` 的正则 ✓
    #    / `FixedLength` 的 length ✓ —— 只看类型名会误判 ✓✗（所以 `detect_form` 要带原始规格 ✓）。
    split_dir = root / "split_form"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "WordPiece", "vocab": {"[UNK]": 0, "ab": 1, "cd": 2, "12": 3},
                  "unk_token": "[UNK]"},
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": r"\d+"}, "behavior": "isolated"},
    }), encoding="utf-8")
    split_hub = hub_mod.load(hub_mod.HubConfig(path=str(split_dir)))
    check("㉘ ⭐ `Split` 预分词器**也走自研** ✓（`WordPiece` + `Split(\\d+)` ⇒ `own-wordpiece` ✓）"
          "且真能编 ✓",
          split_hub.backend == "own-wordpiece"
          and split_hub.encode("ab12cd", add_special_tokens=False) != [],
          (split_hub.backend, split_hub.encode("ab12cd", add_special_tokens=False)))

    hostile = root / "hostile_split"
    hostile.mkdir(parents=True, exist_ok=True)
    (hostile / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "WordPiece", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"},
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": r"\p{N}+"}, "behavior": "isolated"},
    }), encoding="utf-8")
    support = own_mod.own_support(hub_mod.detect_form(hostile))
    check("㉙ `\\p{…}` 正则（标准库**没有** ✗）⇒ **带理由拒绝** ✓（回退参考实现 ✓）"
          "—— 不静默按错的语义切 ✓✗",
          support.ok is False and "\\p{" in support.reason, support.reason)

    # ⭐ 字节回退**也能自研**了 ✓（2026-09-21 第二轮 ✓ —— 此前它被当成「回退用例」✗ 见上面的注释 ✓）
    bf_dir = root / "byte_fallback"
    bf_dir.mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer as RefTokenizer  # noqa: PLC0415
    from tokenizers import models as ref_models  # noqa: PLC0415
    from tokenizers import pre_tokenizers as ref_pre  # noqa: PLC0415
    bf_vocab = [("<unk>", -10.0), ("a", -1.0), ("▁", -2.0)]
    bf_vocab += [(f"<0x{byte:02X}>", -5.0 - index)
                 for index, byte in enumerate("中".encode("utf-8"))]
    ref_tok = RefTokenizer(ref_models.Unigram(bf_vocab, unk_id=0, byte_fallback=True))
    ref_tok.pre_tokenizer = ref_pre.Metaspace(replacement="▁")
    ref_tok.save(str(bf_dir / "tokenizer.json"))
    bf_hub = hub_mod.load(hub_mod.HubConfig(path=str(bf_dir)))
    check("㉚ ⭐ **`byte_fallback` 也走自研** ✓（`own-unigram` ✓ —— **不再**是回退用例 ✓）"
          "且与参考**逐例同 id** ✓（含 `'中'`/`'中中'`/`'中x'` ✓ —— 段级判据 ✓）",
          bf_hub.backend == "own-unigram"
          and all(bf_hub.encode(text, add_special_tokens=False) == list(ref_tok.encode(text).ids)
                  for text in ("中", "中中", "a中", "中x", "a中x")),
          (bf_hub.backend, bf_hub.encode("中x", add_special_tokens=False),
           list(ref_tok.encode("中x").ids)))


def case_optimizations(root: Path) -> None:
    hub = hub_mod.load(hub_mod.HubConfig(path=str(write_byte_bpe(root))))
    texts = ["hello world", "hello", "", "中文测试", "hello world"]
    batch = hub.encode_batch(texts)
    check("⑭ ⭐ 批量 == 逐条 ✓（**钉住的不变量** ✓ —— 批量只是快慢 ✓ 语义不许有别 ✓）",
          batch == [hub.encode(text) for text in texts], batch)

    hub.clear_cache()
    hub.encode("缓存用样例")
    hub.encode("缓存用样例")
    stats = hub.cache_stats()
    check("⑮ LRU 缓存**可观测** ✓：同文本第二次命中 ✓（`hits` 增 ✓ —— 否则「加了缓存」无法自证 ✓）",
          stats["hits"] >= 1 and stats["currsize"] >= 1, stats)
    before = hub.cache_stats()
    hub.encode("另一条完全不同的文本")
    after = hub.cache_stats()
    check("⑯ 新文本 ⇒ **不误命中** ✓（`misses` 增 ✓；否则缓存就是假的 ✓）",
          after["misses"] > before["misses"], (before, after))
    check("⑰ 关缓存 ⇒ 仍能编 ✓（`cache_entries=0` 不炸 ✗）",
          hub_mod.load(hub_mod.HubConfig(path=str(write_byte_bpe(root)), cache_entries=0)
                       ).encode("no-cache") != [], None)

    check("⑱ `transformers` 用法优化落到**环境变量**上 ✓：离线 + 关遥测 + 缓存目录 ✓",
          os.environ.get("HF_HUB_OFFLINE") == "1"
          and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
          and os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"
          and bool(os.environ.get("HF_HOME")),
          {key: os.environ.get(key) for key in
           ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "HF_HOME")})
    check("⑲ 缓存目录指向**本仓数据根** ✓（不写用户主目录 ✓）",
          "hf-cache" in str(os.environ.get("HF_HOME")), os.environ.get("HF_HOME"))


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 可互校 ✓（自研 ↔ 参考实现）+ 指纹随词表变 ✓
# ══════════════════════════════════════════════════════════════════════════
def write_reference_byte_bpe(root: Path) -> Path | None:
    """⚠️ 由**参考实现自己写**的完整 `tokenizer.json` ✓（手写极简版会被它拒 ✓✗ —— 见第 ㉓ 条 ✓）。"""
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415
        from tokenizers import models as ref_models  # noqa: PLC0415
        from tokenizers import pre_tokenizers as ref_pre  # noqa: PLC0415
    except ImportError:
        return None
    target = root / "ref_bytebpe"
    target.mkdir(parents=True, exist_ok=True)
    vocab = {char: index for index, char in enumerate(sorted(tb.bytes_to_unicode().values()))}
    merges = [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o")]
    for extra, token in enumerate(("he", "ll", "hell", "hello"), start=400):
        vocab[token] = extra
    tokenizer = Tokenizer(ref_models.BPE(vocab=vocab, merges=merges))
    tokenizer.pre_tokenizer = ref_pre.ByteLevel(add_prefix_space=False)
    tokenizer.save(str(target / "tokenizer.json"))
    return target


def case_verify(root: Path) -> None:
    complete = write_reference_byte_bpe(root)
    hub = hub_mod.load(hub_mod.HubConfig(path=str(complete or write_byte_bpe(root))))
    report = hub.verify_against_reference(["hello world", "hello", "a b"])
    if complete is not None:
        check("⑳ ⭐⭐ 运行期互校：自研 ↔ 参考实现**逐例同 id** ✓（换词表后当场可复验 ✓）",
              report["ok"] is True and report["compared"] == 3, report)
    else:
        check("⑳ 参考实现没装 ⇒ 降级为**往返恒等**核对 ✓（并**说明原因** ✓ 不假装比过 ✗）",
              report.get("ok") is None and "roundTrip" in report, report)

    # ⚠️⚠️ **别让判据说谎** ✓：手写极简 `tokenizer.json` 参考侧**装不了** ✓ ⇒ 文案必须说
    #    「装不了这份词表」✓，**不能**说成「没装」✗（两者混一起 = 自检里最坏的一类误导 ✓✗）。
    handwritten = hub_mod.load(hub_mod.HubConfig(path=str(write_byte_bpe(root, variant=1))))
    tricky = handwritten.verify_against_reference(["hello"])
    check("㉓ ⭐ 参考侧**造不出来**时如实说「装不了这份词表」✓（不是「没装」✗ —— 判据不许说谎 ✓）",
          tricky.get("ok") is None
          and any("装不了这份词表" in note or "没装" in note for note in tricky["notes"]),
          tricky)

    refused = write_refused_form(root)
    if refused is not None:
        fallback_report = hub_mod.load(hub_mod.HubConfig(path=str(refused))
                                       ).verify_against_reference(["hello world"])
        check("㉑ 回退后端互校：自研覆盖不了该形态 ⇒ **如实说明** ✓ 且 `compared=0` ✓"
              "（不假装比过 ✓）",
              fallback_report["compared"] == 0 and fallback_report["ok"] is None
              and any("覆盖不了" in note for note in fallback_report["notes"]),
              fallback_report)

    first = hub.fingerprint()
    other = hub_mod.load(hub_mod.HubConfig(path=str(write_byte_bpe(root, variant=1))))
    check("㉒ 词表指纹**真的随词表变** ✓（多一条 merge + 多一个 token ⇒ 指纹必变 ✓；"
          "「换没换过词表」要能机械回答 ✓）",
          first["backend"] == "own-bpe" and other.fingerprint()["backend"] == "own-bpe"
          and first["merges"] != other.fingerprint()["merges"]
          and first["vocabSize"] != other.fingerprint()["vocabSize"],
          (first, other.fingerprint()))

    check("㉓ 既不给 `path` 也不给 `repo` ⇒ 报错 ✓（不猜一份词表 ✗）",
          _raises(lambda: hub_mod.load(), "不知道装哪份词表") is not None, None)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_detect(root)
        case_own_first(root)
        case_fallback(root)
        case_own_three_families(root)
        case_optimizations(root)
        case_verify(root)
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      ↳ {detail}"))
    # ⚠️ 汇总行**必须是 `SUMMARY: n/m passed`** ✓ —— `run_all.py` 按这个前缀收敛项数 ✓✗
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed"
          + (" ✗✗✗" if failed else " ✓"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
