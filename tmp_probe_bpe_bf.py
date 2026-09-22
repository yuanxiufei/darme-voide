"""探 BPE 的 `byte_fallback` 触发条件（Unigram 那条已核清 ✓，BPE 血统不同 ✗）。

BPE 走的是**字节级词表** ✓：预分词后的字符已在 byte↔unicode 映射里 ✓ ⇒
「回退」应当只在「某个字节的映射字符**不在词表**里」时发生 ✓ —— 用探针确认 ✓。
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
from tokenizers import Tokenizer  # noqa: E402

from app.services.engine.tokenizer_bpe import bytes_to_unicode  # noqa: E402

#: 本仓那份 byte↔unicode 映射 ✓（GPT-2 算法生成 ✓ 且与参考**逐例对过** ✓ —— 不自己抄表 ✗）
ALPHABET = {value: char for value, char in bytes_to_unicode().items()}
ALPHABET = {byte: bytes_to_unicode()[byte] for byte in range(256)}
print("空格字节 0x20 ⇒", repr(ALPHABET[32]), "| 字节 0x63('c') ⇒", repr(ALPHABET[99]))


def build(root: Path, *, letters: str, fallback: bool, byte_tokens: bool) -> Path:
    vocab: dict[str, int] = {"<unk>": 0}
    for char in letters:
        vocab[ALPHABET[ord(char)]] = len(vocab)
    if byte_tokens:
        for value in range(256):
            vocab.setdefault(f"<0x{value:02X}>", len(vocab))
    payload = {"model": {"type": "BPE", "unk_token": "<unk>", "byte_fallback": fallback,
                         "vocab": vocab, "merges": []},
               "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                                 "trim_offsets": True, "use_regex": True}}
    root.mkdir(parents=True, exist_ok=True)
    path = root / "tokenizer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def build_partial(root: Path, byte_names: list[str], *, letters: str = "ab") -> Path:
    """只放**部分**字节 token ✓（探「缺一个字节」在 BPE 上怎么处理 ✗）。"""
    vocab: dict[str, int] = {"<unk>": 0}
    for char in letters:
        vocab[ALPHABET[ord(char)]] = len(vocab)
    for name in byte_names:
        vocab[name] = len(vocab)
    payload = {"model": {"type": "BPE", "unk_token": "<unk>", "byte_fallback": True,
                         "vocab": vocab, "merges": []},
               "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                                 "trim_offsets": True, "use_regex": True}}
    root.mkdir(parents=True, exist_ok=True)
    path = root / "tokenizer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for label, names in (("只有 <0xC4>", ["<0xC4>"]), ("只有 <0xA0>", ["<0xA0>"]),
                         ("<0xC4>+<0xA0>", ["<0xC4>", "<0xA0>"]),
                         ("只有 <0x63>", ["<0x63>"]), ("<0x63>+<0xC4>", ["<0x63>", "<0xC4>"])):
        path = build_partial(root / label.replace("<", "").replace(">", ""), names)
        tokenizer = Tokenizer.from_file(str(path))
        enc = tokenizer.encode("a c")
        print(f"partial {label:16s} 'a c' ids={enc.ids} tokens={enc.tokens}")

    # ⚠️ 逐**字符**还是逐**片**？（BPE 与 Unigram 那边可能不同 ✗）：
    #    'a' 在词表里 ✓、'c' 可字节展开 ✓、'b' 既不在词表也没字节 token ✗ ⇒ 同片里混着三种 ✓
    only_c = {"<0x63>": 2}
    mixed = build_partial(root / "mixed", ["<0x63>"], letters="a")
    tokenizer = Tokenizer.from_file(str(mixed))
    for text in ("cb", "bc", "cbc", "a c", "acb"):
        enc = tokenizer.encode(text)
        print(f"mixed(只有 a + <0x63>) {text!r:6s} ids={enc.ids} tokens={enc.tokens}")

    for letters in ("ab", "abc"):
        for fallback in (False, True):
            for byte_tokens in (False, True):
                path = build(root / f"{letters}_{fallback}_{byte_tokens}", letters=letters,
                             fallback=fallback, byte_tokens=byte_tokens)
                tokenizer = Tokenizer.from_file(str(path))
                for text in ("abc", "ab", "cb", "a c"):
                    enc = tokenizer.encode(text)
                    print(f"letters={letters:3s} fallback={fallback!s:5s} byteTokens={byte_tokens!s:5s}"
                          f" {text!r:6s} ids={enc.ids} tokens={enc.tokens}")
