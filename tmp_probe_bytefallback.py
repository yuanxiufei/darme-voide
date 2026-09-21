"""探 `byte_fallback` 的参考语义（第二轮：决定实现的判据）。

第一轮结论（已跑 ✓）：`byte_fallback: true` **只有**在词表里**真有 `<0xNN>` 字节 token** 时才生效 ✓
（词表没有它们 ⇒ 仍出 `<unk>` ✓ —— 之前「实测没走字节回退」的拒绝理由其实是**词表缺字节 token** ✗）。
"""
import json
import tempfile
from pathlib import Path

from tokenizers import Tokenizer


def build(root: Path, vocab: dict[str, float]) -> Path:
    payload = {
        "model": {"type": "Unigram", "unk_id": 0, "byte_fallback": True,
                  "vocab": [[token, score] for token, score in vocab.items()]},
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"},
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "tokenizer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def probe(label: str, extra: dict[str, float], texts: list[str]) -> None:
    base = {"<unk>": -10.0, "a": -1.0, "▁": -2.0}
    base.update(extra)
    with tempfile.TemporaryDirectory() as tmp:
        path = build(Path(tmp) / "t", base)
        tokenizer = Tokenizer.from_file(str(path))
        for text in texts:
            ids = tokenizer.encode(text).ids
            print(f"{label:26s} {text!r:12s} ids={ids} "
                  f"tokens={tokenizer.encode(text).tokens}")


ALL_ZHONG = {f"<0x{b:02X}>": -5.0 - i for i, b in enumerate("中".encode("utf-8"))}
ONLY_FIRST = {"<0xE4>": -5.0}
probe("全 3 个字节 token", ALL_ZHONG, ["中", "a中", "中中", "中 a"])
probe("只有 1 个字节 token", ONLY_FIRST, ["中", "中中"])
probe("无字节 token", {}, ["中", "a中"])
probe("字节 token + 该字本身在词表", {**ALL_ZHONG, "中": -1.0}, ["中", "中中", "a中"])
probe("半角空格 / 制表符", ALL_ZHONG, ["中 ", "中\t中"])
# ⚠️ 用户自定义的 `<0x..>` 之外的名字（比如 `&` 前缀）是否也算字节 token ✗
probe("怪名字字节 token", {"<0xZZ>": -5.0}, ["中"])
