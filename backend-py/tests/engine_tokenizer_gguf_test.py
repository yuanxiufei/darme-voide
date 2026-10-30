"""自检：**GGUF 自带词表** → 分词器（``engine/tokenizer_gguf.py`` ＋ ``engine/gguf.read_arrays`` ✓）。

为什么非测不可 ✗：Ollama 库里的模型**只有 GGUF** ✓，词表在 ``tokenizer.ggml.*`` 元数据里 ✓；
而 :func:`gguf.inspect` 对数组**只存前 16 项** ✗✗ ⇒ 「读到了」与「读全了」是两回事 ✓✗。
本文件用**现写的 GGUF 写盘器**造一份 259 项词表的小文件 ✓（不依赖任何真模型 ✓、也不联网 ✓）。

判据（都是「看着能用、其实 id 全错」的形状 ✓✗）：
1. ⭐⭐ **读全**：259 项词表 ⇒ ``vocab_size`` 必须是 259 ✗✗（截断成 16 就是这个 bug 还在 ✗）；
2. ⭐⭐ **预分词规则不许串**：``"1234"`` 在 Qwen2 里是**4 段** ✓、在 GPT-2 里是**1 段** ✗
   （硬套 GPT-2 会**静默编出错 id** ✗✗，不报错 ✓）；
3. ⭐ **无损**：任意文本的预分词结果拼回去必须**逐字等于原文** ✓（丢字符 = 解码后缺字 ✓）；
4. ⭐ **控制符是整体 token**：``<|im_start|>`` 必须编成**一个** id ✓（被字节级拆散 ⇒ 聊天模板失效 ✗✗）；
5. ⭐ **往返**：``decode(encode(中文)) == 原文`` ✓；
6. ⭐ **不加 BOS/EOS**：``add_special_tokens=True`` 与 ``False`` 结果**完全一样** ✓
   （生成侧把 EOS 拼在提示词尾部会直接算错 ✗✗）；
7. ⭐ **具名拒绝**：词表模型 ``llama`` / 预分词 ``llama-bpe`` / 缺 ``token_type`` ⇒ 三条都得报 ✓
   （回落 GPT-2 或静默拆散 = 编出错 id 还看不出来 ✗✗）。

运行::

    python -X utf8 tests/engine_tokenizer_gguf_test.py
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="tokgguf_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import gguf as gguf_mod  # noqa: E402
from app.services.engine import tokenizer_bpe  # noqa: E402
from app.services.engine import tokenizer_gguf  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ── 极简 GGUF 写盘器 ✓（只写 KV、不写张量 ✓ —— 词表测试不需要权重 ✓）
def _str(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(value: object) -> bytes:
    if isinstance(value, bool):                                   # ⚠️ bool 是 int 的子类 ✗ 必须先判 ✓
        return struct.pack("<IB", 7, 1 if value else 0)
    if isinstance(value, int):
        return struct.pack("<II", 4, value)                       # UINT32 ✓
    if isinstance(value, float):
        return struct.pack("<If", 6, value)                       # FLOAT32 ✓
    if isinstance(value, str):
        return struct.pack("<I", 8) + _str(value)                 # STRING ✓
    if isinstance(value, list):
        if not value:                                             # 空数组 ⇒ 按 0 项字符串数组写 ✓
            return struct.pack("<IIQ", 9, 8, 0)
        if isinstance(value[0], str):
            return (struct.pack("<IIQ", 9, 8, len(value))
                    + b"".join(_str(item) for item in value))
        return (struct.pack("<IIQ", 9, 4, len(value))
                + b"".join(struct.pack("<I", int(item)) for item in value))
    raise AssertionError(f"测试写盘器不支持的元数据值：{value!r}")


def write_gguf(path: Path, meta: dict[str, object]) -> None:
    body = b"".join(_str(key) + _value(value) for key, value in meta.items())
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(meta)) + body)


# ── 造一份「像 Qwen 那样」的小词表 ✓
def _toy_vocab() -> tuple[list[str], list[str], list[int]]:
    """256 个字节级 token ＋ 1 条 merge ＋ 2 个控制符 ✓（词表 259 项 ⇒ **远超** inspect 的 16 项截断 ✗）。"""
    mapping = tokenizer_bpe.bytes_to_unicode()
    tokens = [mapping[byte] for byte in range(256)]
    raw = "你".encode("utf-8")                                    # 3 字节 ⇒ 先并前两个 ✓
    merged = mapping[raw[0]] + mapping[raw[1]]
    tokens.append(merged)                                         # id 256 ✓
    merges = [f"{mapping[raw[0]]} {mapping[raw[1]]}"]
    tokens.extend(["<|im_start|>", "<|im_end|>"])                 # id 257 / 258 ✓
    types = [1] * 256 + [1] + [3, 3]
    return tokens, merges, types


def _build(tmp: Path, **override: object) -> Path:
    tokens, merges, types = _toy_vocab()
    meta: dict[str, object] = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.bos_token_id": 257,
        "tokenizer.ggml.eos_token_id": 258,
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.ggml.add_eos_token": False,
    }
    meta.update(override)
    path = tmp / f"toy-{len(meta)}-{abs(hash(tuple(sorted(meta)))) % 10**6}.gguf"
    write_gguf(path, meta)
    return path


def case_read_arrays(tmp: Path) -> None:
    """① 全量读 ✓（这是整个模块的地基 ✗：拿不到正文就什么都编不出 ✓）。"""
    path = _build(tmp)
    meta = gguf_mod.read_arrays(path, ("tokenizer.ggml.tokens", "tokenizer.ggml.merges",
                                       "tokenizer.ggml.token_type"))
    tokens = meta.get("tokenizer.ggml.tokens")
    check("① ⭐⭐ read_arrays 读**全** 259 项（inspect 只给前 16 ✗；截断 ⇒ 这里是 16 ✗✗）",
          isinstance(tokens, list) and len(tokens) == 259, len(tokens) if isinstance(tokens, list) else tokens)
    check("①′ read_arrays 的整数数组也读全（``token_type`` 259 项 ✓）",
          len(meta.get("tokenizer.ggml.token_type") or []) == 259)
    check("①″ 没点名的键**不返回** ✓（不是给空值 ✗）",
          "tokenizer.ggml.model" not in meta)

    hub = tokenizer_gguf.load_gguf_tokenizer(path)
    check("①‴ hub 词表 259 ✓（不走截断源 ✗）", hub.vocab_size == 259, hub.vocab_size)
    check("①⁗ ``required_vocab_size`` = 最大 id + 1 = 259 ✓（含控制符 ✓，拿 vocab_size 建表会越界 ✗）",
          hub.required_vocab_size == 259, hub.required_vocab_size)


def case_suffixless(tmp: Path) -> None:
    """①⁵ **判 GGUF 不许看后缀** ✗✗（Ollama 的文件是 ``blobs\\sha256-<hash>``，一个后缀都没有 ✗）。

    这是真机实测踩到的第二个坑 ✓：按 ``.gguf`` 分派 ⇒ 一路走到 HF 词表装载 ⇒
    报「没有可识别的词表」✗（而权重已经装了两分钟 ✗✗）。
    """
    blob = tmp / "sha256-0123456789abcdef"          # 无后缀 ✓ 就是 Ollama 的形状 ✓
    blob.write_bytes(_build(tmp).read_bytes())
    check("①⁵ ⭐⭐ 无后缀文件按**正文**认成 GGUF ✓（看后缀 ⇒ 认不出 ⇒ 装词表必失败 ✗✗）",
          gguf_mod.looks_like_gguf(blob) is True)
    check("①⁵′ 无后缀也能装出词表 ✓", tokenizer_gguf.load_gguf_tokenizer(blob).vocab_size == 259)
    plain = tmp / "not-a-model.json"
    plain.write_text('{"a": 1}', encoding="utf-8")
    check("①⁵″ 非 GGUF 正文 ⇒ 不认 ✓（别把 JSON 当权重 ✗）",
          gguf_mod.looks_like_gguf(plain) is False)
    check("①⁵‴ 文件不存在 ⇒ 不认 ✓（不抛 ✗）",
          gguf_mod.looks_like_gguf(tmp / "nope.bin") is False)


def case_pretokenize() -> None:
    """② 规则不许串 ✗✗ ＋ ③ 无损 ✓。"""
    qwen_pieces = tokenizer_gguf.pretokenize_qwen2("1234")
    gpt2_pieces = tokenizer_bpe.pretokenize("1234")
    check("② ⭐⭐ Qwen2 的 ``\\p{N}`` 是**单数字** ⇒ ``\"1234\"`` 4 段 ✓",
          qwen_pieces == ["1", "2", "3", "4"], qwen_pieces)
    check("②′ 同一句在 GPT-2 规则下是 **1 段** ✓（两套规则确实不同 ⇒ 串了就是错 id ✗✗）",
          gpt2_pieces == ["1234"], gpt2_pieces)

    check("②″ Qwen2 的 ``\\s*[\\r\\n]+`` 分支：换行前的空格**并入**换行段 ✓",
          tokenizer_gguf.pretokenize_qwen2("a  \r\nb") == ["a", "  \r\n", "b"],
          tokenizer_gguf.pretokenize_qwen2("a  \r\nb"))
    check("②‴ 字母前的可选字符可以是**标点** ✓（GPT-2 的 `` ?`` 只能是空格 ✗）",
          tokenizer_gguf.pretokenize_qwen2("-abc") == ["-abc"],
          tokenizer_gguf.pretokenize_qwen2("-abc"))
    check("②⁗ 空白串尾部**单留一格**给下一段 ✓（``\\s+(?!\\S)`` 的回溯效果 ✓）",
          tokenizer_gguf.pretokenize_qwen2("a  b") == ["a", " ", " b"],
          tokenizer_gguf.pretokenize_qwen2("a  b"))

    samples = ["你好，世界！", "Hello 1234\n\n第二段", "a\tb\t c", "  \n\n ", "emoji 😀 与 ’s 缩写",
               "I'm ok, you're ok", "1234567890", "··—— 标点连打 ……", "多行\r\n换行\r\n见"]
    broken = [text for text in samples if "".join(tokenizer_gguf.pretokenize_qwen2(text)) != text]
    check("③ ⭐ 预分词**无损**：拼回原文逐字相等 ✓（丢字符 ⇒ 解码缺字 ✗）", not broken, broken)


def case_roundtrip(tmp: Path) -> None:
    """④ 控制符整体 ✓ ＋ ⑤ 往返 ✓ ＋ ⑥ 不加 BOS/EOS ✓。"""
    hub = tokenizer_gguf.load_gguf_tokenizer(_build(tmp))
    check("④ ⭐ ``<|im_start|>`` 编成**一个** id = 257 ✓（被字节级拆散 ⇒ 聊天模板失效 ✗✗）",
          hub.encode("<|im_start|>") == [257], hub.encode("<|im_start|>"))
    ids = hub.encode("你")
    check("④′ merge 生效：``你`` = [256(合并), 0xa0 那格] 两段 ✓",
          len(ids) == 2 and ids[0] == 256, ids)
    check("⑤ ⭐ 往返 ``decode(encode(x)) == x`` ✓（中文 ✓）", hub.decode(hub.encode("你好")) == "你好",
          hub.decode(hub.encode("你好")))
    check("⑥ ⭐ ``add_special_tokens`` 对 GGUF 词表**不加** BOS/EOS ⇒ 两种取值结果相同 ✓",
          hub.encode("你好", add_special_tokens=True) == hub.encode("你好", add_special_tokens=False))
    check("⑥′ 控制符解码时被跳过 ✓（``skip_special_tokens=True`` ✓）",
          hub.decode([257]) == "", repr(hub.decode([257])))
    check("⑥″ 自述里写清预分词规则 ✓（排查「id 全错」第一眼看这里 ✓）",
          hub.describe().get("name", "") != "" and
          "qwen2" in str(hub.detail.get("form", {})), hub.detail.get("form"))


def case_refuse(tmp: Path) -> None:
    """⑦ 三条具名拒绝 ✓（回落或静默拆散都会**编出错 id 还看不出来** ✗✗）。"""
    def message_of(path: Path) -> str:
        try:
            tokenizer_gguf.load_gguf_tokenizer(path)
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        return ""

    llama = message_of(_build(tmp, **{"tokenizer.ggml.model": "llama"}))
    check("⑦ ⭐ 词表模型 ``llama``（SPM）⇒ 具名拒绝 ✓（切分规则完全不同 ✗）", "llama" in llama, llama)

    tokens, merges, types = _toy_vocab()
    no_type = message_of(_build(tmp, **{"tokenizer.ggml.token_type": [0]}))
    check("⑦′ 缺 ``token_type``（长度不符 ✓）⇒ 具名拒绝 ✓（认不出控制符 ⇒ 只能拆散 ✗✗）",
          "token_type" in no_type, no_type)

    bad_pre = message_of(_build(tmp, **{"tokenizer.ggml.pre": "llama-bpe"}))
    check("⑦″ 未实现的预分词规则 ``llama-bpe`` ⇒ 具名拒绝 ✓（**不回落 GPT-2** ✗✗）",
          "llama-bpe" in bad_pre, bad_pre)

    no_tokens = message_of(_build(tmp, **{"tokenizer.ggml.tokens": []}))
    check("⑦‴ 没有 ``tokens`` ⇒ 具名拒绝 ✓（不用假桩词表 ✗）", "tokens" in no_tokens, no_tokens)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="tokgguf_") as handle:
        tmp = Path(handle)
        case_read_arrays(tmp)
        case_suffixless(tmp)
        case_pretokenize()
        case_roundtrip(tmp)
        case_refuse(tmp)
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
