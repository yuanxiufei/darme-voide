"""第三库抽文本层自检：三类「跑得过、但结果不对」的逻辑。

1. **OLE2 解析**（``_ole_stream``）：FAT 链、目录项、越界与环。真文件里 WPS 写的
   Word 会把 FAT 指回自己，实测一份 116KB 的文档让解析器原地转死（CPU 100%、不出错、
   也不结束）；现在靠 ``max_sectors`` 上限 + ``seen`` 集合收口，这里要用**自指 FAT**
   与**自指 DIFAT** 反向证明它真能停。
2. **码表**（``_decode_compressed`` / ``_rtf_text``）：Word 压缩分片按 FIB ``lid``、
   RTF ``\\'hh`` 按当前字体 ``\\fcharset``。cp936 与 cp949 的字节区间几乎重叠，
   两边都能"成功"解出字 —— 错的那边是**可打印的乱码**，可打印率闸门拦不住，
   只能靠逐字断言。
3. **可打印率**（``_printable_ratio``）：``str.isprintable()`` 对 ``U+3000`` 返回
   ``False``，而中文剧本整段靠全角空格缩进 ⇒ 一份完好的剧本被判 0.83 分数、一块不写。
   实测因此误隔离 37 份。

运行::

    ./.venv/Scripts/python.exe tests/lib_index_test.py
"""
from __future__ import annotations

import struct
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.scripts.corpus import lib_index as li  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
SECTOR = 512


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ---------------------------------------------------------------------------
# 最小 OLE2 构造函数（全部走普通 FAT：ministart 显式置越界，见 _ole_stream 的分支条件）
# ---------------------------------------------------------------------------
def _dir_entry(name: str, kind: int, start: int, size: int) -> bytes:
    slot = bytearray(128)
    raw = name.encode("utf-16-le")
    slot[:len(raw)] = raw
    struct.pack_into("<H", slot, 0x40, len(raw) + 2)
    slot[0x42] = kind
    struct.pack_into("<I", slot, 0x74, start)
    struct.pack_into("<I", slot, 0x78, size)
    return bytes(slot)


def _ole(streams: list[tuple[str, bytes]], *, self_ref: bool = False) -> bytes:
    """FAT(0) + 目录(1) + 各流数据(从 2 起)。``self_ref`` 让流链指回自己。"""
    fat = [0xFFFFFFFF] * 128
    fat[0] = 0xFFFFFFFD       # FAT 扇区
    fat[1] = 0xFFFFFFFE       # 目录只有一扇区
    entries = [_dir_entry("Root Entry", 5, 0xFFFFFFFE, 0)]
    body = b""
    cursor = 2
    for name, data in streams:
        n = max(1, (len(data) + SECTOR - 1) // SECTOR)
        for k in range(n):
            fat[cursor + k] = cursor if self_ref else (
                cursor + k + 1 if k < n - 1 else 0xFFFFFFFE)
        entries.append(_dir_entry(name, 2, cursor, len(data)))
        body += data.ljust(n * SECTOR, b"\x00")
        cursor += n
    while len(entries) < 4:
        entries.append(bytes(128))
    head = bytearray(512)
    head[0:8] = li._OLE_MAGIC
    struct.pack_into("<HH", head, 0x1E, 9, 6)          # sect_shift=9, mini_shift=6
    struct.pack_into("<II", head, 0x2C, 1, 1)          # nfat=1, dirstart=1
    struct.pack_into("<I", head, 0x38, 4096)           # mini cutoff
    struct.pack_into("<III", head, 0x3C, 0xFFFFFFFE, 0xFFFFFFFE, 0)
    struct.pack_into("<I", head, 0x4C, 0)              # DIFAT[0] = FAT 在扇区 0
    return bytes(head) + struct.pack("<128I", *fat) + b"".join(entries) + body


def _word_ole(text: bytes, *, lid: int, compressed: bool = True,
              self_ref: bool = False) -> bytes:
    """最小 Word：FIB + 1Table 里的单分片 Pcdt。"""
    word = bytearray(0x400)
    struct.pack_into("<H", word, 0x00, 0xA5EC)
    struct.pack_into("<H", word, 0x06, lid)            # 语言 id ⇒ 码表
    struct.pack_into("<H", word, 0x0A, 0x0200)         # fWhichTblStm=1 ⇒ 1Table
    at = 0x300
    word[at:at + len(text)] = text
    chars = len(text) if compressed else len(text) // 2
    fc = ((at * 2) | 0x40000000) if compressed else at
    plc = (struct.pack("<II", 0, chars)
           + struct.pack("<H", 0) + struct.pack("<I", fc) + struct.pack("<H", 0))
    clx = b"\x02" + struct.pack("<I", len(plc)) + plc
    table = bytearray(max(len(clx), 16))
    table[:len(clx)] = clx
    struct.pack_into("<II", word, 0x01A2, 0, len(clx))
    return _ole([("WordDocument", bytes(word)), ("1Table", bytes(table))],
                self_ref=self_ref)


def _in_thread(fn, timeout: float = 3.0):
    box: dict[str, object] = {}
    worker = threading.Thread(target=lambda: box.setdefault("v", fn()), daemon=True)
    worker.start()
    worker.join(timeout)
    return box.get("v", "__TIMEOUT__")


def main() -> int:  # noqa: C901
    # ================= 可打印率：全角空格不是坏字符 =================
    check("可打印率: 全角空格与不换行空格都算可打印（误隔离 37 份的根因）",
          li._printable_ratio("甲\u3000\u3000乙\u00a0丙") == 1.0,
          li._printable_ratio("甲\u3000\u3000乙\u00a0丙"))
    check("可打印率: 控制符扣分", abs(li._printable_ratio("ab\x01") - 2 / 3) < 1e-9)
    check("可打印率: U+FFFD 计入坏字符（isprintable 对它为 True，不能靠它）",
          abs(li._printable_ratio("a\ufffdb") - 2 / 3) < 1e-9)
    check("可打印率: 空串 0.0", li._printable_ratio("") == 0.0)

    # ================= 入库前规范化 =================
    check("规范化: 三种换行统一成 \\n", li._tidy_text("a\r\nb\rc") == "a\nb\nc")
    check("规范化: nbsp→空格、零宽空格删除、控制符删除",
          li._tidy_text("x\xa0y\u200bz\x01w") == "x yzw", repr(li._tidy_text("x\xa0y\u200bz\x01w")))
    check("规范化: 三行以上空行压成一行空行", li._tidy_text("a\n\n\n\nb") == "a\n\nb")

    # ================= Word 压缩分片：码表跟 lid 走 =================
    raw = "剧本正文".encode("cp936")
    check("码表: codepage 生效（cp936）", li._decode_compressed(raw, "cp936") == "剧本正文")
    check("码表: 同一份字节按 cp949 解就不是原文（cp936/cp949 区间重叠，必须靠 lid）",
          li._decode_compressed(raw, "cp949") != "剧本正文")
    check("码表: 无 codepage 时纯 ASCII 原样", li._decode_compressed(b"plain text") == "plain text")
    check("码表: lid 表有中文/韩文/日文三档",
          (li._LID_CP.get(0x0804), li._LID_CP.get(0x0412), li._LID_CP.get(0x0411))
          == ("cp936", "cp949", "cp932"))

    # ================= OLE2：正常取流 =================
    payload = b"Word-Document-" * 500          # > 4096 ⇒ 普通 FAT，跨 24 个扇区
    stream = li._ole_stream(_ole([("WordDocument", payload)]), "WordDocument")
    check("OLE: 跨扇区取流完整", stream == payload, len(stream or b""))
    check("OLE: 没有这条流就 None",
          li._ole_stream(_ole([("WordDocument", payload)]), "0Table") is None)

    # ================= OLE2：畸形/环 =================
    check("OLE: 签名不对 → None", li._ole_stream(b"not ole at all", "WordDocument") is None)
    bad_shift = bytearray(_ole([("WordDocument", payload)]))
    struct.pack_into("<H", bad_shift, 0x1E, 8)          # sect_shift=8 非法
    check("OLE: 头字段非法 → None（不拿歪字段往下走）",
          li._ole_stream(bytes(bad_shift), "WordDocument") is None)
    check("OLE: FAT 自指必须停（真文件里 WPS 会这么写，旧实现原地转死）",
          _in_thread(lambda: li._ole_stream(
              _ole([("WordDocument", payload)], self_ref=True), "WordDocument")) != "__TIMEOUT__")
    looped = bytearray(_ole([("WordDocument", payload)]))
    struct.pack_into("<I", looped, 0x4C, 1)             # DIFAT[0]=1 ⇒ 扇区 1 自指
    struct.pack_into("<II", looped, 0x44, 1, 1)         # difstart=1, ndif=1
    check("OLE: DIFAT 自指必须停", _in_thread(
        lambda: li._ole_stream(bytes(looped), "WordDocument")) != "__TIMEOUT__")

    # ================= FIB 直读：压缩/非压缩 + 码表 =================
    check("FIB: cp936 压缩分片解出中文",
          li._doc_ole_text(_word_ole("剧本正文".encode("cp936"), lid=0x0804)) == "剧本正文")
    check("FIB: 同一份字节 lid=0x0412 解出韩文而非中文（码表真按 lid）",
          li._doc_ole_text(_word_ole("剧本正文".encode("cp936"), lid=0x0412)) != "剧本正文")
    utf16 = "非压缩分片".encode("utf-16-le")
    check("FIB: 非压缩分片按 UTF-16LE 解",
          li._doc_ole_text(_word_ole(utf16, lid=0x0804, compressed=False)) == "非压缩分片")
    no_word = _ole([("SomethingElse", b"x" * 64)])
    try:
        li._doc_ole_text(no_word)
        check("FIB: 缺 WordDocument 流要报可读的错", False, "没抛错")
    except RuntimeError as exc:
        check("FIB: 缺 WordDocument 流要报可读的错", "WordDocument" in str(exc), exc)

    # ================= RTF =================
    def rtf(body: str, *, cpg: int = 936,
            fonts: str = r"{\f0\fnil\fcharset134 SimSun;}") -> bytes:
        """RTF 是字节流：非 ASCII 一律写成 ``\\'hh``（默认字体是 fcharset134 ⇒ cp936）。"""
        escaped = "".join(
            ch if ord(ch) < 128 else "".join(f"\\'{b:02x}" for b in ch.encode("cp936"))
            for ch in body)
        text = r"{\rtf1\ansi\ansicpg%d{\fonttbl%s}%s}" % (cpg, fonts, escaped)
        return text.encode("latin-1")

    check("RTF: \\uc1 后的回退字符被吃掉（正文不能多出问号）",
          li._rtf_text(rtf(r"\uc1\u20013?")).strip() == "中",
          repr(li._rtf_text(rtf(r"\uc1\u20013?"))))
    two_fonts = r"{\f0\fnil\fcharset134 SimSun;}{\f1\fnil\fcharset129 Batang;}"
    mixed = li._rtf_text(rtf(r"\f0 \'d6\'d0 \f1 \'c7\'d1 ", fonts=two_fonts))
    check("RTF: 按 \\fcharset 分字体解（中韩混排各用各的码表）",
          "中" in mixed and "한" in mixed, repr(mixed))
    field = li._rtf_text(rtf(r"{\field{\*\fldinst HYPERLINK \"x\"}{\fldrslt 正文}}"))
    check("RTF: field 里保留 fldrslt 正文、丢掉 fldinst 指令",
          "正文" in field and "HYPERLINK" not in field, repr(field))
    meta = li._rtf_text(rtf(r"{\*\generator Microsoft Word;}{\info{\title 元}}正文"))
    check("RTF: 可忽略 destination（generator/info）不进正文",
          "Microsoft" not in meta and "正文" in meta, repr(meta))

    # ================= _doc_text：先认内容再认扩展名 =================
    with tempfile.TemporaryDirectory(prefix="libidx-") as td:
        fake = Path(td) / "其实是rtf.doc"
        fake.write_bytes(rtf(r"伪装成 doc 的 RTF"))
        text, engine = li._doc_text(fake)
        check("分派: RTF 内容存成 .doc 也走内置 RTF 解", engine == "rtf" and "伪装" in text,
              (engine, text[:20]))
        garbage = Path(td) / "真的不是.doc"
        garbage.write_bytes(b"hello")
        try:
            li._doc_text(garbage)
            check("分派: 既非 OLE 又非 RTF 要如实报头字节", False, "没抛错")
        except RuntimeError as exc:
            check("分派: 既非 OLE 又非 RTF 要如实报头字节",
                  "68656c6c6f" in str(exc), exc)

    # ================= 检索层：fts5 上 path 过滤不能用 LIKE =================
    from app.services import script_library as sl
    clauses, params = sl._path_filters("Chinatown")
    check("检索: path 过滤用 instr（fts5 上 3 字符 ASCII LIKE 会静默返回 0 行）",
          all("instr(path" in c for c in clauses) and "Chinatown" in params, (clauses, params))
    check("检索: 隔离目录过滤也是 instr",
          any("instr(path, ?) = 0" in c for c in clauses) and "90-隔离与待审" in params,
          (clauses, params))
    src = Path(sl.__file__).read_text(encoding="utf-8")
    body = src[src.index("def _path_filters"):src.index("def ", src.index("def _path_filters") + 10)]
    appends = [line.strip() for line in body.splitlines() if "clauses.append(" in line]
    check("检索: _path_filters 的 SQL 片段只剩 instr（注释里的历史写法不算）",
          len(appends) == 2 and all("instr" in line for line in appends), appends)

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
