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

import io
import struct
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.scripts.corpus import lib_index as li  # noqa: E402
from app.services import library_ingest as lgi  # noqa: E402  (上传口，与命令行同一条 _build)
from app.services import library_overlay as ov  # noqa: E402
from app.services import script_library as sl  # noqa: E402

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


def _minimal_pdf(text: str, *, page_chars: int = 0) -> bytes:
    """造一份 **xref 完整**的最小 PDF（Helvetica / ASCII），用来真测 PDF 通道。

    不借第三方生成器：要验的是「我们的分派能不能把这份文件喂进 pypdfium2 并拿回字」，
    生成器一旦也是 pypdfium2，测的就是它自己。``page_chars=0`` 时正文为空 ⇒ 走
    「N 页都没有文字层」那条分支。
    """
    content = f"BT /F1 12 Tf 20 100 Td ({text * page_chars if page_chars else text}) Tj ET"
    stream = content.encode("latin-1")
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]"
        b"/Resources<</Font<</F1 5 0 R>>>>/Contents 4 0 R>>",
        b"<</Length %d>>stream\n" % len(stream) + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for no, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % no + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, xref_at))
    return bytes(out)


def _pptx(slides: list[str]) -> bytes:
    """造一份只有 ``ppt/slides/slideN.xml`` 的 .pptx（与 WPS/PptxGenJS 同一 schema）。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for no, body in enumerate(slides, start=1):
            zf.writestr(
                f"ppt/slides/slide{no}.xml",
                '<?xml version="1.0"?><p:sld xmlns:p="p" xmlns:a="a"><p:cSld>'
                f"<p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{body}</a:t></a:r></a:p>"
                "</p:txBody></p:sp></p:spTree></p:cSld></p:sld>")
        zf.writestr("[Content_Types].xml", "<Types/>")
    return buf.getvalue()


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

    # ================= PDF 通道（pypdfium2） =================
    with tempfile.TemporaryDirectory(prefix="libidx-pdf-") as td:
        good = Path(td) / "样例.pdf"
        good.write_bytes(_minimal_pdf("Hello PDF channel"))
        text, engine = li._pdf_text(good)
        check("PDF: pypdfium2 真抽出文字层（不是空串）",
              engine == "pypdfium2" and "Hello PDF channel" in text, (engine, text[:40]))
        item = {"path": "样例.pdf", "local": good, "name": "样例.pdf", "extension": ".pdf",
                "size": good.stat().st_size, "modifiedNs": good.stat().st_mtime_ns,
                "signature": li._signature(good.stat().st_size, good.stat().st_mtime_ns)}
        outcome = li._build(item)
        check("PDF: 端到端 _build 判 extra 并切出块",
              outcome["status"] == li.STATUS_EXTRA and len(outcome["chunks"]) >= 1,
              (outcome["status"], outcome["chars"]))
        broken = Path(td) / "坏的.pdf"
        broken.write_bytes(b"%PDF-1.4 garbage with no xref at all")
        try:
            li._pdf_text(broken)
            check("PDF: 打不开必须报可读的错（不许静默返回空串）", False, "没抛异常")
        except RuntimeError as exc:
            check("PDF: 打不开必须报可读的错（不许静默返回空串）",
                  "pypdfium2" in str(exc), exc)

    # PDF 两端对齐会把汉字拆开：这是检索失效，不是排版问题
    check("PDF: 汉字间填充空格被去掉（否则「证人席」搜不到「证 人 席」）",
          li._cjk_deglue("证 人 席") == "证人席", li._cjk_deglue("证 人 席"))
    check("PDF: 中文标点两侧空格也去（句末「。」不该留空）",
          li._cjk_deglue("是我过去未曾想过的 。") == "是我过去未曾想过的。")
    check("PDF: 英文单词间空格一个不动",
          li._cjk_deglue("ROMAN HOLIDAY") == "ROMAN HOLIDAY")
    check("PDF: 中英混排只去夹在汉字/标点旁的那些空格",
          li._cjk_deglue("世界 CINEMA 世界") == "世界 CINEMA 世界",
          li._cjk_deglue("世界 CINEMA 世界"))

    # ================= 用户真正关心的那条链：上传 → 落盘 → 检索命中 =================
    with tempfile.TemporaryDirectory(prefix="libidx-upload-") as up_dir:
        lib_root = Path(up_dir) / "库"
        (lib_root / "99-索引与检索规则").mkdir(parents=True)
        extra_db = Path(up_dir) / "extra" / "library-extra.sqlite"
        saved = (sl.LIBRARY_ROOT, sl.METHODS_DB, sl.OCR_DB, sl.EXTRA_DB,
                 li.EXTRA_DB, ov.OVERLAY_DB)
        try:
            sl.LIBRARY_ROOT = lib_root
            sl.METHODS_DB = lib_root / "99-索引与检索规则" / "没有主库.sqlite"
            sl.OCR_DB = Path(up_dir) / "没有OCR.sqlite"
            sl.EXTRA_DB = extra_db
            li.EXTRA_DB = extra_db
            ov.OVERLAY_DB = Path(up_dir) / "overlay" / "library-overlay.sqlite"
            up = lgi.ingest(_minimal_pdf("INT. HOTEL ROOM - NIGHT. THE COURIER WAITS."),
                            "上传剧本.pdf")
            hits = sl.search_methods("COURIER")
        finally:
            (sl.LIBRARY_ROOT, sl.METHODS_DB, sl.OCR_DB, sl.EXTRA_DB,
             li.EXTRA_DB, ov.OVERLAY_DB) = saved
        check("上传: 有文字层的 PDF 判 ingested（不再一律拒绝）",
              up["status"] == "ingested", up)
        check("上传: 落的文件名带 .pdf、正文块真进库",
              up["path"].endswith(".pdf") and up.get("chunks", 0) >= 1, up)
        rows = [row for row in hits.get("results") or [] if row.get("from") == "extra"]
        check("上传: 刚传的 PDF 立刻能被检索到（在 extra 那一路上命中）",
              bool(rows), (hits.get("parts"), hits.get("hint")))

    # ================= HTML / PPTX 通道 =================
    with tempfile.TemporaryDirectory(prefix="libidx-markup-") as td:
        page = Path(td) / "剧本.html"
        page.write_text(
            "<html><head><style>p{color:red}</style><script>var x = 1</script></head>"
            "<body><h1>ROMAN HOLIDAY</h1><p>第一场 &amp; 第二场</p>"
            "<div>INT. HOTEL — DAY</div></body></html>", encoding="utf-8")
        text, engine = li._html_text(page)
        check("HTML: 剥标签后保留正文", "ROMAN HOLIDAY" in text and "INT. HOTEL" in text,
              text[:60])
        check("HTML: script/style 整段丢（否则样式表会进检索）",
              "color:red" not in text and "var x" not in text, text[:60])
        check("HTML: 实体被解码", "&" in text and "&amp;" not in text, text[:80])

        inline = Path(td) / "行内标签.html"
        inline.write_text("<p>INT. HOTEL — <i>DAY</i></p><p>甲  乙</p>", encoding="utf-8")
        glued, _enc = li._html_text(inline)
        check("HTML: 行内标签两侧的空格保住（否则英文剧本整篇词粘连）",
              "INT. HOTEL — DAY" in glued, glued)
        check("HTML: 行内多余空白压成一个（全角空格也压）", "甲 乙" in glued, glued)
        check("HTML: 引擎名带编码（台账要能读出版式来源）",
              engine.startswith("html-"), engine)
        html_item = {"path": "剧本.html", "local": page, "name": "剧本.html",
                     "extension": ".html", "size": page.stat().st_size,
                     "modifiedNs": page.stat().st_mtime_ns,
                     "signature": li._signature(page.stat().st_size, page.stat().st_mtime_ns)}
        html_out = li._build(html_item)
        joined = " ".join(chunk[2] for chunk in html_out["chunks"])  # (page, chunk_no, text)
        check("HTML: 端到端 _build 判 extra，且切出的块里能搜到原文（不是只抽给日志看）",
              html_out["status"] == li.STATUS_EXTRA and "ROMAN HOLIDAY" in joined,
              (html_out["status"], html_out["chars"], joined[:60]))

        deck = Path(td) / "讲座.pptx"
        deck.write_bytes(_pptx(["第一页 剧本结构", "第二页 三幕式"]))
        text, engine = li._pptx_text(deck)
        check("PPTX: 按 slide 号顺序抽 <a:t>",
              engine == "pptx-xml" and text.index("第一页") < text.index("第二页"),
              repr(text[:60]))
        attr = Path(td) / "带属性.pptx"
        with zipfile.ZipFile(attr, "w") as zf:  # 手写 slide：夹具默认把正文套进 <a:t> 了
            zf.writestr("ppt/slides/slide1.xml",
                        '<?xml version="1.0"?><p:sld xmlns:p="p" xmlns:a="a"><a:p><a:r>'
                        '<a:t xml:space="preserve">保留属性 &amp; 实体</a:t>'
                        "</a:r></a:p></p:sld>")
        attr_text, _engine = li._pptx_text(attr)
        check("PPTX: 容得下 <a:t xml:space=…> 并解码实体（产出方不是只有 PowerPoint）",
              "保留属性 & 实体" in attr_text, repr(attr_text))
        silent = Path(td) / "纯图.pptx"
        silent.write_bytes(_pptx([]))
        try:
            li._pptx_text(silent)
            check("PPTX: 抽不出字要如实报错（不装作入库）", False, "没抛异常")
        except RuntimeError as exc:
            check("PPTX: 抽不出字要如实报错（不装作入库）", "slide" in str(exc), exc)

    # ================= 上传白名单与通道同源 =================
    check("上传: 白名单吃下 pdf/html/pptx（此前 PDF 直接被拒）",
          {".pdf", ".html", ".htm", ".pptx"} <= set(lgi.ALLOWED_SUFFIXES),
          lgi.ALLOWED_SUFFIXES)
    check("上传: 提示串反映新类型",
          "PDF" in lgi.ALLOWED_LABEL and "PPTX" in lgi.ALLOWED_LABEL, lgi.ALLOWED_LABEL)
    check("口径: 拒绝文案不再谎称「PDF 走扫描件 OCR 那条线」",
          "PDF 走扫描件 OCR" not in Path(lgi.__file__).read_text(encoding="utf-8"))
    check("口径: HANDLED_ELSEWHERE 不再宣称 PDF 不归本脚本",
          ".pdf" not in li.HANDLED_ELSEWHERE, list(li.HANDLED_ELSEWHERE))

    # 扫描件 PDF 会落到 quarantine，唯一补救路是 OCR 线的待办集合 —— 拿掉就成死件
    import ast

    ocr_src = Path(__file__).resolve().parents[1] / "app/scripts/corpus/ocr_index.py"
    pipeline = ast.parse(ocr_src.read_text(encoding="utf-8"))
    ocr_suffixes: list[str] = []
    for node in ast.walk(pipeline):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", "") == "SUPPORTED_SUFFIXES" for t in node.targets):
            continue
        if isinstance(node.value, ast.Tuple):
            ocr_suffixes = [e.value for e in node.value.elts
                            if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    check("PDF: 抽不出字的件必须落在 OCR 线接手集合里（否则成死件）",
          ".pdf" in ocr_suffixes, ocr_suffixes)

    # ================= 检索层：fts5 上 path 过滤不能用 LIKE =================
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

    # ================= 扫库白名单：与抽文本通道同一个真源 =================
    #: 2026-11-07 事故原样：``_extract`` 已能抽 ``.pdf``，``_candidates`` 的白名单却还是
    #: 手抄的旧四组 ⇒ 批量扫库把 19 份 PDF ＋ 6 份 HTML 记成「未登记抽文本通道」静默跳过，
    #: 只有界面上传那条路能用。这里从两头钉死：同一个真源 ＋ 逐个扩展名真被收。
    check("口径: 上传白名单就是 lib_index.SUPPORTED_EXTS（同一真源，不是两份加法）",
          tuple(lgi.ALLOWED_SUFFIXES) == tuple(li.SUPPORTED_EXTS),
          (lgi.ALLOWED_SUFFIXES, li.SUPPORTED_EXTS))
    with tempfile.TemporaryDirectory(prefix="libidx-scan-") as scan_dir:
        lib = Path(scan_dir) / "扫描库"
        lib.mkdir()
        for ext in li.SUPPORTED_EXTS:
            (lib / f"样本{ext}").write_bytes(b"x")
        (lib / "样本.epub").write_bytes(b"x")
        (lib / "样本.zap").write_bytes(b"x")
        saved = (sl.LIBRARY_ROOT, sl.METHODS_DB, sl.OCR_DB, li.EXTRA_DB)
        try:
            sl.LIBRARY_ROOT = lib
            sl.METHODS_DB = Path(scan_dir) / "没有主库.db"
            sl.OCR_DB = Path(scan_dir) / "没有OCR.db"
            li.EXTRA_DB = Path(scan_dir) / "没有第三库.db"
            got, why = li._candidates()
        finally:
            sl.LIBRARY_ROOT, sl.METHODS_DB, sl.OCR_DB, li.EXTRA_DB = saved
        picked = sorted(row["extension"] for row in got)
        check("扫库: SUPPORTED_EXTS 里每个扩展名都真被当候选（不是只在 _extract 里认）",
              picked == sorted(li.SUPPORTED_EXTS), (picked, why))
        check("扫库: 登记在 HANDLED_ELSEWHERE 的扩展名报的是「为什么不收」，不是「不认识」",
              any(key.startswith(".epub ") and "未登记" not in key for key in why), why)
        check("扫库: 彻底不认识的扩展名也要报出来（静默跳过＝永远查不出漏了）",
              any(".zap 未登记抽文本通道" in key for key in why), why)

    # ================= prune 那边的同一份手抄名单 =================
    #: `prune` 判「扩展名不在通道内」时也曾手抄一份旧四组 ⇒ 一旦真跑 `--apply`，
    #: 刚入库的 PDF/HTML/PPTX 记录会被当成脏数据**删掉**（比"不收录"更坏：明明成功了又消失）。
    with tempfile.TemporaryDirectory(prefix="libidx-prune-") as prune_dir:
        pdir = Path(prune_dir)
        lib2 = pdir / "库"
        (lib2 / "子目录").mkdir(parents=True)
        (lib2 / "子目录" / "剧本.pdf").write_bytes(b"x")
        (lib2 / "子目录" / "清单.csv").write_text("a,b", encoding="utf-8")
        saved = (sl.LIBRARY_ROOT, sl.METHODS_DB, sl.OCR_DB, li.EXTRA_DB)
        try:
            sl.LIBRARY_ROOT = lib2
            li.EXTRA_DB = pdir / "extra.sqlite"
            conn = li._connect(writable=True)
            for rel, ext in (("子目录/剧本.pdf", ".pdf"), ("子目录/清单.csv", ".csv")):
                conn.execute(
                    "INSERT INTO documents (path, name, extension, size, modified_ns,"
                    " signature, pages, status) VALUES (?, ?, ?, 1, 1, 's', 1, 'extra')",
                    (rel, Path(rel).name, ext))
            conn.commit()
            conn.close()
            report = li.prune(apply=False)
        finally:
            sl.LIBRARY_ROOT, sl.METHODS_DB, sl.OCR_DB, li.EXTRA_DB = saved
        doomed = [rel for rel, _why in report["removed"]]
        check("prune: 新登记的 .pdf 记录不许被当成「扩展名不在通道内」清掉（真跑 --apply 就删库）",
              "子目录/剧本.pdf" not in doomed, report["removed"])
        check("prune: 当初误收的 .csv 仍要被报出来（清理能力没退化）",
              "子目录/清单.csv" in doomed, report["removed"])

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
