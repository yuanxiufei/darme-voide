"""扫描件 OCR 的 **docx 那一半**：整页扫描图贴进 Word 的文档也要能补出正文。

为什么单独钉这一套：``ocr_index`` 原先只认 ``.pdf``，代码注释把盘上那 2 份 docx 写成
「坏文件」——2026-11-04 实测它们是**正文层 0 字、内容全在 `word/media/*` 的几十张图里**
（《撕夜》全剧本 43 页图 / 开端 49 页图）。若不改，这两份会永远挂在「待补」上，
而且没有任何报错提示这条路根本没走过。

判据只看**可确定**的东西（不加载 rapidocr）：
* 取图顺序 = ``word/document.xml`` 里 ``<a:blip r:embed>`` 的顺序（＝阅读顺序），不是 zip 名顺序；
* rels 缺失/外链图必须**跳过而不是抛错**（一份文档 40 多张图，坏一张不该整份失败）；
* 小图（页眉徽标）不算页；空白页计入 ``blank`` 且不产块；
* 落库的 ``extension`` 跟着来源走（不许写死 ``.pdf``），``--pages`` 试跑标 ``partial``。

运行::

    ./.venv/Scripts/python.exe tests/ocr_index_test.py
"""
from __future__ import annotations

import io
import os
import sqlite3
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

BACKEND_PY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_PY))

from app.scripts.corpus import ocr_index as oi  # noqa: E402
from app.scripts.corpus import lib_index as li  # noqa: E402
from app.services import script_library as sl  # noqa: E402

CHECKS = 0
FAILED = 0


def check(label: str, ok: bool, detail=None) -> None:
    global CHECKS, FAILED
    CHECKS += 1
    if ok:
        print(f"  ✓ {label}")
    else:
        FAILED += 1
        print(f"  ✗ {label}")
        if detail is not None:
            print(f"      {detail}")


def png(width: int, height: int) -> bytes:
    from PIL import Image  # noqa: PLC0415 —— 与 ocr_index 同一条依赖

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def build_docx(path: Path, blips: list[str], rels: dict[str, str],
               media: dict[str, bytes]) -> Path:
    """造一份最小 docx：``blips`` 是正文里出现的 rId 顺序，``rels`` 是 rId→包内路径。"""
    body = "".join(f'<a:blip r:embed="{rid}"/>' for rid in blips)
    rel_xml = "".join(f'<Relationship Id="{rid}" Target="{target}"/>'
                      for rid, target in rels.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml",
                    f'<w:document><w:body><w:drawing>{body}</w:drawing></w:body></w:document>')
        if rel_xml:
            zf.writestr("word/_rels/document.xml.rels", f"<Relationships>{rel_xml}</Relationships>")
        for name, data in media.items():
            zf.writestr(name, data)
    return path


class FakeOCR:
    """替身 OCR：按调用次序吐预设文字（空串＝空白页），不加载 onnxruntime。"""

    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls = 0

    def __call__(self, image):
        text = self.texts[self.calls] if self.calls < len(self.texts) else ""
        self.calls += 1
        return ([[[0, 0, 0, 0], text, 0.9]] if text else []), 0.9


def case_images() -> None:
    print("\n① 取图顺序与兜底")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        page1, page2, icon = png(600, 800), png(640, 900), png(60, 60)
        docx = build_docx(
            root / "a.docx",
            blips=["rId2", "rId1"],  #: 正文里 rId2 在前（zip 名顺序恰好相反）
            rels={"rId1": "media/image1.png", "rId2": "media/image2.png"},
            media={"word/media/image1.png": page1, "word/media/image2.png": page2},
        )
        got = oi._docx_images(docx)
        check("取图按正文出现顺序（rId2 先于 rId1），不是 zip 名顺序",
              got == [page2, page1], [len(blob) for blob in got])

        no_rels = build_docx(
            root / "b.docx", blips=["rId9"],
            rels={},
            media={"word/media/b.png": page1, "word/media/a.png": page2},
        )
        check("rels 缺失 ⇒ 兜底按 media 名排序，且不抛错",
              oi._docx_images(no_rels) == [page2, page1])

        external = build_docx(
            root / "c.docx", blips=["rId1", "rId2"],
            rels={"rId1": "http://example.com/x.png", "rId2": "media/ok.png"},
            media={"word/media/ok.png": icon},
        )
        check("外链图（Target 是 http）跳过，包内那张照取",
              oi._docx_images(external) == [icon])


def case_docx_ocr() -> None:
    print("\n② docx 的页数/空白/试跑口径")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        big1, big2, big3 = png(600, 800), png(600, 800), png(600, 800)
        docx = build_docx(
            root / "知识库" / "08-编剧方法书籍" / "整页图.docx",
            blips=["rId1", "rId2", "rId3", "rId4"],
            rels={"rId1": "media/1.png", "rId2": "media/2.png",
                  "rId3": "media/3.png", "rId4": "media/4.png"},
            media={"word/media/1.png": big1, "word/media/2.png": png(80, 80),
                   "word/media/3.png": big2, "word/media/4.png": big3},
        )
        item = {"path": f"G:\\电影剧本\\{sl.LOCAL_MARKER}\\08-编剧方法书籍\\整页图.docx",
                "rel": "08-编剧方法书籍/整页图.docx", "name": "整页图.docx",
                "extension": ".docx"}

        original = sl.LIBRARY_ROOT
        sl.LIBRARY_ROOT = root / "知识库"
        try:
            body = "第一页正文。" * 40
            ocr = FakeOCR([body, "", "第三页正文。" * 40])
            outcome = oi._ocr_docx(item, ocr, max_pages=None)
            check("页数只数「够大」的图（60×60 的徽标不算一页 ⇒ 3 页不是 4 页）",
                  outcome["pages"] == 3 and outcome["scanned"] == 3, outcome["pages"])
            check("空白页计入 blank、不产块（块里没有空串）",
                  outcome["blank"] == 1
                  and all(text.strip() for _page, _no, text in outcome["chunks"]),
                  outcome["blank"])
            check("正文进了块，且块序从 1 连续编号",
                  "第一页正文。" in outcome["chunks"][0][2]
                  and [no for _page, no, _text in outcome["chunks"]]
                  == list(range(1, len(outcome["chunks"]) + 1)),
                  [no for _page, no, _text in outcome["chunks"]])

            trial = oi._ocr_docx(item, FakeOCR([body]), max_pages=1)
            check("--pages 试跑：scanned=1 < pages=3（⇒ 落库标 partial，不算整本做完）",
                  (trial["scanned"], trial["pages"]) == (1, 3),
                  (trial["scanned"], trial["pages"]))
        finally:
            sl.LIBRARY_ROOT = original


def case_native() -> None:
    """PDF 自己的文字层要先用：OCR 是最贵的路，只有真没字时才走。"""
    print("\n③ 有文字层的页不 OCR（混合文档）")
    import fitz  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdf = root / "知识库" / "58-短剧待精筛" / "混排.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        try:
            #: ⚠️ 字数必须过 NATIVE_TRUST_CHARS（100）：这条判据是 2026-11-04 为「水印型
            #: 扫描件」加的，样本只写 92 字时会被当水印送去 OCR —— 那时本用例测的就不再是
            #: 「有文字层就不 OCR」，而是「字太少要 OCR」，红得莫名其妙。
            native_text = "NATIVE TEXT LAYER ON PAGE ONE, NO OCR NEEDED. " * 6
            first = doc.new_page(width=595, height=842)
            first.insert_text((60, 90), native_text, fontsize=9)
            second = doc.new_page(width=595, height=842)
            second.insert_image(fitz.Rect(0, 0, 595, 842), stream=png(600, 800))
            doc.save(str(pdf))
        finally:
            doc.close()

        item = {"path": f"G:\\电影剧本\\{sl.LOCAL_MARKER}\\58-短剧待精筛\\混排.pdf",
                "rel": "58-短剧待精筛/混排.pdf", "name": "混排.pdf", "extension": ".pdf"}
        original = sl.LIBRARY_ROOT
        sl.LIBRARY_ROOT = root / "知识库"
        try:
            ocr = FakeOCR(["SCANNED PAGE TWO TEXT VIA OCR ENGINE. " * 2])
            outcome = oi._ocr_one(item, ocr, dpi=96, max_pages=None)
            check("有文字层的页记进 native（2 页里 1 页没走 OCR）",
                  outcome["native"] == 1 and outcome["scanned"] == 2, outcome)
            check("真没字的那页才调引擎（不是整份都 OCR）",
                  ocr.calls == 1, ocr.calls)
            merged = "".join(text for _page, _no, text in outcome["chunks"])
            check("两种来源的正文都进了块（原生页与 OCR 页都不丢）",
                  "NATIVE TEXT" in merged and "SCANNED PAGE" in merged, merged[:80])

            #: 水印型：文字层只有三十来字（页眉水印＋日期）⇒ **不算**原生，必须送 OCR。
            #: 这条钉的是 ``NATIVE_TRUST_CHARS`` —— 没有它，168 份 / 27888 页水印型扫描件
            #: 会被当成「有文字层」白拿 35 字水印入库，正文一个字都进不去。
            watermarked = root / "知识库" / "58-短剧待精筛" / "水印.pdf"
            doc = fitz.open()
            try:
                page = doc.new_page(width=595, height=842)
                page.insert_text((60, 90), "CENTRAL ACADEMY OF DRAMA  07/26/2014", fontsize=9)
                page.insert_image(fitz.Rect(0, 0, 595, 842), stream=png(600, 800))
                doc.save(str(watermarked))
            finally:
                doc.close()
            watermark_item = {"path": f"G:\\电影剧本\\{sl.LOCAL_MARKER}\\58-短剧待精筛\\水印.pdf",
                              "rel": "58-短剧待精筛/水印.pdf", "name": "水印.pdf",
                              "extension": ".pdf"}
            watermark_ocr = FakeOCR(["扫描图里的真正文。" * 20])
            outcome = oi._ocr_one(watermark_item, watermark_ocr, dpi=96, max_pages=None)
            check("水印型文字层（三十来字）不算原生，整页送 OCR",
                  outcome["native"] == 0 and watermark_ocr.calls == 1, outcome)
        finally:
            sl.LIBRARY_ROOT = original


def case_engine() -> None:
    """两代 rapidocr 的回执形状不同，接错了会「每页都识别成空白」而不报错。"""
    print("\n④ 引擎档位、回执形状与线程口径")
    import importlib.util  # noqa: PLC0415
    import os  # noqa: PLC0415

    original = os.environ.get("OCR_ENGINE")
    try:
        os.environ["OCR_ENGINE"] = oi.ENGINE_OLD
        check("显式钉老档 ⇒ engine_name() 就报老档（不建 session 也答得出来）",
              oi.engine_name() == oi.ENGINE_OLD, oi.engine_name())
        os.environ.pop("OCR_ENGINE", None)
        installed = importlib.util.find_spec("rapidocr") is not None
        check("auto：装得上新档就优先新档，装不上退回老档（不许抛异常）",
              oi.engine_name() == (oi.ENGINE_NEW if installed else oi.ENGINE_OLD),
              (installed, oi.engine_name()))
    finally:
        if original is None:
            os.environ.pop("OCR_ENGINE", None)
        else:
            os.environ["OCR_ENGINE"] = original

    class ModernOutput:
        #: 真跑时 ``boxes`` 是 **numpy 数组**，不是 list —— 用 list 当替身就试不出
        #: 「``x or ()`` 对数组求布尔 ⇒ ValueError」（这个坑正是真跑第一份才炸的）。
        boxes = np.zeros((1, 4, 2), dtype=np.float32)
        txts = ("新档识别出来的正文。",)
        scores = np.array([0.93], dtype=np.float32)
        elapse = 0.12

    class Modern:
        def __call__(self, image):
            return ModernOutput()

    class ModernBlank:
        boxes = None
        txts = None
        scores = None
        elapse = 0.01

        def __call__(self, image):
            return self

    lines, elapse = oi._ModernEngine(Modern())(image=None)
    check("新档回执被拍成老档的三列形状（[框, 文本, 分]，numpy 数组也不许炸）",
          len(lines) == 1 and lines[0][1] == "新档识别出来的正文。" and elapse == 0.12,
          (lines, elapse))
    check("新档「一张字都没检出」时给空列表，不许抛（插图页很常见）",
          oi._ModernEngine(ModernBlank())(image=None)[0] == [])

    original_threads = os.environ.get("OCR_THREADS")
    original_batch = os.environ.get("OCR_REC_BATCH")
    try:
        os.environ["OCR_THREADS"] = "5"
        check("OCR_THREADS 能钉死每进程线程数（做对照实验用）",
              oi.engine_threads(8) == 5, oi.engine_threads(8))
        os.environ.pop("OCR_THREADS")
        check("按进程数摊线程、单进程封顶 16（32 线程实测反而更慢）",
              1 <= oi.engine_threads(8) <= 16 and 1 <= oi.engine_threads(1) <= 16,
              (oi.engine_threads(1), oi.engine_threads(8)))
        os.environ["OCR_REC_BATCH"] = "32"
        check("OCR_REC_BATCH 能改批大小（默认 6 时一页 7.68 s，16 时 1.75 s）",
              oi._rec_batch() == 32, oi._rec_batch())
        os.environ["OCR_REC_BATCH"] = "not-a-number"
        check("批大小写坏了退回 16（不许拿 ValueError 把整批跑挂）",
              oi._rec_batch() == 16, oi._rec_batch())
    finally:
        for key, value in (("OCR_THREADS", original_threads),
                           ("OCR_REC_BATCH", original_batch)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def case_image_doc() -> None:
    print("\n⑤ 图片文档：单页 / 多帧 TIFF / 试跑截断")
    from PIL import Image  # noqa: PLC0415 —— 与 ocr_index 同一条依赖

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        kb = root / "知识库"
        folder = kb / "08-编剧方法书籍"
        folder.mkdir(parents=True)
        (folder / "封面.png").write_bytes(png(600, 800))
        frames = [Image.new("RGB", (600, 800), "white") for _ in range(3)]
        frames[0].save(folder / "扫描件.tif", format="TIFF", save_all=True,
                       append_images=frames[1:])

        base = {"path": f"G:\\电影剧本\\{sl.LOCAL_MARKER}\\08-编剧方法书籍\\封面.png",
                "rel": "08-编剧方法书籍/封面.png", "name": "封面.png", "extension": ".png"}
        original = sl.LIBRARY_ROOT
        sl.LIBRARY_ROOT = kb
        try:
            outcome = oi._ocr_image(base, FakeOCR(["封面正文。" * 40]), max_pages=None)
            check("单页图片 ⇒ 1 页且有正文块",
                  outcome["pages"] == 1 and bool(outcome["chunks"]), outcome["pages"])
            check("图片没有文字层可借 ⇒ native 恒 0", outcome["native"] == 0, outcome["native"])

            tif = {**base, "path": base["path"].replace("封面.png", "扫描件.tif"),
                   "rel": "08-编剧方法书籍/扫描件.tif", "name": "扫描件.tif",
                   "extension": ".tif"}
            outcome = oi._ocr_image(tif, FakeOCR(["第一帧。" * 40, "", "第三帧。" * 40]),
                                    max_pages=None)
            check("多帧 TIFF 按帧算页（3 帧 ⇒ 3 页）", outcome["pages"] == 3, outcome["pages"])
            check("空帧计入 blank 且不产空块",
                  outcome["blank"] == 1
                  and all(text.strip() for _page, _no, text in outcome["chunks"]),
                  (outcome["blank"], len(outcome["chunks"])))

            outcome = oi._ocr_image(tif, FakeOCR(["第一帧。" * 40]), max_pages=1)
            check("--pages 1 只读首帧 ⇒ scanned=1 < pages=3（落库记 partial）",
                  (outcome["scanned"], outcome["pages"]) == (1, 3),
                  (outcome["scanned"], outcome["pages"]))
        finally:
            sl.LIBRARY_ROOT = original

    check("待办白名单收齐图片后缀，且仍含 pdf/docx",
          {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
          <= set(oi.SUPPORTED_SUFFIXES), oi.SUPPORTED_SUFFIXES)


def case_dispatch() -> None:
    print("\n⑥ 分派与待办清单")
    calls: list[str] = []

    def stub(tag: str):
        #: ``*args`` 是必须的：``_ocr_text`` **不收引擎**（只解码），签名与另三条路不同。
        def run(item, *args, **kwargs):
            calls.append(tag)
            return {"pages": 1, "scanned": 1, "blank": 0, "chunks": [], "seconds": 0}
        return run

    originals = (oi._ocr_one, oi._ocr_docx, oi._ocr_image, oi._ocr_text)
    oi._ocr_one, oi._ocr_docx, oi._ocr_image, oi._ocr_text = (
        stub("pdf"), stub("docx"), stub("image"), stub("text"))
    try:
        oi._ocr_item({"rel": "a.pdf", "extension": ".pdf"}, None, dpi=150, max_pages=None)
        oi._ocr_item({"rel": "b.docx", "extension": ".docx"}, None, dpi=150, max_pages=None)
        oi._ocr_item({"rel": "c.DOCX"}, None, dpi=150, max_pages=None)  #: 扩展名大小写
        oi._ocr_item({"rel": "d.png", "extension": ".png"}, None, dpi=150, max_pages=None)
        oi._ocr_item({"rel": "e.TIF"}, None, dpi=150, max_pages=None)   #: 没登记 extension 时看 rel
        oi._ocr_item({"rel": "f.txt", "extension": ".txt"}, None, dpi=150, max_pages=None)
        oi._ocr_item({"rel": "g.MD"}, None, dpi=150, max_pages=None)    #: .md 与大小写
        check("_ocr_item 按扩展名分派（pdf / docx / DOCX / png / TIF / txt / MD 各走各的）",
              calls == ["pdf", "docx", "docx", "image", "image", "text", "text"], calls)
        #: 纯文本件只解码 ⇒ 不该为它起引擎（起一次要几秒，还占一份显存会话）
        check("纯文本件不算「要引擎」，PDF 照旧算",
              oi._needs_engine([{"rel": "a.txt", "extension": ".txt"}]) is False
              and oi._needs_engine([{"rel": "a.pdf", "extension": ".pdf"}]) is True
              and oi._needs_engine([]) is False)
    finally:
        oi._ocr_one, oi._ocr_docx, oi._ocr_image, oi._ocr_text = originals

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db = root / "curated.sqlite"
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, name TEXT, "
            "extension TEXT, size INTEGER, modified_ns INTEGER, pages INTEGER);"
            "CREATE TABLE chunks (document_id INTEGER);"
        )
        marker = sl.LOCAL_MARKER
        rows = [
            (1, f"G:\\电影剧本\\{marker}\\08-编剧方法书籍\\扫描.docx", "扫描.docx", ".docx", 10, 11, 0),
            (2, f"G:\\电影剧本\\{marker}\\08-编剧方法书籍\\正文.docx", "正文.docx", ".docx", 10, 12, 0),
            (3, f"G:\\电影剧本\\{marker}\\08-编剧方法书籍\\笔记.txt", "笔记.txt", ".txt", 10, 13, 0),
            (4, f"G:\\电影剧本\\{marker}\\90-隔离与待审\\待审.docx", "待审.docx", ".docx", 10, 14, 0),
        ]
        conn.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?)", rows)
        conn.execute("INSERT INTO chunks VALUES (2)")  #: 2 号有正文 ⇒ 不是待办
        conn.commit()
        conn.close()

        saved = (sl.LIBRARY_ROOT, sl.METHODS_DB, sl.OCR_DB, li.EXTRA_DB)
        sl.LIBRARY_ROOT = root
        sl.METHODS_DB = db
        sl.OCR_DB = root / "ocr" / "sidecar.sqlite"
        #: 第三库也要离开真盘：``_pending`` 现在会从那里接「抽不出字的 docx」
        #: （图画型件主库压根不登记，只有这一条路能把它送进 OCR 队列）。
        #: 不隔离就会把开发机真库那份带进来，本节按「三份」写的断言全错。
        li.EXTRA_DB = root / "ocr" / "不存在-extra.sqlite"
        try:
            plan = oi._pending()
            #: .docx 与 .txt 都要收进来：白名单曾经只有 pdf/docx/图片，纯文本件被
            #: ``continue`` 静默跳过。主库那 80 份恰好在表里已有块（所以从没露馅），
            #: 但只要某份的块丢了，就会变成"列进待办却没人能做"。
            check("待办清单收 .docx 与 .txt（原先只认 .pdf ⇒ 那些份永远补不上）",
                  [item["rel"] for item in plan]
                  == ["08-编剧方法书籍/扫描.docx", "08-编剧方法书籍/笔记.txt"],
                  [item["rel"] for item in plan])
            check("待办清单仍剔掉「有正文」与隔离桶（docx 支持不是放宽筛选）",
                  all("90-隔离与待审" not in item["rel"] and "正文.docx" not in item["rel"]
                      for item in plan))
            check("清单项带 extension（落库的扩展名跟着来源走，不写死 .pdf）",
                  plan[0]["extension"] == ".docx", plan[0])

            oi._save(plan[0], {"pages": 3, "scanned": 3, "blank": 0, "seconds": 1.0,
                              "chunks": [(1, 1, "整页图 OCR 出来的正文")]}, dpi=150, engine="t")
            side = sqlite3.connect(sl.OCR_DB)
            try:
                row = side.execute("SELECT extension, pages, status FROM documents").fetchone()
                text = side.execute("SELECT text FROM chunks").fetchone()[0]
            finally:
                side.close()
            check("落库写的是 .docx 与真实页数，status=ocr（整本做完）",
                  row == (".docx", 3, "ocr"), row)
            check("正文块落进 sidecar（检索时才看得见）", "整页图" in text, text)

            oi._save(plan[0], {"pages": 3, "scanned": 1, "blank": 0, "seconds": 1.0,
                              "chunks": [(1, 1, "只跑了第一页")]}, dpi=150, engine="t")
            side = sqlite3.connect(sl.OCR_DB)
            try:
                status = side.execute("SELECT status FROM documents").fetchone()[0]
            finally:
                side.close()
            check("--pages 试跑重跑同一份 ⇒ status 退回 partial（不许留 ocr 骗过下次跳过）",
                  status == "partial", status)

            #: 再造一份「压根没碰过」的待办 —— 它属于 pending，但**不是**半成品。
            #: 没有这份做对照，``only="partial"`` 就算写成「返回全部」也测不出来。
            touch = sqlite3.connect(db)
            touch.execute(
                "INSERT INTO documents VALUES (5, ?, '未动过.docx', '.docx', 10, 15, 0)",
                [f"G:\\电影剧本\\{marker}\\08-编剧方法书籍\\未动过.docx"])
            touch.commit()
            touch.close()
            every = [item["rel"] for item in oi._pending()]
            half = [item["rel"] for item in oi._pending(only="partial")]
            check("待办清单共三份（半成品 + 没碰过的 + 那份 .txt）", len(every) == 3, every)
            check("--only=partial 只回半成品，不把「没碰过」的带进来",
                  half == ["08-编剧方法书籍/扫描.docx"], half)

            #: 口径重烧（切块/清洗规则改了要按新规则重跑存量）只能靠 --only=done：
            #: --redo 会把 3000+ 份待办一起排进来，--limit 20 时老份一个都轮不上。
            oi._save(plan[0], {"pages": 3, "scanned": 3, "blank": 0, "seconds": 1.0,
                              "chunks": [(1, 1, "整本跑完")]}, dpi=150, engine="t")
            untouched = [item["rel"] for item in oi._pending()]
            again = [item["rel"] for item in oi._pending(only="done")]
            check("--only=done 隐含 redo：签收过的份照样回来（否则口径重烧一份都跑不到）",
                  again == ["08-编剧方法书籍/扫描.docx"] and
                  "08-编剧方法书籍/扫描.docx" not in untouched, (untouched, again))
            check("--only=done 不把「压根没碰过」的份带进来", again == ["08-编剧方法书籍/扫描.docx"],
                  again)

            #: 第三库接了一手：造一份「抽不出字」的 docx 记录（图画型件）。
            #: 主库不登记这类盘面，所以它是**唯一**能把这份送进 OCR 队列的来源；
            #: 少了这条，两份文件都不会报错，只是这本书永远没人 OCR。
            extra_db = root / "ocr" / "extra.sqlite"
            conn = sqlite3.connect(extra_db.as_posix())
            conn.executescript(li._SCHEMA)
            conn.execute(
                "INSERT INTO documents(path, name, extension, size, modified_ns,"
                " signature, pages, status) VALUES(?,?,?,?,?,?,?,?)",
                ("05-短剧/图画教材.docx", "图画教材.docx", ".docx", 99, 7, "s", 0,
                 li.STATUS_QUARANTINE))
            conn.commit()
            conn.close()
            li.EXTRA_DB = extra_db
            picked = [item["rel"] for item in oi._pending()]
            check("第三库抽不出字的 docx 会进 OCR 待办（否则两条通道一起漏）",
                  "05-短剧/图画教材.docx" in picked, picked)
            li.EXTRA_DB = root / "ocr" / "不存在-extra.sqlite"
            plain = [item["rel"] for item in oi._pending()]
            check("第三库正常收下的件不进 OCR 队列（已有正文块，重跑是白烧算力）",
                  set(plain) == set(picked) - {"05-短剧/图画教材.docx"}, (plain, picked))
        finally:
            sl.LIBRARY_ROOT, sl.METHODS_DB, sl.OCR_DB, li.EXTRA_DB = saved

    #: 拼错一个字母必须当场退 2：静默忽略等于从 3135 份头部重跑一遍（几天的 CPU）
    saved_argv = sys.argv
    try:
        sys.argv = ["ocr_index", "--plan", "--only=partal"]
        code = oi.main()
    finally:
        sys.argv = saved_argv
    check("--only 给了不认识的值 ⇒ 退出 2（不许当没事发生）", code == 2, code)


def case_budget() -> None:
    print("\n⑦ 线程预算与推理设备（不超硬件线程数 / 要 GPU 就不许静默回落）")
    import os  # noqa: PLC0415 —— 本文件其余用例按需局部 import，保持同一风格

    cores = oi._CORES
    check("cpu_room() 是逻辑线程数减保留，且永远 ≤ 硬件线程数",
          1 <= oi.cpu_room() <= cores, (oi.cpu_room(), cores))

    keys = ("OCR_DEVICE", "OCR_CPU_RESERVE", "OCR_THREAD_CAP", "OCR_THREADS")
    saved = {key: os.environ.get(key) for key in keys}
    real_ready = oi.cuda_ready
    try:
        os.environ.pop("OCR_THREADS", None)
        os.environ["OCR_DEVICE"] = "cpu"
        os.environ["OCR_CPU_RESERVE"] = "2"
        os.environ.pop("OCR_THREAD_CAP", None)
        check("OCR_CPU_RESERVE 从硬件线程里扣（默认留 2 条给渲染线程与系统）",
              oi.cpu_room() == max(1, cores - 2), (oi.cpu_room(), cores))
        os.environ["OCR_THREAD_CAP"] = "4"
        check("OCR_THREAD_CAP 只能把预算收紧，不能撑大",
              oi.cpu_room() == min(4, max(1, cores - 2)), (oi.cpu_room(), cores))
        os.environ.pop("OCR_THREAD_CAP", None)

        workers = oi.default_workers()
        threads, render = oi.worker_threads(workers)
        check("默认档「进程 ×（识别＋渲染）」≤ 预算（这是「不超线程上限」的唯一保证）",
              workers * (threads + render) <= oi.cpu_room() <= cores,
              (workers, threads, render, oi.cpu_room(), cores))
        check("任意进程数下总额都不越界（1–8 进程逐个核，超订会让机器失去响应）",
              all(w * sum(oi.worker_threads(w)) <= oi.cpu_room() for w in range(1, 9)),
              [w * sum(oi.worker_threads(w)) for w in range(1, 9)])
        check("渲染线程也算进配额、且至少 1 条（否则 GPU 只能干等 CPU 渲染）",
              all(oi.worker_threads(w)[1] >= 1 for w in range(1, 9)),
              [oi.worker_threads(w)[1] for w in range(1, 9)])
        check("渲染线程有上限（再多只是在 PyMuPDF 里排队，还白占内存）",
              all(oi.worker_threads(w)[1] <= oi.MAX_RENDER_THREADS for w in range(1, 9)),
              oi.MAX_RENDER_THREADS)
        os.environ["OCR_RENDER_THREADS"] = "2"
        check("OCR_RENDER_THREADS 能钉死渲染线程数（做对照实验用）",
              oi.render_threads(4) == 2, oi.render_threads(4))
        os.environ.pop("OCR_RENDER_THREADS", None)

        oi.cuda_ready = lambda: False
        os.environ["OCR_DEVICE"] = "cpu"
        check("OCR_DEVICE=cpu ⇒ 有卡也钉在 CPU",
              oi.engine_device() == oi.ENGINE_DEVICE_CPU, oi.engine_device())
        os.environ["OCR_DEVICE"] = "auto"
        check("auto + 这套 onnxruntime 上不了 CUDA ⇒ 如实退回 cpu（不许抛）",
              oi.engine_device() == oi.ENGINE_DEVICE_CPU, oi.engine_device())
        os.environ["OCR_DEVICE"] = "cuda"
        raised = None
        try:
            oi.engine_device()
        except oi.DeviceUnavailable as exc:
            raised = exc
        check("显式要 cuda 却没有 CUDAExecutionProvider ⇒ 当场 DeviceUnavailable",
              raised is not None, raised)

        oi.cuda_ready = lambda: True
        check("auto + CUDA 真可用 ⇒ 选 cuda",
              oi.engine_device() == oi.ENGINE_DEVICE_CUDA, oi.engine_device())
        check("GPU 档 ORT 只给 1–2 条线程（CPU 侧只做解码/后处理，多给只会抢渲染）",
              1 <= oi.engine_threads(1) <= 2, oi.engine_threads(1))
        check("GPU 档把剩下的配额全给渲染（可用进程数内每进程 ≥2 条，不许 CPU 20% 摸鱼）",
              all(oi.render_threads(w) >= 2
                  for w in range(2, oi.GPU_MAX_WORKERS + 1)),
              [oi.render_threads(w) for w in range(2, oi.GPU_MAX_WORKERS + 1)])
        # 显式喂快照：真读 nvidia-smi 时结果随「此刻卡上还剩多少」变（后台批次跑着时
        # 空闲显存可能只够 0 个会话），拿它判常量封顶必然时红时绿。
        cap_snap = {"device": oi.ENGINE_DEVICE_CUDA, "cpuBudget": 30, "gpuSessions": 9}
        check("GPU 档进程数按显存封顶（每个 CUDA 会话驻留 ≈4.2 GiB，堆不出 8 进程）",
              oi.default_workers(cap_snap) == oi.GPU_MAX_WORKERS, oi.default_workers(cap_snap))
        busy_snap = {"device": oi.ENGINE_DEVICE_CUDA, "cpuBudget": 30, "gpuSessions": 0}
        check("卡被别人占满（gpuSessions=0）⇒ 不硬堆 GPU 进程，退回 CPU 档的每进程 4 线程口径",
              oi.default_workers(busy_snap) == min(8, 30 // 4), oi.default_workers(busy_snap))

        class NoSession:
            pass

        class WithSession:
            def __init__(self) -> None:
                self.session = type(
                    "S", (), {"get_providers": lambda self: ["CUDAExecutionProvider", "CPUExecutionProvider"]})()

        check("device_report 找不到会话 ⇒ 空表（不许拿「我设了 use_cuda」冒充在用 GPU）",
              oi.device_report(NoSession()) == [])
        check("device_report 逐字回报引擎实际的 provider 列表",
              oi.device_report(WithSession())
              == [("CUDAExecutionProvider", "CPUExecutionProvider")],
              oi.device_report(WithSession()))
    finally:
        oi.cuda_ready = real_ready
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def case_segments() -> None:
    print("\n⑧ 页级切分（只剩一份大书时也要把进程喂满）")
    #: 段必须**首尾相接、既不重叠也不漏页**：漏一页就是永久丢内容，而整份状态还写「完成」。
    for pages in (920, 401, 120, 40, 7):
        for k in (1, 2, 3, 4, 8):
            ranges = oi._page_ranges(pages, k)
            check(f"{pages} 页切 {k} 段：区间连续且恰好覆盖 0–{pages}",
                  ranges[0][0] == 0 and ranges[-1][1] == pages
                  and all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])),
                  ranges)
            check(f"{pages} 页切 {k} 段：首尾相接不重叠",
                  sum(hi - lo for lo, hi in ranges) == pages, ranges)
    check("段数超过页数时退回「一页一段」（不造空段去白起进程）",
          len(oi._page_ranges(3, 8)) == 3, oi._page_ranges(3, 8))

    def item(name: str, suffix: str = ".pdf") -> dict:
        return {"path": f"D:/x/{name}{suffix}", "rel": f"x/{name}{suffix}",
                "name": f"{name}{suffix}", "extension": suffix, "pages": 0}

    big, mid, small = item("big"), item("mid"), item("small")
    word = item("word", ".docx")
    tasks = oi.plan_segments([big, mid, small, word], 4,
                             {"D:/x/big.pdf": 920, "D:/x/mid.pdf": 300,
                              "D:/x/small.pdf": 50, "D:/x/word.docx": None})
    check("份数 ≥ 进程数 ⇒ 一段不切（按文件分发本来就吃满）",
          len(tasks) == 4 and all(rng is None for _it, rng in tasks), tasks)
    tasks = oi.plan_segments([big], 4, {"D:/x/big.pdf": 920})
    check("只剩一份 920 页的书 ⇒ 切 4 段（否则 3 个进程空转、GPU 18%）",
          len(tasks) == 4 and all(rng is not None for _it, rng in tasks),
          [rng for _it, rng in tasks])
    check("切 4 段时每段 ≈230 页",
          [rng for _it, rng in tasks] == [(0, 230), (230, 460), (460, 690), (690, 920)],
          [rng for _it, rng in tasks])
    tasks = oi.plan_segments([big, small], 4,
                             {"D:/x/big.pdf": 920, "D:/x/small.pdf": 50})
    counts = {}
    for it, _rng in tasks:
        counts[it["path"]] = counts.get(it["path"], 0) + 1
    check("小文件不切（<120 页切段的 open/汇总开销抵不上收益）",
          counts["D:/x/small.pdf"] == 1, counts)
    check("额度全补给大文件（3 任务补齐 4 进程）",
          counts["D:/x/big.pdf"] == 3 and len(tasks) == 4, counts)
    tasks = oi.plan_segments([item("a"), item("b")], 8,
                             {"D:/x/a.pdf": 200, "D:/x/b.pdf": 200})
    check("单段不低于 MIN_SEG_PAGES（切太碎：每段都要重开 PDF、重建会话）",
          all((rng[1] - rng[0]) >= oi.MIN_SEG_PAGES
              for _it, rng in tasks if rng is not None),
          [(rng[1] - rng[0]) for _it, rng in tasks if rng is not None])
    check("docx/图片不参与切分（数不出页，也不值得切）",
          all(it["path"] != "D:/x/word.docx" or rng is None for it, rng in tasks))
    check("workers ≤ 1 ⇒ 不切（单进程切段只是多开几份 PDF）",
          all(rng is None for _it, rng in oi.plan_segments([big], 1, {"D:/x/big.pdf": 920})))

    page = "第一章　测试正文，" + "内容反复出现以凑够字数，" * 8
    parts = [
        {"pages": 6, "scanned": 3, "blank": 0, "native": 0, "escalated": 0, "seconds": 9.0,
         "pagesText": [(4, page), (5, page), (6, page)], "notes": {}},
        {"pages": 6, "scanned": 3, "blank": 1, "native": 1, "escalated": 1, "seconds": 12.0,
         "pagesText": [(1, page), (2, page)], "notes": {3: "upgraded"}},
    ]
    merged = oi._merge_segments(parts)
    order = [p for p, _no, _t in merged["chunks"]]
    check("合并后页序升序（段是乱序回来的，清洗必须按全书顺序做）",
          order == sorted(order), order[:10])
    check("合并后 scanned/blank/native/escalated 是各段合计",
          (merged["scanned"], merged["blank"], merged["native"], merged["escalated"])
          == (6, 1, 1, 1),
          (merged["scanned"], merged["blank"], merged["native"], merged["escalated"]))
    check("合并后 seconds 取最大段（段是并行的，相加＝把同一段时间数两遍）",
          merged["seconds"] == 12.0, merged["seconds"])
    check("合并后仍带 judged（漏了 ⇒ 整份隔离永不触发）", "judged" in merged, sorted(merged))


def case_chunk_pages() -> None:
    print("\n⑨ 块的页号记「起始页」（检索跳页据此翻原文）")
    unit = "第一页的开头。"
    #: 两页各 420 字，合计 840 > CHUNK_CHARS(800) ⇒ 必然切出一个跨两页的块。
    #: 老口径把「切出这块时正在读的页」（=2）写进 ``chunks.page``，于是按页跳整体后移一页。
    packed = oi._page_chunks([(1, unit * 60), (2, "第二页接着写。" * 60)])
    pages = [page for page, _no, _text in packed["chunks"]]
    check("跨页的块记**起始页 1**（老口径会记收尾页 2）",
          bool(pages) and pages[0] == 1, pages)
    check("page 序列在 chunk_no 顺序下非降（出现下降即首尾记反）",
          all(a <= b for a, b in zip(pages, pages[1:])), pages)
    #: 内部两个页号必须分开给：tail 是收尾页、start 是起始页，落库只许取 start
    pieces, _rest, _next = oi._chunks_of_page(unit * 120, 2, unit * 5, 1)
    check("_chunks_of_page 分别给出（收尾页 2, 起始页 1），不许混成一个",
          bool(pieces) and tuple(pieces[0][:2]) == (2, 1), pieces[:1])
    tail_only = oi._page_chunks([(7, "只有一页的短正文，够十二个字。")])
    check("短到进尾巴缓冲的块记它自己的页（7）—— 这条本来就对，锁住别再改回去",
          tail_only["chunks"][0][0] == 7, tail_only["chunks"])


def case_salvage() -> None:
    """整份保底：每页都判废时，不许让一本**有字**的书在库里凭空变空。

    真事（2026-11-04 终局对账抓到）：``8 波琳_凯尔_美国影评家_.pdf`` 只有 1 页、574 字
    中文清清楚楚，判废理由是「符号连续段占 20%（3 处）＋有效字符占比偏低 79%」——
    那页是期刊的装饰线/点线被识别成一排制表符，**字本身认得对**。于是整页丢弃、0 块，
    而 ``status`` 仍是 ``ocr``：``--stats`` 报「已 OCR」、检索页页 0 命中、``--audit``
    只多一行 reject，没有任何一处能把人引到「这本书在库里其实是空的」。
    """
    print("\n⑭ 整份保底：每页都判废时，软性判废降为存疑保正文")
    #: 每句都不同（不同汉字够多）—— 不能拿同一句重复 12 遍凑长度，那样 distinct 只有
    #: 二十来字，样本会变成"疑似刷行页"，压根不是真事里那种正文页。
    prose = "".join(
        f"第{i}段：论电影现实主义与纪录电影语言的本体差异，及其在创作实践中的具体路径。"
        for i in range(1, 13))
    #: 制表符/方框线（U+2500 一类）既不是 `_HARD_JUNK` 也不算 `_VALID`，正是真事里的形态：
    #: junkRunShare 超线、validRatio 掉到 0.8 左右 ⇒ 整页判废，可字是对的。
    line = "─" * 40
    page = prose[:200] + line + prose[200:400] + line
    raw = oi.text_clean.judge_page(page)
    #: 先证明样本真的会触发（否则下面几条是自说自话：没判废当然留得下）
    check("构造的样本先被门禁判废，且理由是**软性**的（符号段密集，不是乱码）",
          raw["verdict"] == "reject"
          and raw["metrics"]["junkRunShare"] >= oi.text_clean.JUNK_RUN_SHARE_REJECT
          and raw["metrics"]["maxRun"] < oi.text_clean.JUNK_RUN_MAX_REJECT
          and raw["metrics"]["validRatio"] >= oi.text_clean.VALID_REJECT
          and raw["metrics"]["junkRatio"] == 0,
          (raw["verdict"], raw["reasons"], raw["metrics"]["validRatio"],
           raw["metrics"]["junkRunShare"]))

    kept = oi._page_chunks([(1, page)])
    check("单页全废但属软性 ⇒ 仍出块（修前：整页丢弃、本份 0 块）",
          bool(kept["chunks"]) and kept["rejected"] == 0,
          (kept["rejected"], len(kept["chunks"]), [r["verdict"] for r in kept["audit"]]))
    check("降级要留痕：audit 记 suspect 并写清是「整份每页都判废」触发的",
          kept["audit"] and all(row["verdict"] == "suspect" for row in kept["audit"])
          and any("整份每页都判废" in row["reasons"] for row in kept["audit"]),
          kept["audit"])

    #: 空白页的判定是 ``ok``（只有 ``reasons=['空白页']``、chars=0）⇒ 若按「所有页都是
    #: reject」触发，这种「1 页封面空白 + 1 页正文判废」的件就漏掉了，症状与真事一模一样。
    two = oi._page_chunks([(1, ""), (2, page)])
    check("空白页 + 一页软废 也算「没有一页带着正文活下来」⇒ 保底必须触发",
          bool(two["chunks"]), (two["rejected"], len(two["chunks"])))

    #: 硬伤（乱码 / 整块被认成符号 / 整页只剩几个字）不许被这条保底放进来：
    #: 那不是"书本来就这样"，是真认错了，进了 fts5 就是纯噪声。
    hard = oi._page_chunks([(1, "\ufffd" * 200)])
    check("硬伤页保底不生效（乱码仍判废、仍 0 块）",
          hard["rejected"] == 1 and not hard["chunks"],
          (hard["rejected"], len(hard["chunks"])))
    check("0 块且**有字**被丢掉 ⇒ 判隔离（不再顶着 ocr 当一本查不到的隐形空书）",
          oi._status_of({**hard, "scanned": 1, "pages": 1}) == oi.STATUS_QUARANTINE,
          (hard.get("rejectedChars"), hard["judged"]))

    #: 期刊「总目录」：篇名后面跟一串点线 ⇒ 有效字符**占比**只有 47%、去重比 7%，
    #: 但有效字有上千个。按占比判会整本丢（实测隔离清单里就有两份这样的总目录）；
    #: 保底改看「实打实的有效字符数」后必须放行。
    titles = "".join(
        f"第{i}章 论电影现实主义与纪录电影语言的本体差异及其实践路径研究"
        + "·" * 40 + f"({i})\n" for i in range(1, 21))
    toc_soft = oi.text_clean.judge_page(titles)
    check("构造的总目录页先被判废，且是「占比低」这类软性理由",
          toc_soft["verdict"] == "reject"
          and toc_soft["metrics"]["validRatio"] < oi.text_clean.VALID_REJECT
          and toc_soft["metrics"]["maxRun"] < oi.text_clean.JUNK_RUN_MAX_REJECT
          and toc_soft["metrics"]["junkRatio"] == 0,
          (toc_soft["verdict"], toc_soft["reasons"], toc_soft["metrics"]["validRatio"]))
    toc_kept = oi._page_chunks([(1, titles)])
    check("占比低但有效字符够多（点线目录）⇒ 保底放行，别整本丢",
          bool(toc_kept["chunks"]) and toc_kept["rejected"] == 0,
          (toc_kept["rejected"], len(toc_kept["chunks"])))

    #: 反面：**硬伤**不进保底。乱码页、整块被认成一排符号的页，即便整本因此变空也不放行
    #: （乱码进了 fts5 就是纯噪声，检索命中一堆读不出意思的片段）。
    check("硬伤页 `_salvageable` 直接为假（乱码／整块被认成符号）",
          oi._salvageable({"junkRatio": 1.0, "maxRun": 3, "chars": 4000, "validRatio": 0.9})
          is False
          and oi._salvageable({"junkRatio": 0.0, "maxRun": 3000, "chars": 4000,
                               "validRatio": 0.9}) is False)
    #: 反面之二：有效字太少（插图页/封面页/真噪声）也不值得为它让整本书"降级活着"。
    check("有效字太少（<120 且占比 <60%）⇒ 不放行",
          oi._salvageable({"junkRatio": 0.0, "maxRun": 5, "chars": 300,
                           "validRatio": 0.2}) is False
          and oi._salvageable({"junkRatio": 0.0, "maxRun": 5, "chars": 200,
                               "validRatio": 0.55}) is False)

    blank = oi._page_chunks([(1, ""), (2, "")])
    check("全空白页不判隔离（本来就没有内容可丢），仍记 ocr",
          oi._status_of({**blank, "scanned": 2, "pages": 2}) == oi.STATUS_OCR,
          (blank.get("rejectedChars"), len(blank["chunks"])))

    #: 隔离的语义随之收窄：它现在只管「这机器读的东西不能用」（硬伤成片），
    #: 而「排版被认成符号但字是对的」一律留内容 —— 与 ``_page_chunks`` 里那句
    #: 「假阳性丢内容比留一点噪声严重得多」同一个口径。
    many = oi._page_chunks([(1, page), (2, page), (3, page)])
    check("成片软废不再整份隔离：3 页全软废 ⇒ 有块且 status=ocr",
          bool(many["chunks"])
          and oi._status_of({**many, "scanned": 3, "pages": 3}) == oi.STATUS_OCR,
          (many["rejected"], many["judged"], len(many["chunks"])))


def case_text() -> None:
    """纯文本件（``.txt``/``.md``）：编码判定 + 伪页 + 「清洗不许删内容」的口径。

    2026-11-04 之前这条路**在分派器上根本不存在**（白名单只有 pdf/docx/图片）。
    先说清事实：主库那 80 份 ``.txt``/``.md`` 早就有正文块、中文完全正确，所以它们
    **不是缺口**，``_pending`` 也不会列它们；补这条路是为了「万一某份的表里块丢了」时
    有出口，而不是补一个不存在的坑（我一度把它误判成 1481 万字的最大缺口，因为扫描脚本
    按扩展名统计、没回主库核块 —— 教训是"缺"必须两边都核）。

    这里的风险也不是"读不进来"，是**读进来的东西已经变形**：
    GBK 被按 UTF-8 读成 ``ËÄ ²»ÊÇ°ëÒ¹ÀïÄÖ¹íÃ´£¿``（不抛异常、页页还能过门禁），
    或者照 PDF 那套清洗把剧本对白串成一大段、把正文首行当跨页页眉剔掉。
    """
    print("\n⑬ 纯文本件（.txt/.md）：编码、伪页、清洗口径")
    #: GBK（主库 30 份的主力编码）被按 UTF-8 硬读：整段变成
    #: ``ËÄ ²»ÊÇ°ëÒ¹ÀïÄÖ¹íÃ´£¿``，门禁一个字都拦不住（它不是"像乱码"，它字字合法）。
    text, encoding = oi._decode_text("四 不是半夜里闹鬼么？\n周萍 你走开！\n".encode("gbk") * 200)
    check("GBK 文本解成中文（按 UTF-8 硬读会得到 ËÄ ²»ÊÇ…）",
          encoding == "gb18030" and text.startswith("四 不是半夜里闹鬼么？")
          and "ËÄ" not in text, (encoding, text[:20]))

    text, encoding = oi._decode_text(("\ufeff" + "第一幕 周家客厅。\n" * 100).encode("utf-8"))
    check("带 BOM 的 UTF-8 去掉 BOM（残留 \\ufeff 会让首个块检索不到）",
          encoding == "utf-8-sig" and not text.startswith("\ufeff")
          and "周家客厅" in text, (encoding, text[:12]))

    #: 老英文稿的另一种形态：高位字节（``0x92`` 弯引号）在 gb18030/big5 里"凑得成一对"，
    #: 于是前三种编码都能"解码成功"，只有按可疑度打分才分得开（真实那 2 份的坏字节是
    #: ``0xE9``/``0xC9``，三种编码全抛异常，直接落到 cp1252 兜底）。
    text, encoding = oi._decode_text("It’s a play about fathers and sons.\n".encode("cp1252") * 50)
    check("Windows-1252 老英文稿不会被「凑得成的 GBK」抢走（打分看拉丁/汉字混排）",
          encoding == "cp1252" and "It’s a play" in text, (encoding, text[:24]))

    lines = [f"第{i}行：这是一段剧本正文，用来凑够字数。" for i in range(1, 401)]
    pages = oi._split_text_pages("\n".join(lines))
    check("切伪页不丢行、不在行中间切（拼回去逐字等于原文）",
          "\n".join(pages) == "\n".join(lines), (len(pages), len(lines)))
    check("伪页按 TEXT_PAGE_CHARS 满页切（只有末页可以短）",
          all(len(page) >= oi.TEXT_PAGE_CHARS for page in pages[:-1]),
          [len(page) for page in pages[:3]])

    #: 同一段文字，「纯文本口径」与「PDF 口径」各跑一遍：这是 plain 参数唯一的判据。
    #: PDF 那套有个「整页被切碎就把句子接回去」的步骤（为 OCR 按字形簇出行而设），
    #: 而剧本/诗歌的短行本来就该逐行 —— 接回去以后检索还命中，但读出来已经不是对白了。
    drama = "周萍推开门走进来\n他看见桌上的信\n愣住了\n" * 40
    plain_text = "\n".join(t for _p, _n, t in oi._page_chunks([(1, drama)], plain=True)["chunks"])
    normal_text = "\n".join(t for _p, _n, t in oi._page_chunks([(1, drama)])["chunks"])
    check("plain 保住逐行（非 plain 会把无句号的短行按「整页被切碎」接成一段）",
          "周萍推开门走进来\n他看见桌上的信" in plain_text
          and "周萍推开门走进来\n他看见桌上的信" not in normal_text, normal_text[:60])

    #: 判废口径：门禁的"符号连续段"是为 OCR 把表格识成一排符号设计的。书里的目录页
    #: 天生就有「目 录……………………………………(1)」这样的**一长串点线**（实测 --audit
    #: 里 82 个被剔除的页全是这种），对它判废＝删真内容；纯文本件必须放行。
    toc = "目录\n" + "第1章 标题" + "…" * 62 + "(37)\n"
    toc_plain = oi._page_chunks([(1, toc)], plain=True)
    toc_normal = oi._page_chunks([(1, toc)])
    #: ⚠️ 2026-11-04 之后 OCR 口径**也不再丢**这条目录页了：门禁仍判它废（点线是判据），
    #: 但「整份每页都判废」会触发保底把它降为存疑 —— 这条测试原先断言 ``rejected == 1
    #: 且 0 块``，改口径后当场变红。判据要盯的是**放行结果**与**原始判定**两件事。
    check("门禁本身仍判目录页废（点线判据照样生效，不是被悄悄放宽）",
          oi.text_clean.judge_page(toc)["verdict"] == "reject")
    check("目录页在纯文本口径直接放行（那是给识别错的行准备的判据，不该删书上的字）",
          toc_plain["rejected"] == 0 and bool(toc_plain["chunks"]),
          (toc_plain["rejected"], len(toc_plain["chunks"])))
    check("OCR 口径的同类件走整份保底：降为存疑留内容，理由写明触发原因",
          toc_normal["rejected"] == 0 and bool(toc_normal["chunks"])
          and any("整份每页都判废" in row["reasons"] for row in toc_normal["audit"]),
          (toc_normal["rejected"], toc_normal["audit"]))
    check("目录页不判废但要留证据：仍记进 audit（verdict=suspect）",
          [row["verdict"] for row in toc_plain["audit"]] == ["suspect"],
          toc_plain["audit"])

    #: 唯一真该判废的是**解码失败**：四种编码全没解对时留下的是替换符，那套字进 fts5
    #: 就是纯噪声。放宽判废口径不能把这条一起放宽 —— 反向验证一下。
    broken = "\ufffd" * 200
    broken_plain = oi._page_chunks([(1, broken)], plain=True)
    check("非法字符页在纯文本口径仍判废（放宽的是「书本来就这样」，不是解码失败）",
          broken_plain["rejected"] == 1 and not broken_plain["chunks"],
          broken_plain["rejected"])

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        #: 每页都以同一行打头：非 plain 清洗会把它当"跨页页眉"全剔掉（＝删内容）。
        #: 每页刻意凑到**正好** TEXT_PAGE_CHARS（标记行 8 字 + 一行 1490 正文 ⇒ 9+1491=1500），
        #: 这样每页都在行末 flush、没有行残留到下一页，标记行必然落在每页第一行。
        one_page = "【每页重复的行】\n" + "正文" * 745
        body = "\n".join(one_page for _ in range(5))
        play = root / "知识库" / "58-短剧待精筛" / "雷雨.txt"
        play.parent.mkdir(parents=True, exist_ok=True)
        play.write_bytes(body.encode("gbk"))
        item = {"path": f"G:\\电影剧本\\{sl.LOCAL_MARKER}\\58-短剧待精筛\\雷雨.txt",
                "rel": "58-短剧待精筛/雷雨.txt", "name": "雷雨.txt", "extension": ".txt"}
        page_count = len(oi._split_text_pages(body))
        original = sl.LIBRARY_ROOT
        sl.LIBRARY_ROOT = root / "知识库"
        try:
            outcome = oi._ocr_text(item)
            merged = "\n".join(text for _page, _no, text in outcome["chunks"])
            check("解出中文正文，乱码指纹一个字都没有",
                  "正文正文" in merged and "ËÄ" not in merged, merged[:40])
            check("不剔跨页版头：每页打头那行原样留着（plain 模式剔它就是删内容）",
                  page_count == 5 and merged.count("【每页重复的行】") == page_count
                  and outcome["droppedHeads"] == [],
                  (merged.count("【每页重复的行】"), page_count))
            check("页数/原生页按伪页记、没走 OCR 也没被判空白",
                  (outcome["pages"], outcome["native"], outcome["blank"])
                  == (page_count, page_count, 0), outcome)
            check("编码留在 outcome 里（报告尾行靠它看出解错编码）",
                  outcome["encoding"] == "gb18030", outcome["encoding"])
            trial = oi._ocr_text(item, max_pages=1)
            check("--pages 试跑：scanned=1 < pages=N（落库标 partial，不算整本做完）",
                  trial["scanned"] == 1 and trial["pages"] == page_count,
                  (trial["scanned"], trial["pages"]))
        finally:
            sl.LIBRARY_ROOT = original


def case_page_threads() -> None:
    print("\n⑫ 并发识别（GPU 队列填深；CPU 档默认不并发）")
    saved = {key: os.environ.get(key) for key in ("OCR_DEVICE", "OCR_PAGE_THREADS")}
    real_gpu = oi.cuda_ready
    try:
        #: 这台机器上有卡也不会在单测里真建会话，直接声明「能上 CUDA」
        oi.cuda_ready = lambda: True
        os.environ["OCR_DEVICE"] = "cuda"
        os.environ.pop("OCR_PAGE_THREADS", None)
        check("GPU 档默认并发 2 页（串行时 GPU 只忙 53%，一次只有一个请求在飞）",
              oi.page_threads() == 2, oi.page_threads())
        os.environ["OCR_PAGE_THREADS"] = "4"
        check("显式 OCR_PAGE_THREADS 压过默认",
              oi.page_threads() == 4, oi.page_threads())
        os.environ["OCR_PAGE_THREADS"] = "0"
        check("给 0 夹到 1（不许起 0 线程池）", oi.page_threads() == 1, oi.page_threads())
        os.environ["OCR_PAGE_THREADS"] = "abc"
        check("写错退回默认，不让一个环境变量把整批跑废",
              oi.page_threads() == 2, oi.page_threads())
        os.environ.pop("OCR_PAGE_THREADS", None)
        os.environ["OCR_DEVICE"] = "cpu"
        check("CPU 档默认 1（那时 ORT 已吃 16 线程，多页并发只会互抢核）",
              oi.page_threads() == 1, oi.page_threads())
    finally:
        oi.cuda_ready = real_gpu
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    #: 并发完成顺序与页序无关 ⇒ 必须显式按页号收。顺序错乱会让「页眉剔除」把正文首行当页眉。
    raw = [(3, "第三页的正文，长度足够。", "vlm 重识别"),
           (1, "第一页的正文，长度足够。", None),
           (2, "短", None),
           (5, "第五页的正文，长度足够。", None),
           (4, "第四页的正文，长度足够。", None)]
    pages_text, notes, blank = oi._digest_pages(raw)
    order = [no for no, _ in pages_text]
    check("乱序收进来也按页序交出（切块要靠跨页统计）", order == [1, 3, 4, 5], order)
    check("短页计入 blank 且不进正文（口径与串行版一致）", blank == 1, blank)
    check("留痕落在它自己那一页，不串页", notes == {3: "vlm 重识别"}, notes)


def case_cli_args() -> None:
    print("\n⑪ 参数形式守卫（--name value 不许静默退回默认）")
    saved = sys.argv[:]
    try:
        sys.argv = ["ocr_index", "--path-substr", "金榜题名", "--run", "--redo"]
        bad = oi._split_value_args()
        check("空格写法被认出来（否则 --path-substr 静默失效 ⇒ 不筛、跑前 20 份）",
              bad == ["--path-substr 金榜题名"], bad)
        sys.argv = ["ocr_index", "--path-substr=金榜题名", "--run"]
        check("等号写法不报", oi._split_value_args() == [], oi._split_value_args())
        sys.argv = ["ocr_index", "--path-substr", "--run"]
        check("后面直接跟下一个开关也算写错（值根本没给）",
              oi._split_value_args() == ["--path-substr"], oi._split_value_args())
        sys.argv = ["ocr_index", "--only", "partial"]
        check("--only 空格写法同样报（原先的 only 校验连值都读不到，等于没校验）",
              oi._split_value_args() == ["--only partial"], oi._split_value_args())
        sys.argv = ["ocr_index", "--run", "--redo", "--json"]
        check("纯开关不受影响", oi._split_value_args() == [], oi._split_value_args())
    finally:
        sys.argv = saved


def case_resources() -> None:
    print("\n⑩ 资源调度（按实时空闲显存与外部 CPU 占用决定开多少进程/线程）")
    saved = {k: os.environ.get(k) for k in ("OCR_DEVICE", "OCR_CPU_BUDGET", "OCR_THREADS",
                                            "OCR_RENDER_THREADS")}
    try:
        #: 探测失败一律当「不知道」，退回实测上限 —— 不许因为 nvidia-smi 没装就少开进程
        cap, why = oi.gpu_session_cap(None)
        check("探测不到显存 ⇒ 按实测上限 4 会话（不猜、也不因为探测失败就缩到 1）",
              cap == oi.GPU_MAX_WORKERS and "上限" in why, (cap, why))
        cap, why = oi.gpu_session_cap((23028, 20000))
        check("空闲 20000 MiB ⇒ 4 会话", cap == 4, (cap, why))
        cap, _why = oi.gpu_session_cap((23028, 6000))
        check("空闲 6000 MiB ⇒ 只开 1 会话（(6000−1024)//4200 = 1）", cap == 1, cap)
        cap, why = oi.gpu_session_cap((23028, oi.GPU_HEADROOM_MIB + oi.GPU_SESSION_MIB))
        check("刚够一个会话（空闲 = 余量 + 一个会话）⇒ 1 会话，不能判成 0",
              cap == 1, (cap, why))
        cap, why = oi.gpu_session_cap((23028, 1500))
        check("空闲 1500 MiB ⇒ 0 会话，且说明里点明是「被占」而不是静默开会话",
              cap == 0 and "占" in why, (cap, why))

        #: 外部占用 60% ⇒ 预算必须收缩，而不是继续按 30 条抢
        os.environ.pop("OCR_CPU_BUDGET", None)
        check("没有预算快照 ⇒ cpu_budget() 退回硬件余量（老调用方行为不变）",
              oi.cpu_budget() == oi.cpu_room(), (oi.cpu_budget(), oi.cpu_room()))
        os.environ["OCR_CPU_BUDGET"] = "12"
        check("快照写了预算 ⇒ cpu_budget() 读它（子进程靠这条环境变量拿到全局一致的配额）",
              oi.cpu_budget() == 12, oi.cpu_budget())

        os.environ["OCR_DEVICE"] = oi.ENGINE_DEVICE_CUDA
        for workers in (1, 2, 4):
            ort, render = oi.worker_threads(workers)
            check(f"预算 12 时 {workers} 进程 ×（{ort}+{render}）不超预算",
                  workers * (ort + render) <= 12, (workers, ort, render))

        check("显存够 4 会话且预算够 ⇒ 4 进程",
              oi.default_workers({"cpuBudget": 30, "device": oi.ENGINE_DEVICE_CUDA,
                                  "gpuSessions": 4}) == 4)
        check("显存只够 2 会话 ⇒ 2 进程（不硬开到 4 撞 OOM）",
              oi.default_workers({"cpuBudget": 30, "device": oi.ENGINE_DEVICE_CUDA,
                                  "gpuSessions": 2}) == 2)
        check("预算 8 条线程 ⇒ 1 进程（8 // 每进程 7 条）",
              oi.default_workers({"cpuBudget": 8, "device": oi.ENGINE_DEVICE_CUDA,
                                  "gpuSessions": 4}) == 1)
        fallback = oi.default_workers({"cpuBudget": 30, "device": oi.ENGINE_DEVICE_CUDA,
                                       "gpuSessions": 0})
        check("一个会话都开不起 ⇒ 回落 CPU 档进程数（不是 0、也不是死等）",
              1 <= fallback <= 8, fallback)

        #: 采样器：跑批期间唯一的「到底谁在瓶颈」证据来源，空样本与各档结论都不能炸
        empty = oi._ResourceSampler()
        check("采样器没采到样也能出报告（不许因 psutil/nvidia-smi 缺失就抛异常）",
              "未采样" in empty.summary(), empty.summary())
        empty.cpu, empty.gpu = [95.0], [10]
        check("CPU 高 GPU 低 ⇒ 结论指向 CPU", "CPU 已吃满" in empty.verdict(), empty.verdict())
        empty.cpu, empty.gpu = [30.0], [96]
        check("GPU 高 ⇒ 结论明说「CPU 占比低是任务常态」，别让人白加线程",
              "GPU 已吃满" in empty.verdict(), empty.verdict())
        empty.cpu, empty.gpu = [22.0], [18]
        check("两边都低 ⇒ 结论指向磁盘/单线程段，而不是让人继续加线程",
              "两边都没吃满" in empty.verdict(), empty.verdict())

        #: ⚠️ 真跑实录：起名 `_stop` 会盖掉 Thread 的内部方法，`join()` 里变成「调用 Event」
        #: ⇒ TypeError，而且单测若不 join 就发现不了。这条必须真起线程、真 join 一次。
        live = oi._ResourceSampler(interval=0.2)
        live.start()
        time.sleep(0.4)
        crashed: Exception | None = None
        try:
            live.stop()
        except Exception as exc:  # noqa: BLE001
            crashed = exc
        check("采样器能干净停下（`_stop` 与 Thread._stop() 撞名 ⇒ join 时炸）",
              crashed is None, crashed)
        check("停下后线程真的退出（不留 daemon 线程挂着）", not live.is_alive())
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def case_reaudit() -> None:
    """``--reaudit``：只重判「有存疑页」的件，别为了清审计噪声把整库重烧一遍。

    起因：门禁给「去重比」加了中文门槛后，旧判据在 368 份里留下 8458 页存疑
    （英文剧本页去重字符只有 30–60 个，必然判「疑大量重复」）。这些页**一个字没丢**
    （``suspect`` 照常写块），但会一直挂在 ``--audit`` 里把真问题淹掉。页文本没单独存，
    重判只能重跑 —— 于是需要一个「只跑这批」的入口，而不是 ``--only=done`` 全库 3138 份。
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        saved = (sl.LIBRARY_ROOT, sl.METHODS_DB, sl.OCR_DB)
        sl.OCR_DB = root / "ocr" / "sidecar.sqlite"
        try:
            conn = oi._open_sidecar()
            try:
                for doc_id, path, verdict in ((1, "a.pdf", "suspect"), (2, "b.pdf", "ok"),
                                              (3, "c.pdf", "reject"), (4, "d.pdf", "suspect")):
                    conn.execute(
                        "INSERT INTO documents (id, path, name, extension, size, modified_ns, "
                        "signature, pages, status, gated) VALUES (?,?,?,?,?,?,?,?,?,1)",
                        (doc_id, path, Path(path).name, ".pdf", 1, 1, "1:1", 10, "ocr"))
                    conn.execute(
                        "INSERT INTO page_audit (document_id, path, name, page, verdict, "
                        "reasons, chars, preview) VALUES (?,?,?,?,?,?,?,?)",
                        (doc_id, path, Path(path).name, 1, verdict, "x", 10, "y"))
                conn.commit()
            finally:
                conn.close()

            paths = oi._suspect_paths()
            check("--reaudit 认的是「有存疑页」的件（reject/ok 都不算）",
                  paths == {"a.pdf", "d.pdf"}, paths)
            plan = [{"path": "a.pdf", "rel": "a.pdf"}, {"path": "b.pdf", "rel": "b.pdf"},
                    {"path": "c.pdf", "rel": "c.pdf"}]
            check("收缩只留存疑件（b/c 不陪跑）",
                  [item["rel"] for item in oi._keep_suspects(plan)] == ["a.pdf"],
                  oi._keep_suspects(plan))
        finally:
            sl.LIBRARY_ROOT, sl.METHODS_DB, sl.OCR_DB = saved

    #: 隐含 --redo 不能只写在文档里：那些件都算"已处理"，不重判就进不了候选
    saved_argv = sys.argv
    called: dict[str, object] = {}
    original = oi._run

    def spy(*args, **kwargs):
        called.update(kwargs)
        if args:  #: limit 是位置参数传进去的
            called["limit"] = args[0]
        return 0

    oi._run = spy
    try:
        sys.argv = ["ocr_index", "--run", "--reaudit", "--limit=5"]
        oi.main()
    finally:
        oi._run = original
        sys.argv = saved_argv
    check("--run --reaudit 自动带 redo=True（否则候选集是空的、一份都不做）",
          called.get("reaudit") is True and called.get("redo") is True, called)

    #: ``--limit=0`` 曾经被 ``_count`` 夹成 1：本意"全做完"、实际只做 1 份，日志还写
    #: 「本次做 1 份」像是正常收尾，差掉的 367 份一声不响（2026-11-04 实测）。
    called.clear()
    oi._run = spy
    try:
        sys.argv = ["ocr_index", "--run", "--limit=0"]
        oi.main()
    finally:
        oi._run = original
        sys.argv = saved_argv
    check("--limit=0 传下去就是 0（＝全部），不许被夹成 1 份",
          called.get("limit") == 0, called)


def case_save_lock() -> None:
    """落库遇写锁：等得住、退得开、不静默丢。

    2026-11-04 真实事故：整批跑到 2739/2988 时被一次探测性的**只读**查询占住锁，
    ``_save`` 直接抛 ``database is locked`` 冲出 ``main()``，9 小时的活儿当场结束
    （已入库的没丢，剩下的 249 份得重跑）。根因不是脏数据，而是**没给长任务留余地**：
    ``sqlite3.connect`` 默认只等 5 秒，而 ``_save`` 又是全批唯一没有兜底的调用点。
    """
    #: ① 连接本身要等够久：默认 5 秒扛不住一次 1.3 GB 库的备份读
    conn = oi._open_sidecar()
    try:
        busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        conn.close()
    check("sidecar 连接 busy_timeout ≥ 60s（默认 5s 会被长读顶掉）", busy >= 60000, busy)

    original = oi._save
    quiet = lambda *_args, **_kwargs: None  # noqa: E731

    #: ② 锁错误连犯两次 ⇒ 第三次落库成功（退让有效，且不是无脑放弃）
    calls: list[int] = []

    def flaky(item, outcome, *, dpi, engine):
        calls.append(1)
        if len(calls) < 3:
            raise sqlite3.OperationalError("database is locked")

    oi._save = flaky
    try:
        oi._save_atomic({}, {}, dpi=200, engine="x", retries=4, backoff=0.01, notify=quiet)
    finally:
        oi._save = original
    check("锁错误重试 3 次后成功落库", len(calls) == 3, len(calls))

    #: ③ 重试到上限仍锁 ⇒ 必须抛出去：上游据此把这份记成失败、下轮重来。
    #:    静默吞掉才是真丢件（这份在 sidecar 里没登记，没人知道它没进去）
    def always_locked(item, outcome, *, dpi, engine):
        raise sqlite3.OperationalError("database is locked")

    oi._save = always_locked
    raised: Exception | None = None
    try:
        oi._save_atomic({}, {}, dpi=200, engine="x", retries=2, backoff=0.01, notify=quiet)
    except sqlite3.OperationalError as exc:
        raised = exc
    finally:
        oi._save = original
    check("重试用尽仍抛（不静默丢这份）", raised is not None, raised)

    #: ④ 非锁的 OperationalError 立刻抛：那是真 bug（表没了、SQL 写错），
    #:    重试只会把错误埋进 60 秒的等待里
    attempts: list[int] = []

    def broken(item, outcome, *, dpi, engine):
        attempts.append(1)
        raise sqlite3.OperationalError("no such table: chunks")

    oi._save = broken
    try:
        oi._save_atomic({}, {}, dpi=200, engine="x", retries=4, backoff=0.01, notify=quiet)
    except sqlite3.OperationalError:
        pass
    finally:
        oi._save = original
    check("非锁错误只试 1 次（别把真 bug 埋进等待里）", len(attempts) == 1, len(attempts))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    case_images()
    case_docx_ocr()
    case_image_doc()
    case_native()
    case_engine()
    case_budget()
    case_dispatch()
    case_segments()
    case_chunk_pages()
    case_salvage()
    case_text()
    case_page_threads()
    case_cli_args()
    case_resources()
    case_reaudit()
    case_save_lock()
    total, failed = CHECKS, FAILED
    print(f"\nSUMMARY: {total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
