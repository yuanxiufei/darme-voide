"""知识库**写能力**自检：覆盖层（编辑/删除/上传台账）＋ 上传链路。

这一层的错误几乎都不抛异常，只是「结果对不上」，所以逐条钉住：

1. **改了就搜得到**：编辑落 ``edits`` ＋ ``edits_fts``，检索要能从第四路命中 ——
   否则用户改完搜旧词还中、搜新词不中，只会以为保存失败。
2. **删除不是物理删**：墓碑只让它在清单/检索/目录树里消失，随时可还原；原库一行不动。
3. **物理删只开给本仓上传件**：库里原有的件是供应商/OCR 的真源，删了下次重建又回来。
4. **块号起点因库而异**：``curated`` 从 0 起、``lib_index`` 从 1 起 ⇒ 越界判据必须看
   「库里真有这一块」，按 ``0–n-1`` 推算会把最后一块判成越界（实测踩过）。
5. **上传三档不许合并**：``ingested`` / ``quarantined``（落盘了但抽不出正文）/
   ``rejected``（没落盘）对用户是三件事，前两档盘上都能找到文件。
6. **相对路径的件要能打开**：``library-extra.sqlite`` 存的是库根下的相对路径，
   没有 ``G:\\…`` 锚点，``_on_disk`` 不认它就会「清单里看得到、点进去说没有」。

夹具是**真 SQLite fts5 库 + 真文件**（不是 mock）—— 出问题的全是「SQL 与盘上现实不一致」。

运行::

    ./.venv/Scripts/python.exe tests/library_overlay_test.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.scripts.corpus import lib_index  # noqa: E402
from app.services import library_ingest as ingest_mod  # noqa: E402
from app.services import library_overlay as ov  # noqa: E402
from app.services import script_library as sl  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []

_G = "G:\\电影剧本\\全球热门与获奖影视剧本知识库"


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _make_db(path: Path, docs: list[tuple], chunks: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, name TEXT, "
        "extension TEXT, size INTEGER, pages INTEGER, status TEXT)")
    conn.execute(
        "CREATE VIRTUAL TABLE chunks USING fts5(document_id, path, name, page, "
        "chunk_no, text, tokenize=trigram)")
    conn.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?)", docs)
    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", chunks)
    conn.commit()
    conn.close()


def _build_fixture(root: Path) -> Path:
    kb = root / "剧本知识库" / "全球热门与获奖影视剧本知识库"
    (kb / "08-编剧方法书籍").mkdir(parents=True)
    (kb / "05-类型片").mkdir(parents=True)
    (kb / "08-编剧方法书籍" / "故事.txt").write_text("x", encoding="utf-8")
    (kb / "05-类型片" / "短剧.txt").write_text("x", encoding="utf-8")
    #: ``05-类型片/补录.doc`` 在盘上（.doc 的真内容无关，_on_disk 只判存在）
    (kb / "05-类型片" / "补录.doc").write_bytes(b"x")

    _make_db(
        kb / "99-索引与检索规则" / "curated-screenplay-methods.sqlite",
        [
            (1, f"{_G}\\08-编剧方法书籍\\故事.txt", "故事.txt", "txt", 100, 1, "indexed"),
            (2, f"{_G}\\05-类型片\\短剧.txt", "短剧.txt", "txt", 100, 1, "indexed"),
        ],
        [
            (1, f"{_G}\\08-编剧方法书籍\\故事.txt", "故事.txt", 1, 0,
             "第一块：激励事件把必备场景投射到观众的想象之中，这是三幕结构的起点。"),
            (1, f"{_G}\\08-编剧方法书籍\\故事.txt", "故事.txt", 1, 1,
             "第二块：原句讲人物的欲望与阻力。"),
            (1, f"{_G}\\08-编剧方法书籍\\故事.txt", "故事.txt", 2, 2,
             "第三块：结尾要兑现承诺。"),
            (2, f"{_G}\\05-类型片\\短剧.txt", "短剧.txt", 7, 0,
             "雨夜霓虹下她转身离开，雨夜之后激励事件才真正落地。"),
        ])
    return kb


def _rows(rel: str, *, source: str = "curated") -> list[tuple]:
    """某一份现在有几块、块号是什么（直接读库，别从函数回执推）。"""
    db = {"curated": sl.METHODS_DB, "extra": sl.EXTRA_DB}[source]
    conn = sqlite3.connect(Path(db).as_posix())
    try:
        return conn.execute(
            "SELECT chunk_no, text FROM chunks WHERE replace(path, '\\', '/') LIKE ? "
            "ORDER BY chunk_no", [f"%{rel}"]).fetchall()
    finally:
        conn.close()


def main() -> int:  # noqa: C901
    root = Path(tempfile.mkdtemp(prefix="overlay_"))
    kb = _build_fixture(root)

    original = (sl.LIBRARY_ROOT, sl.PLAYBOOK_ROOT, sl.METHODS_DB, sl.OCR_DB, sl.EXTRA_DB,
                lib_index.EXTRA_DB, ov.OVERLAY_DB)
    extra_db = root / "extra" / "library-extra.sqlite"
    overlay_db = root / "overlay" / "library-overlay.sqlite"
    sl.LIBRARY_ROOT = kb
    sl.PLAYBOOK_ROOT = root / "no-playbook"
    sl.METHODS_DB = kb / "99-索引与检索规则" / "curated-screenplay-methods.sqlite"
    sl.OCR_DB = root / "no-ocr" / "ocr.sqlite"
    sl.EXTRA_DB = extra_db
    lib_index.EXTRA_DB = extra_db
    ov.OVERLAY_DB = overlay_db
    sl.reset_cache()

    # ================= 1. 覆盖层还没建 =================
    fresh = sl.overlay_report()
    check("空覆盖层: report 如实 available=false 且三个计数都是 0（不编数）",
          fresh["available"] is False and fresh["edits"] == 0
          and fresh["tombstones"] == 0 and fresh["uploads"] == 0, fresh.get("hint"))
    check("空覆盖层: 清单/检索照样能用（没有改动≠没有知识库）",
          sl.list_documents()["total"] == 2
          and sl.search_methods("三幕结构")["results"], sl.list_documents()["total"])

    # ================= 2. document_state（按钮该不该亮） =================
    state = sl.document_state("08-编剧方法书籍/故事.txt")
    check("state: 主库件可编辑可删、但**不能物理删**（不是本仓上传的）",
          state["found"] and state["source"] == "curated" and state["chunks"] == 3
          and state["editable"] and state["deletable"] and not state["hardDeletable"]
          and state["onDisk"] and not state["buried"], state)
    missing = sl.document_state("08-编剧方法书籍/根本没有.txt")
    check("state: 查不到的件 found=false 并给人话理由（不抛错）",
          missing["found"] is False and "索引里没有" in (missing.get("reason") or ""), missing)

    # ================= 3. 编辑：叠加、可撤销、搜得到 =================
    before = _rows("08-编剧方法书籍/故事.txt")
    saved = sl.write_chunk("08-编剧方法书籍/故事.txt", 1,
                           "改后正文里有一个独一无二的关键词：蓝鲸计划。")
    check("改正文: ok 且原库那块一个字没动（改动只在覆盖层）",
          saved["ok"] and _rows("08-编剧方法书籍/故事.txt") == before, saved)

    opened = sl.open_document("08-编剧方法书籍/故事.txt")
    chunk1 = next(item for item in opened["chunks"] if item["chunkNo"] == 1)
    check("翻原文: 读到的是改后正文，同时给出 original 原文",
          chunk1["text"].startswith("改后正文") and chunk1["edited"]
          and "原句讲人物" in (chunk1.get("original") or ""), chunk1)
    check("翻原文: 文档头标 edited/editable/source（前端据此画按钮）",
          opened["document"]["edited"] and opened["document"]["editable"]
          and opened["document"]["source"] == "curated"
          and not opened["document"]["buried"], opened["document"])
    check("翻原文: 没改过的块 edited=false（不能整篇都打「已修改」）",
          [item["edited"] for item in opened["chunks"]] == [False, True, False],
          [item["edited"] for item in opened["chunks"]])

    hit = sl.search_methods("蓝鲸计划")
    check("改完搜得到: 第四路 edits 命中（否则用户会以为没保存）",
          hit["results"] and hit["results"][0].get("from") == "edits"
          and hit["parts"]["edits"]["count"] >= 1, hit.get("parts"))
    listed = sl.list_documents(only="edited")
    check("清单: only=edited 只出改过的那份，且条目带 edited/source/buried 三字段",
          listed["total"] == 1 and listed["items"][0]["path"] == "08-编剧方法书籍/故事.txt"
          and listed["items"][0]["edited"] is True
          and listed["items"][0]["source"] == "curated"
          and listed["items"][0]["buried"] is False, listed["items"])

    bad_range = sl.write_chunk("08-编剧方法书籍/故事.txt", 99, "越界")
    check("改正文: 块号越界当场拒，并报出真实块号区间",
          bad_range["ok"] is False and "越界" in bad_range["reason"]
          and "0–2" in bad_range["reason"], bad_range)
    title = sl.write_chunk("08-编剧方法书籍/故事.txt", ov.META_CHUNK, "故事（第二版）")
    check("改标题: chunk_no=-1 写文档头，翻原文 title 跟着变",
          title["ok"] and sl.open_document("08-编剧方法书籍/故事.txt")
          ["document"]["title"] == "故事（第二版）", title)

    undone = sl.undo_chunk("08-编剧方法书籍/故事.txt", 1)
    reopened = sl.open_document("08-编剧方法书籍/故事.txt")
    check("撤销: 回到原库正文，edited 标记同时消失",
          undone["ok"] and undone["dropped"]
          and not next(item for item in reopened["chunks"] if item["chunkNo"] == 1)["edited"],
          undone)
    again = sl.undo_chunk("08-编剧方法书籍/故事.txt", 1)
    check("撤销: 本来没改过时 dropped=false 但**不算错误**（幂等）",
          again["ok"] and again["dropped"] is False, again)
    check("撤销后: 新词搜不到了（覆盖层 fts 一起删干净，不留孤儿）",
          not sl.search_methods("蓝鲸计划")["results"],
          sl.search_methods("蓝鲸计划")["results"])

    # ================= 4. 块号起点：extra 从 1 起 =================
    extra_res = ingest_mod.ingest(
        "补录正文：独一无二的补录句子，讲一个侠客在雨夜做过的事。".encode("utf-8"),
        "补录.txt")
    check("上传: 文本件 ingested，落在 00-本仓补录/ 下并真的抽出正文",
          extra_res["status"] == "ingested" and extra_res["chunks"] >= 1
          and extra_res["path"].startswith("00-本仓补录/"), extra_res)
    extra_state = sl.document_state(extra_res["path"])
    check("state: 上传件归 extra、盘上找得到 ⇒ **可以物理删**",
          extra_state["found"] and extra_state["source"] == "extra"
          and extra_state["hardDeletable"] and extra_state["uploaded"]
          and extra_state["onDisk"], extra_state)
    extra_rows = _rows(extra_res["path"], source="extra")
    check("上传件的块号是 1 起（lib_index 口径，与供应商库的 0 起不同）",
          [no for no, _ in extra_rows] == [1], extra_rows)
    last_ok = sl.write_chunk(extra_res["path"], 1, "补录正文改过：独一无二的改后句子。")
    zero_bad = sl.write_chunk(extra_res["path"], 0, "不存在的第 0 块")
    check("改正文: 1 起块号的最后一块能改（按 0–n-1 推算的旧写法在这里会误报越界）",
          last_ok["ok"] is True, last_ok)
    check("改正文: 不存在的第 0 块被拒（起点不是 0 的库不能放过它）",
          zero_bad["ok"] is False and "越界" in zero_bad["reason"], zero_bad)
    check("上传件改完也能搜到（第四路不排除 extra 的 path）",
          any(row.get("from") == "edits"
              for row in sl.search_methods("改后句子")["results"]),
          sl.search_methods("改后句子").get("parts"))

    # ================= 5. staleEdits（原库重建过） =================
    sl.write_chunk("08-编剧方法书籍/故事.txt", 2, "改第三块")
    conn = sqlite3.connect(Path(sl.METHODS_DB).as_posix())
    with conn:
        conn.execute("DELETE FROM chunks WHERE document_id = 1 AND chunk_no = 2")
    conn.close()
    sl.reset_cache()
    stale = sl.open_document("08-编剧方法书籍/故事.txt")
    check("staleEdits: 指向已不存在块的编辑如实报出来，不假装还在",
          stale.get("staleEdits") == [2] and "重建" in (stale.get("hint") or ""), stale)
    #: 与 stale 相反的方向：extra 的块号从 1 起，按 total 推算会把**最后一块**误报成 stale。
    extra_open = sl.open_document(extra_res["path"])
    check("staleEdits: 1 起块号的最后一块编辑**不**被误报成 stale",
          "staleEdits" not in extra_open and extra_open["chunks"][0]["edited"], extra_open)

    # ================= 6. 删除（墓碑）与还原 =================
    buried = sl.bury_document("05-类型片/短剧.txt", reason="测试删除")
    check("删除: 默认走墓碑（不物理删），回执说清可还原",
          buried["ok"] and buried["mode"] == "tombstone"
          and (Path(kb) / "05-类型片" / "短剧.txt").exists(), buried)
    live = {item["path"] for item in sl.list_documents()["items"]}
    check("删除: 清单里不再出现（默认档看不见删掉的）",
          "05-类型片/短剧.txt" not in live, sorted(live))
    trash = sl.list_documents(only="buried")
    check("回收站: only=buried 看得见，且条目带 buried=true",
          trash["total"] == 1 and trash["items"][0]["path"] == "05-类型片/短剧.txt"
          and trash["items"][0]["buried"] is True, trash["items"])
    hidden = sl.search_methods("雨夜")
    check("删除: 检索里剔掉并报 hidden.count（不许静默少几条）",
          all(row["path"] != "05-类型片/短剧.txt" for row in hidden["results"])
          and (hidden.get("hidden") or {}).get("count") == 1, hidden)
    check("删除: 目录树的计数也跟着少（树/清单/检索三处同一口径）",
          "05-类型片" not in {group["path"] for group in sl.library_tree(depth=1)["groups"]},
          [group["path"] for group in sl.library_tree(depth=1)["groups"]])
    absent = sl.bury_document("05-类型片/根本没这份.txt")
    check("删除: 查不到的件 ok=false + 理由（不抛异常、也不写墓碑）",
          absent["ok"] is False and ov.buried_map() == {"curated": {"05-类型片/短剧.txt"}},
          absent)

    hard_refused = sl.bury_document("08-编剧方法书籍/故事.txt", hard=True)
    check("物理删: 库里原有的件被拒（只有本仓上传件能彻底删）",
          hard_refused["ok"] is False and "本仓" in hard_refused["reason"], hard_refused)

    restored = sl.raise_document("05-类型片/短剧.txt")
    check("还原: 回到清单、检索重新命中、回收站清空",
          restored["ok"] and restored["restored"]
          and "05-类型片/短剧.txt" in
          {item["path"] for item in sl.list_documents()["items"]}
          and sl.search_methods("雨夜")["results"]
          and sl.list_documents(only="buried")["total"] == 0, restored)

    # ================= 7. 上传：三档与校验 =================
    dup = ingest_mod.ingest("再来一份同名的正文，用来验证不覆盖。".encode("utf-8"), "补录.txt")
    check("上传: 同名不覆盖，自动落成 -2（用户重传不会悄悄盖掉上一份）",
          dup["status"] == "ingested" and dup["path"].endswith("-2.txt"), dup["path"])
    #: 样本必须挑**真不支持**的扩展名。2026-11-07 起 ``.pdf`` 已归第三库抽文字层，
    #: 拿它当「不支持」的样本会让本条判据恒假（假 PDF 会落盘成 ``quarantined``）。
    epub = ingest_mod.ingest(b"PK\x03\x04 not really an ebook", "剧本.epub")
    check("上传: 不支持的扩展名 rejected，并说明本库只收哪些类型",
          epub["status"] == "rejected" and "不支持" in epub["reason"]
          and "TXT" in epub["reason"], epub)
    empty = ingest_mod.ingest(b"", "空.txt")
    check("上传: 0 字节 rejected", empty["status"] == "rejected"
          and "空文件" in empty["reason"], empty)
    keep_limit = ingest_mod.MAX_FILE_SIZE
    ingest_mod.MAX_FILE_SIZE = 32
    big = ingest_mod.ingest("这是一段比上限更长的正文内容".encode("utf-8") * 4, "大.txt")
    ingest_mod.MAX_FILE_SIZE = keep_limit
    check("上传: 超过上限 rejected，理由带上限（由后端判，不靠前端自觉）",
          big["status"] == "rejected" and "太大" in big["reason"], big)
    junk = ingest_mod.ingest(bytes([0, 1, 2, 3, 0, 5, 0, 7]), "垃圾.txt")
    junk_ok = (kb / junk["path"]).exists() if junk["path"] else False
    check("上传: 抽不出正文 quarantine（**文件在盘上**，不是 rejected）",
          junk["status"] == "quarantined" and junk_ok and "已落盘" in junk["reason"], junk)
    #: 假 PDF（正文层抽不出，等同扫描件）**不是** ``rejected``：文件必须留在盘上，
    #: 否则 ``ocr_index._extra_quarantined_docx()`` 那条 OCR 补救回路就永远收不到它。
    scanned = ingest_mod.ingest(b"%PDF-1.4 fake but not a real pdf", "扫描件.pdf")
    scanned_ok = (kb / scanned["path"]).exists() if scanned["path"] else False
    check("上传: 抽不出字层的 PDF 走 quarantine（文件在盘上，留给 OCR 线接手）",
          scanned["status"] == "quarantined" and scanned_ok
          and "已落盘" in scanned["reason"], scanned)
    check("上传: rejected 的理由里**不**说「已落盘」（两档不许说成同一件事）",
          "已落盘" not in epub["reason"] and "已落盘" not in empty["reason"],
          [epub["reason"], empty["reason"]])
    check("上传: safe_name 挡住路径穿越、保住扩展名、过长截断",
          ingest_mod.safe_name("../../evil.doc") == "evil.doc"
          and ingest_mod.safe_name("a/b/c.doc") == "c.doc"
          and ingest_mod.safe_name("长" * 300 + ".docx").endswith(".docx")
          and len(ingest_mod.safe_name("长" * 300 + ".docx")) <= 120,
          ingest_mod.safe_name("../../evil.doc"))

    ledger = {row["status"] for row in ov.list_uploads(limit=50)}
    check("台账: 三档都记下来了（ingested / quarantined / rejected 各成一类）",
          {"ingested", "quarantined"} <= ledger and "rejected" not in ledger, ledger)
    report = sl.overlay_report()
    #: edits=3（故事第 2 块 ＋ 文档头标题 ＋ 上传件第 1 块；撤销掉的两条不该还在）、
    #: uploads=4（补录.txt／补录-2.txt／垃圾.txt／扫描件.pdf）—— ``rejected`` **不进台账**：
    #: 它没落盘，记进去只会让「盘上有多少补录件」对不上数。
    check("盘点: edits/tombstones/uploads 三个计数与操作次数对得上",
          report["available"] and report["edits"] == 3 and report["tombstones"] == 0
          and report["uploads"] == 4 and report["writableSources"] == ["curated", "ocr", "extra"],
          {key: report[key] for key in ("edits", "tombstones", "uploads")})

    # ================= 8. 物理删上传件 =================
    target = extra_res["path"]
    hard = sl.bury_document(target, hard=True)
    check("物理删: 上传件连带索引记录与盘上文件一起没了",
          hard["ok"] and hard["mode"] == "hard" and hard["fileRemoved"]
          and not (kb / target).exists()
          and not _rows(target, source="extra"), hard)
    check("物理删: 删完 document_state 报 found=false（不留一条打不开的记录）",
          sl.document_state(target)["found"] is False, sl.document_state(target))
    check("物理删: 台账里的上传记录**留着**（事后要能查到传过什么）",
          any(row["path"] == target for row in ov.list_uploads(limit=50)),
          [row["path"] for row in ov.list_uploads(limit=50)])

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")

    (sl.LIBRARY_ROOT, sl.PLAYBOOK_ROOT, sl.METHODS_DB, sl.OCR_DB, sl.EXTRA_DB,
     lib_index.EXTRA_DB, ov.OVERLAY_DB) = original
    sl.reset_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    #: Windows 中文控制台是 GBK ⇒ 不重配 stdout 会在第一条 print 就 UnicodeEncodeError，
    #: 一条判据都没跑就退 1（`run_all` 下不露头，因为那边已经重配过）。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
