"""第三方剧本/编剧知识库检索层自检（``app/services/script_library.py``）。

这层最容易犯的错**都不报错、只是结果不对**，所以逐条钉住：

1. **旧盘路径重映射**：索引里是建库机器上的 ``G:\\电影剧本\\…``。搬过盘的机器上如果不重映射，
   结果里全是「盘上不存在」，模型却照样当参考用 ⇒ 必须逐条回报 ``onDisk``，缺文件就是 ``false``。
2. **trigram 的 3 字下限**：1–2 字词走 ``MATCH`` 恒空（不是没内容）⇒ 必须自动回落 ``LIKE``，
   且回执里 ``ranking`` 要如实区分（两种排序不可比）。
3. **隔离目录** ``90-隔离与待审/`` 不许参与检索 —— 现在库内是 0 条，但那是数据现状、不是契约。
4. **缺库降级**：``available=false`` + 安装提示，**不抛错**（否则 Agent 直接跑不动）。
5. **盘点回执**：库多大、缺几份要能自证（与 prompt 语料同一条「不许假绿」纪律）。
6. **命令行入口**（``app/scripts/corpus/library.py``）：多词查询在 PowerShell 里会被空格切开 ——
   位置参数必须**全部**拼进查询，``--q`` 与位置参数同时给则当场退出 2（不许静默丢词）。
7. **目录树/文档清单**（``/knowledge/tree``、``/knowledge/documents`` 的数据源）：``pending``
   必须是「供应商没抽正文、本仓也还没 OCR」的**真数**，不是 ``documents − withText`` 那种减法
   （OCR 已补的那批既非 withText、也不该算待补）；``90-隔离与待审/`` 连计数都不许进。

夹具是一个**真 SQLite fts5 trigram 库** + 真文件（不是 mock）—— 因为出问题的全是
「SQL 与盘上现实不一致」，mock 掉就什么也测不到。

运行::

    ./.venv/Scripts/python.exe tests/script_library_test.py
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.tools.library_tools import create_library_tools  # noqa: E402
from app.scripts.corpus import library as cli  # noqa: E402
from app.services import script_library as sl  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []

_G = "G:\\电影剧本\\全球热门与获奖影视剧本知识库"


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _call(tool_id: str, arguments: dict) -> dict:
    return asyncio.run(create_library_tools()[tool_id].run(arguments))


def _build_fixture(root: Path) -> tuple[Path, Path]:
    """造 ``剧本知识库/…/curated-screenplay-methods.sqlite`` + 盘上真文件 + 参考 md。"""
    kb = root / "剧本知识库" / "全球热门与获奖影视剧本知识库"
    db = kb / "99-索引与检索规则" / "curated-screenplay-methods.sqlite"
    db.parent.mkdir(parents=True)

    (kb / "08-编剧方法书籍").mkdir(parents=True)
    (kb / "05-类型片").mkdir(parents=True)
    (kb / "08-编剧方法书籍" / "故事.txt").write_text("x", encoding="utf-8")
    (kb / "05-类型片" / "短剧.txt").write_text("x", encoding="utf-8")
    # 90-隔离与待审 与「99-缺盘」都**不在盘上**（前者是隔离桶，后者模拟搬盘缺文件）

    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, name TEXT, "
        "extension TEXT, size INTEGER, pages INTEGER, status TEXT)")
    conn.execute(
        "CREATE VIRTUAL TABLE chunks USING fts5(document_id, path, name, page, "
        "chunk_no, text, tokenize=trigram)")
    rows = [
        (1, f"{_G}\\08-编剧方法书籍\\故事.txt", "故事.txt", "txt", 100, 1, "indexed"),
        (2, f"{_G}\\05-类型片\\短剧.txt", "短剧.txt", "txt", 100, 1, "indexed"),
        (3, f"{_G}\\90-隔离与待审\\草稿.txt", "草稿.txt", "txt", 100, 1, "indexed"),
        (4, f"{_G}\\99-缺盘\\不存在.pdf", "不存在.pdf", "pdf", 100, 1, "indexed"),
    ]
    conn.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?)", rows)
    chunks = [
        (1, f"{_G}\\08-编剧方法书籍\\故事.txt", "故事.txt", 1, 0,
         "激励事件把必备场景投射到观众的想象之中，这是三幕结构的起点。"),
        (2, f"{_G}\\05-类型片\\短剧.txt", "短剧.txt", 7, 0,
         "雨夜霓虹下她转身离开。雨夜之后，激励事件才真正落地。"),
        (3, f"{_G}\\90-隔离与待审\\草稿.txt", "草稿.txt", 1, 0,
         "这是待审草稿，也写了三幕结构与激励事件，但绝不该被检索到。"),
        (4, f"{_G}\\99-缺盘\\不存在.pdf", "不存在.pdf", 3, 0,
         "这份文件索引里有三幕结构，但盘上没有。"),
    ]
    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", chunks)
    conn.commit()
    conn.close()

    playbook = root / "playbook"
    acting = playbook / "film-expert-bridge"
    acting.mkdir(parents=True)
    (acting / "SKILL.md").write_text(
        "# 表演层\n\n## 微表情与反应\n\n写“他很伤心”不算表演指令；改成下颌收紧、鼻翼微张、眨眼停顿，"
        "让情绪藏进肌肉。近景优先用微表情，远景优先用身体姿态。\n\n## 动作节拍\n\n"
        "打戏按 1.5 秒一次攻防接触切，开打必须有攻防接触。\n", encoding="utf-8")
    return kb, playbook


def main() -> int:  # noqa: C901
    root = Path(tempfile.mkdtemp(prefix="scriptlib_"))
    kb, playbook = _build_fixture(root)
    #: ``EXTRA_DB`` **不进** ``original``：下面几处 ``= original`` 只在两库夹具之间来回
    #: 恢复，第三库一次设好就不再动 —— 进了元组反而会在某处恢复时把真库放回来。
    original = (sl.LIBRARY_ROOT, sl.PLAYBOOK_ROOT, sl.METHODS_DB, sl.OCR_DB)
    sl.LIBRARY_ROOT = kb
    sl.PLAYBOOK_ROOT = playbook
    sl.METHODS_DB = kb / "99-索引与检索规则" / "curated-screenplay-methods.sqlite"
    #: OCR sidecar 指向夹具里**不存在**的路径：真库里那份是开发机自己跑出来的，
    #: 不隔离就会让「两源都缺」这条断言读到真 OCR 库（实测踩过一次）。
    sl.OCR_DB = kb / "ocr" / "不存在.sqlite"
    #: 第三个库（``app/scripts/corpus/lib_index.py`` 补的 ``.doc``／RTF）同样要隔离。
    #: 它比 sidecar 更容易漏：夹具只重指了 ``METHODS_DB``/``OCR_DB``，两库变三库之后
    #: 这台机器上真库已有 880 份 / 31157 块，于是按「夹具 7 份」写的 totals、清单、
    #: 短词回落等 14 条断言一起红（``documents`` 报 887、清单报 368 条）。
    sl.EXTRA_DB = Path(tempfile.mkdtemp(prefix="no-extra-")) / "library-extra.sqlite"
    sl.reset_cache()

    # ================= 1. 路径重映射 =================
    resolved = sl.resolve_local(f"{_G}\\08-编剧方法书籍\\故事.txt")
    check("重映射: G:\\ 旧盘路径 -> 本仓 LIBRARY_ROOT 下同名相对路径",
          resolved is not None and resolved.as_posix().endswith("08-编剧方法书籍/故事.txt")
          and str(kb).replace("\\", "/") in resolved.as_posix(), resolved)
    check("重映射: 没有锚点的路径 -> None（**不猜**）",
          sl.resolve_local("D:\\别的地方\\故事.txt") is None
          and sl.resolve_local("") is None)

    # ================= 2. fts5 长词 / LIKE 短词 =================
    long_hit = sl.search_methods("三幕结构", limit=5)
    paths = [row["path"] for row in long_hit["results"]]
    check("长词: 走 bm25 且命中数来自真库（2 段；隔离桶那条不算）",
          long_hit["ranking"] == "bm25" and long_hit["matched"] == 2, long_hit["matched"])
    check("长词: 隔离目录 90-隔离与待审 绝不出现",
          not any("90-隔离与待审" in path for path in paths), paths)
    check("长词: snippet 带【】标记且压平空白",
          any("【三幕结构】" in row["excerpt"] for row in long_hit["results"]),
          [row["excerpt"][:40] for row in long_hit["results"]])
    check("长词: 盘上有的 onDisk=true、缺的 onDisk=false（**不假设文件在**）",
          {row["title"]: row["onDisk"] for row in long_hit["results"]}
          == {"故事.txt": True, "不存在.pdf": False},
          {row["title"]: row["onDisk"] for row in long_hit["results"]})

    short_hit = sl.search_methods("雨夜", limit=5)
    check("短词: 2 字词自动回落 LIKE（MATCH 恒空的话这里会是 0 条）",
          short_hit["ranking"] == "like" and short_hit["matched"] is None
          and [row["title"] for row in short_hit["results"]] == ["短剧.txt"],
          short_hit)
    check("短词: LIKE 路线的片段自己标出命中词（没有 snippet() 可用）",
          "【雨夜】" in short_hit["results"][0]["excerpt"], short_hit["results"][0]["excerpt"])

    mixed = sl.search_methods("雨夜 激励事件", limit=5)
    check("混合词: 含短词时整条走 LIKE，且**全部命中**优先（同段共现）",
          mixed["ranking"] == "like" and [row["title"] for row in mixed["results"]] == ["短剧.txt"],
          (mixed["ranking"], [row["title"] for row in mixed["results"]]))

    loose = sl.search_methods("雨夜 三幕结构", limit=5)
    check("放宽: 无同段共现时降到「任一命中」并如实标注 like-or",
          loose["ranking"] in ("like", "like-or")
          and (loose["ranking"] == "like-or") == (len(loose["results"]) > 1),
          (loose["ranking"], len(loose["results"])))

    check("转义: 词里的 % / _ 当字面量（不会把整库都命中）",
          sl.search_methods("100%三幕", limit=5)["results"] == [],
          sl.search_methods("100%三幕", limit=5)["results"])
    check("BOM/零宽: 检索词被清掉（PowerShell 的 --q-file 会带 \\ufeff）",
          [row["title"] for row in sl.search_methods("\ufeff雨夜", limit=5)["results"]]
          == ["短剧.txt"])
    check("path_contains: 只在指定路径下找",
          [row["title"] for row in sl.search_methods("三幕结构", limit=5,
                                                     path_contains="08-编剧方法")["results"]]
          == ["故事.txt"])

    # ================= 3. 方法文本（md 小节）=================
    book = sl.search_playbook("微表情", limit=3)
    check("playbook: 命中小节且标题/正文都参与打分",
          book["results"][0]["title"].startswith("微表情与反应")
          and book["results"][0]["source"] == "playbook", book["results"][0])
    covered = sl.search_playbook("微表情 近景 反应", limit=3)
    check("playbook: 三个词全落在同一小节时 coverage=1.0（覆盖率真的算出来了，不是恒 1）",
          covered["results"][0]["coverage"] == 1.0
          and covered["results"][0]["title"].startswith("微表情与反应"), covered["results"][0])
    partial = sl.search_playbook("微表情 近景 打戏", limit=3)
    check("playbook: 只命中部分词时 coverage < 1，且穷尽性词（打戏）拉进另一小节",
          partial["results"][0]["coverage"] < 1.0
          and any("动作节拍" in row["title"] for row in partial["results"]),
          [(row["title"], row["coverage"]) for row in partial["results"]])
    check("playbook: 只有标题没正文的小节不进索引（检索到它等于没检索）",
          all(len(row["excerpt"]) > 10 for row in book["results"]))

    sections_first = sl._playbook_sections()
    check("缓存: 同一进程内二次检索复用同一份小节表（223 份不重复读盘）",
          sl._playbook_sections() is sections_first)
    skill = playbook / "film-expert-bridge" / "SKILL.md"
    stat = skill.stat()
    os.utime(skill, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    check("缓存: 文件 mtime 变了就必须重读（否则改了资料，检索还给旧内容）",
          sl._playbook_sections() is not sections_first)

    # ================= 4. 盘点 / 降级 =================
    report = sl.library_report()
    check("盘点: methods 的文档数/片段数/缺盘数来自真库（含隔离桶与缺文件各 1）",
          report["methods"]["documents"] == 4 and report["methods"]["chunks"] == 4
          and report["methods"]["missingFiles"] == 2, report["methods"])
    check("盘点: playbook 给出文件数 >= 1 且正文字数 > 0",
          report["playbook"]["files"] >= 1 and report["playbook"]["chars"] > 50,
          report["playbook"])

    merged = sl.search("激励事件", limit=3)
    check("合并: all 同时查两源，逐源给 available/ranking/matched/count",
          set(merged["sources"]) == {"methods", "playbook"}
          and merged["sources"]["methods"]["count"] >= 1
          and merged["corpus"]["methods"]["documents"] == 4, merged["sources"])
    check("合并: 空检索词 -> available 但 results 空 + hint（盘点照给）",
          sl.search("")["results"] == [] and sl.search("")["hint"] == "检索词为空。"
          and sl.search("")["corpus"]["methods"]["documents"] == 4)
    try:
        sl.search("x", source="nonsense")
        check("未知来源: 当场 ValueError（不静默兜成 all）", False)
    except ValueError:
        check("未知来源: 当场 ValueError（不静默兜成 all）", True)

    sl.LIBRARY_ROOT = root / "空" / "不存在"
    sl.PLAYBOOK_ROOT = root / "空" / "也不存在"
    sl.METHODS_DB = sl.LIBRARY_ROOT / "x.sqlite"
    sl.OCR_DB = sl.LIBRARY_ROOT / "x-ocr.sqlite"
    sl.reset_cache()
    missing = sl.search("三幕结构")
    #: playbook 的提示由「未安装」改为「缺失」：2026-11-04 起它随仓库分发
    #: （``data/playbook/``），正常克隆里本来就该有 —— 再喊「未安装」会让人
    #: 去装一个不需要装的东西，而真因通常是目录被误删或被 SCRIPT_PLAYBOOK_ROOT 指偏。
    check("降级: 两源都缺 -> available=false + 安装提示（不抛错）",
          missing["available"] is False and missing["results"] == []
          and "剧本知识库未安装" in missing["hint"]
          and "参考方法文本缺失" in missing["hint"], missing["hint"])
    check("降级: 缺库时 corpus 两个来源都是 None（**不编 0**）",
          missing["corpus"] == {"methods": None, "playbook": None}, missing["corpus"])

    # ================= 5. Agent 工具 =================
    sl.LIBRARY_ROOT, sl.PLAYBOOK_ROOT, sl.METHODS_DB, sl.OCR_DB = original
    sl.LIBRARY_ROOT = kb
    sl.PLAYBOOK_ROOT = playbook
    sl.METHODS_DB = kb / "99-索引与检索规则" / "curated-screenplay-methods.sqlite"
    sl.OCR_DB = kb / "ocr" / "不存在.sqlite"
    sl.reset_cache()
    tools = create_library_tools()
    check("工具: 只挂一个 search_script_library", list(tools) == ["search_script_library"], list(tools))
    check("工具: query 必填，其余可选",
          tools["search_script_library"].input_schema["required"] == ["query"],
          tools["search_script_library"].input_schema)
    output = _call("search_script_library", {"query": "三幕结构", "source": "methods"})
    check("工具: 透传 service 的结果形状（不另造字段）",
          {"available", "results", "sources", "corpus", "hint", "note"} <= set(output)
          and output["sources"]["methods"]["count"] >= 1, list(output))
    check("工具: limit 夹到 [1,8]（传 99 也不会塞爆上下文）",
          len(_call("search_script_library", {"query": "三幕结构", "limit": 99})["results"])
          <= 16)
    check("工具: 非法 source 兜回 all（不报错）",
          _call("search_script_library", {"query": "三幕结构", "source": "xxx"})["source"] == "all")

    #: ⚠️ 后面几节（6/7/8）都必须留在**夹具**上跑。原来这里恢复的是 ``original``（真库），
    #: 而 section 7 只重指了 ``OCR_DB`` ⇒ 主库仍指真库：断言只是**偶然**通过（真库里恰好
    #: 也有这些词），一旦换机器/换库就全红，而且会去读 19 GB 的供应商索引。
    sl.LIBRARY_ROOT, sl.PLAYBOOK_ROOT, sl.METHODS_DB, sl.OCR_DB = original
    sl.LIBRARY_ROOT = kb
    sl.PLAYBOOK_ROOT = playbook
    sl.METHODS_DB = kb / "99-索引与检索规则" / "curated-screenplay-methods.sqlite"
    sl.OCR_DB = kb / "ocr" / "不存在.sqlite"
    sl.reset_cache()

    # ================= 6. 命令行入口 =================
    #: 实测踩过：PowerShell 里 ``--q=雨夜 追逐`` 会被空格切成 ``--q=雨夜`` ＋ ``追逐``，
    #: 而旧写法只看 ``--q`` ⇒ 第二个词**静默丢掉**、检索结果照样出一页（说谎 ✗）。
    def _argv(argv: list[str]):
        saved = sys.argv
        sys.argv = argv
        try:
            return cli._query()
        finally:
            sys.argv = saved

    check("命令行: 位置参数多词 -> 全部拼进查询（不静默丢第二个词）",
          _argv(["prog", "雨夜", "追逐"]) == "雨夜 追逐", _argv(["prog", "雨夜", "追逐"]))
    check("命令行: 引号包住的 --q 原样当整句",
          _argv(["prog", "--q=雨夜 追逐"]) == "雨夜 追逐", _argv(["prog", "--q=雨夜 追逐"]))
    #: 冲突那一支在**检索之前**就返回 2 ⇒ 不会真查盘；stderr 只吞掉这句提示。
    saved, out = sys.argv, io.StringIO()
    sys.argv = ["prog", "--q=雨夜", "追逐"]
    try:
        with contextlib.redirect_stderr(out):
            code = cli.main()
    finally:
        sys.argv = saved
    check("命令行: --q 与位置参数同时给 -> 退出 2（位置参数会被忽略，不许装作没发生）",
          code == 2 and "位置参数" in out.getvalue(), {"code": code, "stderr": out.getvalue()})

    # ================= 7. OCR sidecar（扫描件补进来的正文）=================
    #: 主库里 3139 份扫描件**登记了但没正文**；``ocr_index.py`` OCR 出来的正文进 sidecar。
    #: 这一节盯三件事：① 两库结果能合到一起 ② ``from`` 分得清 ③ 翻原文会落到 sidecar。
    sidecar = kb / "ocr" / "library-ocr.sqlite"
    sidecar.parent.mkdir(parents=True)
    conn = sqlite3.connect(sidecar)
    conn.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT UNIQUE, name TEXT, "
        "extension TEXT, size INTEGER, modified_ns INTEGER, signature TEXT, pages INTEGER, "
        "status TEXT)")
    conn.execute(
        "CREATE VIRTUAL TABLE chunks USING fts5(document_id, path, name, page, "
        "chunk_no, text, tokenize=trigram)")
    conn.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO documents VALUES (1, ?, '扫描件.pdf', 'pdf', 900, 1, '900:1', 5, 'ocr')",
                 [f"{_G}\\08-编剧方法书籍\\扫描件.pdf"])
    conn.execute("INSERT INTO chunks VALUES (1, ?, '扫描件.pdf', 5, 1, ?)",
                 [f"{_G}\\08-编剧方法书籍\\扫描件.pdf",
                  "这份扫描件的正文写了幻影骑士与三幕结构的差异，也谈序列节拍；"
                  "主库里没有这一份的文字层。"])
    conn.execute("INSERT INTO meta VALUES ('engine', 'rapidocr-onnxruntime')")
    conn.commit()
    conn.close()
    sl.OCR_DB = sidecar
    sl.reset_cache()

    merged = sl.search("三幕结构", source="methods", limit=6)
    tags = {row.get("from") for row in merged["results"]}
    check("sidecar: 两库结果合到同一份 methods 里（curated 与 ocr 都在）",
          tags == {"curated", "ocr"}, tags)
    check("sidecar: 回执分列两库的秩与命中数",
          set(merged["sources"]["methods"].get("parts") or {}) == {"curated", "ocr"},
          merged["sources"]["methods"].get("parts"))
    check("sidecar: 只有 OCR 库才有的词也能检索到（'幻影骑士'）",
          [row.get("from") for row in sl.search("幻影骑士", source="methods")["results"]]
          == ["ocr"], sl.search("幻影骑士", source="methods")["results"])
    ocr_doc = sl.open_document("08-编剧方法书籍/扫描件.pdf")
    check("sidecar: 翻原文落到 sidecar，且 origin 标明出处",
          ocr_doc["origin"] == "ocr" and ocr_doc["total"] == 1
          and ocr_doc["chunks"][0]["page"] == 5, ocr_doc)
    curated_doc = sl.open_document("08-编剧方法书籍/故事.txt")
    check("sidecar: 主库有正文的文档仍走 curated（不被 sidecar 抢）",
          curated_doc["origin"] == "curated", curated_doc["origin"])
    check("sidecar: 盘点单列（documents/chunks/pages/partial/engine）",
          (sl.library_report().get("ocr") or {}).get("engine") == "rapidocr-onnxruntime",
          sl.library_report().get("ocr"))
    sl.OCR_DB = kb / "ocr" / "不存在.sqlite"
    sl.reset_cache()

    # ================= 8. 目录树 / 文档清单（前端浏览态的数据源）=================
    #: 真实库的口径：sidecar 只 OCR **主库已登记**的文档（``ocr_index._pending`` 只遍历
    #: ``documents`` 表），所以「登记了、但主库无正文块」才是待补的那批。夹具照此造：
    #: 扫描件.pdf 两边都有（主库无正文 ＋ sidecar 有正文 ⇒ 已补），待补.pdf / 深度.pdf 两边都没有。
    conn = sqlite3.connect(kb / "99-索引与检索规则" / "curated-screenplay-methods.sqlite")
    conn.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?)", [
        (5, f"{_G}\\08-编剧方法书籍\\扫描件.pdf", "扫描件.pdf", "pdf", 900, 5, "indexed"),
        (6, f"{_G}\\06-期刊\\待补.pdf", "待补.pdf", "pdf", 200, 2, "indexed"),
        (7, f"{_G}\\08-编剧方法书籍\\子目录\\深度.pdf", "深度.pdf", "pdf", 300, 3, "indexed"),
        #: 第 9 节要用的那份**半成品**：sidecar 里 status='partial'、只扫了 4 页中的第 1 页。
        #: 它登记在主库（否则 `_scan_documents` 根本扫不到），正文全在 sidecar。
        (8, f"{_G}\\06-期刊\\半成品.pdf", "半成品.pdf", "pdf", 400, 4, "indexed"),
    ])
    conn.commit()
    conn.close()
    side = sqlite3.connect(sidecar)
    side.execute(
        "INSERT INTO documents VALUES (2, ?, '半成品.pdf', 'pdf', 400, 8, '400:8', 4, 'partial')",
        [f"{_G}\\06-期刊\\半成品.pdf"])
    side.execute("INSERT INTO chunks VALUES (2, ?, '半成品.pdf', 1, 1, ?)",
                 [f"{_G}\\06-期刊\\半成品.pdf",
                  "这份只扫了第一页，正文里有青苔灯笼四个字；剩下的三页还没跑。"])
    side.commit()
    side.close()
    sl.OCR_DB = sidecar
    sl.reset_cache()

    tree = sl.library_tree(depth=2)
    check("目录树: totals 七个计数与夹具逐条对得上"
          "（7 份 / 3 有正文 / 1 已 OCR / 1 半成品 / 3 待补 / 2100 字节 / 17 页）",
          tree["available"] and tree["totals"] == {"documents": 7, "withText": 3, "ocr": 1,
                                                  "ocrPartial": 1, "pending": 3,
                                                  "bytes": 2100, "pages": 17},
          tree["totals"])
    check("目录树: 半成品计入 pending 但仍单列 ocrPartial（withText+ocr+pending 才等于总数）",
          tree["totals"]["withText"] + tree["totals"]["ocr"] + tree["totals"]["pending"]
          == tree["totals"]["documents"]
          and tree["totals"]["ocrPartial"] <= tree["totals"]["pending"], tree["totals"])
    check("目录树: 隔离桶 90-隔离与待审 连计数都不进树（documents=7 里没有那条草稿）",
          not any("90-隔离与待审" in group["path"] for group in tree["groups"])
          and tree["totals"]["documents"] == 7, [g["path"] for g in tree["groups"]])
    check("目录树: groups 按文档数降序、同数按路径升序（前端不许自己再排一次）",
          [g["path"] for g in tree["groups"]]
          == ["06-期刊", "08-编剧方法书籍", "05-类型片",
              "08-编剧方法书籍/子目录", "99-缺盘"],
          [g["path"] for g in tree["groups"]])
    group = next(g for g in tree["groups"] if g["path"] == "08-编剧方法书籍")
    check("目录树: 组内 bytes/pages 是该目录的合计（1000 字节 / 6 页）",
          (group["bytes"], group["pages"], group["documents"]) == (1000, 6, 2), group)
    check("目录树: depth=2 把子目录单列出来（08-编剧方法书籍/子目录 只含深度.pdf）",
          next(g for g in tree["groups"]
               if g["path"] == "08-编剧方法书籍/子目录")["pending"] == 1,
          [g for g in tree["groups"] if g["path"].startswith("08-编剧方法书籍")])
    flat = sl.library_tree(depth=1)
    check("目录树: depth=1 把子目录并进一级（08-编剧方法书籍 变 3 份）",
          next(g for g in flat["groups"] if g["path"] == "08-编剧方法书籍")["documents"] == 3,
          flat["groups"])
    check("目录树: depth 越界夹到 [1,3]（0 视同未给 -> 2、9 -> 3，不抛错也不返回空）",
          sl.library_tree(depth=0)["depth"] == 2 and sl.library_tree(depth=9)["depth"] == 3,
          (sl.library_tree(depth=0)["depth"], sl.library_tree(depth=9)["depth"]))
    check("目录树: 分组永远是目录（2 段路径不会退化成一个文件一个组）",
          "08-编剧方法书籍" in [g["path"] for g in tree["groups"]]
          and not any(g["path"].endswith((".txt", ".pdf")) for g in tree["groups"]),
          [g["path"] for g in tree["groups"]])

    pending = sl.list_documents(only="pending")
    check("清单: only=pending 是「两库都没整本正文」的那三份（含半成品；已 OCR 的不许算待补）",
          [item["path"] for item in pending["items"]]
          == ["06-期刊/半成品.pdf", "06-期刊/待补.pdf", "08-编剧方法书籍/子目录/深度.pdf"]
          and pending["total"] == 3
          and next(i for i in pending["items"] if i["path"] == "06-期刊/半成品.pdf")["partial"],
          [(item["path"], item["withText"], item["ocr"], item["partial"])
           for item in pending["items"]])
    ocr_only = sl.list_documents(only="ocr")
    check("清单: only=ocr 只回本仓已补的那 1 份，且 withText=false（两个布尔各记各的）",
          ocr_only["total"] == 1 and ocr_only["items"][0]["path"] == "08-编剧方法书籍/扫描件.pdf"
          and ocr_only["items"][0]["withText"] is False, ocr_only["items"])
    check("清单: only=withText 回 3 份（供应商抽出正文的那批）",
          sl.list_documents(only="withText")["total"] == 3,
          sl.list_documents(only="withText")["items"])
    by_dir = sl.list_documents(category="08-编剧方法书籍")
    check("清单: category 是**前缀**匹配（含子目录 ⇒ 3 份，不是只看一层）",
          by_dir["total"] == 3 and by_dir["category"] == "08-编剧方法书籍", by_dir["total"])
    check("清单: category 深到子目录就只剩 1 份",
          [item["path"] for item in sl.list_documents(
              category="08-编剧方法书籍/子目录")["items"]]
          == ["08-编剧方法书籍/子目录/深度.pdf"])
    check("清单: q 按文件名包含匹配（大小写无关）",
          [item["path"] for item in sl.list_documents(q="扫描")["items"]]
          == ["08-编剧方法书籍/扫描件.pdf"])
    check("清单: 排序是 有正文 → 已 OCR → 待补（前端不用二次排）",
          [item["path"] for item in by_dir["items"]]
          == ["08-编剧方法书籍/故事.txt", "08-编剧方法书籍/扫描件.pdf",
              "08-编剧方法书籍/子目录/深度.pdf"],
          [item["path"] for item in by_dir["items"]])
    page = sl.list_documents(category="08-编剧方法书籍", offset=2, limit=1)
    check("清单: 分页只切窗口、total 恒为命中总数（offset 越界回空但 total 照给）",
          (page["total"], page["offset"], page["limit"], len(page["items"])) == (3, 2, 1, 1)
          and sl.list_documents(category="08-编剧方法书籍", offset=99)["items"] == [],
          (page["total"], page["offset"], page["limit"], len(page["items"])))

    # ================= 9. 半成品（status='partial'）不进检索 =================
    #: 真库实测：80 个块里 49 个来自 partial，其中一份 920 页的书只扫了 2 页。
    #: 那些块**已经写进 fts5**（``_save`` 只对 quarantine 不写块）⇒ 不排除的话，
    #: 模型会拿「第 2 页的一句话」当「整本书的写法」引用。
    hidden = sl.search_methods("青苔灯笼", limit=5)
    check("半成品: 默认不进检索（只扫了 1/4 页的书不许以「全书可查」出现）",
          hidden["results"] == [], hidden["results"])
    shown = sl.search_methods("青苔灯笼", limit=5, include_partial=True)
    row = (shown["results"] or [{}])[0]
    check("半成品: include_partial=True 才给，且如实标覆盖率 1/4 页",
          len(shown["results"]) == 1 and row.get("from") == "ocr"
          and row.get("coverage") == {"pages": 4, "chunkPages": 1}, shown["results"])
    check("半成品: hint 明说「别把片段当成整本可用」",
          "没跑完" in (shown.get("hint") or ""), shown.get("hint"))
    opened = sl.open_document("06-期刊/半成品.pdf")
    check("半成品: 翻原文标明 status 与 coverage（不至于把 pages=4/total=1 读成抽取失败）",
          opened["document"]["status"] == "partial"
          and opened["document"]["coverage"] == {"pages": 4, "chunkPages": 1},
          opened["document"])
    check("清单: only=partial 只回那 1 份半成品（与 only=ocr 严格分开）",
          [i["path"] for i in sl.list_documents(only="partial")["items"]]
          == ["06-期刊/半成品.pdf"], sl.list_documents(only="partial")["items"])

    # ================= 10. 状态字面量同源守卫 =================
    #: services 层不许 import scripts 层，所以 ``OCR_STATUS_*`` 是从
    #: ``ocr_index.py`` 的 ``STATUS_*`` **手抄**的字面量。抄字面就必须有守卫：
    #: 那边一改（比如 ``partial`` 改名），这边会静默把半成品当成别的档。
    import ast
    from pathlib import Path as _Path
    src = (_Path(__file__).resolve().parents[1]
           / "app" / "scripts" / "corpus" / "ocr_index.py").read_text("utf-8")
    literals: dict[str, str] = {}
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id.startswith("STATUS_"):
                literals[tgt.id] = node.value.value
    check("常量: services 的 OCR_STATUS_* 与 ocr_index.py 的 STATUS_* 逐字同源",
          literals.get("STATUS_OCR") == sl.OCR_STATUS_DONE
          and literals.get("STATUS_PARTIAL") == sl.OCR_STATUS_PARTIAL
          and literals.get("STATUS_QUARANTINE") == sl.OCR_STATUS_QUARANTINE,
          literals)

    scan_a = sl._scan_documents()
    check("缓存: 一次扫全主库+sidecar，二次调用复用同一份（5403 份不重复扫）",
          sl._scan_documents() is scan_a)
    stat = sidecar.stat()
    os.utime(sidecar, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    check("缓存: 库文件 mtime 变了就必须重扫（OCR 补完新文档，树要给新数）",
          sl._scan_documents() is not scan_a)

    sl.LIBRARY_ROOT, sl.PLAYBOOK_ROOT, sl.METHODS_DB, sl.OCR_DB = original
    sl.reset_cache()
    sl.METHODS_DB = root / "空" / "x.sqlite"
    sl.OCR_DB = root / "空" / "x-ocr.sqlite"
    sl.reset_cache()
    gone = sl.library_tree()
    check("降级: 缺库时树 available=false + hint、totals 为 None（**不编 0**，不抛错）",
          gone["available"] is False and gone["totals"] is None and gone["groups"] == []
          and "未安装" in (gone["hint"] or ""), gone)
    check("降级: 缺库时清单 available=false + total=0，hint 原样透出",
          sl.list_documents()["available"] is False
          and sl.list_documents()["total"] == 0 and sl.list_documents()["hint"] == gone["hint"],
          sl.list_documents())
    sl.LIBRARY_ROOT, sl.PLAYBOOK_ROOT, sl.METHODS_DB, sl.OCR_DB = original
    sl.reset_cache()

    # ================= 第九节：第三库（extra）融进同一条检索 =================
    # 两库变三库时最容易漏的正是这一段：``search`` / ``library_tree`` / ``list_documents``
    # 各自拼库列表，谁漏一条都不报错，只是「新补进来的 .doc 搜不到 / 不进树」。
    from app.scripts.corpus import lib_index as _li
    extra_db = root / "extra" / "library-extra.sqlite"
    extra_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(extra_db.as_posix())
    conn.executescript(_li._SCHEMA)
    conn.execute(
        "INSERT INTO documents(path, name, extension, size, modified_ns, signature,"
        " pages, status) VALUES(?,?,?,?,?,?,?,?)",
        ("05-类型片/补录剧本.doc", "补录剧本.doc", ".doc", 1024, 0, "sig", 1,
         _li.STATUS_EXTRA))
    conn.execute(
        "INSERT INTO chunks(document_id, path, name, page, chunk_no, text)"
        " VALUES(?,?,?,?,?,?)",
        (1, "05-类型片/补录剧本.doc", "补录剧本.doc", 1, 0,
         "三幕结构里第三幕的激励事件要压到最后十分钟。"))
    conn.commit()
    conn.close()
    sl.EXTRA_DB = extra_db
    sl.reset_cache()
    merged = sl.search_methods("三幕结构", limit=5)
    check("第三库: extra 的块搜得到，from 与 parts 都标得出来",
          any(row.get("from") == "extra" for row in merged["results"])
          and merged["parts"].get("extra", {}).get("count", 0) >= 1, merged.get("parts"))
    check("第三库: 目录树把 extra 的份数算进 totals（不另立一套口径）",
          (sl.library_tree()["totals"] or {}).get("documents", 0) >= 8,
          sl.library_tree()["totals"])
    listing = sl.list_documents(only="extra")
    check("第三库: 清单能按 only=extra 筛出来（前端「本仓补录」那一档）",
          any(row.get("path", "").endswith("补录剧本.doc") for row in listing["items"]),
          (len(listing["items"]), listing.get("total")))
    sl.EXTRA_DB = Path(tempfile.mkdtemp(prefix="no-extra-")) / "library-extra.sqlite"
    sl.reset_cache()

    # ---- playbook 规模：README 自述必须与实测一致 -------------------------------
    # 这里是**两本账**的典型现场：README 写「137 份 / 2332 小节」，`/knowledge/report`
    # 又按实际目录数给一套数。谁往包目录里丢一份文件，两处就开始互相打脸，而没有任何
    # 测试会红（2026-11-04 检查时实测两边一致，但没有守卫）。判断记录见 docs/playbook-audit.md。
    import re as _re
    readme_path = sl.PLAYBOOK_ROOT / "README.md"
    check("playbook: README 存在（它是 137 份的元数据页，也在索引白名单里）",
          readme_path.exists(), readme_path)
    if readme_path.exists():
        report = sl._playbook_report()
        stated = _re.search(r"\|\s*规模\s*\|\s*(\d+)\s*份 Markdown\s*/\s*(\d+)\s*小节",
                            readme_path.read_text("utf-8"))
        check("playbook: README「规模」行的份数/小节 == 实测（对不上就是两本账）",
              bool(stated) and (int(stated.group(1)), int(stated.group(2)))
              == (report["files"], report["sections"]),
              (stated.groups() if stated else None, report["files"], report["sections"]))
        check("playbook: 根目录不留散落 md（散一份就多一条「未索引」）",
              report["unindexedFiles"] == 0, report["unindexedFiles"])

    # ---- general/ 入口索引：它是判据面，写错了会直接骗后来人 --------------------
    # 2026-11-06 实测抓到两处：§1 抬头写「十一族」而表里只有 10 行；「动作与打斗」
    # 标 3 份（把归运镜族的「打斗镜头招式组合库」算了进来），使合计 39≠实际 38。
    # 索引正文第 6 行自己写着「往 general/ 加文件必须在本表补一行」—— 这条把它钉住。
    gen_dir = sl.PLAYBOOK_ROOT / "film-expert-bridge" / "general"
    gen_idx = gen_dir / "通用资料索引.md"
    check("playbook: general 入口索引存在（它是那 38 份的唯一入口）",
          gen_idx.exists(), gen_idx)
    if gen_idx.exists():
        _gen_txt = gen_idx.read_text("utf-8")
        _real = [x for x in gen_dir.glob("*.md") if x.name != gen_idx.name]
        _named = {p.replace("\\", "/").split("/")[-1]
                  for p in _re.findall(r"`([^`\n]+?\.md)`", _gen_txt)}
        _unlisted = sorted(x.name for x in _real if x.name not in _named)
        check("playbook: general 每份原料都被入口索引点名（加了文件忘登 = 这份没人认识）",
              not _unlisted, _unlisted)

        _tot, _fam, _fam_counts = 0, 0, []
        for _line in _gen_txt.split("## 2.")[0].split("\n"):
            _s = _line.strip()
            if not _s.startswith("|"):
                continue
            _cols = [c.strip() for c in _s.strip("|").split("|")]
            if len(_cols) < 2 or not _cols[1].isdigit():
                continue
            _fam += 1
            _tot += int(_cols[1])
            _fam_counts.append(int(_cols[1]))
        check("playbook: general 索引 §1 族表份数合计 == 该层实际原料数（份数写错就是骗人）",
              _tot == len(_real), (_tot, len(_real), _fam))

        # 「十族一览」里的**那个「十」**也要对账。上一轮的原话是「§1 抬头写『十一族』而表里
        # 只有 10 行」，靠人眼翻出来的 —— 同一句抬头里还有个「合计 38 份」，两个数字都是散文，
        # 都没有判据。这条把它钉住：抬头数字 == 表行数 == 盘上文件数。
        _cn = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
               "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}
        _hd = _re.search(r"^## 1\. ([一二三四五六七八九十]+?)族一览（合计 (\d+) 份",
                         _gen_txt, _re.M)
        check("playbook: general §1 抬头「N 族／合计 M 份」与表行数、盘上文件数三方一致",
              bool(_hd) and _cn.get(_hd.group(1)) == _fam
              and int(_hd.group(2)) == len(_real),
              (_hd.groups() if _hd else None, _fam, len(_real)))

        # ---- 合计对得上，不等于每族都对得上 -------------------------------------
        # 2026-11-06 自查：上面那条只比**合计**。一族多报 2、另一族少报 2，合计照样相等，
        # 而 §1 的用途正是「照着族名去取那几份」—— 每族错 1 也够把人指到错的文件上。
        #
        # 判据**不按族名 join §1 表**：§1 写「画法与底层方法」而 §7 抬头写「画法方法」，
        # 按名字对就已经差一个字，往后谁再改个名就静默失效（正是 §9.8 那个「点名名单」的教训）。
        # 改成两条不依赖名字的对账：
        #   ① 结构：**抬头自己声明了（N 份）的小节**就是承担归属的那些节（§0/§1/§11/§12 没声明）
        #      ⇒ 每节「声明数」必须等于「该节真正点名的本层文件数」（§7 是两族合写，取 2+2=4）；
        #   ② 多重集：§1 十行数字与各小节声明数字**排完序必须相同** —— 不关心谁对谁，
        #      但任何一个数错了都会露出来（对调两个相等的数看不见，也无害）。
        _BT = "`"
        _heads = [(m.start(), m.group(0)) for m in _re.finditer(r"^## .*$", _gen_txt, _re.M)]
        _real_names = {x.name for x in _real}
        _owner, _dup, _sec_counts = {}, [], []
        for _i, (_pos, _h) in enumerate(_heads):
            _nums = [int(n) for n in _re.findall(r"（(\d+) 份", _h)]
            if not _nums:
                continue          # 导览/反查节，不承担归属
            _end = _heads[_i + 1][0] if _i + 1 < len(_heads) else len(_gen_txt)
            # 只算「本层文件」：`../modules/shot_grammar.md` 这类跨层引用不是本级的份数
            _mine = {n for n in _re.findall(_BT + r"([^" + _BT + r"\n]+?\.md)" + _BT,
                                            _gen_txt[_pos:_end])
                     if "/" not in n and n in _real_names}
            _sec_counts.extend(_nums)
            if sum(_nums) != len(_mine):
                _dup.append(("份数不符", _h[3:24].strip(), sum(_nums), len(_mine)))
            for _n in _mine:
                if _n in _owner:
                    _dup.append(("重复归属", _n, _owner[_n], _h[3:24].strip()))
                _owner[_n] = _h[3:24].strip()
        check("playbook: general 每节「（N 份）」== 该节真正点名的本层文件数（逐族对，不只对合计）",
              not _dup, _dup[:4])
        check("playbook: general §1 十行数字与各小节声明数字（排序后）逐个相同（族份数不许挪）",
              sorted(_fam_counts) == sorted(_sec_counts),
              (sorted(_fam_counts), sorted(_sec_counts)))
        check("playbook: general 38 份每份都归且只归一个族节、且没有谁都不归属的（索引查不到它）",
              set(_owner) == _real_names,
              {"未归属": sorted(_real_names - set(_owner)),
               "不存在的名字": sorted(set(_owner) - _real_names)})

    # ---- 导航型文档：找得到，但不许霸榜 -----------------------------------------
    # 2026-11-06 加 `general/通用资料索引.md` 之后实测：搜「造型」第一条是「4. 角色造型（7 份）」、
    # 搜「四宫格」第一条是「12. 两个有名字没文件的术语」—— 点开只有表、一句方法都读不到，
    # 却和真资料同分并列。**加索引是为了让人找得到，结果先把检索顶掉了**，这正是 §9.7 那个
    # 「索引不许骗人」的另一面：索引也不许抢正文的位置。
    _pb_root = Path(sl.PLAYBOOK_ROOT)
    _nav_missing = [n for n in sl.PLAYBOOK_NAV_DOCS if not list(_pb_root.rglob(f"{n}.md"))]
    check("playbook: 导航型名单里每份都真实存在（改名即静默失效 = 降权悄悄不生效）",
          not _nav_missing, _nav_missing)

    for _q in ("造型", "特效", "运镜"):
        _top = sl.search_playbook(_q, limit=3).get("results") or []
        _head_is_nav = bool(_top) and any(n in Path(_top[0]["path"]).name
                                         for n in sl.PLAYBOOK_NAV_DOCS)
        check(f"playbook: 搜族名「{_q}」第一条是方法正文、不是入口索引（索引不许霸榜）",
              bool(_top) and not _head_is_nav, _top[0].get("path") if _top else None)

    _nav_hits = sl.search_playbook("资料索引", limit=10).get("results") or []
    check("playbook: 明确找目录（查询带「索引」）时不打折，索引仍排得上前 5",
          any(any(n in Path(r["path"]).name for n in sl.PLAYBOOK_NAV_DOCS)
              for r in _nav_hits[:5]), [r["path"] for r in _nav_hits[:5]])

    # ---- 检索给的路径必须真能打开（「检索 → 打开」是一个闭环）-------------------
    # 2026-11-06 实测：10 个查询 × top8 ＝ 75 条回执，`open_document` 全部打得开。
    # 这条盯的是**两处各自都对、拼起来才错**的那类：`search_playbook` 拼 path 的口径
    # （skill/子目录/文件名）与 `open_document` 解析 path 的口径（`_resolve_playbook` 的
    # 白名单子树判定）一旦分叉，检索会给你一个点开就报错的路径 —— 单看任何一边都正常。
    _opened, _broken = 0, []
    for _q in ("造型", "运镜", "打戏", "场景"):
        for _row in sl.search_playbook(_q, limit=5).get("results") or []:
            _opened += 1
            _resp = sl.open_document(_row["path"])
            #: 判「打得开」必须看回执里**真有正文**（`origin=playbook` ＋ `document` 非空）。
            #: 只判 `available is not False` 会被兜底路放行 —— 解析不出相对路径时它回的是
            #: `available=True, origin="curated", total=0, document=None`、
            #: hint 写着「索引里没有这份文件的正文记录」：肉眼一看就是打不开，判据却当成功。
            #: 这正是本轮实践到的**假绿灯**（2026-11-06 改 path 拼法反验时它没红，才发现）。
            if not _resp or _resp.get("origin") != "playbook" or not _resp.get("document"):
                _broken.append((_q, _row["path"], (_resp or {}).get("hint")))
    check("playbook: 检索回执里的 path 都能被 open_document 打开（检索→打开不许断）",
          _opened >= 20 and not _broken, (_opened, _broken[:3]))

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 GBK ⇒ 不重配 stdout 会在第一条 print 就 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
