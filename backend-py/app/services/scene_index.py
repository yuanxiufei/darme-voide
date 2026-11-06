"""场次索引（A 侧派生）—— 库里的 5403 份剧本 → 场次表。

为什么单独一个 sqlite，而不是进 ``darme.db``
==========================================
定过的规矩：**A 侧可重建、B 侧不可重建**。``kb_scenes`` 是纯派生（``DELETE`` 后重跑即可），
放进项目库会：① 被计算进备份体积；② 被 ``db_upgrade.py`` 当成项目结构去补列；③ 用户切换
数据根时莫名其妙跟着跑丢。所以它跟 OCR sidecar 同住：``data/ocr/library-scenes.sqlite``。

对齐键是 ``path``（相对库根）**不是** ``document_id`` —— 供应商库与本仓 sidecar 各自编号，
同一份剧本在两库里 id 不同，只有 path 是两库共同的锚。

正文从哪来
==========
不读 PDF（那是 ``ocr_index`` 的活），只读两份 ``chunks``。拼接时**必须**检查
``chunk_no`` 是否连续：OCR 的门禁会把整页判 ``reject`` 并且**一块都不写**，于是块号跳号 ——
跳号处很可能正好是一场戏的边界，不记录就会把两场粘成一场还不自知。
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.services import script_library
from app.services.scene_parser import SCENES_SCHEMA_VERSION, parse_scenes

__all__ = [
    "SCENES_DB",
    "build_document",
    "load_text",
    "read_scenes",
    "scene_report",
]

#: 派生索引的落点（``SCRIPT_LIBRARY_SCENES_DB`` 可覆盖，测试夹具也用它）
SCENES_DB = Path(os.environ.get("SCRIPT_LIBRARY_SCENES_DB")
                 or (PROJECT_ROOT / "data" / "ocr" / "library-scenes.sqlite"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scene_docs (
    path TEXT PRIMARY KEY,
    origin TEXT,
    sha256 TEXT,
    parser_sha TEXT,
    scenes INTEGER DEFAULT 0,
    strategy TEXT,
    confidence REAL DEFAULT 0,
    chars INTEGER DEFAULT 0,
    source_status TEXT,
    status TEXT,
    hint TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    origin TEXT,
    scene_no INTEGER,
    no_source TEXT,
    int_ext TEXT,
    location TEXT,
    "time" TEXT,
    cast TEXT,
    opening TEXT,
    core_event TEXT,
    scene_tail TEXT,
    page_from INTEGER,
    page_to INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    chars INTEGER,
    confidence REAL,
    extractor TEXT,
    heading TEXT,
    schema_version INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS scenes_by_path ON scenes(path, scene_no);
"""


def _like_literal(value: str) -> str:
    """把字符串转成 LIKE 的**字面量**（配合 ``ESCAPE '\\'``）。

    ``documents.path`` 存的是历史机位路径（``G:\\电影剧本\\...``），本仓只能按"完整路径的
    尾巴"反查 —— 但 ``LIKE`` 里 ``_`` 是单字符通配、``%`` 是任意串，文件名一旦含下划线
    （``EP001_终稿.pdf`` 很常见）就会**匹配到另一份**，还不是报错，是静默给错文本。
    所以三个字符全部转义。反斜杠必须最先处理，否则会把后面加的转义符再转义一遍。
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def _parser_sha() -> str:
    """当前解析器源码的指纹（12 位 sha1）。

    ``sha256`` 只说明**正文**没变，不说明**解析器**没变 —— 解析器一改（新增路线、改正则），
    旧的派生行就全过期了，而按 sha256 判会一律跳过、把过期结果当成"已是最新"。
    拿源码字节做指纹，改一个字就自动失效，不需要谁记得手动 ``--force``。
    """
    from app.services import scene_parser  # 局部 import：避免循环依赖
    return hashlib.sha1(Path(scene_parser.__file__).read_bytes()).hexdigest()[:12]


#: 本进程的解析器指纹（模块级算一次；改解析器源码 ⇒ 批处理自动重算该份）
PARSER_SHA = _parser_sha()


def _writer() -> sqlite3.Connection:
    """写连接：派生库是本仓自己的资产，可以建、可以删、**不进**供应商目录。"""
    SCENES_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SCENES_DB.as_posix(), timeout=30)
    conn.executescript(_SCHEMA)
    #: 老库补列：``_SCHEMA`` 的 CREATE 只对新库生效，已有库得 ALTER 一次（幂等）。
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(scene_docs)")}
        if "parser_sha" not in columns:
            conn.execute("ALTER TABLE scene_docs ADD COLUMN parser_sha TEXT")
    except sqlite3.Error:  # noqa: BLE001 —— 补列失败不该挡住写入（大不了下次还重算）
        pass
    return conn


def load_text(rel: str) -> dict[str, Any]:
    """相对路径 → 全文 + 页码区间 + 来源状态。

    两库都试：供应商库优先（抽好的文字），0 块时落到 OCR sidecar。与
    ``script_library.open_document`` 的取舍**逐字一致**，否则同一份文件在"看原文"和
    "切场次"两个入口会出现两份不一样的正文字。
    """
    out: dict[str, Any] = {
        "available": False, "text": "", "origin": None, "source_status": None,
        "pages_total": 0, "pages": [], "gaps": [], "hint": "",
    }
    opened = False
    for label, db in (("curated", script_library.METHODS_DB), ("ocr", script_library.OCR_DB)):
        conn = script_library._connect(db)  # noqa: SLF001 —— 同包内复用只读连接
        if conn is None:
            continue
        opened = True
        try:
            row = conn.execute(
                "SELECT id, pages, status FROM documents WHERE path LIKE ? ESCAPE '\\' "
                "ORDER BY id LIMIT 1",
                [f"%{_like_literal(rel.replace('/', chr(92)))}"],
            ).fetchone()
            if row is None:
                continue
            doc_id, pages_total, source_status = row
            #: ⚠️ 排序必须是 ``page, chunk_no`` —— 供应商库里 ``chunk_no`` 是**页内块号**
            #: （每页都从 1 开始，实测 349 页对应 chunk_no 全是 1）。只按 ``chunk_no`` 排
            #: 会把"所有页的第 1 块"排在一起，正文彻底打乱，而且不报错 —— 是最坏的那种错。
            rows = conn.execute(
                'SELECT page, chunk_no, text FROM chunks WHERE document_id = ? '
                "ORDER BY page, chunk_no",
                [doc_id],
            ).fetchall()
        except sqlite3.Error as exc:  # noqa: BLE001 —— 读坏了要说出来，不能静默当空
            out["hint"] = f"{label} 索引读取失败：{exc}"
            continue
        finally:
            conn.close()

        if not rows:
            continue

        parts: list[str] = []
        spans: list[tuple[int, int, int]] = []
        cursor = 0
        previous_no: int | None = None
        previous_page = -1
        gaps: list[int] = []
        tail = ""  # 只留上文的末尾 200 字做重叠检测，别每块都 join 全文（那是 O(n²)）
        for page, chunk_no, text in rows:
            body = (text or "").replace("\r\n", "\n")
            #: 相邻块理论上无重叠（``ocr_index._page_chunks`` 是硬切），但供应商库没这个保证。
            #: 发现长重叠就去掉，否则重复的字会把场次头复制一份、切出幽灵场次。
            if tail and len(body) > 40:
                for size in range(min(200, len(body), len(tail)) // 10 * 10, 39, -10):
                    if tail.endswith(body[:size]):
                        body = body[size:]
                        break
            #: 只在**同一页内**判连续性：跨页时 ``chunk_no`` 本来就从 1 重来，
            #: 跨页比会得出"每页都缺块"的假告警（实测 1653 页 → 1650 条假 gap）。
            if (previous_no is not None and int(page or 0) == previous_page
                    and chunk_no != previous_no + 1):
                gaps.append(int(chunk_no))
            previous_no, previous_page = int(chunk_no), int(page or 0)
            start = cursor
            parts.append(body)
            cursor += len(body)
            tail = (tail + body)[-200:]
            spans.append((start, cursor, int(page or 0)))

        out.update({
            "available": True,
            "text": "".join(parts),
            "origin": label,
            "source_status": source_status,
            "pages_total": int(pages_total or 0),
            "pages": spans,
            "gaps": gaps,
            "hint": "",
        })
        return out

    if not out["hint"]:
        out["hint"] = ("索引里查不到这份文件的正文记录" if opened
                       else "两份索引都不在（方法库与 OCR sidecar 都没找到）。")
    return out


def _page_at(spans: list[tuple[int, int, int]], char_pos: int) -> int | None:
    """字符偏移 → 页号（块区间包含即命中；越界取最近一端的块）。"""
    if not spans:
        return None
    for start, end, page in spans:
        if start <= char_pos < end:
            return page
    return spans[-1][2] if char_pos >= spans[-1][1] else spans[0][2]


def build_document(rel: str, *, force: bool = False) -> dict[str, Any]:
    """解析一份 → 写派生库。

    正文 sha256 **与解析器指纹**都没变且非 ``force`` 时才跳过，可反复调用（可断点续跑）。
    """
    loaded = load_text(rel)
    if not loaded["available"]:
        return {"path": rel, "status": "missing", "scenes": 0, "skipped": False,
                "notes": [loaded["hint"]]}

    text = loaded["text"]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if not force and SCENES_DB.exists():
        conn = sqlite3.connect(SCENES_DB.as_posix(), timeout=30)
        try:
            row = conn.execute(
                "SELECT sha256, parser_sha FROM scene_docs WHERE path = ?", [rel]
            ).fetchone()
        except sqlite3.Error:
            row = None
        finally:
            conn.close()
        #: 正文与解析器**都**没变才跳过：只比 sha256 会把"解析器改了但正文没改"当成最新。
        if row and row[0] == digest and row[1] == PARSER_SHA:
            return {"path": rel, "status": "skipped", "scenes": 0, "skipped": True,
                    "notes": ["正文与解析器都没变，跳过（加 --force 可强制重算）。"]}
        if row and row[0] == digest:
            stale_note = "正文未变但解析器换过 ⇒ 重新解析。"
        else:
            stale_note = ""
    else:
        stale_note = ""

    parsed = parse_scenes(text)
    scenes = parsed["scenes"]
    notes = list(parsed["notes"])
    if stale_note:
        notes.append(stale_note)
    if not text.strip():
        status = "empty"
    elif parsed["strategy"] == "none":
        status = "no-heading"
    else:
        status = "ok"
    if loaded["gaps"]:
        notes.append(f"块号不连续（跳过 {len(loaded['gaps'])} 处），缺口处的场次边界可能缺失。")

    rows = []
    for item in scenes:
        rows.append((
            rel, loaded["origin"], item["scene_no"], item["no_source"], item["int_ext"],
            item["location"], item["time"],
            ", ".join(item["cast"]) if item["cast"] else None,
            item["opening"], item["core_event"], item["scene_tail"],
            _page_at(loaded["pages"], item["char_start"]),
            _page_at(loaded["pages"], max(item["char_start"], item["char_end"] - 1)),
            item["char_start"], item["char_end"], item["chars"],
            item["confidence"], item["extractor"], item["heading"],
            SCENES_SCHEMA_VERSION,
        ))

    conn = _writer()
    try:
        with conn:  # 单事务：中途失败不留半份
            conn.execute("DELETE FROM scenes WHERE path = ?", [rel])
            conn.executemany(
                'INSERT INTO scenes (path, origin, scene_no, no_source, int_ext, location, '
                '"time", "cast", opening, core_event, scene_tail, page_from, page_to, char_start, '
                'char_end, chars, confidence, extractor, heading, schema_version) '
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.execute(
                "INSERT OR REPLACE INTO scene_docs (path, origin, sha256, parser_sha, scenes, "
                "strategy, confidence, chars, source_status, status, hint, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (rel, loaded["origin"], digest, PARSER_SHA, len(rows), parsed["strategy"],
                 parsed["confidence"], len(text), loaded["source_status"], status,
                 "\n".join(notes)[:2000] or None, _now()),
            )
    finally:
        conn.close()

    return {"path": rel, "status": status, "scenes": len(rows),
            "strategy": parsed["strategy"], "confidence": parsed["confidence"],
            "origin": loaded["origin"], "sourceStatus": loaded["source_status"],
            "skipped": False, "notes": notes}


def read_scenes(rel: str) -> dict[str, Any]:
    """读回一份的场次表（不解析、不写入，纯只读 —— HTTP 端点只能走这条路）。"""
    if not SCENES_DB.exists():
        return {"available": False, "scenes": [], "document": None,
                "hint": "场次索引还没建（先跑 ``python -m app.scripts.corpus.scene_index``）。"}
    conn = sqlite3.connect(f"file:{SCENES_DB.as_posix()}?mode=ro", uri=True, timeout=20)
    try:
        doc = conn.execute(
            "SELECT path, origin, scenes, strategy, confidence, chars, source_status, status, "
            "hint, updated_at FROM scene_docs WHERE path = ?", [rel],
        ).fetchone()
        if doc is None:
            return {"available": False, "scenes": [], "document": None,
                    "hint": "这份还没解析过场次（属按需生成，不是漏做）。"}
        rows = conn.execute(
            'SELECT scene_no, no_source, int_ext, location, "time", "cast", opening, core_event, '
            "scene_tail, page_from, page_to, chars, confidence, heading "
            #: 排序必须是**原文位置**，不是 ``scene_no``：短剧的「集-场号」（``1-1``/``2-1``）
            #: 场号每集重置，按场号排会把第 2 集第 1 场插到第 1 集第 2 场前面（混排不报错）。
            #: ``char_start`` 对原文单调递增，补的生产编号 ``scene_no`` 也随它递增 ⇒ 两种都成立。
            "FROM scenes WHERE path = ? ORDER BY char_start, id", [rel],
        ).fetchall()
    finally:
        conn.close()

    keys = ("sceneNo", "noSource", "intExt", "location", "time", "cast", "opening",
            "coreEvent", "sceneTail", "pageFrom", "pageTo", "chars", "confidence", "heading")
    scenes = []
    for row in rows:
        item = dict(zip(keys, row))
        item["cast"] = [name.strip() for name in (item["cast"] or "").split(",") if name.strip()]
        scenes.append(item)

    return {
        "available": True,
        "document": {
            "path": doc[0], "origin": doc[1], "scenes": doc[2], "strategy": doc[3],
            "confidence": doc[4], "chars": doc[5], "sourceStatus": doc[6],
            "status": doc[7], "hint": doc[8], "updatedAt": doc[9],
        },
        "scenes": scenes,
        "hint": None,
    }


def scene_report() -> dict[str, Any]:
    """盘点：多少份解析过、多少份零场次。用于确认"跑没跑完"，不做业务判断。"""
    if not SCENES_DB.exists():
        return {"available": False, "documents": 0, "scenes": 0,
                "hint": "场次索引还没建。"}
    conn = sqlite3.connect(f"file:{SCENES_DB.as_posix()}?mode=ro", uri=True, timeout=20)
    try:
        documents = conn.execute("SELECT count(*) FROM scene_docs").fetchone()[0]
        scenes = conn.execute("SELECT count(*) FROM scenes").fetchone()[0]
        by_status = dict(conn.execute(
            "SELECT status, count(*) FROM scene_docs GROUP BY status").fetchall())
        by_strategy = dict(conn.execute(
            "SELECT strategy, count(*) FROM scene_docs GROUP BY strategy").fetchall())
        try:
            stale = conn.execute(
                "SELECT count(*) FROM scene_docs WHERE coalesce(parser_sha, '') <> ?",
                [PARSER_SHA],
            ).fetchone()[0]
        except sqlite3.Error:  # noqa: BLE001 —— 老库还没补列 ⇒ 全部算过期
            stale = documents
    finally:
        conn.close()
    return {"available": True, "documents": documents, "scenes": scenes,
            "byStatus": by_status, "byStrategy": by_strategy,
            "staleDocuments": stale, "parserSha": PARSER_SHA}
