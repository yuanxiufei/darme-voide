"""``library-overlay.sqlite`` —— 知识库的**可写覆盖层**（编辑 / 删除 / 上传台账）。

## 为什么不是「就地改那三个库」

三库都是**别的进程产出的派生物**，谁都不归这一层写：

| 库 | 谁产 | 什么时候被整体覆盖 |
|---|---|---|
| ``curated`` | 供应商脚本 | 供应商重跑一次就回来 |
| ``ocr`` | ``ocr_index.py`` | ``--redo`` 重烧那一本就回来 |
| ``extra`` | ``lib_index.py`` | ``--all`` 重算就回来 |

就地改的后果不是「改丢了」这么简单，而是**改完之后整库自相矛盾**：盘上原文没变、
索引里那段变了，``onDisk`` 还报 ``true`` —— 用户以为改动写进了书里，其实只写在
一个下次会被覆盖的缓存里。所以写一律落在**独立的第四库**：

* **编辑** ⇒ ``edits`` 表，按 ``(source, path, chunk_no)`` 定位一块正文，读取时叠加；
* **删除** ⇒ ``tombstones`` 表，读取时隐藏，可还原（本仓自己上传的件才允许物理删）；
* **上传** ⇒ ``uploads`` 台账 ＋ 落盘 ＋ 走 ``lib_index`` 同一条抽文本路径入 ``extra``。

好处是：原库怎么重建都不影响这些改动；反过来，覆盖层里指向的行要是没了（比如重跑
供应商库把某份删了），读取时会**如实报 ``stale``**，而不是假装还有这一段。

## 三个必须能区分的状态

1. ``buried``（已删，还在库里）≠ ``missing``（原库没有了）—— 前者能还原，后者还原也是空的；
2. ``edited``（这块被改过）≠ 原件 —— 回执里每块带 ``edited`` 标记，且给 ``original`` 原文；
3. ``uploaded``（本仓上传的）≠ ``extra`` 里原有的 —— 只有前者能物理删。

## 并发

与 ``ocr_index`` 同一套纪律：``timeout=`` 只管第一条语句，整条连接要靠
``PRAGMA busy_timeout``；写走 ``_save_atomic`` 的退让重试形态（锁等 4 次，非锁错误立刻抛）。
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any

#: 覆盖层库的位置。**必须可换**（测试要隔离，绝不能让单测写进开发机的真库）。
_DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "ocr" / "library-overlay.sqlite"
OVERLAY_DB = Path(os.environ.get("SCRIPT_LIBRARY_OVERLAY_DB") or _DEFAULT_DB)

#: 文档头那一格（改标题/备注）用 ``-1``。正文块从 0 起（``lib_index._chunks`` 的 ``chunk_no``），
#: 两边不会撞。
META_CHUNK = -1

#: 三个可写来源。``playbook`` 是 md 文件、直接改文件即真源，不走覆盖层。
WRITABLE_SOURCES = ("curated", "ocr", "extra")

#: 锁退让：第 n 次等 n × 5 秒。与 ``ocr_index.LOCK_RETRIES`` 同口径。
LOCK_RETRIES = 4
BUSY_TIMEOUT_MS = 60000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS edits (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    path TEXT NOT NULL,
    chunk_no INTEGER NOT NULL,
    text TEXT NOT NULL,
    note TEXT,
    created_ns INTEGER NOT NULL,
    updated_ns INTEGER NOT NULL,
    UNIQUE(source, path, chunk_no)
);
CREATE TABLE IF NOT EXISTS tombstones (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    path TEXT NOT NULL,
    reason TEXT,
    created_ns INTEGER NOT NULL,
    UNIQUE(source, path)
);
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    name TEXT,
    size INTEGER,
    mime TEXT,
    sha256 TEXT,
    status TEXT,
    reason TEXT,
    documents INTEGER,
    chunks INTEGER,
    created_ns INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS edits_fts USING fts5(
    path UNINDEXED, name UNINDEXED, page UNINDEXED, chunk_no UNINDEXED,
    source UNINDEXED, text, tokenize='trigram'
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


class OverlayError(ValueError):
    """覆盖层拒绝了这次写（调用方应当把这句原话回给用户，不要吞成 500）。"""


# ===========================================================================
# 连接
# ===========================================================================


def _connect(*, writable: bool = False) -> sqlite3.Connection | None:
    """只读连接在库不存在时返回 ``None``（= 「这台机器还没有任何改动」）。"""
    path = Path(OVERLAY_DB)
    if not writable and not path.exists():
        return None
    if writable:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path.as_posix(), timeout=BUSY_TIMEOUT_MS / 1000)
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        conn.executescript(_SCHEMA)
        return conn
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True,
                           timeout=BUSY_TIMEOUT_MS / 1000)
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    return conn


def _write(mutate) -> Any:
    """在写锁上退让重试；非锁的 ``OperationalError`` 立刻抛（别把真 bug 埋进等待里）。"""
    last: Exception | None = None
    for attempt in range(LOCK_RETRIES):
        conn = _connect(writable=True)
        try:
            with conn:
                return mutate(conn)
        except sqlite3.OperationalError as exc:  # noqa: PERF203
            last = exc
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            if attempt + 1 < LOCK_RETRIES:
                time.sleep((attempt + 1) * 5)
        finally:
            conn.close()
    raise OverlayError(f"覆盖层写入失败（数据库被占用）：{last}")


def stamp() -> tuple:
    """给 ``script_library`` 的进程内缓存当戳 —— 改了覆盖层必须让扫描结果失效。"""
    path = Path(OVERLAY_DB)
    try:
        stat = path.stat()
    except OSError:
        return ("missing",)
    return (path.as_posix(), stat.st_size, stat.st_mtime_ns)


def available() -> bool:
    return Path(OVERLAY_DB).exists()


# ===========================================================================
# 编辑
# ===========================================================================


def _clean(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def save_edit(source: str, path: str, chunk_no: int, text: str, *, note: str | None = None,
              name: str = "", page: int = 0) -> dict[str, Any]:
    """写一块覆盖正文。同一 ``(source, path, chunk_no)`` 重复保存是**更新**，不是插一条新的。

    ``text`` 允许是空串（= 「这块清空」），但不允许 ``None``：空串有语义（这一页确实是空的），
    ``None`` 只会是调用方忘了带。
    """
    source = (source or "").strip()
    if source not in WRITABLE_SOURCES:
        raise OverlayError(f"来源「{source}」不接受编辑（只有 {', '.join(WRITABLE_SOURCES)} 可写）")
    path = (path or "").strip().replace("\\", "/")
    if not path:
        raise OverlayError("缺少 path")
    if text is None:
        raise OverlayError("缺少 text（要清空请传空串）")
    if chunk_no is None:
        raise OverlayError("缺少 chunkNo（改文档头请传 -1）")
    body = _clean(text)
    now = time.time_ns()

    def mutate(conn: sqlite3.Connection) -> dict[str, Any]:
        conn.execute(
            "INSERT INTO edits (source, path, chunk_no, text, note, created_ns, updated_ns)"
            " VALUES (?,?,?,?,?,?,?)"
            " ON CONFLICT(source, path, chunk_no) DO UPDATE SET"
            " text=excluded.text, note=excluded.note, updated_ns=excluded.updated_ns",
            (source, path, int(chunk_no), body, note, now, now),
        )
        row_id = conn.execute(
            "SELECT id FROM edits WHERE source=? AND path=? AND chunk_no=?",
            (source, path, int(chunk_no))).fetchone()[0]
        conn.execute("DELETE FROM edits_fts WHERE path=? AND source=? AND chunk_no=?",
                     (path, source, str(int(chunk_no))))
        conn.execute(
            "INSERT INTO edits_fts (path, name, page, chunk_no, source, text)"
            " VALUES (?,?,?,?,?,?)",
            (path, name or path.rsplit("/", 1)[-1], str(int(page or 0)),
             str(int(chunk_no)), source, body),
        )
        return {"id": row_id, "source": source, "path": path,
                "chunkNo": int(chunk_no), "chars": len(body), "updatedNs": now}

    return _write(mutate)


def drop_edit(source: str, path: str, chunk_no: int) -> dict[str, Any]:
    """撤销一次编辑（回到原库那份正文）。没写过也返回 ``dropped=false``，不当错误。"""
    source = (source or "").strip().replace("\\", "/")
    path = (path or "").strip().replace("\\", "/")

    def mutate(conn: sqlite3.Connection) -> dict[str, Any]:
        cur = conn.execute("DELETE FROM edits WHERE source=? AND path=? AND chunk_no=?",
                           (source, path, int(chunk_no)))
        conn.execute("DELETE FROM edits_fts WHERE path=? AND source=? AND chunk_no=?",
                     (path, source, str(int(chunk_no))))
        return {"dropped": cur.rowcount > 0, "source": source, "path": path,
                "chunkNo": int(chunk_no)}

    return _write(mutate)


def edits_of(source: str, path: str) -> dict[int, dict[str, Any]]:
    """某一份文档的所有覆盖块：``{chunk_no: {...}}``。读不到一律空字典（= 没改过）。"""
    conn = _connect()
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT chunk_no, text, note, updated_ns FROM edits WHERE source=? AND path=?",
            ((source or "").strip(), (path or "").strip().replace("\\", "/"))).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return {int(no): {"text": text or "", "note": note, "updatedNs": ns}
            for no, text, note, ns in rows}


def list_edits(*, limit: int = 200, path_prefix: str = "") -> list[dict[str, Any]]:
    """已改清单（回收站/审计用）。按最近改动倒序。"""
    conn = _connect()
    if conn is None:
        return []
    prefix = (path_prefix or "").strip().replace("\\", "/")
    sql = ("SELECT source, path, chunk_no, length(text), note, updated_ns FROM edits")
    params: list[Any] = []
    if prefix:
        sql += " WHERE instr(path, ?) > 0"
        params.append(prefix)
    sql += " ORDER BY updated_ns DESC LIMIT ?"
    params.append(max(1, min(int(limit or 200), 500)))
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [{"source": source, "path": path, "chunkNo": int(no), "chars": int(chars or 0),
             "note": note, "updatedNs": int(ns)} for source, path, no, chars, note, ns in rows]


def edited_paths() -> dict[str, set[str]]:
    """``{source: {path, …}}`` —— 扫目录时标「这块被改过」，同样一次取走。"""
    conn = _connect()
    if conn is None:
        return {}
    try:
        rows = conn.execute("SELECT DISTINCT source, path FROM edits").fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    out: dict[str, set[str]] = {}
    for source, path in rows:
        out.setdefault(str(source), set()).add(str(path))
    return out


def search_edits(terms: list[str], *, limit: int = 5) -> dict[str, Any]:
    """在**改过的正文**里检索 —— 让「改完就搜得到」。

    ≥3 字走 fts5（trigram，与三个原库同一分词口径），短词回落 ``LIKE``：短词在 trigram
    索引上会静默 0 命中，那种「改了却搜不到」用户只会当成没保存。
    """
    cleaned = [term for term in (terms or []) if term]
    if not cleaned:
        return {"available": True, "results": [], "matched": 0, "ranking": None}
    conn = _connect()
    if conn is None:
        return {"available": True, "results": [], "matched": 0, "ranking": None}
    use_fts = all(len(term) >= 3 for term in cleaned)
    try:
        if use_fts:
            match = " OR ".join('"' + term.replace('"', '""') + '"' for term in cleaned)
            rows = conn.execute(
                "SELECT path, name, page, chunk_no, source, "
                "snippet(edits_fts, 5, '【', '】', '…', 40) AS excerpt, "
                "bm25(edits_fts) AS rank FROM edits_fts WHERE edits_fts MATCH ? "
                "ORDER BY rank LIMIT ?", (match, limit)).fetchall()
            matched = conn.execute("SELECT count(*) FROM edits_fts WHERE edits_fts MATCH ?",
                                   (match,)).fetchone()[0]
            ranking = "bm25"
        else:
            cond = " AND ".join(["text LIKE ?"] * len(cleaned))
            rows = conn.execute(
                f"SELECT path, name, page, chunk_no, source, text, 0 FROM edits_fts "
                f"WHERE {cond} LIMIT ?",
                [f"%{term}%" for term in cleaned] + [limit]).fetchall()
            matched = len(rows)
            ranking = "like"
    except sqlite3.Error:
        return {"available": True, "results": [], "matched": 0, "ranking": None,
                "hint": "覆盖层索引读取失败。"}
    finally:
        conn.close()
    results = []
    for path, name, page, chunk_no, source, excerpt, rank in rows:
        results.append({
            #: ``source`` 是 RRF 融合时的**分源键**：写成 ``methods`` 会让改过的块与
            #: 原库结果共用一个秩计数器（编辑条永远排在后半段），所以单列一源。
            "source": "edits",
            "title": name or "",
            "path": path,
            "page": int(page or 0),
            "chunkNo": int(chunk_no or 0),
            "score": round(-float(rank), 4) if use_fts else float(rank or 0),
            "excerpt": " ".join(str(excerpt or "").split())[:400],
            "onDisk": True,
            "edited": True,
            "editSource": source,
        })
    return {"available": True, "results": results, "matched": int(matched), "ranking": ranking}


# ===========================================================================
# 删除（墓碑）
# ===========================================================================


def bury(source: str, path: str, *, reason: str | None = None) -> dict[str, Any]:
    """隐藏一份（不动物理文件、不动原库）。重复删是幂等的。"""
    source = (source or "").strip()
    if source not in WRITABLE_SOURCES:
        raise OverlayError(f"来源「{source}」不接受删除")
    path = (path or "").strip().replace("\\", "/")
    if not path:
        raise OverlayError("缺少 path")
    now = time.time_ns()

    def mutate(conn: sqlite3.Connection) -> dict[str, Any]:
        conn.execute(
            "INSERT INTO tombstones (source, path, reason, created_ns) VALUES (?,?,?,?)"
            " ON CONFLICT(source, path) DO UPDATE SET reason=excluded.reason",
            (source, path, reason, now))
        return {"buried": True, "source": source, "path": path, "reason": reason}

    return _write(mutate)


def raise_tomb(source: str, path: str) -> dict[str, Any]:
    """把一份从回收站拿回来。"""
    source = (source or "").strip()
    path = (path or "").strip().replace("\\", "/")

    def mutate(conn: sqlite3.Connection) -> dict[str, Any]:
        cur = conn.execute("DELETE FROM tombstones WHERE source=? AND path=?", (source, path))
        return {"restored": cur.rowcount > 0, "source": source, "path": path}

    return _write(mutate)


def buried_map() -> dict[str, set[str]]:
    """``{source: {path, …}}`` —— 扫描目录树时一次性取走，别在循环里逐条查库。"""
    conn = _connect()
    if conn is None:
        return {}
    try:
        rows = conn.execute("SELECT source, path FROM tombstones").fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    out: dict[str, set[str]] = {}
    for source, path in rows:
        out.setdefault(str(source), set()).add(str(path))
    return out


def list_buried(*, limit: int = 200) -> list[dict[str, Any]]:
    conn = _connect()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT source, path, reason, created_ns FROM tombstones "
            "ORDER BY created_ns DESC LIMIT ?", (max(1, min(int(limit or 200), 500)),)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [{"source": source, "path": path, "reason": reason, "buriedNs": int(ns)}
            for source, path, reason, ns in rows]


def is_buried(source: str, path: str) -> bool:
    conn = _connect()
    if conn is None:
        return False
    try:
        row = conn.execute("SELECT 1 FROM tombstones WHERE source=? AND path=?",
                           ((source or "").strip(), (path or "").strip().replace("\\", "/"))
                           ).fetchone()
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    return row is not None


# ===========================================================================
# 上传台账
# ===========================================================================


def record_upload(*, path: str, name: str = "", size: int = 0, mime: str = "",
                  sha256: str = "", status: str = "", reason: str | None = None,
                  documents: int = 0, chunks: int = 0) -> dict[str, Any]:
    """记一条上传结果。``status`` ∈ ``ingested`` / ``quarantined`` / ``rejected``。

    三档必须分开：``quarantined``（落盘了但抽不出可用正文）与 ``rejected``（连落盘都没做）
    对用户是两件事 —— 前者能在盘上找到，后者根本没有。合成一句「失败」就没法处理。
    """
    now = time.time_ns()

    def mutate(conn: sqlite3.Connection) -> dict[str, Any]:
        cur = conn.execute(
            "INSERT INTO uploads (path, name, size, mime, sha256, status, reason,"
            " documents, chunks, created_ns) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (path, name, int(size or 0), mime, sha256, status, reason,
             int(documents or 0), int(chunks or 0), now))
        return {"id": cur.lastrowid, "path": path, "status": status, "createdNs": now}

    return _write(mutate)


def list_uploads(*, limit: int = 20) -> list[dict[str, Any]]:
    conn = _connect()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT path, name, size, mime, status, reason, documents, chunks, created_ns "
            "FROM uploads ORDER BY created_ns DESC LIMIT ?",
            (max(1, min(int(limit or 20), 200)),)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [{"path": path, "name": name, "size": int(size or 0), "mime": mime,
             "status": status, "reason": reason, "documents": int(docs or 0),
             "chunks": int(chunks or 0), "createdNs": int(ns)}
            for path, name, size, mime, status, reason, docs, chunks, ns in rows]


def upload_of(path: str) -> dict[str, Any] | None:
    """本仓**自己上传**的那份（决定能不能物理删）。"""
    conn = _connect()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT path, name, size, status FROM uploads WHERE path=? ORDER BY id DESC LIMIT 1",
            ((path or "").strip().replace("\\", "/"),)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    return {"path": row[0], "name": row[1], "size": int(row[2] or 0), "status": row[3]}


# ===========================================================================
# 盘点
# ===========================================================================


def report() -> dict[str, Any]:
    """覆盖层自身的样子（前端「回收站」与运维看这一个口子）。"""
    path = Path(OVERLAY_DB)
    if not path.exists():
        return {"available": False, "db": path.as_posix(),
                "hint": "还没有任何编辑/删除/上传记录（这一层是写操作时才建的）。",
                "edits": 0, "tombstones": 0, "uploads": 0}
    conn = _connect()
    if conn is None:
        return {"available": False, "db": path.as_posix(), "hint": "覆盖层打不开。",
                "edits": 0, "tombstones": 0, "uploads": 0}
    try:
        data = {
            "available": True,
            "db": path.as_posix(),
            "edits": conn.execute("SELECT count(*) FROM edits").fetchone()[0],
            "tombstones": conn.execute("SELECT count(*) FROM tombstones").fetchone()[0],
            "uploads": conn.execute("SELECT count(*) FROM uploads").fetchone()[0],
            "edited": list_edits(limit=50),
            "buried": list_buried(limit=50),
            "recentUploads": list_uploads(limit=20),
        }
    except sqlite3.Error as exc:  # noqa: BLE001
        data = {"available": False, "db": path.as_posix(), "hint": f"覆盖层读不了：{exc}",
                "edits": 0, "tombstones": 0, "uploads": 0}
    finally:
        conn.close()
    return data
