"""知识台账（B 侧，本仓可写）—— 「一条经验」的唯一归口。

它不是什么
==========
不是 ``script_library``（那 5403 份是**外部**剧本/方法资料，只读，且是供应商预建的库），
也不是 ``library_overlay``（那是**外部资料**的编辑覆盖层）。本库存的是**本仓自己跑出来的**
经验：哪一步白做了、哪个参数配这个题材更好、哪条失败是怎么修的。

四类知识共用一张表
==================
``kind`` 分七类：``fact``（行业/作品事实）、``work``（作品资料，外部知识）、``recipe``（成法）、
``lesson``（经验）、``failure``（失败样本：现象→根因→修法）、``threshold``（阈值建议）、
``prompt_patch``（提示词补丁）。**同一张表、同一套检索**：这样「搜悬疑短剧的节奏」和
「搜这部剧第 3 集为什么崩」能一次命中，而不是让调用方先猜该查哪个库。

三条硬规矩
==========
1. **证据必须可核**：``evidence`` 里每个 id 要么是 ``outcome_index`` 里真实存在的 run，
   要么是盘上真实存在的文件。核不过 ⇒ **整条拒收**（不是"存下来再标记可疑"）——
   这比在提示词里写「不要编造」有效得多。
2. **同一件事只一条**：``id = kind-scope-归一标题`` 的 hash。二次提炼只 ``support+1``，
   不新建。否则"经验"会以每天十几条的速度膨胀成噪声。
3. **状态机单向前进**：``proposed → accepted → deprecated/rejected``。Agent 写进来的
   永远是 ``proposed`` —— 它可以提议，不能自己拍板。

检索口径（与 ``script_library`` 同源）
====================================
``trigram`` 只索引 **3 字符以上**：2 字中文（「悬疑」「糊」）在 trigram 下**一条都不命中且不报错**
⇒ 必须回落 ``LIKE``，并在回执里如实写 ``ranking``。三种排序**不可比**，混在一起展示就是
错误结论（``bm25`` 越小越相关、``like`` 只是包含、``recent`` 与相关性无关）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.core.config import PROJECT_ROOT, get_data_root
from app.services import outcome_index

__all__ = [
    "KINDS",
    "STATUSES",
    "db_path",
    "upsert",
    "search",
    "list_entries",
    "detail",
    "set_status",
    "set_lift",
    "record_usage",
    "usage_summary",
    "stats",
    "entry_id",
]

#: 知识类型（新增一类要同时想清楚：谁会读它、凭什么判它有用）
KINDS = ("fact", "work", "recipe", "lesson", "failure", "threshold", "prompt_patch")

#: 生命周期状态
STATUSES = ("proposed", "accepted", "deprecated", "rejected")

#: 外部知识（``fact``/``work``）的证据是来源，不是 run ⇒ 允许空 evidence
_EXTERNAL_KINDS = ("fact", "work")

#: 状态机（同状态视为幂等成功；不在表里的迁移一律拒）
_TRANSITIONS: dict[str, set[str]] = {
    "proposed": {"accepted", "rejected"},
    "accepted": {"deprecated", "rejected"},
    "deprecated": {"accepted", "rejected"},
    "rejected": {"proposed"},
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    tags TEXT DEFAULT '',
    confidence REAL DEFAULT 0,
    status TEXT DEFAULT 'proposed',
    evidence TEXT DEFAULT '',
    support INTEGER DEFAULT 0,
    contra INTEGER DEFAULT 0,
    lift REAL,
    source_kind TEXT,
    source_ref TEXT,
    created_ns INTEGER NOT NULL,
    updated_ns INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_by_kind ON entries(kind, status, updated_ns);
CREATE INDEX IF NOT EXISTS entries_by_scope ON entries(scope, status);
CREATE TABLE IF NOT EXISTS entry_links (
    entry_id TEXT NOT NULL,
    ref_kind TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    UNIQUE(entry_id, ref_kind, ref_id)
);
CREATE TABLE IF NOT EXISTS entry_usage (
    id INTEGER PRIMARY KEY,
    entry_id TEXT NOT NULL,
    run_id TEXT,
    used_ns INTEGER NOT NULL,
    outcome TEXT
);
CREATE INDEX IF NOT EXISTS usage_by_entry ON entry_usage(entry_id, used_ns);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    entry_id UNINDEXED, title, body, tags, tokenize='trigram'
);
"""

_WHITESPACE = re.compile(r"\s+")


def db_path() -> Path:
    """台账落点（``KNOWLEDGE_LEDGER_DB`` 可覆盖）。

    与 ``outcome_index`` 同理：**每次现算**，不做模块常量（数据根可运行时切换）。
    """
    override = os.environ.get("KNOWLEDGE_LEDGER_DB")
    if override:
        return Path(override)
    return Path(get_data_root()) / "knowledge" / "knowledge-ledger.sqlite"


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path.as_posix(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    try:
        conn.executescript(_FTS_SCHEMA)
    except sqlite3.Error:
        #: 该 Python 构建的 SQLite 没编 FTS5 / 没 trigram：检索降级为纯 LIKE，
        #: 由 ``search`` 如实回 ``ranking="like"``。**不报错、不假装支持**。
        pass
    return conn


def _has_fts(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entries_fts' LIMIT 1"
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def _now_ns() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_title(title: str) -> str:
    """标题归一：压空白、去首尾（``\\s`` 在 Python 里已含全角空格）。"""
    return _WHITESPACE.sub(" ", _text(title)).lower()


def entry_id(kind: str, scope: str, title: str) -> str:
    """稳定 id —— 同一件事（同类、同作用域、同标题）永远同一个 id。"""
    payload = f"{kind}|{scope}|{normalize_title(title)}"
    return f"{kind}-{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def _split_evidence(evidence: Any) -> list[str]:
    if evidence is None or evidence == "":
        return []
    if isinstance(evidence, str):
        return [part.strip() for part in evidence.split(",") if part.strip()]
    if isinstance(evidence, Iterable):
        return [str(item).strip() for item in evidence if str(item).strip()]
    return [str(evidence)]


def check_evidence(evidence: list[str]) -> dict[str, Any]:
    """逐条核证据。返回 ``{ok, bad, checked}``。

    核的三种合法形状：① ``outcome_index`` 里的 run_id；② 盘上存在的路径；
    ③ 相对仓库根/数据根能解析到的路径。**别放行"看起来像 id"的字符串** —— 那等于没核。
    """
    bad: list[str] = []
    for item in evidence:
        if outcome_index.exists(item):
            continue
        candidates = [Path(item), PROJECT_ROOT / item, Path(get_data_root()) / item]
        if any(path.exists() for path in candidates):
            continue
        bad.append(item)
    return {"ok": not bad, "bad": bad, "checked": len(evidence)}


def _sync_fts(conn: sqlite3.Connection, entry: dict[str, Any]) -> None:
    if not _has_fts(conn):
        return
    try:
        conn.execute("DELETE FROM entries_fts WHERE entry_id = ?", [entry["id"]])
        conn.execute(
            "INSERT INTO entries_fts (entry_id, title, body, tags) VALUES (?,?,?,?)",
            [entry["id"], entry["title"], entry["body"], entry.get("tags") or ""],
        )
    except sqlite3.Error:
        pass  # fts 索引坏了只是搜不到，不影响正表


def upsert(*, kind: str, scope: str, title: str, body: str,
           evidence: Any = None, tags: str = "", confidence: float = 0.0,
           status: str = "proposed", source_kind: str = "human",
           source_ref: str = "", links: Iterable[tuple[str, str]] | None = None,
           require_evidence: bool | None = None) -> dict[str, Any]:
    """写入或叠加一条知识（存在则 ``support+1`` 并更新正文）。

    返回 ``{"ok": bool, "id": str, "created": bool, "support": int, "why": str}``。
    **拒收时为什么不抛错**：这条链路会挂在 Agent 工具与定时蒸馏上，
    一次拒收不该让整个 Agent run 失败 —— 但必须把 ``why`` 如实带给调用方。
    """
    if kind not in KINDS:
        return {"ok": False, "id": "", "created": False, "support": 0,
                "why": f"未知知识类型 {kind!r}（可选：{'/'.join(KINDS)}）"}
    if status not in STATUSES:
        return {"ok": False, "id": "", "created": False, "support": 0,
                "why": f"未知状态 {status!r}（可选：{'/'.join(STATUSES)}）"}
    scope = _text(scope) or "global"
    title = _text(title)
    body = _text(body)
    if not title:
        return {"ok": False, "id": "", "created": False, "support": 0, "why": "标题不能为空"}
    if not body:
        return {"ok": False, "id": "", "created": False, "support": 0, "why": "正文不能为空"}

    items = _split_evidence(evidence)
    if require_evidence is None:
        require_evidence = kind not in _EXTERNAL_KINDS

    if require_evidence and not items:
        return {"ok": False, "id": "", "created": False, "support": 0,
                "why": f"{kind} 必须带证据（run_id 或盘上文件路径），空证据的经验不可采信"}
    if items:
        checked = check_evidence(items)
        if not checked["ok"]:
            return {"ok": False, "id": "", "created": False, "support": 0,
                    "why": "证据核不过（这些 id 既不是索引里的 run，也不是盘上的文件）："
                           + ", ".join(checked["bad"][:5])}

    key = entry_id(kind, scope, title)
    now = _now_ns()
    evidence_text = ",".join(items)
    try:
        with _connect() as conn:
            existing = conn.execute("SELECT * FROM entries WHERE id = ?", [key]).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO entries (id, kind, scope, title, body, tags, confidence, status,"
                    " evidence, support, contra, lift, source_kind, source_ref, created_ns, updated_ns)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [key, kind, scope, title, body, _text(tags), float(confidence or 0),
                     status, evidence_text, 1, 0, None, _text(source_kind), _text(source_ref),
                     now, now],
                )
                entry = {"id": key, "title": title, "body": body, "tags": _text(tags)}
                _sync_fts(conn, entry)
                if links:
                    _write_links(conn, key, links)
                return {"ok": True, "id": key, "created": True, "support": 1, "why": ""}

            support = int(existing["support"] or 0) + 1
            #: 已 accepted 的条目被再次提炼 ⇒ 只加 support，**不把状态打回 proposed**
            new_status = existing["status"]
            conn.execute(
                "UPDATE entries SET body=?, tags=?, support=?, confidence=?, evidence=?,"
                " updated_ns=?, source_kind=?, source_ref=? WHERE id=?",
                [body, _text(tags) or existing["tags"], support,
                 max(float(existing["confidence"] or 0), float(confidence or 0)),
                 _merge_evidence(existing["evidence"], evidence_text), now,
                 _text(source_kind) or existing["source_kind"], _text(source_ref) or existing["source_ref"],
                 key],
            )
            _sync_fts(conn, {"id": key, "title": title, "body": body, "tags": _text(tags)})
            if links:
                _write_links(conn, key, links)
            return {"ok": True, "id": key, "created": False, "support": support,
                    "why": "", "status": new_status}
    except sqlite3.Error as exc:
        return {"ok": False, "id": "", "created": False, "support": 0, "why": f"台账不可写：{exc}"}


def _merge_evidence(old: str | None, new: str | None) -> str:
    merged: list[str] = []
    for chunk in (old or "", new or ""):
        for item in chunk.split(","):
            item = item.strip()
            if item and item not in merged:
                merged.append(item)
    return ",".join(merged)


def _write_links(conn: sqlite3.Connection, key: str,
                 links: Iterable[tuple[str, str]]) -> None:
    for ref_kind, ref_id in links:
        ref_kind, ref_id = _text(ref_kind), _text(ref_id)
        if not ref_kind or not ref_id:
            continue
        try:
            conn.execute(
                "INSERT OR IGNORE INTO entry_links (entry_id, ref_kind, ref_id) VALUES (?,?,?)",
                [key, ref_kind, ref_id],
            )
        except sqlite3.Error:
            continue


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

def _like_literal(value: str) -> str:
    """LIKE 字面量（配合 ``ESCAPE '\\'``）——文件名/正文里的 ``_`` ``%`` 必须转义，
    否则会**静默匹配到别的东西**（本仓剧本库已踩过这个坑）。反斜杠要最先处理。"""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _row_to_entry(row: sqlite3.Row) -> dict[str, Any]:
    item = {key: row[key] for key in row.keys()}
    item["evidence"] = [x for x in (item.get("evidence") or "").split(",") if x]
    return item


def search(query: str, *, kind: str | None = None, scope: str | None = None,
           status: str | None = None, limit: int = 10) -> dict[str, Any]:
    """检索知识（三种 ranking 不可比，回执里如实写明）。"""
    q = _text(query)
    limit = max(1, min(int(limit or 10), 50))
    clauses: list[str] = []
    params: list[Any] = []
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if scope:
        clauses.append("(scope = ? OR scope = 'global')")
        params.append(scope)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    try:
        conn = _connect()
    except sqlite3.Error as exc:
        return {"query": q, "ranking": "none", "total": 0, "items": [], "limit": limit,
                "available": False, "hint": f"台账不可用：{exc}"}

    with conn:
        if not q:
            rows = conn.execute(
                f"SELECT * FROM entries{where} ORDER BY updated_ns DESC LIMIT ?",
                [*params, limit]).fetchall()
            total = len(rows)
            return {"query": q, "ranking": "recent", "total": total,
                    "items": [_row_to_entry(r) for r in rows], "limit": limit,
                    "available": True,
                    "hint": "空查询：按最近更新倒序，与相关性无关"}

        ranking = "like"
        rows: list[sqlite3.Row] = []
        if len(q) >= 3 and _has_fts(conn):
            try:
                match = '"' + q.replace('"', '""') + '"'
                fts_rows = conn.execute(
                    "SELECT entry_id, bm25(entries_fts) AS rank FROM entries_fts "
                    "WHERE entries_fts MATCH ? ORDER BY rank LIMIT ?",
                    [match, limit * 3]).fetchall()
                if fts_rows:
                    ranking = "bm25"
                    order = {str(r["entry_id"]): index for index, r in enumerate(fts_rows)}
                    ids = list(order)
                    marks = ",".join("?" for _ in ids)
                    extra = f"{where} AND id IN ({marks})" if where else f" WHERE id IN ({marks})"
                    found = conn.execute(f"SELECT * FROM entries{extra}",
                                         [*params, *ids]).fetchall()
                    rows = sorted(found, key=lambda r: order.get(str(r["id"]), 1_000))
            except sqlite3.Error:
                rows = []

        if not rows:
            ranking = "like"
            literal = f"%{_like_literal(q)}%"
            like_sql = (f"SELECT * FROM entries{where}"
                        f"{' AND' if where else ' WHERE'} "
                        f"(title LIKE ? ESCAPE '\\' OR body LIKE ? ESCAPE '\\' "
                        f"OR tags LIKE ? ESCAPE '\\') ORDER BY updated_ns DESC LIMIT ?")
            rows = conn.execute(like_sql, [*params, literal, literal, literal, limit]).fetchall()

        items = [_row_to_entry(r) for r in rows[:limit]]
        return {
            "query": q, "ranking": ranking, "total": len(items), "items": items,
            "limit": limit, "available": True,
            "hint": ("bm25 序（越小越相关，只覆盖 ≥3 字查询）" if ranking == "bm25"
                     else "LIKE 包含匹配（不支持词序/相关性；1–2 字中文只能走这条）"),
        }


def list_entries(*, kind: str | None = None, scope: str | None = None,
                 status: str | None = None, limit: int = 50,
                 offset: int = 0) -> dict[str, Any]:
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset))
    clauses: list[str] = []
    params: list[Any] = []
    for name, value in (("kind", kind), ("scope", scope), ("status", status)):
        if value:
            clauses.append(f"{name} = ?")
            params.append(value)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    try:
        with _connect() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) FROM entries{where}",
                                     params).fetchone()[0])
            rows = conn.execute(
                f"SELECT * FROM entries{where} ORDER BY updated_ns DESC LIMIT ? OFFSET ?",
                [*params, limit, offset]).fetchall()
    except sqlite3.Error as exc:
        return {"total": 0, "items": [], "limit": limit, "offset": offset,
                "available": False, "hint": f"台账不可用：{exc}"}
    return {"total": total, "items": [_row_to_entry(r) for r in rows],
            "limit": limit, "offset": offset, "available": True}


def detail(entry: str) -> dict[str, Any] | None:
    """条目详情 ＋ 挂载关系 ＋ 被用过多少次。"""
    try:
        with _connect() as conn:
            row = conn.execute("SELECT * FROM entries WHERE id = ?", [entry]).fetchone()
            if row is None:
                return None
            item = _row_to_entry(row)
            item["links"] = [
                {"refKind": r["ref_kind"], "refId": r["ref_id"]}
                for r in conn.execute(
                    "SELECT ref_kind, ref_id FROM entry_links WHERE entry_id = ?", [entry])
            ]
            usage_rows = conn.execute(
                "SELECT run_id, used_ns, outcome FROM entry_usage WHERE entry_id = ? "
                "ORDER BY used_ns DESC LIMIT 50", [entry]).fetchall()
            item["usage"] = [
                {"runId": r["run_id"], "usedNs": r["used_ns"], "outcome": r["outcome"]}
                for r in usage_rows
            ]
            item["usageSummary"] = _summarize_usage(
                [r["outcome"] for r in conn.execute(
                    "SELECT outcome FROM entry_usage WHERE entry_id = ?", [entry])])
    except sqlite3.Error as exc:
        return {"id": entry, "available": False, "hint": f"台账不可用：{exc}"}
    return item


def _summarize_usage(outcomes: list[Any]) -> dict[str, Any]:
    ok = sum(1 for o in outcomes if o == "ok")
    failed = sum(1 for o in outcomes if o == "failed")
    settled = ok + failed
    return {
        "ok": ok, "failed": failed, "unknown": len(outcomes) - settled,
        "settled": settled,
        #: ⚠️ 这里只是「注入后结果的比例」，**不是 lift**：lift 要跟同期未注入的基线比，
        #: 由 ``knowledge_distill`` / 引擎侧计算（此处不编一个好看的数字）。
        "okRate": round(ok / settled, 4) if settled else None,
    }


def set_status(entry: str, status: str, *, reason: str = "") -> dict[str, Any]:
    """状态迁移（人审用）。非法迁移返回 ``ok=False`` 并说明，不静默改。"""
    if status not in STATUSES:
        return {"ok": False, "id": entry, "why": f"未知状态 {status!r}"}
    try:
        with _connect() as conn:
            row = conn.execute("SELECT status FROM entries WHERE id = ?", [entry]).fetchone()
            if row is None:
                return {"ok": False, "id": entry, "why": "条目不存在"}
            current = str(row["status"] or "proposed")
            if current == status:
                return {"ok": True, "id": entry, "from": current, "to": status, "why": "已是该状态"}
            allowed = _TRANSITIONS.get(current, set())
            if status not in allowed:
                return {"ok": False, "id": entry, "from": current,
                        "why": f"不允许 {current} → {status}（可选：{'/'.join(sorted(allowed)) or '无'}）"}
            conn.execute(
                "UPDATE entries SET status = ?, updated_ns = ?, source_ref = "
                "CASE WHEN ? = '' THEN source_ref ELSE source_ref || ' | ' || ? END "
                "WHERE id = ?", [status, _now_ns(), _text(reason), _text(reason), entry])
            return {"ok": True, "id": entry, "from": current, "to": status, "why": _text(reason)}
    except sqlite3.Error as exc:
        return {"ok": False, "id": entry, "why": f"台账不可写：{exc}"}


def set_lift(entry: str, lift: float | None, *, confidence: float | None = None) -> bool:
    """写入效果分。

    ``lift=None`` 的语义是「样本不足，**还没有结论**」—— 不是「0，也就是没影响」。
    这两者混用会让看板把"没测"读成"没用"。
    """
    try:
        with _connect() as conn:
            if confidence is None:
                conn.execute("UPDATE entries SET lift = ?, updated_ns = ? WHERE id = ?",
                             [lift, _now_ns(), entry])
            else:
                conn.execute(
                    "UPDATE entries SET lift = ?, confidence = ?, updated_ns = ? WHERE id = ?",
                    [lift, float(confidence), _now_ns(), entry])
        return True
    except sqlite3.Error:
        return False


def record_usage(entry: str, run_id: str | None, outcome: str | None = None) -> bool:
    """记一次「这条知识被注入到了哪个 run」。

    这是 ``lift`` 的**唯一**数据来源：没有它，知识库就只是个文档柜，
    谁也不知道哪条真有用、哪条是负资产。
    """
    if outcome is not None and outcome not in ("ok", "failed", None):
        outcome = "failed" if outcome in ("error", "fail") else "ok"
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO entry_usage (entry_id, run_id, used_ns, outcome) VALUES (?,?,?,?)",
                [entry, _text(run_id) or None, _now_ns(), outcome])
        return True
    except sqlite3.Error:
        return False


def usage_summary(entry: str) -> dict[str, Any]:
    try:
        with _connect() as conn:
            rows = conn.execute("SELECT outcome FROM entry_usage WHERE entry_id = ?",
                                [entry]).fetchall()
    except sqlite3.Error:
        return {"ok": 0, "failed": 0, "unknown": 0, "settled": 0, "okRate": None}
    return _summarize_usage([r["outcome"] for r in rows])


def stats() -> dict[str, Any]:
    """台账盘点：按 kind × status 计数、按 scope 分布、最近 7 天新增。"""
    try:
        with _connect() as conn:
            by_kind_status = [
                {"kind": r["kind"], "status": r["status"], "count": r["n"]}
                for r in conn.execute(
                    "SELECT kind, status, COUNT(*) AS n FROM entries GROUP BY kind, status")
            ]
            by_scope = [
                {"scope": r["scope"], "count": r["n"]}
                for r in conn.execute(
                    "SELECT scope, COUNT(*) AS n FROM entries GROUP BY scope ORDER BY n DESC")
            ]
            total = int(conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
            recent = int(conn.execute(
                "SELECT COUNT(*) FROM entries WHERE created_ns >= ?",
                [_now_ns() - 7 * 86_400 * 1_000_000_000]).fetchone()[0])
    except sqlite3.Error as exc:
        return {"total": 0, "available": False, "hint": f"台账不可用：{exc}"}
    return {
        "total": total,
        "recent7d": recent,
        "byKindStatus": by_kind_status,
        "byScope": by_scope,
        "db": db_path().as_posix(),
        "available": True,
    }
