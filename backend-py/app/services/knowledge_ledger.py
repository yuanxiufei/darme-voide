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
    "ROLES",
    "ACTIONS",
    "db_path",
    "upsert",
    "search",
    "related",
    "for_context",
    "context_refs",
    "SIMILAR_RATIO",
    "list_entries",
    "detail",
    "set_status",
    "set_lift",
    "record_usage",
    "usage_summary",
    "stats",
    "entry_id",
    "normalize_tags",
    "tag_list",
    "update_entry",
    "soft_delete",
    "restore",
    "purge",
    "versions",
    "version_detail",
    "revert",
    "tag_facets",
    "taxonomy",
    "audit_log",
]

#: 知识类型（新增一类要同时想清楚：谁会读它、凭什么判它有用）
KINDS = ("fact", "work", "recipe", "lesson", "failure", "threshold", "prompt_patch")

#: 类型的**中文名**：分类导航要用，且必须与 ``reasons`` 里的说法一致。
#:
#: 「分类」在本仓**不新造一列自由文本** —— 自由文本分类会在两周内退化成第二个
#: ``tags``（同一类知识被写成「节奏」「节奏问题」「 pacing」三种）。分类就是
#: ``kind``（这是什么知识，闭集）＋``scope``（管哪儿，半开集），两者都是已有列，
#: 可枚举、可计数、可校验 ⇒ 天然可管理。
KIND_LABELS: dict[str, str] = {
    "fact": "行业事实",
    "work": "作品资料",
    "recipe": "成法",
    "lesson": "经验",
    "failure": "失败样本",
    "threshold": "阈值建议",
    "prompt_patch": "提示词补丁",
}

#: 操作角色（**声明式**，不是认证 —— 见 ``docs/knowledge-base-contract.md`` §4）
ROLES = ("viewer", "editor", "owner")

#: 审计动作（写进 ``entry_audit.action``，闭集）
ACTIONS = ("create", "support", "update", "status", "delete", "restore", "revert", "purge")

#: 标签上限：单条 24 字符、每条目 12 个。
#:
#: 不设上限时 ``tags`` 会长成段落 —— 它既进 FTS 索引又进 facet 统计，
#: 一个 200 字的"标签"会让 facet 页出现一行占满屏的条目。
MAX_TAG_LEN = 24
MAX_TAGS = 12

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
CREATE TABLE IF NOT EXISTS entry_versions (
    entry_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    tags TEXT DEFAULT '',
    evidence TEXT DEFAULT '',
    confidence REAL DEFAULT 0,
    status TEXT NOT NULL,
    editor TEXT DEFAULT '',
    reason TEXT DEFAULT '',
    created_ns INTEGER NOT NULL,
    UNIQUE(entry_id, version)
);
CREATE TABLE IF NOT EXISTS entry_audit (
    id INTEGER PRIMARY KEY,
    entry_id TEXT NOT NULL,
    action TEXT NOT NULL,
    operator TEXT DEFAULT '',
    role TEXT DEFAULT '',
    at_ns INTEGER NOT NULL,
    detail TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS audit_by_entry ON entry_audit(entry_id, at_ns);
CREATE INDEX IF NOT EXISTS audit_by_time ON entry_audit(at_ns);
"""

#: 旧库加列（``CREATE TABLE IF NOT EXISTS`` 改不了已存在的表）。
#:
#: 三列各有不可替代的用途，缺一不可：``version`` 是乐观锁的载体；
#: ``deleted_ns`` 让删除可撤销且不丢用量历史；``deleted_by`` 让"谁删的"可回溯
#: （只靠审计表也能查，但列表页要一眼看出状态，不该为一列信息再查一次审计）。
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("version", "ALTER TABLE entries ADD COLUMN version INTEGER DEFAULT 1"),
    ("deleted_ns", "ALTER TABLE entries ADD COLUMN deleted_ns INTEGER DEFAULT 0"),
    ("deleted_by", "ALTER TABLE entries ADD COLUMN deleted_by TEXT DEFAULT ''"),
)

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
    _ensure_columns(conn)
    try:
        conn.executescript(_FTS_SCHEMA)
    except sqlite3.Error:
        #: 该 Python 构建的 SQLite 没编 FTS5 / 没 trigram：检索降级为纯 LIKE，
        #: 由 ``search`` 如实回 ``ranking="like"``。**不报错、不假装支持**。
        pass
    return conn


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """给旧库补列（幂等）。

    ⚠️ 必须逐列 ``PRAGMA table_info`` 判存在，不能靠 ``ALTER TABLE`` 报错兜底 ——
    报错兜底会把"磁盘满/库损坏"这类真错一起吞掉，表现为**静默没有版本号**，
    然后乐观锁永远不生效（``expectedVersion`` 每次都对不上或都对得上，两种都错）。
    """
    try:
        existing = {str(r["name"]) for r in conn.execute("PRAGMA table_info(entries)")}
    except sqlite3.Error:
        return
    for name, ddl in _ADDED_COLUMNS:
        if name in existing:
            continue
        try:
            conn.execute(ddl)
        except sqlite3.Error:
            continue


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


def _col(row: sqlite3.Row, name: str, default: Any = None) -> Any:
    """取列，**列不存在时给默认值**（旧库迁移失败/测试用简化表都会出现这种情况）。

    直接用 ``row[name]`` 在缺列时抛的是 ``IndexError``，``except sqlite3.Error`` 接不住，
    表现为一个"改个标签就把接口打成 500"的诡异 bug。
    """
    try:
        return row[name]
    except (IndexError, KeyError):
        return default


def tag_list(tags: Any) -> list[str]:
    """标签串 → 列表（**只读拆分，不做归一**）。

    归一只在写入口做（``normalize_tags``）：读入口若也"顺手修"，同一份数据会
    在"旧行读出 A、写回后变 B"之间来回抖，审计里就会出现一堆 `update` 却看不出改了什么。
    """
    if isinstance(tags, (list, tuple)):
        return [str(item).strip() for item in tags if str(item).strip()]
    return [part.strip() for part in str(tags or "").split(",") if part.strip()]


def normalize_tags(tags: Any) -> str:
    """标签归一 → 逗号串（落库格式，**不能改成 JSON**：FTS 与前端都按逗号串读）。

    归一四件事：① 全角逗号/顿号/分号都当分隔符（用户从 Word 粘过来就是这样）；
    ② trim 后去空；③ 去重（保序，首个写法为准 —— 「节奏 」与「节奏」是同一个）；
    ④ 单标签截到 ``MAX_TAG_LEN``、整体最多 ``MAX_TAGS`` 个。
    超限**不报错只截断**，且截断本身可被测试检出（回执里的 ``tags`` 就是答案）。
    """
    if isinstance(tags, (list, tuple)):
        raw = [str(item) for item in tags]
    else:
        text = str(tags or "")
        for sep in ("，", "、", "；", ";", "|"):
            text = text.replace(sep, ",")
        raw = text.split(",")
    out: list[str] = []
    for item in raw:
        name = _WHITESPACE.sub(" ", str(item)).strip()
        if not name:
            continue
        name = name[:MAX_TAG_LEN]
        if name not in out:
            out.append(name)
        if len(out) >= MAX_TAGS:
            break
    return ",".join(out)


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


def _audit(conn: sqlite3.Connection, entry: str, action: str, *, operator: str = "",
           role: str = "", detail: str = "") -> None:
    """写一条审计。

    审计写入失败**不让主操作失败**（与 ``outcome_index`` 的记账同一条哲学）：
    一次"改成功了但没留痕"比"改不成"更接近可用；但必须捞异常，不能静默崩在半路。
    """
    try:
        conn.execute(
            "INSERT INTO entry_audit (entry_id, action, operator, role, at_ns, detail)"
            " VALUES (?,?,?,?,?,?)",
            [entry, action, _text(operator), _text(role), _now_ns(), _text(detail)],
        )
    except sqlite3.Error:
        pass


def _snapshot(conn: sqlite3.Connection, row: sqlite3.Row, *, editor: str = "",
              reason: str = "") -> int:
    """把**当前行**存成版本快照，返回它对应的版本号。

    约定：``entries.version`` 是**当前值**的版本号，快照存的是"被替换掉的旧值"。
    于是 ``versions`` 列表 = 1..current-1 全是历史，current 只在 ``entries`` 里 ——
    历史表因此不会因为"每次读都写一遍"而膨胀。
    """
    version = int(_col(row, "version", 1) or 1)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO entry_versions"
            " (entry_id, version, kind, scope, title, body, tags, evidence, confidence,"
            "  status, editor, reason, created_ns)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [row["id"], version, row["kind"], row["scope"], row["title"], row["body"],
             row["tags"] or "", row["evidence"] or "", float(row["confidence"] or 0),
             row["status"], _text(editor), _text(reason), _now_ns()],
        )
    except sqlite3.Error:
        return version
    return version


def upsert(*, kind: str, scope: str, title: str, body: str,
           evidence: Any = None, tags: str = "", confidence: float = 0.0,
           status: str = "proposed", source_kind: str = "human",
           source_ref: str = "", links: Iterable[tuple[str, str]] | None = None,
           require_evidence: bool | None = None,
           operator: str = "system", role: str = "") -> dict[str, Any]:
    """写入或叠加一条知识（存在则 ``support+1`` 并更新正文）。

    返回 ``{"ok": bool, "id": str, "created": bool, "support": int, "why": str}``。
    **拒收时为什么不抛错**：这条链路会挂在 Agent 工具与定时蒸馏上，
    一次拒收不该让整个 Agent run 失败 —— 但必须把 ``why`` 如实带给调用方。
    """
    if kind not in KINDS:
        return {"ok": False, "id": "", "created": False, "support": 0, "code": "invalid",
                "why": f"未知知识类型 {kind!r}（可选：{'/'.join(KINDS)}）"}
    if status not in STATUSES:
        return {"ok": False, "id": "", "created": False, "support": 0, "code": "invalid",
                "why": f"未知状态 {status!r}（可选：{'/'.join(STATUSES)}）"}
    scope = _text(scope) or "global"
    title = _text(title)
    body = _text(body)
    if not title:
        return {"ok": False, "id": "", "created": False, "support": 0, "code": "invalid",
                "why": "标题不能为空"}
    if not body:
        return {"ok": False, "id": "", "created": False, "support": 0, "code": "invalid",
                "why": "正文不能为空"}

    items = _split_evidence(evidence)
    if require_evidence is None:
        require_evidence = kind not in _EXTERNAL_KINDS

    if require_evidence and not items:
        return {"ok": False, "id": "", "created": False, "support": 0, "code": "evidence",
                "why": f"{kind} 必须带证据（run_id 或盘上文件路径），空证据的经验不可采信"}
    if items:
        checked = check_evidence(items)
        if not checked["ok"]:
            return {"ok": False, "id": "", "created": False, "support": 0, "code": "evidence",
                    "why": "证据核不过（这些 id 既不是索引里的 run，也不是盘上的文件）："
                           + ", ".join(checked["bad"][:5])}

    key = entry_id(kind, scope, title)
    now = _now_ns()
    evidence_text = ",".join(items)
    #: 写入口归一（读入口不归一，见 ``tag_list`` 的说明）
    tags_text = normalize_tags(tags)
    try:
        with _connect() as conn:
            existing = conn.execute("SELECT * FROM entries WHERE id = ?", [key]).fetchone()
            if existing is not None and int(_col(existing, "deleted_ns", 0) or 0):
                #: 已删条目被重新提炼：**不复活、不新建**，如实拒收。
                #: 自动复活会让"我删掉的错误经验"被下一次蒸馏悄悄带回来；
                #: 想恢复就走 ``restore``（那是人的动作，有审计）。
                return {"ok": False, "id": key, "created": False, "support": 0,
                        "code": "deleted",
                        "why": f"该条目已被删除（{existing['deleted_by'] or '未知'}），"
                               "要恢复请走 restore，不能靠重新提炼复活"}
            if existing is None:
                conn.execute(
                    "INSERT INTO entries (id, kind, scope, title, body, tags, confidence, status,"
                    " evidence, support, contra, lift, source_kind, source_ref, created_ns,"
                    " updated_ns, version)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [key, kind, scope, title, body, tags_text, float(confidence or 0),
                     status, evidence_text, 1, 0, None, _text(source_kind), _text(source_ref),
                     now, now, 1],
                )
                entry = {"id": key, "title": title, "body": body, "tags": tags_text}
                _sync_fts(conn, entry)
                if links:
                    _write_links(conn, key, links)
                _audit(conn, key, "create", operator=operator, role=role,
                       detail=f"{kind}/{scope}")
                return {"ok": True, "id": key, "created": True, "support": 1, "why": ""}

            support = int(existing["support"] or 0) + 1
            #: 已 accepted 的条目被再次提炼 ⇒ 只加 support，**不把状态打回 proposed**
            new_status = existing["status"]
            conn.execute(
                "UPDATE entries SET body=?, tags=?, support=?, confidence=?, evidence=?,"
                " updated_ns=?, source_kind=?, source_ref=? WHERE id=?",
                [body, tags_text or existing["tags"], support,
                 max(float(existing["confidence"] or 0), float(confidence or 0)),
                 _merge_evidence(existing["evidence"], evidence_text), now,
                 _text(source_kind) or existing["source_kind"], _text(source_ref) or existing["source_ref"],
                 key],
            )
            _sync_fts(conn, {"id": key, "title": title, "body": body,
                             "tags": tags_text or existing["tags"]})
            if links:
                _write_links(conn, key, links)
            _audit(conn, key, "support", operator=operator, role=role,
                   detail=f"support={support}")
            return {"ok": True, "id": key, "created": False, "support": support,
                    "why": "", "status": new_status}
    except sqlite3.Error as exc:
        return {"ok": False, "id": "", "created": False, "support": 0,
                "code": "unavailable", "why": f"台账不可写：{exc}"}


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
    """行 → 前端契约对象。

    **既有字段一律不动**（前端与 131 条闭环测试都在读 ``title``/``body``/``support``…），
    只在后面补派生字段：``tagList``（拆分好的标签）、``version``、``deleted``/``deletedNs``、
    ``deletedBy``。派生字段必须由服务端算：让每个前端各自 ``split(',')``，
    标签里一旦出现半角逗号（归一已挡）或超长串的两端空白，两边显示就会不一致。
    """
    item = {key: row[key] for key in row.keys()}
    item["evidence"] = [x for x in (item.get("evidence") or "").split(",") if x]
    item["tagList"] = tag_list(item.get("tags") or "")
    item["version"] = int(item.get("version") or 1)
    item["deletedNs"] = int(item.get("deleted_ns") or 0)
    item["deleted"] = bool(item["deletedNs"])
    item["deletedBy"] = str(item.get("deleted_by") or "")
    return item


def _snippet(text: str, query: str, *, width: int = 90) -> str:
    """命中片段（一段，去掉换行）。前端不该自己截 —— 各自截法不同，两边看到的首句会不一样。"""
    body = str(text or "").replace("\r", " ").replace("\n", " ")
    body = _WHITESPACE.sub(" ", body).strip()
    if not body:
        return ""
    if not query:
        return body[: width * 2]
    position = body.lower().find(query.lower())
    if position < 0:
        return body[: width * 2]
    start = max(0, position - width)
    end = min(len(body), position + len(query) + width)
    return ("…" if start else "") + body[start:end] + ("…" if end < len(body) else "")


def _tag_clause(tags: Iterable[str] | None) -> tuple[list[str], list[Any]]:
    """标签过滤（AND 语义：多标签 = 同时具备）。

    用 ``',' || tags || ','`` 做**词边界**匹配：直接 ``LIKE %节奏%`` 会把「节奏慢」
    也算成命中「节奏」，facets 里点一个标签却冒出另一批条目，用户只会以为筛选坏了。
    """
    clauses: list[str] = []
    params: list[Any] = []
    #: ⚠️ 这里**必须先把 str 与序列分开**：``list("节奏")`` 得到 ``['节','奏']``，
    #: 于是 ``tags="节奏"`` 会被当成「同时具备 节 与 奏」的 AND 条件 ⇒ 一条都查不到，
    #: 而 facet 照样显示「节奏 2 条」—— 页面能看出毛病，SQL 却不报错。
    raw = tags if isinstance(tags, str) or tags is None else ",".join(str(t) for t in tags)
    for tag in tag_list(raw):
        pattern = f"%{_like_literal(',' + tag + ',')}%"
        clauses.append("(',' || REPLACE(COALESCE(tags,''), ' ', '') || ',') LIKE ? ESCAPE '\\'")
        params.append(pattern)
    return clauses, params


def search(query: str, *, kind: str | None = None, scope: str | None = None,
           status: str | None = None, tags: Iterable[str] | None = None,
           include_deleted: bool = False, limit: int = 10,
           offset: int = 0) -> dict[str, Any]:
    """检索知识（三种 ranking 不可比，回执里如实写明）。

    ``total`` 是**全量命中数**（不是本页条数）—— 分页 UI 要它；旧实现回的是
    ``len(items)``，于是"共 10 条"永远等于每页大小，翻页按钮永远显示只有一页。
    候选集上限 500，超出时 ``totalCapped=True`` 并如实写明"至少 500"。
    """
    q = _text(query)
    limit = max(1, min(int(limit or 10), 50))
    offset = max(0, int(offset))
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
    if not include_deleted:
        #: 软删条目默认不出现在任何"找人看"的路径上；``include_deleted`` 是给审计/回收站用的
        clauses.append("COALESCE(deleted_ns, 0) = 0")
    tag_clauses, tag_params = _tag_clause(tags)
    clauses.extend(tag_clauses)
    params.extend(tag_params)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    try:
        conn = _connect()
    except sqlite3.Error as exc:
        return {"query": q, "ranking": "none", "total": 0, "items": [], "limit": limit,
                "offset": offset, "available": False, "hint": f"台账不可用：{exc}"}

    cap = 500
    with conn:
        if not q:
            total = int(conn.execute(f"SELECT COUNT(*) FROM entries{where}",
                                     params).fetchone()[0])
            rows = conn.execute(
                f"SELECT * FROM entries{where} ORDER BY updated_ns DESC LIMIT ? OFFSET ?",
                [*params, limit, offset]).fetchall()
            return {"query": q, "ranking": "recent", "total": total,
                    "items": [_row_to_entry(r) for r in rows], "limit": limit,
                    "offset": offset, "available": True, "totalCapped": False,
                    "hint": "空查询：按最近更新倒序，与相关性无关"}

        ranking = "like"
        rows: list[sqlite3.Row] = []
        capped = False
        if len(q) >= 3 and _has_fts(conn):
            try:
                match = '"' + q.replace('"', '""') + '"'
                fts_rows = conn.execute(
                    "SELECT entry_id, bm25(entries_fts) AS rank FROM entries_fts "
                    "WHERE entries_fts MATCH ? ORDER BY rank LIMIT ?",
                    [match, cap]).fetchall()
                capped = len(fts_rows) >= cap
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
            rows = conn.execute(like_sql,
                                [*params, literal, literal, literal, cap]).fetchall()
            capped = len(rows) >= cap

        items = []
        for row in rows[offset: offset + limit]:
            item = _row_to_entry(row)
            item["snippet"] = _snippet(item.get("body") or "", q)
            items.append(item)
        return {
            "query": q, "ranking": ranking, "total": len(rows), "items": items,
            "limit": limit, "offset": offset, "available": True, "totalCapped": capped,
            "hint": ("bm25 序（越小越相关，只覆盖 ≥3 字查询）" if ranking == "bm25"
                     else "LIKE 包含匹配（不支持词序/相关性；1–2 字中文只能走这条）"),
        }


# ---------------------------------------------------------------------------
# 关联：把 entry_links 从「只写」变成可正反向读（2026-11-08）
# ---------------------------------------------------------------------------
#
# 缘起：``entry_links`` 自蒸馏上线起就一直在写（按 ``stage``/``shot`` 挂），但**全仓
# 没有一处读它**（``detail`` 只回显本条挂的 ref，没有反向、没有邻居）——写进去的关联
# 等于不存在。参考实现（graphiti 的双向边、cognee 的图补全、mem0 的相近记忆召回）
# 都说明这个能力是「记忆」与「一堆条目」的分水岭，而它们的共同前提不需要向量：
# **边已经在库里了，先把它读出来**。
#
# 两档 ``via`` 的**证据强度不同，回执分开给、不混排成一条分数**：
#   * ``shared``：同一 ``(refKind, refId)`` 上还挂着谁 —— 有事实依据（同阶段／同镜头）；
#   * ``similar``：正文 trigram Jaccard ≥ ``SIMILAR_RATIO`` —— 纯启发式，只作提示，
#     **不当成合并依据**（同题材不同结论的条目字面也很像，合并会抹掉分歧）。

#: 文本相似的默认闸门。0.6 对应「同一件事换个说法」（词序/语气词微调）；
#: 调低会把同题材、结论相反的条目混进来 —— 那是这本台账最值钱的分歧，不能被"去重"。
SIMILAR_RATIO = 0.6

#: 相似比较的候选上限（同 kind 内按最近更新取）。台账规模是百级，
#: 全量两两比较也没问题，但**别让它随库增长变成隐形的慢查询**。
_SIMILAR_POOL = 300

#: ``related`` / ``for_context`` 回执里条目摘要保留的正文长度
_BRIEF_CHARS = 120


def _trigrams(text: Any) -> set[str]:
    """3 字滑窗（大小写与空白先归一）。

    为什么是 trigram 而不是分词：中文没有空格，本仓也不想为此引一个分词器依赖；
    3 字窗对「换词序」「加语气词」稳健，对「换主语」敏感 —— 正好是我们要的粒度。
    """
    plain = _WHITESPACE.sub(" ", str(text or "")).strip().lower()
    if not plain:
        return set()
    if len(plain) <= 3:
        return {plain}
    return {plain[i:i + 3] for i in range(len(plain) - 2)}


def _similarity(left: Any, right: Any) -> float:
    """Jaccard 相似度（0–1）。任一侧为空返回 0.0 —— **不是** 1.0。"""
    a, b = _trigrams(left), _trigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _brief(row: sqlite3.Row) -> dict[str, Any]:
    """条目摘要（不含全文）。

    ``body`` 可能上千字，邻居列表里塞全文会让一个抽屉拉几百 KB；
    但**不能只给标题** —— 判断「这条是不是我要找的那条」几乎全靠头两句。
    """
    item = _row_to_entry(row)
    return {
        "id": item["id"],
        "kind": item["kind"],
        "scope": item["scope"],
        "title": item["title"],
        "status": item["status"],
        "confidence": float(item.get("confidence") or 0.0),
        "support": int(item.get("support") or 0),
        "contra": int(item.get("contra") or 0),
        "lift": item.get("lift"),
        "tags": item["tagList"],
        "version": item["version"],
        "deleted": item["deleted"],
        "updatedNs": int(item.get("updated_ns") or 0),
        "snippet": _WHITESPACE.sub(" ", str(item.get("body") or ""))[:_BRIEF_CHARS],
    }


def _links_of(conn: sqlite3.Connection, entry: str) -> list[dict[str, str]]:
    return [
        {"refKind": str(r["ref_kind"]), "refId": str(r["ref_id"])}
        for r in conn.execute(
            "SELECT ref_kind, ref_id FROM entry_links WHERE entry_id = ?"
            " ORDER BY ref_kind, ref_id", [entry])
    ]


def related(entry: str, *, limit: int = 10,
            include_deleted: bool = False) -> dict[str, Any]:
    """一条知识的邻居：**共享同一 ref 的兄弟条目** ＋ **正文高度相似的条目**。

    ``links`` 原样带回本条自己挂的 ref（前端要显示「挂在 stage:video 上」）；
    ``items`` 只装**别的条目**，每项带 ``via``/``score``/``refKind``/``refId``。

    条目不存在与条目已删**分开报**：前者是 ``found=False``，后者是 ``found=True``
    但 `deleted=True`（回收站里的条目点开邻居不该显示成"这条不存在"）。
    """
    limit = max(1, min(int(limit or 10), 50))
    empty_counts = {"shared": 0, "similar": 0}
    try:
        with _connect() as conn:
            row = conn.execute("SELECT * FROM entries WHERE id = ?", [entry]).fetchone()
            if row is None:
                return {"id": entry, "found": False, "available": True, "links": [],
                        "items": [], "counts": empty_counts,
                        "hint": f"没有这条知识：{entry}"}
            deleted = bool(int(_col(row, "deleted_ns", 0) or 0))
            if deleted and not include_deleted:
                return {"id": entry, "found": True, "deleted": True, "available": True,
                        "links": [], "items": [], "counts": empty_counts,
                        "hint": "这条已软删（回收站视角要 includeDeleted=true）"}
            item = _row_to_entry(row)
            links = _links_of(conn, entry)
            seen = {str(entry)}

            shared: list[dict[str, Any]] = []
            for link in links:
                rows = conn.execute(
                    "SELECT e.* FROM entry_links l JOIN entries e ON e.id = l.entry_id"
                    " WHERE l.ref_kind = ? AND l.ref_id = ? AND l.entry_id <> ?"
                    " ORDER BY e.updated_ns DESC LIMIT ?",
                    [link["refKind"], link["refId"], entry, limit]).fetchall()
                for sibling in rows:
                    key = str(sibling["id"])
                    if key in seen:
                        continue
                    if not include_deleted and int(_col(sibling, "deleted_ns", 0) or 0):
                        continue
                    seen.add(key)
                    shared.append({
                        "via": "shared", "refKind": link["refKind"], "refId": link["refId"],
                        "score": None, "entry": _brief(sibling),
                    })

            #: 相似只在**同 kind** 内比较：`fact` 与 `failure` 字面再像也不是一件事
            #: （一个是外部事实、一个是我们自己的坑），跨类相似只会在列表里制造噪声。
            clause = "SELECT * FROM entries WHERE id <> ? AND kind = ?"
            params: list[Any] = [entry, item.get("kind")]
            if not include_deleted:
                clause += " AND COALESCE(deleted_ns, 0) = 0"
            pool = conn.execute(clause + " ORDER BY updated_ns DESC LIMIT ?",
                                [*params, _SIMILAR_POOL]).fetchall()
            base_title = item.get("title") or ""
            base_body = item.get("body") or ""
            similar: list[dict[str, Any]] = []
            for other in pool:
                key = str(other["id"])
                if key in seen:
                    continue
                score = max(_similarity(base_title, other["title"]),
                            _similarity(base_body, other["body"]))
                if score < SIMILAR_RATIO:
                    continue
                similar.append({"via": "similar", "refKind": "", "refId": "",
                                "score": round(score, 4), "entry": _brief(other)})
            similar.sort(key=lambda x: -float(x["score"] or 0.0))
            similar = similar[:limit]
    except sqlite3.Error as exc:
        return {"id": entry, "found": False, "available": False, "links": [], "items": [],
                "counts": empty_counts, "hint": f"台账不可用：{exc}"}

    items = shared[:limit] + similar
    hint = ""
    if not links:
        hint = "这条没挂任何阶段/镜头（人工写的条目通常如此）：邻居只剩正文相似这一档"
    elif not shared:
        hint = "挂在 ref 上但没有别的条目共用（蒸馏出的同族条目还没出现）"
    return {"id": entry, "found": True, "deleted": deleted, "available": True,
            "links": links, "items": items,
            "counts": {"shared": len(shared), "similar": len(similar)},
            "hint": hint}


def for_context(*, stage: str | None = None, shot: str | None = None,
                status: str | None = None, include_deleted: bool = False,
                limit: int = 10) -> dict[str, Any]:
    """按「阶段 / 镜头」反查经验 —— 读 ``entry_links`` 的反向索引。

    回答的是「**这一阶段／这一镜**踩过什么坑」，与 ``search``（提到某个词的有哪些）
    不可互替：前者是结构化事实，后者是文本匹配。

    ``stage``/``shot`` 都不给时**不猜**（不返回"全部"当成上下文命中）：那会让调用方
    以为自己拿到了针对性经验。要"台账里挂了哪些 ref"走 :func:`context_refs`。
    """
    limit = max(1, min(int(limit or 10), 50))
    refs = [(kind, _text(value)) for kind, value in (("stage", stage), ("shot", shot))]
    refs = [(kind, value) for kind, value in refs if value]
    if not refs:
        return {"available": True, "items": [], "total": 0, "refs": [],
                "hint": "没给 stage/shot：按上下文取知识必须点名其中一个阶段或镜头"}
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with _connect() as conn:
            for ref_kind, ref_id in refs:
                clause = ("SELECT e.* FROM entry_links l JOIN entries e ON e.id = l.entry_id"
                          " WHERE l.ref_kind = ? AND l.ref_id = ?")
                params: list[Any] = [ref_kind, ref_id]
                if status:
                    clause += " AND e.status = ?"
                    params.append(status)
                if not include_deleted:
                    clause += " AND COALESCE(e.deleted_ns, 0) = 0"
                rows = conn.execute(
                    clause + " ORDER BY e.confidence DESC, e.updated_ns DESC LIMIT ?",
                    [*params, limit]).fetchall()
                for row in rows:
                    key = str(row["id"])
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append({"refKind": ref_kind, "refId": ref_id, "entry": _brief(row)})
    except sqlite3.Error as exc:
        return {"available": False, "items": [], "total": 0,
                "refs": [{"refKind": k, "refId": v} for k, v in refs],
                "hint": f"台账不可用：{exc}"}
    return {"available": True, "items": items, "total": len(items),
            "refs": [{"refKind": k, "refId": v} for k, v in refs],
            "hint": "" if items else "这些 ref 上还没有条目（蒸馏还没写过，或都已被删）"}


def context_refs(*, ref_kind: str | None = None, entry: str | None = None,
                 limit: int = 200) -> dict[str, Any]:
    """台账里**真被挂过**的 ref 清单（附条目数）——「有哪些阶段/镜头有经验」。

    没有这条端点时，前端只能猜阶段名（`video`? `h3`? `krea2`?）；猜错的表现是
    "查了没结果"，与"确实没有经验"永远分不开。
    """
    limit = max(1, min(int(limit or 200), 500))
    clauses: list[str] = []
    params: list[Any] = []
    if ref_kind:
        clauses.append("ref_kind = ?")
        params.append(_text(ref_kind))
    if entry:
        clauses.append("entry_id = ?")
        params.append(_text(entry))
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT ref_kind, ref_id, COUNT(*) AS n FROM entry_links"
                f"{where} GROUP BY ref_kind, ref_id ORDER BY n DESC, ref_id LIMIT ?",
                [*params, limit]).fetchall()
    except sqlite3.Error as exc:
        return {"available": False, "items": [], "total": 0, "hint": f"台账不可用：{exc}"}
    items = [{"refKind": str(r["ref_kind"]), "refId": str(r["ref_id"]),
              "entries": int(r["n"])} for r in rows]
    return {"available": True, "items": items, "total": len(items),
            "hint": "" if items else "库里还没有任何 ref（蒸馏尚未写过 stage/shot 关联）"}


def list_entries(*, kind: str | None = None, scope: str | None = None,
                 status: str | None = None, tags: Iterable[str] | None = None,
                 include_deleted: bool = False, limit: int = 50,
                 offset: int = 0) -> dict[str, Any]:
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset))
    clauses: list[str] = []
    params: list[Any] = []
    for name, value in (("kind", kind), ("scope", scope), ("status", status)):
        if value:
            clauses.append(f"{name} = ?")
            params.append(value)
    if not include_deleted:
        clauses.append("COALESCE(deleted_ns, 0) = 0")
    tag_clauses, tag_params = _tag_clause(tags)
    clauses.extend(tag_clauses)
    params.extend(tag_params)
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


def detail(entry: str, *, include_deleted: bool = False) -> dict[str, Any] | None:
    """条目详情 ＋ 挂载关系 ＋ 被用过多少次。

    已删条目默认返回 ``None``（对调用方就是"不存在"）：否则``GET /entries/{id}``
    会变成一个绕过软删的后门 —— 回收站要看得显式传 ``include_deleted=True``。
    """
    try:
        with _connect() as conn:
            row = conn.execute("SELECT * FROM entries WHERE id = ?", [entry]).fetchone()
            if row is None:
                return None
            if int(_col(row, "deleted_ns", 0) or 0) and not include_deleted:
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


def set_status(entry: str, status: str, *, reason: str = "",
               operator: str = "system", role: str = "") -> dict[str, Any]:
    """状态迁移（人审用）。非法迁移返回 ``ok=False`` 并说明，不静默改。"""
    if status not in STATUSES:
        return {"ok": False, "id": entry, "code": "invalid", "why": f"未知状态 {status!r}"}
    try:
        with _connect() as conn:
            row = conn.execute("SELECT status, deleted_ns FROM entries WHERE id = ?",
                               [entry]).fetchone()
            if row is None:
                return {"ok": False, "id": entry, "code": "not_found", "why": "条目不存在"}
            if int(_col(row, "deleted_ns", 0) or 0):
                return {"ok": False, "id": entry, "code": "deleted",
                        "why": "条目已删除 ⇒ 状态迁移无意义；先 restore 再改状态"}
            current = str(row["status"] or "proposed")
            if current == status:
                return {"ok": True, "id": entry, "from": current, "to": status, "why": "已是该状态"}
            allowed = _TRANSITIONS.get(current, set())
            if status not in allowed:
                return {"ok": False, "id": entry, "from": current, "code": "transition",
                        "why": f"不允许 {current} → {status}（可选：{'/'.join(sorted(allowed)) or '无'}）"}
            conn.execute(
                "UPDATE entries SET status = ?, updated_ns = ?, source_ref = "
                "CASE WHEN ? = '' THEN source_ref ELSE source_ref || ' | ' || ? END "
                "WHERE id = ?", [status, _now_ns(), _text(reason), _text(reason), entry])
            _audit(conn, entry, "status", operator=operator, role=role,
                   detail=f"{current}->{status}" + (f" · {_text(reason)}" if _text(reason) else ""))
            return {"ok": True, "id": entry, "from": current, "to": status, "why": _text(reason)}
    except sqlite3.Error as exc:
        return {"ok": False, "id": entry, "code": "unavailable", "why": f"台账不可写：{exc}"}


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
    """台账盘点：按 kind × status 计数、按 scope 分布、最近 7 天新增。

    **软删条目不计入任何一栏**，单独回 ``deleted``：否则"盘点 120 条、点进列表只有 80 条"
    这种对不上的账会让每次盘点都要人工解释一遍。
    """
    alive = "COALESCE(deleted_ns, 0) = 0"
    try:
        with _connect() as conn:
            by_kind_status = [
                {"kind": r["kind"], "status": r["status"], "count": r["n"]}
                for r in conn.execute(
                    "SELECT kind, status, COUNT(*) AS n FROM entries"
                    f" WHERE {alive} GROUP BY kind, status")
            ]
            by_scope = [
                {"scope": r["scope"], "count": r["n"]}
                for r in conn.execute(
                    "SELECT scope, COUNT(*) AS n FROM entries"
                    f" WHERE {alive} GROUP BY scope ORDER BY n DESC")
            ]
            total = int(conn.execute(
                f"SELECT COUNT(*) FROM entries WHERE {alive}").fetchone()[0])
            deleted = int(conn.execute(
                "SELECT COUNT(*) FROM entries WHERE COALESCE(deleted_ns, 0) > 0").fetchone()[0])
            recent = int(conn.execute(
                f"SELECT COUNT(*) FROM entries WHERE {alive} AND created_ns >= ?",
                [_now_ns() - 7 * 86_400 * 1_000_000_000]).fetchone()[0])
    except sqlite3.Error as exc:
        return {"total": 0, "available": False, "hint": f"台账不可用：{exc}"}
    return {
        "total": total,
        "deleted": deleted,
        "recent7d": recent,
        "byKindStatus": by_kind_status,
        "byScope": by_scope,
        "db": db_path().as_posix(),
        "available": True,
    }


# ---------------------------------------------------------------------------
# 管理面：编辑 / 删除 / 恢复 / 版本 / 标签 / 审计
# ---------------------------------------------------------------------------
#
# 与上面的检索面分开写，因为它们的**失败语义不同**：检索失败可以降级（回空 + hint），
# 写操作失败必须显式回 ``ok=False`` + 原因，绝不能静默成功 —— 一次"看起来保存了"
# 的编辑，创作者会以为改到了，直到下次读到旧内容才发现。

def update_entry(entry: str, *, title: str | None = None, body: str | None = None,
                 tags: Any = None, evidence: Any = None, confidence: float | None = None,
                 reason: str = "", expected_version: int | None = None,
                 operator: str = "system", role: str = "") -> dict[str, Any]:
    """编辑一条知识（**不叠加 support**）。

    与 ``upsert`` 的分工必须写清，否则两条路会互相打架：
    * ``upsert`` ＝「又有一条证据支持它」⇒ ``support+1``，正文通常不变；
    * ``update_entry`` ＝「这条写错了/要补充」⇒ 正文变、``support`` 不动、``version+1``。
    把编辑做成 ``upsert`` 会让"改了三次错别字"看起来像"有三份独立证据"，那是灌水。

    ``expected_version`` 是乐观锁：并发编辑时后手必须失败（409），
    否则两个人各改一半，最后一个人的正文会把前一个人的整段覆盖掉且无痕。
    """
    fields: list[str] = []
    params: list[Any] = []
    changed: list[str] = []
    if title is not None:
        title = _text(title)
        if not title:
            return {"ok": False, "code": "invalid",
                    "why": "标题不能为空（要清空标题请删除条目）"}
        fields.append("title = ?")
        params.append(title)
        changed.append("title")
    if body is not None:
        body = _text(body)
        if not body:
            return {"ok": False, "code": "invalid",
                    "why": "正文不能为空（要废弃请改成 failure 或删除）"}
        fields.append("body = ?")
        params.append(body)
        changed.append("body")
    if tags is not None:
        fields.append("tags = ?")
        params.append(normalize_tags(tags))
        changed.append("tags")
    if confidence is not None:
        fields.append("confidence = ?")
        params.append(max(0.0, min(float(confidence), 1.0)))
        changed.append("confidence")
    evidence_items: list[str] | None = None
    if evidence is not None:
        evidence_items = _split_evidence(evidence)
        if evidence_items:
            checked = check_evidence(evidence_items)
            if not checked["ok"]:
                return {"ok": False, "code": "evidence",
                        "why": "证据核不过（这些 id 既不是索引里的 run，"
                               "也不是盘上的文件）："
                               + ", ".join(checked["bad"][:5])}
        fields.append("evidence = ?")
        params.append(",".join(evidence_items))
        changed.append("evidence")
    if not fields:
        return {"ok": False, "code": "invalid",
                "why": "没有要改的字段（title/body/tags/evidence/confidence 至少给一个）"}

    try:
        with _connect() as conn:
            row = conn.execute("SELECT * FROM entries WHERE id = ?", [entry]).fetchone()
            if row is None:
                return {"ok": False, "code": "not_found", "why": "条目不存在"}
            if int(_col(row, "deleted_ns", 0) or 0):
                return {"ok": False, "code": "deleted", "why": "条目已删除 ⇒ 先 restore 再编辑"}
            current = int(_col(row, "version", 1) or 1)
            if expected_version is not None and int(expected_version) != current:
                return {"ok": False, "code": "conflict", "conflict": True, "version": current,
                        "why": f"版本冲突：请求基于 v{int(expected_version)}，"
                               f"当前已是 v{current}（请重新读取后再改）"}
            _snapshot(conn, row, editor=operator, reason=_text(reason))
            fields.append("version = ?")
            params.append(current + 1)
            fields.append("updated_ns = ?")
            params.append(_now_ns())
            conn.execute(f"UPDATE entries SET {', '.join(fields)} WHERE id = ?",
                         [*params, entry])
            fresh = conn.execute("SELECT * FROM entries WHERE id = ?", [entry]).fetchone()
            _sync_fts(conn, {"id": entry, "title": fresh["title"], "body": fresh["body"],
                             "tags": fresh["tags"] or ""})
            _audit(conn, entry, "update", operator=operator, role=role,
                   detail=f"v{current}->v{current + 1}" + (f" · {','.join(changed)}" if changed else ""))
            item = _row_to_entry(fresh)
    except sqlite3.Error as exc:
        return {"ok": False, "code": "unavailable", "why": f"台账不可写：{exc}"}
    return {"ok": True, "id": entry, "version": item["version"], "entry": item,
            "changed": changed, "why": _text(reason)}


def soft_delete(entry: str, *, reason: str = "", operator: str = "system",
                role: str = "") -> dict[str, Any]:
    """软删（可撤销）。重复删幂等：``changed=False`` 而不是报错。"""
    try:
        with _connect() as conn:
            row = conn.execute("SELECT id, deleted_ns FROM entries WHERE id = ?",
                               [entry]).fetchone()
            if row is None:
                return {"ok": False, "code": "not_found", "why": "条目不存在"}
            if int(_col(row, "deleted_ns", 0) or 0):
                return {"ok": True, "id": entry, "changed": False, "why": "已经是删除状态"}
            conn.execute("UPDATE entries SET deleted_ns = ?, deleted_by = ? WHERE id = ?",
                         [_now_ns(), _text(operator), entry])
            _audit(conn, entry, "delete", operator=operator, role=role,
                   detail=_text(reason) or "soft")
    except sqlite3.Error as exc:
        return {"ok": False, "code": "unavailable", "why": f"台账不可写：{exc}"}
    return {"ok": True, "id": entry, "changed": True, "hard": False, "why": _text(reason)}


def restore(entry: str, *, reason: str = "", operator: str = "system",
            role: str = "") -> dict[str, Any]:
    """撤销软删。重复恢复幂等。"""
    try:
        with _connect() as conn:
            row = conn.execute("SELECT id, deleted_ns FROM entries WHERE id = ?",
                               [entry]).fetchone()
            if row is None:
                return {"ok": False, "code": "not_found", "why": "条目不存在"}
            if not int(_col(row, "deleted_ns", 0) or 0):
                return {"ok": True, "id": entry, "changed": False, "why": "本来就没删"}
            conn.execute("UPDATE entries SET deleted_ns = 0, deleted_by = '' WHERE id = ?",
                         [entry])
            _audit(conn, entry, "restore", operator=operator, role=role,
                   detail=_text(reason) or "restore")
    except sqlite3.Error as exc:
        return {"ok": False, "code": "unavailable", "why": f"台账不可写：{exc}"}
    return {"ok": True, "id": entry, "changed": True, "why": _text(reason)}


def purge(entry: str, *, operator: str = "system", role: str = "") -> dict[str, Any]:
    """硬删（物理删除 entries / links / usage / FTS / 版本）。

    只对**已软删**的条目开放：这是防手滑的第二道闸 —— ``DELETE ?hard=true``
    打错一个参数就把一条被引用过几十次的成法连同行用历史一起蒸发，
    那种损失在"操作有审计"的当下也补不回来。审计表**不删**（那是账，不是内容）。
    """
    try:
        with _connect() as conn:
            row = conn.execute("SELECT id, deleted_ns, title FROM entries WHERE id = ?",
                               [entry]).fetchone()
            if row is None:
                return {"ok": False, "code": "not_found", "why": "条目不存在"}
            if not int(_col(row, "deleted_ns", 0) or 0):
                return {"ok": False, "code": "not_deleted",
                        "why": "硬删前必须先软删（这是防手滑的第二道闸）"}
            counts: dict[str, int] = {}
            for table in ("entry_links", "entry_usage", "entry_versions"):
                counts[table] = int(conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE entry_id = ?", [entry]).fetchone()[0])
                conn.execute(f"DELETE FROM {table} WHERE entry_id = ?", [entry])
            #: 审计**不删**（那是账不是内容）：删掉之后"谁在什么时候删了什么"就断了
            counts["entry_audit"] = int(conn.execute(
                "SELECT COUNT(*) FROM entry_audit WHERE entry_id = ?", [entry]).fetchone()[0])
            if _has_fts(conn):
                try:
                    conn.execute("DELETE FROM entries_fts WHERE entry_id = ?", [entry])
                except sqlite3.Error:
                    pass
            conn.execute("DELETE FROM entries WHERE id = ?", [entry])
            _audit(conn, entry, "purge", operator=operator, role=role,
                   detail=f"删前 title={row['title']} · " + ",".join(
                       f"{k}={v}" for k, v in counts.items()))
    except sqlite3.Error as exc:
        return {"ok": False, "code": "unavailable", "why": f"台账不可写：{exc}"}
    return {"ok": True, "id": entry, "purged": True, "counts": counts}


def versions(entry: str) -> dict[str, Any]:
    """版本列表（只有元信息，**不含正文**：一页十条正文会把响应撑到几十 KB）。"""
    try:
        with _connect() as conn:
            row = conn.execute("SELECT version, deleted_ns FROM entries WHERE id = ?",
                               [entry]).fetchone()
            if row is None:
                return {"id": entry, "current": 0, "items": [], "available": True,
                        "deleted": False}
            current = int(_col(row, "version", 1) or 1)
            rows = conn.execute(
                "SELECT version, kind, scope, title, tags, confidence, status, editor,"
                " reason, created_ns, LENGTH(body) AS bodyLen"
                " FROM entry_versions WHERE entry_id = ? ORDER BY version DESC",
                [entry]).fetchall()
    except sqlite3.Error as exc:
        return {"id": entry, "current": 0, "items": [], "available": False,
                "hint": f"台账不可用：{exc}"}
    return {
        "id": entry, "current": current,
        "deleted": bool(int(_col(row, "deleted_ns", 0) or 0)),
        "items": [
            {"version": int(r["version"]), "title": r["title"], "kind": r["kind"],
             "scope": r["scope"], "tags": tag_list(r["tags"] or ""),
             "confidence": float(r["confidence"] or 0), "status": r["status"],
             "editor": r["editor"] or "", "reason": r["reason"] or "",
             "bodyLen": int(r["bodyLen"] or 0), "createdNs": int(r["created_ns"] or 0)}
            for r in rows
        ],
        "available": True,
    }


def version_detail(entry: str, version: int) -> dict[str, Any] | None:
    """单个历史版本（含正文）。当前版本从 ``entries`` 读，历史版本从 ``entry_versions`` 读。

    ★两个来源**必须收成同一个形状**：早先这里当前版直接回 ``_row_to_entry``（snake 列名
    + ``tagList``），历史版直接回 ``entry_versions`` 的原始行（``created_ns``、``tags`` 是串），
    于是同一个「版本详情」端点按 v 等于当前与否吐出两套键名 —— 前端必然只有一条路能跑通，
    另一条静默取到 undefined。现在两条都走这个 canonical 结构（camel 派生字段）。
    """
    try:
        with _connect() as conn:
            row = conn.execute("SELECT * FROM entries WHERE id = ?", [entry]).fetchone()
            if row is None:
                return None
            if int(_col(row, "version", 1) or 1) == int(version):
                current = _row_to_entry(row)
                body = current.get("body") or ""
                return {
                    "id": entry, "version": int(version),
                    "kind": current.get("kind") or "", "scope": current.get("scope") or "",
                    "title": current.get("title") or "", "body": body,
                    "bodyLen": len(body), "tags": current.get("tagList") or [],
                    "evidence": current.get("evidence") or [],
                    "confidence": float(current.get("confidence") or 0),
                    "status": current.get("status") or "",
                    "editor": "", "reason": "",
                    "createdNs": int(current.get("updated_ns") or 0),
                    "isCurrent": True,
                }
            old = conn.execute(
                "SELECT * FROM entry_versions WHERE entry_id = ? AND version = ?",
                [entry, int(version)]).fetchone()
            if old is None:
                return None
    except sqlite3.Error:
        return None
    body = _col(old, "body", "") or ""
    return {
        "id": entry, "version": int(_col(old, "version", 0) or 0),
        "kind": _col(old, "kind", "") or "", "scope": _col(old, "scope", "") or "",
        "title": _col(old, "title", "") or "", "body": body,
        "bodyLen": len(body), "tags": tag_list(_col(old, "tags", "") or ""),
        "evidence": [x for x in str(_col(old, "evidence", "") or "").split(",") if x],
        "confidence": float(_col(old, "confidence", 0) or 0),
        "status": _col(old, "status", "") or "",
        "editor": _col(old, "editor", "") or "", "reason": _col(old, "reason", "") or "",
        "createdNs": int(_col(old, "created_ns", 0) or 0),
        "isCurrent": False,
    }


def revert(entry: str, version: int, *, reason: str = "", operator: str = "system",
           role: str = "") -> dict[str, Any]:
    """回滚到某个历史版本。

    回滚**不删除**之后的版本（删了就没法"再滚回改前"了），它只是一次普通编辑：
    把旧值写进正文 ⇒ 走 ``update_entry`` ⇒ 自动 ``version+1`` + 快照 + 审计。
    """
    target = version_detail(entry, version)
    if target is None:
        return {"ok": False, "code": "not_found", "why": f"{entry} 没有 v{version}"}
    if target.get("isCurrent"):
        return {"ok": True, "id": entry, "changed": False, "why": "已经是这个版本"}
    result = update_entry(
        entry, body=target.get("body") or "", tags=target.get("tags") or "",
        reason=f"revert to v{version}" + (f" · {_text(reason)}" if _text(reason) else ""),
        operator=operator, role=role,
    )
    if result.get("ok"):
        _audit_follow(entry, "revert", operator=operator, role=role, detail=f"to v{version}")
        result["changed"] = True
    return result


def _audit_follow(entry: str, action: str, *, operator: str = "", role: str = "",
                  detail: str = "") -> None:
    """补一条审计（``revert`` 借用 ``update_entry`` 后要说明"这一版是回滚来的"）。"""
    try:
        with _connect() as conn:
            _audit(conn, entry, action, operator=operator, role=role, detail=detail)
    except sqlite3.Error:
        pass


def tag_facets(*, include_deleted: bool = False, limit: int = 200) -> dict[str, Any]:
    """标签 facet（每个标签有多少条）。"""
    clauses = [] if include_deleted else ["COALESCE(deleted_ns, 0) = 0"]
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    try:
        with _connect() as conn:
            rows = conn.execute(f"SELECT tags FROM entries{where}").fetchall()
    except sqlite3.Error as exc:
        return {"items": [], "total": 0, "available": False, "hint": f"台账不可用：{exc}"}
    counter: dict[str, int] = {}
    for row in rows:
        for name in tag_list(row["tags"] or ""):
            counter[name] = counter.get(name, 0) + 1
    items = sorted(({"tag": name, "count": n} for name, n in counter.items()),
                   key=lambda item: (-item["count"], item["tag"]))
    return {"items": items[: max(1, int(limit))], "total": len(counter), "available": True}


def taxonomy(*, include_deleted: bool = False) -> dict[str, Any]:
    """分类导航：``kind``（闭集，带中文名与计数）＋ ``scope``（半开集）＋ 状态 ＋ 标签。

    这是「分类管理」的**唯一入口**：前端不该自己写死中文名，否则服务端加一类，
    前端就多出一串英文 id。
    """
    alive = "" if include_deleted else " WHERE COALESCE(deleted_ns, 0) = 0"
    try:
        with _connect() as conn:
            kind_counts = {str(r["kind"]): int(r["n"]) for r in conn.execute(
                f"SELECT kind, COUNT(*) AS n FROM entries{alive} GROUP BY kind")}
            scope_counts = [
                {"scope": r["scope"], "count": r["n"]} for r in conn.execute(
                    f"SELECT scope, COUNT(*) AS n FROM entries{alive}"
                    " GROUP BY scope ORDER BY n DESC")
            ]
            status_counts = {str(r["status"]): int(r["n"]) for r in conn.execute(
                f"SELECT status, COUNT(*) AS n FROM entries{alive} GROUP BY status")}
    except sqlite3.Error as exc:
        return {"kinds": [], "scopes": [], "statuses": [], "tags": [], "available": False,
                "hint": f"台账不可用：{exc}"}
    facets = tag_facets(include_deleted=include_deleted)
    return {
        "kinds": [
            {"kind": kind, "label": KIND_LABELS.get(kind, kind), "count": kind_counts.get(kind, 0)}
            for kind in KINDS
        ],
        "scopes": scope_counts,
        "statuses": [{"status": name, "count": status_counts.get(name, 0)}
                     for name in STATUSES],
        "tags": facets["items"],
        "tagTotal": facets["total"],
        "available": True,
    }


def audit_log(*, entry: str | None = None, action: str | None = None,
              operator: str | None = None, limit: int = 50,
              offset: int = 0) -> dict[str, Any]:
    """操作审计（倒序）。"""
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset))
    clauses: list[str] = []
    params: list[Any] = []
    if entry:
        clauses.append("entry_id = ?")
        params.append(_text(entry))
    if action:
        clauses.append("action = ?")
        params.append(_text(action))
    if operator:
        clauses.append("operator = ?")
        params.append(_text(operator))
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    try:
        with _connect() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) FROM entry_audit{where}",
                                     params).fetchone()[0])
            rows = conn.execute(
                f"SELECT id, entry_id, action, operator, role, at_ns, detail"
                f" FROM entry_audit{where} ORDER BY at_ns DESC, id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset]).fetchall()
    except sqlite3.Error as exc:
        return {"total": 0, "items": [], "available": False, "hint": f"台账不可用：{exc}"}
    return {
        "total": total,
        "items": [
            {"id": int(r["id"]), "entryId": r["entry_id"], "action": r["action"],
             "operator": r["operator"] or "", "role": r["role"] or "",
             "atNs": int(r["at_ns"] or 0), "detail": r["detail"] or ""}
            for r in rows
        ],
        "limit": limit, "offset": offset, "available": True,
    }
