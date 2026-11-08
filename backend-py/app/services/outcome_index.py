"""运行结果索引（A 侧派生）—— 把散在各处的「一次跑次」归一成一张可查的表。

为什么要这个模块
================
本仓的反馈信号其实一直都在，但**散在四个地方、四个形状**：

* ``traces/<scope>/<id>.jsonl`` —— Agent 跑次，append-only，只能按 ``scope``+``traceId`` 取；
* ``video_quality_checks`` —— 镜头级分数与 issues，在 ``darme.db`` 里（B 侧，不可重建）；
* ``comfyui_runs`` —— 生图/生视频的步骤、耗时、错误串；
* ``api_usage`` —— 文本/图/视频调用的状态、重试、成本。

于是「这个 stage 最近失败率多高」「哪些镜头返工最多」「28 步和 30 步分数差多少」这类
**一个 SQL 就能答**的问题，目前只能靠人翻日志。知识积累、引擎自我完善、模型投喂三条线
都要读它 ⇒ 先把它做成一张表。

为什么单独一个 sqlite，而不是进 ``darme.db``
==========================================
与 ``scene_index.py`` 同一条规矩：**A 侧可重建、B 侧不可重建**。``runs`` 是纯派生
（``DELETE`` 后重跑 ``rebuild()`` 即可），放进项目库会 ① 被算进备份体积；② 被
``db_upgrade.py`` 当成项目结构去补列；③ 用户切数据根时莫名跟着跑丢。

写失败不许挡住生产
==================
``record_run`` 的返回值是 ``bool`` 而不是抛错：索引写不进去最多是「少一条统计」，
而让一次真实生成因为统计失败而失败，是拿主链路的安全换报表的好看。
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.core.config import get_data_root
from app.services.trace_store import get_traces_dir

__all__ = [
    "KINDS",
    "STATUSES",
    "db_path",
    "record_run",
    "ingest_trace",
    "rebuild",
    "query",
    "count",
    "exists",
    "get",
    "stats",
    "samples",
    "link_record",
    "record_of",
]

#: ``kind`` 的取值（新增一类跑次要先登记在这里，否则 ``record_run`` 拒收）
#:
#: ``eval``（2026-11-07 补，知识闭环缺口 8 的另一半）：**评测跑次** —— 基准 case 与成片评测
#: 都落这一类。为什么不让它借用 ``agent`` ✗：基准 case 跑的是 agent，但成片评测**根本不跑
#: agent**（它是拿已有的成片与已有的质检结论算账）；混进 ``agent`` 会让「agent 成功率」把
#: 一次没调模型的评测算进去 ⇒ 那个数字从此不可比 ✓✗。
KINDS = ("agent", "image", "video", "tts", "qc", "train", "repair", "eval")

#: ``status`` 的取值（三档不够用时要先想清楚"partial"和"failed"的区别，别再加第四档）
STATUSES = ("ok", "failed", "partial")

#: 可当 ``WHERE`` 用的列（**白名单**：动态 SQL 里只允许这些名字进来）
_WHEREABLE = (
    "kind",
    "scope",
    "drama_id",
    "episode_id",
    "storyboard_id",
    "shot_id",
    "status",
    "stage",
    "agent_type",
    "model",
    "trace_id",
)

#: 可当 ``GROUP BY`` 用的列（同上）
_GROUPABLE = (
    "kind",
    "scope",
    "stage",
    "status",
    "agent_type",
    "model",
    "drama_id",
    "episode_id",
    "storyboard_id",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    scope TEXT,
    drama_id TEXT,
    episode_id TEXT,
    storyboard_id TEXT,
    shot_id TEXT,
    agent_type TEXT,
    model TEXT,
    stage TEXT,
    status TEXT NOT NULL,
    score REAL,
    cost_amount REAL,
    is_local INTEGER,
    elapsed_ms INTEGER,
    error_code TEXT,
    trace_id TEXT,
    started_ns INTEGER,
    ended_ns INTEGER,
    meta TEXT
);
CREATE INDEX IF NOT EXISTS runs_by_episode ON runs(drama_id, episode_id, kind, started_ns);
CREATE INDEX IF NOT EXISTS runs_by_status ON runs(kind, status, started_ns);
CREATE INDEX IF NOT EXISTS runs_by_stage ON runs(stage, kind, started_ns);
CREATE TABLE IF NOT EXISTS index_state (
    file TEXT PRIMARY KEY,
    mtime_ns INTEGER,
    size INTEGER,
    run_id TEXT,
    updated_ns INTEGER
);
CREATE TABLE IF NOT EXISTS records (
    record_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    run_id TEXT,
    updated_ns INTEGER,
    PRIMARY KEY (record_type, record_id)
);
"""


def db_path() -> Path:
    """索引库落点（``OUTCOME_INDEX_DB`` 可覆盖，测试夹具也用它）。

    ⚠️ **每次调用都现算**，不做模块级常量：数据根可在运行时切换
    （``config.set_data_root``），常量会把索引钉在旧根上 —— 症状是"换了库统计归零"。
    """
    override = os.environ.get("OUTCOME_INDEX_DB")
    if override:
        return Path(override)
    return Path(get_data_root()) / "knowledge" / "outcome-index.sqlite"


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path.as_posix(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# 归一
# ---------------------------------------------------------------------------

def _text(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value).strip()
    return out or None


def _to_ns(value: Any) -> int | None:
    """时间戳归一成纳秒。

    三种输入都有真实来源：trace 里是 ISO 字符串（``now()`` 的 JS 形），
    ``api_usage`` 是秒/毫秒整数，内部调用直接给纳秒。阈值 1e11 / 1e14 是
    「秒 / 毫秒 / 纳秒」的分界 —— 按数值量级判，**不猜**语义。
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = int(value)
        if number <= 0:
            return None
        if number < 100_000_000_000:  # < 1e11 ⇒ 秒
            return number * 1_000_000_000
        if number < 100_000_000_000_000:  # < 1e14 ⇒ 毫秒
            return number * 1_000_000
        return number
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return _to_ns(int(text))
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def _norm_status(raw: Any) -> str:
    """把各家叫法归到三档。

    ⚠️ 认不出来时**原样小写返回**，不硬塞进 ``failed``：把 ``cancelled`` 悄悄算成
    失败会直接污染失败率（本仓评测里"取消"与"失败"必须能分开）。

    ⚠️ **空输入返回空串**（不是 ``partial``）：没写状态不叫"部分完成"，叫"没说"。
    兜成 ``partial`` 会让 ``total`` 涨而不进 ``failed`` —— 失败率被稀释，
    而失败率正是蒸馏第一条判据（失败聚集）的输入 ⇒ 宁可让 ``record_run`` 当场拒收。
    """
    text = (_text(raw) or "").lower()
    if text in ("ok", "success", "succeeded", "done", "completed"):
        return "ok"
    if text in ("failed", "fail", "error", "errored"):
        return "failed"
    if text in ("partial", "cancelled", "canceled", "timeout", "timedout"):
        return "partial"
    return text


def _pick(meta: dict[str, Any], *names: str) -> str | None:
    """按 camelCase / snake_case 双读（trace 的 meta 两种写法都出现过）。"""
    for name in names:
        value = _text(meta.get(name))
        if value:
            return value
    return None


def _pick_float(meta: dict[str, Any], *names: str) -> float | None:
    for name in names:
        raw = meta.get(name)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------

_COLUMNS = (
    "run_id", "kind", "scope", "drama_id", "episode_id", "storyboard_id", "shot_id",
    "agent_type", "model", "stage", "status", "score", "cost_amount", "is_local",
    "elapsed_ms", "error_code", "trace_id", "started_ns", "ended_ns", "meta",
)


def record_run(row: dict[str, Any]) -> bool:
    """写入/覆盖一条跑次（同一 ``run_id`` 后到的覆盖先到的）。

    返回 ``False`` 表示**没写成**（缺必填 / 库不可用 / SQL 错）。调用方按
    「统计少一条」处理即可，**不要**因此让主流程失败。
    """
    run_id = _text(row.get("run_id"))
    kind = _text(row.get("kind"))
    if not run_id or kind not in KINDS:
        return False
    status = _norm_status(row.get("status"))
    if not status:
        #: 没写状态 ⇒ 拒收。塞一个 ``partial`` 会让 total 涨、failed 不涨，失败率被稀释。
        return False
    #: 归一不了的状态（如 ``cancelled`` 之外的第三家叫法）**照样落库并保留原貌**：
    #: 统计按 ``status`` 分组时它会自成一组，看得见，不必塞进三档里假装知道。

    meta = row.get("meta")
    if isinstance(meta, (dict, list)):
        meta = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    elif meta is not None:
        meta = str(meta)

    values = {
        "run_id": run_id,
        "kind": kind,
        "scope": _text(row.get("scope")),
        "drama_id": _text(row.get("drama_id")),
        "episode_id": _text(row.get("episode_id")),
        "storyboard_id": _text(row.get("storyboard_id")),
        "shot_id": _text(row.get("shot_id")),
        "agent_type": _text(row.get("agent_type")),
        "model": _text(row.get("model")),
        "stage": _text(row.get("stage")),
        "status": status,
        "score": _pick_float(row, "score"),
        "cost_amount": _pick_float(row, "cost_amount", "costAmount"),
        "is_local": 1 if row.get("is_local") else 0 if row.get("is_local") is not None else None,
        "elapsed_ms": int(row["elapsed_ms"]) if _pick_float(row, "elapsed_ms") is not None else None,
        "error_code": _text(row.get("error_code")),
        "trace_id": _text(row.get("trace_id")),
        "started_ns": _to_ns(row.get("started_ns")),
        "ended_ns": _to_ns(row.get("ended_ns")),
        "meta": meta,
    }
    placeholders = ",".join("?" for _ in _COLUMNS)
    updates = ",".join(f"{name}=excluded.{name}" for name in _COLUMNS if name != "run_id")
    sql = (f"INSERT INTO runs ({','.join(_COLUMNS)}) VALUES ({placeholders}) "
           f"ON CONFLICT(run_id) DO UPDATE SET {updates}")
    try:
        with _connect() as conn:
            conn.execute(sql, [values[name] for name in _COLUMNS])
        return True
    except sqlite3.Error:
        return False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """结构保真读：逐行解析、坏行跳过（与 ``trace_store.read_trace`` 同口径）。

    这里**不复用** ``read_trace``：它按 ``safe_segment`` 反拼路径，而重建是直接从盘上
    走到的文件；两边路径归一规则一旦不同，就会出现"文件明明在、read 回来 0 条"。
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in raw.split("\n"):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def _row_from_records(scope: str, trace_id: str,
                      records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """把一份 trace 折成一行 ``runs``。"""
    if not records:
        return None

    last_error = -1
    for index, record in enumerate(records):
        if str(record.get("level") or "").upper() == "ERROR":
            last_error = index

    recovered = False
    for record in records[last_error + 1:]:
        action = str(record.get("action") or "")
        if str(record.get("level") or "").upper() == "SUCCESS" or action.startswith("DONE "):
            recovered = True
            break

    if last_error >= 0:
        status = "ok" if recovered else "failed"
    else:
        status = "ok" if any(
            str(r.get("level") or "").upper() == "SUCCESS"
            or str(r.get("action") or "").startswith("DONE ")
            for r in records
        ) else "partial"

    #: meta 从 ``START`` 那条取（那里是调用方给的入参），逐级回落
    meta: dict[str, Any] = {}
    for record in records:
        if str(record.get("action") or "").startswith("START "):
            raw = record.get("meta")
            if isinstance(raw, dict):
                meta = dict(raw)
            break
    if not meta:
        for record in records:
            raw = record.get("meta")
            if isinstance(raw, dict):
                meta = dict(raw)
                break

    error_code = None
    if last_error >= 0:
        error_code = _text(records[last_error].get("action"))
        if error_code:
            error_code = error_code[:80]

    started = _to_ns(records[0].get("ts"))
    ended = _to_ns(records[-1].get("ts"))
    elapsed = _pick_float(records[-1].get("meta") or {}, "totalMs", "elapsedMs")
    if elapsed is None:
        elapsed = _pick_float(meta, "totalMs")

    return {
        "run_id": f"trace:{scope}/{trace_id}" if scope else f"trace:{trace_id}",
        "kind": "agent",
        "scope": scope or None,
        "drama_id": _pick(meta, "dramaId", "drama_id"),
        "episode_id": _pick(meta, "episodeId", "episode_id"),
        "storyboard_id": _pick(meta, "storyboardId", "storyboard_id"),
        "shot_id": _pick(meta, "shotId", "shot_id"),
        "agent_type": _pick(meta, "agentType", "agent_type"),
        "model": _pick(meta, "model"),
        "stage": _pick(meta, "stage"),
        "status": status,
        "score": _pick_float(meta, "score"),
        "cost_amount": _pick_float(meta, "cost", "costAmount"),
        "is_local": meta.get("isLocal"),
        "elapsed_ms": int(elapsed) if elapsed is not None else None,
        "error_code": error_code,
        "trace_id": trace_id,
        "started_ns": started,
        "ended_ns": ended,
        "meta": meta or None,
    }


def ingest_trace(scope: str, trace_id: str) -> dict[str, Any] | None:
    """按 ``scope`` + ``traceId`` 抽一行（读不到返回 ``None``，并**不改库**）。"""
    from app.services.trace_store import read_trace

    records = read_trace(scope, trace_id)
    row = _row_from_records(scope, trace_id, records)
    if row is None:
        return None
    record_run(row)
    return row


def _iter_trace_files(root: Path) -> Iterable[tuple[Path, str, str]]:
    """(文件, scope, traceId) —— scope 是相对 ``traces/`` 的目录段（用 ``/`` 拼）。

    空目录（直接放在 ``traces/`` 下的 jsonl）scope 记空串，不编一个假名字。
    """
    if not root.exists():
        return
    for path in sorted(root.rglob("*.jsonl")):
        rel = path.relative_to(root)
        trace_id = path.stem
        scope = "/".join(rel.parts[:-1])
        yield path, scope, trace_id


def rebuild(*, since_ns: int | None = None, dry_run: bool = False,
            limit: int | None = None) -> dict[str, Any]:
    """扫 ``traces/`` 增量重建（按 (mtime, size) 指纹跳过未变的文件）。

    ``since_ns`` 只跳过**更早**的文件（按 mtime），用于日常增量。
    ``dry_run=True`` 时照常判断"哪些变了"，但**不写库**。
    """
    root = Path(get_traces_dir())
    scanned = changed = written = skipped = failed = 0
    seen: dict[str, tuple[int, int]] = {}

    state: dict[str, tuple[int, int, str | None]] = {}
    try:
        with _connect() as conn:
            for row in conn.execute("SELECT file, mtime_ns, size, run_id FROM index_state"):
                state[str(row["file"])] = (int(row["mtime_ns"] or 0), int(row["size"] or 0),
                                           row["run_id"])
    except sqlite3.Error:
        state = {}

    touched: list[tuple[str, tuple[int, int], str | None]] = []

    for path, scope, trace_id in _iter_trace_files(root):
        scanned += 1
        if limit is not None and scanned > limit:
            break
        try:
            stat = path.stat()
        except OSError:
            failed += 1
            continue
        if since_ns is not None and int(stat.st_mtime * 1_000_000_000) < since_ns:
            skipped += 1
            continue
        key = path.as_posix()
        fingerprint = (stat.st_mtime_ns, stat.st_size)
        previous = state.get(key)
        if previous and previous[0] == fingerprint[0] and previous[1] == fingerprint[1]:
            skipped += 1
            continue
        changed += 1
        seen[key] = fingerprint

        row = _row_from_records(scope, trace_id, _read_jsonl(path))
        if row is None:
            failed += 1
            continue
        if not dry_run and not record_run(row):
            failed += 1
            continue
        if not dry_run:
            touched.append((key, fingerprint, str(row["run_id"])))
        written += 1

    if not dry_run and touched:
        try:
            with _connect() as conn:
                conn.executemany(
                    "INSERT INTO index_state (file, mtime_ns, size, run_id, updated_ns) "
                    "VALUES (?,?,?,?,?) ON CONFLICT(file) DO UPDATE SET "
                    "mtime_ns=excluded.mtime_ns, size=excluded.size, "
                    "run_id=excluded.run_id, updated_ns=excluded.updated_ns",
                    [(key, fp[0], fp[1], run_id, _to_ns(datetime.now(timezone.utc).isoformat()))
                     for key, fp, run_id in touched],
                )
        except sqlite3.Error:
            #: 指纹没写进去只是"下次还会重算这份"，不影响已入的 runs
            pass

    return {
        "tracedir": root.as_posix(),
        "db": db_path().as_posix(),
        "scanned": scanned,
        "changed": changed,
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "dryRun": dry_run,
    }


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------

def _build_where(**kwargs: Any) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for name in _WHEREABLE:
        value = kwargs.get(name)
        if value is None or value == "":
            continue
        clauses.append(f"{name} = ?")
        params.append(str(value))
    since_ns = kwargs.get("since_ns")
    if since_ns:
        clauses.append("COALESCE(started_ns, ended_ns) >= ?")
        params.append(int(since_ns))
    until_ns = kwargs.get("until_ns")
    if until_ns:
        clauses.append("COALESCE(started_ns, ended_ns) <= ?")
        params.append(int(until_ns))
    if kwargs.get("failed_only"):
        clauses.append("status = 'failed'")
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def _row_to_item(row: sqlite3.Row) -> dict[str, Any]:
    item = {key: row[key] for key in row.keys()}
    raw = item.get("meta")
    if isinstance(raw, str) and raw:
        try:
            item["meta"] = json.loads(raw)
        except (ValueError, TypeError):
            pass
    return item


def query(*, limit: int = 100, offset: int = 0, **filters: Any) -> dict[str, Any]:
    """按剧/集/镜/阶段/状态查跑次（新到旧）。

    回执同时给 ``total`` 与 ``items``：**「命中 3 条」和「只看前 3 条」必须能分开**，
    否则前端会把"分页了"读成"就这么多"。
    """
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    where, params = _build_where(**filters)
    try:
        with _connect() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) FROM runs{where}", params).fetchone()[0])
            rows = conn.execute(
                f"SELECT * FROM runs{where} ORDER BY COALESCE(ended_ns, started_ns) DESC, rowid DESC "
                f"LIMIT ? OFFSET ?", [*params, limit, offset],
            ).fetchall()
    except sqlite3.Error as exc:
        return {"total": 0, "items": [], "limit": limit, "offset": offset,
                "available": False, "hint": f"索引库不可用：{exc}"}
    return {
        "total": total,
        "items": [_row_to_item(row) for row in rows],
        "limit": limit,
        "offset": offset,
        "available": True,
        "db": db_path().as_posix(),
    }


def count(**filters: Any) -> int:
    where, params = _build_where(**filters)
    try:
        with _connect() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM runs{where}", params).fetchone()[0])
    except sqlite3.Error:
        return 0


def exists(run_id: str) -> bool:
    """这条跑次真的在索引里吗（证据校验的唯一入口）。"""
    if not _text(run_id):
        return False
    try:
        with _connect() as conn:
            row = conn.execute("SELECT 1 FROM runs WHERE run_id = ? LIMIT 1",
                               [str(run_id)]).fetchone()
            return row is not None
    except sqlite3.Error:
        return False


def get(run_id: str) -> dict[str, Any] | None:
    """按 ``run_id`` 取**一行**（含解好的 ``meta``）；查不到 / 库坏返回 ``None``。

    为什么不复用 :func:`query`：它按时间倒序 + 白名单过滤，问的是"最近哪些跑次"，
    拿不到"指定这一条"；而质检反哺要的是「这条 QC 说的那次生成用了多少步」——
    必须按 id 直取，不能靠时间窗猜。
    """
    key = _text(run_id)
    if not key:
        return None
    try:
        with _connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ? LIMIT 1", [key]).fetchone()
    except sqlite3.Error:
        return None
    return _row_to_item(row) if row is not None else None


# ---------------------------------------------------------------------------
# 业务记录 ⇒ 生产它的那一档（质检反哺的归因键）
# ---------------------------------------------------------------------------

#: ``records.record_type`` 的取值（与业务表同名：``video_generations`` / ``images``）
RECORD_TYPES = ("video", "image")


def link_record(record_type: str, record_id: Any, stage: str, *,
                run_id: str | None = None) -> bool:
    """记下「这条业务记录是谁出的」：``(记录类型, 记录 id) ⇒ (引擎档, 引擎跑次)``。

    为什么要这张表 ✗✗：**QC 行和引擎行没有共同键** ✓——``technical_qc`` 手上只有
    ``video_generation_id`` ✓，而"用了几步 / 哪个引擎档"在 ``engine:<taskId>`` 那行上 ✓；
    ``video_generations.task_id`` 是给**云 provider** 存的轮询号 ✗（引擎这条路本来就不写它 ✓），
    所以两个方向都 join 不上 ✓✗。表由 ``bridge.run_job`` 在提交成功的那一刻写 ✓
    （那时 ``stage`` 与 ``taskId`` 都在手上 ✓）。

    返回 ``False`` ＝没写成（缺参 / 库不可用）——调用方按「这次归因不了」处理即可，
    **不要**因此让生成失败：少一条归因只是少一条统计 ✓。
    """
    kind = _text(record_type)
    rid = _text(record_id)
    name = _text(stage)
    if kind not in RECORD_TYPES or rid is None or not name:
        return False
    sql = ("INSERT INTO records (record_type, record_id, stage, run_id, updated_ns) "
           "VALUES (?,?,?,?,?) ON CONFLICT(record_type, record_id) DO UPDATE SET "
           "stage=excluded.stage, run_id=excluded.run_id, updated_ns=excluded.updated_ns")
    try:
        with _connect() as conn:
            conn.execute(sql, [kind, rid, name, _text(run_id),
                               _to_ns(datetime.now(timezone.utc).isoformat())])
        return True
    except sqlite3.Error:
        return False


def record_of(record_type: str, record_id: Any) -> dict[str, Any] | None:
    """这条业务记录是谁出的：``{"recordType","recordId","stage","runId","updatedNs"}``。

    没登记返回 ``None``（**不编一个 stage**）—— 云 provider 出的片本来就没有"引擎档"，
    硬填一格会让「这一族的质检通过率」把别人的片算进来 ✓✗。
    """
    kind = _text(record_type)
    rid = _text(record_id)
    if kind not in RECORD_TYPES or rid is None:
        return None
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT record_type, record_id, stage, run_id, updated_ns FROM records "
                "WHERE record_type = ? AND record_id = ? LIMIT 1", [kind, rid]).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return {
        "recordType": row["record_type"],
        "recordId": row["record_id"],
        "stage": row["stage"],
        "runId": row["run_id"],
        "updatedNs": row["updated_ns"],
    }


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def stats(*, group_by: str = "stage", since_ns: int | None = None,
          kind: str | None = None) -> dict[str, Any]:
    """按某列聚合：条数 / 失败率 / 平均分 / 耗时 P50·P95。

    失败率的分母是 ``total``（含 partial）—— 说"失败率"就是失败占比，
    不含 partial 的分母会算出比实际更好看的数。
    """
    if group_by not in _GROUPABLE:
        raise ValueError(f"group_by 只能是 {_GROUPABLE}，收到 {group_by!r}")
    where, params = _build_where(kind=kind, since_ns=since_ns)
    try:
        with _connect() as conn:
            rows = conn.execute(
                f"SELECT {group_by} AS key, status, score, elapsed_ms, cost_amount "
                f"FROM runs{where}", params,
            ).fetchall()
    except sqlite3.Error as exc:
        return {"groupBy": group_by, "groups": [], "total": 0,
                "available": False, "hint": f"索引库不可用：{exc}"}

    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["key"] if row["key"] is not None else "(空)")
        bucket = buckets.setdefault(key, {
            "key": key, "total": 0, "ok": 0, "failed": 0, "partial": 0,
            "other": 0, "scores": [], "elapsed": [], "cost": 0.0,
        })
        bucket["total"] += 1
        status = str(row["status"] or "")
        if status in ("ok", "failed", "partial"):
            bucket[status] += 1
        else:
            bucket["other"] += 1
        if row["score"] is not None:
            bucket["scores"].append(float(row["score"]))
        if row["elapsed_ms"] is not None:
            bucket["elapsed"].append(float(row["elapsed_ms"]))
        if row["cost_amount"] is not None:
            bucket["cost"] += float(row["cost_amount"])

    groups: list[dict[str, Any]] = []
    for bucket in buckets.values():
        scores = bucket.pop("scores")
        elapsed = bucket.pop("elapsed")
        total = bucket["total"]
        groups.append({
            **bucket,
            "failRate": round(bucket["failed"] / total, 4) if total else None,
            "avgScore": round(sum(scores) / len(scores), 4) if scores else None,
            "p50ElapsedMs": _percentile(elapsed, 0.5),
            "p95ElapsedMs": _percentile(elapsed, 0.95),
        })
    groups.sort(key=lambda g: (-g["total"], g["key"]))
    return {
        "groupBy": group_by,
        "groups": groups,
        "total": sum(g["total"] for g in groups),
        "available": True,
        "sinceNs": since_ns,
        "kind": kind,
    }


def samples(*, stage: str | None = None, kind: str | None = None,
            status: str | None = None, since_ns: int | None = None,
            limit: int = 2000) -> list[dict[str, Any]]:
    """取明细给蒸馏/参数建议用（**带 limit**：别让一次分析把全库读进内存）。"""
    where, params = _build_where(stage=stage, kind=kind, status=status, since_ns=since_ns)
    try:
        with _connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM runs{where} ORDER BY COALESCE(ended_ns, started_ns) DESC LIMIT ?",
                [*params, max(1, min(int(limit), 20000))],
            ).fetchall()
    except sqlite3.Error:
        return []
    return [_row_to_item(row) for row in rows]
