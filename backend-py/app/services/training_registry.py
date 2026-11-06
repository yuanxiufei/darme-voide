"""训练产物登记表（P4）—— 让「跑过一次训练」这件事在重启之后仍然成立。

要解决的问题
============
:class:`app.services.lora_train.runtime.LoraTrainRuntime` 的任务表是**进程内存**里的
``dict``（``self._tasks`` + ``self._order``）：进程一重启，``GET /lora-train/tasks``
就返回空表，而 ``data/tasks/<taskId>/`` 里的日志、``train.log``、产出的
``*.safetensors`` **全都还在盘上**。于是最常见的两件事都办不到：

1. 「上周那个 30 张图训出来的 LoRA 是用什么参数训的？」——参数只存在于已经消失的内存里；
2. 「现在该用哪个 LoRA？」——没有任何地方记得「这个文件是候选还是已经定为默认」。

本模块只做一件事：把**任务与产物落成一张可查的表**（``training_jobs`` /
``trained_models``），并且坚持三条：

* **记录失败不许挡住训练**：所有写口返回 ``bool`` / ``dict``，不抛（同
  ``outcome_index.record_run`` 的口径 —— 少一条登记不该让一次真训练白跑）；
* **候选不许自动成默认**：扫描出来的产物一律 ``status="candidate"``、``is_default=0``；
  变成 ``accepted`` 必须由 :func:`promote` 显式做，且要 ``basis``（为什么）与
  可核的 ``evidence``（凭据）。**这是"模型投喂"整条线唯一能改变出图行为的地方**，
  静默自动提升的后果是「某天出图风格忽然变了，没人知道是谁改的」；
* **盘上不见了就说不见了**：``missing`` / ``exists=0`` 照记，不因为文件被删就把记录也删掉。

数据来路（``dataset_json``）
===========================
任务与产物两张表都留一格存 ``datasetRef`` 的**核验结果**（``knowledge_feed.resolve_dataset``
那份事实：``ref`` / ``fingerprint`` / ``itemsPath`` / ``count`` / ``roles`` / ``rights``）。
它回答的是「这个 LoRA 是拿哪批知识训的」。产物也存一份，是因为人通常从产物反查，
不会先记得 ``taskId``；而样本文件被清掉时 :func:`problems` 会点名 —— 那时血统已经断了。

为什么单独一个 sqlite
=====================
与 ``outcome_index`` / ``scene_index`` 同一条规矩：**A 侧可重建、B 侧不可重建**。
登记表是派生视图（盘上目录 + 任务事实），放进 ``darme.db`` 会被算进备份体积、
被 ``db_upgrade`` 当成项目结构补列。落点 ``<数据根>/knowledge/training-registry.sqlite``，
``TRAINING_REGISTRY_DB`` 可整体改指向（测试夹具用它隔离）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.core.config import get_data_root

from . import knowledge_ledger

__all__ = [
    "MODEL_STATUSES",
    "JOB_STATUSES",
    "db_path",
    "get_job",
    "list_jobs",
    "list_models",
    "promote",
    "demote",
    "problems",
    "record_artifact",
    "record_job",
    "scan_artifacts",
    "stats",
]

#: 任务状态的**同一套**取值（与 ``lora_train.runtime.TERMINAL`` 对齐，另加 ``queued``）
JOB_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")

#: 产物状态：``candidate`` → ``accepted``（人工）→ ``deprecated``（淘汰但仍留档）
MODEL_STATUSES = ("candidate", "accepted", "deprecated")

#: 摘要上限（bytes）：超过就只记大小不记 sha1 —— 登记表不该为了算摘要而拖住收尾
_HASH_LIMIT = 512 * 1024 * 1024

#: 一条 ``running`` 记录超过这么久还是 ``running`` ⇒ 判定为「进程死掉的僵尸」
_STALE_RUNNING_MS = 24 * 3600 * 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_jobs (
    task_id TEXT PRIMARY KEY,
    model_key TEXT,
    status TEXT,
    stage TEXT,
    error TEXT,
    workspace TEXT,
    log_path TEXT,
    created_ms INTEGER,
    started_ms INTEGER,
    finished_ms INTEGER,
    values_json TEXT,
    plan_json TEXT,
    dataset_json TEXT,
    recorded_ns INTEGER,
    updated_ns INTEGER
);
CREATE INDEX IF NOT EXISTS jobs_by_key ON training_jobs(model_key, created_ms DESC);
CREATE INDEX IF NOT EXISTS jobs_by_status ON training_jobs(status, created_ms DESC);
CREATE TABLE IF NOT EXISTS trained_models (
    model_id TEXT PRIMARY KEY,
    task_id TEXT,
    model_key TEXT,
    kind TEXT,
    path TEXT,
    exists_now INTEGER,
    size_bytes INTEGER,
    sha1 TEXT,
    status TEXT,
    is_default INTEGER,
    note TEXT,
    metrics_json TEXT,
    dataset_json TEXT,
    basis TEXT,
    evidence_json TEXT,
    created_ns INTEGER,
    checked_ns INTEGER
);
CREATE INDEX IF NOT EXISTS models_by_status ON trained_models(status, kind);
"""

#: 后加的列：``CREATE TABLE IF NOT EXISTS`` **不会**改已有表的结构 ⇒ 老库（已经存过
#: 几次训练的）要单独补列。不补的后果不是报错，而是「登记表里查不到 dataset」——
#: 一个看起来正常的空字段，比缺列更难发现。
_ADDED_COLUMNS = (
    ("training_jobs", "dataset_json", "TEXT"),
    ("trained_models", "dataset_json", "TEXT"),
)


def db_path() -> Path:
    """登记库落点（``TRAINING_REGISTRY_DB`` 可覆盖）。

    ⚠️ **每次调用现算**，不做模块级常量：数据根可在运行时切换
    （``config.set_data_root``），常量会把登记表钉在旧根上 ——
    症状是「明明训过的模型，重启后列表是空的」。
    """
    override = str(os.environ.get("TRAINING_REGISTRY_DB") or "").strip()
    if override:
        return Path(override)
    return Path(get_data_root()) / "knowledge" / "training-registry.sqlite"


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path.as_posix(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _ensure_columns(conn)
    return conn


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """给老库补 :data:`_ADDED_COLUMNS` 里的列（已存在就跳过，每次都调、幂等）。"""
    for table, column, decl in _ADDED_COLUMNS:
        names = {str(row["name"]) for row in
                 conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in names:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    conn.commit()


def _text(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value).strip()
    return out or None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _field(job: Any, name: str, default: Any = None) -> Any:
    """读任务对象或 dict 的同名字段（两种形状都收 —— 调用方不用先转 dict）。"""
    if isinstance(job, Mapping):
        return job.get(name, default)
    return getattr(job, name, default)


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return "{}"


def _loads(text: Any, fallback: Any) -> Any:
    if not text:
        return fallback
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return fallback


def _dataset_of(job: Any) -> str | None:
    """取出这次训练用的数据集引用（``dataset`` 块 / 纯字符串 ``datasetRef`` 都收）。

    纯字符串也照记 —— 那是调用方没经 :func:`knowledge_feed.resolve_dataset` 核过的
    引用，记下来仍然比丢掉有用（:func:`problems` 只对**没有 itemsPath 的**引用免检，
    有路径的照样要核在不在盘上）。
    """
    value = (_field(job, "dataset") or _field(job, "datasetRef")
             or _field(job, "dataset_ref"))
    if isinstance(value, Mapping):
        return _dump(dict(value)) if value else None
    text = _text(value)
    return _dump({"ref": text}) if text else None


# ---------------------------------------------------------------------------
# 任务
# ---------------------------------------------------------------------------

def record_job(job: Any) -> bool:
    """把一次训练任务的**当前状态**写进登记表（按 ``task_id`` 幂等 upsert）。

    ``job`` 可以是 :class:`LoraTrainTask` 本体，也可以是 ``to_summary()`` /
    ``to_dict()`` 的字典。字段缺就留空，**不猜**（``createdAt`` 缺失不写 0 ——
    0 会被排序当成"1970 年"，比空更难看出问题）。

    返回 ``False`` 表示没写进去（库坏/权限），**不抛** —— 见模块头第 1 条。
    """
    task_id = _text(_field(job, "id") or _field(job, "task_id") or _field(job, "taskId"))
    if not task_id:
        return False
    now = time.time_ns()
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO training_jobs (task_id, model_key, status, stage, error,
                    workspace, log_path, created_ms, started_ms, finished_ms,
                    values_json, plan_json, dataset_json, recorded_ns, updated_ns)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET
                    model_key=excluded.model_key,
                    status=excluded.status,
                    stage=excluded.stage,
                    error=excluded.error,
                    workspace=excluded.workspace,
                    log_path=excluded.log_path,
                    started_ms=COALESCE(excluded.started_ms, training_jobs.started_ms),
                    finished_ms=COALESCE(excluded.finished_ms, training_jobs.finished_ms),
                    values_json=excluded.values_json,
                    plan_json=excluded.plan_json,
                    dataset_json=COALESCE(excluded.dataset_json, training_jobs.dataset_json),
                    updated_ns=excluded.updated_ns
                """,
                (
                    task_id,
                    _text(_field(job, "modelKey") or _field(job, "model_key")),
                    _text(_field(job, "status")),
                    _text(_field(job, "stage")),
                    _text(_field(job, "error")),
                    _text(_field(job, "workspace")),
                    _text(_field(job, "logPath") or _field(job, "log_path")),
                    _int(_field(job, "createdAt") or _field(job, "created_ms")),
                    _int(_field(job, "startedAt") or _field(job, "started_ms")),
                    _int(_field(job, "finishedAt") or _field(job, "finished_ms")),
                    _dump(_field(job, "values") or {}),
                    _dump(_field(job, "plan") or {}),
                    _dataset_of(job),
                    now,
                    now,
                ),
            )
            conn.commit()
        return True
    except sqlite3.Error:
        return False


def _job_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "taskId": row["task_id"],
        "modelKey": row["model_key"],
        "status": row["status"],
        "stage": row["stage"],
        "error": row["error"],
        "workspace": row["workspace"],
        "logPath": row["log_path"],
        "createdAt": row["created_ms"],
        "startedAt": row["started_ms"],
        "finishedAt": row["finished_ms"],
        "values": _loads(row["values_json"], {}),
        "plan": _loads(row["plan_json"], {}),
        #: 这次训练用的是哪批数据（``knowledge_feed.resolve_dataset`` 核过的那份事实）
        "dataset": _loads(row["dataset_json"], None),
        "recordedNs": row["recorded_ns"],
        "updatedNs": row["updated_ns"],
        #: ⭐ 这一行来自**盘上登记表**而不是进程内存 ⇒ 前端/调用方一眼能看出
        #: 「这是重启前跑的那次」，不会误以为是当前在跑的
        "restored": True,
    }


def get_job(task_id: str) -> dict[str, Any] | None:
    """取一条登记（不认识 ⇒ ``None``，不抛）。"""
    key = _text(task_id)
    if not key:
        return None
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM training_jobs WHERE task_id = ?", (key,)).fetchone()
    except sqlite3.Error:
        return None
    return _job_row(row) if row is not None else None


def list_jobs(*, limit: int = 20, model_key: str | None = None,
              status: str | None = None, skip_ids: Iterable[str] = ()) -> list[dict[str, Any]]:
    """按提交时间倒序列出登记。``skip_ids`` 用来滤掉「内存里已有的」（合并视图用）。"""
    sql = "SELECT * FROM training_jobs"
    where: list[str] = []
    args: list[Any] = []
    if model_key:
        where.append("model_key = ?")
        args.append(model_key)
    if status:
        where.append("status = ?")
        args.append(status)
    skip = [str(item) for item in skip_ids if str(item).strip()]
    if skip:
        where.append(f"task_id NOT IN ({','.join('?' * len(skip))})")
        args.extend(skip)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY COALESCE(created_ms, 0) DESC, task_id DESC LIMIT ?"
    args.append(max(0, int(limit)))
    try:
        with _connect() as conn:
            rows = conn.execute(sql, tuple(args)).fetchall()
    except sqlite3.Error:
        return []
    return [_job_row(row) for row in rows]


# ---------------------------------------------------------------------------
# 产物
# ---------------------------------------------------------------------------

def _sha1(path: Path) -> str | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > _HASH_LIMIT:
        return None
    digest = hashlib.sha1()  # noqa: S324 —— 只做去重指纹，不做安全校验
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def record_artifact(task_id: str, path: str | Path, *, kind: str = "lora",
                    model_key: str | None = None, note: str = "",
                    metrics: Any = None, dataset: Any = None) -> dict[str, Any]:
    """登记一个产物文件。

    **文件不存在也照记**（``existsNow=0``）—— 记录的是"这次训练声称产出了什么"，
    不是"文件现在还在不在"。把不存在的直接拒掉，会让「盘被清过」这件事
    彻底失去痕迹（正是 :func:`problems` 要报的那一类）。

    ``dataset`` 是这次训练的数据集引用，**跟着产物一起存**（而不是只挂在任务上）：
    三个月后问「这个 LoRA 拿什么训的」，人通常是从产物反查，不会先记得 taskId。
    """
    target = Path(str(path))
    exists = target.is_file()
    size = target.stat().st_size if exists else None
    model_id = f"{kind}:{hashlib.sha1(str(target).encode('utf-8')).hexdigest()[:16]}"  # noqa: S324
    now = time.time_ns()
    payload = (
        model_id, _text(task_id), _text(model_key), _text(kind) or "lora",
        target.as_posix(), 1 if exists else 0, size, _sha1(target) if exists else None,
        "candidate", 0, note or None, _dump(metrics or {}),
        _dataset_of({"dataset": dataset}), now, now,
    )
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO trained_models (model_id, task_id, model_key, kind, path,
                    exists_now, size_bytes, sha1, status, is_default, note,
                    metrics_json, dataset_json, created_ns, checked_ns)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(model_id) DO UPDATE SET
                    task_id=excluded.task_id,
                    model_key=COALESCE(excluded.model_key, trained_models.model_key),
                    exists_now=excluded.exists_now,
                    size_bytes=excluded.size_bytes,
                    sha1=COALESCE(excluded.sha1, trained_models.sha1),
                    metrics_json=excluded.metrics_json,
                    dataset_json=COALESCE(excluded.dataset_json, trained_models.dataset_json),
                    checked_ns=excluded.checked_ns
                """,
                payload,
            )
            conn.commit()
    except sqlite3.Error as err:
        return {"ok": False, "why": f"写不进去：{err}"}
    #: ⚠️ ``status``/``is_default`` **不在冲突更新列里**：重新扫一次盘不许把
    #: 已经人工 ``accepted`` 的产物打回 ``candidate``，也不许把默认档撤掉。
    return {"ok": True, "modelId": model_id, "path": target.as_posix(),
            "exists": exists, "sha1": payload[7], "status": "candidate"}


def scan_artifacts(job: Any) -> list[dict[str, Any]]:
    """扫一次任务的产物目录 ⇒ 登记所有 ``*.safetensors``。

    目录来源只认两个**真实字段**：``values["output_dir"]``（``commands.build_plan``
    写进训练命令的那个）与 ``workspace``（日志/中间件所在）。**不做正则解析 argv** ——
    命令格式一改，正则就会静默扫错目录、还报"0 个产物"。
    """
    #: ⚠️ 三种叫法都要认：运行时任务对象是 ``id``、登记行是 ``taskId``、
    #: 直接读 sqlite 行的人会传 ``task_id``。只认一个 ⇒ 从 ``get_job()`` 的结果
    #: 反查时**永远扫出 0 个**（不报错，看起来就是"这次训练没产出"）。
    task_id = _text(_field(job, "id") or _field(job, "taskId") or _field(job, "task_id"))
    if not task_id:
        return []
    values = _field(job, "values") or {}
    model_key = _text(_field(job, "modelKey") or _field(job, "model_key"))
    dataset = _field(job, "dataset") or _field(job, "datasetRef")
    roots: list[Path] = []
    for raw in (values.get("output_dir") if isinstance(values, Mapping) else None,
                _field(job, "workspace") or _field(job, "logPath")):
        text = _text(raw)
        if not text:
            continue
        candidate = Path(text)
        if candidate.suffix:  # 给了 logPath 这种文件路径 ⇒ 退到它所在目录
            candidate = candidate.parent
        if candidate not in roots:
            roots.append(candidate)
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for file in sorted(root.rglob("*.safetensors")):
            key = file.as_posix()
            if key in seen:
                continue
            seen.add(key)
            found.append(record_artifact(task_id, file, model_key=model_key,
                                         note=f"训练产物（扫自 {root.name}）",
                                         dataset=dataset))
    return found


def _model_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "modelId": row["model_id"],
        "taskId": row["task_id"],
        "modelKey": row["model_key"],
        "kind": row["kind"],
        "path": row["path"],
        "exists": bool(row["exists_now"]),
        "sizeBytes": row["size_bytes"],
        "sha1": row["sha1"],
        "status": row["status"],
        "isDefault": bool(row["is_default"]),
        "note": row["note"],
        "metrics": _loads(row["metrics_json"], {}),
        "dataset": _loads(row["dataset_json"], None),
        "basis": row["basis"],
        "evidence": _loads(row["evidence_json"], []),
        "createdNs": row["created_ns"],
        "checkedNs": row["checked_ns"],
    }


def list_models(*, status: str | None = None, kind: str | None = None,
                limit: int = 50) -> list[dict[str, Any]]:
    sql = "SELECT * FROM trained_models"
    where: list[str] = []
    args: list[Any] = []
    if status:
        where.append("status = ?")
        args.append(status)
    if kind:
        where.append("kind = ?")
        args.append(kind)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY is_default DESC, created_ns DESC LIMIT ?"
    args.append(max(0, int(limit)))
    try:
        with _connect() as conn:
            rows = conn.execute(sql, tuple(args)).fetchall()
    except sqlite3.Error:
        return []
    return [_model_row(row) for row in rows]


def promote(model_id: str, *, basis: str, evidence: Any = None,
            by: str = "human") -> dict[str, Any]:
    """把一个产物定为**该 kind 的默认档**（人工动作，唯一能改变出图行为的入口）。

    四条必须同时满足，缺一条就整条拒：

    1. 产物在登记表里（``model_id`` 存在）；
    2. 文件**现在还在盘上**（把已删除的文件设成默认 ⇒ 下次出图崩在装载）；
    3. 有 ``basis``（为什么用它）—— 一句话，人写的；
    4. 有可核的 ``evidence``（``knowledge_ledger.check_evidence``）—— 真实 run_id 或盘上文件。

    ``candidate`` **不会**自动变成默认：本函数之外没有任何代码路径会写
    ``is_default=1``（见模块头第 2 条）。
    """
    key = _text(model_id)
    if not key:
        return {"ok": False, "why": "缺 modelId"}
    text = str(basis or "").strip()
    if not text:
        return {"ok": False, "why": "缺 basis：定为默认必须写清「凭什么用它」"}
    items = evidence if isinstance(evidence, list) else ([evidence] if evidence else [])
    items = [str(item).strip() for item in items if str(item).strip()]
    if not items:
        return {"ok": False, "why": "缺 evidence：没有凭据的提升就是拍脑袋"}
    checked = knowledge_ledger.check_evidence(items)
    if not checked["ok"]:
        return {"ok": False, "why": f"证据核不过：{'、'.join(checked['bad'])}",
                "bad": checked["bad"]}

    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM trained_models WHERE model_id = ?", (key,)).fetchone()
            if row is None:
                return {"ok": False, "why": f"没有这个产物：{key}"}
            path = Path(str(row["path"]))
            if not path.is_file():
                return {"ok": False, "why": f"文件不在盘上，不能设为默认：{path.as_posix()}"}
            kind = str(row["kind"] or "lora")
            demoted = [
                item["model_id"] for item in conn.execute(
                    "SELECT model_id FROM trained_models "
                    "WHERE kind = ? AND is_default = 1 AND model_id != ?", (kind, key)).fetchall()
            ]
            now = time.time_ns()
            conn.execute(
                "UPDATE trained_models SET is_default = 0, status = CASE "
                "WHEN status = 'accepted' THEN 'deprecated' ELSE status END "
                "WHERE kind = ? AND model_id != ? AND is_default = 1", (kind, key))
            conn.execute(
                "UPDATE trained_models SET status = 'accepted', is_default = 1, "
                "basis = ?, evidence_json = ?, note = COALESCE(note, '') || ?, "
                "checked_ns = ? WHERE model_id = ?",
                (text, _dump(items), f"\n[${now}] {by} 定为默认", now, key))
            conn.commit()
    except sqlite3.Error as err:
        return {"ok": False, "why": f"写不进去：{err}"}

    return {"ok": True, "modelId": key, "kind": kind, "by": by,
            "demoted": demoted,
            "hint": "同 kind 的其它默认档已自动改为 deprecated（但文件不删）"}


def demote(model_id: str, *, why: str = "") -> dict[str, Any]:
    """撤掉默认档（文件不动、记录不删 —— 「不用它」不等于「它没存在过」）。"""
    key = _text(model_id)
    if not key:
        return {"ok": False, "why": "缺 modelId"}
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT is_default FROM trained_models WHERE model_id = ?", (key,)).fetchone()
            if row is None:
                return {"ok": False, "why": f"没有这个产物：{key}"}
            conn.execute(
                "UPDATE trained_models SET is_default = 0, status = 'deprecated', "
                "note = COALESCE(note, '') || ? WHERE model_id = ?",
                (f"\n[{time.time_ns()}] 撤下默认：{why.strip() or '未写原因'}", key))
            conn.commit()
    except sqlite3.Error as err:
        return {"ok": False, "why": f"写不进去：{err}"}
    return {"ok": True, "modelId": key, "wasDefault": bool(row["is_default"])}


def stats() -> dict[str, Any]:
    """一眼看清：多少任务、多少产物、几个是默认档。"""
    out: dict[str, Any] = {
        "path": db_path().as_posix(), "exists": db_path().exists(),
        "jobs": {}, "models": {}, "totalJobs": 0, "totalModels": 0,
    }
    try:
        with _connect() as conn:
            for row in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM training_jobs GROUP BY status"):
                out["jobs"][row["status"] or "unknown"] = row["n"]
                out["totalJobs"] += row["n"]
            for row in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM trained_models GROUP BY status"):
                out["models"][row["status"] or "unknown"] = row["n"]
                out["totalModels"] += row["n"]
            #: 有多少次训练**说得出**数据来源（数据集投喂闭环的覆盖率，不是"跑了几次"）
            out["jobsWithDataset"] = conn.execute(
                "SELECT COUNT(*) FROM training_jobs WHERE dataset_json IS NOT NULL"
            ).fetchone()[0]
            out["modelsWithDataset"] = conn.execute(
                "SELECT COUNT(*) FROM trained_models WHERE dataset_json IS NOT NULL"
            ).fetchone()[0]
            out["defaults"] = [
                {"modelId": item["model_id"], "kind": item["kind"], "path": item["path"]}
                for item in conn.execute(
                    "SELECT model_id, kind, path FROM trained_models WHERE is_default = 1"
                ).fetchall()
            ]
    except sqlite3.Error as err:
        out["error"] = str(err)
    return out


def problems() -> list[str]:
    """自查：把「登记自己在骗人」的几种情形逐条报出来。"""
    found: list[str] = []
    try:
        with _connect() as conn:
            models = conn.execute("SELECT * FROM trained_models").fetchall()
            jobs = conn.execute("SELECT * FROM training_jobs").fetchall()
    except sqlite3.Error as err:
        return [f"登记库读不出来：{err}"]

    defaults: dict[str, str] = {}
    for row in models:
        item = _model_row(row)
        if item["isDefault"]:
            kind = item["kind"] or "lora"
            if kind in defaults:
                found.append(f"{kind}: 有多个默认档（{defaults[kind]} 与 {item['modelId']}）")
            defaults[kind] = item["modelId"]
        if not item["path"] or not Path(str(item["path"])).is_file():
            if item["isDefault"] or item["status"] == "accepted":
                found.append(f"{item['modelId']}: 标为「{item['status']}"
                             f"{'/默认' if item['isDefault'] else ''}」但文件不在盘上：{item['path']}")
        if item["status"] == "accepted" and not str(item["basis"] or "").strip():
            found.append(f"{item['modelId']}: accepted 却没有 basis")
        if item["status"] not in MODEL_STATUSES:
            found.append(f"{item['modelId']}: 状态 {item['status']!r} 不在 {MODEL_STATUSES}")

    #: 数据集引用一旦指向不在盘上的样本文件，这条血统就断了 —— 而登记表看上去
    #: 一切正常（`dataset.ref` 还在）。和「默认档文件不在盘上」同一类谎话，必须点名。
    for row in jobs:
        dataset = _loads(row["dataset_json"], None)
        items_path = str((dataset or {}).get("itemsPath") or "") if isinstance(dataset, dict) else ""
        if items_path and not Path(items_path).is_file():
            found.append(f"{row['task_id']}: 数据集样本文件不在盘上：{items_path}"
                         f"（这次训练的来路已经没法核了）")

    now_ms = int(time.time() * 1000)
    for row in jobs:
        status = row["status"]
        if status not in JOB_STATUSES:
            found.append(f"{row['task_id']}: 任务状态 {status!r} 不在 {JOB_STATUSES}")
        if status == "running":
            started = row["started_ms"] or row["created_ms"] or 0
            if started and now_ms - int(started) > _STALE_RUNNING_MS:
                found.append(f"{row['task_id']}: 登记为 running，但已经过了 "
                             f"{(now_ms - int(started)) // 3600000} 小时 —— 进程多半已经死了"
                             f"（重启后内存任务表消失，这行永远是 running）")
    return found
