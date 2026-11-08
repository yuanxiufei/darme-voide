"""Skill 注入台账（纯派生统计）—— 「挂上去的这个 skill 有没有让跑次变好一点」。

它记什么、不记什么
==================
**记**：``load_agent_skills`` 这一次到底注入了哪些 skill、各占多少字符、哪些因预算被裁；
以及那一次 agent 跑次最后是 ``ok`` 还是 ``failed``。

**不记**：模型有没有读它、有没有照它做。本仓**没有**这条观测通路 —— 硬写一个
``hit_rate`` 就是编。所以下面只提供 ``okRate``：**注入该 skill 的那些跑次里成功的比例**。
它是**相关**不是**因果**：跑次失败可能是模型、工具、磁盘任何一处的锅，
skill 好也可能只是被挂在一个本来就会成功的 agent 上。任何看板若把它标成"命中率"或
"贡献度"都是说谎，口径与 ``knowledge_ledger._summarize_usage`` 完全一致。

它影响什么
==========
只有一处：``skills.load_agent_skills`` 在**字符预算装不下**时，用
``priority + penalty`` 决定谁先被裁（``penalty`` 只在样本足够且 ``okRate`` 明显低时非零）。
预算够时**一个字节都不受影响** —— 默认仍按 ``priority`` 全量注入。

为什么是加法不是乘法
====================
设计稿写的是 ``priority × 历史命中率``。``priority`` 是序数量纲（1/2/3），乘一个 0..1 的率
会把 priority=1 与 priority=2 压成没差别（1×0.4=0.4 vs 2×0.4=0.8，都还能被 1×0.9=0.9 顶掉，
但 1 与 2 之间的相对次序已经不是 priority 的语义了）。这里用「降级惩罚」，语义是
**「它不是被降权，而是排到"明显更好"的候选之后」**，且写进 ``why`` 供人复核。
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from app.core.config import get_data_root

__all__ = [
    "MIN_SAMPLES",
    "LOW_OK_RATE",
    "PRIORITY_PENALTY",
    "db_path",
    "record",
    "settle",
    "summary",
    "ok_rates",
    "penalty",
    "stats",
]

#: 少于这么多次**已结算**跑次就不下结论（「没有基线不说话」）
MIN_SAMPLES = 10

#: 低于它才算「明显低」；0.5 是抛硬币，不是"好/坏"的分界线
LOW_OK_RATE = 0.5

#: 降级惩罚（加在 priority 上，只在超预算裁剪时生效）
PRIORITY_PENALTY = 50

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_usage (
    id INTEGER PRIMARY KEY,
    skill_id TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    run_id TEXT,
    chars INTEGER DEFAULT 0,
    dropped INTEGER DEFAULT 0,
    used_ns INTEGER NOT NULL,
    outcome TEXT
);
CREATE INDEX IF NOT EXISTS skill_usage_by_skill ON skill_usage(skill_id, used_ns);
CREATE INDEX IF NOT EXISTS skill_usage_by_run ON skill_usage(run_id);
CREATE INDEX IF NOT EXISTS skill_usage_by_agent ON skill_usage(agent_type, used_ns);
"""


def db_path() -> Path:
    """统计库落点（``SKILL_USAGE_DB`` 可覆盖）。

    与 ``outcome_index`` / ``knowledge_ledger`` 同理：**每次现算**，不做模块常量
    （数据根可在运行时切换，比如测试把它指到 tmp）。
    """
    override = os.environ.get("SKILL_USAGE_DB")
    if override:
        return Path(override)
    return Path(get_data_root()) / "knowledge" / "skill-usage.sqlite"


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path.as_posix(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


@contextmanager
def _session() -> Iterator[sqlite3.Connection]:
    """开一个连接，**退出时 close**。

    不能直接 ``with sqlite3.connect(...) as conn`` —— 那个上下文只提交/回滚**事务**，
    连接不会关。Windows 上句柄留着会让统计库文件锁住（测试里删库直接
    ``PermissionError: WinError 32``，进程长期挂着则是句柄泄漏）。
    """
    conn = _connect()
    try:
        with conn:  # 事务（正常提交 / 异常回滚）
            yield conn
    finally:
        conn.close()


def _now_ns() -> int:
    return int(time.time_ns())


def _norm_outcome(value: Any) -> str | None:
    """把各种失败叫法归一到 ``ok``/``failed``；认不出的记 ``None``（= 还没结论）。"""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in ("ok", "success", "succeeded", "done"):
        return "ok"
    if text in ("failed", "fail", "error", "cancelled", "canceled"):
        return "failed"
    return None


@dataclass(frozen=True)
class _Item:
    skill_id: str
    chars: int = 0
    dropped: bool = False


def _items(raw: Iterable[Any]) -> list[_Item]:
    """接受 ``[{"id","chars","dropped"}]`` 或 ``["id", ...]`` 两种写法（后者按全量注入计）。"""
    out: list[_Item] = []
    for entry in raw:
        if isinstance(entry, str):
            if entry.strip():
                out.append(_Item(entry.strip()))
            continue
        if not isinstance(entry, dict):
            continue
        skill_id = entry.get("id") or entry.get("skillId") or entry.get("skill_id")
        if not skill_id:
            continue
        try:
            chars = int(entry.get("chars") or 0)
        except (TypeError, ValueError):
            chars = 0
        out.append(_Item(str(skill_id), chars, bool(entry.get("dropped"))))
    return out


def record(items: Iterable[Any], *, agent_type: str, run_id: str | None = None,
           outcome: str | None = None) -> dict[str, Any]:
    """记一次注入。

    ``run_id`` 相同 ⇒ **覆盖**（同一个跑次被记两遍不该变成两条样本）。这正是
    ``load_agent_skills`` 在同一跑次里被调用两次（``build_agent_config`` 与 ``default_generate``）
    的安全网。
    """
    parsed = _items(items)
    if not parsed:
        return {"ok": True, "recorded": 0}
    settled = _norm_outcome(outcome)
    rid = (run_id or "").strip() or None
    try:
        with _session() as conn:
            if rid:
                for item in parsed:
                    conn.execute("DELETE FROM skill_usage WHERE skill_id = ? AND run_id = ?",
                                 [item.skill_id, rid])
            for item in parsed:
                conn.execute(
                    "INSERT INTO skill_usage "
                    "(skill_id, agent_type, run_id, chars, dropped, used_ns, outcome) "
                    "VALUES (?,?,?,?,?,?,?)",
                    [item.skill_id, agent_type, rid, item.chars,
                     1 if item.dropped else 0, _now_ns(), settled],
                )
        return {"ok": True, "recorded": len(parsed)}
    except sqlite3.Error as exc:
        return {"ok": False, "recorded": 0, "hint": f"统计库不可用：{exc}"}


def settle(run_id: str, outcome: str, *, agent_type: str | None = None) -> int:
    """给某个跑次补上结果（跑次结束时调）。

    注入时刻还不知道成败 —— 所以写入时 ``outcome`` 是 ``None``，由这里回填。
    返回被回填的行数（0 表示这个跑次没注入过 skill，不是错误）。
    """
    rid = (run_id or "").strip()
    settled = _norm_outcome(outcome)
    if not rid or settled is None:
        return 0
    sql = "UPDATE skill_usage SET outcome = ? WHERE run_id = ?"
    params: list[Any] = [settled, rid]
    if agent_type:
        sql += " AND agent_type = ?"
        params.append(agent_type)
    try:
        with _session() as conn:
            return int(conn.execute(sql, params).rowcount or 0)
    except sqlite3.Error:
        return 0


def _summarize(outcomes: list[Any], *, dropped: int = 0) -> dict[str, Any]:
    ok = sum(1 for o in outcomes if o == "ok")
    failed = sum(1 for o in outcomes if o == "failed")
    settled = ok + failed
    return {
        "ok": ok,
        "failed": failed,
        "unknown": len(outcomes) - settled,
        "settled": settled,
        "samples": settled,
        "dropped": dropped,
        #: ⚠️ 「注入了它的跑次里成功的比例」，**不是** skill 的命中率／贡献度（见模块头）
        "okRate": round(ok / settled, 4) if settled else None,
    }


def summary(*, agent_type: str | None = None, min_samples: int = MIN_SAMPLES) -> list[dict[str, Any]]:
    """按 skill 汇总（只含 ``agent_type`` 匹配的行）。

    ``judged=False`` 表示**样本不足，还没有结论** —— 不是「没用」。
    """
    where, params = "", []
    if agent_type:
        where = " WHERE agent_type = ?"
        params.append(agent_type)
    try:
        with _session() as conn:
            rows = conn.execute(
                "SELECT skill_id, dropped, outcome FROM skill_usage" + where, params
            ).fetchall()
    except sqlite3.Error:
        return []
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = grouped.setdefault(row["skill_id"], {"outcomes": [], "dropped": 0})
        bucket["outcomes"].append(row["outcome"])
        if row["dropped"]:
            bucket["dropped"] += 1
    out: list[dict[str, Any]] = []
    for skill_id in sorted(grouped):
        bucket = grouped[skill_id]
        item = _summarize(bucket["outcomes"], dropped=bucket["dropped"])
        item["skillId"] = skill_id
        item["agentType"] = agent_type
        item["injected"] = len(bucket["outcomes"]) - bucket["dropped"]
        item["judged"] = item["settled"] >= max(1, int(min_samples))
        out.append(item)
    return out


def ok_rates(*, agent_type: str | None = None,
             min_samples: int = MIN_SAMPLES) -> dict[str, float]:
    """只返回**有结论**的 skill 的 ``okRate``。

    这是 ``load_agent_skills`` 唯一的读口，必须**永不抛** —— 统计库坏了顶多退化成
    「按 priority 排序」，不能让 agent 起不来。
    """
    try:
        return {item["skillId"]: item["okRate"] for item in summary(
            agent_type=agent_type, min_samples=min_samples)
            if item["judged"] and item["okRate"] is not None}
    except Exception:  # noqa: BLE001 - 读统计不许影响主流程
        return {}


def penalty(agent_type: str, skill_id: str, *, min_samples: int = MIN_SAMPLES) -> tuple[int, str]:
    """给一个 skill 的降级惩罚与理由（``(0, "")`` 表示不降级）。"""
    rate = ok_rates(agent_type=agent_type, min_samples=min_samples).get(skill_id)
    if rate is None or rate >= LOW_OK_RATE:
        return 0, ""
    return PRIORITY_PENALTY, f"历史成功率 {rate:.0%}（{min_samples}+ 次已结算）"


def stats() -> dict[str, Any]:
    """整体盘点：总行数、被裁次数、有结论的 skill 数。"""
    try:
        with _session() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM skill_usage").fetchone()[0])
            dropped = int(conn.execute(
                "SELECT COUNT(*) FROM skill_usage WHERE dropped = 1").fetchone()[0])
            settled = int(conn.execute(
                "SELECT COUNT(*) FROM skill_usage WHERE outcome IS NOT NULL").fetchone()[0])
            skills = int(conn.execute(
                "SELECT COUNT(DISTINCT skill_id) FROM skill_usage").fetchone()[0])
    except sqlite3.Error as exc:
        return {"available": False, "hint": f"统计库不可用：{exc}"}
    return {
        "available": True,
        "db": db_path().as_posix(),
        "rows": total,
        "droppedRows": dropped,
        "settledRows": settled,
        "skills": skills,
        "minSamples": MIN_SAMPLES,
        "lowOkRate": LOW_OK_RATE,
        "penalty": PRIORITY_PENALTY,
        "note": "okRate 是「注入该 skill 的跑次里成功的比例」，不是命中率／贡献度",
    }
