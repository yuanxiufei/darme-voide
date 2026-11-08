"""优化史 —— 回答「这次改动到底提升了多少」（知识闭环 §1 缺口 3）。

为什么需要
============
优化器 ``agent.evaluation.optimizer.optimize_agent_prompt()`` 每跑一轮都会把分数写进
``benchmarks/history/<时间戳>.json`` ✓，而「当前生效的提示词」落在 ``agent_configs`` ✗ ——
两张嘴各说一半，于是三个问题都回答不了 ✗：

* 「这次改动提升了多少」（要 ``score_before`` / ``score_after`` 成对 ✓）
* 「v4 是拿什么分换来的、它的父版本是哪一版」（要 ``version`` / ``parent_version`` ✓）
* 「历史上回滚过几次、有没有哪版被采纳后又没落库」（要 ``accepted`` 与 ``persisted`` 分开 ✓）

更硬的一条：``agent_configs`` 是**覆盖式**的（``persist_agent_config`` 直接改那一行）⇒
上一次生效的提示词与它的分数**当场就没了** ✗，连对比基准都不存在 ✓。

三条口径
========
1. **一版一行，追加不覆盖** ✓：``agent_configs`` 可以覆盖，事实层不行 —— 优化史是「发生过什么」，
   不是「现在是什么」✓✗。
2. **``accepted`` 与 ``persisted`` 分开** ✓：判决（分数严格更高 ⇔ 采纳）与生效（写成默认值）
   是两件事 ✓。合成一个布尔就会出现「页面说已采纳、实际默认提示词还是旧的」这种最难查的假绿 ✗。
3. **``target`` 必填，现已有两格**：``system_prompt`` 与 ``params``（温度，缺口 4 第一格）✓；
   工具集/技能绑定/QC 阈值/引擎参数仍未开 ✗ —— 留列不留假数据：没实现的目标不会出现在表里 ✓。

写口不抛
========
``record()`` / ``mark_persisted()`` 失败只记 warn 并返回 ``None`` / ``0`` ✓✗：
优化本身是长跑任务（拉模型、跑整套 case），**日志写不进去不许把整个优化跑废掉** ✗；
但读口反过来必须如实报 ``available=False`` ✓（读不出来还给空表，人就会以为「从没优化过」✗）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..core.response import now
from .task_logger import log_task_warn

__all__ = ["REASONS", "TARGETS", "list_records", "mark_persisted", "record", "summary"]

#: 优化目标（落库值 → 人话）✓。新增目标前先想清楚：它的「分数」从哪来 ✗。
TARGETS: dict[str, str] = {
    "system_prompt": "系统提示词",
    # 缺口 4 的第一格：同提示词、只换温度（`agent/evaluation/param_optimizer.py`）。
    # 这一格的「分数」与提示词优化共用同一套 case 判分，所以能放进同一张表比。
    "params": "生成参数（温度）",
}

#: 一轮优化的判决理由（落库值 → 人话）✓。键是**闭集**：页面按它分组/上色 ✓。
REASONS: dict[str, str] = {
    "accepted": "分数严格更高，采纳",
    "no_gain": "分数没涨，保持原版",
    "runtime_mismatch": "运行环境变了，分数不可比",
    # 参数调优专用：均值确实涨了，但涨幅没超过同一档位的抖动带 ⇒ 按「没提升」处理。
    # 单独一档是为了**看得出这是被抖动带挡掉的**，而不是真的没涨（那是两件不同的事）。
    "below_noise": "涨了，但没超过抖动带",
    # 候选本身跑崩/没跑完（模型报错、超时）⇒ 这一档没有可比分数，记「不作数」而不是「没涨」。
    "eval_failed": "评测没跑完，不作数",
}

#: 页面一次最多读多少行（防止老库把端点拖死 ✓）。
MAX_LIMIT = 500


def _prompt_sha(prompt: str) -> str:
    """候选提示词的 sha1 前 16 位 ✓。

    为什么存指纹不存全文 ✗：一段系统提示词几千字，几十版就是几十万字段，而这张表的用途是
    「对比分数」不是「回放内容」✓；真要回放，``benchmarks/history/`` 里的 JSON 有全文 ✓。
    """
    text_ = str(prompt or "")
    return hashlib.sha1(text_.encode("utf-8")).hexdigest()[:16] if text_ else ""


def _json(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _row_to_dict(row: Any) -> dict[str, Any]:
    item = dict(row._mapping)  # noqa: SLF001 —— SQLAlchemy 2.x Row → dict 的公开写法在 1.4 上没有
    return {
        "id": item["id"],
        "agentType": item.get("agent_type") or "",
        "caseId": item.get("case_id") or "",
        "target": item.get("target") or "system_prompt",
        "targetLabel": TARGETS.get(item.get("target") or "system_prompt",
                                   item.get("target") or ""),
        "version": item.get("version"),
        "parentVersion": item.get("parent_version"),
        "scoreBefore": item.get("score_before"),
        "scoreAfter": item.get("score_after"),
        "delta": item.get("delta"),
        "accepted": bool(item.get("accepted")),
        "persisted": bool(item.get("persisted")),
        "reason": item.get("reason") or "",
        "reasonLabel": REASONS.get(item.get("reason") or "", item.get("reason") or ""),
        "runtimeModel": item.get("runtime_model") or "",
        "promptSha": item.get("prompt_sha") or "",
        "evidence": item.get("evidence") or "",
        "createdAt": item.get("created_at"),
    }


def record(conn: Connection, *, agent_type: str, version: int,
           parent_version: int | None = None, score_before: float | None = None,
           score_after: float | None = None, accepted: bool = False, reason: str = "",
           target: str = "system_prompt", case_id: str = "", runtime_model: str = "",
           prompt: str = "", evidence: Any = None) -> dict[str, Any] | None:
    """落一行优化史 ✓；**失败不抛**（返回 ``None``，见模块头「写口不抛」✓）。

    ``delta`` 在这里算而不是让调用方传 ✓：``score_before`` 为空（第一版没有对手 ✓）时
    ``delta=None`` 而不写 0 ✗ —— 写 0 会在页面上显示成「提升了 0」，那是在编数 ✗。
    """
    delta: float | None = None
    if score_before is not None and score_after is not None:
        delta = round(float(score_after) - float(score_before), 4)
    payload = {
        "agent_type": str(agent_type or ""),
        "case_id": str(case_id or ""),
        "target": target if target in TARGETS else "system_prompt",
        "version": int(version),
        "parent_version": None if parent_version is None else int(parent_version),
        "score_before": None if score_before is None else float(score_before),
        "score_after": None if score_after is None else float(score_after),
        "delta": delta,
        "accepted": 1 if accepted else 0,
        "persisted": 0,
        "reason": reason if reason in REASONS else (reason or ""),
        "runtime_model": str(runtime_model or ""),
        "prompt_sha": _prompt_sha(prompt),
        "evidence": _json(evidence),
        "stamp": now(),
    }
    if not payload["agent_type"]:
        log_task_warn("optimization_log", "优化史缺 agent_type，跳过这一行")
        return None
    try:
        conn.execute(
            text("INSERT INTO agent_optimizations (agent_type, case_id, target, version, "
                 "parent_version, score_before, score_after, delta, accepted, persisted, reason, "
                 "runtime_model, prompt_sha, evidence, created_at) "
                 "VALUES (:agent_type, :case_id, :target, :version, :parent_version, :score_before, "
                 ":score_after, :delta, :accepted, :persisted, :reason, :runtime_model, :prompt_sha, "
                 ":evidence, :stamp)"),
            payload,
        )
        found = conn.execute(text("SELECT last_insert_rowid()")).first()
        new_id = int(found[0]) if found else 0
        row = conn.execute(text("SELECT * FROM agent_optimizations WHERE id = :id"),
                           {"id": new_id}).first()
        return _row_to_dict(row) if row is not None else None
    except Exception as err:  # noqa: BLE001 —— 见模块头「写口不抛」：日志不许废掉优化本身
        log_task_warn("optimization_log", f"优化史写入失败（已忽略）：{err}")
        return None


def mark_persisted(conn: Connection, *, agent_type: str, version: int,
                   case_id: str = "") -> int:
    """把某一版标成「真的写成默认值了」✓；返回改到的行数（0＝没这条记录，也算正常 ✓）。

    为什么单独一步 ✗：「被采纳」（分数高）与「被落库」（变成默认）是优化器的两个不同出口 ✓——
    ``persist_agent_config`` 可能因为调用方没给 ``persist`` 参数而根本没跑 ✓，合成一个布尔就会
    出现「页面说已采纳、而默认提示词还是旧的」这种最难查的假绿 ✗。
    """
    sql = ("UPDATE agent_optimizations SET persisted = 1 "
           "WHERE agent_type = :agent_type AND version = :version")
    params: dict[str, Any] = {"agent_type": str(agent_type or ""), "version": int(version)}
    if case_id:
        sql += " AND case_id = :case_id"
        params["case_id"] = str(case_id)
    try:
        result = conn.execute(text(sql), params)
        return int(result.rowcount or 0)
    except Exception as err:  # noqa: BLE001 —— 同上
        log_task_warn("optimization_log", f"优化史标记落库失败（已忽略）：{err}")
        return 0


def _optional_conn(conn: Connection | None) -> tuple[Connection, bool]:
    if conn is not None:
        return conn, False
    from ..core import db  # noqa: PLC0415 —— 只在需要时才碰 DB 单例（测试可换数据根 ✓）
    return db.engine.connect(), True


def summary(conn: Connection | None = None, *, agent_type: str | None = None) -> dict[str, Any]:
    """汇总：跑过几轮、采纳几次、落库几次、最大提升多少 ✓。读不出来如实 ``available=False`` ✓。"""
    own = False
    try:
        conn, own = _optional_conn(conn)
        where = " WHERE agent_type = :at" if agent_type else ""
        params = {"at": agent_type} if agent_type else {}
        rows = conn.execute(
            text("SELECT * FROM agent_optimizations" + where + " ORDER BY id DESC"),
            params,
        ).fetchall()
        items = [_row_to_dict(row) for row in rows]
        counted = [i for i in items if isinstance(i.get("delta"), (int, float))]
        best = max(counted, key=lambda i: i["delta"]) if counted else None
        by_target: dict[str, int] = {}
        for item in items:
            by_target[item["target"]] = by_target.get(item["target"], 0) + 1
        return {
            "available": True,
            "rounds": len(items),
            "accepted": sum(1 for i in items if i["accepted"]),
            "persisted": sum(1 for i in items if i["persisted"]),
            "scored": len(counted),
            "bestDelta": None if best is None else best["delta"],
            "bestVersion": None if best is None else best["version"],
            "lastAt": items[0]["createdAt"] if items else None,
            "byTarget": by_target,
            "targets": TARGETS,
            "reasons": REASONS,
            "generatedAt": now(),
        }
    except Exception as err:  # noqa: BLE001 —— 读口也不许把页面拖崩，但必须说清读不出来
        log_task_warn("optimization_log", f"优化史汇总失败：{err}")
        return {"available": False, "reason": f"读不出来：{err}", "rounds": 0, "accepted": 0,
                "persisted": 0, "scored": 0, "bestDelta": None, "bestVersion": None,
                "lastAt": None, "byTarget": {}, "targets": TARGETS, "reasons": REASONS,
                "generatedAt": now()}
    finally:
        if own and conn is not None:
            conn.close()


def list_records(conn: Connection | None = None, *, agent_type: str | None = None,
                 case_id: str | None = None, limit: int = 100) -> dict[str, Any]:
    """最近的优化记录（倒序 ✓）。返回 ``{available, items, summary}`` 一起给页面 ✓。"""
    own = False
    try:
        conn, own = _optional_conn(conn)
        where: list[str] = []
        params: dict[str, Any] = {"limit": max(1, min(int(limit), MAX_LIMIT))}
        if agent_type:
            where.append("agent_type = :at")
            params["at"] = agent_type
        if case_id:
            where.append("case_id = :cid")
            params["cid"] = case_id
        sql = "SELECT * FROM agent_optimizations"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT :limit"
        rows = conn.execute(text(sql), params).fetchall()
        return {
            "available": True,
            "items": [_row_to_dict(row) for row in rows],
            "summary": summary(conn, agent_type=agent_type),
        }
    except Exception as err:  # noqa: BLE001 —— 同上：读不出来要说出来，不返空表
        log_task_warn("optimization_log", f"优化史读取失败：{err}")
        return {"available": False, "reason": f"读不出来：{err}", "items": [],
                "summary": summary(None, agent_type=agent_type)}
    finally:
        if own and conn is not None:
            conn.close()
