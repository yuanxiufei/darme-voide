"""质检的**人工复核结论** —— 阈值命中率的分母（知识闭环 §4.3 里唯一未做的那条）。

为什么需要
============
``technical_qc.ISSUE_CODES`` 的注释早就写明它的第二个用途：「以后要统计哪个阈值最常被触发时，
分母是 code 而不是文案」✓ —— 但**只有分子**（触发了多少次）是算不出命中率的 ✗：
``black`` 报了 12 次，到底是「这 12 次全是真黑场 ✓」还是「其中 9 次是片头淡入的误杀 ✗」，
机器看不出来，**只有人看过成片才知道** ✓。所以这条缺口缺的不是统计代码，而是**分母**：
一张能记录「人看过这条 QC 结论、判它对还是错」的表。

口径（三条，直接影响数字）
==========================
1. **复核粒度是 ``(check_id, code)``，不是 issue 行** ✓：同一次质检里可能报两段黑场
   （两条 issue、同一个 code），复核人判的是「这条片子的黑场判定准不准」这一件事 ✗。
   所以「触发数」也按 ``(check_id, code)`` 去重计数 ✓ —— 两边口径不一致就是假命中率 ✗。
2. **命中率 = hit / (hit + false_alarm)** ✓：分母只含**触发过的**复核（有 issue 才谈得上判对判错）。
   「漏报」是另一个问题（该报没报），单独给 ``missed`` 计数与 ``missRate``（分母＝该 code
   全部复核条数）✗ 不与命中率混在一个分母里：混进去会让「漏报多」看起来像「命中率下降」，
   而这两种结论要调的阈值方向完全相反（误报多 ⇒ 放宽；漏报多 ⇒ 收紧）✓✗。
3. **样本不足不编数** ✓：``reviewed < MIN_REVIEWS`` 时 ``hitRate`` 照样算出来（这是事实 ✓），
   但 ``verdict`` 字段留 ``"insufficient"`` 并且**不生成调参候选** ✗ —— 3 条复核里有 1 条误报
   不代表阈值就该动 ✓✗。

为什么单开一张表
================
见 ``core/models.py::qc_reviews`` 的注释（漏报没有 issue 行可挂 / 一次复核逐 code 判 /
复核是事实层不该被 QC 重跑覆盖）。

与台账的关系
============
本模块**只统计**，不写知识台账 ✗：命中率是「调参候选」的输入（``engine.tuning.suggest()``），
而候选一律 ``canApply=False``（shadow pending）⇒ 这里算得再准也不会自动改阈值 ✓。
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..core.response import now
from . import technical_qc
from .task_logger import log_task_warn

__all__ = ["MIN_REVIEWS", "VERDICTS", "delete_review", "hit_rate", "list_reviews",
           "pending", "record_review"]

#: 人给的三种结论 ✓。值是人话（页面直接显示 ✓），键是落库值（统计只认这三个 ✗）。
VERDICTS: dict[str, str] = {
    "hit": "判对了（确实有这个缺陷）",
    "false_alarm": "误报（人看着没问题）",
    "missed": "漏报（该判没判）",
}

#: 少于这么多条复核就**不出候选** ✓（只把数字摆出来给人看）。
#: 为什么是 10 而不是 ``tuning.MIN_SAMPLES``（30）✗：复核是**人工成本**，一次要拖片子看完，
#: 30 条意味着这条规则要几周才够格；而误报的分布比跑次分布集中得多（同一条规则的错法很像 ✓）。
#: 这个数字是**门槛**不是结论：样本不足时页面明说「还差 N 条」✓，不假装没有信息 ✗。
MIN_REVIEWS = 10

#: 统计时最多回看多少条 QC 记录（防止老库上全表扫描把端点拖死 ✓）。
SCAN_LIMIT = 2000


def _json_list(raw: Any) -> list[Any]:
    """``issues`` / ``dimensions`` 是 JSON 文本 ✓；坏数据当空表（不让一条脏记录拖垮整个统计 ✓）。"""
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _label(code: str) -> str:
    return technical_qc.ISSUE_CODES.get(code, code)


def _row_to_dict(row: Any) -> dict[str, Any]:
    item = dict(row._mapping)  # noqa: SLF001 —— SQLAlchemy 2.x Row → dict 的公开写法在 1.4 上没有
    return {
        "id": item["id"],
        "checkId": item["check_id"],
        "storyboardId": item.get("storyboard_id"),
        "videoGenerationId": item.get("video_generation_id"),
        "code": item["code"],
        "label": _label(item["code"]),
        "verdict": item["verdict"],
        "verdictLabel": VERDICTS.get(item["verdict"], item["verdict"]),
        "note": item.get("note") or "",
        "reviewer": item.get("reviewer") or "",
        "createdAt": item.get("created_at"),
        "updatedAt": item.get("updated_at"),
    }


def record_review(conn: Connection, *, check_id: int, code: str, verdict: str,
                  note: str = "", reviewer: str = "",
                  storyboard_id: int | None = None,
                  video_generation_id: int | None = None) -> dict[str, Any]:
    """写一条复核结论 ✓；``(check_id, code)`` **已存在就改判**（覆盖式 upsert ✓）。

    为什么覆盖而不是追加 ✗：一个人先判「误报」，看了放大帧后改判「判对了」——
    那是**改判**（同一件事的结论变了 ✓），追加会把它算成两条 ⇒ 命中率被自己的犹豫稀释 ✗。
    真要看改判史，``updated_at`` 与 ``note`` 里自己写（历史不该由统计口径来承担 ✓）。
    """
    code = str(code or "").strip()
    verdict = str(verdict or "").strip()
    if not code:
        raise ValueError("code 不能为空")
    if verdict not in VERDICTS:
        raise ValueError(f"verdict 只认 {sorted(VERDICTS)}，收到 {verdict!r}")
    #: ⚠️ code 必须落在闭集里 ✓：放任意字符串进来 ⇒ 统计表里长出「没人认得的类别」✗，
    #:    而命中率是按 code 分组的 ⇒ 那些行永远没有阈值可调、也永远不会被看 ✗。
    if code not in technical_qc.ISSUE_CODES:
        raise ValueError(f"未知的 issue code：{code!r}（闭集见 technical_qc.ISSUE_CODES）")

    stamp = now()
    existing = conn.execute(
        text("SELECT id FROM qc_reviews WHERE check_id = :cid AND code = :code ORDER BY id LIMIT 1"),
        {"cid": int(check_id), "code": code},
    ).first()
    if existing is not None:
        conn.execute(
            text("UPDATE qc_reviews SET verdict = :verdict, note = :note, reviewer = :reviewer, "
                 "storyboard_id = :sb, video_generation_id = :vg, updated_at = :stamp "
                 "WHERE id = :id"),
            {"verdict": verdict, "note": note or "", "reviewer": reviewer or "",
             "sb": storyboard_id, "vg": video_generation_id, "stamp": stamp,
             "id": int(existing[0])},
        )
        review_id = int(existing[0])
    else:
        conn.execute(
            text("INSERT INTO qc_reviews (check_id, storyboard_id, video_generation_id, code, "
                 "verdict, note, reviewer, created_at, updated_at) "
                 "VALUES (:cid, :sb, :vg, :code, :verdict, :note, :reviewer, :stamp, :stamp)"),
            {"cid": int(check_id), "sb": storyboard_id, "vg": video_generation_id, "code": code,
             "verdict": verdict, "note": note or "", "reviewer": reviewer or "", "stamp": stamp},
        )
        found = conn.execute(text("SELECT last_insert_rowid()")).first()
        review_id = int(found[0]) if found else 0

    row = conn.execute(text("SELECT * FROM qc_reviews WHERE id = :id"),
                       {"id": review_id}).first()
    return _row_to_dict(row) if row is not None else {"id": review_id, "code": code,
                                                      "verdict": verdict}


def delete_review(conn: Connection, review_id: int) -> bool:
    """撤销一条复核（打错码/选错片子 ✓）；返回是否真删掉了 ✓。"""
    found = conn.execute(text("SELECT id FROM qc_reviews WHERE id = :id"),
                         {"id": int(review_id)}).first()
    if found is None:
        return False
    conn.execute(text("DELETE FROM qc_reviews WHERE id = :id"), {"id": int(review_id)})
    return True


def list_reviews(conn: Connection, *, limit: int = 50, check_id: int | None = None,
                 code: str | None = None, verdict: str | None = None) -> list[dict[str, Any]]:
    """最近的复核结论（页面「已复核」列表用 ✓）。"""
    where: list[str] = []
    params: dict[str, Any] = {"limit": max(1, min(int(limit), 500))}
    if check_id is not None:
        where.append("check_id = :cid")
        params["cid"] = int(check_id)
    if code:
        where.append("code = :code")
        params["code"] = code
    if verdict:
        where.append("verdict = :verdict")
        params["verdict"] = verdict
    sql = "SELECT * FROM qc_reviews"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY COALESCE(updated_at, created_at) DESC, id DESC LIMIT :limit"
    rows = conn.execute(text(sql), params).fetchall()
    return [_row_to_dict(row) for row in rows]


def pending(conn: Connection, *, limit: int = 20) -> list[dict[str, Any]]:
    """**待复核**清单：最近若干条 QC 记录 ＋ 每条里的 code 及当前判定 ✓。

    为什么要它 ✗：复核表建好了但没人填，命中率永远是 ``None`` ✓✗ —— 这条缺口真正的
    卡点是「人不知道去核哪一条」✗，所以清单必须**主动把最近报了缺陷的片子摆到人面前** ✓，
    并且标出哪几个 code 还没判过（``verdict=None`` ✓）⇒ 一次点击就能填一条 ✓。
    """
    limit = max(1, min(int(limit), 200))
    records = conn.execute(
        text("SELECT id, storyboard_id, video_generation_id, overall_score, status, issues, "
             "created_at FROM video_quality_checks ORDER BY id DESC LIMIT :limit"),
        {"limit": limit},
    ).fetchall()
    if not records:
        return []

    ids = {int(row._mapping["id"]) for row in records}  # noqa: SLF001
    marks: dict[tuple[int, str], dict[str, Any]] = {}
    #: ⚠️ 不写 ``WHERE check_id IN :ids`` ✓✗：原生 SQL 里 IN 不吃元组参数（要 ``expanding``），
    #:    写成那种形式会在**第二条记录**上直接 OperationalError ✓。复核表是人工录的（几百行量级 ✓），
    #:    全读再按 id 过滤更省事，也少一个方言坑 ✓。
    rows = conn.execute(text("SELECT * FROM qc_reviews")).fetchall()
    for row in rows:
        item = _row_to_dict(row)
        if item["checkId"] in ids:
            marks[(item["checkId"], item["code"])] = item

    out: list[dict[str, Any]] = []
    for row in records:
        mapping = row._mapping  # noqa: SLF001
        codes: list[str] = []
        messages: dict[str, str] = {}
        for issue in _json_list(mapping["issues"]):
            if not isinstance(issue, dict):
                continue
            code = str(issue.get("code") or "")
            if not code:
                # 规则维度的 issue 没有 code（``dimension`` 是 lip_sync 之类）⇒ 不参与命中率 ✓
                continue
            if code not in messages:
                codes.append(code)
            messages.setdefault(code, str(issue.get("message") or ""))
        out.append({
            "checkId": int(mapping["id"]),
            "storyboardId": mapping["storyboard_id"],
            "videoGenerationId": mapping["video_generation_id"],
            "score": mapping["overall_score"],
            "status": mapping["status"],
            "createdAt": mapping["created_at"],
            "issues": [{"code": code, "label": _label(code), "message": messages[code],
                        "review": marks.get((int(mapping["id"]), code))} for code in codes],
            #: 一条都没判过的记录排前面更该被看 ✓；全判过就 ``reviewed=True`` ✓。
            "reviewed": bool(codes) and all((int(mapping["id"]), code) in marks for code in codes),
        })
    return out


def hit_rate(conn: Connection | None = None) -> dict[str, Any]:
    """按 code 统计「触发 / 复核 / 判对 / 误报 / 漏报 / 命中率」✓。

    **读不出来时如实说 ``available=False`` ＋原因** ✗ 不返一张全 0 的表：
    全 0 会被读成「阈值从来没误报过」✓✗ —— 那是最危险的一种假绿 ✗。

    ``conn=None`` 时自开一条只读连接（``tuning.suggest()`` 那条路没有 FastAPI 依赖 ✓）。
    """
    owned = conn is None
    if owned:
        from ..core import db  # noqa: PLC0415 —— 只在需要时才碰 DB 单例（测试可换数据根 ✓）
        conn = db.engine.connect()
    try:
        return _hit_rate(conn)
    except Exception as err:  # noqa: BLE001 —— 统计失败不许把调用方（调参候选）拖崩
        log_task_warn("qc_reviews", f"命中率统计失败：{err}")
        return {"available": False, "reason": f"读不出来：{err}", "codes": [],
                "records": {"total": 0, "reviewed": 0, "unreviewed": 0},
                "minReviews": MIN_REVIEWS, "generatedAt": now(), "notes": []}
    finally:
        if owned and conn is not None:
            conn.close()


def _hit_rate(conn: Connection) -> dict[str, Any]:
    #: ① 触发面：``(check_id, code)`` 去重（与复核粒度同一口径 ✓，见模块头口径 1 ✓）
    triggered: dict[str, set[int]] = {}
    record_ids: set[int] = set()
    rows = conn.execute(
        text("SELECT id, issues FROM video_quality_checks ORDER BY id DESC LIMIT :limit"),
        {"limit": SCAN_LIMIT},
    ).fetchall()
    for row in rows:
        mapping = row._mapping  # noqa: SLF001
        check_id = int(mapping["id"])
        record_ids.add(check_id)
        for issue in _json_list(mapping["issues"]):
            if isinstance(issue, dict) and issue.get("code"):
                triggered.setdefault(str(issue["code"]), set()).add(check_id)

    #: ② 复核面
    reviews = conn.execute(text("SELECT * FROM qc_reviews")).fetchall()
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    reviewed_checks: set[int] = set()
    for row in reviews:
        item = _row_to_dict(row)
        code = item["code"]
        slot = buckets.setdefault(code, {"hit": [], "false_alarm": [], "missed": []})
        slot.setdefault(item["verdict"], []).append(
            {"checkId": item["checkId"], "reviewId": item["id"],
             "storyboardId": item.get("storyboardId")})
        reviewed_checks.add(item["checkId"])

    codes: list[dict[str, Any]] = []
    for code in sorted(set(triggered) | set(buckets)):
        slot = buckets.get(code) or {}
        hit_items = slot.get("hit") or []
        false_items = slot.get("false_alarm") or []
        miss_items = slot.get("missed") or []
        hit, false_alarm, missed = len(hit_items), len(false_items), len(miss_items)
        reviewed = hit + false_alarm + missed
        judged = hit + false_alarm
        target = technical_qc.ISSUE_THRESHOLDS.get(code) or {}
        #: 凭据：调参候选要「可核」✓ ⇒ 把支撑这条结论的复核记录 id 带出去（最多 8 条 ✓）
        samples = (false_items or hit_items or miss_items)[:8]
        codes.append({
            "code": code,
            "label": _label(code),
            "triggeredRecords": len(triggered.get(code) or ()),
            "reviewed": reviewed,
            "hit": hit,
            "falseAlarm": false_alarm,
            "missed": missed,
            #: 只有「触发过的」才进命中率分母 ✓；一条都没判过 ⇒ ``None``（不是 0 ✗）
            "hitRate": round(hit / judged, 4) if judged else None,
            #: 漏报率分母＝该 code 全部复核条数 ✓（漏报自己不进命中率分母 ✓，见口径 2 ✓）
            "missRate": round(missed / reviewed, 4) if reviewed else None,
            "verdict": ("insufficient" if reviewed < MIN_REVIEWS else
                        "false_alarm_heavy" if judged and hit / judged < 0.5 else
                        "miss_heavy" if reviewed and missed / reviewed >= 0.3 else "ok"),
            "threshold": ({"stage": "qc", "field": target["field"],
                           "relax": target.get("relax", "raise"),
                           "aux": target.get("aux")} if target else None),
            "sampleReviews": samples,
        })

    total = len(record_ids)
    return {
        "available": True,
        "generatedAt": now(),
        "scanned": len(rows),
        "scanLimit": SCAN_LIMIT,
        "minReviews": MIN_REVIEWS,
        "records": {"total": total, "reviewed": len(reviewed_checks),
                    "unreviewed": max(0, total - len(reviewed_checks))},
        "codes": codes,
        "notes": [
            f"复核门槛 {MIN_REVIEWS} 条：不足只摆数字、不出调参候选",
            "命中率分母＝「触发过且判过对错」的样本；漏报单独算（分母＝该 code 全部复核）",
            f"最多回看最近 {SCAN_LIMIT} 条 QC 记录（老记录不参与统计，避免全表扫描）",
        ],
    }
