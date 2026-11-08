"""成片评测（知识闭环缺口 8 的另一半：**评成片好坏**）。

问题在哪
========
原评测面只有 ``benchmarks/`` 的基准（extractor / script-rewriter / storyboard-breaker /
voice-assigner，2026-11-07 又补了第 5 份 ``grid-prompt``），全部在**产出层**：提示词写得好不好、
结构化输出数量对不对。没有任何一条判据看**成片**——而本仓真正花钱、真正决定交付质量的那一步
恰好是成片（每镜一次视频生成 + 一次技术质检），它此前只在 ``qc`` 页被人肉看，
**从不进入评测口径** ✓✗。

判据只能从"已经发生的事实"取
==============================
这里**不重新跑 ffmpeg** ✗、不调模型 ✗、不自造"画面好不好"的主观分 ✗：

1. ``storyboards`` —— 这一集**应该**有多少镜、每镜计划多少秒（计划面）；
2. ``video_generations`` —— 每镜最新的 ``completed`` 成片（出片面）；
3. ``video_quality_checks`` —— 技术 QC **已经量过**的东西：``dimensions.video_spec.score``、
   ``issues``（黑场/冻帧/响度/规格）、``overall_score``（已发生的事实面）；
4. ``qc_reviews`` —— 人**已经判过**的结论：``hit`` / ``false_alarm`` / ``missed``（人的判断面）。

为什么不去重跑一遍 ffprobe/blackdetect ✗✗：① 每镜一次 probe + 黑场 + 冻帧三趟 ffmpeg，
一集 57 镜就是 171 次调用，"跑一次评测"变成一次全量转码；② 更要命的是**两处口径会打架**——
QC 页显示"这镜 2 处黑场"，评测页却算出 3 处，人只会信错的那个 ✗。评测要评的是
"**当前这套质检口径下的成片**"，那就必须读同一份质检结论 ✓。

四个维度（等权，缺项不参与）
============================
============================  ==========================================
``coverage``  出片覆盖        有成片的镜 / 计划镜数
``spec``      技术规格        每镜 ``video_spec.score`` 均值（QC 自带 100 起扣）
``defects``   缺陷面          有 error 级 issue 的镜占比 ⇒ ``100*(1-占比)``
``review``    人工复核        命中率 ``hit / (hit + false_alarm)``；漏报**单列**
============================  ==========================================

三条口径（都是"宁可说判不了"）：
* 某一维**一条样本都没有** ⇒ 该维 ``judged=false`` 且 **不参与**总分（不是算 0 ✗，也不是算满分 ✗），
  并在 ``missing`` 里点名 —— 把"没数据"折成任何一个数字都是编 ✓✗；
* 等权是**人为口径**（不是发现）：四维各有各的失败方式，没有哪一维天然更贵重；
  真按权重调，改这里一行、回执与历史里都写着，不需要重跑；
* ``verdict`` 只表达"够不够格交片"：``empty``（这集还没分镜）/ ``partial``（有维度没数据）/
  ``blocked``（分数低于 ``PASS_SCORE``）/ ``ready``。
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from ...services.task_logger import log_task_warn

__all__ = ["DIMENSIONS", "PASS_SCORE", "score_episode"]

#: 四维的名字与中文标签（顺序＝回执里的顺序）
DIMENSIONS: dict[str, str] = {
    "coverage": "出片覆盖",
    "spec": "技术规格",
    "defects": "缺陷面",
    "review": "人工复核",
}

#: 够格交片的分数线（0–100）。**不是**"质量标准"，是"低于这条要不要人先看一眼"
PASS_SCORE = 80.0


def _rows(conn: Connection, sql: str, **params: Any) -> list[dict[str, Any]]:
    """只读查询；库坏了返回空列表（评测算不出东西，但**不许**把页面打 500）。"""
    try:
        result = conn.execute(text(sql), params)
    except SQLAlchemyError as err:
        log_task_warn("FilmCheck", "query-failed", {"error": str(err)})
        return []
    return [dict(row._mapping) for row in result.fetchall()]


def _json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        loaded = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return loaded if isinstance(loaded, list) else []


def _json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _shots(conn: Connection, episode_id: int) -> list[dict[str, Any]]:
    """这一集的镜头清单（计划面）：镜号、计划秒数。"""
    return _rows(conn, """
        SELECT id, storyboard_number, duration
        FROM storyboards
        WHERE episode_id = :eid AND deleted_at IS NULL
        ORDER BY storyboard_number, id
    """, eid=episode_id)


def _latest_takes(conn: Connection, shot_ids: list[int]) -> dict[int, dict[str, Any]]:
    """每镜**最新一条 completed 成片**（同一镜可能重跑过多次，只认最后一条 ✓）。"""
    if not shot_ids:
        return {}
    marks = ",".join(f":s{index}" for index in range(len(shot_ids)))
    params = {f"s{index}": int(sid) for index, sid in enumerate(shot_ids)}
    rows = _rows(conn, f"""
        SELECT g.id, g.storyboard_id, g.duration, g.fps, g.width, g.height,
               g.local_path, g.video_url, g.completed_at
        FROM video_generations g
        JOIN (
            SELECT storyboard_id, MAX(id) AS max_id
            FROM video_generations
            WHERE deleted_at IS NULL AND status = 'completed'
              AND storyboard_id IN ({marks})
            GROUP BY storyboard_id
        ) last ON last.max_id = g.id
    """, **params)
    return {int(row["storyboard_id"]): row for row in rows}


def _checks(conn: Connection, gen_ids: list[int]) -> dict[int, dict[str, Any]]:
    """每条成片最新一条 QC 记录（**已发生的事实面**：不再重跑 ffmpeg）。"""
    if not gen_ids:
        return {}
    marks = ",".join(f":g{index}" for index in range(len(gen_ids)))
    params = {f"g{index}": int(gid) for index, gid in enumerate(gen_ids)}
    rows = _rows(conn, f"""
        SELECT q.id, q.video_generation_id, q.storyboard_id, q.overall_score,
               q.status, q.issues, q.dimensions, q.updated_at
        FROM video_quality_checks q
        JOIN (
            SELECT video_generation_id, MAX(id) AS max_id
            FROM video_quality_checks
            WHERE video_generation_id IN ({marks})
            GROUP BY video_generation_id
        ) last ON last.max_id = q.id
    """, **params)
    return {int(row["video_generation_id"]): row for row in rows}


def _reviews(conn: Connection, check_ids: list[int]) -> list[dict[str, Any]]:
    """人已经判过的结论（按 ``check_id`` 取；没判过就是空）。"""
    if not check_ids:
        return []
    marks = ",".join(f":c{index}" for index in range(len(check_ids)))
    params = {f"c{index}": int(cid) for index, cid in enumerate(check_ids)}
    return _rows(conn, f"""
        SELECT check_id, code, verdict FROM qc_reviews
        WHERE check_id IN ({marks})
    """, **params)


def _dimension(name: str, score: float | None, *, samples: int,
               notes: list[str] | None = None) -> dict[str, Any]:
    """一维结论：``judged=False`` ⇒ 没数据 ⇒ 不参与总分（消费方据 ``judged`` 掉该维）。"""
    return {
        "name": name,
        "label": DIMENSIONS[name],
        "score": None if score is None else round(float(score), 1),
        "samples": int(samples),
        "judged": score is not None,
        "notes": [item for item in (notes or []) if item],
    }


def score_episode(conn: Connection, episode_id: int, *, record: bool = True) -> dict[str, Any]:
    """评一集的成片；``record=True`` 时把结论落成一条 ``kind="eval"`` 跑次（幂等覆盖）。

    跑次用 ``run_id=f"film:{episode_id}"``：同一集重评**覆盖**而不是并列 ✓ ——
    「这集的成片分」只有一个当前值，历史要看 ``meta.*At`` / 别处留痕，
    不能让"我点了五次评测"变成五条样本去稀释统计 ✓✗。
    """
    eid = int(episode_id)
    shots = _shots(conn, eid)
    shot_ids = [int(row["id"]) for row in shots]
    takes = _latest_takes(conn, shot_ids)
    gen_ids = [int(row["id"]) for row in takes.values()]
    checks = _checks(conn, gen_ids)
    reviews = _reviews(conn, [int(row["id"]) for row in checks.values()])

    total = len(shot_ids)
    ready = len(takes)
    missing_shots = [int(row["storyboard_number"] or 0) for row in shots
                     if int(row["id"]) not in takes]

    # ---- spec 维：技术 QC 自己给的规格分（100 起扣，黑场/冻帧各扣 20）
    spec_scores: list[float] = []
    defect_shots: dict[int, set[str]] = {}
    no_check: list[int] = []
    for shot_id, gen in takes.items():
        check = checks.get(int(gen["id"]))
        if check is None:
            no_check.append(shot_id)
            continue
        dims = _json_dict(check.get("dimensions"))
        spec = dims.get("video_spec")
        if isinstance(spec, dict) and isinstance(spec.get("score"), (int, float)):
            spec_scores.append(float(spec["score"]))
        bad = {str(item.get("code")) for item in _json_list(check.get("issues"))
               if isinstance(item, dict) and item.get("code")
               and str(item.get("severity") or "") == "error"}
        if bad:
            defect_shots[shot_id] = bad

    spec_avg = sum(spec_scores) / len(spec_scores) if spec_scores else None

    # ---- defects 维：有 error 级缺陷的镜占比（分母＝**有 QC 的**镜，不是全部镜）
    judged_takes = len(spec_scores)
    defect_rate = (len(defect_shots) / judged_takes) if judged_takes else None
    defect_score = None if defect_rate is None else 100.0 * (1.0 - defect_rate)

    codes: dict[str, dict[str, Any]] = {}
    for shot_id, found in defect_shots.items():
        for code in found:
            slot = codes.setdefault(code, {"code": code, "count": 0, "shots": []})
            slot["count"] += 1
            slot["shots"].append(shot_id)
    defect_codes = sorted(codes.values(), key=lambda item: (-item["count"], item["code"]))

    # ---- review 维：人给的命中率；**漏报单列**（它不进命中率分母，见 qc_reviews 口径 2）
    hits = sum(1 for item in reviews if item.get("verdict") == "hit")
    alarms = sum(1 for item in reviews if item.get("verdict") == "false_alarm")
    missed = sum(1 for item in reviews if item.get("verdict") == "missed")
    judged_reviews = hits + alarms
    review_score = (100.0 * hits / judged_reviews) if judged_reviews else None

    coverage_score = (100.0 * ready / total) if total else None

    dimensions = [
        _dimension("coverage", coverage_score, samples=total,
                   notes=[f"{len(missing_shots)} 镜还没有成片：{missing_shots[:12]}"]
                   if missing_shots else []),
        _dimension("spec", spec_avg, samples=judged_takes,
                   notes=[f"{len(no_check)} 镜有成片但还没跑过技术质检：{no_check[:12]}"]
                   if no_check else []),
        _dimension("defects", defect_score, samples=judged_takes,
                   notes=[f"{item['code']} × {item['count']} 镜" for item in defect_codes[:5]]),
        _dimension("review", review_score, samples=judged_reviews,
                   notes=[f"已复核 {len(reviews)} 条；漏报 {missed} 条"] if reviews else
                         ["还没有人复核过任何一条质检结论"]),
    ]

    # ---- 总分：只在**有值**的维度上等权（缺项不参与，也不补 0）
    scored = [item["score"] for item in dimensions if item["judged"]]
    score = round(sum(scored) / len(scored), 1) if scored else None
    missing = [item["name"] for item in dimensions if not item["judged"]]

    if not total:
        verdict = "empty"
    elif missing:
        verdict = "partial"
    elif score is not None and score >= PASS_SCORE:
        verdict = "ready"
    else:
        verdict = "blocked"

    notes: list[str] = []
    if missing_shots:
        notes.append(f"出片覆盖 {ready}/{total}：{len(missing_shots)} 镜没成片，"
                     f"这一集的分数只代表已出的那部分")
    for item in defect_codes[:3]:
        notes.append(f"`{item['code']}` 命中 {item['count']} 镜（{item['shots'][:8]}）")
    if missed:
        notes.append(f"人工复核里还有 {missed} 条**漏报**：说明该报没报，"
                     f"缺陷面那一维此刻是偏乐观的")
    if not judged_reviews:
        notes.append("没有人工复核样本 ⇒ 命中率无从谈起（这是缺数据，不是通过）")

    report: dict[str, Any] = {
        "episodeId": eid,
        "total": total,
        "ready": ready,
        "missingShots": missing_shots,
        "dimensions": dimensions,
        "score": score,
        "verdict": verdict,
        "passScore": PASS_SCORE,
        "missing": missing,
        "defectCodes": defect_codes,
        "review": {"hit": hits, "falseAlarm": alarms, "missed": missed,
                   "reviewed": len(reviews),
                   "hitRate": round(hits / judged_reviews, 4) if judged_reviews else None},
        "notes": notes,
        "recorded": False,
    }
    if record:
        report["recorded"] = _record(report)
    return report


def _record(report: dict[str, Any]) -> bool:
    """落一条 ``kind="eval"`` 跑次（缺口 8：评测得进得去闭环，否则它只是一张截图）。"""
    try:
        from ...services import outcome_index

        return outcome_index.record_run({
            "run_id": f"film:{report['episodeId']}",
            "kind": "eval",
            "stage": "film",
            "status": "ok" if report["verdict"] != "partial" else "partial",
            "score": None if report["score"] is None else round(report["score"] / 100.0, 4),
            "episode_id": report["episodeId"],
            "meta": {
                "verdict": report["verdict"],
                "total": report["total"],
                "ready": report["ready"],
                "missing": report["missing"],
                "passScore": report["passScore"],
                "dimensions": {item["name"]: item["score"] for item in report["dimensions"]},
                "review": report["review"],
                "defectCodes": [{"code": item["code"], "count": item["count"]}
                                for item in report["defectCodes"]],
            },
        })
    except Exception as err:  # noqa: BLE001 —— 记账失败不影响"已经评出来了"
        log_task_warn("FilmCheck", "record-failed", {"error": str(err)})
        return False
