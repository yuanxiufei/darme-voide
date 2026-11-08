"""质检结论 ⇒ 引擎参数选择（知识闭环缺口 6 的另一半）。

问题
====
两边**各自都是通的** ✓：``technical_qc`` 会把每次质检写成 ``kind="qc"`` 跑次 ✓；
``tuning.suggest()`` 也读跑次出候选 ✓。但两边**从来没有对上过** ✗✗：

* QC 行落在伪档 ``stage="video_spec"`` ✓（它说的是"规格判据"✓，不是"哪族出的"）；
* 引擎行落在真档 ``h3`` / ``wan22`` / ``krea2`` ✓；
* 两边没有共同键 ⇒ 「这一族最近出的片质量掉了」**永远成不了一条关于这一族的结论** ✗。

于是 QC 失败只能触发 ``qc_retry`` 重跑**这一镜** ✓，对"下一镜用什么参数"零影响 ✗。

本模块只做三件事
================
1. **归因**：把 QC 行折到**产出它的引擎档**上。归因键由 ``engine/bridge.run_job`` 在
   提交成功那一刻登记（``outcome_index.records``）✓ —— 那是 ``stage`` 与 ``taskId``
   唯一同时出现在手上的时刻 ✓。
2. **分桶**：同一族内按 ``steps`` 分桶算 **QC 通过率**（不是分数 ✓）。
   为什么盯通过率 ✗：分数是连续量、且引擎跑次**根本不记分数** ✗（``_index_run`` 没有
   ``score`` 字段）⇒ ``tuning`` 的"步数平台期"规则在真机上一直是空转 ✓✗；
   而"过没过"是有人/有判据看过的**离散结论** ✓，正好能当闸门 ✓。
3. **闸门**：给 ``tuning._steps_candidates`` 一个可判的结论 ——
   **降步数会不会把质检通过率打下去**。会 ⇒ 这条候选不出 ✓（并留下理由，不是静默丢弃 ✓）。

口径纪律
========
* 每桶要有 ``MIN_BUCKET`` 条带质检结论的跑次才敢比较 ✓（少于此答"判不了" ✓，**不是**"通过"）。
* 只算**相对下滑**（高档 → 低档），不比绝对值：绝对通过率低是另一件事（该换档 ✓），
  不归本模块管 ✓。
* 索引库/归因表坏了 ⇒ 一律答 ``unavailable`` ✓ 并让调用方退回原行为 ✓
  （主链路不因为"统计不可用"而卡住 ✓）。
* 跑次索引**由服务层登记**（:mod:`.hooks` ✓），引擎自己不 import 服务层 ✓；
  没登记 ⇒ 每个口答 ``unavailable`` ＋「索引没接入」✓，**不许**说成"没有质检数据"✗。
"""

from __future__ import annotations

from typing import Any

from . import hooks

__all__ = [
    "MIN_BUCKET",
    "TOLERANCE",
    "SCAN_LIMIT",
    "stage_stats",
    "pass_by_steps",
    "gate",
    "summary_line",
]

#: 每个步数桶至少要几条**带 QC 结论**的跑次，才允许拿它当闸门
MIN_BUCKET = 5

#: 降一档步数后，QC 通过率允许的下滑（0.1 ＝ 10 个百分点）
TOLERANCE = 0.1

#: 一次分析最多读多少条 QC 明细（`outcome_index.samples` 硬上限 20000）
SCAN_LIMIT = 20000

_NOTE = (
    "QC 通过率按**产出它的引擎档**归因（归因键＝bridge 提交时登记的 records 表）；"
    "未归因的那些是云 provider 或早于归因表上线的跑次，不参与任何一族的口径"
)


def _index():
    """跑次索引模块（**由服务层登记** ✓，没登记 ⇒ ``None`` ✓）。

    ⚠️ 这里原先写 ``from .. import outcome_index`` ✓✗ —— **相对 import 一样是把引擎钉死在
    服务层上** ✓，只是当年的独立性判据 ② 只扫 ``node.module`` 且要求 ``level == 0`` ⇒
    相对写法**整类被放过** ✓✗（2026-11-08 补判据时才露头 ✓）。改成登记 + 如实 ``None`` ✓。
    """
    return hooks.get("run_index")


def _meta_of(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta")
    return meta if isinstance(meta, dict) else {}


def _stage_key(row: dict[str, Any]) -> str:
    return str(_meta_of(row).get("producerStage") or "").strip().lower()


#: 没登记时所有口统一这一句 ✓✗（别写"没有质检数据"✓ —— 那与"索引没接入"是两回事 ✓✗）
_ABSENT = ("跑次索引没接入（引擎走「服务层登记」通道 ✓，``app.main`` 启动时登记 ✓）"
           "⇒ 质检反哺判不了，不是「没有质检」")


def _qc_rows(limit: int = SCAN_LIMIT) -> list[dict[str, Any]]:
    index = _index()
    if index is None:
        return []
    return index.samples(kind="qc", limit=min(int(limit), SCAN_LIMIT))


def _steps_of(run_id: str, cache: dict[str, int | None]) -> int | None:
    """那次引擎生成用了几步（跑次不存在 / meta 没记 ⇒ ``None``，**不猜** ✓）。"""
    if not run_id:
        return None
    if run_id in cache:
        return cache[run_id]
    index = _index()
    if index is None:
        return None
    row = index.get(run_id)
    steps: int | None = None
    if row:
        raw = _meta_of(row).get("steps")
        try:
            steps = int(raw) if raw is not None and str(raw) != "" else None
        except (TypeError, ValueError):
            steps = None
    cache[run_id] = steps
    return steps


def stage_stats(*, limit: int = SCAN_LIMIT) -> dict[str, Any]:
    """按**引擎档**聚合质检结论：条数 / 通过率 / 失败 code 分布。

    回执里 ``attributed`` 与 ``unattributed`` **必须分开报** ✓：只说"20 条 QC"
    会让人以为这 20 条都能拿来判族 ✓✗。
    """
    index = _index()
    if index is None:
        return {
            "available": False,
            "qcTotal": 0,
            "attributed": 0,
            "unattributed": 0,
            "stages": [],
            "hint": _ABSENT,
            "note": _NOTE,
        }
    probe = index.stats(group_by="kind", kind="qc")
    if not probe.get("available"):
        return {
            "available": False,
            "qcTotal": 0,
            "attributed": 0,
            "unattributed": 0,
            "stages": [],
            "hint": probe.get("hint") or "索引库不可用",
            "note": _NOTE,
        }

    rows = _qc_rows(limit)
    buckets: dict[str, dict[str, Any]] = {}
    unattributed = 0
    for row in rows:
        stage = _stage_key(row)
        if not stage:
            unattributed += 1
            continue
        bucket = buckets.setdefault(stage, {
            "stage": stage, "total": 0, "passed": 0, "failed": 0, "codes": {},
        })
        bucket["total"] += 1
        if str(row.get("status")) == "ok":
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
        code = row.get("error_code")
        if code:
            bucket["codes"][code] = bucket["codes"].get(code, 0) + 1

    stages: list[dict[str, Any]] = []
    for bucket in buckets.values():
        total = bucket["total"]
        stages.append({
            "stage": bucket["stage"],
            "total": total,
            "passed": bucket["passed"],
            "failed": bucket["failed"],
            "passRate": round(bucket["passed"] / total, 4) if total else None,
            "codes": [
                {"code": code, "count": count}
                for code, count in sorted(bucket["codes"].items(),
                                          key=lambda kv: (-kv[1], kv[0]))
            ],
        })
    stages.sort(key=lambda item: (-item["total"], item["stage"]))
    return {
        "available": True,
        "qcTotal": int(probe.get("total") or 0),
        "scanned": len(rows),
        "attributed": sum(item["total"] for item in stages),
        "unattributed": unattributed,
        "stages": stages,
        "limit": min(int(limit), SCAN_LIMIT),
        "note": _NOTE,
    }


def pass_by_steps(stage: str, *, limit: int = SCAN_LIMIT) -> dict[str, Any]:
    """该族按 ``steps`` 分桶的 QC 通过率。

    只要 ``meta.steps`` 拿得到的那些 ✓；拿不到的单独计数进 ``missingSteps`` ✓
    （那一批不是"没通过"✓，是"这次比不了"✓）。
    """
    wanted = str(stage or "").strip().lower()
    if not wanted:
        return {"stage": "", "available": False, "buckets": [], "missingSteps": 0,
                "samples": 0, "hint": "没给引擎档"}
    if _index() is None:
        return {"stage": wanted, "available": False, "buckets": [], "missingSteps": 0,
                "samples": 0, "hint": _ABSENT}

    cache: dict[str, int | None] = {}
    buckets: dict[int, dict[str, Any]] = {}
    missing_steps = 0
    considered = 0
    for row in _qc_rows(limit):
        if _stage_key(row) != wanted:
            continue
        considered += 1
        steps = _steps_of(str(_meta_of(row).get("engineRunId") or ""), cache)
        if steps is None:
            missing_steps += 1
            continue
        bucket = buckets.setdefault(steps, {"steps": steps, "samples": 0, "passed": 0})
        bucket["samples"] += 1
        if str(row.get("status")) == "ok":
            bucket["passed"] += 1

    items = []
    for bucket in buckets.values():
        samples = bucket["samples"]
        items.append({
            "steps": bucket["steps"],
            "samples": samples,
            "passed": bucket["passed"],
            "passRate": round(bucket["passed"] / samples, 4) if samples else None,
        })
    items.sort(key=lambda item: item["steps"])
    return {
        "stage": wanted,
        "available": bool(items) or considered > 0,
        "buckets": items,
        "samples": considered,
        "missingSteps": missing_steps,
    }


def _brief(bucket: dict[str, Any] | None) -> dict[str, Any] | None:
    if not bucket:
        return None
    return {"steps": bucket["steps"], "samples": bucket["samples"],
            "passed": bucket["passed"], "passRate": bucket["passRate"]}


def gate(stage: str, high: int, low: int, *,
         min_bucket: int = MIN_BUCKET, tolerance: float = TOLERANCE) -> dict[str, Any]:
    """「从 ``high`` 步降到 ``low`` 步，质检会不会变差」——**降档候选的闸门**。

    三档回执 ✓（调用方按 ``verdict`` 分支 ✓，别看 ``why`` 的字面）：

    * ``insufficient``：任一侧带质检结论的跑次不足 ``min_bucket`` ⇒ **判不了** ✓
      （不是"通过" ✗ —— 把它当通过就是拿空样本降质 ✓✗）。候选照出 ✓，但把
      "闸门没数据"如实带出去 ✓。
    * ``blocked``：低档通过率比高档低出 ``tolerance`` 以上 ⇒ **别降** ✓。
    * ``pass``：低档通过率没有明显下滑 ⇒ 降档可以谈 ✓。
    """
    data = pass_by_steps(stage)
    if data.get("hint") == _ABSENT:
        #: ⚠️ 别落到下面那句「0 条带质检结论」✗ —— 那读起来像「质检没数据」✓，
        #: 真实原因是**索引没接入**⇒ 会把人朝"再攒点 QC 数据"的方向带 ✓✗。
        return {"verdict": "insufficient", "why": f"{stage}：{_ABSENT}",
                "high": None, "low": None}
    buckets = {item["steps"]: item for item in data.get("buckets") or []}
    try:
        hi_steps, lo_steps = int(high), int(low)
    except (TypeError, ValueError):
        return {"verdict": "insufficient", "why": f"{stage}：步数不是整数（{high!r}/{low!r}）⇒ 判不了",
                "high": None, "low": None}
    hi, lo = buckets.get(hi_steps), buckets.get(lo_steps)
    hi_n = int(hi["samples"]) if hi else 0
    lo_n = int(lo["samples"]) if lo else 0
    if not hi or not lo or hi_n < min_bucket or lo_n < min_bucket:
        return {
            "verdict": "insufficient",
            "why": (f"{stage}：{hi_steps} 步 {hi_n} 条 / {lo_steps} 步 {lo_n} 条带质检结论"
                    f"（每桶要 ≥{min_bucket} 条才敢当闸门）⇒ 这次降步数**不看质检**"),
            "high": _brief(hi),
            "low": _brief(lo),
        }

    rate_hi = hi["passed"] / hi_n
    rate_lo = lo["passed"] / lo_n
    drop = rate_hi - rate_lo
    payload = {
        "verdict": "blocked" if drop > tolerance else "pass",
        "high": _brief(hi),
        "low": _brief(lo),
        "drop": round(drop, 4),
        "tolerance": tolerance,
    }
    if drop > tolerance:
        payload["why"] = (
            f"{stage}：{hi_steps} 步质检通过率 {rate_hi:.0%}（{hi['passed']}/{hi_n}），"
            f"{lo_steps} 步只有 {rate_lo:.0%}（{lo['passed']}/{lo_n}），"
            f"掉 {drop * 100:.0f} 个百分点（容忍 {tolerance * 100:.0f}）⇒ 不建议降到 {lo_steps} 步"
        )
    else:
        payload["why"] = (
            f"{stage}：{hi_steps} 步 {rate_hi:.0%}（{hi['passed']}/{hi_n}）/ "
            f"{lo_steps} 步 {rate_lo:.0%}（{lo['passed']}/{lo_n}），"
            f"质检通过率没有明显下滑（{drop * 100:+.0f} 个百分点）⇒ 降档过质检关"
        )
    return payload


def summary_line(stats: dict[str, Any] | None = None) -> str:
    """给人看的一句话：质检有没有、能不能归因到族（写进 ``tuning.suggest`` 的 notes）。"""
    data = stats if stats is not None else stage_stats()
    if not data.get("available"):
        return f"质检反哺：{data.get('hint')} ⇒ 这次降步数**判不了**（不是通过）"
    if not data.get("attributed"):
        return (
            f"质检反哺：{data.get('qcTotal', 0)} 条质检里 0 条带引擎档位"
            "（云 provider 或归因表上线前的旧数据）⇒ 降步数闸门**判不了**（不是通过）"
        )
    parts = "、".join(f"{item['stage']} {item['passed']}/{item['total']}"
                     for item in data["stages"])
    return (
        f"质检反哺：{data['attributed']} 条质检已归因到引擎档（{parts}），"
        f"{data['unattributed']} 条未归因（云 provider / 归因表上线前）"
    )
