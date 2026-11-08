"""质检**人工复核**与阈值命中率（`knowledge-loop.md` §4.3 最后一条缺口）守卫。

为什么要有它
============
``tuning.suggest()`` 的规则一二看的是「跑次失败聚集」与「步数平台期」，两条都是**机器可见**的；
第三条 ``qc-threshold-hit-rate`` 一直留在 ``notEvaluated`` 里，原因不是算不出来，而是
**分母不存在** —— 阈值报了一次黑场，是「真黑」还是「片头淡入误杀」，只有人看过成片才知道。
``video_quality_checks`` 原先没有复核字段 ⇒ 只有触发频次（分子），没有命中率。

本套盯的是这条链上的**每个口径**，而不是"端点能返回 200"：

① 落库：写一条复核、同 ``(checkId, code)`` 改判**不新增**（否则命中率被自己的犹豫稀释）；
② 拒绝：``verdict`` 不在三值里 / ``code`` 不在 ``technical_qc.ISSUE_CODES`` 闭集里 ⇒ 拒并带 why；
③ 口径：触发数按 ``(checkId, code)`` **去重**（与复核粒度同源）；漏报**不进**命中率分母；
④ 不编数：无复核时 ``hitRate`` 是 ``None`` 不是 ``0``（``0`` 会被读成"零误报"）；
⑤ 阈值映射不漂：``ISSUE_THRESHOLDS`` 的每个 code 都在闭集、每个 field 都在 ``TUNABLES``；
⑥ 方向：误报多 ⇒ 放宽（``blackminduration`` 调高）、漏报多 ⇒ 收紧（``fpsmin`` 调高）、
   负值区间不写反（``truepeakdb=-1`` 放宽是 **-0.8** 不是 -1.25）；
⑦ 门槛：复核条数 < ``MIN_REVIEWS`` ⇒ 一个候选都不出（只报还差几条）；
⑧ 候选恒 ``canApply=False``，且凭据是可核的 ``check:<id>``。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/qc_reviews_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"
sys.path.insert(0, str(BP))
os.environ.setdefault("PROXY_TO_NODE", "0")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

TMP = Path(tempfile.mkdtemp(prefix="qc-reviews-"))
#: 四个根全挪进夹具：本套真建库、真写复核、真算候选
os.environ["DATA_ROOT"] = TMP.as_posix()
os.environ["OUTCOME_INDEX_DB"] = (TMP / "outcome.sqlite").as_posix()
os.environ["ENGINE_TUNING_JSON"] = (TMP / "tuning.json").as_posix()
os.environ["KNOWLEDGE_LEDGER_DB"] = (TMP / "ledger.sqlite").as_posix()

from sqlalchemy import text  # noqa: E402

from app.core import db  # noqa: E402
from app.services import qc_reviews, technical_qc  # noqa: E402
from app.services.engine import hooks, tuning  # noqa: E402

#: ⚠️ 2026-11-07：`tuning` 读复核表已改成**注入口**（`hooks.get("qc_reviews")` ✓，
#: 生产由 `app/main.py` 启动时登记 ✓）—— 本套必须补同一步 ✗，否则 ⑦ 那组只会看到
#: 「质检复核没接入 ⇒ 命中率没有分母」而整组红（注入式接线的**可见症状** ✓，不是静默 ✓）。
hooks.provide(qc_reviews=qc_reviews)

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _reset() -> None:
    """清库（每个 case 从零开始 ✓ —— 否则上一个 case 的复核会把命中率带歪 ✗）。"""
    with db.engine.begin() as conn:
        conn.execute(text("DELETE FROM qc_reviews"))
        conn.execute(text("DELETE FROM video_quality_checks"))


def _qc(issues: list[dict], *, score: int = 80) -> int:
    """插一条质检记录（只有 ``created_at``/``updated_at`` 是 NOT NULL ✓）。"""
    stamp = "2026-11-07T00:00:00Z"
    with db.engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO video_quality_checks (storyboard_id, video_generation_id, issues, "
            "overall_score, status, created_at, updated_at) VALUES (1, 2, :issues, :score, "
            "'failed', :stamp, :stamp)"),
            {"issues": json.dumps(issues, ensure_ascii=False), "score": score, "stamp": stamp})
        row = conn.execute(text("SELECT last_insert_rowid()")).first()
        return int(row[0]) if row else 0


def _review(check_id: int, code: str, verdict: str, *, note: str = "") -> dict:
    with db.engine.begin() as conn:
        return qc_reviews.record_review(conn, check_id=check_id, code=code, verdict=verdict,
                                        note=note, reviewer="tester")


def _expect_raises(fn) -> str:
    try:
        fn()
    except ValueError as err:
        return str(err)
    return ""


def _by_code(summary: dict, code: str) -> dict | None:
    return next((item for item in summary.get("codes") or [] if item.get("code") == code), None)


def _suggest_candidates() -> list[dict]:
    return tuning.suggest().get("candidates") or []


# ===================================================================== ①
def case_schema() -> None:
    with db.engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(qc_reviews)")).fetchall()]
    for name in ("check_id", "code", "verdict", "note", "reviewer", "created_at", "updated_at"):
        check(f"① 表结构: 有 `{name}` 列 ✓", name in cols, cols)
    check("① 三值闭集就是 hit / false_alarm / missed ✓",
          sorted(qc_reviews.VERDICTS) == ["false_alarm", "hit", "missed"], qc_reviews.VERDICTS)
    check("① 复核门槛**低于**跑次门槛 ✓（人工成本更贵，不该要求 30 条片子）",
          qc_reviews.MIN_REVIEWS < tuning.MIN_SAMPLES,
          (qc_reviews.MIN_REVIEWS, tuning.MIN_SAMPLES))


# ===================================================================== ②
def case_record() -> None:
    _reset()
    check_id = _qc([{"code": "black", "message": "黑场 0.42s"},
                    {"code": "freeze", "message": "冻帧 1.3s"}])
    item = _review(check_id, "black", "false_alarm", note="片头淡入")
    check("② 写入: 回执带 id / code / verdict / 人话 ✓",
          item.get("id") and item.get("code") == "black"
          and item.get("verdict") == "false_alarm"
          and item.get("verdictLabel") == qc_reviews.VERDICTS["false_alarm"], item)

    again = _review(check_id, "black", "hit", note="放大看确实是黑")
    check("② 改判: 同 (checkId, code) **不新增**（id 不变 ✓ ⇒ 仍是同一条样本）",
          again.get("id") == item.get("id"), (item.get("id"), again.get("id")))
    with db.engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM qc_reviews")).scalar_one()
        listed = qc_reviews.list_reviews(conn, check_id=check_id)
    check("② 改判: 库里只有 1 条 ✓（追加会把命中率稀释 ✓✗）", total == 1, total)
    check("② 改判: 读出的是**新**结论并保留 note ✓",
          listed and listed[0]["verdict"] == "hit" and listed[0]["note"] == "放大看确实是黑",
          listed[0] if listed else None)

    why_verdict = _expect_raises(lambda: _review(check_id, "black", "probably"))
    check("② 拒绝: 未知 verdict ⇒ 带 why ✓（不让统计里长出第四种结论）",
          "verdict" in why_verdict, why_verdict)
    why_code = _expect_raises(lambda: _review(check_id, "black.night", "hit"))
    check("② 拒绝: 未知 code ⇒ 带 why 且点名闭集 ✓（否则那行永远没有阈值可调）",
          "ISSUE_CODES" in why_code, why_code)


# ===================================================================== ③④
def case_hit_rate() -> None:
    _reset()
    #: 同一次质检里两段黑场 ⇒ **一条**样本（触发数与复核粒度同源 ✓）
    first = _qc([{"code": "black", "message": "黑场 0.42s"},
                 {"code": "black", "message": "黑场 1.10s"},
                 {"code": "fps.low", "message": "帧率 12 < 15"}])
    second = _qc([{"code": "black", "message": "黑场 0.30s"}])
    third = _qc([{"code": "black", "message": "黑场 0.51s"}])

    _review(first, "black", "hit")
    _review(second, "black", "false_alarm")
    _review(third, "black", "missed", note="这段是过场黑卡，确实该报")

    summary = qc_reviews.hit_rate()
    black = _by_code(summary, "black") or {}
    check("③ 触发数: 三条记录 ⇒ 3 ✓（同一条里的两段黑场只算 1 ✓✗）",
          black.get("triggeredRecords") == 3, black.get("triggeredRecords"))
    check("③ 计数: hit 1 / false_alarm 1 / missed 1 ✓",
          (black.get("hit"), black.get("falseAlarm"), black.get("missed")) == (1, 1, 1), black)
    check("③ 命中率: 分母只含判过对错的 2 条 ⇒ 0.5 ✓", black.get("hitRate") == 0.5,
          black.get("hitRate"))
    check("③ 漏报率: 分母是全部复核 3 条 ⇒ 0.3333 ✓（漏报自己**不进**命中率分母 ✓）",
          black.get("missRate") == 0.3333, black.get("missRate"))
    check("③ 未复核的 code 只有触发数、命中率是 None ✓（不是 0 ✓✗）",
          (_by_code(summary, "fps.low") or {}).get("triggeredRecords") == 1
          and (_by_code(summary, "fps.low") or {}).get("hitRate") is None,
          _by_code(summary, "fps.low"))
    check("③ 记录面: 3 条记录均有复核 ⇒ unreviewed 0 ✓",
          summary.get("records") == {"total": 3, "reviewed": 3, "unreviewed": 0},
          summary.get("records"))

    _reset()
    empty = qc_reviews.hit_rate()
    check("④ 空库: ``available=True`` 且 codes 为空 ✓（库里没数据 ≠ 库坏了 ✓）",
          empty.get("available") is True and empty.get("codes") == [], empty.get("codes"))
    check("④ 空库: 全 0 而不是编一个 100% ✓",
          empty.get("records") == {"total": 0, "reviewed": 0, "unreviewed": 0},
          empty.get("records"))


# ===================================================================== ⑤
def case_pending() -> None:
    _reset()
    recorded = _qc([{"code": "audio.loudness", "message": "响度 -8.2 LUFS"},
                    {"code": "freeze", "message": "冻帧 1.3s"}])
    #: 规则维度的 issue（没有 code）不参与命中率 ✓
    _qc([{"dimension": "lip_sync", "message": "口型不稳"}, {"code": "black", "message": "黑场"}])

    with db.engine.begin() as conn:
        items = qc_reviews.pending(conn, limit=10)
    target = next((item for item in items if item["checkId"] == recorded), {})
    check("⑤ 待复核: 只列**有 code** 的 issue ✓（规则维度没有阈值可调 ✗）",
          [issue["code"] for issue in (target.get("issues") or [])] == ["audio.loudness", "freeze"],
          target.get("issues"))
    check("⑤ 待复核: 未判过 ⇒ review 为 None 且整条 reviewed=False ✓",
          all(issue.get("review") is None for issue in (target.get("issues") or []))
          and target.get("reviewed") is False, target.get("reviewed"))
    check("⑤ 待复核: 带人话标签（人不用去读 code）✓",
          (target.get("issues") or [{}])[0].get("label") == technical_qc.ISSUE_CODES["audio.loudness"],
          (target.get("issues") or [{}])[0].get("label"))

    _review(recorded, "audio.loudness", "hit")
    _review(recorded, "freeze", "false_alarm")
    with db.engine.begin() as conn:
        items = qc_reviews.pending(conn, limit=10)
    target = next((item for item in items if item["checkId"] == recorded), {})
    check("⑤ 待复核: 两个 code 都判过 ⇒ reviewed=True ✓（页面可以把它从待办里划掉）",
          target.get("reviewed") is True, target.get("reviewed"))
    check("⑤ 待复核: 已判的那条带出结论（能看到上次怎么判的）✓",
          {(issue["code"], (issue.get("review") or {}).get("verdict"))
           for issue in (target.get("issues") or [])}
          == {("audio.loudness", "hit"), ("freeze", "false_alarm")}, target.get("issues"))


# ===================================================================== ⑥
def case_mapping() -> None:
    codes = set(technical_qc.ISSUE_CODES)
    thresholds = set(technical_qc.ISSUE_THRESHOLDS)
    check("⑥ 映射: 每个 code ⊆ 闭集 ✓（映射表里不许有没人认得的类别 ✗）",
          thresholds <= codes, sorted(thresholds - codes))
    bad = [code for code, item in technical_qc.ISSUE_THRESHOLDS.items()
           if ("qc", item["field"]) not in tuning.TUNABLES]
    check("⑥ 映射: 每个 field 都在 ``TUNABLES`` 的 qc 档里 ✓（否则候选指向一个改不了的字段 ✗✗）",
          not bad, bad)
    bad_dir = [code for code, item in technical_qc.ISSUE_THRESHOLDS.items()
               if item.get("relax") not in ("raise", "lower")]
    check("⑥ 映射: ``relax`` 只有 raise/lower 两种 ✓", not bad_dir, bad_dir)
    bad_aux = [(code, item["aux"]) for code, item in technical_qc.ISSUE_THRESHOLDS.items()
               if item.get("aux") and ("qc", item["aux"]) not in tuning.TUNABLES]
    check("⑥ 映射: ``aux`` 也必须在 qc 档里 ✓（提示人去调一个不存在的字段等于害人 ✓✗）",
          not bad_aux, bad_aux)
    check("⑥ 映射: 探测失败的 code **有意不登记** ✓（改阈值只会把 ffprobe 问题藏起来 ✗✗）",
          "spec.unreadable" not in thresholds and "spec.probe_failed" not in thresholds,
          sorted(thresholds))


# ===================================================================== ⑦
def case_candidates() -> None:
    _reset()
    saved = qc_reviews.MIN_REVIEWS
    try:
        qc_reviews.MIN_REVIEWS = 4
        #: ① 误报多 ⇒ 放宽：``blackminduration`` 调高（放过更长的黑场）
        for _ in range(4):
            _review(_qc([{"code": "black", "message": "黑场 0.3s"}]), "black", "false_alarm")
        found = [c for c in _suggest_candidates() if c.get("rule") == "qc-threshold-hit-rate"]
        black = next((c for c in found if c.get("field") == "blackminduration"), None)
        current = tuning.resolve("qc", "blackminduration", 0.25)[0]
        check("⑦ 误报多: 出候选且指向 `blackminduration` ✓", black is not None, found)
        if black:
            check("⑦ 误报多: 方向是**放宽**（阈值调高 ⇒ 少误杀）✓",
                  float(black["proposed"]) > float(current)
                  and black.get("direction") == "relax",
                  (current, black.get("proposed")))
            check("⑦ 候选恒不可应用 ✓（影子验证没做之前不许自动改行为 ✓✗）",
                  black.get("canApply") is False, black.get("canApply"))
            check("⑦ 样本量如实（samples＝复核条数，不是触发条数）✓",
                  black.get("samples") == 4, black.get("samples"))
            check("⑦ 凭据可核: evidence 是 `check:<id>` ✓",
                  black.get("evidence") and all(str(item).startswith("check:")
                                                for item in black["evidence"]),
                  black.get("evidence"))
            check("⑦ why 里写明倍率是**起手值**而不是实测结论 ✓（不把建议说成证据 ✓）",
                  "起手值" in str(black.get("why")), black.get("why"))

        #: ② 漏报多 ⇒ 收紧：``fpsmin`` 是下限 ⇒ 收紧＝调高（与「放宽＝调低」相反 ✓）
        _reset()
        for _ in range(4):
            _review(_qc([{"code": "fps.low", "message": "帧率 12"}]), "fps.low", "missed")
        found = [c for c in _suggest_candidates() if c.get("rule") == "qc-threshold-hit-rate"]
        fps = next((c for c in found if c.get("field") == "fpsmin"), None)
        fps_current = tuning.resolve("qc", "fpsmin", 15)[0]
        check("⑦ 漏报多: 出候选且方向是**收紧**（下限调高 ⇒ 更容易报）✓",
              fps is not None and float(fps["proposed"]) > float(fps_current)
              and fps.get("direction") == "tighten",
              (fps_current, fps.get("proposed") if fps else None))

        #: ③ 负值区间：``truepeakdb`` 默认 -1，「放宽（调高）」⇒ -0.8，**不是** -1.25 ✓✗
        _reset()
        for _ in range(4):
            _review(_qc([{"code": "audio.true_peak", "message": "真峰 -0.2 dBTP"}]),
                    "audio.true_peak", "false_alarm")
        found = [c for c in _suggest_candidates() if c.get("rule") == "qc-threshold-hit-rate"]
        peak = next((c for c in found if c.get("field") == "truepeakdb"), None)
        check("⑦ 负值: `truepeakdb` 放宽是**变大**（-1 → -0.8 ✓，写成 ×1.25 得 -1.25 是**更严** ✗✗）",
              peak is not None and float(peak["proposed"]) > -1.0
              and float(peak["proposed"]) < 0.0,
              peak.get("proposed") if peak else None)

        #: ④ 样本不足 ⇒ 一个候选都不出，但**必须**说还差几条（难 ≠ 没问题 ✓）
        _reset()
        for _ in range(3):
            _review(_qc([{"code": "freeze", "message": "冻帧 2s"}]), "freeze", "false_alarm")
        result = tuning.suggest()
        found = [c for c in (result.get("candidates") or [])
                 if c.get("rule") == "qc-threshold-hit-rate"]
        note = next((item for item in (result.get("notEvaluated") or [])
                     if item.get("rule") == "qc-threshold-hit-rate"), {})
        check("⑦ 样本不足: 3 条 < 门槛 4 ⇒ 不出候选 ✓",
              not [c for c in found if c.get("field") == "freezeminduration"], found)
        check("⑦ 样本不足: 但如实报门槛与已复核数 ✓（沉默会被读成\"没问题\"✓✗）",
              "门槛" in str(note.get("why")), note.get("why"))

        #: ⑤ 没有阈值可映射的 code：即便误报满天飞也不许出候选 ✓
        _reset()
        for _ in range(4):
            _review(_qc([{"code": "spec.unreadable", "message": "ffprobe 读不出"}]),
                    "spec.unreadable", "false_alarm")
        found = [c for c in _suggest_candidates()
                 if c.get("rule") == "qc-threshold-hit-rate"]
        check("⑦ 无阈值的 code: 不出候选 ✓（那是环境问题，改阈值等于把探测失败藏起来 ✗✗）",
              not found, found)
    finally:
        qc_reviews.MIN_REVIEWS = saved


# ===================================================================== ⑧
def case_delete() -> None:
    _reset()
    check_id = _qc([{"code": "black", "message": "黑场 0.4s"}])
    item = _review(check_id, "black", "hit")
    with db.engine.begin() as conn:
        ok = qc_reviews.delete_review(conn, int(item["id"]))
        again = qc_reviews.delete_review(conn, int(item["id"]))
    check("⑧ 撤销: 删掉返回 True ✓，再删返回 False ✓（不静默成功 ✓）", ok and not again,
          (ok, again))
    summary = qc_reviews.hit_rate()
    check("⑧ 撤销: 命中率回到 None ✓（撤销要真的把分母拿掉 ✓）",
          (_by_code(summary, "black") or {}).get("hitRate") is None,
          _by_code(summary, "black"))


# ===================================================================== ⑨
def case_http() -> None:
    """HTTP 层：端点存在、信封形状对、400 与 404 说得清。

    为什么服务层测完还要测一遍路由：`qc_reviews.hit_rate()` 绿只证明**口径**对 ✓，
    而前端 `knowledge-loop.vue` 拿到的是信封里的 `data` ✓ —— 少一层 `data`、
    或者 `POST` 的字段名从 `checkId` 漂成 `check_id`（前端送的是驼峰），
    服务层全都看不出来 ✗（2026-11-07 这页就是靠这条才发现要两边都收 ✓）。
    """
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import app  # noqa: PLC0415

    client = TestClient(app)
    _reset()

    got = client.get("/api/v1/knowledge-loop/qc/hit-rate")
    data = (got.json() or {}).get("data") or {}
    check("⑨ HTTP: GET /qc/hit-rate 200 且信封里两率/门槛齐 ✓",
          got.status_code == 200 and data.get("available") is True
          and data.get("minReviews") == qc_reviews.MIN_REVIEWS and "codes" in data,
          (got.status_code, data))

    check_id = _qc([{"code": "black", "message": "黑场 0.4s"}])
    posted = client.post("/api/v1/knowledge-loop/qc/reviews",
                         json={"checkId": check_id, "code": "black",
                               "verdict": "false_alarm", "note": "片头淡入"})
    body = (posted.json() or {}).get("data") or {}
    check("⑨ HTTP: POST /qc/reviews 收**驼峰** checkId 并回 review ＋ 最新命中率 ✓",
          posted.status_code == 200
          and (body.get("review") or {}).get("verdict") == "false_alarm"
          and (body.get("hitRate") or {}).get("available") is True,
          (posted.status_code, body))
    review_id = (body.get("review") or {}).get("id")

    bad = client.post("/api/v1/knowledge-loop/qc/reviews",
                      json={"checkId": check_id, "code": "black", "verdict": "wrong"})
    check("⑨ HTTP: 非法 verdict ⇒ 400 且带 why ✓（不静默落一条废结论 ✗）",
          bad.status_code == 400 and bool(bad.json()), (bad.status_code, bad.json()))
    bad_code = client.post("/api/v1/knowledge-loop/qc/reviews",
                           json={"checkId": check_id, "code": "__nope__", "verdict": "hit"})
    check("⑨ HTTP: 闭集外的 code ⇒ 400 ✓（否则长出一堆没有阈值可调的行 ✗）",
          bad_code.status_code == 400, (bad_code.status_code, bad_code.json()))

    listing = client.get(f"/api/v1/knowledge-loop/qc/reviews?checkId={check_id}")
    items = ((listing.json() or {}).get("data") or {}).get("items") or []
    check("⑨ HTTP: GET /qc/reviews 能按 checkId 过滤到刚才那条 ✓",
          listing.status_code == 200 and [item.get("code") for item in items] == ["black"],
          items)

    pending = client.get("/api/v1/knowledge-loop/qc/pending?limit=5")
    pdata = (pending.json() or {}).get("data") or {}
    check("⑨ HTTP: GET /qc/pending 带 unreviewed 与三值字典 ✓（页面靠它画按钮）",
          pending.status_code == 200
          and isinstance(pdata.get("unreviewed"), int)
          and set(pdata.get("verdicts") or {}) == set(qc_reviews.VERDICTS),
          pdata.get("verdicts"))

    gone = client.delete(f"/api/v1/knowledge-loop/qc/reviews/{review_id}")
    again = client.delete(f"/api/v1/knowledge-loop/qc/reviews/{review_id}")
    check("⑨ HTTP: DELETE 先 200 再 404 ✓（撤销不静默成功 ✓）",
          gone.status_code == 200 and again.status_code == 404,
          (gone.status_code, again.status_code))


def main() -> int:
    case_schema()
    case_record()
    case_hit_rate()
    case_pending()
    case_mapping()
    case_candidates()
    case_delete()
    case_http()

    failures = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAIL {name}  ← {detail!r}")
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
