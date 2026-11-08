"""成片评测守卫（2026-11-07，`knowledge-loop.md` 缺口 8 的另一半）。

缺口 8 的原话是「评测面只有 4 份基准」，那 4 份全部在**产出层**（提示词/结构化输出），
**没有任何一条判据看**成片 —— 而本仓真正花钱、真正决定交付的就是成片。

这套守卫盯六类失效（全都不会报错，只会给一个"看着挺像"的分数）：

① **缺数据被折成数字**：没成的镜算 0 分、没人复核算 100 分，两种都能凑出一个漂亮总分；
② **分母取错**：缺陷率用"全部镜"还是"有 QC 的镜"作分母，数字完全不同（前者会因没出片而虚高）；
③ **重跑 ffmpeg**：评测本可以"顺手再 probe 一遍"，但那样会和 QC 页口径打架（同一镜两处黑场数不同）；
④ **幂等**：点五次评测落五条跑次 ⇒ "样本量"被自己的点击灌水；
⑤ **软失败**：库坏了抛异常 ⇒ 评测页 500，而它本该答"判不了"；
⑥ **接线**：跑次 `kind` 没登记 `eval` ⇒ 记账静默返回 False，页面显示"已记录"其实没记。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/film_check_test.py
"""
from __future__ import annotations

import ast
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

TMP = Path(tempfile.mkdtemp(prefix="film-check-"))
os.environ["OUTCOME_INDEX_DB"] = (TMP / "outcome.sqlite").as_posix()
os.environ["DATA_ROOT"] = TMP.as_posix()

from sqlalchemy import create_engine, insert, text  # noqa: E402

from app.agent.evaluation import film_check  # noqa: E402
from app.core.models import (episodes, metadata, qc_reviews,  # noqa: E402
                             storyboards, video_generations, video_quality_checks)
from app.core.response import now  # noqa: E402
from app.services import outcome_index  # noqa: E402

SRC_FILM = (BP / "app" / "agent" / "evaluation" / "film_check.py").read_text(encoding="utf-8")
SRC_ROUTER = (BP / "app" / "routers" / "evaluation.py").read_text(encoding="utf-8")

_RESULTS: list[tuple[str, bool, object]] = []
_EPISODE = 1


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _called_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def _fresh_db():
    engine = create_engine("sqlite:///" + (TMP / "darme.sqlite").as_posix())
    metadata.create_all(engine)
    return engine


def _episode(conn, number: int = 1) -> int:
    global _EPISODE
    stamp = now()
    eid = conn.execute(insert(episodes).values(
        drama_id=1, episode_number=number, title=f"EP{number:03d}",
        created_at=stamp, updated_at=stamp,
    )).inserted_primary_key[0]
    _EPISODE = int(eid)
    return _EPISODE


def _shot(conn, episode_id: int, number: int, duration: int = 5) -> int:
    stamp = now()
    return int(conn.execute(insert(storyboards).values(
        episode_id=episode_id, storyboard_number=number, duration=duration,
        created_at=stamp, updated_at=stamp,
    )).inserted_primary_key[0])


def _take(conn, shot_id: int, *, status: str = "completed",
          local_path: str = "static/x.mp4") -> int:
    stamp = now()
    return int(conn.execute(insert(video_generations).values(
        storyboard_id=shot_id, drama_id=1, provider="engine", status=status,
        duration=5, fps=24, local_path=local_path if status == "completed" else None,
        created_at=stamp, updated_at=stamp,
        completed_at=stamp if status == "completed" else None,
    )).inserted_primary_key[0])


def _check_row(conn, shot_id: int, gen_id: int, *, spec_score: float | None = 100,
               issues: list[dict] | None = None, overall: int = 95) -> int:
    stamp = now()
    dims = {}
    if spec_score is not None:
        dims["video_spec"] = {"score": spec_score, "checked": True, "notes": []}
    return int(conn.execute(insert(video_quality_checks).values(
        storyboard_id=shot_id, video_generation_id=gen_id, drama_id=1,
        episode_id=_EPISODE, overall_score=overall, status="passed",
        issues=json.dumps(issues or [], ensure_ascii=False),
        dimensions=json.dumps(dims, ensure_ascii=False),
        created_at=stamp, updated_at=stamp,
    )).inserted_primary_key[0])


def _review(conn, check_id: int, code: str, verdict: str) -> None:
    stamp = now()
    conn.execute(insert(qc_reviews).values(
        check_id=check_id, code=code, verdict=verdict,
        created_at=stamp, updated_at=stamp,
    ))


def _dims(report: dict) -> dict[str, dict]:
    return {item["name"]: item for item in report["dimensions"]}


def main() -> int:
    engine = _fresh_db()

    # ==================================================== ① 空集：判不了，不是 0 分
    with engine.connect() as conn:
        empty_id = _episode(conn, 9)
        conn.commit()
        blank = film_check.score_episode(conn, empty_id, record=False)
    check("① 一镜都没有 ⇒ verdict=empty", blank["verdict"] == "empty", blank)
    check("① 没有总量 ⇒ 总分为 None（不是 0）", blank["score"] is None, blank["score"])
    check("① 四维全部 judged=false",
          [item["judged"] for item in blank["dimensions"]] == [False] * 4, blank["dimensions"])
    check("① 缺项逐条点名", blank["missing"] == list(film_check.DIMENSIONS), blank["missing"])

    # ==================================================== ② 覆盖：只出部分镜
    with engine.connect() as conn:
        eid = _episode(conn, 1)
        shots = [_shot(conn, eid, n) for n in (1, 2, 3, 4)]
        conn.commit()
        g1 = _take(conn, shots[0])
        g2 = _take(conn, shots[1])
        conn.commit()
        partial = film_check.score_episode(conn, eid, record=False)
    dims = _dims(partial)
    check("② 覆盖分 = 2/4", dims["coverage"]["score"] == 50.0, dims["coverage"])
    check("② 缺片的镜号点名（3、4）", partial["missingShots"] == [3, 4],
          partial["missingShots"])
    check("② 没出的镜不算 0 分、也不进 spec 分母（此刻无 QC ⇒ judged=false）",
          dims["spec"]["judged"] is False and dims["spec"]["samples"] == 0, dims["spec"])

    # ==================================================== ③ 同一镜多次成片 ⇒ 只认最新
    with engine.connect() as conn:
        eid = _episode(conn, 2)
        shot = _shot(conn, eid, 1)
        old = _take(conn, shot, local_path="static/old.mp4")
        new = _take(conn, shot, local_path="static/new.mp4")
        conn.commit()
        _check_row(conn, shot, old, spec_score=10)      # 旧那条带劣质 QC
        _check_row(conn, shot, new, spec_score=100)     # 新那条是好的
        conn.commit()
        latest = film_check.score_episode(conn, eid, record=False)
    check("③ 只读最新一条成片（旧 QC 的 10 分不进来）",
          _dims(latest)["spec"]["score"] == 100.0, _dims(latest)["spec"])

    # ==================================================== ④ spec 维 = 各镜技术分均值
    with engine.connect() as conn:
        eid = _episode(conn, 3)
        a, b = _shot(conn, eid, 1), _shot(conn, eid, 2)
        conn.commit()
        ga, gb = _take(conn, a), _take(conn, b)
        conn.commit()
        _check_row(conn, a, ga, spec_score=100)
        _check_row(conn, b, gb, spec_score=80)
        conn.commit()
        two = film_check.score_episode(conn, eid, record=False)
    check("④ 规格分 = (100 + 80) / 2", _dims(two)["spec"]["score"] == 90.0, _dims(two)["spec"])
    check("④ 一镜缺陷都没有 ⇒ 缺陷面满分", _dims(two)["defects"]["score"] == 100.0,
          _dims(two)["defects"])
    check("④ 两镜都有 QC ⇒ 覆盖率满分", _dims(two)["coverage"]["score"] == 100.0)

    # ==================================================== ⑤ 缺陷面：error 才算，warning 不算
    with engine.connect() as conn:
        eid = _episode(conn, 4)
        a, b, c = _shot(conn, eid, 1), _shot(conn, eid, 2), _shot(conn, eid, 3)
        conn.commit()
        ga, gb, gc = _take(conn, a), _take(conn, b), _take(conn, c)
        conn.commit()
        _check_row(conn, a, ga, spec_score=60, issues=[
            {"dimension": "video_spec", "code": "black", "severity": "error", "message": "黑场 1.2s"},
            {"dimension": "video_spec", "code": "black", "severity": "error", "message": "黑场 0.9s"},
        ])
        _check_row(conn, b, gb, spec_score=100, issues=[
            {"dimension": "video_spec", "code": "fps", "severity": "warning", "message": "帧率偏低"},
        ])
        _check_row(conn, c, gc, spec_score=100)
        conn.commit()
        defects = film_check.score_episode(conn, eid, record=False)
    dd = _dims(defects)
    check("⑤ 1/3 镜有 error 缺陷 ⇒ 缺陷面 66.7", dd["defects"]["score"] == 66.7, dd["defects"])
    check("⑤ warning 级不算缺陷（b 镜只有 warning）",
          [item["code"] for item in defects["defectCodes"]] == ["black"], defects["defectCodes"])
    check("⑤ count 与 shots 同源（= 命中镜数，不是 issue 条数 ⇒ 可互相校验）",
          defects["defectCodes"][0]["count"] == 1
          and defects["defectCodes"][0]["shots"] == [a], defects["defectCodes"][0])
    check("⑤ 缺陷面的分母是有 QC 的镜（3），不是全部镜",
          dd["defects"]["samples"] == 3, dd["defects"])

    # ==================================================== ⑥ 人工复核：漏报不进命中率分母
    with engine.connect() as conn:
        eid = _episode(conn, 5)
        a, b = _shot(conn, eid, 1), _shot(conn, eid, 2)
        conn.commit()
        ga, gb = _take(conn, a), _take(conn, b)
        conn.commit()
        ca = _check_row(conn, a, ga, spec_score=60, issues=[
            {"dimension": "video_spec", "code": "black", "severity": "error", "message": "黑场"}])
        cb = _check_row(conn, b, gb, spec_score=60, issues=[
            {"dimension": "video_spec", "code": "freeze", "severity": "error", "message": "冻帧"}])
        conn.commit()
        _review(conn, ca, "black", "hit")
        _review(conn, ca, "freeze", "missed")
        _review(conn, cb, "freeze", "false_alarm")
        conn.commit()
        reviewed = film_check.score_episode(conn, eid, record=False)
    rd = _dims(reviewed)
    check("⑥ 命中率 = hit / (hit + 误报) = 1/2（漏报不进分母）",
          rd["review"]["score"] == 50.0 and reviewed["review"]["hitRate"] == 0.5, reviewed["review"])
    check("⑥ 漏报单独计数并进 notes（提示缺陷面偏乐观）",
          reviewed["review"]["missed"] == 1
          and any("漏报" in note for note in reviewed["notes"]), reviewed["notes"])

    # ==================================================== ⑦ 无复核 ⇒ 该维不参与总分
    check("⑦ 无复核时 review 维 judged=false", _dims(two)["review"]["judged"] is False)
    check("⑦ 无复核时总分只由其余三维等权（(100+90+100)/3）",
          two["score"] == 96.7, two["score"])
    check("⑦ 无复核时 verdict=partial（缺维点名）",
          two["verdict"] == "partial" and "review" in two["missing"], two)

    # ==================================================== ⑧ 分档：ready / blocked
    with engine.connect() as conn:
        eid = _episode(conn, 6)
        shot = _shot(conn, eid, 1)
        conn.commit()
        gen = _take(conn, shot)
        conn.commit()
        check_id = _check_row(conn, shot, gen, spec_score=0, issues=[
            {"dimension": "video_spec", "code": "black", "severity": "error", "message": "黑场"}])
        conn.commit()
        _review(conn, check_id, "black", "hit")
        _review(conn, check_id, "freeze", "hit")
        conn.commit()
        blocked = film_check.score_episode(conn, eid, record=False)
    check("⑧ 技术分崩了 ⇒ 总分低于 80、verdict=blocked",
          blocked["verdict"] == "blocked" and blocked["score"] < film_check.PASS_SCORE, blocked)
    check("⑧ blocked 也照样把四维分数给出来（不吞明细）",
          all(item["judged"] for item in blocked["dimensions"]), blocked["dimensions"])

    # ==================================================== ⑨ 落跑次：eval 类、幂等覆盖
    with engine.connect() as conn:
        eid = _episode(conn, 7)
        shot = _shot(conn, eid, 1)
        conn.commit()
        gen = _take(conn, shot)
        conn.commit()
        _check_row(conn, shot, gen, spec_score=100)
        conn.commit()
        first = film_check.score_episode(conn, eid, record=True)
        conn.commit()
        second = film_check.score_episode(conn, eid, record=True)
        conn.commit()
    check("⑨ 记账成功（recorded=true）", first["recorded"] is True, first["recorded"])
    check("⑨ 落的是 eval 类跑次", "eval" in outcome_index.KINDS)
    row = outcome_index.get(f"film:{eid}") or {}
    check("⑨ 跑次 run_id = film:<集>、stage=film、score 是 0–1",
          row.get("stage") == "film" and row.get("kind") == "eval"
          and isinstance(row.get("score"), float) and row["score"] <= 1.0, row)
    rows = outcome_index.query(limit=50, kind="eval")["items"]
    check("⑨ 评两次只有一条（幂等覆盖，不灌水样本）",
          len([item for item in rows if item.get("run_id") == f"film:{eid}"]) == 1,
          [item.get("run_id") for item in rows])
    check("⑨ meta 里带四维与 verdict（可复核，不是一个孤零零的分）",
          set((row.get("meta") or {}).get("dimensions") or {}) == set(film_check.DIMENSIONS),
          row.get("meta"))
    check("⑨ 重评不改 run_id（两次拿到的是同一条跑次）",
          (outcome_index.get(f"film:{eid}") or {}).get("run_id") == f"film:{eid}"
          and second["score"] == first["score"] == 100.0,
          (first["score"], second["score"]))

    with engine.connect() as conn:
        eid = _episode(conn, 8)
        conn.commit()
        skipped = film_check.score_episode(conn, eid, record=False)
    check("⑨ record=false ⇒ recorded=false 且查不到这条跑次",
          skipped["recorded"] is False and outcome_index.get(f"film:{eid}") is None, skipped["recorded"])

    # ==================================================== ⑩ 不重跑 ffmpeg（机械判据）
    calls = _called_names(SRC_FILM)
    check("⑩ 评测不调 probe/ffmpeg（读数，不重新量）",
          not ({"probe_video_technical_spec", "run_technical_qc", "_run"} & calls), sorted(calls))
    check("⑩ 评测不调模型/agent（它评的是已经出的片）",
          not ({"run_agent", "run_agent_with_instructions", "evaluate_case"} & calls),
          sorted(calls))
    imports: set[str] = set()
    for node in ast.walk(ast.parse(SRC_FILM)):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    check("⑩ 模块没有 import subprocess（想 probe 也 probe 不了）",
          "subprocess" not in imports, sorted(imports))

    # ==================================================== ⑪ 路由：参数校验与位置
    check("⑪ 路由已注册 /film/{episode_id}",
          '@router.post("/film/{episode_id}")' in SRC_ROUTER)
    check("⑪ 非正整数 ⇒ 400（参数错，不是「这一集不存在」）",
          '必须是正整数' in SRC_ROUTER)
    check("⑪ 路由导入了 score_episode",
          "from app.agent.evaluation.film_check import score_episode" in SRC_ROUTER)

    # ==================================================== ⑫ 库坏：答"判不了"，不抛
    broken = create_engine("sqlite:///" + (TMP / "broken.sqlite").as_posix())
    metadata.create_all(broken)
    broken.dispose()
    with create_engine("sqlite:///" + (TMP / "darme.sqlite").as_posix()).connect() as conn:
        conn.execute(text("DROP TABLE video_generations"))
        conn.commit()
        try:
            degraded = film_check.score_episode(conn, eid, record=False)
            crashed = False
        except Exception:  # noqa: BLE001
            degraded, crashed = {}, True
    check("⑫ 表被删 ⇒ 不抛异常（评测页不该 500）", crashed is False, crashed)
    check("⑫ 表被删 ⇒ 当成「一镜都没有」（empty），不编数据",
          isinstance(degraded, dict) and degraded.get("verdict") == "empty", degraded)

    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'✓' if ok else '✗'} {name}" + ("" if ok else f"  → {detail!r}"))
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    engine.dispose()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
