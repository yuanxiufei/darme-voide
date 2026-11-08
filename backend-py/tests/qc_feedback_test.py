"""质检 ⇒ 引擎参数选择的**归因与闸门**守卫（2026-11-07，`knowledge-loop.md` 缺口 6）。

为什么要有它
============
缺口 6 的原话是「QC 失败只触发 `qc_retry` 重跑这一镜，不影响后续**同模型**的参数选择」✓。
根因不在"少一条规则"，而在**归因断了** ✗：

* `technical_qc` 写的是 `stage="video_spec"` ✓（"这是规格判据"✓，不是"哪族出的"）；
* 引擎跑次写的是 `stage="h3"` / `"krea2"` ✓；
* 两边没有共同键 ⇒ 任何"这一族最近质量掉了"都无从谈起 ✗。

所以这套守卫盯三件事：

① **归因必须可核**：`link_record` / `record_of` 折得回 `(档, 引擎跑次)`，
   未登记一律 `None`（**不许**回落到"最近一次引擎档"之类的猜测 ✗）；
② **闸门三档必须分得开**：`insufficient`（样本不足 ⇒ 判不了）≠ `pass`（过了）≠ `blocked`（别降）；
   把 `insufficient` 当 `pass` 就是"拿空样本降质" ✗✗；
③ **接线是真的**：`bridge.run_job` 提交后登记 ✓、`_index_qc_run` 写回时查归因 ✓、
   `tuning._steps_candidates` 出候选前问闸门 ✓（AST 判据，改名字那一刻就红）。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/qc_feedback_test.py
"""
from __future__ import annotations

import ast
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

TMP = Path(tempfile.mkdtemp(prefix="qc-feedback-"))
#: 三个根全挪进夹具：这套会真写索引库（归因表与跑次同库）、真读 tuning 覆盖
os.environ["OUTCOME_INDEX_DB"] = (TMP / "outcome.sqlite").as_posix()
os.environ["ENGINE_TUNING_JSON"] = (TMP / "tuning.json").as_posix()
os.environ["KNOWLEDGE_LEDGER_DB"] = (TMP / "ledger.sqlite").as_posix()
os.environ["DATA_ROOT"] = TMP.as_posix()

from app.services import knowledge_ledger, outcome_index, qc_reviews, technical_qc  # noqa: E402
from app.services.engine import hooks as engine_hooks  # noqa: E402
from app.services.engine import qc_feedback, tuning  # noqa: E402

#: ⚠️ 引擎的三个源改成**由服务层登记** ✓✗（见 `engine/hooks.py`：引擎**不** import 服务层 ✓，
#: 独立性契约扫的是 `ast.walk` ⇒ 连函数里的 late import 也算越界 ✓✗）。线上登记点在
#: ``app/main.py`` 的 lifespan ✓；本套件直连引擎 ⇒ 自己登记 ✓。
#: 不登记的后果**不是报错而是降级** ✓✗：``suggest()`` 只会说"跑次索引没接入" ⇒ ⑥ 那条
#: 「拦下要留理由」拿到的是**没数据**的理由、不是**被闸门拦下**的理由 ✓（症状＝断言红，
#: 报错指向那条断言、不指向真正的原因 ✓✗）。
engine_hooks.provide(run_index=outcome_index, ledger=knowledge_ledger, qc_reviews=qc_reviews)

SRC_TUNING = (BP / "app" / "services" / "engine" / "tuning.py").read_text(encoding="utf-8")
SRC_BRIDGE = (BP / "app" / "services" / "engine" / "bridge.py").read_text(encoding="utf-8")
SRC_QC = (BP / "app" / "services" / "technical_qc.py").read_text(encoding="utf-8")

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _run(run_id: str, stage: str, status: str, *, code: str | None = None,
         score: float | None = None, steps: int | None = None,
         kind: str = "video") -> None:
    meta = {"steps": steps} if steps is not None else {}
    ok = outcome_index.record_run({
        "run_id": run_id, "kind": kind, "stage": stage, "status": status,
        "error_code": code, "score": score, "meta": meta,
    })
    if not ok:
        raise RuntimeError(f"夹具写入失败：{run_id}")


def _qc(run_id: str, stage: str | None, *, passed: bool, run: str | None,
        code: str = "black") -> None:
    meta: dict[str, object] = {}
    if stage:
        meta["producerStage"] = stage
    if run:
        meta["engineRunId"] = run
    ok = outcome_index.record_run({
        "run_id": run_id, "kind": "qc", "stage": "video_spec",
        "status": "ok" if passed else "failed",
        "error_code": None if passed else code,
        "score": 0.9 if passed else 0.3,
        "meta": meta,
    })
    if not ok:
        raise RuntimeError(f"夹具写入失败：{run_id}")


def _called_names(path: Path, func: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            names: set[str] = set()
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    target = inner.func
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        names.add(target.attr)
            return names
    raise AssertionError(f"{path.name} 里找不到函数 {func}")


def main() -> int:
    # ==================================================== ① 归因表：折得回去，折不回就 None
    check("① 未登记的记录不许编一个档位（返回 None）",
          outcome_index.record_of("video", 999) is None)
    check("① 登记成功", outcome_index.link_record("video", 101, "h3", run_id="engine:eng-1") is True)
    found = outcome_index.record_of("video", 101)
    check("① 折得回 (档, 引擎跑次)",
          (found or {}).get("stage") == "h3" and (found or {}).get("runId") == "engine:eng-1",
          found)
    outcome_index.link_record("video", 101, "wan22", run_id="engine:eng-2")
    again = outcome_index.record_of("video", 101)
    check("① 同一条记录重跑＝覆盖（不并列两条）",
          (again or {}).get("stage") == "wan22", again)
    check("① 未知记录类型一律拒（不进表、也不查）",
          outcome_index.link_record("shot", 1, "h3") is False
          and outcome_index.record_of("shot", 1) is None)
    check("① 缺记录 id 不写假行", outcome_index.link_record("video", "", "h3") is False)
    check("① 缺 stage 不写假行", outcome_index.link_record("video", 102, "") is False)

    # ==================================================== ② 质检写回：带上产出它的那一族
    outcome_index.link_record("video", 201, "krea2", run_id="engine:eng-9")
    technical_qc._index_qc_run(
        storyboard_id=7, video_generation_id=201, episode_id=3,
        status="failed", overall=42,
        spec={"blackSegments": [1], "freezeSegments": [], "duration": 4.0,
              "fps": 24.0, "hasAudio": True},
        issues=[{"code": "black", "message": "画面黑场 1.2s"}],
    )
    qc_row = outcome_index.get("qc:7:201") or {}
    qc_meta = qc_row.get("meta") or {}
    check("② 质检行仍落在 video_spec（这条语义不动）",
          qc_row.get("stage") == "video_spec", qc_row.get("stage"))
    check("② 冒出了产出它那一档（krea2）",
          qc_meta.get("producerStage") == "krea2" and qc_meta.get("engineRunId") == "engine:eng-9",
          qc_meta)
    technical_qc._index_qc_run(
        storyboard_id=8, video_generation_id=202, episode_id=3,
        status="passed", overall=95,
        spec={"blackSegments": [], "freezeSegments": [], "duration": 4.0,
              "fps": 24.0, "hasAudio": False},
        issues=[],
    )
    plain = (outcome_index.get("qc:8:202") or {}).get("meta") or {}
    check("② 云 provider 的片**不带** producerStage（没登记就不填 null）",
          "producerStage" not in plain and "engineRunId" not in plain, plain)

    # ==================================================== ③ 按族聚合：归因与未归因必须分开报
    _run("engine:e-step8", "zimage", "ok", score=0.80, steps=8)
    _run("engine:e-step4", "zimage", "ok", score=0.80, steps=4)
    stats = qc_feedback.stage_stats()
    check("③ 索引库读得出来", stats.get("available") is True, stats.get("hint"))
    check("③ 只把带档位的那 1 条算进来",
          stats.get("attributed") == 1 and stats.get("unattributed") == 1, stats)
    check("③ 按族给出条数与通过率",
          [item["stage"] for item in stats["stages"]] == ["krea2"]
          and stats["stages"][0]["total"] == 1 and stats["stages"][0]["passRate"] == 0.0,
          stats["stages"])
    check("③ 失败 code 有分布（不只是数一数几条）",
          stats["stages"][0]["codes"] == [{"code": "black", "count": 1}],
          stats["stages"][0]["codes"])
    check("③ 一句话能说清「几条归因了、几条没有」",
          "1 条质检已归因" in qc_feedback.summary_line(stats)
          and "1 条未归因" in qc_feedback.summary_line(stats),
          qc_feedback.summary_line(stats))

    # ==================================================== ④ 按步数分桶：拿不到步数 ≠ 没通过
    _qc("qc:9:301", "zimage", passed=True, run="engine:e-step8")
    _qc("qc:9:302", "zimage", passed=True, run="engine:e-step8")
    _qc("qc:9:303", "zimage", passed=False, run="engine:e-step4")
    _qc("qc:9:304", "zimage", passed=True, run="engine:missing")  # 跑次不在索引里
    _qc("qc:9:305", "zimage", passed=True, run=None)              # 压根没记跑次号
    by_steps = qc_feedback.pass_by_steps("zimage")
    buckets = {item["steps"]: item for item in by_steps["buckets"]}
    check("④ 8 步桶 2 条全过", buckets.get(8, {}).get("passed") == 2, by_steps)
    check("④ 4 步桶 1 条没过", buckets.get(4, {}).get("passRate") == 0.0, by_steps)
    check("④ 拿不到步数的单独计数（不是算成「没通过」）",
          by_steps.get("missingSteps") == 2, by_steps)
    check("④ 别的族的行不会混进来",
          all(item["steps"] in (4, 8) for item in by_steps["buckets"]), by_steps)
    check("④ 没给档位时如实答「判不了」",
          qc_feedback.pass_by_steps("")["available"] is False)

    # ==================================================== ⑤ 闸门三档：判不了 ≠ 通过
    thin = qc_feedback.gate("zimage", 8, 4)
    check("⑤ 4 步桶只有 1 条 ⇒ insufficient（不许当通过）",
          thin.get("verdict") == "insufficient", thin)
    check("⑤ insufficient 的 why 写清了「每桶要几条」",
          "≥5" in str(thin.get("why")), thin.get("why"))

    for index in range(5):
        _qc(f"qc:10:{400 + index}", "sdxl", passed=True, run="engine:s8")
    for index in range(5):
        _qc(f"qc:11:{500 + index}", "sdxl", passed=False, run="engine:s4")
    _run("engine:s8", "sdxl", "ok", score=0.80, steps=8)
    _run("engine:s4", "sdxl", "ok", score=0.80, steps=4)
    blocked = qc_feedback.gate("sdxl", 8, 4)
    check("⑤ 高档 5/5、低档 0/5 ⇒ blocked",
          blocked.get("verdict") == "blocked" and blocked.get("drop") == 1.0, blocked)
    check("⑤ blocked 的 why 有两个桶的真实条数（可复核）",
          "5/5" in str(blocked.get("why")) and "0/5" in str(blocked.get("why")),
          blocked.get("why"))

    for index in range(5):
        _qc(f"qc:12:{600 + index}", "qwen_image", passed=index < 4, run="engine:q8")
        _qc(f"qc:13:{700 + index}", "qwen_image", passed=index < 4, run="engine:q4")
    _run("engine:q8", "qwen_image", "ok", score=0.80, steps=8)
    _run("engine:q4", "qwen_image", "ok", score=0.80, steps=4)
    passed = qc_feedback.gate("qwen_image", 8, 4)
    check("⑤ 两档同为 4/5 ⇒ pass（没有明显下滑）",
          passed.get("verdict") == "pass" and passed.get("drop") == 0.0, passed)

    # ==================================================== ⑥ 接线：降步数候选必须过这道关
    check("⑥ tuning 出降步候选前问了闸门（AST）",
          "gate" in _called_names(BP / "app" / "services" / "engine" / "tuning.py", "_steps_candidates"),
          sorted(_called_names(BP / "app" / "services" / "engine" / "tuning.py", "_steps_candidates")))
    check("⑥ bridge 提交成功后登记归因（AST）",
          "_remember_record" in _called_names(BP / "app" / "services" / "engine" / "bridge.py", "run_job"),
          sorted(_called_names(BP / "app" / "services" / "engine" / "bridge.py", "run_job")))
    check("⑥ 登记走的是 outcome_index.link_record（AST）",
          "link_record" in _called_names(BP / "app" / "services" / "engine" / "bridge.py",
                                         "_remember_record"))
    check("⑥ 质检写回时查归因（AST）",
          "_producer_of" in _called_names(BP / "app" / "services" / "technical_qc.py", "_index_qc_run")
          and "record_of" in _called_names(BP / "app" / "services" / "technical_qc.py", "_producer_of"))

    #: ⑥b 归因**必须走接线口** ✓✗（2026-11-08 改）：原先 `_remember_record` 在函数里
    #: `from .. import outcome_index` —— 形式上躲开顶层依赖，方向仍是**引擎 → 服务层** ✓✗
    #: （独立性契约 ② 扫 `ast.walk`，函数里的 import 同样算越界）。改走 `hooks.get("run_index")`
    #: 后要验两件事，缺一件这条通路就还能退回旧形态 ✓✗：
    #: ① 登记了 ⇒ 真的交给它（参数逐项对得上，不是"喊了一声没带数据"）；
    #: ② 没登记 ⇒ **不许**偷偷回落到真库（那正是 late import 的形状），只许记一条 warn。
    from app.services.engine import bridge as engine_bridge

    class _SpyIndex:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def link_record(self, *args: object, **kwargs: object) -> bool:
            self.calls.append((args, kwargs))
            return True

    spy = _SpyIndex()
    engine_hooks.provide(run_index=spy)
    engine_bridge._remember_record("VideoTask", 4242, "h3", "eng-spy")
    check("⑥ 登记了跑次索引 ⇒ 归因真交给它（档/记录 id/引擎跑次三项逐项对得上）",
          spy.calls == [(("video", 4242, "h3"), {"run_id": "engine:eng-spy"})], spy.calls)
    engine_hooks.clear()
    engine_bridge._remember_record("VideoTask", 4343, "h3", "eng-none")
    check("⑥ 没登记 ⇒ **不**偷偷回落到真索引库（late import 的旧形态），也不抛",
          outcome_index.record_of("video", 4343) is None, outcome_index.record_of("video", 4343))
    engine_hooks.provide(run_index=outcome_index, ledger=knowledge_ledger, qc_reviews=qc_reviews)
    check("⑥ 源码里没把「样本不足」当通过（不许出现 insufficient→pass 的直通）",
          'or verdict.get("verdict") == "insufficient"' not in SRC_TUNING
          and 'verdict != "blocked"' not in SRC_TUNING)

    # 造一组"降步数不降分"的跑次：步数平台期成立，但质检在低档崩了
    for index in range(4):
        _run(f"engine:z8-{index}", "zimage", "ok", score=0.80, steps=8)
        _run(f"engine:z4-{index}", "zimage", "ok", score=0.80, steps=4)
    for index in range(3):
        _run(f"engine:zz8-{index}", "sdxl", "ok", score=0.80, steps=8)
        _run(f"engine:zz4-{index}", "sdxl", "ok", score=0.80, steps=4)
    suggestions = tuning.suggest(min_samples=8)
    steps_ids = [item["id"] for item in suggestions.get("candidates") or []
                 if item.get("rule") == "steps-diminishing"]
    check("⑥ sdxl 低档质检崩了 ⇒ 不出这条候选（当场拦下）",
          not any(":sdxl" in item for item in steps_ids), steps_ids)
    check("⑥ 拦下这件事必须留下理由，不是静默丢弃",
          any("不建议降到 4 步" in str(note) for note in suggestions.get("notes") or []),
          suggestions.get("notes"))
    kept = [item for item in suggestions.get("candidates") or []
            if item.get("rule") == "steps-diminishing" and item.get("stage") == "zimage"]
    if kept:
        verdict = (kept[0].get("qcGate") or {}).get("verdict")
        check("⑥ 放行的候选把闸门结论一起带出去（人能看到凭据）",
              verdict in ("pass", "insufficient"), kept[0].get("qcGate"))
    else:
        check("⑥ zimage 的候选被样本门槛挡住时，notes 要如实说",
              any("没有哪一族" in str(note) or "质检" in str(note)
                  for note in suggestions.get("notes") or []),
              suggestions.get("notes"))
    check("⑥ suggest 的 notes 里报了质检归因情况",
          any("质检反哺" in str(note) for note in suggestions.get("notes") or []),
          suggestions.get("notes"))

    # ==================================================== ⑦ 坏库降级：判不了就如实说，不炸主链路
    broken = Path(tempfile.mkdtemp(prefix="qc-feedback-bad-"))
    os.environ["OUTCOME_INDEX_DB"] = broken.as_posix()  # 一个目录 ⇒ sqlite 打不开
    broken_stats = qc_feedback.stage_stats()
    check("⑦ 库坏 ⇒ available=false（不是 0 条）", broken_stats.get("available") is False, broken_stats)
    check("⑦ 库坏时闸门答 insufficient 且 why 里点明判不了",
          qc_feedback.gate("zimage", 8, 4).get("verdict") == "insufficient")
    check("⑦ 库坏时 summary_line 明说是「判不了」",
          "判不了" in qc_feedback.summary_line(broken_stats), qc_feedback.summary_line(broken_stats))
    check("⑦ 库坏时 suggest 不抛（只是没有质检闸门）",
          isinstance(tuning.suggest(min_samples=8).get("candidates"), list))

    # ==================================================== ⑧ 没登记源：说「没接入」，不许说「没质检」
    #: ⚠️ 这两件事**看起来一样、修法完全不同** ✓✗：「索引没接入」⇒ 去查 `app.main` 的登记／
    #: 本套件的 `provide(...)`；「没有质检数据」⇒ 去攒 QC。混成一句会把人朝错方向带 ✓✗。
    engine_hooks.clear()
    absent_stats = qc_feedback.stage_stats()
    absent_steps = qc_feedback.pass_by_steps("zimage")
    absent_gate = qc_feedback.gate("zimage", 8, 4)
    absent_line = qc_feedback.summary_line(absent_stats)
    check("⑧ 没登记 ⇒ stage_stats 答 unavailable 且点明「索引没接入」",
          absent_stats.get("available") is False and "没接入" in str(absent_stats.get("hint")),
          absent_stats.get("hint"))
    check("⑧ 没登记 ⇒ pass_by_steps 也点明「没接入」（不是「这一族没样本」）",
          absent_steps.get("available") is False and "没接入" in str(absent_steps.get("hint")),
          absent_steps.get("hint"))
    check("⑧ 没登记 ⇒ gate 的 why 说的是「没接入」而不是「0 条带质检结论」",
          absent_gate.get("verdict") == "insufficient"
          and "没接入" in str(absent_gate.get("why"))
          and "0 条" not in str(absent_gate.get("why")),
          absent_gate.get("why"))
    check("⑧ 没登记 ⇒ summary_line 明说「判不了」并带出真因",
          "判不了" in absent_line and "没接入" in absent_line, absent_line)
    engine_hooks.provide(run_index=outcome_index, ledger=knowledge_ledger, qc_reviews=qc_reviews)
    #: 反验只判「不再说没接入」✓ —— 此刻 ⑦ 把 `OUTCOME_INDEX_DB` 指到了坏目录，索引库本来
    #: 就是坏的 ⇒ 不能要求 `available=True`（那是 ⑦ 的领域 ✗），只要求**归因**回到真因 ✓。
    restored = qc_feedback.stage_stats()
    check("⑧ 反验：登记回来后「没接入」这句话立刻消失（不留全局残态 ✗）",
          "没接入" not in str(restored.get("hint")), restored.get("hint"))

    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'✓' if ok else '✗'} {name}" + ("" if ok else f"  → {detail!r}"))
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
