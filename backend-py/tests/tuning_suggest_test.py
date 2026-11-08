"""调参**候选**（`tuning.suggest`）守卫（2026-11-06）—— `knowledge-loop.md` §4.3 的另一半。

为什么要有它
============
`tuning` 落地后闭环仍缺一半：覆盖层能改行为了，但「该改成多少」还得人去翻跑次。
`suggest()` 把 `outcome_index` 的统计折成候选（失败聚集 / 步数已到平台期）。

它**一行都不写**，所以这套守卫盯的是「别让它偷偷改行为」与「别让它编结论」：

① 写口隔离：除 `_write`/`set_field`/`clear_field`/`reset` 外，没有任何函数调用它们（AST）；
② 样本量闸门：< `MIN_SAMPLES` 只报；`canApply` **恒 False**（影子验证没做）；
③ 归因别越界：失败分散时不许归成一个结论；`meta` 没 `steps` 就不许出步数建议；
④ 凭据可核：候选带的 run_id 必须过 `knowledge_ledger.check_evidence`（不是编的字符串）；
⑤ 抽样要如实：> 500 条时 `sampled=true`（把"扫了 500 条"读成"总共 500 条"就会绕过闸门①）。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/tuning_suggest_test.py
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

TMP = Path(tempfile.mkdtemp(prefix="tuning-suggest-"))
#: 三个根全挪进夹具：这套会真写 tuning.json，也会真建索引库
os.environ["OUTCOME_INDEX_DB"] = (TMP / "outcome.sqlite").as_posix()
os.environ["ENGINE_TUNING_JSON"] = (TMP / "tuning.json").as_posix()
os.environ["KNOWLEDGE_LEDGER_DB"] = (TMP / "ledger.sqlite").as_posix()
os.environ["DATA_ROOT"] = TMP.as_posix()

from app.services import knowledge_ledger, outcome_index  # noqa: E402
from app.services.engine import hooks as engine_hooks  # noqa: E402
from app.services.engine import tuning  # noqa: E402

#: ⚠️ 引擎不再 import 服务层 ✓✗（独立性契约 ② 判的就是这个 ✓）—— 三个源改成**由服务层登记** ✓，
#: 线上的登记点在 ``app/main.py`` 启动钩子 ✓；本套件**直连引擎** ⇒ 得自己登记 ✓，
#: 否则 ``set_field`` 会按"台账没接入 ⇒ 证据核不了 ⇒ 按核不过处理"拒写 ✓（症状是本节读
#: ``tuning.json`` 时 FileNotFoundError ✓，报错指向路径、不指向真正的原因 ✓✗）。
engine_hooks.provide(run_index=outcome_index, ledger=knowledge_ledger)

SRC = (BP / "app" / "services" / "engine" / "tuning.py").read_text(encoding="utf-8")
_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _run(run_id: str, stage: str, status: str, *, code: str | None = None,
         score: float | None = None, steps: int | None = None, kind: str = "image") -> None:
    meta = {"steps": steps} if steps is not None else {}
    ok = outcome_index.record_run({
        "run_id": run_id, "kind": kind, "stage": stage, "status": status,
        "error_code": code, "score": score, "meta": meta,
    })
    if not ok:
        raise RuntimeError(f"夹具写入失败：{run_id}")


def _by_id(candidates: list[dict], candidate_id: str) -> dict | None:
    return next((c for c in candidates if c.get("id") == candidate_id), None)


def _for_stage(candidates: list[dict], stage: str) -> list[dict]:
    return [c for c in candidates if c.get("stage") == stage]


def main() -> int:
    tuning.path().unlink(missing_ok=True)

    # ==================================================== ① 空库：不抛、不编
    empty = tuning.suggest()
    check("① 空库：不抛异常、候选为空", empty.get("candidates") == [], empty.get("candidates"))
    check("① 空库仍 available=true（读得出来，只是没有跑次）", empty.get("available") is True, empty)
    check("① 如实列出三处「算不了的」维度（没检查 ≠ 没问题）",
          [item.get("rule") for item in empty.get("notEvaluated") or []]
          == ["qc-threshold-hit-rate", "sampler-schedule-score", "shadow-validation"],
          empty.get("notEvaluated"))
    check("① 没有覆盖文件时 overrides 为空", empty.get("overrides") == [], empty.get("overrides"))

    # 索引库坏掉（路径是个目录）：必须 available=false 并写清「不是 0 条」
    good_db = os.environ["OUTCOME_INDEX_DB"]
    blocked = TMP / "blocked-db"
    blocked.mkdir()
    os.environ["OUTCOME_INDEX_DB"] = blocked.as_posix()
    broken = tuning.suggest()
    os.environ["OUTCOME_INDEX_DB"] = good_db
    check("① 索引库读不出 ⇒ available=false 且候选为空",
          broken.get("available") is False and broken.get("candidates") == [], broken)
    check("① 坏库不许静默成 0 条：notes 要说明原因",
          any("索引库" in str(note) or "不可用" in str(note) for note in broken.get("notes") or []),
          broken.get("notes"))

    # ==================================================== ② 失败聚集
    for index in range(30):
        _run(f"fail-seedvr2-{index}", "seedvr2", "failed", code="CUDA out of memory")
    clustered = tuning.suggest()
    cand = _by_id(clustered.get("candidates") or [], "failure-cluster:seedvr2")
    check("② 30 条同一错误码 ⇒ 报出失败聚集候选", cand is not None, clustered.get("candidates"))
    cand = cand or {}
    check("② 候选写清样本数/占比/错误码",
          cand.get("samples") == 30 and cand.get("share") == 1.0
          and cand.get("errorCode") == "CUDA out of memory", cand)
    check("② 失败聚集不指向某个可调字段（先归因，别先动 steps）",
          cand.get("field") is None and cand.get("canApply") is False, cand)
    check("② 闸门如实：样本 ok / 影子 n/a（没字段就没影子）",
          cand.get("gate") == {"samples": "ok", "shadow": "n/a", "rollback": "n/a"},
          cand.get("gate"))
    check("② 没能被核的凭据不写：证据过 check_evidence",
          bool(cand.get("evidence"))
          and knowledge_ledger.check_evidence(cand.get("evidence") or []).get("ok") is True,
          knowledge_ledger.check_evidence(cand.get("evidence") or []))
    check("② 扫到的条数与失败总数都写清、未抽样 ⇒ sampled=false",
          cand.get("scanned") == 30 and cand.get("failedTotal") == 30
          and cand.get("sampled") is False, cand)

    # ==================================================== ③ 分散不归因
    for code in ("A", "B", "C"):
        for index in range(5):
            _run(f"spread-flux-{code}-{index}", "flux", "failed", code=f"ERR-{code}")
    spread = tuning.suggest(min_samples=15)
    check("③ 三类各占 1/3（<50%）⇒ 不归成一个结论",
          _for_stage(spread.get("candidates") or [], "flux") == [],
          _for_stage(spread.get("candidates") or [], "flux"))

    # ==================================================== ④ 样本量闸门
    for index in range(20):
        _run(f"few-wan22-{index}", "wan22", "failed", code="shape mismatch")
    few_default = tuning.suggest()
    check("④ 20 条 < 默认 30 ⇒ 只报不应用（不出候选）",
          _for_stage(few_default.get("candidates") or [], "wan22") == [],
          _for_stage(few_default.get("candidates") or [], "wan22"))
    few_lowered = tuning.suggest(min_samples=20)
    check("④ 同一批数据把门槛降到 20 ⇒ 才报（闸门真按参数走）",
          _by_id(few_lowered.get("candidates") or [], "failure-cluster:wan22") is not None,
          few_lowered.get("candidates"))
    check("④ MIN_SAMPLES 就是文档写的那道闸门（30）", tuning.MIN_SAMPLES == 30, tuning.MIN_SAMPLES)

    # ==================================================== ⑤ 抽样如实
    for index in range(501):
        _run(f"big-bigfail-{index}", "bigfail", "failed", code="boom")
    big = tuning.suggest()
    big_cand = _by_id(big.get("candidates") or [], "failure-cluster:bigfail") or {}
    check("⑤ 501 条失败 ⇒ scanned=500 且 sampled=true（别把抽样读成总数）",
          big_cand.get("scanned") == 500 and big_cand.get("failedTotal") == 501
          and big_cand.get("sampled") is True, big_cand)

    # ==================================================== ⑥ 步数平台期
    for index in range(10):
        _run(f"zimage-30-{index}", "zimage", "ok", score=0.900, steps=30)
        _run(f"zimage-20-{index}", "zimage", "ok", score=0.895, steps=20)
    flat = tuning.suggest(min_samples=10)
    steps_cand = _by_id(flat.get("candidates") or [], "steps-diminishing:zimage:steps")
    check("⑥ 30 步比 20 步只多 0.005 分 ⇒ 建议降回 20 步", steps_cand is not None,
          flat.get("candidates"))
    steps_cand = steps_cand or {}
    check("⑥ 候选给出 current/proposed 与差距",
          steps_cand.get("current") == 30 and steps_cand.get("proposed") == 20
          and float(steps_cand.get("gain") or 1) < 0.01, steps_cand)
    check("⑥ 影子验证没做 ⇒ canApply 恒 False（闸门②不许被布尔值顶替）",
          steps_cand.get("canApply") is False
          and (steps_cand.get("gate") or {}).get("shadow") == "pending",
          steps_cand.get("gate"))
    check("⑥ 凭据两侧都取（多步与次高档），且真能核",
          bool(steps_cand.get("evidence"))
          and knowledge_ledger.check_evidence(steps_cand.get("evidence") or []).get("ok") is True,
          steps_cand.get("evidence"))

    # ==================================================== ⑦ 真有增益就不提
    for index in range(10):
        _run(f"qwen-30-{index}", "qwen_image", "ok", score=0.95, steps=30)
        _run(f"qwen-20-{index}", "qwen_image", "ok", score=0.80, steps=20)
    gain = tuning.suggest(min_samples=10)
    check("⑦ 多走 10 步买到 0.15 分 ⇒ 不提降步数（别为了省时间砍质量）",
          _by_id(gain.get("candidates") or [], "steps-diminishing:qwen_image:steps") is None,
          _for_stage(gain.get("candidates") or [], "qwen_image"))

    # ==================================================== ⑧ 没有 steps 维度就不猜
    for index in range(20):
        _run(f"krea2-{index}", "krea2", "ok", score=0.9)
    nometa = tuning.suggest(min_samples=10)
    check("⑧ meta 里没有 steps ⇒ 不出步数建议（不拿 elapsed 反推）",
          _by_id(nometa.get("candidates") or [], "steps-diminishing:krea2:steps") is None,
          _for_stage(nometa.get("candidates") or [], "krea2"))

    # ==================================================== ⑨ 已覆盖成同一值 ⇒ 不再提
    for index in range(10):
        _run(f"sdxl-30-{index}", "sdxl", "ok", score=0.90, steps=30)
        _run(f"sdxl-20-{index}", "sdxl", "ok", score=0.897, steps=20)
    written = tuning.set_field("sdxl", "steps", 20, basis="测试：平台期",
                               evidence=["sdxl-30-0"], by="test")
    check("⑨ 前提：先把 20 写成覆盖（证据用真 run_id）", written.get("ok") is True, written)
    applied = tuning.suggest(min_samples=10)
    check("⑨ 已经是 20 步了 ⇒ 不再重复建议",
          _by_id(applied.get("candidates") or [], "steps-diminishing:sdxl:steps") is None,
          _for_stage(applied.get("candidates") or [], "sdxl"))
    check("⑨ 覆盖列表里能看见这一条", "sdxl.steps" in (applied.get("overrides") or []),
          applied.get("overrides"))

    # ==================================================== ⑩ 只读：一个字节都不写
    before = tuning.path().read_text(encoding="utf-8")
    tuning.suggest()
    tuning.suggest(min_samples=5)
    after = tuning.path().read_text(encoding="utf-8")
    check("⑩ 连跑两次 suggest，tuning.json 逐字节不变", before == after, "文件被改了")
    check("⑩ 也没生出 `_backups/`（备份只在真写上发生）",
          not (tuning.path().parent / "_backups").exists(), "多了备份目录")

    # ==================================================== ⑪ AST：写口只有四个
    tree = ast.parse(SRC)
    writers = {"_write", "set_field", "clear_field", "reset"}
    leaks: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name in writers:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                target = inner.func
                name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
                if name in writers:
                    leaks.append(f"{node.name} -> {name}")
    check("⑪ 除四个写口自身外，没有任何函数调用它们（建议只读）", leaks == [], leaks)

    # ==================================================== ⑫ HTTP 面接线
    from app.routers import knowledge_loop as kl  # noqa: E402

    routes = {(getattr(route, "path", ""), tuple(sorted(getattr(route, "methods", ()) or ())))
              for route in kl.router.routes}
    check("⑫ `GET /tuning/suggest` 真挂在路由表上",
          any(path.endswith("/tuning/suggest") and methods == ("GET",)
              for path, methods in routes), sorted(routes))
    payload = json.loads(kl.get_tuning_suggest(min_samples=10).body)
    check("⑫ 端点走成功信封，且 data 结构与 suggest 一致",
          payload.get("code") == 200 and isinstance(payload.get("data"), dict)
          and "candidates" in payload["data"], payload.get("code"))

    failed = [name for name, ok, _ in _RESULTS if not ok]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  ↦ {detail!r}"))
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
