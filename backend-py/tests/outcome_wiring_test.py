"""跑次落账（`outcome_index` 的**输入侧**）—— 在此之前「调参建议」在真机上是瞎的。

为什么要有它
============
`tuning.suggest()` 的「失败聚集 / 步数平台期」读 `outcome_index`，而 2026-11-06 之前
**没有任何生产代码往里写 image/video/qc 跑次**（只有 `ingest_trace` 灌 agent trace）
⇒ 两条规则在真机上永远只会答「没有哪一族的失败集中」—— 看着像"没问题"，
其实是**没人喂它**。自测里那些行全是夹具造的假行，绿得没有意义。

这条守卫盯的是「真接上、且接对」：

① 两处写点都在（AST）：引擎封口后调 `_index_run`，质检写回后调 `_index_qc_run`；
② 位置对：引擎那次必须在 `with self._wake` **之外**（持锁写 sqlite = 卡住 submit）；
③ 归因码可聚类：`error_code` 只取**类别**（`technical_qc.ISSUE_CODES` / 异常类名式前缀），
   绝不拿带数字的整句话当码（那样每次失败自成一桶 ⇒ 规则永远"没有聚集"）；
④ 重跑不堆行：同一次成品重跑质检覆盖同一条（否则失败率被自己的重试污染）；
⑤ 记账失败不拖累主链：`record_run` 抛异常时两处写点都静默（P0 只加事实）；
⑥ 端到端：落进去的失败真能被 `suggest()` 聚成候选（接线 → 建议，不是两段各说各话）。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/outcome_wiring_test.py
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"
sys.path.insert(0, str(BP))
os.environ.setdefault("PROXY_TO_NODE", "0")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

TMP = Path(tempfile.mkdtemp(prefix="outcome-wiring-"))
os.environ["OUTCOME_INDEX_DB"] = (TMP / "outcome.sqlite").as_posix()
os.environ["ENGINE_TUNING_JSON"] = (TMP / "tuning.json").as_posix()
os.environ["KNOWLEDGE_LEDGER_DB"] = (TMP / "ledger.sqlite").as_posix()
os.environ["DATA_ROOT"] = TMP.as_posix()

from app.services import knowledge_ledger, outcome_index, technical_qc  # noqa: E402
from app.services.engine import hooks, runtime, tuning  # noqa: E402

#: ⚠️ 2026-11-07：引擎读业务数据已改成**注入口**（`hooks.get("run_index"/"ledger")` ✓，
#: 生产由 `app/main.py` 启动时登记 ✓）—— 本套必须补同一步 ✗：否则 ⑧/⑨/⑪ 会集体
#: 「读到空」（`_scan()` 回 `available=False` 时 `suggest()` 不给候选 ✓），看起来像归因逻辑坏了 ✓✗。
hooks.provide(run_index=outcome_index, ledger=knowledge_ledger)

_RUNTIME_SRC = (BP / "app" / "services" / "engine" / "runtime.py").read_text(encoding="utf-8")
_QC_SRC = (BP / "app" / "services" / "technical_qc.py").read_text(encoding="utf-8")
_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _rows(*, stage: str | None = None, kind: str | None = None) -> list[dict]:
    filters = {key: value for key, value in
               {"stage": stage, "kind": kind}.items() if value}
    return outcome_index.query(limit=100, **filters).get("items") or []


def _task(*, stage: str = "qwen_image", outputs: dict | None = None,
          steps: int = 30, status: str = "done", error: dict | None = None,
          task_id: str = "eng-wiring-001"):
    task = runtime.EngineTask(id=task_id, stage=stage,
                              request=SimpleNamespace(seed=777, cfg=4.0,
                                                      sampler="euler", schedule="simple"),
                              assembly={}, createdAt=1_000)
    task.finishedAt = 3_500
    task.status = status
    task.result = {"plan": {"steps": steps, "width": 1024, "height": 1024,
                            "frames": None, "fps": None},
                   "outputs": outputs or {}, "error": error}
    return task


def _engine():
    #: 落账挂在 :class:`EngineRuntime` 上（不是 ``TorchBackend`` ✗ —— 后者只管权重）。
    #: ``_index_run`` 只用到 ``@classmethod _run_kind``（后缀表 / 族表都在类上 ✓），
    #: 不碰任何实例状态 ⇒ ``__new__`` 起个最小实例就够，不必真起线程与队列。
    return runtime.EngineRuntime.__new__(runtime.EngineRuntime)


def _qc(*, status: str = "passed", overall: int = 92, codes: tuple[str, ...] = (),
        run_id_storyboard: str = "sb-1", generation: str = "vg-1") -> None:
    issues = [{"code": code, "severity": "error", "message": f"测试用：{code} 0.42s"}
              for code in codes]
    technical_qc._index_qc_run(
        storyboard_id=run_id_storyboard, video_generation_id=generation,
        episode_id="ep-1", status=status, overall=overall,
        spec={"blackSegments": [], "freezeSegments": [], "duration": 5.0,
              "fps": 24, "hasAudio": True},
        issues=issues,
    )


def _issue_code_gaps(tree: ast.Module) -> tuple[list[int], list[str], int]:
    """每处 ``result["issues"].append({...})`` 的字典必须带**闭集内的字面量** ``code`` ✓。

    返回 ``(缺 code 的行号, 码不在闭集里的表达式, 查到的处数)``。
    ``result["issues"]`` 认**结构**、不认 ``ast.unparse`` 出来的引号形 —— unparse 用单引号 ⇒
    写成 ``endswith('["issues"]')`` 会一处都匹配不到，判据自己变成空转 ✗（本轮实测踩过）。
    """
    def _is_issue_list(value: ast.expr) -> bool:
        return (isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Constant)
                and value.slice.value == "issues")

    missing: list[int] = []
    unknown: list[str] = []
    count = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append" and node.args
                and _is_issue_list(node.func.value)):
            continue
        count += 1
        arg = node.args[0]
        pairs = {key.value: value for key, value in zip(arg.keys, arg.values)
                 if isinstance(key, ast.Constant) and isinstance(key.value, str)} \
            if isinstance(arg, ast.Dict) else {}
        code = pairs.get("code")
        if code is None:
            missing.append(node.lineno)
        elif not (isinstance(code, ast.Constant) and code.value in technical_qc.ISSUE_CODES):
            unknown.append(ast.unparse(code))
    return missing, unknown, count


def main() -> int:
    # ==================================================== ① AST：两处写点都在
    runtime_tree = ast.parse(_RUNTIME_SRC)
    qc_tree = ast.parse(_QC_SRC)

    def calls_of(tree: ast.Module, func_name: str, call_name: str):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                found = [inner for inner in ast.walk(node)
                         if isinstance(inner, ast.Call)
                         and getattr(inner.func, "id", getattr(inner.func, "attr", "")) == call_name]
                return node, found
        return None, []

    loop, loop_calls = calls_of(runtime_tree, "_worker_loop", "_index_run")
    qc_func, qc_calls = calls_of(qc_tree, "run_technical_qc", "_index_qc_run")
    check("① 引擎封口后真的落账（`_worker_loop` 调 `_index_run`）", len(loop_calls) == 1, len(loop_calls))
    check("① 质检写回后真的落账（`run_technical_qc` 调 `_index_qc_run`）",
          len(qc_calls) == 1, len(qc_calls))

    # ==================================================== ② 落账在锁外
    wake_with = None
    for node in ast.walk(loop) if loop is not None else []:
        if isinstance(node, ast.With) and any(
                "self._wake" in ast.unparse(item.context_expr) for item in node.items):
            wake_with = node
    check("② 找到了 `with self._wake` 临界区", wake_with is not None, "没找到")
    if wake_with is not None and loop_calls:
        check("② 落账必须在临界区**之外**（持锁写 sqlite 会卡住 submit）",
              loop_calls[0].lineno > (wake_with.end_lineno or 0),
              (loop_calls[0].lineno, wake_with.end_lineno))
    check("② 两处写点都包着 try（记账失败不许上抛）",
          "except Exception" in _QC_SRC and "except Exception" in _RUNTIME_SRC, "")

    # ==================================================== ③ 归因码是类别不是句子
    src_codes = {node.value for node in ast.walk(qc_tree)
                 if isinstance(node, ast.Constant) and isinstance(node.value, str)
                 and node.value in technical_qc.ISSUE_CODES}
    check("③ ISSUE_CODES 的每个码在源码里真的被用上（没有死登记）",
          src_codes == set(technical_qc.ISSUE_CODES), sorted(set(technical_qc.ISSUE_CODES) - src_codes))
    check("③ 码里不带数字（带数字就聚不到一起）",
          all(not any(char.isdigit() for char in code) for code in technical_qc.ISSUE_CODES),
          list(technical_qc.ISSUE_CODES))

    #: 只查「码没人用」是**单向**的 ✗：新加一处 ``result["issues"].append({...})`` 却忘了
    #: 写 ``code`` 时，闭集里的码仍全被用上 ⇒ 上面那条照样绿，而归因又退回文案 ✗。
    #: 所以反过来再查一遍：每一处 issue 的字典**必须**有 ``code``（且值在闭集内）。
    gaps, unknown, counted = _issue_code_gaps(qc_tree)
    check(f"③ 每处 issue（{counted} 处）都带 `code`（漏一处就退回按文案聚类 ✗）",
          counted >= 7 and not gaps, gaps)
    check("③ 每处 issue 的 `code` 都是闭集里的字面量（不许现拼字符串 ✗）",
          not unknown, unknown)
    #: 反验：真实源码现在是干净的 ⇒ 坏法只能靠喂假 AST 造出来
    fake = ast.parse(
        'def f(result):\n'
        '    result["issues"].append({"severity": "error", "message": "忘写 code"})\n'
        '    result["issues"].append({"code": "自造的码", "message": "不在闭集"})\n'
        '    result["issues"].append({"code": "black", "message": "好的"})\n')
    fake_gaps, fake_unknown, fake_count = _issue_code_gaps(fake)
    check("③ 反验：漏写 code / 码不在闭集 ⇒ 两种都要被点名（否则判据是空转 ✗）",
          fake_count == 3 and len(fake_gaps) == 1 and len(fake_unknown) == 1,
          (fake_count, fake_gaps, fake_unknown))

    # ==================================================== ④ 质检落账：通过
    _qc(status="passed", overall=92)
    rows = _rows(kind="qc")
    check("④ 质检通过 ⇒ 一条 kind=qc 的跑次", len(rows) == 1, rows)
    row = rows[0] if rows else {}
    check("④ 分数统一到 0–1（并保留原值在 meta）",
          row.get("score") == 0.92 and (row.get("meta") or {}).get("overallRaw") == 92, row)
    check("④ 通过时不写 error_code（没有错就没有码）",
          row.get("error_code") is None and row.get("status") == "ok", row)
    check("④ 带上了能下钻的坐标（episode / storyboard）",
          row.get("episode_id") == "ep-1" and row.get("storyboard_id") == "sb-1", row)

    # ==================================================== ⑤ 质检落账：失败取码
    _qc(status="failed", overall=71, codes=("freeze", "black"), run_id_storyboard="sb-2",
        generation="vg-2")
    failed_row = next((item for item in _rows(kind="qc")
                       if item.get("storyboard_id") == "sb-2"), {})
    check("⑤ 失败 ⇒ status=failed 且 error_code 取第一条**类别码**",
          failed_row.get("status") == "failed" and failed_row.get("error_code") == "freeze",
          failed_row)
    check("⑤ 失败原因标了在哪一侧（技术维度 / 规则维度）",
          (failed_row.get("meta") or {}).get("cause") == "video_spec", failed_row.get("meta"))

    # ==================================================== ⑥ 重跑不堆行
    _qc(status="passed", overall=95, run_id_storyboard="sb-2", generation="vg-2")
    same = [item for item in _rows(kind="qc") if item.get("storyboard_id") == "sb-2"]
    check("⑥ 同一次成品重跑质检 ⇒ 覆盖同一条，不新增（否则重试把失败率顶高）",
          len(same) == 1 and same[0].get("score") == 0.95 and same[0].get("status") == "ok", same)

    # ==================================================== ⑦ 规则维度失败没有码
    _qc(status="failed", overall=60, codes=(), run_id_storyboard="sb-3", generation="vg-3")
    rule_row = next((item for item in _rows(kind="qc")
                     if item.get("storyboard_id") == "sb-3"), {})
    check("⑦ 规则维度判失败（只有自然语言）⇒ error_code=None 且 cause=rule（不编伪码）",
          rule_row.get("error_code") is None
          and (rule_row.get("meta") or {}).get("cause") == "rule", rule_row)

    # ==================================================== ⑧ 引擎落账：kind 按产物判
    backend = _engine()
    backend._index_run(_task(task_id="eng-mp4",
                             outputs={"primaryPath": "output/x/video_seed1.mp4"}), "done", None)
    backend._index_run(_task(task_id="eng-wav", stage="music3",
                             outputs={"primaryPath": "output/x/a.wav"}), "done", None)
    backend._index_run(_task(task_id="eng-png", stage="zimage",
                             outputs={"primaryPath": "output/x/i.png"}), "done", None)
    engine_rows = {item.get("run_id"): item for item in _rows() if str(item.get("run_id", "")).startswith("engine:")}
    kinds = sorted({item.get("kind") for item in engine_rows.values()})
    check("⑧ 引擎跑次按**产物后缀**判 kind（mp4⇒video / wav⇒tts / png⇒image）",
          kinds == ["image", "tts", "video"], kinds)
    engine_row = next(iter(engine_rows.values()), {})
    check("⑧ meta 带上了调参要用的维度（steps/宽高/seed）",
          (engine_row.get("meta") or {}).get("steps") == 30
          and (engine_row.get("meta") or {}).get("width") == 1024
          and (engine_row.get("meta") or {}).get("seed") == 777, engine_row.get("meta"))
    check("⑧ 耗时按 task 自己的时间戳算（不是 0）", (engine_row.get("elapsed_ms") or 0) == 2_500,
          engine_row.get("elapsed_ms"))

    # ==================================================== ⑨ 引擎失败取码
    backend._index_run(_task(stage="h3", status="failed",
                             error={"code": "CUDAOutOfMemory", "message": "装不下 36.9 GiB"}),
                       "failed", "装不下")
    oom_row = next((item for item in _rows()
                    if item.get("error_code") == "CUDAOutOfMemory"), None)
    check("⑨ 失败时的 error_code 取 `error.code`", oom_row is not None, _rows()[-1:])
    backend._index_run(_task(stage="h3", status="failed",
                             error={"message": "装不下：36.9 GiB > 22.49 GiB（D:/models/x）"}),
                       "failed", "装不下：36.9 GiB")
    check("⑨ 只有整句话时**不把整句当码**（宁可 None 也不造一堆单例桶）",
          runtime._engine_error_code({"error": {"message": "装不下：36.9 GiB > 22.49 GiB"}}) is None,
          runtime._engine_error_code({"error": {"message": "装不下：36.9 GiB > 22.49 GiB"}}))
    check("⑨ 异常类名式前缀照收（`TorchBackendUnavailable: 人话` ⇒ 类名）",
          runtime._engine_error_code({"error": {"message": "TorchBackendUnavailable: 缺 VAE"}})
          == "TorchBackendUnavailable", runtime._engine_error_code({"error": {"message": "x"}}))

    # ==================================================== ⑩ 记账失败不拖主链
    original = outcome_index.record_run
    outcome_index.record_run = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("库炸了"))
    thrown: list[str] = []
    try:
        backend._index_run(_task(outputs={}), "done", None)
        _qc(status="passed", overall=88, run_id_storyboard="sb-boom", generation="vg-boom")
    except Exception as exc:  # noqa: BLE001 —— 这里就是要抓"它居然抛了"
        thrown.append(repr(exc))
    finally:
        outcome_index.record_run = original
    check("⑩ record_run 抛异常时两处写点都静默（出片/质检结论不受影响）", thrown == [], thrown)

    # ==================================================== ⑪ 端到端：真能被 suggest 聚
    for index in range(30):
        _qc(status="failed", overall=40, codes=("freeze",),
            run_id_storyboard=f"sb-e2e-{index}", generation=f"vg-e2e-{index}")
    report = tuning.suggest()
    cand = next((item for item in report.get("candidates") or []
                 if item.get("id") == "failure-cluster:video_spec"), None)
    check("⑪ 30 条同样的质检失败 ⇒ suggest 报出聚集候选（接线→建议真通）",
          cand is not None, report.get("candidates"))
    cand = cand or {}
    check("⑪ 候选归因到类别码（不是「(未写 error_code)」）",
          cand.get("errorCode") == "freeze" and (cand.get("samples") or 0) >= 30, cand)
    check("⑪ 候选的凭据是索引里的真 run_id（过 check_evidence）",
          bool(cand.get("evidence"))
          and knowledge_ledger.check_evidence(cand.get("evidence") or []).get("ok") is True,
          cand.get("evidence"))

    failed = [name for name, ok, _ in _RESULTS if not ok]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  ↦ {detail!r}"))
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
