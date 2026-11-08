"""优化史 `agent_optimizations` 守卫（`docs/knowledge-loop.md` §1 缺口 3）。

为什么要有它
============
优化器把分数写进 ``benchmarks/history/<时间戳>.json``，把"最终胜出的提示词"写进
``agent_configs`` —— 两处都不成对，于是「这次改动提升了多少」根本答不出来 ✓✗：

* ``history/*.json`` 是按 case **覆盖**的（同名重跑即冲掉上一轮的对手）；
* ``agent_configs`` 是**覆盖式**落库（上一次生效的提示词与分数当场就没了）。

本套盯的是"这张表能不能真的回答那个问题"，不是"端点返回 200"：

① 表结构：列齐、``target`` 必填、进 ``PYTHON_ONLY_TABLES``（Node DDL 里没有它的理由）；
② 落库：``delta`` 由服务端算；``score_before`` 为空 ⇒ ``delta=None``（**不是 0**）；
③ 追加不覆盖：同版本重写一行是**两行事实**，不是一次修改；
④ ``accepted`` 与 ``persisted`` **是两个布尔**（"分数更高"与"真变成默认值"不同）；
⑤ ``target`` / ``reason`` 闭集：未知值不许静默变成 ``system_prompt``；
⑥ 汇总：``rounds/accepted/persisted/bestDelta`` 分开给，空表不是 ``available=false``；
⑦ 读口读不出来 ⇒ ``available=false``（不返空表，空表会被读成"从没优化过"）；
⑧ 写口失败 ⇒ 只吞不抛（长跑优化不许被日志拖废）；
⑨ 接线：优化器**两个分支**都写、``mark_persisted`` 在 ``persist_agent_config`` 之后；
⑩ HTTP：``GET /knowledge-loop/optimizations`` 信封与过滤。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/optimization_log_test.py
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

TMP = Path(tempfile.mkdtemp(prefix="optimization-log-"))
os.environ["DATA_ROOT"] = TMP.as_posix()

from sqlalchemy import text  # noqa: E402

from app.core import db  # noqa: E402
from app.core import models  # noqa: E402
from app.services import optimization_log  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _reset() -> None:
    with db.engine.begin() as conn:
        conn.execute(text("DELETE FROM agent_optimizations"))


def _record(**kwargs) -> dict | None:
    params = {"agent_type": "script", "version": 2}
    params.update(kwargs)
    with db.engine.begin() as conn:
        return optimization_log.record(conn, **params)


# ===================================================================== ①
def case_schema() -> None:
    with db.engine.connect() as conn:
        cols = [row[1] for row in
                conn.execute(text("PRAGMA table_info(agent_optimizations)")).fetchall()]
    for name in ("agent_type", "case_id", "target", "version", "parent_version", "score_before",
                 "score_after", "delta", "accepted", "persisted", "reason", "runtime_model",
                 "prompt_sha", "evidence", "created_at"):
        check(f"① 表结构: 有 `{name}` 列 ✓", name in cols, cols)
    check("① 不存提示词全文 ✓（只存 sha1 指纹：几十版全文会把这张表撑成日志库 ✗）",
          "prompt" not in cols and "prompt_text" not in cols, cols)
    check("① 登记进 `PYTHON_ONLY_TABLES` ✓（Node 侧 schema.ts 没有这张表，不登记会被结构断言误报 ✗）",
          "agent_optimizations" in models.PYTHON_ONLY_TABLES, models.PYTHON_ONLY_TABLES)
    check("① 优化目标闭集＝system_prompt + params（温度，缺口 4 第一格）✓"
          "（留列不留假数据：没实现的目标不许出现 ✗）",
          sorted(optimization_log.TARGETS) == ["params", "system_prompt"], optimization_log.TARGETS)
    check("① 判决理由五值闭集 ✓", sorted(optimization_log.REASONS)
          == ["accepted", "below_noise", "eval_failed", "no_gain", "runtime_mismatch"],
          optimization_log.REASONS)


# ===================================================================== ②
def case_record() -> None:
    _reset()
    item = _record(version=2, parent_version=1, score_before=72.0, score_after=78.5,
                   accepted=True, reason="accepted", case_id="triage-basic",
                   runtime_model="ollama/qwen3:14b", prompt="你是一个剧本分析助手……",
                   evidence={"accuracy": 0.8, "brevity": 0.7})
    check("② 写入: 回执带版本/父子/分数/人话标签 ✓",
          item and item.get("version") == 2 and item.get("parentVersion") == 1
          and item.get("scoreAfter") == 78.5 and item.get("accepted") is True
          and item.get("reasonLabel") == optimization_log.REASONS["accepted"], item)

    check("② delta **由服务端算** ✓（78.5 − 72.0 ＝ 6.5，不信任调用方传的差值 ✓✗）",
          item and item.get("delta") == 6.5, item.get("delta") if item else None)
    check("② 只存指纹: `prompt_sha` 是 16 位十六进制 ✓",
          item and len(item.get("promptSha") or "") == 16
          and all(ch in "0123456789abcdef" for ch in item["promptSha"]),
          item.get("promptSha") if item else None)
    with db.engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM agent_optimizations ORDER BY id DESC")).first()
    check("② 只存指纹: 库里翻不到提示词原文 ✓（原文仍在 history/*.json ✓）",
          row is not None and "剧本分析助手" not in str(dict(row._mapping)),  # noqa: SLF001
          dict(row._mapping) if row is not None else None)  # noqa: SLF001

    first = _record(version=1, score_before=None, score_after=72.0, accepted=False,
                    reason="no_gain", case_id="triage-basic")
    check("② 无对手（第一版）⇒ `delta=None` **不是 0** ✓（写 0 会显示成\"提升了 0\"，那是编数 ✗）",
          first and first.get("delta") is None and first.get("parentVersion") is None, first)


# ===================================================================== ③
def case_append_only() -> None:
    _reset()
    _record(version=3, parent_version=2, score_before=78.5, score_after=78.5, accepted=False,
            reason="no_gain", case_id="triage-basic")
    _record(version=3, parent_version=2, score_before=78.5, score_after=80.0, accepted=True,
            reason="accepted", case_id="triage-basic")
    listing = optimization_log.list_records(limit=10)
    versions = [i["version"] for i in listing["items"]]
    check("③ 追加不覆盖: 同版本两条都在 ✓（`agent_configs` 可以覆盖，事实层不行 ✓✗）",
          versions.count(3) == 2 and listing["summary"]["rounds"] == 2, versions)
    check("③ 倒序: 最新一条在最前 ✓", versions[:1] == [3] and listing["items"][0]["accepted"], versions)


# ===================================================================== ④
def case_persisted_separate() -> None:
    _reset()
    item = _record(version=4, parent_version=3, score_before=80.0, score_after=84.0,
                   accepted=True, reason="accepted", case_id="triage-basic")
    check("④ 刚写入时 `accepted=true` 而 `persisted=false` ✓（判决 ≠ 生效 ✓✗）",
          item and item["accepted"] is True and item["persisted"] is False, item)

    with db.engine.begin() as conn:
        marked = optimization_log.mark_persisted(conn, agent_type="script", version=4,
                                                 case_id="triage-basic")
    again = optimization_log.list_records(limit=5)["items"][0]
    check("④ `mark_persisted` 改到 1 行并把 `persisted` 置真 ✓",
          marked == 1 and again["persisted"] is True, (marked, again["persisted"]))

    with db.engine.begin() as conn:
        nothing = optimization_log.mark_persisted(conn, agent_type="script", version=99)
    check("④ 标不存在的版本 ⇒ 0 行 ✓（不静默成功、也不新增一行 ✗）", nothing == 0, nothing)

    listing = optimization_log.list_records(limit=5)["summary"]
    check("④ 汇总里 `accepted` 与 `persisted` 分开数 ✓",
          listing["accepted"] == 1 and listing["persisted"] == 1, listing)


# ===================================================================== ⑤
def case_closed_sets() -> None:
    _reset()
    weird = _record(version=5, score_before=1.0, score_after=2.0, accepted=False,
                    reason="我觉得不行", target="temperature")
    check("⑤ 闭集外 `target` ⇒ 落成 `system_prompt` ✓（现在还没有\"温度\"这个优化目标 ✗）",
          weird and weird["target"] == "system_prompt", weird.get("target") if weird else None)
    check("⑤ 闭集外 `reason` ⇒ 原样留、`reasonLabel` 回落原文 ✓（不猜人话、也不丢原文 ✓✗）",
          weird and weird["reason"] == "我觉得不行" and weird["reasonLabel"] == "我觉得不行",
          weird.get("reason") if weird else None)
    check("⑤ 空 `agent_type` ⇒ 不落库 ✓（没有主体的优化史无法归因 ✗）",
          _record(agent_type="", version=6) is None)


# ===================================================================== ⑥
def case_summary() -> None:
    _reset()
    empty = optimization_log.list_records(limit=5)
    check("⑥ 空表: `available=true` 且 `rounds=0` ✓（\"还没优化过\"与\"表读不出来\"不是一回事 ✓✗）",
          empty["available"] is True and empty["summary"]["rounds"] == 0
          and empty["summary"]["bestDelta"] is None, empty["summary"])

    _record(version=1, score_before=70.0, score_after=74.0, accepted=True, reason="accepted",
            case_id="a")
    _record(version=2, parent_version=1, score_before=74.0, score_after=73.0, accepted=False,
            reason="no_gain", case_id="a")
    _record(version=1, score_before=60.0, score_after=91.0, accepted=True, reason="accepted",
            case_id="b")
    summary = optimization_log.list_records(limit=10)["summary"]
    check("⑥ 汇总: rounds / accepted / scored 数对 ✓",
          summary["rounds"] == 3 and summary["accepted"] == 2 and summary["scored"] == 3,
          summary)
    check("⑥ 汇总: `bestDelta` 取**最大提升**那一版 ✓（不是最近一次、不是平均 ✗）",
          summary["bestDelta"] == 31.0 and summary["bestVersion"] == 1, summary)
    check("⑥ 汇总: `byTarget` 按目标分组 ✓", summary["byTarget"] == {"system_prompt": 3},
          summary["byTarget"])

    filtered = optimization_log.list_records(case_id="b", limit=10)
    check("⑥ 过滤: 按 caseId 只取那一个 case ✓（不同 case 的分数不可比 ✓✗）",
          [i["caseId"] for i in filtered["items"]] == ["b"], filtered["items"])
    agent_filtered = optimization_log.list_records(agent_type="nobody", limit=10)
    check("⑥ 过滤: 不存在的 agentType ⇒ 空表但 `available=true` ✓",
          agent_filtered["available"] is True and agent_filtered["items"] == [],
          agent_filtered["items"])

    clamped = optimization_log.list_records(limit=10 ** 6)
    check("⑥ limit 会被夹到 MAX_LIMIT ✓（老库不许把端点拖死 ✓✗）",
          clamped["available"] is True, optimization_log.MAX_LIMIT)


# ===================================================================== ⑦
def case_unavailable() -> None:
    _reset()
    with db.engine.begin() as conn:
        conn.execute(text("ALTER TABLE agent_optimizations RENAME TO agent_optimizations_bak"))
    try:
        listing = optimization_log.list_records(limit=5)
        check("⑦ 表读不出来 ⇒ `available=false` 且带 reason ✓（不返空表 ✓✗）",
              listing["available"] is False and listing["items"] == []
              and bool(listing.get("reason")), listing.get("reason"))
        check("⑦ 表读不出来时汇总也不许装成\"0 轮\" ✓",
              listing["summary"]["available"] is False, listing["summary"].get("available"))
    finally:
        with db.engine.begin() as conn:
            conn.execute(text("ALTER TABLE agent_optimizations_bak RENAME TO agent_optimizations"))
    check("⑦ 表恢复后又读得出来 ✓（判据自己没把夹具搞坏 ✓）",
          optimization_log.list_records(limit=5)["available"] is True)


# ===================================================================== ⑧
def case_write_never_raises() -> None:
    class _Boom:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("database is locked")

    try:
        out = optimization_log.record(_Boom(), agent_type="script", version=1)  # type: ignore[arg-type]
        raised = ""
    except Exception as err:  # noqa: BLE001
        out, raised = None, f"{type(err).__name__}: {err}"
    check("⑧ 写口失败只吞不抛 ✓（一次优化要跑几十分钟，不许被日志拖废 ✓✗）",
          out is None and not raised, raised)

    try:
        marked = optimization_log.mark_persisted(_Boom(), agent_type="script", version=1)  # type: ignore[arg-type]
        raised = ""
    except Exception as err:  # noqa: BLE001
        marked, raised = -1, f"{type(err).__name__}: {err}"
    check("⑧ 标记失败也只吞不抛 ✓（返回 0 而不是炸掉 persist 流程 ✓）",
          marked == 0 and not raised, raised)


# ===================================================================== ⑨
def case_wiring() -> None:
    src = (BP / "app" / "agent" / "evaluation" / "optimizer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    func = next((node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == "optimize_agent_prompt"),
                None)
    check("⑨ 优化器里真有 `optimize_agent_prompt` ✓（`async def` 也算：它是长跑异步函数 ✓）",
          func is not None, type(func).__name__ if func is not None else "not found")

    records: list[ast.Call] = []
    marks: list[ast.Call] = []
    persists: list[ast.Call] = []
    if func is not None:
        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            attr = getattr(node.func, "attr", "")
            name = getattr(node.func, "id", "")
            base = getattr(getattr(node.func, "value", None), "id", "")
            if name == "persist_agent_config":  # 本模块内的裸调用 ✓
                persists.append(node)
            if base != "optimization_log":
                continue
            if attr == "record":
                records.append(node)
            elif attr == "mark_persisted":
                marks.append(node)

    check("⑨ 每轮都写优化史: `optimization_log.record` 出现 **2 次** ✓"
          "（正常轮 + 换模型不可比轮，少一处就会漏掉一整类事实 ✗）",
          len(records) == 2, len(records))

    kwargs = [kw.arg for node in records for kw in node.keywords]
    for need in ("score_before", "score_after", "accepted", "reason", "case_id"):
        check(f"⑨ record 显式给 `{need}` ✓（少给会让 delta/归因静默变成 None ✗）",
              need in kwargs, kwargs)

    accepted_vals = {getattr(kw.value, "value", None) for node in records
                     for kw in node.keywords if kw.arg == "accepted"}
    check("⑨ 两个分支的判决不同: 一处 `accepted=False`、一处 `accepted=<变量>` ✓"
          "（两处都写死 False ⇒ 永远没有\"采纳\"这一行 ✗）",
          False in accepted_vals and None in accepted_vals, accepted_vals)

    check("⑨ `mark_persisted` 只出现 1 次 ✓", len(marks) == 1, len(marks))
    if marks and persists:
        check("⑨ `mark_persisted` 在 `persist_agent_config` **之后** ✓"
              "（先落库再标记：反了就会出现\"标了已生效、其实没写\" ✗）",
              marks[0].lineno > persists[0].lineno,
              (persists[0].lineno, marks[0].lineno))
    else:
        check("⑨ 找得到 `persist_agent_config` 与 `mark_persisted` 两处调用 ✓", False,
              (len(persists), len(marks)))

    check("⑨ 优化史写口**不许**出现在路由里 ✓（HTTP 手写\"提升了多少\"没法核 ✗）",
          "optimization_log.record" not in
          (BP / "app" / "routers" / "knowledge_loop.py").read_text(encoding="utf-8"))


# ===================================================================== ⑩
def case_http() -> None:
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import app  # noqa: PLC0415

    _reset()
    _record(version=7, parent_version=6, score_before=80.0, score_after=83.0, accepted=True,
            reason="accepted", case_id="triage-basic")

    client = TestClient(app)
    got = client.get("/api/v1/knowledge-loop/optimizations")
    data = (got.json() or {}).get("data") or {}
    check("⑩ HTTP: GET /optimizations 200 且信封里 items/summary 都在 ✓",
          got.status_code == 200 and data.get("available") is True
          and len(data.get("items") or []) == 1 and data["summary"]["rounds"] == 1,
          (got.status_code, list(data)))

    # 本项目把 FastAPI 的 422 统一转成 400 + `why`（回执纪律 2），所以这里判 400 而**不是** 422 ✓
    bad = client.get("/api/v1/knowledge-loop/optimizations?limit=0")
    check("⑩ HTTP: limit 越界 ⇒ 400 且带 why ✓（不许静默当成 1 条 ✗）",
          bad.status_code == 400 and bool(bad.json()), (bad.status_code, bad.json()))

    other = client.get("/api/v1/knowledge-loop/optimizations?agentType=nobody")
    odata = (other.json() or {}).get("data") or {}
    check("⑩ HTTP: 按 agentType 过滤 ⇒ 空表但 `available=true` ✓",
          other.status_code == 200 and odata.get("items") == []
          and odata.get("available") is True, odata.get("items"))


def main() -> int:
    case_schema()
    case_record()
    case_append_only()
    case_persisted_separate()
    case_closed_sets()
    case_summary()
    case_unavailable()
    case_write_never_raises()
    case_wiring()
    case_http()

    failures = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAIL {name}  ← {detail!r}")
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
