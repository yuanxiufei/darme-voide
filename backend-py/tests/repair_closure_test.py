"""修复闭环守卫：**应用之后那个问题到底走没走**（知识闭环缺口 9）。

钉六件事：
1. **缺口键** ``<kind>:<小写文件名>`` —— 剥目录、只认 ``missing``/``ambiguous``，别的 kind 拒 ✗。
2. **指纹与顺序无关** —— 同一批缺口换个扫描顺序必须同指纹，否则「趋势」是假的。
3. **快照落盘与淘汰** —— 只留最近 N 份、新→旧、坏 json 跳过、``truncated`` 原样透传。
4. **四档判定顺序不可换** —— ``没挂键 → 没基准 → 当时不在 → 现在还在不在``；
   ⚠️ 最要紧的一条：**诊断被截断时「键不在集合里」不判 ``resolved``**（缺席可能只是没列出来）。
5. **记入跑次** —— ``resolved/still/stale/rolled_back`` 对到 ``ok/failed/partial``，
   ``unverified`` 不写；同一提案复诊几次都只一条（``run_id`` 固定）。
6. **接线**（AST）—— ``apply`` 里真调了复诊、``rollback`` 里真记终态、``diagnose`` 真落快照。

运行::

    ./.venv/Scripts/python.exe tests/repair_closure_test.py
"""
from __future__ import annotations

import ast
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _fake(missing: tuple[str, ...] = (), ambiguous: tuple[str, ...] = (), *,
          truncated: bool = False, ok: bool = True) -> dict:
    """造一份 ``repair_diagnose.diagnose()`` 形状的结果（不真扫盘）。"""
    long_used = ["a.json", "b.json", "c.json", "d.json", "e.json", "f.json", "g.json"]
    return {
        "ok": ok,
        "workflowsDir": "W",
        "counts": {"refs": 9, "have": 1, "missing": len(missing), "ambiguous": len(ambiguous)},
        "missing": [{"file": n, "usedBy": list(long_used), "nodes": ["UNETLoader"]} for n in missing],
        "ambiguous": [{"file": n, "usedBy": list(long_used), "nodes": ["LoraLoader"]} for n in ambiguous],
        "truncated": truncated,
    }


def _called_names(path: Path, func: str) -> set[str]:
    """``func`` 函数体里调到的名字（含 ``模块.函数`` 的属性名）✓ —— 钉接线而不记行号。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != func:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            if isinstance(inner.func, ast.Name):
                names.add(inner.func.id)
            elif isinstance(inner.func, ast.Attribute):
                names.add(inner.func.attr)
    return names


def _run_rows(db: Path) -> list[tuple[str, str, str, str]]:
    conn = sqlite3.connect(db.as_posix())
    try:
        return [tuple(row) for row in
                conn.execute("select run_id, status, kind, stage from runs order by run_id")]
    finally:
        conn.close()


def main() -> int:
    sys.path.insert(0, str(BP))
    from app.services import repair_closure as closure  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        os.environ["OUTCOME_INDEX_DB"] = str(root / "outcome.sqlite")

        # ------------------------------------------------------------ 缺口键
        check("键: 剥目录并小写",
              closure.finding_key("missing", "sub/Dir/X.SafeTensors") == "missing:x.safetensors")
        check("键: ambiguous 也建得出来",
              closure.finding_key("ambiguous", "y.safetensors") == "ambiguous:y.safetensors")
        check("键: 不认的 kind 拒（不编键）", closure.finding_key("have", "y.safetensors") == "")
        check("归一: 带 kind 的串",
              closure.normalize_finding("MISSING:A.SafeTensors") == "missing:a.safetensors")
        check("归一: 字典写法",
              closure.normalize_finding({"kind": "ambiguous", "file": "B.safetensors"})
              == "ambiguous:b.safetensors")
        check("归一: 裸文件名按缺失解释",
              closure.normalize_finding("c.safetensors") == "missing:c.safetensors")
        check("归一: 空值给空串（不抛）", closure.normalize_finding(None) == "")

        # ------------------------------------------------------------ 明细与指纹
        rows = closure.findings_of(_fake(("b.safetensors", "a.safetensors"), ("z.safetensors",)))
        check("明细: missing 两条 + ambiguous 一条", len(rows) == 3, [r["key"] for r in rows])
        check("明细: 按 key 排序（稳定）",
              [r["key"] for r in rows] == sorted(r["key"] for r in rows))
        check("明细: usedBy 封顶", len(rows[0]["usedBy"]) == closure.MAX_EVIDENCE)
        check("指纹: 与扫描顺序无关",
              closure.fingerprint(closure.findings_of(_fake(("a", "b"))))
              == closure.fingerprint(closure.findings_of(_fake(("b", "a")))))
        check("指纹: 集合变了值就变",
              closure.fingerprint(closure.findings_of(_fake(("a",))))
              != closure.fingerprint(closure.findings_of(_fake(("b",)))))
        empty_print = closure.fingerprint([])
        check("指纹: 空集合也有指纹（不是空串）", bool(empty_print) and len(empty_print) == 16,
              empty_print)
        check("指纹: 大写", empty_print.isupper(), empty_print)

        # ------------------------------------------------------------ 快照形状
        snap = closure.build_snapshot(_fake(("x.safetensors",), truncated=True))
        check("快照: 五个必备字段齐全",
              all(snap.get(k) for k in ("id", "at", "fingerprint", "counts", "workflowsDir")),
              sorted(snap))
        check("快照: keys 与明细同序", snap["keys"] == [r["key"] for r in snap["findings"]])
        check("快照: 截断标志原样透传", snap["truncated"] is True)
        check("快照: 同 at 两次 id 不撞",
              closure.build_snapshot(_fake(), at="2026-11-07T10:00:00")["id"]
              != closure.build_snapshot(_fake(), at="2026-11-07T10:00:00")["id"])

        # ------------------------------------------------------------ 落盘 / 历史 / 最新
        folder = root / "diagnose"
        first = closure.save_snapshot(closure.build_snapshot(_fake(("a.safetensors",))), folder)
        read_back = json.loads(first.read_text(encoding="utf-8"))
        check("落盘: 文件名就是快照 id", first.stem == read_back["id"])
        check("落盘: 内容可读回且指纹一致",
              read_back["fingerprint"]
              == closure.fingerprint(closure.findings_of(_fake(("a.safetensors",)))))

        for index, at in enumerate(("2026-11-05T10:00:00", "2026-11-06T10:00:00", "2026-11-07T10:00:00")):
            path = closure.save_snapshot(
                closure.build_snapshot(_fake((f"m{index}.safetensors",)), at=at), folder)
            os.utime(path, (1000.0 + index, 1000.0 + index))
        os.utime(first, (999.0, 999.0))  # 第一份是最旧的

        trend = closure.history(folder)
        check("历史: 新→旧",
              [row["at"] for row in trend][:3]
              == ["2026-11-07T10:00:00", "2026-11-06T10:00:00", "2026-11-05T10:00:00"],
              [row["at"] for row in trend])
        check("历史: 最早那份排最后（按 mtime 不按 id）",
              trend[-1]["at"] == read_back["at"], [row["at"] for row in trend])
        check("历史: findings 是条数不是明细", isinstance(trend[0]["findings"], int))
        check("最新: 就是最后那份", closure.latest(folder)["at"] == "2026-11-07T10:00:00")
        check("最新键: 集合 + 截断标志一起给",
              closure.latest_keys(folder) == ({"missing:m2.safetensors"}, False),
              closure.latest_keys(folder))

        prune_dir = root / "prune"
        for index in range(5):
            path = closure.save_snapshot(
                closure.build_snapshot(_fake((f"p{index}.safetensors",)),
                                       at=f"2026-11-0{index + 1}T09:00:00"),
                prune_dir, keep=3)
            os.utime(path, (2000.0 + index, 2000.0 + index))
        check("淘汰: keep=3 只留 3 份",
              len(list(prune_dir.glob("*.json"))) == 3,
              sorted(p.name for p in prune_dir.glob("*.json")))
        check("淘汰: 留的是最新 3 份",
              closure.history(prune_dir)[0]["at"] == "2026-11-05T09:00:00",
              [row["at"] for row in closure.history(prune_dir)])

        check("缺目录: latest 给 None", closure.latest(root / "nope") is None)
        check("缺目录: history 给空表", closure.history(root / "nope") == [])
        check("缺目录: latest_keys 给 None", closure.latest_keys(root / "nope") is None)

        broken = root / "broken"
        broken.mkdir()
        (broken / "broken.json").write_text("{ not json", encoding="utf-8")
        check("坏文件: latest 跳过不抛", closure.latest(broken) is None)
        check("坏文件: history 跳过不抛", closure.history(broken) == [])

        trunc_dir = root / "trunc"
        closure.save_snapshot(closure.build_snapshot(_fake(("t.safetensors",), truncated=True)),
                              trunc_dir)
        check("截断快照: 标志随键一起交出去",
              closure.latest_keys(trunc_dir) == ({"missing:t.safetensors"}, True),
              closure.latest_keys(trunc_dir))

        # ------------------------------------------------------------ 判定
        check("判定: 没挂键 ⇒ unverified",
              closure.verdict(finding="", was_present=True, after_keys=set())["status"]
              == "unverified")
        check("判定: 当时就不在 ⇒ stale",
              closure.verdict(finding="missing:a", was_present=False,
                              after_keys={"missing:a"})["status"] == "stale")
        check("判定: 没基准 ⇒ unverified",
              closure.verdict(finding="missing:a", was_present=None,
                              after_keys={"missing:a"})["status"] == "unverified")
        still = closure.verdict(finding="missing:a", was_present=True, after_keys={"missing:a"},
                                evidence=[{"key": "missing:a", "file": "a"}])
        check("判定: 键还在 ⇒ still 并带证据", still["status"] == "still" and still["evidence"])
        check("判定: 键没了 ⇒ resolved",
              closure.verdict(finding="missing:a", was_present=True,
                              after_keys={"missing:b"})["status"] == "resolved")
        tricky = closure.verdict(finding="missing:a", was_present=True,
                                 after_keys={"missing:b"}, after_truncated=True)
        check("判定: 截断时缺席不许判成功", tricky["status"] == "unverified", tricky)
        check("判定: 传大写也认",
              closure.verdict(finding="MISSING:A", was_present=True,
                              after_keys={"missing:a"})["status"] == "still")

        # ------------------------------------------------------------ 复诊
        calls = {"n": 0}

        def fresh_diag() -> dict:
            calls["n"] += 1
            return _fake(("other.safetensors",))

        outcome = closure.verify(finding="missing:a.safetensors", was_present=True,
                                 diagnose_fn=fresh_diag, directory=folder)
        check("复诊: 真跑了一次诊断", calls["n"] == 1, calls)
        check("复诊: 结论 resolved", outcome["status"] == "resolved", outcome)
        check("复诊: 回执带快照 id / 指纹 / 耗时",
              bool(outcome.get("snapshotId")) and bool(outcome.get("fingerprint"))
              and isinstance(outcome.get("elapsedMs"), int), outcome)
        check("复诊: 快照落盘成功", outcome["saved"] is True)
        check("复诊: 盘上多了一份快照", len(closure.history(folder)) == 5,
              len(closure.history(folder)))

        def boom() -> dict:
            raise RuntimeError("盘不见了")

        crashed = closure.verify(finding="missing:a", was_present=True, diagnose_fn=boom,
                                 directory=folder)
        check("复诊: 抛异常 ⇒ unverified 且 reason 带类名",
              crashed["status"] == "unverified" and "RuntimeError" in crashed["reason"], crashed)
        not_ok = closure.verify(finding="missing:a", was_present=True,
                                diagnose_fn=lambda: {"ok": False, "error": "目录不存在"},
                                directory=folder)
        check("复诊: ok=false ⇒ unverified 带上游文案",
              not_ok["status"] == "unverified" and "目录不存在" in not_ok["reason"], not_ok)

        # ------------------------------------------------------------ 跑次
        check("跑次: resolved → ok",
              closure.record("p1", {"status": "resolved", "finding": "missing:a"}) is True)
        check("跑次: still → failed", closure.record("p2", {"status": "still"}) is True)
        check("跑次: stale → partial", closure.record("p3", {"status": "stale"}) is True)
        check("跑次: rolled_back → partial",
              closure.record("p4", {"status": "rolled_back"}) is True)
        check("跑次: unverified 不写", closure.record("p5", {"status": "unverified"}) is False)
        check("跑次: 没提案号不写", closure.record("", {"status": "resolved"}) is False)

        db = root / "outcome.sqlite"
        before = _run_rows(db)
        closure.record("p2", {"status": "resolved", "finding": "missing:b"})
        after = _run_rows(db)
        check("跑次: 四档各一条（unverified 没进库）", len(before) == 4, before)
        check("跑次: 复诊不新增条目", len(after) == len(before), after)
        check("跑次: 复诊覆盖同一条（状态跟着变）",
              [row for row in after if row[0] == "repair-p2"][0][1] == "ok", after)
        check("跑次: kind 与 stage 都记 repair",
              all(row[2] == "repair" and row[3] == "repair" for row in after), after)

        # ------------------------------------------------------------ 回滚终态
        term = closure.rolled_back("MISSING:A", previous="resolved")
        check("回滚: 终态 status", term["status"] == "rolled_back")
        check("回滚: 保住上一档结论", term["previous"] == "resolved", term)
        check("回滚: 键归一", term["finding"] == "missing:a", term)
        check("回滚: 明确说不重新诊断", term["saved"] is False and bool(term["reason"]), term)

        # ------------------------------------------------------------ 接线（AST）
        router = BP / "app" / "routers" / "repair.py"
        check("接线: apply 里真调复诊", "_verify_proposal" in _called_names(router, "apply"))
        check("接线: propose 里挂缺口", "_attach_finding" in _called_names(router, "propose"))
        check("接线: rollback 里记终态", "rolled_back" in _called_names(router, "rollback"))
        check("接线: diagnose 里落快照", "_persist_snapshot" in _called_names(router, "diagnose"))
        check("接线: /verify 端点真复诊", "_verify_proposal" in _called_names(router, "verify"))
        check("接线: 复诊内部把结论记进跑次", "record" in _called_names(router, "_verify_proposal"))

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed_ok, detail in _RESULTS:
        print(("PASS  " if passed_ok else "FAIL  ") + name
              + ("" if passed_ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
