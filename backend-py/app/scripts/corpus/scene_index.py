#!/usr/bin/env python3
"""场次索引批处理（A 侧）—— 库里剧本 → ``data/ocr/library-scenes.sqlite``。

⚠️ **不重复实现解析**：切场、角色提取、置信度全在 ``app/services/scene_parser.py``；
落库与读回在 ``app/services/scene_index.py``。HTTP 端点 ``GET /knowledge/scenes`` 与
Agent 工具读的是**同一份**派生库 ⇒ 三者不会给出三种答案。

为什么是批处理而不是 HTTP
========================
解析一份要读全本正文（几百页 OCR 文本），5403 份一起跑是小时级任务；藏在 HTTP 端点后面
既会超时，也没法续跑。这里可 Ctrl-C、可 ``--limit`` 试跑、可 ``--force`` 重算，与
``ocr_index.py`` 的断点纪律一致。

用法（⚠️ PowerShell 直接传中文路径会乱码 ⇒ 优先 ``--path-file``）::

    python -m app.scripts.corpus.scene_index --stats
    python -m app.scripts.corpus.scene_index --path-file=tmp/p.txt
    python -m app.scripts.corpus.scene_index --all --limit=50
    python -m app.scripts.corpus.scene_index --all --force

参数：``--path=P``（一份相对路径）｜``--path-file=F``（一行一个，绕开编码问题）
     ｜``--all``（遍历两库所有有正文的文档）｜``--limit=N``｜``--force``（忽略 sha256 缓存）
     ｜``--json``（机器可读）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[3]
if str(BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(BACKEND_PY))

from app.services import scene_index, script_catalog, script_library  # noqa: E402


def _all_paths() -> list[str]:
    """两库里所有登记过的文档 → 相对路径（去重、保序）。

    直接读 ``documents`` 而不是 ``list_documents``：后者带分页与目录过滤，是给界面用的，
    批处理要的是"全集"，用错入口会漏掉后几页的文档。
    """
    seen: dict[str, None] = {}
    for db in (script_library.METHODS_DB, script_library.OCR_DB):
        conn = script_library._connect(db)  # noqa: SLF001 —— 同包内复用只读连接
        if conn is None:
            continue
        try:
            rows = conn.execute("SELECT path FROM documents ORDER BY id").fetchall()
        except Exception:  # noqa: BLE001 —— 库坏了一律当没有，不阻塞另一库
            continue
        finally:
            conn.close()
        for (raw,) in rows:
            rel = script_catalog.normalize(raw, script_library.LOCAL_MARKER)
            if rel:
                seen.setdefault(rel, None)
    return list(seen)


def _render_stats(report: dict[str, Any]) -> None:
    if not report.get("available"):
        print(f"[scenes] 未建：{report.get('hint')}")
        return
    print(f"[scenes] 已解析 {report['documents']} 份 ｜ 场次 {report['scenes']} 场")
    stale = report.get("staleDocuments")
    if stale:
        print(f"          过期 {stale} 份（解析器指纹 {report.get('parserSha')} 与库内不符，"
              f"跑 ``--all`` 会自动重算，不用 ``--force``）")
    for status, count in sorted((report.get("byStatus") or {}).items()):
        print(f"          status={status}: {count}")
    for strategy, count in sorted((report.get("byStrategy") or {}).items()):
        print(f"          strategy={strategy}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="库内剧本 → 场次索引（A 侧派生）")
    parser.add_argument("--stats", action="store_true", help="只看盘点")
    parser.add_argument("--path", default="", help="一份文档的相对路径")
    parser.add_argument("--path-file", default="", help="一行一个路径（优先用这个，绕开 PowerShell 编码）")
    parser.add_argument("--all", action="store_true", help="遍历两库所有有正文的文档")
    parser.add_argument("--limit", type=int, default=0, help="最多处理几份（0 = 不限）")
    parser.add_argument("--force", action="store_true", help="正文 sha256 未变也重算")
    parser.add_argument("--json", action="store_true", help="机器可读输出")
    args = parser.parse_args()

    if args.stats:
        report = scene_index.scene_report()
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            _render_stats(report)
        return 0

    targets: list[str] = []
    if args.path:
        targets.append(args.path.strip())
    if args.path_file:
        #: ``utf-8-sig``：PowerShell 的 UTF8 会写 BOM
        text = Path(args.path_file).read_text(encoding="utf-8-sig")
        targets.extend(line.strip() for line in text.splitlines() if line.strip())
    if args.all:
        targets.extend(_all_paths())

    #: 去重保序：``--path`` 与 ``--all`` 同时给时不要重复解析同一份
    targets = list(dict.fromkeys(targets))
    if args.limit and args.limit > 0:
        targets = targets[:args.limit]
    if not targets:
        parser.print_help()
        return 2

    results: list[dict[str, Any]] = []
    counts = {"ok": 0, "no-heading": 0, "empty": 0, "missing": 0, "skipped": 0}
    for position, rel in enumerate(targets, start=1):
        outcome = scene_index.build_document(rel, force=bool(args.force))
        key = "skipped" if outcome.get("skipped") else str(outcome.get("status"))
        counts[key] = counts.get(key, 0) + 1
        results.append(outcome)
        if not args.json:
            flag = "·" if outcome.get("skipped") else "✓"
            scenes = outcome.get("scenes", 0)
            strategy = outcome.get("strategy", "")
            print(f"{flag} [{position}/{len(targets)}] {outcome.get('status')} "
                  f"{scenes} 场 {strategy} ← {rel}")
            for note in outcome.get("notes") or []:
                print(f"    ! {note}")

    if args.json:
        print(json.dumps({"counts": counts, "results": results}, ensure_ascii=False, indent=2))
    else:
        summary = " ｜ ".join(f"{key} {value}" for key, value in counts.items() if value)
        print(f"\n合计：{summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
