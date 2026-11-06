#!/usr/bin/env python3
"""剧本/编剧知识库检索（命令行）。

⚠️ **不重复实现任何口径**：分词、bm25/LIKE 分流、旧盘路径重映射、盘点全在
``app/services/script_library.py``；Agent 工具 ``app/agent/tools/library_tools.py`` 也调它。
本文件只是「人肉验证 + 排查」的入口 —— 让「命令行看得到」与「Agent 拿得到」不可能分叉
（这也是 prompt 语料那边 ``search.py`` 用注释声明的老约束，这里改用**同一份实现**来消除）。

用法（⚠️ PowerShell 直接传中文参数会乱码 ⇒ 优先 ``--q-file``）::

    python backend-py/app/scripts/corpus/library.py --stats
    python backend-py/app/scripts/corpus/library.py --q-file=tmp/q.txt --top=5
    python backend-py/app/scripts/corpus/library.py --q-file=tmp/q.txt --source=playbook
    python backend-py/app/scripts/corpus/library.py "激励事件" --json

参数：``--top=N``（默认 5）｜``--source=all|methods|playbook``｜``--path=<片段>``（只在该路径下找）
     ｜``--full``（片段的 400 字截断放开到 2000）｜``--json``（机器可读）｜``--stats``（只看盘点）
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[3]
if str(BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(BACKEND_PY))

from app.services import script_library  # noqa: E402 —— 必须在 sys.path 调整之后


def _arg(name: str, default: str = "") -> str:
    prefix = f"--{name}="
    return next((arg[len(prefix):] for arg in sys.argv if arg.startswith(prefix)), default)


def _flag(name: str) -> bool:
    return f"--{name}" in sys.argv


def _positionals() -> list[str]:
    """不以 ``--`` 开头的参数（多词查询在 PowerShell 里会被空格切开，全部收着别丢）。"""
    return [arg for arg in sys.argv[1:] if not arg.startswith("--")]


def _query() -> str:
    explicit = _arg("q")
    if explicit:
        return explicit
    q_file = _arg("q-file") or _arg("q_file")
    if q_file:
        #: ``utf-8-sig``：PowerShell 的 ``Set-Content -Encoding UTF8`` 会写 BOM
        return Path(q_file).read_text(encoding="utf-8-sig").strip()
    positional = _positionals()
    #: 没给 ``--q`` 时，位置参数**全部**拼成一个查询（静默丢掉第二个词＝检索结果说谎 ✗）
    return " ".join(positional) if positional else os.environ.get("LIBRARY_QUERY", "")


def _render_stats(report: dict) -> None:
    methods = report.get("methods")
    if methods:
        print(f"[methods] 文档 {methods['documents']} 部 ｜ 片段 {methods['chunks']} 段 "
              f"｜ 盘上缺失 {methods['missingFiles']} 部")
        print(f"          索引 {methods['db']}（{methods['root']}）")
    else:
        print("[methods] 未安装 / 索引读不到 ⇒ 工具会报 available=false")
    playbook = report.get("playbook")
    if playbook:
        print(f"[playbook] 文件 {playbook['files']} 份 ｜ 小节 {playbook['sections']} 段 "
              f"｜ 正文 {playbook['chars']} 字")
        print(f"          {playbook['root']}")
    else:
        print("[playbook] 未安装 ⇒ 工具会报 available=false")


def main() -> int:
    if _flag("stats"):
        _render_stats(script_library.library_report())
        return 0

    #: ``--q``/``--q-file`` 与位置参数同时给 ⇒ 位置参数会被静默丢弃（实测：``--q=雨夜 追逐``
    #: 在 PowerShell 里成了 ``--q=雨夜`` ＋ ``追逐``）⇒ 当场报错，别让检索结果说谎 ✗。
    if (_arg("q") or _arg("q-file") or _arg("q_file")) and _positionals():
        print("同时给了 --q/--q-file 和位置参数：位置参数会被忽略，请把整句写进 --q"
              "（PowerShell 里带空格要引号：--q=\"雨夜 追逐\"）。", file=sys.stderr)
        return 2

    query = _query().strip()
    if not query:
        print("缺少检索词：用位置参数、--q=、--q-file= 或环境变量 LIBRARY_QUERY。", file=sys.stderr)
        return 2

    source = _arg("source", "all") or "all"
    top = int(_arg("top", "5") or 5)
    path = _arg("path") or None
    full = _flag("full")

    outcome = script_library.search(query, source=source, limit=top, path_contains=path)
    if _flag("json"):
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return 0

    print(f"查询：{query} ｜ 来源：{source} ｜ 可用：{outcome['available']}")
    for name, info in outcome["sources"].items():
        matched = "未统计" if info["matched"] is None else info["matched"]
        print(f"  · {name}: 命中 {matched} 段 ｜ 排序 {info['ranking'] or '-'} "
              f"｜ 本次返回 {info['count']} 段")
    if outcome["hint"]:
        print(f"  ! {outcome['hint']}")
    print()
    for index, row in enumerate(outcome["results"], 1):
        page = f" p{row['page']}" if row.get("page") else ""
        cover = f" ｜ 覆盖 {row['coverage']}" if row.get("coverage") is not None else ""
        flags = "" if row["onDisk"] else "  ⚠️ 盘上找不到该文件"
        print(f"[{index}] {row['source']} ｜ {row['score']} 分{cover}{page}{flags}")
        print(f"    {row['path']} ｜ {row['title']}")
        print("    " + row["excerpt"][: (2000 if full else 400)].replace("\n", "\n    "))
        print()
    print(outcome["note"])
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
