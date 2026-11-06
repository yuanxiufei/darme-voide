r"""**ComfyUI 运行记录保留清理（CLI 薄壳）** ✓ —— 逻辑在 :mod:`app.services.comfyui` ✓。

为什么要有这个入口：``comfyui_runs`` 每跑一次长一条 ✓，附带的产物（``static/comfyui/`` 下的
PNG/MP4）也每个几百 KB 到几十 MB ✓ —— 不清就会一直涨 ✓。但**一键清空是不可接受的** ✗：

* 还有人在跑的（``queued`` / ``processing``）删了 ⇒ 后台任务照跑、产物落盘却没人指向它 ✗；
* 记录删了、文件留着 ⇒ 孤儿文件，再也没人知道该删哪个 ✗；
* 文件删了、记录留着 ⇒ 点开那条运行看到一张破图 ✗。

所以口径全在服务层的 :func:`app.services.comfyui.purge_runs` ✓（行与产物同生共死 ✓），
本文件只做三件事：解析命令行 ✓、人读渲染 ✓、给退出码 ✓。

用法::

    ./.venv/Scripts/python.exe app/scripts/comfyui_retention.py              # 干跑：只算账
    ./.venv/Scripts/python.exe app/scripts/comfyui_retention.py --days 7     # 换个保留期再算
    ./.venv/Scripts/python.exe app/scripts/comfyui_retention.py --apply      # 真删
    ./.venv/Scripts/python.exe app/scripts/comfyui_retention.py --json       # 给自动化用

退出码：``0`` = 干跑完成 / 真删且没有删不掉的产物；``1`` = 有产物删不掉（对应记录已保留 ✓，
需要人看一眼占用情况）；``2`` = 参数不对。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[2]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.core.db import engine  # noqa: E402
from app.services import comfyui as svc  # noqa: E402


def _human_bytes(size: int | float) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def render(report: dict[str, Any], *, applied: bool) -> None:
    print("── 保留清理 ───────────────────────────────────────")
    print(f"保留期：{report['days']} 天（早于 {report['cutoff']} 的都算过期）")
    print(f"模式：{'真删 ✓' if applied else '干跑（什么都没动 ✓）'}")
    if report.get("note"):
        print(f"⚠️ {report['note']}")
    print()
    print(f"过期记录：{report['considered']} 条")
    verb = "已删" if applied else "将删"
    print(f"  {verb}记录：{report['runs']} 条")
    print(f"  {verb}产物：{report['files']} 个 / {_human_bytes(report['bytes'])}")
    if report.get("missingFiles"):
        print(f"  产物已不在盘上：{report['missingFiles']} 条（记录照删 ✓）")

    for title, key in (
        ("仍在跑（一条都没动 ✓）", "keptRunning"),
        ("产物被别的记录共用（留着 ✓）", "keptShared"),
        ("路径不在 static/comfyui/ 下（整行留着 ✓）", "keptOutside"),
        ("产物删不掉（记录也留着 ✓）", "keptFailed"),
        ("删除失败", "failed"),
    ):
        items = report.get(key) or []
        if not items:
            continue
        print(f"\n{title}：{len(items)} 条")
        for item in items[:10]:
            print(f"  · {json.dumps(item, ensure_ascii=False)}")
        if len(items) > 10:
            print(f"  … 另有 {len(items) - 10} 条（--json 看全）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="清理过期的 ComfyUI 运行记录与它们的本地产物（默认干跑）",
    )
    parser.add_argument("--days", type=int, default=None,
                        help=f"保留天数（默认 {svc.RETENTION_DAYS}，读 COMFYUI_RETENTION_DAYS）")
    parser.add_argument("--apply", action="store_true", help="真删（不给就是干跑 ✓）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（给自动化用 ✓）")
    args = parser.parse_args(argv)

    days = args.days
    if days is None:
        raw = os.environ.get("COMFYUI_RETENTION_DAYS")
        if raw:
            try:
                days = int(raw)
            except ValueError:
                print(f"⚠️ COMFYUI_RETENTION_DAYS={raw!r} 不是整数 ⇒ 用默认 {svc.RETENTION_DAYS}",
                      file=sys.stderr)
    if days is not None and days < 0:
        print("--days 不能是负数", file=sys.stderr)
        return 2

    with engine.begin() as conn:
        report = svc.purge_runs(conn, days=days, dry_run=not args.apply)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        render(report, applied=args.apply)
    return 1 if report.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
