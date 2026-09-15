"""评测闭环 CLI —— 对齐 ``evaluation/cli.ts``（64 行）。

用法（与原 TS 的 ``npx tsx src/evaluation/cli.ts ...`` 一一对应）::

    python -m app.services.evaluation.cli evaluate benchmarks/storyboard-breaker.json
    python -m app.services.evaluation.cli optimize benchmarks/storyboard-breaker.json --iterations 3

⚠️ 与原 TS 的四处对应/差异：

1. **case 路径先按 cwd 解析**（``resolve(process.cwd(), filePath)`` ⇒ ``Path.cwd()/filePath``，
   与 Node 同义）；**找不到再回退到项目根** —— Node 那边 cwd 恒为 ``backend/``，``benchmarks/x.json``
   一路可用；S7 把 case 挪到项目根后，从 ``backend-py/`` 里跑同一条命令必须也能用，
   否则命令行要写 ``../benchmarks/x.json``，是个纯粹的 UX 陷阱；
2. 未知 ``kind`` 在**调评测前**就拦掉（``AGENT_BY_KIND`` 查表）并 exit 1；
3. ``--iterations`` 少了「非数字 → NaN」这个坑：TS 里 ``Number('abc')`` 得 NaN，
   会让迭代循环**一次都不跑**（静默 0 次）；这里**回退到默认 3** 并打一行 warn
   —— 与 README 里 ``format_vendor_http_error`` 同类**有意加固**（用户输入错误不该静默变语义）；
4. 评分报告打印格式逐字对齐（``总分 x/100`` + 每条维度一行）。

⚠️ 跑起来会**持有写事务**（``engine.begin()``，与 HTTP 路由同语义）：``optimize`` 要落
``agent_configs`` 与历史文件，必须提交。作为手动运维工具可以接受；不要在常驻服务里这样跑。
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.core.db import engine
from app.core.response import js_number
from app.services.agent_prompts import get_default_instructions
from app.agent.evaluation.catalog import AGENT_BY_KIND
from app.agent.evaluation.evaluator import evaluate_case
from app.agent.evaluation.optimizer import optimize_agent_prompt

__all__ = ["main", "print_report", "resolve_case_path"]

_USAGE = [
    "用法：",
    "  python -m app.services.evaluation.cli evaluate benchmarks/storyboard-breaker.json",
    "  python -m app.services.evaluation.cli optimize benchmarks/storyboard-breaker.json --iterations 3",
]


def resolve_case_path(raw: str) -> Path:
    """case 路径解析：**cwd 优先，回退项目根**（见模块 docstring 第 1 条）。

    两种习惯因此都能用：从项目根跑 ``benchmarks/x.json``、从 ``backend-py/`` 跑同一条命令
    （cwd 里没有 ⇒ 落到项目根），以及 ``x.json``（cwd 就是 ``benchmarks/``）。
    """
    direct = Path.cwd() / raw
    if direct.exists():
        return direct
    return PROJECT_ROOT / raw


def print_report(report: dict[str, Any]) -> None:
    """打印确定性评分报告（``总分 x/100`` + 每维度一行）。"""
    print(f"总分 {report['total']}/100")
    for dimension in report["dimensions"]:
        print(f"  {dimension['name']}: {dimension['score']}/{dimension['max']}  "
              f"{dimension['detail']}")


def _parse_iterations(rest: list[str]) -> int:
    """``--iterations N``：缺省 3；**非数字回退 3 并告警**（不制造 NaN，见模块 docstring）。"""
    if "--iterations" not in rest:
        return 3
    index = rest.index("--iterations")
    raw = rest[index + 1] if index + 1 < len(rest) else None
    if not raw:
        return 3
    parsed = js_number(raw)
    if parsed is None or not math.isfinite(parsed):
        print(f"[warn] --iterations 取值非法（{raw}），回退到默认 3", file=sys.stderr)
        return 3
    return int(parsed)


async def _run(command: str, case_def: dict[str, Any], agent_type: str,
               rest: list[str]) -> None:
    with engine.begin() as conn:
        if command == "evaluate":
            print(f"评测 {agent_type}（Reference 提示词）case={case_def['id']}")
            report = await evaluate_case(conn, case_def, get_default_instructions(agent_type))
            print_report(report)
            return
        await optimize_agent_prompt(conn, agent_type, case_def,
                                    {"iterations": _parse_iterations(rest)})


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程码（0 成功 / 1 用法错误或运行失败）。"""
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if len(args) > 0 else None
    case_path = args[1] if len(args) > 1 else None
    rest = args[2:]

    if not command or not case_path or command not in ("evaluate", "optimize"):
        for line in _USAGE:
            print(line)
        return 1

    case_def = json.loads(resolve_case_path(case_path).read_text(encoding="utf-8"))
    agent_type = AGENT_BY_KIND.get(case_def.get("kind") or "")
    if not agent_type:
        print(f"未知 case kind: {case_def.get('kind')}", file=sys.stderr)
        return 1

    try:
        asyncio.run(_run(command, case_def, agent_type, rest))
    except Exception as exc:  # noqa: BLE001 —— 对应 TS 顶层 catch
        print(f"评测失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
