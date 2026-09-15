#!/usr/bin/env python3
"""一键跑全部仓库自检（三道串联 + 汇总）。

> 本文件是 ``scripts/check-all.mjs`` 的**逐条对齐移植**（2026-09-15，Node 版已删）。

为什么需要
  ``.githooks/pre-commit`` 是**按资产条件触发**的：改 ``skills/`` 只跑引用守卫、改
  ``.codebuddy/memory/`` 只跑记忆守卫。这对手提交是对的（快），但想「整体体检」时
  得手敲三道命令，容易漏跑 —— 而漏跑的那道恰恰可能是红的那道。

与 ``test_guards.py`` 的分工
  · 本脚本 = **跑**守卫（正向体检：真实资产当前是否合规）。
  · ``test_guards.py`` = **测**守卫（负向实证：守卫还能不能报错）—— 它已包含在本脚本第 3 道。

用法::

    python backend-py/app/scripts/check_all.py [--verbose]

``--verbose`` 透传给 ``check_skill_refs.py``（额外列出被跳过的候选，审计其盲区）。
退出码：0 = 三道全过；1 = 任一失败（失败道次在末尾汇总）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # depth-adjusted-to-app
SCRIPTS = Path(__file__).resolve().parent  # 兄弟守卫在同一目录（按自身位置推导）
VERBOSE = "--verbose" in sys.argv

#: 顺序有意义：先跑「真实资产是否合规」，最后跑「守卫是否还能报错」
STEPS = [
    ("skills/ 路径引用完整性（含 docs/ 引用）", "check_skill_refs.py",
     ["--verbose"] if VERBOSE else []),
    (".codebuddy/memory/ 记忆层（8k 预算 + 锚点 + 落点）", "check_memory.py", []),
    ("守卫自检（两套守卫的负向用例）", "test_guards.py", []),
]


def main() -> int:
    failed = 0
    for label, script, args in STEPS:
        print(f"\n=== {label}\n=== → {SCRIPTS.relative_to(ROOT).as_posix()}/{script}")
        # 不捕获输出 —— 让子守卫的彩色/多行输出原样透出，避免二次转述失真
        result = subprocess.run([sys.executable, str(SCRIPTS / script), *args])
        if result.returncode != 0:
            failed += 1

    if failed:
        print(f"\n✗ 自检未通过：{failed}/{len(STEPS)} 道失败")
    else:
        print(f"\n✓ 全部 {len(STEPS)} 道自检通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
