"""旧库 **schema 补列**运维入口（**仅标准库** ✓，与本目录其它工具同款 ✓）。

什么时候用：后端启动日志出现下面这行时（2026-09-25 的真实库就缺 **32 列** ✓）

    [db-py] ⚠️ columns missing in existing db (32 可补 / 0 需人工): ['ALTER TABLE …', …]

═══ 为什么会有这个缺口（是刻意设计，不是 bug ✓✗）═══

``core/db.py::_ensure_tables`` 刻意「**只增不改**」✓：缺的**表**每次启动 ``create_all`` 自动补 ✓；
缺的**列**只报不改 ✗ —— 「改用户的库必须由人决定」✓。本脚本就是那个「人工动作」的入口 ✓。

⚠️ 启动日志的清单**只打前 6 条**（要让启动输出能扫一眼 ✓）⇒ **完整清单走 ``--plan``** ✓。

═══ 用法 ═══

    python backend-py/app/scripts/db_upgrade.py            # 干跑：完整清单（不补列）
    python backend-py/app/scripts/db_upgrade.py --apply    # 先备份，再补列

退出码：``0`` = 已补完 / 无缺口；``1`` = 仍有需人工处理的列，或子进程失败。

═══ 三条口径（都与后端同源 ✓，这里不自己算 ✗）═══

1. **清单与判定一律以 ``app.core.db`` 的输出为准** ✗（本脚本只转发 ✓）——
   判据只有一处（``metadata`` ✓），在这儿重写一份必然漂移 ✓；
2. **库路径也不自己拼** ✗：从子进程输出里**读** ``[db-upgrade] db = …`` ✓
   （路径唯一权威是 ``core/config.py::get_db_path`` ✓，认不出就**中止**而不猜 ✓）；
3. **备份用 ``sqlite3`` 的 backup API** ✓，**不复制文件** ✗：库跑在 WAL 模式 ✓，
   只 copy ``drama.db`` 会丢掉 ``-wal`` 里尚未 checkpoint 的写 ✓✗
   （副本人为「少一个剧集」那种坑 ✓）。

⚠️ ``--plan`` 会**建缺失的表**（`create_all` ✓）—— 这是**任何一次后端启动都会发生**的既定行为 ✓，
但**不动任何已存在的表** ✓（补列只在 ``--apply`` 且备份之后 ✓）。
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

#: ``app/scripts/`` → ``backend-py/``：**跳过 scripts、app 两级** ✓
#: （即 ``parents[2]``；``parents[3]`` 才是仓库根 ✓）
#: ⚠️ 别照本目录 README「路径推导」那张表的 ``scripts/*.py`` 行 ✗ —— 那一行的级数**是过期的** ✗
#: （实测：按它写 ``parents[1]`` 会得到 ``backend-py/app`` ✓✗，报错是
#: ``No module named 'app'`` ✓ 而不是「路径不存在」✗ ⇒ 正是 README 自己警告的那种静默坑 ✓）。
#: 以本目录**实际脚本**为准（``tokenizer_bench.py`` / ``h3_readiness.py`` 都是 ``parents[2]`` ✓）。
BACKEND_PY = Path(__file__).resolve().parents[2]

_DB_LINE_PREFIX = "[db-upgrade] db = "


def _run_backend(flag: str) -> subprocess.CompletedProcess[str]:
    """以 subprocess 跑 ``python -m app.core.db <flag>``（与后端**解耦** ✓：本脚本零第三方依赖 ✓）。

    ⚠️ 解释器用 ``sys.executable`` ✓（脚本跑在哪个 Python，后端就用哪个 ✓）——
    后果是「用错解释器跑脚本」会在**子进程**里报 ``ModuleNotFoundError: sqlalchemy`` ✓，
    故这里**原样透传 stderr** ✗（不吞 ✓），让人一眼看出是环境而不是库的问题 ✓。

    ⚠️⚠️ ``PYTHONPATH`` **必须显式给** ✗（2026-09-25 实测踩到 ✓）：靠 ``-m`` 自己把 cwd
    放进 ``sys.path`` 是靠不住的 ✓ —— 本机 ``PYTHONPATH`` 被设成编辑器 shim 目录时，
    子进程直接 ``ModuleNotFoundError: No module named 'app'`` ✓✗，**而同一条命令手敲却没事** ✓
    （凭 cwd 找得到 ✓）⇒ 那种「只在被调用时才坏」的坑必须靠显式声明堵掉 ✓。
    """
    child_env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",  # 中文日志在管道里别乱码 ✓
        # 项目根**前置** ✓（保留既有 PYTHONPATH 以免影响别人的环境 ✓）
        "PYTHONPATH": os.pathsep.join(
            [str(BACKEND_PY)]
            + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
        ),
    }
    return subprocess.run(
        [sys.executable, "-m", "app.core.db", flag],
        cwd=str(BACKEND_PY),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        check=False,
    )


def _parse_db_path(output: str) -> Path | None:
    """从子进程输出里读库路径 ✓（读不到返回 ``None`` ⇒ 调用方**中止**，不猜 ✗）。"""
    for line in output.splitlines():
        if line.startswith(_DB_LINE_PREFIX):
            return Path(line[len(_DB_LINE_PREFIX):].strip())
    return None


def _backup(db: Path) -> Path:
    """一致性快照备份（含 WAL 里的写 ✓）⇒ ``<db>.bak-YYYYmmdd-HHMMSS`` ✓。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = db.with_name(f"{db.name}.bak-{stamp}")
    source = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        with sqlite3.connect(dest) as target:
            source.backup(target)  # 标准库 API ✓：读到的是「一个一致的点」，不是文件当前字节 ✓
    finally:
        source.close()
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="db_upgrade.py",
        description="旧库 schema 补列（ALTER TABLE ADD COLUMN；默认只干跑）",
    )
    parser.add_argument("--apply", action="store_true",
                        help="备份后真的补列（不加则只打印完整清单）")
    args = parser.parse_args()

    plan = _run_backend("--plan")
    if plan.stdout:
        print(plan.stdout, end="")
    if plan.returncode != 0 and not plan.stdout:
        print(f"✗ 子进程失败（exit {plan.returncode}）：\n{plan.stderr}", file=sys.stderr)
        return 1

    db = _parse_db_path(plan.stdout)
    if db is None:
        print(f"✗ 没能从子进程输出里认出库路径 ⇒ 中止（不猜路径 ✗）\n{plan.stderr}", file=sys.stderr)
        return 1

    if not args.apply:
        print(f"\n[db-upgrade] 干跑结束：未改动 {db} ⇒ 要真补请加 --apply")
        return plan.returncode

    print(f"\n[db-upgrade] 备份 → {_backup(db)}")
    applied = _run_backend("--apply")
    if applied.stdout:
        print(applied.stdout, end="")
    if applied.returncode != 0 and not applied.stdout:
        print(f"✗ 补列失败（exit {applied.returncode}）：\n{applied.stderr}", file=sys.stderr)
        return 1
    return applied.returncode


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
