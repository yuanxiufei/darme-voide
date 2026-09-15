"""S7 自检：TS 源码快照（Python 版）的**反漂移**性质。

删 ``backend/`` 之后，九道守卫靠 ``tests/frozen_ts_source.py`` 继续证明「Python vs TS 当初」
没漂移 ⇒ **快照必须覆盖守卫读到的每一个文件**。这里锁的正是「守卫新增一处读文件 ⇒ 快照悄悄缺件」
这类事故（本项目**真实发生过两次**：``services/consistency-qc.ts`` 的跨字面量形态、
``services/technical-qc.ts`` 的表驱动普通字符串形态）。

2026-09-15 起快照**不再存 ``.ts`` 文件树**（改存 Python 模块，守卫侧物化到临时目录）——
故这里还额外锁「仓库里不再有 frozen_ts/ 目录」与「物化的内容与快照数据逐字一致」。

运行::

    ./.venv/Scripts/python.exe tests/freeze_snapshot_test.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="frozen_"))
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import freeze_ts_snapshot as fz  # noqa: E402
from frozen_ts import load, snapshot_root  # noqa: E402

_R: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _R.append((name, bool(condition), detail))


def main() -> int:
    files_needed, dirs_needed = fz._effective()
    discovered_files, discovered_dirs = fz.discover_refs()
    data = load()

    check("自动发现: 能扫到守卫里 `_SRC_ROOT / \"services\" / \"consistency-qc.ts\"` 这种**跨字面量拼接**",
          "services/consistency-qc.ts" in discovered_files, discovered_files)
    check("自动发现: 也能扫到**表驱动普通字符串**形态（services/technical-qc.ts）",
          "services/technical-qc.ts" in discovered_files, discovered_files)
    check("自动发现: 快照需覆盖的文件 = 手写清单 ∪ 自动发现",
          set(fz.FILES) <= set(files_needed) and set(discovered_files) <= set(files_needed),
          (len(fz.FILES), len(files_needed)))

    absent = [name for name in files_needed if name not in data]
    check("⭐ 快照覆盖: 守卫读到的每个文件**都在快照数据里**（新增守卫忘了重冻 ⇒ 这里红）",
          not absent, absent)
    empty_dirs = [name for name in dirs_needed
                  if not any(key.startswith(name + "/") for key in data)]
    check("快照覆盖: 每个「按目录读」的目录都至少有一个条目", not empty_dirs, empty_dirs)
    check("快照规模: 至少覆盖 routes/ + services/adapters/ + agents/ 三个目录",
          {"routes", "services/adapters", "agents"} <= set(dirs_needed), dirs_needed[:8])

    if fz.TS_SRC.is_dir():
        dangling = [name for name in files_needed if not (fz.TS_SRC / name).is_file()]
        check("真源码一致: 自动发现的路径在 backend/src 下确实存在（无幽灵引用）",
              not dangling, dangling)
    else:
        print("[skip] backend/ 已删除：跳过「真源码存在性」检查")

    check("check(): 完整性自检本身返回 0", fz.check() == 0)

    # ── ⭐ 物化：守卫读到的应当是**快照数据的逐字副本**（不是被转换过的某种摘要） ──
    root = snapshot_root()
    sample = "routes/dramas.ts" if "routes/dramas.ts" in data else sorted(data)[0]
    materialized = (root / sample).read_text(encoding="utf-8")
    check("⭐ 物化: 临时目录里的内容与快照数据**逐字一致**（含换行）",
          materialized == data[sample], sample)
    check("物化: 同一进程内复用同一目录（不重复写盘）", snapshot_root() == root)

    # ── ⭐ 仓库里不再有 .ts 快照树（改存 Python 模块） ──
    check("结构: 已无 `tests/frozen_ts/` 目录（快照改存 frozen_ts_source.py）",
          not (TESTS / "frozen_ts").exists(), sorted(p.name for p in TESTS.glob("frozen_ts*")))

    # ── ⭐ 端到端：冻结模式跑一遍守卫，必须与真源码同结论（0 漂移）──
    # 这条才是真正的「删库保险」验收：它不关心清单怎么写，只看**快照够不够用**。
    env = {**os.environ, "PARITY_USE_FROZEN": "1", "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(
        [sys.executable, str(TESTS / "route_parity_test.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=300, cwd=str(TESTS.parent))
    output = completed.stdout + completed.stderr
    drift_zero = "漂移 0 条" in output and "镜像常量漂移 0 条" in output
    check("⭐ 端到端: **冻结模式**跑守卫 -> 退出码 0 且 0 漂移（快照足以支撑删库后的守卫）",
          completed.returncode == 0 and drift_zero,
          [line.strip() for line in output.splitlines() if "漂移" in line or "FAIL" in line][:4])

    failures = [item for item in _R if not item[1]]
    for name, passed, detail in _R:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_R) - len(failures)}/{len(_R)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
