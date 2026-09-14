"""S7 自检：TS 源码快照（``tests/frozen_ts/``）的**反漂移**性质。

删 ``backend/`` 之后，九道守卫靠这份快照继续证明「Python vs TS 当初」没漂移 ⇒
**快照必须覆盖守卫读到的每一个 .ts**。这里锁的正是「守卫新增一处读文件 ⇒ 快照悄悄缺文件」
这类事故（本项目真实发生过：补写连续性 QC 阈值镜像当天，`services/consistency-qc.ts`
没被纳入手写清单，直到跑冻结模式才炸）。

运行::

    ./.venv/Scripts/python.exe tests/freeze_snapshot_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="frozen_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import freeze_ts_snapshot as fz  # noqa: E402

_R: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _R.append((name, bool(condition), detail))


def main() -> int:
    files_needed, dirs_needed = fz._effective()
    discovered_files, discovered_dirs = fz.discover_refs()

    check("自动发现: 能扫到守卫里 `_SRC_ROOT / \"services\" / \"consistency-qc.ts\"` 这种**跨字面量拼接**",
          "services/consistency-qc.ts" in discovered_files, discovered_files)
    check("自动发现: 快照需覆盖的文件 = 手写清单 ∪ 自动发现",
          set(fz.FILES) <= set(files_needed) and set(discovered_files) <= set(files_needed),
          (len(fz.FILES), len(files_needed)))

    absent = [name for name in files_needed if not (fz.FROZEN / name).is_file()]
    check("⭐ 快照覆盖: 守卫读到的每个文件**都在快照里**（新增守卫忘了重冻 ⇒ 这里红）",
          not absent, absent)
    empty_dirs = [name for name in dirs_needed
                  if not sorted((fz.FROZEN / name).rglob("*.ts"))]
    check("快照覆盖: 每个「按目录读」的目录都至少有一个 .ts", not empty_dirs, empty_dirs)

    if fz.TS_SRC.is_dir():
        dangling = [name for name in files_needed if not (fz.TS_SRC / name).is_file()]
        check("真源码一致: 自动发现的路径在 backend/src 下确实存在（无幽灵引用）",
              not dangling, dangling)
    else:
        print("[skip] backend/ 已删除：跳过「真源码存在性」检查")

    check("check(): 完整性自检本身返回 0", fz.check() == 0)
    check("快照规模: 至少覆盖 routes/ + adapters/ + agents/ 三个目录",
          {"routes", "services/adapters", "agents"} <= set(dirs_needed), dirs_needed[:8])

    # ── ⭐ 端到端：冻结模式跑一遍守卫，必须与真源码同结论（0 漂移）──
    # 这条才是真正的「删库保险」验收：它不关心清单怎么写，只看**快照够不够用**。
    # （本轮就是靠它发现 `services/technical-qc.ts` 漏冻结 —— 上面的存在性检查当时是绿的。）
    import subprocess  # noqa: PLC0415

    env = {**os.environ, "PARITY_USE_FROZEN": "1", "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "route_parity_test.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=300, cwd=str(Path(__file__).resolve().parents[1]))
    output = completed.stdout + completed.stderr
    drift_zero = "漂移 0 条" in output and "镜像常量漂移 0 条" in output
    check("⭐ 端到端: **冻结模式**跑守卫 -> 退出码 0 且 0 漂移（快照足以支撑删库后的守卫）",
          completed.returncode == 0 and drift_zero,
          [line.strip() for line in output.splitlines() if "漂移" in line or "FAIL" in line][:4])

    failed = [item for item in _R if not item[1]]
    for name, passed, detail in _R:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_R) - len(failed)}/{len(_R)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
