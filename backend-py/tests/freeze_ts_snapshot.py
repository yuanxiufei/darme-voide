"""把守卫依赖的 TS 源码**冻结**成快照 —— 「删 ``backend/``，但保住守卫的价值」。

背景：``route_parity_test.py`` 的九道守卫靠**读 TS 源码**来证明「Python 侧的路由表/常量/提示词
没漂移」。删掉 ``backend/`` 之后源码没了，守卫会集体失效。

做法：把守卫真正读到的那些文件**逐一复制**到 ``tests/frozen_ts/``（**保持相对路径不变**），
守卫侧的 ``_SRC_ROOT`` 会「真源码优先、缺失回退快照」——
于是删库后守卫仍然在比「Python 现在 vs TS 当初」，价值完整保留。

用法（**必须在删 ``backend/`` 之前跑一次**）::

    ./.venv/Scripts/python.exe tests/freeze_ts_snapshot.py            # 冻结
    ./.venv/Scripts/python.exe tests/freeze_ts_snapshot.py --check    # 只检查快照完整性

⚠️ 冻结的是**文本**而不是「提取后的结论」：守卫里的正则/抽取逻辑一行都不用改，
   出问题时也能直接看原文对账。
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TS_SRC = REPO / "backend" / "src"
FROZEN = Path(__file__).resolve().parent / "frozen_ts"

#: 守卫读到的**具体文件**（相对 ``backend/src``）
FILES: tuple[str, ...] = (
    "shared/prompt-utils.ts",
    "shared/prompt-blocks.ts",
    "shared/visual-graph.ts",
    "shared/camera-movement-guides.ts",
    "services/text-generation.ts",
    "agents/index.ts",
)

#: 守卫按**目录**读取/列举的（整目录 ``*.ts`` 都要）
DIRS: tuple[str, ...] = (
    "routes",
    "services/adapters",
    "agents",
)

#: 扫描哪些文件来发现「守卫还引用了哪些 .ts」
_GUARD_SOURCES = ("route_parity_test.py", "parity_diff_test.py")

#: ``_SRC_ROOT / "a" / "b.ts"``（**可能由多段字符串字面量拼出来**，故要整链捕获）
_SRC_REF = re.compile(r'_SRC_ROOT((?:\s*/\s*r?"[^"]+")+)')
_SEGMENT = re.compile(r'"([^"]+)"')


def discover_refs() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """从守卫源码里**自动发现**它引用的 TS 文件 / 目录（相对 ``backend/src``）。

    ⚠️ 为什么必须自动发现：``FILES`` 是手写清单，**新增一处「守卫读文件」时极易漏加** ——
    漏了不会立刻报错，直到「删掉 ``backend/`` 后跑冻结模式」才炸。本项目**真实发生过**：
    ``services/consistency-qc.ts`` 就是补守卫（连续性 QC 阈值镜像）当天漏进快照的，
    而快照恰是删库的唯一保险。这里扫 ``_SRC_ROOT / "…" / "…"`` 形态，
    以 ``.ts`` 结尾的当文件、其余当目录，与手写清单**取并集**。
    """
    here = Path(__file__).resolve().parent
    files: set[str] = set()
    dirs: set[str] = set()
    for name in _GUARD_SOURCES:
        path = here / name
        if not path.is_file():
            continue
        for match in _SRC_REF.finditer(path.read_text(encoding="utf-8")):
            relative = "/".join(_SEGMENT.findall(match.group(1)))
            (files if relative.endswith(".ts") else dirs).add(relative)
    return tuple(sorted(files)), tuple(sorted(dirs))


def _effective() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """手写清单 ∪ 自动发现（**顺序即去重后的并集**）。"""
    extra_files, extra_dirs = discover_refs()
    files = tuple(dict.fromkeys((*FILES, *extra_files)))
    dirs = tuple(dict.fromkeys((*DIRS, *extra_dirs)))
    return files, dirs


def freeze() -> int:
    if not TS_SRC.is_dir():
        print(f"❌ 找不到 TS 源码目录：{TS_SRC}（已删 backend/？那就不该再冻结）", file=sys.stderr)
        return 2
    files_to_copy, dirs_to_copy = _effective()
    auto = len(files_to_copy) - len(FILES)
    copied = 0
    for relative in files_to_copy:
        source = TS_SRC / relative
        if not source.is_file():
            print(f"❌ 缺少文件：{relative}", file=sys.stderr)
            return 2
        target = FROZEN / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied += 1
    for relative in dirs_to_copy:
        source_dir = TS_SRC / relative
        # ⚠️ **递归**：守卫会读 `agents/tools/*.ts` 这类子目录（漏了就会「TS 侧抽到空集 ⇒ 满屏 Python 独有」）
        found = sorted(source_dir.rglob("*.ts"))
        if not found:
            print(f"❌ 目录里没有 .ts：{relative}", file=sys.stderr)
            return 2
        for source in found:
            target = FROZEN / relative / source.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied += 1
    total = sum(p.stat().st_size for p in FROZEN.rglob("*.ts"))
    print(f"✅ 冻结完成：{copied} 个文件 / {total / 1024:.0f} KB -> {FROZEN}")
    print(f"   手写清单 {len(FILES)} 个 + 自动发现 {auto} 个文件；目录 {len(dirs_to_copy)} 个")
    return 0


def check() -> int:
    files_needed, dirs_needed = _effective()
    missing: list[str] = []
    for relative in files_needed:
        if not (FROZEN / relative).is_file():
            missing.append(relative)
    for relative in dirs_needed:
        if not sorted((FROZEN / relative).rglob("*.ts")):
            missing.append(relative + "/**/*.ts")
    # 真源码还在时：逐个核对「源码树里的每个 .ts 都有对应快照」（防冻结后又改了 TS 却忘了重冻）
    if TS_SRC.is_dir():
        for relative in dirs_needed:
            for source in (TS_SRC / relative).rglob("*.ts"):
                target = FROZEN / relative / source.relative_to(TS_SRC / relative)
                if not target.is_file():
                    missing.append(str(target.relative_to(FROZEN)))
    if missing:
        print("❌ 快照不完整，缺：" + "、".join(missing), file=sys.stderr)
        return 1
    total = sum(p.stat().st_size for p in FROZEN.rglob("*.ts"))
    print(f"✅ 快照完整（{total / 1024:.0f} KB）；需覆盖 {len(files_needed)} 个文件 + {len(dirs_needed)} 个目录")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只校验快照完整性")
    args = parser.parse_args(argv)
    return check() if args.check else freeze()


if __name__ == "__main__":
    raise SystemExit(main())
