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


def freeze() -> int:
    if not TS_SRC.is_dir():
        print(f"❌ 找不到 TS 源码目录：{TS_SRC}（已删 backend/？那就不该再冻结）", file=sys.stderr)
        return 2
    copied = 0
    for relative in FILES:
        source = TS_SRC / relative
        if not source.is_file():
            print(f"❌ 缺少文件：{relative}", file=sys.stderr)
            return 2
        target = FROZEN / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied += 1
    for relative in DIRS:
        source_dir = TS_SRC / relative
        # ⚠️ **递归**：守卫会读 `agents/tools/*.ts` 这类子目录（漏了就会「TS 侧抽到空集 ⇒ 满屏 Python 独有」）
        files = sorted(source_dir.rglob("*.ts"))
        if not files:
            print(f"❌ 目录里没有 .ts：{relative}", file=sys.stderr)
            return 2
        for source in files:
            target = FROZEN / relative / source.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied += 1
    total = sum(p.stat().st_size for p in FROZEN.rglob("*.ts"))
    print(f"✅ 冻结完成：{copied} 个文件 / {total / 1024:.0f} KB -> {FROZEN}")
    return 0


def check() -> int:
    missing: list[str] = []
    for relative in FILES:
        if not (FROZEN / relative).is_file():
            missing.append(relative)
    for relative in DIRS:
        if not sorted((FROZEN / relative).rglob("*.ts")):
            missing.append(relative + "/**/*.ts")
    # 真源码还在时：逐个核对「源码树里的每个 .ts 都有对应快照」（防冻结后又改了 TS 却忘了重冻）
    if TS_SRC.is_dir():
        for relative in DIRS:
            for source in (TS_SRC / relative).rglob("*.ts"):
                target = FROZEN / relative / source.relative_to(TS_SRC / relative)
                if not target.is_file():
                    missing.append(str(target.relative_to(FROZEN)))
    if missing:
        print("❌ 快照不完整，缺：" + "、".join(missing), file=sys.stderr)
        return 1
    total = sum(p.stat().st_size for p in FROZEN.rglob("*.ts"))
    print(f"✅ 快照完整（{total / 1024:.0f} KB）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只校验快照完整性")
    args = parser.parse_args(argv)
    return check() if args.check else freeze()


if __name__ == "__main__":
    raise SystemExit(main())
