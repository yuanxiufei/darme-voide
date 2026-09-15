"""把守卫依赖的 TS 源码**冻结**成 Python 快照 —— 「删 ``backend/``，但保住守卫的价值」。

背景：``route_parity_test.py`` 的九道守卫靠**读 TS 源码**来证明「Python 侧的路由表/常量/提示词
没漂移」。删掉 ``backend/`` 之后源码没了，守卫会集体失效。

做法：把守卫真正读到的那些文件**逐字**收进一个 Python 模块
（``tests/frozen_ts_source.py``，**生成物**）；守卫侧的 ``_SRC_ROOT`` 在「真源码缺失」或
``PARITY_USE_FROZEN=1`` 时调用 ``frozen_ts.snapshot_root()`` **物化到临时目录** ——
于是 15 处调用点一行都不用改，删库后守卫仍在比「Python 现在 vs TS 当初」。

⚠️ 2026-09-15 起快照**不再存 ``.ts`` 文件树**（改存 Python 模块）：仓库里不再有 TS 文件，
且 diff 能直接看到改的是哪一条文本。要**看原文**用 ``--dump``：

用法（**必须在删 ``backend/`` 之前跑一次**）::

    python tests/freeze_ts_snapshot.py              # 冻结（生成/覆盖 frozen_ts_source.py）
    python tests/freeze_ts_snapshot.py --check       # 只检查覆盖是否完整
    python tests/freeze_ts_snapshot.py --dump tmp/frozen_ts   # 物化成 .ts 文件便于人读
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TS_SRC = REPO / "backend" / "src"
TESTS = Path(__file__).resolve().parent
FROZEN_MODULE = TESTS / "frozen_ts_source.py"

#: 守卫读到的**具体文件**（相对 ``backend/src``）；与自动发现取并集
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
#: 普通字符串字面量形态的相对路径（如 `("services/technical-qc.ts", "CONST", "mod")`）
_PLAIN_TS = re.compile(r'"((?:[\w.-]+/)+[\w.-]+\.ts)"')


def discover_refs() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """从守卫源码里**自动发现**它引用的 TS 文件 / 目录（相对 ``backend/src``）。

    ⚠️ 为什么必须自动发现：``FILES`` 是手写清单，**新增一处「守卫读文件」时极易漏加** ——
    漏了不会立刻报错，直到「删掉 ``backend/`` 后跑冻结模式」才炸。本项目**真实发生过两次**：
    ``services/consistency-qc.ts``（跨字面量拼接形态）与 ``services/technical-qc.ts``
    （表驱动普通字符串形态）。这里**两种形态都扫**，与手写清单**取并集**。

    普通字符串形态只在**真源码还在**时按「文件确实存在」过滤（防注释/文档里的幽灵路径
    被当成必需件）；``_SRC_ROOT`` 形态本就带目录结构，不加存在性过滤。
    """
    files: set[str] = set()
    dirs: set[str] = set()
    for name in _GUARD_SOURCES:
        path = TESTS / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for match in _SRC_REF.finditer(text):
            relative = "/".join(re.findall(r'"([^"]+)"', match.group(1)))
            (files if relative.endswith(".ts") else dirs).add(relative)
        if TS_SRC.is_dir():  # 真源码在 ⇒ 普通字符串形态可校验存在性
            for candidate in _PLAIN_TS.findall(text):
                if (TS_SRC / candidate).is_file():
                    files.add(candidate)
    return tuple(sorted(files)), tuple(sorted(dirs))


def _effective() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """手写清单 ∪ 自动发现（顺序即去重后的并集）。"""
    extra_files, extra_dirs = discover_refs()
    files = tuple(dict.fromkeys((*FILES, *extra_files)))
    dirs = tuple(dict.fromkeys((*DIRS, *extra_dirs)))
    return files, dirs


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _chunks(text: str, size: int = 900) -> list[str]:
    """按行切成小块 ⇒ 生成的模块 diff 可读（一整段 700KB 的单行没法 review）。"""
    pieces: list[str] = []
    buffer = ""
    for line in text.splitlines(keepends=True):
        if buffer and len(buffer) + len(line) > size:
            pieces.append(buffer)
            buffer = ""
        buffer += line
    if buffer or not pieces:
        pieces.append(buffer)
    return pieces


def _write_module(data: dict[str, str]) -> None:
    body = [
        '"""TS 源码快照（**生成物：勿手改**）—— 由 ``tests/freeze_ts_snapshot.py`` 生成。',
        "",
        "用途：删掉 ``backend/`` 之后，``route_parity_test.py`` 的九道守卫仍要读 TS 源码来做",
        "比对；本模块把那些文件**逐字**存下来，守卫侧用 ``frozen_ts.snapshot_root()`` 物化到",
        "临时目录后照常读取（调用点零改动）。",
        "",
        "要改内容 ⇒ 改 ``backend/src`` 下的真源码后**重跑冻结脚本**，不要手改本文件；",
        "要读原文 ⇒ ``python tests/freeze_ts_snapshot.py --dump <dir>``。",
        '"""',
        "from __future__ import annotations",
        "",
        f"GENERATED_AT = {_iso_now()!r}",
        f"SOURCE_ROOT = {str(TS_SRC)!r}",
        "",
        "#: 相对 ``backend/src`` 的路径 -> 文件**逐字**文本（含换行原样）",
        "FILES: dict[str, str] = {",
    ]
    for relative in sorted(data):
        body.append(f"    {relative!r}: (")
        for piece in _chunks(data[relative]):
            body.append(f"        {piece!r}")
        body.append("    ),")
    body.append("}")
    body.append("")
    FROZEN_MODULE.write_text("\n".join(body), encoding="utf-8")


def freeze() -> int:
    if not TS_SRC.is_dir():
        print(f"❌ 找不到 TS 源码目录：{TS_SRC}（已删 backend/？那就不该再冻结）", file=sys.stderr)
        return 2
    files_to_copy, dirs_to_copy = _effective()
    auto = len(files_to_copy) - len(FILES)
    data: dict[str, str] = {}
    for relative in files_to_copy:
        source = TS_SRC / relative
        if not source.is_file():
            print(f"❌ 缺少文件：{relative}", file=sys.stderr)
            return 2
        data[relative] = source.read_text(encoding="utf-8")
    for relative in dirs_to_copy:
        source_dir = TS_SRC / relative
        # ⚠️ **递归**：守卫会读 `agents/tools/*.ts` 这类子目录（漏了就会「TS 侧抽到空集」）
        found = sorted(source_dir.rglob("*.ts"))
        if not found:
            print(f"❌ 目录里没有 .ts：{relative}", file=sys.stderr)
            return 2
        for source in found:
            data[(Path(relative) / source.relative_to(source_dir)).as_posix()] = (
                source.read_text(encoding="utf-8"))
    _write_module(data)

    total = sum(len(text) for text in data.values())
    print(f"✅ 冻结完成：{len(data)} 个文件 / {total / 1024:.0f} KB -> {FROZEN_MODULE.name}"
          f"（手写清单 {len(FILES)} 个 + 自动发现 {auto} 个；目录 {len(dirs_to_copy)} 个）")
    return 0


def check() -> int:
    files_needed, dirs_needed = _effective()
    if not FROZEN_MODULE.is_file():
        print(f"❌ 缺少快照模块：{FROZEN_MODULE.name}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(TESTS))
    from frozen_ts import load  # noqa: PLC0415

    data = load()
    missing: list[str] = []
    for relative in files_needed:
        if relative not in data:
            missing.append(relative)
    for relative in dirs_needed:
        if not any(key.startswith(relative + "/") for key in data):
            missing.append(relative + "/**/*.ts")
    # 真源码还在时：逐个核对「源码树里的每个 .ts 都有对应快照」（防冻结后又改了 TS 却忘了重冻）
    if TS_SRC.is_dir():
        for relative in dirs_needed:
            for source in (TS_SRC / relative).rglob("*.ts"):
                key = (Path(relative) / source.relative_to(TS_SRC / relative)).as_posix()
                if key not in data:
                    missing.append(key)
    if missing:
        print("❌ 快照不完整，缺：" + "、".join(missing), file=sys.stderr)
        return 1
    total = sum(len(text) for text in data.values())
    print(f"✅ 快照完整（{total / 1024:.0f} KB）；需覆盖 {len(files_needed)} 个文件 "
          f"+ {len(dirs_needed)} 个目录，实存 {len(data)} 个文件")
    return 0


def dump(target: str) -> int:
    sys.path.insert(0, str(TESTS))
    from frozen_ts import materialize, load  # noqa: PLC0415

    directory = Path(target)
    directory.mkdir(parents=True, exist_ok=True)
    materialize(directory)
    print(f"✅ 已物化 {len(load())} 个 .ts 到 {directory}（仅供人读；守卫用的是临时目录）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TS 源码快照（Python 版）")
    parser.add_argument("--check", action="store_true", help="只检查覆盖是否完整")
    parser.add_argument("--dump", metavar="DIR", help="把快照物化成 .ts 文件树便于人读")
    args = parser.parse_args()
    if args.dump:
        return dump(args.dump)
    return check() if args.check else freeze()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
