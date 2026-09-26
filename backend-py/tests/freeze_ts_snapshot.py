"""把守卫依赖的 TS 源码**冻结**成 Python 快照 —— 「删 ``backend/``，但保住守卫的价值」。

背景：``route_parity_test.py`` 的九道守卫靠**读 TS 源码**来证明「Python 侧的路由表/常量/提示词
没漂移」。删掉 ``backend/`` 之后源码没了，守卫会集体失效。

做法：把守卫真正读到的那些文件**逐字**收进一个 Python 模块
（``tests/frozen_ts_source.py``，**生成物**）；守卫侧的 ``_SRC_ROOT`` 在「真源码缺失」或
``PARITY_USE_FROZEN=1`` 时调用 ``frozen_ts.snapshot_root()`` **物化到临时目录** ——
于是 15 处调用点一行都不用改，删库后守卫仍在比「Python 现在 vs TS 当初」。

⚠️ 2026-09-15 起快照**不再存 ``.ts`` 文件树**（改存 Python 模块）：仓库里不再有 TS 文件，
且 diff 能直接看到改的是哪一条文本。要**看原文**用 ``--dump``：

用法::

    python tests/freeze_ts_snapshot.py              # 冻结（生成/覆盖 frozen_ts_source.py）
    python tests/freeze_ts_snapshot.py --check       # 只检查覆盖是否完整
    python tests/freeze_ts_snapshot.py --dump tmp/frozen_ts   # 物化成 .ts 文件便于人读

⚠️ **删 ``backend/`` 之后仍可跑**（2026-09-15 补的能力）：取源顺序是
**真源码 → 现有快照（逐字）→ git 历史**（``HEAD`` 里已含删除时，回溯到「最后一个还有该文件」
的提交 —— 删库当天现场常已被提交成删除，只试 ``HEAD`` 会直接 fatal）。
- 「现有快照优先于 git」是**刻意的**：快照冻结于「删库前的现场」，而 git HEAD 可能落后于现场
  （本项目真实发生过：删库前改过 ``services/local-model-scan.ts`` 但没提交 ⇒ 若一律从 git 重建，
  那次改动会被**静默回退成旧版**）。所以已冻结的条目**逐字保留**，只有**新增件**才去 git 取。
- 因此本脚本在删库后仍能「扩快照」（例如给守卫/自检新加一处 TS 读取）。
⚠️ ``SOURCE_ROOT`` 是「**冻结当时的现场路径**」的戳 ✓（写在生成物里 ✓ 供人读 ✓ ——
   **没有任何代码读它** ✗）：它指的 ``backend/src`` 现在**本就不存在**（已拆成 ``backend-py/`` + ``frontend/`` ✓）
   ⇒ **这是正常的、有意保留的** ✓。⚠️ **别为了「修这条路径」重跑本脚本** ✗ —— 重跑只会把它换成
   **另一个同样不存在的路径** ✗✗（本仓现在也没有 ``backend/src`` ✓）并丢掉历史现场 ✓，
   而**功能性收益为零** ✗。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TS_SRC = REPO / "backend" / "src"
TESTS = Path(__file__).resolve().parent
FROZEN_MODULE = TESTS / "frozen_ts_source.py"
sys.path.insert(0, str(TESTS))  # 便于 `from frozen_ts import load`（快照读取器就在同目录）

#: 守卫读到的**具体文件**（相对 ``backend/src``）；与自动发现取并集
FILES: tuple[str, ...] = (
    "shared/prompt-utils.ts",
    "shared/prompt-blocks.ts",
    "shared/visual-graph.ts",
    "shared/camera-movement-guides.ts",
    "services/text-generation.ts",
    "agents/index.ts",
    # ⚠️ 这两份是 **smoke_test.py** 要的（校验「models 表集/列集 == Node DDL」）：守卫不读它们，
    #    但自检读 ⇒ 删 ``backend/`` 后同样必须能在快照里找到（2026-09-15 删库当天补进来）。
    "db/index.ts",
    "db/schema.ts",
)

#: 守卫按**目录**读取/列举的（整目录 ``*.ts`` 都要）
DIRS: tuple[str, ...] = (
    "routes",
    "services/adapters",
    "agents",
)

#: 扫描哪些文件来发现「谁还引用了哪些 .ts」
#: ⚠️ **不止守卫**：任何**自检**里写死的 TS 路径都要在快照里，否则删 ``backend/`` 当天才炸。
#: 本项目真实栽过：``smoke_test.py`` 读 ``db/index.ts`` / ``db/schema.ts``（校验表和列集），
#: 不在这个清单里 ⇒ 自动发现漏了 ⇒ 删库后**导入期 FileNotFoundError**、整套冒烟直接崩（2026-09-15）。
_GUARD_SOURCES = ("route_parity_test.py", "parity_diff_test.py", "smoke_test.py")

#: ``_SRC_ROOT / "a" / "b.ts"``（**可能由多段字符串字面量拼出来**，故要整链捕获）
_SRC_REF = re.compile(r'_SRC_ROOT((?:\s*/\s*r?"[^"]+")+)')
#: 普通字符串字面量形态的相对路径（如 `("services/technical-qc.ts", "CONST", "mod")`）
_PLAIN_TS = re.compile(r'"((?:[\w.-]+/)+[\w.-]+\.ts)"')


def _ts_exists(relative: str) -> bool:
    """该相对路径算不算「守卫真读的文件」。

    * 真源码还在 ⇒ 看 ``backend/src/<relative>`` 是否存在（滤掉注释/文档里的幽灵路径）；
    * 真源码已删（**删库之后**）⇒ 改看**快照里有没有它** —— 这样 ``--check`` 仍能拦住
      「守卫新增了读文件、却没进快照」这类漂移（否则删库后这条能力会**静默消失**）。
    """
    if TS_SRC.is_dir():
        return (TS_SRC / relative).is_file()
    try:
        from frozen_ts import load  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - 快照缺失时按「不认为是必需件」处理，由 check() 报警
        return False
    return relative in load()


def discover_refs() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """从守卫源码里**自动发现**它引用的 TS 文件 / 目录（相对 ``backend/src``）。

    ⚠️ 为什么必须自动发现：``FILES`` 是手写清单，**新增一处「守卫读文件」时极易漏加** ——
    漏了不会立刻报错，直到「删掉 ``backend/`` 后跑冻结模式」才炸。本项目**真实发生过两次**：
    ``services/consistency-qc.ts``（跨字面量拼接形态）与 ``services/technical-qc.ts``
    （表驱动普通字符串形态）。这里**两种形态都扫**，与手写清单**取并集**。

    普通字符串形态按 ``_ts_exists()`` 过滤（真源码在时看文件、删库后看快照）；
    ``_SRC_ROOT`` 形态本就带目录结构，不加存在性过滤。
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
        for candidate in _PLAIN_TS.findall(text):
            if _ts_exists(candidate):
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


def _git_run(*args: str) -> tuple[bool, str]:
    """跑一条**只读** git 命令，返回 ``(是否成功, 输出)``。"""
    try:
        done = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", check=False)
    except OSError:
        return False, ""
    return done.returncode == 0, done.stdout


def _git_text(relative: str) -> str | None:
    """从 git 历史里取该文件的**最后一个还存在的版本**。

    ⚠️ 不能只试 ``HEAD``：删库当天现场往往**已被提交成「删除」** ⇒ ``HEAD:…`` 直接 fatal。
    本项目实测：删掉 ``backend/`` 后 ``HEAD`` 已不含 ``backend/src/db/index.ts``，
    但它在上一个提交里还在（`git log --all -- <path>` 能看到）。所以在那些提交里逐个试。
    """
    path = f"backend/src/{relative}"
    ok, text = _git_run("show", f"HEAD:{path}")
    if ok:
        return text
    ok, revs = _git_run("log", "--all", "--format=%H", "--", path)
    for rev in (revs.split() if ok else []):
        ok, text = _git_run("show", f"{rev}:{path}")
        if ok:
            return text
    return None


def _existing_snapshot() -> dict[str, str]:
    """读现有快照（没有/读不了就空 dict）—— 用于**逐字保留删库前的现场**。"""
    if not FROZEN_MODULE.is_file():
        return {}
    try:
        from frozen_ts import load  # noqa: PLC0415
        return dict(load())
    except Exception:  # noqa: BLE001
        return {}


def _resolve(relative: str, snapshot: dict[str, str]) -> tuple[str | None, str]:
    """取一份 TS 文本：**真源码 → 现有快照 → git 历史**。返回 ``(文本, 来源标签)``。"""
    source = TS_SRC / relative
    if source.is_file():
        return source.read_text(encoding="utf-8"), "真源码"
    if relative in snapshot:
        return snapshot[relative], "快照"
    text = _git_text(relative)
    if text is not None:
        return text, "git 历史"
    return None, "缺失"


def _dir_names(relative: str, snapshot: dict[str, str]) -> list[str]:
    """某目录下要收的 ``*.ts``（真源码 ∪ 现有快照 —— **取并集**，只增不减）。

    ⚠️ 删库后**不再去 git 列目录**：HEAD 已不含那棵树（见 ``_git_text`` 的注释），
    而目录类清单（routes / services/adapters / agents）本来就已经在快照里了 ⇒
    「快照做底 + 真源码在时取并集」既够用，又不会凭空丢条目。
    """
    names: set[str] = set()
    source_dir = TS_SRC / relative
    if source_dir.is_dir():
        # ⚠️ **递归**：守卫会读 `agents/tools/*.ts` 这类子目录（漏了就会「TS 侧抽到空集」）
        names |= {p.relative_to(source_dir).as_posix() for p in source_dir.rglob("*.ts")}
    prefix = relative + "/"
    names |= {key[len(prefix):] for key in snapshot if key.startswith(prefix)}
    return sorted(names)


def freeze() -> int:
    """冻结/扩快照。⚠️ 删 ``backend/`` 后照样能跑（见模块 docstring 的取源顺序）。"""
    files_to_copy, dirs_to_copy = _effective()
    auto = len(files_to_copy) - len(FILES)
    snapshot = _existing_snapshot()
    data: dict[str, str] = {}
    origins: Counter[str] = Counter()
    for relative in files_to_copy:
        text, origin = _resolve(relative, snapshot)
        if text is None:
            print(f"❌ 缺少文件：{relative}（真源码 / 快照 / git 历史 都没有）", file=sys.stderr)
            return 2
        # ⚠️ 只有**首次**写入才计来源：手写清单与目录清单有重叠（如 `agents/index.ts`），
        #    否则「来源统计」会大于实际条目数（本文件刚踩过：76 条却打出 78 ✗）。
        if relative not in data:
            origins[origin] += 1
        data[relative] = text
    for relative in dirs_to_copy:
        names = _dir_names(relative, snapshot)
        if not names:
            print(f"❌ 目录里没有 .ts：{relative}（真源码 / 快照 / git 历史 都没有）", file=sys.stderr)
            return 2
        for name in names:
            key = f"{relative}/{name}"
            text, origin = _resolve(key, snapshot)
            if text is None:
                print(f"❌ 缺少文件：{key}", file=sys.stderr)
                return 2
            if key not in data:  # 同上：重叠条目只计一次
                origins[origin] += 1
            data[key] = text
    _write_module(data)

    total = sum(len(text) for text in data.values())
    how = "、".join(f"{name} {count} 个" for name, count in origins.most_common())
    print(f"✅ 冻结完成：{len(data)} 个文件 / {total / 1024:.0f} KB -> {FROZEN_MODULE.name}")
    print(f"   手写清单 {len(FILES)} 个 + 自动发现 {auto} 个；目录 {len(dirs_to_copy)} 个；取源：{how}"
          + ("" if TS_SRC.is_dir() else "（真源码已删 ⇒ 新增件来自现有快照 / git HEAD）"))
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
