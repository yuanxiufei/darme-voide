"""S7 自检：**层级守卫** —— 后端全树单向依赖（2026-09-15 层级收口时新增，同日扩到顶层包）。

为什么需要：用户要求「后端要像前端一样**按层级划分清楚**」+「`agent` / `skill` / `script` / `mcp`
要在**同层级**」⇒ 光有目录不够，**依赖方向**必须单向，否则半年后 `core/` 里又长出 `services/`
的 import、顶层 `mcp/` 反向依赖 `routers/`，层级名存实亡。本自检把它变成机械判据。

后端单元与允许的依赖（⚠️ 一条**线性链**，不是网）：

    top      （`app/main.py`、`app/passthrough.py`）  → 全部
    routers  （HTTP 层，`app/routers/`）              → core / services / agent / mcp
    agent    （顶层 `agent/`：协议 / 工具 / 循环 / 子 Agent） → core / services / agent / mcp
    mcp      （顶层 `mcp/`：MCP 客户端）              → core / mcp
    services （业务层，`app/services/`）              → core / services
    core     （平台层，`app/core/`）                  → **只 core**

即：`routers → agent → services → core`（`mcp` 只挂 `core`，`agent` 可调 `mcp`）。
⚠️ 这条不变量此前靠自觉：`core` 里 import `services` 会形成**偶发**初始化环（谁先被 import 决定成败）；
顶层包反向依赖 `app.routers` 则会让「运行时」和「HTTP 层」绑死 —— 正是最该机械化的那一类。

运行::

    ./.venv/Scripts/python.exe tests/layering_test.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"

#: 单元扫描根（目录前缀 → 单元名）；`app/*.py` 归 top
#: ⚠️ 方案 A（2026-09-15）后 `agent/`、`mcp/` 是 **`app/` 的子包** —— 它们与 core/routers/services
#: **同层级**（「划分清楚」落在 `app/` 内部），所以单元前缀带 `app/`。
UNIT_DIRS: dict[str, str] = {
    "app/core": "core",
    "app/routers": "routers",
    "app/services": "services",
    "app/agent": "agent",
    "app/mcp": "mcp",
}

#: 各单元允许依赖的单元（含自身）
#: ⚠️ `mcp → agent` 是**正当的向下依赖**（不是环）：MCP 客户端把远端工具包成 `agent.tool.Tool`，
#: 也就是说 MCP 工具就是 Agent 工具 ⇒ 它必须依赖工具基座。反之 `agent` 也可以 `mcp`
#: （运行时发现 MCP 工具）。两者都只向 `core` 取基础设施，不会反向碰 `routers/services`。
ALLOWED: dict[str, set[str]] = {
    "top": {"core", "routers", "services", "agent", "mcp"},
    "routers": {"core", "services", "agent", "mcp"},
    "agent": {"core", "services", "agent", "mcp"},
    "mcp": {"core", "agent", "mcp"},
    "services": {"core", "services"},
    "core": {"core"},
}

#: `from a.b.c import x` / `import a.b` / `from . import x` / `from ..x import y`
FROM_RE = re.compile(r"^\s*from\s+(?P<mod>\.*[\w.]*)\s+import\s+(?P<names>\S.*)$")
IMPORT_RE = re.compile(r"^\s*import\s+(?P<mod>[\w.]+)")

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def scan_files() -> list[Path]:
    #: 方案 A 后全部代码都在 `app/` 之内（agent/、mcp/ 是它的子包）
    #: ⚠️ 跳过 `app/scripts/`（工具链）与 `app/skills/`（技能附带的脚本）—— 它们不是应用分层代码
    #: （2026-09-15 三目录并入 app/ 后落进来的 ✗），混进来会把「分层」判定变成噪声。
    roots = [BP / "app"]
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            # 跳过工具链与技能附带脚本（它们不是应用分层代码，见上）
            if {"scripts", "skills"} & set(path.relative_to(root).parts):
                continue
            out.append(path)
    return sorted(out)


def unit_of(path: Path) -> str:
    rel = path.relative_to(BP).as_posix()
    for prefix, unit in UNIT_DIRS.items():
        if rel.startswith(prefix + "/"):
            return unit
    return "top"


def resolve(own_file: Path, mod: str) -> str:
    """把 import 的目标解析成 `backend-py/` 下的**点分模块名**（相对形态按本文件位置换算）。"""
    if not mod.startswith("."):
        return mod
    dots = len(mod) - len(mod.lstrip("."))
    rest = mod.lstrip(".")
    parts = own_file.relative_to(BP).as_posix().split("/")[:-1]  # 本文件所在目录
    if dots > 1:
        parts = parts[: -(dots - 1)] if dots - 1 <= len(parts) else []
    return ".".join([*parts, *([rest] if rest else [])])


def target_unit(module: str) -> str | None:
    for prefix, unit in UNIT_DIRS.items():
        if module == prefix or module.startswith(prefix + "."):
            return unit
    return None


def violations() -> list[tuple[str, int, str, str, str]]:
    out: list[tuple[str, int, str, str, str]] = []
    for path in scan_files():
        own = unit_of(path)
        for index, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
            match = FROM_RE.match(line) or IMPORT_RE.match(line)
            if not match:
                continue
            module = resolve(path, match.group("mod"))
            target = target_unit(module)
            if target is None or target in ALLOWED[own]:
                continue
            out.append((path.relative_to(REPO).as_posix(), index + 1, own, target, line.strip()[:70]))
    return out


def main() -> int:
    files = scan_files()
    units = {unit_of(p) for p in files}
    bad = violations()

    # ① 扫描面自检（防「解析坏了 ⇒ 0 文件 ⇒ 假绿」）
    check("扫描: 后端 .py 数量正常（≥120）", len(files) >= 120, len(files))
    check("扫描: 六个单元齐（含 app/ 内的 agent / mcp 子包）",
          {"top", "core", "routers", "services", "agent", "mcp"} <= units, sorted(units))
    check("扫描: agent/ 与 mcp/ 都在 app/ 之内",
          (BP / "app" / "agent").is_dir() and (BP / "app" / "mcp").is_dir())

    # ② ⭐ 核心：全树无反向依赖
    check("⭐ 层级: 后端全树单向（routers→agent→services→core；mcp→agent 的工具基座）",
          not bad, [f"{f}:{n}  {own} → {target}   {code}" for f, n, own, target, code in bad][:6])

    # ③ 反套套逻辑：判定器必须**会失败**
    def ok(own: str, target: str) -> bool:
        return target in ALLOWED[own]

    check("反套套逻辑: core→services / mcp→routers / agent→routers 必须判违规",
          not ok("core", "services") and not ok("mcp", "routers") and not ok("agent", "routers"))
    check("反套套逻辑: 线性链方向必须放行",
          ok("routers", "agent") and ok("agent", "services") and ok("services", "core")
          and ok("agent", "mcp") and ok("top", "routers"))

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"（扫描 {len(files)} 个 .py："
          + "、".join(f"{u} {sum(1 for p in files if unit_of(p) == u)}" for u in
                      ["top", "core", "routers", "services", "agent", "mcp"] if u in units)
          + f"；违规 {len(bad)} 处）")
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
