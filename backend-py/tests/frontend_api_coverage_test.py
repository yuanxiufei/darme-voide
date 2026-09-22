"""S7 自检：**前端调用点 ↔ 后端路由**覆盖 —— 删 `backend/` 之后唯一后端的安全网。

为什么需要（2026-09-15 删库当天补）：路由守卫（``route_parity_test.py``）证明的是
「**Node 曾有过的 224 条路径**都被覆盖」，它**看不见前端**。删库后 Python 是唯一服务方 ⇒
只要前端调了一个 Python 没实现的路径，就是**线上 501**（用户点一下按钮就报错）。
本自检把前端源码里的调用点抽出来，逐条对 ``app.openapi()`` 的路由表**反查**，锁住这条不变量。

提取规则（每条都是踩过坑之后定下来的）：
* ``api.get/post/put/del('<path>')``（含泛型 ``api.get<T>('/x')``）⇒ 直接取方法 + 路径；
* ``fetch('/api/v1/…', { method: 'POST' })`` ⇒ **方法必须从调用现场读** —— 先按 GET 处理时，
  4 条 ``POST`` 动作端点被误报成「缺失」✗；
* 路径里 ``${...}`` 归一化成 ``{}``（后端模板是 ``{episode_id}``）；**查询串要剥掉**（``?a=b``）；
* 含空格 / 残留 ``$`` 或 ``{`` 的串（模板三元表达式）直接丢弃 —— 它们不可能是真路径。
* ⚠️⚠️ **注释里的示例调用也会被抽出来** ✗（提取是纯文本扫描 ✓，不看语法上下文 ✓）——
  2026-09-22 实测踩到：`useApi.ts` 里一条"这里曾经写错成 …"的注释写了调用形状 ✓✗ ⇒
  被当成真调用点，反查后端**当然找不到** ⇒ 本套当场红 ✓。⇒ 写注释**别用调用形状** ✗。

基线（2026-09-15 实测）：**166 个调用点 ｜ 166 已注册 ｜ 0 未匹配**。

运行::

    ./.venv/Scripts/python.exe tests/frontend_api_coverage_test.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="frontend_cov_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FRONT = REPO / "frontend" / "app"

_CALL_RE = re.compile(r"api\.(get|post|put|del)\s*(?:<[^>]*>)?\s*\(\s*[`'\"]([^`'\"]+)[`'\"]")
_ABS_RE = re.compile(r"[`'\"](/api/v1/[^`'\"]+)[`'\"]")
_METHOD_RE = re.compile(r"method:\s*['\"]([A-Za-z]+)['\"]")
_TEMPLATE_PARAM = re.compile(r"\$\{[^}]*\}")
_METHODS = {"get": "GET", "post": "POST", "put": "PUT", "del": "DELETE"}

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def normalize(raw: str) -> str | None:
    """前端路径字面量 → 后端路由模板形态；不可能是真路径的返回 ``None``。"""
    path = raw.split("?")[0]
    path = _TEMPLATE_PARAM.sub("{}", path)
    if not path.startswith("/") or " " in path or "$" in path or "{" in path.replace("{}", ""):
        return None
    return path


def collect_calls() -> tuple[dict[tuple[str, str], set[str]], int, int]:
    """返回 ``(调用点, 文件数, 来自 fetch 字面量的条数)``。"""
    calls: dict[tuple[str, str], set[str]] = {}
    absolute = 0
    files = [p for p in FRONT.rglob("*")
             if p.is_file() and p.suffix in (".ts", ".vue", ".js")]
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = file.relative_to(REPO).as_posix()
        for match in _CALL_RE.finditer(text):
            path = normalize(match.group(2))
            if path:
                calls.setdefault((_METHODS[match.group(1)], path), set()).add(rel)
        for match in _ABS_RE.finditer(text):
            path = normalize(match.group(1)[len("/api/v1"):])
            if not path:
                continue
            tail = text[match.end(): match.end() + 240]
            found = _METHOD_RE.search(tail)
            absolute += 1
            calls.setdefault(((found.group(1).upper() if found else "GET"), path), set()).add(rel)
    return calls, len(files), absolute


def backend_patterns() -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for route, ops in app.openapi()["paths"].items():
        for method in ops:
            body = re.sub(r"\{[^}]+\}", "[^/]+",
                          re.escape(route).replace(r"\{", "{").replace(r"\}", "}"))
            patterns.append((method.upper(), re.compile("^" + body + "$")))
    return patterns


def matched(method: str, full: str, patterns: list[tuple[str, re.Pattern[str]]]) -> bool:
    return any(m == method and rx.match(full) for m, rx in patterns)


def main() -> int:
    calls, file_count, absolute = collect_calls()
    patterns = backend_patterns()

    unmatched = [(method, "/api/v1" + raw, sorted(where)[0])
                 for (method, raw), where in sorted(calls.items())
                 if not matched(method, "/api/v1" + raw, patterns)]

    # ① 防「提取规则坏了 ⇒ 0 个调用点 ⇒ 假绿」
    check("提取: 前端调用点数量正常（≥100；基线 166）", len(calls) >= 100, len(calls))
    # ② 防「只扫到 useApi 一种形态」（fetch 绝对字面量那条路也得真在用）
    check("提取: 也扫到了 `fetch('/api/v1/…')` 形态（≥1）", absolute >= 1, absolute)
    check("提取: 后端路由表规模正常（≥200 组合）", len(patterns) >= 200, len(patterns))
    # ③ ⭐ 核心不变量：前端每一个调用点，后端都实现了
    check("⭐ 覆盖: 前端**全部**调用点都能在后端路由表里找到（0 未匹配）",
          not unmatched, unmatched[:6])
    # ④ 反套套逻辑：匹配器必须**会拒绝**（否则上面那条恒真、毫无意义）
    check("反套套逻辑: 故意造一条不存在的路径 -> 匹配器必须拒绝",
          not matched("GET", "/api/v1/__definitely_not_a_route__", patterns)
          and not matched("POST", "/api/v1/storyboards/1/__nope__", patterns))

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"（扫描 {file_count} 个前端文件；调用点 {len(calls)} 个；后端 {len(patterns)} 条 (method,path)）")
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
