"""自检：**引擎 → 服务层的单向接线口**（`app/services/engine/hooks.py`，2026-11-07 补 ✓）。

## 为什么单开一份

`hooks.py` 是为解决一个**假绿**事故而生的 ✓✗：引擎要读三块业务数据（跑次索引 / 知识台账 /
质检复核），原先在函数里 late import 服务层模块 —— 形式上躲开了「顶层依赖」✓，方向却仍是
引擎 → 服务层 ✓✗ ⇒ 引擎被单独拉起（CLI / 纯引擎进程）时隐式拖上 sqlite 与业务表，
`tuning.suggest()` 表现为「ImportError 或空数据」（**不报错地给 0 条建议 = 假绿** ✓✗）。

现在反过来：服务层在 `app/main.py` 启动时登记**模块对象** ✓，引擎只查 `hooks.SOURCES` ✓。
本文件钉十一件事：

1. **三个源名固定** ✓：`run_index` / `ledger` / `qc_reviews`（改名 ⇒ 引擎读空且不报错 ✗）；
2. **没登记就说没登记** ✓：`get()` 回 ``None`` / `provided()` 回 ``False``（调用点必须自己表态 ✓✗）；
3. **登记后是同一个对象** ✓（不是拷贝、不是快照 —— 那三个模块自带缓存 ✓）；
4. **写错名字整体拒** ✓✗：`provide(ledger=x, nope=y)` **不许**先把 `ledger` 登记进去再报错；
5. **生产接线逐字对得上** ✓：`app/main.py` 那次 `provide(...)` 的关键字集合 == :data:`SOURCES`
   （少登记一个 ⇒ 引擎侧静默读到 ``None`` ✓✗）；
6. **下游真的按「没接入」走** ✓：`tuning._scan()` 未登记时回 ``([], 0, False)``，不是 ``([], 0, True)``
   （后者会被读成「没有失败跑次 ⇒ 一切正常」✓✗）；登记了假索引则走真数据 ✓；
7. **引擎侧零越界** ✓：`hooks.py` 顶层只 import ``typing``（不 import 服务层任何业务模块 ✓）；
8. **读的一侧拼错名字也当场炸** ✓✗：`get()` / `provided()` 传未知名字 ⇒ `ValueError`
   （静默回 ``None`` / ``False`` 会把「我拼错了」伪装成「服务层没登记」✓✗）；
   `provide(x=None)` 同样拒 ✓（等价于没登记，却让 `provided()` 说「登记了」⇒ 两口自相矛盾）；
9. **表与读者双向对账** ✓：`hooks.SOURCES` 每一项在 `engine/*.py` 里都有真读者
   （`hooks.get("…")` / `hooks.provided("…")`），且引擎侧每个字面量都在表内
   —— 「表里有、没人读」＝死源，「有人读、表里没有」＝读空 ✓✗；
10. **谁调公开口谁登记** ✓：调 `tuning.suggest/set_field` 或 `qc_feedback.*` 的套件必须自己
   `provide(...)` —— 少写一行不会报「接线问题」，而是以**业务断言红**的形状出现 ✓✗
   （2026-11-07 连踩两次，两次都先怀疑业务逻辑 ✓）；
11. **接线状态看得见** ✓：`hooks.status()` ＋ `doctor` 的「接线」行逐条点名未登记的源，
   并写明是「没接入」**不是**没有数据 ✓✗；`doctor --json` 的 `wiring` 与 `status()` 逐字段相同，
   而 `capabilities()` **不带** wiring（那是静态表，要跟 `docs/env/*.json` 逐字段对 ✓）。

⚠️ 不碰数据库 ✗（本文件全用假对象 ✓）。
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import hooks, tuning  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


class _FakeIndex:
    """跑次索引的替身：只实现 `_scan` 用到的那一个方法 ✓。"""

    def __init__(self, items: list[dict] | None = None, *, available: bool = True) -> None:
        self.items = items or []
        self.available = available
        self.calls: list[dict] = []

    def query(self, **kwargs) -> dict:
        self.calls.append(dict(kwargs))
        return {"items": self.items, "total": len(self.items), "available": self.available}


# ══════════════════════════════════════════════════════════════════════════
# ① 三个源名固定
# ══════════════════════════════════════════════════════════════════════════
def case_source_names() -> None:
    check("① 三个源名逐字固定（改名 ⇒ 引擎读空且不报错 ✗）",
          hooks.SOURCES == ("run_index", "ledger", "qc_reviews"), hooks.SOURCES)
    #: ⚠️ 别把名字再抄一遍 ✓✗（2026-11-08 加 `status()` 时这里就是抄的那份，当场红）：
    #: 从模块自己的 `vars()` 推「公开函数」，只认 `__module__` 是本模块的（`typing.Any` 这类
    #: 被 import 进来的不算 ✓）。
    public = {"SOURCES"} | {
        name for name, obj in vars(hooks).items()
        if not name.startswith("_") and callable(obj)
        and getattr(obj, "__module__", "") == hooks.__name__
    }
    check("① `__all__` 与实现一致（漏一个 ⇒ `from hooks import *` 拿不到 ✗）",
          set(hooks.__all__) == public, sorted(set(hooks.__all__) ^ public))


# ══════════════════════════════════════════════════════════════════════════
# ② 没登记就说没登记（不是给空容器 ✓✗）
# ══════════════════════════════════════════════════════════════════════════
def case_unprovided() -> None:
    hooks.clear()
    for name in hooks.SOURCES:
        check(f"② 未登记 `{name}` ⇒ `get()` 回 None（不是 {{}} / [] ✗✗）",
              hooks.get(name) is None, hooks.get(name))
        check(f"② 未登记 `{name}` ⇒ `provided()` 回 False", hooks.provided(name) is False)


# ══════════════════════════════════════════════════════════════════════════
# ③ 登记后是同一个对象（不是拷贝）
# ══════════════════════════════════════════════════════════════════════════
def case_provide() -> None:
    hooks.clear()
    index, ledger, qc = _FakeIndex(), object(), object()
    receipt = hooks.provide(run_index=index, ledger=ledger, qc_reviews=qc)
    check("③ 登记后 `get()` 回**同一个对象**（拷贝/快照会丢模块自带的缓存 ✓✗）",
          hooks.get("run_index") is index and hooks.get("ledger") is ledger
          and hooks.get("qc_reviews") is qc)
    check("③ 回执把每个源映射到**类型名**（人看得懂 ✓）",
          receipt == {"run_index": "_FakeIndex", "ledger": "object", "qc_reviews": "object"},
          receipt)
    check("③ 三个源 `provided()` 全 True",
          all(hooks.provided(n) for n in hooks.SOURCES))
    hooks.clear()
    check("③ `clear()` 之后全部回到未登记（测试用 ✓）",
          all(hooks.get(n) is None for n in hooks.SOURCES))


# ══════════════════════════════════════════════════════════════════════════
# ④ 写错名字整体拒（不许半登记 ✓✗）
# ══════════════════════════════════════════════════════════════════════════
def case_unknown_name() -> None:
    hooks.clear()
    raised = ""
    try:
        hooks.provide(ledger=object(), nope=object())
    except ValueError as err:
        raised = str(err)
    check("④ 未知源名 ⇒ 当场 ValueError 且点名 `nope` ✗",
          "nope" in raised and "ValueError" not in raised, raised or "没报错")
    check("④ ⭐ 未知源名时**同一个调用里的合法源也不许留下**（半登记 = 更难查 ✓✗）",
          hooks.get("ledger") is None and not hooks.provided("ledger"),
          hooks.get("ledger"))
    check("④ 反验：全是合法名字 ⇒ 不报（判据不许见谁都喊 ✗）",
          hooks.provide(ledger=object())["ledger"] == "object")
    hooks.clear()


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 生产接线逐字对得上
# ══════════════════════════════════════════════════════════════════════════
def case_production_wiring() -> None:
    source = (BACKEND_PY / "app" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    provided: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "provide"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "engine_hooks"):
            provided |= {kw.arg for kw in node.keywords if kw.arg}
    check("⑤ `app/main.py` 真调了 `engine_hooks.provide(...)`（不调 ⇒ 引擎全读 None ✗）",
          bool(provided), sorted(provided))
    check("⑤ ⭐ 登记的键与 `hooks.SOURCES` **逐字相同**（少一个 ⇒ 那条通路静默读空 ✗✗）",
          provided == set(hooks.SOURCES), sorted(provided ^ set(hooks.SOURCES)))


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 下游真的按「没接入」走
# ══════════════════════════════════════════════════════════════════════════
def case_downstream_reports_unavailable() -> None:
    hooks.clear()
    items, total, available = tuning._scan()
    check("⑥ 未登记 ⇒ `tuning._scan()` 回 `([], 0, False)`（False 才是「没接入」✗✗）",
          (items, total, available) == ([], 0, False), (items, total, available))
    check("⑥ ⭐ 不许回 `available=True`（那会被读成「没有失败跑次 ⇒ 一切正常」✓✗）",
          available is False, available)

    fake = _FakeIndex([{"stage": "zimage", "ok": True}])
    hooks.provide(run_index=fake)
    items, total, available = tuning._scan(stage="zimage")
    check("⑥ 登记了假索引 ⇒ 走真数据（`items` / `total` / `available` 三项都来自它 ✓）",
          items == [{"stage": "zimage", "ok": True}] and total == 1 and available is True,
          (items, total, available))
    check("⑥ `total` 与 `len(items)` 分开取（扫到 500 条就说「共 500 条」✗✗）",
          fake.calls and fake.calls[0].get("limit") == tuning._SCAN_LIMIT, fake.calls)
    hooks.clear()


# ══════════════════════════════════════════════════════════════════════════
# ⑦ 引擎侧零越界
# ══════════════════════════════════════════════════════════════════════════
def case_no_service_import() -> None:
    path = Path(hooks.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tops: list[str] = []
    for node in tree.body:  # 只看顶层（函数内的 late import 才是老毛病的形状 ✗）
        if isinstance(node, ast.Import):
            tops += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            tops.append(node.module)
    check("⑦ `hooks.py` 顶层只 import `typing` 与 `__future__`（引擎侧不许碰服务层 ✗）",
          set(tops) <= {"typing", "__future__"}, tops)

    # ⚠️ 必须扫 **AST 的 import 节点**，不能全文 grep 文本 ✓✗：模块 docstring 里就写着
    #    `from app.services.engine import hooks` 这句**用法示例** —— 文本搜会把示例当违规
    #    （2026-11-07 自己踩到：22/23）。扫全文件 import 节点才既能拦「函数内 late import」
    #    又不误伤注释/文档 ✓。
    everything: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            everything += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            everything.append(node.module)
    bad = [m for m in everything if m == "app" or m.startswith("app.services")]
    check("⑦ 全文件（含函数内 late import）零 `app.services.*` —— 只扫 AST 节点，不扫 docstring ✓",
          not bad, bad)


def _readers_in(directory: Path) -> set[str]:
    """扫一个目录里的**字面量**读取点：`hooks.get("…")` / `hooks.provided("…")` ✓。

    ⚠️ 只认 ``hooks.<attr>(<字符串字面量>)`` 这一种形状 ✓：变量名当参数（``hooks.get(name)``）
    扫不到 ⇒ 那本身就该被禁（调用点写死名字才核得动 ✓✗）。
    """
    names: set[str] = set()
    for path in sorted(directory.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr in ("get", "provided")):
                continue
            if not (isinstance(func.value, ast.Name) and func.value.id == "hooks"):
                continue
            if (node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                names.add(node.args[0].value)
    return names


# ══════════════════════════════════════════════════════════════════════════
# ⑧ 读的一侧拼错名字也当场炸
# ══════════════════════════════════════════════════════════════════════════
def case_reader_side_typo() -> None:
    hooks.clear()
    for label, fn in (("get", hooks.get), ("provided", hooks.provided)):
        raised = ""
        try:
            fn("runindex")
        except ValueError as err:
            raised = str(err)
        check(f"⑧ `{label}()` 传未知名字 ⇒ 当场 ValueError，并点名 `runindex` "
              f"（静默回 None/False 会把「我拼错了」伪装成「服务层没登记」✓✗）",
              "runindex" in raised, raised or "没报错")
    check("⑧ 反验：合法名字 + 没登记 ⇒ 只是回 None，**不**抛（没登记 ≠ 拼错 ✓✗）",
          hooks.get("ledger") is None and hooks.provided("ledger") is False)

    raised = ""
    try:
        hooks.provide(run_index=None)
    except ValueError as err:
        raised = str(err)
    check("⑧ `provide(run_index=None)` ⇒ ValueError（等价于没登记，却会让 provided() 说「登记了」✓✗）",
          "run_index" in raised, raised or "没报错")
    check("⑧ ⭐ 被拒的 None 不许留下痕迹（`provided('run_index')` 仍 False ✓✗）",
          hooks.provided("run_index") is False and hooks.get("run_index") is None)
    hooks.clear()


# ══════════════════════════════════════════════════════════════════════════
# ⑨ 表与读者双向对账
# ══════════════════════════════════════════════════════════════════════════
def case_sources_reconcile() -> None:
    engine_dir = BACKEND_PY / "app" / "services" / "engine"
    readers = _readers_in(engine_dir)

    check("⑨ 表与读者**逐字相同**：`hooks.SOURCES` 每一项都有人读（表里有、没人读 = 死源 ✗）、"
          "引擎侧每个字面量都在表内（表里没有 ⇒ 读的那一刻 ValueError ✗）",
          readers == set(hooks.SOURCES), sorted(readers ^ set(hooks.SOURCES)))
    check("⑨ 三个源都被真读到（`run_index` 两处读者不影响本判据：比对的是**名字集合** ✓）",
          {"run_index", "ledger", "qc_reviews"} <= readers, sorted(readers))

    # 反验：扫描器不是恒空（恒空 ⇒ 上面两条会假绿 ✓✗）
    probe = Path(tempfile.mkdtemp(prefix="hooks-probe-"))
    (probe / "ghost.py").write_text(
        "from app.services.engine import hooks\n"
        "def f():\n"
        "    return hooks.get('ghost'), hooks.provided('ghost2')\n",
        encoding="utf-8")
    check("⑨ 反验：扫描器确实能看见读取点（造个 `hooks.get('ghost')` 必须被抓到 ✓✗）",
          _readers_in(probe) == {"ghost", "ghost2"}, sorted(_readers_in(probe)))


# ══════════════════════════════════════════════════════════════════════════
# ⑩ 谁调「要有真数据才谈得上对」的公开口，谁就得先登记
# ══════════════════════════════════════════════════════════════════════════
#: 这些口在**没登记**时的行为是"降级但如实说" ✓ —— 于是测试里少写一行 `provide(...)`
#: 不会被指出来，而是以**业务断言红**的形状出现 ✓✗（2026-11-07 连踩两次：`tuning_suggest`
#: 读不到 `tuning.json`、`qc_feedback` ⑥ 拿到"没数据"的理由而不是"被闸门拦下"的理由 ✓）。
_NEEDS_SOURCE = {"tuning": ("suggest", "set_field"),
                 "qc_feedback": ("stage_stats", "pass_by_steps", "gate", "summary_line")}


def case_callers_register() -> None:
    need: set[str] = set()
    ok: set[str] = set()
    for path in sorted((BACKEND_PY / "tests").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls_source, provides = False, False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.attr in _NEEDS_SOURCE.get(func.value.id, ()):
                    calls_source = True
                if func.attr == "provide" and func.value.id in ("hooks", "engine_hooks"):
                    provides = True
        if calls_source:
            need.add(path.name)
            if provides:
                ok.add(path.name)
    check(f"⑩ 调「要有真数据」口的套件（{len(need)} 份）**都**登记了源"
          f"（少写一行 ⇒ 症状是业务断言红，报错不指向接线 ✓✗）",
          need == ok, sorted(need - ok))
    check("⑩ 反验：这条判据真扫到了套件（恒空 ⇒ 假绿 ✓✗）", len(need) >= 4, sorted(need))


# ══════════════════════════════════════════════════════════════════════════
# ⑪ 接线状态**看得见**（`status()` + `doctor` 的「接线」行）
# ══════════════════════════════════════════════════════════════════════════
def _doctor_run(as_json: bool = False) -> tuple[int, str]:
    """同进程跑一次 `doctor` 把输出抓回来（不启子进程 ✓，省 1 秒且能控登记状态 ✓）。"""
    from app.services.engine import __main__ as cli

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli._cmd_doctor(argparse.Namespace(json=as_json))
    return code, buf.getvalue()


def _wiring_line(text: str) -> str:
    """把「接线」那行抠出来当报错详情 ✓✗。

    ⚠️ 别用整段输出的尾巴（`text[-260:]`）当 detail ✓✗：那永远是最后一行「边界」，
    红了也看不出接线行长什么样（2026-11-08 反验时实测踩到 ✓）。
    """
    for line in text.splitlines():
        if line.lstrip().startswith("接线"):
            return line.strip()
    return "<输出里没有「接线」行>"


def case_status_visible() -> None:
    hooks.clear()
    st = hooks.status()
    check("⑪ 未登记 ⇒ `status()` 三个源全 False、`missing` 就是那三个、`allProvided=False` "
          "（这是**进程态**，不是能力表 ✓✗）",
          st["sources"] == list(hooks.SOURCES) and not st["allProvided"]
          and set(st["missing"]) == set(hooks.SOURCES)
          and not any(st["provided"].values()) and st["registered"] == {}, st)

    code, text = _doctor_run()
    line = _wiring_line(text)
    check("⑪ `doctor` 有一行「接线」并**点名**每个未登记的源（只说「正常」＝外部看不出来 ✗）",
          code == 0 and all(name in line for name in hooks.SOURCES), line)
    check("⑪ `doctor` 明说是「没接入」而**不是**没有数据（两件事修法完全不同 ✓✗）",
          "没接入" in line, line)

    #: 假源就够：本判据只看**状态口径**，不碰 DB ✓
    hooks.provide(run_index=object())
    st2 = hooks.status()
    check("⑪ 登记一个 ⇒ 同一份 `status()` 立刻反映（不是写死的空壳 ✓✗）",
          st2["provided"]["run_index"] is True and "run_index" not in st2["missing"]
          and st2["registered"].get("run_index") == "object" and not st2["allProvided"], st2)

    #: ⚠️ 登记的都是**模块** ⇒ 名字不能一律显示「module」（`type(m).__name__` 就是那个字，
    #: 2026-11-08 实测 `doctor` 打出过 `module@run_index` —— 等于没写 ✓✗）
    fake_module = type(sys)("app.services.fake_source")
    hooks.provide(ledger=fake_module)
    check("⑪ 名字取模块名末段（不许三个源都显示 `module` ✓✗）",
          hooks.status()["registered"].get("ledger") == "fake_source",
          hooks.status()["registered"])
    hooks.provide(ledger=object())

    _code, text2 = _doctor_run()
    line2 = _wiring_line(text2)
    check("⑪ 登记后 `doctor` 改口：已登记／未登记分开写，不再一概说没接入 ✓",
          "已登记" in line2 and "run_index" in line2 and "ledger" in line2, line2)

    _code, raw = _doctor_run(as_json=True)
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = {}
    check("⑪ `doctor --json` 的 `wiring` 与 `hooks.status()` **逐字段相同**"
          "（同一真源；各写一份迟早漂 ✓✗）",
          payload.get("wiring") == hooks.status(), payload.get("wiring"))
    from app.services.engine import facade

    check("⑪ ⭐ `capabilities()` **不带** wiring（静态能力表要跟 `docs/env/*.json` 逐字段对，"
          "塞进程态会让它随进程漂 ✓✗）",
          "wiring" in payload and "wiring" not in facade.capabilities(),
          {"doctor": "wiring" in payload, "caps": "wiring" in facade.capabilities()})

    hooks.clear()
    check("⑪ 反验：清空后 `allProvided` 立刻回 False（状态实时，不是缓存 ✓）",
          hooks.status()["allProvided"] is False and hooks.get("run_index") is None)


def main() -> int:
    case_source_names()
    case_unprovided()
    case_provide()
    case_unknown_name()
    case_production_wiring()
    case_downstream_reports_unavailable()
    case_no_service_import()
    case_reader_side_typo()
    case_sources_reconcile()
    case_callers_register()
    case_status_visible()

    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      实测: {detail}")
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
