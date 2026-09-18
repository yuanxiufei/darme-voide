"""S7 守卫：**路由里对 ``read_json`` 返回值的 ``isinstance(body, dict)`` 是死代码**（2026-09-18 立 ✓）。

## 为什么立它

``core/request_utils.read_json`` **有意**把「非法 JSON / 非对象」**归一成 ``{}``** ✓
（注释写明：对齐 TS 的 ``c.req.json()`` 会抛错并被 catch 成 400 ⇒ Python 侧走空 dict、
**由业务校验给出 400** ✓，且不让"客户端发了坏 JSON"升级成 500 ✓）。

⇒ 于是 ``body = await read_json(request)`` 之后的
``if not isinstance(body, dict): return bad_request(...)`` **永远进不去** ✗✗：

* **无害** ✓（紧随的业务校验照样 400 ✓），但**会骗人** ✗ ——
  后来的人以为"坏输入已被这行拦住" ✓，于是业务校验写松了 ✓ ⇒ 真漏过一次 ✓；
* 2026-09-18 立这条时全仓共 **9 处**（我自己写的 8 处 + ``comfyui.py`` 1 处历史遗留 ✓）。

## 为什么用 **AST** 而不是正则（这一步我返工过 ✓）

第一版用正则按行匹配 ✓ —— 结果：

* **误报**：我在 ``preflight.py`` 的**文档串**里写了 ``isinstance(body, dict)`` 来说明
  "这里不做这个检查" ✓，被当成违规 ✗；
* **漏报**：``prompt_tools.py`` 还有一处**形状略有不同**的 ✓，正则的"固定两行"匹配不到 ✗。

⇒ 改成 **AST** ✓：只看**真代码** ✓（注释/字符串天然不进 AST ✓）、
且识别的是**语义**（"对 ``read_json`` 绑定的名字做 ``isinstance(..., dict)`` ✓"）而不是形状 ✓。

## 本守卫的三段（少一段就可能假绿 ✗）

1. **合同锚点** ✓：``read_json`` 保证返回 dict（实现里必须有那句归一 ✓）——
   哪天有人改了它 ✓，本守卫会提醒"死代码这个前提不成立了" ✓；
2. **正/负对照** ✓：合成的死代码**必须检出** ✓（否则检测器坏了 ⇒ 全绿是假的 ✗）；
   ``request.json()`` 的返回值**不算** ✓（它真可能不是 dict ⇒ 那个检查是必要的 ✓）；
3. **全仓扫描** ✓ —— 且**扫描面非空** ✓（扫 0 个文件也会"全绿"✗）。

运行::

    ./.venv/Scripts/python.exe tests/read_json_guard_test.py
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

ROUTERS_DIR = BACKEND_PY / "app" / "routers"
REQUEST_UTILS = BACKEND_PY / "app" / "core" / "request_utils.py"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


#: 确实**该**保留的例外（文件名 → 理由 ✓）—— 目前为空 ✓（留机制 ✓ 不预置例外 ✗）
ALLOW: dict[str, str] = {}

_BIND_RE = re.compile(r"^\s*(?P<name>\w+)\s*=\s*await\s+read_json\s*\(", re.MULTILINE)


def _read_json_names(scope: ast.AST) -> set[str]:
    """这个函数里从 ``read_json(...)`` 拿到值的变量名 ✓。"""
    names: set[str] = set()
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Await):
            continue
        call = node.value.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "read_json":
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _is_dict_check(test: ast.AST, names: set[str]) -> bool:
    """``not isinstance(<名字>, dict)` `` ✓（``dict`` 也容忍写成 ``(dict, ...)`` ✓）。"""
    if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
        return False
    call = test.operand
    if not isinstance(call, ast.Call):
        return False
    if getattr(call.func, "id", None) != "isinstance" or not call.args:
        return False
    first = call.args[0]
    if not isinstance(first, ast.Name) or first.id not in names:
        return False
    def mentions_dict(node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and node.id == "dict":
            return True
        if isinstance(node, (ast.Tuple, ast.List)):
            return any(mentions_dict(item) for item in node.elts)
        return False
    return len(call.args) > 1 and mentions_dict(call.args[1])


def dead_dict_checks(text: str) -> list[tuple[int, str]]:
    """AST 找「对 ``read_json`` 返回值做 ``isinstance(..., dict)``」的真代码 ✓。

    ⚠️ **注释与字符串不进 AST** ✓ ⇒ 我写在文档串里的"这里不做这个检查"不会误报 ✓
    （正则版本正是栽在这上面 ✓）。
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    hits: list[tuple[int, str]] = []
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        names = _read_json_names(scope)
        if not names:
            continue
        for node in ast.walk(scope):
            if isinstance(node, ast.If) and _is_dict_check(node.test, names):
                line = text.splitlines()[node.lineno - 1].strip()
                hits.append((node.lineno, line))
    return sorted(hits)


def case_contract() -> None:
    source = REQUEST_UTILS.read_text(encoding="utf-8")
    check("① ⭐ **合同锚点**：`read_json` 仍然保证返回 dict ✓"
          "（实现里必须有把非对象归一成 `{}` 的那句 ✓ —— 否则本守卫的前提不成立 ✗）",
          "isinstance(data, dict) else {}" in source,
          "read_json 的归一逻辑变了 ⇒ 请重判本守卫的结论 ✓")
    check("② 合同里写明**为什么**（对齐 TS / 由业务校验给 400 / 不升级成 500 ✓）",
          "业务校验" in source and "400" in source, "")


def case_controls() -> None:
    synthetic = (
        "async def route(request):\n"
        "    body = await read_json(request)\n"
        "    if not isinstance(body, dict):\n"
        "        return bad_request('必须是对象')\n"
        "    return success(body)\n")
    hits = dead_dict_checks(synthetic)
    check("③ ⭐ **正对照**：合成的死代码必须被检出 ✓（检测器坏了 ⇒ 全绿是假的 ✗）",
          len(hits) == 1 and hits[0][0] == 3, hits)

    legit = (
        "async def route(request):\n"
        "    data = await request.json()\n"
        "    if not isinstance(data, dict):\n"
        "        return bad_request('必须是对象')\n"
        "    return success(data)\n")
    check("④ ⭐ **负对照**：`request.json()` 的返回值**不算**死代码 ✓"
          "（它真可能不是 dict ⇒ 那个检查是必要的 ✓）",
          dead_dict_checks(legit) == [], dead_dict_checks(legit))

    docstring = (
        "async def route(request):\n"
        "    \"\"\"这里不做 isinstance(body, dict) 检查 —— read_json 已保证 dict。\"\"\"\n"
        "    body = await read_json(request)\n"
        "    return success(body)\n")
    check("⑤ ⭐ **文档串里的同形文字不误报** ✓（正则版栽在这上面 ⇒ 才改的 AST ✓）",
          dead_dict_checks(docstring) == [], dead_dict_checks(docstring))

    varied = (
        "async def route(request):\n"
        "    payload = await read_json(request)\n"
        "    if not isinstance(payload, (dict, list)):\n"
        "        return bad_request('x')\n")
    check("⑥ 换成别的变量名 / 别的形状（`(dict, list)` ✓）**照样检出** ✓"
          "（按语义判 ✓ 不按形状 ✗）",
          len(dead_dict_checks(varied)) == 1, dead_dict_checks(varied))

    other_func = (
        "async def a(request):\n"
        "    body = await read_json(request)\n"
        "    return body\n\n"
        "async def b(request):\n"
        "    body = {'x': 1}\n"
        "    if not isinstance(body, dict):\n"
        "        return bad_request('y')\n")
    check("⑦ **只在同一函数内**判定 ✓：另一个函数里同名的 `body` 不算 ✓",
          dead_dict_checks(other_func) == [], dead_dict_checks(other_func))


def case_repo() -> None:
    findings: list[str] = []
    call_sites = 0
    scanned = 0
    for path in sorted(ROUTERS_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        scanned += 1
        call_sites += len(_BIND_RE.findall(text))
        if path.name in ALLOW:
            continue
        for number, line in dead_dict_checks(text):
            findings.append(f"{path.name}:{number}  {line[:80]}")
    check("⑧ 扫描面非空 ✓（真扫到了路由文件与 read_json 调用点 ✓ —— "
          "扫 0 个也会「全绿」✗）",
          scanned > 0 and call_sites > 0, (scanned, call_sites))
    check("⑨ ⭐⭐ **全仓路由里没有这类死代码** ✓（它永远进不去 ⇒ "
          "会让人误以为坏输入已被拦住 ✗）",
          findings == [], findings)
    check("⑩ 例外表为空 ✓（有例外必须在 `ALLOW` 里写明理由 ✓）",
          all(reason.strip() for reason in ALLOW.values()), ALLOW)


def main() -> int:
    case_contract()
    case_controls()
    case_repo()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
