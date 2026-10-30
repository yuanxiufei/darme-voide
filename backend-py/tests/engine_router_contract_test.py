r"""自研引擎 · **对外入口契约**（E1.3 唯一入口 / E1.5 错误码单一真源 / E1.6 schema 快照）。

三件事都是「不写测试就一定会漂」的那种：

1. **唯一入口**（E1.3）：HTTP 层若绕过门面直接调 `loader.plan_stage`，CLI 与 HTTP 就会各算一遍 ——
   `schemaVersion`、错误码、显存口径只会在其中一处更新，另一处静默变旧。这里用 AST 判，不靠 review。
2. **错误码单一真源**（E1.5）：仓内任何 `E_XXX` 字符串都必须是 `facade.CODES` 的键 —— 自造一个码，
   前端就永远分流不到它（且没人会报错）。
3. **schema 快照**（E1.6）：门面六个接口的**字段名集合**进 `docs/engine-api.schema.json`。增删改名
   都会红，逼着先升 `SCHEMA_VERSION` —— 「字段悄悄少了」是最难发现的一类破坏。

运行::

    ./.venv/Scripts/python.exe tests/engine_router_contract_test.py
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from app.services.engine import facade  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_ROUTERS = _ROOT / "app" / "routers"
_ENGINE = _ROOT / "app" / "services" / "engine"
_SCHEMA_PATH = _ROOT.parent / "docs" / "engine-api.schema.json"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, ok: bool, detail: object = None) -> None:
    _RESULTS.append((name, bool(ok), detail))


def _calls_in(node: ast.AST) -> set[str]:
    """`node` 内所有两段式调用名（`facade.plan(...)` → ``"facade.plan"``）。"""
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            base = sub.func.value
            if isinstance(base, ast.Name):
                names.add(f"{base.id}.{sub.func.attr}")
    return names


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


# ══════════════════════════════════════════════════════════════════════════
# ① + ② 唯一入口：装载计划与运行时状态必须走门面
# ══════════════════════════════════════════════════════════════════════════
def case_single_entry() -> None:
    source = (_ROUTERS / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = _functions(tree)

    #: ⚠️ 只禁 `loader.plan_stage` ✗：`/readiness` 走 `inv.readiness` 是**另一件事** ✓ ——
    #: 那是项目侧编排（pip 建议 / 超清放大器 / 分镜有无 ✓），引擎门面明写「不含项目侧」✓。
    #: 一刀切禁掉，等于把项目侧职责塞回引擎 ✓✗。
    direct_plan = sorted(
        f"{sub.func.value.id}.{sub.func.attr}"
        for sub in ast.walk(tree)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "plan_stage" and isinstance(sub.func.value, ast.Name)
        and sub.func.value.id == "loader")
    check("① `/load-plan` 不直接调 `loader.plan_stage`（必须走 `facade.plan`）",
          not direct_plan, direct_plan)

    load_plan = funcs.get("load_plan")
    check("② `/load-plan` 函数体里确实调了 `facade.plan`",
          load_plan is not None and "facade.plan" in _calls_in(load_plan),
          sorted(_calls_in(load_plan)) if load_plan else "load_plan 不存在")

    runtime_status = funcs.get("runtime_status")
    check("③ `/runtime` 函数体里确实调了 `facade.status`",
          runtime_status is not None and "facade.status" in _calls_in(runtime_status),
          sorted(_calls_in(runtime_status)) if runtime_status else "runtime_status 不存在")

    #: 门面被 router 真的 import 了（否则上两条只是「写了个名字」）
    imported = any(isinstance(node, ast.ImportFrom) and node.level == 2
                   and node.module == "services.engine"
                   and any(alias.name == "facade" for alias in node.names)
                   for node in tree.body)
    check("④ router 从引擎包 import 了 `facade`", imported, "没 import")


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 错误码单一真源
# ══════════════════════════════════════════════════════════════════════════
def _engine_code_literals(path: Path) -> set[str]:
    """文件里所有**看起来像引擎错误码**的字符串常量（`"E_XXX"` 整串，不含 docstring 长句）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if text.startswith("E_") and text[2:].replace("_", "").isalnum() \
                    and text.upper() == text and len(text) < 40:
                found.add(text)
    return found


def case_error_codes() -> None:
    known = set(facade.CODES)
    offenders: dict[str, list[str]] = {}
    for folder in (_ROUTERS, _ENGINE):
        for path in sorted(folder.glob("*.py")):
            stray = sorted(_engine_code_literals(path) - known)
            if stray:
                offenders[path.name] = stray
    check("⑤ 仓内 `E_*` 字面量全部来自 `facade.CODES`（无自造码）", not offenders, offenders)

    #: 经 `_engine_error` 出去的码必须就是门面那个码（不是 `str(err)` 再被前端猜）
    from app.routers import engine as router_mod  # noqa: PLC0415 —— 只在需要时导入 FastAPI 侧

    err = facade.EngineError("权重不在盘上", code="E_WEIGHT_MISSING", hint="按清单下载")
    body = json.loads(router_mod._engine_error(err).body.decode("utf-8"))
    check("⑥ `_engine_error` 把门面码原样带出（`engineCode`）",
          body.get("engineCode") == "E_WEIGHT_MISSING" and body.get("code") == 400, body)

    weird = facade.EngineError("瞎写的码", code="E_NOT_A_REAL_CODE")
    body2 = json.loads(router_mod._engine_error(weird).body.decode("utf-8"))
    check("⑦ 未知码被 `EngineError` 归一成 `E_ENGINE_UNAVAILABLE`（不会漏到前端）",
          body2.get("engineCode") == "E_ENGINE_UNAVAILABLE", body2)


# ══════════════════════════════════════════════════════════════════════════
# ⑧⑨⑩ schema 快照：字段名集合就是契约
# ══════════════════════════════════════════════════════════════════════════
def _write_safetensors(path: Path) -> None:
    """最小合法 safetensors（两个 bf16 张量）—— 只为拿 `inspect` / `manifest` 的字段集合。"""
    import struct  # noqa: PLC0415

    header = {
        "a": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]},
        "b": {"dtype": "BF16", "shape": [1], "data_offsets": [8, 10]},
    }
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 10)


def _compare(schema: dict[str, Any], interface: str, actual: Any,
             label: str) -> tuple[bool, object]:
    want = schema["interfaces"][interface]
    got = set(actual or {})
    expected = set(want["top"])
    if got != expected:
        return False, {"missing": sorted(expected - got), "unexpected": sorted(got - expected)}
    return True, label


def case_schema() -> None:
    raw = _SCHEMA_PATH.read_text(encoding="utf-8")
    schema = json.loads(raw)

    check("⑧ schema 文件里的版本与代码一致（升版本必须两边同步）",
          schema.get("apiVersion") == facade.API_VERSION
          and int(schema.get("schemaVersion")) == int(facade.SCHEMA_VERSION),
          {"file": [schema.get("apiVersion"), schema.get("schemaVersion")],
           "code": [facade.API_VERSION, facade.SCHEMA_VERSION]})

    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "sample.safetensors"
        _write_safetensors(sample)
        actuals: dict[str, Any] = {
            "capabilities": facade.capabilities(),
            "plan": facade.plan("h3"),
            "readiness": facade.readiness("h3"),
            "status": facade.status(),
            "inspect_weights": facade.inspect_weights(sample),
            "manifest_of": facade.manifest_of(sample),
        }
        wrong: dict[str, Any] = {}
        for interface, data in actuals.items():
            ok, detail = _compare(schema, interface, data, "")
            if not ok:
                wrong[interface] = detail
        check("⑨ 六个接口的**顶层字段集合**与 schema 快照逐一相等", not wrong, wrong)

        nested: dict[str, Any] = {}
        for interface, spec in schema["interfaces"].items():
            for key, expected in (spec.get("nested") or {}).items():
                data = actuals[interface]
                if key.endswith("[]"):
                    items = data.get(key[:-2]) or []
                    if not items:
                        nested[f"{interface}.{key}"] = "样本里没有元素（无法断言）"
                        continue
                    got = set(items[0])
                else:
                    got = set((data.get(key) or {}))
                if got != set(expected):
                    nested[f"{interface}.{key}"] = {
                        "missing": sorted(set(expected) - got),
                        "unexpected": sorted(got - set(expected))}
        check("⑩ 关键嵌套字段集合与 schema 快照相等", not nested, nested)


def case_assembly_keys() -> None:
    """⭐ 装配白名单 vs ``ensure_loaded`` 签名：**手抄副本必须盖住真源** ✗✗。

    ⚠️ 为什么单独一条 ✗：``_ASSEMBLY_KEYS`` 是**逐项手抄**的 ✓，而 ``ensure_loaded`` 的参数才是
    真源 ✓ —— 少抄一项就「开关静默失效」✓✗：``2016-10-29`` 实测 ``int8_offload`` 就是这样丢的
    （``bridge.assembly_from_config`` 认它 ✓、HTTP 白名单不认 ✓✗ ⇒ 同一开关两条路行为不同 ✓✗，
    而 720×1280 只有这一档装得下 ⇒ 用户以为开了、结果照样 OOM 且报告里也没有线索 ✓✗）。
    这条判据把「下次再加一个开关、只加一处」变成红 ✓。
    """
    import inspect  # noqa: PLC0415

    from app.routers import engine as engine_router  # noqa: PLC0415
    from app.services.engine import pipeline as pipe_mod  # noqa: PLC0415
    from app.services.engine import runtime as runtime_mod  # noqa: PLC0415

    #: ⚠️ **豁免现在是空的** ✓（2026-10-29 ✓）：`loras` 已收进白名单 ✓ —— 它不再靠"拒收"来
    #: 避免静默丢掉 ✓，而是收下来走 `bridge.lora_specs_from_settings` 那层归一 ✓（判据 ⑪⁵ ✓）。
    #: ⚠️ 别再给它加豁免 ✗✗：一有豁免，`ensure_loaded` 下次多一个参数就**报不出来** ✓✗。
    exempt: set[str] = set()
    params = {name for name in inspect.signature(runtime_mod.EngineRuntime.ensure_loaded).parameters
              if name != "self"}
    exposed = set(engine_router._ASSEMBLY_KEYS.values())  # noqa: SLF001
    missing = sorted(params - exposed - exempt)
    check("⑪′ ⭐ `ensure_loaded` 的每个参数都进了 HTTP 装配白名单（**零豁免** ✓）"
          "—— 少抄一项就是「开关静默失效」✗✗", not missing, missing)

    on = engine_router._assembly_from({"int8Offload": True})  # noqa: SLF001
    off_str = engine_router._assembly_from({"int8Offload": "false"})  # noqa: SLF001
    absent = engine_router._assembly_from({})  # noqa: SLF001
    check("⑪″ ⭐ ``int8Offload`` 真的透到 `ensure_loaded` ✓；``\"false\"`` 必须读成 **False** ✓"
          "（`bool(\"false\")` 是 True ✓✗ —— 那会让人「以为没开、其实每步多一趟 PCIe」✓✗）；"
          "不给 ⇒ **键不出现** ✓（走签名默认 False ✓，不是塞一个 False 进去 ✓）",
          on.get("int8_offload") is True
          and off_str.get("int8_offload") is False
          and "int8_offload" not in absent,
          {"on": on, "off": off_str, "absent": absent})

    bad: Exception | None = None
    try:
        engine_router._assembly_from({"int8Offload": "也许"})  # noqa: SLF001
    except pipe_mod.StageError as err:
        bad = err
    except Exception as err:  # noqa: BLE001
        bad = err
    check("⑪‴ 开关给了**不认识的值** ⇒ 当场 400（`StageError`）✓，不许猜成 True/False ✗"
          "（同 `audioLatentMode` 那条口径 ✓）",
          isinstance(bad, pipe_mod.StageError), repr(bad)[:120])

    flag_bad: Exception | None = None
    try:
        engine_router._flag({"force": "maybe"}, "force")  # noqa: SLF001
    except pipe_mod.StageError as err:
        flag_bad = err
    check("⑪⁗ ⚠️ `unload` 的 `force` 走同一条严格解析 ✓ —— `\"false\"` 被读成 True 会**强卸**、"
          "把在跑的任务搞死 ✗✗（用户本意正好相反 ✓✗）",
          isinstance(flag_bad, pipe_mod.StageError)
          and engine_router._flag({"force": "false"}, "force") is False  # noqa: SLF001
          and engine_router._flag({}, "force") is False,  # noqa: SLF001
          repr(flag_bad)[:120])

    #: ⚠️ `loras` 现在**收** ✓（2026-10-29 ✓）—— 但收下来**必须走 config 那条路的同一份归一** ✓：
    #:    相对路径按 `modelsDir` 解析 ✓、strength 归一 ✓、`lora` 单条也认 ✓。
    #:    ⚠️ 断言"能装上"不够 ✗✗：真正要盯的是**归一真的发生了** ✓（路径被拼成绝对路径 ✓）——
    #:    只判"没报错"的话，把 `loras` 原样透传也会绿 ✓✗（那就是相对路径按 cwd 解析的坑 ✓✗）。
    with tempfile.TemporaryDirectory() as tmp:
        real = Path(tmp) / "turbo.safetensors"
        real.write_bytes(b"\x00" * 8)
        abs_ok = engine_router._assembly_from({"loras": [str(real)]})  # noqa: SLF001
        rel_ok = engine_router._assembly_from(  # noqa: SLF001
            {"loras": [{"path": "turbo.safetensors", "strength": 0.65}], "modelsDir": tmp})
        single_ok = engine_router._assembly_from({"lora": str(real)})  # noqa: SLF001
        #: ⚠️ 显式 `strength: 0` 必须保持 0.0 ✗✗ —— 以前 torch 后端那句
        #:    `float(... or 1.0)` 会把它抬成 1.0 ✓✗，而装载键按 0.0 记 ✓ ⇒ 键与实挂不一致 ✓✗。
        zero_ok = engine_router._assembly_from(  # noqa: SLF001
            {"loras": [{"path": str(real), "strength": 0}]})
        empty_ok = engine_router._assembly_from({"loras": []})  # noqa: SLF001

        def _first(assembly: dict[str, Any]) -> Any:
            items = assembly.get("loras") or []
            return items[0] if items else None

        def _json_safe(*asmts: dict[str, Any]) -> bool:
            #: ⚠️ 装配**会回显** ✓（`Task.assembly` ⇒ `/tasks` ✓），而 `success()` 走
            #:    `JSONResponse`（**裸 `json.dumps`** ✓，不过 `jsonable_encoder` ✗）⇒
            #:    里面塞一个 `LoraSpec` 就是 **500** ✓✗。这是 2026-10-29 查出的**既有隐患** ✓
            #:    （config 那条路一直塞的是 `LoraSpec` ✓✗）⇒ 所以这里不能只判"归一对了" ✗。
            try:
                for assembly in asmts:
                    json.dumps(assembly)
            except TypeError:
                return False
            return True

        abs_one, rel_one = _first(abs_ok), _first(rel_ok)
        single_one, zero_one = _first(single_ok), _first(zero_ok)
        check("⑪⁵ ⭐ `loras` 走 HTTP 也**真归一** ✓ —— 绝对路径原样 ✓、相对路径按 `modelsDir` "
              "拼成绝对 ✓（只判「没报错」不够 ✗：原样透传也会绿 ✓✗）、`strength` 归一 ✓、"
              "`lora` 单条默认 1.0 ✓、**显式 `strength: 0` 保持 0.0** ✓（抬成 1.0 就是「键说 0.0、"
              "实际按 1.0 挂」✗✗）、给空清单 ⇒ 空清单 ✓ 不打红正常请求 ✗；"
              "⚠️ 且装配必须**能 JSON 出去** ✓（回显时 `LoraSpec` 会 500 ✗✗）",
              abs_one is not None and Path(abs_one["path"]).resolve() == real.resolve()
              and rel_one is not None and Path(rel_one["path"]).resolve() == real.resolve()
              and abs(rel_one["strength"] - 0.65) < 1e-9
              and single_one is not None and abs(single_one["strength"] - 1.0) < 1e-9
              and zero_one is not None and zero_one["strength"] == 0.0
              and (empty_ok.get("loras") or []) == []
              and _json_safe(abs_ok, rel_ok, single_ok, zero_ok, empty_ok),
              {"abs": repr(abs_one)[:80], "rel": repr(rel_one)[:80],
               "relRaw": str((rel_ok.get("loras") or [None])[0]),
               "single": repr(single_one)[:80], "zero": repr(zero_one)[:80],
               "empty": empty_ok.get("loras")})

        #: ⚠️ 坏输入**当场 400** ✓（`StageError` ⇒ `bad_request`）—— 漏出去就是 **500** ✓✗。
        #:    ⚠️ 目录那条盯的是 `is_file()` ≠ `exists()` ✓：只 `exists()` 会把传错的目录放行到装载深处 ✓✗。
        loras_bad: Exception | None = None
        try:
            engine_router._assembly_from({"loras": ["/nonexistent/x.safetensors"]})  # noqa: SLF001
        except pipe_mod.StageError as err:
            loras_bad = err
        dir_bad: Exception | None = None
        try:
            engine_router._assembly_from({"loras": [tmp]})  # noqa: SLF001
        except pipe_mod.StageError as err:
            dir_bad = err
        check("⑪⁶ ⚠️ `loras` 写错 ⇒ **当场 400 并点名那个文件** ✓（不是 500 ✗✗，也**不是静默丢掉** ✗✗"
              "—— 丢掉照样「跑得动、出得来、不是那个档」✓✗）；传进来一个**目录**也必须拒 ✗",
              isinstance(loras_bad, pipe_mod.StageError)
              and "不存在" in str(loras_bad)
              and "x.safetensors" in str(loras_bad)
              and isinstance(dir_bad, pipe_mod.StageError),
              (repr(loras_bad)[:160], repr(dir_bad)[:160]))


# ══════════════════════════════════════════════════════════════════════════
# ⑫ 取消裁决的**单点真源**（E3.2 ✓，2026-10-29 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_cancel_verdict() -> None:
    """`cancel.verdict` 只能由 `runtime.cancel` 定 ✓ —— 路由层**原样透传** ✗✗。

    路由最容易犯的错就是"顺手按 `status` 再判一次" ✓✗：那读的是**另一瞬间**的事实 ✓，
    实测把裁决改成事后按 `status` 判 ⇒ 二次取消会**谎称"又停了一次"**（`engine_runtime_test`
    恰红那一条 ✓）。所以这里判的是形状 ✗ 不是文案 ✓：转发调用在、自己不碰 `status`、
    也不自己拼 `cancel` 块 ✓（拼一份就会与 runtime 那份漂 ✓✗）。
    """
    source = (_ROUTERS / "engine.py").read_text(encoding="utf-8")
    node = _functions(ast.parse(source)).get("cancel_engine_task")
    if node is None:
        check("取消: 路由函数还在（`cancel_engine_task` ✓）", False, "函数不见了")
        return

    #: ⚠️ 只看**代码** ✓ 不看 docstring ✗ —— 说明里**必须**提 `status`（讲清为什么不许用它 ✓），
    #:   连说明一起扫就成了"写清楚即违规" ✓✗。
    code_nodes = [sub for sub in node.body
                  if not (isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Constant)
                          and isinstance(sub.value.value, str))]
    code = "\n".join(ast.get_source_segment(source, sub) or "" for sub in code_nodes)

    check("取消: 裁决**透传** runtime（`engine_runtime.cancel(...)` ✓）",
          any(isinstance(sub, ast.Call) and f"{getattr(sub.func.value, 'id', '')}.{sub.func.attr}"
              == "engine_runtime.cancel"
              for sub in ast.walk(node) if isinstance(sub, ast.Call)
              and isinstance(sub.func, ast.Attribute)), code)
    check("取消: 路由**不许**自己按 `status` 重判（事后读的是另一瞬间 ⇒ 二次取消会谎称又停一次 ✓✗）",
          "status" not in code, code)
    check("取消: 路由**不许**自己拼 `cancel` 裁决块（单点真源 ✓；拼一份就会与 runtime 那份漂 ✗✗）",
          '"cancel"' not in code and "'cancel'" not in code, code)


def main() -> int:
    case_single_entry()
    case_error_codes()
    case_schema()
    case_assembly_keys()
    case_cancel_verdict()

    failed = 0
    for name, ok, detail in _RESULTS:
        mark = "OK  " if ok else "FAIL"
        print(f"{mark} {name}")
        if not ok:
            failed += 1
            print(f"     ↳ {detail}")
    print(f"\nSUMMARY: {len(_RESULTS) - failed}/{len(_RESULTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
