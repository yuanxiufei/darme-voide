"""自检：**stage 注册表 ＋ 装载前兼容性闸门**（E2.1 / E2.2 ✓）。

钉的是四件用户真会遇到的事：

1. **阶段集合相等**（双向）✓：清单有而注册表没有 ⇒ 那个阶段完全没有兼容性判据；
   注册表有而清单没有 ⇒ 那个阶段匹配到 0 个组件、``ready=True`` 是**空就绪**。两个方向都会
   让"体检通过"失去意义，所以两条都要报。
2. **未注册阶段 ⇒ ``E_BAD_CONFIG``**（不是静默按 h3 装、也不是"引擎不可用"）。
3. **四类各一条故障注入**（依赖 / 权重 / 量化族 / 显存）—— 都在**装载前**给可读错误 ＋
   机器可判的码，且经门面 ``translate`` 后码**不变**（翻成 ``E_ENGINE_UNAVAILABLE`` 就是把人
   送去查错方向）。
4. **「没查」≠「通过」** ✓✗：注入 ``{}`` 表示这一类没查 ⇒ 必须出现在 ``unchecked`` 里，
   而不是悄悄算作已通过。

⚠️ 全程不碰 torch ✓（注册表不 import torch：依赖探测走 ``find_spec``、版本走包元数据 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_registry_test.py
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import facade, inventory, registry  # noqa: E402

GIB = 1024 ** 3
_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _required_key() -> str:
    """从清单里取**一个真的必需件**（不硬编码 key ✗ —— 模型换名本测试不该跟着红 ✓）。"""
    for item in inventory.load_catalog()["models"]:
        if item.get("required") and str(item.get("category")) == "video":
            return str(item["key"])
    raise AssertionError("清单里没有任何必需的视频件（这份 configs/models.json 不对劲）")


def _blank() -> dict[str, Any]:
    """四类全部"没查" ✓（用来验 unchecked 口径 ✓）。"""
    return {"deps": {}, "readiness": {}, "plans": {}}


def _report(**overrides: Any) -> dict[str, Any]:
    kwargs = {"deps": {"missing": [], "install": [], "optionalMissing": []},
              "readiness": {}, "plans": {}}
    kwargs.update(overrides)
    return registry.check_compat("h3", **kwargs)


def _codes(report: dict[str, Any], kind: str) -> list[str]:
    return [str(item["code"]) for item in report["blockers"] if item["kind"] == kind]


# ══════════════════════════════════════════════════════════════════════════
def case_zero_deps() -> None:
    check("零依赖: import `registry` **不拖 torch** ✓（无卡机器也要能回答「装得了吗」✓）",
          "torch" not in sys.modules,
          [name for name in ("torch", "numpy") if name in sys.modules])


def case_registration() -> None:
    problems = registry.registration_problems()
    filter_stages = sorted(str(name) for name in inventory.STAGE_FILTERS)
    check("注册表: 阶段集合与 `inventory.STAGE_FILTERS` **双向相等** ✓（少一个方向都会让体检失真 ✓✗）",
          sorted(registry.STAGES) == filter_stages and not problems,
          {"registered": sorted(registry.STAGES), "filters": filter_stages, "problems": problems})
    bad_category = [name for name in registry.STAGES
                    if str((inventory.STAGE_FILTERS.get(name) or {}).get("category") or "")
                    != registry.STAGES[name].category]
    check("注册表: 同名阶段的 `category` 必须与清单规则一致 ✓（不一致 ⇒ 体检的组件集 ≠ 装载的组件集 ✓✗）",
          not bad_category, bad_category)
    claimed = {spec.category for spec in registry.STAGES.values()}
    catalog_categories = {str(item.get("category") or "") for item in
                          inventory.load_catalog()["models"]} - {""}
    unclaimed = registry.unclaimed_categories()
    check("注册表: `unclaimed_categories()` == 清单分类 − 已认领分类 ✓（缺口要看得见，不是要有 0 条 ✓）",
          set(unclaimed) == catalog_categories - claimed,
          {"unclaimed": unclaimed, "catalog": sorted(catalog_categories), "claimed": sorted(claimed)})
    check("注册表: 每个阶段都声明了能力与依赖 ✓（空声明＝没声明 ✓✗）",
          all(spec.capabilities and spec.deps for spec in registry.STAGES.values()),
          {name: {"capabilities": spec.capabilities, "deps": spec.deps}
           for name, spec in registry.STAGES.items()})
    described = registry.describe()
    check("注册表: `describe()` 自洽（阶段数 / 问题 / 四类 / hint 都在 ✓）",
          set(described) == {"stages", "registered", "filterStages", "problems",
                             "unclaimedCategories", "compatKinds", "vramHint"}
          and set(described["stages"]) == set(registry.STAGES)
          and described["vramHint"] == registry.VRAM_HINT
          and described["compatKinds"] == list(registry.COMPAT_KINDS),
          sorted(described))


def case_unknown_stage() -> None:
    try:
        registry.stage_spec("video")
    except registry.RegistryError as err:
        check("未注册阶段: 抛 `RegistryError` 且 `code == E_BAD_CONFIG` ✓（不回落成 h3 ✓✗；"
              "可用阶段在 `hint` 里点名 ✓）",
              err.code == "E_BAD_CONFIG" and "h3" in err.hint and err.detail.get("registered"),
              err.to_dict())
        translated = facade.translate(err)
        check("未注册阶段: 经门面 `translate` 仍是 `E_BAD_CONFIG` ✓（不许变成「引擎不可用」✓✗）",
              translated.code == "E_BAD_CONFIG" and translated.message == err.message,
              translated.to_dict())
    else:
        check("未注册阶段: 抛 `RegistryError` ✓", False, "没抛错")

    try:
        registry.stage_spec("  ")
    except registry.RegistryError as err:
        check("空阶段名: 抛具名错误 ✓（空串回落默认阶段是最糟的静默 ✓✗）",
              err.code == "E_BAD_CONFIG" and "空" in err.message, err.message)
    else:
        check("空阶段名: 抛具名错误 ✓", False, "没抛错")


def case_deps() -> None:
    spec = registry.STAGES["h3"]
    first_dep = spec.deps[0]
    missing = _report(deps={"missing": [first_dep], "install": [f"pip install {first_dep}"],
                            "optionalMissing": []})
    check(f"依赖类: 缺必需依赖 `{first_dep}` ⇒ `E_ENGINE_UNAVAILABLE` ＋ 给出安装命令 ✓",
          _codes(missing, "deps") == ["E_ENGINE_UNAVAILABLE"]
          and f"pip install {first_dep}" in missing["blockers"][0]["hint"],
          missing["blockers"])
    optional = _report(deps={"missing": ["onnxruntime"], "install": [], "optionalMissing": ["onnxruntime"]})
    check("依赖类: 缺**可选**依赖不阻断 ✓（但要在 facts 里看得见 ✓✗）",
          not optional["blockers"]
          and optional["checks"]["deps"]["facts"]["optionalMissing"] == ["onnxruntime"],
          optional["checks"]["deps"]["facts"])

    real = registry.version_of
    try:
        registry.version_of = lambda package: "0.0.1"          # noqa: ARG005
        low = _report()
        check("版本下限: 装的是 `0.0.1` ⇒ 阻断并点名「要求 ≥ …」✓（读得到版本就必须判 ✓✗）",
              _codes(low, "deps") == ["E_ENGINE_UNAVAILABLE"]
              and "版本过低" in low["blockers"][0]["message"], low["blockers"])
        registry.version_of = lambda package: None            # noqa: ARG005
        unknown = _report()
        check("版本下限: 版本**读不出来** ⇒ 进 `unchecked`、**不阻断**也不当通过 ✓✗",
              not [item for item in unknown["blockers"] if item["kind"] == "deps"]
              and any("版本" in note for note in unknown["unchecked"]), unknown["unchecked"])
        registry.version_of = lambda package: "不是版本"      # noqa: ARG005
        weird = _report()
        check("版本下限: 版本字符串判不出数字 ⇒ 进 `unchecked`（不当作通过 ✓✗）",
              any("判不出数字" in note for note in weird["unchecked"]), weird["unchecked"])
    finally:
        registry.version_of = real


def case_weights() -> None:
    key = _required_key()
    missing = _report(readiness={"missingRequired": [key], "brokenRequired": []})
    check(f"权重类: 必需件 `{key}` 不在盘上 ⇒ `E_WEIGHT_MISSING` ✓",
          _codes(missing, "weights") == ["E_WEIGHT_MISSING"], missing["blockers"])
    broken = _report(readiness={"missingRequired": [], "brokenRequired": [key]})
    check(f"权重类: 必需件 `{key}` 读不了 ⇒ `E_WEIGHT_CORRUPT` ✓（与「缺件」分开：一个该下载、一个该重下 ✓）",
          _codes(broken, "weights") == ["E_WEIGHT_CORRUPT"], broken["blockers"])
    corrupt_catalog = _report(readiness={"catalogError": "models.json: bad json"})
    check("权重类: 清单读不了 ⇒ `E_BAD_CONFIG` ✓（不知道怎么装，不是装不了 ✓）",
          _codes(corrupt_catalog, "weights") == ["E_BAD_CONFIG"], corrupt_catalog["blockers"])
    suspect = _report(readiness={"missingRequired": [], "brokenRequired": [],
                                 "suspectRequired": [key], "ready": True})
    check("权重类: 字节数可疑 ⇒ 只警告不阻断 ✓（能读就放行，但要说出来 ✓）",
          not suspect["blockers"] and any("可疑" in item for item in suspect["warnings"]),
          suspect["warnings"])

    original = registry._readiness

    def _boom(stage: str, root: object, capacity: object) -> dict[str, Any]:
        raise RuntimeError("假体检崩溃 ✓")

    try:
        registry._readiness = _boom                       # type: ignore[assignment]
        crashed = registry.check_compat("h3", deps={}, plans={})
        check("权重类: 体检**自己崩了** ⇒ 出阻断、不许静默放行 ✓✗（码报「引擎不可用」而不是「缺件」✓）",
              _codes(crashed, "weights") == ["E_ENGINE_UNAVAILABLE"],
              crashed["blockers"])
    finally:
        registry._readiness = original                # type: ignore[assignment]


def case_quant() -> None:
    key = _required_key()
    plans = {"components": [
        {"key": key, "dequantPlan": {"weights": 12, "supported": False,
                                     "unresolvedCount": 12,
                                     "family": {"name": "mxfp4", "reason": "本仓没实现"}}},
    ]}
    unsupported = _report(plans=plans)
    #: ⚠️ 样本族从 `nvfp4` 换成 `mxfp4`（2026-10-30）：nvfp4 **已实现** ✓，再拿它当「没实现」的例子
    #: 会让读测试的人以为它还是缺口 ✓✗。
    check(f"量化类: 必需件 `{key}` 的族本仓没实现 ⇒ `E_UNSUPPORTED_QUANT` ✓（连同族名一起报 ✓）",
          _codes(unsupported, "quant") == ["E_UNSUPPORTED_QUANT"]
          and "mxfp4" in unsupported["blockers"][0]["message"], unsupported["blockers"])

    optional_plan = {"components": [
        {"key": "not_in_catalog_xyz", "dequantPlan": {"weights": 4, "supported": False,
                                                      "unresolvedCount": 4,
                                                      "family": {"name": "mxfp4"}}},
    ]}
    optional = _report(plans=optional_plan)
    check("量化类: **可选件**反解不了 ⇒ 只警告、不阻断 ✓✗（一律硬拦会把能跑的配置挡在门外 ✓）",
          not [item for item in optional["blockers"] if item["kind"] == "quant"]
          and any("可选件" in item for item in optional["warnings"]), optional["warnings"])

    full_precision = {"components": [
        {"key": key, "dequantPlan": {"weights": 0, "supported": False, "unresolvedCount": 0}},
    ]}
    clean = _report(plans=full_precision)
    check("量化类: **全精度**件（`weights=0` ＋ `supported=False`）不算未实现 ✓✗ —— "
          "只盯 `supported` 会把 sdxl_base 这类全精度权重报成「不支持的族」（本模块自己造成过的假阳性 ✓）",
          not [item for item in clean["blockers"] if item["kind"] == "quant"]
          and not clean["warnings"], {"blockers": clean["blockers"], "warnings": clean["warnings"]})


def case_vram() -> None:
    key = _required_key()
    tight = _report(readiness={"missingRequired": [], "brokenRequired": [],
                               "components": [{"key": key, "bytes": 20 * GIB}]},
                    plans={}, capacity_gib=8)
    codes = _codes(tight, "vram")
    check(f"显存类: 必需件 {20} GiB ＞ 8 GiB 卡 ⇒ `E_VRAM_EXHAUSTED` ＋ hint 就是那份单一口径 ✓",
          codes == ["E_VRAM_EXHAUSTED"]
          and tight["blockers"][0]["hint"] == registry.VRAM_HINT
          and "int8Offload" in tight["blockers"][0]["hint"], tight["blockers"])
    check("显存类: 容量来源要标出来 ✓（`given` vs `default` —— 默认值不是这张卡的实测 ✓✗）",
          tight["checks"]["vram"]["facts"].get("capacitySource") == "given",
          tight["checks"]["vram"]["facts"])
    roomy = _report(readiness={"missingRequired": [], "brokenRequired": [],
                               "components": [{"key": key, "bytes": 2 * GIB}]},
                    plans={}, capacity_gib=48)
    check("显存类: 装得下 ⇒ 不阻断 ✓（估算给的是一份没超，不是一份保证 ✓）",
          not [item for item in roomy["blockers"] if item["kind"] == "vram"],
          roomy["checks"]["vram"]["facts"])
    nothing = _report(readiness={"components": []}, plans={})
    check("显存类: 盘上还没有必需件 ⇒ 进 `unchecked`（算不出要占多少 ✓✗）",
          any("显存" in note for note in nothing["unchecked"]), nothing["unchecked"])


def case_unchecked() -> None:
    blank = registry.check_compat("h3", **_blank())
    check("没查 ≠ 通过: 四类都没注入 ⇒ `ok=True` 但 `unchecked` **必须**点名（否则读者会以为查过了 ✓✗）",
          blank["ok"] and len(blank["unchecked"]) >= 3, blank["unchecked"])
    check("没查 ≠ 通过: `unchecked` 不许影响 `ok`（它是诚实标注，不是第二套判据 ✓）",
          all(item["ok"] for item in blank["checks"].values()), blank["checks"])


def case_gate() -> None:
    key = _required_key()
    plans = {"components": [{"key": key, "dequantPlan": {
        "weights": 3, "supported": False, "unresolvedCount": 3, "family": {"name": "mxfp4"}}}]}
    common = {"deps": {"missing": [], "install": [], "optionalMissing": []},
              "readiness": {}, "plans": plans}

    report = registry.ensure_compat("h3", require=("deps", "weights"), **common)
    check("闸门: 默认只硬拦 `deps`/`weights` ⇒ 量化阻断**不抛**（软阻断进报告 ✓）",
          report["hardBlockers"] == [] and not report["ok"], report["hardBlockers"])

    try:
        registry.ensure_compat("h3", require=None, **common)
    except registry.CompatError as err:
        check("闸门: `require=None` ⇒ 四类全硬拦、`CompatError.code` 取第一条阻断的码 ✓"
              "（全部阻断项在 detail 里 ✓ —— 只报一条会让用户修一个撞一个 ✓✗）",
              err.code == "E_UNSUPPORTED_QUANT" and len(err.detail.get("blockers") or []) == 1
              and "装载前就不兼容" in err.message, err.to_dict())
        translated = facade.translate(err)
        check("闸门: 经门面 `translate` 码不变 ✓（`E_UNSUPPORTED_QUANT` 不许翻成 `E_BAD_CONFIG` ✓✗）",
              translated.code == "E_UNSUPPORTED_QUANT", translated.to_dict())
    else:
        check("闸门: `require=None` ⇒ 抛 `CompatError` ✓", False, "没抛错")

    report = registry.ensure_compat("h3", require=(), **common)
    check("闸门: `require=()` ⇒ 只体检不拦 ✓（`load(check=True)` 之外的自检用法 ✓）",
          not report["ok"] and report["hardBlockers"] == [], report["hardBlockers"])


def _calls(source_path: Path, func_name: str) -> list[tuple[int, str]]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != func_name:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func, parts = inner.func, []
            while isinstance(func, ast.Attribute):
                parts.append(func.attr)
                func = func.value
            if isinstance(func, ast.Name):
                parts.append(func.id)
            if parts:
                sites.append((int(inner.lineno), ".".join(reversed(parts))))
    return sites


def _first(sites: list[tuple[int, str]], name: str) -> int | None:
    lines = [line for line, call in sites if call == name]
    return min(lines) if lines else None


def case_wiring() -> None:
    facade_path = BACKEND_PY / "app/services/engine/facade.py"
    init_path = BACKEND_PY / "app/services/engine/__init__.py"
    load_sites = _calls(facade_path, "load")
    gate = _first(load_sites, "registry_mod.ensure_compat")
    load = _first(load_sites, "runtime_mod.engine_runtime.ensure_loaded")
    check("接线: `Engine.load` **先**过闸门、**再**装载 ✓（顺序反了闸门只是在事故后写报告 ✓✗；"
          "按 `lineno` 判序，不比 `ast.walk` 下标 —— 那是广度优先 ✓）",
          gate is not None and load is not None and gate < load,
          {"ensure_compat": gate, "ensure_loaded": load})
    check("接线: 门面导出 `compat` ✓（E2.2 的入口，不必摸到 engine 内部 ✓）",
          "compat" in facade.__all__ and callable(facade.compat), facade.__all__)

    source = facade_path.read_text(encoding="utf-8")
    registry_source = (BACKEND_PY / "app/services/engine/registry.py").read_text(encoding="utf-8")
    check("单一真源: 显存三条路的提示只在 `registry.VRAM_HINT` 一处 ✓✗（门面 OOM 分支不再抄一遍 ✓）",
          "17.94 GiB" not in source and registry_source.count("17.94 GiB") == 1,
          {"facade": source.count("17.94 GiB"), "registry": registry_source.count("17.94 GiB")})

    keys = {str(item.get("key")) for item in inventory.load_catalog()["models"]}
    tree = ast.parse(registry_source)
    literals = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    copied = sorted({value for value in literals if value in keys})
    check("单一真源: 注册表**代码里**没有任何组件 key 字面量 ✓✗（抄一份＝第二个真源，模型换名就静默过期 ✓；"
          "注释里点名实测件作证据是允许的 ✓，所以判据看 AST 字面量而不看原文 ✓）",
          not copied, copied)

    init_source = init_path.read_text(encoding="utf-8")
    missing = [name for name in ("registry", "lora_cache") if f'"{name}"' not in init_source]
    check("接线: 新模块都登记进 `engine.__all__` ✓（磁盘上有、清单里没有是本仓的老毛病 ✓✗）",
          not missing, missing)


def main() -> int:
    case_zero_deps()
    case_registration()
    case_unknown_stage()
    case_deps()
    case_weights()
    case_quant()
    case_vram()
    case_unchecked()
    case_gate()
    case_wiring()

    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name
              + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
