r"""自研引擎 · **独立性契约**（WBS-1.4 / 1.5 / 2.1 / 2.6 的机器判据）。

「引擎能独立跑」这句话必须能被**机器**验证，否则只是口号：本轮之前的实况是「能力都在、
门面没有」—— 想拿引擎的结论只能起整个后端 + 调 HTTP，于是「引擎自己行不行」永远混着 Web 层。

覆盖：

1. **静态**：`engine/*.py` 不许 import `fastapi` / `starlette` / `pydantic` / `app.routers` /
   `app.services.*`（唯一允许的仓内依赖是 `app.core.*`）—— 用 AST 扫，不靠人记；
2. **门面契约**：`API_VERSION` 语义化；`EngineError` 每个错误码都有中文说明；
3. **无 torch 也能回答**：`capabilities` / `plan` / `inspect` 只靠文件头（子进程里把 `torch`
   打成 `None` 再跑）；
4. **损坏权重不能被误判为有效模型**（WBS G2 前置）：截断 ⇒ `E_WEIGHT_CORRUPT`、
   不存在 ⇒ `E_WEIGHT_MISSING`、未知阶段 ⇒ `E_BAD_CONFIG`；
5. **两处判据不漂**：门面 `readiness` 的未实现族集合 == 项目侧 `engine_readiness.collect()`
   的同一集合（同一条 `comfy_quant` 证据，不该一处一个说法）；
6. **CLI 是独立入口**：`python -m app.services.engine doctor --json` 退出 0 且带 apiVersion；
   `readiness` 在不可装时退出 1（退出码可机器判）；
7. **清单诚实**：`manifest_of` 在清单没有 `license` 时输出 `None` + `licenseNote`（不猜许可）；
   头部哈希对同一文件稳定、对头部改动敏感。

运行::

    ./.venv/Scripts/python.exe tests/engine_independence_test.py
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from app.services.engine import facade  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_ENGINE = _ROOT / "app" / "services" / "engine"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, ok: bool, detail: object = None) -> None:
    _RESULTS.append((name, bool(ok), detail))


# ══════════════════════════════════════════════════════════════════════════
# ① 静态：引擎不许碰 Web 层与项目层
# ══════════════════════════════════════════════════════════════════════════
_FORBIDDEN = ("fastapi", "starlette", "pydantic", "uvicorn", "sqlalchemy",
              "app.routers", "app.services.engine_readiness", "app.services.production",
              "app.db", "app.models")


def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module)
    return found


def case_static() -> None:
    files = sorted(_ENGINE.glob("*.py"))
    violations: list[str] = []
    app_deps: dict[str, set[str]] = {}
    for path in files:
        for name in _imports_of(path):
            if any(name == bad or name.startswith(bad + ".") for bad in _FORBIDDEN):
                violations.append(f"{path.name}: {name}")
            if name == "app" or name.startswith("app."):
                app_deps.setdefault(path.name, set()).add(name)
    check(f"① 引擎 {len(files)} 个模块**不 import** Web 层/项目层（fastapi·pydantic·routers·services）",
          not violations, violations)
    #: 允许的仓内依赖只有两类：`app.core.*`（显存预算这类纯工具）＋ `app.services.engine`**自己**
    #: （包内互引写成绝对路径是正常的 ✓ —— 但它必须是**唯一**的外部前提 ✗）
    def _allowed(dep: str) -> bool:
        return dep == "app.services.engine" or dep.startswith(("app.core", "app.services.engine."))

    bad_app = {name: sorted(deps) for name, deps in app_deps.items()
               if any(not _allowed(dep) for dep in deps)}
    check("② 引擎对仓内的依赖**只有** `app.core.*` ＋ 引擎自己（其余一律不许）", not bad_app, bad_app)


# ══════════════════════════════════════════════════════════════════════════
# ② 门面契约
# ══════════════════════════════════════════════════════════════════════════
def case_contract() -> None:
    parts = facade.API_VERSION.split(".")
    check("③ API_VERSION 是语义化版本（MAJOR.MINOR.PATCH，全数字）",
          len(parts) == 3 and all(part.isdigit() for part in parts), facade.API_VERSION)
    check("④ 每个错误码都带中文说明（机器可判 + 人能读）",
          all(str(text).strip() for text in facade.CODES.values()) and len(facade.CODES) >= 6,
          sorted(facade.CODES))
    caps = facade.capabilities()
    check("⑤ 能力声明把「已验证 / 未验收 / 边界」三档分开写",
          isinstance(caps.get("operations"), list) and caps.get("unverified")
          and caps.get("limits") and caps.get("apiVersion") == facade.API_VERSION,
          {"ops": caps.get("operations"), "unverified": caps.get("unverified")})


# ══════════════════════════════════════════════════════════════════════════
# ③ 不 import torch 也能回答
# ══════════════════════════════════════════════════════════════════════════
def case_no_torch() -> None:
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "from app.services.engine import facade\n"
        "c = facade.capabilities()\n"
        "p = facade.plan('h3')\n"
        "print(c['apiVersion'], len(p['components']), c['backend']['available'])\n"
    )
    proc = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=str(_ROOT),
                          capture_output=True, text=True, encoding="utf-8", timeout=180)
    ok = proc.returncode == 0 and proc.stdout.strip().startswith(facade.API_VERSION)
    check("⑥ 把 torch 打成 None 后，能力/计划**照样**能算（只靠文件头 ⇒ 引擎真独立）",
          ok, (proc.stdout or "").strip()[:200] or (proc.stderr or "").strip()[-300:])


# ══════════════════════════════════════════════════════════════════════════
# ④ 假权重：坏文件必须被当场挡住
# ══════════════════════════════════════════════════════════════════════════
def _write_safetensors(path: Path, tensors: dict[str, dict[str, Any]],
                       *, data: bytes = b"", truncate: bool = False) -> None:
    """手写一个合法 safetensors 头（**不借 torch** —— 本测试要在无 torch 环境也成立）。"""
    offset = 0
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    for name, meta in tensors.items():
        size = int(meta["size"])
        header[name] = {"dtype": meta["dtype"], "shape": list(meta["shape"]),
                        "data_offsets": [offset, offset + size]}
        offset += size
    payload = json.dumps(header).encode("utf-8")
    blob = len(payload).to_bytes(8, "little") + payload + (data or b"\x00" * offset)
    if truncate:
        blob = blob[: 8 + len(payload) // 2]
    path.write_bytes(blob)


def case_bad_weights() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.safetensors"
        _write_safetensors(good, {"w.weight": {"dtype": "F16", "shape": [2, 3], "size": 12}})
        report = facade.inspect_weights(good)
        check("⑦ 合法假权重：结构过 ⇒ ok、张量数对、无量化族（不该凭空报错）",
              report["ok"] and report["tensorCount"] == 1 and not report["formats"],
              {"tensors": report["tensorCount"], "formats": report["formats"]})

        quan = Path(tmp) / "convrot.safetensors"
        #: ⚠️ `comfy_quant` 的 size 必须**按真实 JSON 长度**算 ✗（写死 58 会让头部声明的
        #: 偏移量对不上实际字节 ⇒ 结构校验先失败 ✓✗，于是测的就不是「族识别」而是「头部校验」）
        spec = b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
        _write_safetensors(quan, {
            "w.weight": {"dtype": "I8", "shape": [2, 3], "size": 6},
            "w.weight_scale": {"dtype": "F16", "shape": [2], "size": 4},
            "w.comfy_quant": {"dtype": "U8", "shape": [len(spec)], "size": len(spec)},
        }, data=(b"\x01" * 6 + b"\x00" * 4 + spec))
        #: ⚠️ 2026-10-29 改口径 ✓：convrot **已实现** ✓（`quant.rotate_weight` / 前向搬到激活侧 ✓），
        #: 不再是「未实现族」✗。现在的判据是「**认出来**并且如实报 group」✓ —— 若哪天它被悄悄退回
        #: 未实现族 ✓✗，或 `convrot_group_of` 读不出 256 ✗，这条都会红 ✓。
        #: ⚠️ 这个假 weight 只有 `[2,3]` ✗（末维 3 不能被 group=256 整除 ✓）⇒ 正确结果是
        #: **认得出 convrot=256** ✓，同时**拒绝**把它当普通 per-row ✗（进 `unresolved` ✓）——
        #: 这恰好就是「绝不静默当普通 per-row」那条纪律的活证据 ✓。
        got = facade.inspect_weights(quan)
        check("⑧ 带 `comfy_quant` 的 convrot 权重 ⇒ **认得**（不再进未实现族 ✓）且 "
              "`quant.convrot==256` ✓；形状与 256 组不符 ⇒ **拒绝**当普通 per-row ✓（进 unresolved ✓）",
              got["ok"] and got["family"] is None and got["quant"]["convrot"] == 256
              and "w" in (got["quant"]["unresolved"] or [])
              and not got["quant"]["supported"],
              {"convrot": got["quant"]["convrot"], "quant": got["quant"]})

        cut = Path(tmp) / "cut.safetensors"
        _write_safetensors(cut, {"w.weight": {"dtype": "F16", "shape": [2, 3], "size": 12}},
                           truncate=True)
        try:
            facade.inspect_weights(cut)
            code = "no-error"
        except facade.EngineError as err:
            code = err.code
        check("⑨ 截断文件 ⇒ **E_WEIGHT_CORRUPT**（坏权重不能被当成有效模型 ✓ G2 前置）",
              code == "E_WEIGHT_CORRUPT", code)

        try:
            facade.inspect_weights(Path(tmp) / "nope.safetensors")
            code = "no-error"
        except facade.EngineError as err:
            code = err.code
        check("⑩ 文件不存在 ⇒ **E_WEIGHT_MISSING**（与「坏了」分开，处置方式不同）",
              code == "E_WEIGHT_MISSING", code)

        try:
            facade.readiness("nope-stage")
            code = "no-error"
        except facade.EngineError as err:
            code = err.code
        check("⑪ 未知阶段 ⇒ **E_BAD_CONFIG**（参数错不等于是引擎坏）", code == "E_BAD_CONFIG", code)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 两处判据不漂（门面 vs 项目侧体检）
# ══════════════════════════════════════════════════════════════════════════
def case_no_drift() -> None:
    from app.services import engine_readiness as outer  # noqa: PLC0415 —— 项目侧，只在本测试用

    mine = {item["key"] for item in facade.readiness("h3")["unsupportedQuant"]}
    theirs = {item["key"] for item in (outer.collect()["quantPlan"].get("unsupported") or [])
              if item.get("family")}
    check("⑫ 门面与项目侧体检的「未实现量化族」集合**完全一致**（同一份证据，不许两处两说）",
          mine == theirs, {"facade": sorted(mine), "project": sorted(theirs)})


# ══════════════════════════════════════════════════════════════════════════
# ⑥ CLI 是独立入口 + 退出码可机器判
# ══════════════════════════════════════════════════════════════════════════
def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-X", "utf8", "-m", "app.services.engine", *args],
                          cwd=str(_ROOT), capture_output=True, text=True,
                          encoding="utf-8", timeout=300)


def case_cli() -> None:
    doctor = _cli("doctor", "--json")
    try:
        payload = json.loads(doctor.stdout)
    except ValueError:
        payload = {}
    check("⑬ `python -m app.services.engine doctor --json` 退出 0 且输出带 apiVersion",
          doctor.returncode == 0 and payload.get("apiVersion") == facade.API_VERSION,
          {"exit": doctor.returncode, "head": (doctor.stdout or "")[:120],
           "err": (doctor.stderr or "")[-200:]})

    ready = _cli("readiness", "--json")
    try:
        data = json.loads(ready.stdout)
    except ValueError:
        data = {}
    #: 退出码必须与结论**一致**：不可装 ⇒ 1；可装 ⇒ 0（不许「退出 0 但报告说不能跑」）
    expect = 0 if data.get("ready") else 1
    check("⑭ `readiness` 的退出码跟结论走（0＝可装 / 1＝不可装，脚本能直接判）",
          ready.returncode == expect, {"exit": ready.returncode, "ready": data.get("ready")})


# ══════════════════════════════════════════════════════════════════════════
# ⑦ 清单诚实：许可不猜 / 头部哈希稳定
# ══════════════════════════════════════════════════════════════════════════
def case_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.safetensors"
        _write_safetensors(good, {"w.weight": {"dtype": "F16", "shape": [2, 3], "size": 12}})
        first = facade.manifest_of(good)
        second = facade.manifest_of(good)
        check("⑮ 头部哈希对同一文件**稳定**（否则清单没法当凭据）",
              first["headSha256"] == second["headSha256"] and first["hashScope"] == "header+size",
              first["hashScope"])
        check("⑯ 清单没有 license ⇒ 输出 None + licenseNote（**不猜许可**）",
              first["license"] is None and bool(first["licenseNote"]), first["licenseNote"])
        check("⑰ 没要全量哈希时 `sha256` 是 None（不假装算过全文件）",
              first["sha256"] is None, first["sha256"])

        #: 改头部（加一个张量）⇒ 头部哈希必须变（否则这份凭据毫无鉴别力）
        _write_safetensors(good, {"w.weight": {"dtype": "F16", "shape": [2, 3], "size": 12},
                                  "w.bias": {"dtype": "F16", "shape": [3], "size": 6}})
        third = facade.manifest_of(good)
        check("⑱ 头部一改 ⇒ 头部哈希**必须变**（有鉴别力才算凭据）",
              third["headSha256"] != first["headSha256"],
              {"old": first["headSha256"][:12], "new": third["headSha256"][:12]})


def main() -> int:
    case_static()
    case_contract()
    case_no_torch()
    case_bad_weights()
    case_no_drift()
    case_cli()
    case_manifest()

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
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
