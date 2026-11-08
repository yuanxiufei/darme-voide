"""自检：**引擎产物的兜底落点只有一处真源**（2026-11-07 ✓）。

## 为什么单开一份

四个落盘入口（`torch_backend.write` / `dryrun.write` / `chain.run_chain` /
`sdxl_backend.SdxlBackend.output_path`）此前各自一句兜底 ✓✗：

- 三处 `tempfile.mkdtemp(prefix="engine_*")` ⇒ **每跑一次多一个新目录** ✓，
  图散在 `%TEMP%\\engine_torch_xxxx` ✓、系统一清就没了 ✓，想留还得人工搬 ✓✗；
- `sdxl_backend` 用 `Path.cwd()/"outputs"` ⇒ **换个工作目录换个落点** ✓✗。

现在统一走 :mod:`app.services.engine.output_paths` ✓。本文件钉五件事：

1. **稳定** ✓：同一用途连调两次是**同一个目录**（旧写法每次新目录）；
2. **落点形状** ✓：`<storageRoot>/engine/<kind>/` ✓，干跑另起 `<dataRoot>/engine-dryrun/<kind>/` ✓
   （干跑报告不是产物 ✗ —— 落 `static` 下会被当"成品链接"点开 ✓✗）；
3. **不许穿越** ✓：kind 是拼进路径的 ✓ ⇒ `../` / 子目录 / 空格一律收成单层名 ✓；
4. **业务通路同一棵树** ✓：`bridge.outputs_dir_for` 的根必须就是 `outputs_root()` ✓
   （两处各写一遍 ⇒ 盘上长出两个 `engine/` ✓✗）；
5. **源码不许回退** ✗✗：四处入口的函数体里**再出现** `tempfile` / `mkdtemp` 即红 ✓，
   且全目录不许再有 `mkdtemp(prefix="engine_` ✓。

⚠️ 不跑真生成 ✗（那要真权重 ✓）：本文件只碰路径与源码 ✓。
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
#: ⚠️ 数据根指到临时目录 ✓：本文件要读 `get_storage_root()` / `get_data_root()` 比较，
#: 但**不该**在真数据根下建任何目录 ✓（`Data` 是用户的作品数据 ✓）。
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="engineout_"))
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import output_paths  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _engine_dir() -> Path:
    return BACKEND_PY / "app" / "services" / "engine"


def _function_body(path: Path, qualname: str) -> str | None:
    """取某个函数/方法的**源码片段** ✓（同名方法按文件分开找 ✓）"""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == qualname:
            return ast.get_source_segment(source, node) or ""
    return None


def _identifiers(path: Path, qualname: str | None = None) -> set[str]:
    """函数体（或整文件）里真正被用到的**名字** ✓ —— 只认 AST ✓✗，注释与 docstring 不算 ✓。

    ⚠️ 不能拿源码文本判 ✗：本轮的注释里就写着「不再 `tempfile.mkdtemp`」✓ ⇒
    文本判据会把**说明这句话的注释**当成违规 ✓✗（自己把自己判红 ✓）。
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    scope: ast.AST = tree
    if qualname:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == qualname:
                scope = node
                break
    found: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
    return found


def case_stable() -> None:
    first = output_paths.fallback_outputs_dir("image")
    second = output_paths.fallback_outputs_dir("image")
    check("① 同一用途连调两次 ⇒ **同一个目录** ✓（旧写法 `mkdtemp` 每次一个新目录 ✓✗）",
          first == second and first.is_dir(), {"first": str(first), "second": str(second)})
    check("①′ 没有 override 时根是 `<storageRoot>/engine` ✓（不是 cwd ✓、不是 %TEMP% ✓）",
          first.parent == output_paths.outputs_root()
          and first == output_paths.outputs_root() / "image",
          {"parent": str(first.parent), "root": str(output_paths.outputs_root())})


def case_override() -> None:
    with tempfile.TemporaryDirectory(prefix="engineout_ovr_") as box:
        root = Path(box) / "engine"
        previous = os.environ.get(output_paths.ENV_OVERRIDE)
        os.environ[output_paths.ENV_OVERRIDE] = str(root)
        try:
            target = output_paths.fallback_outputs_dir("video")
            check("② `DARME_ENGINE_OUTPUTS` 整棵改根 ✓（探针/测试隔离用 ✓）",
                  target == root / "video" and target.is_dir(),
                  {"target": str(target), "root": str(root)})
            check("②′ override 时 `outputs_root()` 跟着走 ✓（不是只改落点不改根 ✓）",
                  output_paths.outputs_root() == root, str(output_paths.outputs_root()))
        finally:
            if previous is None:
                os.environ.pop(output_paths.ENV_OVERRIDE, None)
            else:
                os.environ[output_paths.ENV_OVERRIDE] = previous


def case_dryrun_tree() -> None:
    report = output_paths.fallback_outputs_dir("h3", dry_run=True)
    check("③ 干跑落 `<dataRoot>/engine-dryrun/<stage>/` ✓",
          report == output_paths.dryrun_root() / "h3", str(report))
    storage = output_paths.outputs_root()
    check("③′ 干跑**不在**产物树下 ✓✗（落 `static` 会被 `/static` 服务成\"成品链接\" ✓）",
          storage not in report.parents, {"report": str(report), "outputsRoot": str(storage)})

    #: 业务通路给的 `outputs_dir` 是 `<storageRoot>/engine/<kind>-<id>` ✓ ⇒ 照落就是漏进 `static` ✓✗
    given = output_paths.report_dir(storage / "video-eng-abc123")
    check("③″ 调用方给的 `outputs_dir` **只借最后一层名字** ✓✗"
          "（照落 ⇒ 报告进 `static`，`/static` 把 JSON 当成品点开 ✓ —— 2026-11-07 盘上真漏过 4 个 ✓）",
          given == output_paths.dryrun_root() / "video-eng-abc123", str(given))
    staging = output_paths.report_dir(None, "h3")
    check("③‴ 没给 `outputs_dir` ⇒ 按 `stage` 分桶 ✓（与 `fallback_outputs_dir(dry_run=True)` 同路径 ✓）",
          staging == report, {"reportDir": str(staging), "fallback": str(report)})


def case_kind_safe() -> None:
    root = output_paths.outputs_root()
    for raw in ("../escape", "a/b/c", "Video Frame", "..", ""):
        target = output_paths.fallback_outputs_dir(raw, create=False)
        check(f"④ kind 收成单层名 ✓：{raw!r} ⇒ {target.name!r}",
              target.parent == root and ".." not in target.name and "/" not in target.name
              and "\\" not in target.name,
              {"target": str(target), "parent": str(target.parent)})
    check("④′ 空 kind ⇒ `misc` ✓（不拼出 `<root>/` 这种根目录落点 ✓✗）",
          output_paths.fallback_outputs_dir("", create=False).name == output_paths.DEFAULT_KIND,
          output_paths.fallback_outputs_dir("", create=False).name)


def case_create_flag() -> None:
    name = "no-create-probe"
    target = output_paths.outputs_root() / name
    existed = target.exists()
    output_paths.fallback_outputs_dir(name, create=False)
    check("⑤ `create=False` 只算路径不建目录 ✓（调用方要自己 mkdir 时不替它做主 ✓）",
          existed or not target.exists(), str(target))


def case_is_fallback() -> None:
    inside = output_paths.fallback_outputs_dir("image") / "image_seed7.png"
    outside = Path(tempfile.mkdtemp(prefix="engineout_out_")) / "x.png"
    check("⑥ `is_fallback` 认自己家的路径 ✓、不认外面的 ✓",
          output_paths.is_fallback(inside) and not output_paths.is_fallback(outside),
          {"inside": str(inside), "outside": str(outside)})


def case_no_tempfile_regression() -> None:
    #: 干跑报告走 `report_dir` ✓（它内部就是 `dryrun_root()` ✓ —— 比 `fallback_outputs_dir`
    #: 更严：**调用方给了 `outputs_dir` 也只借名字** ✓，否则业务通路的 `static/engine/<kind>-<id>`
    #: 会把报告带进 `static` ✓✗）
    entries = {
        "torch_backend.py": ("write", "fallback_outputs_dir"),
        "dryrun.py": ("write", "report_dir"),
        "chain.py": ("run_chain", "fallback_outputs_dir"),
        "sdxl_backend.py": ("output_path", "fallback_outputs_dir"),
    }
    for filename, (qualname, entry) in entries.items():
        path = _engine_dir() / filename
        body = _function_body(path, qualname)
        used = _identifiers(path, qualname)
        check(f"⑦ {filename}::{qualname} 兜底走 `output_paths.{entry}` ✓",
              body is not None and entry in used,
              "找不到该函数" if body is None else body[:80])
        forbidden = used & {"mkdtemp", "tempfile", "cwd"}
        check(f"⑦′ {filename}::{qualname} **不再**自己 `mkdtemp` / 用 cwd 兜底 ✓✗"
              "（AST 判据 ✓ —— 注释里写「不再 mkdtemp」不算违规 ✓）",
              not forbidden, sorted(forbidden))

    offenders = []
    for path in sorted(_engine_dir().glob("*.py")):
        used = _identifiers(path)
        if "mkdtemp" in used:
            offenders.append(path.name)  # 造出来的目录**没人清** ⇒ 正是"每跑一次留一堆"的根源 ✗
        elif "tempfile" in used and "TemporaryDirectory" not in used:
            offenders.append(f"{path.name}（用了 tempfile 却没有 TemporaryDirectory ✓✗）")
    check("⑦″ 全引擎目录**零** `mkdtemp` ✓✗（\"以后不用搬来搬去\"的硬判据 ✓）；"
          "`tempfile` 只许 `TemporaryDirectory` 那种**自动清理**的用法 ✓"
          "（如 `torch_backend` 抽音轨的中转目录 ✓）",
          offenders == [], offenders)


def case_bridge_same_tree() -> None:
    body = _function_body(_engine_dir() / "bridge.py", "outputs_dir_for")
    check("⑧ `bridge.outputs_dir_for` 的根也来自 `output_paths` ✓（两处各写一遍 ⇒ 盘上两个 engine/ ✓✗）",
          body is not None and "outputs_root()" in body and "dryrun_root()" in body,
          "函数体没走 output_paths")
    try:
        from app.services.engine import bridge  # noqa: PLC0415
    except Exception as error:  # noqa: BLE001
        check("⑧′ `bridge.outputs_dir_for` 真调一次 ✓", False, f"{type(error).__name__}: {error}")
        return
    product = bridge.outputs_dir_for("image", "abc123")
    report = bridge.outputs_dir_for("image", "abc123", dry_run=True)
    check("⑧′ 业务通路 = `<outputsRoot>/<kind>-<id>` ✓（同 seed 连跑不互相覆盖 ✓）",
          Path(product) == output_paths.outputs_root() / "image-abc123", str(product))
    check("⑧″ 干跑通路 = `<dryrunRoot>/<kind>-<id>` ✓",
          Path(report) == output_paths.dryrun_root() / "image-abc123", str(report))


def main() -> int:
    case_stable()
    case_override()
    case_dryrun_tree()
    case_kind_safe()
    case_create_flag()
    case_is_fallback()
    case_no_tempfile_regression()
    case_bridge_same_tree()

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
