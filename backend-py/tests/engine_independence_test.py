r"""自研引擎 · **独立性契约**（WBS-1.4 / 1.5 / 2.1 / 2.6 的机器判据）。

「引擎能独立跑」这句话必须能被**机器**验证，否则只是口号：本轮之前的实况是「能力都在、
门面没有」—— 想拿引擎的结论只能起整个后端 + 调 HTTP，于是「引擎自己行不行」永远混着 Web 层。

覆盖：

1. **静态**：`engine/*.py` 不许 import `fastapi` / `starlette` / `pydantic` / `app.routers` /
   `app.services.*`（唯一允许的仓内依赖是 `app.core.*`）—— 用 AST 扫，不靠人记。
   ★**相对写法同样算数**（`from .. import outcome_index` ⇒ 折算成 `app.services.outcome_index`）：
   2026-11-08 之前只收 `level == 0` ⇒ 这一整类**全被放过** ✓✗（`qc_feedback` 藏了一轮、
   `inventory` 更是写在模块头上 ⇒「引擎一被 import 就必须带着服务层」）。现在补三条：
   ② 服务层依赖必须**逐条**在 `_SEAMS`（含"为什么"＋"怎么撤销"）／③ 表与代码**双向对账**
   （修好的必须删表）／⑤ **模块加载期**不许碰服务层（`bridge.py` 是服务侧桥，豁免）；
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

#: 本包的**绝对名**：相对 import 按它折算 ✓✗（不折算就等于判据只认一种写法 ✓）。
_PKG = "app.services.engine"

#: 服务层里被引擎碰到的**全部**模块，逐条登记 ✓✗ ——
#: `(文件, 绝对模块名) -> (为什么非碰不可, 怎么撤销, 用了它的哪几个名字)`。
#:
#: ⚠️ 这张表是**三向对账**的 ✓：表里有、代码里已经没有了 ⇒ 红（白名单过期 ✗）；
#: 代码里有、表里没有 ⇒ 红（偷偷多一条依赖 ✗）；**`uses` 与真实 import 的名字对不上 ⇒ 红**
#: （③b）。所以「先随手加一条豁免」这条路是通的 ✓，但**撤销条件必须写成人话且可判**
#: （写"以后再说"会被 ④ 拦 ✓）。
#:
#: 为什么要 `uses`（2026-11-08 补 ✓）：`inventory.py` 对扫描层曾经用了**两个**名字
#: （`get_model_paths` + `get_default_roots`）—— 按模块登记的话，悄悄销掉其中一个不会红 ✓✗，
#: 表就停在"两个都要"的旧口径上，「还欠什么」没人知道 ✓✗。降到名字粒度后，销一个**必须**改表
#: （销完就得删条），账目才对得上 ✓。⚠️ 这条规则当天就应验了两次 ✓：`get_model_paths` 先搬去
#: `app.core.model_paths`、随后 `get_default_roots` 连同根表整体搬去 `app.core.model_roots`
#: （连同 `model_folders` / `model_ecosystems` / `ollama_roots`）⇒ `inventory.py` 整条已删 ✓。
#:
#: ⚠️ 现阶段只剩**一条** ✓ —— `bridge.py → task_logger`（任务日志面板的语料，引擎侧没有等价物 ✓、
#: 也**不该**有：那是"给谁看"的业务口径，不是"能不能跑"的引擎口径 ✓）。这不是"迁就现状"的余额 ✓，
#: 而是逐条撤销后剩下的**真缺口**：要销它得先让路由器自己落日志（见该条"怎么撤销"）。
_SEAMS: dict[tuple[str, str], tuple[str, str, tuple[str, ...]]] = {
    ("bridge.py", "app.services.task_logger"): (
        "任务日志（progress/warn/error 三档）是**服务层任务面板**的语料 ✓，"
        "引擎侧没有等价物 ✓，桥负责把引擎的回执翻成面板能读的条目 ✓",
        "引擎回执里已有结构化 notes ⇒ 让调用方（routers）自己落日志，桥不再写盘",
        ("log_task_error", "log_task_progress", "log_task_warn"),
    ),
}


def _seam_of(name: str, dep: str) -> bool:
    return (name, dep) in _SEAMS


def _package_of(path: Path) -> str:
    """该文件**所在包**的绝对名 ✓✗ —— 相对 import 是相对**包**解析的，不是相对模块 ✓。

    （`from ...core.config import x` 里的三个点从 ``app.services.engine`` 往上退两级 ⇒
    ``app.core.config`` ✓；把 stem 也拼进去会错算成 ``app.services.core.config`` ✗。）
    """
    return _PKG


def _absolutize(node: ast.ImportFrom, package: str) -> list[str]:
    """把 ``node`` 折成**绝对模块名** ✓（`level == 0` 原样 ✓）。

    ⚠️ 此前只收 ``level == 0`` ✓✗ ⇒ ``from .. import outcome_index``、``from ..file_storage
    import x`` 这一整类**相对写法全被放过** ✗ —— 判据写着"引擎不 import 服务层"，实际只要
    换个写法就能绕过 ✓（2026-11-08 实测：``qc_feedback`` 藏了一轮、``inventory`` 更是
    写在模块头上 ⇒ 「引擎一被 import 就必须带着服务层」✓✗）。折算规则与 Python 一致：
    ``level=1`` 是本包、``level=N`` 上退 ``N-1`` 级 ✓。
    """
    if node.level == 0:
        return [node.module] if node.module else []
    parts = package.split(".")
    base = ".".join(parts[: len(parts) - (node.level - 1)])
    if node.module:
        return [f"{base}.{node.module}"]
    #: `from .. import X`：**X 可能是模块** ⇒ 只有这种写法需要把名字也展开 ✓
    return [f"{base}.{alias.name}" for alias in node.names]


def _imports_of(path: Path, *, module_level_only: bool = False) -> list[str]:
    """该文件 import 了哪些**绝对模块名** ✓（相对写法先折算 ✓）。

    ``module_level_only`` ⇒ 只收**模块加载期**会执行的那些 ✓（``if``/``try`` 里的算 ✓，
    函数/类体里的不算 ✓）—— 判「一 import 这个模块会不会顺带把服务层拽起来」✓。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _package_of(path)

    nodes: list[ast.AST]
    if module_level_only:
        nodes = []
        stack: list[ast.AST] = list(tree.body)
        while stack:
            node = stack.pop()
            nodes.append(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue  #: 函数体/类体里的 import 是**调用时**才执行 ✓，不算加载期 ✓
            stack.extend(ast.iter_child_nodes(node))
    else:
        nodes = list(ast.walk(tree))

    found: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.extend(_absolutize(node, package))
    return found


def _import_names(path: Path) -> dict[str, set[str]]:
    """``绝对模块名 -> 从它那儿 import 的**名字**集合`` ✓✗。

    * ``from X import a, b`` ⇒ ``{X: {a, b}}``（判断"用了它的哪几个口"✓）；
    * ``from .. import outcome_index`` ⇒ ``{app.services.outcome_index: {outcome_index}}``
      （模块本身就是那个名字 ✓，调用处写的是 ``outcome_index.xxx`` ✓）；
    * ``import a.b`` ⇒ 绑定名是 ``a``（``asname`` 优先 ✓）。

    ⚠️ 别拿「文件全文里出现过 ``get_model_paths`` 这几个字」当对账 ✗：注释、docstring、
    另一个同名变量全都会误命中 ✓✗ —— 只有 AST 里的 ``alias.name`` 是**真的在用** ✓。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _package_of(path)
    names: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.setdefault(alias.name, set()).add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            deps = _absolutize(node, package)
            if node.module:
                #: `from <mod> import a, b` ⇒ 两个名字都挂在同一个 dep 上 ✓
                for dep in deps:
                    names.setdefault(dep, set()).update(a.name for a in node.names)
            else:
                #: `from .. import X` ⇒ `_absolutize` 与 `node.names` **一一对应** ✓
                for dep, alias in zip(deps, node.names):
                    names.setdefault(dep, set()).add(alias.name)
    return names


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

    bad_app = sorted(f"{name}: {dep}" for name, deps in app_deps.items()
                     for dep in deps if not _allowed(dep) and not _seam_of(name, dep))
    check("② 引擎对仓内的依赖**只有** `app.core.*` ＋ 引擎自己；碰服务层必须**逐条**在 `_SEAMS` 里",
          not bad_app, bad_app)

    #: ②b **搬家 ≠ 抄一份** ✓✗：把口搬进 `app.core` 之后，服务层那份必须是**同一个对象**的再导出
    #: （`is`）✓ —— 否则全仓就是两套实现：引擎走新的、业务走旧的，改一边不改另一边没人知道 ✓✗
    #: （`get_absolute_path` 是**防路径穿越**的那道闸 ✓，抄两份必然漂 ✓）。2026-11-08 把
    #: `file_storage` 的两个纯函数搬进 `app.core.media` 时补的这条 ✓。
    from app.core import media as core_media
    from app.services import file_storage as svc_storage

    copied = [name for name in ("get_absolute_path", "parse_data_url")
              if getattr(svc_storage, name, None) is not getattr(core_media, name, None)]
    check("②b 搬进 `app.core` 的媒体口（`get_absolute_path`/`parse_data_url`）在服务层那份"
          "是**同一个对象**（`is` 判；另写一份 = 第二真源 ✗）", not copied, copied)

    #: ②b′ 同一把尺子量**模型落点**那批（2026-11-08 搬 ✓）：引擎现在直接从 `app.core` 取
    #: `comfyui_roots` / `model_roots` / `model_ecosystems` ⇒ 服务层那几份**必须**是同一个对象 ✓✗。
    #: 抄一份的后果特别隐蔽：扫描层与引擎各读一张表 ⇒「清单说装了、加载说没有」✓✗
    #: （本仓 2026-09-26 真机踩过同类：扫描看得见、视频看不见 ✓）。
    from app.core import comfyui_roots as core_comfyui
    from app.core import model_ecosystems as core_eco
    from app.core import model_roots as core_roots
    from app.services import local_model_scan as svc_scan
    from app.services import model_ecosystems as svc_eco

    twins = {
        "local_model_scan.detect_comfyui_roots": (svc_scan.detect_comfyui_roots,
                                                  core_comfyui.detect_comfyui_roots),
        "local_model_scan.comfyui_roots→LOCAL_SERVICES_ROOT": (svc_scan.LOCAL_SERVICES_ROOT,
                                                               core_roots.LOCAL_SERVICES_ROOT),
        "model_ecosystems.ECOSYSTEMS": (svc_eco.ECOSYSTEMS, core_eco.ECOSYSTEMS),
    }
    copied_roots = [f"{name} 不是同一个对象" for name, (a, b) in twins.items() if a is not b]
    check("②b′ 搬进 `app.core` 的模型落点口（安装根探测 / 本地服务根 / 生态落点表）在服务层那份"
          "是**同一个对象**（`is` 判；另写一份 = 第二真源 ✗）", not copied_roots, copied_roots)

    #: ③ 双向对账：表里每一条都必须在代码里真存在（修好了就删表 ⇒ 表不许长霉 ✓✗）
    stale: list[str] = []
    for (name, dep) in _SEAMS:
        path = _ENGINE / name
        if not path.exists():
            stale.append(f"{name}：文件不存在")
        elif dep not in _imports_of(path):
            stale.append(f"{name}：已不再 import {dep} ⇒ 请删掉这条（白名单过期 ✗）")
    check("③ `_SEAMS` 每一条在代码里都还真的存在（已修好的必须删表，不许留下僵尸豁免 ✗）",
          not stale, stale)

    #: ③b **名字粒度**对账 ✓✗：一条依赖常常只用了对方几个函数（`from X import a, b`）。
    #: 只按模块登记的话，悄悄销掉其中一个**不会红** ⇒ 表停留在「a 和 b 都要」的旧口径上，
    #: 「还欠什么」就没人知道了 ✓✗（2026-11-08 把 `models_dir` 搬进 `app.core` 时实测踩到 ✓）。
    wrong_uses: list[str] = []
    for (name, dep), (_why, _exit, uses) in _SEAMS.items():
        path = _ENGINE / name
        if not path.exists():
            continue
        actual = _import_names(path).get(dep, set())
        if actual != set(uses):
            wrong_uses.append(f"{name}→{dep}: 表里 {sorted(uses)} ≠ 实际 {sorted(actual)}")
    check("③b `_SEAMS.uses` 与代码里 `from … import …` 的**名字逐个**一致（销掉一个就得改表 ✓✗）",
          not wrong_uses, wrong_uses)

    #: ③c 空 `uses` 等于回到模块粒度 ⇒ 放它过去，③b 就白做了 ✓✗
    check("③c 每条豁免都写明了用到的名字（空表 = 又退回模块粒度 ✗）",
          all(uses for (_w, _e, uses) in _SEAMS.values()),
          [f"{n}→{d}" for (n, d), (_w, _e, uses) in _SEAMS.items() if not uses])

    #: ④ 每条豁免都得写清「为什么现在非碰不可」＋「怎么撤销」——两句话，别写"以后再说" ✓✗
    weak = [f"{name}→{dep}" for (name, dep), (why, exit_, _uses) in _SEAMS.items()
            if len(str(why).strip()) < 20 or len(str(exit_).strip()) < 20
            or "以后" in str(exit_) or "待定" in str(exit_)]
    check("④ 豁免不能是空话：`为什么` 与 `怎么撤销` 各 ≥20 字，撤销条件不许写「以后/待定」",
          not weak, weak)

    #: ⑤ **模块加载期**只许碰 `app.core` ＋ 自己（服务侧桥 `bridge.py` 例外：它本来就是服务层伸进来的）
    #: —— 这条盯的是「`import app.services.engine.<任何核心模块>` 会不会顺带把服务层拽起来」✓✗，
    #: 与 ② 不同：② 管"依赖存在"，⑤ 管"什么时候发生"✓。此前 `inventory` 的服务层 import 就在模块头 ✗。
    load_time: dict[str, list[str]] = {}
    for path in files:
        if path.name == "bridge.py":  # 服务侧桥：routers 调它，模块头摸服务层是它的本职 ✓
            continue
        deps = sorted(dep for dep in set(_imports_of(path, module_level_only=True))
                      if (dep == "app" or dep.startswith("app.")) and not _allowed(dep))
        if deps:
            load_time[path.name] = deps
    check("⑤ 引擎核心**模块加载期**只碰 `app.core.*` 与自身（服务层依赖必须写在函数里 ✓；"
          "`bridge.py` 是服务侧桥，豁免 ✓）", not load_time, load_time)


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
# ⑥b 任务级可行性入口（§5.3）：退出码 0/1/2 机器可判 + 尺寸委托画布 + 估时不编
# ══════════════════════════════════════════════════════════════════════════
def case_feasibility_cli() -> None:
    plain = _cli("feasibility", "--stage", "krea2", "--json")
    try:
        data = json.loads(plain.stdout)
    except ValueError:
        data = {}
    check("⑲ `feasibility --json` 退出 0 且**三格齐**（verdict / estimate / size）",
          plain.returncode == 0 and bool(data.get("verdict"))
          and isinstance(data.get("estimate"), dict) and "size" in data,
          {"exit": plain.returncode, "keys": sorted(data)[:12],
           "err": (plain.stderr or "")[-200:]})
    estimate = data.get("estimate") or {}
    check("⑳ 没给样本 ⇒ `runSeconds=null` ＋ `basis=unknown`（**不是 0 秒**）",
          estimate.get("runSeconds") is None and estimate.get("basis") == "unknown",
          estimate)
    check("㉑ `loadSeconds` 恒 null（本仓只记整任务耗时 ⇒ 不拆一个假装载秒数）",
          "loadSeconds" in estimate and estimate["loadSeconds"] is None, estimate)

    sized = _cli("feasibility", "--stage", "krea2", "--size", "1000x1000", "--json")
    try:
        sdata = json.loads(sized.stdout)
    except ValueError:
        sdata = {}
    from app.services.engine import canvas as canvas_mod  # noqa: PLC0415

    want = canvas_mod.plan_canvas("krea2", width=1000, height=1000)
    size = sdata.get("size") or {}
    check("㉒ `--size` 的吸附**与 `canvas.plan_canvas` 同值**（就近 992 不是 1024/1008）",
          size.get("snapped") == f"{int(want.width)}x{int(want.height)}"
          and size.get("step") == int(want.grid) and size.get("absorbed") is True,
          size)
    check("㉓ 给尺寸**不改 verdict**（这一栏只报几何；本仓没有逐族激活峰值表）",
          sdata.get("verdict") == data.get("verdict"),
          {"plain": data.get("verdict"), "sized": sdata.get("verdict")})

    measured = _cli("feasibility", "--stage", "krea2", "--measure", "95,102,88", "--json")
    try:
        mdata = json.loads(measured.stdout)
    except ValueError:
        mdata = {}
    mestimate = mdata.get("estimate") or {}
    run = mestimate.get("runSeconds") or {}
    check("㉔ `--measure 95,102,88` ⇒ basis=measured 且 p50/p95 是**出现过的实测值**（秒）",
          mestimate.get("basis") == "measured" and mestimate.get("samples") == 3
          and run.get("p50") in (88.0, 95.0, 102.0) and run.get("p95") == 102.0,
          mestimate)

    bad_stage = _cli("feasibility", "--stage", "nope")
    check("㉖ 未知阶段 ⇒ 退出 2（用法错，不是「跑不了」的 1）", bad_stage.returncode == 2,
          {"exit": bad_stage.returncode, "err": (bad_stage.stderr or "")[-200:]})

    strict = _cli("feasibility", "--stage", "krea2", "--capacity", "4", "--strict")
    loose = _cli("feasibility", "--stage", "krea2", "--capacity", "4")
    check("㉗ `--strict` 只认 perfect/good ⇒ 紧档判 1，不加则能跑（marginal）判 0",
          strict.returncode == 1 and loose.returncode == 0,
          {"strict": strict.returncode, "loose": loose.returncode})

    #: ⭐ 反转判据 ✓✗：``--from-ledger`` **故意不提供** ✗ —— 台账在业务库 ✓，
    #: 引擎内核（`feasibility.py` 零 IO ✓）不许读它 ✓。这条一松，内核就开始依赖库了 ✗✗。
    ledger_cli = _cli("feasibility", "--stage", "krea2", "--from-ledger")
    check("㉘ 内核 CLI **没有** `--from-ledger`（台账归服务层，内核不许读业务库）",
          ledger_cli.returncode == 2,
          {"exit": ledger_cli.returncode, "err": (ledger_cli.stderr or "")[-160:]})

    #: ⚠️ 但**服务层**必须给得出来 ✓（不然"从台账估时"这条能力就是空话 ✓✗）：
    from app.services import engine_readiness  # noqa: PLC0415 —— 这是**项目侧**测试 ✓

    block = engine_readiness.run_samples("krea2")
    check("㉙ 服务层 `run_samples` 三态分明：`available/total/samples/why` 齐全且能自圆其说",
          {"available", "total", "samples", "why"} <= set(block)
          and (bool(block.get("samples")) or bool(block.get("why"))),
          block)
    #: 样本不足闸门（含 0 条）⇒ 估时**必须**是 null ✓（不许把 1～2 条当结论 ✗）
    service = engine_readiness.task_feasibility("krea2", use_ledger=True)
    lestimate = service.get("estimate") or {}
    check("㉚ 服务层 `use_ledger=True` ⇒ 回执带 `ledger`，且样本不足时 `runSeconds=null`",
          "ledger" in service
          and (lestimate.get("samples") or 0) >= 0
          and (lestimate.get("basis") == "measured" or lestimate.get("runSeconds") is None),
          {"samples": lestimate.get("samples"), "basis": lestimate.get("basis")})
    check("㉛ `use_ledger=False`（默认）⇒ 回执里**没有** `ledger`（不碰库就不留痕迹）",
          "ledger" not in engine_readiness.task_feasibility("krea2"),
          sorted(engine_readiness.task_feasibility("krea2")))

    plan = _cli("plan", "--stage", "krea2", "--json")
    try:
        pdata = json.loads(plan.stdout)
    except ValueError:
        pdata = {}
    check("㉜ 老的 `plan`（装载计划）**没被改成** verdict：仍出 `components` 且退出码跟件齐不齐",
          isinstance(pdata.get("components"), (list, tuple)) and plan.returncode in (0, 1),
          {"exit": plan.returncode, "keys": sorted(pdata)[:10]})


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
    case_feasibility_cli()
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
