"""自检：**环境与兼容性落盘事实**不许悄悄过期（E0.1 / E0.2）✓✗。

这三份文件（`docs/env/hardware-report.json` / `docs/env/runtime-versions.json` /
`docs/compatibility-matrix.md`）在代码里**没有读者** ✓✗ —— 实测抓过一次：
旧 `hardware-report.json` 停在 `schemaVersion: 1`，把 `transformers` 记成「可选缺失」、
把 `convrot` 记成「未实现」（其实早已实现，两个 19.53 GiB 主 DiT 就是它）、
`modelsDir` 还指向已经不用的 `data/models` ✓✗。没有判据的文档一定会烂 ✓。

所以本套干四件事：

1. **能力段逐项对账**：文件里的 `schemaVersion` / `stages` / `operations` / `unverified` /
   `backend` / `quant` / `catalog` / `limits` / `knownGaps` == 现场 `facade.capabilities()` ✓；
2. **版本段逐项对账**：`runtime-versions.json` 的 python / torch / cuda / cuDNN / 驱动 / GPU /
   依赖版本 == 本机实测 ✓（换机器、升级 torch 都会红 ✓）；
3. **文档不许自相矛盾**：`compatibility-matrix.md` 里写的那些版本字面量必须就是实测值 ✓；
4. **文档表格不许漏掉已存在的事实**（2026-10-31 补，四条洞全出在这里 ✓✗）：
   `quant.SUPPORTED_FORMATS` 每一族在 §三 都得有「支持」行（**fp8 当时一行都没有** ✗）；
   §二 每个包的版本 == `runtime-versions.json::packages`（**numpy / safetensors / pillow /
   transformers 四个版本原先没有任何判据** ✗）；§四 的显存数字 == `device.memoryTotalMiB`
   （**这个字段从来没人核** ✗）。病根与 `structure_test` 的 ㊆/㊇ 同形：
   判据只核「**点过名的**必须对」，不核「**存在的**必须写到」✗；
5. **口径不许倒退**：GPU 可用时，文档里不许还留着"CPU 轮子 / 无 GPU"的说法
   （那是 E0.3 未裁定时的旧事实 ✓✗）。

⚠️ 判据 3 盯的是**字面量**而不是语义 —— 文档可以改措辞，但不能改数字 ✓。
⚠️ 判据 4 每节都用 :func:`_matrix_section` 切段再扫 ✓：`fp8` / `torch` 在 §一、§四 也有行，
全文匹配会把「GPU 有 fp8 语义」误当成「量化表写了 fp8」✓✗。

运行::

    ./.venv/Scripts/python.exe tests/engine_env_docs_test.py
"""
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import re as _re
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_PY.parent
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

import torch  # noqa: E402  —— 版本/显卡事实只能实测 ✓

from app.services.engine import facade, inventory, quant  # noqa: E402

ENV_DIR = REPO_ROOT / "docs" / "env"
HARDWARE = ENV_DIR / "hardware-report.json"
VERSIONS = ENV_DIR / "runtime-versions.json"
MATRIX = REPO_ROOT / "docs" / "compatibility-matrix.md"

#: 代码里的 format token → §三 表里**称呼它的写法**（多写法命中任一即可 ✓）。
#:
#: 这张表是判据的输入，不是文档的副本 ✓：代码新增一族却没登记 ⇒ 判据报「没登记文档写法」✓，
#: 逼人先想清「读者会按哪个词去查它」✗ —— 这正是漏写 fp8 那次的病根。
DOC_TOKENS: dict[str, tuple[str, ...]] = {
    "nvfp4": ("nvfp4",),
    "convrot_w4a4": ("w4a4", "convrot_w4a4"),
    "float8_e4m3fn": ("fp8",),
    "float8_e5m2": ("fp8",),
}

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def undocumented_families(section: str,
                          catalog: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """§三 表里**没有「支持」行**的已支持族 ✓（纯函数 ⇒ 可反验 ✓）。

    「有一行写着支持」就算覆盖 ✓，要求**那行同时不含「不支持」**✗ —— 因为
    文档里讲「不支持」的行也常顺带点名受支持的族（§三 最后一行就拿 `convrot_w4a4` 做对比 ✓）。
    """
    tokens = DOC_TOKENS if catalog is None else catalog
    return sorted({
        token for token, spellings in tokens.items()
        if token in quant.SUPPORTED_FORMATS
        and not any(line.startswith("|") and "支持" in line and "不支持" not in line
                    and any(spelling in line for spelling in spellings)
                    for line in section.splitlines())})


def package_doc_problems(section: str, packages: dict[str, str]) -> dict[str, object]:
    """§二「依赖 × 实测版本」表 ⇄ ``runtime-versions.json::packages`` ✓（纯函数 ⇒ 可反验 ✓）。

    返回 ``{"缺行": [...], "写错": {包: (文档, 实测)}}``，都空才算对 ✓。

    ⚠️ 为什么非得有这条：③ 那条只用 ``must`` 盯了 **torch / Python / CUDA / cuDNN** ✓ ——
    2026-10-31 实测 §二 里 `numpy` / `safetensors` / `pillow` / `transformers` **四个版本
    没有任何判据** ✗。升级 numpy 之后表里还是旧值，读者照着表去装就装成另一个组合 ✓✗。
    复合行（`fastapi / uvicorn / …`）不逐个核 ✓：它的版本真源本就在 `THIRD_PARTY_NOTICES.md` ✓。
    """
    documented: dict[str, str] = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        name = cells[0].strip("`").strip()
        if name in packages:
            documented[name] = cells[1].strip("`").strip()
    missing = sorted(set(packages) - set(documented))
    wrong = {name: (documented[name], real) for name, real in packages.items()
             if name in documented and documented[name] != real}
    return {"缺行": missing, "写错": wrong}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _matrix_section(text: str, heading: str) -> str:
    """取 ``## <heading>`` 到下一个二级标题之间的那段 ✓（判据只认本节内的表格行 ✓）。

    ⚠️ 不许全文扫 ✗：``fp8`` 在 §一、§四 也有行 ✓，全文匹配会把「GPU 有 fp8 语义」
    误当成「量化族表里写了 fp8」✓✗。
    """
    start = text.find(heading)
    if start < 0:
        return ""
    nxt = text.find("\n## ", start + len(heading))
    return text[start:] if nxt < 0 else text[start:nxt]


def case_hardware_report() -> None:
    """① 能力段：文件 == 现场 `capabilities()` ✓。"""
    check("① 两份落盘事实都在盘上 ✓（缺一份＝E0.1 没交付 ✓✗）",
          HARDWARE.is_file() and VERSIONS.is_file() and MATRIX.is_file(),
          [str(p.name) for p in (HARDWARE, VERSIONS, MATRIX) if not p.is_file()])

    saved = _load(HARDWARE)
    live = facade.capabilities()

    check("① E0.1 硬判据：`device` / `python` / `catalog` 三栏非空 ✓✗",
          bool(_load(VERSIONS).get("device")) and bool(live.get("python"))
          and bool((saved.get("catalog") or {}).get("models")),
          {"runtimeVersionsDevice": bool(_load(VERSIONS).get("device")),
           "python": live.get("python"), "catalogModels": (saved.get("catalog") or {}).get("models")})

    stable = ("schemaVersion", "apiVersion", "engine", "python", "stages", "operations",
              "unverified", "devicePolicy", "limits", "knownGaps")
    drift = {key: {"文件": saved.get(key), "现场": live.get(key)}
             for key in stable if saved.get(key) != live.get(key)}
    check("① `hardware-report.json` 的能力段与**现场**一致 ✓✗（`schemaVersion` 就曾经停在 1 ⇒ "
          "读它的人会按旧契约理解今天的行为 ✓✗）",
          not drift, drift)

    for section, keys in (("backend", ("available", "missingRequired", "optionalMissing")),
                          ("quant", ("supportedLayouts", "unsupportedFamilies", "unsupportedFormats")),
                          ("catalog", ("models", "modelsDir"))):
        got = {key: (saved.get(section) or {}).get(key) for key in keys}
        want = {key: (live.get(section) or {}).get(key) for key in keys}
        check(f"① `{section}` 段逐项一致 ✓✗（实测踩过：文件里把 `convrot` 报成未实现、"
              f"把 `transformers` 报成可选缺失 ✓✗）", got == want, {"文件": got, "现场": want})

    #: ⚠️ 设备策略这一栏**必须答全**，否则等于没接 ✗✗：只报各条的 `device`（"这次在 CPU"）
    #: 会让读的人把**一次降级**读成**能力缺失** ✓✗（本仓 2026-10-31 之前就是这样 ✓）。
    #: 三件事缺一不可 ✓：① 四档策略名与默认档 ✓；② 判定规则（凭什么从卡挪到 CPU ✓）；
    #: ③ **两本账**（分时 / 整链同驻 ✓ —— 混一起报会按最重的单件估 ⇒ 差 4.5 GiB OOM ✓✗）。
    dp = live.get("devicePolicy") or {}
    need = dp.get("needGiB") or {}
    check("①′ 设备策略栏答全四档 + 判定规则 + **三本账** ✓✗"
          "（只有结论没有规则 ⇒ 一次降级被读成能力缺失；只有一本账 ⇒ 按单件估会 OOM）",
          list(dp.get("policies") or []) == ["auto", "gpu", "cpu", "balanced"]
          and dp.get("default") == "auto"
          and len(dp.get("rules") or []) >= 3
          and set(need) == {"part", "chain", "qwenImagePart"}
          and abs(float((need.get("chain") or {}).get("zimage") or 0) - 19.28) < 0.01
          and float((need.get("part") or {}).get("dit") or 0) > 0
          #: ⚠️ **第三本账**必须真报出来 ✓✗（2026-10-31 补 ✓）：2512 的 DiT 38.05 GiB 与 2.1 的
          #: 13.26 差 24.8 GiB ✓ ⇒ 只报前者会让"这台机器能跑 2512"变成想当然 ✓（实际必然 split/CPU）。
          and abs(float((need.get("qwenImagePart") or {}).get("dit") or 0) - 38.05) < 0.01
          and float((need.get("qwenImagePart") or {}).get("dit") or 0)
          > float((need.get("part") or {}).get("dit") or 0), dp)

    #: ⚠️ `capabilities()` 的契约是「零副作用、不 import torch、毫秒级」✓✗ ⇒ 这一栏**不许**
    #: 带当时的卡况读数（那是 `GET /engine/runtime` 的事 ✓）。带了就说明有人为了凑字段
    #: 在这里 probe 显存 ✓✗ —— 一个本该恒定的能力表就变成了随机器负载变化的快照 ✓✗，
    #: 而 `hardware-report.json` 是落盘事实 ⇒ 每次跑测试都得跟着改 ✓✗。
    check("①″ 能力表**只报静态事实** ✓✗：`devicePolicy` 里不许出现当时的卡况（`pressure` / "
          "`freeGiB` / `lastDecision` 属运行时那面 ✓）",
          not any(key in dp for key in ("pressure", "freeGiB", "lastDecision", "phases")),
          sorted(dp))


def case_runtime_versions() -> None:
    """② 版本段：文件 == 本机实测 ✓。"""
    saved = _load(VERSIONS)
    catalog = inventory.load_catalog()["models"]

    check("② `python` == 实测 ✓", str(saved.get("python")) == platform.python_version(),
          {"文件": saved.get("python"), "实测": platform.python_version()})

    packages = saved.get("packages") or {}
    drift = {}
    #: ⚠️ 遍历 `packages` 的**全部键** ✓，不按手抄的五个名字走 ✗ —— 旧写法把
    #: `("torch", "numpy", "safetensors", "pillow", "transformers")` 抄死在这里，
    #: 采集端**多记一个包**（比如以后加 `onnxruntime`）这条判据就完全不看它 ✓✗（同 ㊆/㊇ 那个病）。
    for name in sorted(packages):
        try:
            actual = torch.__version__ if name == "torch" else metadata.version(name)
        except metadata.PackageNotFoundError:
            drift[name] = {"文件": packages.get(name), "实测": "本机未安装"}
            continue
        if str(packages.get(name)) != actual:
            drift[name] = {"文件": packages.get(name), "实测": actual}
    check("② 依赖版本逐项 == 实测 ✓✗（升级依赖 / 换机器后必须刷新本文件 ✓）", not drift, drift)
    check("② `packages` 不许为空 ✓（空表会让上一条与 ②′ 一起空转成「全绿」✓✗）",
          bool(packages), packages)

    torch_saved = saved.get("torch") or {}
    check("② `torch.cudaVersion` == 实测 ✓（为 null 时按 E0.1 必须同时写 `knownGaps` ✓✗ —— "
          "这一栏空着就等于「不知道 CUDA 能不能用」，而文档读者会当成「能用」✓✗）",
          str(torch_saved.get("cudaVersion")) == str(torch.version.cuda)
          and (torch_saved.get("cudaVersion") is not None or bool(saved.get("knownGaps"))),
          {"文件": torch_saved.get("cudaVersion"), "实测": torch.version.cuda})

    device = saved.get("device") or {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        check("② GPU 可用 ⇒ `device.name` / `capability` / `driverVersion` 都得写实测值 ✓✗",
              device.get("name") == torch.cuda.get_device_name(0)
              and device.get("capability") == ".".join(map(str, torch.cuda.get_device_capability(0)))
              and bool(device.get("driverVersion")),
              device)
        #: ⚠️ 上面那条**一直没核显存** ✗（2026-10-31 实测）—— 而 `memoryTotalMiB` 正是
        #: 矩阵 §四 那两个数字（`23028 MiB 标称` / `22.49 GiB`）的来源 ✓。ECC 开关、
        #: 换卡、驱动变化都会改它，落盘却没人对 ⇒ 文档会永远写着旧容量 ✓✗。
        #: ⚠️ 拦 **1 MiB 容差**：实测 `mem_get_info().total` = 24146083840 B = **23027.34 MiB**
        #: ⇒ torch 截断写 23027 ✓、`nvidia-smi` 显示 **23028** ✓ —— 手写栏取哪个都对，
        #: 差 1 MiB 是**取整**不是**过期** ✗（换卡是 GB 级差，1 MiB 容差拦不住任何真问题 ✓）。
        free_bytes, total_bytes = torch.cuda.mem_get_info()   # 与 `runtime.py` 同一把尺子 ✓
        actual_mib = total_bytes // (1024 * 1024)
        saved_mib = device.get("memoryTotalMiB")
        check("② `device.memoryTotalMiB` == `mem_get_info().total`（容差 1 MiB 取整差）✓✗",
              isinstance(saved_mib, int) and abs(saved_mib - actual_mib) <= 1,
              {"文件": saved_mib, "实测MiB": actual_mib, "实测字节": total_bytes})
    check("② `catalog.models` == 清单条数 ✓（清单加一件就红 ✓）",
          (saved.get("catalog") or {}).get("models") == len(catalog),
          {"文件": (saved.get("catalog") or {}).get("models"), "清单": len(catalog)})


def case_matrix() -> None:
    """③ 文档里的版本字面量 == 实测 ✓；④ GPU 可用时不许留 CPU 轮子旧说法 ✓。"""
    text = MATRIX.read_text(encoding="utf-8")
    saved = _load(VERSIONS)

    must = {
        "Python 版本": platform.python_version(),
        "torch 版本": torch.__version__,
        "torch 的 CUDA": str(torch.version.cuda),
        "cuDNN 版本": "%d.%d.%d" % (torch.backends.cudnn.version() // 10000,
                                    torch.backends.cudnn.version() // 100 % 100,
                                    torch.backends.cudnn.version() % 100),
    }
    missing = {label: value for label, value in must.items() if value not in text}
    check("③ 兼容性矩阵里写的是**实测版本**而不是旧值 ✓✗（换机、升级 torch 就红 ✓）",
          not missing, missing)

    check("③ `pip check` 的结论在文档里，且与落盘事实一致 ✓✗（写「无冲突」而实际有冲突，比不写更坏 ✓）",
          "No broken requirements found." in text
          and "No broken requirements found." in str(saved.get("pipCheck")),
          saved.get("pipCheck"))

    check("③ 量化：`mxfp4` 在文档里被明确写成**不支持** ✓✗（含糊其辞⇒读的人会去试，白烧一次装载 ✓）",
          "mxfp4" in text
          and "mxfp4" in (saved.get("knownGaps") and json.dumps(saved, ensure_ascii=False) or ""),
          "mxfp4")
    #: ⚠️ `nvfp4` **2026-10-30 起已实现** ✓ ⇒ 这条判据必须**反过来** ✗（原文写「必须写成不支持」✓✗）：
    #: 还写成不支持，读的人就会去换权重 / 认定 H3 跑不了 —— 比不写更坏 ✓✗。
    nv_supported = [line for line in text.splitlines()
                    if "nvfp4" in line and "支持" in line and "不支持" not in line and "作废" not in line]
    report = _load(HARDWARE).get("quant") or {}
    check("③ 量化：`nvfp4` **已实现** ⇒ 文档里必须有一句写明它**支持** ✓，且报告的 "
          "`unsupportedFormats` 里**不许**再有它 ✗",
          bool(nv_supported) and "nvfp4" not in (report.get("unsupportedFormats") or [])
          and "nvfp4" in (report.get("supportedFormats") or []),
          {"命中行": nv_supported[:1], "报告": report})

    #: ⚠️ 上面几条只盯**文档**与**报告** ✓ —— 可 `runtime-versions.json` 的 `knownGaps` 是**手写维护**的
    #: （它不是 `facade.capabilities()` 的产物 ✗）⇒ 没有判据就一定停在旧事实上 ✓✗（2026-10-30 实测正是
    #: 如此：报告里 `supportedFormats=[nvfp4]` ✓，同一份 env 的 `knownGaps` 却写着「mxfp4 / nvfp4 未实现」
    #: ✓✗ —— 两句话在同一处打架，读者只会信到错的那句 ✓）。
    #: 所以这里做**双向**对账：已支持的族不许被说成「未实现」✗；未支持的族**必须**被点名 ✓
    #: （漏点名＝读者不知道它会被拒 ✓✗）。
    gaps = saved.get("knownGaps") or []
    supported = report.get("supportedFormats") or []
    unsupported = report.get("unsupportedFormats") or []
    #: ⚠️ 判「未实现」必须按**子句**判 ✗，不能按整条 `summary` 判 ✓✗ —— 一条 knownGap 里常常同时讲
    #: 两个族（「mxfp4 未实现 ⇒ … **；** nvfp4 已实现」✓），整条匹配就会把「mxfp4 的未实现」算到
    #: nvfp4 头上 ✓✗（本次实测：第一版就是这么误报的 ✓）。
    def _clauses(text: str) -> list[str]:
        return [part for part in _re.split(r"[；;。\n]", str(text)) if part.strip()]

    stale_gap = [f"{gap.get('id')}:{fmt}" for gap in gaps for fmt in supported
                 for clause in _clauses(gap.get("summary"))
                 if fmt in clause and "未实现" in clause]
    silent_gap = [fmt for fmt in unsupported
                  if not any(fmt in clause for gap in gaps for clause in _clauses(gap.get("summary")))]
    check("③′ `knownGaps` 与 `quant` 报告**双向**一致 ✓✗（已支持的族不许写成「未实现」✗；"
          "未支持的族必须点名 ✓）",
          not stale_gap and not silent_gap,
          {"已支持却被写成未实现": stale_gap, "未支持却没点名": silent_gap,
           "supported": supported, "unsupported": unsupported})

    #: ⚠️ 上面两条只盯**被点名的那两族**（nvfp4 / mxfp4）✓✗ —— 代码新加一族、文档漏写，
    #: 没有任何一行会红 ✓✗。2026-10-31 实测正是如此：`quant.SUPPORTED_FORMATS` 有 4 个键，
    #: §三 表里**没有 fp8 的行** ⇒ 查「fp8 到底支不支持」的人在主题表里找不到答案，
    #: 只能去 §四 读那句「有 fp8 语义、无 fp8 硬件加速」自己猜 ✓。
    #: 所以这里逐族对账：`SUPPORTED_FORMATS` 每一族都必须在 §三 有一行写着「支持」✓。
    section = _matrix_section(text, "## 三、")
    unmapped = [tok for tok in quant.SUPPORTED_FORMATS if tok not in DOC_TOKENS]
    stale_tokens = [tok for tok in DOC_TOKENS if tok not in quant.SUPPORTED_FORMATS]
    undocumented = undocumented_families(section)
    check("③″ `quant.SUPPORTED_FORMATS` 的**每一族**在 §三 表里都有「支持」行 ✓✗"
          "（代码支持、表里没行 ⇒ 读者只能猜 ✓✗）",
          not unmapped and not undocumented,
          {"没登记文档写法": unmapped, "表里缺支持行": undocumented, "节": bool(section)})
    check("③″ `DOC_TOKENS` 自己也没有过期条目（某族已不再支持 ⇒ 该删这行，"
          "否则它会永久「覆盖」一个不存在的族 ✓✗）",
          not stale_tokens, stale_tokens)
    fp8_lines = [line for line in section.splitlines() if "fp8" in line]
    check("③″ 反验：抹掉表里那行 fp8 ⇒ 必须报两族（判据不是照着当前文档写的 ✓）",
          undocumented_families(section.replace(fp8_lines[0] + "\n", "", 1)
                                if fp8_lines else section)
          == ["float8_e4m3fn", "float8_e5m2"], fp8_lines[:1])
    check("③″ 反验：抹掉 4 bit 打包那行 ⇒ 只报 `convrot_w4a4`（末行那句对比"
          "「不支持」不许替它顶包 ✓）",
          undocumented_families(
              "\n".join(line for line in section.splitlines() if "w4a4" not in line
                        or "不支持" in line))
          == ["convrot_w4a4"], None)

    #: ⚠️ ③ 只盯 torch / Python / CUDA / cuDNN 四个字面量 ✓ —— §二 那张「依赖 × 实测版本」
    #: 表里另外 **五个包版本没有任何判据** ✗（2026-10-31 实测）⇒ 升级 numpy 之后表里还是旧值，
    #: 读者照表去装就装成另一个组合 ✓✗。这条把表 ⇄ `packages` 逐项对死 ✓。
    deps = _matrix_section(text, "## 二、")
    pkg = package_doc_problems(deps, saved.get("packages") or {})
    check("②′ §二 依赖表的版本 == `runtime-versions.json::packages` 实测 ✓✗"
          "（升级依赖不出声 ⇒ 表里留旧值 ✓✗）",
          bool(deps) and not pkg["缺行"] and not pkg["写错"],
          {**pkg, "节": bool(deps)})
    #: 反验两条：判据不是照着当前这份表写的 ✓。
    stale = dict(saved.get("packages") or {})
    nv = stale.get("numpy") or ""
    check("②′ 反验：把 `numpy` 版本改成旧值 ⇒ 必须报为「写错」✓",
          "numpy" in package_doc_problems(
              deps.replace(f"| {nv} |", "| 0.0.0 |", 1) if nv else deps,
              stale).get("写错", {}), None)
    check("②′ 反验：抹掉 `pillow` 那行 ⇒ 必须报为「缺行」✓",
          "pillow" in package_doc_problems(
              "\n".join(line for line in deps.splitlines() if "| pillow |" not in line),
              {"pillow": "13.0.0"}).get("缺行", []), None)

    #: §四 那两个显存数字（`23028 MiB 标称` / `22.49 GiB`）也是**手抄**的 ✓，来源是
    #: `device.memoryTotalMiB` ✓ —— ② 现在核了落盘值 == 现场（上一条），这里核**文档也抄对了** ✓。
    #: ⚠️ 只核这两个 ✓：同节里的 `49.44 GiB` 是 `readiness("h3").vram` 的现场估算，
    #: 跑它要读 39 GiB 权重的元数据 ⇒ 由 `engine_readiness_script_test` 管 ✓，不在这里拖时间 ✗。
    section_gpu = _matrix_section(text, "## 四、")
    mib = (saved.get("device") or {}).get("memoryTotalMiB")
    gib = f"{mib / 1024:.2f}" if isinstance(mib, int) else ""
    documented_gpu = (isinstance(mib, int) and bool(section_gpu)
                      and str(mib) in section_gpu and gib in section_gpu)
    check("④′ §四 的显存数字 == `device.memoryTotalMiB`（整 MiB ＋ GiB 两位小数）✓✗"
          "（换卡/ECC 变化 ⇒ 文档不许还写旧容量 ✓✗）",
          documented_gpu, {"MiB": mib, "GiB": gib, "节": bool(section_gpu)})
    check("④′ 反验：把 §四 里的 MiB 数字改掉 ⇒ 必须报 ✓",
          bool(gib) and not (str(mib) in section_gpu.replace(str(mib), "99999", 1)
                             and gib in section_gpu), None)

    if torch.cuda.is_available():
        # ⚠️ 只扫「没标作废」的行 ✓：把旧结论**明确写成已作废**是允许的 ✓（甚至该鼓励 ✓），
        # 不许的是**还当着事实说** ✓✗。所以判据要先滤掉带「作废」的行，再扫词 ✓。
        live_lines = [line for line in text.splitlines() if "作废" not in line]
        stale = [phrase for phrase in ("CPU 轮子", "只能走 CPU", "无 GPU 加速", "仅 CPU")
                 if any(phrase in line for line in live_lines)]
        check("④ GPU **实测可用** ⇒ 文档里不许还当着事实说「CPU 轮子 / 无 GPU」✓✗"
              "（旧结论可以留着，但必须标「已作废」✓ —— 不标的话读者分不清哪句是今天的事实 ✓✗）",
              not stale, stale)
        check("④ `runtime-versions.json` 的 `torch.cudaAvailable` 也必须是 true ✓",
              (saved.get("torch") or {}).get("cudaAvailable") is True,
              saved.get("torch"))


def main() -> int:
    case_hardware_report()
    case_runtime_versions()
    case_matrix()

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
