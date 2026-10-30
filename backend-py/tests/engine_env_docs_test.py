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
4. **口径不许倒退**：GPU 可用时，文档里不许还留着"CPU 轮子 / 无 GPU"的说法
   （那是 E0.3 未裁定时的旧事实 ✓✗）。

⚠️ 判据 3 盯的是**字面量**而不是语义 —— 文档可以改措辞，但不能改数字 ✓。

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

from app.services.engine import facade, inventory  # noqa: E402

ENV_DIR = REPO_ROOT / "docs" / "env"
HARDWARE = ENV_DIR / "hardware-report.json"
VERSIONS = ENV_DIR / "runtime-versions.json"
MATRIX = REPO_ROOT / "docs" / "compatibility-matrix.md"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
              "unverified", "limits", "knownGaps")
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


def case_runtime_versions() -> None:
    """② 版本段：文件 == 本机实测 ✓。"""
    saved = _load(VERSIONS)
    catalog = inventory.load_catalog()["models"]

    check("② `python` == 实测 ✓", str(saved.get("python")) == platform.python_version(),
          {"文件": saved.get("python"), "实测": platform.python_version()})

    packages = saved.get("packages") or {}
    drift = {}
    for name in ("torch", "numpy", "safetensors", "pillow", "transformers"):
        actual = torch.__version__ if name == "torch" else metadata.version(name)
        if str(packages.get(name)) != actual:
            drift[name] = {"文件": packages.get(name), "实测": actual}
    check("② 依赖版本逐项 == 实测 ✓✗（升级依赖 / 换机器后必须刷新本文件 ✓）", not drift, drift)

    torch_saved = saved.get("torch") or {}
    check("② `torch.cudaVersion` == 实测 ✓（为 null 时按 E0.1 必须同时写 `knownGaps` ✓✗ —— "
          "这一栏空着就等于「不知道 CUDA 能不能用」，而文档读者会当成「能用」✓✗）",
          str(torch_saved.get("cudaVersion")) == str(torch.version.cuda)
          and (torch_saved.get("cudaVersion") is not None or bool(saved.get("knownGaps"))),
          {"文件": torch_saved.get("cudaVersion"), "实测": torch.version.cuda})

    device = saved.get("device") or {}
    if torch.cuda.is_available():
        check("② GPU 可用 ⇒ `device.name` / `capability` / `driverVersion` 都得写实测值 ✓✗",
              device.get("name") == torch.cuda.get_device_name(0)
              and device.get("capability") == ".".join(map(str, torch.cuda.get_device_capability(0)))
              and bool(device.get("driverVersion")),
              device)
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
