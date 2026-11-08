"""引擎产物**兜底落点** —— 唯一真源（2026-11-07 ✓）。

## 它治的是什么

调用方没给 `outputs_dir` 时（自检 / `_bench/` 探针 / 直接 `pipeline.run_sync` ✓），
四处入口各自 `tempfile.mkdtemp(prefix="engine_*")` ✓✗ ⇒ **每跑一次就多一个新目录** ✓：
图落在 `%TEMP%\\engine_torch_xxxx` ✓、机器一清理连凭据一起没 ✓，想留就得人工搬 ✓✗。

现在一律落到一个**稳定目录** ✓：

=======================================  ==============================================
落点                                     什么时候
=======================================  ==============================================
`<storageRoot>/engine/<kind>/`           出图 / 出片 / 出音频（`/static` 服务得到 ✓）
`<dataRoot>/engine-dryrun/<stage>/`      干跑报告（**不进** `static` ✓，见下）
=======================================  ==============================================

`DARME_ENGINE_OUTPUTS` 可整棵改根 ✓（探针 / 测试隔离用 ✓）。

## 三条纪律

1. **只兜底** ✓：调用方给了 `outputs_dir` 就一字不改 ✗。
2. **不编时间戳** ✗：同 seed 重跑仍覆盖同名文件 ✓ —— 文件名是回执的一部分 ✓，
   编时间戳会让「同一请求」不再可复现 ✗（要多份就由调用方给不同目录 ✓）。
3. **不 import `bridge`** ✗：`bridge` 反向依赖 `runtime` ✓，从 `torch_backend` 引它会成环 ✗；
   本模块只依赖 :mod:`app.core.config` ✓（那边只读环境变量与 yaml ✓）。

干跑为什么另起一棵树 ✗：干跑报告是**调试凭据** ✓、不是产物 ✗ —— 落到 `static` 之下就会被
`/static` 服务出去 ✓，前端/用户拿它当"成品链接"点开（里面是 JSON ✓✗）。与
:func:`bridge.outputs_dir_for` 的 `dry_run` 分支同一口径 ✓。
"""
from __future__ import annotations

import os
from pathlib import Path

from ...core.config import get_data_root, get_storage_root

__all__ = [
    "ENV_OVERRIDE",
    "ENV_DRYRUN_OVERRIDE",
    "DEFAULT_KIND",
    "outputs_root",
    "dryrun_root",
    "fallback_outputs_dir",
    "report_dir",
    "is_fallback",
]

#: 整棵改根 ✓（只顶替 `<storageRoot>/engine` ✓，干跑那棵仍跟数据根走 ✓）
ENV_OVERRIDE = "DARME_ENGINE_OUTPUTS"

#: 干跑报告树整棵改根 ✓（测试要隔离它时用 ✓ —— 否则 `--dry-run` 会往**真**数据根写 ✓）
ENV_DRYRUN_OVERRIDE = "DARME_ENGINE_DRYRUN"

DEFAULT_KIND = "misc"


def _safe_kind(kind: str | None) -> str:
    """用途名 ⇒ 单层目录名 ✓（去分隔符、去 `..` ✗ —— 这条会拼进路径 ✓）"""
    text = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in str(kind or "").strip())
    text = text.strip("-.")
    return text or DEFAULT_KIND


def outputs_root() -> Path:
    """产物树的根：`DARME_ENGINE_OUTPUTS` > `<storageRoot>/engine` ✓（**不建目录** ✗）"""
    override = (os.environ.get(ENV_OVERRIDE) or "").strip()
    if override:
        return Path(override).expanduser()
    return Path(get_storage_root()) / "engine"


def dryrun_root() -> Path:
    """干跑报告的根：`DARME_ENGINE_DRYRUN` > `<dataRoot>/engine-dryrun` ✓（**不建目录** ✗）"""
    override = (os.environ.get(ENV_DRYRUN_OVERRIDE) or "").strip()
    if override:
        return Path(override).expanduser()
    return Path(get_data_root()) / "engine-dryrun"


def _under(path: str | Path, root: str | Path) -> bool:
    """``path`` 是否落在 ``root`` 之下 ✓（解析失败一律 False ✗ —— 宁可退回稳定名 ✓）"""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False
    return True


def report_dir(outputs_dir: str | Path | None = None, stage: str | None = None) -> Path:
    """**干跑报告**的落点 ✓ —— 永远在 :func:`dryrun_root` 之下 ✓✗，调用方给的目录**只借最后一层名字** ✓。

    ## 为什么连调用方给的 `outputs_dir` 也不照落

    业务通路给的落点来自 :func:`bridge.outputs_dir_for` 的**非** `dry_run` 支 ✓ ⇒
    `<storageRoot>/engine/<kind>-<id>` ✓。报告要"照用"它，就等于 **干跑报告住进 `static`** ✓✗：
    `/static` 会把它服务成一个能点开的链接 ✓，而里面是 JSON（不是画面 ✗）。

    2026-11-07 实测就是这样漏的 ✓：:func:`bridge.outputs_dir_for` 已经补了 `dry_run` 分支 ✓、
    兜底也走对了 ✓，但本函数的前身（`dryrun.write` 里的三元表达式）**先看调用方给没给** ✓ ⇒
    盘上 `data/static/engine/dryrun_seed0.json` 与
    `data/static/engine/video-eng-{436aead6d1f8,d5e25e9ffae8}/dryrun_seed3.json` 四个报告
    一直躺在 `static` 里 ✓✗。

    ## 借名字的前提：它得是**本仓落点**

    借名字是为了保住 `<kind>-<id>` 那层 ✓：同 seed 连跑不互相覆盖 ✓，且与
    :func:`bridge.outputs_dir_for` 的 `dry_run` 支**路径完全重合** ✓（两条路进同一个目录 ✓）。

    ⚠️ 但**不是谁的目录都借** ✓✗（2026-11-07 第二轮修 ✓）：探针与测试常给
    `tempfile.mkdtemp(prefix="engine_pipe_")` ✓ ⇒ 借来的名字是**一次性随机串** ✓
    ⇒ 每跑一次就在干跑树里多一个空壳 ✓。实测盘上积了 **29 个
    `engine-dryrun/engine_pipe_xxxx/`**（全部由 `tests/engine_pipeline_test.py` 一次跑出 ✓，
    因为全仓**没有任何地方**设 `DARME_ENGINE_DRYRUN` ✓✗ —— 那个环境变量当时是死配置 ✓）。

    ⇒ 现在只认**落在两棵树的根下面**的 `outputs_dir` ✓：业务通路
    `<storageRoot>/engine/<kind>-<id>` ✓、干跑支 `<dataRoot>/engine-dryrun/<kind>-<id>` ✓
    都稳定、可复跑 ✓；临时目录一律不借 ✓，退回 `stage`（`krea2` / `h3` / `video` ✓）。
    """
    name = ""
    if outputs_dir:
        candidate = Path(str(outputs_dir))
        if _under(candidate, outputs_root()) or _under(candidate, dryrun_root()):
            name = _safe_kind(candidate.name)
    if not name:
        name = _safe_kind(stage or "dryrun")
    target = dryrun_root() / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def fallback_outputs_dir(kind: str | None = DEFAULT_KIND, *, dry_run: bool = False,
                         create: bool = True) -> Path:
    """没给 `outputs_dir` 时的落点 ✓ —— 稳定、可重复、**不再每次换新目录** ✓。

    :param kind: 用途名 ✓（`image` / `video` / `audio` / `chain` / `sdxl` / 阶段名 ✓）
    :param dry_run: 落 `<dataRoot>/engine-dryrun/<kind>/` ✓（见模块说明）
    :param create: 默认替调用方建目录 ✓（每个调用点后面本来就要 `mkdir` ✓）
    """
    name = _safe_kind(kind)
    root = dryrun_root() if dry_run else outputs_root()
    target = root / name
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def is_fallback(path: str | Path) -> bool:
    """这个路径是不是本模块给的（用来在回执里标「兜底落点」✓、不假装是调用方指定的 ✗）"""
    try:
        target = Path(path).resolve()
    except (OSError, ValueError):
        return False
    for root in (outputs_root(), dryrun_root()):
        try:
            target.relative_to(Path(root).resolve())
        except (OSError, ValueError):
            continue
        return True
    return False
