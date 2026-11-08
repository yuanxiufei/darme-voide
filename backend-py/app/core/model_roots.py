"""「默认扫描根」的**唯一一份**拼装（2026-11-08 从 ``services/local_model_scan.py`` 搬 ✓）。

## 它回答什么

「该去哪些目录找模型」✓ ⇒ 去重保序的**绝对路径**列表 ✓。五类来源（顺序即优先级 ✓）：

1. ``models_dir`` ✓ —— 存放/下载主目录（``MODELS_DIR`` > ``model-paths.json`` > ``<dataRoot>/models``
   ✓；优先级**不在这里写** ✗，走 :func:`app.core.model_paths.models_dir` 那唯一一份 ✓）；
2. ``local_services_dir`` ✓ —— 本仓自带的本地服务落点（``LOCAL_SERVICES_DIR`` / json / 默认 ✓）；
3. ``extra_roots`` ✓ —— 用户**显式**加的任意目录（显式意图优先于约定 ✓）；
4. **用户声明过的任意目录** ✓ —— ComfyUI 的 ``extra_model_paths.yaml`` / ``extra_paths`` /
   ``EXTRA_MODEL_PATHS``（见 :mod:`app.core.model_folders` ✓）；
5. **各生态落点** ✓ —— HuggingFace / ModelScope / LM Studio / GPT4All / Jan / llama.cpp /
   text-generation-webui / SD WebUI / Stability Matrix / InvokeAI / torch hub
   （见 :mod:`app.core.model_ecosystems` ✓；⚠️ 排在 ComfyUI **之前** ✗：ComfyUI 的 ``models/``
   动辄上万文件 ⇒ 后置会被 ``maxFiles`` 上限吃光 ⇒ 生态落点「扫不到也不报错」✗✗）；
6. 每个 ComfyUI 类目录的 ``models/`` ✓（**全部**命中 ✗ 不是第一个 ✓ —— 见
   :mod:`app.core.comfyui_roots` ✓），最后把**显式配的那个根**整体也加上 ✓。

## 为什么它住 ``app/core``（2026-11-08 搬 ✓）

引擎的 ``inventory.candidate_roots()`` 要「清单里那份权重在盘上还有没有别的副本」⇒ 要这份根表 ✓，
而引擎顶层只许 import ``app.core.*`` ✗（``tests/engine_independence_test.py`` 逐条判 ✓）。
上述六类来源读的**全是配置与盘**（用户声明、生态约定、盘符枚举 ✓），不碰业务数据 ✓
⇒ 与 ``app.core.model_paths`` / ``comfyui_roots`` / ``model_folders`` / ``model_ecosystems`` 同一类 ✓。

⚠️ **根表只此一份** ✗✗：扫描层（``local_model_scan.get_default_roots``）与引擎
（``inventory.candidate_roots``）**同调本函数** ✓ —— 各写一份的后果是「扫描能看见、加载看不见」✓✗
（2026-09-26 真机实测过：DiT 躺在 ComfyUI 的 ``diffusion_models/`` 里，扫描看得见、视频看不见 ✓）。

⚠️ 隔离口必须**显式下传** ✗✗（跨模块 monkeypatch 不生效 ✓）：自检靠 patch ``path=``（临时配置）
与 ``comfyui=``（假安装根）注入 ✓；少传一个 ⇒ 本函数走自己的全局 ⇒ 那条判据**静默失效** ✓✗
（2026-11-08 实测：只传 ``path=`` 时「每个 ComfyUI 根都要收进来」当场红 ✓）。
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

from . import comfyui_roots as core_comfyui_roots
from . import config, model_ecosystems, model_folders, model_paths

__all__ = ["LOCAL_SERVICES_ROOT", "default_roots"]

#: 本仓自带的「本地服务」落点 ✓（``model_manager.py`` 会 ``git clone`` 的独立服务 + ``h3`` 薄封装 ✓）。
#: ⚠️ 唯一权威在 :mod:`app.core.config` ✓（``services/local_model_scan.LOCAL_SERVICES_ROOT`` 是再导出的同一个 ✓）。
#: ⚠️ 配置里给了 ``local_services_dir`` 时以配置为准 ✓（见 :func:`default_roots` ✓）——
#: 本常量只是**没人配**时的默认值 ✗ 不是"实际生效值" ✗。
LOCAL_SERVICES_ROOT = config.APP_ROOT / "local_services"


def default_roots(
    *,
    path: Path | None = None,
    comfyui: Callable[[], list[tuple[str, str]]] | None = None,
    extra_roots: Sequence[str] | None = None,
    declared: Sequence[str] | None = None,
    ecosystems: Sequence[tuple[str, str]] | None = None,
) -> list[str]:
    """默认扫描根（去重保序的绝对路径 ✓，**只收盘上真有的** ✓）。见模块头「它回答什么」✓。

    ⚠️ 五个隔离口**全部可注入** ✗✗（自检要能在任何机器上给出确定答案 ✓）：
    ``path=``（配置文件 ✓）、``comfyui=``（安装根表 ✓）、``extra_roots=``（用户显式目录 ✓）、
    ``declared=``（声明表 ✓）、``ecosystems=``（生态落点 ✓）。不传哪个，那个就走本模块自己的全局 ✓
    —— 对**生产**是正确默认 ✓，对**自检**是静默失效 ✓✗（所以调用方必须传全 ✓）。

    参数
    ----
    path
        ``model-paths.json`` 的位置 ✓（``None`` ⇒ 走 :func:`app.core.model_paths.paths_config_file` ✓）。
    comfyui
        返回 ``[(安装根, 来源说明)]`` 的函数 ✓（``None`` ⇒ :func:`app.core.comfyui_roots.comfyui_roots` ✓）。
    extra_roots
        用户显式加的目录（``None`` ⇒ 读配置的 ``extra_roots`` ✓）。
    declared
        声明表读出的目录（``None`` ⇒ :func:`app.core.model_folders.declared_extra_dirs` ✓）。
    ecosystems
        生态落点 ``[(生态 id, 落点)]``（``None`` ⇒ :func:`app.core.model_ecosystems.roots` ✓）。
    """
    cfg = model_paths.read_paths_config(path=path)
    roots: list[str] = []

    #: ⚠️ 优先级**不在这里写** ✓✗ —— 走 ``model_paths.models_dir()`` 那唯一一份 ✓（同一份优先级
    #: 既是「存放主目录」也是本函数的第一根 ⇒ 两处各写一遍必然漂 ✓）。
    #: ⚠️ 必须把 ``path=`` 这个隔离口一起传下去 ✓✗（配置面有三个使用者：读 / 写 / 本函数；
    #: 少传一个，自检里换掉的文件就对它不生效 ⇒ 「换了文件还读到真配置」的假绿 ✓）。
    models_dir = model_paths.models_dir(path=path)
    services_dir = (os.environ.get("LOCAL_SERVICES_DIR") or cfg.get("local_services_dir")
                    or str(LOCAL_SERVICES_ROOT))
    probe = comfyui if comfyui is not None else (
        lambda: core_comfyui_roots.comfyui_roots(path=path))
    # ⚠️ 显式配置（env / json）**优先且靠前** ✓；探测结果只是**补充** ✓（探测错了也不影响显式配置 ✓）
    #    ⚠️ 根表**只此一份** ✗：安装根表由 ``comfyui_roots`` 给 ✓（引擎侧要 ``<根>/comfy/…``
    #    这类**安装内部**落点 ✓，两处各写一份就会「扫描能看见、加载看不见」✓✗）。
    comfyui_root = os.environ.get("COMFYUI_PATH") or cfg.get("comfyui_root") or ""
    install_roots: list[str] = [root for root, _source in probe()]

    def add(candidate: str) -> None:
        if candidate and os.path.exists(candidate):
            resolved = os.path.abspath(candidate)
            if os.path.normcase(resolved) not in {os.path.normcase(r) for r in roots}:
                roots.append(resolved)

    add(models_dir)
    add(services_dir)
    # ⚠️ 用户**显式**加的目录排在约定落点之前 ✓：显式意图优先于约定 ✓
    extras = list(extra_roots) if extra_roots is not None else _extra_roots_of(cfg)
    for candidate in extras:
        add(candidate)
    # ⭐ **用户声明过的任意目录** ✓（ComfyUI 的 ``extra_model_paths.yaml`` ✓、
    #    ``model-paths.json`` 的 ``extra_paths`` ✓、环境变量 ``EXTRA_MODEL_PATHS`` ✓）——
    #    「模型放任意目录，**声明一下就能用**」✓，口径见 ``model_folders`` ✓。
    #    ⚠️ 与 ``extra_roots`` 同属**显式意图** ⇒ 也排在生态落点之前 ✓（显式优先于约定 ✓）。
    #    ⚠️ **环境变量**指向的声明文件读不开 ⇒ 如实抛 ✓（用户自己指的文件 ✗ 不许静默跳过 ✗）；
    #       而 ComfyUI 那边别人的 ``extra_model_paths.yaml`` 坏掉 ⇒ 只跳过 ✓（不连坐 ✗）。
    declared_dirs = list(declared) if declared is not None else model_folders.declared_extra_dirs(
        cfg,
        os.path.dirname(str(path if path is not None else model_paths.paths_config_file())),
        install_roots,
    )
    for candidate in declared_dirs:
        add(candidate)
    # ⭐ **世界各大模型生态**的落点 ✓（只读 ✓、不要求任何生态的服务在跑 ✓）。
    eco = list(ecosystems) if ecosystems is not None else model_ecosystems.roots()
    for _source, candidate in eco:
        add(candidate)
    # ⭐ 修「只取第一个」✗：**每个** ComfyUI 类目录的 ``models/`` 都加 ✓
    for root in install_roots:
        add(os.path.join(root, "models"))
    # ⚠️ **只有显式配的那个根**整体也加上 ✓（``COMFYUI_PATH`` / ``comfyui_root`` ✓）：
    #    探测到多个安装时，各自的整包（web / tests … 动辄上万文件）会把 ``maxFiles`` 吃光 ✗
    #    ⇒ 生态落点又变成「扫不到也不报错」✗✗；而根下除 ``models/`` 外基本是代码/输出 ✓（不是模型 ✓）。
    if comfyui_root:
        add(comfyui_root)

    # 若完全没有可扫描目录，回退到本仓默认模型目录
    if not roots:
        default = os.path.abspath(model_paths.default_models_dir())
        if os.path.exists(default):
            roots.append(default)
    return roots


def _extra_roots_of(cfg: dict) -> list[str]:
    """配置里 ``extra_roots`` 的**字符串**项 ✓（形状不对的条目 ⇒ 丢掉 ✓，不猜 ✓）。"""
    extra = cfg.get("extra_roots")
    if isinstance(extra, list):
        return [x for x in extra if isinstance(x, str) and x.strip()]
    return []
