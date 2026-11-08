"""ComfyUI 口径的「模型目录类别」表 + **任意目录**模型根声明解析（自研重写 ✓）。

⚠️ 口径来源（2026-09-26 逐条核对 ✓）：``ComfyUI/folder_paths.py``（类别表 ✓）与
``ComfyUI/utils/extra_config.py``（``extra_model_paths.yaml`` 语义 ✓）——
⚠️⚠️ 那份参考副本是 **GPL-3.0** ⇒ 这里只搬**语义**（类别名 / 目录名 / 扩展名 / 声明格式 ✓），
代码**自己写** ✗（不拷源码 ✗）。

为什么要有它（2026-09-26 用户口径 ✓：模型可装**任意目录** ✓、也要能扫**任意目录** ✓、不许写死路径 ✗）：
1. :func:`category_of_path`：按**父目录名**（ComfyUI 口径 ✓）再按**文件名**认出「这是哪一类」✓
   —— 比只看文件名准得多 ✓（``…/models/vae/xxx.safetensors`` 一定是 VAE ✓，同名文件在别处就未必 ✓）；
2. :func:`declared_extra_dirs`：读用户**已经声明**的目录 ✓（ComfyUI 的 ``extra_model_paths.yaml`` ✓、
   本仓 ``model-paths.json`` 的 ``extra_paths`` ✓、环境变量 ``EXTRA_MODEL_PATHS`` ✓）
   ⇒ 用户把模型放哪个盘哪个目录，**声明一下就能用** ✓（不用改代码 ✓）；
3. :func:`default_subdir` / :func:`category_extensions`：下载/安装时挑**类别子目录** ✓ 与校验扩展名 ✓
   （``model_manager.py --kind`` 的默认值就取这儿 ✓）。

## ⚠️ 实现已搬 ``app/core/model_folders.py``（2026-11-08 ✓）

本文件**只剩再导出** ✓ —— 下面每一个名字都是**同一个对象** ✓
（``model_folders.FOLDER_SPECS is app.core.model_folders.FOLDER_SPECS`` ✓），不是副本 ✗。

搬家的理由：引擎的 ``inventory.candidate_roots()`` 要「用户声明过的任意目录」那半 ✓，
而引擎顶层只许 import ``app.core.*`` ✗（``tests/engine_independence_test.py`` 逐条判 ✓）。
本模块**零服务层依赖**（只用 ``json`` / ``os`` ✓、读 ComfyUI 约定与用户声明 ✓、不碰业务数据 ✓）
⇒ 与 ``app.core.model_paths`` 同一类，住 ``app/core`` 才是它的位置 ✓。

⚠️ 新增能力请改 ``app/core/model_folders.py`` ✗✗：改在这一份会**静默失效** ✓✗
（调用方拿的是 core 那个对象 ✓）。
"""

from __future__ import annotations

from ..core.model_folders import (
    FOLDER_SPECS,
    LEGACY_CATEGORY_ALIASES,
    PT_EXTENSIONS,
    FolderSpec,
    category_dirs,
    category_extensions,
    category_of_path,
    comfyui_declared_dirs,
    config_declared_dirs,
    declared_extra_dirs,
    default_subdir,
    env_declared_dirs,
    load_extra_paths_file,
    normalize_category,
    resolve_extra_paths,
)

__all__ = [
    "LEGACY_CATEGORY_ALIASES",
    "PT_EXTENSIONS",
    "FOLDER_SPECS",
    "FolderSpec",
    "category_dirs",
    "category_extensions",
    "category_of_path",
    "comfyui_declared_dirs",
    "config_declared_dirs",
    "default_subdir",
    "declared_extra_dirs",
    "env_declared_dirs",
    "load_extra_paths_file",
    "normalize_category",
    "resolve_extra_paths",
]
