"""模型路径**配置面**：配置文件在哪、怎么读写、``models_dir`` 的优先级 ✓。

为什么要单独有这么一层 ✓✗：

* ``configs/model-paths.json`` 读的是**配置**（不是业务数据 ✓），而引擎必须在**不 import 服务层**
  的前提下回答「模型根在哪」✓ —— 以前那句话由 ``app.services.local_model_scan.get_model_paths()``
  提供，等于让引擎每一次 ``plan`` / ``readiness`` / ``doctor`` / ``inspect`` 都先把扫描层（连带
  ``model_folders`` / ``model_ecosystems`` / ``ollama_store``）拽起来 ✓✗；
* 独立性判据 ② 早先只认绝对 import ⇒ 这类相对写法藏了一整轮 ✓（2026-11-08 补判据后露头 ✓）。

**分工**（一层只管一件事 ✓✗）：

* 本模块（``app.core``）：配置文件位置、JSON 读写、``models_dir`` 的**优先级** ✓；
* ``app.services.local_model_scan``：ComfyUI 探测、生态落点、``model_folders`` 声明这些**发现**
  逻辑 ✓ —— 它们要「扫全盘」，只在真扫的时候才需要 ✓。

⚠️ 优先级**只许有一份** ✓✗：``MODELS_DIR`` > ``model-paths.json`` 的 ``models_dir`` >
``<data_root>/models``。谁再抄一遍，就会出现「引擎说这里是根、扫描说那里是根」✓✗
（判据：``tests/local_models_test.py`` 里「优先级只有一份」那条静态守卫 ✓）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, get_data_root

__all__ = [
    "default_models_dir",
    "models_dir",
    "paths_config_file",
    "read_paths_config",
    "write_paths_config",
]

#: 配置檔在仓内的相对位置 ✓（只有这一处写 ✓✗：读写同一个文件，写错就变成两本账 ✓）。
CONFIG_FILE = Path("configs") / "model-paths.json"


def paths_config_file() -> Path:
    """模型路径配置文件的**绝对位置** ✓（读写都从这里取 ✓）。"""
    return PROJECT_ROOT / CONFIG_FILE


def read_paths_config(*, path: Path | None = None) -> dict[str, Any]:
    """读配置（文件不在 / 坏掉 / 不是对象 ⇒ **一律空表** ✓，不抛 ✓）。

    ⚠️ 这里与 ``model_folders`` 的「用户显式指的文件读不开要如实抛」不同 ✓✗：那是**用户当场
    点的路径**，静默跳过会让它以为生效了 ✗；而本文件是本仓自己的状态文件，缺失就是「还没配」✓。

    ``path`` 是**测试隔离口** ✓✗：调用方（``local_model_scan`` 的薄委托）把自己那唯一一处可
    patch 的 ``_paths_config_file()`` 结果传下来 ⇒ 换文件对**读写同时**生效 ✓。2026-11-08 实测
    的教训：只让读走那个口、写仍走本函数的默认路径，一次自检就把 ``configs/model-paths.json``
    的 ``models_dir`` 冲掉、还塞进一条 ``%TEMP%`` 的 ``extra_roots`` ✓✗。
    """
    target = path or paths_config_file()
    try:
        if target.exists():
            parsed = json.loads(target.read_text(encoding="utf-8"))
            return parsed if isinstance(parsed, dict) else {}
    except (OSError, ValueError):
        pass
    return {}


def write_paths_config(cfg: dict[str, Any], *, path: Path | None = None) -> None:
    """写配置 ✓。⚠️ 缩进 2 空格 + 末尾换行（与前端的 ``JSON.stringify(cfg, null, 2) + '\\n'`` 逐字一致）✓。

    ``path`` 同 :func:`read_paths_config`（隔离口，不传就是真身 ✓）。
    """
    target = path or paths_config_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def default_models_dir() -> str:
    """本仓**自己的**模型存储目录 ✓：``<data_root>/models`` ✓。

    「自主」指的是**下载/存放**默认落在这里 ✓；**扫描发现**要覆盖盘上已有的模型（含别人的
    ComfyUI ✓）是另一件事 ✓✗ —— 两件事都归 ``local_model_scan`` 管 ✓。
    """
    return str(Path(get_data_root()) / "models")


def models_dir(*, path: Path | None = None) -> str:
    """``models_dir`` 的**唯一**解析口 ✓：``MODELS_DIR`` > 配置文件 > ``<data_root>/models`` ✓。

    ⚠️ 三项顺序不许改 ✓✗（改了就与 ``local_model_scan.get_default_roots()`` 的第一个根不一致，
    体检说「这里有」、加载却说「那里没有」✓✗）。空串按「没配」处理 ✓（前端清空字段时写的就是空串 ✓）。

    ``path`` 与 :func:`read_paths_config` 同一个**隔离口** ✓✗：配置面三个使用者（读 / 写 / 本函数）
    必须共用同一个文件位置 —— 少传一个，自检里换掉的文件就只对另外两个生效，而 ``models_dir``
    仍去读**真身** ⇒ 「换了文件还是读到真配置」这类假绿 ✓✗（2026-11-08 自检端点判据实测露头）。
    引擎侧（``engine.inventory.models_dir()``）没有隔离口，不传 = 真身 ✓。
    """
    configured = read_paths_config(path=path).get("models_dir")
    return str(os.environ.get("MODELS_DIR") or configured or default_models_dir())
