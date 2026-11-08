"""媒体引用 → 本机路径的**公共口** —— 引擎侧只许认这一份。

为什么这两个纯函数要住在 ``app/core`` ✗✓：

- ``get_absolute_path`` 是**安全关键**的路径穿越防线 ✓（业务侧十来个模块都靠它 ✓）；
- ``parse_data_url`` 只解 ``data:<mime>;base64,...`` 一种形状 ✓。

它们原本住在 ``app/services/file_storage.py`` ✓ —— 而**引擎读参考图**（``engine/bridge.py``
的 ``materialize_reference``）恰恰必须经过这两步 ✓✗：引擎只吃**本机文件**（它不联网 ✗），
所以「``static/...`` 怎么变绝对路径」「data URL 怎么拆包」是引擎的必经口 ✓。
住在服务层 ⇒ 引擎为了解一条路径就得 ``import app.services.file_storage`` ✓✗，那正是
``tests/engine_independence_test.py`` 的 ``_SEAMS`` 里要被逐条解释的那类越界 ✓。

两个函数本来就不碰服务层任何东西（一个只用 stdlib，一个只用 :func:`config.get_storage_root` ✓）
⇒ 搬家零代价 ✓。``app/services/file_storage.py`` 现在只是**再导出同一批对象** ✓（不是副本 ✗）：
业务侧调用点一字不改 ✓，也不存在第二本账 ✓✗。守卫见 ``tests/engine_independence_test.py``
（引擎里再出现 ``app.services.file_storage`` 即红 ✓）与 ``tests/output_layout_test.py``
（``get_absolute_path("x")`` 的落点三路对账 ✓）。

⚠️ ``get_absolute_path`` 的语义**一个字都不许动** ✓✗：``static/`` 打头时基准要上跳到
storageRoot 的**父目录**（``static/xxx`` 这类返回值是「相对数据根」的 ✓），而穿越检查始终
以真实 storageRoot 为界 ✓。搬了家之后这条仍然是全仓唯一实现 ✓。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .config import get_storage_root

__all__ = ["get_absolute_path", "parse_data_url"]


def get_absolute_path(relative_path: str) -> str:
    """相对路径 → 绝对路径（**防路径穿越**）。

    ⚠️ 一处容易看漏的原逻辑：若路径以 ``static/`` 开头，基准要**上跳到 storageRoot 的父目录**
    —— 因为 ``static/xxx`` 这类返回值是「相对数据根」而非「相对 storageRoot」的。
    路径穿越检查最终仍以**真实的 storageRoot** 为界，所以上跳本身不构成逃逸。
    """
    storage_root = Path(get_storage_root()).resolve()
    if relative_path.startswith("static/") or relative_path.startswith("static\\"):
        base = storage_root.parent
    else:
        base = storage_root

    resolved = (base / relative_path).resolve()
    if not (
        str(resolved).startswith(str(storage_root) + os.sep) or str(resolved) == str(storage_root)
    ):
        raise ValueError(f"Path traversal blocked: {relative_path}")
    return str(resolved)


def parse_data_url(data_url: str) -> dict[str, str] | None:
    """``data:image/png;base64,xxxx`` → ``{mimeType, data}``；不匹配返回 None。"""
    match = re.match(r"^data:([^;]+);base64,(.+)$", str(data_url or ""), re.S)
    if not match:
        return None
    return {"mimeType": match.group(1), "data": match.group(2)}
