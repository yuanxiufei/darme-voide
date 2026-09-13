"""数据存储信息 —— 移植 ``backend/src/services/data-storage.ts`` 的**只读部分**。

⚠️ **``changeDataRoot`` 刻意未移植**（对应 ``POST /storage/change`` 不注册 → 走反代）。
理由不是「难」，而是**绞杀期共享状态风险**：

1. 它会写项目级 ``.data-root`` 标记文件，而 **Node 后端启动时也读这个文件**
   ⇒ 从 Python 切一次目录，会连带把 Node 的数据根一起挪走。两个后端并存时这是
   跨进程副作用，切域那一刻的行为会不可预期。
2. 它还要关掉并重开数据库连接。Node 侧是单例全局连接；Python 侧我的 engine 是
   导入期构建的 + ``config`` 的路径在导入期解析 —— 虽然 ``config.set_data_root()``
   已具备运行时切换能力，但「DB 连接重建 + 标记文件 + 两进程并发」三者叠加，
   在绞杀期收益极低、风险却不小。

⇒ 等 Node 完全下线（单后端）再迁这个端点，那时没有共享状态问题。
"""

from __future__ import annotations

import os
from typing import Any

from ..config import get_data_root, get_db_path, get_storage_root


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path) if os.path.exists(path) else 0
    except OSError:
        return 0


def _dir_size(directory: str) -> int:
    """递归统计目录体积；单个子项失败即跳过（与原 TS 一致，返回**部分**和）。"""
    if not os.path.exists(directory):
        return 0
    total = 0

    def walk(path: str) -> None:
        nonlocal total
        try:
            entries = list(os.scandir(path))
        except OSError:
            return
        for entry in entries:
            if entry.is_dir():
                walk(entry.path)
            else:
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass

    walk(directory)
    return total


def get_storage_info() -> dict[str, Any]:
    """当前数据存储信息（键名与 TS 的 ``StorageInfo`` 一致，camelCase）。"""
    db_path = get_db_path()
    storage_path = get_storage_root()
    return {
        "dataRoot": get_data_root(),
        "dbPath": db_path,
        "storagePath": storage_path,
        "dbExists": os.path.exists(db_path),
        "storageExists": os.path.exists(storage_path),
        "dbSizeBytes": _file_size(db_path),
        "storageSizeBytes": _dir_size(storage_path),
    }
