"""数据存储 —— 移植 ``backend/src/services/data-storage.ts``（**整域关闭**）。

两个端点：
* ``GET  /storage/info``   —— 只读信息
* ``POST /storage/change`` —— 切换数据根目录（可选迁移旧数据）

⚠️ 关于 ``change_data_root`` 的**并存期行为**（早先它被列为「刻意未迁移」，理由现状如下）：
它会写项目级 ``.data-root`` 标记文件，而 **Node 后端启动时也读这个文件** ⇒ 两个后端并存时
从 Python 切一次目录，会连带把 Node 的数据根一起挪走，是**跨进程副作用**。
该风险**依然存在**，只是它已成为「删 ``backend/``」的最后一块阻塞（Node 下线后此端点必须
由 Python 提供，否则单后端下前端设置页会直接 501）⇒ 现在迁移，并把风险记在此处：
**并存期不要随手调它**，等到单后端再交由前端正常使用。

逐条对齐 TS 的行为（都有自检钉住）：
1. 空白路径报错（路由层先拦，服务层再兜一道，**两处文案不同**）；
2. ``migrate`` 语义是 ``!== false``（只认字面量 ``False``；``null``/``0`` 都算「要迁移」）；
3. 拒绝把数据目录设成项目根，或放进 ``backend`` / ``frontend`` / ``node_modules`` / ``configs`` / ``.git``；
4. 建目录 + 写探针文件验可写，失败报「目录不可写」；
5. 迁移是**复制**（旧目录保留作备份）：库文件 + ``-wal`` / ``-shm`` 伴生文件、``static/``、``traces/``；
   复制抛错时**回滚**（重开旧库）并把错误包成「迁移失败：…」；
6. 最后写标记文件（重启仍生效）+ 重开连接（对应 Node 的 ``reopenDatabase()`` + ``rebuildDb()``）。
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from ..core import config
from ..core.config import get_data_root, get_db_path, get_storage_root, set_data_root
from ..core.db import close_engine, reopen_engine


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


def copy_db_files(old_db_path: str, new_db_path: str) -> None:
    """复制 SQLite 库文件及 WAL/SHM 伴生文件（缺哪个跳哪个，与 TS 同）。"""
    target_dir = os.path.dirname(new_db_path)
    os.makedirs(target_dir, exist_ok=True)
    for source in (old_db_path, f"{old_db_path}-wal", f"{old_db_path}-shm"):
        if os.path.exists(source):
            shutil.copyfile(source, os.path.join(target_dir, os.path.basename(source)))


def _copy_tree(source: str, target: str) -> None:
    """递归复制目录 —— 对位 TS 的 ``fs.cpSync(src, dst, {recursive: true})``。

    注意 ``dirs_exist_ok=True``：TS 的 ``cpSync`` 是**合并**语义（目标已存在也照写），
    不是「目标必须不存在」。
    """
    shutil.copytree(source, target, dirs_exist_ok=True)


#: 切根时**整目录搬走**的 ``<dataRoot>`` 子目录（元组顺序＝复制顺序）—— 单一真源 ✓
#:
#: 收录判据只有一条：**丢了算不算得回来** ✓
#:   * ``traces/``     —— 跑批观测。重跑能再有，但"当时那次为什么长这样"只有它有 ⇒ 搬
#:   * ``lora-train/`` —— 手调参数 + 训出来的 ``*.safetensors``。重训是几小时卡时，
#:                        且 ``lora_train.config.data_dir()`` 会在新根 ``mkdir`` 出空目录
#:                        ⇒ 不搬的症状是「切个目录，训练集和权重全没了」而旧数据还在旧根 ✓✗
#:
#: **故意不搬**（写下来是为了防后人「顺手补齐」✓✗）：``models/`` 与 ``hf-cache/`` 能重下、
#: ``knowledge/`` 与 ``materialized/`` 能重算、``engine-dryrun/`` 与 ``repair/`` 能重跑/重诊断。
#: 这些一次切根白搬几十 GB（慢、占盘），而不搬**不会丢东西**。
#:
#: 为什么 ``db`` / ``static`` / ``library-overlay`` 不在表里：三者各有覆盖口
#: （``DB_PATH`` / ``STORAGE_PATH`` / 覆盖层的模块常量），只能各自走
#: ``get_db_path()`` / ``get_storage_root()`` / 显式路径算，不能按 ``old_root / 名字`` 拼 ✓✗。
#:
#: 判据：``tests/storage_change_test.py`` ⑨ 双向核对 —— 表内每个名字真的搬了（④ 那条），
#: 且表**不许**混进上面那批「能重下/重算」的目录。
MIGRATED_TREE_SUBDIRS: tuple[str, ...] = ("traces", "lora-train")


def _write_probe(directory: str) -> None:
    """可写性探针（单独成函数：自检要能打桩出「不可写」分支）。"""
    probe = os.path.join(directory, f".write-test-{int(time.time() * 1000)}")
    with open(probe, "w", encoding="utf-8") as handle:
        handle.write("ok")
    os.remove(probe)


def change_data_root(new_root_raw: Any, migrate: bool = True) -> dict[str, Any]:
    """切换数据根目录（默认迁移旧数据，旧目录保留）。

    :raises ValueError: 任何校验/迁移失败，消息与 TS 的 ``throw new Error(...)`` 逐字一致。
    """
    trimmed = str(new_root_raw or "").strip()
    if not trimmed:
        raise ValueError("请填写有效的目录路径")
    # 相对路径按进程 cwd 解析（同 TS 的 path.resolve）
    new_root = Path(trimmed).resolve()

    old_root = Path(get_data_root()).resolve()
    old_db_path = get_db_path()
    old_storage_path = get_storage_root()

    if new_root == old_root:
        raise ValueError("新目录与当前目录相同，无需切换")

    # 目录合法性校验：禁止指向项目根、源码/依赖/配置目录
    project_root = config.PROJECT_ROOT
    if new_root == project_root:
        raise ValueError("数据目录不能设置为项目根目录")
    forbidden = [
        project_root / "backend",
        project_root / "frontend",
        project_root / "node_modules",
        project_root / "configs",
        project_root / ".git",
    ]
    if any(new_root == item or str(new_root).startswith(str(item) + os.sep)
           for item in forbidden):
        raise ValueError("数据目录不能设置在项目源码/依赖/配置目录内")

    # 创建目录并测试可写性
    new_root.mkdir(parents=True, exist_ok=True)
    try:
        _write_probe(str(new_root))
    except OSError as exc:
        raise ValueError("目录不可写，请检查权限或更换目录") from exc

    # ⚠️ 库一律落在新根的 ``db/`` 下（与 ``config.get_db_path`` 同源 ✓）：
    # 这里若仍写 ``new_root / "drama.db"``，切根后的 ``reopen_engine`` 会去 ``db/`` 找不到库 ⇒
    # **当场建一个空库**，用户看到的就是「切了目录数据全没了」✓✗（复制过的文件还躺在顶层没人读）。
    new_db_path = str(new_root / config.DB_SUBDIR / "drama.db")
    new_storage_path = str(new_root / "static")

    # ``library-overlay.sqlite`` 也跟数据根走（编辑/上传台账，不可重建 ✓）——
    # 切根时不同步它 ⇒ 换台机器/换个盘之后「知识库的编辑与上传记录全没了」✓✗。
    old_overlay_path = str(old_root / config.DB_SUBDIR / "library-overlay.sqlite")
    new_overlay_path = str(new_root / config.DB_SUBDIR / "library-overlay.sqlite")

    # 迁移旧数据（复制）。⚠️ 只认字面量 False 才跳过（TS 是 `opts.migrate !== false`）
    if migrate is not False:
        # 先关连接池：Windows 上库文件被占用时复制会失败
        close_engine()
        try:
            if os.path.exists(old_db_path):
                copy_db_files(old_db_path, new_db_path)
            if os.path.exists(old_overlay_path):
                copy_db_files(old_overlay_path, new_overlay_path)
            if os.path.exists(old_storage_path):
                _copy_tree(old_storage_path, new_storage_path)
            # 其余「整目录搬」的走单一真源（收录理由与被排除项都写在常量的注释里）
            for _sub in MIGRATED_TREE_SUBDIRS:
                _src = str(old_root / _sub)
                if os.path.exists(_src):
                    _copy_tree(_src, str(new_root / _sub))
        except Exception as exc:  # noqa: BLE001
            # 迁移失败回滚：重新打开旧数据库（数据根尚未切换）
            try:
                reopen_engine()
            except Exception:  # noqa: BLE001 - 回滚本身失败也不该掩盖原错误
                pass
            raise ValueError(f"迁移失败：{exc}") from exc

    # 更新数据根目录（写标记文件，重启后依然生效）
    set_data_root(str(new_root))

    # 重开连接并指向新库（库文件不存在时顺手建表，同 TS 的 reopenDatabase + rebuildDb）
    reopen_engine()

    # 知识库覆盖层是模块级常量 ⇒ 也要跟着换根，否则编辑/上传台账继续读写旧目录
    from . import library_overlay  # noqa: PLC0415 —— 只为避开 services 包内的导入环

    library_overlay.resync_path()

    return get_storage_info()
