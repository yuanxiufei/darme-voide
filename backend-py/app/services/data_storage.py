"""数据存储 —— 移植 ``backend/src/services/data-storage.ts``（**已接线**）。

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
5. 迁移是**复制**（旧目录保留作备份）：库文件 + ``-wal`` / ``-shm`` 伴生文件、``static/``、
   ``library-overlay.sqlite``，以及 ``MIGRATED_TREE_SUBDIRS`` 那几个**不可重建**的整目录；
   复制抛错时**回滚**（重开旧库）并把错误包成「迁移失败：…」；
6. 最后写标记文件（重启仍生效）+ 重开连接（对应 Node 的 ``reopenDatabase()`` + ``rebuildDb()``）；
7. 复制前先算账：旧数据体积 + 余量 > 目标盘可用空间 ⇒ **当场拒**（2026-11-08 补）。
   这条是本仓自定的 —— 参考侧 anything-llm / ragflow 这类自托管项目都是「配置项 + 重启生效」、
   不做运行时迁移，没有可搬的实现；而本仓是在线切换，半途失败虽会回滚连接、标记也不写，
   但新根下会留下**半份目录**，下次切进该根时 ``_copy_tree`` 的合并语义会把半成品当"已有数据"叠上去 ✓✗。
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


def _human_bytes(size: int) -> str:
    """人话体积（只用于错误文案：用户要能一眼判断「差多少」）。"""
    value = float(max(int(size), 0))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


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
#:   * ``canvas/``     —— 画布草案（``canvas_plan_store.plans_dir()`` ⇒ ``canvas/plans/<16位id>.json``）。
#:                        草案是**用户手作**（节点位置/连线/填好的 values），没有任何源可再生；
#:                        而 ``plans_dir()`` 动态读 ``get_data_root()`` ⇒ 切根后它**立刻指向新根的空目录**，
#:                        症状正是「切个目录，画布全空了」而旧草案还躺在旧根没人读 ✓✗（2026-11-08 补）。
#:   * ``knowledge/``  —— 知识台账 ``knowledge-ledger.sqlite``（Agent 蒸馏 + 人审累积的条目、
#:                        ``entry_versions``、审计）与训练登记 ``training-registry.sqlite``
#:                        （哪个 LoRA 被 ``promote``、依据与凭据）**都不可重建**；而两个模块的
#:                        ``db_path()`` 每次现算 ``get_data_root()`` ⇒ 不搬的症状是
#:                        「切个目录，积累的经验与训练登记全空了」，旧库还躺在旧根没人读 ✓✗
#:                        （2026-11-08 补）。
#:                        ⚠️ 同目录里另有两个**派生**库（``outcome-index`` 可从 ``traces`` 的
#:                        ``rebuild()`` 重放、``skill-usage`` 从跑次重算）。它们跟着一起搬是**故意的**：
#:                        白名单是**目录粒度**，多搬这 100 KB 换来的好处是切根后「优化史／质检命中率
#:                        ／跑次统计」的历史立刻可用 —— 否则要手跑 ``rebuild`` 才补得回来。别为了
#:                        「只搬真源」把它拆成文件级清单（那要多维护一套，收益只是少搬 100 KB）。
#:
#: **故意不搬**（写下来是为了防后人「顺手补齐」✓✗）：``models/`` 与 ``hf-cache/`` 能重下、
#: ``materialized/`` 能重算、``engine-dryrun/`` 与 ``repair/`` 能重跑/重诊断。
#: 这些一次切根白搬几十 GB（慢、占盘），而不搬**不会丢东西**。
#: 判据的反面同样要守住：**不可重建的一律得搬** —— ``canvas/`` 与 ``knowledge/`` 就是照这条补进来的。
#: ⚠️ **别按目录名判、要按目录里的东西判**：``knowledge/`` 曾因为「名字里带 knowledge、看起来像
#: 索引缓存」被整目录划进「能重算」（连 ``data/README.md`` 的落点表都写着「下次检索时自动重建」），
#: 而真相是四个库里两个不可重建。``canvas/`` 是同一个形状：只有几百 KB、名字像临时缓存，
#: 里面却是人一场一场摆出来的布局。
#:
#: 为什么 ``db`` / ``static`` / ``library-overlay`` 不在表里：三者各有覆盖口
#: （``DB_PATH`` / ``STORAGE_PATH`` / 覆盖层的模块常量），只能各自走
#: ``get_db_path()`` / ``get_storage_root()`` / 显式路径算，不能按 ``old_root / 名字`` 拼 ✓✗。
#:
#: 判据：``tests/storage_change_test.py`` ⑨ 双向核对 —— 表内每个名字真的搬了（④ 那条），
#: 且表**不许**混进上面那批「能重下/重算」的目录。
MIGRATED_TREE_SUBDIRS: tuple[str, ...] = ("traces", "lora-train", "canvas", "knowledge")

#: 迁移前的空间余量：复制完还要写标记、建库表、写 WAL —— 卡在「刚刚好」上会以更难看的方式失败。
#: 64 MiB 是**故意偏小**的：它是防「目标盘明显装不下」的，不是精确容量规划；
#: 真按 GB 级余量卡会在边界上误拒合法目录（旧数据本身还能被清走再切）。
MIGRATION_HEADROOM_BYTES = 64 * 1024 * 1024


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
        # ⚠️ 复制前先算账：目标盘装不下就**当场拒**（理由写在文件头第 7 条）。
        # 这里刻意在 ``close_engine()`` **之前**——此时连接池还开着、状态一点没动，
        # 抛出去就是干净的拒绝；若放到关连接之后，报错路径还得再回滚一次。
        need = (
            _file_size(old_db_path)
            + _file_size(f"{old_db_path}-wal")
            + _file_size(f"{old_db_path}-shm")
            + _file_size(old_overlay_path)
            + _dir_size(old_storage_path)
            + sum(_dir_size(str(old_root / name)) for name in MIGRATED_TREE_SUBDIRS)
        )
        try:
            free = shutil.disk_usage(str(new_root)).free
        except OSError:
            # 取不到可用空间（网络盘/特殊文件系统）就**不拦**：宁可不预检，也不误拒合法目录。
            free = -1
        if free >= 0 and need + MIGRATION_HEADROOM_BYTES > free:
            raise ValueError(
                f"目标目录可用空间不足：需要约 {_human_bytes(need)}，"
                f"可用 {_human_bytes(free)}（旧目录会保留，可先清理再试）"
            )
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
