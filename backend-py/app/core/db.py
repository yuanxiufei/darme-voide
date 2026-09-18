"""SQLite 连接层 —— 对齐 ``backend/src/db/connection.ts``。

与 Node 版一致的两条硬约束：
* ``journal_mode = WAL``（Node 侧同款；两个后端并存时必须一致，否则会出现锁竞争）
* ``busy_timeout = 30000``（30s，Node 侧 ``new Database(path, { timeout: 30000 })``）

**建表策略（2026-09-17 修订 ✓）**：``create_all(checkfirst=True)`` **每次启动都跑** —— 它只
**创建缺失的表** ✓、既不 ALTER 也不 DROP ✓，因此对已有库是安全的 ✓。

⚠️ 为什么改（旧策略的**理由已过期** ✗）：原来的策略是「只有库文件不存在才建表，已有库一律不动」，
理由是「Node 侧的 ``db/index.ts`` 用 ``CREATE TABLE IF NOT EXISTS`` + 历史 ALTER 补丁维护 schema，
Python 侧不重复这套逻辑，避免两边 DDL 漂移」✗ —— 但 ``backend/`` **已经被删掉了** ✓，
于是这条理由**不再成立** ✗，而后果是**静默且致命**的：
Python 侧新加的表在**已有库**上根本不会被创建 ✗ ⇒ 任何碰新表的代码（例如启动期的崩溃恢复）
在真机上直接 `no such table` 把服务拖崩 ✗（2026-09-17 冒烟测试当场抓到 ✓）。

⚠️ **仍然不做**列迁移 ✗：给已存在的表**加列**不会被自动应用 ✓ —— 这类缺口现在会被**明确报出来**
（见 :func:`_ensure_tables` 的列体检 ✓），由人决定 ALTER，而不是我们偷偷改用户的库 ✗。

``engine`` 是**模块级单例**（``get_conn`` / ``get_tx`` 在调用时才取它）⇒
``POST /storage/change`` 切换数据根目录后必须 ``reopen_engine()``，后续请求才会打到新库；
这对应 Node 侧的 ``reopenDatabase()`` + ``rebuildDb()``（见 ``services/data_storage.py``）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Connection, Engine

from .config import get_db_path
from .models import metadata


def _set_sqlite_pragmas(dbapi_conn, _connection_record) -> None:  # noqa: ANN001
    cur = dbapi_conn.cursor()
    try:
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
    finally:
        cur.close()


def _make_engine(db_path: Path) -> Engine:
    """建 engine 并挂上 PRAGMA 监听。

    抽成函数是因为**重开连接也要挂监听**（早先监听靠装饰器挂在初始 engine 上，
    换 engine 就会静默丢掉 WAL/busy_timeout 两条硬约束）。
    """
    eng = create_engine(
        f"sqlite+pysqlite:///{db_path.as_posix()}",
        future=True,
        # check_same_thread=False：FastAPI 的同步端点跑在线程池里，连接会跨线程复用
        connect_args={"timeout": 30, "check_same_thread": False},
    )
    event.listen(eng, "connect", _set_sqlite_pragmas)
    return eng


def _ensure_tables(eng: Engine, *, fresh: bool) -> None:
    """**只增不改**地调和 schema ✓：建缺失的表 + 报出缺失的列（不自动 ALTER ✗）。

    两条输出都只在**有问题**时打印 ✓（正常启动保持安静 ✓）：
    * ``created``：本次新建了哪些表（已有库升级到新版时最需要看到 ✓）；
    * ``columns missing``：已存在表上**缺列**（新加的列不会自动生效 ✗）⇒ 打印可执行的 ALTER 提示 ✓。
    """
    from sqlalchemy import inspect as sa_inspect

    existing = set(sa_inspect(eng).get_table_names())
    expected = set(metadata.tables)
    created = sorted(expected - existing)

    metadata.create_all(eng, checkfirst=True)  # 只建缺失的表 ✓ 幂等 ✓

    if fresh:
        print(f"[db-py] fresh db initialized: {_db_path}")
        return
    if created:
        print(f"[db-py] schema upgraded: created {created} in {_db_path}")
    # 列体检：**只报不改** ✗（改用户的库必须由人决定 ✓）
    inspector = sa_inspect(eng)
    gaps: list[str] = []
    for name in sorted(expected & existing):
        have = {column["name"] for column in inspector.get_columns(name)}
        want = {column.name for column in metadata.tables[name].columns}
        for column in sorted(want - have):
            spec = metadata.tables[name].columns[column]
            gaps.append(f'{name}.{column} (ALTER TABLE "{name}" ADD COLUMN "{column}" '
                        f'{spec.type.compile(eng.dialect)}'
                        f'{" NOT NULL DEFAULT ..." if not spec.nullable else ""})')
    if gaps:
        print(f"[db-py] ⚠️ columns missing in existing db ({len(gaps)}): {gaps[:6]}"
              f"{' …' if len(gaps) > 6 else ''}")
    if not created and not gaps:
        print(f"[db-py] reusing existing db: {_db_path}")


_db_path = Path(get_db_path())
_db_path.parent.mkdir(parents=True, exist_ok=True)
_is_fresh_db = not _db_path.exists()

engine: Engine = _make_engine(_db_path)

_ensure_tables(engine, fresh=_is_fresh_db)


def close_engine() -> None:
    """释放当前连接池（切目录**迁移前**调用：Windows 上库文件被占用时复制会失败）。"""
    engine.dispose()


def reopen_engine() -> None:
    """按**当前配置**的数据根重建连接 —— ``POST /storage/change`` 切目录后的收尾。

    与 ``close_engine()`` 的区别：这里把模块级 ``engine`` 换成指向新库的新实例
    （``get_conn`` / ``get_tx`` 引用的就是这个全局名，所以换掉即全体生效）。
    建表策略与导入期一致：**只增不改**（``_ensure_tables`` ✓）—— 切到旧库也要把缺的表补上 ✓，
    否则「切目录后某些功能报 no such table」会变成只有用户才能发现的坑 ✗。
    """
    global engine, _db_path, _is_fresh_db
    engine.dispose()
    _db_path = Path(get_db_path())
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    _is_fresh_db = not _db_path.exists()
    engine = _make_engine(_db_path)
    _ensure_tables(engine, fresh=_is_fresh_db)
    print(f"[db-py] engine reopened: {_db_path}")


def get_conn() -> Iterator[Connection]:
    """FastAPI 依赖：只读连接（用完自动归还连接池）。"""
    with engine.connect() as conn:
        yield conn


def get_tx() -> Iterator[Connection]:
    """FastAPI 依赖：事务连接（正常返回时 commit，异常时 rollback）。"""
    with engine.begin() as conn:
        yield conn
