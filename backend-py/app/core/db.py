"""SQLite 连接层 —— 对齐 ``backend/src/db/connection.ts``。

与 Node 版一致的两条硬约束：
* ``journal_mode = WAL``（Node 侧同款；两个后端并存时必须一致，否则会出现锁竞争）
* ``busy_timeout = 30000``（30s，Node 侧 ``new Database(path, { timeout: 30000 })``）

**建表策略**：只有库文件不存在时才 ``create_all``（全新克隆场景）。
已有库一律不动 —— Node 侧的 ``db/index.ts`` 用 ``CREATE TABLE IF NOT EXISTS`` +
历史 ALTER 补丁维护 schema，Python 侧不重复这套逻辑，避免两边 DDL 漂移。

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


_db_path = Path(get_db_path())
_db_path.parent.mkdir(parents=True, exist_ok=True)
_is_fresh_db = not _db_path.exists()

engine: Engine = _make_engine(_db_path)

if _is_fresh_db:
    # 仅全新库：与 Node 侧首次启动等价的建表行为
    metadata.create_all(engine)
    print(f"[db-py] fresh db initialized: {_db_path}")
else:
    # 刻意用 ASCII：Windows 控制台默认代码页会把中文打成乱码（见 .codebuddy/memory 工具踩坑）
    print(f"[db-py] reusing existing db: {_db_path}")


def close_engine() -> None:
    """释放当前连接池（切目录**迁移前**调用：Windows 上库文件被占用时复制会失败）。"""
    engine.dispose()


def reopen_engine() -> None:
    """按**当前配置**的数据根重建连接 —— ``POST /storage/change`` 切目录后的收尾。

    与 ``close_engine()`` 的区别：这里把模块级 ``engine`` 换成指向新库的新实例
    （``get_conn`` / ``get_tx`` 引用的就是这个全局名，所以换掉即全体生效）。
    建表策略与导入期一致：**只有库文件本来不存在**才 ``create_all``，已有库不动。
    """
    global engine, _db_path, _is_fresh_db
    engine.dispose()
    _db_path = Path(get_db_path())
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    _is_fresh_db = not _db_path.exists()
    engine = _make_engine(_db_path)
    if _is_fresh_db:
        metadata.create_all(engine)
    print(f"[db-py] engine reopened: {_db_path}")


def get_conn() -> Iterator[Connection]:
    """FastAPI 依赖：只读连接（用完自动归还连接池）。"""
    with engine.connect() as conn:
        yield conn


def get_tx() -> Iterator[Connection]:
    """FastAPI 依赖：事务连接（正常返回时 commit，异常时 rollback）。"""
    with engine.begin() as conn:
        yield conn
