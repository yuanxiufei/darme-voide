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


def plan_schema_upgrade(eng: Engine | None = None) -> tuple[list[str], list[str]]:
    """算出「已存在的表上缺的列」对应的 ``ALTER TABLE … ADD COLUMN``（**只算不执行** ✗）。

    返回 ``(statements, blockers)``：

    * ``statements``：**可直接执行**的语句 ✓（顺序按 ``metadata`` 的声明顺序 ⇒ 跑两遍一致 ✓）；
    * ``blockers``：SQLite 的 ``ADD COLUMN`` **加不了**的列 ✗ —— ``PRIMARY KEY`` / ``UNIQUE``
      加不上 ✓，``NOT NULL`` 也得先有默认值 ✓。这类**具名报出、不做猜测** ✓
      （不偷偷塞 ``DEFAULT ''`` ✗：那是把「结构变化」伪装成「数据默认值」✗，
      正是本仓「判不出来就拒绝」那条纪律要拦的 ✓）。

    判据与 :func:`_ensure_tables` 的启动体检**同源** ✓（都读 ``metadata`` 这个唯一权威 ✓），
    区别只在用途：那里打日志（**只取前 6 条** ✓，扫一眼就知道有事 ✓），这里给运维**完整清单** ✓
    （``app/scripts/db_upgrade.py`` ✓）—— ⚠️ 2026-09-25 实测：真实库缺 **32 列**，
    日志截断导致「到底要补哪些」得另外写探针才拿得到 ✗。
    """
    from sqlalchemy import inspect as sa_inspect  # noqa: PLC0415 —— 只在体检用 ✓

    target = eng if eng is not None else engine
    inspector = sa_inspect(target)
    existing = set(inspector.get_table_names())
    statements: list[str] = []
    blockers: list[str] = []
    for name in sorted(set(metadata.tables) & existing):
        have = {column["name"] for column in inspector.get_columns(name)}
        for column in metadata.tables[name].columns:
            if column.name in have:
                continue
            where = f"{name}.{column.name}"
            if column.primary_key or column.unique:
                blockers.append(f"{where}（PRIMARY KEY / UNIQUE ⇒ SQLite 加不了，需重建表）")
                continue
            if not column.nullable:
                blockers.append(f"{where}（NOT NULL ⇒ 必须先定默认值，由人决定）")
                continue
            statements.append(f'ALTER TABLE "{name}" ADD COLUMN "{column.name}" '
                              f"{column.type.compile(target.dialect)}")
    return statements, blockers


def apply_schema_upgrade(eng: Engine | None = None) -> list[str]:
    """**显式**执行补列，返回已执行的语句 ✓（只应由 ``db_upgrade.py --apply`` 在备份后调用 ✓）。

    **全有或全无** ✓：只要有一条 ``blocker`` 就**一条都不执行** ✗ ——
    半升级的库比旧库更难诊断 ✓（与 ``services/continuity_store.py`` 同口径 ✓）。
    """
    target = eng if eng is not None else engine
    statements, blockers = plan_schema_upgrade(target)
    if blockers:
        raise RuntimeError(
            "有列无法用 ALTER 添加 ⇒ 已拒绝执行任何一条（请人工处理）：\n  "
            + "\n  ".join(blockers)
        )
    with target.begin() as conn:
        for sql in statements:
            conn.exec_driver_sql(sql)
    return statements


def _ensure_tables(eng: Engine, *, fresh: bool) -> None:
    """**只增不改**地调和 schema ✓：建缺失的表 + 报出缺失的列（不自动 ALTER ✗）。

    两条输出都只在**有问题**时打印 ✓（正常启动保持安静 ✓）：
    * ``created``：本次新建了哪些表（已有库升级到新版时最需要看到 ✓）；
    * ``columns missing``：已存在表上**缺列**（新加的列不会自动生效 ✗）⇒ 打印可执行的 ALTER 提示 ✓。

    ⚠️ 提示**只取前 6 条**（日志要能扫一眼 ✓）⇒ 真要动手请用 ``plan_schema_upgrade()``
    或 ``app/scripts/db_upgrade.py`` 拿**完整清单** ✓（2026-09-25：真实库那 32 条就是被截断的 ✓）。
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
    statements, blockers = plan_schema_upgrade(eng)
    if statements or blockers:
        print(f"[db-py] ⚠️ columns missing in existing db "
              f"({len(statements)} 可补 / {len(blockers)} 需人工): {statements[:6]}"
              f"{' …' if len(statements) > 6 else ''}")
        for item in blockers:
            print(f"[db-py]   ✗ {item}")
        print("[db-py]   补列入口: python app/scripts/db_upgrade.py --plan | --apply")
    if not created and not statements and not blockers:
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


def _main(argv: list[str] | None = None) -> int:
    """补列入口（**供 ``app/scripts/db_upgrade.py`` 以 subprocess 调用** ✓，别直接当 API 用 ✗）。

    ⚠️ 为什么入口在这儿而不是全写在 ``app/scripts/`` 里 ✗：那个目录**不是包** ✓
    （``tests/engine_readiness_script_test.py`` 记着这条教训 ✓），而补列的「缺哪列」只能从
    ``metadata``（唯一权威 ✓）算出来 ⇒ 逻辑留在本模块 ✓，脚本只负责**备份 + 转发** ✓。
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="python -m app.core.db",
        description="旧库 schema 补列（只做 ALTER TABLE ADD COLUMN，不建表/不删列）",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true", help="只打印完整补列清单（不动库）")
    group.add_argument("--apply", action="store_true", help="执行补列（调用方须先备份）")
    args = parser.parse_args(argv)

    print(f"[db-upgrade] db = {_db_path}")
    statements, blockers = plan_schema_upgrade()

    if args.plan:
        for sql in statements:
            print(f"[db-upgrade]   {sql};")
        print(f"[db-upgrade] 待补 {len(statements)} 列；需人工 {len(blockers)} 列")
    else:
        applied = apply_schema_upgrade()
        print(f"[db-upgrade] 已补 {len(applied)} 列")
    for item in blockers:
        print(f"[db-upgrade]   ✗ {item}")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(_main())
