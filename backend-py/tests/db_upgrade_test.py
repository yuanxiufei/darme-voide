"""库升级脚本守卫：**``.bak-*`` 备份只留最近 N 份**（``app/scripts/db_upgrade.py`）。

补列是「一次性迁移」里会反复做的事 ⇒ 每次 ``--apply`` 都留一份 ``<db>.bak-<时间戳>``，
不清就一路累积（2026-11-07 实测盘上已躺两份，最老那份是 0925 缺 32 列的老库，
与当前库差好几个版本 ⇒ 留着只会让人分不清「升级前」是哪份）。

钉五件事：

1. **留最新、删最旧** —— 排序读**文件名里的时间戳**，不读 mtime
   （复制/解压/同步都会把 mtime 抹平 ⇒ 按 mtime 排会删掉真正最新的那份）；
2. **刚做的那份永远在** —— 裁剪排在新建之后；判据是「先 ``_backup`` 再 ``_prune``」这一真实顺序；
3. **只碰自己造的形状** —— ``<db>.bak-<YYYYmmdd-HHMMSS>``；同一目录里还有整套备份
   ``_backup_<时间戳>/``、``-wal``/``-shm`` 旁文件、人手改名留下的 ``.bak-<别的>``
   ⇒ 一个都不许删，也**不许把它们的份数算进去**（算进去会让真正的旧备份永远排不进清理位）；
4. **删不掉不许拖垮补列** —— 备份被占用/无权限时跳过并继续（本脚本主任务是补列）；
5. **反向验证** —— ``keep`` 真被用上（``keep=1`` 时只剩一份），否则「常量摆着不生效」查不出来。

运行::

    ./.venv/Scripts/python.exe tests/db_upgrade_test.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_PY))

from app.scripts import db_upgrade as up  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def make_db(root: Path) -> Path:
    """造一个**真 sqlite 库** ✓（``_backup`` 走 ``sqlite3.backup`` API，不是写文件 ⇒ 假文件跑不通）。

    ⚠️ 连接必须**显式 close** ✗：``with sqlite3.connect(...)`` 只提交事务、不关连接 ✓✗，
    Windows 上句柄留着 ⇒ ``TemporaryDirectory`` 收尾删文件当场 ``WinError 32`` ✓
    （本套第一版就栽在这儿 ✓ —— 与 ``_backup`` 里那处是同一个坑 ✓）。
    """
    db = root / "drama.db"
    conn = sqlite3.connect(db)
    try:
        with conn:
            conn.execute("CREATE TABLE demo (id INTEGER PRIMARY KEY)")
    finally:
        conn.close()
    return db


def make_bak(db: Path, stamp: str, *, mtime: float | None = None) -> Path:
    """按给定时间戳造一份备份 ✓（``mtime`` 可单独拧 ⇒ 才验得动「不按 mtime 排」）。"""
    path = db.with_name(f"{db.name}.bak-{stamp}")
    path.write_bytes(b"backup")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def names(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


def main() -> int:
    # ── ① 五份 ⇒ 留最新三份，删最旧两份（乱序创建 ⇒ 排除「按创建顺序」的巧合） ──
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db = make_db(root)
        stamps = ["20250925-153222", "20260101-000000", "20261107-101010",
                  "20250601-080000", "20260401-120000"]
        # 故意按乱序落盘，最后结果只由时间戳决定
        for stamp in ["20261107-101010", "20250925-153222", "20260401-120000",
                      "20250601-080000", "20260101-000000"]:
            make_bak(db, stamp)

        kept_before = names(up._backups_of(db))
        check("① `_backups_of` 由新到旧（不看创建顺序）",
              kept_before == ["drama.db.bak-20261107-101010", "drama.db.bak-20260401-120000",
                              "drama.db.bak-20260101-000000", "drama.db.bak-20250925-153222",
                              "drama.db.bak-20250601-080000"], kept_before)

        removed = up._prune_backups(db)
        check("① 删掉的正是最旧两份",
              names(removed) == ["drama.db.bak-20250925-153222", "drama.db.bak-20250601-080000"],
              names(removed))
        check("① 盘上只剩最新三份",
              names(up._backups_of(db)) == ["drama.db.bak-20261107-101010",
                                            "drama.db.bak-20260401-120000",
                                            "drama.db.bak-20260101-000000"],
              names(up._backups_of(db)))
        check("① 删掉的文件真不在盘上（不是只少了个返回值）",
              not (root / "drama.db.bak-20250925-153222").exists())
        check("① 主库本身没被碰", db.exists() and db.stat().st_size > 0)

    # ── ② 排序读时间戳、不读 mtime ──
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db = make_db(root)
        make_bak(db, "20250101-000000", mtime=2_000_000_000)   # 最老的时间戳，最新颖的 mtime
        make_bak(db, "20261107-000000", mtime=1_000_000_000)
        make_bak(db, "20260601-000000", mtime=1_500_000_000)
        make_bak(db, "20260301-000000", mtime=1_700_000_000)
        removed = up._prune_backups(db)
        check("② mtime 最新但时间戳最老的那份照样被删（判据是文件名不是 mtime）",
              names(removed) == ["drama.db.bak-20250101-000000"], names(removed))

    # ── ③ 先备份再裁剪 ⇒ 刚做的那份不会被自己删掉 ──
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db = make_db(root)
        for stamp in ["20250101-000000", "20250301-000000", "20250501-000000"]:
            make_bak(db, stamp)
        fresh = up._backup(db)
        removed = up._prune_backups(db)
        check("③ 新鲜备份仍在（裁剪排在新建之后）", fresh.exists(), fresh.name)
        check("③ 只删到最老那一份、其余不动",
              names(removed) == ["drama.db.bak-20250101-000000"], names(removed))
        check("③ 恰好 N 份时一份不删",
              up._prune_backups(db) == [], names(up._backups_of(db)))
        check("③ `_backup` 造的备份真含库内容（sqlite backup API 而非空文件）",
              fresh.stat().st_size > 0 and fresh.stat().st_size != 6, fresh.stat().st_size)

    # ── ④ 只认自己造的形状：别人的备份 / 目录 / wal 文件一律不碰也不计数 ──
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db = make_db(root)
        for stamp in ["20250101-000000", "20250201-000000", "20250301-000000", "20250401-000000"]:
            make_bak(db, stamp)
        manual = make_bak(db, "manual-keep")                    # 人手命名
        short = make_bak(db, "2025")                            # 形状不全
        wal = db.with_name("drama.db-wal")
        wal.write_bytes(b"")
        shm = db.with_name("drama.db-shm")
        shm.write_bytes(b"")
        folder = root / "_backup_20260101-000000"               # 整套备份（目录）
        folder.mkdir()
        (folder / "drama.db").write_bytes(b"whole")
        other = root / "other.db.bak-20250101-000000"           # 别的库
        other.write_bytes(b"other")

        check("④ 认出的只有 4 份自己造的（别人的一律不计数）",
              len(up._backups_of(db)) == 4, names(up._backups_of(db)))
        removed = up._prune_backups(db)
        check("④ 只删自己造的最旧那份",
              names(removed) == ["drama.db.bak-20250101-000000"], names(removed))
        check("④ 人手命名的 .bak 不删", manual.exists(), manual.name)
        check("④ 形状不全的 .bak 不删", short.exists(), short.name)
        check("④ -wal / -shm 不删", wal.exists() and shm.exists())
        check("④ 整套备份目录还在（连里面的库也在）",
              folder.is_dir() and (folder / "drama.db").read_bytes() == b"whole")
        check("④ 别的库的备份不串", other.exists(), other.name)

    # ── ⑤ 反向验证：keep 真被用上 + 删不掉不拖垮 ──
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db = make_db(root)
        for stamp in ["20250101-000000", "20250201-000000", "20250301-000000"]:
            make_bak(db, stamp)
        removed = up._prune_backups(db, keep=1)
        check("⑤ keep=1 ⇒ 只留最新一份（常量真被用上）",
              len(removed) == 2 and names(up._backups_of(db)) == ["drama.db.bak-20250301-000000"],
              names(up._backups_of(db)))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db = make_db(root)
        for stamp in ["20250101-000000", "20250201-000000", "20250301-000000", "20250401-000000"]:
            make_bak(db, stamp)
        original = Path.unlink

        def boom(self: Path, *args: object, **kwargs: object) -> None:
            raise PermissionError(32, "被别的进程占着")

        Path.unlink = boom  # type: ignore[method-assign]
        try:
            removed = up._prune_backups(db)
        finally:
            Path.unlink = original  # type: ignore[method-assign]
        check("⑤ 删不掉时返回空表、不抛（补列是主任务）", removed == [], names(removed))
        check("⑤ 删不掉时备份全都还在", len(up._backups_of(db)) == 4)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed_ok, detail in _RESULTS:
        print(("PASS  " if passed_ok else "FAIL  ") + name
              + ("" if passed_ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
