"""S7 自检：数据根目录切换（``data-storage.ts`` → `data_storage.py`）
+ ``GET /storage/info`` / ``POST /storage/change`` 两条端点（**整域关闭**）。

这是「删 ``backend/``」的**最后一块阻塞**：它是唯一未注册端点，Node 下线后单后端必须自己提供，
否则前端设置页会直接 501 ⇒ 行为要逐条钉住，尤其是「**迁移是复制不是移动**」「失败必须回滚」
「``migrate !== false`` 只认字面量 false」这三条容易写歪的语义。

⚠️ **安全前提**：真实项目根的 ``.data-root`` 标记文件**绝不能被写**。故整个过程把
``config.PROJECT_ROOT`` 与 ``config.DATA_ROOT_MARKER`` 都指向临时目录，并在末尾断言
真项目根那个文件**内容未变**。

运行::

    ./.venv/Scripts/python.exe tests/storage_change_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="storechg_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, inspect  # noqa: E402

from app import config  # noqa: E402
from app import db  # noqa: E402
from app.main import app  # noqa: E402
from app.services import data_storage as ds  # noqa: E402

# ⚠️ **必须用 `db.engine`（模块属性），不能 `from app.db import engine`**：
#    `reopen_engine()` 换的是 `app.db.engine` 这个**名字**，而 `from ... import` 拿到的是
#    导入那一刻的对象**快照** ⇒ 用它断言「已指向新库」会永远看到旧库（我第一版就这么错，
#    还让「新库已建表」那条**假通过**了：它在旧库上当然有表）。

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _expect_error(message: str, call) -> tuple[bool, str]:
    """服务层错误：**文案必须逐字一致**（TS 的 `throw new Error(...)`）。"""
    try:
        call()
    except ValueError as exc:
        return str(exc) == message, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"抛了非 ValueError：{exc!r}"
    return False, "没有抛错"


# ── 沙箱：假项目根 + 假标记文件路径（真仓库标记文件全程只读） ──
_SANDBOX = Path(tempfile.mkdtemp(prefix="storechg_sandbox_")).resolve()
_FAKE_PROJECT = _SANDBOX / "proj"
_REAL_PROJECT_ROOT = config.PROJECT_ROOT
_REAL_MARKER = config.DATA_ROOT_MARKER
_REAL_MARKER_BEFORE = _REAL_MARKER.read_text(encoding="utf-8") if _REAL_MARKER.exists() else None

_FAKE_PROJECT.mkdir(parents=True, exist_ok=True)
for _name in ("backend", "frontend", "configs", "node_modules", ".git"):
    (_FAKE_PROJECT / _name).mkdir(parents=True, exist_ok=True)
config.PROJECT_ROOT = _FAKE_PROJECT
config.DATA_ROOT_MARKER = _FAKE_PROJECT / ".data-root"

_ROOT0 = Path(config.get_data_root()).resolve()
_TARGET_A = (_SANDBOX / "target_a").resolve()
_TARGET_B = (_SANDBOX / "target_b").resolve()
_TARGET_C = (_SANDBOX / "target_c").resolve()
_TARGET_D = (_SANDBOX / "target_d").resolve()


def _seed(path: Path, relative: str, payload: str) -> None:
    target = path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")


def main() -> int:
    # ── ① 服务层校验（都不改状态） ──
    for raw in ("", "   ", None):
        ok, detail = _expect_error("请填写有效的目录路径", lambda r=raw: ds.change_data_root(r))
        check(f"校验: 空白路径 {raw!r} -> 请填写有效的目录路径", ok, detail)

    ok, detail = _expect_error("新目录与当前目录相同，无需切换",
                               lambda: ds.change_data_root(str(_ROOT0)))
    check("校验: 与当前目录相同（比较的是 resolve 后的路径）", ok, detail)

    ok, detail = _expect_error("数据目录不能设置为项目根目录",
                               lambda: ds.change_data_root(str(config.PROJECT_ROOT)))
    check("校验: 项目根目录被拒", ok, detail)

    for name in ("backend", "frontend", "configs", "node_modules", ".git"):
        ok, detail = _expect_error(
            "数据目录不能设置在项目源码/依赖/配置目录内",
            lambda n=name: ds.change_data_root(str(config.PROJECT_ROOT / n)))
        check(f"校验: 项目内 {name}/ 被拒", ok, detail)
    ok, detail = _expect_error(
        "数据目录不能设置在项目源码/依赖/配置目录内",
        lambda: ds.change_data_root(str(config.PROJECT_ROOT / "backend" / "deep" / "nested")))
    check("校验: 项目内目录的**子目录**同样被拒（前缀判定）", ok, detail)

    original_probe = ds._write_probe
    ds._write_probe = lambda _directory: (_ for _ in ()).throw(OSError("denied"))
    try:
        ok, detail = _expect_error("目录不可写，请检查权限或更换目录",
                                   lambda: ds.change_data_root(str(_TARGET_A)))
    finally:
        ds._write_probe = original_probe
    check("校验: 探针写失败 -> 目录不可写（且**未**切换数据根）",
          ok and Path(config.get_data_root()).resolve() == _ROOT0, detail)

    # ── ② copy_db_files 单元（含 -wal / -shm 伴生文件；不碰真库，故不担心 WAL 内容） ──
    # ⚠️ 目标文件名用的是**源文件的 basename**（TS 写的是 `path.join(dirname(newDb), basename(src))`）——
    #    真实调用里两边都叫 `drama.db` 所以看不出来；夹具**故意沿用同名**，别用 `old.db -> new.db`
    #    这种异名夹具，那会测到一个 TS 根本不会走的分支（我第一版就这么写的，被实测打回）。
    unit = _SANDBOX / "unit"
    unit.mkdir(parents=True, exist_ok=True)
    (unit / "drama.db").write_bytes(b"DB")
    (unit / "drama.db-wal").write_bytes(b"WAL")
    (unit / "drama.db-shm").write_bytes(b"SHM")
    ds.copy_db_files(str(unit / "drama.db"), str(unit / "sub" / "drama.db"))
    check("copy_db_files: 库文件 + -wal + -shm 三个都复制（缺哪个跳哪个）",
          [(unit / "sub" / n).read_bytes() for n in ("drama.db", "drama.db-wal", "drama.db-shm")]
          == [b"DB", b"WAL", b"SHM"], sorted(p.name for p in (unit / "sub").iterdir()))
    ds.copy_db_files(str(unit / "missing.db"), str(unit / "sub2" / "drama.db"))
    check("copy_db_files: 源不存在也不抛（只建目录）",
          (unit / "sub2").is_dir() and not any((unit / "sub2").iterdir()))

    # ── ③ 迁移失败必须回滚（数据根与 engine 都不动） ──
    # ⚠️ **必须先造出可复制的内容**：迁移分支里 `static/` 与 `traces/` 都是「存在才复制」，
    #    夹具空着的话打桩的 `_copy_tree` 根本不会被调用 ⇒ 迁移反而成功、断言全错位
    #    （我第一版就踩了这个：回滚用例「通过」了，实际是它压根没跑到迁移）。
    _seed(_ROOT0, "static/a.png", "IMG")
    _seed(_ROOT0, "traces/run.jsonl", '{"t":1}')
    old_db_size = (_ROOT0 / "drama.db").stat().st_size

    original_tree = ds._copy_tree
    ds._copy_tree = lambda _s, _t: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        ok, detail = _expect_error("迁移失败：boom", lambda: ds.change_data_root(str(_TARGET_A)))
    finally:
        ds._copy_tree = original_tree
    check("回滚: 迁移抛错 -> 包成「迁移失败：…」", ok, detail)
    check("回滚: 数据根**仍是旧根**", Path(config.get_data_root()).resolve() == _ROOT0,
          config.get_data_root())
    check("回滚: engine 重新指回旧库",
          Path(db.engine.url.database).resolve() == (_ROOT0 / "drama.db").resolve(),
          db.engine.url.database)

    # ── ④ 正常切换 + 迁移（复制，旧目录保留） ──
    info = ds.change_data_root(str(_TARGET_A))

    check("迁移: 数据根切到目标目录", Path(info["dataRoot"]).resolve() == _TARGET_A, info["dataRoot"])
    check("迁移: db / static / traces 三样都复制到新根",
          (_TARGET_A / "drama.db").is_file()
          and (_TARGET_A / "static" / "a.png").read_text(encoding="utf-8") == "IMG"
          and (_TARGET_A / "traces" / "run.jsonl").read_text(encoding="utf-8") == '{"t":1}',
          sorted(p.name for p in _TARGET_A.iterdir()))
    # ⚠️ **别比主库文件体积**：WAL 模式 + `close_engine()` 会触发检查点 ⇒ 同一个库在
    #    「复制前 / 复制后」体积都会变（本次实测 4096 → 159744）。要比就比**逻辑等价**。
    check("迁移: 是**复制**不是移动（旧库与旧 static 都还在）",
          (_ROOT0 / "drama.db").is_file() and (_ROOT0 / "static" / "a.png").is_file(),
          sorted(p.name for p in _ROOT0.iterdir()))
    old_engine = create_engine(f"sqlite+pysqlite:///{(_ROOT0 / 'drama.db').as_posix()}")
    try:
        check("迁移: 新旧两库**逻辑等价**（都能打开且表结构在）",
              inspect(old_engine).has_table("dramas") and inspect(db.engine).has_table("dramas"),
              [inspect(old_engine).get_table_names()[:3], old_db_size])
    finally:
        old_engine.dispose()
    check("迁移: 返回的 info 键名与 TS 的 StorageInfo 一致（camelCase 7 键）",
          sorted(info) == ["dataRoot", "dbExists", "dbPath", "dbSizeBytes",
                           "storageExists", "storagePath", "storageSizeBytes"], sorted(info))
    check("迁移: info.dbExists / storageExists 都为真",
          info["dbExists"] and info["storageExists"])
    check("切换后: get_db_path / get_storage_root 都跟着变",
          Path(config.get_db_path()).resolve() == (_TARGET_A / "drama.db").resolve()
          and Path(config.get_storage_root()).resolve() == (_TARGET_A / "static").resolve(),
          (config.get_db_path(), config.get_storage_root()))
    check("切换后: engine 指向新库（reopen_engine 生效）",
          Path(db.engine.url.database).resolve() == (_TARGET_A / "drama.db").resolve(),
          db.engine.url.database)
    check("切换后: 新库可用（表结构就位）",
          inspect(db.engine).has_table("dramas"), inspect(db.engine).get_table_names()[:6])
    check("切换后: 标记文件内容 = 新根（重启仍生效）",
          config.DATA_ROOT_MARKER.read_text(encoding="utf-8") == str(_TARGET_A),
          config.DATA_ROOT_MARKER.read_text(encoding="utf-8"))

    # ── ⑤ migrate=False：只切、不复制 ──
    info_b = ds.change_data_root(str(_TARGET_B), False)
    check("migrate=False: 不复制 static（新根里没有 a.png）",
          not (_TARGET_B / "static" / "a.png").exists(), sorted(p.name for p in _TARGET_B.iterdir()))
    check("migrate=False: info.storageExists 为假，但库已新建并建表",
          info_b["storageExists"] is False and info_b["dbExists"] is True
          and inspect(db.engine).has_table("dramas"), info_b)
    check("migrate=False: 旧根内容**原样保留**", (_TARGET_A / "static" / "a.png").is_file())

    # ── ⑥ 端点层（TestClient） ──
    client = TestClient(app)

    response = client.post("/api/v1/storage/change", json={})
    check("端点: 空 body -> 400 且文案是**路由层**的「请填写目标目录路径」",
          response.status_code == 400
          and response.json()["message"] == "请填写目标目录路径", response.text[:120])

    response = client.post("/api/v1/storage/change", json={"path": "   "})
    check("端点: 全空白 path -> 400（同上文案）",
          response.status_code == 400
          and response.json()["message"] == "请填写目标目录路径", response.text[:120])

    # ⚠️ 用**当前**根（不是 `_ROOT0`）：到这一步根已经过 ④⑤ 两轮切换，写 `_ROOT0` 反而会**真的切回去**
    #    （第一版就是这么写的：期望 400 却拿到 200，还把后续用例的根偷偷改掉了）。
    response = client.post("/api/v1/storage/change", json={"path": config.get_data_root()})
    check("端点: 服务层校验失败 -> 400 且用**服务层**文案",
          response.status_code == 400
          and response.json()["message"] == "新目录与当前目录相同，无需切换", response.text[:120])

    response = client.post("/api/v1/storage/change",
                           json={"path": str(_TARGET_C), "migrate": False})
    body = response.json()
    check("端点: 正常切换 -> 200 且信封为 {code,data,message}",
          response.status_code == 200 and body["code"] == 200
          and body["message"] == "success"
          and Path(body["data"]["dataRoot"]).resolve() == _TARGET_C, response.text[:160])

    # `migrate: 0` ⇒ TS 的 `0 !== false` ⇒ **要迁移**（只有字面量 false 才跳过）
    _seed(_TARGET_C, "static/pin.txt", "PIN")
    response = client.post("/api/v1/storage/change", json={"path": str(_TARGET_D), "migrate": 0})
    check("端点: migrate=0 按「要迁移」处理（`!== false` 只认字面量 false）",
          response.status_code == 200
          and (_TARGET_D / "static" / "pin.txt").read_text(encoding="utf-8") == "PIN",
          response.text[:160])

    response = client.post("/api/v1/storage/change", content=b"not-json",
                           headers={"content-type": "application/json"})
    check("端点: 坏 JSON -> 400（走空 dict 后由必填校验拦下，与 TS 状态码一致）",
          response.status_code == 400, response.text[:120])

    response = client.get("/api/v1/storage/info")
    check("端点: GET /info 仍正常（整域 2/2）",
          response.status_code == 200 and response.json()["data"]["dataRoot"] == str(_TARGET_D),
          response.text[:160])

    # ── ⑦ 安全收口：真项目根标记文件从未被写 ──
    check("安全: 真仓库 .data-root **内容未变**（全程只写沙箱）",
          (_REAL_MARKER.read_text(encoding="utf-8") if _REAL_MARKER.exists() else None)
          == _REAL_MARKER_BEFORE, _REAL_MARKER)

    # 还原（同进程后续不再用配置，但保持规矩）
    config.PROJECT_ROOT = _REAL_PROJECT_ROOT
    config.DATA_ROOT_MARKER = _REAL_MARKER

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
