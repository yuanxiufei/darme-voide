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

from app.core import config  # noqa: E402
from app.core import db  # noqa: E402
from app.main import app  # noqa: E402
from app.services import data_storage as ds  # noqa: E402

# ⚠️ **必须用 `db.engine`（模块属性），不能 `from app.core.db import engine`**：
#    `reopen_engine()` 换的是 `app.core.db.engine` 这个**名字**，而 `from ... import` 拿到的是
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
    # ⚠️ **必须先造出可复制的内容**：迁移分支里 `static/`、`traces/`、`lora-train/` 都是
    #    「存在才复制」，夹具空着的话打桩的 `_copy_tree` 根本不会被调用 ⇒ 迁移反而成功、
    #    断言全错位（我第一版就踩了这个：回滚用例「通过」了，实际是它压根没跑到迁移）。
    _seed(_ROOT0, "static/a.png", "IMG")
    _seed(_ROOT0, "traces/run.jsonl", '{"t":1}')
    _seed(_ROOT0, "lora-train/settings.toml", "LR")
    # 画布草案：用**真形状**的 id（`canvas_plan_store.PLAN_ID_PATTERN` = 16 位十六进制），
    # 这样夹具与线上 `canvas/plans/<id>.json` 同构，不会测到一个线上不存在的路径。
    _seed(_ROOT0, "canvas/plans/0123456789abcdef.json", '{"nodes":[]}')
    # 知识库：夹具用**真文件名**（`knowledge_ledger.db_path()` 与 `training_registry` 的落点
    # 就是这两个），与 lora-train / canvas 同款同构写法。
    _seed(_ROOT0, "knowledge/knowledge-ledger.sqlite", "LEDGER")
    _seed(_ROOT0, "knowledge/training-registry.sqlite", "REGISTRY")
    # ⚠️ 库在 ``<数据根>/db/`` 下（2026-11-07 分层：顶层只放目录、不放散文件）——
    #    这行曾是 ``_ROOT0 / "drama.db"``，切根后当场 FileNotFoundError。
    old_db_size = (_ROOT0 / "db" / "drama.db").stat().st_size

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
          Path(db.engine.url.database).resolve() == (_ROOT0 / "db" / "drama.db").resolve(),
          db.engine.url.database)

    # ── ④ 正常切换 + 迁移（复制，旧目录保留） ──
    info = ds.change_data_root(str(_TARGET_A))

    check("迁移: 数据根切到目标目录", Path(info["dataRoot"]).resolve() == _TARGET_A, info["dataRoot"])
    # ⚠️ 每项先 `is_file()` 再读：漏复制时**必须报红**，不能 `read_text` 抛
    #    `FileNotFoundError` 把整套炸掉 —— 那样反向验证只会看到一条 traceback、
    #    后面的判据全不跑（2026-11-07 实测踩到）。
    def _content(path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    check("迁移: db / static / traces / lora-train / canvas / knowledge 六样都复制到新根",
          (_TARGET_A / "db" / "drama.db").is_file()
          and _content(_TARGET_A / "static" / "a.png") == "IMG"
          and _content(_TARGET_A / "traces" / "run.jsonl") == '{"t":1}'
          and _content(_TARGET_A / "lora-train" / "settings.toml") == "LR"
          and _content(_TARGET_A / "canvas" / "plans" / "0123456789abcdef.json")
          == '{"nodes":[]}'
          and _content(_TARGET_A / "knowledge" / "knowledge-ledger.sqlite") == "LEDGER"
          and _content(_TARGET_A / "knowledge" / "training-registry.sqlite") == "REGISTRY",
          sorted(p.name for p in _TARGET_A.iterdir()))
    # ⚠️ 「六样」里 `lora-train` 是 2026-11-07 补的：手调的参数 + 训出来的权重**不可重建**
    #    （重训是几小时卡时），而 `lora_train.config.data_dir()` 就是 `<dataRoot>/lora-train`
    #    且会 `mkdir` ⇒ 不同步的话新根下是个**空目录**，用户看到「训练集和权重全没了」，
    #    旧数据还在旧根却没人指向 ✓✗。对照：`models/` / `hf-cache/` 能重下、
    #    `materialized/` 能重算 ⇒ 那几个故意不搬。
    # ⚠️ `canvas` 同理（2026-11-08 补）：草案是用户一场一场摆出来的节点与连线，**无源可再生**；
    #    `canvas_plan_store.plans_dir()` 动态读 `get_data_root()` ⇒ 不搬的话切根瞬间
    #    「画布全空了」，且旧草案还躺在旧根（用户以为坏了，其实是被丢下）。
    # ⚠️ `knowledge`（2026-11-08 同日补，是**改判**不是新增认知）：原先这里写着
    #    「`knowledge/` 能重算 ⇒ 故意不搬」—— **错了**。那个目录里四个库只有两个可重建
    #    （`outcome-index` 能从 traces `rebuild()`、`skill-usage` 从跑次重算），另两个是**真源**：
    #    `knowledge-ledger.sqlite`（Agent 蒸馏 + 人审累积的条目／版本／审计）与
    #    `training-registry.sqlite`（哪个 LoRA 被 promote、依据与凭据）。两个模块的 `db_path()`
    #    都现算 `get_data_root()` ⇒ 不搬的症状与 canvas 逐字相同：「切个目录，积累的经验全空了」。
    #    教训是**别按目录名判、要按目录里的东西判** —— 名字像缓存、里面有真源，这一格只能靠读落点。

    # ⚠️ **别比主库文件体积**：WAL 模式 + `close_engine()` 会触发检查点 ⇒ 同一个库在
    #    「复制前 / 复制后」体积都会变（本次实测 4096 → 159744）。要比就比**逻辑等价**。
    check("迁移: 是**复制**不是移动（旧库、旧 static、旧 lora-train 都还在）",
          (_ROOT0 / "db" / "drama.db").is_file()
          and (_ROOT0 / "static" / "a.png").is_file()
          and (_ROOT0 / "lora-train" / "settings.toml").is_file(),
          sorted(p.name for p in _ROOT0.iterdir()))
    # ── ⑨ 复制清单是**双向**的：该搬的真搬了（上面 ④），不该搬的不许混进来 ──
    # 白名单从 3 个 if 提成 `data_storage.MIGRATED_TREE_SUBDIRS` 之后，这里同时钉住
    # 「表里是哪些」与「表里不许有哪些」—— 后者防的是有人为"补齐"把 models/hf-cache
    # 也加进去：一次切根白搬几十 GB，而这些目录不搬**不会丢东西** ✓。
    check("⑨ 整目录搬的白名单＝('traces','lora-train','canvas','knowledge')（db/static/覆盖层各有覆盖口，单独算）",
          ds.MIGRATED_TREE_SUBDIRS == ("traces", "lora-train", "canvas", "knowledge"),
          ds.MIGRATED_TREE_SUBDIRS)
    check("⑨ 能重下/重算/重跑的目录**不许**进白名单（也不许重复收录 static/db）",
          not set(ds.MIGRATED_TREE_SUBDIRS)
          & {"models", "hf-cache", "materialized", "engine-dryrun", "repair",
             "static", "db"},
          ds.MIGRATED_TREE_SUBDIRS)
    # ── ⑨′ 反向守卫：不问「表里写了哪些」，问「已知的真源切根后还在不在」──
    # 上面两条判的是**表的内容**，所以漏一个真源只能靠人记得改那张表（`knowledge/` 就是这么漏的）。
    # 这条改成现算各模块 ``db_path()`` 的**父目录名**：凡是落在白名单外的真源，当场报红。
    # 将来任何模块把真源挪去新目录、或新开一个真源目录，这里都会自己亮，不必依赖记性 ✓✗
    from app.services import knowledge_ledger, training_registry as _training_registry
    truth_paths = {"knowledge_ledger": knowledge_ledger.db_path(),
                   "training_registry": _training_registry.db_path()}
    missed = {name: path.as_posix() for name, path in truth_paths.items()
              if path.parent.name not in ds.MIGRATED_TREE_SUBDIRS}
    check("⑨′ 知识台账 / 训练登记这两个真源库，其父目录必须在整目录搬的白名单里",
          not missed, missed or {k: v.as_posix() for k, v in truth_paths.items()})
    old_engine = create_engine(f"sqlite+pysqlite:///{(_ROOT0 / 'db' / 'drama.db').as_posix()}")
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
          Path(config.get_db_path()).resolve() == (_TARGET_A / "db" / "drama.db").resolve()
          and Path(config.get_storage_root()).resolve() == (_TARGET_A / "static").resolve(),
          (config.get_db_path(), config.get_storage_root()))
    check("切换后: engine 指向新库（reopen_engine 生效）",
          Path(db.engine.url.database).resolve() == (_TARGET_A / "db" / "drama.db").resolve(),
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

    # ── ⑩ 空间预检：装不下就当场拒，而且**一点状态都不许动** ──
    # 打桩真正的 `shutil.disk_usage`（`ds.shutil` 就是那个模块对象本身）⇒ **必须还原**，
    # 否则后面的用例全按 1 KiB 可用空间算。
    import shutil as _shutil  # noqa: PLC0415

    _real_usage = _shutil.disk_usage
    _before_root = config.get_data_root()
    _shutil.disk_usage = lambda _path: type("_Tiny", (), {"free": 1024})()
    try:
        try:
            ds.change_data_root(str(_SANDBOX / "target_full"), True)
            _space_err = None
        except ValueError as exc:
            _space_err = str(exc)
    finally:
        _shutil.disk_usage = _real_usage
    check("⑩ 目标盘装不下 -> 报「可用空间不足」并给出需求/可用两个体积",
          isinstance(_space_err, str) and "可用空间不足" in _space_err
          and "需要约" in _space_err and "可用" in _space_err, _space_err)
    check("⑩ 拒之前状态一点没动（根未切、库未建）",
          config.get_data_root() == _before_root
          and not (_SANDBOX / "target_full" / "db" / "drama.db").exists(),
          (config.get_data_root(), _before_root))

    # 反方向同样要守住：**取不到可用空间**（网络盘/特殊文件系统）时不许拦 ——
    # 误拒一个合法目录比「没预检」更糟（用户完全无法切换，且报错指向不到真因）。
    _shutil.disk_usage = lambda _path: (_ for _ in ()).throw(OSError("no such device"))
    try:
        _net_info = ds.change_data_root(str(_SANDBOX / "target_netdisk"), True)
    finally:
        _shutil.disk_usage = _real_usage
    check("⑩ 取不到可用空间 -> 不拦，照常切换",
          Path(_net_info["dataRoot"]).resolve() == (_SANDBOX / "target_netdisk").resolve(),
          _net_info)

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
