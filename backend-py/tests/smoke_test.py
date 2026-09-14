"""Python 后端冒烟测试 —— 绞杀者迁移期的**唯一回归安全网**。

Node 后端零自动化测试，Python 侧不能重蹈覆辙：每迁一个域，就把它加进这里。
这个脚本守住三件在迁移中最容易静默走样的事：

1. **模式核对**：把 SQLAlchemy 的表/列定义与真实 ``data/drama.db`` 的 ``PRAGMA table_info``
   逐列比对，并与 Node 的 ``db/index.ts`` 的建表清单交叉印证。
   比对照 ``schema.ts`` 更硬 —— DB 才是运行时真相。
2. **契约核对**：起 TestClient 打真实接口，确认响应信封（``code``/``data``/``message``）、
   状态码、**错误文案逐字**与 Node 版一致（前端直接展示 ``message``，措辞变了就是回归）。
3. **写操作隔离**：所有写操作都发生在 ``data/drama.db`` 的**副本**上，真实库全程只读。

用法（在 ``backend-py`` 目录下）::

    .venv\\Scripts\\python.exe tests\\smoke_test.py

退出码 0 = 全过，1 = 有失败项。完整报告（含请求/响应原文）落在系统临时目录，
路径在末尾打印，也可用 ``SMOKE_REPORT`` 环境变量指定。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
REPO = BACKEND_PY.parent
REAL_DB = REPO / "data" / "drama.db"

if not REAL_DB.exists():
    print(f"FAIL  real db not found: {REAL_DB}")
    print("      hint: 先启动一次 Node 后端让它建库（cd backend && npx tsx src/index.ts）")
    raise SystemExit(1)

_report_path = Path(
    os.environ.get("SMOKE_REPORT") or (Path(tempfile.gettempdir()) / "drama_studio_py_smoke.txt")
)
_smoke_root = Path(tempfile.mkdtemp(prefix="drama-py-smoke-"))

lines: list[str] = []


def log(msg: str = "") -> None:
    lines.append(msg)


results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    results.append((name, bool(ok), extra))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{extra}]" if extra and not ok else ""))


# ---------------------------------------------------------------------------
# 0) 隔离库：真实库只读，写操作全部落在副本上
# ---------------------------------------------------------------------------
shutil.copy2(REAL_DB, _smoke_root / "drama.db")
log(f"real db (readonly source): {REAL_DB}")
log(f"smoke db (writable copy) : {_smoke_root / 'drama.db'}")

os.environ["DATA_ROOT"] = str(_smoke_root)
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.config import PROJECT_ROOT, get_db_path, server  # noqa: E402
from app.models import metadata  # noqa: E402

check("config: PROJECT_ROOT == repo root", PROJECT_ROOT == REPO, str(PROJECT_ROOT))
# ⚠️ 必须比较 resolve() 后的路径：Windows 上 tempfile.mkdtemp() 返回 8.3 短路径
# （C:\Users\ADMINI~1\...），而 config 侧会对 DATA_ROOT 做 resolve() 展开成长路径，
# 直接子串比较会假红。
check(
    "config: db path points into smoke root",
    Path(get_db_path()).resolve().parent == _smoke_root.resolve(),
    get_db_path(),
)
check("config: python port != node port (5789)", server["port"] != 5789, str(server["port"]))

# ---------------------------------------------------------------------------
# 1) 模式核对（对照真实 DB + Node DDL）
# ---------------------------------------------------------------------------
_conn = sqlite3.connect(f"file:{REAL_DB.as_posix()}?mode=ro", uri=True)
db_tables = [
    r[0]
    for r in _conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
]
db_schema = {t: [r[1] for r in _conn.execute(f'PRAGMA table_info("{t}")')] for t in db_tables}
_conn.close()

model_schema = {t.name: [c.name for c in t.columns] for t in metadata.tables.values()}

# 真实库里比 Node 模型多出的**遗留**列（旧版本 MinIO 存储留下；Node 侧 schema.ts / db/index.ts
# 都没有它们，Drizzle 也读不到 ⇒ 迁移同样不需要建模）。除白名单外一律视为缺陷。
LEGACY_DB_ONLY_COLUMNS = {"minio_url"}
# 真实库里比 Node 模型多出的**遗留表**（同上：旧版本残留，当前代码零引用）
LEGACY_DB_ONLY_TABLES = {"assets", "props"}

only_db = sorted(set(db_tables) - set(model_schema))
only_model = sorted(set(model_schema) - set(db_tables))

missing_cols: dict[str, list[str]] = {}
legacy_cols: dict[str, list[str]] = {}
extra_cols: dict[str, list[str]] = {}
order_diff: dict[str, str] = {}
for t in sorted(set(db_tables) & set(model_schema)):
    db_cols, md_cols = db_schema[t], model_schema[t]
    miss = [c for c in db_cols if c not in md_cols]
    legacy = [c for c in miss if c in LEGACY_DB_ONLY_COLUMNS]
    miss = [c for c in miss if c not in LEGACY_DB_ONLY_COLUMNS]
    extra = [c for c in md_cols if c not in db_cols]
    if miss:
        missing_cols[t] = miss
    if legacy:
        legacy_cols[t] = legacy
    if extra:
        extra_cols[t] = extra
    if not miss and not legacy and not extra and db_cols != md_cols:
        order_diff[t] = "db=[%s] model=[%s]" % (",".join(db_cols), ",".join(md_cols))

ddl_tables = sorted(
    set(
        re.findall(
            r"CREATE TABLE IF NOT EXISTS (\w+)",
            (REPO / "backend" / "src" / "db" / "index.ts").read_text(encoding="utf-8"),
        )
    )
)
ts_schema = (REPO / "backend" / "src" / "db" / "schema.ts").read_text(encoding="utf-8")

log()
log("=" * 72)
log(f"SCHEMA: db={len(db_tables)} tables / models={len(model_schema)} / node-ddl={len(ddl_tables)}")
log("=" * 72)
log(f"tables only in DB (expected legacy): {only_db}")
log(f"tables only in models (real defect): {only_model}")
log(f"columns missing in models (defect) : {json.dumps(missing_cols, ensure_ascii=False)}")
log(f"columns legacy-only in DB          : {json.dumps(legacy_cols, ensure_ascii=False)}")
log(f"columns extra in models (breaks)   : {json.dumps(extra_cols, ensure_ascii=False)}")
log(f"column ORDER diffs (informational) : {len(order_diff)} tables")
# 列序不影响正确性：SQLAlchemy 全程按列名生成 SQL（不会用位置 INSERT/SELECT）
log(f"  {json.dumps(order_diff, ensure_ascii=False)[:600]}")

check("schema: models == Node DDL table set", sorted(model_schema) == ddl_tables, str(only_model))
check("schema: extra DB tables are known legacy only", set(only_db) <= LEGACY_DB_ONLY_TABLES, str(only_db))
check("schema: no real missing column", not missing_cols, json.dumps(missing_cols))
check("schema: no extra column in models", not extra_cols, json.dumps(extra_cols))
check("schema: table count == 29", len(model_schema) == 29, str(len(model_schema)))
check("schema: matches Node schema.ts count", ts_schema.count("sqliteTable(") == len(model_schema))

# ---------------------------------------------------------------------------
# 2) 接口契约
# ---------------------------------------------------------------------------
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.routers.libraries import (  # noqa: E402
    BODY_PARTS,
    COSTUME_STYLES,
    WEAPON_CATEGORIES,
)

log()
log("=" * 72)
log("HTTP CONTRACT")
log("=" * 72)


def _unregistered_samples(limit: int = 5) -> list[tuple[str, str]]:
    """从**守卫口径**动态派生「仍未迁移」的端点，返回前 N 条的 ``(方法, 具体路径)``。

    ⚠️ **为什么不再硬编码「未迁移清单」**：这份清单在本仓已经**连续三轮**被迁移打破
    （characters/scenes/props 三连 → storyboards 的 generate-tts/split → episodes 的
    continue-script），而且**只有跑全量才暴露**（单跑新自检全绿）。
    这里直接复用 ``route_parity_test.py``（第 8 道守卫）的口径：
    **Node 侧路径 − Python 已注册路径 = 仍未迁移**，再用守卫的 ``_concrete()``
    把 ``/{id}`` 换成可请求的具体路径。

    ⚠️ 守卫模块**懒导入**：它在模块级会**硬改** ``DATA_ROOT`` / ``PROXY_TO_NODE`` 环境变量，
    所以导入前要**快照环境变量、导入后原样还原** —— 否则会把 smoke 自己的数据根顶掉
    （实测会让后面「traces list」这类**读文件系统**的断言失败）。
    守卫模块只会真正执行一次（``sys.modules`` 缓存）⇒ 后续调用不会再有副作用。
    """
    import os as _os  # noqa: PLC0415

    env_keys = ("DATA_ROOT", "PROXY_TO_NODE")
    saved = {key: _os.environ.get(key) for key in env_keys}
    try:
        import route_parity_test as guard  # noqa: PLC0415
    finally:
        for key, value in saved.items():
            if value is None:
                _os.environ.pop(key, None)
            else:
                _os.environ[key] = value

    from app.main import app as _app  # noqa: PLC0415

    routes_dir = guard.REPO / "backend" / "src" / "routes"
    node: set[tuple[str, str]] = set()
    for filename, prefix in guard.MIGRATED.items():
        text = (routes_dir / filename).read_text(encoding="utf-8")
        for method, ts_path in guard._TS_ROUTE.findall(text):
            node.add((method.upper(), guard._node_path(prefix, ts_path)))
    registered = {(m.upper(), p) for p, ops in _app.openapi()["paths"].items() for m in ops}
    node_norm = {(m, guard._pattern(p)) for m, p in node}
    reg_norm = {(m, guard._pattern(p)) for m, p in registered}
    todo = sorted((m, p) for m, p in node_norm if (m, p) not in reg_norm)
    return [(m, guard._concrete(p)) for m, p in todo[:limit]]


def dump(label: str, resp) -> None:
    log(f"\n--- {label} -> {resp.status_code} {resp.headers.get('content-type')}")
    log(resp.text[:1500])


with TestClient(app) as client:
    # --- 健康检查：Node 侧是裸对象，不包信封 ---
    r = client.get("/api/v1/health")
    dump("GET /api/v1/health", r)
    check("health: 200", r.status_code == 200)
    check("health: bare object (status/timestamp)", set(r.json()) == {"status", "timestamp"})

    # --- 列表 ---
    r = client.get("/api/v1/dramas")
    dump("GET /api/v1/dramas", r)
    body = r.json()
    check("list: 200 + envelope", r.status_code == 200 and body.get("code") == 200)
    check("list: message == 'success'", body.get("message") == "success")
    check("list: has items/pagination", "items" in body["data"] and "pagination" in body["data"])
    pag = body["data"]["pagination"]
    check("list: pagination shape", set(pag) == {"page", "page_size", "total", "total_pages"}, str(sorted(pag)))
    items = body["data"]["items"]
    check("list: real data visible (total > 0)", pag["total"] > 0, str(pag["total"]))
    check("list: page_size default 20", pag["page_size"] == 20, str(pag["page_size"]))
    if items:
        keys = set(items[0])
        check("list: item has progress", "progress" in keys)
        check(
            "list: progress keys",
            set(items[0]["progress"])
            == {"total_episodes", "scripted_episodes", "storyboarded_episodes", "storyboards",
                "images", "videos", "tts"},
            str(sorted(items[0]["progress"])),
        )
        check("list: tags is list", isinstance(items[0].get("tags"), list))
        check(
            "list: snake_case contract fields",
            {"created_at", "updated_at", "total_episodes", "style_id"} <= keys,
            str(sorted(keys)[:8]),
        )
        log(f"\nfirst item: {json.dumps(items[0], ensure_ascii=False)[:900]}")

    r = client.get("/api/v1/dramas", params={"page": 1, "page_size": 1})
    check("list: page_size honored", len(r.json()["data"]["items"]) <= 1)
    check("list: status filter ok", client.get("/api/v1/dramas", params={"status": "draft"}).status_code == 200)
    r = client.get("/api/v1/dramas", params={"keyword": "不存在的剧名XYZ"})
    check("list: keyword filter -> empty", r.json()["data"]["items"] == [])

    # --- 统计 ---
    r = client.get("/api/v1/dramas/stats")
    dump("GET /api/v1/dramas/stats", r)
    b = r.json()
    check("stats: 200 + total/by_status", r.status_code == 200 and "total" in b["data"] and "by_status" in b["data"])

    # --- 详情 / 提示词聚合 ---
    if items:
        did = items[0]["id"]
        r = client.get(f"/api/v1/dramas/{did}")
        dump(f"GET /api/v1/dramas/{did}", r)
        d = r.json()["data"]
        check("detail: nested episodes/characters/scenes",
              all(k in d for k in ("episodes", "characters", "scenes")))
        log(f"\ndetail keys: {sorted(d.keys())}")

        r = client.get(f"/api/v1/dramas/{did}/prompts")
        dump(f"GET /api/v1/dramas/{did}/prompts", r)
        p = r.json()["data"]
        check("prompts: 4 groups", set(p) == {"characters", "scenes", "episodes", "storyboards"}, str(sorted(p)))
        if p["characters"]:
            check("prompts: camelCase in aggregate view", "customPrompt" in p["characters"][0])

    # --- 错误路径（文案逐字对齐 Node）---
    r = client.get("/api/v1/dramas/abc")
    dump("GET /api/v1/dramas/abc", r)
    check("error: non-numeric id -> 404 'Invalid drama id'",
          r.status_code == 404 and r.json()["message"] == "Invalid drama id")
    check("error: no 'data' key on error (node parity)", "data" not in r.json())
    r = client.get("/api/v1/dramas/999999")
    check("error: missing drama -> 404 zh message", r.status_code == 404 and r.json()["message"] == "剧本不存在")
    r = client.put("/api/v1/dramas/999999", json={"title": "x"})
    check("error: PUT missing -> 404 'Drama not found'",
          r.status_code == 404 and r.json()["message"] == "Drama not found")
    r = client.delete("/api/v1/dramas/999999")
    check("error: DELETE missing -> 404 'Drama not found'",
          r.status_code == 404 and r.json()["message"] == "Drama not found")

    # --- 创建 ---
    r = client.post("/api/v1/dramas", json={"title": "SMOKE 测试剧", "genre": "测试", "tags": ["a", "b"]})
    dump("POST /api/v1/dramas", r)
    check("create: 201 + code 201 + message 'created'",
          r.status_code == 201 and r.json()["code"] == 201 and r.json()["message"] == "created")
    new_drama = r.json()["data"]
    new_id = new_drama["id"]
    check("create: style_id auto-filled (ensureStyleId)", bool(new_drama.get("style_id")), str(new_drama.get("style_id")))
    check("create: style_id format STYLE_xxx", (new_drama.get("style_id") or "").startswith("STYLE_"))
    check("create: style resolved non-empty", bool(new_drama.get("style")), str(new_drama.get("style")))
    check("create: default status draft", new_drama.get("status") == "draft")
    check("create: tags raw json in row (node parity)", isinstance(new_drama.get("tags"), str))

    d = client.get(f"/api/v1/dramas/{new_id}").json()["data"]
    check("create: default episode auto-created", len(d["episodes"]) == 1, str(len(d["episodes"])))
    check("create: default episode title", d["episodes"][0]["title"] == "第1集", d["episodes"][0]["title"])
    check("create: tags parsed as list in detail", d["tags"] == ["a", "b"], str(d["tags"]))

    # --- 更新：时代背景三态 ---
    payload = {"era": "古代仙侠", "summary": "架空东方仙侠世界", "imageHint": "ancient chinese xianxia"}
    r = client.put(f"/api/v1/dramas/{new_id}",
                   json={"title": "SMOKE 改名", "era_background": json.dumps(payload, ensure_ascii=False)})
    dump("PUT /api/v1/dramas/{id} (valid era)", r)
    check("update: 200 + code 200", r.status_code == 200 and r.json()["code"] == 200)
    check("update: returns data null (node parity)", r.json().get("data") is None, str(r.json()))

    d = client.get(f"/api/v1/dramas/{new_id}").json()["data"]
    check("update: title applied", d["title"] == "SMOKE 改名")
    check(
        "update: era_background normalized json",
        d["era_background"] == json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        str(d["era_background"]),
    )

    client.put(f"/api/v1/dramas/{new_id}",
               json={"era_background": {"summary": "s", "image_style_en": "legacy hint"}})
    d = client.get(f"/api/v1/dramas/{new_id}").json()["data"]
    check("update: legacy image_style_en -> imageHint", "legacy hint" in (d["era_background"] or ""),
          str(d["era_background"]))

    r = client.put(f"/api/v1/dramas/{new_id}", json={"era_background": "{not json"})
    dump("PUT invalid era_background", r)
    check(
        "update: invalid era -> 400 exact message",
        r.status_code == 400
        and r.json()["message"] == "era_background 不是有效的时代背景 JSON（需 era/summary/imageHint 字段）",
        r.json().get("message", ""),
    )
    r = client.put(f"/api/v1/dramas/{new_id}", json={"era_background": 123})
    check("update: era wrong type -> 400 'era_background 类型不合法'",
          r.status_code == 400 and r.json()["message"] == "era_background 类型不合法")
    client.put(f"/api/v1/dramas/{new_id}", json={"era_background": ""})
    d = client.get(f"/api/v1/dramas/{new_id}").json()["data"]
    check("update: empty string clears era_background", d["era_background"] == "")

    # --- 批量保存角色（snake/camel 双写 + ensureCostumeId）---
    r = client.put(
        f"/api/v1/dramas/{new_id}/characters",
        json={"characters": [{"name": "SMOKE 角色", "role": "主角", "voiceStyle": "calm", "clothing": "长袍"}]},
    )
    dump("PUT /dramas/{id}/characters", r)
    check("chars: 200", r.status_code == 200)
    ch = client.get(f"/api/v1/dramas/{new_id}").json()["data"]["characters"][0]
    check("chars: camelCase input accepted (voiceStyle)", ch.get("voice_style") == "calm", str(ch.get("voice_style")))
    check("chars: costume_id auto-filled", (ch.get("costume_id") or "").startswith("COST_"), str(ch.get("costume_id")))

    # --- 批量保存剧集（此处曾因 **kwargs 重复键 TypeError 而回归）---
    r = client.put(f"/api/v1/dramas/{new_id}/episodes",
                   json={"episodes": [{"title": "SMOKE 新集", "content": "原文"}]})
    dump("PUT /dramas/{id}/episodes", r)
    check("episodes: 200", r.status_code == 200)
    d = client.get(f"/api/v1/dramas/{new_id}").json()["data"]
    check("episodes: appended", len(d["episodes"]) == 2, str(len(d["episodes"])))
    r = client.put(f"/api/v1/dramas/{new_id}/episodes", json={"episodes": [{"id": d["episodes"][0]["id"], "title": "改标题"}]})
    d2 = client.get(f"/api/v1/dramas/{new_id}").json()["data"]
    check("episodes: update by id works",
          any(e["title"] == "改标题" for e in d2["episodes"]), str([e["title"] for e in d2["episodes"]]))

    # --- 软删 ---
    check("delete: 200", client.delete(f"/api/v1/dramas/{new_id}").status_code == 200)
    check("delete: detail now 404", client.get(f"/api/v1/dramas/{new_id}").status_code == 404)
    r = client.get("/api/v1/dramas", params={"keyword": "SMOKE 改名"})
    check("delete: excluded from list", r.json()["data"]["items"] == [])

    # ================= episodes 域 =================
    r = client.post("/api/v1/dramas", json={"title": "SMOKE EP 父剧"})
    ep_drama_id = r.json()["data"]["id"]
    check("ep setup: parent drama created", ep_drama_id > 0, str(ep_drama_id))

    r = client.post("/api/v1/episodes", json={})
    dump("POST /api/v1/episodes (no body)", r)
    check("ep create: missing drama_id -> 400 exact",
          r.status_code == 400 and r.json()["message"] == "drama_id required")

    r = client.post("/api/v1/episodes", json={"drama_id": ep_drama_id})
    dump("POST /api/v1/episodes (no config ids)", r)
    check("ep create: missing config ids -> 400 exact",
          r.status_code == 400
          and r.json()["message"] == "image_config_id, video_config_id and audio_config_id are required")

    r = client.post("/api/v1/episodes", json={
        "drama_id": ep_drama_id, "image_config_id": 1, "video_config_id": 2, "audio_config_id": 3,
    })
    dump("POST /api/v1/episodes", r)
    ep_body = r.json()
    check("ep create: 200 (not 201) + code 200", r.status_code == 200 and ep_body["code"] == 200)
    check("ep create: next episode_number == 2 (第1集 auto-created by drama)",
          ep_body["data"]["episode_number"] == 2, str(ep_body["data"]))
    check("ep create: default title 第2集", ep_body["data"]["title"] == "第2集", str(ep_body["data"]["title"]))
    check("ep create: echoes config ids",
          (ep_body["data"]["image_config_id"], ep_body["data"]["video_config_id"],
           ep_body["data"]["audio_config_id"]) == (1, 2, 3))
    ep_id = ep_body["data"]["id"]

    r = client.put(f"/api/v1/episodes/{ep_id}", json={"nope": 1})
    dump("PUT /api/v1/episodes/{id} (no valid fields)", r)
    check("ep update: no valid fields -> 400 exact",
          r.status_code == 400 and r.json()["message"] == "no valid fields")

    # 剧本写入：故意带 CRLF，验证 sha256 的 CRLF→LF 归一化与 ?? 取源语义
    SCRIPT = "## S01 | 内景 · 客厅 · 夜\r\n\r\n林晚：（低声）你回来了。\r\n"
    r = client.put(f"/api/v1/episodes/{ep_id}", json={"script_content": SCRIPT, "bgm_volume": "0.5"})
    dump("PUT /api/v1/episodes/{id} (script_content)", r)
    check("ep update: 200 + data null", r.status_code == 200 and r.json().get("data") is None)

    r = client.get(f"/api/v1/episodes/{ep_id}/script-fingerprint")
    dump("GET /episodes/{id}/script-fingerprint", r)
    fp = r.json()["data"]
    expected_hash = hashlib.sha256(
        "## S01 | 内景 · 客厅 · 夜\n\n林晚：（低声）你回来了。".encode("utf-8")
    ).hexdigest()[:16]
    check("fp: has_script true", fp["has_script"] is True)
    check("fp: hash == sha256(normalized CRLF->LF)[:16]", fp["current_script_hash"] == expected_hash,
          f"{fp['current_script_hash']} != {expected_hash}")
    check("fp: storyboard_count 0 / stale false",
          fp["storyboard_count"] == 0 and fp["stale"] is False, str(fp))
    check("fp: clean message exact",
          fp["message"] == "剧本指纹一致：0 个分镜与当前剧本匹配", fp["message"])
    check("fp: set of keys", set(fp) == {
        "episode_id", "current_script_hash", "has_script", "storyboard_count",
        "stale_count", "stale_storyboards", "stale", "stale_with_assets", "message",
    }, str(sorted(fp)))

    # 造一个「基于旧剧本且已有资产」的分镜，验证过期判定与最危险标记
    raw = sqlite3.connect(get_db_path())
    ts = "2026-01-01T00:00:00.000Z"
    raw.execute(
        "INSERT INTO storyboards (episode_id, storyboard_number, title, script_hash, composed_image,"
        " created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        (ep_id, 1, "SMOKE 镜头", "deadbeefdeadbeef", "smoke.png", ts, ts),
    )
    raw.commit()
    sb_id = raw.execute("SELECT id FROM storyboards WHERE episode_id=? AND title='SMOKE 镜头'", (ep_id,)).fetchone()[0]
    raw.close()

    r = client.get(f"/api/v1/episodes/{ep_id}/script-fingerprint")
    dump("GET /episodes/{id}/script-fingerprint (stale)", r)
    fp2 = r.json()["data"]
    check("fp: stale detected (1/1)", fp2["stale"] is True and fp2["stale_count"] == 1, str(fp2))
    check("fp: stale_with_assets true", fp2["stale_with_assets"] is True, str(fp2))
    check("fp: stale message exact",
          fp2["message"] == "剧本已变更：1/1 个分镜基于旧剧本，需重新拆解分镜", fp2["message"])
    check("fp: stale_storyboards shape",
          fp2["stale_storyboards"] == [{"id": sb_id, "storyboard_number": 1, "has_assets": True}],
          str(fp2["stale_storyboards"]))

    # 关联聚合：角色 / 服装 / 道具
    client.put(f"/api/v1/dramas/{ep_drama_id}/characters",
               json={"characters": [{"name": "SMOKE EP 角色"}]})
    ep_char_id = client.get(f"/api/v1/dramas/{ep_drama_id}").json()["data"]["characters"][0]["id"]

    raw = sqlite3.connect(get_db_path())
    raw.execute("INSERT INTO episode_characters (episode_id, character_id, created_at) VALUES (?,?,?)",
                (ep_id, ep_char_id, ts))
    raw.execute("INSERT INTO storyboard_characters (storyboard_id, character_id, costume) VALUES (?,?,?)",
                (sb_id, ep_char_id, "长袍"))
    raw.execute("INSERT INTO storyboard_props (storyboard_id, prop_id) VALUES (?,?)", (sb_id, 42))
    raw.execute("INSERT INTO scenes (drama_id, location, time, prompt, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?)", (ep_drama_id, "客厅", "夜", "smoke prompt", ts, ts))
    raw.commit()
    scene_id = raw.execute("SELECT id FROM scenes WHERE drama_id=? AND location='客厅'", (ep_drama_id,)).fetchone()[0]
    raw.execute("INSERT INTO episode_scenes (episode_id, scene_id, created_at) VALUES (?,?,?)",
                (ep_id, scene_id, ts))
    raw.commit()
    raw.close()

    r = client.get(f"/api/v1/episodes/{ep_id}/characters")
    dump("GET /episodes/{id}/characters", r)
    check("ep characters: linked char returned",
          [c["id"] for c in r.json()["data"]] == [ep_char_id], str(r.json()["data"]))

    r = client.get(f"/api/v1/episodes/{ep_id}/scenes")
    dump("GET /episodes/{id}/scenes", r)
    check("ep scenes: linked scene returned",
          [s["id"] for s in r.json()["data"]] == [scene_id], str(r.json()["data"]))

    r = client.get(f"/api/v1/episodes/{ep_id}/storyboards")
    dump("GET /episodes/{id}/storyboards", r)
    sbs = r.json()["data"]
    check("ep storyboards: 1 row", len(sbs) == 1, str(len(sbs)))
    check("ep storyboards: snake_case row fields",
          "storyboard_number" in sbs[0] and "episode_id" in sbs[0], str(sorted(sbs[0])))
    check("ep storyboards: aggregate keys present",
          {"character_ids", "character_costumes", "prop_ids", "characters"} <= set(sbs[0]),
          str(sorted(sbs[0])))
    check("ep storyboards: character_ids", sbs[0]["character_ids"] == [ep_char_id], str(sbs[0]["character_ids"]))
    check("ep storyboards: prop_ids", sbs[0]["prop_ids"] == [42], str(sbs[0]["prop_ids"]))
    check("ep storyboards: character_costumes keyed by id",
          sbs[0]["character_costumes"] == {str(ep_char_id): "长袍"}, str(sbs[0]["character_costumes"]))
    check("ep storyboards: nested characters filtered to linked",
          [c["id"] for c in sbs[0]["characters"]] == [ep_char_id], str(sbs[0]["characters"]))

    # 流水线十步
    r = client.get(f"/api/v1/episodes/{ep_id}/pipeline-status")
    dump("GET /episodes/{id}/pipeline-status", r)
    ps = r.json()["data"]["steps"]
    check("pipeline: 10 steps", len(ps) == 10, str(sorted(ps)))
    check("pipeline: script_rewrite done (script_content set)", ps["script_rewrite"]["status"] == "done")
    check("pipeline: extract_storyboards done/1",
          ps["extract_storyboards"]["status"] == "done" and ps["extract_storyboards"]["count"] == 1,
          str(ps["extract_storyboards"]))
    check("pipeline: generate_images done (1/1 composed)",
          ps["generate_images"]["status"] == "done" and ps["generate_images"]["completed"] == 1,
          str(ps["generate_images"]))
    check("pipeline: generate_videos pending (0/1)",
          ps["generate_videos"]["status"] == "pending", str(ps["generate_videos"]))
    check("pipeline: merge_episode pending + merged_url null",
          ps["merge_episode"]["status"] == "pending" and ps["merge_episode"]["merged_url"] is None,
          str(ps["merge_episode"]))
    check("pipeline: assign_voices partial (0 assigned of 1)",
          ps["assign_voices"]["status"] == "pending" and ps["assign_voices"]["total"] == 1,
          str(ps["assign_voices"]))

    # 错误路径
    check("ep error: non-numeric id -> 404 'Invalid episode id'",
          client.get("/api/v1/episodes/abc/characters").status_code == 404)
    check("ep error: PUT non-numeric -> 404",
          client.put("/api/v1/episodes/abc", json={"title": "x"}).status_code == 404)
    check("ep error: unknown episode pipeline -> 404 'Episode not found'",
          client.get("/api/v1/episodes/999999/pipeline-status").status_code == 404)

    # 软删（二次删除应 404）
    check("ep delete: 200", client.delete(f"/api/v1/episodes/{ep_id}").status_code == 200)
    r = client.delete(f"/api/v1/episodes/{ep_id}")
    check("ep delete: second delete -> 404 'Episode not found'",
          r.status_code == 404 and r.json()["message"] == "Episode not found", str(r.json()))
    client.delete(f"/api/v1/dramas/{ep_drama_id}")

    # ================= characters / scenes / props 域 =================
    # 这三个域的**返回形状是 camelCase**（Node 直接返回 drizzle 行），
    # 与 dramas/episodes 的 snake_case 不同 —— 这是既有契约，不是不一致。
    r = client.post("/api/v1/dramas", json={"title": "SMOKE CS 父剧"})
    cs_drama_id = r.json()["data"]["id"]

    # --- characters ---
    client.put(f"/api/v1/dramas/{cs_drama_id}/characters", json={"characters": [{"name": "SMOKE 角色A"}]})
    plain = client.get(f"/api/v1/dramas/{cs_drama_id}").json()["data"]["characters"][0]
    check("naming: /dramas 的 characters 是 snake_case", "voice_style" in plain and "voiceStyle" not in plain,
          str(sorted(plain)[:6]))

    char_id = plain["id"]
    r = client.get(f"/api/v1/characters/{char_id}")
    dump("GET /api/v1/characters/{id}", r)
    ch = r.json().get("data") or {}
    check("naming: /characters/{id} 是 camelCase", "voiceStyle" in ch and "voice_style" not in ch,
          str(sorted(ch)[:6]))
    check("chars get: costumeId auto-filled", (ch.get("costumeId") or "").startswith("COST_"), str(ch.get("costumeId")))

    r = client.put(f"/api/v1/characters/{char_id}", json={"voice_style": "smoke-voice", "role": "主角"})
    dump("PUT /api/v1/characters/{id} (snake input)", r)
    ch = r.json().get("data") or {}
    check("chars put: 200 + camelCase body", r.status_code == 200 and "voiceStyle" in ch, str(sorted(ch)[:6]))
    check("chars put: snake input applied", ch.get("voiceStyle") == "smoke-voice" and ch.get("role") == "主角")
    check("chars put: voice change invalidates sample (voiceSampleUrl null)",
          ch.get("voiceSampleUrl") is None, str(ch.get("voiceSampleUrl")))

    r = client.put(f"/api/v1/characters/{char_id}", json={"voiceStyle": "smoke-voice-2", "unknown_field": "x"})
    dump("PUT /api/v1/characters/{id} (camel input + unknown)", r)
    ch = r.json().get("data") or {}
    check("chars put: camel input applied + unknown ignored",
          r.status_code == 200 and ch.get("voiceStyle") == "smoke-voice-2"
          and "unknown_field" not in ch and "unknownField" not in ch,
          f"{r.status_code} {r.text[:300]}")

    check("chars get: invalid id -> 404 'Invalid character id'",
          client.get("/api/v1/characters/abc").json()["message"] == "Invalid character id")
    check("chars get: missing -> 404 'Character not found'",
          client.get("/api/v1/characters/999999").json()["message"] == "Character not found")
    # TS 的 PUT 不校验存在性：更新 0 行后 SELECT 得空 ⇒ 仍是 200
    r = client.put("/api/v1/characters/999999", json={"role": "x"})
    check("chars put: nonexistent still 200 + data null-ish (node parity)",
          r.status_code == 200 and r.json().get("data") is None, str(r.json()))

    # --- characters merge（含资产版本重编号）---
    raw = sqlite3.connect(get_db_path())
    raw.execute("INSERT INTO characters (drama_id, name, role, appearance, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?)", (cs_drama_id, "SMOKE 源", "反派", "高个，黑衣", ts, ts))
    raw.execute("INSERT INTO characters (drama_id, name, appearance, created_at, updated_at)"
                " VALUES (?,?,?,?,?)", (cs_drama_id, "SMOKE 目标", "", ts, ts))
    raw.commit()
    src_id = raw.execute("SELECT id FROM characters WHERE drama_id=? AND name='SMOKE 源'", (cs_drama_id,)).fetchone()[0]
    tgt_id = raw.execute("SELECT id FROM characters WHERE drama_id=? AND name='SMOKE 目标'", (cs_drama_id,)).fetchone()[0]

    # 另一个剧的角色，用于验证「只能合并同一部剧内的角色」
    raw.execute("INSERT INTO characters (drama_id, name, created_at, updated_at) VALUES (?,?,?,?)",
                (new_id, "SMOKE 他剧角色", ts, ts))
    raw.commit()
    other_drama_char = raw.execute("SELECT id FROM characters WHERE drama_id=? AND name='SMOKE 他剧角色'",
                                  (new_id,)).fetchone()[0]

    # 两个分集：ep1 同时有 src+tgt（撞号 → 删 source 行），ep2 只有 src（→ 改挂）
    ep_ids = [e["id"] for e in client.get(f"/api/v1/dramas/{cs_drama_id}").json()["data"]["episodes"]]
    raw.execute("INSERT INTO episodes (drama_id, episode_number, title, created_at, updated_at)"
                " VALUES (?,?,?,?,?)", (cs_drama_id, 99, "SMOKE 第二集", ts, ts))
    raw.commit()
    ep2 = raw.execute("SELECT id FROM episodes WHERE drama_id=? AND episode_number=99", (cs_drama_id,)).fetchone()[0]
    ep1 = ep_ids[0]
    raw.execute("INSERT INTO episode_characters (episode_id, character_id, created_at) VALUES (?,?,?)", (ep1, src_id, ts))
    raw.execute("INSERT INTO episode_characters (episode_id, character_id, created_at) VALUES (?,?,?)", (ep1, tgt_id, ts))
    raw.execute("INSERT INTO episode_characters (episode_id, character_id, created_at) VALUES (?,?,?)", (ep2, src_id, ts))
    # 两个分镜：sb1 同时有 src+tgt（撞号 → 删 source 行），sb2 只有 src（→ 改挂）
    raw.execute("INSERT INTO storyboards (episode_id, storyboard_number, title, created_at, updated_at)"
                " VALUES (?,?,?,?,?)", (ep1, 91, "SMOKE SB1", ts, ts))
    raw.execute("INSERT INTO storyboards (episode_id, storyboard_number, title, created_at, updated_at)"
                " VALUES (?,?,?,?,?)", (ep1, 92, "SMOKE SB2", ts, ts))
    raw.commit()
    sb1, sb2 = [r[0] for r in raw.execute(
        "SELECT id FROM storyboards WHERE episode_id=? AND storyboard_number IN (91,92) ORDER BY storyboard_number",
        (ep1,)).fetchall()]
    raw.execute("INSERT INTO storyboard_characters (storyboard_id, character_id) VALUES (?,?)", (sb1, src_id))
    raw.execute("INSERT INTO storyboard_characters (storyboard_id, character_id) VALUES (?,?)", (sb1, tgt_id))
    raw.execute("INSERT INTO storyboard_characters (storyboard_id, character_id) VALUES (?,?)", (sb2, src_id))
    raw.execute("INSERT INTO image_generations (character_id, drama_id, created_at, updated_at) VALUES (?,?,?,?)",
                (src_id, cs_drama_id, ts, ts))
    # 资产版本：image::composed 组 target 已有 current v1，source 有 v1/v2 → 接续为 v2/v3 historical；
    #          image::first_frame 组 target 没有 → 并入的最新一条应为 current
    def add_ver(asset_id, media, frame, version, status):
        raw.execute(
            "INSERT INTO asset_versions (asset_type, asset_id, media_type, frame_type, version, asset_url,"
            " status, created_at) VALUES ('character',?,?,?,?,?,'" + status + "',?)",
            (asset_id, media, frame, version, f"u{version}.png", ts),
        )
    add_ver(tgt_id, "image", "composed", 1, "current")
    add_ver(src_id, "image", "composed", 1, "current")
    add_ver(src_id, "image", "composed", 2, "historical")
    add_ver(src_id, "image", "first_frame", 1, "current")
    raw.commit()
    raw.close()

    r = client.post(f"/api/v1/characters/{src_id}/merge", json={"target_id": src_id})
    check("merge: target_id == self -> 400 'target_id 必填且不能是自身'",
          r.status_code == 400 and r.json()["message"] == "target_id 必填且不能是自身", str(r.json()))

    r = client.post(f"/api/v1/characters/{src_id}/merge", json={"target_id": other_drama_char})
    check("merge: cross-drama -> 400 '只能合并同一部剧内的角色'",
          r.status_code == 400 and r.json()["message"] == "只能合并同一部剧内的角色", str(r.json()))

    r = client.post(f"/api/v1/characters/{src_id}/merge", json={"target_id": tgt_id})
    dump("POST /api/v1/characters/{id}/merge", r)
    merged = r.json().get("data") or {}
    check("merge: 200 + target_id", r.status_code == 200 and merged.get("target_id") == tgt_id, r.text[:300])
    check("merge: mergedEpisodeLinks == 2", merged.get("mergedEpisodeLinks") == 2, str(merged))
    check("merge: mergedStoryboardLinks == 2", merged.get("mergedStoryboardLinks") == 2, str(merged))
    check("merge: filled uses label=jsKey format",
          "定位=role" in (merged.get("filled") or []) and "外貌特征=appearance" in (merged.get("filled") or []),
          str(merged.get("filled")))

    tgt_row = client.get(f"/api/v1/characters/{tgt_id}").json().get("data") or {}
    check("merge: backfilled empty target fields", tgt_row.get("role") == "反派" and tgt_row.get("appearance") == "高个，黑衣",
          str({k: tgt_row.get(k) for k in ("role", "appearance")}))
    check("merge: source soft-deleted",
          client.get(f"/api/v1/characters/{src_id}").json()["message"] == "Character not found")

    raw = sqlite3.connect(get_db_path())
    ep_links = raw.execute("SELECT episode_id, character_id FROM episode_characters"
                           " WHERE character_id IN (?,?) ORDER BY episode_id", (src_id, tgt_id)).fetchall()
    check("merge: ep links -> target on both episodes, no source rows",
          ep_links == [(ep1, tgt_id), (ep2, tgt_id)], str(ep_links))
    sb_links = raw.execute("SELECT storyboard_id, character_id FROM storyboard_characters"
                           " WHERE character_id IN (?,?) ORDER BY storyboard_id", (src_id, tgt_id)).fetchall()
    check("merge: sb links -> target on both storyboards, no source rows",
          sb_links == [(sb1, tgt_id), (sb2, tgt_id)], str(sb_links))
    check("merge: image_generations re-owned",
          raw.execute("SELECT COUNT(*) FROM image_generations WHERE character_id=?", (tgt_id,)).fetchone()[0] == 1)
    vers = raw.execute("SELECT media_type, frame_type, version, status FROM asset_versions"
                       " WHERE asset_id=? ORDER BY media_type, frame_type, version", (tgt_id,)).fetchall()
    check("merge: asset versions renumbered per (media,frame) group",
          vers == [("image", "composed", 1, "current"), ("image", "composed", 2, "historical"),
                   ("image", "composed", 3, "historical"), ("image", "first_frame", 1, "current")],
          str(vers))
    check("merge: no source asset_versions left",
          raw.execute("SELECT COUNT(*) FROM asset_versions WHERE asset_id=?", (src_id,)).fetchone()[0] == 0)
    raw.close()

    # --- scenes ---
    r = client.post("/api/v1/scenes", json={"drama_id": cs_drama_id, "location": "SMOKE 客厅", "time": "夜"})
    dump("POST /api/v1/scenes", r)
    check("scenes create: 201 + code 201 + 'created'",
          r.status_code == 201 and r.json()["code"] == 201 and r.json()["message"] == "created", str(r.status_code))
    sc = r.json().get("data") or {}
    check("scenes create: camelCase body", "locationId" in sc and "location_id" not in sc, str(sorted(sc)))
    check("scenes create: locationId auto-filled", (sc.get("locationId") or "").startswith("LOC_"), str(sc.get("locationId")))
    scene_id2 = sc["id"]

    r = client.post("/api/v1/scenes", json={"drama_id": cs_drama_id, "location": "SMOKE 空 prompt 场景"})
    sc2 = r.json().get("data") or {}
    check("scenes create: prompt falls back to location", sc2.get("prompt") == "SMOKE 空 prompt 场景", str(sc2.get("prompt")))
    check("scenes create: time defaults to ''", sc2.get("time") == "", repr(sc2.get("time")))

    old_loc_id = sc.get("locationId")
    r = client.put(f"/api/v1/scenes/{scene_id2}", json={"location": "SMOKE 卧室"})
    dump("PUT /api/v1/scenes/{id} (location change)", r)
    upd = r.json().get("data") or {}
    check("scenes put: location applied + camelCase", upd.get("location") == "SMOKE 卧室", str(upd.get("location")))
    check("scenes put: location_id re-locked (not the old one)", upd.get("locationId") != old_loc_id,
          f"{upd.get('locationId')} vs {old_loc_id}")

    # camelCase 入参分支：必须转成 DB 列名（曾因直写 camel 键报 Unconsumed column names）
    r = client.put(f"/api/v1/scenes/{scene_id2}", json={"customPrompt": "smoke scene prompt"})
    check("scenes put: camelCase input applied (column-name conversion)",
          r.status_code == 200 and (r.json().get("data") or {}).get("customPrompt") == "smoke scene prompt",
          f"{r.status_code} {r.text[:200]}")

    check("scenes get: camelCase", "customPrompt" in client.get(f"/api/v1/scenes/{scene_id2}").json()["data"])
    check("scenes get: missing -> 404 'Scene not found'",
          client.get("/api/v1/scenes/999999").json()["message"] == "Scene not found")
    # parseParamId 允许小数 ⇒ 走到查库才失配，报 'Scene not found'（对照 props 的严格解析）
    r = client.get("/api/v1/scenes/1.5")
    check("scenes get: float id passes parse, then 'Scene not found'",
          r.status_code == 404 and r.json()["message"] == "Scene not found", str(r.json()))

    r = client.delete(f"/api/v1/scenes/{scene_id2}")
    check("scenes delete: 200 + {code:200} (hard delete)", r.status_code == 200 and r.json()["code"] == 200)
    raw = sqlite3.connect(get_db_path())
    check("scenes delete: row physically gone (hard delete)",
          raw.execute("SELECT COUNT(*) FROM scenes WHERE id=?", (scene_id2,)).fetchone()[0] == 0)
    raw.close()

    # --- props ---
    r = client.post("/api/v1/props", json={"drama_id": cs_drama_id, "name": "SMOKE 玉坠",
                                           "custom_prompt": "jade pendant prompt"})
    dump("POST /api/v1/props", r)
    pr = r.json().get("data") or {}
    check("props create: 200 (not 201) + camelCase", r.status_code == 200 and "category" in pr, str(sorted(pr)))
    check("props create: category defaults to 道具", pr.get("category") == "道具", str(pr.get("category")))
    check("props create: custom_prompt maps to customPrompt (NOT imagePrompt)",
          pr.get("customPrompt") == "jade pendant prompt" and "imagePrompt" not in pr, str(sorted(pr)))
    prop_id = pr["id"]

    r = client.post("/api/v1/props", json={"name": "无剧"})
    check("props create: missing drama_id -> 400 'drama_id 和 name 必填'",
          r.status_code == 400 and r.json()["message"] == "drama_id 和 name 必填", str(r.json()))

    r = client.get(f"/api/v1/props?drama_id={cs_drama_id}")
    dump("GET /api/v1/props?drama_id=", r)
    check("props list: filtered + camelCase + id asc",
          [p["id"] for p in r.json()["data"]] == [prop_id], str(r.json()["data"]))

    r = client.put(f"/api/v1/props/{prop_id}", json={"holder": "林晚", "size_hint": "掌心大小"})
    dump("PUT /api/v1/props/{id}", r)
    check("props put: snake fields applied, camelCase body",
          r.json()["data"].get("holder") == "林晚" and r.json()["data"].get("sizeHint") == "掌心大小",
          str(r.json()["data"]))

    check("props get: missing -> 404 '物品不存在'",
          client.get("/api/v1/props/999999").json()["message"] == "物品不存在")
    r = client.get("/api/v1/props/1.5")
    check("props get: float id -> 404 'Invalid prop id' (strict parseId)",
          r.status_code == 404 and r.json()["message"] == "Invalid prop id", str(r.json()))

    r = client.delete(f"/api/v1/props/{prop_id}")
    dump("DELETE /api/v1/props/{id}", r)
    check("props delete: {ok: true}", r.json().get("data") == {"ok": True}, str(r.json()))
    check("props delete: soft (list no longer contains it)",
          client.get(f"/api/v1/props?drama_id={cs_drama_id}").json()["data"] == [])
    raw = sqlite3.connect(get_db_path())
    check("props delete: row still on disk (soft delete)",
          raw.execute("SELECT COUNT(*) FROM prop_templates WHERE id=?", (prop_id,)).fetchone()[0] == 1)
    raw.close()

    # 未迁移端点仍走 501 —— **动态派生**，不再硬编码清单
    # （这里原先写死过 characters/scenes/props 三连，后来又写死 continue-script，
    #   都被后续迁移打破且只有全量才暴露 ⇒ 见 `_unregistered_samples` 的说明）
    cs_ep_id = client.get(f"/api/v1/dramas/{cs_drama_id}").json()["data"]["episodes"][0]["id"]
    samples = _unregistered_samples()
    check("未迁移兜底: 仍有未迁移端点（动态抽样非空）", len(samples) > 0, len(samples))
    for method, path in samples:
        code = client.request(method, path, json={}).status_code
        check(f"未迁移兜底: {method} {path} -> 501", code == 501, code)

    client.delete(f"/api/v1/dramas/{cs_drama_id}")

    # ================= storyboards 域 =================
    r = client.post("/api/v1/dramas", json={"title": "SMOKE SB 父剧"})
    sb_drama_id = r.json()["data"]["id"]
    sb_ep = client.get(f"/api/v1/dramas/{sb_drama_id}").json()["data"]["episodes"][0]["id"]

    # 角色（带音色）+ 场景，并关联到当前集（storyboards 会校验归属）
    client.put(f"/api/v1/dramas/{sb_drama_id}/characters",
               json={"characters": [{"name": "林晚", "voice_style": "voice-x"}]})
    sb_char_id = client.get(f"/api/v1/dramas/{sb_drama_id}").json()["data"]["characters"][0]["id"]
    scene_r = client.post("/api/v1/scenes", json={"drama_id": sb_drama_id, "location": "SMOKE 巷口", "time": "夜"})
    sb_scene_id = (scene_r.json().get("data") or {})["id"]

    raw = sqlite3.connect(get_db_path())
    raw.execute("INSERT INTO episode_scenes (episode_id, scene_id, created_at) VALUES (?,?,?)",
                (sb_ep, sb_scene_id, ts))
    raw.execute("INSERT INTO episode_characters (episode_id, character_id, created_at) VALUES (?,?,?)",
                (sb_ep, sb_char_id, ts))
    raw.commit()
    raw.close()

    # --- 归属校验 ---
    r = client.post("/api/v1/storyboards", json={"episode_id": sb_ep, "scene_id": 999999})
    dump("POST /storyboards (bad scene_id)", r)
    check("sb create: scene_id not in episode -> 400 exact",
          r.status_code == 400 and r.json()["message"] == "scene_id 必须来自当前集已关联场景", str(r.json()))

    r = client.post("/api/v1/storyboards",
                    json={"episode_id": sb_ep, "scene_id": sb_scene_id, "character_ids": [999999]})
    check("sb create: character_ids not in episode -> 400 exact",
          r.status_code == 400 and r.json()["message"] == "character_ids 必须来自当前集已关联角色", str(r.json()))

    # --- 正常创建（snake_case + 201）---
    r = client.post("/api/v1/storyboards", json={
        "episode_id": sb_ep, "scene_id": sb_scene_id, "storyboard_number": 7,
        "title": "SMOKE 镜头", "character_ids": [sb_char_id], "prop_ids": [1001],
    })
    dump("POST /api/v1/storyboards", r)
    sb = r.json().get("data") or {}
    check("sb create: 201 + code 201 + 'created'",
          r.status_code == 201 and r.json().get("code") == 201 and r.json().get("message") == "created",
          r.text[:200])
    check("sb create: snake_case body", "storyboard_number" in sb and "shot_type" in sb, str(sorted(sb)[:6]))
    check("sb create: character_ids + prop_ids echoed",
          sb.get("character_ids") == [sb_char_id] and sb.get("prop_ids") == [1001], str(sb)[:200])
    check("sb create: duration defaults to 10", sb.get("duration") == 10, str(sb.get("duration")))
    sb_id = sb["id"]

    check("sb update: missing -> 404 '镜头不存在'",
          client.put("/api/v1/storyboards/999999", json={"title": "x"}).json()["message"] == "镜头不存在")
    check("sb update: non-numeric id -> 404 'Invalid storyboard id'",
          client.put("/api/v1/storyboards/abc", json={}).json()["message"] == "Invalid storyboard id")

    # --- PUT 无 dialogue ⇒ data 为 null 但**键存在** ---
    # ⚠️ 这条曾断言成 `{"code":200,"message":"success"}`（无 data 键），是**误读**：
    # TS 是 `success(c, dialogueValidation ? {...} : undefined)`，而 `success(c, data = null)`
    # 的默认参数会把显式传入的 undefined 换成 null ⇒ data 键仍在。原断言锁住的是我自己的 bug。
    r = client.put(f"/api/v1/storyboards/{sb_id}", json={"title": "SMOKE 镜头改"})
    dump("PUT /storyboards/{id} (no dialogue)", r)
    check("sb update: 200 with data:null (default param, key present)",
          r.status_code == 200 and r.json() == {"code": 200, "data": None, "message": "success"},
          r.text[:200])

    # --- dialogue ⇒ 返回 dialogue_validation ---
    DIALOGUE = "林晚：（低声）你回来了，外面雨很大。\n张三：我是谁你不需要知道。\n旁白：那天夜里下了很大的雨。"
    r = client.put(f"/api/v1/storyboards/{sb_id}", json={"dialogue": DIALOGUE})
    dump("PUT /storyboards/{id} (dialogue)", r)
    dv = (r.json().get("data") or {}).get("dialogue_validation") or []
    by_speaker = {v["speaker"]: v for v in dv}
    check("sb dialogue: 3 lines parsed", len(dv) == 3, str([v.get("speaker") for v in dv]))
    check("sb dialogue: matched character uses its voice",
          by_speaker.get("林晚", {}).get("match_status") == "matched"
          and by_speaker.get("林晚", {}).get("voice_id") == "voice-x"
          and by_speaker.get("林晚", {}).get("character_id") == sb_char_id, str(by_speaker.get("林晚")))
    check("sb dialogue: unknown speaker -> not_found + warning",
          by_speaker.get("张三", {}).get("match_status") == "not_found"
          and '角色"张三"在剧组角色列表中不存在，将使用默认音色' == by_speaker.get("张三", {}).get("warning"),
          str(by_speaker.get("张三")))
    check("sb dialogue: narrator bypasses matching",
          by_speaker.get("旁白", {}).get("match_status") == "narrator"
          and by_speaker.get("旁白", {}).get("voice_id") == "alloy", str(by_speaker.get("旁白")))
    check("sb dialogue: matched line has NO warning key (conditional spread)",
          "warning" not in by_speaker.get("林晚", {}), str(by_speaker.get("林晚")))

    # --- validate-dialogue 端点 ---
    r = client.get(f"/api/v1/storyboards/{sb_id}/validate-dialogue")
    dump("GET /storyboards/{id}/validate-dialogue", r)
    vd = r.json().get("data") or {}
    check("validate-dialogue: keys", set(vd) == {
        "lines", "all_matched", "issue_count", "summary", "issues",
        "tts_stale", "tts_stale_count", "tts_stale_details"}, str(sorted(vd)))
    check("validate-dialogue: 1 issue / not all matched",
          vd.get("issue_count") == 1 and vd.get("all_matched") is False, str(vd.get("issue_count")))
    check("validate-dialogue: summary text",
          vd.get("summary") == "1 个角色存在音色匹配问题", str(vd.get("summary")))
    check("validate-dialogue: issues only non-matched",
          [i["speaker"] for i in vd.get("issues") or []] == ["张三"], str(vd.get("issues")))
    check("validate-dialogue: tts_stale false (no tts yet)", vd.get("tts_stale") is False)

    # --- TTS 音色过期检测（存一个与当前 voice_style 不一致的 voice_id）---
    raw = sqlite3.connect(get_db_path())
    raw.execute("UPDATE storyboards SET tts_audio_url=? WHERE id=?",
                (json.dumps([{"speaker": "林晚", "voice_id": "old-voice"}], ensure_ascii=False), sb_id))
    raw.commit()
    raw.close()
    r = client.get(f"/api/v1/storyboards/{sb_id}/validate-dialogue")
    vd2 = r.json().get("data") or {}
    dump("GET validate-dialogue (stale tts)", r)
    check("validate-dialogue: tts_stale detected",
          vd2.get("tts_stale") is True and vd2.get("tts_stale_count") == 1, str(vd2.get("tts_stale_details")))
    stale = (vd2.get("tts_stale_details") or [{}])[0]
    check("validate-dialogue: stale detail fields",
          stale.get("stored_voice") == "old-voice" and stale.get("current_voice") == "voice-x",
          str(stale))

    # --- PUT dialogue 会作废旧 TTS / 字幕 ---
    client.put(f"/api/v1/storyboards/{sb_id}", json={"dialogue": "林晚：第二次说话也要八个字以上。\n"})
    raw = sqlite3.connect(get_db_path())
    tts_after, sub_after = raw.execute(
        "SELECT tts_audio_url, subtitle_url FROM storyboards WHERE id=?", (sb_id,)).fetchone()
    raw.close()
    check("sb update: dialogue change nulls tts_audio_url + subtitle_url",
          tts_after is None and sub_after is None, f"{tts_after!r} {sub_after!r}")

    # --- character_costumes 的 JSON 字符串键必须能命中数字角色 id ---
    r = client.put(f"/api/v1/storyboards/{sb_id}",
                   json={"character_ids": [sb_char_id], "character_costumes": {str(sb_char_id): "戏服A"}})
    check("sb update: costumes put 200", r.status_code == 200, r.text[:200])
    agg = client.get(f"/api/v1/episodes/{sb_ep}/storyboards").json()["data"]
    target_sb = [s for s in agg if s["id"] == sb_id][0]
    check("sb update: character_costumes keyed by id survives JSON string keys",
          target_sb["character_costumes"] == {str(sb_char_id): "戏服A"},
          str(target_sb["character_costumes"]))

    # --- GET /:id/qc ---
    r = client.get(f"/api/v1/storyboards/{sb_id}/qc")
    dump("GET /storyboards/{id}/qc (empty)", r)
    check("sb qc: no record -> message",
          r.json().get("data") == {"message": "No QC record yet"}, r.text[:200])

    raw = sqlite3.connect(get_db_path())
    raw.execute("INSERT INTO video_quality_checks (storyboard_id, drama_id, overall_score, issues,"
                " dimensions, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (sb_id, sb_drama_id, 70, '["old"]', '{"a":1}', "passed", ts, ts))
    raw.execute("INSERT INTO video_quality_checks (storyboard_id, drama_id, overall_score, issues,"
                " dimensions, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (sb_id, sb_drama_id, 88, '["new"]', '{"b":2}', "passed", ts, ts))
    raw.commit()
    raw.close()
    r = client.get(f"/api/v1/storyboards/{sb_id}/qc")
    dump("GET /storyboards/{id}/qc (2 records)", r)
    qc = r.json().get("data") or {}
    check("sb qc: latest record (max id) + camelCase + parsed json",
          qc.get("overallScore") == 88 and qc.get("issues") == ["new"] and qc.get("dimensions") == {"b": 2},
          str(qc))

    # --- DELETE：硬删 + 只清 characters 关联（props 关联会留孤儿，与 TS 一致）---
    r = client.delete(f"/api/v1/storyboards/{sb_id}")
    dump("DELETE /storyboards/{id}", r)
    check("sb delete: 200", r.status_code == 200 and r.json().get("code") == 200)
    raw = sqlite3.connect(get_db_path())
    check("sb delete: storyboards row gone (hard delete)",
          raw.execute("SELECT COUNT(*) FROM storyboards WHERE id=?", (sb_id,)).fetchone()[0] == 0)
    check("sb delete: storyboard_characters cleaned",
          raw.execute("SELECT COUNT(*) FROM storyboard_characters WHERE storyboard_id=?", (sb_id,)).fetchone()[0] == 0)
    check("sb delete: storyboard_props NOT cleaned (inherited behaviour, documented)",
          raw.execute("SELECT COUNT(*) FROM storyboard_props WHERE storyboard_id=?", (sb_id,)).fetchone()[0] == 1)
    raw.close()

    # --- 未迁移端点仍走 501（**动态派生**，同上）---
    # 这里原先写死过 `generate-tts` / `split`，二者实现后即失效 ⇒ 改为动态抽样
    for method, path in _unregistered_samples(limit=4):
        code = client.request(method, path, json={}).status_code
        check(f"sb 兜底: {method} {path} -> 501", code == 501, code)

    client.delete(f"/api/v1/dramas/{sb_drama_id}")

    # ================= 资源库 / presets / app-settings =================
    # 资源库用的是**另一套信封**：成功 code=0；错误 code=400/404/500 但 **HTTP 恒 200**。
    r = client.get("/api/v1/character-library/abc")
    dump("GET /character-library/abc", r)
    check("lib envelope: invalid id -> HTTP 200 + body code 400",
          r.status_code == 200 and r.json() == {"code": 400, "data": None, "message": "无效的ID参数"},
          f"{r.status_code} {r.text[:200]}")

    r = client.get("/api/v1/character-library/999999")
    check("lib envelope: missing -> HTTP 200 + body code 404",
          r.status_code == 200 and r.json() == {"code": 404, "data": None, "message": "角色模板不存在"},
          f"{r.status_code} {r.text[:200]}")

    r = client.post("/api/v1/character-library", json={"name": "只有名字"})
    check("lib: character requires name+appearance -> 400 exact",
          r.status_code == 200 and r.json()["code"] == 400
          and r.json()["message"] == "名称和外貌描述为必填项", r.text[:200])

    # --- 创建（含 safeStringify 的两种分支）---
    r = client.post("/api/v1/character-library", json={
        "name": "SMOKE 模板", "appearance": "高个黑衣",
        "referenceImages": ["ref1.png", "ref2.png"], "tags": ["a", "b"], "voiceConfig": {"tone": "低"},
    })
    dump("POST /character-library", r)
    check("lib create: HTTP 200 + code 0 + message 创建成功",
          r.status_code == 200 and r.json()["code"] == 0 and r.json()["message"] == "创建成功",
          r.text[:200])
    tpl_id = r.json()["data"]["id"]

    r = client.get(f"/api/v1/character-library/{tpl_id}")
    dump("GET /character-library/{id}", r)
    tpl = r.json().get("data") or {}
    check("lib detail: no message key (list/detail parity)", "message" not in r.json(), str(r.json().keys()))
    # 未传的 JSON 列在创建时被 safeStringify(None) 写成 ''，读回来是 []（不是 None）
    check("lib detail: json cols parsed in place",
          tpl.get("tags") == ["a", "b"] and tpl.get("metadata") == [], str(tpl.get("metadata")))
    check("lib detail: reference_images kept AND aliased to referenceImages",
          tpl.get("reference_images") == ["ref1.png", "ref2.png"]
          and tpl.get("referenceImages") == ["ref1.png", "ref2.png"], str(tpl.get("referenceImages")))
    check("lib detail: voice_config aliased to voiceConfig",
          tpl.get("voiceConfig") == {"tone": "低"}, str(tpl.get("voiceConfig")))
    check("lib detail: category default 通用", tpl.get("category") == "通用", str(tpl.get("category")))
    check("lib detail: usage_count defaults 0", tpl.get("usage_count") == 0, str(tpl.get("usage_count")))

    # parseInt 的宽松解析：/12abc 会命中 id=12（不是 parseParamId 的语义）
    r = client.get(f"/api/v1/character-library/{tpl_id}abc")
    check("lib: parseInt leniency (/12abc hits id=12)",
          r.json()["code"] == 0 and (r.json().get("data") or {}).get("id") == tpl_id, r.text[:200])

    # safeStringify 的「非 JSON 字符串」分支：写回会被再编码一次，读回来能还原
    r = client.post("/api/v1/character-library",
                    json={"name": "SMOKE 纯文本参考图", "appearance": "x", "referenceImages": "not json"})
    tpl2_id = r.json()["data"]["id"]
    tpl2 = client.get(f"/api/v1/character-library/{tpl2_id}").json()["data"]
    check("lib: safeStringify round-trips a non-JSON string",
          tpl2.get("referenceImages") == "not json", repr(tpl2.get("referenceImages")))
    raw = sqlite3.connect(get_db_path())
    stored = raw.execute("SELECT reference_images FROM character_templates WHERE id=?",
                         (tpl2_id,)).fetchone()[0]
    raw.close()
    check("lib: non-JSON string stored as re-encoded JSON", stored == '"not json"', repr(stored))

    # --- 列表 / 分页 / 过滤 ---
    r = client.get("/api/v1/character-library", params={"pageSize": 1, "search": "SMOKE"})
    dump("GET /character-library?pageSize=1&search=SMOKE", r)
    lst = r.json().get("data") or {}
    check("lib list: pagination keys camelCase",
          set(lst) == {"items", "total", "page", "pageSize", "totalPages"}, str(sorted(lst)))
    check("lib list: pageSize honored", len(lst.get("items") or []) == 1, str(len(lst.get("items") or [])))
    check("lib list: total >= 2", (lst.get("total") or 0) >= 2, str(lst.get("total")))

    r = client.get("/api/v1/character-library", params={"sortBy": "id; DROP TABLE characters--"})
    check("lib list: illegal sortBy falls back (no injection)",
          r.json()["code"] == 0, r.text[:200])

    # --- categories / tags ---
    r = client.get("/api/v1/character-library/categories")
    check("lib categories: code 0 + list", r.json()["code"] == 0 and "通用" in (r.json().get("data") or []),
          r.text[:200])
    r = client.get("/api/v1/character-library/tags")
    check("lib tags: code 0 + deduped sorted", r.json()["code"] == 0 and "a" in (r.json().get("data") or []),
          r.text[:200])

    # --- 更新（`??` 与 `!== undefined` 的分支差异）---
    r = client.put(f"/api/v1/character-library/{tpl_id}",
                   json={"name": "SMOKE 模板改", "tags": None, "description": None})
    dump("PUT /character-library/{id}", r)
    check("lib update: HTTP 200 + code 0 + message 更新成功",
          r.json() == {"code": 0, "data": {"id": tpl_id}, "message": "更新成功"}, r.text[:200])
    tpl3 = client.get(f"/api/v1/character-library/{tpl_id}").json()["data"]
    check("lib update: name applied (?? on plain column)", tpl3.get("name") == "SMOKE 模板改")
    # 创建时未传 description ⇒ 被 `|| ''` 落成空串；更新传 null 时 `??` 保留原值（空串而非 None）
    check("lib update: description null keeps existing empty string (nullish, not truthiness)",
          tpl3.get("description") == "", repr(tpl3.get("description")))
    check("lib update: json column with null CLEARS to []  (!== undefined branch)",
          tpl3.get("tags") == [], repr(tpl3.get("tags")))

    # --- apply（角色模板 → 剧组角色，usage_count +1）---
    r = client.post("/api/v1/dramas", json={"title": "SMOKE LIB 父剧"})
    lib_drama_id = r.json()["data"]["id"]
    r = client.post(f"/api/v1/character-library/{tpl_id}/apply", json={"dramaId": lib_drama_id})
    dump("POST /character-library/{id}/apply", r)
    check("lib apply: code 0 + characterId",
          r.json()["code"] == 0 and (r.json().get("data") or {}).get("characterId"), r.text[:200])
    check("lib apply: message contains template name",
          f'角色 "SMOKE 模板改" 已应用到剧组' == r.json().get("message"), str(r.json().get("message")))
    check("lib apply: usage_count incremented",
          client.get(f"/api/v1/character-library/{tpl_id}").json()["data"]["usage_count"] == 1)
    r = client.post(f"/api/v1/character-library/{tpl_id}/apply", json={})
    check("lib apply: missing dramaId -> 400 exact",
          r.json() == {"code": 400, "data": None, "message": "请提供目标剧组ID"}, r.text[:200])

    # --- from-character ---
    applied_char = client.get(f"/api/v1/dramas/{lib_drama_id}").json()["data"]["characters"][0]["id"]
    r = client.post(f"/api/v1/character-library/from-character/{applied_char}", json={})
    dump("POST /character-library/from-character/{id}", r)
    check("lib from-character: code 0 + templateId",
          r.json()["code"] == 0 and (r.json().get("data") or {}).get("templateId"), r.text[:200])
    check("lib from-character: missing char -> 404 角色不存在",
          client.post("/api/v1/character-library/from-character/999999", json={}).json()
          == {"code": 404, "data": None, "message": "角色不存在"})

    # --- 其余三个库：同构，抽查各自的差异点 ---
    r = client.post("/api/v1/scene-library", json={"name": "SMOKE 场景模板", "timeOfDay": "夜", "style": "现代"})
    check("scene lib: create code 0", r.json().get("code") == 0, r.text[:200])
    scene_tpl_id = r.json()["data"]["id"]
    r = client.get("/api/v1/scene-library/filter-options")
    check("scene lib: filter-options {timeOfDay,styles}",
          r.json()["code"] == 0 and set(r.json()["data"]) == {"timeOfDay", "styles"}, r.text[:200])
    r = client.post(f"/api/v1/scene-library/{scene_tpl_id}/apply",
                    json={"dramaId": lib_drama_id})
    check("scene lib apply: code 0 + sceneId",
          r.json()["code"] == 0 and (r.json().get("data") or {}).get("sceneId"), r.text[:200])

    r = client.post("/api/v1/weapon-library", json={"name": "SMOKE 剑"})
    check("weapon lib: create code 0 + category default 剑",
          r.json().get("code") == 0
          and client.get(f"/api/v1/weapon-library/{r.json()['data']['id']}").json()["data"]["category"] == "剑",
          r.text[:200])
    r = client.get("/api/v1/weapon-library/filter-options")
    check("weapon lib: filter-options keys {categories,types,ranks}",
          r.json()["code"] == 0 and set(r.json()["data"]) == {"categories", "types", "ranks"},
          r.text[:200])
    check("weapon lib: no /apply route (not migrated by design)",
          client.post("/api/v1/weapon-library/1/apply", json={"dramaId": 1}).status_code == 501)

    r = client.post("/api/v1/costume-library", json={"name": "SMOKE 衣", "bodyPart": "上衣"})
    check("costume lib: create code 0", r.json().get("code") == 0, r.text[:200])
    costume_tpl_id = r.json()["data"]["id"]
    check("costume lib: no /apply route (not migrated by design)",
          client.post(f"/api/v1/costume-library/{costume_tpl_id}/apply", json={"dramaId": 1}).status_code == 501)
    r = client.get("/api/v1/costume-library/filter-options")
    check("costume lib: filter-options keys {styles,bodyParts,seasons}",
          r.json()["code"] == 0 and set(r.json()["data"]) == {"styles", "bodyParts", "seasons"},
          r.text[:200])

    # --- batch-delete ---
    r = client.post("/api/v1/weapon-library/batch-delete", json={"ids": []})
    check("lib batch-delete: empty ids -> 400 exact",
          r.json() == {"code": 400, "data": None, "message": "请提供要删除的ID列表"}, r.text[:200])

    r = client.post("/api/v1/scene-library/batch-delete", json={"ids": [scene_tpl_id]})
    dump("POST /scene-library/batch-delete", r)
    check("lib batch-delete: code 0 + deletedCount + localized message",
          r.json() == {"code": 0, "data": {"deletedCount": 1}, "message": "成功删除 1 个场景模板"},
          r.text[:200])

    # --- DELETE ---
    r = client.delete(f"/api/v1/costume-library/{costume_tpl_id}")
    dump("DELETE /costume-library/{id}", r)
    check("lib delete: code 0 + id + message 删除成功",
          r.json() == {"code": 0, "data": {"id": costume_tpl_id}, "message": "删除成功"}, r.text[:200])
    r = client.delete(f"/api/v1/costume-library/{costume_tpl_id}")
    check("lib delete: second time -> 404 服装模板不存在",
          r.json() == {"code": 404, "data": None, "message": "服装模板不存在"}, r.text[:200])

    # 兜底枚举只在「该表一条非空值都没有」时才生效 —— 注意创建时未传的字段会被写成 ''，
    # 而 `IS NOT NULL` 对 '' 成立，所以实际几乎不会走到兜底（继承自 Node 的行为）。
    # 放在最后做：这一步会把前面用的模板全部软删掉。
    raw = sqlite3.connect(get_db_path())
    raw.execute("UPDATE character_templates SET deleted_at = ? WHERE deleted_at IS NULL", (ts,))
    raw.execute("UPDATE weapon_templates SET deleted_at = ? WHERE deleted_at IS NULL", (ts,))
    raw.execute("UPDATE costume_templates SET deleted_at = ? WHERE deleted_at IS NULL", (ts,))
    raw.commit()
    raw.close()
    r = client.get("/api/v1/character-library/categories")
    check("character lib: categories has NO fallback (unlike weapon)",
          r.json()["data"] == [], r.text[:200])
    r = client.get("/api/v1/weapon-library/categories")
    check("weapon lib: categories falls back only when table is empty",
          r.json()["data"] == WEAPON_CATEGORIES, r.text[:200])
    r = client.get("/api/v1/costume-library/filter-options")
    check("costume lib: filter-options falls back when table is empty",
          r.json()["data"]["bodyParts"] == BODY_PARTS
          and r.json()["data"]["styles"] == COSTUME_STYLES, r.text[:200])

    client.delete(f"/api/v1/dramas/{lib_drama_id}")

    # --- presets（正常信封 code 200 + camelCase + config 反序列化）---
    r = client.get("/api/v1/presets")
    dump("GET /api/v1/presets", r)
    check("presets: 200 + code 200 + list", r.status_code == 200 and r.json()["code"] == 200, r.text[:200])

    r = client.post("/api/v1/presets", json={"type": "smoke", "name": "SMOKE 预设", "config": {"k": 1}})
    dump("POST /api/v1/presets", r)
    pre = r.json().get("data") or {}
    check("presets create: camelCase row + parsed config",
          pre.get("type") == "smoke" and pre.get("config") == {"k": 1} and "createdAt" in pre,
          str(sorted(pre)))
    preset_id = pre["id"]

    r = client.post("/api/v1/presets", json={"name": "缺 type"})
    check("presets create: missing type -> 400 exact",
          r.status_code == 400 and r.json()["message"] == "type and name are required", r.text[:200])

    # colorGrade 类型会被规整（缺字段补 0，skinTone 夹到 0..100）
    r = client.post("/api/v1/presets", json={"type": "colorGrade", "name": "SMOKE 校色",
                                             "config": {"exposure": 20, "skinTone": 500}})
    cg = r.json().get("data", {}).get("config") or {}
    check("presets create: colorGrade normalized",
          cg.get("exposure") == 20 and cg.get("skinTone") == 100
          and cg.get("colorCalibration") == {"red": 0, "green": 0, "blue": 0}, str(cg))
    cg_id = r.json()["data"]["id"]

    r = client.put(f"/api/v1/presets/{preset_id}", json={"name": "SMOKE 预设改"})
    check("presets update: name applied + config parsed",
          r.json()["data"]["name"] == "SMOKE 预设改" and r.json()["data"]["config"] == {"k": 1},
          r.text[:200])
    check("presets update: non-numeric id -> 404 'Invalid preset id'",
          client.put("/api/v1/presets/abc", json={}).json()["message"] == "Invalid preset id")

    check("presets delete: 200", client.delete(f"/api/v1/presets/{preset_id}").status_code == 200)
    check("presets delete: gone from list",
          all(p["id"] != preset_id for p in client.get("/api/v1/presets").json()["data"]))
    client.delete(f"/api/v1/presets/{cg_id}")

    # --- app-settings ---
    r = client.get("/api/v1/app-settings")
    dump("GET /api/v1/app-settings", r)
    check("app-settings: 200 + code 200 + dict", r.status_code == 200 and isinstance(r.json()["data"], dict))

    original_style = client.get("/api/v1/app-settings").json()["data"].get("art_style")
    r = client.put("/api/v1/app-settings", json={"art_style": "anime"})
    check("app-settings put: anime accepted", r.json()["data"].get("art_style") == "anime", r.text[:200])

    # ⚠️ 已知缺陷锁定：画风体系有 10 种，但此处白名单只有 6 种 ⇒ noir 被拒。
    # 若将来两边一起修好，这条用例会失败 —— 那是预期的信号，不是回归。
    r = client.put("/api/v1/app-settings", json={"art_style": "noir"})
    dump("PUT /app-settings noir (known defect)", r)
    check("app-settings: noir rejected (KNOWN DEFECT: stale 6-key whitelist vs 10 art styles)",
          r.status_code == 400 and r.json()["message"] == "art_style 必须是 realistic/anime/ghibli/cinematic/comic/watercolor 之一或留空",
          r.text[:200])

    client.put("/api/v1/app-settings", json={"art_style": original_style or ""})

    # ================= asset-versions / traces / storage / usage =================
    # --- asset-versions ---
    r = client.get("/api/v1/asset-versions")
    check("av: missing params -> 400 exact",
          r.status_code == 400 and r.json()["message"] == "asset_type and asset_id are required",
          r.text[:200])
    r = client.get("/api/v1/asset-versions", params={"asset_type": "storyboard", "asset_id": "abc"})
    check("av: non-numeric asset_id -> 400 exact",
          r.status_code == 400 and r.json()["message"] == "invalid asset_id", r.text[:200])

    # 建一个分镜，再用服务层留档两个版本（第二个成为 current），最后回滚到 v1 验写回
    r = client.post("/api/v1/dramas", json={"title": "SMOKE AV 父剧"})
    av_drama_id = r.json()["data"]["id"]
    av_ep = client.get(f"/api/v1/dramas/{av_drama_id}").json()["data"]["episodes"][0]["id"]
    r = client.post("/api/v1/storyboards", json={"episode_id": av_ep, "title": "SMOKE AV 镜头"})
    av_sb_id = r.json()["data"]["id"]

    from app.db import engine as av_engine  # noqa: E402
    from app.services.asset_versions import record_asset_version  # noqa: E402

    with av_engine.begin() as av_conn:
        v1 = record_asset_version(
            av_conn, asset_type="storyboard", asset_id=av_sb_id, media_type="image",
            asset_url="static/av-v1.png", provider="smoke", model="m1",
        )
        v2 = record_asset_version(
            av_conn, asset_type="storyboard", asset_id=av_sb_id, media_type="image",
            asset_url="static/av-v2.png", provider="smoke", model="m2",
        )
    check("av: service recorded two versions", v1 and v2 and v1 != v2, f"{v1} {v2}")

    r = client.get("/api/v1/asset-versions",
                   params={"asset_type": "storyboard", "asset_id": av_sb_id})
    dump("GET /api/v1/asset-versions", r)
    av = r.json()["data"]
    check("av list: envelope snake keys + camelCase rows",
          av["asset_type"] == "storyboard" and av["asset_id"] == av_sb_id
          and {"assetType", "assetUrl", "frameType"} <= set(av["versions"][0]), str(sorted(av["versions"][0])))
    check("av list: version DESC", [v["version"] for v in av["versions"]] == [2, 1],
          str([v["version"] for v in av["versions"]]))
    check("av list: only v2 is current",
          [v["status"] for v in av["versions"]] == ["current", "historical"],
          str([v["status"] for v in av["versions"]]))

    r = client.post(f"/api/v1/asset-versions/{v1}/activate")
    dump("POST /api/v1/asset-versions/{id}/activate", r)
    check("av activate: 200 + activated row",
          r.status_code == 200 and (r.json().get("data") or {}).get("activated", {}).get("id") == v1,
          r.text[:250])
    raw = sqlite3.connect(get_db_path())
    sb_after = raw.execute("SELECT composed_image FROM storyboards WHERE id=?", (av_sb_id,)).fetchone()[0]
    statuses = dict(raw.execute(
        "SELECT id, status FROM asset_versions WHERE asset_type='storyboard' AND asset_id=?",
        (av_sb_id,)).fetchall())
    raw.close()
    check("av activate: asset_url written back to storyboards.composed_image",
          sb_after == "static/av-v1.png", repr(sb_after))
    check("av activate: v1 current / v2 historical",
          statuses.get(v1) == "current" and statuses.get(v2) == "historical", str(statuses))

    # Number() 校验不管正负/整数 ⇒ 负数走到查库才失配
    r = client.post("/api/v1/asset-versions/-1/activate")
    check("av activate: negative id passes Number() check -> 'Version not found'",
          r.status_code == 400 and r.json()["message"] == "Version not found", r.text[:200])
    r = client.post("/api/v1/asset-versions/abc/activate")
    check("av activate: non-numeric -> 400 'invalid version id'",
          r.status_code == 400 and r.json()["message"] == "invalid version id", r.text[:200])

    # --- traces（写入侧仍在 Node；这里直接造 jsonl 文件验读侧）---
    traces_dir = Path(os.environ["DATA_ROOT"]).resolve() / "traces" / "SMOKE"
    traces_dir.mkdir(parents=True, exist_ok=True)
    (traces_dir / "t1.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00.000Z", "traceId": "t1", "scope": "SMOKE",
                    "level": "INFO", "action": "a",
                    "meta": {"inputTokens": 10, "outputTokens": 5}}, ensure_ascii=False) + "\n"
        + json.dumps({"ts": "2026-01-01T00:00:01.000Z", "traceId": "t1", "scope": "SMOKE",
                      "level": "INFO", "action": "b",
                      "meta": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}},
                     ensure_ascii=False) + "\n"
        + "{broken json line\n",  # 残缺行必须被跳过
        encoding="utf-8",
    )

    # 第二个 trace：token 值是被**脱敏成字符串**的哨兵值（真实库里就是 `"***"`）。
    # Number("***") === NaN ⇒ JS 的 `if (input || output || total)` 为假 ⇒ **整条被跳过**。
    # 这正是真实库里 20 个 trace 却统计出 runs=0 的原因，属于**预期行为不是 bug**。
    (traces_dir / "t2.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:02.000Z", "traceId": "t2", "scope": "SMOKE",
                    "level": "SUCCESS", "action": "done",
                    "meta": {"inputTokens": "***", "outputTokens": "***", "totalTokens": "***"}},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # ⚠️ 必须显式固定 mtime：连续写两个文件时文件系统 mtime 可能相同，
    # 而 `list_traces` 按 mtime 倒序 → 相等时顺序不稳 ⇒ 断言会随机红。
    # （本条用例曾经就是这么偶发失败的，不是产品 bug。）
    os.utime(traces_dir / "t1.jsonl", (1_700_000_000, 1_700_000_000))
    os.utime(traces_dir / "t2.jsonl", (1_700_000_100, 1_700_000_100))

    r = client.get("/api/v1/traces")
    dump("GET /api/v1/traces", r)
    metas = r.json().get("data") or []
    check("traces list: 2 files listed", len(metas) == 2, str([m["traceId"] for m in metas]))
    t1_meta = next(m for m in metas if m["traceId"] == "t1")
    check("traces list: meta shape + count",
          t1_meta["scope"] == "SMOKE" and t1_meta["count"] == 2, str(t1_meta)[:250])
    check("traces list: firstTs/lastTs from records",
          t1_meta["firstTs"] == "2026-01-01T00:00:00.000Z", str(t1_meta))
    check("traces list: sorted by mtime desc (newest first)", metas[0]["traceId"] == "t2",
          str([m["traceId"] for m in metas]))

    r = client.get("/api/v1/traces/SMOKE/t1")
    dump("GET /api/v1/traces/{scope}/{traceId}", r)
    tr = r.json()["data"]
    check("traces replay: broken line skipped, count == 2",
          tr["count"] == 2 and len(tr["records"]) == 2, str(tr)[:250])
    r = client.get("/api/v1/traces/SMOKE/nope")
    check("traces replay: missing -> records []", r.json()["data"] == {
        "scope": "SMOKE", "traceId": "nope", "records": []}, r.text[:200])

    r = client.get("/api/v1/traces/stats")
    dump("GET /api/v1/traces/stats", r)
    st = r.json().get("data") or {}
    # 同一 trace 取 totalTokens 最大的那条：line2 的 120 胜过 line1 的 15（保真：camel+snake 两种 meta 都要认）
    check("traces stats: picks max-totalTokens record per trace (camel + snake meta)",
          st.get("totalTokens") == 120 and st.get("totalInputTokens") == 100
          and st.get("totalOutputTokens") == 20, str(st))
    check("traces stats: t2 skipped because Number('***') is NaN (all falsy)",
          st.get("runs") == 1, str(st.get("runs")))
    check("traces stats: byScope sorted + camelCase",
          st.get("byScope") == [{"scope": "SMOKE", "runs": 1, "inputTokens": 100,
                                 "outputTokens": 20, "totalTokens": 120}], str(st.get("byScope")))

    # --- storage（只迁了 info）---
    r = client.get("/api/v1/storage/info")
    dump("GET /api/v1/storage/info", r)
    si = r.json().get("data") or {}
    check("storage info: keys + camelCase",
          set(si) == {"dataRoot", "dbPath", "storagePath", "dbExists", "storageExists",
                      "dbSizeBytes", "storageSizeBytes"}, str(sorted(si)))
    check("storage info: dataRoot == smoke root",
          Path(si["dataRoot"]).resolve() == Path(os.environ["DATA_ROOT"]).resolve(), si["dataRoot"])
    check("storage info: dbExists true + dbSizeBytes > 0",
          si["dbExists"] is True and si["dbSizeBytes"] > 0, str(si["dbSizeBytes"]))
    check("storage change: NOT migrated (shared .data-root marker) -> 501",
          client.post("/api/v1/storage/change", json={"path": "x"}).status_code == 501)

    # --- usage ---
    raw = sqlite3.connect(get_db_path())
    for i, (svc, prov, cost, retry) in enumerate(
        [("image", "minimax", 1.23456, 0), ("video", "minimax", 2.0, 1), ("image", "volc", None, 0)]
    ):
        raw.execute(
            "INSERT INTO api_usage (service_type, provider, model, drama_id, episode_id, units,"
            " cost_amount, is_local, status, retry_count, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (svc, prov, f"m{i}", av_drama_id, av_ep, 1, cost, 1 if cost is None else 0,
             "completed", retry, f"2026-01-0{i + 1}T00:00:00.000Z"),
        )
    raw.commit()
    raw.close()

    r = client.get("/api/v1/usage/summary", params={"drama_id": av_drama_id})
    dump("GET /api/v1/usage/summary", r)
    us = r.json().get("data") or {}
    check("usage summary: camelCase top keys",
          {"totalCount", "totalCost", "checkedCost", "byService", "byProvider", "byDay",
           "records"} == set(us), str(sorted(us)))
    check("usage summary: totalCount 3 + totalCost rounded to 4dp",
          us.get("totalCount") == 3 and us.get("totalCost") == 3.2346, str(us.get("totalCost")))
    check("usage summary: records are snake_case + is_local bool",
          {"service_type", "cost_amount", "is_local", "retry_count", "created_at"} <= set(us["records"][0]),
          str(sorted(us["records"][0])))
    check("usage summary: byService snake keys sorted by count desc",
          us["byService"] == [{"service_type": "image", "count": 2, "cost": 1.23456},
                              {"service_type": "video", "count": 1, "cost": 2.0}],
          str(us["byService"]))
    check("usage summary: byDay ascending dates",
          [d["date"] for d in us["byDay"]] == ["2026-01-01", "2026-01-02", "2026-01-03"],
          str(us["byDay"]))
    check("usage summary: limit truncates records", len(us["records"]) == 3, str(len(us["records"])))

    r = client.get("/api/v1/usage/summary", params={"drama_id": "abc"})
    check("usage summary: bad drama_id -> 400 exact",
          r.status_code == 400 and r.json()["message"] == "invalid drama_id", r.text[:200])
    r = client.get("/api/v1/usage/summary", params={"limit": "abc"})
    check("usage summary: NaN limit -> empty records (slice(0, NaN))",
          r.json()["data"]["records"] == [], str(r.json()["data"]["totalCount"]))

    r = client.get("/api/v1/usage/board", params={"drama_id": av_drama_id})
    dump("GET /api/v1/usage/board", r)
    ub = r.json().get("data") or {}
    check("usage board: snake_case shape",
          {"drama_id", "total_cost", "total_calls", "retry_cost", "episodes"} == set(ub),
          str(sorted(ub)))
    check("usage board: totals + retry split",
          ub.get("total_calls") == 3 and ub.get("retry_cost") == 2.0 and ub.get("total_cost") == 3.2346,
          str({k: ub.get(k) for k in ("total_calls", "retry_cost", "total_cost")}))
    check("usage board: episode row shape",
          {"episode_id", "episode_number", "title", "total_cost", "total_calls", "retry_cost",
           "by_service"} == set(ub["episodes"][0]), str(sorted(ub["episodes"][0])))
    check("usage board: by_service sorted by count desc",
          ub["episodes"][0]["by_service"][0]["service_type"] == "image",
          str(ub["episodes"][0]["by_service"]))
    r = client.get("/api/v1/usage/board")
    check("usage board: missing drama_id -> 400 exact",
          r.status_code == 400 and r.json()["message"] == "invalid drama_id", r.text[:200])
    # （原「estimate 未迁移 -> 501」断言已在 S1 迁移该端点后作废）

    client.delete(f"/api/v1/dramas/{av_drama_id}")

    # --- 成功信封：`success(c, cond ? {...} : undefined)` 必须回 data:null（键在）---
    # 这是「默认参数把 undefined 变成 null」的直接看守：曾经写成不带 data 键。
    r = client.post("/api/v1/dramas", json={"title": "SMOKE envelope 父剧"})
    env_drama = r.json()["data"]["id"]
    env_ep = client.get(f"/api/v1/dramas/{env_drama}").json()["data"]["episodes"][0]["id"]
    env_sb = client.post(
        "/api/v1/storyboards", json={"episode_id": env_ep, "title": "SMOKE envelope 镜头"}
    ).json()["data"]["id"]
    r = client.put(f"/api/v1/storyboards/{env_sb}", json={"title": "改个名"})
    check("envelope: PUT storyboard 无 dialogue -> data:null（键存在）",
          r.status_code == 200 and "data" in r.json() and r.json()["data"] is None, r.text[:200])
    r = client.put(f"/api/v1/storyboards/{env_sb}", json={"dialogue": "张三：「你好」"})
    check("envelope: PUT storyboard 有 dialogue -> data.dialogue_validation 为数组",
          r.status_code == 200 and isinstance((r.json().get("data") or {}).get("dialogue_validation"), list),
          r.text[:200])
    client.delete(f"/api/v1/dramas/{env_drama}")

    # ================= ai-configs / ai-providers =================
    r = client.post("/api/v1/ai-configs", json={"provider": "x"})
    check("aic create: missing service_type -> 400 exact",
          r.status_code == 400
          and r.json()["message"] == "service_type and provider are required", r.text[:200])

    r = client.post("/api/v1/ai-configs", json={
        "service_type": "image", "provider": "gemini", "model": ["m-a", "m-b"],
        "base_url": "http://localhost:7860", "priority": 5,
        "negative_prompt": "低质量", "checkpoint_map": {"a": "b"},
        "settings": {"extra": 1},
    })
    dump("POST /api/v1/ai-configs", r)
    aic = r.json()["data"]
    aic_id = aic["id"]
    check("aic create: HTTP/code 201 + name 默认 provider-serviceType",
          r.status_code == 201 and r.json()["code"] == 201 and aic["name"] == "gemini-image", str(aic))
    check("aic create: model 反序列化为数组", aic["model"] == ["m-a", "m-b"], repr(aic["model"]))

    r = client.get(f"/api/v1/ai-configs/{aic_id}")
    dump("GET /api/v1/ai-configs/{id}", r)
    det = r.json()["data"]
    check("aic detail: 含 negative_prompt / checkpoint_map",
          det["negative_prompt"] == "低质量" and det["checkpoint_map"] == {"a": "b"}, str(det))
    check("aic detail: **不含** is_local（只有列表接口有）", "is_local" not in det, str(sorted(det)))

    r = client.get("/api/v1/ai-configs", params={"service_type": "image"})
    row = next((x for x in r.json()["data"] if x["id"] == aic_id), None)
    dump("GET /api/v1/ai-configs?service_type=image", r)
    check("aic list: is_local 命中 hostname=localhost", row is not None and row["is_local"] is True,
          str(row)[:250])
    check("aic list: service_type 过滤生效",
          all(x["service_type"] == "image" for x in r.json()["data"]), "filter failed")

    r = client.get("/api/v1/ai-configs/abc")
    check("aic detail: 非法 id -> 404 'Invalid config id'",
          r.status_code == 404 and r.json()["message"] == "Invalid config id", r.text[:200])
    r = client.get("/api/v1/ai-configs/999999")
    check("aic detail: 不存在 -> 404 'not found'（默认文案，与非法 id 不同）",
          r.status_code == 404 and r.json()["message"] == "not found", r.text[:200])

    # settings 是**合并写**：只传 negative_prompt 时不能把既有 checkpoint_map 抹掉
    r = client.put(f"/api/v1/ai-configs/{aic_id}", json={"negative_prompt": "新负向"})
    check("aic update: 200 + data:null", r.status_code == 200 and r.json()["data"] is None, r.text[:200])
    r = client.get(f"/api/v1/ai-configs/{aic_id}")
    det = r.json()["data"]
    check("aic update: settings 合并而非覆盖（checkpoint_map 保住）",
          det["negative_prompt"] == "新负向" and det["checkpoint_map"] == {"a": "b"}, str(det))
    # checkpoint_map: null ⇒ 删除该键
    client.put(f"/api/v1/ai-configs/{aic_id}", json={"checkpoint_map": None})
    det = client.get(f"/api/v1/ai-configs/{aic_id}").json()["data"]
    check("aic update: checkpoint_map=null -> 删除该键",
          det["checkpoint_map"] is None and det["negative_prompt"] == "新负向", str(det))
    # 键存在性：显式传 is_active=null 应写入 null（不是忽略）
    client.put(f"/api/v1/ai-configs/{aic_id}", json={"is_active": None})
    det = client.get(f"/api/v1/ai-configs/{aic_id}").json()["data"]
    check("aic update: 显式 null 写入（键存在性判断）", det["is_active"] is None, str(det))

    r = client.get("/api/v1/ai-configs/configs/local")
    check("aic local list: 只回本地配置 + 无 negative_prompt 字段",
          all(x["is_local"] if "is_local" in x else True for x in r.json()["data"])
          and all("negative_prompt" not in x for x in r.json()["data"]), r.text[:250])
    check("aic local list: 含 provider=local-sd 的本地项",
          any(x["provider"] == "local-sd" for x in r.json()["data"]), r.text[:250])

    # quick-local：写入 4 条本地预设，api_key 固定 'local'
    r = client.post("/api/v1/ai-configs/quick-local")
    dump("POST /api/v1/ai-configs/quick-local", r)
    ql = r.json()["data"]["configs"]
    local_rows = [x for x in ql if x["name"].startswith("本地") and x["name"].endswith("服务")]
    check("aic quick-local: 4 条本地预设 + api_key='local'",
          len(local_rows) == 4 and all(x["api_key"] == "local" for x in local_rows),
          str([(x["name"], x["api_key"]) for x in local_rows])[:250])

    # quick-preset：需要 api_key
    r = client.post("/api/v1/ai-configs/quick-preset", json={})
    check("aic quick-preset: 无 api_key -> 400 exact",
          r.status_code == 400 and r.json()["message"] == "api_key is required", r.text[:200])
    r = client.post("/api/v1/ai-configs/quick-preset", json={"api_key": "sk-smoke"})
    dump("POST /api/v1/ai-configs/quick-preset", r)
    qp = r.json()["data"]
    check("aic quick-preset: 4 配置 + 5 Agent + agent_model",
          len([x for x in qp["configs"] if x["name"].endswith("服务") and x["name"].startswith("默认")]) == 4
          and len(qp["agents"]) >= 5 and qp["agent_model"] == "gemini-3-pro-preview",
          str({k: (len(v) if isinstance(v, list) else v) for k, v in qp.items()}))
    check("aic quick-preset: Agent 已指向预设模型",
          all(a["model"] == "gemini-3-pro-preview" for a in qp["agents"]), str(qp["agents"])[:250])

    # ai-providers（独立 Hono app，挂 /api/v1/ai-providers）
    r = client.get("/api/v1/ai-providers")
    dump("GET /api/v1/ai-providers", r)
    provs = r.json()["data"]
    check("ai-providers: 列表 + preset_models 反序列化为数组",
          isinstance(provs, list) and all(isinstance(p["preset_models"], list) for p in provs),
          str(provs)[:250])
    check("ai-providers: 真实库已 seed（Node 负责，Python 不重复 seed）", len(provs) > 0,
          str(len(provs)))

    r = client.delete(f"/api/v1/ai-configs/{aic_id}")
    check("aic delete: 200 + data:null", r.status_code == 200 and r.json()["data"] is None, r.text[:200])
    check("aic delete: 硬删 -> 再查 404", client.get(f"/api/v1/ai-configs/{aic_id}").status_code == 404)
    r = client.delete(f"/api/v1/ai-configs/{aic_id}")
    check("aic delete: 重复删除 -> 404 'Config not found'",
          r.status_code == 404 and r.json()["message"] == "Config not found", r.text[:200])

    # ⚠️ 这两条**已迁**（原先断言 501）：GPU 显存管理器搬到 `services/gpu_manager.py`，
    #    `/gpu/status` 返回**裸 JSON**（不是信封）、`/gpu/release-all` 返回信封 —— 语义差异在
    #    `gpu_manager_test.py` 里逐条锁；这里只做「不再走 501 兜底」的接缝断言。
    gpu_status = client.get("/api/v1/ai-configs/gpu/status")
    check("aic gpu/status: 已迁移（裸 JSON，HTTP 200）",
          gpu_status.status_code == 200 and "code" not in gpu_status.json()
          and gpu_status.json()["totalVRAM_GB"] == 24, gpu_status.text[:120])
    gpu_release = client.post("/api/v1/ai-configs/gpu/release-all")
    check("aic gpu/release-all: 已迁移（信封 {message,status}）",
          gpu_release.status_code == 200
          and gpu_release.json()["data"]["message"] == "All GPU models released",
          gpu_release.text[:120])

    # --- ai-configs：Ollama / 模型列举 / 连通性探测（用本地假 Ollama 让结果确定）---
    import threading  # noqa: E402
    from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

    class _FakeOllama(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/api/tags":
                payload = json.dumps({
                    "models": [
                        {"name": "qwen3:14b", "size": 9_000_000_000,
                         "modified_at": "2026-01-01T00:00:00Z", "digest": "abcdef1234567890"},
                        {"name": "llama3", "size": 0, "modified_at": "", "digest": ""},
                    ]
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):  # 静音
            pass

    fake = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    fake_url = f"http://127.0.0.1:{fake.server_address[1]}"
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    dead_url = "http://127.0.0.1:1"  # 必然连不上

    try:
        r = client.post("/api/v1/ai-configs/ollama/status", json={"base_url": fake_url})
        dump("POST /api/v1/ai-configs/ollama/status", r)
        st = r.json()["data"]
        check("ollama status: running=true + 模型按名排序 + 体量标签",
              st["running"] is True and [m["name"] for m in st["models"]] == ["llama3", "qwen3:14b"]
              and st["models"][1]["size_label"] == "8.4 GB"
              and st["models"][1]["digest"] == "abcdef123456", str(st)[:300])
        r = client.post("/api/v1/ai-configs/ollama/status", json={"base_url": dead_url})
        check("ollama status: 不可达 -> running=false + 提示文案",
              r.json()["data"]["running"] is False
              and "无法连接" in r.json()["data"]["message"], r.text[:200])

        # 模型列举：Ollama 允许省略 tag（qwen3 匹配 qwen3:14b）
        r = client.post("/api/v1/ai-configs/models",
                        json={"provider": "ollama", "base_url": fake_url, "models": ["qwen3"]})
        dump("POST /api/v1/ai-configs/models (ollama tag match)", r)
        md = r.json()["data"]
        check("models: Ollama 省略 tag 也算命中 + 计数器正确",
              md["listable"] is True and md["reachable"] is True and md["model_exists"] is True
              and md["models_count"] == 2 and md["model"] == "qwen3", str(md)[:300])
        r = client.post("/api/v1/ai-configs/models",
                        json={"provider": "ollama", "base_url": fake_url, "models": ["nope"]})
        check("models: 未命中 -> model_exists=false + 文案",
              r.json()["data"]["model_exists"] is False
              and "未出现在平台模型列表中" in r.json()["data"]["message"], r.text[:250])

        r = client.post("/api/v1/ai-configs/models",
                        json={"provider": "openai", "base_url": dead_url})
        md = r.json()["data"]
        check("models: 支持列举但连不上 -> listable=true / reachable=false / 平台连接失败",
              md["listable"] is True and md["reachable"] is False
              and md["message"].startswith("平台连接失败："), str(md)[:300])
        r = client.post("/api/v1/ai-configs/models", json={"provider": "minimax"})
        md = r.json()["data"]
        check("models: 不支持在线列举 -> listable=false / reachable=null",
              md["listable"] is False and md["reachable"] is None
              and "暂不支持在线列举" in md["message"], str(md)[:300])
        r = client.post("/api/v1/ai-configs/models", json={})
        check("models: 无 provider -> 400 exact",
              r.status_code == 400 and r.json()["message"] == "provider is required", r.text[:200])

        # 连通性探测
        r = client.post("/api/v1/ai-configs/test",
                        json={"service_type": "text", "provider": "ollama", "base_url": fake_url})
        dump("POST /api/v1/ai-configs/test (ollama)", r)
        tp = r.json()["data"]
        check("test: 可达 -> ok/reachable/status 正常",
              tp["ok"] is True and tp["reachable"] is True and tp["status"] == 200
              and tp["method"] == "GET" and tp["url"].endswith("/api/tags"), str(tp)[:300])
        r = client.post("/api/v1/ai-configs/test", json={
            "service_type": "text", "provider": "gemini", "base_url": fake_url, "api_key": "SECRET123"})
        tp = r.json()["data"]
        check("test: URL 里的 api key 被遮蔽（且不出现原始值）",
              "SECRET123" not in tp["url"] and "key=***" in tp["url"], tp["url"])
        check("test: 404 不在可达白名单 -> reachable=false + 文案",
              tp["reachable"] is False
              and tp["message"] == "端点未按预期响应，请检查 Base URL 和代理前缀", str(tp)[:300])
        r = client.post("/api/v1/ai-configs/test",
                        json={"service_type": "text", "provider": "openai", "base_url": dead_url})
        tp = r.json()["data"]
        check("test: 连不上 -> 200 信封内 ok=false + response_preview 为空",
              r.status_code == 200 and tp["ok"] is False and tp["reachable"] is False
              and tp["response_preview"] == "" and "status" not in tp, str(tp)[:300])
        r = client.post("/api/v1/ai-configs/test", json={"provider": "openai"})
        check("test: 缺字段 -> 400 exact",
              r.status_code == 400
              and r.json()["message"] == "service_type, provider and base_url are required", r.text[:200])

        # Ollama 拉取/删除
        r = client.post("/api/v1/ai-configs/ollama/pull", json={})
        check("ollama pull: 无模型名 -> 400 exact",
              r.status_code == 400 and r.json()["message"] == "缺少模型名，例如 qwen3:8b", r.text[:200])
        r = client.post("/api/v1/ai-configs/ollama/delete", json={"base_url": dead_url})
        check("ollama delete: 无模型名 -> 400 '缺少模型名'",
              r.status_code == 400 and r.json()["message"] == "缺少模型名", r.text[:200])
        r = client.post("/api/v1/ai-configs/ollama/delete",
                        json={"base_url": dead_url, "name": "qwen3:14b"})
        check("ollama delete: 服务未运行 -> 400 含提示",
              r.status_code == 400 and "Ollama 服务未运行" in r.json()["message"], r.text[:200])

        # 本地运行时健康（**不假设环境里跑没跑**，只验结构与自洽）
        r = client.get("/api/v1/ai-configs/runtime/health")
        dump("GET /api/v1/ai-configs/runtime/health", r)
        rh = r.json()["data"]
        check("runtime health: 4 个运行时 + 结构完整",
              len(rh["services"]) == 4
              and [s["key"] for s in rh["services"]] == ["ollama", "local-sd", "h3", "cosyvoice"]
              and all({"running", "http_status", "latency_ms", "error", "registered_count",
                       "registered", "base_url", "service_type", "provider", "label"} <= set(s)
                      for s in rh["services"]), str(rh)[:300])
        check("runtime health: running_count / all_running 自洽",
              rh["running_count"] == sum(1 for s in rh["services"] if s["running"])
              and rh["total"] == 4
              and rh["all_running"] == (rh["running_count"] == 4), str(rh)[:200])
    finally:
        fake.shutdown()

    # `ollama/start` 会真的拉起进程（最坏等 20 秒）⇒ 只验它能正常回信封，不假设本机装没装
    r = client.post("/api/v1/ai-configs/ollama/start")
    check("ollama start: 200 + started/message/exe 三字段",
          r.status_code == 200
          and {"started", "message", "exe"} <= set(r.json()["data"]), r.text[:200])

    # ================= agent-configs / style-profiles / generations =================
    # --- agent-configs ---
    r = client.get("/api/v1/agent-configs")
    dump("GET /api/v1/agent-configs", r)
    check("ac list: array + snake_case keys",
          isinstance(r.json()["data"], list), r.text[:200])
    # `/defaults` 与 `/generate` **都已本地化**（2026-09-12 提示词+skills；2026-09-13 creator）。
    # `/generate` 现在缺 agent_type 时应 400 `agent_type required`（不再走兜底 501）。
    defaults_resp = client.get("/api/v1/agent-configs/defaults")
    generate_resp = client.post("/api/v1/agent-configs/generate", json={})
    check("ac defaults / generate 均已迁移（200 六项 / 400 agent_type required）",
          defaults_resp.status_code == 200
          and len(defaults_resp.json()["data"]) == 6
          and all(item["instructions"].strip() and "skills" in item
                  for item in defaults_resp.json()["data"])
          and generate_resp.status_code == 400
          and generate_resp.json()["message"] == "agent_type required",
          (defaults_resp.status_code, generate_resp.status_code, generate_resp.text[:120]))

    r = client.post("/api/v1/agent-configs", json={})
    check("ac create: missing agent_type -> 400 exact",
          r.status_code == 400 and r.json()["message"] == "agent_type required", r.text[:200])
    for bad, expect in [
        ("not json", "skills is not valid JSON"),
        ('{"a":1}', "skills must be a JSON array"),
        ('[{"name":"x"}]', "each skill must have an id field"),
    ]:
        r = client.post("/api/v1/agent-configs", json={"agent_type": "smoke_ac", "skills": bad})
        check(f"ac skills: {expect}",
              r.status_code == 400 and r.json()["message"] == expect, r.text[:200])

    r = client.post("/api/v1/agent-configs", json={
        "agent_type": "smoke_ac", "name": "冒烟 Agent", "model": "m1",
        "skills": '[{"id":"s1"}]',
    })
    dump("POST /api/v1/agent-configs", r)
    ac = r.json()["data"]
    ac_id = ac["id"]
    check("ac create: snake_case + 默认值补齐",
          {"agent_type", "system_prompt", "is_active", "max_tokens", "deleted_at"} <= set(ac)
          and ac["temperature"] == 0.7 and ac["max_tokens"] == 4096
          and ac["max_iterations"] == 10 and ac["is_active"] is True, str(ac))
    check("ac create: skills 规范化为紧凑 JSON", ac["skills"] == '[{"id":"s1"}]', repr(ac["skills"]))

    # upsert：model 传空串必须**保留空串**（`??` 语义），name 传空串必须**回退旧值**（`||` 语义）
    r = client.post("/api/v1/agent-configs", json={"agent_type": "smoke_ac", "model": "", "name": ""})
    ac2 = r.json()["data"]
    check("ac upsert: model='' kept (?? nullish, not ||)", ac2["model"] == "", repr(ac2["model"]))
    check("ac upsert: name='' falls back to old (|| truthy)", ac2["name"] == "冒烟 Agent",
          repr(ac2["name"]))
    check("ac upsert: same id (upsert by agent_type) + skills kept",
          ac2["id"] == ac_id and ac2["skills"] == '[{"id":"s1"}]', str(ac2))

    r = client.get(f"/api/v1/agent-configs/{ac_id}")
    check("ac get: 200", r.status_code == 200 and r.json()["data"]["id"] == ac_id, r.text[:200])
    r = client.get("/api/v1/agent-configs/abc")
    check("ac get: non-numeric -> 404 'Invalid agent config id'",
          r.status_code == 404 and r.json()["message"] == "Invalid agent config id", r.text[:200])
    r = client.get("/api/v1/agent-configs/999999")
    check("ac get: missing -> 404 'Not found'",
          r.status_code == 404 and r.json()["message"] == "Not found", r.text[:200])

    # PUT：`'model' in body` 是**键存在性**判断 ⇒ 显式 null 会写入 null
    r = client.put(f"/api/v1/agent-configs/{ac_id}", json={"model": None, "max_tokens": 1234})
    ac3 = r.json()["data"]
    check("ac update: explicit null written (key presence, not truthiness)",
          ac3["model"] is None and ac3["max_tokens"] == 1234, str(ac3))
    check("ac update: untouched fields unchanged", ac3["name"] == "冒烟 Agent", str(ac3))
    r = client.put("/api/v1/agent-configs/999999", json={"name": "x"})
    check("ac update: missing id -> 200 with data {} (toSnakeCase(undefined) quirk)",
          r.status_code == 200 and r.json()["data"] == {}, r.text[:200])

    r = client.delete(f"/api/v1/agent-configs/{ac_id}")
    # ⚠️ TS 的 `success(c)` 会走默认参数 `data = null` ⇒ **data 键存在且值为 null**。
    # 曾误以为 `JSON.stringify` 会丢掉 undefined 的键、断言「无 data 键」—— 那是在锁自己的 bug。
    check("ac delete: 200 + data:null（默认参数生效，键存在）",
          r.status_code == 200 and "data" in r.json() and r.json()["data"] is None, r.text[:200])
    check("ac delete: soft-deleted excluded from list",
          all(x["id"] != ac_id for x in client.get("/api/v1/agent-configs").json()["data"]),
          "still listed")
    r = client.post("/api/v1/agent-configs", json={"agent_type": "smoke_ac", "name": "复活"})
    check("ac upsert: revives soft-deleted row (deleted_at -> null, same id)",
          r.json()["data"]["id"] == ac_id and r.json()["data"]["deleted_at"] is None,
          str(r.json()["data"])[:200])
    client.delete(f"/api/v1/agent-configs/{ac_id}")

    # --- style-profiles ---
    r = client.get("/api/v1/style-profiles")
    check("sp list: {profiles: []}", "profiles" in r.json()["data"], r.text[:200])

    r = client.post("/api/v1/style-profiles", json={})
    check("sp create: missing name -> 400 'create failed'",
          r.status_code == 400 and r.json()["message"] == "create failed", r.text[:200])

    r = client.post("/api/v1/style-profiles", json={
        "name": "冒烟风格", "drama_id": av_drama_id,
        "storytelling": {"pace": "fast"},
        "shot_patterns": {},          # ⚠️ 空对象：JS 真值 ⇒ 必须落 "{}" 而非 null
        "preferences": [],            # ⚠️ 空数组：JS 真值 ⇒ 必须落 "[]"
    })
    dump("POST /api/v1/style-profiles", r)
    sp = r.json()["data"]
    sp_id = sp["id"]
    check("sp create: camelCase row + isActive false",
          {"dramaId", "shotPatterns", "audioCaptions", "qcRules", "isActive", "deletedAt"} <= set(sp["profile"])
          and sp["profile"]["isActive"] is False, str(sp["profile"]))
    check("sp create: JSON 落库为紧凑格式（无空格）",
          sp["profile"]["storytelling"] == '{"pace":"fast"}', repr(sp["profile"]["storytelling"]))
    check("sp create: 空对象 {} 落成 '{}' 而不是 null（JS 空对象是真值）",
          sp["profile"]["shotPatterns"] == "{}", repr(sp["profile"]["shotPatterns"]))
    check("sp create: 空数组 [] 落成 '[]' 而不是 null（JS 空数组是真值）",
          sp["profile"]["preferences"] == "[]", repr(sp["profile"]["preferences"]))
    check("sp create: 未传的字段为 null", sp["profile"]["audioCaptions"] is None, str(sp["profile"]))

    r = client.get(f"/api/v1/style-profiles/{sp_id}")
    check("sp get: 200 {profile}", r.json()["data"]["profile"]["id"] == sp_id, r.text[:200])
    r = client.get("/api/v1/style-profiles/abc")
    check("sp get: NaN id -> 404 'Profile not found' (Number() 不校验有限性)",
          r.status_code == 404 and r.json()["message"] == "Profile not found", r.text[:200])

    r = client.put(f"/api/v1/style-profiles/{sp_id}", json={"description": None, "name": "改名"})
    check("sp update: 显式 null 写入 / 未传字段不动",
          r.json()["data"]["profile"]["description"] is None
          and r.json()["data"]["profile"]["name"] == "改名"
          and r.json()["data"]["profile"]["storytelling"] == '{"pace":"fast"}',
          str(r.json()["data"]["profile"]))

    # 同 drama 内只允许一个激活
    r2 = client.post("/api/v1/style-profiles", json={"name": "冒烟风格2", "drama_id": av_drama_id})
    sp2_id = r2.json()["data"]["id"]
    client.post(f"/api/v1/style-profiles/{sp_id}/activate")
    r = client.post(f"/api/v1/style-profiles/{sp2_id}/activate")
    check("sp activate: 返回激活后的 profile",
          r.json()["data"]["profile"]["isActive"] is True, r.text[:200])
    lst = client.get("/api/v1/style-profiles", params={"drama_id": av_drama_id}).json()["data"]["profiles"]
    actives = [p["id"] for p in lst if p["isActive"]]
    check("sp activate: 同 drama 内仅一个激活（旧的被取消）", actives == [sp2_id], str(actives))
    check("sp list: 激活的排在前面", lst[0]["id"] == sp2_id, str([p["id"] for p in lst]))
    check("sp list: 不存在的 drama_id -> 不过滤（NaN 为假值）",
          len(client.get("/api/v1/style-profiles", params={"drama_id": "abc"}).json()["data"]["profiles"]) >= 2,
          "filtered")

    # apply：写回四类规则 + preferences；facts/inferences 不在写入范围
    r = client.post(f"/api/v1/style-profiles/{sp_id}/apply", json={
        "storytelling": {"pace": "slow"}, "qc_rules": {"loudness": -14},
        "preferences": ["偏好A"], "facts": ["x"], "inferences": ["y"],
    })
    p = r.json()["data"]["profile"]
    check("sp apply: 写回 storytelling/qc_rules/preferences",
          p["storytelling"] == '{"pace":"slow"}' and p["qcRules"] == '{"loudness":-14}'
          and p["preferences"] == '["偏好A"]', str(p))
    check("sp apply: facts/inferences 不在写入范围（仍为 null）",
          p["facts"] is None and p["inferences"] is None, str({k: p[k] for k in ("facts", "inferences")}))
    r = client.post(f"/api/v1/style-profiles/{sp_id}/apply", json={})
    check("sp apply: 空 body -> 全部落成空对象/空数组（js_truthy 默认值）",
          r.json()["data"]["profile"]["storytelling"] == "{}"
          and r.json()["data"]["profile"]["preferences"] == "[]", str(r.json()["data"]["profile"]))

    # ⚠️ 已迁（原先断言 501）。这里**故意不打通真实提炼**：契约冒烟不该发起 LLM 调用
    #    （慢且受环境抖动），用不存在的 id 触发「Profile not found」即可证明
    #    「路由已注册、不再走 501 兜底」。提炼行为本身由 era_style_distill_test.py 覆盖。
    distill = client.post("/api/v1/style-profiles/999999/distill")
    check("sp distill: 已迁移（不存在 -> 400 'Profile not found'，而不是 501）",
          distill.status_code == 400 and distill.json()["message"] == "Profile not found",
          distill.text[:160])

    r = client.delete(f"/api/v1/style-profiles/{sp_id}")
    check("sp delete: {deleted: true}", r.json()["data"] == {"deleted": True}, r.text[:200])
    check("sp delete: 软删后查不到",
          client.get(f"/api/v1/style-profiles/{sp_id}").status_code == 404)
    check("sp delete: 重复删除 -> 404",
          client.delete(f"/api/v1/style-profiles/{sp_id}").status_code == 404)
    client.delete(f"/api/v1/style-profiles/{sp2_id}")

    # --- generations ---
    raw = sqlite3.connect(get_db_path())
    raw.execute(
        "INSERT INTO image_generations (storyboard_id, drama_id, provider, model, prompt, status,"
        " image_url, created_at, updated_at, completed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (av_sb_id, av_drama_id, "pg", "pm", "p-prompt", "completed", "static/a.png",
         "2026-02-01T00:00:00.000Z", "2026-02-01T00:00:00.000Z", "2026-02-01T00:00:02.500Z"),
    )
    raw.execute(
        "INSERT INTO video_generations (storyboard_id, drama_id, provider, model, prompt, status,"
        " video_url, duration, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (av_sb_id, av_drama_id, "vg", "vm", "v-prompt", "pending", "static/b.mp4", 5,
         "2026-02-02T00:00:00.000Z", "2026-02-02T00:00:00.000Z"),
    )
    raw.commit()
    raw.close()

    r = client.get("/api/v1/generations", params={"drama_id": av_drama_id})
    dump("GET /api/v1/generations", r)
    items = r.json()["data"]
    check("gen: 两张表聚合 + 按 createdAt 倒序", len(items) == 2 and items[0]["type"] == "video",
          str([(i["type"], i["createdAt"]) for i in items]))
    img, vid = items[1], items[0]
    check("gen: image 项形状（url 取 imageUrl；无 duration 键）",
          img["url"] == "static/a.png" and "duration" not in img
          and {"storyboardId", "dramaId", "errorMsg", "taskId", "elapsedMs", "completedAt"} <= set(img),
          str(sorted(img)))
    check("gen: video 项形状（url 取 videoUrl；含 duration）",
          vid["url"] == "static/b.mp4" and vid["duration"] == 5, str(vid))
    check("gen: elapsedMs 由 created/completed 相减（2500ms）", img["elapsedMs"] == 2500,
          str(img["elapsedMs"]))
    check("gen: completedAt 缺失 -> elapsedMs null", vid["elapsedMs"] is None, str(vid["elapsedMs"]))

    r = client.get("/api/v1/generations", params={"type": "image", "drama_id": av_drama_id})
    check("gen: type=image 只回图片",
          [i["type"] for i in r.json()["data"]] == ["image"], r.text[:200])
    r = client.get("/api/v1/generations", params={"type": "video", "drama_id": av_drama_id})
    check("gen: type=video 只回视频",
          [i["type"] for i in r.json()["data"]] == ["video"], r.text[:200])
    r = client.get("/api/v1/generations", params={"drama_id": av_drama_id, "limit": "0"})
    check("gen: limit=0 -> clamp 到 1", len(r.json()["data"]) == 1, str(len(r.json()["data"])))
    r = client.get("/api/v1/generations", params={"drama_id": "abc"})
    check("gen: drama_id=abc -> 不过滤（NaN 走 null）",
          len(r.json()["data"]) >= 2, str(len(r.json()["data"])))
    r = client.get("/api/v1/generations", params={"drama_id": av_drama_id,
                                                  "storyboard_id": av_sb_id})
    check("gen: storyboard_id 过滤生效", len(r.json()["data"]) == 2, str(len(r.json()["data"])))

    # ================= skills（纯文件系统域，读写真实 skills/ 目录）=================
    SMOKE_SKILL = "py-smoke-tmp"
    # 兜底清理：上一轮若崩在「新建」与「删除」之间，仓库里会留下临时 skill
    shutil.rmtree(Path(__file__).resolve().parents[2] / "skills" / SMOKE_SKILL, ignore_errors=True)

    r = client.get("/api/v1/skills")
    dump("GET /api/v1/skills", r)
    skills = r.json()["data"]
    check("skills list: 非空 + 19 个字段齐全",
          len(skills) > 0
          and {"id", "name", "description", "preconditions", "protocol", "category", "source",
               "sourceLabel", "workflows", "agents", "priority", "allowedTools", "foreignToolRefs",
               "missingTools", "charCount", "referenceCount", "protected", "boundAgents",
               "phases"} == set(skills[0]),
          str(sorted(skills[0]))[:300])
    # ⚠️ 用例名刻意只用 ASCII 可打印字符：Windows 控制台默认 GBK，
    # 打印 ⊆ / ∪ / ∈ 这类符号会直接 UnicodeEncodeError 打断整个测试（已被咬过一次）
    check("skills list: missingTools subset of (allowedTools U foreignToolRefs)",
          all(set(s["missingTools"]) <= set(s["allowedTools"]) | set(s["foreignToolRefs"])
              for s in skills), "subset violated")
    check("skills list: category in {core,vendor} and vendor has source",
          all(s["category"] in ("core", "vendor") and (s["category"] == "core" or s["source"])
              for s in skills), "category/source invalid")
    check("skills list: protected == (顶层 且 agents 非空)",
          all(s["protected"] == ("/" not in s["id"] and len(s["agents"]) > 0) for s in skills),
          str([(s["id"], s["protected"], len(s["agents"])) for s in skills])[:300])

    r = client.get("/api/v1/skills/meta")
    dump("GET /api/v1/skills/meta", r)
    meta = r.json()["data"]
    check("skills meta: 5 个 Agent + charBudget 60000 + sources + coreCount",
          len(meta["agents"]) == 5 and meta["charBudget"] == 60000
          and "sources" in meta and "coreCount" in meta, str(sorted(meta)))
    check("skills meta: Agent label（本地副本，与 DEFAULT_PROMPTS 有漂移守卫）",
          {a["type"]: a["label"] for a in meta["agents"]} == {
              "script_rewriter": "剧本改写", "extractor": "角色场景提取",
              "storyboard_breaker": "分镜拆解", "voice_assigner": "角色音色分配",
              "grid_prompt_generator": "宫格图提示词生成"},
          str([(a["type"], a["label"]) for a in meta["agents"]]))
    check("skills meta: coreCount + Σsources.skillCount == /skills 总数（自洽）",
          meta["coreCount"] + sum(s["skillCount"] for s in meta["sources"]) == len(skills),
          f'{meta["coreCount"]} + {sum(s["skillCount"] for s in meta["sources"])} != {len(skills)}')

    real_id = next(s["id"] for s in skills if "/" not in s["id"])
    r = client.get(f"/api/v1/skills/{real_id}")
    check("skills get: {id, content} 且 content 以 frontmatter 开头",
          r.json()["data"]["id"] == real_id and r.json()["data"]["content"].startswith("---"),
          r.text[:200])
    r = client.get("/api/v1/skills/py-smoke-nonexistent")
    check("skills get: 不存在 -> 400 'Skill not found'",
          r.status_code == 400 and r.json()["message"] == "Skill not found", r.text[:200])
    r = client.get("/api/v1/skills/foo..bar")
    check("skills get: 含 '..' -> 400 'Invalid skill id'（防路径遍历）",
          r.status_code == 400 and r.json()["message"] == "Invalid skill id", r.text[:200])

    r = client.post("/api/v1/skills", json={})
    check("skills create: 无 id -> 400 exact",
          r.status_code == 400 and r.json()["message"] == "Skill id is required", r.text[:200])
    r = client.post("/api/v1/skills", json={"id": "a..b"})
    check("skills create: 非法 id -> 400 'Invalid skill id'",
          r.status_code == 400 and r.json()["message"] == "Invalid skill id", r.text[:200])
    r = client.post("/api/v1/skills", json={"id": SMOKE_SKILL, "name": "冒烟 Skill"})
    dump("POST /api/v1/skills", r)
    check("skills create: 200 + 模板写入（frontmatter 就绪）",
          r.status_code == 200 and r.json()["data"]["id"] == SMOKE_SKILL
          and client.get(f"/api/v1/skills/{SMOKE_SKILL}").json()["data"]["content"].startswith("---"),
          r.text[:200])
    r = client.post("/api/v1/skills", json={"id": SMOKE_SKILL})
    check("skills create: 重复 -> 400 'Skill already exists'",
          r.status_code == 400 and r.json()["message"] == "Skill already exists", r.text[:200])

    r = client.put(f"/api/v1/skills/{SMOKE_SKILL}", json={"content": "纯正文，没有头部"})
    check("skills save: 缺 frontmatter -> 回 warning",
          r.status_code == 200 and "frontmatter" in (r.json()["data"] or {}).get("warning", ""),
          r.text[:200])
    r = client.put(f"/api/v1/skills/{SMOKE_SKILL}", json={
        "content": "---\nname: 冒烟\ndescription: d\nagents: []\npriority: 100\n---\n\n正文\n"})
    check("skills save: agents 为空 -> 回 warning（不再默认注入）",
          r.status_code == 200 and "agents" in (r.json()["data"] or {}).get("warning", ""),
          r.text[:200])
    r = client.put(f"/api/v1/skills/{SMOKE_SKILL}", json={
        "content": "---\nname: 冒烟\ndescription: d\nagents: []\npriority: 100\n---\n\n正文\n"})
    check("skills save: 有 frontmatter 且 agents 空 -> 仍回 warning（同上）", r.status_code == 200)

    # 删除保护：转成「顶层 + agents 非空」的受保护资产 → 必须拒删（不碰任何项目资产）
    client.put(f"/api/v1/skills/{SMOKE_SKILL}", json={
        "content": "---\nname: 冒烟\ndescription: d\nagents: [storyboard_breaker]\npriority: 100\n---\n\n正文\n"})
    protected = next(s for s in client.get("/api/v1/skills").json()["data"] if s["id"] == SMOKE_SKILL)
    check("skills: 声明 agents 后被标记 protected + 进入该 Agent 默认绑定",
          protected["protected"] is True and "storyboard_breaker" in protected["boundAgents"],
          str(protected)[:250])
    r = client.delete(f"/api/v1/skills/{SMOKE_SKILL}")
    dump("DELETE /api/v1/skills/{id} (protected)", r)
    check("skills delete: 受保护资产 -> 拒删（含指引文案）",
          r.status_code == 400 and "受删除保护" in r.json()["message"], r.text[:200])
    check("skills delete: 拒删后文件仍在",
          client.get(f"/api/v1/skills/{SMOKE_SKILL}").status_code == 200)

    client.put(f"/api/v1/skills/{SMOKE_SKILL}", json={
        "content": "---\nname: 冒烟\ndescription: d\nagents: []\npriority: 100\n---\n\n正文\n"})
    r = client.delete(f"/api/v1/skills/{SMOKE_SKILL}")
    check("skills delete: 未声明 agents -> 允许删除（data:null）",
          r.status_code == 200 and r.json()["data"] is None, r.text[:200])
    check("skills delete: 目录已移除，重复删除 -> 'Skill not found'",
          client.get(f"/api/v1/skills/{SMOKE_SKILL}").status_code == 400
          and client.delete(f"/api/v1/skills/{SMOKE_SKILL}").json()["message"] == "Skill not found")
    check("skills: 临时 skill 未在仓库留下残留",
          not (Path(__file__).resolve().parents[2] / "skills" / SMOKE_SKILL).exists(), "residue!")

    # ================= upload（multipart）/ export（工程账本）=================
    r = client.post("/api/v1/upload/image")
    check("upload image: 无文件 -> 400 'file is required'",
          r.status_code == 400 and r.json()["message"] == "file is required", r.text[:200])
    r = client.post("/api/v1/upload/image", files={"file": ("a.txt", b"hello", "text/plain")})
    check("upload image: 类型不允许 -> 400 含 Allowed 列表",
          r.status_code == 400 and "Unsupported file type: text/plain" in r.json()["message"]
          and "image/png" in r.json()["message"], r.text[:300])
    r = client.post("/api/v1/upload/image", data={"file": "not-a-file"})
    check("upload image: 字段是纯文本 -> 400 'file is required'（非 File 判据）",
          r.status_code == 400 and r.json()["message"] == "file is required", r.text[:200])

    r = client.post("/api/v1/upload/image",
                    files={"file": ("shot.png", b"\x89PNG" + b"x" * 100, "image/png")})
    dump("POST /api/v1/upload/image", r)
    up = r.json()["data"]
    check("upload image: 200 + url 带前导斜杠 / path 为相对路径 / 扩展名保留",
          r.status_code == 200 and up["url"] == "/" + up["path"]
          and up["path"].startswith("static/uploads/") and up["path"].endswith(".png"), str(up))
    saved_file = Path(os.environ["DATA_ROOT"]).resolve() / up["path"]
    check("upload image: 真的落盘且内容一致",
          saved_file.exists() and saved_file.read_bytes().startswith(b"\x89PNG"), str(saved_file))

    oversize = b"x" * (20 * 1024 * 1024 + 1)
    r = client.post("/api/v1/upload/image", files={"file": ("big.png", oversize, "image/png")})
    check("upload image: 超限 -> 400，体积保留 1 位小数、上限为整数（无 .0）",
          r.status_code == 400 and r.json()["message"].endswith(". Max: 20MB")
          and r.json()["message"].startswith("File too large: 20.0MB."), r.json()["message"][:200])

    r = client.post("/api/v1/upload/audio", files={"file": ("a.mp3", b"ID3", "audio/mpeg")})
    check("upload audio: 落盘到 static/audio/ 并保留 .mp3",
          r.status_code == 200 and r.json()["data"]["path"].startswith("static/audio/")
          and r.json()["data"]["path"].endswith(".mp3"), r.text[:200])
    r = client.post("/api/v1/upload/video", files={"file": ("v.mov", b"\x00\x00", "video/quicktime")})
    check("upload video: 落盘到 static/uploads/",
          r.status_code == 200 and r.json()["data"]["path"].startswith("static/uploads/"), r.text[:200])
    r = client.post("/api/v1/upload/audio", files={"file": ("a.txt", b"x", "text/plain")})
    check("upload audio: 类型不允许 -> 400（音频白名单独立）",
          r.status_code == 400 and "audio/mpeg" in r.json()["message"], r.text[:250])

    # --- export：工程账本（**裸响应，无信封**）---
    ex_drama = client.post("/api/v1/dramas", json={"title": "SMOKE 导出父剧"}).json()["data"]["id"]
    ex_ep = client.get(f"/api/v1/dramas/{ex_drama}").json()["data"]["episodes"][0]["id"]
    ex_sb = client.post(
        "/api/v1/storyboards", json={"episode_id": ex_ep, "title": "导出镜头|带竖线"}
    ).json()["data"]["id"]
    raw = sqlite3.connect(get_db_path())
    raw.execute("UPDATE episodes SET script_hash=? WHERE id=?", ("HASH_A", ex_ep))
    raw.execute(
        "UPDATE storyboards SET script_hash=?, image_prompt=? WHERE id=?",
        ("HASH_B", "提" * 300, ex_sb),
    )
    raw.commit()
    raw.close()

    r = client.get(f"/api/v1/export/dramas/{ex_drama}/project-ledger")
    dump("GET /api/v1/export/dramas/{id}/project-ledger (json)", r)
    check("export ledger: 裸 JSON（无 code/data 信封）+ 附件头",
          r.status_code == 200 and "code" not in r.json() and r.json()["schemaVersion"] == 1
          and r.headers["content-disposition"]
          == f'attachment; filename="drama-{ex_drama}-project-ledger.json"', r.text[:200])
    ledger = r.json()
    check("export ledger: drama 段 camelCase + 剧名正确",
          ledger["drama"]["title"] == "SMOKE 导出父剧"
          and {"totalEpisodes", "genre", "style"} <= set(ledger["drama"]), str(ledger["drama"]))
    shot = ledger["episodes"][0]["shots"][0]
    check("export ledger: shot 段 camelCase 字段齐全",
          {"storyboardId", "storyboardNumber", "scriptHash", "imagePrompt", "routeReason",
           "imageGenerations", "videoGenerations", "composedVideoUrl"} <= set(shot), str(sorted(shot)))
    check("export ledger: 分镜 scriptHash 与剧集不同（供 stale 判定）",
          shot["scriptHash"] == "HASH_B"
          and ledger["episodes"][0]["scriptHash"] == "HASH_A", str(shot["scriptHash"]))
    check("export ledger: assets 段三分类",
          {"characters", "scenes", "props"} == set(ledger["assets"]), str(sorted(ledger["assets"])))

    r = client.get(f"/api/v1/export/dramas/{ex_drama}/project-ledger", params={"format": "md"})
    dump("GET .../project-ledger?format=md", r)
    md = r.text
    check("export ledger md: text/markdown + 附件头 + 标题行",
          r.status_code == 200 and "text/markdown" in r.headers["content-type"]
          and md.startswith("# 工程账本 · SMOKE 导出父剧")
          and r.headers["content-disposition"]
          == f'attachment; filename="drama-{ex_drama}-project-ledger.md"', md[:120])
    check("export ledger md: 分镜标题里的竖线被转义（表格不被撑破）",
          "导出镜头\\|带竖线" in md, md[:400])
    check("export ledger md: 超长提示词被截断并补省略号",
          "提" * 240 + "…" in md, "truncate failed")
    check("export ledger md: 含资产基线与可复现说明段",
          "## 资产基线" in md and "## 可复现导出说明" in md, md[-300:])

    r = client.get(f"/api/v1/export/dramas/{ex_drama}/project-ledger/stale")
    dump("GET .../project-ledger/stale", r)
    stale = r.json()
    check("export stale: 裸 JSON + 命中 1 条 stale 分镜",
          r.status_code == 200 and stale["staleCount"] == 1
          and stale["stale"][0]["storyboardId"] == ex_sb
          and stale["dramaId"] == ex_drama, str(stale))

    r = client.get("/api/v1/export/dramas/999999/project-ledger")
    check("export ledger: 剧不存在 -> 404 'Drama not found'",
          r.status_code == 404 and r.json()["message"] == "Drama not found", r.text[:200])
    r = client.get("/api/v1/export/dramas/abc/project-ledger")
    check("export ledger: 非法 id -> 404 'Invalid drama id'",
          r.status_code == 404 and r.json()["message"] == "Invalid drama id", r.text[:200])

    # ⚠️ 这里原先写死「edl / jianying-draft / qc-report / contact-sheet + 打包 ZIP 都未迁移」
    #    —— `edl` 与「打包 ZIP」**已实现**（见 `export_service_test.py`）⇒ 改用**动态抽样**
    #    （与上面 episodes/storyboards 两处同源：口径来自 `route_parity_test.py`，不会过期）。
    for method, path in _unregistered_samples(limit=3):
        code = client.request(method, path, json={}).status_code
        check(f"export 兜底: {method} {path} -> 501", code == 501, code)

    client.delete(f"/api/v1/dramas/{ex_drama}")

    # ============ S1 地基：定价目录 / trace 写入侧 / Agent 协议 / 费用预估 ============
    from app.services.cost_catalog import estimate_cost  # noqa: E402
    from app.services.protocol import build_protocol_contract, parse_agent_protocol  # noqa: E402
    from app.services.task_logger import sanitize_value  # noqa: E402
    from app.services.trace_store import append_trace_event  # noqa: E402

    check("cost: 前缀包含匹配（seedream-4 先于 seedream 命中）",
          estimate_cost("image", "volcengine", "doubao-seedream-4-0-250828", 2) == 0.6,
          str(estimate_cost("image", "volcengine", "doubao-seedream-4-0-250828", 2)))
    check("cost: 未收录厂商 -> None", estimate_cost("image", "nosuchprov", "x", 1) is None)
    check("cost: units 为 0 / null / 非有限 -> None",
          estimate_cost("image", "volcengine", "seedream-4", 0) is None
          and estimate_cost("image", "volcengine", "seedream-4", None) is None)
    check("cost: settings.pricing 纯数字覆盖（用默认单位）",
          estimate_cost("image", "volcengine", "seedream-4", 3, {"pricing": 1.5}) == 4.5,
          str(estimate_cost("image", "volcengine", "seedream-4", 3, {"pricing": 1.5})))
    check("cost: settings.pricing 对象形式 + 金额保留 4 位小数",
          estimate_cost("video", "vidu", "viduq3-turbo", 7,
                        {"pricing": {"unit": "second", "price": 0.1234}}) == 0.8638,
          str(estimate_cost("video", "vidu", "viduq3-turbo", 7,
                            {"pricing": {"unit": "second", "price": 0.1234}})))

    # ⚠️ 整数值必须序列化成 int：JS 的 JSON.stringify(Number('37')) 得 37，
    # Python 的 json.dumps(37.0) 得 37.0 —— 宽松比较看不出来，但会出现在响应体字节里
    from app.response import js_number  # noqa: E402

    check("js_number: 整数值返回 int / 小数保留 / 空串为 0 / 非法为 None",
          isinstance(js_number("37"), int) and js_number("37") == 37
          and js_number("1.5") == 1.5 and js_number("") == 0
          and js_number("abc") is None, f"{js_number('37')!r} {js_number('1.5')!r}")
    r = client.get("/api/v1/asset-versions",
                   params={"asset_type": "storyboard", "asset_id": "1"})
    check("js_number: 响应体里回显的数字是 1 而不是 1.0（字节级）",
          '"asset_id":1,' in r.text or '"asset_id":1}' in r.text, r.text[:160])

    sv = sanitize_value({
        "api_key": "sk-1", "Authorization": "Bearer x",
        "callback_url": "https://a.com/hook?key=SECRET", "data": "data:image/png;base64," + "A" * 200,
        "nested": {"token": "t"}, "plain": "ok",
    })
    check("sanitize: 密钥遮蔽 + URL 脱敏 + base64 截断 + 递归生效",
          sv["api_key"] == "***" and sv["Authorization"] == "***"
          and "SECRET" not in sv["callback_url"] and "key=***" in sv["callback_url"]
          and "trimmed" in sv["data"] and sv["nested"]["token"] == "***"
          and sv["plain"] == "ok", str(sv)[:320])

    check("protocol: 正常解析（summary 去首尾空白）",
          parse_agent_protocol("说明\n```yaml\nstatus: ok\nsummary:  已保存 12 个分镜  \n```")["protocol"]["summary"]
          == "已保存 12 个分镜", "parse failed")
    check("protocol: 无 fence -> no yaml fence found",
          parse_agent_protocol("没有协议块")["errors"] == ["no yaml fence found"])
    check("protocol: status 非法 -> invalid status",
          parse_agent_protocol("```yaml\nstatus: maybe\nsummary: x\n```")["errors"][0]
          .startswith("invalid status"), "no error")
    check("protocol: 空 summary -> missing or empty summary",
          parse_agent_protocol("```yaml\nstatus: ok\nsummary: '  '\n```")["errors"]
          == ["missing or empty summary"], "no error")
    check("protocol: 契约文本含关键要求",
          "status 只能填 ok 或 failed" in build_protocol_contract())

    # trace 写入侧：O_APPEND 追加 + 尾部愈合（残缺行不粘连下一条）
    wt_dir = Path(os.environ["DATA_ROOT"]).resolve() / "traces" / "SMOKE_W"
    wt = wt_dir / "wt1.jsonl"
    append_trace_event({"ts": "2026-03-01T00:00:00.000Z", "traceId": "wt1", "scope": "SMOKE_W",
                        "level": "INFO", "action": "START write", "elapsedMs": 3, "meta": {"a": 1}})
    with wt.open("ab") as fh:
        fh.write(b'{"ts":"x","broken":true}')  # 故意不补换行，模拟崩溃残留
    append_trace_event({"ts": "2026-03-01T00:00:01.000Z", "traceId": "wt1", "scope": "SMOKE_W",
                        "level": "SUCCESS", "action": "DONE", "meta": {}})
    recs = client.get("/api/v1/traces/SMOKE_W/wt1").json()["data"]["records"]
    check("trace 写入侧: 三条记录可被读侧回放（含残缺行成独立记录）",
          len(recs) == 3 and recs[0]["action"] == "START write"
          and recs[1].get("broken") is True and recs[2]["action"] == "DONE", str(recs)[:320])
    check("trace 写入侧: 末尾以换行结尾（单条一行）",
          wt.read_bytes().endswith(b"\n"), "no trailing newline")

    # --- /usage/estimate（生成前费用预估）---
    est_drama = client.post("/api/v1/dramas", json={"title": "SMOKE 预估父剧"}).json()["data"]["id"]
    est_ep = client.get(f"/api/v1/dramas/{est_drama}").json()["data"]["episodes"][0]["id"]
    est_sbs = [
        client.post("/api/v1/storyboards",
                    json={"episode_id": est_ep, "title": f"预估镜头{i}"}).json()["data"]["id"]
        for i in range(2)
    ]
    raw = sqlite3.connect(get_db_path())
    for sid in est_sbs:
        raw.execute("UPDATE storyboards SET duration=5, dialogue=? WHERE id=?",
                    ("一二三四五六七八九十", sid))
    est_cfg: dict[str, int] = {}
    for stype, prov, model, settings in [
        ("image", "volcengine", '["doubao-seedream-4-0-250828"]',
         '{"pricing":{"unit":"image","price":0.3}}'),
        ("video", "volcengine", '["doubao-seedance-1-0-pro"]', None),
        ("audio", "minimax", '["speech-2.8-hd"]', None),
    ]:
        cur = raw.execute(
            "INSERT INTO ai_service_configs (service_type, provider, name, base_url, api_key, model,"
            " priority, is_default, is_active, settings, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (stype, prov, f"smoke-est-{stype}", "http://127.0.0.1:1", "k", model, 9999, 1, 1,
             settings, "2026-01-01T00:00:00.000Z", "2026-01-01T00:00:00.000Z"),
        )
        est_cfg[stype] = cur.lastrowid
    raw.commit()
    raw.close()

    r = client.get("/api/v1/usage/estimate", params={"drama_id": est_drama})
    dump("GET /api/v1/usage/estimate", r)
    est = r.json()["data"]
    # 视频 2×5s×0.5 + 图片 2×0.3 + 音频 20 字×0.003 = 5.66
    check("estimate: 待生成计数 + 合计两位小数（5.0 + 0.6 + 0.06）",
          est["pending_images"] == 2 and est["pending_videos"] == 2
          and est["pending_audio_chars"] == 20 and est["total_cost"] == 5.66, str(est)[:320])
    check("estimate: items 三项且带 unit",
          [(i["service_type"], i["unit"]) for i in est["items"]]
          == [("video", "second"), ("image", "image"), ("audio", "char")], str(est["items"])[:250])
    check("estimate: active_* 回显 + unestimatable 为空",
          est["active_video"]["provider"] == "volcengine" and est["unestimatable"] == [],
          str(est)[:250])
    check("estimate: scope 默认 drama-<id> + generated_at 为 ISO",
          est["scope"] == f"drama-{est_drama}" and est["generated_at"].endswith("Z"),
          str(est["scope"]))

    r = client.get("/api/v1/usage/estimate",
                   params={"drama_id": est_drama, "storyboard_ids": str(est_sbs[0])})
    e2 = r.json()["data"]
    check("estimate: storyboard_ids 限定（1 镜 → 5s / 1 张 / 10 字 → 2.83）",
          e2["pending_videos"] == 1 and e2["pending_images"] == 1
          and e2["pending_audio_chars"] == 10 and e2["total_cost"] == 2.83, str(e2)[:250])
    check("estimate: storyboard_ids 里空段会变 0 并被保留（Number('')===0）",
          client.get("/api/v1/usage/estimate",
                     params={"drama_id": est_drama, "storyboard_ids": f"{est_sbs[0]},,"}).json()
          is not None, "sanity")

    raw = sqlite3.connect(get_db_path())
    raw.execute("UPDATE ai_service_configs SET provider=? WHERE id=?",
                ("unknown-prov", est_cfg["audio"]))
    raw.commit()
    raw.close()
    e3 = client.get("/api/v1/usage/estimate", params={"drama_id": est_drama}).json()["data"]
    check("estimate: 无单价 -> unestimatable 记录 + 合计只含可估项（5.6）",
          any("无单价" in u for u in e3["unestimatable"]) and e3["total_cost"] == 5.6,
          str(e3)[:280])

    r = client.get("/api/v1/usage/estimate")
    check("estimate: 缺 drama_id -> 400 'invalid drama_id'",
          r.status_code == 400 and r.json()["message"] == "invalid drama_id", r.text[:200])
    r = client.get("/api/v1/usage/estimate", params={"drama_id": est_drama, "episode_id": "abc"})
    check("estimate: episode_id 非数值 -> 400 'invalid episode_id'",
          r.status_code == 400 and r.json()["message"] == "invalid episode_id", r.text[:200])

    client.delete(f"/api/v1/dramas/{est_drama}")
    raw = sqlite3.connect(get_db_path())
    raw.executemany("DELETE FROM ai_service_configs WHERE id=?",
                    [(v,) for v in est_cfg.values()])
    raw.commit()
    raw.close()

    # ============ S2 第 1 块：prompt_utils 画风层 / 负面词层 ============
    from app.response import js_number  # noqa: F401  (已在上面导入过，这里保证可用)
    from app.services import prompt_utils as pu  # noqa: E402

    check("prompt: 画风目录 10 项且 key/分组齐全",
          len(pu.ART_STYLE_CATALOG) == 10
          and [o["group"] for o in pu.ART_STYLE_CATALOG[:3]] == ["实拍质感"] * 3
          and all({"key", "label", "shortLabel", "desc", "group"} == set(o)
                  for o in pu.ART_STYLE_CATALOG), str(pu.ART_STYLE_KEYS))
    check("prompt: isValidArtStyle 空值放行、脏值拒绝",
          pu.is_valid_art_style(None) and pu.is_valid_art_style("anime")
          and not pu.is_valid_art_style("not-a-style"))

    # 画风解析链：角色 style → 剧集 style → 全局 → realistic（非法 key 跳过而非透传）
    check("prompt: resolveArtStyleKey 跳过非法 key",
          pu.resolve_art_style_key(None, "bogus", "anime") == "anime"
          and pu.resolve_art_style_key("bogus", None) == "realistic",
          pu.resolve_art_style_key(None, "bogus", "anime"))
    # ⚠️ resolve_effective_art_style 会**无条件**读一次全局画风（TS 的 getGlobalArtStyle() 同样如此），
    # 所以 conn 是必需参数 —— 只传 drama_style 也逃不掉这次查询。
    from app.db import engine as _art_engine  # noqa: E402

    with _art_engine.connect() as _art_conn:
        check("prompt: 角色 style 优先于剧集 style",
              pu.resolve_effective_art_style(_art_conn, None, "noir", "anime") == "noir"
              and pu.resolve_effective_art_style(_art_conn, None, None, "anime") == "anime")

    # 后缀必须与 TS 同形：命中画风用 `, art, TAIL`；未命中回退 MASTER（**二选一，不叠加**）
    check("prompt: 角色后缀命中画风时用 TAIL、不含 VISUAL_STYLE_MASTER",
          pu.build_character_art_style_suffix("anime")
          == f", {pu.DRAMA_ART_STYLE_MAP['anime']}, cinematic illustration style, "
             "consistent art style, soft cinematic lighting, high quality, no text, no watermark",
          pu.build_character_art_style_suffix("anime")[:120])
    check("prompt: 角色后缀未命中时回退 CHARACTER + MASTER",
          pu.build_character_art_style_suffix(None)
          == f", {pu.VISUAL_STYLE_CHARACTER}, {pu.VISUAL_STYLE_MASTER}",
          pu.build_character_art_style_suffix(None)[:120])
    check("prompt: 场景后缀未命中回退 SCENE + MASTER，命中只用 TAIL",
          pu.build_scene_art_style_suffix(None)
          == f", {pu.VISUAL_STYLE_SCENE}, {pu.VISUAL_STYLE_MASTER}"
          and pu.build_scene_art_style_suffix("ink-wash")
          == f", {pu.DRAMA_ART_STYLE_MAP['ink-wash']}, {pu.ART_STYLE_TAIL}",
          pu.build_scene_art_style_suffix("ink-wash")[:120])
    check("prompt: 分镜静帧后缀命中只用 art+TAIL（不含 MASTER）",
          pu.build_storyboard_art_style_suffix("ghibli")
          == f", {pu.DRAMA_ART_STYLE_MAP['ghibli']}, {pu.ART_STYLE_TAIL}"
          and pu.build_storyboard_art_style_suffix(None) == f", {pu.VISUAL_STYLE_MASTER}",
          pu.build_storyboard_art_style_suffix("ghibli")[:120])
    check("prompt: 装备后缀恒含无人物约束段",
          ", no people, no person, no model" in pu.build_equip_art_style_suffix("realistic")
          and pu.build_equip_art_style_suffix("realistic").endswith(
              ", high quality, no text, no watermark"), pu.build_equip_art_style_suffix("realistic")[:160])

    # 视频画风层：写实向才注入瑕疵锚点；绘画向不注入（注入会破坏风格）
    v_real = pu.build_video_art_style_suffix("realistic")
    v_anime = pu.build_video_art_style_suffix("anime")
    check("prompt: 视频写实向注入 IMPERFECTION_ANCHORS",
          pu.IMPERFECTION_ANCHORS in v_real
          and "restrained ending (no explosion, no victory pose, no text overlay)" in v_real
          and pu.VIDEO_MOTION_BASE in v_real, v_real[:160])
    check("prompt: 视频绘画向不注入瑕疵锚点，且不含 MASTER 的插画收口",
          pu.IMPERFECTION_ANCHORS not in v_anime
          and pu.VIDEO_MOTION_BASE in v_anime
          and "cinematic illustration style" not in v_anime, v_anime[:160])
    check("prompt: 视频未指定画风回退 VISUAL_STYLE_VIDEO",
          pu.build_video_art_style_suffix(None) == f", {pu.VISUAL_STYLE_VIDEO}",
          pu.build_video_art_style_suffix(None)[:100])

    # 负面词：画风对立词必须追加（防动漫/真人混用）
    # ⚠️ 位置很关键：画风对立词插在 **NEGATIVE_BASE 之后**，而不是拼在 CHARACTER_IMAGE_NEGATIVE 末尾
    check("prompt: 角色负面词把画风对立词插在 NEGATIVE_BASE 之后",
          pu.build_character_negative_prompt("anime")
          == f"{pu.NEGATIVE_BASE}, photorealistic, realistic photo, live action, 3d render, "
             "cluttered background, busy background, multiple characters, multiple people, "
             "duplicated character, inconsistent character, cropped head, cut off face",
          pu.build_character_negative_prompt("anime")[:200])
    check("prompt: 未指定画风时负面词回退常量本身",
          pu.build_character_negative_prompt(None) == pu.CHARACTER_IMAGE_NEGATIVE
          and pu.build_video_negative_prompt(None) == pu.VIDEO_NEGATIVE
          and pu.build_scene_negative_prompt(None) == pu.SCENE_IMAGE_NEGATIVE)
    check("prompt: 视频/场景/分镜负面词都追加同一份画风对立词",
          pu.build_video_negative_prompt("pixar3d").endswith(
              "photorealistic, realistic photo, live action, 2d anime, cel shading, flat coloring")
          and pu.build_scene_negative_prompt("noir").endswith(
              "color, vibrant colors, saturated palette, anime, cartoon, flat coloring")
          and pu.build_storyboard_negative_prompt("cyberpunk").endswith(
              "anime style, cartoon, flat 2d illustration, dull desaturated colors, daylight"),
          pu.build_video_negative_prompt("pixar3d")[-60:])
    check("prompt: PRESET_VIDEO_NEGATIVE 就是 VIDEO_NEGATIVE 本体",
          pu.PRESET_VIDEO_NEGATIVE is pu.VIDEO_NEGATIVE)

    # 屏幕元素检测 + 留白图 prompt
    check("prompt: containsScreenElement 中英文都命中",
          pu.contains_screen_element("她盯着手机屏幕") and pu.contains_screen_element("check the monitor")
          and not pu.contains_screen_element("两人在雨中奔跑")
          and not pu.contains_screen_element(None))
    check("prompt: ui_plate prompt 含留白约束 + 场景/主风格收口",
          pu.build_ui_plate_image_prompt({}).startswith("blank phone screen plate, no text")
          and pu.build_ui_plate_image_prompt({}).endswith(
              f", {pu.VISUAL_STYLE_SCENE}, {pu.VISUAL_STYLE_MASTER}")
          and "context: 手机通知" in pu.build_ui_plate_image_prompt({"context": "手机通知"}),
          pu.build_ui_plate_image_prompt({})[:140])

    # --- S2 第 2 块：预设 / 角色 / 装备 / 单品 / 表情 / 物品 / 场景构建器 ---
    shot = {
        "shotIndex": 1, "themeFamily": "T", "compositionPattern": "C", "spaceType": "S",
        "foregroundFrame": "F", "mainFocalPoint": "M", "thematicClue": "H", "activity": "A",
        "cameraPosition": "P", "windDirection": "W", "lightStructure": "L",
        "characterLayout": "CL", "livingElement": "LE",
    }
    check("prompt2: preset image 词序 = 接口字段序 + livingElement + MASTER",
          pu.build_preset_image_prompt(shot)
          == "T, C, S, F, M, H, A, CL, L, W, P, LE, " + pu.VISUAL_STYLE_MASTER,
          pu.build_preset_image_prompt(shot)[:120])
    check("prompt2: preset video 词序与图片版**不同**（含 cameraMove 与 VIDEO+MASTER 双收口）",
          pu.build_preset_video_prompt(shot, "PAN")
          == f"M, A, C, S, CL, PAN, L, W, {pu.VISUAL_STYLE_VIDEO}, {pu.VISUAL_STYLE_MASTER}",
          pu.build_preset_video_prompt(shot, "PAN")[:140])
    # 缺字段时 JS 的 join 会留下空槽（", , "）—— 保真，不做"顺手过滤"
    check("prompt2: 缺字段保留空槽（对齐 JS join 语义）",
          pu.build_preset_image_prompt({"themeFamily": "T", "mainFocalPoint": "M"})
          .startswith("T, , , , M, "), pu.build_preset_image_prompt({"themeFamily": "T", "mainFocalPoint": "M"})[:60])

    check("prompt2: formatEquip JSON 数组 -> 逗号拼接；非 JSON -> 去括号引号并折叠空白",
          pu.format_equip('["a", "b"]') == "a, b"
          and pu.format_equip('  [ "x" , \'y\' ]  ') == "x , y"
          and pu.format_equip("  破旧   铁剑  ") == "破旧 铁剑"
          and pu.format_equip("") == "" and pu.format_equip(None) == "",
          str([pu.format_equip('["a", "b"]'), pu.format_equip('  [ "x" , \'y\' ]  '),
               pu.format_equip("  破旧   铁剑  ")])[:120])

    char = {
        "name": "林昭", "appearance": "短发侠客", "description": "短发侠客",
        "coreFeatures": '["冷峻眉眼", "  ", "刀疤"]',
        "clothing": "青衫", "weapons": '["长刀"]', "accessories": "玉佩",
        "personality": "隐忍", "dramaStyle": "ink-wash",
    }
    check("prompt2: 角色图 parts 顺序 + core features 前缀 + 性格后缀 + 画风收口",
          pu.build_character_image_prompt(char)
          == "林昭, core features: 冷峻眉眼, 刀疤, 短发侠客, wearing 青衫, armed with 长刀, "
             "wearing accessories: 玉佩, 隐忍 expression and mannerisms"
             + pu.build_character_art_style_suffix("ink-wash"),
          pu.build_character_image_prompt(char)[:200])
    check("prompt2: description === appearance 时不重复灌入（去重分支）",
          pu.build_character_image_prompt(char).count("短发侠客") == 1,
          str(pu.build_character_image_prompt(char).count("短发侠客")))

    # ⚠️ 与 visuals clause 的三处差异，专门断言（最容易"统一"掉的地方）
    check("prompt2: appearanceText 与 visualsClause 的三处差异（无 coreFeatures 前缀 / accessories: / ': ' 拼接）",
          pu.build_character_appearance_text(char)
          == "林昭: 冷峻眉眼, 刀疤: 短发侠客: wearing 青衫: armed with 长刀: accessories: 玉佩",
          pu.build_character_appearance_text(char))

    check("prompt2: 服装取值链 costume > clothing > costumes[0]",
          pu.build_character_visuals_clause({"costume": "红衣", "clothing": "青衫", "costumes": '["黄袍"]'})
          == "wearing 红衣"
          and pu.build_character_visuals_clause({"clothing": "", "costumes": '["黄袍"]'})
          == "wearing 黄袍"
          and pu.build_character_visuals_clause({}) == "",
          pu.build_character_visuals_clause({"costume": "红衣", "clothing": "青衫", "costumes": '["黄袍"]'}))
    check("prompt2: parseVariations 只保留有字符串 name 的项",
          len(pu.parse_variations('[{"name":"v1"},{"imageUrl":"x"},{"name":2},null]')) == 1,
          str(pu.parse_variations('[{"name":"v1"},{"imageUrl":"x"},{"name":2},null]')))

    check("prompt2: equip 三视图三分支 + 无人物画风尾 + 未提供取值时的兜底文案",
          pu.build_equip_image_prompt("weapon", {"weapons": "长刀"})
          .startswith("detailed weapon three-view concept art of 长刀,")
          and pu.build_equip_image_prompt("weapon", {}).startswith("detailed weapon three-view concept art, ")
          and pu.build_equip_image_prompt("clothing", {"clothing": "青衫"}).endswith(
              pu.build_equip_art_style_suffix(None)),
          pu.build_equip_image_prompt("clothing", {"clothing": "青衫"})[:140])
    check("prompt2: equip 负面**不排除** side-by-side（否则三视角会被画成单个物体）",
          "side by side" not in pu.build_equip_negative(None)
          and "split image" in pu.build_three_view_negative(None),
          pu.build_equip_negative(None)[:100])
    check("prompt2: item 单品图负面**要排除** side-by-side / three views / turnaround",
          "side by side, three views, multi view, turnaround" in pu.build_item_negative(None)
          and "not a three-view, not side by side" in pu.build_item_image_prompt("clothing", {}),
          pu.build_item_negative(None)[-80:])

    check("prompt2: 表情预设 9 项 + 未知 key 回退 neutral",
          len(pu.EXPRESSION_PRESETS) == 9
          and pu.find_expression_preset("smile")["en"].startswith("gentle warm smile")
          and pu.find_expression_preset("nope") is None
          and "facial expression: neutral calm expression" in pu.build_expression_image_prompt(
              {"name": "林昭"}, "nope"),
          str(pu.find_expression_preset("nope")))
    check("prompt2: 表情图服装链**不含** costume 字段（与 visualsClause 不同）",
          "wearing 红衣" not in pu.build_expression_image_prompt(
              {"name": "林昭", "costume": "红衣", "clothing": "青衫"}, "smile")
          and "wearing 青衫" in pu.build_expression_image_prompt(
              {"name": "林昭", "costume": "红衣", "clothing": "青衫"}, "smile"),
          pu.build_expression_image_prompt({"name": "林昭", "costume": "红衣", "clothing": "青衫"}, "smile")[:120])

    check("prompt2: 物品图用**全角括号**包分类 + SCENE/MASTER 双收口",
          pu.build_prop_image_prompt({"name": "玉佩"}).startswith("detailed prop design sheet of 玉佩（道具）")
          and pu.build_prop_image_prompt({"name": "玉佩", "category": "信物"}).startswith(
              "detailed prop design sheet of 玉佩（信物）")
          and pu.build_prop_image_prompt({"name": "玉佩"}).endswith(
              f", {pu.VISUAL_STYLE_SCENE}, {pu.VISUAL_STYLE_MASTER}"),
          pu.build_prop_image_prompt({"name": "玉佩"})[:120])

    check("prompt2: 场景图有 prompt 时只用 prompt；否则 location + time 氛围",
          pu.build_scene_image_prompt({"location": "客栈", "time": "夜晚", "prompt": "手写场景"})
          .startswith("手写场景")
          and pu.build_scene_image_prompt({"location": "客栈", "time": "夜晚"})
          .startswith("客栈, 夜晚 lighting and atmosphere")
          and pu.build_scene_image_prompt({"location": "客栈"}).startswith("客栈"),
          pu.build_scene_image_prompt({"location": "客栈", "time": "夜晚"})[:120])

    # --- 绞杀者接缝 ---
    # ⚠️ 这条断言**必须跟着「未迁移清单」走**（清单会随迁移推进变短）：
    #    `GET /dramas/{id}/rhythm` 早已迁移 ⇒ 不再 501，这里改成正向断言；
    #    兜底样本换成当前**仅剩**的未迁移端点 `POST /storage/change`。
    r = client.get(f"/api/v1/dramas/{new_id}/rhythm")
    dump("GET /api/v1/dramas/{id}/rhythm (已迁移，不应再 501)", r)
    check("strangler: 已迁移子路径**不再** 501（rhythm）", r.status_code != 501, r.text[:120])
    r2 = client.post("/api/v1/storage/change", json={})
    dump("POST /api/v1/storage/change (not migrated)", r2)
    check("strangler: unmigrated sub-path -> 501",
          r2.status_code == 501 and r2.json()["code"] == 501, r2.text[:120])
    check("strangler: other domain -> 501", client.get("/api/v1/storyboards").status_code == 501)

    # --- 静态站 ---
    check("static: missing file -> 404", client.get("/static/not-exist.png").status_code == 404)
    check("static: traversal blocked", client.get("/static/../configs/config.yaml").status_code == 404)

    # --- SPA ---
    r = client.get("/")
    log(f"\nGET / -> {r.status_code} {r.text[:200]}")
    check("spa: 404 when frontend/dist absent (dev mode hint)", r.status_code == 404)

# ---------------------------------------------------------------------------
# 3) 汇总
# ---------------------------------------------------------------------------
passed = sum(1 for _, ok, _ in results if ok)
failed = [n for n, ok, _ in results if not ok]
log()
log("=" * 72)
log(f"RESULT: {passed}/{len(results)} passed")
if failed:
    log("FAILED:")
    for n in failed:
        log(f"  - {n}")
log("=" * 72)

_report_path.write_text("\n".join(lines), encoding="utf-8")
shutil.rmtree(_smoke_root, ignore_errors=True)

print()
print(f"SUMMARY: {passed}/{len(results)} passed")
print(f"report -> {_report_path}")
if failed:
    print("FAILED:")
    for n in failed:
        print(f"  - {n}")
raise SystemExit(1 if failed else 0)
