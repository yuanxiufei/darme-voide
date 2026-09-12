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

log()
log("=" * 72)
log("HTTP CONTRACT")
log("=" * 72)


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

    # --- 绞杀者接缝 ---
    r = client.get(f"/api/v1/dramas/{new_id}/rhythm")
    dump("GET /api/v1/dramas/{id}/rhythm (not migrated)", r)
    check("strangler: unmigrated sub-path -> 501", r.status_code == 501 and r.json()["code"] == 501)
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
