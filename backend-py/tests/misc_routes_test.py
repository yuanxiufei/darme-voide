"""多域自检：`visual-graph` / `images` / `webhooks`（S4 收尾的三个零散域）。

三处**错了不报错、只是结果不对**的地方：

1. **visual-graph 的返回形态与其它域不同**：前三个端点是**裸 JSON**（无 code/data 信封）、
   ``Content-Type`` 带 charset、``/guidance`` 是 **text/plain**；``category`` 非法时
   ``GET /`` 返回**全部**而 ``GET /terms`` 直接 400；
2. **images 的负面词二选一**：有 ``storyboard_id`` 用分镜负面词，**没有则用 ``NEGATIVE_BASE``**；
   返回的是 **camelCase 行**、查不到是 ``null``；
3. **webhooks 的三种返回码**：缺 task_id → 400；**task_id 查不到 → 200**（防重复回调）；
   ``state=failed`` → **200**（已记录）；下载失败才 400，且要落 ``failed`` + 中文归因错误。

运行::

    ./.venv/Scripts/python.exe tests/misc_routes_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="misc_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import (  # noqa: E402
    characters,
    image_generations,
    scenes,
    storyboard_characters,
    storyboards,
    video_generations,
)
from app.core.response import now  # noqa: E402
from app.routers import images as images_router  # noqa: E402
from app.routers import webhooks as webhooks_router  # noqa: E402
from app.services import prompt_utils as pu  # noqa: E402
from app.services import visual_graph as vg  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


_IMAGE_CALLS: list[dict] = []
_DOWNLOAD_CALLS: list[tuple] = []
_DOWNLOAD_FAIL = [False]
_DURATION = [12]


async def _fake_generate_image(conn, params: dict) -> int:
    _IMAGE_CALLS.append(dict(params))
    with engine.begin() as c:
        return int(c.execute(image_generations.insert().values(
            drama_id=params.get("dramaId"), prompt=params.get("prompt"),
            storyboard_id=params.get("storyboardId"),
            status="completed", local_path="static/images/out.png",
            frame_type=params.get("frameType"), provider="minimax",
            created_at=now(), updated_at=now(),
        )).lastrowid)


async def _fake_download(url: str, sub_dir: str) -> str:
    _DOWNLOAD_CALLS.append((url, sub_dir))
    if _DOWNLOAD_FAIL[0]:
        raise RuntimeError("boom download")
    return "static/videos/vidu.mp4"


async def _fake_probe(_path: str) -> int:
    return _DURATION[0]


def _row(table, row_id):
    with engine.begin() as conn:
        return conn.execute(select(table).where(table.c.id == row_id)).first()


def main() -> int:  # noqa: C901
    # ⚠️ patch **路由模块**里的名字（路由是 `from ... import x`，改服务模块对它无效）
    images_router.generate_image = _fake_generate_image  # type: ignore[assignment]
    webhooks_router.download_file = _fake_download  # type: ignore[assignment]
    webhooks_router.probe_video_duration = _fake_probe  # type: ignore[assignment]

    client = TestClient(app)

    # ================= visual-graph =================
    root = client.get("/api/v1/visual-graph")
    root_body = root.json()
    check(
        "图谱: GET / 是**裸 JSON**（无 code/data 信封）+ 显式 charset",
        set(root_body) == {"graph", "categories"}
        and root.json() and root.headers["content-type"] == "application/json; charset=utf-8",
        sorted(root_body),
    )
    check(
        "图谱: categories 四类且 node 总数 41",
        root_body["categories"] == ["shot_size", "composition", "movement", "lighting"]
        and sum(len(v) for v in root_body["graph"].values()) == 41,
        {k: len(v) for k, v in root_body["graph"].items()},
    )
    only_movement = client.get("/api/v1/visual-graph?category=movement").json()
    check("图谱: category 过滤生效（只回 movement 一类）",
          set(only_movement["graph"]) == {"movement"}, list(only_movement["graph"]))
    check("图谱: category 非法 -> **返回全部**（不当错）",
          len(client.get("/api/v1/visual-graph?category=bogus").json()["graph"]) == 4)
    check("图谱: /resolve 命中术语", client.get(
        "/api/v1/visual-graph/resolve?text=近景").json() == {"zh": "近景", "en": "close-up"})
    check("图谱: /resolve 未命中 -> en 为 null",
          client.get("/api/v1/visual-graph/resolve?text=不存在词").json()["en"] is None)
    check("图谱: /resolve 缺 text -> 400 固定文案",
          client.get("/api/v1/visual-graph/resolve").json()["message"] == 'query param "text" is required')
    check("图谱: /terms 非法 category -> 400 且列出可选值",
          client.get("/api/v1/visual-graph/terms?category=x").json()["message"]
          == "category must be one of: shot_size | composition | movement | lighting")
    check("图谱: /terms 合法 -> 与图谱节点数一致",
          len(client.get("/api/v1/visual-graph/terms?category=lighting").json()["terms"])
          == len(vg.VISUAL_GRAPH["lighting"]))
    guidance = client.get("/api/v1/visual-graph/guidance")
    check(
        "图谱: /guidance 是 **text/plain** 且非空",
        guidance.status_code == 200 and guidance.headers["content-type"].startswith("text/plain")
        and len(guidance.text) > 50,
        guidance.headers.get("content-type"),
    )

    # ================= 造数据 =================
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE misc"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]
    check("图谱: /guidance 传不存在的 drama_id -> 404",
          client.get("/api/v1/visual-graph/guidance?drama_id=999999").status_code == 404)
    ok_guidance = client.get(f"/api/v1/visual-graph/guidance?drama_id={drama_id}")
    check("图谱: /guidance 传存在的 drama_id -> 200 text/plain",
          ok_guidance.status_code == 200 and ok_guidance.headers["content-type"].startswith("text/plain"))
    check("图谱: /guidance 非数字 drama_id -> 走默认引导（200）",
          client.get("/api/v1/visual-graph/guidance?drama_id=abc").status_code == 200)

    char_id = _insert_character(drama_id, "林昭", image_url="static/images/c.png")
    scene_id = client.post("/api/v1/scenes", json={
        "drama_id": drama_id, "location": "客栈", "time": "夜晚"}).json()["data"]["id"]
    with engine.begin() as conn:
        conn.execute(scenes.update().where(scenes.c.id == scene_id)
                     .values(image_url="static/images/s.png"))
    sb_id = client.post("/api/v1/storyboards", json={
        "episode_id": episode_id, "title": "镜头1", "storyboard_number": 1}).json()["data"]["id"]
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_id).values(
            scene_id=scene_id, description="少年入门", location="客栈",
            shot_type="wide", angle="eye level"))
        conn.execute(storyboard_characters.insert().values(storyboard_id=sb_id, character_id=char_id))

    # ================= images =================
    check("images: 缺 prompt -> 400",
          client.post("/api/v1/images", json={}).json()["message"] == "prompt is required")

    _IMAGE_CALLS.clear()
    created = client.post("/api/v1/images", json={
        "storyboard_id": sb_id, "drama_id": drama_id, "prompt": "两人对视"})
    call = _IMAGE_CALLS[-1]
    check("images: 201 + **camelCase** 行（storyboardId 等）",
          created.status_code == 201 and "storyboardId" in created.json()["data"],
          created.status_code)
    check(
        "images: 有 storyboard_id -> 富化 prompt（角色外观 + 场景 + 分镜字段）",
        "Characters" in call["prompt"] and "少年入门" in call["prompt"],
        call["prompt"][:120],
    )
    check(
        "images: 有 storyboard_id -> 负面词用**分镜那套**（含画风对立词）",
        call["negativePrompt"] == pu.build_storyboard_negative_prompt("realistic")
        or call["negativePrompt"] != pu.NEGATIVE_BASE,
        call["negativePrompt"][:60],
    )
    check(
        "images: 自动收集参考图（角色立绘 + 场景图）",
        len(call["referenceImages"] or []) >= 2, call["referenceImages"],
    )

    _IMAGE_CALLS.clear()
    client.post("/api/v1/images", json={"prompt": "裸出图"})
    bare = _IMAGE_CALLS[-1]
    check(
        "images: **无** storyboard_id -> 负面词是 NEGATIVE_BASE（与分镜那套不同）",
        bare["negativePrompt"] == pu.NEGATIVE_BASE, bare["negativePrompt"][:60],
    )
    _IMAGE_CALLS.clear()
    client.post("/api/v1/images", json={
        "storyboard_id": sb_id, "prompt": "原样", "_skip_enrich": True})
    check("images: _skip_enrich -> prompt 不被富化（原样带走）",
          _IMAGE_CALLS[-1]["prompt"] == "原样", _IMAGE_CALLS[-1]["prompt"])

    gen_id = created.json()["data"]["id"]
    got = client.get(f"/api/v1/images/{gen_id}").json()
    check("images: GET /:id 命中 -> camelCase 行", got["data"]["id"] == gen_id and "localPath" in got["data"],
          sorted(got["data"])[:6])
    check("images: GET /:id 查不到 -> success(null)（不是 404）",
          client.get("/api/v1/images/999999").json()["data"] is None)
    check("images: GET /:id 非法 id -> 404", client.get("/api/v1/images/abc").status_code == 404)
    listed = client.get(f"/api/v1/images?storyboard_id={sb_id}").json()["data"]
    check("images: GET / 按 storyboard_id 过滤",
          listed and all(row["storyboardId"] == sb_id for row in listed), len(listed))
    check("images: 非法过滤值当没传（不过滤）",
          len(client.get("/api/v1/images?storyboard_id=abc").json()["data"])
          == len(client.get("/api/v1/images").json()["data"]))
    check("images: DELETE 生效",
          client.delete(f"/api/v1/images/{gen_id}").status_code == 200
          and client.get(f"/api/v1/images/{gen_id}").json()["data"] is None)

    # ================= webhooks =================
    check("webhooks: 缺 task_id -> 400 Missing task_id",
          client.post("/api/v1/webhooks/vidu", json={"state": "success"}).json()["message"]
          == "Missing task_id")
    not_found = client.post("/api/v1/webhooks/vidu", json={"task_id": "nope", "state": "success"})
    check(
        "webhooks: task_id 查不到 -> **200**（防厂商重复回调）",
        not_found.status_code == 200 and not_found.json()["data"] == {"message": "Task not found"},
        not_found.json(),
    )

    gen = _insert_video_generation(drama_id, storyboard_id=sb_id, task_id="tk-1",
                                   status="processing")
    done = client.post("/api/v1/webhooks/vidu", json={
        "task_id": "tk-1", "state": "success", "video_url": "https://cdn.example.com/v.mp4"})
    row = _row(video_generations, gen)
    sb_row = _row(storyboards, sb_id)
    check(
        "webhooks: 成功回调 -> 下载 + 记录 completed（local_path/video_url）",
        done.status_code == 200 and row.status == "completed"
        and row.local_path == "static/videos/vidu.mp4"
        and row.video_url == "https://cdn.example.com/v.mp4",
        (row.status, row.local_path),
    )
    check("webhooks: 下载目录是 `videos`", _DOWNLOAD_CALLS[-1] == ("https://cdn.example.com/v.mp4", "videos"),
          _DOWNLOAD_CALLS[-1])
    check(
        "webhooks: 回写 storyboards（video_url = 本地路径 + ffprobe 探测的时长）",
        sb_row.video_url == "static/videos/vidu.mp4" and sb_row.duration == 12,
        (sb_row.video_url, sb_row.duration),
    )

    # 探测不到时长 -> **不写** duration 列（对齐 JS 的 undefined 丢键）
    _DURATION[0] = 0
    gen2 = _insert_video_generation(drama_id, storyboard_id=sb_id, task_id="tk-2", status="processing")
    client.post("/api/v1/webhooks/vidu", json={
        "task_id": "tk-2", "state": "success", "video_url": "https://cdn.example.com/v2.mp4"})
    row2 = _row(video_generations, gen2)
    sb_row2 = _row(storyboards, sb_id)
    check(
        "webhooks: 探不到时长 -> duration 保持原值（不写 null）",
        row2.status == "completed" and sb_row2.duration == 12,
        sb_row2.duration,
    )
    _DURATION[0] = 12

    gen3 = _insert_video_generation(drama_id, storyboard_id=sb_id, task_id="tk-3", status="processing")
    failed = client.post("/api/v1/webhooks/vidu", json={
        "task_id": "tk-3", "state": "failed", "error": {"code": "Moderation", "message": "内容不合规"}})
    row3 = _row(video_generations, gen3)
    check(
        "webhooks: state=failed -> **200** + 中文归因错误落库（命中内容安全分支）",
        failed.status_code == 200 and failed.json()["data"] == {"message": "Error recorded"}
        and row3.status == "failed"
        # ⚠️ 归因层识别为「内容安全」=> 落库的是**可行动的中文提示**，不是厂商原文
        and "内容安全" in (row3.error_msg or ""),
        (failed.status_code, row3.error_msg),
    )

    gen4 = _insert_video_generation(drama_id, storyboard_id=sb_id, task_id="tk-4", status="processing")
    noted = client.post("/api/v1/webhooks/vidu", json={"task_id": "tk-4", "state": "processing"})
    row4 = _row(video_generations, gen4)
    check(
        "webhooks: 其它状态 -> 200 Status noted，且**不动**记录",
        noted.json()["data"] == {"message": "Status noted"} and row4.status == "processing",
        row4.status,
    )

    _DOWNLOAD_FAIL[0] = True
    gen5 = _insert_video_generation(drama_id, storyboard_id=sb_id, task_id="tk-5", status="processing")
    broke = client.post("/api/v1/webhooks/vidu", json={
        "task_id": "tk-5", "state": "success", "video_url": "https://cdn.example.com/v5.mp4"})
    row5 = _row(video_generations, gen5)
    _DOWNLOAD_FAIL[0] = False
    check(
        "webhooks: 下载失败 -> 400，且记录 failed + `Webhook download failed:` 前缀",
        broke.status_code == 400 and row5.status == "failed"
        and (row5.error_msg or "").startswith("Webhook download failed:"),
        (broke.status_code, row5.error_msg),
    )

    # 共享密钥（模块级常量：测试里直接改属性）
    original_secret = webhooks_router.WEBHOOK_SECRET
    webhooks_router.WEBHOOK_SECRET = "s3cret"  # type: ignore[assignment]
    unauth = client.post("/api/v1/webhooks/vidu", json={"task_id": "tk-9", "state": "success"})
    check("webhooks: 配了密钥但没带头 -> 400 Unauthorized",
          unauth.status_code == 400 and unauth.json()["message"] == "Unauthorized", unauth.json())
    authed = client.post("/api/v1/webhooks/vidu", json={"task_id": "tk-9", "state": "success"},
                         headers={"x-webhook-secret": "s3cret"})
    check("webhooks: 带对头 -> 放行（走到 task 查不到分支 => 200）",
          authed.status_code == 200, authed.status_code)
    webhooks_router.WEBHOOK_SECRET = original_secret  # type: ignore[assignment]

    # ================= 汇总 =================
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


def _insert_character(drama_id: int, name: str, **extra) -> int:
    values: dict[str, object] = {"drama_id": drama_id, "name": name}
    for column in ("created_at", "updated_at"):
        if column in characters.c:
            values[column] = now()
    values.update(extra)
    with engine.begin() as conn:
        return int(conn.execute(characters.insert().values(**values)).lastrowid)


def _insert_video_generation(drama_id: int, **fields) -> int:
    # ⚠️ updated_at 是 NOT NULL
    values: dict[str, object] = {
        "drama_id": drama_id, "status": "processing",
        "created_at": now(), "updated_at": now(),
    }
    values.update(fields)
    with engine.begin() as conn:
        return int(conn.execute(video_generations.insert().values(**values)).lastrowid)


if __name__ == "__main__":
    raise SystemExit(main())
