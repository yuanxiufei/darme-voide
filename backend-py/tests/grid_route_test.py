"""grid 自检：宫格 prompt / 出图 / 切分 / 状态（``routers/grid.py`` + ``services/grid_split.py``）。

三块**错了不报错、只是结果不对**的逻辑：

1. **Agent 载荷归一**：``grid_prompt`` 与 ``cell_prompts`` 都有 camel/snake 两套键名，
   且**没有 prompt 的格子要被丢掉**、``grid_prompt`` 为空则整体判为「不可用」⇒ 回落本地构建器；
   还要能从 ```json 围栏、裸 ```{...}```、嵌套对象/数组、甚至**字符串化的 JSON** 里抠出来；
2. **宫格出图参数**：画布 ``960*cols x 540*rows``、``frameType=grid_<mode>_<r>x<c>``、
   负面词用**分镜那套**（含画风对立词）、参考图 = 收集到的资产路径；
3. **切分回写**：``reference`` 分支是把格子**追加**到已有数组（且该列与 Node 共用 ⇒ 紧凑 JSON）。

运行::

    ./.venv/Scripts/python.exe tests/grid_route_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="gridroute_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import characters, image_generations, scenes, storyboard_characters, storyboards  # noqa: E402
from app.response import now  # noqa: E402
from app.routers import grid as grid_router  # noqa: E402
from app.services import grid_split as gs  # noqa: E402
from app.services import prompt_utils as pu  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


_IMAGE_CALLS: list[dict] = []
_SPLIT_CALLS: list[tuple] = []
_FAKE_CELLS = [
    {"index": 0, "localPath": "static/grid-cells/cell_1_0.png"},
    {"index": 1, "localPath": "static/grid-cells/cell_1_1.png"},
    {"index": 2, "localPath": "static/grid-cells/cell_1_2.png"},
]


async def _fake_generate_image(conn, params: dict) -> int:
    _IMAGE_CALLS.append(dict(params))
    with engine.begin() as c:
        return int(c.execute(image_generations.insert().values(
            drama_id=params.get("dramaId"), prompt=params.get("prompt"),
            status="completed", local_path="static/grid/grid.png",
            frame_type=params.get("frameType"), created_at=now(), updated_at=now(),
        )).lastrowid)


async def _fake_split(image_path: str, rows: int, cols: int) -> list[dict]:
    _SPLIT_CALLS.append((image_path, rows, cols))
    return _FAKE_CELLS[: rows * cols]


def main() -> int:  # noqa: C901
    grid_router.generate_image = _fake_generate_image  # type: ignore[assignment]
    # ⚠️ 必须 patch **路由模块**里的名字：它 `from ... import split_grid_image`，
    #    改 `gs.split_grid_image` 对它无效（名字已绑定）。
    grid_router.split_grid_image = _fake_split  # type: ignore[assignment]

    # ================= 尺寸校验（调**真**的切分函数，未被替换前的引用） =================
    for bad_rows, bad_cols in ((0, 2), (2, 0), (1.5, 2)):
        err = ""
        try:
            asyncio.run(_REAL_SPLIT("static/grid/g.png", bad_rows, bad_cols))
        except ValueError as exc:
            err = str(exc)
        check(
            f"切分校验: rows={bad_rows}, cols={bad_cols} -> 正整数校验报错",
            f"Invalid grid dimensions: rows={bad_rows}, cols={bad_cols}." in err,
            err,
        )

    # ================= Agent 载荷归一 =================
    norm = grid_router._normalize_grid_payload
    find = grid_router._find_grid_payload
    check(
        "载荷: snake/camel 两套键名都能收（cellPrompts + shotNumber/frameType）",
        norm({"gridPrompt": " P ", "cellPrompts": [
            {"shotNumber": 2, "frameType": "last_frame", "prompt": " x "}]})
        == {"grid_prompt": "P",
            "cell_prompts": [{"shot_number": 2, "frame_type": "last_frame", "prompt": "x"}]},
    )
    check(
        "载荷: 没有 prompt 的格子被丢掉；grid_prompt 为空 -> 整体判不可用（None）",
        norm({"grid_prompt": "", "cell_prompts": [{"prompt": "x"}]}) is None
        and norm({"grid_prompt": "G", "cell_prompts": [{"shot_number": 1}, {"prompt": "ok"}]})
        == {"grid_prompt": "G", "cell_prompts": [{"shot_number": 0,
                                                  "frame_type": "first_frame", "prompt": "ok"}]},
    )
    check("载荷: 非对象 -> None", norm("abc") is None and norm(None) is None)
    check(
        "载荷: 字符串化的 JSON 也能递归解析",
        find('{"grid_prompt":"G","cell_prompts":[]}') == {"grid_prompt": "G", "cell_prompts": []},
    )
    check(
        "载荷: ```json 围栏里的 JSON 能抠出来",
        find('前言\n```json\n{"grid_prompt":"F","cell_prompts":[]}\n```\n后记')
        == {"grid_prompt": "F", "cell_prompts": []},
    )
    check("载荷: 裸 {...} 也能抠出来",
          find('prefix {"grid_prompt":"B"} suffix') == {"grid_prompt": "B", "cell_prompts": []})
    check("载荷: 嵌套数组/对象里能找到", find([{"a": 1}, {"b": {"grid_prompt": "N"}}])
          == {"grid_prompt": "N", "cell_prompts": []})
    check("载荷: 字符串 'null' / 非 JSON 文本 -> None",
          find("null") is None and find("完全没有 JSON") is None)
    check(
        "载荷: shot_number 非数字 -> 0（`Number(x) || 0`）",
        norm({"grid_prompt": "G", "cell_prompts": [{"shot_number": "abc", "prompt": "p"}]})
        ["cell_prompts"][0]["shot_number"] == 0,
    )

    # ================= 造数据 =================
    client = TestClient(app)
    client.post("/api/v1/ai-configs", json={
        "service_type": "image", "provider": "minimax", "base_url": "https://x.example.com",
        "api_key": "k", "model": ["image-01"], "is_active": True,
    })
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 宫格路由"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]

    char_id = _insert_character(drama_id, "林昭", image_url="static/images/c.png")
    scene_id = client.post("/api/v1/scenes", json={
        "drama_id": drama_id, "location": "客栈", "time": "夜晚"}).json()["data"]["id"]
    with engine.begin() as conn:
        conn.execute(scenes.update().where(scenes.c.id == scene_id)
                     .values(image_url="static/images/s.png"))

    def new_sb(number: int, **fields) -> int:
        sid = client.post("/api/v1/storyboards", json={
            "episode_id": episode_id, "title": f"镜头{number}", "storyboard_number": number,
        }).json()["data"]["id"]
        if fields:
            with engine.begin() as conn:
                conn.execute(storyboards.update().where(storyboards.c.id == sid).values(**fields))
        return sid

    sb1 = new_sb(1, scene_id=scene_id, first_frame_image="static/images/f1.png",
                 description="少年入门", movement="推镜")
    sb2 = new_sb(2, scene_id=scene_id, first_frame_image="static/images/f2.png",
                 description="回头一瞥", movement="左摇")
    with engine.begin() as conn:
        conn.execute(storyboard_characters.insert().values(storyboard_id=sb1, character_id=char_id))
        conn.execute(storyboard_characters.insert().values(storyboard_id=sb2, character_id=char_id))

    # ================= POST /grid/prompt =================
    check("prompt: 缺 storyboard_ids -> 400",
          client.post("/api/v1/grid/prompt", json={"rows": 1, "cols": 2}).status_code == 400)
    check("prompt: 缺 rows/cols -> 400",
          client.post("/api/v1/grid/prompt", json={"storyboard_ids": [sb1]}).status_code == 400)

    resp = client.post("/api/v1/grid/prompt", json={
        "storyboard_ids": [sb1, sb2], "drama_id": drama_id, "episode_id": episode_id,
        "rows": 1, "cols": 2, "mode": "first_frame"})
    body = resp.json()["data"]
    check(
        "prompt: Agent 未移植 -> **回落**本地构建器（source=fallback）",
        resp.status_code == 200 and body["source"] == "fallback", body.get("source"),
    )
    check("prompt: 返回 grid/storyboard_ids/mode 三件套",
          body["grid"] == {"rows": 1, "cols": 2}
          and body["storyboard_ids"] == [sb1, sb2] and body["mode"] == "first_frame",
          body["grid"])
    check(
        "prompt: grid_prompt 是本地构建器产物（含画风层与格标签）",
        "1x2 grid layout" in body["grid_prompt"] and "格1（row 1 col 1）" in body["grid_prompt"],
        body["grid_prompt"][:80],
    )
    check("prompt: cell_prompts 逐格（本例 2 格）", len(body["cell_prompts"]) == 2,
          len(body["cell_prompts"]))
    check(
        "prompt: 分镜里存的首帧被收成参考图（图1/图2 进 legend）",
        "图片1" in body["grid_prompt"] or "参考图片" in body["grid_prompt"],
        body["grid_prompt"][:120],
    )
    check("prompt: 空 storyboard_ids 数组 -> 400",
          client.post("/api/v1/grid/prompt", json={
              "storyboard_ids": [], "rows": 1, "cols": 1}).status_code == 400)

    # ================= POST /grid/generate =================
    _IMAGE_CALLS.clear()
    gen = client.post("/api/v1/grid/generate", json={
        "storyboard_ids": [sb1, sb2], "drama_id": drama_id, "rows": 2, "cols": 2,
        "mode": "first_last"})
    gen_body = gen.json()["data"]
    call = _IMAGE_CALLS[-1]
    check(
        "generate: 画布 = 960*cols x 540*rows（2x2 -> 1920x1080）",
        call["size"] == "1920x1080", call["size"],
    )
    check("generate: frameType = `grid_<mode>_<r>x<c>`",
          call["frameType"] == "grid_first_last_2x2", call["frameType"])
    with engine.begin() as conn:
        resolved_style = pu.resolve_effective_art_style(conn, drama_id)
    check(
        "generate: 负面词用**分镜那套**（同一 resolved 画风 => 与分镜静帧负面词逐字相同）",
        call["negativePrompt"] == pu.build_storyboard_negative_prompt(resolved_style),
        call["negativePrompt"][:80],
    )
    check(
        "generate: 参考图 = 收集到的资产路径（含角色立绘/场景图/分镜首帧）",
        len(call["referenceImages"]) >= 3 and "static/images/c.png" in call["referenceImages"],
        call["referenceImages"],
    )
    check(
        "generate: 响应含 image_generation_id/prompt/reference_images",
        set(gen_body) == {"image_generation_id", "grid", "mode", "storyboard_ids",
                          "prompt", "reference_images"}
        and gen_body["grid"] == {"rows": 2, "cols": 2},
        sorted(gen_body),
    )
    _IMAGE_CALLS.clear()
    client.post("/api/v1/grid/generate", json={
        "storyboard_ids": [sb1], "drama_id": drama_id, "rows": 1, "cols": 1,
        "custom_prompt": "我的手写提示词"})
    check("generate: custom_prompt 覆盖构建器", _IMAGE_CALLS[-1]["prompt"] == "我的手写提示词",
          _IMAGE_CALLS[-1]["prompt"])
    check("generate: 缺 rows/cols -> 400",
          client.post("/api/v1/grid/generate", json={"storyboard_ids": [sb1]}).status_code == 400)

    # ================= POST /grid/split =================
    gen_id = _insert_generation(drama_id, status="completed", local_path="static/grid/grid.png")
    check("split: 缺 image_generation_id -> 400",
          client.post("/api/v1/grid/split", json={"rows": 1, "cols": 2}).status_code == 400)
    check("split: 缺 assignments -> 400",
          client.post("/api/v1/grid/split", json={
              "image_generation_id": gen_id, "rows": 1, "cols": 2}).status_code == 400)
    check("split: 图片记录不存在 -> 400",
          client.post("/api/v1/grid/split", json={
              "image_generation_id": 999999, "rows": 1, "cols": 1,
              "assignments": [{"storyboard_id": sb1, "frame_type": "first_frame"}]}).status_code == 400)
    pending_id = _insert_generation(drama_id, status="pending", local_path="static/grid/p.png")
    not_done = client.post("/api/v1/grid/split", json={
        "image_generation_id": pending_id, "rows": 1, "cols": 1,
        "assignments": [{"storyboard_id": sb1, "frame_type": "first_frame"}]})
    check("split: 图片还没完成 -> 400 `Image status: pending`",
          "Image status: pending" in not_done.json()["message"], not_done.json().get("message"))
    no_file_id = _insert_generation(drama_id, status="completed", local_path=None)
    check("split: 没有本地文件 -> 400 No local image file",
          "No local image file" in client.post("/api/v1/grid/split", json={
              "image_generation_id": no_file_id, "rows": 1, "cols": 1,
              "assignments": [{"storyboard_id": sb1, "frame_type": "first_frame"}]}
          ).json()["message"])

    # 先给 sb1 放一个已有的参考图数组，验证 reference 是**追加**
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb1)
                     .values(reference_images='["static/images/old.png"]'))
    _SPLIT_CALLS.clear()
    split = client.post("/api/v1/grid/split", json={
        "image_generation_id": gen_id, "rows": 1, "cols": 3,
        "assignments": [
            {"storyboard_id": sb1, "frame_type": "first_frame"},
            {"storyboard_id": sb2, "frame_type": "last_frame"},
            {"storyboard_id": sb1, "frame_type": "reference"},
        ]})
    split_body = split.json()["data"]
    check("split: 三格全部回写（cells 三条）", len(split_body["cells"]) == 3, split_body["cells"])
    check("split: 切图入参是记录里的 local_path + 行列",
          _SPLIT_CALLS[-1] == ("static/grid/grid.png", 1, 3), _SPLIT_CALLS[-1])
    with engine.begin() as conn:
        row1 = conn.execute(select(storyboards).where(storyboards.c.id == sb1)).first()
        row2 = conn.execute(select(storyboards).where(storyboards.c.id == sb2)).first()
    check("split: first_frame -> first_frame_image", row1.first_frame_image == _FAKE_CELLS[0]["localPath"],
          row1.first_frame_image)
    check("split: last_frame -> last_frame_image", row2.last_frame_image == _FAKE_CELLS[1]["localPath"],
          row2.last_frame_image)
    check(
        "split: reference -> **追加**到已有数组末尾，且是紧凑 JSON（与 Node 共用该列）",
        row1.reference_images == '["static/images/old.png","static/grid-cells/cell_1_2.png"]',
        row1.reference_images,
    )
    check(
        "split: 响应字段是 snake_case（storyboard_id/frame_type/local_path）",
        set(split_body["cells"][0]) == {"storyboard_id", "frame_type", "local_path"},
        sorted(split_body["cells"][0]),
    )
    truncated = client.post("/api/v1/grid/split", json={
        "image_generation_id": gen_id, "rows": 1, "cols": 1,
        "assignments": [{"storyboard_id": sb1, "frame_type": "first_frame"},
                        {"storyboard_id": sb2, "frame_type": "first_frame"}]}
    ).json()["data"]
    check(
        "split: assignments 比格子多时**按格子数截断**（1x1 => 只回写 1 格，不越界）",
        len(truncated["cells"]) == 1, truncated["cells"],
    )

    # ================= GET /grid/status/{id} =================
    status_body = client.get(f"/api/v1/grid/status/{gen_id}").json()["data"]
    check(
        "status: 返回 id/status/local_path/image_url/error_msg 五字段",
        set(status_body) == {"id", "status", "local_path", "image_url", "error_msg"}
        and status_body["status"] == "completed",
        sorted(status_body),
    )
    check("status: 非法 id -> 404", client.get("/api/v1/grid/status/abc").status_code == 404)
    check("status: 记录不存在 -> 404", client.get("/api/v1/grid/status/999999").status_code == 404)

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


#: 真切分函数（``main`` 里会把 ``gs.split_grid_image`` 换掉，这里留引用测校验分支）
_REAL_SPLIT = gs.split_grid_image


def _insert_generation(drama_id: int, **fields) -> int:
    # ⚠️ updated_at 是 NOT NULL（漏了会 IntegrityError）
    values: dict[str, object] = {
        "drama_id": drama_id, "status": "completed",
        "created_at": now(), "updated_at": now(),
    }
    values.update(fields)
    with engine.begin() as conn:
        return int(conn.execute(image_generations.insert().values(**values)).lastrowid)


def _insert_character(drama_id: int, name: str, **extra) -> int:
    values: dict[str, object] = {"drama_id": drama_id, "name": name}
    for column in ("created_at", "updated_at"):
        if column in characters.c:
            values[column] = now()
    values.update(extra)
    with engine.begin() as conn:
        return int(conn.execute(characters.insert().values(**values)).lastrowid)


if __name__ == "__main__":
    raise SystemExit(main())
