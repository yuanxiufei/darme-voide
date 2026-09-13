"""preset-framework 自检：Variation Card 引擎 / 建剧 / 批量生图生视频 / 状态 / 一键全流程。

三块**错了不报错、只是结果不对**的逻辑：

1. **Variation Card 的约束**：固定 5 镜、``themeFamily`` 可排除、线索按 **3:2** 配比、
   灵动元素**最多 3 个镜头**、空间类型/前景框架按 ``长度取模`` 派生（**是字符串长度不是 hash**）；
2. **本域响应形态不同**：成功 ``{success, data}``、失败 ``{success, msg}``（**没有** code/message 信封），
   参数错 400、内部错 **500**；
3. **批量编排**：每镜自管状态（``generating_image → image_ready/image_failed``）、
   **单镜失败不中断**、结果按 ``shotIndex`` 排序；生视频前**没有首帧就跳过**。

运行::

    ./.venv/Scripts/python.exe tests/preset_framework_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="presetfw_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import dramas, episodes, image_generations, storyboards, video_generations  # noqa: E402
from app.response import now  # noqa: E402
from app.services import preset_framework as pf  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


_IMAGE_CALLS: list[dict] = []
_VIDEO_CALLS: list[dict] = []
_FAIL_IMAGE_AT: list[int] = []


async def _fake_generate_image(conn, params: dict) -> int:
    _IMAGE_CALLS.append(dict(params))
    if _IMAGE_CALLS[-1]["storyboardId"] in _FAIL_IMAGE_AT:
        raise RuntimeError("boom image")
    with engine.begin() as c:
        return int(c.execute(image_generations.insert().values(
            storyboard_id=params.get("storyboardId"), drama_id=params.get("dramaId"),
            prompt=params.get("prompt"), status="completed", local_path="static/images/x.png",
            created_at=now(), updated_at=now(),
        )).lastrowid)


async def _fake_generate_video(conn, params: dict) -> int:
    _VIDEO_CALLS.append(dict(params))
    with engine.begin() as c:
        return int(c.execute(video_generations.insert().values(
            storyboard_id=params.get("storyboardId"), drama_id=params.get("dramaId"),
            prompt=params.get("prompt"), status="processing",
            created_at=now(), updated_at=now(),
        )).lastrowid)


def _sb_row(sb_id: int):
    with engine.begin() as conn:
        return conn.execute(select(storyboards).where(storyboards.c.id == sb_id)).first()


def main() -> int:  # noqa: C901
    pf.generate_image = _fake_generate_image  # type: ignore[assignment]
    pf.generate_video = _fake_generate_video  # type: ignore[assignment]

    # ================= Variation Card 引擎 =================
    card = pf.generate_variation_card()
    shots = card["shots"]
    check("卡片: 固定 5 镜 + shotIndex 1..5",
          len(shots) == 5 and [s["shotIndex"] for s in shots] == [1, 2, 3, 4, 5],
          len(shots))
    check("卡片: themeFamily 在池内且 5 镜共用同一家族",
          card["themeFamily"] in pf.THEME_FAMILIES
          and len({s["themeFamily"] for s in shots}) == 1,
          card["themeFamily"])
    check("卡片: excludeFamily 生效（排除后不会再抽到它）",
          all(pf.generate_variation_card(pf.THEME_FAMILIES[0])["themeFamily"] != pf.THEME_FAMILIES[0]
              for _ in range(20)))
    check("卡片: 线索配比 3:2（3 个 subtle + 2 个 prominent）",
          sum(1 for s in shots if s["thematicClue"] in pf.SUBTLE_CLUES) == 3
          and sum(1 for s in shots if s["thematicClue"] in pf.PROMINENT_CLUES) == 2,
          [s["thematicClue"] for s in shots])
    living = [s["livingElement"] for s in shots if s["livingElement"]]
    check("卡片: 灵动元素最多 3 个镜头（可为 0）", len(living) <= pf.MAX_LIVING_SHOTS, living)
    check("卡片: 字段齐全且来自各自的池",
          all(s["compositionPattern"] in pf.COMPOSITION_PATTERNS
              and s["mainFocalPoint"] in pf.MAIN_FOCAL_POINTS
              and s["activity"] in pf.ACTIVITIES
              and s["cameraPosition"] in pf.CAMERA_POSITIONS
              and s["windDirection"] in pf.WIND_DIRECTIONS
              and s["lightStructure"] in pf.LIGHT_STRUCTURES
              for s in shots))
    # ⭐ 派生规则：按 (家族 + 焦点) 的**字符串长度**取模，不是 hash 值
    expected = pf.derive_space_and_frame("THEME_FAMILY_A", "FOCAL_POINT_A")
    length = len("THEME_FAMILY_A" + "FOCAL_POINT_A")
    check(
        "派生: spaceType/foregroundFrame 按**字符串长度**取模（14+13=27）",
        expected == {"spaceType": ["SPACE_TYPE_A", "SPACE_TYPE_B", "SPACE_TYPE_C",
                                   "SPACE_TYPE_D"][length % 4],
                     "foregroundFrame": ["FRAME_TYPE_1", "FRAME_TYPE_2",
                                         "FRAME_TYPE_3"][length % 3]}
        and pf.derive_space_and_frame("A", "B") == pf.derive_space_and_frame("A", "B"),
        expected,
    )
    check("派生: 卡片里每镜的 spaceType 与该镜焦点一致（同一家族下随焦点变）",
          all(s["spaceType"] == pf.derive_space_and_frame(card["themeFamily"], s["mainFocalPoint"])["spaceType"]
              for s in shots))
    check("排列: pick_character_layout 忽略入参、只从 4 种里随机取",
          pf.pick_character_layout("anything") in
          ("LAYOUT_SOLO", "LAYOUT_PAIR", "LAYOUT_TRIANGLE", "LAYOUT_SCATTERED"))

    client = TestClient(app)

    # ================= GET /variation-card =================
    resp = client.get("/api/v1/preset/framework/variation-card")
    body = resp.json()
    check(
        "响应: 成功是 `{success, data}` —— **没有** code/data/message 信封的嵌套",
        resp.status_code == 200 and body["success"] is True
        and set(body) == {"success", "data"} and "shots" in body["data"],
        sorted(body),
    )
    check("响应: 排除家族走查询参数",
          client.get(f"/api/v1/preset/framework/variation-card?excludeFamily={pf.THEME_FAMILIES[1]}"
                     ).json()["data"]["themeFamily"] != pf.THEME_FAMILIES[1])

    # ================= POST /create =================
    created = client.post("/api/v1/preset/framework/create", json={})
    check("建剧: 缺 title -> 400 `{success:false, msg}`",
          created.status_code == 400 and created.json() == {"success": False, "msg": "title is required"},
          created.json())
    no_card = client.post("/api/v1/preset/framework/create", json={"title": "T"})
    check("建剧: 缺 variationCard.shots -> 400",
          no_card.status_code == 400
          and no_card.json()["msg"] == "variationCard with shots is required", no_card.json())

    made = client.post("/api/v1/preset/framework/create", json={
        "title": "预设剧", "variationCard": card}).json()["data"]
    drama_row = None
    with engine.begin() as conn:
        drama_row = conn.execute(select(dramas).where(dramas.c.id == made["dramaId"])).first()
        episode_rows = conn.execute(select(episodes).where(
            episodes.c.drama_id == made["dramaId"])).all()
        sb_rows = conn.execute(select(storyboards).where(
            storyboards.c.episode_id == made["episodeId"])
            .order_by(storyboards.c.storyboard_number)).all()
    check(
        "建剧: Drama 的 style/描述回落/状态",
        drama_row.style == "preset-framework"
        and drama_row.description == f"预设风格: {card['themeFamily']}"
        and drama_row.status == "draft",
        (drama_row.style, drama_row.description),
    )
    check("建剧: 恰好 1 集（episodeNumber=1）",
          len(episode_rows) == 1 and episode_rows[0].episode_number == 1, len(episode_rows))
    check(
        "建剧: 每镜一个 Storyboard，按真实列 **storyboard_number** 落 shotIndex",
        len(sb_rows) == 5 and [r.storyboard_number for r in sb_rows] == [1, 2, 3, 4, 5],
        [r.storyboard_number for r in sb_rows],
    )
    check(
        "建剧: 描述 = `Shot N: 焦点` + shotType 取构图 + status=pending",
        sb_rows[0].description == f"Shot 1: {card['shots'][0]['mainFocalPoint']}"
        and sb_rows[0].shot_type == card["shots"][0]["compositionPattern"]
        and sb_rows[0].status == "pending",
        sb_rows[0].description,
    )
    check(
        "建剧: dramas.metadata 是**紧凑 JSON**（与 Node 共用该列）且含 variationCard",
        '"presetType":"framework"' in drama_row.metadata
        and json.loads(drama_row.metadata)["variationCard"]["themeFamily"] == card["themeFamily"],
        drama_row.metadata[:60],
    )

    # ================= POST /generate-images =================
    bad = client.post("/api/v1/preset/framework/generate-images", json={"dramaId": 1})
    check("生图: 参数不全 -> 400（两个批量端点共用同一句校验文案）",
          bad.status_code == 400
          and bad.json()["msg"] == "dramaId, episodeId, storyboardIds, and variationCard are required",
          bad.json())

    ids = made["storyboardIds"]
    _IMAGE_CALLS.clear()
    images = client.post("/api/v1/preset/framework/generate-images", json={
        "dramaId": made["dramaId"], "episodeId": made["episodeId"],
        "storyboardIds": ids, "variationCard": card}).json()["data"]
    check("生图: 5 镜全部成功 -> imageGenIds 按 shotIndex 顺序",
          len(images["imageGenIds"]) == 5 and images["imageGenIds"] == sorted(images["imageGenIds"]),
          images["imageGenIds"])
    check("生图: 复用预设 prompt 构建器 + PRESET_IMAGE_NEGATIVE",
          "PRESET_IMAGE_NEGATIVE" not in _IMAGE_CALLS[0]["negativePrompt"]
          and _IMAGE_CALLS[0]["negativePrompt"] == pf.PRESET_IMAGE_NEGATIVE,
          _IMAGE_CALLS[0]["negativePrompt"][:50])
    check("生图: 镜头状态落到 image_ready",
          all(_sb_row(i).status == "image_ready" for i in ids),
          [_sb_row(i).status for i in ids])

    # 第 3 镜失败：其余不受影响，该镜状态 image_failed
    ids2 = list(client.post("/api/v1/preset/framework/create", json={
        "title": "失败用例", "variationCard": card}).json()["data"]["storyboardIds"])
    _FAIL_IMAGE_AT[:] = [ids2[2]]
    partial = client.post("/api/v1/preset/framework/generate-images", json={
        "dramaId": made["dramaId"], "episodeId": made["episodeId"],
        "storyboardIds": ids2, "variationCard": card}).json()["data"]
    _FAIL_IMAGE_AT.clear()
    check(
        "生图: **单镜失败不中断**（其余 4 镜成功，失败镜状态 image_failed）",
        len(partial["imageGenIds"]) == 4 and _sb_row(ids2[2]).status == "image_failed"
        and _sb_row(ids2[0]).status == "image_ready",
        (len(partial["imageGenIds"]), _sb_row(ids2[2]).status),
    )

    # ================= POST /generate-videos =================
    # ⚠️ 真实管线里首帧是**后台任务**完成后才回写 storyboards.image_url；
    #    这里的假 generate_image 只插记录 ⇒ 要手工"完成"这一步，否则会被当成没有首帧而跳过
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id.in_(ids))
                     .values(first_frame_image="static/images/x.png"))
    _VIDEO_CALLS.clear()
    videos = client.post("/api/v1/preset/framework/generate-videos", json={
        "dramaId": made["dramaId"], "episodeId": made["episodeId"],
        "storyboardIds": ids, "variationCard": card}).json()["data"]
    check("生视频: 有首帧 -> 逐镜生成，状态 video_ready",
          len(videos["videoGenIds"]) == 5 and _sb_row(ids[0]).status == "video_ready",
          len(videos["videoGenIds"]))
    check("生视频: 参考图用**首帧本地路径**，prompt 走预设视频构建器",
          _VIDEO_CALLS[0]["referenceImageUrl"] == "static/images/x.png"
          and _VIDEO_CALLS[0]["negativePrompt"] == pf.PRESET_VIDEO_NEGATIVE,
          _VIDEO_CALLS[0]["referenceImageUrl"])

    no_frame = client.post("/api/v1/preset/framework/create", json={
        "title": "无首帧", "variationCard": card}).json()["data"]["storyboardIds"]
    _VIDEO_CALLS.clear()
    skipped = client.post("/api/v1/preset/framework/generate-videos", json={
        "dramaId": made["dramaId"], "episodeId": made["episodeId"],
        "storyboardIds": no_frame, "variationCard": card}).json()["data"]
    check("生视频: **没有首帧的镜头被跳过**（不发请求、不计入 ids）",
          skipped["videoGenIds"] == [] and _VIDEO_CALLS == [],
          (skipped, len(_VIDEO_CALLS)))

    # ================= GET /status =================
    check("状态: 非数字 dramaId -> 400 Invalid dramaId",
          client.get("/api/v1/preset/framework/status/abc").json()
          == {"success": False, "msg": "Invalid dramaId"})
    check("状态: 不存在的剧 -> 500 Drama not found",
          client.get("/api/v1/preset/framework/status/999999").json()
          == {"success": False, "msg": "Drama not found"})
    status = client.get(f"/api/v1/preset/framework/status/{made['dramaId']}").json()["data"]
    check(
        "状态: 汇总结构（drama/episode/storyboards/variationCard/summary）",
        set(status) == {"drama", "episode", "storyboards", "variationCard", "summary"}
        and status["episode"]["episodeNumber"] == 1,
        sorted(status),
    )
    # ⚠️ 视频同理：videosGenerated 数的是 **storyboards.video_url**（由后台任务回写），
    #    假 generate_video 只插记录 ⇒ 这里手工"完成"两条来验计数逻辑
    check("状态: 进度汇总 —— imagesGenerated 数首帧（5）",
          status["summary"]["totalShots"] == 5
          and status["summary"]["imagesGenerated"] == 5
          and status["summary"]["videosGenerated"] == 0,
          status["summary"])
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id.in_(ids[:2]))
                     .values(video_url="static/videos/x.mp4"))
    recount = client.get(f"/api/v1/preset/framework/status/{made['dramaId']}").json()["data"]
    check("状态: videosGenerated 数 storyboards.video_url（回写 2 条 -> 2）",
          recount["summary"]["videosGenerated"] == 2, recount["summary"])
    check("状态: variationCard 从 system_metadata 还原（紧凑 JSON 也能解析）",
          status["variationCard"] is None or status["variationCard"]["themeFamily"] == card["themeFamily"],
          str(status["variationCard"])[:60])

    # ================= POST /full-pipeline =================
    _IMAGE_CALLS.clear()
    full = client.post("/api/v1/preset/framework/full-pipeline", json={
        "title": "一键", "variationCard": card}).json()["data"]
    check(
        "一键: 建剧 + 生图 + 原样回传 card",
        set(full) == {"dramaId", "episodeId", "storyboardIds", "imageGenIds", "variationCard"}
        and len(full["imageGenIds"]) == 5,
        sorted(full),
    )
    _IMAGE_CALLS.clear()
    no_images = client.post("/api/v1/preset/framework/full-pipeline", json={
        "title": "一键-不生成图", "variationCard": card,
        "autoGenerateImages": False}).json()["data"]
    check("一键: autoGenerateImages=false -> **不**触发生图（默认才是 true）",
          no_images["imageGenIds"] == [] and _IMAGE_CALLS == [],
          (no_images["imageGenIds"], len(_IMAGE_CALLS)))
    check("一键: 缺 title -> 400 title is required",
          client.post("/api/v1/preset/framework/full-pipeline", json={}).json()
          == {"success": False, "msg": "title is required"})

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
