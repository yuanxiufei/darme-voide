"""逐镜路由 + videos 域路由自检（``shot_router.py`` / ``routers/videos.py``）。

重点覆盖两块**决策类**逻辑（错了不会报错、只会生成路线不对或丢上下文）：

1. **`decide_shot_route` 的优先级**：blocked → T2V/prevTail → FL2VA → R2V(对话) →
   R2V(Ref2VA 音频) → keyframe → 默认 I2V。``prevTail`` 必须在 T2V **之前**消费，
   否则顺接帧不会被 adapter 派发（尾帧顺接形同虚设）；
2. **`POST /videos` 的上下文富化 + 帧来源统一 + referenceMode 降级** —— 这一段是修 bug 的：
   决策读 DB 帧、请求只读 body 帧会产出「判定 first_last 却一帧都不带」的坏请求。

运行::

    ./.venv/Scripts/python.exe tests/videos_route_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="vroute_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import characters, scenes, storyboard_characters, storyboards, video_generations  # noqa: E402
from app.core.response import now  # noqa: E402
from app.services import shot_router as sr  # noqa: E402
from app.services import video_generation as vg  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:  # noqa: C901
    client = TestClient(app)
    # 路由内部的 generate_video 是 fire-and-forget ⇒ 隔离掉后台任务（否则会真去请求厂商）
    vg._spawn = lambda coro: coro.close()  # type: ignore[assignment]

    # 活跃视频配置：provider 用 volcengine（在 MULTI_REF 白名单里，R2V 才能触发）
    client.post("/api/v1/ai-configs", json={
        "service_type": "video", "provider": "volcengine", "base_url": "https://ark.example.com",
        "api_key": "k", "model": ["doubao-seedance-1-5-pro-251215"], "is_active": True,
    })

    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 视频路由"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]

    def new_storyboard(**overrides) -> int:
        body = {"episode_id": episode_id, "title": "镜头", **overrides}
        return client.post("/api/v1/storyboards", json=body).json()["data"]["id"]

    # ================= decide_shot_route 优先级 =================
    def decide(sb_id: int, **kwargs) -> dict:
        base = {
            "storyboardId": sb_id, "sceneType": None, "firstFrameImage": None,
            "lastFrameImage": None, "keyframeImage": None, "blocked": False,
            "provider": "volcengine", "canMultiRef": True, "referenceImages": [],
            "referenceAudioUrls": [], "referenceVideoUrls": [], "prevTail": None,
        }
        base.update(kwargs)
        with engine.begin() as conn:
            return sr.decide_shot_route(conn, base)

    sb = new_storyboard()
    d = decide(sb, blocked=True)
    check(
        "路由: ① blocked 最高优先（即便有首尾帧也阻断）",
        d == {"route": "blocked",
              "reason": "first frame asset needs_regeneration，视频生成被资产门禁阻断",
              "referenceMode": "none"},
        d,
    )
    d = decide(sb, prevTail="static/images/prev.png", firstFrameImage=None)
    check(
        "路由: ② 无首帧 + 有 prevTail -> I2V 起帧（**不能**落 T2V，否则顺接帧白给）",
        d["route"] == "first_frame_to_video" and d["referenceMode"] == "single"
        and "同场景顺接提供上一镜尾帧" in d["reason"],
        d,
    )
    d = decide(sb, firstFrameImage=None)
    check(
        "路由: ②' 无首帧且无 prevTail -> T2V",
        d["route"] == "text_to_video" and d["referenceMode"] == "none",
        d,
    )
    d = decide(sb, firstFrameImage="f.png", lastFrameImage="l.png")
    check(
        "路由: ③ 有尾帧 -> FL2VA（首尾帧连接）",
        d["route"] == "first_last_frame" and d["referenceMode"] == "first_last",
        d,
    )
    d = decide(sb, firstFrameImage="f.png", lastFrameImage="l.png", keyframeImage="k.png")
    check("路由: ③ 优先级高于 keyframe（尾帧更明确）", d["route"] == "first_last_frame")

    d = decide(sb, firstFrameImage="f.png", sceneType="dialogue",
               referenceImages=["a.png", "b.png"])
    check(
        "路由: ④ 对话场景 + 多参考 + 有参考图 -> R2V/multiple",
        d["route"] == "reference_to_video" and d["referenceMode"] == "multiple"
        and "对话/多人场景（dialogue）" in d["reason"] and "2 张" in d["reason"],
        d,
    )
    d = decide(sb, firstFrameImage="f.png", sceneType="dialogue", referenceImages=["a.png"])
    check(
        "路由: ④ 中文「多人」也命中（DIALOGUE_PATTERN 含中文）",
        decide(sb, firstFrameImage="f.png", sceneType="多人混战", referenceImages=["a.png"])["route"]
        == "reference_to_video",
    )
    d = decide(sb, firstFrameImage="f.png", sceneType="action", referenceAudioUrls=["v.mp3"],
               provider="minimax")
    check(
        "路由: ④b minimax + 参考音频（无对话）-> Ref2VA，referenceMode 按有无参考图取 single",
        d["route"] == "reference_to_video" and d["referenceMode"] == "single"
        and "Ref2VA，1 条声线样本" in d["reason"],
        d,
    )
    d = decide(sb, firstFrameImage="f.png", sceneType="action", referenceAudioUrls=["v.mp3"],
               referenceImages=["a.png"], provider="minimax")
    check("路由: ④b 有参考图时 Ref2VA 用 multiple", d["referenceMode"] == "multiple", d)
    # ⚠️ 只挂**参考视频**也是一种 Ref2VA 诉求 ✓✗（H3 的 ``ref_videos`` 与参考图/音频平级 ✓）；
    #    漏算 ⇒ 路由写「首帧图生」、适配器却挑 Ref2VA ✓ ⇒ 两处说法互相打脸 ✓✗。
    d = decide(sb, firstFrameImage="f.png", sceneType="action", referenceVideoUrls=["v.mp4"],
               provider="minimax")
    check("路由: ④b 只挂参考视频（无声线样本）⇒ 也是 Ref2VA、理由点名「参考视频」✓",
          d["route"] == "reference_to_video" and "1 段参考视频" in d["reason"],
          d)
    check("路由: ④b 视频+音频同时有 ⇒ 理由两样都点出来 ✓",
          (lambda dd: "1 条声线样本" in dd["reason"] and "2 段参考视频" in dd["reason"])(
              decide(sb, firstFrameImage="f.png", sceneType="action",
                     referenceAudioUrls=["v.mp3"], referenceVideoUrls=["a.mp4", "b.mp4"],
                     provider="minimax")))
    d = decide(sb, firstFrameImage="f.png", sceneType="action", referenceAudioUrls=["v.mp3"],
               provider="volcengine")
    check(
        "路由: ④b 只在 provider==minimax 时生效（别家带音频不算）",
        d["route"] == "first_frame_to_video",
        d,
    )
    d = decide(sb, firstFrameImage="f.png", keyframeImage="k.png")
    check(
        "路由: ⑤ 关键帧 -> I2V-K",
        d["route"] == "keyframe_to_video" and d["referenceMode"] == "single",
        d,
    )
    d = decide(sb, firstFrameImage="f.png", keyframeImage="k.png", referenceImages=["a.png"])
    check("路由: ⑤ 支持多参考且有参考图时 keyframe 也用 multiple", d["referenceMode"] == "multiple", d)
    d = decide(sb, firstFrameImage="f.png")
    check(
        "路由: ⑥ 默认 I2V（无顺接提示时 reason 不带分号段）",
        d["route"] == "first_frame_to_video" and d["reason"].endswith("（I2V）"),
        d["reason"],
    )
    d = decide(sb, firstFrameImage="f.png", prevTail="p.png")
    check(
        "路由: ⑥ 默认 I2V + 顺接提示",
        d["reason"].endswith("（I2V）；同场景顺接，以上一镜尾帧衔接起帧"),
        d["reason"],
    )

    # 决策会回写分镜
    with engine.begin() as conn:
        row = conn.execute(select(storyboards).where(storyboards.c.id == sb)).first()
    check(
        "路由: 决策**回写** storyboards.route / route_reason",
        row.route == "first_frame_to_video" and row.route_reason == d["reason"],
        (row.route, row.route_reason),
    )

    # recomputeEpisodeRoutes：保守输入（provider=default / 无参考）
    sb2 = new_storyboard()
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb2).values(
            first_frame_image="f2.png"))
    with engine.begin() as conn:
        sr.recompute_episode_routes(conn, episode_id)
    with engine.begin() as conn:
        r2 = conn.execute(select(storyboards).where(storyboards.c.id == sb2)).first()
    check(
        "路由: recomputeEpisodeRoutes 用保守输入（无首帧的落 T2V、有首帧的落 I2V）",
        r2.route == "first_frame_to_video", r2.route,
    )
    check("路由: SHOT_ROUTES 七种类型齐全（含保留的 video_editor）",
          len(sr.SHOT_ROUTES) == 7 and "video_editor" in sr.SHOT_ROUTES)
    check(
        "路由: 本文件判据比 shot_router 的窄（videos.ts 那份没有「多人/中文」）",
        bool(sr.VIDEO_ROUTE_DIALOGUE_PATTERN.search("dialogue"))
        and not bool(sr.VIDEO_ROUTE_DIALOGUE_PATTERN.search("对话"))
        and bool(sr.DIALOGUE_PATTERN.search("对话")),
    )

    # ================= POST /videos =================
    check("POST: 缺 prompt -> 400 prompt is required",
          client.post("/api/v1/videos", json={}).status_code == 400)

    # 造上下文：角色（有立绘）+ 场景（有图）+ 分镜关联
    char_id = _insert_character(drama_id, "林昭", image_url="static/images/c.png")
    scene_id = client.post("/api/v1/scenes", json={
        "drama_id": drama_id, "location": "客栈", "time": "夜晚"}).json()["data"]["id"]
    with engine.begin() as conn:
        conn.execute(scenes.update().where(scenes.c.id == scene_id).values(image_url="static/images/s.png"))
        conn.execute(storyboard_characters.insert().values(storyboard_id=sb, character_id=char_id))
        conn.execute(storyboards.update().where(storyboards.c.id == sb).values(
            scene_id=scene_id, description="视觉描述", action="拔剑", movement="推镜",
            scene_type="dialogue", first_frame_image="static/images/ff.png"))

    resp = client.post("/api/v1/videos", json={
        "storyboard_id": sb, "drama_id": drama_id, "prompt": "<location>客栈</location>两人对视",
        "duration": 6,
    })
    check("POST: 201 + camelCase 行", resp.status_code == 201 and "storyboardId" in resp.json()["data"],
          resp.status_code)
    created_row = resp.json()["data"]
    check(
        "POST: prompt 已富化（角色一致性 + Setting + 动作 + 运镜 + 画风层）",
        "Characters (maintain strict visual consistency): 林昭" in created_row["prompt"]
        and "Setting: 客栈" in created_row["prompt"]
        and "Action: 拔剑" in created_row["prompt"]
        and "Camera movement: dolly in" in created_row["prompt"]
        and "[background_audio]" not in created_row["prompt"],
        created_row["prompt"][:160],
    )
    check(
        "POST: 标签已剥离（富化前先洗 prompt）",
        "<location>" not in created_row["prompt"],
    )
    check(
        "POST: 自动收集参考图（角色立绘 + 场景图）",
        json.loads(created_row["referenceImageUrls"] or "[]")
        == ["static/images/c.png", "static/images/s.png"],
        created_row["referenceImageUrls"],
    )
    check(
        "POST: 逐镜路由决策落库（对话 + 多参考 + 有参考图 -> R2V/multiple）",
        created_row["route"] == "reference_to_video"
        and created_row["referenceMode"] == "multiple"
        and "R2V" in (created_row["routeReason"] or ""),
        (created_row["route"], created_row["referenceMode"]),
    )
    # ⚠️⚠️ 参考**视频**与参考图档位必须**过路由层** ✓✗（2026-10-27 补）：修前路由只读
    #    ``reference_image_urls`` / ``reference_audio_urls`` ✗ ⇒ 前端带上来的这两样
    #    在**第一层**就被丢掉 ✓（下游连一句告警都没有 ✓，看回执只能猜 ✓）。
    #    ⚠️ 用**新分镜**跑：``sb`` 的 take 预算本套后面几条要用满 ✓✗（首次实现挂在 ``sb`` 上
    #    就多占一次 ⇒ 后面那条 400「take 预算耗尽」✓✗，与字段转发毫无关系的一条红 ✓）。
    sb_v = new_storyboard()
    resp_v = client.post("/api/v1/videos", json={
        "storyboard_id": sb_v, "drama_id": drama_id, "prompt": "换了参考视频的一镜",
        "duration": 6, "reference_video_urls": ["static/videos/ref1.mp4"],
        "reference_image_size": "max",
    })
    row_v = resp_v.json()["data"]
    check("POST: ``reference_video_urls`` / ``reference_image_size`` 原样过路由层 ✓",
          resp_v.status_code == 201
          and json.loads(row_v["referenceVideoUrls"] or "null") == ["static/videos/ref1.mp4"]
          and row_v["referenceImageSize"] == "max",
          (resp_v.status_code, row_v.get("referenceVideoUrls"), row_v.get("referenceImageSize")))
    check(
        "POST: 逐镜路由决策落库（对话 + 多参考 + 有参考图 -> R2V/multiple）",
        created_row["route"] == "reference_to_video"
        and created_row["referenceMode"] == "multiple"
        and "R2V" in (created_row["routeReason"] or ""),
        (created_row["route"], created_row["referenceMode"]),
    )
    # ⚠️ 该请求落到了 **multiple**（R2V）—— 按设计「multiple 走 reference_image_urls、
    #    **不用帧字段**」，所以分镜里存的 first_frame_image 在这里**不该**被消费。
    check(
        "POST: multiple 分支不消费帧字段（即便分镜存了首帧）",
        created_row["imageUrl"] is None and created_row["firstFrameUrl"] is None,
        (created_row["imageUrl"], created_row["firstFrameUrl"]),
    )

    # 换一个**非对话**场景的分镜（不会触发 R2V）⇒ 落 single，此时才验「帧来源统一」
    sb_single = new_storyboard()
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_single).values(
            first_frame_image="static/images/stored.png", scene_type="action"))
    resp = client.post("/api/v1/videos", json={"storyboard_id": sb_single, "prompt": "打斗"})
    single_row = resp.json()["data"]
    check(
        "POST: 帧来源统一 —— 分镜 stored 首帧被当作候选帧（single 时落到 imageUrl）",
        single_row["referenceMode"] == "single"
        and single_row["imageUrl"] == "static/images/stored.png",
        (single_row["referenceMode"], single_row["imageUrl"]),
    )

    # referenceMode 降级：请求 first_last 但只有首帧 -> single
    resp = client.post("/api/v1/videos", json={
        "storyboard_id": sb, "prompt": "x", "reference_mode": "first_last",
        "first_frame_url": "static/images/only-first.png",
    })
    degraded = resp.json()["data"]
    check(
        "POST: 目标 first_last 但缺尾帧 -> 降级 single（避免缺首帧的坏请求）",
        degraded["referenceMode"] == "single"
        and degraded["imageUrl"] == "static/images/only-first.png"
        and degraded["lastFrameUrl"] is None,
        (degraded["referenceMode"], degraded["imageUrl"], degraded["lastFrameUrl"]),
    )
    # 首尾帧齐备 -> 保留 first_last，且**不写 imageUrl**
    resp = client.post("/api/v1/videos", json={
        "prompt": "x", "reference_mode": "first_last",
        "first_frame_url": "static/images/f.png", "last_frame_url": "static/images/l.png",
    })
    fl = resp.json()["data"]
    check(
        "POST: 首尾帧齐备 -> first_last，且 imageUrl 为 null（adapter 只认首尾帧）",
        fl["referenceMode"] == "first_last" and fl["imageUrl"] is None
        and fl["firstFrameUrl"] == "static/images/f.png"
        and fl["lastFrameUrl"] == "static/images/l.png",
        (fl["referenceMode"], fl["imageUrl"]),
    )

    # _skip_enrich：不注入上下文（但标签仍会被洗掉）
    resp = client.post("/api/v1/videos", json={
        "storyboard_id": sb, "prompt": "<role>林昭</role>原样", "_skip_enrich": True,
    })
    skipped = resp.json()["data"]
    check(
        "POST: _skip_enrich 跳过上下文注入（prompt 不被富化）",
        skipped["prompt"] == "林昭原样" and "Characters (maintain" not in skipped["prompt"],
        skipped["prompt"],
    )

    # 无 storyboard：不做路由决策，route 为 null
    resp = client.post("/api/v1/videos", json={"prompt": "裸生成"})
    bare = resp.json()["data"]
    check(
        "POST: 无 storyboard_id -> 不做路由决策（route/routeReason 为 null）",
        bare["route"] is None and bare["routeReason"] is None
        and bare["referenceMode"] == "none",
        (bare["route"], bare["referenceMode"]),
    )

    # ================= GET / PUT / DELETE / regenerate =================
    vid = fl["id"]
    got = client.get(f"/api/v1/videos/{vid}")
    check("GET /:id: camelCase 行", got.status_code == 200 and got.json()["data"]["id"] == vid)
    check(
        "GET /:id: 查不到 -> success(null)（**不是 404**，与 TS 一致）",
        client.get("/api/v1/videos/999999").json()["data"] is None,
    )
    check("GET /:id: 非法 id -> 404", client.get("/api/v1/videos/abc").status_code == 404)

    listed = client.get(f"/api/v1/videos?storyboard_id={sb}").json()["data"]
    check(
        "GET /: 按 storyboard_id 过滤",
        len(listed) >= 2 and all(row["storyboardId"] == sb for row in listed),
        len(listed),
    )
    listed_by_drama = client.get(f"/api/v1/videos?drama_id={drama_id}").json()["data"]
    check("GET /: 按 drama_id 过滤", all(row["dramaId"] == drama_id for row in listed_by_drama))
    check("GET /: 非法过滤值当没传（不过滤）",
          len(client.get("/api/v1/videos?storyboard_id=abc").json()["data"]) == len(client.get("/api/v1/videos").json()["data"]))

    put = client.put(f"/api/v1/videos/{vid}", json={"prompt": "改过", "duration": 9})
    check("PUT: 白名单字段生效", put.status_code == 200 and put.json()["data"]["prompt"] == "改过"
          and put.json()["data"]["duration"] == 9, put.json()["data"]["prompt"])
    check("PUT: 空 body -> 400 No fields to update",
          client.put(f"/api/v1/videos/{vid}", json={}).status_code == 400)
    check("PUT: 不存在的记录 -> 400 视频记录不存在",
          client.put("/api/v1/videos/999999", json={"prompt": "x"}).status_code == 400)

    # ⚠️⚠️ 参考素材四件必须能过 ``PUT`` 落库 ✓✗（2026-10-27 补）：白名单原来**一项都没有** ✗
    #    ⇒ 编辑器走「更新已有记录」那条路时，改参考图 / 参考视频 / 参考图档位 = **静默丢弃** ✓✗
    #    （回读一字未变 ✓、也不报错 ✓，界面还写着「已保存」✓✗ —— 前端有值、库里有旧值、没人报错）。
    #    顺带核「数组型别把 list 直接绑进 sqlite」✓（那会 InterfaceError ⇒ 400 ✗）。
    #    ⚠️ 拿 ``degraded`` 那条来改 ✓ —— ``vid`` 后面还要用它的原参数核 regenerate ✗。
    dv = degraded["id"]
    put_refs = client.put(f"/api/v1/videos/{dv}", json={
        "reference_image_urls": ["static/images/r1.png", "static/images/r2.png"],
        "reference_video_urls": ["static/videos/r1.mp4"],
        "reference_image_size": "max",
    })
    row_refs = put_refs.json()["data"]
    check(
        "PUT: 参考图 / 参考视频 / 参考图档位 能落库（数组存紧凑 JSON）",
        put_refs.status_code == 200
        and json.loads(row_refs["referenceImageUrls"]) == ["static/images/r1.png", "static/images/r2.png"]
        and json.loads(row_refs["referenceVideoUrls"]) == ["static/videos/r1.mp4"]
        and row_refs["referenceImageSize"] == "max",
        (put_refs.status_code, row_refs.get("referenceImageUrls"),
         row_refs.get("referenceVideoUrls"), row_refs.get("referenceImageSize")),
    )
    check(
        "PUT: 参考素材显式置空 -> NULL（不是 \"[]\"）",
        client.put(f"/api/v1/videos/{dv}", json={"reference_video_urls": []})
        .json()["data"]["referenceVideoUrls"] is None,
    )
    check(
        "PUT: 档位发空串 = 清空 -> NULL（库里不留 '' 这种第二形态的空）",
        client.put(f"/api/v1/videos/{dv}", json={"reference_image_size": ""})
        .json()["data"]["referenceImageSize"] is None,
    )
    check(
        "PUT: 已序列化的字符串原样放行（不二次转义）",
        client.put(f"/api/v1/videos/{dv}", json={"reference_audio_urls": "[\"static/audio/a.mp3\"]"})
        .json()["data"]["referenceAudioUrls"] == "[\"static/audio/a.mp3\"]",
    )
    check(
        "PUT: 参考音频数组 → 紧凑 JSON（与图/视频同口径）",
        client.put(f"/api/v1/videos/{dv}", json={
            "reference_audio_urls": ["static/audio/b.mp3", "static/audio/c.mp3"],
        }).json()["data"]["referenceAudioUrls"]
        == '["static/audio/b.mp3","static/audio/c.mp3"]',
    )
    # ⚠️ 显式 ``[]`` = **清空** ✓✗：后端 regenerate 原来用 ``or`` 取默认值 ⇒ 空数组被当
    #    「没给」✓ ⇒ 用户把参考视频删光、界面也显示空了，一重跑**旧素材又回来了** ✓✗。
    #    这里核 ``_pick`` 的语义（真跑一次 regenerate 会挤 take 预算、且这里只想锁语义 ✓）。
    from app.routers.videos import _pick

    check(
        "regenerate: 显式 [] 不被 or 吞掉（清空 vs 未给 分开）",
        _pick({"reference_video_urls": []}, "reference_video_urls", ["static/videos/old.mp4"]) == []
        and _pick({}, "reference_video_urls", ["static/videos/old.mp4"]) == ["static/videos/old.mp4"],
    )

    # ⚠️⚠️ 参考**音频**的「显式 [] = 清空」**两头**都得认 ✓✗（2026-10-27 补）：
    #    POST 有**自动注入**（对话镜头自动带出场角色声线样本）、regenerate 原来用 ``or`` 回落 ——
    #    两处都按 falsy 判 ⇒ 用户把声线样本删光、界面也显示空了，一提交**旧样本又注回来** ✓✗。
    voice_char = _insert_character(
        drama_id, "声线角色", voice_sample_url="https://cdn.example.com/v.mp3")
    sb_audio = new_storyboard()
    with engine.begin() as conn:
        conn.execute(storyboard_characters.insert().values(
            storyboard_id=sb_audio, character_id=voice_char))
        conn.execute(storyboards.update().where(storyboards.c.id == sb_audio).values(
            scene_type="dialogue", first_frame_image="static/images/af.png",
            take_budget=10))
    auto = client.post("/api/v1/videos", json={
        "storyboard_id": sb_audio, "prompt": "对话自动声线", "duration": 5,
    }).json()["data"]
    check(
        "POST: 对话镜头**未给**参考音频 ⇒ 自动注入出场角色声线样本（原有行为保住）",
        json.loads(auto["referenceAudioUrls"] or "null") == ["https://cdn.example.com/v.mp3"],
        auto["referenceAudioUrls"],
    )
    cleared = client.post("/api/v1/videos", json={
        "storyboard_id": sb_audio, "prompt": "对话但不要声线", "duration": 5,
        "reference_audio_urls": [],
    }).json()["data"]
    check(
        "POST: 显式 ``[]`` 参考音频 ⇒ **不**自动注入（「清空」≠「没给」）",
        cleared["referenceAudioUrls"] is None, repr(cleared["referenceAudioUrls"]),
    )
    regen_audio = client.post(f"/api/v1/videos/{cleared['id']}/regenerate", json={
        "reference_audio_urls": [],
    })
    check(
        "regenerate: 显式 ``[]`` 参考音频 ⇒ 新记录也不带（原来 ``or`` 会去分镜注回声线样本）",
        regen_audio.status_code == 201
        and regen_audio.json()["data"]["referenceAudioUrls"] is None,
        (regen_audio.status_code, regen_audio.json().get("data", {}).get("referenceAudioUrls")),
    )

    # ⚠️⚠️ 角色关联（`character_ids`）此前**三处都没人管** ✗✗（2026-10-27 补）：
    #    ① 前端编辑器勾了角色、`character_ids` 也发上来了 ✓；② `videos.py` 的 PUT 白名单有它 ✓；
    #    ③ 但 `POST /videos` **一个字都不读**、`generate_video` 的插入列表里**没有这一列**、
    #    `regenerate` 也不复制 ⇒ 落库恒 NULL ⇒ 重开编辑器勾选全空（谁都不报错 ✓✗）。
    #    ⚠️ 前端回填还叠了一层：`'[1,2]'.split(',')` ⇒ `[NaN,NaN]`（本次一并修）。
    sb_cast = new_storyboard()
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_cast).values(take_budget=10))
    cast_row = client.post("/api/v1/videos", json={
        "prompt": "带角色关联的一镜", "storyboard_id": sb_cast,
        "character_ids": [char_id, voice_char],
    }).json()["data"]
    check(
        "POST: ``character_ids`` 落库（紧凑 JSON；原来 POST 不读这字段 ⇒ 静默丢弃）",
        cast_row["characterIds"] == f"[{char_id},{voice_char}]",
        repr(cast_row["characterIds"]),
    )
    cast_regen = client.post(f"/api/v1/videos/{cast_row['id']}/regenerate", json={}).json()["data"]
    check(
        "regenerate: ``character_ids`` 未给 ⇒ 沿用原记录（新记录不丢出场角色）",
        cast_regen["characterIds"] == cast_row["characterIds"],
        repr(cast_regen["characterIds"]),
    )
    # ⚠️⚠️ 二次转义 ✓✗：regenerate 复制过来的参考素材是 **DB 里的 JSON 字符串**，
    #    过 ``json.dumps`` 一次就变成 `'"[\"a\"]"'` ⇒ 落库成了「一个字符串」而不是数组 ✓✗
    #    （前端 parseUrlList 解出字符串、指纹 `_refs_map` 解出空 ⇒ 重跑出来的记录参考素材静默消失）。
    ref_row = client.post("/api/v1/videos", json={
        "prompt": "带参考图", "storyboard_id": sb_cast,
        "reference_image_urls": ["static/images/x1.png", "static/images/x2.png"],
    }).json()["data"]
    ref_regen = client.post(f"/api/v1/videos/{ref_row['id']}/regenerate", json={}).json()["data"]
    check(
        "regenerate: 复制的参考图 JSON 串**不被二次转义**（json.loads 仍是两条）",
        json.loads(ref_regen["referenceImageUrls"])
        == ["static/images/x1.png", "static/images/x2.png"],
        repr(ref_regen["referenceImageUrls"]),
    )

    regen = client.post(f"/api/v1/videos/{vid}/regenerate", json={})
    regen_row = regen.json()["data"]
    check(
        "regenerate: 未给的参数沿用原记录，且**新建**一条（id 不同）",
        regen.status_code == 201 and regen_row["id"] != vid
        and regen_row["prompt"] == "改过" and regen_row["duration"] == 9
        and regen_row["referenceMode"] == "first_last",
        (regen_row["id"], regen_row["prompt"], regen_row["duration"], regen_row["referenceMode"]),
    )
    check(
        "regenerate: 原记录负面词优先（可复现）",
        regen_row["negativePrompt"] == fl["negativePrompt"],
        (regen_row["negativePrompt"], fl["negativePrompt"]),
    )
    check("regenerate: 记录不存在 -> 400 视频记录不存在",
          client.post("/api/v1/videos/999999/regenerate", json={}).status_code == 400)

    check("DELETE: 删掉后查不到", client.delete(f"/api/v1/videos/{vid}").status_code == 200
          and client.get(f"/api/v1/videos/{vid}").json()["data"] is None)

    client.delete(f"/api/v1/dramas/{drama_id}")

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


def _insert_character(drama_id: int, name: str, **extra) -> int:
    values: dict[str, object] = {"drama_id": drama_id, "name": name}
    for column in ("created_at", "updated_at"):
        if column in characters.c:
            values[column] = now()
    values.update(extra)
    with engine.begin() as conn:
        return int(conn.execute(characters.insert().values(**values)).lastrowid)


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
