"""S6 自检：全自动管线编排器 + 4 端点（``services/auto-pipeline.ts`` 814 行 + ``routes/auto-pipeline.ts``）。

这是**最长的编排链路**（8 阶段：4 个 Agent + 4 个媒体），四类错都会**静默伤钱/伤数据**：

1. **阶段顺序**：``scripting → extracting → voicing → storyboarding``（**配音在分镜之前**），
   而 ``STAGE_SEQUENCE`` 就是幂等判据 —— 顺序错了等于跳过/重跑错阶段；
2. **依赖补全**：只勾 ``withMerge`` 也要把图片/视频/合成阶段跑起来（否则 merge 永远缺料）；
3. **幂等**：媒体阶段提交前必须查已有产物/进行中任务，否则崩溃重启会**重复提交重复扣费**；
4. **只写真实尾帧**：``tail_frame_image`` 作为下一镜起帧，**绝不覆盖** ``last_frame_image``。

Agent 阶段与媒体提交全部打桩（不打网络、不真等 5 秒轮询）。

运行::

    ./.venv/Scripts/python.exe tests/auto_pipeline_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="autopipe_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import dramas, episodes, storyboards, video_generations  # noqa: E402
from app.core.response import now  # noqa: E402
from app.routers import auto_pipeline as ap_route  # noqa: E402
from app.agent import auto_pipeline as ap  # noqa: E402
from app.services import sse_hub  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


async def _no_sleep(_seconds: float) -> None:
    return None


def _make_episode(drama_id: int, status: str, number: int = 1) -> int:
    with engine.begin() as conn:
        return int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=number, title="测试集",
            content="梗概", status=status, created_at=now(), updated_at=now(),
        )).lastrowid)


def _make_drama(title: str = "测试剧", metadata: str | None = None) -> int:
    with engine.begin() as conn:
        return int(conn.execute(dramas.insert().values(
            title=title, status="auto_generating", metadata=metadata,
            created_at=now(), updated_at=now(),
        )).lastrowid)


def main() -> int:  # noqa: C901
    # ================= 纯逻辑：选项归一 =================
    norm = ap.normalize_options({"premise": "  一句话梗概  "})
    check("归一: 缺省 episodeCount=1、genre=短剧、style=realistic、四个开关为假",
          norm["episodeCount"] == 1 and norm["genre"] == "短剧" and norm["style"] == "realistic"
          and not any(norm[key] for key in ("withImages", "withVideos", "withCompose", "withMerge")),
          {k: norm[k] for k in ("episodeCount", "genre", "style")})
    check("归一: title 缺省取 premise **前 24 字符**（不是 trim 后截断）",
          ap.normalize_options({"premise": "字" * 30})["title"] == "字" * 24,
          ap.normalize_options({"premise": "字" * 30})["title"])
    check("归一: episodeCount 非正/非数字 -> 1；小数取 floor",
          [ap.normalize_options({"premise": "x", "episodeCount": n})["episodeCount"]
           for n in (0, -3, "abc", 2.9, 3)] == [1, 1, 1, 2, 3])
    check("归一: 显式 title/genre/style 会 trim 后采用（空白则回落）",
          ap.normalize_options({"premise": "梗", "title": "  我的剧  ", "genre": " 悬疑 ",
                                "style": "  "})["title"] == "我的剧"
          and ap.normalize_options({"premise": "梗", "genre": " 悬疑 "})["genre"] == "悬疑"
          and ap.normalize_options({"premise": "梗"})["style"] == "realistic")
    check("归一: 键名保持 **camelCase**（它要写进 dramas.metadata 再读回）",
          "episodeCount" in norm and "withImages" in norm
          and "episode_count" not in norm)

    # ================= 纯逻辑：依赖补全 =================
    check("补全: `{}` -> 四个阶段全不需要", ap.media_needs({}) == {
        "needImage": False, "needVideo": False, "needCompose": False, "needMerge": False})
    check("补全: `withImages` -> 只要图片", ap.media_needs({"withImages": True}) == {
        "needImage": True, "needVideo": False, "needCompose": False, "needMerge": False})
    check("补全: `withVideos` -> **隐含图片**", ap.media_needs({"withVideos": True}) == {
        "needImage": True, "needVideo": True, "needCompose": False, "needMerge": False})
    check("补全: `withCompose` -> 隐含图片+视频", ap.media_needs({"withCompose": True}) == {
        "needImage": True, "needVideo": True, "needCompose": True, "needMerge": False})
    check("补全: `withMerge` -> **全部**（否则 merge 永远缺料）",
          all(ap.media_needs({"withMerge": True}).values()))

    # ================= 纯逻辑：幂等判据 =================
    check("幂等: 无状态/failed -> 任何阶段都算**未完成**（会重跑）",
          ap.is_stage_completed(None, "auto:scripting") is False
          and ap.is_stage_completed("auto:failed", "auto:scripting") is False)
    check("幂等: 等于 next -> 已完成；早于 next -> 未完成；晚于 next -> 已完成",
          ap.is_stage_completed("auto:imaging", "auto:imaging") is True
          and ap.is_stage_completed("auto:voicing", "auto:imaging") is False
          and ap.is_stage_completed("auto:done", "auto:merging") is True)
    check("幂等: 未知的 next -> True、未知的 current -> False（与原 TS 一致）",
          ap.is_stage_completed("auto:done", "auto:没有这个") is True
          and ap.is_stage_completed("auto:没有这个", "auto:scripting") is False)

    # ================= 纯逻辑：阶段表 =================
    keys = [stage["key"] for stage in ap.build_stages(1, 1, {})]
    check("阶段表: 无媒体时只有 4 个 Agent 阶段，且顺序是 script→extract→**voice→storyboard**",
          keys == ["script", "extract", "voice", "storyboard"], keys)
    media_keys = [stage["key"] for stage in ap.build_stages(1, 1, {"withMerge": True})]
    check("阶段表: `withMerge` 补齐 image→video→compose→merge",
          media_keys == ["script", "extract", "voice", "storyboard",
                         "image", "video", "compose", "merge"], media_keys)
    no_media = {stage["key"]: stage for stage in ap.build_stages(1, 1, {})}
    check("阶段表: 无媒体时 storyboard 的 next 直接是 `auto:done`",
          no_media["storyboard"]["next"] == "auto:done"
          and no_media["storyboard"]["status"] == "auto:storyboarding",
          no_media["storyboard"]["next"])
    full = {stage["key"]: stage for stage in ap.build_stages(1, 1, {"withMerge": True})}
    check("阶段表: 有媒体时 next 链是 imaging→videoing→composing→merging→done",
          (full["storyboard"]["next"], full["image"]["next"], full["video"]["next"],
           full["compose"]["next"], full["merge"]["next"])
          == ("auto:imaging", "auto:videoing", "auto:composing", "auto:merging", "auto:done"),
          [full[k]["next"] for k in ("storyboard", "image", "video", "compose", "merge")])

    # ================= 状态机（阶段打桩）=================
    ap._sleep = _no_sleep  # type: ignore[assignment]  # noqa: SLF001
    called: list[str] = []

    def _install_stages(*, fail_at: str | None = None) -> None:
        def _make_runner(key: str):
            async def _runner(*_args, **_kwargs) -> None:
                called.append(key)
                if fail_at == key:
                    raise RuntimeError(f"{key} 阶段炸了")

            return _runner

        for key, name in (("script", "run_script_stage"), ("extract", "run_extract_stage"),
                          ("voice", "run_voice_stage"), ("storyboard", "run_storyboard_stage"),
                          ("image", "run_image_stage"), ("video", "run_video_stage")):
            setattr(ap, name, _make_runner(key))

        async def _compose(episode_id: int) -> None:
            called.append("compose")

        async def _merge(episode_id: int, drama_id: int) -> None:
            called.append("merge")

        ap.run_compose_stage = _compose  # type: ignore[assignment]
        ap.run_merge_stage = _merge  # type: ignore[assignment]

    drama_id = _make_drama()
    episode_id = _make_episode(drama_id, ap.AUTO_STATUS["queued"])
    events: list[dict] = []
    unsubscribe = sse_hub.subscribe_pipeline(drama_id, lambda evt: events.append(evt))

    called.clear()
    _install_stages()
    asyncio.run(ap.execute_episode_pipeline(episode_id, drama_id,
                                            ap.normalize_options({"premise": "梗"})))
    with engine.begin() as conn:
        final_status = conn.execute(select(episodes.c.status)
                                    .where(episodes.c.id == episode_id)).first()[0]
    check("状态机: 无媒体时跑完 4 个 Agent 阶段并落到 `auto:done`",
          called == ["script", "extract", "voice", "storyboard"]
          and final_status == ap.AUTO_STATUS["done"], (called, final_status))
    check("状态机: 每次推进都发 SSE `status` 事件（含 episodeId/status）",
          len(events) >= 5 and events[0]["type"] == "status"
          and events[0]["episodeId"] == episode_id
          and events[0]["status"] == ap.AUTO_STATUS["scripting"]
          and events[-1]["status"] == ap.AUTO_STATUS["done"],
          events[:2])
    unsubscribe()

    # 幂等续跑：从中间态起，前面的 Agent 阶段应被跳过
    episode2 = _make_episode(drama_id, ap.AUTO_STATUS["imaging"], number=2)
    called.clear()
    _install_stages()
    asyncio.run(ap.execute_episode_pipeline(episode2, drama_id,
                                           ap.normalize_options({"premise": "梗",
                                                                 "withMerge": True})))
    check("幂等: 从 `auto:imaging` 续跑 -> **跳过全部 Agent 阶段**，只跑媒体阶段",
          called == ["image", "video", "compose", "merge"], called)

    # 已完成 -> 立即返回
    episode3 = _make_episode(drama_id, ap.AUTO_STATUS["done"], number=3)
    called.clear()
    _install_stages()
    asyncio.run(ap.execute_episode_pipeline(episode3, drama_id,
                                            ap.normalize_options({"premise": "梗"})))
    check("幂等: `auto:done` 的集**直接返回**（不重跑任何阶段）", called == [], called)

    # 阶段失败 -> 标记 failed 并抛出
    episode4 = _make_episode(drama_id, ap.AUTO_STATUS["queued"], number=4)
    called.clear()
    _install_stages(fail_at="storyboard")
    failed_error = ""
    try:
        asyncio.run(ap.execute_episode_pipeline(episode4, drama_id,
                                                ap.normalize_options({"premise": "梗"})))
    except Exception as err:  # noqa: BLE001
        failed_error = str(err)
    with engine.begin() as conn:
        failed_status = conn.execute(select(episodes.c.status)
                                     .where(episodes.c.id == episode4)).first()[0]
    check("状态机: 阶段抛错 -> 标记 `auto:failed` 且**异常继续上抛**（交给上层记日志）",
          called == ["script", "extract", "voice", "storyboard"]
          and failed_error == "storyboard 阶段炸了"
          and failed_status == ap.AUTO_STATUS["failed"],
          (failed_error, failed_status))

    # 软删保护
    deleted_drama = _make_drama("已删剧")
    with engine.begin() as conn:
        conn.execute(dramas.update().where(dramas.c.id == deleted_drama)
                     .values(deleted_at=now()))
    deleted_episode = _make_episode(deleted_drama, ap.AUTO_STATUS["queued"])
    called.clear()
    _install_stages()
    asyncio.run(ap.execute_episode_pipeline(deleted_episode, deleted_drama,
                                            ap.normalize_options({"premise": "梗"})))
    check("删除竞态: drama 已软删 -> **一个阶段都不跑**（防管线复活已删对象）",
          called == [], called)

    # 整剧串行 + 失败隔离
    drama2 = _make_drama("整剧")
    ep_a = _make_episode(drama2, ap.AUTO_STATUS["queued"], number=1)
    ep_b = _make_episode(drama2, ap.AUTO_STATUS["queued"], number=2)
    order: list[int] = []

    async def _fake_episode(episode_id: int, drama_id: int, opts: dict) -> None:  # noqa: ANN001
        order.append(episode_id)
        if episode_id == ep_a:
            raise RuntimeError("第一集挂了")

    original_episode = ap.execute_episode_pipeline
    ap.execute_episode_pipeline = _fake_episode  # type: ignore[assignment]
    asyncio.run(ap.execute_pipeline(drama2, ap.normalize_options({"premise": "梗"})))
    ap.execute_episode_pipeline = original_episode  # type: ignore[assignment]
    check("整剧: **按集号串行**、单集失败不阻断后续集",
          order == [ep_a, ep_b], order)

    # ================= 状态查询 =================
    check("状态: drama 不存在 -> None（路由回 404）", ap.get_auto_pipeline_status(999_999) is None)
    status = ap.get_auto_pipeline_status(drama_id)
    check("状态: 顶层键 camelCase 且带 running/totalEpisodes/doneCount/failedCount",
          set(status) == {"dramaId", "title", "status", "totalEpisodes", "doneCount",
                          "failedCount", "running", "episodes"},
          sorted(status))
    check("状态: running = (done + failed) < 总集数",
          status["running"] is (status["doneCount"] + status["failedCount"]
                                < status["totalEpisodes"]),
          (status["doneCount"], status["failedCount"], status["totalEpisodes"]))
    first = status["episodes"][0]
    check("状态: 每集明细含 11 个 camelCase 字段（含 blocked/composed 计数）",
          set(first) == {"id", "episodeNumber", "status", "hasScript", "hasVideo",
                         "storyboardCount", "characterCount", "sceneCount",
                         "imageReadyCount", "videoReadyCount", "videoBlockedCount",
                         "composedCount"},
          sorted(first))

    # ================= run / resume =================
    spawned: list[int] = []

    async def _fake_pipeline(drama_id: int, opts: dict) -> None:  # noqa: ANN001
        spawned.append(drama_id)

    ap.execute_pipeline = _fake_pipeline  # type: ignore[assignment]

    empty_error = ""
    try:
        ap.run_auto_pipeline({"premise": "   "})
    except Exception as err:  # noqa: BLE001
        empty_error = str(err)
    check("run: premise 为空/空白 -> `premise 不能为空`", empty_error == "premise 不能为空",
          empty_error)

    # ⚠️ `_spawn` 用 `asyncio.ensure_future` ⇒ **必须在事件循环里调用**（生产是 async 路由里调用）。
    #    所以这一段的断言整体收进一次 `asyncio.run`，否则 fire-and-forget 任务会挂到已关闭的循环上。
    async def _run_and_resume() -> dict:
        created = ap.run_auto_pipeline({"premise": "三集的故事大纲", "episodeCount": 3,
                                        "title": "我的三部曲", "imageConfigId": 7})
        await asyncio.sleep(0.05)  # 让 fire-and-forget 任务跑掉
        return created

    result = asyncio.run(_run_and_resume())
    with engine.begin() as conn:
        drama_row = conn.execute(select(dramas).where(
            dramas.c.id == result["dramaId"])).first()
        episode_rows = conn.execute(
            select(episodes).where(episodes.c.drama_id == result["dramaId"])
            .order_by(episodes.c.episode_number)).all()
    check("run: 建 Drama（title/desc/genre/style/totalEpisodes/status=auto_generating）",
          drama_row.title == "我的三部曲" and drama_row.description == "三集的故事大纲"
          and drama_row.genre == "短剧" and drama_row.total_episodes == 3
          and drama_row.status == "auto_generating",
          (drama_row.title, drama_row.total_episodes, drama_row.status))
    check("run: 建 N 集（`第n集` 标题、`【第n集】` 内容前缀、状态 auto:queued、透传 configId）",
          len(episode_rows) == 3
          and [row.title for row in episode_rows]
          == ["我的三部曲 第1集", "我的三部曲 第2集", "我的三部曲 第3集"]
          and episode_rows[0].content == "【第1集】三集的故事大纲"
          and all(row.status == ap.AUTO_STATUS["queued"] for row in episode_rows)
          and all(row.image_config_id == 7 for row in episode_rows),
          [(row.title, row.content) for row in episode_rows])
    check("run: **立即返回** {dramaId, episodeIds} 并已触发后台管线",
          result["episodeIds"] == [row.id for row in episode_rows]
          and spawned == [result["dramaId"]], (result, spawned))
    check("run: metadata 里存的是**紧凑 JSON**（可原样读回）",
          " " not in (drama_row.metadata or "")[:40]
          and ap.load_pipeline_options(result["dramaId"])["episodeCount"] == 3,
          (drama_row.metadata or "")[:40])

    # resume：done 但媒体缺失 -> 精确降级；failed -> 回到 queued
    resume_drama = _make_drama("续跑剧", metadata=json.dumps(
        {"premise": "梗", "withImages": True}, ensure_ascii=False))
    done_ep = _make_episode(resume_drama, ap.AUTO_STATUS["done"], number=1)
    failed_ep = _make_episode(resume_drama, ap.AUTO_STATUS["failed"], number=2)
    with engine.begin() as conn:
        conn.execute(storyboards.insert().values(
            episode_id=done_ep, storyboard_number=1, title="t",
            created_at=now(), updated_at=now()))
    spawned.clear()

    async def _resume() -> None:
        ap.resume_auto_pipeline(resume_drama, {"withImages": True})
        await asyncio.sleep(0.05)

    asyncio.run(_resume())
    with engine.begin() as conn:
        statuses = [row[0] for row in conn.execute(
            select(episodes.c.status).where(episodes.c.drama_id == resume_drama)
            .order_by(episodes.c.episode_number)).all()]
    check("resume: `done` 但缺首帧 -> **精确降级到 auto:imaging**；`failed` -> 回到 auto:queued",
          statuses == [ap.AUTO_STATUS["imaging"], ap.AUTO_STATUS["queued"]], statuses)
    check("resume: override 会合并并**持久化**回 metadata；并触发后台管线",
          ap.load_pipeline_options(resume_drama)["withImages"] is True
          and spawned == [resume_drama], spawned)

    no_config_error = ""
    try:
        ap.resume_auto_pipeline(_make_drama("没有 metadata"))
    except Exception as err:  # noqa: BLE001
        no_config_error = str(err)
    check("resume: 缺 pipeline 配置 -> 报错带 dramaId",
          no_config_error.startswith("Drama ") and "缺少 pipeline 配置（metadata）" in no_config_error,
          no_config_error)

    # 补跑判定
    check("补跑: 无媒体需求 -> None（不降级、不重跑）",
          ap.media_missing_status(done_ep, ap.normalize_options({"premise": "梗"})) is None)
    check("补跑: 缺首帧 -> imaging",
          ap.media_missing_status(
              done_ep, {"premise": "梗", "withImages": True}) == ap.AUTO_STATUS["imaging"],
          ap.media_missing_status(done_ep, {"premise": "梗", "withImages": True}))
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.episode_id == done_ep)
                     .values(first_frame_image="static/frames/a.jpg"))
    check("补跑: 首帧齐了但缺视频 -> videoing（**逐级往下**，不回头重跑图片）",
          ap.media_missing_status(done_ep, {"premise": "梗", "withImages": True,
                                            "withVideos": True}) == ap.AUTO_STATUS["videoing"],
          ap.media_missing_status(done_ep, {"premise": "梗", "withImages": True,
                                            "withVideos": True}))

    # ================= 端点 =================
    client = TestClient(app)
    check("端点: POST /run 缺 premise -> 400 `premise 不能为空`",
          client.post("/api/v1/auto-pipeline/run", json={}).json()
          == {"code": 400, "message": "premise 不能为空"})
    check("端点: POST /run premise 非字符串（123）-> 400（truthy 判定）",
          client.post("/api/v1/auto-pipeline/run", json={"premise": 123}).status_code == 400)
    run_ok = client.post("/api/v1/auto-pipeline/run", json={"premise": "端到端跑一次"})
    check("端点: POST /run 成功 -> 200 + {dramaId, episodeIds}",
          run_ok.status_code == 200 and set(run_ok.json()["data"]) == {"dramaId", "episodeIds"},
          run_ok.json().get("data"))
    check("端点: GET /status 非法 id -> 400 `invalid dramaId`；不存在 -> 404",
          client.get("/api/v1/auto-pipeline/status/abc").json()
          == {"code": 400, "message": "invalid dramaId"}
          and client.get("/api/v1/auto-pipeline/status/999999").json()
          == {"code": 404, "message": "drama not found"})
    status_ok = client.get(f"/api/v1/auto-pipeline/status/{result['dramaId']}")
    check("端点: GET /status 正常 -> 200 且 running 字段在",
          status_ok.status_code == 200 and "running" in status_ok.json()["data"])
    check("端点: POST /resume 非法 id -> 400；缺配置 -> **500**（不是 400）",
          client.post("/api/v1/auto-pipeline/resume/abc").json()
          == {"code": 400, "message": "invalid dramaId"}
          and client.post(f"/api/v1/auto-pipeline/resume/{_make_drama('无配置')}").status_code == 500)

    # SSE —— ⚠️ 不通过 TestClient 读流（portal 与永不结束的流容易互相等死），
    # 直接调用路由函数拿 `StreamingResponse`，自己按帧迭代（带超时，卡住就报错而不是挂死）。
    ap_route._HEARTBEAT_INTERVAL_SECONDS = 0.05  # type: ignore[attr-defined]  # noqa: SLF001

    async def _read_sse() -> tuple[Any, list[str]]:
        response = await ap_route.stream_pipeline(str(result["dramaId"]))
        iterator = response.body_iterator
        frames: list[str] = []
        first = await asyncio.wait_for(iterator.__anext__(), 2)
        # ⚠️ 帧以空行结尾 ⇒ 收集时过滤掉空行（否则索引会被空行插位）
        frames.extend([line for line in str(first).splitlines() if line])
        # 订阅后再推事件：应作为增量出现（回放协议 = 快照 + 增量）
        sse_hub.publish_pipeline_event(result["dramaId"], {
            "type": "status", "episodeId": 1, "status": "auto:scripting",
        })
        second = await asyncio.wait_for(iterator.__anext__(), 2)
        frames.extend([line for line in str(second).splitlines() if line])
        await iterator.aclose()
        return response, frames

    sse_response, frames = asyncio.run(_read_sse())
    check("SSE: 200 + text/event-stream + 禁缓冲/不缓存头",
          sse_response.media_type == "text/event-stream"
          and sse_response.headers.get("x-accel-buffering") == "no"
          and sse_response.headers.get("cache-control") == "no-cache",
          dict(sse_response.headers))
    check("SSE: **第一帧是 snapshot**（历史状态走快照，流只推增量）",
          frames[0] == "event: snapshot" and frames[1].startswith("data: {")
          and json.loads(frames[1][6:])["dramaId"] == result["dramaId"],
          frames[:2])
    check("SSE: 订阅后发布的事件以**增量帧**出现（事件名 = 事件 type）",
          frames[2] == "event: status"
          and json.loads(frames[3][6:])["status"] == "auto:scripting",
          frames[2:4])
    check("SSE: 关闭流后**订阅被清理**（不留频道、不留心跳任务）",
          sse_hub._channels.get(result["dramaId"]) is None,  # noqa: SLF001
          list(sse_hub._channels))  # noqa: SLF001

    sse_hub._channels.clear()  # noqa: SLF001
    check("内存: 事件总线在订阅者全部取消后**不留频道**（防泄漏）",
          sse_hub._channels == {})  # noqa: SLF001

    # ================= 汇总 =================
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
