"""merge 自检：整集拼接流水线 + 两个端点（``ffmpeg_merge.py`` / ``routers/merge.py``）。

**不装 ffmpeg、不拼真视频**：``_run_ffmpeg`` 换成假的（记录参数 + 落空文件），
``probe_video_duration`` / ``_video_has_audio`` 也换成假的，于是能精确锁住：

1. **前置校验**：必须**全部**分镜都已合成（``Only composed storyboards can be merged (n/m ready)``），
   一个都没有时是 ``No videos to merge``；
2. concat 列表文件的**内容与切片顺序**（按 ``storyboard_number`` 升序，不是 id）；
3. 三段 ffmpeg 的参数：concat 拼接、BGM 混音（``volume/fade/amix/alimiter``）、响度归一化；
4. 中间文件清理 + ``video_merges`` / ``episodes.video_url`` 回写；
5. BGM 失败**回退**（不阻断），无音轨**跳过** loudnorm。

运行::

    ./.venv/Scripts/python.exe tests/merge_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="merge_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import get_storage_root  # noqa: E402
from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import episodes, storyboards, video_merges  # noqa: E402
from app.response import now  # noqa: E402
from app.routers import merge as merge_router  # noqa: E402
from app.services import ffmpeg_merge as fm  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


_FFMPEG_CALLS: list[list[str]] = []
_FAIL_FFMPEG = [False]
_HAS_AUDIO = [True]
_DURATION = [42]


async def _fake_run_ffmpeg(args: list[str]) -> None:
    _FFMPEG_CALLS.append(list(args))
    if _FAIL_FFMPEG[0]:
        raise RuntimeError("ffmpeg exited with code 1: boom")
    # 输出永远是最后一个参数
    Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
    Path(args[-1]).write_bytes(b"fake-mp4")


async def _fake_probe(_path: str) -> int:
    return _DURATION[0]


async def _fake_has_audio(_path: str) -> bool:
    return _HAS_AUDIO[0]


async def _noop_batch(*_args, **_kwargs) -> None:
    """隔离后台拼接：真身开自己的事务，会与测试连接在 SQLite（单写者）上互锁。"""


#: 真身后台入口（patch 前留一份：失败路径要单独驱动它）
_REAL_GUARDED = fm._do_merge_guarded


def _row(table, row_id: int, key):
    with engine.begin() as conn:
        return conn.execute(select(table).where(key == row_id)).first()


async def _call_service(episode_id: int, drama_id: int) -> int:
    """在事件循环里调服务（``merge_episode_videos`` 内部用 ``asyncio.create_task``，
    而路由是 async 端点的天然有循环；同步的测试 ``main`` 里没有）。"""
    merge_id = fm.merge_episode_videos(episode_id, drama_id)
    await asyncio.sleep(0)  # 让火忘任务跑完（已被换成 no-op）
    return merge_id


def main() -> int:  # noqa: C901
    fm._run_ffmpeg = _fake_run_ffmpeg  # type: ignore[assignment]
    fm.probe_video_duration = _fake_probe  # type: ignore[assignment]
    fm._video_has_audio = _fake_has_audio  # type: ignore[assignment]
    fm._do_merge_guarded = _noop_batch  # type: ignore[assignment]

    client = TestClient(app)
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE merge"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]

    def new_sb(number: int, **fields) -> int:
        sid = client.post("/api/v1/storyboards", json={
            "episode_id": episode_id, "title": f"镜头{number}", "storyboard_number": number,
        }).json()["data"]["id"]
        if fields:
            with engine.begin() as conn:
                conn.execute(storyboards.update().where(storyboards.c.id == sid).values(**fields))
        return sid

    # 故意乱序创建：拼接必须按 storyboard_number 排，不是 id
    sb3 = new_sb(3, composed_video_url="static/composed/c3.mp4")
    sb1 = new_sb(1, composed_video_url="static/composed/c1.mp4")
    sb2 = new_sb(2, composed_video_url="static/composed/c2.mp4")

    # ================= 前置校验 =================
    sb_uncomposed = new_sb(4)  # 没有 composed_video_url
    err = ""
    try:
        asyncio.run(_call_service(episode_id, drama_id))
    except ValueError as exc:
        err = str(exc)
    check(
        "校验: 有人没合成 -> 报 `Only composed storyboards can be merged (3/4 ready)`",
        err == "Only composed storyboards can be merged (3/4 ready)", err,
    )
    with engine.begin() as conn:
        conn.execute(storyboards.delete().where(storyboards.c.id == sb_uncomposed))

    empty_episode = _insert_episode(drama_id, "空集")
    err_empty = ""
    try:
        asyncio.run(_call_service(empty_episode, drama_id))
    except ValueError as exc:
        err_empty = str(exc)
    check("校验: 一个分镜都没有 -> `No videos to merge`（不是 n/m 那条）",
          err_empty == "No videos to merge", err_empty)

    # ================= 建记录 + 切片顺序 =================
    with engine.begin() as conn:
        conn.execute(episodes.update().where(episodes.c.id == episode_id).values(
            bgm_url="static/audio/bgm.mp3", bgm_volume=0.4,
            bgm_fade_in=1.0, bgm_fade_out=2.0))
    (Path(get_storage_root()) / "audio").mkdir(parents=True, exist_ok=True)
    (Path(get_storage_root()) / "audio" / "bgm.mp3").write_bytes(b"bgm")

    _FFMPEG_CALLS.clear()
    merge_id = asyncio.run(_call_service(episode_id, drama_id))
    merge_row = _row(video_merges, merge_id, video_merges.c.id)
    check(
        "记录: 插入 video_merges（processing + ffmpeg 供应商 + 标题/模型）",
        merge_row.status == "processing" and merge_row.provider == "ffmpeg"
        and merge_row.model == "ffmpeg-concat-h264-aac"
        and merge_row.title == f"Episode {episode_id} Merge",
        (merge_row.status, merge_row.provider, merge_row.model),
    )
    check(
        "记录: scenes 列是**紧凑 JSON**（与 Node 的 JSON.stringify 同形）且按镜号升序",
        merge_row.scenes
        == '["static/composed/c1.mp4","static/composed/c2.mp4","static/composed/c3.mp4"]',
        merge_row.scenes,
    )
    check("后台: 拼接任务不阻塞返回（记录已是 processing，ffmpeg 还没跑）",
          _FFMPEG_CALLS == [], _FFMPEG_CALLS)

    # ================= 流水线（直接调 _do_merge） =================
    _FFMPEG_CALLS.clear()
    asyncio.run(fm._do_merge(merge_id, episode_id,
                             ["static/composed/c1.mp4", "static/composed/c2.mp4",
                              "static/composed/c3.mp4"],
                             {"bgmUrl": "static/audio/bgm.mp3", "bgmVolume": 0.4,
                              "bgmFadeIn": 1.0, "bgmFadeOut": 2.0}))
    done = _row(video_merges, merge_id, video_merges.c.id)
    ep_row = _row(episodes, episode_id, episodes.c.id)
    concat_call = _FFMPEG_CALLS[0]

    check(
        "拼接: 命令是 concat 解复用器（`-f concat -safe 0 -i <list>`）",
        concat_call[1:6] == ["-f", "concat", "-safe", "0", "-i"], concat_call[1:6],
    )
    check(
        "拼接: 编码参数 `+genpts / libx264 / preset medium / crf 23 / aac 48k 192k / faststart`",
        "+genpts" in concat_call and "medium" in concat_call
        and concat_call[concat_call.index("-ar") + 1] == "48000"
        and concat_call[concat_call.index("-b:a") + 1] == "192k"
        and "+faststart" in concat_call,
        concat_call,
    )
    check("拼接: 输出落在 static/merged/<uuid>.mp4",
          concat_call[-1].replace("\\", "/").find("/merged/") >= 0
          and concat_call[-1].endswith(".mp4"), concat_call[-1])
    check(
        "清理: concat 列表文件用完即删（temp 目录不留垃圾）",
        not list((Path(get_storage_root()) / "temp").glob("*.txt")),
        list((Path(get_storage_root()) / "temp").glob("*")),
    )

    bgm_call = _FFMPEG_CALLS[1]
    complex_filter = bgm_call[bgm_call.index("-filter_complex") + 1]
    check(
        "BGM: 用 `-stream_loop -1` 循环 BGM（属于后加入的那个输入）",
        "-stream_loop" in bgm_call and bgm_call[bgm_call.index("-stream_loop") + 1] == "-1",
        bgm_call[:8],
    )
    check(
        "BGM: 滤镜链含 volume(3 位小数) / afade in+out / amix / alimiter",
        "volume=0.400" in complex_filter and "afade=t=in:st=0:d=1.00" in complex_filter
        and "afade=t=out:st=40.00:d=2.00" in complex_filter
        and "amix=inputs=2:duration=first:normalize=0" in complex_filter
        and "alimiter=limit=0.95" in complex_filter,
        complex_filter,
    )
    check("BGM: 视频流直接 copy + 映射 `0:v` 与 `[aout]`",
          bgm_call[bgm_call.index("-c:v") + 1] == "copy"
          and "0:v" in bgm_call and "[aout]" in bgm_call, bgm_call,
    )
    check("BGM: 混音成功后删掉无 BGM 的中间文件",
          not Path(concat_call[-1]).exists(), concat_call[-1])

    loud_call = _FFMPEG_CALLS[2]
    check(
        "响度: `loudnorm=I=-14:TP=-1.5:LRA=11`（对齐社媒验收）",
        loud_call[loud_call.index("-af") + 1] == "loudnorm=I=-14:TP=-1.5:LRA=11",
        loud_call,
    )
    check("响度: 视频流 copy（不重编码画面）", loud_call[loud_call.index("-c:v") + 1] == "copy")

    check(
        "完成: merge 记录 -> completed + merged_url + duration + completed_at",
        done.status == "completed" and (done.merged_url or "").startswith("static/merged/")
        and done.duration == 42 and done.completed_at is not None,
        (done.status, done.merged_url, done.duration),
    )
    check(
        "完成: 同一路径回写 episodes.video_url（成片即剧集视频）",
        ep_row.video_url == done.merged_url, (ep_row.video_url, done.merged_url),
    )
    check("完成: 最终文件名与 merged_url 一致（BGM/loudnorm 会换新文件）",
          done.merged_url == f"static/merged/{Path(loud_call[-1]).name}",
          (done.merged_url, Path(loud_call[-1]).name))

    # ================= 无音轨：跳过 loudnorm =================
    _HAS_AUDIO[0] = False
    _FFMPEG_CALLS.clear()
    asyncio.run(fm._do_merge(merge_id, episode_id, ["static/composed/c1.mp4"], None))
    check(
        "无音轨: 未开启 BGM 时只有 1 次 ffmpeg（concat），**不做** loudnorm",
        len(_FFMPEG_CALLS) == 1, _FFMPEG_CALLS,
    )
    _HAS_AUDIO[0] = True

    # 有音轨但无 BGM -> 2 次（concat + loudnorm）
    _FFMPEG_CALLS.clear()
    asyncio.run(fm._do_merge(merge_id, episode_id, ["static/composed/c1.mp4"], None))
    check("有音轨无 BGM: 恰好 2 次 ffmpeg（concat + loudnorm）",
          len(_FFMPEG_CALLS) == 2, len(_FFMPEG_CALLS))

    # ================= BGM 失败回退 =================
    _FFMPEG_CALLS.clear()
    asyncio.run(fm._do_merge(merge_id, episode_id, ["static/composed/c1.mp4"],
                             {"bgmUrl": "static/audio/does-not-exist.mp3", "bgmVolume": 0.3,
                              "bgmFadeIn": 1.5, "bgmFadeOut": 2.0}))
    recovered = _row(video_merges, merge_id, video_merges.c.id)
    check(
        "BGM 失败: 只告警并**回退无 BGM 版本**（不阻断），记录仍 completed",
        recovered.status == "completed" and len(_FFMPEG_CALLS) == 2,
        (recovered.status, len(_FFMPEG_CALLS)),
    )

    # ================= 失败路径 =================
    _FAIL_FFMPEG[0] = True
    fail_id = asyncio.run(_call_service(episode_id, drama_id))
    try:
        asyncio.run(_REAL_GUARDED(fail_id, episode_id, ["static/composed/c1.mp4"], None))
    finally:
        _FAIL_FFMPEG[0] = False
    failed_row = _row(video_merges, fail_id, video_merges.c.id)
    check(
        "失败: merge 记录 -> failed + error_msg（异常被吞在 guard 里）",
        failed_row.status == "failed" and "ffmpeg exited with code 1" in (failed_row.error_msg or ""),
        (failed_row.status, failed_row.error_msg),
    )

    # ================= 路由 =================
    # 隔离后台任务（真身会与测试事务在 SQLite 上互锁）
    original = fm.merge_episode_videos

    def _fake_service(ep_id: int, dr_id: int) -> int:
        with engine.begin() as conn:
            return int(conn.execute(video_merges.insert().values(
                episode_id=ep_id, drama_id=dr_id, title="fake", provider="ffmpeg",
                model="m", status="processing", scenes="[]", created_at=now(),
            )).lastrowid)

    merge_router.merge_episode_videos = _fake_service  # type: ignore[assignment]
    resp = client.post(f"/api/v1/merge/episodes/{episode_id}/merge")
    body = resp.json()
    check(
        "路由: POST /merge 挂在 /api/v1/merge 下，返回 **snake_case** {merge_id,status}",
        resp.status_code == 200 and set(body["data"]) == {"merge_id", "status"}
        and body["data"]["status"] == "processing",
        body,
    )
    check("路由: POST 非法 episode id -> 404",
          client.post("/api/v1/merge/episodes/abc/merge").status_code == 404)
    check("路由: POST 不存在的剧集 -> 400 Episode not found",
          client.post("/api/v1/merge/episodes/999999/merge").status_code == 400)

    merge_router.merge_episode_videos = original  # type: ignore[assignment]
    # 此时 sb1..sb3 已合成，但上面 `_do_merge` 的假实验删过中间文件 ⇒ 真实校验依然过；
    # 用一个新分镜（未合成）把它顶成「不齐」再验错误消息
    sb_extra = new_sb(9)
    not_ready = client.post(f"/api/v1/merge/episodes/{episode_id}/merge")
    check(
        "路由: 有镜头未合成 -> 400，且消息透出进度 `(n/m ready)`",
        not_ready.status_code == 400 and "ready)" in not_ready.json()["message"],
        (not_ready.status_code, not_ready.json().get("message")),
    )
    with engine.begin() as conn:
        conn.execute(storyboards.delete().where(storyboards.c.id == sb_extra))

    got = client.get(f"/api/v1/merge/episodes/{episode_id}/merge").json()["data"]
    check(
        "路由: GET 取**最新**一条（按 id 倒序）且为 snake_case 全字段",
        got["id"] == body["data"]["merge_id"] and "merged_url" in got
        and "episode_id" in got and "created_at" in got,
        sorted(got)[:6],
    )
    empty_status = client.get(f"/api/v1/merge/episodes/{empty_episode}/merge").json()["data"]
    check("路由: GET 无记录 -> success(null)（**不是 404**）", empty_status is None, empty_status)
    check("路由: GET 非法 id -> 404",
          client.get("/api/v1/merge/episodes/abc/merge").status_code == 404)

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


def _insert_episode(drama_id: int, title: str) -> int:
    values: dict[str, object] = {"drama_id": drama_id}
    if "title" in episodes.c:
        values["title"] = title
    if "episode_number" in episodes.c:
        with engine.begin() as conn:
            values["episode_number"] = len(conn.execute(select(episodes.c.id)).all()) + 90
    for column in ("created_at", "updated_at"):
        if column in episodes.c:
            values[column] = now()
    with engine.begin() as conn:
        return int(conn.execute(episodes.insert().values(**values)).lastrowid)


if __name__ == "__main__":
    raise SystemExit(main())
