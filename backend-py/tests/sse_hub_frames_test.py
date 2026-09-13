"""S6 自检：SSE 事件总线 + 尾帧提取（``utils/sse-hub.ts`` 57 行 + ``services/frame-extractor.ts`` 86 行）。

两块都是 auto-pipeline 的前置，且各有**一条不能破的红线**：

* **sse-hub**：``publishPipelineEvent`` **永不抛**、订阅者异常自吞、频道空了自动清理
  ---- 「进度推送」这个簿记动作绝不能拖垮主流程；
* **尾帧提取**：只写 ``tail_frame_image``（**真实尾帧**），**绝不碰** ``last_frame_image``
  （**设计尾帧**）---- 混了就会让真实产物污染 shot-router 的 FL2VA 决策。

ffmpeg/ffprobe 用替换缝（``_run_ffmpeg`` / ``_probe_duration_seconds``）打桩 > 无需真装。

运行::

    ./.venv/Scripts/python.exe tests/sse_hub_frames_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="ssframes_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import get_data_root, get_storage_root  # noqa: E402
from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import storyboards  # noqa: E402
from app.response import now  # noqa: E402
from app.services import frame_extractor as fr  # noqa: E402
from app.services import sse_hub  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:  # noqa: C901
    # ================= sse-hub =================
    sse_hub._channels.clear()  # noqa: SLF001
    sse_hub.publish_pipeline_event(999, {"type": "status", "episodeId": 1})  # 无订阅者
    check("总线: 频道无订阅者时 publish **静默返回**（不建频道、不缓存、不抛）",
          sse_hub._channels == {})  # noqa: SLF001

    received: list[dict] = []
    unsubscribe = sse_hub.subscribe_pipeline(7, lambda evt: received.append(evt))
    sse_hub.publish_pipeline_event(7, {"type": "status", "episodeId": 1, "status": "auto:script"})
    check("总线: 订阅者收到事件，且被补上 `ts`（ISO 串）",
          len(received) == 1 and received[0]["type"] == "status"
          and received[0]["status"] == "auto:script"
          and isinstance(received[0]["ts"], str) and received[0]["ts"].endswith("Z"),
          received)
    check("总线: **按 dramaId 分频道**（别的剧的事件不会串到本频道）",
          (sse_hub.publish_pipeline_event(8, {"type": "status", "episodeId": 2}),
           len(received) == 1)[1])

    second: list[dict] = []
    unsubscribe2 = sse_hub.subscribe_pipeline(7, lambda evt: second.append(evt))
    sse_hub.publish_pipeline_event(7, {"type": "media-progress", "episodeId": 1,
                                       "ready": 3, "total": 5})
    check("总线: 多订阅者都收到（同一次 publish 广播）",
          len(received) == 2 and len(second) == 1 and second[0]["ready"] == 3)

    unsubscribe()
    sse_hub.publish_pipeline_event(7, {"type": "status", "episodeId": 1})
    check("总线: 取消订阅后不再收到；且**频道仍在**（还有别的订阅者）",
          len(received) == 2 and len(second) == 2)

    unsubscribe2()
    check("总线: 最后一个订阅者退出 -> **频道被清理**（防内存泄漏）",
          sse_hub._channels == {})  # noqa: SLF001
    unsubscribe2()  # 重复取消
    check("总线: 重复取消订阅不抛（幂等）", True)

    # 订阅者异常自吞
    survived: list[dict] = []

    def _boom(_event: dict) -> None:
        raise RuntimeError("订阅者炸了")

    unsubscribe3 = sse_hub.subscribe_pipeline(7, _boom)
    unsubscribe4 = sse_hub.subscribe_pipeline(7, lambda evt: survived.append(evt))
    raised = ""
    try:
        sse_hub.publish_pipeline_event(7, {"type": "status", "episodeId": 1})
    except Exception as err:  # noqa: BLE001
        raised = str(err)
    check("总线: 单个订阅者抛错 **不影响其他订阅者、也绝不抛回发布方**",
          raised == "" and len(survived) == 1, (raised, survived))
    unsubscribe3()
    unsubscribe4()

    # ================= 路径映射 =================
    check("路径: `storage_root` 就是 `<data_root>/static`（两套写法等价的前提）",
          Path(get_storage_root()) == Path(get_data_root()) / "static",
          (get_data_root(), get_storage_root()))
    check("路径: 绝对路径原样；`static/x` 走 data_root；其余走 storage_root",
          fr.to_abs_media_path("/abs/x.mp4") == "/abs/x.mp4"
          and fr.to_abs_media_path("static/videos/a.mp4")
          == str(Path(get_data_root()) / "static/videos/a.mp4")
          and fr.to_abs_media_path("frames/b.jpg")
          == str(Path(get_storage_root()) / "frames/b.jpg"),
          [fr.to_abs_media_path(x) for x in ("/abs/x.mp4", "static/videos/a.mp4", "frames/b.jpg")])

    # ================= 尾帧提取（打桩 ffmpeg/ffprobe）=================
    video_dir = Path(get_storage_root()) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_file = video_dir / "ep.mp4"
    video_file.write_bytes(b"fake")

    ffmpeg_calls: list[list[str]] = []

    async def _fake_ffmpeg(args: list[str]) -> None:
        ffmpeg_calls.append(args)
        Path(args[-1]).write_bytes(b"jpg")  # 假装落了一张图

    async def _probe_5(_path: str) -> float:
        return 5.0

    fr._run_ffmpeg = _fake_ffmpeg  # type: ignore[assignment]  # noqa: SLF001
    fr._probe_duration_seconds = _probe_5  # type: ignore[assignment]  # noqa: SLF001

    check("提取: 文件不存在 -> None（**不调 ffmpeg**）",
          asyncio.run(fr.extract_tail_frame("static/videos/不存在.mp4")) is None
          and ffmpeg_calls == [])

    frame = asyncio.run(fr.extract_tail_frame("static/videos/ep.mp4"))
    args = ffmpeg_calls[0]
    check("提取: 返回 `static/frames/<uuid>.jpg` 且文件真的落在 storage_root/frames 下",
          isinstance(frame, str) and frame.startswith("static/frames/")
          and frame.endswith(".jpg")
          and Path(fr.to_abs_media_path(frame)).exists(),
          frame)
    check("提取: ffmpeg 参数 ---- `-ss` 在 `-i` 之前（fluence 的 seekInput 语义）、"
          "`-frames:v 1`、`-q:v 2`",
          args[0] == "-ss" and args[2] == "-i"
          and "-frames:v" in args and args[args.index("-frames:v") + 1] == "1"
          and "-q:v" in args and args[args.index("-q:v") + 1] == "2",
          args)
    check("提取: seek = 时长 - 0.2（5.0 -> 4.8）",
          args[1] == "4.8", args[1])

    async def _probe_zero(_path: str) -> float:
        return 0.0

    fr._probe_duration_seconds = _probe_zero  # type: ignore[assignment]  # noqa: SLF001
    ffmpeg_calls.clear()
    check("提取: 拿不到时长（<=0）-> None（不产出半截图）",
          asyncio.run(fr.extract_tail_frame("static/videos/ep.mp4")) is None
          and ffmpeg_calls == [])

    async def _ffmpeg_boom(_args: list[str]) -> None:
        raise RuntimeError("ffmpeg exited with code 1")

    fr._probe_duration_seconds = _probe_5  # type: ignore[assignment]  # noqa: SLF001
    fr._run_ffmpeg = _ffmpeg_boom  # type: ignore[assignment]  # noqa: SLF001
    check("提取: ffmpeg 失败 -> None（**不抛**，调用方照常跑下去）",
          asyncio.run(fr.extract_tail_frame("static/videos/ep.mp4")) is None)

    # ================= 批量提取（红线：只写 tail_frame_image）=================
    client = TestClient(app)
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 尾帧"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]
    other_episode = _insert_episode(drama_id, "别的集")

    with engine.begin() as conn:
        no_video = _insert_storyboard(conn, episode_id, 1)
        with_video = _insert_storyboard(conn, episode_id, 2,
                                        video_url="static/videos/ep.mp4",
                                        last_frame_image="static/images/设计尾帧.png")
        deleted = _insert_storyboard(conn, episode_id, 3, video_url="static/videos/ep.mp4")
        conn.execute(storyboards.update().where(storyboards.c.id == deleted)
                     .values(deleted_at=now()))
        other_row = _insert_storyboard(conn, other_episode, 1,
                                       video_url="static/videos/ep.mp4")

    async def _fake_ffmpeg_ok(args: list[str]) -> None:
        Path(args[-1]).write_bytes(b"jpg")

    fr._run_ffmpeg = _fake_ffmpeg_ok  # type: ignore[assignment]  # noqa: SLF001
    done = asyncio.run(fr.extract_storyboard_tail_frames(episode_id, drama_id))
    check("批量: 只处理「有视频且未软删」的分镜（1 个成功）", done == 1, done)

    with engine.begin() as conn:
        rows = {
            row.id: row for row in conn.execute(
                select(storyboards).where(storyboards.c.id.in_(
                    [no_video, with_video, deleted, other_row]))
            ).all()
        }
    check("批量: 写入了 `tail_frame_image`（真实尾帧）",
          rows[with_video].tail_frame_image is not None
          and rows[with_video].tail_frame_image.startswith("static/frames/"),
          rows[with_video].tail_frame_image)
    check("红线: **设计尾帧 `last_frame_image` 原封不动**（真实产物不冒充设计尾帧）",
          rows[with_video].last_frame_image == "static/images/设计尾帧.png",
          rows[with_video].last_frame_image)
    check("批量: 无视频的分镜被跳过（tail 仍为空）",
          rows[no_video].tail_frame_image is None)
    check("批量: **软删的分镜不处理**", rows[deleted].tail_frame_image is None)
    check("批量: 别的集的分镜不受影响", rows[other_row].tail_frame_image is None)

    # 幂等/刷新：再跑一次会覆盖成新值
    first_tail = rows[with_video].tail_frame_image
    done2 = asyncio.run(fr.extract_storyboard_tail_frames(episode_id, drama_id))
    with engine.begin() as conn:
        refreshed = conn.execute(select(storyboards.c.tail_frame_image).where(
            storyboards.c.id == with_video)).first()[0]
    check("批量: 可重复跑（幂等刷新）---- 视频重生成后尾帧会被刷新成新文件",
          done2 == 1 and refreshed != first_tail and refreshed.startswith("static/frames/"),
          (first_tail, refreshed))

    # 全部失败时返回 0 且不写库
    fr._run_ffmpeg = _ffmpeg_boom  # type: ignore[assignment]  # noqa: SLF001
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == with_video)
                     .values(tail_frame_image=None))
    failed_done = asyncio.run(fr.extract_storyboard_tail_frames(episode_id, drama_id))
    with engine.begin() as conn:
        after = conn.execute(select(storyboards.c.tail_frame_image).where(
            storyboards.c.id == with_video)).first()[0]
    check("批量: 提取全失败 -> 返回 0 且**不改库**", failed_done == 0 and after is None,
          (failed_done, after))

    # ================= 汇总 =================
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


def _insert_episode(drama_id: int, title: str) -> int:
    from app.models import episodes

    values: dict[str, object] = {"drama_id": drama_id, "created_at": now(), "updated_at": now()}
    if "title" in episodes.c:
        values["title"] = title
    if "episode_number" in episodes.c:
        with engine.begin() as conn:
            values["episode_number"] = len(conn.execute(select(episodes.c.id)).all()) + 900
    with engine.begin() as conn:
        return int(conn.execute(episodes.insert().values(**values)).lastrowid)


def _insert_storyboard(conn, episode_id: int, number: int, **extra) -> int:
    values: dict[str, object] = {
        "episode_id": episode_id, "storyboard_number": number,
        "created_at": now(), "updated_at": now(),
    }
    values.update(extra)
    return int(conn.execute(storyboards.insert().values(**values)).lastrowid)


if __name__ == "__main__":
    raise SystemExit(main())
