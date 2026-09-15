"""S7 自检：设置分镜首/尾帧（``POST /storyboards/{id}/set-frame``）+ 抽帧泛化（``extract_frame``）。

锁三件事：**首帧不 seek / 尾帧从末尾回退 0.2s**、**只有 `last_frame` 走尾帧（其余一律 first_frame）**、
**视频素材才抽帧、图片素材直接采用**。``_run_ffmpeg`` 全打桩 ⇒ 没装 ffmpeg 也能跑。

运行::

    ./.venv/Scripts/python.exe tests/set_frame_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="setf_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import get_data_root  # noqa: E402
from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import dramas, episodes, storyboards  # noqa: E402
from app.core.response import now  # noqa: E402
from app.services import frame_extractor as fx  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:  # noqa: C901
    calls: list[list[str]] = []

    async def fake_ffmpeg(args):
        calls.append(list(args))
        # 模拟 ffmpeg 真的写出了文件（便于断言返回路径存在）
        Path(args[-1]).write_bytes(b"jpeg")

    fx._run_ffmpeg = fake_ffmpeg  # type: ignore[assignment]

    async def fake_duration(path):
        return 10.0

    fx._probe_duration_seconds = fake_duration  # type: ignore[assignment]

    # 落一个「假视频」文件（内容无所谓，只要存在）。
    # ⚠️ 路径必须是 `<data_root>/static/videos/...`：`to_abs_media_path("static/videos/x")`
    #    会解析到 `<data_root>/**static**/videos/x`（数据根下有个 static 子目录）。
    media_dir = Path(get_data_root()) / "static" / "videos"
    media_dir.mkdir(parents=True, exist_ok=True)
    (media_dir / "clip.mp4").write_bytes(b"fake")

    calls.clear()
    first = asyncio.run(fx.extract_frame("static/videos/clip.mp4", "first_frame"))
    check("抽帧: 首帧**不 seek**（参数就是 -i … -frames:v 1 -q:v 2）+ 落 static/frames/",
          calls and "-ss" not in calls[0] and calls[0][0] == "-i"
          and "-frames:v" in calls[0] and calls[0][-1].endswith(".jpg")
          and str(first).startswith("static/frames/"), calls)

    calls.clear()
    last = asyncio.run(fx.extract_frame("static/videos/clip.mp4", "last_frame"))
    check("抽帧: 尾帧 seek = 时长 - 0.2s（10s -> 9.8s），`-ss` 在 `-i` **之前**",
          calls and calls[0][:2] == ["-ss", "9.8"] and calls[0][2] == "-i"
          and str(last).startswith("static/frames/"), calls)

    calls.clear()
    check("抽帧: 未传 frame_type 时按**首帧**处理（不是尾帧）",
          asyncio.run(fx.extract_frame("static/videos/clip.mp4")) is not None
          and "-ss" not in calls[0], calls)

    async def zero_duration(path):
        return 0.0

    fx._probe_duration_seconds = zero_duration  # type: ignore[assignment]
    calls.clear()
    check("抽帧: 尾帧且**时长探测失败** -> None（且不调 ffmpeg）",
          asyncio.run(fx.extract_frame("static/videos/clip.mp4", "last_frame")) is None
          and calls == [], calls)
    fx._probe_duration_seconds = fake_duration  # type: ignore[assignment]

    check("抽帧: 文件不存在 -> None", asyncio.run(fx.extract_frame("static/videos/nope.mp4")) is None)

    async def boom(args):
        raise RuntimeError("ffmpeg exited with code 1")

    fx._run_ffmpeg = boom  # type: ignore[assignment]
    check("抽帧: ffmpeg 失败 -> None（不抛，交给调用方决定）",
          asyncio.run(fx.extract_frame("static/videos/clip.mp4", "first_frame")) is None)
    fx._run_ffmpeg = fake_ffmpeg  # type: ignore[assignment]
    calls.clear()
    asyncio.run(fx.extract_tail_frame("static/videos/clip.mp4"))
    check("抽帧: `extract_tail_frame` 就是 last_frame 的薄封装（保留既有调用点语义）",
          calls and calls[0][:2] == ["-ss", "9.8"], calls)

    # ── 端点 ──
    stamp = now()
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="设帧剧", created_at=stamp, updated_at=stamp)).lastrowid)
        ep = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="第一集", content="x",
            created_at=stamp, updated_at=stamp)).lastrowid)
        sb = int(conn.execute(storyboards.insert().values(
            episode_id=ep, storyboard_number=1, title="镜一",
            created_at=stamp, updated_at=stamp)).lastrowid)

    client = TestClient(app)
    check("端点: 非法 id -> 404 'Invalid storyboard id'",
          client.post("/api/v1/storyboards/abc/set-frame",
                      json={"source_url": "static/videos/clip.mp4"}).status_code == 404)
    check("端点: 镜头不存在 -> 404 '镜头不存在'",
          client.post("/api/v1/storyboards/999999/set-frame",
                      json={"source_url": "x.mp4"}).json()
          == {"code": 404, "message": "镜头不存在"})
    check("端点: 缺 source_url -> 400 'source_url required'",
          client.post(f"/api/v1/storyboards/{sb}/set-frame", json={}).json()
          == {"code": 400, "message": "source_url required"})

    # 图片素材：不抽帧，直接采用
    img = client.post(f"/api/v1/storyboards/{sb}/set-frame",
                      json={"source_url": "static/images/pic.png",
                            "frame_type": "last_frame"})
    with engine.begin() as conn:
        row = conn.execute(select(storyboards).where(storyboards.c.id == sb)).first()
    check("端点: **图片素材不抽帧**（frame_url 原样）且写进 last_frame_image",
          img.json()["data"] == {"frame_type": "last_frame",
                                 "frame_url": "static/images/pic.png"}
          and row.last_frame_image == "static/images/pic.png"
          and row.first_frame_image is None, img.json()["data"])

    # 视频素材 + last_frame：抽帧后写入
    vid = client.post(f"/api/v1/storyboards/{sb}/set-frame",
                      json={"source_url": "static/videos/clip.mp4", "frame_type": "last_frame"})
    with engine.begin() as conn:
        row2 = conn.execute(select(storyboards).where(storyboards.c.id == sb)).first()
    check("端点: 视频素材抽帧 -> 写 last_frame_image，响应回 frame_url",
          vid.json()["data"]["frame_type"] == "last_frame"
          and str(vid.json()["data"]["frame_url"]).startswith("static/frames/")
          and row2.last_frame_image == vid.json()["data"]["frame_url"], vid.json()["data"])

    # 非 last_frame 的取值一律当 first_frame（含未知值）
    odd = client.post(f"/api/v1/storyboards/{sb}/set-frame",
                      json={"source_url": "static/images/pic.png", "frame_type": "keyframe"})
    with engine.begin() as conn:
        row3 = conn.execute(select(storyboards).where(storyboards.c.id == sb)).first()
    check("端点: `frame_type=keyframe`（未知值）当 **first_frame** 处理",
          odd.json()["data"]["frame_type"] == "first_frame"
          and row3.first_frame_image == "static/images/pic.png", odd.json()["data"])

    # 抽帧失败 -> 400
    fx._run_ffmpeg = boom  # type: ignore[assignment]
    failed = client.post(f"/api/v1/storyboards/{sb}/set-frame",
                         json={"source_url": "static/videos/clip.mp4"})
    check("端点: 抽帧失败 -> 400（不写库、不返回空 frame_url）",
          failed.status_code == 400 and "抽帧失败" in failed.json()["message"],
          failed.json())
    fx._run_ffmpeg = fake_ffmpeg  # type: ignore[assignment]

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
