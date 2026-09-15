"""S6 自检：``export_service`` + ``GET /export/dramas/{id}/edl`` + ``GET /export/dramas/{id}``。

导出链路是**交付物**（给 Premiere/达芬奇/ZIP），四类错都静默：路径 sanitize、扩展名推断、
「未落盘媒体要跳过」、**EDL 时间码的 round 语义**。

媒体文件在临时目录里**真实落盘**（`static/...` → 由 `to_abs_media_path` 反推位置），
ffprobe 探测路径**不打桩但走不到**（种子里给显式 duration，另一条故意不给 ⇒ 断言被跳过）。

运行::

    ./.venv/Scripts/python.exe tests/export_service_test.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="expsvc_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import characters, dramas, episodes, scenes, storyboards  # noqa: E402
from app.core.response import now  # noqa: E402
from app.services.export_service import (  # noqa: E402
    build_export_zip,
    collect_drama_export_files,
    ext_of,
    sanitize_name,
    seconds_to_timecode,
    to_local_abs_path,
)
from app.services.frame_extractor import to_abs_media_path  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _write_media(relative: str, payload: bytes = b"media") -> str:
    """在 ``static`` 下真实落盘，返回相对路径。"""
    absolute = to_abs_media_path(relative)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    with open(absolute, "wb") as handle:
        handle.write(payload)
    return relative


def main() -> int:  # noqa: C901
    # ================= 纯函数 =================
    check("名字: 非法字符换 _、保留中英文数字，空值回退 untitled",
          sanitize_name('三娘/第一集:"a?<>|') == "三娘_第一集__a____"
          and sanitize_name("  ") == "untitled" and sanitize_name(None) == "untitled"
          and sanitize_name("林晚 01") == "林晚 01", sanitize_name('三娘/第一集:"a?<>|'))
    check("扩展名: 吃掉 query、转小写、认不出回 bin",
          ext_of("a/b.MP4?token=1") == "mp4" and ext_of("a/b") == "bin"
          and ext_of("x.webm?x=1&y=2") == "webm" and ext_of(None) == "bin")
    check("路径: data:/http(s) 一律 None（未落盘）",
          to_local_abs_path("data:image/png;base64,AAA") is None
          and to_local_abs_path("http://x/a.png") is None
          and to_local_abs_path("https://x/a.png") is None
          and to_local_abs_path(None) is None)
    check("路径: 落盘文件返回绝对路径、缺失文件返回 None",
          to_local_abs_path(_write_media("static/ok.png")).endswith("ok.png")
          and to_local_abs_path("static/not-there.png") is None)
    check("时间码: HH:MM:SS:FF（fps 默认 25；1.5s -> 总帧 37.5 -> **38** 即 :13）",
          seconds_to_timecode(0) == "00:00:00:00"
          and seconds_to_timecode(1) == "00:00:01:00"
          and seconds_to_timecode(1.5) == "00:00:01:13"
          and seconds_to_timecode(3661.04) == "01:01:01:01"
          and seconds_to_timecode(0.5, fps=30) == "00:00:00:15",
          seconds_to_timecode(1.5))
    check("时间码: 负数夹到 0；**``Math.round`` 语义（floor(x+.5)，不是银行家舍入）**",
          seconds_to_timecode(-3) == "00:00:00:00"
          and seconds_to_timecode(0.02) == "00:00:00:01"  # 0.5 帧 -> 进 1（Python round 会给 0）
          and seconds_to_timecode(0.0199) == "00:00:00:00",
          seconds_to_timecode(0.02))

    # ================= 种一棵可导出的树 =================
    ts = now()
    video = _write_media("static/videos/ep1.mp4", b"v" * 32)
    char_img = _write_media("static/images/char.png", b"c" * 16)
    scene_img = _write_media("static/images/scene.png", b"s" * 8)
    first = _write_media("static/frames/first.png", b"f")
    last = _write_media("static/frames/last.png", b"l")
    tail = _write_media("static/frames/tail.png", b"t")
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="导出测试", created_at=ts, updated_at=ts)).lastrowid)
        episode_id = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="第一集/开场", content="x",
            video_url=video, status="completed", created_at=ts, updated_at=ts)).lastrowid)
        conn.execute(characters.insert().values(
            drama_id=drama_id, name="林晚", image_url=char_img,
            created_at=ts, updated_at=ts))
        conn.execute(characters.insert().values(
            drama_id=drama_id, name="已删角色", image_url=char_img, deleted_at=ts,
            created_at=ts, updated_at=ts))
        conn.execute(characters.insert().values(
            drama_id=drama_id, name="远程角色", image_url="https://cdn.example.com/a.png",
            created_at=ts, updated_at=ts))
        conn.execute(scenes.insert().values(
            drama_id=drama_id, location="客栈/大堂", time="夜", prompt="",
            image_url=scene_img, created_at=ts, updated_at=ts))
        conn.execute(storyboards.insert().values(
            episode_id=episode_id, storyboard_number=1, title="第一镜", description="描述",
            video_url=video, duration=2.5, first_frame_image=first,
            last_frame_image=last, tail_frame_image=tail,
            created_at=ts, updated_at=ts))
        conn.execute(storyboards.insert().values(
            episode_id=episode_id, storyboard_number=2, title="第二镜",
            video_url=video, duration=0, created_at=ts, updated_at=ts))

    with engine.begin() as conn:
        files = collect_drama_export_files(conn, drama_id, "all")
    zip_paths = [entry["zipPath"] for entry in files]
    check("收集: 整集成片归到 `videos/ep01_标题.ext`（集号 2 位补零 + 标题 sanitize）",
          "videos/ep01_第一集_开场.mp4" in zip_paths, zip_paths)
    check("收集: 角色立绘**排除软删**、**排除远程 URL**",
          "characters/林晚.png" in zip_paths
          and not any("已删角色" in p for p in zip_paths)
          and not any("远程角色" in p for p in zip_paths), zip_paths)
    check("收集: 场景图（地点 sanitize）",
          "scenes/客栈_大堂.png" in zip_paths, zip_paths)
    check("收集: 分镜三帧各自成条目（first / last / **tail 真实尾帧单独导出**）",
          all(p in zip_paths for p in ("storyboards/ep01/sb01_first.png",
                                       "storyboards/ep01/sb01_last.png",
                                       "storyboards/ep01/sb01_tail.png")), zip_paths)
    with engine.begin() as conn:
        only_video = [e["zipPath"] for e in collect_drama_export_files(conn, drama_id, "video")]
        only_assets = [e["zipPath"] for e in collect_drama_export_files(conn, drama_id, "assets")]
    check("scope: `video` 只打成片；`assets` 只要源素材（互不包含）",
          only_video == ["videos/ep01_第一集_开场.mp4"]
          and all(p.startswith(("characters/", "scenes/", "storyboards/")) for p in only_assets)
          and len(only_assets) == 5, (only_video, len(only_assets)))

    built = build_export_zip(files)
    try:
        with zipfile.ZipFile(built) as archive:
            infos = archive.infolist()
            check("ZIP: 条目与 zipPath 一一对应，且内容字节一致",
                  sorted(i.filename for i in infos) == sorted(zip_paths)
                  and archive.read("videos/ep01_第一集_开场.mp4") == b"v" * 32)
            check("ZIP: 用 deflate（对齐 TS 的 level=6）",
                  all(i.compress_type == zipfile.ZIP_DEFLATED for i in infos))
    finally:
        os.remove(built)

    # ================= 端点 =================
    client = TestClient(app)
    check("端点: EDL 非法 id / 剧不存在 -> 404",
          client.get("/api/v1/export/dramas/abc/edl").status_code == 404
          and client.get("/api/v1/export/dramas/999999/edl").status_code == 404)
    edl = client.get(f"/api/v1/export/dramas/{drama_id}/edl")
    text = edl.text
    check("端点: EDL 是**裸 text/plain + attachment**（不是统一信封）",
          edl.status_code == 200
          and edl.headers["content-type"].startswith("text/plain")
          and edl.headers["content-disposition"] == f'attachment; filename="drama-{drama_id}.edl"'
          and json_like(text) is None, edl.headers.get("content-disposition"))
    check("端点: EDL 头部两行 + 事件行（001 起始、带时间码、CLIP NAME / COMMENT）",
          text.splitlines()[0] == "TITLE: 导出测试"
          and text.splitlines()[1] == "FCM: NON-DROP FRAME"
          and "001  AX       V     C        00:00:00:00 00:00:02:13 00:00:00:00 00:00:02:13"
          in text
          and "* FROM CLIP NAME: ep1.mp4" in text
          and "* COMMENT: 第一镜 - 描述" in text, text[:200])
    check("端点: 时长缺失的分镜（且探测失败）**被跳过** —— 只有 1 条事件",
          text.count("AX       V     C") == 1, text.count("AX       V     C"))
    check("端点: `fps=30` 生效（2.5s * 30 = 75 帧 -> 00:00:02:15）",
          "00:00:02:15" in client.get(
              f"/api/v1/export/dramas/{drama_id}/edl", params={"fps": "30"}).text)
    check("端点: ZIP scope 非法 -> 400（文案带枚举）",
          client.get(f"/api/v1/export/dramas/{drama_id}",
                     params={"scope": "nope"}).json()
          == {"code": 400, "message": "scope must be one of: all | video | assets"})
    with engine.begin() as conn:
        empty_drama = int(conn.execute(dramas.insert().values(
            title="空剧", created_at=ts, updated_at=ts)).lastrowid)
    check("端点: 没有可导出素材 -> 400（文案固定）",
          client.get(f"/api/v1/export/dramas/{empty_drama}").json()
          == {"code": 400,
              "message": "No exportable assets found (videos/assets not generated yet)"})
    zip_resp = client.get(f"/api/v1/export/dramas/{drama_id}", params={"scope": "video"})
    check("端点: ZIP 是裸 application/zip + attachment + **X-Export-Count**",
          zip_resp.status_code == 200
          and zip_resp.headers["content-type"] == "application/zip"
          and zip_resp.headers["x-export-count"] == "1"
          and zip_resp.headers["content-disposition"]
          == f'attachment; filename="drama-{drama_id}-export.zip"', dict(zip_resp.headers))
    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as archive:
        check("端点: 返回的字节是**合法 ZIP**，内含成片",
              archive.namelist() == ["videos/ep01_第一集_开场.mp4"])

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


def json_like(text: str) -> str | None:
    """EDL 不该是 JSON 信封（返回 None 表示「不是 JSON」）。"""
    return text if text.lstrip().startswith("{") else None


if __name__ == "__main__":
    raise SystemExit(main())
