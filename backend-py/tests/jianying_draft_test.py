"""S6 自检：剪映草稿导出（``jianying-draft.ts`` 339 行）+ ``GET /export/dramas/{id}/jianying-draft``。

草稿是**给剪映吃的私有格式**（微秒计时 / UUID 引用 / materials+tracks 三段式），抄错只会
「剪映打不开」而不会报错，故锁死：草稿目录结构、素材命名 `video_001`、UUID 引用一致性、
`us_of` 的 nullish 与**最小 1 微秒**、以及 files 清单的 ZIP 内路径前缀。

素材在 ``static``/storage 下**真实落盘**（复制进 ``.draft/media``）；不打网络、不调模型。

运行::

    ./.venv/Scripts/python.exe tests/jianying_draft_test.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="jyd_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_data_root  # noqa: E402
from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import dramas, episodes, storyboards, video_generations  # noqa: E402
from app.response import now  # noqa: E402
from app.services import jianying_draft as jd  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _write_static(relative: str, payload: bytes) -> str:
    """在数据根下的 ``static/...`` 真实落盘，返回相对路径（`static/...`）。"""
    absolute = os.path.join(get_data_root(), relative)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    with open(absolute, "wb") as handle:
        handle.write(payload)
    return relative


def main() -> int:  # noqa: C901
    # ================= 助手（阶段 1）=================
    check("助手: sanitize 非法字符→_、去空白、**截断 40 字符**、空回退 episode",
          jd.sanitize('第一集/开场:"a?b') == "第一集_开场__a_b"
          and jd.sanitize("x" * 60) == "x" * 40
          and jd.sanitize("   ") == "episode" and jd.sanitize(None) == "episode",
          jd.sanitize('第一集/开场:"a?b'))
    check("助手: us_of 是 nullish（只有 None 才给 5s）+ 最小 1 微秒 + Math.round 语义",
          (jd.us_of(None), jd.us_of(0), jd.us_of(1.5), jd.us_of(2.0000005))
          == (5_000_000, 1, 1_500_000, 2_000_001), jd.us_of(2.0000005))
    media_ok = _write_static("static/a.mp4", b"a")
    check("助手: to_media_path —— static/ 挂**数据根**、缺失回 None",
          jd.to_media_path(media_ok) == os.path.join(get_data_root(), media_ok)
          and jd.to_media_path(None) is None
          and jd.to_media_path("static/missing.mp4") is None, jd.to_media_path(media_ok))
    check("助手: copy_to_media 命名 `kind_001` 三位补零、失败返回 None 且不抛",
          (lambda: (lambda copied: copied.endswith("video_002.mp4"))(
              jd.copy_to_media(jd.to_media_path("static/a.mp4"),
                               os.path.join(get_data_root(), "m"), 2, "video")))()
          and jd.copy_to_media(os.path.join(get_data_root(), "nope.mp4"),
                               os.path.join(get_data_root(), "m"), 0, "audio") is None)
    check("助手: segment/text_segment 用**大写 UUID**，且字段集不同（字幕无 speed/volume、多 transform）",
          len(jd.segment("m", 0, 10)["id"]) == 36
          and jd.segment("m", 0, 10)["id"] == jd.segment("m", 0, 10)["id"].upper()
          and "speed" in jd.segment("m", 0, 10) and "transform" not in jd.segment("m", 0, 10)
          and "transform" in jd.text_segment("m", 0, 10)
          and "speed" not in jd.text_segment("m", 0, 10))

    # ================= 种子 =================
    ts = now()
    composed = _write_static("static/composed/sb1.mp4", b"V1" * 8)
    composed2 = _write_static("static/composed/sb2.mp4", b"V2" * 8)
    tts = _write_static("static/audio/sb1.mp3", b"A1" * 4)
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="草稿剧", created_at=ts, updated_at=ts)).lastrowid)
        ep1 = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="第一集/开场", content="x",
            status="completed", created_at=ts, updated_at=ts)).lastrowid)
        ep2 = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=2, title="第二集", content="x",
            status="completed", created_at=ts, updated_at=ts)).lastrowid)
        sb1 = int(conn.execute(storyboards.insert().values(
            episode_id=ep1, storyboard_number=1, title="镜一", dialogue="你好呀，世界。",
            composed_video_url=composed, tts_audio_url=tts, duration=2,
            created_at=ts, updated_at=ts)).lastrowid)
        conn.execute(storyboards.insert().values(
            episode_id=ep1, storyboard_number=2, title="镜二",
            composed_video_url=composed, duration=None, created_at=ts, updated_at=ts))
        sb3 = int(conn.execute(storyboards.insert().values(
            episode_id=ep2, storyboard_number=1, title="镜三",
            composed_video_url=composed2, duration=1, created_at=ts, updated_at=ts)).lastrowid)
        # 画布尺寸来源：该分镜的 video_generations 里**最后一条有宽高**的
        for width, height, created in ((1920, 1080, "2026-01-01"), (720, 1280, "2026-02-01"),
                                       (None, None, "2026-03-01")):
            conn.execute(video_generations.insert().values(
                drama_id=drama_id, storyboard_id=sb1, status="completed", prompt="p",
                width=width, height=height, created_at=created, updated_at=ts))
        conn.execute(storyboards.insert().values(
            episode_id=ep1, storyboard_number=9, title="无合成视频",
            duration=1, created_at=ts, updated_at=ts))
    with engine.begin() as conn:
        drafts = jd.build_jianying_draft(conn, drama_id, ep1)
    draft = drafts[0]
    check("构建: 单集过滤生效（只 1 个草稿），草稿名 = dramaN_epNN_标题sanitize",
          len(drafts) == 1 and draft["draftName"] == f"drama{drama_id}_ep01_第一集_开场",
          draft["draftName"])
    check("构建: 草稿目录 `<storageRoot>/jianying/<name>.draft/` + media/ 子目录",
          draft["draftDir"].endswith(f"{draft['draftName']}.draft")
          and os.path.isdir(os.path.join(draft["draftDir"], "media")), draft["draftDir"])
    check("构建: 时长累计 = 2s + (duration 缺省 5s)（微秒）",
          draft["durationUs"] == 7_000_000, draft["durationUs"])
    check("构建: 素材复制命名 video_001 / video_002（跳过分镜号 9 那条没有合成视频的）",
          sorted(os.listdir(os.path.join(draft["draftDir"], "media")))
          == ["audio_001.mp3", "video_001.mp4", "video_002.mp4"],
          sorted(os.listdir(os.path.join(draft["draftDir"], "media"))))
    with open(os.path.join(draft["draftDir"], "draft_content.json"), encoding="utf-8") as handle:
        content = json.load(handle)
    check("内容: 三段式齐备（canvas_config/materials/tracks）+ fps=30 + duration=总时长",
          content["fps"] == 30 and content["duration"] == 7_000_000
          and {"canvas_config", "materials", "tracks"} <= set(content))
    check("内容: 画布取**最后一条有宽高**的视频生成记录（720x1280，最后那条无宽高被跳过）",
          content["canvas_config"] == {"width": 720, "height": 1280, "ratio": "original"},
          content["canvas_config"])
    check("内容: 视频 2 条 / 音频 1 条 / 字幕 1 条（只有 sb1 有 dialogue）",
          len(content["materials"]["videos"]) == 2 and len(content["materials"]["audios"]) == 1
          and len(content["materials"]["texts"]) == 1
          and content["materials"]["texts"][0]["content"] == "你好呀，世界。")
    video_ids = {m["id"] for m in content["materials"]["videos"]}
    check("内容: 片段**引用同一批 UUID**（tracks.segments.material_id 必须在 materials 里）",
          all(s["material_id"] in video_ids for s in content["tracks"][0]["segments"])
          and len(content["tracks"]) == 3
          and [t["type"] for t in content["tracks"]] == ["video", "audio", "text"],
          [t["type"] for t in content["tracks"]])
    check("内容: 片段 target_timerange 连续铺开（0..2s、2..7s）",
          [(s["target_timerange"]["start"], s["target_timerange"]["duration"])
           for s in content["tracks"][0]["segments"]] == [(0, 2_000_000), (2_000_000, 5_000_000)],
          [(s["target_timerange"]["start"], s["target_timerange"]["duration"])
           for s in content["tracks"][0]["segments"]])
    with open(os.path.join(draft["draftDir"], "draft_info.json"), encoding="utf-8") as handle:
        info = json.load(handle)
    check("info: draft_name/root_path 正确，project_has_audio 反映音频素材数",
          info["draft_name"] == draft["draftName"] and info["draft_root_path"] == draft["draftDir"]
          and info["project_has_audio"] is True)
    check("文件清单: 2 个 JSON + 3 个素材，zipPath 一律以 `<name>.draft/` 开头",
          len(draft["files"]) == 5
          and all(f["zipPath"].startswith(f"{draft['draftName']}.draft/") for f in draft["files"])
          and all(os.path.exists(f["absPath"]) for f in draft["files"]),
          [f["zipPath"] for f in draft["files"]])
    with engine.begin() as conn:
        all_drafts = jd.build_jianying_draft(conn, drama_id)
    check("构建: 全剧导出（不给 episodeId）→ **每集一个草稿**（此刻 2 集）",
          len(all_drafts) == 2
          and {d["draftName"] for d in all_drafts}
          == {draft["draftName"], f"drama{drama_id}_ep02_第二集"},
          [d["draftName"] for d in all_drafts])

    # 异常路径
    with engine.begin() as conn:
        try:
            jd.build_jianying_draft(conn, 999999)
            check("构建: 没有集 -> 抛 No episodes found", False)
        except ValueError as err:
            check("构建: 没有集 -> 抛 `No episodes found for draft export`",
                  str(err) == "No episodes found for draft export", str(err))
        empty_ep = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=7, title="空集", content="x",
            created_at=ts, updated_at=ts)).lastrowid)
        try:
            jd.build_jianying_draft(conn, drama_id, empty_ep)
            check("构建: 无合成视频 -> 抛错", False)
        except ValueError as err:
            check("构建: 无合成视频 -> 抛 `Episode 7 has no composed videos`",
                  str(err) == "Episode 7 has no composed videos", str(err))

    # ================= 端点（阶段 3）=================
    client = TestClient(app)
    base = "/api/v1/export"
    check("端点: 非法 id / 剧不存在 -> 404",
          client.get(f"{base}/dramas/abc/jianying-draft").status_code == 404
          and client.get(f"{base}/dramas/999999/jianying-draft").status_code == 404)
    check("端点: episodeId 指向**别的剧/不存在的集** -> 400（无集可导）",
          client.get(f"{base}/dramas/{drama_id}/jianying-draft",
                     params={"episodeId": "999999"}).json()
          == {"code": 400, "message": "No episodes found for draft export"})
    resp = client.get(f"{base}/dramas/{drama_id}/jianying-draft", params={"episodeId": str(ep1)})
    check("端点: 成功 -> application/zip + attachment 文件名带 episode + X-Draft-Count=1",
          resp.status_code == 200 and resp.headers["content-type"] == "application/zip"
          and resp.headers["x-draft-count"] == "1"
          and resp.headers["content-disposition"]
          == f'attachment; filename="drama-{drama_id}-episode-{ep1}-jianying-draft.zip"',
          dict(resp.headers))
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        names = archive.namelist()
        payload = archive.read(f"{draft['draftName']}.draft/draft_content.json")
    check("端点: ZIP 内是 `<name>.draft/{draft_content.json,draft_info.json,media/*}`",
          f"{draft['draftName']}.draft/draft_content.json" in names
          and f"{draft['draftName']}.draft/media/video_001.mp4" in names, names)
    check("端点: 草稿 JSON 是**缩进 2 空格**（剪映读的就是这个，有意 indent=2）",
          payload.decode("utf-8").splitlines()[1].startswith("  \""), payload[:60])
    whole = client.get(f"{base}/dramas/{drama_id}/jianying-draft")
    check("端点: 不带 episodeId -> 全剧（X-Draft-Count=2）且文件名不带 episode",
          whole.headers["x-draft-count"] == "2"
          and whole.headers["content-disposition"]
          == f'attachment; filename="drama-{drama_id}-jianying-draft.zip"',
          whole.headers.get("x-draft-count"))

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
