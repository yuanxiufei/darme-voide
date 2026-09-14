"""S7 自检：审片重跑闭环（``qc-retry.ts`` 175 行 → `qc_retry.py`）+ ``POST /storyboards/{id}/retry-qc``。

三条最容易错的语义：**首帧等待是「值变了才算」**（超时上限 5 分钟，自检调 0）、
**参考图/参考音频的开关条件不同**（图：对白 + 三个 provider 之一；音：对白 + 仅 minimax）、
**FL2VA 判定只看设计首帧 + `last_frame_image`**（真实尾帧 `tail_frame_image` 不参与）。

⚠️ 三个重依赖（agent / 生图 / 生视频）全部**打桩**（模块级替换）⇒ 不打网络、不起任务。

运行::

    ./.venv/Scripts/python.exe tests/qc_retry_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="qcr_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import (  # noqa: E402
    dramas,
    episodes,
    image_generations,
    storyboards,
    video_generations,
    video_quality_checks,
)
from app.response import now  # noqa: E402
from app.services import qc_retry as qr  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SEEN: dict[str, object] = {}


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:  # noqa: C901
    qr.FIRST_FRAME_WAIT_TIMEOUT_MS = 0  # 首帧轮询直接超时（不阻塞），只验「超时有 warn、流程继续」

    class _AgentResult:
        tool_calls = [{"toolName": "update_storyboard"}, {"toolName": "noop"}]

    async def fake_agent(conn, agent_type, episode_id, drama_id, message, options=None):
        _SEEN["agent"] = {"type": agent_type, "episodeId": episode_id, "dramaId": drama_id,
                          "message": message, "options": options}
        # 模拟 LLM 只改了 video_prompt（顺带改 storyboard 行，供 rewrittenFields 比对）。
        # ⚠️ 只在第一个 case 改：``_SEEN["mutate"]`` 为 False 时模拟「LLM 什么都没改」。
        if _SEEN.get("mutate", True):
            conn.execute(storyboards.update().where(storyboards.c.id == _SEEN["sb_id"])
                         .values(video_prompt="修复后的提示词"))
        return _AgentResult()

    async def fake_image(conn, params):
        _SEEN["image"] = params
        return 101

    async def fake_video(conn, params):
        _SEEN["video"] = params
        return 202

    qr.run_agent_with_retry = fake_agent  # type: ignore[assignment]
    qr.generate_image = fake_image  # type: ignore[assignment]
    qr.generate_video = fake_video  # type: ignore[assignment]

    stamp = now()
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="重跑剧", created_at=stamp, updated_at=stamp)).lastrowid)
        ep = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="第一集", content="x",
            created_at=stamp, updated_at=stamp)).lastrowid)
        sb = int(conn.execute(storyboards.insert().values(
            episode_id=ep, storyboard_number=3, title="问题镜", duration=5,
            video_prompt="旧提示词", first_frame_image="static/frames/old.jpg",
            last_frame_image="static/frames/design-last.jpg",
            tail_frame_image="static/frames/real-tail.jpg",  # ⚠️ 不参与 FL2VA 判定
            created_at=stamp, updated_at=stamp)).lastrowid)
        _SEEN["sb_id"] = sb
        conn.execute(video_quality_checks.insert().values(
            storyboard_id=sb, status="failed", overall_score=40,
            issues=json.dumps([{"severity": "error", "dimension": "continuity",
                                "message": "相邻镜头状态跳变"}]),
            dimensions="{}", created_at=stamp, updated_at=stamp))
        conn.execute(image_generations.insert().values(
            drama_id=drama_id, storyboard_id=sb, status="completed", prompt="p",
            created_at=stamp, updated_at=stamp))
        conn.execute(video_generations.insert().values(
            drama_id=drama_id, storyboard_id=sb, status="completed", prompt="p",
            video_url="static/videos/old.mp4", created_at=stamp, updated_at=stamp))

    async def run():
        with engine.begin() as conn:
            return await qr.retry_failed_storyboard(conn, sb)

    result = asyncio.run(run())

    check("闭环: 返回 ok + 消息含镜头号/新任务 id + rewrittenFields",
          result["ok"] is True and "#3" in result["message"]
          and result["imageGenerationId"] == 101 and result["videoGenerationId"] == 202
          and result["rewrittenFields"] == ["videoPrompt"]
          and result["storyboardNumber"] == 3, result)
    check("闭环: 交给 agent 的是 **storyboard_breaker** + maxSteps 20 + 该剧集 id",
          _SEEN["agent"]["type"] == "storyboard_breaker"
          and _SEEN["agent"]["options"] == {"maxSteps": 20}
          and _SEEN["agent"]["episodeId"] == ep, _SEEN["agent"]["options"])
    check("闭环: 提示词含 QC issues 原文 + 「禁止 save_storyboards 重建整集」",
          "[error] continuity: 相邻镜头状态跳变" in str(_SEEN["agent"]["message"])
          and "禁止调用 save_storyboards 重建整集" in str(_SEEN["agent"]["message"]))
    check("闭环: 生图参数（frameType=first_frame、带 dramaId/configId）"
          "＋ agent 只改了 video_prompt ⇒ prompt 走**兜底文案**（first_frame_prompt 仍为空）",
          _SEEN["image"]["frameType"] == "first_frame"
          and _SEEN["image"]["storyboardId"] == sb
          and _SEEN["image"]["dramaId"] == drama_id
          and _SEEN["image"]["prompt"] == "短剧分镜画面，电影感构图", _SEEN["image"]["prompt"])
    check("闭环: **FL2VA**（设计首帧 + last_frame_image 都在）⇒ referenceMode=first_last"
          "（真实尾帧 tail 不参与）",
          _SEEN["video"]["referenceMode"] == "first_last"
          and _SEEN["video"]["firstFrameUrl"] == "static/frames/old.jpg"
          and _SEEN["video"]["lastFrameUrl"] == "static/frames/design-last.jpg"
          and "imageUrl" not in _SEEN["video"], _SEEN["video"])
    check("闭环: 视频用新 prompt（agent 改过的）",
          _SEEN["video"]["prompt"] == "修复后的提示词", _SEEN["video"]["prompt"])

    with engine.begin() as conn:
        imgs = conn.execute(select(image_generations).where(
            image_generations.c.storyboard_id == sb)).all()
        vids = conn.execute(select(video_generations).where(
            video_generations.c.storyboard_id == sb)).all()
    check("闭环: **软删旧产物**（图片/视频记录的 deleted_at 都写上，不是物理删）",
          all(r.deleted_at is not None for r in imgs)
          and all(r.deleted_at is not None for r in vids)
          and len(imgs) == 1 and len(vids) == 1, [(r.deleted_at, r.updated_at) for r in imgs])

    # 分支：非对白场景 + 非多参考 provider ⇒ 不带参考图/音频；仅首帧 ⇒ single
    with engine.begin() as conn:
        sb2 = int(conn.execute(storyboards.insert().values(
            episode_id=ep, storyboard_number=4, title="普通镜", duration=4,
            scene_type="landscape", first_frame_image="static/frames/f2.jpg",
            created_at=stamp, updated_at=stamp)).lastrowid)
        conn.execute(video_quality_checks.insert().values(
            storyboard_id=sb2, status="failed", overall_score=30, issues="",
            dimensions="{}", created_at=stamp, updated_at=stamp))
    _SEEN.clear()
    _SEEN["sb_id"] = sb2
    _SEEN["mutate"] = False  # 这个分支专门验「LLM 什么都没改」

    async def run_sb2():
        with engine.begin() as conn:
            return await qr.retry_failed_storyboard(conn, sb2)
    result2 = asyncio.run(run_sb2())
    check("分支: 无 QC issues 时用固定兜底文案（不是空）",
          "（无具体问题描述，请复核画面与对白质量后自行修复明显缺陷）"
          in str(_SEEN["agent"]["message"]))
    check("分支: 非对白场景 + 无 last_frame ⇒ referenceMode=**single** 且带 imageUrl",
          _SEEN["video"]["referenceMode"] == "single"
          and _SEEN["video"]["imageUrl"] == "static/frames/f2.jpg"
          and "referenceImageUrls" not in _SEEN["video"]
          and "referenceAudioUrls" not in _SEEN["video"], _SEEN["video"])
    check("分支: 没改任何字段 ⇒ rewrittenFields=[] 且消息里写「无」",
          result2["rewrittenFields"] == [] and "修改字段：无" in result2["message"],
          result2["message"])

    # 分支：镜头不存在 / 已软删
    async def run_missing():
        with engine.begin() as conn:
            return await qr.retry_failed_storyboard(conn, 999999)
    check("分支: 镜头不存在 -> ok:false + 'Storyboard not found'",
          asyncio.run(run_missing()) == {"ok": False, "message": "Storyboard not found",
                                         "storyboardId": 999999})

    # ── 端点 ──
    client = TestClient(app)
    _SEEN.clear()
    _SEEN["sb_id"] = sb
    ok = client.post(f"/api/v1/storyboards/{sb}/retry-qc")
    check("端点: 200 + data 为闭环结果（ok/新任务 id）",
          ok.status_code == 200 and ok.json()["code"] == 200
          and ok.json()["data"]["ok"] is True
          and ok.json()["data"]["videoGenerationId"] == 202, ok.text[:160])
    check("端点: 非法 id -> 404 'Invalid storyboard id'",
          client.post("/api/v1/storyboards/abc/retry-qc").status_code == 404)
    check("端点: 镜头不存在 -> 404 '镜头不存在'",
          client.post("/api/v1/storyboards/999999/retry-qc").json()
          == {"code": 404, "message": "镜头不存在"})

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok_, detail in _RESULTS:
        print(("PASS  " if ok_ else "FAIL  ") + name + ("" if ok_ else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
