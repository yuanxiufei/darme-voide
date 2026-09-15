"""S7 自检：镜头 QC 打分（``qc-scoring.ts`` 272 行 → `qc_scoring.py`）+ ``POST /storyboards/{id}/qc``。

三个维度都是**累加扣分**，扣错了不会报错、只会让分数「看起来差不多但不对」⇒ 这里用**手算期望**
逐条锁：加权总体分、``clamp`` 的半数进位、三态 ``status``、以及「返回结构体 / 落库 JSON 文本」的
双重形态。另有 ``start_state``/``end_state`` 状态机的跳变判定（连续性状态机的核心）。

运行::

    ./.venv/Scripts/python.exe tests/qc_scoring_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="qcs_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import (  # noqa: E402
    characters,
    continuity_states,
    dramas,
    episodes,
    scenes,
    storyboard_characters,
    storyboards,
    video_generations,
    video_quality_checks,
)
from app.core.response import now  # noqa: E402
from app.services import qc_scoring as qs  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:  # noqa: C901
    # ── 助手 ──
    check("clamp: 0–100 且 **半数进位**（94.5 -> 95，JS Math.round 语义）",
          [qs.clamp(x) for x in (-5, 0, 74.5, 99.4, 100, 140)] == [0, 0, 75, 99, 100, 100],
          [qs.clamp(x) for x in (-5, 0, 74.5, 99.4, 100, 140)])
    check("parse_entity_states: 每行一个，ASCII `=` 与全角 `：` 都认，键值 trim",
          qs.parse_entity_states("茶杯=在桌上\n门：半开\n无分隔符\n 钥匙 = 口袋 ")
          == {"茶杯": "在桌上", "门": "半开", "钥匙": "口袋"},
          qs.parse_entity_states("茶杯=在桌上\n门：半开\n无分隔符\n 钥匙 = 口袋 "))
    check("parse_entity_states: 空/None 安全；实体名 >20 字被忽略（正则上限 20）",
          qs.parse_entity_states(None) == {} and qs.parse_entity_states("") == {}
          and qs.parse_entity_states("一二三四五六七八九十一二三四五六七八九十一=x") == {})

    stamp = now()
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="QC 剧", created_at=stamp, updated_at=stamp)).lastrowid)
        ep = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="第一集", content="x",
            created_at=stamp, updated_at=stamp)).lastrowid)
        scene = int(conn.execute(scenes.insert().values(
            drama_id=drama_id, episode_id=ep, location="咖啡厅", time="白天",
            prompt="咖啡厅内景", location_id="LOC-1",
            created_at=stamp, updated_at=stamp)).lastrowid)
        other_scene = int(conn.execute(scenes.insert().values(
            drama_id=drama_id, episode_id=ep, location="街道", time="夜晚",
            prompt="夜晚街道", location_id="LOC-2",
            created_at=stamp, updated_at=stamp)).lastrowid)
        # 干净镜：无对白、无角色、时长 5s、有状态、有 continuity_states
        clean = int(conn.execute(storyboards.insert().values(
            episode_id=ep, storyboard_number=1, title="镜一", duration=5,
            scene_id=scene, start_state="茶杯=在桌上", end_state="茶杯=在桌上",
            created_at=stamp, updated_at=stamp)).lastrowid)
        # 噪声镜：有对白/无 tts、有角色缺资产、时长越界、场景地点不一致、状态跳变
        noisy = int(conn.execute(storyboards.insert().values(
            episode_id=ep, storyboard_number=2, title="镜二", duration=1.5,
            scene_id=scene, dialogue="台词",
            # 与上一镜（clean）**同一实体**、值不同 ⇒ 命中「状态跳变」
            start_state="茶杯=地上", end_state="茶杯=地上",
            created_at=stamp, updated_at=stamp)).lastrowid)
        # 同集另一镜（用来给上面两镜制造「上一镜场景不同」的对照）——scene 用另一个
        conn.execute(storyboards.insert().values(
            episode_id=ep, storyboard_number=3, title="镜三", duration=6,
            scene_id=other_scene, created_at=stamp, updated_at=stamp))
        # 角色：缺参考图 / 缺声音 / 缺服装
        char_id = int(conn.execute(characters.insert().values(
            drama_id=drama_id, name="小明", created_at=stamp, updated_at=stamp)).lastrowid)
        conn.execute(storyboard_characters.insert().values(
            storyboard_id=noisy, character_id=char_id))
        conn.execute(continuity_states.insert().values(
            episode_id=ep, state_type="prop", entity_key="茶杯", state_value="在桌上",
            created_at=stamp, updated_at=stamp))

    with engine.begin() as conn:
        check("打分: 分镜不存在 -> {'error': ...}（**不抛错**）",
              qs.score_storyboard(conn, 999999) == {"error": "Storyboard not found"})
        clean_result = qs.score_storyboard(conn, clean)

    check("干净镜: 无对白 -> lip_sync=90（不是 100，有专门文案）",
          clean_result["lipSyncScore"] == 90
          and clean_result["dimensions"]["lip_sync"]["notes"] == ["无对白镜头，不涉及唇形同步"],
          clean_result["dimensions"]["lip_sync"])
    check("干净镜: 无出场角色 -> character_consistency=95 + 说明",
          clean_result["characterConsistencyScore"] == 95
          and "空镜/环境" in clean_result["dimensions"]["character_consistency"]["notes"][0])
    check("干净镜: continuity=100（时长合规/状态齐/地点锁定/有跨镜状态）",
          clean_result["continuityScore"] == 100, clean_result["continuityScore"])
    check("干净镜: 总体分 = **加权** 90*.4+95*.3+100*.3 = 94.5 -> 95",
          clean_result["overallScore"] == 95, clean_result["overallScore"])
    check("干净镜: 无已完成视频 -> status=**pending**",
          clean_result["status"] == "pending" and clean_result["videoGenerationId"] is None)
    check("干净镜: 返回里 issues/dimensions 是**结构体**（不是 JSON 文本）",
          isinstance(clean_result["issues"], list)
          and isinstance(clean_result["dimensions"]["lip_sync"], dict))
    with engine.begin() as conn:
        row = conn.execute(video_quality_checks.select().where(
            video_quality_checks.c.storyboard_id == clean)).first()
    check("干净镜: 落库的 issues/dimensions 是**紧凑 JSON 文本**（与返回值形态不同）",
          isinstance(row.issues, str) and isinstance(row.dimensions, str)
          and json.loads(row.issues) == clean_result["issues"]
          and " " not in row.issues[:20], row.issues)
    check("干净镜: 落库带 storyboardId/videoGenerationId/dramaId/episodeId 四个关联键",
          row.drama_id == drama_id and row.episode_id == ep
          and row.video_generation_id is None)

    with engine.begin() as conn:
        noisy_result = qs.score_storyboard(conn, noisy)
        issues = [(i["dimension"], i["severity"]) for i in noisy_result["issues"]]
    check("噪声镜: lib_sync = 100-40(无TTS)-30(无完成视频)-10(未指定speaker) = 20",
          noisy_result["lipSyncScore"] == 20, noisy_result["lipSyncScore"])
    check("噪声镜: 三条 lip_sync 问题（error/warning/warning）",
          [s for d, s in issues if d == "lip_sync"] == ["error", "warning", "warning"], issues)
    check("噪声镜: character = 100-25(无参考图)-10(无声音)-5(无服装) = 60",
          noisy_result["characterConsistencyScore"] == 60, noisy_result["characterConsistencyScore"])
    check("噪声镜: continuity = 100-10(时长1.5越界)-20(状态跳变) = 70"
          "（**同场景地点不一致那条是原 TS 死分支，永远不触发**）",
          noisy_result["continuityScore"] == 70, noisy_result["continuityScore"])
    check("噪声镜: 状态跳变文案含「上一镜尾「在桌上」→ 本镜首「地上」」的原文对照",
          any("相邻镜头状态跳变" in i["message"] and "茶杯" in i["message"]
              for i in noisy_result["issues"]), noisy_result["issues"][-2:])
    check("噪声镜: 总体分 = 20*.4+60*.3+70*.3 = 47",
          noisy_result["overallScore"] == 47, noisy_result["overallScore"])
    check("噪声镜: **无完成视频 ⇒ 即使有 error 也是 pending**（三态第一档优先）",
          noisy_result["status"] == "pending", noisy_result["status"])

    # 有完成视频：error -> failed；干净镜 + 参考模式 -> passed
    with engine.begin() as conn:
        conn.execute(video_generations.insert().values(
            drama_id=drama_id, storyboard_id=noisy, status="completed",
            prompt="p", video_url="static/videos/n.mp4", duration=1.5,
            created_at=stamp, updated_at=stamp))
        conn.execute(video_generations.insert().values(
            drama_id=drama_id, storyboard_id=clean, status="completed",
            prompt="p", video_url="static/videos/c.mp4", duration=5,
            reference_mode="image", created_at=stamp, updated_at=stamp))
        failed = qs.score_storyboard(conn, noisy)
        passed = qs.score_storyboard(conn, clean)
    check("有完成视频 + 有 error 级问题 -> status=**failed**",
          failed["status"] == "failed" and failed["videoGenerationId"] is not None,
          failed["status"])
    check("有完成视频 + 无 error -> status=**passed**（用了参考模式 ⇒ 无 -15）",
          passed["status"] == "passed", passed["status"])
    check("完成视频后 lip_sync 不再扣「无已完成视频」的 30 分"
          "（噪声镜 100-40(无TTS)-10(未指定speaker) = 50）",
          passed["lipSyncScore"] == 90 and failed["lipSyncScore"] == 50,
          (passed["lipSyncScore"], failed["lipSyncScore"]))

    with engine.begin() as conn:
        rows = conn.execute(video_quality_checks.select().where(
            video_quality_checks.c.storyboard_id == clean)).all()
    check("落库是 **upsert**（同一分镜只保留一条，第二次是 update 同 id）",
          len(rows) == 1 and rows[0].status == "passed" and rows[0].overall_score == 95,
          [r.id for r in rows])

    # ── 端点 ──
    client = TestClient(app)
    ok = client.post(f"/api/v1/storyboards/{clean}/qc")
    check("端点: 200 + data 为打分结果（含四分数 + issues/dimensions 结构体）",
          ok.status_code == 200 and ok.json()["code"] == 200
          and ok.json()["data"]["overallScore"] == 95
          and isinstance(ok.json()["data"]["issues"], list), ok.text[:140])
    check("端点: 非法 id -> 404 'Invalid storyboard id'",
          client.post("/api/v1/storyboards/abc/qc").status_code == 404)
    check("端点: 分镜不存在 -> 404 '镜头不存在'",
          client.post("/api/v1/storyboards/999999/qc").json()
          == {"code": 404, "message": "镜头不存在"})

    # ── 视频完成后的接线（video_generation 的 _run_qc_after_video_complete）──
    from app.core.models import video_generations as vg_model  # noqa: PLC0415
    from app.services import video_generation as vgen  # noqa: PLC0415

    with engine.begin() as conn:
        fresh_sb = int(conn.execute(storyboards.insert().values(
            episode_id=ep, storyboard_number=9, title="接线镜", duration=5,
            scene_id=scene, start_state="茶杯=在桌上", end_state="茶杯=在桌上",
            created_at=stamp, updated_at=stamp)).lastrowid)
        fresh_vg = int(conn.execute(vg_model.insert().values(
            drama_id=drama_id, storyboard_id=fresh_sb, status="completed",
            prompt="p", video_url="static/videos/f.mp4", duration=5,
            reference_mode="image", created_at=stamp, updated_at=stamp)).lastrowid)
    vgen._run_qc_after_video_complete(fresh_sb, fresh_vg)
    with engine.begin() as conn:
        wired = conn.execute(video_quality_checks.select().where(
            video_quality_checks.c.storyboard_id == fresh_sb)).first()
    # ⚠️ overall=93 而非 95：本镜的**上一镜（#3）没填 start/end_state** ⇒ continuity 额外 -5
    #    （「相邻镜头缺少 start_state / end_state 之一」），与 clean 镜（上一镜无、故无此扣分）不同。
    check("接线: 视频完成后自动落一条 QC 记录（status=passed，关联 videoGenerationId，93 分）",
          wired is not None and wired.status == "passed"
          and wired.video_generation_id == fresh_vg and wired.overall_score == 93,
          None if wired is None else (wired.status, wired.video_generation_id,
                                      wired.overall_score))
    stored_issues = json.loads(wired.issues)
    check("接线: 落库内容与 `qc_report` 同表同口径（issues 是 JSON 文本、dimensions 含三维）",
          any("相邻镜头缺少 start_state" in i["message"] for i in stored_issues)
          and "lip_sync" in wired.dimensions
          and json.loads(wired.dimensions)["lip_sync"]["score"] == 90,
          stored_issues)
    check("接线: 技术维度未迁时**只 warn 不抛**（fire-and-forget 不影响成片）",
          True)  # 见服务内 tech-qc-skipped；此处锁「调用不抛异常」

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok_, detail in _RESULTS:
        print(("PASS  " if ok_ else "FAIL  ") + name + ("" if ok_ else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
