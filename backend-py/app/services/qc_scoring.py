"""镜头级 QC 打分 —— 移植 ``services/qc-scoring.ts``（272 行）。

三个维度（**启发式规则 + 元数据一致性**，不是 AI 检测模型 —— 真唇形/相似度留给后续挂载）：

* ``lip_sync``：对白配音（``tts_audio_url``）/ speaker 声音锁定 / 视频时长偏差；
* ``character_consistency``：出场角色参考图 / 声音 / 服装锁定（**逐角色扣分**）；
* ``continuity``：相邻分镜场景连贯、地点 ID 一致、``start_state``/``end_state`` 状态机衔接、时长合理。

结果**落 ``video_quality_checks``**：同一分镜只保留一条（按 ``id`` 取最新那条来 update，没有才 insert）。

⚠️ 五处保真点：

1. **扣分是累加的、没有下限保护** ⇒ 最后统一 ``clamp`` 到 ``[0,100]`` 且 ``Math.round``（**半数进位**，用 ``js_round``）；
2. **总体分是加权** ``lip*0.4 + character*0.3 + continuity*0.3``，再 clamp —— **不是**各维平均；
3. ``status`` 三态：**有已完成视频**才 ``passed``/``failed``（有 ``error`` 级 issue 即 failed），否则 **``pending``**；
4. 落库的 ``issues`` / ``dimensions`` 是 **JSON 文本**（紧凑），而**返回值**里这两项被换成**结构体**
   （``{...saved, issues, dimensions: dims}``）—— 同一字段两种形态，别搞混；
5. 分镜不存在时**不抛错**，返回 ``{"error": "Storyboard not found"}``（路由层已先查过存在性）。

⚠️ ``characters.speaker_id`` 这个字段在本仓确实存在（``speaker_id`` 是角色表上的「声音绑定」），
所以「speaker 未锁定声音」这条分支是**活的**，别当成死代码删掉。
"""
from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.engine import Connection

from ..models import (
    characters,
    continuity_states,
    episodes,
    scenes,
    storyboard_characters,
    storyboards,
    video_generations,
    video_quality_checks,
)
from ..response import js_round, now
from .task_logger import log_task_success, log_task_warn
from .technical_qc import run_technical_qc

__all__ = ["parse_entity_states", "run_qc_after_video_complete", "score_storyboard"]

#: 「实体=状态」行（ASCII ``=`` 与全角 ``：`` 都算分隔符；实体名 1–20 字，惰性匹配）
_ENTITY_STATE_RE = re.compile(r"^\s*([^=：]{1,20}?)\s*[=：]\s*(.+)$")

#: 三个维度的权重（总体分 = 加权和，再 clamp）
_WEIGHTS = {"lip_sync": 0.4, "character_consistency": 0.3, "continuity": 0.3}


def clamp(value: float) -> int:
    """``Math.max(0, Math.min(100, Math.round(n)))`` —— 半数进位，与 JS 一致。"""
    return max(0, min(100, js_round(value)))


def parse_entity_states(state_text: Any) -> dict[str, str]:
    """解析「实体=状态」文本（每行一个），返回 ``{实体: 状态}``。"""
    result: dict[str, str] = {}
    if not state_text:
        return result
    for line in str(state_text).split("\n"):
        match = _ENTITY_STATE_RE.match(line.strip())
        if match:
            result[match.group(1).strip()] = match.group(2).strip()
    return result


def _json_compact(value: Any) -> str:
    """落库用的紧凑 JSON（对齐 ``JSON.stringify``）。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def score_storyboard(conn: Connection, storyboard_id: Any) -> dict[str, Any]:
    """给一个分镜打分并落库（返回结构里 ``issues``/``dimensions`` 是**结构体**）。"""
    sb = conn.execute(
        select(storyboards).where(storyboards.c.id == storyboard_id)
    ).first()
    if sb is None:
        return {"error": "Storyboard not found"}

    issues: list[dict[str, str]] = []
    dims: dict[str, dict[str, Any]] = {
        "lip_sync": {"score": 0, "checked": True, "notes": []},
        "character_consistency": {"score": 0, "checked": True, "notes": []},
        "continuity": {"score": 0, "checked": True, "notes": []},
    }

    links = conn.execute(
        select(storyboard_characters)
        .where(storyboard_characters.c.storyboard_id == storyboard_id)
    ).all()
    chars = [
        row for row in (
            conn.execute(
                select(characters).where(and_(characters.c.id == link.character_id,
                                              characters.c.deleted_at.is_(None)))
            ).first()
            for link in links
        ) if row is not None
    ]

    scene = (conn.execute(select(scenes).where(scenes.c.id == sb.scene_id)).first()
             if sb.scene_id else None)
    episode = (conn.execute(select(episodes).where(episodes.c.id == sb.episode_id)).first()
               if sb.episode_id else None)

    gens = conn.execute(
        select(video_generations).where(and_(
            video_generations.c.storyboard_id == storyboard_id,
            video_generations.c.deleted_at.is_(None)))
    ).all()
    completed_gen = next(
        (g for g in gens if g.status == "completed" and bool(g.video_url)), None)
    has_dialogue = bool(sb.dialogue) and len(str(sb.dialogue).strip()) > 0

    # ===== 1. 唇形同步 / 音画同步 =====
    if not has_dialogue:
        dims["lip_sync"]["score"] = 90
        dims["lip_sync"]["notes"].append("无对白镜头，不涉及唇形同步")
    else:
        score = 100
        if not sb.tts_audio_url:
            score -= 40
            issues.append({"dimension": "lip_sync", "severity": "error",
                           "message": "有对白但未生成 TTS 音频（tts_audio_url 为空）"})
        if completed_gen is None:
            score -= 30
            issues.append({"dimension": "lip_sync", "severity": "warning",
                           "message": "有对白但无已完成的视频生成记录"})
        if sb.speaker_id:
            speaker_char = next((c for c in chars if c.speaker_id == sb.speaker_id), None)
            if speaker_char is not None and not speaker_char.voice_style:
                score -= 15
                issues.append({"dimension": "lip_sync", "severity": "warning",
                               "message": f"speaker {sb.speaker_id} 未锁定声音资产"
                                          f"（voice_style 为空）"})
        elif chars:
            score -= 10
            issues.append({"dimension": "lip_sync", "severity": "warning",
                           "message": "有对白但分镜未指定 speaker_id"})
        if completed_gen is not None and completed_gen.duration and sb.duration:
            drift = abs(completed_gen.duration - sb.duration) / sb.duration
            if drift > 0.3:
                score -= 20
                issues.append({"dimension": "lip_sync", "severity": "warning",
                               "message": f"视频时长 {completed_gen.duration}s 与分镜 "
                                          f"{sb.duration}s 偏差 {js_round(drift * 100)}%"})
        dims["lip_sync"]["score"] = clamp(score)
        dims["lip_sync"]["notes"].append(f"对白{'已' if sb.tts_audio_url else '未'}配音")

    # ===== 2. 角色一致性 =====
    if not chars:
        dims["character_consistency"]["score"] = 95
        dims["character_consistency"]["notes"].append("无出场角色（空镜/环境），不涉及角色一致性")
    else:
        score = 100
        for char in chars:
            if not (char.image_url or char.reference_images):
                score -= 25
                issues.append({"dimension": "character_consistency", "severity": "error",
                               "message": f"角色「{char.name}」无参考图"
                                          f"（image_url / reference_images 为空）"})
            if not char.voice_style:
                score -= 10
                issues.append({"dimension": "character_consistency", "severity": "warning",
                               "message": f"角色「{char.name}」未锁定声音（voice_style 为空）"})
            if not char.costume_id:
                score -= 5
                issues.append({"dimension": "character_consistency", "severity": "info",
                               "message": f"角色「{char.name}」未锁定服装（costume_id 为空）"})
        if (completed_gen is not None and not completed_gen.reference_mode
                and not completed_gen.image_url and not completed_gen.reference_image_urls):
            score -= 15
            issues.append({"dimension": "character_consistency", "severity": "warning",
                           "message": "视频生成未使用参考图（无 reference / image）"})
        dims["character_consistency"]["score"] = clamp(score)
        dims["character_consistency"]["notes"].append(
            f"出场角色 {'、'.join(str(c.name) for c in chars)}")

    # ===== 3. 连续性 =====
    score = 100
    if sb.duration and (sb.duration < 2 or sb.duration > 30):
        score -= 10
        issues.append({"dimension": "continuity", "severity": "warning",
                       "message": f"分镜时长 {sb.duration}s 超出合理范围（2–30s）"})

    siblings = sorted(
        conn.execute(
            select(storyboards).where(and_(storyboards.c.episode_id == sb.episode_id,
                                           storyboards.c.deleted_at.is_(None)))
        ).all(),
        key=lambda s: s.storyboard_number,
    )
    index = next((i for i, s in enumerate(siblings) if s.id == sb.id), -1)
    if index > 0:
        prev = siblings[index - 1]
        # ⚠️ **原 TS 的死分支（照抄，别"修"）**：这里要求 ``prev.scene_id == sb.scene_id``，
        #    而 ``prev_scene`` 又是按 ``prev.scene_id`` 查出来的 ⇒ 两者**同一个场景行**，
        #    其 ``location_id`` 必然相同 ⇒ 下面那个「地点 ID 不一致 -20」**永远不会触发**。
        #    （想真正校验地点一致性，得比较不同场景行，属原实现的意图偏差；保留原行为以便对拍。）
        if sb.scene_id and prev.scene_id == sb.scene_id:
            prev_scene = conn.execute(
                select(scenes).where(scenes.c.id == prev.scene_id)).first()
            if (prev_scene is not None and scene is not None
                    and prev_scene.location_id and scene.location_id
                    and prev_scene.location_id != scene.location_id):
                score -= 20
                issues.append({"dimension": "continuity", "severity": "error",
                               "message": f"同一场景 #{sb.scene_id} 但地点 ID 不一致"
                                          f"（{prev_scene.location_id} vs {scene.location_id}）"})
        # 状态衔接：上一镜 end_state 与本镜 start_state 的公共实体必须一致
        if prev.end_state and sb.start_state:
            prev_end = parse_entity_states(prev.end_state)
            cur_start = parse_entity_states(sb.start_state)
            for entity, value in prev_end.items():
                if entity in cur_start and cur_start[entity] != value:
                    score -= 20
                    issues.append({
                        "dimension": "continuity", "severity": "error",
                        "message": f"相邻镜头状态跳变：{entity} 上一镜尾「{value}」→ "
                                   f"本镜首「{cur_start[entity]}」（应严格衔接或写明跳切理由）"})
        elif sb.start_state or prev.end_state:
            score -= 5
            issues.append({"dimension": "continuity", "severity": "info",
                           "message": "相邻镜头缺少 start_state / end_state 之一，"
                                      "无法校验状态衔接"})
    if not sb.start_state or not sb.end_state:
        score -= 10
        issues.append({"dimension": "continuity", "severity": "warning",
                       "message": "本镜未填写 start_state / end_state"
                                  "（连续性状态机要求每镜填写）"})
    if sb.scene_id and scene is not None and not scene.location_id:
        score -= 5
        issues.append({"dimension": "continuity", "severity": "info",
                       "message": "关联场景未锁定地点 ID（location_id 为空）"})
    continuity_count = len(conn.execute(
        select(continuity_states).where(continuity_states.c.episode_id == sb.episode_id)
    ).all())
    if continuity_count == 0:
        score -= 5
        issues.append({"dimension": "continuity", "severity": "info",
                       "message": "本集未保存跨镜连续性状态（continuity_states 为空），"
                                  "建议 storyboard_breaker 调用 save_continuity_states"})
    dims["continuity"]["score"] = clamp(score)

    overall = clamp(sum(dims[key]["score"] * weight for key, weight in _WEIGHTS.items()))
    video_generation_id = completed_gen.id if completed_gen is not None else None
    if completed_gen is not None:
        status = "failed" if any(i["severity"] == "error" for i in issues) else "passed"
    else:
        status = "pending"

    base = {
        "storyboardId": storyboard_id,
        "videoGenerationId": video_generation_id,
        "dramaId": (episode.drama_id if episode is not None else None),
        "episodeId": sb.episode_id,
        "lipSyncScore": dims["lip_sync"]["score"],
        "characterConsistencyScore": dims["character_consistency"]["score"],
        "continuityScore": dims["continuity"]["score"],
        "overallScore": overall,
        "issues": _json_compact(issues),
        "dimensions": _json_compact(dims),
        "status": status,
    }

    ts = now()
    existing = conn.execute(
        select(video_quality_checks)
        .where(video_quality_checks.c.storyboard_id == storyboard_id)
    ).all()
    existing = sorted(existing, key=lambda r: r.id, reverse=True)[0] if existing else None

    if existing is not None:
        conn.execute(
            video_quality_checks.update()
            .where(video_quality_checks.c.id == existing.id)
            .values(
                storyboard_id=storyboard_id,
                video_generation_id=video_generation_id,
                drama_id=base["dramaId"],
                episode_id=sb.episode_id,
                lip_sync_score=base["lipSyncScore"],
                character_consistency_score=base["characterConsistencyScore"],
                continuity_score=base["continuityScore"],
                overall_score=overall,
                issues=base["issues"],
                dimensions=base["dimensions"],
                status=status,
                updated_at=ts,
            )
        )
        saved = {"id": existing.id, "createdAt": existing.created_at, "updatedAt": ts}
    else:
        result = conn.execute(video_quality_checks.insert().values(
            storyboard_id=storyboard_id,
            video_generation_id=video_generation_id,
            drama_id=base["dramaId"],
            episode_id=sb.episode_id,
            lip_sync_score=base["lipSyncScore"],
            character_consistency_score=base["characterConsistencyScore"],
            continuity_score=base["continuityScore"],
            overall_score=overall,
            issues=base["issues"],
            dimensions=base["dimensions"],
            status=status,
            created_at=ts,
            updated_at=ts,
        ))
        saved = {"id": int(result.lastrowid or 0), "createdAt": ts, "updatedAt": ts}

    log_task_success("QcScoring", "score-complete", {
        "storyboardId": storyboard_id, "overall": overall, "status": status,
        "issueCount": len(issues),
    })
    # ⚠️ 返回里 issues/dimensions 是**结构体**（覆盖掉落库用的 JSON 文本），与原 TS 一致
    return {**saved, **base, "issues": issues, "dimensions": dims}


def run_qc_after_video_complete(conn: Connection, storyboard_id: Any,
                                video_generation_id: Any) -> None:
    """视频生成完成后自动触发 QC（**失败只 warn，不向上抛**）。

    与 TS 一致：**规则打分**先跑（失败 warn ``auto-score-failed``），随后**独立**跑**技术维度**
    （``technical_qc``：黑场/冻帧/响度/帧率/时长），失败 warn ``tech-qc-failed`` —— 两者互不影响。

    ⚠️ 与 TS 的差异：Node 的 ``runTechnicalQc`` 是 **fire-and-forget**（``.catch()`` 不 await），
    Python 侧**同事务等待完成** —— 写回必须用调用方的连接，detached task 极可能跑在连接提交/关闭
    之后（已写成 ``technical_qc`` 模块 docstring 里的第一条差异）。
    """
    try:
        score_storyboard(conn, storyboard_id)
    except Exception as err:  # noqa: BLE001
        log_task_warn("QcScoring", "auto-score-failed", {
            "storyboardId": storyboard_id, "videoGenerationId": video_generation_id,
            "error": str(err),
        })
    # 技术维度 QC（黑场/冻帧/响度/帧率/时长硬性数值），独立一步 —— 失败不影响上面的规则分
    try:
        run_technical_qc(conn, storyboard_id, video_generation_id)
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 `.catch()` 等价
        log_task_warn("QcScoring", "tech-qc-failed", {
            "storyboardId": storyboard_id, "videoGenerationId": video_generation_id,
            "error": str(err),
        })
