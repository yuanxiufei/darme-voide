"""storyboards 域 —— 与 ``backend/src/routes/storyboards.ts`` 对齐。

该文件在 Node 里有 **14 个端点**，但可迁的纯 DB 逻辑只有 **5 个**（其余全依赖
TTS / 出图 / ffmpeg / LLM / 视觉 QC）：

======  ==============================  ==========================================
方法    路径                            说明
======  ==============================  ==========================================
POST    ``/``                          新建（校验 scene/character 归属当前集）
PUT     ``/{id}``                      更新（33 字段 + 关联同步 + 台词匹配校验）
GET     ``/{id}/validate-dialogue``    台词角色匹配 + TTS 音色过期检测
DELETE  ``/{id}``                      硬删（连带 storyboard_characters）
GET     ``/{id}/qc``                   最近一次 QC 打分
======  ==============================  ==========================================

**已整域迁移（14/14）** —— 2026-09-15 校正：此处原先列着 9 个「未迁移」端点
（``generate-tts`` / ``regenerate-image`` / ``regenerate-frame`` / ``set-frame`` /
``action-suggestion`` / ``split`` / ``optimize-prompt`` / ``qc`` / ``retry-qc``），
它们**后来全部迁完**（出图 / 出音 / ffmpeg 抽帧 / LLM 链路都已落地）。
⚠️ 「哪些没迁」**以 ``tests/route_parity_test.py`` 的机械扫描为准**，别再信任文件头的手写清单
（本仓多处手写清单已过期 —— 已在清理）。

⚠️ 本域的三处注意点：

1. ``POST /`` 返回 **snake_case**（原 TS 显式 ``toSnakeCase``），且是 ``created``（HTTP 201）；
   而 ``GET /{id}/qc`` 直接返回 drizzle 行 ⇒ **camelCase**。同文件两种形状并存，照抄。
2. ``PUT`` 的 ``character_costumes`` 来自 JSON，**键是字符串**。JS 里
   ``costumes[characterId]`` 会把数字键强制转成字符串所以能命中；Python 必须显式
   把键转回 int，否则 ``3 in {"3": ...}`` 恒为 False ⇒ 服装变体静默丢失。
3. ``DELETE`` **只清 storyboard_characters，不清 storyboard_props** —— 原 TS 就是如此
   （会留下 props 关联孤儿行）。照抄不改，但要知情。
"""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import and_, delete, select, update
from sqlalchemy.engine import Connection

from ..db import get_conn, get_tx
from ..models import (
    characters,
    dramas,
    episodes,
    scenes,
    storyboard_characters,
    storyboards,
    video_quality_checks,
)
from ..request_utils import read_json
from ..response import (
    bad_request,
    created,
    js_round,
    not_found,
    now,
    parse_param_id,
    row_to_camel,
    row_to_dict,
    success,
)
from ..services.character_match import match_character_by_speaker_name
from ..services.storyboard_helpers import (
    get_storyboard_character_costumes,
    get_storyboard_character_ids,
    get_storyboard_prop_ids,
    parse_dialogue_for_tts,
    parse_dialogue_lines,
    sync_storyboard_characters,
    sync_storyboard_props,
    validate_dialogue_lines,
    validate_storyboard_bindings,
    validate_tts_speaker,
)
from ..services.frame_extractor import extract_frame
from ..services.image_generation import generate_image
from ..services.qc_retry import retry_failed_storyboard
from ..services.qc_scoring import score_storyboard
from ..services.prompt_utils import (
    build_storyboard_image_prompt,
    build_storyboard_negative_prompt,
    get_storyboard_character_appearances,
    get_storyboard_reference_images,
    get_storyboard_scene_description,
    resolve_effective_art_style,
)
from ..services.task_logger import (
    log_task_error,
    log_task_payload,
    log_task_start,
    log_task_success,
)
from ..services.text_generation import (
    generate_action_suggestion,
    optimize_video_prompt,
    split_shot_into_sub_shots,
)
from ..services.tts_generation import generate_tts

router = APIRouter(prefix="/api/v1/storyboards", tags=["storyboards"])

#: PUT 白名单（请求体键 = DB 列名）。
#: TS 里写成 `{ snakeKey: camelKey }` 映射，但 storyboards 表的 drizzle 属性名
#: 恰好都等于 camelCase(列名)（`tmp/schema-pairs.mjs` 实测 0 处例外），故这里只需列名清单。
_UPDATE_COLUMNS = [
    "title", "description", "shot_type", "angle", "movement", "action",
    "dialogue", "duration", "video_prompt", "image_prompt", "scene_id", "location",
    "time", "atmosphere", "result", "bgm_prompt", "sound_effect",
    "custom_image_prompt", "custom_video_prompt", "negative_prompt",
    "first_frame_prompt", "last_frame_prompt",
    "transition_type", "transition_duration", "transition_motive",
    "first_frame_image", "last_frame_image",
    "keyframe_prompt", "keyframe_image", "asset_status",
    "start_state", "end_state", "constraints",
]


def _fetch_storyboard(conn: Connection, storyboard_id: int | float):
    return conn.execute(
        select(storyboards).where(storyboards.c.id == storyboard_id)
    ).first()


def _normalize_costumes(raw: Any) -> dict[Any, str] | None:
    """把 JSON 对象里字符串化的数字键转回 int（对齐 JS 的数字键强制转换）。"""
    if not isinstance(raw, dict):
        return None
    out: dict[Any, str] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# POST / — 新建
# ---------------------------------------------------------------------------

@router.post("")
async def create_storyboard(request: Request, conn: Connection = Depends(get_tx)):
    try:
        body = await read_json(request)
        ts = now()
        # 先校验归属（scene_id / character_ids 必须来自当前集），失败即 400
        validate_storyboard_bindings(
            conn, body.get("episode_id"), body.get("scene_id"), body.get("character_ids")
        )

        result = conn.execute(
            storyboards.insert().values(
                episode_id=body.get("episode_id"),
                storyboard_number=body.get("storyboard_number") or 1,
                title=body.get("title"),
                description=body.get("description"),
                action=body.get("action"),
                dialogue=body.get("dialogue"),
                scene_id=body.get("scene_id"),
                duration=body.get("duration") or 10,
                created_at=ts,
                updated_at=ts,
            )
        )
        sb_id = int(result.inserted_primary_key[0])
        sync_storyboard_characters(conn, sb_id, body.get("character_ids") or [])
        if "prop_ids" in body:
            sync_storyboard_props(conn, sb_id, body.get("prop_ids") or [])

        row = _fetch_storyboard(conn, sb_id)
        payload = row_to_dict(row) if row is not None else {}
        # 原 TS：created(c, { ...toSnakeCase(result), character_ids, prop_ids }) ⇒ 全 snake_case
        payload["character_ids"] = get_storyboard_character_ids(conn, sb_id)
        payload["prop_ids"] = get_storyboard_prop_ids(conn, sb_id)
        return created(payload)
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc) or "Failed to create storyboard")


# ---------------------------------------------------------------------------
# PUT /{id} — 更新
# ---------------------------------------------------------------------------

@router.put("/{storyboard_id}")
async def update_storyboard(storyboard_id: str, request: Request, conn: Connection = Depends(get_tx)):
    try:
        sid = parse_param_id(storyboard_id)
        if sid is None:
            return not_found("Invalid storyboard id")

        row = _fetch_storyboard(conn, sid)
        if row is None:
            return not_found("镜头不存在")

        body = await read_json(request)
        updates: dict[str, Any] = {"updated_at": now()}
        for column in _UPDATE_COLUMNS:
            if column in body:
                updates[column] = body[column]

        # 台词变更 ⇒ 旧的 TTS 音频与字幕作废
        if "dialogue" in body:
            updates["tts_audio_url"] = None
            updates["subtitle_url"] = None

        validate_storyboard_bindings(
            conn,
            row.episode_id,
            body["scene_id"] if "scene_id" in body else row.scene_id,
            body["character_ids"] if "character_ids" in body else get_storyboard_character_ids(conn, sid),
        )

        conn.execute(update(storyboards).where(storyboards.c.id == sid).values(**updates))

        if "character_ids" in body or "character_costumes" in body:
            if "character_ids" in body:
                current_ids = body.get("character_ids") or []
            else:
                current_ids = get_storyboard_character_ids(conn, sid)
            # TS 用 `??`（空值合并）：显式传 {} 会被采用，只有 undefined/null 才回落查库
            if body.get("character_costumes") is not None:
                costumes = _normalize_costumes(body.get("character_costumes")) or {}
            else:
                costumes = get_storyboard_character_costumes(conn, sid)
            sync_storyboard_characters(conn, sid, current_ids, costumes)

        if "prop_ids" in body:
            prop_ids = body.get("prop_ids")
            sync_storyboard_props(conn, sid, prop_ids if isinstance(prop_ids, list) else [])

        # 对话角色名验证：保存时检测 dialogue 中的角色名是否存在于剧组角色列表
        dialogue_validation: list[dict[str, Any]] | None = None
        if "dialogue" in body and body.get("dialogue"):
            ep = conn.execute(
                select(episodes.c.drama_id).where(episodes.c.id == row.episode_id)
            ).first()
            lines = parse_dialogue_lines(body["dialogue"])
            if not lines:
                parsed = parse_dialogue_for_tts(body["dialogue"])
                if not parsed["ignorable"] and parsed["speaker"]:
                    lines.append({"speaker": parsed["speaker"], "text": parsed["pureText"]})
            if lines:
                dialogue_validation = validate_dialogue_lines(conn, lines, ep.drama_id if ep else 0)

        # TS: `success(c, dialogueValidation ? { dialogue_validation } : undefined)`
        # ⚠️ 无校验结果时 JS 会把 undefined 交给 success 的**默认参数**（`data = null`），
        # 所以返回的是 `{"code":200,"data":null,...}` —— **data 键仍然存在**。
        # （曾误以为 JSON.stringify 会丢掉它而写成「不带 data 键」，已修。）
        if dialogue_validation:
            return success({"dialogue_validation": dialogue_validation})
        return success()
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc) or "Failed to update storyboard")


# ---------------------------------------------------------------------------
# GET /{id}/validate-dialogue
# ---------------------------------------------------------------------------

@router.get("/{storyboard_id}/validate-dialogue")
def validate_dialogue(storyboard_id: str, conn: Connection = Depends(get_conn)):
    sid = parse_param_id(storyboard_id)
    if sid is None:
        return not_found("Invalid storyboard id")

    sb = _fetch_storyboard(conn, sid)
    if sb is None:
        return not_found("镜头不存在")

    ep = conn.execute(select(episodes.c.drama_id).where(episodes.c.id == sb.episode_id)).first()
    drama_id = ep.drama_id if ep else 0

    lines = parse_dialogue_lines(sb.dialogue)
    if not lines:
        parsed = parse_dialogue_for_tts(sb.dialogue)
        # 拆不出可用行 ⇒ 直接返回「无可验证」（注意这里 data 仍是完整对象）
        if parsed["ignorable"] or not parsed["speaker"]:
            return success(
                {"lines": [], "all_matched": True, "issue_count": 0, "summary": "无可验证的对话行"}
            )
        lines.append({"speaker": parsed["speaker"], "text": parsed["pureText"]})

    validations = validate_dialogue_lines(conn, lines, drama_id)
    issues = [v for v in validations if v["match_status"] in ("not_found", "no_voice")]

    # TTS 过期检测：已生成 TTS 的 voice_id 是否与当前角色 voiceStyle 一致
    tts_stale_details = _detect_stale_tts(conn, sb.tts_audio_url, drama_id)

    return success(
        {
            "lines": validations,
            "all_matched": len(issues) == 0,
            "issue_count": len(issues),
            "summary": "全部角色匹配成功" if not issues else f"{len(issues)} 个角色存在音色匹配问题",
            "issues": [
                {"speaker": v["speaker"], "status": v["match_status"], "warning": v.get("warning")}
                for v in issues
            ],
            "tts_stale": len(tts_stale_details) > 0,
            "tts_stale_count": len(tts_stale_details),
            "tts_stale_details": tts_stale_details,
        }
    )


def _detect_stale_tts(conn: Connection, tts_audio_url: Any, drama_id: int) -> list[dict[str, Any]]:
    """检查已生成 TTS 的音色是否与角色当前 voiceStyle 一致；不一致即「过期」。"""
    details: list[dict[str, Any]] = []
    if not tts_audio_url:
        return details
    try:
        tts_lines = json.loads(tts_audio_url)
    except (ValueError, TypeError):
        return details
    if not isinstance(tts_lines, list):
        # 单人模式（字段是 static/... 路径）无法精确检测，原 TS 也是 skipped
        return details

    for line in tts_lines:
        if not isinstance(line, dict):
            continue
        if not line.get("voice_id") or not line.get("speaker"):
            continue
        chars = conn.execute(
            select(characters).where(
                characters.c.drama_id == drama_id,
                characters.c.deleted_at.is_(None),
            )
        ).all()
        char = match_character_by_speaker_name(chars, line["speaker"])
        if char is not None and char.voice_style and char.voice_style != line["voice_id"]:
            details.append(
                {
                    "speaker": line["speaker"],
                    "stored_voice": line["voice_id"],
                    "current_voice": char.voice_style,
                    "warning": (
                        f'角色"{line["speaker"]}"的 TTS 音色已过期'
                        f"（当前音色 {char.voice_style}，配音使用 {line['voice_id']}）"
                    ),
                }
            )
    return details


# ---------------------------------------------------------------------------
# DELETE /{id} — 硬删
# ---------------------------------------------------------------------------

@router.delete("/{storyboard_id}")
def delete_storyboard(storyboard_id: str, conn: Connection = Depends(get_tx)):
    try:
        sid = parse_param_id(storyboard_id)
        if sid is None:
            return not_found("Invalid storyboard id")
        # ⚠️ 只清 storyboard_characters，**不清 storyboard_props** —— 与 TS 一致
        conn.execute(
            delete(storyboard_characters).where(storyboard_characters.c.storyboard_id == sid)
        )
        conn.execute(delete(storyboards).where(storyboards.c.id == sid))
        return success()
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# GET /{id}/qc — 最近一次 QC 打分
# ---------------------------------------------------------------------------

@router.get("/{storyboard_id}/qc")
def get_storyboard_qc(storyboard_id: str, conn: Connection = Depends(get_conn)):
    sid = parse_param_id(storyboard_id)
    if sid is None:
        return not_found("Invalid storyboard id")

    rows = conn.execute(
        select(video_quality_checks).where(video_quality_checks.c.storyboard_id == sid)
    ).all()
    if not rows:
        return success({"message": "No QC record yet"})
    # 原 TS 无 ORDER BY，取回后按 id 降序取首条
    record = max(rows, key=lambda r: r.id)
    payload = row_to_camel(record, "video_quality_checks")
    payload["issues"] = json.loads(record.issues) if record.issues else []
    payload["dimensions"] = json.loads(record.dimensions) if record.dimensions else {}
    return success(payload)


# ---------------------------------------------------------------------------
# POST /{id}/generate-tts — 逐角色生成配音（多人对话为多文件）
# ---------------------------------------------------------------------------

def _compact_json(value: Any) -> str:
    """``JSON.stringify(value)`` —— **紧凑分隔符**（守卫在盯，别写默认分隔符）。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _get_character_voice_params(conn: Connection, character_id: Any) -> dict[str, Any]:
    """角色个性化声音参数（``voiceSpeed`` / ``voiceEmotion`` / ``voicePitch`` / ``voiceModel``）。

    ⚠️ 原 TS 的第二个参数 ``dramaId`` **实际没用到**（签名里留着而已），这里直接省掉。
    """
    if not character_id:
        return {}
    char = conn.execute(
        select(characters).where(and_(characters.c.id == character_id,
                                      characters.c.deleted_at.is_(None)))
    ).first()
    if char is None:
        return {}
    return {
        "speed": char.voice_speed,
        "emotion": char.voice_emotion,
        "pitch": char.voice_pitch,
        "model": char.voice_model,
    }


@router.post("/{storyboard_id}/qc")
def score_storyboard_qc(storyboard_id: str, conn: Connection = Depends(get_tx)):
    """镜头级 QC 打分（三规则维度 + 落 ``video_quality_checks``）。"""
    try:
        sid = parse_param_id(storyboard_id)
        if sid is None:
            return not_found("Invalid storyboard id")
        exists = conn.execute(
            select(storyboards.c.id).where(storyboards.c.id == sid)).first()
        if exists is None:
            return not_found("镜头不存在")
        return success(score_storyboard(conn, sid))
    except Exception as exc:  # noqa: BLE001
        log_task_error("StoryboardAPI", "qc-score",
                       {"storyboardId": storyboard_id, "error": str(exc)})
        return bad_request(str(exc))


#: 视频素材的扩展名判定（原 TS 的裸正则，含 m4v/avi/mpeg/mkv）
_VIDEO_EXT_RE = re.compile(r"\.(mp4|mov|webm|m4v|avi|mpeg|mkv)$", re.IGNORECASE)


@router.post("/{storyboard_id}/set-frame")
async def set_storyboard_frame(storyboard_id: str, request: Request,
                              conn: Connection = Depends(get_tx)):
    """把一段**素材**设成该镜头的首帧或尾帧（视频素材先抽帧，图片素材直接采用）。"""
    try:
        sid = parse_param_id(storyboard_id)
        if sid is None:
            return not_found("Invalid storyboard id")
        body = await read_json(request)
        # ⚠️ 只认 `last_frame`，其余一律 first_frame（原 TS 是真值判断，不是白名单报错）
        frame_type = "last_frame" if body.get("frame_type") == "last_frame" else "first_frame"
        source_url = body.get("source_url") or ""
        exists = conn.execute(select(storyboards.c.id).where(storyboards.c.id == sid)).first()
        if exists is None:
            return not_found("镜头不存在")
        if not source_url:
            return bad_request("source_url required")

        is_video = bool(_VIDEO_EXT_RE.search(source_url))
        frame_url = await extract_frame(source_url, frame_type) if is_video else source_url
        if not frame_url:
            # 抽帧失败：原 TS 的 extractVideoFrame 会 reject ⇒ 这里同样抛成 400
            raise ValueError("抽帧失败（ffmpeg 提取不到该帧）")

        column = "first_frame_image" if frame_type == "first_frame" else "last_frame_image"
        conn.execute(storyboards.update().where(storyboards.c.id == sid)
                     .values(**{column: frame_url}, updated_at=now()))
        log_task_success("StoryboardAPI", "set-frame", {
            "storyboardId": sid, "frameType": frame_type, "sourceUrl": source_url,
            "frameUrl": frame_url, "isVideo": is_video,
        })
        return success({"frame_type": frame_type, "frame_url": frame_url})
    except Exception as exc:  # noqa: BLE001
        log_task_error("StoryboardAPI", "set-frame",
                       {"storyboardId": storyboard_id, "error": str(exc)})
        return bad_request(str(exc))


@router.post("/{storyboard_id}/retry-qc")
async def retry_storyboard_qc(storyboard_id: str, conn: Connection = Depends(get_tx)):
    """审片重跑：只重写该失败镜 → 软删旧产物 → 重提首帧/视频（闭环）。"""
    try:
        sid = parse_param_id(storyboard_id)
        if sid is None:
            return not_found("Invalid storyboard id")
        exists = conn.execute(
            select(storyboards.c.id).where(storyboards.c.id == sid)).first()
        if exists is None:
            return not_found("镜头不存在")
        return success(await retry_failed_storyboard(conn, sid))
    except Exception as exc:  # noqa: BLE001
        log_task_error("StoryboardAPI", "retry-qc",
                       {"storyboardId": storyboard_id, "error": str(exc)})
        return bad_request(str(exc) or "Failed to retry storyboard")


@router.post("/{storyboard_id}/generate-tts")
async def generate_storyboard_tts(storyboard_id: str, request: Request,
                                  conn: Connection = Depends(get_tx)):
    """生成镜头配音。

    * **多角色**（能拆出 >1 条「角色名：台词」）⇒ 每个角色**一个音频文件**，
      结果数组以 JSON 存进 ``tts_audio_url``，回执 ``{lines: [...]}``；
    * **单人 / 拆不出**⇒ 走旧逻辑（``parse_dialogue_for_tts``），回执是**扁平对象**
      （``tts_audio_url`` / ``voice_id`` / ``match_status`` / ``speaker`` / ``text`` + 条件 ``warning``）；
    * 单人模式下若解析结果 ``ignorable``（纯环境音/无对白）⇒ **400** 提前返回（**不落日志**）。
    """
    sid = parse_param_id(storyboard_id)
    if sid is None:
        return not_found("Invalid storyboard id")
    sb = _fetch_storyboard(conn, sid)
    if sb is None:
        return not_found("镜头不存在")

    dialogue_lines = parse_dialogue_lines(sb.dialogue)
    use_multi_mode = len(dialogue_lines) > 1
    if not use_multi_mode and len(dialogue_lines) == 0:
        parsed = parse_dialogue_for_tts(sb.dialogue)
        if parsed["ignorable"]:
            return bad_request("该镜头没有可生成的对白或旁白")

    log_task_start("StoryboardAPI", "generate-tts", {
        "storyboardId": sid,
        "episodeId": sb.episode_id,
        "dialoguePreview": (sb.dialogue or "")[:40],
        "lineCount": len(dialogue_lines),
        "speakers": [line["speaker"] for line in dialogue_lines],
    })
    log_task_payload("StoryboardAPI", "generate-tts input", {
        "storyboardId": sid,
        "episodeId": sb.episode_id,
        "dialogue": sb.dialogue,
        "parsedLines": dialogue_lines,
    })

    episode = conn.execute(
        select(episodes).where(episodes.c.id == sb.episode_id)).first()
    drama_id = episode.drama_id if episode is not None else 0
    audio_config_id = (episode.audio_config_id if episode is not None else None) or None

    try:
        if use_multi_mode:
            results: list[dict[str, Any]] = []
            validations = validate_dialogue_lines(conn, dialogue_lines, drama_id)

            # 记录不匹配的警告
            warnings = [item for item in validations if item.get("warning")]
            if warnings:
                log_task_payload("StoryboardAPI", "generate-tts validation warnings", warnings)

            for validation in validations:
                # 从角色记录中获取个性化声音参数
                voice = _get_character_voice_params(conn, validation.get("character_id"))
                audio_path = await generate_tts(conn, {
                    "text": validation["text"],
                    "voice": validation["voice_id"],
                    "speed": voice.get("speed"),
                    "emotion": voice.get("emotion"),
                    "pitch": voice.get("pitch"),
                    "model": voice.get("model"),
                    "configId": audio_config_id,
                })
                item = {
                    "speaker": validation["speaker"],
                    "text": validation["text"],
                    "tts_audio_url": audio_path,
                    "voice_id": validation["voice_id"],
                    "match_status": validation["match_status"],
                }
                # TS 是条件展开 `...(val.warning ? { warning } : {})`
                if validation.get("warning"):
                    item["warning"] = validation["warning"]
                results.append(item)

            # 多条音频 URL 以 JSON 存入 ttsAudioUrl 字段
            conn.execute(update(storyboards).where(storyboards.c.id == sid)
                         .values(tts_audio_url=_compact_json(results), updated_at=now()))
            log_task_success("StoryboardAPI", "generate-tts", {
                "storyboardId": sid, "lineCount": len(results),
                "speakers": [item["speaker"] for item in results],
            })
            return success({"lines": results})

        # 单人/无法拆分：用旧逻辑直接生成
        if len(dialogue_lines) == 1:
            speaker = dialogue_lines[0]["speaker"]
            text = dialogue_lines[0]["text"]
        else:
            parsed = parse_dialogue_for_tts(sb.dialogue)
            speaker = parsed["speaker"]
            text = parsed["pureText"]

        validation = validate_tts_speaker(conn, speaker, drama_id)
        voice = _get_character_voice_params(conn, validation.get("characterId"))
        audio_path = await generate_tts(conn, {
            "text": text,
            "voice": validation["voiceId"],
            "configId": audio_config_id,
            "speed": voice.get("speed"),
            "emotion": voice.get("emotion"),
            "pitch": voice.get("pitch"),
            "model": voice.get("model"),
        })

        conn.execute(update(storyboards).where(storyboards.c.id == sid)
                     .values(tts_audio_url=audio_path, updated_at=now()))
        log_task_success("StoryboardAPI", "generate-tts", {
            "storyboardId": sid, "voiceId": validation["voiceId"],
            "path": audio_path, "textLength": len(text),
        })

        payload: dict[str, Any] = {
            "tts_audio_url": audio_path,
            "voice_id": validation["voiceId"],
            "match_status": validation["match_status"],
            "speaker": speaker,
            "text": text,
        }
        if validation["match_status"] == "not_found":
            payload["warning"] = f'角色"{speaker}"在剧组角色列表中不存在'
        if validation["match_status"] == "no_voice":
            payload["warning"] = f'角色"{speaker}"尚未配置音色'
        return success(payload)
    except Exception as exc:  # noqa: BLE001
        log_task_error("StoryboardAPI", "generate-tts",
                       {"storyboardId": sid, "error": str(exc)})
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# POST /{id}/regenerate-image — 重新生成镜头首帧图
# ---------------------------------------------------------------------------

@router.post("/{storyboard_id}/regenerate-image")
async def regenerate_storyboard_image(storyboard_id: str, request: Request,
                                      conn: Connection = Depends(get_tx)):
    """重新生成分镜首帧图（注入角色外观 + 场景描述 + 角色/场景参考图，保证一致性）。"""
    sid = parse_param_id(storyboard_id)
    if sid is None:
        return not_found("Invalid storyboard id")
    body = await read_json(request)
    sb = _fetch_storyboard(conn, sid)
    if sb is None:
        return not_found("镜头不存在")
    episode = conn.execute(
        select(episodes).where(episodes.c.id == sb.episode_id)).first()
    if episode is None:
        return bad_request("Episode not found")

    try:
        # 注入角色外观 + 场景描述 + 角色/场景参考图，保证人物与场景一致
        char_appearances = get_storyboard_character_appearances(conn, sid)
        scene_desc = get_storyboard_scene_description(conn, sid)
        explicit_refs = body.get("reference_images")
        reference_images = (explicit_refs if explicit_refs
                            else get_storyboard_reference_images(conn, sid))

        # 画风收口：统一解析链（请求体 style → 剧集 style → 全局默认 → realistic），
        # 正/负提示词**同源**，避免「画风靠请求体、负面词却用通用词」的错配
        drama_style = resolve_effective_art_style(conn, episode.drama_id, None, body.get("style"))

        # 自定义 prompt 优先 → 分镜级 customImagePrompt → 标准构建器
        if body.get("prompt"):
            prompt = body["prompt"]
        elif sb.custom_image_prompt:
            prompt = sb.custom_image_prompt
        else:
            prompt = build_storyboard_image_prompt({
                "description": sb.description or body.get("character_description") or "",
                "storyboardDescription": sb.description,
                "characterDescription": "；".join(char_appearances) if char_appearances else None,
                "sceneDescription": (body.get("scene_description") or scene_desc
                                     or sb.location or ""),
                "shotType": sb.shot_type or body.get("shot_type") or "",
                "cameraAngle": sb.angle or body.get("camera_angle") or "",
                "dramaStyle": drama_style,
            })

        log_task_start("StoryboardAPI", "regenerate-image", {
            "storyboardId": sid, "episodeId": sb.episode_id, "dramaId": episode.drama_id,
            "model": body.get("model") or "default",
        })

        gen_id = await generate_image(conn, {
            "storyboardId": sid,
            "dramaId": episode.drama_id,
            "prompt": prompt,
            "negativePrompt": (body.get("negative_prompt") or sb.negative_prompt
                               or build_storyboard_negative_prompt(drama_style)),
            "model": body.get("model"),
            "referenceImages": reference_images,
            "configId": episode.image_config_id,
            "force": body.get("force"),
        })

        log_task_success("StoryboardAPI", "regenerate-image",
                         {"storyboardId": sid, "generationId": gen_id})
        return success({"image_generation_id": gen_id})
    except Exception as exc:  # noqa: BLE001
        log_task_error("StoryboardAPI", "regenerate-image",
                       {"storyboardId": sid, "error": str(exc)})
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# POST /{id}/regenerate-frame — 重新生成镜头首帧/尾帧/关键帧
# ---------------------------------------------------------------------------

#: 帧类型白名单（原 TS：`['last_frame','keyframe'].includes(...)` ⇒ 其余一律 first_frame）
_FRAME_TYPES = ("last_frame", "keyframe")

#: 三种帧的画面提示词（英文，逐字对齐 TS）
_FRAME_HINTS = {
    "first_frame": "opening frame, establishing the scene, subject at start position, "
                   "beginning of the shot",
    "last_frame": "closing frame, final composition, subject at end position, end of the shot",
    "keyframe": "mid-action keyframe, subject mid-motion, action or prop state in transition, "
                "intermediate moment of the shot",
}


@router.post("/{storyboard_id}/regenerate-frame")
async def regenerate_storyboard_frame(storyboard_id: str, request: Request,
                                     conn: Connection = Depends(get_tx)):
    """重新生成首帧/尾帧/关键帧（注入角色外观 + 场景 + 参考图；与 regenerate-image 同一条画风链）。"""
    sid = parse_param_id(storyboard_id)
    if sid is None:
        return not_found("Invalid storyboard id")
    body = await read_json(request)
    # ⚠️ 白名单判定：只有 last_frame/keyframe 被认，其余（含未传）一律 first_frame
    raw_type = body.get("frame_type")
    frame_type = raw_type if raw_type in _FRAME_TYPES else "first_frame"
    sb = _fetch_storyboard(conn, sid)
    if sb is None:
        return not_found("镜头不存在")
    episode = conn.execute(
        select(episodes).where(episodes.c.id == sb.episode_id)).first()
    if episode is None:
        return bad_request("Episode not found")

    try:
        # 注入角色外观 + 场景描述 + 角色/场景参考图，保证首尾帧人物与场景一致
        char_appearances = get_storyboard_character_appearances(conn, sid)
        scene_desc = get_storyboard_scene_description(conn, sid)
        explicit_refs = body.get("reference_images")
        reference_images = (explicit_refs if explicit_refs
                            else get_storyboard_reference_images(conn, sid))

        # 画风收口：与 regenerate-image 同一条解析链，保证同一镜头不同帧不会换画风
        drama_style = resolve_effective_art_style(conn, episode.drama_id, None, body.get("style"))

        # 帧画面内容：**请求体 prompt 优先**，其次分镜存库的对应帧 prompt
        stored_frame_prompt = getattr(sb, {
            "first_frame": "first_frame_prompt",
            "last_frame": "last_frame_prompt",
            "keyframe": "keyframe_prompt",
        }[frame_type])
        frame_content = body.get("prompt") or stored_frame_prompt

        # 画面基底：标准构建器（始终注入角色外观 + 场景），帧画面内容作为附加描述叠加
        base_prompt = build_storyboard_image_prompt({
            "description": sb.description or body.get("character_description") or "",
            "storyboardDescription": sb.description,
            "characterDescription": "；".join(char_appearances) if char_appearances else None,
            "sceneDescription": (body.get("scene_description") or scene_desc
                                 or sb.location or ""),
            "shotType": sb.shot_type or body.get("shot_type") or "",
            "cameraAngle": sb.angle or body.get("camera_angle") or "",
            "dramaStyle": drama_style,
        })
        prompt = ", ".join(
            part for part in (base_prompt, frame_content, _FRAME_HINTS[frame_type])
            if part
        )

        log_task_start("StoryboardAPI", "regenerate-frame", {
            "storyboardId": sid, "episodeId": sb.episode_id, "dramaId": episode.drama_id,
            "frameType": frame_type, "model": body.get("model") or "default",
        })

        gen_id = await generate_image(conn, {
            "storyboardId": sid,
            "dramaId": episode.drama_id,
            "prompt": prompt,
            "negativePrompt": (body.get("negative_prompt") or sb.negative_prompt
                               or build_storyboard_negative_prompt(drama_style)),
            "model": body.get("model"),
            "frameType": frame_type,
            "referenceImages": reference_images,
            "configId": episode.image_config_id,
            "force": body.get("force"),
        })

        log_task_success("StoryboardAPI", "regenerate-frame",
                         {"storyboardId": sid, "frameType": frame_type, "generationId": gen_id})
        return success({"image_generation_id": gen_id, "frame_type": frame_type})
    except Exception as exc:  # noqa: BLE001
        log_task_error("StoryboardAPI", "regenerate-frame",
                       {"storyboardId": sid, "frameType": frame_type, "error": str(exc)})
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# POST /{id}/action-suggestion — AI 生成运镜/动作建议
# ---------------------------------------------------------------------------

@router.post("/{storyboard_id}/action-suggestion")
async def storyboard_action_suggestion(storyboard_id: str,
                                       conn: Connection = Depends(get_tx)):
    """AI 生成运镜/动作建议（纯 LLM，**不落库**）。"""
    sid = parse_param_id(storyboard_id)
    if sid is None:
        return not_found("Invalid storyboard id")
    try:
        sb = _fetch_storyboard(conn, sid)
        if sb is None:
            return not_found("镜头不存在")

        suggestion = await generate_action_suggestion(conn, {
            "title": sb.title,
            "description": sb.description,
            "action": sb.action,
            "imagePrompt": sb.custom_image_prompt or sb.image_prompt,
            "atmosphere": sb.atmosphere,
            "shotType": sb.shot_type,
            "movement": sb.movement,
            "angle": sb.angle,
        })
        return success({"suggestion": suggestion})
    except Exception as exc:  # noqa: BLE001
        log_task_error("StoryboardAPI", "action-suggestion",
                       {"storyboardId": sid, "error": str(exc)})
        return bad_request(str(exc) or "Failed to generate action suggestion")


# ---------------------------------------------------------------------------
# POST /{id}/split — AI 拆分长镜头为多个子镜头
# ---------------------------------------------------------------------------

def _shot_context(conn: Connection, sb: Any) -> tuple[dict[str, Any], str, list[str], list[int]]:
    """拆分/优化 prompt 共用的上下文：场景信息、视觉风格、出场角色名与 id。

    ⚠️ 视觉风格链是 **storyboard → episode → drama**（取 ``dramas.style``）；
    有 ``scene_id`` 时场景信息**整体替换**成场景表的（含 ``location``），否则用分镜自身的。
    """
    visual_style = ""
    episode = conn.execute(
        select(episodes).where(episodes.c.id == sb.episode_id)).first()
    if episode is not None:
        drama = conn.execute(select(dramas).where(dramas.c.id == episode.drama_id)).first()
        if drama is not None:
            visual_style = drama.style or ""

    scene_info = {"location": sb.location or "", "time": sb.time or "",
                  "atmosphere": sb.atmosphere or ""}
    if sb.scene_id:
        scene = conn.execute(select(scenes).where(scenes.c.id == sb.scene_id)).first()
        if scene is not None:
            scene_info = {"location": scene.location, "time": scene.time or "",
                          "atmosphere": scene.atmosphere or ""}

    char_ids = get_storyboard_character_ids(conn, sb.id)
    character_names: list[str] = []
    if char_ids:
        character_names = [row.name for row in conn.execute(
            select(characters).where(characters.c.id.in_(char_ids))).all()]
    return scene_info, visual_style, character_names, char_ids


@router.post("/{storyboard_id}/split")
async def split_storyboard(storyboard_id: str, conn: Connection = Depends(get_tx)):
    """AI 拆分长镜头为多个子镜头（**保留原镜头**，新镜头追加其后，后续分镜号顺延）。"""
    sid = parse_param_id(storyboard_id)
    if sid is None:
        return not_found("Invalid storyboard id")
    try:
        sb = _fetch_storyboard(conn, sid)
        if sb is None:
            return not_found("镜头不存在")

        scene_info, visual_style, character_names, char_ids = _shot_context(conn, sb)
        sub_shots = await split_shot_into_sub_shots(conn, {
            "title": sb.title,
            "action": sb.action,
            "description": sb.description,
            "shotType": sb.shot_type,
            "atmosphere": sb.atmosphere,
            "dialogue": sb.dialogue,
            "sceneInfo": scene_info,
            "characterNames": character_names,
            "visualStyle": visual_style,
        })

        # 落库：保留原镜头 + 新子镜头追加在原镜头之后 + 后续分镜号顺延
        ts = now()
        base_number = sb.storyboard_number
        shift = len(sub_shots)
        if shift > 0:
            later = conn.execute(
                select(storyboards).where(and_(
                    storyboards.c.episode_id == sb.episode_id,
                    storyboards.c.storyboard_number > base_number))
            ).all()
            for row in later:
                conn.execute(update(storyboards)
                             .where(storyboards.c.id == row.id)
                             .values(storyboard_number=row.storyboard_number + shift))

        created_shots: list[dict[str, Any]] = []
        for index, sub in enumerate(sub_shots):
            new_id = int(conn.execute(storyboards.insert().values(
                episode_id=sb.episode_id,
                scene_id=sb.scene_id,
                storyboard_number=base_number + 1 + index,
                title=f"{sb.title or '镜头'} · {sub['shotSize']}",
                description=sub.get("visualFocus") or sub["actionSummary"],
                action=sub["actionSummary"],
                shot_type=sub["shotSize"],
                movement=sub.get("cameraMovement") or sb.movement,
                atmosphere=sb.atmosphere,
                dialogue=None,
                # `Math.round((sb.duration || 4) / subShots.length)` 再夹到 [2, 4]
                duration=max(2, min(4, js_round((sb.duration or 4) / len(sub_shots)))),
                status="pending",
                created_at=ts,
                updated_at=ts,
            )).lastrowid)
            if char_ids:
                sync_storyboard_characters(conn, new_id, char_ids)
            created_shots.append({
                "id": new_id,
                "storyboardNumber": base_number + 1 + index,
                "shotSize": sub["shotSize"],
                "cameraMovement": sub.get("cameraMovement"),
                "actionSummary": sub["actionSummary"],
                "visualFocus": sub.get("visualFocus"),
            })

        log_task_success("StoryboardAPI", "split",
                         {"storyboardId": sid, "count": len(created_shots)})
        return success({"subShots": created_shots})
    except Exception as exc:  # noqa: BLE001
        log_task_error("StoryboardAPI", "split", {"storyboardId": sid, "error": str(exc)})
        return bad_request(str(exc) or "Failed to split shot")


# ---------------------------------------------------------------------------
# POST /{id}/optimize-prompt — AI 优化视频生成提示词
# ---------------------------------------------------------------------------

@router.post("/{storyboard_id}/optimize-prompt")
async def optimize_storyboard_prompt(storyboard_id: str, request: Request,
                                     conn: Connection = Depends(get_tx)):
    """AI 优化视频生成提示词（用户主动触发；**不落库**，回执 ``{optimizedPrompt}``）。"""
    sid = parse_param_id(storyboard_id)
    if sid is None:
        return not_found("Invalid storyboard id")
    try:
        sb = _fetch_storyboard(conn, sid)
        if sb is None:
            return not_found("镜头不存在")
        body = await read_json(request)

        scene_info, visual_style, character_names, _char_ids = _shot_context(conn, sb)
        optimized = await optimize_video_prompt(conn, {
            "currentPrompt": body.get("currentPrompt"),
            "title": sb.title,
            "action": sb.action,
            "description": sb.description,
            "shotType": sb.shot_type,
            "movement": sb.movement,
            "atmosphere": sb.atmosphere,
            "sceneInfo": scene_info,
            "characterNames": character_names,
            "visualStyle": visual_style,
        })

        log_task_success("StoryboardAPI", "optimize-prompt", {"storyboardId": sid})
        return success({"optimizedPrompt": optimized})
    except Exception as exc:  # noqa: BLE001
        log_task_error("StoryboardAPI", "optimize-prompt",
                       {"storyboardId": sid, "error": str(exc)})
        return bad_request(str(exc) or "Failed to optimize prompt")
