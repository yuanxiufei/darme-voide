"""aiVoices 域 —— 与 ``backend/src/routes/aiVoices.ts``（437 行）对齐。

**5 个端点全部迁移**（前缀 ``/api/v1/ai-voices``）：

* ``GET  /``                        音色列表（按 ``provider`` 过滤，默认 ``minimax``）；
* ``POST /preview``                 固定文案试听；
* ``POST /clone``                   **multipart** 上传参考音频 → 复刻为可复用音色；
* ``POST /generate-from-characters`` 按剧组角色批量生成专属音色（即时合成 ≥10s 参考音频再克隆）；
* ``POST /sync``                    从 MiniMax ``get_voice`` 同步系统音色 + AI 打标。

⚠️ 七处保真点：

1. ``GET /`` 的 ``description`` 是**裸 JSON.parse**（没有 try）——原实现遇到脏数据会抛错，
   这里同样不兜底（``role_tags`` 才走容错解析）；
2. ``/clone`` 的 provider 只支持 ``minimax`` / ``cosyvoice``，且 **cosyvoice 必须给
   ``prompt_text``**（零样本克隆的参考文本）；
3. 音色 id 生成：``ds_``/``cv_`` 前缀 + **base36 时间戳** + 4 位 base36 随机；手填 id 要过
   ``^[a-zA-Z][a-zA-Z0-9_-]{7,255}$`` 且**不能以 ``-``/``_`` 结尾**；
4. 克隆音色名称回落 ``克隆音色 <id 后 6 位>``；角色音色名用角色名；
5. ``/generate-from-characters`` 对 **``ds_``/``cv_`` 开头的角色音色直接 skip**（避免重复克隆），
   参考音频是**即时合成的约 15 秒长文本**（MiniMax 克隆要求 ≥10 秒，试听音频只有 4 秒会报
   ``voice duration too short``）；
6. ``/sync`` 会**先删光**该 provider 的旧音色再插入；AI 打标失败**不阻塞**同步；
7. 写音色库一律 ``onConflictDoNothing``（**不覆盖**已有音色）。
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, Request
from sqlalchemy import and_, delete as sql_delete, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection

from ..config import get_storage_root
from ..db import get_conn, get_tx
from ..models import ai_service_configs, ai_voices, characters, dramas, episodes
from ..request_utils import read_json
from ..response import bad_request, not_found, now, success
from ..services.adapters.url import join_provider_url
from ..services.ai_providers import get_audio_config
from ..services.task_logger import log_task_error
from ..services.text_generation import infer_voice_role_tags
from ..services.tts_generation import generate_tts
from ..services.vendor_errors import vendor_response_error
from ..services.voice_clone import clone_voice, clone_voice_cosyvoice

router = APIRouter(prefix="/api/v1/ai-voices", tags=["ai-voices"])

#: `/generate-from-characters` 用的参考音频文本（≈15 秒；MiniMax 克隆要求 ≥10 秒）
REF_TEXT = (
    "在这座城市里，每天都有许多故事在上演。清晨的阳光洒在街道上，午后微风拂过树梢，"
    "傍晚霞光映红了天边。我愿意把最温暖的声音带给你，陪伴你度过每一个平凡而美好的日子。"
)

#: 试听默认文案
DEFAULT_SAMPLE_TEXT = "你好，欢迎来到短剧工坊，这是我的声音试听。"

#: 克隆音色 id 校验（首字符英文字母，8-256 位，仅字母/数字/-/_，且不以 -/_ 结尾）
_CLONE_ID_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{7,255}$")

#: MiniMax 系统音色里的排除项（儿童音/配音腔/客服腔等，不适合短剧）
_EXCLUDED_VOICE_PATTERNS = (
    "jingpin", "-beta", "cartoon_pig", "cute_boy", "lovely_girl", "clever_boy",
    "robot_armor", "news_anchor", "male_announcer", "radio_host", "hk_flight_attendant",
)


def _base36(value: int) -> str:
    """JS ``Number.prototype.toString(36)`` 的等价实现。"""
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    out = ""
    while value > 0:
        value, remainder = divmod(value, 36)
        out = digits[remainder] + out
    return out


def _random_base36(length: int) -> str:
    """``Math.random().toString(36).slice(2, 2+n)`` 的等价实现（小写字母/数字）。"""
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(digits) for _ in range(length))


def generate_clone_voice_id(raw: str, prefix: str = "ds_") -> str:
    """手填 id 就校验后原样用（非法抛错）；否则生成 ``<prefix><base36 时间戳>_<4 位随机>``。"""
    if raw:
        value = raw.strip()
        if _CLONE_ID_RE.match(value) and not value.endswith(("-", "_")):
            return value
        raise ValueError("voice_id 需 8-256 字符，首字符为英文字母，仅含字母/数字/-/_")
    return f"{prefix}{_base36(int(time.time() * 1000))}_{_random_base36(4)}"


def parse_json_array(raw: Any) -> list[str]:
    """容错解析 JSON 数组（非数组/脏数据 → 空）。"""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def extract_language(voice_id: str, voice_name: str) -> str:
    """从 voice_id / voice_name 推断语言（顺序敏感：先特殊语言，中文最后）。"""
    text = f"{voice_id} {voice_name}".lower()
    for needles, language in (
        (("cantonese", "粤"), "粤语"),
        (("english", "aussie"), "英语"),
        (("japanese", "日语"), "日语"),
        (("korean", "韩"), "韩语"),
        (("spanish",), "西班牙语"),
        (("portuguese",), "葡萄牙语"),
        (("french",), "法语"),
        (("indonesian",), "印尼语"),
        (("german",), "德语"),
        (("russian",), "俄语"),
        (("italian",), "意大利语"),
        (("arabic",), "阿拉伯语"),
        (("turkish",), "土耳其语"),
        (("ukrainian",), "乌克兰语"),
        (("dutch",), "荷兰语"),
        (("vietnamese",), "越南语"),
        (("chinese", "mandarin", "中文"), "中文"),
    ):
        if any(needle in text for needle in needles):
            return language
    return "其他"


def should_keep_voice(voice_id: str, voice_name: str) -> bool:
    """只留中文/粤语，且不含排除项（儿童音/播音腔等）。"""
    if extract_language(voice_id, voice_name) not in ("中文", "粤语"):
        return False
    text = f"{voice_id} {voice_name}".lower()
    return not any(pattern in text for pattern in _EXCLUDED_VOICE_PATTERNS)


def resolve_local_audio_path(url: str) -> str | None:
    """``/static/...`` 相对 URL → 本地绝对路径（去 query、剥 ``static/`` 前缀）。"""
    if not url:
        return None
    cleaned = url.split("?", 1)[0]
    rel = re.sub(r"^/?static/", "", cleaned)
    rel = re.sub(r"^/", "", rel)
    if not rel:
        return None
    return str(Path(get_storage_root()) / rel)


def resolve_drama_config_id(conn: Connection, drama_id: int, field: str) -> Any:
    """取剧下**任一**剧集里的该配置 id（找不到返回 None）。"""
    rows = conn.execute(
        select(episodes.c.image_config_id, episodes.c.audio_config_id)
        .where(and_(episodes.c.drama_id == drama_id, episodes.c.deleted_at.is_(None)))
    ).all()
    index = 0 if field == "imageConfigId" else 1
    for row in rows:
        if row[index] is not None:
            return row[index]
    return None


def map_role_to_tag(role: str | None) -> str:
    """角色 role → 音色 role_tags 四类（龙套归入配角）。"""
    value = (role or "").strip()
    return value if value in ("主角", "反派", "旁白") else "配角"


def _insert_voice_ignore_conflict(conn: Connection, values: dict[str, Any]) -> None:
    """写音色库，冲突（voice_id 已存在）**不覆盖**。"""
    conn.execute(sqlite_insert(ai_voices).values(**values).on_conflict_do_nothing(
        index_elements=["voice_id"]
    ))


@router.get("")
def list_voices(request: Request, conn: Connection = Depends(get_conn)):
    """音色列表（``provider`` 默认 minimax）。"""
    provider = request.query_params.get("provider") or "minimax"
    rows = conn.execute(
        select(ai_voices).where(ai_voices.c.provider == provider)
    ).all()
    return success([
        {
            "voice_id": row.voice_id,
            "voice_name": row.voice_name,
            # ⚠️ 裸 JSON.parse（无兜底）：脏数据在这里就是抛错，与原实现一致
            "description": json.loads(row.description) if row.description else [],
            "role_tags": parse_json_array(row.role_tags),
            "language": row.language,
            "provider": row.provider,
        }
        for row in rows
    ])


@router.post("/preview")
async def preview_voice(request: Request, conn: Connection = Depends(get_conn)):
    """固定文案试听（不绑定角色）。"""
    body = await read_json(request)
    voice_id = body.get("voice_id") or body.get("voiceId")
    if not voice_id:
        return bad_request("voice_id is required")
    try:
        sample_text = body.get("text") or DEFAULT_SAMPLE_TEXT
        audio_path = await generate_tts(conn, {
            "text": sample_text, "voice": voice_id,
            "configId": body.get("config_id"),
        })
        return success({"voice_id": voice_id, "url": audio_path})
    except Exception as err:  # noqa: BLE001
        log_task_error("AiVoices", "preview", {"voiceId": voice_id, "error": str(err)})
        return bad_request(f"试听生成失败: {err}")


@router.post("/clone")
async def clone_voice_endpoint(request: Request, conn: Connection = Depends(get_tx)):
    """音色快速复刻（multipart 上传参考音频）。"""
    try:
        form = await request.form()
        upload = form.get("file")
        voice_name = str(form.get("voice_name") or form.get("voiceName") or "").strip()
        demo_text = form.get("demo_text") or form.get("demoText")
        prompt_text = str(form.get("prompt_text") or form.get("promptText") or "").strip()
        raw_voice_id = str(form.get("voice_id") or form.get("voiceId") or "")

        if upload is None or not hasattr(upload, "read"):
            return bad_request("file is required")

        config = get_audio_config(conn)
        is_cosy_voice = config.get("provider") == "cosyvoice"
        if config.get("provider") != "minimax" and not is_cosy_voice:
            return bad_request(
                f"音色复刻当前仅支持 MiniMax / CosyVoice 音频服务（当前 {config.get('provider')}）"
            )
        if is_cosy_voice and not prompt_text:
            return bad_request("CosyVoice 零样本克隆需要提供参考音频文本 prompt_text")

        file_buffer = await upload.read()
        voice_id = generate_clone_voice_id(raw_voice_id, "cv_" if is_cosy_voice else "ds_")

        demo_audio: str | None = None
        reference_audio: str | None = None
        ref_prompt_text: str | None = None

        if is_cosy_voice:
            sample_text = str(demo_text or DEFAULT_SAMPLE_TEXT)
            result = await clone_voice_cosyvoice({
                "baseUrl": config.get("baseUrl"),
                "fileBuffer": file_buffer,
                "promptText": prompt_text,
                "demoText": sample_text,
                "model": config.get("model"),
            })
            # demo 音频（base64）落盘为可试听 URL
            if result.get("demoAudio"):
                import base64

                audio_dir = Path(get_storage_root()) / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                demo_name = f"{uuid4()}.mp3"
                (audio_dir / demo_name).write_bytes(base64.b64decode(result["demoAudio"]))
                demo_audio = f"static/audio/{demo_name}"
            # 参考音频落盘，供后续 TTS 零样本复用
            voices_dir = Path(get_storage_root()) / "voices"
            voices_dir.mkdir(parents=True, exist_ok=True)
            ref_name = f"{uuid4()}.wav"
            (voices_dir / ref_name).write_bytes(file_buffer)
            reference_audio = f"static/voices/{ref_name}"
            ref_prompt_text = prompt_text
        else:
            result = await clone_voice({
                "baseUrl": config.get("baseUrl"),
                "apiKey": config.get("apiKey"),
                "fileBuffer": file_buffer,
                "filename": getattr(upload, "filename", None),
                "voiceId": voice_id,
                "demoText": str(demo_text) if demo_text else None,
                "model": config.get("model"),
            })
            demo_audio = result.get("demoAudio")

        _insert_voice_ignore_conflict(conn, {
            "voice_id": voice_id,
            "voice_name": voice_name or f"克隆音色 {voice_id[-6:]}",
            "description": json.dumps(["克隆音色"], ensure_ascii=False, separators=(",", ":")),
            "language": "中文",
            "provider": config.get("provider"),
            "reference_audio": reference_audio,
            "prompt_text": ref_prompt_text,
            "created_at": now(),
        })

        return success({"voice_id": voice_id, "demo_audio": demo_audio})
    except Exception as err:  # noqa: BLE001
        log_task_error("AiVoices", "clone", {"error": str(err)})
        return bad_request(f"音色克隆失败: {err}")


@router.post("/generate-from-characters")
async def generate_from_characters(request: Request, conn: Connection = Depends(get_tx)):
    """按剧角色批量生成专属音色（即时合成 ≥10s 参考音频再克隆）。"""
    try:
        body = await read_json(request)
        drama_id = body.get("drama_id") or body.get("dramaId")
        try:
            drama_id = int(drama_id)
        except (TypeError, ValueError):
            drama_id = 0
        if not drama_id:
            return bad_request("drama_id is required")

        drama = conn.execute(select(dramas).where(dramas.c.id == drama_id)).first()
        if drama is None:
            return not_found("Drama not found")

        char_rows = conn.execute(
            select(characters).where(
                and_(characters.c.drama_id == drama_id, characters.c.deleted_at.is_(None))
            )
        ).all()
        char_rows = [row for row in char_rows if row.voice_style]

        if not char_rows:
            return bad_request("该剧暂无已分配音色的角色，请先在角色页分配音色")

        config = get_audio_config(conn)
        is_cosy_voice = config.get("provider") == "cosyvoice"
        audio_config_id = resolve_drama_config_id(conn, drama_id, "audioConfigId")

        results: list[dict[str, Any]] = []
        ok_count = 0

        for char in char_rows:
            name = char.name or "角色"
            try:
                # 已是克隆专属音色（ds_/cv_ 前缀）则跳过，避免重复生成
                if re.match(r"^(ds_|cv_)", char.voice_style or ""):
                    results.append({
                        "character_id": char.id, "name": name, "status": "skipped",
                        "voice_id": char.voice_style, "reason": "已有专属音色",
                    })
                    continue

                # 即时合成 ≥10s 参考音频（MiniMax voice_clone 最低 10 秒）
                ref_path = await generate_tts(conn, {
                    "text": REF_TEXT, "voice": char.voice_style, "configId": audio_config_id,
                })
                file_path = resolve_local_audio_path(ref_path)
                if not file_path or not Path(file_path).exists():
                    results.append({
                        "character_id": char.id, "name": name, "status": "failed",
                        "error": "参考音频落盘失败",
                    })
                    continue

                file_buffer = Path(file_path).read_bytes()
                demo_text = f"你好，我是{name}。很高兴认识你，这是我的专属音色。"
                voice_id = generate_clone_voice_id("", "cv_" if is_cosy_voice else "ds_")

                reference_audio: str | None = None
                ref_prompt_text: str | None = None

                if is_cosy_voice:
                    await clone_voice_cosyvoice({
                        "baseUrl": config.get("baseUrl"),
                        "fileBuffer": file_buffer,
                        "promptText": REF_TEXT,
                        "demoText": demo_text,
                        "model": config.get("model"),
                    })
                    reference_audio = ref_path
                    ref_prompt_text = REF_TEXT
                else:
                    await clone_voice({
                        "baseUrl": config.get("baseUrl"),
                        "apiKey": config.get("apiKey"),
                        "fileBuffer": file_buffer,
                        "filename": Path(ref_path).name,
                        "voiceId": voice_id,
                        "voiceName": name,
                        "demoText": demo_text,
                        "model": config.get("model"),
                    })

                _insert_voice_ignore_conflict(conn, {
                    "voice_id": voice_id,
                    "voice_name": name,
                    "description": json.dumps([char.role or "角色音色"], ensure_ascii=False,
                                              separators=(",", ":")),
                    "language": "中文",
                    "provider": config.get("provider"),
                    "role_tags": json.dumps([map_role_to_tag(char.role)], ensure_ascii=False,
                                            separators=(",", ":")),
                    "reference_audio": reference_audio,
                    "prompt_text": ref_prompt_text,
                    "created_at": now(),
                })

                # 角色改用专属音色
                conn.execute(
                    update(characters).where(characters.c.id == char.id).values(
                        voice_style=voice_id, voice_provider=config.get("provider"),
                        updated_at=now(),
                    )
                )

                results.append({
                    "character_id": char.id, "name": name,
                    "status": "success", "voice_id": voice_id,
                })
                ok_count += 1
            except Exception as err:  # noqa: BLE001 —— 单个角色失败不影响其余
                log_task_error("AiVoices", "generate-from-characters", {
                    "characterId": char.id, "name": name, "error": str(err),
                })
                results.append({
                    "character_id": char.id, "name": name, "status": "failed", "error": str(err),
                })

        return success({"total": len(char_rows), "success_count": ok_count, "results": results})
    except Exception as err:  # noqa: BLE001
        log_task_error("AiVoices", "generate-from-characters", {"error": str(err)})
        return bad_request(f"生成音色失败: {err}")


async def fetch_minimax_voices(base_url: str, api_key: str) -> tuple[dict[str, Any] | None, str | None]:
    """POST MiniMax ``/v1/get_voice``；返回 ``(数据, 归因后的错误文案)``。

    抽成独立函数便于测试替换（真身要在测里发真请求）。
    """
    url = join_provider_url(base_url or "", "/v1", "/get_voice")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            # ⚠️ 发厂商的请求体：紧凑分隔符（对齐 JSON.stringify）
            content=json.dumps({"voice_type": "all"}, separators=(",", ":")),
        )
    if response.status_code >= 400:
        return None, await vendor_response_error(response, "audio")
    return response.json(), None


@router.post("/sync")
async def sync_voices(conn: Connection = Depends(get_tx)):
    """从 MiniMax ``get_voice`` 同步系统音色（先清空旧数据再批量插入 + AI 打标）。"""
    try:
        rows = conn.execute(
            select(ai_service_configs).where(ai_service_configs.c.service_type == "audio")
        ).all()
        candidates = [row for row in rows if row.is_active and row.provider == "minimax"]
        if not candidates:
            return bad_request("No active minimax audio config found")

        config = candidates[0]
        if not config.api_key:
            return bad_request("MiniMax API key not configured")

        result, fetch_error = await fetch_minimax_voices(config.base_url or "", config.api_key)
        if fetch_error is not None:
            return bad_request(fetch_error)

        result = result or {}
        base_resp = result.get("base_resp") or {}
        if base_resp.get("status_code") != 0:
            return bad_request(base_resp.get("status_msg") or "Failed to fetch voices")

        voices = [
            voice for voice in (result.get("system_voice") or [])
            if should_keep_voice(str(voice.get("voice_id", "")), str(voice.get("voice_name", "")))
        ]
        ts = now()

        # 先清空旧数据
        conn.execute(sql_delete(ai_voices).where(ai_voices.c.provider == "minimax"))

        insert_rows = [
            {
                "voice_id": voice.get("voice_id"),
                "voice_name": voice.get("voice_name"),
                "description": json.dumps(voice.get("description") or [], ensure_ascii=False,
                                          separators=(",", ":")),
                "language": extract_language(str(voice.get("voice_id", "")),
                                             str(voice.get("voice_name", ""))),
                "provider": "minimax",
                "created_at": ts,
            }
            for voice in voices
        ]

        if insert_rows:
            conn.execute(ai_voices.insert(), insert_rows)

            # 批量 AI 打标（旁白/主角/反派/配角）；失败不阻塞 sync，role_tags 留空由前端回退正则
            try:
                tags = await infer_voice_role_tags([
                    {
                        "voiceId": row["voice_id"],
                        "voiceName": row["voice_name"],
                        "description": json.loads(row["description"] or "[]"),
                    }
                    for row in insert_rows
                ])
                for voice_id, tag_list in (tags or {}).items():
                    conn.execute(
                        update(ai_voices).where(ai_voices.c.voice_id == voice_id).values(
                            role_tags=json.dumps(tag_list, ensure_ascii=False, separators=(",", ":"))
                        )
                    )
            except Exception as err:  # noqa: BLE001
                log_task_error("AiVoices", "tag-voices", {"error": str(err)})

        return success({
            "count": len(insert_rows),
            "message": f"Synced {len(insert_rows)} voices",
        })
    except Exception as err:  # noqa: BLE001
        log_task_error("AiVoices", "sync-minimax", {"error": str(err)})
        return bad_request(str(err) or "Failed to sync voices")
