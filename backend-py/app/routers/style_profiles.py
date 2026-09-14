"""``/api/v1/style-profiles`` —— 与 ``backend/src/routes/style-profiles.ts`` 对齐（7 / 8 端点）。

**已迁移 7 个**：列表 / 详情 / 创建 / 更新 / 激活 / 写入提炼结果 / 软删
**未迁移 1 个**：``POST /:id/distill``（LLM 分析 + ffprobe，属媒体/Agent 域 ⇒ 走反代）

⚠️ 两处 ``Number(param)`` **不带**有限性校验（与 ``parseParamId`` 不同）：
``Number("abc")`` 得到 NaN，JS 侧会一路带到 SQL 查出空结果 ⇒ 返回
**404 ``Profile not found``**（不是 ``Invalid id``）。这里对 NaN 直接判空，可观察行为一致。

⚠️ 请求体混用 snake/camel：**入参一律 snake_case**（``drama_id`` / ``shot_patterns``），
**出参一律 camelCase**（``dramaId`` / ``shotPatterns``）。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.engine import Connection

from ..db import get_conn, get_tx
from ..response import bad_request, js_number, js_truthy, not_found, success
from ..services.style_profiles import (
    UNSET,
    activate_style_profile,
    apply_distill_result,
    create_style_profile,
    delete_style_profile,
    distill_style_profile,
    get_style_profile,
    list_style_profiles,
    update_style_profile,
)
from ..services.task_logger import log_task_error

router = APIRouter(prefix="/api/v1/style-profiles", tags=["style-profiles"])


def _num(raw: str | None) -> float | None:
    """``Number(raw)``：非有限值返回 None（调用方据此判「查不到」）。"""
    if raw is None or raw == "":
        return None
    value = js_number(raw)
    return None if value is None else value


@router.get("")
def list_profiles(request: Request, conn: Connection = Depends(get_conn)):
    try:
        raw = request.query_params.get("drama_id")
        drama_id = js_number(raw) if js_truthy(raw) else None
        return success({"profiles": list_style_profiles(conn, drama_id)})
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc) or "list style profiles failed")


@router.get("/{profile_id}")
def get_profile(profile_id: str, conn: Connection = Depends(get_conn)):
    profile = get_style_profile(conn, _num(profile_id))
    if profile is None:
        return not_found("Profile not found")
    return success({"profile": profile})


@router.post("")
def create_profile(body: dict[str, Any], conn: Connection = Depends(get_tx)):
    try:
        new_id = create_style_profile(
            conn,
            name=body.get("name"),
            drama_id=body.get("drama_id"),
            description=body.get("description"),
            source=body.get("source"),
            storytelling=body.get("storytelling"),
            shot_patterns=body.get("shot_patterns"),
            audio_captions=body.get("audio_captions"),
            qc_rules=body.get("qc_rules"),
            preferences=body.get("preferences"),
        )
        if new_id is None:
            return bad_request("create failed")
        return success({"id": new_id, "profile": get_style_profile(conn, new_id)})
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc) or "create style profile failed")


@router.put("/{profile_id}")
def update_profile(profile_id: str, body: dict[str, Any], conn: Connection = Depends(get_tx)):
    try:
        pid = _num(profile_id)
        # `=== undefined ? undefined : value` ⇒ 只有**键存在**才进 patch（显式 null 要落 null）
        patch_input: dict[str, Any] = {}
        for key, column in (
            ("name", "name"),
            ("description", "description"),
            ("source", "source"),
            ("storytelling", "storytelling"),
            ("shot_patterns", "shot_patterns"),
            ("audio_captions", "audio_captions"),
            ("qc_rules", "qc_rules"),
            ("preferences", "preferences"),
        ):
            if key in body:
                patch_input[column] = body[key]

        if not update_style_profile(conn, pid, patch_input):
            return not_found("Profile not found")
        return success({"profile": get_style_profile(conn, pid)})
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc) or "update style profile failed")


@router.post("/{profile_id}/activate")
def activate_profile(profile_id: str, conn: Connection = Depends(get_tx)):
    profile = activate_style_profile(conn, _num(profile_id))
    if profile is None:
        return not_found("Profile not found")
    return success({"profile": profile})


@router.post("/{profile_id}/distill")
async def distill_profile(profile_id: str, conn: Connection = Depends(get_tx)):
    """用 LLM 分析参考素材，提炼 house style（**不落库**，返回待用户确认的结果）。"""
    try:
        # ⚠️ 原 TS 是裸 ``Number(id)``（``'abc'`` -> NaN）⇒ Python 用 ``_num`` 给 None，
        #    两者都会在 ``get_style_profile`` 里「查不到」而返回同一句 Profile not found。
        result = await distill_style_profile(conn, _num(profile_id))
        if not result.get("ok"):
            return bad_request(result.get("error") or "distill failed")
        return success({"result": result.get("result")})
    except Exception as exc:  # noqa: BLE001
        log_task_error("StyleProfilesAPI", "distill", {"error": str(exc)})
        return bad_request(str(exc) or "distill failed")


@router.post("/{profile_id}/apply")
def apply_profile(profile_id: str, body: dict[str, Any], conn: Connection = Depends(get_tx)):
    """把（Node 侧提炼并人工确认后的）结果写入 Profile。

    ``body.x || 默认值``：JS 里空 ``{}`` / ``[]`` 是**真值** ⇒ 用 ``js_truthy`` 判断，
    否则显式传空对象会被误判成「没传」。
    """
    try:
        pid = _num(profile_id)
        result = {
            "storytelling": body["storytelling"] if js_truthy(body.get("storytelling")) else {},
            "shot_patterns": body["shot_patterns"] if js_truthy(body.get("shot_patterns")) else {},
            "audio_captions": body["audio_captions"] if js_truthy(body.get("audio_captions")) else {},
            "qc_rules": body["qc_rules"] if js_truthy(body.get("qc_rules")) else {},
            "facts": body["facts"] if js_truthy(body.get("facts")) else [],
            "inferences": body["inferences"] if js_truthy(body.get("inferences")) else [],
            "preferences": body["preferences"] if js_truthy(body.get("preferences")) else [],
        }
        if not apply_distill_result(conn, pid, result):
            return not_found("Profile not found")
        return success({"profile": get_style_profile(conn, pid)})
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc) or "apply distill result failed")


@router.delete("/{profile_id}")
def remove_profile(profile_id: str, conn: Connection = Depends(get_tx)):
    if not delete_style_profile(conn, _num(profile_id)):
        return not_found("Profile not found")
    return success({"deleted": True})


#: 供类型检查器引用（``UNSET`` 由 service 导出，这里显式再导出便于调用方统一入口）
__all__ = ["router", "UNSET"]
