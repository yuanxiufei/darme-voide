"""风格 Profile CRUD + 提炼 —— 移植 ``backend/src/services/style-profiles.ts``（307 行，**整域关闭**）。

从参考素材提炼可复用的 house style，分四类规则：``storytelling``（叙事节奏）、
``shot_patterns``（景别/机位/运镜）、``audio_captions``（音效/配乐/字幕）、``qc_rules``（验收标准）。
来源三分类：measurement facts（客观测量）/ visual inference（模型推断）/ user preference（用户偏好）。

⚠️ 提炼（``distill_style_profile``）**不落库**：返回结果等用户确认，再由 ``/apply`` 写入
   （对齐 H3-Codex-Drama 的 user confirmation gate）。

⚠️ 与 TS 版的两处**有意差异**（都是「用仓内既有链路替代外部依赖」）：

1. TS 用 ``@mastra`` 的 ``Agent`` + ``@ai-sdk`` 的 ``createOpenAI`` **直连** provider
   （绕过自家 adapter）；Python 侧统一走 ``text_generation.generate_text``
   —— 它自带多模型 fallback，且是全仓「直连 LLM」的既定入口 ⇒ 行为等价、错误面更小。
2. TS 用 ``fluent-ffmpeg`` 的 ``ffprobe`` 探测；Python 侧直接起 ``ffprobe`` 进程
   （与 ``frame_extractor`` / ``qc_report`` 同一套做法）。

⚠️ 行形状是 **camelCase**（原 TS 用 ``mapRow`` 显式改名，返回的不是 drizzle 原始行）。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from sqlalchemy import and_, select, update
from sqlalchemy.engine import Connection

from ..core.models import style_profiles
from ..core.response import js_round, js_truthy, now, row_to_dict
from .file_storage import get_absolute_path
from .task_logger import log_task_error

#: 五个 JSON 列的列名（原 TS 里 ``preferences`` 与其余四个一样走 JSON.stringify）
_JSON_FIELDS = ("storytelling", "shot_patterns", "audio_captions", "qc_rules", "preferences")
_PLAIN_FIELDS = ("name", "description", "source")

#: 内部哨兵：区分「调用方没传这个键」与「传了 null」（对齐 TS 的 ``undefined``）
UNSET: Any = object()


def _dump_json(value: Any) -> str | None:
    """对齐 ``value ? JSON.stringify(value) : null`` —— 注意用 ``js_truthy``，
    显式传入的空 ``{}`` / ``[]`` 在 JS 里是真值，必须落成 ``"{}"`` / ``"[]"`` 而不是 null。"""
    if not js_truthy(value):
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _map_row(row: Any) -> dict[str, Any]:
    """对齐 TS 的 ``mapRow``：列名 → camelCase，``isActive`` 转真布尔。"""
    d = row_to_dict(row)
    return {
        "id": d["id"],
        "dramaId": d["drama_id"],
        "name": d["name"],
        "description": d["description"],
        "source": d["source"],
        "storytelling": d["storytelling"],
        "shotPatterns": d["shot_patterns"],
        "audioCaptions": d["audio_captions"],
        "qcRules": d["qc_rules"],
        "facts": d["facts"],
        "inferences": d["inferences"],
        "preferences": d["preferences"],
        "isActive": bool(d["is_active"]),
        "createdAt": d["created_at"],
        "updatedAt": d["updated_at"],
        "deletedAt": d["deleted_at"],
    }


def list_style_profiles(conn: Connection, drama_id: Any = None) -> list[dict[str, Any]]:
    """列表（激活的排前面）。

    ⚠️ ``drama_id`` 走 JS 真值判断：``0`` / ``NaN`` 都视为「不过滤」
    （原 TS 是 ``dramaId ? eq(...) : undefined``，而 NaN 在 JS 里是假值）。
    """
    conds = [style_profiles.c.deleted_at.is_(None)]
    if js_truthy(drama_id):
        conds.append(style_profiles.c.drama_id == drama_id)

    rows = conn.execute(select(style_profiles).where(and_(*conds))).all()
    mapped = [_map_row(r) for r in rows]
    # TS 的比较器 (a.isActive?-1:1)-(b.isActive?-1:1) 等价于「激活的排前，其余保持原序」
    return sorted(mapped, key=lambda p: 0 if p["isActive"] else 1)


def get_active_profile_for_drama(conn: Connection, drama_id: Any = None) -> dict[str, Any] | None:
    """当前生效的风格 Profile（对齐 TS ``getActiveProfileForDrama``）。

    ⚠️ **两级回退**，顺序不能反：先找**本剧专属**且激活的；找不到再找**全局**
    （``drama_id IS NULL``）且激活的。只挑其中一份，不合并。
    与 ``list_style_profiles`` **不同**（那个是「列全部、激活的排前」，没有全局兜底）。
    """
    if drama_id:
        row = conn.execute(
            select(style_profiles).where(and_(
                style_profiles.c.drama_id == drama_id,
                style_profiles.c.is_active.is_(True),
                style_profiles.c.deleted_at.is_(None),
            ))
        ).first()
        if row is not None:
            return _map_row(row)
    global_row = conn.execute(
        select(style_profiles).where(and_(
            style_profiles.c.drama_id.is_(None),
            style_profiles.c.is_active.is_(True),
            style_profiles.c.deleted_at.is_(None),
        ))
    ).first()
    return _map_row(global_row) if global_row is not None else None


def get_style_profile(conn: Connection, profile_id: Any) -> dict[str, Any] | None:
    if profile_id is None or profile_id != profile_id:  # None / NaN
        return None
    row = conn.execute(
        select(style_profiles).where(
            and_(style_profiles.c.id == profile_id, style_profiles.c.deleted_at.is_(None))
        )
    ).first()
    return _map_row(row) if row is not None else None


def create_style_profile(
    conn: Connection,
    *,
    name: Any = None,
    drama_id: Any = None,
    description: Any = None,
    source: Any = None,
    storytelling: Any = None,
    shot_patterns: Any = None,
    audio_captions: Any = None,
    qc_rules: Any = None,
    preferences: Any = None,
) -> int | None:
    """创建（失败返回 None —— 路由据此回 400 ``create failed``）。

    注：``name`` 缺失时按原 TS 一样不补默认值 ⇒ 触发 NOT NULL 约束 ⇒ 返回 None ⇒ 400，
    与 Node 的可观察行为一致。
    """
    ts = now()
    try:
        result = conn.execute(
            style_profiles.insert().values(
                drama_id=None if drama_id is None else drama_id,
                name=name,
                description=None if description is None else description,
                source=None if source is None else source,
                storytelling=_dump_json(storytelling),
                shot_patterns=_dump_json(shot_patterns),
                audio_captions=_dump_json(audio_captions),
                qc_rules=_dump_json(qc_rules),
                preferences=_dump_json(preferences),
                is_active=False,
                created_at=ts,
                updated_at=ts,
            )
        )
        return int(result.inserted_primary_key[0])
    except Exception:  # noqa: BLE001 - 与原 TS 一致：失败仅告警
        return None


def update_style_profile(conn: Connection, profile_id: Any, patch_input: dict[str, Any]) -> bool:
    """更新。``patch_input`` 里**没有的键**视为 ``undefined``（不动）；
    值为 ``null`` 则按原逻辑落成 null。"""
    if get_style_profile(conn, profile_id) is None:
        return False

    patch: dict[str, Any] = {"updated_at": now()}
    for key in _JSON_FIELDS:
        value = patch_input.get(key, UNSET)
        if value is UNSET:
            continue
        patch[key] = _dump_json(value)
    for key in _PLAIN_FIELDS:
        value = patch_input.get(key, UNSET)
        if value is UNSET:
            continue
        patch[key] = value

    conn.execute(update(style_profiles).where(style_profiles.c.id == profile_id).values(**patch))
    return True


def delete_style_profile(conn: Connection, profile_id: Any) -> bool:
    """软删除（同时取消激活）。"""
    if get_style_profile(conn, profile_id) is None:
        return False
    conn.execute(
        update(style_profiles)
        .where(style_profiles.c.id == profile_id)
        .values(deleted_at=now(), is_active=False)
    )
    return True


def activate_style_profile(conn: Connection, profile_id: Any) -> dict[str, Any] | None:
    """激活（**同 drama 内仅一个**；未绑定 drama 的全局 Profile 不受此约束）。"""
    profile = get_style_profile(conn, profile_id)
    if profile is None:
        return None

    if js_truthy(profile["dramaId"]):
        conn.execute(
            update(style_profiles)
            .where(
                and_(
                    style_profiles.c.drama_id == profile["dramaId"],
                    style_profiles.c.is_active.is_(True),
                )
            )
            .values(is_active=False)
        )

    conn.execute(
        update(style_profiles)
        .where(style_profiles.c.id == profile_id)
        .values(is_active=True, updated_at=now())
    )
    return get_style_profile(conn, profile_id)


def apply_distill_result(conn: Connection, profile_id: Any, result: dict[str, Any]) -> bool:
    """把（用户确认后的）提炼结果写入 Profile。

    ⚠️ 与原 TS 一致：这里五个键**总是**写入（不做 undefined 判断），
    所以 ``facts`` / ``inferences`` 两列**不在**写入范围内（它们由提炼流程单独维护）。
    """
    return update_style_profile(
        conn,
        profile_id,
        {
            "storytelling": result.get("storytelling", {}),
            "shot_patterns": result.get("shot_patterns", {}),
            "audio_captions": result.get("audio_captions", {}),
            "qc_rules": result.get("qc_rules", {}),
            "preferences": result.get("preferences", []),
        },
    )


# ===========================================================================
# 提炼（对齐 H3-Codex-Drama distill_house_style 的 LLM 分析）
# ===========================================================================

#: 提炼用的系统提示词（**逐字**对齐 TS 的 ``instructions``）
_DISTILL_INSTRUCTIONS = """你是资深影视风格分析专家。根据用户提供的参考素材描述，提炼可复用的「house style」短片风格档案。

必须严格输出 JSON（不要包含任何其他文字、markdown 代码块或注释），结构如下：
{
  "storytelling": { 叙事节奏、悬念密度、转场动机偏好、信息揭示节奏 },
  "shot_patterns": { 景别分布偏好、机位、运镜、构图习惯、镜头时长规律 },
  "audio_captions": { 配乐风格、音效密度、字幕风格、静场偏好 },
  "qc_rules": { 验收时的画面/音频/连续性硬性标准，如「同场景相邻镜头相似度须≥0.55」「响度 I=-14 LUFS」 },
  "facts": ["可测量的客观事实，来自参考素材本身，如分辨率/时长/镜头数等"],
  "inferences": ["你从素材风格推断出的结论（可能是主观判断，需人工复核）"],
  "preferences": ["用户明确表达的风格偏好（最高优先级）"]
}

原则：
- facts 只放可验证的客观测量；inferences 是推断；preferences 来自用户原话的偏好
- qc_rules 要具体到可执行数值/判定标准
- 中文输出，JSON 字段名保持英文"""


def _int_if_integral(value: Any) -> Any:
    """ffprobe 给的是整数语义的数值 —— 落成 int，避免 JSON 里出现 ``340000.0``。"""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


async def _probe_measurement_facts(source: str | None) -> dict[str, Any]:
    """用 ffprobe 探测参考视频的**测量事实**（可选，失败静默返回 ``{}``）。

    ⚠️ 与 ``qc_report.ffprobe_info`` 的形状**不同**（那份是归一化后的全字段），
    这份是「有才写」的稀疏事实，且 ``fps`` 保留 ffprobe 的**原始分数串**（如 ``30000/1001``）
    —— 给 LLM 看的是事实原文，不做换算，照抄 TS。
    """
    if not source:
        return {}
    facts: dict[str, Any] = {}
    try:
        absolute = get_absolute_path(source)
        process = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", absolute,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await process.communicate()
        if process.returncode != 0:
            return {}
        meta = json.loads(stdout.decode("utf-8", errors="replace") or "")
    except Exception:  # noqa: BLE001 —— 非本地文件/探测失败时忽略（与 TS 的裸 catch 等价）
        return {}
    if not isinstance(meta, dict):
        return {}

    streams = meta.get("streams") if isinstance(meta.get("streams"), list) else []
    video = next((s for s in streams
                  if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    audio = next((s for s in streams
                  if isinstance(s, dict) and s.get("codec_type") == "audio"), None)
    if video:
        facts["resolution"] = f"{video.get('width')}x{video.get('height')}"
        facts["fps"] = video.get("r_frame_rate") or video.get("avg_frame_rate") or None
        facts["video_codec"] = video.get("codec_name") or None
    if audio:
        facts["audio_codec"] = audio.get("codec_name") or None

    fmt = meta.get("format") if isinstance(meta.get("format"), dict) else {}
    if fmt.get("duration"):
        facts["duration_seconds"] = js_round(float(fmt["duration"]))
    if fmt.get("size"):
        facts["file_size_bytes"] = _int_if_integral(float(fmt["size"]))
    return facts


async def distill_style_profile(conn: Connection, profile_id: Any) -> dict[str, Any]:
    """用文本 LLM 分析参考素材，提炼 house style（四类规则 + 三分类来源）。

    **不修改数据**，返回 ``{"ok": bool, "result"?: ..., "error"?: str}`` 供用户确认后写入。
    ⚠️ 与原 TS 一致：**所有异常都吞成 ``{"ok": False, "error": ...}``**（路由据此回 400）。
    """
    profile = get_style_profile(conn, profile_id)
    if profile is None:
        return {"ok": False, "error": "Profile not found"}

    try:
        measurements = await _probe_measurement_facts(profile.get("source"))

        user_msg = "\n\n".join([part for part in [
            f"参考素材说明：{profile.get('source') or '（未提供，仅凭 Profile 名称/描述）'}",
            f"Profile 名称：{profile.get('name')}",
            f"描述：{profile['description']}" if profile.get("description") else "",
            # ⚠️ indent=2 是**有意**的：对齐 TS 的 `JSON.stringify(measurements, null, 2)`
            (f"参考素材测量事实（ffprobe 探测，可信）："
             f"{json.dumps(measurements, ensure_ascii=False, indent=2)}") if measurements else "",
            f"已知用户偏好（若有）：{profile.get('preferences') or '（无）'}",
        ] if part])

        from .text_generation import generate_text  # noqa: PLC0415 —— 惰性导入避免环

        text = await generate_text(conn, user_msg, {"system": _DISTILL_INSTRUCTIONS})
        cleaned = re.sub(r"^```(?:json)?\s*", "", text or "", flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < 0:
            return {"ok": False, "error": "LLM output is not JSON"}

        parsed = json.loads(cleaned[start:end + 1])
        if not isinstance(parsed, dict):
            return {"ok": False, "error": "LLM output is not JSON"}

        def _obj(key: str) -> dict[str, Any]:
            value = parsed.get(key)
            return value if isinstance(value, dict) else {}

        def _list(key: str) -> list[Any]:
            value = parsed.get(key)
            return value if isinstance(value, list) else []

        return {"ok": True, "result": {
            "storytelling": _obj("storytelling"),
            "shot_patterns": _obj("shot_patterns"),
            "audio_captions": _obj("audio_captions"),
            "qc_rules": _obj("qc_rules"),
            "facts": _list("facts"),
            "inferences": _list("inferences"),
            "preferences": _list("preferences"),
        }}
    except Exception as exc:  # noqa: BLE001
        log_task_error("StyleProfile", "distill-failed",
                       {"profileId": profile_id, "error": str(exc)})
        return {"ok": False, "error": str(exc)}
