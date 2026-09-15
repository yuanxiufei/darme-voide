"""characters 域 —— 与 ``backend/src/routes/characters.ts`` 对齐。**12 端点全迁完**。

======  ==============================  ==================================================
方法    路径                            说明
======  ==============================  ==================================================
GET     ``/{id}``                       详情
PUT     ``/{id}``                       更新（35 字段白名单 + 音色变更作废试听）
POST    ``/{id}/merge``                 跨集一致性修复：source 并入 target
DELETE  ``/{id}``                       软删
POST    ``/{id}/auto-split-visuals``    智能拆分外貌 → 服装/武器/首饰（LLM）
POST    ``/{id}/generate-voice-sample`` TTS 试听音频
POST    ``/{id}/generate-image``        角色立绘（纯文生图基线，仅显式参考图）
POST    ``/{id}/generate-prompt``       预览提示词（**不落库**）
POST    ``/{id}/generate-three-views``  三视图合成**一张横向长图**
POST    ``/{id}/generate-equip-image``  装备设定图（view 三视角 / single 单件道具）
POST    ``/{id}/generate-expressions``  表情头像组（批量，逐个独立成败）
POST    ``/batch-generate-images``      按集批量出角色立绘
======  ==============================  ==================================================

⚠️ **本域返回 camelCase**（对齐 drizzle 行的 JS 属性名），不是 dramas/episodes 那种 snake_case。
见 ``response.dict_to_camel`` 的说明。

⚠️ 六处极易抄错的地方（都有用例锁）：

1. **id 非法一律 404 ``Invalid character id``**，但**找不到角色**时是 **400 ``Character not found``**
   （除 ``auto-split-visuals`` 是 404 —— 原 TS 就这样，不要统一）；
2. **错误处理不统一**：多数端点 `try/catch → 400`；``generate-three-views`` **没有 try/catch**
   （异常直接 500）；``batch-generate-images`` 是**逐条 `catch {}` 静默跳过**；
3. **配置回退不同**：立绘/三视图/装备/表情走 ``ep.imageConfigId ?? resolveDramaConfigId(...)``，
   而 **批量接口只用 ``ep.imageConfigId``**（不回退到 drama 级）；
4. **自定义 prompt 也要收口画风**：``resolveCharacterPrompt`` 会给手写 prompt 也追加画风后缀，
   并强制注入「智能拆分」的服装/武器/首饰子句（否则自定义 prompt 会丢信息、画风漂移）；
5. **视觉锚定**（``resolveCharacterAnchors``）：立绘只锚主立绘；三视图/装备/表情优先锚
   「三视图 combined → 主立绘」，``anchor='none'`` 不锚，显式 ``reference_images`` 最高优先（最多 6 张）；
6. **装备图的负向词是嵌套三元**：单件模式用物品负向；三视图模式若用户负向**缺人物排除词**
   （或命中旧版提示词）就**强制换成**装备三视图负向，防止装备图里出现人物。
"""
from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import and_, select, update
from sqlalchemy.engine import Connection

from ..core.db import get_conn, get_tx
from ..core.models import (
    asset_versions,
    characters,
    episode_characters,
    episodes,
    image_generations,
    storyboard_characters,
)
from ..core.request_utils import read_json
from ..core.response import (
    bad_request,
    camel_to_snake,
    js_number,
    not_found,
    now,
    parse_param_id,
    row_to_camel,
    success,
)
from ..services.bible_ids import ensure_costume_id
from ..services.image_generation import generate_image
from ..services.prompt_utils import (
    EXPRESSION_PRESETS,
    ITEM_IMAGE_SIZE,
    THREE_VIEW_COMBINED_LAYOUT,
    THREE_VIEW_SIZE,
    build_character_art_style_suffix,
    build_character_image_prompt,
    build_character_negative_prompt,
    build_character_visuals_clause,
    build_equip_art_style_suffix,
    build_equip_image_prompt,
    build_equip_negative,
    build_expression_image_prompt,
    build_expression_negative,
    build_item_image_prompt,
    build_item_negative,
    build_three_view_negative,
    find_expression_preset,
    get_drama_art_style,
    resolve_effective_art_style,
)
from ..services.task_logger import log_task_error, log_task_start, log_task_success
from ..services.text_generation import split_character_visuals
from ..services.tts_generation import generate_voice_sample

router = APIRouter(prefix="/api/v1/characters", tags=["characters"])

# PUT /characters/:id 白名单（对齐 TS 的 key 列表，camelCase 原样保留以便双写兼容）
_UPDATE_KEYS = [
    "name", "role", "roleType", "description", "appearance", "personality",
    "voiceStyle", "voiceProvider", "voiceSpeed", "voiceEmotion",
    "voicePitch", "voiceModel", "clothing", "weapons", "accessories",
    "customPrompt", "negativePrompt", "style", "coreFeatures", "costumes",
    "variations", "threeViews", "imageUrl", "localPath", "referenceImages",
    "expressions", "itemImages",
    "clothingPrompt", "clothingNegativePrompt",
    "weaponPrompt", "weaponNegativePrompt",
    "accessoryPrompt", "accessoryNegativePrompt",
]

#: merge 时可回填的资产字段：(DB 列名, TS 属性名, 中文标签)
#: 中文标签保留原样 —— TS 会把 `标签=属性名` 拼进响应体的 filled 数组，前端可能展示。
_MERGE_FILLABLE: list[tuple[str, str, str]] = [
    ("role", "role", "定位"),
    ("role_type", "roleType", "角色类型"),
    ("description", "description", "人物简介"),
    ("appearance", "appearance", "外貌特征"),
    ("personality", "personality", "性格"),
    ("clothing", "clothing", "服装"),
    ("weapons", "weapons", "武器"),
    ("costumes", "costumes", "服装设定集"),
    ("variations", "variations", "变体"),
    ("accessories", "accessories", "首饰"),
    ("three_views", "threeViews", "三视图"),
    ("equip_images", "equipImages", "装备图"),
    ("item_images", "itemImages", "道具图"),
    ("expressions", "expressions", "表情头像组"),
    ("custom_prompt", "customPrompt", "自定义提示词"),
    ("negative_prompt", "negativePrompt", "负面提示词"),
    ("style", "style", "画风"),
    ("image_url", "imageUrl", "形象图"),
    ("local_path", "localPath", "形象图路径"),
    ("reference_images", "referenceImages", "参考图集"),
    ("seed_value", "seedValue", "种子值"),
    ("voice_style", "voiceStyle", "音色"),
    ("voice_provider", "voiceProvider", "音色服务商"),
    ("voice_model", "voiceModel", "音色模型"),
    ("voice_speed", "voiceSpeed", "语速"),
    ("voice_emotion", "voiceEmotion", "情感"),
    ("voice_pitch", "voicePitch", "音调"),
    ("voice_sample_url", "voiceSampleUrl", "试听音频"),
    ("speaker_id", "speakerId", "说话人 ID"),
]


def _fetch_character(conn: Connection, character_id: int | float):
    return conn.execute(
        select(characters).where(
            and_(characters.c.id == character_id, characters.c.deleted_at.is_(None))
        )
    ).first()


# ---------------------------------------------------------------------------
# GET /{id}
# ---------------------------------------------------------------------------

@router.get("/{character_id}")
def get_character(character_id: str, conn: Connection = Depends(get_conn)):
    try:
        cid = parse_param_id(character_id)
        if cid is None:
            return not_found("Invalid character id")
        row = _fetch_character(conn, cid)
        if row is None:
            return not_found("Character not found")
        return success(row_to_camel(row, "characters"))
    except Exception as exc:  # noqa: BLE001
        # TS 此处是 {code:500, data:null, message}
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


# ---------------------------------------------------------------------------
# PUT /{id}
# ---------------------------------------------------------------------------

@router.put("/{character_id}")
async def update_character(character_id: str, request: Request, conn: Connection = Depends(get_tx)):
    try:
        cid = parse_param_id(character_id)
        if cid is None:
            return not_found("Invalid character id")

        body = await read_json(request)
        updates: dict[str, Any] = {"updated_at": now()}
        for key in _UPDATE_KEYS:
            snake = camel_to_snake(key)
            if snake in body:
                updates[snake] = body[snake]
            elif key in body:
                # ⚠️ 值取自 camelCase 键，但**列名必须写成 snake_case**。
                # TS 里 `updates[key] = body[key]` 的 key 是 drizzle **属性名**（camelCase），
                # 直接照抄到 Python 会让 SQLAlchemy 收到不存在的列名：
                # 「Unconsumed column names: voiceStyle」（本域实测踩到过）。
                updates[snake] = body[key]

        # 音色变更 ⇒ 旧试听文件作废（强制重新生成）。
        # 注意：即使 voice_style 不在白名单里也会被这条规则命中，与 TS 一致。
        if "voice_style" in body or "voiceStyle" in body:
            updates["voice_sample_url"] = None

        conn.execute(
            update(characters)
            .where(and_(characters.c.id == cid, characters.c.deleted_at.is_(None)))
            .values(**updates)
        )
        ensure_costume_id(conn, int(cid))

        row = _fetch_character(conn, cid)
        return success(row_to_camel(row, "characters") if row is not None else None)
    except Exception as exc:  # noqa: BLE001
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# POST /{id}/merge — 跨集一致性修复
# ---------------------------------------------------------------------------

@router.post("/{character_id}/merge")
async def merge_character(character_id: str, request: Request, conn: Connection = Depends(get_tx)):
    """把「当前角色(source)」合并进「全剧已有角色(target)」。

    语义（照抄 TS 注释）：target 为主角色；source 中 target 为空的资产字段回填
    （**不覆盖** target 既有形象/声线）；source 的全部关联迁移到 target；最后 source 软删。
    """
    try:
        cid = parse_param_id(character_id)
        if cid is None:
            return not_found("Invalid character id")

        body = await read_json(request)
        # TS: Number(body.target_id) —— NaN/0 都会被 `!targetId` 拦下
        try:
            target_id: int | float | None = int(float(body.get("target_id")))
        except (TypeError, ValueError):
            target_id = None
        if not target_id or target_id == cid:
            return bad_request("target_id 必填且不能是自身")

        source = _fetch_character(conn, cid)
        target = _fetch_character(conn, target_id)
        if source is None or target is None:
            return not_found("角色不存在")
        if source.drama_id != target.drama_id:
            return bad_request("只能合并同一部剧内的角色")

        ts = now()

        # 1) 资产字段回填：target 为空才从 source 补
        backfill: dict[str, Any] = {}
        filled: list[str] = []
        for column, js_key, label in _MERGE_FILLABLE:
            sv = getattr(source, column)
            tv = getattr(target, column)
            tv_empty = tv is None or tv == ""
            if tv_empty and sv is not None and sv != "":
                backfill[column] = sv
                filled.append(f"{label}={js_key}")
        if backfill:
            conn.execute(
                update(characters)
                .where(characters.c.id == target_id)
                .values(**backfill, updated_at=ts)
            )
        ensure_costume_id(conn, int(target_id))

        # 2) episode 绑定迁移：该集已有 target 则删掉 source 行，否则改挂 target
        src_ep_links = conn.execute(
            select(episode_characters.c.id, episode_characters.c.episode_id).where(
                episode_characters.c.character_id == cid
            )
        ).all()
        for link in src_ep_links:
            dup = conn.execute(
                select(episode_characters.c.id).where(
                    and_(
                        episode_characters.c.episode_id == link.episode_id,
                        episode_characters.c.character_id == target_id,
                    )
                )
            ).first()
            if dup is not None:
                conn.execute(
                    episode_characters.delete().where(episode_characters.c.id == link.id)
                )
            else:
                conn.execute(
                    update(episode_characters)
                    .where(episode_characters.c.id == link.id)
                    .values(character_id=target_id)
                )

        # 3) 分镜绑定迁移：同一分镜同时绑了 target 时删除 source 行，否则改挂
        src_sb_links = conn.execute(
            select(storyboard_characters.c.storyboard_id).where(
                storyboard_characters.c.character_id == cid
            )
        ).all()
        for link in src_sb_links:
            dup = conn.execute(
                select(storyboard_characters.c.character_id).where(
                    and_(
                        storyboard_characters.c.storyboard_id == link.storyboard_id,
                        storyboard_characters.c.character_id == target_id,
                    )
                )
            ).first()
            if dup is not None:
                conn.execute(
                    storyboard_characters.delete().where(
                        and_(
                            storyboard_characters.c.storyboard_id == link.storyboard_id,
                            storyboard_characters.c.character_id == cid,
                        )
                    )
                )
            else:
                conn.execute(
                    update(storyboard_characters)
                    .where(
                        and_(
                            storyboard_characters.c.storyboard_id == link.storyboard_id,
                            storyboard_characters.c.character_id == cid,
                        )
                    )
                    .values(character_id=target_id)
                )

        # 4a) 生成历史归属迁移
        conn.execute(
            update(image_generations)
            .where(image_generations.c.character_id == cid)
            .values(character_id=target_id)
        )

        # 4b) 资产版本：asset_versions 对 (asset_type, asset_id, media_type, frame_type) 没有唯一约束，
        # 直接改挂会让 source 的 v1/v2 与 target 的 v1/v2 撞号、且同组出现多条 status='current'
        # ⇒ 前端版本列表重号、「回滚到 vN」语义歧义。故按 (media_type, frame_type) 分组把 source
        # 的版本号**接续到 target 之后**，并入的历史统一置 historical；target 该分组原本没有
        # current 时，保留并入的最新一条为 current。
        src_versions = conn.execute(
            select(asset_versions).where(
                and_(asset_versions.c.asset_type == "character", asset_versions.c.asset_id == cid)
            )
        ).all()
        if src_versions:
            tgt_versions = conn.execute(
                select(asset_versions).where(
                    and_(
                        asset_versions.c.asset_type == "character",
                        asset_versions.c.asset_id == target_id,
                    )
                )
            ).all()

            def group_key(media_type: str | None, frame_type: str | None) -> str:
                # TS: `${v.mediaType}::${v.frameType ?? ''}` —— 空值合并，不是逻辑或
                return f"{media_type}::{frame_type if frame_type is not None else ''}"

            next_by_group: dict[str, int] = {}
            current_groups: set[str] = set()
            for v in tgt_versions:
                k = group_key(v.media_type, v.frame_type)
                next_by_group[k] = max(next_by_group.get(k, 0), v.version)
                if v.status == "current":
                    current_groups.add(k)

            sorted_src = sorted(src_versions, key=lambda v: v.version)
            max_by_group: dict[str, int] = {}
            for v in sorted_src:
                k = group_key(v.media_type, v.frame_type)
                max_by_group[k] = max(max_by_group.get(k, 0), v.version)

            for v in sorted_src:
                k = group_key(v.media_type, v.frame_type)
                nxt = next_by_group.get(k, 0) + 1
                next_by_group[k] = nxt
                # target 从未在该位置生成过（该组无 current）时，保并入的最新一条为 current，
                # 否则整组只剩 historical，版本列表失去「当前生效」语义
                keep_current = (k not in current_groups) and v.version == max_by_group.get(k)
                conn.execute(
                    update(asset_versions)
                    .where(asset_versions.c.id == v.id)
                    .values(
                        asset_id=target_id,
                        version=nxt,
                        status="current" if keep_current else "historical",
                    )
                )

        # 5) source 软删
        conn.execute(
            update(characters)
            .where(characters.c.id == cid)
            .values(deleted_at=ts, updated_at=ts)
        )

        return success(
            {
                "target_id": target_id,
                "filled": filled,
                "mergedEpisodeLinks": len(src_ep_links),
                "mergedStoryboardLinks": len(src_sb_links),
            }
        )
    except Exception as exc:  # noqa: BLE001
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


# ---------------------------------------------------------------------------
# DELETE /{id}
# ---------------------------------------------------------------------------

@router.delete("/{character_id}")
def delete_character(character_id: str, conn: Connection = Depends(get_tx)):
    try:
        cid = parse_param_id(character_id)
        if cid is None:
            return not_found("Invalid character id")
        # 注意：与 GET/PUT 不同，DELETE 不校验存在性，直接打 deleted_at（幂等）
        conn.execute(
            update(characters).where(characters.c.id == cid).values(deleted_at=now())
        )
        return success()
    except Exception as exc:  # noqa: BLE001
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"code": 500, "data": None, "message": str(exc)})


# ---------------------------------------------------------------------------
# 共用辅助（对齐 TS 文件头的同名单函数）
# ---------------------------------------------------------------------------

def _resolve_drama_config_id(conn: Connection, drama_id: Any, field: str) -> Any:
    """drama 级共享生成配置：取该剧**第一个已配置**的 episode configId。

    角色/场景的图片与音色试听是 **drama 级共享资源**，不绑定具体 episode；
    找不到就返回 None（⇒ 走全局默认配置）。
    """
    rows = conn.execute(
        select(episodes).where(and_(episodes.c.drama_id == drama_id,
                                    episodes.c.deleted_at.is_(None)))
    ).all()
    for row in rows:
        value = getattr(row, field)
        if value is not None:
            return value
    return None


def _get_character_three_view_combined_image(char: Any) -> str | None:
    """角色三视图横向长图的 URL（``combined`` → ``front`` → ``side`` → ``back``）。"""
    if char is None or not char.three_views:
        return None
    try:
        views = json.loads(char.three_views)
    except (TypeError, ValueError):
        return None
    if not isinstance(views, dict):
        return None
    combined = (views.get("combined") or views.get("front")
                or views.get("side") or views.get("back"))
    if isinstance(combined, dict):
        return combined.get("imageUrl") or None
    return None


def _parse_body_ref_images(body: dict[str, Any]) -> list[str] | None:
    """请求体里的显式参考图（``reference_images`` → ``referenceImages``，支持数组与 JSON 字符串）。"""
    raw = body.get("reference_images")
    if raw is None:  # `??` 语义：null/缺键都回落到 camelCase 键
        raw = body.get("referenceImages")
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            return None
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    return None


def _resolve_character_anchors(char: Any, body: dict[str, Any],
                               mode: str = "character") -> list[str] | None:
    """视觉锚定：把角色**已生成的基准图**作为参考图随生成下发。

    * 显式 ``reference_images`` → 原样使用（**最多 6 张**，优先级最高）
    * ``anchor='none'`` → 不锚定；``'main'/'image'`` → 仅主立绘；``'three_views'`` → 三视图优先
    * ``'auto'``（默认）→ ``mode='main'``（三视图生成）锚主立绘，其余锚「三视图 combined → 主立绘」
    """
    explicit = _parse_body_ref_images(body)
    # ⚠️ 必须用 `is not None`：JS 里**空数组是真**，`if (explicit) return explicit.slice(0,6)`
    #    会直接返回空列表（而不是回落到 anchor 逻辑）
    if explicit is not None:
        return explicit[:6]
    anchor = body.get("anchor") or "auto"
    if anchor == "none":
        return None

    combined = _get_character_three_view_combined_image(char)
    main = char.image_url or char.local_path or None

    if anchor in ("main", "image"):
        if main:
            return [main]
        return [combined] if (combined and mode != "main") else None
    if anchor == "three_views":
        if combined:
            return [combined]
        return [main] if main else None
    if mode == "main":
        return [main] if main else None
    if combined:
        return [combined]
    return [main] if main else None


def _resolve_character_prompt(user_prompt: Any, auto_prompt: str, drama_style: str | None,
                              visuals_clause: str = "") -> str:
    """角色图 prompt 收口：**即使走自定义 prompt 也强制追加画风后缀**，并注入视觉子句。"""
    user = str(user_prompt or "").strip()
    if not user:
        return auto_prompt
    base = f"{user}, {visuals_clause}" if visuals_clause else user
    return f"{base}{build_character_art_style_suffix(drama_style)}"


def _resolve_character_negative(user_negative: Any, drama_style: str | None) -> str:
    """角色图负向收口：用户未填时按剧集画风给默认负向词。"""
    user = str(user_negative or "").strip()
    return user or build_character_negative_prompt(drama_style)


def _is_legacy_equip_prompt(prompt: Any) -> bool:
    """是否为旧版「单图/含人物」装备提示词（历史遗留，会让装备三视图生成出人）。"""
    text = str(prompt or "").strip()
    if not text:
        return False
    if "character appearance" in text or "for the character" in text:
        return True
    if "single view" in text and "side by side" not in text and "three" not in text:
        return True
    if (("realistic cinematic character design" in text or "natural skin texture" in text)
            and "side by side" not in text):
        return True
    return False


def _resolve_equip_prompt(user_prompt: Any, auto_prompt: str,
                          drama_style: str | None) -> str:
    """装备图 prompt 收口：旧版文本一律丢弃走无人三视图构建器；手写/新版保留但追加画风尾。"""
    user = str(user_prompt or "").strip()
    if not user or _is_legacy_equip_prompt(user):
        return auto_prompt
    return f"{user}{build_equip_art_style_suffix(drama_style)}"


def _lookup_episode(conn: Connection, body: dict[str, Any]) -> tuple[Any, Any]:
    """按 ``body.episode_id`` 查集（未传 → ``(None, None)``）。

    ⚠️ 第二个返回值非空即应直接 400 ``Episode not found``（含 ``episode_id`` 是
    非数字的情况 —— 对应 TS 的 ``Number('abc') = NaN`` 查不到行）。
    """
    raw = body.get("episode_id")
    if not raw:
        return None, None
    number = js_number(raw)
    if number is None:
        return None, bad_request("Episode not found")
    episode = conn.execute(select(episodes).where(episodes.c.id == number)).first()
    if episode is None:
        return None, bad_request("Episode not found")
    return episode, None


# ---------------------------------------------------------------------------
# POST /{id}/auto-split-visuals
# ---------------------------------------------------------------------------

@router.post("/{character_id}/auto-split-visuals")
async def auto_split_visuals(character_id: str, request: Request,
                             conn: Connection = Depends(get_tx)):
    """智能拆分「外貌特征」→ 服装/武器/首饰（LLM 链路）。"""
    cid = parse_param_id(character_id)
    if cid is None:
        return not_found("Invalid character id")
    try:
        char = _fetch_character(conn, cid)
        if char is None:
            return not_found("Character not found")
        body = await read_json(request)
        appearance = str(body.get("appearance") or char.appearance or "").strip()
        if not appearance:
            return bad_request("请先填写「外貌特征」")

        log_task_start("SplitVisuals", "route", {"characterId": cid})
        result = await split_character_visuals(conn, {"appearance": appearance})
        log_task_success("SplitVisuals", "route", {"characterId": cid, "result": result})
        return success(result)
    except Exception as exc:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("SplitVisuals", "route", {"error": str(exc), "id": character_id})
        return bad_request(f"智能拆分失败: {exc}")


# ---------------------------------------------------------------------------
# POST /{id}/generate-voice-sample
# ---------------------------------------------------------------------------

@router.post("/{character_id}/generate-voice-sample")
async def generate_voice_sample_route(character_id: str, request: Request,
                                      conn: Connection = Depends(get_tx)):
    """生成角色试听音频（TTS，固定文案）。"""
    cid = parse_param_id(character_id)
    if cid is None:
        return not_found("Invalid character id")
    body = await read_json(request)
    char = _fetch_character(conn, cid)
    # ⚠️ 这里找不到角色是 **400**（不像 auto-split-visuals 是 404）
    if char is None:
        return bad_request("Character not found")
    if not char.voice_style:
        return bad_request("请先分配音色")
    episode, episode_error = _lookup_episode(conn, body)
    if episode_error is not None:
        return episode_error

    try:
        log_task_start("VoiceSample", "generate", {
            "characterId": cid, "characterName": char.name,
            "episodeId": episode.id if episode is not None else None,
            "voice": char.voice_style,
        })
        config_id = episode.audio_config_id if episode is not None else None
        if config_id is None:
            config_id = _resolve_drama_config_id(conn, char.drama_id, "audio_config_id")
        audio_path = await generate_voice_sample(conn, char.name, char.voice_style, config_id)
        conn.execute(update(characters).where(characters.c.id == cid)
                     .values(voice_sample_url=audio_path, updated_at=now()))
        log_task_success("VoiceSample", "generate", {"characterId": cid, "path": audio_path})
        return success({"voice_sample_url": audio_path})
    except Exception as exc:  # noqa: BLE001
        log_task_error("VoiceSample", "generate", {"characterId": cid, "error": str(exc)})
        return bad_request(f"TTS 生成失败: {exc}")


# ---------------------------------------------------------------------------
# POST /{id}/generate-image
# ---------------------------------------------------------------------------

@router.post("/{character_id}/generate-image")
async def generate_character_image(character_id: str, request: Request,
                                   conn: Connection = Depends(get_tx)):
    """角色立绘生成。

    ⚠️ 立绘重生成**仅支持显式参考图**（``reference_images``），不做自动锚定 —— 保持纯文生图基线。
    """
    cid = parse_param_id(character_id)
    if cid is None:
        return not_found("Invalid character id")
    body = await read_json(request)
    char = _fetch_character(conn, cid)
    if char is None:
        return bad_request("Character not found")
    episode, episode_error = _lookup_episode(conn, body)
    if episode_error is not None:
        return episode_error

    # 自定义 prompt 优先 → 角色 customPrompt → 统一构建器；**无论哪条路径都强制追加画风后缀**
    drama_style = resolve_effective_art_style(conn, char.drama_id, char.style)
    merged = {**row_to_camel(char), **body, "dramaStyle": drama_style}
    auto_prompt = build_character_image_prompt(merged)
    visuals_clause = build_character_visuals_clause(merged)
    prompt = _resolve_character_prompt(body.get("prompt") or char.custom_prompt, auto_prompt,
                                       drama_style, visuals_clause)

    try:
        log_task_start("CharacterImage", "generate", {
            "characterId": cid, "episodeId": episode.id if episode is not None else None,
            "dramaId": char.drama_id, "model": body.get("model") or "default",
        })
        config_id = episode.image_config_id if episode is not None else None
        if config_id is None:
            config_id = _resolve_drama_config_id(conn, char.drama_id, "image_config_id")
        gen_id = await generate_image(conn, {
            "characterId": cid,
            "dramaId": char.drama_id,
            "prompt": prompt,
            "negativePrompt": _resolve_character_negative(
                body.get("negative_prompt") or char.negative_prompt, drama_style),
            "model": body.get("model"),
            "referenceImages": _parse_body_ref_images(body),
            "costume": body.get("costume"),
            "colorGrade": (body.get("color_grade") if body.get("color_grade") is not None
                           else body.get("colorGrade")),
            "configId": config_id,
        })
        log_task_success("CharacterImage", "generate", {"characterId": cid, "generationId": gen_id})
        return success({"image_generation_id": gen_id})
    except Exception as exc:  # noqa: BLE001
        log_task_error("CharacterImage", "generate", {"characterId": cid, "error": str(exc)})
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# POST /{id}/generate-prompt（纯预览，**不落库**）
# ---------------------------------------------------------------------------

@router.post("/{character_id}/generate-prompt")
async def generate_character_prompt(character_id: str, request: Request,
                                    conn: Connection = Depends(get_conn)):
    """按对象类型构建正/负向提示词预览（``type``: character/clothing/weapon/accessory）。"""
    cid = parse_param_id(character_id)
    if cid is None:
        return not_found("Invalid character id")
    body = await read_json(request)
    char = _fetch_character(conn, cid)
    if char is None:
        return bad_request("Character not found")

    object_type = body.get("type") or "character"
    if object_type not in ("character", "clothing", "weapon", "accessory"):
        return bad_request("type 必须是 character/clothing/weapon/accessory 之一")

    # 前端表单最新值覆盖（**camelCase 优先，兼容 snake_case**），未传字段保持库内原值
    merged: dict[str, Any] = row_to_camel(char)
    for key in ("name", "role", "appearance", "personality", "description",
                "clothing", "weapons", "accessories", "costumes", "coreFeatures", "style"):
        snake_key = camel_to_snake(key)
        if snake_key in body:
            merged[key] = body[snake_key]
        elif key in body:
            merged[key] = body[key]

    drama_style = resolve_effective_art_style(conn, char.drama_id, merged.get("style"))
    if object_type == "character":
        prompt = build_character_image_prompt({**merged, "dramaStyle": drama_style})
        negative_prompt = build_character_negative_prompt(drama_style)
    else:
        prompt = build_equip_image_prompt(object_type, {**merged, "dramaStyle": drama_style})
        negative_prompt = build_equip_negative(drama_style)
    return success({"type": object_type, "prompt": prompt, "negativePrompt": negative_prompt})


# ---------------------------------------------------------------------------
# POST /{id}/generate-three-views
# ---------------------------------------------------------------------------

@router.post("/{character_id}/generate-three-views")
async def generate_three_views(character_id: str, request: Request,
                               conn: Connection = Depends(get_tx)):
    """角色三视图：正/侧/背合成**一张横向长图**。

    ⚠️ 原 TS **没有 try/catch** —— 异常直接冒到 500（与相邻端点不同），这里照抄。
    """
    cid = parse_param_id(character_id)
    if cid is None:
        return not_found("Invalid character id")
    body = await read_json(request)
    char = _fetch_character(conn, cid)
    if char is None:
        return bad_request("Character not found")
    episode, episode_error = _lookup_episode(conn, body)
    if episode_error is not None:
        return episode_error

    # 兼容旧参数：可指定视角子集，但**始终合成一张**横向三视图长图
    raw_views = (body.get("views") if isinstance(body.get("views"), list)
                 else ["front", "side", "back"])
    views = [view for view in raw_views if view in ("front", "side", "back")]
    if not views:
        return bad_request("views 必须包含 front/side/back 之一")

    drama_style = resolve_effective_art_style(conn, char.drama_id, char.style)
    merged = {**row_to_camel(char), **body, "dramaStyle": drama_style}
    auto_prompt = build_character_image_prompt(merged)
    visuals_clause = build_character_visuals_clause(merged)
    base_prompt = _resolve_character_prompt(body.get("prompt") or char.custom_prompt, auto_prompt,
                                           drama_style, visuals_clause)
    negative = _resolve_character_negative(
        body.get("negative_prompt") or char.negative_prompt, drama_style)
    # 用户未提供负向词时，追加「组合图」排除词，防止三个视角被拆成多张图/分屏/带文字
    three_view_negative = (negative if (body.get("negative_prompt") or char.negative_prompt)
                           else build_three_view_negative(drama_style))
    color_grade = (body.get("color_grade") if body.get("color_grade") is not None
                   else body.get("colorGrade"))
    config_id = episode.image_config_id if episode is not None else None
    if config_id is None:
        config_id = _resolve_drama_config_id(conn, char.drama_id, "image_config_id")

    # 单张横向长图：正面在左、侧面在中、背面在右
    combined_prompt = f"{base_prompt}, {THREE_VIEW_COMBINED_LAYOUT}"
    gen_id = await generate_image(conn, {
        "characterId": cid,
        "dramaId": char.drama_id,
        "prompt": combined_prompt,
        "negativePrompt": three_view_negative,
        "model": body.get("model"),
        # 视觉锚定：以已生成的角色**主立绘**为参考，锁定五官/服装 → 三视角不漂移
        "referenceImages": _resolve_character_anchors(char, body, "main"),
        "size": body.get("size") or THREE_VIEW_SIZE,
        "viewType": "combined",
        "colorGrade": color_grade,
        "configId": config_id,
    })
    log_task_success("CharacterImage", "three-view-generate",
                     {"characterId": cid, "views": views})
    return success({"count": 1,
                    "results": [{"view": "combined", "image_generation_id": gen_id}]})


# ---------------------------------------------------------------------------
# POST /{id}/generate-equip-image
# ---------------------------------------------------------------------------

@router.post("/{character_id}/generate-equip-image")
async def generate_equip_image(character_id: str, request: Request,
                               conn: Connection = Depends(get_tx)):
    """服装/武器/首饰设定图。

    ``mode='view'``（默认）→ 三视角并排设定图；``mode='single'`` → 单件高清道具图（**纯物品无人**）。
    """
    cid = parse_param_id(character_id)
    if cid is None:
        return not_found("Invalid character id")
    body = await read_json(request)
    object_type = body.get("type")
    # ⚠️ type 校验排在**角色查询之前**（原 TS 顺序）
    if object_type not in ("clothing", "weapon", "accessory"):
        return bad_request("type 必须是 clothing/weapon/accessory 之一")
    char = _fetch_character(conn, cid)
    if char is None:
        return bad_request("Character not found")
    episode, episode_error = _lookup_episode(conn, body)
    if episode_error is not None:
        return episode_error

    drama_style = resolve_effective_art_style(conn, char.drama_id, char.style)
    single_mode = (body.get("mode") == "single" or body.get("item_mode") is True
                   or body.get("single") is True)
    # 各对象类型**独立提示词**字段（角色立绘仍走 customPrompt）
    prompt_field = {"clothing": "clothing_prompt", "weapon": "weapon_prompt",
                    "accessory": "accessory_prompt"}[object_type]
    neg_field = {"clothing": "clothing_negative_prompt", "weapon": "weapon_negative_prompt",
                 "accessory": "accessory_negative_prompt"}[object_type]
    user_prompt = body.get("prompt") or getattr(char, prompt_field)
    user_negative = body.get("negative_prompt") or getattr(char, neg_field)

    # 表单最新值覆盖（未保存也能按当前填写内容生成）；未传字段保持库内值
    merged_char: dict[str, Any] = row_to_camel(char)
    for key in ("clothing", "weapons", "accessories", "costumes", "appearance", "coreFeatures"):
        if key in body:
            merged_char[key] = body[key]
    merged_char["dramaStyle"] = drama_style

    auto_prompt = (build_item_image_prompt(object_type, merged_char) if single_mode
                   else build_equip_image_prompt(object_type, merged_char))
    prompt = _resolve_equip_prompt(user_prompt, auto_prompt, drama_style)
    # 负向词三段嵌套（原 TS 的嵌套三元，逐条对齐）：
    #  单件：用户负向 → 角色负向收口；否则物品负向
    #  三视图：用户负向缺「人物排除词」（或命中旧版提示词）→ 强制装备三视图负向；否则角色负向收口
    if single_mode:
        negative = (_resolve_character_negative(user_negative, drama_style) if user_negative
                    else build_item_negative(drama_style))
    elif user_negative:
        negative = (build_equip_negative(drama_style)
                    if (_is_legacy_equip_prompt(user_negative)
                        or not re.search(r"person|human|hand", user_negative, re.IGNORECASE))
                    else _resolve_character_negative(user_negative, drama_style))
    else:
        negative = build_equip_negative(drama_style)

    try:
        log_task_start("CharacterImage", "equip-generate", {
            "characterId": cid, "type": object_type,
            "mode": "single" if single_mode else "view",
        })
        config_id = episode.image_config_id if episode is not None else None
        if config_id is None:
            config_id = _resolve_drama_config_id(conn, char.drama_id, "image_config_id")
        gen_id = await generate_image(conn, {
            "characterId": cid,
            "dramaId": char.drama_id,
            "prompt": prompt,
            "negativePrompt": negative,
            "model": body.get("model"),
            # 视觉锚定（仅三视图）：锚「角色三视图 combined（无则主立绘）」⇒ 穿的是同一个角色；
            # 单件道具图**不做**自动锚定（避免把角色带进画面）
            "referenceImages": None if single_mode else _resolve_character_anchors(char, body),
            # 单件道具图方形 1:1；三视图沿用横向长图比例（2048x896）保证三视角不挤压
            "size": body.get("size") or (ITEM_IMAGE_SIZE if single_mode else THREE_VIEW_SIZE),
            "equipType": object_type,
            "itemType": object_type if single_mode else None,
            "colorGrade": (body.get("color_grade") if body.get("color_grade") is not None
                           else body.get("colorGrade")),
            "configId": config_id,
        })
        log_task_success("CharacterImage", "equip-generate", {
            "characterId": cid, "type": object_type, "generationId": gen_id,
        })
        return success({"image_generation_id": gen_id})
    except Exception as exc:  # noqa: BLE001
        log_task_error("CharacterImage", "equip-generate", {
            "characterId": cid, "type": object_type, "error": str(exc),
        })
        return bad_request(str(exc))


# ---------------------------------------------------------------------------
# POST /{id}/generate-expressions
# ---------------------------------------------------------------------------

@router.post("/{character_id}/generate-expressions")
async def generate_expressions(character_id: str, request: Request,
                               conn: Connection = Depends(get_tx)):
    """批量生成表情头像组（``body.keys`` 省略则全部预设；传单个 key 即单图重生成）。

    ⚠️ **逐张独立成败**：某张失败只记日志、不中断整批（结果里少一条）。
    """
    cid = parse_param_id(character_id)
    if cid is None:
        return not_found("Invalid character id")
    body = await read_json(request)
    char = _fetch_character(conn, cid)
    if char is None:
        return bad_request("Character not found")
    episode, episode_error = _lookup_episode(conn, body)
    if episode_error is not None:
        return episode_error

    raw_keys = body.get("keys")
    candidates = (raw_keys if (isinstance(raw_keys, list) and raw_keys)
                  else [preset["key"] for preset in EXPRESSION_PRESETS])
    keys = [key for key in candidates if find_expression_preset(key)]
    if not keys:
        return bad_request("keys 中没有合法的表情 key")

    drama_style = resolve_effective_art_style(conn, char.drama_id, char.style)
    merged_char: dict[str, Any] = row_to_camel(char)
    for key in ("appearance", "clothing", "costumes", "coreFeatures", "description"):
        if key in body:
            merged_char[key] = body[key]

    anchors = _resolve_character_anchors(char, body)
    color_grade = (body.get("color_grade") if body.get("color_grade") is not None
                   else body.get("colorGrade"))
    config_id = episode.image_config_id if episode is not None else None
    if config_id is None:
        config_id = _resolve_drama_config_id(conn, char.drama_id, "image_config_id")

    results: list[dict[str, Any]] = []
    for key in keys:
        preset = find_expression_preset(key)
        try:
            log_task_start("CharacterImage", "expression-generate",
                           {"characterId": cid, "expression": key})
            prompt = build_expression_image_prompt(merged_char, key, drama_style)
            negative = (_resolve_character_negative(body.get("negative_prompt"), drama_style)
                        if body.get("negative_prompt") else build_expression_negative(drama_style))
            gen_id = await generate_image(conn, {
                "characterId": cid,
                "dramaId": char.drama_id,
                "prompt": prompt,
                "negativePrompt": negative,
                "model": body.get("model"),
                "referenceImages": anchors,
                "size": body.get("size") or "1024x1024",
                "expression": key,
                "colorGrade": color_grade,
                "configId": config_id,
            })
            results.append({"key": key, "label": preset["label"], "image_generation_id": gen_id})
        except Exception as exc:  # noqa: BLE001 —— 单张失败不中断整批
            log_task_error("CharacterImage", "expression-generate",
                           {"characterId": cid, "expression": key, "error": str(exc)})
    log_task_success("CharacterImage", "expressions-generate", {
        "characterId": cid, "requested": len(keys), "started": len(results),
    })
    return success({"count": len(results), "results": results})


# ---------------------------------------------------------------------------
# POST /batch-generate-images
# ---------------------------------------------------------------------------

@router.post("/batch-generate-images")
async def batch_generate_images(request: Request, conn: Connection = Depends(get_tx)):
    """按集批量出角色立绘（逐个**静默跳过**失败项）。"""
    body = await read_json(request)
    ids = body.get("character_ids") or []
    if not body.get("episode_id"):
        return bad_request("episode_id is required")
    number = js_number(body.get("episode_id"))
    episode = (conn.execute(select(episodes).where(episodes.c.id == number)).first()
               if number is not None else None)
    if episode is None:
        return bad_request("Episode not found")

    results: list[Any] = []
    drama_base_style = get_drama_art_style(conn, episode.drama_id)
    for cid in ids:
        char = conn.execute(
            select(characters).where(and_(characters.c.id == cid,
                                          characters.c.deleted_at.is_(None)))
        ).first()
        if char is None:
            continue
        drama_style = resolve_effective_art_style(conn, episode.drama_id, char.style,
                                                  drama_base_style)
        prompt = build_character_image_prompt({**row_to_camel(char), "dramaStyle": drama_style})
        try:
            gen_id = await generate_image(conn, {
                "characterId": cid,
                "dramaId": char.drama_id,
                "prompt": prompt,
                "negativePrompt": _resolve_character_negative(
                    body.get("negative_prompt") or char.negative_prompt, drama_style),
                "colorGrade": (body.get("color_grade") if body.get("color_grade") is not None
                               else body.get("colorGrade")),
                # ⚠️ 批量接口**只用** `ep.imageConfigId`，不回退到 drama 级（与单张接口不同）
                "configId": episode.image_config_id,
            })
            results.append(gen_id)
        except Exception:  # noqa: BLE001 —— 原 TS 是裸 `catch {}`
            pass
    log_task_success("CharacterImage", "batch-generate", {
        "episodeId": episode.id, "requested": len(ids), "started": len(results),
    })
    return success({"count": len(results), "ids": results})
