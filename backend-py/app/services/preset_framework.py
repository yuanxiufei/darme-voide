"""预设框架服务（移植自 ``backend/src/services/preset-framework.ts``，475 行）。

通用骨架：**Variation Card 引擎** + Drama/Episode/Storyboard 创建 + 复用既有的
image/video 生成管线。

⚠️ 两处「原样保留」的特性（别当成 bug 去修）：

1. **数据池全是占位符**（``THEME_FAMILY_A`` / ``COMPOSITION_TYPE_1`` …）——原实现就是
   占位数据池，等具体领域知识替换；数量与顺序会影响随机组合结果，**别"顺手改成中文"**；
2. ``derive_space_and_frame`` 用 ``(themeFamily + focalPoint).length`` 取模派生 —— 是
   **字符串长度**（不是 hash 值），原实现如此；``pick_character_layout`` 干脆**忽略入参**。

⚠️ **列名适配（重要）**：原 TS 写的是 ``dramas.systemMetadata`` / ``storyboards.sceneNumber`` /
``storyboards.cameraMovement`` —— **这三个列在共享 schema 里都不存在**（这个域是"从未接线"的骨架，
Node 侧调用同样会因缺列报错）。这里按真实列名落地：

* ``dramas.metadata`` ← ``{variationCard, presetType}``（紧凑 JSON，与 Node 共用该列）；
* ``storyboards.storyboard_number`` ← ``shotIndex``（``/status`` 本就是按它排序的，写它才自洽）；
* ``storyboards.movement`` ← 随机运镜（Python 侧列名是 ``movement``，提示词构建器读的也是它）；
* 原 TS 想写的「每镜 shotConfig JSON」**没有对应列** ⇒ 不写（只在 drama.metadata 里留整张 card）。

⚠️ 并发编排：原 TS 用 ``utils/merge-queue`` 的 ``runMerged``（并发 + 顺序事件流），
这里用 ``asyncio.gather(return_exceptions=True)`` 实现同形语义（每镜自管状态与异常、
**失败不抛出**），最后按 ``shotIndex`` 恢复提交顺序。
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.engine import Connection

from ..core.db import engine
from ..core.models import dramas, episodes, storyboards
from ..core.response import now
from .image_generation import generate_image
from .prompt_utils import (
    PRESET_IMAGE_NEGATIVE,
    PRESET_VIDEO_NEGATIVE,
    build_preset_image_prompt,
    build_preset_video_prompt,
)
from .video_generation import generate_video

__all__ = [
    "create_preset_drama",
    "generate_variation_card",
    "get_preset_pipeline_status",
    "trigger_image_generation",
    "trigger_video_generation",
]

# ===========================================================================
# 占位符数据池（原样保留，别替换成"更好"的词）
# ===========================================================================

THEME_FAMILIES = [
    "THEME_FAMILY_A", "THEME_FAMILY_B", "THEME_FAMILY_C", "THEME_FAMILY_D",
    "THEME_FAMILY_E", "THEME_FAMILY_F", "THEME_FAMILY_G", "THEME_FAMILY_H",
]

COMPOSITION_PATTERNS = [
    "COMPOSITION_TYPE_1", "COMPOSITION_TYPE_2", "COMPOSITION_TYPE_3", "COMPOSITION_TYPE_4",
    "COMPOSITION_TYPE_5", "COMPOSITION_TYPE_6", "COMPOSITION_TYPE_7", "COMPOSITION_TYPE_8",
    "COMPOSITION_TYPE_9", "COMPOSITION_TYPE_10",
]

MAIN_FOCAL_POINTS = [
    "FOCAL_POINT_A", "FOCAL_POINT_B", "FOCAL_POINT_C",
    "FOCAL_POINT_D", "FOCAL_POINT_E", "FOCAL_POINT_F",
]

SUBTLE_CLUES = ["SUBTLE_CLUE_1", "SUBTLE_CLUE_2", "SUBTLE_CLUE_3", "SUBTLE_CLUE_4"]

PROMINENT_CLUES = ["PROMINENT_CLUE_1", "PROMINENT_CLUE_2"]

ACTIVITIES = [
    "ACTIVITY_1", "ACTIVITY_2", "ACTIVITY_3", "ACTIVITY_4", "ACTIVITY_5",
    "ACTIVITY_6", "ACTIVITY_7", "ACTIVITY_8", "ACTIVITY_9",
]

LIVING_ELEMENTS = ["LIVING_ELEMENT_A", "LIVING_ELEMENT_B", "LIVING_ELEMENT_C"]

LIGHT_STRUCTURES = [
    "LIGHT_STRUCTURE_1", "LIGHT_STRUCTURE_2", "LIGHT_STRUCTURE_3",
    "LIGHT_STRUCTURE_4", "LIGHT_STRUCTURE_5", "LIGHT_STRUCTURE_6",
]

CAMERA_POSITIONS = [
    "CAMERA_POSITION_1", "CAMERA_POSITION_2", "CAMERA_POSITION_3", "CAMERA_POSITION_4",
    "CAMERA_POSITION_5", "CAMERA_POSITION_6", "CAMERA_POSITION_7", "CAMERA_POSITION_8",
    "CAMERA_POSITION_9",
]

WIND_DIRECTIONS = [
    "WIND_DIRECTION_LEFT", "WIND_DIRECTION_RIGHT", "WIND_DIRECTION_CENTER",
]

CAMERA_MOVES = [
    "CAMERA_MOVE_SLOW_LEFT", "CAMERA_MOVE_SLOW_RIGHT", "CAMERA_MOVE_PUSH_IN",
    "CAMERA_MOVE_PULL_OUT", "CAMERA_MOVE_STATIC",
]

#: 配比约束
MAX_LIVING_SHOTS = 3
CLUE_RATIO = {"subtle": 3, "prominent": 2}  # 5 镜中 subtle:prominent ≈ 3:2


# ===========================================================================
# 工具函数
# ===========================================================================

def _random_pick(items: list[Any]) -> Any:
    """``arr[Math.floor(Math.random() * arr.length)]``。"""
    return items[int(random.random() * len(items))]


def _random_pick_n(items: list[Any], n: int) -> list[Any]:
    """洗牌取前 n 个（``[...arr].sort(() => Math.random() - 0.5)`` 的等价：**不修改原数组**）。"""
    shuffled = list(items)
    random.shuffle(shuffled)
    return shuffled[:n]


def _random_number(minimum: int, maximum: int) -> int:
    """``Math.floor(Math.random() * (max - min + 1)) + min``（**含两端**）。"""
    return int(random.random() * (maximum - minimum + 1)) + minimum


def derive_space_and_frame(theme_family: str, focal_point: str) -> dict[str, str]:
    """按 ``(themeFamily + focalPoint)`` 的**字符串长度**取模派生空间类型与前景框架。"""
    hash_value = len(theme_family + focal_point)
    types = ["SPACE_TYPE_A", "SPACE_TYPE_B", "SPACE_TYPE_C", "SPACE_TYPE_D"]
    frames = ["FRAME_TYPE_1", "FRAME_TYPE_2", "FRAME_TYPE_3"]
    return {
        "spaceType": types[hash_value % len(types)],
        "foregroundFrame": frames[hash_value % len(frames)],
    }


def pick_character_layout(_activity: str) -> str:
    """⚠️ 入参**被忽略**（原实现如此），随机取一种人物排列。"""
    return _random_pick(["LAYOUT_SOLO", "LAYOUT_PAIR", "LAYOUT_TRIANGLE", "LAYOUT_SCATTERED"])


# ===========================================================================
# Variation Card 引擎
# ===========================================================================

def generate_variation_card(exclude_family: str | None = None) -> dict[str, Any]:
    """生成一张 Variation Card（**固定 5 个 Shot**）。

    ``exclude_family`` 用于排除上一张的家族，避免连续重复。
    """
    # 1. 主题家族（可排除）
    family_pool = [f for f in THEME_FAMILIES if f != exclude_family] if exclude_family else list(THEME_FAMILIES)
    theme_family = _random_pick(family_pool)

    # 2. 5 个 shot 各自独立分配
    compositions = _random_pick_n(COMPOSITION_PATTERNS, 5)
    focal_points = _random_pick_n(MAIN_FOCAL_POINTS, 5)
    activities = _random_pick_n(ACTIVITIES, 5)

    # 3. 主题线索按**配比**分配（subtle 3 : prominent 2）
    clue_pool: list[str] = []
    for _ in range(CLUE_RATIO["subtle"]):
        clue_pool.append(_random_pick(SUBTLE_CLUES))
    for _ in range(CLUE_RATIO["prominent"]):
        clue_pool.append(_random_pick(PROMINENT_CLUES))
    shuffled_clues = clue_pool
    random.shuffle(shuffled_clues)

    # 4. 灵动元素（最多 MAX_LIVING_SHOTS 个镜头出现）
    living_pool: list[str | None] = [None] * 5
    living_count = _random_number(0, MAX_LIVING_SHOTS)
    if living_count > 0:
        for index in _random_pick_n([0, 1, 2, 3, 4], living_count):
            living_pool[index] = _random_pick(LIVING_ELEMENTS)

    # 5. 组装 5 个 Shot
    shots = []
    for index in range(5):
        derived = derive_space_and_frame(theme_family, focal_points[index])
        shots.append({
            "shotIndex": index + 1,
            "themeFamily": theme_family,
            "compositionPattern": compositions[index],
            "spaceType": derived["spaceType"],
            "foregroundFrame": derived["foregroundFrame"],
            "mainFocalPoint": focal_points[index],
            "thematicClue": shuffled_clues[index],
            "activity": activities[index],
            "cameraPosition": _random_pick(CAMERA_POSITIONS),
            "windDirection": _random_pick(WIND_DIRECTIONS),
            "lightStructure": _random_pick(LIGHT_STRUCTURES),
            "characterLayout": pick_character_layout(activities[index]),
            "livingElement": living_pool[index],
        })

    return {"themeFamily": theme_family, "shots": shots}


# ===========================================================================
# Drama / Episode / Storyboard 创建
# ===========================================================================

def _compact(value: Any) -> str:
    """与 Node 共用的 JSON 列 ⇒ 紧凑分隔符。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def create_preset_drama(
    title: str, description: str, variation_card: dict[str, Any]
) -> dict[str, Any]:
    """基于 Variation Card 创建 Drama → Episode → N 个 Storyboard。"""
    ts = now()
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title=title,
            description=description or f"预设风格: {variation_card.get('themeFamily')}",
            style="preset-framework",
            total_episodes=1,
            # ⚠️ TS 写的是 `systemMetadata`，但**该列在共享 schema 里不存在**（这个域是没接线的骨架）
            #    ⇒ 落到真实列 `metadata`（同一语义）
            metadata=_compact({"variationCard": variation_card, "presetType": "framework"}),
            status="draft",
            created_at=ts, updated_at=ts,
        )).lastrowid)

        episode_id = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title=f"{title} - 第1集",
            status="draft", created_at=ts, updated_at=ts,
        )).lastrowid)

        storyboard_ids: list[int] = []
        for shot in variation_card.get("shots") or []:
            storyboard_ids.append(int(conn.execute(storyboards.insert().values(
                episode_id=episode_id,
                # ⚠️ TS 写 `sceneNumber`（列不存在）⇒ 落到真实列 `storyboard_number`；
                #    而且 /status 就是按 storyboardNumber 排序的，这里必须写它才自洽
                storyboard_number=shot.get("shotIndex"),
                shot_type=shot.get("compositionPattern"),
                description=f"Shot {shot.get('shotIndex')}: {shot.get('mainFocalPoint')}",
                movement=_random_pick(CAMERA_MOVES),
                status="pending",
                created_at=ts, updated_at=ts,
            )).lastrowid))

    return {"dramaId": drama_id, "episodeId": episode_id, "storyboardIds": storyboard_ids}


# ===========================================================================
# 管线编排（每镜独立 producer：自管状态、失败不抛出）
# ===========================================================================

async def _run_merged(tasks: list[Any]) -> list[Any]:
    """并发跑所有 producer，**异常被吞**（对应 ``runMerged`` 的 rejected 事件）。"""
    results = await asyncio.gather(*(task() for task in tasks), return_exceptions=True)
    return [None if isinstance(item, BaseException) else item for item in results]


def _set_storyboard_status(storyboard_id: int, status: str) -> None:
    """独立短事务写状态（每镜自管，失败不影响其它镜头）。"""
    with engine.begin() as conn:
        conn.execute(update(storyboards).where(storyboards.c.id == storyboard_id).values(
            status=status, updated_at=now()
        ))


async def trigger_image_generation(
    drama_id: int, episode_id: int, storyboard_ids: list[int], variation_card: dict[str, Any]
) -> dict[str, Any]:
    """批量生成各 Shot 的首帧图（并发；**失败的镜头返回 ok=False，不抛出**）。"""
    shots = variation_card.get("shots") or []

    async def produce(storyboard_id: int, index: int) -> dict[str, Any]:
        shot = next((s for s in shots if s.get("shotIndex") == index + 1), None)
        if shot is None:
            return {"sbId": storyboard_id, "shotIndex": index + 1, "ok": False}
        try:
            _set_storyboard_status(storyboard_id, "generating_image")
            prompt = build_preset_image_prompt(shot)
            with engine.begin() as conn:
                image_id = await generate_image(conn, {
                    "dramaId": drama_id,
                    "storyboardId": storyboard_id,
                    "prompt": prompt,
                    "negativePrompt": PRESET_IMAGE_NEGATIVE,
                })
            if image_id:
                _set_storyboard_status(storyboard_id, "image_ready")
            return {"sbId": storyboard_id, "shotIndex": index + 1, "ok": True, "imageId": image_id}
        except Exception:  # noqa: BLE001 —— 单镜失败不影响其它
            _set_storyboard_status(storyboard_id, "image_failed")
            return {"sbId": storyboard_id, "shotIndex": index + 1, "ok": False}

    results = await _run_merged([lambda i=i, sb=sb: produce(sb, i) for i, sb in enumerate(storyboard_ids)])

    # 按 shotIndex 恢复提交顺序，与串行行为一致
    image_gen_ids = [
        item["imageId"]
        for item in sorted([r for r in results if r and r.get("ok") and r.get("imageId") is not None],
                           key=lambda r: r["shotIndex"])
    ]
    return {"imageGenIds": image_gen_ids}


async def trigger_video_generation(
    drama_id: int, episode_id: int, storyboard_ids: list[int], variation_card: dict[str, Any]
) -> dict[str, Any]:
    """批量生成各 Shot 的视频（图生视频，**需要首帧已就绪**）。"""
    shots = variation_card.get("shots") or []

    async def produce(storyboard_id: int, index: int) -> dict[str, Any]:
        shot = next((s for s in shots if s.get("shotIndex") == index + 1), None)
        if shot is None:
            return {"sbId": storyboard_id, "shotIndex": index + 1, "ok": False}

        with engine.begin() as conn:
            row = conn.execute(
                # ⚠️ TS 读的是 `imageUrl`（该列不存在）⇒ 本 schema 的"镜头首帧"是 first_frame_image
                select(storyboards.c.first_frame_image).where(storyboards.c.id == storyboard_id)
            ).first()
        if row is None or not row[0]:
            return {"sbId": storyboard_id, "shotIndex": index + 1, "ok": False}

        try:
            _set_storyboard_status(storyboard_id, "generating_video")
            camera_move = _random_pick(CAMERA_MOVES)
            prompt = build_preset_video_prompt(shot, camera_move)
            with engine.begin() as conn:
                video_id = await generate_video(conn, {
                    "dramaId": drama_id,
                    "storyboardId": storyboard_id,
                    "prompt": prompt,
                    "negativePrompt": PRESET_VIDEO_NEGATIVE,
                    "referenceImageUrl": row[0],
                })
            # 修正原串行实现瑕疵：videoId 为空时**不应**标记 video_ready
            _set_storyboard_status(storyboard_id, "video_ready" if video_id else "video_failed")
            return {"sbId": storyboard_id, "shotIndex": index + 1, "ok": bool(video_id),
                    "videoId": video_id}
        except Exception:  # noqa: BLE001
            _set_storyboard_status(storyboard_id, "video_failed")
            return {"sbId": storyboard_id, "shotIndex": index + 1, "ok": False}

    results = await _run_merged([lambda i=i, sb=sb: produce(sb, i) for i, sb in enumerate(storyboard_ids)])

    video_gen_ids = [
        item["videoId"]
        for item in sorted([r for r in results if r and r.get("ok") and r.get("videoId") is not None],
                           key=lambda r: r["shotIndex"])
    ]
    return {"videoGenIds": video_gen_ids}


# ===========================================================================
# 状态查询
# ===========================================================================

def get_preset_pipeline_status(conn: Connection, drama_id: int) -> dict[str, Any]:
    """管线状态：Drama / Episode / Storyboards / Variation Card / 生成进度汇总。"""
    drama = conn.execute(select(dramas).where(dramas.c.id == drama_id)).first()
    if drama is None:
        raise ValueError("Drama not found")

    episode = conn.execute(
        select(episodes).where(episodes.c.drama_id == drama_id)
    ).first()

    storyboard_rows = []
    if episode is not None:
        storyboard_rows = conn.execute(
            select(
                storyboards.c.id, storyboards.c.storyboard_number,
                storyboards.c.first_frame_image, storyboards.c.video_url, storyboards.c.status,
            )
            .where(storyboards.c.episode_id == episode.id)
            .order_by(storyboards.c.storyboard_number)
        ).all()

    # ⚠️ 写入侧存的是 JSON 字符串；若该列被声明为 JSON 类型则会读成 dict ⇒ 两种都兼容
    metadata = drama.metadata
    variation_card = None
    if isinstance(metadata, dict):
        variation_card = metadata.get("variationCard")
    elif isinstance(metadata, str) and metadata:
        try:
            parsed = json.loads(metadata)
            variation_card = parsed.get("variationCard") if isinstance(parsed, dict) else None
        except (ValueError, TypeError):
            variation_card = None

    return {
        "drama": {"id": drama.id, "title": drama.title, "style": drama.style},
        "episode": ({"id": episode.id, "episodeNumber": episode.episode_number}
                    if episode is not None else None),
        "storyboards": [
            {
                "id": row[0], "storyboardNumber": row[1],
                "imageUrl": row[2], "videoUrl": row[3], "status": row[4],
            }
            for row in storyboard_rows
        ],
        "variationCard": variation_card,
        "summary": {
            "totalShots": len(storyboard_rows),
            "imagesGenerated": len([r for r in storyboard_rows if r[2]]),
            "videosGenerated": len([r for r in storyboard_rows if r[3]]),
        },
    }
