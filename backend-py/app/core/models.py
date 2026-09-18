"""SQLAlchemy Core 表定义 —— 与 ``backend/src/db/schema.ts`` 逐列对齐（29 张表）。

两个刻意的设计选择：

1. **用 Core（Table）而不是 ORM（declarative 类）**
   HTTP 契约就是 snake_case，而 snake_case 正是 DB 列名 —— Core 的 ``Row`` 按列名取值，
   转成 dict 即得到契约要求的形状。原 Node 侧的 ``toSnakeCase()`` 转换层在 Python 侧
   不需要存在，**少一层字段漂移源**。同时也不需要维护「Python 属性名 ↔ DB 列名」映射。

2. **不调用 create_all 覆盖既有库**
   Python 后端与 Node 后端共用同一个 SQLite 文件。只有库文件不存在（全新克隆）时才会
   建表，已有库一律不动 —— 见 ``db.py``。

⚠️ ``dramas`` / ``character_templates`` / ``scene_templates`` / ``weapon_templates`` /
``costume_templates`` 有名为 ``metadata`` 的列，访问时请用 ``tbl.c["metadata"]``，
避开 SQLAlchemy 自己挂在 ``Table`` 上的 ``metadata``（MetaData 对象）。
"""

from __future__ import annotations

from sqlalchemy import (
    REAL,
    Boolean,
    Column,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    text,
)

metadata = MetaData()


# ---------------------------------------------------------------------------
# 小工具：让表定义保持紧凑且不易打错列名
# ---------------------------------------------------------------------------

def _dflt(value: object):
    """把 Python 默认值翻成 SQL 字面量。"""
    if isinstance(value, str):
        return text(f"'{value}'")
    if isinstance(value, bool):
        return text("1" if value else "0")
    return text(str(value))


def _pk() -> Column:
    return Column("id", Integer, primary_key=True, autoincrement=True)


def _txt(name: str, nn: bool = False, default: object = None) -> Column:
    return Column(name, Text, nullable=not nn, server_default=_dflt(default) if default is not None else None)


def _int(name: str, nn: bool = False, default: object = None) -> Column:
    return Column(name, Integer, nullable=not nn, server_default=_dflt(default) if default is not None else None)


def _real(name: str, nn: bool = False, default: object = None) -> Column:
    # SQLAlchemy 2.0 顶层导出的是 REAL（不是 Real）
    return Column(name, REAL, nullable=not nn, server_default=_dflt(default) if default is not None else None)


def _bool(name: str, default: bool = False) -> Column:
    """布尔列（对齐 drizzle ``integer({mode:'boolean'})``）。

    ⚠️ **必须用 ``Boolean`` 而不是 ``Integer``**：存储层两者都是 SQLite INTEGER，完全兼容；
    但 drizzle 的 ``mode:'boolean'`` 在 JS 侧读出的是**真布尔**，``JSON.stringify`` 得
    ``true``/``false``；用 ``Integer`` 会让 Python 原样吐出 ``1``/``0`` —— 前端只要用
    ``=== true`` 严格比较就会挂。这个偏差曾真实存在于所有已迁移域（冒烟测试才发现）。
    """
    return Column(name, Boolean, server_default=_dflt(default))


# ===== 核心实体 =====

dramas = Table(
    "dramas", metadata,
    _pk(),
    _txt("title", nn=True),
    _txt("description"),
    _txt("genre"),
    _txt("style", default="realistic"),
    _txt("style_id"),
    _txt("era_background"),
    _int("total_episodes", default=1),
    _int("total_duration", default=0),
    _txt("status", nn=True, default="draft"),
    _txt("thumbnail"),
    _txt("tags"),
    _txt("metadata"),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("deleted_at"),
)

episodes = Table(
    "episodes", metadata,
    _pk(),
    _int("drama_id", nn=True),
    _int("episode_number", nn=True),
    _txt("title", nn=True),
    _txt("content"),
    _txt("script_content"),
    _txt("description"),
    _int("duration", default=0),
    _txt("status", default="draft"),
    _txt("video_url"),
    _txt("thumbnail"),
    _int("image_config_id"),
    _int("video_config_id"),
    _int("audio_config_id"),
    _txt("bgm_url"),
    _real("bgm_volume", default=0.3),
    _real("bgm_fade_in", default=1.5),
    _real("bgm_fade_out", default=2.0),
    _txt("script_hash"),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("deleted_at"),
)

characters = Table(
    "characters", metadata,
    _pk(),
    _int("drama_id", nn=True),
    _txt("name", nn=True),
    _txt("role"),
    _txt("role_type"),
    _txt("description"),
    _txt("appearance"),
    _txt("personality"),
    _txt("voice_style"),
    _txt("speaker_id"),
    _txt("costume_id"),
    _txt("image_url"),
    _txt("reference_images"),
    _txt("seed_value"),
    _int("sort_order"),
    _txt("local_path"),
    _txt("voice_sample_url"),
    _txt("voice_provider"),
    _real("voice_speed", default=1.0),
    _txt("voice_emotion", default="happy"),
    _real("voice_pitch", default=0),
    _txt("clothing"),
    _txt("weapons"),
    _txt("custom_prompt"),
    _txt("negative_prompt"),
    _txt("style"),
    _txt("core_features"),
    _txt("costumes"),
    _txt("variations"),
    _txt("accessories"),
    _txt("three_views"),
    _txt("equip_images"),
    _txt("item_images"),
    _txt("clothing_prompt"),
    _txt("clothing_negative_prompt"),
    _txt("weapon_prompt"),
    _txt("weapon_negative_prompt"),
    _txt("accessory_prompt"),
    _txt("accessory_negative_prompt"),
    _txt("expressions"),
    _txt("voice_model", default="speech-2.8-hd"),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("deleted_at"),
)

episode_characters = Table(
    "episode_characters", metadata,
    _pk(),
    _int("episode_id", nn=True),
    _int("character_id", nn=True),
    _txt("created_at", nn=True),
)

episode_scenes = Table(
    "episode_scenes", metadata,
    _pk(),
    _int("episode_id", nn=True),
    _int("scene_id", nn=True),
    _txt("created_at", nn=True),
)

scenes = Table(
    "scenes", metadata,
    _pk(),
    _int("drama_id", nn=True),
    _int("episode_id"),
    _txt("location", nn=True),
    _txt("location_id"),
    _txt("time", nn=True),
    _txt("prompt", nn=True),
    _int("storyboard_count", default=1),
    _txt("image_url"),
    _txt("status", default="pending"),
    _txt("local_path"),
    _txt("description"),
    _txt("atmosphere"),
    _txt("lighting"),
    _txt("weather"),
    _txt("season"),
    _txt("style"),
    _txt("custom_prompt"),
    _txt("negative_prompt"),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("deleted_at"),
)

storyboards = Table(
    "storyboards", metadata,
    _pk(),
    _int("episode_id", nn=True),
    _int("scene_id"),
    _int("storyboard_number", nn=True),
    _txt("title"),
    _txt("location"),
    _txt("time"),
    _txt("shot_type"),
    _txt("scene_type"),
    _txt("speaker_id"),
    _txt("angle"),
    _txt("movement"),
    _txt("action"),
    _txt("result"),
    _txt("atmosphere"),
    _txt("image_prompt"),
    _txt("video_prompt"),
    _txt("bgm_prompt"),
    _txt("sound_effect"),
    _txt("dialogue"),
    _txt("description"),
    _int("duration", default=0),
    _txt("composed_image"),
    _txt("first_frame_image"),
    _txt("last_frame_image"),
    _txt("tail_frame_image"),
    _txt("keyframe_prompt"),
    _txt("keyframe_image"),
    _txt("reference_images"),
    _txt("video_url"),
    _txt("tts_audio_url"),
    _txt("subtitle_url"),
    _txt("composed_video_url"),
    _txt("custom_image_prompt"),
    _txt("custom_video_prompt"),
    _txt("negative_prompt"),
    _txt("first_frame_prompt"),
    _txt("last_frame_prompt"),
    _txt("transition_type", default="cut"),
    _real("transition_duration", default=0.5),
    _txt("transition_motive"),
    _txt("start_state"),
    _txt("end_state"),
    _txt("constraints"),
    _txt("status", default="pending"),
    _txt("asset_status", default="missing"),
    _txt("script_hash"),
    _int("take_count", default=0),
    _int("take_budget", default=3),
    _txt("rhythm_phase"),
    _txt("route"),
    _txt("route_reason"),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("deleted_at"),
)

storyboard_characters = Table(
    "storyboard_characters", metadata,
    _int("storyboard_id", nn=True),
    _int("character_id", nn=True),
    _txt("costume"),
    PrimaryKeyConstraint("storyboard_id", "character_id"),
)

continuity_states = Table(
    "continuity_states", metadata,
    _pk(),
    _int("episode_id", nn=True),
    _int("storyboard_id"),
    _int("scene_id"),
    _txt("state_type", nn=True),
    _txt("entity_key", nn=True),
    _txt("state_value", nn=True),
    _txt("constraints"),
    _txt("meta"),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
)

# ===== AI 配置 / 音色 / Agent 配置 =====

ai_service_configs = Table(
    "ai_service_configs", metadata,
    _pk(),
    _txt("service_type", nn=True),
    _txt("provider"),
    _txt("name", nn=True),
    _txt("base_url", nn=True),
    _txt("api_key", nn=True),
    _txt("model"),
    _txt("endpoint"),
    _txt("query_endpoint"),
    _int("priority", default=0),
    _bool("is_default", False),
    _bool("is_active", True),
    _txt("settings"),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    # 注意：此表无 deleted_at（对齐 Node 侧注释）
)

ai_service_providers = Table(
    "ai_service_providers", metadata,
    _pk(),
    _txt("name", nn=True),
    _txt("display_name"),
    _txt("service_type", nn=True),
    _txt("provider", nn=True),
    _txt("default_url"),
    _txt("preset_models"),
    _txt("description"),
    _txt("endpoint_prefix"),
    _bool("is_recommended", False),
    _bool("is_active", True),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
)

ai_voices = Table(
    "ai_voices", metadata,
    _pk(),
    Column("voice_id", Text, nullable=False, unique=True),
    _txt("voice_name", nn=True),
    _txt("description"),
    _txt("language"),
    _txt("provider", nn=True),
    _txt("role_tags"),
    _txt("reference_audio"),
    _txt("prompt_text"),
    _txt("created_at", nn=True),
)

agent_configs = Table(
    "agent_configs", metadata,
    _pk(),
    _txt("agent_type", nn=True),
    _txt("name", nn=True),
    _txt("description"),
    _txt("model"),
    _txt("system_prompt"),
    _real("temperature"),
    _int("max_tokens"),
    _int("max_iterations"),
    _txt("skills"),
    _bool("is_active", True),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("deleted_at"),
)

# ===== 生成任务 =====

image_generations = Table(
    "image_generations", metadata,
    _pk(),
    _int("storyboard_id"),
    _int("drama_id"),
    _int("scene_id"),
    _int("character_id"),
    _int("prop_id"),
    _txt("image_type"),
    _txt("frame_type"),
    _txt("provider"),
    _txt("prompt"),
    _txt("negative_prompt"),
    _txt("model"),
    _txt("size"),
    _txt("quality"),
    _txt("style"),
    _int("steps"),
    _real("cfg_scale"),
    _int("seed"),
    _txt("image_url"),
    _txt("local_path"),
    _txt("status", default="pending"),
    _txt("task_id"),
    _txt("error_msg"),
    _int("width"),
    _int("height"),
    _txt("reference_images"),
    _txt("costume"),
    _txt("color_grade"),
    _txt("view_type"),
    _txt("equip_type"),
    _txt("item_type"),
    _txt("expression"),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("completed_at"),
    _txt("deleted_at"),
)

video_generations = Table(
    "video_generations", metadata,
    _pk(),
    _int("storyboard_id"),
    _int("drama_id"),
    _txt("provider"),
    _txt("prompt"),
    _txt("negative_prompt"),
    _txt("model"),
    _int("image_gen_id"),
    _txt("reference_mode"),
    _txt("image_url"),
    _txt("first_frame_url"),
    _txt("last_frame_url"),
    _txt("reference_image_urls"),
    _int("duration"),
    _int("fps"),
    _txt("resolution"),
    _txt("aspect_ratio"),
    _txt("style"),
    _int("motion_level"),
    _txt("camera_motion"),
    _int("seed"),
    _txt("video_url"),
    _txt("local_path"),
    _txt("status", default="pending"),
    _txt("task_id"),
    _txt("error_msg"),
    _int("width"),
    _int("height"),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("completed_at"),
    _txt("deleted_at"),
    _txt("character_ids"),
    _txt("scene_type"),
    _txt("reference_audio_urls"),
    _txt("block_reason"),
    _txt("route"),
    _txt("route_reason"),
)

video_merges = Table(
    "video_merges", metadata,
    _pk(),
    _int("episode_id"),
    _int("drama_id"),
    _txt("title"),
    _txt("provider", nn=True),
    _txt("model", nn=True),
    _txt("status", default="pending"),
    _txt("scenes"),
    _txt("merged_url"),
    _int("duration"),
    _txt("task_id"),
    _txt("error_msg"),
    _txt("created_at", nn=True),
    _txt("completed_at"),
    _txt("deleted_at"),
)

video_quality_checks = Table(
    "video_quality_checks", metadata,
    _pk(),
    _int("storyboard_id"),
    _int("video_generation_id"),
    _int("drama_id"),
    _int("episode_id"),
    _int("lip_sync_score"),
    _int("character_consistency_score"),
    _int("continuity_score"),
    _int("overall_score"),
    _txt("issues"),
    _txt("dimensions"),
    _txt("status", default="pending"),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
)

# ===== 资源库模板表 =====

character_templates = Table(
    "character_templates", metadata,
    _pk(),
    _txt("name", nn=True),
    _txt("category", default="通用"),
    _txt("description"),
    _txt("appearance", nn=True),
    _txt("personality"),
    _txt("clothing_style"),
    _txt("expression"),
    _txt("gender"),
    _txt("age_group"),
    _txt("image_url"),
    _txt("reference_images"),
    _txt("voice_style"),
    _txt("voice_provider"),
    _txt("voice_config"),
    _txt("tags"),
    _txt("metadata"),
    _int("source_drama_id"),
    _int("usage_count", default=0),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("deleted_at"),
)

scene_templates = Table(
    "scene_templates", metadata,
    _pk(),
    _txt("name", nn=True),
    _txt("category", default="通用"),
    _txt("description"),
    _txt("location"),
    _txt("atmosphere"),
    _txt("lighting"),
    _txt("time_of_day"),
    _txt("style"),
    _txt("season"),
    _txt("weather"),
    _txt("image_url"),
    _txt("reference_images"),
    _txt("prompt"),
    _txt("tags"),
    _txt("metadata"),
    _int("source_drama_id"),
    _int("usage_count", default=0),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("deleted_at"),
)

# 物品库（props，连续性状态机 v3）。注意 DB 列名是 image_prompt 而不是 custom_prompt
prop_templates = Table(
    "prop_templates", metadata,
    _pk(),
    _int("drama_id", nn=True),
    _txt("name", nn=True),
    _txt("category", default="道具"),
    _txt("description"),
    _txt("appearance"),
    _txt("size_hint"),
    _txt("holder"),
    _txt("key_clue"),
    _txt("image_prompt"),
    _txt("negative_prompt"),
    _txt("image_url"),
    _txt("reference_images"),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("deleted_at"),
)

episode_props = Table(
    "episode_props", metadata,
    _int("episode_id", nn=True),
    _int("prop_id", nn=True),
    PrimaryKeyConstraint("episode_id", "prop_id"),
)

storyboard_props = Table(
    "storyboard_props", metadata,
    _int("storyboard_id", nn=True),
    _int("prop_id", nn=True),
    PrimaryKeyConstraint("storyboard_id", "prop_id"),
)

weapon_templates = Table(
    "weapon_templates", metadata,
    _pk(),
    _txt("name", nn=True),
    _txt("category", default="剑"),
    _txt("type"),
    _txt("description"),
    _txt("appearance"),
    _txt("material"),
    _txt("attributes"),
    _txt("rank"),
    _txt("owner_character_name"),
    _txt("image_url"),
    _txt("reference_images"),
    _txt("tags"),
    _txt("metadata"),
    _int("source_drama_id"),
    _int("usage_count", default=0),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("deleted_at"),
)

costume_templates = Table(
    "costume_templates", metadata,
    _pk(),
    _txt("name", nn=True),
    _txt("category", default="通用"),
    _txt("description"),
    _txt("style"),
    _txt("body_part"),
    _txt("material"),
    _txt("color_scheme"),
    _txt("season"),
    _txt("appearance"),
    _txt("image_url"),
    _txt("reference_images"),
    _txt("tags"),
    _txt("metadata"),
    _int("source_drama_id"),
    _int("usage_count", default=0),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("deleted_at"),
)

# ===== 预设 / 设置 / 账本 =====

presets = Table(
    "presets", metadata,
    _pk(),
    _txt("type", nn=True),
    _txt("name", nn=True),
    _txt("config"),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
)

app_settings = Table(
    "app_settings", metadata,
    Column("key", Text, primary_key=True),
    _txt("value"),
    _txt("updated_at", nn=True),
)

api_usage = Table(
    "api_usage", metadata,
    _pk(),
    _txt("service_type", nn=True),
    _txt("provider", nn=True),
    _txt("model", nn=True),
    _int("drama_id"),
    _int("episode_id"),
    _int("storyboard_id"),
    _int("image_generation_id"),
    _int("video_generation_id"),
    _int("units"),
    _real("cost_amount"),
    _txt("currency", default="CNY"),
    _bool("is_local", False),
    _txt("status", default="submitted"),
    _int("retry_count", default=0),
    _txt("meta"),
    _txt("created_at", nn=True),
)

#: ComfyUI 工作流运行记录（2026-09-17）—— **这是「实现」而不是「透传」的关键** ✓：
#: 把「跑一次工作流」变成**我们系统的一等公民**（有 DB 记录、能崩溃恢复、能记账、产物落在我们数据根 ✓），
#: 而不是把 ComfyUI 的历史/队列原样透给别人 ✗。
#: 状态机：``queued`` → ``processing`` → ``succeeded`` / ``failed``（与本项目其它生成链路同形 ✓）。
comfyui_runs = Table(
    "comfyui_runs", metadata,
    _pk(),
    #: ``h3``（H3 视频薄封装）或 ``workflow``（任意工作流）✓
    _txt("kind", nn=True, default="workflow"),
    _txt("workflow_name"),
    _txt("source_prompt"),
    #: 请求参数与注入清单（JSON 文本 ✓）
    _txt("params"),
    _txt("status", nn=True, default="queued"),
    #: 人类可读的当前步骤（``submit`` / ``poll`` / ``download`` … ✓）
    _txt("step"),
    #: **上游**（8765 服务）的任务 id —— 崩溃恢复靠它续询 ✓
    _txt("remote_task_id"),
    _txt("outputs"),
    _txt("primary_url"),
    #: 落进**我们数据根**的相对路径（``static/comfyui/<uuid>.mp4`` ✓ 前端可直接播 ✓）
    _txt("local_path"),
    _txt("error_msg"),
    _txt("warnings"),
    _txt("settings"),
    _int("elapsed_ms"),
    _real("requested_seconds"),
    _real("used_seconds"),
    _bool("is_local", True),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
)

asset_versions = Table(
    "asset_versions", metadata,
    _pk(),
    _txt("asset_type", nn=True),
    _int("asset_id", nn=True),
    _txt("media_type", nn=True),
    _txt("frame_type"),
    _int("version", nn=True),
    _txt("asset_url", nn=True),
    _txt("provider"),
    _txt("model"),
    _txt("prompt"),
    _int("generation_id"),
    _txt("meta"),
    _txt("status", default="current"),
    _txt("created_at", nn=True),
)

style_profiles = Table(
    "style_profiles", metadata,
    _pk(),
    _int("drama_id"),
    _txt("name", nn=True),
    _txt("description"),
    _txt("source"),
    _txt("storytelling"),
    _txt("shot_patterns"),
    _txt("audio_captions"),
    _txt("qc_rules"),
    _txt("facts"),
    _txt("inferences"),
    _txt("preferences"),
    _bool("is_active", False),
    _txt("created_at", nn=True),
    _txt("updated_at", nn=True),
    _txt("deleted_at"),
)


#: Node 侧 ``schema.ts`` 的全部表，供自检脚本比对「表数量是否与 Node 一致」。
ALL_TABLES = tuple(metadata.tables.keys())
