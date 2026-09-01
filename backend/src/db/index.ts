import { drizzle } from 'drizzle-orm/better-sqlite3'
import * as schema from './schema.js'
import { getSqlite } from './connection.js'

function runMigrations(): void {
  getSqlite().exec(`
  CREATE TABLE IF NOT EXISTS dramas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    genre TEXT,
    style TEXT DEFAULT 'realistic',
    total_episodes INTEGER DEFAULT 1,
    total_duration INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'draft',
    thumbnail TEXT,
    tags TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  );

  CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drama_id INTEGER NOT NULL,
    episode_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    script_content TEXT,
    description TEXT,
    duration INTEGER DEFAULT 0,
    status TEXT DEFAULT 'draft',
    video_url TEXT,
    thumbnail TEXT,
    image_config_id INTEGER,
    video_config_id INTEGER,
    audio_config_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  );

  CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drama_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    role TEXT,
    description TEXT,
    appearance TEXT,
    personality TEXT,
    voice_style TEXT,
    image_url TEXT,
    reference_images TEXT,
    seed_value TEXT,
    sort_order INTEGER,
    local_path TEXT,
    voice_sample_url TEXT,
    voice_provider TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  );

  CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drama_id INTEGER NOT NULL,
    episode_id INTEGER,
    location TEXT NOT NULL,
    time TEXT NOT NULL,
    prompt TEXT NOT NULL,
    storyboard_count INTEGER DEFAULT 1,
    image_url TEXT,
    status TEXT DEFAULT 'pending',
    local_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  );

  CREATE TABLE IF NOT EXISTS storyboards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL,
    scene_id INTEGER,
    storyboard_number INTEGER NOT NULL,
    title TEXT,
    location TEXT,
    time TEXT,
    shot_type TEXT,
    angle TEXT,
    movement TEXT,
    action TEXT,
    result TEXT,
    atmosphere TEXT,
    image_prompt TEXT,
    video_prompt TEXT,
    bgm_prompt TEXT,
    sound_effect TEXT,
    dialogue TEXT,
    description TEXT,
    duration INTEGER DEFAULT 0,
    composed_image TEXT,
    first_frame_image TEXT,
    last_frame_image TEXT,
    reference_images TEXT,
    video_url TEXT,
    tts_audio_url TEXT,
    subtitle_url TEXT,
    composed_video_url TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  );

  CREATE TABLE IF NOT EXISTS episode_characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL,
    character_id INTEGER NOT NULL,
    created_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_episode_characters_episode_id
    ON episode_characters (episode_id);
  CREATE INDEX IF NOT EXISTS idx_episode_characters_character_id
    ON episode_characters (character_id);

  CREATE TABLE IF NOT EXISTS episode_scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL,
    scene_id INTEGER NOT NULL,
    created_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_episode_scenes_episode_id
    ON episode_scenes (episode_id);
  CREATE INDEX IF NOT EXISTS idx_episode_scenes_scene_id
    ON episode_scenes (scene_id);

  CREATE TABLE IF NOT EXISTS storyboard_characters (
    storyboard_id INTEGER NOT NULL,
    character_id INTEGER NOT NULL,
    PRIMARY KEY (storyboard_id, character_id)
  );
  CREATE INDEX IF NOT EXISTS idx_storyboard_characters_storyboard_id
    ON storyboard_characters (storyboard_id);
  CREATE INDEX IF NOT EXISTS idx_storyboard_characters_character_id
    ON storyboard_characters (character_id);

  CREATE TABLE IF NOT EXISTS ai_service_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_type TEXT NOT NULL,
    provider TEXT,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    api_key TEXT NOT NULL,
    model TEXT,
    endpoint TEXT,
    query_endpoint TEXT,
    priority INTEGER DEFAULT 0,
    is_default INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,
    settings TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS ai_service_providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    display_name TEXT,
    service_type TEXT NOT NULL,
    provider TEXT NOT NULL,
    default_url TEXT,
    preset_models TEXT,
    description TEXT,
    endpoint_prefix TEXT,
    is_recommended INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS ai_voices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    voice_id TEXT NOT NULL UNIQUE,
    voice_name TEXT NOT NULL,
    description TEXT,
    language TEXT,
    provider TEXT NOT NULL,
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS agent_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    model TEXT,
    system_prompt TEXT,
    temperature REAL,
    max_tokens INTEGER,
    max_iterations INTEGER,
    is_active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  );

  CREATE TABLE IF NOT EXISTS image_generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    storyboard_id INTEGER,
    drama_id INTEGER,
    scene_id INTEGER,
    character_id INTEGER,
    image_type TEXT,
    frame_type TEXT,
    provider TEXT,
    prompt TEXT,
    negative_prompt TEXT,
    model TEXT,
    size TEXT,
    quality TEXT,
    style TEXT,
    steps INTEGER,
    cfg_scale REAL,
    seed INTEGER,
    image_url TEXT,
    local_path TEXT,
    status TEXT DEFAULT 'pending',
    task_id TEXT,
    error_msg TEXT,
    width INTEGER,
    height INTEGER,
    reference_images TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
  );

  CREATE TABLE IF NOT EXISTS video_generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    storyboard_id INTEGER,
    drama_id INTEGER,
    provider TEXT,
    prompt TEXT,
    negative_prompt TEXT,
    model TEXT,
    image_gen_id INTEGER,
    reference_mode TEXT,
    image_url TEXT,
    first_frame_url TEXT,
    last_frame_url TEXT,
    reference_image_urls TEXT,
    duration INTEGER,
    fps INTEGER,
    resolution TEXT,
    aspect_ratio TEXT,
    style TEXT,
    motion_level INTEGER,
    camera_motion TEXT,
    seed INTEGER,
    video_url TEXT,
    local_path TEXT,
    status TEXT DEFAULT 'pending',
    task_id TEXT,
    error_msg TEXT,
    width INTEGER,
    height INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    deleted_at TEXT
  );

  CREATE TABLE IF NOT EXISTS video_merges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER,
    drama_id INTEGER,
    title TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    scenes TEXT,
    merged_url TEXT,
    duration INTEGER,
    task_id TEXT,
    error_msg TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    deleted_at TEXT
  );

  CREATE TABLE IF NOT EXISTS video_quality_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    storyboard_id INTEGER,
    video_generation_id INTEGER,
    drama_id INTEGER,
    episode_id INTEGER,
    lip_sync_score INTEGER,
    character_consistency_score INTEGER,
    continuity_score INTEGER,
    overall_score INTEGER,
    issues TEXT,
    dimensions TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_video_quality_checks_storyboard_id
    ON video_quality_checks (storyboard_id);

  -- ====== 连续性状态机（v3）======
  CREATE TABLE IF NOT EXISTS continuity_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL,
    storyboard_id INTEGER,
    scene_id INTEGER,
    state_type TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    state_value TEXT NOT NULL,
    constraints TEXT,
    meta TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_continuity_states_episode_id
    ON continuity_states (episode_id);
  CREATE INDEX IF NOT EXISTS idx_continuity_states_storyboard_id
    ON continuity_states (storyboard_id);

  -- ====== 物品库（props）======
  CREATE TABLE IF NOT EXISTS prop_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drama_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    category TEXT DEFAULT '道具',
    description TEXT,
    appearance TEXT,
    size_hint TEXT,
    holder TEXT,
    key_clue TEXT,
    image_prompt TEXT,
    negative_prompt TEXT,
    image_url TEXT,
    reference_images TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  );
  CREATE INDEX IF NOT EXISTS idx_prop_templates_drama_id
    ON prop_templates (drama_id);
  CREATE TABLE IF NOT EXISTS episode_props (
    episode_id INTEGER NOT NULL,
    prop_id INTEGER NOT NULL,
    PRIMARY KEY (episode_id, prop_id)
  );
  CREATE TABLE IF NOT EXISTS storyboard_props (
    storyboard_id INTEGER NOT NULL,
    prop_id INTEGER NOT NULL,
    PRIMARY KEY (storyboard_id, prop_id)
  );

  -- ====== 资源库模板表 ======

  CREATE TABLE IF NOT EXISTS character_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT DEFAULT '通用',
    description TEXT,
    appearance TEXT NOT NULL,
    personality TEXT,
    clothing_style TEXT,
    expression TEXT,
    gender TEXT,
    age_group TEXT,
    image_url TEXT,
    reference_images TEXT,
    voice_style TEXT,
    voice_provider TEXT,
    voice_config TEXT,
    tags TEXT,
    metadata TEXT,
    source_drama_id INTEGER,
    usage_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  );

  CREATE TABLE IF NOT EXISTS scene_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT DEFAULT '通用',
    description TEXT,
    location TEXT,
    atmosphere TEXT,
    lighting TEXT,
    time_of_day TEXT,
    style TEXT,
    season TEXT,
    weather TEXT,
    image_url TEXT,
    reference_images TEXT,
    prompt TEXT,
    tags TEXT,
    metadata TEXT,
    source_drama_id INTEGER,
    usage_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  );

  CREATE TABLE IF NOT EXISTS weapon_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT DEFAULT '剑',
    type TEXT,
    description TEXT,
    appearance TEXT,
    material TEXT,
    attributes TEXT,
    rank TEXT,
    owner_character_name TEXT,
    image_url TEXT,
    reference_images TEXT,
    tags TEXT,
    metadata TEXT,
    source_drama_id INTEGER,
    usage_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  );

  CREATE TABLE IF NOT EXISTS costume_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT DEFAULT '通用',
    description TEXT,
    style TEXT,
    body_part TEXT,
    material TEXT,
    color_scheme TEXT,
    season TEXT,
    appearance TEXT,
    image_url TEXT,
    reference_images TEXT,
    tags TEXT,
    metadata TEXT,
    source_drama_id INTEGER,
    usage_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  );

  CREATE TABLE IF NOT EXISTS presets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    config TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );

  -- ====== 用量统计与成本估算（对齐参考项目 ArcReel usage_repo）======
  CREATE TABLE IF NOT EXISTS api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_type TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    drama_id INTEGER,
    episode_id INTEGER,
    storyboard_id INTEGER,
    image_generation_id INTEGER,
    video_generation_id INTEGER,
    units INTEGER,
    cost_amount REAL,
    currency TEXT DEFAULT 'CNY',
    is_local INTEGER DEFAULT 0,
    status TEXT DEFAULT 'submitted',
    retry_count INTEGER DEFAULT 0,
    meta TEXT,
    created_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_api_usage_drama_id
    ON api_usage (drama_id);
  CREATE INDEX IF NOT EXISTS idx_api_usage_episode_id
    ON api_usage (episode_id);
  CREATE INDEX IF NOT EXISTS idx_api_usage_created_at
    ON api_usage (created_at);

  -- ====== 资产版本历史/回滚（对齐参考项目 ArcReel artifact_version_provenance）======
  CREATE TABLE IF NOT EXISTS asset_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_type TEXT NOT NULL,
    asset_id INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    frame_type TEXT,
    version INTEGER NOT NULL,
    asset_url TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    prompt TEXT,
    generation_id INTEGER,
    meta TEXT,
    status TEXT DEFAULT 'current',
    created_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_asset_versions_asset
    ON asset_versions (asset_type, asset_id, media_type, frame_type, version);
  CREATE INDEX IF NOT EXISTS idx_asset_versions_status
    ON asset_versions (asset_type, asset_id, status);

  -- ====== 风格 Profile 提炼（对齐参考项目 H3-Codex-Drama Profile Distiller）======
  CREATE TABLE IF NOT EXISTS style_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drama_id INTEGER,
    name TEXT NOT NULL,
    description TEXT,
    source TEXT,
    storytelling TEXT,
    shot_patterns TEXT,
    audio_captions TEXT,
    qc_rules TEXT,
    facts TEXT,
    inferences TEXT,
    preferences TEXT,
    is_active INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  );
  CREATE INDEX IF NOT EXISTS idx_style_profiles_drama
    ON style_profiles (drama_id, deleted_at);
`)

  function ensureColumn(table: string, column: string, definition: string) {
    const s = getSqlite()
    const tableExists = s.prepare(
      `SELECT 1 as ok FROM sqlite_master WHERE type='table' AND name=? LIMIT 1`,
    ).get(table) as { ok: number } | undefined
    if (!tableExists) return
    const columns = s.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>
    if (!columns.some(col => col.name === column)) {
      s.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`)
    }
  }

  ensureColumn('episodes', 'image_config_id', 'INTEGER')
  ensureColumn('episodes', 'video_config_id', 'INTEGER')
  ensureColumn('episodes', 'audio_config_id', 'INTEGER')
  ensureColumn('episodes', 'bgm_url', 'TEXT')
  ensureColumn('episodes', 'bgm_volume', 'REAL DEFAULT 0.3')
  ensureColumn('episodes', 'bgm_fade_in', 'REAL DEFAULT 1.5')
  ensureColumn('episodes', 'bgm_fade_out', 'REAL DEFAULT 2.0')
  ensureColumn('agent_configs', 'skills', 'TEXT')

  // ====== v2 角色/场景/分镜/视频增强字段 ======
  ensureColumn('characters', 'voice_speed', 'REAL DEFAULT 1.0')
  ensureColumn('characters', 'voice_emotion', "TEXT DEFAULT 'happy'")
  ensureColumn('characters', 'voice_pitch', 'REAL DEFAULT 0')
  ensureColumn('characters', 'clothing', 'TEXT')
  ensureColumn('characters', 'weapons', 'TEXT')
  ensureColumn('characters', 'custom_prompt', 'TEXT')
  ensureColumn('characters', 'core_features', 'TEXT')
  ensureColumn('characters', 'costumes', 'TEXT')
  ensureColumn('characters', 'variations', 'TEXT')
  ensureColumn('characters', 'voice_model', "TEXT DEFAULT 'speech-2.8-hd'")
  ensureColumn('characters', 'negative_prompt', 'TEXT')
  ensureColumn('characters', 'speaker_id', 'TEXT')
  ensureColumn('characters', 'accessories', 'TEXT')
  ensureColumn('characters', 'three_views', 'TEXT')
  ensureColumn('characters', 'equip_images', 'TEXT')

  ensureColumn('scenes', 'description', 'TEXT')
  ensureColumn('scenes', 'atmosphere', 'TEXT')
  ensureColumn('scenes', 'lighting', 'TEXT')
  ensureColumn('scenes', 'weather', 'TEXT')
  ensureColumn('scenes', 'season', 'TEXT')
  ensureColumn('scenes', 'style', 'TEXT')
  ensureColumn('scenes', 'custom_prompt', 'TEXT')
  ensureColumn('scenes', 'negative_prompt', 'TEXT')

  ensureColumn('storyboards', 'custom_image_prompt', 'TEXT')
  ensureColumn('storyboards', 'custom_video_prompt', 'TEXT')
  ensureColumn('storyboards', 'negative_prompt', 'TEXT')
  ensureColumn('storyboards', 'first_frame_prompt', 'TEXT')
  ensureColumn('storyboards', 'last_frame_prompt', 'TEXT')
  ensureColumn('storyboards', 'transition_type', "TEXT DEFAULT 'cut'")
  ensureColumn('storyboards', 'transition_duration', 'REAL DEFAULT 0.5')
  ensureColumn('storyboards', 'transition_motive', 'TEXT')
  ensureColumn('storyboards', 'scene_type', 'TEXT')
  ensureColumn('storyboards', 'speaker_id', 'TEXT')
  ensureColumn('storyboards', 'start_state', 'TEXT')
  ensureColumn('storyboards', 'end_state', 'TEXT')
  ensureColumn('storyboards', 'constraints', 'TEXT')
  ensureColumn('image_generations', 'deleted_at', 'TEXT')
  ensureColumn('image_generations', 'prop_id', 'INTEGER')

  ensureColumn('video_generations', 'character_ids', 'TEXT')
  ensureColumn('video_generations', 'negative_prompt', 'TEXT')
  ensureColumn('video_generations', 'scene_type', 'TEXT')
  ensureColumn('video_generations', 'reference_audio_urls', 'TEXT')
  ensureColumn('video_generations', 'block_reason', 'TEXT')

  // ====== 资产验收门禁（asset review gate）======
  ensureColumn('storyboards', 'asset_status', "TEXT DEFAULT 'missing'")

  // ====== 剧本内容指纹门禁（script content fingerprint gate）======
  ensureColumn('episodes', 'script_hash', 'TEXT')
  ensureColumn('storyboards', 'script_hash', 'TEXT')

  // ====== per-shot take 预算 ======
  ensureColumn('storyboards', 'take_count', "INTEGER DEFAULT 0")
  ensureColumn('storyboards', 'take_budget', "INTEGER DEFAULT 3")

  // ====== 多集节奏相位 ======
  ensureColumn('storyboards', 'rhythm_phase', 'TEXT')

  // ====== 逐镜路由（shot routing，对齐 H3-Codex-Drama）======
  ensureColumn('storyboards', 'route', 'TEXT')
  ensureColumn('storyboards', 'route_reason', 'TEXT')
  ensureColumn('video_generations', 'route', 'TEXT')
  ensureColumn('video_generations', 'route_reason', 'TEXT')

  // ====== 关键帧扩展（keyframe）======
  ensureColumn('storyboards', 'keyframe_prompt', 'TEXT')
  ensureColumn('storyboards', 'keyframe_image', 'TEXT')

  ensureColumn('storyboard_characters', 'costume', 'TEXT')
  ensureColumn('image_generations', 'costume', 'TEXT')
  ensureColumn('image_generations', 'color_grade', 'TEXT')
  ensureColumn('image_generations', 'view_type', 'TEXT')
  ensureColumn('image_generations', 'equip_type', 'TEXT')

  ensureColumn('ai_service_providers', 'endpoint_prefix', 'TEXT')
  ensureColumn('ai_service_providers', 'is_recommended', 'INTEGER DEFAULT 0')

  ensureColumn('ai_voices', 'role_tags', 'TEXT')
  ensureColumn('ai_voices', 'reference_audio', 'TEXT')
  ensureColumn('ai_voices', 'prompt_text', 'TEXT')
  ensureColumn('characters', 'role_type', 'TEXT')

  // ====== 六键 Bible 三键（LOCATION_ID / COSTUME_ID / STYLE_ID）======
  ensureColumn('dramas', 'style_id', 'TEXT')
  ensureColumn('characters', 'costume_id', 'TEXT')
  ensureColumn('scenes', 'location_id', 'TEXT')
}

runMigrations()

let currentDb = drizzle(getSqlite(), { schema })

export function rebuildDb(): void {
  runMigrations()
  currentDb = drizzle(getSqlite(), { schema })
}

export const db = new Proxy({} as typeof currentDb, {
  get(_target, prop) {
    const v = (currentDb as any)[prop]
    if (typeof v === 'function') return v.bind(currentDb)
    return v
  },
  set(_target, prop, value) {
    ;(currentDb as any)[prop] = value
    return true
  },
})

export { schema }
export type DB = typeof currentDb
