/**
 * Drizzle schema — 精确匹配现有 SQLite 数据库列名
 * 从 PRAGMA table_info() 逆向生成
 */
import { sqliteTable, text, integer, real, primaryKey } from 'drizzle-orm/sqlite-core'

export const dramas = sqliteTable('dramas', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  title: text('title').notNull(),
  description: text('description'),
  genre: text('genre'),
  style: text('style').default('realistic'),
  styleId: text('style_id'),
  totalEpisodes: integer('total_episodes').default(1),
  totalDuration: integer('total_duration').default(0),
  status: text('status').notNull().default('draft'),
  thumbnail: text('thumbnail'),
  tags: text('tags'),
  metadata: text('metadata'),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  deletedAt: text('deleted_at'),
})

export const episodes = sqliteTable('episodes', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  dramaId: integer('drama_id').notNull(),
  episodeNumber: integer('episode_number').notNull(),
  title: text('title').notNull(),
  content: text('content'),
  scriptContent: text('script_content'),
  description: text('description'),
  duration: integer('duration').default(0),
  status: text('status').default('draft'),
  videoUrl: text('video_url'),
  thumbnail: text('thumbnail'),
  imageConfigId: integer('image_config_id'),
  videoConfigId: integer('video_config_id'),
  audioConfigId: integer('audio_config_id'),
  bgmUrl: text('bgm_url'),
  bgmVolume: real('bgm_volume').default(0.3),
  bgmFadeIn: real('bgm_fade_in').default(1.5),
  bgmFadeOut: real('bgm_fade_out').default(2.0),
  // 剧本内容指纹门禁：script_hash 记录剧本内容快照，
  // 剧本变更后自动重算，用于检测下游分镜/资产是否过期（对齐 H3-Codex-Drama asset gate）
  scriptHash: text('script_hash'),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  deletedAt: text('deleted_at'),
})

export const characters = sqliteTable('characters', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  dramaId: integer('drama_id').notNull(),
  name: text('name').notNull(),
  role: text('role'),
  roleType: text('role_type'),
  description: text('description'),
  appearance: text('appearance'),
  personality: text('personality'),
  voiceStyle: text('voice_style'),
  speakerId: text('speaker_id'),
  costumeId: text('costume_id'),
  imageUrl: text('image_url'),
  referenceImages: text('reference_images'),
  seedValue: text('seed_value'),
  sortOrder: integer('sort_order'),
  localPath: text('local_path'),
  voiceSampleUrl: text('voice_sample_url'),
  voiceProvider: text('voice_provider'),
  voiceSpeed: real('voice_speed').default(1.0),
  voiceEmotion: text('voice_emotion').default('happy'),
  voicePitch: real('voice_pitch').default(0),
  clothing: text('clothing'),
  weapons: text('weapons'),
  customPrompt: text('custom_prompt'),
  negativePrompt: text('negative_prompt'),
  coreFeatures: text('core_features'),
  costumes: text('costumes'),
  variations: text('variations'),
  accessories: text('accessories'),
  threeViews: text('three_views'),
  equipImages: text('equip_images'),
  voiceModel: text('voice_model').default('speech-2.8-hd'),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  deletedAt: text('deleted_at'),
})

// Episode-Character many-to-many
export const episodeCharacters = sqliteTable('episode_characters', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  episodeId: integer('episode_id').notNull(),
  characterId: integer('character_id').notNull(),
  createdAt: text('created_at').notNull(),
})

// Episode-Scene many-to-many
export const episodeScenes = sqliteTable('episode_scenes', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  episodeId: integer('episode_id').notNull(),
  sceneId: integer('scene_id').notNull(),
  createdAt: text('created_at').notNull(),
})

export const scenes = sqliteTable('scenes', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  dramaId: integer('drama_id').notNull(),
  episodeId: integer('episode_id'),
  location: text('location').notNull(),
  locationId: text('location_id'),
  time: text('time').notNull(),
  prompt: text('prompt').notNull(),
  storyboardCount: integer('storyboard_count').default(1),
  imageUrl: text('image_url'),
  status: text('status').default('pending'),
  localPath: text('local_path'),
  description: text('description'),
  atmosphere: text('atmosphere'),
  lighting: text('lighting'),
  weather: text('weather'),
  season: text('season'),
  style: text('style'),
  customPrompt: text('custom_prompt'),
  negativePrompt: text('negative_prompt'),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  deletedAt: text('deleted_at'),
})

export const storyboards = sqliteTable('storyboards', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  episodeId: integer('episode_id').notNull(),
  sceneId: integer('scene_id'),
  storyboardNumber: integer('storyboard_number').notNull(),
  title: text('title'),
  location: text('location'),
  time: text('time'),
  shotType: text('shot_type'),
  sceneType: text('scene_type'),
  speakerId: text('speaker_id'),
  angle: text('angle'),
  movement: text('movement'),
  action: text('action'),
  result: text('result'),
  atmosphere: text('atmosphere'),
  imagePrompt: text('image_prompt'),
  videoPrompt: text('video_prompt'),
  bgmPrompt: text('bgm_prompt'),
  soundEffect: text('sound_effect'),
  dialogue: text('dialogue'),
  description: text('description'),
  duration: integer('duration').default(0),
  composedImage: text('composed_image'),
  firstFrameImage: text('first_frame_image'),
  lastFrameImage: text('last_frame_image'),
  // 关键帧扩展（对齐参考项目 8-16 张关键帧理念）：中段关键帧锁定动作/道具/机位中间态
  keyframePrompt: text('keyframe_prompt'),
  keyframeImage: text('keyframe_image'),
  referenceImages: text('reference_images'),
  videoUrl: text('video_url'),
  ttsAudioUrl: text('tts_audio_url'),
  subtitleUrl: text('subtitle_url'),
  composedVideoUrl: text('composed_video_url'),
  customImagePrompt: text('custom_image_prompt'),
  customVideoPrompt: text('custom_video_prompt'),
  negativePrompt: text('negative_prompt'),
  firstFramePrompt: text('first_frame_prompt'),
  lastFramePrompt: text('last_frame_prompt'),
  transitionType: text('transition_type').default('cut'),
  transitionDuration: real('transition_duration').default(0.5),
  transitionMotive: text('transition_motive'),
  // 连续性状态机字段（v3）
  startState: text('start_state'),
  endState: text('end_state'),
  constraints: text('constraints'),
  status: text('status').default('pending'),
  // 资产验收门禁（对齐参考项目 asset review）：missing=未生成 / approved=验收通过 / needs_regeneration=需重生成
  assetStatus: text('asset_status').default('missing'),
  // 剧本内容指纹门禁：记录本分镜生成时对应的剧本 script_hash，
  // 与 episodes.script_hash 不一致即视为剧本变更后过期（stale），需重新拆解
  scriptHash: text('script_hash'),
  // per-shot take 预算：每个分镜允许的生成尝试次数（默认 3），超预算阻断生成，
  // 重新拆解分镜时重置；对齐参考项目 per-shot take budget
  takeCount: integer('take_count').default(0),
  takeBudget: integer('take_budget').default(3),
  // 多集节奏相位：setup/development/climax/resolution，按分镜在集内的时长位置自动分配；
  // 跨集统计节奏分布，供 storyboard_breaker 保持节奏一致（对齐参考项目 multi-episode rhythm phase）
  rhythmPhase: text('rhythm_phase'),
  // 逐镜路由（对齐参考项目 H3-Codex-Drama shot routing）：每镜显式决策视频生成路线，
  // text_to_video / first_frame_to_video / first_last_frame / reference_to_video / keyframe_to_video / video_editor，
  // 记录决策结果与原因，供管线提交/可复现账本引用
  route: text('route'),
  routeReason: text('route_reason'),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  deletedAt: text('deleted_at'),
})

export const storyboardCharacters = sqliteTable('storyboard_characters', {
  storyboardId: integer('storyboard_id').notNull(),
  characterId: integer('character_id').notNull(),
  costume: text('costume'),
}, (table) => ({
  pk: primaryKey({ columns: [table.storyboardId, table.characterId] }),
}))

// ====== 连续性状态机（v3）======
// 跨镜持久状态：场景空间 / 角色姿态 / 道具状态 / 线索揭示 / 动作因果 / 转场动机
export const continuityStates = sqliteTable('continuity_states', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  episodeId: integer('episode_id').notNull(),
  storyboardId: integer('storyboard_id'),
  sceneId: integer('scene_id'),
  stateType: text('state_type').notNull(), // scene_space / character_pose / prop_state / clue_reveal / causal_link / transition_motive
  entityKey: text('entity_key').notNull(), // 如 "角色_林晚" / "道具_玉坠" / "线索_玉佩" / "场景_大堂"
  stateValue: text('state_value').notNull(),
  constraints: text('constraints'),        // 禁止变化清单（服装/站位/灯光等）
  meta: text('meta'),                      // JSON 扩展
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
})

export const aiServiceConfigs = sqliteTable('ai_service_configs', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  serviceType: text('service_type').notNull(),
  provider: text('provider'),
  name: text('name').notNull(),
  baseUrl: text('base_url').notNull(),
  apiKey: text('api_key').notNull(),
  model: text('model'),
  endpoint: text('endpoint'),
  queryEndpoint: text('query_endpoint'),
  priority: integer('priority').default(0),
  isDefault: integer('is_default', { mode: 'boolean' }).default(false),
  isActive: integer('is_active', { mode: 'boolean' }).default(true),
  settings: text('settings'),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  // 注意: 此表无 deleted_at
})

export const aiServiceProviders = sqliteTable('ai_service_providers', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  name: text('name').notNull(),
  displayName: text('display_name'),
  serviceType: text('service_type').notNull(),
  provider: text('provider').notNull(),
  defaultUrl: text('default_url'),
  presetModels: text('preset_models'),
  description: text('description'),
  endpointPrefix: text('endpoint_prefix'),
  isRecommended: integer('is_recommended', { mode: 'boolean' }).default(false),
  isActive: integer('is_active', { mode: 'boolean' }).default(true),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
})

export const aiVoices = sqliteTable('ai_voices', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  voiceId: text('voice_id').notNull().unique(),   // MiniMax voice_id
  voiceName: text('voice_name').notNull(),         // 中文名
  description: text('description'),                // 描述数组 JSON
  language: text('language'),                     // 语言标签
  provider: text('provider').notNull(),           // minimax
  roleTags: text('role_tags'),                    // 角色类型标签 JSON 数组（旁白/主角/反派/配角）
  referenceAudio: text('reference_audio'),         // 零样本克隆参考音频相对路径（CosyVoice）
  promptText: text('prompt_text'),                 // 参考音频对应文本（CosyVoice 零样本）
  createdAt: text('created_at').notNull(),
})

export const agentConfigs = sqliteTable('agent_configs', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  agentType: text('agent_type').notNull(),
  name: text('name').notNull(),
  description: text('description'),
  model: text('model'),
  systemPrompt: text('system_prompt'),
  temperature: real('temperature'),
  maxTokens: integer('max_tokens'),
  maxIterations: integer('max_iterations'),
  skills: text('skills'),             // JSON: [{ id, enabled, priority }]
  isActive: integer('is_active', { mode: 'boolean' }).default(true),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  deletedAt: text('deleted_at'),
})

export const imageGenerations = sqliteTable('image_generations', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  storyboardId: integer('storyboard_id'),
  dramaId: integer('drama_id'),
  sceneId: integer('scene_id'),
  characterId: integer('character_id'),
  propId: integer('prop_id'),
  imageType: text('image_type'),
  frameType: text('frame_type'),
  provider: text('provider'),
  prompt: text('prompt'),
  negativePrompt: text('negative_prompt'),
  model: text('model'),
  size: text('size'),
  quality: text('quality'),
  style: text('style'),
  steps: integer('steps'),
  cfgScale: real('cfg_scale'),
  seed: integer('seed'),
  imageUrl: text('image_url'),
  localPath: text('local_path'),
  status: text('status').default('pending'),
  taskId: text('task_id'),
  errorMsg: text('error_msg'),
  width: integer('width'),
  height: integer('height'),
  referenceImages: text('reference_images'),
  costume: text('costume'),
  colorGrade: text('color_grade'),
  viewType: text('view_type'),
  equipType: text('equip_type'),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  completedAt: text('completed_at'),
  deletedAt: text('deleted_at'),
})

export const videoGenerations = sqliteTable('video_generations', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  storyboardId: integer('storyboard_id'),
  dramaId: integer('drama_id'),
  provider: text('provider'),
  prompt: text('prompt'),
  negativePrompt: text('negative_prompt'),
  model: text('model'),
  imageGenId: integer('image_gen_id'),
  referenceMode: text('reference_mode'),
  imageUrl: text('image_url'),
  firstFrameUrl: text('first_frame_url'),
  lastFrameUrl: text('last_frame_url'),
  referenceImageUrls: text('reference_image_urls'),
  duration: integer('duration'),
  fps: integer('fps'),
  resolution: text('resolution'),
  aspectRatio: text('aspect_ratio'),
  style: text('style'),
  motionLevel: integer('motion_level'),
  cameraMotion: text('camera_motion'),
  seed: integer('seed'),
  videoUrl: text('video_url'),
  localPath: text('local_path'),
  status: text('status').default('pending'),
  taskId: text('task_id'),
  errorMsg: text('error_msg'),
  width: integer('width'),
  height: integer('height'),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  completedAt: text('completed_at'),
  deletedAt: text('deleted_at'),
  characterIds: text('character_ids'),
  sceneType: text('scene_type'),
  referenceAudioUrls: text('reference_audio_urls'),
  // 资产验收门禁：视频任务因资产缺失被阻断时的原因（如 blocked_by_missing_asset）
  blockReason: text('block_reason'),
  // 逐镜路由快照：本次提交时采用的生成路线（text_to_video / first_frame_to_video / ...）
  route: text('route'),
  routeReason: text('route_reason'),
})

export const videoMerges = sqliteTable('video_merges', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  episodeId: integer('episode_id'),
  dramaId: integer('drama_id'),
  title: text('title'),
  provider: text('provider').notNull(),
  model: text('model').notNull(),
  status: text('status').default('pending'),
  scenes: text('scenes'), // JSON
  mergedUrl: text('merged_url'),
  duration: integer('duration'),
  taskId: text('task_id'),
  errorMsg: text('error_msg'),
  createdAt: text('created_at').notNull(),
  completedAt: text('completed_at'),
  deletedAt: text('deleted_at'),
})

// 镜头级 QC 打分（唇形同步 / 角色一致性 / 连续性）
export const videoQualityChecks = sqliteTable('video_quality_checks', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  storyboardId: integer('storyboard_id'),
  videoGenerationId: integer('video_generation_id'),
  dramaId: integer('drama_id'),
  episodeId: integer('episode_id'),
  lipSyncScore: integer('lip_sync_score'),
  characterConsistencyScore: integer('character_consistency_score'),
  continuityScore: integer('continuity_score'),
  overallScore: integer('overall_score'),
  issues: text('issues'),
  dimensions: text('dimensions'),
  status: text('status').default('pending'),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
})

// ====== 资源库模板表 ======

// 角色库模板
export const characterTemplates = sqliteTable('character_templates', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  name: text('name').notNull(),
  category: text('category').default('通用'),
  description: text('description'),
  // 视觉形象
  appearance: text('appearance').notNull(),
  personality: text('personality'),
  clothingStyle: text('clothing_style'),
  expression: text('expression'),
  gender: text('gender'),
  ageGroup: text('age_group'),
  imageUrl: text('image_url'),
  referenceImages: text('reference_images'),
  // 声音配置
  voiceStyle: text('voice_style'),
  voiceProvider: text('voice_provider'),
  voiceConfig: text('voice_config'), // JSON: { tone, pitch, speed, emotion, ... }
  // 标签与元数据
  tags: text('tags'),               // JSON string array
  metadata: text('metadata'),       // 扩展 JSON
  sourceDramaId: integer('source_drama_id'),
  usageCount: integer('usage_count').default(0),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  deletedAt: text('deleted_at'),
})

// 场景库模板
export const sceneTemplates = sqliteTable('scene_templates', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  name: text('name').notNull(),
  category: text('category').default('通用'),
  description: text('description'),
  // 视觉环境
  location: text('location'),
  atmosphere: text('atmosphere'),
  lighting: text('lighting'),
  timeOfDay: text('time_of_day'),  // 清晨/白天/黄昏/夜晚/凌晨
  style: text('style'),
  season: text('season'),
  weather: text('weather'),
  imageUrl: text('image_url'),
  referenceImages: text('reference_images'),
  prompt: text('prompt'),          // 完整图片生成 prompt
  // 标签与元数据
  tags: text('tags'),
  metadata: text('metadata'),
  sourceDramaId: integer('source_drama_id'),
  usageCount: integer('usage_count').default(0),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  deletedAt: text('deleted_at'),
})

// ====== 物品库（props，连续性状态机 v3）======
export const propTemplates = sqliteTable('prop_templates', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  dramaId: integer('drama_id').notNull(),
  name: text('name').notNull(),
  category: text('category').default('道具'),  // 道具/信物/线索/文件/法器/食物/随身物/其他
  description: text('description'),
  appearance: text('appearance'),               // 外观描述（颜色/材质/形状/大小）
  sizeHint: text('size_hint'),                  // 尺寸参照（如「掌心大小」「半人高」）
  holder: text('holder'),                       // 惯常持有者（角色名）
  keyClue: text('key_clue'),                    // 是否关键线索：是/否
  customPrompt: text('image_prompt'),
  negativePrompt: text('negative_prompt'),
  imageUrl: text('image_url'),
  referenceImages: text('reference_images'),    // JSON 数组
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  deletedAt: text('deleted_at'),
})

// 物品-剧集关联
export const episodeProps = sqliteTable('episode_props', {
  episodeId: integer('episode_id').notNull(),
  propId: integer('prop_id').notNull(),
}, (table) => ({
  pk: primaryKey({ columns: [table.episodeId, table.propId] }),
}))

// 物品-分镜关联
export const storyboardProps = sqliteTable('storyboard_props', {
  storyboardId: integer('storyboard_id').notNull(),
  propId: integer('prop_id').notNull(),
}, (table) => ({
  pk: primaryKey({ columns: [table.storyboardId, table.propId] }),
}))

// 兵器库模板
export const weaponTemplates = sqliteTable('weapon_templates', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  name: text('name').notNull(),
  category: text('category').default('剑'),   // 剑/刀/枪/棍/斧/锤/弓/弩/扇/鞭/杖/暗器/法宝/其他
  type: text('type'),                          // 近战/远程/暗器/法宝
  description: text('description'),
  // 外观与属性
  appearance: text('appearance'),
  material: text('material'),
  attributes: text('attributes'),              // JSON: { power, rarity, element, special, ... }
  rank: text('rank'),                          // 品级: 凡品/灵品/仙品/神品
  ownerCharacterName: text('owner_character_name'),
  imageUrl: text('image_url'),
  referenceImages: text('reference_images'),
  // 标签与元数据
  tags: text('tags'),
  metadata: text('metadata'),
  sourceDramaId: integer('source_drama_id'),
  usageCount: integer('usage_count').default(0),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  deletedAt: text('deleted_at'),
})

// 服装库模板
export const costumeTemplates = sqliteTable('costume_templates', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  name: text('name').notNull(),
  category: text('category').default('通用'),
  description: text('description'),
  // 分类维度
  style: text('style'),                        // 古风/现代/仙侠/武侠/科幻/宫廷/民俗/其他
  bodyPart: text('body_part'),                 // 全身/上衣/下装/外套/鞋履/头饰/配饰
  material: text('material'),
  colorScheme: text('color_scheme'),
  season: text('season'),                      // 春/夏/秋/冬
  // 外观描述
  appearance: text('appearance'),
  imageUrl: text('image_url'),
  referenceImages: text('reference_images'),
  // 标签与元数据
  tags: text('tags'),
  metadata: text('metadata'),
  sourceDramaId: integer('source_drama_id'),
  usageCount: integer('usage_count').default(0),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  deletedAt: text('deleted_at'),
})

// 预设表（校色预设、生成预设等）
export const presets = sqliteTable('presets', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  type: text('type').notNull(),          // colorGrade / character / generation / ...
  name: text('name').notNull(),
  config: text('config'),                // JSON 配置
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
})

// ====== 用量统计与成本估算（对齐参考项目 ArcReel usage_repo）======
// 每次模型调用记一条账；cost_amount 为按单价目录估算（settings.pricing 可覆盖）；
// 本地模型 is_local=1 不计费。支持按项目/集/服务类型/提供商汇总，回答「这集花了多少、重拍烧了多少」。
export const apiUsage = sqliteTable('api_usage', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  serviceType: text('service_type').notNull(), // image / video / audio / text
  provider: text('provider').notNull(),
  model: text('model').notNull(),
  // 归属维度（可空：项目级/集级/镜头级）
  dramaId: integer('drama_id'),
  episodeId: integer('episode_id'),
  storyboardId: integer('storyboard_id'),
  // 关联生成任务（用于追溯重拍/fallback 成本）
  imageGenerationId: integer('image_generation_id'),
  videoGenerationId: integer('video_generation_id'),
  // 用量与成本
  units: integer('units'),              // 计费单位数（图片张数 / 视频秒数 / 音频字符数）
  costAmount: real('cost_amount'),      // 估算成本（元），查不到单价时为 null
  currency: text('currency').default('CNY'),
  isLocal: integer('is_local', { mode: 'boolean' }).default(false),
  status: text('status').default('submitted'), // submitted / completed / failed
  retryCount: integer('retry_count').default(0), // 同一任务第几次尝试（0=首次，含模型 fallback）
  meta: text('meta'),                   // JSON 扩展
  createdAt: text('created_at').notNull(),
})

// ====== 资产版本历史/回滚（对齐参考项目 ArcReel artifact_version_provenance）======
// 每次图片/视频生成成功自动留档，支持查看历史版本与回滚切换。
// 同一资产（asset_type + asset_id + media_type + frame_type）只有一个 status='current'。
export const assetVersions = sqliteTable('asset_versions', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  assetType: text('asset_type').notNull(),   // storyboard / character / scene / prop
  assetId: integer('asset_id').notNull(),
  mediaType: text('media_type').notNull(),   // image / video
  frameType: text('frame_type'),             // 分镜图片子位置：composed / first_frame / last_frame / keyframe
  version: integer('version').notNull(),     // 同资产递增，从 1 开始
  assetUrl: text('asset_url').notNull(),     // 当前生效的 URL / 本地路径
  provider: text('provider'),
  model: text('model'),
  prompt: text('prompt'),
  generationId: integer('generation_id'),    // image_generations / video_generations 的 id
  meta: text('meta'),                        // JSON 扩展
  status: text('status').default('current'), // current / historical
  createdAt: text('created_at').notNull(),
})

// ====== 风格 Profile 提炼（对齐参考项目 H3-Codex-Drama Profile Distiller）======
// 从参考视频/素材提炼可复用的 house style：叙事节奏、镜头语言、音频字幕、验收规则。
// 区分三类来源：measurement facts（测量事实）/ visual inference（视觉推断）/ user preference（用户偏好）。
// 激活后注入 storyboard_breaker 等生成 Agent 的指令，跨集保持统一风格。
export const styleProfiles = sqliteTable('style_profiles', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  dramaId: integer('drama_id'),
  name: text('name').notNull(),
  description: text('description'),
  source: text('source'),                    // 参考素材说明（视频路径/链接/文字描述）
  storytelling: text('storytelling'),        // JSON：叙事节奏/悬念密度/转场偏好
  shotPatterns: text('shot_patterns'),       // JSON：景别分布/机位/运镜偏好
  audioCaptions: text('audio_captions'),     // JSON：音效/配乐/字幕风格
  qcRules: text('qc_rules'),                 // JSON：验收规则覆盖（阈值等）
  facts: text('facts'),                      // JSON：测量事实（可从 ffprobe 提取）
  inferences: text('inferences'),            // JSON：视觉推断
  preferences: text('preferences'),          // JSON：用户偏好（手工标注）
  isActive: integer('is_active', { mode: 'boolean' }).default(false),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  deletedAt: text('deleted_at'),
})
