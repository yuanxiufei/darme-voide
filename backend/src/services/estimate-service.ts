/**
 * 生成前费用预估（对齐参考项目 ArcReel 的生成前预估 + 生成后核算双环）
 *
 * 定位：在发起批量生成前（整剧 / 整集 / 指定分镜），按当前激活的 provider/model
 * 单价与待生成工作负载（图片张数 / 视频秒数 / TTS 字符数）预估总费用，
 * 回答「这一批要烧多少钱、预算还剩多少」。
 *
 * 与 usage-tracking.ts（生成后实际记账 api_usage）配套：估算是花前，记算是花后。
 */
import { eq, and, inArray } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { estimateCost } from './cost-catalog.js'
import { logTaskStart, logTaskSuccess } from '../utils/task-logger.js'

export interface EstimateItem {
  service_type: 'image' | 'video' | 'audio' | 'text'
  provider: string
  model: string
  units: number
  unit: string
  cost: number | null
}

export interface EstimateResult {
  drama_id: number
  episode_id?: number
  scope: string
  generated_at: string
  pending_images: number
  pending_videos: number
  pending_audio_chars: number
  active_image: { provider: string; model: string } | null
  active_video: { provider: string; model: string } | null
  active_audio: { provider: string; model: string } | null
  items: EstimateItem[]
  total_cost: number | null
  unestimatable: string[]
}

/** 读取某服务类型优先级最高的激活配置（isActive + 默认优先 + priority） */
function activeConfig(serviceType: string): { provider: string; model: string; settings: Record<string, any> | null } | null {
  const rows = db
    .select()
    .from(schema.aiServiceConfigs)
    .where(and(
      eq(schema.aiServiceConfigs.serviceType, serviceType),
      eq(schema.aiServiceConfigs.isActive, true),
    ))
    .all()
  if (!rows.length) return null
  rows.sort((a, b) => (b.isDefault ? 1 : 0) - (a.isDefault ? 1 : 0) || (b.priority ?? 0) - (a.priority ?? 0))
  const cfg = rows[0]
  let settings: Record<string, any> | null = null
  if (cfg.settings) {
    try { settings = JSON.parse(cfg.settings) } catch { settings = null }
  }
  return { provider: cfg.provider || '', model: cfg.model || '', settings }
}

/** 查询某分镜是否已有完成的视频生成 */
function hasCompletedVideo(storyboardId: number): boolean {
  return db
    .select({ id: schema.videoGenerations.id })
    .from(schema.videoGenerations)
    .where(and(
      eq(schema.videoGenerations.storyboardId, storyboardId),
      eq(schema.videoGenerations.status, 'completed'),
    ))
    .all().length > 0
}

/**
 * 预估一批待生成分镜的费用
 * @param dramaId 剧 ID
 * @param opts.storyboardIds 可选：限定指定分镜（视为全部待生成）
 * @param opts.episodeId 可选：限定某集（该集未生成视频的分镜为待生成）
 * @param opts.scope 范围描述（覆盖默认）
 */
export function estimatePendingCosts(
  dramaId: number,
  opts: { scope?: string; episodeId?: number; storyboardIds?: number[] } = {},
): EstimateResult {
  logTaskStart('Estimate', 'estimate', { dramaId, episodeId: opts.episodeId })

  // 收集目标分镜
  let storyboards: Array<{ id: number; duration: number | null; dialogue: string | null }>
  if (opts.storyboardIds?.length) {
    storyboards = db
      .select({ id: schema.storyboards.id, duration: schema.storyboards.duration, dialogue: schema.storyboards.dialogue })
      .from(schema.storyboards)
      .where(inArray(schema.storyboards.id, opts.storyboardIds))
      .all()
  } else if (opts.episodeId) {
    storyboards = db
      .select({ id: schema.storyboards.id, duration: schema.storyboards.duration, dialogue: schema.storyboards.dialogue })
      .from(schema.storyboards)
      .where(eq(schema.storyboards.episodeId, opts.episodeId))
      .all()
  } else {
    // 全剧：先取该剧所有集 ID，再查分镜
    const episodeIds = db
      .select({ id: schema.episodes.id })
      .from(schema.episodes)
      .where(eq(schema.episodes.dramaId, dramaId))
      .all()
      .map(e => e.id)
    storyboards = episodeIds.length
      ? db
        .select({ id: schema.storyboards.id, duration: schema.storyboards.duration, dialogue: schema.storyboards.dialogue })
        .from(schema.storyboards)
        .where(inArray(schema.storyboards.episodeId, episodeIds))
        .all()
      : []
  }

  // 待生成：指定分镜全部计入；否则只统计无完成视频的分镜
  const pending = opts.storyboardIds?.length
    ? storyboards
    : storyboards.filter(sb => !hasCompletedVideo(sb.id))

  const items: EstimateItem[] = []
  const unestimatable: string[] = []
  let total = 0
  let hasCost = false

  // 视频（按秒）
  const videoCfg = activeConfig('video')
  const videoSeconds = pending.reduce((s, sb) => s + (sb.duration ?? 5), 0)
  if (videoCfg) {
    const cost = estimateCost('video', videoCfg.provider, videoCfg.model, videoSeconds, videoCfg.settings)
    if (cost == null) unestimatable.push(`video(${videoCfg.provider}/${videoCfg.model}) 无单价`)
    else { total += cost; hasCost = true }
    items.push({ service_type: 'video', provider: videoCfg.provider, model: videoCfg.model, units: videoSeconds, unit: 'second', cost })
  } else {
    unestimatable.push('video 未配置激活服务')
  }

  // 图片（按张 = 待生成分镜数）
  const imageCfg = activeConfig('image')
  const pendingImages = pending.length
  if (imageCfg) {
    const cost = estimateCost('image', imageCfg.provider, imageCfg.model, pendingImages, imageCfg.settings)
    if (cost == null) unestimatable.push(`image(${imageCfg.provider}/${imageCfg.model}) 无单价`)
    else { total += cost; hasCost = true }
    items.push({ service_type: 'image', provider: imageCfg.provider, model: imageCfg.model, units: pendingImages, unit: 'image', cost })
  } else {
    unestimatable.push('image 未配置激活服务')
  }

  // 音频 TTS（按字）
  const audioCfg = activeConfig('audio')
  const audioChars = pending.reduce((s, sb) => s + (sb.dialogue?.length ?? 0), 0)
  if (audioCfg && audioChars > 0) {
    const cost = estimateCost('audio', audioCfg.provider, audioCfg.model, audioChars, audioCfg.settings)
    if (cost == null) unestimatable.push(`audio(${audioCfg.provider}/${audioCfg.model}) 无单价`)
    else { total += cost; hasCost = true }
    items.push({ service_type: 'audio', provider: audioCfg.provider, model: audioCfg.model, units: audioChars, unit: 'char', cost })
  } else if (audioChars === 0) {
    // 无对白则不预估音频
  } else {
    unestimatable.push('audio 未配置激活服务')
  }

  logTaskSuccess('Estimate', 'estimated', {
    dramaId, episodeId: opts.episodeId, pendingVideos: pending.length, videoSeconds, audioChars, totalCost: hasCost ? total : null,
  })

  return {
    drama_id: dramaId,
    episode_id: opts.episodeId,
    scope: opts.scope || (opts.episodeId ? `episode-${opts.episodeId}` : `drama-${dramaId}`),
    generated_at: new Date().toISOString(),
    pending_images: pendingImages,
    pending_videos: pending.length,
    pending_audio_chars: audioChars,
    active_image: imageCfg ? { provider: imageCfg.provider, model: imageCfg.model } : null,
    active_video: videoCfg ? { provider: videoCfg.provider, model: videoCfg.model } : null,
    active_audio: audioCfg ? { provider: audioCfg.provider, model: audioCfg.model } : null,
    items,
    total_cost: hasCost ? Math.round(total * 100) / 100 : null,
    unestimatable,
  }
}
