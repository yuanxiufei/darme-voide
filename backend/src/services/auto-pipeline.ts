/**
 * 全自动管线编排器（Auto Pipeline Orchestrator）
 *
 * 「一句话梗概 → 整集短剧」：自动串联 4 个 LLM Agent
 *   script_rewriter → extractor → voice_assigner → storyboard_breaker
 * 并可选接入媒体生成阶段：图片首帧 → 图生视频 → 单镜头合成（TTS+字幕）→ 整集拼接。
 * 全程无人值守，前端只需轮询 episode.status。
 *
 * ## 幂等与崩溃恢复设计
 * 1. 以 `episode.status`（`auto:*` 前缀）为单向状态机，每个阶段执行成功才推进到下一阶段。
 * 2. Agent 阶段工具本身幂等（save_script 覆盖、save_dedup_* 去重、save_storyboards 重建、
 *    assign_voice 覆盖），重跑安全，不会产生脏数据。
 * 3. 媒体阶段提交前检查已有产物（firstFrameImage / videoUrl / composedVideoUrl）或已有
 *    processing 任务，崩溃重启后不会重复提交（避免重复扣费）。
 * 4. 启动时 `recoverAutoPipelineOnStartup()` 扫描中间态 episode 自动续跑（幂等）。
 */
import { db, schema } from '../db/index.js'
import { eq, and, inArray, asc } from 'drizzle-orm'
import { now } from '../utils/response.js'
import { runAgentWithRetry } from '../agents/index.js'
import { generateImage } from './image-generation.js'
import { generateVideo } from './video-generation.js'
import { composeStoryboard } from './ffmpeg-compose.js'
import { mergeEpisodeVideos } from './ffmpeg-merge.js'
import { logTaskStart, logTaskSuccess, logTaskError, logTaskWarn, logTaskProgress } from '../utils/task-logger.js'
import { publishPipelineEvent } from '../utils/sse-hub.js'
import { STORYBOARD_IMAGE_NEGATIVE, VIDEO_NEGATIVE, getStoryboardReferenceImages, getStoryboardReferenceAudioUrls } from '../shared/prompt-utils.js'
import { getActiveConfig, getConfigById } from './ai.js'

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

/** 媒体阶段轮询参数 */
const MEDIA_WAIT_TIMEOUT_MS = 30 * 60 * 1000
const MEDIA_POLL_INTERVAL_MS = 5000

/** 支持多参考图（referenceMode='multiple'）的视频提供商；其余仅首尾帧 */
const MULTI_REFERENCE_PROVIDERS = new Set(['volcengine', 'vidu', 'minimax'])

/** 管线阶段状态（存 episode.status，`auto:` 前缀避免与业务状态 draft/completed 冲突） */
export const AUTO_STATUS = {
  queued: 'auto:queued',
  scripting: 'auto:scripting',
  extracting: 'auto:extracting',
  storyboarding: 'auto:storyboarding',
  voicing: 'auto:voicing',
  imaging: 'auto:imaging',
  videoing: 'auto:videoing',
  composing: 'auto:composing',
  merging: 'auto:merging',
  done: 'auto:done',
  failed: 'auto:failed',
} as const

/** 阶段顺序（用于幂等判断：当前 status 索引 >= 某阶段 next 索引即视为该阶段已完成） */
const STAGE_SEQUENCE: string[] = [
  AUTO_STATUS.queued,
  AUTO_STATUS.scripting,
  AUTO_STATUS.extracting,
  AUTO_STATUS.voicing,
  AUTO_STATUS.storyboarding,
  AUTO_STATUS.imaging,
  AUTO_STATUS.videoing,
  AUTO_STATUS.composing,
  AUTO_STATUS.merging,
  AUTO_STATUS.done,
]

/** 中间态（崩溃恢复扫描范围） */
const IN_FLIGHT_STATUSES: string[] = [
  AUTO_STATUS.queued,
  AUTO_STATUS.scripting,
  AUTO_STATUS.extracting,
  AUTO_STATUS.voicing,
  AUTO_STATUS.storyboarding,
  AUTO_STATUS.imaging,
  AUTO_STATUS.videoing,
  AUTO_STATUS.composing,
  AUTO_STATUS.merging,
]

export interface AutoPipelineOptions {
  /** 一句话梗概 / 原始剧情内容 */
  premise: string
  title?: string
  genre?: string
  style?: string
  episodeCount?: number
  imageConfigId?: number
  videoConfigId?: number
  audioConfigId?: number
  /** 媒体阶段开关（依赖自动补全：video 隐含 image，compose 隐含 image+video，merge 隐含全部） */
  withImages?: boolean
  withVideos?: boolean
  withCompose?: boolean
  withMerge?: boolean
}

type EpisodeRow = typeof schema.episodes.$inferSelect
type StoryboardRow = typeof schema.storyboards.$inferSelect

function normalizeOptions(opts: AutoPipelineOptions): Required<Pick<AutoPipelineOptions, 'premise'>> & AutoPipelineOptions {
  return {
    ...opts,
    episodeCount: opts.episodeCount && opts.episodeCount > 0 ? Math.floor(opts.episodeCount) : 1,
    title: opts.title?.trim() || opts.premise.slice(0, 24),
    genre: opts.genre?.trim() || '短剧',
    style: opts.style?.trim() || 'realistic',
    withImages: !!opts.withImages,
    withVideos: !!opts.withVideos,
    withCompose: !!opts.withCompose,
    withMerge: !!opts.withMerge,
  }
}

/** 依赖补全：计算各媒体阶段是否真正需要执行 */
function mediaNeeds(opts: AutoPipelineOptions) {
  const needImage = !!(opts.withImages || opts.withVideos || opts.withCompose || opts.withMerge)
  const needVideo = !!(opts.withVideos || opts.withCompose || opts.withMerge)
  const needCompose = !!(opts.withCompose || opts.withMerge)
  const needMerge = !!opts.withMerge
  return { needImage, needVideo, needCompose, needMerge }
}

function getEpisode(episodeId: number): EpisodeRow | undefined {
  return db.select().from(schema.episodes).where(eq(schema.episodes.id, episodeId)).get()
}

/** drama 是否已软删（删除竞态保护：已删剧本不再写数据，防止管线复活已删对象） */
function isDramaDeleted(dramaId: number): boolean {
  const drama = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).get()
  return !drama || !!drama.deletedAt
}

function getStoryboards(episodeId: number): StoryboardRow[] {
  return db
    .select()
    .from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, episodeId))
    .orderBy(asc(schema.storyboards.storyboardNumber))
    .all()
}

function setStatus(episodeId: number, status: string) {
  db.update(schema.episodes).set({ status, updatedAt: now() }).where(eq(schema.episodes.id, episodeId)).run()
  // 进度推送：状态推进即发布 SSE 事件（publish 永不 throw，绝不拖垮管线）
  const dramaId = getEpisode(episodeId)?.dramaId
  if (dramaId != null) publishPipelineEvent(dramaId, { type: 'status', episodeId, status })
}

/** 当前 status 是否已越过 nextStatus（即该阶段已完成） */
function isStageCompleted(currentStatus: string | null, nextStatus: string): boolean {
  if (!currentStatus || currentStatus === AUTO_STATUS.failed) return false
  const curIdx = STAGE_SEQUENCE.indexOf(currentStatus)
  const nextIdx = STAGE_SEQUENCE.indexOf(nextStatus)
  if (nextIdx === -1) return true
  if (curIdx === -1) return false
  return curIdx >= nextIdx
}

// ============================================================
// Agent 阶段
// ============================================================

async function runScriptStage(episodeId: number, dramaId: number): Promise<void> {
  await runAgentWithRetry(
    'script_rewriter',
    episodeId,
    dramaId,
    '请读取本集的原始剧情内容（梗概/小说），将其改写为符合平台规范的完整短剧剧本（含场景划分、对白、动作与心理描写），并保存到本集。',
  )
  const ep = getEpisode(episodeId)
  if (!ep?.scriptContent?.trim()) {
    throw new Error('script_rewriter 未产出剧本（script_content 为空）')
  }
}

async function runExtractStage(episodeId: number, dramaId: number): Promise<void> {
  await runAgentWithRetry(
    'extractor',
    episodeId,
    dramaId,
    '请从本集剧本中提取去重后的角色与场景，并保存关联到本集。',
  )
}

async function runStoryboardStage(episodeId: number, dramaId: number): Promise<void> {
  await runAgentWithRetry(
    'storyboard_breaker',
    episodeId,
    dramaId,
    '请将本集剧本按场景拆分为分镜，为每个分镜生成画面描述（image_prompt）与运镜提示（video_prompt）及对白，并保存。',
  )
  const sbs = getStoryboards(episodeId)
  if (sbs.length === 0) {
    throw new Error('storyboard_breaker 未产出分镜（storyboards 为空）')
  }
}

async function runVoiceStage(episodeId: number, dramaId: number): Promise<void> {
  await runAgentWithRetry(
    'voice_assigner',
    episodeId,
    dramaId,
    '请从可用音色库中为本剧角色分配最匹配的音色，并保存。',
  )
}

// ============================================================
// 媒体阶段
// ============================================================

/** 提交缺失的首帧图（幂等：已有首帧 / 已有 processing/completed 任务则跳过） */
async function submitMissingImages(episodeId: number, dramaId: number, configId?: number): Promise<number> {
  const sbs = getStoryboards(episodeId)
  let submitted = 0
  for (const sb of sbs) {
    if (sb.firstFrameImage) continue
    const existing = db
      .select()
      .from(schema.imageGenerations)
      .where(and(eq(schema.imageGenerations.storyboardId, sb.id), eq(schema.imageGenerations.frameType, 'first_frame')))
      .all()
    if (existing.some((g) => g.status === 'processing' || g.status === 'pending' || g.status === 'completed')) continue
    const prompt = sb.imagePrompt || sb.description || '短剧分镜画面，电影感构图'
    await generateImage({
      storyboardId: sb.id,
      dramaId,
      prompt,
      negativePrompt: STORYBOARD_IMAGE_NEGATIVE,
      frameType: 'first_frame',
      configId,
    })
    submitted++
  }
  return submitted
}

/** 提交缺失的视频（幂等：需首帧已就绪 + 无 processing/completed 任务） */
async function submitMissingVideos(episodeId: number, dramaId: number, configId?: number): Promise<number> {
  const sbs = getStoryboards(episodeId)
  const config = configId ? getConfigById(configId) : getActiveConfig('video')
  const provider = (config?.provider || '').toLowerCase()
  const canMultiRef = MULTI_REFERENCE_PROVIDERS.has(provider)
  let submitted = 0
  for (const sb of sbs) {
    if (sb.videoUrl) continue
    if (!sb.firstFrameImage) continue
    const existing = db
      .select()
      .from(schema.videoGenerations)
      .where(eq(schema.videoGenerations.storyboardId, sb.id))
      .all()
    if (existing.some((v) => v.status === 'processing' || v.status === 'pending' || v.status === 'completed')) continue

    // scene_type 路由：对话/多人/争吵类场景在支持多参考图的 provider 上传「角色+场景」参考图，
    // 保证多人同框一致；其余场景保持首帧图生视频（MiniMax/万相仅支持首尾帧，自动降级为 single）
    const sceneType = sb.sceneType || ''
    const isDialogue = /dialogue|meeting|argument|conversation|multi/i.test(sceneType)
    const referenceImages = isDialogue && canMultiRef ? getStoryboardReferenceImages(sb.id).slice(0, 9) : []
    // H3 音视频联合生成：对话类镜头带出场角色声线样本作为参考音频（Ref2VA reference conditioning）
    const referenceAudioUrls = isDialogue && provider === 'minimax' ? getStoryboardReferenceAudioUrls(sb.id) : []

    const prompt = sb.videoPrompt || sb.imagePrompt || sb.description || '镜头缓慢推进，人物自然表演'
    await generateVideo({
      storyboardId: sb.id,
      dramaId,
      prompt,
      negativePrompt: VIDEO_NEGATIVE,
      referenceMode: referenceImages.length ? 'multiple' : 'single',
      imageUrl: sb.firstFrameImage,
      referenceImageUrls: referenceImages.length ? referenceImages : undefined,
      sceneType: sceneType || undefined,
      referenceAudioUrls: referenceAudioUrls.length ? referenceAudioUrls : undefined,
      configId,
    })
    submitted++
  }
  return submitted
}

/** 轮询等待所有分镜首帧图就绪；超时返回 false（部分完成不阻断） */
async function waitForImages(episodeId: number, dramaId: number, timeoutMs = MEDIA_WAIT_TIMEOUT_MS): Promise<boolean> {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    const sbs = getStoryboards(episodeId)
    const ready = sbs.filter((sb) => !!sb.firstFrameImage).length
    if (ready >= sbs.length) return true
    publishPipelineEvent(dramaId, { type: 'media-progress', episodeId, ready, total: sbs.length })
    await sleep(MEDIA_POLL_INTERVAL_MS)
  }
  return false
}

/** 轮询等待所有分镜视频就绪；超时返回 false */
async function waitForVideos(episodeId: number, dramaId: number, timeoutMs = MEDIA_WAIT_TIMEOUT_MS): Promise<boolean> {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    const sbs = getStoryboards(episodeId)
    const ready = sbs.filter((sb) => !!sb.videoUrl).length
    if (ready >= sbs.length) return true
    publishPipelineEvent(dramaId, { type: 'media-progress', episodeId, ready, total: sbs.length })
    await sleep(MEDIA_POLL_INTERVAL_MS)
  }
  return false
}

async function runImageStage(episodeId: number, dramaId: number, opts: AutoPipelineOptions): Promise<void> {
  const configId = getEpisode(episodeId)?.imageConfigId ?? opts.imageConfigId
  const submitted = await submitMissingImages(episodeId, dramaId, configId)
  logTaskProgress('AutoPipeline', 'image-stage', { episodeId, submitted })
  const ok = await waitForImages(episodeId, dramaId)
  if (!ok) {
    const pending = getStoryboards(episodeId).filter((sb) => !sb.firstFrameImage).length
    logTaskWarn('AutoPipeline', 'image-timeout', { episodeId, pendingCount: pending })
  }
}

async function runVideoStage(episodeId: number, dramaId: number, opts: AutoPipelineOptions): Promise<void> {
  const configId = getEpisode(episodeId)?.videoConfigId ?? opts.videoConfigId
  const submitted = await submitMissingVideos(episodeId, dramaId, configId)
  logTaskProgress('AutoPipeline', 'video-stage', { episodeId, submitted })
  const ok = await waitForVideos(episodeId, dramaId)
  if (!ok) {
    const pending = getStoryboards(episodeId).filter((sb) => !sb.videoUrl).length
    logTaskWarn('AutoPipeline', 'video-timeout', { episodeId, pendingCount: pending })
  }
}

async function runComposeStage(episodeId: number): Promise<void> {
  const sbs = getStoryboards(episodeId)
  for (const sb of sbs) {
    if (sb.composedVideoUrl) continue
    if (!sb.videoUrl) continue
    await composeStoryboard(sb.id)
  }
}

async function runMergeStage(episodeId: number, dramaId: number): Promise<void> {
  const ep = getEpisode(episodeId)
  if (ep?.videoUrl) return // 已拼接
  const completed = db
    .select()
    .from(schema.videoMerges)
    .where(and(eq(schema.videoMerges.episodeId, episodeId), eq(schema.videoMerges.status, 'completed')))
    .all()
  if (completed.length > 0) return
  const all = getStoryboards(episodeId)
  const ready = all.filter((sb) => sb.composedVideoUrl)
  if (all.length === 0 || ready.length !== all.length) {
    logTaskWarn('AutoPipeline', 'merge-skip-not-ready', { episodeId, ready: ready.length, total: all.length })
    return
  }
  await mergeEpisodeVideos(episodeId, dramaId)
}

// ============================================================
// 单集管线（幂等）
// ============================================================

interface StageDef {
  key: string
  status: string
  next: string
  run: () => Promise<void>
}

function buildStages(episodeId: number, dramaId: number, opts: AutoPipelineOptions): StageDef[] {
  const { needImage, needVideo, needCompose, needMerge } = mediaNeeds(opts)
  const done = AUTO_STATUS.done
  const stages: StageDef[] = [
    { key: 'script', status: AUTO_STATUS.scripting, next: AUTO_STATUS.extracting, run: () => runScriptStage(episodeId, dramaId) },
    { key: 'extract', status: AUTO_STATUS.extracting, next: AUTO_STATUS.voicing, run: () => runExtractStage(episodeId, dramaId) },
    { key: 'voice', status: AUTO_STATUS.voicing, next: AUTO_STATUS.storyboarding, run: () => runVoiceStage(episodeId, dramaId) },
    { key: 'storyboard', status: AUTO_STATUS.storyboarding, next: needImage ? AUTO_STATUS.imaging : done, run: () => runStoryboardStage(episodeId, dramaId) },
  ]
  if (needImage) {
    stages.push({ key: 'image', status: AUTO_STATUS.imaging, next: needVideo ? AUTO_STATUS.videoing : done, run: () => runImageStage(episodeId, dramaId, opts) })
  }
  if (needVideo) {
    stages.push({ key: 'video', status: AUTO_STATUS.videoing, next: needCompose ? AUTO_STATUS.composing : done, run: () => runVideoStage(episodeId, dramaId, opts) })
  }
  if (needCompose) {
    stages.push({ key: 'compose', status: AUTO_STATUS.composing, next: needMerge ? AUTO_STATUS.merging : done, run: () => runComposeStage(episodeId) })
  }
  if (needMerge) {
    stages.push({ key: 'merge', status: AUTO_STATUS.merging, next: done, run: () => runMergeStage(episodeId, dramaId) })
  }
  return stages
}

/** 同进程防重入锁：崩溃恢复与手动触发并发时，同一 episode 只执行一次（幂等，等现有执行完成即可） */
const runningEpisodes = new Set<number>()
/** follow-up 队列：执行期间再次触发时入队（而非静默丢弃），当前轮完成后自动重跑一轮以应用最新配置 */
const queuedEpisodes = new Set<number>()

/** 执行单集管线（幂等，可反复调用自动续跑）。忙时入队而非静默丢弃。 */
export async function executeEpisodePipeline(episodeId: number, dramaId: number, opts: AutoPipelineOptions): Promise<void> {
  if (runningEpisodes.has(episodeId)) {
    queuedEpisodes.add(episodeId)
    return
  }
  runningEpisodes.add(episodeId)
  try {
    while (true) {
      await executeEpisodePipelineInner(episodeId, dramaId, opts)
      if (!queuedEpisodes.delete(episodeId)) break
      logTaskProgress('AutoPipeline', 'follow-up-requeue', { episodeId })
    }
  } finally {
    runningEpisodes.delete(episodeId)
  }
}

async function executeEpisodePipelineInner(episodeId: number, dramaId: number, opts: AutoPipelineOptions): Promise<void> {
  if (isDramaDeleted(dramaId)) {
    logTaskWarn('AutoPipeline', 'drama-deleted-skip', { episodeId, dramaId })
    return
  }
  const ep = getEpisode(episodeId)
  if (!ep) throw new Error(`Episode ${episodeId} not found`)
  if (ep.status === AUTO_STATUS.done) return

  const stages = buildStages(episodeId, dramaId, opts)
  let currentStatus = ep.status

  logTaskStart('AutoPipeline', 'episode', { episodeId, dramaId, fromStatus: currentStatus })

  for (const stage of stages) {
    if (isStageCompleted(currentStatus, stage.next)) continue
    setStatus(episodeId, stage.status)
    currentStatus = stage.status
    logTaskProgress('AutoPipeline', `stage-${stage.key}`, { episodeId })
    try {
      await stage.run()
      setStatus(episodeId, stage.next)
      currentStatus = stage.next
      logTaskProgress('AutoPipeline', `stage-${stage.key}-done`, { episodeId })
    } catch (err) {
      setStatus(episodeId, AUTO_STATUS.failed)
      logTaskError('AutoPipeline', `stage-${stage.key}-failed`, { episodeId, error: (err as Error).message })
      throw err
    }
  }

  setStatus(episodeId, AUTO_STATUS.done)
  logTaskSuccess('AutoPipeline', 'episode-done', { episodeId, dramaId })
}

// ============================================================
// 整剧执行主体
// ============================================================

function loadPipelineOptions(dramaId: number): AutoPipelineOptions | null {
  const drama = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).get()
  if (!drama?.metadata) return null
  try {
    return JSON.parse(drama.metadata) as AutoPipelineOptions
  } catch {
    return null
  }
}

function savePipelineOptions(dramaId: number, opts: AutoPipelineOptions) {
  db.update(schema.dramas)
    .set({ metadata: JSON.stringify(opts), updatedAt: now() })
    .where(eq(schema.dramas.id, dramaId))
    .run()
}

/** 后台串行执行整剧各集（单集失败标记 auto:failed，不阻断后续集） */
export async function executePipeline(dramaId: number, opts: AutoPipelineOptions): Promise<void> {
  const episodes = db
    .select()
    .from(schema.episodes)
    .where(eq(schema.episodes.dramaId, dramaId))
    .orderBy(asc(schema.episodes.episodeNumber))
    .all()

  logTaskStart('AutoPipeline', 'drama', { dramaId, episodeCount: episodes.length })
  for (const ep of episodes) {
    try {
      await executeEpisodePipeline(ep.id, dramaId, opts)
    } catch (err) {
      logTaskError('AutoPipeline', 'episode-failed', { episodeId: ep.id, dramaId, error: (err as Error).message })
    }
  }
  logTaskSuccess('AutoPipeline', 'drama-done', { dramaId })
}

// ============================================================
// 对外入口
// ============================================================

/** 创建 Drama + Episodes 并触发后台全自动管线，立即返回 { dramaId, episodeIds } */
export function runAutoPipeline(opts: AutoPipelineOptions): { dramaId: number; episodeIds: number[] } {
  const o = normalizeOptions(opts)
  if (!o.premise?.trim()) throw new Error('premise 不能为空')

  const ts = now()

  // 1. 创建 Drama
  const dramaRes = db
    .insert(schema.dramas)
    .values({
      title: o.title!,
      description: o.premise.trim(),
      genre: o.genre!,
      style: o.style!,
      totalEpisodes: o.episodeCount!,
      status: 'auto_generating',
      metadata: JSON.stringify(o),
      createdAt: ts,
      updatedAt: ts,
    })
    .run()
  const dramaId = Number(dramaRes.lastInsertRowid)

  // 2. 创建 Episodes
  const episodeIds: number[] = []
  for (let i = 0; i < o.episodeCount!; i++) {
    const n = i + 1
    const content = o.episodeCount! > 1 ? `【第${n}集】${o.premise.trim()}` : o.premise.trim()
    const epRes = db
      .insert(schema.episodes)
      .values({
        dramaId,
        episodeNumber: n,
        title: `${o.title} 第${n}集`,
        content,
        status: AUTO_STATUS.queued,
        imageConfigId: o.imageConfigId,
        videoConfigId: o.videoConfigId,
        audioConfigId: o.audioConfigId,
        createdAt: ts,
        updatedAt: ts,
      })
      .run()
    episodeIds.push(Number(epRes.lastInsertRowid))
  }

  // 3. fire-and-forget 后台执行
  executePipeline(dramaId, o)
    .then(() => logTaskSuccess('AutoPipeline', 'run-complete', { dramaId }))
    .catch((err) => logTaskError('AutoPipeline', 'run-failed', { dramaId, error: (err as Error).message }))

  return { dramaId, episodeIds }
}

/** 续跑（幂等）：用于崩溃恢复、手动重跑、媒体补跑 */
export function resumeAutoPipeline(dramaId: number, override?: Partial<AutoPipelineOptions>): void {
  const base = loadPipelineOptions(dramaId)
  if (!base) throw new Error(`Drama ${dramaId} 缺少 pipeline 配置（metadata）`)
  const opts = normalizeOptions({ ...base, ...override })
  savePipelineOptions(dramaId, opts)

  const episodes = db.select().from(schema.episodes).where(eq(schema.episodes.dramaId, dramaId)).all()
  for (const ep of episodes) {
    if (ep.status === AUTO_STATUS.done) {
      const missing = mediaMissingStatus(ep.id, opts)
      if (missing) setStatus(ep.id, missing)
    } else if (ep.status === AUTO_STATUS.failed) {
      setStatus(ep.id, AUTO_STATUS.queued) // 从失败恢复：从头幂等重跑
    }
  }

  executePipeline(dramaId, opts)
    .then(() => logTaskSuccess('AutoPipeline', 'resume-complete', { dramaId }))
    .catch((err) => logTaskError('AutoPipeline', 'resume-failed', { dramaId, error: (err as Error).message }))
}

/** done 状态下，若媒体产物缺失，精确降级到缺失媒体阶段（避免重跑 Agent 链） */
function mediaMissingStatus(episodeId: number, opts: AutoPipelineOptions): string | null {
  const { needImage, needVideo, needCompose, needMerge } = mediaNeeds(opts)
  if (!needImage) return null
  const sbs = getStoryboards(episodeId)
  if (sbs.length === 0) return null
  if (needImage && sbs.some((sb) => !sb.firstFrameImage)) return AUTO_STATUS.imaging
  if (needVideo && sbs.some((sb) => !sb.videoUrl)) return AUTO_STATUS.videoing
  if (needCompose && sbs.some((sb) => !sb.composedVideoUrl)) return AUTO_STATUS.composing
  if (needMerge) {
    const completed = db
      .select()
      .from(schema.videoMerges)
      .where(and(eq(schema.videoMerges.episodeId, episodeId), eq(schema.videoMerges.status, 'completed')))
      .all()
    if (completed.length === 0) return AUTO_STATUS.merging
  }
  return null
}

/** 查询整剧管线进度 */
export function getAutoPipelineStatus(dramaId: number) {
  const drama = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).get()
  if (!drama) return null
  const episodes = db
    .select()
    .from(schema.episodes)
    .where(eq(schema.episodes.dramaId, dramaId))
    .orderBy(asc(schema.episodes.episodeNumber))
    .all()

  const episodesDetail = episodes.map((ep) => {
    const sbs = getStoryboards(ep.id)
    const characters = db
      .select()
      .from(schema.episodeCharacters)
      .where(eq(schema.episodeCharacters.episodeId, ep.id))
      .all()
    const scenes = db
      .select()
      .from(schema.episodeScenes)
      .where(eq(schema.episodeScenes.episodeId, ep.id))
      .all()
    return {
      id: ep.id,
      episodeNumber: ep.episodeNumber,
      status: ep.status,
      hasScript: !!ep.scriptContent?.trim(),
      hasVideo: !!ep.videoUrl,
      storyboardCount: sbs.length,
      characterCount: characters.length,
      sceneCount: scenes.length,
      imageReadyCount: sbs.filter((sb) => !!sb.firstFrameImage).length,
      videoReadyCount: sbs.filter((sb) => !!sb.videoUrl).length,
      composedCount: sbs.filter((sb) => !!sb.composedVideoUrl).length,
    }
  })

  const doneCount = episodesDetail.filter((e) => e.status === AUTO_STATUS.done).length
  const failedCount = episodesDetail.filter((e) => e.status === AUTO_STATUS.failed).length

  return {
    dramaId,
    title: drama.title,
    status: drama.status,
    totalEpisodes: episodes.length,
    doneCount,
    failedCount,
    running: doneCount + failedCount < episodes.length,
    episodes: episodesDetail,
  }
}

/** 启动崩溃恢复：扫描中间态 episode 自动续跑（幂等） */
export function recoverAutoPipelineOnStartup(): void {
  const rows = db
    .select()
    .from(schema.episodes)
    .where(inArray(schema.episodes.status, IN_FLIGHT_STATUSES))
    .all()
  if (rows.length === 0) return

  // 按 drama 分组，每个 drama 只触发一次 executePipeline（内部逐集幂等）
  const dramaIds = [...new Set(rows.map((r) => r.dramaId))]
  logTaskStart('AutoPipeline', 'recover', { episodeCount: rows.length, dramaCount: dramaIds.length })

  for (const dramaId of dramaIds) {
    const opts = loadPipelineOptions(dramaId)
    if (!opts) {
      logTaskWarn('AutoPipeline', 'recover-no-config', { dramaId })
      continue
    }
    executePipeline(dramaId, opts)
      .then(() => logTaskSuccess('AutoPipeline', 'recover-done', { dramaId }))
      .catch((err) => logTaskError('AutoPipeline', 'recover-failed', { dramaId, error: (err as Error).message }))
  }
}
