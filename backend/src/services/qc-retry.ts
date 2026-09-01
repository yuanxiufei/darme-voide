/**
 * 审片重跑闭环（QC Retry）
 *
 * 当镜头 QC 审片未通过（failed）时，只重写该失败分镜的 prompt，不重建整集：
 *   1. 组装 QC issues → 交给 storyboard_breaker 仅调用 update_storyboard 修复该镜
 *   2. 软删该镜旧图片/视频生成记录（避免 QC 复用旧产物）
 *   3. 重新提交首帧图 → 轮询等新首帧就绪 → 重新提交视频（fire-and-forget）
 *   4. 视频完成时 webhook 自动触发 QC 重新打分，形成闭环
 */
import { and, eq, isNull } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { now } from '../utils/response.js'
import { runAgentWithRetry } from '../agents/index.js'
import { generateImage } from './image-generation.js'
import { generateVideo } from './video-generation.js'
import { getActiveConfig, getConfigById } from './ai.js'
import { logTaskProgress, logTaskSuccess, logTaskWarn } from '../utils/task-logger.js'
import {
  STORYBOARD_IMAGE_NEGATIVE,
  VIDEO_NEGATIVE,
  getStoryboardReferenceImages,
  getStoryboardReferenceAudioUrls,
} from '../shared/prompt-utils.js'

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

/** 支持多参考图的视频提供商；其余仅首尾帧 */
const MULTI_REFERENCE_PROVIDERS = new Set(['volcengine', 'vidu', 'minimax'])

const FIRST_FRAME_WAIT_TIMEOUT_MS = 5 * 60 * 1000
const FIRST_FRAME_POLL_INTERVAL_MS = 3000

export interface QcRetryResult {
  ok: boolean
  message: string
  storyboardId: number
  storyboardNumber?: number
  rewrittenFields?: string[]
  imageGenerationId?: number
  videoGenerationId?: number
}

export async function retryFailedStoryboard(storyboardId: number): Promise<QcRetryResult> {
  const [sb] = db
    .select()
    .from(schema.storyboards)
    .where(and(eq(schema.storyboards.id, storyboardId), isNull(schema.storyboards.deletedAt)))
    .all()
  if (!sb) return { ok: false, message: 'Storyboard not found', storyboardId }
  const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, sb.episodeId)).all()
  if (!ep) return { ok: false, message: 'Episode not found', storyboardId }

  const qcs = db
    .select()
    .from(schema.videoQualityChecks)
    .where(eq(schema.videoQualityChecks.storyboardId, storyboardId))
    .all()
    .sort((a, b) => b.id - a.id)
  const qc = qcs[0]
  const issues: Array<{ severity: string; dimension: string; message: string }> = qc?.issues
    ? (() => {
        try { return JSON.parse(qc.issues) } catch { return [] }
      })()
    : []

  // ===== 1. LLM 只重写该失败分镜 =====
  const issuesText = issues.length
    ? issues.map((i) => `[${i.severity}] ${i.dimension}: ${i.message}`).join('\n')
    : '（无具体问题描述，请复核画面与对白质量后自行修复明显缺陷）'
  const message = `本镜 QC 审片未通过，需要修复。分镜 ID=${storyboardId}（镜头 #${sb.storyboardNumber}「${sb.title || ''}」）。\n审片问题：\n${issuesText}\n\n修复要求：\n1. 只调用 update_storyboard 工具更新该分镜（storyboard_id=${storyboardId}），可修改 first_frame_prompt / last_frame_prompt / video_prompt / image_prompt / negative_prompt / constraints / start_state / end_state / action / description 等字段\n2. 禁止调用 save_storyboards 重建整集，禁止修改其他分镜\n3. 针对性修复：画面重复则改构图、首尾帧雷同则强化差异、状态跳变则修正 start_state/end_state、负面元素则强化 negative_prompt\n4. 修复完成后报告修改了哪些字段`

  logTaskProgress('QcRetry', 'rewrite-begin', { storyboardId, episodeId: sb.episodeId, issueCount: issues.length })
  const agentRet = await runAgentWithRetry('storyboard_breaker', sb.episodeId, ep.dramaId, message, { maxSteps: 20 })

  const [updated] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, storyboardId)).all()
  if (!updated) return { ok: false, message: 'Storyboard disappeared during retry', storyboardId }

  // 报告实际被修改的字段
  const rewrittenFields = (['firstFramePrompt', 'lastFramePrompt', 'videoPrompt', 'imagePrompt', 'negativePrompt', 'constraints', 'startState', 'endState'] as const)
    .filter((k) => (updated as any)[k] !== (sb as any)[k])
    .map((k) => k)

  // ===== 2. 软删旧图片/视频生成记录（防止 QC 复用旧产物）=====
  const softDeleteTs = now()
  db.update(schema.imageGenerations)
    .set({ deletedAt: softDeleteTs, updatedAt: softDeleteTs })
    .where(eq(schema.imageGenerations.storyboardId, storyboardId))
    .run()
  db.update(schema.videoGenerations)
    .set({ deletedAt: softDeleteTs, updatedAt: softDeleteTs })
    .where(eq(schema.videoGenerations.storyboardId, storyboardId))
    .run()

  // ===== 3. 重新提交首帧图并等待就绪 =====
  const oldFirstFrame = updated.firstFrameImage
  const prompt = updated.firstFramePrompt || updated.imagePrompt || updated.description || '短剧分镜画面，电影感构图'
  const refImages = getStoryboardReferenceImages(storyboardId)
  const imageId = await generateImage({
    storyboardId,
    dramaId: ep.dramaId,
    prompt,
    negativePrompt: updated.negativePrompt || STORYBOARD_IMAGE_NEGATIVE,
    frameType: 'first_frame',
    referenceImages: refImages.length ? refImages : undefined,
    configId: ep.imageConfigId ?? undefined,
  })

  const newFirstFrameReady = await (async () => {
    const start = Date.now()
    while (Date.now() - start < FIRST_FRAME_WAIT_TIMEOUT_MS) {
      const [row] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, storyboardId)).all()
      if (row?.firstFrameImage && row.firstFrameImage !== oldFirstFrame) return true
      await sleep(FIRST_FRAME_POLL_INTERVAL_MS)
    }
    return false
  })()

  if (!newFirstFrameReady) {
    logTaskWarn('QcRetry', 'first-frame-timeout', { storyboardId, imageGenerationId: imageId })
  }

  // ===== 4. 重新提交视频（fire-and-forget，完成后 webhook 自动重新 QC）=====
  const [finalSb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, storyboardId)).all()
  const sceneType = finalSb?.sceneType || ''
  const isDialogue = /dialogue|meeting|argument|conversation|multi/i.test(sceneType)
  const config = ep.videoConfigId ? getConfigById(ep.videoConfigId) : getActiveConfig('video')
  const provider = (config?.provider || '').toLowerCase()
  const canMultiRef = MULTI_REFERENCE_PROVIDERS.has(provider)
  const referenceImages = isDialogue && canMultiRef ? getStoryboardReferenceImages(storyboardId).slice(0, 9) : []
  const referenceAudioUrls = isDialogue && provider === 'minimax' ? getStoryboardReferenceAudioUrls(storyboardId) : []

  const videoPrompt = finalSb?.videoPrompt || finalSb?.imagePrompt || finalSb?.description || '镜头缓慢推进，人物自然表演'
  const videoId = await generateVideo({
    storyboardId,
    dramaId: ep.dramaId,
    prompt: videoPrompt,
    negativePrompt: VIDEO_NEGATIVE,
    referenceMode: referenceImages.length ? 'multiple' : 'single',
    imageUrl: finalSb?.firstFrameImage || undefined,
    referenceImageUrls: referenceImages.length ? referenceImages : undefined,
    sceneType: sceneType || undefined,
    referenceAudioUrls: referenceAudioUrls.length ? referenceAudioUrls : undefined,
    configId: ep.videoConfigId ?? undefined,
  })

  logTaskSuccess('QcRetry', 'retry-complete', {
    storyboardId,
    rewrittenFields: rewrittenFields.join(','),
    imageGenerationId: imageId,
    videoGenerationId: videoId,
    agentToolCalls: agentRet?.toolCalls.length ?? 0,
  })

  const fieldsText = rewrittenFields.length ? rewrittenFields.join('、') : '无'

  return {
    ok: true,
    message: `已重写并重新生成镜头 #${sb.storyboardNumber}（修改字段：${fieldsText}），新首帧图任务 ${imageId}、新视频任务 ${videoId} 已提交，完成后将自动重新审片`,
    storyboardId,
    storyboardNumber: sb.storyboardNumber,
    rewrittenFields,
    imageGenerationId: imageId,
    videoGenerationId: videoId,
  }
}
