import { db, schema } from '../db/index.js'
import { eq } from 'drizzle-orm'
import { getActiveConfig, getConfigById } from './ai.js'
import { now } from '../utils/response.js'
import { downloadFile, readImageAsCompressedDataUrl } from '../utils/storage.js'
import { getVideoAdapter } from './adapters/registry'
import type { AIConfig } from './adapters/types'
import { logTaskError, logTaskPayload, logTaskProgress, logTaskStart, logTaskSuccess, logTaskWarn, redactUrl } from '../utils/task-logger.js'
import { gpuManager, type GpuLease, isLocalConfig } from './gpu-manager.js'

/** 后台任务 GPU 租约映射（fire-and-forget 模式用 id 追踪租约） */
const videoGpuLeases = new Map<number, GpuLease>()

interface GenerateVideoParams {
  storyboardId?: number
  dramaId?: number
  prompt: string
  model?: string
  referenceMode?: string
  imageUrl?: string
  firstFrameUrl?: string
  lastFrameUrl?: string
  referenceImageUrls?: string[]
  duration?: number
  aspectRatio?: string
  configId?: number
}

export async function generateVideo(params: GenerateVideoParams): Promise<number> {
  const ts = now()
  const config = params.configId
    ? getConfigById(params.configId)
    : getActiveConfig('video')
  if (!config) throw new Error('No active video AI config')

  const res = db.insert(schema.videoGenerations).values({
    storyboardId: params.storyboardId,
    dramaId: params.dramaId,
    prompt: params.prompt,
    model: params.model || config.model,
    provider: config.provider,
    referenceMode: params.referenceMode || 'none',
    imageUrl: params.imageUrl,
    firstFrameUrl: params.firstFrameUrl,
    lastFrameUrl: params.lastFrameUrl,
    referenceImageUrls: params.referenceImageUrls ? JSON.stringify(params.referenceImageUrls) : null,
    duration: params.duration || 5,
    aspectRatio: params.aspectRatio || '16:9',
    status: 'processing',
    createdAt: ts,
    updatedAt: ts,
  }).run()

  const lastId = Number(res.lastInsertRowid)
  logTaskStart('VideoTask', 'enqueue', {
    id: lastId,
    provider: config.provider,
    storyboardId: params.storyboardId,
    dramaId: params.dramaId,
    referenceMode: params.referenceMode || 'none',
    duration: params.duration || 5,
  })
  logTaskPayload('VideoTask', 'enqueue params', {
    id: lastId,
    config: {
      provider: config.provider,
      model: config.model,
      baseUrl: config.baseUrl,
    },
    params,
  })
  processVideoGeneration(lastId, config).catch(err => {
    logTaskError('VideoTask', 'process', { id: lastId, error: err.message })
    console.error(`Video generation ${lastId} failed:`, err)
  })
  return lastId
}

/**
 * 处理视频生成任务（含模型自动 fallback）
 *
 * 遍历 config.models 数组，任一模型成功即视为完成。
 * Vidu 等无轮询端点的提供商，fetch 成功后返回（依赖 Webhook），失败则 fallback。
 */
async function processVideoGeneration(id: number, config: AIConfig) {
  // ── 预加载记录和参考图 URL（所有模型尝试共享）──
  const rows = db.select().from(schema.videoGenerations).where(eq(schema.videoGenerations.id, id)).all()
  const record = rows[0]
  if (!record) return
  const resolvedImageUrl = await normalizeVideoReferenceUrl(record.imageUrl)
  const resolvedFirstFrameUrl = await normalizeVideoReferenceUrl(record.firstFrameUrl)
  const resolvedLastFrameUrl = await normalizeVideoReferenceUrl(record.lastFrameUrl)
  const resolvedReferenceImageUrls = await normalizeVideoReferenceUrls(record.referenceImageUrls)

  const models = config.models?.length ? config.models : [config.model]

  for (let attempt = 0; attempt < models.length; attempt++) {
    const model = models[attempt]
    const attemptConfig: AIConfig = { ...config, model }

    // ── 更新 DB 当前使用的模型 ──
    db.update(schema.videoGenerations)
      .set({ model, updatedAt: now() })
      .where(eq(schema.videoGenerations.id, id)).run()

    // ── 非首次尝试：释放上一次的 GPU 租约 ──
    if (attempt > 0) releaseVideoGpuLease(id)

    // ── 本地 GPU 模型：获取显存租约 ──
    if (isLocalConfig(config.baseUrl, config.provider)) {
      try {
        const lease = await gpuManager.acquire('video', config.provider, model, config.baseUrl)
        videoGpuLeases.set(id, lease)
      } catch (err: any) {
        logTaskWarn('VideoTask', 'gpu-acquire-failed', { id, model, error: err.message })
      }
    }

    try {
      logTaskProgress('VideoTask', 'build-request', {
        id, provider: config.provider, model, attempt: attempt + 1,
        storyboardId: record.storyboardId, referenceMode: record.referenceMode,
      })

      const adapter = getVideoAdapter(config.provider)
      const { url, method, headers, body } = adapter.buildGenerateRequest(attemptConfig, {
        id: record.id,
        model,
        prompt: record.prompt,
        referenceMode: record.referenceMode,
        imageUrl: resolvedImageUrl,
        firstFrameUrl: resolvedFirstFrameUrl,
        lastFrameUrl: resolvedLastFrameUrl,
        referenceImageUrls: resolvedReferenceImageUrls ? JSON.stringify(resolvedReferenceImageUrls) : null,
        duration: record.duration,
        aspectRatio: record.aspectRatio,
      })

      logTaskProgress('VideoTask', 'request', {
        id, provider: config.provider, method, url: redactUrl(url), model, referenceMode: record.referenceMode,
      })
      logTaskPayload('VideoTask', 'request payload', { id, method, url, headers, body })

      const resp = await fetch(url, { method, headers, body: JSON.stringify(body) })

      if (!resp.ok) throw new Error(`API error ${resp.status}: ${await resp.text()}`)
      const result = await resp.json() as any

      const { isAsync, taskId, videoUrl } = adapter.parseGenerateResponse(result)

      if (!isAsync && videoUrl) {
        logTaskProgress('VideoTask', 'sync-complete', { id, model, videoUrl })
        await handleVideoComplete(id, videoUrl, record.duration)
        return
      }

      // 异步模式：更新 taskId
      db.update(schema.videoGenerations)
        .set({ taskId, status: 'processing', updatedAt: now() })
        .where(eq(schema.videoGenerations.id, id)).run()
      logTaskProgress('VideoTask', 'poll-start', { id, taskId, provider: config.provider, model })

      // Vidu 没有轮询端点，成功提交后返回（依赖 Webhook 回调完成）
      if (adapter.provider === 'vidu') {
        logTaskProgress('VideoTask', 'webhook-wait', { id, taskId, provider: adapter.provider })
        return
      }

      await pollVideoTask(id, attemptConfig, taskId!, record.storyboardId)
      return
    } catch (err: any) {
      const isLastAttempt = attempt === models.length - 1
      logTaskWarn('VideoTask', isLastAttempt ? 'all-models-failed' : 'model-fallback', {
        id, attempt: attempt + 1, totalModels: models.length,
        failedModel: model, error: err.message,
        ...(isLastAttempt ? {} : { nextModel: models[attempt + 1] }),
      })

      if (isLastAttempt) {
        releaseVideoGpuLease(id)
        logTaskError('VideoTask', 'process', { id, provider: config.provider, attemptedModels: models, error: err.message })
        db.update(schema.videoGenerations)
          .set({ status: 'failed', errorMsg: `All models failed. Last error: ${err.message}`, updatedAt: now() })
          .where(eq(schema.videoGenerations.id, id)).run()
      }
    }
  }
}

async function normalizeVideoReferenceUrl(value: string | null | undefined): Promise<string | null> {
  const raw = String(value || '').trim()
  if (!raw) return null
  if (raw.startsWith('data:image/')) return raw
  if (raw.startsWith('static/') || raw.startsWith('/static/')) {
    const localPath = raw.startsWith('/static/') ? raw.slice(1) : raw
    try {
      return await readImageAsCompressedDataUrl(localPath, {
        maxWidth: 768,
        maxHeight: 768,
        quality: 68,
      })
    } catch (err) {
      logTaskWarn('VideoTask', 'reference-read-failed', { path: localPath, error: (err as Error).message })
      return null
    }
  }
  return raw
}

async function normalizeVideoReferenceUrls(raw: string | null | undefined): Promise<string[]> {
  if (!raw) return []
  let refs: string[] = []
  try {
    refs = JSON.parse(raw)
  } catch {
    refs = []
  }
  const normalized = await Promise.all(
    Array.from(new Set(refs.map((item) => String(item || '').trim()).filter(Boolean))).map((item) => normalizeVideoReferenceUrl(item)),
  )
  return normalized.filter((item): item is string => !!item)
}

/**
 * 轮询异步视频任务直到完成或失败。
 * 成功时通过 handleVideoComplete 保存视频并返回。
 * 最终失败时 **抛出错误**，由外层 retry 循环捕获后切换到下一个模型。
 */
async function pollVideoTask(id: number, config: AIConfig, taskId: string, storyboardId?: number | null): Promise<void> {
  const adapter = getVideoAdapter(config.provider)

  for (let i = 0; i < 300; i++) {
    await new Promise(r => setTimeout(r, 10000))
    try {
      const { url, method, headers } = adapter.buildPollRequest(config, taskId)
      logTaskProgress('VideoTask', 'poll-request', {
        id, taskId, provider: config.provider, method, url: redactUrl(url), attempt: i + 1,
      })
      const resp = await fetch(url, { method, headers })
      if (!resp.ok) continue
      const result = await resp.json() as any

      const pollResp = adapter.parsePollResponse(result)

      if (pollResp.status === 'completed' && pollResp.videoUrl) {
        logTaskSuccess('VideoTask', 'poll-complete', { id, taskId, videoUrl: pollResp.videoUrl })
        await handleVideoComplete(id, pollResp.videoUrl, null, storyboardId)
        return
      }
      if (pollResp.status === 'failed') {
        logTaskError('VideoTask', 'poll-failed', { id, taskId, error: pollResp.error || 'Video generation failed' })
        throw new Error(pollResp.error || 'Video generation failed')
      }
    } catch (err: any) {
      if (i === 299) {
        throw err // 重新抛出给外层 retry 循环
      }
      logTaskWarn('VideoTask', 'poll-retry', { id, taskId, attempt: i + 1, error: err.message })
    }
  }
}

/** 释放视频任务的 GPU 租约 */
function releaseVideoGpuLease(id: number): void {
  const lease = videoGpuLeases.get(id)
  if (lease) {
    lease.release()
    videoGpuLeases.delete(id)
  }
}

async function handleVideoComplete(id: number, videoUrl: string, duration: number | null | undefined, storyboardId?: number | null) {
  releaseVideoGpuLease(id)
  const localPath = await downloadFile(videoUrl, 'videos')
  db.update(schema.videoGenerations)
    .set({ videoUrl, localPath, status: 'completed', completedAt: now(), updatedAt: now() })
    .where(eq(schema.videoGenerations.id, id))
    .run()
  logTaskSuccess('VideoTask', 'downloaded', { id, localPath, storyboardId, duration })

  if (storyboardId) {
    db.update(schema.storyboards)
      .set({ videoUrl: localPath, duration: duration || undefined, updatedAt: now() })
      .where(eq(schema.storyboards.id, storyboardId))
      .run()
  }
}
