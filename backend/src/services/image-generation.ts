import { db, schema } from '../db/index.js'
import { and, eq, isNull } from 'drizzle-orm'
import { getActiveConfig, getActiveConfigByProvider, getConfigById } from './ai.js'
import { now } from '../utils/response.js'
import { downloadFile, readImageAsCompressedDataUrl, saveBase64Image } from '../utils/storage.js'
import { applyColorGradeToFile, hasColorGrade, type ColorGradeParams } from './color-grade.js'
import { getImageAdapter, imageAdapters } from './adapters/registry'
import type { AIConfig } from './adapters/types'
import { logTaskError, logTaskPayload, logTaskProgress, logTaskStart, logTaskSuccess, logTaskWarn, redactUrl } from '../utils/task-logger.js'
import { fetchWithRetry, formatVendorHttpError, formatVendorTaskError, isNonRetryableHttpError } from '../utils/vendor-errors.js'
import { gpuManager, type GpuLease, isLocalConfig } from './gpu-manager.js'

/** 后台任务 GPU 租约映射（fire-and-forget 模式用 id 追踪租约） */
const imageGpuLeases = new Map<number, GpuLease>()

interface GenerateImageParams {
  storyboardId?: number
  dramaId?: number
  sceneId?: number
  characterId?: number
  prompt: string
  /** 负面提示词；不传则继承当前图片配置的默认值 */
  negativePrompt?: string
  model?: string
  size?: string
  referenceImages?: string[]
  frameType?: string
  configId?: number
  /** 角色服装变体名；存在时生成结果写入 variations 对应项而非主图 */
  costume?: string
  /** 校色参数；存在时生成完成后用 sharp 后处理 */
  colorGrade?: ColorGradeParams
  /** 三视图类型：front/side/back；存在时结果写入 characters.threeViews 对应项而非主图 */
  viewType?: string
  /** 装备特写类型：clothing/weapon/accessory；存在时结果写入 characters.equipImages 对应项而非主图 */
  equipType?: string
}

export async function generateImage(params: GenerateImageParams): Promise<number> {
  const ts = now()
  const config = params.configId
    ? getConfigById(params.configId)
    : getActiveConfig('image')
  if (!config) throw new Error('No active image AI config')

  const res = db.insert(schema.imageGenerations).values({
    storyboardId: params.storyboardId,
    dramaId: params.dramaId,
    sceneId: params.sceneId,
    characterId: params.characterId,
    prompt: params.prompt,
    negativePrompt: params.negativePrompt ?? config.negativePrompt ?? null,
    model: params.model || config.model,
    provider: config.provider,
    size: params.size || '1920x1080',
    frameType: params.frameType,
    referenceImages: params.referenceImages ? JSON.stringify(params.referenceImages) : null,
    costume: params.costume ?? null,
    colorGrade: params.colorGrade && hasColorGrade(params.colorGrade) ? JSON.stringify(params.colorGrade) : null,
    viewType: params.viewType ?? null,
    status: 'processing',
    createdAt: ts,
    updatedAt: ts,
  }).run()

  const lastId = Number(res.lastInsertRowid)
  logTaskStart('ImageTask', 'enqueue', {
    id: lastId,
    provider: config.provider,
    storyboardId: params.storyboardId,
    sceneId: params.sceneId,
    characterId: params.characterId,
    frameType: params.frameType,
    model: params.model || config.model,
  })
  logTaskPayload('ImageTask', 'enqueue params', {
    id: lastId,
    config: {
      provider: config.provider,
      model: config.model,
      baseUrl: config.baseUrl,
    },
    params,
  })
  processImageGeneration(lastId, config).catch(err => {
    logTaskError('ImageTask', 'process', { id: lastId, error: err.message })
    console.error(`Image generation ${lastId} failed:`, err)
  })
  return lastId
}

/**
 * 处理图片生成任务（含模型自动 fallback）
 *
 * 遍历 config.models 数组，任一模型成功即视为完成。
 * 当前模型失败时自动释放 GPU 租约，切换到下一个模型重试。
 */
async function processImageGeneration(id: number, config: AIConfig) {
  // ── 预加载记录和参考图（所有模型尝试共享）──
  const rows = db.select().from(schema.imageGenerations).where(eq(schema.imageGenerations.id, id)).all()
  const record = rows[0]
  if (!record) return
  const resolvedReferenceImages = await normalizeReferenceImages(record.referenceImages)

  const models = config.models?.length ? config.models : [config.model]

  for (let attempt = 0; attempt < models.length; attempt++) {
    const model = models[attempt]
    const attemptConfig: AIConfig = { ...config, model }

    // ── 更新 DB 当前使用的模型 ──
    db.update(schema.imageGenerations)
      .set({ model, updatedAt: now() })
      .where(eq(schema.imageGenerations.id, id)).run()

    // ── 非首次尝试：释放上一次的 GPU 租约 ──
    if (attempt > 0) releaseImageGpuLease(id)

    // ── 本地 GPU 模型：获取显存租约 ──
    if (isLocalConfig(config.baseUrl, config.provider)) {
      try {
        const lease = await gpuManager.acquire('image', config.provider, model, config.baseUrl)
        imageGpuLeases.set(id, lease)
      } catch (err: any) {
        logTaskWarn('ImageTask', 'gpu-acquire-failed', { id, model, error: err.message })
      }
    }

    try {
      logTaskProgress('ImageTask', 'build-request', {
        id, provider: config.provider, model, attempt: attempt + 1,
        storyboardId: record.storyboardId, sceneId: record.sceneId,
        characterId: record.characterId, frameType: record.frameType,
      })

      const adapter = getImageAdapter(config.provider)
      const { url, method, headers, body } = adapter.buildGenerateRequest(attemptConfig, {
        id: record.id,
        model,
        prompt: record.prompt,
        negativePrompt: record.negativePrompt,
        size: record.size,
        frameType: record.frameType,
        referenceImages: resolvedReferenceImages ? JSON.stringify(resolvedReferenceImages) : null,
      })

      logTaskProgress('ImageTask', 'request', {
        id, provider: config.provider, method, url: redactUrl(url), model,
      })
      logTaskPayload('ImageTask', 'request payload', { id, method, url, headers, body })

      const resp = await fetchWithRetry(url, {
        method, headers,
        body: JSON.stringify(body),
      }, 'image', {
        timeoutMs: 600_000,
        onRetry: (attempt, delayMs, reason) => logTaskWarn('ImageTask', 'request-retry', { id, model, attempt, delayMs, reason }),
      })
      const result = await resp.json() as any
      logTaskPayload('ImageTask', 'response payload', { id, provider: config.provider, result })

      const { isAsync, taskId, imageUrl } = adapter.parseGenerateResponse(result)

      if (!isAsync && imageUrl) {
        logTaskProgress('ImageTask', 'sync-complete', { id, model, imageUrl })
        await handleImageComplete(id, config.provider, imageUrl)
        return
      }

      if (!isAsync && !imageUrl) {
        const b64 = adapter.extractImageBase64(result)
        if (b64) {
          logTaskProgress('ImageTask', 'sync-base64-complete', { id, model, mimeType: b64.mimeType })
          await handleImageCompleteBase64(id, config.provider, b64.data, b64.mimeType)
          return
        }
        throw new Error('No image URL or base64 data in response')
      }

      // 异步模式：更新 taskId，阻塞等待轮询完成
      db.update(schema.imageGenerations)
        .set({ taskId, status: 'processing', updatedAt: now() })
        .where(eq(schema.imageGenerations.id, id)).run()
      logTaskProgress('ImageTask', 'poll-start', { id, taskId, provider: config.provider, model })

      await pollImageTask(id, attemptConfig, taskId!)
      return // 轮询成功完成
    } catch (err: any) {
      const isLastAttempt = attempt === models.length - 1
      logTaskWarn('ImageTask', isLastAttempt ? 'all-models-failed' : 'model-fallback', {
        id, attempt: attempt + 1, totalModels: models.length,
        failedModel: model, error: err.message,
        ...(isLastAttempt ? {} : { nextModel: models[attempt + 1] }),
      })

      if (isLastAttempt) {
        releaseImageGpuLease(id)
        logTaskError('ImageTask', 'process', { id, provider: config.provider, attemptedModels: models, error: err.message })
        db.update(schema.imageGenerations)
          .set({ status: 'failed', errorMsg: `All models failed. Last error: ${err.message}`, updatedAt: now() })
          .where(eq(schema.imageGenerations.id, id)).run()
      }
    }
  }
}

async function normalizeReferenceImages(raw: string | null | undefined): Promise<string[]> {
  if (!raw) return []
  let refs: string[] = []
  try {
    refs = JSON.parse(raw)
  } catch {
    refs = []
  }

  const deduped = Array.from(
    new Set(
      refs
        .map((item) => String(item || '').trim())
        .filter(Boolean),
    ),
  )

  const normalized = await Promise.all(deduped.map(async (value) => {
    if (value.startsWith('data:image/')) return value
    if (value.startsWith('static/') || value.startsWith('/static/')) {
      const localPath = value.startsWith('/static/') ? value.slice(1) : value
      try {
        return await readImageAsCompressedDataUrl(localPath, {
          maxWidth: 768,
          maxHeight: 768,
          quality: 68,
        })
      } catch (err) {
        logTaskWarn('ImageTask', 'reference-read-failed', { path: localPath, error: (err as Error).message })
        return null
      }
    }
    return value
  }))

  return normalized.filter((item): item is string => !!item).slice(0, 6)
}

/**
 * 轮询异步图片任务直到完成或失败。
 * 成功时通过 handleImageComplete 保存图片并返回。
 * 最终失败时 **抛出错误**，由外层 retry 循环捕获后切换到下一个模型。
 */
async function pollImageTask(id: number, config: AIConfig, taskId: string): Promise<void> {
  const adapter = getImageAdapter(config.provider)
  const startedAt = Date.now()
  const maxDurationMs = 600_000

  for (let i = 0; i < 120; i++) {
    if (Date.now() - startedAt >= maxDurationMs) {
      throw new Error('Polling exceeded 10 minutes')
    }
    await new Promise(r => setTimeout(r, 5000))
    if (Date.now() - startedAt >= maxDurationMs) {
      throw new Error('Polling exceeded 10 minutes')
    }
    try {
      const { url, method, headers } = adapter.buildPollRequest(config, taskId)
      logTaskProgress('ImageTask', 'poll-request', {
        id, taskId, provider: config.provider, method, url: redactUrl(url), attempt: i + 1,
      })
      const remainingMs = Math.max(1_000, maxDurationMs - (Date.now() - startedAt))
      const resp = await fetch(url, { method, headers, signal: AbortSignal.timeout(remainingMs) })
      if (!resp.ok) {
        const bodyText = await resp.text()
        if (isNonRetryableHttpError(resp.status, bodyText)) {
          throw new Error(formatVendorHttpError(resp.status, bodyText, 'image'))
        }
        continue
      }
      const result = await resp.json() as any

      const pollResp = adapter.parsePollResponse(result)

      if (pollResp.status === 'completed' && pollResp.imageUrl) {
        logTaskSuccess('ImageTask', 'poll-complete', { id, taskId, imageUrl: pollResp.imageUrl })
        await handleImageComplete(id, config.provider, pollResp.imageUrl)
        return
      }
      if (pollResp.status === 'completed' && adapter.provider === 'gemini') {
        const b64 = adapter.extractImageBase64(result)
        if (b64) {
          logTaskSuccess('ImageTask', 'poll-base64-complete', { id, taskId, mimeType: b64.mimeType })
          await handleImageCompleteBase64(id, config.provider, b64.data, b64.mimeType)
          return
        }
      }
      if (pollResp.status === 'failed') {
        const msg = formatVendorTaskError(pollResp.error, 'image')
        logTaskError('ImageTask', 'poll-failed', { id, taskId, error: msg })
        throw new Error(msg)
      }
    } catch (err: any) {
      if (i === 119 || Date.now() - startedAt >= maxDurationMs) {
        throw err // 重新抛出给外层 retry 循环
      }
      logTaskWarn('ImageTask', 'poll-retry', { id, taskId, attempt: i + 1, error: err.message })
    }
  }
}

/** 释放图片任务的 GPU 租约 */
function releaseImageGpuLease(id: number): void {
  const lease = imageGpuLeases.get(id)
  if (lease) {
    lease.release()
    imageGpuLeases.delete(id)
  }
}

/**
 * 更新角色图片：若本次生成指定了 costume（服装变体），写入 characters.variations
 * 对应项的 imageUrl；否则覆盖主图。用于支持「同一角色多套造型独立立绘」。
 */
function updateCharacterImage(record: any, localPath: string) {
  if (!record?.characterId) return
  const charRows = db.select().from(schema.characters)
    .where(and(eq(schema.characters.id, record.characterId), isNull(schema.characters.deletedAt)))
    .all()
  const char = charRows[0]
  if (!char) return

  // 三视图：写入 threeViews 对应项而非主图
  if (record.viewType === 'front' || record.viewType === 'side' || record.viewType === 'back') {
    let views: Record<string, { view: string; imageUrl: string; prompt?: string | null; generatedAt: string }> = {}
    try {
      views = JSON.parse(char.threeViews || '{}')
    } catch {
      views = {}
    }
    views[record.viewType] = { view: record.viewType, imageUrl: localPath, prompt: record.prompt, generatedAt: now() }
    db.update(schema.characters)
      .set({ threeViews: JSON.stringify(views), updatedAt: now() })
      .where(eq(schema.characters.id, record.characterId)).run()
    return
  }

  // 装备特写（服装/武器/首饰）：写入 equipImages 对应项而非主图
  if (record.equipType === 'clothing' || record.equipType === 'weapon' || record.equipType === 'accessory') {
    let imgs: Record<string, { type: string; imageUrl: string; prompt?: string | null; generatedAt: string }> = {}
    try {
      imgs = JSON.parse(char.equipImages || '{}')
    } catch {
      imgs = {}
    }
    imgs[record.equipType] = { type: record.equipType, imageUrl: localPath, prompt: record.prompt, generatedAt: now() }
    db.update(schema.characters)
      .set({ equipImages: JSON.stringify(imgs), updatedAt: now() })
      .where(eq(schema.characters.id, record.characterId)).run()
    return
  }

  if (record.costume) {
    let variations: Array<{ name: string; imageUrl: string | null }> = []
    try {
      variations = JSON.parse(char.variations || '[]')
    } catch {
      variations = []
    }
    const idx = variations.findIndex((v) => v.name === record.costume)
    if (idx >= 0) {
      variations[idx] = { ...variations[idx], imageUrl: localPath }
    } else {
      variations.push({ name: record.costume, imageUrl: localPath })
    }
    db.update(schema.characters)
      .set({ variations: JSON.stringify(variations), updatedAt: now() })
      .where(eq(schema.characters.id, record.characterId)).run()
  } else {
    db.update(schema.characters)
      .set({ imageUrl: localPath, updatedAt: now() })
      .where(eq(schema.characters.id, record.characterId)).run()
  }
}

async function handleImageComplete(id: number, provider: string, imageUrl: string) {
  releaseImageGpuLease(id)
  const localPath = await downloadFile(imageUrl, 'images')
  const rows = db.select().from(schema.imageGenerations).where(eq(schema.imageGenerations.id, id)).all()
  const record = rows[0]

  // 校色后处理（若记录带 colorGrade）
  let finalPath = localPath
  if (record?.colorGrade) {
    try {
      finalPath = await applyColorGradeToFile(localPath, record.colorGrade)
    } catch (err: any) {
      logTaskWarn('ImageTask', 'color-grade-failed', { id, error: err?.message || String(err) })
    }
  }

  db.update(schema.imageGenerations)
    .set({ imageUrl, localPath: finalPath, status: 'completed', updatedAt: now() })
    .where(eq(schema.imageGenerations.id, id))
    .run()
  logTaskSuccess('ImageTask', 'downloaded', { id, provider, localPath: finalPath })

  // 更新关联表
  if (record?.storyboardId) {
    const sbUpdate: Record<string, any> = { updatedAt: now() }
    if (record.frameType === 'first_frame') sbUpdate.firstFrameImage = finalPath
    else if (record.frameType === 'last_frame') sbUpdate.lastFrameImage = finalPath
    else sbUpdate.composedImage = finalPath
    db.update(schema.storyboards).set(sbUpdate).where(eq(schema.storyboards.id, record.storyboardId)).run()
  }
  if (record?.characterId) {
    updateCharacterImage(record, finalPath)
  }
  if (record?.sceneId) {
    db.update(schema.scenes).set({ imageUrl: finalPath, status: 'completed', updatedAt: now() }).where(eq(schema.scenes.id, record.sceneId)).run()
  }
}

async function handleImageCompleteBase64(id: number, provider: string, base64Data: string, mimeType: string) {
  releaseImageGpuLease(id)
  const localPath = await saveBase64Image(base64Data, mimeType, 'images')
  const rows = db.select().from(schema.imageGenerations).where(eq(schema.imageGenerations.id, id)).all()
  const record = rows[0]

  // 校色后处理（若记录带 colorGrade）
  let finalPath = localPath
  if (record?.colorGrade) {
    try {
      finalPath = await applyColorGradeToFile(localPath, record.colorGrade)
    } catch (err: any) {
      logTaskWarn('ImageTask', 'color-grade-failed', { id, error: err?.message || String(err) })
    }
  }

  db.update(schema.imageGenerations)
    .set({ localPath: finalPath, status: 'completed', updatedAt: now() })
    .where(eq(schema.imageGenerations.id, id))
    .run()
  logTaskSuccess('ImageTask', 'saved-base64', { id, provider, mimeType, localPath: finalPath })

  // 更新关联表
  if (record?.storyboardId) {
    const sbUpdate: Record<string, any> = { updatedAt: now() }
    if (record.frameType === 'first_frame') sbUpdate.firstFrameImage = finalPath
    else if (record.frameType === 'last_frame') sbUpdate.lastFrameImage = finalPath
    else sbUpdate.composedImage = finalPath
    db.update(schema.storyboards).set(sbUpdate).where(eq(schema.storyboards.id, record.storyboardId)).run()
  }
  if (record?.characterId) {
    updateCharacterImage(record, finalPath)
  }
  if (record?.sceneId) {
    db.update(schema.scenes).set({ imageUrl: finalPath, status: 'completed', updatedAt: now() }).where(eq(schema.scenes.id, record.sceneId)).run()
  }
}

/** 恢复超时阈值：processing 状态超过该时长（毫秒）无进展，判定为孤儿任务直接失败 */
const RECOVER_TIMEOUT_MS = 60 * 60 * 1000 // 60 分钟

/**
 * 服务启动时恢复被中断的图片生成任务（崩溃恢复）。
 *
 * 根因：generateImage 是 fire-and-forget，pollImageTask 用内存 setTimeout 轮询，
 * 进程一旦重启这些轮询循环直接蒸发，status='processing' 的记录会永久卡死。
 *
 * 恢复策略（幂等，绝不重新提交，避免按次计费的厂商重复扣费）：
 *  - 无 taskId：提交请求前进程就崩了（或同步模式下载中途崩）→ 判 failed
 *  - 未知 provider：显式判 failed（getImageAdapter 已改为 throw）
 *  - 其余轮询型：按记录 provider 匹配活跃配置，恢复 pollImageTask 续跑
 *  - 超时兜底：updatedAt 超过 RECOVER_TIMEOUT_MS 的 processing 直接判 failed
 */
export function recoverImageTasksOnStartup() {
  const orphans = db.select().from(schema.imageGenerations)
    .where(eq(schema.imageGenerations.status, 'processing'))
    .all()

  if (orphans.length === 0) return
  logTaskWarn('ImageTask', 'recover-start', { count: orphans.length })

  for (const r of orphans) {
    try {
      // 超时兜底：超过阈值直接失败（厂商任务大概率已过期，无法续跑）
      if (r.updatedAt && Date.now() - new Date(r.updatedAt).getTime() > RECOVER_TIMEOUT_MS) {
        markImageRecoverFailed(r.id, `Interrupted and expired (idle > ${RECOVER_TIMEOUT_MS / 60000}min)`)
        continue
      }

      const provider = (r.provider || '').toLowerCase()

      if (!r.taskId) {
        // 提交请求前进程崩溃（或同步模式下载中途崩），无 taskId 无法续跑
        markImageRecoverFailed(r.id, 'Interrupted before task submit')
        continue
      }

      // 未知 provider 显式失败（getImageAdapter 已改为 throw，不再静默 fallback）
      if (!imageAdapters[provider]) {
        markImageRecoverFailed(r.id, `Unknown image provider: ${r.provider || '(empty)'}`)
        continue
      }

      // 轮询型：按 provider 精确匹配活跃配置，恢复轮询
      const config = getActiveConfigByProvider('image', provider)
      if (!config) {
        // 无匹配配置：保持 processing，待配置补上后下次重启再恢复
        logTaskWarn('ImageTask', 'recover-no-config', { id: r.id, taskId: r.taskId, provider })
        continue
      }
      // 用记录里当初提交的模型覆盖，保证续跑同一模型
      if (r.model) config.model = r.model

      logTaskWarn('ImageTask', 'recover-resume-poll', { id: r.id, taskId: r.taskId, provider, model: r.model })
      pollImageTask(r.id, config, r.taskId).catch(err => {
        logTaskError('ImageTask', 'recover-poll-failed', { id: r.id, error: err.message })
        markImageRecoverFailed(r.id, `Recover poll failed: ${err.message}`)
      })
    } catch (err: any) {
      logTaskError('ImageTask', 'recover-error', { id: r.id, error: err.message })
    }
  }
}

/** 标记恢复失败（复用 processImageGeneration 的 failed 更新语义） */
function markImageRecoverFailed(id: number, reason: string) {
  db.update(schema.imageGenerations)
    .set({ status: 'failed', errorMsg: reason, updatedAt: now() })
    .where(eq(schema.imageGenerations.id, id))
    .run()
  logTaskError('ImageTask', 'recover-failed', { id, reason })
}
