import { db, schema } from '../db/index.js'
import { and, eq } from 'drizzle-orm'
import { getActiveConfig, getActiveConfigByProvider, getConfigById } from './ai.js'
import { now } from '../utils/response.js'
import { downloadFile, readImageAsCompressedDataUrl } from '../utils/storage.js'
import { probeVideoDuration } from '../utils/video-probe.js'
import { getVideoAdapter, videoAdapters } from './adapters/registry'
import type { AIConfig } from './adapters/types'
import { logTaskError, logTaskPayload, logTaskProgress, logTaskStart, logTaskSuccess, logTaskWarn, redactUrl } from '../utils/task-logger.js'
import { fetchWithRetry, formatVendorHttpError, formatVendorTaskError, isNonRetryableHttpError } from '../utils/vendor-errors.js'
import { gpuManager, type GpuLease, isLocalConfig } from './gpu-manager.js'
import { stripVideoPromptTags } from '../shared/prompt-utils.js'
import { runQcAfterVideoComplete } from './qc-scoring.js'
import { recordUsage, type UsageStatus } from './usage-tracking.js'
import { recordAssetVersion } from './asset-versions.js'

/** 后台任务 GPU 租约映射（fire-and-forget 模式用 id 追踪租约） */
const videoGpuLeases = new Map<number, GpuLease>()

interface GenerateVideoParams {
  storyboardId?: number
  dramaId?: number
  prompt: string
  negativePrompt?: string
  model?: string
  referenceMode?: string
  imageUrl?: string
  firstFrameUrl?: string
  lastFrameUrl?: string
  referenceImageUrls?: string[]
  sceneType?: string
  referenceAudioUrls?: string[]
  duration?: number
  aspectRatio?: string
  configId?: number
  /** 剧本内容指纹门禁：设为 true 跳过门禁（确认基于旧分镜继续） */
  force?: boolean
  /** 逐镜路由快照（对齐 H3-Codex-Drama shot routing）：本镜采用的生成路线与决策原因 */
  route?: string
  routeReason?: string
}

export async function generateVideo(params: GenerateVideoParams): Promise<number> {
  const ts = now()
  const config = params.configId
    ? getConfigById(params.configId)
    : getActiveConfig('video')
  if (!config) throw new Error('No active video AI config')

  // per-shot take 预算：分镜生成尝试超预算则阻断（force 可放行）
  if (params.storyboardId && !params.force) {
    const { checkTakeBudget } = await import('./take-budget.js')
    const budget = checkTakeBudget(params.storyboardId)
    if (!budget.allowed) {
      logTaskWarn('VideoTask', 'take-budget-exhausted', { storyboardId: params.storyboardId, reason: budget.reason })
      throw new Error(budget.reason)
    }
  }

  // 剧本内容指纹门禁：分镜媒体生成前校验剧本未变更，过期分镜阻断（force 可放行）
  if (params.storyboardId && !params.force) {
    const { checkStoryboardGate } = await import('./script-fingerprint.js')
    const gate = checkStoryboardGate(params.storyboardId)
    if (!gate.allowed) {
      logTaskWarn('VideoTask', 'gate-blocked', { storyboardId: params.storyboardId, reason: gate.reason })
      throw new Error(gate.reason)
    }
  }

  // 剥离分镜视频提示词中的结构化标签（<location>/<role>/<voice>/<n>），
  // 让视频生成模型拿到干净的纯自然语言 prompt（标签是给 agent/程序解析用的 DSL）
  let prompt = stripVideoPromptTags(params.prompt)

  // 连续性状态机 v3：逐镜禁止变化清单注入（该镜画面中必须保持不变的元素）
  if (params.storyboardId) {
    const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, params.storyboardId)).all()
    if (sb?.constraints) {
      prompt = `${prompt} -- 逐镜禁止变化（画面中以下元素必须始终保持不变，不得增减或改变）: ${sb.constraints}`
    }
    // H3 原生场景声标记 [background_audio]（对齐参考项目 minimax-h3-comfyui 语法）：
    // 分镜配置了 sound_effect 且 prompt 尚未包含该标记时注入，H3 会在成片时同步合成环境音。
    // 幂等：videos.ts 富化路径已注入过则不再重复追加。
    if (sb?.soundEffect?.trim() && !prompt.includes('[background_audio]')) {
      prompt = `${prompt} [background_audio] ${sb.soundEffect.trim()}`
    }
  }

  const res = db.insert(schema.videoGenerations).values({
    storyboardId: params.storyboardId,
    dramaId: params.dramaId,
    prompt,
    negativePrompt: params.negativePrompt ?? config.negativePrompt ?? null,
    model: params.model || config.model,
    provider: config.provider,
    referenceMode: params.referenceMode || 'none',
    imageUrl: params.imageUrl,
    firstFrameUrl: params.firstFrameUrl,
    lastFrameUrl: params.lastFrameUrl,
    referenceImageUrls: params.referenceImageUrls ? JSON.stringify(params.referenceImageUrls) : null,
    sceneType: params.sceneType || null,
    referenceAudioUrls: params.referenceAudioUrls?.length ? JSON.stringify(params.referenceAudioUrls) : null,
    route: params.route || null,
    routeReason: params.routeReason || null,
    duration: params.duration || 5,
    aspectRatio: params.aspectRatio || '16:9',
    status: 'processing',
    createdAt: ts,
    updatedAt: ts,
  }).run()

  const lastId = Number(res.lastInsertRowid)

  // per-shot take 预算：任务提交成功即消耗一次 take（无论成败，算一次尝试）
  if (params.storyboardId) {
    await import('./take-budget.js').then(m => m.consumeTake(params.storyboardId))
  }

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
    const isLocal = isLocalConfig(config.baseUrl, config.provider)
    if (isLocal) {
      try {
        const lease = await gpuManager.acquire('video', config.provider, model, config.baseUrl)
        videoGpuLeases.set(id, lease)
      } catch (err: any) {
        logTaskWarn('VideoTask', 'gpu-acquire-failed', { id, model, error: err.message })
      }
    }

    // ── 用量记账：每次模型尝试（含 fallback）记一条 submitted，完成/失败后收口 ──
    recordUsage({
      serviceType: 'video',
      provider: config.provider,
      model,
      dramaId: record.dramaId,
      storyboardId: record.storyboardId,
      videoGenerationId: id,
      units: record.duration ?? null,
      isLocal,
      status: 'submitted',
      retryCount: attempt,
      settings: config.settings,
    })

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
        negativePrompt: record.negativePrompt,
        referenceMode: record.referenceMode,
        imageUrl: resolvedImageUrl,
        firstFrameUrl: resolvedFirstFrameUrl,
        lastFrameUrl: resolvedLastFrameUrl,
        referenceImageUrls: resolvedReferenceImageUrls ? JSON.stringify(resolvedReferenceImageUrls) : null,
        sceneType: record.sceneType,
        referenceAudioUrls: record.referenceAudioUrls,
        duration: record.duration,
        aspectRatio: record.aspectRatio,
      })

      logTaskProgress('VideoTask', 'request', {
        id, provider: config.provider, method, url: redactUrl(url), model, referenceMode: record.referenceMode,
      })
      logTaskPayload('VideoTask', 'request payload', { id, method, url, headers, body })

      const resp = await fetchWithRetry(url, { method, headers, body: JSON.stringify(body) }, 'video', {
        onRetry: (attempt, delayMs, reason) => logTaskWarn('VideoTask', 'request-retry', { id, model, attempt, delayMs, reason }),
      })
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
        markUsageByVideoGen(id, 'failed')
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
      if (!resp.ok) {
        const bodyText = await resp.text()
        if (isNonRetryableHttpError(resp.status, bodyText)) {
          throw new Error(formatVendorHttpError(resp.status, bodyText, 'video'))
        }
        continue
      }
      const result = await resp.json() as any

      const pollResp = adapter.parsePollResponse(result)

      if (pollResp.status === 'completed' && pollResp.videoUrl) {
        logTaskSuccess('VideoTask', 'poll-complete', { id, taskId, videoUrl: pollResp.videoUrl })
        await handleVideoComplete(id, pollResp.videoUrl, null, storyboardId)
        return
      }
      if (pollResp.status === 'failed') {
        const msg = formatVendorTaskError(pollResp.error, 'video')
        logTaskError('VideoTask', 'poll-failed', { id, taskId, error: msg })
        throw new Error(msg)
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

/** 收口某个视频生成任务的所有 submitted 用量记录（完成/失败） */
function markUsageByVideoGen(videoGenerationId: number, status: UsageStatus) {
  db.update(schema.apiUsage)
    .set({ status })
    .where(and(eq(schema.apiUsage.videoGenerationId, videoGenerationId), eq(schema.apiUsage.status, 'submitted')))
    .run()
}

async function handleVideoComplete(id: number, videoUrl: string, duration: number | null | undefined, storyboardId?: number | null) {
  releaseVideoGpuLease(id)
  markUsageByVideoGen(id, 'completed')
  const localPath = await downloadFile(videoUrl, 'videos')

  // 异步提供商（轮询/Webhook）不返回时长时，用 ffprobe 探测本地文件实际时长
  let resolvedDuration = duration ?? null
  if (resolvedDuration == null) {
    const probed = await probeVideoDuration(localPath)
    if (probed > 0) resolvedDuration = probed
  }

  db.update(schema.videoGenerations)
    .set({ videoUrl, localPath, status: 'completed', completedAt: now(), updatedAt: now() })
    .where(eq(schema.videoGenerations.id, id))
    .run()
  logTaskSuccess('VideoTask', 'downloaded', { id, localPath, storyboardId, duration: resolvedDuration })

  if (storyboardId) {
    db.update(schema.storyboards)
      .set({ videoUrl: localPath, duration: resolvedDuration ?? undefined, updatedAt: now() })
      .where(eq(schema.storyboards.id, storyboardId))
      .run()
  }

  // 资产版本历史留档（分镜视频）
  const [gen] = db.select().from(schema.videoGenerations).where(eq(schema.videoGenerations.id, id)).all()
  const versionSbId = storyboardId ?? gen?.storyboardId
  if (versionSbId) {
    recordAssetVersion({
      assetType: 'storyboard',
      assetId: versionSbId,
      mediaType: 'video',
      frameType: null,
      assetUrl: localPath,
      provider: gen?.provider,
      model: gen?.model,
      prompt: gen?.prompt,
      generationId: id,
      meta: gen?.duration ? { duration: gen.duration } : undefined,
    })
  }

  // 触发镜头级 QC 打分（fire-and-forget）
  const qcStoryboardId = storyboardId ?? gen?.storyboardId
  if (qcStoryboardId) runQcAfterVideoComplete(qcStoryboardId, id)
}

/** 恢复超时阈值：processing 状态超过该时长（毫秒）无进展，判定为孤儿任务直接失败 */
const RECOVER_TIMEOUT_MS = 60 * 60 * 1000 // 60 分钟

/**
 * 服务启动时恢复被中断的视频生成任务（崩溃恢复）。
 *
 * 根因：generateVideo 是 fire-and-forget，pollVideoTask 用内存 setTimeout 轮询，
 * 进程一旦重启这些轮询循环直接蒸发，status='processing' 的记录会永久卡死。
 *
 * 恢复策略（幂等，绝不重新提交，避免按次计费的厂商重复扣费）：
 *  - 无 taskId：提交请求前进程就崩了 → 判 failed
 *  - vidu（Webhook 型）：外部回调仍会到达 → 保持 processing 等待
 *  - 其余轮询型：按记录 provider 匹配活跃配置，恢复 pollVideoTask 续跑
 *  - 超时兜底：updatedAt 超过 RECOVER_TIMEOUT_MS 的 processing 直接判 failed
 */
export function recoverVideoTasksOnStartup() {
  const orphans = db.select().from(schema.videoGenerations)
    .where(eq(schema.videoGenerations.status, 'processing'))
    .all()

  if (orphans.length === 0) return
  logTaskWarn('VideoTask', 'recover-start', { count: orphans.length })

  for (const r of orphans) {
    try {
      // 超时兜底：超过阈值直接失败（厂商任务大概率已过期，无法续跑）
      if (r.updatedAt && Date.now() - new Date(r.updatedAt).getTime() > RECOVER_TIMEOUT_MS) {
        markRecoverFailed(r.id, `Interrupted and expired (idle > ${RECOVER_TIMEOUT_MS / 60000}min)`)
        continue
      }

      const provider = (r.provider || '').toLowerCase()

      if (!r.taskId) {
        // 提交请求前进程崩溃，无 taskId 无法续跑
        markRecoverFailed(r.id, 'Interrupted before task submit')
        continue
      }

      // 未知 provider 显式失败（getVideoAdapter 已改为 throw，不再静默 fallback minimax）
      if (!videoAdapters[provider]) {
        markRecoverFailed(r.id, `Unknown video provider: ${r.provider || '(empty)'}`)
        continue
      }

      if (provider === 'vidu') {
        // Webhook 型：外部回调仍会到达，保持 processing 等待
        logTaskWarn('VideoTask', 'recover-webhook-wait', { id: r.id, taskId: r.taskId, provider })
        continue
      }

      // 轮询型：按 provider 精确匹配活跃配置，恢复轮询
      const config = getActiveConfigByProvider('video', provider)
      if (!config) {
        // 无匹配配置：保持 processing，待配置补上后下次重启再恢复
        logTaskWarn('VideoTask', 'recover-no-config', { id: r.id, taskId: r.taskId, provider })
        continue
      }
      // 用记录里当初提交的模型覆盖，保证续跑同一模型
      if (r.model) config.model = r.model

      logTaskWarn('VideoTask', 'recover-resume-poll', { id: r.id, taskId: r.taskId, provider, model: r.model })
      pollVideoTask(r.id, config, r.taskId, r.storyboardId).catch(err => {
        logTaskError('VideoTask', 'recover-poll-failed', { id: r.id, error: err.message })
        markRecoverFailed(r.id, `Recover poll failed: ${err.message}`)
      })
    } catch (err: any) {
      logTaskError('VideoTask', 'recover-error', { id: r.id, error: err.message })
    }
  }
}

/** 标记恢复失败（复用 processVideoGeneration 的 failed 更新语义） */
function markRecoverFailed(id: number, reason: string) {
  db.update(schema.videoGenerations)
    .set({ status: 'failed', errorMsg: reason, updatedAt: now() })
    .where(eq(schema.videoGenerations.id, id))
    .run()
  logTaskError('VideoTask', 'recover-failed', { id, reason })
}
