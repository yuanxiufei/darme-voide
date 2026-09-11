import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, created, badRequest, notFound, now, parseParamId } from '../utils/response.js'
import { generateVideo } from '../services/video-generation.js'
import { getActiveConfig, getConfigById } from '../services/ai.js'
import { logTaskError, logTaskPayload, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import {
  buildStoryboardVideoPrompt,
  getStoryboardCharacterAppearances,
  getStoryboardSceneDescription,
  getStoryboardReferenceImages,
  getStoryboardReferenceAudioUrls,
  VIDEO_NEGATIVE,
} from '../shared/prompt-utils.js'
import { decideShotRoute } from '../services/shot-router.js'

const app = new Hono()

// POST /videos — Generate video
app.post('/', async (c) => {
  const body = await c.req.json()
  if (!body.prompt) return badRequest(c, 'prompt is required')

  try {
    let configId: number | undefined = body.config_id
    let prompt = body.prompt
    let firstFrameUrl = body.first_frame_url
    let referenceImageUrls = body.reference_image_urls
    let sceneType: string | undefined = body.scene_type
    let referenceAudioUrls: string[] | undefined = body.reference_audio_urls
    // 分镜行：上下文注入与逐镜路由共用同一次查询，保证「决策输入」与「请求输入」同源
    let sbRow: typeof schema.storyboards.$inferSelect | undefined = undefined

    // 分镜视频生成：注入角色外观+场景+风格上下文
    if (body.storyboard_id) {
      const sbId = Number(body.storyboard_id)
      sbRow = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, sbId)).all()[0]
      const sb = sbRow
      if (sb) {
        const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, sb.episodeId)).all()
        if (ep?.videoConfigId != null) configId = ep.videoConfigId

        // 自动注入角色外观和场景描述到 prompt（除非明确跳过）
        if (!body._skip_enrich) {
          const charAppearances = getStoryboardCharacterAppearances(sbId)
          const sceneDesc = getStoryboardSceneDescription(sbId)

          prompt = buildStoryboardVideoPrompt({
            description: body.prompt,
            storyboardDescription: sb.description,
            characterAppearances: charAppearances,
            scenePrompt: sceneDesc,
            action: sb.action,
            movement: sb.movement,
            dramaStyle: undefined,
            backgroundAudio: sb.soundEffect, // H3 原生 [background_audio] 场景声标记
          })

          // 自动添加参考图作为 reference_images（角色立绘 + 场景图 + 道具图，与 auto-pipeline 批量
          // 路径同源，保证手动/批量两种触发方式收到同一套跨集一致性参考；首帧已由前端传）
          if (!referenceImageUrls?.length) {
            const sbRefs = getStoryboardReferenceImages(sbId)
            if (sbRefs.length) referenceImageUrls = sbRefs
          }
        }

        if (!sceneType && sb.sceneType) sceneType = sb.sceneType
        // H3 音视频联合生成：对话类镜头自动带出场角色声线样本作为参考音频（Ref2VA）
        if (!referenceAudioUrls?.length && /dialogue|meeting|argument|conversation|multi/i.test(sb.sceneType || '')) {
          referenceAudioUrls = getStoryboardReferenceAudioUrls(sbId)
        }
      }
    }

    logTaskStart('VideoAPI', 'generate', {
      storyboardId: body.storyboard_id,
      dramaId: body.drama_id,
      referenceMode: body.reference_mode,
      duration: body.duration,
    })
    logTaskPayload('VideoAPI', 'enriched prompt', { original: body.prompt, enriched: prompt, hasCharRefs: !!referenceImageUrls?.length })

    // —— 帧来源统一：body 显式传帧优先 → 分镜已存帧兜底 ——
    // 此前路由决策读 DB 帧、生成请求只读 body 帧，两者不同源，会出现「决策判定 first_last/single，
    // 实发请求却一帧都不带」的坏请求（例如调用方只传 first_frame_url，而 adapter 的单帧分支只认
    // image_url）。这里统一成同一份候选帧，决策与请求共用。
    const sbFirstFrame = sbRow?.firstFrameImage || undefined
    const sbLastFrame = sbRow?.lastFrameImage || undefined
    const candFirstFrameUrl = firstFrameUrl || sbFirstFrame || undefined
    const candLastFrameUrl = body.last_frame_url || sbLastFrame || undefined

    // 逐镜路由（对齐 H3-Codex-Drama shot routing）：手动提交时按分镜属性决策生成路线，
    // 并回写 storyboards.route / route_reason + video_generations 快照（供可复现账本追溯）。
    let routeDecision: { route?: string; reason?: string; referenceMode?: 'none' | 'single' | 'first_last' | 'multiple' } = {}
    if (sbRow) {
      const provider = (body.config_id
        ? getConfigById(body.config_id)?.provider
        : getActiveConfig('video')?.provider) || 'default'
      const canMultiRef = ['volcengine', 'vidu', 'minimax'].includes(provider.toLowerCase()) || !!body.reference_mode
      routeDecision = decideShotRoute({
        storyboardId: sbRow.id,
        sceneType: sbRow.sceneType,
        firstFrameImage: candFirstFrameUrl,
        lastFrameImage: candLastFrameUrl,
        keyframeImage: sbRow.keyframeImage,
        blocked: sbRow.assetStatus === 'needs_regeneration' && !candFirstFrameUrl,
        provider: provider || 'default',
        canMultiRef,
        referenceImages: referenceImageUrls || [],
        referenceAudioUrls: referenceAudioUrls || [],
        prevTail: body.image_url && !body.first_frame_url ? body.image_url : undefined,
      })
    }

    // referenceMode 兜底对齐 adapter 契约：first_last（FL2VA）必须首尾帧齐备才会被 adapter 派发；
    // 只有单帧可用而目标模式落 first_last 时降级 single（用首帧起图），避免产生缺首帧的坏请求。
    const requestedMode = body.reference_mode || routeDecision.referenceMode || 'none'
    const effectiveMode =
      requestedMode === 'first_last' && !(candFirstFrameUrl && candLastFrameUrl)
        ? (candFirstFrameUrl || body.image_url ? 'single' : 'none')
        : requestedMode

    // 仅 single / first_last 消费帧字段（multiple 走 reference_image_urls、none 不用帧），
    // 这两种模式不注入兜底帧，保持既有请求体不变。
    const frameMode = effectiveMode === 'single' || effectiveMode === 'first_last'
    const outFirstFrameUrl = frameMode ? candFirstFrameUrl : firstFrameUrl
    const outLastFrameUrl = effectiveMode === 'first_last' ? candLastFrameUrl : body.last_frame_url
    // adapter 单帧分支只读 imageUrl（无 first_frame_url 入参），故 single 时把首帧统一落到 imageUrl
    const outImageUrl = effectiveMode === 'first_last'
      ? undefined
      : (body.image_url || (frameMode ? candFirstFrameUrl : undefined) || undefined)

    const id = await generateVideo({
      storyboardId: body.storyboard_id,
      dramaId: body.drama_id,
      prompt,
      negativePrompt: body.negative_prompt || VIDEO_NEGATIVE,
      model: body.model,
      referenceMode: effectiveMode,
      imageUrl: outImageUrl,
      firstFrameUrl: outFirstFrameUrl,
      lastFrameUrl: outLastFrameUrl,
      referenceImageUrls,
      sceneType,
      referenceAudioUrls,
      duration: body.duration,
      aspectRatio: body.aspect_ratio,
      configId,
      force: body.force,
      route: routeDecision.route,
      routeReason: routeDecision.reason,
    })

    const [record] = db.select().from(schema.videoGenerations)
      .where(eq(schema.videoGenerations.id, id)).all()
    logTaskSuccess('VideoAPI', 'generate', { generationId: id, provider: record?.provider })
    return created(c, record)
  } catch (err: any) {
    logTaskError('VideoAPI', 'generate', { error: err.message })
    return badRequest(c, err.message)
  }
})

// GET /videos/:id
app.get('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid video id')
  const [row] = db.select().from(schema.videoGenerations)
    .where(eq(schema.videoGenerations.id, id)).all()
  return success(c, row || null)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// GET /videos — List by storyboard_id or drama_id
app.get('/', async (c) => {
  try {
  const storyboardId = c.req.query('storyboard_id')
  const dramaId = c.req.query('drama_id')

  let rows = db.select().from(schema.videoGenerations).all()

  if (storyboardId) { const sid = Number(storyboardId); if (Number.isFinite(sid)) rows = rows.filter(r => r.storyboardId === sid) }
  if (dramaId) { const did = Number(dramaId); if (Number.isFinite(did)) rows = rows.filter(r => r.dramaId === did) }

  return success(c, rows)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// PUT /videos/:id — 更新视频生成参数
app.put('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid video id')
  const body = await c.req.json()
  const [row] = db.select().from(schema.videoGenerations).where(eq(schema.videoGenerations.id, id)).all()
  if (!row) return badRequest(c, '视频记录不存在')

  // snake_case API keys → drizzle camelCase column names
  const fieldMap: Record<string, string> = {
    prompt: 'prompt', negative_prompt: 'negativePrompt', model: 'model', duration: 'duration',
    reference_mode: 'referenceMode', aspect_ratio: 'aspectRatio',
    character_ids: 'characterIds', image_url: 'imageUrl',
    first_frame_url: 'firstFrameUrl', last_frame_url: 'lastFrameUrl',
  }
  const updates: Record<string, any> = {}
  for (const [apiKey, colName] of Object.entries(fieldMap)) {
    if (apiKey in body) updates[colName] = body[apiKey]
  }
  if (Object.keys(updates).length === 0) return badRequest(c, 'No fields to update')
  db.update(schema.videoGenerations).set(updates).where(eq(schema.videoGenerations.id, id)).run()
  const [updated] = db.select().from(schema.videoGenerations).where(eq(schema.videoGenerations.id, id)).all()
  return success(c, updated)
  } catch (err: any) { logTaskError('VideoAPI', 'update', { error: err.message, id: c.req.param('id') }); return badRequest(c, err.message) }
})

// POST /videos/:id/regenerate — 重新生成视频（可选模型+参数）
app.post('/:id/regenerate', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid video id')
  const body = await c.req.json()
  const [row] = db.select().from(schema.videoGenerations).where(eq(schema.videoGenerations.id, id)).all()
  if (!row) return badRequest(c, '视频记录不存在')

  const prompt = body.prompt || row.prompt
  if (!prompt) return badRequest(c, 'prompt is required')

  logTaskStart('VideoAPI', 'regenerate', {
    generationId: id, storyboardId: row.storyboardId, dramaId: row.dramaId,
    model: body.model || row.model || 'default',
  })

  // 获取 configId：优先从请求体 → 从原记录关联的 episode
  let configId: number | undefined = body.config_id
  if (!configId && row.storyboardId) {
    const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, row.storyboardId)).all()
    if (sb) {
      const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, sb.episodeId)).all()
      if (ep?.videoConfigId != null) configId = ep.videoConfigId
    }
  }

  const newId = await generateVideo({
    storyboardId: row.storyboardId ?? undefined,
    dramaId: row.dramaId ?? undefined,
    prompt,
    negativePrompt: body.negative_prompt || row.negativePrompt || VIDEO_NEGATIVE,
    model: body.model || row.model,
    referenceMode: body.reference_mode || row.referenceMode,
    imageUrl: body.image_url || row.imageUrl,
    firstFrameUrl: body.first_frame_url || row.firstFrameUrl,
    lastFrameUrl: body.last_frame_url || row.lastFrameUrl,
    referenceImageUrls: body.reference_image_urls || row.referenceImageUrls,
    sceneType: body.scene_type || row.sceneType || undefined,
    referenceAudioUrls: body.reference_audio_urls
      || (row.storyboardId ? getStoryboardReferenceAudioUrls(row.storyboardId) : undefined),
    duration: body.duration || row.duration,
    aspectRatio: body.aspect_ratio || row.aspectRatio,
    configId,  // ✅ 传递 configId
    force: body.force,
  })

  const [record] = db.select().from(schema.videoGenerations)
    .where(eq(schema.videoGenerations.id, newId)).all()
  logTaskSuccess('VideoAPI', 'regenerate', { generationId: newId, oldId: id, provider: record?.provider })
  return created(c, record)
  } catch (err: any) {
    logTaskError('VideoAPI', 'regenerate', { generationId: c.req.param('id'), error: err.message })
    return badRequest(c, err.message)
  }
})

// DELETE /videos/:id
app.delete('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid video id')
  db.delete(schema.videoGenerations).where(eq(schema.videoGenerations.id, id)).run()
  return success(c)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

export default app
