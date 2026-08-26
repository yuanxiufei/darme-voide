import { Hono } from 'hono'
import { and, eq, isNull } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, badRequest, notFound, now, parseParamId } from '../utils/response.js'
import { generateVoiceSample } from '../services/tts-generation.js'
import { generateImage } from '../services/image-generation.js'
import { logTaskError, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import { buildCharacterImagePrompt, buildEquipImagePrompt, CHARACTER_IMAGE_NEGATIVE } from '../shared/prompt-utils.js'
import { splitCharacterVisuals } from '../services/text-generation.js'
import { ensureCostumeId } from '../services/bible-ids.js'

const app = new Hono()

// 解析 drama 级生成配置：角色/场景的图片与音色试听是 drama 级共享资源，不绑定具体 episode。
// 优先取该 drama 下第一个已配置的 episode configId，否则返回 undefined（走全局默认配置）。
function resolveDramaConfigId(dramaId: number, field: 'imageConfigId' | 'audioConfigId'): number | undefined {
  const eps = db.select().from(schema.episodes)
    .where(and(eq(schema.episodes.dramaId, dramaId), isNull(schema.episodes.deletedAt)))
    .all()
  for (const e of eps) {
    const v = e[field] as number | null
    if (v != null) return v
  }
  return undefined
}

// GET /characters/:id
app.get('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  if (!char) return notFound(c, 'Character not found')
  return success(c, char)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// PUT /characters/:id
app.put('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const body = await c.req.json()
  const updates: Record<string, any> = { updatedAt: now() }
  for (const key of [
    'name', 'role', 'roleType', 'description', 'appearance', 'personality',
    'voiceStyle', 'voiceProvider', 'voiceSpeed', 'voiceEmotion',
    'voicePitch', 'voiceModel', 'clothing', 'weapons', 'accessories',
    'customPrompt', 'negativePrompt', 'coreFeatures', 'costumes', 'variations', 'threeViews', 'imageUrl', 'localPath',
  ]) {
    const snakeKey = key.replace(/[A-Z]/g, m => '_' + m.toLowerCase())
    if (snakeKey in body) updates[key] = body[snakeKey]
    else if (key in body) updates[key] = body[key]
  }
  if ('voice_style' in body || 'voiceStyle' in body) {
    updates.voiceSampleUrl = null
  }
  db.update(schema.characters).set(updates).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).run()
  ensureCostumeId(id)
  const [updated] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  return success(c, updated)
  } catch (err: any) { logTaskError('CharacterAPI', 'update', { error: err.message, id: c.req.param('id') }); return badRequest(c, err.message) }
})

// POST /characters/:id/auto-split-visuals — 从外貌特征智能拆分服装/武器/首饰并回填
app.post('/:id/auto-split-visuals', async (c) => {
  try {
    const id = parseParamId(c)
    if (id == null) return notFound(c, 'Invalid character id')
    const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
    if (!char) return notFound(c, 'Character not found')
    const body = await c.req.json().catch(() => ({}))
    const appearance = ((body as any).appearance || char.appearance || '').trim()
    if (!appearance) return badRequest(c, '请先填写「外貌特征」')

    logTaskStart('SplitVisuals', 'route', { characterId: id })
    const result = await splitCharacterVisuals({ appearance })
    logTaskSuccess('SplitVisuals', 'route', { characterId: id, result })
    return success(c, result)
  } catch (err: any) {
    logTaskError('SplitVisuals', 'route', { error: err.message, id: c.req.param('id') })
    return badRequest(c, `智能拆分失败: ${err.message}`)
  }
})

// DELETE /characters/:id
app.delete('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  db.update(schema.characters).set({ deletedAt: now() }).where(eq(schema.characters.id, id)).run()
  return success(c)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// POST /characters/:id/generate-voice-sample — 生成角色音色试听
app.post('/:id/generate-voice-sample', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const body = await c.req.json().catch(() => ({}))
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  if (!char) return badRequest(c, 'Character not found')
  if (!char.voiceStyle) return badRequest(c, '请先分配音色')
  const ep = body.episode_id
    ? db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()[0]
    : undefined
  if (body.episode_id && !ep) return badRequest(c, 'Episode not found')

  try {
    logTaskStart('VoiceSample', 'generate', { characterId: id, characterName: char.name, episodeId: ep?.id, voice: char.voiceStyle })
    const audioPath = await generateVoiceSample(char.name, char.voiceStyle, ep?.audioConfigId ?? resolveDramaConfigId(char.dramaId, 'audioConfigId'))
    db.update(schema.characters)
      .set({ voiceSampleUrl: audioPath, updatedAt: now() })
      .where(eq(schema.characters.id, id)).run()
    logTaskSuccess('VoiceSample', 'generate', { characterId: id, path: audioPath })
    return success(c, { voice_sample_url: audioPath })
  } catch (err: any) {
    logTaskError('VoiceSample', 'generate', { characterId: id, error: err.message })
    return badRequest(c, `TTS 生成失败: ${err.message}`)
  }
})

// POST /characters/:id/generate-image
app.post('/:id/generate-image', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const body = await c.req.json()
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  if (!char) return badRequest(c, 'Character not found')
  const ep = body.episode_id
    ? db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()[0]
    : undefined
  if (body.episode_id && !ep) return badRequest(c, 'Episode not found')

  // 自定义 prompt 优先 → 角色 customPrompt → 统一构建器
  const prompt = body.prompt
    || char.customPrompt
    || buildCharacterImagePrompt({ ...char, ...body })

  try {
    logTaskStart('CharacterImage', 'generate', { characterId: id, episodeId: ep?.id, dramaId: char.dramaId, model: body.model || 'default' })
    const genId = await generateImage({
      characterId: id,
      dramaId: char.dramaId,
      prompt,
      negativePrompt: body.negative_prompt || char.negativePrompt || CHARACTER_IMAGE_NEGATIVE,
      model: body.model,
      costume: body.costume,
      colorGrade: body.color_grade ?? body.colorGrade,
      configId: ep?.imageConfigId ?? resolveDramaConfigId(char.dramaId, 'imageConfigId'),
    })
    logTaskSuccess('CharacterImage', 'generate', { characterId: id, generationId: genId })
    return success(c, { image_generation_id: genId })
  } catch (err: any) {
    logTaskError('CharacterImage', 'generate', { characterId: id, error: err.message })
    return badRequest(c, err.message)
  }
})

// POST /characters/:id/generate-three-views — 生成角色三视图（正面/侧面/背面）
app.post('/:id/generate-three-views', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const body = await c.req.json().catch(() => ({}))
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  if (!char) return badRequest(c, 'Character not found')
  const ep = body.episode_id
    ? db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()[0]
    : undefined
  if (body.episode_id && !ep) return badRequest(c, 'Episode not found')

  // 可选指定视图，默认三个视图全生成
  const views = (body.views && Array.isArray(body.views) ? body.views : ['front', 'side', 'back'])
    .filter((v: string) => ['front', 'side', 'back'].includes(v))
  if (views.length === 0) return badRequest(c, 'views 必须包含 front/side/back 之一')

  const basePrompt = body.prompt || char.customPrompt || buildCharacterImagePrompt({ ...char, ...body })
  const negative = body.negative_prompt || char.negativePrompt || CHARACTER_IMAGE_NEGATIVE
  const colorGrade = body.color_grade ?? body.colorGrade
  const configId = ep?.imageConfigId ?? resolveDramaConfigId(char.dramaId, 'imageConfigId')
  const VIEW_LABELS: Record<string, string> = { front: '正面', side: '侧面', back: '背面' }

  const results: Array<{ view: string; image_generation_id: number }> = []
  for (const view of views) {
    const viewPrompt = `${basePrompt}，${VIEW_LABELS[view]}全身三视图，人物居中站立，完整展示服饰、首饰与武器细节`
    try {
      const genId = await generateImage({
        characterId: id,
        dramaId: char.dramaId,
        prompt: viewPrompt,
        negativePrompt: negative,
        model: body.model,
        viewType: view,
        colorGrade,
        configId,
      })
      results.push({ view, image_generation_id: genId })
    } catch (err: any) {
      logTaskError('CharacterImage', 'three-view-generate', { characterId: id, view, error: err.message })
    }
  }
  logTaskSuccess('CharacterImage', 'three-view-generate', { characterId: id, views: results.map(r => r.view) })
  return success(c, { count: results.length, results })
})

// POST /characters/:id/generate-equip-image — 生成服装/武器/首饰独立设定图
app.post('/:id/generate-equip-image', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const body = await c.req.json().catch(() => ({}))
  const type = body.type
  if (!['clothing', 'weapon', 'accessory'].includes(type)) {
    return badRequest(c, 'type 必须是 clothing/weapon/accessory 之一')
  }
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  if (!char) return badRequest(c, 'Character not found')
  const ep = body.episode_id
    ? db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()[0]
    : undefined
  if (body.episode_id && !ep) return badRequest(c, 'Episode not found')

  const prompt = body.prompt || buildEquipImagePrompt(type, char)
  const negative = body.negative_prompt || char.negativePrompt || CHARACTER_IMAGE_NEGATIVE

  try {
    logTaskStart('CharacterImage', 'equip-generate', { characterId: id, type })
    const genId = await generateImage({
      characterId: id,
      dramaId: char.dramaId,
      prompt,
      negativePrompt: negative,
      model: body.model,
      equipType: type,
      colorGrade: body.color_grade ?? body.colorGrade,
      configId: ep?.imageConfigId ?? resolveDramaConfigId(char.dramaId, 'imageConfigId'),
    })
    logTaskSuccess('CharacterImage', 'equip-generate', { characterId: id, type, generationId: genId })
    return success(c, { image_generation_id: genId })
  } catch (err: any) {
    logTaskError('CharacterImage', 'equip-generate', { characterId: id, type, error: err.message })
    return badRequest(c, err.message)
  }
})

// POST /characters/batch-generate-images
app.post('/batch-generate-images', async (c) => {
  const body = await c.req.json()
  const ids: number[] = body.character_ids || []
  if (!body.episode_id) return badRequest(c, 'episode_id is required')
  const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()
  if (!ep) return badRequest(c, 'Episode not found')
  const results: number[] = []
  for (const cid of ids) {
    const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, cid), isNull(schema.characters.deletedAt))).all()
    if (!char) continue
    const prompt = buildCharacterImagePrompt(char)
    try {
      const genId = await generateImage({
        characterId: cid,
        dramaId: char.dramaId,
        prompt,
        negativePrompt: body.negative_prompt || char.negativePrompt || CHARACTER_IMAGE_NEGATIVE,
        colorGrade: body.color_grade ?? body.colorGrade,
        configId: ep.imageConfigId ?? undefined,
      })
      results.push(genId)
    } catch {}
  }
  logTaskSuccess('CharacterImage', 'batch-generate', { episodeId: ep.id, requested: ids.length, started: results.length })
  return success(c, { count: results.length, ids: results })
})

export default app
