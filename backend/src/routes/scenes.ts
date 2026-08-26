import { Hono } from 'hono'
import { and, eq, isNull } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, created, badRequest, notFound, now, parseParamId } from '../utils/response.js'
import { generateImage } from '../services/image-generation.js'
import { logTaskError, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import { buildSceneImagePrompt, SCENE_IMAGE_NEGATIVE } from '../shared/prompt-utils.js'
import { ensureLocationId } from '../services/bible-ids.js'

const app = new Hono()

// 解析 drama 级生成配置：场景图是 drama 级共享资源，不绑定具体 episode。
// 优先取该 drama 下第一个已配置 imageConfigId 的 episode，否则返回 undefined（走全局默认配置）。
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

// GET /scenes/:id
app.get('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid scene id')
  const [scene] = db.select().from(schema.scenes).where(eq(schema.scenes.id, id)).all()
  if (!scene) return notFound(c, 'Scene not found')
  return success(c, scene)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// POST /scenes
app.post('/', async (c) => {
  try {
  const body = await c.req.json()
  const ts = now()
  const res = db.insert(schema.scenes).values({
    dramaId: body.drama_id,
    episodeId: body.episode_id,
    location: body.location,
    time: body.time || '',
    prompt: body.prompt || body.location,
    createdAt: ts,
    updatedAt: ts,
  }).run()
  const sceneId = Number(res.lastInsertRowid)
  ensureLocationId(sceneId)
  const [result] = db.select().from(schema.scenes)
    .where(eq(schema.scenes.id, sceneId)).all()
  return created(c, result)
  } catch (err: any) { logTaskError('SceneAPI', 'create', { error: err.message }); return badRequest(c, err.message) }
})

// PUT /scenes/:id
app.put('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid scene id')
  const body = await c.req.json()
  const updates: Record<string, any> = { updatedAt: now() }
  for (const key of [
    'location', 'time', 'prompt', 'description', 'atmosphere',
    'lighting', 'weather', 'season', 'style', 'customPrompt', 'negativePrompt', 'imageUrl',
  ]) {
    const snakeKey = key.replace(/[A-Z]/g, m => '_' + m.toLowerCase())
    if (snakeKey in body) updates[key] = body[snakeKey]
    else if (key in body) updates[key] = body[key]
  }
  db.update(schema.scenes).set(updates).where(eq(schema.scenes.id, id)).run()
  // 地点变化 → 重新锁定 location_id（六键 Bible 跨集锁定）
  if (updates.location !== undefined) {
    db.update(schema.scenes).set({ locationId: null }).where(eq(schema.scenes.id, id)).run()
  }
  ensureLocationId(id)
  const [updated] = db.select().from(schema.scenes).where(eq(schema.scenes.id, id)).all()
  return success(c, updated)
  } catch (err: any) { logTaskError('SceneAPI', 'update', { error: err.message, id: c.req.param('id') }); return badRequest(c, err.message) }
})

// POST /scenes/:id/generate-image
app.post('/:id/generate-image', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid scene id')
  const body = await c.req.json()
  const [scene] = db.select().from(schema.scenes).where(eq(schema.scenes.id, id)).all()
  if (!scene) return badRequest(c, 'Scene not found')
  const ep = body.episode_id
    ? db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()[0]
    : undefined
  if (body.episode_id && !ep) return badRequest(c, 'Episode not found')

  const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, scene.dramaId)).all()
  const prompt = body.prompt
    || scene.customPrompt
    || buildSceneImagePrompt({
      location: scene.location,
      time: scene.time,
      prompt: scene.prompt,
      dramaStyle: drama?.style || undefined,
    })
  try {
    logTaskStart('SceneImage', 'generate', { sceneId: id, episodeId: ep?.id, dramaId: scene.dramaId, location: scene.location, model: body.model || 'default' })
    db.update(schema.scenes).set({ status: 'processing', updatedAt: now() }).where(eq(schema.scenes.id, id)).run()
    const genId = await generateImage({ sceneId: id, dramaId: scene.dramaId, prompt, negativePrompt: body.negative_prompt || scene.negativePrompt || SCENE_IMAGE_NEGATIVE, model: body.model, configId: ep?.imageConfigId ?? resolveDramaConfigId(scene.dramaId, 'imageConfigId') })
    logTaskSuccess('SceneImage', 'generate', { sceneId: id, generationId: genId })
    return success(c, { image_generation_id: genId })
  } catch (err: any) {
    logTaskError('SceneImage', 'generate', { sceneId: id, error: err.message })
    db.update(schema.scenes).set({ status: 'failed', updatedAt: now() }).where(eq(schema.scenes.id, id)).run()
    return badRequest(c, err.message)
  }
})

// DELETE /scenes/:id
app.delete('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid scene id')
  db.delete(schema.scenes).where(eq(schema.scenes.id, id)).run()
  return success(c)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

export default app
