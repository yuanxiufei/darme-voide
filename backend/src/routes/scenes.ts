import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, created, badRequest, notFound, now, parseParamId } from '../utils/response.js'
import { generateImage } from '../services/image-generation.js'
import { logTaskError, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import { buildSceneImagePrompt } from '../shared/prompt-utils.js'

const app = new Hono()

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
  const [result] = db.select().from(schema.scenes)
    .where(eq(schema.scenes.id, Number(res.lastInsertRowid))).all()
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
    'lighting', 'weather', 'season', 'style', 'customPrompt', 'imageUrl',
  ]) {
    const snakeKey = key.replace(/[A-Z]/g, m => '_' + m.toLowerCase())
    if (snakeKey in body) updates[key] = body[snakeKey]
    else if (key in body) updates[key] = body[key]
  }
  db.update(schema.scenes).set(updates).where(eq(schema.scenes.id, id)).run()
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
  if (!body.episode_id) return badRequest(c, 'episode_id is required')
  const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()
  if (!ep) return badRequest(c, 'Episode not found')

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
    logTaskStart('SceneImage', 'generate', { sceneId: id, episodeId: ep.id, dramaId: scene.dramaId, location: scene.location, model: body.model || 'default' })
    db.update(schema.scenes).set({ status: 'processing', updatedAt: now() }).where(eq(schema.scenes.id, id)).run()
    const genId = await generateImage({ sceneId: id, dramaId: scene.dramaId, prompt, model: body.model, configId: ep.imageConfigId ?? undefined })
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
