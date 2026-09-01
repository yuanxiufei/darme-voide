import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, created, now, badRequest, notFound, parseParamId } from '../utils/response.js'
import { generateImage } from '../services/image-generation.js'
import { logTaskError, logTaskPayload, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import {
  buildStoryboardImagePrompt,
  getStoryboardCharacterAppearances,
  getStoryboardSceneDescription,
  getStoryboardReferenceImages,
  STORYBOARD_IMAGE_NEGATIVE,
  NEGATIVE_BASE,
} from '../shared/prompt-utils.js'

const app = new Hono()

// POST /images — Generate image
app.post('/', async (c) => {
  const body = await c.req.json()
  if (!body.prompt) return badRequest(c, 'prompt is required')

  try {
    let configId: number | undefined = body.config_id
    let prompt = body.prompt
    let referenceImages = body.reference_images

    // 分镜图片生成：注入角色外观+场景+风格上下文
    if (body.storyboard_id) {
      const sbId = Number(body.storyboard_id)
      const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, sbId)).all()
      if (sb) {
        const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, sb.episodeId)).all()
        if (ep?.imageConfigId != null) configId = ep.imageConfigId

        // 自动注入角色外观和场景描述到 prompt
        if (!body._skip_enrich) {
          const charAppearances = getStoryboardCharacterAppearances(sbId)
          const sceneDesc = getStoryboardSceneDescription(sbId)

          prompt = buildStoryboardImagePrompt({
            description: body.prompt,
            storyboardDescription: sb.description,
            characterDescription: charAppearances.length ? charAppearances.join('；') : null,
            sceneDescription: sceneDesc,
            location: sb.location,
            shotType: sb.shotType,
            cameraAngle: sb.angle,
            dramaStyle: undefined, // 可通过 episode->drama 补充
          })

          // 自动添加角色图片 + 场景图作为 reference_images，保证人物与场景一致
          if (!referenceImages?.length) {
            const refUrls = getStoryboardReferenceImages(sbId)
            if (refUrls.length) referenceImages = refUrls
          }
        }
      }
    }

    logTaskStart('ImageAPI', 'generate', {
      storyboardId: body.storyboard_id,
      sceneId: body.scene_id,
      characterId: body.character_id,
      dramaId: body.drama_id,
      frameType: body.frame_type,
    })
    logTaskPayload('ImageAPI', 'enriched prompt', { original: body.prompt, enriched: prompt, referenceCount: referenceImages?.length || 0 })
    const id = await generateImage({
      storyboardId: body.storyboard_id,
      dramaId: body.drama_id,
      sceneId: body.scene_id,
      characterId: body.character_id,
      prompt,
      negativePrompt: body.negative_prompt || (body.storyboard_id ? STORYBOARD_IMAGE_NEGATIVE : NEGATIVE_BASE),
      model: body.model,
      size: body.size,
      referenceImages,
      frameType: body.frame_type,
      configId,
      force: body.force,
    })

    const [record] = db.select().from(schema.imageGenerations)
      .where(eq(schema.imageGenerations.id, id)).all()
    logTaskSuccess('ImageAPI', 'generate', { generationId: id, provider: record?.provider })
    return created(c, record)
  } catch (err: any) {
    logTaskError('ImageAPI', 'generate', { error: err.message })
    return badRequest(c, err.message)
  }
})

// GET /images/:id
app.get('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid image id')
  const [row] = db.select().from(schema.imageGenerations)
    .where(eq(schema.imageGenerations.id, id)).all()
  return success(c, row || null)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// GET /images — List by storyboard_id or drama_id
app.get('/', async (c) => {
  try {
  const storyboardId = c.req.query('storyboard_id')
  const dramaId = c.req.query('drama_id')

  let rows = db.select().from(schema.imageGenerations).all()

  if (storyboardId) { const sid = Number(storyboardId); if (Number.isFinite(sid)) rows = rows.filter(r => r.storyboardId === sid) }
  if (dramaId) { const did = Number(dramaId); if (Number.isFinite(did)) rows = rows.filter(r => r.dramaId === did) }

  return success(c, rows)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// DELETE /images/:id
app.delete('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid image id')
  db.delete(schema.imageGenerations).where(eq(schema.imageGenerations.id, id)).run()
  return success(c)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

export default app
