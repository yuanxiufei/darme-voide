import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, created, now, badRequest } from '../utils/response.js'
import { generateImage } from '../services/image-generation.js'
import { logTaskError, logTaskPayload, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import {
  buildStoryboardImagePrompt,
  getStoryboardCharacterAppearances,
  getStoryboardSceneDescription,
  getStoryboardCharacterImageUrls,
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
            storyboardDescription: sb.storyboardDescription,
            characterDescription: charAppearances.length ? charAppearances.join('；') : null,
            sceneDescription: sceneDesc,
            location: sb.location,
            shotType: sb.shotType,
            cameraAngle: sb.cameraAngle,
            dramaStyle: undefined, // 可通过 episode->drama 补充
          })

          // 自动添加角色图片作为 reference_images
          if (!referenceImages?.length) {
            const charUrls = getStoryboardCharacterImageUrls(sbId)
            if (charUrls.length) referenceImages = charUrls
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
      model: body.model,
      size: body.size,
      referenceImages,
      frameType: body.frame_type,
      configId,
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
  const id = Number(c.req.param('id'))
  const [row] = db.select().from(schema.imageGenerations)
    .where(eq(schema.imageGenerations.id, id)).all()
  return success(c, row || null)
})

// GET /images — List by storyboard_id or drama_id
app.get('/', async (c) => {
  const storyboardId = c.req.query('storyboard_id')
  const dramaId = c.req.query('drama_id')

  let rows = db.select().from(schema.imageGenerations).all()

  if (storyboardId) rows = rows.filter(r => r.storyboardId === Number(storyboardId))
  if (dramaId) rows = rows.filter(r => r.dramaId === Number(dramaId))

  return success(c, rows)
})

// DELETE /images/:id
app.delete('/:id', async (c) => {
  const id = Number(c.req.param('id'))
  db.delete(schema.imageGenerations).where(eq(schema.imageGenerations.id, id)).run()
  return success(c)
})

export default app
