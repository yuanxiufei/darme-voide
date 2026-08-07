import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, created, badRequest } from '../utils/response.js'
import { generateVideo } from '../services/video-generation.js'
import { logTaskError, logTaskPayload, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import {
  buildStoryboardVideoPrompt,
  getStoryboardCharacterAppearances,
  getStoryboardSceneDescription,
  getStoryboardCharacterImageUrls,
} from '../shared/prompt-utils.js'

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

    // 分镜视频生成：注入角色外观+场景+风格上下文
    if (body.storyboard_id) {
      const sbId = Number(body.storyboard_id)
      const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, sbId)).all()
      if (sb) {
        const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, sb.episodeId)).all()
        if (ep?.videoConfigId != null) configId = ep.videoConfigId

        // 自动注入角色外观和场景描述到 prompt（除非明确跳过）
        if (!body._skip_enrich) {
          const charAppearances = getStoryboardCharacterAppearances(sbId)
          const sceneDesc = getStoryboardSceneDescription(sbId)

          prompt = buildStoryboardVideoPrompt({
            description: body.prompt,
            storyboardDescription: sb.storyboardDescription,
            characterAppearances: charAppearances,
            scenePrompt: sceneDesc,
            action: sb.action,
            movement: sb.movement,
            dramaStyle: undefined,
          })

          // 自动添加角色图片作为 reference_images（首帧已由前端传）
          if (!referenceImageUrls?.length) {
            const charUrls = getStoryboardCharacterImageUrls(sbId)
            if (charUrls.length) referenceImageUrls = charUrls
          }
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
    const id = await generateVideo({
      storyboardId: body.storyboard_id,
      dramaId: body.drama_id,
      prompt,
      model: body.model,
      referenceMode: body.reference_mode,
      imageUrl: body.image_url,
      firstFrameUrl,
      lastFrameUrl: body.last_frame_url,
      referenceImageUrls,
      duration: body.duration,
      aspectRatio: body.aspect_ratio,
      configId,
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
  const id = Number(c.req.param('id'))
  const [row] = db.select().from(schema.videoGenerations)
    .where(eq(schema.videoGenerations.id, id)).all()
  return success(c, row || null)
})

// GET /videos — List by storyboard_id or drama_id
app.get('/', async (c) => {
  const storyboardId = c.req.query('storyboard_id')
  const dramaId = c.req.query('drama_id')

  let rows = db.select().from(schema.videoGenerations).all()

  if (storyboardId) rows = rows.filter(r => r.storyboardId === Number(storyboardId))
  if (dramaId) rows = rows.filter(r => r.dramaId === Number(dramaId))

  return success(c, rows)
})

// DELETE /videos/:id
app.delete('/:id', async (c) => {
  const id = Number(c.req.param('id'))
  db.delete(schema.videoGenerations).where(eq(schema.videoGenerations.id, id)).run()
  return success(c)
})

export default app
