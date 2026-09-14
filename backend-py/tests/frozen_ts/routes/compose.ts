import { Hono } from 'hono'
import { eq, inArray } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, badRequest, notFound, parseParamId } from '../utils/response.js'
import { composeStoryboard } from '../services/ffmpeg-compose.js'
import { logTaskError, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import { toSnakeCase } from '../utils/transform.js'

/**
 * compose = 单镜合成：把单个镜头的视频 + 配音/音频合成为成片片段（composed_video_url）。
 * 注意与 merge（整集拼接）区分：merge 把多个已合成镜头串接为完整剧集。
 */
const app = new Hono()

// POST /storyboards/:id/compose — 合成单个镜头
app.post('/storyboards/:id/compose', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  try {
    logTaskStart('ComposeAPI', 'single-compose', { storyboardId: id })
    const composedUrl = await composeStoryboard(id)
    logTaskSuccess('ComposeAPI', 'single-compose', { storyboardId: id, output: composedUrl })
    return success(c, { id, composed_video_url: composedUrl })
  } catch (err: any) {
    logTaskError('ComposeAPI', 'single-compose', { storyboardId: id, error: err.message })
    return badRequest(c, err.message)
  }
})

// POST /episodes/:id/compose-all — 批量合成全部镜头
app.post('/episodes/:id/compose-all', async (c) => {
  try {
  const episodeId = parseParamId(c)
  if (episodeId == null) return notFound(c, 'Invalid episode id')
  const storyboards = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, episodeId))
    .orderBy(schema.storyboards.storyboardNumber)
    .all()

  if (storyboards.length === 0) return badRequest(c, 'No storyboards found')

  const withVideo = storyboards.filter(sb => sb.videoUrl)
  if (withVideo.length === 0) return badRequest(c, 'No storyboards have video yet')

  logTaskStart('ComposeAPI', 'batch-compose', { episodeId, total: withVideo.length })

  // 只标记有视频的 storyboard 为合成中（无视频的保持原状）
  db.update(schema.storyboards)
    .set({ status: 'compose_processing' })
    .where(inArray(schema.storyboards.id, withVideo.map(sb => sb.id)))
    .run()

  ;(async () => {
    let ok = 0; let fail = 0
    for (const sb of withVideo) {
      try {
        await composeStoryboard(sb.id)
        ok++
      } catch (err: any) {
        fail++
        logTaskError('ComposeAPI', 'batch-item', { storyboardId: sb.id, episodeId, error: err.message })
      }
    }
    logTaskSuccess('ComposeAPI', 'batch-compose', { episodeId, total: withVideo.length, ok, fail })
  })()

  return success(c, {
    message: `Started composing ${withVideo.length} storyboards`,
    total: withVideo.length,
  })
  } catch (err: any) { logTaskError('ComposeAPI', 'batch-compose', { episodeId: c.req.param('id'), error: err.message }); return badRequest(c, err.message) }
})

// GET /episodes/:id/compose-status — 查询批量合成状态
app.get('/episodes/:id/compose-status', async (c) => {
  try {
  const episodeId = parseParamId(c)
  if (episodeId == null) return notFound(c, 'Invalid episode id')
  const storyboards = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, episodeId))
    .orderBy(schema.storyboards.storyboardNumber)
    .all()

  const withVideo = storyboards.filter(sb => !!sb.videoUrl)
  const completed = withVideo.filter(sb => sb.status === 'compose_completed' && !!sb.composedVideoUrl)
  const failed = withVideo.filter(sb => sb.status === 'compose_failed')
  const processing = withVideo.filter(sb => sb.status === 'compose_processing')
  const idle = withVideo.filter(sb => !sb.status || !String(sb.status).startsWith('compose_'))

  return success(c, {
    total: withVideo.length,
    completed: completed.length,
    failed: failed.length,
    processing: processing.length,
    idle: idle.length,
    items: withVideo.map((sb) => toSnakeCase({
      id: sb.id,
      storyboardNumber: sb.storyboardNumber,
      status: sb.status || 'pending',
      composedVideoUrl: sb.composedVideoUrl,
      errorMsg: sb.status === 'compose_failed' ? '视频合成失败，请检查视频、配音或字幕素材' : '',
    })),
  })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

export default app
