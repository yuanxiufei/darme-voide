import { Hono } from 'hono'
import { desc, eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, badRequest, notFound, parseParamId } from '../utils/response.js'
import { mergeEpisodeVideos } from '../services/ffmpeg-merge.js'
import { toSnakeCase } from '../utils/transform.js'
import { logTaskError, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'

/**
 * merge = 整集拼接：把多个已合成镜头（composed_video_url）串接为完整剧集视频。
 * 注意与 compose（单镜合成）区分：compose 把视频 + 音频合成为一个镜头片段。
 */
const app = new Hono()

// POST /episodes/:id/merge — 拼接全集视频
app.post('/episodes/:id/merge', async (c) => {
  try {
  const episodeId = parseParamId(c)
  if (episodeId == null) return notFound(c, 'Invalid episode id')
  const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, episodeId)).all()
  if (!ep) return badRequest(c, 'Episode not found')

  logTaskStart('MergeAPI', 'episode-merge', { episodeId, dramaId: ep.dramaId })
  const mergeId = await mergeEpisodeVideos(episodeId, ep.dramaId)
  logTaskSuccess('MergeAPI', 'episode-merge', { episodeId, mergeId })
  return success(c, { merge_id: mergeId, status: 'processing' })
  } catch (err: any) {
    logTaskError('MergeAPI', 'episode-merge', { episodeId: c.req.param('id'), error: err.message })
    return badRequest(c, err.message)
  }
})

// GET /episodes/:id/merge — 查询拼接状态
app.get('/episodes/:id/merge', async (c) => {
  try {
  const episodeId = parseParamId(c)
  if (episodeId == null) return notFound(c, 'Invalid episode id')
  const merges = db.select().from(schema.videoMerges)
    .where(eq(schema.videoMerges.episodeId, episodeId))
    .orderBy(desc(schema.videoMerges.id))
    .all()

  const latest = merges[0]
  if (!latest) return success(c, null)

  return success(c, toSnakeCase(latest))
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

export default app
