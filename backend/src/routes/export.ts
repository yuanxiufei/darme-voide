import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { badRequest, notFound, parseParamId } from '../utils/response.js'
import { logTaskError } from '../utils/task-logger.js'
import {
  buildEdl,
  buildExportZip,
  collectDramaExportFiles,
  type ExportScope,
} from '../services/export-service.js'

const app = new Hono()

// GET /export/dramas/:dramaId/edl?episodeId=&fps=
// 生成 CMX3600 格式 EDL（编辑决策列表），供 Premiere / DaVinci Resolve 导入
app.get('/dramas/:dramaId/edl', async (c) => {
  try {
    const dramaId = parseParamId(c, 'dramaId')
    if (dramaId == null) return notFound(c, 'Invalid drama id')

    const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
    if (!drama) return notFound(c, 'Drama not found')

    const episodeIdRaw = c.req.query('episodeId')
    const episodeId = episodeIdRaw && !Number.isNaN(Number(episodeIdRaw)) ? Number(episodeIdRaw) : undefined
    const fpsRaw = c.req.query('fps')
    const fps = fpsRaw && !Number.isNaN(Number(fpsRaw)) ? Number(fpsRaw) : undefined

    const edl = await buildEdl(dramaId, { episodeId, fps })

    c.header('Content-Type', 'text/plain; charset=utf-8')
    c.header('Content-Disposition', `attachment; filename="drama-${dramaId}.edl"`)
    return c.body(edl)
  } catch (err: any) {
    logTaskError('ExportAPI', 'edl', { error: err.message, dramaId: c.req.param('dramaId') })
    return badRequest(c, err.message)
  }
})

// GET /export/dramas/:dramaId?scope=all|video|assets
// scope=video  仅打成片视频（episodes.videoUrl）
// scope=assets 仅打源素材（角色立绘 + 场景图 + 分镜关键帧）
// scope=all    全部（默认）
app.get('/dramas/:dramaId', (c) => {
  try {
    const dramaId = parseParamId(c, 'dramaId')
    if (dramaId == null) return notFound(c, 'Invalid drama id')

    const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
    if (!drama) return notFound(c, 'Drama not found')

    const scope = (c.req.query('scope') ?? 'all') as ExportScope
    if (scope !== 'all' && scope !== 'video' && scope !== 'assets') {
      return badRequest(c, 'scope must be one of: all | video | assets')
    }

    const files = collectDramaExportFiles(dramaId, scope)
    if (files.length === 0) {
      return badRequest(c, 'No exportable assets found (videos/assets not generated yet)')
    }

    const stream = buildExportZip(files)
    c.header('Content-Type', 'application/zip')
    c.header('Content-Disposition', `attachment; filename="drama-${dramaId}-export.zip"`)
    c.header('X-Export-Count', String(files.length))
    return c.body(stream)
  } catch (err: any) {
    logTaskError('ExportAPI', 'download', { error: err.message, dramaId: c.req.param('dramaId') })
    return badRequest(c, err.message)
  }
})

export default app
