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
import { buildJianYingDraft } from '../services/jianying-draft.js'
import { buildProjectLedger, buildProjectLedgerMarkdown, scanStaleShots } from '../services/project-ledger.js'
import { buildContactSheetHtml, buildQcReport, buildQcReportHtml, buildQcReportMarkdown } from '../services/qc-report.js'

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

// GET /export/dramas/:dramaId/jianying-draft?episodeId=
// 导出剪映草稿（.draft 文件夹打包 ZIP）：对齐 ArcReel 的剪映草稿交付能力，
// 用户可在剪映中继续调整字幕 / 配音 / 节奏 / 转场。
app.get('/dramas/:dramaId/jianying-draft', (c) => {
  try {
    const dramaId = parseParamId(c, 'dramaId')
    if (dramaId == null) return notFound(c, 'Invalid drama id')

    const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
    if (!drama) return notFound(c, 'Drama not found')

    const episodeIdRaw = c.req.query('episodeId')
    const episodeId = episodeIdRaw && !Number.isNaN(Number(episodeIdRaw)) ? Number(episodeIdRaw) : undefined

    const drafts = buildJianYingDraft(dramaId, episodeId)

    const files = drafts.flatMap(d => d.files)
    if (files.length === 0) return badRequest(c, 'No composed videos found for draft export')

    const stream = buildExportZip(files)
    const filename = episodeId ? `drama-${dramaId}-episode-${episodeId}-jianying-draft.zip` : `drama-${dramaId}-jianying-draft.zip`
    c.header('Content-Type', 'application/zip')
    c.header('Content-Disposition', `attachment; filename="${filename}"`)
    c.header('X-Draft-Count', String(drafts.length))
    return c.body(stream)
  } catch (err: any) {
    logTaskError('ExportAPI', 'jianying-draft', { error: err.message, dramaId: c.req.param('dramaId') })
    return badRequest(c, err.message)
  }
})

// GET /export/dramas/:dramaId/project-ledger?format=json|md&episodeId=
// 导出工程账本（对齐 H3-Codex-Drama 可复现工程 project.yaml）：
// - format=json：完整结构化账本（提示词/输入指纹/路由/生成记录/成本），供断点续作与程序消费
// - format=md：人类可读 Markdown 报告，交付制片/审计
// GET /export/dramas/:dramaId/project-ledger/stale?episodeId=
// 断点续作扫描：返回「账本指纹 vs 当前剧本」不一致的 stale 分镜清单
app.get('/dramas/:dramaId/project-ledger/stale', (c) => {
  try {
    const dramaId = parseParamId(c, 'dramaId')
    if (dramaId == null) return notFound(c, 'Invalid drama id')

    const episodeIdRaw = c.req.query('episodeId')
    const episodeId = episodeIdRaw && !Number.isNaN(Number(episodeIdRaw)) ? Number(episodeIdRaw) : undefined

    const result = scanStaleShots(dramaId, episodeId)
    c.header('Content-Type', 'application/json; charset=utf-8')
    return c.json(result)
  } catch (err: any) {
    logTaskError('ExportAPI', 'project-ledger-stale', { error: err.message, dramaId: c.req.param('dramaId') })
    return badRequest(c, err.message)
  }
})

app.get('/dramas/:dramaId/project-ledger', (c) => {
  try {
    const dramaId = parseParamId(c, 'dramaId')
    if (dramaId == null) return notFound(c, 'Invalid drama id')

    const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
    if (!drama) return notFound(c, 'Drama not found')

    const episodeIdRaw = c.req.query('episodeId')
    const episodeId = episodeIdRaw && !Number.isNaN(Number(episodeIdRaw)) ? Number(episodeIdRaw) : undefined
    const format = c.req.query('format') ?? 'json'

    if (format === 'md' || format === 'markdown') {
      const md = buildProjectLedgerMarkdown(dramaId, episodeId)
      c.header('Content-Type', 'text/markdown; charset=utf-8')
      c.header('Content-Disposition', `attachment; filename="drama-${dramaId}-project-ledger.md"`)
      return c.body(md)
    }

    const ledger = buildProjectLedger(dramaId, episodeId)
    c.header('Content-Type', 'application/json; charset=utf-8')
    c.header('Content-Disposition', `attachment; filename="drama-${dramaId}-project-ledger.json"`)
    return c.json(ledger)
  } catch (err: any) {
    logTaskError('ExportAPI', 'project-ledger', { error: err.message, dramaId: c.req.param('dramaId') })
    return badRequest(c, err.message)
  }
})

// GET /export/dramas/:dramaId/qc-report?episodeId=&format=json|md|html
// 导出 QC 报告交付物（对齐参考项目 QC 报告交付物）：每镜 QC 分数 + 问题清单 + 时间线汇总
app.get('/dramas/:dramaId/qc-report', async (c) => {
  try {
    const dramaId = parseParamId(c, 'dramaId')
    if (dramaId == null) return notFound(c, 'Invalid drama id')

    const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
    if (!drama) return notFound(c, 'Drama not found')

    const episodeIdRaw = c.req.query('episodeId')
    const episodeId = episodeIdRaw && !Number.isNaN(Number(episodeIdRaw)) ? Number(episodeIdRaw) : undefined
    const format = c.req.query('format') ?? 'json'
    const opts = { episodeId }

    if (format === 'md' || format === 'markdown') {
      const md = await buildQcReportMarkdown(dramaId, opts)
      c.header('Content-Type', 'text/markdown; charset=utf-8')
      c.header('Content-Disposition', `attachment; filename="drama-${dramaId}-qc-report.md"`)
      return c.body(md)
    }
    if (format === 'html') {
      const html = await buildQcReportHtml(dramaId, opts)
      c.header('Content-Type', 'text/html; charset=utf-8')
      c.header('Content-Disposition', `attachment; filename="drama-${dramaId}-qc-report.html"`)
      return c.body(html)
    }

    const report = await buildQcReport(dramaId, opts)
    c.header('Content-Type', 'application/json; charset=utf-8')
    c.header('Content-Disposition', `attachment; filename="drama-${dramaId}-qc-report.json"`)
    return c.json(report)
  } catch (err: any) {
    logTaskError('ExportAPI', 'qc-report', { error: err.message, dramaId: c.req.param('dramaId') })
    return badRequest(c, err.message)
  }
})

// GET /export/dramas/:dramaId/contact-sheet?episodeId=
// 导出联系表 HTML（每镜首帧/尾帧缩略图 + 元信息网格，可浏览器打印/转 PDF）
app.get('/dramas/:dramaId/contact-sheet', async (c) => {
  try {
    const dramaId = parseParamId(c, 'dramaId')
    if (dramaId == null) return notFound(c, 'Invalid drama id')

    const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
    if (!drama) return notFound(c, 'Drama not found')

    const episodeIdRaw = c.req.query('episodeId')
    const episodeId = episodeIdRaw && !Number.isNaN(Number(episodeIdRaw)) ? Number(episodeIdRaw) : undefined

    const html = await buildContactSheetHtml(dramaId, { episodeId })
    c.header('Content-Type', 'text/html; charset=utf-8')
    const filename = episodeId
      ? `drama-${dramaId}-episode-${episodeId}-contact-sheet.html`
      : `drama-${dramaId}-contact-sheet.html`
    c.header('Content-Disposition', `attachment; filename="${filename}"`)
    return c.body(html)
  } catch (err: any) {
    logTaskError('ExportAPI', 'contact-sheet', { error: err.message, dramaId: c.req.param('dramaId') })
    return badRequest(c, err.message)
  }
})

export default app
