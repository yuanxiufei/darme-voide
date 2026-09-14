/**
 * 生成历史聚合路由 — 对齐 gcc 的 RenderLogsModal / renderLogService。
 * 聚合 image_generations + video_generations 两张表，返回统一结构的历史记录，
 * 供前端「生成历史」面板展示（按时间倒序，支持类型 / 项目 / 分镜过滤）。
 */
import { Hono } from 'hono'
import { and, desc, eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success } from '../utils/response.js'

const app = new Hono()

function elapsed(createdAt?: string | null, completedAt?: string | null): number | null {
  if (!createdAt || !completedAt) return null
  const start = new Date(createdAt).getTime()
  const end = new Date(completedAt).getTime()
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) return null
  return end - start
}

// GET /generations — 聚合图片 / 视频生成历史（按 createdAt 倒序）
app.get('/', async (c) => {
  const type = c.req.query('type') // 'image' | 'video'（缺省=全部）
  const limitRaw = Number(c.req.query('limit') || 100)
  const limit = Number.isFinite(limitRaw) ? Math.min(Math.max(limitRaw, 1), 500) : 100

  const dramaIdRaw = Number(c.req.query('drama_id') || NaN)
  const storyboardIdRaw = Number(c.req.query('storyboard_id') || NaN)
  const dramaId = Number.isFinite(dramaIdRaw) ? dramaIdRaw : null
  const storyboardId = Number.isFinite(storyboardIdRaw) ? storyboardIdRaw : null

  const wantImage = !type || type === 'image'
  const wantVideo = !type || type === 'video'

  // 过滤与排序下推到 SQL 层，每张表只取前 limit 条，避免全表加载
  const items: any[] = []

  if (wantImage) {
    const conds: any[] = []
    if (dramaId != null) conds.push(eq(schema.imageGenerations.dramaId, dramaId))
    if (storyboardId != null) conds.push(eq(schema.imageGenerations.storyboardId, storyboardId))
    const where = conds.length === 0 ? undefined : conds.length === 1 ? conds[0] : and(...conds)
    const rows = db.select().from(schema.imageGenerations)
      .where(where)
      .orderBy(desc(schema.imageGenerations.createdAt))
      .limit(limit)
      .all()
    for (const r of rows) {
      items.push({
        id: r.id,
        type: 'image',
        storyboardId: r.storyboardId ?? null,
        dramaId: r.dramaId ?? null,
        provider: r.provider ?? null,
        model: r.model ?? null,
        prompt: r.prompt ?? null,
        status: r.status ?? 'pending',
        errorMsg: r.errorMsg ?? null,
        taskId: r.taskId ?? null,
        url: r.imageUrl ?? null,
        createdAt: r.createdAt,
        completedAt: r.completedAt ?? null,
        elapsedMs: elapsed(r.createdAt, r.completedAt),
      })
    }
  }

  if (wantVideo) {
    const conds: any[] = []
    if (dramaId != null) conds.push(eq(schema.videoGenerations.dramaId, dramaId))
    if (storyboardId != null) conds.push(eq(schema.videoGenerations.storyboardId, storyboardId))
    const where = conds.length === 0 ? undefined : conds.length === 1 ? conds[0] : and(...conds)
    const rows = db.select().from(schema.videoGenerations)
      .where(where)
      .orderBy(desc(schema.videoGenerations.createdAt))
      .limit(limit)
      .all()
    for (const r of rows) {
      items.push({
        id: r.id,
        type: 'video',
        storyboardId: r.storyboardId ?? null,
        dramaId: r.dramaId ?? null,
        provider: r.provider ?? null,
        model: r.model ?? null,
        prompt: r.prompt ?? null,
        status: r.status ?? 'pending',
        errorMsg: r.errorMsg ?? null,
        taskId: r.taskId ?? null,
        url: r.videoUrl ?? null,
        duration: r.duration ?? null,
        createdAt: r.createdAt,
        completedAt: r.completedAt ?? null,
        elapsedMs: elapsed(r.createdAt, r.completedAt),
      })
    }
  }

  items.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
  return success(c, items.slice(0, limit))
})

export default app
