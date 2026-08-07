/**
 * 服装库路由 - 管理服装模板的 CRUD、搜索筛选
 */
import { Hono } from 'hono'
import { db } from '../db/index.js'
import { qAll, qGet, qRun } from '../db/queryHelper.js'

const Q_ALL = (...args: any[]) => (qAll as any)(...args)
const Q_GET = (...args: any[]) => (qGet as any)(...args)
const Q_RUN = (...args: any[]) => (qRun as any)(...args)
import { now } from '../utils/response.js'

const router = new Hono()

const COSTUME_STYLES = ['古风', '现代', '仙侠', '武侠', '科幻', '宫廷', '民俗', '其他']
const BODY_PARTS = ['全身', '上衣', '下装', '外套', '鞋履', '头饰', '配饰']
const SEASONS = ['春', '夏', '秋', '冬', '通用']

function safeParseJson(v: any) {
  if (!v) return v === '' ? [] : (Array.isArray(v) ? [] : null)
  if (typeof v === 'object') return v
  try { return JSON.parse(v) } catch { return v }
}
function safeStringify(v: any): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'string') { try { JSON.parse(v); return v } catch { return JSON.stringify(v) } }
  return JSON.stringify(v)
}

function parseQuery(q: Record<string, string | string[]>) {
  return {
    page: Math.max(1, parseInt(q.page as string) || 1),
    pageSize: Math.min(100, Math.max(1, parseInt(q.pageSize as string) || 20)),
    search: (q.search as string) || '',
    category: (q.category as string) || '',
    style: (q.style as string) || '',
    bodyPart: (q.bodyPart as string) || '',
    tags: (q.tags as string) ? (q.tags as string).split(',').map(t => t.trim()).filter(Boolean) : [],
    sortBy: (q.sortBy as string) || 'updated_at',
    sortOrder: ((q.sortOrder as string) || 'desc') === 'asc' ? 'asc' : 'desc',
  }
}

router.get('/', async (c) => {
  try {
    const q = parseQuery(c.req.query())
    const params: any[] = []
    let where = 'WHERE deleted_at IS NULL'

    if (q.search) {
      const kw = `%${q.search}%`
      where += ' AND (name LIKE ? OR description LIKE ? OR tags LIKE ? OR appearance LIKE ? OR material LIKE ? OR color_scheme LIKE ?)'
      params.push(kw, kw, kw, kw, kw, kw)
    }
    q.tags.forEach(t => { where += ' AND tags LIKE ?'; params.push(`%${t}%`) })
    if (q.category) {
      const cats = q.category.split(',').filter(Boolean)
      if (cats.length) { where += ` AND (${cats.map(() => 'category = ?').join(' OR ')})`; params.push(...cats) }
    }
    if (q.style) { where += ' AND style = ?'; params.push(q.style) }
    if (q.bodyPart) { where += ' AND body_part = ?'; params.push(q.bodyPart) }

    const allowedSorts = ['name', 'category', 'style', 'body_part', 'season', 'usage_count', 'updated_at', 'created_at']
    const sf = allowedSorts.includes(q.sortBy) ? q.sortBy : 'updated_at'
    const sd = q.sortOrder === 'asc' ? 'ASC' : 'DESC'

    const total = (Q_ALL(`SELECT COUNT(*) as total FROM costume_templates ${where}`, ...params) as any[])[0]?.total ?? 0
    const offset = (q.page - 1) * q.pageSize
    const items = Q_ALL(`SELECT * FROM costume_templates ${where} ORDER BY ${sf} ${sd} LIMIT ? OFFSET ?`, ...params, q.pageSize, offset) as any[]
    const parsed = items.map(item => ({ ...item, tags: safeParseJson(item.tags), metadata: safeParseJson(item.metadata), referenceImages: safeParseJson(item.reference_images) }))
    return c.json({ code: 0, data: { items: parsed, total, page: q.page, pageSize: q.pageSize, totalPages: Math.ceil(total / q.pageSize) } })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

router.get('/filter-options', async (c) => {
  const styles = (Q_ALL('SELECT DISTINCT style FROM costume_templates WHERE deleted_at IS NULL AND style IS NOT NULL ORDER BY style') as any[]).map(r => r.style)
  const bodyParts = (Q_ALL('SELECT DISTINCT body_part FROM costume_templates WHERE deleted_at IS NULL AND body_part IS NOT NULL ORDER BY body_part') as any[]).map(r => r.body_part)
  const seasons = (Q_ALL('SELECT DISTINCT season FROM costume_templates WHERE deleted_at IS NULL AND season IS NOT NULL ORDER BY season') as any[]).map(r => r.season)
  return c.json({ code: 0, data: { styles: styles.length ? styles : COSTUME_STYLES, bodyParts: bodyParts.length ? bodyParts : BODY_PARTS, seasons: seasons.length ? seasons : SEASONS } })
})

router.get('/categories', async (c) => {
  const rows = Q_ALL('SELECT DISTINCT category FROM costume_templates WHERE deleted_at IS NULL ORDER BY category') as any[]
  return c.json({ code: 0, data: rows.map(r => r.category) })
})

router.get('/tags', async (c) => {
  const rows = Q_ALL('SELECT tags FROM costume_templates WHERE deleted_at IS NULL AND tags IS NOT NULL') as any[]
  const tagSet = new Set<string>()
  rows.forEach(r => { const p = safeParseJson(r.tags); if (Array.isArray(p)) p.forEach((t: string) => tagSet.add(t)) })
  return c.json({ code: 0, data: [...tagSet].sort() })
})

router.get('/:id', async (c) => {
  const id = parseInt(c.req.param('id'))
  const row = qGet('SELECT * FROM costume_templates WHERE id = ? AND deleted_at IS NULL', id) as any
  if (!row) return c.json({ code: 404, data: null, message: '服装模板不存在' })
  row.tags = safeParseJson(row.tags); row.metadata = safeParseJson(row.metadata); row.referenceImages = safeParseJson(row.reference_images)
  return c.json({ code: 0, data: row })
})

router.post('/', async (c) => {
  try {
    const body = await c.req.json()
    const { name, category, description, style, bodyPart, material, colorScheme, season, appearance, imageUrl, referenceImages, tags, metadata, sourceDramaId } = body
    if (!name) return c.json({ code: 400, data: null, message: '名称为必填项' })
    const t = now()
    const r = Q_RUN(
      `INSERT INTO costume_templates (name,category,description,style,body_part,material,color_scheme,season,appearance,image_url,reference_images,tags,metadata,source_drama_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
      name, category || '通用', description || '', style || '', bodyPart || '', material || '', colorScheme || '', season || '', appearance || '', imageUrl || '', safeStringify(referenceImages), safeStringify(tags), safeStringify(metadata), sourceDramaId || null, t, t,
    )
    return c.json({ code: 0, data: { id: r.lastInsertRowid }, message: '创建成功' })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

router.put('/:id', async (c) => {
  try {
    const id = parseInt(c.req.param('id'))
    const existing = qGet('SELECT * FROM costume_templates WHERE id = ? AND deleted_at IS NULL', id) as any
    if (!existing) return c.json({ code: 404, data: null, message: '服装模板不存在' })
    const body = await c.req.json()
    const { name, category, description, style, bodyPart, material, colorScheme, season, appearance, imageUrl, referenceImages, tags, metadata } = body
    const t = now()
    Q_RUN(
      `UPDATE costume_templates SET name=?,category=?,description=?,style=?,body_part=?,material=?,color_scheme=?,season=?,appearance=?,image_url=?,reference_images=?,tags=?,metadata=?,updated_at=? WHERE id=?`,
      name ?? existing.name, category ?? existing.category, description ?? existing.description, style ?? existing.style, bodyPart ?? existing.body_part, material ?? existing.material, colorScheme ?? existing.color_scheme, season ?? existing.season, appearance ?? existing.appearance, imageUrl ?? existing.image_url, referenceImages !== undefined ? safeStringify(referenceImages) : existing.reference_images, tags !== undefined ? safeStringify(tags) : existing.tags, metadata !== undefined ? safeStringify(metadata) : existing.metadata, t, id,
    )
    return c.json({ code: 0, data: { id }, message: '更新成功' })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

router.delete('/:id', async (c) => {
  const id = parseInt(c.req.param('id'))
  const r = Q_RUN('UPDATE costume_templates SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL', now(), id)
  if (r.changes === 0) return c.json({ code: 404, data: null, message: '服装模板不存在' })
  return c.json({ code: 0, data: { id }, message: '删除成功' })
})

router.post('/batch-delete', async (c) => {
  const { ids } = await c.req.json()
  if (!Array.isArray(ids) || ids.length === 0) return c.json({ code: 400, data: null, message: '请提供要删除的ID列表' })
  const t = now(); const p = ids.map(() => '?').join(',')
  const r = Q_RUN(`UPDATE costume_templates SET deleted_at = ? WHERE id IN (${p}) AND deleted_at IS NULL`, t, ...ids)
  return c.json({ code: 0, data: { deletedCount: r.changes }, message: `成功删除 ${r.changes} 个服装模板` })
})

export default router
