/**
 * 场景库路由 - 管理场景模板的 CRUD、搜索筛选、应用到项目
 */
import { Hono } from 'hono'
import { db } from '../db/index.js'
import { now } from '../utils/response.js'

const router = new Hono()

function safeParseJson(v: any) {
  if (!v) return v === '' ? [] : (Array.isArray(v) ? [] : null)
  if (typeof v === 'object') return v
  try { return JSON.parse(v) } catch { return v }
}

function safeStringify(v: any): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'string') {
    try { JSON.parse(v); return v } catch { return JSON.stringify(v) }
  }
  return JSON.stringify(v)
}

function parseQuery(q: Record<string, string | string[]>) {
  return {
    page: Math.max(1, parseInt(q.page as string) || 1),
    pageSize: Math.min(100, Math.max(1, parseInt(q.pageSize as string) || 20)),
    search: (q.search as string) || '',
    category: (q.category as string) || '',
    timeOfDay: (q.timeOfDay as string) || '',
    style: (q.style as string) || '',
    tags: (q.tags as string) ? (q.tags as string).split(',').map(t => t.trim()).filter(Boolean) : [],
    sortBy: (q.sortBy as string) || 'updated_at',
    sortOrder: ((q.sortOrder as string) || 'desc') === 'asc' ? 'asc' : 'desc',
  }
}

// ====== GET / ======
router.get('/', async (c) => {
  try {
    const q = parseQuery(c.req.query())
    const params: any[] = []
    let where = 'WHERE deleted_at IS NULL'

    if (q.search) {
      const kw = `%${q.search}%`
      where += ' AND (name LIKE ? OR description LIKE ? OR tags LIKE ? OR location LIKE ? OR atmosphere LIKE ? OR lighting LIKE ? OR prompt LIKE ?)'
      params.push(kw, kw, kw, kw, kw, kw, kw)
    }
    q.tags.forEach(t => { where += ' AND tags LIKE ?'; params.push(`%${t}%`) })
    if (q.category) {
      const cats = q.category.split(',').filter(Boolean)
      if (cats.length) { where += ` AND (${cats.map(() => 'category = ?').join(' OR ')})`; params.push(...cats) }
    }
    if (q.timeOfDay) { where += ' AND time_of_day = ?'; params.push(q.timeOfDay) }
    if (q.style) { where += ' AND style = ?'; params.push(q.style) }

    const allowedSorts = ['name', 'category', 'time_of_day', 'style', 'usage_count', 'updated_at', 'created_at']
    const sf = allowedSorts.includes(q.sortBy) ? q.sortBy : 'updated_at'
    const sd = q.sortOrder === 'asc' ? 'ASC' : 'DESC'

    const total = (db.all(`SELECT COUNT(*) as total FROM scene_templates ${where}`, ...params) as any[])[0]?.total ?? 0
    const offset = (q.page - 1) * q.pageSize

    const items = db.all(`SELECT * FROM scene_templates ${where} ORDER BY ${sf} ${sd} LIMIT ? OFFSET ?`, ...params, q.pageSize, offset) as any[]
    const parsed = items.map(item => ({ ...item, tags: safeParseJson(item.tags), metadata: safeParseJson(item.metadata), referenceImages: safeParseJson(item.reference_images) }))

    return c.json({ code: 0, data: { items: parsed, total, page: q.page, pageSize: q.pageSize, totalPages: Math.ceil(total / q.pageSize) } })
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message })
  }
})

router.get('/categories', async (c) => {
  const rows = db.all('SELECT DISTINCT category FROM scene_templates WHERE deleted_at IS NULL ORDER BY category') as any[]
  return c.json({ code: 0, data: rows.map(r => r.category) })
})

router.get('/filter-options', async (c) => {
  const timeOfDay = (db.all('SELECT DISTINCT time_of_day FROM scene_templates WHERE deleted_at IS NULL AND time_of_day IS NOT NULL ORDER BY time_of_day') as any[]).map(r => r.time_of_day)
  const styles = (db.all('SELECT DISTINCT style FROM scene_templates WHERE deleted_at IS NULL AND style IS NOT NULL ORDER BY style') as any[]).map(r => r.style)
  return c.json({ code: 0, data: { timeOfDay, styles } })
})

router.get('/tags', async (c) => {
  const rows = db.all('SELECT tags FROM scene_templates WHERE deleted_at IS NULL AND tags IS NOT NULL') as any[]
  const tagSet = new Set<string>()
  rows.forEach(r => { const p = safeParseJson(r.tags); if (Array.isArray(p)) p.forEach((t: string) => tagSet.add(t)) })
  return c.json({ code: 0, data: [...tagSet].sort() })
})

router.get('/:id', async (c) => {
  const id = parseInt(c.req.param('id'))
  const row = db.get('SELECT * FROM scene_templates WHERE id = ? AND deleted_at IS NULL', id) as any
  if (!row) return c.json({ code: 404, data: null, message: '场景模板不存在' })
  row.tags = safeParseJson(row.tags); row.metadata = safeParseJson(row.metadata); row.referenceImages = safeParseJson(row.reference_images)
  return c.json({ code: 0, data: row })
})

router.post('/', async (c) => {
  try {
    const body = await c.req.json()
    const { name, category, description, location, atmosphere, lighting, timeOfDay, style, season, weather, imageUrl, referenceImages, prompt, tags, metadata, sourceDramaId } = body
    if (!name) return c.json({ code: 400, data: null, message: '名称为必填项' })
    const t = now()
    const r = db.run(
      `INSERT INTO scene_templates (name,category,description,location,atmosphere,lighting,time_of_day,style,season,weather,image_url,reference_images,prompt,tags,metadata,source_drama_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
      name, category || '通用', description || '', location || '', atmosphere || '', lighting || '', timeOfDay || '', style || '', season || '', weather || '', imageUrl || '', safeStringify(referenceImages), prompt || '', safeStringify(tags), safeStringify(metadata), sourceDramaId || null, t, t,
    )
    return c.json({ code: 0, data: { id: r.lastInsertRowid }, message: '创建成功' })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

router.put('/:id', async (c) => {
  try {
    const id = parseInt(c.req.param('id'))
    const existing = db.get('SELECT * FROM scene_templates WHERE id = ? AND deleted_at IS NULL', id) as any
    if (!existing) return c.json({ code: 404, data: null, message: '场景模板不存在' })
    const body = await c.req.json()
    const { name, category, description, location, atmosphere, lighting, timeOfDay, style, season, weather, imageUrl, referenceImages, prompt, tags, metadata } = body
    const t = now()
    db.run(
      `UPDATE scene_templates SET name=?,category=?,description=?,location=?,atmosphere=?,lighting=?,time_of_day=?,style=?,season=?,weather=?,image_url=?,reference_images=?,prompt=?,tags=?,metadata=?,updated_at=? WHERE id=?`,
      name ?? existing.name, category ?? existing.category, description ?? existing.description, location ?? existing.location, atmosphere ?? existing.atmosphere, lighting ?? existing.lighting, timeOfDay ?? existing.time_of_day, style ?? existing.style, season ?? existing.season, weather ?? existing.weather, imageUrl ?? existing.image_url, referenceImages !== undefined ? safeStringify(referenceImages) : existing.reference_images, prompt ?? existing.prompt, tags !== undefined ? safeStringify(tags) : existing.tags, metadata !== undefined ? safeStringify(metadata) : existing.metadata, t, id,
    )
    return c.json({ code: 0, data: { id }, message: '更新成功' })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

router.delete('/:id', async (c) => {
  const id = parseInt(c.req.param('id'))
  const r = db.run('UPDATE scene_templates SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL', now(), id)
  if (r.changes === 0) return c.json({ code: 404, data: null, message: '场景模板不存在' })
  return c.json({ code: 0, data: { id }, message: '删除成功' })
})

router.post('/batch-delete', async (c) => {
  const { ids } = await c.req.json()
  if (!Array.isArray(ids) || ids.length === 0) return c.json({ code: 400, data: null, message: '请提供要删除的ID列表' })
  const t = now(); const p = ids.map(() => '?').join(',')
  const r = db.run(`UPDATE scene_templates SET deleted_at = ? WHERE id IN (${p}) AND deleted_at IS NULL`, t, ...ids)
  return c.json({ code: 0, data: { deletedCount: r.changes }, message: `成功删除 ${r.changes} 个场景模板` })
})

router.post('/:id/apply', async (c) => {
  try {
    const id = parseInt(c.req.param('id'))
    const { dramaId, episodeId } = await c.req.json()
    if (!dramaId) return c.json({ code: 400, data: null, message: '请提供目标剧组ID' })
    const tpl = db.get('SELECT * FROM scene_templates WHERE id = ? AND deleted_at IS NULL', id) as any
    if (!tpl) return c.json({ code: 404, data: null, message: '场景模板不存在' })
    const t = now()
    const r = db.run(
      `INSERT INTO scenes (drama_id,episode_id,location,time,prompt,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)`,
      dramaId, episodeId || null, tpl.location || tpl.name, tpl.time_of_day || '白天', tpl.prompt || `${tpl.location}，${tpl.atmosphere}，${tpl.lighting}`, 'pending', t, t,
    )
    db.run('UPDATE scene_templates SET usage_count = usage_count + 1 WHERE id = ?', id)
    return c.json({ code: 0, data: { sceneId: r.lastInsertRowid }, message: `场景 "${tpl.name}" 已应用到剧组` })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

router.post('/from-scene/:sceneId', async (c) => {
  try {
    const sid = parseInt(c.req.param('sceneId'))
    const sc = db.get('SELECT * FROM scenes WHERE id = ? AND deleted_at IS NULL', sid) as any
    if (!sc) return c.json({ code: 404, data: null, message: '场景不存在' })
    const t = now()
    const r = db.run(`INSERT INTO scene_templates (name,category,location,time_of_day,prompt,image_url,source_drama_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)`,
      sc.location || '未命名场景', '通用', sc.location, sc.time, sc.prompt, sc.image_url || '', sc.drama_id, t, t)
    return c.json({ code: 0, data: { templateId: r.lastInsertRowid }, message: '场景已保存到场景库' })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

export default router
