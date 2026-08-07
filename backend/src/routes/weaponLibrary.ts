/**
 * 兵器库路由 - 管理兵器模板的 CRUD、搜索筛选
 */
import { Hono } from 'hono'
import { db } from '../db/index.js'
import { qAll, qGet, qRun } from '../db/queryHelper.js'

const Q_ALL = qAll as any
const Q_GET = qGet as any
const Q_RUN = qRun as any
import { now } from '../utils/response.js'

const router = new Hono()

const WEAPON_CATEGORIES = ['剑', '刀', '枪', '棍', '斧', '锤', '弓', '弩', '扇', '鞭', '杖', '暗器', '法宝', '其他']
const WEAPON_TYPES = ['近战', '远程', '暗器', '法宝']
const WEAPON_RANKS = ['凡品', '灵品', '仙品', '神品']

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
    type: (q.type as string) || '',
    rank: (q.rank as string) || '',
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
      where += ' AND (name LIKE ? OR description LIKE ? OR tags LIKE ? OR appearance LIKE ? OR material LIKE ? OR attributes LIKE ?)'
      params.push(kw, kw, kw, kw, kw, kw)
    }
    q.tags.forEach(t => { where += ' AND tags LIKE ?'; params.push(`%${t}%`) })
    if (q.category) {
      const cats = q.category.split(',').filter(Boolean)
      if (cats.length) { where += ` AND (${cats.map(() => 'category = ?').join(' OR ')})`; params.push(...cats) }
    }
    if (q.type) { where += ' AND type = ?'; params.push(q.type) }
    if (q.rank) { where += ' AND rank = ?'; params.push(q.rank) }

    const allowedSorts = ['name', 'category', 'type', 'rank', 'usage_count', 'updated_at', 'created_at']
    const sf = allowedSorts.includes(q.sortBy) ? q.sortBy : 'updated_at'
    const sd = q.sortOrder === 'asc' ? 'ASC' : 'DESC'

    const total = (Q_ALL(`SELECT COUNT(*) as total FROM weapon_templates ${where}`, ...params) as any[])[0]?.total ?? 0
    const offset = (q.page - 1) * q.pageSize
    const items = Q_ALL(`SELECT * FROM weapon_templates ${where} ORDER BY ${sf} ${sd} LIMIT ? OFFSET ?`, ...params, q.pageSize, offset) as any[]
    const parsed = items.map(item => ({ ...item, tags: safeParseJson(item.tags), attributes: safeParseJson(item.attributes), metadata: safeParseJson(item.metadata), referenceImages: safeParseJson(item.reference_images) }))
    return c.json({ code: 0, data: { items: parsed, total, page: q.page, pageSize: q.pageSize, totalPages: Math.ceil(total / q.pageSize) } })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

router.get('/filter-options', async (c) => {
  const cats = (Q_ALL('SELECT DISTINCT category FROM weapon_templates WHERE deleted_at IS NULL AND category IS NOT NULL ORDER BY category') as any[]).map(r => r.category)
  const types = (Q_ALL('SELECT DISTINCT type FROM weapon_templates WHERE deleted_at IS NULL AND type IS NOT NULL ORDER BY type') as any[]).map(r => r.type)
  const ranks = (Q_ALL('SELECT DISTINCT rank FROM weapon_templates WHERE deleted_at IS NULL AND rank IS NOT NULL ORDER BY rank') as any[]).map(r => r.rank)
  return c.json({ code: 0, data: { categories: cats.length ? cats : WEAPON_CATEGORIES, types: types.length ? types : WEAPON_TYPES, ranks: ranks.length ? ranks : WEAPON_RANKS } })
})

router.get('/categories', async (c) => {
  const rows = Q_ALL('SELECT DISTINCT category FROM weapon_templates WHERE deleted_at IS NULL ORDER BY category') as any[]
  const cats = rows.map(r => r.category)
  return c.json({ code: 0, data: cats.length ? cats : WEAPON_CATEGORIES })
})

router.get('/tags', async (c) => {
  const rows = Q_ALL('SELECT tags FROM weapon_templates WHERE deleted_at IS NULL AND tags IS NOT NULL') as any[]
  const tagSet = new Set<string>()
  rows.forEach(r => { const p = safeParseJson(r.tags); if (Array.isArray(p)) p.forEach((t: string) => tagSet.add(t)) })
  return c.json({ code: 0, data: [...tagSet].sort() })
})

router.get('/:id', async (c) => {
  const id = parseInt(c.req.param('id'))
  const row = Q_GET('SELECT * FROM weapon_templates WHERE id = ? AND deleted_at IS NULL', id) as any
  if (!row) return c.json({ code: 404, data: null, message: '兵器模板不存在' })
  row.tags = safeParseJson(row.tags); row.attributes = safeParseJson(row.attributes); row.metadata = safeParseJson(row.metadata); row.referenceImages = safeParseJson(row.reference_images)
  return c.json({ code: 0, data: row })
})

router.post('/', async (c) => {
  try {
    const body = await c.req.json()
    const { name, category, type, description, appearance, material, attributes, rank, ownerCharacterName, imageUrl, referenceImages, tags, metadata, sourceDramaId } = body
    if (!name) return c.json({ code: 400, data: null, message: '名称为必填项' })
    const t = now()
    const r = Q_RUN(
      `INSERT INTO weapon_templates (name,category,type,description,appearance,material,attributes,rank,owner_character_name,image_url,reference_images,tags,metadata,source_drama_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
      name, category || '剑', type || '', description || '', appearance || '', material || '', safeStringify(attributes), rank || '', ownerCharacterName || '', imageUrl || '', safeStringify(referenceImages), safeStringify(tags), safeStringify(metadata), sourceDramaId || null, t, t,
    )
    return c.json({ code: 0, data: { id: r.lastInsertRowid }, message: '创建成功' })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

router.put('/:id', async (c) => {
  try {
    const id = parseInt(c.req.param('id'))
    const existing = Q_GET('SELECT * FROM weapon_templates WHERE id = ? AND deleted_at IS NULL', id) as any
    if (!existing) return c.json({ code: 404, data: null, message: '兵器模板不存在' })
    const body = await c.req.json()
    const { name, category, type, description, appearance, material, attributes, rank, ownerCharacterName, imageUrl, referenceImages, tags, metadata } = body
    const t = now()
    Q_RUN(
      `UPDATE weapon_templates SET name=?,category=?,type=?,description=?,appearance=?,material=?,attributes=?,rank=?,owner_character_name=?,image_url=?,reference_images=?,tags=?,metadata=?,updated_at=? WHERE id=?`,
      name ?? existing.name, category ?? existing.category, type ?? existing.type, description ?? existing.description, appearance ?? existing.appearance, material ?? existing.material, attributes !== undefined ? safeStringify(attributes) : existing.attributes, rank ?? existing.rank, ownerCharacterName ?? existing.owner_character_name, imageUrl ?? existing.image_url, referenceImages !== undefined ? safeStringify(referenceImages) : existing.reference_images, tags !== undefined ? safeStringify(tags) : existing.tags, metadata !== undefined ? safeStringify(metadata) : existing.metadata, t, id,
    )
    return c.json({ code: 0, data: { id }, message: '更新成功' })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

router.delete('/:id', async (c) => {
  const id = parseInt(c.req.param('id'))
  const r = Q_RUN('UPDATE weapon_templates SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL', now(), id)
  if (r.changes === 0) return c.json({ code: 404, data: null, message: '兵器模板不存在' })
  return c.json({ code: 0, data: { id }, message: '删除成功' })
})

router.post('/batch-delete', async (c) => {
  const { ids } = await c.req.json()
  if (!Array.isArray(ids) || ids.length === 0) return c.json({ code: 400, data: null, message: '请提供要删除的ID列表' })
  const t = now(); const p = ids.map(() => '?').join(',')
  const r = Q_RUN(`UPDATE weapon_templates SET deleted_at = ? WHERE id IN (${p}) AND deleted_at IS NULL`, t, ...ids)
  return c.json({ code: 0, data: { deletedCount: r.changes }, message: `成功删除 ${r.changes} 个兵器模板` })
})

router.post('/from-prop/:propId', async (c) => {
  try {
    const pid = parseInt(c.req.param('propId'))
    const prop = Q_GET('SELECT * FROM props WHERE id = ? AND deleted_at IS NULL', pid) as any
    if (!prop) return c.json({ code: 404, data: null, message: '道具不存在' })
    const t = now()
    const r = Q_RUN(`INSERT INTO weapon_templates (name,category,description,appearance,image_url,reference_images,source_drama_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)`,
      prop.name, prop.type || '其他', prop.description || '', prop.prompt || '', prop.image_url || '', prop.reference_images || '', prop.drama_id, t, t)
    return c.json({ code: 0, data: { templateId: r.lastInsertRowid }, message: `道具 "${prop.name}" 已保存到兵器库` })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

export default router
