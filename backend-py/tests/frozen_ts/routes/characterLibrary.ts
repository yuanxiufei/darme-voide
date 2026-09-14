/**
 * 角色库路由 - 管理角色模板的 CRUD、搜索筛选、应用到项目
 */
import { Hono } from 'hono'
import { qAll, qGet, qRun } from '../db/queryHelper.js'

const Q_ALL = qAll as any
const Q_GET = qGet as any
const Q_RUN = qRun as any
import { now } from '../utils/response.js'

const router = new Hono()

// ====== 辅助函数 ======
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

interface LibraryQuery {
  page: number; pageSize: number; search: string; category: string
  tags: string[]; sortBy: string; sortOrder: 'asc' | 'desc'
}

function parseLibraryQuery(q: Record<string, string | string[]>): LibraryQuery {
  return {
    page: Math.max(1, parseInt(q.page as string) || 1),
    pageSize: Math.min(100, Math.max(1, parseInt(q.pageSize as string) || 20)),
    search: (q.search as string) || '',
    category: (q.category as string) || '',
    tags: (q.tags as string) ? (q.tags as string).split(',').map(t => t.trim()).filter(Boolean) : [],
    sortBy: (q.sortBy as string) || 'updated_at',
    sortOrder: ((q.sortOrder as string) || 'desc') === 'asc' ? 'asc' : 'desc',
  }
}

// ====== GET / - 分页列表 ======
router.get('/', async (c) => {
  try {
    const q = parseLibraryQuery(c.req.query())
    const params: any[] = []
    let where = 'WHERE deleted_at IS NULL'

    if (q.search) {
      const kw = `%${q.search}%`
      where += ` AND (name LIKE ? OR description LIKE ? OR tags LIKE ? OR appearance LIKE ? OR personality LIKE ? OR voice_style LIKE ?)`
      params.push(kw, kw, kw, kw, kw, kw)
    }
    if (q.tags.length > 0) {
      q.tags.forEach(t => { where += ' AND tags LIKE ?'; params.push(`%${t}%`) })
    }
    if (q.category) {
      const cats = q.category.split(',').filter(Boolean)
      if (cats.length > 0) {
        where += ` AND (${cats.map(() => 'category = ?').join(' OR ')})`
        params.push(...cats)
      }
    }

    const allowedSorts = ['name', 'category', 'usage_count', 'updated_at', 'created_at']
    const sortField = allowedSorts.includes(q.sortBy) ? q.sortBy : 'updated_at'
    const sortDir = q.sortOrder === 'asc' ? 'ASC' : 'DESC'

    const countResult = Q_ALL(`SELECT COUNT(*) as total FROM character_templates ${where}`, ...params) as any[]
    const total = countResult[0]?.total ?? 0
    const offset = (q.page - 1) * q.pageSize

    const items = Q_ALL(
      `SELECT * FROM character_templates ${where} ORDER BY ${sortField} ${sortDir} LIMIT ? OFFSET ?`,
      ...params, q.pageSize, offset,
    ) as any[]

    const parsed = items.map(item => ({
      ...item,
      tags: safeParseJson(item.tags),
      voiceConfig: safeParseJson(item.voice_config),
      metadata: safeParseJson(item.metadata),
      referenceImages: safeParseJson(item.reference_images),
    }))

    return c.json({ code: 0, data: { items: parsed, total, page: q.page, pageSize: q.pageSize, totalPages: Math.ceil(total / q.pageSize) } })
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message || '获取角色库列表失败' })
  }
})

// ====== GET /categories ======
router.get('/categories', async (c) => {
  try {
    const rows = Q_ALL('SELECT DISTINCT category FROM character_templates WHERE deleted_at IS NULL ORDER BY category') as any[]
    return c.json({ code: 0, data: rows.map(r => r.category) })
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message })
  }
})

// ====== GET /tags ======
router.get('/tags', async (c) => {
  try {
    const rows = Q_ALL('SELECT tags FROM character_templates WHERE deleted_at IS NULL AND tags IS NOT NULL') as any[]
    const tagSet = new Set<string>()
    rows.forEach(r => {
      const parsed = safeParseJson(r.tags)
      if (Array.isArray(parsed)) parsed.forEach((t: string) => tagSet.add(t))
    })
    return c.json({ code: 0, data: [...tagSet].sort() })
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message })
  }
})

// ====== GET /:id ======
router.get('/:id', async (c) => {
  try {
    const id = parseInt(c.req.param('id'))
    if (isNaN(id)) return c.json({ code: 400, data: null, message: '无效的ID参数' })
    const row = Q_GET('SELECT * FROM character_templates WHERE id = ? AND deleted_at IS NULL', id) as any
    if (!row) return c.json({ code: 404, data: null, message: '角色模板不存在' })
    row.tags = safeParseJson(row.tags)
    row.voiceConfig = safeParseJson(row.voice_config)
    row.metadata = safeParseJson(row.metadata)
    row.referenceImages = safeParseJson(row.reference_images)
    return c.json({ code: 0, data: row })
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message })
  }
})

// ====== POST / ======
router.post('/', async (c) => {
  try {
    const body = await c.req.json()
    const { name, category, description, appearance, personality, clothingStyle,
      expression, gender, ageGroup, imageUrl, referenceImages,
      voiceStyle, voiceProvider, voiceConfig, tags, metadata, sourceDramaId } = body

    if (!name || !appearance) return c.json({ code: 400, data: null, message: '名称和外貌描述为必填项' })

    const t = now()
    const result = Q_RUN(
      `INSERT INTO character_templates
        (name, category, description, appearance, personality, clothing_style,
         expression, gender, age_group, image_url, reference_images,
         voice_style, voice_provider, voice_config, tags, metadata,
         source_drama_id, created_at, updated_at)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
      name, category || '通用', description || '', appearance,
      personality || '', clothingStyle || '', expression || '',
      gender || '', ageGroup || '', imageUrl || '',
      safeStringify(referenceImages), voiceStyle || '', voiceProvider || '',
      safeStringify(voiceConfig), safeStringify(tags), safeStringify(metadata),
      sourceDramaId || null, t, t,
    )
    return c.json({ code: 0, data: { id: result.lastInsertRowid }, message: '创建成功' })
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message || '创建角色模板失败' })
  }
})

// ====== PUT /:id ======
router.put('/:id', async (c) => {
  try {
    const id = parseInt(c.req.param('id'))
    if (isNaN(id)) return c.json({ code: 400, data: null, message: '无效的ID参数' })
    const existing = Q_GET('SELECT * FROM character_templates WHERE id = ? AND deleted_at IS NULL', id) as any
    if (!existing) return c.json({ code: 404, data: null, message: '角色模板不存在' })

    const body = await c.req.json()
    const { name, category, description, appearance, personality, clothingStyle,
      expression, gender, ageGroup, imageUrl, referenceImages,
      voiceStyle, voiceProvider, voiceConfig, tags, metadata } = body

    const t = now()
    Q_RUN(
      `UPDATE character_templates SET
        name=?, category=?, description=?, appearance=?, personality=?,
        clothing_style=?, expression=?, gender=?, age_group=?,
        image_url=?, reference_images=?, voice_style=?, voice_provider=?,
        voice_config=?, tags=?, metadata=?, updated_at=?
       WHERE id=? AND deleted_at IS NULL`,
      name ?? existing.name, category ?? existing.category,
      description ?? existing.description, appearance ?? existing.appearance,
      personality ?? existing.personality, clothingStyle ?? existing.clothing_style,
      expression ?? existing.expression, gender ?? existing.gender,
      ageGroup ?? existing.age_group, imageUrl ?? existing.image_url,
      referenceImages !== undefined ? safeStringify(referenceImages) : existing.reference_images,
      voiceStyle ?? existing.voice_style,
      voiceProvider ?? existing.voice_provider,
      voiceConfig !== undefined ? safeStringify(voiceConfig) : existing.voice_config,
      tags !== undefined ? safeStringify(tags) : existing.tags,
      metadata !== undefined ? safeStringify(metadata) : existing.metadata,
      t, id,
    )
    return c.json({ code: 0, data: { id }, message: '更新成功' })
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message || '更新角色模板失败' })
  }
})

// ====== DELETE /:id ======
router.delete('/:id', async (c) => {
  try {
    const id = parseInt(c.req.param('id'))
    if (isNaN(id)) return c.json({ code: 400, data: null, message: '无效的ID参数' })
    const t = now()
    const result = Q_RUN('UPDATE character_templates SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL', t, id)
    if (result.changes === 0) return c.json({ code: 404, data: null, message: '角色模板不存在' })
    return c.json({ code: 0, data: { id }, message: '删除成功' })
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message || '删除角色模板失败' })
  }
})

// ====== POST /batch-delete ======
router.post('/batch-delete', async (c) => {
  try {
    const { ids } = await c.req.json()
    if (!Array.isArray(ids) || ids.length === 0) return c.json({ code: 400, data: null, message: '请提供要删除的ID列表' })
    const t = now()
    const placeholders = ids.map(() => '?').join(',')
    const result = Q_RUN(`UPDATE character_templates SET deleted_at = ? WHERE id IN (${placeholders}) AND deleted_at IS NULL`, t, ...ids)
    return c.json({ code: 0, data: { deletedCount: result.changes }, message: `成功删除 ${result.changes} 个角色模板` })
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message || '批量删除失败' })
  }
})

// ====== POST /:id/apply - 应用到剧组 ======
router.post('/:id/apply', async (c) => {
  try {
    const id = parseInt(c.req.param('id'))
    const { dramaId } = await c.req.json()
    if (!dramaId) return c.json({ code: 400, data: null, message: '请提供目标剧组ID' })

    const template = Q_GET('SELECT * FROM character_templates WHERE id = ? AND deleted_at IS NULL', id) as any
    if (!template) return c.json({ code: 404, data: null, message: '角色模板不存在' })

    const drama = Q_GET('SELECT id FROM dramas WHERE id = ? AND deleted_at IS NULL', dramaId) as any
    if (!drama) return c.json({ code: 404, data: null, message: '剧组不存在' })

    const t = now()
    const result = Q_RUN(
      `INSERT INTO characters
        (drama_id, name, role, description, appearance, personality,
         voice_style, voice_provider, image_url, reference_images, created_at, updated_at)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)`,
      dramaId, template.name, template.category || '',
      template.description || '', template.appearance,
      template.personality || '', template.voice_style || '',
      template.voice_provider || '', template.image_url || '',
      template.reference_images || '', t, t,
    )
    Q_RUN('UPDATE character_templates SET usage_count = usage_count + 1 WHERE id = ?', id)
    return c.json({ code: 0, data: { characterId: result.lastInsertRowid }, message: `角色 "${template.name}" 已应用到剧组` })
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message || '应用角色模板失败' })
  }
})

// ====== POST /from-character/:characterId - 从项目角色保存到模板库 ======
router.post('/from-character/:characterId', async (c) => {
  try {
    const charId = parseInt(c.req.param('characterId'))
    const character = Q_GET('SELECT * FROM characters WHERE id = ? AND deleted_at IS NULL', charId) as any
    if (!character) return c.json({ code: 404, data: null, message: '角色不存在' })

    const t = now()
    const result = Q_RUN(
      `INSERT INTO character_templates
        (name, category, description, appearance, personality,
         clothing_style, image_url, reference_images,
         voice_style, voice_provider, source_drama_id, created_at, updated_at)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)`,
      character.name, character.role || '通用', character.description || '',
      character.appearance || '', character.personality || '',
      '', character.image_url || '', character.reference_images || '',
      character.voice_style || '', character.voice_provider || '',
      character.drama_id, t, t,
    )
    return c.json({ code: 0, data: { templateId: result.lastInsertRowid }, message: `角色 "${character.name}" 已保存到角色库` })
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message || '保存角色到模板库失败' })
  }
})

export default router
