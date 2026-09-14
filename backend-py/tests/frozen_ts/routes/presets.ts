import { Hono } from 'hono'
import { desc, eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, badRequest, notFound, now, parseParamId } from '../utils/response.js'
import { normalizeColorGrade } from '../services/color-grade.js'

const app = new Hono()

// 将 DB 行转为 API 响应：config 反序列化为对象
function toApi(row: any) {
  let config: any = null
  if (row.config) {
    try { config = JSON.parse(row.config) } catch { config = null }
  }
  return { ...row, config }
}

// GET /presets?type=colorGrade — 列出预设（可按类型过滤）
app.get('/', async (c) => {
  try {
    const type = c.req.query('type')
    const rows = type
      ? db.select().from(schema.presets).where(eq(schema.presets.type, type)).orderBy(desc(schema.presets.updatedAt)).all()
      : db.select().from(schema.presets).orderBy(desc(schema.presets.updatedAt)).all()
    return success(c, rows.map(toApi))
  } catch (err: any) { return badRequest(c, err.message) }
})

// POST /presets — 创建预设
app.post('/', async (c) => {
  try {
    const body = await c.req.json()
    const type: string | undefined = body.type
    const name: string | undefined = body.name
    if (!type || !name) return badRequest(c, 'type and name are required')
    // colorGrade 类型：规整 config 为合法校色参数
    let config = body.config ?? null
    if (type === 'colorGrade' && config) config = normalizeColorGrade(config)
    const ts = now()
    const r = db.insert(schema.presets).values({
      type,
      name,
      config: config ? JSON.stringify(config) : null,
      createdAt: ts,
      updatedAt: ts,
    }).run()
    const [row] = db.select().from(schema.presets).where(eq(schema.presets.id, Number(r.lastInsertRowid))).all()
    return success(c, toApi(row))
  } catch (err: any) { return badRequest(c, err.message) }
})

// PUT /presets/:id — 更新预设
app.put('/:id', async (c) => {
  try {
    const id = parseParamId(c)
    if (id == null) return notFound(c, 'Invalid preset id')
    const body = await c.req.json()
    const updates: Record<string, any> = { updatedAt: now() }
    if (body.name != null) updates.name = body.name
    if (body.type != null) updates.type = body.type
    if (body.config != null) {
      const [existing] = db.select().from(schema.presets).where(eq(schema.presets.id, id)).all()
      if (!existing) return notFound(c, 'Preset not found')
      const type = body.type ?? existing.type
      updates.config = JSON.stringify(type === 'colorGrade' ? normalizeColorGrade(body.config) : body.config)
    }
    db.update(schema.presets).set(updates).where(eq(schema.presets.id, id)).run()
    const [row] = db.select().from(schema.presets).where(eq(schema.presets.id, id)).all()
    return success(c, toApi(row))
  } catch (err: any) { return badRequest(c, err.message) }
})

// DELETE /presets/:id — 删除预设
app.delete('/:id', async (c) => {
  try {
    const id = parseParamId(c)
    if (id == null) return notFound(c, 'Invalid preset id')
    db.delete(schema.presets).where(eq(schema.presets.id, id)).run()
    return success(c)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

export default app
