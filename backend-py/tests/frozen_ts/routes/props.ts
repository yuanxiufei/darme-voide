/**
 * 物品库（props）路由 —— 连续性状态机 v3
 *
 * 道具/信物/线索/法器资产：CRUD + 设定图生成。
 * extractor 提取的关键物品落在 prop_templates，
 * 分镜通过 storyboard_props 关联，图片/视频生成时作为参考图注入。
 */
import { Hono } from 'hono'
import { and, eq, isNull } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { generateImage } from '../services/image-generation.js'
import { buildPropImagePrompt, buildUiPlateImagePrompt, UI_PLATE_CATEGORY } from '../shared/prompt-utils.js'
import { notFound, badRequest, success } from '../utils/response.js'
import { logTaskError, logTaskStart } from '../utils/task-logger.js'

const app = new Hono()

const parseId = (v: string): number | null => {
  const n = Number(v)
  return Number.isInteger(n) && n > 0 ? n : null
}

// GET /props?drama_id= — 列表（可过滤）
app.get('/', async (c) => {
  const dramaId = Number(c.req.query('drama_id') || '0')
  const rows = db
    .select()
    .from(schema.propTemplates)
    .where(isNull(schema.propTemplates.deletedAt))
    .all()
  const list = rows
    .filter((p) => !dramaId || p.dramaId === dramaId)
    .sort((a, b) => a.id - b.id)
  return success(c, list)
})

// GET /props/:id — 详情
app.get('/:id', async (c) => {
  const id = parseId(c.req.param('id'))
  if (!id) return notFound(c, 'Invalid prop id')
  const [p] = db
    .select()
    .from(schema.propTemplates)
    .where(and(eq(schema.propTemplates.id, id), isNull(schema.propTemplates.deletedAt)))
    .all()
  if (!p) return notFound(c, '物品不存在')
  return success(c, p)
})

// POST /props — 创建
app.post('/', async (c) => {
  const body = await c.req.json().catch(() => ({}))
  const { drama_id, name, category, description, appearance, size_hint, holder, key_clue, custom_prompt, negative_prompt } = body
  if (!drama_id || !name) return badRequest(c, 'drama_id 和 name 必填')
  const ts = new Date().toISOString()
  const [ins] = db
    .insert(schema.propTemplates)
    .values({
      dramaId: Number(drama_id),
      name: String(name),
      category: category || '道具',
      description,
      appearance,
      sizeHint: size_hint,
      holder,
      keyClue: key_clue,
      customPrompt: custom_prompt,
      negativePrompt: negative_prompt,
      createdAt: ts,
      updatedAt: ts,
    })
    .returning()
    .all()
  return success(c, ins)
})

// PUT /props/:id — 更新
app.put('/:id', async (c) => {
  const id = parseId(c.req.param('id'))
  if (!id) return notFound(c, 'Invalid prop id')
  const body = await c.req.json().catch(() => ({}))
  const updates: Record<string, unknown> = { updatedAt: new Date().toISOString() }
  const fieldMap: Record<string, string> = {
    name: 'name', category: 'category', description: 'description',
    appearance: 'appearance', size_hint: 'sizeHint', holder: 'holder',
    key_clue: 'keyClue', custom_prompt: 'customPrompt', negative_prompt: 'negativePrompt',
    image_url: 'imageUrl',
  }
  for (const [k, dbk] of Object.entries(fieldMap)) {
    if (body[k] !== undefined) updates[dbk] = body[k]
  }
  db.update(schema.propTemplates).set(updates).where(eq(schema.propTemplates.id, id)).run()
  const [row] = db.select().from(schema.propTemplates).where(eq(schema.propTemplates.id, id)).all()
  return success(c, row)
})

// DELETE /props/:id — 软删除
app.delete('/:id', async (c) => {
  const id = parseId(c.req.param('id'))
  if (!id) return notFound(c, 'Invalid prop id')
  const ts = new Date().toISOString()
  db.update(schema.propTemplates)
    .set({ deletedAt: ts, updatedAt: ts })
    .where(eq(schema.propTemplates.id, id))
    .run()
  return success(c, { ok: true })
})

// POST /props/:id/generate-image — 生成物品设定图
app.post('/:id/generate-image', async (c) => {
  const id = parseId(c.req.param('id'))
  if (!id) return notFound(c, 'Invalid prop id')
  const [p] = db
    .select()
    .from(schema.propTemplates)
    .where(and(eq(schema.propTemplates.id, id), isNull(schema.propTemplates.deletedAt)))
    .all()
  if (!p) return notFound(c, '物品不存在')
  const body = await c.req.json().catch(() => ({}))
  // 屏幕留白类物品（ui_plate）使用留白图 prompt 构建器，供后期叠加文字
  const prompt = body.prompt || (p.category === UI_PLATE_CATEGORY
    ? buildUiPlateImagePrompt({ type: p.name, context: p.description || p.appearance })
    : buildPropImagePrompt(p))
  const configId = body.config_id ? Number(body.config_id) : undefined
  try {
    logTaskStart('PropsAPI', 'generate-image', { propId: id, name: p.name })
    const imageId = await generateImage({
      propId: id,
      dramaId: p.dramaId,
      prompt,
      negativePrompt: p.negativePrompt || undefined,
      configId,
    })
    return success(c, { imageGenerationId: imageId, prompt })
  } catch (err: any) {
    logTaskError('PropsAPI', 'generate-image', { propId: id, error: err.message })
    return badRequest(c, err.message || '生成失败')
  }
})

export default app
