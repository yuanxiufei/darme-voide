/**
 * 视觉图谱 API — 对齐参考项目「视觉图谱：景别/构图/运镜/灯光，由 Adviser 图谱驱动提示词」
 *
 * GET /visual-graph                图谱全量（四类知识节点，供前端图谱库展示）
 * GET /visual-graph/resolve?text=  中文视觉术语 → 英文电影术语（供 prompt 构造调试）
 * GET /visual-graph/guidance?drama_id=  按剧类型/风格生成拆镜视觉图谱引导文本
 */
import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { badRequest, notFound, parseParamId } from '../utils/response.js'
import { logTaskError } from '../utils/task-logger.js'
import {
  buildVisualGraphGuidance,
  getVisualGraph,
  listVisualTerms,
  resolveVisualTerm,
  type VisualCategory,
} from '../shared/visual-graph.js'

const app = new Hono()

const CATEGORIES: VisualCategory[] = ['shot_size', 'composition', 'movement', 'lighting']

app.get('/', (c) => {
  const categoryRaw = c.req.query('category')
  const category = CATEGORIES.includes(categoryRaw as VisualCategory) ? (categoryRaw as VisualCategory) : undefined
  c.header('Content-Type', 'application/json; charset=utf-8')
  return c.json({ graph: getVisualGraph(category), categories: CATEGORIES })
})

app.get('/resolve', (c) => {
  const text = c.req.query('text')
  if (!text) return badRequest(c, 'query param "text" is required')
  c.header('Content-Type', 'application/json; charset=utf-8')
  return c.json({ zh: text, en: resolveVisualTerm(text) })
})

app.get('/terms', (c) => {
  const categoryRaw = c.req.query('category')
  if (!CATEGORIES.includes(categoryRaw as VisualCategory)) {
    return badRequest(c, `category must be one of: ${CATEGORIES.join(' | ')}`)
  }
  c.header('Content-Type', 'application/json; charset=utf-8')
  return c.json({ category: categoryRaw, terms: listVisualTerms(categoryRaw as VisualCategory) })
})

app.get('/guidance', (c) => {
  const dramaIdRaw = c.req.query('drama_id')
  if (dramaIdRaw && !Number.isNaN(Number(dramaIdRaw))) {
    const dramaId = Number(dramaIdRaw)
    const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
    if (!drama) return notFound(c, 'Drama not found')
    c.header('Content-Type', 'text/plain; charset=utf-8')
    return c.body(buildVisualGraphGuidance(drama.genre, drama.style))
  }
  // 无 drama_id 时使用全局默认引导（无类型匹配，仅通用规则）
  c.header('Content-Type', 'text/plain; charset=utf-8')
  return c.body(buildVisualGraphGuidance(null, null))
})

export default app

// 保留 parseParamId 引用避免未使用告警（后续扩展按 id 查询时使用）
void parseParamId
