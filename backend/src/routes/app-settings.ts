import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, badRequest, now } from '../utils/response.js'

const app = new Hono()

/** 画风 key 白名单（与前端创建/编辑剧集时的 6 选一致） */
export const ART_STYLE_KEYS = ['realistic', 'anime', 'ghibli', 'cinematic', 'comic', 'watercolor'] as const

function allSettings(): Record<string, string | null> {
  const rows = db.select().from(schema.appSettings).all()
  const data: Record<string, string | null> = {}
  for (const r of rows) data[r.key] = r.value
  return data
}

// GET /app-settings — 返回全部应用设置（目前含 art_style 全局默认画风）
app.get('/', (c) => success(c, allSettings()))

// PUT /app-settings — 更新应用设置（支持 art_style：6 种画风之一或空串=未设置）
app.put('/', async (c) => {
  const body = await c.req.json().catch(() => ({}))
  if (body.art_style !== undefined) {
    const v = String(body.art_style || '')
    if (v && !(ART_STYLE_KEYS as readonly string[]).includes(v)) {
      return badRequest(c, `art_style 必须是 ${ART_STYLE_KEYS.join('/')} 之一或留空`)
    }
    const ts = now()
    const existing = db.select().from(schema.appSettings).where(eq(schema.appSettings.key, 'art_style')).all()
    if (existing.length) {
      db.update(schema.appSettings).set({ value: v || null, updatedAt: ts })
        .where(eq(schema.appSettings.key, 'art_style')).run()
    } else {
      db.insert(schema.appSettings).values({ key: 'art_style', value: v || null, updatedAt: ts }).run()
    }
  }
  return success(c, allSettings())
})

export default app
