import { Hono } from 'hono'
import { and, eq, isNull, like, desc } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, badRequest, notFound, created, now, parseParamId } from '../utils/response.js'
import { toSnakeCase, toSnakeCaseArray } from '../utils/transform.js'
import { logTaskError } from '../utils/task-logger.js'

const app = new Hono()

// GET /dramas - List dramas
app.get('/', async (c) => {
  try {
  const page = Number(c.req.query('page') || 1)
  const pageSize = Number(c.req.query('page_size') || 20)
  const status = c.req.query('status')
  const keyword = c.req.query('keyword')

  let query = db.select().from(schema.dramas).where(isNull(schema.dramas.deletedAt))

  const allRows = await query.orderBy(desc(schema.dramas.updatedAt))
  let filtered = allRows

  if (status) filtered = filtered.filter(d => d.status === status)
  if (keyword) filtered = filtered.filter(d => d.title.includes(keyword))

  const total = filtered.length
  const items = filtered.slice((page - 1) * pageSize, page * pageSize)

  // Attach episode/character/scene counts
  const enriched = await Promise.all(items.map(async (drama) => {
    const eps = await db.select().from(schema.episodes)
      .where(eq(schema.episodes.dramaId, drama.id))
    const chars = await db.select().from(schema.characters)
      .where(eq(schema.characters.dramaId, drama.id))
    const scns = await db.select().from(schema.scenes)
      .where(eq(schema.scenes.dramaId, drama.id))
    return {
      ...toSnakeCase(drama),
      tags: drama.tags ? JSON.parse(drama.tags) : [],
      total_episodes: eps.length,
      episodes: toSnakeCaseArray(eps),
      characters: toSnakeCaseArray(chars),
      scenes: toSnakeCaseArray(scns),
    }
  }))

  return success(c, {
    items: enriched,
    pagination: { page, page_size: pageSize, total, total_pages: Math.ceil(total / pageSize) },
  })
  } catch (err: any) { logTaskError('DramasAPI', 'list', { error: err.message }); return badRequest(c, err.message) }
})

// POST /dramas - Create drama
app.post('/', async (c) => {
  try {
  const body = await c.req.json()
  const ts = now()
  const res = db.insert(schema.dramas).values({
    title: body.title,
    description: body.description,
    genre: body.genre,
    style: body.style,
    tags: body.tags ? JSON.stringify(body.tags) : null,
    metadata: body.metadata,
    status: 'draft',
    createdAt: ts,
    updatedAt: ts,
  }).run()

  const [result] = db.select().from(schema.dramas)
    .where(eq(schema.dramas.id, Number(res.lastInsertRowid))).all()

  // Create default episodes
  const totalEpisodes = body.total_episodes || 1
  for (let i = 1; i <= totalEpisodes; i++) {
    db.insert(schema.episodes).values({
      dramaId: result.id,
      episodeNumber: i,
      title: `第${i}集`,
      status: 'draft',
      createdAt: ts,
      updatedAt: ts,
    }).run()
  }

  return created(c, toSnakeCase(result))
  } catch (err: any) { logTaskError('DramasAPI', 'create', { error: err.message }); return badRequest(c, err.message) }
})


// GET /dramas/stats — must be before /:id
app.get('/stats', async (c) => {
  try {
  const all = db.select().from(schema.dramas).where(isNull(schema.dramas.deletedAt)).all()
  const byStatus = Object.entries(
    all.reduce((acc, d) => {
      acc[d.status || 'draft'] = (acc[d.status || 'draft'] || 0) + 1
      return acc
    }, {} as Record<string, number>)
  ).map(([status, count]) => ({ status, count }))
  return success(c, { total: all.length, by_status: byStatus })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// GET /dramas/:id - Get drama detail
app.get('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid drama id')
  const [drama] = await db.select().from(schema.dramas).where(eq(schema.dramas.id, id))
  if (!drama) return notFound(c, '剧本不存在')

  const eps = await db.select().from(schema.episodes)
    .where(eq(schema.episodes.dramaId, id))
  const chars = await db.select().from(schema.characters)
    .where(eq(schema.characters.dramaId, id))
  const scns = await db.select().from(schema.scenes)
    .where(eq(schema.scenes.dramaId, id))
  const prps = await db.select().from(schema.props)
    .where(eq(schema.props.dramaId, id))

  return success(c, {
    ...toSnakeCase(drama),
    tags: drama.tags ? JSON.parse(drama.tags) : [],
    episodes: toSnakeCaseArray(eps),
    characters: toSnakeCaseArray(chars),
    scenes: toSnakeCaseArray(scns),
    props: toSnakeCaseArray(prps),
  })
  } catch (err: any) { logTaskError('DramasAPI', 'get', { error: err.message, id: c.req.param('id') }); return badRequest(c, err.message) }
})

// PUT /dramas/:id - Update drama
app.put('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid drama id')
  const [drama] = db.select().from(schema.dramas).where(and(eq(schema.dramas.id, id), isNull(schema.dramas.deletedAt))).all()
  if (!drama) return notFound(c, 'Drama not found')

  const body = await c.req.json()
  const updates: Record<string, any> = { updatedAt: now() }
  if (body.title !== undefined) updates.title = body.title
  if (body.description !== undefined) updates.description = body.description
  if (body.genre !== undefined) updates.genre = body.genre
  if (body.style !== undefined) updates.style = body.style
  if (body.status !== undefined) updates.status = body.status
  if (body.tags !== undefined) updates.tags = JSON.stringify(body.tags)
  if (body.metadata !== undefined) updates.metadata = body.metadata
  db.update(schema.dramas).set(updates).where(eq(schema.dramas.id, id)).run()
  return success(c)
  } catch (err: any) { logTaskError('DramasAPI', 'update', { error: err.message, id: c.req.param('id') }); return badRequest(c, err.message) }
})

// DELETE /dramas/:id - Soft delete
app.delete('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid drama id')
  const [drama] = db.select().from(schema.dramas).where(and(eq(schema.dramas.id, id), isNull(schema.dramas.deletedAt))).all()
  if (!drama) return notFound(c, 'Drama not found')
  db.update(schema.dramas).set({ deletedAt: now() }).where(eq(schema.dramas.id, id)).run()
  return success(c)
  } catch (err: any) { logTaskError('DramasAPI', 'delete', { error: err.message, id: c.req.param('id') }); return badRequest(c, err.message) }
})

// PUT /dramas/:id/characters - Save characters
app.put('/:id/characters', async (c) => {
  try {
  const dramaId = parseParamId(c)
  if (dramaId == null) return notFound(c, 'Invalid drama id')
  const body = await c.req.json()
  const chars = body.characters || []
  const ts = now()

  for (const char of chars) {
    if (char.id) {
      await db.update(schema.characters).set({ ...char, updatedAt: ts }).where(eq(schema.characters.id, char.id))
    } else {
      await db.insert(schema.characters).values({ ...char, dramaId, createdAt: ts, updatedAt: ts })
    }
  }
  return success(c)
  } catch (err: any) { logTaskError('DramasAPI', 'save-characters', { error: err.message }); return badRequest(c, err.message) }
})

// PUT /dramas/:id/episodes - Save episodes
app.put('/:id/episodes', async (c) => {
  try {
  const dramaId = parseParamId(c)
  if (dramaId == null) return notFound(c, 'Invalid drama id')
  const body = await c.req.json()
  const episodes = body.episodes || []
  const ts = now()

  for (const ep of episodes) {
    if (ep.id) {
      await db.update(schema.episodes).set({ ...ep, updatedAt: ts }).where(eq(schema.episodes.id, ep.id))
    } else {
      await db.insert(schema.episodes).values({
        ...ep,
        dramaId,
        episodeNumber: ep.episode_number || ep.episodeNumber || 1,
        title: ep.title || '未命名',
        createdAt: ts,
        updatedAt: ts,
      })
    }
  }
  return success(c)
  } catch (err: any) { logTaskError('DramasAPI', 'save-episodes', { error: err.message }); return badRequest(c, err.message) }
})

export default app
