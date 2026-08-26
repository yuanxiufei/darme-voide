import { Hono } from 'hono'
import { and, eq, isNull, like, desc, inArray } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, badRequest, notFound, conflict, created, now, parseParamId } from '../utils/response.js'
import { toSnakeCase, toSnakeCaseArray } from '../utils/transform.js'
import { logTaskError } from '../utils/task-logger.js'
import { ensureStyleId, ensureCostumeId } from '../services/bible-ids.js'

const app = new Hono()

// 白名单字段映射：snake_case/camelCase 双写 → drizzle camelCase 列名
// 返回 any：动态 key 无法被 drizzle 静态推断，交由运行时校验
function pickFields(src: any, camelKeys: string[]): any {
  const out: Record<string, any> = {}
  for (const key of camelKeys) {
    const snakeKey = key.replace(/[A-Z]/g, (m) => '_' + m.toLowerCase())
    if (snakeKey in src) out[key] = src[snakeKey]
    else if (key in src) out[key] = src[key]
  }
  return out
}

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

  // Attach episode/character/scene counts + multi-stage production progress
  const enriched = await Promise.all(items.map(async (drama) => {
    const eps = await db.select().from(schema.episodes)
      .where(and(eq(schema.episodes.dramaId, drama.id), isNull(schema.episodes.deletedAt)))
    const chars = await db.select().from(schema.characters)
      .where(and(eq(schema.characters.dramaId, drama.id), isNull(schema.characters.deletedAt)))
    const scns = await db.select().from(schema.scenes)
      .where(and(eq(schema.scenes.dramaId, drama.id), isNull(schema.scenes.deletedAt)))

    // 分镜进度统计：剧本 → 分镜 → 图片 → 视频 → 配音
    let sbs: any[] = []
    if (eps.length) {
      sbs = await db.select().from(schema.storyboards)
        .where(and(inArray(schema.storyboards.episodeId, eps.map(e => e.id)), isNull(schema.storyboards.deletedAt)))
    }
    const scripted = eps.filter(e => e.scriptContent).length
    const storyboarded = new Set(sbs.map(s => s.episodeId)).size
    const withImage = sbs.filter(s => s.composedImage || s.firstFrameImage).length
    const withVideo = sbs.filter(s => s.videoUrl || s.composedVideoUrl).length
    const withTts = sbs.filter(s => s.ttsAudioUrl).length

    return {
      ...toSnakeCase(drama),
      tags: drama.tags ? JSON.parse(drama.tags) : [],
      total_episodes: eps.length,
      episodes: toSnakeCaseArray(eps),
      characters: toSnakeCaseArray(chars),
      scenes: toSnakeCaseArray(scns),
      progress: {
        total_episodes: eps.length,
        scripted_episodes: scripted,
        storyboarded_episodes: storyboarded,
        storyboards: sbs.length,
        images: withImage,
        videos: withVideo,
        tts: withTts,
      },
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

  const dramaId = Number(res.lastInsertRowid)
  ensureStyleId(dramaId)
  const [result] = db.select().from(schema.dramas)
    .where(eq(schema.dramas.id, dramaId)).all()

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
  const [drama] = await db.select().from(schema.dramas).where(and(eq(schema.dramas.id, id), isNull(schema.dramas.deletedAt)))
  if (!drama) return notFound(c, '剧本不存在')

  const eps = await db.select().from(schema.episodes)
    .where(and(eq(schema.episodes.dramaId, id), isNull(schema.episodes.deletedAt)))
  const chars = await db.select().from(schema.characters)
    .where(and(eq(schema.characters.dramaId, id), isNull(schema.characters.deletedAt)))
  const scns = await db.select().from(schema.scenes)
    .where(and(eq(schema.scenes.dramaId, id), isNull(schema.scenes.deletedAt)))

  return success(c, {
    ...toSnakeCase(drama),
    tags: drama.tags ? JSON.parse(drama.tags) : [],
    episodes: toSnakeCaseArray(eps),
    characters: toSnakeCaseArray(chars),
    scenes: toSnakeCaseArray(scns),
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

  // 删除竞态保护：管线执行中的 episode 拒绝删除（避免已删剧本继续写数据/复活）
  const inFlight = db.select().from(schema.episodes)
    .where(and(eq(schema.episodes.dramaId, id), isNull(schema.episodes.deletedAt))).all()
    .filter(ep => ep.status?.startsWith('auto:') && ep.status !== 'auto:done' && ep.status !== 'auto:failed')
  if (inFlight.length) {
    return conflict(c, `剧本正在自动生成中（${inFlight.length} 集执行中），请等待完成后再删除`)
  }

  db.update(schema.dramas).set({ deletedAt: now() }).where(eq(schema.dramas.id, id)).run()
  return success(c)
  } catch (err: any) { logTaskError('DramasAPI', 'delete', { error: err.message, id: c.req.param('id') }); return badRequest(c, err.message) }
})

// PUT /dramas/:id/characters - Save characters（白名单字段映射，避免 no such column）
app.put('/:id/characters', async (c) => {
  try {
  const dramaId = parseParamId(c)
  if (dramaId == null) return notFound(c, 'Invalid drama id')
  const body = await c.req.json()
  const chars = body.characters || []
  const ts = now()

  const CHAR_KEYS = [
    'name', 'role', 'roleType', 'description', 'appearance', 'personality',
    'voiceStyle', 'imageUrl', 'referenceImages', 'seedValue', 'sortOrder',
    'localPath', 'voiceSampleUrl', 'voiceProvider', 'voiceSpeed', 'voiceEmotion',
    'voicePitch', 'clothing', 'weapons', 'customPrompt', 'voiceModel', 'costumeId',
  ]

  for (const char of chars) {
    const fields = pickFields(char, CHAR_KEYS)
    if (char.id) {
      await db.update(schema.characters).set({ ...fields, updatedAt: ts }).where(and(eq(schema.characters.id, char.id), isNull(schema.characters.deletedAt)))
      ensureCostumeId(char.id)
    } else {
      const ins = db.insert(schema.characters).values({ ...fields, dramaId, createdAt: ts, updatedAt: ts }).run()
      ensureCostumeId(Number(ins.lastInsertRowid))
    }
  }
  return success(c)
  } catch (err: any) { logTaskError('DramasAPI', 'save-characters', { error: err.message }); return badRequest(c, err.message) }
})

// PUT /dramas/:id/episodes - Save episodes（白名单字段映射，避免 no such column）
app.put('/:id/episodes', async (c) => {
  try {
  const dramaId = parseParamId(c)
  if (dramaId == null) return notFound(c, 'Invalid drama id')
  const body = await c.req.json()
  const episodes = body.episodes || []
  const ts = now()

  const EP_KEYS = [
    'episodeNumber', 'title', 'content', 'scriptContent', 'description',
    'duration', 'status', 'videoUrl', 'thumbnail',
    'imageConfigId', 'videoConfigId', 'audioConfigId',
  ]

  for (const ep of episodes) {
    const fields = pickFields(ep, EP_KEYS)
    if (ep.id) {
      await db.update(schema.episodes).set({ ...fields, updatedAt: ts }).where(eq(schema.episodes.id, ep.id))
    } else {
      await db.insert(schema.episodes).values({
        ...fields,
        dramaId,
        episodeNumber: fields.episodeNumber || 1,
        title: fields.title || '未命名',
        createdAt: ts,
        updatedAt: ts,
      })
    }
  }
  return success(c)
  } catch (err: any) { logTaskError('DramasAPI', 'save-episodes', { error: err.message }); return badRequest(c, err.message) }
})

// GET /dramas/:id/prompts — 聚合该剧下所有提示词（角色/场景/分镜图片+视频），供统一提示词管理视图使用
app.get('/:id/prompts', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid drama id')
  try {
    const [drama] = db.select().from(schema.dramas).where(and(eq(schema.dramas.id, id), isNull(schema.dramas.deletedAt))).all()
    if (!drama) return notFound(c, 'Drama not found')

    // 角色（软删除过滤，按 sortOrder 排序）
    const characters = db.select().from(schema.characters)
      .where(and(eq(schema.characters.dramaId, id), isNull(schema.characters.deletedAt)))
      .orderBy(schema.characters.sortOrder).all()

    // 场景（软删除过滤）
    const scenes = db.select().from(schema.scenes)
      .where(and(eq(schema.scenes.dramaId, id), isNull(schema.scenes.deletedAt)))
      .orderBy(schema.scenes.id).all()

    // 分镜（通过 episode 关联到 drama）
    const episodes = db.select().from(schema.episodes)
      .where(and(eq(schema.episodes.dramaId, id), isNull(schema.episodes.deletedAt)))
      .orderBy(schema.episodes.episodeNumber).all()
    const epIds = episodes.map(e => e.id)

    let storyboards: any[] = []
    if (epIds.length) {
      storyboards = db.select().from(schema.storyboards)
        .where(inArray(schema.storyboards.episodeId, epIds))
        .orderBy(schema.storyboards.episodeId, schema.storyboards.storyboardNumber).all()
    }

    return success(c, {
      characters: characters.map(ch => ({
        id: ch.id, name: ch.name, role: ch.role,
        customPrompt: ch.customPrompt, imageUrl: ch.imageUrl,
      })),
      scenes: scenes.map(sc => ({
        id: sc.id, location: sc.location, time: sc.time,
        prompt: sc.prompt, customPrompt: sc.customPrompt, imageUrl: sc.imageUrl,
      })),
      episodes: episodes.map(ep => ({ id: ep.id, episodeNumber: ep.episodeNumber, title: ep.title })),
      storyboards: storyboards.map(sb => ({
        id: sb.id, episodeId: sb.episodeId, storyboardNumber: sb.storyboardNumber, title: sb.title,
        imagePrompt: sb.imagePrompt, videoPrompt: sb.videoPrompt,
        customImagePrompt: sb.customImagePrompt, customVideoPrompt: sb.customVideoPrompt,
        status: sb.status,
      })),
    })
  } catch (err: any) {
    logTaskError('DramasAPI', 'prompts', { error: err.message, id })
    return badRequest(c, err.message)
  }
})

export default app
