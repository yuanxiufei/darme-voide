import { Hono } from 'hono'
import { success, badRequest, notFound } from '../utils/response.js'
import {
  listStyleProfiles, getStyleProfile, createStyleProfile, updateStyleProfile,
  deleteStyleProfile, activateStyleProfile, distillStyleProfile, applyDistillResult,
} from '../services/style-profiles.js'
import { logTaskError } from '../utils/task-logger.js'

const app = new Hono()

// GET /style-profiles?drama_id=1 — 列表（激活的在前）
app.get('/', async (c) => {
  try {
    const dramaId = c.req.query('drama_id')
    const profiles = listStyleProfiles(dramaId ? Number(dramaId) : null)
    return success(c, { profiles })
  } catch (err: any) {
    return badRequest(c, err?.message || 'list style profiles failed')
  }
})

// GET /style-profiles/:id — 详情
app.get('/:id', async (c) => {
  const p = getStyleProfile(Number(c.req.param('id')))
  if (!p) return notFound(c, 'Profile not found')
  return success(c, { profile: p })
})

// POST /style-profiles — 创建
app.post('/', async (c) => {
  try {
    const body = await c.req.json()
    const id = createStyleProfile({
      dramaId: body.drama_id ?? null,
      name: body.name,
      description: body.description ?? null,
      source: body.source ?? null,
      storytelling: body.storytelling ?? null,
      shotPatterns: body.shot_patterns ?? null,
      audioCaptions: body.audio_captions ?? null,
      qcRules: body.qc_rules ?? null,
      preferences: body.preferences ?? null,
    })
    if (id == null) return badRequest(c, 'create failed')
    return success(c, { id, profile: getStyleProfile(id) })
  } catch (err: any) {
    return badRequest(c, err?.message || 'create style profile failed')
  }
})

// PUT /style-profiles/:id — 更新
app.put('/:id', async (c) => {
  try {
    const id = Number(c.req.param('id'))
    const body = await c.req.json()
    const ok = updateStyleProfile(id, {
      name: body.name,
      description: body.description === undefined ? undefined : body.description,
      source: body.source === undefined ? undefined : body.source,
      storytelling: body.storytelling === undefined ? undefined : body.storytelling,
      shotPatterns: body.shot_patterns === undefined ? undefined : body.shot_patterns,
      audioCaptions: body.audio_captions === undefined ? undefined : body.audio_captions,
      qcRules: body.qc_rules === undefined ? undefined : body.qc_rules,
      preferences: body.preferences === undefined ? undefined : body.preferences,
    })
    if (!ok) return notFound(c, 'Profile not found')
    return success(c, { profile: getStyleProfile(id) })
  } catch (err: any) {
    return badRequest(c, err?.message || 'update style profile failed')
  }
})

// POST /style-profiles/:id/activate — 激活（同 drama 内唯一）
app.post('/:id/activate', async (c) => {
  const p = activateStyleProfile(Number(c.req.param('id')))
  if (!p) return notFound(c, 'Profile not found')
  return success(c, { profile: p })
})

// POST /style-profiles/:id/distill — 用 LLM 分析参考素材，提炼 house style（不落库，返回待确认）
app.post('/:id/distill', async (c) => {
  try {
    const id = Number(c.req.param('id'))
    const result = await distillStyleProfile(id)
    if (!result.ok) return badRequest(c, result.error || 'distill failed')
    return success(c, { result: result.result })
  } catch (err: any) {
    logTaskError('StyleProfilesAPI', 'distill', { error: err?.message || String(err) })
    return badRequest(c, err?.message || 'distill failed')
  }
})

// POST /style-profiles/:id/apply — 把确认后的提炼结果写入 Profile
app.post('/:id/apply', async (c) => {
  try {
    const id = Number(c.req.param('id'))
    const body = await c.req.json()
    const ok = applyDistillResult(id, {
      storytelling: body.storytelling || {},
      shot_patterns: body.shot_patterns || {},
      audio_captions: body.audio_captions || {},
      qc_rules: body.qc_rules || {},
      facts: body.facts || [],
      inferences: body.inferences || [],
      preferences: body.preferences || [],
    })
    if (!ok) return notFound(c, 'Profile not found')
    return success(c, { profile: getStyleProfile(id) })
  } catch (err: any) {
    return badRequest(c, err?.message || 'apply distill result failed')
  }
})

// DELETE /style-profiles/:id — 软删除
app.delete('/:id', async (c) => {
  const ok = deleteStyleProfile(Number(c.req.param('id')))
  if (!ok) return notFound(c, 'Profile not found')
  return success(c, { deleted: true })
})

export default app
