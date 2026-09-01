import { Hono } from 'hono'
import { success, badRequest } from '../utils/response.js'
import { getUsageSummary, getEpisodeCostBoard } from '../services/usage-tracking.js'
import { estimatePendingCosts } from '../services/estimate-service.js'

const app = new Hono()

// GET /usage/summary?drama_id=&episode_id=&limit=
// 用量与成本汇总：总量 / 总成本 / 按服务类型 / 按提供商 / 按天 / 最近明细
app.get('/summary', async (c) => {
  try {
    const q = c.req.query()
    const dramaId = q.drama_id ? Number(q.drama_id) : null
    const episodeId = q.episode_id ? Number(q.episode_id) : null
    if (q.drama_id && !Number.isFinite(dramaId)) return badRequest(c, 'invalid drama_id')
    if (q.episode_id && !Number.isFinite(episodeId)) return badRequest(c, 'invalid episode_id')

    const limit = q.limit ? Number(q.limit) : 200
    const summary = getUsageSummary({ dramaId, episodeId, limit })
    return success(c, summary)
  } catch (err: any) {
    return badRequest(c, err?.message || 'usage summary failed')
  }
})

// GET /usage/board?drama_id=
// 多集成本看板：按剧聚合每集成本（总成本 / 总调用 / 重拍成本 / 按服务类型拆分）
app.get('/board', (c) => {
  try {
    const q = c.req.query()
    const dramaId = q.drama_id ? Number(q.drama_id) : null
    if (dramaId == null || !Number.isFinite(dramaId)) return badRequest(c, 'invalid drama_id')
    return success(c, getEpisodeCostBoard(dramaId))
  } catch (err: any) {
    return badRequest(c, err?.message || 'cost board failed')
  }
})

// GET /usage/estimate?drama_id=&episode_id=&storyboard_ids=
// 生成前费用预估（对齐 ArcReel 生成前预估 + 生成后核算双环）：
// 按当前激活的 provider/model 单价 × 待生成工作量（视频秒数 / 图片张数 / TTS 字符数）估算。
app.get('/estimate', (c) => {
  try {
    const q = c.req.query()
    const dramaId = q.drama_id ? Number(q.drama_id) : null
    if (dramaId == null || !Number.isFinite(dramaId)) return badRequest(c, 'invalid drama_id')

    const episodeId = q.episode_id ? Number(q.episode_id) : undefined
    if (q.episode_id && !Number.isFinite(episodeId)) return badRequest(c, 'invalid episode_id')

    const storyboardIds = q.storyboard_ids
      ? q.storyboard_ids.split(',').map(Number).filter(Number.isFinite)
      : undefined

    const result = estimatePendingCosts(dramaId, { episodeId, storyboardIds })
    return success(c, result)
  } catch (err: any) {
    return badRequest(c, err?.message || 'estimate failed')
  }
})

export default app
