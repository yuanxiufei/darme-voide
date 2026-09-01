/**
 * 用量统计与成本估算
 *
 * 对齐参考项目 ArcReel usage_repo：
 *  - 每次模型调用记一条账（provider/model/归属维度/用量/估算成本/重试次数）
 *  - 本地模型 is_local=1 不计费（cost_amount 为 null）
 *  - 汇总口径：按项目/集/服务类型/提供商/按天
 *  - 重拍与模型 fallback 会记 retry_count，可回答「审片重拍多烧了多少钱」
 */
import { db, schema } from '../db/index.js'
import { eq } from 'drizzle-orm'
import { now } from '../utils/response.js'
import { logTaskWarn } from '../utils/task-logger.js'
import { estimateCost } from './cost-catalog.js'

export type UsageServiceType = 'image' | 'video' | 'audio' | 'text'
export type UsageStatus = 'submitted' | 'completed' | 'failed'

export interface RecordUsageInput {
  serviceType: UsageServiceType
  provider: string
  model: string
  dramaId?: number | null
  episodeId?: number | null
  storyboardId?: number | null
  sceneId?: number | null
  characterId?: number | null
  imageGenerationId?: number | null
  videoGenerationId?: number | null
  /** 计费单位数：图片张数 / 视频秒数 / 音频字符数 / 千 token 数 */
  units?: number | null
  isLocal?: boolean
  status?: UsageStatus
  /** 同一任务第几次尝试（0=首次；模型 fallback / QC 重跑递增） */
  retryCount?: number
  /** ai_service_configs.settings 解析对象，可含 pricing 覆盖 */
  settings?: Record<string, any> | null
  meta?: Record<string, any>
}

/** 记录一次模型调用（同步写入，不抛错，失败仅告警不影响主流程） */
export function recordUsage(input: RecordUsageInput): number | null {
  try {
    const units = input.units ?? null
    const costAmount = input.isLocal
      ? null
      : estimateCost(input.serviceType, input.provider, input.model, units, input.settings)
    const res = db.insert(schema.apiUsage).values({
      serviceType: input.serviceType,
      provider: input.provider,
      model: input.model,
      dramaId: input.dramaId ?? null,
      episodeId: input.episodeId ?? null,
      storyboardId: input.storyboardId ?? null,
      imageGenerationId: input.imageGenerationId ?? null,
      videoGenerationId: input.videoGenerationId ?? null,
      units,
      costAmount,
      isLocal: input.isLocal ?? false,
      status: input.status ?? 'submitted',
      retryCount: input.retryCount ?? 0,
      meta: input.meta ? JSON.stringify(input.meta) : null,
      createdAt: now(),
    }).run()
    return Number(res.lastInsertRowid)
  } catch (err: any) {
    logTaskWarn('Usage', 'record-failed', {
      serviceType: input.serviceType,
      provider: input.provider,
      model: input.model,
      error: err?.message || String(err),
    })
    return null
  }
}

/** 更新一条用量记录的状态（submitted → completed / failed） */
export function updateUsageStatus(id: number | null, status: UsageStatus, meta?: Record<string, any>): void {
  if (!id) return
  try {
    const patch: Record<string, any> = { status }
    if (meta) patch.meta = JSON.stringify(meta)
    db.update(schema.apiUsage).set(patch).where(eq(schema.apiUsage.id, id)).run()
  } catch (err: any) {
    logTaskWarn('Usage', 'update-failed', { id, error: err?.message || String(err) })
  }
}

export interface UsageSummary {
  totalCount: number
  totalCost: number
  checkedCost: boolean // false=所有记录都无 cost（如只有本地/未知单价）
  byService: Array<{ service_type: string; count: number; cost: number }>
  byProvider: Array<{ provider: string; count: number; cost: number }>
  byDay: Array<{ date: string; count: number; cost: number }>
  records: Array<{
    id: number
    service_type: string
    provider: string
    model: string
    drama_id: number | null
    episode_id: number | null
    storyboard_id: number | null
    units: number | null
    cost_amount: number | null
    is_local: boolean
    status: string
    retry_count: number
    created_at: string
  }>
}

export interface UsageFilter {
  dramaId?: number | null
  episodeId?: number | null
  limit?: number
}

/** 汇总：按项目/集过滤，返回总数、总成本、分类口径与明细（默认最近 200 条） */
export function getUsageSummary(filter: UsageFilter = {}): UsageSummary {
  const q = db.select().from(schema.apiUsage)
  const rows = q.all().filter((r) => {
    if (filter.dramaId != null && r.dramaId !== filter.dramaId) return false
    if (filter.episodeId != null && r.episodeId !== filter.episodeId) return false
    return true
  })

  const byService = new Map<string, { count: number; cost: number }>()
  const byProvider = new Map<string, { count: number; cost: number }>()
  const byDay = new Map<string, { count: number; cost: number }>()
  let totalCost = 0
  let costed = 0

  for (const r of rows) {
    const cost = r.costAmount ?? 0
    totalCost += cost
    if (r.costAmount != null) costed++

    const svc = byService.get(r.serviceType) || { count: 0, cost: 0 }
    svc.count++; svc.cost += cost
    byService.set(r.serviceType, svc)

    const prov = byProvider.get(r.provider) || { count: 0, cost: 0 }
    prov.count++; prov.cost += cost
    byProvider.set(r.provider, prov)

    const date = (r.createdAt || '').slice(0, 10)
    const day = byDay.get(date) || { count: 0, cost: 0 }
    day.count++; day.cost += cost
    byDay.set(date, day)
  }

  const sortDesc = (a: [string, { count: number; cost: number }], b: [string, { count: number; cost: number }]) =>
    b[1].count - a[1].count

  const limit = Math.max(1, filter.limit ?? 200)
  const recent = [...rows].sort((a, b) => b.id - a.id).slice(0, limit)

  return {
    totalCount: rows.length,
    totalCost: Math.round(totalCost * 10000) / 10000,
    checkedCost: costed > 0,
    byService: [...byService.entries()].sort(sortDesc).map(([service_type, v]) => ({ service_type, ...v })),
    byProvider: [...byProvider.entries()].sort(sortDesc).map(([provider, v]) => ({ provider, ...v })),
    byDay: [...byDay.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([date, v]) => ({ date, ...v })),
    records: recent.map((r) => ({
      id: r.id,
      service_type: r.serviceType,
      provider: r.provider,
      model: r.model,
      drama_id: r.dramaId,
      episode_id: r.episodeId,
      storyboard_id: r.storyboardId,
      units: r.units,
      cost_amount: r.costAmount,
      is_local: !!r.isLocal,
      status: r.status ?? '',
      retry_count: r.retryCount ?? 0,
      created_at: r.createdAt,
    })),
  }
}

export interface EpisodeCostBoard {
  drama_id: number
  total_cost: number
  total_calls: number
  retry_cost: number // 重拍/fallback 带来的额外成本（retry_count >= 1 的调用成本）
  episodes: Array<{
    episode_id: number
    episode_number: number
    title: string | null
    total_cost: number
    total_calls: number
    retry_cost: number
    by_service: Array<{ service_type: string; count: number; cost: number }>
  }>
}

/**
 * 多集成本看板（对齐参考项目多集成本汇总口径）
 * 按剧聚合每集成本：总成本 / 总调用 / 重拍成本 / 每集按服务类型拆分。
 * 重拍成本 = 同一任务 retry_count >= 1 的调用成本，可回答「重拍烧了多少」。
 */
export function getEpisodeCostBoard(dramaId: number): EpisodeCostBoard {
  const usageRows = db
    .select()
    .from(schema.apiUsage)
    .all()
    .filter(r => r.dramaId === dramaId && r.episodeId != null)

  const episodes = db
    .select({ id: schema.episodes.id, episodeNumber: schema.episodes.episodeNumber, title: schema.episodes.title })
    .from(schema.episodes)
    .where(eq(schema.episodes.dramaId, dramaId))
    .orderBy(schema.episodes.episodeNumber)
    .all()

  const map = new Map<number, EpisodeCostBoard['episodes'][number]>()
  for (const ep of episodes) {
    map.set(ep.id, {
      episode_id: ep.id,
      episode_number: ep.episodeNumber,
      title: ep.title,
      total_cost: 0,
      total_calls: 0,
      retry_cost: 0,
      by_service: [],
    })
  }

  let dramaTotal = 0
  let dramaCalls = 0
  let dramaRetry = 0

  for (const r of usageRows) {
    const epRow = map.get(r.episodeId ?? -1)
    if (!epRow) continue
    const cost = r.costAmount ?? 0
    const isRetry = (r.retryCount ?? 0) >= 1

    epRow.total_cost += cost
    epRow.total_calls += 1
    if (isRetry) epRow.retry_cost += cost

    dramaTotal += cost
    dramaCalls += 1
    if (isRetry) dramaRetry += cost

    const svcIdx = epRow.by_service.findIndex(s => s.service_type === r.serviceType)
    if (svcIdx >= 0) {
      epRow.by_service[svcIdx].count += 1
      epRow.by_service[svcIdx].cost += cost
    } else {
      epRow.by_service.push({ service_type: r.serviceType, count: 1, cost })
    }
  }

  const round = (n: number) => Math.round(n * 10000) / 10000
  for (const epRow of map.values()) {
    epRow.total_cost = round(epRow.total_cost)
    epRow.retry_cost = round(epRow.retry_cost)
    epRow.by_service.sort((a, b) => b.count - a.count)
  }

  return {
    drama_id: dramaId,
    total_cost: round(dramaTotal),
    total_calls: dramaCalls,
    retry_cost: round(dramaRetry),
    episodes: [...map.values()],
  }
}
