/**
 * per-shot take 预算（对齐参考项目 per-shot take budget）
 *
 * 每个分镜（shot）允许的生成尝试次数由 take_budget 控制（默认 3）。
 * 每次提交图片/视频生成（无论成败）消耗一次 take；超预算后阻断继续生成，
 * 强制用户重新拆解分镜（此时重置预算）或显式 force 覆盖。
 *
 * 价值：防止失败重试/反复调参导致同一分镜无限消耗算力，将资源收敛到
 * 「预算内重试 → 预算耗尽 → 重新拆解」的收敛循环。
 */
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { now } from '../utils/response.js'

export const DEFAULT_TAKE_BUDGET = 3

/** 取分镜 take 状态 */
export function getTakeStatus(storyboardId: number): {
  storyboard_id: number
  take_count: number
  take_budget: number
  remaining: number
  exhausted: boolean
} | null {
  const [sb] = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.id, storyboardId)).all()
  if (!sb) return null
  const budget = sb.takeBudget ?? DEFAULT_TAKE_BUDGET
  const count = sb.takeCount ?? 0
  return {
    storyboard_id: storyboardId,
    take_count: count,
    take_budget: budget,
    remaining: Math.max(0, budget - count),
    exhausted: count >= budget,
  }
}

/** 消耗一次 take（媒体生成提交成功后调用） */
export function consumeTake(storyboardId: number | undefined | null): void {
  if (!storyboardId) return
  const [sb] = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.id, storyboardId)).all()
  if (!sb) return
  db.update(schema.storyboards)
    .set({ takeCount: (sb.takeCount ?? 0) + 1, updatedAt: now() })
    .where(eq(schema.storyboards.id, storyboardId)).run()
}

/**
 * 门禁检查：预算是否耗尽。返回 { allowed, reason, take_status }。
 * 未绑定 storyboard 或预算耗尽时 force=true 可放行。
 */
export function checkTakeBudget(storyboardId: number | null | undefined, force?: boolean): {
  allowed: boolean
  reason?: string
  status?: NonNullable<ReturnType<typeof getTakeStatus>>
} {
  if (!storyboardId) return { allowed: true }
  const status = getTakeStatus(storyboardId)
  if (!status) return { allowed: true }
  if (status.exhausted && !force) {
    return {
      allowed: false,
      reason: `该分镜 take 预算已耗尽（${status.take_count}/${status.take_budget}）。请重新拆解分镜以重置预算，或传 force=true 强制继续。`,
      status,
    }
  }
  return { allowed: true, status }
}

/** 重置分镜 take 预算（重新拆解分镜后调用） */
export function resetTakeBudget(storyboardId: number): void {
  db.update(schema.storyboards)
    .set({ takeCount: 0, updatedAt: now() })
    .where(eq(schema.storyboards.id, storyboardId)).run()
}

/** 重置某集所有分镜 take 预算 */
export function resetTakeBudgetForEpisode(episodeId: number): void {
  const sbs = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, episodeId)).all()
  for (const sb of sbs) {
    resetTakeBudget(sb.id)
  }
}
