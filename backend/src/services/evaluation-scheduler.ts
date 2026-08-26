/**
 * 评测→优化 无人值守定时调度器（对齐 PenguinHarness 自进化闭环）
 *
 * - 用原生 setTimeout 实现「每日 HH:MM 定时」，无外部 cron 依赖
 * - 串行跑全部基准 case（避免并发 LLM 调用引发限流 / 费用激增）
 * - 防重入：上一轮未跑完则跳过本次触发
 * - 失败隔离：单个 case 失败不影响其余
 * - 仅在 evaluation.auto_optimize.enabled 时启动（默认关闭，避免意外烧钱）
 */
import { config } from '../config.js'
import { AGENT_BY_KIND, listBenchmarkCases, loadCaseById } from '../evaluation/catalog.js'
import { optimizeAgentPrompt } from '../evaluation/optimizer.js'
import { logTask, logTaskSuccess, logTaskWarn, logTaskError } from '../utils/task-logger.js'

export interface RunResult {
  caseId: string
  agentType: string
  ok: boolean
  error?: string
  persisted?: { version: number; score: number; name: string } | null
}

export interface SchedulerState {
  enabled: boolean
  running: boolean
  lastRunAt: string | null
  nextRunAt: string | null
  lastResults: RunResult[]
  schedule: { hour: number; minute: number; iterations: number; runOnStartup: boolean }
}

const schedule = config.evaluation.autoOptimize

const state: SchedulerState = {
  enabled: schedule.enabled,
  running: false,
  lastRunAt: null,
  nextRunAt: null,
  lastResults: [],
  schedule: {
    hour: schedule.hour,
    minute: schedule.minute,
    iterations: schedule.iterations,
    runOnStartup: schedule.runOnStartup,
  },
}

let timer: ReturnType<typeof setTimeout> | null = null

/** 计算距下一个 HH:MM 的毫秒数（已过则顺延到次日） */
function nextRunDelay(hour: number, minute: number): number {
  const now = new Date()
  const next = new Date(now)
  next.setHours(hour, minute, 0, 0)
  if (next.getTime() <= now.getTime()) next.setDate(next.getDate() + 1)
  return next.getTime() - now.getTime()
}

/** 跑一轮全量优化：串行遍历全部基准 case */
export async function runEvaluationCycle(): Promise<RunResult[]> {
  if (state.running) {
    logTaskWarn('EvalScheduler', '上一轮仍在运行，跳过本次触发')
    return state.lastResults
  }

  state.running = true
  state.lastRunAt = new Date().toISOString()
  const results: RunResult[] = []
  const cases = listBenchmarkCases()
  logTask('EvalScheduler', `START 全量优化（${cases.length} 个 case）`)

  for (const c of cases) {
    try {
      const caseDef = loadCaseById(c.id)
      if (!caseDef) throw new Error(`case 加载失败：${c.id}`)
      const agentType = AGENT_BY_KIND[caseDef.kind]
      if (!agentType) throw new Error(`未知 case kind：${caseDef.kind}`)

      const history = await optimizeAgentPrompt(agentType, caseDef, {
        iterations: state.schedule.iterations,
        autoPersist: true,
      })
      results.push({
        caseId: c.id,
        agentType,
        ok: true,
        persisted: history.persisted ?? null,
      })
      logTask('EvalScheduler', `DONE ${c.id}`, {
        agentType,
        bestScore: history.best.score,
        persisted: history.persisted ? 'yes' : 'no',
      })
    } catch (err: any) {
      logTaskError('EvalScheduler', `case ${c.id} 优化失败`, { error: err?.message || String(err) })
      results.push({ caseId: c.id, agentType: c.agentType, ok: false, error: err?.message || String(err) })
    }
  }

  state.lastResults = results
  state.running = false
  const okCount = results.filter(r => r.ok).length
  logTaskSuccess('EvalScheduler', `全量优化完成 ${okCount}/${results.length}`)
  return results
}

/** 调度下一次运行 */
function arm() {
  if (!state.enabled) return
  const delay = nextRunDelay(state.schedule.hour, state.schedule.minute)
  state.nextRunAt = new Date(Date.now() + delay).toISOString()
  logTask('EvalScheduler', `下次优化定于 ${state.nextRunAt}`)
  timer = setTimeout(async () => {
    await runEvaluationCycle()
    arm()
  }, delay)
}

/** 查询调度器状态（供 /evaluation/scheduler 展示） */
export function getSchedulerState(): SchedulerState {
  return state
}

/**
 * 启动调度器（index.ts 调用一次）。
 * - enabled 时按每日 HH:MM 定时
 * - runOnStartup 时启动后立即跑一轮（与定时互不冲突，靠防重入保证）
 */
export function startEvaluationScheduler(): void {
  if (!state.enabled) {
    logTaskWarn('EvalScheduler', '未启用（evaluation.auto_optimize.enabled=false），跳过')
    return
  }
  logTask('EvalScheduler', `启用每日定时优化（${state.schedule.hour}:${String(state.schedule.minute).padStart(2, '0')}，iterations=${state.schedule.iterations}）`)
  arm()
  if (state.schedule.runOnStartup) {
    logTask('EvalScheduler', 'run_on_startup=true，启动即跑一轮')
    runEvaluationCycle()
  }
}
