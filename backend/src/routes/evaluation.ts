import { Hono } from 'hono'
import { success, badRequest, notFound } from '../utils/response.js'
import { logTaskError } from '../utils/task-logger.js'
import { getDefaultInstructions } from '../agents/index.js'
import { AGENT_BY_KIND, listBenchmarkCases, loadCaseById } from '../evaluation/catalog.js'
import { evaluateCase } from '../evaluation/evaluator.js'
import { optimizeAgentPrompt } from '../evaluation/optimizer.js'
import { getSchedulerState, runEvaluationCycle } from '../services/evaluation-scheduler.js'

const app = new Hono()

// GET /evaluation/cases —— 列出可用基准 case（id / kind / agentType）
app.get('/cases', (c) => {
  return success(c, listBenchmarkCases())
})

// POST /evaluation/evaluate/:caseId —— 用 Reference 提示词评测指定 case
app.post('/evaluate/:caseId', async (c) => {
  try {
    const caseId = c.req.param('caseId')
    const caseDef = loadCaseById(caseId)
    if (!caseDef) return notFound(c, `未知基准 case：${caseId}`)
    const agentType = AGENT_BY_KIND[caseDef.kind]
    if (!agentType) return badRequest(c, `未知 case kind：${caseDef.kind}`)

    const report = await evaluateCase(caseDef, getDefaultInstructions(agentType))
    return success(c, report)
  } catch (err: any) {
    logTaskError('EvaluationAPI', 'evaluate', { error: err.message })
    return badRequest(c, err.message)
  }
})

// POST /evaluation/optimize/:caseId —— 状态机优化（best 严格高于 Reference 时自动落库）
// body: { iterations?: number, auto_persist?: boolean }
app.post('/optimize/:caseId', async (c) => {
  try {
    const caseId = c.req.param('caseId')
    const caseDef = loadCaseById(caseId)
    if (!caseDef) return notFound(c, `未知基准 case：${caseId}`)
    const agentType = AGENT_BY_KIND[caseDef.kind]
    if (!agentType) return badRequest(c, `未知 case kind：${caseDef.kind}`)

    const body = await c.req.json().catch(() => ({}))
    const iterations = Number.isInteger(body.iterations) && body.iterations > 0 ? body.iterations : 3
    const autoPersist = body.auto_persist !== false

    const history = await optimizeAgentPrompt(agentType, caseDef, { iterations, autoPersist })
    return success(c, history)
  } catch (err: any) {
    logTaskError('EvaluationAPI', 'optimize', { error: err.message })
    return badRequest(c, err.message)
  }
})

// GET /evaluation/scheduler —— 查询定时调度器状态（enabled/running/nextRunAt/lastResults）
app.get('/scheduler', (c) => {
  return success(c, getSchedulerState())
})

// POST /evaluation/run —— 手动触发一轮全量优化（与定时共用同一执行入口，防重入）
app.post('/run', async (c) => {
  try {
    const results = await runEvaluationCycle()
    return success(c, results)
  } catch (err: any) {
    logTaskError('EvaluationAPI', 'run', { error: err.message })
    return badRequest(c, err.message)
  }
})

export default app
