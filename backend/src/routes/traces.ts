/**
 * Trace 回放路由 — 对齐 PenguinHarness 第 9 章「Trace 写入与回放」。
 * 暴露 Agent 执行轨迹的查询与回放能力，供前端/排查崩溃前后的执行过程。
 */
import { Hono } from 'hono'
import { success, badRequest } from '../utils/response.js'
import { listTraces, readTrace } from '../utils/trace-store.js'

const app = new Hono()

// GET /traces/stats — 聚合所有 Agent 调用的 token 用量（按 scope 分组）
app.get('/stats', async (c) => {
  const metas = await listTraces()
  const byScope = new Map<string, { runs: number; inputTokens: number; outputTokens: number; totalTokens: number }>()
  let totalInput = 0
  let totalOutput = 0
  let totalTokens = 0
  let totalRuns = 0

  for (const meta of metas) {
    const records = await readTrace(meta.scope, meta.traceId)
    let best: { inputTokens: number; outputTokens: number; totalTokens: number } | null = null
    for (const r of records) {
      const m = r.meta as Record<string, unknown> | undefined
      if (!m) continue
      const input = Number(m.inputTokens ?? m.input_tokens ?? 0)
      const output = Number(m.outputTokens ?? m.output_tokens ?? 0)
      const total = Number(m.totalTokens ?? m.total_tokens ?? input + output)
      if (input || output || total) {
        if (!best || total > best.totalTokens) best = { inputTokens: input, outputTokens: output, totalTokens: total }
      }
    }
    if (best) {
      const scope = meta.scope || 'unknown'
      const cur = byScope.get(scope) || { runs: 0, inputTokens: 0, outputTokens: 0, totalTokens: 0 }
      cur.runs += 1
      cur.inputTokens += best.inputTokens
      cur.outputTokens += best.outputTokens
      cur.totalTokens += best.totalTokens
      byScope.set(scope, cur)
      totalInput += best.inputTokens
      totalOutput += best.outputTokens
      totalTokens += best.totalTokens
      totalRuns += 1
    }
  }

  const scopes = Array.from(byScope.entries())
    .map(([scope, v]) => ({ scope, ...v }))
    .sort((a, b) => b.totalTokens - a.totalTokens)

  return success(c, {
    totalInputTokens: totalInput,
    totalOutputTokens: totalOutput,
    totalTokens,
    runs: totalRuns,
    byScope: scopes,
  })
})

// GET /traces?scope=Agent — 列出 trace 元信息（按时间倒序），可选按 scope 过滤
app.get('/', async (c) => {
  const scope = c.req.query('scope')
  const metas = await listTraces(scope || undefined)
  return success(c, metas)
})

// GET /traces/:scope/:traceId — 回放单个 trace 的完整事件序列（结构保真）
app.get('/:scope/:traceId', async (c) => {
  const scope = c.req.param('scope')
  const traceId = c.req.param('traceId')
  if (!scope || !traceId) return badRequest(c, 'scope and traceId are required')
  const records = await readTrace(scope, traceId)
  if (records.length === 0) return success(c, { scope, traceId, records: [] })
  return success(c, { scope, traceId, count: records.length, records })
})

export default app
