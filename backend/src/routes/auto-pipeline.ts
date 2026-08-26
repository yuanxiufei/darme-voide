/**
 * 全自动管线编排路由（Auto Pipeline Routes）
 *
 * 一句话梗概 → 整集短剧 全自动管线的 HTTP 入口：
 *   - POST /api/v1/auto-pipeline/run           创建 Drama + 触发后台管线
 *   - GET  /api/v1/auto-pipeline/status/:id    查询整剧进度
 *   - POST /api/v1/auto-pipeline/resume/:id    幂等续跑（崩溃恢复/媒体补跑）
 */
import { Hono } from 'hono'
import { streamSSE } from 'hono/streaming'
import { runAutoPipeline, resumeAutoPipeline, getAutoPipelineStatus } from '../services/auto-pipeline.js'
import { subscribePipeline } from '../utils/sse-hub.js'
import { success, badRequest, notFound, serverError, parseParamId } from '../utils/response.js'

const router = new Hono()

/**
 * POST /auto-pipeline/run
 * body: { premise, title?, genre?, style?, episodeCount?, imageConfigId?, videoConfigId?, audioConfigId?, withImages?, withVideos?, withCompose?, withMerge? }
 */
router.post('/run', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}))
    if (!body?.premise?.trim()) return badRequest(c, 'premise 不能为空')
    const result = runAutoPipeline(body)
    return success(c, result)
  } catch (err: any) {
    return serverError(c, err?.message || 'run failed')
  }
})

/** GET /auto-pipeline/status/:dramaId */
router.get('/status/:dramaId', (c) => {
  const dramaId = parseParamId(c, 'dramaId')
  if (!dramaId) return badRequest(c, 'invalid dramaId')
  const status = getAutoPipelineStatus(dramaId)
  if (!status) return notFound(c, 'drama not found')
  return success(c, status)
})

/**
 * GET /auto-pipeline/stream/:dramaId
 * SSE 进度流（对齐 PenguinHarness 第 16.4 节「SSE 端点的交付保证」）：
 *   - 心跳：每 20s 写一次 ping，保持长连接不被代理/浏览器断开
 *   - 回放协议：先 subscribe 再发初始快照，避免"订阅→快照"窗口内的广播竞争丢事件；
 *     历史状态由快照提供，SSE 只推增量（不缓存、不回放 buffer）
 *   - 写失败即断连：客户端断开（writeSSE reject）时清理定时器与订阅
 *   - X-Accel-Buffering: no 禁用反向代理缓冲
 */
router.get('/stream/:dramaId', (c) => {
  const dramaId = parseParamId(c, 'dramaId')
  if (!dramaId) return badRequest(c, 'invalid dramaId')
  if (!getAutoPipelineStatus(dramaId)) return notFound(c, 'drama not found')

  c.header('X-Accel-Buffering', 'no')
  c.header('Cache-Control', 'no-cache')
  c.header('Connection', 'keep-alive')

  return streamSSE(c, async (stream) => {
    let closed = false
    const heartbeat = setInterval(() => {
      if (closed) return
      stream.writeSSE({ event: 'ping', data: '' }).catch(() => {})
    }, 20000)

    // 先订阅再发快照：避免"订阅→快照"窗口内发生的事件丢失
    const unsubscribe = subscribePipeline(dramaId, (evt) => {
      if (closed) return
      stream.writeSSE({ event: evt.type, data: JSON.stringify(evt) }).catch(() => {})
    })

    stream.onAbort(() => {
      closed = true
      clearInterval(heartbeat)
      unsubscribe()
    })

    // 初始快照（对齐 16.4 回放协议：历史状态走快照，流只推增量）
    const snapshot = getAutoPipelineStatus(dramaId)
    await stream.writeSSE({ event: 'snapshot', data: JSON.stringify(snapshot) }).catch(() => {})
  })
})

/**
 * POST /auto-pipeline/resume/:dramaId
 * body: 可选的 override（如 { withVideos: true } 补跑媒体阶段）
 */
router.post('/resume/:dramaId', async (c) => {
  const dramaId = parseParamId(c, 'dramaId')
  if (!dramaId) return badRequest(c, 'invalid dramaId')
  try {
    const body = await c.req.json().catch(() => ({}))
    resumeAutoPipeline(dramaId, body)
    return success(c, { dramaId, resumed: true })
  } catch (err: any) {
    return serverError(c, err?.message || 'resume failed')
  }
})

export default router
