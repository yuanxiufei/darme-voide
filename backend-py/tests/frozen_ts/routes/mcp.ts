import { Hono } from 'hono'
import { success, badRequest } from '../utils/response.js'
import { refreshMcp, getMcpStatus, testMcpServer } from '../agents/mcp.js'

const app = new Hono()

// GET /mcp/status — 查看已配置与已连接的 MCP server 及工具
app.get('/status', (c) => success(c, getMcpStatus()))

// POST /mcp/refresh — 关闭所有连接并重新发现
app.post('/refresh', async (c) => {
  try {
    await refreshMcp()
    return success(c, getMcpStatus())
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message })
  }
})

// POST /mcp/test — 测试连接单个 server（独立临时连接，测完即关）
app.post('/test', async (c) => {
  try {
    const body = await c.req.json()
    if (!body || typeof body !== 'object' || !body.name) return badRequest(c, 'server name is required')
    const result = await testMcpServer(body)
    return success(c, result)
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message })
  }
})

export default app
