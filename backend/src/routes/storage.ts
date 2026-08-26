/**
 * 数据存储路由 — 查询 / 切换数据根目录（SQLite 数据库 + 图片/视频/音频等生成文件）
 */
import { Hono } from 'hono'
import { success, badRequest } from '../utils/response.js'
import { getStorageInfo, changeDataRoot } from '../services/data-storage.js'

const router = new Hono()

// GET /storage/info — 当前数据存储信息
router.get('/info', (c) => {
  try {
    return success(c, getStorageInfo())
  } catch (err: any) {
    return c.json({ code: 500, data: null, message: err.message })
  }
})

// POST /storage/change — 切换数据目录（默认自动迁移旧数据，旧目录保留）
router.post('/change', async (c) => {
  try {
    const body = await c.req.json()
    const { path: newPath, migrate = true } = body || {}
    if (!newPath || !String(newPath).trim()) return badRequest(c, '请填写目标目录路径')
    const info = changeDataRoot(String(newPath), { migrate: migrate !== false })
    return success(c, info)
  } catch (err: any) {
    return badRequest(c, err.message)
  }
})

export default router
