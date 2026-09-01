import { serve } from '@hono/node-server'
import { serveStatic } from '@hono/node-server/serve-static'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

import dramas from './routes/dramas.js'
import episodes from './routes/episodes.js'
import storyboards from './routes/storyboards.js'
import scenes from './routes/scenes.js'
import characters from './routes/characters.js'
import props from './routes/props.js'
import images from './routes/images.js'
import videos from './routes/videos.js'
import generations from './routes/generations.js'
import upload from './routes/upload.js'
import aiConfigs, { aiProviders, seedServiceProviders } from './routes/aiConfigs.js'
import agentConfigs from './routes/agentConfigs.js'
import agent from './routes/agent.js'
import compose from './routes/compose.js'
import merge from './routes/merge.js'
import grid from './routes/grid.js'
import skills from './routes/skills.js'
import traces from './routes/traces.js'
import mcpRoutes from './routes/mcp.js'
import presetFramework from './routes/preset-framework.js'
import autoPipeline from './routes/auto-pipeline.js'
import evaluation from './routes/evaluation.js'
import exportRoute from './routes/export.js'
import webhooks from './routes/webhooks.js'
import aiVoices from './routes/aiVoices.js'
import characterLibrary from './routes/characterLibrary.js'
import sceneLibrary from './routes/sceneLibrary.js'
import weaponLibrary from './routes/weaponLibrary.js'
import costumeLibrary from './routes/costumeLibrary.js'
import presets from './routes/presets.js'
import storageRoute from './routes/storage.js'
import localModels from './routes/localModels.js'
import { requestLogger, errorHandler } from './middleware/logger.js'
import { config, getStorageRoot } from './config.js'
import { recoverVideoTasksOnStartup } from './services/video-generation.js'
import { recoverImageTasksOnStartup } from './services/image-generation.js'
import { recoverAutoPipelineOnStartup } from './services/auto-pipeline.js'
import { startEvaluationScheduler } from './services/evaluation-scheduler.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(__dirname, '../..')

const app = new Hono()

// Middleware
app.use('*', cors({
  origin: config.server.corsOrigins,
  credentials: true,
}))
app.use('*', requestLogger)
app.use('*', errorHandler)

// Health check
app.get('/api/v1/health', (c) => c.json({ status: 'ok', timestamp: new Date().toISOString() }))

// API routes
const api = new Hono()
api.route('/dramas', dramas)
api.route('/episodes', episodes)
api.route('/storyboards', storyboards)
api.route('/scenes', scenes)
api.route('/characters', characters)
api.route('/props', props)
api.route('/images', images)
api.route('/videos', videos)
api.route('/generations', generations)
api.route('/upload', upload)
api.route('/ai-configs', aiConfigs)
api.route('/ai-providers', aiProviders)
api.route('/local-models', localModels)
api.route('/agent-configs', agentConfigs)
api.route('/agent', agent)
api.route('/compose', compose)
api.route('/merge', merge)
api.route('/grid', grid)
api.route('/skills', skills)
api.route('/traces', traces)
api.route('/mcp', mcpRoutes)
api.route('/ai-voices', aiVoices)
api.route('/preset/framework', presetFramework)
api.route('/auto-pipeline', autoPipeline)
api.route('/evaluation', evaluation)
api.route('/export', exportRoute)
api.route('/character-library', characterLibrary)
api.route('/scene-library', sceneLibrary)
api.route('/weapon-library', weaponLibrary)
api.route('/costume-library', costumeLibrary)
api.route('/presets', presets)
api.route('/storage', storageRoute)

app.route('/api/v1', api)

// Webhook callbacks (Vidu, etc.) - outside /api/v1
app.route('/webhooks', webhooks)

// Serve static files (storage) — 动态读取当前数据根目录，支持 Range（视频拖动进度条）
const STATIC_MIME: Record<string, string> = {
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp',
  '.gif': 'image/gif', '.svg': 'image/svg+xml', '.bmp': 'image/bmp', '.ico': 'image/x-icon',
  '.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime', '.mkv': 'video/x-matroska',
  '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.ogg': 'audio/ogg', '.m4a': 'audio/mp4', '.aac': 'audio/aac',
  '.srt': 'application/x-subrip', '.vtt': 'text/vtt', '.txt': 'text/plain',
  '.json': 'application/json', '.pdf': 'application/pdf', '.zip': 'application/zip',
}

app.use('/static/*', async (c) => {
  const rel = c.req.path.replace(/^\/static\/?/, '')
  const storageRoot = path.resolve(getStorageRoot())
  const abs = path.resolve(storageRoot, rel)
  // 路径穿越防护
  if (abs !== storageRoot && !abs.startsWith(storageRoot + path.sep)) {
    return c.json({ code: 404, message: 'not found' }, 404)
  }
  let stat: fs.Stats
  try { stat = fs.statSync(abs) } catch { return c.json({ code: 404, message: 'not found' }, 404) }
  if (!stat.isFile()) return c.json({ code: 404, message: 'not found' }, 404)

  const ext = path.extname(abs).toLowerCase()
  const mime = STATIC_MIME[ext] || 'application/octet-stream'
  const range = c.req.header('Range')

  if (range) {
    const m = /bytes=(\d*)-(\d*)/.exec(range)
    if (m && (m[1] || m[2])) {
      let start = m[1] ? parseInt(m[1], 10) : 0
      let end = m[2] ? parseInt(m[2], 10) : stat.size - 1
      if (isNaN(start) || start < 0) start = 0
      if (isNaN(end) || end >= stat.size) end = stat.size - 1
      if (start > end) {
        return new Response(null, { status: 416, headers: { 'Content-Range': `bytes */${stat.size}` } })
      }
      const len = end - start + 1
      const buf = Buffer.alloc(len)
      const fd = fs.openSync(abs, 'r')
      try { fs.readSync(fd, buf, 0, len, start) } finally { fs.closeSync(fd) }
      return new Response(buf, {
        status: 206,
        headers: {
          'Content-Type': mime,
          'Content-Range': `bytes ${start}-${end}/${stat.size}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': String(len),
        },
      })
    }
  }

  const data = fs.readFileSync(abs)
  return new Response(data, {
    headers: {
      'Content-Type': mime,
      'Content-Length': String(stat.size),
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'public, max-age=86400',
    },
  })
})

// Serve frontend (production build)
const distPath = path.join(projectRoot, 'frontend', 'dist')
app.use('*', serveStatic({ root: distPath }))
app.get('*', serveStatic({ root: distPath, path: 'index.html' }))

// 幂等 seed 服务商目录（前端「设置」页数据源）
seedServiceProviders()

// 崩溃恢复：恢复上次进程退出时被中断的视频/图片生成任务 + 自动管线中间态续跑
recoverVideoTasksOnStartup()
recoverImageTasksOnStartup()
recoverAutoPipelineOnStartup()

// 自进化闭环：每日定时全量优化（配置 evaluation.auto_optimize.enabled 开启，默认关闭）
startEvaluationScheduler()

const port = config.server.port
console.log(`🚀 Drama Studio server on http://localhost:${port}`)
serve({ fetch: app.fetch, port, hostname: config.server.host })
