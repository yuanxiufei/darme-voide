import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, notFound, created, badRequest, now, parseParamId } from '../utils/response.js'
import { toSnakeCase } from '../utils/transform.js'
import { joinProviderUrl } from '../services/adapters/url.js'
import { redactUrl, logTaskError, logTaskProgress, logTaskSuccess } from '../utils/task-logger.js'
import { gpuManager, isLocalConfig, type GpuStatus } from '../services/gpu-manager.js'

const app = new Hono()

const PRESET_SERVICES = [
  { serviceType: 'text', label: '文本', provider: 'chatfire', baseUrl: 'https://api.chatfire.site', model: 'gemini-3-pro-preview', priority: 100 },
  { serviceType: 'image', label: '图片', provider: 'gemini', baseUrl: 'https://api.chatfire.site', model: 'gemini-3-pro-image-preview', priority: 99 },
  { serviceType: 'video', label: '视频', provider: 'minimax', baseUrl: 'https://api.chatfire.site/minimax', model: 'hailuo-02', priority: 98 },
  { serviceType: 'audio', label: '音频', provider: 'minimax', baseUrl: 'https://api.chatfire.site/minimax', model: 'speech-2.8-hd', priority: 97 },
] as const

/** 本地模型预设 — 无需 API Key，通过 /quick-local 一键初始化 */
const LOCAL_PRESET_SERVICES = [
  { serviceType: 'text', label: '文本(本地)', provider: 'openai', baseUrl: 'http://localhost:11434', model: 'qwen3:14b', priority: 85 },
  { serviceType: 'image', label: '图片(本地)', provider: 'local-sd', baseUrl: 'http://localhost:7860', model: 'sdxl-base', priority: 84 },
  { serviceType: 'video', label: '视频(本地H3)', provider: 'minimax', baseUrl: 'http://localhost:8765', model: 'hailuo-02', priority: 83 },
  { serviceType: 'audio', label: '音频(本地)', provider: 'cosyvoice', baseUrl: 'http://localhost:9880', model: 'cosyvoice-v2', priority: 82 },
] as const

const PRESET_AGENT_DEFAULTS = [
  { agentType: 'script_rewriter', name: '剧本改写' },
  { agentType: 'extractor', name: '角色场景提取' },
  { agentType: 'storyboard_breaker', name: '分镜拆解' },
  { agentType: 'voice_assigner', name: '音色分配' },
  { agentType: 'grid_prompt_generator', name: '图片提示词生成' },
] as const

const PRESET_AGENT_MODEL = 'gemini-3-pro-preview'

function bearerHeaders(apiKey?: string, withJson = false) {
  const headers: Record<string, string> = {}
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`
  if (withJson) headers['Content-Type'] = 'application/json'
  return headers
}

function geminiHeaders(apiKey?: string, withJson = false) {
  const headers: Record<string, string> = {}
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`
    headers['x-goog-api-key'] = apiKey
  }
  if (withJson) headers['Content-Type'] = 'application/json'
  return headers
}

function viduHeaders(apiKey?: string, withJson = false) {
  const headers: Record<string, string> = {}
  if (apiKey) headers.Authorization = `Token ${apiKey}`
  if (withJson) headers['Content-Type'] = 'application/json'
  return headers
}

function buildProbe(serviceType: string, provider: string, baseUrl: string, model?: string, apiKey?: string) {
  const p = provider.toLowerCase()
  const m = model || ''

  if (p === 'gemini') {
    const url = new URL(joinProviderUrl(baseUrl, '/v1beta', `/models/${m || 'gemini-2.5-flash'}:generateContent`))
    if (apiKey) url.searchParams.set('key', apiKey)
    return { method: 'POST', url: url.toString(), headers: geminiHeaders(apiKey, true), body: {} }
  }

  if (p === 'openai' || p === 'openrouter' || p === 'chatfire') {
    return {
      method: 'GET',
      url: joinProviderUrl(baseUrl, '/v1', '/models'),
      headers: bearerHeaders(apiKey),
      body: undefined,
    }
  }

  if (p === 'ali') {
    return {
      method: 'POST',
      url: joinProviderUrl(baseUrl, '/api/v1', serviceType === 'video'
        ? '/services/aigc/video-generation/video-synthesis'
        : '/services/aigc/image-generation/generation'),
      headers: bearerHeaders(apiKey, true),
      body: {},
    }
  }

  if (p === 'volcengine') {
    const path = serviceType === 'video'
      ? '/contents/generations/tasks'
      : '/images/generations'
    return {
      method: 'POST',
      url: joinProviderUrl(baseUrl, '/api/v3', path),
      headers: bearerHeaders(apiKey, true),
      body: {},
    }
  }

  if (p === 'minimax') {
    const path = serviceType === 'audio'
      ? '/t2a_v2'
      : serviceType === 'video'
        ? '/video_generation'
        : '/image_generation'
    return {
      method: 'POST',
      url: joinProviderUrl(baseUrl, '/v1', path),
      headers: bearerHeaders(apiKey, true),
      body: {},
    }
  }

  if (p === 'vidu') {
    return {
      method: 'POST',
      url: joinProviderUrl(baseUrl, '', '/ent/v2/img2video'),
      headers: viduHeaders(apiKey, true),
      body: {},
    }
  }

  // 本地服务 probe
  if (p === 'local-sd') {
    return {
      method: 'GET',
      url: joinProviderUrl(baseUrl, '', '/sdapi/v1/samplers'),
      headers: {},
      body: undefined,
    }
  }

  if (p === 'cosyvoice') {
    return {
      method: 'GET',
      url: joinProviderUrl(baseUrl, '', '/'),
      headers: {},
      body: undefined,
    }
  }

  if (p === 'ollama') {
    return {
      method: 'GET',
      url: joinProviderUrl(baseUrl, '', '/api/tags'),
      headers: {},
      body: undefined,
    }
  }

  return {
    method: 'GET',
    url: joinProviderUrl(baseUrl, '', m ? `/${m}` : '/'),
    headers: bearerHeaders(apiKey),
    body: undefined,
  }
}

// GET /ai-configs?service_type=text
app.get('/', async (c) => {
  try {
  const serviceType = c.req.query('service_type')
  let rows = db.select().from(schema.aiServiceConfigs).all()
  if (serviceType) rows = rows.filter(r => r.serviceType === serviceType)

  const parsed = rows.map(r => ({
    ...toSnakeCase(r),
    model: r.model ? JSON.parse(r.model) : [],
  }))
  return success(c, parsed)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// POST /ai-configs
app.post('/', async (c) => {
  try {
  const body = await c.req.json()
  const ts = now()

  // 验证必填字段
  if (!body.service_type || !body.provider) {
    return badRequest(c, 'service_type and provider are required')
  }

  const res = db.insert(schema.aiServiceConfigs).values({
    serviceType: body.service_type,
    provider: body.provider,
    name: body.name || `${body.provider}-${body.service_type}`,
    baseUrl: body.base_url || '',
    apiKey: body.api_key || '',
    model: JSON.stringify(body.model || []),
    priority: body.priority || 0,
    isActive: true,
    createdAt: ts,
    updatedAt: ts,
  }).run()

  const [row] = db.select().from(schema.aiServiceConfigs)
    .where(eq(schema.aiServiceConfigs.id, Number(res.lastInsertRowid))).all()

  return created(c, {
    ...toSnakeCase(row),
    model: row.model ? JSON.parse(row.model) : [],
  })
  } catch (err: any) { logTaskError('AIConfig', 'create', { error: err.message }); return badRequest(c, err.message) }
})

// POST /ai-configs/quick-preset
app.post('/quick-preset', async (c) => {
  try {
  const body = await c.req.json()
  const apiKey = String(body.api_key || '').trim()
  if (!apiKey) return badRequest(c, 'api_key is required')

  const ts = now()

  for (const preset of PRESET_SERVICES) {
    const [existing] = db.select().from(schema.aiServiceConfigs).where(eq(schema.aiServiceConfigs.serviceType, preset.serviceType)).all()
      .filter(row => row.provider === preset.provider)

    const values = {
      serviceType: preset.serviceType,
      provider: preset.provider,
      name: `默认${preset.label}服务`,
      baseUrl: preset.baseUrl,
      apiKey,
      model: JSON.stringify([preset.model]),
      priority: preset.priority,
      isActive: true,
      updatedAt: ts,
    }

    if (existing) {
      db.update(schema.aiServiceConfigs).set(values).where(eq(schema.aiServiceConfigs.id, existing.id)).run()
    } else {
      db.insert(schema.aiServiceConfigs).values({
        ...values,
        createdAt: ts,
      }).run()
    }
  }

  for (const agent of PRESET_AGENT_DEFAULTS) {
    const [existing] = db.select().from(schema.agentConfigs).where(eq(schema.agentConfigs.agentType, agent.agentType)).all()
    const values = {
      name: agent.name,
      model: PRESET_AGENT_MODEL,
      isActive: true,
      updatedAt: ts,
    }

    if (existing) {
      db.update(schema.agentConfigs).set(values).where(eq(schema.agentConfigs.id, existing.id)).run()
    } else {
      db.insert(schema.agentConfigs).values({
        agentType: agent.agentType,
        description: '',
        model: PRESET_AGENT_MODEL,
        name: agent.name,
        systemPrompt: '',
        temperature: 0.7,
        maxTokens: 4096,
        maxIterations: 10,
        isActive: true,
        createdAt: ts,
        updatedAt: ts,
      }).run()
    }
  }

  const configs = db.select().from(schema.aiServiceConfigs).all().map(row => ({
    ...toSnakeCase(row),
    model: row.model ? JSON.parse(row.model) : [],
  }))
  const agents = db.select().from(schema.agentConfigs).all().map(row => toSnakeCase(row))

  logTaskSuccess('AIConfig', 'quick-preset-applied', {
    serviceCount: PRESET_SERVICES.length,
    agentCount: PRESET_AGENT_DEFAULTS.length,
  })

  return success(c, {
    configs,
    agents,
    agent_model: PRESET_AGENT_MODEL,
  })
  } catch (err: any) { logTaskError('AIConfig', 'quick-preset', { error: err.message }); return badRequest(c, err.message) }
})

// POST /ai-configs/quick-local — 一键初始化本地模型（无需 API Key）
app.post('/quick-local', async (c) => {
  try {
  const ts = now()

  for (const preset of LOCAL_PRESET_SERVICES) {
    const [existing] = db.select().from(schema.aiServiceConfigs)
      .where(eq(schema.aiServiceConfigs.serviceType, preset.serviceType))
      .all()
      .filter(row => row.provider === preset.provider)

    const values = {
      serviceType: preset.serviceType,
      provider: preset.provider,
      name: `本地${preset.label}服务`,
      baseUrl: preset.baseUrl,
      apiKey: 'local', // 本地服务不需要真实 key，但字段不可为空
      model: JSON.stringify([preset.model]),
      priority: preset.priority,
      isActive: true,
      updatedAt: ts,
    }

    if (existing) {
      db.update(schema.aiServiceConfigs).set(values).where(eq(schema.aiServiceConfigs.id, existing.id)).run()
    } else {
      db.insert(schema.aiServiceConfigs).values({
        ...values,
        createdAt: ts,
      }).run()
    }
  }

  const configs = db.select().from(schema.aiServiceConfigs).all().map(row => ({
    ...toSnakeCase(row),
    model: row.model ? JSON.parse(row.model) : [],
  }))

  logTaskSuccess('AIConfig', 'quick-local-applied', {
    localServiceCount: LOCAL_PRESET_SERVICES.length,
  })

  return success(c, { configs })
  } catch (err: any) { logTaskError('AIConfig', 'quick-local', { error: err.message }); return badRequest(c, err.message) }
})

// POST /ai-configs/test
app.post('/test', async (c) => {
  const body = await c.req.json()
  if (!body.service_type || !body.provider || !body.base_url) {
    return badRequest(c, 'service_type, provider and base_url are required')
  }

  const model = Array.isArray(body.model) ? body.model[0] : body.model
  const probe = buildProbe(body.service_type, body.provider, body.base_url, model, body.api_key)
  const probeUrl = redactUrl(probe.url)

  logTaskProgress('AIConfig', 'probe-start', {
    serviceType: body.service_type,
    provider: body.provider,
    method: probe.method,
    url: probeUrl,
  })

  try {
    const resp = await fetch(probe.url, {
      method: probe.method,
      headers: probe.headers,
      body: probe.body ? JSON.stringify(probe.body) : undefined,
      signal: AbortSignal.timeout(30_000),
    })
    const text = await resp.text()
    const reachable = [200, 204, 400, 401, 403].includes(resp.status)
    const payload = {
      ok: resp.ok,
      reachable,
      status: resp.status,
      status_text: resp.statusText,
      method: probe.method,
      url: probeUrl,
      message: reachable
        ? (resp.ok ? '端点可访问，认证与路径基本正常' : '端点已响应，请根据状态码判断认证或路径是否正确')
        : '端点未按预期响应，请检查 Base URL 和代理前缀',
      response_preview: text.slice(0, 240),
    }
    if (reachable) {
      logTaskSuccess('AIConfig', 'probe-done', {
        provider: body.provider,
        status: resp.status,
        url: probeUrl,
      })
    } else {
      logTaskError('AIConfig', 'probe-unexpected', {
        provider: body.provider,
        status: resp.status,
        url: probeUrl,
      })
    }
    return success(c, payload)
  } catch (error: any) {
    logTaskError('AIConfig', 'probe-failed', {
      provider: body.provider,
      url: probeUrl,
      error: error.message,
    })
    return success(c, {
      ok: false,
      reachable: false,
      method: probe.method,
      url: probeUrl,
      message: error.message || '请求失败',
      response_preview: '',
    })
  }
})

// GET /ai-configs/:id
app.get('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid config id')
  const [row] = db.select().from(schema.aiServiceConfigs).where(eq(schema.aiServiceConfigs.id, id)).all()
  if (!row) return notFound(c)
  return success(c, {
    ...toSnakeCase(row),
    model: row.model ? JSON.parse(row.model) : [],
  })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// PUT /ai-configs/:id
app.put('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid config id')
  const body = await c.req.json()
  const updates: Record<string, any> = { updatedAt: now() }

  if ('provider' in body) updates.provider = body.provider
  if ('name' in body) updates.name = body.name
  if ('base_url' in body) updates.baseUrl = body.base_url
  if ('api_key' in body) updates.apiKey = body.api_key
  if ('model' in body) updates.model = JSON.stringify(body.model)
  if ('priority' in body) updates.priority = body.priority
  if ('is_active' in body) updates.isActive = body.is_active

  db.update(schema.aiServiceConfigs).set(updates).where(eq(schema.aiServiceConfigs.id, id)).run()
  return success(c)
  } catch (err: any) { logTaskError('AIConfig', 'update', { error: err.message, id: c.req.param('id') }); return badRequest(c, err.message) }
})

// DELETE /ai-configs/:id
app.delete('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid config id')
  const [existing] = db.select().from(schema.aiServiceConfigs).where(eq(schema.aiServiceConfigs.id, id)).all()
  if (!existing) return notFound(c, 'Config not found')
  db.delete(schema.aiServiceConfigs).where(eq(schema.aiServiceConfigs.id, id)).run()
  return success(c)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// GET /ai-providers
export const aiProviders = new Hono()
aiProviders.get('/', async (c) => {
  const rows = db.select().from(schema.aiServiceProviders).all()
  const parsed = rows.map(r => ({
    ...toSnakeCase(r),
    preset_models: r.presetModels ? JSON.parse(r.presetModels) : [],
  }))
  return success(c, parsed)
})

// ─── GPU 显存监控 ──────────────────────────────────────────────

// GET /gpu/status — GPU 显存使用状态
app.get('/gpu/status', (c) => {
  try {
  const status = gpuManager.getStatus()

  // 尝试获取 nvidia-smi 实时数据（Windows）
  const smiPromise = getNvidiaSmi().catch(() => null)

  return smiPromise.then(smi => {
    const result: any = {
      ...status,
      isLocalMode: smi !== null,
      hardware: smi ? {
        gpuName: smi.gpuName,
        totalMemoryMB: smi.totalMemoryMB,
        usedMemoryMB: smi.usedMemoryMB,
        freeMemoryMB: smi.freeMemoryMB,
        utilizationPercent: smi.utilizationPercent,
        temperatureC: smi.temperatureC,
      } : null,
    }
    return c.json(result)
  })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// POST /gpu/release-all — 强制释放所有本地模型显存（调试用）
app.post('/gpu/release-all', async (c) => {
  try {
  await gpuManager.releaseAll()
  return success(c, { message: 'All GPU models released', status: gpuManager.getStatus() })
  } catch (err: any) { logTaskError('AIConfig', 'gpu-release-all', { error: err.message }); return badRequest(c, err.message) }
})

// GET /configs/local — 列出所有本地配置
app.get('/configs/local', (c) => {
  try {
  const configs = db.select().from(schema.aiServiceConfigs).all()
  const localConfigs = configs
    .filter(row => isLocalConfig(row.baseUrl, row.provider))
    .map(row => ({
      ...toSnakeCase(row),
      model: row.model ? JSON.parse(row.model) : [],
    }))
  return success(c, localConfigs)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// ─── nvidia-smi 辅助 ──────────────────────────────────────────
interface NvidiaSmiInfo {
  gpuName: string
  totalMemoryMB: number
  usedMemoryMB: number
  freeMemoryMB: number
  utilizationPercent: number
  temperatureC: number
}

async function getNvidiaSmi(): Promise<NvidiaSmiInfo> {
  const isWindows = process.platform === 'win32'
  const cmd = isWindows
    ? 'nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits'
    : 'nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits'

  const { execSync } = await import('child_process')
  const output = execSync(cmd, { encoding: 'utf-8', timeout: 5000 })
  const [line] = output.trim().split('\n')
  const parts = line.split(',').map(s => s.trim())

  return {
    gpuName: parts[0],
    totalMemoryMB: parseInt(parts[1], 10),
    usedMemoryMB: parseInt(parts[2], 10),
    freeMemoryMB: parseInt(parts[3], 10),
    utilizationPercent: parseInt(parts[4], 10),
    temperatureC: parseInt(parts[5], 10),
  }
}

export default app
