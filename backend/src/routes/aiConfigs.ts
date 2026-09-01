import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { existsSync } from 'node:fs'
import { spawn, execSync } from 'node:child_process'
import { stream } from 'hono/streaming'
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
  { serviceType: 'video', label: '视频', provider: 'volcengine', baseUrl: 'https://api.chatfire.site/volcengine', model: 'doubao-seedance-1-5-pro-251215', priority: 98 },
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

/**
 * 服务商目录 — 前端「设置」页服务商下拉框与预设推荐的唯一数据源。
 * 每项 = 一个 (provider, service_type) 组合；recommended=true 的 4 项对应
 * 「一键配置」预设卡片，与 PRESET_SERVICES 保持一致。
 */
const PROVIDER_CATALOG = [
  // 文本
  { serviceType: 'text', provider: 'chatfire', label: '推荐', baseUrl: 'https://api.chatfire.site', models: ['gemini-3-pro-preview'], endpointPrefix: '/v1', recommended: true, description: '文本生成 - ChatFire 代理' },
  { serviceType: 'text', provider: 'openrouter', label: 'OpenRouter 推荐', baseUrl: 'https://openrouter.ai/api', models: ['google/gemini-3-flash-preview'], endpointPrefix: '/v1', recommended: false, description: '文本生成 - OpenRouter' },
  { serviceType: 'text', provider: 'openai', label: 'OpenAI 推荐', baseUrl: 'https://api.openai.com', models: ['gpt-4.1-mini'], endpointPrefix: '/v1', recommended: false, description: '文本生成 - OpenAI' },
  { serviceType: 'text', provider: 'ollama', label: 'Ollama 本地', baseUrl: 'http://localhost:11434', models: [], endpointPrefix: '/v1', recommended: false, description: '文本生成 - Ollama 本地模型' },
  // 图片
  { serviceType: 'image', provider: 'chatfire', label: '推荐', baseUrl: 'https://api.chatfire.site', models: ['doubao-seedream-4-5-251128'], endpointPrefix: '/v1', recommended: false, description: '图片生成 - Seedream（ChatFire 代理）' },
  { serviceType: 'image', provider: 'gemini', label: 'Gemini 推荐', baseUrl: 'https://api.chatfire.site', models: ['gemini-3-pro-image-preview'], endpointPrefix: '/v1beta', recommended: true, description: '图片生成 - Gemini' },
  { serviceType: 'image', provider: 'volcengine', label: '火山推荐', baseUrl: 'https://ark.cn-beijing.volces.com', models: ['doubao-seedream-4-0-250828'], endpointPrefix: '/api/v3', recommended: false, description: '图片生成 - 火山方舟' },
  // 视频
  { serviceType: 'video', provider: 'volcengine', label: '火山引擎', baseUrl: 'https://api.chatfire.site/volcengine', models: ['doubao-seedance-1-5-pro-251215'], endpointPrefix: '/api/v3', recommended: true, description: '视频生成 - Seedance' },
  { serviceType: 'video', provider: 'vidu', label: 'Vidu 推荐', baseUrl: 'https://api.vidu.com', models: ['viduq3-turbo'], endpointPrefix: '/ent/v2', recommended: false, description: '视频生成 - Vidu' },
  { serviceType: 'video', provider: 'ali', label: '阿里推荐', baseUrl: 'https://dashscope.aliyuncs.com', models: ['wan2.6-i2v-flash'], endpointPrefix: '/api/v1', recommended: false, description: '视频生成 - 阿里云百炼' },
  // 音频
  { serviceType: 'audio', provider: 'minimax', label: 'MiniMax', baseUrl: 'https://api.chatfire.site/minimax', models: ['speech-2.8-hd'], endpointPrefix: '/v1', recommended: true, description: '语音生成 - MiniMax' },
] as const

/**
 * 幂等 seed 服务商目录到 ai_service_providers 表。
 * 按 name 去重（`${provider}-${serviceType}`），已存在则跳过。
 */
export function seedServiceProviders(): number {
  const ts = now()
  let inserted = 0
  for (const item of PROVIDER_CATALOG) {
    const name = `${item.provider}-${item.serviceType}`
    const [existing] = db.select().from(schema.aiServiceProviders)
      .where(eq(schema.aiServiceProviders.name, name)).all()
    if (existing) continue

    db.insert(schema.aiServiceProviders).values({
      name,
      displayName: item.label,
      serviceType: item.serviceType,
      provider: item.provider,
      defaultUrl: item.baseUrl,
      presetModels: JSON.stringify(item.models),
      description: item.description,
      endpointPrefix: item.endpointPrefix,
      isRecommended: item.recommended,
      isActive: true,
      createdAt: ts,
      updatedAt: ts,
    }).run()
    inserted++
  }
  if (inserted > 0) logTaskSuccess('AIConfig', 'seed-providers', { inserted, total: PROVIDER_CATALOG.length })
  return inserted
}

/** 解析 ai_service_configs.settings JSON 为对象（损坏时回退空对象） */
function parseSettingsObject(settings: string | null | undefined): Record<string, any> {
  try { return JSON.parse(settings || '{}') } catch { return {} }
}

/**
 * 组装 ai_service_configs.settings JSON：
 * - 保留既有 settings（含 checkpoint_map 等扩展配置），避免覆盖丢失
 * - negative_prompt / checkpoint_map 作为首类字段合并；settings 作为兜底整体合并
 */
function buildSettings(input: {
  existing?: string | null
  negative_prompt?: string
  checkpoint_map?: Record<string, string> | null
  settings?: Record<string, any> | null
}): string {
  const next = parseSettingsObject(input.existing)
  if (input.negative_prompt !== undefined) next.negative_prompt = input.negative_prompt || ''
  if (input.checkpoint_map !== undefined) {
    if (input.checkpoint_map == null) delete next.checkpoint_map
    else next.checkpoint_map = input.checkpoint_map
  }
  if (input.settings && typeof input.settings === 'object') Object.assign(next, input.settings)
  return JSON.stringify(next)
}

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

// ──────────────────────────────────────────────────────────────
// 通用「平台检测 + 模型列举」能力：识别各平台可用模型、检测模型是否存在
// ──────────────────────────────────────────────────────────────

type ModelListDescriptor = {
  url: string
  method: string
  headers: Record<string, string>
  parse: (data: any) => string[]
} | null

/** 根据 provider 构造「列出可用模型」的请求描述；不支持在线列举的返回 null */
function buildListModels(provider: string, baseUrl: string, apiKey?: string): ModelListDescriptor {
  const p = provider.toLowerCase()

  // OpenAI 兼容平台（OpenAI / OpenRouter / ChatFire）→ GET /v1/models
  if (p === 'openai' || p === 'openrouter' || p === 'chatfire') {
    return {
      method: 'GET',
      url: joinProviderUrl(baseUrl, '/v1', '/models'),
      headers: bearerHeaders(apiKey),
      parse: (d) => (Array.isArray(d?.data) ? d.data.map((m: any) => m?.id).filter(Boolean) : []),
    }
  }

  // Google Gemini → GET /v1beta/models（key 走 query）
  if (p === 'gemini') {
    const url = new URL(joinProviderUrl(baseUrl, '/v1beta', '/models'))
    if (apiKey) url.searchParams.set('key', apiKey)
    return {
      method: 'GET',
      url: url.toString(),
      headers: {},
      parse: (d) => (Array.isArray(d?.models)
        ? d.models.map((m: any) => String(m?.name || '').replace(/^models\//, '')).filter(Boolean)
        : []),
    }
  }

  // Ollama 本地 → GET /api/tags
  if (p === 'ollama') {
    return {
      method: 'GET',
      url: joinProviderUrl(baseUrl, '', '/api/tags'),
      headers: {},
      parse: (d) => (Array.isArray(d?.models) ? d.models.map((m: any) => m?.name).filter(Boolean) : []),
    }
  }

  // 阿里百炼 / 火山方舟 / MiniMax / Vidu / 本地 SD / CosyVoice 等无公开 list models 接口
  return null
}

/** 执行模型列举，返回 { listable, models, error } */
async function listProviderModels(provider: string, baseUrl: string, apiKey?: string) {
  const desc = buildListModels(provider, baseUrl, apiKey)
  if (!desc) return { listable: false, models: [] as string[], error: '' }
  try {
    const resp = await fetch(desc.url, {
      method: desc.method,
      headers: desc.headers,
      signal: AbortSignal.timeout(20_000),
    })
    if (!resp.ok) {
      return { listable: true, models: [] as string[], error: `HTTP ${resp.status} ${resp.statusText}`.trim() }
    }
    const data = await resp.json().catch(() => ({}))
    return { listable: true, models: desc.parse(data), error: '' }
  } catch (err: any) {
    return { listable: true, models: [] as string[], error: err.message }
  }
}

// GET /ai-configs?service_type=text
app.get('/', async (c) => {
  try {
  const serviceType = c.req.query('service_type')
  let rows = db.select().from(schema.aiServiceConfigs).all()
  if (serviceType) rows = rows.filter(r => r.serviceType === serviceType)

  const parsed = rows.map(r => {
    const settingsObj = parseSettingsObject(r.settings)
    return {
      ...toSnakeCase(r),
      model: r.model ? JSON.parse(r.model) : [],
      negative_prompt: settingsObj.negative_prompt || '',
      checkpoint_map: settingsObj.checkpoint_map ?? null,
      is_local: isLocalConfig(r.baseUrl ?? '', r.provider ?? ''),
    }
  })
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
    settings: buildSettings({ negative_prompt: body.negative_prompt, checkpoint_map: body.checkpoint_map, settings: body.settings }),
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

// ──────────────────────────────────────────────────────────────
// Ollama 本地模型管理：识别本机所有模型 + 下载模型 + 启动服务
// ──────────────────────────────────────────────────────────────

const DEFAULT_OLLAMA_URL = 'http://localhost:11434'

function normalizeOllamaUrl(baseUrl?: string): string {
  return String(baseUrl || DEFAULT_OLLAMA_URL).replace(/\/+$/, '')
}

function formatBytes(bytes?: number): string {
  if (!bytes || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let i = 0
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i++ }
  return `${value.toFixed(1)} ${units[i]}`
}

/** 探测 Ollama 可执行文件位置（Windows 优先，兼容 PATH） */
function findOllamaExe(): string | null {
  try {
    const where = execSync('where ollama', { encoding: 'utf8', shell: 'cmd.exe', timeout: 3000 }).trim()
    const first = where.split(/\r?\n/).find(Boolean)
    if (first) return first
  } catch { /* 不在 PATH 中 */ }
  const candidates = [
    process.env.LOCALAPPDATA ? `${process.env.LOCALAPPDATA}\\Programs\\Ollama\\ollama.exe` : '',
    'C:\\Program Files\\Ollama\\ollama.exe',
    'C:\\Program Files (x86)\\Ollama\\ollama.exe',
    'D:\\app\\ollama\\ollama.exe',
  ].filter(Boolean)
  for (const p of candidates) if (existsSync(p)) return p
  return null
}

async function isOllamaReachable(baseUrl: string): Promise<boolean> {
  try {
    const resp = await fetch(`${baseUrl}/api/tags`, { signal: AbortSignal.timeout(2500) })
    return resp.ok
  } catch { return false }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

/** 尝试启动本地 Ollama 服务并等待就绪 */
async function tryStartOllama(): Promise<{ started: boolean; message: string; exe?: string | null }> {
  if (await isOllamaReachable(DEFAULT_OLLAMA_URL)) {
    return { started: false, message: 'Ollama 已在运行', exe: findOllamaExe() }
  }
  const exe = findOllamaExe()
  if (!exe) return { started: false, message: '未找到 Ollama，请先安装（ollama.com）', exe: null }
  try {
    const child = spawn(exe, ['serve'], { detached: true, stdio: 'ignore', windowsHide: true })
    child.unref()
  } catch (err: any) {
    return { started: false, message: `启动失败: ${err.message}`, exe }
  }
  for (let i = 0; i < 40; i++) {
    await sleep(500)
    if (await isOllamaReachable(DEFAULT_OLLAMA_URL)) {
      return { started: true, message: 'Ollama 已启动并就绪', exe }
    }
  }
  return { started: true, message: '启动命令已发出，若 10 秒内未就绪请检查 Ollama 安装', exe }
}

// POST /ai-configs/ollama/status — 检测 Ollama 运行状态并列出本机所有已安装模型
app.post('/ollama/status', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}))
    const baseUrl = normalizeOllamaUrl(body.base_url)
    const exe = findOllamaExe()
    if (!(await isOllamaReachable(baseUrl))) {
      return success(c, { running: false, base_url: baseUrl, models: [], exe, message: `无法连接 ${baseUrl}（Ollama 服务未运行）` })
    }
    const resp = await fetch(`${baseUrl}/api/tags`, { signal: AbortSignal.timeout(5000) })
    if (!resp.ok) {
      return success(c, { running: true, base_url: baseUrl, models: [], exe, message: `Ollama 响应异常 (HTTP ${resp.status})` })
    }
    const data = await resp.json() as { models?: Array<{ name: string; size?: number; modified_at?: string; digest?: string }> }
    const models = (data.models || [])
      .map((m) => ({
        name: m.name,
        size: m.size || 0,
        size_label: formatBytes(m.size),
        modified_at: m.modified_at || '',
        digest: (m.digest || '').slice(0, 12),
      }))
      .sort((a, b) => a.name.localeCompare(b.name))
    return success(c, { running: true, base_url: baseUrl, models, exe, message: `检测到 ${models.length} 个本地模型` })
  } catch (err: any) {
    logTaskError('AIConfig', 'ollama-status', { error: err.message })
    return badRequest(c, err.message)
  }
})

// POST /ai-configs/ollama/start — 启动本地 Ollama 服务
app.post('/ollama/start', async (c) => {
  try {
    const result = await tryStartOllama()
    return success(c, result)
  } catch (err: any) {
    logTaskError('AIConfig', 'ollama-start', { error: err.message })
    return badRequest(c, err.message)
  }
})

// POST /ai-configs/ollama/pull — 下载/拉取模型，NDJSON 流式返回进度
app.post('/ollama/pull', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}))
    const baseUrl = normalizeOllamaUrl(body.base_url)
    const name = String(body.name || '').trim()
    if (!name) return badRequest(c, '缺少模型名，例如 qwen3:8b')
    if (!(await isOllamaReachable(baseUrl))) {
      return badRequest(c, `Ollama 服务未运行（${baseUrl}），请先在下方启动`)
    }
    const resp = await fetch(`${baseUrl}/api/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, stream: true }),
      signal: AbortSignal.timeout(60 * 60_000),
    })
    if (!resp.ok || !resp.body) {
      const text = await resp.text().catch(() => '')
      return success(c, { ok: false, error: (text || `HTTP ${resp.status}`).slice(0, 400) })
    }
    c.header('Content-Type', 'application/x-ndjson; charset=utf-8')
    return stream(c, async (s) => {
      const reader = resp.body!.getReader()
      const decoder = new TextDecoder()
      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          await s.write(decoder.decode(value, { stream: true }))
        }
      } finally { reader.releaseLock() }
    })
  } catch (err: any) {
    logTaskError('AIConfig', 'ollama-pull', { error: err.message })
    return success(c, { ok: false, error: err.message })
  }
})

// POST /ai-configs/ollama/delete — 删除本地 Ollama 模型
app.post('/ollama/delete', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}))
    const baseUrl = normalizeOllamaUrl(body.base_url)
    const name = String(body.name || '').trim()
    if (!name) return badRequest(c, '缺少模型名')
    if (!(await isOllamaReachable(baseUrl))) {
      return badRequest(c, `Ollama 服务未运行（${baseUrl}）`)
    }
    const resp = await fetch(`${baseUrl}/api/delete`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
      signal: AbortSignal.timeout(120_000),
    })
    if (!resp.ok) {
      const text = await resp.text().catch(() => '')
      return badRequest(c, (text || `HTTP ${resp.status}`).slice(0, 400))
    }
    return success(c, { ok: true, deleted: name })
  } catch (err: any) {
    logTaskError('AIConfig', 'ollama-delete', { error: err.message })
    return badRequest(c, err.message)
  }
})

// POST /ai-configs/models — 检测平台连通性 + 列出可用模型 + 检测指定模型是否存在
app.post('/models', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}))
    const provider = String(body.provider || '').trim()
    const baseUrl = String(body.base_url || '').trim()
    if (!provider) return badRequest(c, 'provider is required')

    const apiKey = String(body.api_key || '')
    // 支持单 model 字符串（兼容旧调用）或 models 数组（批量检测）
    const wantedList: string[] = (Array.isArray(body.models) ? body.models : body.model ? [body.model] : [])
      .map((m: any) => String(m || '').trim())
      .filter(Boolean)
    const p = provider.toLowerCase()

    const { listable, models, error } = await listProviderModels(provider, baseUrl, apiKey)

    // 检测模型是否存在：精确匹配；Ollama 允许省略 tag（qwen3 匹配 qwen3:14b）
    const modelChecks = wantedList.map((wanted) => {
      const w = wanted.toLowerCase()
      const exists = models.some((id) => {
        const idl = String(id).toLowerCase()
        if (idl === w) return true
        if (p === 'ollama' && idl.startsWith(`${w}:`)) return true
        return false
      })
      return { model: wanted, exists }
    })

    const firstMissing = modelChecks.find((c) => !c.exists)
    const message = error
      ? `平台连接失败：${error}`
      : !listable
        ? '该平台暂不支持在线列举模型，可手动填写模型名（可用「测试配置」验证连通性）'
        : modelChecks.length === 0
          ? `已检测到 ${models.length} 个可用模型`
          : firstMissing
            ? `模型 ${firstMissing.model} 未出现在平台模型列表中，请核对模型名`
            : modelChecks.length === 1
              ? `模型 ${modelChecks[0].model} 可用 ✓`
              : `${modelChecks.length} 个模型均可用 ✓`

    logTaskProgress('AIConfig', 'models-list', {
      provider,
      listable,
      modelsCount: models.length,
      wantedCount: modelChecks.length,
      missingCount: modelChecks.filter((c) => !c.exists).length,
    })

    return success(c, {
      provider,
      base_url: baseUrl,
      listable,
      reachable: listable ? !error : null,
      models: models.slice(0, 500),
      models_count: models.length,
      model: modelChecks[0]?.model || null,
      model_exists: modelChecks[0]?.exists ?? false,
      model_checks: modelChecks,
      message,
    })
  } catch (err: any) {
    logTaskError('AIConfig', 'models', { error: err.message })
    return badRequest(c, err.message)
  }
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
  const settingsObj = parseSettingsObject(row.settings)
  return success(c, {
    ...toSnakeCase(row),
    model: row.model ? JSON.parse(row.model) : [],
    negative_prompt: settingsObj.negative_prompt || '',
    checkpoint_map: settingsObj.checkpoint_map ?? null,
  })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// PUT /ai-configs/:id
app.put('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid config id')
  const [row] = db.select().from(schema.aiServiceConfigs).where(eq(schema.aiServiceConfigs.id, id)).all()
  if (!row) return notFound(c)
  const body = await c.req.json()
  const updates: Record<string, any> = { updatedAt: now() }

  if ('provider' in body) updates.provider = body.provider
  if ('name' in body) updates.name = body.name
  if ('base_url' in body) updates.baseUrl = body.base_url
  if ('api_key' in body) updates.apiKey = body.api_key
  if ('model' in body) updates.model = JSON.stringify(body.model)
  if ('priority' in body) updates.priority = body.priority
  if ('is_active' in body) updates.isActive = body.is_active
  // settings 合并写入：negative_prompt / checkpoint_map / settings 任一出现即重算，避免覆盖既有扩展配置
  if ('negative_prompt' in body || 'checkpoint_map' in body || 'settings' in body) {
    updates.settings = buildSettings({
      existing: row.settings,
      negative_prompt: body.negative_prompt,
      checkpoint_map: body.checkpoint_map,
      settings: body.settings,
    })
  }

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
app.get('/gpu/status', async (c) => {
  try {
  const status = gpuManager.getStatus()

  // 尝试获取 nvidia-smi 实时数据（Windows）
  const smi = await getNvidiaSmi().catch(() => null)

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
    .filter(row => isLocalConfig(row.baseUrl ?? '', row.provider ?? ''))
    .map(row => ({
      ...toSnakeCase(row),
      model: row.model ? JSON.parse(row.model) : [],
    }))
  return success(c, localConfigs)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// ─── 本地运行时健康检查 ───────────────────────────────────────
// 与 LOCAL_PRESET_SERVICES 的 baseUrl 保持一致，作为「一键配置」之外的健康探测来源。
const LOCAL_RUNTIMES = [
  { key: 'ollama', label: '文本 · Ollama', serviceType: 'text', provider: 'ollama', baseUrl: 'http://localhost:11434', probePath: '/api/tags' },
  { key: 'local-sd', label: '图像 · Stable Diffusion', serviceType: 'image', provider: 'local-sd', baseUrl: 'http://localhost:7860', probePath: '/sdapi/v1/samplers' },
  { key: 'h3', label: '视频 · MiniMax H3', serviceType: 'video', provider: 'minimax', baseUrl: 'http://localhost:8765', probePath: '/' },
  { key: 'cosyvoice', label: '语音 · CosyVoice', serviceType: 'audio', provider: 'cosyvoice', baseUrl: 'http://localhost:9880', probePath: '/' },
] as const

async function probeLocalRuntime(baseUrl: string, probePath: string, timeoutMs = 2500) {
  const startedAt = Date.now()
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const resp = await fetch(`${baseUrl}${probePath}`, { method: 'GET', signal: controller.signal })
    return { reachable: true, httpStatus: resp.status, latencyMs: Date.now() - startedAt, error: '' }
  } catch (err: any) {
    const aborted = err?.name === 'AbortError'
    return { reachable: false, httpStatus: null, latencyMs: Date.now() - startedAt, error: aborted ? `超时(${timeoutMs}ms)` : (err?.message || '连接失败') }
  } finally {
    clearTimeout(timer)
  }
}

// GET /runtime/health — 统一探测本地四大运行时（文本/图像/视频/语音）
app.get('/runtime/health', async (c) => {
  try {
    const configs = db.select().from(schema.aiServiceConfigs).all()
    const services = await Promise.all(LOCAL_RUNTIMES.map(async (rt) => {
      const probe = await probeLocalRuntime(rt.baseUrl, rt.probePath)
      const registered = configs.filter(row =>
        isLocalConfig(row.baseUrl ?? '', row.provider ?? '') && row.serviceType === rt.serviceType
      )
      return {
        key: rt.key,
        label: rt.label,
        service_type: rt.serviceType,
        provider: rt.provider,
        base_url: rt.baseUrl,
        running: probe.reachable,
        http_status: probe.httpStatus,
        latency_ms: probe.latencyMs,
        error: probe.error,
        registered_count: registered.length,
        registered: registered.map(r => ({ id: r.id, name: r.name, provider: r.provider, base_url: r.baseUrl })),
      }
    }))
    const runningCount = services.filter(s => s.running).length
    return success(c, {
      services,
      running_count: runningCount,
      total: services.length,
      all_running: runningCount === services.length,
    })
  } catch (err: any) {
    logTaskError('AIConfig', 'runtime-health', { error: err.message })
    return badRequest(c, err.message)
  }
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

  const { exec } = await import('child_process')
  const output = await new Promise<string>((resolve, reject) => {
    exec(cmd, { encoding: 'utf-8', timeout: 5000 }, (err, stdout) => {
      if (err) reject(err)
      else resolve(stdout)
    })
  })
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
