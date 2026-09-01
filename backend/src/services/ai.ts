/**
 * AI 服务抽象层 — 从数据库配置中获取 provider 和 API key
 */
import { db, schema } from '../db/index.js'
import { eq } from 'drizzle-orm'
import { logTaskProgress, logTaskWarn } from '../utils/task-logger.js'
import { joinProviderUrl } from './adapters/url.js'

export type ServiceType = 'text' | 'image' | 'video' | 'audio'

export interface AIConfig {
  provider: string
  baseUrl: string
  apiKey: string
  model: string
  /** 所有可用模型（按优先级排序），失败时自动 fallback */
  models: string[]
  /** 图片生成负面提示词（从配置 settings JSON 解析，可空） */
  negativePrompt?: string | null
  /** 配置原始 settings JSON 解析结果（含 checkpoint_map 等扩展配置） */
  settings?: Record<string, any> | null
}

/** 从 ai_service_configs.settings JSON 中解析负面提示词 */
function parseNegativePrompt(settings: string | null | undefined): string | null {
  if (!settings) return null
  try {
    const s = JSON.parse(settings)
    return typeof s?.negative_prompt === 'string' && s.negative_prompt.trim()
      ? s.negative_prompt.trim()
      : null
  } catch {
    return null
  }
}

/** 解析配置 settings JSON 为对象（供 adapter 读取 checkpoint_map 等扩展配置） */
function parseSettings(settings: string | null | undefined): Record<string, any> | null {
  if (!settings) return null
  try {
    const s = JSON.parse(settings)
    return s && typeof s === 'object' ? s : null
  } catch {
    return null
  }
}

export function getTextProviderBaseUrl(config: AIConfig) {
  const provider = config.provider.toLowerCase()

  if (provider === 'openai' || provider === 'openrouter' || provider === 'chatfire' || provider === 'ollama') {
    return joinProviderUrl(config.baseUrl, '/v1', '')
  }

  if (provider === 'volcengine') {
    return joinProviderUrl(config.baseUrl, '/api/v3', '')
  }

  if (provider === 'ali') {
    // 阿里百炼文本走 OpenAI 兼容模式端点（区别于图片/视频的 DashScope 原生 /api/v1）
    return joinProviderUrl(config.baseUrl, '/compatible-mode/v1', '')
  }

  if (provider === 'minimax') {
    // MiniMax 文本走 OpenAI 兼容 /v1 端点（与 adapters/openai-compatible-text 的 PREFIX_BY_PROVIDER 对齐）
    return joinProviderUrl(config.baseUrl, '/v1', '')
  }

  return config.baseUrl
}

export function getActiveConfig(serviceType: ServiceType): AIConfig | null {
  const rows = db.select().from(schema.aiServiceConfigs)
    .where(eq(schema.aiServiceConfigs.serviceType, serviceType))
    .all()
    .filter(r => r.isActive)
    .sort((a, b) => (b.priority || 0) - (a.priority || 0)) // 高优先级优先

  const active = rows[0]
  if (!active) {
    logTaskWarn('AIConfig', 'active-config-missing', { serviceType })
    return null
  }

  let models: string[] = []
  if (active.model) {
    try { models = JSON.parse(active.model) } catch { /* keep empty array */ }
  }
  logTaskProgress('AIConfig', 'active-config-selected', {
    serviceType,
    configId: active.id,
    provider: active.provider,
    model: models[0] || '',
    priority: active.priority,
  })
  return {
    provider: active.provider || '',
    baseUrl: active.baseUrl,
    apiKey: active.apiKey,
    model: models[0] || '',
    models: models.filter(Boolean),
    negativePrompt: parseNegativePrompt(active.settings),
    settings: parseSettings(active.settings),
  }
}

export function getTextConfig(): AIConfig {
  const config = getActiveConfig('text')
  if (!config) throw new Error('No active text AI config')
  return config
}

export function getAudioConfig(): AIConfig {
  const config = getActiveConfig('audio')
  if (!config) throw new Error('No active audio AI config — 请在设置中添加音频服务')
  return config
}

export function getAudioConfigById(id?: number | null): AIConfig {
  if (id) {
    const config = getConfigById(id)
    if (config) return config
  }
  return getAudioConfig()
}

export function getConfigById(id: number): AIConfig | null {
  const [row] = db.select().from(schema.aiServiceConfigs)
    .where(eq(schema.aiServiceConfigs.id, id)).all()
  if (!row || !row.isActive) {
    logTaskWarn('AIConfig', 'config-by-id-missing', { configId: id })
    return null
  }
  let models: string[] = []
  if (row.model) {
    try { models = JSON.parse(row.model) } catch { /* keep empty array */ }
  }
  logTaskProgress('AIConfig', 'config-by-id-selected', {
    configId: id,
    provider: row.provider,
    model: models[0] || '',
    serviceType: row.serviceType,
  })
  return {
    provider: row.provider || '',
    baseUrl: row.baseUrl,
    apiKey: row.apiKey,
    model: models[0] || '',
    models: models.filter(Boolean),
    negativePrompt: parseNegativePrompt(row.settings),
    settings: parseSettings(row.settings),
  }
}

/**
 * 按 provider 精确匹配活跃配置。
 * 用于崩溃恢复：记录里只存了 provider，需据此找到对应厂商的 baseUrl/apiKey 续跑任务。
 */
export function getActiveConfigByProvider(serviceType: ServiceType, provider: string): AIConfig | null {
  const rows = db.select().from(schema.aiServiceConfigs)
    .where(eq(schema.aiServiceConfigs.serviceType, serviceType))
    .all()
    .filter(r => r.isActive && (r.provider || '').toLowerCase() === provider.toLowerCase())
    .sort((a, b) => (b.priority || 0) - (a.priority || 0))

  const active = rows[0]
  if (!active) {
    logTaskWarn('AIConfig', 'active-config-by-provider-missing', { serviceType, provider })
    return null
  }

  let models: string[] = []
  if (active.model) {
    try { models = JSON.parse(active.model) } catch { /* keep empty array */ }
  }
  return {
    provider: active.provider || '',
    baseUrl: active.baseUrl,
    apiKey: active.apiKey,
    model: models[0] || '',
    models: models.filter(Boolean),
    negativePrompt: parseNegativePrompt(active.settings),
    settings: parseSettings(active.settings),
  }
}
