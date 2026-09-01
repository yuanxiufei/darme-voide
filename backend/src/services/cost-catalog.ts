/**
 * 成本单价目录（估算值，按公开定价近似，非精确账单）
 *
 * 定位：让「多集长剧 + QC 自动重拍」的烧钱情况可量化。
 * 单价仅用于估算 cost_amount，真实账单以厂商结算为准。
 * 每项可按 provider/model 前缀匹配；也可在 ai_service_configs.settings.pricing 覆盖：
 *   settings: { "pricing": { "unit": "second", "price": 0.5 } }  或  { "pricing": 0.3 }（用默认单位）
 */

export type PricingUnit = 'image' | 'second' | 'char' | 'k-token'

export interface PricingRule {
  /** 计费单位：image=每张图；second=每秒视频/音频；char=每字；k-token=每千 token */
  unit: PricingUnit
  /** 每单位价格（元） */
  price: number
}

/** 默认单位映射（无 pricing 覆盖时，不同类型任务取什么单位计数） */
export const DEFAULT_UNITS: Record<string, PricingUnit> = {
  image: 'image',
  video: 'second',
  audio: 'char',
  text: 'k-token',
}

/** 单价目录：provider → [模型前缀 → 单价]。模型前缀与 model 做包含匹配（首个命中）。 */
export const COST_CATALOG: Record<string, Array<{ key: string; rule: PricingRule }>> = {
  minimax: [
    // 视频（Hailuo / H3）
    { key: 'hailuo', rule: { unit: 'second', price: 0.3 } },
    { key: 'h3', rule: { unit: 'second', price: 0.3 } },
    { key: 'video', rule: { unit: 'second', price: 0.3 } },
    // 图片
    { key: 'image', rule: { unit: 'image', price: 0.1 } },
    // TTS（按字）
    { key: 'speech', rule: { unit: 'char', price: 0.003 } },
    // LLM（按千 token，输入/输出混合估算）
    { key: 'abab', rule: { unit: 'k-token', price: 0.01 } },
    { key: 'mini-max', rule: { unit: 'k-token', price: 0.01 } },
  ],
  volcengine: [
    // 图片（Seedream 系列）
    { key: 'seedream-4', rule: { unit: 'image', price: 0.3 } },
    { key: 'seedream-3', rule: { unit: 'image', price: 0.15 } },
    { key: 'seedream', rule: { unit: 'image', price: 0.15 } },
    // 视频（Seedance 系列）
    { key: 'seedance-1-0-pro', rule: { unit: 'second', price: 0.5 } },
    { key: 'seedance-1-0', rule: { unit: 'second', price: 0.35 } },
    { key: 'seedance', rule: { unit: 'second', price: 0.35 } },
    // LLM（Doubao 系列）
    { key: 'doubao', rule: { unit: 'k-token', price: 0.003 } },
  ],
  vidu: [
    { key: 'vidu', rule: { unit: 'second', price: 0.3 } },
  ],
  ali: [
    { key: 'wanx', rule: { unit: 'image', price: 0.2 } },
    { key: 'wan', rule: { unit: 'second', price: 0.3 } },
    { key: 'qwen', rule: { unit: 'k-token', price: 0.002 } },
    { key: 'cosyvoice', rule: { unit: 'char', price: 0.002 } },
  ],
  openai: [
    { key: 'gpt-4o-mini', rule: { unit: 'k-token', price: 0.006 } },
    { key: 'gpt-4o', rule: { unit: 'k-token', price: 0.03 } },
    { key: 'tts', rule: { unit: 'char', price: 0.002 } },
  ],
  openrouter: [
    { key: 'openai', rule: { unit: 'k-token', price: 0.02 } },
    { key: 'anthropic', rule: { unit: 'k-token', price: 0.03 } },
    { key: 'deepseek', rule: { unit: 'k-token', price: 0.003 } },
  ],
  deepseek: [
    { key: 'deepseek', rule: { unit: 'k-token', price: 0.002 } },
  ],
  chatfire: [
    { key: 'claude', rule: { unit: 'k-token', price: 0.03 } },
    { key: 'gpt', rule: { unit: 'k-token', price: 0.02 } },
  ],
  ollama: [],
  local: [],
}

/** 解析 settings.pricing 覆盖（对象 {unit, price} 或纯数字=单价，用默认单位） */
function parsePricingOverride(settings: Record<string, any> | null | undefined, defaultUnit: PricingUnit): PricingRule | null {
  const pricing = settings?.pricing
  if (pricing == null) return null
  if (typeof pricing === 'number') {
    return { unit: defaultUnit, price: pricing }
  }
  if (typeof pricing === 'object') {
    const price = Number(pricing.price)
    if (Number.isFinite(price) && price >= 0) {
      return {
        unit: (pricing.unit as PricingUnit) || defaultUnit,
        price,
      }
    }
  }
  return null
}

/**
 * 估算一次调用的成本（元）。
 * @param serviceType image / video / audio / text
 * @param provider 提供商（小写不敏感）
 * @param model 模型名
 * @param units 计费单位数（图片张数 / 视频秒数 / 音频字符数 / 千 token 数）
 * @param settings ai_service_configs.settings 解析对象，可含 pricing 覆盖
 * @returns 估算金额（元），查不到单价返回 null
 */
export function estimateCost(
  serviceType: string,
  provider: string,
  model: string,
  units: number | null | undefined,
  settings?: Record<string, any> | null,
): number | null {
  if (!model || units == null || !Number.isFinite(units) || units <= 0) return null
  const providerKey = (provider || '').toLowerCase()
  const defaultUnit = DEFAULT_UNITS[serviceType] || 'request'

  const override = parsePricingOverride(settings, defaultUnit)
  if (override) return roundMoney(override.price * units)

  const entries = COST_CATALOG[providerKey] || []
  const m = String(model).toLowerCase()
  const hit = entries.find((e) => m.includes(e.key))
  if (!hit) return null
  return roundMoney(hit.rule.price * units)
}

function roundMoney(n: number): number {
  return Math.round(n * 10000) / 10000
}
