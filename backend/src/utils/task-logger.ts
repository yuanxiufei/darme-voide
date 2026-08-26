import { randomUUID } from 'node:crypto'
import { appendTraceEvent } from './trace-store.js'

type LogLevel = 'INFO' | 'WARN' | 'ERROR' | 'SUCCESS'

const C = {
  reset: '\x1b[0m',
  dim: '\x1b[2m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
  blue: '\x1b[34m',
}

function colorFor(level: LogLevel) {
  if (level === 'SUCCESS') return C.green
  if (level === 'WARN') return C.yellow
  if (level === 'ERROR') return C.red
  return C.cyan
}

function timeText() {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false })
}

function safeValue(value: unknown) {
  if (value == null) return value
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return value
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function formatMeta(meta?: Record<string, unknown>) {
  if (!meta) return ''
  const entries = Object.entries(meta)
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) => `${key}=${safeValue(value)}`)
  return entries.length ? ` | ${entries.join(' ')}` : ''
}

export function redactUrl(rawUrl: string) {
  try {
    const url = new URL(rawUrl)
    for (const key of ['key', 'api_key', 'apikey', 'token', 'access_token']) {
      if (url.searchParams.has(key)) {
        url.searchParams.set(key, '***')
      }
    }
    return url.toString()
  } catch {
    return rawUrl
      .replace(/([?&](?:key|api_key|apikey|token|access_token)=)[^&]+/gi, '$1***')
  }
}

function sanitizeValue(value: unknown): unknown {
  if (value == null) return value
  if (typeof value === 'string') return truncateString(value)
  if (typeof value === 'number' || typeof value === 'boolean') return value

  if (Array.isArray(value)) {
    return value.map(item => sanitizeValue(item))
  }

  if (typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
      const lower = key.toLowerCase()
      if (['authorization', 'api_key', 'apikey', 'apiKey', 'token', 'access_token'].includes(key) ||
        lower.includes('authorization') || lower.includes('token') || lower.includes('apikey') || lower.includes('api_key')) {
        out[key] = '***'
        continue
      }

      if (typeof raw === 'string' && (lower === 'url' || lower.endsWith('url'))) {
        out[key] = redactUrl(raw)
        continue
      }

      if (typeof raw === 'string' && (
        lower === 'data' ||
        lower === 'b64_json' ||
        lower.includes('base64') ||
        lower.includes('audiohex') ||
        lower.includes('inline') ||
        raw.startsWith('data:image/')
      )) {
        out[key] = truncateString(raw, 48)
        continue
      }

      out[key] = sanitizeValue(raw)
    }
    return out
  }

  try {
    return JSON.parse(JSON.stringify(value))
  } catch {
    return String(value)
  }
}

function truncateString(value: string, edge = 120) {
  if (value.length <= edge * 2 + 24) return value
  return `${value.slice(0, edge)}...<trimmed ${value.length} chars>...${value.slice(-edge)}`
}

export function logTask(scope: string, action: string, meta?: Record<string, unknown>, level: LogLevel = 'INFO') {
  const color = colorFor(level)
  console.log(`${C.dim}${timeText()}${C.reset} ${color}[${scope}]${C.reset} ${action}${formatMeta(meta)}`)
}

export function logTaskStart(scope: string, action: string, meta?: Record<string, unknown>) {
  logTask(scope, `START ${action}`, meta, 'INFO')
}

export function logTaskProgress(scope: string, action: string, meta?: Record<string, unknown>) {
  logTask(scope, action, meta, 'INFO')
}

export function logTaskSuccess(scope: string, action: string, meta?: Record<string, unknown>) {
  logTask(scope, `DONE ${action}`, meta, 'SUCCESS')
}

export function logTaskWarn(scope: string, action: string, meta?: Record<string, unknown>) {
  logTask(scope, action, meta, 'WARN')
}

export function logTaskError(scope: string, action: string, meta?: Record<string, unknown>) {
  logTask(scope, `ERROR ${action}`, meta, 'ERROR')
}

export function logTaskPayload(scope: string, action: string, payload: unknown) {
  const sanitized = sanitizeValue(payload)
  const serialized = typeof sanitized === 'string'
    ? sanitized
    : JSON.stringify(sanitized, null, 2)
  console.log(`${C.dim}${timeText()}${C.reset} ${C.blue}[${scope}]${C.reset} ${action}\n${serialized}`)
}

export interface TraceHandle {
  traceId: string
  progress: (action: string, meta?: Record<string, unknown>) => void
  success: (action: string, meta?: Record<string, unknown>) => void
  warn: (action: string, meta?: Record<string, unknown>) => void
  error: (action: string, meta?: Record<string, unknown>) => void
  end: () => void
}

/**
 * 开启一次可追踪的执行追踪（trace）。
 * 同一 trace 内的所有日志自动附带 traceId 与 elapsedMs，便于把一次 Agent run
 * 的多条日志串起来，对齐 PenguinHarness 的"单一事实来源"可观测性思想。
 */
export function startTrace(scope: string, action: string, meta?: Record<string, unknown>): TraceHandle {
  const traceId = randomUUID().slice(0, 8)
  const startAt = Date.now()
  const withTrace = (m?: Record<string, unknown>): Record<string, unknown> => ({
    ...(m || {}),
    traceId,
    elapsedMs: Date.now() - startAt,
  })

  // 单点 emit：console 输出 + append-only JSONL 落盘（对齐第 9 章「单一事实来源」可观测）
  const emit = (level: LogLevel, a: string, m: Record<string, unknown>) => {
    logTask(scope, a, m, level)
    appendTraceEvent({
      ts: new Date().toISOString(),
      traceId,
      scope,
      level,
      action: a,
      elapsedMs: m.elapsedMs as number | undefined,
      meta: sanitizeValue(m) as Record<string, unknown>,
    })
  }

  emit('INFO', `START ${action}`, withTrace(meta))

  return {
    traceId,
    progress: (a, m) => emit('INFO', a, withTrace(m)),
    success: (a, m) => emit('SUCCESS', `DONE ${a}`, withTrace(m)),
    warn: (a, m) => emit('WARN', a, withTrace(m)),
    error: (a, m) => emit('ERROR', `ERROR ${a}`, withTrace(m)),
    end: () => emit('INFO', `END trace ${traceId}`, { ...withTrace(meta), totalMs: Date.now() - startAt }),
  }
}
