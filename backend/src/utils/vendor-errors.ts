/**
 * 上游厂商 HTTP 错误 → 用户可读中文的统一映射层。
 *
 * 设计动机（对齐参照项目 gcc-printfilm-main 的 services/videoHttpErrors.ts 思路）：
 *   不再把上游返回的英文 JSON 直接透传给前端，而是按 HTTP status + body
 *   归因成「用户能看懂、能行动」的中文说明。
 *
 * 重点覆盖两类原先被笼统处理的高频错误：
 *   1. 内容审核拦截 —— 尤其「上传图/首帧被识别为真实人物」这类与文字描述
 *      无关的归因（对应火山引擎 Seedance 的
 *      InputImageSensitiveContentDetected.PrivacyInformation 等）。
 *   2. 模型与接口不匹配 —— 图片模型填在视频栏、模型名拼错等。
 *
 * 同时提供 isNonRetryableHttpError 供轮询循环判定：哪些错误应立即判失败，
 * 哪些（5xx / 网络抖动）应继续重试，避免对审核/参数错误空转数十分钟。
 */

export type VendorErrorKind = 'video' | 'image' | 'audio' | 'text'

interface ParsedApiError {
  message?: string
  code?: string
  type?: string
}

const KIND_LABEL: Record<VendorErrorKind, string> = {
  video: '视频',
  image: '图片',
  audio: '语音',
  text: '文本',
}

/** 各场景的「内容审核拦截」通用文案 */
const MODERATION_HINT: Record<VendorErrorKind, string> = {
  video: '内容安全拦截：该视频提示词或参考图可能包含不安全内容。请编辑分镜视频提示词，避免暴力、血腥、敏感描述，或更换参考图后重试。',
  image: '内容安全拦截：该图片提示词或参考图可能包含不安全内容。请编辑关键帧/角色提示词，避免暴力、血腥、敏感描述后重试。',
  audio: '内容安全拦截：该配音文本可能包含不安全内容。请修改台词文本后重试。',
  text: '内容安全拦截：该提示词可能包含不安全内容。请修改后重试。',
}

/** 各场景的「上传图含人物触发审核」专属文案 */
const UPLOADS_MODERATION_HINT: Record<VendorErrorKind, string> = {
  video: '视频生成被平台内容审核拦截：原因指向「上传的参考图/首帧」（如角色图、分镜图）。平台会单独审核上传画面，与文字描述无直接关系。可尝试：去掉或更换参考图、避免画面中清晰可辨识的真实人物，或改为纯文生视频后重试。',
  image: '图片生成被平台内容审核拦截：原因指向「上传的参考图」。可尝试：去掉或更换参考图、避免画面中清晰可辨识的真实人物后重试。',
  audio: '语音生成被平台内容审核拦截。请检查台词语音文本后重试。',
  text: '生成被平台内容审核拦截。请检查提示词后重试。',
}

/** 识别「上传图含人物」类审核（与文字提示词无直接关系） */
function bodySuggestsPeopleInUserUploads(text: string): boolean {
  const t = (text || '').toLowerCase()
  return (
    t.includes('people-in-user-uploads') ||
    t.includes('people in user uploads') ||
    (t.includes('user-upload') && t.includes('people') && t.includes('moderation')) ||
    // 火山引擎 Seedance 隐私审核：首帧/参考图被识别为真实人物照片
    t.includes('inputimagesensitivecontentdetected') ||
    t.includes('input image sensitive content') ||
    t.includes('privacyinformation') ||
    t.includes('privacy information') ||
    (t.includes('sensitive') && t.includes('image') && (t.includes('detected') || t.includes('content'))) ||
    /图片.*(人物|人像|隐私|敏感)|(人物|人像).*图片|上传.*(人物|人像)|(人物|人像).*上传/.test(text || '')
  )
}

/** 识别「内容审核拦截」类错误（含文字与图片） */
function looksLikeContentModeration(apiErr: ParsedApiError | undefined, rawText: string, code?: string): boolean {
  const msg = (apiErr?.message || rawText || '').toLowerCase()
  const c = (code || apiErr?.code || '').toLowerCase()
  const type = (apiErr?.type || '').toLowerCase()
  if (c.includes('moderation') || type.includes('moderation') || msg.includes('moderation')) return true
  if (msg.includes('content_policy') || msg.includes('content policy')) return true
  if (msg.includes('safety') && (msg.includes('filter') || msg.includes('system'))) return true
  if (msg.includes('policy') && msg.includes('violation')) return true
  if (msg.includes('sensitive content') || c.includes('sensitivecontent')) return true
  return /不安全|违规|内容审核|敏感内容|风控|内容安全/.test(apiErr?.message || rawText || '')
}

function parseErrorBody(text: string): { error?: ParsedApiError; topCode?: string; topMessage?: string } | null {
  try {
    const obj = JSON.parse(text)
    if (obj && typeof obj === 'object') {
      const err = obj.error && typeof obj.error === 'object' ? obj.error : null
      return {
        error: err
          ? { message: typeof err.message === 'string' ? err.message : undefined, code: err.code, type: err.type }
          : undefined,
        topCode: obj.code,
        topMessage: typeof obj.message === 'string' ? obj.message : undefined,
      }
    }
    return null
  } catch {
    return null
  }
}

/** 去掉 Request id 等冗长后缀，避免刷屏 */
function stripRequestIdTail(s: string): string {
  return s.replace(/\s*Request id:\s*[a-f0-9]+.*$/i, '').trim()
}

/** 从英文 message 里尝试取出「模型名」片段，用于提示 */
function extractModelNameFromMessage(msg: string): string | null {
  const m =
    msg.match(/requested model\s+([^\s.]+)/i) ||
    msg.match(/model[`:：\s]+([a-z0-9_.-]+)/i) ||
    msg.match(/`model`\s*[^:]*:\s*([a-z0-9_.-]+)/i)
  return m ? m[1] : null
}

function clipMessage(rawMsg: string, max = 120): string {
  return rawMsg.length > max ? rawMsg.slice(0, max - 3).replace(/\s+\S*$/, '') + '…' : rawMsg
}

/**
 * 将上游 status + body 转为用户可读中文（不返回整段英文 JSON）。
 */
export function formatVendorHttpError(status: number, bodyText: string, kind: VendorErrorKind = 'video'): string {
  const trimmedBody = (bodyText || '').trim()
  const parsed = parseErrorBody(trimmedBody)
  const apiErr = parsed?.error
  const code = ((apiErr?.code || parsed?.topCode) || '').toLowerCase()
  const type = (apiErr?.type || '').toLowerCase()
  const rawMsg = stripRequestIdTail((apiErr?.message || parsed?.topMessage || '').trim())
  const msgLower = rawMsg.toLowerCase()
  const combinedForUploadCheck = `${trimmedBody}\n${rawMsg}`

  // 1. 内容审核拦截（含上传图归因细分）
  if (looksLikeContentModeration(apiErr, trimmedBody, code)) {
    if (bodySuggestsPeopleInUserUploads(combinedForUploadCheck)) {
      return UPLOADS_MODERATION_HINT[kind]
    }
    return MODERATION_HINT[kind]
  }

  // 2. 认证 / 权限 / 额度
  if (status === 401) return 'API Key 无效或未通过校验，请在 AI 服务配置中检查密钥是否正确。'
  if (status === 403) return '没有权限调用该能力，请检查账号是否已开通对应模型或接口。'
  if (status === 429) return '请求过于频繁或额度不足，请稍后再试。'

  // 3. 服务端瞬时异常
  if (status === 500 || status === 502 || status === 503 || status === 504) {
    return '当前请求较多或服务暂时异常，请稍后重试。'
  }

  // 4. 模型与接口不匹配（图片模型填视频栏、错误模型名）
  const isModelMismatch =
    status === 400 &&
    (code === 'invalidparameter' ||
      type === 'badrequest' ||
      msgLower.includes('does not support this api') ||
      (msgLower.includes('not support') && msgLower.includes('api')) ||
      msgLower.includes('invalidparameter') ||
      (msgLower.includes('parameter') && msgLower.includes('model') && msgLower.includes('valid')) ||
      msgLower.includes('not found') ||
      msgLower.includes('modelnotfound'))
  if (isModelMismatch) {
    const modelName = extractModelNameFromMessage(rawMsg)
    if (/seedream/i.test(rawMsg)) {
      return '当前使用的是豆包 Seedream（图片模型），不能用于生成视频。请在 AI 服务配置中改用 Seedance、MiniMax H3 等视频模型。'
    }
    if (msgLower.includes('not found') || msgLower.includes('modelnotfound')) {
      return modelName
        ? `找不到模型「${modelName}」，请核对 AI 服务配置中的模型 ID 是否填写正确。`
        : '找不到所填写的模型名称，请核对上游平台文档中的模型 ID 是否填写正确。'
    }
    if (modelName) {
      return `模型「${modelName}」不支持当前接口。请在 AI 服务配置中更换为支持该类型生成的模型。`
    }
    return '当前选择的模型与该接口不匹配。请检查 AI 服务配置中的「模型」是否为对应类型的模型（勿将图片模型填在视频栏）。'
  }

  // 5. 缺少必填参数
  if (status === 400 && (msgLower.includes('prompt') || msgLower.includes('parameter')) && msgLower.includes('required')) {
    return '缺少必填参数（如提示词或参考图），请检查任务参数是否完整。'
  }

  // 6. 有结构化 message：尽量不贴大段英文技术原文
  if (rawMsg && rawMsg.length > 0) {
    if (rawMsg.startsWith('{')) {
      return '服务返回了异常数据，请稍后重试或联系平台。'
    }
    const hasChinese = /[\u4e00-\u9fff]/.test(rawMsg)
    if (!hasChinese && rawMsg.length > 80) {
      return '服务拒绝了本次请求，常见原因：模型名称错误、该模型不支持当前接口、或参数不符合平台要求。请在 AI 服务配置中核对「模型」是否正确。'
    }
    const short = clipMessage(rawMsg)
    return `${KIND_LABEL[kind]}生成未成功：${short}`
  }

  // 7. 无法解析 JSON 时的纯文本 body
  if (!parsed && trimmedBody.length > 0 && !trimmedBody.startsWith('{')) {
    const plain = stripRequestIdTail(trimmedBody.replace(/\s+/g, ' '))
    const clip = clipMessage(plain)
    return `服务暂时无法完成请求（HTTP ${status}）。${clip ? `说明：${clip}` : '请稍后重试。'}`
  }

  return `${KIND_LABEL[kind]}生成失败（HTTP ${status}），请检查网络、API Key 与模型配置后重试。`
}

/**
 * 判断轮询过程中是否应立即判失败（而非继续重试）。
 *
 * 可重试：5xx 服务端瞬时错误、网络抖动、404（任务尚未就绪）。
 * 不可重试：401/403/429（认证/权限/额度，重试无意义）、
 *          400 + 审核拦截或参数/模型错误（重试不会改变结果）。
 */
export function isNonRetryableHttpError(status: number, bodyText: string): boolean {
  if (status === 401 || status === 403 || status === 429) return true
  if (status === 400) {
    const trimmed = (bodyText || '').trim()
    const parsed = parseErrorBody(trimmed)
    const apiErr = parsed?.error
    const code = ((apiErr?.code || parsed?.topCode) || '').toLowerCase()
    const rawMsg = ((apiErr?.message || parsed?.topMessage) || '').toLowerCase()
    const isModeration = looksLikeContentModeration(apiErr, trimmed, code)
    const isParamOrModelError =
      code === 'invalidparameter' ||
      rawMsg.includes('invalidparameter') ||
      rawMsg.includes('does not support this api') ||
      rawMsg.includes('model') ||
      rawMsg.includes('parameter') ||
      rawMsg.includes('prompt')
    return isModeration || isParamOrModelError
  }
  return false
}

/**
 * 读取 Response 的 body 后，构造带用户可读中文的 Error（不直接透传英文 JSON）。
 */
export async function vendorResponseError(resp: Response, kind: VendorErrorKind = 'video'): Promise<Error> {
  const bodyText = await resp.text()
  return new Error(formatVendorHttpError(resp.status, bodyText, kind))
}

/** 尽力把任意值转成字符串（对象无 message 时的兜底），JSON 序列化失败则退回 String() */
function safeStringify(v: unknown): string {
  try {
    return JSON.stringify(v)
  } catch {
    return String(v)
  }
}

/**
 * 将「轮询结果里的 error 字段」归因为用户可读中文。
 *
 * 与 formatVendorHttpError 的区别：轮询阶段拿不到 HTTP status，error 来自任务状态体
 * （HTTP 200 但 status=failed），且经常是厂商自定义结构 —— 可能是 string、可能是
 * `{ code, message }` 对象（火山引擎 Seedance 的 result.error 就是对象），甚至 JSON 字符串。
 *
 * 原先 `throw new Error(pollResp.error)` 遇到对象会退化成 `[object Object]`，
 * 把审核/失败原因全部吞掉，这里统一归一化 + 审核归因。
 */
export function formatVendorTaskError(raw: unknown, kind: VendorErrorKind = 'video'): string {
  let code = ''
  let message = ''
  let rawText = ''

  if (raw == null) {
    // 无 error 字段，走兜底
  } else if (typeof raw === 'string') {
    rawText = raw
    const parsed = parseErrorBody(raw)
    if (parsed?.error) {
      code = parsed.error.code || ''
      message = parsed.error.message || ''
    } else if (parsed?.topMessage) {
      message = parsed.topMessage
    } else {
      message = raw
    }
  } else if (typeof raw === 'object') {
    const obj = raw as Record<string, any>
    code = String(obj.code ?? obj.error_code ?? obj.err_code ?? obj.type ?? '')
    message = String(obj.message ?? obj.msg ?? obj.error_msg ?? obj.errorMessage ?? obj.detail ?? '')
    // 厂商 error 字段又套一层 error 对象的情况（{ error: { code, message } }）
    if (!message && typeof obj.error === 'string') message = obj.error
    if (!message && obj.error && typeof obj.error === 'object') {
      const e = obj.error as Record<string, any>
      code = code || String(e.code ?? e.error_code ?? '')
      message = String(e.message ?? e.msg ?? '')
    }
    rawText = message ? `${code} ${message}` : safeStringify(raw)
  }

  const combined = `${code}\n${message}\n${rawText}`

  // 1. 内容审核归因（含「上传图含人物」细分，复用 HTTP 层判断）
  const apiErr: ParsedApiError = { message: message || undefined, code: code || undefined }
  if (looksLikeContentModeration(apiErr, rawText, code)) {
    if (bodySuggestsPeopleInUserUploads(combined)) return UPLOADS_MODERATION_HINT[kind]
    return MODERATION_HINT[kind]
  }

  // 2. 提取可读 message 精简返回，避免贴大段英文技术原文
  const cleaned = stripRequestIdTail(message || rawText || '').trim()
  if (cleaned) {
    if (cleaned.startsWith('{') || cleaned.startsWith('[')) {
      return `${KIND_LABEL[kind]}生成失败，请稍后重试或检查提示词与参考图。`
    }
    const clip = clipMessage(cleaned)
    if (cleaned.length > 80 && !/[\u4e00-\u9fff]/.test(cleaned)) {
      return `${KIND_LABEL[kind]}生成失败，请稍后重试或检查提示词、参考图与模型配置。`
    }
    return `${KIND_LABEL[kind]}生成失败：${clip}`
  }

  return `${KIND_LABEL[kind]}生成失败，请稍后重试。`
}

// ── create 请求的指数退避重试 ─────────────────────────────────────────

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

/** create 阶段可退避重试的状态码：429 限流 / 5xx 瞬时异常 */
function isRetryableStatus(status: number): boolean {
  return status === 429 || status === 502 || status === 503 || status === 504 || status >= 500
}

/** 网络层错误（fetch reject）是否可退避重试 */
function isRetryableNetworkError(err: unknown): boolean {
  const msg = String((err as any)?.message ?? (err as any)?.name ?? err ?? '').toLowerCase()
  return (
    msg.includes('timeout') ||
    msg.includes('etimedout') ||
    msg.includes('econnreset') ||
    msg.includes('econnrefused') ||
    msg.includes('network') ||
    msg.includes('enotfound') ||
    msg.includes('eai_again') ||
    msg.includes('fetch failed') ||
    msg.includes('socket hang up') ||
    msg.includes('aborted')
  )
}

export interface FetchWithRetryOptions {
  /** 最大尝试次数（含首次），默认 3 */
  maxRetries?: number
  /** 首次退避延迟（毫秒），指数增长 2^n，默认 2000 */
  baseDelayMs?: number
  /** 单次请求超时（毫秒）；每次重试都会新建 signal，避免复用已 abort 的 signal */
  timeoutMs?: number
  /** 每次退避重试前的回调，用于记录结构化日志 */
  onRetry?: (attempt: number, delayMs: number, reason: string) => void
}

/**
 * 带指数退避重试的 fetch，用于「创建生成任务」请求（create 阶段）。
 *
 * 与轮询阶段的 isNonRetryableHttpError 区分：
 *   - create 阶段遇到 429（限流）/ 5xx（瞬时异常）/ 网络抖动，退避后重试同一模型是合理的；
 *   - 401/403/400 审核或参数错误则立即归因中文抛出，不浪费时间重试。
 *
 * 对齐参照项目 gcc-printfilm-main services/geminiService.ts 的 retryOperation：
 * 2s/4s/8s 指数退避，默认 3 次。多模型 fallback 外层仍会切换下一个模型兜底，
 * 因此单模型配置（models 数组只有一项）也能从「遇限流即失败」变为「短暂等待后自愈」。
 */
export async function fetchWithRetry(
  url: string,
  init: RequestInit,
  kind: VendorErrorKind = 'video',
  options: FetchWithRetryOptions = {},
): Promise<Response> {
  const { maxRetries = 3, baseDelayMs = 2000, timeoutMs, onRetry } = options

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      // 每次重试都重建 signal：AbortSignal.timeout 一旦触发即永久 abort，
      // 复用同一个 signal 会令后续重试立即失败。
      const signal = timeoutMs ? AbortSignal.timeout(timeoutMs) : init.signal
      const resp = await fetch(url, { ...init, signal })

      if (resp.ok) return resp

      // 非 2xx：可重试状态码则退避重试，否则（或最后一次）归因中文抛出
      if (isRetryableStatus(resp.status) && attempt < maxRetries - 1) {
        await resp.text().catch(() => {}) // 消费 body 以便连接复用
        const delayMs = baseDelayMs * 2 ** attempt
        onRetry?.(attempt + 1, delayMs, `HTTP ${resp.status}`)
        await sleep(delayMs)
        continue
      }
      throw await vendorResponseError(resp, kind)
    } catch (err: any) {
      // 网络层错误（fetch reject）：可重试则退避重试，否则原样抛出
      if (isRetryableNetworkError(err) && attempt < maxRetries - 1) {
        const delayMs = baseDelayMs * 2 ** attempt
        onRetry?.(attempt + 1, delayMs, err.message || String(err))
        await sleep(delayMs)
        continue
      }
      throw err
    }
  }

  throw new Error(`${KIND_LABEL[kind]}请求失败：已重试 ${maxRetries} 次仍无法完成。`)
}
