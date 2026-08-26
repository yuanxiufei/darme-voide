import { getTextConfig } from './ai.js'
import { fetchWithRetry, vendorResponseError } from '../utils/vendor-errors.js'
import { logTaskError, logTaskProgress, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import { getTextAdapter } from './adapters/registry.js'
import { gpuManager, isLocalConfig, type GpuLease } from './gpu-manager.js'

interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface GenerateTextOptions {
  temperature?: number
  maxTokens?: number
  system?: string
}

/**
 * 轻量文本生成：通过 Provider Adapter 层调用 text 配置对应厂商，
 * 支持多模型自动 fallback。用于「动作建议」这类无需 Agent 工具链的简单文本任务，
 * 避免为一次性需求新增 Mastra Agent + 协议契约的复杂度。
 */
export async function generateText(userPrompt: string, options: GenerateTextOptions = {}): Promise<string> {
  const config = getTextConfig()
  const models = (config.models && config.models.length > 0)
    ? config.models
    : [config.model].filter(Boolean)

  if (models.length === 0) throw new Error('文本模型未配置 — 请在设置中添加文本服务')

  const adapter = getTextAdapter(config.provider)
  const isLocal = isLocalConfig(config.baseUrl, config.provider)
  const messages: ChatMessage[] = []
  if (options.system) messages.push({ role: 'system', content: options.system })
  messages.push({ role: 'user', content: userPrompt })

  let lastError: Error | null = null

  for (const model of models) {
    // 本地 GPU 文本模型（如 Ollama qwen3）：获取显存租约，参与模型启动/卸载调度
    let lease: GpuLease | null = null
    if (isLocal) {
      lease = await gpuManager.acquire('text', config.provider, model, config.baseUrl)
    }

    try {
      const request = adapter.buildRequest(config, {
        model,
        messages,
        temperature: options.temperature,
        maxTokens: options.maxTokens,
      })

      const resp = await fetchWithRetry(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body ? JSON.stringify(request.body) : undefined,
      }, 'text', { timeoutMs: isLocal ? 180_000 : 60_000, maxRetries: isLocal ? 1 : 3 })

      if (!resp.ok) {
        lastError = await vendorResponseError(resp, 'text')
        logTaskError('TextGen', 'http-error', { model, error: lastError.message })
        continue
      }

      const data = await resp.json()
      const content = adapter.parseResponse(data)
      if (!content) {
        lastError = new Error('文本模型返回为空')
        logTaskError('TextGen', 'empty-response', { model })
        continue
      }

      logTaskSuccess('TextGen', 'done', { model, chars: content.length })
      return content.trim()
    } catch (err: any) {
      lastError = err
      logTaskError('TextGen', 'model-error', { model, error: err.message })
    } finally {
      if (lease) lease.release()
    }
  }

  throw lastError || new Error('文本生成失败')
}

/** 分镜动作建议输入 */
export interface ActionSuggestionInput {
  title?: string | null
  description?: string | null
  action?: string | null
  imagePrompt?: string | null
  atmosphere?: string | null
  shotType?: string | null
  movement?: string | null
  angle?: string | null
}

const ACTION_SYSTEM_PROMPT =
  '你是一名专业的短视频分镜导演。根据给定的镜头信息，为这个镜头设计 1-2 句简洁的' +
  '中文运镜与动作建议，直接可用于视频生成提示词。要求：' +
  '（1）聚焦画面内的主体动作与镜头运动，例如推/拉/摇/移/跟、慢动作、特写推进等；' +
  '（2）语言精炼，不超过 30 字；' +
  '（3）只输出建议文本本身，不要解释、不要序号、不要引号。'

/** 生成分镜动作/运镜建议 */
export async function generateActionSuggestion(input: ActionSuggestionInput): Promise<string> {
  const parts: string[] = []
  if (input.title) parts.push(`镜头标题：${input.title}`)
  if (input.shotType) parts.push(`景别：${input.shotType}`)
  if (input.angle) parts.push(`拍摄角度：${input.angle}`)
  if (input.atmosphere) parts.push(`氛围：${input.atmosphere}`)
  const sceneDesc = input.imagePrompt || input.description
  if (sceneDesc) parts.push(`画面内容：${sceneDesc}`)
  if (input.action) parts.push(`已有动作描述：${input.action}`)
  if (input.movement) parts.push(`已有运镜：${input.movement}`)

  const userPrompt = parts.length > 0
    ? parts.join('\n') + '\n\n请为以上镜头生成运镜与动作建议。'
    : '请为这个镜头生成运镜与动作建议。'

  logTaskStart('ActionSuggestion', 'generate', { title: input.title || '' })
  const result = await generateText(userPrompt, { system: ACTION_SYSTEM_PROMPT, temperature: 0.8, maxTokens: 120 })
  logTaskProgress('ActionSuggestion', 'generated', { length: result.length })
  return result
}

/** 子镜头结构 */
export interface SubShot {
  shotSize: string
  cameraMovement: string
  actionSummary: string
  visualFocus: string
}

/** 拆分镜头输入 */
export interface ShotSplitInput {
  title?: string | null
  action?: string | null
  description?: string | null
  shotType?: string | null
  atmosphere?: string | null
  dialogue?: string | null
  sceneInfo?: { location: string; time: string; atmosphere: string }
  characterNames?: string[]
  visualStyle?: string
}

const SPLIT_SYSTEM_PROMPT =
  '你是一位专业的电影分镜师。你的任务是把一个粗略的镜头描述，拆分为多个细致、专业的子镜头。' +
  '每个子镜头只负责一个视角或动作细节，时长约 2-4 秒。' +
  '合理运用远景、全景、中景、近景、特写等不同景别，子镜头之间保持叙事连贯。' +
  '只输出 JSON，不要包含任何解释或 markdown 代码块标记。'

function cleanJsonString(raw: string): string {
  let s = raw.trim()
  // 去掉 ```json ... ``` 包裹
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)```/i)
  if (fence) s = fence[1].trim()
  const start = s.indexOf('{')
  const end = s.lastIndexOf('}')
  if (start >= 0 && end > start) s = s.slice(start, end + 1)
  return s
}

/** 将一个粗略镜头拆分为多个子镜头 */
export async function splitShotIntoSubShots(input: ShotSplitInput): Promise<SubShot[]> {
  const styleDesc = input.visualStyle || '电影写实风格'
  const lines: string[] = []
  if (input.sceneInfo) {
    lines.push(`场景地点：${input.sceneInfo.location}`)
    lines.push(`场景时间：${input.sceneInfo.time}`)
    if (input.sceneInfo.atmosphere) lines.push(`场景氛围：${input.sceneInfo.atmosphere}`)
  }
  if (input.title) lines.push(`镜头标题：${input.title}`)
  if (input.shotType) lines.push(`原始景别：${input.shotType}`)
  if (input.characterNames && input.characterNames.length) lines.push(`出场角色：${input.characterNames.join('、')}`)
  lines.push(`视觉风格：${styleDesc}`)
  const sceneDesc = input.description || input.action
  if (sceneDesc) lines.push(`原始动作/画面描述：${sceneDesc}`)
  if (input.dialogue) lines.push(`对白：${input.dialogue}（请将对白放入最合适的子镜头，通常是角色说话的中景或近景）`)
  lines.push('')
  lines.push('请将以上镜头拆分为 2-5 个子镜头，输出 JSON：')
  lines.push('{"subShots":[{"shotSize":"全景","cameraMovement":"静止","actionSummary":"60-100字的动作与画面描述","visualFocus":"视觉焦点"}]}')

  const userPrompt = lines.join('\n')

  logTaskStart('ShotSplit', 'generate', { title: input.title || '' })
  const result = await generateText(userPrompt, { system: SPLIT_SYSTEM_PROMPT, temperature: 0.8, maxTokens: 2000 })

  let parsed: { subShots?: SubShot[] }
  try {
    parsed = JSON.parse(cleanJsonString(result))
  } catch {
    throw new Error('AI 返回的拆分结果不是有效 JSON')
  }

  const subShots = parsed.subShots
  if (!subShots || !Array.isArray(subShots) || subShots.length === 0) {
    throw new Error('AI 拆分结果为空')
  }
  for (const s of subShots) {
    if (!s.shotSize || !s.actionSummary) {
      throw new Error('子镜头缺少必要字段（shotSize / actionSummary）')
    }
  }

  logTaskSuccess('ShotSplit', 'generated', { count: subShots.length })
  return subShots
}

/** 续写剧本输入 */
export interface ContinueScriptInput {
  /** 已有内容（原始剧本或格式化剧本） */
  text: string
  /** 续写类型：raw 原始剧本 / script 格式化剧本 */
  mode: 'raw' | 'script'
}

const CONTINUE_RAW_SYSTEM_PROMPT =
  '你是一位专业的短剧编剧。根据用户提供的已有剧本内容，自然地续写后续剧情。要求：' +
  '（1）延续已有的文风、人称、叙事节奏与角色语气，保持前后一致；' +
  '（2）剧情推进合理、有新情节发展，不重复已写内容；' +
  '（3）若已有内容以对白或台词收尾，续写应包含新的情节推进或对白；' +
  '（4）只输出续写正文本身，不要解释、不要"续写："之类的前缀、不要加引号包裹。'

const CONTINUE_SCRIPT_SYSTEM_PROMPT =
  '你是一位专业的短剧编剧。根据用户提供的已有格式化剧本（场次剧本）内容，自然地续写后续场次。要求：' +
  '（1）严格延续已有的场次格式（场号、地点、时间、角色、对白/动作的排版结构）；' +
  '（2）延续已有角色的性格与说话语气，剧情推进合理；' +
  '（3）只输出续写的场次正文，不要解释、不要加 markdown 代码块、不要加"续写："前缀。'

/** AI 续写剧本（原始内容或格式化剧本） */
export async function continueScript(input: ContinueScriptInput): Promise<string> {
  const trimmed = (input.text || '').trim()
  if (!trimmed) throw new Error('内容为空，无法续写')

  // 只取末尾 2400 字作为上下文，控制 token 且保证前后衔接
  const tail = trimmed.length > 2400 ? trimmed.slice(-2400) : trimmed
  const system = input.mode === 'script' ? CONTINUE_SCRIPT_SYSTEM_PROMPT : CONTINUE_RAW_SYSTEM_PROMPT

  const userPrompt = `以下是已有内容：\n\n${tail}\n\n请直接继续往下写，输出续写内容。`

  logTaskStart('ContinueScript', 'generate', { mode: input.mode, length: trimmed.length })
  const result = await generateText(userPrompt, { system, temperature: 0.85, maxTokens: 1200 })
  logTaskSuccess('ContinueScript', 'generated', { length: result.length })
  return result
}

/** 优化视频提示词输入 */
export interface OptimizePromptInput {
  /** 用户当前已填写的提示词（可能简短、粗糙或为空） */
  currentPrompt?: string | null
  title?: string | null
  action?: string | null
  description?: string | null
  shotType?: string | null
  movement?: string | null
  atmosphere?: string | null
  sceneInfo?: { location: string; time: string; atmosphere: string }
  characterNames?: string[]
  visualStyle?: string
}

const OPTIMIZE_PROMPT_SYSTEM_PROMPT =
  '你是一位专业的 AI 视频生成提示词优化师。用户会给你一段视频生成提示词（可能简短、粗糙、缺少细节），' +
  '以及该镜头的辅助信息（场景、角色、景别、运镜、风格等）。你的任务是把它们融合、扩写成一段专业、' +
  '可直接用于视频生成模型的高质量中文提示词。要求：' +
  '（1）保留用户原始意图，不新增用户未指定的剧情或动作；' +
  '（2）补全并细化画面主体、动作、镜头运动、景别、光线、氛围、节奏等电影化细节；' +
  '（3）融合提供的场景地点/时间/氛围、出场角色名、视觉风格；' +
  '（4）语言精炼流畅，总长控制在 200 字以内；' +
  '（5）只输出优化后的提示词正文本身，不要解释、不要加「优化后」之类前缀、不要引号包裹。'

/**
 * 分镜视频提示词（含时间轴分段 DSL）专用优化指令：
 * 分镜 agent 生成的 video_prompt 用 <n> 分隔时间段（0-3秒/3-6秒…），并用
 * <location>/<role>/<voice> 标签承载地点/角色/配音，这些是程序解析依赖的结构，
 * 优化时严禁抹平，只能对每个时间段内部的描述文字做润色扩写。
 */
const OPTIMIZE_TIMELINE_SYSTEM_PROMPT =
  '你是一位专业的 AI 视频生成提示词优化师。用户会给你一段「分镜视频提示词」，它由多个时间段（例如「0-3秒」「3-6秒」）组成，' +
  '用 <n> 作为分隔符，并可能包含 <location>（地点）、<role>（角色）、<voice>（配音）等结构化标签；' +
  '这些时间段与标签是程序解析依赖的结构，必须原样保留。' +
  '你的任务是：只对每个时间段内部的画面描述文字做扩写与润色，补全画面主体、动作、镜头运动、景别、光线、氛围、节奏等电影化细节，并融合辅助信息中的场景/角色/风格。' +
  '硬性要求：' +
  '（1）所有时间段标注（如 0-3秒、3-6秒、6-9秒）原样保留，不得新增、删除、合并或改动时间段数量与顺序；' +
  '（2）所有 <n> 分隔符、<location>…</location>、<role>…</role>、<voice>…</voice> 标签原样保留；' +
  '（3）只扩写各段内的描述文字，不改变各段叙事顺序与剧情；' +
  '（4）语言精炼，每段控制在 90 字以内，不新增用户未指定的剧情或动作；' +
  '（5）只输出优化后的提示词正文本身，不要解释、不要加前缀、不要引号包裹。'

/**
 * 优化视频生成提示词（对齐 gcc KeyframeEditor 的 AI 优化提示词）。
 * 注意：这是用户主动点击「AI 优化」触发的增强，与项目约定「审核失败时不自动改写提示词」无关。
 */
export async function optimizeVideoPrompt(input: OptimizePromptInput): Promise<string> {
  const lines: string[] = []
  if (input.sceneInfo) {
    lines.push(`场景地点：${input.sceneInfo.location}`)
    if (input.sceneInfo.time) lines.push(`场景时间：${input.sceneInfo.time}`)
    if (input.sceneInfo.atmosphere) lines.push(`场景氛围：${input.sceneInfo.atmosphere}`)
  }
  if (input.characterNames && input.characterNames.length) lines.push(`出场角色：${input.characterNames.join('、')}`)
  if (input.title) lines.push(`镜头标题：${input.title}`)
  if (input.shotType) lines.push(`景别：${input.shotType}`)
  if (input.movement) lines.push(`运镜：${input.movement}`)
  if (input.atmosphere) lines.push(`氛围：${input.atmosphere}`)
  if (input.visualStyle) lines.push(`视觉风格：${input.visualStyle}`)
  const sceneDesc = input.description || input.action
  if (sceneDesc) lines.push(`画面/动作描述：${sceneDesc}`)
  lines.push('')
  if (input.currentPrompt && input.currentPrompt.trim()) {
    lines.push(`用户当前提示词：\n${input.currentPrompt.trim()}`)
  } else {
    lines.push('用户当前未填写提示词，请根据以上信息生成。')
  }
  lines.push('')
  lines.push('请输出优化后的视频生成提示词。')

  logTaskStart('OptimizePrompt', 'generate', { title: input.title || '' })
  // 检测用户当前提示词是否为「分镜视频提示词」（含时间轴分段 DSL），若是则走保留结构的专用指令
  const current = input.currentPrompt?.trim() || ''
  const hasTimelineDsl = /<n\s*\/?>|<\/?(?:location|role|voice)>|\d+\s*-\s*\d+\s*秒/i.test(current)
  const result = await generateText(lines.join('\n'), {
    system: hasTimelineDsl ? OPTIMIZE_TIMELINE_SYSTEM_PROMPT : OPTIMIZE_PROMPT_SYSTEM_PROMPT,
    temperature: 0.7,
    maxTokens: hasTimelineDsl ? 900 : 400,
  })
  logTaskSuccess('OptimizePrompt', 'generated', { length: result.length })
  return result
}

/** 智能拆分角色视觉信息输入 */
export interface SplitCharacterVisualsInput {
  appearance?: string | null
}

const SPLIT_VISUALS_SYSTEM_PROMPT =
  '你是一位专业的角色设定分析师。用户会给你一段角色的「外貌特征」描述，' +
  '这段描述里可能混杂了角色的服装穿着、武器装备、首饰配饰等信息。' +
  '请把它们拆分成三个独立字段：' +
  '（1）clothing 服装风格：角色的穿着，如「青色古风长衫」；' +
  '（2）weapons 武器装备：角色携带或使用的武器，如「三尺青锋长剑」；' +
  '（3）accessories 首饰配饰：角色佩戴的首饰、发饰、束发带等配饰。' +
  '要求：只输出 JSON，格式为 {"clothing":"","weapons":"","accessories":""}；' +
  '只提炼原文确实存在的描述，不得凭空新增；每项措辞精简，不超过 20 字；' +
  '若原文未提及某类信息，对应字段输出空字符串；' +
  '不要输出任何解释或 markdown 代码块标记。'

/**
 * 从「外貌特征」文本智能拆分服装/武器/首饰三个字段，
 * 用于角色详情页「智能拆分」按钮自动回填，减少重复填写。
 */
export async function splitCharacterVisuals(input: SplitCharacterVisualsInput): Promise<{ clothing: string; weapons: string; accessories: string }> {
  const appearance = (input.appearance || '').trim()
  if (!appearance) throw new Error('外貌特征为空')

  const userPrompt = `角色外貌特征描述：\n${appearance}\n\n请拆分出 clothing / weapons / accessories 三个字段。`

  logTaskStart('SplitVisuals', 'generate', {})
  const result = await generateText(userPrompt, { system: SPLIT_VISUALS_SYSTEM_PROMPT, temperature: 0.2, maxTokens: 300 })

  let parsed: { clothing?: unknown; weapons?: unknown; accessories?: unknown }
  try {
    parsed = JSON.parse(cleanJsonString(result))
  } catch {
    throw new Error('AI 返回的拆分结果不是有效 JSON')
  }

  const out = {
    clothing: String(parsed.clothing || '').trim(),
    weapons: String(parsed.weapons || '').trim(),
    accessories: String(parsed.accessories || '').trim(),
  }
  logTaskSuccess('SplitVisuals', 'generated', out)
  return out
}

/** 音色角色类型标签（4 类，与前端 ROLE_TAGS 对齐） */
export const VOICE_ROLE_TAGS = ['旁白', '主角', '反派', '配角'] as const

const VOICE_TAG_SYSTEM_PROMPT =
  '你是专业的配音导演，擅长判断一个音色适合配音的角色类型。' +
  '可选标签只有 4 个：旁白、主角、反派、配角。' +
  '请根据音色的名称和官方描述判断，为每个音色打 1-3 个最贴切的标签。' +
  '只输出 JSON，格式为 {"voice_id":["标签1","标签2"],...}，' +
  'voice_id 必须与输入完全一致，不要包含任何解释或 markdown 代码块标记。'

export interface VoiceRoleTagInput {
  voiceId: string
  voiceName: string
  description: string[]
}

/**
 * 批量推断音色适合的角色类型，返回 { voice_id: string[] }。
 * 用于音色库按角色类型筛选，替代前端纯正则推断。
 */
export async function inferVoiceRoleTags(voices: VoiceRoleTagInput[]): Promise<Record<string, string[]>> {
  if (!voices.length) return {}

  const list = voices.map((v) => {
    const desc = (v.description || []).join('、')
    return `- voice_id=${v.voiceId}，名称=${v.voiceName}${desc ? `，描述=${desc}` : ''}`
  }).join('\n')

  const userPrompt = `请为以下音色打角色类型标签：\n${list}\n\n标签只能从「旁白 / 主角 / 反派 / 配角」中选择。`

  logTaskStart('VoiceTag', 'generate', { count: voices.length })
  const result = await generateText(userPrompt, { system: VOICE_TAG_SYSTEM_PROMPT, temperature: 0.1, maxTokens: 2000 })

  let parsed: Record<string, unknown>
  try {
    parsed = JSON.parse(cleanJsonString(result))
  } catch {
    throw new Error('AI 返回的打标结果不是有效 JSON')
  }

  const out: Record<string, string[]> = {}
  for (const [id, tags] of Object.entries(parsed)) {
    const arr = Array.isArray(tags)
      ? tags.filter((t) => (VOICE_ROLE_TAGS as readonly string[]).includes(String(t)))
      : []
    if (arr.length) out[id] = arr
  }
  logTaskSuccess('VoiceTag', 'generated', { tagged: Object.keys(out).length })
  return out
}
