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
  '你是一位专业的角色视觉设定解析器。用户会给你一段角色的外貌特征描述，其中混杂了神态气质、外貌体型、服装穿着、武器装备、首饰配饰、随身器物等内容。\n' +
  '你的任务：只精确提取以下三个视觉字段，其余内容一律忽略：\n' +
  '（1）clothing 服装：身上穿着的衣物鞋帽（含服装配件），如「朴素古意的青色长衫」「玄色劲装」「白色西装」；\n' +
  '（2）weapons 武器装备：随身携带或使用的武器（含法器），如「三尺青锋」「龙首紫檀法杖」「长弓」；\n' +
  '（3）accessories 首饰配饰：佩戴在身上的饰品与发饰，如「简单束带」「白玉发簪」「蓝宝石项链」「玉佩」。\n' +
  '拆分规则：\n' +
  '1. 先通读全文，逐句扫描，凡涉及上述三类的信息必须全部提取，不得遗漏；\n' +
  '2. 提取时必须保留原文对物体的完整描述：名物词、数量词与修饰语（如「朴素古意的青色长衫」「三尺青锋，剑鞘古朴无华，却蕴含着慑人的锋芒」），只去掉「身着/身穿/头戴/手持/腰间别着/身旁横放」等动词引导词；\n' +
  '3. 不把神态、气质、环境、场景、动作过程带进任何字段，如「沉静如枯木」「与周围格格不入」「快逾闪电」应忽略；\n' +
  '4. 武器配件（剑鞘/刀鞘/箭袋等）不单独拆成独立条目，但属于武器细节的描述（如「剑鞘古朴无华」「却蕴含着慑人的锋芒」）要保留在 weapons 字段中；\n' +
  '5. 发带/发簪/耳环/项链/玉佩等佩戴类饰品一律归入 accessories，不归入 clothing；腰带/靴子/帽子归入 clothing；\n' +
  '6. 法器/法杖/飞剑等进攻性器物归入 weapons；折扇、酒葫芦、拂尘、罗盘、乐器、书卷等非武器器物既不属于服装也不属于配饰，一律忽略、不输出；\n' +
  '7. 同一类出现多项时用中文逗号分隔；原文未提及某类则输出空字符串，绝不编造；\n' +
  '8. 只输出 JSON：{"clothing":"","weapons":"","accessories":""}，禁止 markdown 代码块和任何解释。'

/** 拆分 few-shot 示例：贴近真实长文本（神态/环境与装备混排），让模型模仿「只留名物短语」的输出风格 */
const SPLIT_VISUALS_EXAMPLES =
  '示例 1：\n' +
  '输入：「少年身着朴素古意的青色长衫，长发以简单束带松松挽于脑后，腰间别着一柄三尺青锋长剑，手中把玩着一把白面折扇。」\n' +
  '输出：{"clothing":"朴素古意的青色长衫","weapons":"三尺青锋长剑","accessories":"简单束带"}\n\n' +
  '示例 2：\n' +
  '输入：「老者鹤发童颜，双目精光湛湛，一副仙风道骨模样。身穿玄色道袍，腰系玉带，脚蹬云纹靴。右手拄着一根龙首紫檀法杖，杖身刻满符文，隐隐泛着灵光，左手还托着一个刻满铭文的黄铜罗盘。左腕戴着一串佛珠，偶尔转动。」\n' +
  '输出：{"clothing":"玄色道袍，玉带，云纹靴","weapons":"龙首紫檀法杖","accessories":"佛珠"}\n\n' +
  '示例 3：\n' +
  '输入：「她踩着高跟鞋走进会场，一袭白色晚礼服裙摆曳地，颈间坠着蓝宝石项链，耳垂上悬着细钻耳环。手提包里藏着一把袖珍手枪，作为最后防身之物。」\n' +
  '输出：{"clothing":"白色晚礼服，高跟鞋","weapons":"袖珍手枪","accessories":"蓝宝石项链，细钻耳环"}\n\n' +
  '示例 4：\n' +
  '输入：「少年身着一袭朴素古意的青色长衫，长发以简单束带松松挽于脑后，身旁横放一柄三尺青锋，剑鞘古朴无华，却蕴含着慑人的锋芒，腰间别着一只温润的羊脂白玉酒葫芦。」\n' +
  '输出：{"clothing":"朴素古意的青色长衫","weapons":"三尺青锋，剑鞘古朴无华，却蕴含着慑人的锋芒","accessories":"简单束带"}\n'

// ====== 本地规则兜底：AI 输出异常或漏字段时按关键词从原文提取 ======
// 关键词先长后短：优先命中完整名物词，避免「衣/剑/枪」等单字提前匹配
const SPLIT_CLOTHING_KEYWORDS = ['长衫', '汉服', '唐装', '戏服', '劲装', '布衣', '古衣', '战甲', '燕尾服', '西装', '礼服', '外套', '大衣', '风衣', '马甲', '衬衫', '卫衣', '牛仔裤', '短裤', '披风', '斗篷', '长裙', '围巾', '领带', '腰带', '护腕', '皮鞋', '运动鞋', '靴', '帽', '裙', '袍', '衫', '衣', '裤', '甲', '铠', '盔']
const SPLIT_WEAPON_KEYWORDS = ['狼牙棒', '飞镖', '暗器', '火铳', '狙击枪', '冲锋枪', '机枪', '手枪', '步枪', '长剑', '青锋', '佩剑', '短剑', '匕首', '法杖', '长弓', '弓箭', '弓弩', '剑匣', '剑', '刀', '棍', '枪', '矛', '戟', '锤', '斧', '盾', '鞭', '锏', '弩', '杖']
// 武器屏蔽词：命中时说明该句在描述「出剑/挥剑」等动作过程而非武器本体
const SPLIT_WEAPON_BANNED = ['出剑', '拔剑', '挥剑', '使剑', '用剑', '剑尖', '剑光']
// 神态/环境/动作过程杂质词：命中时说明该子句与物品本体无关，合并邻近句时剔除
const SPLIT_NOISE_KEYWORDS = ['气质', '神态', '神情', '神色', '眼神', '目光', '面色', '面容', '模样', '场景', '环境', '格格不入', '气势', '沉静', '淡漠', '精光', '不敢直视', '出剑', '拔剑', '挥剑', '使剑', '用剑', '剑尖', '剑光']
const SPLIT_ACCESSORY_KEYWORDS = ['束发带', '发带', '发簪', '玉簪', '头冠', '凤冠', '王冠', '璎珞', '香囊', '荷包', '项链', '吊坠', '挂坠', '戒指', '扳指', '手串', '手链', '手镯', '脚链', '耳环', '耳钉', '花钿', '纶巾', '玉佩', '红绳', '簪', '钗', '冠', '束带']
// 随身器物（非武器、非穿戴，如折扇/酒葫芦/罗盘）：不再作为独立字段输出，仅用于阻止其污染服装/武器/首饰的合并片段
const SPLIT_PROP_NOISE_KEYWORDS = ['羊脂白玉酒葫芦', '芭蕉扇', '油纸伞', '紫金葫芦', '酒葫芦', '乾坤袋', '储物袋', '褡裢', '折扇', '团扇', '蒲扇', '羽扇', '拂尘', '罗盘', '花灯', '宫灯', '灯笼', '玉箫', '洞箫', '长箫', '横笛', '竹笛', '牧笛', '玉笛', '古琴', '瑶琴', '七弦琴', '焦尾琴', '琵琶', '古筝', '三弦', '书卷', '画卷', '书简', '卷轴', '经卷', '经书', '铜镜', '古镜', '香炉', '丹炉', '砚台', '算盘', '玉如意', '铜铃', '木鱼', '药瓶', '瓷瓶', '玉瓶', '酒坛', '茶壶', '酒樽', '金樽', '夜光杯', '水囊', '钱袋', '钱囊', '锦囊', '包袱', '药箱', '药篓', '手帕', '绣帕', '信笺', '令牌', '玉玺', '印玺', '官印', '圣旨', '如意']
// 名物清洗时可去掉的常见动词/数量词引导与冗余后缀（先长后短）
const SPLIT_REDUNDANT_PREFIXES = ['腰间别着一柄', '腰间别着', '腰间悬着一柄', '腰间悬着', '腰间佩着', '手里拿着', '手里拄着', '手执一柄', '手执', '手持一柄', '手持', '手中握着', '手上戴', '背后背着', '肩上扛着', '身旁横放一柄', '身旁横放', '横放', '头上戴', '头戴', '脚蹬一双', '脚蹬', '身着', '穿着', '身穿', '身披', '披着', '腰系', '腰缠', '长发以', '手腕上系着', '悬着', '挎着', '别着', '佩着', '戴着', '佩戴', '腰挂', '腰佩', '颈间坠着', '颈间戴着', '耳垂上悬着', '一手拄着', '拄着', '一串', '一柄', '一把', '一根', '一支', '一件', '一袭', '一身', '背着', '拿着', '握着', '脑后']
const SPLIT_REDUNDANT_SUFFIXES = ['松松挽于脑后', '挽于脑后', '松松地挽着', '松松挽着', '松散地挽着', '斜挎在腰', '别在腰间', '挂在腰间', '挎在腰间', '握在手中', '提在手中', '垂在身侧', '拄于地面', '立在地上', '悬在腰间', '佩在腰间', '裙摆曳地', '裙摆轻摇', '无风自动', '随风飘动']

/**
 * 清洗单条视觉片段：去掉动词/数量词引导与冗余后缀，保留名物短语。
 * 兼容逗号分隔的多词值（逐段清洗后重新拼接，不破坏分隔）。
 */
function cleanVisualFragment(raw: string): string {
  const parts = (raw || '').split(/[，,]/).map((p) => {
    let fragment = p.replace(/[。；：、\n]/g, '').trim()
    if (!fragment) return ''
    // 循环去掉引导词（如「穿着一身」→ 依次去掉「穿着」「一身」）
    let changed = true
    while (changed) {
      changed = false
      for (const prefix of SPLIT_REDUNDANT_PREFIXES) {
        if (fragment.startsWith(prefix)) { fragment = fragment.slice(prefix.length).trim(); changed = true; break }
      }
    }
    for (const suffix of SPLIT_REDUNDANT_SUFFIXES) {
      if (fragment.endsWith(suffix)) { fragment = fragment.slice(0, -suffix.length).trim(); break }
    }
    return fragment
  }).filter(Boolean)
  return parts.join('，')
}

function splitVisualsByRules(appearance: string): { clothing: string; weapons: string; accessories: string } {
  const sentences = appearance.split(/[，。；、\n,;：:！!？?]/).map(s => s.trim()).filter(Boolean)
  const pick = (keywords: string[], banned: string[] = [], otherKeywords: string[] = []): string => {
    const items: string[] = []
    for (let i = 0; i < sentences.length; i++) {
      const s = sentences[i]
      const hit = keywords.find(k => s.includes(k))
      if (!hit) continue
      if (banned.some(b => s.includes(b)) && hit.length <= 2) continue
      // 同类已提取过 → 跳过（如原文重复粘贴同一武器）
      if (items.some(it => it.includes(hit))) continue
      // 合并前后相邻的描述性子句：遇神态/环境/动作杂质词或他类关键词即停，
      // 保留完整物品描述（如「三尺青锋，剑鞘古朴无华，却蕴含着慑人的锋芒」）
      const fragParts: string[] = [s]
      let total = s.length
      const canMerge = (t: string) =>
        !SPLIT_NOISE_KEYWORDS.some(k => t.includes(k)) && !otherKeywords.some(k => t.includes(k))
      for (let j = i - 1; j >= 0; j--) {
        const adj = sentences[j]
        if (!canMerge(adj)) break
        if (total + adj.length + 1 > 34) break
        total += adj.length + 1
        fragParts.unshift(adj)
      }
      for (let j = i + 1; j < sentences.length; j++) {
        const adj = sentences[j]
        if (!canMerge(adj)) break
        if (total + adj.length + 1 > 34) break
        total += adj.length + 1
        fragParts.push(adj)
      }
      const cleaned = cleanVisualFragment(fragParts.join('，'))
      if (cleaned && !items.includes(cleaned)) items.push(cleaned)
    }
    return items.join('，')
  }
  const allOther = [...SPLIT_CLOTHING_KEYWORDS, ...SPLIT_WEAPON_KEYWORDS, ...SPLIT_ACCESSORY_KEYWORDS, ...SPLIT_PROP_NOISE_KEYWORDS]
  return {
    clothing: pick(SPLIT_CLOTHING_KEYWORDS, [], allOther),
    weapons: pick(SPLIT_WEAPON_KEYWORDS, SPLIT_WEAPON_BANNED, [...SPLIT_CLOTHING_KEYWORDS, ...SPLIT_ACCESSORY_KEYWORDS, ...SPLIT_PROP_NOISE_KEYWORDS]),
    accessories: pick(SPLIT_ACCESSORY_KEYWORDS, [], [...SPLIT_CLOTHING_KEYWORDS, ...SPLIT_WEAPON_KEYWORDS, ...SPLIT_PROP_NOISE_KEYWORDS]),
  }
}

/**
 * 从「外貌特征」文本智能拆分服装/武器/首饰三个字段，
 * 用于角色详情页「智能拆分」按钮自动回填，减少重复填写。
 *
 * 融合策略（保证完整描述 + 防幻觉）：
 * 1. 本地规则先拆：名物短语精确、无动词引导/神态杂质（对常见词覆盖稳定）；
 * 2. AI 增强：AI 能保留原文完整物品描述（含剑鞘/锋芒等细节），覆盖规则词表盲区；
 * 3. 每字段「AI 优先，规则兜底」：AI 值逐段校验均来自原文才采用（防「三尺青芒」类幻觉），否则回退规则值。
 */
export async function splitCharacterVisuals(input: SplitCharacterVisualsInput): Promise<{ clothing: string; weapons: string; accessories: string }> {
  const appearance = (input.appearance || '').trim()
  if (!appearance) throw new Error('外貌特征为空')

  // 规则先拆：稳定、精确、无杂质
  const ruleResult = splitVisualsByRules(appearance)

  const userPrompt =
    SPLIT_VISUALS_EXAMPLES +
    `现在拆分以下角色描述：\n${appearance}\n\n请严格按规则只输出 JSON。`

  logTaskStart('SplitVisuals', 'generate', {})
  let ai: { clothing?: unknown; weapons?: unknown; accessories?: unknown } = {}
  try {
    const result = await generateText(userPrompt, { system: SPLIT_VISUALS_SYSTEM_PROMPT, temperature: 0.1, maxTokens: 300 })
    ai = JSON.parse(cleanJsonString(result))
    logTaskProgress('SplitVisuals', 'ai', { ...ai })
  } catch (err: any) {
    // AI 输出异常 → 直接使用规则结果，保证按钮始终有输出
    logTaskProgress('SplitVisuals', 'fallback', { reason: 'ai-failed', error: err.message })
  }

  // AI 值统一清洗：去掉动词引导/冗余后缀
  const cleanAi = {
    clothing: cleanVisualFragment(String(ai.clothing || '')),
    weapons: cleanVisualFragment(String(ai.weapons || '')),
    accessories: cleanVisualFragment(String(ai.accessories || '')),
  }

  // AI 优先 + 原文一致性校验：AI 值每个名物片段必须能在原文中找到（防「三尺青芒」类幻觉），否则回退规则值
  const appearsInSource = (value: string): boolean => {
    const segs = value.split(/[，,]/).map(s => s.trim()).filter(Boolean)
    return segs.length > 0 && segs.every(seg => seg.length >= 2 && appearance.includes(seg))
  }
  const out = {
    clothing: appearsInSource(cleanAi.clothing) ? cleanAi.clothing : ruleResult.clothing,
    weapons: appearsInSource(cleanAi.weapons) ? cleanAi.weapons : ruleResult.weapons,
    accessories: appearsInSource(cleanAi.accessories) ? cleanAi.accessories : ruleResult.accessories,
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
