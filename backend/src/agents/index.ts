/**
 * Mastra Agent 工厂
 * 每次请求动态创建 agent，注入 episodeId/dramaId 到工具闭包
 * 从 agent_configs 表读取 prompt/model/temperature 配置
 */
import { Agent } from '@mastra/core/agent'
import { createOpenAI } from '@ai-sdk/openai'
import { eq, isNull, and } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { getTextConfig, getTextProviderBaseUrl } from '../services/ai.js'
import { logTaskError, logTaskProgress, startTrace } from '../utils/task-logger.js'
import { buildProtocolContract, parseAgentProtocol, type AgentProtocol } from './protocol.js'
import { createScriptTools } from './tools/script-tools.js'
import { createExtractTools } from './tools/extract-tools.js'
import { createStoryboardTools } from './tools/storyboard-tools.js'
import { createVoiceTools } from './tools/voice-tools.js'
import { createGridPromptTools } from './tools/grid-prompt-tools.js'
import { createRunSubagentTool } from './subagent.js'
import { loadAgentSkills } from './skills.js'
import { discoverMcpTools } from './mcp.js'

// Default prompts (used when DB has no config)
const DEFAULT_PROMPTS: Record<string, { name: string; instructions: string }> = {
  script_rewriter: {
    name: '剧本改写',
    instructions: `你是专业编剧，擅长将小说改编为短剧剧本。

工作流程：
1. 调用 read_episode_script 读取原始内容
2. 根据读取到的内容，自己进行改写（输出格式化剧本格式）
3. 调用 save_script 保存改写后的完整剧本

格式化剧本格式：
- 场景头：## S编号 | 内景/外景 · 地点 | 时间段
- 动作描写：自然段落，不包含镜头语言
- 对白：角色名：（状态/表情）台词内容
- 每个场景 30-60 秒内容

注意：你必须自己完成改写工作，不要只返回指令。读取内容后直接输出改写结果并保存。`,
  },
  extractor: {
    name: '角色场景提取',
    instructions: `你是制片助理，擅长从剧本中提取角色和场景信息，并在提取时与项目已有数据进行智能去重。

工作流程：
1. 调用 read_script_for_extraction 读取格式化剧本
2. 调用 read_existing_characters 读取项目中已存在的角色列表，以及当前集已关联角色
3. 调用 read_existing_scenes 读取项目中已存在的场景列表，以及当前集已关联场景
4. 优先围绕当前集剧本，分析本集实际出现的角色和场景
5. 对每个角色：若同名已存在则合并更新，若不存在则新增
6. 调用 save_dedup_characters 保存角色（去重合并，自动处理新增和更新，并关联到当前集）
7. 分析剧本内容，提取本集涉及的所有场景信息
8. 对每个场景：若同地点+时间段已存在则复用，若不存在则新增
9. 调用 save_dedup_scenes 保存场景（去重合并，自动处理新增和复用，并关联到当前集）

去重规则：
- 角色：按名字精确匹配，同名保留现有（合并信息）
- 场景：按【地点+时间段】精确匹配；同地点不同时段视为新场景

提取要求：
- 只提取当前集真实出现或被明确提及、且对当前集叙事有效的角色和场景
- 角色要包含完整的外貌特征描述（发型、服装、体态等），并单独填写服装（clothing）、武器（weapons）、首饰（accessories）、核心视觉特征（core_features，字符串数组）、服装变化（costumes，字符串数组）这些独立字段；不要把服装/武器/首饰信息只堆进 appearance 而遗漏独立字段
- 没有武器/首饰/多套服装的角色，weapons/accessories/costumes 填空字符串或空数组，但 clothing 必须尽量填写
- 场景要包含光线（lighting）、色调、氛围（atmosphere）、天气（weather）、季节（season）、风格（style）等视觉信息
- 不要遗漏任何有台词或重要动作的角色

提示词生成（必须在分析剧本内容之后、基于剧本真实信息生成，禁止套用固定模板）：
- 每个角色都必须生成 image_prompt（英文正向提示词）：融合角色外貌/服装/武器/首饰/风格，用于生成角色立绘，保证与其他角色风格统一
- 每个角色都必须生成 negative_prompt（英文反向提示词）：针对该角色排除不需要的元素（如 photorealistic、text、watermark、multiple people、畸形肢体等）
- 每个场景都必须生成 image_prompt（英文正向提示词）：融合地点/时间/光线/氛围/天气/季节/风格，用于生成场景图
- 每个场景都必须生成 negative_prompt（英文反向提示词）：排除不需要的元素（如 people、text、watermark、flat composition 等）
- 上述 image_prompt 和 negative_prompt 是硬性要求，每个角色、每个场景都必须填写，禁止省略或留空`,
  },
  storyboard_breaker: {
    name: '分镜拆解',
    instructions: `你是资深影视分镜师，擅长将剧本拆解为分镜方案。

工作流程：
1. 调用 read_storyboard_context 读取剧本、角色列表、场景列表
2. 将剧本拆解为镜头序列（每个镜头 10-15 秒，总体保持剧情完整连续）
3. 为每个镜头补全完整分镜字段，而不只是 video_prompt
4. 调用 save_storyboards 保存所有分镜

每个镜头必须尽量完整填写以下字段：
- title：3-8 字镜头标题
- shot_type：景别，如全景/中景/近景/特写
- angle：机位角度，如平视/仰视/俯视/侧拍
- movement：运镜，如固定/推镜/拉镜/摇镜/跟拍
- location：镜头地点，应与 scenes 中已有地点保持一致
- time：时间段，应与 scenes 中已有时间保持一致
- character_ids：当前镜头涉及的角色 ID 列表，可以为空，也可以包含多个角色；必须从 characters 中选择
- action：角色动作与表演
- dialogue：该镜头实际发生的对白或旁白；旁白可写为“旁白：内容”
- description：镜头概述，用于前端阅读和镜头编辑
- result：该镜头结束时的画面结果或状态变化
- atmosphere：氛围、光线、色调、环境感受
- image_prompt：镜头代表性静态画面提示词，用于生成镜头主图/封面
- first_frame_prompt：首帧画面提示词，描述该镜头开始时（动作起点）的静态画面，须包含角色外貌/服装、场景、构图、光线氛围
- last_frame_prompt：尾帧画面提示词，描述该镜头结束时（动作终点）的静态画面，须体现与首帧的差异（动作完成/状态变化/构图变化）
- video_prompt：用于视频生成的动态提示词
- negative_prompt：该镜头画面的反向提示词（英文），根据镜头内容排除不需要的元素（文字、水印、畸形肢体、多余人、模糊等）
- bgm_prompt：该镜头适合的配乐风格
- sound_effect：该镜头关键音效
- duration：时长，优先 10-15 秒
- scene_id：若可匹配到 scenes 中已有场景，必须填写正确 scene_id

视频提示词格式：
- 按 3 秒为一段，用时间标记分隔
- 使用 <location>地点</location> 标记场景
- 使用 <role>角色名</role> 标记角色
- 使用 <voice>角色名</voice> 标记画外音
- 用 <n> 分隔不同时间段

示例：
"0-3秒：<location>咖啡厅</location>，近景，<role>小明</role>低头看手机。<n>3-6秒：全景，<role>小红</role>推门走入。"

额外要求：
- 优先复用 read_storyboard_context 返回的 scene_id，不要凭空创造新场景
- 镜头角色绑定必须来自 read_storyboard_context 返回的角色列表；无角色的空镜头可传空数组
- 镜头描述必须能支撑后续图片、视频、配音、音效、合成流程
- 若一个镜头没有对白，可将 dialogue 置空，但 description / action / video_prompt / image_prompt 仍必须完整
- 每个镜头都必须生成 first_frame_prompt 和 last_frame_prompt，且两者画面要有明确差异（首帧=动作起点，尾帧=动作终点），否则视频生成会首尾帧雷同、失去动势
- 如果已有 existing_storyboards，仅在用户明确要求增量修改时参考；默认按当前剧本重新完整生成并保存整集分镜。`,
  },
  voice_assigner: {
    name: '角色音色分配',
    instructions: `你是配音导演，擅长为角色选择合适的音色。

工作流程：
1. 调用 list_voices 获取可用音色列表
2. 调用 get_characters 获取所有角色信息
3. 根据每个角色的性别、性格、年龄、角色定位，选择最匹配的音色
4. 对每个角色调用 assign_voice 分配音色，并说明选择理由

注意：每个角色都必须分配音色，不要遗漏。`,
  },
  grid_prompt_generator: {
    name: '宫格图提示词生成',
    instructions: `你是专业的 AI 图像提示词工程师，擅长为宫格图生成高质量的英文提示词。

## 宫格图提示词（参考 skills/grid-image-generator/SKILL.md）

工作流程：
1. 调用 read_shots_for_grid 读取选中镜头的详细信息
2. 根据 mode 调用 generate_grid_prompt：
   - first_frame 模式：按用户指定的 rows x cols 生成首帧风格宫格
   - first_last 模式：按用户指定的 rows x cols 生成首尾帧节奏感宫格
   - multi_ref 模式：按用户指定的 rows x cols 生成同一镜头的多角度宫格
3. 返回 grid_prompt（整体提示词）和 cell_prompts（每格提示词）

提示词规范：
- 使用英文提示词
- 必须严格遵守用户指定的 rows 和 cols
- 必须明确写出 "exactly N visible panels"
- 必须明确约束 "no merged panels, no missing panels"
- 宫格位置统一写成“格1/格2/...”，参考图统一写成“图片1/图片2/...”
- 必须包含 "consistent art style" 保持风格统一
- 必须包含 "cinematic quality"
- 避免出现文字或水印
- 宫格图片强调整体布局一致性`,
  },
  orchestrator: {
    name: '短剧制作总调度',
    instructions: `你是短剧制作的总调度（orchestrator），负责把用户的复杂需求拆解为多个子任务，并调度领域专家 Agent 协作完成，产出完整的短剧制作成果。

可用子 Agent（通过 run_subagent 工具委托，先调用 list_available_agents 确认）：
- script_rewriter（剧本改写）：小说/原始内容 → 格式化短剧剧本
- extractor（角色场景提取）：剧本 → 角色 + 场景（智能去重）
- storyboard_breaker（分镜拆解）：剧本 → 分镜方案
- voice_assigner（角色音色分配）：为角色分配音色
- grid_prompt_generator（图片提示词生成）：角色/场景/宫格图提示词

工作流程：
1. 调用 list_available_agents 确认当前可用的子 Agent
2. 理解用户需求，规划需要哪些子 Agent 及其执行顺序
3. 按依赖顺序调用 run_subagent，把每个子任务委托给对应专家；把上一个子 Agent 的结果摘要（尤其是已保存的 ID、关键结论）传给下一个
4. 汇总所有子 Agent 的产出，向用户报告整体进度与成果

调度规则：
- 有先后依赖的子任务必须串行（等上一个完成再委托下一个）
- 相互独立的子任务可以并行委托
- 不要自己做领域工作（改写/提取/分镜/配音/提示词），一律委托给专家 Agent
- 每个 run_subagent 的 task 要具体、包含足够上下文，例如"把上一阶段保存的剧本进行角色场景提取"
- 委托失败（返回 error）时，分析原因并重试或降级处理`,
  },
}

export const validAgentTypes = Object.keys(DEFAULT_PROMPTS)

/** 获取 Agent 默认提示词（无 DB 配置时的兜底，也是优化闭环的 Reference 基准） */
export function getDefaultInstructions(type: string): string {
  return DEFAULT_PROMPTS[type]?.instructions || ''
}

/** 获取 Agent 默认显示名（无 DB 配置时的兜底，供 agent-creation 生成 name 兜底） */
export function getDefaultName(type: string): string {
  return DEFAULT_PROMPTS[type]?.name || type
}

function getAgentConfig(agentType: string) {
  const rows = db.select().from(schema.agentConfigs)
    .where(and(eq(schema.agentConfigs.agentType, agentType), isNull(schema.agentConfigs.deletedAt)))
    .all()
  return rows.find(r => r.isActive) || rows[0] || null
}

/** 获取 Agent 工具，供 createAgent 和 runAgentWithRetry 共用 */
export function createAgentTools(
  type: string,
  episodeId: number,
  dramaId: number,
): Record<string, any> | null {
  switch (type) {
    case 'script_rewriter': return createScriptTools(episodeId)
    case 'extractor': return createExtractTools(episodeId, dramaId)
    case 'storyboard_breaker': return createStoryboardTools(episodeId, dramaId)
    case 'voice_assigner': return createVoiceTools(episodeId, dramaId)
    case 'grid_prompt_generator': return createGridPromptTools(episodeId, dramaId)
    case 'orchestrator': return createRunSubagentTool({ dramaId, episodeId, parentType: type })
    default: return null
  }
}

interface AgentConfig {
  textConfig: ReturnType<typeof getTextConfig>
  dbConfig: ReturnType<typeof getAgentConfig>
  defaults: (typeof DEFAULT_PROMPTS)[string]
  /** 纯基础指令（不含 skill / 协议契约），供 runAgentWithInstructions 作为唯一组装起点 */
  baseInstructions: string
  /** 完整指令（base + skill + 协议契约），供 createAgent 直接使用 */
  instructions: string
  name: string
  tools: Record<string, any>
  resolvedBaseURL: string
}

/** 统一组装完整 instructions：base + skill + 输出协议契约（单一事实来源，避免重复注入） */
function assembleInstructions(baseInstructions: string, skillInstructions: string | null): string {
  return [baseInstructions, skillInstructions, buildProtocolContract()]
    .filter(Boolean)
    .join('\n\n')
}

function buildAgentConfig(type: string, episodeId: number, dramaId: number): AgentConfig | null {
  const defaults = DEFAULT_PROMPTS[type]
  if (!defaults) return null

  const dbConfig = getAgentConfig(type)
  const baseInstructions = dbConfig?.systemPrompt?.trim() || defaults.instructions
  const skillInstructions = loadAgentSkills(type, dbConfig?.skills)
  const instructions = assembleInstructions(baseInstructions, skillInstructions)
  const name = dbConfig?.name || defaults.name
  const tools = createAgentTools(type, episodeId, dramaId)
  if (!tools) return null

  const textConfig = getTextConfig()
  const resolvedBaseURL = getTextProviderBaseUrl(textConfig)

  return { textConfig, dbConfig, defaults, baseInstructions, instructions, name, tools, resolvedBaseURL }
}

export function createAgent(type: string, episodeId: number, dramaId: number): Agent | null {
  const built = buildAgentConfig(type, episodeId, dramaId)
  if (!built) return null

  const modelName = built.dbConfig?.model || built.textConfig.model
  logTaskProgress('AIConfig', 'text-model-endpoint', {
    provider: built.textConfig.provider,
    baseUrl: built.resolvedBaseURL,
    model: modelName,
  })

  const provider = createOpenAI({
    baseURL: built.resolvedBaseURL,
    apiKey: built.textConfig.apiKey,
  } as any)

  return new Agent({
    id: type,
    name: built.name,
    instructions: built.instructions,
    model: provider.chat(modelName),
    tools: built.tools,
  })
}

function normalizeToolNick(tc: any): string {
  // Mastra 的 result.toolCalls/toolResults 元素是 { type, payload: { toolName, args, result } }
  // 工具名与参数都在 payload 里，优先从 payload 取
  return tc?.payload?.toolName || tc?.toolName || tc?.tool?.toolName || tc?.tool?.id || tc?.name || tc?.type || null
}

function normalizeToolOutput(tr: any): string {
  const out = tr?.payload?.result ?? tr?.result ?? tr?.output ?? tr?.data ?? null
  return typeof out === 'string' ? out : JSON.stringify(out)
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

/** 错误分类结果：transient（瞬态，可退避重试）/ fatal（非瞬态，立即失败） */
type LLMErrorClass = 'transient' | 'fatal'

/** 单个模型的瞬态重试上限（超出后换下一个模型） */
const MAX_TRANSIENT_RETRIES = 3

/**
 * 失败分类器（对齐 PenguinHarness 的 StopReason 归一化）：
 * - fatal：认证/授权、参数错误、上下文超限、内容审核拦截 —— 重试或换模型都无意义，立即失败省时间省费用。
 * - transient：限流/配额/过载/超时/网络抖动 —— 指数退避重试，耗尽后再换模型。
 */
function classifyLLMError(err: any): LLMErrorClass {
  const msg = `${err?.message || ''} ${err?.name || ''} ${err?.code || ''}`.toLowerCase()
  const status = err?.status ?? err?.statusCode ?? err?.response?.status ?? err?.response?.statusCode

  if (status === 401 || status === 403) return 'fatal'
  if (status === 400) return 'fatal'
  if (msg.includes('context length') || msg.includes('context_length') || msg.includes('maximum context')
    || msg.includes('max tokens') || msg.includes('too many tokens') || msg.includes('context window')
    || msg.includes('token limit') || msg.includes('exceed')) return 'fatal'
  if (msg.includes('content filter') || msg.includes('content_policy') || msg.includes('contentpolicy')
    || msg.includes('safety') || msg.includes('recitation') || msg.includes('moderation')
    || msg.includes('sensitive') || msg.includes('unsafe')) return 'fatal'

  if (status === 429 || status === 500 || status === 502 || status === 503 || status === 504) return 'transient'
  if (msg.includes('rate limit') || msg.includes('rate_limit') || msg.includes('ratelimit')
    || msg.includes('too many requests')) return 'transient'
  if (msg.includes('overloaded') || msg.includes('busy') || msg.includes('capacity')
    || msg.includes('insufficient_quota') || msg.includes('quota')) return 'transient'
  if (msg.includes('timeout') || msg.includes('timed out') || msg.includes('econnreset')
    || msg.includes('etimedout') || msg.includes('enotfound') || msg.includes('econnrefused')
    || msg.includes('network') || msg.includes('socket') || msg.includes('aborted')
    || msg.includes('fetch failed') || msg.includes('connection')) return 'transient'

  // 默认 transient：宁重试，避免分类不全误伤可用模型
  return 'transient'
}

/** 指数退避：2s 起、×2、30s 上限 */
function backoffDelay(retryIndex: number): number {
  return Math.min(2000 * 2 ** retryIndex, 30000)
}

/**
 * 带模型自动 fallback 的 Agent 执行
 *
 * 遍历 textConfig.models，任一模型成功即返回结果。
 * 每次失败时重新创建 Agent，使用配置中下一个模型重试。
 */
/** 单次 Agent 调用的 token 用量统计（对齐 gcc renderLogService 的 input/output/totalTokens） */
export interface TokenUsage {
  inputTokens: number
  outputTokens: number
  totalTokens: number
}

/**
 * 从 Mastra generate 结果中提取 token 用量。
 * 优先累加所有 step 的 usage（总用量），回退到最后一步的 usage；
 * 兼容 AI SDK v5（inputTokens/outputTokens）与 v4（promptTokens/completionTokens）字段命名。
 */
function extractTokenUsage(result: any): TokenUsage | null {
  const usages: any[] = []
  if (Array.isArray(result?.steps)) {
    for (const step of result.steps) {
      if (step?.usage) usages.push(step.usage)
    }
  }
  if (usages.length === 0 && result?.usage) usages.push(result.usage)
  if (usages.length === 0) return null

  let inputTokens = 0
  let outputTokens = 0
  let totalTokens = 0
  for (const u of usages) {
    const i = u.inputTokens ?? u.promptTokens ?? 0
    const o = u.outputTokens ?? u.completionTokens ?? 0
    const t = u.totalTokens ?? (i + o)
    inputTokens += i
    outputTokens += o
    totalTokens += t
  }
  return { inputTokens, outputTokens, totalTokens }
}

export interface AgentRunResult {
  /** 实际成功使用的模型名（供评测 runtime 一致性校验） */
  model: string
  text: string
  toolCalls: Array<{ toolName: string; args: any }>
  toolResults: Array<{ toolName: string; result: string }>
  /** 解析后的输出协议；缺失或校验失败时为 null（见 protocolErrors） */
  protocol: AgentProtocol | null
  protocolErrors: string[]
  /** token 用量统计；上游未返回时为 null */
  usage: TokenUsage | null
}

/**
 * 用自定义 instructions 执行 Agent（带模型自动 fallback）。
 * 供评测/优化闭环使用：直接传入候选提示词（纯 base，不含 skill/协议契约），
 * 由本函数统一组装，保证评测上下文与真实运行完全一致，且不依赖、不污染 DB 的 agent_configs。
 */
export async function runAgentWithInstructions(
  type: string,
  episodeId: number,
  dramaId: number,
  instructions: string,
  message: string,
  options?: { maxSteps?: number },
): Promise<AgentRunResult> {
  const built = buildAgentConfig(type, episodeId, dramaId)
  if (!built) throw new Error(`Invalid agent type: ${type}`)

  const trace = startTrace('Agent', `run-${type}`, { agentType: type, episodeId, dramaId })

  // 统一组装：base + skill + 协议契约（与 buildAgentConfig 同源，避免重复注入 skill）
  const skillInstructions = loadAgentSkills(type, built.dbConfig?.skills)
  const fullInstructions = assembleInstructions(instructions, skillInstructions)

  // 构建模型列表：textConfig.models 为基础，dbConfig.model 作为最高优先级前置
  const baseModels = built.textConfig.models?.length ? built.textConfig.models : [built.textConfig.model]
  // 如果 dbConfig 有指定 model，将其作为首选（去重）
  const models: string[] = built.dbConfig?.model
    ? [built.dbConfig.model, ...baseModels.filter(m => m !== built.dbConfig!.model)]
    : baseModels

  if (models.length === 0) {
    trace.error('no-model', { agentType: type })
    throw new Error(`No model configured for agent type ${type}`)
  }

  const maxSteps = options?.maxSteps ?? 20

  // MCP 外部工具接入：懒连接 single-flight 发现（失败只降级，绝不阻断 Agent run）
  const mcpTools = await discoverMcpTools()
  const mergedTools: Record<string, any> = Object.keys(mcpTools).length > 0
    ? { ...built.tools, ...mcpTools }
    : built.tools

  let lastError: any = null

  for (let attempt = 0; attempt < models.length; attempt++) {
    const modelName = models[attempt]

    trace.progress('model-fallback-attempt', {
      attempt: attempt + 1, totalModels: models.length, model: modelName,
    })

    const provider = createOpenAI({ baseURL: built.resolvedBaseURL, apiKey: built.textConfig.apiKey } as any)
    const agent = new Agent({
      id: type, name: built.name, instructions: fullInstructions,
      model: provider.chat(modelName), tools: mergedTools,
    })

    // 同一模型内：瞬态错误指数退避重试，非瞬态立即终止
    for (let retry = 0; retry <= MAX_TRANSIENT_RETRIES; retry++) {
      try {
        const result = await agent.generate(
          [{ role: 'user', content: message }],
          { maxSteps },
        )

        const allToolCalls = result.toolCalls || []
        const allToolResults = result.toolResults || []
        const text = result.text || ''
        const usage = extractTokenUsage(result)
        const { protocol, errors } = parseAgentProtocol(text)

        trace.success('completed', {
          model: modelName,
          toolCalls: allToolCalls.length,
          protocolStatus: protocol?.status ?? 'missing',
          protocolErrors: errors.length,
          ...(usage
            ? { inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, totalTokens: usage.totalTokens }
            : {}),
        })

        return {
          model: modelName,
          text,
          toolCalls: allToolCalls.map((tc: any) => ({
            toolName: normalizeToolNick(tc),
            args: tc?.payload?.args ?? tc?.args ?? tc?.input ?? null,
          })),
          toolResults: allToolResults.map((tr: any) => ({
            toolName: normalizeToolNick(tr),
            result: normalizeToolOutput(tr),
          })),
          protocol,
          protocolErrors: errors,
          usage,
        }
      } catch (err: any) {
        lastError = err
        const cls = classifyLLMError(err)

        // 非瞬态错误：重试/换模型都无意义，立即失败（省时间省费用）
        if (cls === 'fatal') {
          trace.error('fatal-error', { attempt: attempt + 1, model: modelName, error: err.message })
          throw err
        }

        // 瞬态错误：优先同一模型指数退避重试，耗尽后换下一个模型
        if (retry < MAX_TRANSIENT_RETRIES) {
          const delay = backoffDelay(retry)
          trace.progress('backoff-retry', {
            attempt: attempt + 1, retry: retry + 1, model: modelName, delayMs: delay, error: err.message,
          })
          await sleep(delay)
          continue
        }

        const isLast = attempt === models.length - 1
        trace.error(isLast ? 'all-models-failed' : 'model-fallback-error', {
          attempt: attempt + 1, model: modelName, error: err.message,
        })
        if (isLast) throw err
        break
      }
    }
  }

  throw lastError || new Error('All models failed')
}

export async function runAgentWithRetry(
  type: string,
  episodeId: number,
  dramaId: number,
  message: string,
  options?: { maxSteps?: number },
): Promise<AgentRunResult> {
  const built = buildAgentConfig(type, episodeId, dramaId)
  if (!built) throw new Error(`Invalid agent type: ${type}`)
  // 传纯 baseInstructions，skill/协议契约由 runAgentWithInstructions 统一组装，避免重复注入
  return runAgentWithInstructions(type, episodeId, dramaId, built.baseInstructions, message, options)
}
