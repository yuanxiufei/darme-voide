/**
 * Agent 创建器（对齐 PenguinHarness 自进化闭环第一环 agent-creation）
 *
 * 一句话需求 → Agent 配置（name / description / systemPrompt / skills）
 *
 * 前置契约（缺失即报错）：
 * - agentType 必须属于 validAgentTypes（现有 5 个领域 Agent）
 * - requirement 非空
 *
 * 协议化输出：
 * - LLM 必须返回严格 JSON（字段名固定），解析失败即报错
 * - 生成的 systemPrompt 保持原工作流程与工具调用语义，只针对需求强化
 *
 * 防作弊隔离：生成器看不到评测 rubric，只看默认 prompt + 用户需求（与 optimizer 一致）
 */
import { Agent } from '@mastra/core/agent'
import { createOpenAI } from '@ai-sdk/openai'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { getTextConfig, getTextProviderBaseUrl } from '../services/ai.js'
import { llmFetch } from '../utils/llm-fetch.js'
import { getDefaultInstructions, getDefaultName, validAgentTypes } from './index.js'
import { AGENT_SKILL_MAP, listSkillIds } from './skills.js'
import { now } from '../utils/response.js'

export interface GeneratedAgentConfig {
  agentType: string
  name: string
  description: string
  systemPrompt: string
  /** 标准化后的 skill 绑定（只含合法 id，enabled=true） */
  skills: Array<{ id: string; enabled: boolean; priority: number }>
}

/** 全部可用 skill id 集合（扫描 skills/ 目录，含 minimax 技能库；不再受默认映射子集限制） */
const AVAILABLE_SKILL_IDS = listSkillIds()

const CREATOR_INSTRUCTIONS = `你是一名 AI Agent 配置生成专家。根据用户的一句话需求，为指定类型的 Agent 生成优化后的配置。

约束：
1. 保持原有工作流程与工具调用语义不变（不修改工具名、不改变调用顺序）
2. 只针对用户需求做针对性强化（补充规则、示例、强调易错点）
3. 输出必须是严格 JSON，不要输出任何解释、前言，也不要包 markdown 代码块围栏`

/** 健壮 JSON 提取：剥 markdown 围栏 + 取首个 { 到末个 } */
function extractJson(text: string): any {
  let t = text.trim()
  const fence = t.match(/^```[a-zA-Z]*\n([\s\S]*?)\n```$/)
  if (fence) t = fence[1].trim()
  const start = t.indexOf('{')
  const end = t.lastIndexOf('}')
  if (start === -1 || end === -1 || end <= start) {
    throw new Error('生成器未返回有效 JSON')
  }
  return JSON.parse(t.slice(start, end + 1))
}

/** 一句话需求 → 生成 Agent 配置（不落库） */
export async function generateAgentConfig(
  agentType: string,
  requirement: string,
): Promise<GeneratedAgentConfig> {
  if (!validAgentTypes.includes(agentType)) {
    throw new Error(`未知 Agent 类型：${agentType}（可用：${validAgentTypes.join(', ')}）`)
  }
  if (!requirement?.trim()) {
    throw new Error('requirement 不能为空')
  }

  const defaultInstructions = getDefaultInstructions(agentType)
  const textConfig = getTextConfig()
  const provider = createOpenAI({
    baseURL: getTextProviderBaseUrl(textConfig),
    apiKey: textConfig.apiKey,
    fetch: llmFetch,
  } as any)

  const agent = new Agent({
    id: 'agent-creator',
    name: 'Agent 配置生成器',
    instructions: CREATOR_INSTRUCTIONS,
    model: provider.chat(textConfig.model),
  })

  const userMessage = [
    `Agent 类型：${agentType}`,
    `当前默认提示词：`,
    `--- 开始 ---`,
    defaultInstructions,
    `--- 结束 ---`,
    ``,
    `可用 skill 集合：${AVAILABLE_SKILL_IDS.join(', ')}`,
    ``,
    `用户需求：${requirement}`,
    ``,
    `请严格输出 JSON（字段名固定）：`,
    `{"name": "简短中文显示名", "description": "一句话职责", "system_prompt": "完整系统提示词（可直接落库）", "skills": ["从可用集合选出的 skill id 列表"]}`,
  ].join('\n')

  const result = await agent.generate([{ role: 'user', content: userMessage }])
  const parsed = extractJson(result.text || '')

  const systemPrompt = String(parsed.system_prompt ?? '').trim()
  if (systemPrompt.length < 50) {
    throw new Error(`生成器产出无效 systemPrompt（length=${systemPrompt.length}）`)
  }

  // skills 过滤：只保留合法 id，非法静默丢弃；空则回退该类型的默认映射
  const rawSkills: unknown = parsed.skills
  const requestedIds = Array.isArray(rawSkills)
    ? rawSkills.map(s => String(s)).filter(id => AVAILABLE_SKILL_IDS.includes(id))
    : []
  const finalIds = requestedIds.length ? requestedIds : (AGENT_SKILL_MAP[agentType] || [])
  const skills = finalIds.map((id, i) => ({ id, enabled: true, priority: i + 1 }))

  return {
    agentType,
    name: String(parsed.name ?? '').trim() || getDefaultName(agentType),
    description: String(parsed.description ?? '').trim(),
    systemPrompt,
    skills,
  }
}

/** 将生成的配置 upsert 到 agent_configs（保留 model/temperature 等既有值），返回落库行 */
export function persistAgentConfig(candidate: GeneratedAgentConfig): any {
  const ts = now()
  const skillsStr = JSON.stringify(candidate.skills)

  const [existing] = db.select().from(schema.agentConfigs)
    .where(eq(schema.agentConfigs.agentType, candidate.agentType)).all()

  if (existing) {
    db.update(schema.agentConfigs).set({
      name: candidate.name,
      description: candidate.description,
      systemPrompt: candidate.systemPrompt,
      skills: skillsStr,
      isActive: true,
      deletedAt: null,
      updatedAt: ts,
    }).where(eq(schema.agentConfigs.id, existing.id)).run()
    const [row] = db.select().from(schema.agentConfigs).where(eq(schema.agentConfigs.id, existing.id)).all()
    return row
  }

  const res = db.insert(schema.agentConfigs).values({
    agentType: candidate.agentType,
    name: candidate.name,
    description: candidate.description,
    model: '',
    systemPrompt: candidate.systemPrompt,
    temperature: 0.7,
    maxTokens: 4096,
    maxIterations: 10,
    skills: skillsStr,
    isActive: true,
    createdAt: ts,
    updatedAt: ts,
  }).run()
  const [row] = db.select().from(schema.agentConfigs)
    .where(eq(schema.agentConfigs.id, Number(res.lastInsertRowid))).all()
  return row
}
