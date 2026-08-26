/**
 * 提示词优化器：状态机闭环（对齐 PenguinHarness 的 agent-optimization）
 *
 *   evidence（评测报告）→ hypothesis（失分维度）→ candidate（LLM 改写）
 *   → evaluation（跑评测）→ accept（总分严格更高）/ rollback（保留最佳）
 *
 * 防作弊隔离：
 * - 优化器只能看到「维度名 + 分数」，看不到 rubric 的金标准（如「漏提了谁」）。
 * - 候选提示词直接传入评测器，不写 DB，回滚零成本（无 tar.gz 快照的必要）。
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { Agent } from '@mastra/core/agent'
import { createOpenAI } from '@ai-sdk/openai'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { getDefaultInstructions, getDefaultName } from '../agents/index.js'
import { AGENT_SKILL_MAP } from '../agents/skills.js'
import { persistAgentConfig, type GeneratedAgentConfig } from '../agents/creator.js'
import { getTextConfig, getTextProviderBaseUrl } from '../services/ai.js'
import { evaluateCase } from './evaluator.js'
import type { BenchmarkCase, OptimizationHistory, ScoreReport } from './types.js'

const OPTIMIZER_INSTRUCTIONS = `你是一名专业的 AI Agent 提示词优化专家。你的任务是改进给定 Agent 的系统提示词，使其在评测中得分更高。

约束：
1. 保持原有的工作流程与工具调用语义不变（不修改工具名、调用顺序）
2. 只针对给定的失分维度做针对性强化（明确字段要求、补充示例、强调易错点）
3. 输出必须是完整的、可直接作为系统提示词的文本
4. 只输出新提示词本身，不要输出任何解释、前言，也不要包 markdown 代码块围栏`

/** 只暴露「维度名 + 分数」，不暴露 detail（detail 里可能含金标准答案） */
function formatFeedback(report: ScoreReport): string {
  return report.dimensions.map(d => `- ${d.name}：${d.score}/${d.max}`).join('\n')
}

/** LLM 偶尔会套 markdown 围栏，剥掉 */
function stripCodeFence(text: string): string {
  const t = text.trim()
  const m = t.match(/^```[a-zA-Z]*\n([\s\S]*?)\n```$/)
  return m ? m[1].trim() : t
}

/**
 * 将优化出的最佳提示词组装为可落库配置。
 * 优化器只改 systemPrompt，因此 name/description/skills 优先沿用 DB 既有值，
 * 避免误覆盖用户自定义的显示名、描述与 skill 绑定；无既有值时回退默认映射。
 */
function buildPersistCandidate(agentType: string, bestPrompt: string): GeneratedAgentConfig {
  const existing = db.select().from(schema.agentConfigs)
    .where(eq(schema.agentConfigs.agentType, agentType)).all()[0]

  let skills: Array<{ id: string; enabled: boolean; priority: number }> =
    (AGENT_SKILL_MAP[agentType] || []).map((id, i) => ({ id, enabled: true, priority: i + 1 }))
  if (existing?.skills) {
    try {
      const parsed = JSON.parse(existing.skills)
      if (Array.isArray(parsed)) skills = parsed as typeof skills
    } catch {
      // skills JSON 损坏时静默回退默认映射
    }
  }

  return {
    agentType,
    name: existing?.name || getDefaultName(agentType),
    description: existing?.description || '',
    systemPrompt: bestPrompt,
    skills,
  }
}

async function generateCandidatePrompt(
  agentType: string,
  currentPrompt: string,
  version: number,
  feedbackReport: ScoreReport,
): Promise<string> {
  const textConfig = getTextConfig()
  const provider = createOpenAI({
    baseURL: getTextProviderBaseUrl(textConfig),
    apiKey: textConfig.apiKey,
  } as any)

  const agent = new Agent({
    id: 'prompt-optimizer',
    name: '提示词优化器',
    instructions: OPTIMIZER_INSTRUCTIONS,
    model: provider.chat(textConfig.model),
  })

  const userMessage = [
    `Agent 类型：${agentType}`,
    `当前提示词（v${version}）：`,
    `--- 开始 ---`,
    currentPrompt,
    `--- 结束 ---`,
    ``,
    `最近一次评测的失分维度（只看分数，不要猜测金标准答案）：`,
    formatFeedback(feedbackReport),
    ``,
    `请针对上述失分维度改进提示词，输出改进后的完整提示词。`,
  ].join('\n')

  const result = await agent.generate([{ role: 'user', content: userMessage }])
  const candidate = stripCodeFence(result.text || '')
  if (candidate.length < 50) {
    throw new Error(`Optimizer produced invalid candidate (length=${candidate.length})`)
  }
  return candidate
}

function printReport(report: ScoreReport) {
  console.log(`  总分 ${report.total}/100`)
  for (const d of report.dimensions) {
    console.log(`    ${d.name}: ${d.score}/${d.max}  ${d.detail}`)
  }
}

/**
 * 优化指定 Agent 的提示词。
 * @param agentType storyboard_breaker | extractor | script_rewriter | voice_assigner
 * @param caseDef   基准 case（statement + rubric）
 */
export async function optimizeAgentPrompt(
  agentType: 'storyboard_breaker' | 'extractor' | 'script_rewriter' | 'voice_assigner',
  caseDef: BenchmarkCase,
  options: { iterations?: number; historyDir?: string; autoPersist?: boolean } = {},
): Promise<OptimizationHistory> {
  const iterations = options.iterations ?? 3
  const historyDir = options.historyDir ?? join(process.cwd(), 'benchmarks', 'history')
  const autoPersist = options.autoPersist ?? true
  const referencePrompt = getDefaultInstructions(agentType)

  // 1. 评测 Reference 基准（v0）
  console.log(`\n===== 评测 Reference 提示词（v0，当前默认）=====`)
  const referenceReport = await evaluateCase(caseDef, referencePrompt)
  printReport(referenceReport)
  const referenceModel = referenceReport.runtimeModel

  const history: OptimizationHistory = {
    agentType,
    caseId: caseDef.id,
    reference: { score: referenceReport.total, dimensions: referenceReport.dimensions, prompt: referencePrompt },
    iterations: [],
    best: { version: 0, score: referenceReport.total, prompt: referencePrompt },
  }
  let bestReport = referenceReport

  // 2. 状态机：evidence → hypothesis → candidate → evaluation → accept/rollback
  for (let v = 1; v <= iterations; v++) {
    console.log(`\n===== 迭代 ${v}/${iterations}：生成候选提示词 =====`)
    const candidate = await generateCandidatePrompt(agentType, history.best.prompt, history.best.version, bestReport)

    console.log(`===== 迭代 ${v}/${iterations}：评测候选 =====`)
    const candidateReport = await evaluateCase(caseDef, candidate)
    printReport(candidateReport)

    // runtime 一致性校验：候选与 Reference 必须同 provider/model，否则分数不可比（评测期间 runtime 漂移）
    if (referenceModel && candidateReport.runtimeModel && candidateReport.runtimeModel !== referenceModel) {
      console.warn(`[RUNTIME-MISMATCH] 候选=${candidateReport.runtimeModel} vs Reference=${referenceModel}，本轮结果不可比，跳过（不 accept）`)
      history.iterations.push({
        version: v,
        score: candidateReport.total,
        accepted: false,
        prompt: candidate,
        dimensions: candidateReport.dimensions,
      })
      continue
    }

    const accepted = candidateReport.total > history.best.score
    history.iterations.push({
      version: v,
      score: candidateReport.total,
      accepted,
      prompt: candidate,
      dimensions: candidateReport.dimensions,
    })

    if (accepted) {
      console.log(`[ACCEPT] ${candidateReport.total} > 最佳 ${history.best.score}，候选晋升为 v${v}`)
      history.best = { version: v, score: candidateReport.total, prompt: candidate }
      bestReport = candidateReport
    } else {
      console.log(`[ROLLBACK] ${candidateReport.total} <= 最佳 ${history.best.score}，保留 v${history.best.version}`)
    }
  }

  // 3. 持久化历史（含候选提示词，便于审计回放）
  mkdirSync(historyDir, { recursive: true })
  const outPath = join(historyDir, `${agentType}-${caseDef.id}.json`)
  writeFileSync(outPath, JSON.stringify(history, null, 2), 'utf-8')

  // 4. 自动落库：best 严格高于 Reference 才生效（对齐「总分严格高于 Reference 才接受」）
  if (autoPersist && history.best.version > 0 && history.best.score > history.reference.score) {
    const candidate = buildPersistCandidate(agentType, history.best.prompt)
    const row = persistAgentConfig(candidate)
    history.persisted = { version: history.best.version, score: history.best.score, name: row.name }
    console.log(`[PERSIST] 最佳候选 v${history.best.version} 已自动落库（agent_type=${agentType}, name=${row.name}）`)
  } else {
    history.persisted = null
    if (autoPersist) {
      console.log(`[SKIP] 无严格更优候选，不落库（best=${history.best.score}, reference=${history.reference.score}）`)
    }
  }

  console.log(`\n===== 优化结束 =====`)
  console.log(`最佳：v${history.best.version}，得分 ${history.best.score}（Reference v0 得分 ${history.reference.score}）`)
  console.log(`历史已写入：${outPath}`)
  return history
}
