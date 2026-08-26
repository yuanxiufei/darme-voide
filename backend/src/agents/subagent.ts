/**
 * 子 Agent 调度（对齐 PenguinHarness 第 8 章「子 Agent 调度」）
 *
 * 提供 run_subagent 工具：让主控 Agent（orchestrator）在执行过程中，
 * 自主把子任务委托给领域专家 Agent，实现"Agent 调用 Agent"的递归能力。
 *
 * 硬约束（对齐参考文档）：
 * - agent_type 必须 ∈ validAgentTypes（运行时动态 import 校验）
 * - 禁止自调用（agent_type === parentType）
 * - MAX_SUBAGENT_DEPTH 深度上限，防无限递归
 * - AsyncLocalStorage 调用链追踪，防环形调用（A→B→A）
 */
import { AsyncLocalStorage } from 'node:async_hooks'
import { createTool } from '@mastra/core/tools'
import { z } from 'zod'

/** 领域专家 Agent 能力清单（供 orchestrator 决策调度） */
export const SUBAGENT_REGISTRY: Array<{ type: string; name: string; capability: string }> = [
  { type: 'script_rewriter', name: '剧本改写', capability: '将小说/原始内容改写为格式化短剧剧本并保存到当前集' },
  { type: 'extractor', name: '角色场景提取', capability: '从剧本提取角色与场景（同名/同地点智能去重）并保存' },
  { type: 'storyboard_breaker', name: '分镜拆解', capability: '将剧本拆解为带完整字段的分镜方案并保存' },
  { type: 'voice_assigner', name: '角色音色分配', capability: '为每个角色匹配最合适的音色并保存' },
  { type: 'grid_prompt_generator', name: '图片提示词生成', capability: '生成角色/场景/宫格图的英文提示词' },
]

/** 最大委托深度：orchestrator → 领域 Agent（领域 Agent 无 run_subagent，天然终止） */
export const MAX_SUBAGENT_DEPTH = 2

interface SubagentChainContext {
  chain: string[]
}

/** 调用链追踪：跨 async 传播，用于防环形调用 + 深度上限 */
const subagentALS = new AsyncLocalStorage<SubagentChainContext>()

export interface RunSubagentTools {
  run_subagent: ReturnType<typeof createTool>
  list_available_agents: ReturnType<typeof createTool>
}

/**
 * 创建子 Agent 调度工具集
 * @param dramaId / episodeId - 委托给子 Agent 时透传的上下文 ID
 * @param parentType - 主控 Agent 类型（用于防自调用 + 调用链起点）
 */
export function createRunSubagentTool(options: {
  dramaId: number
  episodeId: number
  parentType: string
}): RunSubagentTools {
  const { dramaId, episodeId, parentType } = options

  const runSubagent = createTool({
    id: 'run_subagent',
    description:
      'Delegate a subtask to a specialist agent and return its result. Use this to break a complex job into domain-expert tasks executed in order.',
    inputSchema: z.object({
      agent_type: z.string().describe('Target specialist agent type (see list_available_agents)'),
      task: z.string().describe('The concrete instruction for the subagent, with enough context'),
    }),
    execute: async ({ agent_type, task }) => {
      // 动态 import 打破循环依赖（subagent.ts ↔ index.ts）
      const { runAgentWithRetry, validAgentTypes } = await import('./index.js')

      if (!validAgentTypes.includes(agent_type)) {
        return { error: `Unknown agent type: ${agent_type}. Available: ${validAgentTypes.join(', ')}` }
      }
      if (agent_type === parentType) {
        return { error: `Cannot delegate to self (${agent_type})` }
      }

      const current = subagentALS.getStore()
      const chain = current?.chain ?? [parentType]
      if (chain.includes(agent_type)) {
        return { error: `Circular delegation detected: ${[...chain, agent_type].join(' -> ')}` }
      }
      if (chain.length >= MAX_SUBAGENT_DEPTH) {
        return { error: `Max subagent depth reached (${MAX_SUBAGENT_DEPTH})` }
      }

      try {
        const result = await subagentALS.run({ chain: [...chain, agent_type] }, () =>
          runAgentWithRetry(agent_type, episodeId, dramaId, task),
        )
        return {
          status: result.protocol?.status ?? 'completed',
          summary: result.protocol?.summary ?? '',
          text: result.text,
          tool_calls_count: result.toolCalls.length,
        }
      } catch (err: any) {
        return { error: `Subagent ${agent_type} failed: ${err.message}` }
      }
    },
  })

  const listAgents = createTool({
    id: 'list_available_agents',
    description: 'List the specialist agents available for delegation and their capabilities.',
    inputSchema: z.object({}),
    execute: async () => ({
      agents: SUBAGENT_REGISTRY.filter(a => a.type !== parentType),
    }),
  })

  return { run_subagent: runSubagent, list_available_agents: listAgents }
}
