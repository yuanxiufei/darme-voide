/**
 * Agent 输出协议（P3 落地：Skill 输出协议化）
 *
 * 借鉴 PenguinHarness agent-evaluation 的"纯协议输出"思想：
 * 让每个 Agent 完成任务后在最终回复里输出结构化 YAML 协议块，
 * 下游（前端 / 评测 / 优化 / 管线）据此稳定消费，不再依赖 LLM 自由文本。
 *
 * 协议契约由 instructions 注入（见 buildProtocolContract），
 * Agent 代码只负责解析与校验，二者解耦——协议字段语义由契约文本定义，不在代码里硬编码。
 */
import { parse as parseYaml } from 'yaml'

export interface AgentProtocol {
  /** ok=成功完成，failed=失败 */
  status: 'ok' | 'failed'
  /** 一句话概括完成情况与关键数字 */
  summary: string
  /** 允许 Agent 附加结构化统计字段，供下游交叉验证 */
  [key: string]: unknown
}

/** 追加到 instructions 末尾的输出协议契约（所有 Agent 类型通用） */
export function buildProtocolContract(): string {
  return [
    '## 输出协议（必须遵守）',
    '',
    '完成任务后，你必须在最后一条回复中输出一个 YAML 代码块作为收尾协议，格式如下：',
    '',
    '```yaml',
    'status: ok        # ok 表示成功完成，failed 表示失败',
    'summary: 一句话概括完成情况和关键数字',
    '```',
    '',
    '要求：',
    '- status 只能填 ok 或 failed',
    '- summary 用一句话概括（例如：已保存 12 个分镜）',
    '- 即使中途工具调用失败，也要输出协议并如实填写 status: failed',
  ].join('\n')
}

/** 从 Agent 最终文本中提取并解析协议块 */
export function parseAgentProtocol(text: string): {
  protocol: AgentProtocol | null
  errors: string[]
} {
  if (!text) return { protocol: null, errors: ['empty text'] }

  // 提取 YAML fence 块
  const fenceRe = /```ya?ml\s*\n?([\s\S]*?)```/i
  const match = text.match(fenceRe)
  if (!match) return { protocol: null, errors: ['no yaml fence found'] }

  let parsed: unknown
  try {
    parsed = parseYaml(match[1])
  } catch (err) {
    return { protocol: null, errors: [`yaml parse error: ${(err as Error).message}`] }
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { protocol: null, errors: ['protocol is not an object'] }
  }

  const obj = parsed as Record<string, unknown>
  const status = obj.status
  const summary = obj.summary

  if (status !== 'ok' && status !== 'failed') {
    return { protocol: null, errors: [`invalid status: ${JSON.stringify(status)}`] }
  }
  if (typeof summary !== 'string' || !summary.trim()) {
    return { protocol: null, errors: ['missing or empty summary'] }
  }

  return {
    protocol: { ...obj, status, summary: summary.trim() } as AgentProtocol,
    errors: [],
  }
}
