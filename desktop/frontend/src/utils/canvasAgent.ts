/**
 * 画布 AI —— 一句话 ⇒ 后端 Agent 编译出 op ⇒ 前端落盘。
 *
 * 三段职责分得很清楚，改哪段都不会牵动别处：
 *
 * | 段 | 谁做 | 在哪 |
 * | --- | --- | --- |
 * | 话 → op | 模型 + `canvas_ops.build_request` | `backend-py/app/services/canvas_agent.py` |
 * | op → 状态 | `applyCanvasOps` | `utils/canvasOps.ts` |
 * | op → 生成任务 | 调用方注入的 `onGenerate` | 画布页（生成域，要工作流与档位） |
 *
 * ## 两个必须如实交代的地方
 *
 * 1. **后端返回的 op 此刻还没落地** —— 别把 `reply.text` 当执行结果播给用户，
 *    要等 `applyCanvasOps` 回来再说「已落 N 条」✓
 * 2. **模型说的 ≠ 画布真做到的** —— 前端拒绝的 op（连线三不许、节点不存在）
 *    必须一起报出来，否则用户看到「AI 说连上了」但画布没线 ✗
 */
import { canvasAPI } from '../composables/useApi'
import { applyCanvasOps, canvasSnapshot, type ApplyResult, type CanvasOp } from './canvasOps'

export interface CanvasAgentOutcome {
  /** 模型的回话（说明它做了什么） */
  text: string
  /** 真正落到画布上的 op 条数 */
  applied: number
  /** 后端编译失败（模型参数不对，画布没动） */
  errors: string[]
  /** 前端落地被拒（连线被拒、节点不存在等） */
  rejected: string[]
  /** 等着用户点一下「运行」的配置节点（没注入 `onGenerate` 时才会有） */
  pendingRun: string[]
  /** 模型调用过的工具（排查用） */
  toolCalls: Array<{ toolName: string; args: Record<string, unknown> }>
}

/**
 * 跑一次画布 AI ✓。
 *
 * `includeSnapshot` 默认 **true**：模型看不到画布就会凭空编节点 id（那是最常见的翻车方式）。
 * 只有画布是空的、或调用方明确知道不需要时才关掉。
 */
export async function runCanvasAgent(
  message: string,
  handlers: { onGenerate?: (op: CanvasOp) => void; includeSnapshot?: boolean } = {},
): Promise<CanvasAgentOutcome> {
  const text = (message || '').trim()
  if (!text) throw new Error('请先说一句要做什么')

  const snapshot = handlers.includeSnapshot === false ? undefined : canvasSnapshot()
  const reply = await canvasAPI.agent({ message: text, snapshot })

  const ops = (reply.ops || []) as CanvasOp[]
  const apply: ApplyResult = applyCanvasOps(ops, { onGenerate: handlers.onGenerate })

  // `run_generation` 被跳过有两种原因：没给回调（⇒ 等用户手动点，这是**正常**的），
  // 或者缺 nodeId（⇒ 是真错误）。混在一起报会给用户一条看不懂的告警 ✗
  const pendingRun: string[] = []
  const rejected: string[] = []
  for (const item of apply.skipped) {
    if (!handlers.onGenerate && item.op.type === 'run_generation' && item.op.nodeId) {
      pendingRun.push(item.op.nodeId)
      continue
    }
    rejected.push(`${item.op.type}：${item.reason}`)
  }

  return {
    text: reply.text || '',
    applied: apply.applied,
    errors: reply.errors || [],
    rejected,
    pendingRun,
    toolCalls: reply.toolCalls || [],
  }
}
