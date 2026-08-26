/**
 * 超长输入保护（对齐 PenguinHarness 的上下文压缩思想，确定性滑窗版）
 *
 * 剧本/小说超长时若整段注入 LLM，会直接触发 context length 超限（fatal，不可重试）。
 * 这里用「保留前 70% + 后 30%，中间省略」的确定性滑窗截断：
 * - 剧本开头（人物出场、设定）与结尾（冲突高潮、结局）通常最关键，优先保留。
 * - 中间省略处显式提示，让 Agent 基于首尾关键情节继续，而不是拿到被硬截断的半句。
 *
 * 默认 24000 字符（中文约 2 字/token → 约 12k token），远低于常见 32k 上下文。
 */

const DEFAULT_MAX_CHARS = 24000

export interface SlicedText {
  text: string
  /** 是否发生了截断 */
  truncated: boolean
  /** 原始总字符数 */
  total_chars: number
  /** 实际保留字符数（含省略提示） */
  kept_chars: number
}

export function sliceLongText(content: string, maxChars = DEFAULT_MAX_CHARS): SlicedText {
  const total = content?.length ?? 0
  if (!content || total <= maxChars) {
    return { text: content ?? '', truncated: false, total_chars: total, kept_chars: total }
  }

  const head = Math.floor(maxChars * 0.7)
  const tail = maxChars - head
  const omitted = total - maxChars
  const text =
    content.slice(0, head) +
    `\n\n……（中间 ${omitted} 字因内容超长已省略，请基于上文开头与下文结尾的关键情节继续）……\n\n` +
    content.slice(-tail)

  return { text, truncated: true, total_chars: total, kept_chars: maxChars }
}
