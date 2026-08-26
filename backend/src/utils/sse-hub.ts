/**
 * SSE 进度推送事件总线 — 对齐 PenguinHarness 第 16.4 节「SSE 端点的交付保证」。
 *
 * 设计约束（忠实于参考文档）：
 * 1. 内存 pub/sub，按 dramaId 分频道（每个短剧生成任务一个频道）。
 * 2. 发布端 fire-and-forget：`publishPipelineEvent` 永不 throw，订阅者 handler 内部异常
 *    自吞，保证"进度推送"这个簿记动作绝不拖垮 auto-pipeline 主流程
 *    （对齐 16.3 节 `drive` 收尾不变量——簿记绝不能拖垮 run 的生命周期）。
 * 3. 单进程单实例，频道无订阅者时自动清理，避免内存泄漏。
 * 4. 事件不落盘、不缓存（历史状态由 GET /status 快照提供，SSE 只推增量——
 *    对齐 16.4 节"新订阅不回放 buffer，历史由 messages 端点提供"）。
 */

/** 推送事件类型：status=阶段状态推进 / media-progress=媒体等待的中间进度 */
export interface PipelineEvent {
  type: 'status' | 'media-progress'
  episodeId: number
  /** type=status 时：episode.status 新值（auto:* 单向状态机） */
  status?: string
  /** type=media-progress 时：已就绪数量 */
  ready?: number
  /** type=media-progress 时：总数 */
  total?: number
  ts: string
}

type Handler = (event: PipelineEvent) => void

const channels = new Map<number, Set<Handler>>()

/** 发布进度事件到指定 drama 频道（永不 throw，订阅者异常自吞） */
export function publishPipelineEvent(dramaId: number, event: Omit<PipelineEvent, 'ts'>): void {
  const subscribers = channels.get(dramaId)
  if (!subscribers || subscribers.size === 0) return
  const full: PipelineEvent = { ...event, ts: new Date().toISOString() }
  for (const handler of subscribers) {
    try {
      handler(full)
    } catch {
      // 单个订阅者异常不影响其他订阅者，也绝不抛回发布方
    }
  }
}

/** 订阅 drama 频道，返回取消函数 */
export function subscribePipeline(dramaId: number, handler: Handler): () => void {
  let set = channels.get(dramaId)
  if (!set) {
    set = new Set()
    channels.set(dramaId, set)
  }
  set.add(handler)
  return () => {
    set.delete(handler)
    if (set.size === 0) channels.delete(dramaId)
  }
}
