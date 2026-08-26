/**
 * MergeQueue — 并发合并队列
 *
 * 移植自 PenguinHarness（packages/core/src/engine/context-engine.ts 的 MergeQueue）核心思想：
 *   多个并发 producer 各自 push 事件，单个 consumer 按 push 顺序（= 完成顺序）yield，
 *   全部 producer 结束且队列排空后返回 null 结束。
 *
 * 在本项目中的落地点：
 *   预设框架的「批量生成 5 个 Shot」此前是串行 for 循环（triggerImageGeneration /
 *   triggerVideoGeneration）。改用 runMerged 后，5 个独立生成任务 fire-and-forget 并发执行，
 *   各自的完成/失败事件合并成一条顺序事件流，支撑「一句话 → 整集短剧」全自动管线。
 *
 * 语义对齐：
 *   - addProducer/removeProducer/push/next/signal 与 PenguinHarness 完全一致（生产者计数）。
 *   - 消费者阻塞在 next() 直到有事件可消费或所有 producer 已结束。
 */

/** 单个任务完成/失败事件。index 为任务提交序号，用于恢复提交顺序。 */
export interface MergeEvent<T> {
  index: number
  status: 'fulfilled' | 'rejected'
  value?: T
  error?: unknown
}

/** 并发合并队列（生产者-消费者）。 */
export class MergeQueue<T> {
  private items: T[] = []
  private producers = 0
  private wake: (() => void) | null = null

  addProducer(): void {
    this.producers += 1
  }

  removeProducer(): void {
    this.producers -= 1
    this.signal()
  }

  push(msg: T): void {
    this.items.push(msg)
    this.signal()
  }

  private signal(): void {
    if (this.wake) {
      const w = this.wake
      this.wake = null
      w()
    }
  }

  /** 拉取下一条事件；队列空但还有 producer 则等待；全部结束且排空返回 null。 */
  async next(): Promise<T | null> {
    for (;;) {
      if (this.items.length > 0) return this.items.shift()!
      if (this.producers === 0) return null
      await new Promise<void>((resolve) => { this.wake = resolve })
    }
  }
}

/**
 * 并发执行 N 个异步任务，按完成顺序合并成一条事件流。
 *
 * @param tasks  任务工厂数组（立即并发启动，fire-and-forget）。
 * @param onEvent 可选，按完成顺序逐事件回调（用于实时进度/日志推送）。
 * @returns 按完成顺序排列的完整事件列表（含每个任务的 index，可据此恢复提交顺序）。
 */
export async function runMerged<T>(
  tasks: Array<() => Promise<T>>,
  onEvent?: (event: MergeEvent<T>) => void | Promise<void>,
): Promise<MergeEvent<T>[]> {
  const queue = new MergeQueue<MergeEvent<T>>()
  const events: MergeEvent<T>[] = []

  tasks.forEach((task, index) => {
    queue.addProducer()
    void (async () => {
      try {
        const value = await task()
        queue.push({ index, status: 'fulfilled', value })
      } catch (error) {
        queue.push({ index, status: 'rejected', error })
      } finally {
        queue.removeProducer()
      }
    })()
  })

  for (;;) {
    const event = await queue.next()
    if (event === null) break
    events.push(event)
    if (onEvent) await onEvent(event)
  }

  return events
}
