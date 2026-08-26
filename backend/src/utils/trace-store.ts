/**
 * Trace 持久化存储 — 对齐 PenguinHarness 第 9 章「Trace 写入与回放（崩溃恢复）」。
 *
 * 写入端三约束：
 * 1. append-only JSONL，每行一条完整记录（首尾 ts 字段，天然自描述）
 * 2. 刻意不用 fs.appendFile：直接用 O_APPEND 打开 → 单次 write(2) 写完 → 关闭，
 *    避免大 payload 被拆成多次 write 后进程崩溃留下残缺记录
 * 3. 尾部愈合：append 前探测末字节，非 '\n' 则补 '\n'，防止残缺末行粘连下一条新记录
 *
 * 串行化：全局 promise 链，保证多异步生产者串行写入、不交错（单实例单队列）。
 * 回放端：结构保真、非字节保真——逐行 JSON.parse，跳过空白与残缺/损坏行，
 * 不因单条坏记录整体失败（这正是崩溃恢复可回放的关键）。
 */
import fsp from 'fs/promises'
import path from 'path'
import { getDataRoot } from '../config.js'

/** 数据根目录下的 traces 子目录（跟随数据目录切换，纳入迁移范围） */
function getTracesDir(): string {
  return path.join(getDataRoot(), 'traces')
}

export interface TraceRecord {
  ts: string
  traceId: string
  scope: string
  level: 'INFO' | 'WARN' | 'ERROR' | 'SUCCESS'
  action: string
  elapsedMs?: number
  meta?: Record<string, unknown>
}

export interface TraceMeta {
  scope: string
  traceId: string
  file: string
  size: number
  mtimeMs: number
  count: number
  firstTs?: string
  lastTs?: string
}

/** scope / traceId 可能含路径分隔符或 ..，统一净化防止路径遍历 */
function safeSegment(s: string): string {
  return s.replace(/[^a-zA-Z0-9_-]/g, '_').replace(/_{2,}/g, '_')
}

function traceFilePath(scope: string, traceId: string): string {
  return path.join(getTracesDir(), safeSegment(scope), `${safeSegment(traceId)}.jsonl`)
}

// 全局串行化写入链：保证 append 顺序稳定、不交错
let writeChain: Promise<unknown> = Promise.resolve()

/**
 * 追加一条 trace 记录。fire-and-forget 语义下仍串行落盘，不阻塞调用方主流程。
 */
export function appendTraceEvent(record: TraceRecord): void {
  const filePath = traceFilePath(record.scope, record.traceId)
  const line = JSON.stringify(record) + '\n'
  writeChain = writeChain
    .catch(() => {}) // 上一条失败不阻断后续
    .then(() => appendJsonLine(filePath, line))
    .catch((err) => {
      console.warn(`[trace-store] 写入失败 ${filePath}:`, err?.message)
    })
}

/** O_APPEND 打开 → 尾部愈合 → 单次 write → 关闭 */
async function appendJsonLine(filePath: string, line: string): Promise<void> {
  await fsp.mkdir(path.dirname(filePath), { recursive: true })

  // 尾部愈合：探测末字节，非换行则补换行
  let heal = ''
  try {
    const stat = await fsp.stat(filePath)
    if (stat.size > 0) {
      const fh = await fsp.open(filePath, 'r')
      try {
        const buf = Buffer.alloc(1)
        const { bytesRead } = await fh.read(buf, 0, 1, stat.size - 1)
        if (bytesRead === 1 && buf[0] !== 0x0a) heal = '\n'
      } finally {
        await fh.close()
      }
    }
  } catch {
    /* 文件尚不存在，无需愈合 */
  }

  const fh = await fsp.open(filePath, 'a') // O_APPEND
  try {
    await fh.write(heal + line) // 单次 write(2) 写完
  } finally {
    await fh.close()
  }
}

/** 结构保真回放：逐行解析，跳过空白与残缺行 */
export async function readTrace(scope: string, traceId: string): Promise<TraceRecord[]> {
  const filePath = traceFilePath(scope, traceId)
  let content: string
  try {
    content = await fsp.readFile(filePath, 'utf-8')
  } catch {
    return []
  }
  const records: TraceRecord[] = []
  for (const line of content.split('\n')) {
    const t = line.trim()
    if (!t) continue
    try {
      records.push(JSON.parse(t) as TraceRecord)
    } catch {
      // 残缺末行（进程崩溃写一半）：跳过，保持结构保真
    }
  }
  return records
}

/** 列出所有（或某 scope 下）的 trace 元信息，按修改时间倒序 */
export async function listTraces(scope?: string): Promise<TraceMeta[]> {
  const tracesDir = getTracesDir()
  const base = scope ? path.join(tracesDir, safeSegment(scope)) : tracesDir
  let files: string[] = []
  try {
    files = await collectJsonl(base)
  } catch {
    return []
  }
  const metas: TraceMeta[] = []
  for (const file of files) {
    try {
      const stat = await fsp.stat(file)
      const rel = path.relative(tracesDir, file)
      const parts = rel.split(path.sep)
      const traceId = path.basename(file, '.jsonl')
      const scopeName = parts.length > 1 ? parts.slice(0, -1).join('/') : ''
      const records = await readTrace(scopeName, traceId)
      metas.push({
        scope: scopeName,
        traceId,
        file: rel,
        size: stat.size,
        mtimeMs: stat.mtimeMs,
        count: records.length,
        firstTs: records[0]?.ts,
        lastTs: records[records.length - 1]?.ts,
      })
    } catch {
      /* 忽略单文件错误 */
    }
  }
  metas.sort((a, b) => b.mtimeMs - a.mtimeMs)
  return metas
}

async function collectJsonl(dir: string): Promise<string[]> {
  const out: string[] = []
  const entries = await fsp.readdir(dir, { withFileTypes: true })
  for (const e of entries) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) out.push(...(await collectJsonl(p)))
    else if (e.isFile() && e.name.endsWith('.jsonl')) out.push(p)
  }
  return out
}
