/**
 * 评测基准 case 目录（单一来源）
 *
 * - AGENT_BY_KIND：case.kind → Agent 类型（CLI 与 HTTP 路由共用，避免重复映射）
 * - listBenchmarkCases / loadCaseById：运行时发现与按 id 加载（供 HTTP 路由使用）
 */
import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { dirname, resolve, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { BenchmarkCase } from './types.js'

/** case.kind → Agent 类型 */
export const AGENT_BY_KIND: Record<
  string,
  'storyboard_breaker' | 'extractor' | 'script_rewriter' | 'voice_assigner'
> = {
  storyboard: 'storyboard_breaker',
  extractor: 'extractor',
  script_rewriter: 'script_rewriter',
  voice_assigner: 'voice_assigner',
}

/** benchmarks/ 目录定位（与 cwd 解耦，兼容 Docker WORKDIR /app），对齐 skills.ts 写法 */
const BENCHMARKS_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../../benchmarks')

interface CaseFile {
  id: string
  kind: string
  agentType: string
  file: string
}

function scanCaseFiles(): CaseFile[] {
  if (!existsSync(BENCHMARKS_DIR)) return []
  return readdirSync(BENCHMARKS_DIR, { withFileTypes: true })
    .filter(d => d.isFile() && d.name.endsWith('.json'))
    .map(d => {
      const file = join(BENCHMARKS_DIR, d.name)
      const def = JSON.parse(readFileSync(file, 'utf-8')) as BenchmarkCase
      return { id: def.id, kind: def.kind, agentType: AGENT_BY_KIND[def.kind] || '', file }
    })
    .filter(c => c.agentType)
}

/** 列出全部基准 case 元信息（供 /evaluation/cases 展示） */
export function listBenchmarkCases(): Array<{ id: string; kind: string; agentType: string }> {
  return scanCaseFiles().map(({ id, kind, agentType }) => ({ id, kind, agentType }))
}

/** 按 case id 加载完整 case（全目录扫描，case 文件极少，无性能顾虑） */
export function loadCaseById(caseId: string): BenchmarkCase | null {
  const hit = scanCaseFiles().find(c => c.id === caseId)
  if (!hit) return null
  return JSON.parse(readFileSync(hit.file, 'utf-8')) as BenchmarkCase
}
