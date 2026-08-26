/**
 * 评测闭环 CLI 入口
 *
 * 用法：
 *   npx tsx src/evaluation/cli.ts evaluate  benchmarks/storyboard-breaker.json
 *   npx tsx src/evaluation/cli.ts optimize  benchmarks/storyboard-breaker.json --iterations 3
 *   npx tsx src/evaluation/cli.ts evaluate  benchmarks/extractor.json
 *   npx tsx src/evaluation/cli.ts optimize  benchmarks/extractor.json --iterations 3
 *   npx tsx src/evaluation/cli.ts evaluate  benchmarks/script-rewriter.json
 *   npx tsx src/evaluation/cli.ts evaluate  benchmarks/voice-assigner.json
 */
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { getDefaultInstructions } from '../agents/index.js'
import { evaluateCase } from './evaluator.js'
import { optimizeAgentPrompt } from './optimizer.js'
import { AGENT_BY_KIND } from './catalog.js'
import type { BenchmarkCase } from './types.js'

function loadCase(filePath: string): BenchmarkCase {
  const abs = resolve(process.cwd(), filePath)
  return JSON.parse(readFileSync(abs, 'utf-8')) as BenchmarkCase
}

function printReport(report: { total: number; dimensions: Array<{ name: string; score: number; max: number; detail: string }> }) {
  console.log(`总分 ${report.total}/100`)
  for (const d of report.dimensions) {
    console.log(`  ${d.name}: ${d.score}/${d.max}  ${d.detail}`)
  }
}

async function main() {
  const [, , cmd, casePath, ...rest] = process.argv
  if (!cmd || !casePath || !['evaluate', 'optimize'].includes(cmd)) {
    console.log('用法：')
    console.log('  npx tsx src/evaluation/cli.ts evaluate  benchmarks/storyboard-breaker.json')
    console.log('  npx tsx src/evaluation/cli.ts optimize  benchmarks/storyboard-breaker.json --iterations 3')
    process.exit(1)
  }

  const caseDef = loadCase(casePath)
  const agentType = AGENT_BY_KIND[caseDef.kind]
  if (!agentType) {
    console.error(`未知 case kind: ${caseDef.kind}`)
    process.exit(1)
  }

  if (cmd === 'evaluate') {
    console.log(`评测 ${agentType}（Reference 提示词）case=${caseDef.id}`)
    const report = await evaluateCase(caseDef, getDefaultInstructions(agentType))
    printReport(report)
    return
  }

  const idx = rest.indexOf('--iterations')
  const iterations = idx >= 0 && rest[idx + 1] ? Number(rest[idx + 1]) : 3
  await optimizeAgentPrompt(agentType, caseDef, { iterations })
}

main().catch(err => {
  console.error('评测失败：', err)
  process.exit(1)
})
