#!/usr/bin/env node
/**
 * 一键跑全部仓库自检（三道串联 + 汇总）
 *
 * 为什么需要
 *   `.githooks/pre-commit` 是**按资产条件触发**的：改 `skills/` 只跑引用守卫、改
 *   `.codebuddy/memory/` 只跑记忆守卫。这对手提交是对的（快），但想「整体体检」时
 *   得手敲三道命令，容易漏跑 —— 而漏跑的那道恰恰可能是红的那道。
 *
 * 与 test-guards.mjs 的分工
 *   · 本脚本 = **跑**守卫（正向体检：真实资产当前是否合规）。
 *   · `test-guards.mjs` = **测**守卫（负向实证：守卫还能不能报错）—— 它已包含在本脚本第 3 道。
 *
 * 用法
 *   node scripts/check-all.mjs [--verbose]
 *     --verbose 透传给 `check-skill-refs.mjs`（额外列出被跳过的候选，审计其盲区）
 *   退出码：0 = 三道全过；1 = 任一失败（失败道次在末尾汇总）
 */
import { spawnSync } from 'node:child_process'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const VERBOSE = process.argv.includes('--verbose')

/** 顺序有意义：先跑「真实资产是否合规」，最后跑「守卫是否还能报错」 */
const STEPS = [
  {
    label: 'skills/ 路径引用完整性（含 docs/ 引用）',
    script: 'check-skill-refs.mjs',
    args: VERBOSE ? ['--verbose'] : [],
  },
  { label: '.codebuddy/memory/ 记忆层（8k 预算 + 锚点 + 落点）', script: 'check-memory.mjs', args: [] },
  { label: '守卫自检（两套守卫的负向用例）', script: 'test-guards.mjs', args: [] },
]

let failed = 0
for (const s of STEPS) {
  console.log(`\n=== ${s.label}\n=== → scripts/${s.script}`)
  // stdio: 'inherit' —— 让子守卫的彩色/多行输出原样透出，避免二次转述失真
  const r = spawnSync(process.execPath, [join(ROOT, 'scripts', s.script), ...s.args], { stdio: 'inherit' })
  if (r.status !== 0) failed++
}

console.log(
  failed
    ? `\n✗ 自检未通过：${failed}/${STEPS.length} 道失败`
    : `\n✓ 全部 ${STEPS.length} 道自检通过`
)
process.exit(failed ? 1 : 0)
