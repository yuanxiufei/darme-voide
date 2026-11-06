#!/usr/bin/env node
/**
 * 响应式静态守卫（Web 端 frontend/ 与桌面端 desktop/frontend/ 一起扫）
 *
 * 四条硬判据（任一不满足则退出码 1）：
 *   A. 两端 `assets/studio.css` 的「响应式基础层」逐字一致
 *      —— 这是全仓唯一的断点真源，跑偏一处就会一边好一边坏；
 *   B. pages/ 下每个页面文件至少有一条 `@media (max-width: ...)` 收敛规则
 *      —— 页面级布局（固定列数、固定面板宽度）只有页面自己知道；
 *   C. 不出现 `repeat(auto-fill|auto-fit, minmax(<Npx>, 1fr))`
 *      —— 320px 视口下 N 大于容器宽度会直接把页面撑出横向滚动，
 *         必须写成 `minmax(min(Npx, 100%), 1fr)`。
 *   D. 本脚本在两端的副本必须逐字相同（各存一份是为了互不依赖；漂移必须报）
 *
 * 用法：node scripts/check-responsive.mjs [--verbose]（两端各自的 `scripts/` 下各有一份）
 */
import { createHash } from 'node:crypto'
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

//: 两端**各存一份**（`frontend/scripts/` 与 `desktop/frontend/scripts/`，逐字相同 ✓）：
//: 各自 self-contained ⇒ 删掉任一端的 scripts/ 都不会让另一端报错 ✓；漂移由判据 D 判红 ✓。
//: ⚠️ 别搬去仓库根 `scripts/`：structure_test 的 ㊃ 明令禁止（README 有承诺）✓。
const SELF = fileURLToPath(import.meta.url)

/** 从脚本位置向上找仓库根：认「同时含 frontend/app 与 desktop/frontend/src」的目录 ✓ */
function findRepo(from) {
  let dir = from
  for (let i = 0; i < 8; i += 1) {
    if (existsSync(join(dir, 'frontend', 'app')) && existsSync(join(dir, 'desktop', 'frontend', 'src'))) return dir
    const up = resolve(dir, '..')
    if (up === dir) break
    dir = up
  }
  throw new Error(`找不到仓库根：从 ${from} 向上未发现同时含 frontend/app 与 desktop/frontend/src 的目录`)
}

const root = findRepo(dirname(SELF))
const verbose = process.argv.includes('--verbose')

const WEB = join(root, 'frontend', 'app')
const DESK = join(root, 'desktop', 'frontend', 'src')

const MARKER = '响应式基础层（多设备适配的唯一真源'

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === '.git' || name === 'dist') continue
    const p = join(dir, name)
    if (statSync(p).isDirectory()) walk(p, out)
    else out.push(p)
  }
  return out
}

const problems = []
const notes = []

/* --- D. 两端各一份脚本 ⇒ 必须逐字相同（漂移会让两侧给出不同结论 ✗）--------- */
const sha = (p) => createHash('sha256').update(readFileSync(p)).digest('hex')
const selfSha = sha(SELF)
for (const p of [
  join(root, 'frontend', 'scripts', 'check-responsive.mjs'),
  join(root, 'desktop', 'frontend', 'scripts', 'check-responsive.mjs'),
].filter((x) => resolve(x) !== resolve(SELF))) {
  const rel = relative(root, p).split('\\').join('/')
  if (!existsSync(p)) problems.push(`D 副本缺失：${rel}（两端各一份尺子，缺一侧等于单边护栏）`)
  else if (sha(p) !== selfSha) problems.push(`D 两份尺子已漂移：${rel} 与本文件不同 —— 改一处必须同步另一处`)
}

/* --- A. 两端 studio.css 的响应式基础层必须逐字一致 ------------------------- */
function responsiveLayer(path) {
  const css = readFileSync(path, 'utf8')
  const start = css.indexOf(MARKER)
  if (start < 0) return null
  // 回退到该段落上面的注释框起始行，保证比较起点完整
  const boxStart = css.lastIndexOf('/* ====', start)
  return css.slice(boxStart < 0 ? start : boxStart).trimEnd()
}

const webCss = join(WEB, 'assets', 'studio.css')
const deskCss = join(DESK, 'assets', 'studio.css')
const a = responsiveLayer(webCss)
const b = responsiveLayer(deskCss)
if (!a || !b) {
  problems.push('studio.css 缺少「响应式基础层」段落：' + (!a ? 'Web 端' : '桌面端'))
} else if (a !== b) {
  const la = a.split('\n')
  const lb = b.split('\n')
  let i = 0
  while (i < la.length && i < lb.length && la[i] === lb[i]) i++
  problems.push(
    `两端 studio.css 的响应式基础层不一致（第 ${i + 1} 行起）：\n` +
      `  web : ${(la[i] ?? '<EOF>').trim()}\n` +
      `  desk: ${(lb[i] ?? '<EOF>').trim()}`,
  )
}

/* --- B/C. 逐页扫描 -------------------------------------------------------- */
const AUTO_FILL = /repeat\(\s*(auto-fill|auto-fit)\s*,\s*minmax\(\s*(\d+)px/g

for (const [label, base] of [['web', WEB], ['desktop', DESK]]) {
  const pages = walk(join(base, 'pages')).filter((f) => f.endsWith('.vue'))
  for (const file of pages) {
    const src = readFileSync(file, 'utf8')
    const rel = relative(root, file).replace(/\\/g, '/')

    if (!/@media\s*\(max-width:\s*\d+px\)/.test(src)) {
      problems.push(`${rel}: 页面没有任何 @media (max-width: ...) 收敛规则`)
    }

    AUTO_FILL.lastIndex = 0
    let m
    while ((m = AUTO_FILL.exec(src))) {
      const [, kind, px] = m
      // 合法写法：minmax(min(320px, 100%), 1fr)
      const tail = src.slice(m.index, m.index + 160)
      if (!/minmax\(\s*min\(/.test(tail)) {
        const line = src.slice(0, m.index).split('\n').length
        problems.push(`${rel}:${line}: repeat(${kind}, minmax(${px}px, …)) 缺 min() 兜底，320px 视口会横向溢出`)
      }
    }

    const breaks = [...src.matchAll(/@media\s*\(max-width:\s*(\d+)px\)/g)].map((x) => Number(x[1]))
    if (breaks.length) {
      const uniq = [...new Set(breaks)].sort((x, y) => x - y)
      notes.push(`${label} ${rel}: ${uniq.map((n) => n + 'px').join(' / ')}`)
    }
  }
}

/* --- 输出 ---------------------------------------------------------------- */
if (verbose) {
  console.log('断点使用情况：')
  for (const n of notes) console.log('  ' + n)
}

if (problems.length) {
  console.error(`response check: ${problems.length} 项不通过`)
  for (const p of problems) console.error('  ✗ ' + p)
  process.exit(1)
}

console.log('response check: OK（studio.css 真源一致 · 页面均有窄屏规则 · 无 minmax 硬宽度）')
