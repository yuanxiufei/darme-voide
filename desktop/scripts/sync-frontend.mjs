#!/usr/bin/env node
/**
 * 桌面端前端产物的构建与体积统计。
 *
 * ## 2026-11 变更：桌面端不再使用 Nuxt 构建
 *
 * 旧链路：`frontend/` 跑 `nuxt generate`（带 `NUXT_PUBLIC_HASH_ROUTER=1`）
 *        → 产物在 `frontend/.output/public` → 复制到 `desktop/dist-frontend`。
 *        两个问题：
 *        1) 两端共享同一份源码，Web 端任何改动都会顺带改变桌面端行为
 *        2) 依赖 Nuxt 两个隐式契约（`router.options.hashMode`、`.output/public` 布局）
 *
 * 新链路：`desktop/frontend/` 是**独立 Vite 工程**，其 `build.outDir` 就是
 *        `desktop/dist-frontend` ⇒ 本脚本**不再需要复制步骤**，只做构建触发与校验。
 *
 * 归属关系：
 *   Web 端产物  → `frontend/.output/public`（`nuxt build/generate`）
 *   桌面端产物  → `desktop/dist-frontend`（`vite build`，本脚本管辖）
 *   两者物理隔离，互不读写。
 *
 * 用法：
 *   node scripts/sync-frontend.mjs            # 只校验 + 统计已有产物
 *   node scripts/sync-frontend.mjs --build    # 先构建（含 vue-tsc 类型检查）再校验
 *   node scripts/sync-frontend.mjs --build --fast   # 跳过类型检查，仅供本地快跑
 */
import { existsSync } from 'node:fs'
import { readdir, readFile, stat } from 'node:fs/promises'
import { dirname, extname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFileSync } from 'node:child_process'
import { gzipSync } from 'node:zlib'

const here = dirname(fileURLToPath(import.meta.url))
const desktopDir = resolve(here, '..')
const uiDir = join(desktopDir, 'frontend')
const targetDir = join(desktopDir, 'dist-frontend')

const flags = new Set(process.argv.slice(2))

if (flags.has('--build')) {
  const script = flags.has('--fast') ? 'build:fast' : 'build'
  console.log(`[sync] 构建桌面端界面（独立 Vite 工程，npm run ${script}）…`)
  execFileSync('npm', ['run', script], {
    cwd: uiDir,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  })
}

if (!existsSync(join(targetDir, 'index.html'))) {
  console.error(`[sync] 找不到桌面端产物：${join(targetDir, 'index.html')}`)
  console.error('       先跑 `npm run build`（在 desktop/frontend），或用本脚本的 --build')
  process.exit(1)
}

// 断言产物确实是本工程产出：Nuxt 产物会引用 /_nuxt/，Vite 产物引用 ./assets/
const html = await readFile(join(targetDir, 'index.html'), 'utf8')
if (/_nuxt\//.test(html)) {
  console.error('[sync] dist-frontend 里是 Nuxt 产物（含 /_nuxt/）—— 说明构建链没切过来')
  console.error('       桌面端必须由 desktop/frontend 的 vite build 产出')
  process.exit(1)
}
if (!/\.\/assets\//.test(html)) {
  console.error('[sync] index.html 不含 ./assets/ 相对引用，不符合桌面端产物特征')
  process.exit(1)
}

async function walk(dir) {
  const out = []
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name)
    if (entry.isDirectory()) out.push(...(await walk(full)))
    else if (entry.isFile()) out.push(full)
  }
  return out
}

const files = await walk(targetDir)
let raw = 0
let gz = 0
const rows = []
for (const file of files) {
  const info = await stat(file)
  raw += info.size
  // 只对文本类资源估 gzip；图片/字体本身已压缩，跳过以免误判预算
  const ext = extname(file).toLowerCase()
  const compressible = ['.js', '.css', '.html', '.json', '.svg', '.map'].includes(ext)
  const packed = compressible ? gzipSync(await readFile(file)).length : info.size
  gz += packed
  rows.push({ path: relative(targetDir, file), size: info.size, packed })
}

rows.sort((a, b) => b.size - a.size)
const mb = (n) => `${(n / 1024 / 1024).toFixed(2)} MB`
console.log(`[sync] ${files.length} 个文件 → ${targetDir}`)
console.log(`[sync] 原始 ${mb(raw)} ／ 传输量（估算）${mb(gz)}`)
console.log('[sync] 体积前 8：')
for (const row of rows.slice(0, 8)) {
  console.log(`        ${mb(row.size).padStart(9)}  ${row.path}`)
}
