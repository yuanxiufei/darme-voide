#!/usr/bin/env node
/**
 * 安装包体积预算守卫。
 *
 * 存在的意义：体积是**会悄悄涨的**指标（多一个插件 +1.2 MB、图标没压 +0.4 MB、
 * 前端多引一个 UI 库 +3 MB），没有闸门就一定会失守。所以把它做成与测试同级的检查项：
 * 超限 ⇒ 退出码 1（可直接进 CI）。
 *
 * 用法：
 *   node scripts/size-budget.mjs          # 有产物就核对，没有就提示
 *   node scripts/size-budget.mjs --strict # 缺产物也算失败（发布前跑这个）
 */
import { existsSync } from 'node:fs'
import { readdir, readFile, stat } from 'node:fs/promises'
import { dirname, extname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { gzipSync } from 'node:zlib'

const here = dirname(fileURLToPath(import.meta.url))
const desktopDir = resolve(here, '..')
const targetDir = join(desktopDir, 'src-tauri', 'target', 'release')
const distDir = join(desktopDir, 'dist-frontend')

const MB = 1024 * 1024
/**
 * 上限（MB）。指标依据、实测基线与收紧过程见 docs/desktop-tauri-selection.md §3.2。
 * 2026-11-02 Windows x64 首次真机出包实测 ⇒ 按「实测 × 约 2」收紧（只许收紧不许放宽）：
 *   壳 3.35 / NSIS 1.39 / 前端 1.05（gz 0.33）。
 *
 * 2026-11-02 二次出包（后端源码进包）：
 *   壳 3.36 / NSIS 3.56 / 后端源码 9.86（586 文件，已排除 .venv 3.2 GB、output、tests）。
 *   NSIS 这一档是**功能面扩张**（纯壳 → 壳 + 可运行的 backend-py 源码），不是同功能变胖，
 *   所以按「实测 + 约 25% 余量」定 4.5，而不是照 ×2 放到 7 MB（那等于把闸门关掉）。
 *   后端源码单列一项：它涨了就说明又混进了运行期用不上的东西，必须当场归因。
 * macOS 与 Linux 尚未真机出包，保留预估上限，实测后同样收紧。
 */
const BUDGET = {
  shellBinary: 6,
  frontendRaw: 2,
  frontendGz: 0.75,
  backendSource: 12,
  windowsNsis: 4.5,
  macosDmg: 14,
  linuxAppImage: 18,
  linuxDeb: 10,
}

const strict = process.argv.includes('--strict')

async function walk(dir) {
  if (!existsSync(dir)) return []
  const out = []
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name)
    if (entry.isDirectory()) out.push(...(await walk(full)))
    else if (entry.isFile()) out.push(full)
  }
  return out
}

const checks = []
function check(label, bytes, limitMb) {
  const mb = bytes / MB
  checks.push({ label, mb, limitMb, ok: mb <= limitMb })
}

/* 1) 壳二进制 */
const binaryNames =
  process.platform === 'win32'
    ? ['DramaStudio.exe', 'darme-voide-desktop.exe']
    : ['DramaStudio', 'darme-voide-desktop']
let binaryFound = false
for (const name of binaryNames) {
  const full = join(targetDir, name)
  if (existsSync(full)) {
    const info = await stat(full)
    check(`壳二进制 ${name}`, info.size, BUDGET.shellBinary)
    binaryFound = true
    break
  }
}

/* 2) 安装包 */
const bundleDir = join(targetDir, 'bundle')
for (const file of await walk(bundleDir)) {
  const ext = extname(file).toLowerCase()
  const size = (await stat(file)).size
  const rel = relative(bundleDir, file)
  if (ext === '.exe' && rel.includes('nsis')) check(`NSIS 安装包 ${rel}`, size, BUDGET.windowsNsis)
  else if (ext === '.msi') check(`MSI 安装包 ${rel}`, size, BUDGET.windowsNsis)
  else if (ext === '.dmg') check(`macOS DMG ${rel}`, size, BUDGET.macosDmg)
  else if (ext === '.appimage') check(`Linux AppImage ${rel}`, size, BUDGET.linuxAppImage)
  else if (ext === '.deb') check(`Linux deb ${rel}`, size, BUDGET.linuxDeb)
}

/* 3) 前端资源 */
let rawTotal = 0
let gzTotal = 0
let frontendFound = false
if (existsSync(distDir)) {
  const files = await walk(distDir)
  if (files.length) {
    frontendFound = true
    for (const file of files) {
      const size = (await stat(file)).size
      rawTotal += size
      const ext = extname(file).toLowerCase()
      const compressible = ['.js', '.css', '.html', '.json', '.svg', '.map'].includes(ext)
      gzTotal += compressible ? gzipSync(await readFile(file), { level: 9 }).length : size
    }
    check('前端资源（未压缩）', rawTotal, BUDGET.frontendRaw)
    check('前端资源（传输量估算）', gzTotal, BUDGET.frontendGz)
  }
}

/* 4) 打包进安装包的后端源码 */
const backendDir = join(desktopDir, 'dist-backend', 'backend-py')
if (existsSync(backendDir)) {
  const files = await walk(backendDir)
  let backendTotal = 0
  for (const file of files) backendTotal += (await stat(file)).size
  if (files.length) check(`后端源码 ${files.length} 个文件`, backendTotal, BUDGET.backendSource)
}

const mb = (n) => `${n.toFixed(2)} MB`

if (!checks.length) {
  console.log('[budget] 还没有可比对的产物。先跑：')
  console.log('         node scripts/gen-icons.mjs && node scripts/sync-frontend.mjs --build')
  console.log('         npx @tauri-apps/cli build')
  process.exit(strict ? 1 : 0)
}

console.log('[budget] 体积核对')
let failed = 0
for (const row of checks) {
  const status = row.ok ? 'OK  ' : 'OVER'
  if (!row.ok) failed++
  console.log(
    `  ${status} ${row.label.padEnd(44)} ${mb(row.mb).padStart(10)} / 上限 ${String(row.limitMb).padStart(5)} MB`,
  )
}

if (!binaryFound && !frontendFound) console.log('[budget] 提示：未找到壳二进制与前端资源，只核对了安装包。')

if (failed) {
  console.error(`[budget] ${failed} 项超限 —— 请先看 docs/desktop-tauri-selection.md §3 的瘦身顺序`)
  process.exit(1)
}
console.log('[budget] 全部在预算内')
