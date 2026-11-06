#!/usr/bin/env node
/**
 * 把后端源码同步进桌面壳的打包资源目录 `dist-backend/backend-py`。
 *
 * 为什么需要这一步：`backend-py` 平摊在仓库里，里面混着运行期**绝不能进包**的东西
 * （`.venv` 3.2 GB、`.venv-cpu-backup` 0.7 GB、`output/`、`tmp/`、`__pycache__/`）。
 * 直接 `resources: ["../../backend-py/**"]` 会让安装包从 1.4 MB 涨到 GB 级，
 * 而用户机器上真正需要的只有 `app/**` + `requirements.txt`。
 *
 * 这也是「装完打不开」的第一个真原因：`resolve_backend_dir()` 以前只认仓库布局，
 * 装到别人机器上必然报「找不到后端目录」。现在它多认一条打包布局
 * `<resource_dir>/backend-py`，本脚本负责把那份目录喂饱。
 *
 * 用法：
 *   node scripts/sync-backend.mjs          # 同步源码
 *   node scripts/sync-backend.mjs --report # 只报体积，不复制
 */
import { existsSync } from 'node:fs'
import { copyFile, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const desktopDir = resolve(here, '..')
const repoRoot = resolve(desktopDir, '..')
const sourceDir = join(repoRoot, 'backend-py')
const targetDir = join(desktopDir, 'dist-backend', 'backend-py')

const reportOnly = process.argv.includes('--report')

/**
 * 运行期不需要、且体积或隐私代价明确的目录。
 * - `.venv*`：依赖树，本该由解释器侧提供（见 docs/desktop-tauri-selection.md §5）
 * - `output` / `tmp`：用户产物与临时文件
 * - `tests`：测试套件（15.9 MB），排障看日志就够
 * - `_bench`：引擎基准脚本与结果（2.3 MB），开发期工具，不进安装包
 * - `__pycache__` / `.pytest_cache` / `.mypy_cache` / `.ruff_cache`：可重建
 */
const EXCLUDE_DIRS = new Set([
  '.venv',
  '.venv-cpu-backup',
  '.pytest_cache',
  '.mypy_cache',
  '.ruff_cache',
  '__pycache__',
  'node_modules',
  'output',
  'tmp',
  'tests',
  '_bench',
])

/** 运行期不需要的散文件（探针脚本、备份、日志）。 */
const EXCLUDE_FILES = new Set(['_bwprobe.py', '_run_qwen_image.py'])

const EXCLUDE_EXTS = new Set(['.pyc', '.pyo', '.log', '.tmp'])

function excluded(entryPath, name) {
  if (EXCLUDE_DIRS.has(name)) return true
  if (EXCLUDE_FILES.has(name)) return true
  const dot = name.lastIndexOf('.')
  if (dot > 0 && EXCLUDE_EXTS.has(name.slice(dot).toLowerCase())) return true
  return false
}

/** 深度优先收集要复制/统计的文件。 */
async function walk(dir, out = []) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name)
    if (excluded(full, entry.name)) continue
    if (entry.isDirectory()) await walk(full, out)
    else if (entry.isFile()) out.push(full)
  }
  return out
}

if (!existsSync(sourceDir)) {
  console.error(`[backend] 找不到后端源码：${sourceDir}`)
  process.exit(1)
}

const files = await walk(sourceDir)
let total = 0
for (const file of files) total += (await stat(file)).size

const mb = (n) => `${(n / 1024 / 1024).toFixed(2)} MB`

if (!files.some((f) => relative(sourceDir, f).replaceAll('\\', '/') === 'app/main.py')) {
  console.error('[backend] 源码里没有 app/main.py —— backend-py 结构变了？')
  process.exit(1)
}

if (reportOnly) {
  console.log(`[backend] ${files.length} 个文件，合计 ${mb(total)}`)
  process.exit(0)
}

// 增量同步，**不整目录删除**：本地环境的批量删除保护会在一次删 500+ 个文件时中断构建，
// 而这里本来也不需要删 —— 源码删掉的文件靠 manifest 差集精确清理（通常 0~几 个）。
const manifestPath = join(desktopDir, 'dist-backend', '.manifest.json')
let previous = []
try {
  previous = JSON.parse(await readFile(manifestPath, 'utf8'))
} catch {
  // 首次运行或 manifest 损坏：当作空清单，只覆盖不删除
}

const current = files.map((file) => relative(sourceDir, file).replaceAll('\\', '/'))
const currentSet = new Set(current)

let removed = 0
for (const stale of previous) {
  if (currentSet.has(stale)) continue
  await rm(join(targetDir, stale), { force: true })
  removed += 1
}

for (const file of files) {
  const dest = join(targetDir, relative(sourceDir, file))
  await mkdir(dirname(dest), { recursive: true })
  await copyFile(file, dest)
}
await writeFile(manifestPath, JSON.stringify([...current].sort()))

// 顶层清单便于排障：装完发现少了东西时，一眼能看出打进包的是什么
const topLevel = [...new Set(files.map((f) => relative(sourceDir, f).split(/[\\/]/)[0]))].sort()
console.log(`[backend] ${files.length} 个文件 → ${targetDir}（清理陈旧 ${removed} 个）`)
console.log(`[backend] 合计 ${mb(total)}（排除 .venv / output / tmp / tests / _bench / __pycache__）`)
console.log(`[backend] 顶层：${topLevel.join('、')}`)
