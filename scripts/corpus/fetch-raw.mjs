#!/usr/bin/env node
/**
 * 语料采集（只下载原始文件，不解析）—— 第一阶段
 *
 * 为什么单独一步
 *   下载受网络波动影响最大，且**必须可重跑**；解析则依赖对每个源结构的实测。
 *   把两者分开：本脚本只负责"把原始字节拿到本地并记账"，解析器随后按实测定稿。
 *
 * 落盘约定（沿用项目既有规矩，见 scripts/README.md「语料分析脚本」）
 *   data/prompt-corpus/<源 id>/raw/<原始相对路径>   ← 原始文件（data/prompt-corpus/ 已 gitignore）
 *   data/prompt-corpus/<源 id>/SOURCE.md            ← 来源 / 许可 / 采集时间 记账（可追溯，**要进仓库**）
 *
 * 为什么把 SOURCE.md 写进 gitignored 目录之外
 *   ⚠️ 原始文件因第三方版权**不得入库**，但「这些文件是从哪来的、什么许可」必须留痕
 *   （CC BY 4.0 要求署名）。所以 SOURCE.md 落在 data/prompt-corpus/ 下也会被忽略 ——
 *   需要溯源时按本脚本头的 SOURCES 表重建即可，故这里同时打印到日志。
 *
 * 用法
 *   node scripts/corpus/fetch-raw.mjs            # 跳过已存在的文件
 *   node scripts/corpus/fetch-raw.mjs --force    # 重新下载
 *   node scripts/corpus/fetch-raw.mjs --only=<id 前缀>   # 只跑某个源
 *
 * 退出码：0 = 全部成功（含跳过）；1 = 有源失败（失败清单在日志末尾）
 */
import { existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..')
const CORPUS = join(ROOT, 'data', 'prompt-corpus')
const FORCE = process.argv.includes('--force')
const ONLY = (process.argv.find((a) => a.startsWith('--only=')) || '').slice(7)

/**
 * 只收**许可证明确可商用或可内部使用**的源。
 * ⚠️ 已核实但**刻意不收**的源（勿随手加回来）：
 *   - TIP-I2V                CC BY-NC 4.0（禁商用）→ 只能内部统计，不进语料库
 *   - Semonxue/awesome-video-prompts   无 LICENSE（4.88 GB）→ 仅内部参考
 *   - HitPaw-Official / geekjourneyx / fantasylights  实测为空壳（24 KB / 24 KB / 4 KB，0 条数据）
 * 见 docs/video-prompt-data-sources.md。
 */
const SOURCES = [
  {
    id: 'seedance-prompt-ericgood',
    label: 'Seedance Prompt (Ericgood)',
    license: 'CC-BY-4.0',
    repo: 'https://github.com/Ericgood/seedance-prompt',
    base: 'https://raw.githubusercontent.com/Ericgood/seedance-prompt/main/',
    /** prompts.json 是唯一数据文件；assets/ 与 videos/ 是媒体，**不采** */
    files: ['prompts.json', 'LICENSE'],
  },
  {
    id: 'awesome-seedance-2-5-flaqai',
    label: 'Awesome Seedance 2.5 (flaqai)',
    license: 'MIT',
    repo: 'https://github.com/flaqai/awesome_seedance_2_5',
    base: 'https://raw.githubusercontent.com/flaqai/awesome_seedance_2_5/main/',
    /** 120 个场景分布在 7 个 markdown 里；i18n/ 15 语言暂不采（体积大、增益低） */
    files: [
      'prompts/README.md',
      'prompts/prompt-library.md',
      'prompts/extended-scenarios.md',
      'prompts/advanced-workflows.en.md',
      'prompts/creative-techniques.en.md',
      'prompts/genre-social-experiments.en.md',
      'prompts/multilingual-pack.md',
      'LICENSE',
    ],
  },
]

const log = []
const say = (s) => {
  log.push(s)
  console.log(s)
}

/** 下载单个文件到 dest；已存在且非 --force 时跳过（下载常被转后台，跳过逻辑保证可重入） */
async function fetchOne(url, dest) {
  if (existsSync(dest) && !FORCE) return { skipped: true, bytes: statSync(dest).size }
  const res = await fetch(url, { redirect: 'follow' })
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`)
  const buf = Buffer.from(await res.arrayBuffer())
  if (buf.length === 0) throw new Error('empty body')
  mkdirSync(dirname(dest), { recursive: true })
  writeFileSync(dest, buf)
  return { skipped: false, bytes: buf.length }
}

const failed = []
say(`# 语料采集 ${new Date().toISOString()}`)
say(`# force=${FORCE} only=${ONLY || '(all)'}`)

for (const s of SOURCES) {
  if (ONLY && !s.id.startsWith(ONLY)) continue
  say('')
  say(`=== ${s.id}  [${s.license}]  ${s.repo}`)
  for (const f of s.files) {
    const dest = join(CORPUS, s.id, 'raw', f)
    try {
      const r = await fetchOne(s.base + f, dest)
      say(`  ${r.skipped ? 'skip' : 'get '}  ${String(r.bytes).padStart(8)} B  ${f}`)
    } catch (e) {
      failed.push(`${s.id}/${f}`)
      say(`  FAIL  ${f}  -> ${e.message}`)
    }
  }
}

say('')
say(failed.length ? `# 失败 ${failed.length} 个：${failed.join(', ')}` : '# 全部成功')

mkdirSync(join(ROOT, 'tmp'), { recursive: true })
writeFileSync(join(ROOT, 'tmp', 'fetch-raw.log'), log.join('\n'), 'utf8')
process.exit(failed.length ? 1 : 0)
