#!/usr/bin/env node
/**
 * 语料归一化：各源原始文件 → 统一 JSONL（供检索层消费）
 *
 * ── 设计取舍（重要，勿随手加维度）────────────────────────────────────────
 * **只抽"能检索的最小字段集"**：id / source / license / lang / title / prompt / category / tags / mode / duration / aspect。
 *
 * 为什么不做 11 维全标注：1.2 万条 × 11 维是周级重活，而且**没有 ground truth 可验收**
 * 标注质量（质量评分维度更是需要"生成结果"才对得上，而语料里的配套视频我们没下）。
 * 而"能不能检索到相关镜头"这件事，用上面这几个字段就足以验证。
 * ⇒ 先证明检索有用，再按需补标注。路线讨论见 docs/video-prompt-data-sources.md。
 *
 * ── 数据来源与许可 ───────────────────────────────────────────────────────
 *   seedance-prompt-ericgood      CC BY 4.0（可商用，需署名）  结构：数组 JSON
 *   awesome-seedance-2-5-flaqai   MIT（宽松）                  结构：Markdown（层级不一致，见下）
 *
 * ⚠️ 原始正文**不得进仓库**（第三方版权红线）：本脚本产物落在 data/prompt-corpus/（已 gitignore）。
 *
 * 用法
 *   node scripts/corpus/normalize.mjs            # 归一化全部已采集的源
 *   node scripts/corpus/normalize.mjs --source=flaqai
 * 产物
 *   data/prompt-corpus/_normalized/prompts.jsonl   （每行一条，UTF-8）
 *   data/prompt-corpus/_normalized/report.txt      （字段填充率与分布，供人工核对）
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..')
const CORPUS = join(ROOT, 'data', 'prompt-corpus')
const OUT_DIR = join(CORPUS, '_normalized')
const ONLY = (process.argv.find((a) => a.startsWith('--source=')) || '').slice(9)

const report = []
const say = (s) => report.push(s)

const cjkCount = (s) => (String(s).match(/[\u4e00-\u9fff]/g) || []).length

/** 数组取首个非空：不同源字段名不同，用候选名列表兜住 */
function pick(obj, names) {
  for (const n of names) {
    for (const k of Object.keys(obj)) {
      if (k.toLowerCase() === n.toLowerCase()) {
        const v = obj[k]
        if (v !== undefined && v !== null && String(v).trim() !== '') return v
      }
    }
  }
  return undefined
}

/** 把任意 tag 形态（数组 / 逗号串 / 分号串）规整成字符串数组 */
function toTags(v) {
  if (Array.isArray(v)) return v.map((x) => String(x).trim()).filter(Boolean)
  if (typeof v === 'string') return v.split(/[,;，；|]/).map((s) => s.trim()).filter(Boolean)
  return []
}

// ── 源 1：Ericgood seedance-prompt（数组 JSON）─────────────────────────────
function parseEricgood() {
  const P = join(CORPUS, 'seedance-prompt-ericgood', 'raw', 'prompts.json')
  if (!existsSync(P)) return []
  const j = JSON.parse(readFileSync(P, 'utf8'))
  const arr = Array.isArray(j) ? j : j.prompts || j.data || j.items || []
  const out = []
  arr.forEach((r, i) => {
    const prompt = pick(r, ['prompt', 'prompt_zh', 'content', 'text', 'description'])
    if (!prompt || !String(prompt).trim()) return
    const zhTitle = String(r.title_zh ?? '').trim()
    const enTitle = String(r.title ?? '').trim()
    out.push({
      id: `ericgood:${String(r.id ?? i + 1)}`,
      source: 'seedance-prompt-ericgood',
      license: 'CC-BY-4.0',
      source_url: 'https://github.com/Ericgood/seedance-prompt',
      // 实测：该源 prompt 是**英文**（112 条里仅 1 条含中文），但带成套中文元数据
      // （title_zh / description_zh / tag_zh）⇒ prompt 语言记 en，
      // 中文侧另存 title / summary_zh / tags，保证中文检索能命中。
      lang: cjkCount(prompt) > 0 ? 'zh' : 'en',
      title: zhTitle || enTitle,
      title_en: enTitle,
      summary_zh: String(r.description_zh ?? '').trim(),
      prompt: String(prompt).trim(),
      category: String(r.category ?? '').trim(),
      tags: [...toTags(r.tag), ...toTags(r.tag_zh), String(r.model ?? '').trim()].filter(Boolean),
      mode: String(r.mode ?? '').trim(),
      duration: String(r.duration ?? '').trim(),
      aspect: String(r.aspect ?? '').trim(),
    })
  })
  return out
}

// ── 源 2：flaqai awesome-seedance-2-5（Markdown，中文 ### / 英文 ## 混用）──
const SCENE_RE = /^(#{2,3})\s*(\d+)\.\s*(.+?)\s*$/
const META_RE = /\*\*([^*]+?)\s*[:：]\s*\*\*\s*([^·\n]+)/g
const FENCE_RE = /```(?:text|txt|markdown)?\s*\n([\s\S]*?)```/

function parseFlaqai() {
  const dir = join(CORPUS, 'awesome-seedance-2-5-flaqai', 'raw', 'prompts')
  if (!existsSync(dir)) return []
  const out = []
  for (const f of readdirSync(dir).filter((x) => x.endsWith('.md') && x !== 'README.md')) {
    const text = readFileSync(join(dir, f), 'utf8')
    // ⚠️ 必须**全文**切块：```text 围栏跨行，逐行匹配永远命中不了（首版即踩此坑，产出 0 条）
    const marks = []
    for (const m of text.matchAll(new RegExp(SCENE_RE.source, 'gm'))) {
      marks.push({ index: m.index, num: m[2], title: m[3] })
    }
    // 场景可能用 ## 或 ###（中文 ###、英文 ##），故分类取「本标题之前最近的 非场景 标题行」
    const heads = []
    for (const m of text.matchAll(/^#{1,2}\s+(.+?)\s*$/gm)) {
      if (!SCENE_RE.test(m[0])) heads.push({ index: m.index, text: m[1].trim() })
    }
    const catOf = (idx) => {
      let cur = ''
      for (const h of heads) if (h.index < idx) cur = h.text
      return cur
    }
    marks.forEach((mk, i) => {
      const end = i + 1 < marks.length ? marks[i + 1].index : text.length
      const block = text.slice(mk.index, end)
      const fence = block.match(FENCE_RE)
      const body = fence ? fence[1].trim() : ''
      if (!body) return
      const meta = {}
      for (const line of block.split('\n')) {
        if (line.includes('```')) break
        for (const mm of line.matchAll(META_RE)) meta[mm[1].trim().toLowerCase()] = mm[2].trim()
      }
      out.push({
        id: `flaqai:${mk.num}`,
        source: 'awesome-seedance-2-5-flaqai',
        license: 'MIT',
        source_url: 'https://github.com/flaqai/awesome_seedance_2_5',
        lang: cjkCount(body) > 0 ? 'zh' : 'en',
        title: mk.title,
        summary_zh: '',
        prompt: body,
        category: catOf(mk.index),
        tags: [f.replace(/\.md$/, '')],
        mode: meta['mode'] || meta['模式'] || '',
        duration: meta['duration'] || meta['时长'] || '',
        aspect: meta['format'] || meta['画幅'] || '',
      })
    })
  }
  return out
}

// ── 源 3：Seedance 2（本地已下载的 8755 条 JSONL）─────────────────────────
function parseSeedance2() {
  const P = join(CORPUS, 'seedance2', 'metadata.jsonl')
  if (!existsSync(P)) return []
  const out = []
  readFileSync(P, 'utf8')
    .split('\n')
    .forEach((l, i) => {
      if (!l.trim()) return
      let r
      try {
        r = JSON.parse(l)
      } catch {
        return
      }
      const z = (r.i18n || {}).zh || {}
      const p = typeof z.p === 'string' && z.p.trim() ? z.p : r.raw_p
      if (!p || !String(p).trim()) return
      const d = (r.spec || {}).duration
      // ⚠️ spec.duration 有脏值（最小 -3.69e17、最大 887.75）⇒ 只在 0 < d <= 300 时采信
      const dur = typeof d === 'number' && d > 0 && d <= 300 ? Math.round(d) + 's' : ''
      out.push({
        id: `sd2:${i + 1}`,
        source: 'seedance2',
        license: 'CC-BY-4.0',
        source_url: 'https://huggingface.co/datasets/GokuScraper/seedance-2-prompts-datasets',
        lang: cjkCount(p) > 0 ? 'zh' : 'en',
        title: String(z.t || '').trim(),
        summary_zh: '',
        prompt: String(p).trim(),
        category: String(r.category || '').trim(),
        tags: [...toTags(z.tags), String((r.model_info || {}).name || '').trim()].filter(Boolean),
        mode: '',
        duration: dur,
        aspect: (r.spec || {}).ratio ? String(r.spec.ratio) : '',
      })
    })
  return out
}

const SOURCES = [
  { key: 'ericgood', fn: parseEricgood },
  { key: 'flaqai', fn: parseFlaqai },
  { key: 'seedance2', fn: parseSeedance2 },
]

let all = []
say(`# 语料归一化 ${new Date().toISOString()}`)
for (const s of SOURCES) {
  if (ONLY && !s.key.startsWith(ONLY)) continue
  const rows = s.fn()
  say(`\n=== ${s.key}  → ${rows.length} 条`)
  const fields = ['title', 'category', 'mode', 'duration', 'aspect', 'tags']
  for (const f of fields) {
    const filled = rows.filter((r) => (Array.isArray(r[f]) ? r[f].length : String(r[f] || '').trim())).length
    say(`  fill ${f.padEnd(10)} ${filled}/${rows.length}`)
  }
  const zh = rows.filter((r) => r.lang === 'zh').length
  say(`  lang zh/en   ${zh}/${rows.length - zh}`)
  const lens = rows.map((r) => r.prompt.length).sort((a, b) => a - b)
  if (lens.length) say(`  prompt len min/med/max = ${lens[0]} / ${lens[Math.floor(lens.length / 2)]} / ${lens[lens.length - 1]}`)
  all = all.concat(rows)
}

mkdirSync(OUT_DIR, { recursive: true })
writeFileSync(join(OUT_DIR, 'prompts.jsonl'), all.map((r) => JSON.stringify(r)).join('\n'), 'utf8')
say(`\n# 合计 ${all.length} 条 → data/prompt-corpus/_normalized/prompts.jsonl`)
writeFileSync(join(OUT_DIR, 'report.txt'), report.join('\n'), 'utf8')
console.log(report.join('\n'))
console.log(`\nTOTAL=${all.length}`)
