#!/usr/bin/env node
/**
 * 语料检索（最小闭环验证）—— 归一化后的本地 prompt 库上做「找相似镜头」
 *
 * 定位：**先证明检索有没有用**，不追求工业级检索。
 *   8987 条 / 约 2.5 MB 文本，全量读内存 + 命中打分就是毫秒级，**无需 SQLite / 向量库**。
 *   若实测有效，再决定要不要上 embedding（本地已装 `nomic-embed-text`）与持久索引。
 *
 * 打分口径（可调，别在没数据前过度设计）
 *   标题命中 5 ・ tags 命中 3 ・ 分类命中 2 ・ 中文摘要命中 2 ・ 正文命中 1
 *   中文用 **2-gram** 切词：中文无空格，2-gram 是最省事且够用的近似
 *   （"雨夜霓虹" → 雨夜/夜霓/霓虹，因此「霓虹雨夜」也能互相命中）。
 *
 * 用法（⚠️ PowerShell 直接传中文参数会乱码 ⇒ 优先用 --q-file）
 *   node scripts/corpus/search.mjs --q-file=tmp/q.txt --top=5
 *   node scripts/corpus/search.mjs "rainy neon street" --top=5
 *   $env:CORPUS_QUERY='雨夜霓虹'; node scripts/corpus/search.mjs      # 环境变量亦可
 *
 * 参数：--top=N（默认 5）｜--lang=zh|en｜--source=<id 前缀>｜--full（打印完整 prompt）
 */
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..')
const JSONL = join(ROOT, 'data', 'prompt-corpus', '_normalized', 'prompts.jsonl')
const arg = (n, d) => {
  const hit = process.argv.find((a) => a.startsWith(`--${n}=`))
  return hit ? hit.slice(n.length + 3) : d
}
const qFile = arg('q-file', '')
const TOP = Number(arg('top', '5'))
const LANG = arg('lang', '')
const SOURCE = arg('source', '')
const FULL = process.argv.includes('--full')

let QUERY = arg('q', '') || process.env.CORPUS_QUERY || process.argv[2] || ''
if (qFile && existsSync(join(ROOT, qFile))) QUERY = readFileSync(join(ROOT, qFile), 'utf8').trim()
if (!QUERY) {
  console.error('缺少查询词。用 --q-file=tmp/q.txt 或 CORPUS_QUERY 环境变量（PowerShell 直传中文会乱码）')
  process.exit(2)
}

if (!existsSync(JSONL)) {
  console.error('缺少 data/prompt-corpus/_normalized/prompts.jsonl —— 先跑 node scripts/corpus/normalize.mjs')
  process.exit(2)
}
let rows = readFileSync(JSONL, 'utf8')
  .split('\n')
  .filter(Boolean)
  .map((l) => {
    try {
      return JSON.parse(l)
    } catch {
      return null
    }
  })
  .filter(Boolean)
if (LANG) rows = rows.filter((r) => r.lang === LANG)
if (SOURCE) rows = rows.filter((r) => String(r.source).startsWith(SOURCE))

/** 查询分词：拉丁词按词边界；CJK 取 2-gram */
function tokenize(s) {
  const out = new Set()
  const lower = String(s).toLowerCase()
  for (const w of lower.match(/[a-z0-9][a-z0-9._-]+/g) || []) out.add(w)
  for (const seg of lower.replace(/[^\u4e00-\u9fff]+/g, ' ').split(/\s+/)) {
    if (!seg) continue
    if (seg.length === 1) out.add(seg)
    for (let i = 0; i + 1 < seg.length; i++) out.add(seg.slice(i, i + 2))
  }
  return [...out]
}

const tokens = tokenize(QUERY)
const scored = []
for (const r of rows) {
  const t = (r.title || '').toLowerCase()
  const tg = (r.tags || []).join(' ').toLowerCase()
  const c = (r.category || '').toLowerCase()
  const s = (r.summary_zh || '').toLowerCase()
  const b = (r.prompt || '').toLowerCase()
  let sc = 0
  let hits = 0
  for (const k of tokens) {
    let got = 0
    if (t.includes(k)) got += 5
    if (tg.includes(k)) got += 3
    if (c.includes(k)) got += 2
    if (s.includes(k)) got += 2
    if (b.includes(k)) got += 1
    if (got) hits++
    sc += got
  }
  if (sc > 0) scored.push({ r, sc, hits, cover: hits / tokens.length })
}
scored.sort((a, b) => b.sc - a.sc)

console.log(`查询：${QUERY}`)
console.log(`词元 ${tokens.length} 个 ｜ 候选 ${rows.length} 条 ｜ 命中 ${scored.length} 条\n`)
for (const { r, sc, cover } of scored.slice(0, TOP)) {
  console.log(`[${sc} 分 · 覆盖 ${(cover * 100).toFixed(0)}%] ${r.source} · ${r.lang} · ${r.category || '-'}`)
  console.log(`  ${r.title || '(无标题)'}${r.mode ? '  ·  ' + r.mode : ''}${r.duration ? ' · ' + r.duration : ''}${r.aspect ? ' · ' + r.aspect : ''}`)
  const body = FULL ? r.prompt : r.prompt.replace(/\s+/g, ' ').slice(0, 220) + (r.prompt.length > 220 ? ' …' : '')
  console.log(`  ${body.split('\n').join('\n  ')}`)
  if (r.tags && r.tags.length) console.log(`  tags: ${r.tags.slice(0, 8).join(', ')}`)
  console.log('')
}
