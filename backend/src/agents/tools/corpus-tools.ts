/**
 * 语料检索工具 —— 让 Agent 能「找相似镜头」
 *
 * 数据源：`data/prompt-corpus/_normalized/prompts.jsonl`
 * （由 `scripts/corpus/fetch-raw.mjs` + `normalize.mjs` 产出，见 scripts/README.md）
 *
 * ⚠️ 该目录**已 gitignore**（第三方语料不进仓库）⇒ 新克隆的仓库里文件不存在。
 * 因此本工具**必须在语料缺失时优雅降级**：返回 available=false + 提示，
 * 而不是抛错 —— 否则会让「没采语料」直接变成 Agent 跑不动。
 *
 * 打分口径与 `scripts/corpus/search.mjs` 保持一致（2-gram + 位置加权），
 * 改这里时请同步改那边，否则「命令行验证有效」与「Agent 实际效果」会分叉。
 */
import { createTool } from '@mastra/core/tools'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { z } from 'zod'

/** 从本文件位置上溯到仓库根（与 agents/skills.ts 的 SKILLS_DIR 同法，兼容任意 cwd / Docker） */
const CORPUS_JSONL = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../../../data/prompt-corpus/_normalized/prompts.jsonl',
)

interface CorpusRow {
  id: string
  source: string
  license: string
  lang: string
  title: string
  summary_zh: string
  prompt: string
  category: string
  tags: string[]
  mode: string
  duration: string
  aspect: string
}

/** 懒加载 + 进程内缓存：语料 ~2.5 MB 文本，启动时读一次即可（缺失则缓存 null，不反复 stat） */
let cache: CorpusRow[] | null | undefined

function loadCorpus(): CorpusRow[] | null {
  if (cache !== undefined) return cache
  if (!existsSync(CORPUS_JSONL)) {
    cache = null
    return cache
  }
  try {
    cache = readFileSync(CORPUS_JSONL, 'utf8')
      .split('\n')
      .filter(Boolean)
      .map((l) => JSON.parse(l) as CorpusRow)
  } catch {
    cache = null
  }
  return cache
}

/** 查询分词：拉丁词按词边界；CJK 取 2-gram（中文无空格，2-gram 是最省事且够用的近似） */
function tokenize(s: string): string[] {
  const out = new Set<string>()
  const lower = String(s).toLowerCase()
  for (const w of lower.match(/[a-z0-9][a-z0-9._-]+/g) || []) out.add(w)
  for (const seg of lower.replace(/[^\u4e00-\u9fff]+/g, ' ').split(/\s+/)) {
    if (!seg) continue
    if (seg.length === 1) out.add(seg)
    for (let i = 0; i + 1 < seg.length; i++) out.add(seg.slice(i, i + 2))
  }
  return [...out]
}

export function createCorpusTools() {
  const searchReferencePrompts = createTool({
    id: 'search_reference_prompts',
    description:
      '在本地提示词语料库（约 9000 条真实视频提示词，含 Seedance 短剧/电影级写法）中检索与当前镜头相似的参考写法。' +
      '当你需要具体可抄的运镜/光线/一致性写法，或想让画面描述更专业时调用；返回若干条参考提示词（已截断）。',
    inputSchema: z.object({
      query: z.string().describe('检索词，建议用中文关键词组合，例如「雨夜 霓虹 街道 特写 手持」'),
      top: z.number().optional().describe('返回条数，默认 3，最多 5'),
    }),
    execute: async ({ query, top }) => {
      const rows = loadCorpus()
      if (!rows) {
        return {
          available: false,
          results: [],
          hint:
            '本地语料未安装（data/prompt-corpus/ 已 gitignore，不在仓库内）。' +
            '请先执行 node scripts/corpus/fetch-raw.mjs 采集、node scripts/corpus/normalize.mjs 归一化。' +
            '在语料就绪前，请改用 SKILL 里已有的词表与锚点，不要假设能检索到参考。',
        }
      }

      const tokens = tokenize(query)
      if (!tokens.length) return { available: true, results: [], hint: '检索词为空。' }

      const limit = Math.min(Math.max(top ?? 3, 1), 5)
      const scored: Array<{ r: CorpusRow; sc: number; cover: number }> = []
      for (const r of rows) {
        const t = (r.title || '').toLowerCase()
        const tg = (r.tags || []).join(' ').toLowerCase()
        const c = (r.category || '').toLowerCase()
        const sm = (r.summary_zh || '').toLowerCase()
        const b = (r.prompt || '').toLowerCase()
        let sc = 0
        let hits = 0
        for (const k of tokens) {
          let got = 0
          if (t.includes(k)) got += 5
          if (tg.includes(k)) got += 3
          if (c.includes(k)) got += 2
          if (sm.includes(k)) got += 2
          if (b.includes(k)) got += 1
          if (got) hits++
          sc += got
        }
        if (sc > 0) scored.push({ r, sc, cover: hits / tokens.length })
      }
      scored.sort((a, b) => b.sc - a.sc)

      return {
        available: true,
        total_matched: scored.length,
        results: scored.slice(0, limit).map(({ r, sc, cover }) => ({
          score: sc,
          coverage: Number(cover.toFixed(2)),
          source: r.source,
          license: r.license,
          lang: r.lang,
          title: r.title,
          category: r.category,
          tags: (r.tags || []).slice(0, 8),
          mode: r.mode,
          duration: r.duration,
          aspect: r.aspect,
          // 截断：参考是给模型看的，过长会挤占上下文；要全文自行按 id 到语料里取
          prompt_excerpt: r.prompt.replace(/\s+/g, ' ').slice(0, 400),
        })),
        note: '以上为**参考写法**，不是可直接照抄的剧本内容：请只借鉴镜头/光线/结构手法，不要搬运原文（合规红线见 skills/video-prompt-library 6.9）。',
      }
    },
  })

  return { searchReferencePrompts }
}
