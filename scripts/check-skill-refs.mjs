#!/usr/bin/env node
/**
 * skills 库引用完整性检查 —— 防止「改名 / 挪库」造成的静默断链。
 *
 * 为什么需要它：skill 库按 `skills/<库名>/<skill名>/` 组织，库层级与 skill 名都可能变动。
 * 而正文里的路径引用**改坏了不会报任何错** —— 加载器只读 SKILL.md，没人会去点那些路径，
 * 断链只会在读者真的走到那一行时表现为「指向空处」。本脚本把引用全部拉出来逐个验存在性。
 *
 * 用法（仓库根目录）：  node scripts/check-skill-refs.mjs [--verbose]
 *   --verbose 额外列出被跳过的候选（上游路径 / 基准不明），便于审计本脚本自身的盲区。
 *   环境变量 `SKILL_REFS_DIR` 可把扫描根指向别处（仅供 `scripts/test-guards.mjs` 在副本上做负向实证）。
 * 退出码：0 = 无致命断链；1 = 存在致命断链（可用作提交前自检 / CI 步骤）
 *
 * ── 解析基准（本文档实测只有这几种，混用会被误判）────────────────────────
 *   ① `references/x.md` / `scripts/x.mjs` / `agents/x.yaml` —— 相对 **skill 根目录**（SKILL.md 所在目录）。
 *      注意：写在 `references/foo.md` 内部的这种引用同样指 skill 根，而不是 references/ 自身。
 *   ② `../other-skill/references/x.md` —— 相对 **当前文档目录**（真·相对路径）。
 *   ③ `backend/…` / `frontend/…` / `skills/…` / `docs/…` —— repo 根相对，无歧义。
 *
 * ── 分级 ────────────────────────────────────────────────────────────────
 *   · 致命：`references/…`、跨 skill `../…`、repo 根相对路径指向不存在 ⇒ 退出码 1。
 *   · 非致命：`scripts/…` 指向不存在。外部技能库只随行 SKILL.md + references/，
 *     上游 scripts/ 普遍未 vendored（且 scripts/ 与 references/ 一样对 Agent 不可达），
 *     故只列出、不判失败 —— 避免守卫长期红灯而被忽略。
 *   · 跳过：上游/外来宿主路径（`.ci/`、`spec/`、`.opencode-v2/`、`.claude/` 等）、
 *     含通配符的模式、**示意引用**（同一行紧邻的 `e.g.` / `such as` / `例如` 表明只是举例而非依赖，
 *     见 ILLUSTRATIVE_RE）、以及基准不明确的候选（不在任何 skill 内 / 非资产目录开头，如 `export/…`）。
 *     各类跳过都会分别计数；`--verbose` 可逐条审计，避免「跳过」变成静默丢弃。
 */
import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
/**
 * 可被环境变量 `SKILL_REFS_DIR` 覆盖 —— 供 `scripts/test-guards.mjs` 在**临时副本**上
 * 做负向实证（真仓库全程只读）。只覆盖 skills 树：`docs/` 等 repo 根相对引用
 * 仍按真实 REPO_ROOT 解析，故用例贴近实战。
 */
const SKILLS_DIR = process.env.SKILL_REFS_DIR
  ? path.resolve(process.env.SKILL_REFS_DIR)
  : path.join(REPO_ROOT, 'skills')
const VERBOSE = process.argv.includes('--verbose')

const FILE_EXT = 'md|json|ya?ml|txt|py|js|ts|mjs|cjs|sh|jsonl'
const PATH_TOKEN_RE = new RegExp(`^(?:\\.\\./)*[\\w.-]+(?:/[\\w.*-]+)*\\.(?:${FILE_EXT})$`)
/** 上游 Hub 仓库 / 外来宿主平台的专有前缀：本就不属于本项目 */
const UPSTREAM_PREFIXES = ['.ci/', 'spec/', 'hub-skill-market', '.opencode-v2/', '.agents/', '.claude/']
/**
 * 明确相对 repo 根的路径前缀。
 * `docs/` 于 2026-09-12 补入：skill 正文常引用 `docs/*.md` 作为「细节落点」
 * （如 prompt-style-library 指向 docs/prompt-style-sources.md），
 * 此前这类 token 落进「基准不明」被静默跳过 ⇒ 删/改 docs 文件名不会报任何错。
 */
const REPO_ROOT_PREFIXES = ['skills/', 'backend/', 'frontend/', 'docs/']
/** 基准为 skill 根的资产目录（文档不在任何 skill 内时基准不明，跳过） */
const SKILL_ROOT_ASSET_DIRS = ['references', 'scripts', 'agents']
/** 这些目录的缺失只提示、不判失败 */
const NON_FATAL_ASSET_DIRS = ['scripts']
/**
 * 「示意引用」标记：紧邻 token 之前的措辞表明该路径只是**举例**，不是本 skill 的依赖。
 * 典型：voiceover-direction「Any script path the user mentioned (e.g. `scripts/explainer-v3.md`)」
 * —— 不跳过就会误报「缺脚本」，长期把真信号淹没在噪声里。
 *
 * 必须**锚定在 token 紧前方**：标记与路径之间只允许「非字母数字、非汉字」的字符（空白、括号、
 * 反引号、冒号、逗号…—— 注意 token 前的开启反引号也在这一串里）。
 * 放宽成「同一行出现过 e.g.」会误伤真依赖 —— 实测构造 `无 e.g. 标记：\`references/x.md\``
 * 这类行时，该断链会被静默跳过（连守卫自己一起漏报），故收紧到紧邻。
 */
const ILLUSTRATIVE_RE = /(?:e\.g\.|eg\.|for example|such as|例如|比如|譬如)[^A-Za-z0-9\u4e00-\u9fa5]*$/i

function walk(dir, out = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) walk(p, out)
    else out.push(p)
  }
  return out
}

/** 路径候选 → 首次出现偏移：反引号 token + markdown 链接目标 */
function collectTokens(text) {
  const map = new Map()
  for (const re of [/`([^`\n]+)`/g, /\]\(([^)\n]+)\)/g]) {
    for (const m of text.matchAll(re)) {
      const token = m[1].trim()
      if (!map.has(token)) map.set(token, m.index + m[0].indexOf(m[1]))
    }
  }
  return map
}

/** 文档所属 skill 根（最近的存在 SKILL.md 的祖先）；不在任何 skill 内 → undefined */
function skillRootOf(docDir) {
  let d = docDir
  while (d.startsWith(SKILLS_DIR)) {
    if (fs.existsSync(path.join(d, 'SKILL.md'))) return d
    if (d === SKILLS_DIR) break
    d = path.dirname(d)
  }
  return undefined
}

/** 偏移 → 行号（二分，先建行首表避免逐 token 重扫全文） */
function makeLineLocator(text) {
  const starts = [0]
  for (let i = 0; i < text.length; i++) if (text[i] === '\n') starts.push(i + 1)
  return off => {
    let lo = 0
    let hi = starts.length - 1
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1
      if (starts[mid] <= off) lo = mid
      else hi = mid - 1
    }
    return lo + 1
  }
}

const rel = p => path.relative(REPO_ROOT, p).split(path.sep).join('/')
const docs = walk(SKILLS_DIR).filter(f => f.endsWith('.md'))
const fatal = []
const notes = []
/** 被跳过的候选 —— `--verbose` 时列出，用于审计守卫自身的盲区 */
const skipped = []
let checked = 0
let upstream = 0
let illustrative = 0
let ambiguous = 0

for (const doc of docs) {
  const text = fs.readFileSync(doc, 'utf8')
  const lineOf = makeLineLocator(text)
  const docDir = path.dirname(doc)
  const skillRoot = skillRootOf(docDir)

  for (const [token, offset] of collectTokens(text)) {
    if (!token.includes('/') || token.includes('*') || !PATH_TOKEN_RE.test(token)) continue
    const clean = token.replace(/^\.\//, '')
    const pos = `${rel(doc)}:${lineOf(offset)}`

    if (UPSTREAM_PREFIXES.some(p => clean.startsWith(p))) {
      upstream++
      skipped.push(`[上游/外来宿主路径] ${pos}  →  ${clean}`)
      continue
    }

    // 同一行内、token 之前若出现「举例」措辞 → 该路径泛指而非依赖，跳过（避免淹没真信号）
    if (ILLUSTRATIVE_RE.test(text.slice(text.lastIndexOf('\n', offset - 1) + 1, offset))) {
      illustrative++
      skipped.push(`[示意引用：举例而非依赖] ${pos}  →  ${clean}`)
      continue
    }

    const firstSeg = clean.replace(/^(?:\.\.\/)+/, '').split('/')[0]
    let target
    if (REPO_ROOT_PREFIXES.some(p => clean.startsWith(p))) {
      target = path.join(REPO_ROOT, clean)
    } else if (clean.startsWith('../')) {
      target = path.resolve(docDir, clean)
    } else if (SKILL_ROOT_ASSET_DIRS.includes(firstSeg)) {
      if (!skillRoot) {
        ambiguous++
        skipped.push(`[基准不明：不在任何 skill 内] ${pos}  →  ${clean}`)
        continue
      }
      target = path.resolve(skillRoot, clean)
    } else {
      ambiguous++
      skipped.push(`[基准不明：非资产目录开头] ${pos}  →  ${clean}`)
      continue
    }

    checked++
    if (fs.existsSync(target)) continue
    const item = { pos, token }
    if (NON_FATAL_ASSET_DIRS.includes(firstSeg)) notes.push(item)
    else fatal.push(item)
  }
}

for (const b of fatal.sort((a, b) => a.pos.localeCompare(b.pos))) console.log(`断链  ${b.pos}  →  ${b.token}`)
if (fatal.length) console.log('')
for (const b of notes.sort((a, b) => a.pos.localeCompare(b.pos))) console.log(`缺脚本（非致命）  ${b.pos}  →  ${b.token}`)
if (notes.length) console.log('')

if (VERBOSE) {
  for (const s of skipped.sort()) console.log(s)
  if (skipped.length) console.log('')
}

console.log(`扫描 skills/**/*.md 共 ${docs.length} 个文件`)
console.log(
  `待校验 ${checked} 处 ｜ 致命断链 ${fatal.length} 处 ｜ 非致命缺脚本 ${notes.length} 处 ｜ ` +
  `跳过：上游/外来宿主 ${upstream} 处 / 示意引用 ${illustrative} 处 / 基准不明 ${ambiguous} 处`
)
process.exit(fatal.length ? 1 : 0)
