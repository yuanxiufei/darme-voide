#!/usr/bin/env node
/**
 * 桌面端 ↔ Web 端的**隔离守卫**。
 *
 * ## 为什么需要它
 *
 * `desktop/frontend/src/` 的代码是从 `frontend/app/` **复制**来的。复制动作本身
 * 不产生耦合，真正产生耦合的是复制之后有人图省事写了一句：
 *
 *     import { useApi } from '../../../frontend/app/composables/useApi'
 *
 * 这一句会让两个工程重新绑死：改 Web 端的文件会悄悄改变桌面端行为，而桌面端
 * 构建时又必须能读到 `frontend/`。守卫就是把这条线钉死。
 *
 * ## 七道判据（全部机械可判，不依赖人读）
 *
 * A. 源码越界：`src/**` 里的相对/别名 import 解析后必须落在 `src/` 内
 * B. Nuxt 残留：不得出现 Nuxt 专有 API（Vite 下会直接构建失败或静默不生效）
 * B2. `useState` 例外：允许调用，但实现必须来自本工程 `utils/nuxtCompat.ts`
 * C. 依赖隔离：package.json 不得依赖 nuxt，不得用 file:/link:/workspace: 指向本仓其他工程
 * D. 产物归属：dist-frontend/index.html 必须是 Vite 产物（含 ./assets/ 相对引用）
 * E. 锁文件独立：本工程有自己的 package-lock.json，且不含 Web 端条目
 * F. 媒体地址：不得手拼根相对路径（`'/' + p`），也不得把后端字段**裸绑**进 `:src`，
 *    必须过 `utils/apiBase.ts::mediaUrl`（画布卡片的 `node.url` 就在这条线上）
 * G. 外链资源：CSS/HTML 里不得出现 http(s) 的外链样式、字体、脚本、图片
 *
 * 注意：判据 B 会**先剥离注释**再匹配 —— 迁移说明里会正当地提到 NuxtLink 等名字。
 *
 * 用法：node scripts/check-isolation.mjs [--deps-only] [--dist]
 * 退出码：0 = 全绿；1 = 有 FAIL
 */
import { readdir, readFile, stat } from 'node:fs/promises'
import { dirname, join, relative, resolve, extname, isAbsolute } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '..') // desktop/frontend
const SRC = join(ROOT, 'src')
const REPO = resolve(ROOT, '../..') // darme-voide
const DIST = resolve(ROOT, '../dist-frontend') // desktop/dist-frontend

const argv = process.argv.slice(2)
const DEPS_ONLY = argv.includes('--deps-only')
const CHECK_DIST = argv.includes('--dist')

const fail = []
const pass = []
const ok = (msg) => pass.push(msg)
const bad = (msg) => fail.push(msg)

/** Nuxt 专有 API：一旦出现在本工程源码里，说明是复制未清理或又抄回去了 */
const NUXT_TOKENS = [
  'NuxtLink',
  'NuxtPage',
  'NuxtLayout',
  'definePageMeta',
  'defineNuxtPlugin',
  'useRuntimeConfig',
  'useAsyncData',
  'useFetch',
  'useHead',
  'useNuxtApp',
  "from '#app'",
  "from '#imports'",
  'import.meta.client',
  'import.meta.server',
]

/**
 * 媒体地址手拼的两类写法：`'/' + p`、`` `/` + p ``、`` `/static/${p}` ``。
 *
 * 后端返回的媒体路径是根相对的（`/static/...`）。Web 同源下这样写没错，
 * 桌面壳的页面 origin 却是 `tauri://localhost` ⇒ 打包后请求打到壳自己的静态目录、
 * 整片裂图；而 dev 阶段会被 Vite 的 `/static` proxy 兜住，**看不出来**。
 */
const MEDIA_CONCAT = /(['"`])\/\1\s*\+/
const MEDIA_TEMPLATE = /`\/\$\{/
/** 内联样式里的图片插值：`backgroundImage: \`url(${x})\`` —— 形如 `'/' + p` 的第二形态 */
const CSS_URL_INTERP = /url\(\s*["']?\$\{/
/**
 * 第三形态：模板绑定 `:src="x"` / `:poster="x"` 直接把后端字段塞进标签属性。
 * 前面两条判据只认字符串拼接与模板串插值，`<img :src="item.imageUrl">` 一条都盖不到
 * ——`pages/library/props.vue` 就这么漏了整整一轮（打包端道具缩略图全裂）。
 * 判据：绑定表达式里出现后端媒体字段名，且同一行没有收口函数。
 */
const MEDIA_BIND_ATTR = /:(?:src|poster)\s*=\s*"([^"]*)"/
const MEDIA_FIELD_NAME =
  /[Ii]mage_?[Uu]rl|\basset_url\b|\blocalPath\b|[Aa]udioUrl|\bmergeUrl\b|\bnode\.url\b|get(?:StoryboardCover|FirstFrame|Keyframe|LastFrame)/
const MEDIA_HELPER = /(?:mediaUrl|resolveImageUrl|resolveUrl|normPUrl|normEpUrl)\s*\(/

/**
 * 外链资源：桌面壳的 CSP 只放行 `'self' data: blob:` 与 `http://127.0.0.1`/`localhost`，
 * 任何 `https://` 的样式表、字体、脚本、图片都会被拦（实测：`studio.css` 里的
 * Google Fonts `@import` 让打包端每次启动都报一条 security error，字体静默失效）。
 * 这里只盯**资源类**引用；`<a href="https://…">` 这类对外跳转不在此列（CSP 不管导航）。
 */
const EXTERNAL_CSS = /@import\s+(?:url\(\s*)?['"]?https?:\/\//i
const EXTERNAL_CSS_URL = /url\(\s*['"]?https?:\/\//i
const EXTERNAL_TAG =
  /<(?:link|script|img|source|video|audio|iframe|embed)\b[^>]*\s(?:src|href|poster)\s*=\s*['"]https?:\/\//i

/** 允许的裸模块（必须出现在 package.json 的 deps 里） */
const ALLOWED_EXTERNAL = new Set(['vue', 'vue-router', 'vue-sonner', 'lucide-vue-next'])

const CODE_EXT = new Set(['.ts', '.vue', '.js', '.mjs', '.tsx'])

/** 守卫 G 要扫的扩展名（比 CODE_EXT 多出真正会写 `url()` 的样式与入口 HTML） */
const RESOURCE_EXT = new Set([...CODE_EXT, '.css', '.html'])

async function walk(dir, exts = CODE_EXT) {
  const out = []
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name)
    if (entry.isDirectory()) {
      if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue
      out.push(...(await walk(p, exts)))
    } else if (exts.has(extname(entry.name))) {
      out.push(p)
    }
  }
  return out
}

/**
 * 剥离注释（HTML 多行 / 块 / 行）。
 * 必须剥离：本工程大量文件头注释在解释"这里替代了 Nuxt 的什么"，
 * 直接匹配会让守卫对自己写的文档产生误报。
 */
function stripComments(code) {
  return code
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^[ \t]*\/\/.*$/gm, '')
}

/** 抽取源码里的 import/export 来源说明符（静态 import + 动态 import + require） */
function specifiersOf(code) {
  const specs = []
  const re = /(?:^|\n)\s*(?:import|export)[\s\S]*?from\s+['"]([^'"]+)['"]/g
  let m
  while ((m = re.exec(code))) specs.push(m[1])
  const re2 = /import\s*\(\s*['"]([^'"]+)['"]\s*\)/g
  while ((m = re2.exec(code))) specs.push(m[1])
  const re3 = /require\s*\(\s*['"]([^'"]+)['"]\s*\)/g
  while ((m = re3.exec(code))) specs.push(m[1])
  return specs
}

/** 把别名/相对说明符解析回工程内的真实路径；返回 null 表示是裸模块 */
function resolveSpec(spec, fromFile) {
  if (spec.startsWith('~/') || spec.startsWith('@/')) return join(SRC, spec.slice(2))
  if (spec === '~contracts') return join(SRC, 'types/contracts.ts')
  if (spec.startsWith('.')) return resolve(dirname(fromFile), spec)
  return null
}

function inside(child, parent) {
  const rel = relative(parent, child)
  return rel !== '' && !rel.startsWith('..') && !isAbsolute(rel)
}

// ---------------------------------------------------------------- A + B + B2
async function checkSource() {
  const files = await walk(SRC)
  const violations = []
  const nuxtHits = []
  const stateUsers = []
  const mediaHits = []

  for (const file of files) {
    const raw = await readFile(file, 'utf8')
    const code = stripComments(raw)
    const rel = relative(ROOT, file).replace(/\\/g, '/')

    // A. 越界 import
    for (const spec of specifiersOf(code)) {
      const resolved = resolveSpec(spec, file)
      if (resolved === null) {
        const pkg = spec.startsWith('@')
          ? spec.split('/').slice(0, 2).join('/')
          : spec.split('/')[0]
        if (!ALLOWED_EXTERNAL.has(pkg) && !pkg.startsWith('node:')) {
          violations.push(`${rel}: 非白名单裸模块 "${spec}"`)
        }
        continue
      }
      if (!inside(resolved, SRC)) {
        violations.push(`${rel}: import "${spec}" 解析到工程外 (${relative(REPO, resolved)})`)
      }
    }

    // B. Nuxt 残留（已剥注释）
    for (const token of NUXT_TOKENS) {
      if (code.includes(token)) nuxtHits.push(`${rel}: "${token}"`)
    }

    // B2. useState 调用登记（实现是否合法在循环外统一判）
    if (/(^|[^\w.])useState\s*\(/.test(code)) stateUsers.push(rel)

    // F. 媒体地址手拼（模板字符串 / 字符串拼接 / CSS 里的 url(${...}) 插值）
    if (MEDIA_CONCAT.test(code) || MEDIA_TEMPLATE.test(code)) mediaHits.push(rel)
    raw.split(/\r?\n/).forEach((line, i) => {
      const t = line.trim()
      if (t.startsWith('//') || t.startsWith('*') || t.startsWith('/*')) return
      if (CSS_URL_INTERP.test(line) && !MEDIA_HELPER.test(line)) {
        mediaHits.push(`${rel}:${i + 1}（内联样式 url() 插值未走 mediaUrl）`)
        return
      }
      const bind = MEDIA_BIND_ATTR.exec(line)
      if (bind && MEDIA_FIELD_NAME.test(bind[1]) && !MEDIA_HELPER.test(bind[1])) {
        mediaHits.push(`${rel}:${i + 1}（模板绑定 ${bind[1].trim()} 未走 mediaUrl）`)
      }
    })
  }

  if (violations.length) {
    bad(`A. 源码越界 import ${violations.length} 处`)
    violations.forEach((v) => fail.push('    ' + v))
  } else {
    ok(`A. 源码边界：${files.length} 个文件，零越界 import`)
  }

  if (nuxtHits.length) {
    bad(`B. Nuxt 专有 API 残留 ${nuxtHits.length} 处`)
    nuxtHits.forEach((v) => fail.push('    ' + v))
  } else {
    ok('B. 无 Nuxt 专有 API 残留')
  }

  // B2：useState 是允许的（本工程提供同名 shim），但实现必须自己拥有
  if (stateUsers.length) {
    let shimOk = false
    try {
      const shim = await readFile(join(SRC, 'utils/nuxtCompat.ts'), 'utf8')
      shimOk = /export function useState/.test(shim) && /export function navigateTo/.test(shim)
    } catch {
      shimOk = false
    }
    if (shimOk) {
      ok(`B2. useState/navigateTo 由本工程 shim 提供（useState 调用 ${stateUsers.length} 处）`)
    } else {
      bad(`B2. 有 ${stateUsers.length} 处 useState 调用，但 utils/nuxtCompat.ts 未提供完整实现`)
    }
  }

  //: F 自检 ✓✗：字段名正则必须认得画布节点的 `node.url`。上一轮正是漏了它 ——
  //: `<img/video/audio :src="props.node.url">` 全是**裸绑后端字段**，一条判据都不触发，
  //: 打包后画布上的图片/视频/音频卡整片 404（dev 被 Vite 的 `/static` proxy 掩盖 ✗）。
  //: 这一步不依赖人读：正则哪天被改窄了，这条先红 ✓。
  if (!MEDIA_FIELD_NAME.test(':src="props.node.url"')) {
    fail.push('    F 判据自检失败：MEDIA_FIELD_NAME 认不出 node.url（画布媒体卡的裸绑会静默漏掉 ✗）')
  }

  if (mediaHits.length) {
    bad(`F. 媒体地址手拼根相对路径 ${mediaHits.length} 处（须改用 utils/apiBase.ts 的 mediaUrl）`)
    mediaHits.forEach((v) => fail.push('    ' + v))
  } else {
    ok('F. 媒体地址统一走 mediaUrl（无根相对手拼，含模板裸绑后端字段）')
  }

  // G. 外链资源（CSP 会拦，且断网时必然失效 ⇒ 一律改本地）
  const resourceFiles = await walk(SRC, RESOURCE_EXT)
  const externalHits = []
  for (const file of resourceFiles) {
    const code = stripComments(await readFile(file, 'utf8'))
    const rel = relative(ROOT, file).replace(/\\/g, '/')
    if (EXTERNAL_CSS.test(code) || EXTERNAL_CSS_URL.test(code) || EXTERNAL_TAG.test(code)) {
      externalHits.push(rel)
    }
  }
  if (externalHits.length) {
    bad(`G. CSS/HTML 里有 http(s) 外链资源 ${externalHits.length} 处（CSP 会拦、断网必失效）`)
    externalHits.forEach((v) => fail.push('    ' + v))
  } else {
    ok(`G. 无外链运行时资源（扫了 ${resourceFiles.length} 个样式/模板/脚本文件）`)
  }
}

// ---------------------------------------------------------------- C + E
async function checkDeps() {
  const pkg = JSON.parse(await readFile(join(ROOT, 'package.json'), 'utf8'))
  const all = { ...(pkg.dependencies || {}), ...(pkg.devDependencies || {}) }

  const problems = []
  if (all.nuxt) problems.push('依赖了 nuxt（桌面端不得依赖 Nuxt）')
  for (const [name, ver] of Object.entries(all)) {
    if (/^(file|link|portal|workspace):/.test(ver)) {
      problems.push(`${name} 使用了本地路径协议 "${ver}"（会与 Web 端耦合）`)
    }
  }
  if (problems.length) {
    bad('C. 依赖隔离失败')
    problems.forEach((p) => fail.push('    ' + p))
  } else {
    ok(`C. 依赖隔离：${Object.keys(all).length} 个依赖，无 nuxt / 无本地路径协议`)
  }

  try {
    const lock = await readFile(join(ROOT, 'package-lock.json'), 'utf8')
    if (lock.includes('packages/frontend') || lock.includes('../frontend')) {
      bad('E. 锁文件里出现了指向 Web 端工程的条目')
    } else {
      ok(`E. 锁文件独立存在（${(lock.length / 1024).toFixed(0)} KB）`)
    }
  } catch {
    bad('E. 缺少本工程自己的 package-lock.json')
  }
}

// ---------------------------------------------------------------- D
async function checkDist() {
  const index = join(DIST, 'index.html')
  try {
    const html = await readFile(index, 'utf8')
    if (/_nuxt\//.test(html)) {
      bad('D. dist-frontend 仍是 Nuxt 产物（含 /_nuxt/），会被误当桌面端构建结果')
    } else if (!/\.\/assets\//.test(html)) {
      bad('D. dist-frontend/index.html 不含 ./assets/ 相对引用，可能不是本工程产出')
    } else {
      const s = await stat(index)
      ok(`D. 产物归属：dist-frontend 是 Vite 产物（index.html ${s.size} B）`)
    }
  } catch {
    bad('D. 未找到 dist-frontend/index.html（先跑 npm run build）')
  }
}

// ---------------------------------------------------------------- H
// 模板里的自动导入（navigateTo 等）只有开了 vueTemplate 才会注入；
// 关掉它的症状是「点了没反应」+ 控制台 `_ctx.xxx is not a function`，很难一眼看出来。
async function checkAutoImport() {
  try {
    const text = await readFile(join(ROOT, 'vite.config.ts'), 'utf8')
    if (!/vueTemplate\s*:\s*true/.test(text)) {
      bad('H. vite.config.ts 缺 vueTemplate: true —— 模板内的自动导入不注入，点击会静默失效')
    } else {
      ok('H. 模板自动导入已开启（vueTemplate: true）')
    }
  } catch {
    bad('H. 读不到 vite.config.ts')
  }
}

// ---------------------------------------------------------------- main
await checkDeps()
if (!DEPS_ONLY) {
  await checkAutoImport()
  await checkSource()
  if (CHECK_DIST) await checkDist()
}

console.log('')
for (const p of pass) console.log('  OK   ' + p)
for (const f of fail) console.log('  FAIL ' + f)
console.log('')
console.log(fail.length ? `隔离守卫：${fail.length} 项失败` : '隔离守卫：全部通过')
process.exit(fail.length ? 1 : 0)
