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
 * ## 十一道判据（全部机械可判，不依赖人读）
 *
 * A. 源码越界：`src/**` 里的相对/别名 import 解析后必须落在 `src/` 内
 * A2. 资源越界：`.css`/`.html` 里的相对引用（url()/@import/src=/href=）不得跑出本工程
 *     —— A 只解析 JS 的 import 语法，换成 `@import '../../../frontend/app/…'` 就没有守卫
 * B. Nuxt 残留：不得出现 Nuxt 专有 API（Vite 下会直接构建失败或静默不生效）
 * B2. `useState`/`navigateTo` 例外：允许调用，但实现必须来自本工程 `utils/nuxtCompat.ts`
 * B3. 上面两个符号全仓零 import ⇒ `AutoImport.dirs` 必须覆盖 `src/utils`，
 *     否则运行期变 undefined，而 `vite build` 不报错
 * B4. `AutoImport.imports` 必须含 vue + vue-router、`dts` 必须声明 ⇒ 全仓裸标识符
 *     有来源（实测移走生成物后 typecheck 报 481 处 `Cannot find name`）
 * C. 依赖隔离：package.json 不得依赖 nuxt，不得用 file:/link:/workspace: 指向本仓其他工程
 * D. 产物归属：dist-frontend/index.html 必须是 Vite 产物（含 ./assets/ 相对引用）
 * E. 锁文件独立：本工程有自己的 package-lock.json，且不含 Web 端条目
 * F. 媒体地址：字符串拼接 / 模板串（含 `/static/…` 带目录那一族）、模板裸绑后端字段
 *    四种形态全禁，必须过 `utils/apiBase.ts::mediaUrl`
 *    （画布卡片的 `node.url` 就在这条线上）
 * G. 外链资源：CSS/HTML 里不得出现 http(s) 的外链样式、字体、脚本、图片
 *    （**含桌面壳 `desktop/frontend/index.html`** —— 打进包里的入口就是它）
 * H. 模板自动导入：`vueTemplate: true` 必须在（模板里的标识符靠它注入，
 *    缺了症状是「点击没反应」+ `_ctx.xxx is not a function`）
 *
 * 注意：判据 B / B4 / H 都会**先剥离注释**再匹配 —— 迁移说明与配置注释里
 * 会正当地提到 `NuxtLink`、`vueTemplate: true` 这些字面，不剥就会假绿。
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
/**
 * 手拼媒体的四种形态。前两条只认「斜杠紧跟引号」（`'/' + p`、`` `/${p}` ``），
 * 后两条补上「斜杠后面还带目录」那一族 —— `'/static/' + p`、`` `/static/${p}` ``、
 * `'/static/uploads/' + p` 这三种更常见的写法，旧正则**一条都匹配不到**（实测三漏），
 * 而症状与前者完全相同：dev 有 `/static` proxy 兜着看不出来，打包后整片裂图。
 * `mediaUrl()` 收的是根相对路径（`static/xxx` 或 `/static/xxx` 都行），**自己加前缀**
 * 正是它的职责 ⇒ 字符串里出现 `/static/` 又拿去拼接/插值，一定是绕过了它。
 * 注意不能只凭「出现 `'/static/…'`」判 —— `mediaUrl('/static/a.png')` 是合法调用；
 * 必须与 `+` 或 `${` 同时出现才算手拼。
 */
const MEDIA_LITERAL = [
  { re: /(['"`])\/\1\s*\+/, why: "字符串 '/' + 拼接" },
  { re: /`\/\$\{/, why: '模板串 `/${…}`' },
  { re: /['"]\/static\/[^'"\n]*['"]\s*\+/, why: "字符串 '/static/…' + 拼接" },
  { re: /`\/static\/(?:[^`\n]*)\$\{/, why: '模板串 `/static/…${…}`' },
]
/** 内联样式里的图片插值：`backgroundImage: \`url(${x})\`` —— 形如 `'/' + p` 的第二形态 */
const CSS_URL_INTERP = /url\(\s*["']?\$\{/
/**
 * 第三形态：模板绑定 `:src="x"` / `:poster="x"` 直接把后端字段塞进标签属性。
 * 前面两条判据只认字符串拼接与模板串插值，`<img :src="item.imageUrl">` 一条都盖不到
 * ——`pages/library/props.vue` 就这么漏了整整一轮（打包端道具缩略图全裂）。
 * 判据：绑定表达式里出现后端媒体字段名，且同一行没有收口函数。
 * 属性写法要认全：`:src` / `v-bind:src`、单引号 / 双引号都合法，
 * 只认 `:src="…"` 一种相当于把另一种引号写成法整个放过去。
 */
const MEDIA_BIND_ATTR = /(?::|v-bind:)(?:src|poster)\s*=\s*(["'])([^"']*)\1/
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

/**
 * 资源文件里的相对引用（A2 用）：`url(…)` / `@import …` / `src="…"` / `href="…"`。
 * 只取 `./`、`../` 开头的 —— `http(s)`、`data:`、`/static/…` 都不是相对路径，
 * 分别归 G 与 F 管，在这里天然不命中。
 */
const RELATIVE_REF =
  /(?:@import\s+(?:url\(\s*)?|url\(\s*|(?:src|href)\s*=\s*|import\s+from\s*)\s*['"]?(\.{1,2}\/[^'")\s>]+)/gi

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
 * 把注释**遮蔽成等长空格**（HTML 多行 / 块 / 整行 `//`），而不是删掉。
 * 为什么不删：删字会让后面的行整体左移 —— 报告里的 `文件:行` 全错，而判据 F 是
 * 逐行判的，行号错一位就等于指错地方。换空格 ⇒ 长度不变、换行不变、行号不变，
 * 而注释里的 `import` / `NuxtLink` / `'/static/' + p` 一律匹配不到 ✓。
 * 只遮蔽**整行** `//`：行内 `//` 后面很可能是 URL（`'https://…'`、`'//static'`），
 * 按它切会把真代码一起吃掉 ✗。
 * ⚠️ 还**必须认字符串**（2026-11-08 收口）：旧写法是三条正则直接抠注释，而正则看不见引号，
 * 字面量里的注释起始符会让遮蔽一路吃到文件尾 —— 例：`const GLOB = 'src/**` + `/*.ts'`
 * （注释起始符就藏在通配符里）之后没有收尾符 ⇒ 该文件**之后所有行对判据完全隐身**（假绿 ✗）。
 * 现在逐字符走一遍，只有「不在引号里」的注释起始符才算注释。
 * 本工程大量文件头注释在解释"这里替代了 Nuxt 的什么"，不遮蔽就会让守卫对自己写的文档误报。
 */
function maskComments(code) {
  const blank = (s) => s.replace(/[^\n]/g, ' ')
  let out = ''
  let i = 0
  let quote = null
  let home = true // 本行到当前字符为止只有空白（整行 `//` 的判定）
  while (i < code.length) {
    const c = code[i]
    const n = code[i + 1]
    if (quote) {
      out += c
      if (c === '\\') {
        out += n ?? ''
        i += 2
        continue
      }
      if (c === quote) {
        quote = null
        home = false // 引号之后就算同一行有内容了
      }
      i += 1
      continue
    }
    if (c === '\n') {
      out += c
      i += 1
      home = true
      continue
    }
    if (c === ' ' || c === '\t') {
      out += c
      i += 1
      continue
    }
    if (code.startsWith('<!--', i)) {
      const end = code.indexOf('-->', i + 4)
      const to = end < 0 ? code.length : end + 3
      const masked = blank(code.slice(i, to))
      out += masked
      home = masked.includes('\n')
      i = to
      continue
    }
    if (c === '/' && n === '*') {
      const end = code.indexOf('*/', i + 2)
      const to = end < 0 ? code.length : end + 2
      const masked = blank(code.slice(i, to))
      out += masked
      home = masked.includes('\n')
      i = to
      continue
    }
    if (c === '/' && n === '/' && home) {
      let to = i
      while (to < code.length && code[to] !== '\n') to += 1
      out += blank(code.slice(i, to))
      i = to
      continue
    }
    if (c === '"' || c === "'" || c === '`') {
      quote = c
      out += c
      i += 1
      continue
    }
    out += c
    i += 1
    home = false
  }
  return out
}

/** 抽取源码里的 import/export 来源说明符（静态 import + 动态 import + require + 副作用导入） */
function specifiersOf(code) {
  const specs = []
  const re = /(?:^|\n)\s*(?:import|export)[\s\S]*?from\s+['"]([^'"]+)['"]/g
  let m
  while ((m = re.exec(code))) specs.push(m[1])
  const re2 = /import\s*\(\s*['"]([^'"]+)['"]\s*\)/g
  while ((m = re2.exec(code))) specs.push(m[1])
  const re3 = /require\s*\(\s*['"]([^'"]+)['"]\s*\)/g
  while ((m = re3.exec(code))) specs.push(m[1])
  //: 副作用导入（`import './studio.css'`，没有 `from`、没有括号）——上面三条都要求
  //: `from` 或圆括号 ⇒ 这一种**谁也匹配不到**，越界的 CSS/JSON 能静默过关 ✗。
  //: 桌面端现在的 CSS 走 index.html，暂无实例；但 A 是「零关系」承诺的唯一机器守卫
  //: （vite.config.ts 头注释），漏一种语法就留一条缝 ⇒ 补上这 2 行便宜得多 ✓。
  const re4 = /(?:^|\n)\s*import\s+['"]([^'"]+)['"]/g
  while ((m = re4.exec(code))) specs.push(m[1])
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
/**
 * 读 `vite.config.ts` 里 `AutoImport({...})` 的四个关键开关。
 *
 * 为什么用正则读原文而不 import 配置：本脚本是纯 ESM，配置是 TS，直接 import
 * 要先过一遍转译器；判据只关心「这几行还在不在」，正则够用，而且**配置被注释掉
 * 或值被改成 false** 时同样报得出来（`dirs` 被注释掉 ⇒ 读不到 ⇒ 判红）。
 *
 * 返回 `null` = 找不到 `AutoImport(...)`（等于「自动导入整个没了」）。
 */
async function autoImportConfig() {
  let raw
  try {
    //: ⚠️ 必须先剥注释：把 `vueTemplate: true` **注释掉**之后，原文里那串字还在，
    //: 正则照样命中 ⇒ 判据绿、注入已停（H 以前正是这么假绿的）。
    raw = maskComments(await readFile(join(ROOT, 'vite.config.ts'), 'utf8'))
  } catch {
    return null
  }
  //: 参数对象里嵌着 `eslintrc: { enabled: false }`，所以不能贪心到最后一个 `}`；
  //: 非贪婪 + `}` 后必须紧跟 `)` ⇒ 会停在真正收口的那一行。
  const block = /AutoImport\s*\(\s*\{([\s\S]*?)\}\s*\)/.exec(raw)
  if (!block) return null
  const body = block[1]
  const list = (key) => {
    const m = new RegExp(`\\b${key}\\s*:\\s*\\[([^\\]]*)\\]`).exec(body)
    if (!m) return null
    return m[1]
      .split(',')
      .map((s) => s.trim().replace(/^['"`]|['"`]$/g, ''))
      .filter(Boolean)
  }
  const dts = /\bdts\s*:\s*['"`]([^'"`]+)['"`]/.exec(body)
  return {
    imports: list('imports'),
    dirs: list('dirs'),
    vueTemplate: /\bvueTemplate\s*:\s*true\b/.test(body),
    dts: dts ? dts[1] : null,
  }
}

async function checkSource() {
  const files = await walk(SRC)
  const violations = []
  const nuxtHits = []
  const shimUsers = []
  const mediaHits = []

  for (const file of files) {
    const raw = await readFile(file, 'utf8')
    //: `code` 与 `raw` **逐字符对齐**（只有注释位置是空格）⇒ 下标可以直接当行号 ✓
    const code = maskComments(raw)
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

    // B2. Nuxt 同名 API 的**裸调用**登记（实现是否合法、能否被解析，在循环外统一判）
    //: ⚠️ 关键前提：这两个符号在桌面端**全仓零 import** —— 它们靠
    //: `vite.config.ts` 的 `AutoImport.dirs: ['src/utils']` 注入。于是
    //: 「有调用」不等于「能解析」：shim 被删、或 auto-import 配置被收窄，症状
    //: 都是**构建不报错、点一下没反应**（`navigateTo is not a function`）。
    //: 所以 B2 判实现、B3 判配置，两条缺一不可。
    //: 自己那份实现（`export function navigateTo(`）不算调用点，按文件名排掉。
    //: ⚠️ `rel` 是 `relative(ROOT, file)` ⇒ 带 `src/` 前缀（写成 `utils/...` 排不掉自己，
    //: 会把 shim 自己的**定义行**算成第 7 个调用文件 ✗）。
    if (rel !== 'src/utils/nuxtCompat.ts') {
      if (/(^|[^\w.])useState\s*\(/.test(code)) shimUsers.push(rel)
      if (/(^|[^\w.])navigateTo\s*\(/.test(code)) shimUsers.push(rel)
    }

    // F. 媒体地址手拼（字符串拼接 / 模板串 / CSS 里的 url(${...}) 插值 / 裸绑后端字段）
    //: 全部改在 `code`（注释已遮蔽）上逐行判：
    //:  · 旧写法先拿整份原文跑两条正则（只报文件名、不报行号），再拿 **raw** 逐行跑
    //:    另外两条 —— 后者只跳行首 `//`/`*`/`/*`，于是块注释包住的
    //:    `<img :src="item.imageUrl">` 被当成活代码喊出来（实测假红）；
    //:  · 行号取遮蔽文本的下标，与原文逐字符对齐，不再需要「行首像不像注释」这种猜法。
    code.split('\n').forEach((line, i) => {
      const literal = MEDIA_LITERAL.find((x) => x.re.test(line))
      if (literal && !MEDIA_HELPER.test(line)) {
        mediaHits.push(`${rel}:${i + 1}（${literal.why} 未走 mediaUrl）`)
        return
      }
      if (CSS_URL_INTERP.test(line) && !MEDIA_HELPER.test(line)) {
        mediaHits.push(`${rel}:${i + 1}（内联样式 url() 插值未走 mediaUrl）`)
        return
      }
      const bind = MEDIA_BIND_ATTR.exec(line)
      if (bind && MEDIA_FIELD_NAME.test(bind[2]) && !MEDIA_HELPER.test(bind[2])) {
        mediaHits.push(`${rel}:${i + 1}（模板绑定 ${bind[2].trim()} 未走 mediaUrl）`)
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

  // B2：`useState`/`navigateTo` 是允许的（本工程提供同名 shim），但**实现必须自己拥有**
  // B3：既然调用点零 import，能解析这件事全靠 `vite.config.ts` 的 `AutoImport.dirs`
  //     —— 配置收窄时 shim 文件还在、typecheck 还是绿、构建也能过，只是运行期
  //     `navigateTo` 变成 undefined（症状是「点一下没反应」）。
  const cfg = await autoImportConfig()
  const callSites = [...new Set(shimUsers)]
  if (callSites.length) {
    let shimOk = false
    try {
      //: 与全仓其它判据一致：**先遮蔽注释再匹配** —— 否则把 `export function navigateTo(`
      //: 那一行注释掉之后，字面仍在、判据照旧绿，而运行期就是 undefined（与 H 同一个病根）。
      const shim = maskComments(await readFile(join(SRC, 'utils/nuxtCompat.ts'), 'utf8'))
      //: ⚠️ 必须带 `[<(]` 收尾：只写 `/export function navigateTo/` 的话，
      //: 把导出改名成 `navigateToRenamed` 照样命中（前缀重合）⇒ 判据绿、功能死 ✗。
      //: 反向验证：改名 ⇒ 必须红（2026-11-07 实测旧写法漏掉这一种）。
      shimOk =
        /export\s+function\s+useState\s*[(<]/.test(shim) &&
        /export\s+function\s+navigateTo\s*[(<]/.test(shim)
    } catch {
      shimOk = false
    }
    if (shimOk) {
      ok(`B2. navigateTo/useState 由本工程 shim 提供（裸调用 ${callSites.length} 个文件）`)
    } else {
      bad(
        `B2. 有 ${callSites.length} 个文件裸调 navigateTo/useState，` +
          `但 utils/nuxtCompat.ts 缺失或未同时导出这两个函数`,
      )
    }

    const dirs = cfg?.dirs
    if (dirs && dirs.some((d) => /(^|\/)utils\/?$/.test(d))) {
      ok(`B3. AutoImport.dirs 覆盖 src/utils（${callSites.length} 个文件的裸标识符才有来源）`)
    } else {
      bad(
        `B3. vite.config.ts 的 AutoImport.dirs ${dirs ? `是 [${dirs.join(', ')}]` : '读不到'}，` +
          `未覆盖 src/utils ⇒ ${callSites.length} 个文件里的 navigateTo/useState 会变 undefined` +
          `（vite build 不报错、本机 typecheck 也可能被陈旧 d.ts 遮蔽，运行期点不动）`,
      )
    }
  }

  // B4. auto-import 的**架构性**开关：本工程 84 个文件只有 33 个显式 `from 'vue'`，
  //     裸 `ref(`/`reactive(`/`useRoute(` 遍布全仓 —— 所以 `imports` 不是可选优化。
  //     实测：把 `src/auto-imports.d.ts`（gitignore 的生成物）移走后 typecheck 报
  //     **481 处 `Cannot find name`**，这才是这四项各自的真实权重。
  //     判据不依赖 d.ts 在不在（它在干净克隆上本来就不存在），直接读配置原文。
  //     `dirs` 那一项与 B3 重复判是有意的：B3 只在真有 navigateTo/useState 裸调用时
  //     才启动，而 `dirs` 同时承载 composables 的裸调用（`useApi()` 等）⇒ 两条都要在。
  if (!cfg) {
    bad('B4. vite.config.ts 里找不到 AutoImport({...}) ⇒ 全仓裸标识符（ref/useState/navigateTo…）全部失效')
  } else {
    const missing = []
    if (!cfg.imports || !cfg.imports.includes('vue')) {
      missing.push(
        `imports 缺 'vue'（现为 ${cfg.imports ? `[${cfg.imports.join(', ')}]` : '未声明'}）` +
          `⇒ 裸 ref/reactive/computed 变 undefined`,
      )
    }
    if (!cfg.imports || !cfg.imports.includes('vue-router')) {
      missing.push(`imports 缺 'vue-router' ⇒ 裸 useRoute/useRouter 变 undefined`)
    }
    if (!cfg.dts) {
      missing.push(
        'dts 未声明 ⇒ 不再生成 src/auto-imports.d.ts，typecheck 会报数百处 Cannot find name' +
          '（不是静默，但工程直接不可用）',
      )
    }
    if (missing.length) {
      bad(`B4. AutoImport 配置缺 ${missing.length} 项：${missing.join('；')}`)
    } else {
      //: 只判这两项：`dirs` 归 B3（它还要覆盖 composables 的裸调用）、
      //: `vueTemplate` 归 H。同一件事不设两道尺子 —— 改一处忘另一处就是两套口径。
      ok(`B4. AutoImport 的 imports(vue+vue-router) 与 dts(${cfg.dts}) 都在（dirs 见 B3、vueTemplate 见 H）`)
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
  //: 这里原来只走 `src/`，而 `src/` 下**一份 .html 都没有**（桌面壳的入口
  //: `desktop/frontend/index.html` 在上一级）⇒ 判据名里的「HTML」那半边一直空转：
  //: 谁往壳的 `<head>` 里贴一行 `<link href="https://fonts…">`，CSP 照样拦、字体照样
  //: 静默失效，而 G 全绿。打包出来的东西正是从这个文件起的，必须扫。
  const resourceFiles = await walk(SRC, RESOURCE_EXT)
  const shellHtml = join(ROOT, 'index.html')
  try {
    await stat(shellHtml)
    resourceFiles.push(shellHtml)
  } catch {
    /* 没有壳 HTML 就不扫（不静默假装扫过，下面 ok 行报的是真实文件数） */
  }
  const externalHits = []
  // A2. 资源文件里的**相对引用**必须落在桌面工程内（`@import '../../../frontend/…'`
  //     这类跨端引用是 A 的同类问题，但 A 只解析 JS 的 import 语法 ⇒ 换到 CSS/HTML
  //     里就完全没有守卫）。只判「跑出 ROOT 之外」：`url(./x.svg)`、`<a href="../library">`
  //     都还在工程内 ⇒ 不喊；`http(s)`、`data:`、`/static/…` 不以 `./`/`../` 开头 ⇒ 天然不命中。
  const escapeRefs = []
  for (const file of resourceFiles) {
    const text = maskComments(await readFile(file, 'utf8'))
    const rel = relative(ROOT, file).replace(/\\/g, '/')
    for (const m of text.matchAll(RELATIVE_REF)) {
      const target = resolve(dirname(file), m[1])
      if (!inside(target, ROOT)) {
        escapeRefs.push(`${rel}: "${m[1]}" 解析到桌面工程外 (${relative(REPO, target)})`)
      }
    }
    if (EXTERNAL_CSS.test(text) || EXTERNAL_CSS_URL.test(text) || EXTERNAL_TAG.test(text)) {
      externalHits.push(rel)
    }
  }
  if (escapeRefs.length) {
    bad(`A2. 资源文件引用越界 ${escapeRefs.length} 处（CSS/HTML 里引到了桌面工程之外）`)
    escapeRefs.forEach((v) => fail.push('    ' + v))
  } else {
    ok(`A2. 资源引用边界：${resourceFiles.length} 个样式/模板文件，无跨工程相对引用`)
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
  //: 与 B4 共用 `autoImportConfig()`（它会先剥注释）⇒ 单一解析器。
  //: 以前的写法是 `readFile + /vueTemplate\s*:\s*true/` 直接扫全文 ⇒ 把那行**注释掉**
  //: 之后判据仍然绿（注释里留着原文），而注入其实已经停了。
  const cfg = await autoImportConfig()
  if (!cfg) {
    bad('H. 读不到 vite.config.ts 里的 AutoImport({...})')
  } else if (!cfg.vueTemplate) {
    bad('H. vite.config.ts 的 AutoImport 缺 vueTemplate: true —— 模板内的自动导入不注入，点击会静默失效')
  } else {
    ok('H. 模板自动导入已开启（vueTemplate: true）')
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
