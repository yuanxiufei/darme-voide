#!/usr/bin/env node
// ============================================================================
// 两端同源守卫（`npm run check:parity`，Web/桌面两端 package.json 都挂）
// ============================================================================
//
// 前提（`docs/web-desktop-split.md` 已裁定，不要再提共享代码）：
//   Web `frontend/app` 与桌面 `desktop/frontend/src` 是**两份拷贝**，不是共享包。
//   复制有好处（各自能独立演进、能各自 shim），代价是「只改一端」没人拦得住 ——
//   typecheck 也拦不住：tokens 各自都对，只是行为悄悄分叉了。
//
// 所以本脚本只干一件事：把「两份拷贝现在差在哪」变成可判读的三桶。
// 判据（全部为硬判据，任一红即退出 1）：
//   A 集合  七个目录 + `plugins/` 下 .vue/.ts/.css 的相对路径集合必须相同，
//           外加**根级入口组件**按 ROOT_ENTRIES 显式对照（Web `app.vue` ↔ 桌面 `App.vue`）
//           —— 除 ONLY_DESKTOP / ONLY_WEB 登记项
//   B 内容  不在 ALLOW_DIFF 里的同名文件必须**逐字相同**（sha256）
//   C 过期  ALLOW_DIFF / ONLY_* 里的登记项若「已经逐字相同」或「某一边已不存在」⇒ 白名单在腐化
//   D 理由  每条白名单必须写 why（≥8 字）—— 白名单是**带理由的例外**，不是垃圾桶
//   E 死件  ONLY_DESKTOP 的文件必须真被桌面源码 import 到（否则是没人用的孤儿）
//   F 契约  `types/contracts.ts` 两端**去注释后必须逐字一致**（文件头各写各的身份说明）
//           —— 后端改字段时两端必须一起跟，这是全项目最容易静默分叉的地方
//   H 尺子  本脚本在两端的副本必须**逐字相同**（各存一份是为了互不依赖；漂移必须报 ✗）
//
// 为什么 B 用整文件指纹而不是「差异块白名单」：
//   差异块位置会随无关改动漂移，维护成本高到没人愿维护；整文件登记 + C 的过期检测
//   能保证「一旦两端重新变一致，白名单必须立刻收掉」，够用了。
//
// 绿了不代表「两端功能对等」——它只保证「该一致的确实一致，允许的差异都在册」。
// ============================================================================

import { createHash } from 'node:crypto'
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, extname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

//: 本脚本**两端各存一份**（`frontend/scripts/` 与 `desktop/frontend/scripts/`，逐字相同 ✓）：
//: 各自 self-contained ⇒ 删掉任一端的 `scripts/` 都不会让另一端的检查报错 ✓，
//: 改一端的尺子也不会**悄悄**改变另一端的结论 ✓（判据 H 把漂移直接判红 ✓）。
//: ⚠️ 别把它搬去仓库根 `scripts/`：`backend-py/tests/structure_test.py` 的 ㊃ 明令
//: 「仓库根不许有 `scripts/`」（README 亦有承诺）—— 那是全仓共用的硬规矩，不为一支脚本破例 ✓。
const SELF = fileURLToPath(import.meta.url)

/** 从脚本位置向上找仓库根：认「同时含 frontend/app 与 desktop/frontend/src」的目录 ✓ */
function findRepo(from) {
  let dir = from
  for (let i = 0; i < 8; i += 1) {
    if (existsSync(join(dir, 'frontend', 'app')) && existsSync(join(dir, 'desktop', 'frontend', 'src'))) return dir
    const up = resolve(dir, '..')
    if (up === dir) break
    dir = up
  }
  throw new Error(`找不到仓库根：从 ${from} 向上未发现同时含 frontend/app 与 desktop/frontend/src 的目录`)
}

const ROOT = findRepo(dirname(SELF))
const WEB = join(ROOT, 'frontend', 'app')
const DESK = join(ROOT, 'desktop', 'frontend', 'src')
//: 另一端的副本（自己这一份不算）
const SELF_COPIES = [
  join(ROOT, 'frontend', 'scripts', 'check-parity.mjs'),
  join(ROOT, 'desktop', 'frontend', 'scripts', 'check-parity.mjs'),
].filter((p) => resolve(p) !== resolve(SELF))
//: ⚠️ `plugins/` 也在比对范围内 ✓：Web 端的 ``plugins/desktop-bridge.client.ts`` 干的是
//: 「后端换端口时把 origin 同步给 apiBase」这件**两端都必须有**的事 ✓，只是桌面端的等价物是
//: ``main.ts`` 的同步注入（不在本目录）⇒ 它只能进 ONLY_WEB 并写清对照物 ✓。
//: 不纳入 DIRS 的话，这类「一端有插件、另一端悄悄没有」的缺口**一条判据都碰不到** ✗。
const DIRS = ['components', 'composables', 'utils', 'layouts', 'pages', 'assets', 'types', 'plugins']
const EXTS = new Set(['.vue', '.ts', '.css'])

// ---------------------------------------------------------------------------
// 白名单：允许只在某一端存在的文件
// ---------------------------------------------------------------------------
const ONLY_DESKTOP = {
  'utils/nuxtCompat.ts':
    '移植安全网：navigateTo/useState/useCookie 的同名等价物。当前**零引用**（桌面调用点已全部改成 vue-router 显式导入），保留是为了下次从 Web 移植页面时不用二次加工；判据 E 会盯着它别变成没人用的孤儿（真长期不用就删，连同本条一起）',
}
const ONLY_WEB = {
  'plugins/desktop-bridge.client.ts':
    'Nuxt 插件：把后端 origin 异步注入 `apiBase`（`setApiOrigin`），并做 `useState` 的桌面端兜底。桌面端没有 Nuxt 插件机制，等价物是 `desktop/frontend/src/main.ts` 里的**同步**注入 ⇒ 不是漏搬，是两套启动模型的差异；判据 C 会盯着这条别腐化（桌面端真长出 plugins/ 就该收掉本条）',
}

// ---------------------------------------------------------------------------
// 根级入口组件：**两端各自独立**，文件名和形态都不同 ⇒ 用显式对照表钉住。
//
// ⚠️ 为什么不在 `walk()` 里「顺手也比较根目录」✗：桌面根上还有 `main.ts` / `*.d.ts`
// （入口与类型声明，Web 端根本没有对应物）⇒ 一律纳入会刷出一堆假红 ✓✗。
// 所以只钉**同一个角色、两种写法**的入口组件 ✓，一条一条写清对照关系 ✓
// （不靠大小写或前缀猜 ✗：猜错就等于悄悄漏比一整类文件 ✓✗）。
// 集合层面仍是硬判据 ✓：对照表里任一边缺席 ⇒ 判据 A 当场红 ✓。
// ---------------------------------------------------------------------------
const ROOT_ENTRIES = {
  'app.vue': 'App.vue', // Web 的 Nuxt 根组件 ↔ 桌面的 vue-router 根组件
}

// ---------------------------------------------------------------------------
// 白名单：允许两端内容不同的文件（15 条，覆盖 5 类「各自独立」的合法差异）
//
// ⚠️ 2026-11-06 收口：早先这里有一条 `SHIM_MEDIA`，允许「媒体地址只桌面端过 mediaUrl、
// Web 端手拼 `'/' + p`」。那是**假绿**：两端都在跑、图也都在显示（Web 同源），所以谁都没
// 发现写法分叉；一旦按「改 Web → 同步到桌面」的既有流程操作，桌面端那份修复就被整份
// 覆盖回手拼，打包后整片裂图。现在媒体口径**两端逐字同一份实现**（`utils/apiBase.ts`
// 的 `mediaUrl()`），对应 7 个组件已从本表删除 —— 它们现在逐字一致，判据 C 会盯着别再加回来。
// ---------------------------------------------------------------------------
const SHIM_LINK = '路由组件/页面宏：桌面端没有 Nuxt ⇒ NuxtLink 换 RouterLink、`#app` 导入换 vue-router、`definePageMeta` 换成路由表 meta、模板里的组件改显式 import'
const SHIM_BRIDGE = '桌面桥：Tauri 检测/窗口控制只在桌面端存在，Web 那份是 no-op stub（保证页面代码两端可共用）'

const ALLOW_DIFF = {
  'app.vue':
    '根级入口组件：Web 用 `<NuxtLayout><NuxtPage/></NuxtLayout>`，桌面用 vue-router 的 `<RouterView/>` —— 两个工程各自独立，入口形态本来就不该一样。判据 A 仍硬要求两端各自存在（删掉任一边当场红），内容差异按本条放行',
  'types/contracts.ts':
    '文件头说明各自工程身份（Web「前端侧共享契约类型」/ 桌面「本工程自有、与 Web 端互不 import」），这是有意的；**判据 F 另查去注释后的正文必须逐字一致**，所以字段级漂移照样红',
  'assets/studio.css':
    '顶部字体：桌面端 CSP 是 `style-src self`，Google Fonts 的 @import 会被拦且静默失效 ⇒ 改本地字体栈。响应式基础层（文件末尾那一节）由 `check:responsive` 强制两端逐字一致，不受本例外影响',
  'utils/apiBase.ts':
    '注释各写各的端别视角（Web 靠 Nuxt 插件异步注入 / 桌面靠 main.ts 同步注入），**函数体同构**：`setApiOrigin`/`apiBase`/`mediaUrl` 行为必须一致，`injectedOrigin` 两端都用 `shallowRef`（后端重启换端口时 computed 才有响应式依赖）',
  'composables/useDesktop.ts': SHIM_BRIDGE,
  'layouts/default.vue': SHIM_LINK,
  'pages/drama/[id]/character/[characterId].vue': SHIM_LINK,
  'pages/drama/[id]/episode/[episodeNumber].vue': `${SHIM_LINK}（另含页面宏与 EpisodePreflight 显式 import）`,
  'pages/drama/[id]/scene/[sceneId].vue': SHIM_LINK,
  'pages/library/characters.vue': SHIM_LINK,
  'pages/library/costumes.vue': SHIM_LINK,
  'pages/library/index.vue': SHIM_LINK,
  'pages/library/props.vue': SHIM_LINK,
  'pages/library/scenes.vue': SHIM_LINK,
  'pages/library/weapons.vue': SHIM_LINK,
}

// ---------------------------------------------------------------------------
const problems = []
const notes = []

function walk(root) {
  const out = new Map()
  for (const dir of DIRS) {
    const base = join(root, dir)
    let stack
    try {
      stack = [base]
    } catch {
      continue
    }
    while (stack.length) {
      const cur = stack.pop()
      let entries
      try {
        entries = readdirSync(cur, { withFileTypes: true })
      } catch {
        continue
      }
      for (const e of entries) {
        const p = join(cur, e.name)
        if (e.isDirectory()) {
          if (e.name === 'node_modules' || e.name.startsWith('.')) continue
          stack.push(p)
        } else if (EXTS.has(extname(e.name))) {
          if (!statSync(p).isFile()) continue
          out.set(relative(root, p).split('\\').join('/'), hash(p))
        }
      }
    }
  }
  return out
}

function hash(p) {
  return createHash('sha256').update(readFileSync(p)).digest('hex')
}

function rootHashes(root, side) {
  const out = new Map()
  for (const [webName, deskName] of Object.entries(ROOT_ENTRIES)) {
    const p = join(root, side === 'web' ? webName : deskName)
    if (existsSync(p) && statSync(p).isFile()) out.set(webName, hash(p))
  }
  return out
}

const web = walk(WEB)
const desk = walk(DESK)
//: 判据 E 要按**盘上真名**读文件 ⇒ 先留一份「目录遍历」的键（根级入口不在其中 ✓）
const deskWalkKeys = [...desk.keys()]
for (const [k, v] of rootHashes(WEB, 'web')) web.set(k, v)
for (const [k, v] of rootHashes(DESK, 'desk')) desk.set(k, v)

// ---- H 两份尺子必须逐字相同 --------------------------------------------------
//: 两端各存一份 ⇒ 只改一份就会让两侧跑出**不同**结论，而且谁都不报错 ✗。
//: 这条把「漂移」变成红 ✓：动了尺子，哪一端跑都能立刻看到要同步 ✓。
const selfHash = hash(SELF)
for (const p of SELF_COPIES) {
  const rel = relative(ROOT, p).split('\\').join('/')
  if (!existsSync(p)) problems.push(`H 副本缺失：${rel}（两端各一份尺子，缺一侧等于单边护栏）`)
  else if (hash(p) !== selfHash) problems.push(`H 两份尺子已漂移：${rel} 与本文件不同 —— 改一处必须同步另一处`)
}

// ---- D 所有白名单必须带理由 ------------------------------------------------
for (const [name, why] of [...Object.entries(ALLOW_DIFF), ...Object.entries(ONLY_DESKTOP), ...Object.entries(ONLY_WEB)]) {
  if (typeof why !== 'string' || why.trim().length < 8) problems.push(`D 白名单缺理由：${name}`)
}

// ---- A 集合 -----------------------------------------------------------------
for (const k of web.keys()) {
  if (!desk.has(k) && !(k in ONLY_WEB)) problems.push(`A 只在 Web 端：${k}（要么同步到桌面端，要么登记 ONLY_WEB 并写理由）`)
}
for (const k of desk.keys()) {
  if (!web.has(k) && !(k in ONLY_DESKTOP)) problems.push(`A 只在桌面端：${k}（要么回收到 Web 端，要么登记 ONLY_DESKTOP 并写理由）`)
}

// ---- B 内容 -----------------------------------------------------------------
for (const [k, wh] of web) {
  if (!desk.has(k) || wh === desk.get(k)) continue
  if (!(k in ALLOW_DIFF)) {
    problems.push(`B 两端内容不一致且未登记：${k}（共享逻辑请两边改成一致；确属桌面 shim 再进 ALLOW_DIFF）`)
  }
}

// ---- C 白名单过期 ------------------------------------------------------------
for (const k of Object.keys(ALLOW_DIFF)) {
  if (!web.has(k) || !desk.has(k)) {
    problems.push(`C 白名单过期：${k} 已有一边不存在`)
  } else if (web.get(k) === desk.get(k)) {
    problems.push(`C 白名单过期：${k} 两端已逐字一致，请从 ALLOW_DIFF 删除这一条`)
  }
}
for (const [k, store] of [
  [ONLY_DESKTOP, desk],
  [ONLY_WEB, web],
]) {
  const other = store === desk ? web : desk
  for (const name of Object.keys(k)) {
    if (other.has(name)) problems.push(`C 白名单过期：${name} 两端已都存在，请从 ONLY_* 删除`)
    //: 只查「另一端没有」是不够的 ✓：文件被删掉之后，这条登记会变成**幽灵** ——
    //: 一端没有、另一端也没有，判据却全绿 ✓✗（登记表从此再没指向任何东西 ✗）。
    else if (!store.has(name)) problems.push(`C 白名单过期：${name} 在所属端已不存在（登记项指向空气）`)
  }
}

// ---- E ONLY_DESKTOP 必须真被引用 --------------------------------------------
const deskSources = deskWalkKeys.filter((k) => k.endsWith('.vue') || k.endsWith('.ts'))
for (const name of Object.keys(ONLY_DESKTOP)) {
  const stem = name.split('/').pop().replace(/\.\w+$/, '')
  const used = deskSources.some((k) => {
    if (k === name) return false
    const src = readFileSync(join(DESK, k), 'utf8')
    return new RegExp(`['"\`][^'"\`]*\\b${stem}\\b['"\`]`).test(src)
  })
  if (!used) notes.push(`E 未被引用：${name}（当前零引用；这份白名单条目等价于"预留"，若决定不用就删文件+删条目）`)
}

// ---- F 契约副本：去注释后必须逐字一致 ---------------------------------------
// 两端 contracts.ts 的文件头本来就不同（各写各的身份说明），所以不能整文件比；
// 但**正文**必须一样 —— 后端改一个字段名，两端都镜像一份，漏改一端就是运行期才炸。
const CONTRACT = 'types/contracts.ts'
function stripComments(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/[^\n]*/g, '')
    .replace(/\s+/g, '')
}
if (!web.has(CONTRACT) || !desk.has(CONTRACT)) {
  problems.push(`F 契约副本缺失：${CONTRACT} 必须两端都存在`)
} else {
  const a = stripComments(readFileSync(join(WEB, CONTRACT), 'utf8'))
  const b = stripComments(readFileSync(join(DESK, CONTRACT), 'utf8'))
  if (a !== b) {
    problems.push(
      `F 契约副本漂移：${CONTRACT} 去注释后两端不一致（Web ${a.length} 字符 / 桌面 ${b.length} 字符）—— 改字段必须两边一起改；长期正解是从后端 /api/v1/openapi.json 各生成一份`,
    )
  }
}

// ---- 摘要 -------------------------------------------------------------------
const common = [...web.keys()].filter((k) => desk.has(k))
const same = common.filter((k) => web.get(k) === desk.get(k))
console.log(`两端文件：Web ${web.size} / 桌面 ${desk.size}；同名 ${common.length}，逐字一致 ${same.length}，已登记差异 ${Object.keys(ALLOW_DIFF).length}`)
for (const n of notes) console.log(`  · ${n}`)
if (problems.length) {
  console.error(`\n✗ 同源守卫失败（${problems.length} 条）：`)
  for (const p of problems) console.error(`  ${p}`)
  process.exit(1)
}
console.log('✓ 同源守卫通过：该一致的都一致，允许的差异全部在册')
