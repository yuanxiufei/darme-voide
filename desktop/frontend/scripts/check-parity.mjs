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
//   A 集合  `DIRS` 九个目录下 `EXTS`（.vue/.ts/.css/.svg）的相对路径集合必须相同，
//           外加**根级入口组件**按 ROOT_ENTRIES 显式对照（Web `app.vue` ↔ 桌面 `App.vue`）
//           —— 除 ONLY_DESKTOP / ONLY_WEB 登记项
//   I 目录  `DIRS` 必须**恰好等于**两端顶层目录的并集：多了 ⇒ 漏登记（该目录整片不受
//           A/B/G 约束）；少了 ⇒ 表过期（目录已删/改名却还留着）。A 只在 DIRS 里走，
//           没有这一条，新建的目录（如 `stores/`）会静默逃出全部判据 ✗
//           （2026-11-08 加：`router/` 此前正是这样漏在表外的）
//   J 扩展名 与 I 同一个病根的第二层：目录登记了，里面**没登记的扩展名**照样整类隐身——
//           遍历只收 `EXTS`，`assets/brand-logo.svg` 两端各一份却谁也不比 ✗
//           （2026-11-08 加：`.svg` 进 EXTS，并把这层漏洞本身做成判据）
//   B 内容  不在 ALLOW_DIFF 里的同名文件必须**逐字相同**（sha256）
//   C 过期  ALLOW_DIFF / ONLY_* 里的登记项若「已经逐字相同」或「某一边已不存在」⇒ 白名单在腐化
//   D 理由  每条白名单必须写 why（≥8 字）—— 白名单是**带理由的例外**，不是垃圾桶
//   E 死件  ONLY_DESKTOP 的文件必须真被用到 —— import 其路径**或**调用其导出函数
//           （后者是桌面端自动导入的常态：零 import，路径字面根本不出现）
//   F 契约  `types/contracts.ts` 两端**去注释后必须逐字一致**（文件头各写各的身份说明）
//           —— 后端改字段时两端必须一起跟，这是全项目最容易静默分叉的地方
//   G 有界  进了 ALLOW_DIFF 只说明「可以不同」，还必须**恰好差这么多**：
//           剔注释后的代码行多重集差异必须等于 ALLOW_DIFF_BUDGET（登记值＝当前实测）
//           —— 没有这一条，「漏跟一半」「整块被覆盖」「只删一侧 shim」都会全绿
//   H 尺子  本脚本在两端的副本必须**逐字相同**（各存一份是为了互不依赖；漂移必须报 ✗）
//           ⚠️ 只守「自己」这一份，别在这里再列上别的成对脚本 —— `check-responsive.mjs`
//           的判据 D 是同一角色、守的是它自己那两支。每支脚本守自己，清单就不会漂
//           （2026-11-07 我在这条上多列了一次 responsive，等于同一件事两道尺子，已回退 ✗）。
//
// 为什么 B 用整文件指纹而不是「差异块白名单」：
//   差异块位置会随无关改动漂移，维护成本高到没人愿维护；整文件登记 + C 的过期检测
//   能保证「一旦两端重新变一致，白名单必须立刻收掉」，够用了。
//   代价是 B 不再区分「差一行」和「差三百行」⇒ 由 **G 的差异预算**补上这一问，
//   两者合起来才是完整口径：C 管「该不该在册」、G 管「在册的能差多少」。
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
//: 2026-11-08 收口：`router/` 此前**不在表内** ⇒ 桌面端 `router/index.ts`（400+ 行、
//: 整张 vue-router 表）对 A/B/G 完全不可见，谁都能改而守卫一句不吭 ✗。现在纳入并在
//: ONLY_DESKTOP 登记（Web 端由 Nuxt 按文件系统生成路由，没有可比对的文件）。
//: 表本身也不能再靠人工维护 —— 判据 I 把「两端顶层目录并集 ≠ DIRS」变成红 ✓。
const DIRS = ['components', 'composables', 'utils', 'layouts', 'pages', 'assets', 'types', 'plugins', 'router']
//: 受 A/B 约束的扩展名。2026-11-08 补 `.svg`：`assets/brand-logo.svg` 两端各一份、**此前两边都
//: 不在表里** ⇒ A/B 一句也不管它（换掉桌面端那份旧 logo，守卫照样全绿）。与 DIRS 同理，
//: 这张表也不能靠人工维护 —— 判据 **J** 把「DIRS 里出现了没登记的扩展名」变成红 ✓。
const EXTS = new Set(['.vue', '.ts', '.css', '.svg'])

// ---------------------------------------------------------------------------
// 白名单：允许只在某一端存在的文件
// ---------------------------------------------------------------------------
const ONLY_DESKTOP = {
  'router/index.ts':
    '桌面端路由表：Web 端由 Nuxt 按 `pages/` 文件系统自动生成路由，**没有对应文件可比**；桌面端是 vue-router 手写表（`createWebHashHistory` + 每页一个 component 懒加载 + `meta` 标题）。两端页面集合是否一致由判据 A 的**页面清单**保证（同名 `pages/*.vue` 必须两端都在），本表只放行「路由表这个容器本身」⇒ 它是**唯一**的桌面端入口，删了等于整个应用打不开（判据 E 会报孤儿）。',
  'utils/nuxtCompat.ts':
    'Nuxt 两个全局 API 的桌面端落地实现：`navigateTo`（6 个文件 12 处）+ `useState`（2 处）全是**裸调用、零 import**，靠 `vite.config.ts` 的 `AutoImport.dirs: ["src/utils"]` 注入。两端逐字同构的 `Onboarding.vue` / `layouts/default.vue` 正是靠它才能一个字不改地跑到桌面端 ⇒ **不是零引用，不能删**：删掉等于 14 处跳转与全局状态变 undefined，而 `vite build` 不报错、本机 `vue-tsc` 也被 `src/auto-imports.d.ts`（gitignore 的生成物、陈旧也照样声明）遮蔽（`check:isolation` 的 B2/B3/B4/H 守这一组）。仅提供这两件，没有 `useCookie`。',
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
// 差异预算：`ALLOW_DIFF` 只说「这两个文件**可以**不同」，**没说能差多少** —— 这是洞。
// 于是这两种情况都照样全绿，而它们恰恰是本脚本存在的理由：
//   ① 「Web 端改了、桌面端只跟了一半」（判据 B 只看 sha256，不等即放行 ✗）
//   ② 桌面端那份被整块删掉/被旧版本覆盖（两端仍「不等」，白名单照旧放行 ✗）
// 判据 G 把「可以不同」收紧成「差异行数必须**恰好**等于登记值」：
//   · 先剔注释再按**行**做多重集差（注释差异是有意的：两端各写各的端别说明）；
//   · **变大**红 ⇒ 漏跟一半 / 被旧版本覆盖；
//   · **变小**也红 ⇒ 只删掉了一侧的 shim 行（另一端还是 `NuxtLink`，运行期就是
//     undefined 组件），这种「修一半不如不修」的收窄最容易被当成清理 ✓✗；
//   · 只有「两侧同改、差异行数不变」才是常态，不需要碰这里的数字。
// 维护：新加一处 shim（`NuxtLink`→`RouterLink` 成对出现，+2）会让某条超预算 ⇒
//   **把数字调大这个动作本身就是「确认这处差异属于 shim」的签字**；别为了变绿盲放大数。
// ---------------------------------------------------------------------------
const ALLOW_DIFF_BUDGET = {
  'app.vue': 25, // 入口形态本来就不同（NuxtLayout+NuxtPage ↔ RouterView）+ 顶部身份说明
  'types/contracts.ts': 0, // 正文已由判据 F 要求去注释后逐字一致 ⇒ 行级差异必须为 0
  'assets/studio.css': 5, // 仅顶部字体：Web 是 Google Fonts，桌面换本地栈
  'utils/apiBase.ts': 0, // 媒体口径两端同一份实现（见本表上方 2026-11-06 那条）
  'composables/useDesktop.ts': 2, // Tauri 检测一行（`import.meta.client` ↔ `typeof window`）
  'layouts/default.vue': 44, // 22 处 NuxtLink ↔ RouterLink（导航壳是两端最容易分叉的地方）
  'pages/drama/[id]/character/[characterId].vue': 2, // `#app` ↔ `vue-router`
  //: 4 → 6（2026-11-08，**口径变化不是内容变化**）：这一页的内容一处没动，变的是度量方式 ——
  //: 旧 `codeLinesOf` 用正则删块注释，多行注释连换行一起删掉 ⇒ 前后代码被粘成一行、差异被吞；
  //: 换成等长遮蔽后行结构由原文决定，真实差异显形 = 仅 Web 3 行（`<NuxtLink` / `</NuxtLink>` /
  //: `definePageMeta`）对仅桌面 3 行（`<RouterLink` 对 / `import EpisodePreflight`）。
  //: 两端 shim 集合未变（`git diff` 该文件为空），所以是「数原来就少算了 2」，不是新漂移。
  'pages/drama/[id]/episode/[episodeNumber].vue': 6,
  'pages/drama/[id]/scene/[sceneId].vue': 2, // `#app` ↔ `vue-router`
  'pages/library/characters.vue': 2, // 面包屑 1 处 Link
  'pages/library/costumes.vue': 2, // 面包屑 1 处 Link
  'pages/library/index.vue': 20, // 10 张入口卡（每张一对 Link）
  'pages/library/props.vue': 2,
  'pages/library/scenes.vue': 2,
  'pages/library/weapons.vue': 2,
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

// ---- I 目录表必须覆盖两端顶层目录的并集 ---------------------------------------
//: A 判据只在 `DIRS` 列出的目录里走 ⇒ **新建一个目录而没登记**，那个目录下的
//: 全部文件就整片脱离 A/B/G（白名单式遍历的固有弱点，且不会有任何提示 ✗）。
//: 把「并集 ≠ DIRS」直接判红，这两个方向都堵住：
//:   · 并集有而 DIRS 没有 ⇒ 漏登记（真分叉就在里面，守卫却看不见）；
//:   · DIRS 有而两端都没有 ⇒ 表过期（目录改名/删除后残留，会让人以为还在比）。
//: 只算「递归下有受检文件（`EXTS`）」的目录 —— 空目录不参与比对、报它只会误导
//: （删完文件留下的空壳、构建残留的空目录都会踩中）。有文件的才真会影响守卫视野。
const hasTracked = (dir) => {
  const stack = [dir]
  while (stack.length) {
    const cur = stack.pop()
    let entries
    try {
      entries = readdirSync(cur, { withFileTypes: true })
    } catch {
      continue
    }
    for (const e of entries) {
      if (e.isDirectory()) {
        if (e.name !== 'node_modules' && !e.name.startsWith('.')) stack.push(join(cur, e.name))
      } else if (EXTS.has(extname(e.name))) {
        return true
      }
    }
  }
  return false
}
const topDirs = (root) => {
  try {
    return readdirSync(root, { withFileTypes: true })
      .filter((e) => e.isDirectory() && !e.name.startsWith('.') && hasTracked(join(root, e.name)))
      .map((e) => e.name)
  } catch {
    return []
  }
}
const unionDirs = new Set([...topDirs(WEB), ...topDirs(DESK)])
for (const d of unionDirs) {
  if (!DIRS.includes(d)) {
    problems.push(`I 目录未纳入比对：${d}/（两端顶层目录并集里有它，但 DIRS 没列 ⇒ 该目录下所有文件都不受 A/B/G 约束）`)
  }
}
for (const d of DIRS) {
  if (!unionDirs.has(d)) problems.push(`I 目录表过期：DIRS 列了 ${d}/，但两端都没有这个目录`)
}

// ---- J 扩展名表必须覆盖 DIRS 里出现的全部文件类型 -----------------------------
//: 与 I 同一个病根的第二层：`walk()` 只收 `EXTS` 里的类型 ⇒ 目录进了表，里面**别的扩展名**
//: 照样整类隐身（2026-11-08 实测 `assets/brand-logo.svg` 两端各一份，却谁也不比）。
//: 这里报出「哪一端、哪种扩展名、例哪份文件」，人再决定是「两端同源件」（加进 EXTS）
//: 还是「本就各自独立」（那就不该放在 DIRS 里，或写明它不是同源件）。
const strayExts = new Map()
for (const [root, side] of [[WEB, 'Web'], [DESK, '桌面']]) {
  for (const dir of DIRS) {
    const stack = [join(root, dir)]
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
        } else if (!EXTS.has(extname(e.name))) {
          const key = `${extname(e.name) || '(无扩展名)'}|${side}`
          if (!strayExts.has(key)) strayExts.set(key, relative(ROOT, p).split('\\').join('/'))
        }
      }
    }
  }
}
for (const [key, sample] of strayExts) {
  const [ext, side] = key.split('|')
  problems.push(
    `J 扩展名未纳入比对：${ext}（${side}端，例 ${sample}）—— 它在 DIRS 里却不在 EXTS ⇒ 这类文件不受 A/B 约束；` +
      `确是两端同源件就把 ${ext} 加进 EXTS，否则别放在 DIRS 里`,
  )
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
//: 「被用到」有两种成立方式，只做①会**误报零引用**（2026-11-07 真踩到）：
//:   ① 有文件 import 它的**路径**（`from '.../nuxtCompat'`）—— 常规依赖；
//:   ② 它的**导出函数**在别处被裸调用 —— 桌面端靠 vite 的 `AutoImport.dirs`
//:      注入（`navigateTo()` 在 6 个文件里 12 处、`useState()` 2 处，**全仓零
//:      import**），路径字面根本不会出现。只看①会把 14 处活调用判成孤儿，
//:      而这条 note 写着「真长期不用就删」—— 照着做，桌面端所有路由跳转与
//:      Onboarding 状态立刻变 undefined，且构建不报错 ✗✗。
const deskSources = deskWalkKeys.filter((k) => k.endsWith('.vue') || k.endsWith('.ts'))
//: 同样**先遮蔽注释**：注释里留着的 `navigateTo(` 不算「在用」——
//: 否则把唯一的调用点注释掉，E 仍说「被引用」，而运行期就是 undefined（假绿 ✗）。
const deskTexts = deskSources.map((k) => [k, maskComments(readFileSync(join(DESK, k), 'utf8'), true)])
for (const name of Object.keys(ONLY_DESKTOP)) {
  const stem = name.split('/').pop().replace(/\.\w+$/, '')
  const own = maskComments(readFileSync(join(DESK, name), 'utf8'), true)
  //: 只认「导出函数」不认 const：常量被裸调用无从判读，写成函数就自然进得来 ✓
  const symbols = [...own.matchAll(/export\s+(?:async\s+)?function\s+([A-Za-z_$][\w$]*)/g)].map((m) => m[1])
  const others = deskTexts.filter(([k]) => k !== name)
  const byPath = others.some(([, src]) => new RegExp(`['"\`][^'"\`]*\\b${stem}\\b['"\`]`).test(src))
  const bySymbol = symbols.some((s) =>
    others.some(([, src]) => new RegExp(`(^|[^\\w.$])${s}\\s*\\(`, 'm').test(src)),
  )
  if (!byPath && !bySymbol) {
    notes.push(
      `E 未被引用：${name}（既没人 import 其路径，也没有任何文件调用它的导出 ` +
        `[${symbols.join(', ') || '无导出函数'}] ⇒ 真孤儿，可删文件 + 删条目）`,
    )
  }
}

// ---- F 契约副本：去注释后必须逐字一致 ---------------------------------------
// 两端 contracts.ts 的文件头本来就不同（各写各的身份说明），所以不能整文件比；
// 但**正文**必须一样 —— 后端改一个字段名，两端都镜像一份，漏改一端就是运行期才炸。
const CONTRACT = 'types/contracts.ts'

/**
 * 把注释**遮蔽成等长空格**（HTML 注释 / 块注释 / `//`），并且**认字符串**。
 *
 * 为什么不删：删字会让后面的行左移，F 的字符数与 G 的「第几行」都会错位。
 * 为什么必须认引号：正则看不见引号，字面量里的注释起始符会把后面的真代码一起吃掉
 * （`const GLOB = 'src/**` + `/*.ts'` 这种通配符，或 `content: "/*"`），
 * 一路遮蔽到文件尾 ⇒ 那段内容对判据**完全隐身**（假绿，且没有任何提示 ✗）。
 * `inlineSlash=true` 时行内 `//` 也算注释（F/G 要比「去注释后的正文」）；
 * 引号内的 `'https://…'`、`'//static'` 永远不算 ✓。
 */
function maskComments(code, inlineSlash = false) {
  const blank = (s) => s.replace(/[^\n]/g, ' ')
  let out = ''
  let i = 0
  let quote = null
  let home = true
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
        home = false
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
    if (c === '/' && n === '/' && (inlineSlash || home)) {
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

function stripComments(src) {
  return maskComments(src, true).replace(/\s+/g, '')
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

// ---- G 白名单文件的差异必须有界 ----------------------------------------------
//: 行级剔注释：**不能复用上面的 `stripComments`** —— 它把 `\s+` 压成空串、连换行
//: 一起吃掉（那是给判据 F 做字符级逐字比用的），拿来分行就只剩一行，判不出「差几行」。
//: 改用 `maskComments`（等长遮蔽、认引号）再分行：注释行变空白行被过滤、行尾注释被 trim 掉，
//: 而「行号/行结构」由原文决定 ⇒ 差几行仍数得准 ✓。
function codeLinesOf(p) {
  //: 与 F 共用 `maskComments`（同一个「什么算注释」的口径），差别只有两点：
  //: ① 这里要**行结构** ⇒ 遮蔽成空格而不是压成空串；② 行内 `//` 也剔（注释不该算进代码行差）。
  return maskComments(readFileSync(p, 'utf8'), true)
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean)
}
//: 多重集差（不排顺序）：同一行出现两次算两次，所以「重复行被删掉一处」也数得出来 ✓
function diffTotal(a, b) {
  const rest = new Map()
  for (const l of a) rest.set(l, (rest.get(l) ?? 0) + 1)
  let deskOnly = 0
  for (const l of b) {
    const n = rest.get(l) ?? 0
    if (n > 0) rest.set(l, n - 1)
    else deskOnly += 1
  }
  let webOnly = 0
  for (const [, n] of rest) webOnly += n
  return { webOnly, deskOnly, total: webOnly + deskOnly }
}
for (const name of Object.keys(ALLOW_DIFF_BUDGET)) {
  if (!(name in ALLOW_DIFF)) problems.push(`G 预算过期：${name} 不在 ALLOW_DIFF 里（白名单已删条目，预算也得删）`)
}
let widest = { name: '（无）', total: 0, budget: 0 }
for (const name of Object.keys(ALLOW_DIFF)) {
  const budget = ALLOW_DIFF_BUDGET[name]
  if (typeof budget !== 'number') {
    problems.push(
      `G 未登记差异预算：${name}（在 ALLOW_DIFF_BUDGET 写明当前差异行数 —— ` +
        `只登记「可以不同」而不登记「能差多少」，等于这条白名单没有上限 ✓✗）`,
    )
    continue
  }
  if (!web.has(name) || !desk.has(name)) continue //: 缺边由判据 C 报，别重复刷同一条
  const d = diffTotal(codeLinesOf(join(WEB, name)), codeLinesOf(join(DESK, name)))
  const detail = `现在 ${d.total} 行（仅 Web ${d.webOnly} / 仅桌面 ${d.deskOnly}）`
  if (d.total > budget) {
    problems.push(
      `G 差异变大：${name} ${detail} > 登记 ${budget} 行 —— 一端改了另一端没跟，` +
        `或被旧版本整块覆盖；确认是新增 shim 再调大登记值`,
    )
  } else if (d.total < budget) {
    problems.push(
      `G 差异变小：${name} ${detail} < 登记 ${budget} 行 —— 多半是**只删掉了一侧**的 shim 行` +
        `（另一端还是 NuxtLink，运行期就是 undefined 组件）；确认是有意收窄再调小登记值`,
    )
  }
  if (d.total > widest.total) widest = { name, total: d.total, budget }
}

// ---- 摘要 -------------------------------------------------------------------
const common = [...web.keys()].filter((k) => desk.has(k))
const same = common.filter((k) => web.get(k) === desk.get(k))
console.log(
  `两端文件：Web ${web.size} / 桌面 ${desk.size}；同名 ${common.length}，逐字一致 ${same.length}，` +
    `已登记差异 ${Object.keys(ALLOW_DIFF).length}（差异行数最紧：${widest.name} ${widest.total}/${widest.budget}）`,
)
for (const n of notes) console.log(`  · ${n}`)
if (problems.length) {
  console.error(`\n✗ 同源守卫失败（${problems.length} 条）：`)
  for (const p of problems) console.error(`  ${p}`)
  process.exit(1)
}
console.log('✓ 同源守卫通过：该一致的都一致，允许的差异全部在册')
