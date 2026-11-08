# Web 端与桌面端的工程拆分

> 状态：**已落地**（桌面端独立工程可构建、可类型检查、隔离守卫全绿）
> 更新时间：2026-11-04
> 替代：`docs/desktop-tauri-selection.md` 里"复用同一套 Nuxt SPA"的决定

---

## 1. 结论先行

两端现在是**两个互不读写的工程**：

| | Web 端 | 桌面端 |
|---|---|---|
| 位置 | `frontend/` | `desktop/frontend/` |
| 构建器 | Nuxt 3（`nuxt build` / `nuxt generate`） | Vite 8（`vite build`） |
| 产物 | `frontend/.output/public` | `desktop/dist-frontend` |
| 路由 | history（`createWebHistory`） | hash（`createWebHashHistory`） |
| 源码 | 84 文件 | 86 文件（**自有副本，非引用**） |
| 锁文件 | `frontend/package-lock.json` | `desktop/frontend/package-lock.json`（60 KB，独立） |
| 依赖数 | Nuxt 全家桶 | 11 个（无 nuxt） |
| 产物体积 | 77 文件 / 1.05 MB | **66 文件 / 1.00 MB**（gzip 估算 0.30 MB） |
| 类型检查 | Nuxt `typeCheck: true` | `vue-tsc`（**0 错误**） |

**唯一共享的是后端** `backend-py/`，两端通过 HTTP 契约通信，不共享任何前端代码。

---

## 2. 目录结构

```
darme-voide/
├── backend-py/                      # 唯一后端（FastAPI），两端共用
│
├── frontend/                        # ██ Web 端（Nuxt）
│   ├── package.json
│   ├── package-lock.json
│   ├── nuxt.config.ts
│   └── app/
│       ├── app.vue                  # NuxtLayout + NuxtPage
│       ├── pages/        (19)
│       ├── components/   (34)
│       ├── composables/  (8)
│       ├── utils/        (17)
│       ├── layouts/      (2)
│       ├── types/contracts.ts
│       └── assets/
│
└── desktop/                         # ██ 桌面端（Tauri）
    ├── package.json                 # 仅 @tauri-apps/cli
    ├── scripts/
    │   ├── sync-frontend.mjs        # 触发 desktop/frontend 构建 + 校验产物归属
    │   ├── sync-backend.mjs
    │   ├── size-budget.mjs
    │   └── gen-icons.mjs
    ├── src-tauri/                   # Rust 壳
    │   ├── tauri.conf.json          # frontendDist = ../dist-frontend
    │   ├── capabilities/
    │   └── src/*.rs
    ├── dist-frontend/               # ← vite build 产物（不进版本库）
    └── frontend/                    # ★ 桌面端独立前端工程
        ├── package.json
        ├── package-lock.json        # 独立锁文件
        ├── vite.config.ts
        ├── tsconfig.json
        ├── index.html
        ├── scripts/
        │   └── check-isolation.mjs  # 隔离守卫
        └── src/
            ├── main.ts              # await initDesktopBridge() → mount
            ├── App.vue              # 动态 layout + Toaster + TaskCenter
            ├── router/index.ts      # 20 条显式路由（hash）
            ├── layouts/      (2)
            ├── pages/        (20)   # 副本，此后独立演进
            ├── components/   (34)
            ├── composables/  (8)
            ├── utils/        (17)   # 含 nuxtCompat.ts（navigateTo/useState 实现，靠 auto-import 注入）
            ├── types/contracts.ts   # 副本
            └── assets/      (3)
```

`desktop/frontend/` 与 `frontend/app/` **没有任何路径引用关系**。守卫 A 项强制这一点。

---

## 3. 独立依赖管理

### 3.1 没有 monorepo

仓库根目录**没有** `package.json`、没有 `pnpm-workspace.yaml`、没有 `turbo.json`。三个包彼此不可见：

```
frontend/          package.json + package-lock.json
desktop/           package.json（Tauri CLI）
desktop/frontend/  package.json + package-lock.json
```

这是刻意的。一旦引入 workspace，`npm install` 会把三个包提升到根 `node_modules`，版本冲突就从"各自失败"变成"全局失败"。

### 3.2 版本可以各自演进

| 包 | Web 端 | 桌面端 | 说明 |
|---|---|---|---|
| vue | 3.5.x | 3.5.x | 可不同步，各自 `npm update` |
| vue-router | 4.5.x | 4.5.x | |
| nuxt | 3.21.x | **不存在** | 桌面端不装 Nuxt |
| vite | 由 Nuxt 间接引入 | 8.3.5（直接依赖） | |
| unplugin-auto-import | **不存在** | 21.1.0 | Nuxt 用自研 `unimport` |
| unplugin-vue-components | **不存在** | 32.1.0 | Nuxt 用 `@nuxt/components` |

> 注意：桌面端**必须**自带两个 unplugin。迁移过来的 86 个文件里只有 33 个显式 `from 'vue'`，
> 其余全靠自动导入；组件也一样（34 个组件通常只有个位数显式 import）。

### 3.3 安装与升级互不影响

```bash
# Web 端
cd frontend && npm install

# 桌面端（不会读也不会写 frontend/node_modules）
cd desktop/frontend && npm install
```

---

## 4. 迁移时如何避免代码耦合

耦合不会在"复制"那一刻产生，而是复制之后有人写了一句跨工程 import。所以拆成**五层防线**：前四层拦"跨工程耦合"（源码文本 / 构建期模块集合），第五层拦"两端静默漂移"（只改一端，各自都编译得过的分叉）。

### 第一层：物理隔离（复制而非引用）

迁移是**复制文件**，不是软链、不是 workspace、不是 `file:` 协议。复制后两份代码各自拥有独立历史。

### 第二层：身份隔离（同名兼容层）

| 原 Nuxt API | 桌面端处理 | 调用点改动 |
|---|---|---|
| `<NuxtLink>` | → `<RouterLink>`（vue-router 全局组件） | 33 处标签，批量替换 |
| `navigateTo()` | `utils/nuxtCompat.ts` 提供**同名**实现（内部 `router.push`） | **0 处**（现 6 文件 12 处调用） |
| `useState()` | `utils/nuxtCompat.ts` 提供**同名**实现（模块级 Map 单例） | **0 处**（现 2 处调用） |
| `definePageMeta({layout})` | 删除，改为 `router/index.ts` 的 `meta: { layout }` | 1 处 |
| `#app` | → `vue-router` | 2 处 |
| `import.meta.client` | → `typeof window === 'undefined'` | 1 处 |
| `~/assets/*` | 别名 `~` 指向 `src/`，**无需改** | 0 处 |

同名的价值：调用点零改动，未来从 Web 端再移植代码时不需要二次加工。

⚠️ 这两个符号在桌面端**全仓零 `import`**——能解析全靠 `vite.config.ts` 的 `AutoImport`
（`dirs` 含 `src/utils`）。于是「shim 文件在不在」和「配置还盖不盖得住」是两件独立的事：
任一不成立，症状都是**点一下没反应**（`navigateTo is not a function`），而 `vite build`
不报错（它只是注入不到）。

`vue-tsc` 也兜不住：`src/auto-imports.d.ts` 是 unplugin 的生成物、被 `.gitignore` 忽略，
本机留着它（**陈旧也照样声明**）⇒ typecheck 一路绿；只有干净克隆（没跑过 dev/build）
才会报。实测把它移走后 typecheck 报 **481 处 `Cannot find name`**（`ref`/`computed`/
`navigateTo` 全在内）——这同时说明这套注入是**架构依赖**，不是可选优化。

所以守卫必须是脚本级的四条：B2（shim 实现齐全）、B3（`dirs` 覆盖 `src/utils`）、
B4（`imports` 含 vue+vue-router 且 `dts` 在）、H（`vueTemplate: true`），全部做过反向验证。

### 第三层：机器守卫（`check-isolation.mjs`）

```bash
cd desktop/frontend
npm run check:isolation        # 十一项判据（A/A2/B/B2/B3/B4/C/E/F/G/H）
npm run check:isolation -- --dist   # 额外校验产物归属（D）
```

| 判据 | 拦什么 |
|---|---|
| **A** 源码越界 | 任何相对/别名 import 解析后必须落在 `src/` 内；裸模块必须在白名单。抽四种写法：`from '…'`、`import('…')`、`require('…')`、**无 `from` 的副作用导入 `import '…'`**（最后一种此前漏抽，越界的 CSS/JSON 能静默过关） |
| **A2** 资源越界 | **非 JS 文件**（`.css` / `.html`）里的相对引用（`url(…)` / `@import` / `src=` / `href=`）解析后必须落在桌面工程内 —— A 只解析 JS 的 import 语法，换成 `@import '../../../frontend/app/assets/studio.css'` 就完全没有守卫 |
| **B** Nuxt 残留 | `NuxtLink`/`definePageMeta`/`useHead`/`#app`/`import.meta.client` 等 |
| **B2** shim 归属 | 允许 `useState(`/`navigateTo(` 裸调用，但实现必须来自本工程 `nuxtCompat.ts`（导出名带词边界判，`navigateToRenamed` 之类骗不过） |
| **B3** 注入配置 | 有裸调用 ⇒ `vite.config.ts` 的 `AutoImport.dirs` 必须覆盖 `src/utils`（否则运行期 undefined，构建不报错） |
| **B4** 注入来源 | `AutoImport.imports` 必须含 `vue` + `vue-router`，`dts` 必须声明（全仓裸 `ref(`/`useRoute(` 靠它；实测移走生成物后 typecheck 报 481 处 `Cannot find name`） |
| **C** 依赖隔离 | 不得依赖 `nuxt`；不得用 `file:`/`link:`/`workspace:`/`portal:` 指向别处 |
| **D** 产物归属 | `dist-frontend/index.html` 必须是 Vite 产物（含 `./assets/`），出现 `/_nuxt/` 即失败 |
| **E** 锁文件独立 | 本工程有自己的 `package-lock.json`，且不含 Web 端条目 |
| **F** 媒体地址 | 四种形态都禁：`'/' + p`、`` `/${p}` ``、**`'/static/' + p`**、**`` `/static/${p}` ``**（后两种是 2026-11-08 补的，旧正则只认「斜杠紧跟引号」）；标签属性不得把后端字段**裸绑**（`:src` / `v-bind:src`、单双引号皆判），一律过 `utils/apiBase.ts::mediaUrl` |
| **G** 外链资源 | CSS/HTML 里不得出现 `http(s)` 的样式表、字体、脚本、图片（CSP 会拦、断网必失效）。2026-11-08 起**扫到桌面壳 `index.html`** —— 此前只走 `src/`，而 `src/` 下一份 `.html` 都没有，HTML 那半边判据一直空转 |
| **H** 模板注入 | `vueTemplate: true` 必须在（缺了症状是"点击没反应"+`_ctx.xxx is not a function`） |

**反向验证已通过**：故意写入 `import ... from '../../../frontend/app/composables/useApi'`
和 `useHead` 引用 → 守卫 A、B 同时报错，退出码 1；删除后恢复全绿、退出码 0。
B2/B3/B4/H 四项另做过反向验证：shim 导出改名 → B2 红；`dirs` 去掉 `src/utils` → B3 红；
`imports` 去掉 `vue-router` + 删 `dts` → B4 报两项；把 `vueTemplate: true` **注释掉** → H 红
（这一条以前是假绿：旧写法扫全文，注释里留着原文也照样命中）。

A 另有补充（2026-11-07）：**副作用导入**（`import '…'`，无 `from`、无圆括号）此前三条正则
都不命中 ⇒ 越界的 CSS/JSON 能静默过关。已补第四条正则并实测：夹具
`src/_probe_isolation.ts` 写 `import '../../../frontend/app/composables/useApi'`
→ `FAIL A. 源码越界 import 1 处 … 解析到工程外 (frontend\app\composables\useApi)`，退出码 1；
删夹具后复原全绿。

> **2026-11-08 收口（F 三漏 + 一处假红；G 空转；新增 A2）**：用「同一坏法拆成五种写法各占
> 一个夹具」的办法实测，F 的两条正则只认「斜杠**紧跟**引号」⇒ 五种写法只抓两种：
>
> | 夹具 | 旧 F |
> |---|---|
> | `'/' + p` | 抓到 |
> | `` `/${p}` `` | 抓到 |
> | `'/static/' + p` | **漏** |
> | `` `/static/${p}` `` | **漏** |
> | `'/static/uploads/' + p` | **漏** |
>
> 后三种在实际代码里更常见（上传接口返回的就是 `static/uploads/…`），而症状与前两种一模一样：
> dev 有 `/static` proxy 兜着看不出来，打包后整片裂图。现在改成**四种形态一张表**
> `MEDIA_LITERAL`、逐行判。同时修两个反向毛病：
> ① **假红** —— 旧写法拿 `raw` 逐行扫，只跳行首 `//`/`*`/`/*`，块注释包住的
> `<img :src="item.imageUrl">` 会被当成活代码喊出来；改成在**注释遮蔽后**的文本上逐行判；
> ② `:src` 只认双引号简写 ⇒ `:src='…'` 与 `v-bind:src="…"` 整个漏（已实测补上，静态
> `<img src="./assets/…">` 不误报）。
>
> `stripComments()` 因此升级为 **`maskComments()`**：把注释**逐字符换成等长空格**而不是删掉。
> 删字会让后面的行整体左移，而 F 是逐行判的、行号错一位就指错地方；换空格则长度、换行、
> 行号全不变，注释里的 `import` / `NuxtLink` / `'/static/' + p` 一样匹配不到。
> 只遮蔽**整行** `//`（行内 `//` 后面常是 `'https://…'`，切了会吃掉真代码）。
>
> G 此前只走 `src/`，而 `src/` 下**一份 `.html` 都没有** ⇒ 判据名里的「HTML」半边一直空转；
> 打进包里的壳其实是 `desktop/frontend/index.html`。现在把它并进扫描：实测往 `<head>` 里加
> 一行 `fonts.googleapis.com` 的 `<link>` → G 立刻红（以前全绿）。
> 同时补 **A2**（CSS/HTML 里的相对引用不得跑出工程）：实测 `@import url('../../../frontend/app/assets/studio.css')`
> → `A2` 红，而 `@import url('./studio.css')` 不误报。

> 判据 B / B4 / H 都会**先剥离注释**再匹配——迁移说明里会正当地提到 `NuxtLink`、
> 配置注释里也会写着 `vueTemplate: true` 这些字面，不剥离就会误报或假绿。

> **2026-11-08 第二轮收口：遮蔽器必须认引号（三个脚本同一病根）**。「先剥注释再匹配」只做了一半 ——
> 遮蔽器原本是三条正则（`<!-- -->` / `/* */` / 整行 `//`），而**正则看不见引号**：
>
> | 夹具（一行） | 旧遮蔽器 | 后果 |
> |---|---|---|
> | `export const OPEN = '/**'` 之后出现真注释 `/* … */` | 从**字符串里**的 `/*` 一路遮蔽到那句注释的 `*/` | 中间所有真代码变空格 ⇒ **判据对这一段彻底失明**（假绿，且无任何提示） |
> | `content: "/*"`、`const GLOB = 'src/**` + `/*.ts'` | 找不到收尾符 ⇒ **遮蔽到文件尾** | 该文件剩下的行全部隐身 |
>
> 取证：往 `src/utils/` 放这样一个 7 行夹具，旧遮蔽器输出只剩第一行半截、`'/static/' + p` 整个消失（F 全绿）；
> 换成认引号的逐字符扫描后，F 精确指到第 6 行。现在 `maskComments()` 逐字符走一遍，
> 只有**不在引号里**的注释起始符才算注释（`'https://…'`、`'//static'`、`"/*"` 原样保留），
> 且 `//` 的取舍由参数定：`inlineSlash=false` 只吃**整行** `//`（`check-isolation` 用，源码里有正则字面量），
> `inlineSlash=true` 连行内 `//` 也吃（`check-parity` 的 F/G 要比「去注释后的正文」）。
>
> 同一轮顺着这个病根把 `check-parity` 的三处补齐（都做过**双向**反验）：
> ① **F**（契约副本逐字比）与 **G**（行级差异）改用同一份 `maskComments(src, true)`；
> ② **E**（`ONLY_DESKTOP` 必须真被引用）也先遮蔽 —— 否则把唯一的调用点注释掉，E 仍说「被引用」，
> 而运行期就是 `undefined`（夹具实测：注释里写 `_pZzz('x')` ⇒ 报孤儿；改成真调用 ⇒ 静默 ✓）；
> ③ `check-isolation` 的 **B2** 读了 `utils/nuxtCompat.ts` 却用**原文**匹配 `export function navigateTo(` ——
> 把那行注释掉，判据照旧绿。现在同样走遮蔽后才匹配。
>
> **G 的登记值因此重定了一处**：`pages/drama/[id]/episode/[episodeNumber].vue` 4 → 6。
> 这是**度量口径变了、内容没变**（该文件两端 `git diff` 为空）：旧实现删块注释时把换行一起删掉，
> 多行注释前后的两行代码被**粘成一行**，差异被吞掉 2 行；换成等长遮蔽后行结构由原文决定，
> 真实差异显形 = 仅 Web 3 行（`<NuxtLink` / `</NuxtLink>` / `definePageMeta`）对仅桌面 3 行
> （`<RouterLink` 对 / `import EpisodePreflight`）。其余 14 条登记值一字未动。

> **2026-11-08 第三轮收口：`router/` 整片游离于所有判据之外，加判据 I**。`DIRS` 原先只列八个目录，
> 而 `desktop/frontend/src/router/index.ts`（400+ 行手写 vue-router 表，**应用唯一入口**）就在第九个
> 目录里 ⇒ 它从不在 A/B/G 的遍历范围内：改坏一个 component 指向、漏跟一页路由，本脚本一声不响。
> 现状：`DIRS` 补 `router`，`router/index.ts` 登记进 `ONLY_DESKTOP`（Web 端路由由 Nuxt 按 `pages/`
> 文件系统生成，**没有对应文件可比**；两端页面集合是否一致另有 A 的 `pages/*.vue` 保证）。
>
> 只补一个目录治不了病根 —— 病根是「**白名单式遍历**：`DIRS` 之外的目录永远逃出全部判据」，
> 新增一个目录（比如将来加 `stores/`）会重演同一次失明。故加判据 **I**：两端顶层目录**并集**必须
> 恰好等于 `DIRS`（双向，两个方向都实测反验过）：
>
> | 动作 | I 的判定 |
> |---|---|
> | 新建 `frontend/app/stores/x.ts`（`DIRS` 没这项） | 红：`I 目录未纳入比对：stores/（…）` —— 强制登记后才能进判据 |
> | 在 `DIRS` 里加一项 `_p_ghost`（两端都没这目录） | 红：`I 目录表过期：DIRS 列了 _p_ghost/，但两端都没有这个目录` —— 防止表本身腐化成「看着很全」 |
>
> `topDirs()` 只看**递归下真有 `EXTS` 文件**的目录：空目录/构建残留不应制造噪声红（反验时删夹具
> 只删了文件、目录还在，I 当场误报过一次 ⇒ 加此过滤）。判据 I 落地后文件数从 87/87 变成 **87/88**。
>
> **同一轮把第二层漏洞也堵上（判据 J）**：目录进了表还不够，`walk()` 只收 `EXTS` 里的类型 ⇒
> 登记目录里**别的扩展名**照样整类隐身。实测就有一份：`assets/brand-logo.svg` 两端各存一份、
> 内容当时确实相同，可 `.svg` 不在 `EXTS` 里 ⇒ 换掉桌面端那份旧 logo，本脚本一句话都不说。
> 处置：`.svg` 进 `EXTS`（B 从此逐字比它，反验＝只改桌面端那份 ⇒ 精确报 B 内容不一致），
> 并加判据 **J**：`DIRS` 里出现未登记扩展名即红。反验＝在 `assets/` 放一个 `.json` ⇒
> 报 `J 扩展名未纳入比对：.json（Web端，例 frontend/app/assets/_probe_extra.json）`。
> 加完后文件数 **88/89**（两端各 +1，就是 `assets/brand-logo.svg`），逐字一致 71 → **72**。

### 第四层：构建期自包含断言（`vite.config.ts`）

第三层查**源码文本**，能被 `import(变量)` 这类动态写法绕过。第四层查**打包器实际读到的
模块集合**（`generateBundle` 里各 chunk 的 `modules` 键），绕不过：

> 任何被打进产物的模块，路径必须落在 `desktop/frontend/` 内。

失败时 `vite build` 退出 1，并把越界清单落到 `frontend/isolation-outsiders.txt`（已 gitignore）。

**反向验证已通过**：在 `desktop/` 下建 `_probe_external.mjs` 并让 `main.ts` 引用它
→ `vite build` 退出 1，清单精确指出 `desktop/_probe_external.mjs`；移除后恢复 0。

> 两层分工要分清：**纯类型**的跨工程 import（如 `import '../../../frontend/app/types/contracts'`）
> 会被完全 tree-shake，不出现在 `chunk.modules` 里 ⇒ 第四层不报，由第三层 A 项报。
> 第四层专拦**运行时代码**被引进来——那才会真的污染体积与行为。
>
> 踩过的坑：判据早期写成 `resolve(projectRoot)` 直接 `startsWith` 比较，而 Windows 上
> `resolve()` 给反斜杠、打包器模块 id 用正斜杠 ⇒ 本工程自己的 3 个文件被误判为"外部"，
> 构建必挂。现已统一转正斜杠后比较。

### 第五层：同源漂移守卫（`check-parity.mjs`）

前四层都在拦"两端互相引用"。但它们拦不住另一种更常见的坏法：**只改一端**——
另一端还是老逻辑，各自 `vue-tsc` 都零错、各自页面都能跑，只是行为悄悄分叉了。
这类分叉没有任何编译器能发现，只能靠比对。

```bash
npm run check:parity        # 两端各自的 scripts/check-parity.mjs（两份逐字相同）
```

> **脚本位置（2026-11-06 定稿）**：`check-parity.mjs` 与 `check-responsive.mjs` **两端各存一份**，
> 分别在 `frontend/scripts/` 与 `desktop/frontend/scripts/`，两份**逐字相同**（sha256 一致）。
>
> 为什么不是「一份、两端都调它」：原先桌面端写的是 `node ../../frontend/scripts/check-parity.mjs`
> ——那只眼睛长在 Web 端目录里，删/改名 `frontend/scripts/` 会让桌面端的检查命令直接报错，
> 改它又会悄悄改变桌面端的门。现在各跑各的副本，谁的 `scripts/` 都不进对方的调用链。
>
> 为什么不是「放仓库根 `scripts/`」：`backend-py/tests/structure_test.py` 的 **㊃** 明令
> 「仓库根不许有 `scripts/`」（README 亦有承诺），这是全仓共用硬规矩，不为一支脚本破例。
>
> 各存一份的代价是可能漂移，所以两边各加了一条判据盯着：`check-parity` 的 **H** 与
> `check-responsive` 的 **D** —— 副本缺失或与本文件哈希不同，两端跑都会红。
> 脚本按**自身位置**向上认仓库根（认「同时含 `frontend/app` 与 `desktop/frontend/src`」的目录），
> 所以放在哪一端、从哪个 `cwd` 调用，结论都一样。

比对 `components/ composables/ utils/ layouts/ pages/ assets/ types/ plugins/ router/` 九个目录
（`EXTS`：`.vue/.ts/.css/.svg`），外加**根级入口组件的显式对照**（Web `app.vue` ↔ 桌面 `App.vue`，见 `ROOT_ENTRIES`），
十条硬判据（A–J），任一红即退出 1：

| 判据 | 拦什么 |
|---|---|
| **A** 集合 | 只在某一端存在的文件（除 `ONLY_DESKTOP` / `ONLY_WEB` 登记项）。根级入口按 `ROOT_ENTRIES` 的对照键比较，任一边缺席即红（不靠文件名大小写猜） |
| **B** 内容 | 不在 `ALLOW_DIFF` 里的同名文件必须**逐字相同**（sha256） |
| **C** 过期 | 白名单腐化：已逐字一致却还登记着 / 某一边已不存在 / 登记项在**所属端**已不存在（指向空气） |
| **D** 理由 | 每条白名单必须写 `why`（白名单是**带理由的例外**，不是垃圾桶） |
| **E** 死件 | `ONLY_DESKTOP` 的文件必须真被桌面源码 import 到（只报不拦） |
| **F** 契约 | `types/contracts.ts` 两端**去注释后必须逐字一致**（文件头各写各的身份说明） |
| **G** 有界 | 进了 `ALLOW_DIFF` 只说明「可以不同」，剔注释后的**代码行多重集差异必须恰好等于** `ALLOW_DIFF_BUDGET` 登记值（变大＝漏跟一半/被旧版覆盖；变小＝只删了一侧的 shim 行） |
| **H** 尺子 | 本脚本在两端的两份副本必须**逐字相同**（副本缺失或哈希不同即红——各存一份是为了互不依赖，漂移不能静默） |
| **I** 目录表 | 两端**顶层目录并集 ≠ `DIRS`** 即红（双向：盘点出新目录却没登记 → 「整片逃出全部判据」；`DIRS` 里留着已不存在的目录 → 表腐化）。只看**递归下有 `EXTS` 文件**的目录，空目录/构建残留不参与 |
| **J** 扩展名 | 与 I 同一个病根的第二层：目录登记了，里面**没登记的扩展名**照样整类隐身。`DIRS` 里出现 `EXTS` 之外的文件类型即红（报「哪一端、哪种扩展名、例哪份文件」）—— 是两端同源件就加进 `EXTS`，不是就别放在 `DIRS` 里 |

当前状态：Web 88 / 桌面 89；同名 87，**逐字一致 72**，登记差异 15（`ALLOW_DIFF` 15 条）。
<small>桌面端多出的那一个就是 `router/index.ts`（`ONLY_DESKTOP`，理由见脚本注释：Web 端路由由 Nuxt 文件系统生成，没有对应文件）。</small>
15 条差异归 5 类合法 shim（每条在脚本里都写了 why）：`NuxtLink→RouterLink`／`#app→vue-router`／
`definePageMeta→路由表 meta`、`typeof window` 守卫（桌面桥）、`studio.css` 字体、
`contracts.ts` 的文件头，以及**根级入口组件**（`NuxtLayout/NuxtPage` vs `RouterView`）。

> **2026-11-07 第四次收口（判据 G）**：`ALLOW_DIFF` 此前只登记「这两个文件**可以**不同」，
> **没说能差多少** —— 于是「Web 端改了、桌面端只跟一半」和「桌面端那份被旧版本整块覆盖」
> 都照样全绿，而这两件正是本脚本存在的理由（B 只看 sha256，不等即放行）。
> 现在每条差异另有 `ALLOW_DIFF_BUDGET`：剔注释后的**代码行多重集差异必须恰好等于**登记值，
> 两侧合计一分余量都不留。**变大**红（漏跟/被覆盖）；**变小**也红 —— 那通常意味着
> 只删掉了一侧的 shim 行，另一端还是 `NuxtLink`，运行期就是 undefined 组件。
> 只有「两端同改、差异行数不变」是常态，不需要碰那些数字。当前最紧的是
> `layouts/default.vue` 44/44（22 处 `NuxtLink↔RouterLink`）。

> **2026-11-06 第三次收口（覆盖范围）**：此前 `plugins/` 与根级 `app.vue` **完全不在比对范围内**——
> 它们不是 `DIRS` 里的子目录，A/B/C 一条都碰不到。于是「Web 端加了插件、桌面端悄悄没有」和
> 「只在 Web 端改了根组件」这两类分叉，守卫一直是绿的。现在 `plugins/` 纳入 `DIRS`
> （`plugins/desktop-bridge.client.ts` 登记进 `ONLY_WEB`，写明桌面端等价物是 `main.ts` 的同步注入），
> 根级 `.vue` 入口按 `ROOT_ENTRIES` 显式对照。**没有把桌面的 `main.ts` / `*.d.ts` 一起纳入**：
> 那是两套启动模型的差异，纳入只会刷假红。

> **2026-11-06 第二次收口**：早先这里有一类「媒体地址走 `mediaUrl()`」的例外，意思是
> 「桌面端过 `mediaUrl`、Web 端手拼 `'/' + p`」。这是**假绿**——两端各自都能跑、图都能显示
> （Web 同源），编译器与守卫都不报，于是没人发现写法已经分叉；一旦按既有流程「改 Web →
> 同步到桌面」，桌面那份修复就被整份覆盖回手拼，**打包后整片裂图**。现在 `mediaUrl()`
> 是两端**逐字同一份实现**，对应 7 个组件（`AssetCardGrid` / `CharacterEditor` /
> `DubbingPanel` / `SceneEditor` / `ShotMediaPanel` / `VideoEditor` / `PanoramaNodeContent`）
> 已从 `ALLOW_DIFF` 删除 —— 它们逐字一致，判据 C 会盯着别再加回来。
> 顺带把 `composables/useApi.ts` 的知识库上传/覆盖层 API 定义同步到桌面端（纯 TS 增量，
> 两端 import 路径本来就相同）。

`ONLY_DESKTOP` 只有 `utils/nuxtCompat.ts`（**14 处活调用的实现者**，6 个文件里的
`navigateTo` 12 处 + `useState` 2 处全靠它 + `AutoImport.dirs`）。判据 E 认两种「被用到」：
import 其**路径**，或调用其**导出函数**——后者是桌面端的常态（零 import），
2026-11-07 之前只判前一种，于是这条被打成「零引用、真长期不用就删」。
照着删会让整端路由与 Onboarding 状态全变 undefined，构建仍绿。现已按符号判并做过反向验证。

**反向验证已通过**（每条都单独做过红）：只在一端新增文件 → A 红；只改一端一个字节 → B 红；
把已一致的文件留在白名单 → C 红；给契约副本多加一个 type → F 红；
G 两个方向都验过：桌面端 `props.vue` 多一条 CSS 规则 → `G 差异变大：… 3 行 > 登记 2 行`；
删掉桌面端那条 `<RouterLink to="/library">` → `G 差异变小：… 1 行 < 登记 2 行`；
全部复原后退出 0。

> 与第三层的分工要分清：第三层查"桌面端有没有越界引用 Web"（耦合）；
> 第五层查"两端有没有各改各的"（漂移）。前者防污染，后者防分叉。

### 迁移一个新功能的固定流程

1. 在 Web 端 `frontend/app/` 完成并验证
2. **复制**目标文件到 `desktop/frontend/src/` 对应位置
3. 跑 `npm run check:isolation`（A/B 必须绿）
4. 跑 `npm run typecheck`（`vue-tsc` 零错误）
5. 跑 `npm run check:parity`（A/B/C 必须绿）；若该文件**确属**桌面 shim，进 `ALLOW_DIFF` 并写 why
6. 若有 Nuxt API 命中：优先加 shim（第二层表），其次逐处改写
7. 提交时 Web 端与桌面端**分开两次**，便于单独回退

---

## 5. 契约管理（唯一需要防漂移的地方）

代码独立后，两端会各自演进，这是允许的。但**后端 API 的字段不能靠人记**，否则一端改了另一端运行期才炸。

- 单一真源：后端 FastAPI 的 OpenAPI（`/api/v1/openapi.json`）
- 当前状态：两端各持一份 `types/contracts.ts` **独立副本**（10 个 interface）。
  桌面端那份的文件头已改为"本工程自有、与 Web 端互不 import"，
  不再残留"从 Nuxt alias 引用"这类指向对方的描述。
- 现有机器判据：`check:parity` 的 **F** 项 —— 两端 `contracts.ts` **去注释后必须逐字一致**
  （2026-11-04 实测两边归一后同为 1792 字符），字段级漂移立刻红；文件头允许各写各的身份说明。
  这样"抄来的副本"不会静默分叉，也仍然不需要共享代码。
- 建议演进（尚未实施）：两端各自用 `openapi-typescript` 从 OpenAPI 生成 `types/api.generated.ts`，
  生成命令写进各自 `package.json`。这样契约变更 → 重新生成 → 编译期报错，
  而不是运行期静默错位。

> 这一步**不是**共享源码：两端各自生成各自的文件，只是输入相同。

---

## 6. 构建与发布

### Web 端（零改动）

```bash
cd frontend
npm run build        # → frontend/.output/public
```

### 桌面端

```bash
cd desktop
npm run build        # = prebuild(gen-icons → sync-frontend --build → sync-backend) + tauri build
```

`sync-frontend.mjs` 已改造：

- 旧：`frontend/` 跑 `nuxt generate`（`NUXT_PUBLIC_HASH_ROUTER=1`）→ 复制 `.output/public` → `dist-frontend`
- 新：`desktop/frontend/` 跑 `npm run build` → 产物**直接**落在 `dist-frontend`（Vite `outDir` 就是它），**无复制步骤**

脚本还加了两道断言：
- `dist-frontend/index.html` 含 `/_nuxt/` ⇒ 报错退出（说明构建链没切过来）
- 不含 `./assets/` 相对引用 ⇒ 报错退出

`tauri.conf.json` 的 `frontendDist: "../dist-frontend"` 继续成立（Vite `outDir` 就指向它）。

### 本地开发（两个终端）

`devUrl` 与 Vite `server.port` 都是 **3014**，刻意避开 Web 端 `nuxt dev` 占用的 3013
（Vite 开了 `strictPort`，撞端口会直接起不来而不是静默换端口）：

```bash
# 终端 1：桌面端界面（Vite，3014）
cd desktop && npm run dev:web

# 终端 2：Tauri 窗口（连 3014）
cd desktop && npm run dev

# Web 端开发（Nuxt，3013，可与上面同时跑）
cd frontend && npm run dev
```

> `dev:web` 已从 `npm --prefix ../frontend run dev`（Nuxt）改为
> `npm --prefix frontend run dev`（desktop/frontend 的 Vite）。
> 若忘了改这条，`tauri dev` 会连到空端口、表现为白屏。

> ⚠️ **改端口要改三处**：`tauri.conf.json` 的 `devUrl`、Vite 的 `server.port`、
> 以及 `src-tauri/src/config.rs` 的 `DESKTOP_ORIGINS`（CORS 白名单）。
> 第三处最容易漏，而漏了的症状极具误导性：**界面全空、控制台一片 CORS 报错，
> 但 `curl 127.0.0.1:5790/api/v1/health` 是通的**，很容易误判成"后端没起来"。
> 2026-11-04 从 3013 改 3014 时真踩过一次，现由 `cargo test` 的
> `config::tests::dev_origin_is_allowed` 守着——它读 `tauri.conf.json` 的实际
> `devUrl` 去比对白名单，两处一漂移就报。

### 桌面端的运行时资源（媒体路径 / CSP / 字体）

这三件事在 Web 端"自动是对的"，在桌面端全要显式处理：

**① 媒体路径必须变绝对地址。** 后端返回的图片/音频/合集字段是**根相对**的
（`static/images/<uuid>.png`，不含域名）。桌面端 origin 是 `tauri.localhost`，根相对会打到
壳自己的静态目录 ⇒ **整片裂图**；更坑的是 dev 阶段看不出来：Vite 配了 `/static` proxy，
根相对在 3014 下依然 200。

⇒ 统一过 `mediaUrl()`（两端 `utils/apiBase.ts` 各一份，**行为与函数体逐字一致**），并由守卫 **F**
禁止再手拼 `'/' + p`。`resolveImageUrl`/`resolveUrl`/`normPUrl`/`normEpUrl` 与 `openImageViewer`
都收敛到它，页面里约 30 处调用点因此不必各写一套。已是绝对地址（`http:`/`data:`/`blob:`…）
的一律原样返回，所以调用方重复拼也不会拼坏。

⚠️ Web 端**不是**「同源所以不用管」：早先 Web 端那 20 余处写的是 `'/' + p` 手拼，与桌面端
分叉但两边都能跑（Web 同源 ⇒ 看不出问题），于是按「改 Web → 同步到桌面」的流程操作时，
桌面端的修复会被整份覆盖回手拼。2026-11-06 已把 Web 端全部改成 `mediaUrl()`（含 6 个编辑/媒体
组件、3 个 library 页、角色/场景/剧集三页、画布页与节点卡片），`check:parity` 的白名单同步
收掉这 7 条。**任何新页面写媒体地址时只准调 `mediaUrl()`，不要判断"Web 同源无所谓"。**

**画布节点卡片是这条纪律的漏网口**（2026-11-06 补上）。`CanvasNodeCard.vue` 的
`<img>` / `<video>` / `<audio>` 三处一直写 `:src="props.node.url"` —— **裸绑后端字段**：
既不是 `'/' + p`，也不是模板串插值，守卫 F 的前两种形态一条都盖不到；而 `node.url` 装的
正是 `primaryUrl`（`/static/engine/audio_seed777.wav` 这种根相对路径）原样存进
localStorage 的值 ⇒ 桌面端打包后画布上的图片/视频/音频卡**整片 404**，dev 阶段同样被
Vite 的 `/static` proxy 掩盖得干干净净。
修法：三处标签 + 裁剪对话框 + 画布页的对比视图 / 重绘蒙版底图 / 切分与取帧（这几处都在
**浏览器里**加载像素）一律过 `mediaUrl`；而**送后端**当参考图的地址仍保持根相对
（后端只认自己的路径，喂绝对地址会多一层解析 ✗）。
守卫 F 的字段名正则补 `node.url`，并加一条自检 —— 正则哪天被改窄，自检先红。
反向验证：把 audio 那处改回裸绑 ⇒ 精确报 `CanvasNodeCard.vue:717（模板绑定 props.node.url 未走 mediaUrl）`。

origin 的来源是**运行时注入**，不是构建期常量：`main.ts` 先 `await initDesktopBridge()`
（内部 `invoke('app_bootstrap')` 拿 `apiOrigin`）**再** `createApp().mount()`；
Web 端由 `plugins/desktop-bridge.client.ts` 承担同一职责。注入必须早于首屏，
否则第一批请求会打到 `tauri://localhost/api/v1` 上 404（症状是"首屏空白、手动刷新一次又好了"）。

> `injectedOrigin` 存的是 **`shallowRef`**：后端重启会换端口（5790 被占就落 5791，
> 见「启动后端」按钮的 `startBackend()`），拼在 computed 里的媒体地址必须跟着重算。
> 早期写成普通变量时，接口会跟着新端口走、图片却钉在旧端口上，**症状是"数据都在、图全裂"**，
> 极易误判成后端挂了。

> 附带修掉一个跨源坑：`<a download>` 在跨源时被浏览器忽略（会变成直接打开），
> 合集下载改为 `fetch → blob → <a download>`。这条依赖 CORS：`config.rs::DESKTOP_ORIGINS`
> 里已含 `tauri://localhost`、`http(s)://tauri.localhost`（Windows 打包端真实 origin）
> 与 `localhost:3013/3014`，漏了就是"界面全空 + 控制台一片 CORS 报错"。

**② CSP 的 `img-src` 放开 `https:`，其余一概不放开。** 现行：

```
img-src 'self' data: blob: https: http://127.0.0.1:* http://localhost:*
media-src 'self' data: blob: http://127.0.0.1:* http://localhost:*
connect-src 'self' ipc: http://ipc.localhost http://127.0.0.1:* http://localhost:* ws://…
```

`https:` 只加在 `img-src` 这一处，是因为**素材库的图片字段仍收自由输入的地址**：
`pages/library/characters.vue` 的「头像」与 `pages/library/scenes.vue` 的「场景图」都是一个
可手填的输入框（Web 端一直允许外链图，桌面端拦掉就成了"填了不显示、且没有任何提示"的静默裂图）。

> 2026-11-04 这两个字段的主路径已从「手填 URL」改成 **本地上传**（见下面 ④）：上传走
> `/api/v1/upload/image`，落成 `static/uploads/<uuid>.png`，完全离线自洽。但 URL 输入**保留**
> （老数据、外部图仍有人用）⇒ `https:` 这一格**继续留着**，不能因为"上传够用"就撤掉。

`media-src` / `connect-src` / `script-src` / `style-src` 仍然只认本机后端与 `'self'`。

> 这条走过一次弯路：上一轮判"没有外部图使用方"时，检索的是 `mediaUrl(` 的调用点，
> 而这两处是 `backgroundImage: url(${item.imageUrl})` 的内联样式，两个判据都不覆盖 ⇒ 漏判。
> 现在守卫 **F** 补了第三种形态（`url(${…})` 插值行里必须出现 `mediaUrl(`），
> 顺手把这三处（含画布全景节点）都接到了 `mediaUrl`，用户填相对路径也能用。

**④ 素材库图片改成上传（本轮）**。`characters.vue` / `scenes.vue`（**两端同构，改一处必须同步另一处**）
的图片字段现在是「输入框 + 上传按钮 + 预览图」：选文件 → `uploadAPI.image(file)` →
回填 `form.imageUrl`（后端回的是 `/static/uploads/<uuid>.png`，与既有 `static/images/…` 同口径，
`mediaUrl()` 两种情况都幂等处理）。三个配套点：

- **列表头像不用 CSS 背景图**。`background-image` 加载失败**没有任何事件**（CSP 拦了、文件被清掉了，
  界面都只是静静地空一格）；改 `<img @error>` + `brokenAvatars` 集合，失败回退姓名首字 / 占位块，
  「填了但加载不出来」这件事在界面上是看得见的。
- **表单字段也要 `@error`**：`formAvatarBroken` / `formImageBroken` 在打开新建/编辑时清零，
  否则上一次失败的状态会传给下一张。
- **`.field-row > input` 必须显式 `width:auto`**：`modal input { width:100% }` 会把上传按钮挤出容器。

同期修掉一个**后端契约缺口**（`services/resource_library.py::_decorate`）：写入侧
`create_item`/`update_item` 一律按 `camel(column)` 取 body（`imageUrl`/`clothingStyle`…），
读出侧原先**只给 `out_aliases`（`reference_images` 那几个）补 camel 别名**，普通列只回 snake_case
⇒ 前端 `item.imageUrl` 恒为 `undefined`，症状是「图存得进、界面上永远空着、编辑表单被清空」。
现改为遍历 `(*spec.columns, *spec.out_aliases)`，snake 与 camel 两种拼写并存
（旧调用方不改），`tests/smoke_test.py` 有两条断言钉住「详情/列表都能拿到 camelCase」。

**⑤ 构建纪律：改了 `backend-py` 只能走 `npm run build`（本轮踩到的真坑）。**
`tauri.conf.json` 的 `bundle.resources` 把 `desktop/dist-backend/backend-py` 作为资源打进包，
构建时 Tauri 会把它复制到 `target/release/backend-py`，而运行时 `resolve_backend_dir()` 的
**第一优先**就是"exe 同级 `backend-py`"（安装版靠这一条找到后端）。

⇒ 直接 `npx tauri build` **不会重新同步后端**：它只会拿上次 `sync:backend` 留下的
`dist-backend` 快照。实测症状极像"代码没生效"——改了 `resource_library.py`，用 release 壳验证，
界面依旧是"图存得进、不显示"，因为跑的是 `target/release/backend-py/…` 里**几周前的副本**。
判据：看那份副本的 `LastWriteTime`，以及壳拉起的 python 是不是 `backend-py/.venv` 里那个
（那份副本没有 `.venv` ⇒ 壳退回 PATH 上的系统 python）。

- 正确入口：`cd desktop && npm run build`（`prebuild` = 图标 → `sync:frontend --build` →
  `sync:backend` → `tauri build`），跑完再起壳验证。
- 只重建壳（改了 Rust 或要复测）才可以用 `npx tauri build --no-bundle`，但**前提是刚跑过 `npm run sync:backend`**。

> 已知边界（既有，非本轮引入）：跨源 `fetch()` 读 `/static/…` **可能**命中 `<img>` 先前缓存下来的
> 那份响应（`Cache-Control: public, max-age=86400`，无 `Vary: Origin`）而被 CORS 拒绝；
> 带 cache-busting 的冷请求 200 正常。影响面仅"下载/导出"这类 `fetch → blob` 路径，
> `<img>` / `<video>` 不经 CORS、不受影响（素材库图片显示已实测 `complete=true` + `naturalWidth` 正确）。

**⑥ 模板里的自动导入必须开 `vueTemplate`（"项目卡片点不进去"的真因）。**
症状：桌面端点首页项目卡片**完全没有反应**（不跳转、不报红），控制台只有一条
`TypeError: _ctx.navigateTo is not a function`。

根因：`unplugin-auto-import` 的 `vueTemplate` 默认 `false`，只扫 `<script>`——模板里的
标识符不被注入，Vue 编译器把 `@click="navigateTo('/drama/1')"` 编成 `_ctx.navigateTo(...)`，
运行时是 `undefined`。之所以长期没暴露：`mediaUrl()`、`apiBase()` 这类函数在 script 里
也有调用点（或文件里显式 `import` 过），注入顺带成立；而 **`navigateTo` 全仓零显式 import**，
`pages/index.vue` 的卡片点击正好落在盲区。

修法（`desktop/frontend/vite.config.ts`）：`AutoImport({ ..., vueTemplate: true })`。
Web 端（Nuxt）不受影响，它的自动导入默认覆盖模板。

验证：重打壳后 CDP 程序化 `card.click()` → `location.hash` 由 `#/` 变 `#/drama/1`，
详情页（返回/角色/场景/提示词/编辑剧集/添加集）正常渲染，`Runtime.exceptionThrown` 为 0。

回归守卫：`check:isolation` 判据 **H**。它先剥注释再读 `AutoImport({...})`，
所以把那行**注释掉**也会红——旧写法直接扫全文，注释里留着 `vueTemplate: true` 照样命中，
属于假绿（本条踩过的坑，2026-11-07 补的剥注释）。

> 同族风险：任何**只在模板里出现**的 auto-import（组件名外的函数，如 `fmtDate`/`useToast`）
> 都会踩同一个坑。判据是"打开控制台看有没有 `_ctx.xxx is not a function`"，
> 而不是去页面里逐个试点击。

**③ 字体不许走外链。** `studio.css` 原本 `@import` Google Fonts，被
`style-src 'self' 'unsafe-inline'` 直接拦掉，每次启动报一条 security error、字体静默失效
（断网时本来也是空的）。桌面端这份改为**本地字体栈**（标题中文明体 / 正文系统无衬线），
由守卫 **G** 保证 CSS/HTML 里不再出现 http(s) 资源。这也正是桌面端与 Web 端
`studio.css` 的**唯一**差异——Web 端照旧走 webfont。

**打包端怎么观测**：release 没有 devtools，用
`$env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS="--remote-debugging-port=9222"` 起壳，
再连 `http://127.0.0.1:9222/json` 走 CDP 读 `document.images`（`complete`/`naturalWidth`）
与 security 日志——这是目前唯一能"看见"打包端渲染结果的口子（截图人看得见、机器读不了）。

### 两者互不影响

| 动作 | Web 端影响 | 桌面端影响 |
|---|---|---|
| `frontend && npm install` | 自身 | 无 |
| `desktop/frontend && npm install` | 无 | 自身 |
| `frontend && npm run build` | 自身产物 | 无 |
| `desktop && npm run build` | 无（不读 `frontend/`） | 自身产物 |

---

## 7. 已验证 / 未验证

### 已验证（本轮实测）

| 项 | 结果 |
|---|---|
| 桌面端 `vite build` | 成功，20 个页面各自成 chunk（`episode` 218.5 KB） |
| 桌面端 `vue-tsc` | exit 0，**零错误** |
| 产物完整性 | 66 文件 / 1.00 MB，入口 `./assets/...` 相对路径 |
| 隔离守卫 | 11 项全绿；反向验证多项均被拦住（A/B 跨工程 import、A2 跨工程 `@import`、F 四种手拼形态、F 裸绑字段、G 壳 HTML 外链字体） |
| 依赖安装 | 独立 `package-lock.json` 60 KB，无 nuxt |
| `sync-frontend.mjs` | 产物归属断言通过 |
| `npm run build`（typecheck + build） | exit 0，1.22s |
| `tauri build --no-bundle` | exit 0，48~49s，产出 `DramaStudio.exe` **3.35 MB**（上限 6 MB） |
| 体积预算 | 5 项全部达标（前端 1.00 MB / 2 MB，传输 0.30 / 0.75 MB） |
| **产物缺席实验** | 把 `frontend/.output` 与 `frontend/dist` 临时改名后，桌面端 `npm run build` 仍 exit 0（1.17s） |
| **耦合审计** | `desktop/` 内对 `frontend/` 的命中**全部在注释**，零可执行引用；三个 `node_modules` 均为真目录（无软链/共享） |
| **构建期自包含断言** | 正常构建 `outsiders=0`；塞入 `desktop/_probe_external.mjs` 后 `vite build` exit 1 且清单精确点名该文件；移除后恢复 0 |
| **Web 端反向钩子** | `hashMode` 死代码已删；`frontend/package.json` 从未设置该环境变量（删前已核）；删后 Web `npm run build` **exit 0** |
| **路由策略互不影响** | Web 产物全量 chunk 零 `createWebHashHistory` 特征（history）；桌面入口 chunk 含 `.hash`（hash）⇒ 两端各自演进 |
| **端口冲突** | Web dev 3013 与桌面 dev 原同为 3013 ⇒ 桌面端改 3014，`devUrl` 与 Vite `server.port` 同步 |
| **`tauri dev` 冒烟** | 窗口标题「短剧工坊」创建成功；壳自启后端监听 5791；首屏 9 个 API（`dramas`/`stats`/`local-models/roots`/`skills`/`ai-configs/*`/`style-profiles`/`visual-graph`）**全部 200** |
| **dev origin 的 CORS** | `OPTIONS`/`GET` 带 `Origin: http://localhost:3014` ⇒ 响应头 `access-control-allow-origin: http://localhost:3014` |
| **退出回收** | 发 `WM_CLOSE` 后壳进程、uvicorn 子进程、5791 端口**全部消失**，无孤儿 |
| **`cargo test --lib`** | 10 项全过（含新增的 `dev_origin_is_allowed`，反向验证：移除白名单里的 3014 ⇒ exit 101 并点名） |
| **打包端媒体加载** | 起真实 `DramaStudio.exe`（`http://tauri.localhost/#/drama/1/character/1`），CDP 读页面：`<img src="http://127.0.0.1:5790/static/images/dd706aaa-….png">`、`complete:true`、`naturalWidth×Height = 2048×1536`；后端日志同步出现该 `GET … 200`。⇒ `mediaUrl()` 与 CSP `img-src` 双双成立 |
| **CSP 拦住外链字体** | 同一轮 CDP 的 security 日志抓到 `https://fonts.googleapis.com/css2?… violates … style-src`（`studio.css` 的 `@import`）。改为本地字体栈后 → 该条消失、`getComputedStyle(body).fontFamily` 落到 `system-ui/Segoe UI/Microsoft YaHei` 兜底 |
| **`dragDropEnabled`** | 已改 `false`。依据：Tauri 自带 schema 原文「Disabling it is required to use HTML5 drag and drop on the frontend on Windows since we replace the drag drop handler of WebView2」；前端拖放走的是 HTML5（`CanvasNodeCard.vue` 读 `ev.dataTransfer?.files?.[0]`），且全仓零 `onDragDropEvent`/`getCurrentWebview` 调用 ⇒ 原 `true` 会让画布拖文件收不到 `File` |
| **打包端素材库图片（两种形态）** | 起真实壳走 CDP 读 `#/library/characters` 与 `#/library/scenes`：老数据 `static/images/dd706aaa-….png` → `http://127.0.0.1:5790/static/images/…`、上传新数据 `/static/uploads/smoke-upload.png`（**带前导斜杠**）→ `http://127.0.0.1:5790/static/uploads/…`，两张都 `complete:true`、`naturalWidth×Height = 2048×1536`，`@error` 回退分支未触发 ⇒ `mediaUrl()` 对两种写法都幂等 |
| **打包端上传接口** | 在壳内页面执行 `POST /api/v1/upload/image`（multipart，等价 `uploadAPI.image`）→ `200`、回 `{"url":"/static/uploads/<uuid>.png","path":"static/uploads/<uuid>.png"}`，再用 `new Image()` 加载该 url → `loaded 1x1` ⇒ CSP `connect-src`、CORS、落盘目录与 `/static` 路由四处对齐 |
| **后端 camelCase 别名** | 临时库 `POST /api/v1/character-library`（body 用 `imageUrl`/`clothingStyle`）→ 详情与列表都同时回 `imageUrl`+`image_url`、`clothingStyle`+`clothing_style`；`tests/smoke_test.py` **484/484** 通过 |

### 未验证（需真机执行）

| 项 | 原因 |
|---|---|
| 跨 OS（macOS WKWebView / Linux WebKitGTK） | Vite `target` 已取三者交集（`chrome105`/`safari15`/`firefox115`），但未在真机跑 |

## 8. Web 端残留的桌面端钩子（不影响独立性）

桌面端独立后，Web 端仍有 4 处"当初为桌面端存在"的代码。它们**不构成共享源码**——
桌面端不 import 它们，它们只在 Web 端自己的构建里生效——但需要知情：

| 位置 | 作用 | 现状 |
|---|---|---|
| ~~`nuxt.config.ts` 的 `hashMode`~~ | 供**旧** Nuxt 桌面构建切 hash 路由 | **已删除**（2026-11-04）。`frontend/package.json` 的 `build`/`generate` 从不设置该环境变量，故该分支从未在正式构建中生效；桌面端改用 Vite hash 路由后彻底无用 |
| `app/app.vue` 的 `<DesktopStatusBanner />` | 在 Tauri 里显示后端状态条 | 保留（Web 端被桌面壳加载的回退路径仍需要） |
| `app/plugins/desktop-bridge.client.ts` | 注入 API base 到 `window` | 同上 |
| `app/composables/useDesktop.ts`、`app/utils/apiBase.ts` | 探测 Tauri 运行时 | 同上 |

剩下的三项是 Web 端**自己的**运行时兼容逻辑（浏览器里是 no-op），不是桌面端的依赖，
因此不构成耦合；只有 `hashMode` 那条是"Web 端为桌面端留的构建开关"，已清除。

### 已退役（保留作回退）

`frontend/nuxt.config.ts` 的 `NUXT_PUBLIC_HASH_ROUTER` 分支与
`desktop/scripts/sync-frontend.mjs` 的旧 Nuxt 路径已不再使用。
Web 端不设该环境变量时行为完全不变，故**未删除**，真机验证通过后可清理。

---

## 9. 响应式（两端唯一的样式同源）

断开工程后**只有一件事必须逐字相同**：两端 `assets/studio.css` 末尾的「响应式基础层」
（Web `frontend/app/assets/studio.css` ↔ 桌面 `desktop/frontend/src/assets/studio.css`），
它是断点与留白的唯一真源；两份文件的差异只允许是顶部字体 `@import`（桌面端为 CSP 改系统字体栈）。

- 机器守卫：`npm run check:responsive`（两端各一份 `scripts/check-responsive.mjs`，逐字相同），
  比对基础层是否逐字一致、每个页面是否都有窄屏规则、是否出现缺 `min()` 兜底的硬宽度网格；
  「窄屏规则 / 硬宽度网格」两条都在**剔除注释后**的文本上判（注释遮蔽见 `docs/responsive-adaptation.md` §8）；
- 改样式的固定顺序：**改 Web 端 → 把 `<style` 之后整段同步到桌面端同名文件 → 跑守卫 → 跑多视口实测**；
- 设备范围、关键页面清单、验收判据与实测记录见 `docs/responsive-adaptation.md`。
