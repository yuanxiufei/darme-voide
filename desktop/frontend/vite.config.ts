/**
 * 桌面端独立构建配置。
 *
 * ## 与 Web 端（Nuxt）的关系
 *
 * **零关系**。本工程不 import `frontend/` 下任何文件，`resolve.alias` 只指向本工程
 * `src/`。源码在迁移时是**复制**过来的，此后各自演进（`check:isolation` 守这条）。
 *
 * ## 三个不可省的插件/配置
 *
 * 1. `unplugin-auto-import` —— 源码里 84 个文件只有 33 个显式 `from 'vue'`，
 *    其余全靠自动导入；且页面直接调 `useApi()` 等 composable 而不 import。
 *    Nuxt 用自研 `unimport` 做这事，独立工程必须自带等价能力。
 * 2. `unplugin-vue-components` —— 54 个组件里只有 7 处显式 import。
 * 3. `base: './'` —— Tauri 用自定义协议提供资源（Windows `http://tauri.localhost`、
 *    mac/linux `tauri://localhost`）。相对路径不依赖协议根，比绝对路径稳。
 *
 * ## 为什么 `outDir` 指向 `../dist-frontend`
 *
 * `src-tauri/tauri.conf.json` 的 `frontendDist` 是 `../dist-frontend`（相对
 * `src-tauri/`）⇒ 即 `desktop/dist-frontend`。本工程在 `desktop/frontend/`，
 * 所以 `../dist-frontend` 正好落在同一处，**tauri.conf.json 无需改动**。
 * Web 端产物在 `frontend/.output/public`，两者物理隔离。
 */
import { writeFileSync } from 'node:fs'
import { isAbsolute, resolve } from 'node:path'
import { fileURLToPath, URL } from 'node:url'
import { defineConfig, type Plugin } from 'vite'
import vue from '@vitejs/plugin-vue'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'

const src = fileURLToPath(new URL('./src', import.meta.url))
const projectRoot = fileURLToPath(new URL('.', import.meta.url))

/**
 * 独立性硬约束：构建期自证「只吃本工程的文件」。
 *
 * 与 `scripts/check-isolation.mjs` 的分工：
 * - 那个查**源码文本**（import 语句的字面路径），能被 `import(变量)` 这类动态写法绕过；
 * - 这个查 **Rollup 实际读到的模块集合**，绕不过——它由打包器的解析结果直接给出。
 *
 * 判据：除 Rollup 虚拟模块外，每个模块都必须落在本工程目录内。
 * 一旦有人写了 `import '../../../frontend/app/...'`，或把依赖装到仓库根
 * `node_modules` 被提升上来，构建当场失败，而不是等打进 exe 才发现体积被污染。
 */
function assertSelfContained(): Plugin {
  // 统一成正斜杠再比较：Windows 上 resolve() 给反斜杠，而打包器的模块 id 用正斜杠，
  // 直接 startsWith 会把本工程自己的文件全判成「外部」。
  const root = resolve(projectRoot).replace(/\\/g, '/')
  const prefix = root.endsWith('/') ? root : root + '/'
  return {
    name: 'desktop-assert-self-contained',
    apply: 'build',
    // 用 generateBundle 而不是 buildEnd + getModuleIds：Vite 8 底层是 rolldown，
    // 插件上下文对 getModuleIds 的支持不稳；chunk.modules 是两种打包器都稳定给的清单。
    generateBundle(_options, bundle) {
      const outsiders = new Set<string>()
      for (const output of Object.values(bundle)) {
        if (output.type !== 'chunk') continue
        for (const id of Object.keys(output.modules)) {
          // `\0` = 打包器虚拟模块；非绝对路径 = 虚拟 id
          if (id.startsWith('\0') || !isAbsolute(id)) continue
          if (!id.replace(/\\/g, '/').startsWith(prefix)) outsiders.add(id)
        }
      }
      if (outsiders.size > 0) {
        const list = [...outsiders].sort()
        // 打包器会把错误文本吞掉，落一份清单便于定位
        const report = resolve(root, 'isolation-outsiders.txt')
        writeFileSync(report, list.join('\n'), 'utf8')
        throw new Error(
          `[isolation] 构建读到了 ${outsiders.size} 个本工程之外的模块，` +
            `桌面端必须自包含（允许范围：${root}），清单见 ${report}：\n` +
            list.slice(0, 10).map((p) => `  - ${p}`).join('\n'),
        )
      }
    },
  }
}

export default defineConfig({
  base: './',
  plugins: [
    assertSelfContained(),
    vue(),
    AutoImport({
      imports: ['vue', 'vue-router'],
      dirs: ['src/composables', 'src/utils'],
      // ⚠️ 必须开：默认 false 只扫描 <script>，**模板里的标识符不注入**。
      // 症状是「点击没有任何反应」，控制台一条
      // `TypeError: _ctx.navigateTo is not a function`（项目卡片进不去就是这个）。
      // 为什么之前没暴露：`mediaUrl()` 那些调用点大多在 script 里也用到、或文件里显式
      // `import` 过，注入顺带成立；而 `navigateTo` 全仓零显式 import，
      // `pages/index.vue` 的卡片 `@click="navigateTo(...)"` 就落进了这个盲区。
      vueTemplate: true,
      dts: 'src/auto-imports.d.ts',
      // 生成物不参与类型检查的循环依赖
      eslintrc: { enabled: false },
    }),
    Components({
      dirs: ['src/components'],
      dts: 'src/components.d.ts',
      // canvas/ 等子目录也要自动注册（Nuxt 默认就是 deep）
      deep: true,
      extensions: ['vue'],
    }),
  ],
  resolve: {
    alias: {
      '@': src,
      '~': src,
      // Web 端由 nuxt.config.ts 的 alias 提供；这里同名重建，
      // 使迁移过来的 useApi.ts 无需改 import 语句
      '~contracts': fileURLToPath(new URL('./src/types/contracts.ts', import.meta.url)),
    },
  },
  server: {
    // 必须与 src-tauri/tauri.conf.json 的 devUrl 一致（3014），
    // 否则 `tauri dev` 起的窗口会连到一个空端口、白屏。
    // 刻意避开 3013：那是 Web 端 `nuxt dev` 的端口，撞上会因 strictPort 直接起不来。
    port: 3014,
    strictPort: true,
    // 桌面 dev 模式下前端仍需一个后端；生产走 Tauri 拉起的内置 Python 服务
    proxy: {
      '/api': { target: 'http://127.0.0.1:5790', changeOrigin: true },
      '/static': { target: 'http://127.0.0.1:5790', changeOrigin: true },
    },
  },
  build: {
    outDir: '../dist-frontend',
    emptyOutDir: true,
    // 跨 OS 兼容基线：Windows 是 Chromium(WebView2)，macOS 是 WKWebView(Safari)、
    // Linux 是 WebKitGTK。取三者交集，而不是只按 Chromium 写。
    target: ['chrome105', 'safari15', 'firefox115'],
    chunkSizeWarningLimit: 1200,
  },
})
