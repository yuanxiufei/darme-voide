import { fileURLToPath } from 'node:url'

// 后端地址：Python 后端默认 **5790**（Node 旧后端 5789 已随 `backend/` 删除）。
// ⚠️ 2026-09-15 从 5789 切到 5790 —— Python 侧已**全量覆盖**（227 端点 / 未注册 0），
// 未实现的路径会回 **501 + 说明**（不再有反代兜底）。
// ⚠️⚠️ 这里**必须**被下面 `vite.server.proxy` 真正引用：2026-09-15 修过一次「改了本变量、
// 却忘了改 proxy 里硬编码的 5789」⇒ dev 模式整套代理打到已删除的 Node 上（静默失效 ✗）。
// 需要临时指向别的上游（如自建服务）：`NUXT_API_TARGET=http://localhost:1234 npm run dev`
// （注意这只是 **dev 代理**；生产静态产物走同源，由部署侧决定）
const backendTarget = process.env.NUXT_API_TARGET || 'http://localhost:5790'

export default defineNuxtConfig({
  srcDir: 'app/',
  ssr: false,
  // 共享契约类型（`app/types/contracts.ts`）：import type 别名。
  // ⚠️ 2026-09-15 从 `backend/src/shared/contracts.ts` 迁到前端 —— 那是 TS 旧后端的位置，
  // 删 `backend/` 会让**前端构建失败**。字段的**权威**在 Python 后端，本文件是前端侧镜像。
  alias: {
    '~contracts': fileURLToPath(new URL('./app/types/contracts.ts', import.meta.url)),
  },
  typescript: {
    tsConfig: {
      compilerOptions: {
        paths: {
          '~contracts': ['./app/types/contracts.ts'],
        },
      },
    },
  },
  devtools: { enabled: false },
  experimental: {
    appManifest: false,
  },
  app: {
    head: {
      title: '短剧工坊',
      meta: [{ name: 'viewport', content: 'width=device-width, initial-scale=1' }],
      link: [
        { rel: 'icon', type: 'image/png', href: '/favicon.png' },
        { rel: 'shortcut icon', type: 'image/png', href: '/favicon.png' },
      ],
    },
  },
  vite: {
    server: {
      // ⚠️ 一律用 `backendTarget`（**别再硬编码**：上一版这里写死 5789 ⇒ 删库后 dev 代理全打到空端口）
      proxy: {
        '/api': { target: backendTarget, changeOrigin: true },
        '/static': { target: backendTarget, changeOrigin: true },
      },
    },
  },
  compatibilityDate: '2025-05-15',
})
