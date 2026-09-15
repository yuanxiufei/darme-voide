import { fileURLToPath } from 'node:url'

// 后端地址：Python 后端默认 **5790**（Node 旧后端是 5789）。
// ⚠️ 2026-09-15 从 5789 切到 5790 —— Python 侧已**全量覆盖**（227 端点 / 未注册 0），
// 且未注册路径仍会由它反代给 Node（PROXY_TO_NODE）⇒ 前端对着它跑是自足的。
// 需要临时对着 Node 调试：`NUXT_API_TARGET=http://localhost:5789 npm run dev`
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
      proxy: {
        '/api': { target: 'http://localhost:5789', changeOrigin: true },
        '/static': { target: 'http://localhost:5789', changeOrigin: true },
      },
    },
  },
  compatibilityDate: '2025-05-15',
})
