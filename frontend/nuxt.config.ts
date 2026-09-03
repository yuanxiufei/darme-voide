import { fileURLToPath } from 'node:url'

export default defineNuxtConfig({
  srcDir: 'app/',
  ssr: false,
  // 共享契约类型（backend/src/shared/contracts.ts）：import type 别名，字段单一来源
  alias: {
    '~contracts': fileURLToPath(new URL('../backend/src/shared/contracts.ts', import.meta.url)),
  },
  typescript: {
    tsConfig: {
      compilerOptions: {
        paths: {
          '~contracts': ['../backend/src/shared/contracts.ts'],
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
