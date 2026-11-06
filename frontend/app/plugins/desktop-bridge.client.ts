/**
 * 桌面平台桥的初始化时机 —— **必须是首屏渲染之前**。
 *
 * `enforce: 'pre'` + `async setup`：Nuxt 会 await 本插件再挂载 app ⇒ 页面里第一个请求
 * （`useApi` 的 fetch、`EventSource`）就已经拿到桌面端注入的绝对后端 origin ✓。
 * 若改成 `onMounted` 里初始化，首屏那几个请求会打到 `tauri://localhost/api/v1` 上 404 ✗
 * （症状：桌面端首屏空白，但手动刷新一次又"好了"）。
 */
import { initDesktopBridge } from '../composables/useDesktop'

export default defineNuxtPlugin({
  name: 'desktop-bridge',
  enforce: 'pre',
  async setup() {
    await initDesktopBridge()
  },
})
