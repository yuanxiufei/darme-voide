/**
 * 桌面端入口。
 *
 * ## 与 Nuxt 的时序差异（重要）
 *
 * Web 端用 `plugins/desktop-bridge.client.ts` 的 `enforce: 'pre'` + `async setup`
 * 保证"注入后端 origin"早于首屏渲染。
 *
 * 桌面端没有 Nuxt 插件系统，等价做法是：`await initDesktopBridge()` **先于**
 * `createApp().mount()`。若改成 mount 后再等（或 onMounted 里做），首屏那几个
 * 请求会打到 `tauri://localhost/api/v1` 上 404 —— 症状是"首屏空白，手动刷新一次又好了"。
 */
import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import { initDesktopBridge } from './composables/useDesktop'

async function boot(): Promise<void> {
  // 1) 先注入后端 origin（非 Tauri 环境下是 no-op）
  await initDesktopBridge()

  // 2) 再挂载，确保首个请求已拿到绝对基址
  createApp(App).use(router).mount('#app')
}

void boot()
