/**
 * 后端 API 基址的**唯一真源**（Web 同源 / 桌面注入）。
 *
 * ## 为什么要有这一层
 *
 * Web 形态下，前端与后端**同源**：dev 由 `nuxt.config.ts` 的 `vite.server.proxy` 把
 * `/api` 转到 `localhost:5790`，生产由部署侧反代 ⇒ 相对路径 `/api/v1` 恒成立 ✓。
 *
 * 桌面形态下**不同源**：Tauri 壳把打包好的前端资源用自定义协议提供（Windows 上是
 * `http://tauri.localhost`，macOS/Linux 上是 `tauri://localhost`），而后端是
 * `http://127.0.0.1:<port>` ⇒ 相对路径会打到壳自己的协议栈上、必定 404 ✗。
 * ⇒ 由壳在启动时注入**绝对基址**（见 `composables/useDesktop.ts` 的 `initDesktopBridge`）。
 *
 * ## 三条纪律
 *
 * 1. **不许在别处再写死 `/api/v1`** —— 桌面端会整套请求打空（症状是"界面全空、控制台
 *    一堆 404"，但后端明明在跑）。
 * 2. **媒体路径（`/static/...`）同样受影响**，而且**后端返回的就是根相对路径**
 *    （`engine.py` 的 `primaryUrl`、上传接口的 `"/" + saved_path` 都是 `/static/...`）。
 *    ⇒ 一切要喂给 `<img>` / `<audio>` / `<video>` / `poster` 的地址，都必须过
 *    `mediaUrl()`；直接写 `'/' + path` 在 dev 下会被 Vite 的 `/static` proxy 掩盖，
 *    **打包后才暴露成一整片裂图**（后端日志里连一条 `/static` 请求都看不到）。
 * 3. 注入必须**早于首次请求**：Web 端靠 `plugins/desktop-bridge.client.ts` 这个 async 插件，
 *    Nuxt 会等它 resolve 再渲染页面 ✓；桌面端靠 `main.ts` 里 `await initDesktopBridge()`
 *    **先于** `createApp().mount()`（改成挂载后再等 ⇒ 首屏请求打到 `tauri://localhost` 上 404，
 *    症状是「首屏空白、手动刷新一次又好了」）。
 */
import { shallowRef } from 'vue'

/**
 * 注入的后端 origin（`http://127.0.0.1:5790` 这种，不带尾斜杠）；`null` = 同源。
 *
 * ⚠️ 用 `shallowRef` 而不是普通变量：后端**重启会换端口**（5790 被占就落到 5791，
 * 见 `startBackend()` 回执），媒体地址是拼在 computed / 渲染函数里的。
 * 普通变量的话 `mediaUrl()` 读不到变化 ⇒ 接口跟着新端口走、图片却钉在旧端口上，
 * 症状是「数据都在、图全裂」，看起来像后端挂了。`desktopApiOrigin` 那个 computed
 * 同理（普通变量 ⇒ 无响应式依赖 ⇒ 永远只算一次）。
 */
const injectedOrigin = shallowRef<string | null>(null)

/** 由桌面壳 / 运行时配置注入后端 origin；传 `null` 恢复同源行为 */
export function setApiOrigin(origin: string | null | undefined): void {
  if (!origin) {
    injectedOrigin.value = null
    return
  }
  // 容忍尾斜杠与误传的 `/api/v1` 后缀：只保留 scheme://host:port
  injectedOrigin.value = origin.replace(/\/+$/, '').replace(/\/api\/v1$/, '')
}

/** 当前注入的后端 origin；`null` 表示走同源相对路径 */
export function apiOrigin(): string | null {
  return injectedOrigin.value
}

/**
 * 拼请求用的基址。
 *
 * ⚠️ 每次调用都现算（不是模块级常量）—— 注入发生在插件阶段，若在模块顶层求值，
 * 早于注入的模块会永久拿到 `/api/v1` ✗。
 */
export function apiBase(): string {
  const origin = injectedOrigin.value
  return origin ? `${origin}/api/v1` : '/api/v1'
}

/**
 * 媒体地址（`/static/...`、`static/...`）→ 能直接喂给 `<img>` / `<audio>` / `<video>` 的 URL。
 *
 * 后端给的媒体路径是**根相对**的；Web 同源下这是对的，桌面壳下会打到 `tauri://localhost`
 * 的静态目录上 ⇒ 404。注入了 origin 就补成绝对地址，没注入（Web）原样返回。
 *
 * 幂等：已是绝对地址（`http:`/`https:`/`data:`/`blob:`/`tauri:`/`file:`）一律原样返回，
 * 所以「调用方已经拼过一次」也不会拼坏。
 */
export function mediaUrl(path?: string | null): string {
  const raw = String(path ?? '').trim()
  if (!raw) return ''
  // 带 scheme 或协议相对 ⇒ 已经是可直接加载的地址
  if (/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(raw)) return raw
  const origin = injectedOrigin.value
  if (!origin) return raw
  return `${origin}/${raw.replace(/^\/+/, '')}`
}
