/**
 * 桌面壳（Tauri 2）平台桥 —— **同一套 UI 跑 Web 与桌面的唯一分叉点**。
 *
 * ## 设计约束（三条）
 *
 * 1. **零 npm 依赖**：不引 `@tauri-apps/api`，直接走壳注入的 `window.__TAURI_INTERNALS__.invoke`。
 *    理由：安装包体积是本项目的硬指标（见 `docs/desktop-tauri-selection.md`），而这里只需要
 *    6 个命令、不需要事件/窗口/菜单等完整 SDK。代价是内部 API 非公开语义 ⇒ **所有调用点
 *    集中在本文件**，将来换官方 SDK 只改这一个文件。
 * 2. **Web 端零副作用**：所有导出在浏览器里都退化成"不是桌面端 + 明确报错"，Web 构建不会
 *    因为缺壳而崩，也不会多打任何请求。
 * 3. **绝不静默降级**：桌面端拉不到 bootstrap 时**不假装成功**（那会让用户看到一堆
 *    "网络连接失败"却不知道后端没起）⇒ 保留 `desktopError` 供设置页/首屏横幅展示。
 */
import { computed, ref } from 'vue'
import { apiOrigin, setApiOrigin } from '../utils/apiBase'

/** 桌面壳命令名（与 `desktop/src-tauri/src/commands.rs` 逐字对应，改一处必改两处） */
export type DesktopCommand =
  | 'app_bootstrap'
  | 'backend_status'
  | 'backend_start'
  | 'backend_stop'
  | 'open_path'
  | 'pick_directory'

export interface BackendStatus {
  /** `stopped` / `starting` / `running` / `failed` / `external` */
  state: string
  /** 后端 origin，如 `http://127.0.0.1:5790` */
  origin: string | null
  port: number | null
  pid: number | null
  /** 是否由本壳拉起（`false` = 复用已在跑的外部后端） */
  owned: boolean
  /** 人话说明（失败原因 / 复用原因） */
  message: string
  /** 后端日志文件绝对路径 */
  logPath: string | null
}

export interface DesktopBootstrap {
  apiOrigin: string | null
  platform: string
  appVersion: string
  dataDir: string
  backend: BackendStatus
}

interface TauriInternals {
  invoke: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>
}

function internals(): TauriInternals | null {
  // 桌面端没有 `import.meta.client`（Nuxt 专有）；本工程只跑在浏览器/WebView 里，
  // 用通用判断即可，同时保留在 Node 侧被 import 时不炸的能力。
  if (typeof window === 'undefined') return null
  const raw = (window as any).__TAURI_INTERNALS__
  return raw && typeof raw.invoke === 'function' ? (raw as TauriInternals) : null
}

/** 是否运行在桌面壳里（Web 端恒为 false） */
export const isDesktop = ref<boolean>(internals() !== null)

const bootstrapped = ref(false)
const bootstrapData = ref<DesktopBootstrap | null>(null)
const desktopError = ref<string | null>(null)

/** 后端状态（桌面端设置页/首屏横幅读它；Web 端恒为 null） */
export const backendStatus = computed<BackendStatus | null>(() => bootstrapData.value?.backend ?? null)

/**
 * 调壳命令。
 *
 * ⚠️ 不在桌面端时**抛错而不是返回空值** —— 调用方若 `await` 一个 `undefined` 会继续往下跑，
 * 症状是"点了没反应"（比报错更难查）。
 */
export async function invokeDesktop<T = unknown>(cmd: DesktopCommand, args?: Record<string, unknown>): Promise<T> {
  const api = internals()
  if (!api) throw new Error(`桌面能力不可用（当前为 Web 形态）：${cmd}`)
  return (await api.invoke(cmd, args)) as T
}

/**
 * 初始化平台桥：拿后端 origin 注入给 `apiBase()`。
 *
 * 由 `plugins/desktop-bridge.client.ts` 在首屏渲染**之前** await ⇒ 页面里的第一次 `fetch`
 * 就已经带着正确基址。
 */
export async function initDesktopBridge(): Promise<void> {
  if (bootstrapped.value) return
  bootstrapped.value = true
  if (!internals()) return // Web：什么都不做，保持同源相对路径

  try {
    const data = await invokeDesktop<DesktopBootstrap>('app_bootstrap')
    bootstrapData.value = data
    setApiOrigin(data.apiOrigin)
  } catch (err: any) {
    // 不设 origin ⇒ 退回同源；把错误摆到界面上，而不是吞掉
    desktopError.value = err?.message || String(err)
    setApiOrigin(null)
  }
}

/** 重新拉一次 bootstrap（后端重启后端口可能变） */
export async function refreshDesktopBootstrap(): Promise<void> {
  bootstrapped.value = false
  await initDesktopBridge()
}

export async function startBackend(): Promise<BackendStatus> {
  const st = await invokeDesktop<BackendStatus>('backend_start')
  if (bootstrapData.value) bootstrapData.value.backend = st
  if (st.origin) setApiOrigin(st.origin)
  return st
}

export async function stopBackend(): Promise<BackendStatus> {
  const st = await invokeDesktop<BackendStatus>('backend_stop')
  if (bootstrapData.value) bootstrapData.value.backend = st
  return st
}

export async function getBackendStatus(): Promise<BackendStatus> {
  const st = await invokeDesktop<BackendStatus>('backend_status')
  if (bootstrapData.value) bootstrapData.value.backend = st
  return st
}

/** 用系统文件管理器打开目录/文件（日志、输出目录、模型目录） */
export async function openPath(path: string): Promise<void> {
  await invokeDesktop('open_path', { path })
}

/** 原生目录选择器；用户取消时返回 `null`（不是抛错） */
export async function pickDirectory(title?: string): Promise<string | null> {
  return await invokeDesktop<string | null>('pick_directory', { title: title ?? '选择目录' })
}

/** 壳/后端诊断信息（设置页"关于"用） */
export const desktopInfo = computed(() => bootstrapData.value)
export const desktopBridgeError = computed(() => desktopError.value)
/** 当前生效的后端 origin；Web 端为 null（同源） */
export const desktopApiOrigin = computed(() => apiOrigin())
