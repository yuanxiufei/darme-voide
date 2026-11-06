/**
 * 画布插件注册表 —— **前端侧单一真源** ✓。
 *
 * 与后端 `backend-py/app/services/canvas_plugins.py::BUILTIN_PLUGINS` 逐项同源，
 * 由 `backend-py/tests/canvas_plugins_test.py` 的「前端对拍」断言防止两套漂开。
 *
 * ## 为什么清单要抄一份到前端（不是重复代码）
 *
 * 卡片渲染是**同步**决定的：`CanvasNodeCard.vue` 拿到节点就得当场知道「这个 type 用哪个组件、
 * 默认多大」。注册表在 HTTP 上 ⇒ 等请求回来再渲染会出现「先白框、后填内容」的闪烁，
 * 首屏与离线还直接裂开 ✗。所以清单与组件名在前端留一份，**由测试保证与后端一字不差**。
 *
 * ## 插件自定义键落哪里
 *
 * 落 `node.values`（一律字符串，与 config 卡同规矩）✓ ——
 * **不是** `node.metadata`：本仓节点字段是平铺的，多造一层 `metadata` 只会让
 * 「读 x 读不到、写 content 写不进去」这类错位重演 ✗。后端的
 * `RESERVED_METADATA_KEYS` 就是挡「插件键占用平铺真字段」，与这里一一对应。
 */

import type { CanvasPluginNodeType } from '../composables/useCanvas'

/** `<pluginId>:<name>` —— 与后端 `NODE_TYPE_PATTERN` **同一套规则** */
export const PLUGIN_TYPE_PATTERN = /^[a-z0-9][a-z0-9-]*:[a-z0-9][a-z0-9-]+$/

export interface CanvasPluginNodeSpec {
  /** 节点类型，形如 `sticky-note:note` */
  type: CanvasPluginNodeType
  pluginId: string
  title: string
  /** 字符串图标（emoji 或 lucide 名）—— 源的 ReactNode 在 Vue 侧吃不下 ✗ */
  icon: string
  defaultSize: { w: number; h: number }
  /** 落到 `node.values` 的默认键（**一律字符串**） */
  defaultMetadata: Record<string, string>
  minimapColor: string
  /** 渲染组件名 —— 与后端 `contentComponent` 同名，改一处必须两处一起改 */
  contentComponent: string
}

/**
 * 5 个内置插件节点（源的 6 个里 `template` 不迁：它是给插件作者的起步样板，注释原文「改我」）。
 *
 * ⚠️ 顺序与后端 `BUILTIN_PLUGINS` 的节点顺序一致 —— 对拍测试按 type 查表，顺序不影响判定，
 * 但保持同步能让 diff 一眼看出「谁被漏加了」✓。
 */
export const PLUGIN_NODES: CanvasPluginNodeSpec[] = [
  {
    type: 'html:render',
    pluginId: 'html',
    title: 'HTML',
    icon: '🌐',
    defaultSize: { w: 420, h: 320 },
    defaultMetadata: { content: '' },
    minimapColor: '#ec4899',
    contentComponent: 'HtmlNodeContent',
  },
  {
    type: 'markdown:doc',
    pluginId: 'markdown',
    title: 'Markdown',
    icon: '📝',
    defaultSize: { w: 360, h: 300 },
    defaultMetadata: { content: '' },
    minimapColor: '#6366f1',
    contentComponent: 'MarkdownNodeContent',
  },
  {
    type: 'panorama:viewer',
    pluginId: 'panorama',
    title: '3D 全景',
    icon: '🌐',
    defaultSize: { w: 480, h: 300 },
    defaultMetadata: {},
    minimapColor: '#0ea5e9',
    contentComponent: 'PanoramaNodeContent',
  },
  {
    type: 'sticky-note:note',
    pluginId: 'sticky-note',
    title: '便利贴',
    icon: '📌',
    defaultSize: { w: 240, h: 200 },
    defaultMetadata: { content: '', pluginColor: '#fde68a' },
    minimapColor: '#f59e0b',
    contentComponent: 'StickyNoteNodeContent',
  },
  {
    type: 'svg:vector',
    pluginId: 'svg',
    title: 'SVG',
    icon: '🔷',
    defaultSize: { w: 320, h: 320 },
    defaultMetadata: {},
    minimapColor: '#14b8a6',
    contentComponent: 'SvgNodeContent',
  },
]

const BY_TYPE = new Map<string, CanvasPluginNodeSpec>(PLUGIN_NODES.map((item) => [item.type, item]))

/** 是不是插件节点类型（`<id>:<name>`）✓ */
export function isPluginNodeType(type: unknown): type is CanvasPluginNodeType {
  return typeof type === 'string' && PLUGIN_TYPE_PATTERN.test(type)
}

/** 查清单（未登记 ⇒ `undefined`：渲染层据此显示「未安装的插件」，**不猜**组件 ✗） */
export function pluginOf(type: unknown): CanvasPluginNodeSpec | undefined {
  return typeof type === 'string' ? BY_TYPE.get(type) : undefined
}

/** 全部已登记的插件节点 type ✓ */
export const PLUGIN_NODE_TYPES: CanvasPluginNodeType[] = PLUGIN_NODES.map((item) => item.type)

/**
 * 插件节点的有效 `values`（默认值 + 已有值）✓。
 *
 * 只补**清单里声明过**的键：插件升级加了新键才会自动出现默认值，
 * 而用户自己写的键原样保留（不拿默认值覆盖用户输入 ✗）。
 */
export function pluginValuesOf(
  type: unknown,
  values: Record<string, string> | undefined,
): Record<string, string> {
  const spec = pluginOf(type)
  if (!spec) return values ? { ...values } : {}
  return { ...spec.defaultMetadata, ...(values || {}) }
}
