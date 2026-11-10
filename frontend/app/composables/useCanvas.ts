/**
 * 无限画布的状态层（Vue 3 组合式，**无第三方依赖**）。
 *
 * 移植自上游 `ComfyUI-Local-Infinite-Canvas`（MIT）的画布概念：文本 / 图片 / 视频 / 配置 /
 * 分组五类节点 + 连线 + 框选 + 自动保存。原实现是 React + Zustand + Tailwind，
 * 与本仓前端（Nuxt3 + Vue3 + 纯 CSS）栈不同 ⇒ 这里是**重写**，只搬概念与交互口径。
 *
 * ## 三处「写错了不报错、只是手感不对」的口径
 *
 * 1. **缩放要以光标为锚点** ✗：直接在 `scale` 上乘会让内容朝左上跑。
 *    正确是 `newOffset = cursor - (cursor - offset) * (newScale / oldScale)` ✓
 *    （`zoomAt` 收的是**屏幕坐标**，不是世界坐标 ✗）。
 * 2. **屏幕 → 世界**要**先减平移再除缩放** ✓（顺序反了 ⇒ 拖动时节点越拖越快 ✗）。
 * 3. **连线三不许** ✓：不许自连、不许重复、不许连不存在的节点 ——
 *    都要在**写入前**挡掉（写进去再清理会让撤销/持久化脏掉 ✗）。
 *
 * 状态是**模块级单例** ✓：工具栏、画布、侧栏各自 `useCanvas()` 拿到的是同一份，
 * 不需要 provide/inject。
 */
import { ref, watch } from 'vue'
import {
  canGroupSelectedNodes,
  canUngroupSelectedNodes,
  focusViewportForNode,
  getGroupWrapRect,
  normalizeConnection,
  snapNodesIntoGroup,
  type FocusFrame,
} from '../utils/canvasGeometry'
import { DEFAULT_NODE_SIZE } from '../utils/canvasNodeSize'
import { isPluginNodeType, pluginOf, pluginValuesOf } from '../utils/canvasPlugins'
import { recallNodeSettings, rememberNodeSettings } from '../utils/nodeSettingsMemory'

/**
 * 内置六类节点 ✓
 *
 * `audio` 是 2026-11-06 补的 ✓：自研引擎 Music3 档真出 wav ✓，画布原先只认图片/视频 ⇒
 * 歌跑完了只会被按 `.wav` 当图片卡塞进来（一张裂图 ✗）。后端 `canvas_ops.NODE_TYPES`
 * 与这里**逐项一致**，两边任一边漏就会在 `canvas_ops_test` 的对拍判据上报 ✓。
 */
export type CanvasNodeType = 'text' | 'image' | 'video' | 'audio' | 'config' | 'group'

/**
 * 插件节点类型 ✓：**必须**是 `<pluginId>:<name>`。
 *
 * 用模板字面量类型而不是裸 `string`：这样「内置五类 ∪ 带冒号的插件类型」正好就是
 * `CanvasNode['type']` 的合法域 —— 把 `type` 直接放开成 `string` 会让 `NODE_TYPE_LABEL[type]`
 * 之类的查表悄悄返回 `undefined`，而编译器一句话都不说 ✗。
 */
export type CanvasPluginNodeType = `${string}:${string}`

/** 画布上能出现的全部节点类型 ✓ */
export type CanvasAnyNodeType = CanvasNodeType | CanvasPluginNodeType

export interface CanvasNode {
  id: string
  type: CanvasAnyNodeType
  title: string
  /** 世界坐标（不受缩放影响 ✓） */
  x: number
  y: number
  w: number
  h: number
  /** text 节点的正文；config 节点的**提示词**（工作流跑起来就是它 ✓） */
  text?: string
  /** image / video 节点的资源地址 */
  url?: string
  /** 所属分组（成组后写入 ✓；解组时清空 ✓） */
  groupId?: string
  /** config 卡绑定的工作流（注册表 id ✓；空 = 还没选 ✓） */
  workflowId?: string
  /** 工作流变体（空 = 主形态 ✓） */
  variant?: string
  /**
   * 节点上的键值 ✓：config 卡是**工作流参数**，插件节点是它自己的**自定义键**
   * （`utils/canvasPlugins.ts` 的 `defaultMetadata` 就落在这里 ✓）。
   *
   * **一律存字符串** —— 存 number/boolean 会让 localStorage 与后端 JSON 两侧类型漂开 ✗，
   * 提交前按参数类型转 ✓。**这层不是 `metadata`**：本仓节点字段是平铺的，
   * 再套一层只会让「读 x 读不到」重演 ✗。
   */
  values?: Record<string, string>
  /**
   * config 卡的**运行方式** ✓：`engine` = 本仓自研引擎（默认 ✓，不依赖外部服务 ✓）；
   * `comfyui` = 外部 ComfyUI 工作流（注册表 `workflowId` + 8765 上游 ✓）。
   *
   * ⚠️ 老档没有这个键 ⇒ 按 `workflowId` 有没有值推断 ✓（`useCanvas` 的存量数据不迁移 ✗）。
   */
  runMode?: 'engine' | 'comfyui'
  /** 引擎档位（`GET /engine/stages` 的 `stage` ✓；空 = 用清单里的第一个可用档 ✓） */
  engineStage?: string
  /**
   * **上一次没跑起来的原因** ✓ —— 随卡持久化（对应源 `metadata.errorDetails` ✓）。
   *
   * 为什么非要写进节点：闸口拒绝（档位类别不符 / 缺首帧 / 缺件 / 空提示词）时
   * **一条请求都没发** ⇒ 没有 taskId 可回读、没有 `engine.state` 可显示 ✓✗。
   * 只弹 toast 的话，toast 一消失就彻底无痕（刷新更没 ✗），用户过一会儿只见
   * 「点了没反应」；而那句理由恰好是「照做就能跑」的指引（「先连一张图片卡过来」）✓。
   *
   * 落盘后还有两个免费好处：① 卡上看得见（`CanvasNodeCard` 的「上次没跑起来」）；
   * ② 进 `canvasSnapshot()` ⇒ `compact_snapshot` 带进模型视野，下一轮模型自己知道
   * 上一次为什么没成（源 `agent-site-tools.ts` 的 `error` 同义）✓。
   *
   * 成功提交即清空 ✓、再失败就覆盖 ✓ —— 留着旧的会让人以为这次又没跑起来 ✗。
   */
  lastError?: string
}

export interface CanvasEdge {
  id: string
  from: string
  to: string
}

export interface CanvasViewport {
  x: number
  y: number
  scale: number
}

export interface CanvasDoc {
  /** 结构版本 ✓（老档没有这个键 ⇒ 当 1 ✓） */
  version?: number
  /**
   * **内容版本** ✓ —— 每变一次内容 +1（单调，不回落）。与 `version` 完全不同：
   * 那是「文件格式第几版」，这是「这份画布被改过几次」。
   *
   * 它是给 CAS 用的：AI 出草案时记下当时的 `revision`，落地前比一次 —— 对不上说明
   * 这期间用户自己动过画布，那批 op 是按旧画布算的，直接落会覆盖/错位 ✗。
   */
  revision?: number
  nodes: CanvasNode[]
  edges: CanvasEdge[]
  viewport: CanvasViewport
}

/** 画布文件的结构版本（字段搬家时靠它分流 ✓） */
export const CANVAS_DOC_VERSION = 1

const STORAGE_KEY = 'drama-studio:canvas'
const MIN_SCALE = 0.2
const MAX_SCALE = 3

const nodes = ref<CanvasNode[]>([])
const edges = ref<CanvasEdge[]>([])
const viewport = ref<CanvasViewport>({ x: 0, y: 0, scale: 1 })
const selection = ref<string[]>([])
/**
 * 画布内容版本 ✓（单调递增，**只随 nodes/edges 变**）。
 *
 * 为什么**不**把 viewport 算进去：平移放大不改变画布内容，把视口算进版本会让
 * 「用户滚了下滚轮」被当成「画布被改了」，于是 AI 草案每次确认都报冲突 ✗。
 */
const revision = ref(0)
let hydrated = false
let revisionWatched = false
let timer: ReturnType<typeof setTimeout> | null = null

// 默认尺寸走单一真源（源 `NODE_DEFAULT_SIZE` ✓）：自己再拍一套会让同一张图被压扁 ✗
const DEFAULT_SIZE = DEFAULT_NODE_SIZE

/** 内置六类（**插件类型不在这里**，判定见 `isKnownNodeType`）✓ */
const NODE_TYPES = new Set<CanvasNodeType>(['text', 'image', 'video', 'audio', 'config', 'group'])

export const NODE_TYPE_LABEL: Record<CanvasNodeType, string> = {
  text: '文本',
  image: '图片',
  video: '视频',
  audio: '音频',
  config: '配置',
  group: '分组',
}

/** 类型标签 ✓：内置查表，插件类型查清单标题，都不中才回落到 type 本身（不 swallow ✗） */
export function nodeTypeLabel(type: CanvasAnyNodeType): string {
  const builtin = (NODE_TYPE_LABEL as Record<string, string>)[type]
  if (builtin) return builtin
  return pluginOf(type)?.title ?? String(type)
}

/** 是不是内置五类 ✓（尺寸记忆、默认尺寸表都只认这些） */
export function isBuiltinNodeType(type: unknown): type is CanvasNodeType {
  return typeof type === 'string' && NODE_TYPES.has(type as CanvasNodeType)
}

/** 能不能进画布 ✓：内置五类 + 已登记的插件类型 */
export function isKnownNodeType(type: unknown): type is CanvasAnyNodeType {
  return isBuiltinNodeType(type) || isPluginNodeType(type)
}

function uid(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

function persist() {
  if (timer) clearTimeout(timer)
  // 拖动会连续触发 ⇒ 防抖，别每帧写一次 localStorage ✗
  timer = setTimeout(() => {
    try {
      const doc: CanvasDoc = {
        nodes: nodes.value,
        edges: edges.value,
        viewport: viewport.value,
        revision: revision.value,
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(doc))
    } catch {
      // 隐私模式 / 配额满：画布照用，只是不落盘（不弹错打断创作 ✗）
    }
  }, 300)
}

function finite(value: unknown, fallback = 0) {
  const number = Number(value)
  return Number.isFinite(number) ? number : fallback
}

/**
 * 校验并清洗一份画布文档 ✓。
 *
 * ⭐ 两种策略**必须分开**：
 * * **导入**（`strict`）⇒ 有问题**整份拒**，现有画布一个字都不改 ✓
 *   （源 `canvas-import` 的口径："validate the complete archive before writing"；
 *   半截失败的画布比导不进来更难受 ✗）；
 * * **本地读档**（非 `strict`）⇒ **丢坏的、留好的** ✓（localStorage 坏一两个节点就清空整张画布，
 *   是拿用户刚摆好的稿子惩罚他 ✗）。
 */
/** config 卡的 ``values``：**只收字符串** ✓（数字/布尔一律 ``String()``，对象/null 丢掉 ✓） */
function normalizeValues(raw: unknown): Record<string, string> | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const out: Record<string, string> = {}
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof value === 'string') out[key] = value
    else if (typeof value === 'number' || typeof value === 'boolean') out[key] = String(value)
  }
  return Object.keys(out).length ? out : undefined
}

function normalizeDoc(raw: unknown, strict: boolean): CanvasDoc {
  const doc = (raw || {}) as Partial<CanvasDoc>
  if (!Array.isArray(doc.nodes) || !Array.isArray(doc.edges)) {
    if (strict) throw new Error('不是画布文件（缺 nodes / edges）')
    return { nodes: [], edges: [], viewport: { x: 0, y: 0, scale: 1 } }
  }

  const nodes: CanvasNode[] = []
  const ids = new Set<string>()
  for (const item of doc.nodes as unknown[]) {
    const node = (item || {}) as Partial<CanvasNode>
    const type = node.type as CanvasAnyNodeType
    const id = typeof node.id === 'string' ? node.id.trim() : ''
    const w = finite(node.w)
    const h = finite(node.h)
    if (!isKnownNodeType(type) || !id || w <= 0 || h <= 0) {
      if (strict) {
        // 未登记的插件类型要单独说清：一律报「节点不合法」会让人跑去查坐标/尺寸 ✗
        const why = isPluginNodeType(type) && !pluginOf(type) ? '插件未安装' : '节点不合法'
        throw new Error(`${why}：${id || '(无 id)'}`)
      }
      continue
    }
    if (ids.has(id)) {
      if (strict) throw new Error(`节点 id 重复：${id}`)
      continue
    }
    ids.add(id)
    nodes.push({
      id,
      type,
      title: typeof node.title === 'string' && node.title ? node.title : nodeTypeLabel(type),
      x: finite(node.x),
      y: finite(node.y),
      w,
      h,
      text: typeof node.text === 'string' ? node.text : undefined,
      url: typeof node.url === 'string' ? node.url : undefined,
      groupId: typeof node.groupId === 'string' && node.groupId ? node.groupId : undefined,
      workflowId: typeof node.workflowId === 'string' && node.workflowId ? node.workflowId : undefined,
      variant: typeof node.variant === 'string' && node.variant ? node.variant : undefined,
      // ⚠️ 这里是一张**白名单** ⇒ `CanvasNode` 每加一个字段都必须搬进来 ✓✗
      // （漏搬的后果不是报错，是**静默丢档**：`engineStage` 漏在这里时，用户选了 Music3、
      //   刷新一次就回落成清单第一个可用档（Z-Image），点生成跑出来的是一张图 ✓✗；
      //   而 `persist()` 会把清洗后的对象写回盘 ⇒ 丢是**永久**的 ✓✗。判据见
      //   `backend-py/tests/canvas_ops_test.py` 的「白名单覆盖接口字段」那条 ✓）
      runMode: node.runMode === 'comfyui' ? 'comfyui' : node.runMode === 'engine' ? 'engine' : undefined,
      engineStage: typeof node.engineStage === 'string' && node.engineStage ? node.engineStage : undefined,
      lastError: typeof node.lastError === 'string' && node.lastError ? node.lastError : undefined,
      values: normalizeValues(node.values),
    })
  }

  const edges: CanvasEdge[] = []
  const pairs = new Set<string>()
  for (const item of doc.edges as unknown[]) {
    const edge = (item || {}) as Partial<CanvasEdge>
    const from = typeof edge.from === 'string' ? edge.from : ''
    const to = typeof edge.to === 'string' ? edge.to : ''
    // 悬空连线（指向已删节点）一律不认 ✓ —— 留着会在渲染层画出「看不见的线头」✗
    if (!ids.has(from) || !ids.has(to) || from === to) {
      if (strict) throw new Error(`连线指向不存在的节点：${from || '(空)'} → ${to || '(空)'}`)
      continue
    }
    const pair = `${from}->${to}`
    if (pairs.has(pair)) continue
    pairs.add(pair)
    edges.push({ id: typeof edge.id === 'string' && edge.id ? edge.id : uid('edge'), from, to })
  }

  const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, finite(doc.viewport?.scale, 1) || 1))
  return { nodes, edges, viewport: { x: finite(doc.viewport?.x), y: finite(doc.viewport?.y), scale } }
}

/**
 * 版本号自增 ✓ —— **只注册一次**（模块级单例），并且**同步**生效。
 *
 * 两个细节都是必要的：
 *
 * - 只注册一次：`useCanvas()` 会被多个组件各调一次，若在 `useCanvas()` 里 watch，
 *   同一次改动会按调用次数重复 +1。单调递增本身不算错，但版本号跳得莫名其妙 ✓✗；
 * - 同步：CAS 靠它判「草案期间画布有没有被改」。若等防抖（300ms）再涨，
 *   AI 计划正好在这段窗口里落地就会拿旧版本号比对通过 —— 冲突漏判 ✗。
 */
function watchRevision() {
  if (revisionWatched) return
  revisionWatched = true
  watch([nodes, edges], () => {
    revision.value += 1
  }, { deep: true, flush: 'sync' })
}

function hydrate() {
  if (hydrated) return
  hydrated = true
  // 先挂 watch 再灌数据：灌进来的这批也会触发一次 +1，但那**不是**用户改动，
  // 所以紧接着用落盘的 revision 覆盖它 ✓（顺序不能反，否则 hydrate 完版本会多 1）
  watchRevision()
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return
    const parsed = JSON.parse(raw)
    const doc = normalizeDoc(parsed, false)
    nodes.value = doc.nodes
    edges.value = doc.edges
    viewport.value = doc.viewport
    const stored = Number(parsed?.revision)
    // 老档没有 revision ⇒ 从 0 起。**不猜**它被改过几次 ✓
    revision.value = Number.isFinite(stored) && stored > 0 ? Math.floor(stored) : 0
  } catch {
    // 坏档就当没有，不拦着进页面 ✗
  }
}

export function useCanvas() {
  hydrate()
  watch([nodes, edges, viewport], persist, { deep: true })

  /** 屏幕坐标 → 世界坐标（**先减平移、再除缩放** ✓） */
  function toWorld(sx: number, sy: number) {
    return { x: (sx - viewport.value.x) / viewport.value.scale, y: (sy - viewport.value.y) / viewport.value.scale }
  }

  function toScreen(wx: number, wy: number) {
    return { x: wx * viewport.value.scale + viewport.value.x, y: wy * viewport.value.scale + viewport.value.y }
  }

  function addNode(type: CanvasAnyNodeType, world: { x: number; y: number }) {
    // 尺寸 / 配置参数沿用「这类节点上次的设置」✓（批量摆素材时不用每张重新拉 ✗）
    // 插件节点**不记**：默认尺寸来自它的清单（`canvasPlugins.ts`），记进按 type 的尺寸表
    // 会让插件升级后仍用旧尺寸 ✗
    const builtin = isBuiltinNodeType(type)
    const spec = pluginOf(type)
    const remembered = builtin ? recallNodeSettings(type) : undefined
    const fallback = builtin
      ? DEFAULT_SIZE[type]
      : spec?.defaultSize ?? DEFAULT_SIZE.text
    const size = { w: remembered?.w || fallback.w, h: remembered?.h || fallback.h }
    const node: CanvasNode = {
      // id 里不留冒号（`html:render-xxx` 读起来像命名空间，排障时容易看错 ✗）
      id: uid(String(type).replace(':', '-')),
      type,
      title: nodeTypeLabel(type),
      // ⭐ 以落点为中心 ✓（左上角对齐会让双击建卡时卡片偏到右下，手感"点哪儿都不对" ✗）
      x: Math.round(world.x - size.w / 2),
      y: Math.round(world.y - size.h / 2),
      w: size.w,
      h: size.h,
      text: type === 'text' ? '' : type === 'config' ? remembered?.text ?? '' : undefined,
      // 插件默认键先落进 `values` ✓（渲染组件直接读得到，不必各自兜一遍默认值）
      values: spec ? pluginValuesOf(type, undefined) : undefined,
    }
    nodes.value = [...nodes.value, node]
    selection.value = [node.id]
    return node.id
  }

  function updateNode(id: string, patch: Partial<CanvasNode>) {
    nodes.value = nodes.value.map((n) => (n.id === id ? { ...n, ...patch } : n))
    // 记住尺寸（含素材自适应后的 ✓）与配置卡正文 ✓ —— 只在这两类字段真变了时才记
    const next = nodes.value.find((n) => n.id === id)
    // 只记内置五类：插件节点的默认尺寸来自它的清单（`canvasPlugins.ts`），
    // 记进「按 type 记住的尺寸」反而会让插件升级后仍用旧尺寸 ✗
    if (
      next &&
      isBuiltinNodeType(next.type) &&
      (patch.w !== undefined || patch.h !== undefined || (next.type === 'config' && patch.text !== undefined))
    ) {
      rememberNodeSettings(next.type, next)
    }
  }

  function removeNodes(ids: string[]) {
    const doomed = new Set(ids)
    nodes.value = nodes.value.filter((n) => !doomed.has(n.id))
    // 节点没了，挂在它身上的连线必须一起走（留着就是指向空处的幽灵线 ✗）
    edges.value = edges.value.filter((e) => !doomed.has(e.from) && !doomed.has(e.to))
    selection.value = selection.value.filter((id) => !doomed.has(id))
  }

  function moveNodes(ids: string[], dx: number, dy: number) {
    const moving = new Set(ids)
    nodes.value = nodes.value.map((n) => (moving.has(n.id) ? { ...n, x: n.x + dx, y: n.y + dy } : n))
  }

  function connect(from: string, to: string) {
    // 连线三不许（自连 / 指向分组 / 配置卡互连）都在**写入前**挡掉 ✓
    const pair = normalizeConnection(from, to, nodes.value)
    if (!pair) return false
    if (edges.value.some((e) => e.from === pair.from && e.to === pair.to)) return false
    edges.value = [...edges.value, { id: uid('edge'), from: pair.from, to: pair.to }]
    return true
  }

  /** 把选中的节点收进一个新分组（≥2 个才能成组 ✓，上留白更大以露出标题 ✓） */
  function groupSelected() {
    if (!canGroupSelectedNodes(selection.value, nodes.value)) return null
    const selected = new Set(selection.value)
    const members = nodes.value.filter((n) => n.type !== 'group' && selected.has(n.id))
    const rect = getGroupWrapRect(members)
    const groupId = uid('group')
    nodes.value = [
      ...nodes.value.map((n) => (selected.has(n.id) && n.type !== 'group' ? { ...n, groupId } : n)),
      { id: groupId, type: 'group', title: '分组', x: rect.x, y: rect.y, w: rect.w, h: rect.h },
    ]
    selection.value = [groupId]
    return groupId
  }

  /** 解组：清成员的 ``groupId`` 并删掉分组框 ✓ */
  function ungroupSelected() {
    if (!canUngroupSelectedNodes(selection.value, nodes.value)) return false
    const selected = new Set(selection.value)
    const groupIds = new Set(nodes.value.filter((n) => selected.has(n.id) && n.type === 'group').map((n) => n.id))
    nodes.value = nodes.value
      .filter((n) => !(selected.has(n.id) && n.type === 'group'))
      .map((n) => (groupIds.has(n.groupId || '') || (selected.has(n.id) && n.groupId) ? { ...n, groupId: undefined } : n))
    selection.value = []
    return true
  }

  /** 拖动结束时把落在分组上的节点收进去（**整体平移**，成员相对位置不变 ✓） */
  function snapIntoGroup(ids: string[], group: CanvasNode) {
    nodes.value = snapNodesIntoGroup(ids, nodes.value, group)
  }

  /** 聚焦到某个节点：节点占视口 ≤60%、只缩小不放大 ✓ */
  function focusNode(nodeId: string, size: { width: number; height: number }, frame: FocusFrame = {}) {
    const node = nodes.value.find((n) => n.id === nodeId)
    if (!node) return
    viewport.value = focusViewportForNode(node, size, frame)
  }

  function removeEdge(id: string) {
    edges.value = edges.value.filter((e) => e.id !== id)
  }

  /** 以**屏幕坐标**为锚缩放 ✓（不锚光标 ⇒ 视图会朝左上跑 ✗） */
  function zoomAt(factor: number, sx: number, sy: number) {
    const old = viewport.value.scale
    const next = Math.min(MAX_SCALE, Math.max(MIN_SCALE, old * factor))
    if (next === old) return
    const ratio = next / old
    viewport.value = {
      scale: next,
      x: sx - (sx - viewport.value.x) * ratio,
      y: sy - (sy - viewport.value.y) * ratio,
    }
  }

  function panBy(dx: number, dy: number) {
    viewport.value = { ...viewport.value, x: viewport.value.x + dx, y: viewport.value.y + dy }
  }

  function resetView() {
    viewport.value = { x: 0, y: 0, scale: 1 }
  }

  function setSelection(ids: string[]) {
    selection.value = ids
  }

  function exportDoc(): string {
    return JSON.stringify(
      { version: CANVAS_DOC_VERSION, nodes: nodes.value, edges: edges.value, viewport: viewport.value },
      null,
      2,
    )
  }

  function importDoc(text: string) {
    let parsed: unknown
    try {
      parsed = JSON.parse(text)
    } catch {
      throw new Error('不是合法的 JSON 文件')
    }
    // ⭐ 先整份校验、再落盘 ✓：宁可整份拒，也不留半坏的画布 ✗
    const doc = normalizeDoc(parsed, true)
    nodes.value = doc.nodes
    edges.value = doc.edges
    viewport.value = doc.viewport
    selection.value = []
  }

  function clearAll() {
    nodes.value = []
    edges.value = []
    selection.value = []
  }

  return {
    nodes,
    edges,
    viewport,
    selection,
    /** 内容版本 ✓（CAS 用；只随 nodes/edges 变，视口不算） */
    revision,
    addNode,
    updateNode,
    removeNodes,
    moveNodes,
    connect,
    removeEdge,
    groupSelected,
    ungroupSelected,
    snapIntoGroup,
    focusNode,
    zoomAt,
    panBy,
    resetView,
    setSelection,
    exportDoc,
    importDoc,
    clearAll,
    toWorld,
    toScreen,
  }
}

/**
 * 当前画布内容版本 ✓（不建 composable 也能读）。
 *
 * 给 `applyCanvasOps` 用：它要做 CAS，但不需要拿整个画布 API；
 * 走 `useCanvas()` 会在 utils 层制造循环依赖（`canvasOps` ↔ `useCanvas` 已经互引）✗。
 */
export function canvasRevision(): number {
  hydrate()
  return revision.value
}

