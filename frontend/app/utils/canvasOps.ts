/**
 * 画布 op 执行器 —— 把后端编译好的 op 列表落到 `useCanvas` 的状态上。
 *
 * 与 `backend-py/app/services/canvas_ops.py` **成对存在**：那边只产出 op（纯函数、可单测），
 * 这边只落 op（要写 `ref` 才能触发 Vue 更新）。中间那份 JSON 就是两边的契约。
 *
 * ## 三处必须自己判的地方（后端判不了）
 *
 * 1. **`add_node` 的 x/y 是左上角**，而 `useCanvas.addNode()` 收的是**落点中心**
 *    （它内部减 w/2、h/2）⇒ 走 `addNode` 会让每个节点偏到右下 ✗，这里直接写 `nodes.value`。
 * 2. **连线会被静默拒**：`connect()` 挡自连/指向分组/配置互卡互连，返回 `false`。
 *    不检查返回值 = 用户看到「AI 说连上了」但画布没线 ✗ ⇒ 每条都要记 skipped。
 * 3. **`run_generation` 不在这里跑**：那是生成域（要带工作流、档位、参考图），
 *    由调用方注入 `onGenerate` 回调 ✓ —— 执行器保持纯粹，不偷偷起任务。
 */
import type { CanvasAnyNodeType, CanvasDoc, CanvasEdge, CanvasNode, CanvasViewport } from '../composables/useCanvas'
import { isBuiltinNodeType, useCanvas } from '../composables/useCanvas'
import { isPluginNodeType, pluginOf, pluginValuesOf } from './canvasPlugins'
import { DEFAULT_NODE_SIZE } from './canvasNodeSize'

export type CanvasOpType =
  | 'add_node'
  | 'update_node'
  | 'delete_nodes'
  | 'delete_edges'
  | 'connect_nodes'
  | 'select_nodes'
  | 'set_viewport'
  | 'run_generation'

export interface CanvasOp {
  type: CanvasOpType
  nodeType?: CanvasAnyNodeType
  id?: string
  title?: string
  text?: string
  url?: string
  workflowId?: string
  variant?: string
  runMode?: 'engine' | 'comfyui'
  engineStage?: string
  values?: Record<string, string>
  x?: number
  y?: number
  w?: number
  h?: number
  patch?: Partial<CanvasNode>
  ids?: string[]
  connections?: Array<{ from: string; to: string }>
  viewport?: CanvasViewport
  nodeId?: string
  mode?: 'text' | 'image' | 'video' | 'audio'
  prompt?: string
}

export interface SkippedOp {
  op: CanvasOp
  reason: string
}

export interface ApplyResult {
  applied: number
  skipped: SkippedOp[]
}

const MIN_SCALE = 0.2
const MAX_SCALE = 3

/** 与 `useCanvas` 的 `uid()` 同形（时间戳 + 随机，避免与既有 id 撞车） */
function uid(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

function finite(value: unknown, fallback = 0): number {
  const number = Number(value)
  return Number.isFinite(number) ? number : fallback
}

/**
 * 应用一批 op ✓。**不做整批回滚** —— op 之间常有先后依赖（先建节点再连线），
 * 逐条落地 + 逐条记 `skipped` 比整批放弃对用户更有用；要撤销请用画布的撤销栈。
 */
export function applyCanvasOps(
  ops: CanvasOp[],
  handlers: { onGenerate?: (op: CanvasOp) => void } = {},
): ApplyResult {
  const canvas = useCanvas()
  const result: ApplyResult = { applied: 0, skipped: [] }

  for (const op of ops) {
    try {
      if (op.type === 'add_node') {
        const nodeType = op.nodeType
        // 内置五类 + 插件类型（`<pluginId>:<name>`）✓。插件类型不要求已登记：
        // 没登记 ⇒ 没有默认尺寸/默认值，退回文本规格，卡片会显示「未安装的插件」而不是消失 ✓
        if (!nodeType || (!isBuiltinNodeType(nodeType) && !isPluginNodeType(nodeType))) {
          result.skipped.push({ op, reason: `节点类型不认识：${String(nodeType)}` })
          continue
        }
        const spec = pluginOf(nodeType)
        const base = isBuiltinNodeType(nodeType)
          ? DEFAULT_NODE_SIZE[nodeType]
          : spec?.defaultSize ?? DEFAULT_NODE_SIZE.text
        const node: CanvasNode = {
          id: op.id || uid(String(nodeType).replace(':', '-')),
          type: nodeType,
          title: op.title || spec?.title || '',
          x: finite(op.x),
          y: finite(op.y),
          w: op.w ?? base.w,
          h: op.h ?? base.h,
        }
        if (op.text !== undefined) node.text = op.text
        if (op.url !== undefined) node.url = op.url
        if (op.workflowId !== undefined) node.workflowId = op.workflowId
        if (op.variant !== undefined) node.variant = op.variant
        if (op.runMode !== undefined) node.runMode = op.runMode
        if (op.engineStage !== undefined) node.engineStage = op.engineStage
        // 插件节点的默认键先补齐 ✓（同 `useCanvas.addNode`）—— 渲染组件只读 `values`
        if (op.values !== undefined) node.values = op.values
        else if (spec) node.values = pluginValuesOf(nodeType, undefined)
        // 同 id 已存在 ⇒ 覆盖（重放同一套 op 不该长出两套节点 ✓）
        const exists = canvas.nodes.value.some((item) => item.id === node.id)
        canvas.nodes.value = exists
          ? canvas.nodes.value.map((item) => (item.id === node.id ? node : item))
          : [...canvas.nodes.value, node]
        result.applied += 1
      } else if (op.type === 'update_node') {
        if (!op.id) {
          result.skipped.push({ op, reason: 'update_node 缺 id' })
          continue
        }
        const known = canvas.nodes.value.some((item) => item.id === op.id)
        if (!known) {
          // 静默忽略 = 用户以为改了其实没改 ✗
          result.skipped.push({ op, reason: `节点不存在：${op.id}` })
          continue
        }
        canvas.updateNode(op.id, op.patch || {})
        result.applied += 1
      } else if (op.type === 'delete_nodes') {
        const ids = (op.ids || []).filter((id) => canvas.nodes.value.some((item) => item.id === id))
        if (!ids.length) {
          result.skipped.push({ op, reason: 'ids 为空或都不存在' })
          continue
        }
        canvas.removeNodes(ids)
        result.applied += 1
      } else if (op.type === 'delete_edges') {
        const known = new Set(canvas.edges.value.map((edge) => edge.id))
        const ids = (op.ids || []).filter((id) => known.has(id))
        if (!ids.length) {
          result.skipped.push({ op, reason: '连线 id 都不存在' })
          continue
        }
        for (const id of ids) canvas.removeEdge(id)
        result.applied += 1
      } else if (op.type === 'connect_nodes') {
        const pairs = op.connections || []
        if (!pairs.length) {
          result.skipped.push({ op, reason: 'connections 为空' })
          continue
        }
        const rejected: string[] = []
        for (const pair of pairs) {
          // ⭐ 必须看返回值：false = 被「连线三不许」拒了
          if (!canvas.connect(pair.from, pair.to)) rejected.push(`${pair.from} → ${pair.to}`)
        }
        if (rejected.length) result.skipped.push({ op, reason: `连线被拒：${rejected.join('、')}` })
        if (rejected.length < pairs.length) result.applied += 1
      } else if (op.type === 'select_nodes') {
        canvas.setSelection(op.ids || [])
        result.applied += 1
      } else if (op.type === 'set_viewport') {
        const viewport = op.viewport
        if (!viewport) {
          result.skipped.push({ op, reason: '缺 viewport' })
          continue
        }
        canvas.viewport.value = {
          x: finite(viewport.x),
          y: finite(viewport.y),
          scale: Math.min(MAX_SCALE, Math.max(MIN_SCALE, finite(viewport.scale, 1) || 1)),
        }
        result.applied += 1
      } else if (op.type === 'run_generation') {
        if (!op.nodeId) {
          result.skipped.push({ op, reason: 'run_generation 缺 nodeId' })
          continue
        }
        if (!handlers.onGenerate) {
          // 没有生成回调时**如实说没跑**，不能假装跑了 ✗
          result.skipped.push({ op, reason: '未提供生成回调（本次只落画布，未开跑）' })
          continue
        }
        handlers.onGenerate(op)
        result.applied += 1
      } else {
        result.skipped.push({ op, reason: `未知 op 类型：${String((op as { type: string }).type)}` })
      }
    } catch (error) {
      result.skipped.push({ op, reason: error instanceof Error ? error.message : String(error) })
    }
  }

  return result
}

/** 画当前快照（发给后端做坐标推算 / 让模型理解布局）✓ */
export function canvasSnapshot(): {
  nodes: CanvasNode[]
  edges: CanvasEdge[]
  viewport: CanvasViewport
  selection: string[]
} {
  const canvas = useCanvas()
  return {
    nodes: canvas.nodes.value,
    edges: canvas.edges.value,
    viewport: canvas.viewport.value,
    selection: canvas.selection.value,
  }
}

/** 导入一份画布文档（走 `importDoc` 的 strict 口径：坏档整份拒 ✓） */
export function importCanvasDoc(doc: CanvasDoc | string): void {
  useCanvas().importDoc(typeof doc === 'string' ? doc : JSON.stringify(doc))
}
