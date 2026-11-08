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
 *    但**必须等它的答复**（`await`，答复可以是个 promise）✓：卡片可能还没挂载，
 *    同步回答不了「提交了没有」；不等就只能乐观记账 ⇒ 面板谎报「已提交」✗。
 *    这也是 `applyCanvasOps` 是 `async` 的唯一理由。
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

/** CAS 冲突 ✓ —— 这批 op 是按**旧画布**算出来的，整批没落地。 */
export interface ApplyConflict {
  /** 调用方以为画布停在哪一版 */
  baseRevision: number
  /** 画布实际停在哪一版 */
  currentRevision: number
}

export interface ApplyResult {
  applied: number
  skipped: SkippedOp[]
  /**
   * 这批 op 真正碰到的节点 id（新建/修改/删除/连线两端/选中）✓。
   *
   * 为什么要交出来：AI 一次落 5 条 op 之后，用户**找不到**是哪几张卡动了 ——
   * 画布可能早已滚到别处、卡片又长得都一样。有了这个列表，界面才能
   * 「选中刚改的那几张」或「飞过去看一眼」，否则只能让用户自己扫一遍 ✗。
   */
  touched: string[]
  /** 有值 ⇒ **一条都没落**，调用方应提示用户重看计划 ✗ */
  conflict?: ApplyConflict
  /** 有值 ⇒ 这批 op 之前已经落过，本次是**重放**，`applied` 记的是上次的条数 ✓ */
  replayed?: boolean
}

const MIN_SCALE = 0.2
const MAX_SCALE = 3

/** 幂等簿的上限 ✓ —— 只留最近若干批，防止长会话把内存吃光 */
const APPLIED_LIMIT = 50

/**
 * 已落地的批次 ✓（`applyId` ⇒ 结果）。
 *
 * 为什么需要它：一份被确认的计划可能因为「用户手滑点两下确认」或「网络重试」
 * 而送到这里两次。没有这张簿，第二次会**再建一套节点**（id 相同则覆盖，看着没事；
 * 但 `connect_nodes`/`select_nodes` 会重复计数，`run_generation` 更是会重复起任务 ✗）。
 */
const appliedBatches = new Map<string, ApplyResult>()

/** 清空幂等簿 ✓（测试用；正常流程不必调） */
export function resetAppliedBatches(): void {
  appliedBatches.clear()
}

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
 *
 * 但有两件事是**整批**判的，因为它们不是「某一条 op 坏了」，而是「这批 op 不该落」：
 *
 * - ``applyId``：同 id 的批次只落一次，重放直接返回上次结果（``replayed: true``）✓；
 * - ``baseRevision``：与当前画布版本不符 ⇒ 整批拒绝，返回 ``conflict`` ✓。
 *   典型场景：AI 出一份计划要 30 秒，用户等不及自己拖了两个节点 ——
 *   那批 op 是按旧画布算的坐标与连线，硬落就是错位加断线 ✗。
 */
/**
 * 对话里「代你点生成」时 op 带下来的两格覆盖 ✓（源同口径：`generateNodeRef(nodeId, mode, prompt)`）。
 *
 * ⚠️ 为什么不是"只按 nodeId 点一下"：模型经常先 `canvas_create_text_node` 写新提示词、
 * 再 `canvas_run_generation` 指向那张配置卡 —— 只认 nodeId 就等于把它刚写的那句话丢掉 ✗。
 */
/**
 * 「跑完了，这是成品」这一跳的载荷 ✓（卡片 emit → 画布板 → 画布页）。
 *
 * `sourceId` 是后加的，不是可有可无的装饰：画布页要据此**贴边落位并连线**，
 * 否则一份成品只知道自己是「图片」，不知道自己是哪张卡跑出来的 ✗ ——
 * 连了线的成品，下一轮对话 agent 读快照时才追得回来 ✓。
 */
export interface GeneratedPayload {
  kind: 'image' | 'video' | 'audio'
  url: string
  runId: number | null
  /** 出品卡 id ✓（缺省时退回视口中心：工具栏直接跑工作流那条路） */
  sourceId?: string
}

export interface RunOverride {
  /** 生成类别（`text|image|video|audio`）✓ —— 与卡上档位类别不符就如实拒绝，不静默换档 ✗ */
  mode?: string
  /** 本次提示词 ✓ —— 优先于卡上正文；仍是引用串时由画布解析成正文再送 */
  prompt?: string
}

export async function applyCanvasOps(
  ops: CanvasOp[],
  handlers: {
    /**
     * 返回 `false` ⇒ 明确"没提交成"（缺档位/缺件/缺首帧），执行器会记 skipped ✓。
     *
     * ⭐ 也可以返回一个 **promise**：同一批 op 刚建出来的卡此刻还没挂载，
     * 它没法同步回答「提交了没有」，只能等挂上后跑一次再如实回答 ✓ ——
     * 这正是本函数改成 `async` 的唯一理由（见下面 `run_generation` 分支）。
     */
    onGenerate?: (op: CanvasOp) => boolean | void | Promise<boolean>
    /** 幂等键 ✓（计划 id 天然合适） */
    applyId?: string
    /** 这批 op 是基于哪一版画布算的 ✓（不给 = 不校验，兼容老调用方） */
    baseRevision?: number
  } = {},
): Promise<ApplyResult> {
  const canvas = useCanvas()
  const result: ApplyResult = { applied: 0, skipped: [], touched: [] }
  /** 同一次落地里可能反复碰到同一张卡（先建再连再选中）⇒ 去重但保序 ✓ */
  const touched = new Set<string>()
  const touch = (id?: string | null) => {
    if (id) touched.add(id)
  }

  if (handlers.applyId) {
    const previous = appliedBatches.get(handlers.applyId)
    if (previous) return { ...previous, replayed: true }
  }

  if (handlers.baseRevision !== undefined) {
    const current = canvas.revision.value
    if (current !== handlers.baseRevision) {
      // 一条都不落：半批落地比整批不落更难收拾 ✓
      return {
        applied: 0,
        skipped: ops.map((op) => ({ op, reason: '画布已被修改，这批动作已过期' })),
        touched: [],
        conflict: { baseRevision: handlers.baseRevision, currentRevision: current },
      }
    }
  }

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
        touch(node.id)
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
        touch(op.id)
        result.applied += 1
      } else if (op.type === 'delete_nodes') {
        const ids = (op.ids || []).filter((id) => canvas.nodes.value.some((item) => item.id === id))
        if (!ids.length) {
          result.skipped.push({ op, reason: 'ids 为空或都不存在' })
          continue
        }
        // 删掉的 id 也记：撤销后要「跳回去看它原本在哪」，界面得知道是哪一个 ✗ 记了才发现不在了
        ids.forEach(touch)
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
        if (rejected.length < pairs.length) {
          // 只记真连上的那两端（被拒的连不上，别让界面去选中它们）
          for (const pair of pairs) {
            if (rejected.some((item) => item === `${pair.from} → ${pair.to}`)) continue
            touch(pair.from)
            touch(pair.to)
          }
          result.applied += 1
        }
      } else if (op.type === 'select_nodes') {
        canvas.setSelection(op.ids || [])
        ;(op.ids || []).forEach(touch)
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
        // 先定位再提交：闸没过也要把卡选中，用户才知道是哪张卡没跑起来 ✓
        touch(op.nodeId)
        /*
         * ⭐ 必须 `await` ✓✗ —— 这里的 promise 不是风格问题，是**谎报的根源**：
         * 同一批 op 刚建的卡此刻还没挂载（Vue 渲染是异步的），画布板只能先给一个
         * promise「等挂上后跑一次再答复」。不 await 就等于把「排队」当「已提交」，
         * 闸没过时对话面板会写「已提交生成 1」而实际一条请求都没发 ✗
         * （实测踩过：model 让 zimage 卡跑 video，卡片如实拒绝，回执却报已提交）。
         */
        const verdict = await handlers.onGenerate(op)
        if (verdict === false) {
          // 回调说"没提交成"（缺档位/缺件/缺首帧）⇒ 不能记成已落地 ✗
          result.skipped.push({ op, reason: '生成没提交成：看卡片上的缺项提示' })
          continue
        }
        result.applied += 1
      } else {
        result.skipped.push({ op, reason: `未知 op 类型：${String((op as { type: string }).type)}` })
      }
    } catch (error) {
      result.skipped.push({ op, reason: error instanceof Error ? error.message : String(error) })
    }
  }

  result.touched = [...touched]

  if (handlers.applyId) {
    // 记的是**这一次**的结果（含 skipped）—— 重放时如实返回「上次落了 N 条、跳了 M 条」，
    // 而不是假装又成功了一次 ✓
    appliedBatches.set(handlers.applyId, { applied: result.applied, touched: result.touched, skipped: result.skipped })
    while (appliedBatches.size > APPLIED_LIMIT) {
      const oldest = appliedBatches.keys().next()
      if (oldest.done) break
      appliedBatches.delete(oldest.value)
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
