/**
 * 画布 AI —— 一句话 ⇒ 后端 Agent 编译出 op ⇒ 前端落盘。
 *
 * 三段职责分得很清楚，改哪段都不会牵动别处：
 *
 * | 段 | 谁做 | 在哪 |
 * | --- | --- | --- |
 * | 话 → op | 模型 + `canvas_ops.build_request` | `backend-py/app/services/canvas_agent.py` |
 * | op → 状态 | `applyCanvasOps` | `utils/canvasOps.ts` |
 * | op → 生成任务 | 调用方注入的 `onGenerate` | 画布页（生成域，要工作流与档位） |
 *
 * ## 两个必须如实交代的地方
 *
 * 1. **后端返回的 op 此刻还没落地** —— 别把 `reply.text` 当执行结果播给用户，
 *    要等 `applyCanvasOps` 回来再说「已落 N 条」✓
 * 2. **模型说的 ≠ 画布真做到的** —— 前端拒绝的 op（连线三不许、节点不存在）
 *    必须一起报出来，否则用户看到「AI 说连上了」但画布没线 ✗
 */
import { canvasAPI } from '../composables/useApi'
import { canvasRevision } from '../composables/useCanvas'
import { applyCanvasOps, canvasSnapshot, type ApplyResult, type CanvasOp, type RunAnswer } from './canvasOps'

export interface CanvasAgentOutcome {
  /** 模型的回话（说明它做了什么） */
  text: string
  /** 真正落到画布上的 op 条数 */
  applied: number
  /** 后端编译失败（模型参数不对，画布没动） */
  errors: string[]
  /** 前端落地被拒（连线被拒、节点不存在等） */
  rejected: string[]
  /** 不影响落地、但用户该知道的事（例如整批自检没跑成） */
  warnings: string[]
  /** 等着用户点一下「运行」的配置节点（没注入 `onGenerate` 时才会有） */
  pendingRun: string[]
  /** 这次真正被改到的节点 id ✓（对话面板据此「选中刚动的卡」并飞过去） */
  touched: string[]
  /** 模型调用过的工具（排查用） */
  toolCalls: Array<{ toolName: string; args: Record<string, unknown> }>
}

/**
 * 跑一次画布 AI ✓。
 *
 * `includeSnapshot` 默认 **true**：模型看不到画布就会凭空编节点 id（那是最常见的翻车方式）。
 * 只有画布是空的、或调用方明确知道不需要时才关掉。
 */
export async function runCanvasAgent(
  message: string,
  handlers: {
    /**
     * 可以返回 promise：卡片还没挂载时，它要等挂上跑一次才知道真答案 ✓。
     * 可以返回 `{ ok, reason }`：没提交成的人话理由会进「未生效」明细 ✓。
     */
    onGenerate?: (op: CanvasOp) => RunAnswer | Promise<RunAnswer>
    includeSnapshot?: boolean
    /** 用户点「停止」时中止 ✓（抛 `AbortError`，画布一条都不动） */
    signal?: AbortSignal
  } = {},
): Promise<CanvasAgentOutcome> {
  const text = (message || '').trim()
  if (!text) throw new Error('请先说一句要做什么')

  const snapshot = handlers.includeSnapshot === false ? undefined : canvasSnapshot()
  const reply = await canvasAPI.agent({ message: text, snapshot }, handlers.signal)

  const ops = (reply.ops || []) as CanvasOp[]

  /*
   * 落地前**整批自检** ✓（`POST /canvas/ops/validate`，这个接口一直有、此前没人调）。
   *
   * 为什么必须补这一步：`applyCanvasOps` 是逐条 try 的，遇到「指向不存在的节点」
   * 只跳过那一条 ⇒ 一批 6 条落 5 条，用户看到半套节点 + 一句看不懂的告警 ✗；
   * 后端是**整批判**，有问题就一条都不落、错误逐条列出来 ✓。
   * 它顺带把字段名归一（`node_type`→`nodeType`、坐标转数字），落地用它的产出更稳。
   */
  const warnings: string[] = []
  let verified = ops
  if (ops.length) {
    try {
      const check = await canvasAPI.validate({ ops, snapshot })
      if (check.ok && Array.isArray(check.ops)) {
        verified = check.ops as CanvasOp[]
      } else {
        // 整批拒：画布一条不动，把问题原样交给用户（比落一半好收拾 ✓）
        return {
          text: reply.text || '',
          applied: 0,
          errors: [...(reply.errors || []), ...(check.errors || [])],
          rejected: [],
          warnings,
          pendingRun: [],
          touched: [],
          toolCalls: reply.toolCalls || [],
        }
      }
    } catch (err: any) {
      /*
       * 自检自己挂了（后端 500 / 断网）不该把用户这一句话吞掉：
       * 退回「照原样落地」，前端逐条兜底仍在，但要如实说一声 ✓
       */
      warnings.push(`整批自检没跑成（${err?.message || err}）：这批按原样落地，可能有半批被拒`)
    }
  }

  const apply: ApplyResult = await applyCanvasOps(verified, { onGenerate: handlers.onGenerate })

  // `run_generation` 被跳过有两种原因：没给回调（⇒ 等用户手动点，这是**正常**的），
  // 或者缺 nodeId（⇒ 是真错误）。混在一起报会给用户一条看不懂的告警 ✗
  const pendingRun: string[] = []
  const rejected: string[] = []
  for (const item of apply.skipped) {
    if (!handlers.onGenerate && item.op.type === 'run_generation' && item.op.nodeId) {
      pendingRun.push(item.op.nodeId)
      continue
    }
    rejected.push(`${item.op.type}：${item.reason}`)
  }

  return {
    text: reply.text || '',
    applied: apply.applied,
    errors: reply.errors || [],
    rejected,
    warnings,
    pendingRun,
    touched: apply.touched,
    toolCalls: reply.toolCalls || [],
  }
}

// ---------------------------------------------------------------- 草案 → 确认（两段式）

/**
 * 草案 ✓ —— 一句话换来「AI 打算做什么」，**画布一动不动**。
 *
 * 与 `runCanvasAgent` 的取舍：
 *
 * | | 一句话 ⇒ 直接落 | 一句话 ⇒ 草案 ⇒ 确认 |
 * | --- | --- | --- |
 * | 步数 | 1 | 2 |
 * | 反悔机会 | 没有（落了才看见） | 有（先看 preflight 的数字） |
 * | 坐标 | 模型自己算 | 编译器算 |
 * | 适合 | 小动作（加一张图、连根线） | 一次建一批（开场三镜、整段分镜） |
 *
 * 所以两条都留着，别用一个替掉另一个 ✓。
 */
export interface CanvasPlanDraft {
  planId: string
  /** 用户原话（后端逐字保留，会回填进计划） */
  goal: string
  status: string
  /** true ⇒ 还没确认，画布没动 */
  requiresConfirmation: boolean
  /** true ⇒ 这句子之前确认过，同一份计划不会再下发一遍 */
  alreadyConfirmed?: boolean
  /** 草案时的画布版本（参考值；真正的 CAS 用确认时返回的那个） */
  expectedRevision: number | null
  preflight: {
    total: number
    creates: number
    links: number
    runs: number
    byOp: Record<string, number>
    byNodeType: Record<string, number>
  }
  layout: Record<string, any>
  warnings: string[]
  note?: string
}

/** 起草 ✓ —— 不入画布。`baseRevision` 记下当下画布版本，供确认时比对。 */
export async function draftCanvasPlan(
  message: string,
  handlers: { includeSnapshot?: boolean } = {},
): Promise<CanvasPlanDraft> {
  const text = (message || '').trim()
  if (!text) throw new Error('请先说一句要做什么')

  const snapshot = handlers.includeSnapshot === false ? undefined : canvasSnapshot()
  return canvasAPI.planAuto({
    message: text,
    snapshot,
    baseRevision: canvasRevision(),
  })
}

export interface CanvasPlanOutcome {
  /** 真正落到画布上的条数 */
  applied: number
  /** 后端在确认时重新发现的问题（画布变了导致计划对不上） */
  errors: string[]
  /** 前端落地被拒（连线被拒、节点不存在等） */
  rejected: string[]
  /** 没给 `onGenerate` 时挂起的生成节点（等用户手动点，这是正常的） */
  pendingRun: string[]
  /** 有值 ⇒ **一条都没落**：确认到落地之间画布被改过 */
  conflict?: { baseRevision: number; currentRevision: number }
  /** true ⇒ 这批 op 之前已经落过（重复确认 / 网络重试），本次什么也没做 */
  replayed: boolean
}

/**
 * 确认并落地 ✓ —— 唯一会写画布的那一步。
 *
 * 三个细节都是刻意的：
 *
 * 1. **确认时把当下快照再送一次**：后端会用它**重新编译**，所以「草案期间画布变了」
 *    不是靠拒绝处理，而是靠重算消化掉 —— 计划里要连的节点被删了，这一步就会报错 ✓；
 * 2. **`expectedRevision` 来自后端的确认响应**（= 重编译所依据的版本），不是草案时那个。
 *    CAS 拦的是「确认到落地这几十毫秒里画布又被改」这个窗口 ✓；
 * 3. **`applyId = planId`**：重复确认拿到的是后端缓存的同一批 op，前端靠幂等簿确保
 *    不会建第二套节点、更不会重复起生成任务 ✓。
 */
export async function applyCanvasPlan(
  planId: string,
  handlers: {
    onGenerate?: (op: CanvasOp) => boolean | void | Promise<boolean>
    includeSnapshot?: boolean
  } = {},
): Promise<CanvasPlanOutcome> {
  const snapshot = handlers.includeSnapshot === false ? undefined : canvasSnapshot()
  const confirmed = await canvasAPI.confirmPlan(planId, {
    snapshot,
    baseRevision: canvasRevision(),
  })

  const apply = await applyCanvasOps(confirmed.ops as CanvasOp[], {
    onGenerate: handlers.onGenerate,
    applyId: planId,
    baseRevision: confirmed.expectedRevision ?? undefined,
  })

  const pendingRun: string[] = []
  const rejected: string[] = []
  for (const item of apply.skipped) {
    if (!handlers.onGenerate && item.op.type === 'run_generation' && item.op.nodeId) {
      pendingRun.push(item.op.nodeId)
      continue
    }
    rejected.push(`${item.op.type}：${item.reason}`)
  }

  return {
    applied: apply.applied,
    errors: confirmed.warnings || [],
    rejected,
    pendingRun,
    conflict: apply.conflict,
    replayed: apply.replayed === true,
  }
}
