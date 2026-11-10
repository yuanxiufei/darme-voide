/**
 * 本仓**自研引擎**的运行编排（画布 → `POST /engine/generate` → 轮询 `POST/GET /engine/tasks/{id}`）。
 *
 * 与 `useComfyRun` 的分工 ✗✗：
 * * `useComfyRun` = **外部 ComfyUI 工作流**（8765 上游 + 注册表 `workflowId` + SSE）✓；
 * * 本模块 = **进程内引擎**（`stage` + 参数 + 权重自动装配 + 事件轮询）✓ —— 不依赖任何外部服务 ✓。
 *
 * ## 四件必须做对的事（写错都是"看着在跑、其实没接上"）
 *
 * 1. ⭐ **进度只能读「事件」** ✓：引擎不报 `percent` ✓，只报 `{stage, step, total}` ✓ ——
 *    自己按阶段给区间（plan→write）再按 step/total 在区间内插值 ✓；
 *    没有 `sample` 事件时**如实按阶段走**（不画一个永远 0% 的环 ✗）。
 * 2. ⭐ **产物是 URL 不是路径** ✓：后端 `primaryPath` 是**本机绝对路径** 🚫，
 *    能直接给 `<img>` 的是 `primaryUrl`（`/static/engine/x.png` ✓）。
 * 3. ⭐ **终态判据是 `terminal`** ✓（`done` / `failed` / `cancelled`）——
 *    别用 `status !== 'running'` ✗：`queued` 也不是运行中 ✓✗，那样第一拍就停轮询 ✓✗。
 * 4. ⭐ **轮询失败≠任务失败** ✓：一次网络抖动就把卡片标成"失败"，用户会重跑一次几十秒的活 ✗。
 *
 * 状态挂在**模块级单例**（与 `useCanvas` / `useComfyRun` 同口径 ✓）。
 */
import { computed, ref, type ComputedRef } from 'vue'
import { api } from './useApi'

/** 轮询间隔（引擎每步都会 append 事件；1.2 秒够看出"在动"又不打爆后端 ✓） */
const POLL_MS = 1200

/** 档位 = 一次 `ensure_loaded` 的装配口径 ✓（后端 `/engine/stages` 的 `stage` 值 ✓） */
export interface EngineStage {
  stage: string
  label: string
  /** 装配会挑到的件（**文件名** ✓，后端不吐绝对路径 ✗） */
  components: Record<string, string>
  /** 件齐不齐 ✓（清单层面） */
  ready: boolean
  /** 装不装得下 ✓（显存层面）—— `null` = 没测出来 ✓，**别当 false** ✗ */
  fits: boolean | null
  estimatedGiB: number | null
  requiredGiB: number | null
  /** 缺哪些件 ✓（人话 ✓） */
  missing: string[]
  /** 自动装配的说明 ✓（挑了哪些件 / 词表从哪来 ✓） */
  note: string | null
}

export type EngineStatus = 'idle' | 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export interface EngineRunState {
  taskId: string | null
  status: EngineStatus
  percent: number
  step: number
  total: number
  /** 当前阶段（`plan` / `encode` / `sample` / `decode` … ✓ 展示时过 `phaseLabel` ✓） */
  phase: string
  error: string | null
  /** 可直接给 `<img>` / `<video>` 的地址 ✓（来自回执 `primaryUrl` ✓） */
  primaryUrl: string | null
  primaryPath: string | null
  elapsedMs: number | null
  /** ⚠️ 跑的是干跑后端 / 参考件未训练 ⇒ 界面必须打标 ✗（别让人当成成品 ✓） */
  synthetic: boolean
  /** 提交回执的 `autoNote` ✓（这一档自动挑了哪些件、缺什么 ✓） */
  note: string | null
  /**
   * 引擎对这次请求的**说明** ✓（任务结果里的 `plan.warnings` ✓）。
   *
   * ⚠️ 不取就等于白算 ✗✗：这里面装的正是「尺寸被吸附成多少」「参考图被**忽略**」
   * 「这一档给不了 24fps ⇒ 换成 16 并说明」「denoise<1 ⇒ 实跑步数比名义少」这类
   * **用户必须知道、界面别处看不到**的话 ✓。后端刻意不把它们写成 error（都不是失败 ✓）
   * ⇒ 前端不显示，就永远没人看见 ✓✗。
   */
  warnings: string[]
  /** 已收到的事件条数 ✓（"在动"的最直接证据 ✓） */
  eventCount: number
  /**
   * 这条状态是**刷新后恢复**出来的、且恢复时**已经跑完** ✓。
   *
   * ⚠️ 这种状态**不许**再走"产物回卡" ✗✗：产物在刷新前就已经加进画布并持久化了 ✓，
   * 再 emit 一次就会凭空多出一张一模一样的卡 ✓✗（视频档一次要几分钟，最容易踩 ✓）。
   */
  restored: boolean
}

const PHASE_LABEL: Record<string, string> = {
  plan: '计划', encode: '编码', init: '起点', condition: '条件',
  sample: '采样', refine: '精修', decode: '解码', write: '写出',
}

/**
 * 阶段 → 进度区间 ✓（**只用于显示** ✗：引擎的真实步数在 `sample` 事件里 ✓）。
 *
 * 为什么按区间而不是 `step/total`：一次出图 8 步采样只占全程 1/3 ✓，
 * 把 8 步直接当 100% 会让进度条在解码那段"卡住不动"✗。
 */
const PHASE_RANGE: Record<string, [number, number]> = {
  plan: [0, 3], encode: [3, 15], init: [15, 20], condition: [20, 25],
  sample: [25, 88], refine: [88, 92], decode: [92, 98], write: [98, 99],
}

const TERMINAL: EngineStatus[] = ['succeeded', 'failed', 'cancelled']

const stages = ref<EngineStage[]>([])
const stagesLoading = ref(false)
const stagesError = ref<string | null>(null)
const states = ref<Record<string, EngineRunState>>({})
const timers = new Map<string, number>()
let loadPromise: Promise<void> | null = null

function idleState(): EngineRunState {
  return {
    taskId: null, status: 'idle', percent: 0, step: 0, total: 0, phase: '',
    error: null, primaryUrl: null, primaryPath: null, elapsedMs: null,
    synthetic: false, note: null, warnings: [], eventCount: 0, restored: false,
  }
}

/**
 * 任务号**本地留档** ✓ —— 视频档一次跑几分钟 ✓，刷新一下界面就把 `taskId` 丢了 ✓✗：
 * 引擎还在跑、进度还在涨，画布却回到"未生成" ✓✗（用户只能再点一次，白等一遍）。
 *
 * 存的是 `{ [nodeId]: taskId }` ✓（按卡分开 ✓）；写失败（隐私模式/配额满）只丢"刷新可恢复" ✓，
 * **不影响本次生成** ✗。
 */
const STORE_KEY = 'darme:canvas:engine-tasks'

function readRegistry(): Record<string, string> {
  if (typeof localStorage === 'undefined') return {}
  try {
    const parsed = JSON.parse(localStorage.getItem(STORE_KEY) || '{}')
    if (!parsed || typeof parsed !== 'object') return {}
    return Object.fromEntries(
      Object.entries(parsed).filter(([, value]) => typeof value === 'string' && value),
    ) as Record<string, string>
  } catch {
    return {}
  }
}

function writeRegistry(map: Record<string, string>) {
  if (typeof localStorage === 'undefined') return
  try {
    if (Object.keys(map).length) localStorage.setItem(STORE_KEY, JSON.stringify(map))
    else localStorage.removeItem(STORE_KEY)
  } catch { /* 隐私模式 / 配额满 ⇒ 只是失去"刷新可恢复"，不打断生成 ✓ */ }
}

/** 记住 / 忘掉某张卡的任务号 ✓（`null` ⇒ 忘掉 ✓） */
function rememberTask(key: string, taskId: string | null) {
  const map = readRegistry()
  if (taskId) map[key] = taskId
  else delete map[key]
  writeRegistry(map)
}

function isTerminal(status: EngineStatus): boolean {
  return TERMINAL.includes(status)
}

function patch(key: string, next: Partial<EngineRunState>) {
  states.value = {
    ...states.value,
    [key]: { ...(states.value[key] || idleState()), ...next },
  }
}

function stop(key: string) {
  const timer = timers.get(key)
  if (timer !== undefined) {
    clearTimeout(timer)
    timers.delete(key)
  }
}

/** 后端状态名 → 前端状态名 ✓（后端叫 `done` ✓，前端叫 `succeeded` ✓ —— 两套名字只在这一处换 ✓） */
function mapStatus(raw: unknown): EngineStatus {
  const text = String(raw || '').toLowerCase()
  if (text === 'done' || text === 'succeeded') return 'succeeded'
  if (text === 'failed') return 'failed'
  if (text === 'cancelled' || text === 'canceled') return 'cancelled'
  if (text === 'running') return 'running'
  if (text === 'queued') return 'queued'
  return 'idle'
}

interface EngineEvent {
  stage?: string
  step?: number
  total?: number
  note?: string
}

/** 事件 → 进度 ✓（最后一个**带 stage** 的事件说了算 ✓） */
function progressOf(events: EngineEvent[], status: EngineStatus): {
  percent: number; step: number; total: number; phase: string
} {
  if (status === 'succeeded') {
    return { percent: 100, step: 0, total: 0, phase: 'write' }
  }
  let last: EngineEvent | null = null
  for (const event of events) {
    if (event && typeof event.stage === 'string') last = event
  }
  if (!last || !last.stage) return { percent: 0, step: 0, total: 0, phase: '' }
  const [lo, hi] = PHASE_RANGE[last.stage] || [0, 0]
  const total = Number(last.total) || 0
  const step = Number(last.step) || 0
  const inner = total > 0 && step > 0 ? Math.min(1, step / total) : 0
  return {
    percent: Math.round((lo + (hi - lo) * inner) * 10) / 10,
    step,
    total,
    phase: last.stage,
  }
}

/**
 * 拉一次任务详情 ✓ ⇒ 三种结局**必须分开** ✗✗：
 *
 * * `ok` ＝ 拿到了（不管跑没跑完 ✓）；
 * * `missing` ＝ 后端说**没有这个任务** ✓（`400` ✓：引擎重启过 / 被清过 ✓）——
 *   这时继续轮询就是**永远转圈** ✓✗，必须停并如实报 ✓；
 * * `error` ＝ 网络/超时 ✓ —— **不算任务失败** ✓（见模块头第 4 条 ✓），下一拍再问 ✓。
 */
async function syncDetail(key: string, taskId: string): Promise<'ok' | 'missing' | 'error'> {
  try {
    const task: any = await api.get(`/engine/tasks/${taskId}`)
    const status = mapStatus(task?.status)
    const events: EngineEvent[] = Array.isArray(task?.events) ? task.events : []
    const progress = progressOf(events, status)
    patch(key, {
      status,
      ...progress,
      error: task?.error || null,
      primaryUrl: task?.primaryUrl || null,
      primaryPath: task?.primaryPath || null,
      elapsedMs: task?.elapsedMs ?? null,
      synthetic: Boolean(task?.result?.synthetic),
      // ⚠️ 路径是 `result.plan.warnings` ✓（`RunResult.to_dict()` 把整个 plan 带上 ✓）——
      //    不是 `result.warnings` ✗、也不是 `task.warnings` ✗（都没有那一层 ✓）。
      //    只在**有内容**时覆盖 ✓：中途一次拿不到不该把已经显示的说明擦掉 ✗。
      //: ⭐⭐ 再并上 `result.alerts` ✓✗（2026-11-09 补 ✗）：那是**跑完才知道**的结论 ✓
      //:   （「音轨几乎是静音 ✗」「mp4 里音轨被截掉 0.6 秒 ✗」「落盘帧数没核对过 ✗」…）✓ ——
      //:   后端同时把它写进 `result.note` ✓，而本文件此前**一个字都不取** ✗ ⇒ 界面上
      //:   一句都看不到 ✓✗（后端说了、用户没看见 ＝ 白说 ✓）。
      //:   两路合成同一个列表 ✓：`warnings` 的渲染已有「⚠️ 前缀 ＋ 只显示前两条 ＋ 悬停看全量」✓。
      ...(() => {
        const plan = Array.isArray(task?.result?.plan?.warnings)
          ? task.result.plan.warnings.map((item: unknown) => String(item))
          : null
        const alerts = Array.isArray(task?.result?.alerts)
          ? task.result.alerts.map((item: unknown) => String(item))
          : []
        if (!plan && !alerts.length) return {}
        return { warnings: [...(plan ?? []), ...alerts] }
      })(),
      eventCount: Number(task?.eventCount) || events.length,
    })
    return 'ok'
  } catch (err: any) {
    // 后端对"没有这个任务"回 **400** ✓（`bad_request` ✓），与网络错误分得开 ✓
    if (Number(err?.status) === 400) return 'missing'
    return 'error'
  }
}

function poll(key: string, taskId: string) {
  const tick = async () => {
    const current = states.value[key]
    if (!current || current.taskId !== taskId) return
    if (isTerminal(current.status)) return
    const outcome = await syncDetail(key, taskId)
    const after = states.value[key]
    if (!after || after.taskId !== taskId) return
    if (outcome === 'missing') {
      // ⚠️ 任务没了就别再转 ✗：如实报"后端不认识这个任务"，并**忘掉**它 ✓
      patch(key, {
        status: 'failed',
        error: '后端已没有这个任务（引擎重启或任务被清理）⇒ 请重新生成',
      })
      rememberTask(key, null)
      return
    }
    if (isTerminal(after.status)) {
      if (outcome === 'ok') rememberTask(key, taskId)  // 留档 ✓：刷新后还能看到"完成/失败"的结论 ✓
      return
    }
    timers.set(key, setTimeout(tick, POLL_MS) as unknown as number)
  }
  // 提交后**先等一小拍**再问 ✓：引擎要装载权重，立刻查只会拿到一条"排队中"✓（无害但白跑一趟 ✓）
  timers.set(key, setTimeout(tick, 400) as unknown as number)
}

export function useEngineRun() {
  /** 档位清单（**一次拉取、模块级缓存** ✓：画布每张卡各拉一遍会把后端打爆 ✗） */
  async function loadStages(force = false) {
    if (stages.value.length && !force) return stages.value
    if (loadPromise) {
      await loadPromise
      return stages.value
    }
    stagesLoading.value = true
    stagesError.value = null
    loadPromise = (async () => {
      try {
        const data: any = await api.get('/engine/stages')
        stages.value = Array.isArray(data?.stages) ? data.stages : []
      } catch (err: any) {
        stagesError.value = err?.message || String(err)
      } finally {
        stagesLoading.value = false
        loadPromise = null
      }
    })()
    await loadPromise
    return stages.value
  }

  function stateOf(key: string): ComputedRef<EngineRunState> {
    return computed(() => states.value[key] || idleState())
  }

  /**
   * 起一次引擎生成 ✓。
   *
   * `payload` 直接就是 `/engine/generate` 的 body ✓（`stage` / `prompt` / `width`… ，
   * **不需要**传权重路径 ✗ —— 后端按档位从清单装配 ✓，缺件会写进 `autoNote` ✓）。
   */
  async function start(key: string, payload: Record<string, unknown>) {
    stop(key)
    patch(key, { ...idleState(), status: 'queued' })
    try {
      const result: any = await api.post('/engine/generate', payload)
      const taskId = String(result?.taskId || '')
      if (!taskId) throw new Error('后端没有返回任务 id')
      patch(key, { taskId, status: 'queued', note: result?.autoNote || null })
      rememberTask(key, taskId)
      poll(key, taskId)
      return taskId
    } catch (err: any) {
      // ⚠️ **队满不是失败** ✗✗（`E_ENGINE_BUSY` ✓）：引擎正跑着画布上**另一张卡**的任务 ✓ ——
      //    标成 `failed` 会让人以为这张卡坏了 ✓✗，其实是"等一会儿就好" ✓（后者才是真相 ✓）。
      /*
       * ⚠️ 必须读 `engineCode` ✗✗ —— 信封里的 `code` 是 **HTTP 状态码**（恒 400，
       * 见 `core/response.py::bad_request`），引擎码另在 `engineCode` ✓。
       * 旧写法只读 `code` ⇒ 这一支**从来没执行过** ✓✗ ⇒ 队满被标成 `failed`
       * （用户以为这张卡坏了，而真相是另一张卡在跑、等会儿再点 ✓）。
       */
      if (String(err?.engineCode || err?.code || '') === 'E_ENGINE_BUSY') {
        patch(key, idleState())
        throw new Error('引擎正忙（另一个任务在跑）⇒ 等它跑完再点，或先取消它')
      }
      patch(key, { status: 'failed', error: err?.message || String(err) })
      throw err
    }
  }

  async function cancel(key: string) {
    const taskId = states.value[key]?.taskId
    if (!taskId) return
    try {
      await api.post(`/engine/tasks/${taskId}/cancel`)
    } finally {
      stop(key)
      await syncDetail(key, taskId)
    }
  }

  function reset(key: string) {
    stop(key)
    rememberTask(key, null)
    const next = { ...states.value }
    delete next[key]
    states.value = next
  }

  /**
   * 刷新后把某张卡的任务**接回来** ✓（画布挂载时逐卡调 ✓）。
   *
   * 三种结局都要处理对：
   * * 还在跑 ⇒ 恢复进度、继续轮询 ✓（跑完照常回卡 ✓）；
   * * 已经跑完 ⇒ 只恢复**状态文本**并打 `restored` ✗（产物在刷新前就回卡了 ✓，
   *   再来一次会多出一张一样的卡 ✓✗）；
   * * 任务号已失效（后端重启过 / 被清过）⇒ `syncDetail` 拿不到 ⇒ 状态留 `queued` 会一直转 ✓✗
   *   ⇒ 这里**当场忘掉**并退回 idle ✓。
   */
  async function restore(key: string) {
    if (states.value[key]?.taskId) return
    const taskId = readRegistry()[key]
    if (!taskId) return
    /*
     * ⚠️ `restored` 必须在**查详情之前**就置上 ✗✗（不能等拿到终态再补）：
     * `syncDetail` 会把 `status` 直接写成 `succeeded`，而卡片的 watch 是**异步** flush 的 ——
     * 可能赶在"补写 restored"之前进回调 ⇒ 那时还是 false ⇒ 照常 `emit('generated')`
     * ⇒ 凭空多一张一样的产物卡 ✓✗（comfy 通道实测过：节点数 1 → 2）。
     */
    patch(key, { ...idleState(), taskId, status: 'queued', restored: true })
    const outcome = await syncDetail(key, taskId)
    const after = states.value[key]
    if (!after || after.taskId !== taskId) return
    if (outcome === 'missing') {
      // 后端不认识这个号（引擎重启过 / 任务被清）⇒ 忘掉并回 idle ✓（不留一个假"排队中" ✗）
      rememberTask(key, null)
      reset(key)
      return
    }
    if (isTerminal(after.status)) return  // 号已收尾 ⇒ restored 保持 true ✓（产物刷新前就回卡了）
    // 还在跑 ⇒ 标记交还给正常流程 ✓（这一轮真跑完时产物**要**照常回卡 ✓）
    patch(key, { restored: false })
    poll(key, taskId)
  }

  function phaseLabel(phase: string): string {
    return PHASE_LABEL[phase] || phase || ''
  }

  return {
    stages,
    stagesLoading,
    stagesError,
    loadStages,
    stateOf,
    start,
    cancel,
    reset,
    restore,
    phaseLabel,
    isTerminal,
  }
}
