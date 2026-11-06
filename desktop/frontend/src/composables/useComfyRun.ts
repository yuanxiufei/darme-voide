/**
 * ComfyUI 运行编排（画布 → 后端 → SSE 进度/预览 → 产物回卡）。
 *
 * 移植自上游 `ComfyUI-Local-Infinite-Canvas` 的 `canvas-image-generation` /
 * `canvas-node-progress-ring` 的**交互语义**（MIT），实现按本仓后端的接口重写。
 *
 * ## 四件必须做对的事（写错都是"看着在跑、其实没接上"）
 *
 * 1. ⭐ **入队与订阅分开** ✓：`POST /comfyui/runs` **立刻**返回 id（后台跑），
 *    真正的进度走 `GET /comfyui/runs/{id}/events`（SSE）。等 POST 返回结果再更新 UI，
 *    用户会盯着一个不动的按钮 3 分钟 ✗。
 * 2. ⭐ **收尾要拉一次详情** ✓：`done` 事件只说"跑完了"，**产物在 run 记录里**
 *    （`localPath` / `outputs`）。只听事件不取详情 ⇒ 永远拿不到图 ✗。
 * 3. ⭐ **流断了不等于失败** ✓：EventSource 断线（反代超时/切网）会触发 `onerror`，
 *    这跟"工作流失败"是两件事 ✗。这里退化成 2s 轮询，直到拿到终态 ✓。
 * 4. ⭐ **服务端失败事件叫 `failed` 不叫 `error`** ✓：`EventSource` 自己就有
 *    `error`（连接错误）⇒ 同名会把两者混成一条，报错信息还互相覆盖 ✗。
 *
 * 状态挂在**模块级单例**（与 `useCanvas` 同口径 ✓）：卡片、工具栏、任务中心读到的是同一份。
 */
import { computed, ref, type ComputedRef } from 'vue'
import { api } from './useApi'
// 后端基址：Web 同源、桌面端由壳注入绝对 origin（唯一真源见 `utils/apiBase.ts`）——
// ⚠️ EventSource 也吃这个基址：桌面端若写成相对路径，SSE 会连到壳自己的协议栈上 ✗
import { apiBase } from '../utils/apiBase'
/** EventSource 断线后的轮询间隔（SSE 正常时不会用到 ✓） */
const FALLBACK_POLL_MS = 2000

export interface ComfyWorkflowParam {
  key: string
  label?: string
  type?: string
  default?: unknown
  min?: number
  max?: number
  step?: number
  options?: unknown[]
  description?: string
}

export interface ComfyWorkflow {
  id: string
  name: string
  type: 'image' | 'video' | string
  category?: string
  description?: string
  file?: string
  variants: Array<{ id: string; label: string }>
  parameters: ComfyWorkflowParam[]
  minMedia?: unknown
}

export type RunStatus = 'idle' | 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export interface ComfyRunState {
  runId: number | null
  status: RunStatus
  /** 0–100 ✓（服务端已经 round 到 1 位；这里不再二次取整 ✗） */
  percent: number
  step: number
  total: number
  /** 采样中途的预览帧（data URL ✓ 可直接给 `<img>`） */
  preview: string | null
  error: string | null
  primaryUrl: string | null
  localPath: string | null
  outputs: Array<{ kind?: string; url?: string }>
  elapsedMs: number | null
  /** 后端是否**真的**有逐帧进度 ✓（无 websockets ⇒ false，界面别画一个永远 0% 的环 ✗） */
  progressAvailable: boolean
}

function idleState(): ComfyRunState {
  return {
    runId: null, status: 'idle', percent: 0, step: 0, total: 0, preview: null,
    error: null, primaryUrl: null, localPath: null, outputs: [], elapsedMs: null,
    progressAvailable: false,
  }
}

const workflows = ref<ComfyWorkflow[]>([])
const workflowsLoading = ref(false)
const workflowsError = ref<string | null>(null)
const states = ref<Record<string, ComfyRunState>>({})
const sources = new Map<string, EventSource>()
const timers = new Map<string, ReturnType<typeof setTimeout>>()
let loadPromise: Promise<void> | null = null

function patch(key: string, next: Partial<ComfyRunState>) {
  const current = states.value[key] || idleState()
  states.value = { ...states.value, [key]: { ...current, ...next } }
}

function close(key: string) {
  const source = sources.get(key)
  if (source) {
    source.close()
    sources.delete(key)
  }
  const timer = timers.get(key)
  if (timer) {
    clearTimeout(timer)
    timers.delete(key)
  }
}

function isTerminal(status: RunStatus) {
  return status === 'succeeded' || status === 'failed' || status === 'cancelled'
}

/** 拉一次 run 详情 ⇒ 产物/错误/耗时（**收尾必做** ✓，见模块头第 2 条） */
async function syncDetail(key: string, runId: number) {
  try {
    const run: any = await api.get(`/comfyui/runs/${runId}`)
    patch(key, {
      status: (run.status || 'running') as RunStatus,
      step: Number(run.step) || 0,
      error: run.errorMsg || null,
      primaryUrl: run.primaryUrl || null,
      localPath: run.localPath || null,
      outputs: Array.isArray(run.outputs) ? run.outputs : [],
      elapsedMs: run.elapsedMs ?? null,
    })
    if (run.status === 'succeeded') patch(key, { percent: 100 })
  } catch {
    // 查不到就保持现状 ✓（不要把一次网络抖动记成运行失败 ✗）
  }
}

/** EventSource 断线后的兜底：轮询到终态为止 ✓ */
function pollUntilDone(key: string, runId: number) {
  const tick = async () => {
    if (!states.value[key] || isTerminal(states.value[key].status)) return
    await syncDetail(key, runId)
    if (isTerminal(states.value[key].status)) return
    timers.set(key, setTimeout(tick, FALLBACK_POLL_MS))
  }
  timers.set(key, setTimeout(tick, FALLBACK_POLL_MS))
}

function subscribe(key: string, runId: number) {
  close(key)
  const source = new EventSource(`${apiBase()}/comfyui/runs/${runId}/events`)
  sources.set(key, source)

  const parse = (ev: MessageEvent) => {
    try {
      return JSON.parse(ev.data || '{}')
    } catch {
      return null
    }
  }

  source.addEventListener('snapshot', (ev) => {
    const data = parse(ev as MessageEvent)
    if (!data) return
    patch(key, {
      status: (data.status || 'queued') as RunStatus,
      percent: Number(data.percent) || 0,
      step: Number(data.step) || 0,
      total: Number(data.total) || 0,
      preview: data.preview?.image || null,
      progressAvailable: Boolean(data.progressAvailable),
    })
  })

  source.addEventListener('progress', (ev) => {
    const data = parse(ev as MessageEvent)
    if (!data) return
    patch(key, {
      status: 'running',
      percent: Number(data.percent) || 0,
      step: Number(data.step) || 0,
      total: Number(data.total) || 0,
      progressAvailable: true,
    })
  })

  source.addEventListener('preview', (ev) => {
    const data = parse(ev as MessageEvent)
    if (!data?.image) return
    patch(key, { preview: data.image, status: 'running' })
  })

  source.addEventListener('done', async () => {
    close(key)
    await syncDetail(key, runId)
  })

  source.addEventListener('failed', async (ev) => {
    const data = parse(ev as MessageEvent)
    close(key)
    patch(key, { status: 'failed', error: data?.message || '运行失败' })
    await syncDetail(key, runId)
  })

  source.addEventListener('cancelled', async () => {
    close(key)
    patch(key, { status: 'cancelled' })
    await syncDetail(key, runId)
  })

  // ⚠️ 这是**连接**错误（不是工作流失败）⇒ 退化成轮询 ✓（见模块头第 3 条）
  source.onerror = () => {
    if (!sources.has(key)) return
    close(key)
    pollUntilDone(key, runId)
  }
}

export function useComfyRun() {
  /** 工作流清单（**一次拉取、模块级缓存** ✓：画布上每张卡都拉一遍会把后端打爆 ✗） */
  async function loadWorkflows(force = false) {
    if (workflows.value.length && !force) return workflows.value
    if (loadPromise) {
      await loadPromise
      return workflows.value
    }
    workflowsLoading.value = true
    workflowsError.value = null
    loadPromise = (async () => {
      try {
        const data: any = await api.get('/comfyui/catalog/workflows')
        workflows.value = Array.isArray(data?.workflows) ? data.workflows : []
      } catch (err: any) {
        workflowsError.value = err?.message || String(err)
      } finally {
        workflowsLoading.value = false
        loadPromise = null
      }
    })()
    await loadPromise
    return workflows.value
  }

  function stateOf(key: string): ComputedRef<ComfyRunState> {
    return computed(() => states.value[key] || idleState())
  }

  /**
   * 起一次运行 ✓。
   *
   * ``payload`` 走 ``POST /comfyui/runs`` 的**注册表形态**：
   * ``{workflowId, variant, prompt, size, quality, count, values}`` ——
   * 后端 ``build_registry_graph`` 负责把参数写进图上正确的节点 ✓
   * （前端**不要**自己拼整张图：那是把 bindings 复刻一遍，迟早漂 ✗）。
   */
  async function start(key: string, payload: Record<string, unknown>) {
    close(key)
    patch(key, { ...idleState(), status: 'queued' })
    try {
      const run: any = await api.post('/comfyui/runs', { kind: 'workflow', ...payload })
      const runId = Number(run?.id)
      if (!runId) throw new Error('后端没有返回运行 id')
      patch(key, { runId, status: (run.status || 'queued') as RunStatus })
      subscribe(key, runId)
      return runId
    } catch (err: any) {
      patch(key, { status: 'failed', error: err?.message || String(err) })
      throw err
    }
  }

  async function cancel(key: string) {
    const runId = states.value[key]?.runId
    if (!runId) return
    try {
      await api.post(`/comfyui/runs/${runId}/cancel`)
    } finally {
      close(key)
      await syncDetail(key, runId)
    }
  }

  /** 只校验不提交 ✓（「生成」按钮点下去之前就能知道缺什么，不用等队列 ✗） */
  async function validate(payload: Record<string, unknown>) {
    return api.post('/comfyui/validate', { kind: 'workflow', ...payload })
  }

  function reset(key: string) {
    close(key)
    const next = { ...states.value }
    delete next[key]
    states.value = next
  }

  return { workflows, workflowsLoading, workflowsError, loadWorkflows, states, stateOf, start, cancel, validate, reset, close }
}
