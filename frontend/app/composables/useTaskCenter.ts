/**
 * 全局任务台账 —— 右下角「任务中心」的唯一真源（2026-10-27 立）。
 *
 * 病根（为什么要有它）：本仓的生成动作全是「点一下 → toast 一条 → 各页面自己拿一个
 * `pending*Ids` 数组盯着」，四处散落 ✗：
 *   ① `pendingVideoIds` / `pendingComposeIds` / `pendingShotFrameKeys` 都是**页面内 ref**
 *      ⇒ 切页即丢，回来只能刷新猜结果 ✓✗；
 *   ② 进度靠 `watchAsyncResult` 每 2.5–4 秒拉整个 episode 比对 URL ⇒ 只知道「好没好」，
 *      不知道「在干什么、到第几个」✓✗；
 *   ③ 失败只 `toast.error(e.message)` ⇒ 一条网络抖动和「缺首帧」长得一模一样，
 *      用户拿不到可操作的下一步 ✓✗。
 * 竞品（Seko / LibTV / Flova）把批量生成做成「批次 + 任务列表 + 单条可中断/可重试」，
 * 我们把它补成同构的一层，但**只做前端台账**：真源仍是后端 job/generation 记录，
 * 这里只是「谁在跑、跑成什么样」的可见面，不做调度 ★（调度在后端，见 auto_pipeline）。
 *
 * 设计约束：
 *   - `ssr: false`（nuxt.config）⇒ 模块级单例是安全的，不需要 `useState`；
 *   - 任务用 `key` 去重：同一镜头的同一个动作重复触发（含「重试」）**复用同一条目**，
 *     否则批量跑 20 镜 + 重试 3 次就能把列表撑成 60 条 ✓✗；
 *   - `progress` 允许 `null`（= 不知道百分比），此时界面显示 `stage` 文字，
 *     **不编造 95%** —— 这是本仓反复吃过的假进度坑 ✓✗。
 */

export type TaskKind = 'image' | 'video' | 'tts' | 'compose' | 'merge' | 'agent' | 'export' | 'pipeline'
export type TaskStatus = 'queued' | 'running' | 'success' | 'failed' | 'canceled'

/** 失败归因（决定给用户哪个按钮，而不是只给一句抱怨） */
export type FailureKind = 'network' | 'quota' | 'params' | 'model' | 'unknown'

export interface TaskFailure {
  kind: FailureKind
  /** 归因短标签，直接渲染在失败行上 */
  label: string
  /** 原始错误文案（可展开看） */
  message: string
  /** 人话建议：下一步该干什么 */
  hint: string
  /** 主行动 */
  action: { kind: 'retry' | 'goto'; label: string }
  /**
   * 兜底落地页：任务自己没给 `route` 时用它（目前只有「换模型 / 去配置」两族有）。
   * 为什么要有：`quota` / `model` 这类错的解法**不在当前页**，只给一句「换模型吧」
   * 等于让用户自己翻菜单 ✓✗ —— 竞品都是直接把人送过去。
   */
  route?: string
}

export interface TaskItem {
  id: string
  /** 去重键：同 key 的重复触发复用同一条目 */
  key: string
  kind: TaskKind
  title: string
  subtitle: string
  status: TaskStatus
  /** 阶段文字（「排队中」「生成中」「轮询 12/120」） */
  stage: string
  /** 0–100；null = 进度未知（不编造） */
  progress: number | null
  /** 去处理后跳到哪（本页时留空） */
  route: string
  failure: TaskFailure | null
  retry: (() => any) | null
  /**
   * 「取消」动作 —— **只有声明了它的任务才画取消按钮**。
   *
   * 为什么不给所有任务都画：已经提交到后端/引擎的渲染是撤不回来的 ✗，画一个
   * 「取消」却什么都没取消，比不给按钮更坏。只有**批次调度**这种「还没提交下一批」
   * 的任务能真停下来，所以由发起方自己声明（`batchVideos` 就是这种）★
   */
  onCancel: (() => any) | null
  startedAt: number
  finishedAt: number | null
}

export interface StartOptions {
  kind: TaskKind
  title: string
  subtitle?: string
  key?: string
  stage?: string
  route?: string
  retry?: (() => any) | null
  onCancel?: (() => any) | null
}

/** 成功的条目只留最近这么多条（失败的**不自动清**，那是用户要看的） */
const MAX_SUCCESS = 40

const state = reactive({
  items: [] as TaskItem[],
  seq: 0,
})

function find(id: string): TaskItem | undefined {
  return state.items.find(t => t.id === id)
}

function nowMs() {
  return Date.now()
}

/**
 * 只留「最近 MAX_SUCCESS 条成功」，失败/进行中一律保留。
 *
 * ⚠️ 这里**必须原地 splice**，不能 `state.items = state.items.filter(...)` ✗：
 * `useTaskCenter()` 把 `state.items` 这个**数组引用**交给组件，整体替换后
 * 组件里那份引用就指向旧数组了 ⇒ 列表再也不更新（而且不报错，最难查 ✓✗）。
 */
function trimSuccess() {
  const successes = state.items.filter(t => t.status === 'success')
  if (successes.length <= MAX_SUCCESS) return
  const drop = new Set(successes.slice(0, successes.length - MAX_SUCCESS).map(t => t.id))
  for (let i = state.items.length - 1; i >= 0; i--) {
    const item = state.items[i]
    if (item && drop.has(item.id)) state.items.splice(i, 1)
  }
}

/**
 * 错误归类 —— **本文件的核心价值**。
 *
 * 判据全部落在「后端真实会说的话」上（中文文案 + HTTP 状态 + `useApi.ts` 打的
 * `isNetworkError` 标记），不猜。归成五类，每类只给**一个**主行动：
 *   - `network`   网络/连接 → 重试（多半下一次就好）
 *   - `quota`     额度/余额/配额 → 去配置（重试一百次也没用）
 *   - `params`    缺参数/资源不存在 → 去改参数（重试也是同一个错）
 *   - `model`     模型/引擎/显存/不支持 → 换模型或降规格
 *   - `unknown`   认不出来 → 允许重试一次，并提示查设置
 */
export function classifyFailure(err: unknown): TaskFailure {
  const e: any = err
  const message = (e?.message || String(err || '未知错误')).trim() || '未知错误'
  const status = Number(e?.status || 0)
  const isNetwork = !!e?.isNetworkError
  const m = message.toLowerCase()

  if (isNetwork || /网络|连接失败|failed to fetch|fetch failed|econnrefused|networkerror|offline/.test(m)) {
    return {
      kind: 'network',
      label: '网络',
      message,
      hint: '后端没连上或连接被中断。确认服务在跑后重试即可。',
      action: { kind: 'retry', label: '重试' },
    }
  }
  if (/额度|余额|配额|积分|quota|insufficient|rate limit|限流|402/.test(m)) {
    return {
      kind: 'quota',
      label: '额度',
      message,
      hint: '额度或配额不足。到「模型」页换一条可用通道，或充值后再说。',
      action: { kind: 'goto', label: '去配置' },
      route: '/models',
    }
  }
  if (/超时|timeout|timed out/.test(m)) {
    return {
      kind: 'network',
      label: '超时',
      message,
      hint: '等太久没回来，多半是上游慢。可直接重试；连续超时就走小批量。',
      action: { kind: 'retry', label: '重试' },
    }
  }
  if (/显存|oom|out of memory|cuda|引擎|模型不支持|不支持|unsupported|not support|comfyui|503|500/.test(m)) {
    return {
      kind: 'model',
      label: '模型',
      message,
      hint: '当前模型/引擎跑不动这一条。换模型或降分辨率后重试。',
      action: { kind: 'goto', label: '换模型' },
      route: '/models',
    }
  }
  if (
    status === 404
    || /缺少|缺失|必填|不能为空|请先(选择|设置|填写|上传|生成)|不存在|未找到|not found|invalid|参数/.test(m)
  ) {
    return {
      kind: 'params',
      label: '缺参数',
      message,
      hint: '输入不满足要求（缺首帧/缺对白/资源被删等）。补齐后再点，重试同一个请求不会有变化。',
      action: { kind: 'goto', label: '去处理' },
    }
  }
  return {
    kind: 'unknown',
    label: '未知',
    message,
    hint: '没认出来的错。先重试一次；仍失败就到「设置」看服务状态与日志。',
    action: { kind: 'retry', label: '重试' },
  }
}

export function useTaskCenter() {
  /** 开一个任务（同 key 复用并重置，见文件头约束） */
  function start(opts: StartOptions): TaskItem {
    const key = opts.key || `${opts.kind}:${opts.title}`
    const existing = state.items.find(t => t.key === key)
    if (existing) {
      existing.title = opts.title
      existing.subtitle = opts.subtitle ?? existing.subtitle
      existing.status = 'running'
      existing.stage = opts.stage || '排队中'
      existing.progress = null
      existing.failure = null
      existing.route = opts.route || existing.route
      if (opts.retry !== undefined) existing.retry = opts.retry
      if (opts.onCancel !== undefined) existing.onCancel = opts.onCancel
      existing.startedAt = nowMs()
      existing.finishedAt = null
      return existing
    }
    state.seq += 1
    const item: TaskItem = {
      id: `t${state.seq}`,
      key,
      kind: opts.kind,
      title: opts.title,
      subtitle: opts.subtitle || '',
      status: 'running',
      stage: opts.stage || '排队中',
      progress: null,
      route: opts.route || '',
      failure: null,
      retry: opts.retry ?? null,
      onCancel: opts.onCancel ?? null,
      startedAt: nowMs(),
      finishedAt: null,
    }
    state.items.unshift(item)
    return item
  }

  /** 更新阶段文字与（可选）真实百分比 */
  function stage(id: string, text: string, progress?: number | null) {
    const t = find(id)
    if (!t || t.status === 'success' || t.status === 'failed' || t.status === 'canceled') return
    t.stage = text
    if (progress !== undefined) t.progress = progress
  }

  function done(id: string, stageText?: string) {
    const t = find(id)
    if (!t) return
    t.status = 'success'
    t.stage = stageText || '完成'
    // 完成恒为 100%：流水线是按「第 i/n 步」推进的，收尾时最后上报的是 (n-1)/n，
    // 若顺着它留着，成功的条目会永远停在 86% 这种读数上 ✓✗
    t.progress = 100
    t.failure = null
    t.onCancel = null
    t.finishedAt = nowMs()
    trimSuccess()
  }

  function fail(id: string, err: unknown, opts?: { retry?: (() => any) | null; route?: string }) {
    const t = find(id)
    if (!t) return
    t.status = 'failed'
    t.failure = classifyFailure(err)
    t.stage = '失败'
    t.onCancel = null
    t.finishedAt = nowMs()
    if (opts?.retry !== undefined) t.retry = opts.retry
    if (opts?.route) t.route = opts.route
  }

  function cancel(id: string, stageText = '已取消') {
    const t = find(id)
    if (!t) return
    t.status = 'canceled'
    t.stage = stageText
    t.onCancel = null
    t.finishedAt = nowMs()
  }

  /**
   * 用户点「取消」：只有**自己声明了 `onCancel`** 的任务走到这里，动作由发起方实现
   * （例如批量视频 = 不再提交下一批）。已提交的那些本来就不能撤，发起方必须在文案里说清。
   */
  async function requestCancel(id: string) {
    const t = find(id)
    if (!t?.onCancel) return
    const fn = t.onCancel
    t.onCancel = null // 防连点：先摘掉，动作失败也不重入
    try {
      await fn()
    } finally {
      // 动作自己收尾了（可能置成 canceled）就不再覆盖；还挂着 running 才兜底
      const cur = find(id)
      if (cur && (cur.status === 'running' || cur.status === 'queued')) cancel(id, '已取消')
    }
  }

  /** 用户点「重试」：复位后重跑原动作（原动作内部 start 同 key ⇒ 复用本条目） */
  async function retryTask(id: string) {
    const t = find(id)
    if (!t || !t.retry) return
    const fn = t.retry
    t.status = 'running'
    t.failure = null
    t.stage = '重试中…'
    t.progress = null
    t.startedAt = nowMs()
    t.finishedAt = null
    try {
      await fn()
    } catch (e) {
      fail(id, e)
    }
  }

  function remove(id: string) {
    const i = state.items.findIndex(t => t.id === id)
    if (i >= 0) state.items.splice(i, 1)
  }

  function clearFinished() {
    for (let i = state.items.length - 1; i >= 0; i--) {
      const st = state.items[i]?.status
      if (st && st !== 'running' && st !== 'queued') state.items.splice(i, 1)
    }
  }

  const summary = computed(() => {
    let running = 0
    let failed = 0
    for (const t of state.items) {
      if (t.status === 'running' || t.status === 'queued') running += 1
      else if (t.status === 'failed') failed += 1
    }
    return { running, failed, total: state.items.length }
  })

  return {
    items: state.items,
    summary,
    start,
    stage,
    done,
    fail,
    cancel,
    requestCancel,
    retryTask,
    remove,
    clearFinished,
  }
}

/**
 * 这条失败该把人送到哪 —— 任务自带落点优先（如「本集 · 镜头图片」），
 * 其次是归因兜底（如「模型」页），都没有就返回空 ⇒ 界面不画那个按钮
 * （**绝不画一个点了没反应的按钮** ★）。
 */
export function failureRoute(t: TaskItem): string {
  return t.route || t.failure?.route || ''
}

/** 耗时文案（面板每秒刷新 `now` 才有「12 秒」这种读数） */
export function taskElapsed(t: TaskItem, now: number): string {
  const end = t.finishedAt ?? now
  const ms = Math.max(0, end - t.startedAt)
  if (ms < 1000) return '<1 秒'
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s} 秒`
  const m = Math.floor(s / 60)
  const rest = s % 60
  return rest ? `${m} 分 ${rest} 秒` : `${m} 分`
}
