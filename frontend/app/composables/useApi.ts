// 共享契约：字段权威在后端（Python），本文件只是前端侧镜像（见 app/types/contracts.ts）——
// 禁止在前端各处再手工双写类型
import type { DramaDetailDTO, DramaListResponse, DramaUpdateBody, EraBackground } from '~contracts'
// 后端基址：Web 同源、桌面端由壳注入绝对 origin（唯一真源与理由见 `utils/apiBase.ts` 文件头）
import { apiBase } from '../utils/apiBase'

/**
 * 唯一出口（JSON）。
 *
 * `signal` 只有「能被用户中途叫停」的慢接口才传（画布 AI 一次要 5–20 秒）✓。
 * 中止时 `fetch` 抛 `AbortError`，本函数原样透传 —— 调用方靠 `name === 'AbortError'`
 * 区分「用户叫停」与「真失败」，所以这里**不能**把它包装成别的错误 ✗。
 */
async function req<T = any>(method: string, path: string, body?: any, signal?: AbortSignal): Promise<T> {
  const opts: RequestInit = { method, headers: { 'Content-Type': 'application/json' } }
  if (body) opts.body = JSON.stringify(body)
  if (signal) opts.signal = signal

  const start = performance.now()
  console.log(`%c[API] %c${method} %c${path}`, 'color:#888', 'color:#2dd4bf;font-weight:bold', 'color:#ccc', body || '')

  try {
    const resp = await fetch(`${apiBase()}${path}`, opts)
    const json = await resp.json()
    const ms = Math.round(performance.now() - start)

    if (!resp.ok || (json.code && json.code >= 400)) {
      console.log(`%c[API] %c${method} ${path} %c${resp.status} %c${ms}ms`, 'color:#888', 'color:#ef5350', 'color:#ef5350;font-weight:bold', 'color:#888', json.message || '')
      // 增强错误信息：包含状态码和路径，便于调试
      // ⚠️ `json.msg` 是 `/preset/framework` 域的失败字段（全项目只有它用 msg 而不是 message）——
      //    不兜底就会把后端原话吞成「请求失败 (400)」，而那句 msg 恰好是唯一的病因说明
      const error = new Error(json.message || json.msg || `请求失败 (${resp.status})`)
      ;(error as any).status = resp.status
      ;(error as any).code = json.code
      /*
       * ⚠️ 信封里的 `code` 是 **HTTP 状态码**（400 ✓），引擎错误码另在 `engineCode` 那一格 ✓
       * （真源 `core/response.py::bad_request` ✓ —— 它明写「两者不是一回事」✓）。
       * 不转发这一格 ⇒ 调用方只能拿到 400 ⇒ 所有按引擎码分流的地方都**静默失效** ✓✗
       * （已中招一次：`useEngineRun` 的队满分支恒不成立，队满被标成「这张卡坏了」✓✗）。
       */
      ;(error as any).engineCode = json.engineCode
      ;(error as any).path = path
      throw error
    }

    console.log(`%c[API] %c${method} ${path} %c${resp.status} %c${ms}ms`, 'color:#888', 'color:#66bb6a', 'color:#66bb6a;font-weight:bold', 'color:#888')
    return json.data ?? json
  } catch (err: any) {
    // 区分网络错误和业务错误
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      const ms = Math.round(performance.now() - start)
      console.log(`%c[API] %c${method} ${path} %cNETWORK ERROR %c${ms}ms`, 'color:#888', 'color:#ff9800', 'color:#ff9800;font-weight:bold', 'color:#888', err.message)
      const networkError = new Error('网络连接失败，请检查网络或服务器状态')
      ;(networkError as any).isNetworkError = true
      throw networkError
    }
    if (!err.message?.match(/^\d{3}$/)) {
      const ms = Math.round(performance.now() - start)
      console.log(`%c[API] %c${method} ${path} %cERROR %c${ms}ms`, 'color:#888', 'color:#ef5350', 'color:#ef5350;font-weight:bold', 'color:#888', err.message)
    }
    throw err
  }
}

async function reqForm<T = any>(path: string, formData: FormData): Promise<T> {
  const resp = await fetch(`${apiBase()}${path}`, { method: 'POST', body: formData })
  const json = await resp.json()
  if (!resp.ok || (json.code && json.code >= 400)) {
    const error = new Error(json.message || `请求失败 (${resp.status})`)
    ;(error as any).status = resp.status
    ;(error as any).code = json.code
    ;(error as any).engineCode = json.engineCode
    throw error
  }
  return json.data ?? json
}

/**
 * 带进度的 multipart 上传。
 *
 * 为什么不用 `reqForm`：`fetch` **报不出上传进度**（没有 upload.onprogress 那一层），
 * 只能「转圈 → 突然完成」。大文件（几十 MB 的 docx）要几十秒，用户看不到百分比就会
 * 再点一次，于是同一份被传两遍（后端会落成 `-2`）。
 * 进度只表示「字节发完了」，之后还有服务端抽正文的耗时，所以界面在 100% 时改说「解析中」。
 */
function uploadFiles<T = any>(
  path: string,
  files: File[],
  onProgress?: (pct: number) => void,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const form = new FormData()
    for (const file of files) form.append('files', file, file.name)
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${apiBase()}${path}`)
    xhr.upload.onprogress = (ev: ProgressEvent) => {
      if (ev.lengthComputable && onProgress) onProgress(Math.round((ev.loaded / ev.total) * 100))
    }
    xhr.onload = () => {
      let json: any = null
      try {
        json = JSON.parse(xhr.responseText)
      } catch {
        json = null
      }
      if (xhr.status >= 200 && xhr.status < 300 && json && !(json.code >= 400)) {
        resolve((json.data ?? json) as T)
        return
      }
      const error = new Error(json?.message || `上传失败 (${xhr.status})`)
      ;(error as any).status = xhr.status
      reject(error)
    }
    xhr.onerror = () => reject(new Error('网络连接失败，请检查网络或服务器状态'))
    xhr.send(form)
  })
}

export const api = {
  get: <T = any>(p: string) => req<T>('GET', p),
  post: <T = any>(p: string, b?: any) => req<T>('POST', p, b),
  put: <T = any>(p: string, b?: any) => req<T>('PUT', p, b),
  /**
   * 部分字段更新。与 PUT 的差别是**语义**，不是风格：后端 `PATCH /knowledge-loop/entries/{id}`
   * 有意不提供 PUT —— 忘记带 `body` 的 PUT 会把几百字正文清空（见 `docs/knowledge-loop.md`）。
   */
  patch: <T = any>(p: string, b?: any) => req<T>('PATCH', p, b),
  del: <T = any>(p: string) => req<T>('DELETE', p),
}

export const dramaAPI = {
  list: () => api.get<DramaListResponse>('/dramas'),
  get: (id: number) => api.get<DramaDetailDTO>(`/dramas/${id}`),
  create: (data: any) => api.post('/dramas', data),
  update: (id: number, data: DramaUpdateBody) => api.put(`/dramas/${id}`, data),
  del: (id: number) => api.del(`/dramas/${id}`),
  stats: () => api.get<{ total: number; by_status: { status: string; count: number }[] }>('/dramas/stats'),
  prompts: (id: number) => api.get(`/dramas/${id}/prompts`),
  /** AI 从剧本提炼时代背景并落库（与 PUT 时代背景同构，三字段规范见 `~contracts` 的 `EraBackground`） */
  eraExtract: (id: number, data?: { source_text?: string }) => api.post<EraBackground>(`/dramas/${id}/era-background/extract`, data || {}),
}

// 全局应用设置（KV：art_style 全局默认画风等）
export const appSettingsAPI = {
  get: () => api.get<{ art_style?: string | null }>('/app-settings'),
  setArtStyle: (artStyle: string) => api.put('/app-settings', { art_style: artStyle }),
}

export const episodeAPI = {
  create: (data: any) => api.post('/episodes', data),
  update: (id: number, data: any) => api.put(`/episodes/${id}`, data),
  del: (id: number) => api.del(`/episodes/${id}`),
  characters: (id: number) => api.get(`/episodes/${id}/characters`),
  scenes: (id: number) => api.get(`/episodes/${id}/scenes`),
  storyboards: (id: number) => api.get(`/episodes/${id}/storyboards`),
  pipelineStatus: (id: number) => api.get(`/episodes/${id}/pipeline-status`),
  continueScript: (id: number, data: { mode: 'raw' | 'script'; text: string }) => api.post(`/episodes/${id}/continue-script`, data),
}

export const storyboardAPI = {
  create: (data: any) => api.post('/storyboards', data),
  update: (id: number, data: any) => api.put(`/storyboards/${id}`, data),
  generateTTS: (id: number) => api.post(`/storyboards/${id}/generate-tts`),
  validateDialogue: (id: number) => api.get(`/storyboards/${id}/validate-dialogue`),
  regenerateImage: (id: number, data: any) => api.post(`/storyboards/${id}/regenerate-image`, data),
  regenerateFrame: (id: number, data: any) => api.post(`/storyboards/${id}/regenerate-frame`, data),
  setFrame: (id: number, data: any) => api.post(`/storyboards/${id}/set-frame`, data),
  del: (id: number) => api.del(`/storyboards/${id}`),
}

export const characterAPI = {
  get: (id: number) => api.get(`/characters/${id}`),
  update: (id: number, data: any) => api.put(`/characters/${id}`, data),
  voiceSample: (id: number, episodeId?: number) => api.post(`/characters/${id}/generate-voice-sample`, { episode_id: episodeId }),
  generateImage: (id: number, episodeId?: number, data?: { prompt?: string; negative_prompt?: string; model?: string; costume?: string; color_grade?: any }) =>
    api.post(`/characters/${id}/generate-image`, { episode_id: episodeId, ...data }),
  generatePrompt: (id: number, data?: { type?: 'character' | 'clothing' | 'weapon' | 'accessory'; name?: string; appearance?: string; personality?: string; description?: string; clothing?: string; weapons?: string; accessories?: string; costumes?: string; style?: string }) =>
    api.post(`/characters/${id}/generate-prompt`, data),
  generateThreeViews: (id: number, episodeId?: number, data?: { prompt?: string; negative_prompt?: string; model?: string; views?: string[]; clothing?: string; weapons?: string; accessories?: string; costumes?: string; style?: string; anchor?: string; reference_images?: string[] }) =>
    api.post(`/characters/${id}/generate-three-views`, { episode_id: episodeId, ...data }),
  generateEquipImage: (id: number, episodeId?: number, data?: { type: string; prompt?: string; negative_prompt?: string; model?: string; color_grade?: any; clothing?: string; weapons?: string; accessories?: string; costumes?: string; anchor?: string; mode?: 'single' | 'view'; reference_images?: string[] }) =>
    api.post(`/characters/${id}/generate-equip-image`, { episode_id: episodeId, ...data }),
  // 批量生成表情头像特写组：body.keys 为空默认全部 9 个；传单 key 单独重生成
  generateExpressions: (id: number, data?: { keys?: string[]; appearance?: string; clothing?: string; costumes?: string; model?: string; episodeId?: number; anchor?: string; reference_images?: string[] }) =>
    api.post(`/characters/${id}/generate-expressions`, data),
  batchImages: (ids: number[], episodeId: number) => api.post('/characters/batch-generate-images', { character_ids: ids, episode_id: episodeId }),
  autoSplitVisuals: (id: number, appearance?: string) => api.post(`/characters/${id}/auto-split-visuals`, { appearance }),
  /** 跨集一致性修复：把 source 角色(id) 并入全剧已有 target 角色（target 资产优先，source 关联迁移后软删） */
  merge: (id: number, targetId: number) => api.post(`/characters/${id}/merge`, { target_id: targetId }),
}

export const sceneAPI = {
  get: (id: number) => api.get(`/scenes/${id}`),
  update: (id: number, data: any) => api.put(`/scenes/${id}`, data),
  generateImage: (id: number, episodeId?: number, data?: { prompt?: string; negative_prompt?: string; model?: string }) =>
    api.post(`/scenes/${id}/generate-image`, { episode_id: episodeId, ...data }),
}

export const imageAPI = {
  generate: (d: any) => api.post('/images', d),
  list: (params?: { drama_id?: number; storyboard_id?: number }) => {
    const query = new URLSearchParams()
    if (params?.drama_id) query.set('drama_id', String(params.drama_id))
    if (params?.storyboard_id) query.set('storyboard_id', String(params.storyboard_id))
    return api.get(`/images${query.size ? `?${query.toString()}` : ''}`)
  },
}

// ====== 生成历史（聚合图片 / 视频生成记录） ======

export interface GenerationRecord {
  id: number
  type: 'image' | 'video'
  storyboardId: number | null
  dramaId: number | null
  provider: string | null
  model: string | null
  prompt: string | null
  status: string
  errorMsg: string | null
  taskId: string | null
  url: string | null
  duration?: number | null
  createdAt: string
  completedAt: string | null
  elapsedMs: number | null
}

export const generationsAPI = {
  list: (params?: { type?: 'image' | 'video'; drama_id?: number; storyboard_id?: number; limit?: number }) => {
    const query = new URLSearchParams()
    if (params?.type) query.set('type', params.type)
    if (params?.drama_id) query.set('drama_id', String(params.drama_id))
    if (params?.storyboard_id) query.set('storyboard_id', String(params.storyboard_id))
    if (params?.limit) query.set('limit', String(params.limit))
    return api.get<GenerationRecord[]>(`/generations${query.size ? `?${query.toString()}` : ''}`)
  },
}
/** 视觉图谱里的一个术语节点（数据契约在 `backend-py/app/services/visual_graph.py::VISUAL_GRAPH`） */
export interface VisualTerm {
  /** 中文别名，**第一个是主名**；匹配时长短要区分（「中景」不能吃掉「中近景」） */
  zh: string[]
  /** 英文电影术语 —— 会直接进 image/video prompt */
  en: string
  /** 使用场景说明 */
  usage: string
  /** 常搭配的英文术语（部分节点没有） */
  pairsWith?: string[]
}
export type VisualCategory = 'shot_size' | 'composition' | 'movement' | 'lighting'
export interface VisualGraphResponse {
  graph: Partial<Record<VisualCategory, VisualTerm[]>>
  categories: VisualCategory[]
}

export const visualGraphAPI = {
  /** 图谱全量（传 category 只取一类；非法值后端会**退回全部**，不会报错） */
  graph: (category?: VisualCategory) =>
    api.get<VisualGraphResponse>(`/visual-graph${category ? `?category=${category}` : ''}`),
  /** 中文术语 → 英文电影术语；没命中时 `en` 为 null（不是抛错） */
  resolve: (text: string) => api.get<{ zh: string; en: string | null }>(`/visual-graph/resolve?text=${encodeURIComponent(text)}`),
  /** 某类的全部主名（不含别名） */
  terms: (category: VisualCategory) => api.get<{ category: string; terms: string[] }>(`/visual-graph/terms?category=${category}`),
  /**
   * 按剧类型/风格取「拆镜引导」**纯文本**。
   *
   * ⚠️ 这个端点回的是 `text/plain`，走不了 `api.get` —— 那个封装第一句就是 `resp.json()`，
   * 对非 JSON 会直接抛解析错误（错误信息还会长得像网络问题，极难查 ✓✗）。
   * 另外：`drama_id` 不存在时后端是 **404**，这里要把它变成能看懂的话。
   */
  guidance: async (dramaId?: number): Promise<string> => {
    const resp = await fetch(`${apiBase()}/visual-graph/guidance${dramaId ? `?drama_id=${dramaId}` : ''}`)
    if (!resp.ok) {
      const json = await resp.json().catch(() => null)
      const msg = json?.message || (resp.status === 404 ? '剧集不存在（可能已删除）' : `引导文本获取失败 (HTTP ${resp.status})`)
      throw new Error(msg)
    }
    return resp.text()
  },
}

export const gridAPI = {
  prompt: (d: any) => api.post('/grid/prompt', d),
  generate: (d: any) => api.post('/grid/generate', d),
  status: (id: number) => api.get(`/grid/status/${id}`),
  split: (d: any) => api.post('/grid/split', d),
}
export const videoAPI = {
  generate: (d: any) => api.post('/videos', d),
  get: (id: number) => api.get(`/videos/${id}`),
  update: (id: number, d: any) => api.put(`/videos/${id}`, d),
  regenerate: (id: number, d: any) => api.post(`/videos/${id}/regenerate`, d),
}
export const composeAPI = {
  shot: (id: number) => api.post(`/compose/storyboards/${id}/compose`),
  all: (epId: number) => api.post(`/compose/episodes/${epId}/compose-all`),
  status: (epId: number) => api.get(`/compose/episodes/${epId}/compose-status`),
}
export const mergeAPI = {
  merge: (epId: number) => api.post(`/merge/episodes/${epId}/merge`),
  status: (epId: number) => api.get(`/merge/episodes/${epId}/merge`),
}
export const aiConfigAPI = {
  list: (t?: string) => api.get(`/ai-configs${t ? `?service_type=${t}` : ''}`),
  create: (d: any) => api.post('/ai-configs', d),
  update: (id: number, d: any) => api.put(`/ai-configs/${id}`, d),
  del: (id: number) => api.del(`/ai-configs/${id}`),
  test: (d: any) => api.post('/ai-configs/test', d),
  models: (d: any) => api.post('/ai-configs/models', d),
  quickPreset: (apiKey: string) => api.post('/ai-configs/quick-preset', { api_key: apiKey }),
  quickLocal: () => api.post('/ai-configs/quick-local'),
  configsLocal: () => api.get('/ai-configs/configs/local'),
  gpuStatus: () => api.get('/ai-configs/gpu/status'),
  gpuReleaseAll: () => api.post('/ai-configs/gpu/release-all'),
  ollamaStatus: (baseUrl?: string) => api.post('/ai-configs/ollama/status', { base_url: baseUrl }),
  ollamaStart: () => api.post('/ai-configs/ollama/start'),
  /** 统一探测本地四大运行时（文本/图像/视频/语音）健康状态 */
  runtimeHealth: () => api.get('/ai-configs/runtime/health'),
  /**
   * 自研文本（LLM）后端的档位回执。
   *
   * ⚠️ 这条是 **GET 且不装载**：`kvMode`/`kvWindow`/`gpuLayers` 只活在文本服务的 settings 里，
   * 而「已装的那个后端」是按**装载那一刻**的档位建的 ⇒ 改完配置它不会跟着变（缓存键按档位
   * 分格 ⇒ 下次生成另装一份）。回执把 `requested` 与 `backends` 并排 + `matches`/`mismatch`，
   * 否则「我配了呀，怎么显存一动不动」只能靠猜。`duplicates` 报**同权重多档位各占一份显存**。
   */
  engineLlmStatus: () => api.get('/engine/llm'),
  /** 删除本地 Ollama 模型 */
  ollamaDelete: (name: string, baseUrl?: string) => api.post('/ai-configs/ollama/delete', { name, base_url: baseUrl }),
  /** 下载 Ollama 模型，NDJSON 流式返回进度事件 { status, completed, total, error } */
  ollamaPull: async (name: string, baseUrl?: string, onEvent?: (ev: any) => void): Promise<{ ok: boolean; error?: string }> => {
    const resp = await fetch(`${apiBase()}/ai-configs/ollama/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, base_url: baseUrl }),
    })
    if (!resp.ok) {
      const json = await resp.json().catch(() => null)
      throw new Error(json?.message || `下载失败 (HTTP ${resp.status})`)
    }
    const ct = resp.headers.get('content-type') || ''
    if (!ct.includes('ndjson') || !resp.body) {
      const json = await resp.json().catch(() => null)
      return json?.data ?? json
    }
    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, idx).trim()
        buf = buf.slice(idx + 1)
        if (!line) continue
        try {
          const ev = JSON.parse(line)
          if (ev.error) return { ok: false, error: ev.error }
          onEvent?.(ev)
        } catch { /* 忽略非 JSON 行 */ }
      }
    }
    return { ok: true }
  },
}

export const localModelsAPI = {
  /** 扫描本地模型文件并识别类型，返回 { roots, models, total, truncated, elapsedMs, byKind } */
  scan: (params?: { roots?: string[]; kinds?: string[]; maxDepth?: number; maxFiles?: number }) => {
    const q = new URLSearchParams()
    if (params?.roots?.length) q.set('roots', JSON.stringify(params.roots))
    if (params?.kinds?.length) q.set('kinds', params.kinds.join(','))
    if (params?.maxDepth) q.set('maxDepth', String(params.maxDepth))
    if (params?.maxFiles) q.set('maxFiles', String(params.maxFiles))
    const qs = q.toString()
    return api.get(`/local-models/scan${qs ? `?${qs}` : ''}`)
  },
  /** 返回默认扫描根目录、探测状态及完整路径配置（paths.models_dir 等） */
  roots: () => api.get('/local-models/roots'),
  /** 保存模型路径配置（partial：roots=额外扫描目录，models_dir=模型存储目录） */
  savePaths: (patch: { roots?: string[]; models_dir?: string }) => api.put('/local-models/roots', patch),
  /** 将扫描到的模型注册进 ai_service_configs，返回 { created, updated, skipped, configs } */
  register: (models: any[], name?: string) => api.post('/local-models/register', { models, name }),
  /** 列出可用磁盘盘符，返回 { drives: [{ letter, root }] } */
  drives: () => api.get('/local-models/drives'),
  /** 启动后台异步扫描任务，返回 { taskId } */
  scanAsync: (params?: { roots?: string[]; kinds?: string[]; maxDepth?: number; maxFiles?: number }) =>
    api.post('/local-models/scan', params || {}),
  /** 查询扫描任务进度/结果，返回 { taskId, progress } */
  scanStatus: (taskId: string) => api.get(`/local-models/scan/status?taskId=${encodeURIComponent(taskId)}`),
  /** 取消扫描任务 */
  scanCancel: (taskId: string) => api.post(`/local-models/scan/cancel?taskId=${encodeURIComponent(taskId)}`),
  /** 列出仓库文件（按来源），返回 { repo, source, revision, files: [{ name, size }] } */
  hfFiles: (repo: string, revision?: string, source?: string) => api.post('/local-models/hf/files', { repo, revision, source }),
  /** 删除「模型存储目录」下的本地模型文件/目录，返回 { deleted, failed } */
  delFiles: (paths: string[]) => api.post('/local-models/delete', { paths }),
  /** 下载模型到「模型存储目录」（按来源），NDJSON 流式进度事件 { status, file, downloaded, total, overall, ... } */
  hfDownload: async (repo: string, files: string[], revision: string, source: string, onEvent?: (ev: any) => void): Promise<{ ok: boolean }> => {
    const resp = await fetch(`${apiBase()}/local-models/hf/download`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo, files, revision, source }),
    })
    if (!resp.ok || !resp.body) {
      const json = await resp.json().catch(() => null)
      throw new Error(json?.message || `下载失败 (HTTP ${resp.status})`)
    }
    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, idx).trim()
        buf = buf.slice(idx + 1)
        if (!line) continue
        try { onEvent?.(JSON.parse(line)) } catch { /* 忽略非 JSON 行 */ }
      }
    }
    return { ok: true }
  },
}

/** 画布工具域（后端只**编译** op，落盘由 `utils/canvasOps.ts` 做） */
export const canvasAPI = {
  /** 工具清单（function-calling 形态：22 个已迁 + 未迁的逐条理由） */
  tools: () => api.get('/canvas/tools'),
  /** 编译一次工具调用 ⇒ ops（不执行、不起任务） */
  tool: (d: { name: string; input?: Record<string, any>; snapshot?: any }) =>
    api.post('/canvas/tool', d) as Promise<{ ops: any[]; errors: string[]; needsSnapshot?: boolean }>,
  /** 执行前整批自检：有错就整批放弃，别半批落地 */
  validate: (d: { ops: any[]; snapshot?: any }) =>
    api.post('/canvas/ops/validate', d) as Promise<{ ops: any[]; errors: string[]; ok: boolean }>,
  /** 压缩快照（正文截断，只留判断所需字段） */
  compact: (d: { snapshot: any }) => api.post('/canvas/ops/snapshot', d),
  /**
   * 一句话 ⇒ 一串待执行 op（`ops` 此刻**还没落地**）。
   *
   * `signal` 是给对话面板「停止」用的：模型这一趟要 5–20 秒，用户等不下去时必须能真的
   * 停下 —— 中止后请求被丢弃、`runCanvasAgent` 抛 `AbortError`、**一条 op 都不会落** ✓。
   * 只做「界面不转圈、但结果照落」是假停止 ✗。
   */
  agent: (d: { message: string; snapshot?: any; scene?: string }, signal?: AbortSignal) =>
    req('POST', '/canvas/agent', d, signal) as Promise<{
      text: string
      ops: any[]
      errors: string[]
      toolCalls: Array<{ toolName: string; args: Record<string, any> }>
    }>,

  // —— 计划通路（意图 → 计划 → op，中间有一道**人工确认**）——
  //
  // 与 `agent` 的区别值得留意：`agent` 一句话回来的是能直接落地的 op（快，但没有反悔机会）；
  // `plan` 回来的是一份**草案**（有 preflight 计数、**没有 ops**），用户确认后才走 confirm 拿 op ✓。
  // 要「AI 改画布但我先看一眼」就用这条；要「快，别问」就还用 `agent` ✓。

  /** 一句话 ⇒ **草案**（模型只出计划，编译器出 op，此刻**不下发 op**） */
  planAuto: (d: { message: string; snapshot?: any; baseRevision?: number }) =>
    api.post('/canvas/plan/auto', d) as Promise<{
      planId: string
      goal: string
      status: string
      preflight: { total: number; creates: number; links: number; runs: number; byOp: Record<string, number>; byNodeType: Record<string, number> }
      layout: Record<string, any>
      warnings: string[]
      expectedRevision: number | null
      requiresConfirmation: boolean
      alreadyConfirmed?: boolean
      modelText?: string
      note?: string
    }>,
  /** 交一份**已写好的**计划（工具链/测试用；用户侧走 `planAuto`） */
  plan: (d: { plan: any; snapshot?: any; baseRevision?: number }) =>
    api.post('/canvas/plan', d) as Promise<{
      planId: string
      goal: string
      status: string
      preflight: { total: number; creates: number; links: number; runs: number; byOp: Record<string, number>; byNodeType: Record<string, number> }
      layout: Record<string, any>
      warnings: string[]
      expectedRevision: number | null
      requiresConfirmation: boolean
      alreadyConfirmed?: boolean
      note?: string
    }>,
  /** 草案列表（新的在前） */
  plans: (limit = 20) => api.get(`/canvas/plans?limit=${limit}`),
  /** 单份草案详情（draft 阶段同样不含 ops） */
  planDetail: (planId: string) => api.get(`/canvas/plans/${planId}`),
  /** 确认：用**当下**快照重编译后下发 ops；重复确认返回上次缓存（`replayed=true`） */
  confirmPlan: (planId: string, d: { snapshot?: any; baseRevision?: number } = {}) =>
    api.post(`/canvas/plans/${planId}/confirm`, d) as Promise<{
      planId: string
      goal: string
      status: string
      ops: any[]
      refs: Record<string, string>
      preflight: { total: number; creates: number; links: number; runs: number; byOp: Record<string, number>; byNodeType: Record<string, number> }
      warnings: string[]
      expectedRevision: number | null
      replayed: boolean
      note?: string
    }>,
}

export const agentConfigAPI = {
  list: () => api.get('/agent-configs'),
  /** 出厂默认配置（默认提示词 + 默认 Skill 绑定），来源为后端单一事实来源 */
  defaults: () => api.get('/agent-configs/defaults'),
  get: (id: number) => api.get(`/agent-configs/${id}`),
  create: (d: any) => api.post('/agent-configs', d),
  update: (id: number, d: any) => api.put(`/agent-configs/${id}`, d),
  del: (id: number) => api.del(`/agent-configs/${id}`),
}

export const skillsAPI = {
  list: () => api.get('/skills'),
  /** 侧栏元信息：主流程 Agent 列表 + 外部技能库列表（均由后端目录结构推导，前端不硬编码） */
  meta: () => api.get('/skills/meta'),
  get: (id: string) => api.get(`/skills/${id}`),
  create: (data: { id: string; name: string; description?: string }) => api.post('/skills', data),
  update: (id: string, content: string) => api.put(`/skills/${id}`, { content }),
  del: (id: string) => api.del(`/skills/${id}`),
}

export const voicesAPI = {
  list: (provider?: string) => api.get(`/ai-voices${provider ? `?provider=${provider}` : ''}`),
  sync: () => api.post('/ai-voices/sync', {}),
  preview: (voiceId: string, text?: string) => api.post('/ai-voices/preview', { voice_id: voiceId, text }),
  clone: (formData: FormData) => reqForm('/ai-voices/clone', formData),
  generateFromCharacters: (dramaId: number) => api.post('/ai-voices/generate-from-characters', { drama_id: dramaId }),
}

export const aiProvidersAPI = {
  list: () => api.get('/ai-providers'),
}

async function uploadFile(file: File, endpoint: string): Promise<{ url: string; path: string }> {
  const fd = new FormData()
  fd.append('file', file)
  const resp = await fetch(`${apiBase()}${endpoint}`, { method: 'POST', body: fd })
  const json = await resp.json()
  if (!resp.ok || (json.code && json.code >= 400)) {
    throw new Error(json.message || `上传失败 (${resp.status})`)
  }
  return (json.data ?? json) as { url: string; path: string }
}

export const uploadAPI = {
  image: (file: File) => uploadFile(file, '/upload/image'),
  audio: (file: File) => uploadFile(file, '/upload/audio'),
  video: (file: File) => uploadFile(file, '/upload/video'),
}

// ====== 资源库 API ======

export const characterLibraryAPI = {
  list: (params?: Record<string, any>) => {
    const q = new URLSearchParams()
    if (params) Object.entries(params).forEach(([k, v]) => v != null && v !== '' && q.set(k, String(v)))
    return api.get(`/character-library${q.size ? `?${q}` : ''}`)
  },
  get: (id: number) => api.get(`/character-library/${id}`),
  create: (data: any) => api.post('/character-library', data),
  update: (id: number, data: any) => api.put(`/character-library/${id}`, data),
  del: (id: number) => api.del(`/character-library/${id}`),
  batchDelete: (ids: number[]) => api.post('/character-library/batch-delete', { ids }),
  categories: () => api.get<string[]>('/character-library/categories'),
  tags: () => api.get<string[]>('/character-library/tags'),
  apply: (id: number, dramaId: number) => api.post(`/character-library/${id}/apply`, { dramaId }),
  fromCharacter: (characterId: number) => api.post(`/character-library/from-character/${characterId}`),
}

export const sceneLibraryAPI = {
  list: (params?: Record<string, any>) => {
    const q = new URLSearchParams()
    if (params) Object.entries(params).forEach(([k, v]) => v != null && v !== '' && q.set(k, String(v)))
    return api.get(`/scene-library${q.size ? `?${q}` : ''}`)
  },
  get: (id: number) => api.get(`/scene-library/${id}`),
  create: (data: any) => api.post('/scene-library', data),
  update: (id: number, data: any) => api.put(`/scene-library/${id}`, data),
  del: (id: number) => api.del(`/scene-library/${id}`),
  batchDelete: (ids: number[]) => api.post('/scene-library/batch-delete', { ids }),
  categories: () => api.get<string[]>('/scene-library/categories'),
  filterOptions: () => api.get('/scene-library/filter-options'),
  tags: () => api.get<string[]>('/scene-library/tags'),
  apply: (id: number, dramaId: number, episodeId?: number) => api.post(`/scene-library/${id}/apply`, { dramaId, episodeId }),
  fromScene: (sceneId: number) => api.post(`/scene-library/from-scene/${sceneId}`),
}

export const weaponLibraryAPI = {
  list: (params?: Record<string, any>) => {
    const q = new URLSearchParams()
    if (params) Object.entries(params).forEach(([k, v]) => v != null && v !== '' && q.set(k, String(v)))
    return api.get(`/weapon-library${q.size ? `?${q}` : ''}`)
  },
  get: (id: number) => api.get(`/weapon-library/${id}`),
  create: (data: any) => api.post('/weapon-library', data),
  update: (id: number, data: any) => api.put(`/weapon-library/${id}`, data),
  del: (id: number) => api.del(`/weapon-library/${id}`),
  batchDelete: (ids: number[]) => api.post('/weapon-library/batch-delete', { ids }),
  categories: () => api.get<string[]>('/weapon-library/categories'),
  filterOptions: () => api.get('/weapon-library/filter-options'),
  tags: () => api.get<string[]>('/weapon-library/tags'),
  fromCharacter: (characterId: number, data?: any) => api.post(`/weapon-library/from-character/${characterId}`, data),
}

export const costumeLibraryAPI = {
  list: (params?: Record<string, any>) => {
    const q = new URLSearchParams()
    if (params) Object.entries(params).forEach(([k, v]) => v != null && v !== '' && q.set(k, String(v)))
    return api.get(`/costume-library${q.size ? `?${q}` : ''}`)
  },
  get: (id: number) => api.get(`/costume-library/${id}`),
  create: (data: any) => api.post('/costume-library', data),
  update: (id: number, data: any) => api.put(`/costume-library/${id}`, data),
  del: (id: number) => api.del(`/costume-library/${id}`),
  batchDelete: (ids: number[]) => api.post('/costume-library/batch-delete', { ids }),
  categories: () => api.get<string[]>('/costume-library/categories'),
  filterOptions: () => api.get('/costume-library/filter-options'),
  tags: () => api.get<string[]>('/costume-library/tags'),
  fromCharacter: (characterId: number, data?: any) => api.post(`/costume-library/from-character/${characterId}`, data),
}

export const propsLibraryAPI = {
  list: (params?: { drama_id?: number }) => {
    const q = new URLSearchParams()
    if (params?.drama_id) q.set('drama_id', String(params.drama_id))
    return api.get(`/props${q.size ? `?${q}` : ''}`)
  },
  get: (id: number) => api.get(`/props/${id}`),
  create: (data: any) => api.post('/props', data),
  update: (id: number, data: any) => api.put(`/props/${id}`, data),
  del: (id: number) => api.del(`/props/${id}`),
  generateImage: (id: number, data?: { prompt?: string; config_id?: number }) => api.post(`/props/${id}/generate-image`, data || {}),
}

export const presetsAPI = {
  list: (type?: string) => api.get(`/presets${type ? `?type=${type}` : ''}`),
  create: (data: { type: string; name: string; config?: any }) => api.post('/presets', data),
  update: (id: number, data: any) => api.put(`/presets/${id}`, data),
  del: (id: number) => api.del(`/presets/${id}`),
}

// ====== 预设框架 API（Variation Card → 建剧 → 批量首帧 → 批量视频） ======
//
// 后端 `app/routers/preset_framework.py` 的 6 条端点，此前**前端零调用**（整域只有接口没有入口）。
//
// ⚠️ 本域响应形态与全项目都不同：成功 `{success:true,data}`、失败 `{success:false,msg}`
//    （不是 `message`）——`req()` 的错误分支已补 `json.msg` 兜底。
// ⚠️ Card 里的 `THEME_FAMILY_A` / `ACTIVITY_1` 这类是后端**占位数据池**
//    （原 Node 实现即如此，等具体领域知识替换）⇒ 界面上要如实说明，别当乱码去"修"。
// ⚠️ 5 镜固定：`generate_variation_card()` 恒出 5 条，前端不许假设别的条数。

export interface VariationShot {
  shotIndex: number
  themeFamily?: string
  compositionPattern: string
  spaceType: string
  foregroundFrame: string
  mainFocalPoint: string
  thematicClue: string
  activity: string
  cameraPosition: string
  windDirection: string
  lightStructure: string
  characterLayout: string
  livingElement: string | null
}

export interface VariationCard {
  themeFamily: string
  shots: VariationShot[]
}

export interface PresetFrameworkStatus {
  drama: { id: number; title: string; style: string | null }
  episode: { id: number; episodeNumber: number } | null
  storyboards: {
    id: number
    storyboardNumber: number | null
    imageUrl: string | null
    videoUrl: string | null
    status: string
  }[]
  variationCard: VariationCard | null
  summary: { totalShots: number; imagesGenerated: number; videosGenerated: number }
}

export interface PresetFullPipelineResult {
  dramaId: number
  episodeId: number
  storyboardIds: number[]
  imageGenIds: number[]
  variationCard: VariationCard
}

export const presetFrameworkAPI = {
  /** 随机换一张卡；`excludeFamily` 挡掉上一张的家族（连点「换一张」不会撞同一个） */
  variationCard: (excludeFamily?: string) =>
    api.get<VariationCard>(
      `/preset/framework/variation-card${excludeFamily ? `?excludeFamily=${encodeURIComponent(excludeFamily)}` : ''}`,
    ),
  /** 只建结构：Drama + 第 1 集 + N 条分镜（不生图） */
  create: (d: { title: string; description?: string; variationCard: VariationCard }) =>
    api.post<{ dramaId: number; episodeId: number; storyboardIds: number[] }>('/preset/framework/create', d),
  /** 批量生首帧（并发；**失败不中断**，失败的镜头状态置 image_failed） */
  generateImages: (d: { dramaId: number; episodeId: number; storyboardIds: number[]; variationCard: VariationCard }) =>
    api.post<{ dramaId: number; episodeId: number; imageGenIds: number[] }>('/preset/framework/generate-images', d),
  /** 批量生视频（图生视频；**没有首帧的镜头会被跳过**，不计入 videoGenIds） */
  generateVideos: (d: { dramaId: number; episodeId: number; storyboardIds: number[]; variationCard: VariationCard }) =>
    api.post<{ dramaId: number; episodeId: number; videoGenIds: number[] }>('/preset/framework/generate-videos', d),
  /** 管线状态与进度汇总（每镜首帧/视频/状态） */
  status: (dramaId: number) => api.get<PresetFrameworkStatus>(`/preset/framework/status/${dramaId}`),
  /** 一键：建剧 + 批量生首帧（`autoGenerateImages` 默认 true，**只有显式 false 才跳过**） */
  fullPipeline: (d: {
    title: string
    description?: string
    variationCard: VariationCard
    autoGenerateImages?: boolean
  }) => api.post<PresetFullPipelineResult>('/preset/framework/full-pipeline', d),
}

// ====== 全自动管线 API（一句话 → 整集短剧） ======

export interface AutoPipelineStatus {
  dramaId: number
  title: string
  status: string
  totalEpisodes: number
  doneCount: number
  failedCount: number
  running: boolean
  episodes: {
    id: number
    episodeNumber: number
    status: string
    hasScript: boolean
    hasVideo: boolean
    storyboardCount: number
    characterCount: number
    sceneCount: number
    imageReadyCount: number
    videoReadyCount: number
    composedCount: number
  }[]
}

export const autoPipelineAPI = {
  run: (d: any) => api.post<{ dramaId: number; episodeIds: number[] }>('/auto-pipeline/run', d),
  status: (dramaId: number) => api.get<AutoPipelineStatus>(`/auto-pipeline/status/${dramaId}`),
  resume: (dramaId: number, override?: any) => api.post(`/auto-pipeline/resume/${dramaId}`, override),
  /** SSE 进度流地址（供 EventSource 订阅，替代轮询） */
  streamUrl: (dramaId: number) => `${apiBase()}/auto-pipeline/stream/${dramaId}`,
}

export interface TokenStats {
  totalInputTokens: number
  totalOutputTokens: number
  totalTokens: number
  runs: number
  byScope: { scope: string; runs: number; inputTokens: number; outputTokens: number; totalTokens: number }[]
}

export const traceAPI = {
  list: (scope?: string) => api.get(scope ? `/traces?scope=${scope}` : '/traces'),
  stats: () => api.get<TokenStats>('/traces/stats'),
}

// ====== 数据存储 API（数据根目录查询 / 切换） ======

export interface StorageInfo {
  dataRoot: string
  dbPath: string
  storagePath: string
  dbExists: boolean
  storageExists: boolean
  dbSizeBytes: number
  storageSizeBytes: number
}

export const storageAPI = {
  info: () => api.get<StorageInfo>('/storage/info'),
  change: (path: string, migrate = true) => api.post<StorageInfo>('/storage/change', { path, migrate }),
}

// ====== 用量统计 + 成本预估（花后核算 / 花前估算） ======

export interface UsageRecord {
  id: number
  service_type: string
  provider: string
  model: string
  drama_id: number | null
  episode_id: number | null
  storyboard_id: number | null
  units: number | null
  cost_amount: number | null
  is_local: boolean
  status: string
  retry_count: number
  created_at: string
}

export interface UsageSummary {
  totalCount: number
  totalCost: number
  checkedCost: boolean
  byService: Array<{ service_type: string; count: number; cost: number }>
  byProvider: Array<{ provider: string; count: number; cost: number }>
  byDay: Array<{ date: string; count: number; cost: number }>
  records: UsageRecord[]
}

export interface EpisodeCostBoard {
  drama_id: number
  total_cost: number
  total_calls: number
  retry_cost: number
  episodes: Array<{
    episode_id: number
    episode_number: number
    title: string | null
    total_cost: number
    total_calls: number
    retry_cost: number
    by_service: Array<{ service_type: string; count: number; cost: number }>
  }>
}

export interface EstimateResult {
  drama_id: number
  episode_id?: number
  scope: string
  generated_at: string
  pending_images: number
  pending_videos: number
  pending_audio_chars: number
  active_image: { provider: string; model: string } | null
  active_video: { provider: string; model: string } | null
  active_audio: { provider: string; model: string } | null
  items: Array<{ service_type: string; provider: string; model: string; units: number; unit: string; cost: number | null }>
  total_cost: number | null
  unestimatable: string[]
}

export const usageAPI = {
  /** 用量汇总：总量/总成本/按服务/按提供商/按天/最近明细 */
  summary: (params?: { drama_id?: number; episode_id?: number; limit?: number }) => {
    const q = new URLSearchParams()
    if (params?.drama_id) q.set('drama_id', String(params.drama_id))
    if (params?.episode_id) q.set('episode_id', String(params.episode_id))
    if (params?.limit) q.set('limit', String(params.limit))
    return api.get<UsageSummary>(`/usage/summary${q.size ? `?${q}` : ''}`)
  },
  /** 多集成本看板：每集总成本/总调用/重拍成本/按服务拆分 */
  board: (dramaId: number) => api.get<EpisodeCostBoard>(`/usage/board?drama_id=${dramaId}`),
  /** 生成前费用预估：当前激活单价 × 待生成工作量 */
  estimate: (params: { drama_id: number; episode_id?: number; storyboard_ids?: number[] }) => {
    const q = new URLSearchParams()
    q.set('drama_id', String(params.drama_id))
    if (params.episode_id) q.set('episode_id', String(params.episode_id))
    if (params.storyboard_ids?.length) q.set('storyboard_ids', params.storyboard_ids.join(','))
    return api.get<EstimateResult>(`/usage/estimate?${q}`)
  },
}

// ====== 资产版本历史 / 回滚 ======

export interface AssetVersion {
  id: number
  asset_type: string
  asset_id: number
  media_type: string | null
  frame_type: string | null
  version: number
  asset_url: string | null
  provider: string | null
  model: string | null
  prompt: string | null
  generation_id: number | null
  status: string
  created_at: string
}

export const assetVersionsAPI = {
  list: (assetType: string, assetId: number) =>
    api.get<{ asset_type: string; asset_id: number; versions: AssetVersion[] }>(
      `/asset-versions?asset_type=${encodeURIComponent(assetType)}&asset_id=${assetId}`,
    ),
  activate: (versionId: number) => api.post(`/asset-versions/${versionId}/activate`),
}

// ====== 风格 Profile（house style 提炼） ======

export interface StyleProfile {
  id: number
  drama_id: number | null
  name: string
  description: string | null
  source: string | null
  storytelling: string | null
  shot_patterns: string | null
  audio_captions: string | null
  qc_rules: string | null
  facts: string | null
  inferences: string | null
  preferences: string | null
  is_active: boolean
  created_at: string
  updated_at: string
}

// 生产体检：**开跑前**的一条链（占位符 → 质感 → 连续性 → 验收门 ✓）。
// ⚠️ 后端只要 `episodeId` 就**自己取数组装** ✓ ⇒ 前端**不必**知道 plan/manifest 的形状 ✓
//    （这正是第 88→90 步接出来的接口 ✓：调用方只给 id ✓）。
export const productionAPI = {
  /** 按集体检 ✓ 返回 { ready, blockers, warnings, nextActions, sections, source } */
  preflightForEpisode: (episodeId: number) =>
    api.post('/production/preflight', { episodeId }),
  /** 显式 payload（调试/外部用 ✓）；一般用上面那个 ✓ */
  preflight: (payload: Record<string, any>) => api.post('/production/preflight', payload),
  /** 体检接受哪些段、按什么顺序跑 ✓ */
  schema: () => api.get('/production/preflight/schema'),
  /**
   * **机器侧**就绪（依赖 / 权重 / 显存 ✓ 2026-09-21 新增 ✓）
   *
   * ⚠️ 与 `preflightForEpisode` 不是一回事 ✗：那个查**内容/结构** ✓，这个查
   * 「**这台机器能不能跑**」✓ ⇒ 两个 `ready` 各自独立 ✓，别在界面上混成一个 ✗。
   * ⚠️ 真权重预检（键名/形状核对）**不在这个端点** ✗（那要传文件路径 ✓ ⇒ 等于开放任意路径读取 ✗）
   * ⇒ 后端响应里的 `hint` 会指明用 CLI ✓。
   */
  // ⚠️⚠️ `api.get` **只收一个路径参数** ✗（签名见上面的 `get: <T>(p: string)` ✓）——
  //     2026-09-21 打开类型检查时才发现这里曾经**多传了一个查询对象** ✓✗ ⇒ 第二个参数被
  //     **静默忽略** ✗ ⇒ `stage` 从来没发出去过 ✓✗（后端只会用默认档 ✓，界面看着"正常" ✓）。
  //     ⇒ 查询串按本文件既有写法**拼进路径** ✓（与 `imageAPI.list` 同口径 ✓）。
  //     ⚠️ 注释里**不要写调用形状** ✗（`tests/frontend_api_coverage_test.py` 会把**注释里的**
  //     `api.<method>('…')` 当**真调用点**抽出来反查后端路由 ✓✗ ⇒ 一条不存在的"路径"会让它红 ✓）。
  engineReadiness: (stage: string = 'h3') =>
    api.get(`/production/engine-readiness?stage=${encodeURIComponent(stage)}`),
}

// 提示词生产工具：**纯函数、毫秒级**（不落库、不调模型）⇒ 可在用户编辑时**实时**调 ✓。
// ⚠️ 返回形状按后端真键写（勿猜）：resolve → { text, ok, used, unresolved, durationsMs,
//    residualTags, bareIds, problems }；polish → { sections, text, ok, issues, vagueTerms, inserted }
export const promptToolsAPI = {
  /** 占位符 → **具体内容**；`ok=false` ⇒ 还不能直接送模型 ✗（原因看 `problems`） */
  resolve: (payload: {
    text: string
    maps?: Record<string, Record<string, string>>
    durationTemplate?: string
    flagBareIds?: boolean
  }) => api.post('/prompts/resolve', payload),
  /** Mx-Shell **五段质感层**（字段一律 camelCase）；`issues` = 哪一段约束不够硬 */
  polish: (payload: Record<string, any>) => api.post('/prompts/polish', payload),
}

/** 知识库：目录树 / 文档清单 / 检索 / 翻原文（**只读**，后端故意不提供上传与重建索引） */
export interface KnowledgeGroup {
  path: string
  name: string
  documents: number
  withText: number
  /** 整本跑完的 OCR 份数（只认 sidecar 的 status='ocr'） */
  ocr: number
  /** 有 OCR 记录但只做了前几页；**计入 pending**，这里单列只为让它别消失 */
  ocrPartial: number
  pending: number
  bytes: number
  pages: number
}
export interface KnowledgeTree {
  available: boolean
  hint: string | null
  depth: number
  totals: Omit<KnowledgeGroup, 'path' | 'name'> | null
  groups: KnowledgeGroup[]
}
export interface KnowledgeDoc {
  name: string
  path: string
  size: number
  pages: number
  withText: boolean
  ocr: boolean
  /** 半成品：有 sidecar 记录但只 OCR 了前几页（仍算 pending） */
  partial: boolean
  /** 本仓补录库里的件（上传进来的）—— 与供应商/OCR 的件能一眼分开 */
  extra?: boolean
  /** 归哪个库：`curated` 供应商 / `ocr` 补出来的 / `extra` 本仓补录 */
  source?: 'curated' | 'ocr' | 'extra'
  /** 有覆盖层改动的块（清单上打「已修改」） */
  edited?: boolean
  /** 在回收站（只有 `only=buried` 才会出现） */
  buried?: boolean
}
export interface KnowledgeDocs {
  available: boolean
  hint: string | null
  category: string | null
  only: string
  total: number
  offset: number
  limit: number
  items: KnowledgeDoc[]
}
export interface KnowledgeHit {
  title: string
  path: string
  page: number | null
  chunkNo: number | null
  score: number
  excerpt: string
  from?: 'curated' | 'ocr'
  source: 'methods' | 'playbook'
  onDisk?: boolean
  /**
   * ⚠️ **同一个字段名，两条源两种语义**（2026-11-06 实测）：
   * - 剧本库命中（`source:"methods"`）：仅 `includePartial=true` 才可能带对象
   *   `{ pages, chunkPages }` ＝ 总页数 / 真有正文的页数；
   * - 方法库命中（`source:"playbook"`）：恒为 **0–1 的数字**，是「命中了多少查询词」，
   *   也就是方法库的排序主键。
   * 直接 `hit.coverage.pages` 用在数字上会渲染出 `undefined/undefined`，务必先判类型。
   */
  coverage?: { pages: number; chunkPages: number } | number | null
}
export interface KnowledgeSearch {
  query: string
  results: KnowledgeHit[]
  sources: Record<string, any>
  fusion?: any
  hint?: string
}
/** `/knowledge/facets`：六个目录维度的可选值（值 → 条数），别让调用方猜词 */
export interface KnowledgeFacets {
  available: boolean
  hint?: string | null
  entries?: number
  dimensions: Record<string, Record<string, number>>
  /**
   * 方法库的技能目录名 → 小节数（就是 `skill=` 的取值，会随人加目录而变 ⇒ 由后端下发）。
   * **是对象不是数组**：后端 `playbook_skills()` 返回 `dict[str, int]`，
   * 当成数组 `v-for` 会去遍历「值」，渲染出几个数字下拉还查不到东西。
   */
  playbookSkills: Record<string, number>
}
/**
 * `/knowledge/document/state`：**按钮该不该亮**。
 *
 * 界面先在打开文档时问一次，而不是「画满按钮、让人点了才报错」：
 * - `editable` 恒 true（正文可改，改动落覆盖层，原库不动）；
 * - `hardDeletable` 只对「本仓上传、且文件还在盘上」的件成立（其余只能进回收站）；
 * - `buried` = 在回收站 ⇒ 只该有「还原」，不该再有「删除」。
 */
export interface KnowledgeDocState {
  found: boolean
  reason?: string
  path?: string
  name?: string
  source?: 'curated' | 'ocr' | 'extra'
  chunks?: number
  onDisk?: boolean
  editable?: boolean
  deletable?: boolean
  hardDeletable?: boolean
  uploaded?: boolean
  buried?: boolean
  edits?: number
}
/** 覆盖层盘点（`/knowledge/overlay`）＋ 上传口径（同一响应下发，前端不再抄一份） */
export interface KnowledgeOverlay {
  available: boolean
  hint?: string | null
  edits: number
  tombstones: number
  uploads: number
  writableSources: string[]
  /** 已改清单（最近 50 条） */
  edited?: Array<{ source: string; path: string; chunkNo: number; chars: number; note?: string | null; updatedNs: number }>
  /** 回收站（最近 50 条）—— 还原用 `path`，`chunkNo` 不参与 */
  buried?: Array<{ source: string; path: string; reason?: string | null; buriedNs: number }>
  recentUploads?: Array<{ path: string; name: string; size: number; mime?: string; status: string; reason?: string | null;
    documents?: number; chunks?: number; createdNs?: number }>
  upload: {
    accept: string[]
    label: string
    maxBytes: number
    maxFiles: number
    dir: string
  }
}
/**
 * 上传一份的结果。**三档不许合并成「失败」**：
 * - `ingested` 正文抽出来了，立刻能搜;
 * - `quarantined` **文件已在盘上**，只是抽不出正文（扫描件/图片型）⇒ 要走 OCR；
 * - `rejected` 连落盘都没做（类型不支持/太大/空文件）。
 */
export interface KnowledgeUploadResult {
  ok: boolean
  status: 'ingested' | 'quarantined' | 'rejected'
  name: string
  path: string | null
  size: number
  chunks?: number
  pages?: number
  chars?: number
  sha256?: string
  engine?: string
  printable?: number
  reason?: string | null
  seconds?: number
}
export interface KnowledgeUploadBatch {
  results: KnowledgeUploadResult[]
  total: number
  ingested: number
  quarantined: number
  rejected: number
  allowed: string
  maxSize: number
  maxFiles: number
  hint?: string
}
/** 一场。`confidence` 是解析器自评（0–1），`noSource` 说明 `sceneNo` 为什么是空的 */
export interface KnowledgeScene {
  sceneNo: number | null
  noSource?: string | null
  intExt: string | null
  location: string | null
  time: string | null
  cast: string[]
  opening: string | null
  coreEvent: string | null
  sceneTail: string | null
  pageFrom: number | null
  pageTo: number | null
  chars: number | null
  confidence: number | null
  heading: string | null
}
/** `/knowledge/scenes`：三种回执必须能区分（索引没建 / 这份没解析过 / 解析过但零场次） */
export interface KnowledgeScenes {
  available: boolean
  hint: string | null
  document: {
    path: string
    scenes: number
    strategy: string
    confidence: number | null
    chars: number | null
    status: string
    hint?: string | null
    updatedAt?: string
  } | null
  scenes: KnowledgeScene[]
  total: number
  page: number
  limit: number
}

export const knowledgeAPI = {
  report: () => api.get<any>('/knowledge/report'),
  tree: (depth = 2) => api.get<KnowledgeTree>(`/knowledge/tree?depth=${depth}`),
  documents: (p: { category?: string; q?: string; only?: string; offset?: number; limit?: number }) => {
    const qs = new URLSearchParams()
    if (p.category) qs.set('category', p.category)
    if (p.q) qs.set('q', p.q)
    if (p.only && p.only !== 'all') qs.set('only', p.only)
    qs.set('offset', String(p.offset ?? 0))
    qs.set('limit', String(p.limit ?? 30))
    return api.get<KnowledgeDocs>(`/knowledge/documents?${qs.toString()}`)
  },
  /**
   * 检索。六个目录维度**只筛剧本库**、`skill` **只筛方法库** —— 这是后端的硬边界，
   * 给了不适用的一侧会回 `filtersApplied=false`（页面据此提示「条件没生效」，
   * 不能假装筛过了）。`includePartial` 才会把「只 OCR 了前几页」的半成品带出来。
   */
  search: (p: {
    q: string
    source?: string
    limit?: number
    path?: string
    country?: string
    genre?: string
    category?: string
    docType?: string
    permission?: string
    festival?: string
    skill?: string
    includePartial?: boolean
  }) => {
    const qs = new URLSearchParams({ q: p.q, limit: String(p.limit ?? 10) })
    if (p.source && p.source !== 'all') qs.set('source', p.source)
    if (p.path) qs.set('path', p.path)
    for (const key of ['country', 'genre', 'category', 'permission', 'festival'] as const) {
      if (p[key]) qs.set(key, p[key] as string)
    }
    //: 后端参数名是 `docType`（camelCase），URLSearchParams 里也得原样写，别顺手 snake 化
    if (p.docType) qs.set('docType', p.docType)
    if (p.skill) qs.set('skill', p.skill)
    if (p.includePartial) qs.set('includePartial', 'true')
    return api.get<KnowledgeSearch>(`/knowledge/search?${qs.toString()}`)
  },
  document: (path: string, offset = 0, limit = 20) =>
    api.get<any>(`/knowledge/document?path=${encodeURIComponent(path)}&offset=${offset}&limit=${limit}`),
  facets: () => api.get<KnowledgeFacets>('/knowledge/facets'),
  /** 只读回派生好的场次表；没建索引 / 这份没解析过都走 `available=false` + `hint` */
  scenes: (path: string, page = 1, limit = 200) =>
    api.get<KnowledgeScenes>(
      `/knowledge/scenes?path=${encodeURIComponent(path)}&page=${page}&limit=${limit}`,
    ),
  /** 打开文档时问一次「按钮该不该亮」（可编辑 / 可删 / 可彻底删 / 在回收站） */
  state: (path: string) =>
    api.get<KnowledgeDocState>(`/knowledge/document/state?path=${encodeURIComponent(path)}`),
  /**
   * 改一块正文（`chunkNo = -1` 改标题）。
   *
   * 必须走 **JSON body**：一块正文可上千字，塞进 query 会被各级代理的 URL 上限截断，
   * 症状是「提示保存成功、但后半段没了」—— 这种错最难查，所以后端直接把 query 版删掉。
   */
  saveChunk: (path: string, chunkNo: number, text: string, note?: string) =>
    api.post<{ ok: boolean; reason?: string; chunkNo?: number; source?: string }>(
      '/knowledge/document/chunk',
      { path, chunkNo, text, note: note || undefined },
    ),
  /** 撤销某块改动（回到原库正文） */
  undoChunk: (path: string, chunkNo: number) =>
    api.del(
      `/knowledge/document/chunk?path=${encodeURIComponent(path)}&chunkNo=${chunkNo}`,
    ),
  /** 删除一份：默认进回收站（可还原）；`hard` 只对本仓上传件有效 */
  bury: (path: string, hard = false, reason?: string) =>
    api.del(
      `/knowledge/document?path=${encodeURIComponent(path)}`
      + (hard ? '&hard=true' : '')
      + (reason ? `&reason=${encodeURIComponent(reason)}` : ''),
    ),
  restore: (path: string) =>
    api.post(`/knowledge/document/restore?path=${encodeURIComponent(path)}`),
  overlay: () => api.get<KnowledgeOverlay>('/knowledge/overlay'),
  /** 上传语料（补录库），回执按 `ingested/quarantined/rejected` 三档分别给结果 */
  upload: (files: File[], onProgress?: (pct: number) => void) =>
    uploadFiles<KnowledgeUploadBatch>('/knowledge/upload', files, onProgress),
}

// ── 知识闭环（`/api/v1/knowledge-loop/*` ＋成片评测）─────────────────────────
//
// 与上面的 `knowledgeAPI`（`/knowledge/*`：语料库的只读钻取）是**两套对象**，别混：
// 这一族是可写台账，写口一律带 `operator`/`role`（审计行自洽）与 `basis`/`evidence`
// （可归因），缺凭据后端直接拒（400/403/409/422），前端**不替它补默认值**。
// 契约真源：`backend-py/app/routers/knowledge_loop.py`；说明：`docs/knowledge-loop.md`。

/** 台账一条。派生字段（`tagList`/`version`/`deleted*`）由服务端算，前端不自己拆标签 */
export interface KnowledgeEntry {
  id: string
  kind: string
  scope: string
  title: string
  body: string
  tags: string
  tagList: string[]
  evidence: string[]
  confidence: number
  status: string
  support: number
  contra: number
  lift: number | null
  created_ns: number
  updated_ns: number
  version: number
  deleted: boolean
  deletedNs: number
  deletedBy: string
  source_kind?: string | null
  source_ref?: string | null
}

/** 详情＝条目＋挂载关系＋用过几次（`okRate` 是注入后结果占比，**不是 lift**） */
export interface KnowledgeEntryDetail extends KnowledgeEntry {
  links: Array<{ refKind: string; refId: string }>
  usage: Array<{ runId: string; usedNs: number; outcome: string }>
  usageSummary: {
    ok: number
    failed: number
    unknown: number
    settled: number
    okRate: number | null
  }
}

export interface KnowledgeEntryList {
  total: number
  items: KnowledgeEntry[]
  limit: number
  offset: number
  available: boolean
  hint?: string
}

/** 邻居条目的摘要：不给全文（列表里塞全文会把抽屉撑到几百 KB），但**必须给头 120 字** */
export interface KnowledgeRelatedBrief {
  id: string
  kind: string
  scope: string
  title: string
  status: string
  confidence: number
  support: number
  contra: number
  lift: number | null
  tags: string[]
  version: number
  deleted: boolean
  updatedNs: number
  snippet: string
}

/**
 * 邻居一项。`via` 两档**证据强度不同，不能混读**：
 * - `shared`：共用同一 `stage`/`shot`（有事实依据，`score` 恒为 `null`）；
 * - `similar`：正文 trigram 相似（启发式，只作提示，**不是合并依据**）。
 */
export interface KnowledgeRelatedItem {
  via: 'shared' | 'similar'
  refKind: string
  refId: string
  score: number | null
  entry: KnowledgeRelatedBrief
}

/** `GET /entries/{id}/related`：`found=false` 是「没这条」，不是「库坏了」 */
export interface KnowledgeRelated {
  id: string
  found: boolean
  deleted: boolean
  available: boolean
  links: Array<{ refKind: string; refId: string }>
  items: KnowledgeRelatedItem[]
  counts: { shared: number; similar: number }
  hint?: string
}

/** 版本列表只给元信息（不带正文：一页十条正文会把响应撑到几十 KB） */
export interface KnowledgeEntryVersion {
  version: number
  title: string
  kind: string
  scope: string
  tags: string[]
  confidence: number
  status: string
  editor: string
  reason: string
  bodyLen: number
  createdNs: number
}

export interface KnowledgeEntryVersions {
  id: string
  current: number
  deleted: boolean
  items: KnowledgeEntryVersion[]
  available: boolean
  hint?: string
}

export interface KnowledgeEntryVersionDetail {
  id: string
  version: number
  kind: string
  scope: string
  title: string
  body: string
  bodyLen: number
  tags: string[]
  evidence: string[]
  confidence: number
  status: string
  editor: string
  reason: string
  createdNs: number
  isCurrent: boolean
}

export interface KnowledgeAuditItem {
  id: number
  entryId: string
  action: string
  operator: string
  role: string
  atNs: number
  detail: string
}

export interface KnowledgeAuditList {
  total: number
  items: KnowledgeAuditItem[]
  limit: number
  offset: number
  available: boolean
  hint?: string
}

export interface KnowledgeLedgerStats {
  total: number
  deleted?: number
  recent7d?: number
  accepted?: number
  proposed?: number
  byKindStatus?: Array<{ kind: string; status: string; count: number }>
  byScope?: Array<{ scope: string; count: number }>
  available: boolean
  hint?: string
}

/** 词表闭集（kind/scope/status 的中文名与计数都由后端下发，前端不硬编码第二份） */
export interface KnowledgeTaxonomy {
  kinds: Array<{ kind: string; label: string; count: number }>
  scopes: Array<{ scope: string; count: number }>
  statuses: Array<{ status: string; count: number }>
  tags: Array<{ tag: string; count: number }>
  tagTotal: number
  available: boolean
  hint?: string
}

export interface KnowledgeLoopSearch {
  query: string
  ranking: string
  total: number
  items: KnowledgeEntry[]
  limit: number
  offset?: number
  available: boolean
  totalCapped?: boolean
  hint?: string
}

/** 调参台账的一行：`value` 为 null 表示「没设过，走 default」 */
export interface KnowledgeTunable {
  stage: string
  field: string
  default: any
  value: any
  source: string
  basis?: string
  evidence?: string[]
  updatedAt?: string
  note?: string
}

/** 蒸馏/调参候选：`canApply=false` 的会带 `why`（拒的理由），照实显示，别只显示前半句 */
export interface KnowledgeCandidate {
  id?: string
  key?: string
  pattern?: string
  rule?: string
  kind?: string
  stage?: string
  field?: string
  current?: any
  proposed?: any
  why?: string
  evidence?: string[]
  samples?: number
  stats?: Record<string, any>
  gate?: { samples?: string; needed?: string; [key: string]: any }
  canApply?: boolean
  qcGate?: { verdict: string; [key: string]: any }
}

/** 写口回执：`ok=false` 时 `why` 是后端原话（HTTP 层通常已抛，这里是兜底的软失败） */
export interface KnowledgeWriteResult {
  ok?: boolean
  /** 机器码（`conflict`/`evidence`/`invalid`…）；HTTP 层已分派成 400/403/409/422 */
  code?: string
  why?: string
  id?: string
  version?: number
  entry?: KnowledgeEntry
  /**
   * 「真的有变化吗」——**两种形态**，别当同一种读：
   * `PATCH` 给的是改动字段名数组（`[]` ＝ 合法但没变），删/恢复/回滚给的是 bool。
   */
  changed?: boolean | string[]
  /** `POST /entries` 回执：`created=false` 表示命中去重合并到既有条目，不是新建 */
  created?: boolean
  /** 同一条目并发写撞车（`code=conflict` 时另带 `version`＝库里现版本） */
  conflict?: boolean
  support?: number
  hint?: string
}

export interface FilmEvalDimension {
  name: string
  label: string
  score: number | null
  samples: number
  judged: boolean
  notes: string[]
}

/** 成片四维评测（`POST /evaluation/film/{id}`）。`score=null` 表示一维都判不了，**不是 0 分** */
export interface FilmEvalReport {
  episodeId: number
  total: number
  ready: number
  missingShots: number[]
  dimensions: FilmEvalDimension[]
  score: number | null
  verdict: string
  passScore: number
  missing: string[]
  defectCodes: Array<{ code: string; count: number; shots?: number[] }>
  review: {
    hit: number
    falseAlarm: number
    missed: number
    reviewed: number
    hitRate: number | null
  }
  notes: string[]
  recorded: boolean
}

export const knowledgeLoopAPI = {
  /** 闭环总览。`includeCandidates` 才把蒸馏候选一起算（贵：要扫跑次统计） */
  report: (includeCandidates = false) =>
    api.get<any>(`/knowledge-loop/report?includeCandidates=${includeCandidates ? 'true' : 'false'}`),
  /** 定期蒸馏/提升的排期事实（下一次什么时候跑、上次跑成什么） */
  scheduler: () => api.get<any>('/knowledge-loop/scheduler'),
  rebuildRuns: (p: { dryRun?: boolean; limit?: number } = {}) =>
    api.post<any>('/knowledge-loop/runs/rebuild', p),
  /** 蒸馏候选 → `proposed` 条目。`dryRun` 只看不写 */
  distill: (p: { dryRun?: boolean; limit?: number } = {}) =>
    api.post<any>('/knowledge-loop/distill', p),
  refreshLift: (p: { limit?: number } = {}) =>
    api.post<any>('/knowledge-loop/lift/refresh', p),
  optimizations: (p: { agentType?: string; caseId?: string; limit?: number } = {}) =>
    api.get<any>(
      `/knowledge-loop/optimizations?limit=${p.limit ?? 50}`
      + (p.agentType ? `&agentType=${encodeURIComponent(p.agentType)}` : '')
      + (p.caseId ? `&caseId=${encodeURIComponent(p.caseId)}` : ''),
    ),
  taxonomy: (p: { includeDeleted?: boolean } = {}) =>
    api.get<KnowledgeTaxonomy>(
      `/knowledge-loop/taxonomy?includeDeleted=${p.includeDeleted ? 'true' : 'false'}`,
    ),
  entries: (p: {
    kind?: string
    scope?: string
    status?: string
    tags?: string
    includeDeleted?: boolean
    limit?: number
    offset?: number
  } = {}) =>
    api.get<KnowledgeEntryList>(
      `/knowledge-loop/entries?limit=${p.limit ?? 50}&offset=${p.offset ?? 0}`
      + (p.includeDeleted ? '&includeDeleted=true' : '')
      + (p.kind ? `&kind=${encodeURIComponent(p.kind)}` : '')
      + (p.status ? `&status=${encodeURIComponent(p.status)}` : '')
      + (p.scope ? `&scope=${encodeURIComponent(p.scope)}` : '')
      + (p.tags ? `&tags=${encodeURIComponent(p.tags)}` : ''),
    ),
  entry: (id: string, includeDeleted = false) =>
    api.get<KnowledgeEntryDetail>(
      `/knowledge-loop/entries/${encodeURIComponent(id)}`
      + `?includeDeleted=${includeDeleted ? 'true' : 'false'}`,
    ),
  /** 一条知识的邻居：共用同一 stage/shot（有事实依据）＋ 正文相似（启发式，仅提示） */
  relatedEntries: (id: string, p: { limit?: number; includeDeleted?: boolean } = {}) =>
    api.get<KnowledgeRelated>(
      `/knowledge-loop/entries/${encodeURIComponent(id)}/related?limit=${p.limit ?? 10}`
      + (p.includeDeleted ? '&includeDeleted=true' : ''),
    ),
  /** **部分字段**改一条：没出现的字段不动（`null` 与「不传」同义）。清标签要显式传 `tags: ''` */
  updateEntry: (
    id: string,
    p: {
      title?: string
      body?: string
      tags?: string
      evidence?: string[]
      confidence?: number
      reason?: string
      operator?: string
      role?: string
      /** 乐观锁：与库里的 `version` 不符一律 409，防「两个人各改一半」互相覆盖 */
      expectedVersion?: number
    },
  ) => api.patch<KnowledgeWriteResult>(`/knowledge-loop/entries/${encodeURIComponent(id)}`, p),
  /** 默认软删（可还原）；`hard` 是物理删除，必须先软删过，且需要 owner */
  deleteEntry: (
    id: string,
    p: { hard?: boolean; reason?: string; operator?: string; role?: string } = {},
  ) =>
    api.del<KnowledgeWriteResult>(
      `/knowledge-loop/entries/${encodeURIComponent(id)}?hard=${p.hard ? 'true' : 'false'}`
      + (p.reason ? `&reason=${encodeURIComponent(p.reason)}` : '')
      + (p.operator ? `&operator=${encodeURIComponent(p.operator)}` : '')
      + (p.role ? `&role=${encodeURIComponent(p.role)}` : ''),
    ),
  restoreEntry: (id: string, p: { operator?: string; role?: string; reason?: string } = {}) =>
    api.post<KnowledgeWriteResult>(`/knowledge-loop/entries/${encodeURIComponent(id)}/restore`, p),
  entryVersions: (id: string) =>
    api.get<KnowledgeEntryVersions>(`/knowledge-loop/entries/${encodeURIComponent(id)}/versions`),
  entryVersion: (id: string, version: number) =>
    api.get<KnowledgeEntryVersionDetail>(
      `/knowledge-loop/entries/${encodeURIComponent(id)}/versions/${version}`,
    ),
  /** 回滚到某个历史版本（**追加**一个新版本，不抹掉中间的历史） */
  revertEntry: (
    id: string,
    p: { version: number; reason?: string; operator?: string; role?: string },
  ) => api.post<KnowledgeWriteResult>(`/knowledge-loop/entries/${encodeURIComponent(id)}/revert`, p),
  audit: (p: { entryId?: string; action?: string; operator?: string; limit?: number; offset?: number } = {}) =>
    api.get<KnowledgeAuditList>(
      `/knowledge-loop/audit?limit=${p.limit ?? 50}&offset=${p.offset ?? 0}`
      + (p.entryId ? `&entryId=${encodeURIComponent(p.entryId)}` : '')
      + (p.action ? `&action=${encodeURIComponent(p.action)}` : '')
      + (p.operator ? `&operator=${encodeURIComponent(p.operator)}` : ''),
    ),
  /** 状态迁移：后端只有四条边（proposed→accepted/rejected、accepted↔deprecated），越界 400 */
  setEntryStatus: (
    id: string,
    status: string,
    p: { reason?: string; operator?: string; role?: string } = {},
  ) => api.post<KnowledgeWriteResult>(`/knowledge-loop/entries/${encodeURIComponent(id)}/status`, { status, ...p }),
  createEntry: (p: {
    kind: string
    scope?: string
    title: string
    body: string
    tags?: string
    evidence?: string[]
    confidence?: number
    operator?: string
    role?: string
  }) => api.post<KnowledgeWriteResult>('/knowledge-loop/entries', p),
  search: (p: { q: string; kind?: string; status?: string; scope?: string; limit?: number }) =>
    api.get<KnowledgeLoopSearch>(
      `/knowledge-loop/search?q=${encodeURIComponent(p.q)}&limit=${p.limit ?? 10}`
      + (p.kind ? `&kind=${encodeURIComponent(p.kind)}` : '')
      + (p.status ? `&status=${encodeURIComponent(p.status)}` : '')
      + (p.scope ? `&scope=${encodeURIComponent(p.scope)}` : ''),
    ),
  /**
   * 一次喂给某角色的知识包（编译期预算内，超了按分值截断并计数）。
   * `episodeId`/`dramaId` 决定「本集/本剧」作用域能否进来 —— 不传就只剩全局成法，
   * 预览会说"没有知识"，而生产侧其实注入了本集那几条（同一功能两个口径，最易误判）。
   */
  feedPreview: (p: {
    agentType?: string; episodeId?: number; dramaId?: number
    scopes?: string; budgetChars?: number; limit?: number
  } = {}) =>
    api.get<any>(
      `/knowledge-loop/feed/preview?budgetChars=${p.budgetChars ?? 600}&limit=${p.limit ?? 8}`
      + (p.agentType ? `&agentType=${encodeURIComponent(p.agentType)}` : '')
      + (p.episodeId ? `&episodeId=${p.episodeId}` : '')
      + (p.dramaId ? `&dramaId=${p.dramaId}` : '')
      + (p.scopes ? `&scopes=${encodeURIComponent(p.scopes)}` : ''),
    ),
  datasets: () => api.get<any>('/knowledge-loop/datasets'),
  /** L3 导出：`target=runs` 导高分跑次，`target=knowledge` 导「指令 → 做法」对 */
  feedExport: (p: { target: 'runs' | 'knowledge'; limit?: number } = { target: 'knowledge' }) =>
    api.post<any>('/knowledge-loop/feed/export', p),
  /** 把 `datasetRef` 解析成盘上真路径（提交训练前先核一遍，比训练跑完才知道缺文件强） */
  datasetResolve: (ref: string) =>
    api.get<any>(`/knowledge-loop/datasets/resolve?ref=${encodeURIComponent(ref)}`),
  tuning: () => api.get<any>('/knowledge-loop/tuning'),
  tuningSuggest: (minSamples = 30) =>
    api.get<any>(`/knowledge-loop/tuning/suggest?minSamples=${minSamples}`),
  /** 落一条调参（`basis`/`evidence` 必填：没有凭据的「我觉得这个好」会让下次风格变化无从归因） */
  setTuning: (p: {
    stage: string
    field: string
    value: any
    basis: string
    evidence?: string[]
    by?: string
  }) => api.post<any>('/knowledge-loop/tuning', p),
  clearTuning: (p: { stage: string; field: string }) =>
    api.del<any>(
      `/knowledge-loop/tuning?stage=${encodeURIComponent(p.stage)}`
      + `&field=${encodeURIComponent(p.field)}`,
    ),
  qcHitRate: () => api.get<any>('/knowledge-loop/qc/hit-rate'),
  qcPending: (limit = 20) => api.get<any>(`/knowledge-loop/qc/pending?limit=${limit}`),
  /** 录一条复核：`(checkId, code)` 已存在就是**改判**（覆盖，不追加），别当第二条样本 */
  qcReview: (p: {
    checkId: number
    code: string
    verdict: 'hit' | 'false_alarm' | 'missed'
    note?: string
    reviewer?: string
    storyboardId?: number
    videoGenerationId?: number
  }) => api.post<any>('/knowledge-loop/qc/reviews', p),
  deleteQcReview: (id: number) => api.del<any>(`/knowledge-loop/qc/reviews/${id}`),
  training: (p: { limit?: number; status?: string } = {}) =>
    api.get<any>(
      `/knowledge-loop/training?limit=${p.limit ?? 50}`
      + (p.status ? `&status=${encodeURIComponent(p.status)}` : ''),
    ),
  trainingScan: (taskId: string | number) =>
    api.post<any>('/knowledge-loop/training/scan', { taskId: String(taskId) }),
  /** 定为该类默认档——**唯一能改变出图行为的入口**，必带 `basis` 与 `evidence` */
  promoteModel: (id: string, p: { basis: string; evidence?: string[]; by?: string }) =>
    api.post<any>(`/knowledge-loop/training/models/${encodeURIComponent(id)}/promote`, p),
  /** 撤掉默认档（文件与记录都不删：「不用它」不等于「它没存在过」） */
  demoteModel: (id: string, p: { why?: string } = {}) =>
    api.post<any>(`/knowledge-loop/training/models/${encodeURIComponent(id)}/demote`, p),
  /** 成片四维评测。`record=true` 会落一条 `kind=eval` 跑次（同一集重评是**覆盖**） */
  filmEval: (episodeId: number, record = true) =>
    api.post<FilmEvalReport>(
      `/evaluation/film/${episodeId}?record=${record ? 'true' : 'false'}`,
    ),
}

export const styleProfileAPI = {
  list: (dramaId?: number) => api.get<{ profiles: StyleProfile[] }>(`/style-profiles${dramaId ? `?drama_id=${dramaId}` : ''}`),
  get: (id: number) => api.get<{ profile: StyleProfile }>(`/style-profiles/${id}`),
  create: (data: Partial<StyleProfile>) => api.post('/style-profiles', data),
  update: (id: number, data: Partial<StyleProfile>) => api.put(`/style-profiles/${id}`, data),
  del: (id: number) => api.del(`/style-profiles/${id}`),
  activate: (id: number) => api.post(`/style-profiles/${id}/activate`),
  distill: (id: number) => api.post<{ result: any }>(`/style-profiles/${id}/distill`),
  apply: (id: number, data: any) => api.post(`/style-profiles/${id}/apply`, data),
}
