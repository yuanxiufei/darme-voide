// 共享契约：字段唯一来源，禁止在前端再手工双写一遍（见 backend/src/shared/contracts.ts）
import type { DramaDetailDTO, DramaListResponse, DramaUpdateBody, EraBackground } from '~contracts'

const BASE = '/api/v1'

async function req<T = any>(method: string, path: string, body?: any): Promise<T> {
  const opts: RequestInit = { method, headers: { 'Content-Type': 'application/json' } }
  if (body) opts.body = JSON.stringify(body)

  const start = performance.now()
  console.log(`%c[API] %c${method} %c${path}`, 'color:#888', 'color:#2dd4bf;font-weight:bold', 'color:#ccc', body || '')

  try {
    const resp = await fetch(`${BASE}${path}`, opts)
    const json = await resp.json()
    const ms = Math.round(performance.now() - start)

    if (!resp.ok || (json.code && json.code >= 400)) {
      console.log(`%c[API] %c${method} ${path} %c${resp.status} %c${ms}ms`, 'color:#888', 'color:#ef5350', 'color:#ef5350;font-weight:bold', 'color:#888', json.message || '')
      // 增强错误信息：包含状态码和路径，便于调试
      const error = new Error(json.message || `请求失败 (${resp.status})`)
      ;(error as any).status = resp.status
      ;(error as any).code = json.code
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
  const resp = await fetch(`${BASE}${path}`, { method: 'POST', body: formData })
  const json = await resp.json()
  if (!resp.ok || (json.code && json.code >= 400)) {
    const error = new Error(json.message || `请求失败 (${resp.status})`)
    ;(error as any).status = resp.status
    ;(error as any).code = json.code
    throw error
  }
  return json.data ?? json
}

export const api = {
  get: <T = any>(p: string) => req<T>('GET', p),
  post: <T = any>(p: string, b?: any) => req<T>('POST', p, b),
  put: <T = any>(p: string, b?: any) => req<T>('PUT', p, b),
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
  /** AI 从剧本提炼时代背景并落库（与 PUT 时代背景同构，三字段规范见 shared/contracts EraBackground） */
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
  /** 删除本地 Ollama 模型 */
  ollamaDelete: (name: string, baseUrl?: string) => api.post('/ai-configs/ollama/delete', { name, base_url: baseUrl }),
  /** 下载 Ollama 模型，NDJSON 流式返回进度事件 { status, completed, total, error } */
  ollamaPull: async (name: string, baseUrl?: string, onEvent?: (ev: any) => void): Promise<{ ok: boolean; error?: string }> => {
    const resp = await fetch(`${BASE}/ai-configs/ollama/pull`, {
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
    const resp = await fetch(`${BASE}/local-models/hf/download`, {
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

export const agentConfigAPI = {
  list: () => api.get('/agent-configs'),
  get: (id: number) => api.get(`/agent-configs/${id}`),
  create: (d: any) => api.post('/agent-configs', d),
  update: (id: number, d: any) => api.put(`/agent-configs/${id}`, d),
  del: (id: number) => api.del(`/agent-configs/${id}`),
}

export const skillsAPI = {
  list: () => api.get('/skills'),
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
  const resp = await fetch(`${BASE}${endpoint}`, { method: 'POST', body: fd })
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
  streamUrl: (dramaId: number) => `${BASE}/auto-pipeline/stream/${dramaId}`,
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
