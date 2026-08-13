const BASE = '/api/v1'

async function req<T = any>(method: string, path: string, body?: any): Promise<T> {
  const opts: RequestInit = { method, headers: { 'Content-Type': 'application/json' } }
  if (body) opts.body = JSON.stringify(body)

  const start = performance.now()
  console.log(`%c[API] %c${method} %c${path}`, 'color:#888', 'color:#4fc3f7;font-weight:bold', 'color:#ccc', body || '')

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

export const api = {
  get: <T = any>(p: string) => req<T>('GET', p),
  post: <T = any>(p: string, b?: any) => req<T>('POST', p, b),
  put: <T = any>(p: string, b?: any) => req<T>('PUT', p, b),
  del: <T = any>(p: string) => req<T>('DELETE', p),
}

export const dramaAPI = {
  list: () => api.get<{ items: any[] }>('/dramas'),
  get: (id: number) => api.get(`/dramas/${id}`),
  create: (data: any) => api.post('/dramas', data),
  update: (id: number, data: any) => api.put(`/dramas/${id}`, data),
  del: (id: number) => api.del(`/dramas/${id}`),
}

export const episodeAPI = {
  create: (data: any) => api.post('/episodes', data),
  update: (id: number, data: any) => api.put(`/episodes/${id}`, data),
  characters: (id: number) => api.get(`/episodes/${id}/characters`),
  scenes: (id: number) => api.get(`/episodes/${id}/scenes`),
  storyboards: (id: number) => api.get(`/episodes/${id}/storyboards`),
  pipelineStatus: (id: number) => api.get(`/episodes/${id}/pipeline-status`),
}

export const storyboardAPI = {
  create: (data: any) => api.post('/storyboards', data),
  update: (id: number, data: any) => api.put(`/storyboards/${id}`, data),
  generateTTS: (id: number) => api.post(`/storyboards/${id}/generate-tts`),
  validateDialogue: (id: number) => api.get(`/storyboards/${id}/validate-dialogue`),
  regenerateImage: (id: number, data: any) => api.post(`/storyboards/${id}/regenerate-image`, data),
  del: (id: number) => api.del(`/storyboards/${id}`),
}

export const characterAPI = {
  get: (id: number) => api.get(`/characters/${id}`),
  update: (id: number, data: any) => api.put(`/characters/${id}`, data),
  voiceSample: (id: number, episodeId: number) => api.post(`/characters/${id}/generate-voice-sample`, { episode_id: episodeId }),
  generateImage: (id: number, episodeId: number, data?: { prompt?: string; model?: string }) =>
    api.post(`/characters/${id}/generate-image`, { episode_id: episodeId, ...data }),
  batchImages: (ids: number[], episodeId: number) => api.post('/characters/batch-generate-images', { character_ids: ids, episode_id: episodeId }),
}

export const sceneAPI = {
  get: (id: number) => api.get(`/scenes/${id}`),
  update: (id: number, data: any) => api.put(`/scenes/${id}`, data),
  generateImage: (id: number, episodeId: number, data?: { prompt?: string; model?: string }) =>
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
  quickPreset: (apiKey: string) => api.post('/ai-configs/quick-preset', { api_key: apiKey }),
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
  fromProp: (propId: number) => api.post(`/weapon-library/from-prop/${propId}`),
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
}
