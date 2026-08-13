/**
 * GPU 显存管理器 — 云端/本地双模式下的智能显存调度
 *
 * 核心职责：
 * 1. 追踪当前 GPU 上加载了哪些本地模型及其估算显存
 * 2. 在切换模型前主动释放上一个模型的显存（避免 OOM）
 * 3. Mutex 锁确保同一时刻只有一个 GPU 重负载任务在运行
 * 4. 提供 GPU 状态监控（nvidia-smi）
 *
 * 使用方式：
 *   const lease = await gpuManager.acquire('image', config)
 *   try { ... 执行图片生成 ... } finally { lease.release() }
 *
 * 本地服务卸载机制：
 *   - Ollama: POST /api/generate { keep_alive: 0 } → 响应后立即卸载
 *   - SD WebUI: POST /sdapi/v1/unload-checkpoint（best-effort，部分版本支持）
 *   - CosyVoice: 轻量级（~4GB），不强制卸载，仅跟踪状态
 *   - MiniMax H3 本地: 跟踪状态，由外部管理生命周期
 */

import { logTaskProgress, logTaskWarn } from '../utils/task-logger.js'

// ─── 本地 Provider 标记 ────────────────────────────────────────
/** 判断 provider 是否为本地 GPU 服务 */
export function isLocalProvider(provider: string): boolean {
  return LOCAL_PROVIDERS.has(provider.toLowerCase())
}

/** 本地 provider 集合（这些是运行在本机 GPU 上的服务） */
const LOCAL_PROVIDERS = new Set([
  'ollama',
  'openai',       // Ollama OpenAI 兼容接口
  'local-sd',
  'cosyvoice',
  // minimax 本地 H3 也走 minimax provider，通过 baseUrl 中的 localhost 判断
])

/** 判断 AIConfig 是否指向本地服务（检查 baseUrl 含 localhost/127.0.0.1） */
export function isLocalConfig(baseUrl: string, provider: string): boolean {
  if (LOCAL_PROVIDERS.has(provider.toLowerCase())) return true
  // minimax 等通用 provider 但 baseUrl 指向本地
  try {
    const host = new URL(baseUrl).hostname
    return host === 'localhost' || host === '127.0.0.1' || host.startsWith('192.168.')
  } catch {
    return false
  }
}

// ─── VRAM 配置 ────────────────────────────────────────────────
export interface VramProfile {
  /** 估算显存占用（GB） */
  vramGB: number
  /** 卸载策略 */
  unloadStrategy: 'ollama-keep-alive' | 'sd-unload-checkpoint' | 'passive' | 'none'
  /** 服务 baseUrl（用于发送卸载请求） */
  baseUrl?: string
  /** 卸载等待时间（ms），给服务端缓冲 */
  cooldownMs: number
}

/** 每个 provider+model 组合的显存估算（GB） */
const VRAM_ESTIMATES: Record<string, VramProfile> = {
  // Ollama 文本模型
  'ollama:qwen3:14b':       { vramGB: 9,  unloadStrategy: 'ollama-keep-alive',  cooldownMs: 2000 },
  'ollama:qwen3:32b':       { vramGB: 16, unloadStrategy: 'ollama-keep-alive',  cooldownMs: 3000 },
  'ollama:qwen3:8b':        { vramGB: 5,  unloadStrategy: 'ollama-keep-alive',  cooldownMs: 1000 },
  'openai:qwen3:14b':       { vramGB: 9,  unloadStrategy: 'ollama-keep-alive',  cooldownMs: 2000 },
  'openai:qwen3:32b':       { vramGB: 16, unloadStrategy: 'ollama-keep-alive',  cooldownMs: 3000 },
  // SD WebUI 图片模型
  'local-sd:sdxl-base':     { vramGB: 12, unloadStrategy: 'sd-unload-checkpoint', cooldownMs: 3000 },
  'local-sd:sd15':          { vramGB: 5,  unloadStrategy: 'sd-unload-checkpoint', cooldownMs: 1500 },
  // CosyVoice 语音模型（轻量）
  'cosyvoice:cosyvoice-v2': { vramGB: 4,  unloadStrategy: 'passive',              cooldownMs: 500 },
  // MiniMax H3 本地视频模型
  'minimax:hailuo-02':      { vramGB: 15, unloadStrategy: 'passive',              cooldownMs: 5000 },
  // 默认兜底
  'default':                { vramGB: 8,  unloadStrategy: 'none',                 cooldownMs: 1000 },
}

const GPU_TOTAL_GB = 24
const GPU_SAFE_MARGIN_GB = 2 // 安全余量，避免精确填满

// ─── GPU 租约 ──────────────────────────────────────────────────
export interface GpuLease {
  readonly serviceType: string
  readonly modelKey: string
  readonly vramGB: number
  /** 释放 GPU 占用（使用完毕后必须调用） */
  release: () => void
}

// ─── 状态类型 ──────────────────────────────────────────────────
interface GpuState {
  /** 当前锁持有者 */
  holder: string | null
  /** 锁等待队列 */
  queue: Array<{ serviceType: string; resolve: () => void }>
  /** 当前已加载的模型列表（可能有多个轻量模型共存） */
  loadedModels: Map<string, VramProfile>
}

// ─── GPU 管理器单例 ───────────────────────────────────────────
class GpuMemoryManager {
  private state: GpuState = {
    holder: null,
    queue: [],
    loadedModels: new Map(),
  }

  /** 当前 GPU 总占用估算（GB） */
  get totalVramGB(): number {
    let total = 0
    this.state.loadedModels.forEach(p => { total += p.vramGB })
    return Math.round(total * 10) / 10
  }

  /** 当前可用显存（GB） */
  get availableVramGB(): number {
    return Math.round((GPU_TOTAL_GB - GPU_SAFE_MARGIN_GB - this.totalVramGB) * 10) / 10
  }

  /** 是否被锁定 */
  get isLocked(): boolean {
    return this.state.holder !== null
  }

  /** 当前锁持有者 */
  get currentHolder(): string | null {
    return this.state.holder
  }

  /** 已加载模型列表 */
  get loadedModelKeys(): string[] {
    return Array.from(this.state.loadedModels.keys())
  }

  /**
   * 获取 VRAM 配置
   */
  private getProfile(provider: string, model: string): VramProfile {
    const key = `${provider.toLowerCase()}:${model}`
    return VRAM_ESTIMATES[key] || VRAM_ESTIMATES['default'] || {
      vramGB: 8, unloadStrategy: 'none', cooldownMs: 1000,
    }
  }

  /**
   * 尝试卸载指定模型
   */
  private async unloadModel(modelKey: string, profile: VramProfile): Promise<boolean> {
    logTaskProgress('GpuManager', 'unload-attempt', { modelKey, strategy: profile.unloadStrategy })

    try {
      switch (profile.unloadStrategy) {
        case 'ollama-keep-alive': {
          // Ollama: 发送 keep_alive=0 的 generate 请求，模型在响应后立即卸载
          const [_, modelName] = modelKey.split(':')
          const ollamaUrl = profile.baseUrl || 'http://localhost:11434'
          const resp = await fetch(`${ollamaUrl}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: modelName || 'qwen3:14b',
              prompt: '.',
              keep_alive: 0,
              max_tokens: 1,
              stream: false,
            }),
            signal: AbortSignal.timeout(10_000),
          })
          if (resp.ok) {
            // 读取响应以触发 keep_alive=0 卸载
            await resp.text()
            logTaskProgress('GpuManager', 'unload-ollama-ok', { modelKey })
          }
          return true
        }

        case 'sd-unload-checkpoint': {
          // SD WebUI: 尝试卸载模型
          const sdUrl = profile.baseUrl || 'http://localhost:7860'
          try {
            await fetch(`${sdUrl}/sdapi/v1/unload-checkpoint`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: '{}',
              signal: AbortSignal.timeout(5_000),
            })
            logTaskProgress('GpuManager', 'unload-sd-ok', { modelKey })
          } catch {
            // unload-checkpoint 可能不支持，记录但不报错
            logTaskWarn('GpuManager', 'unload-sd-unsupported', { modelKey })
          }
          return true
        }

        case 'passive':
          // 被动模式：仅从跟踪列表中移除，由外部服务自行管理
          logTaskProgress('GpuManager', 'unload-passive', { modelKey })
          return true

        case 'none':
        default:
          return false
      }
    } catch (err: any) {
      logTaskWarn('GpuManager', 'unload-error', { modelKey, error: err.message })
      return false
    }
  }

  /**
   * 释放所有冲突模型，为新的 GPU 任务腾出空间
   */
  private async evictIfNeeded(neededGB: number, newModelKey: string): Promise<void> {
    if (this.totalVramGB + neededGB <= GPU_TOTAL_GB - GPU_SAFE_MARGIN_GB) {
      return // 空间足够，无需驱逐
    }

    const needToFree = this.totalVramGB + neededGB - (GPU_TOTAL_GB - GPU_SAFE_MARGIN_GB)
    logTaskProgress('GpuManager', 'evict-needed', {
      currentVram: this.totalVramGB,
      needed: neededGB,
      needToFree: Math.round(needToFree * 10) / 10,
    })

    // 按显存占用从小到大排序，优先卸载小的（快速释放）
    // 但跳过即将要加载的新模型
    const candidates = Array.from(this.state.loadedModels.entries())
      .filter(([key]) => key !== newModelKey)
      .sort((a, b) => a[1].vramGB - b[1].vramGB)

    let freed = 0
    for (const [key, profile] of candidates) {
      if (freed >= needToFree) break
      const success = await this.unloadModel(key, profile)
      if (success) {
        freed += profile.vramGB
        this.state.loadedModels.delete(key)
        logTaskProgress('GpuManager', 'evicted', { modelKey: key, freedGB: profile.vramGB })
      }
      // 卸载冷却时间
      await new Promise(r => setTimeout(r, profile.cooldownMs))
    }

    if (freed < needToFree) {
      logTaskWarn('GpuManager', 'evict-insufficient', {
        needToFree: Math.round(needToFree * 10) / 10,
        freed: Math.round(freed * 10) / 10,
        remainingModels: Array.from(this.state.loadedModels.keys()),
      })
    }
  }

  /**
   * 获取 GPU 独占锁
   *
   * @param serviceType  任务类型 ('text' | 'image' | 'video' | 'audio')
   * @param provider     厂商名称
   * @param model        模型名称
   * @param baseUrl      服务地址（用于发送卸载请求）
   * @returns GpuLease 租约，使用完毕需调用 lease.release()
   */
  async acquire(
    serviceType: string,
    provider: string,
    model: string,
    baseUrl?: string,
  ): Promise<GpuLease> {
    const modelKey = `${provider.toLowerCase()}:${model}`
    const baseProfile = this.getProfile(provider, model)
    // 浅拷贝，防止修改全局共享的 VRAM_ESTIMATES 配置对象
    const profile: VramProfile = { ...baseProfile }
    if (baseUrl) profile.baseUrl = baseUrl

    logTaskProgress('GpuManager', 'acquire-request', {
      serviceType,
      modelKey,
      vramGB: profile.vramGB,
      currentVram: this.totalVramGB,
      holder: this.state.holder,
    })

    // 等待锁释放（简单队列实现，避免竞态）
    if (this.state.holder) {
      await new Promise<void>(resolve => {
        this.state.queue.push({ serviceType, resolve })
        logTaskProgress('GpuManager', 'queued', {
          serviceType,
          queueLength: this.state.queue.length,
          holder: this.state.holder,
        })
      })
    }

    // 获取锁
    this.state.holder = modelKey

    // 驱逐冲突模型
    await this.evictIfNeeded(profile.vramGB, modelKey)

    // 标记新模型已加载
    this.state.loadedModels.set(modelKey, profile)

    logTaskProgress('GpuManager', 'acquired', {
      serviceType,
      modelKey,
      vramGB: profile.vramGB,
      totalVram: this.totalVramGB,
    })

    return {
      serviceType,
      modelKey,
      vramGB: profile.vramGB,
      release: () => this.release(modelKey, profile),
    }
  }

  /**
   * 释放 GPU 锁
   */
  private release(modelKey: string, profile: VramProfile): void {
    logTaskProgress('GpuManager', 'release', {
      modelKey,
      strategy: profile.unloadStrategy,
    })

    // 如果是 ollama-keep-alive，主动卸载
    if (profile.unloadStrategy === 'ollama-keep-alive') {
      this.unloadModel(modelKey, profile).catch(err => {
        logTaskError('GpuManager', 'unload-on-release', { modelKey, error: err.message })
      })
      this.state.loadedModels.delete(modelKey)
    } else {
      // 其他策略：暂时保留在 loadedModels 中（可能复用）
      // 当需要空间时由 evictIfNeeded 卸载
    }

    // 标记锁释放
    this.state.holder = null

    // 处理等待队列
    const next = this.state.queue.shift()
    if (next) {
      logTaskProgress('GpuManager', 'dequeue', {
        next: next.serviceType,
        remaining: this.state.queue.length,
      })
      next.resolve()
    }
  }

  /**
   * 批量释放所有已加载模型
   */
  async releaseAll(): Promise<void> {
    logTaskProgress('GpuManager', 'release-all', {
      count: this.state.loadedModels.size,
    })

    for (const [key, profile] of this.state.loadedModels) {
      await this.unloadModel(key, profile)
    }
    this.state.loadedModels.clear()
    this.state.holder = null

    // 清空队列
    this.state.queue.forEach(q => q.resolve())
    this.state.queue = []
  }

  /**
   * 获取当前状态快照（供 API 返回）
   */
  getStatus(): GpuStatus {
    return {
      totalVRAM_GB: GPU_TOTAL_GB,
      safeMargin_GB: GPU_SAFE_MARGIN_GB,
      usedVRAM_GB: this.totalVramGB,
      availableVRAM_GB: this.availableVramGB,
      isLocked: this.isLocked,
      holder: this.state.holder,
      queueLength: this.state.queue.length,
      loadedModels: this.loadedModelKeys,
    }
  }
}

export interface GpuStatus {
  totalVRAM_GB: number
  safeMargin_GB: number
  usedVRAM_GB: number
  availableVRAM_GB: number
  isLocked: boolean
  holder: string | null
  queueLength: number
  loadedModels: string[]
}

// ─── 单例导出 ──────────────────────────────────────────────────
export const gpuManager = new GpuMemoryManager()

/**
 * GPU 租约辅助：自动在 finally 中释放
 *
 * 用法：
 *   await withGpuLease('image', config, async (lease) => {
 *     // 执行 GPU 操作
 *   })
 */
export async function withGpuLease(
  serviceType: string,
  provider: string,
  model: string,
  baseUrl: string,
  fn: (lease: GpuLease) => Promise<void>,
): Promise<void> {
  const lease = await gpuManager.acquire(serviceType, provider, model, baseUrl)
  try {
    await fn(lease)
  } finally {
    lease.release()
  }
}
