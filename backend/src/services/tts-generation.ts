/**
 * TTS 语音合成服务
 * 支持 MiniMax TTS (hex 音频响应) 和 OpenAI 兼容 /audio/speech
 */
import fs from 'fs'
import path from 'path'
import { v4 as uuid } from 'uuid'
import { getAudioConfigById } from './ai.js'
import { getTTSAdapter } from './adapters/registry.js'
import { logTaskError, logTaskPayload, logTaskProgress, logTaskStart, logTaskSuccess, logTaskWarn, redactUrl } from '../utils/task-logger.js'
import { fetchWithRetry } from '../utils/vendor-errors.js'
import { gpuManager, isLocalConfig } from './gpu-manager.js'
import { getStorageRoot } from '../config.js'

interface TTSParams {
  text: string
  voice: string
  model?: string
  speed?: number
  emotion?: string
  pitch?: number
  configId?: number | null
}

/**
 * 生成 TTS 音频，返回本地文件路径。
 * 支持多模型自动 fallback：一个模型失败时自动尝试配置中的下一个模型。
 */
export async function generateTTS(params: TTSParams): Promise<string> {
  const config = getAudioConfigById(params.configId)
  const models = config.models?.length ? config.models : [config.model]
  const isLocal = isLocalConfig(config.baseUrl, config.provider)

  for (let attempt = 0; attempt < models.length; attempt++) {
    const model = params.model || models[attempt]
    const attemptConfig = { ...config, model }

    logTaskStart('AudioTask', 'tts-generate', {
      provider: config.provider,
      voice: params.voice,
      model,
      attempt: attempt + 1,
      textPreview: params.text.slice(0, 50),
      textLength: params.text.length,
      isLocal,
    })
    logTaskPayload('AudioTask', 'tts params', {
      config: { provider: config.provider, model, baseUrl: config.baseUrl },
      params,
    })

    // ── 本地 GPU 模型：获取显存租约 ──
    let lease: any = null
    if (isLocal) {
      lease = await gpuManager.acquire('audio', config.provider, model, config.baseUrl)
    }

    try {
      const adapter = getTTSAdapter(config.provider)
      const { url, method, headers, body } = adapter.buildGenerateRequest(attemptConfig, { ...params, model })
      logTaskProgress('AudioTask', 'request', {
        provider: config.provider, voice: params.voice,
        method, url: redactUrl(url), model,
      })
      logTaskPayload('AudioTask', 'request payload', { method, url, headers, body })

      const resp = await fetchWithRetry(url, { method, headers, body: JSON.stringify(body) }, 'audio', {
        onRetry: (attempt, delayMs, reason) => logTaskWarn('AudioTask', 'request-retry', { model, attempt, delayMs, reason }),
      })

      const result = await resp.json()
      const parsed = adapter.parseResponse(result)

      // 将 hex 解码为二进制
      const buffer = Buffer.from(parsed.audioHex, 'hex')

      // 保存到本地
      const audioDir = path.join(getStorageRoot(), 'audio')
      fs.mkdirSync(audioDir, { recursive: true })
      const filename = `${uuid()}.${parsed.format || 'mp3'}`
      const filePath = path.join(audioDir, filename)
      fs.writeFileSync(filePath, buffer)

      const relativePath = `static/audio/${filename}`
      logTaskSuccess('AudioTask', 'tts-saved', {
        provider: config.provider, model, voice: params.voice,
        path: relativePath, bytes: buffer.length, audioMs: parsed.audioLength,
      })
      return relativePath
    } catch (err: any) {
      const isLastAttempt = attempt === models.length - 1
      logTaskError('AudioTask', isLastAttempt ? 'all-models-failed' : 'model-fallback', {
        provider: config.provider, voice: params.voice, model, attempt,
        error: err.message,
        ...(isLastAttempt ? {} : { nextModel: models[attempt + 1] }),
      })
      if (isLastAttempt) throw err
    } finally {
      if (lease) lease.release()
    }
  }

  throw new Error('All TTS models failed')
}

/**
 * 为角色生成试听音频
 */
export async function generateVoiceSample(characterName: string, voiceId: string, configId?: number | null): Promise<string> {
  const sampleText = `你好，我是${characterName}。很高兴认识你，这是我的声音试听。`
  return generateTTS({ text: sampleText, voice: voiceId, configId })
}
