/**
 * 音色快速复刻（Voice Clone）
 * MiniMax：POST /v1/files/upload（上传样本）→ POST /v1/voice_clone（复刻）
 * 复刻得到的 voice_id 写入音色库后，可直接用于 TTS 合成。
 */
import { fetchWithRetry } from '../utils/vendor-errors.js'
import { logTaskError, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'

export interface VoiceCloneInput {
  baseUrl: string
  apiKey: string
  fileBuffer: Buffer
  filename: string
  voiceId: string
  voiceName?: string
  demoText?: string
  model?: string
}

export interface VoiceCloneResult {
  voiceId: string
  demoAudio?: string
}

/** 根据文件名推断音频 MIME（MiniMax 支持 mp3/m4a/wav） */
function inferAudioMime(filename: string): string {
  if (/\.wav$/i.test(filename)) return 'audio/wav'
  if (/\.m4a$|\.mp4$/i.test(filename)) return 'audio/m4a'
  return 'audio/mpeg'
}

/** 上传音频样本到 MiniMax，返回 file_id */
async function uploadFileToMiniMax(baseUrl: string, apiKey: string, fileBuffer: Buffer, filename: string): Promise<string> {
  const url = `${baseUrl.replace(/\/+$/, '')}/v1/files/upload`
  const form = new FormData()
  form.append('purpose', 'voice_clone')
  form.append('file', new Blob([new Uint8Array(fileBuffer)], { type: inferAudioMime(filename) }), filename)

  const resp = await fetchWithRetry(url, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${apiKey}` },
    body: form,
  }, 'audio', { maxRetries: 2 })

  const data = await resp.json().catch(() => ({}))
  if (data?.base_resp?.status_code !== 0) {
    throw new Error(data?.base_resp?.status_msg || 'MiniMax 文件上传失败')
  }
  const fileId = data?.file?.file_id
  if (!fileId) throw new Error('MiniMax 未返回 file_id')
  return String(fileId)
}

/** 调用 MiniMax 音色快速复刻 */
export async function cloneVoice(input: VoiceCloneInput): Promise<VoiceCloneResult> {
  logTaskStart('VoiceClone', 'clone', { voiceId: input.voiceId, filename: input.filename })
  try {
    const fileId = await uploadFileToMiniMax(input.baseUrl, input.apiKey, input.fileBuffer, input.filename)

    const url = `${input.baseUrl.replace(/\/+$/, '')}/v1/voice_clone`
    const body: Record<string, unknown> = {
      file_id: Number(fileId),
      voice_id: input.voiceId,
      need_noise_reduction: true,
      need_volume_normalization: true,
    }
    if (input.voiceName) body.voice_name = input.voiceName
    if (input.demoText) {
      body.text = input.demoText
      body.model = input.model || 'speech-2.8-hd'
    }

    const resp = await fetchWithRetry(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${input.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    }, 'audio', { maxRetries: 2 })

    const data = await resp.json().catch(() => ({}))
    if (data?.base_resp?.status_code !== 0) {
      throw new Error(data?.base_resp?.status_msg || 'MiniMax 音色复刻失败')
    }

    const result: VoiceCloneResult = { voiceId: input.voiceId, demoAudio: data?.demo_audio || undefined }
    logTaskSuccess('VoiceClone', 'clone', { voiceId: result.voiceId })
    return result
  } catch (err: any) {
    logTaskError('VoiceClone', 'clone', { error: err.message })
    throw err
  }
}

/**
 * CosyVoice 2 零样本克隆（本地）：POST /inference_zero_shot
 * 无需训练，传入参考音频（prompt_audio）+ 参考文本（prompt_text）即时模仿音色合成 demo。
 * 接口约定参考 CosyVoice 官方 FastAPI 封装；本地服务（http://localhost:9880）部署后需按实际接口核对。
 */
export async function cloneVoiceCosyVoice(input: {
  baseUrl: string
  fileBuffer: Buffer
  promptText: string
  demoText: string
  model?: string
}): Promise<{ demoAudio?: string }> {
  logTaskStart('VoiceClone', 'cosyvoice-zero-shot', { promptText: input.promptText.slice(0, 20) })
  try {
    const url = `${input.baseUrl.replace(/\/+$/, '')}/inference_zero_shot`
    const body: Record<string, unknown> = {
      tts_text: input.demoText,
      prompt_text: input.promptText,
      prompt_audio: input.fileBuffer.toString('base64'),
      model: input.model || 'cosyvoice-v2',
      stream: false,
      speed: 1.0,
    }

    const resp = await fetchWithRetry(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }, 'audio', { maxRetries: 2 })

    const data = await resp.json().catch(() => ({}))
    const audio = data?.audio || data?.data?.audio
    const result: { demoAudio?: string } = { demoAudio: audio || undefined }
    logTaskSuccess('VoiceClone', 'cosyvoice-zero-shot', {})
    return result
  } catch (err: any) {
    logTaskError('VoiceClone', 'cosyvoice-zero-shot', { error: err.message })
    throw err
  }
}
