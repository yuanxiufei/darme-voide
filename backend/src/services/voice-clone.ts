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
