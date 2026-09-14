/**
 * CosyVoice 2 本地 TTS Adapter
 * 假设 CosyVoice 运行在本机 HTTP 服务，暴露 REST 接口
 *
 * 预期 API：
 *   POST /tts  { text, voice_id, speed, emotion }
 *   响应: { audio: "<base64 hex>", format: "mp3", sample_rate: 32000 }
 */
import type { TTSProviderAdapter } from './types'
import type { AIConfig } from './types'
import { joinProviderUrl } from './url'

export interface CosyVoiceTTSParams {
  text: string
  voice: string
  speed?: number
  model?: string
  emotion?: string
  promptAudio?: string   // 零样本克隆：参考音频 base64
  promptText?: string    // 零样本克隆：参考音频对应文本
}

export class CosyVoiceTTSAdapter implements TTSProviderAdapter {
  readonly provider = 'cosyvoice'

  buildGenerateRequest(config: AIConfig, params: CosyVoiceTTSParams): {
    url: string
    method: string
    headers: Record<string, string>
    body: any
  } {
    // 零样本模式：带参考音频时走 CosyVoice 官方 /inference_zero_shot
    if (params.promptAudio) {
      return {
        url: joinProviderUrl(config.baseUrl, '', '/inference_zero_shot'),
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: {
          tts_text: params.text,
          prompt_text: params.promptText || '',
          prompt_audio: params.promptAudio,
          model: config.model || 'cosyvoice-v2',
          stream: false,
          speed: params.speed ?? 1.0,
        },
      }
    }

    return {
      url: joinProviderUrl(config.baseUrl, '', '/tts'),
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: {
        text: params.text,
        voice_id: params.voice,
        speed: params.speed ?? 1.0,
        emotion: params.emotion || 'neutral',
        model: config.model || 'cosyvoice-v2',
        format: 'mp3',
        sample_rate: 32000,
      },
    }
  }

  parseResponse(result: any): {
    audioHex: string
    audioLength: number
    sampleRate: number
    bitrate: number
    format: string
    channel: number
  } {
    const raw = result.audio || result.data?.audio || ''
    if (!raw) throw new Error('CosyVoice 未返回音频数据')

    // CosyVoice 返回的可能是 base64，自动转 hex（兼容下游 Buffer.from(audioHex, 'hex')）
    let audioHex = raw
    if (!/^[0-9a-fA-F]+$/.test(raw.slice(0, 100))) {
      // 非纯 hex → 视为 base64 → 转 hex
      audioHex = Buffer.from(raw, 'base64').toString('hex')
    }

    return {
      audioHex,
      audioLength: result.audio_length || result.duration || 0,
      sampleRate: result.sample_rate || 32000,
      bitrate: result.bitrate || 128000,
      format: result.format || 'mp3',
      channel: result.channel || 1,
    }
  }
}
