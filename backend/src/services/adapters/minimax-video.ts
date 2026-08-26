/**
 * MiniMax H3 视频生成 Adapter
 * MiniMax H3 是大模型，使用原生 API 格式：
 *   POST /v1/video_generation  { model, prompt, aspect_ratio, duration, first_frame_image? }
 */
import type {
  VideoProviderAdapter,
  ProviderRequest,
  AIConfig,
  VideoGenerationRecord,
  VideoGenResponse,
  VideoPollResponse,
} from './types'
import { joinProviderUrl } from './url'

export class MiniMaxVideoAdapter implements VideoProviderAdapter {
  provider = 'minimax'

  buildGenerateRequest(config: AIConfig, record: VideoGenerationRecord): ProviderRequest {
    // MiniMax H3 原生 body：prompt 是纯字符串，不做 content 数组包装
    const body: any = {
      model: resolveH3Checkpoint(config, record),
      prompt: record.prompt || '',
      aspect_ratio: record.aspectRatio || '16:9',
      duration: record.duration || 5,
    }

    // 参考图：MiniMax 原生字段 first_frame_image / last_frame_image
    if (record.referenceMode === 'single' && record.imageUrl) {
      body.first_frame_image = record.imageUrl
    } else if (record.referenceMode === 'first_last') {
      if (record.firstFrameUrl) body.first_frame_image = record.firstFrameUrl
      if (record.lastFrameUrl) body.last_frame_image = record.lastFrameUrl
    } else if (record.referenceMode === 'multiple' && record.referenceImageUrls) {
      // Ref2VA：多参考图（角色立绘 + 场景图）用 subject_reference 锁定主体
      try {
        const refs = JSON.parse(record.referenceImageUrls)
        if (Array.isArray(refs) && refs.length) body.subject_reference = refs.map(String).slice(0, 9)
      } catch { /* ignore malformed json */ }
    }

    // 参考音频（H3 音视频联合生成 / Ref2VA reference conditioning）：角色声线样本，最多 3 条
    // 字段名 reference_audio 为本地 H3 服务的约定；云端 MiniMax 若不支持会忽略该字段
    if (record.referenceAudioUrls) {
      try {
        const audios = JSON.parse(record.referenceAudioUrls)
        if (Array.isArray(audios) && audios.length) body.reference_audio = audios.map(String).slice(0, 3)
      } catch { /* ignore malformed json */ }
    }

    return {
      url: joinProviderUrl(config.baseUrl, '/v1', '/video_generation'),
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${config.apiKey}`,
      },
      body,
    }
  }

  parseGenerateResponse(result: any): VideoGenResponse {
    const taskId = result.task_id || result.id || result.data?.id
    if (!taskId) {
      // 同步返回
      const videoUrl = result.video_url || result.data?.video_url || result.content?.video_url
      if (videoUrl) {
        return { isAsync: false, videoUrl }
      }
      throw new Error('No task_id or video_url in response')
    }
    return { isAsync: true, taskId }
  }

  buildPollRequest(config: AIConfig, taskId: string): ProviderRequest {
    return {
      url: joinProviderUrl(config.baseUrl, '/v1', `/video_generation/task/${taskId}`),
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${config.apiKey}`,
      },
      body: undefined,
    }
  }

  parsePollResponse(result: any): VideoPollResponse {
    const status = result.status || result.state || result.data?.status
    if (status === 'completed' || status === 'succeeded') {
      return {
        status: 'completed',
        videoUrl: result.video_url || result.data?.video_url || result.content?.video_url,
      }
    }
    if (status === 'failed' || status === 'error') {
      return { status: 'failed', error: result.error_msg || result.error || 'Video generation failed' }
    }
    return { status: status || 'processing' }
  }

  extractVideoUrl(result: any): string | null {
    return result.video_url || result.data?.video_url || result.content?.video_url || null
  }
}

/**
 * H3 双 checkpoint 路由：根据 scene_type 选择 FL2VA / Ref2VA。
 * 需在 ai_service_configs.settings 里声明 checkpoint_map，例如：
 *   { "checkpoint_map": { "fl2va": "minimax_h3_fl2va_pruned_int8_convrot", "ref2va": "minimax_h3_ref2va_pruned_int8_convrot" } }
 * 未声明 checkpoint_map 时退化为 record.model / config.model（单 checkpoint 模式，云端亦然）。
 */
function resolveH3Checkpoint(config: AIConfig, record: VideoGenerationRecord): string {
  const base = record.model || config.model
  const map = config.settings?.checkpoint_map
  if (!map || typeof map !== 'object') return base
  const st = (record.sceneType || '').toLowerCase()
  // FL2VA（图生视频/首尾帧）：动作 / 静默 / 空镜 / 转场；其余（single/dialogue/meeting/argument/long_dialogue）走 Ref2VA
  const isFl2va = /action|silent|transition|establishing|empty/.test(st)
  const key = isFl2va ? 'fl2va' : 'ref2va'
  const mapped = map[key] || map.fl2va || map.ref2va
  return typeof mapped === 'string' && mapped ? mapped : base
}
