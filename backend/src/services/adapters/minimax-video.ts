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
      model: record.model || config.model,
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
