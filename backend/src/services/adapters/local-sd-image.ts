/**
 * 本地 Stable Diffusion 图片生成 Adapter
 * 对接 SD WebUI API：POST /sdapi/v1/txt2img 或 /sdapi/v1/img2img
 */
import type {
  ImageProviderAdapter,
  ProviderRequest,
  AIConfig,
  ImageGenerationRecord,
  ImageGenResponse,
  ImagePollResponse,
} from './types'
import { joinProviderUrl } from './url'

export class LocalSDImageAdapter implements ImageProviderAdapter {
  readonly provider = 'local-sd'

  buildGenerateRequest(config: AIConfig, record: ImageGenerationRecord): ProviderRequest {
    const prompt = record.prompt || ''
    const negative =
      record.negativePrompt ||
      'low quality, blurry, text, watermark, distorted face, deformed, bad anatomy, disfigured'

    const [w, h] = (record.size || '1024x1024').split('x').map(Number)
    const width = w || 1024
    const height = h || 1024

    const body: any = {
      prompt,
      negative_prompt: negative,
      steps: 25,
      width,
      height,
      cfg_scale: 7,
      sampler_name: 'DPM++ 2M Karras',
      seed: -1,
    }

    // 有参考图 → img2img 模式
    let endpoint = '/sdapi/v1/txt2img'
    if (record.referenceImages) {
      try {
        const refs = JSON.parse(record.referenceImages)
        if (Array.isArray(refs) && refs.length > 0) {
          body.init_images = [refs[0]]
          body.denoising_strength = 0.55
          endpoint = '/sdapi/v1/img2img'
        }
      } catch {
        // 参考图解析失败，fallback 到 txt2img
      }
    }

    return {
      url: joinProviderUrl(config.baseUrl, '', endpoint),
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    }
  }

  parseGenerateResponse(result: any): ImageGenResponse {
    if (!result.images || result.images.length === 0) {
      throw new Error('SD 未返回图片')
    }
    // SD 同步返回 base64，直接标记完成
    return { isAsync: false }
  }

  buildPollRequest(_config: AIConfig, _taskId: string): ProviderRequest {
    throw new Error('SD WebUI 同步模式，无需轮询')
  }

  parsePollResponse(_result: any): ImagePollResponse {
    throw new Error('SD WebUI 同步模式，无需轮询')
  }

  extractImageUrl(_result: any): string | null {
    // SD 返回的是 base64，不是 URL
    return null
  }

  extractImageBase64(result: any): { data: string; mimeType: string } | null {
    const b64 = result.images?.[0]
    if (!b64 || typeof b64 !== 'string') return null
    return { data: b64, mimeType: 'image/png' }
  }
}
