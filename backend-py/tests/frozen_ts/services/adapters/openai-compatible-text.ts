/**
 * OpenAI 兼容文本生成 Adapter
 * 覆盖所有走 /chat/completions 的文本服务商，仅端前缀不同：
 *   - openai / openrouter / chatfire / ollama → /v1
 *   - volcengine（火山方舟，model 用 Endpoint ID）→ /api/v3
 *   - ali（阿里百炼 OpenAI 兼容模式）→ /compatible-mode/v1
 * 响应格式: { choices: [{ message: { content } }] }
 */
import type {
  TextProviderAdapter,
  ProviderRequest,
  AIConfig,
  TextGenerateParams,
} from './types'
import { joinProviderUrl } from './url'

const PREFIX_BY_PROVIDER: Record<string, string> = {
  openai: '/v1',
  openrouter: '/v1',
  chatfire: '/v1',
  ollama: '/v1',
  volcengine: '/api/v3',
  ali: '/compatible-mode/v1',
  minimax: '/v1',
}

export class OpenAICompatibleTextAdapter implements TextProviderAdapter {
  provider = 'openai-compatible'

  buildRequest(config: AIConfig, params: TextGenerateParams): ProviderRequest {
    const prefix = PREFIX_BY_PROVIDER[config.provider.toLowerCase()] || '/v1'

    const body: any = {
      model: params.model,
      messages: params.messages,
      temperature: params.temperature ?? 0.7,
    }
    if (params.maxTokens) body.max_tokens = params.maxTokens

    return {
      url: joinProviderUrl(config.baseUrl, prefix, '/chat/completions'),
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${config.apiKey}`,
      },
      body,
    }
  }

  parseResponse(result: any): string {
    const content = result?.choices?.[0]?.message?.content
    if (typeof content === 'string') return content

    // 某些兼容实现返回数组 content（多模态），取第一段 text
    if (Array.isArray(content)) {
      for (const part of content) {
        if (part && typeof part === 'object' && typeof part.text === 'string') return part.text
      }
    }
    return ''
  }
}
