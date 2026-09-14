/**
 * Gemini 文本生成 Adapter（原生 :generateContent 格式，非 OpenAI 兼容）
 * 认证: URL Query ?key= + Header x-goog-api-key
 * 请求: contents[].parts[] 结构，system 消息映射到 systemInstruction
 * 响应: candidates[0].content.parts[].text
 */
import type {
  TextProviderAdapter,
  ProviderRequest,
  AIConfig,
  TextGenerateParams,
} from './types'
import { joinProviderUrl } from './url'

export class GeminiTextAdapter implements TextProviderAdapter {
  provider = 'gemini'

  buildRequest(config: AIConfig, params: TextGenerateParams): ProviderRequest {
    const modelName = params.model.startsWith('models/') ? params.model : `models/${params.model}`

    const systemParts = params.messages
      .filter(m => m.role === 'system')
      .map(m => ({ text: m.content }))
    const chatContents = params.messages
      .filter(m => m.role !== 'system')
      .map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      }))

    const body: any = {
      contents: chatContents,
      generationConfig: {
        temperature: params.temperature ?? 0.7,
      },
    }
    if (systemParts.length > 0) {
      body.systemInstruction = { parts: systemParts }
    }
    if (params.maxTokens) body.generationConfig.maxOutputTokens = params.maxTokens

    const url = new URL(joinProviderUrl(config.baseUrl, '/v1beta', `/${modelName}:generateContent`))
    url.searchParams.set('key', config.apiKey)

    return {
      url: url.toString(),
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': config.apiKey,
      },
      body,
    }
  }

  parseResponse(result: any): string {
    const parts = result?.candidates?.[0]?.content?.parts || []
    const texts: string[] = []
    for (const part of parts) {
      if (part && typeof part.text === 'string') texts.push(part.text)
    }
    return texts.join('')
  }
}
