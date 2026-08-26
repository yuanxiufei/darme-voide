/**
 * Provider Adapter 注册表
 * 根据 provider 名称返回对应的 Adapter 实例
 */
import { MiniMaxImageAdapter } from './minimax-image'
import { MiniMaxVideoAdapter } from './minimax-video'
import { MiniMaxTTSAdapter } from './minimax-tts'
import { OpenAIImageAdapter } from './openai-image'
import { GeminiImageAdapter } from './gemini-image'
import { VolcEngineImageAdapter } from './volcengine-image'
import { VolcEngineVideoAdapter } from './volcengine-video'
import { ViduVideoAdapter } from './vidu-video'
import { AliImageAdapter } from './ali-image'
import { AliVideoAdapter } from './ali-video'
import { LocalSDImageAdapter } from './local-sd-image'
import { CosyVoiceTTSAdapter } from './cosyvoice-tts'
import { OpenAICompatibleTextAdapter } from './openai-compatible-text'
import { GeminiTextAdapter } from './gemini-text'
import type { ImageProviderAdapter, VideoProviderAdapter, TTSProviderAdapter, TextProviderAdapter } from './types'

// 图片 Adapter 注册表
export const imageAdapters: Record<string, ImageProviderAdapter> = {
  minimax: new MiniMaxImageAdapter(),
  openai: new OpenAIImageAdapter(),
  gemini: new GeminiImageAdapter(),
  volcengine: new VolcEngineImageAdapter(),
  ali: new AliImageAdapter(),
  // 第三方代理 - 兼容 OpenAI 格式
  chatfire: new OpenAIImageAdapter(),
  // 本地 SD WebUI
  'local-sd': new LocalSDImageAdapter(),
}

// 视频 Adapter 注册表
// 注意：MiniMax H3 开源后本地启动也走 minimax Adapter，只需改 baseUrl 指向 localhost
export const videoAdapters: Record<string, VideoProviderAdapter> = {
  minimax: new MiniMaxVideoAdapter(),
  volcengine: new VolcEngineVideoAdapter(),
  vidu: new ViduVideoAdapter(),
  ali: new AliVideoAdapter(),
}

// TTS Adapter 注册表
export const ttsAdapters: Record<string, TTSProviderAdapter> = {
  minimax: new MiniMaxTTSAdapter(),
  cosyvoice: new CosyVoiceTTSAdapter(),
}

// 文本 Adapter 注册表
// openai/openrouter/chatfire/ollama/volcengine/ali/minimax 均为 OpenAI 兼容（仅端点前缀不同），共享一个实例
const openAICompatibleText = new OpenAICompatibleTextAdapter()
export const textAdapters: Record<string, TextProviderAdapter> = {
  openai: openAICompatibleText,
  openrouter: openAICompatibleText,
  chatfire: openAICompatibleText,
  ollama: openAICompatibleText,
  volcengine: openAICompatibleText,
  ali: openAICompatibleText,
  minimax: openAICompatibleText,
  gemini: new GeminiTextAdapter(),
}

/**
 * 获取文本 Adapter
 * @param provider 厂商名称
 * @throws 未知厂商显式报错
 */
export function getTextAdapter(provider: string): TextProviderAdapter {
  return resolveAdapter(textAdapters, provider, 'text')
}

/**
 * 从注册表解析 Adapter。
 * 未知 provider **显式 throw**（不再静默 fallback 到 MiniMax），
 * 让配置拼写错误在提交时尽早暴露，而非"成功走到 MiniMax"难以排查。
 */
function resolveAdapter<T>(registry: Record<string, T>, provider: string, kind: string): T {
  const key = provider.toLowerCase()
  const adapter = registry[key]
  if (!adapter) {
    throw new Error(`Unknown ${kind} provider "${provider}". Available: ${Object.keys(registry).join(', ')}`)
  }
  return adapter
}

export function getTTSAdapter(provider: string): TTSProviderAdapter {
  return resolveAdapter(ttsAdapters, provider, 'TTS')
}

/**
 * 获取图片 Adapter
 * @param provider 厂商名称
 * @throws 未知厂商显式报错
 */
export function getImageAdapter(provider: string): ImageProviderAdapter {
  return resolveAdapter(imageAdapters, provider, 'image')
}

/**
 * 获取视频 Adapter
 * @param provider 厂商名称
 * @throws 未知厂商显式报错
 */
export function getVideoAdapter(provider: string): VideoProviderAdapter {
  return resolveAdapter(videoAdapters, provider, 'video')
}
