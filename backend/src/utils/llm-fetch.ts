/**
 * LLM 专用 fetch
 *
 * 背景：`createOpenAI({ baseURL, apiKey })` 未显式指定 fetch 时，走 Node 全局
 * fetch（undici），其默认 headersTimeout / bodyTimeout 均为 300s。本地慢推理模型
 * （如 qwen3:32b）在非流式模式下，Ollama 会等完整生成结束才返回响应头，单次
 * agent.generate 常超 5 分钟，从而触发 "Headers Timeout Error"。
 *
 * 这里用 undici 自定义 Agent 把 headers/body 超时拉长，并仅作用于 LLM provider，
 * 不污染全局 dispatcher（其余内部 fetch 仍保持默认超时）。
 */
import { Agent as UndiciAgent, fetch as undiciFetch } from 'undici'

/** 等待响应头的最长时间（本地慢推理单次生成可能数分钟～十几分钟） */
const LLM_HEADERS_TIMEOUT = 30 * 60 * 1000 // 30 分钟
/** body 数据块之间的最长间隔 */
const LLM_BODY_TIMEOUT = 10 * 60 * 1000 // 10 分钟
/** TCP 建连超时 */
const LLM_CONNECT_TIMEOUT = 30 * 1000 // 30 秒

const dispatcher = new UndiciAgent({
  headersTimeout: LLM_HEADERS_TIMEOUT,
  bodyTimeout: LLM_BODY_TIMEOUT,
  connect: { timeout: LLM_CONNECT_TIMEOUT },
})

/** 供 createOpenAI({ fetch }) 使用；签名对齐 FetchFunction = typeof fetch */
export const llmFetch: typeof fetch = (input, init) =>
  undiciFetch(input as any, { ...init, dispatcher } as any) as unknown as Promise<Response>
