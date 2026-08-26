/**
 * MCP（Model Context Protocol）工具接入层 —— 对齐 PenguinHarness 第 10 章
 *
 * 设计精髓（单一事实来源：外部工具生态接入，单点故障绝不拖垮整个 Agent Session）：
 * 1. 三传输 transport 推断：command → stdio；url → http(streamable)；显式 sse → sse
 * 2. 容错：非法条目只告警跳过、连接失败只降级、绝不 throw —— 外部工具挂了不影响主流程
 * 3. 命名隔离：`mcp__<server>__<tool>` 三段式扁平命名空间，与业务工具零冲突
 * 4. 懒连接 single-flight：会话组装时才连接，所有 server 并行连接，连接+握手+发现共用
 *    一个 connectTimeoutMs 预算（单一计时器，不做多层超时叠加）
 * 5. 权限方向：MCP 的 readOnlyHint 是不可信 hint，本层默认按更严格的 rw 处理（不据此放行）
 *
 * 配置来源：configs/mcp-servers.json（JSON 数组，见 configs/mcp-servers.example.json）。
 * 文件不存在或解析失败时视为空列表，对现有 Agent 零影响。
 */
import fs from 'fs'
import path from 'path'
import { createTool } from '@mastra/core/tools'
import { z } from 'zod'
import { Client } from '@modelcontextprotocol/sdk/client'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js'
import { config } from '../config.js'

export type McpTransport = 'stdio' | 'http' | 'sse'

export interface McpServerConfig {
  /** 唯一名，用于命名隔离前缀 mcp__<name>__<tool> */
  name: string
  /** 缺省时按 command/url 推断 */
  transport?: McpTransport
  /** stdio 传输：可执行文件路径 */
  command?: string
  args?: string[]
  env?: Record<string, string>
  cwd?: string
  /** http/sse 传输：服务端点 */
  url?: string
  headers?: Record<string, string>
  /** 连接 + 握手 + 发现共用的单一超时预算，默认 10s */
  connectTimeoutMs?: number
  enabled?: boolean
}

const MCP_TOOL_PREFIX = 'mcp__'
const DEFAULT_CONNECT_TIMEOUT_MS = 10_000

const mcpConfigPath =
  process.env.MCP_SERVERS_PATH || path.join(config.projectRoot, 'configs', 'mcp-servers.json')

interface ConnectedServer {
  config: McpServerConfig
  client: Client
  /** 已发现的 Mastra 工具（key = mcp__<server>__<tool>） */
  tools: Record<string, unknown>
}

/** 已连接 server 缓存（key = server name） */
const connectedServers = new Map<string, ConnectedServer>()
/** 当前配置指纹，用于检测配置变更失效缓存 */
let cacheFingerprint = ''
/** single-flight：并发调用共享同一次发现 */
let discoveryInFlight: Promise<Record<string, unknown>> | null = null

/** 读取并净化 MCP server 配置；非法条目跳过、文件缺失返回空列表 */
function loadMcpServers(): McpServerConfig[] {
  try {
    const raw = JSON.parse(fs.readFileSync(mcpConfigPath, 'utf-8'))
    if (!Array.isArray(raw)) return []
    const servers: McpServerConfig[] = []
    for (const item of raw) {
      if (!item || typeof item !== 'object' || typeof item.name !== 'string' || !item.name.trim()) {
        console.warn('[mcp] invalid server entry skipped (missing name):', JSON.stringify(item))
        continue
      }
      if (item.enabled === false) continue
      servers.push({ ...item, name: item.name.trim() })
    }
    return servers
  } catch {
    // 文件不存在或解析失败 → 空列表，零影响
    return []
  }
}

function inferTransport(cfg: McpServerConfig): McpTransport {
  if (cfg.transport) return cfg.transport
  if (cfg.command) return 'stdio'
  return 'http'
}

/** 净化工具名/服务名，仅保留安全字符，避免 Mastra 工具 id 非法 */
function sanitizeSegment(s: string): string {
  return s.replace(/[^a-zA-Z0-9_-]/g, '_')
}

function buildTransport(cfg: McpServerConfig) {
  const t = inferTransport(cfg)
  if (t === 'stdio') {
    if (!cfg.command) throw new Error(`MCP "${cfg.name}": stdio transport missing command`)
    return new StdioClientTransport({
      command: cfg.command,
      args: cfg.args,
      env: cfg.env,
      cwd: cfg.cwd,
      stderr: 'ignore',
    })
  }
  if (!cfg.url) throw new Error(`MCP "${cfg.name}": ${t} transport missing url`)
  const headers = cfg.headers ? { ...cfg.headers } : undefined
  const requestInit = headers ? { headers } : undefined
  if (t === 'sse') {
    return new SSEClientTransport(new URL(cfg.url), { requestInit } as any)
  }
  return new StreamableHTTPClientTransport(new URL(cfg.url), { requestInit } as any)
}

/** 单一计时器超时：连接/握手/发现共用一个预算 */
function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)
    p.then(
      (v) => { clearTimeout(timer); resolve(v) },
      (e) => { clearTimeout(timer); reject(e) },
    )
  })
}

/** 将 MCP 工具结果序列化为 LLM 可读文本 */
function serializeToolResult(result: any): Record<string, unknown> {
  const isError = !!result?.isError
  if (result?.structuredContent) {
    return { ...(result.structuredContent as Record<string, unknown>), isError }
  }
  const content = result?.content
  if (Array.isArray(content)) {
    const texts = content
      .filter((c: any) => c?.type === 'text' && typeof c.text === 'string')
      .map((c: any) => c.text)
    if (texts.length) return { text: texts.join('\n'), isError }
  }
  return { raw: JSON.stringify(result ?? null), isError }
}

/** 把一个 MCP tool 转成 Mastra tool（命名隔离 + 宽松入参透传） */
function toMastraTool(serverName: string, client: Client, mcpTool: any) {
  const fullName = `${MCP_TOOL_PREFIX}${sanitizeSegment(serverName)}__${sanitizeSegment(mcpTool.name)}`
  const sourceDesc = mcpTool.description?.trim()
  return createTool({
    id: fullName,
    description: sourceDesc
      ? `[MCP/${serverName}] ${sourceDesc}`
      : `[MCP/${serverName}] external tool ${mcpTool.name}`,
    // 宽松入参：不逐字段还原 JSON Schema，避免转换脆弱；参数透传给 MCP server 校验
    inputSchema: z.record(z.string(), z.any()),
    execute: async (args) => {
      const res = await client.callTool({
        name: mcpTool.name,
        arguments: (args ?? {}) as Record<string, unknown>,
      })
      return serializeToolResult(res)
    },
  })
}

/** 连接单个 server 并发现工具（失败抛出，由 allSettled 兜底降级） */
async function connectServer(cfg: McpServerConfig): Promise<void> {
  const client = new Client({ name: 'drama-studio', version: '1.0.0' }, { capabilities: {} } as any)
  const transport = buildTransport(cfg)
  const timeoutMs = cfg.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS

  try {
    await withTimeout(client.connect(transport), timeoutMs, `MCP "${cfg.name}" connect`)
    const list: any = await withTimeout(client.listTools(), timeoutMs, `MCP "${cfg.name}" listTools`)

    const tools: Record<string, unknown> = {}
    for (const t of list.tools || []) {
      const fullName = `${MCP_TOOL_PREFIX}${sanitizeSegment(cfg.name)}__${sanitizeSegment(t.name)}`
      if (tools[fullName]) {
        console.warn(`[mcp] duplicate tool "${fullName}" skipped`)
        continue
      }
      tools[fullName] = toMastraTool(cfg.name, client, t)
    }
    connectedServers.set(cfg.name, { config: cfg, client, tools })
  } catch (err: any) {
    // 连接失败：关闭残留连接，向上抛给 allSettled 降级
    try { await client.close() } catch { /* ignore */ }
    throw err
  }
}

function mergeCachedTools(): Record<string, unknown> {
  const merged: Record<string, unknown> = {}
  for (const s of connectedServers.values()) {
    for (const [k, v] of Object.entries(s.tools)) {
      if (merged[k]) {
        console.warn(`[mcp] tool name collision "${k}" — latter skipped`)
        continue
      }
      merged[k] = v
    }
  }
  return merged
}

async function closeAllServers(): Promise<void> {
  const closers = [...connectedServers.values()].map((s) => s.client.close().catch(() => {}))
  await Promise.all(closers)
  connectedServers.clear()
}

async function doDiscover(): Promise<Record<string, unknown>> {
  const servers = loadMcpServers()
  const fingerprint = JSON.stringify(servers)

  // 配置未变且已连接过 → 直接复用缓存
  if (fingerprint === cacheFingerprint) return mergeCachedTools()

  // 配置变更或首次：关闭旧连接，重建
  await closeAllServers()
  cacheFingerprint = fingerprint

  if (servers.length === 0) return {}

  const results = await Promise.allSettled(servers.map((s) => connectServer(s)))
  for (const r of results) {
    if (r.status === 'rejected') {
      // 单点故障只降级，绝不 throw —— 外部工具挂了不影响主流程
      console.warn('[mcp] server connect failed:', (r.reason as any)?.message || r.reason)
    }
  }
  return mergeCachedTools()
}

/**
 * 发现所有已配置 MCP server 的工具，返回可注入 Agent 的工具映射。
 * 懒连接 single-flight + 缓存：首次调用触发连接，后续复用；绝不 throw。
 */
export async function discoverMcpTools(): Promise<Record<string, unknown>> {
  if (discoveryInFlight) return discoveryInFlight
  discoveryInFlight = doDiscover()
  try {
    return await discoveryInFlight
  } finally {
    discoveryInFlight = null
  }
}

/** 手动刷新：关闭所有连接并重新发现（供路由触发） */
export async function refreshMcp(): Promise<void> {
  await closeAllServers()
  cacheFingerprint = ''
  await doDiscover()
}

/** 查询当前 MCP 接入状态（供路由/运维） */
export function getMcpStatus() {
  const configured = loadMcpServers().map((c) => c.name)
  const connected = [...connectedServers.values()].map((s) => ({
    server: s.config.name,
    transport: inferTransport(s.config),
    toolCount: Object.keys(s.tools).length,
    tools: Object.keys(s.tools),
  }))
  return { configured, connected }
}

/** 测试连接单个 server（独立临时连接，测完即关，不写入缓存） */
export async function testMcpServer(cfg: McpServerConfig): Promise<Record<string, unknown>> {
  try {
    if (!cfg?.name || typeof cfg.name !== 'string') return { ok: false, error: 'server name is required' }
    const client = new Client({ name: 'drama-studio', version: '1.0.0' }, { capabilities: {} } as any)
    const transport = buildTransport(cfg)
    const timeoutMs = cfg.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS
    await withTimeout(client.connect(transport), timeoutMs, `MCP "${cfg.name}" connect`)
    const list: any = await withTimeout(client.listTools(), timeoutMs, `MCP "${cfg.name}" listTools`)
    const tools = (list.tools || []).map((t: any) => t.name)
    try { await client.close() } catch { /* ignore */ }
    return { ok: true, server: cfg.name, transport: inferTransport(cfg), toolCount: tools.length, tools }
  } catch (err: any) {
    return { ok: false, server: cfg?.name, error: err?.message || String(err) }
  }
}
