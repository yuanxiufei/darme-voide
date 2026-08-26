/**
 * 全局配置加载 — 优先级：环境变量 > configs/config.yaml > 硬编码默认值
 * 使 config.yaml 真正生效（此前是死配置，代码只用环境变量 + 硬编码）
 */
import { parse } from 'yaml'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(__dirname, '../..')

// config.yaml 位置（可用 CONFIG_PATH 环境变量覆盖，便于测试/多环境）
const configPath = process.env.CONFIG_PATH || path.join(projectRoot, 'configs', 'config.yaml')

// 解析 config.yaml；缺失/解析失败时降级为空对象（全部走默认值，不阻塞启动）
let raw: any = {}
try {
  raw = parse(fs.readFileSync(configPath, 'utf-8')) ?? {}
} catch (err) {
  console.warn(`[config] 读取 ${configPath} 失败，回退默认配置:`, (err as Error).message)
}

// config.yaml 中的相对路径统一相对项目根解析
function resolvePath(p: unknown): string {
  if (typeof p !== 'string' || !p) return ''
  return path.isAbsolute(p) ? p : path.resolve(projectRoot, p)
}

const server = raw.server ?? {}
const database = raw.database ?? {}
const evaluation = raw.evaluation ?? {}
const autoOptimize = evaluation.auto_optimize ?? {}

// ===== 统一数据根目录（SQLite 数据库 + 图片/视频/音频等所有生成文件）=====
// 优先级：.data-root 标记文件 > DATA_ROOT 环境变量 > config.yaml database.path 所在目录 > 默认 data/
// 数据库固定为 <dataRoot>/drama.db，生成文件固定为 <dataRoot>/static
const DATA_ROOT_MARKER = path.join(projectRoot, '.data-root')

function readDataRootMarker(): string {
  try {
    const v = fs.readFileSync(DATA_ROOT_MARKER, 'utf-8').trim()
    if (v && fs.existsSync(v)) return v
  } catch {}
  return ''
}

const defaultDataDir = path.join(projectRoot, 'data')
const yamlDbPath = resolvePath(database.path)
const yamlDbDir = yamlDbPath ? path.dirname(yamlDbPath) : ''

// 是否通过「标记文件 / DATA_ROOT 环境变量 / UI 切换」显式指定过数据根目录。
// 显式指定后，DB_PATH / STORAGE_PATH 环境变量不再覆盖，确保 UI 切换目录真正生效。
let dataRootExplicit = false
const markerRoot = readDataRootMarker()
let dataRoot: string

if (markerRoot) {
  dataRoot = markerRoot
  dataRootExplicit = true
} else if (process.env.DATA_ROOT) {
  dataRoot = path.resolve(projectRoot, process.env.DATA_ROOT)
  dataRootExplicit = true
} else {
  dataRoot = yamlDbDir || defaultDataDir
  dataRootExplicit = false
}

/** 当前数据根目录（含数据库与全部生成文件） */
export function getDataRoot(): string {
  return dataRoot
}

/** 当前 SQLite 数据库文件绝对路径 */
export function getDbPath(): string {
  if (!dataRootExplicit && process.env.DB_PATH) return process.env.DB_PATH
  return path.join(dataRoot, 'drama.db')
}

/** 当前生成文件存储根目录绝对路径（<dataRoot>/static） */
export function getStorageRoot(): string {
  if (!dataRootExplicit && process.env.STORAGE_PATH) return process.env.STORAGE_PATH
  return path.join(dataRoot, 'static')
}

/** 运行时切换数据根目录（写标记文件，重启后依然生效） */
export function setDataRoot(newRoot: string): void {
  const abs = path.resolve(newRoot)
  dataRoot = abs
  dataRootExplicit = true
  fs.mkdirSync(abs, { recursive: true })
  fs.writeFileSync(DATA_ROOT_MARKER, abs, 'utf-8')
}

export const config = {
  projectRoot,
  configPath,
  server: {
    port: Number(process.env.PORT || server.port || 5789),
    host: String(process.env.HOST || server.host || '0.0.0.0'),
    corsOrigins: (process.env.CORS_ORIGINS
      ? process.env.CORS_ORIGINS.split(',').map((s) => s.trim()).filter(Boolean)
      : (server.cors_origins ?? ['http://localhost:3013', 'http://localhost:5789'])) as string[],
  },
  database: {
    get path() { return getDbPath() },
  },
  storage: {
    get localPath() { return getStorageRoot() },
  },
  evaluation: {
    autoOptimize: {
      enabled: process.env.AUTO_OPTIMIZE_ENABLED === 'true' || autoOptimize.enabled === true,
      hour: Number(process.env.AUTO_OPTIMIZE_HOUR ?? autoOptimize.hour ?? 3),
      minute: Number(process.env.AUTO_OPTIMIZE_MINUTE ?? autoOptimize.minute ?? 0),
      iterations: Number(process.env.AUTO_OPTIMIZE_ITERATIONS ?? autoOptimize.iterations ?? 3),
      runOnStartup: process.env.AUTO_OPTIMIZE_RUN_ON_STARTUP === 'true' || autoOptimize.run_on_startup === true,
    },
  },
}
