/**
 * 本地模型路由 — 扫描本机任意目录下的模型文件并识别类型。
 *
 * 阶段 1 仅提供只读接口：
 *  - GET /local-models/scan   扫描模型（支持 roots/kinds/maxDepth/maxFiles 参数）
 *  - GET /local-models/roots  返回默认扫描根目录及探测状态
 * 阶段 2 将补充 POST /local-models/register 注册到 ai_service_configs。
 */
import { Hono } from 'hono'
import fs from 'fs'
import path from 'path'
import { randomUUID } from 'node:crypto'
import { stream } from 'hono/streaming'
import { eq } from 'drizzle-orm'
import { success, badRequest, now } from '../utils/response.js'
import { toSnakeCase } from '../utils/transform.js'
import { logTaskSuccess, logTaskError } from '../utils/task-logger.js'
import { db, schema } from '../db/index.js'
import {
  scanLocalModels,
  scanLocalModelsAsync,
  listDrives,
  getDefaultRoots,
  getExtraRoots,
  saveExtraRoots,
  getModelPaths,
  saveModelPaths,
  ScanCancelledError,
  type ModelKind,
  type ScanProgress,
} from '../services/local-model-scan.js'

const app = new Hono()

const VALID_KINDS: ModelKind[] = ['text', 'image', 'video', 'audio', 'unknown']

/** 解析 roots 参数：逗号分隔，或 JSON 数组字符串 */
function parseRoots(raw: string | undefined): string[] | undefined {
  if (!raw) return undefined
  const trimmed = raw.trim()
  if (!trimmed) return undefined
  if (trimmed.startsWith('[')) {
    try {
      const arr = JSON.parse(trimmed)
      if (Array.isArray(arr)) return arr.filter((x) => typeof x === 'string' && x)
    } catch { /* 降级为逗号分隔 */ }
  }
  return trimmed.split(',').map((s) => s.trim()).filter(Boolean)
}

/** 解析 kinds 参数 */
function parseKinds(raw: string | undefined): ModelKind[] | undefined {
  if (!raw) return undefined
  const kinds = raw.split(',').map((s) => s.trim().toLowerCase()) as ModelKind[]
  const valid = kinds.filter((k) => VALID_KINDS.includes(k))
  return valid.length ? valid : undefined
}

// 扫描本地模型
app.get('/scan', (c) => {
  const roots = parseRoots(c.req.query('roots'))
  const kinds = parseKinds(c.req.query('kinds'))
  const maxDepth = Number(c.req.query('maxDepth')) || 5
  const maxFiles = Number(c.req.query('maxFiles')) || 8000

  if (roots && roots.length === 0) {
    return badRequest(c, 'roots 参数为空或格式不正确')
  }

  const result = scanLocalModels({ roots, kinds, maxDepth, maxFiles })
  return success(c, result)
})

// 返回默认扫描根目录及探测状态
app.get('/roots', (c) => {
  const roots = getDefaultRoots()
  const detail = roots.map((r) => {
    let sizeHint = ''
    try {
      if (fs.statSync(r).isDirectory()) {
        const parent = path.basename(r)
        sizeHint = parent // 仅展示目录名，避免遍历磁盘
      }
    } catch { /* 忽略 */ }
    return { path: r, name: sizeHint || path.basename(r) }
  })
  return success(c, { roots, detail, extra: getExtraRoots(), paths: getModelPaths() })
})

// 保存模型路径配置（partial 更新：roots=额外扫描目录，models_dir=模型存储目录）
app.put('/roots', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}))
    let extra = getExtraRoots()

    // 额外扫描目录：仅当显式传入 roots 数组时才更新（否则保留现状）
    if (Array.isArray(body.roots)) {
      extra = saveExtraRoots(
        body.roots.filter((x: any) => typeof x === 'string' && x.trim()),
      )
    }

    // 模型存储目录：仅当显式传入 models_dir 时才更新
    if (body.models_dir !== undefined) {
      saveModelPaths({ models_dir: body.models_dir })
    }

    logTaskSuccess('LocalModels', 'saveRoots', {
      extra: extra.length,
      models_dir: body.models_dir ?? '(unchanged)',
    })
    return success(c, { extra, roots: getDefaultRoots(), paths: getModelPaths() })
  } catch (err: any) {
    logTaskError('LocalModels', 'saveRoots', { error: err.message })
    return badRequest(c, err.message)
  }
})

// ──────────────────────────────────────────────────────────────
// 阶段 2：POST /local-models/register — 将扫描到的模型注册进 ai_service_configs
// ──────────────────────────────────────────────────────────────

const SERVICE_LABELS: Record<string, string> = {
  text: '文本',
  image: '图像',
  video: '视频',
  audio: '音频',
}

/** 安全解析 settings JSON 字段 */
function parseSettings(raw: unknown): Record<string, any> {
  if (!raw) return {}
  if (typeof raw === 'object') return raw as Record<string, any>
  try {
    const v = JSON.parse(String(raw))
    return v && typeof v === 'object' ? v : {}
  } catch {
    return {}
  }
}

/** 从文件名派生 H3 checkpoint_map 键（fl2va / ref2va） */
function h3CheckpointKey(filename: string): 'fl2va' | 'ref2va' | null {
  const n = filename.toLowerCase()
  if (n.includes('fl2va')) return 'fl2va'
  if (n.includes('ref2va')) return 'ref2va'
  return null
}

/**
 * 注册本地模型。
 * body: { models: ScannedModel[], name?: string }
 *  - standalone 模型：按 (serviceType|provider|baseUrl) 去重合并，upsert 进 ai_service_configs。
 *  - component 模型：不单独注册，返回 skipped 并说明原因（由主模型 checkpoint_map / ComfyUI 工作流引用）。
 *  - H3 视频 DiT：从文件名派生 fl2va/ref2va，写入 settings.checkpoint_map。
 */
app.post('/register', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}))
    const models: any[] = Array.isArray(body.models) ? body.models : []
    const overrideName = typeof body.name === 'string' ? body.name.trim() : ''

    if (models.length === 0) {
      return badRequest(c, 'models 不能为空')
    }

    const ts = now()
    const created: string[] = []
    const updated: string[] = []
    const skipped: { filename: string; reason: string }[] = []

    // 按去重键分组 standalone 模型
    const groups = new Map<string, any[]>()

    for (const m of models) {
      const s = m?.suggested
      const role = m?.role ?? s?.role
      const filename = typeof m?.filename === 'string' && m.filename
        ? m.filename
        : path.basename(m?.path || '')

      if (role === 'component') {
        skipped.push({
          filename,
          reason: s?.note || '组件模型，需在 ComfyUI 工作流中引用，不单独注册',
        })
        continue
      }
      // ComfyUI 仅作「只读的可选扫描来源」，不注册为可调用后端
      if (s?.runtime === 'comfyui' || s?.callable === false) {
        skipped.push({
          filename,
          reason: s?.note || 'ComfyUI 仅作只读扫描来源，不直接调用',
        })
        continue
      }
      if (!s) {
        skipped.push({ filename, reason: '无法识别模型类型，未生成注册建议' })
        continue
      }
      const key = `${s.serviceType}|${s.provider}|${s.baseUrl}`
      if (!groups.has(key)) groups.set(key, [])
      groups.get(key)!.push({ suggested: s, filename })
    }

    for (const [key, items] of groups.entries()) {
      const [serviceType, provider, baseUrl] = key.split('|')
      const first = items[0].suggested

      // 收集 model 名（去重）与 H3 checkpoint_map
      const modelNames: string[] = []
      const checkpointMap: Record<string, string> = {}
      for (const it of items) {
        const modelName = it.suggested.model
        if (modelName && !modelNames.includes(modelName)) modelNames.push(modelName)
        if (first.runtime === 'h3' && serviceType === 'video') {
          const ck = h3CheckpointKey(it.filename)
          if (ck) checkpointMap[ck] = it.filename.replace(/\.[^.]+$/, '')
        }
      }

      const name = overrideName || `本地${SERVICE_LABELS[serviceType] || serviceType}服务`

      // 查现有同 serviceType+provider+baseUrl 配置
      const existing = db.select().from(schema.aiServiceConfigs)
        .where(eq(schema.aiServiceConfigs.serviceType, serviceType))
        .all()
        .filter((row) => row.provider === provider && row.baseUrl === baseUrl)

      const row = existing[0]
      const isNew = existing.length === 0

      // 合并 settings（checkpoint_map）
      let settings = parseSettings(row?.settings)
      if (Object.keys(checkpointMap).length > 0) {
        settings = {
          ...settings,
          checkpoint_map: { ...(settings.checkpoint_map || {}), ...checkpointMap },
        }
      }

      // 合并 model 数组（去重）
      let mergedModels = modelNames
      if (row?.model) {
        try {
          const prev = JSON.parse(row.model)
          if (Array.isArray(prev)) {
            mergedModels = [...new Set([...prev, ...modelNames])]
          }
        } catch { /* 保留新模型名 */ }
      }

      const values = {
        serviceType,
        provider,
        name,
        baseUrl,
        apiKey: row?.apiKey || 'local',
        model: JSON.stringify(mergedModels),
        priority: row?.priority ?? 80,
        settings: JSON.stringify(settings),
        isActive: true,
        updatedAt: ts,
      }

      if (isNew) {
        db.insert(schema.aiServiceConfigs).values({ ...values, createdAt: ts }).run()
        created.push(name)
      } else {
        db.update(schema.aiServiceConfigs).set(values).where(eq(schema.aiServiceConfigs.id, row!.id)).run()
        updated.push(name)
      }
    }

    logTaskSuccess('LocalModels', 'register', {
      created: created.length,
      updated: updated.length,
      skipped: skipped.length,
    })

    const configs = db.select().from(schema.aiServiceConfigs).all().map((row) => ({
      ...toSnakeCase(row),
      model: row.model ? JSON.parse(row.model) : [],
    }))

    return success(c, { created, updated, skipped, configs })
  } catch (err: any) {
    logTaskError('LocalModels', 'register', { error: err.message })
    return badRequest(c, err.message)
  }
})

// ──────────────────────────────────────────────────────────────
// 阶段 3：磁盘/全盘异步扫描（后台任务 + 进度轮询 + 取消）
// ──────────────────────────────────────────────────────────────

interface ScanTaskState {
  id: string
  progress: ScanProgress
  cancel: () => void
}

/** 内存任务表（本地单用户，任务量小，无需持久化） */
const scanTasks = new Map<string, ScanTaskState>()

// 列出可用磁盘盘符
app.get('/drives', (c) => {
  const drives = listDrives()
  return success(c, { drives })
})

// 启动异步扫描任务（body 可传 roots/kinds/maxDepth/maxFiles）
app.post('/scan', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}))
    const roots = Array.isArray(body.roots)
      ? body.roots.filter((x: any) => typeof x === 'string' && x.trim())
      : undefined
    const kinds = Array.isArray(body.kinds)
      ? body.kinds.filter((k: any) => VALID_KINDS.includes(k))
      : undefined
    const maxDepth = Number(body.maxDepth) > 0 ? Number(body.maxDepth) : undefined
    const maxFiles = Number(body.maxFiles) > 0 ? Number(body.maxFiles) : undefined

    if (roots && roots.length === 0) {
      return badRequest(c, 'roots 参数为空或格式不正确')
    }

    const taskId = randomUUID()
    const cancelled = { value: false }
    const state: ScanTaskState = {
      id: taskId,
      progress: {
        scannedFiles: 0,
        foundModels: 0,
        currentDir: '',
        done: false,
        cancelled: false,
      },
      cancel: () => { cancelled.value = true },
    }
    scanTasks.set(taskId, state)

    const cleanup = () => {
      // 完成后保留 10 分钟供前端拉取结果，之后清理
      setTimeout(() => { scanTasks.delete(taskId) }, 10 * 60 * 1000)
    }

    scanLocalModelsAsync({
      roots,
      kinds,
      maxDepth,
      maxFiles,
      shouldCancel: () => cancelled.value,
      onProgress: (p) => { state.progress = p },
    })
      .then((result) => {
        state.progress = { ...state.progress, done: true, result }
        cleanup()
      })
      .catch((err: any) => {
        if (err instanceof ScanCancelledError) {
          state.progress = { ...state.progress, done: true, cancelled: true }
        } else {
          state.progress = { ...state.progress, done: true, error: err?.message || String(err) }
        }
        cleanup()
      })

    return success(c, { taskId })
  } catch (err: any) {
    return badRequest(c, err.message)
  }
})

// 查询扫描任务进度/结果
app.get('/scan/status', (c) => {
  const taskId = c.req.query('taskId')
  if (!taskId) return badRequest(c, 'taskId 不能为空')
  const state = scanTasks.get(taskId)
  if (!state) return badRequest(c, '任务不存在或已过期')
  return success(c, { taskId, progress: state.progress })
})

// 取消扫描任务
app.post('/scan/cancel', (c) => {
  const taskId = c.req.query('taskId')
  if (!taskId) return badRequest(c, 'taskId 不能为空')
  const state = scanTasks.get(taskId)
  if (!state) return badRequest(c, '任务不存在或已过期')
  state.cancel()
  return success(c, { taskId, cancelled: true })
})

// ──────────────────────────────────────────────────────────────
// 阶段 4：HuggingFace 权重下载到「模型存储目录」
// ──────────────────────────────────────────────────────────────

/** 下载来源：HF 官方 / hf-mirror 国内镜像 / ModelScope 魔搭 */
type DownloadSource = 'hf' | 'hf_mirror' | 'modelscope'

const SOURCES: Record<DownloadSource, { base: string; label: string }> = {
  hf: { base: 'https://huggingface.co', label: 'HuggingFace 官方' },
  hf_mirror: { base: 'https://hf-mirror.com', label: 'hf-mirror 镜像' },
  modelscope: { base: 'https://modelscope.cn', label: 'ModelScope 魔搭' },
}

/** 解析下载来源参数，非法值回退为 hf */
function parseSource(raw: unknown): DownloadSource {
  const s = String(raw || '').trim().toLowerCase()
  return (s === 'hf_mirror' || s === 'modelscope' || s === 'hf') ? s : 'hf'
}

/** 默认分支：ModelScope 用 master，HF 系列用 main */
function defaultRevision(source: DownloadSource, revision: string): string {
  const r = revision.trim()
  if (r) return r
  return source === 'modelscope' ? 'master' : 'main'
}

/** 校验并归一化仓库名（owner/name），防止路径穿越 */
function normalizeRepo(raw: unknown): string {
  const repo = String(raw || '').trim().replace(/^\/+|\/+$/g, '')
  if (!repo || !repo.includes('/')) throw new Error('仓库名格式应为 owner/name，例如 Qwen/Qwen3-4B')
  if (repo.includes('..') || repo.includes('\\')) throw new Error('仓库名不合法')
  return repo
}

/** 通用 JSON 请求封装（失败抛错并截断错误信息） */
async function fetchJson(url: string, timeoutMs = 30_000): Promise<any> {
  const resp = await fetch(url, {
    headers: { Accept: 'application/json' },
    signal: AbortSignal.timeout(timeoutMs),
  })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error((text || `HTTP ${resp.status}`).slice(0, 400))
  }
  return resp.json().catch(() => null)
}

/** 通过 HF 系列 API 列出仓库文件（含大小） */
async function listHfFiles(repo: string, source: DownloadSource): Promise<{ name: string; size: number }[]> {
  const base = SOURCES[source].base
  const json: any = await fetchJson(`${base}/api/models/${repo}?blobs=true`)
  const list: any[] = Array.isArray(json?.files) ? json.files : Array.isArray(json?.siblings) ? json.siblings : []
  return list
    .map((f) => ({ name: String(f?.rfilename || ''), size: Number(f?.size) || 0 }))
    .filter((f) => f.name && !f.name.endsWith('/'))
}

/** 通过 ModelScope API 列出仓库文件（含大小），仅返回 blob 文件 */
async function listModelScopeFiles(repo: string, revision: string): Promise<{ name: string; size: number }[]> {
  const url = `https://modelscope.cn/api/v1/models/${repo}/repo/files?Revision=${encodeURIComponent(revision)}&Recursive=true`
  const json: any = await fetchJson(url)
  const files: any[] = Array.isArray(json?.Data?.Files) ? json.Data.Files : []
  return files
    .filter((f) => f?.Type === 'blob' && String(f?.Path || ''))
    .map((f) => ({ name: String(f.Path), size: Number(f.Size) || 0 }))
}

/** 按来源列出仓库文件 */
function listRemoteFiles(repo: string, source: DownloadSource, revision: string) {
  return source === 'modelscope'
    ? listModelScopeFiles(repo, revision)
    : listHfFiles(repo, source)
}

/** 按来源构造下载直链（modelscope 走 /models/.../resolve 302 重定向） */
function buildDownloadUrl(source: DownloadSource, repo: string, revision: string, filePath: string): string {
  const encRev = encodeURIComponent(revision)
  const encPath = filePath.split('/').map(encodeURIComponent).join('/')
  if (source === 'modelscope') {
    return `https://modelscope.cn/models/${repo}/resolve/${encRev}/${encPath}`
  }
  return `${SOURCES[source].base}/${repo}/resolve/${encRev}/${encPath}`
}

/**
 * 通过 ModelScope API 获取单文件签名下载 URL（用于 LFS 大文件稳定下载）。
 * 失败返回 null，由调用方回退到 /resolve/ 302 直链。
 */
async function getModelScopeDownloadUrl(repo: string, revision: string, filePath: string): Promise<string | null> {
  try {
    const url = `https://modelscope.cn/api/v1/models/${repo}/repo?FilePath=${encodeURIComponent(filePath)}&Revision=${encodeURIComponent(revision)}`
    const json: any = await fetchJson(url, 15_000)
    const u = json?.Data?.Url
    return typeof u === 'string' && u ? u : null
  } catch {
    return null
  }
}

// 列出仓库文件（按来源）
app.post('/hf/files', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}))
    const repo = normalizeRepo(body.repo)
    const source = parseSource(body.source)
    const revision = defaultRevision(source, String(body.revision || ''))
    const files = await listRemoteFiles(repo, source, revision)
    return success(c, { repo, source, revision, files })
  } catch (err: any) {
    return badRequest(c, err.message)
  }
})

// 下载模型到「模型存储目录」，NDJSON 流式返回进度（按来源）
app.post('/hf/download', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}))
    const repo = normalizeRepo(body.repo)
    const source = parseSource(body.source)
    const revision = defaultRevision(source, String(body.revision || ''))
    const requested: string[] = (Array.isArray(body.files) ? body.files : [])
      .filter((x: any) => typeof x === 'string' && x.trim())

    const { models_dir } = getModelPaths()
    if (!models_dir) return badRequest(c, '请先在「模型存储目录」中设置下载目录')

    const all = await listRemoteFiles(repo, source, revision)
    if (all.length === 0) return badRequest(c, `仓库 ${repo} 没有可下载的文件`)

    const targets = requested.length
      ? requested.map((name) => {
          const hit = all.find((f) => f.name === name)
          if (!hit) throw new Error(`仓库中不存在文件：${name}`)
          return hit
        })
      : all

    const destDir = path.join(models_dir, repo.replace('/', '__'))
    fs.mkdirSync(destDir, { recursive: true })

    c.header('Content-Type', 'application/x-ndjson; charset=utf-8')
    return stream(c, async (s) => {
      const send = (obj: any) => s.write(JSON.stringify(obj) + '\n')
      let totalBytes = targets.reduce((acc, f) => acc + (f.size || 0), 0)
      let overall = 0
      const failed: string[] = []
      await send({ status: 'start', repo, revision, source, files: targets.map((f) => f.name), total: totalBytes })

      for (const f of targets) {
        let url = buildDownloadUrl(source, repo, revision, f.name)
        // ModelScope 优先取签名 URL 下载 LFS 大文件，失败回退 /resolve/ 直链
        if (source === 'modelscope') {
          const signed = await getModelScopeDownloadUrl(repo, revision, f.name)
          if (signed) url = signed
        }
        const outPath = path.join(destDir, f.name)
        const partPath = outPath + '.part'
        let fileDone = 0
        let size = f.size || 0
        try {
          fs.mkdirSync(path.dirname(outPath), { recursive: true })

          // 已存在目标文件且大小满足要求：跳过，避免重复下载
          let existing = 0
          try { existing = fs.statSync(outPath).size } catch { /* 不存在 */ }
          if (existing > 0 && (size === 0 || existing >= size)) {
            overall += existing
            await send({ status: 'file_done', file: f.name, downloaded: existing, skipped: true })
            continue
          }

          // 断点续传：读取 .part 部分文件大小
          let partSize = 0
          try { partSize = fs.statSync(partPath).size } catch { /* 不存在 */ }
          if (partSize > 0 && size > 0 && partSize >= size) {
            overall += partSize
            fs.renameSync(partPath, outPath)
            await send({ status: 'file_done', file: f.name, downloaded: partSize })
            continue
          }

          const headers: Record<string, string> = { Accept: '*/*' }
          if (partSize > 0) headers['Range'] = `bytes=${partSize}-`
          const resp = await fetch(url, {
            headers,
            signal: AbortSignal.timeout(4 * 60 * 60_000),
          })
          if (!resp.ok || !resp.body) {
            await send({ status: 'error', file: f.name, error: `HTTP ${resp.status}` })
            failed.push(f.name)
            continue
          }

          // 服务器返回 206 表示续传生效；返回 200 表示不支持 Range，从头重下
          const isPartial = resp.status === 206
          fileDone = isPartial ? partSize : 0
          if (isPartial) overall += partSize

          const clen = Number(resp.headers.get('content-length'))
          if (isPartial && clen > 0) {
            const fullSize = fileDone + clen
            if (size > 0 && fullSize !== size) totalBytes += fullSize - size
            size = fullSize
          } else if (clen > 0 && clen !== size) {
            totalBytes += clen - size
            size = clen
          }

          const writer = fs.createWriteStream(partPath, { flags: isPartial ? 'a' : 'w' })
          await send({ status: 'file_start', file: f.name, total: size, resumed: isPartial, downloaded: fileDone })

          const reader = resp.body.getReader()
          try {
            while (true) {
              const { done, value } = await reader.read()
              if (done) break
              const buf = Buffer.from(value as Uint8Array)
              if (!writer.write(buf)) await new Promise((r) => writer.once('drain', r))
              fileDone += buf.length
              overall += buf.length
              await send({ status: 'progress', file: f.name, downloaded: fileDone, total: size, overall, overallTotal: totalBytes })
            }
          } finally {
            reader.releaseLock()
          }
          await new Promise((resolve, reject) => {
            writer.on('finish', resolve)
            writer.on('error', reject)
            writer.end()
          })
          fs.renameSync(partPath, outPath)
          await send({ status: 'file_done', file: f.name, downloaded: fileDone })
        } catch (err: any) {
          // 保留 .part 供下次续传，不删除
          await send({ status: 'error', file: f.name, error: err?.message || String(err) })
          failed.push(f.name)
        }
      }

      await send({ status: 'done', overall, total: totalBytes, failed })
    })
  } catch (err: any) {
    logTaskError('LocalModels', 'hf-download', { error: err.message })
    return badRequest(c, err.message)
  }
})

// ──────────────────────────────────────────────────────────────
// 阶段 5：删除「模型存储目录」下的本地模型文件/目录
// ──────────────────────────────────────────────────────────────

/** 校验目标路径落在 models_dir 内，返回规范化绝对路径（防路径穿越） */
function resolveModelPath(raw: unknown): string {
  const target = String(raw || '').trim()
  if (!target) throw new Error('目标路径不能为空')
  const { models_dir } = getModelPaths()
  if (!models_dir) throw new Error('请先在「模型存储目录」中设置目录')
  const base = path.resolve(models_dir)
  const abs = path.resolve(base, target)
  if (abs === base) throw new Error('不能删除整个模型存储目录')
  if (!abs.startsWith(base + path.sep)) throw new Error('目标路径不在模型存储目录内')
  return abs
}

app.post('/delete', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}))
    const rawPaths: unknown[] = Array.isArray(body.paths) ? body.paths : [body.path]
    const paths = rawPaths.filter((x) => typeof x === 'string' && String(x).trim())
    if (paths.length === 0) return badRequest(c, '请指定要删除的文件或目录')

    const deleted: string[] = []
    const failed: { path: string; error: string }[] = []
    for (const p of paths) {
      const label = String(p).trim()
      try {
        const abs = resolveModelPath(label)
        if (!fs.existsSync(abs)) {
          failed.push({ path: label, error: '路径不存在' })
          continue
        }
        fs.rmSync(abs, { recursive: true, force: true })
        deleted.push(label)
      } catch (err: any) {
        failed.push({ path: label, error: err.message })
      }
    }
    logTaskSuccess('LocalModels', 'delete', { deleted: deleted.length, failed: failed.length })
    return success(c, { deleted, failed })
  } catch (err: any) {
    logTaskError('LocalModels', 'delete', { error: err.message })
    return badRequest(c, err.message)
  }
})

export default app
