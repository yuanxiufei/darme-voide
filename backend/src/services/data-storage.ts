/**
 * 数据存储目录管理 — 统一管理数据根目录（SQLite 数据库 + 图片/视频/音频等生成文件）
 * 支持运行时切换目录并自动迁移旧数据（复制，旧目录保留作为安全备份）
 */
import fs from 'fs'
import path from 'path'
import { getDataRoot, getDbPath, getStorageRoot, setDataRoot, config } from '../config.js'
import { closeDatabase, reopenDatabase } from '../db/connection.js'
import { rebuildDb } from '../db/index.js'

export interface StorageInfo {
  dataRoot: string
  dbPath: string
  storagePath: string
  dbExists: boolean
  storageExists: boolean
  dbSizeBytes: number
  storageSizeBytes: number
}

function fileSize(p: string): number {
  try { return fs.existsSync(p) ? fs.statSync(p).size : 0 } catch { return 0 }
}

function dirSize(dir: string): number {
  if (!fs.existsSync(dir)) return 0
  let total = 0
  const walk = (p: string) => {
    let entries: fs.Dirent[]
    try { entries = fs.readdirSync(p, { withFileTypes: true }) } catch { return }
    for (const e of entries) {
      const full = path.join(p, e.name)
      if (e.isDirectory()) walk(full)
      else { try { total += fs.statSync(full).size } catch {} }
    }
  }
  walk(dir)
  return total
}

/** 获取当前数据存储信息 */
export function getStorageInfo(): StorageInfo {
  const dbPath = getDbPath()
  const storagePath = getStorageRoot()
  return {
    dataRoot: getDataRoot(),
    dbPath,
    storagePath,
    dbExists: fs.existsSync(dbPath),
    storageExists: fs.existsSync(storagePath),
    dbSizeBytes: fileSize(dbPath),
    storageSizeBytes: dirSize(storagePath),
  }
}

/** 复制 SQLite 数据库文件（含 WAL/SHM 伴生文件） */
function copyDbFiles(oldDbPath: string, newDbPath: string): void {
  const targetDir = path.dirname(newDbPath)
  fs.mkdirSync(targetDir, { recursive: true })
  const candidates = [oldDbPath, `${oldDbPath}-wal`, `${oldDbPath}-shm`]
  for (const src of candidates) {
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, path.join(targetDir, path.basename(src)))
    }
  }
}

/**
 * 切换数据根目录
 * @param newRootRaw 新目录路径（相对路径按后端进程 cwd 解析）
 * @param opts.migrate 是否迁移旧数据（默认 true，复制旧库与 static 目录，旧目录保留）
 */
export function changeDataRoot(newRootRaw: string, opts: { migrate?: boolean } = {}): StorageInfo {
  const trimmed = String(newRootRaw || '').trim()
  if (!trimmed) throw new Error('请填写有效的目录路径')
  const newRoot = path.resolve(trimmed)

  const oldRoot = getDataRoot()
  const oldDbPath = getDbPath()
  const oldStoragePath = getStorageRoot()

  if (newRoot === oldRoot) throw new Error('新目录与当前目录相同，无需切换')

  // 目录合法性校验：禁止指向项目根、源码/依赖/配置目录
  const projectRoot = config.projectRoot
  if (newRoot === projectRoot) throw new Error('数据目录不能设置为项目根目录')
  const forbidden = [
    path.join(projectRoot, 'backend'),
    path.join(projectRoot, 'frontend'),
    path.join(projectRoot, 'node_modules'),
    path.join(projectRoot, 'configs'),
    path.join(projectRoot, '.git'),
  ]
  if (forbidden.some((f) => newRoot === f || newRoot.startsWith(f + path.sep))) {
    throw new Error('数据目录不能设置在项目源码/依赖/配置目录内')
  }

  // 创建目录并测试可写性
  fs.mkdirSync(newRoot, { recursive: true })
  const probe = path.join(newRoot, `.write-test-${Date.now()}`)
  try {
    fs.writeFileSync(probe, 'ok')
    fs.rmSync(probe)
  } catch {
    throw new Error('目录不可写，请检查权限或更换目录')
  }

  const newDbPath = path.join(newRoot, 'drama.db')
  const newStoragePath = path.join(newRoot, 'static')

  // 迁移旧数据（复制）
  if (opts.migrate !== false) {
    closeDatabase()
    try {
      if (fs.existsSync(oldDbPath)) copyDbFiles(oldDbPath, newDbPath)
      if (fs.existsSync(oldStoragePath)) {
        fs.cpSync(oldStoragePath, newStoragePath, { recursive: true })
      }
      const oldTracesPath = path.join(oldRoot, 'traces')
      if (fs.existsSync(oldTracesPath)) {
        fs.cpSync(oldTracesPath, path.join(newRoot, 'traces'), { recursive: true })
      }
    } catch (err) {
      // 迁移失败回滚：重新打开旧数据库
      try { reopenDatabase(); rebuildDb() } catch {}
      throw new Error(`迁移失败：${(err as Error).message}`)
    }
  }

  // 更新数据根目录（写标记文件，重启后依然生效）
  setDataRoot(newRoot)

  // 重新打开新数据库并重建建表（新库会自动执行迁移 SQL）
  reopenDatabase()
  rebuildDb()

  return getStorageInfo()
}
