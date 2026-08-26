/**
 * 底层 better-sqlite3 连接实例
 * 独立模块，供 db/index.ts（drizzle-orm）和 queryHelper.ts（直接 SQL 等）共享
 * 支持运行时切换数据库文件（closeDatabase / reopenDatabase）
 */
import Database from 'better-sqlite3'
import path from 'path'
import fs from 'fs'
import { getDbPath } from '../config.js'

type SqliteDatabase = InstanceType<typeof Database>

let sqliteInstance: SqliteDatabase | null = null

function openDb(dbPath: string): SqliteDatabase {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true })
  const inst = new Database(dbPath, { timeout: 30000 })
  inst.pragma('journal_mode = WAL')
  inst.pragma('busy_timeout = 30000')
  return inst
}

sqliteInstance = openDb(getDbPath())

/** 获取当前数据库实例（惰性重开，避免空引用） */
export function getSqlite(): SqliteDatabase {
  if (!sqliteInstance) sqliteInstance = openDb(getDbPath())
  return sqliteInstance
}

/** 关闭当前数据库连接（迁移目录前调用，确保 WAL 落盘） */
export function closeDatabase(): void {
  if (sqliteInstance?.open) {
    try { sqliteInstance.pragma('wal_checkpoint(TRUNCATE)') } catch {}
    try { sqliteInstance.close() } catch {}
  }
  sqliteInstance = null
}

/** 按当前配置路径重新打开数据库（迁移目录后调用，先关闭旧连接避免泄漏） */
export function reopenDatabase(): SqliteDatabase {
  closeDatabase()
  sqliteInstance = openDb(getDbPath())
  return sqliteInstance
}

// 兼容既有的 `import { sqlite }` 用法（queryHelper 等），透明转发到当前实例
export const sqlite: SqliteDatabase = new Proxy({} as SqliteDatabase, {
  get(_target, prop) {
    const inst = getSqlite()
    const v = (inst as any)[prop]
    if (typeof v === 'function') return v.bind(inst)
    return v
  },
  set(_target, prop, value) {
    ;(getSqlite() as any)[prop] = value
    return true
  },
})
