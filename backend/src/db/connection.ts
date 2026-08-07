/**
 * 底层 better-sqlite3 连接实例
 * 独立模块，供 db/index.ts（drizzle-orm）和 queryHelper.ts（直接 SQL 等）共享
 */
import Database from 'better-sqlite3'
import path from 'path'
import { fileURLToPath } from 'url'
import fs from 'fs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const DB_PATH = process.env.DB_PATH || path.resolve(__dirname, '../../../data/drama.db')

fs.mkdirSync(path.dirname(DB_PATH), { recursive: true })

export const sqlite = new Database(DB_PATH, { timeout: 30000 })
sqlite.pragma('journal_mode = WAL')
sqlite.pragma('busy_timeout = 30000')
