/**
 * 参数化 SQL 查询辅助模块
 * drizzle-orm 的 db.all()/db.get()/db.run() 只接受 1 个参数（SQL 字符串），
 * 不能传递 ? 占位符的值。此模块直接使用底层 better-sqlite3 实例。
 */
import { sqlite } from './index.js'

export function qAll(sql: string, ...params: any[]): any[] {
  return (sqlite.prepare(sql) as any).all(...params)
}

export function qGet(sql: string, ...params: any[]): any {
  return (sqlite.prepare(sql) as any).get(...params)
}

export function qRun(sql: string, ...params: any[]): any {
  return (sqlite.prepare(sql) as any).run(...params)
}
