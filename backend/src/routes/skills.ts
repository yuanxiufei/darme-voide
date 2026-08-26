import { Hono } from 'hono'
import type { Context } from 'hono'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { success, badRequest } from '../utils/response.js'
import { parseSkill } from '../agents/skill-parser.js'

const app = new Hono()
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SKILLS_DIR = path.resolve(__dirname, '../../../skills')

/** 验证 skill id 是安全的，防止路径遍历攻击 */
function validateSkillId(rawId: string): string | null {
  // 剔除 null 字节 / 换行等危险字符
  const sanitized = rawId.replace(/[\0\r\n]/g, '')
  // 只允许字母、数字、横线、下划线、点号、斜线，不允许连续 ".." 或绝对路径
  if (!/^[a-zA-Z0-9_\-./]+$/.test(sanitized)) return null
  if (sanitized.startsWith('/') || sanitized.startsWith('\\')) return null
  if (sanitized.includes('..')) return null
  // 最终解析路径必须在 SKILLS_DIR 内
  const resolved = path.resolve(SKILLS_DIR, sanitized)
  if (!resolved.startsWith(SKILLS_DIR + path.sep) && resolved !== SKILLS_DIR) return null
  return sanitized
}

/** 安全获取 skill 文件路径，id 必须通过 validateSkillId 校验 */
function safeSkillPath(id: string): string {
  return path.join(SKILLS_DIR, id, 'SKILL.md')
}

/** 安全获取 skill 目录路径 */
function safeSkillDir(id: string): string {
  return path.join(SKILLS_DIR, id)
}

/** 从通配符路由提取匹配到的 skill id（从 routePath 动态推导挂载前缀，避免硬编码 '/api/v1/skills/'） */
function wildcardId(c: Context): string {
  const routePath = c.req.routePath // 形如 '/api/v1/skills/*'
  const star = routePath.indexOf('*')
  const prefix = star >= 0 ? routePath.slice(0, star) : ''
  return c.req.path.slice(prefix.length)
}

// GET /skills — List all skills (recursive, supports nested dirs)
app.get('/', async (c) => {
  try {
  const skills: { id: string; name: string; description: string; preconditions: string[]; protocol: string[] }[] = []

  if (!fs.existsSync(SKILLS_DIR)) {
    return success(c, skills)
  }

  function scanDir(dir: string, prefix = '') {
    const entries = fs.readdirSync(dir, { withFileTypes: true })
    for (const entry of entries) {
      if (!entry.isDirectory()) continue
      const fullPath = path.join(dir, entry.name)
      const skillPath = path.join(fullPath, 'SKILL.md')
      if (fs.existsSync(skillPath)) {
        const content = fs.readFileSync(skillPath, 'utf-8')
        const parsed = parseSkill(content, entry.name)
        const id = prefix ? `${prefix}/${entry.name}` : entry.name
        skills.push({
          id,
          name: parsed.metadata.name,
          description: parsed.metadata.description,
          preconditions: parsed.metadata.preconditions,
          protocol: parsed.metadata.protocol,
        })
      }
      // Always recurse — nested skills may exist even if this dir has SKILL.md
      scanDir(fullPath, prefix ? `${prefix}/${entry.name}` : entry.name)
    }
  }

  scanDir(SKILLS_DIR)
  return success(c, skills)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// GET /skills/:id — Get skill content
app.get('/*', async (c) => {
  try {
  const rawId = wildcardId(c)
  const id = validateSkillId(rawId)
  if (!id) return badRequest(c, 'Invalid skill id')
  const skillPath = safeSkillPath(id)
  if (!fs.existsSync(skillPath)) return badRequest(c, 'Skill not found')
  const content = fs.readFileSync(skillPath, 'utf-8')
  return success(c, { id, content })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// PUT /skills/:id — Update skill content
app.put('/*', async (c) => {
  try {
  const rawId = wildcardId(c)
  const id = validateSkillId(rawId)
  if (!id) return badRequest(c, 'Invalid skill id')
  const body = await c.req.json()
  const skillDir = safeSkillDir(id)
  const skillPath = safeSkillPath(id)
  if (!fs.existsSync(skillDir)) fs.mkdirSync(skillDir, { recursive: true })
  fs.writeFileSync(skillPath, body.content, 'utf-8')
  return success(c)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// POST /skills — Create new skill directory
app.post('/', async (c) => {
  try {
  const body = await c.req.json()
  const { id, name, description } = body
  if (!id) return badRequest(c, 'Skill id is required')
  const validId = validateSkillId(id)
  if (!validId) return badRequest(c, 'Invalid skill id')

  const skillDir = safeSkillDir(validId)
  if (fs.existsSync(skillDir)) return badRequest(c, 'Skill already exists')

  fs.mkdirSync(skillDir, { recursive: true })
  const content = `---
name: ${name || validId}
description: ${description || ''}
preconditions: []
protocol: []
---

# ${name || validId}

Write your skill content here.
`
  fs.writeFileSync(path.join(skillDir, 'SKILL.md'), content, 'utf-8')
  return success(c, { id: validId, name: name || validId, description: description || '' })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// DELETE /skills/:id — Delete skill directory
app.delete('/*', async (c) => {
  try {
  const rawId = wildcardId(c)
  const id = validateSkillId(rawId)
  if (!id) return badRequest(c, 'Invalid skill id')
  const skillDir = safeSkillDir(id)
  if (!fs.existsSync(skillDir)) return badRequest(c, 'Skill not found')
  fs.rmSync(skillDir, { recursive: true, force: true })
  return success(c)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

export default app
