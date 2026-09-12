import { Hono } from 'hono'
import type { Context } from 'hono'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { isNull } from 'drizzle-orm'
import { success, badRequest } from '../utils/response.js'
import { parseSkill, renderSkill } from '../agents/skill-parser.js'
import { AGENT_PHASES, getDefaultName, getHostToolNames } from '../agents/index.js'
import { listSkillIds, resolveDefaultSkills, SKILL_CHAR_BUDGET } from '../agents/skills.js'
import { parse as parseYaml } from 'yaml'
import { db, schema } from '../db/index.js'

const app = new Hono()
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SKILLS_DIR = path.resolve(__dirname, '../../../skills')

/** 外部技能库声明文件名：顶层目录含此文件 = 一个技能库（vendor） */
const LIBRARY_FILE = 'library.yaml'

/** 技能库元数据（来自 `skills/<lib>/library.yaml`） */
interface LibraryMeta {
  /** 库标识：API 与前端分组的稳定 key（声明里的 name，缺省 = 目录名） */
  id: string
  /** 展示名：可随时改，不影响任何引用（声明里的 label，缺省 = id） */
  label: string
  description: string
}

/**
 * 读取技能库声明。**库靠显式声明识别，不靠目录名、也不靠层级猜测**：
 *   skills/<lib>/library.yaml   → `<lib>` 是外部技能库（vendor），库内 skill 一律不默认注入
 *   skills/<name>/SKILL.md      → 项目自有 skill（core），可由 frontmatter `agents:` 默认注入
 * 于是「加库 / 换库 / 改展示名」都是纯文件操作，零代码改动。
 */
function readLibrary(dirName: string): LibraryMeta {
  const fallback: LibraryMeta = { id: dirName, label: dirName, description: '' }
  const file = path.join(SKILLS_DIR, dirName, LIBRARY_FILE)
  if (!fs.existsSync(file)) return fallback
  try {
    const raw = parseYaml(fs.readFileSync(file, 'utf-8')) as Record<string, unknown> | null
    if (!raw || typeof raw !== 'object') return fallback
    const id = typeof raw.name === 'string' && raw.name.trim() ? raw.name.trim() : dirName
    const label = typeof raw.label === 'string' && raw.label.trim() ? raw.label.trim() : id
    const description = typeof raw.description === 'string' ? raw.description.trim() : ''
    return { id, label, description }
  } catch {
    return fallback // 声明损坏不该让整个列表接口失败
  }
}

/** 扫描 skills/ 顶层，列出全部外部技能库（按库标识排序） */
function listLibraries(): LibraryMeta[] {
  if (!fs.existsSync(SKILLS_DIR)) return []
  return fs.readdirSync(SKILLS_DIR, { withFileTypes: true })
    .filter(e => e.isDirectory() && fs.existsSync(path.join(SKILLS_DIR, e.name, LIBRARY_FILE)))
    .map(e => readLibrary(e.name))
    .sort((a, b) => a.id.localeCompare(b.id))
}

/**
 * 推导 skill 来源（换库 / 加库 / 改库名零代码改动）：
 *   core   —— 顶层目录自身含 SKILL.md，即项目自有 skill（与工作流、工具名、字段契约对齐）
 *   vendor —— 顶层目录含 library.yaml，即外部技能库；source = 库标识（来自声明，可与目录名不同）
 */
function classify(id: string): { category: 'core' | 'vendor'; source: string | null } {
  if (!id.includes('/')) return { category: 'core', source: null }
  const top = id.split('/')[0]
  // 顶层目录自身含 SKILL.md → 自有 skill 的子目录，其子项仍按自有处理
  if (fs.existsSync(path.join(SKILLS_DIR, top, 'SKILL.md'))) return { category: 'core', source: null }
  // 含库声明 → 外部技能库
  if (fs.existsSync(path.join(SKILLS_DIR, top, LIBRARY_FILE))) {
    return { category: 'vendor', source: readLibrary(top).id }
  }
  // 兜底：无声明文件的容器目录，退化为「目录名即库名」
  return { category: 'vendor', source: top }
}

/** 反向查询 skill 绑定关系：skillId → agent_type 列表（含已禁用的）
 *  优先 DB agent_configs.skills（用户显式配置）；DB 未配置（null）的 agent 回退该 skill
 *  frontmatter `agents:` 里的自描述声明（与 loadAgentSkills 的兜底共用 resolveDefaultSkills，保证一致）。 */
function loadSkillBindings(): Map<string, string[]> {
  const map = new Map<string, string[]>()
  // 自描述默认绑定兜底（只含自有 skill，外部库不默认注入）
  for (const agent of Object.keys(AGENT_PHASES)) {
    for (const id of resolveDefaultSkills(agent)) {
      const list = map.get(id) ?? []
      if (!list.includes(agent)) list.push(agent)
      map.set(id, list)
    }
  }
  const rows = db.select().from(schema.agentConfigs).where(isNull(schema.agentConfigs.deletedAt)).all()
  for (const row of rows) {
    if (!row.skills) continue // 未配置 → 保留默认映射
    try {
      const arr = JSON.parse(row.skills)
      if (!Array.isArray(arr)) continue
      // 用户显式配置：先移除该 agent 的默认绑定，再按用户配置重建
      for (const agents of map.values()) {
        const i = agents.indexOf(row.agentType)
        if (i >= 0) agents.splice(i, 1)
      }
      for (const item of arr) {
        if (!item || typeof item.id !== 'string') continue
        const list = map.get(item.id) ?? []
        if (!list.includes(row.agentType)) list.push(row.agentType)
        map.set(item.id, list)
      }
    } catch { /* 忽略损坏的 skills JSON */ }
  }
  return map
}

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

/**
 * 删除保护判定（删除闸）。保护对象 = **被默认注入的项目资产**，判据两项都要满足：
 *   ① 顶层目录（id 不含 '/'）—— 「Agent 根 skill」或「提示词库」；
 *   ② frontmatter `agents:` 声明非空 —— 确实参与默认注入，删除会直接改变 Agent 行为，
 *      其中未纳入版本控制的部分（如 prompt-style-library）删掉即**永久丢失**。
 * 用户自建 skill（`skills/<agent>/<name>/`，id 含 '/'）不受保护；**顶层但未声明 agents**
 * （手工建目录 / 接口直连创建，不参与默认注入）同样可删 —— 否则合法删除会被永久误拒。
 * `agents` 传 undefined（文件读不到 / 解析失败）时保守按资产处理。
 */
function isProtectedSkill(id: string, agents?: string[]): boolean {
  if (id.includes('/')) return false
  if (!agents) return true
  return agents.length > 0
}

/** 统计 skill 目录下 references/ 内的参考文件数（注入时不会展开 → 前端据此提示"引用不可达"） */
function countReferenceFiles(skillDir: string): number {
  const refDir = path.join(skillDir, 'references')
  if (!fs.existsSync(refDir)) return 0
  let count = 0
  const walk = (dir: string) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.isDirectory()) walk(path.join(dir, entry.name))
      else count++
    }
  }
  try { walk(refDir) } catch { /* 读取失败按 0 处理，不阻断列表 */ }
  return count
}

/**
 * 真实注入体量（字符）= `renderSkill` 后的长度，与 `loadAgentSkills` 的预算闸**同口径**
 * （不能直接用文件字节数：前置契约 / 协议字段段的拼接同样计入注入）。
 */
function injectedChars(parsed: ReturnType<typeof parseSkill>): number {
  try { return renderSkill(parsed).length } catch { return 0 }
}

/** 读取并计算单个 skill 的注入体量（文件缺失 / 解析失败 → 0） */
function injectedCharsOf(id: string): number {
  const file = safeSkillPath(id)
  if (!fs.existsSync(file)) return 0
  try { return injectedChars(parseSkill(fs.readFileSync(file, 'utf-8'), id)) } catch { return 0 }
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
  const bindings = loadSkillBindings()
  const libraryLabels = new Map(listLibraries().map(l => [l.id, l.label]))
  // 宿主已注册的工具集（懒计算并缓存），用于判定各 skill 依赖的工具是否被满足
  const hostTools = getHostToolNames()
  const skills: {
    id: string
    name: string
    description: string
    preconditions: string[]
    protocol: string[]
    /** 来源分类：core（项目自有 skill）/ vendor（外部技能库） */
    category: 'core' | 'vendor'
    /** 所属技能库标识（仅 vendor 有，取自 library.yaml 的 name）；core 为 null */
    source: string | null
    /** 所属技能库展示名（取自 library.yaml 的 label，供 UI 直接渲染）；core 为 null */
    sourceLabel: string | null
    /** frontmatter workflows: 声明的适用工作流/阶段 */
    workflows: string[]
    /** frontmatter agents: 自描述的默认绑定 Agent（决定默认注入） */
    agents: string[]
    /** frontmatter priority: 注入顺序（越小越靠前） */
    priority: number
    /** 反向查询：绑定了该 skill 的 agent_type 列表（含 DB 显式配置） */
    boundAgents: string[]
    /** 工作流制作阶段 = 绑定 Agent 对应阶段 + frontmatter 声明（去重） */
    phases: string[]
    /** 实际注入体量（字符，含前置契约/协议字段段；与注入时的预算闸同口径） */
    charCount: number
    /** 目录内 references/ 的参考文件数（加载器只读 SKILL.md ⇒ 这些文件对 Agent 不可达） */
    referenceCount: number
    /** 是否为受保护的顶层自有 skill（前端据此隐藏删除入口，避免误删项目资产） */
    protected: boolean
    /** frontmatter `allowed-tools:` 声明的工具名（声明式依赖，未做可用性判定） */
    allowedTools: string[]
    /** 正文中引用的外来宿主工具名（散落在「先调用 xxx」这类指令里，与 allowedTools 互补） */
    foreignToolRefs: string[]
    /**
     * 该 skill **依赖、但本项目（宿主）未注册**的工具（= allowedTools ∪ foreignToolRefs，再减去宿主已有）。
     * 非空意味着它的指令在教 Agent 调用本项目不存在的工具 ⇒ 绑定后必然无法按其指令执行，
     * UI 需要显式提示（外部库多按另一套宿主平台编写，这是常见情况而非异常）。
     * 注意 MCP 外部工具是运行时异步发现的，可能存在误报 ⇒ 定位为**预警**，不做硬拦截。
     */
    missingTools: string[]
  }[] = []

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
        const boundAgents = bindings.get(id) || []
        const { category, source } = classify(id)
        const allowedTools = parsed.metadata.allowedTools
        // 该 skill 真实依赖的工具 = frontmatter 声明 ∪ 正文引用 —— 二者都可能指向宿主未注册的工具
        const requiredTools = Array.from(new Set([...allowedTools, ...parsed.foreignToolRefs]))
        skills.push({
          id,
          name: parsed.metadata.name,
          description: parsed.metadata.description,
          preconditions: parsed.metadata.preconditions,
          protocol: parsed.metadata.protocol,
          category,
          source,
          sourceLabel: source ? (libraryLabels.get(source) ?? source) : null,
          workflows: parsed.metadata.workflows,
          agents: parsed.metadata.agents,
          priority: parsed.metadata.priority,
          allowedTools,
          foreignToolRefs: parsed.foreignToolRefs,
          missingTools: requiredTools.filter(t => !hostTools.has(t)),
          charCount: injectedChars(parsed),
          referenceCount: countReferenceFiles(fullPath),
          protected: isProtectedSkill(id, parsed.metadata.agents),
          boundAgents,
          phases: Array.from(new Set([
            ...boundAgents.map(a => AGENT_PHASES[a]).filter(Boolean),
            ...parsed.metadata.workflows,
          ])),
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

// GET /skills/meta — 侧栏元信息：主流程 Agent + 外部技能库（全部由目录结构推导，前端不再硬编码）
// 注意：必须注册在 GET /* 之前，否则 'meta' 会被当成 skill id
app.get('/meta', async (c) => {
  try {
    const declared = listLibraries()
    const byId = new Map(declared.map(l => [l.id, l]))
    let coreCount = 0
    const counters = new Map<string, number>()
    for (const id of listSkillIds()) {
      const { category, source } = classify(id)
      if (category === 'vendor' && source) {
        counters.set(source, (counters.get(source) ?? 0) + 1)
        // 兜底库（顶层目录无 library.yaml）：补一个库条目，保证「core + 各库 = 总数」
        // 自洽，且这些 skill 在侧栏有入口（否则会从计数里消失、UI 也不可达）
        if (!byId.has(source)) byId.set(source, { id: source, label: source, description: '' })
      } else coreCount++
    }

    return success(c, {
      /** 「按 Agent 分组」：主流程 Agent + 各自默认绑定的 skill 数（含阶段名，供 UI 展示） */
      agents: Object.keys(AGENT_PHASES).map(type => {
        const ids = resolveDefaultSkills(type)
        return {
          type,
          label: getDefaultName(type),
          phase: AGENT_PHASES[type],
          skillCount: ids.length,
          /** 默认绑定的实际注入体量合计（与 charBudget 对比即可看出余量/超预算风险） */
          charCount: ids.reduce((sum, id) => sum + injectedCharsOf(id), 0),
        }
      }),
      /** 注入体量硬闸（字符）：超出者按优先级跳过（见 loadAgentSkills），供前端做超预算预警 */
      charBudget: SKILL_CHAR_BUDGET,
      /** 外部技能库（vendor）：由 skills/<lib>/library.yaml 显式声明，与目录名解耦 ——
       *  加库 / 换库 / 改展示名皆为零代码的文件操作 */
      sources: [...byId.values()]
        .map(lib => ({
          id: lib.id,
          label: lib.label,
          description: lib.description,
          skillCount: counters.get(lib.id) ?? 0,
          /** 是否带 library.yaml 声明（false = 无声明的兜底库，展示名即目录名） */
          declared: declared.some(d => d.id === lib.id),
        }))
        .sort((a, b) => a.id.localeCompare(b.id)),
      coreCount,
    })
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
  // 保存后校验 frontmatter：正文里的头部被误删/改坏，会让 name/description 丢失、
  // agents 声明失效（该 Skill 静默退出默认注入）—— 这类"静默行为变更"必须回告前端
  let warning: string | undefined
  if (!/^\s*---/.test(body.content || '')) {
    warning = '内容缺少 frontmatter（--- 包裹的头部），name / description / agents 声明将无法识别。'
  } else if (parseSkill(body.content, id).metadata.agents.length === 0) {
    warning = 'frontmatter 的 agents 声明为空，该 Skill 不再默认注入任何 Agent（可在「Agent 配置 → 绑定 Skills」中手动启用）。'
  }
  return success(c, warning ? { warning } : undefined)
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
# agents: 声明默认注入哪些 Agent（本项目为 script_rewriter / extractor / storyboard_breaker /
#         voice_assigner / grid_prompt_generator）；留空 = 不默认注入，仅在前端按需手动绑定
agents: []
# priority: 注入顺序，越小越靠前（缺省 100）
priority: 100
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
  // 受保护 = 顶层且被 frontmatter `agents:` 默认注入的资产（详见 isProtectedSkill）；
  // 解析失败传 undefined → 保守保护，宁可拒删也不误删项目资产
  let declaredAgents: string[] | undefined
  try { declaredAgents = parseSkill(fs.readFileSync(safeSkillPath(id), 'utf-8'), id).metadata.agents }
  catch { declaredAgents = undefined }
  if (isProtectedSkill(id, declaredAgents)) {
    return badRequest(c, `「${id}」是项目自有 Skill（由 frontmatter agents 声明默认注入，受删除保护），不支持删除。如需停用，请在「Agent 配置 → 绑定 Skills」中取消勾选。`)
  }
  fs.rmSync(skillDir, { recursive: true, force: true })
  return success(c)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

export default app
