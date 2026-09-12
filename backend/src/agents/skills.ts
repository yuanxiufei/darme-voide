/**
 * Agent Skill 加载器
 * 支持 DB 配置（用户可控）+ 硬编码默认值（兜底）
 */
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseSkill, renderSkill, type ParsedSkill } from './skill-parser.js'

// ── 默认绑定：由 SKILL.md 自己声明，代码里不维护任何「谁绑谁」的映射 ─────
//
// skill 在 frontmatter 里自描述：
//     agents:   [storyboard_breaker]   # 默认注入哪些 Agent（不写 = 不默认注入，仅按需手动绑）
//     priority: 20                     # 注入顺序，越小越靠前（默认 100）
//
// 于是「新增 / 调整默认绑定」= 改 SKILL.md，**零代码改动**；绑定关系写在文件里，
// 与 skill 同生共死（不会出现"代码里绑了一个不存在的 skill"这种漂移）。
//
// ## 目录约定（决定 skill 属于「自有」还是「外部库」，见 listCoreSkillIds）
//     skills/<name>/SKILL.md           → 自有（core）：与工作流、工具名、字段契约对齐，可默认注入
//     skills/<lib>/library.yaml        → 外部技能库（vendor）：**库由声明文件识别**，
//                                        库标识取声明里的 name（与目录名解耦），库内 skill 不默认注入
//     详见 skills/README.md 与 routes/skills.ts 的 readLibrary / classify
//
// 外部库**不默认注入**，三条理由：
//   1. 体量失控：单文件最大 42 KB，曾使 storyboard_breaker 单次注入 ≈68 KB、
//      grid_prompt_generator ≈81 KB，且 DEFAULT_PROMPTS 里已有同主题规范 → 重复且烧 token；
//   2. 语义错位：它们是"按触发词独立启动"的会话式技能（自带 STEP 1/2/3 与 "Not for: ..."），
//      不是"常驻注入"的提示词层；
//   3. 注入不完整：其正文大量引用 references/*.md，而加载器只读 SKILL.md
//      → agent 顺着引用找文件必然落空，等于注入半截指令。
// 仍可在前端「Agent 配置 → 绑定 Skills」按需手动启用（DB agent_configs.skills 优先于默认绑定）。

// ── 类型定义 ────────────────────────────────────────────
interface SkillBinding {
  id: string
  enabled?: boolean   // 默认 true
  priority?: number   // 默认按数组顺序
}

/**
 * 解析 DB 存储的 skills JSON 字符串
 * 返回标准化后的 SkillBinding 数组，过滤掉无效项
 */
function parseSkillsConfig(raw: string | null | undefined): SkillBinding[] {
  if (!raw) return []
  try {
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter((item): item is SkillBinding =>
        typeof item === 'object' && item !== null && typeof (item as SkillBinding).id === 'string'
      )
      .map(item => ({
        id: item.id,
        enabled: item.enabled !== false,
        priority: typeof item.priority === 'number' ? item.priority : 0,
      }))
  } catch {
    return []
  }
}

/**
 * 将默认 skillId 列表转为 SkillBinding 格式（用于回退场景）
 */
function defaultBindings(skillIds: string[]): SkillBinding[] {
  return skillIds.map((id, index) => ({ id, enabled: true, priority: index + 1 }))
}

// skills/ 目录定位：从本文件位置上溯到项目根（与 process.cwd() 解耦，
// 兼容 Docker 容器 WORKDIR /app 及任意 cwd 启动），对齐 routes/skills.ts 的写法
const SKILLS_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../../../skills')

/** SKILL.md 解析缓存（按 mtimeMs 失效）：默认绑定查询会在每次 Agent 生成时扫一遍目录 */
const parsedCache = new Map<string, { mtimeMs: number; parsed: ParsedSkill | null }>()

/**
 * 加载并解析指定 Skill 的 Markdown（分离 frontmatter 元数据与正文），带 mtime 缓存。
 * 文件缺失或读取失败 → null（调用方据此产出「未注入（缺失）」诊断）。
 */
function loadSkill(skillId: string): ParsedSkill | null {
  const filePath = resolve(SKILLS_DIR, skillId, 'SKILL.md')
  let mtimeMs: number
  try {
    mtimeMs = statSync(filePath).mtimeMs
  } catch {
    parsedCache.delete(skillId)
    return null
  }

  const hit = parsedCache.get(skillId)
  if (hit && hit.mtimeMs === mtimeMs) return hit.parsed

  let parsed: ParsedSkill | null = null
  try {
    parsed = parseSkill(readFileSync(filePath, 'utf-8'), skillId)
  } catch {
    parsed = null
  }
  parsedCache.set(skillId, { mtimeMs, parsed })
  return parsed
}

/**
 * 扫描 skills/ 顶层，返回「自有 skill」id（= 目录自身含 SKILL.md 的那些）。
 * 顶层不含 SKILL.md 的目录视为**外部技能库容器**（vendor），其子项不进 core 列表。
 * 这条规则不依赖任何外部库的名字，换库 / 加库都无需改代码。
 */
export function listCoreSkillIds(): string[] {
  if (!existsSync(SKILLS_DIR)) return []
  return readdirSync(SKILLS_DIR, { withFileTypes: true })
    .filter(entry => entry.isDirectory() && existsSync(resolve(SKILLS_DIR, entry.name, 'SKILL.md')))
    .map(entry => entry.name)
    .sort()
}

/**
 * 某 Agent 的默认 skill 绑定 = 所有 frontmatter 里声明了该 agent 的自有 skill。
 * 按 priority 升序（同序按 id 字典序），保证注入顺序稳定可复现。
 */
export function resolveDefaultSkills(agentType: string): string[] {
  if (!agentType) return []
  return listCoreSkillIds()
    .map(id => ({ id, parsed: loadSkill(id) }))
    .filter(({ parsed }) => parsed?.metadata.agents.includes(agentType))
    .map(({ id, parsed }) => ({ id, priority: parsed!.metadata.priority }))
    .sort((a, b) => a.priority - b.priority || a.id.localeCompare(b.id))
    .map(entry => entry.id)
}

/**
 * 核心：为指定 Agent 加载 Skill 指令文本
 *
 * @param agentType - Agent 类型标识
 * @param dbSkillsRaw - 从 DB agent_configs.skills 读取的原始 JSON 字符串（可选）
 * @returns 拼接好的 Skill Markdown 文本，或 null（无可用 Skill）
 *
 * 优先级：
 *   1. dbSkillsRaw 能解析出配置项 → 只加载 enabled=true 的 Skill（**全部关闭则不注入，不回退默认**）
 *   2. dbSkillsRaw 为空 / 空数组 / 解析失败 → 回退该 Agent 的默认绑定（SKILL.md frontmatter `agents:`）
 *   3. 都没有 → 返回 null
 */
/**
 * Skill 注入体量上限（字符数，默认 60000 ≈ 3 万 token 量级）。
 * 目的：给"skill 越挂越多"设一道硬闸 —— 超出的 skill 按优先级被跳过并在注入文本里注明，
 * 便于排查"某个 skill 为什么没生效"。用环境变量 AGENT_SKILL_BUDGET 覆盖，设为 0 关闭限制。
 */
export const SKILL_CHAR_BUDGET = (() => {
  const raw = Number(process.env.AGENT_SKILL_BUDGET)
  return Number.isFinite(raw) && raw > 0 ? raw : 60_000
})()

export function loadAgentSkills(agentType: string, dbSkillsRaw?: string | null): string | null {
  let bindings: SkillBinding[] = []
  // DB 里存在「配置项」= 用户已做过选择，其意图优先 —— 即使最终一个都不注入，也不回退默认
  let userConfigured = false

  // 优先用 DB 配置
  if (dbSkillsRaw && dbSkillsRaw.trim()) {
    const parsed = parseSkillsConfig(dbSkillsRaw)
    // ⚠️ 判据必须是「解析出的配置项是否非空」，不能看「过滤 enabled 后是否为空」：
    // 后者会把「用户全部取消勾选」误判成「从来没配过」→ 回退默认绑定，
    // 于是"一个都不注入"反向执行为"注入全部默认 skill"（前端勾选框与面板文案均承诺"仅勾选的会注入"）。
    if (parsed.length) {
      userConfigured = true
      bindings = parsed.filter(b => b.enabled).sort((a, b) => (a.priority || 0) - (b.priority || 0))
    }
  }

  // 无用户配置（DB 为 null / 空数组 / 解析失败）→ 回退 skill 自描述默认绑定
  if (!userConfigured) {
    const defaultIds = resolveDefaultSkills(agentType)
    if (!defaultIds.length) return null
    bindings = defaultBindings(defaultIds)
  }

  // 用户显式关闭了全部 skill → 不注入任何 skill（这是"最小化注入"的正当诉求）
  if (!bindings.length) return null

  // 加载每个 Skill 的内容（只注入正文，frontmatter 元数据经 renderSkill 转为前置契约/协议字段段）
  // 同时执行体量预算闸：按优先级顺序累加，超出预算者跳过并记录 —— 保证无论默认映射
  // 还是用户/DB 挂了多长的 skill，都不会把 system prompt 撑爆（可诊断、可回退）。
  const sections: string[] = []
  const skipped: string[] = []
  let used = 0
  for (const binding of bindings) {
    const parsed = loadSkill(binding.id)
    if (!parsed) {
      skipped.push(`${binding.id}（缺失）`)
      continue
    }
    const rendered = renderSkill(parsed)
    if (SKILL_CHAR_BUDGET > 0 && used + rendered.length > SKILL_CHAR_BUDGET) {
      skipped.push(`${binding.id}（超预算）`)
      continue
    }
    used += rendered.length
    sections.push(rendered)
  }

  if (!sections.length) return null

  const notice = skipped.length
    ? [`> 未注入：${skipped.join('、')}（文件缺失或超出 ${SKILL_CHAR_BUDGET} 字符预算）`, '']
    : []

  return [
    '## Available Skills',
    '',
    ...notice,
    ...sections.map(s => `---\n${s}\n`),
    '---',
  ].join('\n')
}

/**
 * 扫描 skills/ 目录，返回全部可用 skill id（**含外部库的嵌套目录**，如 `<lib>/installed/<name>`）。
 * 用于「全部 Skill」浏览与用户手动绑定。
 *
 * 注意：**默认注入请用 `resolveDefaultSkills`（只含自有 core skill）**。
 * 外部库是"按触发词独立启动"的会话式技能，绑定给常驻 Agent 只会注入半截指令（见文件头注释）。
 */
export function listSkillIds(): string[] {
  if (!existsSync(SKILLS_DIR)) return []
  const out: string[] = []
  function scan(dir: string, prefix = '') {
    const entries = readdirSync(dir, { withFileTypes: true })
    for (const entry of entries) {
      if (!entry.isDirectory()) continue
      const full = resolve(dir, entry.name)
      const id = prefix ? `${prefix}/${entry.name}` : entry.name
      if (existsSync(resolve(full, 'SKILL.md'))) out.push(id)
      scan(full, id)
    }
  }
  scan(SKILLS_DIR)
  return out
}
