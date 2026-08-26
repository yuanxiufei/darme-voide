/**
 * Agent Skill 加载器
 * 支持 DB 配置（用户可控）+ 硬编码默认值（兜底）
 */
import { readFileSync, existsSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseSkill, renderSkill, type ParsedSkill } from './skill-parser.js'

// ── 硬编码默认映射（DB 无配置时使用） ─────────────────
export const AGENT_SKILL_MAP: Record<string, string[]> = {
  script_rewriter: ['script_rewriter'],
  extractor: ['extractor'],
  storyboard_breaker: ['storyboard_breaker', 'extractor'],
  voice_assigner: ['voice_assigner'],
  grid_prompt_generator: ['grid_prompt_generator'],
  orchestrator: [],
}

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

/**
 * 加载并解析指定 Skill 的 Markdown（分离 frontmatter 元数据与正文）
 */
function loadSkill(skillId: string): ParsedSkill | null {
  const filePath = resolve(SKILLS_DIR, skillId, 'SKILL.md')
  if (!existsSync(filePath)) return null
  try {
    return parseSkill(readFileSync(filePath, 'utf-8'), skillId)
  } catch {
    return null
  }
}

/**
 * 核心：为指定 Agent 加载 Skill 指令文本
 *
 * @param agentType - Agent 类型标识
 * @param dbSkillsRaw - 从 DB agent_configs.skills 读取的原始 JSON 字符串（可选）
 * @returns 拼接好的 Skill Markdown 文本，或 null（无可用 Skill）
 *
 * 优先级：
 *   1. dbSkillsRaw 有值 → 解析后只加载 enabled=true 的 Skill
 *   2. dbSkillsRaw 为空 → 回退 AGENT_SKILL_MAP[agentType] 硬编码默认值
 *   3. 都没有 → 返回 null
 */
export function loadAgentSkills(agentType: string, dbSkillsRaw?: string | null): string | null {
  let bindings: SkillBinding[] = []

  // 优先用 DB 配置
  if (dbSkillsRaw && dbSkillsRaw.trim()) {
    bindings = parseSkillsConfig(dbSkillsRaw)
    // 过滤出启用的，按优先级排序
    bindings = bindings.filter(b => b.enabled).sort((a, b) => (a.priority || 0) - (b.priority || 0))
  }

  // 回退到硬编码默认值
  if (!bindings.length) {
    const defaultIds = AGENT_SKILL_MAP[agentType]
    if (!defaultIds?.length) return null
    bindings = defaultBindings(defaultIds)
  }

  // 加载每个 Skill 的内容（只注入正文，frontmatter 元数据经 renderSkill 转为前置契约/协议字段段）
  const sections: string[] = []
  for (const binding of bindings) {
    const parsed = loadSkill(binding.id)
    if (parsed) {
      sections.push(renderSkill(parsed))
    }
  }

  if (!sections.length) return null

  return [
    '## Available Skills',
    '',
    ...sections.map(s => `---\n${s}\n`),
    '---',
  ].join('\n')
}
