/**
 * Agent Skill 加载器
 * 支持 DB 配置（用户可控）+ 硬编码默认值（兜底）
 */
import { readFileSync, existsSync } from 'node:fs'
import { resolve } from 'node:path'

// ── 硬编码默认映射（DB 无配置时使用） ─────────────────
export const AGENT_SKILL_MAP: Record<string, string[]> = {
  script_rewriter: ['script_rewriter'],
  extractor: ['extractor'],
  storyboard_breaker: ['storyboard_breaker', 'extractor'],
  voice_assigner: ['voice_assigner'],
  grid_prompt_generator: ['grid_prompt_generator'],
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

/**
 * 加载指定 Skill 的 Markdown 内容
 */
function loadSkillContent(skillId: string): string | null {
  const filePath = resolve(process.cwd(), 'skills', skillId, 'SKILL.md')
  if (!existsSync(filePath)) return null
  try {
    return readFileSync(filePath, 'utf-8')
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

  // 加载每个 Skill 的内容
  const sections: string[] = []
  for (const binding of bindings) {
    const content = loadSkillContent(binding.id)
    if (content) {
      sections.push(content)
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
