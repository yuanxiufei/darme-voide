/**
 * SKILL.md 结构化解析器
 *
 * 对齐 PenguinHarness 架构参考文档第 11 章（自进化闭环）的 SKILL.md 规范：
 * - frontmatter 承载元数据（name / description / preconditions / protocol）
 * - 正文承载「如何做」的规范
 * - 前置契约（preconditions）：执行该 Skill 前必须满足的条件，缺失则停下询问
 * - 协议字段（protocol）：完成后必须在 YAML 协议块中额外汇报的字段，实现 skill 与 agent 解耦
 *
 * 单一事实来源：skills.ts（注入 Agent prompt）与 routes/skills.ts（元数据 API）共用本解析器，
 * 避免两处用脆弱正则重复解析 frontmatter。
 */
import { parse as parseYaml } from 'yaml'

export interface SkillMetadata {
  name: string
  description: string
  /** 前置契约：执行该 Skill 前必须满足的条件（空数组 = 无） */
  preconditions: string[]
  /** 输出协议字段：完成后需在 YAML 协议块中额外汇报的字段名（空数组 = 仅 status/summary） */
  protocol: string[]
}

export interface ParsedSkill {
  metadata: SkillMetadata
  /** 纯正文（不含 frontmatter），用于注入 Agent prompt */
  body: string
}

/** 匹配开头的 YAML frontmatter（--- 包裹），非贪婪，避免误匹配正文里的 --- 分隔线 */
const FRONTMATTER_RE = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/

/** 将 frontmatter 里的任意值规范化为字符串数组（兼容字符串 / 数组 / 省略三种写法） */
function toStrArray(v: unknown): string[] {
  if (v == null) return []
  if (Array.isArray(v)) return v.map(x => String(x).trim()).filter(Boolean)
  if (typeof v === 'string') {
    return v
      .split('\n')
      .map(s => s.trim().replace(/^-\s*/, ''))
      .filter(Boolean)
  }
  return [String(v)]
}

/**
 * 解析 SKILL.md 原始文本为元数据 + 正文。
 * frontmatter 解析失败时降级为空元数据（name 回退为 fallbackId），正文照常使用。
 */
export function parseSkill(content: string, fallbackId: string): ParsedSkill {
  const match = content.match(FRONTMATTER_RE)
  const rawFront = match ? match[1] : ''
  const body = match ? content.slice(match[0].length) : content

  let fm: Record<string, unknown> = {}
  if (rawFront.trim()) {
    try {
      const parsed = parseYaml(rawFront)
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        fm = parsed as Record<string, unknown>
      }
    } catch {
      fm = {}
    }
  }

  const metadata: SkillMetadata = {
    name: typeof fm.name === 'string' && fm.name.trim() ? fm.name.trim() : fallbackId,
    description: typeof fm.description === 'string' ? fm.description.trim() : '',
    preconditions: toStrArray(fm.preconditions),
    protocol: toStrArray(fm.protocol),
  }

  return { metadata, body: body.trim() }
}

/**
 * 将解析后的 Skill 渲染为注入 Agent prompt 的文本：
 * 正文 + 前置契约段 + 输出协议字段段。
 *
 * 协议字段段与 protocol.ts 的 buildProtocolContract（全局 status/summary 契约）衔接——
 * skill 只声明「额外」字段，全局契约保证 status/summary 始终存在，二者解耦。
 */
export function renderSkill(parsed: ParsedSkill): string {
  const parts: string[] = [parsed.body]

  if (parsed.metadata.preconditions.length) {
    parts.push(
      '## 前置契约（执行前必须满足）\n' +
        '以下条件不满足时，停下并向用户说明缺失项，不要臆造数据继续执行：\n' +
        parsed.metadata.preconditions.map(p => `- ${p}`).join('\n')
    )
  }

  if (parsed.metadata.protocol.length) {
    parts.push(
      '## 输出协议字段\n' +
        '在收尾的 YAML 协议块中，除 status / summary 外，还必须额外汇报以下字段：\n' +
        parsed.metadata.protocol.map(p => `- ${p}`).join('\n')
    )
  }

  return parts.join('\n\n')
}
