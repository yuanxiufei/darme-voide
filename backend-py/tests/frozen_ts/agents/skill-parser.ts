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
  /** 适用工作流/制作阶段（frontmatter `workflows:` 可选声明，如：剧本编写 / 分镜拆解） */
  workflows: string[]
  /**
   * 默认注入哪些 Agent（frontmatter `agents:`）。
   * **skill 自描述绑定关系**：写了即默认注入，绑定关系不再由代码维护（见 skills.ts）。
   * 空数组 = 不默认注入，只能在前端「Agent 配置 → 绑定 Skills」按需手动启用。
   */
  agents: string[]
  /** 注入顺序（frontmatter `priority:`，越小越靠前，默认 100） */
  priority: number
  /**
   * 该 skill 假设宿主提供的工具（frontmatter `allowed-tools:`）。
   * 外部技能库多按「另一套宿主平台」编写，声明的工具名在本项目未必注册；
   * 本解析器只负责解析，是否可用由 routes/skills.ts 拿宿主工具集比对后判定（产出 missingTools）。
   */
  allowedTools: string[]
}

export interface ParsedSkill {
  metadata: SkillMetadata
  /** 纯正文（不含 frontmatter），用于注入 Agent prompt */
  body: string
  /**
   * 正文中引用的**外来宿主工具**名（见 FOREIGN_TOOL_PREFIXES）。
   * 与 `metadata.allowedTools`（声明式依赖）互补 —— 二者并集才是该 skill 真正假设宿主提供的工具集；
   * 是否被满足由 routes/skills.ts 与宿主工具集比对后判定（产出 missingTools）。
   */
  foreignToolRefs: string[]
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
 * 将 frontmatter `allowed-tools` 规范化为工具名列表。
 * 必须同时兼容三种写法（外部库三种都出现过）：
 *   allowed-tools: [hub_read, hub_write]   → YAML 解析为数组
 *   allowed-tools: hub_read, hub_write     → 解析为**纯量字符串**（YAML 不按逗号切分，须自己切）
 *   allowed-tools:\n  - hub_read           → 换行列表
 * 故在数组/字符串归一后，再统一按「换行 + 逗号」切分，并剥掉列表符与方括号/引号。
 */
function toToolList(v: unknown): string[] {
  const parts = (Array.isArray(v) ? v : [v])
    .filter(x => x != null)
    .flatMap(x => String(x).split(/[\n,]/))
  const names = parts
    .map(s => s.trim()
      .replace(/^-\s*/, '')
      .replace(/^\[|\]$/g, '')
      .replace(/^["']|["']$/g, '')
      .trim())
    .filter(Boolean)
  return Array.from(new Set(names))
}

/**
 * 「外来宿主平台」的工具名前缀。
 *
 * 外部技能库按另一套宿主平台编写，其工具统一带 `hub_` 命名空间前缀；而本项目宿主工具
 * （`read_script_for_extraction` / `save_dedup_characters` / `generate_grid_prompt` …）
 * 无一以此开头 ⇒ 前缀命中即外来，实测 36 个 SKILL.md 零误报。
 *
 * 刻意**只按前缀识别**，不做「正文里未知 snake_case token 即工具」的宽泛猜测：正文中大量
 * 反引号 token 其实是字段名与文件名（`shot_type` / `bgm_url` / `prompt_budget` / `references/`…），
 * 宽泛匹配会把它们统统判成缺失工具，让预警彻底失去意义。
 * 代价：正文里裸写的 `read` / `write` / `task` 等平台通用工具识别不到（这类词在正文中噪声极大），
 * 它们只在 frontmatter `allowed-tools` 覆盖范围内（如 voice-clone 声明的 `read`）。
 */
const FOREIGN_TOOL_PREFIXES = ['hub_']

/**
 * 从**正文**提取外来宿主工具引用（反引号包裹，如「先调用 `hub_list_capabilities`」）。
 *
 * 与 frontmatter `allowed-tools` 互补，缺一不可：实测 13 个外部 skill 正文出现 `hub_*`，
 * 其中 6 个**根本没声明** `allowed-tools` ⇒ 只查声明会整片漏掉；另有 multi-shot / n-storyboard
 * 的正文调用的工具与其声明不一致（声明 `hub_submit_dag`，正文却调 `hub_query_dag_result`）。
 */
export function extractForeignToolRefs(body: string): string[] {
  const found = new Set<string>()
  for (const m of body.matchAll(/`([a-z][a-z0-9_]{2,})`/g)) {
    if (FOREIGN_TOOL_PREFIXES.some(p => m[1].startsWith(p))) found.add(m[1])
  }
  return Array.from(found).sort()
}

/** 将 frontmatter priority 规范化为正数（缺省 / 非法 → 100，越小越靠前） */
function normalizePriority(v: unknown): number {
  const n = Number(v)
  return Number.isFinite(n) && n > 0 ? n : 100
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
    workflows: toStrArray(fm.workflows),
    agents: toStrArray(fm.agents),
    priority: normalizePriority(fm.priority),
    allowedTools: toToolList(fm['allowed-tools']),
  }

  return { metadata, body: body.trim(), foreignToolRefs: extractForeignToolRefs(body) }
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
