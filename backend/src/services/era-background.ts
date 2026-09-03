/**
 * 剧集时代背景（era background）服务
 *
 * 概念：整部剧共享一个「时代背景」设定（朝代/世界观/环境/美术方向），
 * 由 AI 从剧本原文提炼后存 dramas.era_background（JSON 文本），
 * 在所有视觉资产生成（角色立绘/三视图/装备/表情/场景/分镜）前自动注入，
 * 保证不同资产、不同镜头生成出的画面时代感一致，不会出现古代戏里现代装饰这类串味。
 *
 * era_background JSON 结构：
 * {
 *   "era":        "时代标签（中文，简短）",
 *   "summary":    "中文概述（世界观/地域/年代/社会风貌）",
 *   "imageHint":  "英文画面指令（文生图用，注入 prompt）"
 * }
 */
import { and, eq, isNull } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { now } from '../utils/response.js'
import { logTaskError, logTaskProgress, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import type { EraBackground } from '../shared/contracts.js'

// 向后兼容：仍可通过本文件 import 该类型，但定义以 shared/contracts.ts 为准
export type { EraBackground }

/** 解析 dramas.era_background（JSON 文本）为结构化对象；空/非法返回 null */
export function parseEraBackground(raw: string | null | undefined): EraBackground | null {
  if (!raw) return null
  let obj: any
  try {
    obj = JSON.parse(raw)
  } catch {
    return null
  }
  if (!obj || typeof obj !== 'object') return null
  const out: EraBackground = {
    era: String(obj.era || obj.summary || '').trim().slice(0, 80),
    summary: String(obj.summary || obj.raw || obj.era || '').trim().slice(0, 2000),
    imageHint: String(obj.image_style_en || obj.imageHint || obj.summary || '').trim().slice(0, 1500),
  }
  if (!out.summary && !out.imageHint) return null
  return out
}

/** 读取某剧的时代背景；未提炼/不存在返回 null */
export function getEraBackground(dramaId: number | null | undefined): EraBackground | null {
  if (!dramaId) return null
  const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
  return drama ? parseEraBackground(drama.eraBackground) : null
}

/**
 * 时代背景注入：imageHint 存在时作为「时代/环境画面指令」追加到 prompt 尾部。
 * 不改变调用方原有 prompt 结构，无背景时原样返回。
 */
export function applyEraImageClause(prompt: string, dramaId?: number | null): string {
  if (!dramaId) return prompt
  const era = getEraBackground(dramaId)
  const hint = era?.imageHint?.trim()
  if (!hint) return prompt
  const clause = hint.endsWith('.') ? hint : `${hint}.`
  return `${prompt}, ${clause}`
}

const ERA_EXTRACT_SYSTEM_PROMPT =
  '你是资深影视美术指导。请从剧本中提炼整部剧的时代背景设定，输出会用于文生图模型保持全剧时代感一致。' +
  '只输出 JSON，不要任何解释或 markdown 代码块标记，格式：' +
  '{"era":"时代标签（中文、简短，如 古代仙侠 / 现代都市·赛博朋克 / 民国谍战 / 中世纪奇幻）",' +
  '"summary":"中文概述 60-120 字：世界观、地域、年代、社会风貌、常见场景等",' +
  '"image_style_en":"英文 2-4 句画面指令：描述生成角色/服装/建筑/道具/环境图时必须体现的时代特征与美术风格方向，可包含材质/配色/年代细节词，不得描述任何具体人物"}' +
  '若剧本包含架空/奇幻/科幻设定，优先交代其特殊规则（如灵力体系、机械义体）。'

/** 去掉 LLM 可能包裹的 ```json ``` 代码块 */
function stripJsonFence(raw: string): string {
  let s = raw.trim()
  const fenceMatch = s.match(/^```[a-zA-Z]*\s*([\s\S]*?)```$/)
  if (fenceMatch) s = fenceMatch[1].trim()
  const firstBrace = s.indexOf('{')
  const lastBrace = s.lastIndexOf('}')
  if (firstBrace >= 0 && lastBrace > firstBrace) s = s.slice(firstBrace, lastBrace + 1)
  return s
}

/**
 * 从剧本原文 AI 提炼时代背景并落库 dramas.era_background。
 * sourceText 未传时自动聚合该剧全部集数的 scriptContent/content。
 */
export async function extractDramaEraBackground(dramaId: number, sourceText?: string): Promise<EraBackground> {
  const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
  if (!drama) throw new Error('Drama not found')

  let text = (sourceText || '').trim()
  if (!text) {
    const eps = db.select().from(schema.episodes)
      .where(and(eq(schema.episodes.dramaId, dramaId), isNull(schema.episodes.deletedAt)))
      .all()
    text = eps
      .map(e => [`第${e.episodeNumber}集${e.title ? ` ${e.title}` : ''}`, e.scriptContent || e.content || ''].filter(Boolean).join('\n'))
      .join('\n\n')
      .trim()
  }
  if (!text) throw new Error('该剧暂无剧本内容，无法提炼时代背景。请先写剧本，或粘贴剧本原文后重试')

  logTaskStart('EraBackground', 'extract', { dramaId, sourceChars: text.length })

  // 剧本输入截断：避免超出模型上下文（约 8000 字符，超出取首尾保设定与结局）
  const MAX_SOURCE_CHARS = 8000
  let truncated = text
  if (truncated.length > MAX_SOURCE_CHARS) {
    truncated = `${text.slice(0, Math.floor(MAX_SOURCE_CHARS * 0.7))}\n……（中间省略）……\n${text.slice(-Math.floor(MAX_SOURCE_CHARS * 0.3))}`
  }

  const userPrompt = `剧名：${drama.title || '(未命名)'}\n以下是剧本内容：\n${truncated}\n\n请提炼这部剧的时代背景设定，只输出 JSON。`

  try {
    const { generateText } = await import('./text-generation.js')
    const result = await generateText(userPrompt, {
      system: ERA_EXTRACT_SYSTEM_PROMPT,
      temperature: 0.2,
      maxTokens: 800,
    })
    logTaskProgress('EraBackground', 'ai-result', { dramaId, length: result.length })
    const obj = JSON.parse(stripJsonFence(result)) as any
    const parsed = parseEraBackground(JSON.stringify(obj))
    if (!parsed) throw new Error('AI 输出缺少有效时代背景字段')

    db.update(schema.dramas)
      .set({ eraBackground: JSON.stringify(parsed), updatedAt: now() })
      .where(eq(schema.dramas.id, dramaId)).run()
    logTaskSuccess('EraBackground', 'extract', { dramaId, era: parsed.era })
    return parsed
  } catch (err: any) {
    logTaskError('EraBackground', 'extract', { dramaId, error: err.message })
    throw new Error(`时代背景提炼失败: ${err.message}`)
  }
}
