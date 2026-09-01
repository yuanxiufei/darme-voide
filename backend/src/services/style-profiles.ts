/**
 * 风格 Profile 提炼（对齐参考项目 H3-Codex-Drama Profile Distiller）
 *
 * 从参考视频/素材提炼可复用的 house style，分四类规则：
 *  - storytelling：叙事节奏 / 悬念密度 / 转场动机偏好
 *  - shot_patterns：景别分布 / 机位 / 运镜偏好
 *  - audio_captions：音效 / 配乐 / 字幕风格
 *  - qc_rules：验收规则覆盖（阈值等）
 *
 * 来源三分类（对齐 H3-Codex-Drama 的 provenance 三分类）：
 *  - measurement facts：可测量的客观事实（如 ffprobe 探测的时长/分辨率/fps/响度）
 *  - visual inference：模型从素材视觉推断的结论（需人审复核）
 *  - user preference：用户明确表达的风格偏好（最高优先级）
 *
 * Profile 激活后注入 storyboard_breaker 等生成 Agent 指令（见 agents/index.ts），
 * 跨集保持统一风格（对齐 H3-Codex-Drama 跨集 locked 机制）。
 */
import { eq, and, isNull } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { now } from '../utils/response.js'
import { logTaskError, logTaskWarn } from '../utils/task-logger.js'
import { getTextConfig, getTextProviderBaseUrl } from './ai.js'
import { llmFetch } from '../utils/llm-fetch.js'
import { createOpenAI } from '@ai-sdk/openai'
import { Agent } from '@mastra/core/agent'
import ffmpeg from 'fluent-ffmpeg'
import { getAbsolutePath } from '../utils/storage.js'

// ===== 类型 =====

export interface StyleProfileRow {
  id: number
  dramaId: number | null
  name: string
  description: string | null
  source: string | null
  storytelling: string | null
  shotPatterns: string | null
  audioCaptions: string | null
  qcRules: string | null
  facts: string | null
  inferences: string | null
  preferences: string | null
  isActive: boolean
  createdAt: string
  updatedAt: string
  deletedAt: string | null
}

/** 四类规则 + 三分类来源（JSON 结构约定） */
export interface DistillResult {
  storytelling: Record<string, unknown>
  shot_patterns: Record<string, unknown>
  audio_captions: Record<string, unknown>
  qc_rules: Record<string, unknown>
  facts: string[]
  inferences: string[]
  preferences: string[]
}

function mapRow(r: any): StyleProfileRow {
  return {
    id: r.id,
    dramaId: r.dramaId,
    name: r.name,
    description: r.description,
    source: r.source,
    storytelling: r.storytelling,
    shotPatterns: r.shotPatterns,
    audioCaptions: r.audioCaptions,
    qcRules: r.qcRules,
    facts: r.facts,
    inferences: r.inferences,
    preferences: r.preferences,
    isActive: !!r.isActive,
    createdAt: r.createdAt,
    updatedAt: r.updatedAt,
    deletedAt: r.deletedAt,
  }
}

// ===== CRUD =====

export function listStyleProfiles(dramaId?: number | null): StyleProfileRow[] {
  const base = db.select().from(schema.styleProfiles)
    .where(and(isNull(schema.styleProfiles.deletedAt), dramaId ? eq(schema.styleProfiles.dramaId, dramaId) : undefined))
  return (base.all() as any[]).map(mapRow).sort((a, b) => (a.isActive ? -1 : 1) - (b.isActive ? -1 : 1))
}

export function getStyleProfile(id: number): StyleProfileRow | null {
  const [row] = db.select().from(schema.styleProfiles)
    .where(and(eq(schema.styleProfiles.id, id), isNull(schema.styleProfiles.deletedAt))).all()
  return row ? mapRow(row) : null
}

export function createStyleProfile(input: {
  dramaId?: number | null
  name: string
  description?: string | null
  source?: string | null
  storytelling?: Record<string, unknown> | null
  shotPatterns?: Record<string, unknown> | null
  audioCaptions?: Record<string, unknown> | null
  qcRules?: Record<string, unknown> | null
  preferences?: string[] | null
}): number | null {
  const ts = now()
  try {
    const res = db.insert(schema.styleProfiles).values({
      dramaId: input.dramaId ?? null,
      name: input.name,
      description: input.description ?? null,
      source: input.source ?? null,
      storytelling: input.storytelling ? JSON.stringify(input.storytelling) : null,
      shotPatterns: input.shotPatterns ? JSON.stringify(input.shotPatterns) : null,
      audioCaptions: input.audioCaptions ? JSON.stringify(input.audioCaptions) : null,
      qcRules: input.qcRules ? JSON.stringify(input.qcRules) : null,
      preferences: input.preferences ? JSON.stringify(input.preferences) : null,
      isActive: false,
      createdAt: ts,
      updatedAt: ts,
    }).run()
    return Number(res.lastInsertRowid)
  } catch (err: any) {
    logTaskWarn('StyleProfile', 'create-failed', { error: err?.message || String(err) })
    return null
  }
}

export function updateStyleProfile(id: number, input: Partial<{
  name: string
  description: string | null
  source: string | null
  storytelling: Record<string, unknown> | null
  shotPatterns: Record<string, unknown> | null
  audioCaptions: Record<string, unknown> | null
  qcRules: Record<string, unknown> | null
  preferences: string[] | null
}>): boolean {
  const existing = getStyleProfile(id)
  if (!existing) return false

  const patch: Record<string, any> = { updatedAt: now() }
  const jsonFields: Array<[string, string | null | undefined]> = [
    ['storytelling', input.storytelling === undefined ? undefined : (input.storytelling ? JSON.stringify(input.storytelling) : null)],
    ['shotPatterns', input.shotPatterns === undefined ? undefined : (input.shotPatterns ? JSON.stringify(input.shotPatterns) : null)],
    ['audioCaptions', input.audioCaptions === undefined ? undefined : (input.audioCaptions ? JSON.stringify(input.audioCaptions) : null)],
    ['qcRules', input.qcRules === undefined ? undefined : (input.qcRules ? JSON.stringify(input.qcRules) : null)],
    ['preferences', input.preferences === undefined ? undefined : (input.preferences ? JSON.stringify(input.preferences) : null)],
  ]
  for (const [key, value] of jsonFields) {
    if (value !== undefined) patch[key] = value
  }
  if (input.name !== undefined) patch.name = input.name
  if (input.description !== undefined) patch.description = input.description
  if (input.source !== undefined) patch.source = input.source

  db.update(schema.styleProfiles).set(patch).where(eq(schema.styleProfiles.id, id)).run()
  return true
}

export function deleteStyleProfile(id: number): boolean {
  const existing = getStyleProfile(id)
  if (!existing) return false
  db.update(schema.styleProfiles).set({ deletedAt: now(), isActive: false })
    .where(eq(schema.styleProfiles.id, id)).run()
  return true
}

/** 激活某个 Profile（同 drama 内仅一个激活），返回激活后 Profile */
export function activateStyleProfile(id: number): StyleProfileRow | null {
  const p = getStyleProfile(id)
  if (!p) return null

  if (p.dramaId) {
    db.update(schema.styleProfiles).set({ isActive: false })
      .where(and(eq(schema.styleProfiles.dramaId, p.dramaId), eq(schema.styleProfiles.isActive, true))).run()
  }
  db.update(schema.styleProfiles).set({ isActive: true, updatedAt: now() })
    .where(eq(schema.styleProfiles.id, id)).run()
  return getStyleProfile(id)
}

/** 取某剧当前激活的 Profile（未绑定剧的全局 Profile 也视为候选） */
export function getActiveProfileForDrama(dramaId?: number | null): StyleProfileRow | null {
  if (dramaId) {
    const [dramaScoped] = db.select().from(schema.styleProfiles)
      .where(and(eq(schema.styleProfiles.dramaId, dramaId), eq(schema.styleProfiles.isActive, true), isNull(schema.styleProfiles.deletedAt)))
      .all()
    if (dramaScoped) return mapRow(dramaScoped)
  }
  const [global] = db.select().from(schema.styleProfiles)
    .where(and(isNull(schema.styleProfiles.dramaId), eq(schema.styleProfiles.isActive, true), isNull(schema.styleProfiles.deletedAt)))
    .all()
  return global ? mapRow(global) : null
}

// ===== 提炼（对齐 H3-Codex-Drama distill_house_style 的 LLM 分析） =====

/** 用 ffprobe 探测参考视频的测量事实（可选，失败静默） */
async function probeMeasurementFacts(source: string | null): Promise<Record<string, unknown>> {
  if (!source) return {}
  const facts: Record<string, unknown> = {}
  try {
    const abs = getAbsolutePath(source)
    const meta: any = await new Promise((resolve, reject) => {
      ffmpeg.ffprobe(abs, (err, m) => (err ? reject(err) : resolve(m)))
    })
    const v = meta.streams?.find((s: any) => s.codec_type === 'video')
    const a = meta.streams?.find((s: any) => s.codec_type === 'audio')
    if (v) {
      facts.resolution = `${v.width}x${v.height}`
      facts.fps = v.r_frame_rate || v.avg_frame_rate || null
      facts.video_codec = v.codec_name || null
    }
    if (a) facts.audio_codec = a.codec_name || null
    if (meta.format?.duration) facts.duration_seconds = Math.round(meta.format.duration)
    if (meta.format?.size) facts.file_size_bytes = meta.format.size
  } catch { /* 非本地文件/探测失败时忽略 */ }
  return facts
}

/**
 * 用文本 LLM 分析参考素材，提炼 house style（四类规则 + 三分类来源）。
 * 不修改数据，返回提炼结果供用户确认后写入（对齐 H3-Codex-Drama 的 user confirmation gate）。
 */
export async function distillStyleProfile(id: number): Promise<{ ok: boolean; error?: string; result?: DistillResult }> {
  const p = getStyleProfile(id)
  if (!p) return { ok: false, error: 'Profile not found' }

  try {
    const measurements = await probeMeasurementFacts(p.source)

    const textConfig = getTextConfig()
    const resolvedBaseURL = getTextProviderBaseUrl(textConfig)
    const model = textConfig.models[0] || textConfig.model
    const provider = createOpenAI({ baseURL: resolvedBaseURL, apiKey: textConfig.apiKey, fetch: llmFetch } as any)

    const agent = new Agent({
      id: 'style_profile_distiller',
      name: '风格提炼',
      instructions: `你是资深影视风格分析专家。根据用户提供的参考素材描述，提炼可复用的「house style」短片风格档案。

必须严格输出 JSON（不要包含任何其他文字、markdown 代码块或注释），结构如下：
{
  "storytelling": { 叙事节奏、悬念密度、转场动机偏好、信息揭示节奏 },
  "shot_patterns": { 景别分布偏好、机位、运镜、构图习惯、镜头时长规律 },
  "audio_captions": { 配乐风格、音效密度、字幕风格、静场偏好 },
  "qc_rules": { 验收时的画面/音频/连续性硬性标准，如「同场景相邻镜头相似度须≥0.55」「响度 I=-14 LUFS」 },
  "facts": ["可测量的客观事实，来自参考素材本身，如分辨率/时长/镜头数等"],
  "inferences": ["你从素材风格推断出的结论（可能是主观判断，需人工复核）"],
  "preferences": ["用户明确表达的风格偏好（最高优先级）"]
}

原则：
- facts 只放可验证的客观测量；inferences 是推断；preferences 来自用户原话的偏好
- qc_rules 要具体到可执行数值/判定标准
- 中文输出，JSON 字段名保持英文`,
      model: provider.chat(model),
    })

    const userMsg = [
      `参考素材说明：${p.source || '（未提供，仅凭 Profile 名称/描述）'}`,
      `Profile 名称：${p.name}`,
      p.description ? `描述：${p.description}` : '',
      Object.keys(measurements).length
        ? `参考素材测量事实（ffprobe 探测，可信）：${JSON.stringify(measurements, null, 2)}`
        : '',
      `已知用户偏好（若有）：${p.preferences || '（无）'}`,
    ].filter(Boolean).join('\n\n')

    const result = await agent.generate([{ role: 'user', content: userMsg }], { maxSteps: 1 })
    const text = result.text || ''
    const cleaned = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim()

    const start = cleaned.indexOf('{')
    const end = cleaned.lastIndexOf('}')
    if (start < 0 || end < 0) return { ok: false, error: 'LLM output is not JSON' }

    const parsed = JSON.parse(cleaned.slice(start, end + 1)) as Partial<DistillResult>
    const resultObj: DistillResult = {
      storytelling: (parsed.storytelling || {}) as Record<string, unknown>,
      shot_patterns: (parsed.shot_patterns || {}) as Record<string, unknown>,
      audio_captions: (parsed.audio_captions || {}) as Record<string, unknown>,
      qc_rules: (parsed.qc_rules || {}) as Record<string, unknown>,
      facts: Array.isArray(parsed.facts) ? parsed.facts : [],
      inferences: Array.isArray(parsed.inferences) ? parsed.inferences : [],
      preferences: Array.isArray(parsed.preferences) ? parsed.preferences : [],
    }
    return { ok: true, result: resultObj }
  } catch (err: any) {
    logTaskError('StyleProfile', 'distill-failed', { profileId: id, error: err?.message || String(err) })
    return { ok: false, error: err?.message || String(err) }
  }
}

/** 应用确认后的提炼结果写入 Profile（含人工复核后的偏好） */
export function applyDistillResult(id: number, result: DistillResult): boolean {
  return updateStyleProfile(id, {
    storytelling: result.storytelling,
    shotPatterns: result.shot_patterns,
    audioCaptions: result.audio_captions,
    qcRules: result.qc_rules,
    preferences: result.preferences,
  })
}
