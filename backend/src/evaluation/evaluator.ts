/**
 * 评测执行器：seed 临时剧组 → 用指定提示词跑 Agent → 捕获工具调用 → 确定性评分 → 清理。
 *
 * 隔离性：每次评测都用独立的临时 drama/episode，跑完物理删除，不污染真实数据。
 * 可重复：同一 case + 同一提示词，评分应稳定（LLM 本身有随机性，属可接受方差）。
 */
import { eq, inArray } from 'drizzle-orm'
import { db } from '../db/index.js'
import * as schema from '../db/schema.js'
import { now } from '../utils/response.js'
import { runAgentWithInstructions } from '../agents/index.js'
import { scoreStoryboards, scoreExtraction, scoreScriptRewrite, scoreVoiceAssignment } from './scorer.js'
import type {
  BenchmarkCase,
  ScoreReport,
  StoryboardShot,
  ExtractedCharacter,
  ExtractedScene,
} from './types.js'

const STORYBOARD_MESSAGE = '请对当前集剧本进行分镜拆解并保存。'
const EXTRACTOR_MESSAGE = '请提取当前集剧本中出现的角色和场景并保存。'
const SCRIPT_REWRITER_MESSAGE = '请将当前集的原始内容改写为格式化剧本并保存。'
const VOICE_ASSIGNER_MESSAGE = '请为当前剧组的角色分配合适的音色。'

interface SeedResult {
  dramaId: number
  episodeId: number
  legal: { characterIds: Set<number>; sceneIds: Set<number> }
}

function seedCase(opts: {
  caseId: string
  script?: string
  content?: string
  characters?: Array<{ name: string; role: string; appearance?: string; personality?: string }>
  scenes?: Array<{ location: string; time: string; prompt: string }>
}): SeedResult {
  const ts = now()

  const drama = db.insert(schema.dramas)
    .values({ title: `[bench] ${opts.caseId}`, status: 'draft', createdAt: ts, updatedAt: ts })
    .run()
  const dramaId = Number(drama.lastInsertRowid)

  const ep = db.insert(schema.episodes)
    .values({ dramaId, episodeNumber: 1, title: '评测集', content: opts.content, scriptContent: opts.script, createdAt: ts, updatedAt: ts })
    .run()
  const episodeId = Number(ep.lastInsertRowid)

  const characterIds = new Set<number>()
  for (const c of opts.characters || []) {
    const r = db.insert(schema.characters)
      .values({ dramaId, name: c.name, role: c.role, appearance: c.appearance, personality: c.personality, createdAt: ts, updatedAt: ts })
      .run()
    const cid = Number(r.lastInsertRowid)
    characterIds.add(cid)
    db.insert(schema.episodeCharacters).values({ episodeId, characterId: cid, createdAt: ts }).run()
  }

  const sceneIds = new Set<number>()
  for (const s of opts.scenes || []) {
    const r = db.insert(schema.scenes)
      .values({ dramaId, episodeId, location: s.location, time: s.time, prompt: s.prompt, createdAt: ts, updatedAt: ts })
      .run()
    const sid = Number(r.lastInsertRowid)
    sceneIds.add(sid)
    db.insert(schema.episodeScenes).values({ episodeId, sceneId: sid, createdAt: ts }).run()
  }

  return { dramaId, episodeId, legal: { characterIds, sceneIds } }
}

/** 物理删除评测临时数据（seed 的 drama 及其全部关联，含 Agent 跑出来的分镜/角色/场景） */
function cleanupCase(seed: SeedResult) {
  const { dramaId, episodeId } = seed
  try {
    const sbs = db.select({ id: schema.storyboards.id })
      .from(schema.storyboards).where(eq(schema.storyboards.episodeId, episodeId)).all()
    if (sbs.length) {
      db.delete(schema.storyboardCharacters)
        .where(inArray(schema.storyboardCharacters.storyboardId, sbs.map(s => s.id))).run()
    }
    db.delete(schema.storyboards).where(eq(schema.storyboards.episodeId, episodeId)).run()
    db.delete(schema.episodeScenes).where(eq(schema.episodeScenes.episodeId, episodeId)).run()
    db.delete(schema.episodeCharacters).where(eq(schema.episodeCharacters.episodeId, episodeId)).run()
    db.delete(schema.scenes).where(eq(schema.scenes.dramaId, dramaId)).run()
    db.delete(schema.characters).where(eq(schema.characters.dramaId, dramaId)).run()
    db.delete(schema.episodes).where(eq(schema.episodes.id, episodeId)).run()
    db.delete(schema.dramas).where(eq(schema.dramas.id, dramaId)).run()
  } catch (e) {
    console.warn(`[bench] cleanup failed: ${(e as Error).message}`)
  }
}

/**
 * 归一化工具名：忽略大小写与非字母数字字符。
 * Mastra 的 payload.toolName 可能是驼峰（saveStoryboards），而工具 id 是下划线（save_storyboards），
 * 归一化后二者一致，避免因命名风格差异导致提取 0 命中。
 */
function normToolName(name: string): string {
  return (name || '').toLowerCase().replace(/[^a-z0-9]/g, '')
}

function extractStoryboards(toolCalls: Array<{ toolName: string; args: any }>): StoryboardShot[] {
  const calls = toolCalls.filter(tc => normToolName(tc.toolName) === 'savestoryboards')
  const last = calls[calls.length - 1]
  return (last?.args?.storyboards as StoryboardShot[]) || []
}

function extractCharacters(toolCalls: Array<{ toolName: string; args: any }>): ExtractedCharacter[] {
  const all: ExtractedCharacter[] = []
  for (const tc of toolCalls) {
    if (normToolName(tc.toolName) === 'savededupcharacters') all.push(...(tc.args?.characters || []))
  }
  return all
}

function extractScenes(toolCalls: Array<{ toolName: string; args: any }>): ExtractedScene[] {
  const all: ExtractedScene[] = []
  for (const tc of toolCalls) {
    if (normToolName(tc.toolName) === 'savededupscenes') all.push(...(tc.args?.scenes || []))
  }
  return all
}

/** script_rewriter 的产出：save_script 提交的格式化剧本（取最后一次） */
function extractScript(toolCalls: Array<{ toolName: string; args: any }>): string {
  const calls = toolCalls.filter(tc => normToolName(tc.toolName) === 'savescript')
  const last = calls[calls.length - 1]
  return (last?.args?.content as string) || ''
}

/** voice_assigner 的产出：assign_voice 提交的音色分配 */
function extractVoiceAssignments(
  toolCalls: Array<{ toolName: string; args: any }>,
): Array<{ character_id?: number; voice_id?: string; reason?: string }> {
  const all: Array<{ character_id?: number; voice_id?: string; reason?: string }> = []
  for (const tc of toolCalls) {
    if (normToolName(tc.toolName) === 'assignvoice') {
      all.push({ character_id: tc.args?.character_id, voice_id: tc.args?.voice_id, reason: tc.args?.reason })
    }
  }
  return all
}

/**
 * 用指定 instructions 评测一个 case，返回确定性评分报告。
 * 不依赖、也不修改 DB 里的 agent_configs（候选提示词直接传入）。
 */
export async function evaluateCase(caseDef: BenchmarkCase, instructions: string): Promise<ScoreReport> {
  switch (caseDef.kind) {
    case 'storyboard': {
      const seed = seedCase({
        caseId: caseDef.id,
        script: caseDef.statement.script,
        characters: caseDef.statement.characters,
        scenes: caseDef.statement.scenes,
      })
      try {
        const result = await runAgentWithInstructions(
          'storyboard_breaker', seed.episodeId, seed.dramaId, instructions, STORYBOARD_MESSAGE,
        )
        const shots = extractStoryboards(result.toolCalls)
        const report = scoreStoryboards(shots, caseDef.rubric, seed.legal)
        report.caseId = caseDef.id
        report.runtimeModel = result.model
        return report
      } finally {
        cleanupCase(seed)
      }
    }

    case 'extractor': {
      const seed = seedCase({ caseId: caseDef.id, script: caseDef.statement.script })
      try {
        const result = await runAgentWithInstructions(
          'extractor', seed.episodeId, seed.dramaId, instructions, EXTRACTOR_MESSAGE,
        )
        const chars = extractCharacters(result.toolCalls)
        const scenes = extractScenes(result.toolCalls)
        const report = scoreExtraction(chars, scenes, caseDef.rubric)
        report.caseId = caseDef.id
        report.runtimeModel = result.model
        return report
      } finally {
        cleanupCase(seed)
      }
    }

    case 'script_rewriter': {
      const seed = seedCase({ caseId: caseDef.id, content: caseDef.statement.content })
      try {
        const result = await runAgentWithInstructions(
          'script_rewriter', seed.episodeId, seed.dramaId, instructions, SCRIPT_REWRITER_MESSAGE,
        )
        const content = extractScript(result.toolCalls)
        const report = scoreScriptRewrite(content, caseDef.rubric)
        report.caseId = caseDef.id
        report.runtimeModel = result.model
        return report
      } finally {
        cleanupCase(seed)
      }
    }

    case 'voice_assigner': {
      const seed = seedCase({
        caseId: caseDef.id,
        script: caseDef.statement.script,
        characters: caseDef.statement.characters.map(c => ({
          name: c.name, role: c.role, personality: c.personality,
        })),
      })
      try {
        const result = await runAgentWithInstructions(
          'voice_assigner', seed.episodeId, seed.dramaId, instructions, VOICE_ASSIGNER_MESSAGE,
        )
        const assignments = extractVoiceAssignments(result.toolCalls)
        const report = scoreVoiceAssignment(assignments, caseDef.rubric, seed.legal)
        report.caseId = caseDef.id
        report.runtimeModel = result.model
        return report
      } finally {
        cleanupCase(seed)
      }
    }
  }
}
