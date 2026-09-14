/**
 * 角色音色分配 Agent 工具
 */
import { createTool } from '@mastra/core/tools'
import { z } from 'zod'
import { db, schema } from '../../db/index.js'
import { and, eq, isNull } from 'drizzle-orm'
import { now } from '../../utils/response.js'
import { logTaskProgress, logTaskSuccess, logTaskWarn } from '../../utils/task-logger.js'

export function createVoiceTools(episodeId: number, dramaId: number) {
  function getEpisodeAudioProvider() {
    const [episode] = db.select().from(schema.episodes).where(eq(schema.episodes.id, episodeId)).all()
    if (!episode?.audioConfigId) return null
    const [config] = db.select().from(schema.aiServiceConfigs).where(eq(schema.aiServiceConfigs.id, episode.audioConfigId)).all()
    return config?.provider || null
  }

  const getCharacters = createTool({
    id: 'get_characters',
    description: 'Get all characters for the current drama with their current voice assignments.',
    inputSchema: z.object({}),
    execute: async () => {
      const chars = db.select().from(schema.characters)
        .where(and(eq(schema.characters.dramaId, dramaId), isNull(schema.characters.deletedAt))).all()
      const payload = {
        characters: chars.map(c => ({
          id: c.id,
          name: c.name,
          role: c.role,
          personality: c.personality,
          description: c.description,
          current_voice: c.voiceStyle || '未分配',
          speaker_id: c.speakerId || '',
        })),
      }
      logTaskSuccess('VoiceTool', 'get-characters', { episodeId, dramaId, count: payload.characters.length })
      return payload
    },
  })

  const listVoices = createTool({
    id: 'list_voices',
    description: 'List all available voice options for TTS, optionally filtered by role type.',
    inputSchema: z.object({
      role_tag: z.string().optional().describe('按角色类型筛选音色：旁白 / 主角 / 反派 / 配角'),
    }),
    execute: async ({ role_tag }) => {
      const provider = getEpisodeAudioProvider() || 'minimax'
      const rows = db.select().from(schema.aiVoices).where(eq(schema.aiVoices.provider, provider)).all()
      const voices = rows.length ? rows.map(v => {
        const desc = v.description ? JSON.parse(v.description) : []
        return {
          id: v.voiceId,
          name: v.voiceName,
          gender: inferGender(v.voiceName, desc),
          traits: Array.isArray(desc) && desc.length ? desc.slice(0, 2).join('、') : `${v.language || '多语言'}音色`,
          suitable_for: Array.isArray(desc) && desc.length > 2 ? desc.slice(2).join('、') : `${v.language || '通用'}角色`,
          role_tags: parseRoleTags(v.roleTags),
          language: v.language,
          provider,
        }
      }) : [
        { id: 'alloy', name: 'Alloy', gender: '中性', traits: '平衡自然', suitable_for: '旁白、通用', role_tags: ['旁白', '配角'], language: '多语言', provider },
        { id: 'echo', name: 'Echo', gender: '男声', traits: '低沉稳重', suitable_for: '成熟男性、旁白', role_tags: ['旁白', '主角'], language: '多语言', provider },
        { id: 'fable', name: 'Fable', gender: '男声', traits: '温暖富有表现力', suitable_for: '年轻男性、故事叙述', role_tags: ['主角', '旁白'], language: '多语言', provider },
        { id: 'onyx', name: 'Onyx', gender: '男声', traits: '深沉有力', suitable_for: '权威角色、反派', role_tags: ['反派', '配角'], language: '多语言', provider },
        { id: 'nova', name: 'Nova', gender: '女声', traits: '温柔甜美', suitable_for: '年轻女性、女主', role_tags: ['主角'], language: '多语言', provider },
        { id: 'shimmer', name: 'Shimmer', gender: '女声', traits: '明亮活泼', suitable_for: '活泼女性、少女', role_tags: ['主角', '配角'], language: '多语言', provider },
      ].filter(v => !role_tag || v.role_tags.includes(role_tag))

      const payload = {
        provider,
        voices,
        instruction: '根据角色的性别、性格、年龄来匹配最合适的音色，并且只能从当前集音频配置可用的音色列表中选择。',
      }
      logTaskSuccess('VoiceTool', 'list-voices', { episodeId, provider, count: payload.voices.length })
      return payload
    },
  })

  const assignVoice = createTool({
    id: 'assign_voice',
    description: 'Assign a voice to a character.',
    inputSchema: z.object({
      character_id: z.number().describe('Character ID'),
      voice_id: z.string().describe('Voice ID from list_voices'),
      speaker_id: z.string().optional().describe('Global speaker ID (S1, S2, ...) unique per character and stable across episodes'),
      reason: z.string().optional().describe('Why this voice fits'),
    }),
    execute: async ({ character_id, voice_id, speaker_id, reason }) => {
      const provider = getEpisodeAudioProvider() || 'minimax'
      logTaskProgress('VoiceTool', 'assign-begin', { episodeId, dramaId, characterId: character_id, voiceId: voice_id, speakerId: speaker_id, provider, reason })

      // 硬规则 1：一角色一音色，禁止两个角色共用同一音色（跨集锁定由 voiceStyle 存于角色级天然保证）
      const voiceConflict = db.select().from(schema.characters)
        .where(and(eq(schema.characters.dramaId, dramaId), eq(schema.characters.voiceStyle, voice_id), isNull(schema.characters.deletedAt)))
        .all()
        .find(c => c.id !== character_id)
      if (voiceConflict) {
        logTaskWarn('VoiceTool', 'assign-voice-conflict', { episodeId, characterId: character_id, voiceId: voice_id, conflictName: voiceConflict.name })
        return { error: `音色 "${voice_id}" 已分配给角色「${voiceConflict.name}」(id=${voiceConflict.id})，禁止共用，请改选其他音色` }
      }

      // 硬规则 2：speaker_id 全局唯一（一个说话人 = 一个角色，跨集稳定）
      if (speaker_id) {
        const speakerConflict = db.select().from(schema.characters)
          .where(and(eq(schema.characters.dramaId, dramaId), eq(schema.characters.speakerId, speaker_id), isNull(schema.characters.deletedAt)))
          .all()
          .find(c => c.id !== character_id)
        if (speakerConflict) {
          logTaskWarn('VoiceTool', 'assign-speaker-conflict', { episodeId, characterId: character_id, speakerId: speaker_id, conflictName: speakerConflict.name })
          return { error: `speaker_id "${speaker_id}" 已分配给角色「${speakerConflict.name}」(id=${speakerConflict.id})，请勿复用` }
        }
      }

      const update: Record<string, any> = { voiceStyle: voice_id, voiceProvider: provider, voiceSampleUrl: null, updatedAt: now() }
      if (speaker_id) update.speakerId = speaker_id
      db.update(schema.characters)
        .set(update)
        .where(eq(schema.characters.id, character_id))
        .run()
      logTaskSuccess('VoiceTool', 'assign-complete', { episodeId, characterId: character_id, voiceId: voice_id, speakerId: speaker_id, provider })
      return { message: `Assigned voice "${voice_id}" (speaker "${speaker_id || 'n/a'}") to character ${character_id}`, reason }
    },
  })

  return { getCharacters, listVoices, assignVoice }
}

function inferGender(name: string, desc: unknown) {
  const description = Array.isArray(desc) ? desc.join(' ') : ''
  const text = `${name} ${description}`
  if (/[男|青年|大爷|学长|boy|man|male]/i.test(text)) return '男声'
  if (/[女|少女|御姐|奶奶|girl|woman|female]/i.test(text)) return '女声'
  return '中性'
}

function parseRoleTags(raw: string | null): string[] {
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.map(String) : []
  } catch {
    return []
  }
}
