import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, created, now, badRequest } from '../utils/response.js'
import { toSnakeCase } from '../utils/transform.js'
import { generateTTS } from '../services/tts-generation.js'
import { logTaskError, logTaskPayload, logTaskProgress, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import ffmpeg from 'fluent-ffmpeg'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { v4 as uuid } from 'uuid'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const STORAGE_ROOT = process.env.STORAGE_PATH || path.resolve(__dirname, '../../../data/static')

const app = new Hono()

const IGNORE_TTS_SPEAKERS = /^(环境音|环境声|音效|效果音|sfx|sound ?effect|bgm|背景音|背景音乐|ambient)$/i
const IGNORE_TTS_TEXT = /^(无|无对白|无台词|无旁白|无需配音|无需对白|none|null|n\/a|na|环境音|环境声|音效|效果音|纯音效|纯环境音|只有环境音|仅环境音|背景音|背景音乐|bgm|sfx|ambient)$/i

/**
 * 将一段包含多角色对话的文本拆分为多条独立对话行
 * 只匹配真正的"角色名：较长台词"模式，避免把句子中的标点/短句误拆
 * 例："三娘：你好呀今天天气不错。姐姐：小雪别闹了。小雪：我知道了姐姐。"
 * → [{speaker:"三娘", text:"你好呀今天天气不错"}, {speaker:"姐姐", text:"小雪别闹了"}, {speaker:"小雪", text:"我知道了姐姐"}]
 *
 * 关键规则：冒号后面必须 >= 8 个字符且不含冒号，才认为是有效对话行
 */
function parseDialogueLines(dialogue?: string | null): Array<{ speaker: string; text: string }> {
  const raw = dialogue?.trim() || ''
  if (!raw) return []

  const lines: Array<{ speaker: string; text: string }> = []
  // 全局匹配：角色名(不含冒号) + 冒号 + 至少8个字符的台词(不含冒号)
  const regex = /([^\n:：]{1,10}?)[:：]([^:：\n]{8,})/g
  let match: RegExpExecArray | null

  while ((match = regex.exec(raw)) !== null) {
    let speaker = match[1].replace(/[（(].+?[)）]/g, '').trim()
    const text = match[2].trim()
    // 过滤：角色名太长不像名字、或属于忽略列表
    if (!speaker || speaker.length > 10 || IGNORE_TTS_SPEAKERS.test(speaker)) continue
    if (IGNORE_TTS_TEXT.test(text)) continue
    lines.push({ speaker, text })
  }

  return lines
}

function parseDialogueForTTS(dialogue?: string | null) {
  const raw = dialogue?.trim() || ''
  if (!raw) return { speaker: '', pureText: '', ignorable: true }
  const speakerMatch = raw.match(/^(.+?)[:：]/)
  const speaker = speakerMatch ? speakerMatch[1].replace(/[（(].+?[)）]/g, '').trim() : ''
  const pureText = raw.replace(/^.+?[:：]\s*/, '').replace(/[（(].+?[)）]/g, '').trim()
  const ignorable = (!!speaker && IGNORE_TTS_SPEAKERS.test(speaker)) || !pureText || IGNORE_TTS_TEXT.test(pureText)
  return { speaker, pureText, ignorable }
}

/**
 * TTS 匹配状态
 * - matched:   角色存在于剧组且已配置音色，音色与角色形象一致
 * - no_voice:  角色存在但未配置音色，将 fallback 到默认音色 'alloy'
 * - not_found: 台词中出现的角色名在剧组角色列表中不存在
 * - narrator:  旁白/画外音（无需角色匹配）
 */
type TTSMatchStatus = 'matched' | 'no_voice' | 'not_found' | 'narrator'

interface TTSLineValidation {
  speaker: string
  text: string
  match_status: TTSMatchStatus
  voice_id: string
  character_id?: number | null
  warning?: string
}

/**
 * 验证单个台词行：角色名→剧组角色记录→音色配置 三连匹配
 */
function validateTTSSpeaker(speaker: string, dramaId: number): TTSLineValidation['match_status'] & { voiceId: string; characterId?: number | null } {
  // 旁白/画外音直接视为通过
  if (/^(旁白|画外音|narrator)$/i.test(speaker)) {
    return { match_status: 'narrator' as TTSLineValidation['match_status'], voiceId: 'alloy' }
  }

  const chars = db.select().from(schema.characters)
    .where(eq(schema.characters.dramaId, dramaId)).all()
  const found = chars.find((char) => char.name === speaker)

  if (!found) {
    return { match_status: 'not_found', voiceId: 'alloy', characterId: null }
  }

  if (!found.voiceStyle) {
    return { match_status: 'no_voice', voiceId: 'alloy', characterId: found.id }
  }

  return { match_status: 'matched', voiceId: found.voiceStyle, characterId: found.id }
}

/**
 * 获取角色的 voiceStyle，用于 TTS 音色选择
 * @deprecated 请使用 validateTTSSpeaker 获取完整验证信息
 */
function getCharacterVoiceId(characterName: string, dramaId: number): string {
  const result = validateTTSSpeaker(characterName, dramaId)
  return result.voiceId
}

/**
 * 验证整个 dialogue 字段中所有角色台词行的匹配状态
 */
function validateDialogueLines(speakerLines: Array<{ speaker: string; text: string }>, dramaId: number): TTSLineValidation[] {
  return speakerLines.map(line => {
    const validation = validateTTSSpeaker(line.speaker, dramaId)
    let warning: string | undefined
    if (validation.match_status === 'not_found') {
      warning = `角色"${line.speaker}"在剧组角色列表中不存在，将使用默认音色`
    } else if (validation.match_status === 'no_voice') {
      warning = `角色"${line.speaker}"尚未配置音色，将使用默认音色`
    }
    return {
      speaker: line.speaker,
      text: line.text,
      match_status: validation.match_status,
      voice_id: validation.voiceId,
      character_id: validation.characterId ?? null,
      ...(warning ? { warning } : {}),
    }
  })
}

function syncStoryboardCharacters(storyboardId: number, characterIds: number[]) {
  db.delete(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .run()

  const uniqueIds = [...new Set((characterIds || []).filter(Boolean))]
  if (!uniqueIds.length) return

  for (const characterId of uniqueIds) {
    db.insert(schema.storyboardCharacters).values({
      storyboardId,
      characterId,
    }).run()
  }
}

function getStoryboardCharacterIds(storyboardId: number) {
  return db.select().from(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId)).all()
    .map(link => link.characterId)
}

function validateStoryboardBindings(episodeId: number, sceneId: number | null | undefined, characterIds: number[] | undefined) {
  const episodeSceneIds = new Set(
    db.select().from(schema.episodeScenes)
      .where(eq(schema.episodeScenes.episodeId, episodeId)).all()
      .map(link => link.sceneId),
  )
  const episodeCharacterIds = new Set(
    db.select().from(schema.episodeCharacters)
      .where(eq(schema.episodeCharacters.episodeId, episodeId)).all()
      .map(link => link.characterId),
  )

  if (sceneId != null && !episodeSceneIds.has(sceneId)) {
    throw new Error('scene_id 必须来自当前集已关联场景')
  }

  const invalidCharacterIds = (characterIds || []).filter(id => !episodeCharacterIds.has(id))
  if (invalidCharacterIds.length) {
    throw new Error('character_ids 必须来自当前集已关联角色')
  }
}

// POST /storyboards
app.post('/', async (c) => {
  const body = await c.req.json()
  const ts = now()
  logTaskStart('StoryboardAPI', 'create', {
    episodeId: body.episode_id,
    shotNumber: body.storyboard_number || 1,
    sceneId: body.scene_id,
    characterIds: body.character_ids,
  })
  logTaskPayload('StoryboardAPI', 'create body', body)
  validateStoryboardBindings(body.episode_id, body.scene_id, body.character_ids)
  const res = db.insert(schema.storyboards).values({
    episodeId: body.episode_id,
    storyboardNumber: body.storyboard_number || 1,
    title: body.title,
    description: body.description,
    action: body.action,
    dialogue: body.dialogue,
    sceneId: body.scene_id,
    duration: body.duration || 10,
    createdAt: ts,
    updatedAt: ts,
  }).run()
  syncStoryboardCharacters(Number(res.lastInsertRowid), body.character_ids || [])
  const [result] = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.id, Number(res.lastInsertRowid))).all()
  logTaskSuccess('StoryboardAPI', 'create', {
    storyboardId: result.id,
    episodeId: result.episodeId,
    shotNumber: result.storyboardNumber,
  })
  return created(c, {
    ...toSnakeCase(result),
    character_ids: getStoryboardCharacterIds(result.id),
  })
})

// PUT /storyboards/:id
app.put('/:id', async (c) => {
  const id = Number(c.req.param('id'))
  const body = await c.req.json()
  const [storyboard] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
  if (!storyboard) return badRequest(c, '镜头不存在')
  logTaskStart('StoryboardAPI', 'update', {
    storyboardId: id,
    episodeId: storyboard.episodeId,
    fields: Object.keys(body),
  })
  logTaskPayload('StoryboardAPI', 'update body', body)

  const fieldMap: Record<string, string> = {
    title: 'title', description: 'description', shot_type: 'shotType',
    angle: 'angle', movement: 'movement', action: 'action',
    dialogue: 'dialogue', duration: 'duration', video_prompt: 'videoPrompt',
    image_prompt: 'imagePrompt', scene_id: 'sceneId', location: 'location',
    time: 'time', atmosphere: 'atmosphere', result: 'result',
    bgm_prompt: 'bgmPrompt', sound_effect: 'soundEffect',
  }

  const updates: Record<string, any> = { updatedAt: now() }
  for (const [snakeKey, camelKey] of Object.entries(fieldMap)) {
    if (snakeKey in body) updates[camelKey] = body[snakeKey]
  }

  if ('dialogue' in body) {
    updates.ttsAudioUrl = null
    updates.subtitleUrl = null
  }

  validateStoryboardBindings(
    storyboard.episodeId,
    'scene_id' in body ? body.scene_id : storyboard.sceneId,
    'character_ids' in body ? body.character_ids : getStoryboardCharacterIds(id),
  )

  db.update(schema.storyboards).set(updates).where(eq(schema.storyboards.id, id)).run()
  if ('character_ids' in body) syncStoryboardCharacters(id, body.character_ids || [])

  // 对话角色名验证：保存时检测 dialogue 中的角色名是否存在于剧组角色列表
  let dialogueValidation: any = null
  if ('dialogue' in body && body.dialogue) {
    const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, storyboard.episodeId)).all()
    const lines = parseDialogueLines(body.dialogue)
    if (lines.length === 0) {
      const parsed = parseDialogueForTTS(body.dialogue)
      if (!parsed.ignorable && parsed.speaker) lines.push({ speaker: parsed.speaker, text: parsed.pureText })
    }
    if (lines.length > 0) {
      dialogueValidation = validateDialogueLines(lines, ep?.dramaId || 0)
    }
  }

  logTaskSuccess('StoryboardAPI', 'update', {
    storyboardId: id,
    updatedFields: Object.keys(updates),
    characterIds: body.character_ids,
    dialogueValidation: dialogueValidation?.filter((v: any) => v.match_status !== 'matched' && v.match_status !== 'narrator'),
  })
  return success(c, dialogueValidation ? { dialogue_validation: dialogueValidation } : undefined)
})

// POST /storyboards/:id/generate-tts
app.post('/:id/generate-tts', async (c) => {
  const id = Number(c.req.param('id'))
  const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
  if (!sb) return badRequest(c, '镜头不存在')

  const dialogueLines = parseDialogueLines(sb.dialogue)
  // fallback：拆分不出多角色对话行时，用旧的单人逻辑
  const useMultiMode = dialogueLines.length > 1
  if (!useMultiMode && dialogueLines.length === 0) {
    const parsed = parseDialogueForTTS(sb.dialogue)
    if (parsed.ignorable) return badRequest(c, '该镜头没有可生成的对白或旁白')
  }

  logTaskStart('StoryboardAPI', 'generate-tts', {
    storyboardId: id,
    episodeId: sb.episodeId,
    dialoguePreview: (sb.dialogue || '').slice(0, 40),
    lineCount: dialogueLines.length,
    speakers: dialogueLines.map(l => l.speaker),
  })
  logTaskPayload('StoryboardAPI', 'generate-tts input', {
    storyboardId: id,
    episodeId: sb.episodeId,
    dialogue: sb.dialogue,
    parsedLines: dialogueLines,
  })

  const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, sb.episodeId)).all()

  try {
    // 多人对话：每个角色生成独立的音频文件
    if (useMultiMode) {
      const results = []
      const validations = validateDialogueLines(dialogueLines, ep?.dramaId || 0)

      // 记录不匹配的警告
      const warnings = validations.filter(v => v.warning)
      if (warnings.length > 0) {
        logTaskPayload('StoryboardAPI', 'generate-tts validation warnings', warnings)
      }

      for (const val of validations) {
        const audioPath = await generateTTS({
          text: val.text,
          voice: val.voice_id,
          configId: ep?.audioConfigId || null,
        })

        results.push({
          speaker: val.speaker,
          text: val.text,
          tts_audio_url: audioPath,
          voice_id: val.voice_id,
          match_status: val.match_status,
          ...(val.warning ? { warning: val.warning } : {}),
        })
      }

      // 将多条音频 URL 以 JSON 存入 ttsAudioUrl 字段
      db.update(schema.storyboards)
        .set({ ttsAudioUrl: JSON.stringify(results), updatedAt: now() })
        .where(eq(schema.storyboards.id, id))
        .run()

      logTaskSuccess('StoryboardAPI', 'generate-tts', {
        storyboardId: id,
        lineCount: results.length,
        speakers: results.map(r => r.speaker),
      })

      return success(c, { lines: results })
    } else {
      // 单人/无法拆分：用旧逻辑直接生成
      const parsed = dialogueLines.length === 1
        ? dialogueLines[0]
        : parseDialogueForTTS(sb.dialogue)
      const speaker = parsed.speaker || parsed.pureText ? parsed.speaker : ''
      const text = parsed.pureText || (dialogueLines.length === 1 ? dialogueLines[0].text : '')
      const validation = validateTTSSpeaker(speaker, ep?.dramaId || 0)

      const audioPath = await generateTTS({ text: text, voice: validation.voiceId, configId: ep?.audioConfigId || null })

      db.update(schema.storyboards)
        .set({ ttsAudioUrl: audioPath, updatedAt: now() })
        .where(eq(schema.storyboards.id, id))
        .run()

      logTaskSuccess('StoryboardAPI', 'generate-tts', {
        storyboardId: id,
        voiceId,
        path: audioPath,
        textLength: text.length,
      })

      return success(c, {
        tts_audio_url: audioPath,
        voice_id: validation.voiceId,
        match_status: validation.match_status,
        speaker,
        text,
        ...(validation.match_status === 'not_found' ? { warning: `角色"${speaker}"在剧组角色列表中不存在` } : {}),
        ...(validation.match_status === 'no_voice' ? { warning: `角色"${speaker}"尚未配置音色` } : {}),
      })
    }
  } catch (err: any) {
    logTaskError('StoryboardAPI', 'generate-tts', { storyboardId: id, error: err.message })
    return badRequest(c, err.message)
  }
})

// GET /storyboards/:id/validate-dialogue
// 验证分镜 dialogue 中所有角色台词行的匹配状态
app.get('/:id/validate-dialogue', async (c) => {
  const id = Number(c.req.param('id'))
  const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
  if (!sb) return badRequest(c, '镜头不存在')

  const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, sb.episodeId)).all()

  const dialogueLines = parseDialogueLines(sb.dialogue)
  // 拆分不出多角色对话行时，用旧逻辑提取单人
  if (dialogueLines.length === 0) {
    const parsed = parseDialogueForTTS(sb.dialogue)
    if (parsed.ignorable || !parsed.speaker) {
      return success(c, { lines: [], all_matched: true, issue_count: 0, summary: '无可验证的对话行' })
    }
    dialogueLines.push({ speaker: parsed.speaker, text: parsed.pureText })
  }

  const validations = validateDialogueLines(dialogueLines, ep?.dramaId || 0)
  const issues = validations.filter(v => v.match_status === 'not_found' || v.match_status === 'no_voice')

  // TTS 过期检测：检查已生成 TTS 的 voice_id 是否与当前角色 voiceStyle 一致
  let ttsStaleCount = 0
  let ttsStaleDetails: any[] = []
  if (sb.ttsAudioUrl) {
    try {
      const ttsLines = JSON.parse(sb.ttsAudioUrl)
      if (Array.isArray(ttsLines)) {
        for (const line of ttsLines) {
          if (!line.voice_id || !line.speaker) continue
          // 查找当前角色的最新 voiceStyle
          const chars = db.select().from(schema.characters)
            .where(eq(schema.characters.dramaId, ep?.dramaId || 0)).all()
          const char = chars.find(c => c.name === line.speaker)
          if (char?.voiceStyle && char.voiceStyle !== line.voice_id) {
            ttsStaleCount++
            ttsStaleDetails.push({
              speaker: line.speaker,
              stored_voice: line.voice_id,
              current_voice: char.voiceStyle,
              warning: `角色"${line.speaker}"的 TTS 音色已过期（当前音色 ${char.voiceStyle}，配音使用 ${line.voice_id}）`,
            })
          }
        }
      } else if (typeof sb.ttsAudioUrl === 'string' && sb.ttsAudioUrl.startsWith('static/')) {
        // 单人模式：无法精确检测，skipped
      }
    } catch (_) { /* 非 JSON 格式，跳过 */ }
  }

  return success(c, {
    lines: validations,
    all_matched: issues.length === 0,
    issue_count: issues.length,
    summary: issues.length === 0
      ? '全部角色匹配成功'
      : `${issues.length} 个角色存在音色匹配问题`,
    issues: issues.map(v => ({ speaker: v.speaker, status: v.match_status, warning: v.warning })),
    tts_stale: ttsStaleDetails.length > 0,
    tts_stale_count: ttsStaleCount,
    tts_stale_details: ttsStaleDetails,
  })
})

// DELETE /storyboards/:id
app.delete('/:id', async (c) => {
  const id = Number(c.req.param('id'))
  logTaskStart('StoryboardAPI', 'delete', { storyboardId: id })
  db.delete(schema.storyboardCharacters).where(eq(schema.storyboardCharacters.storyboardId, id)).run()
  db.delete(schema.storyboards).where(eq(schema.storyboards.id, id)).run()
  logTaskSuccess('StoryboardAPI', 'delete', { storyboardId: id })
  return success(c)
})

export default app
