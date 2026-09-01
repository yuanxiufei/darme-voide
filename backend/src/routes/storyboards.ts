import { Hono } from 'hono'
import { and, eq, gt, inArray, isNull } from 'drizzle-orm'
import ffmpeg from 'fluent-ffmpeg'
import fs from 'fs'
import path from 'path'
import { v4 as uuid } from 'uuid'
import { db, schema } from '../db/index.js'
import { getDataRoot, getStorageRoot } from '../config.js'
import { success, created, now, badRequest, notFound, parseParamId } from '../utils/response.js'
import { toSnakeCase } from '../utils/transform.js'
import { generateTTS } from '../services/tts-generation.js'
import { generateImage } from '../services/image-generation.js'
import { generateActionSuggestion, splitShotIntoSubShots, optimizeVideoPrompt } from '../services/text-generation.js'
import {
  buildStoryboardImagePrompt,
  STORYBOARD_IMAGE_NEGATIVE,
  getStoryboardCharacterAppearances,
  getStoryboardSceneDescription,
  getStoryboardReferenceImages,
} from '../shared/prompt-utils.js'
import { logTaskError, logTaskPayload, logTaskProgress, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import { scoreStoryboard } from '../services/qc-scoring.js'
import { retryFailedStoryboard } from '../services/qc-retry.js'
import { syncStoryboardProps, getStoryboardPropIds } from '../agents/tools/storyboard-tools.js'

const app = new Hono()

const IGNORE_TTS_SPEAKERS = /^(环境音|环境声|音效|效果音|sfx|sound ?effect|bgm|背景音|背景音乐|ambient)$/i
const IGNORE_TTS_TEXT = /^(无|无对白|无台词|无旁白|无需配音|无需对白|none|null|n\/a|na|环境音|环境声|音效|效果音|纯音效|纯环境音|只有环境音|仅环境音|背景音|背景音乐|bgm|sfx|ambient)$/i

/** 获取角色个性化声音参数（emotion/speed/pitch/voiceModel） */
function getCharacterVoiceParams(characterId: number | undefined, dramaId: number): {
  speed?: number; emotion?: string; pitch?: number; model?: string
} {
  if (!characterId) return {}
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, characterId), isNull(schema.characters.deletedAt))).all()
  if (!char) return {}
  return {
    speed: char.voiceSpeed ?? undefined,
    emotion: char.voiceEmotion ?? undefined,
    pitch: char.voicePitch ?? undefined,
    model: char.voiceModel ?? undefined,
  }
}

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

interface TTSValidationResult {
  match_status: TTSMatchStatus
  voiceId: string
  characterId?: number | null
}

/**
 * 验证单个台词行：角色名→剧组角色记录→音色配置 三连匹配
 */
function validateTTSSpeaker(speaker: string, dramaId: number): TTSValidationResult {
  // 旁白/画外音直接视为通过
  if (/^(旁白|画外音|narrator)$/i.test(speaker)) {
    return { match_status: 'narrator', voiceId: 'alloy' }
  }

  const chars = db.select().from(schema.characters)
    .where(and(eq(schema.characters.dramaId, dramaId), isNull(schema.characters.deletedAt))).all()
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

function syncStoryboardCharacters(storyboardId: number, characterIds: number[], costumes?: Record<number, string>) {
  db.delete(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .run()

  const uniqueIds = [...new Set((characterIds || []).filter(Boolean))]
  if (!uniqueIds.length) return

  for (const characterId of uniqueIds) {
    db.insert(schema.storyboardCharacters).values({
      storyboardId,
      characterId,
      costume: costumes?.[characterId] ?? null,
    }).run()
  }
}

function getStoryboardCharacterIds(storyboardId: number) {
  return db.select().from(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId)).all()
    .map(link => link.characterId)
}

/** 镜头级角色服装变体映射 { characterId: costume } */
function getStoryboardCharacterCostumes(storyboardId: number): Record<number, string> {
  const map: Record<number, string> = {}
  const links = db.select().from(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId)).all()
  for (const link of links) {
    if (link.costume) map[link.characterId] = link.costume
  }
  return map
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
  try {
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
  if ('prop_ids' in body) syncStoryboardProps(Number(res.lastInsertRowid), body.prop_ids || [])
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
    prop_ids: getStoryboardPropIds(result.id),
  })
  } catch (err: any) {
    logTaskError('StoryboardAPI', 'create', { error: err.message, stack: err.stack })
    return badRequest(c, err.message || 'Failed to create storyboard')
  }
})

// PUT /storyboards/:id
app.put('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  const body = await c.req.json()
  const [storyboard] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
  if (!storyboard) return notFound(c, '镜头不存在')
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
    custom_image_prompt: 'customImagePrompt', custom_video_prompt: 'customVideoPrompt',
    negative_prompt: 'negativePrompt',
    first_frame_prompt: 'firstFramePrompt', last_frame_prompt: 'lastFramePrompt',
    transition_type: 'transitionType', transition_duration: 'transitionDuration',
    transition_motive: 'transitionMotive',
    first_frame_image: 'firstFrameImage', last_frame_image: 'lastFrameImage',
    keyframe_prompt: 'keyframePrompt', keyframe_image: 'keyframeImage',
    asset_status: 'assetStatus',
    start_state: 'startState', end_state: 'endState', constraints: 'constraints',
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
  if ('character_ids' in body || 'character_costumes' in body) {
    const currentIds = 'character_ids' in body ? (body.character_ids || []) : getStoryboardCharacterIds(id)
    const costumes: Record<number, string> = body.character_costumes ?? getStoryboardCharacterCostumes(id)
    syncStoryboardCharacters(id, currentIds, costumes)
  }
  if ('prop_ids' in body) {
    syncStoryboardProps(id, Array.isArray(body.prop_ids) ? body.prop_ids : [])
  }

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
  } catch (err: any) {
    logTaskError('StoryboardAPI', 'update', { error: err.message, stack: err.stack })
    return badRequest(c, err.message || 'Failed to update storyboard')
  }
})

// POST /storyboards/:id/generate-tts
app.post('/:id/generate-tts', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
  if (!sb) return notFound(c, '镜头不存在')

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
        // 从角色记录中获取个性化声音参数
        const charVoiceParams = getCharacterVoiceParams(val.character_id ?? undefined, ep?.dramaId ?? 0)
        const audioPath = await generateTTS({
          text: val.text,
          voice: val.voice_id,
          speed: charVoiceParams.speed,
          emotion: charVoiceParams.emotion,
          pitch: charVoiceParams.pitch,
          model: charVoiceParams.model,
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
      let speaker: string
      let text: string
      if (dialogueLines.length === 1) {
        speaker = dialogueLines[0].speaker
        text = dialogueLines[0].text
      } else {
        const parsed = parseDialogueForTTS(sb.dialogue)
        speaker = parsed.speaker
        text = parsed.pureText
      }
      const validation = validateTTSSpeaker(speaker, ep?.dramaId || 0)
      const charVoiceParams = getCharacterVoiceParams(validation.characterId ?? undefined, ep?.dramaId ?? 0)

      const audioPath = await generateTTS({
        text, voice: validation.voiceId, configId: ep?.audioConfigId || null,
        speed: charVoiceParams.speed, emotion: charVoiceParams.emotion, pitch: charVoiceParams.pitch,
        model: charVoiceParams.model,
      })

      db.update(schema.storyboards)
        .set({ ttsAudioUrl: audioPath, updatedAt: now() })
        .where(eq(schema.storyboards.id, id))
        .run()

      logTaskSuccess('StoryboardAPI', 'generate-tts', {
        storyboardId: id,
        voiceId: validation.voiceId,
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
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
  if (!sb) return notFound(c, '镜头不存在')

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
            .where(and(eq(schema.characters.dramaId, ep?.dramaId || 0), isNull(schema.characters.deletedAt))).all()
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

// POST /storyboards/:id/regenerate-image — 单镜头图片重新生成（可选模型选择）
app.post('/:id/regenerate-image', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  const body = await c.req.json()
  const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
  if (!sb) return notFound(c, '镜头不存在')

  const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, sb.episodeId)).all()
  if (!ep) return badRequest(c, 'Episode not found')

  try {
    // 注入角色外观 + 场景描述 + 角色/场景参考图，保证人物与场景一致
    const charAppearances = getStoryboardCharacterAppearances(id)
    const sceneDesc = getStoryboardSceneDescription(id)
    const referenceImages = body.reference_images?.length
      ? body.reference_images
      : getStoryboardReferenceImages(id)

    // 自定义 prompt 优先 → 分镜级 customImagePrompt → 标准构建器
    const prompt = body.prompt
      || sb.customImagePrompt
      || buildStoryboardImagePrompt({
        description: sb.description || body.character_description || '',
        storyboardDescription: sb.description,
        characterDescription: charAppearances.length ? charAppearances.join('；') : null,
        sceneDescription: body.scene_description || sceneDesc || sb.location || '',
        shotType: sb.shotType || body.shot_type || '',
        cameraAngle: sb.angle || body.camera_angle || '',
        dramaStyle: body.style,
      })

    logTaskStart('StoryboardAPI', 'regenerate-image', {
      storyboardId: id, episodeId: sb.episodeId, dramaId: ep.dramaId, model: body.model || 'default',
    })

    const genId = await generateImage({
      storyboardId: id,
      dramaId: ep.dramaId,
      prompt,
      negativePrompt: body.negative_prompt || sb.negativePrompt || STORYBOARD_IMAGE_NEGATIVE,
      model: body.model,
      referenceImages,
      configId: ep.imageConfigId ?? undefined,
      force: body.force,
    })

    logTaskSuccess('StoryboardAPI', 'regenerate-image', { storyboardId: id, generationId: genId })
    return success(c, { image_generation_id: genId })
  } catch (err: any) {
    logTaskError('StoryboardAPI', 'regenerate-image', { storyboardId: id, error: err.message })
    return badRequest(c, err.message)
  }
})

// POST /storyboards/:id/regenerate-frame — 独立重新生成首帧/尾帧/关键帧图（frame_type: first_frame | last_frame | keyframe）
app.post('/:id/regenerate-frame', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  const body = await c.req.json()
  const frameType = ['last_frame', 'keyframe'].includes(body.frame_type) ? body.frame_type : 'first_frame'
  const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
  if (!sb) return notFound(c, '镜头不存在')

  const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, sb.episodeId)).all()
  if (!ep) return badRequest(c, 'Episode not found')

  try {
    // 注入角色外观 + 场景描述 + 角色/场景参考图，保证首尾帧人物与场景一致
    const charAppearances = getStoryboardCharacterAppearances(id)
    const sceneDesc = getStoryboardSceneDescription(id)
    const referenceImages = body.reference_images?.length
      ? body.reference_images
      : getStoryboardReferenceImages(id)

    // 首/尾/关键帧画面内容：优先请求体 prompt，其次分镜存库对应帧 prompt
    const frameContent = body.prompt
      || (frameType === 'first_frame' ? sb.firstFramePrompt
        : frameType === 'last_frame' ? sb.lastFramePrompt : sb.keyframePrompt)

    // 画面基底：标准构建器（始终注入角色外观 + 场景），帧画面内容作为附加描述叠加
    const basePrompt = buildStoryboardImagePrompt({
      description: sb.description || body.character_description || '',
      storyboardDescription: sb.description,
      characterDescription: charAppearances.length ? charAppearances.join('；') : null,
      sceneDescription: body.scene_description || sceneDesc || sb.location || '',
      shotType: sb.shotType || body.shot_type || '',
      cameraAngle: sb.angle || body.camera_angle || '',
      dramaStyle: body.style,
    })
    const frameHint = frameType === 'first_frame'
      ? 'opening frame, establishing the scene, subject at start position, beginning of the shot'
      : frameType === 'last_frame'
        ? 'closing frame, final composition, subject at end position, end of the shot'
        : 'mid-action keyframe, subject mid-motion, action or prop state in transition, intermediate moment of the shot'
    const prompt = [basePrompt, frameContent, frameHint].filter(Boolean).join(', ')

    logTaskStart('StoryboardAPI', 'regenerate-frame', {
      storyboardId: id, episodeId: sb.episodeId, dramaId: ep.dramaId, frameType, model: body.model || 'default',
    })

    const genId = await generateImage({
      storyboardId: id,
      dramaId: ep.dramaId,
      prompt,
      negativePrompt: body.negative_prompt || sb.negativePrompt || STORYBOARD_IMAGE_NEGATIVE,
      model: body.model,
      frameType,
      referenceImages,
      configId: ep.imageConfigId ?? undefined,
      force: body.force,
    })

    logTaskSuccess('StoryboardAPI', 'regenerate-frame', { storyboardId: id, frameType, generationId: genId })
    return success(c, { image_generation_id: genId, frame_type: frameType })
  } catch (err: any) {
    logTaskError('StoryboardAPI', 'regenerate-frame', { storyboardId: id, frameType, error: err.message })
    return badRequest(c, err.message)
  }
})

// POST /storyboards/:id/set-frame — 用户上传自定义图片/视频素材，自动匹配适配到首帧/尾帧
app.post('/:id/set-frame', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  const body = await c.req.json()
  const frameType = body.frame_type === 'last_frame' ? 'last_frame' : 'first_frame'
  const sourceUrl: string = body.source_url || ''
  const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
  if (!sb) return notFound(c, '镜头不存在')
  if (!sourceUrl) return badRequest(c, 'source_url required')

  try {
    // 视频素材：抽取首帧/尾帧作为该镜头的首/尾帧图；图片素材：直接使用
    const isVideo = /\.(mp4|mov|webm|m4v|avi|mpeg|mkv)$/i.test(sourceUrl)
    const frameUrl = isVideo ? await extractVideoFrame(sourceUrl, frameType) : sourceUrl

    const update: Record<string, any> = { updatedAt: now() }
    if (frameType === 'first_frame') update.firstFrameImage = frameUrl
    else update.lastFrameImage = frameUrl
    db.update(schema.storyboards).set(update).where(eq(schema.storyboards.id, id)).run()

    logTaskSuccess('StoryboardAPI', 'set-frame', { storyboardId: id, frameType, sourceUrl, frameUrl, isVideo })
    return success(c, { frame_type: frameType, frame_url: frameUrl })
  } catch (err: any) {
    logTaskError('StoryboardAPI', 'set-frame', { storyboardId: id, frameType, error: err.message })
    return badRequest(c, err.message)
  }
})

// DELETE /storyboards/:id
app.delete('/:id', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  logTaskStart('StoryboardAPI', 'delete', { storyboardId: id })
  db.delete(schema.storyboardCharacters).where(eq(schema.storyboardCharacters.storyboardId, id)).run()
  db.delete(schema.storyboards).where(eq(schema.storyboards.id, id)).run()
  logTaskSuccess('StoryboardAPI', 'delete', { storyboardId: id })
  return success(c)
})

// POST /storyboards/:id/action-suggestion — AI 生成运镜/动作建议
app.post('/:id/action-suggestion', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  try {
    const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
    if (!sb) return notFound(c, '镜头不存在')

    const suggestion = await generateActionSuggestion({
      title: sb.title,
      description: sb.description,
      action: sb.action,
      imagePrompt: sb.customImagePrompt || sb.imagePrompt,
      atmosphere: sb.atmosphere,
      shotType: sb.shotType,
      movement: sb.movement,
      angle: sb.angle,
    })

    return success(c, { suggestion })
  } catch (err: any) {
    logTaskError('StoryboardAPI', 'action-suggestion', { storyboardId: id, error: err.message })
    return badRequest(c, err.message || 'Failed to generate action suggestion')
  }
})

// POST /storyboards/:id/split — AI 拆分长镜头为多个子镜头（保留原镜头，追加新镜头）
app.post('/:id/split', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  try {
    const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
    if (!sb) return notFound(c, '镜头不存在')

    // 视觉风格：storyboard → episode → drama
    let visualStyle = ''
    const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, sb.episodeId)).all()
    if (ep) {
      const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, ep.dramaId)).all()
      if (drama) visualStyle = drama.style || ''
    }

    // 场景信息
    let sceneInfo = { location: sb.location || '', time: sb.time || '', atmosphere: sb.atmosphere || '' }
    if (sb.sceneId) {
      const [sc] = db.select().from(schema.scenes).where(eq(schema.scenes.id, sb.sceneId)).all()
      if (sc) sceneInfo = { location: sc.location, time: sc.time || '', atmosphere: sc.atmosphere || '' }
    }

    // 出场角色名
    const charIds = getStoryboardCharacterIds(id)
    let characterNames: string[] = []
    if (charIds.length) {
      characterNames = db.select().from(schema.characters)
        .where(inArray(schema.characters.id, charIds)).all()
        .map(ch => ch.name)
    }

    const subShots = await splitShotIntoSubShots({
      title: sb.title,
      action: sb.action,
      description: sb.description,
      shotType: sb.shotType,
      atmosphere: sb.atmosphere,
      dialogue: sb.dialogue,
      sceneInfo,
      characterNames,
      visualStyle,
    })

    // 落库：保留原镜头 + 新子镜头追加在原镜头之后 + 后续分镜号顺延
    const ts = now()
    const baseNumber = sb.storyboardNumber

    const shift = subShots.length
    if (shift > 0) {
      const later = db.select().from(schema.storyboards)
        .where(and(eq(schema.storyboards.episodeId, sb.episodeId), gt(schema.storyboards.storyboardNumber, baseNumber))).all()
      for (const s of later) {
        db.update(schema.storyboards).set({ storyboardNumber: s.storyboardNumber + shift })
          .where(eq(schema.storyboards.id, s.id)).run()
      }
    }

    const createdShots = subShots.map((sub, i) => {
      const res = db.insert(schema.storyboards).values({
        episodeId: sb.episodeId,
        sceneId: sb.sceneId,
        storyboardNumber: baseNumber + 1 + i,
        title: `${sb.title || '镜头'} · ${sub.shotSize}`,
        description: sub.visualFocus || sub.actionSummary,
        action: sub.actionSummary,
        shotType: sub.shotSize,
        movement: sub.cameraMovement || sb.movement,
        atmosphere: sb.atmosphere,
        dialogue: null,
        duration: Math.max(2, Math.min(4, Math.round((sb.duration || 4) / subShots.length))),
        status: 'pending',
        createdAt: ts,
        updatedAt: ts,
      }).run()
      const newId = Number(res.lastInsertRowid)
      if (charIds.length) syncStoryboardCharacters(newId, charIds)
      return {
        id: newId,
        storyboardNumber: baseNumber + 1 + i,
        shotSize: sub.shotSize,
        cameraMovement: sub.cameraMovement,
        actionSummary: sub.actionSummary,
        visualFocus: sub.visualFocus,
      }
    })

    logTaskSuccess('StoryboardAPI', 'split', { storyboardId: id, count: createdShots.length })
    return success(c, { subShots: createdShots })
  } catch (err: any) {
    logTaskError('StoryboardAPI', 'split', { storyboardId: id, error: err.message })
    return badRequest(c, err.message || 'Failed to split shot')
  }
})

// POST /storyboards/:id/optimize-prompt — AI 优化视频生成提示词（用户主动触发，对齐 gcc KeyframeEditor 的 AI 优化）
app.post('/:id/optimize-prompt', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  try {
    const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
    if (!sb) return notFound(c, '镜头不存在')

    const body = await c.req.json()

    // 视觉风格：storyboard → episode → drama
    let visualStyle = ''
    const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, sb.episodeId)).all()
    if (ep) {
      const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, ep.dramaId)).all()
      if (drama) visualStyle = drama.style || ''
    }

    // 场景信息
    let sceneInfo = { location: sb.location || '', time: sb.time || '', atmosphere: sb.atmosphere || '' }
    if (sb.sceneId) {
      const [sc] = db.select().from(schema.scenes).where(eq(schema.scenes.id, sb.sceneId)).all()
      if (sc) sceneInfo = { location: sc.location, time: sc.time || '', atmosphere: sc.atmosphere || '' }
    }

    // 出场角色名
    const charIds = getStoryboardCharacterIds(id)
    let characterNames: string[] = []
    if (charIds.length) {
      characterNames = db.select().from(schema.characters)
        .where(inArray(schema.characters.id, charIds)).all()
        .map(ch => ch.name)
    }

    const optimizedPrompt = await optimizeVideoPrompt({
      currentPrompt: body?.currentPrompt,
      title: sb.title,
      action: sb.action,
      description: sb.description,
      shotType: sb.shotType,
      movement: sb.movement,
      atmosphere: sb.atmosphere,
      sceneInfo,
      characterNames,
      visualStyle,
    })

    logTaskSuccess('StoryboardAPI', 'optimize-prompt', { storyboardId: id })
    return success(c, { optimizedPrompt })
  } catch (err: any) {
    logTaskError('StoryboardAPI', 'optimize-prompt', { storyboardId: id, error: err.message })
    return badRequest(c, err.message || 'Failed to optimize prompt')
  }
})

// POST /storyboards/:id/qc — 触发镜头级 QC 打分（唇形同步 / 角色一致性 / 连续性）
app.post('/:id/qc', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  try {
    const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
    if (!sb) return notFound(c, '镜头不存在')
    const result = scoreStoryboard(id)
    return success(c, result)
  } catch (err: any) {
    logTaskError('StoryboardAPI', 'qc-score', { storyboardId: id, error: err.message })
    return badRequest(c, err.message || 'Failed to score storyboard')
  }
})

// POST /storyboards/:id/retry-qc — 审片重跑：QC 未通过时只重写该失败分镜并重新生成图片/视频
app.post('/:id/retry-qc', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  try {
    const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
    if (!sb) return notFound(c, '镜头不存在')
    const result = await retryFailedStoryboard(id)
    return success(c, result)
  } catch (err: any) {
    logTaskError('StoryboardAPI', 'retry-qc', { storyboardId: id, error: err.message })
    return badRequest(c, err.message || 'Failed to retry storyboard')
  }
})

// GET /storyboards/:id/qc — 获取最近一次 QC 打分结果
app.get('/:id/qc', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid storyboard id')
  const [record] = db
    .select()
    .from(schema.videoQualityChecks)
    .where(eq(schema.videoQualityChecks.storyboardId, id))
    .all()
    .sort((a, b) => b.id - a.id)
  if (!record) return success(c, { message: 'No QC record yet' })
  return success(c, {
    ...record,
    issues: record.issues ? JSON.parse(record.issues) : [],
    dimensions: record.dimensions ? JSON.parse(record.dimensions) : {},
  })
})

/** 将相对媒体路径转为绝对路径 */
function toAbsMediaPath(relativePath: string): string {
  if (path.isAbsolute(relativePath)) return relativePath
  if (relativePath.startsWith('static/')) return path.join(getDataRoot(), relativePath)
  return path.join(getStorageRoot(), relativePath)
}

/**
 * 从视频素材抽取首帧/尾帧，返回可访问的相对图片路径。
 * 用于用户上传自定义视频素材后自动匹配适配到镜头首/尾帧。
 */
async function extractVideoFrame(sourceUrl: string, frameType: 'first_frame' | 'last_frame'): Promise<string> {
  const absPath = toAbsMediaPath(sourceUrl)
  if (!fs.existsSync(absPath)) throw new Error(`素材文件不存在: ${sourceUrl}`)

  const duration = await new Promise<number>((resolve) => {
    ffmpeg.ffprobe(absPath, (err, metadata) => {
      if (err) { resolve(0); return }
      resolve(metadata.format.duration || 0)
    })
  })

  const outputDir = path.join(getStorageRoot(), 'frames')
  fs.mkdirSync(outputDir, { recursive: true })
  const filename = `${uuid()}.jpg`
  const outPath = path.join(outputDir, filename)
  const seek = frameType === 'first_frame' ? 0 : Math.max(0, duration - 0.2)

  await new Promise<void>((resolve, reject) => {
    ffmpeg(absPath)
      .seekInput(seek)
      .outputOptions(['-frames:v', '1', '-q:v', '2'])
      .output(outPath)
      .on('end', () => resolve())
      .on('error', (err) => reject(err))
      .run()
  })

  return `static/frames/${filename}`
}

export default app
