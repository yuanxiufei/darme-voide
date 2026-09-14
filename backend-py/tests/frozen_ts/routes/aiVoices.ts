/**
 * AI 音色管理
 * GET  /api/v1/ai-voices       - 获取音色列表
 * POST /api/v1/ai-voices/sync  - 从 MiniMax 同步音色
 */
import fs from 'fs'
import path from 'path'
import { v4 as uuid } from 'uuid'
import { Hono } from 'hono'
import { eq, and, isNull } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { getStorageRoot } from '../config.js'
import { success, badRequest, notFound, now } from '../utils/response.js'
import { joinProviderUrl } from '../services/adapters/url.js'
import { getAudioConfig } from '../services/ai.js'
import { inferVoiceRoleTags } from '../services/text-generation.js'
import { generateTTS } from '../services/tts-generation.js'
import { cloneVoice, cloneVoiceCosyVoice } from '../services/voice-clone.js'
import { logTaskError } from '../utils/task-logger.js'
import { vendorResponseError } from '../utils/vendor-errors.js'

const app = new Hono()

// GET /ai-voices?provider=minimax
app.get('/', async (c) => {
  const provider = c.req.query('provider') || 'minimax'
  const rows = db.select().from(schema.aiVoices)
    .where(eq(schema.aiVoices.provider, provider))
    .all()

  const parsed = rows.map(r => ({
    voice_id: r.voiceId,
    voice_name: r.voiceName,
    description: r.description ? JSON.parse(r.description) : [],
    role_tags: parseJsonArray(r.roleTags),
    language: r.language,
    provider: r.provider,
  }))

  return success(c, parsed)
})

// POST /ai-voices/preview - 生成音色试听（不绑定角色，用固定文案）
app.post('/preview', async (c) => {
  const body = await c.req.json().catch(() => ({}))
  const voiceId = body.voice_id || body.voiceId
  if (!voiceId) return badRequest(c, 'voice_id is required')

  try {
    const sampleText = body.text || '你好，欢迎来到短剧工坊，这是我的声音试听。'
    const audioPath = await generateTTS({ text: sampleText, voice: voiceId, configId: body.config_id ?? null })
    return success(c, { voice_id: voiceId, url: audioPath })
  } catch (err: any) {
    logTaskError('AiVoices', 'preview', { voiceId, error: err.message })
    return badRequest(c, `试听生成失败: ${err.message}`)
  }
})

// POST /ai-voices/clone - 音色快速复刻（上传参考音频，克隆为可复用音色）
// minimax：MiniMax voice_clone 注册式克隆；cosyvoice：本地零样本克隆（参考音频 + 参考文本）
app.post('/clone', async (c) => {
  try {
    const body = await c.req.parseBody()
    const file = body['file']
    const voiceName = String(body['voice_name'] || body['voiceName'] || '').trim()
    const demoText = body['demo_text'] || body['demoText'] || undefined
    const promptText = String(body['prompt_text'] || body['promptText'] || '').trim()
    const rawVoiceId = String(body['voice_id'] || body['voiceId'] || '')

    if (!file || !(file instanceof File)) return badRequest(c, 'file is required')

    const config = getAudioConfig()
    const isCosyVoice = config.provider === 'cosyvoice'
    if (config.provider !== 'minimax' && !isCosyVoice) {
      return badRequest(c, `音色复刻当前仅支持 MiniMax / CosyVoice 音频服务（当前 ${config.provider}）`)
    }
    if (isCosyVoice && !promptText) {
      return badRequest(c, 'CosyVoice 零样本克隆需要提供参考音频文本 prompt_text')
    }

    const fileBuffer = Buffer.from(await file.arrayBuffer())
    const voiceId = generateCloneVoiceId(rawVoiceId, isCosyVoice ? 'cv_' : 'ds_')

    let demoAudio: string | undefined
    let referenceAudio: string | null = null
    let refPromptText: string | null = null

    if (isCosyVoice) {
      const sampleText = String(demoText || '你好，欢迎来到短剧工坊，这是我的声音试听。')
      const r = await cloneVoiceCosyVoice({
        baseUrl: config.baseUrl,
        fileBuffer,
        promptText,
        demoText: sampleText,
        model: config.model,
      })
      // demo 音频（base64）落盘为可试听 URL
      if (r.demoAudio) {
        const audioDir = path.join(getStorageRoot(), 'audio')
        fs.mkdirSync(audioDir, { recursive: true })
        const demoName = `${uuid()}.mp3`
        fs.writeFileSync(path.join(audioDir, demoName), Buffer.from(r.demoAudio, 'base64'))
        demoAudio = `static/audio/${demoName}`
      }
      // 参考音频落盘，供后续 TTS 零样本复用
      const voicesDir = path.join(getStorageRoot(), 'voices')
      fs.mkdirSync(voicesDir, { recursive: true })
      const refName = `${uuid()}.wav`
      fs.writeFileSync(path.join(voicesDir, refName), fileBuffer)
      referenceAudio = `static/voices/${refName}`
      refPromptText = promptText
    } else {
      const result = await cloneVoice({
        baseUrl: config.baseUrl,
        apiKey: config.apiKey,
        fileBuffer,
        filename: file.name,
        voiceId,
        demoText: demoText ? String(demoText) : undefined,
        model: config.model,
      })
      demoAudio = result.demoAudio
    }

    // 写入音色库（不覆盖已有）
    db.insert(schema.aiVoices).values({
      voiceId,
      voiceName: voiceName || `克隆音色 ${voiceId.slice(-6)}`,
      description: JSON.stringify(['克隆音色']),
      language: '中文',
      provider: config.provider,
      referenceAudio,
      promptText: refPromptText,
      createdAt: now(),
    }).onConflictDoNothing({ target: schema.aiVoices.voiceId }).run()

    return success(c, { voice_id: voiceId, demo_audio: demoAudio })
  } catch (err: any) {
    logTaskError('AiVoices', 'clone', { error: err.message })
    return badRequest(c, `音色克隆失败: ${err.message}`)
  }
})

// POST /ai-voices/generate-from-characters - 根据剧人物批量生成专属音色（用角色试听音频做声线克隆）
app.post('/generate-from-characters', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}))
    const dramaId = Number(body.drama_id || body.dramaId)
    if (!dramaId) return badRequest(c, 'drama_id is required')

    const drama = db.select().from(schema.dramas)
      .where(eq(schema.dramas.id, dramaId))
      .get()
    if (!drama) return notFound(c, 'Drama not found')

    const characters = db.select().from(schema.characters)
      .where(and(eq(schema.characters.dramaId, dramaId), isNull(schema.characters.deletedAt)))
      .all()
      .filter(ch => ch.voiceStyle)

    if (characters.length === 0) {
      return badRequest(c, '该剧暂无已分配音色的角色，请先在角色页分配音色')
    }

    const config = getAudioConfig()
    const isCosyVoice = config.provider === 'cosyvoice'
    const audioConfigId = resolveDramaConfigId(dramaId, 'audioConfigId')

    // 声线克隆要求参考音频 ≥10 秒，现有试听仅约 4 秒会报 "voice duration too short"，
    // 故改用角色音色即时合成一段约 15 秒的参考音频用于克隆。
    const REF_TEXT = '在这座城市里，每天都有许多故事在上演。清晨的阳光洒在街道上，午后微风拂过树梢，傍晚霞光映红了天边。我愿意把最温暖的声音带给你，陪伴你度过每一个平凡而美好的日子。'

    const results: any[] = []
    let okCount = 0

    for (const ch of characters) {
      const name = ch.name || '角色'
      try {
        // 已是克隆专属音色（ds_/cv_ 前缀）则跳过，避免重复生成
        if (/^(ds_|cv_)/.test(ch.voiceStyle!)) {
          results.push({ character_id: ch.id, name, status: 'skipped', voice_id: ch.voiceStyle, reason: '已有专属音色' })
          continue
        }

        // 即时合成 ≥10s 参考音频（MiniMax voice_clone 最低 10 秒）
        const refPath = await generateTTS({ text: REF_TEXT, voice: ch.voiceStyle!, configId: audioConfigId })
        const filePath = resolveLocalAudioPath(refPath)
        if (!filePath || !fs.existsSync(filePath)) {
          results.push({ character_id: ch.id, name, status: 'failed', error: '参考音频落盘失败' })
          continue
        }

        const fileBuffer = fs.readFileSync(filePath)
        // demo 试听文案（短句）；参考文本仅 CosyVoice 零样本需要
        const demoText = `你好，我是${name}。很高兴认识你，这是我的专属音色。`
        const voiceId = generateCloneVoiceId('', isCosyVoice ? 'cv_' : 'ds_')

        let referenceAudio: string | null = null
        let refPromptText: string | null = null

        if (isCosyVoice) {
          await cloneVoiceCosyVoice({
            baseUrl: config.baseUrl,
            fileBuffer,
            promptText: REF_TEXT,
            demoText,
            model: config.model,
          })
          referenceAudio = refPath
          refPromptText = REF_TEXT
        } else {
          await cloneVoice({
            baseUrl: config.baseUrl,
            apiKey: config.apiKey,
            fileBuffer,
            filename: path.basename(refPath),
            voiceId,
            voiceName: name,
            demoText,
            model: config.model,
          })
        }

        // 写入音色库（不覆盖已有）
        db.insert(schema.aiVoices).values({
          voiceId,
          voiceName: name,
          description: JSON.stringify([ch.role || '角色音色']),
          language: '中文',
          provider: config.provider,
          roleTags: JSON.stringify([mapRoleToTag(ch.role)]),
          referenceAudio,
          promptText: refPromptText,
          createdAt: now(),
        }).onConflictDoNothing({ target: schema.aiVoices.voiceId }).run()

        // 角色改用专属音色
        db.update(schema.characters)
          .set({ voiceStyle: voiceId, voiceProvider: config.provider, updatedAt: now() })
          .where(eq(schema.characters.id, ch.id))
          .run()

        results.push({ character_id: ch.id, name, status: 'success', voice_id: voiceId })
        okCount++
      } catch (err: any) {
        logTaskError('AiVoices', 'generate-from-characters', { characterId: ch.id, name, error: err.message })
        results.push({ character_id: ch.id, name, status: 'failed', error: err.message })
      }
    }

    return success(c, { total: characters.length, success_count: okCount, results })
  } catch (err: any) {
    logTaskError('AiVoices', 'generate-from-characters', { error: err.message })
    return badRequest(c, `生成音色失败: ${err.message}`)
  }
})

// POST /ai-voices/sync
app.post('/sync', async (c) => {
  try {
  // 从数据库获取 minimax 的音频配置
  const rows = db.select().from(schema.aiServiceConfigs)
    .where(eq(schema.aiServiceConfigs.serviceType, 'audio'))
    .all()
    .filter(r => r.isActive && r.provider === 'minimax')

  if (rows.length === 0) {
    return badRequest(c, 'No active minimax audio config found')
  }

  const config = rows[0]
  if (!config.apiKey) {
    return badRequest(c, 'MiniMax API key not configured')
  }

  // 调用 MiniMax get_voice API
  const resp = await fetch(joinProviderUrl(config.baseUrl, '/v1', '/get_voice'), {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${config.apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ voice_type: 'all' }),
    signal: AbortSignal.timeout(30_000),
  })

  if (!resp.ok) {
    const err = await vendorResponseError(resp, 'audio')
    return badRequest(c, err.message)
  }

  const result = await resp.json() as any
  if (result.base_resp?.status_code !== 0) {
    return badRequest(c, result.base_resp?.status_msg || 'Failed to fetch voices')
  }

  const voices = (result.system_voice || []).filter((v: any) => shouldKeepVoice(v))
  const ts = now()

  // 先清空旧数据
  db.delete(schema.aiVoices).where(eq(schema.aiVoices.provider, 'minimax')).run()

  // 批量插入新数据
  const insertRows = voices.map((v: any) => ({
    voiceId: v.voice_id,
    voiceName: v.voice_name,
    description: JSON.stringify(v.description || []),
    language: extractLanguage(v.voice_id, v.voice_name),
    provider: 'minimax',
    createdAt: ts,
  }))

  if (insertRows.length > 0) {
    db.insert(schema.aiVoices).values(insertRows).run()

    // 批量 AI 打标（旁白/主角/反派/配角）；失败不阻塞 sync，role_tags 留空由前端回退正则
    try {
      const tags = await inferVoiceRoleTags(insertRows.map((r: any) => ({
        voiceId: r.voiceId,
        voiceName: r.voiceName,
        description: JSON.parse(r.description || '[]'),
      })))
      for (const [voiceId, tagsArr] of Object.entries(tags)) {
        db.update(schema.aiVoices)
          .set({ roleTags: JSON.stringify(tagsArr) })
          .where(eq(schema.aiVoices.voiceId, voiceId))
          .run()
      }
    } catch (err: any) {
      logTaskError('AiVoices', 'tag-voices', { error: err.message })
    }
  }

  return success(c, { count: insertRows.length, message: `Synced ${insertRows.length} voices` })
  } catch (err: any) {
    logTaskError('AiVoices', 'sync-minimax', { error: err.message, stack: err.stack })
    return badRequest(c, err.message || 'Failed to sync voices')
  }
})

/**
 * 从 voice_id 或 voice_name 推断语言
 */
function extractLanguage(voiceId: string, voiceName: string): string {
  const text = `${voiceId} ${voiceName}`.toLowerCase()
  if (text.includes('cantonese') || text.includes('粤')) return '粤语'
  if (text.includes('english') || text.includes('aussie')) return '英语'
  if (text.includes('japanese') || text.includes('日语')) return '日语'
  if (text.includes('korean') || text.includes('韩')) return '韩语'
  if (text.includes('spanish')) return '西班牙语'
  if (text.includes('portuguese')) return '葡萄牙语'
  if (text.includes('french')) return '法语'
  if (text.includes('indonesian')) return '印尼语'
  if (text.includes('german')) return '德语'
  if (text.includes('russian')) return '俄语'
  if (text.includes('italian')) return '意大利语'
  if (text.includes('arabic')) return '阿拉伯语'
  if (text.includes('turkish')) return '土耳其语'
  if (text.includes('ukrainian')) return '乌克兰语'
  if (text.includes('dutch')) return '荷兰语'
  if (text.includes('vietnamese')) return '越南语'
  if (text.includes('chinese') || text.includes('mandarin') || text.includes('中文')) return '中文'
  return '其他'
}

function shouldKeepVoice(voice: { voice_id: string, voice_name: string }) {
  const language = extractLanguage(voice.voice_id, voice.voice_name)
  if (language !== '中文' && language !== '粤语') return false

  const text = `${voice.voice_id} ${voice.voice_name}`.toLowerCase()

  const excludedPatterns = [
    'jingpin',
    '-beta',
    'cartoon_pig',
    'cute_boy',
    'lovely_girl',
    'clever_boy',
    'robot_armor',
    'news_anchor',
    'male_announcer',
    'radio_host',
    'hk_flight_attendant',
  ]

  return !excludedPatterns.some(pattern => text.includes(pattern))
}

// 把 /static/... 相对 URL 解析为本地磁盘绝对路径
function resolveLocalAudioPath(url: string): string | null {
  if (!url) return null
  const cleaned = url.split('?')[0]
  const rel = cleaned.replace(/^\/?static\//, '').replace(/^\//, '')
  if (!rel) return null
  return path.join(getStorageRoot(), rel)
}

// 解析剧集级配置 id（音色生成用剧的 audioConfigId，找不到则走默认）
function resolveDramaConfigId(dramaId: number, field: 'imageConfigId' | 'audioConfigId'): number | undefined {
  const eps = db.select().from(schema.episodes)
    .where(and(eq(schema.episodes.dramaId, dramaId), isNull(schema.episodes.deletedAt)))
    .all()
  for (const e of eps) {
    const v = e[field] as number | null
    if (v != null) return v
  }
  return undefined
}

// 角色 role（主角/反派/龙套/旁白）→ 音色 role_tags 四类（龙套归入配角）
function mapRoleToTag(role: string | null): string {
  const r = (role || '').trim()
  if (r === '主角' || r === '反派' || r === '旁白') return r
  return '配角'
}

function generateCloneVoiceId(raw: string, prefix = 'ds_'): string {
  if (raw) {
    const v = raw.trim()
    if (/^[a-zA-Z][a-zA-Z0-9_-]{7,255}$/.test(v) && !/[_-]$/.test(v)) return v
    throw new Error('voice_id 需 8-256 字符，首字符为英文字母，仅含字母/数字/-/_')
  }
  return prefix + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 6)
}

function parseJsonArray(raw: string | null): string[] {
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.map(String) : []
  } catch {
    return []
  }
}

export default app
