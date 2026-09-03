import { Hono } from 'hono'
import { and, eq, isNull } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, badRequest, notFound, now, parseParamId } from '../utils/response.js'
import { generateVoiceSample } from '../services/tts-generation.js'
import { generateImage } from '../services/image-generation.js'
import { logTaskError, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import { buildCharacterImagePrompt, buildCharacterVisualsClause, buildEquipImagePrompt, buildEquipArtStyleSuffix, buildCharacterArtStyleSuffix, buildCharacterNegativePrompt, buildThreeViewNegative, buildEquipNegative, buildExpressionImagePrompt, buildExpressionNegative, buildItemImagePrompt, buildItemNegative, ITEM_IMAGE_SIZE, EXPRESSION_PRESETS, findExpressionPreset, THREE_VIEW_COMBINED_LAYOUT, THREE_VIEW_SIZE } from '../shared/prompt-utils.js'
import { splitCharacterVisuals } from '../services/text-generation.js'
import { ensureCostumeId } from '../services/bible-ids.js'

const app = new Hono()

// 解析 drama 级生成配置：角色/场景的图片与音色试听是 drama 级共享资源，不绑定具体 episode。
// 优先取该 drama 下第一个已配置的 episode configId，否则返回 undefined（走全局默认配置）。
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

/** 读取剧集视觉风格（dramas.style：realistic/anime/ghibli/cinematic/comic/watercolor） */
function getDramaStyle(dramaId: number): string | null {
  const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
  return drama?.style || null
}

/** 读取全局默认画风（app_settings.art_style），未设置返回 null */
function getGlobalArtStyle(): string | null {
  const [row] = db.select().from(schema.appSettings).where(eq(schema.appSettings.key, 'art_style')).all()
  return row?.value || null
}

/**
 * 三级画风解析：角色 style → 剧集 style → 全局默认 → realistic（与 dramas.style 默认一致）。
 * 角色/剧集均未指定时使用全局默认画风，保证全链路兜底一致。
 */
function getEffectiveArtStyle(charStyle: string | null | undefined, dramaStyle: string | null): string {
  return charStyle || dramaStyle || getGlobalArtStyle() || 'realistic'
}

/** 角色三视图横向长图（combined）localPath/URL 提取 */
function getCharacterThreeViewCombinedImage(char: any): string | null {
  if (!char?.threeViews) return null
  try {
    const views = JSON.parse(char.threeViews)
    if (!views) return null
    const combined = views.combined || views.front || views.side || views.back
    return combined?.imageUrl || null
  } catch {
    return null
  }
}

/** 解析请求体中的显式参考图（body.reference_images / body.referenceImages），支持数组与 JSON 字符串 */
function parseBodyRefImages(body: any): string[] | undefined {
  const raw = body?.reference_images ?? body?.referenceImages
  if (Array.isArray(raw)) return raw.map((x: any) => String(x).trim()).filter(Boolean)
  if (typeof raw === 'string' && raw.trim()) {
    try {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) return parsed.map((x: any) => String(x).trim()).filter(Boolean)
    } catch { /* not json string */ }
  }
  return undefined
}

/**
 * 角色图「视觉锚定」参考图解析（自动把角色已生成的基准图作为参考图随生成下发）：
 * - 显式传 reference_images → 原样使用（优先级最高）
 * - body.anchor：
 *   - 'none' → 不锚定
 *   - 'main'/'image' → 仅主立绘（imageUrl/localPath）
 *   - 'three_views' → 优先三视图 combined，无则主立绘
 *   - 'auto'（默认）→ mode='main'（三视图生成）以主立绘为锚；
 *     其余（装备/表情）以「三视图 combined → 主立绘」为锚
 */
function resolveCharacterAnchors(char: any, body: any, mode: 'main' | 'character' = 'character'): string[] | undefined {
  const explicit = parseBodyRefImages(body)
  if (explicit) return explicit.slice(0, 6)
  const anchor = (body?.anchor as string | undefined) || 'auto'
  if (anchor === 'none') return undefined

  const combined = getCharacterThreeViewCombinedImage(char)
  const main = char?.imageUrl || char?.localPath || null

  if (anchor === 'main' || anchor === 'image') {
    if (main) return [main]
    return combined && mode !== 'main' ? [combined] : undefined
  }
  if (anchor === 'three_views') {
    if (combined) return [combined]
    return main ? [main] : undefined
  }
  // auto
  if (mode === 'main') return main ? [main] : undefined
  if (combined) return [combined]
  return main ? [main] : undefined
}

/**
 * 角色图 prompt 统一收口：即使走了自定义 prompt（body.prompt / char.customPrompt），
 * 也强制追加统一画风后缀，防止手写 prompt 导致动漫/真人风格漂移；
 * 同时把智能拆分出的服装/武器/首饰（visualsClause）注入，
 * 确保提取信息在自定义 prompt 场景下也不会丢失。
 */
function resolveCharacterPrompt(
  userPrompt: string | null | undefined,
  autoPrompt: string,
  dramaStyle: string | null,
  visualsClause = '',
): string {
  const user = (userPrompt || '').trim()
  if (!user) return autoPrompt
  const base = visualsClause ? `${user}, ${visualsClause}` : user
  return `${base}${buildCharacterArtStyleSuffix(dramaStyle)}`
}

/** 角色图负面提示词收口：用户未填时按剧集画风给默认负面词（排除对立风格） */
function resolveCharacterNegative(userNegative: string | null | undefined, dramaStyle: string | null): string {
  const user = (userNegative || '').trim()
  return user || buildCharacterNegativePrompt(dramaStyle)
}

/** 判断是否为旧版「单图/含人物」装备提示词（历史遗留，会造成装备三视图生成出人） */
function isLegacyEquipPrompt(p: string | null | undefined): boolean {
  const text = (p || '').trim()
  if (!text) return false
  // 旧提示词把整段角色外貌（character appearance: ...）/ 角色名（for the character X）灌入
  if (text.includes('character appearance') || text.includes('for the character')) return true
  // 旧「单视角」提示词（单件物体单张图，非三视图并排）
  if (text.includes('single view') && !text.includes('side by side') && !text.includes('three')) return true
  // 旧版含人物语义词且缺失无人三视图结构
  if ((text.includes('realistic cinematic character design') || text.includes('natural skin texture'))
    && !text.includes('side by side')) return true
  return false
}

/** 装备图 prompt 收口：legacy 旧文本一律丢弃走无人三视图构建器；
 * 用户手写/新版提示词保留，但统一追加无人物画风尾并补充防人物负向。 */
function resolveEquipPrompt(
  userPrompt: string | null | undefined,
  autoPrompt: string,
  dramaStyle: string | null,
): string {
  const user = (userPrompt || '').trim()
  if (!user || isLegacyEquipPrompt(user)) return autoPrompt
  return `${user}${buildEquipArtStyleSuffix(dramaStyle)}`
}

// GET /characters/:id
app.get('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  if (!char) return notFound(c, 'Character not found')
  return success(c, char)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// PUT /characters/:id
app.put('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const body = await c.req.json()
  const updates: Record<string, any> = { updatedAt: now() }
  for (const key of [
    'name', 'role', 'roleType', 'description', 'appearance', 'personality',
    'voiceStyle', 'voiceProvider', 'voiceSpeed', 'voiceEmotion',
    'voicePitch', 'voiceModel', 'clothing', 'weapons', 'accessories',
    'customPrompt', 'negativePrompt', 'style', 'coreFeatures', 'costumes', 'variations', 'threeViews', 'imageUrl', 'localPath', 'referenceImages',
    'expressions', 'itemImages',
    'clothingPrompt', 'clothingNegativePrompt', 'weaponPrompt', 'weaponNegativePrompt', 'accessoryPrompt', 'accessoryNegativePrompt',
  ]) {
    const snakeKey = key.replace(/[A-Z]/g, m => '_' + m.toLowerCase())
    if (snakeKey in body) updates[key] = body[snakeKey]
    else if (key in body) updates[key] = body[key]
  }
  if ('voice_style' in body || 'voiceStyle' in body) {
    updates.voiceSampleUrl = null
  }
  db.update(schema.characters).set(updates).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).run()
  ensureCostumeId(id)
  const [updated] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  return success(c, updated)
  } catch (err: any) { logTaskError('CharacterAPI', 'update', { error: err.message, id: c.req.param('id') }); return badRequest(c, err.message) }
})

// POST /characters/:id/auto-split-visuals — 从外貌特征智能拆分服装/武器/首饰并回填
app.post('/:id/auto-split-visuals', async (c) => {
  try {
    const id = parseParamId(c)
    if (id == null) return notFound(c, 'Invalid character id')
    const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
    if (!char) return notFound(c, 'Character not found')
    const body = await c.req.json().catch(() => ({}))
    const appearance = ((body as any).appearance || char.appearance || '').trim()
    if (!appearance) return badRequest(c, '请先填写「外貌特征」')

    logTaskStart('SplitVisuals', 'route', { characterId: id })
    const result = await splitCharacterVisuals({ appearance })
    logTaskSuccess('SplitVisuals', 'route', { characterId: id, result })
    return success(c, result)
  } catch (err: any) {
    logTaskError('SplitVisuals', 'route', { error: err.message, id: c.req.param('id') })
    return badRequest(c, `智能拆分失败: ${err.message}`)
  }
})

// DELETE /characters/:id
app.delete('/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  db.update(schema.characters).set({ deletedAt: now() }).where(eq(schema.characters.id, id)).run()
  return success(c)
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

// POST /characters/:id/generate-voice-sample — 生成角色音色试听
app.post('/:id/generate-voice-sample', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const body = await c.req.json().catch(() => ({}))
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  if (!char) return badRequest(c, 'Character not found')
  if (!char.voiceStyle) return badRequest(c, '请先分配音色')
  const ep = body.episode_id
    ? db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()[0]
    : undefined
  if (body.episode_id && !ep) return badRequest(c, 'Episode not found')

  try {
    logTaskStart('VoiceSample', 'generate', { characterId: id, characterName: char.name, episodeId: ep?.id, voice: char.voiceStyle })
    const audioPath = await generateVoiceSample(char.name, char.voiceStyle, ep?.audioConfigId ?? resolveDramaConfigId(char.dramaId, 'audioConfigId'))
    db.update(schema.characters)
      .set({ voiceSampleUrl: audioPath, updatedAt: now() })
      .where(eq(schema.characters.id, id)).run()
    logTaskSuccess('VoiceSample', 'generate', { characterId: id, path: audioPath })
    return success(c, { voice_sample_url: audioPath })
  } catch (err: any) {
    logTaskError('VoiceSample', 'generate', { characterId: id, error: err.message })
    return badRequest(c, `TTS 生成失败: ${err.message}`)
  }
})

// POST /characters/:id/generate-image
app.post('/:id/generate-image', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const body = await c.req.json()
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  if (!char) return badRequest(c, 'Character not found')
  const ep = body.episode_id
    ? db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()[0]
    : undefined
  if (body.episode_id && !ep) return badRequest(c, 'Episode not found')

  // 自定义 prompt 优先 → 角色 customPrompt → 统一构建器；无论哪条路径都强制追加统一画风后缀
  const dramaStyle = getEffectiveArtStyle(char.style, getDramaStyle(char.dramaId))
  const autoPrompt = buildCharacterImagePrompt({ ...char, ...body, dramaStyle })
  const visualsClause = buildCharacterVisualsClause({ ...char, ...body })
  const prompt = resolveCharacterPrompt(body.prompt || char.customPrompt, autoPrompt, dramaStyle, visualsClause)

  try {
    logTaskStart('CharacterImage', 'generate', { characterId: id, episodeId: ep?.id, dramaId: char.dramaId, model: body.model || 'default' })
    const genId = await generateImage({
      characterId: id,
      dramaId: char.dramaId,
      prompt,
      negativePrompt: resolveCharacterNegative(body.negative_prompt || char.negativePrompt, dramaStyle),
      model: body.model,
      // 立绘重生成仅支持显式参考图（reference_images），不做自动锚定，保持纯文生图基线
      referenceImages: parseBodyRefImages(body),
      costume: body.costume,
      colorGrade: body.color_grade ?? body.colorGrade,
      configId: ep?.imageConfigId ?? resolveDramaConfigId(char.dramaId, 'imageConfigId'),
    })
    logTaskSuccess('CharacterImage', 'generate', { characterId: id, generationId: genId })
    return success(c, { image_generation_id: genId })
  } catch (err: any) {
    logTaskError('CharacterImage', 'generate', { characterId: id, error: err.message })
    return badRequest(c, err.message)
  }
})

// POST /characters/:id/generate-prompt — 根据形象设定生成/预览提示词（不落库）
// type: character（角色立绘）/ clothing / weapon / accessory（装备设定图）
// body 可携带表单未保存的最新字段，服务端合并后按对象类型构建正反向提示词
app.post('/:id/generate-prompt', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const body = await c.req.json().catch(() => ({}))
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  if (!char) return badRequest(c, 'Character not found')

  const type = body.type || 'character'
  if (!['character', 'clothing', 'weapon', 'accessory'].includes(type)) {
    return badRequest(c, 'type 必须是 character/clothing/weapon/accessory 之一')
  }

  // 前端表单最新值覆盖（camelCase 优先，兼容 snake_case），未传字段保持库内原值
  const merged: any = { ...char }
  for (const key of ['name', 'role', 'appearance', 'personality', 'description',
    'clothing', 'weapons', 'accessories', 'costumes', 'coreFeatures', 'style']) {
    const snakeKey = key.replace(/[A-Z]/g, m => '_' + m.toLowerCase())
    if (snakeKey in body) merged[key] = body[snakeKey]
    else if (key in body) merged[key] = body[key]
  }

  const dramaStyle = getEffectiveArtStyle(merged.style, getDramaStyle(char.dramaId))
  let prompt: string
  let negativePrompt: string
  if (type === 'character') {
    prompt = buildCharacterImagePrompt({ ...merged, dramaStyle })
    negativePrompt = buildCharacterNegativePrompt(dramaStyle)
  } else {
    prompt = buildEquipImagePrompt(type, { ...merged, dramaStyle })
    negativePrompt = buildEquipNegative(dramaStyle)
  }
  return success(c, { type, prompt, negativePrompt })
})

// POST /characters/:id/generate-three-views — 生成角色三视图（正面/侧面/背面合成一张横向长图）
app.post('/:id/generate-three-views', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const body = await c.req.json().catch(() => ({}))
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  if (!char) return badRequest(c, 'Character not found')
  const ep = body.episode_id
    ? db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()[0]
    : undefined
  if (body.episode_id && !ep) return badRequest(c, 'Episode not found')

  // 兼容旧参数：可指定视角子集，但始终合成一张横向三视图长图
  const views = (body.views && Array.isArray(body.views) ? body.views : ['front', 'side', 'back'])
    .filter((v: string) => ['front', 'side', 'back'].includes(v))
  if (views.length === 0) return badRequest(c, 'views 必须包含 front/side/back 之一')

  const dramaStyle = getEffectiveArtStyle(char.style, getDramaStyle(char.dramaId))
  const autoPrompt = buildCharacterImagePrompt({ ...char, ...body, dramaStyle })
  const visualsClause = buildCharacterVisualsClause({ ...char, ...body })
  const basePrompt = resolveCharacterPrompt(body.prompt || char.customPrompt, autoPrompt, dramaStyle, visualsClause)
  const negative = resolveCharacterNegative(body.negative_prompt || char.negativePrompt, dramaStyle)
  // 用户未提供负向词时，追加「组合图」排除词，防止三个视角被拆成多张图/分屏/带文字
  const threeViewNegative = body.negative_prompt || char.negativePrompt
    ? negative
    : buildThreeViewNegative(dramaStyle)
  const colorGrade = body.color_grade ?? body.colorGrade
  const configId = ep?.imageConfigId ?? resolveDramaConfigId(char.dramaId, 'imageConfigId')

  // 单张横向长图：正面在左、侧面在中、背面在右
  const combinedPrompt = `${basePrompt}, ${THREE_VIEW_COMBINED_LAYOUT}`
  const genId = await generateImage({
    characterId: id,
    dramaId: char.dramaId,
    prompt: combinedPrompt,
    negativePrompt: threeViewNegative,
    model: body.model,
    // 视觉锚定：以已生成的角色主立绘为参考，锁定五官/服装 → 三视角不漂移
    referenceImages: resolveCharacterAnchors(char, body, 'main'),
    size: body.size || THREE_VIEW_SIZE,
    viewType: 'combined',
    colorGrade,
    configId,
  })
  logTaskSuccess('CharacterImage', 'three-view-generate', { characterId: id, views })
  return success(c, { count: 1, results: [{ view: 'combined', image_generation_id: genId }] })
})

// POST /characters/:id/generate-equip-image — 生成服装/武器/首饰三视图设定图（单张横向长图，三个视角并排）
app.post('/:id/generate-equip-image', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const body = await c.req.json().catch(() => ({}))
  const type = body.type
  if (!['clothing', 'weapon', 'accessory'].includes(type)) {
    return badRequest(c, 'type 必须是 clothing/weapon/accessory 之一')
  }
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  if (!char) return badRequest(c, 'Character not found')
  const ep = body.episode_id
    ? db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()[0]
    : undefined
  if (body.episode_id && !ep) return badRequest(c, 'Episode not found')

  const dramaStyle = getEffectiveArtStyle(char.style, getDramaStyle(char.dramaId))
  // 模式：view（默认，三视角设定图，设计稿用）/ single（单件高清道具图，纯物品无人物，入库/分镜用）
  const singleMode = body.mode === 'single' || body.item_mode === true || body.single === true
  // 各对象类型独立提示词：服装/武器/首饰分别读取各自的 prompt 字段（角色立绘仍走 customPrompt）
  const promptField = type === 'clothing' ? 'clothingPrompt' : type === 'weapon' ? 'weaponPrompt' : 'accessoryPrompt'
  const negField = type === 'clothing' ? 'clothingNegativePrompt' : type === 'weapon' ? 'weaponNegativePrompt' : 'accessoryNegativePrompt'
  const userPrompt = body.prompt || (char as any)[promptField]
  const userNegative = body.negative_prompt || (char as any)[negField]
  // 表单最新值覆盖（未保存也能按当前填写内容生成）；未传字段保持库内值
  const mergedChar: any = { ...char }
  for (const k of ['clothing', 'weapons', 'accessories', 'costumes', 'appearance', 'coreFeatures']) {
    if (body[k] !== undefined) mergedChar[k] = body[k]
  }
  mergedChar.dramaStyle = dramaStyle
  const autoPrompt = singleMode
    ? buildItemImagePrompt(type, mergedChar)
    : buildEquipImagePrompt(type, mergedChar)
  const prompt = resolveEquipPrompt(userPrompt, autoPrompt, dramaStyle)
  // 用户未提供负向词 → 用装备三视图负向（排除人物/手持/边框但不抑制三视角并排）；
  // 用户提供负向词 → 若为旧版通用负向（不含人物排除），也强制补人物/手持排除，防止装备图出人
  const negative = singleMode
    ? userNegative
      ? resolveCharacterNegative(userNegative, dramaStyle)
      : buildItemNegative(dramaStyle)
    : userNegative
      ? isLegacyEquipPrompt(userNegative) || !/person|human|hand/i.test(userNegative)
        ? buildEquipNegative(dramaStyle)
        : resolveCharacterNegative(userNegative, dramaStyle)
      : buildEquipNegative(dramaStyle)

  try {
    logTaskStart('CharacterImage', 'equip-generate', { characterId: id, type, mode: singleMode ? 'single' : 'view' })
    const genId = await generateImage({
      characterId: id,
      dramaId: char.dramaId,
      prompt,
      negativePrompt: negative,
      model: body.model,
      // 视觉锚定（仅三视图设定图）：以「角色三视图 combined（无则主立绘）」为参考，保证穿的是同一个角色；
      // 单件道具图为纯物品生成不做自动锚定（避免把角色带进画面），需要时可显式传 reference_images
      referenceImages: singleMode ? undefined : resolveCharacterAnchors(char, body),
      // 单件道具图方形高清 1:1；三视图设定图沿用横向长图同比例（2048x896）保证三视角横向排开不挤压
      size: body.size || (singleMode ? ITEM_IMAGE_SIZE : THREE_VIEW_SIZE),
      equipType: type,
      itemType: singleMode ? type : undefined,
      colorGrade: body.color_grade ?? body.colorGrade,
      configId: ep?.imageConfigId ?? resolveDramaConfigId(char.dramaId, 'imageConfigId'),
    })
    logTaskSuccess('CharacterImage', 'equip-generate', { characterId: id, type, generationId: genId })
    return success(c, { image_generation_id: genId })
  } catch (err: any) {
    logTaskError('CharacterImage', 'equip-generate', { characterId: id, type, error: err.message })
    return badRequest(c, err.message)
  }
})

// POST /characters/:id/generate-expressions — 批量生成角色表情头像特写组
// body.keys?: string[]（默认全部 9 个表情；传单个 key 实现单图重生成）
// 每张表情独立异步生成，完成后写回 characters.expressions[key]，用于分镜/表情演出。
app.post('/:id/generate-expressions', async (c) => {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid character id')
  const body = await c.req.json().catch(() => ({}))
  const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
  if (!char) return badRequest(c, 'Character not found')
  const ep = body.episode_id
    ? db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()[0]
    : undefined
  if (body.episode_id && !ep) return badRequest(c, 'Episode not found')

  const keys = (Array.isArray(body.keys) && body.keys.length ? body.keys : EXPRESSION_PRESETS.map(p => p.key))
    .filter((k: string) => findExpressionPreset(k))
  if (keys.length === 0) return badRequest(c, 'keys 中没有合法的表情 key')

  const dramaStyle = getEffectiveArtStyle(char.style, getDramaStyle(char.dramaId))
  // 表单最新值覆盖（未保存也能按当前外貌/服装生成表情），未传字段保持库内值
  const mergedChar: any = { ...char }
  for (const k of ['appearance', 'clothing', 'costumes', 'coreFeatures', 'description']) {
    if (body[k] !== undefined) mergedChar[k] = body[k]
  }
  // 视觉锚定：表情头像组以「角色三视图 combined（无则主立绘）」为参考，同一批次/跨批保持一致脸型
  const anchors = resolveCharacterAnchors(char, body)
  const results: Array<{ key: string; label: string; image_generation_id: number }> = []
  for (const key of keys) {
    const preset = findExpressionPreset(key)!
    try {
      logTaskStart('CharacterImage', 'expression-generate', { characterId: id, expression: key })
      const prompt = buildExpressionImagePrompt(mergedChar, key, dramaStyle)
      const negative = body.negative_prompt
        ? resolveCharacterNegative(body.negative_prompt, dramaStyle)
        : buildExpressionNegative(dramaStyle)
      const genId = await generateImage({
        characterId: id,
        dramaId: char.dramaId,
        prompt,
        negativePrompt: negative,
        model: body.model,
        referenceImages: anchors,
        size: body.size || '1024x1024',
        expression: key,
        colorGrade: body.color_grade ?? body.colorGrade,
        configId: ep?.imageConfigId ?? resolveDramaConfigId(char.dramaId, 'imageConfigId'),
      })
      results.push({ key, label: preset.label, image_generation_id: genId })
    } catch (err: any) {
      logTaskError('CharacterImage', 'expression-generate', { characterId: id, expression: key, error: err.message })
    }
  }
  logTaskSuccess('CharacterImage', 'expressions-generate', { characterId: id, requested: keys.length, started: results.length })
  return success(c, { count: results.length, results })
})

// POST /characters/batch-generate-images
app.post('/batch-generate-images', async (c) => {
  const body = await c.req.json()
  const ids: number[] = body.character_ids || []
  if (!body.episode_id) return badRequest(c, 'episode_id is required')
  const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, Number(body.episode_id))).all()
  if (!ep) return badRequest(c, 'Episode not found')
  const results: number[] = []
  const dramaBaseStyle = getDramaStyle(ep.dramaId)
  for (const cid of ids) {
    const [char] = db.select().from(schema.characters).where(and(eq(schema.characters.id, cid), isNull(schema.characters.deletedAt))).all()
    if (!char) continue
    const dramaStyle = getEffectiveArtStyle(char.style, dramaBaseStyle)
    const prompt = buildCharacterImagePrompt({ ...char, dramaStyle })
    const negative = resolveCharacterNegative(char.negativePrompt, dramaStyle)
    try {
      const genId = await generateImage({
        characterId: cid,
        dramaId: char.dramaId,
        prompt,
        negativePrompt: resolveCharacterNegative(body.negative_prompt || char.negativePrompt, dramaStyle),
        colorGrade: body.color_grade ?? body.colorGrade,
        configId: ep.imageConfigId ?? undefined,
      })
      results.push(genId)
    } catch {}
  }
  logTaskSuccess('CharacterImage', 'batch-generate', { episodeId: ep.id, requested: ids.length, started: results.length })
  return success(c, { count: results.length, ids: results })
})

export default app
