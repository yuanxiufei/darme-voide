/**
 * 统一视觉 Prompt 工具
 * 
 * 解决全项目 prompt 风格碎片化问题：
 * - 角色图片：cinematic illustration style
 * - 场景图片：高质量场景, 电影感
 * - 网格图片：cinematic lighting, high quality
 * - 视频 prompt：无固定风格
 * 
 * 所有生成路径统一使用本模块的风格预设和 prompt 构建器
 */

import { db, schema } from '../db/index.js'
import { and, eq, inArray, isNull } from 'drizzle-orm'
import { getCameraMovementComposition } from './camera-movement-guides.js'

// ============================================================
// 风格预设常量（全域一致）
// ============================================================

/**
 * 主视觉风格标签 - 所有角色/场景/分镜图像生成共用
 */
export const VISUAL_STYLE_MASTER = 'cinematic illustration style, consistent art style, soft cinematic lighting, high quality, no text, no watermark'

/**
 * 角色立绘专用风格标签
 */
export const VISUAL_STYLE_CHARACTER = 'stylized character design, front-facing character portrait, clean white background, illustration style, not photorealistic'

/**
 * 场景背景专用风格标签
 */
export const VISUAL_STYLE_SCENE = 'highly detailed cinematic environment, atmospheric lighting, movie quality composition, depth of field'

/**
 * 视频生成风格引导前缀
 */
export const VISUAL_STYLE_VIDEO = 'cinematic motion, smooth camera movement, consistent character design, lighting continuity'

// ============================================================
// 通用预设类型定义（Preset Framework Types）
// ============================================================

/** Variation Card 的单个 Shot 配置 */
export interface PresetVariationCardShot {
  shotIndex: number
  themeFamily: string
  compositionPattern: string
  spaceType: string
  foregroundFrame: string
  mainFocalPoint: string
  thematicClue: string
  activity: string
  cameraPosition: string
  windDirection: string
  lightStructure: string
  characterLayout: string
  livingElement: string | null
}

/** Variation Card（5 镜配置） */
export interface PresetVariationCard {
  themeFamily: string
  shots: PresetVariationCardShot[]
}

// ============================================================
// 统一负面提示词（Negative Prompt）体系
// 每一条正向提示词都配套对应的反向提示词，明确排除不需要的内容/风格/元素。
// 按生成类型拆分为针对性预设，各调用方在构建正向 prompt 时成对引用。
// ============================================================

/**
 * 通用负面基础（所有生成类型共用）
 * 排除低质、常见瑕疵与文字/水印等通用干扰项
 */
export const NEGATIVE_BASE = 'text, watermark, signature, logo, subtitles, low quality, blurry, out of focus, pixelated, jpeg artifacts, oversaturated, distorted, deformed, disfigured, bad anatomy, extra limbs, extra fingers, mutated hands, bad proportions, duplicated elements'

/**
 * 角色立绘负面提示词
 * 排除写实照片质感（角色走 stylized 插画风）、杂乱/多人背景、角色重复与裁切
 */
export const CHARACTER_IMAGE_NEGATIVE = `${NEGATIVE_BASE}, photorealistic, realistic photo, 3d render, cluttered background, busy background, multiple characters, multiple people, duplicated character, inconsistent character, cropped head, cut off face`

/**
 * 场景背景负面提示词
 * 排除人物角色（场景图只保留环境）、特写人脸与空洞扁平构图
 */
export const SCENE_IMAGE_NEGATIVE = `${NEGATIVE_BASE}, people, person, human figure, characters, portrait, close-up face, face, flat composition, empty boring layout`

/**
 * 分镜图片负面提示词
 * 排除角色不一致、多余人物与时代/现代元素穿帮
 */
export const STORYBOARD_IMAGE_NEGATIVE = `${NEGATIVE_BASE}, inconsistent character, character mismatch, extra characters, wrong number of people, anachronism, modern objects, modern clothing`

/**
 * 视频负面提示词（分镜视频 / 预设视频 / 手动视频共用）
 * 排除运动瑕疵、闪烁、形变、角色漂移与镜头抖动
 */
export const VIDEO_NEGATIVE = `${NEGATIVE_BASE}, motion blur, jittery, flickering, flicker, warping, morphing, melting, distorted face, inconsistent character, character drift, frame inconsistency, jump cuts, camera shake, static image, frozen frame`

/** 预设图片负面提示词（兼容旧引用，基于统一负面基础） */
export const PRESET_IMAGE_NEGATIVE = `${NEGATIVE_BASE}, mutated body parts`

/** 预设视频负面提示词 */
export const PRESET_VIDEO_NEGATIVE = VIDEO_NEGATIVE

/**
 * 根据 Shot 配置构建预设图片生成 prompt
 */
export function buildPresetImagePrompt(shot: PresetVariationCardShot): string {
  const parts = [
    shot.themeFamily,
    shot.compositionPattern,
    shot.spaceType,
    shot.foregroundFrame,
    shot.mainFocalPoint,
    shot.thematicClue,
    shot.activity,
    shot.characterLayout,
    shot.lightStructure,
    shot.windDirection,
    shot.cameraPosition,
  ]
  if (shot.livingElement) parts.push(shot.livingElement)
  parts.push(VISUAL_STYLE_MASTER)
  return parts.join(', ')
}

/**
 * 根据 Shot 配置 + 运镜方式构建预设视频生成 prompt
 */
export function buildPresetVideoPrompt(shot: PresetVariationCardShot, cameraMove: string): string {
  const parts = [
    shot.mainFocalPoint,
    shot.activity,
    shot.compositionPattern,
    shot.spaceType,
    shot.characterLayout,
    cameraMove,
    shot.lightStructure,
    shot.windDirection,
  ]
  parts.push(VISUAL_STYLE_VIDEO, VISUAL_STYLE_MASTER)
  return parts.join(', ')
}

// ============================================================
// 角色视觉 Prompt 构建
// ============================================================

/** 解析角色核心特征标签（JSON 数组字符串 → 字符串数组） */
function parseCoreFeatures(coreFeatures?: string | null): string[] {
  if (!coreFeatures) return []
  try {
    const arr = JSON.parse(coreFeatures)
    if (!Array.isArray(arr)) return []
    return arr.filter((x): x is string => typeof x === 'string' && !!x.trim()).map(x => x.trim())
  } catch {
    return []
  }
}

/** 解析角色多套服装变体（JSON 数组字符串 → 字符串数组） */
function parseCostumes(costumes?: string | null): string[] {
  if (!costumes) return []
  try {
    const arr = JSON.parse(costumes)
    if (!Array.isArray(arr)) return []
    return arr.filter((x): x is string => typeof x === 'string' && !!x.trim()).map(x => x.trim())
  } catch {
    return []
  }
}

/** 归一化装备/首饰文本：JSON 数组 → 逗号拼接；纯文本 → 去引号原样返回 */
function formatEquip(text?: string | null): string {
  if (!text) return ''
  const arr = parseCoreFeatures(text)
  if (arr.length) return arr.join(', ')
  return text.replace(/[\[\]"']/g, '').replace(/\s+/g, ' ').trim()
}

/** 解析角色多变体立绘（JSON 数组字符串 → {name, imageUrl}[]） */
function parseVariations(variations?: string | null): Array<{ name: string; imageUrl: string | null }> {
  if (!variations) return []
  try {
    const arr = JSON.parse(variations)
    if (!Array.isArray(arr)) return []
    return arr.filter((x): x is { name: string; imageUrl: string | null } =>
      x && typeof x === 'object' && typeof x.name === 'string')
  } catch {
    return []
  }
}

/**
 * 构建角色图片生成 prompt
 * @param name 角色名
 * @param appearance 外貌描述
 * @param description 补充描述
 * @param personality 性格特征（可选，用于表情氛围）
 * @returns 完整的角色图片 prompt
 */
export function buildCharacterImagePrompt(char: {
  name: string
  appearance?: string | null
  description?: string | null
  personality?: string | null
  coreFeatures?: string | null
  clothing?: string | null
  costume?: string | null
  costumes?: string | null
  weapons?: string | null
  accessories?: string | null
}): string {
  const parts: string[] = [char.name]
  const core = parseCoreFeatures(char.coreFeatures)
  if (core.length) parts.push(`core features: ${core.join(', ')}`)
  if (char.appearance) parts.push(char.appearance)
  // 服装：优先本次选中的 costume，其次单套 clothing，最后回退多套 costumes 首套
  const costume = char.costume || char.clothing || parseCostumes(char.costumes)[0]
  if (costume) parts.push(`wearing ${costume}`)
  const weapons = formatEquip(char.weapons)
  if (weapons) parts.push(`armed with ${weapons}`)
  const accessories = formatEquip(char.accessories)
  if (accessories) parts.push(`wearing accessories: ${accessories}`)
  if (char.description && char.description !== char.appearance) parts.push(char.description)

  // 性格特征影响表情氛围
  if (char.personality) {
    parts.push(`${char.personality} expression and mannerisms`)
  }

  return `${parts.join(', ')}, ${VISUAL_STYLE_CHARACTER}, ${VISUAL_STYLE_MASTER}`
}

/**
 * 构建角色外观描述文本（用于分镜/视频 prompt 中引用）
 * 提取角色核心视觉特征，不含风格标签
 */
export function buildCharacterAppearanceText(char: {
  name: string
  appearance?: string | null
  description?: string | null
  coreFeatures?: string | null
  clothing?: string | null
  costume?: string | null
  costumes?: string | null
  weapons?: string | null
  accessories?: string | null
}): string {
  const parts: string[] = [char.name]
  const core = parseCoreFeatures(char.coreFeatures)
  if (core.length) parts.push(core.join(', '))
  if (char.appearance) parts.push(char.appearance)
  const costume = char.costume || char.clothing || parseCostumes(char.costumes)[0]
  if (costume) parts.push(`wearing ${costume}`)
  const weapons = formatEquip(char.weapons)
  if (weapons) parts.push(`armed with ${weapons}`)
  const accessories = formatEquip(char.accessories)
  if (accessories) parts.push(`accessories: ${accessories}`)
  if (char.description && char.description !== char.appearance) parts.push(char.description)
  return parts.join(': ')
}

/**
 * 构建角色「装备/服饰特写图」prompt（服装/武器/首饰独立设定图）
 * @param type clothing/weapon/accessory
 */
export function buildEquipImagePrompt(
  type: 'clothing' | 'weapon' | 'accessory',
  char: {
    name: string
    appearance?: string | null
    coreFeatures?: string | null
    clothing?: string | null
    weapons?: string | null
    accessories?: string | null
  },
): string {
  const parts: string[] = []
  if (type === 'clothing') {
    const c = formatEquip(char.clothing)
    parts.push(c ? `detailed costume design sheet of ${c}` : 'detailed costume design sheet')
  } else if (type === 'weapon') {
    const w = formatEquip(char.weapons)
    parts.push(w ? `detailed weapon concept art of ${w}` : 'detailed weapon concept art')
  } else {
    const a = formatEquip(char.accessories)
    parts.push(a ? `detailed accessory jewelry design of ${a}` : 'detailed accessory jewelry design')
  }
  const core = parseCoreFeatures(char.coreFeatures)
  if (core.length) parts.push(`matching the character's core features: ${core.join(', ')}`)
  if (char.appearance) parts.push(`character appearance: ${char.appearance}`)
  parts.push(`for the character ${char.name}`)
  return `${parts.join(', ')}, ${VISUAL_STYLE_CHARACTER}, ${VISUAL_STYLE_MASTER}`
}

// ============================================================
// 场景视觉 Prompt 构建
// ============================================================

/**
 * 构建场景图片生成 prompt
 * @param location 地点名称
 * @param time 时间（白天/夜晚等）
 * @param prompt 场景描述
 * @param dramaStyle 戏剧风格标签（来自 dramas.style）
 */
export function buildSceneImagePrompt(options: {
  location: string
  time?: string
  prompt?: string
  dramaStyle?: string
}): string {
  const parts: string[] = []
  if (options.prompt) parts.push(options.prompt)
  else {
    parts.push(options.location)
    if (options.time) parts.push(`${options.time} lighting and atmosphere`)
  }
  if (options.dramaStyle) parts.push(`${options.dramaStyle} visual style`)

  return `${parts.join(', ')}, ${VISUAL_STYLE_SCENE}, ${VISUAL_STYLE_MASTER}`
}

// ============================================================
// 分镜视觉 Prompt 构建
// ============================================================

/**
 * 构建分镜叙事图片 prompt（融合角色+场景+镜头描述）
 * 
 * 核心修复：之前分镜图片生成只传 description，没有角色外观和场景信息，
 * 导致角色在不同分镜中视觉不一致
 */
export function buildStoryboardImagePrompt(options: {
  description?: string | null         // 分镜叙事描述
  storyboardDescription?: string | null  // 分镜视觉描述
  characterDescription?: string | null   // 角色描述
  sceneDescription?: string | null       // 场景描述
  location?: string | null               // 地点
  shotType?: string | null               // 景别
  cameraAngle?: string | null            // 机位
  dramaStyle?: string                    // 戏剧风格
}): string {
  const parts: string[] = []

  // 1. 场景环境
  if (options.sceneDescription) {
    parts.push(`Scene: ${options.sceneDescription}`)
  } else if (options.location) {
    parts.push(`Location: ${options.location}`)
  }

  // 2. 角色外观
  if (options.characterDescription) {
    parts.push(`Characters: ${options.characterDescription}`)
  }

  // 3. 镜头描述
  const cinematicHints: string[] = []
  if (options.shotType) cinematicHints.push(options.shotType)
  if (options.cameraAngle) cinematicHints.push(options.cameraAngle)
  if (cinematicHints.length) parts.push(`Camera: ${cinematicHints.join(', ')}`)

  // 4. 叙事内容
  if (options.storyboardDescription) {
    parts.push(options.storyboardDescription)
  } else if (options.description) {
    parts.push(options.description)
  }

  // 5. 风格
  if (options.dramaStyle) parts.push(`${options.dramaStyle} visual style`)

  // 底线
  if (parts.length === 0) return `cinematic shot, ${VISUAL_STYLE_MASTER}`

  return `${parts.join('. ')}, ${VISUAL_STYLE_MASTER}`
}

/**
 * 构建分镜视频生成 prompt（融合角色+场景+叙事+动作）
 * 
 * 核心修复：之前视频生成只用 storyboardDescription，完全没有角色外观上下文，
 * 导致视频中角色形象突变
 */
export function buildStoryboardVideoPrompt(options: {
  description?: string | null
  storyboardDescription?: string | null
  characterAppearances?: string[]          // 各角色外观描述列表
  scenePrompt?: string | null              // 场景视觉描述
  action?: string | null                   // 动作描述
  movement?: string | null                 // 运镜
  dramaStyle?: string                      // 戏剧风格
}): string {
  const parts: string[] = []

  // 1. 角色一致性 - 最关键的部分
  if (options.characterAppearances?.length) {
    parts.push(`Characters (maintain strict visual consistency): ${options.characterAppearances.join('; ')}`)
  }

  // 2. 场景
  if (options.scenePrompt) {
    parts.push(`Setting: ${options.scenePrompt}`)
  }

  // 3. 叙事
  if (options.storyboardDescription) {
    parts.push(options.storyboardDescription)
  } else if (options.description) {
    parts.push(options.description)
  }

  // 4. 动作与运镜
  if (options.action) parts.push(`Action: ${options.action}`)
  if (options.movement) parts.push(`Camera movement: ${options.movement}`)

  // 5. 风格
  if (options.dramaStyle) parts.push(`${options.dramaStyle} cinematic style`)

  return `${parts.join('. ')}, ${VISUAL_STYLE_VIDEO}, ${VISUAL_STYLE_MASTER}`
}

// ============================================================
// 数据库查询辅助
// ============================================================

/**
 * 根据分镜ID获取关联的角色外表描述列表
 */
export function getStoryboardCharacterAppearances(storyboardId: number): string[] {
  const spChars = db.select().from(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .all()

  if (!spChars.length) return []

  // 镜头级服装变体：每个角色可指定本镜头穿的造型
  const costumeByCharId = new Map<number, string>()
  for (const link of spChars) {
    if (link.costume) costumeByCharId.set(link.characterId, link.costume)
  }

  const characterIds = spChars.map(link => link.characterId)
  // 使用 inArray 替代内存过滤
  const characters = characterIds.length > 0
    ? db.select().from(schema.characters)
        .where(and(inArray(schema.characters.id, characterIds), isNull(schema.characters.deletedAt)))
        .all()
    : []

  return characters.map(char =>
    buildCharacterAppearanceText({
      name: char.name,
      appearance: char.appearance,
      description: char.description,
      coreFeatures: char.coreFeatures,
      clothing: char.clothing,
      costume: costumeByCharId.get(char.id),
      costumes: char.costumes,
    })
  )
}

/**
 * 根据分镜ID获取关联的场景描述
 */
export function getStoryboardSceneDescription(storyboardId: number): string | null {
  const [sb] = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.id, storyboardId))
    .all()

  if (!sb?.sceneId) return null

  const [scene] = db.select().from(schema.scenes)
    .where(eq(schema.scenes.id, sb.sceneId))
    .all()

  if (!scene) return null

  return buildSceneImagePrompt({
    location: scene.location,
    time: scene.time,
    prompt: scene.prompt,
  })
}

/**
 * 根据角色ID列表获取所有角色的图片URL（用于视频/图片 reference_images）
 */
export function getCharacterImageUrls(characterIds: number[]): string[] {
  if (!characterIds.length) return []
  const characters = db.select().from(schema.characters)
    .where(and(inArray(schema.characters.id, characterIds), isNull(schema.characters.deletedAt)))
    .all()
    .filter(char => char.imageUrl)

  return characters.map(char => char.imageUrl!)
}

/**
 * 根据分镜ID获取关联角色的图片URL
 */
export function getStoryboardCharacterImageUrls(storyboardId: number): string[] {
  const spChars = db.select().from(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .all()
  if (!spChars.length) return []

  // 镜头级服装变体：若角色指定了变体且有对应立绘，用变体图；否则主图
  const costumeByCharId = new Map<number, string>()
  for (const link of spChars) {
    if (link.costume) costumeByCharId.set(link.characterId, link.costume)
  }

  const characterIds = spChars.map(link => link.characterId)
  const characters = db.select().from(schema.characters)
    .where(and(inArray(schema.characters.id, characterIds), isNull(schema.characters.deletedAt)))
    .all()

  const urls: string[] = []
  for (const char of characters) {
    const costume = costumeByCharId.get(char.id)
    if (costume) {
      const variation = parseVariations(char.variations).find(v => v.name === costume)
      if (variation?.imageUrl) {
        urls.push(variation.imageUrl)
        continue
      }
    }
    if (char.imageUrl) urls.push(char.imageUrl)
  }
  return urls
}

/**
 * 根据分镜ID获取参考图（角色立绘 + 场景图），用于分镜图片/视频生成保持人物与场景一致
 */
export function getStoryboardReferenceImages(storyboardId: number): string[] {
  const urls = getStoryboardCharacterImageUrls(storyboardId)

  const [sb] = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.id, storyboardId))
    .all()
  if (sb?.sceneId) {
    const [scene] = db.select().from(schema.scenes)
      .where(eq(schema.scenes.id, sb.sceneId))
      .all()
    if (scene?.imageUrl) urls.push(scene.imageUrl)
  }

  return urls
}

/**
 * 将媒体相对路径转成可被本地/远程服务访问的绝对 URL（本地 H3 服务需能拉取参考音频）
 */
function toPublicMediaUrl(p: string): string {
  if (/^(https?:|data:)/.test(p)) return p
  const base = (process.env.PUBLIC_BASE_URL || 'http://localhost:5789').replace(/\/+$/, '')
  return `${base}/${p.replace(/^\/+/, '')}`
}

/**
 * 收集分镜出场角色的声线样本音频 URL（H3 Ref2VA 参考音频 / reference conditioning，最多 3 条）
 */
export function getStoryboardReferenceAudioUrls(storyboardId: number): string[] {
  const links = db.select().from(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .all()
  if (!links.length) return []

  const characterIds = links.map(l => l.characterId)
  const characters = db.select().from(schema.characters)
    .where(and(inArray(schema.characters.id, characterIds), isNull(schema.characters.deletedAt)))
    .all()

  return characters
    .map(c => c.voiceSampleUrl)
    .filter((u): u is string => !!u && !!u.trim())
    .map(toPublicMediaUrl)
    .slice(0, 3)
}

// ============================================================
// 对话-角色一致性验证
// ============================================================

/**
 * 验证分镜 dialogue 中的角色名是否与分镜关联的 character_ids 匹配
 * @param dialogue 对话文本
 * @param storyboardId 分镜ID
 * @returns { mismatches: 不匹配的角色名列表, matchCount: 匹配的角色名数量 }
 */
export function validateDialogueCharacterConsistency(
  dialogue: string,
  storyboardId: number,
): { mismatches: string[]; matchCount: number; allMatch: boolean } {
  if (!dialogue) return { mismatches: [], matchCount: 0, allMatch: true }

  // 提取对话中的角色名
  const regex = /([^\n:：]{1,10}?)[:：]([^:：\n]{8,})/g
  const speakerNames = new Set<string>()
  let match
  while ((match = regex.exec(dialogue)) !== null) {
    speakerNames.add(match[1].trim())
  }

  // 如果正则没匹配到，尝试用 parseDialogueForTTS 逻辑
  if (speakerNames.size === 0) {
    const parsed = parseDialogueForTTSLocal(dialogue)
    if (parsed.speaker && !parsed.ignorable) {
      speakerNames.add(parsed.speaker)
    }
  }

  if (speakerNames.size === 0) return { mismatches: [], matchCount: 0, allMatch: true }

  // 获取分镜关联的角色（使用 WHERE 查询替代全表扫描）
  const spChars = db.select().from(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .all()

  const characterIds = spChars.map(link => link.characterId)
  const characters = characterIds.length > 0
    ? db.select().from(schema.characters)
        .where(and(inArray(schema.characters.id, characterIds), isNull(schema.characters.deletedAt)))
        .all()
    : []

  const charNames = new Set(characters.map(c => c.name))

  const mismatches: string[] = []
  let matchCount = 0

  for (const name of speakerNames) {
    // 旁白/画外音不校验
    if (/^(旁白|画外音|narrator)$/i.test(name)) continue
    if (charNames.has(name)) {
      matchCount++
    } else {
      mismatches.push(name)
    }
  }

  return { mismatches, matchCount, allMatch: mismatches.length === 0 }
}

/**
 * parseDialogueForTTS 本地副本，避免循环依赖
 */
function parseDialogueForTTSLocal(dialogue: string): { ignorable: boolean; speaker: string; pureText: string } {
  const text = (dialogue || '').trim()
  if (!text) return { ignorable: true, speaker: '', pureText: '' }

  const labelMatch = text.match(/^(.*?)[:：]\s*(.+)/)
  if (labelMatch) {
    return {
      ignorable: false,
      speaker: labelMatch[1].trim(),
      pureText: labelMatch[2].trim(),
    }
  }
  return { ignorable: false, speaker: '', pureText: text }
}

// ============================================================
// 视频提示词标签剥离
// ============================================================

/**
 * 剥离视频提示词中的结构化标记标签，只保留标签内的自然语言内容。
 *
 * 背景：分镜 agent 生成的 video_prompt 使用 <location>/<role>/<voice>/<n>
 * 作为结构化 DSL，供程序解析（分段、角色绑定、场景提取）使用。
 * 但视频扩散模型不识别这些 XML 标签，原样保留只会成为文本噪声，
 * 占用注意力并可能干扰语义理解。因此在发送给视频生成模型前应剥离。
 *
 * 规则：
 * - <location>…</location> / <role>…</role> / <voice>…</voice> → 去掉开闭标签，保留中间内容
 * - <n>（时间段分隔符）→ 换成换行，让每个时间段独立成段
 * - 清理多余空白
 */
export function stripVideoPromptTags(prompt: string): string {
  if (!prompt) return prompt
  return prompt
    .replace(/<\/?(?:location|role|voice)>/gi, '')
    .replace(/<n\s*\/?>/gi, '\n')
    .replace(/[ \t]+/g, ' ')
    .replace(/\s*\n\s*/g, '\n')
    .trim()
}

// ============================================================
// 宫格图 Prompt 构建（唯一入口）
//
// 宫格图 prompt 曾有三处重复实现（grid 路由内嵌模板、grid-prompt-tools、
// storyboard-tools），现统一收敛到本模块。grid 路由与 grid_prompt_generator
// agent 工具均调用 buildGridPrompt / buildGridCellPrompts，避免逻辑分叉。
// ============================================================

export type GridReferenceAsset = {
  path: string
  label: string
  kind: 'scene' | 'character' | 'storyboard'
  imageLabel: string
  sceneId?: number
  characterId?: number
  storyboardId?: number
}

function safeParseJsonArray(value: any): string[] {
  if (!value) return []
  try {
    const parsed = JSON.parse(value)
    return Array.isArray(parsed) ? parsed.filter(Boolean) : []
  } catch {
    return []
  }
}

function posLabel(i: number, rows: number, cols: number) {
  const r = Math.floor(i / cols), c = i % cols
  return `row ${r + 1} col ${c + 1}`
}

function cellLabel(i: number, rows: number, cols: number) {
  return `格${i + 1}（${posLabel(i, rows, cols)}）`
}

/** 获取每个分镜关联的角色 ID 列表（用于注入角色外观与参考图） */
function getStoryboardCharacterIds(storyboardIds: number[]) {
  if (!storyboardIds.length) return new Map<number, number[]>()
  const links = db.select().from(schema.storyboardCharacters).all()
    .filter((link) => storyboardIds.includes(link.storyboardId))
  const map = new Map<number, number[]>()
  for (const link of links) {
    const arr = map.get(link.storyboardId) || []
    arr.push(link.characterId)
    map.set(link.storyboardId, arr)
  }
  return map
}

/** 构建角色 ID → 外观描述文本的映射，用于将角色视觉特征注入分镜描述 */
function buildCharacterAppearanceMap(storyboardCharacterIds: Map<number, number[]>) {
  const allCharIds = new Set<number>()
  for (const ids of storyboardCharacterIds.values()) ids.forEach(id => allCharIds.add(id))
  if (!allCharIds.size) return new Map<number, string>()

  const chars = db.select().from(schema.characters)
    .where(and(inArray(schema.characters.id, [...allCharIds]), isNull(schema.characters.deletedAt)))
    .all()

  const map = new Map<number, string>()
  for (const char of chars) {
    map.set(char.id, buildCharacterAppearanceText(char))
  }
  return map
}

/** 构建分镜关联的参考图提示（角色立绘 + 场景图 + 已有分镜图） */
function buildStoryboardReferenceHints(
  sb: any,
  referenceAssets: GridReferenceAsset[],
  storyboardCharacterIds: Map<number, number[]>,
) {
  const hints: string[] = []
  const charIds = storyboardCharacterIds.get(sb.id) || []

  for (const asset of referenceAssets) {
    if (asset.kind === 'scene' && sb.sceneId && asset.sceneId === sb.sceneId) {
      hints.push(`${asset.imageLabel}（${asset.label}）`)
    }
    if (asset.kind === 'character') {
      if (asset.characterId && charIds.includes(asset.characterId)) {
        hints.push(`${asset.imageLabel}（${asset.label}）`)
      }
    }
    if (asset.kind === 'storyboard' && asset.storyboardId === sb.id) {
      hints.push(`${asset.imageLabel}（${asset.label}）`)
    }
  }

  return [...new Set(hints)].slice(0, 4)
}

/** 构建增强的分镜 cell 描述：注入角色外观 + 参考图 + 分镜自身描述 */
function buildEnrichedCellDescription(
  sb: any,
  index: number,
  storyboardCharacterIds: Map<number, number[]>,
  charAppearanceMap: Map<number, string>,
  referenceAssets: GridReferenceAsset[],
): string {
  const parts: string[] = []

  const charIds = storyboardCharacterIds.get(sb.id) || []
  const appearances: string[] = []
  for (const cid of charIds) {
    const appearance = charAppearanceMap.get(cid)
    if (appearance) appearances.push(appearance)
  }
  if (appearances.length) {
    parts.push(`Characters: ${appearances.join('; ')}`)
  }

  const refs = buildStoryboardReferenceHints(sb, referenceAssets, storyboardCharacterIds)
  if (refs.length) {
    parts.push(`参考${refs.join('、')}`)
  }

  const desc = sb.imagePrompt || sb.description || sb.title || `shot ${index + 1}`
  parts.push(desc)

  return parts.join('. ')
}

/** 收集宫格图涉及的参考图（首尾帧/镜头图/参考图 + 场景图 + 角色立绘，上限 6 张） */
export function collectGridReferenceAssets(storyboards: any[]): GridReferenceAsset[] {
  const storyboardIds = storyboards.map((sb) => sb.id)
  const storyboardCharacterIds = getStoryboardCharacterIds(storyboardIds)
  const sceneIds = [...new Set(storyboards.map((sb) => sb.sceneId).filter(Boolean))]
  const characterIds = [...new Set([...storyboardCharacterIds.values()].flat().filter(Boolean))]

  const scenes = sceneIds.length > 0
    ? db.select().from(schema.scenes).where(inArray(schema.scenes.id, sceneIds)).all()
    : []
  const characters = characterIds.length > 0
    ? db.select().from(schema.characters).where(and(inArray(schema.characters.id, characterIds), isNull(schema.characters.deletedAt))).all()
    : []

  const assets: GridReferenceAsset[] = []
  const seen = new Set<string>()
  const pushAsset = (
    path: string | null | undefined,
    label: string,
    kind: 'scene' | 'character' | 'storyboard',
    extra: { sceneId?: number; characterId?: number; storyboardId?: number } = {},
  ) => {
    if (!path || seen.has(path) || assets.length >= 6) return
    seen.add(path)
    assets.push({ path, label, kind, ...extra } as GridReferenceAsset)
  }

  for (const sb of storyboards) {
    pushAsset(sb.firstFrameImage, `镜头${sb.storyboardNumber}首帧`, 'storyboard', { storyboardId: sb.id })
    pushAsset(sb.lastFrameImage, `镜头${sb.storyboardNumber}尾帧`, 'storyboard', { storyboardId: sb.id })
    pushAsset(sb.composedImage, `镜头${sb.storyboardNumber}镜头图`, 'storyboard', { storyboardId: sb.id })
    for (const ref of safeParseJsonArray(sb.referenceImages)) {
      pushAsset(ref, `镜头${sb.storyboardNumber}参考图`, 'storyboard', { storyboardId: sb.id })
    }
  }
  for (const scene of scenes) {
    pushAsset(scene.imageUrl, `${scene.location}${scene.time ? `（${scene.time}）` : ''}场景`, 'scene', { sceneId: scene.id })
  }
  for (const char of characters) {
    pushAsset(char.imageUrl, `${char.name}角色`, 'character', { characterId: char.id })
  }

  return assets.map((asset, index) => ({
    ...asset,
    imageIndex: index + 1,
    imageLabel: `图片${index + 1}`,
  }))
}

export function buildReferenceLegend(referenceAssets: Array<{ imageLabel: string; label: string }>) {
  if (!referenceAssets.length) return ''
  return referenceAssets.map((asset) => `${asset.imageLabel}=${asset.label}`).join('；')
}

const GRID_ANGLES = [
  'wide establishing shot', 'medium shot character focus',
  'close-up detail', 'dramatic low angle', 'over-the-shoulder view',
  'bird eye view', 'side profile', 'atmospheric detail',
  'extreme close-up', 'dutch angle', 'silhouette shot',
  'depth of field focus', 'symmetrical composition', 'leading lines',
  'negative space', 'high angle looking down', 'ground level',
  'panoramic wide', 'intimate two-shot', 'reflection shot',
  'shadow play', 'backlit silhouette', 'macro detail',
  'split lighting', 'rim light portrait',
]

/**
 * 构建宫格图整体 prompt（三种模式：first_frame / first_last / multi_ref）
 * 统一注入：风格、参考图映射、角色外观、运镜构图。
 */
export function buildGridPrompt(
  mode: string,
  storyboards: any[],
  rows: number,
  cols: number,
  dramaStyle: string,
  referenceAssets: GridReferenceAsset[],
): string {
  const style = dramaStyle || 'cinematic'
  const legend = buildReferenceLegend(referenceAssets)
  const storyboardCharacterIds = getStoryboardCharacterIds(storyboards.map((sb) => sb.id))
  const charAppearanceMap = buildCharacterAppearanceMap(storyboardCharacterIds)

  if (mode === 'first_frame') {
    const cells = storyboards.map((sb, i) => {
      const desc = buildEnrichedCellDescription(sb, i, storyboardCharacterIds, charAppearanceMap, referenceAssets)
      return `${cellLabel(i, rows, cols)}: ${desc}`
    })
    return [
      `${rows}x${cols} grid layout, consistent art style, ${style},`,
      legend ? `参考图映射：${legend}` : '',
      '当画面涉及角色或场景时，优先使用对应的图片编号来约束一致性。',
      ...cells,
      'high quality, cinematic lighting, no text, no watermark',
    ].filter(Boolean).join('\n')
  }

  if (mode === 'first_last') {
    const totalCells = rows * cols
    const cells = Array.from({ length: totalCells }, (_, i) => {
      const sb = storyboards[i % storyboards.length]
      const desc = sb.imagePrompt || sb.description || sb.title || `shot ${i + 1}`
      const action = sb.action || sb.movement || ''
      const refs = buildStoryboardReferenceHints(sb, referenceAssets, storyboardCharacterIds)
      const isFirst = i % 2 === 0
      const composition = getCameraMovementComposition(sb.movement || '', isFirst ? 'start' : 'end')
      const comp = composition ? `, ${composition}` : ''
      const frameHint = isFirst
        ? 'opening moment'
        : `${action ? `${action}, ` : ''}closing moment, subtle motion change`
      return `${cellLabel(i, rows, cols)}: ${refs.length ? `参考${refs.join('、')}，` : ''}${desc}, ${frameHint}${comp}`
    })
    return [
      `${rows}x${cols} grid layout, consistent art style, ${style},`,
      legend ? `参考图映射：${legend}` : '',
      'first/last frame visual rhythm, alternating opening and closing beats across the grid,',
      ...cells,
      'continuous motion implied between left and right, high quality, no text',
    ].filter(Boolean).join('\n')
  }

  if (mode === 'multi_ref') {
    const sb = storyboards[0]
    const desc = sb.imagePrompt || sb.description || sb.title || 'scene'
    const totalCells = rows * cols
    const cells = Array.from({ length: totalCells }, (_, i) => {
      return `${cellLabel(i, rows, cols)}: ${legend ? `参考${legend}，` : ''}${desc}, ${GRID_ANGLES[i % GRID_ANGLES.length]}`
    })
    return [
      `${rows}x${cols} grid layout, same scene different angles and compositions, ${style},`,
      legend ? `参考图映射：${legend}` : '',
      `main scene: ${desc},`,
      ...cells,
      'consistent lighting and color palette, high quality, no text',
    ].filter(Boolean).join('\n')
  }

  return `${rows}x${cols} grid, ${style}, storyboard frames, high quality`
}

/**
 * 构建宫格图逐格 prompt（与 buildGridPrompt 同源，供 split 回写/逐格生成使用）
 */
export function buildGridCellPrompts(
  mode: string,
  storyboards: any[],
  rows: number,
  cols: number,
  referenceAssets: GridReferenceAsset[],
) {
  if (!storyboards.length) return []
  const storyboardCharacterIds = getStoryboardCharacterIds(storyboards.map((sb) => sb.id))

  if (mode === 'multi_ref') {
    const sb = storyboards[0]
    const desc = sb.imagePrompt || sb.description || sb.title || 'scene'
    return Array.from({ length: rows * cols }, (_, i) => {
      const refs = buildStoryboardReferenceHints(sb, referenceAssets, storyboardCharacterIds)
      return {
        shot_number: sb.storyboardNumber,
        frame_type: 'reference',
        prompt: `${cellLabel(i, rows, cols)}: ${refs.length ? `参考${refs.join('、')}，` : ''}${desc}, ${GRID_ANGLES[i % GRID_ANGLES.length]}`,
      }
    })
  }

  if (mode === 'first_last') {
    return Array.from({ length: rows * cols }, (_, i) => {
      const sb = storyboards[i % storyboards.length]
      const desc = sb.imagePrompt || sb.description || sb.title || `shot ${sb.storyboardNumber || ''}`
      const motion = sb.action || sb.movement || ''
      const refs = buildStoryboardReferenceHints(sb, referenceAssets, storyboardCharacterIds)
      const isFirst = i % 2 === 0
      const composition = getCameraMovementComposition(sb.movement || '', isFirst ? 'start' : 'end')
      const comp = composition ? `, ${composition}` : ''
      return {
        shot_number: sb.storyboardNumber,
        frame_type: isFirst ? 'first_frame' : 'last_frame',
        prompt: isFirst
          ? `${cellLabel(i, rows, cols)}，首帧：${refs.length ? `参考${refs.join('、')}，` : ''}${desc}${sb.location ? `, ${sb.location}` : ''}${sb.shotType ? `, ${sb.shotType}` : ''}${comp}`
          : `${cellLabel(i, rows, cols)}，尾帧：${refs.length ? `参考${refs.join('、')}，` : ''}${desc}${motion ? `, ${motion}` : ''}${sb.location ? `, ${sb.location}` : ''}${sb.shotType ? `, ${sb.shotType}` : ''}${comp}`,
      }
    })
  }

  return storyboards.slice(0, rows * cols).map((sb, index) => {
    const desc = sb.imagePrompt || sb.description || sb.title || `shot ${sb.storyboardNumber || ''}`
    const refs = buildStoryboardReferenceHints(sb, referenceAssets, storyboardCharacterIds)
    const composition = getCameraMovementComposition(sb.movement || '', 'start')
    const comp = composition ? `, ${composition}` : ''
    return {
      shot_number: sb.storyboardNumber,
      frame_type: 'first_frame',
      prompt: `${cellLabel(index, rows, cols)}：${refs.length ? `参考${refs.join('、')}，` : ''}${desc}${sb.location ? `, ${sb.location}` : ''}${sb.shotType ? `, ${sb.shotType}` : ''}, opening scene${comp}`,
    }
  })
}

