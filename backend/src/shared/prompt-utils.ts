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
import { eq } from 'drizzle-orm'

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
// 角色视觉 Prompt 构建
// ============================================================

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
}): string {
  const parts: string[] = [char.name]
  if (char.appearance) parts.push(char.appearance)
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
}): string {
  const parts: string[] = [char.name]
  if (char.appearance) parts.push(char.appearance)
  if (char.description && char.description !== char.appearance) parts.push(char.description)
  return parts.join(': ')
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
    .all()
    .filter(link => link.storyboardId === storyboardId)
  
  if (!spChars.length) return []

  const characterIds = spChars.map(link => link.characterId)
  const characters = db.select().from(schema.characters)
    .all()
    .filter(char => characterIds.includes(char.id))

  return characters.map(char =>
    buildCharacterAppearanceText({
      name: char.name,
      appearance: char.appearance,
      description: char.description,
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
    .all()
    .filter(char => characterIds.includes(char.id) && char.imageUrl)

  return characters.map(char => char.imageUrl!)
}

/**
 * 根据分镜ID获取关联角色的图片URL
 */
export function getStoryboardCharacterImageUrls(storyboardId: number): string[] {
  const spChars = db.select().from(schema.storyboardCharacters)
    .all()
    .filter(link => link.storyboardId === storyboardId)
  
  const characterIds = spChars.map(link => link.characterId)
  return getCharacterImageUrls(characterIds)
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

  // 获取分镜关联的角色
  const spChars = db.select().from(schema.storyboardCharacters)
    .all()
    .filter(link => link.storyboardId === storyboardId)

  const characterIds = spChars.map(link => link.characterId)
  const characters = characterIds.length
    ? db.select().from(schema.characters)
        .all()
        .filter(char => characterIds.includes(char.id))
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
