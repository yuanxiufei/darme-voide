/**
 * 预设框架服务（Preset Framework Service）
 *
 * 通用骨架：提供 Variation Card 引擎 + 管线编排逻辑。
 * 所有领域内容（主题家族、构图模式、核心焦点等）在此处仅作为占位符数据池，
 * 实际使用时替换为具体领域知识。
 *
 * 核心架构：
 *   1. Variation Card 生成（随机组合 + 去重 + 配比约束）
 *   2. Drama/Episode/Storyboard 创建
 *   3. 复用现有 image-generation / video-generation 管线
 */

import { db, schema } from '../db/index.js'
import { eq } from 'drizzle-orm'
import type { PresetVariationCard } from '../shared/prompt-utils.js'
import { generateImage } from './image-generation.js'
import { generateVideo } from './video-generation.js'

// ============================================================
// 占位符数据池（使用时替换为具体领域数据）
// ============================================================

const THEME_FAMILIES = [
  'THEME_FAMILY_A',
  'THEME_FAMILY_B',
  'THEME_FAMILY_C',
  'THEME_FAMILY_D',
  'THEME_FAMILY_E',
  'THEME_FAMILY_F',
  'THEME_FAMILY_G',
  'THEME_FAMILY_H',
]

const COMPOSITION_PATTERNS = [
  'COMPOSITION_TYPE_1',
  'COMPOSITION_TYPE_2',
  'COMPOSITION_TYPE_3',
  'COMPOSITION_TYPE_4',
  'COMPOSITION_TYPE_5',
  'COMPOSITION_TYPE_6',
  'COMPOSITION_TYPE_7',
  'COMPOSITION_TYPE_8',
  'COMPOSITION_TYPE_9',
  'COMPOSITION_TYPE_10',
]

const MAIN_FOCAL_POINTS = [
  'FOCAL_POINT_A',
  'FOCAL_POINT_B',
  'FOCAL_POINT_C',
  'FOCAL_POINT_D',
  'FOCAL_POINT_E',
  'FOCAL_POINT_F',
]

const SUBTLE_CLUES = [
  'SUBTLE_CLUE_1',
  'SUBTLE_CLUE_2',
  'SUBTLE_CLUE_3',
  'SUBTLE_CLUE_4',
]

const PROMINENT_CLUES = [
  'PROMINENT_CLUE_1',
  'PROMINENT_CLUE_2',
]

const ACTIVITIES = [
  'ACTIVITY_1',
  'ACTIVITY_2',
  'ACTIVITY_3',
  'ACTIVITY_4',
  'ACTIVITY_5',
  'ACTIVITY_6',
  'ACTIVITY_7',
  'ACTIVITY_8',
  'ACTIVITY_9',
]

const LIVING_ELEMENTS = [
  'LIVING_ELEMENT_A',
  'LIVING_ELEMENT_B',
  'LIVING_ELEMENT_C',
]

const LIGHT_STRUCTURES = [
  'LIGHT_STRUCTURE_1',
  'LIGHT_STRUCTURE_2',
  'LIGHT_STRUCTURE_3',
  'LIGHT_STRUCTURE_4',
  'LIGHT_STRUCTURE_5',
  'LIGHT_STRUCTURE_6',
]

const CAMERA_POSITIONS = [
  'CAMERA_POSITION_1',
  'CAMERA_POSITION_2',
  'CAMERA_POSITION_3',
  'CAMERA_POSITION_4',
  'CAMERA_POSITION_5',
  'CAMERA_POSITION_6',
  'CAMERA_POSITION_7',
  'CAMERA_POSITION_8',
  'CAMERA_POSITION_9',
]

const WIND_DIRECTIONS = [
  'WIND_DIRECTION_LEFT',
  'WIND_DIRECTION_RIGHT',
  'WIND_DIRECTION_CENTER',
]

const CAMERA_MOVES = [
  'CAMERA_MOVE_SLOW_LEFT',
  'CAMERA_MOVE_SLOW_RIGHT',
  'CAMERA_MOVE_PUSH_IN',
  'CAMERA_MOVE_PULL_OUT',
  'CAMERA_MOVE_STATIC',
]

// 配比约束
const MAX_LIVING_SHOTS = 3
const CLUE_RATIO = { subtle: 3, prominent: 2 } // 5镜中 subtle:prominent ≈ 3:2

// ============================================================
// 工具函数
// ============================================================

function randomPick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

function randomPickN<T>(arr: T[], n: number): T[] {
  const shuffled = [...arr].sort(() => Math.random() - 0.5)
  return shuffled.slice(0, n)
}

function randomNumber(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min
}

/**
 * 根据 themeFamily + mainFocalPoint 派生空间类型和前景框架
 * 【占位符】实际使用时替换为具体领域逻辑
 */
function deriveSpaceAndFrame(themeFamily: string, focalPoint: string): {
  spaceType: string
  foregroundFrame: string
} {
  const hash = (themeFamily + focalPoint).length
  const types = ['SPACE_TYPE_A', 'SPACE_TYPE_B', 'SPACE_TYPE_C', 'SPACE_TYPE_D']
  const frames = ['FRAME_TYPE_1', 'FRAME_TYPE_2', 'FRAME_TYPE_3']
  return {
    spaceType: types[hash % types.length],
    foregroundFrame: frames[hash % frames.length],
  }
}

/**
 * 根据 activity 选取人物排列方式
 * 【占位符】实际使用时替换为具体领域逻辑
 */
function pickCharacterLayout(_activity: string): string {
  const layouts = [
    'LAYOUT_SOLO',
    'LAYOUT_PAIR',
    'LAYOUT_TRIANGLE',
    'LAYOUT_SCATTERED',
  ]
  return randomPick(layouts)
}

// ============================================================
// Variation Card 引擎
// ============================================================

/**
 * 生成一张 Variation Card（5 个 Shot 配置）
 *
 * @param excludeFamily 可选，排除上次使用的家族，避免连续重复
 */
export function generateVariationCard(excludeFamily?: string): PresetVariationCard {
  // 1. 选取主题家族
  let familyPool = [...THEME_FAMILIES]
  if (excludeFamily) {
    familyPool = familyPool.filter(f => f !== excludeFamily)
  }
  const themeFamily = randomPick(familyPool)

  // 2. 5 个 shot 各自独立分配
  const compositions = randomPickN(COMPOSITION_PATTERNS, 5)
  const focalPoints = randomPickN(MAIN_FOCAL_POINTS, 5)
  const activities = randomPickN(ACTIVITIES, 5)

  // 3. 主题线索按配比分配
  const totalSubtle = CLUE_RATIO.subtle
  const totalProminent = CLUE_RATIO.prominent
  const cluePool: string[] = []
  for (let i = 0; i < totalSubtle; i++) cluePool.push(randomPick(SUBTLE_CLUES))
  for (let i = 0; i < totalProminent; i++) cluePool.push(randomPick(PROMINENT_CLUES))
  const shuffledClues = cluePool.sort(() => Math.random() - 0.5)

  // 4. 灵动元素分配（最多 MAX_LIVING_SHOTS 个镜头出现）
  const livingPool: (string | null)[] = Array(5).fill(null)
  const livingCount = randomNumber(0, MAX_LIVING_SHOTS)
  if (livingCount > 0) {
    const indices = randomPickN([0, 1, 2, 3, 4], livingCount)
    for (const idx of indices) {
      livingPool[idx] = randomPick(LIVING_ELEMENTS)
    }
  }

  // 5. 组装 5 个 Shot
  const shots = Array.from({ length: 5 }, (_, i) => {
    const { spaceType, foregroundFrame } = deriveSpaceAndFrame(themeFamily, focalPoints[i])

    return {
      shotIndex: i + 1,
      themeFamily,
      compositionPattern: compositions[i],
      spaceType,
      foregroundFrame,
      mainFocalPoint: focalPoints[i],
      thematicClue: shuffledClues[i],
      activity: activities[i],
      cameraPosition: randomPick(CAMERA_POSITIONS),
      windDirection: randomPick(WIND_DIRECTIONS),
      lightStructure: randomPick(LIGHT_STRUCTURES),
      characterLayout: pickCharacterLayout(activities[i]),
      livingElement: livingPool[i],
    }
  })

  return { themeFamily, shots }
}

// ============================================================
// Drama 创建（复用现有 Schema）
// ============================================================

/**
 * 基于 Variation Card 创建 Drama → Episode → 5 Storyboards
 */
export async function createPresetDrama(
  title: string,
  description: string,
  variationCard: PresetVariationCard
): Promise<{
  dramaId: number
  episodeId: number
  storyboardIds: number[]
}> {
  // 创建 Drama
  const [drama] = await db.insert(schema.dramas).values({
    title,
    description: description || `预设风格: ${variationCard.themeFamily}`,
    style: 'preset-framework',
    totalEpisodes: 1,
    systemMetadata: JSON.stringify({ variationCard, presetType: 'framework' }),
    status: 'draft',
  } as any).returning({ id: schema.dramas.id })

  // 创建 Episode
  const [episode] = await db.insert(schema.episodes).values({
    dramaId: drama.id,
    episodeNumber: 1,
    title: `${title} - 第1集`,
    status: 'draft',
  } as any).returning({ id: schema.episodes.id })

  // 创建 5 个 Storyboard
  const storyboardIds: number[] = []
  for (const shot of variationCard.shots) {
    const [sb] = await db.insert(schema.storyboards).values({
      episodeId: episode.id,
      sceneNumber: shot.shotIndex,
      shotType: shot.compositionPattern,
      description: `Shot ${shot.shotIndex}: ${shot.mainFocalPoint}`,
      cameraMovement: randomPick(CAMERA_MOVES),
      systemMetadata: JSON.stringify({ shotConfig: shot }),
      status: 'pending',
    } as any).returning({ id: schema.storyboards.id })
    storyboardIds.push(sb.id)
  }

  return { dramaId: drama.id, episodeId: episode.id, storyboardIds }
}

// ============================================================
// 管线编排（复用现有服务）
// ============================================================

/**
 * 批量生成 5 个 Shot 的首帧图片
 */
export async function triggerImageGeneration(
  dramaId: number,
  episodeId: number,
  storyboardIds: number[],
  variationCard: PresetVariationCard
): Promise<{ imageGenIds: number[] }> {
  const imageGenIds: number[] = []

  for (const sbId of storyboardIds) {
    const shot = variationCard.shots.find(s => s.shotIndex === storyboardIds.indexOf(sbId) + 1)
    if (!shot) continue

    try {
      // 更新 Storyboard 状态
      await db.update(schema.storyboards)
        .set({ status: 'generating_image' } as any)
        .where(eq(schema.storyboards.id, sbId))

      // 构建 prompt（调用 prompt-utils 中的构建器）
      const { buildPresetImagePrompt, PRESET_IMAGE_NEGATIVE } = await import('../shared/prompt-utils.js')
      const prompt = buildPresetImagePrompt(shot)

      // 调用现有图片生成服务（返回 image record ID）
      const imageId = await generateImage({
        dramaId,
        storyboardId: sbId,
        prompt,
        negativePrompt: PRESET_IMAGE_NEGATIVE,
      } as any)

      if (imageId) {
        imageGenIds.push(imageId)
        await db.update(schema.storyboards)
          .set({ status: 'image_ready' } as any)
          .where(eq(schema.storyboards.id, sbId))
      }
    } catch (err) {
      console.error(`[PresetFramework] Image generation failed for shot ${shot.shotIndex}:`, err)
      await db.update(schema.storyboards)
        .set({ status: 'image_failed' } as any)
        .where(eq(schema.storyboards.id, sbId))
    }
  }

  return { imageGenIds }
}

/**
 * 批量生成 5 个 Shot 的视频（图生视频，需要首帧已生成）
 */
export async function triggerVideoGeneration(
  dramaId: number,
  episodeId: number,
  storyboardIds: number[],
  variationCard: PresetVariationCard
): Promise<{ videoGenIds: number[] }> {
  const videoGenIds: number[] = []

  for (const sbId of storyboardIds) {
    const shot = variationCard.shots.find(s => s.shotIndex === storyboardIds.indexOf(sbId) + 1)
    if (!shot) continue

    // 检查首帧是否已就绪
    const [sb] = await db.select()
      .from(schema.storyboards)
      .where(eq(schema.storyboards.id, sbId))

    if (!sb || !(sb as any).imageUrl) {
      console.warn(`[PresetFramework] Skipping video for shot ${shot.shotIndex}: no first frame`)
      continue
    }

    try {
      await db.update(schema.storyboards)
        .set({ status: 'generating_video' } as any)
        .where(eq(schema.storyboards.id, sbId))

      const cameraMove = randomPick(CAMERA_MOVES)

      const { buildPresetVideoPrompt } = await import('../shared/prompt-utils.js')
      const prompt = buildPresetVideoPrompt(shot, cameraMove)

      const videoId = await generateVideo({
        dramaId,
        storyboardId: sbId,
        prompt,
        referenceImageUrl: (sb as any).imageUrl,
      } as any)

      if (videoId) videoGenIds.push(videoId)

      await db.update(schema.storyboards)
        .set({ status: 'video_ready' } as any)
        .where(eq(schema.storyboards.id, sbId))
    } catch (err) {
      console.error(`[PresetFramework] Video generation failed for shot ${shot.shotIndex}:`, err)
      await db.update(schema.storyboards)
        .set({ status: 'video_failed' } as any)
        .where(eq(schema.storyboards.id, sbId))
    }
  }

  return { videoGenIds }
}

// ============================================================
// 状态查询
// ============================================================

export async function getPresetPipelineStatus(dramaId: number) {
  const [drama] = await db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId))
  if (!drama) throw new Error('Drama not found')

  const episodes = await db.select().from(schema.episodes).where(eq(schema.episodes.dramaId, dramaId))
  const episode = episodes[0]

  let storyboards: any[] = []
  if (episode) {
    storyboards = await db.select().from(schema.storyboards)
      .where(eq(schema.storyboards.episodeId, episode.id))
      .orderBy(schema.storyboards.storyboardNumber)
  }

  const variationCard = (drama as any).systemMetadata?.variationCard || null

  return {
    drama: { id: drama.id, title: drama.title, style: drama.style },
    episode: episode ? { id: episode.id, episodeNumber: episode.episodeNumber } : null,
    storyboards: storyboards.map(s => ({
      id: s.id,
      storyboardNumber: s.storyboardNumber,
      imageUrl: (s as any).imageUrl,
      videoUrl: (s as any).videoUrl,
      status: s.status,
    })),
    variationCard,
    summary: {
      totalShots: storyboards.length,
      imagesGenerated: storyboards.filter(s => (s as any).imageUrl).length,
      videosGenerated: storyboards.filter(s => (s as any).videoUrl).length,
    },
  }
}
