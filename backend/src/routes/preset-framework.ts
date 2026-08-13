/**
 * 预设框架路由（Preset Framework Routes）
 *
 * 通用骨架端点，提供 Variation Card 生成 → Drama 创建 → 管线编排的完整 API。
 * 所有端点名称已通用化，便于后续根据实际需求进行定制和扩展。
 */

import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import {
  generateVariationCard,
  createPresetDrama,
  triggerImageGeneration,
  triggerVideoGeneration,
  getPresetPipelineStatus,
} from '../services/preset-framework.js'
import type { PresetVariationCard } from '../shared/prompt-utils.js'
import { logTaskError } from '../utils/task-logger.js'

const router = new Hono()

// ============================================================
// Variation Card 生成
// ============================================================

/**
 * GET /preset/framework/variation-card
 *
 * 随机生成一张 Variation Card（5 镜配置）。
 * Query: ?excludeFamily=THEME_X  排除上次使用的家族，避免连续重复
 */
router.get('/variation-card', (c) => {
  const excludeFamily = c.req.query('excludeFamily') || undefined

  try {
    const card = generateVariationCard(excludeFamily)
    return c.json({ success: true, data: card })
  } catch (err: any) {
    return c.json({ success: false, msg: err.message }, 500)
  }
})

// ============================================================
// Drama 创建
// ============================================================

/**
 * POST /preset/framework/create
 *
 * 基于 Variation Card 创建 Drama + Episode + 5 Storyboards。
 *
 * Body:
 * {
 *   title: string,
 *   description?: string,
 *   variationCard: PresetVariationCard
 * }
 *
 * Response:
 * { dramaId, episodeId, storyboardIds }
 */
router.post('/create', async (c) => {
  try {
    const body = await c.req.json()
    const { title, description, variationCard } = body

    if (!title) {
      return c.json({ success: false, msg: 'title is required' }, 400)
    }

    if (!variationCard || !variationCard.shots || variationCard.shots.length === 0) {
      return c.json({ success: false, msg: 'variationCard with shots is required' }, 400)
    }

    const result = await createPresetDrama(
      title,
      description || '',
      variationCard as PresetVariationCard
    )

    return c.json({ success: true, data: result })
  } catch (err: any) {
    logTaskError('PresetFramework', 'create-drama', { error: err.message, stack: err.stack })
    return c.json({ success: false, msg: err.message }, 500)
  }
})

// ============================================================
// 管线编排
// ============================================================

/**
 * POST /preset/framework/generate-images
 *
 * 批量生成 5 个 Storyboard 的首帧图片（复用现有 image-generation 管线）。
 *
 * Body:
 * {
 *   dramaId: number,
 *   episodeId: number,
 *   storyboardIds: number[],
 *   variationCard: PresetVariationCard
 * }
 */
router.post('/generate-images', async (c) => {
  try {
    const body = await c.req.json()
    const { dramaId, episodeId, storyboardIds, variationCard } = body

    if (!dramaId || !episodeId || !storyboardIds?.length || !variationCard) {
      return c.json({ success: false, msg: 'dramaId, episodeId, storyboardIds, and variationCard are required' }, 400)
    }

    const result = await triggerImageGeneration(
      dramaId, episodeId, storyboardIds,
      variationCard as PresetVariationCard
    )

    return c.json({ success: true, data: { dramaId, episodeId, ...result } })
  } catch (err: any) {
    console.error('[PresetFramework] Generate images error:', err)
    return c.json({ success: false, msg: err.message }, 500)
  }
})

/**
 * POST /preset/framework/generate-videos
 *
 * 批量生成 5 个 Storyboard 的视频（图生视频，需要首帧已生成）。
 *
 * Body:
 * {
 *   dramaId: number,
 *   episodeId: number,
 *   storyboardIds: number[],
 *   variationCard: PresetVariationCard
 * }
 */
router.post('/generate-videos', async (c) => {
  try {
    const body = await c.req.json()
    const { dramaId, episodeId, storyboardIds, variationCard } = body

    if (!dramaId || !episodeId || !storyboardIds?.length || !variationCard) {
      return c.json({ success: false, msg: 'dramaId, episodeId, storyboardIds, and variationCard are required' }, 400)
    }

    const result = await triggerVideoGeneration(
      dramaId, episodeId, storyboardIds,
      variationCard as PresetVariationCard
    )

    return c.json({ success: true, data: { dramaId, episodeId, ...result } })
  } catch (err: any) {
    logTaskError('PresetFramework', 'generate-videos', { error: err.message, stack: err.stack })
    return c.json({ success: false, msg: err.message }, 500)
  }
})

// ============================================================
// 状态查询
// ============================================================

/**
 * GET /preset/framework/status/:dramaId
 *
 * 查询管线当前状态：Drama / Episode / Storyboards / Variation Card / 生成进度汇总
 */
router.get('/status/:dramaId', async (c) => {
  try {
    const dramaId = parseInt(c.req.param('dramaId'))
    if (isNaN(dramaId)) {
      return c.json({ success: false, msg: 'Invalid dramaId' }, 400)
    }

    const status = await getPresetPipelineStatus(dramaId)
    return c.json({ success: true, data: status })
  } catch (err: any) {
    console.error('[PresetFramework] Get status error:', err)
    return c.json({ success: false, msg: err.message }, 500)
  }
})

// ============================================================
// 一键全流程
// ============================================================

/**
 * POST /preset/framework/full-pipeline
 *
 * 一键执行：生成 Card → 创建 Drama → 批量生图。
 *
 * Body:
 * {
 *   title: string,
 *   description?: string,
 *   variationCard: PresetVariationCard,
 *   autoGenerateImages?: boolean  默认 true
 * }
 */
router.post('/full-pipeline', async (c) => {
  try {
    const body = await c.req.json()
    const { title, description, variationCard, autoGenerateImages = true } = body

    if (!title) {
      return c.json({ success: false, msg: 'title is required' }, 400)
    }

    if (!variationCard || !variationCard.shots || variationCard.shots.length === 0) {
      return c.json({ success: false, msg: 'variationCard with shots is required' }, 400)
    }

    // Step 1: 创建 Drama + Storyboards
    const { dramaId, episodeId, storyboardIds } = await createPresetDrama(
      title, description || '', variationCard as PresetVariationCard
    )

    // Step 2: 批量生图
    let imageGenIds: number[] = []

    if (autoGenerateImages) {
      const result = await triggerImageGeneration(
        dramaId, episodeId, storyboardIds,
        variationCard as PresetVariationCard
      )
      imageGenIds = result.imageGenIds
    }

    return c.json({
      success: true,
      data: {
        dramaId,
        episodeId,
        storyboardIds,
        imageGenIds,
        variationCard,
      },
    })
  } catch (err: any) {
    logTaskError('PresetFramework', 'full-pipeline', { error: err.message, stack: err.stack })
    return c.json({ success: false, msg: err.message }, 500)
  }
})

export default router
