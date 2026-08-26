/**
 * 宫格图提示词生成 Agent 工具
 * 工厂函数模式 — 注入 episodeId + dramaId
 *
 * 宫格图 prompt 构建已统一收敛到 shared/prompt-utils.ts（buildGridPrompt /
 * buildGridCellPrompts / collectGridReferenceAssets）。本工具只负责读取镜头
 * 数据并调用统一构建器，避免与 grid 路由层重复实现、逻辑分叉。
 */
import { createTool } from '@mastra/core/tools'
import { buildGridPrompt, buildGridCellPrompts, collectGridReferenceAssets } from '../../shared/prompt-utils.js'
import { z } from 'zod'
import { db, schema } from '../../db/index.js'
import { eq } from 'drizzle-orm'

export function createGridPromptTools(episodeId: number, dramaId: number) {
  const readShotsForGrid = createTool({
    id: 'read_shots_for_grid',
    description: '读取选中镜头的详细信息，用于生成宫格图提示词。',
    inputSchema: z.object({
      shot_ids: z.array(z.number()),
    }),
    execute: async ({ shot_ids }) => {
      if (!shot_ids.length) return { shots: [] }
      const shots = db.select().from(schema.storyboards)
        .where(eq(schema.storyboards.episodeId, episodeId)).all()
        .filter(sb => shot_ids.includes(sb.id))
        .map(sb => ({
          shot_number: sb.storyboardNumber,
          description: sb.description || sb.title || '',
          shot_type: sb.shotType || '',
          dialogue: sb.dialogue || '',
          location: sb.location || '',
          time: sb.time || '',
          movement: sb.movement || '',
          first_frame_prompt: sb.firstFramePrompt || '',
          last_frame_prompt: sb.lastFramePrompt || '',
        }))
      return { shots }
    },
  })

  const generateGridPrompt = createTool({
    id: 'generate_grid_prompt',
    description: '为宫格图生成整体画面描述和每个格子的独立提示词。遵循 grid-image-generator SKILL.md 的三种模式规范。',
    inputSchema: z.object({
      shots: z.array(z.object({
        shot_number: z.number(),
        description: z.string().optional(),
        shot_type: z.string().optional(),
        dialogue: z.string().optional(),
        location: z.string().optional(),
        time: z.string().optional(),
        movement: z.string().optional(),
        first_frame_prompt: z.string().optional(),
        last_frame_prompt: z.string().optional(),
      })),
      rows: z.number(),
      cols: z.number(),
      mode: z.string(), // 'first_frame' | 'first_last' | 'multi_ref'
    }),
    execute: async ({ shots, rows, cols, mode }) => {
      if (!shots.length) return { error: 'No shots provided', grid_prompt: '', cell_prompts: [] }

      // 反查 DB 拿完整 storyboard（含参考图、角色外观、场景等字段），
      // 确保 agent 路径与路由 fallback 使用完全一致的构建逻辑。
      const shotNumbers = shots.map(s => s.shot_number)
      const allStoryboards = db.select().from(schema.storyboards)
        .where(eq(schema.storyboards.episodeId, episodeId)).all()
      const ordered = shotNumbers
        .map(n => allStoryboards.find(sb => sb.storyboardNumber === n))
        .filter(Boolean)

      const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
      const dramaStyle = drama?.style || ''

      const referenceAssets = collectGridReferenceAssets(ordered)
      const gridPrompt = buildGridPrompt(mode, ordered, rows, cols, dramaStyle, referenceAssets)
      const cellPrompts = buildGridCellPrompts(mode, ordered, rows, cols, referenceAssets)

      return {
        grid_prompt: gridPrompt,
        cell_prompts: cellPrompts,
        reference_assets: referenceAssets.map(a => ({ image_label: a.imageLabel, path: a.path, label: a.label })),
      }
    },
  })

  return {
    readShotsForGrid,
    generateGridPrompt,
  }
}
