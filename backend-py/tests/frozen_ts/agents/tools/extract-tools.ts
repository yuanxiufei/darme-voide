/**
 * 角色/场景提取 Agent 工具
 * 工厂函数模式 — 注入 episodeId + dramaId
 *
 * 单 Agent 一步流程：
 * 1. 读取剧本内容
 * 2. 读取项目中已存在的角色/场景（用于去重）
 * 3. 提取角色/场景并智能去重后直接保存
 */
import { createTool } from '@mastra/core/tools'
import { z } from 'zod'
import { db, schema } from '../../db/index.js'
import { eq, and } from 'drizzle-orm'
import { now } from '../../utils/response.js'
import { logTaskProgress, logTaskSuccess } from '../../utils/task-logger.js'
import { sliceLongText } from '../../utils/text-slice.js'
import {
  buildCharacterImagePrompt,
  buildSceneImagePrompt,
  buildPropImagePrompt,
  buildCharacterNegativePrompt,
  SCENE_IMAGE_NEGATIVE,
} from '../../shared/prompt-utils.js'

// ─── 关联辅助 ────────────────────────────────────────────────
function linkCharToEpisode(episodeId: number, characterId: number) {
  const ts = now()
  const existing = db.select().from(schema.episodeCharacters)
    .where(and(eq(schema.episodeCharacters.episodeId, episodeId), eq(schema.episodeCharacters.characterId, characterId)))
    .all()
  if (!existing.length) {
    db.insert(schema.episodeCharacters).values({ episodeId, characterId, createdAt: ts }).run()
  }
}

function linkSceneToEpisode(episodeId: number, sceneId: number) {
  const ts = now()
  const existing = db.select().from(schema.episodeScenes)
    .where(and(eq(schema.episodeScenes.episodeId, episodeId), eq(schema.episodeScenes.sceneId, sceneId)))
    .all()
  if (!existing.length) {
    db.insert(schema.episodeScenes).values({ episodeId, sceneId, createdAt: ts }).run()
  }
}

function linkEpisodeToProp(episodeId: number, propId: number) {
  const existing = db.select().from(schema.episodeProps)
    .where(and(eq(schema.episodeProps.episodeId, episodeId), eq(schema.episodeProps.propId, propId)))
    .all()
  if (!existing.length) {
    db.insert(schema.episodeProps).values({ episodeId, propId }).run()
  }
}

/** 从自由文本 role 推断标准化角色类型（主角/反派/配角/旁白/其他） */
function inferRoleType(role: string | undefined | null): string {
  if (!role) return ''
  if (/主角|男主|女主|hero|protagonist|lead/i.test(role)) return '主角'
  if (/反派|坏|恶|villain|antagonist|boss/i.test(role)) return '反派'
  if (/龙套|路人/i.test(role)) return '龙套'
  if (/配角|supporting|side/i.test(role)) return '配角'
  if (/旁白|叙述|narrator|画外音/i.test(role)) return '旁白'
  return '其他'
}

export function createExtractTools(episodeId: number, dramaId: number) {

  // 1. 读取剧本内容
  const readScriptForExtraction = createTool({
    id: 'read_script_for_extraction',
    description: 'Read the formatted screenplay for character/scene extraction.',
    inputSchema: z.object({}),
    execute: async () => {
      const [ep] = db.select().from(schema.episodes)
        .where(eq(schema.episodes.id, episodeId)).all()
      if (!ep) return { error: 'Episode not found' }
      const content = ep.scriptContent || ep.content
      if (!content) return { error: 'Episode has no script content' }
      const sliced = sliceLongText(content)
      logTaskSuccess('ExtractTool', 'read-script', { episodeId, dramaId, scriptLength: content.length })
      return { script: sliced.text, truncated: sliced.truncated, total_chars: sliced.total_chars }
    },
  })

  // 2. 读取项目中已存在的角色（用于去重判断）
  const readExistingCharacters = createTool({
    id: 'read_existing_characters',
    description: 'Read all characters already existing in this drama project (for deduplication).',
    inputSchema: z.object({}),
    execute: async () => {
      const linkedIds = new Set(
        db.select().from(schema.episodeCharacters)
          .where(eq(schema.episodeCharacters.episodeId, episodeId)).all()
          .map(link => link.characterId),
      )
      const chars = db.select().from(schema.characters)
        .where(eq(schema.characters.dramaId, dramaId)).all()
        .filter(c => !c.deletedAt)
      const payload = {
        count: chars.length,
        characters: chars,
        current_episode_characters: chars.filter(c => linkedIds.has(c.id)),
      }
      logTaskSuccess('ExtractTool', 'read-characters', {
        episodeId,
        dramaId,
        projectCharacters: payload.count,
        episodeCharacters: payload.current_episode_characters.length,
      })
      return payload
    },
  })

  // 3. 读取项目中已存在的场景（用于去重判断）
  const readExistingScenes = createTool({
    id: 'read_existing_scenes',
    description: 'Read all scenes already existing in this drama project (for deduplication).',
    inputSchema: z.object({}),
    execute: async () => {
      const linkedIds = new Set(
        db.select().from(schema.episodeScenes)
          .where(eq(schema.episodeScenes.episodeId, episodeId)).all()
          .map(link => link.sceneId),
      )
      const scenes = db.select().from(schema.scenes)
        .where(eq(schema.scenes.dramaId, dramaId)).all()
        .filter(s => !s.deletedAt)
      const payload = {
        count: scenes.length,
        scenes,
        current_episode_scenes: scenes.filter(s => linkedIds.has(s.id)),
      }
      logTaskSuccess('ExtractTool', 'read-scenes', {
        episodeId,
        dramaId,
        projectScenes: payload.count,
        episodeScenes: payload.current_episode_scenes.length,
      })
      return payload
    },
  })

  // 4. 智能保存角色（按名字去重，与现有数据合并）
  const saveDedupCharacters = createTool({
    id: 'save_dedup_characters',
    description: 'Save extracted characters with deduplication. Existing characters (same name) are merged/updated; new ones are created. All are linked to the current episode.',
    inputSchema: z.object({
      characters: z.array(z.object({
        name: z.string(),
        role: z.string().optional(),
        role_type: z.string().optional(),
        description: z.string().optional(),
        appearance: z.string().optional(),
        personality: z.string().optional(),
        clothing: z.string().optional(),
        weapons: z.string().optional(),
        accessories: z.string().optional(),
        core_features: z.array(z.string()).optional(),
        costumes: z.array(z.string()).optional(),
        image_prompt: z.string().optional(),
        negative_prompt: z.string().optional(),
      })),
    }),
    execute: async ({ characters }) => {
      const ts = now()
      const results = { created: 0, merged: 0 }
      const [dramaRow] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
      const [globalRow] = db.select().from(schema.appSettings).where(eq(schema.appSettings.key, 'art_style')).all()
      // 剧集未设画风时回退全局默认画风，保证提取兜底 prompt 与生成链路一致
      const dramaStyle = dramaRow?.style || globalRow?.value || null
      logTaskProgress('ExtractTool', 'save-characters-begin', {
        episodeId,
        dramaId,
        names: characters.map(char => char.name).join(','),
      })

      for (const char of characters) {
        const existing = db.select().from(schema.characters)
          .where(eq(schema.characters.dramaId, dramaId)).all()
          .filter(c => !c.deletedAt)
          .find(c => c.name === char.name)

        // 合并字段：agent 本次提供的值优先，缺省回退已有数据
        const merged = {
          name: char.name,
          role: char.role || existing?.role || '',
          roleType: char.role_type || inferRoleType(char.role || existing?.role || ''),
          description: char.description || existing?.description || '',
          appearance: char.appearance || existing?.appearance || '',
          personality: char.personality || existing?.personality || '',
          clothing: char.clothing || existing?.clothing || '',
          weapons: char.weapons || existing?.weapons || '',
          accessories: char.accessories || existing?.accessories || '',
          coreFeatures: char.core_features?.length ? JSON.stringify(char.core_features) : (existing?.coreFeatures || ''),
          costumes: char.costumes?.length ? JSON.stringify(char.costumes) : (existing?.costumes || ''),
        }

        // 提示词兜底：agent 值 > 已有值 > 基于合并字段自动构建（不依赖 LLM 必填，避免漏填导致留空）
        const customPrompt = char.image_prompt || existing?.customPrompt || buildCharacterImagePrompt({
          name: merged.name,
          appearance: merged.appearance,
          description: merged.description,
          personality: merged.personality,
          coreFeatures: merged.coreFeatures || undefined,
          clothing: merged.clothing,
          costumes: merged.costumes || undefined,
          dramaStyle,
        })
        const negativePrompt = char.negative_prompt || existing?.negativePrompt || buildCharacterNegativePrompt(dramaStyle)

        if (existing) {
          // 已存在：合并信息，保留 ID
          db.update(schema.characters).set({
            role: merged.role,
            roleType: merged.roleType,
            description: merged.description,
            appearance: merged.appearance,
            personality: merged.personality,
            clothing: merged.clothing,
            weapons: merged.weapons,
            accessories: merged.accessories,
            coreFeatures: merged.coreFeatures,
            costumes: merged.costumes,
            customPrompt,
            negativePrompt,
            updatedAt: ts,
          }).where(eq(schema.characters.id, existing.id)).run()
          linkCharToEpisode(episodeId, existing.id)
          results.merged++
        } else {
          // 新增角色
          const res = db.insert(schema.characters).values({
            name: merged.name,
            role: merged.role,
            roleType: merged.roleType,
            description: merged.description,
            appearance: merged.appearance,
            personality: merged.personality,
            clothing: merged.clothing,
            weapons: merged.weapons,
            accessories: merged.accessories,
            coreFeatures: merged.coreFeatures,
            costumes: merged.costumes,
            customPrompt,
            negativePrompt,
            dramaId,
            createdAt: ts,
            updatedAt: ts,
          }).run()
          const charId = Number(res.lastInsertRowid)
          linkCharToEpisode(episodeId, charId)
          results.created++
        }
      }

      const payload = {
        message: `角色保存完成：新增 ${results.created}，合并更新 ${results.merged}`,
        ...results,
      }
      logTaskSuccess('ExtractTool', 'save-characters-complete', { episodeId, ...results })
      return payload
    },
  })

  // 5. 智能保存场景（按地点+时间段去重，与现有数据合并）
  const saveDedupScenes = createTool({
    id: 'save_dedup_scenes',
    description: 'Save extracted scenes with deduplication. Existing scenes (same location+time) are reused; new ones are created. All are linked to the current episode.',
    inputSchema: z.object({
      scenes: z.array(z.object({
        location: z.string(),
        time: z.string().optional(),
        prompt: z.string().optional(),
        description: z.string().optional(),
        atmosphere: z.string().optional(),
        lighting: z.string().optional(),
        weather: z.string().optional(),
        season: z.string().optional(),
        style: z.string().optional(),
        image_prompt: z.string().optional(),
        negative_prompt: z.string().optional(),
      })),
    }),
    execute: async ({ scenes }) => {
      const ts = now()
      const results = { created: 0, reused: 0 }
      logTaskProgress('ExtractTool', 'save-scenes-begin', {
        episodeId,
        dramaId,
        scenes: scenes.map(scene => `${scene.location}@${scene.time || ''}`).join(','),
      })

      for (const scene of scenes) {
        // 按地点+时间段精确匹配
        const existing = db.select().from(schema.scenes)
          .where(eq(schema.scenes.dramaId, dramaId)).all()
          .filter(s => !s.deletedAt)
          .find(s => s.location === scene.location && s.time === (scene.time || ''))

        // 提示词兜底：agent 值 > 已有值 > 基于场景字段自动构建（不依赖 LLM 必填）
        const fallbackImagePrompt = buildSceneImagePrompt({
          location: scene.location,
          time: scene.time,
          prompt: scene.prompt || scene.description,
        })

        if (existing) {
          // 已存在完全匹配的场景：合并更新细节与提示词后关联
          db.update(schema.scenes).set({
            description: scene.description || existing.description,
            atmosphere: scene.atmosphere || existing.atmosphere,
            lighting: scene.lighting || existing.lighting,
            weather: scene.weather || existing.weather,
            season: scene.season || existing.season,
            style: scene.style || existing.style,
            customPrompt: scene.image_prompt || existing.customPrompt || fallbackImagePrompt,
            negativePrompt: scene.negative_prompt || existing.negativePrompt || SCENE_IMAGE_NEGATIVE,
            updatedAt: ts,
          }).where(eq(schema.scenes.id, existing.id)).run()
          linkSceneToEpisode(episodeId, existing.id)
          results.reused++
        } else {
          // 检查是否有同地点不同时段（保留现有，新增独立场景）
          const sameLocation = db.select().from(schema.scenes)
            .where(eq(schema.scenes.dramaId, dramaId)).all()
            .filter(s => !s.deletedAt)
            .find(s => s.location === scene.location)

          const res = db.insert(schema.scenes).values({
            dramaId,
            location: scene.location,
            time: scene.time || '',
            prompt: scene.prompt || scene.location,
            description: scene.description || '',
            atmosphere: scene.atmosphere || '',
            lighting: scene.lighting || '',
            weather: scene.weather || '',
            season: scene.season || '',
            style: scene.style || '',
            customPrompt: scene.image_prompt || fallbackImagePrompt,
            negativePrompt: scene.negative_prompt || SCENE_IMAGE_NEGATIVE,
            createdAt: ts,
            updatedAt: ts,
          }).run()
          const sceneId = Number(res.lastInsertRowid)
          linkSceneToEpisode(episodeId, sceneId)
          results.created++
        }
      }

      const payload = {
        message: `场景保存完成：新增 ${results.created}，复用已有 ${results.reused}`,
        ...results,
      }
      logTaskSuccess('ExtractTool', 'save-scenes-complete', { episodeId, ...results })
      return payload
    },
  })

  const readExistingProps = createTool({
    id: 'read_existing_props',
    description: 'Read props (items/clues) already extracted for this episode, for dedup when saving new props.',
    inputSchema: z.object({}),
    execute: async () => {
      const links = db.select().from(schema.episodeProps).where(eq(schema.episodeProps.episodeId, episodeId)).all()
      if (!links.length) return { props: [] }
      const props = links
        .map((l) => {
          const [p] = db.select().from(schema.propTemplates).where(eq(schema.propTemplates.id, l.propId)).all()
          return p && !p.deletedAt
            ? { id: p.id, name: p.name, category: p.category, description: p.description, holder: p.holder, keyClue: p.keyClue }
            : null
        })
        .filter((p): p is NonNullable<typeof p> => p != null)
      return { props }
    },
  })

  const saveDedupProps = createTool({
    id: 'save_dedup_props',
    description: 'Save extracted props (items/clues/treasure/props) with deduplication by name within the drama. All are linked to the current episode.',
    inputSchema: z.object({
      props: z.array(z.object({
        name: z.string(),
        category: z.string().optional(),
        description: z.string().optional(),
        appearance: z.string().optional(),
        size_hint: z.string().optional(),
        holder: z.string().optional(),
        key_clue: z.string().optional(),
        image_prompt: z.string().optional(),
        negative_prompt: z.string().optional(),
      })),
    }),
    execute: async ({ props }) => {
      const ts = now()
      const results = { created: 0, reused: 0 }
      logTaskProgress('ExtractTool', 'save-props-begin', {
        episodeId,
        dramaId,
        props: props.map((p) => p.name).join(','),
      })

      for (const prop of props) {
        const existing = db.select().from(schema.propTemplates)
          .where(eq(schema.propTemplates.dramaId, dramaId)).all()
          .filter((p) => !p.deletedAt)
          .find((p) => p.name === prop.name)

        const fallbackPrompt = buildPropImagePrompt({
          name: prop.name,
          category: prop.category,
          description: prop.description,
          appearance: prop.appearance,
          sizeHint: prop.size_hint,
          holder: prop.holder,
        })

        if (existing) {
          db.update(schema.propTemplates).set({
            category: prop.category || existing.category || '道具',
            description: prop.description || existing.description,
            appearance: prop.appearance || existing.appearance,
            sizeHint: prop.size_hint || existing.sizeHint,
            holder: prop.holder || existing.holder,
            keyClue: prop.key_clue || existing.keyClue,
            customPrompt: prop.image_prompt || existing.customPrompt || fallbackPrompt,
            negativePrompt: prop.negative_prompt || existing.negativePrompt || '',
            updatedAt: ts,
          }).where(eq(schema.propTemplates.id, existing.id)).run()
          linkEpisodeToProp(episodeId, existing.id)
          results.reused++
        } else {
          const res = db.insert(schema.propTemplates).values({
            dramaId,
            name: prop.name,
            category: prop.category || '道具',
            description: prop.description || '',
            appearance: prop.appearance || '',
            sizeHint: prop.size_hint || '',
            holder: prop.holder || '',
            keyClue: prop.key_clue || '否',
            customPrompt: prop.image_prompt || fallbackPrompt,
            negativePrompt: prop.negative_prompt || '',
            createdAt: ts,
            updatedAt: ts,
          }).run()
          const propId = Number(res.lastInsertRowid)
          linkEpisodeToProp(episodeId, propId)
          results.created++
        }
      }

      const payload = {
        message: `物品保存完成：新增 ${results.created}，复用已有 ${results.reused}`,
        ...results,
      }
      logTaskSuccess('ExtractTool', 'save-props-complete', { episodeId, ...results })
      return payload
    },
  })

  return {
    readScriptForExtraction,
    readExistingCharacters,
    readExistingScenes,
    saveDedupCharacters,
    saveDedupScenes,
    readExistingProps,
    saveDedupProps,
  }
}
