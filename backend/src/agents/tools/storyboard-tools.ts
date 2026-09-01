/**
 * 分镜拆解 Agent 工具
 * 工厂函数模式 — 注入 episodeId + dramaId
 */
import { createTool } from '@mastra/core/tools'
import { z } from 'zod'
import { db, schema } from '../../db/index.js'
import { and, eq, isNull } from 'drizzle-orm'
import { now } from '../../utils/response.js'
import { logTaskProgress, logTaskSuccess } from '../../utils/task-logger.js'
import { sliceLongText } from '../../utils/text-slice.js'

function syncStoryboardCharacters(storyboardId: number, characterIds: number[]) {
  db.delete(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .run()

  const uniqueIds = [...new Set(characterIds.filter(Boolean))]
  if (!uniqueIds.length) return

  for (const characterId of uniqueIds) {
    db.insert(schema.storyboardCharacters).values({
      storyboardId,
      characterId,
    }).run()
  }
}

export function syncStoryboardProps(storyboardId: number, propIds: number[]) {
  db.delete(schema.storyboardProps)
    .where(eq(schema.storyboardProps.storyboardId, storyboardId))
    .run()

  const uniqueIds = [...new Set(propIds.filter(Boolean))]
  if (!uniqueIds.length) return

  for (const propId of uniqueIds) {
    db.insert(schema.storyboardProps).values({
      storyboardId,
      propId,
    }).run()
  }
}

export function getStoryboardPropIds(storyboardId: number): number[] {
  return db
    .select({ propId: schema.storyboardProps.propId })
    .from(schema.storyboardProps)
    .where(eq(schema.storyboardProps.storyboardId, storyboardId))
    .all()
    .map((r) => r.propId)
}

function getEpisodeSceneIds(episodeId: number) {
  return new Set(
    db.select().from(schema.episodeScenes)
      .where(eq(schema.episodeScenes.episodeId, episodeId)).all()
      .map(link => link.sceneId),
  )
}

function getEpisodeCharacterIds(episodeId: number) {
  return new Set(
    db.select().from(schema.episodeCharacters)
      .where(eq(schema.episodeCharacters.episodeId, episodeId)).all()
      .map(link => link.characterId),
  )
}

function validateStoryboardBindings(episodeId: number, sceneId: number | null | undefined, characterIds: number[] | undefined) {
  const episodeSceneIds = getEpisodeSceneIds(episodeId)
  const episodeCharacterIds = getEpisodeCharacterIds(episodeId)

  if (sceneId != null && !episodeSceneIds.has(sceneId)) {
    throw new Error(`scene_id ${sceneId} 不属于当前集`)
  }

  const invalidCharacterIds = (characterIds || []).filter(id => !episodeCharacterIds.has(id))
  if (invalidCharacterIds.length) {
    throw new Error(`character_ids 不属于当前集: ${invalidCharacterIds.join(', ')}`)
  }
}

/** 从对白文本解析第一个「角色名：台词」前缀的说话人名字，旁白/画外音返回 null */
function extractSpeakerName(dialogue: string | undefined): string | null {
  if (!dialogue) return null
  const match = /^\s*([^\n:：]{1,12}?)[:：]/.exec(dialogue)
  if (!match) return null
  const name = match[1].trim()
  if (!name || /^(旁白|画外音|narrator)$/i.test(name)) return null
  return name
}

export function createStoryboardTools(episodeId: number, dramaId: number) {
  const readStoryboardContext = createTool({
    id: 'read_storyboard_context',
    description: 'Read the screenplay, characters, and scenes for storyboard breakdown.',
    inputSchema: z.object({}),
    execute: async () => {
      const [ep] = db.select().from(schema.episodes)
        .where(eq(schema.episodes.id, episodeId)).all()
      if (!ep) return { error: 'Episode not found' }
      const script = ep.scriptContent || ep.content
      if (!script) return { error: 'Episode has no script' }
      const slicedScript = sliceLongText(script)

      const charLinks = db.select().from(schema.episodeCharacters)
        .where(eq(schema.episodeCharacters.episodeId, episodeId)).all()
      const sceneLinks = db.select().from(schema.episodeScenes)
        .where(eq(schema.episodeScenes.episodeId, episodeId)).all()

      const linkedCharacterIds = new Set(charLinks.map(link => link.characterId))
      const linkedSceneIds = new Set(sceneLinks.map(link => link.sceneId))

      const chars = db.select().from(schema.characters)
        .where(and(eq(schema.characters.dramaId, dramaId), isNull(schema.characters.deletedAt))).all()
      const scns = db.select().from(schema.scenes)
        .where(eq(schema.scenes.dramaId, dramaId)).all()
      const existingStoryboards = db.select().from(schema.storyboards)
        .where(eq(schema.storyboards.episodeId, episodeId)).all()

      const characters = chars
        .filter(c => !c.deletedAt)
        .filter(c => !linkedCharacterIds.size || linkedCharacterIds.has(c.id))
        .map(c => ({
          id: c.id,
          name: c.name,
          role: c.role || '',
          description: c.description || '',
          appearance: c.appearance || '',
          personality: c.personality || '',
          voice_style: c.voiceStyle || '',
          speaker_id: c.speakerId || '',
          costume_id: c.costumeId || '',
          image_url: c.imageUrl || '',
          reference_images: c.referenceImages || '',
          // 重要提示：生成 image_prompt 和 video_prompt 时，必须用 appearance 确保角色视觉一致
          appearance_hint: `${c.name}: ${c.appearance || c.description || c.role || ''}`,
        }))

      const scenes = scns
        .filter(s => !s.deletedAt)
        .filter(s => !linkedSceneIds.size || linkedSceneIds.has(s.id))
        .map(s => ({
          id: s.id,
          location: s.location,
          location_id: s.locationId || '',
          time: s.time,
          prompt: s.prompt || '',
          image_url: s.imageUrl || '',
          storyboard_count: s.storyboardCount || 0,
        }))

      const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
      const payload = {
        style_id: drama?.styleId || '',
        episode: {
          id: ep.id,
          title: ep.title,
          episode_number: ep.episodeNumber,
          description: ep.description || '',
        },
        script: slicedScript.text,
        script_truncated: slicedScript.truncated,
        script_total_chars: slicedScript.total_chars,
        characters,
        scenes,
        existing_storyboards: existingStoryboards
          .filter(sb => !sb.deletedAt)
          .map(sb => ({
            id: sb.id,
            shot_number: sb.storyboardNumber,
            title: sb.title || '',
            scene_id: sb.sceneId,
            character_ids: db.select().from(schema.storyboardCharacters)
              .where(eq(schema.storyboardCharacters.storyboardId, sb.id)).all()
              .map(link => link.characterId),
            shot_type: sb.shotType || '',
            duration: sb.duration || 0,
            start_state: sb.startState || '',
            end_state: sb.endState || '',
          })),
        continuity_states: db.select().from(schema.continuityStates)
          .where(and(eq(schema.continuityStates.episodeId, episodeId)))
          .orderBy(schema.continuityStates.id)
          .all()
          .map(st => ({
            state_type: st.stateType,
            entity_key: st.entityKey,
            state_value: st.stateValue,
            constraints: st.constraints || '',
            storyboard_id: st.storyboardId,
          })),
        props: db.select().from(schema.episodeProps)
          .where(eq(schema.episodeProps.episodeId, episodeId))
          .all()
          .map(link => {
            const [p] = db.select().from(schema.propTemplates)
              .where(eq(schema.propTemplates.id, link.propId)).all()
            return p && !p.deletedAt
              ? { id: p.id, name: p.name, category: p.category, appearance: p.appearance, holder: p.holder }
              : null
          })
          .filter((p): p is NonNullable<typeof p> => p != null),
      }
      logTaskSuccess('StoryboardTool', 'read-context', {
        episodeId,
        dramaId,
        characters: characters.length,
        scenes: scenes.length,
        existingStoryboards: payload.existing_storyboards.length,
        scriptLength: script.length,
      })
      return payload
    },
  })

  const saveContinuityStates = createTool({
    id: 'save_continuity_states',
    description:
      'Save/replace the persistent continuity states (scene space layout, character pose, prop state, clue reveal) for this episode. ' +
      'Call after save_storyboards to lock cross-shot consistency. Replaces all previous states (idempotent).',
    inputSchema: z.object({
      states: z.array(z.object({
        state_type: z.string(),
        entity_key: z.string(),
        state_value: z.string(),
        constraints: z.string().optional(),
        storyboard_id: z.number().nullable().optional(),
        scene_id: z.number().nullable().optional(),
      })),
    }),
    execute: async ({ states }) => {
      const ts = now()
      db.delete(schema.continuityStates).where(eq(schema.continuityStates.episodeId, episodeId)).run()
      for (const st of states) {
        db.insert(schema.continuityStates).values({
          episodeId,
          storyboardId: st.storyboard_id ?? null,
          sceneId: st.scene_id ?? null,
          stateType: st.state_type,
          entityKey: st.entity_key,
          stateValue: st.state_value,
          constraints: st.constraints || '',
          createdAt: ts,
          updatedAt: ts,
        }).run()
      }
      logTaskSuccess('StoryboardTool', 'save-continuity', {
        episodeId,
        dramaId,
        count: states.length,
        types: [...new Set(states.map((s) => s.state_type))].join(','),
      })
      return { message: `Saved ${states.length} continuity states`, count: states.length }
    },
  })

  const saveStoryboards = createTool({
    id: 'save_storyboards',
    description: 'Save generated storyboards. Replaces all existing storyboards for this episode.',
    inputSchema: z.object({
      storyboards: z.array(z.object({
        shot_number: z.number(),
        title: z.string().optional(),
        shot_type: z.string().optional(),
        angle: z.string().optional(),
        movement: z.string().optional(),
        location: z.string().optional(),
        time: z.string().optional(),
        action: z.string().optional(),
        dialogue: z.string().optional(),
        description: z.string().optional(),
        result: z.string().optional(),
        atmosphere: z.string().optional(),
        image_prompt: z.string().optional(),
        first_frame_prompt: z.string().optional(),
        last_frame_prompt: z.string().optional(),
        video_prompt: z.string().optional(),
        negative_prompt: z.string().optional(),
        bgm_prompt: z.string().optional(),
        sound_effect: z.string().optional(),
        duration: z.number().optional(),
        scene_id: z.number().nullable().optional(),
        character_ids: z.array(z.number()).optional(),
        scene_type: z.string().optional(),
        speaker_id: z.string().optional(),
        start_state: z.string().optional(),
        end_state: z.string().optional(),
        constraints: z.string().optional(),
        transition_motive: z.string().optional(),
        // 关键帧（中段）：锁定动作/道具/机位的中间状态，供视频生成参考
        keyframe_prompt: z.string().optional(),
        prop_ids: z.array(z.number()).optional(),
      })),
    }),
    execute: async ({ storyboards }) => {
      const ts = now()
      logTaskProgress('StoryboardTool', 'save-begin', {
        episodeId,
        dramaId,
        count: storyboards.length,
        shotNumbers: storyboards.map(sb => sb.shot_number).join(','),
      })
      const existingStoryboardIds = db.select().from(schema.storyboards)
        .where(eq(schema.storyboards.episodeId, episodeId)).all()
        .map(sb => sb.id)
      for (const storyboardId of existingStoryboardIds) {
        db.delete(schema.storyboardCharacters)
          .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
          .run()
      }
      db.delete(schema.storyboards).where(eq(schema.storyboards.episodeId, episodeId)).run()

      let totalDuration = 0
      let dialogueIssueCount = 0
      const episodeCharNames = new Set<string>()
      const nameToSpeaker = new Map<string, string>()
      for (const link of db.select().from(schema.episodeCharacters)
        .where(eq(schema.episodeCharacters.episodeId, episodeId)).all()) {
        const [char] = db.select().from(schema.characters)
          .where(and(eq(schema.characters.id, link.characterId), isNull(schema.characters.deletedAt))).all()
        if (char?.name) {
          episodeCharNames.add(char.name)
          if (char.speakerId) nameToSpeaker.set(char.name, char.speakerId)
        }
      }

      for (const sb of storyboards) {
        validateStoryboardBindings(episodeId, sb.scene_id, sb.character_ids)

        // 说话人绑定闭环：speaker_id 未填时，从对白「角色名：台词」自动解析并绑定说话人
        let resolvedSpeakerId = sb.speaker_id || ''
        if (!resolvedSpeakerId && sb.dialogue) {
          const speakerName = extractSpeakerName(sb.dialogue)
          if (speakerName) {
            resolvedSpeakerId = nameToSpeaker.get(speakerName) || ''
            if (resolvedSpeakerId) {
              logTaskProgress('StoryboardTool', 'speaker-auto-bind', {
                shotNumber: sb.shot_number,
                speakerName,
                speakerId: resolvedSpeakerId,
              })
            }
          }
        }

        // 对话角色一致性校验：dialogue 中的角色名必须存在于本集角色或 character_ids 中
        if (sb.dialogue) {
          const sbCharNames = new Set(
            (sb.character_ids || []).map(id => {
              const [char] = db.select().from(schema.characters)
                .where(and(eq(schema.characters.id, id), isNull(schema.characters.deletedAt))).all()
              return char?.name || ''
            }).filter(Boolean),
          )
          const regex = /([^\n:：]{1,10}?)[:：]([^:：\n]{8,})/g
          let match
          while ((match = regex.exec(sb.dialogue)) !== null) {
            const name = match[1].trim()
            if (/^(旁白|画外音|narrator)$/i.test(name)) continue
            if (!sbCharNames.has(name) && !episodeCharNames.has(name)) {
              dialogueIssueCount++
              logTaskProgress('StoryboardTool', 'dialogue-warn', {
                shotNumber: sb.shot_number,
                unknownCharacter: name,
                availableNames: [...sbCharNames, ...episodeCharNames].join(','),
              })
            }
          }
        }
        const res = db.insert(schema.storyboards).values({
          episodeId,
          storyboardNumber: sb.shot_number,
          title: sb.title, shotType: sb.shot_type,
          angle: sb.angle, movement: sb.movement,
          location: sb.location, time: sb.time,
          action: sb.action, dialogue: sb.dialogue,
          description: sb.description, result: sb.result,
          atmosphere: sb.atmosphere, imagePrompt: sb.image_prompt,
          firstFramePrompt: sb.first_frame_prompt, lastFramePrompt: sb.last_frame_prompt,
          videoPrompt: sb.video_prompt, negativePrompt: sb.negative_prompt,
          bgmPrompt: sb.bgm_prompt, soundEffect: sb.sound_effect,
          sceneId: sb.scene_id, duration: sb.duration || 10,
          sceneType: sb.scene_type, speakerId: resolvedSpeakerId,
          startState: sb.start_state, endState: sb.end_state,
          constraints: sb.constraints,
          transitionMotive: sb.transition_motive,
          keyframePrompt: sb.keyframe_prompt,
          createdAt: ts, updatedAt: ts,
        }).run()
        syncStoryboardCharacters(Number(res.lastInsertRowid), sb.character_ids || [])
        syncStoryboardProps(Number(res.lastInsertRowid), sb.prop_ids || [])
        totalDuration += sb.duration || 10
      }

      db.update(schema.episodes)
        .set({ duration: Math.ceil(totalDuration / 60), updatedAt: ts })
        .where(eq(schema.episodes.id, episodeId)).run()

      logTaskSuccess('StoryboardTool', 'save-complete', {
        episodeId,
        count: storyboards.length,
        totalDuration,
        dialogueIssues: dialogueIssueCount,
      })
      return {
        message: `Saved ${storyboards.length} storyboards${dialogueIssueCount ? ` (${dialogueIssueCount} dialogue character mismatches detected)` : ''}`,
        count: storyboards.length,
        total_duration: totalDuration,
        dialogue_issues: dialogueIssueCount,
      }
    },
  })

  const updateStoryboard = createTool({
    id: 'update_storyboard',
    description: 'Update a specific storyboard shot.',
    inputSchema: z.object({
      storyboard_id: z.number(),
      title: z.string().optional(),
      shot_type: z.string().optional(),
      angle: z.string().optional(),
      movement: z.string().optional(),
      location: z.string().optional(),
      time: z.string().optional(),
      action: z.string().optional(),
      result: z.string().optional(),
      atmosphere: z.string().optional(),
      image_prompt: z.string().optional(),
      first_frame_prompt: z.string().optional(),
      last_frame_prompt: z.string().optional(),
      video_prompt: z.string().optional(),
      negative_prompt: z.string().optional(),
      bgm_prompt: z.string().optional(),
      sound_effect: z.string().optional(),
      description: z.string().optional(),
      dialogue: z.string().optional(),
      scene_id: z.number().nullable().optional(),
      character_ids: z.array(z.number()).optional(),
      duration: z.number().optional(),
      scene_type: z.string().nullable().optional(),
      speaker_id: z.string().nullable().optional(),
      start_state: z.string().optional(),
      end_state: z.string().optional(),
      constraints: z.string().optional(),
      transition_motive: z.string().optional(),
    }),
    execute: async ({ storyboard_id, ...fields }) => {
      const [storyboard] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, storyboard_id)).all()
      if (!storyboard) return { error: `Storyboard ${storyboard_id} not found` }
      logTaskProgress('StoryboardTool', 'update-begin', {
        episodeId,
        storyboardId: storyboard_id,
        fields: Object.keys(fields),
      })

      validateStoryboardBindings(
        episodeId,
        'scene_id' in fields ? fields.scene_id : storyboard.sceneId,
        'character_ids' in fields
          ? fields.character_ids
          : db.select().from(schema.storyboardCharacters)
              .where(eq(schema.storyboardCharacters.storyboardId, storyboard_id)).all()
              .map(link => link.characterId),
      )

      const updates: Record<string, any> = { updatedAt: now() }
      if ('title' in fields) updates.title = fields.title
      if ('shot_type' in fields) updates.shotType = fields.shot_type
      if ('angle' in fields) updates.angle = fields.angle
      if ('movement' in fields) updates.movement = fields.movement
      if ('location' in fields) updates.location = fields.location
      if ('time' in fields) updates.time = fields.time
      if ('action' in fields) updates.action = fields.action
      if ('result' in fields) updates.result = fields.result
      if ('atmosphere' in fields) updates.atmosphere = fields.atmosphere
      if ('image_prompt' in fields) updates.imagePrompt = fields.image_prompt
      if ('first_frame_prompt' in fields) updates.firstFramePrompt = fields.first_frame_prompt
      if ('last_frame_prompt' in fields) updates.lastFramePrompt = fields.last_frame_prompt
      if ('video_prompt' in fields) updates.videoPrompt = fields.video_prompt
      if ('negative_prompt' in fields) updates.negativePrompt = fields.negative_prompt
      if ('bgm_prompt' in fields) updates.bgmPrompt = fields.bgm_prompt
      if ('sound_effect' in fields) updates.soundEffect = fields.sound_effect
      if ('description' in fields) updates.description = fields.description
      if ('dialogue' in fields) updates.dialogue = fields.dialogue
      if ('scene_id' in fields) updates.sceneId = fields.scene_id
      if ('duration' in fields) updates.duration = fields.duration
      if ('scene_type' in fields) updates.sceneType = fields.scene_type
      if ('speaker_id' in fields) updates.speakerId = fields.speaker_id
      if ('start_state' in fields) updates.startState = fields.start_state
      if ('end_state' in fields) updates.endState = fields.end_state
      if ('constraints' in fields) updates.constraints = fields.constraints
      if ('transition_motive' in fields) updates.transitionMotive = fields.transition_motive
      if ('keyframe_prompt' in fields) updates.keyframePrompt = fields.keyframe_prompt
      db.update(schema.storyboards).set(updates).where(eq(schema.storyboards.id, storyboard_id)).run()
      if ('character_ids' in fields) syncStoryboardCharacters(storyboard_id, fields.character_ids || [])
      logTaskSuccess('StoryboardTool', 'update-complete', {
        episodeId,
        storyboardId: storyboard_id,
        updatedFields: Object.keys(updates),
        characterIds: 'character_ids' in fields ? (fields.character_ids || []).join(',') : undefined,
      })
      return { message: `Storyboard ${storyboard_id} updated` }
    },
  })

  return { readStoryboardContext, saveStoryboards, updateStoryboard, saveContinuityStates }
}
