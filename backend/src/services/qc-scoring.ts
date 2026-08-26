import { and, eq, isNull } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { now } from '../utils/response.js'
import { logTaskSuccess, logTaskWarn } from '../utils/task-logger.js'

interface QcIssue {
  dimension: 'lip_sync' | 'character_consistency' | 'continuity'
  severity: 'error' | 'warning' | 'info'
  message: string
}

interface QcDimension {
  score: number
  checked: boolean
  notes: string[]
}

function clamp(n: number): number {
  return Math.max(0, Math.min(100, Math.round(n)))
}

/**
 * 镜头级 QC 打分（启发式规则 + 元数据一致性）
 * - lip_sync：对白配音（tts_audio_url）、speaker 声音锁定、视频时长偏差
 * - character_consistency：出场角色参考图 / 声音 / 服装锁定
 * - continuity：相邻分镜场景连贯、地点 ID 一致、时长合理
 *
 * 说明：真实唇形对齐、角色相似度需 AI 检测模型（后续可挂载），
 * 此处为可落地的规则分，issues 供人工复核。
 */
export function scoreStoryboard(storyboardId: number): any {
  const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, storyboardId)).all()
  if (!sb) return { error: 'Storyboard not found' }

  const issues: QcIssue[] = []
  const dims: Record<'lip_sync' | 'character_consistency' | 'continuity', QcDimension> = {
    lip_sync: { score: 0, checked: true, notes: [] },
    character_consistency: { score: 0, checked: true, notes: [] },
    continuity: { score: 0, checked: true, notes: [] },
  }

  const links = db
    .select()
    .from(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .all()
  const chars = links
    .map((l) =>
      db
        .select()
        .from(schema.characters)
        .where(and(eq(schema.characters.id, l.characterId), isNull(schema.characters.deletedAt)))
        .all()[0],
    )
    .filter(Boolean)

  const scene = sb.sceneId
    ? db.select().from(schema.scenes).where(eq(schema.scenes.id, sb.sceneId)).all()[0]
    : undefined
  const ep = sb.episodeId
    ? db.select().from(schema.episodes).where(eq(schema.episodes.id, sb.episodeId)).all()[0]
    : undefined

  const gens = db
    .select()
    .from(schema.videoGenerations)
    .where(and(eq(schema.videoGenerations.storyboardId, storyboardId), isNull(schema.videoGenerations.deletedAt)))
    .all()
  const completedGen = gens.find((g) => g.status === 'completed' && !!g.videoUrl)
  const hasDialogue = !!sb.dialogue && sb.dialogue.trim().length > 0

  // ===== 1. 唇形同步 / 音画同步 =====
  if (!hasDialogue) {
    dims.lip_sync.score = 90
    dims.lip_sync.notes.push('无对白镜头，不涉及唇形同步')
  } else {
    let score = 100
    if (!sb.ttsAudioUrl) {
      score -= 40
      issues.push({ dimension: 'lip_sync', severity: 'error', message: '有对白但未生成 TTS 音频（tts_audio_url 为空）' })
    }
    if (!completedGen) {
      score -= 30
      issues.push({ dimension: 'lip_sync', severity: 'warning', message: '有对白但无已完成的视频生成记录' })
    }
    if (sb.speakerId) {
      const speakerChar = chars.find((c) => c.speakerId === sb.speakerId)
      if (speakerChar && !speakerChar.voiceStyle) {
        score -= 15
        issues.push({ dimension: 'lip_sync', severity: 'warning', message: `speaker ${sb.speakerId} 未锁定声音资产（voice_style 为空）` })
      }
    } else if (chars.length > 0) {
      score -= 10
      issues.push({ dimension: 'lip_sync', severity: 'warning', message: '有对白但分镜未指定 speaker_id' })
    }
    if (completedGen?.duration && sb.duration) {
      const drift = Math.abs(completedGen.duration - sb.duration) / sb.duration
      if (drift > 0.3) {
        score -= 20
        issues.push({ dimension: 'lip_sync', severity: 'warning', message: `视频时长 ${completedGen.duration}s 与分镜 ${sb.duration}s 偏差 ${Math.round(drift * 100)}%` })
      }
    }
    dims.lip_sync.score = clamp(score)
    dims.lip_sync.notes.push(`对白${sb.ttsAudioUrl ? '已' : '未'}配音`)
  }

  // ===== 2. 角色一致性 =====
  if (chars.length === 0) {
    dims.character_consistency.score = 95
    dims.character_consistency.notes.push('无出场角色（空镜/环境），不涉及角色一致性')
  } else {
    let score = 100
    for (const c of chars) {
      const hasRef = !!c.imageUrl || !!c.referenceImages
      if (!hasRef) {
        score -= 25
        issues.push({ dimension: 'character_consistency', severity: 'error', message: `角色「${c.name}」无参考图（image_url / reference_images 为空）` })
      }
      if (!c.voiceStyle) {
        score -= 10
        issues.push({ dimension: 'character_consistency', severity: 'warning', message: `角色「${c.name}」未锁定声音（voice_style 为空）` })
      }
      if (!c.costumeId) {
        score -= 5
        issues.push({ dimension: 'character_consistency', severity: 'info', message: `角色「${c.name}」未锁定服装（costume_id 为空）` })
      }
    }
    if (completedGen && !completedGen.referenceMode && !completedGen.imageUrl && !completedGen.referenceImageUrls) {
      score -= 15
      issues.push({ dimension: 'character_consistency', severity: 'warning', message: '视频生成未使用参考图（无 reference / image）' })
    }
    dims.character_consistency.score = clamp(score)
    dims.character_consistency.notes.push(`出场角色 ${chars.map((c) => c.name).join('、')}`)
  }

  // ===== 3. 连续性 =====
  {
    let score = 100
    if (sb.duration && (sb.duration < 2 || sb.duration > 30)) {
      score -= 10
      issues.push({ dimension: 'continuity', severity: 'warning', message: `分镜时长 ${sb.duration}s 超出合理范围（2–30s）` })
    }
    const siblings = db
      .select()
      .from(schema.storyboards)
      .where(and(eq(schema.storyboards.episodeId, sb.episodeId), isNull(schema.storyboards.deletedAt)))
      .all()
      .sort((a, b) => a.storyboardNumber - b.storyboardNumber)
    const idx = siblings.findIndex((s) => s.id === sb.id)
    if (idx > 0) {
      const prev = siblings[idx - 1]
      if (sb.sceneId && prev.sceneId === sb.sceneId) {
        const prevScene = db.select().from(schema.scenes).where(eq(schema.scenes.id, prev.sceneId)).all()[0]
        if (prevScene && scene && prevScene.locationId && scene.locationId && prevScene.locationId !== scene.locationId) {
          score -= 20
          issues.push({ dimension: 'continuity', severity: 'error', message: `同一场景 #${sb.sceneId} 但地点 ID 不一致（${prevScene.locationId} vs ${scene.locationId}）` })
        }
      }
    }
    if (sb.sceneId && scene && !scene.locationId) {
      score -= 5
      issues.push({ dimension: 'continuity', severity: 'info', message: '关联场景未锁定地点 ID（location_id 为空）' })
    }
    dims.continuity.score = clamp(score)
  }

  const overall = clamp(dims.lip_sync.score * 0.4 + dims.character_consistency.score * 0.3 + dims.continuity.score * 0.3)
  const videoGenerationId = completedGen?.id ?? null
  const status = completedGen ? (issues.some((i) => i.severity === 'error') ? 'failed' : 'passed') : 'pending'

  const base = {
    storyboardId,
    videoGenerationId,
    dramaId: ep?.dramaId ?? null,
    episodeId: sb.episodeId,
    lipSyncScore: dims.lip_sync.score,
    characterConsistencyScore: dims.character_consistency.score,
    continuityScore: dims.continuity.score,
    overallScore: overall,
    issues: JSON.stringify(issues),
    dimensions: JSON.stringify(dims),
    status,
  }

  const ts = now()
  const existing = db
    .select()
    .from(schema.videoQualityChecks)
    .where(eq(schema.videoQualityChecks.storyboardId, storyboardId))
    .all()
    .sort((a, b) => b.id - a.id)[0]

  let saved: any
  if (existing) {
    db.update(schema.videoQualityChecks)
      .set({ ...base, updatedAt: ts })
      .where(eq(schema.videoQualityChecks.id, existing.id))
      .run()
    saved = { ...base, id: existing.id, createdAt: existing.createdAt, updatedAt: ts }
  } else {
    const res = db.insert(schema.videoQualityChecks).values({ ...base, createdAt: ts, updatedAt: ts }).run()
    saved = { ...base, id: Number(res.lastInsertRowid), createdAt: ts, updatedAt: ts }
  }

  logTaskSuccess('QcScoring', 'score-complete', {
    storyboardId,
    overall,
    status,
    issueCount: issues.length,
  })
  return { ...saved, issues, dimensions: dims }
}

/** 视频生成完成后自动触发 QC（fire-and-forget） */
export function runQcAfterVideoComplete(storyboardId: number, videoGenerationId: number): void {
  try {
    scoreStoryboard(storyboardId)
  } catch (err: any) {
    logTaskWarn('QcScoring', 'auto-score-failed', {
      storyboardId,
      videoGenerationId,
      error: err?.message || String(err),
    })
  }
}
