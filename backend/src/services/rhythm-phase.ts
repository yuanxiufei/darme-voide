/**
 * 多集节奏相位（Multi-Episode Rhythm Phase）
 *
 * 对齐参考项目 multi-episode rhythm phase：把每集的分镜按其在集内的时长位置
 * 自动划分到四类节奏相位，跨集统计节奏分布，帮助 storyboard_breaker 保持节奏一致性。
 *
 * 相位划分（按累计时长占比）：
 *  - setup       0%–25%：建置，引入人物/场景/冲突前提
 *  - development 25%–70%：推进，矛盾升级/伏笔展开
 *  - climax      70%–90%：高潮，冲突爆发
 *  - resolution  90%–100%：收束，落点/钩子
 *
 * 用法：
 *  1. save_storyboards 重建分镜后调用 assignRhythmPhases(episodeId) 自动打相位
 *  2. storyboard_breaker 注入 rhythm_guidance_for_episode() 生成的跨集节奏引导
 */
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'

export type RhythmPhase = 'setup' | 'development' | 'climax' | 'resolution'

export const RHYTHM_PHASES: RhythmPhase[] = ['setup', 'development', 'climax', 'resolution']
export const RHYTHM_PHASE_LABELS: Record<RhythmPhase, string> = {
  setup: '建置',
  development: '推进',
  climax: '高潮',
  resolution: '收束',
}

/** 按累计时长占比返回相位 */
export function phaseForPosition(ratio: number): RhythmPhase {
  if (ratio < 0.25) return 'setup'
  if (ratio < 0.70) return 'development'
  if (ratio < 0.90) return 'climax'
  return 'resolution'
}

/** 为某集所有分镜分配节奏相位（按累计时长占比；无时长数据则按序号占比） */
export function assignRhythmPhases(episodeId: number): number {
  const sbs = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, episodeId))
    .orderBy(schema.storyboards.storyboardNumber).all()
  if (!sbs.length) return 0

  const totalDuration = sbs.reduce((sum, s) => sum + (s.duration ?? 0), 0)
  let cumulative = 0
  let assigned = 0
  for (const sb of sbs) {
    let ratio: number
    if (totalDuration > 0) {
      ratio = cumulative / totalDuration
      cumulative += sb.duration ?? 0
    } else {
      ratio = (sb.storyboardNumber - 1) / sbs.length
    }
    const phase = phaseForPosition(ratio)
    db.update(schema.storyboards)
      .set({ rhythmPhase: phase })
      .where(eq(schema.storyboards.id, sb.id)).run()
    assigned++
  }
  return assigned
}

export interface EpisodeRhythm {
  episode_id: number
  episode_number: number
  title: string
  total_duration: number
  storyboard_count: number
  phases: Partial<Record<RhythmPhase, number>>
  // 高潮段分镜编号（前若干条），用于 QC 重点抽查
  climax_storyboards: number[]
  complete: boolean // 四相位是否都覆盖（>=0.9 时长占比场景可容忍 resolution 缺省）
}

export interface DramaRhythmReport {
  drama_id: number
  total_episodes: number
  total_duration: number
  episodes: EpisodeRhythm[]
  // 跨集节奏一致性：每集四相位平均占比方差较小说明节奏稳定
  phase_balance: Partial<Record<RhythmPhase, number>>
  guidance: string
}

/** 统计某集节奏分布 */
export function getEpisodeRhythm(episodeId: number): EpisodeRhythm | null {
  const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, episodeId)).all()
  if (!ep) return null
  const sbs = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, episodeId))
    .orderBy(schema.storyboards.storyboardNumber).all()

  const phases: Partial<Record<RhythmPhase, number>> = {}
  const climaxSbs: number[] = []
  let totalDuration = 0
  for (const sb of sbs) {
    totalDuration += sb.duration ?? 0
    const phase = (sb.rhythmPhase as RhythmPhase) || phaseForPosition(0.5)
    phases[phase] = (phases[phase] ?? 0) + 1
    if (phase === 'climax') climaxSbs.push(sb.storyboardNumber)
  }

  return {
    episode_id: episodeId,
    episode_number: ep.episodeNumber,
    title: ep.title,
    total_duration: totalDuration,
    storyboard_count: sbs.length,
    phases,
    climax_storyboards: climaxSbs.slice(0, 5),
    complete: RHYTHM_PHASES.every(p => (phases[p] ?? 0) > 0),
  }
}

/** 生成跨集节奏引导文本（注入 storyboard_breaker） */
export function rhythmGuidanceForEpisode(episodeId: number): string {
  const [ep] = db.select().from(schema.episodes).where(eq(schema.episodes.id, episodeId)).all()
  if (!ep) return ''
  const dramaRhythm = getDramaRhythm(ep.dramaId)
  const prev = dramaRhythm.episodes
    .filter(e => e.episode_number < ep.episodeNumber)
    .sort((a, b) => b.episode_number - a.episode_number)[0]

  const lines: string[] = []
  lines.push('【多集节奏相位要求】')
  lines.push('请将本集分镜按节奏划分为四相：setup(建置,前25%) / development(推进,25%-70%) / climax(高潮,70%-90%) / resolution(收束,末10%)。')

  if (prev) {
    const prevPhases = RHYTHM_PHASES.map(p => `${RHYTHM_PHASE_LABELS[p]}${prev.phases[p] ?? 0}镜`).join('、')
    lines.push(`上一集（第${prev.episode_number}集）节奏分布：${prevPhases}，总时长${Math.round(prev.total_duration)}秒。请保持节奏基调一致。`)
  }

  const avg = dramaRhythm.phase_balance
  if (dramaRhythm.episodes.length > 0) {
    lines.push(`该剧各集平均节奏：` +
      RHYTHM_PHASES.map(p => `${RHYTHM_PHASE_LABELS[p]}占比${Math.round((avg[p] ?? 0) * 100)}%`).join('、') +
      `。若本集为高潮集，可加大 climax 占比。`)
  }

  lines.push('若上一集以悬念/钩子收尾，本集开局需回应；若上一集是普通推进，本集结尾应埋下钩子。')
  return lines.join('\n')
}

/** 生成全剧节奏报告 */
export function getDramaRhythm(dramaId: number): DramaRhythmReport {
  const eps = db.select().from(schema.episodes)
    .where(eq(schema.episodes.dramaId, dramaId))
    .orderBy(schema.episodes.episodeNumber).all()

  const episodeRhythms = eps.map(ep => getEpisodeRhythm(ep.id)).filter(Boolean) as EpisodeRhythm[]

  // 平均占比（按故事板数量权重）
  const totalSbs = episodeRhythms.reduce((s, e) => s + e.storyboard_count, 0)
  const phaseBalance: Partial<Record<RhythmPhase, number>> = {}
  if (totalSbs > 0) {
    for (const p of RHYTHM_PHASES) {
      const count = episodeRhythms.reduce((s, e) => s + (e.phases[p] ?? 0), 0)
      phaseBalance[p] = count / totalSbs
    }
  }

  const totalDuration = episodeRhythms.reduce((s, e) => s + e.total_duration, 0)
  return {
    drama_id: dramaId,
    total_episodes: episodeRhythms.length,
    total_duration: totalDuration,
    episodes: episodeRhythms,
    phase_balance: phaseBalance,
    guidance: rhythmGuidanceForEpisode(eps[eps.length - 1]?.id ?? 0),
  }
}
