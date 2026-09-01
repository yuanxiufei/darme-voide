/**
 * 剧本内容指纹门禁（Script Content Fingerprint Gate）
 *
 * 剧本（episodes.script_content）是生成链路源头：script_rewriter → extractor
 * → voice_assigner → storyboard_breaker → 媒体生成。剧本一旦被修改，基于旧剧本
 * 生成的分镜/图片/视频在语义上全部过期（stale）。
 *
 * 机制（对齐 H3-Codex-Drama asset gate）：
 *  - `episodes.script_hash`：当前剧本内容的 sha256 指纹（剧本写入时自动重算）
 *  - `storyboards.script_hash`：该分镜生成时对应的剧本指纹（save_storyboards 写入）
 *  - 门禁检查：媒体生成前比较两者，不一致则阻断并提示重新拆解分镜；
 *    支持 force 显式放行（用户确认「我就是要基于旧分镜继续」）。
 *
 * 指纹用 sha256 摘要的截断值即可，碰撞概率可忽略；空剧本返回 null（不设门禁）。
 */
import { createHash } from 'node:crypto'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { now } from '../utils/response.js'

/** 取剧本指纹内容源：优先 script_content，回退 content */
export function getEpisodeScriptSource(ep: { scriptContent: string | null; content: string | null }): string {
  return (ep.scriptContent ?? ep.content ?? '').trim()
}

/** 计算剧本内容指纹（sha256 前 16 位十六进制） */
export function computeScriptHash(content: string): string | null {
  const normalized = content.replace(/\r\n/g, '\n').trim()
  if (!normalized) return null
  return createHash('sha256').update(normalized, 'utf8').digest('hex').slice(0, 16)
}

/** 重算某集当前剧本指纹并写回 episodes.script_hash；返回新指纹（空剧本返回 null） */
export function refreshEpisodeScriptHash(episodeId: number): string | null {
  const [ep] = db.select().from(schema.episodes)
    .where(eq(schema.episodes.id, episodeId)).all()
  if (!ep) return null
  const hash = computeScriptHash(getEpisodeScriptSource(ep))
  db.update(schema.episodes)
    .set({ scriptHash: hash, updatedAt: now() })
    .where(eq(schema.episodes.id, episodeId)).run()
  return hash
}

/** 取某集当前剧本指纹（不重算） */
export function getEpisodeScriptHash(episodeId: number): string | null {
  const [ep] = db.select().from(schema.episodes)
    .where(eq(schema.episodes.id, episodeId)).all()
  if (!ep) return null
  return ep.scriptHash ?? computeScriptHash(getEpisodeScriptSource(ep))
}

/** 把当前剧本指纹写入该集全部分镜（save_storyboards 后调用） */
export function stampStoryboardsScriptHash(episodeId: number): void {
  const hash = getEpisodeScriptHash(episodeId)
  const sbs = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, episodeId)).all()
  for (const sb of sbs) {
    db.update(schema.storyboards)
      .set({ scriptHash: hash, updatedAt: now() })
      .where(eq(schema.storyboards.id, sb.id)).run()
  }
}

export interface StoryboardStaleInfo {
  id: number
  storyboard_number: number
  has_assets: boolean // composedImage / videoUrl / composedVideoUrl 任一存在
}

export interface EpisodeFingerprintStatus {
  episode_id: number
  current_script_hash: string | null
  has_script: boolean
  storyboard_count: number
  stale_count: number
  stale_storyboards: StoryboardStaleInfo[]
  stale: boolean
  stale_with_assets: boolean // 有旧资产且剧本已变（最危险，需强提示）
  message: string
}

/** 检查某集剧本指纹状态：当前指纹 vs 各分镜指纹 */
export function checkEpisodeFingerprint(episodeId: number): EpisodeFingerprintStatus {
  const [ep] = db.select().from(schema.episodes)
    .where(eq(schema.episodes.id, episodeId)).all()
  if (!ep) {
    return {
      episode_id: episodeId,
      current_script_hash: null,
      has_script: false,
      storyboard_count: 0,
      stale_count: 0,
      stale_storyboards: [],
      stale: false,
      stale_with_assets: false,
      message: 'Episode not found',
    }
  }

  const currentHash = ep.scriptHash ?? computeScriptHash(getEpisodeScriptSource(ep))
  const sbs = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, episodeId)).all()

  const staleSbs: StoryboardStaleInfo[] = []
  if (currentHash) {
    for (const sb of sbs) {
      // 分镜无指纹视为「门禁启用前产物」，不计为过期（保持兼容）
      if (sb.scriptHash && sb.scriptHash !== currentHash) {
        staleSbs.push({
          id: sb.id,
          storyboard_number: sb.storyboardNumber,
          has_assets: !!(sb.composedImage || sb.videoUrl || sb.composedVideoUrl),
        })
      }
    }
  }

  const staleWithAssets = staleSbs.some(s => s.has_assets)
  return {
    episode_id: episodeId,
    current_script_hash: currentHash,
    has_script: !!currentHash,
    storyboard_count: sbs.length,
    stale_count: staleSbs.length,
    stale_storyboards: staleSbs,
    stale: staleSbs.length > 0,
    stale_with_assets: staleWithAssets,
    message: staleSbs.length > 0
      ? `剧本已变更：${staleSbs.length}/${sbs.length} 个分镜基于旧剧本，需重新拆解分镜`
      : `剧本指纹一致：${sbs.length} 个分镜与当前剧本匹配`,
  }
}

/** 单分镜门禁检查：媒体生成前调用 */
export function checkStoryboardGate(storyboardId: number): { allowed: boolean; reason?: string } {
  const [sb] = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.id, storyboardId)).all()
  if (!sb) return { allowed: false, reason: 'Storyboard not found' }
  const currentHash = getEpisodeScriptHash(sb.episodeId)
  // 无剧本指纹或分镜无指纹 → 不设门禁
  if (!currentHash || !sb.scriptHash) return { allowed: true }
  if (sb.scriptHash !== currentHash) {
    return {
      allowed: false,
      reason: `分镜基于旧剧本生成（剧本已变更），请先重新拆解分镜。如确认继续请传 force=true。`,
    }
  }
  return { allowed: true }
}
