/**
 * 工程账本（Project Ledger）——对齐参考项目 H3-Codex-Drama「可复现工程（project.yaml）」
 *
 * 将整剧的制作全链路落成一份可复现账本：剧本哈希、分镜提示词、逐镜路由决策、
 * 输入资产引用、生成任务（image/video 的 prompt/model/status）、资产版本历史、成本明细。
 *
 * 用途：
 * 1. 可复现导出：拿到账本即可知道「每镜怎么生成的、用了什么提示词/参考图/路由、花了多少」。
 * 2. 断点续作：以 script_hash 等指纹比对剧本是否变更，定位 stale 分镜（配合 script-fingerprint）。
 * 3. 审计交付：生成 Markdown 报告交付给制片/审计方（联系表 + 媒体信息 + 生成记录）。
 *
 * 输出两种格式：
 * - JSON：完整结构化账本（供程序消费 / 断点续作恢复）
 * - Markdown：人类可读报告（供制片/审计交付）
 */
import { db, schema } from '../db/index.js'
import { eq, asc } from 'drizzle-orm'
import { getEpisodeCostBoard, type EpisodeCostBoard } from './usage-tracking.js'

// ============================================================
// 账本结构
// ============================================================

export interface LedgerAssetRef {
  kind: 'storyboard' | 'character' | 'scene' | 'prop'
  id: number
  url: string | null
}

export interface LedgerShotRecord {
  storyboardId: number
  storyboardNumber: number
  title: string | null
  sceneType: string | null
  shotType: string | null
  duration: number | null
  rhythmPhase: string | null
  /** 逐镜路由决策（T2V / I2V / FL2VA / R2V / K-Frame） */
  route: string | null
  routeReason: string | null
  /** 剧本指纹（分镜生成时对应的 script_hash） */
  scriptHash: string | null
  /** 输入提示词（生成时快照） */
  imagePrompt: string | null
  videoPrompt: string | null
  /** 输入资产引用 */
  firstFrameImage: string | null
  lastFrameImage: string | null
  keyframeImage: string | null
  referenceImages: string | null
  referenceAudioUrls: string | null
  /** 产出 */
  imageGenerations: Array<{
    id: number
    provider: string | null
    model: string | null
    prompt: string | null
    frameType: string | null
    status: string | null
    imageUrl: string | null
    createdAt: string
  }>
  videoGenerations: Array<{
    id: number
    provider: string | null
    model: string | null
    prompt: string | null
    referenceMode: string | null
    route: string | null
    routeReason: string | null
    status: string | null
    videoUrl: string | null
    blockReason: string | null
    createdAt: string
  }>
  composedVideoUrl: string | null
}

export interface LedgerEpisodeRecord {
  id: number
  episodeNumber: number
  title: string
  status: string | null
  scriptHash: string | null
  duration: number | null
  videoUrl: string | null
  bgmUrl: string | null
  imageConfigId: number | null
  videoConfigId: number | null
  audioConfigId: number | null
  cost: EpisodeCostBoard['episodes'][number] | null
  shots: LedgerShotRecord[]
}

export interface ProjectLedger {
  /** 账本元信息 */
  schemaVersion: 1
  exportedAt: string
  /** 剧集元信息 */
  drama: {
    id: number
    title: string
    genre: string | null
    style: string | null
    totalEpisodes: number | null
    status: string | null
  }
  /** 资产（角色/场景/道具），供账本读者了解全集资产基线 */
  assets: {
    characters: Array<{ id: number; name: string; imageUrl: string | null; voiceSampleUrl: string | null }>
    scenes: Array<{ id: number; location: string; imageUrl: string | null }>
    props: Array<{ id: number; name: string; category: string | null; imageUrl: string | null }>
  }
  episodes: LedgerEpisodeRecord[]
}

// ============================================================
// 账本构建
// ============================================================

function safeJsonArray(value: string | null): any[] {
  if (!value) return []
  try {
    const parsed = JSON.parse(value)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

/**
 * 构建整剧工程账本（JSON 结构）。
 * @param dramaId 剧集 ID
 * @param episodeId 可选：只导出指定集
 */
export function buildProjectLedger(dramaId: number, episodeId?: number): ProjectLedger {
  const ts = new Date().toISOString()

  const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
  if (!drama) throw new Error(`Drama ${dramaId} not found`)

  // 资产基线
  const characters = db.select().from(schema.characters)
    .where(eq(schema.characters.dramaId, dramaId)).all()
  const scenes = db.select().from(schema.scenes)
    .where(eq(schema.scenes.dramaId, dramaId)).all()
  const props = db.select().from(schema.propTemplates)
    .where(eq(schema.propTemplates.dramaId, dramaId)).all()

  // 剧集
  let eps = db.select().from(schema.episodes)
    .where(eq(schema.episodes.dramaId, dramaId))
    .all()
    .sort((a, b) => a.episodeNumber - b.episodeNumber)
  if (episodeId) eps = eps.filter(e => e.id === episodeId)

  // 多集成本看板（按剧聚合，供每集成本摘要）
  let costBoard: EpisodeCostBoard | null = null
  try {
    costBoard = getEpisodeCostBoard(dramaId)
  } catch {
    costBoard = null
  }
  const costByEpisode = new Map<number, EpisodeCostBoard['episodes'][number]>()
  for (const epRow of costBoard?.episodes ?? []) costByEpisode.set(epRow.episode_id, epRow)

  const episodeRecords: LedgerEpisodeRecord[] = eps.map((ep) => {
    const sbs = db.select().from(schema.storyboards)
      .where(eq(schema.storyboards.episodeId, ep.id))
      .orderBy(asc(schema.storyboards.storyboardNumber))
      .all()

    const shotRecords: LedgerShotRecord[] = sbs.map((sb) => {
      const images = db.select().from(schema.imageGenerations)
        .where(eq(schema.imageGenerations.storyboardId, sb.id))
        .all()
      const videos = db.select().from(schema.videoGenerations)
        .where(eq(schema.videoGenerations.storyboardId, sb.id))
        .all()

      return {
        storyboardId: sb.id,
        storyboardNumber: sb.storyboardNumber,
        title: sb.title,
        sceneType: sb.sceneType,
        shotType: sb.shotType,
        duration: sb.duration,
        rhythmPhase: sb.rhythmPhase,
        route: sb.route,
        routeReason: sb.routeReason,
        scriptHash: sb.scriptHash,
        imagePrompt: sb.imagePrompt,
        videoPrompt: sb.videoPrompt,
        firstFrameImage: sb.firstFrameImage,
        lastFrameImage: sb.lastFrameImage,
        keyframeImage: sb.keyframeImage,
        referenceImages: sb.referenceImages,
        referenceAudioUrls: videos.length ? videos[videos.length - 1].referenceAudioUrls : null,
        imageGenerations: images.map(g => ({
          id: g.id,
          provider: g.provider,
          model: g.model,
          prompt: g.prompt,
          frameType: g.frameType,
          status: g.status,
          imageUrl: g.imageUrl,
          createdAt: g.createdAt,
        })),
        videoGenerations: videos.map(v => ({
          id: v.id,
          provider: v.provider,
          model: v.model,
          prompt: v.prompt,
          referenceMode: v.referenceMode,
          route: v.route,
          routeReason: v.routeReason,
          status: v.status,
          videoUrl: v.videoUrl,
          blockReason: v.blockReason,
          createdAt: v.createdAt,
        })),
        composedVideoUrl: sb.composedVideoUrl,
      }
    })

    const cost = costByEpisode.get(ep.id) ?? null

    return {
      id: ep.id,
      episodeNumber: ep.episodeNumber,
      title: ep.title,
      status: ep.status,
      scriptHash: ep.scriptHash,
      duration: ep.duration,
      videoUrl: ep.videoUrl,
      bgmUrl: ep.bgmUrl,
      imageConfigId: ep.imageConfigId,
      videoConfigId: ep.videoConfigId,
      audioConfigId: ep.audioConfigId,
      cost,
      shots: shotRecords,
    }
  })

  return {
    schemaVersion: 1,
    exportedAt: ts,
    drama: {
      id: drama.id,
      title: drama.title,
      genre: drama.genre,
      style: drama.style,
      totalEpisodes: drama.totalEpisodes,
      status: drama.status,
    },
    assets: {
      characters: characters.map(c => ({ id: c.id, name: c.name, imageUrl: c.imageUrl, voiceSampleUrl: c.voiceSampleUrl })),
      scenes: scenes.map(s => ({ id: s.id, location: s.location, imageUrl: s.imageUrl })),
      props: props.map(p => ({ id: p.id, name: p.name, category: p.category, imageUrl: p.imageUrl })),
    },
    episodes: episodeRecords,
  }
}

// ============================================================
// Markdown 可复现报告
// ============================================================

const ROUTE_LABEL: Record<string, string> = {
  text_to_video: 'T2V 纯文本生成',
  first_frame_to_video: 'I2V 首帧图生视频',
  first_last_frame: 'FL2VA 首尾帧连接',
  reference_to_video: 'R2V 多参考图/参考音频',
  keyframe_to_video: 'I2V-K 关键帧参考',
  video_editor: 'Editor 视频编辑',
  blocked: '资产门禁阻断',
}

/** 路由摘要（用于 Markdown 表头） */
export function routeLabel(route: string | null): string {
  return route ? (ROUTE_LABEL[route] || route) : '—'
}

/**
 * 生成可复现 Markdown 报告（人类可读，交付制片/审计）。
 */
export function buildProjectLedgerMarkdown(dramaId: number, episodeId?: number): string {
  const ledger = buildProjectLedger(dramaId, episodeId)
  const lines: string[] = []

  lines.push(`# 工程账本 · ${ledger.drama.title}`)
  lines.push('')
  lines.push(`- 导出时间：${ledger.exportedAt}`)
  lines.push(`- 剧集类型：${ledger.drama.genre || '—'} / 风格：${ledger.drama.style || '—'}`)
  lines.push(`- 总集数：${ledger.drama.totalEpisodes ?? '—'}`)
  lines.push(`- 状态：${ledger.drama.status || '—'}`)
  lines.push('')
  lines.push('> 本报告为可复现工程账本：每镜的提示词、输入指纹、路由决策、参考资产与成本均留档，'
    + '可用于断点续作与审计。')
  lines.push('')

  // 资产基线
  lines.push('## 资产基线')
  lines.push('')
  lines.push(`- 角色：${ledger.assets.characters.length} 个`
    + `（${ledger.assets.characters.map(c => c.name).join('、') || '—'}）`)
  lines.push(`- 场景：${ledger.assets.scenes.length} 个`
    + `（${ledger.assets.scenes.map(s => s.location).join('、') || '—'}）`)
  lines.push(`- 道具：${ledger.assets.props.length} 个`
    + `（${ledger.assets.props.map(p => p.name).join('、') || '—'}）`)
  lines.push('')

  for (const ep of ledger.episodes) {
    lines.push(`## 第 ${ep.episodeNumber} 集 · ${ep.title}`)
    lines.push('')
    lines.push(`- 状态：${ep.status || '—'}　|　时长：${ep.duration ?? '—'}s　|　指纹：${ep.scriptHash || '—'}`)
    if (ep.videoUrl) lines.push(`- 成片：${ep.videoUrl}`)
    if (ep.bgmUrl) lines.push(`- BGM：${ep.bgmUrl}`)
    lines.push(`- 配置：image#${ep.imageConfigId ?? '—'} / video#${ep.videoConfigId ?? '—'} / audio#${ep.audioConfigId ?? '—'}`)

    // 成本摘要
    if (ep.cost) {
      lines.push(`- 成本：${ep.cost.total_cost ?? '—'} 元`
        + `（调用 ${ep.cost.total_calls ?? 0} 次${(ep.cost.retry_cost ?? 0) > 0 ? `，重拍 ${ep.cost.retry_cost} 元` : ''}）`)
    }
    lines.push('')

    // 分镜明细表
    lines.push('| # | 分镜 | 时长 | 相位 | 路由 | 指纹 | 视频状态 |')
    lines.push('|---|------|------|------|------|------|----------|')
    for (const shot of ep.shots) {
      const videoStatus = shot.videoGenerations
        .filter(v => v.status === 'completed' || v.status === 'processing' || v.status === 'blocked_by_missing_asset')
        .map(v => v.status)
        .join('/') || '—'
      const phase = shot.rhythmPhase || '—'
      lines.push(
        `| ${shot.storyboardNumber} | ${(shot.title || `分镜 ${shot.storyboardNumber}`).replace(/\|/g, '\\|')} `
        + `| ${shot.duration ?? '—'}s | ${phase} | ${routeLabel(shot.route)} `
        + `| ${shot.scriptHash?.slice(0, 8) || '—'} | ${videoStatus} |`
      )
    }
    lines.push('')

    // 每镜明细（提示词 + 决策原因）
    for (const shot of ep.shots) {
      lines.push(`### 分镜 ${shot.storyboardNumber}${shot.title ? ` · ${shot.title}` : ''}`)
      lines.push('')
      if (shot.routeReason) lines.push(`- **路由**：${routeLabel(shot.route)} — ${shot.routeReason}`)
      if (shot.scriptHash) lines.push(`- **指纹**：\`${shot.scriptHash}\``)
      if (shot.imagePrompt) lines.push(`- **图片提示词**：\`${shot.imagePrompt.slice(0, 240)}${shot.imagePrompt.length > 240 ? '…' : ''}\``)
      if (shot.videoPrompt) lines.push(`- **视频提示词**：\`${shot.videoPrompt.slice(0, 240)}${shot.videoPrompt.length > 240 ? '…' : ''}\``)
      const assets: string[] = []
      if (shot.firstFrameImage) assets.push(`首帧 \`${shot.firstFrameImage}\``)
      if (shot.lastFrameImage) assets.push(`尾帧 \`${shot.lastFrameImage}\``)
      if (shot.keyframeImage) assets.push(`关键帧 \`${shot.keyframeImage}\``)
      for (const ref of safeJsonArray(shot.referenceImages)) assets.push(`参考图 \`${ref}\``)
      for (const ref of safeJsonArray(shot.referenceAudioUrls)) assets.push(`参考音频 \`${ref}\``)
      if (assets.length) {
        lines.push(`- **输入资产**：`)
        for (const a of assets) lines.push(`  - ${a}`)
      }
      if (shot.videoGenerations.length) {
        lines.push(`- **生成记录**（${shot.videoGenerations.length} 次视频尝试）：`)
        for (const v of shot.videoGenerations) {
          lines.push(`  - #${v.id} \`${v.status}\` route=${v.route || '—'} model=${v.model || '—'}`
            + (v.blockReason ? ` 阻断：${v.blockReason}` : '')
            + (v.videoUrl ? ` 产物：${v.videoUrl}` : ''))
        }
      }
      if (shot.composedVideoUrl) lines.push(`- **成片**：${shot.composedVideoUrl}`)
      lines.push('')
    }
  }

  // 可复现导出说明
  lines.push('## 可复现导出说明')
  lines.push('')
  lines.push('若需断点续作：以各分镜 `指纹`（script_hash）与当前 `episodes.script_hash` 比对，'
    + '不一致即视为剧本变更后的 stale 分镜，需重新拆解。生成时按 `路由` 列选择对应路线，'
    + '参考资产按「输入资产」列表注入。')
  lines.push('')

  return lines.join('\n')
}

/**
 * 断点续作扫描：返回「账本指纹 vs 当前剧本」不一致的分镜清单。
 * @param dramaId 剧集 ID
 * @param episodeId 可选：只扫描指定集
 */
export function scanStaleShots(dramaId: number, episodeId?: number) {
  const ledger = buildProjectLedger(dramaId, episodeId)
  const stale: Array<{ episodeId: number; episodeNumber: number; storyboardId: number; storyboardNumber: number }> = []
  for (const ep of ledger.episodes) {
    for (const shot of ep.shots) {
      if (shot.scriptHash && shot.scriptHash !== ep.scriptHash) {
        stale.push({
          episodeId: ep.id,
          episodeNumber: ep.episodeNumber,
          storyboardId: shot.storyboardId,
          storyboardNumber: shot.storyboardNumber,
        })
      }
    }
  }
  return { dramaId, exportedAt: ledger.exportedAt, staleCount: stale.length, stale }
}
