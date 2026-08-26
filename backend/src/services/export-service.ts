/**
 * 素材打包导出服务 — 把一部剧的成片视频 + 源素材（角色/场景/关键帧）打包成 ZIP
 * 对齐 gcc exportService 的 downloadMasterVideo（成片） / downloadSourceAssets（源素材）两场景。
 *
 * 关键前提：voide 所有媒体（图片/视频）在生成完成后都会回填为本地路径 `static/...`
 * （见 image-generation.ts / video-generation.ts / ffmpeg-compose.ts / ffmpeg-merge.ts），
 * 因此导出时直接按本地路径流式读取即可，无需再下载远程 URL。
 */
import { ZipArchive } from 'archiver'
import { Readable } from 'node:stream'
import fs from 'fs'
import { and, eq, inArray, isNull } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { getAbsolutePath } from '../utils/storage.js'
import { probeVideoDuration } from '../utils/video-probe.js'

export type ExportScope = 'all' | 'video' | 'assets'

export interface ExportFileEntry {
  /** 本地绝对路径 */
  absPath: string
  /** ZIP 内的相对路径 */
  zipPath: string
}

/** 清理文件名中的非法字符，保留中英文数字下划线连字符，空值回退 untitled */
function sanitizeName(name: string | null | undefined): string {
  return (name || '').replace(/[\\/:*?"<>|\r\n\t]/g, '_').trim() || 'untitled'
}

/** 从媒体路径推断扩展名（去掉 query），默认 bin */
function extOf(p: string): string {
  const m = /\.([a-zA-Z0-9]+)(?:\?.*)?$/.exec(p || '')
  return m ? m[1].toLowerCase() : 'bin'
}

/** 把媒体路径（static/xxx）转成本地绝对路径；远程 URL / data: 等未落盘形态返回 null */
function toLocalAbsPath(p: string | null | undefined): string | null {
  if (!p) return null
  if (p.startsWith('data:') || p.startsWith('http://') || p.startsWith('https://')) return null
  try {
    const abs = getAbsolutePath(p)
    return fs.existsSync(abs) ? abs : null
  } catch {
    return null
  }
}

/** 收集该 drama 下需要导出的文件清单 */
export function collectDramaExportFiles(dramaId: number, scope: ExportScope): ExportFileEntry[] {
  const files: ExportFileEntry[] = []

  const episodes = db
    .select()
    .from(schema.episodes)
    .where(and(eq(schema.episodes.dramaId, dramaId), isNull(schema.episodes.deletedAt)))
    .orderBy(schema.episodes.episodeNumber)
    .all()
  const epNumById = new Map(episodes.map((e) => [e.id, e.episodeNumber]))

  // 成片视频（整集）
  if (scope === 'all' || scope === 'video') {
    for (const ep of episodes) {
      if (!ep.videoUrl) continue
      const abs = toLocalAbsPath(ep.videoUrl)
      if (!abs) continue
      const epLabel = String(ep.episodeNumber).padStart(2, '0')
      files.push({
        absPath: abs,
        zipPath: `videos/ep${epLabel}_${sanitizeName(ep.title)}.${extOf(ep.videoUrl)}`,
      })
    }
  }

  // 源素材
  if (scope === 'all' || scope === 'assets') {
    // 角色立绘
    const characters = db
      .select()
      .from(schema.characters)
      .where(and(eq(schema.characters.dramaId, dramaId), isNull(schema.characters.deletedAt)))
      .all()
    for (const ch of characters) {
      if (!ch.imageUrl) continue
      const abs = toLocalAbsPath(ch.imageUrl)
      if (!abs) continue
      files.push({ absPath: abs, zipPath: `characters/${sanitizeName(ch.name)}.${extOf(ch.imageUrl)}` })
    }

    // 场景图
    const scenes = db.select().from(schema.scenes).where(eq(schema.scenes.dramaId, dramaId)).all()
    for (const sc of scenes) {
      if (!sc.imageUrl) continue
      const abs = toLocalAbsPath(sc.imageUrl)
      if (!abs) continue
      files.push({ absPath: abs, zipPath: `scenes/${sanitizeName(sc.location)}.${extOf(sc.imageUrl)}` })
    }

    // 分镜关键帧（首帧/尾帧）
    if (episodes.length > 0) {
      const storyboards = db
        .select()
        .from(schema.storyboards)
        .where(inArray(schema.storyboards.episodeId, episodes.map((e) => e.id)))
        .orderBy(schema.storyboards.episodeId, schema.storyboards.storyboardNumber)
        .all()
      for (const sb of storyboards) {
        const epLabel = String(epNumById.get(sb.episodeId) ?? 0).padStart(2, '0')
        const sbLabel = String(sb.storyboardNumber).padStart(2, '0')
        const dir = `storyboards/ep${epLabel}`
        if (sb.firstFrameImage) {
          const abs = toLocalAbsPath(sb.firstFrameImage)
          if (abs) files.push({ absPath: abs, zipPath: `${dir}/sb${sbLabel}_first.${extOf(sb.firstFrameImage)}` })
        }
        if (sb.lastFrameImage) {
          const abs = toLocalAbsPath(sb.lastFrameImage)
          if (abs) files.push({ absPath: abs, zipPath: `${dir}/sb${sbLabel}_last.${extOf(sb.lastFrameImage)}` })
        }
      }
    }
  }

  return files
}

/** 构建 ZIP 流（流式打包，边生成边输出，避免把大视频读进内存） */
export function buildExportZip(files: ExportFileEntry[]): ReadableStream {
  const archive = new ZipArchive({ zlib: { level: 6 } })
  archive.on('warning', (err) => {
    // ENOENT 表示某文件在打包时被移除，忽略；其余告警打印
    if (err.code !== 'ENOENT') console.warn('[export] archiver warning:', err.message)
  })
  archive.on('error', (err) => {
    console.error('[export] archiver error:', err.message)
  })
  for (const f of files) {
    archive.file(f.absPath, { name: f.zipPath })
  }
  // 通知无更多输入，后台完成打包（数据边生成边流出），不 await 以免阻塞到内存
  archive.finalize().catch((err) => console.error('[export] finalize failed:', (err as Error).message))
  return Readable.toWeb(archive) as unknown as ReadableStream
}

/** 秒 → 时间码 HH:MM:SS:FF（fps 默认 25，PAL） */
export function secondsToTimecode(seconds: number, fps = 25): string {
  const totalFrames = Math.max(0, Math.round(seconds * fps))
  const ff = totalFrames % fps
  const totalSec = Math.floor(totalFrames / fps)
  const ss = totalSec % 60
  const mm = Math.floor(totalSec / 60) % 60
  const hh = Math.floor(totalSec / 3600)
  const p = (n: number, w = 2) => String(n).padStart(w, '0')
  return `${p(hh)}:${p(mm)}:${p(ss)}:${p(ff)}`
}

/**
 * 生成 CMX3600 格式 EDL（编辑决策列表），供 Premiere / DaVinci Resolve 等剪辑软件导入。
 * 每个有视频的分镜对应一条事件，按「集号 → 分镜号」顺序铺在时间线上。
 */
export async function buildEdl(dramaId: number, opts?: { episodeId?: number; fps?: number }): Promise<string> {
  const fps = opts?.fps || 25
  const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()

  const epWhere = opts?.episodeId
    ? and(eq(schema.episodes.id, opts.episodeId), eq(schema.episodes.dramaId, dramaId))
    : eq(schema.episodes.dramaId, dramaId)
  const episodes = db
    .select()
    .from(schema.episodes)
    .where(and(epWhere, isNull(schema.episodes.deletedAt)))
    .orderBy(schema.episodes.episodeNumber)
    .all()

  const lines: string[] = [`TITLE: ${drama?.title || 'Drama Studio Export'}`, 'FCM: NON-DROP FRAME', '']
  let eventNo = 0
  let recordInSec = 0

  for (const ep of episodes) {
    const sbs = db
      .select()
      .from(schema.storyboards)
      .where(and(eq(schema.storyboards.episodeId, ep.id), isNull(schema.storyboards.deletedAt)))
      .orderBy(schema.storyboards.storyboardNumber)
      .all()
    for (const sb of sbs) {
      if (!sb.videoUrl) continue
      let dur = sb.duration || 0
      if (dur <= 0) dur = await probeVideoDuration(sb.videoUrl)
      if (dur <= 0) continue

      eventNo++
      const clipName = sb.videoUrl.split('/').pop() || `ep${ep.episodeNumber}_sb${sb.storyboardNumber}.mp4`
      const srcIn = secondsToTimecode(0, fps)
      const srcOut = secondsToTimecode(dur, fps)
      const recIn = secondsToTimecode(recordInSec, fps)
      const recOut = secondsToTimecode(recordInSec + dur, fps)

      lines.push(`${String(eventNo).padStart(3, '0')}  AX       V     C        ${srcIn} ${srcOut} ${recIn} ${recOut}`)
      lines.push(`* FROM CLIP NAME: ${clipName}`)
      const comment = [sb.title, sb.description].filter(Boolean).join(' - ')
      if (comment) lines.push(`* COMMENT: ${comment}`)
      recordInSec += dur
    }
  }

  lines.push('')
  return lines.join('\n')
}
