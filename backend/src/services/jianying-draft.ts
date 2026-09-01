/**
 * 剪映草稿（JianYing Draft）导出服务 — 对齐 ArcReel 的交付侧能力
 *
 * 目标：把一集的合成分镜视频 + TTS 对白 + 字幕导出为剪映可直接打开的草稿，
 * 用户在剪映里继续调整字幕 / 配音 / 节奏 / 转场。
 *
 * 剪映草稿本质是一个 `<name>.draft/` 文件夹，必含：
 *  - draft_content.json：时间线与素材仓库（引用式架构，UUID 关联，时间单位微秒）
 *  - draft_info.json   ：草稿元信息
 *  - draft_meta_info.json：剪映打开时会自动补全，无需生成
 *
 * 素材路径使用「复制进 .draft/media/ 后以绝对路径引用」策略：
 *  - 本机使用：绝对路径直接生效（主要场景，自托管工作台）
 *  - 跨机器分发：把整个 .draft 文件夹打包分发，剪映按草稿目录自动搜索/手动重链素材
 *
 * 兼容性注意：剪映格式为私有格式且随版本演进，此处按社区通用结构生成（微秒计时、
 * canvas_config / materials / tracks 三段式）；若剪映大版本升级导致结构不兼容，
 * 以剪映自建的草稿为基准微调本服务的字段即可。
 */
import { v4 as uuidv4 } from 'uuid'
import fs from 'fs'
import path from 'path'
import { eq, and } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { getDataRoot, getStorageRoot } from '../config.js'
import { logTaskStart, logTaskSuccess, logTaskWarn } from '../utils/task-logger.js'

export interface JianYingDraftFile {
  /** 生成后的本地绝对路径 */
  absPath: string
  /** 打包到 ZIP 内的相对路径（顶层为 <draftName>.draft/） */
  zipPath: string
}

export interface JianYingDraftResult {
  /** .draft 目录绝对路径 */
  draftDir: string
  draftName: string
  durationUs: number
  videoCount: number
  audioCount: number
  textCount: number
  files: JianYingDraftFile[]
}

interface DraftVideo {
  id: string
  path: string
  duration: number
  width: number
  height: number
  type: 'video'
  has_audio: boolean
}

interface DraftAudio {
  id: string
  path: string
  duration: number
  type: 'music'
}

interface DraftText {
  id: string
  content: string
  duration: number
  font_size: number
  x: number
  y: number
  alignment: number
  type: 'text'
  font_name: string
  is_bold: number
  is_italic: number
  outline: { color: { r: number; g: number; b: number; a: number }; size: number; softness: number }
  stroke: { color: { r: number; g: number; b: number; a: number }; size: number; softness: number }
  transform: { x: number; y: number }
  fixed_scale: number
}

/** 把相对/绝对媒体路径转本地绝对路径（不存在返回 null） */
function toMediaPath(p: string | null | undefined): string | null {
  if (!p) return null
  let abs: string
  if (path.isAbsolute(p)) abs = p
  else if (p.startsWith('static/')) abs = path.join(getDataRoot(), p)
  else abs = path.join(getStorageRoot(), p)
  return fs.existsSync(abs) ? abs : null
}

/** 复制文件进 media 目录，返回复制后绝对路径（失败返回 null） */
function copyToMedia(src: string, mediaDir: string, index: number, kind: 'video' | 'audio'): string | null {
  const ext = path.extname(src) || (kind === 'video' ? '.mp4' : '.mp3')
  const dest = path.join(mediaDir, `${kind}_${String(index).padStart(3, '0')}${ext}`)
  try {
    fs.copyFileSync(src, dest)
    return dest
  } catch (err: any) {
    logTaskWarn('JianYingDraft', 'copy-failed', { from: src, error: err?.message })
    return null
  }
}

/** 微秒时长（不小于 1ms 避免 0 时长素材） */
function usOf(duration: number | null | undefined): number {
  return Math.max(1, Math.round((duration ?? 5) * 1_000_000))
}

function sanitize(name: string): string {
  return name.replace(/[\\/:*?"<>|\r\n\t]/g, '_').trim().slice(0, 40) || 'episode'
}

/** 时间线片段（视频/音频共用） */
function segment(materialId: string, startUs: number, durationUs: number): any {
  return {
    id: uuidv4().toUpperCase(),
    material_id: materialId,
    extra_material_refs: [],
    source_timerange: { start: 0, duration: durationUs },
    target_timerange: { start: startUs, duration: durationUs },
    clip: { alpha: 1, scale: { x: 1, y: 1 }, rotation: 0 },
    speed: 1,
    volume: 1,
    visible: true,
  }
}

/** 字幕片段 */
function textSegment(materialId: string, startUs: number, durationUs: number): any {
  return {
    id: uuidv4().toUpperCase(),
    material_id: materialId,
    extra_material_refs: [],
    source_timerange: { start: 0, duration: durationUs },
    target_timerange: { start: startUs, duration: durationUs },
    clip: { alpha: 1, scale: { x: 1, y: 1 }, rotation: 0 },
    transform: { x: 0, y: 0 },
    visible: true,
  }
}

/**
 * 生成一集（或全剧）的剪映草稿
 * @param dramaId   剧 ID
 * @param episodeId 可选：只导出该集；缺省导出全剧（每集一个草稿）
 */
export function buildJianYingDraft(dramaId: number, episodeId?: number): JianYingDraftResult[] {
  const epWhere = episodeId
    ? and(eq(schema.episodes.id, episodeId), eq(schema.episodes.dramaId, dramaId))
    : eq(schema.episodes.dramaId, dramaId)
  const episodes = db
    .select()
    .from(schema.episodes)
    .where(epWhere)
    .orderBy(schema.episodes.episodeNumber)
    .all()

  if (!episodes.length) throw new Error('No episodes found for draft export')

  return episodes.map(ep => buildDraftForEpisode(dramaId, ep))
}

function buildDraftForEpisode(dramaId: number, ep: { id: number; episodeNumber: number; title: string | null }): JianYingDraftResult {
  logTaskStart('JianYingDraft', 'build', { dramaId, episodeId: ep.id, episodeNumber: ep.episodeNumber })

  const storyboards = db
    .select()
    .from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, ep.id))
    .orderBy(schema.storyboards.storyboardNumber)
    .all()

  const ready = storyboards.filter(sb => !!sb.composedVideoUrl)
  if (!ready.length) throw new Error(`Episode ${ep.episodeNumber} has no composed videos`)

  const draftName = `drama${dramaId}_ep${String(ep.episodeNumber).padStart(2, '0')}_${sanitize(ep.title || 'episode')}`
  const draftRoot = path.join(getStorageRoot(), 'jianying')
  fs.mkdirSync(draftRoot, { recursive: true })
  const draftDir = path.join(draftRoot, `${draftName}.draft`)
  const mediaDir = path.join(draftDir, 'media')
  fs.rmSync(draftDir, { recursive: true, force: true })
  fs.mkdirSync(mediaDir, { recursive: true })

  const videos: DraftVideo[] = []
  const audios: DraftAudio[] = []
  const texts: DraftText[] = []
  let cursorUs = 0
  let videoIdx = 0
  let audioIdx = 0

  // 先复制素材，把「分镜 → 视频/音频」映射固定下来
  const shotTimings: Array<{ startUs: number; durationUs: number; videoId: string | null; audioId: string | null }> = []

  // 取首个分镜的最新已完成视频生成记录作为画布尺寸来源（storyboards 无 resolution 字段）
  const firstVideoGen = ready[0]
    ? db.select().from(schema.videoGenerations)
      .where(eq(schema.videoGenerations.storyboardId, ready[0].id))
      .orderBy(schema.videoGenerations.createdAt)
      .all()
      .filter(v => v.width && v.height)
      .pop()
    : undefined
  const canvasW = firstVideoGen?.width ?? 1920
  const canvasH = firstVideoGen?.height ?? 1080

  for (const sb of ready) {
    const durationUs = usOf(sb.duration)

    // 视频素材：合成镜头
    let videoId: string | null = null
    const videoSrc = toMediaPath(sb.composedVideoUrl)
    if (videoSrc) {
      const copied = copyToMedia(videoSrc, mediaDir, videoIdx++, 'video')
      if (copied) {
        videoId = uuidv4().toUpperCase()
        videos.push({ id: videoId, path: copied, duration: durationUs, width: canvasW, height: canvasH, type: 'video', has_audio: true })
      }
    }

    // 音频素材：独立 TTS 对白
    let audioId: string | null = null
    const audioSrc = toMediaPath(sb.ttsAudioUrl)
    if (audioSrc) {
      const copied = copyToMedia(audioSrc, mediaDir, audioIdx++, 'audio')
      if (copied) {
        audioId = uuidv4().toUpperCase()
        audios.push({ id: audioId, path: copied, duration: durationUs, type: 'music' })
      }
    }

    // 字幕素材：对白文本
    if (sb.dialogue) {
      texts.push({
        id: uuidv4().toUpperCase(),
        content: sb.dialogue,
        duration: durationUs,
        font_size: 80,
        x: canvasW / 2,
        y: canvasH - 120,
        alignment: 1,
        type: 'text',
        font_name: 'Microsoft YaHei',
        is_bold: 0,
        is_italic: 0,
        outline: { color: { r: 0, g: 0, b: 0, a: 1 }, size: 0.15, softness: 0 },
        stroke: { color: { r: 0, g: 0, b: 0, a: 1 }, size: 0, softness: 0 },
        transform: { x: 0, y: 0 },
        fixed_scale: 1,
      })
    }

    shotTimings.push({ startUs: cursorUs, durationUs, videoId, audioId })
    cursorUs += durationUs
  }

  if (!videos.length) throw new Error(`Episode ${ep.episodeNumber}: no video materials copied`)

  const totalUs = cursorUs
  const fps = 30
  const canvas = { width: canvasW, height: canvasH }

  // 视频/音频/字幕段（分别按素材顺序与 shotTimings 对齐）
  const videoSegments: any[] = []
  const audioSegments: any[] = []
  const textSegments: any[] = []
  let videoShotIdx = 0
  let audioShotIdx = 0
  let textShotIdx = 0
  for (const t of shotTimings) {
    if (t.videoId) videoSegments.push(segment(t.videoId, t.startUs, t.durationUs))
    if (t.audioId) audioSegments.push(segment(t.audioId, t.startUs, t.durationUs))
    if (texts[textShotIdx]) textSegments.push(textSegment(texts[textShotIdx].id, t.startUs, t.durationUs))
    textShotIdx++
    videoShotIdx++
    audioShotIdx++
  }
  void videoShotIdx
  void audioShotIdx

  const content = {
    canvas_config: { width: canvas.width, height: canvas.height, ratio: 'original' },
    color_space: 0,
    config: {
      maintrack_adsorb: true,
      video_mute: false,
      subtitle_sync: true,
      lyrics_sync: true,
      material_save_mode: 0,
    },
    duration: totalUs,
    fps,
    id: uuidv4().toUpperCase(),
    keyframes: {},
    materials: {
      videos,
      audios,
      texts,
      speeds: [],
      canvases: [],
      audio_fades: [],
      material_animation: [],
      sound_channel_mapping: [],
      vocal_separation: [],
    },
    name: draftName,
    platform: {},
    tracks: [
      { id: uuidv4().toUpperCase(), type: 'video', attribute: 0, is_default_name: true, name: '视频', segments: videoSegments },
      { id: uuidv4().toUpperCase(), type: 'audio', attribute: 0, is_default_name: true, name: '音频', segments: audioSegments },
      ...(textSegments.length ? [{ id: uuidv4().toUpperCase(), type: 'text', attribute: 0, is_default_name: true, name: '字幕', segments: textSegments }] : []),
    ],
    version: 1,
  }

  const draftId = uuidv4().toUpperCase()
  const info = {
    created_at: 0,
    edited: 0,
    tm_draft_enter_edit_time: 0,
    tm_draft_exit_edit_time: 0,
    tm_draft_last_modified: 0,
    tm_draft_last_modified_mtime: 0,
    tm_draft_last_opened: 0,
    tm_draft_removed: 0,
    tm_draft_used: 0,
    draft_fold_path: draftDir,
    draft_id: draftId,
    draft_name: draftName,
    draft_removed: false,
    draft_root_path: draftDir,
    draft_selection: [],
    draft_selection_end: -1,
    draft_selection_start: -1,
    draft_timeline_materials: [],
    draft_used_tracks: [],
    extension: {},
    folder_id: '',
    is_imported_draft: false,
    is_new_simple_draft: false,
    is_short_video_mode: false,
    need_download: false,
    new_version: 0,
    path: draftDir,
    project_has_audio: audios.length > 0,
    project_source: '',
    purchase: { pack: '', version: 0 },
    recover_draft_info: {},
    storage: 'local',
    use_speeches_to_score: false,
    use_shortcut_engine: false,
  }

  fs.writeFileSync(path.join(draftDir, 'draft_content.json'), JSON.stringify(content, null, 2), 'utf-8')
  fs.writeFileSync(path.join(draftDir, 'draft_info.json'), JSON.stringify(info, null, 2), 'utf-8')

  // 文件清单（供 ZIP 打包）
  const files: JianYingDraftFile[] = ['draft_content.json', 'draft_info.json'].map(f => ({
    absPath: path.join(draftDir, f),
    zipPath: path.posix.join(`${draftName}.draft`, f),
  }))
  for (const m of [...videos, ...audios]) {
    if (m.path.startsWith(draftDir)) {
      const rel = path.relative(draftDir, m.path).split(path.sep).join('/')
      files.push({ absPath: m.path, zipPath: path.posix.join(`${draftName}.draft`, rel) })
    }
  }

  logTaskSuccess('JianYingDraft', 'built', {
    dramaId, episodeId: ep.id, draftName, durationUs: totalUs,
    videos: videos.length, audios: audios.length, texts: texts.length,
  })
  return { draftDir, draftName, durationUs: totalUs, videoCount: videos.length, audioCount: audios.length, textCount: texts.length, files }
}
