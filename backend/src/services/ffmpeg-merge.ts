/**
 * FFmpeg 多镜头拼接 — 将所有合成后的镜头视频拼接为一集
 */
import ffmpeg from 'fluent-ffmpeg'
import fs from 'fs'
import path from 'path'
import { v4 as uuid } from 'uuid'
import { db, schema } from '../db/index.js'
import { eq } from 'drizzle-orm'
import { now } from '../utils/response.js'
import { logTaskError, logTaskStart, logTaskSuccess } from '../utils/task-logger.js'
import { getDataRoot, getStorageRoot } from '../config.js'

/**
 * 合并前自动触发图像连续性检测（fire-and-forget，report 模式）：
 * 所有分镜画面就绪后、开始拼接前对相邻镜头做穿帮筛查，结果写入各分镜 QC 记录。
 * 与用户手动触发的 POST /episodes/:id/consistency-qc 等价。
 */
function runConsistencyQcBeforeMerge(episodeId: number, dramaId: number): void {
  import('./consistency-qc.js').then(({ runEpisodeConsistencyQc }) =>
    runEpisodeConsistencyQc(episodeId, dramaId).catch(() => {})
  ).catch(() => {})
}

function toAbsPath(relativePath: string): string {
  if (path.isAbsolute(relativePath)) return relativePath
  if (relativePath.startsWith('static/')) return path.join(getDataRoot(), relativePath)
  return path.join(getStorageRoot(), relativePath)
}

/**
 * 拼接一集的所有合成镜头视频
 */
export async function mergeEpisodeVideos(episodeId: number, dramaId: number): Promise<number> {
  const storyboards = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, episodeId))
    .orderBy(schema.storyboards.storyboardNumber)
    .all()

  const composedStoryboards = storyboards.filter(sb => !!sb.composedVideoUrl)
  if (composedStoryboards.length !== storyboards.length) {
    throw new Error(`Only composed storyboards can be merged (${composedStoryboards.length}/${storyboards.length} ready)`)
  }
  const videos = composedStoryboards
    .map(sb => sb.composedVideoUrl)
    .filter(Boolean) as string[]

  if (videos.length === 0) throw new Error('No videos to merge')

  // 读取本集统一配乐方案：一条 BGM 贯穿全片（而非逐镜头切换），跨镜头自然过渡
  const [episode] = db.select().from(schema.episodes)
    .where(eq(schema.episodes.id, episodeId)).all()
  const bgm = episode?.bgmUrl
    ? {
        bgmUrl: episode.bgmUrl,
        bgmVolume: episode.bgmVolume ?? 0.3,
        bgmFadeIn: episode.bgmFadeIn ?? 1.5,
        bgmFadeOut: episode.bgmFadeOut ?? 2.0,
      }
    : null

  logTaskStart('MergeTask', 'episode-merge', { episodeId, dramaId, clips: videos.length, bgm: !!bgm })

  // 合并前自动做图像连续性检测（穿帮筛查，不影响拼接主流程）
  runConsistencyQcBeforeMerge(episodeId, dramaId)

  // 创建 merge 记录
  const ts = now()
  const res = db.insert(schema.videoMerges).values({
    episodeId,
    dramaId,
    title: `Episode ${episodeId} Merge`,
    provider: 'ffmpeg',
    model: 'ffmpeg-concat-h264-aac',
    status: 'processing',
    scenes: JSON.stringify(videos),
    createdAt: ts,
  }).run()
  const mergeId = Number(res.lastInsertRowid)

  // 异步执行
  doMerge(mergeId, episodeId, videos, bgm).catch(err => {
    logTaskError('MergeTask', 'episode-merge', { mergeId, episodeId, error: err.message })
    console.error(`[Merge] Failed:`, err)
    db.update(schema.videoMerges)
      .set({ status: 'failed', errorMsg: err.message })
      .where(eq(schema.videoMerges.id, mergeId)).run()
  })

  return mergeId
}

async function doMerge(
  mergeId: number,
  episodeId: number,
  videos: string[],
  bgm: { bgmUrl: string; bgmVolume: number; bgmFadeIn: number; bgmFadeOut: number } | null,
) {
  // 生成 concat 列表文件
  const listDir = path.join(getStorageRoot(), 'temp')
  fs.mkdirSync(listDir, { recursive: true })
  const listPath = path.join(listDir, `${uuid()}.txt`)

  const listContent = videos
    .map(v => `file '${toAbsPath(v)}'`)
    .join('\n')
  fs.writeFileSync(listPath, listContent, 'utf-8')

  // 输出文件
  const outputDir = path.join(getStorageRoot(), 'merged')
  fs.mkdirSync(outputDir, { recursive: true })
  const outputFilename = `${uuid()}.mp4`
  const outputPath = path.join(outputDir, outputFilename)

  await new Promise<void>((resolve, reject) => {
    ffmpeg()
      .input(listPath)
      .inputOptions(['-f', 'concat', '-safe', '0'])
      .outputOptions([
        '-fflags', '+genpts',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'aac',
        '-ar', '48000',
        '-b:a', '192k',
        '-movflags', '+faststart',
      ])
      .output(outputPath)
      .on('end', () => resolve())
      .on('error', (err) => reject(err))
      .run()
  })

  // 清理临时文件
  fs.unlinkSync(listPath)

  // 统一配乐混音：BGM 循环贯穿全片 + 首尾淡入淡出，与对白平滑叠加
  let finalPath = outputPath
  if (bgm) {
    try {
      finalPath = await mixBgm(outputPath, bgm)
    } catch (err: any) {
      console.warn(`[Merge] BGM mix failed, falling back to no-bgm:`, err.message)
      logTaskError('MergeTask', 'bgm-mix', { mergeId, episodeId, error: err.message })
    }
  }

  // 成片响度归一化：对齐社媒平台验收标准 I=-14 LUFS / TP=-1.5dB / LRA=11
  // （对齐参考项目 H3-Codex-Drama final master 阶段 loudnorm）
  try {
    finalPath = await normalizeLoudness(finalPath)
  } catch (err: any) {
    console.warn(`[Merge] Loudness normalization failed, keep as-is:`, err.message)
    logTaskError('MergeTask', 'loudnorm', { mergeId, episodeId, error: err.message })
  }

  // 获取时长
  const duration = await getVideoDuration(finalPath)

  const mergedRelative = `static/merged/${path.basename(finalPath)}`

  // 更新 merge 记录
  db.update(schema.videoMerges)
    .set({ status: 'completed', mergedUrl: mergedRelative, duration, completedAt: now() })
    .where(eq(schema.videoMerges.id, mergeId)).run()

  // 更新 episode
  db.update(schema.episodes)
    .set({ videoUrl: mergedRelative, updatedAt: now() })
    .where(eq(schema.episodes.id, episodeId)).run()

  logTaskSuccess('MergeTask', 'episode-merge', { mergeId, episodeId, output: mergedRelative, duration, clips: videos.length })
}

/**
 * 将统一配乐混入成片：BGM 循环播放覆盖全程，音量压低、首尾淡入淡出，与对白平滑叠加。
 * 成功后输出为新文件并删除无 BGM 的中间文件（失败抛错，由调用方回退到无 BGM 版本）。
 */
async function mixBgm(
  videoPath: string,
  bgm: { bgmUrl: string; bgmVolume: number; bgmFadeIn: number; bgmFadeOut: number },
): Promise<string> {
  const bgmAbs = toAbsPath(bgm.bgmUrl)
  if (!fs.existsSync(bgmAbs)) throw new Error(`BGM file not found: ${bgm.bgmUrl}`)

  const duration = await getVideoDuration(videoPath)
  const volume = Math.max(0, Math.min(1, bgm.bgmVolume ?? 0.3))
  // 淡入淡出时长不超过成片一半，避免短片段超界
  const fadeIn = Math.min(Math.max(0, bgm.bgmFadeIn ?? 1.5), Math.max(0, duration / 2))
  const fadeOut = Math.min(Math.max(0, bgm.bgmFadeOut ?? 2.0), Math.max(0, duration / 2))
  const fadeOutStart = Math.max(0, duration - fadeOut)

  const outPath = path.join(path.dirname(videoPath), `${uuid()}.mp4`)

  await new Promise<void>((resolve, reject) => {
    ffmpeg()
      .input(videoPath)
      .input(bgmAbs)
      .inputOptions(['-stream_loop', '-1'])
      .complexFilter([
        `[1:a]volume=${volume.toFixed(3)},afade=t=in:st=0:d=${fadeIn.toFixed(2)},afade=t=out:st=${fadeOutStart.toFixed(2)}:d=${fadeOut.toFixed(2)}[bgm]`,
        `[0:a][bgm]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[aout]`,
      ])
      .outputOptions([
        '-map', '0:v',
        '-map', '[aout]',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-ar', '48000',
        '-b:a', '192k',
        '-movflags', '+faststart',
      ])
      .output(outPath)
      .on('end', () => resolve())
      .on('error', (err) => reject(err))
      .run()
  })

  // 成功后删除无 BGM 的中间文件
  try { if (outPath !== videoPath) fs.unlinkSync(videoPath) } catch {}

  return outPath
}

function getVideoDuration(filePath: string): Promise<number> {
  return new Promise((resolve) => {
    ffmpeg.ffprobe(filePath, (err, metadata) => {
      if (err) { resolve(0); return }
      resolve(Math.round(metadata.format.duration || 0))
    })
  })
}

/** 探测视频是否存在音轨 */
function videoHasAudio(filePath: string): Promise<boolean> {
  return new Promise((resolve) => {
    ffmpeg.ffprobe(filePath, (err, metadata) => {
      if (err) { resolve(false); return }
      const stream = metadata.streams?.find((s) => s.codec_type === 'audio')
      resolve(!!stream)
    })
  })
}

/**
 * 成片响度归一化（I=-14 LUFS / TP=-1.5dB / LRA=11）。
 * 对齐社媒平台验收标准（抖音/小红书等常见参考门限）与参考项目 H3-Codex-Drama 的 final master 阶段。
 * 无音轨视频直接返回原路径；loudnorm 失败抛错由调用方兜底。
 * 成功时输出为新文件并删除旧文件。
 */
async function normalizeLoudness(videoPath: string): Promise<string> {
  if (!(await videoHasAudio(videoPath))) return videoPath

  const outPath = path.join(path.dirname(videoPath), `${uuid()}.mp4`)

  await new Promise<void>((resolve, reject) => {
    ffmpeg()
      .input(videoPath)
      .outputOptions([
        '-c:v', 'copy',
        '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11',
        '-c:a', 'aac',
        '-ar', '48000',
        '-b:a', '192k',
        '-movflags', '+faststart',
      ])
      .output(outPath)
      .on('end', () => resolve())
      .on('error', (err) => reject(err))
      .run()
  })

  try { if (outPath !== videoPath) fs.unlinkSync(videoPath) } catch {}

  return outPath
}
