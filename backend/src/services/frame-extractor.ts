/**
 * 尾帧提取（连续性状态机 v3 的衔接链路）
 *
 * 视频生成完成后，从已产出视频中提取真实尾帧写入 storyboards.last_frame_image，
 * 供下一镜头视频生成时作为「上一镜尾帧」起帧参考（submitMissingVideos 的 tail-link）。
 */
import ffmpeg from 'fluent-ffmpeg'
import fs from 'fs'
import path from 'path'
import { v4 as uuid } from 'uuid'
import { and, eq, isNull } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { getDataRoot, getStorageRoot } from '../config.js'
import { now } from '../utils/response.js'
import { logTaskSuccess, logTaskWarn } from '../utils/task-logger.js'

/** 将相对媒体路径转为绝对路径 */
function toAbsMediaPath(relativePath: string): string {
  if (path.isAbsolute(relativePath)) return relativePath
  if (relativePath.startsWith('static/')) return path.join(getDataRoot(), relativePath)
  return path.join(getStorageRoot(), relativePath)
}

/** 从视频提取尾帧，返回可访问的相对图片路径；失败返回 null */
async function extractTailFrame(videoUrl: string): Promise<string | null> {
  const absPath = toAbsMediaPath(videoUrl)
  if (!fs.existsSync(absPath)) return null
  const duration = await new Promise<number>((resolve) => {
    ffmpeg.ffprobe(absPath, (err, metadata) => {
      if (err) { resolve(0); return }
      resolve(metadata.format.duration || 0)
    })
  })
  if (duration <= 0) return null
  const outputDir = path.join(getStorageRoot(), 'frames')
  fs.mkdirSync(outputDir, { recursive: true })
  const filename = `${uuid()}.jpg`
  const outPath = path.join(outputDir, filename)
  const seek = Math.max(0, duration - 0.2)
  try {
    await new Promise<void>((resolve, reject) => {
      ffmpeg(absPath)
        .seekInput(seek)
        .outputOptions(['-frames:v', '1', '-q:v', '2'])
        .output(outPath)
        .on('end', () => resolve())
        .on('error', (err) => reject(err))
        .run()
    })
    return `static/frames/${filename}`
  } catch {
    return null
  }
}

/** 提取本集所有「已有视频但缺尾帧」分镜的尾帧；返回成功提取数量（幂等） */
export async function extractStoryboardTailFrames(episodeId: number, dramaId: number): Promise<number> {
  const sbs = db
    .select()
    .from(schema.storyboards)
    .where(and(eq(schema.storyboards.episodeId, episodeId), isNull(schema.storyboards.deletedAt)))
    .all()
  let done = 0
  let failed = 0
  for (const sb of sbs) {
    if (!sb.videoUrl || sb.lastFrameImage) continue
    const frame = await extractTailFrame(sb.videoUrl)
    if (!frame) { failed++; continue }
    db.update(schema.storyboards)
      .set({ lastFrameImage: frame, updatedAt: now() })
      .where(eq(schema.storyboards.id, sb.id))
      .run()
    done++
  }
  if (done > 0 || failed > 0) {
    logTaskSuccess('TailFrame', 'extract', { episodeId, dramaId, extracted: done, failed })
  }
  return done
}
