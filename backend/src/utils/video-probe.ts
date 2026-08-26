import path from 'path'
import ffmpeg from 'fluent-ffmpeg'
import { getStorageRoot } from '../config.js'

/** 探测本地视频文件实际时长（秒），供异步提供商（轮询/Webhook）未返回 duration 时使用 */
export function probeVideoDuration(localPath: string): Promise<number> {
  const absPath = path.isAbsolute(localPath)
    ? localPath
    : path.join(getStorageRoot(), localPath.replace(/^static[\\/]/, ''))
  return new Promise((resolve) => {
    ffmpeg.ffprobe(absPath, (err, metadata) => {
      if (err) { resolve(0); return }
      resolve(Math.round(metadata.format.duration || 0))
    })
  })
}
