import { Hono } from 'hono'
import { success, badRequest } from '../utils/response.js'
import { saveUploadedFile } from '../utils/storage.js'
import { logTaskError } from '../utils/task-logger.js'

const app = new Hono()

const MAX_IMAGE_SIZE = 20 * 1024 * 1024 // 20MB
const ALLOWED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif']

const MAX_AUDIO_SIZE = 50 * 1024 * 1024 // 50MB
const ALLOWED_AUDIO_TYPES = [
  'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav',
  'audio/mp4', 'audio/m4a', 'audio/x-m4a', 'audio/aac',
  'audio/flac', 'audio/ogg', 'audio/webm',
]

const MAX_VIDEO_SIZE = 200 * 1024 * 1024 // 200MB
const ALLOWED_VIDEO_TYPES = [
  'video/mp4', 'video/quicktime', 'video/webm',
  'video/x-m4v', 'video/x-msvideo', 'video/mpeg',
]

/** 通用上传处理：校验类型/大小 → 落盘 → 返回相对路径 */
async function handleUpload(
  c: any,
  allowedTypes: string[],
  maxSize: number,
  dir: string,
): Promise<Response> {
  const body = await c.req.parseBody()
  const file = body['file']

  if (!file || !(file instanceof File)) {
    return badRequest(c, 'file is required')
  }
  if (!allowedTypes.includes(file.type)) {
    return badRequest(c, `Unsupported file type: ${file.type}. Allowed: ${allowedTypes.join(', ')}`)
  }
  if (file.size > maxSize) {
    return badRequest(c, `File too large: ${(file.size / 1024 / 1024).toFixed(1)}MB. Max: ${maxSize / 1024 / 1024}MB`)
  }

  const buffer = await file.arrayBuffer()
  const savedPath = await saveUploadedFile(buffer, dir, file.name)
  return success(c, { url: `/${savedPath}`, path: savedPath })
}

// POST /upload/image
app.post('/image', async (c) => {
  try {
    return await handleUpload(c, ALLOWED_IMAGE_TYPES, MAX_IMAGE_SIZE, 'uploads')
  } catch (err: any) { logTaskError('UploadAPI', 'image', { error: err.message }); return badRequest(c, err.message) }
})

// POST /upload/audio — 配乐/背景音乐素材
app.post('/audio', async (c) => {
  try {
    return await handleUpload(c, ALLOWED_AUDIO_TYPES, MAX_AUDIO_SIZE, 'audio')
  } catch (err: any) { logTaskError('UploadAPI', 'audio', { error: err.message }); return badRequest(c, err.message) }
})

// POST /upload/video — 自定义首尾帧视频素材
app.post('/video', async (c) => {
  try {
    return await handleUpload(c, ALLOWED_VIDEO_TYPES, MAX_VIDEO_SIZE, 'uploads')
  } catch (err: any) { logTaskError('UploadAPI', 'video', { error: err.message }); return badRequest(c, err.message) }
})

export default app
