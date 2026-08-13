import { Hono } from 'hono'
import { success, badRequest } from '../utils/response.js'
import { saveUploadedFile } from '../utils/storage.js'
import { logTaskError } from '../utils/task-logger.js'

const app = new Hono()

const MAX_UPLOAD_SIZE = 20 * 1024 * 1024 // 20MB
const ALLOWED_MIME_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif']

// POST /upload/image
app.post('/image', async (c) => {
  try {
  const body = await c.req.parseBody()
  const file = body['file']

  if (!file || !(file instanceof File)) {
    return badRequest(c, 'file is required')
  }

  if (!ALLOWED_MIME_TYPES.includes(file.type)) {
    return badRequest(c, `Unsupported file type: ${file.type}. Allowed: ${ALLOWED_MIME_TYPES.join(', ')}`)
  }

  if (file.size > MAX_UPLOAD_SIZE) {
    return badRequest(c, `File too large: ${(file.size / 1024 / 1024).toFixed(1)}MB. Max: ${MAX_UPLOAD_SIZE / 1024 / 1024}MB`)
  }

  const buffer = await file.arrayBuffer()
  const savedPath = await saveUploadedFile(buffer, 'uploads', file.name)
  return success(c, { url: `/${savedPath}`, path: savedPath })
  } catch (err: any) { logTaskError('UploadAPI', 'image', { error: err.message }); return badRequest(c, err.message) }
})

export default app
