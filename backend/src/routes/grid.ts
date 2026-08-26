import { Hono } from 'hono'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { success, badRequest, notFound, now, parseParamId } from '../utils/response.js'
import { generateImage } from '../services/image-generation.js'
import { splitGridImage } from '../services/grid-split.js'
import { runAgentWithRetry } from '../agents/index.js'
import { logTaskError, logTaskPayload, logTaskProgress } from '../utils/task-logger.js'
import { STORYBOARD_IMAGE_NEGATIVE, buildGridPrompt, buildGridCellPrompts, collectGridReferenceAssets, buildReferenceLegend } from '../shared/prompt-utils.js'

const app = new Hono()

function extractJsonCandidate(text: string) {
  const fenced = text.match(/```json\s*([\s\S]*?)```/i)
  if (fenced?.[1]) return fenced[1].trim()

  const plain = text.match(/\{[\s\S]*\}/)
  return plain?.[0]?.trim() || ''
}

function normalizeGridPayload(payload: any) {
  if (!payload || typeof payload !== 'object') return null
  const gridPrompt = typeof payload.grid_prompt === 'string'
    ? payload.grid_prompt.trim()
    : typeof payload.gridPrompt === 'string'
      ? payload.gridPrompt.trim()
      : ''
  const rawCells = Array.isArray(payload.cell_prompts)
    ? payload.cell_prompts
    : Array.isArray(payload.cellPrompts)
      ? payload.cellPrompts
      : []
  const cellPrompts = rawCells.map((cell: any) => ({
    shot_number: Number(cell?.shot_number ?? cell?.shotNumber ?? 0) || 0,
    frame_type: String(cell?.frame_type ?? cell?.frameType ?? 'first_frame'),
    prompt: String(cell?.prompt ?? '').trim(),
  })).filter((cell: any) => cell.prompt)

  if (!gridPrompt) return null
  return { grid_prompt: gridPrompt, cell_prompts: cellPrompts }
}

function findGridPayload(value: any): { grid_prompt: string; cell_prompts: any[] } | null {
  if (!value) return null

  const normalized = normalizeGridPayload(value)
  if (normalized) return normalized

  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed || trimmed === 'null') return null
    try {
      const parsed = JSON.parse(trimmed)
      return findGridPayload(parsed)
    } catch {
      const candidate = extractJsonCandidate(trimmed)
      if (!candidate) return null
      try {
        return findGridPayload(JSON.parse(candidate))
      } catch {
        return null
      }
    }
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      const found = findGridPayload(item)
      if (found) return found
    }
    return null
  }

  if (typeof value === 'object') {
    for (const nested of Object.values(value)) {
      const found = findGridPayload(nested)
      if (found) return found
    }
  }

  return null
}

async function tryAgentGridPrompt(
  episodeId: number,
  dramaId: number,
  storyboardIds: number[],
  rows: number,
  cols: number,
  mode: string,
  referenceLegend: string,
) {
  const message = [
    '请为宫格图生成提示词，并优先调用工具完成。',
    `选中镜头ID：${JSON.stringify(storyboardIds)}`,
    `行数：${rows}`,
    `列数：${cols}`,
    `模式：${mode}`,
    referenceLegend ? `参考图映射：${referenceLegend}` : '',
    '当提示词涉及到某个角色或场景时，直接把对应的图片编号写进提示词，例如：图片1中的角色A站了起来，图片3中的房间场景。不要只写名字，不写图片编号。',
    `必须严格按 ${rows}x${cols} 生成，总共 exactly ${rows * cols} visible panels。不要合并格子，不要缺格。`,
    '必须返回 JSON，结构为：{"grid_prompt":"...","cell_prompts":[{"shot_number":1,"frame_type":"first_frame","prompt":"..."}]}',
  ].filter(Boolean).join('\n')

  try {
    const ret = await runAgentWithRetry('grid_prompt_generator', episodeId, dramaId, message, { maxSteps: 10 })
    const fromTools = findGridPayload(ret.toolResults)
    if (fromTools) return fromTools
    const fromText = findGridPayload(ret.text)
    if (fromText) return fromText
    return null
  } catch {
    return null
  }
}

// POST /grid/prompt
app.post('/prompt', async (c) => {
  try {
  const body = await c.req.json()
  const {
    storyboard_ids,
    drama_id,
    episode_id,
    rows,
    cols,
    mode = 'first_frame',
  } = body

  if (!storyboard_ids?.length) return badRequest(c, 'storyboard_ids required')
  if (!rows || !cols) return badRequest(c, 'rows and cols required')

  const storyboards = storyboard_ids.map((id: number) => {
    const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
    return sb
  }).filter(Boolean)

  if (!storyboards.length) return badRequest(c, 'No storyboards found')

  let dramaStyle = ''
  if (drama_id) {
    const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, drama_id)).all()
    dramaStyle = drama?.style || ''
  }

  const actualCols = cols
  const actualRows = rows
  const resolvedEpisodeId = Number(episode_id || storyboards[0]?.episodeId || 0)
  const referenceAssets = collectGridReferenceAssets(storyboards)
  const referenceLegend = buildReferenceLegend(referenceAssets)

  if (!resolvedEpisodeId) {
    return badRequest(c, 'episode_id required')
  }

  try {
    const agentPayload = await tryAgentGridPrompt(
      resolvedEpisodeId,
      Number(drama_id || 0),
      storyboard_ids,
      actualRows,
      actualCols,
      mode,
      referenceLegend,
    )

    if (agentPayload?.grid_prompt) {
      logTaskProgress('GridPrompt', 'agent-success', {
        episodeId: resolvedEpisodeId,
        dramaId: drama_id,
        mode,
        rows: actualRows,
        cols: actualCols,
        storyboardCount: storyboard_ids.length,
      })
      logTaskPayload('GridPrompt', 'agent-result', agentPayload)
      return success(c, {
        ...agentPayload,
        source: 'agent',
        grid: { rows: actualRows, cols: actualCols },
        storyboard_ids,
        mode,
      })
    }
  } catch (err: any) {
    logTaskError('GridPrompt', 'agent-failed', {
      episodeId: resolvedEpisodeId,
      dramaId: drama_id,
      error: err.message,
    })
  }

  const gridPrompt = buildGridPrompt(mode, storyboards, actualRows, actualCols, dramaStyle, referenceAssets)
  const cellPrompts = buildGridCellPrompts(mode, storyboards, actualRows, actualCols, referenceAssets)
  logTaskProgress('GridPrompt', 'fallback-used', {
    episodeId: resolvedEpisodeId,
    dramaId: drama_id,
    mode,
    rows: actualRows,
    cols: actualCols,
    storyboardCount: storyboard_ids.length,
  })

  return success(c, {
    grid_prompt: gridPrompt,
    cell_prompts: cellPrompts,
    source: 'fallback',
    grid: { rows: actualRows, cols: actualCols },
    storyboard_ids,
    mode,
  })
  } catch (err: any) { logTaskError('GridPrompt', 'prompt', { error: err.message }); return badRequest(c, err.message) }
})

// POST /grid/generate
app.post('/generate', async (c) => {
  const body = await c.req.json()
  const {
    storyboard_ids,
    drama_id,
    rows,
    cols,
    mode = 'first_frame', // first_frame | first_last | multi_ref
    custom_prompt,
  } = body

  if (!storyboard_ids?.length) return badRequest(c, 'storyboard_ids required')
  if (!rows || !cols) return badRequest(c, 'rows and cols required')

  const storyboards = storyboard_ids.map((id: number) => {
    const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, id)).all()
    return sb
  }).filter(Boolean)

  if (!storyboards.length) return badRequest(c, 'No storyboards found')

  // Get drama style
  let dramaStyle = ''
  if (drama_id) {
    const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, drama_id)).all()
    dramaStyle = drama?.style || ''
  }

  const referenceAssets = collectGridReferenceAssets(storyboards)
  const prompt = custom_prompt || buildGridPrompt(mode, storyboards, rows, cols, dramaStyle, referenceAssets)
  const referenceImages = referenceAssets.map((asset) => asset.path)

  // Size: first_last mode uses Nx2 layout
  const cellW = 960, cellH = 540
  const actualCols = cols
  const actualRows = rows
  const size = `${cellW * actualCols}x${cellH * actualRows}`

  try {
    const genId = await generateImage({
      dramaId: drama_id,
      prompt,
      negativePrompt: STORYBOARD_IMAGE_NEGATIVE,
      size,
      frameType: `grid_${mode}_${actualRows}x${actualCols}`,
      referenceImages,
    })

    logTaskProgress('GridGenerate', 'reference-images', {
      dramaId: drama_id,
      mode,
      rows: actualRows,
      cols: actualCols,
      referenceCount: referenceImages.length,
    })

    return success(c, {
      image_generation_id: genId,
      grid: { rows: actualRows, cols: actualCols },
      mode,
      storyboard_ids,
      prompt,
      reference_images: referenceImages,
    })
  } catch (err: any) {
    return badRequest(c, err.message)
  }
})

// POST /grid/split
app.post('/split', async (c) => {
  const body = await c.req.json()
  const {
    image_generation_id,
    rows,
    cols,
    assignments, // [{storyboard_id, frame_type: 'first_frame'|'last_frame'|'reference'}]
  } = body

  if (!image_generation_id) return badRequest(c, 'image_generation_id required')
  if (!rows || !cols) return badRequest(c, 'rows and cols required')
  if (!assignments?.length) return badRequest(c, 'assignments required')

  const [imgRecord] = db.select().from(schema.imageGenerations)
    .where(eq(schema.imageGenerations.id, image_generation_id)).all()

  if (!imgRecord) return badRequest(c, 'Image generation not found')
  if (imgRecord.status !== 'completed') return badRequest(c, `Image status: ${imgRecord.status}`)
  if (!imgRecord.localPath) return badRequest(c, 'No local image file')

  try {
    const cells = await splitGridImage(imgRecord.localPath, rows, cols)

    const results: any[] = []
    for (let i = 0; i < assignments.length && i < cells.length; i++) {
      const { storyboard_id, frame_type } = assignments[i]
      const cell = cells[i]
      if (!storyboard_id) continue

      const update: Record<string, any> = { updatedAt: now() }
      if (frame_type === 'first_frame') update.firstFrameImage = cell.localPath
      else if (frame_type === 'last_frame') update.lastFrameImage = cell.localPath
      else if (frame_type === 'reference') {
        const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, storyboard_id)).all()
        const existing = sb?.referenceImages ? JSON.parse(sb.referenceImages) : []
        existing.push(cell.localPath)
        update.referenceImages = JSON.stringify(existing)
      }

      db.update(schema.storyboards).set(update).where(eq(schema.storyboards.id, storyboard_id)).run()
      results.push({ storyboard_id, frame_type, local_path: cell.localPath })
    }

    return success(c, { cells: results })
  } catch (err: any) {
    return badRequest(c, err.message)
  }
})

// GET /grid/status/:id
app.get('/status/:id', async (c) => {
  try {
  const id = parseParamId(c)
  if (id == null) return notFound(c, 'Invalid image generation id')
  const [row] = db.select().from(schema.imageGenerations)
    .where(eq(schema.imageGenerations.id, id)).all()
  if (!row) return notFound(c, 'Not found')
  return success(c, {
    id: row.id,
    status: row.status,
    local_path: row.localPath,
    image_url: row.imageUrl,
    error_msg: row.errorMsg,
  })
  } catch (err: any) { return c.json({ code: 500, data: null, message: err.message }) }
})

export default app
