/**
 * 资产版本历史/回滚（对齐参考项目 ArcReel artifact_version_provenance / version_restore）
 *
 * 每次图片/视频生成成功自动留档一条 asset_versions 记录（current）；
 * 同一资产再次生成时旧版本自动降为 historical，新版本成为 current。
 * 回滚 = 把某历史版本重新置为 current，并把其 asset_url 写回主表字段。
 *
 * 资产标识：asset_type（storyboard/character/scene/prop）+ asset_id
 * 分镜图片还按 frame_type（composed/first_frame/last_frame/keyframe）分组版本，
 * 即同一条分镜的合成图、首帧、尾帧各自独立版本历史。
 */
import { and, eq, isNull, desc } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { now } from '../utils/response.js'
import { logTaskWarn } from '../utils/task-logger.js'

export type AssetType = 'storyboard' | 'character' | 'scene' | 'prop'
export type MediaType = 'image' | 'video'
export type StoryboardFrameType = 'composed' | 'first_frame' | 'last_frame' | 'keyframe' | null

export interface RecordAssetVersionInput {
  assetType: AssetType
  assetId: number
  mediaType: MediaType
  frameType?: StoryboardFrameType
  assetUrl: string
  provider?: string | null
  model?: string | null
  prompt?: string | null
  generationId?: number | null
  meta?: Record<string, unknown>
}

/** 生成成功时留档：旧版本降级为 historical，新版本成为 current */
export function recordAssetVersion(input: RecordAssetVersionInput): number | null {
  try {
    const frameType = input.frameType ?? null
    const existing = db.select().from(schema.assetVersions)
      .where(and(
        eq(schema.assetVersions.assetType, input.assetType),
        eq(schema.assetVersions.assetId, input.assetId),
        eq(schema.assetVersions.mediaType, input.mediaType),
        frameType === null ? isNull(schema.assetVersions.frameType) : eq(schema.assetVersions.frameType, frameType),
      ))
      .all()

    // 旧的 current 降级为 historical
    const currentIds = existing.filter((v) => v.status === 'current').map((v) => v.id)
    if (currentIds.length) {
      for (const id of currentIds) {
        db.update(schema.assetVersions)
          .set({ status: 'historical' })
          .where(eq(schema.assetVersions.id, id)).run()
      }
    }

    const nextVersion = existing.length
      ? Math.max(...existing.map((v) => v.version)) + 1
      : 1

    const ts = now()
    const res = db.insert(schema.assetVersions).values({
      assetType: input.assetType,
      assetId: input.assetId,
      mediaType: input.mediaType,
      frameType,
      version: nextVersion,
      assetUrl: input.assetUrl,
      provider: input.provider ?? null,
      model: input.model ?? null,
      prompt: input.prompt ?? null,
      generationId: input.generationId ?? null,
      meta: input.meta ? JSON.stringify(input.meta) : null,
      status: 'current',
      createdAt: ts,
    }).run()
    return Number(res.lastInsertRowid)
  } catch (err: any) {
    logTaskWarn('AssetVersions', 'record-failed', {
      assetType: input.assetType, assetId: input.assetId,
      mediaType: input.mediaType, frameType: input.frameType ?? null,
      error: err?.message || String(err),
    })
    return null
  }
}

export interface AssetVersionRow {
  id: number
  assetType: string
  assetId: number
  mediaType: string
  frameType: string | null
  version: number
  assetUrl: string
  provider: string | null
  model: string | null
  prompt: string | null
  generationId: number | null
  meta: string | null
  status: string
  createdAt: string
}

/** 资产版本列表（版本号倒序，最新在前） */
export function listAssetVersions(assetType: string, assetId: number): AssetVersionRow[] {
  return db.select().from(schema.assetVersions)
    .where(and(
      eq(schema.assetVersions.assetType, assetType),
      eq(schema.assetVersions.assetId, assetId),
    ))
    .orderBy(desc(schema.assetVersions.version))
    .all() as unknown as AssetVersionRow[]
}

/** 回滚：把指定版本置为 current，asset_url 写回主表对应字段 */
export function activateAssetVersion(versionId: number): { ok: boolean; error?: string; row?: AssetVersionRow } {
  const [row] = db.select().from(schema.assetVersions)
    .where(eq(schema.assetVersions.id, versionId)).all()
  if (!row) return { ok: false, error: 'Version not found' }

  const asRow = row as unknown as AssetVersionRow
  const ts = now()

  // 同组其他 current 降级
  db.update(schema.assetVersions)
    .set({ status: 'historical' })
    .where(and(
      eq(schema.assetVersions.assetType, asRow.assetType),
      eq(schema.assetVersions.assetId, asRow.assetId),
      eq(schema.assetVersions.mediaType, asRow.mediaType),
      asRow.frameType === null
        ? isNull(schema.assetVersions.frameType)
        : eq(schema.assetVersions.frameType, asRow.frameType),
      eq(schema.assetVersions.status, 'current'),
    )).run()

  db.update(schema.assetVersions)
    .set({ status: 'current' })
    .where(eq(schema.assetVersions.id, versionId)).run()

  // 写回主表
  try {
    applyAssetToEntity(asRow, ts)
  } catch (err: any) {
    logTaskWarn('AssetVersions', 'apply-to-entity-failed', {
      versionId, error: err?.message || String(err),
    })
    return { ok: false, error: err?.message || String(err), row: asRow }
  }

  return { ok: true, row: asRow }
}

/** 把资产 URL 写回对应主表字段（回滚生效） */
function applyAssetToEntity(row: AssetVersionRow, ts: string): void {
  switch (row.assetType) {
    case 'storyboard': {
      const update: Record<string, any> = { updatedAt: ts }
      if (row.mediaType === 'video') {
        update.videoUrl = row.assetUrl
      } else if (row.frameType === 'first_frame') {
        update.firstFrameImage = row.assetUrl
      } else if (row.frameType === 'last_frame') {
        update.lastFrameImage = row.assetUrl
      } else if (row.frameType === 'keyframe') {
        update.keyframeImage = row.assetUrl
      } else {
        update.composedImage = row.assetUrl
      }
      db.update(schema.storyboards).set(update).where(eq(schema.storyboards.id, row.assetId)).run()
      return
    }
    case 'character': {
      db.update(schema.characters).set({ imageUrl: row.assetUrl, updatedAt: ts })
        .where(eq(schema.characters.id, row.assetId)).run()
      return
    }
    case 'scene': {
      db.update(schema.scenes).set({ imageUrl: row.assetUrl, status: 'completed', updatedAt: ts })
        .where(eq(schema.scenes.id, row.assetId)).run()
      return
    }
    case 'prop': {
      db.update(schema.propTemplates).set({ imageUrl: row.assetUrl, updatedAt: ts })
        .where(eq(schema.propTemplates.id, row.assetId)).run()
      return
    }
    default:
      logTaskWarn('AssetVersions', 'unknown-asset-type', { assetType: row.assetType })
  }
}

/** 从分镜生成记录推导分镜图片的 frameType（无则视为 composed） */
export function resolveStoryboardFrameType(frameType?: string | null): StoryboardFrameType {
  if (frameType === 'first_frame' || frameType === 'last_frame' || frameType === 'keyframe') {
    return frameType
  }
  return 'composed'
}
