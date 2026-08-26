import { and, eq, isNull } from 'drizzle-orm'
import { db, schema } from '../db/index.js'

function pad3(n: number): string {
  return String(n).padStart(3, '0')
}

/**
 * 剧级视觉风格 ID（六键 Bible 的 STYLE_ID，跨集锁定，一剧一 ID）
 * 由 style 文本归一化而来，后续跨集复用同一 ID。
 */
export function ensureStyleId(dramaId: number): string {
  const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()
  if (!drama) return ''
  if (drama.styleId) return drama.styleId
  const styleId = `STYLE_${pad3(dramaId)}`
  db.update(schema.dramas).set({ styleId }).where(eq(schema.dramas.id, dramaId)).run()
  return styleId
}

/**
 * 角色标准服装 ID（六键 Bible 的 COSTUME_ID，跨集锁定，一角色一 ID）
 * 角色换装时才显式变更 costumeId。
 */
export function ensureCostumeId(characterId: number): string {
  const [char] = db.select().from(schema.characters).where(eq(schema.characters.id, characterId)).all()
  if (!char) return ''
  if (char.costumeId) return char.costumeId
  const costumeId = `COST_${pad3(characterId)}`
  db.update(schema.characters).set({ costumeId }).where(eq(schema.characters.id, characterId)).run()
  return costumeId
}

/**
 * 场景地点 ID（六键 Bible 的 LOCATION_ID，跨集锁定）
 * 同一 drama 内 location 文本归一化相同 → 复用同一 ID，否则生成新 ID。
 */
export function ensureLocationId(sceneId: number): string {
  const [scene] = db.select().from(schema.scenes).where(eq(schema.scenes.id, sceneId)).all()
  if (!scene) return ''
  if (scene.locationId) return scene.locationId
  const norm = (scene.location || '').trim()
  const siblings = db
    .select()
    .from(schema.scenes)
    .where(and(eq(schema.scenes.dramaId, scene.dramaId), isNull(schema.scenes.deletedAt)))
    .all()

  // 复用同地点已有 ID
  const existing = siblings.find(
    (s) => s.id !== scene.id && !!s.locationId && (s.location || '').trim() === norm,
  )

  let locationId: string
  if (existing) {
    locationId = existing.locationId!
  } else {
    const maxSeq = siblings.reduce((max, s) => {
      const m = s.locationId?.match(/^LOC_(\d+)$/)
      return m ? Math.max(max, Number(m[1])) : max
    }, 0)
    locationId = `LOC_${pad3(maxSeq + 1)}`
  }

  db.update(schema.scenes).set({ locationId }).where(eq(schema.scenes.id, sceneId)).run()
  return locationId
}
