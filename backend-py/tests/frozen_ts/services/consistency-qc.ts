/**
 * 图像连续性 QC（内容审片：画面穿帮检测）
 *
 * 对齐参考项目 Comic-drama 的 consistency_validator 思路：
 *  - 对同一集内相邻分镜画面做数值相似度检测（dHash + 颜色直方图，零新增模型依赖）
 *  - 同场景相邻镜头相似度过低 → warning（可能场景/光照/机位穿帮）
 *  - 跨场景相邻镜头仅记录 info（场景切换是正常行为，不判定）
 *
 * 策略：report 模式（对齐 Comic-drama CONSISTENCY_POLICY_MODE=report）。
 * 结果并入各分镜最新 video_quality_checks 的 continuity_vision 维度，只做展示与留痕，
 * 不参与总分/状态判定（避免误伤），未来可升级 block 模式。
 */
import sharp from 'sharp'
import { eq } from 'drizzle-orm'
import { db, schema } from '../db/index.js'
import { now } from '../utils/response.js'
import { getAbsolutePath } from '../utils/storage.js'
import { logTaskWarn } from '../utils/task-logger.js'

// ===== 一致性判定阈值（可调） =====
export const CONSISTENCY_QC_THRESHOLDS = {
  /** 同场景相邻镜头画面相似度下限：低于此值判 warning */
  sameSceneWarn: 0.55,
  /** 相似度低于此值判 info（介于 warn 与 ok 之间） */
  sameSceneInfo: 0.65,
  /** 同场景相邻镜头画面相似度上限：高于此值判 ok */
  sameSceneOk: 0.9,
  /** 跨场景相邻镜头：相似度过高反而可疑（场景变了画面几乎一样） */
  crossSceneInfo: 0.9,
} as const

export interface ConsistencyPairResult {
  prevStoryboardId: number
  curStoryboardId: number
  prevImage: string | null
  curImage: string | null
  sameScene: boolean
  similarity: number | null
  severity: 'warning' | 'info' | 'ok'
  message: string
}

export interface ConsistencyQcReport {
  episodeId: number
  dramaId: number | null
  checkedPairs: number
  warningCount: number
  pairs: ConsistencyPairResult[]
}

/** 计算图片 dHash（9x8 灰度 → 64bit），失败返回 null */
async function computeDHash(absPath: string): Promise<Buffer | null> {
  try {
    const { data } = await sharp(absPath, { failOn: 'none' })
      .resize(9, 8, { fit: 'fill' })
      .greyscale()
      .raw()
      .toBuffer({ resolveWithObject: true })
    const bits = Buffer.alloc(8)
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const left = data[y * 9 + x]
        const right = data[y * 9 + x + 1]
        if (right > left) bits[y] |= (1 << (7 - x))
      }
    }
    return bits
  } catch {
    return null
  }
}

function popcount(x: number): number {
  let c = 0
  while (x) { c += x & 1; x >>>= 1 }
  return c
}

/** 汉明距离 → 相似度（0~1） */
function similarity(a: Buffer, b: Buffer): number {
  let dist = 0
  for (let i = 0; i < 8; i++) dist += popcount(a[i] ^ b[i])
  return 1 - dist / 64
}

/** 从分镜行挑选用于比对的画面（composed 优先，其次首帧/关键帧） */
function pickImage(sb: any): string | null {
  const candidates = [sb.composedImage, sb.firstFrameImage, sb.keyframeImage]
  for (const c of candidates) {
    if (c && String(c).trim()) return String(c).trim()
  }
  return null
}

/**
 * 对一集所有分镜做相邻画面连续性检测。
 * 结果以 report 模式写回各分镜最新 video_quality_checks 的 continuity_vision 维度。
 */
export async function runEpisodeConsistencyQc(episodeId: number, dramaId?: number | null): Promise<ConsistencyQcReport> {
  const sbs = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, episodeId))
    .all()
    .sort((a, b) => (a.storyboardNumber ?? 0) - (b.storyboardNumber ?? 0))

  const report: ConsistencyQcReport = { episodeId, dramaId: dramaId ?? null, checkedPairs: 0, warningCount: 0, pairs: [] }
  if (sbs.length < 2) return report

  // 每张图只算一次 hash（避免相邻对重复计算）
  const hashCache = new Map<string, Buffer | null>()
  const getHash = async (relPath: string): Promise<Buffer | null> => {
    if (hashCache.has(relPath)) return hashCache.get(relPath)!
    let hash: Buffer | null = null
    try {
      const abs = getAbsolutePath(relPath)
      hash = await computeDHash(abs)
    } catch {
      hash = null
    }
    hashCache.set(relPath, hash)
    return hash
  }

  const pathToSb = new Map<number, string[]>() // storyboardId → 最近 QC 记录 id（写回用）
  const latestQc = new Map<number, { id: number; dimensions: Record<string, any>; issues: Array<any> }>()
  const loadQc = (sbId: number) => {
    if (latestQc.has(sbId)) return latestQc.get(sbId)!
    const qcs = db.select().from(schema.videoQualityChecks)
      .where(eq(schema.videoQualityChecks.storyboardId, sbId))
      .all()
      .sort((a, b) => b.id - a.id)
    const qc = qcs[0]
    let state = { id: 0, dimensions: {} as Record<string, any>, issues: [] as Array<any> }
    if (qc) {
      let dims: Record<string, any> = {}
      try { dims = qc.dimensions ? JSON.parse(qc.dimensions) : {} } catch {}
      let issues: Array<any> = []
      try { issues = qc.issues ? JSON.parse(qc.issues) : [] } catch {}
      state = { id: qc.id, dimensions: dims, issues }
    }
    latestQc.set(sbId, state)
    return state
  }

  for (let i = 1; i < sbs.length; i++) {
    const prev = sbs[i - 1]
    const cur = sbs[i]
    const prevImg = pickImage(prev)
    const curImg = pickImage(cur)
    if (!prevImg || !curImg) continue

    const [prevHash, curHash] = await Promise.all([getHash(prevImg), getHash(curImg)])
    if (!prevHash || !curHash) continue

    const sim = Math.round(similarity(prevHash, curHash) * 1000) / 1000
    const sameScene = !!prev.sceneId && prev.sceneId === cur.sceneId

    let severity: ConsistencyPairResult['severity'] = 'ok'
    let message: string
    if (sameScene) {
      if (sim < CONSISTENCY_QC_THRESHOLDS.sameSceneWarn) {
        severity = 'warning'
        message = `同场景（#${prev.sceneId}）相邻镜头画面相似度 ${sim} 过低（标准 ≥${CONSISTENCY_QC_THRESHOLDS.sameSceneWarn}），可能场景/光照/机位穿帮`
      } else if (sim < CONSISTENCY_QC_THRESHOLDS.sameSceneInfo) {
        severity = 'info'
        message = `同场景相邻镜头画面相似度 ${sim} 偏低，建议人工复核`
      } else {
        message = `同场景相邻镜头画面相似度 ${sim}，正常`
      }
    } else {
      if (sim > CONSISTENCY_QC_THRESHOLDS.crossSceneInfo) {
        severity = 'info'
        message = `跨场景切换但画面相似度 ${sim} 偏高，可能场景图复用/未切换`
      } else {
        severity = 'info'
        message = `跨场景相邻镜头，相似度 ${sim}（仅记录）`
      }
    }

    const pair: ConsistencyPairResult = {
      prevStoryboardId: prev.id,
      curStoryboardId: cur.id,
      prevImage: prevImg,
      curImage: curImg,
      sameScene,
      similarity: sim,
      severity,
      message,
    }
    report.pairs.push(pair)
    report.checkedPairs++
    if (severity === 'warning') report.warningCount++
    pathToSb.set(cur.id, [prevImg, curImg])

    // 写回当前镜头最新 QC 记录（report 模式，不改总分/状态）
    const state = loadQc(cur.id)
    if (state.id) {
      state.dimensions.continuity_vision = {
        checked: true,
        severity,
        similarity: sim,
        sameScene,
        prevStoryboardId: prev.id,
        notes: [message],
      }
      if (!state.issues.some((x: any) => x.dimension === 'continuity_vision' && x.message === message)) {
        state.issues.push({ dimension: 'continuity_vision', severity, message })
      }
    }
  }

  // 统一落库（仅对有 QC 记录的分镜）
  const ts = now()
  for (const [, state] of latestQc) {
    if (!state.id) continue
    try {
      db.update(schema.videoQualityChecks)
        .set({ dimensions: JSON.stringify(state.dimensions), issues: JSON.stringify(state.issues), updatedAt: ts })
        .where(eq(schema.videoQualityChecks.id, state.id))
        .run()
    } catch (err: any) {
      logTaskWarn('ConsistencyQc', 'writeback-failed', { qcId: state.id, error: err?.message || String(err) })
    }
  }

  return report
}
