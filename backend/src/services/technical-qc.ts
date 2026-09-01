/**
 * 技术维度 QC（视频规格硬性数值检测）
 *
 * 对齐参考项目（MiniMax-H3-Codex-Drama）的 QC 硬性标准：
 *   - 黑场间隔（black interval）：> 0.25s 记为缺陷
 *   - 冻帧（freeze）：> 1.0s 记为缺陷
 *   - 音频真峰（true peak）：> -1dBTP 记为缺陷
 *   - 集成响度（integrated loudness）：偏离 -14 LUFS ±1 记为缺陷
 *   - 帧率（fps）：明显异常（< 15）记为缺陷
 *   - 时长偏差：与期望值偏差过大记为缺陷
 *
 * 仅在本地存在视频文件时执行（webhook/轮询已完成下载）。
 * 结果为 fire-and-forget，写入 video_quality_checks 记录的 video_spec 维度。
 */
import { and, eq } from 'drizzle-orm'
import { execFile } from 'child_process'
import { db, schema } from '../db/index.js'
import { now } from '../utils/response.js'
import { getAbsolutePath } from '../utils/storage.js'
import { logTaskWarn } from '../utils/task-logger.js'

// ===== 硬性数值标准（可调） =====
export const TECH_QC_THRESHOLDS = {
  blackMinDuration: 0.25, // 单段黑场最小判定时长（s）
  blackPixThreshold: 0.1, // 黑场像素阈值
  freezeMinDuration: 1.0, // 单段冻帧最小判定时长（s）
  freezeNoiseThreshold: -60, // 冻帧噪声阈值（dB）
  truePeakDb: -1, // 音频真峰上限（dBTP）
  loudnessTarget: -14, // 集成响度目标（LUFS）
  loudnessTolerance: 1, // 集成响度容差（LUFS）
  fpsMin: 15, // 合理帧率下限
  durationToleranceSec: 1, // 时长偏差容差（s）
} as const

export interface TechQcResult {
  duration: number | null
  fps: number | null
  hasAudio: boolean
  maxVolumeDb: number | null // 近似真峰（volumedetect max_volume）
  integratedLoudness: number | null // ebur128 集成响度（若可用）
  blackSegments: Array<{ start: number; duration: number }>
  freezeSegments: Array<{ start: number; duration: number }>
  issues: Array<{ severity: 'error' | 'warning' | 'info'; message: string }>
  score: number
  error?: string
}

function execFileP(bin: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    execFile(bin, args, { maxBuffer: 16 * 1024 * 1024, windowsHide: true }, (err, stdout, stderr) => {
      // ffmpeg 滤镜检测正常结束会输出到 stderr；非零退出码才视为失败
      if (err && !stderr) reject(err)
      else resolve({ stdout: String(stdout || ''), stderr: String(stderr || '') })
    })
  })
}

/** 解析黑场检测输出（stderr 中 black_start/black_end/black_duration 行） */
function parseBlackSegments(stderr: string): Array<{ start: number; duration: number }> {
  const segs: Array<{ start: number; duration: number }> = []
  const re = /black_start:\s*([0-9.]+)\s*black_end:\s*([0-9.]+)\s*black_duration:\s*([0-9.]+)/g
  let m: RegExpExecArray | null
  while ((m = re.exec(stderr)) !== null) {
    segs.push({ start: Number(m[1]), duration: Number(m[3]) })
  }
  return segs
}

/** 解析冻帧检测输出（stderr 中 freeze_start/freeze_duration 行） */
function parseFreezeSegments(stderr: string): Array<{ start: number; duration: number }> {
  const segs: Array<{ start: number; duration: number }> = []
  const re = /freeze_start:\s*([0-9.]+)\s*freeze_duration:\s*([0-9.]+)/g
  let m: RegExpExecArray | null
  while ((m = re.exec(stderr)) !== null) {
    segs.push({ start: Number(m[1]), duration: Number(m[2]) })
  }
  return segs
}

/** 解析 volumedetect 输出（stderr 中 mean_volume/max_volume 行） */
function parseVolume(stderr: string): { mean: number | null; max: number | null } {
  const meanM = /mean_volume:\s*(-?[0-9.]+) dB/.exec(stderr)
  const maxM = /max_volume:\s*(-?[0-9.]+) dB/.exec(stderr)
  return { mean: meanM ? Number(meanM[1]) : null, max: maxM ? Number(maxM[1]) : null }
}

/** 解析 ebur128 summary（stderr 中 Integrated loudness / True peak 行） */
function parseEbur128(stderr: string): { integrated: number | null; truePeak: number | null } {
  const intM = /Integrated loudness:\s*(-?[0-9.]+) LUFS/.exec(stderr)
  const peakM = /True peak:\s*(-?[0-9.]+) dBTP/.exec(stderr)
  return { integrated: intM ? Number(intM[1]) : null, truePeak: peakM ? Number(peakM[1]) : null }
}

/**
 * 对本地视频文件执行技术规格检测。
 * @param localPath 相对路径（static/...）或绝对路径
 * @param expectedDuration 期望时长（分镜声明），可为空
 */
export async function probeVideoTechnicalSpec(localPath: string, expectedDuration?: number | null): Promise<TechQcResult> {
  let absPath: string
  try {
    absPath = getAbsolutePath(localPath)
  } catch (err: any) {
    return { duration: null, fps: null, hasAudio: false, maxVolumeDb: null, integratedLoudness: null, blackSegments: [], freezeSegments: [], issues: [], score: 100, error: err.message }
  }

  const result: TechQcResult = {
    duration: null,
    fps: null,
    hasAudio: false,
    maxVolumeDb: null,
    integratedLoudness: null,
    blackSegments: [],
    freezeSegments: [],
    issues: [],
    score: 100,
  }

  // ===== 1. ffprobe：时长 / 帧率 / 音轨存在性 =====
  try {
    const probe = await execFileP('ffprobe', [
      '-v', 'error',
      '-select_streams', 'v:0',
      '-show_entries', 'stream=avg_frame_rate:format=duration',
      '-of', 'json', absPath,
    ])
    const meta = JSON.parse(probe.stdout || '{}')
    const stream = meta?.streams?.[0]
    result.duration = Number(meta?.format?.duration) || null
    if (stream?.avg_frame_rate && stream.avg_frame_rate !== '0/0') {
      const [n, d] = String(stream.avg_frame_rate).split('/').map(Number)
      if (n && d) result.fps = Number((n / d).toFixed(3))
    }
  } catch {
    result.score -= 15
    result.issues.push({ severity: 'warning', message: '无法读取视频规格（ffprobe 失败），跳过技术检测' })
    return result
  }

  // ===== 2. 黑场检测（blackdetect） =====
  try {
    const { stderr } = await execFileP('ffmpeg', [
      '-i', absPath,
      '-vf', `blackdetect=d=${TECH_QC_THRESHOLDS.blackMinDuration}:pix_th=${TECH_QC_THRESHOLDS.blackPixThreshold}`,
      '-an', '-f', 'null', '-',
    ])
    result.blackSegments = parseBlackSegments(stderr)
    for (const seg of result.blackSegments) {
      result.score -= 20
      result.issues.push({
        severity: 'error',
        message: `画面黑场 ${seg.duration.toFixed(2)}s（第 ${seg.start.toFixed(1)}s 起，标准 >${TECH_QC_THRESHOLDS.blackMinDuration}s 即判缺陷）`,
      })
    }
  } catch { /* 忽略检测失败，不重复扣分 */ }

  // ===== 3. 冻帧检测（freezedetect） =====
  try {
    const { stderr } = await execFileP('ffmpeg', [
      '-i', absPath,
      '-vf', `freezedetect=n=${TECH_QC_THRESHOLDS.freezeNoiseThreshold}dB:d=${TECH_QC_THRESHOLDS.freezeMinDuration}`,
      '-map', '0:v', '-f', 'null', '-',
    ])
    result.freezeSegments = parseFreezeSegments(stderr)
    for (const seg of result.freezeSegments) {
      result.score -= 20
      result.issues.push({
        severity: 'error',
        message: `画面冻帧 ${seg.duration.toFixed(2)}s（第 ${seg.start.toFixed(1)}s 起，标准 >${TECH_QC_THRESHOLDS.freezeMinDuration}s 即判缺陷）`,
      })
    }
  } catch { /* 忽略 */ }

  // ===== 4. 音频响度检测（volumedetect 近似真峰 + ebur128 集成响度） =====
  try {
    const { stderr } = await execFileP('ffmpeg', ['-i', absPath, '-af', 'volumedetect', '-f', 'null', '-'])
    const { max } = parseVolume(stderr)
    result.maxVolumeDb = max
    if (max != null) result.hasAudio = true
  } catch { /* 无音轨或检测失败 */ }
  if (result.hasAudio) {
    try {
      const { stderr } = await execFileP('ffmpeg', ['-i', absPath, '-af', 'ebur128=peak=true', '-f', 'null', '-'])
      const { integrated, truePeak } = parseEbur128(stderr)
      result.integratedLoudness = integrated
      if (truePeak != null && truePeak > TECH_QC_THRESHOLDS.truePeakDb) {
        result.score -= 15
        result.issues.push({ severity: 'warning', message: `音频真峰 ${truePeak.toFixed(1)}dBTP 超过 ${TECH_QC_THRESHOLDS.truePeakDb}dBTP 标准（防削波）` })
      }
      if (integrated != null && Math.abs(integrated - TECH_QC_THRESHOLDS.loudnessTarget) > TECH_QC_THRESHOLDS.loudnessTolerance) {
        result.score -= 10
        result.issues.push({ severity: 'info', message: `集成响度 ${integrated.toFixed(1)}LUFS 偏离标准 ${TECH_QC_THRESHOLDS.loudnessTarget}±${TECH_QC_THRESHOLDS.loudnessTolerance}LUFS` })
      }
    } catch { /* ebur128 可能因多声道失败，忽略 */ }
  }

  // ===== 5. 帧率 / 时长合理性 =====
  if (result.fps != null && result.fps < TECH_QC_THRESHOLDS.fpsMin) {
    result.score -= 10
    result.issues.push({ severity: 'warning', message: `帧率 ${result.fps}fps 过低（标准 ≥${TECH_QC_THRESHOLDS.fpsMin}fps）` })
  }
  if (expectedDuration && result.duration != null) {
    const diff = Math.abs(result.duration - expectedDuration)
    if (diff > TECH_QC_THRESHOLDS.durationToleranceSec) {
      result.score -= 10
      result.issues.push({ severity: 'warning', message: `视频实际时长 ${result.duration.toFixed(1)}s 与期望 ${expectedDuration}s 偏差 ${diff.toFixed(1)}s（标准 ≤${TECH_QC_THRESHOLDS.durationToleranceSec}s）` })
    }
  }

  result.score = Math.max(0, Math.min(100, Math.round(result.score)))
  return result
}

/**
 * 技术 QC（fire-and-forget）：视频生成完成后对本地成品做硬性规格检测，
 * 结果并入该分镜最近一条 video_quality_checks 记录的 video_spec 维度。
 */
export async function runTechnicalQc(storyboardId: number, videoGenerationId: number): Promise<void> {
  try {
    const [gen] = db
      .select()
      .from(schema.videoGenerations)
      .where(and(eq(schema.videoGenerations.id, videoGenerationId), eq(schema.videoGenerations.status, 'completed')))
      .all()
    if (!gen?.localPath) {
      logTaskWarn('TechQc', 'no-local-file', { storyboardId, videoGenerationId })
      return
    }
    const [sb] = db.select().from(schema.storyboards).where(eq(schema.storyboards.id, storyboardId)).all()
    const spec = await probeVideoTechnicalSpec(gen.localPath, sb?.duration)

    // 找到该分镜最新 QC 记录，读取并合并 video_spec 维度
    const qcs = db
      .select()
      .from(schema.videoQualityChecks)
      .where(eq(schema.videoQualityChecks.storyboardId, storyboardId))
      .all()
      .sort((a, b) => b.id - a.id)
    const qc = qcs[0]
    if (!qc) return

    let dims: Record<string, any> = {}
    try { dims = qc.dimensions ? JSON.parse(qc.dimensions) : {} } catch {}
    dims.video_spec = {
      score: spec.score,
      checked: true,
      notes: spec.issues.length ? spec.issues.map((i) => i.message) : ['技术规格检测通过'],
      duration: spec.duration,
      fps: spec.fps,
      hasAudio: spec.hasAudio,
      maxVolumeDb: spec.maxVolumeDb,
      integratedLoudness: spec.integratedLoudness,
      blackSegments: spec.blackSegments.length,
      freezeSegments: spec.freezeSegments.length,
    }

    let issues: Array<{ dimension: string; severity: string; message: string }> = []
    try { issues = qc.issues ? JSON.parse(qc.issues) : [] } catch {}
    for (const i of spec.issues) {
      if (!issues.some((x) => x.dimension === 'video_spec' && x.message === i.message)) {
        issues.push({ dimension: 'video_spec', severity: i.severity, message: i.message })
      }
    }

    // 重算总分：原三维平均 70% + 技术维度 30%
    const ruleScores = [dims.lip_sync?.score ?? 0, dims.character_consistency?.score ?? 0, dims.continuity?.score ?? 0]
    const ruleAvg = ruleScores.reduce((a, b) => a + b, 0) / 3
    const overall = Math.round(ruleAvg * 0.7 + spec.score * 0.3)
    const hasError = issues.some((i) => i.severity === 'error')
    const status = hasError ? 'failed' : 'passed'

    const ts = now()
    db.update(schema.videoQualityChecks)
      .set({ dimensions: JSON.stringify(dims), issues: JSON.stringify(issues), overallScore: overall, status, updatedAt: ts })
      .where(eq(schema.videoQualityChecks.id, qc.id))
      .run()
  } catch (err: any) {
    logTaskWarn('TechQc', 'run-failed', { storyboardId, videoGenerationId, error: err?.message || String(err) })
  }
}
