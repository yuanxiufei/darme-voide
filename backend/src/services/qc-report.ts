/**
 * QC 报告交付物服务 — 对齐参考项目「QC 报告交付物：联系表 + 媒体信息 + QC 报告 + 时间线导出文件」
 *
 * 提供三类交付物：
 * 1. Contact Sheet（联系表）：每镜缩略图（首帧/尾帧/关键帧）+ 元信息网格，
 *    供制片/导演快速浏览整集画面节奏与完成度。
 * 2. QC 报告：聚合 video_quality_checks（唇形/角色一致性/连续性/总体分）+ 技术维度，
 *    输出每镜 QC 状态与问题清单，支持 Markdown / HTML / JSON。
 * 3. Media Info（媒体信息）：每镜视频/音频文件的规格（分辨率/帧率/时长/码率/编码），
 *    从 video_generations 表读取实测值；未生成时回退 ffprobe 探测。
 *
 * 时间线导出文件已由 EDL（export-service）与剪映草稿（jianying-draft）承担，
 * 本模块的 QC 报告包含 timeline 汇总（每镜时长/起始/路由/状态），补足「交付物」闭环。
 */
import { and, eq, isNull } from 'drizzle-orm'
import { execFile } from 'child_process'
import { db, schema } from '../db/index.js'
import { logTaskWarn } from '../utils/task-logger.js'
import { probeVideoDuration } from '../utils/video-probe.js'
import { getAbsolutePath } from '../utils/storage.js'
import fs from 'fs'

/** 调 ffprobe 获取媒体规格（JSON 输出），失败返回 null */
async function ffprobeInfo(absPath: string): Promise<MediaInfo | null> {
  return new Promise((resolve) => {
    execFile(
      'ffprobe',
      ['-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', absPath],
      { maxBuffer: 16 * 1024 * 1024, windowsHide: true },
      (err, stdout) => {
        if (err) return resolve(null)
        try {
          const j = JSON.parse(String(stdout || ''))
          const vStream = (j.streams || []).find((s: any) => s.codec_type === 'video')
          const aStream = (j.streams || []).find((s: any) => s.codec_type === 'audio')
          const fmt = j.format || {}
          const width = vStream?.width ? Number(vStream.width) : null
          const height = vStream?.height ? Number(vStream.height) : null
          const fpsRaw = vStream?.avg_frame_rate || vStream?.r_frame_rate
          let fps: number | null = null
          if (fpsRaw && fpsRaw !== '0/0') {
            const [n, d] = String(fpsRaw).split('/').map(Number)
            if (n && d) fps = Math.round((n / d) * 100) / 100
          }
          const durationSec = fmt?.duration ? Number(fmt.duration) : null
          const bitrate = fmt?.bit_rate ? `${(Number(fmt.bit_rate) / 1024).toFixed(0)} kbps` : null
          const fileSize = fmt?.size ? Number(fmt.size) : null
          return resolve({
            durationSec,
            fps,
            resolution: width && height ? `${width}x${height}` : null,
            width,
            height,
            bitrate,
            codec: vStream?.codec_name || null,
            audioCodec: aStream?.codec_name || null,
            fileSize,
          })
        } catch {
          return resolve(null)
        }
      },
    )
  })
}

// ===== 类型 =====

export interface QcShotEntry {
  storyboardNumber: number
  title: string | null
  description: string | null
  shotType: string | null
  angle: string | null
  movement: string | null
  route: string | null
  routeReason: string | null
  duration: number | null
  durationSec: number
  dialogue: string | null
  status: string | null
  assetStatus: string | null
  takeCount: number | null
  takeBudget: number | null
  videoUrl: string | null
  composedVideoUrl: string | null
  ttsAudioUrl: string | null
  firstFrameImage: string | null
  lastFrameImage: string | null
  keyframeImage: string | null
  videoGen: VideoGenInfo | null
  qc: QcInfo | null
  media: MediaInfo | null
}

export interface VideoGenInfo {
  provider: string | null
  model: string | null
  duration: number | null
  fps: number | null
  resolution: string | null
  width: number | null
  height: number | null
  status: string | null
  seed: number | null
}

export interface QcInfo {
  lipSyncScore: number | null
  characterConsistencyScore: number | null
  continuityScore: number | null
  overallScore: number | null
  status: string | null
  issues: string[]
  dimensions: Record<string, any> | null
}

export interface MediaInfo {
  durationSec: number | null
  fps: number | null
  resolution: string | null
  width: number | null
  height: number | null
  bitrate: string | null
  codec: string | null
  audioCodec: string | null
  fileSize: number | null
}

export interface QcReport {
  dramaId: number
  dramaTitle: string | null
  episodeId: number | null
  episodeNumber: number | null
  episodeTitle: string | null
  generatedAt: string
  summary: {
    totalShots: number
    shotsWithVideo: number
    shotsWithQc: number
    avgLipSync: number | null
    avgCharacterConsistency: number | null
    avgContinuity: number | null
    avgOverall: number | null
    passed: number
    failed: number
  }
  shots: QcShotEntry[]
  timeline: Array<{ storyboardNumber: number; startSec: number; durationSec: number; status: string | null }>
}

// ===== 工具 =====

function parseJsonArray<T>(raw: string | null): T[] {
  if (!raw) return []
  try {
    const v = JSON.parse(raw)
    return Array.isArray(v) ? v : []
  } catch {
    return []
  }
}

function parseJsonObj(raw: string | null): Record<string, any> | null {
  if (!raw) return null
  try {
    const v = JSON.parse(raw)
    return v && typeof v === 'object' ? v : null
  } catch {
    return null
  }
}

function pct(score: number | null): number | null {
  return score == null ? null : Math.max(0, Math.min(100, score))
}

/** 本地媒体绝对路径（static/... 或相对 data 根）；远程/未落盘返回 null */
function toLocalAbsPath(p: string | null | undefined): string | null {
  if (!p) return null
  if (p.startsWith('data:') || p.startsWith('http://') || p.startsWith('https://')) return null
  try {
    const abs = getAbsolutePath(p)
    return fs.existsSync(abs) ? abs : null
  } catch {
    return null
  }
}

/** 取分镜最新（按 created_at 倒序）一条视频生成记录 */
function latestVideoGen(storyboardId: number): any | null {
  const rows = db
    .select()
    .from(schema.videoGenerations)
    .where(eq(schema.videoGenerations.storyboardId, storyboardId))
    .orderBy(schema.videoGenerations.createdAt)
    .all()
  return rows.length ? rows[rows.length - 1] : null
}

/** 取分镜最新一条 QC 记录 */
function latestQc(storyboardId: number): any | null {
  const rows = db
    .select()
    .from(schema.videoQualityChecks)
    .where(eq(schema.videoQualityChecks.storyboardId, storyboardId))
    .orderBy(schema.videoQualityChecks.createdAt)
    .all()
  return rows.length ? rows[rows.length - 1] : null
}

/** 媒体信息：优先读视频生成记录的实测规格，fallback ffprobe 探测 */
async function buildMediaInfo(sb: any, vg: any): Promise<MediaInfo | null> {
  let durationSec: number | null = vg?.duration || null
  let fps: number | null = vg?.fps || null
  let resolution = vg?.resolution || null
  let width = vg?.width || null
  let height = vg?.height || null
  let bitrate: string | null = null
  let codec: string | null = null
  let audioCodec: string | null = null
  let fileSize: number | null = null

  const src = toLocalAbsPath(sb?.videoUrl || sb?.composedVideoUrl)
  if (src) {
    const info = await ffprobeInfo(src)
    if (info) {
      durationSec = durationSec ?? info.durationSec ?? null
      fps = fps ?? info.fps ?? null
      resolution = resolution ?? info.resolution ?? null
      width = width ?? info.width ?? null
      height = height ?? info.height ?? null
      bitrate = bitrate ?? info.bitrate ?? null
      codec = codec ?? info.codec ?? null
      audioCodec = audioCodec ?? info.audioCodec ?? null
      fileSize = fileSize ?? info.fileSize ?? null
    }
    if (fileSize == null) {
      try {
        fileSize = fs.statSync(src).size
      } catch {
        /* ignore */
      }
    }
  }

  if (durationSec == null && sb?.duration) durationSec = sb.duration

  return {
    durationSec,
    fps,
    resolution,
    width,
    height,
    bitrate,
    codec,
    audioCodec,
    fileSize,
  }
}

// ===== 主构建 =====

/** 构建 QC 报告（结构化数据） */
export async function buildQcReport(
  dramaId: number,
  opts?: { episodeId?: number },
): Promise<QcReport> {
  const [drama] = db.select().from(schema.dramas).where(eq(schema.dramas.id, dramaId)).all()

  const epWhere = opts?.episodeId
    ? and(eq(schema.episodes.id, opts.episodeId), eq(schema.episodes.dramaId, dramaId))
    : eq(schema.episodes.dramaId, dramaId)
  const episodes = db
    .select()
    .from(schema.episodes)
    .where(and(epWhere, isNull(schema.episodes.deletedAt)))
    .orderBy(schema.episodes.episodeNumber)
    .all()

  if (!episodes.length) throw new Error('No episodes found for QC report')

  // 支持多集时聚合全部；report.episodeId 为 opts 指定或首集
  const episodeId = opts?.episodeId ?? episodes[0].id
  const ep = episodes.find((e) => e.id === episodeId) || episodes[0]

  const storyboards = db
    .select()
    .from(schema.storyboards)
    .where(and(eq(schema.storyboards.episodeId, ep.id), isNull(schema.storyboards.deletedAt)))
    .orderBy(schema.storyboards.storyboardNumber)
    .all()

  const shots: QcShotEntry[] = []
  const timeline: QcReport['timeline'] = []
  let cursorSec = 0

  for (const sb of storyboards) {
    const vg = latestVideoGen(sb.id)
    const qc = latestQc(sb.id)
    const media = await buildMediaInfo(sb, vg)

    let durationSec = sb.duration || media?.durationSec || 0
    if (durationSec <= 0 && sb.videoUrl) {
      try {
        durationSec = await probeVideoDuration(sb.videoUrl)
      } catch {
        /* ignore */
      }
    }

    timeline.push({
      storyboardNumber: sb.storyboardNumber,
      startSec: cursorSec,
      durationSec,
      status: sb.status,
    })
    cursorSec += durationSec

    shots.push({
      storyboardNumber: sb.storyboardNumber,
      title: sb.title,
      description: sb.description,
      shotType: sb.shotType,
      angle: sb.angle,
      movement: sb.movement,
      route: sb.route,
      routeReason: sb.routeReason,
      duration: sb.duration,
      durationSec,
      dialogue: sb.dialogue,
      status: sb.status,
      assetStatus: sb.assetStatus,
      takeCount: sb.takeCount,
      takeBudget: sb.takeBudget,
      videoUrl: sb.videoUrl,
      composedVideoUrl: sb.composedVideoUrl,
      ttsAudioUrl: sb.ttsAudioUrl,
      firstFrameImage: sb.firstFrameImage,
      lastFrameImage: sb.lastFrameImage,
      keyframeImage: sb.keyframeImage,
      videoGen: vg
        ? {
            provider: vg.provider,
            model: vg.model,
            duration: vg.duration,
            fps: vg.fps,
            resolution: vg.resolution,
            width: vg.width,
            height: vg.height,
            status: vg.status,
            seed: vg.seed,
          }
        : null,
      qc: qc
        ? {
            lipSyncScore: pct(qc.lipSyncScore),
            characterConsistencyScore: pct(qc.characterConsistencyScore),
            continuityScore: pct(qc.continuityScore),
            overallScore: pct(qc.overallScore),
            status: qc.status,
            issues: parseJsonArray<string>(qc.issues),
            dimensions: parseJsonObj(qc.dimensions),
          }
        : null,
      media,
    })
  }

  // 汇总
  const withQc = shots.filter((s) => s.qc)
  const avg = (sel: (s: QcShotEntry) => number | null) => {
    const vals = withQc.map(sel).filter((v): v is number => v != null)
    if (!vals.length) return null
    return Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 10) / 10
  }
  const passed = shots.filter((s) => s.qc && (s.qc!.overallScore ?? 0) >= 60).length
  const failed = shots.filter((s) => s.qc && (s.qc!.overallScore ?? 0) < 60).length

  const report: QcReport = {
    dramaId,
    dramaTitle: drama?.title || null,
    episodeId: ep.id,
    episodeNumber: ep.episodeNumber,
    episodeTitle: ep.title || null,
    generatedAt: new Date().toISOString(),
    summary: {
      totalShots: shots.length,
      shotsWithVideo: shots.filter((s) => s.videoUrl || s.composedVideoUrl).length,
      shotsWithQc: withQc.length,
      avgLipSync: avg((s) => s.qc?.lipSyncScore ?? null),
      avgCharacterConsistency: avg((s) => s.qc?.characterConsistencyScore ?? null),
      avgContinuity: avg((s) => s.qc?.continuityScore ?? null),
      avgOverall: avg((s) => s.qc?.overallScore ?? null),
      passed,
      failed,
    },
    shots,
    timeline,
  }
  return report
}

// ===== Markdown 渲染 =====

/** 渲染为 Markdown QC 报告（可交付给制片/审计） */
export async function buildQcReportMarkdown(dramaId: number, opts?: { episodeId?: number }): Promise<string> {
  const report = await buildQcReport(dramaId, opts)
  const L: string[] = []
  L.push(`# QC 报告：${report.dramaTitle || `Drama #${report.dramaId}`}`)
  L.push('')
  L.push(`- 剧集：第 ${report.episodeNumber} 集「${report.episodeTitle || ''}」`)
  L.push(`- 生成时间：${report.generatedAt}`)
  L.push(`- 分镜总数：${report.summary.totalShots} ｜ 有视频：${report.summary.shotsWithVideo} ｜ 有 QC：${report.summary.shotsWithQc}`)
  L.push(`- 平均分：唇形 ${report.summary.avgLipSync ?? '—'} ｜ 角色一致性 ${report.summary.avgCharacterConsistency ?? '—'} ｜ 连续性 ${report.summary.avgContinuity ?? '—'} ｜ 总体 ${report.summary.avgOverall ?? '—'}`)
  L.push(`- 达标（≥60）：${report.summary.passed} ｜ 未达标：${report.summary.failed}`)
  L.push('')
  L.push('## 分镜明细')
  L.push('')
  L.push('| # | 景别 | 机位 | 运镜 | 路由 | 时长s | 视频 | QC总体 | 状态 |')
  L.push('|---|------|------|------|------|-------|------|--------|------|')
  for (const s of report.shots) {
    const statusEmoji = s.videoUrl ? '✅' : s.status === 'completed' ? '⚠️' : '⛔'
    L.push(
      `| ${s.storyboardNumber} | ${s.shotType || '—'} | ${s.angle || '—'} | ${s.movement || '—'} | ` +
      `${s.route || '—'} | ${s.durationSec?.toFixed(1) || '—'} | ${statusEmoji} | ${s.qc?.overallScore ?? '—'} | ${s.status || '—'} |`,
    )
  }
  L.push('')
  L.push('## 问题清单')
  L.push('')
  const issues = report.shots.filter((s) => s.qc?.issues?.length)
  if (!issues.length) {
    L.push('无记录问题。')
  } else {
    for (const s of issues) {
      L.push(`- **Shot #${s.storyboardNumber}**（${s.title || ''}）`)
      for (const issue of s.qc!.issues) L.push(`  - ${issue}`)
    }
  }
  L.push('')
  L.push('## 时间线汇总')
  L.push('')
  L.push('| # | 起始s | 时长s | 状态 |')
  L.push('|---|-------|-------|------|')
  for (const t of report.timeline) {
    L.push(`| ${t.storyboardNumber} | ${t.startSec.toFixed(1)} | ${t.durationSec.toFixed(1)} | ${t.status || '—'} |`)
  }
  return L.join('\n')
}

// ===== HTML 渲染（联系表） =====

function imgTagOrPlaceholder(src: string | null, label: string): string {
  if (!src) return `<div class="thumb placeholder"><span>${label}</span></div>`
  return `<div class="thumb"><img src="${escapeHtml(src)}" alt="${label}" loading="lazy" /></div>`
}

function escapeHtml(s: string | null | undefined): string {
  return (s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/** 构建联系表 HTML（每镜缩略图 + 元信息网格，可浏览器打印/转 PDF） */
export async function buildContactSheetHtml(dramaId: number, opts?: { episodeId?: number }): Promise<string> {
  const report = await buildQcReport(dramaId, opts)
  const L: string[] = []
  L.push(`<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>Contact Sheet - ${escapeHtml(report.dramaTitle)}</title>`)
  L.push(`<style>
    body{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:24px;color:#222;background:#fff}
    h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:24px 0 8px;border-bottom:2px solid #eee;padding-bottom:4px}
    .meta{color:#666;font-size:12px;margin:0 0 16px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
    .card{border:1px solid #e3e3e3;border-radius:8px;overflow:hidden;break-inside:avoid}
    .card .header{display:flex;justify-content:space-between;align-items:center;padding:6px 10px;background:#f7f7f7;font-size:12px;font-weight:600}
    .route-tag{font-family:ui-monospace,Consolas,monospace;font-size:10px;padding:1px 6px;border:1px solid #4fc3f7;color:#0288d1;border-radius:4px;background:#e1f5fe}
    .route-tag.bad{border-color:#e57373;color:#c62828;background:#ffebee}
    .thumbs{display:grid;grid-template-columns:1fr 1fr;gap:4px;padding:4px}
    .thumb{aspect-ratio:16/9;background:#fafafa;border:1px solid #eee;display:flex;align-items:center;justify-content:center;font-size:10px;color:#999;overflow:hidden}
    .thumb img{width:100%;height:100%;object-fit:cover}
    .info{padding:0 10px 10px;font-size:11px;color:#444;line-height:1.6}
    .info b{color:#111}
    .qc-ok{color:#2e7d32;font-weight:700}.qc-bad{color:#c62828;font-weight:700}.qc-na{color:#999}
    .summary{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#333;margin-bottom:8px}
    .summary span{padding:4px 10px;border-radius:20px;background:#f0f0f0}
    @media print{.card{break-inside:avoid}body{margin:12px}}
  </style></head><body>`)
  L.push(`<h1>联系表 Contact Sheet</h1>`)
  L.push(`<p class="meta">${escapeHtml(report.dramaTitle)} · 第 ${report.episodeNumber} 集「${escapeHtml(report.episodeTitle)}」 · ${report.generatedAt}</p>`)
  L.push(`<div class="summary">
    <span>分镜 ${report.summary.totalShots}</span>
    <span>有视频 ${report.summary.shotsWithVideo}</span>
    <span>QC 达标 ${report.summary.passed} / 未达标 ${report.summary.failed}</span>
    <span>总体均分 ${report.summary.avgOverall ?? '—'}</span>
  </div>`)
  L.push(`<div class="grid">`)
  for (const s of report.shots) {
    const qcCls = s.qc?.overallScore == null ? 'qc-na' : s.qc.overallScore >= 60 ? 'qc-ok' : 'qc-bad'
    L.push(`<div class="card">
      <div class="header"><span>#${s.storyboardNumber} ${escapeHtml(s.title || '')}</span>
      ${s.route ? `<span class="route-tag${s.route === 'blocked' ? ' bad' : ''}">${escapeHtml(s.route)}</span>` : ''}</div>
      <div class="thumbs">${imgTagOrPlaceholder(s.firstFrameImage, '首帧')}${imgTagOrPlaceholder(s.lastFrameImage, '尾帧')}</div>
      <div class="info">
        <div>${escapeHtml(s.shotType || '—')} / ${escapeHtml(s.angle || '—')} / ${escapeHtml(s.movement || '—')} ｜ 时长 ${s.durationSec?.toFixed(1) || '—'}s</div>
        <div>视频：${s.videoUrl ? '✅ 已生成' : '—'} ｜ 合成：${s.composedVideoUrl ? '✅' : '—'} ｜ take ${s.takeCount}/${s.takeBudget}</div>
        <div>QC：唇形 <b>${s.qc?.lipSyncScore ?? '—'}</b> ｜ 角色 <b>${s.qc?.characterConsistencyScore ?? '—'}</b> ｜ 连续 <b>${s.qc?.continuityScore ?? '—'}</b> ｜ <span class="${qcCls}">总体 ${s.qc?.overallScore ?? '未跑'}</span></div>
        ${s.media?.resolution ? `<div>规格：${escapeHtml(s.media.resolution)} ${s.media.fps ? `@${s.media.fps}fps` : ''}${s.media.durationSec ? ` ${s.media.durationSec.toFixed(1)}s` : ''}</div>` : ''}
        ${s.qc?.issues?.length ? `<div style="color:#c62828">问题：${escapeHtml(s.qc.issues.slice(0, 3).join('；'))}</div>` : ''}
        <div style="color:#888;font-size:10px">${escapeHtml((s.description || '').slice(0, 120))}${(s.description || '').length > 120 ? '…' : ''}</div>
      </div>
    </div>`)
  }
  L.push(`</div></body></html>`)
  return L.join('\n')
}

/** 渲染为 HTML QC 报告（含全部明细，可浏览器打印/转 PDF） */
export async function buildQcReportHtml(dramaId: number, opts?: { episodeId?: number }): Promise<string> {
  const md = await buildQcReportMarkdown(dramaId, opts)
  // 简单转义后按行渲染，表格保留纯文本；交付审计场景 Markdown 版更通用
  return `<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>QC Report</title>
<style>body{font-family:ui-monospace,Consolas,monospace;margin:24px;font-size:13px;line-height:1.7;color:#222}pre{white-space:pre-wrap;word-break:break-all}</style>
</head><body><pre>${escapeHtml(md)}</pre></body></html>`
}

/** 生成交付物文件清单（联系表 HTML + QC 报告 MD/HTML），供打包下载 */
export async function buildQcReportFiles(dramaId: number, opts?: { episodeId?: number }): Promise<Array<{ name: string; content: string }>> {
  const [contactSheet, qcMd, qcHtml] = await Promise.all([
    buildContactSheetHtml(dramaId, opts),
    buildQcReportMarkdown(dramaId, opts),
    buildQcReportHtml(dramaId, opts),
  ])
  const epLabel = opts?.episodeId ? `ep${opts.episodeId}` : 'all'
  return [
    { name: `contact-sheet_${epLabel}.html`, content: contactSheet },
    { name: `qc-report_${epLabel}.md`, content: qcMd },
    { name: `qc-report_${epLabel}.html`, content: qcHtml },
  ]
}

// ===== 媒体信息报告 =====

/** 构建媒体信息报告（纯媒体规格，供交付/转码核对） */
export async function buildMediaInfoReport(dramaId: number, opts?: { episodeId?: number }): Promise<{ generatedAt: string; files: QcShotEntry['media'][] }> {
  const report = await buildQcReport(dramaId, opts)
  return {
    generatedAt: report.generatedAt,
    files: report.shots.map((s) => s.media),
  }
}
