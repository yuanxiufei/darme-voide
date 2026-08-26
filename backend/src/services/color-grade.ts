/**
 * 校色后处理服务 — 用 sharp 对生成图片应用色彩校准/色调映射/白平衡/曝光/饱和度/对比度/肤色/阴影高光。
 *
 * 说明：sharp 无精确的 shadows/highlights 分离与肤色检测 API，
 * 阴影高光用 gamma 近似、肤色还原用通道增益近似，属「实用级」校色（非 Photoshop 级局部调整）。
 * 所有参数默认 0（中性），范围 -100..100（skinTone 为 0..100）。
 */
import fs from 'fs'
import sharp from 'sharp'
import { getAbsolutePath } from '../utils/storage.js'

/** 校色参数结构。所有数值默认 0（中性）。 */
export interface ColorGradeParams {
  /** 色彩校准：RGB 三通道增益 */
  colorCalibration?: { red?: number; green?: number; blue?: number }
  /** 色调映射：中间调 gamma */
  toneMapping?: { gamma?: number }
  /** 白平衡修正：temperature 冷(-)暖(+)，tint 绿(-)品红(+) */
  whiteBalance?: { temperature?: number; tint?: number }
  /** 曝光补偿 */
  exposure?: number
  /** 饱和度微调 */
  saturation?: number
  /** 对比度优化 */
  contrast?: number
  /** 肤色还原（0..100，0=关闭） */
  skinTone?: number
  /** 阴影与高光细节保留 */
  shadowsHighlights?: { shadows?: number; highlights?: number }
}

export const EMPTY_COLOR_GRADE: ColorGradeParams = {
  colorCalibration: { red: 0, green: 0, blue: 0 },
  toneMapping: { gamma: 0 },
  whiteBalance: { temperature: 0, tint: 0 },
  exposure: 0,
  saturation: 0,
  contrast: 0,
  skinTone: 0,
  shadowsHighlights: { shadows: 0, highlights: 0 },
}

const n = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0)

/** 规整校色参数：过滤非法值、补齐默认值 */
export function normalizeColorGrade(input?: ColorGradeParams | null): ColorGradeParams {
  if (!input) return structuredClone(EMPTY_COLOR_GRADE)
  const cc = input.colorCalibration
  const wb = input.whiteBalance
  const sh = input.shadowsHighlights
  const tm = input.toneMapping
  return {
    colorCalibration: { red: n(cc?.red), green: n(cc?.green), blue: n(cc?.blue) },
    toneMapping: { gamma: n(tm?.gamma) },
    whiteBalance: { temperature: n(wb?.temperature), tint: n(wb?.tint) },
    exposure: n(input.exposure),
    saturation: n(input.saturation),
    contrast: n(input.contrast),
    skinTone: Math.max(0, Math.min(100, n(input.skinTone))),
    shadowsHighlights: { shadows: n(sh?.shadows), highlights: n(sh?.highlights) },
  }
}

/** 是否有任何非中性的校色调整 */
export function hasColorGrade(params?: ColorGradeParams | null): boolean {
  const p = normalizeColorGrade(params)
  return !!(
    p.colorCalibration?.red || p.colorCalibration?.green || p.colorCalibration?.blue ||
    p.toneMapping?.gamma ||
    p.whiteBalance?.temperature || p.whiteBalance?.tint ||
    p.exposure || p.saturation || p.contrast || p.skinTone ||
    p.shadowsHighlights?.shadows || p.shadowsHighlights?.highlights
  )
}

/** 从 DB 字符串解析校色参数；无效或无调整时返回 null */
export function parseColorGrade(json?: string | null): ColorGradeParams | null {
  if (!json) return null
  try {
    const normalized = normalizeColorGrade(JSON.parse(json))
    return hasColorGrade(normalized) ? normalized : null
  } catch {
    return null
  }
}

/** 对图片 buffer 应用校色，返回新 buffer（保持原格式，移除 alpha） */
export async function applyColorGrade(buffer: Buffer, params: ColorGradeParams): Promise<Buffer> {
  const p = normalizeColorGrade(params)
  let img = sharp(buffer).rotate().removeAlpha()

  // 1. 色彩校准（RGB 通道增益）
  const cc = p.colorCalibration!
  if (cc.red || cc.green || cc.blue) {
    img = img.recomb([
      [1 + cc.red! / 100, 0, 0],
      [0, 1 + cc.green! / 100, 0],
      [0, 0, 1 + cc.blue! / 100],
    ])
  }

  // 2. 白平衡修正（色温：暖=+红-蓝 / 冷=-红+蓝；色调：品红=+红+蓝 / 绿=-红-蓝）
  const wb = p.whiteBalance!
  if (wb.temperature || wb.tint) {
    const t = wb.temperature! / 100
    const ti = wb.tint! / 100
    img = img.recomb([
      [1 + t * 0.2 + ti * 0.1, 0, 0],
      [0, 1 - ti * 0.1, 0],
      [0, 0, 1 - t * 0.2 + ti * 0.1],
    ])
  }

  // 3. 曝光补偿（brightness 乘法增益）
  if (p.exposure) img = img.modulate({ brightness: 1 + p.exposure / 100 })

  // 4. 饱和度微调
  if (p.saturation) img = img.modulate({ saturation: 1 + p.saturation / 100 })

  // 5. 对比度优化（保持中间灰 128 不变）
  if (p.contrast) {
    const slope = 1 + p.contrast / 100
    img = img.linear(slope, 128 * (1 - slope))
  }

  // 6. 色调映射（gamma，sharp 中性值为 2.2；>0 提亮、<0 压暗）
  const tm = p.toneMapping!
  if (tm.gamma) img = img.gamma(2.2 - (tm.gamma! / 100) * 1.2)

  // 7. 阴影与高光细节保留（gamma 近似：阴影提亮降低 gamma，高光压制提高 gamma）
  const sh = p.shadowsHighlights!
  if (sh.shadows || sh.highlights) {
    const g = 2.2 - (sh.shadows! / 100) * 0.7 + (sh.highlights! / 100) * 0.7
    img = img.gamma(g)
  }

  // 8. 肤色还原（轻微暖化 + 红通道增益，保护肤色自然红润）
  if (p.skinTone) {
    const k = p.skinTone / 100
    img = img.recomb([
      [1 + k * 0.06, 0, 0],
      [0, 1 + k * 0.02, 0],
      [0, 0, 1 - k * 0.04],
    ])
  }

  return img.toBuffer()
}

/** 对本地图片文件应用校色（原地覆盖），无调整时原样返回 */
export async function applyColorGradeToFile(localPath: string, colorGradeJson?: string | null): Promise<string> {
  const params = parseColorGrade(colorGradeJson)
  if (!params) return localPath
  const abs = getAbsolutePath(localPath)
  const input = fs.readFileSync(abs)
  const output = await applyColorGrade(input, params)
  fs.writeFileSync(abs, output)
  return localPath
}
