/**
 * 扩图的几何口径（移植自上游 `canvas-node-outpaint-dialog.tsx` 的语义，MIT；自研重写 ✓）。
 *
 * ## 为什么要把"补多少"算成一张确定计划
 *
 * 上游工作流 `image-qwen21-outpaint.json` 是**两段**尺寸：
 * 1. 节点 2 `ImagePadForOutpaint(left/top/right/bottom, feathering)` 先把画布撑大（羽化边缘）；
 * 2. 节点 18 `ImageScale(outpaintWidth, outpaintHeight)` 再把结果**硬缩到目标尺寸** ✓。
 *
 * ⇒ 若"撑大后的尺寸"和"目标尺寸"比例不一致，第 2 步会**整体拉伸变形**（人眼第一眼就看得出 ✗）。
 * 所以这里只算一次，`width/height` 同时喂给两个口子；四边留白只由用户改 ✓。
 *
 * ## 两条硬约束（照抄上游，不是自创）
 *
 * - 长边 ≤ 4096、总像素 ≤ 8 MP：再大 Qwen-Image 一次前向就装不下（22.49 GiB 卡实测）✗
 * - 尺寸吸 **32**：VAE 8 倍下采样 × DiT patch 2 × 2 ⇒ 不吸会让边缘出现半格噪声 ✓
 */

/** 长边上限 */
export const OUTPAINT_MAX_EDGE = 4096
/** 总像素上限（8 MP） */
export const OUTPAINT_MAX_PIXELS = 8_000_000
/** 每边最小保留（对齐后仍不许小于它 ⇒ 太小等于没扩，模型会自己编一圈边） */
export const OUTPAINT_MIN_SIDE = 256
/** 尺寸吸附步长 */
export const OUTPAINT_ALIGN = 32
/** 羽化默认（工作流节点默认值，改大了接缝会糊） */
export const OUTPAINT_FEATHERING = 40
/** 参照像素预算默认（工作流节点默认值） */
export const OUTPAINT_MEGAPIXELS = 1.5

export interface Paddings {
  left: number
  top: number
  right: number
  bottom: number
}

export interface ImageSize {
  width: number
  height: number
}

export interface OutpaintOptions {
  feathering?: number
  megapixels?: number
}

export interface OutpaintPlan {
  paddings: Paddings
  /** 撑大后 = 缩回后的目标尺寸（两者恒等 ✓） */
  width: number
  height: number
  feathering: number
  megapixels: number
  /** 人话提醒（超限被压、尺寸被吸）—— 界面要原样显示，不许静默改数 ✗ */
  warnings: string[]
}

export const ZERO_PADDINGS: Paddings = { left: 0, top: 0, right: 0, bottom: 0 }

export function alignUp(value: number, step = OUTPAINT_ALIGN): number {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.ceil(value / step) * step)
}

export function clampPaddings(input: Partial<Paddings> | null | undefined): Paddings {
  const pick = (value: unknown) => {
    const num = Number(value)
    if (!Number.isFinite(num) || num <= 0) return 0
    return Math.round(num)
  }
  return {
    left: pick(input?.left),
    top: pick(input?.top),
    right: pick(input?.right),
    bottom: pick(input?.bottom),
  }
}

/** 按目标宽高比把原图"居中放进"新画布 ⇒ 返回四边留白 ✓（点一下比例键就该得到结果） */
export function ratioPaddings(image: ImageSize, ratio: number): Paddings {
  if (!(ratio > 0) || !(image.width > 0) || !(image.height > 0)) return { ...ZERO_PADDINGS }
  const current = image.width / image.height
  if (Math.abs(current - ratio) < 1e-6) return { ...ZERO_PADDINGS }
  if (ratio > current) {
    // 需要更宽 ⇒ 加左右（高度不动的前提下）
    const extra = image.height * ratio - image.width
    const side = Math.max(0, Math.round(extra / 2))
    return { left: side, top: 0, right: side, bottom: 0 }
  }
  // 需要更高 ⇒ 加下上
  const extra = image.width / ratio - image.height
  const side = Math.max(0, Math.round(extra / 2))
  return { left: 0, top: side, right: 0, bottom: side }
}

/**
 * 得出最终计划 ✓：先按像素/边长预算压留白，再吸 32，最后把吸附余量补到右/下边。
 */
export function outpaintPlan(
  image: ImageSize,
  paddings: Paddings,
  options: OutpaintOptions = {},
): OutpaintPlan {
  const warnings: string[] = []
  const imageWidth = Math.max(1, Math.round(image.width))
  const imageHeight = Math.max(1, Math.round(image.height))
  const clean = clampPaddings(paddings)

  // 目标比例由「原图 + 用户留白」决定 ⇒ 边长上限按这个比例反算，
  // 否则 16:9 与 9:16 会被同一根尺子卡住 ✗
  const draftHeight = Math.max(1, imageHeight + clean.top + clean.bottom)
  const aspect = (imageWidth + clean.left + clean.right) / draftHeight
  const maxWidth = Math.min(OUTPAINT_MAX_EDGE, Math.floor(Math.sqrt(OUTPAINT_MAX_PIXELS * aspect)))
  const maxHeight = Math.min(OUTPAINT_MAX_EDGE, Math.floor(Math.sqrt(OUTPAINT_MAX_PIXELS / Math.max(1e-6, aspect))))

  // 原图自己就超预算 ⇒ 留白只能全给 0（不是"少给一点"，是这张图本身已经超了）
  const budgetWidth = Math.max(0, maxWidth - imageWidth)
  const budgetHeight = Math.max(0, maxHeight - imageHeight)
  if (clean.left + clean.right > budgetWidth && clean.left + clean.right > 0) {
    const scale = budgetWidth / (clean.left + clean.right)
    clean.left = Math.floor(clean.left * scale)
    clean.right = Math.floor(clean.right * scale)
    warnings.push(`左右留白已压到 ${clean.left}/${clean.right} px：长边上限 ${OUTPAINT_MAX_EDGE}、总像素上限 8 MP`)
  }
  if (clean.top + clean.bottom > budgetHeight && clean.top + clean.bottom > 0) {
    const scale = budgetHeight / (clean.top + clean.bottom)
    clean.top = Math.floor(clean.top * scale)
    clean.bottom = Math.floor(clean.bottom * scale)
    warnings.push(`上下留白已压到 ${clean.top}/${clean.bottom} px：长边上限 ${OUTPAINT_MAX_EDGE}、总像素上限 8 MP`)
  }

  // 吸 32：余量补到右/下（补到左/上会让原图整体位移，用户会以为算错 ✗）
  let width = alignUp(imageWidth + clean.left + clean.right)
  let height = alignUp(imageHeight + clean.top + clean.bottom)
  const wantedWidth = imageWidth + clean.left + clean.right
  const wantedHeight = imageHeight + clean.top + clean.bottom
  if (width !== wantedWidth) {
    clean.right += Math.max(0, width - wantedWidth)
    warnings.push(`宽度已吸到 ${OUTPAINT_ALIGN} 的整数倍（实际 ${width}）`)
  }
  if (height !== wantedHeight) {
    clean.bottom += Math.max(0, height - wantedHeight)
    warnings.push(`高度已吸到 ${OUTPAINT_ALIGN} 的整数倍（实际 ${height}）`)
  }

  // 吸附可能把尺寸顶出长边上限 ⇒ 再压一次（只压补出来的那一边）
  if (width > OUTPAINT_MAX_EDGE) {
    width = OUTPAINT_MAX_EDGE
    clean.right = Math.max(0, width - imageWidth - clean.left)
    warnings.push(`宽度已达长边上限 ${OUTPAINT_MAX_EDGE}`)
  }
  if (height > OUTPAINT_MAX_EDGE) {
    height = OUTPAINT_MAX_EDGE
    clean.bottom = Math.max(0, height - imageHeight - clean.top)
    warnings.push(`高度已达长边上限 ${OUTPAINT_MAX_EDGE}`)
  }

  // 每边最小保留：压过头的地方补回来（否则"扩图"变成了"改尺寸"）
  if (width < OUTPAINT_MIN_SIDE) {
    clean.right += OUTPAINT_MIN_SIDE - width
    width = alignUp(imageWidth + clean.left + clean.right)
    warnings.push(`宽度不足 ${OUTPAINT_MIN_SIDE}，已补到 ${width}`)
  }
  if (height < OUTPAINT_MIN_SIDE) {
    clean.bottom += OUTPAINT_MIN_SIDE - height
    height = alignUp(imageHeight + clean.top + clean.bottom)
    warnings.push(`高度不足 ${OUTPAINT_MIN_SIDE}，已补到 ${height}`)
  }

  const feathering = Math.max(0, Math.round(Number(options.feathering ?? OUTPAINT_FEATHERING) || 0))
  const megapixels = Math.max(0.1, Number(options.megapixels ?? OUTPAINT_MEGAPIXELS) || OUTPAINT_MEGAPIXELS)

  return {
    // 不变量：四边之和 + 原图 **恒等于** 目标尺寸 ✓（工作流两段尺寸因此永不拉伸）
    paddings: {
      left: clean.left,
      top: clean.top,
      right: Math.max(0, width - imageWidth - clean.left),
      bottom: Math.max(0, height - imageHeight - clean.top),
    },
    width,
    height,
    feathering,
    megapixels,
    warnings,
  }
}
