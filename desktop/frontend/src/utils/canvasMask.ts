/**
 * 局部重绘的蒙版口径（移植自上游 `canvas-node-mask-edit-dialog.tsx` 的交互语义，MIT；自研重写 ✓）。
 *
 * ## 为什么口子必须是「白 = 重绘」
 *
 * 上游工作流 `image-qwen21-inpaint.json` 的节点 3 是 `ImageToMask(channel="red")` ⇒
 * **红通道亮的区域才是重绘区** ✓。画布上刷成蓝的（预览色）只是为了让人看得清底图，
 * 真正上传的那张蒙版必须另画一张「黑底 + 白笔」✗ 不能把预览图直接送去。
 *
 * ## 三条口径
 *
 * 1. **笔迹一律存原图像素坐标** ✓：卡片缩放/窗口大小变了，同一份笔迹重放出来
 *    还是同一块区域 ✓（存显示坐标会在换窗口后整体偏移 ✗）。
 * 2. **撤销走「重放笔迹」而不是存快照** ✓：2048² 的一份 ImageData 就是 16 MB，
 *    存 10 步等于 160 MB ✗；笔迹数组几十字节/段，重放成本只有几十毫秒 ✓。
 * 3. **橡皮擦在蒙版里画黑、在预览里挖洞** ✓：蒙版要的是"不重绘"这个确定结论
 *    （透明像素在 `ImageToMask` 里同样是 0，但黑底截图/另存更不容易被误判 ✗）。
 */

/** 蒙版底（不重绘）：黑 */
export const MASK_BG = '#000000'
/** 蒙版笔（重绘）：白 */
export const MASK_FG = '#ffffff'
/** 预览色：蓝（与底图的暖色拉开，刷没刷上一眼就能看出来 ✓） */
export const MASK_PREVIEW_COLOR = 'rgba(37, 99, 235, 0.45)'

export const DEFAULT_BRUSH_SIZE = 100
export const MIN_BRUSH_SIZE = 4
export const MAX_BRUSH_SIZE = 512

export interface MaskPoint {
  x: number
  y: number
}

/** 一段连续笔迹 ✓（提笔即封段 —— 撤销以「一笔」为单位才符合手感） */
export interface MaskStroke {
  /** true = 橡皮（擦回不重绘区） */
  erase: boolean
  /** 笔宽，原图像素 */
  size: number
  points: MaskPoint[]
}

export interface MaskSize {
  width: number
  height: number
}

export function clampBrushSize(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_BRUSH_SIZE
  return Math.min(MAX_BRUSH_SIZE, Math.max(MIN_BRUSH_SIZE, Math.round(value)))
}

/** 笔宽 → 半径（canvas 的 `lineWidth` 才是直径 ✓） */
export function brushRadius(size: number): number {
  return Math.max(1, clampBrushSize(size) / 2)
}

/** 显示坐标 → 原图坐标 ✓（画布被 CSS 缩放时 rect 与原图分辨率不等，必须按比例换） */
export function toImagePoint(
  point: MaskPoint,
  display: MaskSize,
  natural: MaskSize,
): MaskPoint {
  const sx = natural.width / Math.max(1, display.width)
  const sy = natural.height / Math.max(1, display.height)
  return { x: point.x * sx, y: point.y * sy }
}

/**
 * 把笔迹重放到任意 2D 上下文 ✓（蒙版与预览共用一份绘制逻辑，
 * 两处各写一遍必然漂开 ✗）。
 *
 * @param eraseColor 橡皮用什么色覆盖；`null` = 用 `destination-out` 挖成透明（预览层用）
 */
export function replayStrokes(
  ctx: CanvasRenderingContext2D,
  strokes: MaskStroke[],
  paintColor: string,
  eraseColor: string | null,
): void {
  ctx.save()
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  for (const stroke of strokes) {
    if (!stroke.points.length) continue
    ctx.globalCompositeOperation = stroke.erase
      ? eraseColor === null
        ? 'destination-out'
        : 'source-over'
      : 'source-over'
    ctx.strokeStyle = stroke.erase ? eraseColor ?? '#000' : paintColor
    ctx.fillStyle = ctx.strokeStyle
    ctx.lineWidth = Math.max(1, stroke.size)
    const [first, ...rest] = stroke.points
    // 落笔只有一个点 ⇒ 画圆点（只 moveTo/lineTo 会什么都不显示 ✗）
    if (!rest.length) {
      ctx.beginPath()
      ctx.arc(first!.x, first!.y, ctx.lineWidth / 2, 0, Math.PI * 2)
      ctx.fill()
      continue
    }
    ctx.beginPath()
    ctx.moveTo(first!.x, first!.y)
    for (const point of rest) ctx.lineTo(point.x, point.y)
    ctx.stroke()
  }
  ctx.restore()
}

/** 离屏画布（浏览器专用；纯逻辑测试不要调它） */
export function createCanvas(size: MaskSize): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(size.width))
  canvas.height = Math.max(1, Math.round(size.height))
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('浏览器不支持 2D 画布')
  return { canvas, ctx }
}

/** 黑底 + 白笔 = 直接能送工作流的蒙版 PNG ✓ */
export function maskCanvas(size: MaskSize, strokes: MaskStroke[]): HTMLCanvasElement {
  const { canvas, ctx } = createCanvas(size)
  ctx.fillStyle = MASK_BG
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  replayStrokes(ctx, strokes, MASK_FG, MASK_BG)
  return canvas
}

/** 预览层：只有笔迹本身（底图单独一张 `<img>` 垫在下面，重绘时才不用反复画大图 ✓） */
export function previewCanvas(size: MaskSize, strokes: MaskStroke[]): HTMLCanvasElement {
  const { canvas, ctx } = createCanvas(size)
  replayStrokes(ctx, strokes, MASK_PREVIEW_COLOR, null)
  return canvas
}

/** 蒙版覆盖率 0–1 ✓（用来判断"一笔没刷"和显示进度，采样步长 4 像素够用） */
export function maskCoverage(canvas: HTMLCanvasElement, step = 4): number {
  const ctx = canvas.getContext('2d')
  if (!ctx) return 0
  const { width, height } = canvas
  if (!width || !height) return 0
  const data = ctx.getImageData(0, 0, width, height).data
  let painted = 0
  let total = 0
  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const index = (y * width + x) * 4
      total += 1
      if (data[index]! > 127) painted += 1
    }
  }
  return total ? painted / total : 0
}

export function toPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error('蒙版导出失败'))
    }, 'image/png')
  })
}
