/**
 * 画布图片算子：裁剪 / 旋转 / 放大 / 切分（移植自上游 `lib/canvas/canvas-image-data.ts`，MIT；自研重写 ✓）。
 *
 * ## 与源的四处差异（都是**必须**的，不是顺手改 ✗）
 *
 * 1. ⭐ 输出 **`Blob` 而不是 dataURL** ✓：本仓画布节点的 `url` 要能持久化，
 *    dataURL 塞进 localStorage 两三张 4K 就超配额（5MB ✗），且刷新后才报错、看不出是谁的锅 ✗。
 *    拿 Blob 直接走 `uploadAPI.image` 落成 `/uploads/xxx.png` 才是本仓的形状 ✓。
 * 2. ⭐ `loadImage` **必须能失败** ✓：源的实现只 `onload` 不 `onerror` ⇒ 图坏了会**永远挂起**
 *    （UI 上表现为"点了没反应"，最难查 ✗）。
 * 3. ⭐ PNG 太大时**回退 JPEG** ✓：上传接口有 20MB 上限（`routers/upload.py`），
 *    4096 长边的 PNG 轻松超 ⇒ 与其让用户看到"文件过大"，不如回退有损格式。
 * 4. 源的 `transformAngleDataUrl`（伪 3D 摆角 + 广角暗角）**不搬** ✓：
 *    它要一组叙事参数（horizontalAngle/pitch/cameraDistance/wideAngle），本仓画布没有这个面板宿主，
 *    搬进来就是没有调用方的死代码 ✗。这里给的是**正交旋转 90°/180°/270°**（画布真正需要的那件事 ✓）。
 */

/** 放大后的长边上限（源 `MAX_UPSCALE_LONG_EDGE` ✓） */
export const MAX_UPSCALE_LONG_EDGE = 4096

/** PNG 超过这个体积就回退 JPEG ✓（上传上限 20MB，留出余量） */
const PNG_FALLBACK_BYTES = 12 * 1024 * 1024

/** 裁剪矩形：**归一化 0–1**（与画布缩放无关 ✓） */
export interface CropRect {
  x: number
  y: number
  width: number
  height: number
}

export type UpscaleAlgorithm = 'nearest' | 'bilinear' | 'high'

export interface SplitParams {
  rows: number
  columns: number
  /** 归一化切点（0–1）✓；不给就等分 */
  horizontalLines?: number[]
  verticalLines?: number[]
}

export interface ImageResult {
  blob: Blob
  width: number
  height: number
}

/** `Blob | URL 字符串` 都吃 ✓（画布节点存的是 URL，刚生成的是 Blob） */
export type ImageInput = Blob | string

function inputToUrl(source: ImageInput): string {
  return typeof source === 'string' ? source : URL.createObjectURL(source)
}

function releaseInput(source: ImageInput, url: string) {
  if (typeof source !== 'string') URL.revokeObjectURL(url)
}

/** 加载图片；**失败就 reject** ✓（源缺 `onerror`，坏图会永久挂起 ✗） */
function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.decoding = 'async'
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('图片加载失败（可能已失效或跨域被拒）'))
    image.src = url
  })
}

function context2d(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const context = canvas.getContext('2d')
  if (!context) throw new Error('当前浏览器不支持 canvas 2d')
  return context
}

/** 画布 → Blob；PNG 过大回退 JPEG ✓（透明底会变白，注释里说清 ✓） */
async function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  const png = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
  if (png && png.size <= PNG_FALLBACK_BYTES) return png
  const jpeg = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92))
  if (jpeg) return jpeg
  if (png) return png
  throw new Error('图片编码失败')
}

/**
 * 裁剪。`crop` 不给 ⇒ **居中正方形** ✓（源同口径：这是"裁成方形"最常用的一刀）。
 */
export async function cropImage(source: ImageInput, crop?: CropRect): Promise<ImageResult> {
  const url = inputToUrl(source)
  try {
    const image = await loadImage(url)
    const rect = crop
      ? {
          sx: Math.floor(crop.x * image.width),
          sy: Math.floor(crop.y * image.height),
          sw: Math.ceil(crop.width * image.width),
          sh: Math.ceil(crop.height * image.height),
        }
      : (() => {
          const size = Math.min(image.width, image.height)
          return {
            sx: Math.max(0, Math.floor((image.width - size) / 2)),
            sy: Math.max(0, Math.floor((image.height - size) / 2)),
            sw: size,
            sh: size,
          }
        })()
    if (rect.sw <= 0 || rect.sh <= 0) throw new Error('裁剪区域为空')
    const canvas = document.createElement('canvas')
    canvas.width = rect.sw
    canvas.height = rect.sh
    context2d(canvas).drawImage(image, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, rect.sw, rect.sh)
    return { blob: await canvasToBlob(canvas), width: canvas.width, height: canvas.height }
  } finally {
    releaseInput(source, url)
  }
}

/**
 * 正交旋转。**非 90 倍数按最近的 90 倍数归一** ✓：
 * 任意角度旋转要么裁掉四角、要么留透明边，画布卡片里两者都难看 ✗。
 */
export async function rotateImage(source: ImageInput, degrees: number): Promise<ImageResult> {
  const url = inputToUrl(source)
  try {
    const image = await loadImage(url)
    const normalized = ((Math.round(degrees / 90) * 90) % 360 + 360) % 360
    const swap = normalized === 90 || normalized === 270
    const canvas = document.createElement('canvas')
    canvas.width = swap ? image.height : image.width
    canvas.height = swap ? image.width : image.height
    const context = context2d(canvas)
    context.translate(canvas.width / 2, canvas.height / 2)
    context.rotate((normalized * Math.PI) / 180)
    context.drawImage(image, -image.width / 2, -image.height / 2)
    return { blob: await canvasToBlob(canvas), width: canvas.width, height: canvas.height }
  } finally {
    releaseInput(source, url)
  }
}

/** 按长边求目标尺寸（夹到 1–4096 ✓，比例不变 ✓） */
export function resolveUpscaleSize(width: number, height: number, targetLongEdge: number) {
  const longEdge = Math.max(1, width, height)
  const target = Math.min(MAX_UPSCALE_LONG_EDGE, Math.max(1, Math.round(targetLongEdge)))
  const scale = target / longEdge
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) }
}

/**
 * 放大到指定长边。
 *
 * `high` 走**分段两倍逼近** ✓（源 `drawStepUpscale` 口径）：一次从 512 拉到 4096 会糊成一片，
 * 每步 ×2 让插值有中间信息可用 ✓。
 */
export async function upscaleImage(
  source: ImageInput,
  targetLongEdge: number,
  algorithm: UpscaleAlgorithm = 'high',
): Promise<ImageResult> {
  const url = inputToUrl(source)
  try {
    const image = await loadImage(url)
    const { width, height } = resolveUpscaleSize(image.width, image.height, targetLongEdge)
    if (width === image.width && height === image.height) {
      // 目标不大于原图 ⇒ 原样返回，不白跑一次重采样 ✓
      const canvas = document.createElement('canvas')
      canvas.width = image.width
      canvas.height = image.height
      context2d(canvas).drawImage(image, 0, 0)
      return { blob: await canvasToBlob(canvas), width: canvas.width, height: canvas.height }
    }
    if (algorithm !== 'high') {
      return { blob: await resizeToBlob(image, image.width, image.height, width, height, algorithm), width, height }
    }
    let current: CanvasImageSource = image
    let currentWidth = image.width
    let currentHeight = image.height
    while (currentWidth * 2 < width && currentHeight * 2 < height) {
      currentWidth *= 2
      currentHeight *= 2
      current = resizeToCanvas(current, currentWidth / 2, currentHeight / 2, currentWidth, currentHeight, 'high')
    }
    return {
      blob: await resizeToBlob(current, currentWidth, currentHeight, width, height, 'high'),
      width,
      height,
    }
  } finally {
    releaseInput(source, url)
  }
}

/**
 * 切分（分镜切图）。`lines` 是**归一化切点** ✓（0–1），不给就等分 ✓。
 * 返回按「先行后列」排序的块 ✓（与源一致：左上→右上→下一行）。
 */
export async function splitImage(source: ImageInput, params: SplitParams): Promise<Array<{ row: number; column: number; blob: Blob }>> {
  const url = inputToUrl(source)
  try {
    const image = await loadImage(url)
    const xCuts = buildSplitCuts(params.verticalLines, image.width, Math.max(1, Math.floor(params.columns)))
    const yCuts = buildSplitCuts(params.horizontalLines, image.height, Math.max(1, Math.floor(params.rows)))
    const pieces: Array<{ row: number; column: number; blob: Blob }> = []
    for (let row = 0; row < yCuts.length - 1; row += 1) {
      for (let column = 0; column < xCuts.length - 1; column += 1) {
        const sx = xCuts[column]
        const sy = yCuts[row]
        const sw = xCuts[column + 1] - sx
        const sh = yCuts[row + 1] - sy
        if (sw <= 0 || sh <= 0) continue
        const canvas = document.createElement('canvas')
        canvas.width = sw
        canvas.height = sh
        context2d(canvas).drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh)
        pieces.push({ row, column, blob: await canvasToBlob(canvas) })
      }
    }
    return pieces
  } finally {
    releaseInput(source, url)
  }
}

/** 切点：给了归一化线就用线（过滤边界外的 ✓），否则等分 ✓ */
function buildSplitCuts(lines: number[] | undefined, size: number, count: number): number[] {
  if (!lines?.length) {
    return Array.from({ length: count + 1 }, (_, index) => Math.floor((index * size) / count))
  }
  const inner = lines
    .map((line) => Math.round(line * size))
    .filter((line) => line > 0 && line < size)
    .sort((a, b) => a - b)
  return [0, ...inner, size]
}

function resizeToCanvas(
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
  algorithm: UpscaleAlgorithm,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = context2d(canvas)
  context.imageSmoothingEnabled = algorithm !== 'nearest'
  context.imageSmoothingQuality = algorithm === 'bilinear' ? 'medium' : 'high'
  context.drawImage(source, 0, 0, sourceWidth, sourceHeight, 0, 0, width, height)
  return canvas
}

async function resizeToBlob(
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
  algorithm: UpscaleAlgorithm,
): Promise<Blob> {
  return canvasToBlob(resizeToCanvas(source, sourceWidth, sourceHeight, width, height, algorithm))
}
